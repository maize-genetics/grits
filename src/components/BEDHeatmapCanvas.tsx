import React, { useRef, useEffect, useCallback, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import { getBackend } from '../platform';

/** Visible range information reported by the canvas */
export interface BEDVisibleRange {
  startRegionIdx: number;
  endRegionIdx: number;
  startCol: number;
  endCol: number;
}

/** A genomic region from the BED file */
export interface BEDRegion {
  start: number;
  end: number;
}

/** Export options for PNG generation */
export interface BEDExportOptions {
  fileId: string;
  chromosome: string;
  /** Custom title (overrides auto-generated one) */
  title?: string;
  /** Export width in logical pixels */
  width?: number;
  /** Export height in logical pixels (heatmap area, excluding title/legend) */
  height?: number;
  /** DPI scale factor (default 3) */
  scale?: number;
  /** Whether to include the cell value legend (default true) */
  includeCellValueLegend?: boolean;
  /** Whether to include the path overlay legend (default true) */
  includePathLegend?: boolean;
  /** Per-path visibility overrides keyed by label */
  pathVisibility?: Record<string, boolean>;
  /** Start position for the exported range, or null/omitted for the current viewport */
  startPosition?: number | null;
  /** End position for the exported range, or null/omitted for the current viewport */
  endPosition?: number | null;
}

/** Methods exposed via ref */
export interface BEDHeatmapCanvasHandle {
  exportToPng: (options: BEDExportOptions) => Promise<void>;
}

interface BEDHeatmapCanvasProps {
  /** Matrix data: rows = parents, columns = regions. Values: 0=empty, 1=parent1, 2=parent2, 3=both */
  matrix: number[][];
  /** Region labels for x-axis */
  regions: BEDRegion[];
  /** Parent names for y-axis */
  parentNames: string[];
  /** For each column, the row index of parent1 */
  parent1Path: number[];
  /** For each column, the row index of parent2 */
  parent2Path: number[];
  /** Whether to show the parent 1 path line */
  showParent1Path: boolean;
  /** Whether to show the parent 2 path line */
  showParent2Path: boolean;
  /** Current zoom level (1 = 100%) */
  zoomLevel: number;
  /** Horizontal scroll offset (0-1) */
  scrollOffset: number;
  onScrollChange: (offset: number) => void;
  onZoomChange?: (zoom: number) => void;
  onVisibleRangeChange?: (range: BEDVisibleRange) => void;
  baseCellSize?: number;
  cellWidthMultiplier?: number;
  /** Callback when cell width multiplier changes via Ctrl+Shift+scroll */
  onCellWidthChange?: (multiplier: number) => void;
  cellHeightMultiplier?: number;
  showGridLines?: boolean;
}

// Cell fill colors — pale variants for softer heatmap background
const COLOR_PARENT1 = '#FFAB91'; // Pale orange
const COLOR_PARENT2 = '#81D4FA'; // Pale blue
const COLOR_BOTH = '#CE93D8';    // Pale purple
const COLOR_EMPTY = '#F0F0F0';   // Light grey

// Path line colors — darker/contrasting variants so paths are distinct from cell fills
const PATH_COLOR_PARENT1 = '#B81D00'; // Deep red-orange
const PATH_COLOR_PARENT2 = '#0B4F6C'; // Dark teal

interface TooltipContentProps {
  mousePos: { x: number; y: number };
  containerSize: { width: number; height: number };
  parentNames: string[];
  regions: BEDRegion[];
  hoveredCell: { row: number; col: number; value: number };
}

const TooltipContent: React.FC<TooltipContentProps> = ({
  mousePos,
  containerSize,
  parentNames,
  regions,
  hoveredCell,
}) => {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tooltipSize, setTooltipSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (tooltipRef.current) {
      const rect = tooltipRef.current.getBoundingClientRect();
      setTooltipSize({ width: rect.width, height: rect.height });
    }
  }, [hoveredCell]);

  const tooltipPosition = useMemo(() => {
    const offset = 15;
    const padding = 8;
    let left = mousePos.x + offset;
    let top = mousePos.y - 10;

    if (tooltipSize.width > 0 && left + tooltipSize.width + padding > containerSize.width) {
      left = mousePos.x - tooltipSize.width - offset;
    }
    if (tooltipSize.height > 0 && top + tooltipSize.height + padding > containerSize.height) {
      top = mousePos.y - tooltipSize.height - offset;
    }
    if (left < padding) left = padding;
    if (top < padding) top = padding;
    return { left, top };
  }, [mousePos, tooltipSize, containerSize]);

  const region = regions[hoveredCell.col];
  const parentName = parentNames[hoveredCell.row];
  const roleLabel = hoveredCell.value === 1 ? 'Parent 1' :
                    hoveredCell.value === 2 ? 'Parent 2' :
                    hoveredCell.value === 3 ? 'Both Parents' : 'Empty';

  return (
    <div
      ref={tooltipRef}
      className="heatmap-tooltip"
      style={{
        position: 'absolute',
        left: tooltipPosition.left,
        top: tooltipPosition.top,
        background: 'var(--md-sys-color-inverse-surface, #322f35)',
        color: 'var(--md-sys-color-inverse-on-surface, #f5eff7)',
        padding: '6px 10px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: '"Roboto", sans-serif',
        pointerEvents: 'none',
        zIndex: 100,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        whiteSpace: 'nowrap',
      }}
    >
      <div><strong>{parentName}</strong></div>
      {region && (
        <div>Region: {region.start.toLocaleString()} - {region.end.toLocaleString()} bp</div>
      )}
      <div>Role: {roleLabel}</div>
    </div>
  );
};

const BEDHeatmapCanvas = forwardRef<BEDHeatmapCanvasHandle, BEDHeatmapCanvasProps>(({
  matrix,
  regions,
  parentNames,
  parent1Path,
  parent2Path,
  showParent1Path,
  showParent2Path,
  zoomLevel,
  scrollOffset,
  onScrollChange,
  onZoomChange,
  onCellWidthChange,
  onVisibleRangeChange,
  baseCellSize = 12,
  cellWidthMultiplier = 1,
  cellHeightMultiplier = 1,
  showGridLines = true,
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 400 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartOffset, setDragStartOffset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const LABEL_MARGIN_LEFT = 100;
  const LABEL_MARGIN_TOP = 10;
  const PADDING = 10;

  const cellHeight = baseCellSize * zoomLevel * cellHeightMultiplier;
  const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
  const numRows = matrix.length;
  const numCols = regions.length;
  const totalMatrixWidth = numCols * cellWidth;
  const totalMatrixHeight = numRows * cellHeight;
  const viewportWidth = containerSize.width - LABEL_MARGIN_LEFT - PADDING * 2;

  const maxScrollOffset = Math.max(0, totalMatrixWidth - viewportWidth);
  const scrollX = scrollOffset * maxScrollOffset;
  const startCol = Math.floor(scrollX / cellWidth);
  const endCol = Math.min(numCols, Math.ceil((scrollX + viewportWidth) / cellWidth) + 1);

  // Monitor container size
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const requiredHeight = totalMatrixHeight + LABEL_MARGIN_TOP + PADDING * 2;
        setContainerSize({
          width: rect.width || 800,
          height: requiredHeight,
        });
      }
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [totalMatrixHeight]);

  // Report visible range changes
  useEffect(() => {
    if (onVisibleRangeChange && regions.length > 0) {
      const allDataVisible = maxScrollOffset === 0;
      if (allDataVisible) {
        onVisibleRangeChange({
          startRegionIdx: 0,
          endRegionIdx: regions.length - 1,
          startCol: 0,
          endCol: regions.length - 1,
        });
      } else {
        const lastVisibleCol = Math.min(endCol - 2, numCols - 2);
        onVisibleRangeChange({
          startRegionIdx: startCol,
          endRegionIdx: lastVisibleCol,
          startCol,
          endCol: lastVisibleCol,
        });
      }
    }
  }, [onVisibleRangeChange, startCol, endCol, numCols, regions, maxScrollOffset]);

  // Get cell color based on value
  const getCellColor = useCallback((value: number): string => {
    switch (value) {
      case 1: return COLOR_PARENT1;
      case 2: return COLOR_PARENT2;
      case 3: return COLOR_BOTH;
      default: return COLOR_EMPTY;
    }
  }, []);

  // Draw a parent path on the canvas
  const drawParentPath = useCallback((
    ctx: CanvasRenderingContext2D,
    pathData: number[],
    color: string,
    dashPattern: number[],
  ) => {
    if (pathData.length < 2) return;

    // Collect visible points
    const points: { x: number; y: number }[] = [];
    for (let col = startCol; col < endCol && col < pathData.length; col++) {
      const rowIdx = pathData[col];
      const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth + cellWidth / 2;
      const y = LABEL_MARGIN_TOP + PADDING + rowIdx * cellHeight + cellHeight / 2;
      points.push({ x, y });
    }

    if (points.length < 2) return;

    // Decimate: keep only transition boundary points to reduce noise when zoomed out
    const visibleCount = endCol - startCol;
    let drawPoints = points;
    if (visibleCount > 2) {
      const decimated: { x: number; y: number }[] = [points[0]];
      for (let i = 1; i < points.length - 1; i++) {
        const prevRow = pathData[startCol + i - 1];
        const currRow = pathData[startCol + i];
        const nextRow = pathData[startCol + i + 1];
        if (currRow !== prevRow || currRow !== nextRow) {
          decimated.push(points[i]);
        }
      }
      decimated.push(points[points.length - 1]);
      drawPoints = decimated;
    }

    // Adaptive sizing based on cell dimensions
    const adaptiveLineWidth = Math.max(1, Math.min(2.5, cellWidth * 0.4));
    const dashScale = Math.max(0.3, Math.min(1, cellWidth / 8));
    const scaledDash = dashPattern.map(d => d * dashScale);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = adaptiveLineWidth;
    ctx.setLineDash(scaledDash);
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.moveTo(drawPoints[0].x, drawPoints[0].y);
    for (let i = 1; i < drawPoints.length; i++) {
      ctx.lineTo(drawPoints[i].x, drawPoints[i].y);
    }
    ctx.stroke();

    // Draw circles at transition nodes only when cells are large enough to be useful
    if (cellWidth >= 8) {
      const circleRadius = Math.max(2, Math.min(3.5, cellWidth * 0.3));
      const strokeWidth = Math.max(0.5, Math.min(1.5, cellWidth * 0.15));
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.9;
      for (const pt of drawPoints) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, circleRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = strokeWidth;
        ctx.stroke();
      }
    }
    ctx.restore();
  }, [startCol, endCol, cellWidth, cellHeight, LABEL_MARGIN_LEFT, LABEL_MARGIN_TOP, PADDING]);

  // Export to PNG
  useImperativeHandle(ref, () => ({
    exportToPng: async (options: BEDExportOptions) => {
      try {
        const { fileId, chromosome } = options;

        // Resolve export column range from position overrides or current viewport
        let expStartCol = startCol;
        let expEndCol = endCol;
        if (options.startPosition != null && options.endPosition != null) {
          expStartCol = regions.findIndex(r => r.start >= options.startPosition!);
          if (expStartCol < 0) expStartCol = 0;
          expEndCol = regions.findIndex(r => r.start > options.endPosition!);
          if (expEndCol < 0) expEndCol = numCols;
        }

        const expStartRegion = regions[expStartCol] ?? regions[0];
        const expLastVisibleCol = Math.min(expEndCol - 1, numCols - 1);
        const expEndRegion = regions[expLastVisibleCol] ?? regions[regions.length - 1];

        const titleText = options.title ??
          `${fileId} | ${chromosome} | ${expStartRegion.start.toLocaleString()} - ${expEndRegion.end.toLocaleString()} bp`;

        const TITLE_HEIGHT = 50;
        const includeCellValueLegend = options.includeCellValueLegend !== false;
        const includePathLegend = options.includePathLegend !== false;
        const includeLegend = includeCellValueLegend || includePathLegend;
        const showP1 = options.pathVisibility
          ? (options.pathVisibility['Parent 1 Path'] ?? showParent1Path)
          : showParent1Path;
        const showP2 = options.pathVisibility
          ? (options.pathVisibility['Parent 2 Path'] ?? showParent2Path)
          : showParent2Path;
        const hasVisiblePathsExport = showP1 || showP2;
        const LEGEND_HEIGHT = includeLegend ? 50 : 0;

        const expW = options.width ?? containerSize.width;
        const expH = (options.height ?? containerSize.height) + TITLE_HEIGHT + LEGEND_HEIGHT;

        const heatmapAreaHeight = expH - TITLE_HEIGHT - LEGEND_HEIGHT;

        const numVisibleCols = expEndCol - expStartCol;
        const heatmapDrawWidth = expW - LABEL_MARGIN_LEFT - PADDING * 2;
        const heatmapDrawHeight = heatmapAreaHeight - LABEL_MARGIN_TOP - PADDING * 2;
        const expCellWidth = numVisibleCols > 0 ? heatmapDrawWidth / numVisibleCols : cellWidth;
        const expCellHeight = numRows > 0 ? heatmapDrawHeight / numRows : cellHeight;
        const expTotalMatrixHeight = numRows * expCellHeight;

        const exportCanvas = document.createElement('canvas');
        const exportScale = options.scale ?? 3;
        exportCanvas.width = expW * exportScale;
        exportCanvas.height = expH * exportScale;

        const ctx = exportCanvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(exportScale, exportScale);

        const surfaceColor = '#ffffff';
        const onSurfaceColor = '#1d1b20';
        const outlineVariantColor = '#cac4d0';

        ctx.fillStyle = surfaceColor;
        ctx.fillRect(0, 0, expW, expH);

        // Title
        ctx.fillStyle = onSurfaceColor;
        ctx.font = 'bold 16px "Roboto", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(titleText, expW / 2, TITLE_HEIGHT / 2);

        ctx.save();
        ctx.translate(0, TITLE_HEIGHT);

        // Y-axis labels
        ctx.fillStyle = onSurfaceColor;
        ctx.font = `${Math.min(11, expCellHeight * 0.8)}px "Roboto", sans-serif`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * expCellHeight + expCellHeight / 2;
          if (y < heatmapAreaHeight) {
            const label = parentNames[row] || `Parent ${row}`;
            const truncated = label.length > 12 ? label.substring(0, 10) + '...' : label;
            ctx.fillText(truncated, LABEL_MARGIN_LEFT - 8, y);
          }
        }

        const expViewportWidth = numVisibleCols * expCellWidth;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(LABEL_MARGIN_LEFT + PADDING, LABEL_MARGIN_TOP + PADDING, expViewportWidth, expTotalMatrixHeight);

        // Cells
        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * expCellHeight;
          if (y > heatmapAreaHeight) break;
          for (let col = expStartCol; col < expEndCol; col++) {
            const x = LABEL_MARGIN_LEFT + PADDING + (col - expStartCol) * expCellWidth;
            if (x > expW - PADDING) break;
            const value = matrix[row]?.[col] ?? 0;
            ctx.fillStyle = getCellColor(value);
            ctx.fillRect(x, y, expCellWidth - (showGridLines ? 1 : 0), expCellHeight - (showGridLines ? 1 : 0));
          }
        }

        // Grid lines
        if (showGridLines && Math.min(expCellWidth, expCellHeight) >= 4) {
          ctx.strokeStyle = outlineVariantColor;
          ctx.lineWidth = 0.5;
          for (let col = expStartCol; col <= expEndCol; col++) {
            const x = LABEL_MARGIN_LEFT + PADDING + (col - expStartCol) * expCellWidth;
            if (x <= expW - PADDING) {
              ctx.beginPath();
              ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
              ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + expTotalMatrixHeight);
              ctx.stroke();
            }
          }
          for (let row = 0; row <= numRows; row++) {
            const y = LABEL_MARGIN_TOP + PADDING + row * expCellHeight;
            if (y <= heatmapAreaHeight) {
              ctx.beginPath();
              ctx.moveTo(LABEL_MARGIN_LEFT + PADDING, y);
              ctx.lineTo(LABEL_MARGIN_LEFT + PADDING + expViewportWidth, y);
              ctx.stroke();
            }
          }
        }

        // Paths with optional visibility overrides (showP1, showP2 already defined above)
        const exportPathEntries: { data: number[]; color: string; dash: number[] }[] = [];
        if (showP1) exportPathEntries.push({ data: parent1Path, color: PATH_COLOR_PARENT1, dash: [8, 4] });
        if (showP2) exportPathEntries.push({ data: parent2Path, color: PATH_COLOR_PARENT2, dash: [4, 8] });

        for (const { data: pathData, color, dash: dashPattern } of exportPathEntries) {
          if (pathData.length < 2) continue;

          const points: { x: number; y: number }[] = [];
          for (let col = expStartCol; col < expEndCol && col < pathData.length; col++) {
            const rowIdx = pathData[col];
            const x = LABEL_MARGIN_LEFT + PADDING + (col - expStartCol) * expCellWidth + expCellWidth / 2;
            const y = LABEL_MARGIN_TOP + PADDING + rowIdx * expCellHeight + expCellHeight / 2;
            points.push({ x, y });
          }
          if (points.length < 2) continue;

          const decimated: { x: number; y: number }[] = [points[0]];
          for (let i = 1; i < points.length - 1; i++) {
            const prevRow = pathData[expStartCol + i - 1];
            const currRow = pathData[expStartCol + i];
            const nextRow = pathData[expStartCol + i + 1];
            if (currRow !== prevRow || currRow !== nextRow) {
              decimated.push(points[i]);
            }
          }
          decimated.push(points[points.length - 1]);

          const adaptiveLineWidth = Math.max(1, Math.min(2.5, expCellWidth * 0.4));
          const dashScale = Math.max(0.3, Math.min(1, expCellWidth / 8));
          const scaledDash = dashPattern.map(d => d * dashScale);

          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = adaptiveLineWidth;
          ctx.setLineDash(scaledDash);
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.moveTo(decimated[0].x, decimated[0].y);
          for (let i = 1; i < decimated.length; i++) {
            ctx.lineTo(decimated[i].x, decimated[i].y);
          }
          ctx.stroke();

          if (expCellWidth >= 8) {
            const circleRadius = Math.max(2, Math.min(3.5, expCellWidth * 0.3));
            const strokeWidth = Math.max(0.5, Math.min(1.5, expCellWidth * 0.15));
            ctx.setLineDash([]);
            ctx.globalAlpha = 0.9;
            for (const pt of decimated) {
              ctx.beginPath();
              ctx.arc(pt.x, pt.y, circleRadius, 0, Math.PI * 2);
              ctx.fillStyle = color;
              ctx.fill();
              ctx.strokeStyle = '#fff';
              ctx.lineWidth = strokeWidth;
              ctx.stroke();
            }
          }
          ctx.restore();
        }

        ctx.restore();

        // Legend
        if (includeLegend) {
          const legendY = TITLE_HEIGHT + heatmapAreaHeight + LEGEND_HEIGHT / 2;
          ctx.font = '12px "Roboto", sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          const legendStartX = expW / 2 - 180;
          let nextX = legendStartX;

          if (includeCellValueLegend) {
            ctx.fillStyle = COLOR_PARENT1;
            ctx.fillRect(legendStartX, legendY - 8, 16, 16);
            ctx.fillStyle = onSurfaceColor;
            ctx.fillText('Parent 1', legendStartX + 24, legendY);

            ctx.fillStyle = COLOR_PARENT2;
            ctx.fillRect(legendStartX + 110, legendY - 8, 16, 16);
            ctx.fillStyle = onSurfaceColor;
            ctx.fillText('Parent 2', legendStartX + 134, legendY);

            ctx.fillStyle = COLOR_BOTH;
            ctx.fillRect(legendStartX + 220, legendY - 8, 16, 16);
            ctx.fillStyle = onSurfaceColor;
            ctx.fillText('Both', legendStartX + 244, legendY);

            ctx.fillStyle = COLOR_EMPTY;
            ctx.fillRect(legendStartX + 300, legendY - 8, 16, 16);
            ctx.strokeStyle = outlineVariantColor;
            ctx.strokeRect(legendStartX + 300, legendY - 8, 16, 16);
            ctx.fillStyle = onSurfaceColor;
            ctx.fillText('Empty', legendStartX + 324, legendY);
            nextX = legendStartX + 380;
          }

          // Path overlay legend (when paths are visible) — side by side
          if (includePathLegend && hasVisiblePathsExport) {
            nextX += 16; // gap before path items
            if (showP1) {
              ctx.fillStyle = PATH_COLOR_PARENT1;
              ctx.fillRect(nextX, legendY - 6, 12, 3);
              ctx.fillStyle = onSurfaceColor;
              ctx.font = '11px "Roboto", sans-serif';
              ctx.fillText('Parent 1 Path', nextX + 18, legendY);
              nextX += 18 + ctx.measureText('Parent 1 Path').width + 16;
            }
            if (showP2) {
              ctx.fillStyle = PATH_COLOR_PARENT2;
              ctx.fillRect(nextX, legendY - 6, 12, 3);
              ctx.fillStyle = onSurfaceColor;
              ctx.fillText('Parent 2 Path', nextX + 18, legendY);
            }
          }
        }

        const sanitizedFileId = fileId.replace(/[^a-zA-Z0-9_-]/g, '_');
        const sanitizedChromosome = chromosome.replace(/[^a-zA-Z0-9_-]/g, '_');
        const suggestedFilename = `${sanitizedFileId}_${sanitizedChromosome}_heatmap.png`;

        const blob = await new Promise<Blob | null>((resolve) => {
          exportCanvas.toBlob(resolve, 'image/png');
        });
        if (!blob) return;

        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const backend = await getBackend();
        await backend.saveFile(uint8Array, {
          defaultName: suggestedFilename,
          filters: [{ name: 'PNG Image', extensions: ['png'] }],
        });
      } catch (error) {
        console.error('Failed to export BED heatmap PNG:', error);
        throw error;
      }
    }
  }), [
    matrix, regions, parentNames, parent1Path, parent2Path, showParent1Path, showParent2Path,
    containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols,
    showGridLines, getCellColor,
  ]);

  // Draw the heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = containerSize.width * dpr;
    canvas.height = containerSize.height * dpr;
    ctx.scale(dpr, dpr);

    const computedStyle = getComputedStyle(document.documentElement);
    const surfaceColor = computedStyle.getPropertyValue('--md-sys-color-surface').trim() || '#fef7ff';
    const onSurfaceColor = computedStyle.getPropertyValue('--md-sys-color-on-surface').trim() || '#1d1b20';
    const outlineVariantColor = computedStyle.getPropertyValue('--md-sys-color-outline-variant').trim() || '#cac4d0';

    // Clear
    ctx.fillStyle = surfaceColor;
    ctx.fillRect(0, 0, containerSize.width, containerSize.height);

    // Y-axis labels
    ctx.fillStyle = onSurfaceColor;
    ctx.font = `${Math.min(11, cellHeight * 0.8)}px "Roboto", sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight + cellHeight / 2;
      if (y < containerSize.height) {
        const label = parentNames[row] || `Parent ${row}`;
        const truncated = label.length > 12 ? label.substring(0, 10) + '...' : label;
        ctx.fillText(truncated, LABEL_MARGIN_LEFT - 8, y);
      }
    }

    // White background behind cells
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(LABEL_MARGIN_LEFT + PADDING, LABEL_MARGIN_TOP + PADDING, viewportWidth, totalMatrixHeight);

    // Draw cells
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
      if (y > containerSize.height) break;
      for (let col = startCol; col < endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x > containerSize.width - PADDING) break;
        const value = matrix[row]?.[col] ?? 0;
        ctx.fillStyle = getCellColor(value);
        ctx.fillRect(x, y, cellWidth - (showGridLines ? 1 : 0), cellHeight - (showGridLines ? 1 : 0));
      }
    }

    // Grid lines
    if (showGridLines && Math.min(cellWidth, cellHeight) >= 4) {
      ctx.strokeStyle = outlineVariantColor;
      ctx.lineWidth = 0.5;
      for (let col = startCol; col <= endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x <= containerSize.width - PADDING) {
          ctx.beginPath();
          ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
          ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + totalMatrixHeight);
          ctx.stroke();
        }
      }
      for (let row = 0; row <= numRows; row++) {
        const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
        if (y <= containerSize.height) {
          ctx.beginPath();
          ctx.moveTo(LABEL_MARGIN_LEFT + PADDING, y);
          ctx.lineTo(LABEL_MARGIN_LEFT + PADDING + viewportWidth, y);
          ctx.stroke();
        }
      }
    }

    // Draw parent paths
    if (showParent1Path) {
      drawParentPath(ctx, parent1Path, PATH_COLOR_PARENT1, [8, 4]);
    }
    if (showParent2Path) {
      drawParentPath(ctx, parent2Path, PATH_COLOR_PARENT2, [4, 8]);
    }

    // Hover highlight
    if (hoveredCell) {
      const { row, col } = hoveredCell;
      if (col >= startCol && col < endCol) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
        const tertiaryColor = computedStyle.getPropertyValue('--md-sys-color-tertiary').trim() || '#7d5260';
        ctx.strokeStyle = tertiaryColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      }
    }
  }, [matrix, regions, parentNames, parent1Path, parent2Path, showParent1Path, showParent2Path,
      containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols,
      totalMatrixHeight, viewportWidth, showGridLines, getCellColor, hoveredCell, drawParentPath]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStartX(e.clientX);
      setDragStartOffset(scrollOffset);
    }
  }, [scrollOffset]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    if (isDragging) {
      const deltaX = e.clientX - dragStartX;
      const deltaOffset = -deltaX / (maxScrollOffset || 1);
      const newOffset = Math.max(0, Math.min(1, dragStartOffset + deltaOffset));
      onScrollChange(newOffset);
    }

    const cellX = x - LABEL_MARGIN_LEFT - PADDING;
    const cellY = y - LABEL_MARGIN_TOP - PADDING;
    if (cellX >= 0 && cellY >= 0) {
      const col = startCol + Math.floor(cellX / cellWidth);
      const row = Math.floor(cellY / cellHeight);
      if (row >= 0 && row < numRows && col >= 0 && col < numCols) {
        const value = matrix[row]?.[col] ?? 0;
        setHoveredCell({ row, col, value });
      } else {
        setHoveredCell(null);
      }
    } else {
      setHoveredCell(null);
    }
  }, [isDragging, dragStartX, dragStartOffset, maxScrollOffset, onScrollChange,
      cellWidth, cellHeight, startCol, numRows, numCols, matrix]);

  const handleMouseUp = useCallback(() => { setIsDragging(false); }, []);
  const handleMouseLeave = useCallback(() => { setIsDragging(false); setHoveredCell(null); }, []);

  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // Native wheel events
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleNativeWheel = (e: WheelEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
        // Column width with Ctrl/Cmd + Shift + scroll
        e.preventDefault();
        if (onCellWidthChange) {
          const rawDelta = e.deltaY !== 0 ? e.deltaY : e.deltaX;
          const delta = rawDelta > 0 ? -0.05 : 0.05;
          onCellWidthChange(Math.max(0.05, Math.min(4, cellWidthMultiplier + delta)));
        }
      } else if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        if (onZoomChange) {
          const delta = e.deltaY > 0 ? -0.1 : 0.1;
          const newZoom = Math.max(0.25, Math.min(4, zoomLevel + delta));
          if (newZoom === zoomLevel) return;

          // Zoom toward cursor: keep the data position under the mouse fixed
          const rect = canvas.getBoundingClientRect();
          const cursorX = e.clientX - rect.left - LABEL_MARGIN_LEFT - PADDING;
          const dataX = scrollX + cursorX;

          const ratio = newZoom / zoomLevel;
          const newDataX = dataX * ratio;
          const newCellWidth = baseCellSize * newZoom * cellWidthMultiplier;
          const newTotalWidth = numCols * newCellWidth;
          const newMaxScroll = Math.max(0, newTotalWidth - viewportWidth);
          const newScrollX = newDataX - cursorX;
          const newOffset = newMaxScroll > 0
            ? Math.max(0, Math.min(1, newScrollX / newMaxScroll))
            : 0;

          onZoomChange(newZoom);
          onScrollChange(newOffset);
        }
      } else if (e.shiftKey) {
        e.preventDefault();
        const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        const scrollDelta = delta / (maxScrollOffset || 1);
        const newOffset = Math.max(0, Math.min(1, scrollOffset + scrollDelta * 0.1));
        onScrollChange(newOffset);
      }
    };
    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleNativeWheel);
  }, [scrollOffset, scrollX, maxScrollOffset, zoomLevel, baseCellSize, cellWidthMultiplier,
      numCols, viewportWidth, onScrollChange, onZoomChange, onCellWidthChange,
      LABEL_MARGIN_LEFT, PADDING]);

  const requiredHeight = totalMatrixHeight + LABEL_MARGIN_TOP + PADDING * 2;

  return (
    <div
      ref={containerRef}
      className="heatmap-canvas-container"
      style={{
        width: '100%',
        height: requiredHeight,
        position: 'relative',
        cursor: isDragging ? 'grabbing' : 'grab',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: containerSize.width, height: containerSize.height }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {hoveredCell && (
        <TooltipContent
          mousePos={mousePos}
          containerSize={containerSize}
          parentNames={parentNames}
          regions={regions}
          hoveredCell={hoveredCell}
        />
      )}
    </div>
  );
});

BEDHeatmapCanvas.displayName = 'BEDHeatmapCanvas';

export default BEDHeatmapCanvas;
