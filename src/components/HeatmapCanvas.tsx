import React, { useRef, useEffect, useCallback, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import { getBackend } from '../platform';

/** Visible range information reported by the canvas */
export interface VisibleRange {
  startPos: number;
  endPos: number;
  startCol: number;
  endCol: number;
}

/** Export options for PNG generation */
export interface ExportOptions {
  /** File ID (typically the filename) */
  fileId: string;
  /** Chromosome being viewed */
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

/** Methods exposed by HeatmapCanvas via ref */
export interface HeatmapCanvasHandle {
  /** Export the current visible view as a PNG image */
  exportToPng: (options: ExportOptions) => Promise<void>;
}

/** A path overlay line drawn on top of the heatmap */
export interface PathOverlay {
  /** Row index per position (column) in the heatmap matrix */
  data: number[];
  /** Stroke color */
  color: string;
  /** Canvas dash pattern */
  dashPattern: number[];
  /** Base line width multiplier (observed paths thicker, predicted thinner) */
  lineWidth: number;
  /** Display label */
  label: string;
  /** Whether this path is currently visible */
  visible: boolean;
}

interface HeatmapCanvasProps {
  /** Matrix data: rows = gametes, columns = positions. Values are counts (0 = no data) */
  matrix: number[][];
  /** Position labels for x-axis (binned positions) */
  positions: number[];
  /** Gamete names for y-axis */
  gameteNames: string[];
  /** Current zoom level (1 = 100%) */
  zoomLevel: number;
  /** Horizontal scroll offset (0-1 range, percentage of total width) */
  scrollOffset: number;
  /** Callback when scroll position changes via drag/wheel */
  onScrollChange: (offset: number) => void;
  /** Callback when zoom changes via wheel */
  onZoomChange?: (zoom: number) => void;
  /** Callback when visible range changes */
  onVisibleRangeChange?: (range: VisibleRange) => void;
  /** Cell height at zoom level 1 */
  baseCellSize?: number;
  /** Cell width multiplier (relative to baseCellSize, default 1) */
  cellWidthMultiplier?: number;
  /** Callback when cell width multiplier changes via Ctrl+Shift+scroll */
  onCellWidthChange?: (multiplier: number) => void;
  /** Cell height multiplier (relative to baseCellSize, default 1) */
  cellHeightMultiplier?: number;
  /** Whether to show grid lines */
  showGridLines?: boolean;
  /** Color scheme: 'binary' (gray/white) or 'intensity' (color gradient) */
  colorScheme?: 'binary' | 'intensity';
  /** Optional path overlays (true/predicted paths from NumPy data) */
  pathOverlays?: PathOverlay[];
}

// Tooltip component with smart positioning to avoid overflow
interface TooltipContentProps {
  mousePos: { x: number; y: number };
  containerSize: { width: number; height: number };
  gameteNames: string[];
  positions: number[];
  hoveredCell: { row: number; col: number; value: number };
  pathOverlays?: PathOverlay[];
}

const TooltipContent: React.FC<TooltipContentProps> = ({
  mousePos,
  containerSize,
  gameteNames,
  positions,
  hoveredCell,
  pathOverlays,
}) => {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tooltipSize, setTooltipSize] = useState({ width: 0, height: 0 });

  // Measure tooltip size after render
  useEffect(() => {
    if (tooltipRef.current) {
      const rect = tooltipRef.current.getBoundingClientRect();
      setTooltipSize({ width: rect.width, height: rect.height });
    }
  }, [hoveredCell]);

  // Calculate position with edge detection
  const tooltipPosition = useMemo(() => {
    const offset = 15;
    const padding = 8; // Minimum distance from container edge
    
    // Default: position to the right and slightly above cursor
    let left = mousePos.x + offset;
    let top = mousePos.y - 10;
    
    // Check right edge overflow - flip to left side of cursor
    if (tooltipSize.width > 0 && left + tooltipSize.width + padding > containerSize.width) {
      left = mousePos.x - tooltipSize.width - offset;
    }
    
    // Check bottom edge overflow - flip to above cursor
    if (tooltipSize.height > 0 && top + tooltipSize.height + padding > containerSize.height) {
      top = mousePos.y - tooltipSize.height - offset;
    }
    
    // Ensure tooltip doesn't go off the left edge
    if (left < padding) {
      left = padding;
    }
    
    // Ensure tooltip doesn't go off the top edge
    if (top < padding) {
      top = padding;
    }
    
    return { left, top };
  }, [mousePos, tooltipSize, containerSize]);

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
      <div><strong>{gameteNames[hoveredCell.row]}</strong></div>
      <div>Position: {positions[hoveredCell.col]?.toLocaleString()}</div>
      <div>Count: {hoveredCell.value}</div>
      {pathOverlays && pathOverlays.filter(o => o.visible && o.data[hoveredCell.col] === hoveredCell.row).length > 0 && (
        <div style={{ marginTop: 3, borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 3 }}>
          {pathOverlays.filter(o => o.visible && o.data[hoveredCell.col] === hoveredCell.row).map(o => (
            <div key={o.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                display: 'inline-block',
                width: 10,
                height: 3,
                background: o.color,
                borderRadius: 1,
                flexShrink: 0,
              }} />
              <span>{o.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const HeatmapCanvas = forwardRef<HeatmapCanvasHandle, HeatmapCanvasProps>(({
  matrix,
  positions,
  gameteNames,
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
  colorScheme = 'binary',
  pathOverlays,
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 400 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartOffset, setDragStartOffset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Layout constants
  const LABEL_MARGIN_LEFT = 100; // Space for gamete labels
  const LABEL_MARGIN_TOP = 10;   // Minimal top margin (no position labels)
  const PADDING = 10;

  // Calculate dimensions - separate width and height for cells
  const cellHeight = baseCellSize * zoomLevel * cellHeightMultiplier;
  const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
  const numRows = matrix.length;
  const numCols = positions.length;
  const totalMatrixWidth = numCols * cellWidth;
  const totalMatrixHeight = numRows * cellHeight;
  const viewportWidth = containerSize.width - LABEL_MARGIN_LEFT - PADDING * 2;

  // Calculate visible range based on scroll offset
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
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, [totalMatrixHeight]);

  // Report visible range changes to parent
  useEffect(() => {
    if (onVisibleRangeChange && positions.length > 0) {
      // When all data fits in the viewport, report the full range
      const allDataVisible = maxScrollOffset === 0;
      
      if (allDataVisible) {
        onVisibleRangeChange({
          startPos: positions[0],
          endPos: positions[positions.length - 1],
          startCol: 0,
          endCol: positions.length - 1,
        });
      } else {
        // "end" column + padding cuts off last actual column (-1) hence why we subtract 2 here:
        const lastVisibleCol = Math.min(endCol - 2, numCols - 2);
        const startPos = positions[startCol] ?? positions[0];
        const endPos = positions[lastVisibleCol] ?? positions[positions.length - 2];
        
        onVisibleRangeChange({
          startPos,
          endPos,
          startCol,
          endCol: lastVisibleCol,
        });
      }
    }
  }, [onVisibleRangeChange, startCol, endCol, numCols, positions, maxScrollOffset]);

  // Color stops for the plasma-style gradient: black → purple → red → orange → yellow
  const colorStops = React.useMemo(() => [
    { pos: 0.0,  r: 0,   g: 0,   b: 0   },   // black
    { pos: 0.2,  r: 63,  g: 0,   b: 92  },   // dark purple
    { pos: 0.4,  r: 148, g: 23,  b: 81  },   // magenta-purple
    { pos: 0.55, r: 199, g: 52,  b: 44  },   // red
    { pos: 0.7,  r: 237, g: 117, b: 15  },   // orange
    { pos: 0.85, r: 251, g: 191, b: 36  },   // light orange
    { pos: 1.0,  r: 252, g: 253, b: 141 },   // yellow
  ], []);

  // Interpolate between color stops
  const interpolateColor = useCallback((t: number): string => {
    // Clamp t to [0, 1]
    t = Math.max(0, Math.min(1, t));
    
    // Find the two color stops to interpolate between
    let lower = colorStops[0];
    let upper = colorStops[colorStops.length - 1];
    
    for (let i = 0; i < colorStops.length - 1; i++) {
      if (t >= colorStops[i].pos && t <= colorStops[i + 1].pos) {
        lower = colorStops[i];
        upper = colorStops[i + 1];
        break;
      }
    }
    
    // Calculate interpolation factor
    const range = upper.pos - lower.pos;
    const factor = range > 0 ? (t - lower.pos) / range : 0;
    
    // Interpolate RGB values
    const r = Math.round(lower.r + (upper.r - lower.r) * factor);
    const g = Math.round(lower.g + (upper.g - lower.g) * factor);
    const b = Math.round(lower.b + (upper.b - lower.b) * factor);
    
    return `rgb(${r}, ${g}, ${b})`;
  }, [colorStops]);

  // Get color for cell value - primaryColor is passed in since CSS vars don't work in Canvas
  const getCellColor = useCallback((value: number, maxValue: number, primaryColor: string): string => {
    if (value === 0) {
      return '#ffffff'; // White for no data
    }
    
    if (colorScheme === 'binary') {
      return primaryColor;
    }
    
    // Intensity-based coloring with logarithmic normalization
    // Log normalization prevents large values from dwarfing smaller ones
    // Using log1p (log(1 + x)) to handle values smoothly including small values
    const logValue = Math.log1p(value);
    const logMax = Math.log1p(maxValue);
    const intensity = logMax > 0 ? logValue / logMax : 0;
    
    return interpolateColor(intensity);
  }, [colorScheme, interpolateColor]);

  // Get max value for intensity scaling
  const maxValue = React.useMemo(() => {
    let max = 1;
    for (const row of matrix) {
      for (const val of row) {
        if (val > max) max = val;
      }
    }
    return max;
  }, [matrix]);

  // Draw a path overlay line on the canvas (adapted from BEDHeatmapCanvas)
  const drawPath = useCallback((
    ctx: CanvasRenderingContext2D,
    pathData: number[],
    color: string,
    dashPattern: number[],
    lineWidthMultiplier: number,
  ) => {
    if (pathData.length < 2) return;

    const points: { x: number; y: number }[] = [];
    for (let col = startCol; col < endCol && col < pathData.length; col++) {
      const rowIdx = pathData[col];
      if (rowIdx < 0 || rowIdx >= numRows) continue;
      const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth + cellWidth / 2;
      const y = LABEL_MARGIN_TOP + PADDING + rowIdx * cellHeight + cellHeight / 2;
      points.push({ x, y });
    }

    if (points.length < 2) return;

    // Decimate: keep only transition boundary points to reduce noise when zoomed out
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

    const baseWidth = Math.max(2, Math.min(4, cellWidth * 0.5));
    const adaptiveLineWidth = baseWidth * lineWidthMultiplier;
    const dashScale = Math.max(0.5, Math.min(1.5, cellWidth / 6));
    const scaledDash = dashPattern.map(d => d * dashScale);

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Build the path once, then stroke twice
    ctx.beginPath();
    ctx.moveTo(decimated[0].x, decimated[0].y);
    for (let i = 1; i < decimated.length; i++) {
      ctx.lineTo(decimated[i].x, decimated[i].y);
    }

    // Pass 1: dark outline for contrast against any heatmap background
    ctx.setLineDash(scaledDash);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.lineWidth = adaptiveLineWidth + 2;
    ctx.stroke();

    // Pass 2: colored path on top
    ctx.strokeStyle = color;
    ctx.lineWidth = adaptiveLineWidth;
    ctx.stroke();

    if (cellWidth >= 8) {
      const circleRadius = Math.max(2.5, Math.min(4, cellWidth * 0.35));
      const strokeWidth = Math.max(0.5, Math.min(1.5, cellWidth * 0.15));
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
  }, [startCol, endCol, cellWidth, cellHeight, numRows, LABEL_MARGIN_LEFT, LABEL_MARGIN_TOP, PADDING]);

  // Export to PNG functionality
  useImperativeHandle(ref, () => ({
    exportToPng: async (options: ExportOptions) => {
      try {
        const { fileId, chromosome } = options;

        // Resolve export column range from position overrides or current viewport
        let expStartCol = startCol;
        let expEndCol = endCol;
        if (options.startPosition != null && options.endPosition != null) {
          expStartCol = positions.findIndex(p => p >= options.startPosition!);
          if (expStartCol < 0) expStartCol = 0;
          expEndCol = positions.findIndex(p => p > options.endPosition!);
          if (expEndCol < 0) expEndCol = numCols;
        }

        const expStartPos = positions[expStartCol] ?? positions[0];
        const expLastVisibleCol = Math.min(expEndCol - 1, numCols - 1);
        const expEndPos = positions[expLastVisibleCol] ?? positions[positions.length - 1];

        const titleText = options.title ??
          `${fileId} | ${chromosome} | ${expStartPos.toLocaleString()} - ${expEndPos.toLocaleString()} rbp`;

        const TITLE_HEIGHT = 50;
        const includeCellValueLegend = options.includeCellValueLegend !== false;
        const includePathLegend = options.includePathLegend !== false;
        const includeLegend = includeCellValueLegend || includePathLegend;
        const resolvePathVisible = (overlay: PathOverlay): boolean => {
          if (options.pathVisibility && overlay.label in options.pathVisibility) {
            return options.pathVisibility[overlay.label];
          }
          return overlay.visible;
        };
        const hasVisiblePathsExport = pathOverlays?.some(o => resolvePathVisible(o)) ?? false;
        const LEGEND_HEIGHT = includeLegend ? 40 : 0;

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
        const primaryColor = '#6750a4';
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

        // Gamete labels (y-axis)
        ctx.fillStyle = onSurfaceColor;
        ctx.font = `${Math.min(11, expCellHeight * 0.8)}px "Roboto", sans-serif`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * expCellHeight + expCellHeight / 2;
          if (y < heatmapAreaHeight) {
            const label = gameteNames[row] || `Gamete ${row}`;
            const truncatedLabel = label.length > 12 ? label.substring(0, 10) + '...' : label;
            ctx.fillText(truncatedLabel, LABEL_MARGIN_LEFT - 8, y);
          }
        }

        const expViewportWidth = numVisibleCols * expCellWidth;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(
          LABEL_MARGIN_LEFT + PADDING,
          LABEL_MARGIN_TOP + PADDING,
          expViewportWidth,
          expTotalMatrixHeight
        );

        // Cells
        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * expCellHeight;
          if (y > heatmapAreaHeight) break;

          for (let col = expStartCol; col < expEndCol; col++) {
            const x = LABEL_MARGIN_LEFT + PADDING + (col - expStartCol) * expCellWidth;
            if (x > expW - PADDING) break;

            const value = matrix[row]?.[col] ?? 0;
            ctx.fillStyle = getCellColor(value, maxValue, primaryColor);
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

        // Determine per-path visibility with optional overrides (resolvePathVisible already defined above)
        if (hasVisiblePathsExport) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
          ctx.fillRect(
            LABEL_MARGIN_LEFT + PADDING,
            LABEL_MARGIN_TOP + PADDING,
            expViewportWidth,
            expTotalMatrixHeight,
          );
        }

        if (pathOverlays) {
          for (const overlay of pathOverlays) {
            if (!resolvePathVisible(overlay)) continue;
            const pathData = overlay.data;
            if (pathData.length < 2) continue;

            const points: { x: number; y: number }[] = [];
            for (let col = expStartCol; col < expEndCol && col < pathData.length; col++) {
              const rowIdx = pathData[col];
              if (rowIdx < 0 || rowIdx >= numRows) continue;
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

            const baseWidth = Math.max(2, Math.min(4, expCellWidth * 0.5));
            const adaptiveLineWidth = baseWidth * overlay.lineWidth;
            const dashScale = Math.max(0.5, Math.min(1.5, expCellWidth / 6));
            const scaledDash = overlay.dashPattern.map(d => d * dashScale);

            ctx.save();
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';

            ctx.beginPath();
            ctx.moveTo(decimated[0].x, decimated[0].y);
            for (let i = 1; i < decimated.length; i++) {
              ctx.lineTo(decimated[i].x, decimated[i].y);
            }

            ctx.setLineDash(scaledDash);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.lineWidth = adaptiveLineWidth + 2;
            ctx.stroke();

            ctx.strokeStyle = overlay.color;
            ctx.lineWidth = adaptiveLineWidth;
            ctx.stroke();

            if (expCellWidth >= 8) {
              const circleRadius = Math.max(2.5, Math.min(4, expCellWidth * 0.35));
              const strokeWidth = Math.max(0.5, Math.min(1.5, expCellWidth * 0.15));
              ctx.setLineDash([]);
              ctx.globalAlpha = 0.9;
              for (const pt of decimated) {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, circleRadius, 0, Math.PI * 2);
                ctx.fillStyle = overlay.color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = strokeWidth;
                ctx.stroke();
              }
            }
            ctx.restore();
          }
        }

        ctx.restore();

        // Legend
        if (includeLegend) {
          const legendY = TITLE_HEIGHT + heatmapAreaHeight + LEGEND_HEIGHT / 2;
          ctx.fillStyle = onSurfaceColor;
          ctx.font = '12px "Roboto", sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';

          let nextX = expW / 2;
          if (includeCellValueLegend) {
            if (colorScheme === 'binary') {
              const legendStartX = nextX - 100;
              ctx.fillStyle = '#ffffff';
              ctx.fillRect(legendStartX, legendY - 8, 16, 16);
              ctx.strokeStyle = outlineVariantColor;
              ctx.strokeRect(legendStartX, legendY - 8, 16, 16);
              ctx.fillStyle = onSurfaceColor;
              ctx.fillText('No data', legendStartX + 24, legendY);

              ctx.fillStyle = primaryColor;
              ctx.fillRect(legendStartX + 100, legendY - 8, 16, 16);
              ctx.fillStyle = onSurfaceColor;
              ctx.fillText('Has reads', legendStartX + 124, legendY);
              nextX = legendStartX + 200;
            } else {
              const gradientStartX = nextX - 80;
              const gradientWidth = 100;

              ctx.fillText('Low', gradientStartX - 30, legendY);

              const gradient = ctx.createLinearGradient(gradientStartX, 0, gradientStartX + gradientWidth, 0);
              gradient.addColorStop(0, 'rgb(0, 0, 0)');
              gradient.addColorStop(0.2, 'rgb(63, 0, 92)');
              gradient.addColorStop(0.4, 'rgb(148, 23, 81)');
              gradient.addColorStop(0.55, 'rgb(199, 52, 44)');
              gradient.addColorStop(0.7, 'rgb(237, 117, 15)');
              gradient.addColorStop(0.85, 'rgb(251, 191, 36)');
              gradient.addColorStop(1, 'rgb(252, 253, 141)');

              ctx.fillStyle = gradient;
              ctx.fillRect(gradientStartX, legendY - 8, gradientWidth, 16);
              ctx.strokeStyle = outlineVariantColor;
              ctx.strokeRect(gradientStartX, legendY - 8, gradientWidth, 16);

              ctx.fillStyle = onSurfaceColor;
              ctx.fillText('High', gradientStartX + gradientWidth + 8, legendY);
              nextX = gradientStartX + gradientWidth + 50;
            }
          }

          // Path overlay legend (when paths are visible) — side by side
          if (includePathLegend && hasVisiblePathsExport && pathOverlays) {
            nextX += 16; // gap before path items
            for (const overlay of pathOverlays) {
              if (!resolvePathVisible(overlay)) continue;
              ctx.fillStyle = overlay.color;
              ctx.fillRect(nextX, legendY - 6, 12, 3);
              ctx.fillStyle = onSurfaceColor;
              ctx.font = '11px "Roboto", sans-serif';
              ctx.fillText(overlay.label, nextX + 18, legendY);
              nextX += 18 + ctx.measureText(overlay.label).width + 16;
            }
          }
        }

        const sanitizedFileId = fileId.replace(/[^a-zA-Z0-9_-]/g, '_');
        const sanitizedChromosome = chromosome.replace(/[^a-zA-Z0-9_-]/g, '_');
        const suggestedFilename = `${sanitizedFileId}_${sanitizedChromosome}_${expStartPos}-${expEndPos}.png`;

        const blob = await new Promise<Blob | null>((resolve) => {
          exportCanvas.toBlob(resolve, 'image/png');
        });

        if (!blob) {
          console.error('Failed to create PNG blob');
          return;
        }

        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const backend = await getBackend();
        await backend.saveFile(uint8Array, {
          defaultName: suggestedFilename,
          filters: [{ name: 'PNG Image', extensions: ['png'] }],
        });
      } catch (error) {
        console.error('Failed to export PNG:', error);
        throw error;
      }
    }
  }), [
    matrix, positions, gameteNames, containerSize, cellWidth, cellHeight,
    startCol, endCol, numRows, numCols,
    showGridLines, colorScheme, getCellColor, maxValue, pathOverlays,
  ]);

  // Draw the heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size with device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = containerSize.width * dpr;
    canvas.height = containerSize.height * dpr;
    ctx.scale(dpr, dpr);

    // Get computed CSS variable colors
    const computedStyle = getComputedStyle(document.documentElement);
    const surfaceColor = computedStyle.getPropertyValue('--md-sys-color-surface').trim() || '#fef7ff';
    const onSurfaceColor = computedStyle.getPropertyValue('--md-sys-color-on-surface').trim() || '#1d1b20';
    const outlineVariantColor = computedStyle.getPropertyValue('--md-sys-color-outline-variant').trim() || '#cac4d0';
    const primaryColor = computedStyle.getPropertyValue('--md-sys-color-primary').trim() || '#6750a4';

    // Clear canvas with surface color
    ctx.fillStyle = surfaceColor;
    ctx.fillRect(0, 0, containerSize.width, containerSize.height);

    // Draw gamete labels (y-axis)
    ctx.fillStyle = onSurfaceColor;
    ctx.font = `${Math.min(11, cellHeight * 0.8)}px "Roboto", sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight + cellHeight / 2;
      if (y < containerSize.height) {
        const label = gameteNames[row] || `Gamete ${row}`;
        const truncatedLabel = label.length > 12 ? label.substring(0, 10) + '...' : label;
        ctx.fillText(truncatedLabel, LABEL_MARGIN_LEFT - 8, y);
      }
    }

    // Draw white background behind cells (ensures no dark lines show through in dark theme)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(
      LABEL_MARGIN_LEFT + PADDING,
      LABEL_MARGIN_TOP + PADDING,
      viewportWidth,
      totalMatrixHeight
    );

    // Draw cells
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
      if (y > containerSize.height) break;

      for (let col = startCol; col < endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x > containerSize.width - PADDING) break;

        const value = matrix[row]?.[col] ?? 0;
        ctx.fillStyle = getCellColor(value, maxValue, primaryColor);
        ctx.fillRect(x, y, cellWidth - (showGridLines ? 1 : 0), cellHeight - (showGridLines ? 1 : 0));
      }
    }

    // Draw grid lines
    if (showGridLines && Math.min(cellWidth, cellHeight) >= 4) {
      ctx.strokeStyle = outlineVariantColor;
      ctx.lineWidth = 0.5;

      // Vertical lines
      for (let col = startCol; col <= endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x <= containerSize.width - PADDING) {
          ctx.beginPath();
          ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
          ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + totalMatrixHeight);
          ctx.stroke();
        }
      }

      // Horizontal lines
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

    // Desaturate cells when path overlays are visible so paths stand out
    const hasVisiblePaths = pathOverlays?.some(o => o.visible);
    if (hasVisiblePaths) {
      ctx.save();
      ctx.globalCompositeOperation = 'source-atop';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
      const renderedWidth = (endCol - startCol) * cellWidth;
      ctx.fillRect(
        LABEL_MARGIN_LEFT + PADDING,
        LABEL_MARGIN_TOP + PADDING,
        renderedWidth,
        totalMatrixHeight,
      );
      ctx.restore();
    }

    // Draw path overlays
    if (pathOverlays) {
      for (const overlay of pathOverlays) {
        if (overlay.visible) {
          drawPath(ctx, overlay.data, overlay.color, overlay.dashPattern, overlay.lineWidth);
        }
      }
    }

    // Draw hover highlight
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

  }, [matrix, positions, gameteNames, containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols, 
      totalMatrixHeight, viewportWidth, showGridLines, getCellColor, maxValue, hoveredCell, pathOverlays, drawPath]);

  // Handle mouse down for drag scrolling
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setIsDragging(true);
      setDragStartX(e.clientX);
      setDragStartOffset(scrollOffset);
    }
  }, [scrollOffset]);

  // Handle mouse move for drag scrolling and hover
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    // Update dragging
    if (isDragging) {
      const deltaX = e.clientX - dragStartX;
      const deltaOffset = -deltaX / (maxScrollOffset || 1);
      const newOffset = Math.max(0, Math.min(1, dragStartOffset + deltaOffset));
      onScrollChange(newOffset);
    }

    // Update hover cell
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

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
    setHoveredCell(null);
  }, []);

  // Global mouse up listener
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // Native wheel event listener for proper preventDefault support
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
        // Zoom with Ctrl/Cmd + scroll
        e.preventDefault();
        if (onZoomChange) {
          const delta = e.deltaY > 0 ? -0.1 : 0.1;
          onZoomChange(Math.max(0.25, Math.min(4, zoomLevel + delta)));
        }
      } else if (e.shiftKey) {
        // Horizontal scroll with Shift + scroll
        // Note: Some browsers convert deltaY to deltaX when shift is held
        e.preventDefault();
        const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        const scrollDelta = delta / (maxScrollOffset || 1);
        const newOffset = Math.max(0, Math.min(1, scrollOffset + scrollDelta * 0.1));
        onScrollChange(newOffset);
      }
      // Normal scroll (no modifier) - let browser handle vertical scrolling
    };

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleNativeWheel);
  }, [scrollOffset, maxScrollOffset, zoomLevel, cellWidthMultiplier, onScrollChange, onZoomChange, onCellWidthChange]);

  // Calculate the required height for the canvas
  const requiredHeight = totalMatrixHeight + LABEL_MARGIN_TOP + PADDING * 2;

  return (
    <div 
      ref={containerRef} 
      className="heatmap-canvas-container"
      style={{ 
        width: '100%', 
        minHeight: Math.max(300, requiredHeight),
        height: requiredHeight,
        position: 'relative',
        cursor: isDragging ? 'grabbing' : 'grab',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: containerSize.width,
          height: containerSize.height,
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      
      {/* Tooltip */}
      {hoveredCell && (
        <TooltipContent
          mousePos={mousePos}
          containerSize={containerSize}
          gameteNames={gameteNames}
          positions={positions}
          hoveredCell={hoveredCell}
          pathOverlays={pathOverlays}
        />
      )}
    </div>
  );
});

HeatmapCanvas.displayName = 'HeatmapCanvas';

export default HeatmapCanvas;

