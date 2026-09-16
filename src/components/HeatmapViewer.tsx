import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import Icon from '@mdi/react';
import { getBackend } from '../platform';
import type { FileHandle, FileInput, ChromosomeMatrixResult, ChromosomeMatrixProgress, NpyOverlayResult, ColumnMode } from '../platform';
import { mdiChartTimeline, mdiAlertCircle, mdiChevronDown, mdiDownload, mdiPlay, mdiPause, mdiHelpCircleOutline, mdiEye, mdiEyeOff, mdiArrowExpandVertical, mdiKeyboard, mdiClose } from '@mdi/js';
import HeatmapCanvas, { VisibleRange, HeatmapCanvasHandle, PathOverlay } from './HeatmapCanvas';
import HeatmapControls from './HeatmapControls';
import PositionSearch from './PositionSearch';
import ExportModal, { ExportSettings, PathOverlayInfo } from './ExportModal';
import { findNearestColumnIndex, calculateScrollOffset } from '../utils/positionSearch';
import './HeatmapViewer.css';

/**
 * Natural sort comparison function for strings.
 * Handles numeric substrings so that "2" < "10" (unlike lexicographic sort).
 * For purely numeric IDs (e.g., "21", "0"), performs numeric comparison.
 */
function naturalSortCompare(a: string, b: string): number {
  // Split strings into chunks of digits and non-digits
  const splitRegex = /(\d+)/g;
  const aParts = a.split(splitRegex).filter(Boolean);
  const bParts = b.split(splitRegex).filter(Boolean);
  
  const maxLen = Math.max(aParts.length, bParts.length);
  
  for (let i = 0; i < maxLen; i++) {
    const aPart = aParts[i] || '';
    const bPart = bParts[i] || '';
    
    const aIsNum = /^\d+$/.test(aPart);
    const bIsNum = /^\d+$/.test(bPart);
    
    if (aIsNum && bIsNum) {
      // Compare as numbers
      const diff = parseInt(aPart, 10) - parseInt(bPart, 10);
      if (diff !== 0) return diff;
    } else if (aIsNum) {
      // Numbers come before letters
      return -1;
    } else if (bIsNum) {
      return 1;
    } else {
      // Compare as strings (case-insensitive)
      const cmp = aPart.toLowerCase().localeCompare(bPart.toLowerCase());
      if (cmp !== 0) return cmp;
    }
  }
  
  return 0;
}

import type { PS4GMetadata, PS4GSummary } from '../platform';

// Colorblind-safe palette (Wong 2011) chosen to contrast with plasma heatmap
const PATH_COLORS = {
  true1: '#0072B2',   // blue
  true2: '#56B4E9',   // sky blue
  pred1: '#009E73',   // bluish green
  pred2: '#CC79A7',   // reddish purple
} as const;

const PATH_DASHES = {
  true1: [1, 0],       // solid (observed haplotype 1)
  true2: [1, 0],       // solid (observed haplotype 2)
  pred1: [10, 4],      // dashed (predicted haplotype 1)
  pred2: [4, 3, 1, 3], // dot-dash (predicted haplotype 2)
} as const;

const PATH_WIDTHS = {
  true1: 1.4,
  true2: 1.4,
  pred1: 0.7,
  pred2: 0.7,
} as const;

interface HeatmapViewerProps {
  filePath: string;
  fileHandle: FileHandle | null;
  metadata: PS4GMetadata;
  summary: PS4GSummary;
  overlayModalOpen?: boolean;
  onOverlayModalClose?: () => void;
}

const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  filePath,
  fileHandle,
  metadata: _metadata,
  summary,
  overlayModalOpen = false,
  onOverlayModalClose,
}) => {
  // State
  const [selectedChromosome, setSelectedChromosome] = useState<string>(summary.chromosomes[0] || '');
  const [matrixData, setMatrixData] = useState<ChromosomeMatrixResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadProgress, setLoadProgress] = useState<ChromosomeMatrixProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // View state
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [scrollOffset, setScrollOffset] = useState<number>(0);
  const [cellWidthMultiplier, setCellWidthMultiplier] = useState<number>(1);
  const [cellHeightMultiplier, setCellHeightMultiplier] = useState<number>(1);
  const [showGridLines, setShowGridLines] = useState<boolean>(true);
  const [colorScheme, setColorScheme] = useState<'binary' | 'intensity'>('intensity');
  const [showShortcuts, setShowShortcuts] = useState<boolean>(false);

  // Column model: 'binned' collapses same-bin PS4G rows into one column
  // (historical/default behavior); 'row' gives one column per PS4G data
  // row. Tracks the chromosome a load actually landed on, so a toggle that
  // reloads the SAME chromosome (mode change) can be told apart from an
  // actual chromosome change -- only the latter should reset zoom/scroll.
  const [columnMode, setColumnMode] = useState<ColumnMode>('binned');
  const loadedChromosomeRef = useRef<string | null>(null);
  // Genomic position to re-center on after a mode toggle lands, in place of
  // carrying over scrollOffset directly -- scrollOffset is a fraction of a
  // total width that shifts with the column count, so reusing it verbatim
  // would drift by however much the toggle changed that count.
  const pendingAnchorPosRef = useRef<number | null>(null);
  
  // Auto-scroll state
  const [isAutoScrolling, setIsAutoScrolling] = useState<boolean>(false);
  const [autoScrollSpeed, setAutoScrollSpeed] = useState<number>(0.5);
  const autoScrollRef = useRef<number | null>(null);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HeatmapCanvasHandle>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);
  const [visibleRange, setVisibleRange] = useState<VisibleRange | null>(null);
  
  // Top-level controls visibility (hide to maximize heatmap viewspace)
  const [topControlsVisible, setTopControlsVisible] = useState<boolean>(true);

  // Export modal
  const [showExportModal, setShowExportModal] = useState<boolean>(false);

  // NumPy overlay state
  const [observedNpyPath, setObservedNpyPath] = useState<string>('');
  const [predictionsNpyPath, setPredictionsNpyPath] = useState<string>('');
  const observedNpyFileRef = useRef<FileInput | null>(null);
  const predictionsNpyFileRef = useRef<FileInput | null>(null);
  const [overlayData, setOverlayData] = useState<NpyOverlayResult | null>(null);
  const [overlayLoading, setOverlayLoading] = useState<boolean>(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [showTruePath1, setShowTruePath1] = useState<boolean>(true);
  const [showTruePath2, setShowTruePath2] = useState<boolean>(true);
  const [showPredPath1, setShowPredPath1] = useState<boolean>(true);
  const [showPredPath2, setShowPredPath2] = useState<boolean>(true);

  // Monitor container width for viewport calculation
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.getBoundingClientRect().width);
      }
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, []);

  // Load chromosome data when selection or column mode changes
  const loadChromosomeData = useCallback(async (chromosome: string, mode: ColumnMode) => {
    if (!chromosome || !fileHandle) return;

    setIsLoading(true);
    setError(null);
    setLoadProgress(null);

    try {
      const backend = await getBackend();
      const result = await backend.getChromosomeMatrix(fileHandle, chromosome, mode, (p) => {
        setLoadProgress(p);
      });

      if (result.success) {
        const isChromosomeChange = loadedChromosomeRef.current !== chromosome;
        loadedChromosomeRef.current = chromosome;
        setMatrixData(result);
        if (isChromosomeChange) {
          // A real chromosome change: reset the view. A mode-only reload
          // preserves zoom/scroll (see the anchor-restore effect below for
          // scroll, which needs the new column count first).
          setZoomLevel(1);
          setScrollOffset(0);
        }
        // The column count differs across modes (and always across
        // chromosomes), so any loaded overlay no longer lines up either way.
        setOverlayData(null);
        setOverlayError(null);
      } else {
        setError(result.error || 'Failed to load chromosome data');
        setMatrixData(null);
      }
    } catch (err) {
      console.error('Error loading chromosome matrix:', err);
      setError(`Error loading data: ${err}`);
      setMatrixData(null);
    } finally {
      setIsLoading(false);
      setLoadProgress(null);
    }
  }, [fileHandle]);

  // Load data when chromosome selection or column mode changes
  useEffect(() => {
    if (selectedChromosome) {
      loadChromosomeData(selectedChromosome, columnMode);
    }
  }, [selectedChromosome, columnMode, loadChromosomeData]);

  // Restore an anchor position across a mode toggle (see
  // pendingAnchorPosRef) once the new matrix -- and its column count --
  // has landed.
  useEffect(() => {
    if (!matrixData || pendingAnchorPosRef.current == null) return;
    const anchor = pendingAnchorPosRef.current;
    pendingAnchorPosRef.current = null;
    if (matrixData.positions.length === 0) return;
    const idx = findNearestColumnIndex(anchor, (i) => matrixData.positions[i], matrixData.positions.length);
    if (idx < 0) return;
    const offset = calculateScrollOffset(idx, matrixData.num_positions, zoomLevel, cellWidthMultiplier, containerWidth);
    if (offset !== null) setScrollOffset(offset);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  // Toggle the column model, remembering the currently-visible start
  // position so the anchor-restore effect above can re-center on it.
  const handleToggleColumnMode = useCallback(() => {
    pendingAnchorPosRef.current = visibleRange?.startPos ?? null;
    setColumnMode(prev => (prev === 'binned' ? 'row' : 'binned'));
  }, [visibleRange]);

  // Auto-scroll effect
  useEffect(() => {
    if (isAutoScrolling) {
      const scrollStep = 0.0005 * autoScrollSpeed; // Base step adjusted by speed
      
      autoScrollRef.current = window.setInterval(() => {
        setScrollOffset(prev => {
          const next = prev + scrollStep;
          // Stop at the end and pause
          if (next >= 1) {
            setIsAutoScrolling(false);
            return 1;
          }
          return next;
        });
      }, 50); // Update every 50ms for smooth scrolling
    } else {
      if (autoScrollRef.current) {
        clearInterval(autoScrollRef.current);
        autoScrollRef.current = null;
      }
    }
    
    return () => {
      if (autoScrollRef.current) {
        clearInterval(autoScrollRef.current);
        autoScrollRef.current = null;
      }
    };
  }, [isAutoScrolling, autoScrollSpeed]);

  // Spacebar keyboard shortcut for play/pause auto-scroll
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && 
          !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        if (!containerRef.current || containerRef.current.offsetParent === null) return;
        e.preventDefault();
        setIsAutoScrolling(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Handle chromosome selection
  const handleChromosomeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedChromosome(e.target.value);
  }, []);

  // Reset view to defaults. Deliberately leaves columnMode untouched: it
  // selects a data model (and triggers a reload), not a view setting, so a
  // "reset view" click shouldn't stall on a refetch.
  const handleResetView = useCallback(() => {
    setZoomLevel(1);
    setScrollOffset(0);
    setCellWidthMultiplier(1);
    setCellHeightMultiplier(1);
    setShowGridLines(true);
    setColorScheme('intensity');
    setIsAutoScrolling(false);
    setAutoScrollSpeed(0.5);
  }, []);

  const handleAutoFitHeight = useCallback(() => {
    if (!wrapperRef.current || !matrixData) return;
    const availableHeight = wrapperRef.current.clientHeight;
    const LABEL_MARGIN_TOP = 10;
    const PADDING = 10;
    const overhead = LABEL_MARGIN_TOP + PADDING * 2;
    const targetMatrixHeight = availableHeight - overhead;
    if (targetMatrixHeight <= 0) return;
    const numRows = matrixData.matrix.length;
    const baseCellSize = 12;
    const targetCellHeight = Math.floor(targetMatrixHeight / numRows);
    const newMultiplier = targetCellHeight / (baseCellSize * zoomLevel);
    setCellHeightMultiplier(Math.max(0.1, newMultiplier));
  }, [matrixData, zoomLevel]);

  // Calculate viewport width percentage
  const calculateViewportWidthPercent = useCallback((): number => {
    if (!matrixData) return 1;
    const baseCellSize = 12;
    const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
    const totalWidth = matrixData.num_positions * cellWidth;
    const labelMargin = 100;
    const padding = 20;
    const viewportWidth = Math.max(containerWidth - labelMargin - padding, 100);
    return Math.min(1, viewportWidth / totalWidth);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  // Handle visible range updates from the canvas
  const handleVisibleRangeChange = useCallback((range: VisibleRange) => {
    setVisibleRange(range);
  }, []);

  const handlePositionSearch = useCallback((targetPos: number) => {
    if (!matrixData || matrixData.positions.length === 0) return;
    const idx = findNearestColumnIndex(targetPos, (i) => matrixData.positions[i], matrixData.positions.length);
    if (idx < 0) return;
    const offset = calculateScrollOffset(idx, matrixData.num_positions, zoomLevel, cellWidthMultiplier, containerWidth);
    if (offset !== null) setScrollOffset(offset);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  // Extract filename from file path
  const getFileName = useCallback((path: string): string => {
    return path.split(/[/\\]/).pop() || path;
  }, []);

  // Handle PNG export via modal settings
  const handleExportWithSettings = useCallback(async (settings: ExportSettings) => {
    setShowExportModal(false);
    if (canvasRef.current && selectedChromosome) {
      try {
        await canvasRef.current.exportToPng({
          fileId: getFileName(filePath),
          chromosome: selectedChromosome,
          title: settings.title,
          width: settings.width,
          height: settings.height,
          scale: settings.scale,
          includeCellValueLegend: settings.includeCellValueLegend,
          includePathLegend: settings.includePathLegend,
          pathVisibility: settings.pathVisibility,
          startPosition: settings.startPosition,
          endPosition: settings.endPosition,
        });
      } catch (error) {
        console.error('Export failed:', error);
      }
    }
  }, [filePath, selectedChromosome, getFileName]);

  // Sort gamete names alphabetically (with natural sort for numeric IDs) and reorder matrix rows
  const sortedData = useMemo(() => {
    if (!matrixData) return null;
    
    const { gamete_names, matrix } = matrixData;
    
    // Create array of indices and sort by gamete name using natural sort
    const sortedIndices = gamete_names
      .map((name, index) => ({ name, index }))
      .sort((a, b) => naturalSortCompare(a.name, b.name))
      .map(item => item.index);
    
    // Reorder gamete names and matrix rows based on sorted indices
    const sortedGameteNames = sortedIndices.map(i => gamete_names[i]);
    const sortedMatrix = sortedIndices.map(i => matrix[i]);
    
    return {
      gameteNames: sortedGameteNames,
      matrix: sortedMatrix,
    };
  }, [matrixData]);

  // Build inverse sort map: gameteIndex -> sorted row position.
  // The Rust backend returns gametes sorted by gamete_index, so gamete_index IS the
  // original row. sortedIndices[sortedRow] = originalRow. We need the inverse.
  const gameteIndexToSortedRow = useMemo(() => {
    if (!matrixData) return null;
    const { gamete_names } = matrixData;

    const sortedIndices = gamete_names
      .map((name, index) => ({ name, index }))
      .sort((a, b) => naturalSortCompare(a.name, b.name))
      .map(item => item.index);

    const inverse = new Array<number>(gamete_names.length);
    for (let sortedRow = 0; sortedRow < sortedIndices.length; sortedRow++) {
      inverse[sortedIndices[sortedRow]] = sortedRow;
    }
    return inverse;
  }, [matrixData]);

  // Remap a raw gamete-index path to sorted-row path
  const remapPath = useCallback((rawPath: number[]): number[] => {
    if (!gameteIndexToSortedRow) return rawPath;
    return rawPath.map(idx => gameteIndexToSortedRow[idx] ?? idx);
  }, [gameteIndexToSortedRow]);

  // Build path overlays for HeatmapCanvas
  const pathOverlays = useMemo((): PathOverlay[] | undefined => {
    if (!overlayData) return undefined;
    const overlays: PathOverlay[] = [];

    if (overlayData.true_paths.length >= 1) {
      overlays.push({
        data: remapPath(overlayData.true_paths[0]),
        color: PATH_COLORS.true1,
        dashPattern: [...PATH_DASHES.true1],
        lineWidth: PATH_WIDTHS.true1,
        label: 'True 1',
        visible: showTruePath1,
      });
    }
    if (overlayData.true_paths.length >= 2) {
      overlays.push({
        data: remapPath(overlayData.true_paths[1]),
        color: PATH_COLORS.true2,
        dashPattern: [...PATH_DASHES.true2],
        lineWidth: PATH_WIDTHS.true2,
        label: 'True 2',
        visible: showTruePath2,
      });
    }
    if (overlayData.predicted_paths.length >= 1) {
      overlays.push({
        data: remapPath(overlayData.predicted_paths[0]),
        color: PATH_COLORS.pred1,
        dashPattern: [...PATH_DASHES.pred1],
        lineWidth: PATH_WIDTHS.pred1,
        label: 'Pred 1',
        visible: showPredPath1,
      });
    }
    if (overlayData.predicted_paths.length >= 2) {
      overlays.push({
        data: remapPath(overlayData.predicted_paths[1]),
        color: PATH_COLORS.pred2,
        dashPattern: [...PATH_DASHES.pred2],
        lineWidth: PATH_WIDTHS.pred2,
        label: 'Pred 2',
        visible: showPredPath2,
      });
    }

    return overlays.length > 0 ? overlays : undefined;
  }, [overlayData, remapPath, showTruePath1, showTruePath2, showPredPath1, showPredPath2]);

  const selectNpyFile = useCallback(async (
    setter: (path: string) => void,
    fileRef: React.MutableRefObject<FileInput | null>,
    title: string,
  ) => {
    try {
      const backend = await getBackend();
      const selected = await backend.openFile({
        title,
        filters: [
          { name: 'NumPy Files (*.npy)', extensions: ['npy'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (selected) {
        const displayName = typeof selected === 'string'
          ? selected
          : (selected as File).name;
        setter(displayName);
        fileRef.current = selected;
      }
    } catch (err) {
      console.error('File selection error:', err);
    }
  }, []);

  // Load overlay data from the selected .npy files
  const loadOverlay = useCallback(async () => {
    if (!matrixData) return;
    if (!observedNpyFileRef.current && !predictionsNpyFileRef.current) {
      setOverlayError('Select at least one .npy file');
      return;
    }

    setOverlayLoading(true);
    setOverlayError(null);

    try {
      const backend = await getBackend();
      const result = await backend.loadNpyOverlay(
        observedNpyFileRef.current,
        predictionsNpyFileRef.current,
        matrixData.num_positions,
        matrixData.num_gametes,
        // Lets a genome-wide .npy (one row per PS4G data row, across every
        // chromosome) align to this chromosome's columns, in addition to a
        // chromosome-sized .npy matching num_positions directly.
        matrixData.source_rows,
        summary.total_rows,
      );

      if (result.success) {
        setOverlayData(result);
      } else {
        setOverlayError(result.error || 'Failed to load overlay data');
        setOverlayData(null);
      }
    } catch (err) {
      console.error('Error loading overlay:', err);
      setOverlayError(`Error: ${err}`);
      setOverlayData(null);
    } finally {
      setOverlayLoading(false);
    }
  }, [matrixData, summary.total_rows]);

  const clearOverlay = useCallback(() => {
    setOverlayData(null);
    setOverlayError(null);
    setObservedNpyPath('');
    setPredictionsNpyPath('');
    observedNpyFileRef.current = null;
    predictionsNpyFileRef.current = null;
  }, []);

  // Format count
  const formatNumber = (num: number): string => num.toLocaleString();

  const showTopControls = topControlsVisible || !matrixData || isLoading || !!error;

  return (
    <div className="heatmap-viewer" ref={containerRef}>
      {/* Header with chromosome selector - hidden when controls collapsed to maximize heatmap */}
      {showTopControls && (
        <div className="heatmap-header">
          <div className="chromosome-selector">
            <label htmlFor="chromosome-select">Chromosome:</label>
            <div className="select-wrapper">
              <select
                id="chromosome-select"
                value={selectedChromosome}
                onChange={handleChromosomeChange}
                disabled={isLoading}
              >
                {summary.chromosomes.map(chr => (
                  <option key={chr} value={chr}>
                    {chr} ({formatNumber(summary.chromosome_counts[chr] || 0)} observations)
                  </option>
                ))}
              </select>
              <Icon path={mdiChevronDown} size={0.8} className="select-icon" />
            </div>
          </div>

          {matrixData && !isLoading && (
            <PositionSearch
              onSearch={handlePositionSearch}
              inputId="ps4g-position-search-input"
            />
          )}

          {matrixData && !isLoading && (
            <div className="matrix-info">
              <span className="info-item">
                <strong>{matrixData.num_gametes}</strong> gametes
              </span>
              <span className="info-divider">×</span>
              <span className="info-item">
                <strong>{formatNumber(matrixData.num_positions)}</strong>{' '}
                {matrixData.column_mode === 'row' ? 'columns (1 per PS4G row)' : 'columns (1 per bin)'}
              </span>
              <span className="info-item position-range">
                ({formatNumber(matrixData.position_range[0])} - {formatNumber(matrixData.position_range[1])} rpb)
              </span>
            </div>
          )}
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="heatmap-loading">
          <div className="spinner"></div>
          <span>Loading {selectedChromosome}...</span>
          {loadProgress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${Math.min(loadProgress.percent, 100)}%` }}
                ></div>
              </div>
              <div className="progress-stats">
                <span className="progress-percent">{loadProgress.percent.toFixed(1)}%</span>
                <span className="progress-rows">{formatNumber(loadProgress.rows_processed)} rows processed</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error state */}
      {error && !isLoading && (
        <div className="heatmap-error">
          <Icon path={mdiAlertCircle} size={1.5} />
          <span>{error}</span>
          <button onClick={() => loadChromosomeData(selectedChromosome, columnMode)}>Retry</button>
        </div>
      )}

      {/* Heatmap content */}
      {matrixData && !isLoading && !error && (
        <>
          {topControlsVisible && (<>
            <div className="ps4g-heatmap-controls-row">
              <HeatmapControls
                zoomLevel={zoomLevel}
                onZoomChange={setZoomLevel}
                cellWidthMultiplier={cellWidthMultiplier}
                onCellWidthChange={setCellWidthMultiplier}
                cellHeightMultiplier={cellHeightMultiplier}
                onCellHeightChange={setCellHeightMultiplier}
                showGridLines={showGridLines}
                onToggleGridLines={() => setShowGridLines(prev => !prev)}
                colorScheme={colorScheme}
                onToggleColorScheme={() => setColorScheme(prev => prev === 'binary' ? 'intensity' : 'binary')}
                columnMode={columnMode}
                onToggleColumnMode={handleToggleColumnMode}
                onResetView={handleResetView}
              />
              <div className="ps4g-auto-fit">
                <button
                  className="control-button auto-fit-button"
                  onClick={handleAutoFitHeight}
                  title="Fit heatmap height to view"
                >
                  <Icon path={mdiArrowExpandVertical} size={0.5} />
                  <span className="button-label">Fit Height</span>
                </button>
              </div>
            </div>

            {/* Path Overlay Modal */}
            {overlayModalOpen && (
              <div className="overlay-modal-backdrop" onClick={onOverlayModalClose}>
                <div className="overlay-modal" onClick={(e) => e.stopPropagation()}>
                  <div className="overlay-modal-header">
                    <h3>Path Overlay</h3>
                    <button className="overlay-modal-close" onClick={onOverlayModalClose} title="Close">
                      <Icon path={mdiClose} size={0.9} />
                    </button>
                  </div>
                  <div className="overlay-modal-body">
                    <div className="overlay-modal-field">
                      <label className="overlay-modal-label">Observed</label>
                      <div className="overlay-modal-input-row">
                        <input
                          type="text"
                          className="overlay-modal-path"
                          value={observedNpyPath ? observedNpyPath.split(/[/\\]/).pop() || '' : ''}
                          readOnly
                          placeholder="observed.npy"
                          title={observedNpyPath || 'No file selected'}
                        />
                        <button
                          className="control-button overlay-browse-button"
                          onClick={() => selectNpyFile(setObservedNpyPath, observedNpyFileRef, 'Select Observed .npy File')}
                          disabled={overlayLoading}
                        >
                          Browse
                        </button>
                      </div>
                    </div>
                    <div className="overlay-modal-field">
                      <label className="overlay-modal-label">Predictions</label>
                      <div className="overlay-modal-input-row">
                        <input
                          type="text"
                          className="overlay-modal-path"
                          value={predictionsNpyPath ? predictionsNpyPath.split(/[/\\]/).pop() || '' : ''}
                          readOnly
                          placeholder="predictions.npy"
                          title={predictionsNpyPath || 'No file selected'}
                        />
                        <button
                          className="control-button overlay-browse-button"
                          onClick={() => selectNpyFile(setPredictionsNpyPath, predictionsNpyFileRef, 'Select Predictions .npy File')}
                          disabled={overlayLoading}
                        >
                          Browse
                        </button>
                      </div>
                    </div>
                    {overlayError && (
                      <div className="overlay-modal-error">
                        <Icon path={mdiAlertCircle} size={0.6} />
                        <span>{overlayError}</span>
                      </div>
                    )}
                  </div>
                  <div className="overlay-modal-footer">
                    {overlayData && (
                      <button
                        className="control-button overlay-clear-button"
                        onClick={clearOverlay}
                        title="Clear overlay paths"
                      >
                        Clear
                      </button>
                    )}
                    <button
                      className="control-button overlay-load-button"
                      onClick={async () => { await loadOverlay(); onOverlayModalClose?.(); }}
                      disabled={overlayLoading || (!observedNpyPath && !predictionsNpyPath)}
                      title="Load overlay paths from selected .npy files"
                    >
                      {overlayLoading ? 'Loading...' : 'Load'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>)}

          <div className="heatmap-canvas-wrapper" ref={wrapperRef}>
            <HeatmapCanvas
              ref={canvasRef}
              matrix={sortedData?.matrix ?? matrixData.matrix}
              positions={matrixData.positions}
              gameteNames={sortedData?.gameteNames ?? matrixData.gamete_names}
              zoomLevel={zoomLevel}
              scrollOffset={scrollOffset}
              onScrollChange={setScrollOffset}
              onZoomChange={setZoomLevel}
              onVisibleRangeChange={handleVisibleRangeChange}
              cellWidthMultiplier={cellWidthMultiplier}
              onCellWidthChange={setCellWidthMultiplier}
              cellHeightMultiplier={cellHeightMultiplier}
              showGridLines={showGridLines}
              colorScheme={colorScheme}
              pathOverlays={pathOverlays}
            />
          </div>

          <div className="heatmap-bottom-bar">
            {/* Top row: Auto-scroll (left), Viewing (center), Export (right) */}
            <div className="bottom-bar-top-row">
              <div className="bottom-bar-section">
                <span className="section-label">Auto-scroll</span>
                <button 
                  className={`section-button ${isAutoScrolling ? 'active' : ''}`}
                  onClick={() => setIsAutoScrolling(prev => !prev)}
                  title={isAutoScrolling ? 'Pause auto-scroll' : 'Start auto-scroll'}
                >
                  <Icon path={isAutoScrolling ? mdiPause : mdiPlay} size={0.8} />
                </button>
                <div className="speed-slider-container">
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={autoScrollSpeed}
                    onChange={(e) => setAutoScrollSpeed(parseFloat(e.target.value))}
                    className="speed-slider"
                    title="Scroll speed"
                  />
                  <span className="speed-value">{autoScrollSpeed.toFixed(1)}x</span>
                </div>
              </div>
              
              {visibleRange && (
                <div className="visible-range-indicator">
                  <span className="visible-range-label">Viewing:</span>
                  <span className="visible-range-value">
                    {formatNumber(visibleRange.startPos)} - {formatNumber(visibleRange.endPos)} rpb
                  </span>
                </div>
              )}
              
              <div className="bottom-bar-right-group">
                <div className="bottom-bar-section">
                  <span className="section-label">Export</span>
                  <button
                    className="section-button with-label"
                    onClick={() => setShowExportModal(true)}
                    title="Export heatmap as PNG"
                  >
                    <Icon path={mdiDownload} size={0.7} />
                    <span>PNG</span>
                  </button>
                </div>

                <button
                  type="button"
                  className="heatmap-controls-toggle bottom-bar-toggle"
                  onClick={() => setTopControlsVisible(prev => !prev)}
                  title={topControlsVisible ? 'Hide header and tool bar to maximize heatmap view' : 'Show header and tool bar'}
                >
                  <Icon path={topControlsVisible ? mdiEyeOff : mdiEye} size={0.85} />
                </button>
              </div>
            </div>
            
            {/* Position scale above the slider */}
            <div className="position-scale">
              <span className="position-label">
                {formatNumber(matrixData.position_range[0])} rpb
                <span className="rpb-help-icon">
                  <Icon path={mdiHelpCircleOutline} size={0.5} />
                  <span className="rpb-tooltip">"rpb" represents the number found in the <code>refPosBinned</code> column of the PS4G file.</span>
                </span>
              </span>
              <span className="position-label">{formatNumber(Math.round((matrixData.position_range[0] + matrixData.position_range[1]) / 2))} rpb</span>
              <span className="position-label">{formatNumber(matrixData.position_range[1])} rpb</span>
            </div>
            
            {/* Tickmarks between position labels and slider */}
            <div className="slider-tickmarks">
              {[0, 25, 50, 75, 100].map((percent) => (
                <div 
                  key={percent} 
                  className={`slider-tick ${percent === 0 || percent === 50 || percent === 100 ? 'major' : 'minor'}`}
                  style={{ left: `${percent}%` }} 
                />
              ))}
            </div>
            
            <div className="bottom-position-slider">
              <div 
                className="bottom-slider-track"
                onMouseDown={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const viewportPercent = calculateViewportWidthPercent();
                  const thumbWidth = Math.max(viewportPercent * rect.width, 20);
                  const maxThumbPos = rect.width - thumbWidth;
                  const clickX = e.clientX - rect.left;
                  const newOffset = Math.max(0, Math.min(1, (clickX - thumbWidth / 2) / maxThumbPos));
                  setScrollOffset(newOffset);
                  
                  const handleDrag = (moveEvent: MouseEvent) => {
                    const moveX = moveEvent.clientX - rect.left;
                    const dragOffset = Math.max(0, Math.min(1, (moveX - thumbWidth / 2) / maxThumbPos));
                    setScrollOffset(dragOffset);
                  };
                  
                  const handleUp = () => {
                    window.removeEventListener('mousemove', handleDrag);
                    window.removeEventListener('mouseup', handleUp);
                  };
                  
                  window.addEventListener('mousemove', handleDrag);
                  window.addEventListener('mouseup', handleUp);
                }}
              >
                <div 
                  className="bottom-slider-thumb"
                  style={{
                    left: `${scrollOffset * (100 - Math.max(calculateViewportWidthPercent() * 100, 2))}%`,
                    width: `${Math.max(calculateViewportWidthPercent() * 100, 2)}%`,
                  }}
                />
              </div>
            </div>
            
            <div className="bottom-bar-info">
              <div className="legend-item">
                <span className="legend-color legend-no-data"></span>
                <span className="legend-label">No data</span>
              </div>
              {colorScheme === 'binary' ? (
                <div className="legend-item">
                  <span className="legend-color legend-has-data"></span>
                  <span className="legend-label">Has reads</span>
                </div>
              ) : (
                <div className="legend-item legend-gradient-item">
                  <span className="legend-label">Low</span>
                  <span className="legend-gradient"></span>
                  <span className="legend-label">High</span>
                </div>
              )}
              {overlayData && (
                <div className="ps4g-path-toggle-inline">
                  <span className="path-toggle-group-label">Paths</span>
                  <div className="path-toggle-group">
                    {overlayData.true_paths.length >= 1 && (
                      <button
                        className={`control-button path-toggle-button ${showTruePath1 ? 'active' : ''}`}
                        style={{ '--path-color': PATH_COLORS.true1 } as React.CSSProperties}
                        onClick={() => setShowTruePath1(prev => !prev)}
                        title={showTruePath1 ? 'Hide True 1 path' : 'Show True 1 path'}
                      >
                        <Icon path={showTruePath1 ? mdiEye : mdiEyeOff} size={0.5} />
                        <span className="button-label">T1</span>
                      </button>
                    )}
                    {overlayData.true_paths.length >= 2 && (
                      <button
                        className={`control-button path-toggle-button ${showTruePath2 ? 'active' : ''}`}
                        style={{ '--path-color': PATH_COLORS.true2 } as React.CSSProperties}
                        onClick={() => setShowTruePath2(prev => !prev)}
                        title={showTruePath2 ? 'Hide True 2 path' : 'Show True 2 path'}
                      >
                        <Icon path={showTruePath2 ? mdiEye : mdiEyeOff} size={0.5} />
                        <span className="button-label">T2</span>
                      </button>
                    )}
                    {overlayData.predicted_paths.length >= 1 && (
                      <button
                        className={`control-button path-toggle-button ${showPredPath1 ? 'active' : ''}`}
                        style={{ '--path-color': PATH_COLORS.pred1 } as React.CSSProperties}
                        onClick={() => setShowPredPath1(prev => !prev)}
                        title={showPredPath1 ? 'Hide Predicted 1 path' : 'Show Predicted 1 path'}
                      >
                        <Icon path={showPredPath1 ? mdiEye : mdiEyeOff} size={0.5} />
                        <span className="button-label">P1</span>
                      </button>
                    )}
                    {overlayData.predicted_paths.length >= 2 && (
                      <button
                        className={`control-button path-toggle-button ${showPredPath2 ? 'active' : ''}`}
                        style={{ '--path-color': PATH_COLORS.pred2 } as React.CSSProperties}
                        onClick={() => setShowPredPath2(prev => !prev)}
                        title={showPredPath2 ? 'Hide Predicted 2 path' : 'Show Predicted 2 path'}
                      >
                        <Icon path={showPredPath2 ? mdiEye : mdiEyeOff} size={0.5} />
                        <span className="button-label">P2</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
              <div className="shortcuts-trigger-wrapper">
                <button
                  className="control-button shortcuts-trigger"
                  onClick={() => setShowShortcuts(prev => !prev)}
                  title="Keyboard & mouse shortcuts"
                >
                  <Icon path={mdiKeyboard} size={0.65} />
                </button>
                {showShortcuts && (
                  <div className="shortcuts-popup">
                    <div className="shortcuts-popup-header">
                      <span>Shortcuts</span>
                      <button className="shortcuts-popup-close" onClick={() => setShowShortcuts(false)}>&times;</button>
                    </div>
                    <div className="shortcuts-popup-body">
                      <div className="shortcut-row"><kbd>Scroll</kbd><span>Vertical pan</span></div>
                      <div className="shortcut-row"><kbd>Shift + Scroll</kbd><span>Horizontal pan</span></div>
                      <div className="shortcut-row"><kbd>Ctrl + Scroll</kbd><span>Zoom</span></div>
                      <div className="shortcut-row"><kbd>Ctrl + Shift + Scroll</kbd><span>Column width</span></div>
                      <div className="shortcut-row"><kbd>Space</kbd><span>Play / Pause auto-scroll</span></div>
                      <div className="shortcut-row"><kbd>Click + Drag</kbd><span>Pan horizontally</span></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!matrixData && !isLoading && !error && (
        <div className="heatmap-empty">
          <Icon path={mdiChartTimeline} size={3} />
          <h3>Select a Chromosome</h3>
          <p>Choose a chromosome from the dropdown to visualize its heatmap</p>
        </div>
      )}

      {showExportModal && matrixData && selectedChromosome && (() => {
        const fileId = getFileName(filePath);
        const visStart = visibleRange?.startPos ?? matrixData.position_range[0];
        const visEnd = visibleRange?.endPos ?? matrixData.position_range[1];
        const defaultTitle = `${fileId} | ${selectedChromosome} | ${visStart.toLocaleString()} - ${visEnd.toLocaleString()} rbp`;

        const overlayInfos: PathOverlayInfo[] = (pathOverlays ?? []).map(p => ({
          label: p.label,
          visible: p.visible,
        }));

        return (
          <ExportModal
            defaultTitle={defaultTitle}
            defaultWidth={containerWidth}
            defaultHeight={400}
            visibleStartPos={visStart}
            visibleEndPos={visEnd}
            fullRangeStart={matrixData.position_range[0]}
            fullRangeEnd={matrixData.position_range[1]}
            pathOverlays={overlayInfos}
            onExport={handleExportWithSettings}
            onClose={() => setShowExportModal(false)}
          />
        );
      })()}
    </div>
  );
};

export default HeatmapViewer;

