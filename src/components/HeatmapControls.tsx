import React, { useCallback } from 'react';
import Icon from '@mdi/react';
import {
  mdiMagnifyPlus,
  mdiMagnifyMinus,
  mdiRefresh,
  mdiViewGrid,
  mdiViewGridOutline,
  mdiArrowCollapseHorizontal,
  mdiArrowExpandHorizontal
} from '@mdi/js';
import type { ColumnMode } from '../platform';

interface HeatmapControlsProps {
  /** Current zoom level (0.25 - 4) */
  zoomLevel: number;
  /** Callback when zoom changes */
  onZoomChange: (zoom: number) => void;
  /** Cell width multiplier (0.25 - 4) */
  cellWidthMultiplier: number;
  /** Callback when cell width multiplier changes */
  onCellWidthChange: (width: number) => void;
  /** Cell height multiplier (0.25 - 4) */
  cellHeightMultiplier: number;
  /** Callback when cell height multiplier changes */
  onCellHeightChange: (height: number) => void;
  /** Whether grid lines are shown */
  showGridLines: boolean;
  /** Toggle grid lines */
  onToggleGridLines: () => void;
  /** Color scheme (omit to hide the toggle) */
  colorScheme?: 'binary' | 'intensity';
  /** Toggle color scheme */
  onToggleColorScheme?: () => void;
  /** Column model (omit to hide the toggle) */
  columnMode?: ColumnMode;
  /** Toggle column model */
  onToggleColumnMode?: () => void;
  /** Reset view to defaults */
  onResetView: () => void;
}

const ZOOM_PRESETS = [0.5, 1, 2, 4];

const HeatmapControls: React.FC<HeatmapControlsProps> = ({
  zoomLevel,
  onZoomChange,
  cellWidthMultiplier,
  onCellWidthChange,
  cellHeightMultiplier,
  onCellHeightChange,
  showGridLines,
  onToggleGridLines,
  colorScheme,
  onToggleColorScheme,
  columnMode,
  onToggleColumnMode,
  onResetView,
}) => {
  // Handle zoom in/out buttons
  const handleZoomIn = useCallback(() => {
    const nextPreset = ZOOM_PRESETS.find(z => z > zoomLevel);
    onZoomChange(nextPreset || Math.min(4, zoomLevel + 0.5));
  }, [zoomLevel, onZoomChange]);

  const handleZoomOut = useCallback(() => {
    const prevPreset = [...ZOOM_PRESETS].reverse().find(z => z < zoomLevel);
    onZoomChange(prevPreset || Math.max(0.25, zoomLevel - 0.5));
  }, [zoomLevel, onZoomChange]);

  return (
    <div className="heatmap-controls">
      {/* Display Controls */}
      <div className="controls-section view-section">
        <span className="section-label">Display</span>
        <button 
          className={`control-button ${showGridLines ? 'active' : ''}`}
          onClick={onToggleGridLines}
          title={showGridLines ? 'Hide grid lines' : 'Show grid lines'}
        >
          <Icon path={showGridLines ? mdiViewGrid : mdiViewGridOutline} size={0.6} />
        </button>
        
        {colorScheme && onToggleColorScheme && (
          <button
            className={`control-button color-scheme-toggle ${colorScheme === 'intensity' ? 'active' : ''}`}
            onClick={onToggleColorScheme}
            title={colorScheme === 'binary' ? 'Switch to intensity colors' : 'Switch to binary colors'}
          >
            <span className="color-indicator">
              {colorScheme === 'binary' ? 'BIN' : 'INT'}
            </span>
          </button>
        )}

        {columnMode && onToggleColumnMode && (
          <button
            className={`control-button column-mode-toggle ${columnMode === 'row' ? 'active' : ''}`}
            onClick={onToggleColumnMode}
            title={columnMode === 'binned'
              ? 'Switch to one column per PS4G row'
              : 'Switch to one column per bin position'}
          >
            <Icon path={columnMode === 'binned' ? mdiArrowCollapseHorizontal : mdiArrowExpandHorizontal} size={0.6} />
            <span className="button-label">
              {columnMode === 'binned' ? 'Col: Bin' : 'Col: Row'}
            </span>
          </button>
        )}

        <button
          className="control-button reset-button" 
          onClick={onResetView}
          title="Reset view"
        >
          <Icon path={mdiRefresh} size={0.6} />
        </button>
      </div>

      <div className="controls-separator" />

      {/* Zoom Controls */}
      <div className="controls-section zoom-section">
        <span className="section-label">Zoom</span>
        <div className="zoom-buttons">
          <button 
            className="control-button" 
            onClick={handleZoomOut}
            disabled={zoomLevel <= 0.25}
            title="Zoom out"
          >
            <Icon path={mdiMagnifyMinus} size={0.6} />
          </button>
          
          <div className="zoom-presets">
            {ZOOM_PRESETS.map(preset => (
              <button
                key={preset}
                className={`preset-button ${Math.abs(zoomLevel - preset) < 0.1 ? 'active' : ''}`}
                onClick={() => onZoomChange(preset)}
              >
                {preset}x
              </button>
            ))}
          </div>
          
          <button 
            className="control-button" 
            onClick={handleZoomIn}
            disabled={zoomLevel >= 4}
            title="Zoom in"
          >
            <Icon path={mdiMagnifyPlus} size={0.6} />
          </button>
        </div>
        
        <div className="zoom-slider-container">
          <input
            type="range"
            min="0.25"
            max="4"
            step="0.05"
            value={zoomLevel}
            onChange={(e) => onZoomChange(parseFloat(e.target.value))}
            className="zoom-slider"
          />
          <span className="zoom-value">{(zoomLevel * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div className="controls-separator" />

      {/* Column Width Controls */}
      <div className="controls-section width-section">
        <span className="section-label">Col Width</span>
        <div className="width-slider-container">
          <input
            type="range"
            min="0.05"
            max="4"
            step="0.05"
            value={cellWidthMultiplier}
            onChange={(e) => onCellWidthChange(parseFloat(e.target.value))}
            className="width-slider"
          />
          <span className="width-value">{cellWidthMultiplier.toFixed(2)}x</span>
        </div>
      </div>

      {/* Row Height Controls */}
      <div className="controls-section height-section">
        <span className="section-label">Row Height</span>
        <div className="height-slider-container">
          <input
            type="range"
            min="0.05"
            max="4"
            step="0.05"
            value={cellHeightMultiplier}
            onChange={(e) => onCellHeightChange(parseFloat(e.target.value))}
            className="height-slider"
          />
          <span className="height-value">{cellHeightMultiplier.toFixed(2)}x</span>
        </div>
      </div>
    </div>
  );
};

export default HeatmapControls;
