import React, { useState, useCallback, useEffect } from 'react';
import Icon from '@mdi/react';
import { mdiClose, mdiLink, mdiLinkOff, mdiArrowExpandHorizontal, mdiArrowExpandVertical, mdiAlertOutline, mdiArrowRightBoldBox } from '@mdi/js';
import './ExportModal.css';

export interface PathOverlayInfo {
  label: string;
  visible: boolean;
}

export interface ExportSettings {
  title: string;
  width: number;
  height: number;
  scale: number;
  includePathLegend: boolean;
  includeCellValueLegend: boolean;
  pathVisibility: Record<string, boolean>;
  /** Custom export range, or null to use the currently visible range. */
  startPosition: number | null;
  endPosition: number | null;
}

interface ExportModalProps {
  defaultTitle: string;
  defaultWidth: number;
  defaultHeight: number;
  visibleStartPos: number;
  visibleEndPos: number;
  fullRangeStart: number;
  fullRangeEnd: number;
  pathOverlays: PathOverlayInfo[];
  onExport: (settings: ExportSettings) => void;
  onClose: () => void;
}

const SCALE_OPTIONS = [1, 2, 3, 4];

const ExportModal: React.FC<ExportModalProps> = ({
  defaultTitle,
  defaultWidth,
  defaultHeight,
  visibleStartPos,
  visibleEndPos,
  fullRangeStart,
  fullRangeEnd,
  pathOverlays,
  onExport,
  onClose,
}) => {
  const [title, setTitle] = useState(defaultTitle);
  const [width, setWidth] = useState(Math.round(defaultWidth));
  const [height, setHeight] = useState(Math.round(defaultHeight));
  const [scale, setScale] = useState(3);
  const [lockAspect, setLockAspect] = useState(true);
  const [includePathLegend, setIncludePathLegend] = useState(true);
  const [includeCellValueLegend, setIncludeCellValueLegend] = useState(true);
  const [useCustomPositionRange, setUseCustomPositionRange] = useState(false);
  const [showRangeWarningTooltip, setShowRangeWarningTooltip] = useState(false);
  const [startPos, setStartPos] = useState(visibleStartPos);
  const [endPos, setEndPos] = useState(visibleEndPos);
  const [pathVisibility, setPathVisibility] = useState<Record<string, boolean>>(() => {
    const vis: Record<string, boolean> = {};
    for (const p of pathOverlays) {
      vis[p.label] = p.visible;
    }
    return vis;
  });

  const aspectRatio = defaultWidth / defaultHeight;

  const handleWidthChange = useCallback((newWidth: number) => {
    setWidth(newWidth);
    if (lockAspect && newWidth > 0) {
      setHeight(Math.round(newWidth / aspectRatio));
    }
  }, [lockAspect, aspectRatio]);

  const handleHeightChange = useCallback((newHeight: number) => {
    setHeight(newHeight);
    if (lockAspect && newHeight > 0) {
      setWidth(Math.round(newHeight * aspectRatio));
    }
  }, [lockAspect, aspectRatio]);

  const handlePathToggle = useCallback((label: string) => {
    setPathVisibility(prev => ({ ...prev, [label]: !prev[label] }));
  }, []);

  const handleExport = useCallback(() => {
    // null (rather than the visible range's own bounds) tells the canvas to
    // use whatever is currently on screen at export time -- passing the
    // visible bounds explicitly here is indistinguishable, downstream, from
    // an intentional custom range, and can snap the export to the wrong
    // columns when positions repeat (row column mode).
    onExport({
      title,
      width,
      height,
      scale,
      includePathLegend,
      includeCellValueLegend,
      pathVisibility,
      startPosition: useCustomPositionRange ? startPos : null,
      endPosition: useCustomPositionRange ? endPos : null,
    });
  }, [title, width, height, scale, includePathLegend, includeCellValueLegend, pathVisibility, useCustomPositionRange, startPos, endPos, onExport]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const positionError = useCustomPositionRange
    ? (startPos >= endPos
        ? 'Start position must be less than end position'
        : startPos < fullRangeStart || endPos > fullRangeEnd
          ? `Positions must be within ${fullRangeStart.toLocaleString()} – ${fullRangeEnd.toLocaleString()}`
          : null)
    : null;

  return (
    <div className="export-modal-backdrop" onClick={onClose}>
      <div className="export-modal" onClick={(e) => e.stopPropagation()}>
        <div className="export-modal-header">
          <h3>Export Image</h3>
          <button className="export-modal-close" onClick={onClose} title="Close">
            <Icon path={mdiClose} size={0.9} />
          </button>
        </div>

        <div className="export-modal-body">
          {/* Title */}
          <div className="export-modal-field">
            <label className="export-modal-label">Title</label>
            <input
              type="text"
              className="export-modal-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Image title"
            />
          </div>

          {/* Dimensions */}
          <div className="export-modal-field">
            <label className="export-modal-label">Dimensions (px)</label>
            <div className="export-modal-dimensions-row">
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label" title="Width">
                  <Icon path={mdiArrowExpandHorizontal} size={0.65} />
                  <span>Width</span>
                </span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={width}
                  min={100}
                  onChange={(e) => handleWidthChange(parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label" title="Height">
                  <Icon path={mdiArrowExpandVertical} size={0.65} />
                  <span>Height</span>
                </span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={height}
                  min={100}
                  onChange={(e) => handleHeightChange(parseInt(e.target.value) || 0)}
                />
              </div>
              <button
                className={`export-modal-aspect-lock ${lockAspect ? 'locked' : ''}`}
                onClick={() => setLockAspect(prev => !prev)}
                title={lockAspect ? 'Unlock aspect ratio' : 'Lock aspect ratio'}
              >
                <span className="export-modal-aspect-lock-icon">
                  <Icon path={lockAspect ? mdiLink : mdiLinkOff} size={0.5} />
                </span>
              </button>
            </div>
          </div>

          {/* Scale */}
          <div className="export-modal-field">
            <label className="export-modal-label">Scale Factor</label>
            <div className="export-modal-scale-row">
              {SCALE_OPTIONS.map((s) => (
                <button
                  key={s}
                  className={`export-modal-scale-btn ${scale === s ? 'active' : ''}`}
                  onClick={() => setScale(s)}
                >
                  {s}x
                </button>
              ))}
            </div>
            <span className="export-modal-hint">
              Output: {(width * scale).toLocaleString()} × {(height * scale).toLocaleString()} px
            </span>
          </div>

          {/* Legend toggles */}
          <div className="export-modal-field">
            <label className="export-modal-label">
              {pathOverlays.length > 0 ? 'Legends' : 'Legend'}
            </label>
            <div className="export-modal-legend-toggles">
              <label className="export-modal-checkbox-row">
                <input
                  type="checkbox"
                  checked={includeCellValueLegend}
                  onChange={(e) => setIncludeCellValueLegend(e.target.checked)}
                />
                <span>Cell values</span>
              </label>
              {pathOverlays.length > 0 && (
                <label className="export-modal-checkbox-row">
                  <input
                    type="checkbox"
                    checked={includePathLegend}
                    onChange={(e) => setIncludePathLegend(e.target.checked)}
                  />
                  <span>Paths</span>
                </label>
              )}
            </div>
          </div>

          {/* Path overlays */}
          {pathOverlays.length > 0 && (
            <div className="export-modal-field">
              <label className="export-modal-label">Path Overlays</label>
              <div className="export-modal-path-toggles">
                {pathOverlays.map((p) => (
                  <label key={p.label} className="export-modal-checkbox-row">
                    <input
                      type="checkbox"
                      checked={pathVisibility[p.label] ?? false}
                      onChange={() => handlePathToggle(p.label)}
                    />
                    <span>{p.label}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Position Range */}
          <div className="export-modal-field export-modal-field-separated">
            <label className="export-modal-checkbox-row export-modal-position-engage">
              <input
                type="checkbox"
                checked={useCustomPositionRange}
                onChange={(e) => setUseCustomPositionRange(e.target.checked)}
              />
              <span>Use custom position range</span>
              {useCustomPositionRange && (
                <span
                  className="export-modal-range-warning-trigger"
                  onMouseEnter={() => setShowRangeWarningTooltip(true)}
                  onMouseLeave={() => setShowRangeWarningTooltip(false)}
                >
                  <Icon path={mdiAlertOutline} size={0.65} />
                  {showRangeWarningTooltip && (
                    <span className="export-modal-range-warning-tooltip">
                      Modifying the range values will <strong>not</strong> update the default title!
                    </span>
                  )}
                </span>
              )}
            </label>
            {useCustomPositionRange ? (
              <>
                <div className="export-modal-range-row">
                  <div className="export-modal-range-input">
                    <span className="export-modal-range-label">Start</span>
                    <input
                      type="number"
                      className="export-modal-input export-modal-input-number"
                      value={startPos}
                      min={fullRangeStart}
                      max={fullRangeEnd}
                      onChange={(e) => setStartPos(parseInt(e.target.value) || 0)}
                    />
                  </div>
                  <span className="export-modal-range-separator"><Icon path={mdiArrowRightBoldBox} size={0.6} /></span>
                  <div className="export-modal-range-input">
                    <span className="export-modal-range-label">End</span>
                    <input
                      type="number"
                      className="export-modal-input export-modal-input-number"
                      value={endPos}
                      min={fullRangeStart}
                      max={fullRangeEnd}
                      onChange={(e) => setEndPos(parseInt(e.target.value) || 0)}
                    />
                  </div>
                </div>
                {positionError && (
                  <span className="export-modal-error">{positionError}</span>
                )}
              </>
            ) : (
              <span className="export-modal-hint">
                Export will use the currently visible range ({visibleStartPos.toLocaleString()} – {visibleEndPos.toLocaleString()})
              </span>
            )}
          </div>
        </div>

        <div className="export-modal-footer">
          <button className="export-modal-btn export-modal-btn-cancel" onClick={onClose}>
            Cancel
          </button>
          <button
            className="export-modal-btn export-modal-btn-export"
            onClick={handleExport}
            disabled={!!positionError || width < 100 || height < 100}
          >
            Export
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExportModal;
