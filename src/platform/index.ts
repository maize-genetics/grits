import type { PlatformBackend } from './types';

export type { PlatformBackend } from './types';
export type {
  FileInput,
  FileHandle,
  OpenFileOptions,
  SaveFileOptions,
  ProgressCallback,
  PS4GProgress,
  PS4GParseResult,
  PS4GMetadata,
  PS4GSummary,
  PS4GDataRow,
  GameteInfo,
  ChromosomeMatrixResult,
  ChromosomeMatrixProgress,
  ColumnMode,
  BEDProgress,
  BEDParseResult,
  BEDSummary,
  BEDDataRow,
  ParentStats,
  BEDChromosomeMatrixResult,
  BEDMatrixProgress,
  BEDRegion,
  NpyOverlayResult,
} from './types';

export const isTauri =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

let _backend: PlatformBackend | null = null;

export async function getBackend(): Promise<PlatformBackend> {
  if (!_backend) {
    if (isTauri) {
      _backend = (await import('./tauri')).default;
    } else {
      _backend = (await import('./web')).default;
    }
  }
  return _backend;
}
