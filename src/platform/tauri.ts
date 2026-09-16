import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { open, save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';
import type {
  PlatformBackend,
  FileHandle,
  FileInput,
  OpenFileOptions,
  SaveFileOptions,
  ProgressCallback,
  PS4GProgress,
  PS4GParseResult,
  ChromosomeMatrixResult,
  ChromosomeMatrixProgress,
  ColumnMode,
  BEDProgress,
  BEDParseResult,
  BEDChromosomeMatrixResult,
  BEDMatrixProgress,
  NpyOverlayResult,
} from './types';

const tauriBackend: PlatformBackend = {
  async parsePS4GFile(
    file: FileInput,
    onProgress?: ProgressCallback<PS4GProgress>,
  ) {
    const filePath = file as string;
    let unlisten: UnlistenFn | undefined;
    if (onProgress) {
      unlisten = await listen<PS4GProgress>('ps4g-progress', (event) => {
        onProgress(event.payload);
      });
    }
    try {
      const result = await invoke<PS4GParseResult>('parse_ps4g_file', {
        filePath,
      });
      return { result, handle: filePath as FileHandle };
    } finally {
      unlisten?.();
    }
  },

  async getChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    columnMode: ColumnMode,
    onProgress?: ProgressCallback<ChromosomeMatrixProgress>,
  ) {
    const filePath = handle as string;
    let unlisten: UnlistenFn | undefined;
    if (onProgress) {
      unlisten = await listen<ChromosomeMatrixProgress>(
        'chromosome-matrix-progress',
        (event) => {
          onProgress(event.payload);
        },
      );
    }
    try {
      return await invoke<ChromosomeMatrixResult>('get_chromosome_matrix', {
        filePath,
        chromosome,
        columnMode,
      });
    } finally {
      unlisten?.();
    }
  },

  async parseBEDFile(
    file: FileInput,
    onProgress?: ProgressCallback<BEDProgress>,
  ) {
    const filePath = file as string;
    let unlisten: UnlistenFn | undefined;
    if (onProgress) {
      unlisten = await listen<BEDProgress>('bed-progress', (event) => {
        onProgress(event.payload);
      });
    }
    try {
      const result = await invoke<BEDParseResult>('parse_bed_file', {
        filePath,
      });
      return { result, handle: filePath as FileHandle };
    } finally {
      unlisten?.();
    }
  },

  async getBEDChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    onProgress?: ProgressCallback<BEDMatrixProgress>,
  ) {
    const filePath = handle as string;
    let unlisten: UnlistenFn | undefined;
    if (onProgress) {
      unlisten = await listen<BEDMatrixProgress>(
        'bed-matrix-progress',
        (event) => {
          onProgress(event.payload);
        },
      );
    }
    try {
      return await invoke<BEDChromosomeMatrixResult>(
        'get_bed_chromosome_matrix',
        { filePath, chromosome },
      );
    } finally {
      unlisten?.();
    }
  },

  async loadNpyOverlay(
    observed: FileInput | null,
    predictions: FileInput | null,
    expectedNumPositions: number,
    expectedNumGametes: number,
    sourceRows: number[] | null,
    totalRows: number,
  ) {
    return invoke<NpyOverlayResult>('load_npy_overlay', {
      observedPath: observed as string | null,
      predictionsPath: predictions as string | null,
      expectedNumPositions,
      expectedNumGametes,
      sourceRows,
      totalRows,
    });
  },

  async openFile(options?: OpenFileOptions): Promise<FileInput | null> {
    const selected = await open({
      title: options?.title,
      multiple: options?.multiple ?? false,
      filters: options?.filters,
    });
    if (selected && typeof selected === 'string') {
      return selected;
    }
    return null;
  },

  async saveFile(data: Uint8Array, options?: SaveFileOptions): Promise<void> {
    const path = await save({
      defaultPath: options?.defaultName,
      filters: options?.filters,
    });
    if (path) {
      await writeFile(path, data);
    }
  },
};

export default tauriBackend;
