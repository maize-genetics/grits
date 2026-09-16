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

// ---------------------------------------------------------------------------
// Worker-based WASM bridge
// ---------------------------------------------------------------------------

let worker: Worker | null = null;
let requestId = 0;

interface PendingRequest {
  resolve: (value: Record<string, unknown>) => void;
  reject: (reason: unknown) => void;
  onProgress?: (p: unknown) => void;
}

const pending = new Map<number, PendingRequest>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(
      new URL('./parser.worker.ts', import.meta.url),
      { type: 'module' },
    );
    worker.addEventListener('message', onWorkerMessage);
  }
  return worker;
}

function onWorkerMessage(e: MessageEvent) {
  const { type, id } = e.data;
  const req = pending.get(id);
  if (!req) return;

  switch (type) {
    case 'progress':
      req.onProgress?.(e.data.progress);
      break;
    case 'result':
      pending.delete(id);
      req.resolve(e.data);
      break;
    case 'error':
      pending.delete(id);
      req.reject(new Error(e.data.error));
      break;
  }
}

function sendToWorker(
  message: Record<string, unknown>,
  onProgress?: (p: unknown) => void,
  transfer?: Transferable[],
): Promise<Record<string, unknown>> {
  const id = ++requestId;
  const w = getWorker();

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    w.postMessage({ ...message, id }, transfer ?? []);
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}

// Worker-backed handle: wraps the numeric handle ID living in the worker.
interface WorkerFileHandle {
  __workerHandleId: number;
}

// ---------------------------------------------------------------------------
// Web backend
// ---------------------------------------------------------------------------

const webBackend: PlatformBackend = {
  async parsePS4GFile(
    file: FileInput,
    onProgress?: ProgressCallback<PS4GProgress>,
  ) {
    const browserFile = file as File;
    const buffer = await readFileAsArrayBuffer(browserFile);

    const response = await sendToWorker(
      { type: 'parsePS4G', data: buffer, fileSize: browserFile.size },
      onProgress as ((p: unknown) => void) | undefined,
      [buffer],
    );

    const result = response.result as PS4GParseResult;
    const handle: WorkerFileHandle = {
      __workerHandleId: response.handleId as number,
    };

    return { result, handle: handle as unknown as FileHandle };
  },

  async getChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    columnMode: ColumnMode,
    onProgress?: ProgressCallback<ChromosomeMatrixProgress>,
  ) {
    const wfh = handle as unknown as WorkerFileHandle;

    const response = await sendToWorker(
      {
        type: 'getChromosomeMatrix',
        handleId: wfh.__workerHandleId,
        chromosome,
        columnMode,
      },
      onProgress as ((p: unknown) => void) | undefined,
    );

    return response.result as ChromosomeMatrixResult;
  },

  async parseBEDFile(
    file: FileInput,
    onProgress?: ProgressCallback<BEDProgress>,
  ) {
    const browserFile = file as File;
    const buffer = await readFileAsArrayBuffer(browserFile);

    const response = await sendToWorker(
      { type: 'parseBED', data: buffer, fileSize: browserFile.size },
      onProgress as ((p: unknown) => void) | undefined,
      [buffer],
    );

    const result = response.result as BEDParseResult;
    const handle: WorkerFileHandle = {
      __workerHandleId: response.handleId as number,
    };

    return { result, handle: handle as unknown as FileHandle };
  },

  async getBEDChromosomeMatrix(
    handle: FileHandle,
    chromosome: string,
    onProgress?: ProgressCallback<BEDMatrixProgress>,
  ) {
    const wfh = handle as unknown as WorkerFileHandle;

    const response = await sendToWorker(
      {
        type: 'getChromosomeMatrix',
        handleId: wfh.__workerHandleId,
        chromosome,
      },
      onProgress as ((p: unknown) => void) | undefined,
    );

    return response.result as BEDChromosomeMatrixResult;
  },

  async loadNpyOverlay(
    observed: FileInput | null,
    predictions: FileInput | null,
    expectedNumPositions: number,
    expectedNumGametes: number,
    sourceRows: number[] | null,
    totalRows: number,
  ) {
    let observedBuf = new ArrayBuffer(0);
    let predictionsBuf = new ArrayBuffer(0);

    if (observed && observed instanceof File) {
      observedBuf = await readFileAsArrayBuffer(observed);
    }
    if (predictions && predictions instanceof File) {
      predictionsBuf = await readFileAsArrayBuffer(predictions);
    }

    const transfer: Transferable[] = [];
    if (observedBuf.byteLength > 0) transfer.push(observedBuf);
    if (predictionsBuf.byteLength > 0) transfer.push(predictionsBuf);

    const response = await sendToWorker(
      {
        type: 'loadNpyOverlay',
        observedData: observedBuf,
        predictionsData: predictionsBuf,
        expectedNumPositions,
        expectedNumGametes,
        sourceRows: sourceRows ?? [],
        totalRows,
      },
      undefined,
      transfer,
    );

    return response.result as NpyOverlayResult;
  },

  async openFile(options?: OpenFileOptions): Promise<FileInput | null> {
    return new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.style.display = 'none';

      if (options?.filters?.length) {
        const extensions = options.filters
          .flatMap((f) => f.extensions.filter((e) => e !== '*'))
          .map((e) => `.${e}`);
        if (extensions.length > 0) {
          input.accept = extensions.join(',');
        }
      }

      input.multiple = options?.multiple ?? false;

      input.addEventListener('change', () => {
        const file = input.files?.[0] ?? null;
        document.body.removeChild(input);
        resolve(file);
      });

      input.addEventListener('cancel', () => {
        document.body.removeChild(input);
        resolve(null);
      });

      document.body.appendChild(input);
      input.click();
    });
  },

  async saveFile(data: Uint8Array, options?: SaveFileOptions): Promise<void> {
    const blob = new Blob([data]);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = options?.defaultName ?? 'download';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },
};

export default webBackend;
