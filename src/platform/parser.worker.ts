import init, {
  parse_ps4g_file,
  parse_ps4g_to_handle,
  parse_bed_file,
  parse_bed_to_handle,
  load_npy_overlay,
  type PS4GFileHandle,
  type BEDFileHandle,
} from '../wasm/pkg/parser_wasm';

let wasmReady: Promise<void> | null = null;

function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = init().then(() => {});
  }
  return wasmReady;
}

type AnyHandle = PS4GFileHandle | BEDFileHandle;

let nextHandleId = 1;
const handles = new Map<number, AnyHandle>();
const handleTypes = new Map<number, 'ps4g' | 'bed'>();

function storeHandle(handle: AnyHandle, kind: 'ps4g' | 'bed'): number {
  const id = nextHandleId++;
  handles.set(id, handle);
  handleTypes.set(id, kind);
  return id;
}

function postProgress(id: number, progress: unknown) {
  self.postMessage({ type: 'progress', id, progress });
}

async function handleMessage(msg: MessageEvent) {
  const { data } = msg;
  const id: number = data.id;

  try {
    await ensureWasm();

    switch (data.type) {
      case 'parsePS4G': {
        const bytes = new Uint8Array(data.data as ArrayBuffer);
        const fileSize = BigInt(data.fileSize);

        const result = parse_ps4g_file(bytes, fileSize, (p: unknown) => {
          postProgress(id, p);
        });

        const handle = parse_ps4g_to_handle(bytes, fileSize, undefined);
        const handleId = storeHandle(handle, 'ps4g');

        self.postMessage({ type: 'result', id, result, handleId });
        break;
      }

      case 'parseBED': {
        const bytes = new Uint8Array(data.data as ArrayBuffer);
        const fileSize = BigInt(data.fileSize);

        const result = parse_bed_file(bytes, fileSize, (p: unknown) => {
          postProgress(id, p);
        });

        const handle = parse_bed_to_handle(bytes, fileSize, undefined);
        const handleId = storeHandle(handle, 'bed');

        self.postMessage({ type: 'result', id, result, handleId });
        break;
      }

      case 'getChromosomeMatrix': {
        const handle = handles.get(data.handleId);
        if (!handle) {
          throw new Error(`Handle ${data.handleId} not found`);
        }

        const kind = handleTypes.get(data.handleId);
        let result;

        if (kind === 'ps4g') {
          result = (handle as PS4GFileHandle).get_chromosome_matrix(
            data.chromosome,
            data.columnMode,
          );
        } else {
          result = (handle as BEDFileHandle).get_chromosome_matrix(
            data.chromosome,
            (p: unknown) => postProgress(id, p),
          );
        }

        self.postMessage({ type: 'result', id, result });
        break;
      }

      case 'loadNpyOverlay': {
        const observed = new Uint8Array(data.observedData as ArrayBuffer);
        const predictions = new Uint8Array(data.predictionsData as ArrayBuffer);
        const sourceRows = new Uint32Array((data.sourceRows as number[] | undefined) ?? []);

        const result = load_npy_overlay(
          observed,
          predictions,
          data.expectedNumPositions,
          data.expectedNumGametes,
          sourceRows,
          data.totalRows ?? 0,
        );

        self.postMessage({ type: 'result', id, result });
        break;
      }

      case 'freeHandle': {
        const handle = handles.get(data.handleId);
        if (handle) {
          handle.free();
          handles.delete(data.handleId);
          handleTypes.delete(data.handleId);
        }
        break;
      }
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'error', id, error: errorMsg });
  }
}

self.addEventListener('message', handleMessage);
