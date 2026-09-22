/**
 * One safe way to call a local LLM that runs on the same machine you are using.
 *
 * Three call sites in the original codebase had each grown their own copy of this request with
 * different safety settings: one paced itself and unloaded the model, one unloaded but never paced,
 * and one did neither — it left the model squatting in VRAM between calls. None of the three capped
 * the output, so a 57-word email could generate for as long as the model felt like, which is the
 * load that makes a workstation unusable while you are working on it.
 *
 * The rules, in one place so a call site cannot opt out:
 *   - keep_alive holds the model RESIDENT for the run, and the run unloads once at the end via
 *     `unloadLocalModel`. This reverses the obvious `keep_alive: 0`, which unloads after every call:
 *     a batch run is ~50 generations, so that meant reloading a 4.7GB model ~50 times — hundreds of
 *     GB of disk reads and constant VRAM churn. That thrash, not the inference, is what made the
 *     desktop unusable. Occupied VRAM during a run is what a GPU is for; reloading it fifty times
 *     is not.
 *   - max_tokens is REQUIRED and bounded. An uncapped generation is the whole problem.
 *   - num_ctx is never raised. VRAM is the constraint; 16384 returned 502s and bogged the machine
 *     down, so prompts get trimmed to fit the server default instead.
 *   - a shared pacer serializes calls and enforces a cooldown between them, including after a
 *     failure, so an error loop cannot hammer the GPU.
 *
 * Targets any OpenAI-compatible `/v1/chat/completions` endpoint in front of a local runtime
 * (Ollama, llama.cpp, LM Studio, vLLM).
 */

/** Above this a "cap" is not a cap — it is more generation than any caller has a use for. */
export const MAX_ALLOWED_OUTPUT_TOKENS = 2048;

/**
 * How long the model stays resident between calls. Long enough to cover a cooldown plus the next
 * generation, short enough that an abandoned run frees the machine's VRAM on its own.
 */
export const RESIDENT_KEEP_ALIVE = "5m";

/**
 * GPU layers offloaded by default. Zero, because the GPU that would run those layers is the same one
 * drawing your screen: residency fixed the reload thrash, but a full-offload generation still pins
 * the GPU for its entire duration, and that is what you feel as lag while working.
 *
 * CPU-only trades wall time for a usable desktop — roughly 5-10 tok/s instead of 40+. Batch runs are
 * detached and paced anyway, so the wall time is not on anyone's critical path. Raise `gpuLayers`
 * deliberately (a flag, not a default) when the machine is idle.
 */
export const DEFAULT_GPU_LAYERS = 0;
/** Leaves cores free for the desktop. A run that saturates every core lags it as badly as the GPU. */
export const DEFAULT_THREADS = 4;

/**
 * A deadline that scales with the output cap it is guarding.
 *
 * Measured on a live CPU-only server, cold: a 4.5k-char prompt capped at 260 tokens took 113.0s wall
 * — roughly 55s to load the model plus ~3.4 tok/s of generation. Callers used a flat 120s, which left
 * seven seconds of margin on the SHORTEST call in a multi-section job and was under water on a
 * 900-token section. The first live sectioned run died on its opening call and reported it as "the
 * model returned nothing".
 *
 * A fixed number cannot be right for both a 260-token lede and a 900-token section, so callers pass
 * the cap and this converts it. Rates are deliberately pessimistic: a timeout that fires on healthy
 * work costs a whole job, while one that fires late costs only wall time on a detached run.
 *
 * The rate was 2.5 tok/s on the first pass and that was still too tight. A five-call job measured
 * 2.15, 2.04 and 2.01 tok/s on three consecutive sections and then missed a 298s deadline on the
 * fourth — the same prompt shape at under 1.8 tok/s. The generation rate is stable but not
 * guaranteed, so the constant is set below the slowest run actually observed rather than near the
 * average of the healthy ones.
 */
export const LOAD_HEADROOM_MS = 90_000;
export const SLOWEST_TOKENS_PER_SEC = 1.2;

/**
 * Servers in front of a local runtime typically clamp every request DOWN to their own ceiling and
 * never up, so a client deadline above this can never be reached — the server aborts first. Keeping
 * the clamp here stops callers advertising a budget they do not have.
 *
 * Must match the request timeout configured on your server. That setting, not this file, is the
 * lever if a call genuinely needs longer.
 */
export const SERVER_CEILING_MS = Number(process.env.LOCAL_LLM_CEILING_MS || "420000");

export function cpuCallTimeoutMs(maxOutputTokens: number): number {
  const wanted = LOAD_HEADROOM_MS + Math.round((maxOutputTokens / SLOWEST_TOKENS_PER_SEC) * 1000);
  return Math.min(wanted, SERVER_CEILING_MS);
}

/**
 * The largest output cap that can finish inside the server ceiling at the slowest observed rate,
 * once the model is resident (the load headroom is only paid by the first call of a run). Callers
 * sizing a new prompt should stay at or under this; above it, a slow run cannot complete at all.
 */
export const MAX_COMPLETABLE_TOKENS = Math.floor((SERVER_CEILING_MS / 1000) * SLOWEST_TOKENS_PER_SEC);

export interface LocalModelBody {
  model: string;
  temperature: number;
  keep_alive: string | 0;
  max_tokens: number;
  num_gpu: number;
  num_thread: number;
  /**
   * Forbids the server from falling back to a paid cloud provider when the local runtime is
   * unavailable. A local-first pipeline that silently degrades to a metered API is a billing
   * surprise, not a fallback — so the refusal is asserted by the client on every request rather
   * than left to server configuration.
   */
  local_only: true;
  messages: { role: "user"; content: string }[];
}

export function localModelRequest(input: {
  model: string;
  prompt: string;
  temperature: number;
  maxOutputTokens: number;
  keepAlive?: string;
  gpuLayers?: number;
  threads?: number;
}): LocalModelBody {
  const cap = input.maxOutputTokens;
  if (!Number.isInteger(cap) || cap < 1 || cap > MAX_ALLOWED_OUTPUT_TOKENS) {
    throw new Error(
      `local model needs an output cap between 1 and ${MAX_ALLOWED_OUTPUT_TOKENS} tokens; got ${cap}`
    );
  }
  const gpuLayers = input.gpuLayers ?? DEFAULT_GPU_LAYERS;
  if (!Number.isInteger(gpuLayers) || gpuLayers < 0) {
    throw new Error(`local model gpu layer count must be a non-negative integer; got ${gpuLayers}`);
  }
  const threads = input.threads ?? DEFAULT_THREADS;
  if (!Number.isInteger(threads) || threads < 1) {
    throw new Error(`local model thread count must be at least 1; got ${threads}`);
  }
  return {
    model: input.model,
    temperature: input.temperature,
    keep_alive: input.keepAlive ?? RESIDENT_KEEP_ALIVE,
    max_tokens: cap,
    num_gpu: gpuLayers,
    num_thread: threads,
    local_only: true,
    messages: [{ role: "user", content: input.prompt }],
  };
}

/**
 * Frees the machine's VRAM. Call once when a run finishes — including on failure and on Ctrl-C —
 * because the resident model would otherwise sit there until RESIDENT_KEEP_ALIVE elapses.
 */
export async function unloadLocalModel(
  options: { url: string; model: string },
  deps: LocalModelDeps = liveDeps
): Promise<void> {
  try {
    await deps.fetch(options.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: options.model,
        keep_alive: 0,
        max_tokens: 1,
        local_only: true,
        messages: [{ role: "user", content: "" }],
      }),
      signal: AbortSignal.timeout(20_000),
    });
  } catch {
    // An unload that fails is not worth failing a finished run over; keep_alive expires on its own.
  }
}

/**
 * Mutable so every call site sharing one object shares one cooldown. `null` means no call has run
 * yet, so the first one starts immediately — seeding the timestamp with 0 instead gets that by
 * accident, by comparing against a `Date.now()` in the trillions.
 */
export interface Pacer {
  lastFinished: number | null;
  /**
   * Serializes overlapping callers. `lastFinished` alone only ever *delayed* a call — it never
   * stopped two from running at once, because N concurrent workers all read the same timestamp,
   * computed the same wait, slept it, and then fired together. A `--concurrency 3` batch did exactly
   * that and put three simultaneous CPU generations on the desktop, wedging it. The cooldown was
   * never the thing that broke; it just was not a lock. Every call now queues on this chain, so the
   * machine sees one generation at a time no matter how many workers a call site runs. Undefined
   * until the first call.
   */
  chain?: Promise<void>;
}

export interface LocalModelDeps {
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
  now: () => number;
  sleep: (ms: number) => Promise<void>;
}

const liveDeps: LocalModelDeps = {
  fetch: (url, init) => fetch(url, init),
  now: () => Date.now(),
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

export async function callLocalModel(
  prompt: string,
  options: {
    url: string;
    model: string;
    temperature: number;
    maxOutputTokens: number;
    cooldownMs: number;
    pacer: Pacer;
    timeoutMs?: number;
    gpuLayers?: number;
    threads?: number;
    /**
     * Why a call returned null. Every failure here used to collapse into a bare `null`, so a run
     * that died on a 120s timeout logged the same "the model returned nothing" as one that got a
     * 502 — and the first diagnosis went to the wrong layer. Callers log this.
     */
    onFailure?: (reason: string) => void;
  },
  deps: LocalModelDeps = liveDeps
): Promise<string | null> {
  const body = localModelRequest({
    model: options.model,
    prompt,
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
    gpuLayers: options.gpuLayers,
    threads: options.threads,
  });
  // Wait AND request both sit inside the chained turn. Chaining only the wait would not help: each
  // waiter would time its cooldown from when the previous waiter stopped sleeping rather than from
  // when the previous generation actually ended, and they would overlap again.
  const turn = (options.pacer.chain ?? Promise.resolve()).then(() => attempt());
  // Swallow here only — the caller still sees the real settlement through `turn`. A rejected chain
  // must not poison every later call on this pacer.
  options.pacer.chain = turn.then(
    () => undefined,
    () => undefined
  );
  return turn;

  async function attempt(): Promise<string | null> {
    const since = options.pacer.lastFinished;
    const wait = since === null ? 0 : since + options.cooldownMs - deps.now();
    if (wait > 0) await deps.sleep(wait);
    try {
      const response = await deps.fetch(options.url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(options.timeoutMs ?? 180_000),
      });
      if (!response.ok) {
        options.onFailure?.(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
        return null;
      }
      const data = (await response.json()) as { choices?: { message?: { content?: string } }[] };
      const content = data?.choices?.[0]?.message?.content?.trim() || null;
      if (content === null) options.onFailure?.("HTTP 200 with empty content");
      return content;
    } catch (error) {
      // A TimeoutError here is the one failure that is not the server's fault: CPU inference is slow
      // (DEFAULT_GPU_LAYERS is 0), so naming it separately keeps the next reader from restarting a
      // healthy server.
      const e = error as Error;
      options.onFailure?.(
        e.name === "TimeoutError"
          ? `timed out after ${(options.timeoutMs ?? 180_000) / 1000}s`
          : `${e.name}: ${e.message}`
      );
      return null;
    } finally {
      // In the finally block on purpose: a timeout or a 500 still ran the GPU, and a retry loop that
      // skipped the cooldown on failure would be the worst case for the machine, not the best.
      options.pacer.lastFinished = deps.now();
    }
  }
}
