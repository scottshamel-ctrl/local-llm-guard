# local-llm-guard

Call a local LLM on the workstation you are also using, without making it unusable.

Zero dependencies. One file. 29 tests. Works against any OpenAI-compatible `/v1/chat/completions`
endpoint in front of a local runtime — Ollama, llama.cpp, LM Studio, vLLM.

```bash
bun add local-llm-guard   # or: npm i local-llm-guard
```

## The problem

Running a model on your own desktop is cheap until the desktop is also the machine you work on. A
batch job that drafts fifty pieces of text is not a latency problem — it is a *shared-resource*
problem, and the failure modes are not the ones you expect:

- `keep_alive: 0` looks like the polite setting. It unloads the model after every call, so a
  fifty-generation run reloads a 4.7 GB model fifty times. Hundreds of gigabytes of disk reads and
  constant VRAM churn — the thrash, not the inference, is what makes the machine unusable.
- A cooldown between calls is not a lock. Three workers reading the same `lastFinished` timestamp
  compute the same wait, sleep in lockstep, and fire simultaneously. A `--concurrency 3` batch put
  three CPU generations on one desktop at once and wedged it.
- A flat request timeout cannot be right for two different output caps. A 120 s deadline that clears
  a 260-token call by seven seconds is already underwater on a 900-token one.
- An uncapped generation runs for as long as the model feels like. A 57-word email does not need
  4,096 tokens of budget.

This module is the single call path that makes those mistakes impossible to re-introduce from a call
site.

## Usage

```ts
import { callLocalModel, cpuCallTimeoutMs, unloadLocalModel } from "local-llm-guard";

// One pacer per run. Every caller sharing it is serialized against the same machine.
const pacer = { lastFinished: null };
const url = "http://127.0.0.1:11434/v1/chat/completions";
const model = "llama3:8b";

try {
  for (const section of sections) {
    const text = await callLocalModel(section.prompt, {
      url,
      model,
      temperature: 0.7,
      maxOutputTokens: section.cap,          // required, and bounded
      timeoutMs: cpuCallTimeoutMs(section.cap), // deadline derived from the cap
      cooldownMs: 30_000,
      pacer,
      onFailure: (reason) => console.error(`[${section.id}] ${reason}`),
    });
    if (text === null) continue; // onFailure already said why
    await save(section.id, text);
  }
} finally {
  // Free VRAM once, at the end — including on failure and on Ctrl-C.
  await unloadLocalModel({ url, model });
}
```

`callLocalModel` returns `string | null`. It never throws on a transport or server error; it reports
the reason through `onFailure` and returns `null`, so one bad call in a fifty-call batch does not
take the batch down.

## What it guarantees

| Guarantee | Mechanism |
|---|---|
| Never more than one generation in flight per pacer | Promise chain, not a timestamp comparison |
| Cooldown is honored after failures too | `lastFinished` set in a `finally` block |
| One caller's crash cannot poison the pacer | The stored chain swallows rejection; the caller still sees it |
| Every request is output-capped | `maxOutputTokens` is required and bounded at 2048 |
| The model loads once per run, not once per call | `keep_alive: "5m"` + one explicit unload |
| An abandoned run still frees the GPU | Residency is bounded, not indefinite |
| The deadline scales with the cap | `cpuCallTimeoutMs(cap)`, clamped to the server ceiling |
| A `null` always says why | Timeout, HTTP status + body, and empty-200 are distinct reasons |
| No silent fallback to a paid API | `local_only: true` asserted client-side on every request |

## The constants are measurements, not guesses

Every tuning number in `src/index.ts` carries the observation that set it, and the tests pin them
against those observations so a later edit cannot quietly walk one back:

- **`LOAD_HEADROOM_MS = 90_000`** — a 4.5k-char prompt capped at 260 tokens took **113.0 s** cold on
  a CPU-only server: roughly 55 s to load the model plus ~3.4 tok/s of generation.
- **`SLOWEST_TOKENS_PER_SEC = 1.2`** — a five-call job measured 2.15, 2.04 and 2.01 tok/s on three
  consecutive sections, then missed a 298 s deadline on the fourth at under 1.8 tok/s. The constant
  sits below the slowest rate actually observed, not near the average of the healthy ones. A timeout
  that fires on healthy work costs a whole job; one that fires late costs only wall time on a
  detached run.
- **`DEFAULT_GPU_LAYERS = 0`** — the GPU that would run those layers is the one drawing your screen.
  Residency fixed the reload thrash, but a full-offload generation still pins the GPU for its whole
  duration. CPU-only trades ~40 tok/s for ~5-10 tok/s and a usable desktop. Raise it deliberately
  when the machine is idle.
- **`DEFAULT_THREADS = 4`** — a run that saturates every core lags the desktop as badly as the GPU.
- **`MAX_COMPLETABLE_TOKENS`** — derived, not configured: the largest cap that can finish inside the
  server ceiling at the slowest observed rate. It is well *below* the 2048 hard cap, and the two are
  not interchangeable. Size CPU prompts against this one.

## Testing

```bash
bun install
bun test        # 29 tests
bun run typecheck
```

The module takes its I/O through an injected `LocalModelDeps` (`fetch`, `now`, `sleep`), so the
suite runs on a virtual clock with no network and no sleeping. Concurrency safety is asserted by
counting peak in-flight requests rather than by timing, which is why it is deterministic.

## Origin

Extracted from a production content pipeline where the local runtime, the batch workers, and the
developer's desktop were the same machine. The comments preserve the incident each rule came from —
they are the point of the file, not decoration.

## License

MIT
