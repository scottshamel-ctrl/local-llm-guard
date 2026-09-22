import { describe, expect, test } from "bun:test";
import {
  callLocalModel,
  cpuCallTimeoutMs,
  localModelRequest,
  unloadLocalModel,
  RESIDENT_KEEP_ALIVE,
  SERVER_CEILING_MS,
  MAX_COMPLETABLE_TOKENS,
  SLOWEST_TOKENS_PER_SEC,
  MAX_ALLOWED_OUTPUT_TOKENS,
} from "./index";

const OK = (content: string) =>
  new Response(JSON.stringify({ choices: [{ message: { content } }] }), { status: 200 });

describe("localModelRequest", () => {
  // Was: keep_alive === 0 on every call. That freed VRAM but reloaded 4.7GB per generation, ~50
  // times a run, which is what actually bogged the machine down. VRAM is now freed once at the end
  // by unloadLocalModel; residency is bounded by RESIDENT_KEEP_ALIVE so an abandoned run still
  // releases the GPU on its own.
  test("never leaves the model squatting in VRAM indefinitely", () => {
    const body = localModelRequest({ model: "llama3:8b", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.keep_alive).toBe(RESIDENT_KEEP_ALIVE);
    expect(String(body.keep_alive)).toMatch(/^\d+m$/);
  });

  test("every call is output-capped — an uncapped generation is what bogs the machine down", () => {
    const body = localModelRequest({ model: "llama3:8b", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.max_tokens).toBe(220);
  });

  test("the cap is required, so no call site can forget it", () => {
    // @ts-expect-error maxOutputTokens is not optional
    expect(() => localModelRequest({ model: "m", prompt: "hi", temperature: 0 })).toThrow(/cap/i);
  });

  test("rejects a cap large enough to defeat the point", () => {
    expect(() =>
      localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 8000 })
    ).toThrow(/cap/i);
  });

  test("does not raise num_ctx — VRAM is the constraint, prompts get trimmed instead", () => {
    const body = localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body).not.toHaveProperty("num_ctx");
  });

  // The GPU also draws your screen. Residency stopped the reload thrash, but a full-offload
  // generation still pins the GPU for its whole duration, which is felt as lag while you work.
  // num_gpu is the only setting that takes the run off that GPU entirely.
  test("defaults to CPU-only so a run cannot pin the display GPU", () => {
    const body = localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.num_gpu).toBe(0);
  });

  test("leaves CPU headroom too — a run that eats every core lags the machine just as badly", () => {
    const body = localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.num_thread).toBe(4);
  });

  test("never permits the server to use a paid fallback", () => {
    const body = localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.local_only).toBe(true);
  });

  test("GPU layers can be raised deliberately when the machine is idle", () => {
    const body = localModelRequest({
      model: "m",
      prompt: "hi",
      temperature: 0,
      maxOutputTokens: 220,
      gpuLayers: 20,
      threads: 8,
    });
    expect(body.num_gpu).toBe(20);
    expect(body.num_thread).toBe(8);
  });

  test("rejects a negative layer count rather than sending nonsense to the runtime", () => {
    expect(() =>
      localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220, gpuLayers: -1 })
    ).toThrow(/gpu/i);
  });

  test("rejects a thread count below one", () => {
    expect(() =>
      localModelRequest({ model: "m", prompt: "hi", temperature: 0, maxOutputTokens: 220, threads: 0 })
    ).toThrow(/thread/i);
  });
});

describe("callLocalModel pacing", () => {
  test("waits the full cooldown between calls on the same pacer", async () => {
    let now = 0;
    const slept: number[] = [];
    const deps = {
      fetch: async () => OK("out"),
      now: () => now,
      sleep: async (ms: number) => {
        slept.push(ms);
        now += ms;
      },
    };
    const pacer: { lastFinished: number | null } = { lastFinished: null };
    const opts = { url: "http://x/v1/chat/completions", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 60_000 };

    await callLocalModel("one", { ...opts, pacer }, deps);
    await callLocalModel("two", { ...opts, pacer }, deps);

    expect(slept).toEqual([60_000]);
  });

  test("concurrent callers never overlap on the machine, they queue", async () => {
    // The crash this exists to prevent: a `--concurrency 3` batch ran three workers on one pacer.
    // `lastFinished` delayed them all by the same amount from the same timestamp, so they slept in
    // lockstep and then fired together — three CPU generations at once. Assert the invariant that
    // actually protects the machine: never more than one call in flight.
    let now = 0;
    let inFlight = 0;
    let peak = 0;
    const deps = {
      fetch: async () => {
        inFlight++;
        peak = Math.max(peak, inFlight);
        await new Promise((r) => setTimeout(r, 5));
        inFlight--;
        return OK("out");
      },
      now: () => now,
      sleep: async (ms: number) => {
        now += ms;
      },
    };
    const pacer: { lastFinished: number | null } = { lastFinished: null };
    const opts = { url: "http://x", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 20_000 };

    await Promise.all([
      callLocalModel("a", { ...opts, pacer }, deps),
      callLocalModel("b", { ...opts, pacer }, deps),
      callLocalModel("c", { ...opts, pacer }, deps),
    ]);

    expect(peak).toBe(1);
  });

  test("one caller's failure does not poison later calls on the same pacer", async () => {
    // The chain is reassigned on every call, so a rejected turn must be caught before it becomes
    // the next call's predecessor — otherwise one thrown fetch would break the pacer for the run.
    let now = 0;
    let first = true;
    const deps = {
      fetch: async () => {
        if (first) {
          first = false;
          throw new Error("socket died");
        }
        return OK("second");
      },
      now: () => now,
      sleep: async (ms: number) => {
        now += ms;
      },
    };
    const pacer: { lastFinished: number | null } = { lastFinished: null };
    const opts = { url: "http://x", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 0 };

    expect(await callLocalModel("one", { ...opts, pacer }, deps)).toBeNull();
    expect(await callLocalModel("two", { ...opts, pacer }, deps)).toBe("second");
  });

  test("does not sleep when the cooldown has already elapsed", async () => {
    let now = 0;
    const slept: number[] = [];
    const deps = {
      fetch: async () => OK("out"),
      now: () => now,
      sleep: async (ms: number) => {
        slept.push(ms);
        now += ms;
      },
    };
    const pacer: { lastFinished: number | null } = { lastFinished: null };
    const opts = { url: "http://x", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 60_000 };

    await callLocalModel("one", { ...opts, pacer }, deps);
    now += 90_000;
    await callLocalModel("two", { ...opts, pacer }, deps);

    expect(slept).toEqual([]);
  });

  test("a failed call still starts the cooldown, so an error loop cannot hammer the GPU", async () => {
    let now = 0;
    const slept: number[] = [];
    const deps = {
      fetch: async () => new Response("boom", { status: 500 }),
      now: () => now,
      sleep: async (ms: number) => {
        slept.push(ms);
        now += ms;
      },
    };
    const pacer: { lastFinished: number | null } = { lastFinished: null };
    const opts = { url: "http://x", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 60_000 };

    expect(await callLocalModel("one", { ...opts, pacer }, deps)).toBeNull();
    await callLocalModel("two", { ...opts, pacer }, deps);

    expect(slept).toEqual([60_000]);
  });
});

// A run died on "the model returned nothing" and that string was the whole diagnosis, so the first
// fix went to the wrong layer. A null must always be able to say why it is null.
describe("a null says why", () => {
  const base = { url: "http://x", model: "m", temperature: 0, maxOutputTokens: 220, cooldownMs: 0 };
  const stub = (fetch: () => Promise<Response>) => ({ fetch, now: () => 0, sleep: async () => {} });

  test("a timeout is named as a timeout, not as a dead server", async () => {
    const reasons: string[] = [];
    const deps = stub(async () => {
      throw Object.assign(new Error("The operation timed out."), { name: "TimeoutError" });
    });
    const result = await callLocalModel("p", { ...base, pacer: { lastFinished: null }, timeoutMs: 120_000, onFailure: (r) => reasons.push(r) }, deps);

    expect(result).toBeNull();
    expect(reasons).toEqual(["timed out after 120s"]);
  });

  test("an HTTP error reports its status and body", async () => {
    const reasons: string[] = [];
    const deps = stub(async () => new Response("model not found", { status: 502 }));

    expect(await callLocalModel("p", { ...base, pacer: { lastFinished: null }, onFailure: (r) => reasons.push(r) }, deps)).toBeNull();
    expect(reasons).toEqual(["HTTP 502: model not found"]);
  });

  // A 200 carrying an empty completion looks identical to a timeout at the call site and needs a
  // different fix — the prompt, not the deadline.
  test("an empty 200 is distinguished from a failed request", async () => {
    const reasons: string[] = [];
    const deps = stub(async () => OK("   "));

    expect(await callLocalModel("p", { ...base, pacer: { lastFinished: null }, onFailure: (r) => reasons.push(r) }, deps)).toBeNull();
    expect(reasons).toEqual(["HTTP 200 with empty content"]);
  });

  test("a successful call reports no failure", async () => {
    const reasons: string[] = [];
    const deps = stub(async () => OK("body"));

    expect(await callLocalModel("p", { ...base, pacer: { lastFinished: null }, onFailure: (r) => reasons.push(r) }, deps)).toBe("body");
    expect(reasons).toEqual([]);
  });
});

// A drafting job timed out on a call the server answered successfully in 113.0s, because callers
// hardcoded 120s for every cap. These pin the deadline against that measurement so a future edit
// cannot quietly walk it back under the observed latency.
describe("cpuCallTimeoutMs", () => {
  // The live 260-token lede: 113.0s cold, on the flat 120s deadline it was ~7s from failing.
  test("clears the measured cold-start latency with real margin", () => {
    const budget = cpuCallTimeoutMs(260);
    expect(budget).toBeGreaterThan(113_000 * 1.5);
    expect(budget).toBeLessThan(900_000);
  });

  // A 520-token section ran past a 298s deadline on a live job — under 1.8 tok/s, where its three
  // predecessors had held ~2.7. The budget has to cover the slow tail, not the healthy average.
  test("a 520-token section survives the slowest rate actually observed", () => {
    expect(cpuCallTimeoutMs(520)).toBeGreaterThan(298_000 * 1.4);
  });

  // The old flat deadline was the defect: one number cannot cover both calls in a job.
  test("a longer cap earns a longer deadline", () => {
    expect(cpuCallTimeoutMs(520)).toBeGreaterThan(cpuCallTimeoutMs(260));
  });

  // MAX_ALLOWED_OUTPUT_TOKENS (2048) is far above what the CPU path can finish inside the server
  // ceiling (~504). That is not a contradiction — the ceiling binds only when gpuLayers is 0 — but a
  // caller picking a cap for a CPU run has to size against MAX_COMPLETABLE_TOKENS, not the hard cap.
  test("the hard cap exceeds what the CPU path can finish, so the two are not interchangeable", () => {
    expect(MAX_COMPLETABLE_TOKENS).toBeLessThan(MAX_ALLOWED_OUTPUT_TOKENS);
    expect(cpuCallTimeoutMs(MAX_ALLOWED_OUTPUT_TOKENS)).toBe(SERVER_CEILING_MS);
  });

  // The server clamps every request DOWN to its own ceiling and never up, so a client budget above
  // it is a number that can never be reached — the server aborts first.
  test("never advertises a budget past the server ceiling", () => {
    expect(cpuCallTimeoutMs(2048)).toBeLessThanOrEqual(SERVER_CEILING_MS);
    expect(cpuCallTimeoutMs(MAX_ALLOWED_OUTPUT_TOKENS)).toBeLessThanOrEqual(SERVER_CEILING_MS);
  });

  // Above this, a slow run cannot finish inside the ceiling at all — no deadline fixes it, only a
  // smaller cap or a larger ceiling on the server.
  test("names the largest cap that can actually complete", () => {
    expect(MAX_COMPLETABLE_TOKENS).toBe(Math.floor((SERVER_CEILING_MS / 1000) * SLOWEST_TOKENS_PER_SEC));
    expect(cpuCallTimeoutMs(MAX_COMPLETABLE_TOKENS)).toBeLessThanOrEqual(SERVER_CEILING_MS);
  });
});

// keep_alive: 0 was added to protect the machine and did the opposite. A batch run is ~50
// generations; unloading after every one means reloading 4.7GB from disk ~50 times, which is what
// actually made the machine unusable. The model should load once, stay resident for the run, and be
// unloaded once at the end.
describe("the model loads once per run, not once per call", () => {
  test("a call inside a run keeps the model resident", () => {
    const body = localModelRequest({
      model: "llama3:8b",
      prompt: "hi",
      temperature: 0,
      maxOutputTokens: 220,
      keepAlive: "5m",
    });
    expect(body.keep_alive).toBe("5m");
  });

  test("the default is still resident, so no call site re-creates the thrash by omission", () => {
    const body = localModelRequest({ model: "llama3:8b", prompt: "hi", temperature: 0, maxOutputTokens: 220 });
    expect(body.keep_alive).not.toBe(0);
  });

  test("a run can still free VRAM explicitly when it finishes", async () => {
    const seen: string[] = [];
    const deps = {
      fetch: async (_url: string, init?: RequestInit) => {
        seen.push(String(init?.body ?? ""));
        return OK("");
      },
      now: () => 0,
      sleep: async () => {},
    };
    await unloadLocalModel({ url: "http://x/v1/chat/completions", model: "llama3:8b" }, deps as never);
    expect(seen).toHaveLength(1);
    expect(JSON.parse(seen[0])).toMatchObject({ keep_alive: 0, local_only: true });
  });
});
