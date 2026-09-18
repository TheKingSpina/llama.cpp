# Project State

This file is the visible, repository-local counterpart of the assistant's persistent MCP Memory.

## Project

Apple-Silicon-focused llama.cpp runtime.

## Current phase

Phase 2 - opt-in Apple Silicon infrastructure.

## Current task

Add memory-pressure classification and tier reservation bookkeeping (memory_budget) on top of the telemetry snapshot; keep default inference behavior unchanged.

## Accelerated order

1. Frozen baseline.
2. Optional telemetry.
3. Explicit memory tiers.
4. Allocation and memory-pressure metrics.
5. Minimal node registration and capability reporting.
6. Loopback/TCP transport abstraction.
7. Heartbeat, timeout, and controlled shutdown.
8. Experimental local scheduler.
9. Artificial distributed test before real tensor movement.
10. Remote memory, SSD streaming, and MoE afterward.

## Acceleration policy

- Follow the accelerated roadmap: build opt-in infrastructure in parallel with measurement work.
- Optimize for Apple Silicon and Metal first, with MPS-compatible design principles where applicable: unified-memory locality, asynchronous command submission, minimal synchronization, predictable tensor layouts, and low allocation overhead.
- Treat MPS as a compatibility target and measurement reference, not as proof that a change improves the native ggml Metal backend.
- Preserve CPU and non-Apple fallbacks. New paths must be optional until correctness and benchmark evidence support promotion.

## Memory status

MCP Memory is available. This file is kept in the repository so the project state is visible in VS Code and reviewable by a human.

## Build status

- CMake and Ninja installed with Homebrew.
- CMake configuration succeeded in `build/` with Release and Metal enabled.
- Build succeeded on 2026-08-27 through target `llama` (`604/604`).
- Smoke test passed: `llama-cli --version` reports commit `deae5ee13`, Darwin arm64, AppleClang 21.0.0.21000101.
- Non-blocking warnings: OpenMP and ccache are not installed.

## Initial baseline

- Model: `models/baseline/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (469 MB, SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`).
- Commit: `deae5ee13`, M3, AppleClang 21.0.0.21000101.
- `llama-bench`, 3 repetitions, `p=512`, `n=128`, `batch=512`, `ubatch=512`.
- Explicit-device follow-up, 5 repetitions: BLAS device pp 452.99 +/- 25.93 t/s and tg 45.06 +/- 5.75 t/s; MTL0 device pp 1236.36 +/- 133.51 t/s and tg 51.34 +/- 3.21 t/s.
- The benchmark backend column still reports all compiled backends (`MTL,BLAS`), but the `dev` column confirms the selected execution device (`BLAS` or `MTL0`). These results supersede the earlier 3-repetition comparison for baseline use.
- Normal-power rerun on AC power with `lowpowermode 0`, 5 repetitions: BLAS device pp 1053.96 +/- 47.41 t/s and tg 144.35 +/- 0.30 t/s; MTL0 device pp 2546.14 +/- 25.03 t/s and tg 147.15 +/- 2.06 t/s.
- Clean rerun report: `benchmarks/baseline/2026-08-27-normal-power.md`. Current rerun measured AC power with `lowpowermode 0`: BLAS device pp 1040.92 +/- 35.24 t/s and tg 129.00 +/- 6.36 t/s; MTL0 device pp 2558.55 +/- 16.70 t/s and tg 141.54 +/- 11.63 t/s.
- The earlier explicit-device run is retained as the Low Power / prior-state baseline per the user's report. Power state was not captured during that earlier run, so its label is user-reported rather than independently verified.
- Baseline artifacts are documented in `benchmarks/baseline/README.md`.
- Validation: `test-batch-alloc` passed (30 tests, 198 assertions); `test-quant-type-selection` passed 10/10 model sections with 1 external metadata fetch skipped.
- `test-backend-ops --help` returned exit 1; this was not treated as a backend test result because the executable does not expose that help mode.
- Follow-up validation on AC power with `lowpowermode 0`: `test-batch-alloc` passed (30 tests, 198 assertions); `test-quant-type-selection` passed 10/10 model sections with 1 skipped external metadata fetch. No backend-ops result was recorded because its help invocation exits 1.
- Deterministic bounded smoke with seed 123 produced the same visible answer (`Correct.`) on CPU and Metal for the baseline prompt. This is an output smoke gate, not a logits-level numerical comparison.
- Metal diagnostic run completed with `GGML_METAL_GRAPH_DEBUG=1` and `GGML_METAL_FUSION_DEBUG=1`; it produced valid output and reported 689.1 prompt t/s and 104.2 generation t/s for the short run. No graph/fusion details were emitted at the selected verbosity.
- `xctrace` is present, but cannot run because the active developer directory is CommandLineTools; full Xcode is required for Instruments traces. No kernel change was made.
- Detailed diagnostic artifact: `benchmarks/baseline/2026-08-27-metal-profile.md`.
- Numerical backend correctness: `test-backend-ops test -b MTL0 -j 1` passed `14283/14283` cases, including 240 `fa_vec (Q,NE)` cases; CPU and BLAS were skipped by the explicit Metal selection. This validates Metal operations against the test reference, but is not a full model logits comparison.
- Bounded long Metal run completed on AC with `lowpowermode 1` (therefore separate from the verified normal-power baseline): prompt 565.1 t/s and generation 131.1 t/s. Artifact: `benchmarks/baseline/2026-08-27-metal-long-profile.md`.
- `xctrace list templates` failed with exit 1 because full Xcode is not installed/selected; no System Trace was captured.
- Xcode is installed at `/Applications/Xcode.app`; using `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, `xctrace` captured two Metal traces (system trace 2.108 s, extended 3.969 s, 77 MB). GPU counters and shader timeline were disabled on this configuration, so they were lifecycle evidence only. The 120 MB trace bundles were deleted in the cleanup pass; the conclusions above and the TOC text artifact remain.
- The dedicated `fa-vec` tuner was started with `q4_0`, `dk=128`, 3 reps, and thermal safeguards. It was stopped after more than 5 minutes because the broad shape sweep exceeded the bounded experiment budget. The log showed substantial thermal drift/noise and marked cells untrusted; no tuning row was accepted as a benchmark result. Partial artifact: `benchmarks/baseline/2026-08-27-fa-vec-q4-dk128.log`.
- Targeted `test-backend-ops perf` invocations for q4_K `n=1` and `n=32` completed with backend correctness status `OK`, but emitted no timing rows because the exact case filters did not match the compiled perf case list. They are retained as failed measurement attempts, not performance data: `benchmarks/baseline/2026-08-27-mul-mv-q4k.log` and `benchmarks/baseline/2026-08-27-mul-mm-q4k.log`.
- Backend-ops filter reconnaissance: `--help` documents `-p <params regex>`, but the implementation compares the complete `op_desc(vars)` string by exact equality; it is not a regex filter. `--list-ops` lists only operation names, not perf cases. `perf -b MTL0 -o MUL_MAT --output csv` was started to enumerate cases, but the built-in perf suite exceeded the 120 s bound and was stopped; its partial CSV contains no trusted q4_K case row. Therefore no new q4_K `mul_mv` or `mul_mm` timing was accepted. The earlier no-match logs remain the evidence.
- Current power check: AC connected, `pmset` reports AC `lowpowermode 0` (battery profile still reports `1`). No new kernel benchmark was accepted because the case enumeration did not finish within the safety bound. No thermal-stability conclusion was inferred from the incomplete run.
- Flash Attention tuner: not launched in this block. No real CLI option was found that bounds its shape/grid sweep; the prior `fa-vec` run demonstrated that the broad sweep exceeds the experiment budget and has thermal drift. Launching another unrestricted tuner would violate the bounded-measurement requirement.
- Existing Metal trace analysis: `xctrace export --toc` succeeded on the now-deleted system trace bundle. It confirmed `llama-cli` exited 0 on the M3, trace duration 2.108253 s, Metal System Trace template, and the exact model/CLI arguments. The trace recorded `Counter Set: (null)` and `Shader Timeline: Disabled`, so it provided lifecycle/execution evidence only, not GPU counters, shader timings, or a kernel speedup.

## Hardware

- M3 MacBook Air, 16 GB: development machine
- M1 Mac mini, 8 GB: memory-constrained test and worker node
- M4 Mac mini, 16 GB: compute and benchmark node

## Latest measurement block (2026-08-27)

- `test-backend-ops` parameter filtering was inspected. `-p` filters the test-case `vars()` string, while `-o` requires the complete `op_desc(vars)` string for a specific case.
- The attempted q4_K filters did not match any generated performance case and produced CSV headers only. They are non-measurements and must not be used for optimization decisions.
- The unfiltered `MUL_MAT` run was stopped because it is not bounded enough for this measurement block; no isolated q4_K result is available.
- The extended Xcode Metal trace confirms `metal-shader-profiler-intervals` and `metal-application-encoders-list` tables exist, but recording settings show `Counter Set: (null)` and `Shader Timeline: Disabled`. It is therefore useful for trace provenance and encoder presence, not kernel timing or GPU-counter analysis.

## Decision after measurement review (2026-08-27)

- No source change is applied in this iteration. The measurements identify dispatch paths, but do not isolate a kernel bottleneck or prove that a threshold, threadgroup size, or Metal kernel variant is suboptimal.
- The q4_K `n=1` case uses the extended `mul_mv` path. Small-batch cases use the extended `mul_mv` path, while larger cases compile and use the regular `mul_mm` or large-batch paths.
- A source change without GPU counters, shader timing, or a controlled A/B benchmark would be speculative. The current implementation remains the reference candidate.

## Infrastructure block (2026-08-27)

- Added `common/apple-runtime.{h,cpp}` with caller-driven system snapshots and explicit GPU, unified RAM, local SSD, remote RAM, and remote SSD tier labels.
- The snapshot is bounded and synchronous; it starts no worker thread and is not connected to the inference path, preserving default performance and semantics.
- Apple builds use `sysctl` and Mach task/host statistics; non-Apple builds return safe zero/false defaults.
- Build validation passed for `llama-common` and `llama-cli`; `llama-cli --version` reports commit `deae5ee13`.
- Existing `test-batch-alloc` passed: 30 tests, 198 assertions, 0 failures.
- Extended the snapshot with active, wired, compressed, and peak process-resident memory metrics.
- The frozen baseline remains unchanged; no inference or Metal execution path was modified.
- Added standalone `common/node-runtime.{h,cpp}` capability reporting. It reuses the telemetry snapshot, reports local identity plus CPU, Apple Silicon, unified-memory, chip, and physical-memory fields, and provides compact JSON formatting.
- Extended node runtime with an in-memory versioned registration message, explicit registration state, and registration JSON formatting. It performs no I/O, starts no threads, and is not connected to inference or networking.
- Added an opt-in synchronous loopback TCP transport in `common/node-runtime.{h,cpp}`. It supports listen, accept, localhost-only connect, complete bounded sends, timed or blocking receives, move ownership, and close. It has no background thread, external dependency, or inference integration.
- Added caller-driven node lifecycle state with versioned heartbeats, configurable interval and timeout checks, timeout state transition, and explicit shutdown-requested/stopped transitions. Timestamps are supplied by callers; there are no background threads, timers, daemons, or inference changes.
- Validation for the node capability block: `llama-common`, `llama-cli`, and existing `test-batch-alloc` build/test succeeded; `git diff --check` passed. No new tests were added under `tests/`.

## Next steps

1. Measure the physical link and storage tiers before distributed benchmarks.
2. Keep power-state labels separate and do not compare unverified low-power results with normal-power results.

## Measurement block completed (2026-08-27)

- Corrected backend-ops filtering: `-p 'type_a=q4_K'` matches the actual `test_mul_mat::vars()` field and produced seven direct q4_K `MUL_MAT` rows plus related `MUL_MAT_ID` rows.
- Fresh AC-power q4_K baseline artifact: `benchmarks/baseline/2026-08-27-mul-q4k-perf-console-rerun.log`.
- Representative decode-like case (`m=4096,n=1,k=14336`): 403.01 us/run, 291.41 GFLOPS. Batched cases: n=2 856.78 us/run, n=5 3136.47 us/run; n=8 exercised `mul_mm` according to pipeline compilation.
- Full Metal backend correctness was rerun successfully: 14283/14283.
- Representative bounded runtime profile under AC: prompt 844.2 t/s, generation 139.3 t/s. It is not a replacement for the repeated baseline benchmark.
- No source or kernel changes were justified by this block. The q4_K measurements are now sufficient for a future single-variable optimization experiment.

## Artificial distributed milestone (2026-08-27)

- Added `common/node-runtime-selftest.cpp`, wired as the `node-runtime-selftest` executable from `common/CMakeLists.txt`.
- The bounded, synchronous self-test validates local registration state, localhost TCP listen/connect/accept, heartbeat exchange, timeout detection, and controlled shutdown transitions.
- The test exchanges only a heartbeat struct and a small payload. It moves no tensors, starts no threads, and never connects outside loopback.
- Added a disabled-by-default standalone local scheduler API for artificial tasks and capability candidates. It uses compute, memory, network, and storage cost inputs, rejects candidates without capacity, and selects deterministically by lowest cost then candidate ID. It has no ggml or inference integration.
- Validation target for this milestone: build `node-runtime-selftest` and `test-batch-alloc`, run both, and run `git diff --check`.
- Extended the self-test with a bounded coordinator/worker simulation over loopback TCP. It exchanges length-framed registration JSON, uses the artificial scheduler to select the worker, exchanges fixed synthetic task/result messages, and transitions both lifecycle objects through shutdown to stopped. No tensors, inference, threads, or unbounded waits are involved.
- Replaced self-test-local framing helpers with a reusable versioned bounded message API. Frames validate magic, protocol version, nonzero type, expected type, and a 4096-byte maximum before payload allocation; sends and receives use exact completion semantics. The self-test now uses frames for registration, synthetic task/result, heartbeat, and payload exchanges and rejects oversized and invalid outbound messages. Transport remains synchronous and loopback-only.
- Added bounded loopback fault coverage to the self-test for wrong protocol version, wrong expected message type, truncated/closed frame, and receive timeout. Existing lifecycle checks cover timeout, shutdown request, and stopped transitions without threads or unbounded waits.
- Added an experimental standalone scheduler plan report. It ranks all artificial candidates deterministically, places feasible candidates first, reports total plus compute/memory/network/storage costs, and leaves infeasible candidates visible. It has no default-runtime, ggml, inference, thread, or network side effects.
- Extended `node-runtime-selftest` to validate feasible ranking, infeasible candidates, cost components, and deterministic candidate-ID ties.
- Completed the real Qwen smoke gate with `models/baseline/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`: CPU produced `Baseline-ok`; Metal produced `baseline-ok`. Both exited normally with the expected semantic response. Captured Metal artifact: `benchmarks/baseline/2026-08-27-qwen-real-metal.log`.
- Added `--apple-telemetry [on|off]` / `LLAMA_ARG_APPLE_TELEMETRY` as an opt-in common CLI option. It reports the local snapshot from the existing parameter-info path; default remains off and inference behavior is unchanged.
- Qwen Metal with `--apple-telemetry on -lv 3` completed normally, emitted the telemetry and node capability report, and produced `baseline-ok`. Snapshot: Apple M3, 16 GiB physical memory, 1604 MiB free, 1337 MiB active, 494 MiB wired, 513 MiB compressed, 68 MiB RSS. The run measured 589.5 prompt t/s and 86.4 generation t/s; this is a correctness and telemetry-output smoke gate, not a replacement for the frozen repeated baseline.
- Added standalone `llama-node` with loopback-only registration acknowledgement, `--port`, `--once`, bounded accept, and controlled lifecycle shutdown. It has no inference or tensor integration.
- Node protocol smoke passed on port 49123: registration frame accepted, capability acknowledgement received, and process stopped cleanly. An earlier port-0 probe correctly exposed that the dynamically selected port must be published out-of-band; no result was counted from that failed probe.
- Final Qwen Metal telemetry regression passed: `Baseline-ok`, 632.9 prompt t/s, 93.5 generation t/s. Telemetry was visible and reported Apple M3, 16 GiB physical memory, 1545 MiB free, 1411 MiB active, 481 MiB wired, 510 MiB compressed, and 69 MiB RSS. Artifact: `benchmarks/baseline/2026-08-27-qwen-telemetry-metal-final.log`.
- Extended the opt-in telemetry report with the registered GGML backend device count and per-device name, description, and free/total memory. It uses the existing backend device enumeration and memory APIs only in the telemetry path; default inference behavior is unchanged.
- Added standalone `llama-node`, linked to `llama-common`. It binds loopback, accepts one bounded registration frame, replies with capability registration JSON as an acknowledgement frame, and performs caller-driven heartbeat/shutdown transitions. `--once` bounds the accept wait to five seconds; no inference, tensors, threads, or external network are involved.
- Added standalone `node-runtime-bench`, a bounded synchronous loopback benchmark for synthetic framed payloads. It accepts `[message_bytes] [message_count]`, defaults to 1024 bytes and 1000 exchanges, uses `steady_clock`, and reports average/min/max round-trip latency plus round-trip message and MiB throughput. It has no threads, tensor movement, inference changes, or external-network path.
- Loopback benchmark on the M3 with 256-byte payloads and 100 exchanges: average round-trip latency 0.130 ms, min 0.063 ms, max 0.194 ms, 7665.16 messages/s, 3.74 MiB/s. This is a local transport reference only, not a distributed-network result.
- Qwen Metal regression after backend capability reporting passed with telemetry visible: `baseline-ok`, 615.7 prompt t/s, 84.5 generation t/s, backend device count 3. Artifact: `benchmarks/baseline/2026-08-27-qwen-transport-regression.log`.
- Cluster remains standby; no M1/M4 connections or distributed benchmarks were run.
- Local-only Qwen Metal gate passed with telemetry visible and output `baseline-ok`: 711.3 prompt t/s and 106.9 generation t/s. Snapshot: 1548 MiB free, 1386 MiB active, 506 MiB wired, 508 MiB compressed, 68 MiB RSS. Artifact: `benchmarks/baseline/2026-08-27-qwen-local-final.log`.
- Local validation passed after the transport block: `node-runtime-selftest`, `node-runtime-bench 256 100` (0.146 ms average round-trip, 6867.36 messages/s, 3.35 MiB/s), `test-batch-alloc` (30 tests, 198 assertions), and `git diff --check`.

## Memory pressure block (2026-09-18)

- Extended `common/apple-runtime.{h,cpp}` with `memory_pressure_config`, `memory_pressure_level` (normal/warning/critical), and `memory_pressure_assessment`. `assess_memory_pressure` is a pure classification of one snapshot: wired or compressed ratio at or above a threshold raises the level, a free ratio below the floor raises critical directly. Defaults are conservative: wired 0.66/0.80, compressed 0.20/0.32, free floor 0.10. A zero-physical-memory snapshot classifies as normal.
- Added an experimental caller-driven `memory_budget` for planned reservations per `memory_tier` with explicit per-tier capacities, duplicate-label rejection, unknown-tier rejection, a bounded reservation table (default 16), and insertion-ordered listing. It allocates no real memory and is not consulted by inference.
- The opt-in telemetry report now prints one `memory_pressure` line with level and wired/compressed/free/resident ratios (INFO level, visible with `-lv 3`). Default inference behavior is unchanged.
- Extended `node-runtime-selftest` with deterministic pressure-threshold coverage (warning/critical on wired, compressed, and free, custom config, zero-memory snapshot) and full `memory_budget` coverage (reserve, duplicate, exhaust, release, unknown tier, table bound, insertion order).
- A first selftest run failed because the strict-config assertion inherited a low free-memory value from a previous scenario. Fixed by resetting the scenario state; the classification itself was correct.
- Validation: `node-runtime-selftest` PASS; `test-batch-alloc` passed (30 tests, 198 assertions, 0 failures); `node-runtime-bench 256 100` averaged 0.148 ms round-trip (previous 0.146 ms); `git diff --check` PASS; build clean for `llama-cli` and `node-runtime-selftest`.
- Real correctness/telemetry smoke on battery power (not a benchmark result): Qwen Q4_K_M on Metal produced the expected `pressure-ok` output. Telemetry reported `memory_pressure: level=critical free_ratio=0.088 wired_ratio=0.030 compressed_ratio=0.043`; the machine genuinely had little free memory at run time, so the critical classification is a true positive. Artifact: `benchmarks/baseline/2026-09-18-memory-pressure-smoke.log`.
- No default inference path was modified; no performance comparison is claimed from this block.

## Chunked transport block (2026-09-18)

- Extended the node message framing with experimental chunked transfer: `send_chunked`/`receive_chunked` emit one metadata frame plus ordered data frames and validate sequence, per-frame size, and total on receipt. Limits: chunk 1 MiB, total 256 MiB, chunk type 64/65 reserved.
- `send_message`/`receive_message` gained an optional per-call `max_payload` parameter; the default remains the original 4096-byte limit.
- Enabled TCP_NODELAY on transport sockets. Without it, many small frames interacted with Nagle and delayed ACK and produced multi-second stalls.
- Found and fixed a real single-thread deadlock in the benchmark: sending a payload larger than the socket buffers blocks in send while nobody drains the peer. The server leg of the chunked benchmark now runs on its own thread, matching the real deployment where the two sides are separate processes. The constraint is documented in the header; the selftest stays single-threaded with an 8 KiB payload.
- Also fixed an inverted zero-size assertion in the selftest (empty chunk transfer must produce an empty buffer).
- Benchmarks on loopback, M3, battery (single-run relative references only, not normal-power baselines): 1 MiB round trip with 64 KiB chunks = 2.13 GiB/s and with 1 MiB chunks = 2.58 GiB/s; 4096-byte frames = 174.35 MiB/s; 256-byte frames = 8.51 MiB/s. Before TCP_NODELAY the 1 MiB run showed 1.5-5 second stalls.
- Validation: `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS, clean build for bench, selftest, and `llama-cli`.

## Departure handshake block (2026-09-18)

- Added graceful node departure: new `departure_message_type` (frame type 6, payload = node_id), `node_registry::mark_departed` setting the entry to `stopped`, and coordinator dispatch that closes the link and records the clean stop instead of waiting for heartbeat timeout.
- Registry semantics: a stopped node stays stopped until re-registration returns it to running; heartbeats do not resurrect a stopped entry; `mark_departed` returns false for unknown node ids.
- The coordinator receive path now dispatches by frame type (heartbeat, departure, unexpected type closes the link as a protocol violation) instead of expecting only heartbeats.
- `node-runtime-client` sends the departure frame after its monitor window ends and prints `departure_sent node=<id>`.
- Selftest coverage: unknown-id rejection, stopped transition, heartbeat non-resurrection, re-registration recovery, and the wire departure frame round trip.
- End-to-end loopback smoke on the M3: coordinator `llama-node --port 49155 --monitor 5 --nodes 1` with one `node-runtime-client --monitor 3`; client sent 4 heartbeats, then `departure_sent`; coordinator logged `node ... departed cleanly` and the final registry JSON reported `state=2` (stopped) with the full capability record.
- Validation: `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS, clean build for selftest, `llama-node`, and `node-runtime-client`.

## Cleanup pass (2026-09-18)

- Removed the unused `registration` field from the coordinator `node_entry` struct (written, never read); the registry already holds the authoritative record.
- Deleted the two Instruments trace bundles (120 MB total) from `benchmarks/baseline/`; their conclusions are already distilled in this file and the TOC text remains. Baseline directory is now 3.9 MB.
- Revalidated after cleanup: `node-runtime-selftest` PASS, build clean for `llama-node`.
