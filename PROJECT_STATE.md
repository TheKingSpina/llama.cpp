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

## Expert catalog block (2026-09-18, branch research/moe-expert-sharding)

- New branch `research/moe-expert-sharding` cut from `research/apple-silicon-baseline` at `c8577a5bf` to explore the replicated-weights MoE design: model files live on every node, only the experts selected by the router are loaded, executed, and unloaded. No tensor movement over the network; hidden states stay per node.
- Added `common/expert-catalog.{h,cpp}` on the same standalone pattern as `node-runtime`: reads GGUF metadata only (`no_alloc`), never tensor data, starts no threads, and has no inference integration.
- `expert_catalog::load_from_gguf` scans the tensor directory for fused expert tensors with the canonical names from `src/llama-arch.cpp`: `blk.N.ffn_gate_up_exps`, `blk.N.ffn_down_exps` (fused variants), `blk.N.ffn_gate_up_ff` / `blk.N.ffn_down_ff` (merged variants), and the separate `blk.N.ffn_gate_exps` / `blk.N.ffn_up_exps` / `blk.N.ffn_down_exps` used by Qwen3-MoE. Expert count comes from the 3D tensor shape (`ne[2]`) first, then from `<arch>.expert_count` metadata; disagreeing layers are skipped, not guessed. Dense and attention tensors are ignored.
- Each layer entry reports per-expert slice bytes per tensor plus full tensor totals; the summary reports layer count, expert count, and total expert bytes. Compact JSON report for diagnostics and future registration messages.
- Added the experimental `expert_placement_table`: caller-driven expert-to-node assignments with bounds validation, re-assignment replacing (not duplicating), unassign, per-node totals in first-seen order, bounded entry count, and a compact JSON report. Bookkeeping only; it moves nothing.
- Added `common/expert-catalog-selftest.cpp` as `expert-catalog-selftest`. It writes synthetic GGUF files with the gguf writer (real contexts and tensors, metadata-only output), then covers: separate gate/up/down MoE layout, fused-shape parsing, metadata expert-count fallback, mismatched-layer rejection, missing-file failure, dense-tensor exclusion, and all placement-table behaviors. Temporary files are removed on exit.
- Known limitation, deliberate for this block: the GGUF stores experts fused per tensor (`[n_embd, n_ff, n_expert]`), so per-expert granularity here is bookkeeping on slices, not independent tensors. Selective loading of a subset of experts requires the loader change planned for the next block.
- Not yet measured on a real MoE GGUF: no local MoE model file exists (only the dense Qwen2.5-0.5B baseline). Download a small MoE (for example Qwen3-30B-A3B or smaller) and rerun the catalog read as a real-file check.
- Validation: `expert-catalog-selftest` PASS, `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS, clean build for `llama-cli`.

## Selective expert reader block (2026-09-18, branch research/moe-expert-sharding)

- `expert_catalog::load_from_gguf` now records, for each expert tensor, the absolute file offset of the expert-0 plane (data offset + tensor offset). Each `expert_layer` exposes the per-tensor plane offsets.
- Added `expert_catalog::expert_reader`: opens the file and reads one expert's per-tensor planes at `data offset + tensor offset + slice x expert index`. Synchronous, no threads, no cache, no inference integration. Empty tensors (2D shapes without an expert axis) produce empty vectors; out-of-bounds experts and unknown layers are rejected.
- Real-file verification on OLMoE-1B-7B q8_0 (6.86 GiB, `models/moe/`): catalog reports 16 MoE layers, 64 experts per layer, 6.38 MiB per expert, 6.38 GiB of expert weights total. A full top-1 pass (16 experts, 102 MiB) reads in 23.7 ms on the M3, battery, from SSD; warm repeated reads reach 0.79 ms per expert (8.0 GiB/s, page cache). Reading a 1 GiB working set (160 experts) takes 426 ms cold-ish, 2.66 ms per expert.
- Selftest now writes real tensor data into the synthetic GGUF (distinct fill per tensor) and covers the reader: byte-exact slice reads, adjacent-slice difference, out-of-bounds expert, unknown layer, and unopened-reader failure.
- Two real-file bugs fixed in the same block and committed as `270f2b33` (recorded above): the `.weight` name component and the quantized slice arithmetic.
- The synthetic GGUF fill is now non-periodic so two 512-byte expert planes in one tensor are never byte-identical; the adjacent-slice-differ assertion is real coverage, not a tautology.
- Validation: `expert-catalog-selftest` PASS, `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS.

## LRU expert cache block (2026-09-18, branch research/moe-expert-sharding)

- Added `expert_cache` over one bound `expert_reader`: byte-bounded LRU with an entry bound. `serve` returns resident planes on hit (LRU refresh) and reads plus evicts on miss. Oversized experts and zero-capacity caches are served from a scratch buffer without being cached; failure paths propagate as miss plus false. Cumulative statistics survive `clear()`.
- Selftest covers deterministic eviction order, exact byte accounting (loaded versus served), the hit-rate invariant, the clear path, the bypass path, the oversized path, and failure propagation.
- Real-file trace on OLMoE-1B-7B q8_0 (deterministic LCG trace, seed fixed, 1000 requests, 80 percent from a 40-expert hot set spread over all 16 layers, 20 percent uniform over the 1024-expert key space; M3, battery):
  - 64 MiB cache (10 experts): 19.5 percent hit rate, 795 evictions, 5.13 GiB loaded from SSD, 0.75-0.84 ms per request.
  - 320 MiB cache (50 experts): 84.7 percent hit rate, 103 evictions, 975 MiB loaded, 0.14 ms per request.
- Read of the results: the hot set (40 experts, 255 MiB) barely fits in 320 MiB, so LRU mostly holds it and the hit rate approaches the hot-share bound; at 64 MiB the ratio 10 hot experts over 40 predicts about 20 percent, which matches the measured 19.5 percent. Cold requests still evict hot entries under plain LRU; a hit-rate-driven or hot/cold-segmented policy is the recorded candidate improvement.
- The per-request cost gap (0.75 ms versus 0.14 ms) is the measurable cache payoff; a top-1 decode pass touches 16 experts (one per layer), so a cache that holds the hot set turns a 24 ms I/O pass into a few ms.
- Validation: `expert-catalog-selftest` PASS, `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS.

## Expert forward bit-exactness gate (2026-09-18, branch research/moe-expert-sharding)

- Added `common/expert-forward-check.cpp` as the `expert-forward-check` executable, the correctness gate for the selective-loading design: it proves that expert weights rebuilt plane by plane from `expert_reader` slice reads produce the same output as the full fused 3D tensors through the real `ggml_mul_mat_id` graph.
- Setup: synthetic GGUF with one MoE layer, separate gate/up/down expert tensors (F32, 32x24 and 24x32 planes, 4 experts), deterministic fill. The graph is `gate = gate_exps[:,:,id] @ b`, `up = up_exps[:,:,id] @ b`, `act = swiglu(gate, up)`, `out = down_exps[:,:,id] @ act` with two tokens, two experts each, all four experts participating. b is [n_embd, n_used, n_tokens] reshaped from the token input, matching the `mul_mat_id` contract.
- The selective path reads experts 3, 1, 0, 2 in shuffled order and rebuilds the 3D tensors at `e * plane_bytes` offsets. Result: output is bit-exact (128 of 128 elements equal) against the full-tensor path.
- The gate is self-checking: one corrupted byte in an expert plane used by a token must change the output; the tool fails if corruption goes undetected.
- This establishes the invariant the future cache-to-inference integration must preserve: slices placed at `e * expert_bytes` in a 3D tensor compute identically to the fused layout.
- Validation: `expert-forward-check` PASS (bit-exact), `expert-catalog-selftest` PASS, `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS, `llama-cli` builds clean.

## Cache-backed MoE runner block (2026-09-18, branch research/moe-expert-sharding)

- `expert_cache` now binds its catalog at construction; `serve` needs no catalog argument (three-argument form). Selftest updated.
- Added `expert_moe_runner`: gathers the experts named by router ids through the cache. Duplicates count once (ordered unique); gathered planes are owned copies because a later serve in the same gather may evict an earlier one and dangling pointers were a real observed bug (bit-exact failure on the first run, fixed by deep copy at serve time).
- `expert-forward-check` extends to three paths: full-tensor reference, reader-slice rebuild, and cache-backed gather with eviction pressure (capacity two experts, gather four, exactly two evictions). All three outputs are bit-exact against each other through the real `ggml_mul_mat_id` graph.
- The cache path is the integration blueprint for a real MoE layer: per token, gather the routed experts from cache, place planes at `e * expert_bytes` in a 3D tensor, compute, drop. Correctness invariant holds under eviction.
- Validation: `expert-forward-check` PASS (bit-exact on all three paths, 2 evictions), `expert-catalog-selftest` PASS, `node-runtime-selftest` PASS, `test-batch-alloc` 30/198/0 PASS, `git diff --check` PASS, `llama-cli` builds clean.

## M4 mini transport baseline block (2026-09-23, Role B)

- First setup of the M4 mini (16 GB, AC power) per RUNBOOK.md Role B: branch `research/moe-expert-sharding` at `c9ae4b931`, cmake 4.4.3 installed via pip (no prior cmake on the machine), configured with GGML_METAL=ON, built expert-catalog-selftest, node-runtime-selftest, test-batch-alloc, node-runtime-bench.
- Validation gate PASS: expert-catalog-selftest exit 0, node-runtime-selftest exit 0, test-batch-alloc failures 0; `git diff --check` PASS.
- Loopback transport baseline (M4 mini, AC, 2026-09-23, artifact `benchmarks/baseline/2026-09-23-m4-loopback.log`): 1024-byte frames x 2000 exchanges, average RTT 0.014 ms (min 0.007, max 0.054), 70460.26 messages/s, 137.62 MiB/s. Local reference for the M4; not comparable with LAN numbers.
- Negotiated media recorded: 1000baseT full-duplex on en0, MTU 1500.
- Physical-link battery (1 KiB / 64 KiB / 1 MiB x 20 rounds between the two minis) not yet run: requires both minis on the switch simultaneously.

## Physical-link battery attempt (2026-09-23, M4 mini)

- LAN sweep of 192.168.178.0/24 (60 addresses pinged): no M1 mini found. Reachable hosts: M4 (minidiaessandro, .51), M3 Air (airdialessandro, .59), Raspberry Pi (.27), router/hub, and two other devices. `mac.wind3.hub` (.34) responds to ping but refuses port 22 (Remote Login off); mDNS shows no other Mac advertising SSH.
- Conclusion: the M1 mini is powered off or disconnected from the switch. The physical-link battery (1 KiB / 64 KiB / 1 MiB x 20 rounds between the two minis) is blocked on that machine being available, per RUNBOOK Role B step 3.
- Recorded as a finding, not a failure. No LAN numbers were produced, so no LAN claim is made. Loopback M4 baseline from the previous block stands as the only transport reference for the M4.
- Next step unchanged: when the M1 mini is on the switch (and SSH reachable, or the user runs the commands locally there), run the three-size battery and append raw numbers to benchmarks/baseline/ in a dated file.

## LAN sweep refinement (2026-09-23, M4 mini, follow-up)

- Full subnet sweep .1-.254 (after user noted a mini does not sleep like a laptop): 14 hosts identified. The M1 mini is definitively NOT on the network, not a sleep/power-saving artifact. Found: M3 Air (.93 via DHCP on the Air, not .59 as ARP first showed), MBP (.170), iPhone (.198), iPad (.228), ESP32 (.176), APs (.30, .188), router (.1), Pi (.27), unknown printer-like host (.250), and this M4 (.51).
- Correction to the previous block: `airdialessandro.wind3.hub` resolved to .59 in ARP cache but reverse DNS now gives .93 for `MacBookAir.wind3.hub` — the Air's DHCP lease may have moved; treat .59 as stale cache.
- Conclusion stands: physical-link battery remains blocked on the M1 mini being physically connected. User informed; the machine likely needs manual power/cable attention.

## M1 node reconnaissance (2026-08-30, recovered 2026-10-06)

- Recovered from the M1 machine, where these blocks had been an uncommitted local edit with no copy in git. Preserved verbatim; see the recovery note in the 2026-10-06 blocks below.
- Host: Macmini9,1, Apple M1, 8 GiB physical memory, 8 CPUs, arm64.
- OS: macOS 26.0, Darwin 25.0, build 25A353.
- Repository: branch `research/apple-silicon-baseline`, clean worktree, commit `ddb7a83e1` (`feat(apple): add opt-in runtime and node infrastructure`).
- Active default route: Ethernet `en0`, IP `192.168.178.70`, gateway `192.168.178.1`, MTU 1500. The physical link speed and switch path were not yet measured at that time.
- Thunderbolt Bridge `bridge0` and Ethernet adapters `en4`/`en5` are present, but no active IP/link path was established for them during reconnaissance.
- Root volume: 228 GiB total, 106 GiB available. This is above the 30 GiB safety floor; no model or cache data was written.
- Toolchain gap: `xcode-select` points to `/Library/Developer/CommandLineTools`; `cmake` and `ninja` are not currently available in PATH on this node.

## M1 build validation (2026-08-30, recovered 2026-10-06)

- Installed Homebrew `cmake 4.4.3` and `ninja 1.13.2` under `/opt/homebrew`; no sudo was required.
- Configured isolated `build-m1/` Release build with Metal, native ARM tuning, tests, examples, and server enabled.
- Build completed for `llama-cli`, `llama-bench`, `test-batch-alloc`, and `node-runtime-selftest` using `-j2`.
- `llama-cli --version` confirms commit `ddb7a83e1`, AppleClang 17.0.0.17000319, Darwin arm64.
- `node-runtime-selftest` passed.
- `test-batch-alloc` passed: 30 tests, 198 assertions, 0 failures.
- Configuration warnings: OpenMP and ccache are unavailable. Accelerate and Metal were detected. No inference benchmark was run yet on M1.

## M1 model staging (2026-08-30, recovered 2026-10-06)

- Downloaded `models/baseline/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` from the public `lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF` repository.
- Size: 379 MiB. SHA-256: `fa4d41b65761ed565cac6b5f62e35135d050408b033114a128ab308c02b2e83a`.
- The checksum and size differ from the M3 baseline artifact (469 MB, SHA-256 `74a4da8c...`). Treat this as a staged M1 model until metadata and model identity are compared; do not compare benchmark numbers across the two files as if they were identical.
- Internal free space after download: 105 GiB. The failed initial authenticated URL attempt left no model file; only a small Hugging Face cache directory remains.

## Correction of the M1 availability finding (2026-10-06, M4 mini)

- The two blocks above are wrong on the central point: the M1 mini was on the LAN the whole time. A passive mDNS service browse (`dns-sd -B _ssh._tcp local`) found `Mac mini di Alessandro (2)` immediately, and it resolves to `192.168.178.70`. The 2026-09-23 identification method (ping plus reverse DNS) missed it because the host had no reverse-DNS name; it appeared in the ARP cache only as the unnamed entry for `.70`. Discovery by mDNS is the correct method for a machine that is always on and advertises SSH.
- The user states the two minis are on the same wired network with a router in between. The measurement does not support the router hop for this pair: `route -n get 192.168.178.70` on the M4 returns a direct host route on en0 with `LLINFO`, no gateway. So M4 and M1 are L2 adjacent on the same `192.168.178.0/24`. Any router in the topology is not on this path.
- Verified M1 identity over SSH: `Mac-mini-di-Alessandro-2.local`, Apple M1, 8 GB, 8 CPUs, macOS 26.0, on AC power. Negotiated media on the M1: 1000baseT full-duplex flow-control, MTU 1500, matching the M4.
- The M1 also moved from `192.168.178.51`-era assumptions: the M4 is `.50`, the Air `.93`. Addresses in the 2026-09-23 block (`.51`, `.59`) are stale.
- Method finding, kept for the record: this block was produced with an agent-driven SSH session using an `SSH_ASKPASS` helper and a user-supplied password. No key was installed and no credential file was written.

## LAN Role A smoke (2026-10-06, M4 mini coordinator, M1 mini worker)

- First real M4/M1 connection; the state file previously said the cluster was on standby with no connections run.
- M1 brought onto `research/moe-expert-sharding` @ `c9ae4b931` (it was on `research/apple-silicon-baseline` @ `86e2b4891` with binaries from 2026-08-31 and none of the departure-handshake code). Its uncommitted local `PROJECT_STATE.md` edit was first saved to `benchmarks/baseline/2026-10-06-m1-local-project-state.patch` (3397 bytes), then the worktree was cleaned and the branch checked out. The patch is untracked on the M1 and is the only copy of those 2026-08-30 M1 blocks.
- M1 build note: `cmake` is installed under `/opt/homebrew/bin` and is absent from the non-interactive SSH PATH, so remote builds need an explicit PATH. The M1 has two build dirs; `build-m1/` is the RUNBOOK-configured one and was used.
- Validation gate PASS on both machines: `node-runtime-selftest` exit 0 (M1 via `build-m1/`, M4 via `build/`).
- Coordinator on the M4 needed `--bind 192.168.178.50`; the default bind is `127.0.0.1` and would not have been reachable from the M1.
- Result: registration JSON with `node_id` `Mac-mini-di-Alessandro-2.local`, 5 heartbeats at 5 s, `departure_sent` from the client, coordinator logged `departed cleanly`, `state=2` (stopped), `reconnections=0`, final registry JSON complete. Clean loop over the real LAN.

## LAN transport battery (2026-10-06, M4 mini to M1 mini, 20 rounds per size)

- Blocked before this block: `node-runtime-bench` hardcoded `client.connect("127.0.0.1", ...)`, so the RUNBOOK Role B step 3 instruction to run it over the LAN could not be executed. The bench gained `--listen [--bind ADDRESS] PORT [message_count] [chunk_bytes]` and `--connect HOST PORT [message_bytes] [message_count] [chunk_bytes]`. The loopback default path is unchanged and was checked against the existing baseline.
- Design constraints found while doing this, recorded because they are not obvious:
  - `chunk_bytes` is the size the sender uses for data frames and the receiver validates against it (`receive_chunked` rejects a peer whose `meta.chunk_bytes` exceeds the limit passed in). Client and server must therefore be given the same value by hand; the bench does not negotiate it.
  - The server does not pick the payload size, the client does. The server echoes whatever it receives.
  - The listening side needs an explicit bind address to be reachable at all, since the default is loopback.
  - `--listen` recycles its listener and serves one connection at a time, so a single server process covers a whole battery. This was verified with three clients in sequence.
- LAN numbers, artifact `benchmarks/baseline/2026-10-06-m4-m1-lan-battery.log`, both machines on AC power, M4 as measuring client, M1 as echo server:
  - 1 KiB, framed path: RTT avg 0.621 ms, min 0.525, max 1.160, 3.14 MiB/s.
  - 64 KiB, chunked with chunk 65536: RTT avg 2.240 ms, min 1.996, max 3.965, 55.81 MiB/s.
  - 1 MiB, chunked with chunk 1048576: RTT avg 18.820 ms, min 18.438, max 20.916, 106.27 MiB/s.
- Reading: the small-frame RTT is roughly 45x the M4 loopback reference (0.014 ms at 1024 bytes) and the 1 MiB round trip settles near 106 MiB/s, far below the loopback 2.1-3.6 GiB/s range. These are single 20-round runs, taken once, with no repetition and no power or thermal guard beyond both machines being on AC. They are a first LAN reference, not a characterized link, and no claim is made about the switch or the router.
- No inference path, ggml graph, or Metal kernel was touched. The only code change is the bench tool.
- Not committed: no `git commit` and no push were run on either machine.

## Recovery of the M1 local state file (2026-10-06, M4 mini)

- The three M1 blocks above (reconnaissance, build validation, model staging, all 2026-08-30) existed only as an uncommitted edit on the M1, saved before the branch realignment as `benchmarks/baseline/2026-10-06-m1-local-project-state.patch` (3397 bytes, sha256 `8decebbf1989c6e79af270b4f9ebc298f5db4808b3f86062ecceb14d01352e9b`). They are now in this file, so the only copy is no longer an untracked local file. The patch file is still on the M1 and was not deleted.

## Real MoE model acquisition (2026-10-06, M4 mini)

- Acquired `models/moe/Qwen1.5-MoE-A2.7B_q4_k_m.gguf` from `gdax/Qwen1.5-MoE-A2.7B_gguf`. Size 9490338464 bytes, SHA-256 `09555491f85f40208e3999f3ced97b064f7c7c0b88bb3fedcbd5df51aad7d1ed`, matching the Hugging Face LFS oid exactly.
- Chosen over the smaller-quant alternatives because the architecture is one the project already supports (`qwen2moe`). Note the real size is 9.49 GB, not the ~3 GB estimated when the choice was proposed: Qwen1.5-MoE-A2.7B has 14.3B total parameters and only 2.7B active, so every expert is stored even though few are used per token. Quantization level is irrelevant to the catalog, reader, and cache checks, which only read bytes.
- First download attempt stalled at 5.74 GB with an empty error log, caused by the background process dying with its parent shell rather than by a network error. Restarted with `nohup` plus `disown` and resumed with `curl -C -`; checksum then matched.

## Real-file catalog and reader verification (2026-10-06, Qwen1.5-MoE-A2.7B q4_k_m)

- `expert_catalog::load_from_gguf` recognizes the real file: architecture `qwen2moe`, 24 MoE layers, 60 experts per layer, 8304721920 bytes of expert weights total. This closes the "not yet measured on a real MoE GGUF" gap recorded in the expert catalog block.
- The layout exercised is the separate gate/up/down one (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), not the fused variant: the `gate_up` plane is empty and `down`+`gate`+`up` sum to `expert_bytes`. Two `down` slice sizes appear across layers, 3063808 and 1982464 bytes, so both shapes were checked.
- Verified tools were built as throwaway binaries in `build/bin` and `build-m1/bin`, not added to the repository and not wired into any CMake target.

## Real-file byte-exact gate (2026-10-06)

- Added a temporary verifier that compares the selective path against the full path independently: each expert tensor is read as one contiguous block from the file, and every expert plane returned by `expert_reader` is compared with the matching region of that block. The full read does not go through `expert_reader`, so an error in the catalog offset cannot hide itself.
- Layer 0 on the M4: gate, up, and down each compared across all 60 experts, 180 planes, 0 mismatches.
- Layers 3, 12, and 23 on the M4, covering both `down` slice sizes: 180 planes each, 0 mismatches. Total on the M4: 720 planes, 0 mismatches.
- Layers 0 and 3 on the M1: 180 planes each, 0 mismatches.
- Conclusion, scope: the catalog offsets and the selective reader are byte-exact on this real Q4 file, across both slice shapes and on both machines. This proves the offset arithmetic and the read path only. It does not yet prove that weights rebuilt plane by plane produce the same forward result through `ggml_mul_mat_id` on real quantized data, which is the stronger gate and remains open.
- No inference path was touched. No file under `tests/` was added. Nothing committed or pushed.

## MoE model replication M4 to M1 (2026-10-06)

- The design on this branch is replicated-weights: the model file lives on every node, each node loads only the router-selected experts, and no expert weights travel over the network. The RUNBOOK Role C already stated the file is on the M3 only and must be replicated per node, so replication is a manual per-node step by design, not a missing feature.
- Replicated the Qwen model by copying from the M4 over the LAN rather than re-downloading from Hugging Face on the M1. Destination directory had to be created first and the `scp` destination path had to be absolute: `scp` resolves a relative destination against the login home, not the repository, and the first two attempts failed on that.
- Result: 9490338464 bytes on the M1, SHA-256 identical to the M4 copy. `real-catalog` on the M1 returned the same numbers as the M4, and the first 16 bytes of expert 0 and expert 1 `down` planes were byte for byte identical to the M4 read. The two nodes are verified equivalent for this file.
- Sustained throughput observed during the copy: 9490338464 bytes in roughly 82 s, about 116 MB/s. This is consistent with the 106 MiB/s measured by the 1 MiB transport battery, so the link behaves the same for bulk transfer and for round trips. Practical consequence: replicating a 7 to 9 GB model between these nodes takes one to two minutes, which is what makes replicated-weights viable on an 8 GB M1.

## Open items after the 2026-10-06 blocks

- The stronger real-file gate is closed. A forward `ggml_mul_mat_id` bit-exactness check on real quantized expert planes passes on both nodes; see the 2026-10-06 forward gate block for coverage and limits. `expert-forward-check` itself still generates only its own synthetic F32 file and was left unchanged, so promoting the real-file check into that permanent tool is remaining work.
- `common/node-runtime-bench.cpp` is modified but uncommitted on both the M4 and the M1, with identical content. If either machine is realigned with origin the change is lost. The commit message is the user's to write.
- SSH to the M1 uses an `SSH_ASKPASS` helper with a user-supplied password and no installed key. Installing the user's public key on the M1 would remove the password from the command path.
- The measured bottleneck is the expert cache policy, not correctness. Decode is I/O bound at roughly 80 to 320 ms per token with near-zero hit rate, so the next step with real value is a hit-rate-driven or hot/cold segmented policy measured against this model. The routing traces used so far are synthetic and say nothing about real routing locality.
- No commit and no push were performed in any of these blocks.

## Selective-load viability measurement (2026-10-06, M4 mini, real Q4 model)

- Question: a full load of the 9.49 GB model does not fit the 8 GB M1. Does the replicated-weights selective-load design actually make the model runnable, or does it only move the problem?
- Model metadata that drives the answer: `qwen2moe.expert_used_count = 4`, `expert_count = 60`, 24 MoE layers, so a decode step touches 4 experts per layer. `expert_count` and `expert_used_count` were read from GGUF metadata, not assumed.
- Arithmetic check: one decode step of top-4 across all 24 layers is about 0.51 GiB of expert weights, which fits the M1 with room to spare. This is the per-token figure and it turned out to be misleading, see below.
- Measured with `expert_moe_runner` over `expert_cache` against the real file, serving real planes and consuming them so evictions happen:
  - Worst case, routing changes every token: 8 tokens, 24 layers, top-4, 768 planes served, 4.12 GiB served and all of it re-read, hit rate 0.0%, about 317 ms per token of I/O with both a 512 MiB and a 2048 MiB cache. Evictions 675 and 395 respectively.
  - Hot-set profile, 80 percent of requests drawn from a 12-expert set per layer, 32 tokens: hit rate still 0.0 percent at 512 MiB and 1.7 percent at 1024 MiB, about 106 and 80 ms per token, 16 GiB loaded.
- Reading, stated plainly: the design does reduce the memory peak, because a node loads a few experts at a time instead of the whole 9.49 GB file, and that is what makes the M1 a candidate at all. It does not make decode faster. Decode stays I/O bound at roughly 80 to 320 ms per token depending on the working set, because the memory saving does not translate into speed without routing locality. A hot set of 12 experts per layer across 24 layers is about 1.5 GB of distinct experts, which does not fit the 512 MiB cache tested, so hit rate stayed near zero.
- This is a finding, not a failure. It does not invalidate the design, it sets the expectation: the selective-load path is a memory-fit mechanism, and a hit-rate-driven or hot/cold segmented policy remains the recorded candidate improvement from the LRU block. The routing locality of a real tokenizer on real text was not measured; the traces here are synthetic routing and must not be read as a prediction of real hit rates.
- The measurement tools were throwaway binaries in `build/bin`, built and then deleted. Nothing was added to the repository or to `tests/`.

## Real-file forward bit-exact gate (2026-10-06)

- This closes the gate left open by the previous block. The property that matters for production is now proven on real quantized data, not only on synthetic F32: weights rebuilt plane by plane from the selective reader compute the same result as the full tensors through the real `ggml_mul_mat_id` graph.
- Graph, one MoE layer, same shape as the synthetic gate: `gate = gate_exps[:,:,id] @ b`, `up = up_exps[:,:,id] @ b`, `act = swiglu_split(gate, up)`, `out = down_exps[:,:,id] @ act`, with `b` reshaped to `[n_embd, n_used, n_tokens]` as `mul_mat_id` requires. Two token slots, both routed to expert 0.
- Real shapes and mixed quantization from the file: `gate` and `up` are `q4_K` with ne `[2048, 1408, 60]`, `down` is `q8_0` with ne `[1408, 2048, 60]`. This is the mixed-type case the synthetic F32 gate could never exercise.
- Alignment check performed before building the gate: every expert plane is an exact multiple of its block size (gate and up: 11264 blocks of the q4_K type per plane, 1622016 bytes; down: 90112 blocks, 3063808 bytes), so placing planes at `e * expert_bytes` never splits a quantization block. Rows are also block aligned (2048 elements is 8 blocks), which is what makes the per-row quant layout valid.
- Result on the M4: 4096 output elements, 0 differing, bit-exact between the full contiguous read and the selectively rebuilt tensors. Same result on the M1. Both PASS.
- Scope and limits, stated plainly: this covers one layer, one routed expert, two token slots, and CPU execution. It does not cover all 24 layers, a batch of tokens, the Metal backend, or more than one expert participating per token. Bit-exactness here means identical inputs to `mul_mat_id` produce identical outputs, which is the invariant selective loading must preserve; it is not a claim about numerical equivalence between quantized and unexpertised inference.
- The tool was a throwaway binary, built and deleted. Nothing added to `tests/` or to the repository. No inference path was modified. Nothing committed or pushed.
