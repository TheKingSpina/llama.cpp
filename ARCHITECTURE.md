# llama.cpp Architecture Reconnaissance

Status: Phase 0 reconnaissance. This document describes the current repository at commit `deae5ee133a3c4c56fbd46c17c8c2103af3bd643`.

## Apple Silicon acceleration policy

The experimental direction is Apple-Silicon-first, with native Metal as the primary execution target and MPS-compatible design principles for broader Mac support. New components should favor unified-memory locality, asynchronous execution, low synchronization overhead, reusable allocations, and layouts that avoid unnecessary CPU/GPU copies. MPS compatibility must be validated independently; it does not replace native Metal benchmarks.

Experimental paths remain opt-in and must preserve CPU and non-Apple fallbacks until correctness and performance are demonstrated.

The first shared infrastructure block is `common/apple-runtime.{h,cpp}`. It provides a bounded, caller-driven system snapshot and explicit memory-tier labels. It has no background thread, allocation pool, or inference-path hook, so the default runtime remains unchanged. Tier labels keep local unified RAM, GPU-visible memory, SSD, and future remote tiers distinct.

The same module classifies one snapshot into normal, warning, and critical memory-pressure levels from wired, compressed, and free fractions of physical memory. Thresholds are caller-supplied with conservative defaults and the classification is a pure function of the snapshot it receives. An experimental caller-driven `memory_budget` records planned reservations per memory tier with explicit capacities, duplicate-label rejection, and a bounded reservation count. This is bookkeeping for the future unified-memory manager: it allocates nothing and is not consulted by inference.

The opt-in node capability block is `common/node-runtime.{h,cpp}`. It builds a bounded local identity and hardware/memory snapshot from the existing telemetry API and formats it as compact JSON. It also exposes an in-memory versioned registration message and explicit registration state. The same module now provides a small synchronous TCP transport for localhost experiments. It has explicit listen, accept, connect, bounded send/receive, and close operations, starts no threads, and is not connected to inference.

When `--apple-telemetry on` is used, `common_params_print_info` also reports the registered GGML backend device count and each device's name, description, and free/total memory through the backend device APIs. This is caller-driven reporting only; it does not initialize devices beyond the existing parameter-info behavior and is not used by inference.

The node module also provides a caller-driven `node_lifecycle` state machine. It creates versioned heartbeat messages, tracks heartbeat intervals, detects bounded timeout, and exposes explicit shutdown-requested and stopped states. Callers supply timestamps and invoke the methods; no background thread, timer, daemon, or inference integration is introduced.

Transport framing now supports experimental chunked transfer for payloads larger than one frame: one metadata frame followed by ordered data frames with sequence, size, and total validation on receipt. The peer must drain the stream concurrently, because a single-threaded send-then-receive over one connection deadlocks once bytes in flight exceed the socket buffers. Sockets enable TCP_NODELAY so many small frames do not interact with Nagle and delayed ACK.

The experimental local scheduler is also standalone in `common/node-runtime`. It scores artificial tasks using compute work, memory feasibility, and network/storage penalties, then selects the lowest-cost candidate with deterministic candidate-ID tie breaking. It does not inspect or schedule ggml tensors and is disabled by default; the self-test covers only this artificial API.

`common/node-runtime-selftest.cpp` provides an artificial loopback milestone. It registers the local capability message, opens a localhost-only TCP listener, connects and accepts synchronously, exchanges a heartbeat and a payload, and checks timeout plus shutdown transitions. It does not move tensors, use background threads, or access non-loopback hosts.

The self-test now simulates one bounded coordinator/worker exchange. The worker sends its registration JSON, the coordinator uses the experimental scheduler to select the worker, then sends a fixed synthetic task and receives a fixed synthetic result. Both caller-driven lifecycle objects are transitioned to stopped before transport cleanup. This is a protocol/lifecycle exercise only: it has no tensor movement, inference integration, background thread, or unbounded wait.

The loopback protocol now uses a reusable bounded frame API. Each frame has a magic value, protocol version, nonzero message type, and 32-bit payload length. Send completes the full header and payload or fails; receive reads the exact frame, validates all header fields and the 4096-byte payload limit before allocation, and can require a specific message type. Invalid and oversized outbound cases are covered by the self-test. This remains synchronous, loopback-only, and tensor-free.

The node self-test also covers bounded protocol faults: an unsupported version, an unexpected message type, a truncated frame, and a receive timeout. These cases must fail without allocation or an unbounded wait. Lifecycle tests cover timeout, shutdown request, and stopped transitions for both an active and an immediately shutting-down node.

`llama-node` is a minimal standalone node process for protocol smoke tests. It binds
to loopback, accepts one registration frame, replies with local capability JSON in
an acknowledgement frame, and transitions through controlled shutdown. `--once`
limits the accept wait to five seconds; the normal mode is also bounded to thirty
seconds. It has no inference, tensor, worker-thread, or external-network path.

The scheduler exposes an experimental plan report for artificial tasks. It returns every candidate in deterministic order, with feasible candidates ranked before infeasible candidates, then by total cost and candidate ID. Feasible reports include compute, memory, network, and storage cost components. The API is disabled by default, has no threads or network side effects, and does not inspect ggml or inference state.

`node-runtime-bench` is a bounded standalone loopback transport benchmark. It accepts message bytes and message count (defaults 1024 and 1000, with the existing 4096-byte frame limit), exchanges synthetic framed payloads synchronously, and reports round-trip latency plus throughput using `std::chrono::steady_clock`. It uses no background threads, external hosts, tensors, or inference paths. Its results measure transport/protocol overhead only and are not distributed-compute results.

## Existing architecture

The public API is defined in `include/llama.h`. The main inference implementation is in `src/`, with common CLI/server support in `common/` and `tools/`.

The runtime flow is:

1. CLI or server parses parameters.
2. `llama_backend_init()` initializes the backend layer.
3. `llama_model_loader` reads GGUF metadata and maps model tensors.
4. `llama_model` selects the architecture-specific model implementation.
5. `llama_context` owns context parameters, memory state, scheduler state, and outputs.
6. `llama_graph` builds an architecture-specific GGML graph.
7. KV/memory code reserves and updates cache slots.
8. `ggml_backend_sched` assigns graph nodes to available backends and creates splits/copies.
9. `ggml_backend_graph_compute()` executes the graph.
10. Outputs are read and consumed by sampling, embeddings, or the server.

Key files: `src/llama.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-batch.cpp`.

## Execution and scheduler

GGML provides backend abstraction, graph execution, tensor buffers, and scheduling. The scheduler is implemented in `ggml/src/ggml-backend.cpp` and uses `ggml_backend_sched` and `ggml_backend_sched_split` to assign nodes, split graphs, and insert copies.

The scheduler already supports multiple local backends and pipeline copies. Future storage or distributed work must integrate with this layer instead of bypassing it.

## Memory and KV cache

Model mapping and tensor loading are implemented by `src/llama-model-loader.cpp`, `src/llama-mmap.cpp`, and `src/llama-model.cpp`. GGUF conversion and metadata tooling are under `gguf-py/` and the conversion scripts.

KV and recurrent state are represented by the memory subsystem in `src/llama-memory*.{cpp,h}`, `src/llama-kv-cache*.{cpp,h}`, and `src/llama-kv-cells.h`. The repository supports regular KV cache, sliding-window attention, recurrent/hybrid memory, and architecture-specific cache variants.

Current cache storage is local backend-buffer-based. SSD-backed and remote-RAM cache tiers are not implemented as a general policy system.

## Metal backend

Metal is implemented under `ggml/src/ggml-metal/`. Important files are `ggml-metal.cpp`, `ggml-metal-device.cpp`, `ggml-metal-context.m`, `ggml-metal-ops.cpp`, and `ggml-metal-tuning.cpp`. Kernels include quantized matrix operations, matrix-vector operations, Flash Attention, RoPE, normalization, softmax, reductions, and quantization/dequantization.

The current build on the M3 Air detects Metal and Accelerate successfully. No kernel profiling has been performed; suspected optimization targets are quantized GEMM, token-generation launch overhead, synchronization, tensor copies, and device-specific tuning.

## Quantization

Offline/model-format quantization is implemented through `src/llama-quant.cpp` and GGML quantization code under `ggml/src/`. The repository supports multiple F16/BF16/F32, Q2-Q8, K-quant, IQ, and related formats.

A general runtime policy for keeping multiple cached representations of a tensor is not currently present.

## Server and networking

The HTTP/OpenAI-compatible server is under `tools/server/`, including request queues, slots, streaming, metrics, and model management. It is an application-level service, not a tensor or expert transport. A future node transport should remain separate from HTTP request handling.

## Tests and benchmarks

Tests are configured in `tests/CMakeLists.txt` and include tokenizer, sampling, grammar, chat, batch, recurrent, quantization, backend, and model coverage. Server tests are under `tools/server/tests/`. `tools/llama-bench/` provides prompt-processing and generation benchmarks.

There is no recorded M3 Air throughput baseline yet.

## Current risks and bottlenecks

These are investigation targets, not measured conclusions:

- Metal quantized GEMM, Flash Attention, launch, copy, and synchronization costs.
- KV growth and allocation pressure, especially on the 8 GB M1.
- mmap paging versus explicit SSD streaming and prefetch.
- Network transfer and synchronization cost for token-by-token distributed execution.
- Heterogeneous M1/M3/M4 performance and memory capacity.
- Internal 256 GB SSD capacity and uncontrolled benchmark/cache artifacts.

## Proposed incremental architecture

Preserve the existing llama.cpp and GGML layers. Add optional components in this order:

1. Measurement and telemetry for local execution.
2. Apple/Metal tuning based on profiles.
3. Explicit local memory residency metadata.
4. Experimental SSD prefetch/cache with hard size and free-space limits.
5. KV policy and telemetry.
6. Separate node runtime and reliable transport.
7. Network-aware scheduler extensions.
8. MoE expert placement and batched routing.

Remote RAM and SSD must remain explicitly more expensive than local RAM and must not be represented as equivalent memory.

## Smallest useful milestone

Establish a correctness-gated M3 Air baseline using one small GGUF model. Record commit, hardware, macOS, compiler, model quantization, context, batch, GPU layers, prompt length, generation length, prompt throughput, generation throughput, TTFT, total latency, and peak memory. Compare CPU and Metal builds before changing code.
