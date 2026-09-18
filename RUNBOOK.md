# RUNBOOK.md - multi-Mac agent instructions

Operative runbook for AI agents working on the other machines of the
NewLLAMA cluster. The M3 MacBook Air is the development machine: it owns
the branch `research/moe-expert-sharding` and pushes to
`origin/research/apple-silicon-baseline` only when the user approves.
Machines here are consumers of that branch, never pushers.

## Fleet

| Machine | Chip | RAM | Role | Branch |
|---|---|---|---|---|
| MacBook Air | M3 | 16 GB | development, coordinator, reference | research/moe-expert-sharding |
| Mac mini | M1 | 8 GB | worker, memory-constrained tests | follows branch |
| Mac mini | M4 | 16 GB | benchmark node | follows branch |

All three connect over ethernet (gigabit, switch 2.5G). Never compare
battery results with AC results. Never compare unverified low-power
results with normal-power results.

## Standard setup on a mini (first time or after branch updates)

Run from the repo root on the mini:

```bash
git fetch origin
git checkout research/moe-expert-sharding 2>/dev/null || \
  git checkout -b research/moe-expert-sharding origin/research/moe-expert-sharding
git pull --ff-only origin research/moe-expert-sharding
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --target expert-catalog-selftest node-runtime-selftest \
  test-batch-alloc -j8
```

Validation gate before any experiment (all must pass, else stop and
report):

```bash
./build/bin/expert-catalog-selftest;  echo "exit must be 0"
./build/bin/node-runtime-selftest;    echo "exit must be 0"
./build/bin/test-batch-alloc | tail -3   # failures: 0
```

The M1 mini has 8 GB RAM. Do not run Metal inference and expert-cache
experiments at the same time. Prefer `GGML_METAL=OFF` builds when the
work is CPU/IO only; Metal init reserves wired memory.

## Role A: worker node (M1 mini, 8 GB)

The coordinator runs on the M3. The worker registers itself, sends
heartbeats, and leaves cleanly:

```bash
# coordinator first (M3, on AC power):
./build/bin/llama-node --port 49155 --monitor 120 --nodes 4

# then on the M1 mini (substitute the M3 LAN IP for HOST):
./build/bin/node-runtime-client <M3-IP> 49155 --monitor 60 --interval 5
```

Expected on the coordinator: registration JSON with `node_id` from the
mini hostname, heartbeats arriving at the interval, then a clean
departure (`state=2` in the final registry JSON). A non-graceful exit
(timeout instead of departure) is a finding, not a failure to hide.

## Role B: benchmark node (M4 mini, 16 GB)

1. Same setup and validation gate as Role A.
2. Run the transport baseline before anything else, and append raw
   numbers to `benchmarks/baseline/` in a dated file:

```bash
./build/bin/node-runtime-bench 1024 2000 > \
  benchmarks/baseline/$(date +%F)-m4-loopback.log 2>&1
```

3. The physical-link battery (ethernet) once both minis are on the
   switch. Record negotiated media on each machine first:

```bash
networksetup -getMediaOptions "Ethernet" ; ifconfig en0 | grep -E 'media|mtu'
```

Then run `node-runtime-bench` over the LAN between the two minis at
payload sizes 1 KiB, 64 KiB, 1 MiB (20 rounds each) and record: mean
RTT, min RTT, max RTT, MiB/s. The M3 numbers (loopback 1 MiB, 2.1-2.6
GiB/s) are the local reference, not a LAN expectation.

## Role C: MoE selective-load verification (either mini)

The OLMoE GGUF (6.86 GiB) is on the M3 only at
`models/moe/olmoe-1b-7b-0924-instruct-q8_0.gguf`. To replicate on a mini:

```bash
mkdir -p models/moe
curl -sL -C - -o models/moe/olmoe-1b-7b-0924-instruct-q8_0.gguf \
  "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q8_0.gguf"
# verify size: exactly 7359943136 bytes
stat -f%z models/moe/olmoe-1b-7b-0924-instruct-q8_0.gguf
```

Expected catalog read (any machine):

```bash
# expected: 16 MoE layers, 64 experts, 6.38 MiB per expert (2228224 bytes
# per tensor plane)
```

The M1 mini at 8 GB: keep cache capacity at or below 1 GiB during
expert-cache experiments and watch `memory_pressure` telemetry; a
warning-level pressure state invalidates the run.

## Rules for any agent on any machine

1. Measure, never assume. Label every number with machine, power state,
   and date.
2. Never push to origin. Commit locally with
   `Assisted-by: NewLLAMA (Ori)` only if the human asked for it.
3. Do not create files under `tests/`. New checks go into
   `common/*-selftest.cpp` wired in `common/CMakeLists.txt`.
4. Comments in code: short, simple English, ASCII only, no em dash.
5. Validation before done: selftests + `git diff --check`. Report
   honestly what passed and what did not run.
6. One behavior change per commit. No drive-by refactors.
7. If a run blocks more than 120 seconds on a network read, kill it and
   record the stall (this is a finding, not a failure to hide).
8. `PROJECT_STATE.md` is the source of truth. Read the last two blocks
   before starting anything; append a block after any measurement.

## Escalation

Report back to the M3 (human runs the coordinator side there) with:
machine, git SHA, exact commands, raw output, and one-sentence reading.
No conclusions without the raw numbers attached.
