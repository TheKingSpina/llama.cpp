# Baseline benchmark artifacts

All results must include the power state. Use these labels:

- `normal-power`: AC power with `lowpowermode 0`.
- `low-power`: Low Power Mode enabled. Record whether on battery or AC.
- `unknown-power`: historical result where the state was not captured.

Current model: `models/baseline/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`.

Current verified normal-power report: `2026-08-27-normal-power.md`.

This baseline is frozen for comparison. New implementation work must not
overwrite these artifacts; later results must use a new dated report and
include the same power-state and correctness gates.

Benchmark parameters for the current report: 5 repetitions, prompt 512, generation 128, batch 512, microbatch 512, explicit `BLAS` and `MTL0` devices.

The 2026-08-27 Metal diagnostic profile is limited to runtime graph diagnostics because
`xctrace` requires full Xcode and the machine currently has CommandLineTools selected.

Metal operation correctness passed with `test-backend-ops test -b MTL0 -j 1`:
14,283/14,283 cases, including 240 quantized Flash Attention vector cases.

The long runtime profile is stored in `2026-08-27-metal-long-profile.md` and is
labelled separately because it ran with AC connected but `lowpowermode 1`.

Metal System Trace is stored in `2026-08-27-metal-system-trace.trace`. It was
captured with Xcode 16.0 using `DEVELOPER_DIR`; the target exited with status 0.
GPU counters and shader timeline were disabled in this capture, so no kernel
performance conclusion is drawn from it.

An extended capture using `--instrument 'Metal Application'` is stored in
`2026-08-27-metal-extended.trace`. The target exited normally, but Instruments
still reported no counter set and a disabled shader timeline.

The `fa-vec` tuner sweep was intentionally stopped after exceeding the bounded
time budget. Its log records thermal drift/noisy cells; partial rows are not a
trusted benchmark and must not be used for tuning decisions.

The first targeted `mul_mv`/`mul_mm` filter attempt produced no timing rows
because the filters did not match the available perf cases. The logs are kept
as evidence of the attempt, not as benchmark results.

## 2026-08-27 bounded Metal measurement block

Power state at the check: AC connected, `lowpowermode 0` on the AC profile.

`test-backend-ops --help` confirms that `-p` is called a params regex, but the
current implementation compares the complete operation description and vars
string for exact equality. `--list-ops` enumerates operation names only; it does
not enumerate the perf-case parameter strings. This explains why guessed
q4_K `mul_mv` and `mul_mm` filters produced no rows.

An unfiltered `perf -b MTL0 -o MUL_MAT --output csv` enumeration was started
with a 120-second bound. The perf suite did not finish and was stopped. The
partial artifact is `2026-08-27-mul-mat-list.csv`; it is not a benchmark result
and contains no accepted q4_K timing row. Consequently this block measured no
verified q4_K `mul_mv` n=1 case and no verified q4_K `mul_mm` n>1 case. No
thermal or performance conclusion is drawn from the incomplete command.

The Flash Attention tuner was not launched. No supported option was identified
to constrain its grid, and the previous broad sweep exceeded the time budget
with thermal drift/noisy cells. A new unrestricted sweep would not be a bounded
measurement.

The existing trace was exported with `xctrace export --toc`. It verifies a
successful `llama-cli` run, 2.108253 s duration, the M3 target, and the exact
Metal System Trace configuration. Instruments recorded no counter set and had
the shader timeline disabled. Thus the trace is useful for provenance and
execution lifecycle, but cannot support kernel timing, GPU utilization, or a
performance claim.
