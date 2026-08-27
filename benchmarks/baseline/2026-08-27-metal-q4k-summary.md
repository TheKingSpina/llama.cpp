# q4_K Metal microbenchmark summary

- Date: 2026-08-27
- Hardware: MacBook Air M3, AC power
- Backend: MTL0
- Correctness gate: `test-backend-ops test -b MTL0 -j 1`, 14283/14283
- Case filter: `-p 'type_a=q4_K'`
- Decode-like case: m=4096, n=1, k=14336, 403.01 us/run, 291.41 GFLOPS
- Batched cases: n=2 856.78 us/run, 274.15 GFLOPS; n=5 3136.47 us/run, 187.22 GFLOPS; n=8 used the Metal `mul_mm` path; n=512 used the large-batch path.
- The console log contains 39 timing rows including `MUL_MAT_ID` cases. Results are baseline measurements only; no kernel change was made.
- Runtime profile: prompt 844.2 t/s, generation 139.3 t/s. This is a single bounded run, not a replacement for the 5-repetition baseline.
