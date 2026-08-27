# Metal diagnostic profile

- Date: 2026-08-27
- Hardware: Apple M3, 16 GB
- Power: Now drawing from 'AC Power'  -InternalBattery-0 (id=22347875)	75%; charging; 0:52 remaining present: true ; lowpowermode 1
- Commit: deae5ee13
- Tooling: xctrace unavailable (CommandLineTools active; full Xcode required)

## Runtime graph diagnostics

0.00.042.538 W srv  llama_server: -----------------
0.00.042.539 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.042.539 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.042.540 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.042.540 W srv  llama_server: -----------------
0.00.249.215 W load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
