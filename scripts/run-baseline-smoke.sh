#!/bin/sh

set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model=${1:-"$repo_dir/models/baseline/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"}
mode=${2:-metal}

if [ ! -f "$model" ]; then
    printf 'model not found: %s\n' "$model" >&2
    exit 1
fi

case "$mode" in
    cpu)
        exec "$repo_dir/build/bin/llama-cli" -m "$model" -ngl 0 -n 8 \
            -p 'Return exactly: baseline-ok' --no-display-prompt --single-turn
        ;;
    metal)
        exec "$repo_dir/build/bin/llama-cli" -m "$model" -ngl -1 -n 8 \
            -p 'Return exactly: baseline-ok' --no-display-prompt --single-turn
        ;;
    *)
        printf 'usage: %s [model.gguf] [cpu|metal]\n' "$0" >&2
        exit 2
        ;;
esac