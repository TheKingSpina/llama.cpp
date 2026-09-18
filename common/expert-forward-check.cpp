// Bit-exactness gate: the selectively loaded expert path must produce the
// same output as the full fused tensor path through ggml_mul_mat_id.

#include "expert-catalog.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool check(bool condition) {
    return condition;
}

constexpr int64_t n_embd = 32;
constexpr int64_t n_ff = 24;
constexpr int64_t n_expert = 4;
constexpr int n_tokens = 2;
constexpr int n_used = 2;
constexpr int64_t gate_plane_elems = n_ff * n_embd; // gate/up planes
constexpr int64_t down_plane_elems = n_embd * n_ff; // down planes, same size

// Deterministic fill value for plane t, expert e, element i.
float fill_value(size_t tensor_id, int64_t expert, int64_t element) {
    const uint64_t x = static_cast<uint64_t>(tensor_id * 1000003 +
                                             expert * 131 + element + 1) *
                       6364136223846793005ull;
    return static_cast<float>((x >> 33) % 4096) / 4096.0f;
}

// Write the synthetic MoE GGUF with F32 expert planes.
bool write_synthetic_moe(const std::string & path) {
    ggml_init_params gparams;
    gparams.mem_size = 8 * 1024 * 1024;
    gparams.mem_buffer = nullptr;
    gparams.no_alloc = false;
    ggml_context * ctx = ggml_init(gparams);
    if (!check(ctx != nullptr)) return false;
    gguf_context * gguf = gguf_init_empty();
    if (!check(gguf != nullptr)) {
        ggml_free(ctx);
        return false;
    }
    gguf_set_val_str(gguf, "general.architecture", "testmoe");

    const char * names[] = {
        "blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps",
    };
    const int64_t rows[] = {n_ff, n_ff, n_embd};
    const int64_t cols[] = {n_embd, n_embd, n_ff};
    for (size_t t = 0; t < 3; ++t) {
        ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols[t],
                                                  rows[t], n_expert);
        ggml_set_name(tensor, names[t]);
        float * data = static_cast<float *>(ggml_get_data(tensor));
        const int64_t plane = rows[t] * cols[t];
        for (int64_t e = 0; e < n_expert; ++e) {
            for (int64_t i = 0; i < plane; ++i) {
                data[e * plane + i] = fill_value(t, e, i);
            }
        }
        gguf_add_tensor(gguf, tensor);
    }
    const bool written = gguf_write_to_file(gguf, path.c_str(), false);
    gguf_free(gguf);
    ggml_free(ctx);
    return written;
}

// Build and run the one-layer MoE FFN graph (out is [n_embd, n_used,
// n_tokens]):
//   gate = gate_exps[:,:,id] @ b, up = up_exps[:,:,id] @ b
//   act = swiglu(gate, up), out = down_exps[:,:,id] @ act
// With slices set, the expert weights are rebuilt from loaded plane bytes;
// otherwise they come from the full tensors.
bool build_and_run(const float * full_gate, const float * full_up,
                   const float * full_down, const expert_catalog::loaded_expert * slices,
                   const float * inp_data, const int32_t * ids_data,
                   std::vector<float> & out_data) {
    ggml_init_params gparams;
    gparams.mem_size = ggml_tensor_overhead() * 512;
    gparams.mem_buffer = nullptr;
    gparams.no_alloc = true;
    ggml_context * ctx = ggml_init(gparams);
    if (!check(ctx != nullptr)) return false;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!check(backend != nullptr)) {
        ggml_free(ctx);
        return false;
    }

    // Expert weights: full tensors or rebuilt planes.
    ggml_tensor * w_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd,
                                              n_ff, n_expert);
    ggml_tensor * w_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd,
                                            n_ff, n_expert);
    ggml_tensor * w_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff,
                                              n_embd, n_expert);
    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!check(weights_buffer != nullptr)) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    if (slices == nullptr) {
        ggml_backend_tensor_set(w_gate, full_gate, 0,
                                ggml_nbytes(w_gate));
        ggml_backend_tensor_set(w_up, full_up, 0, ggml_nbytes(w_up));
        ggml_backend_tensor_set(w_down, full_down, 0, ggml_nbytes(w_down));
    } else {
        // Rebuild the 3D expert tensors plane by plane. The plane for
        // expert e must land at offset e * plane bytes, matching the file
        // layout the catalog describes.
        for (int64_t e = 0; e < n_expert; ++e) {
            ggml_backend_tensor_set(w_gate, slices[e].gate.data(),
                                    e * n_ff * n_embd * sizeof(float),
                                    n_ff * n_embd * sizeof(float));
            ggml_backend_tensor_set(w_up, slices[e].up.data(),
                                    e * n_ff * n_embd * sizeof(float),
                                    n_ff * n_embd * sizeof(float));
            ggml_backend_tensor_set(w_down, slices[e].down.data(),
                                    e * n_ff * n_embd * sizeof(float),
                                    n_ff * n_embd * sizeof(float));
        }
    }

    // Graph: gate/up/down routed by ids. mul_mat_id wants b as 3D
    // [cols, n_used, n_tokens], so the token input is 2D [n_embd,
    // n_used*n_tokens] reshaped. inp and ids live in their own buffer so
    // they can be filled before compute.
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           n_embd, n_used * n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_backend_buffer_t io_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!check(io_buffer != nullptr)) {
        ggml_backend_buffer_free(weights_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_tensor * b = ggml_reshape_3d(ctx, inp, n_embd, n_used, n_tokens);
    ggml_tensor * t_gate = ggml_mul_mat_id(ctx, w_gate, b, ids);
    ggml_tensor * t_up = ggml_mul_mat_id(ctx, w_up, b, ids);
    ggml_tensor * act = ggml_swiglu_split(ctx, t_gate, t_up);
    // act is [n_ff, n_used, n_tokens]; down projects back to n_embd.
    ggml_tensor * out = ggml_mul_mat_id(ctx, w_down, act, ids);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!check(ggml_gallocr_reserve(galloc, graph))) {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(weights_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    if (!check(ggml_gallocr_alloc_graph(galloc, graph))) {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(weights_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp, inp_data, 0,
                            ggml_nbytes(inp));
    ggml_backend_tensor_set(ids, ids_data, 0, ggml_nbytes(ids));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (!check(status == GGML_STATUS_SUCCESS)) {
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(weights_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    out_data.resize(n_embd * n_used * n_tokens);
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(io_buffer);
    ggml_backend_buffer_free(weights_buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

} // namespace

int main() {
    const std::string path = "expert-forward-check.gguf";
    if (!check(write_synthetic_moe(path))) return 1;

    // Deterministic input and router ids. The b tensor is [n_embd,
    // n_used, n_tokens]: column t of the 2D input holds the n_used copies
    // of token t, so both copies carry the same token values.
    std::vector<float> inp(n_embd * n_used * n_tokens);
    for (size_t i = 0; i < inp.size(); ++i) {
        const int64_t token = static_cast<int64_t>(i) / (n_embd * n_used);
        const int64_t elem = static_cast<int64_t>(i) % n_embd;
        inp[i] = fill_value(9, token, elem);
    }
    std::vector<int32_t> ids(n_used * n_tokens);
    // Column-major ids [e0, e1] per token: token 0 uses experts 1 and 2,
    // token 1 uses experts 3 and 0. All four experts participate.
    ids = {1, 2, 3, 0};

    // Full-tensor reference: all four experts resident.
    std::vector<float> full_gate(n_ff * n_embd * n_expert);
    std::vector<float> full_up(n_ff * n_embd * n_expert);
    std::vector<float> full_down(n_ff * n_embd * n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        const int64_t gate_plane = n_ff * n_embd;
        for (int64_t i = 0; i < gate_plane; ++i) {
            full_gate[e * gate_plane + i] = fill_value(0, e, i);
            full_up[e * gate_plane + i] = fill_value(1, e, i);
        }
        for (int64_t i = 0; i < down_plane_elems; ++i) {
            full_down[e * down_plane_elems + i] = fill_value(2, e, i);
        }
    }
    std::vector<float> out_full;
    if (!check(build_and_run(full_gate.data(), full_up.data(), full_down.data(),
                             nullptr, inp.data(), ids.data(), out_full))) {
        return 1;
    }

    // Selective path: experts 1, 2, 3, 0 read from the file in a shuffled
    // order to prove per-plane placement, not sequential copy.
    std::vector<expert_catalog::loaded_expert> slices(n_expert);
    {
        expert_catalog::expert_catalog catalog;
        if (!check(catalog.load_from_gguf(path))) return 1;
        expert_catalog::expert_reader reader;
        if (!check(reader.open(path))) return 1;
        const uint32_t order[] = {3, 1, 0, 2};
        for (const uint32_t e : order) {
            if (!check(reader.read_expert(catalog, 0, e, slices[e]))) return 1;
        }
    }
    std::vector<float> out_sel;
    if (!check(build_and_run(nullptr, nullptr, nullptr, slices.data(),
                             inp.data(), ids.data(), out_sel))) {
        std::remove(path.c_str());
        return 1;
    }

    // Cache path: same weights served through expert_cache under eviction
    // pressure (capacity holds two experts, gather serves four).
    std::vector<float> out_cache;
    uint64_t evictions_seen = 0;
    {
        expert_catalog::expert_catalog catalog;
        if (!check(catalog.load_from_gguf(path))) return 1;
        expert_catalog::expert_reader reader;
        if (!check(reader.open(path))) return 1;
        expert_catalog::expert_cache cache(catalog, reader, 2 * 9216);
        expert_catalog::expert_moe_runner runner(cache, 0);
        const uint32_t ids4[] = {3, 1, 0, 2};
        if (!check(runner.gather(ids4, 4))) return 1;
        std::vector<expert_catalog::loaded_expert> rebuilt(n_expert);
        for (size_t s = 0; s < runner.served_count(); ++s) {
            rebuilt[runner.served_id(s)] = runner.planes(s);
        }
        if (!check(build_and_run(nullptr, nullptr, nullptr, rebuilt.data(),
                                 inp.data(), ids.data(), out_cache))) {
            std::remove(path.c_str());
            return 1;
        }
        const expert_catalog::cache_stats & stats = cache.stats();
        if (!check(stats.evictions >= 2) || !check(stats.hits == 0)) {
            std::remove(path.c_str());
            return 1;
        }
        evictions_seen = stats.evictions;
    }

    if (!check(out_full == out_sel)) {
        std::fprintf(stderr, "forward mismatch: full %zu vs selective %zu\n",
                     out_full.size(), out_sel.size());
        for (size_t i = 0; i < out_full.size() && i < out_sel.size(); ++i) {
            if (out_full[i] != out_sel[i]) {
                std::printf("first diff at %zu: full=%.8f sel=%.8f\n", i,
                            out_full[i], out_sel[i]);
                break;
            }
        }
        std::remove(path.c_str());
        return 1;
    }
    std::printf("bit-exact: %zu output elements match\n", out_full.size());

    // The cache path must also be bit-exact, with real eviction pressure:
    // capacity holds only two of the four experts.
    if (!check(out_cache == out_full)) {
        std::fprintf(stderr, "cache path mismatch\n");
        std::remove(path.c_str());
        return 1;
    }
    if (!check(evictions_seen == 2)) {
        std::remove(path.c_str());
        return 1;
    }
    std::printf("cache path bit-exact too, evictions=%llu\n",
                (unsigned long long)evictions_seen);

    // Gate self-check: one corrupted expert plane must change the output.
    // Expert 1 is used by token 0.
    {
        std::vector<expert_catalog::loaded_expert> corrupt(slices);
        std::vector<uint8_t> & plane = corrupt[1].gate;
        const size_t mid = plane.size() / 2;
        plane[mid] = static_cast<uint8_t>(plane[mid] ^ 0xff);
        std::vector<float> out_corrupt;
        if (!check(build_and_run(nullptr, nullptr, nullptr, corrupt.data(),
                                 inp.data(), ids.data(), out_corrupt))) {
            std::remove(path.c_str());
            return 1;
        }
        if (!check(out_corrupt != out_full)) {
            std::fprintf(stderr, "gate not sensitive: corruption went undetected\n");
            std::remove(path.c_str());
            return 1;
        }
    }

    std::remove(path.c_str());
    return 0;
}