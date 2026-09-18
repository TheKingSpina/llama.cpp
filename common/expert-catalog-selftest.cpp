#include "expert-catalog.h"

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

struct tensor_spec {
    std::string name;
    int64_t ne[3];
    ggml_type type;
};

// Write a metadata-only GGUF file. The catalog reads metadata only, so the
// tensor data blob is never needed.
bool write_synthetic_gguf(const std::string & path,
                          const std::string & architecture,
                          bool include_expert_count_kv,
                          uint32_t expert_count_kv,
                          const std::vector<tensor_spec> & tensors) {
    ggml_init_params gparams;
    gparams.mem_size = 1024 * 1024;
    gparams.mem_buffer = nullptr;
    gparams.no_alloc = true;
    ggml_context * ctx = ggml_init(gparams);
    if (ctx == nullptr) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    gguf_set_val_str(gguf, "general.architecture", architecture.c_str());
    if (include_expert_count_kv) {
        gguf_set_val_u32(gguf, (architecture + ".expert_count").c_str(),
                         expert_count_kv);
    }
    for (const tensor_spec & spec : tensors) {
        ggml_tensor * tensor = ggml_new_tensor(ctx, spec.type, 3, spec.ne);
        ggml_set_name(tensor, spec.name.c_str());
        gguf_add_tensor(gguf, tensor);
    }
    const bool written = gguf_write_to_file(gguf, path.c_str(), true);
    gguf_free(gguf);
    ggml_free(ctx);
    return written;
}

tensor_spec gate_exps(uint32_t layer, int64_t n_embd, int64_t n_ff,
                      int64_t experts) {
    return {"blk." + std::to_string(layer) + ".ffn_gate_exps",
            {n_embd, n_ff, experts}, GGML_TYPE_F32};
}

tensor_spec up_exps(uint32_t layer, int64_t n_embd, int64_t n_ff,
                    int64_t experts) {
    return {"blk." + std::to_string(layer) + ".ffn_up_exps",
            {n_embd, n_ff, experts}, GGML_TYPE_F32};
}

tensor_spec down_exps(uint32_t layer, int64_t n_ff, int64_t n_embd,
                      int64_t experts) {
    return {"blk." + std::to_string(layer) + ".ffn_down_exps",
            {n_ff, n_embd, experts}, GGML_TYPE_F32};
}

tensor_spec dense_ffn(uint32_t layer, int64_t n_ff, int64_t n_embd) {
    return {"blk." + std::to_string(layer) + ".ffn_down.weight",
            {n_ff, n_embd, 1}, GGML_TYPE_F32};
}

tensor_spec attn(uint32_t layer, int64_t n_embd) {
    return {"blk." + std::to_string(layer) + ".attn_q.weight",
            {n_embd, n_embd, 1}, GGML_TYPE_F32};
}

} // namespace

int main() {
    const std::string moe_path = "expert-catalog-selftest-moe.gguf";
    const std::string mismatch_path = "expert-catalog-selftest-mismatch.gguf";
    const std::string metadata_path = "expert-catalog-selftest-metadata.gguf";

    // MoE file: layers 0 and 1 with separate gate/up/down expert tensors,
    // one dense layer and attention tensors that must be ignored.
    {
        const int64_t n_embd = 8;
        const int64_t n_ff = 16;
        const int64_t experts = 4;
        std::vector<tensor_spec> tensors;
        tensors.push_back(attn(0, n_embd));
        tensors.push_back(dense_ffn(0, n_ff, n_embd));
        tensors.push_back(gate_exps(0, n_embd, n_ff, experts));
        tensors.push_back(up_exps(0, n_embd, n_ff, experts));
        tensors.push_back(down_exps(0, n_ff, n_embd, experts));
        tensors.push_back(gate_exps(1, n_embd, n_ff, experts));
        tensors.push_back(up_exps(1, n_embd, n_ff, experts));
        tensors.push_back(down_exps(1, n_ff, n_embd, experts));
        if (!check(write_synthetic_gguf(moe_path, "qwen3moe", true, experts,
                                        tensors))) {
            return 1;
        }

        expert_catalog::expert_catalog catalog;
        if (!check(catalog.load_from_gguf(moe_path))) return 1;
        if (!check(catalog.valid())) return 1;
        const expert_catalog::catalog_summary & summary = catalog.summary();
        if (!check(summary.architecture == "qwen3moe") ||
            !check(summary.layer_count == 2) ||
            !check(summary.expert_count == 4)) return 1;

        // Slice bytes: gate and up are [8, 16, 4] F32 = 2048 total,
        // 512 per expert; down is the same total.
        const expert_catalog::expert_layer * layer0 = catalog.find_layer(0);
        if (!check(layer0 != nullptr) || !check(layer0->expert_count == 4) ||
            !check(layer0->gate_bytes == 512) || !check(layer0->up_bytes == 512) ||
            !check(layer0->down_bytes == 512) ||
            !check(layer0->expert_bytes() == 1536)) return 1;
        if (!check(catalog.expert_bytes(1) == 1536)) return 1;
        if (!check(catalog.find_layer(2) == nullptr) ||
            !check(catalog.expert_bytes(2) == 0)) return 1;

        // Dense layer total must not leak into the expert totals.
        if (!check(summary.experts_total_bytes == 2 * 4 * 1536)) return 1;

        const std::string json = catalog.json();
        if (!check(json.find("\"architecture\":\"qwen3moe\"") != std::string::npos) ||
            !check(json.find("\"layer_count\":2") != std::string::npos) ||
            !check(json.find("\"expert_count\":4") != std::string::npos) ||
            !check(json.find("\"experts_total_bytes\":12288") != std::string::npos) ||
            !check(json.find("\"layer_index\":0") != std::string::npos)) return 1;
    }

    // Mismatched expert counts across tensors of one layer: the layer is
    // skipped, not guessed.
    {
        std::vector<tensor_spec> tensors;
        tensors.push_back(gate_exps(0, 8, 16, 4));
        tensors.push_back(down_exps(0, 16, 8, 8));
        if (!check(write_synthetic_gguf(mismatch_path, "testmoe", true, 4,
                                        tensors))) {
            return 1;
        }
        expert_catalog::expert_catalog catalog;
        // No usable expert tensors means the load reports false.
        if (!check(!catalog.load_from_gguf(mismatch_path))) return 1;
        if (!check(catalog.find_layer(0) == nullptr)) return 1;
        if (!check(catalog.summary().layer_count == 0)) return 1;
    }

    // Expert count from metadata when tensor shapes carry no expert axis.
    {
        std::vector<tensor_spec> tensors;
        // Two expert-count-sized tensors per layer without a 3D shape.
        tensors.push_back({"blk.0.ffn_gate_exps", {32, 8, 1}, GGML_TYPE_F32});
        tensors.push_back({"blk.0.ffn_down_exps", {32, 8, 1}, GGML_TYPE_F32});
        if (!check(write_synthetic_gguf(metadata_path, "testmoe", true, 8,
                                        tensors))) {
            return 1;
        }
        expert_catalog::expert_catalog catalog;
        const bool loaded_meta = catalog.load_from_gguf(metadata_path);
        if (!check(loaded_meta)) return 1;
        if (!check(catalog.valid())) return 1;
        const expert_catalog::expert_layer * layer0 = catalog.find_layer(0);
        if (layer0 == nullptr) return 1;
        if (!check(layer0->expert_count == 8) ||
            !check(layer0->gate_bytes == 128) || !check(layer0->down_bytes == 128)) {
            return 1;
        }
        if (!check(catalog.summary().expert_count == 8)) return 1;
    }

    // Missing file must fail cleanly.
    {
        expert_catalog::expert_catalog catalog;
        const bool missing = catalog.load_from_gguf("expert-catalog-selftest-missing.gguf");
        if (!check(!missing)) {
            return 1;
        }
    }

    // Placement table.
    {
        expert_catalog::expert_catalog catalog;
        if (!check(catalog.load_from_gguf(moe_path))) return 1;

        expert_catalog::expert_placement_table table(catalog, 8);
        if (!check(table.assign(0, 0, "node-a")) ||
            !check(table.assign(0, 1, "node-a")) ||
            !check(table.assign(1, 3, "node-b"))) return 1;
        if (!check(table.node_for(0, 0) == "node-a") ||
            !check(table.node_for(1, 3) == "node-b") ||
            !check(table.node_for(0, 2).empty())) return 1;

        // Re-assignment replaces the node, it does not duplicate.
        if (!check(table.assign(0, 0, "node-c")) || !check(table.size() == 3) ||
            !check(table.node_for(0, 0) == "node-c")) return 1;

        // Bounds and input validation.
        if (!check(!table.assign(9, 0, "node-a"))) return 1;
        if (!check(!table.assign(0, 4, "node-a"))) return 1;
        if (!check(!table.assign(0, 0, ""))) return 1;
        if (!check(table.size() == 3)) return 1;

        // Per-node totals in first-seen order. After the re-assignment of
        // expert (0,0) to node-c, each node holds one expert of 1536 bytes.
        std::vector<expert_catalog::node_placement_summary> summaries =
            table.node_summaries();
        if (!check(summaries.size() == 3)) return 1;
        if (!check(summaries[0].node_id == "node-c") ||
            !check(summaries[1].node_id == "node-a") ||
            !check(summaries[2].node_id == "node-b")) return 1;
        for (const expert_catalog::node_placement_summary & summary : summaries) {
            if (!check(summary.experts == 1) || !check(summary.bytes == 1536)) {
                return 1;
            }
        }

        const std::string json = table.json();
        if (!check(json.find("\"protocol_version\":1") != std::string::npos) ||
            !check(json.find("\"expert_index\":3") != std::string::npos) ||
            !check(json.find("\"node_id\":\"node-c\"") != std::string::npos)) return 1;

        if (!check(table.unassign(0, 0)) || !check(table.size() == 2) ||
            !check(!table.unassign(0, 0)) ||
            !check(table.node_for(0, 0).empty())) return 1;

        // Table capacity is enforced.
        expert_catalog::expert_placement_table small(catalog, 2);
        if (!check(small.assign(0, 0, "node-a")) ||
            !check(small.assign(0, 1, "node-a")) ||
            !check(!small.assign(0, 2, "node-a"))) return 1;
    }

    std::remove(moe_path.c_str());
    std::remove(mismatch_path.c_str());
    std::remove(metadata_path.c_str());
    return 0;
}
