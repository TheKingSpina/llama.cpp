#include "expert-catalog.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace expert_catalog {

namespace {

std::string json_escape(const std::string & value) {
    std::ostringstream result;
    result << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') {
            result << '\\' << c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            result << "\\u" << std::hex << static_cast<int>(c) << std::dec;
        } else {
            result << c;
        }
    }
    result << '"';
    return result.str();
}

enum fused_kind : uint8_t {
    fused_none = 0,
    fused_gate_up_exps,
    fused_down_exps,
    fused_gate_up_ff,
    fused_down_ff,
    fused_gate_exps,
    fused_up_exps,
};

struct fused_name {
    fused_kind kind = fused_none;
    uint32_t layer_index = 0;
};

// Match "blk.<N>.<suffix>" for the fused expert tensor suffixes. The
// suffixes follow the canonical names in src/llama-arch.cpp.
fused_name classify_expert_tensor(const char * name) {
    fused_name result;
    const std::string text(name);
    if (text.compare(0, 4, "blk.") != 0) {
        return result;
    }
    size_t digits_end = 4;
    while (digits_end < text.size() &&
           text[digits_end] >= '0' && text[digits_end] <= '9') {
        ++digits_end;
    }
    if (digits_end == 4 || digits_end >= text.size() || text[digits_end] != '.') {
        return result;
    }
    const std::string suffix = text.substr(digits_end + 1);
    if (suffix == "ffn_gate_up_exps") {
        result.kind = fused_gate_up_exps;
    } else if (suffix == "ffn_down_exps") {
        result.kind = fused_down_exps;
    } else if (suffix == "ffn_gate_up_ff") {
        result.kind = fused_gate_up_ff;
    } else if (suffix == "ffn_down_ff") {
        result.kind = fused_down_ff;
    } else if (suffix == "ffn_gate_exps") {
        result.kind = fused_gate_exps;
    } else if (suffix == "ffn_up_exps") {
        result.kind = fused_up_exps;
    } else {
        return result;
    }
    uint64_t layer = 0;
    for (size_t i = 4; i < digits_end; ++i) {
        layer = layer * 10 + static_cast<uint64_t>(text[i] - '0');
        if (layer > 0xffffffffull) {
            result.kind = fused_none;
            return result;
        }
    }
    result.layer_index = static_cast<uint32_t>(layer);
    return result;
}

struct layer_accumulator {
    bool has_gate_up = false;
    bool has_down = false;
    bool has_gate = false;
    bool has_up = false;
    uint64_t gate_up_bytes = 0;
    uint64_t down_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t up_bytes = 0;
    uint32_t gate_up_experts = 0;
    uint32_t down_experts = 0;
    uint32_t gate_experts = 0;
    uint32_t up_experts = 0;
};

// Expert count from tensor shape: fused expert tensors are 3D with the
// expert count in ne[2]. Returns 0 when the shape does not say it.
uint32_t tensor_expert_count(const gguf_context * ctx, int64_t tensor_id) {
    const int64_t * ne = gguf_get_tensor_ne(ctx, tensor_id);
    int dims = 1;
    for (int d = GGML_MAX_DIMS - 1; d > 0; --d) {
        if (ne[d] != 1) {
            dims = d + 1;
            break;
        }
    }
    if (dims == 3 && ne[2] > 0 && ne[2] <= 0xffffffffll) {
        return static_cast<uint32_t>(ne[2]);
    }
    return 0;
}

} // namespace

bool expert_catalog::load_from_gguf(const std::string & path) {
    layers_.clear();
    summary_ = {};

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        return false;
    }

    const int64_t arch_key = gguf_find_key(ctx, "general.architecture");
    if (arch_key >= 0) {
        summary_.architecture = gguf_get_val_str(ctx, arch_key);
    }

    // Model-wide expert count from metadata, used when a tensor shape does
    // not carry it (2D per-expert storage variants).
    uint32_t metadata_expert_count = 0;
    if (!summary_.architecture.empty()) {
        const std::string key = summary_.architecture + ".expert_count";
        const int64_t expert_key = gguf_find_key(ctx, key.c_str());
        if (expert_key >= 0 &&
            gguf_get_kv_type(ctx, expert_key) == GGUF_TYPE_UINT32) {
            metadata_expert_count = gguf_get_val_u32(ctx, expert_key);
        }
    }

    std::vector<layer_accumulator> accumulators;
    const int64_t tensor_count = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < tensor_count; ++i) {
        const fused_name fused = classify_expert_tensor(gguf_get_tensor_name(ctx, i));
        if (fused.kind == fused_none) {
            continue;
        }
        if (fused.layer_index >= accumulators.size()) {
            accumulators.resize(fused.layer_index + 1);
        }
        layer_accumulator & acc = accumulators[fused.layer_index];
        const uint64_t tensor_bytes = gguf_get_tensor_size(ctx, i);
        const uint32_t experts = tensor_expert_count(ctx, i);
        switch (fused.kind) {
            case fused_gate_up_exps:
            case fused_gate_up_ff:
                acc.has_gate_up = true;
                acc.gate_up_bytes = tensor_bytes;
                acc.gate_up_experts = experts;
                break;
            case fused_down_exps:
            case fused_down_ff:
                acc.has_down = true;
                acc.down_bytes = tensor_bytes;
                acc.down_experts = experts;
                break;
            case fused_gate_exps:
                acc.has_gate = true;
                acc.gate_bytes = tensor_bytes;
                acc.gate_experts = experts;
                break;
            case fused_up_exps:
                acc.has_up = true;
                acc.up_bytes = tensor_bytes;
                acc.up_experts = experts;
                break;
            default:
                break;
        }
    }
    gguf_free(ctx);

    for (size_t layer = 0; layer < accumulators.size(); ++layer) {
        const layer_accumulator & acc = accumulators[layer];
        if (!acc.has_gate_up && !acc.has_down) {
            continue;
        }
        // All present tensor shapes must agree; disagreeing layers are
        // skipped, not guessed.
        const uint32_t shape_counts[4] = {
            acc.gate_up_experts, acc.down_experts, acc.gate_experts, acc.up_experts,
        };
        const bool shape_present[4] = {
            acc.has_gate_up, acc.has_down, acc.has_gate, acc.has_up,
        };
        uint32_t expert_count = 0;
        bool consistent = true;
        for (int t = 0; t < 4; ++t) {
            if (!shape_present[t] || shape_counts[t] == 0) {
                continue;
            }
            if (expert_count == 0) {
                expert_count = shape_counts[t];
            } else if (shape_counts[t] != expert_count) {
                consistent = false;
            }
        }
        if (!consistent) {
            continue;
        }
        if (expert_count == 0) {
            expert_count = metadata_expert_count;
        }
        if (expert_count == 0) {
            continue;
        }
        // Byte size of one expert slice in each fused tensor.
        expert_layer entry;
        entry.layer_index = static_cast<uint32_t>(layer);
        entry.expert_count = expert_count;
        entry.gate_up_total_bytes = acc.gate_up_bytes;
        entry.down_total_bytes = acc.down_bytes;
        entry.gate_total_bytes = acc.gate_bytes;
        entry.up_total_bytes = acc.up_bytes;
        if (acc.has_gate_up) {
            if (acc.gate_up_bytes % expert_count != 0) {
                continue;
            }
            entry.gate_up_bytes = acc.gate_up_bytes / expert_count;
        }
        if (acc.has_down) {
            if (acc.down_bytes % expert_count != 0) {
                continue;
            }
            entry.down_bytes = acc.down_bytes / expert_count;
        }
        if (acc.has_gate) {
            if (acc.gate_bytes % expert_count != 0) {
                continue;
            }
            entry.gate_bytes = acc.gate_bytes / expert_count;
        }
        if (acc.has_up) {
            if (acc.up_bytes % expert_count != 0) {
                continue;
            }
            entry.up_bytes = acc.up_bytes / expert_count;
        }
        layers_.push_back(entry);
    }
    std::stable_sort(layers_.begin(), layers_.end(),
        [](const expert_layer & left, const expert_layer & right) {
            return left.layer_index < right.layer_index;
        });

    if (!layers_.empty()) {
        summary_.layer_count = static_cast<uint32_t>(layers_.size());
        summary_.expert_count = layers_.front().expert_count;
        for (const expert_layer & entry : layers_) {
            summary_.experts_total_bytes +=
                entry.expert_count * entry.expert_bytes();
        }
    }
    return valid();
}

const expert_layer * expert_catalog::find_layer(uint32_t layer_index) const {
    for (const expert_layer & entry : layers_) {
        if (entry.layer_index == layer_index) {
            return &entry;
        }
    }
    return nullptr;
}

uint64_t expert_catalog::expert_bytes(uint32_t layer_index) const {
    const expert_layer * entry = find_layer(layer_index);
    return entry == nullptr ? 0 : entry->expert_bytes();
}

std::string expert_catalog::json() const {
    std::ostringstream result;
    result << "{\"architecture\":" << json_escape(summary_.architecture)
           << ",\"layer_count\":" << summary_.layer_count
           << ",\"expert_count\":" << summary_.expert_count
           << ",\"experts_total_bytes\":" << summary_.experts_total_bytes
           << ",\"layers\":[";
    for (size_t i = 0; i < layers_.size(); ++i) {
        const expert_layer & entry = layers_[i];
        if (i > 0) {
            result << ',';
        }
        result << "{\"layer_index\":" << entry.layer_index
               << ",\"expert_count\":" << entry.expert_count
               << ",\"expert_bytes\":" << entry.expert_bytes() << '}';
    }
    result << "]}";
    return result.str();
}

expert_placement_table::expert_placement_table(const expert_catalog & catalog,
                                               size_t max_entries)
    : catalog_(catalog), max_entries_(max_entries) {}

bool expert_placement_table::assign(uint32_t layer_index, uint32_t expert_index,
                                    const std::string & node_id) {
    const expert_layer * layer = catalog_.find_layer(layer_index);
    if (layer == nullptr || expert_index >= layer->expert_count ||
        node_id.empty() || entries_.size() >= max_entries_) {
        return false;
    }
    for (expert_placement & entry : entries_) {
        if (entry.layer_index == layer_index && entry.expert_index == expert_index) {
            entry.node_id = node_id;
            entry.bytes = layer->expert_bytes();
            return true;
        }
    }
    expert_placement entry;
    entry.layer_index = layer_index;
    entry.expert_index = expert_index;
    entry.node_id = node_id;
    entry.bytes = layer->expert_bytes();
    entries_.push_back(entry);
    return true;
}

const std::string & expert_placement_table::node_for(uint32_t layer_index,
                                                     uint32_t expert_index) const {
    static const std::string empty;
    for (const expert_placement & entry : entries_) {
        if (entry.layer_index == layer_index && entry.expert_index == expert_index) {
            return entry.node_id;
        }
    }
    return empty;
}

bool expert_placement_table::unassign(uint32_t layer_index, uint32_t expert_index) {
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].layer_index == layer_index &&
            entries_[i].expert_index == expert_index) {
            entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(i));
            return true;
        }
    }
    return false;
}

std::vector<node_placement_summary> expert_placement_table::node_summaries() const {
    std::vector<node_placement_summary> summaries;
    for (const expert_placement & entry : entries_) {
        node_placement_summary * summary = nullptr;
        for (node_placement_summary & existing : summaries) {
            if (existing.node_id == entry.node_id) {
                summary = &existing;
                break;
            }
        }
        if (summary == nullptr) {
            node_placement_summary added;
            added.node_id = entry.node_id;
            summaries.push_back(added);
            summary = &summaries.back();
        }
        ++summary->experts;
        summary->bytes += entry.bytes;
    }
    return summaries;
}

std::string expert_placement_table::json() const {
    std::ostringstream result;
    result << "{\"protocol_version\":" << placement_protocol_version
           << ",\"entries\":[";
    for (size_t i = 0; i < entries_.size(); ++i) {
        const expert_placement & entry = entries_[i];
        if (i > 0) {
            result << ',';
        }
        result << "{\"layer_index\":" << entry.layer_index
               << ",\"expert_index\":" << entry.expert_index
               << ",\"node_id\":" << json_escape(entry.node_id)
               << ",\"bytes\":" << entry.bytes << '}';
    }
    result << "]}";
    return result.str();
}

} // namespace expert_catalog