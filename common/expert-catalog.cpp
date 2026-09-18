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

// Match "blk.<N>.<suffix>" for the expert tensor suffixes. The suffixes
// follow the canonical names in src/llama-arch.cpp.
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
    std::string suffix = text.substr(digits_end + 1);
    // Strip the trailing .weight or .bias component from real GGUF names.
    const size_t dot = suffix.rfind('.');
    if (dot != std::string::npos) {
        suffix = suffix.substr(0, dot);
    }
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
    uint64_t gate_up_total_bytes = 0;
    uint64_t down_total_bytes = 0;
    uint64_t gate_total_bytes = 0;
    uint64_t up_total_bytes = 0;
    // Exact per-expert slice bytes for 3D expert tensors; 0 when unknown.
    uint64_t gate_up_slice_bytes = 0;
    uint64_t down_slice_bytes = 0;
    uint64_t gate_slice_bytes = 0;
    uint64_t up_slice_bytes = 0;
    uint32_t gate_up_experts = 0;
    uint32_t down_experts = 0;
    uint32_t gate_experts = 0;
    uint32_t up_experts = 0;
};

// Effective dimension count of the tensor.
int tensor_dims(const int64_t * ne) {
    for (int d = GGML_MAX_DIMS - 1; d > 0; --d) {
        if (ne[d] != 1) {
            return d + 1;
        }
    }
    return 1;
}

// Expert count from tensor shape: expert tensors are 3D with the expert
// count in ne[2]. Returns 0 when the shape does not say it.
uint32_t tensor_expert_count(const gguf_context * ctx, int64_t tensor_id) {
    const int64_t * ne = gguf_get_tensor_ne(ctx, tensor_id);
    if (tensor_dims(ne) == 3 && ne[2] > 0 && ne[2] <= 0xffffffffll) {
        return static_cast<uint32_t>(ne[2]);
    }
    return 0;
}

// Byte size of one expert slice along ne[2] of a 3D tensor. The data
// layout is row-major over ne[0] and ne[1], so the slice is the bytes of
// one expert plane. Dividing the total by the expert count would be wrong
// for quantized types: each row is a whole number of type blocks.
uint64_t tensor_slice_bytes(const gguf_context * ctx, int64_t tensor_id) {
    const int64_t * ne = gguf_get_tensor_ne(ctx, tensor_id);
    if (tensor_dims(ne) != 3) {
        return 0;
    }
    const ggml_type type = gguf_get_tensor_type(ctx, tensor_id);
    const uint64_t plane = static_cast<uint64_t>(ne[0]) * static_cast<uint64_t>(ne[1]);
    const uint64_t blck = static_cast<uint64_t>(ggml_blck_size(type));
    if (blck == 0 || plane % blck != 0) {
        return 0;
    }
    return (plane / blck) * ggml_type_size(type);
}

void accumulate(layer_accumulator & acc, fused_kind kind,
                uint64_t total_bytes, uint64_t slice_bytes,
                uint32_t experts) {
    switch (kind) {
        case fused_gate_up_exps:
        case fused_gate_up_ff:
            acc.has_gate_up = true;
            acc.gate_up_total_bytes = total_bytes;
            acc.gate_up_slice_bytes = slice_bytes;
            acc.gate_up_experts = experts;
            break;
        case fused_down_exps:
        case fused_down_ff:
            acc.has_down = true;
            acc.down_total_bytes = total_bytes;
            acc.down_slice_bytes = slice_bytes;
            acc.down_experts = experts;
            break;
        case fused_gate_exps:
            acc.has_gate = true;
            acc.gate_total_bytes = total_bytes;
            acc.gate_slice_bytes = slice_bytes;
            acc.gate_experts = experts;
            break;
        case fused_up_exps:
            acc.has_up = true;
            acc.up_total_bytes = total_bytes;
            acc.up_slice_bytes = slice_bytes;
            acc.up_experts = experts;
            break;
        default:
            break;
    }
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

    // Model-wide expert count from metadata, used when the tensor shapes do
    // not carry an expert axis.
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
        accumulate(accumulators[fused.layer_index], fused.kind,
                   gguf_get_tensor_size(ctx, i), tensor_slice_bytes(ctx, i),
                   tensor_expert_count(ctx, i));
    }
    gguf_free(ctx);

    for (size_t layer = 0; layer < accumulators.size(); ++layer) {
        const layer_accumulator & acc = accumulators[layer];
        if (!acc.has_gate_up && !acc.has_down && !acc.has_gate && !acc.has_up) {
            continue;
        }
        // All present tensor shapes must agree; disagreeing layers are
        // skipped, not guessed.
        const uint32_t shape_counts[4] = {
            acc.gate_up_experts, acc.down_experts, acc.gate_experts, acc.up_experts,
        };
        uint32_t expert_count = 0;
        bool consistent = true;
        for (const uint32_t count : shape_counts) {
            if (count == 0) {
                continue;
            }
            if (expert_count == 0) {
                expert_count = count;
            } else if (count != expert_count) {
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
        expert_layer entry;
        entry.layer_index = static_cast<uint32_t>(layer);
        entry.expert_count = expert_count;
        entry.gate_up_total_bytes = acc.gate_up_total_bytes;
        entry.down_total_bytes = acc.down_total_bytes;
        entry.gate_total_bytes = acc.gate_total_bytes;
        entry.up_total_bytes = acc.up_total_bytes;
        // Per-expert slice bytes: exact plane size for 3D expert tensors,
        // total divided by the expert count when the shape carries no
        // expert axis. A layer with an indivisible tensor is skipped.
        if (acc.has_gate_up) {
            entry.gate_up_bytes = acc.gate_up_slice_bytes != 0
                ? acc.gate_up_slice_bytes
                : (acc.gate_up_total_bytes % expert_count == 0
                    ? acc.gate_up_total_bytes / expert_count : 0);
            if (entry.gate_up_bytes == 0) {
                continue;
            }
        }
        if (acc.has_down) {
            entry.down_bytes = acc.down_slice_bytes != 0
                ? acc.down_slice_bytes
                : (acc.down_total_bytes % expert_count == 0
                    ? acc.down_total_bytes / expert_count : 0);
            if (entry.down_bytes == 0) {
                continue;
            }
        }
        if (acc.has_gate) {
            entry.gate_bytes = acc.gate_slice_bytes != 0
                ? acc.gate_slice_bytes
                : (acc.gate_total_bytes % expert_count == 0
                    ? acc.gate_total_bytes / expert_count : 0);
            if (entry.gate_bytes == 0) {
                continue;
            }
        }
        if (acc.has_up) {
            entry.up_bytes = acc.up_slice_bytes != 0
                ? acc.up_slice_bytes
                : (acc.up_total_bytes % expert_count == 0
                    ? acc.up_total_bytes / expert_count : 0);
            if (entry.up_bytes == 0) {
                continue;
            }
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