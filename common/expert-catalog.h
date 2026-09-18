// Experimental MoE expert catalog. Reads GGUF metadata only and builds a
// layer list with per-expert byte sizes plus an experimental placement table.
// It loads no tensor data, no model weights, and starts no threads.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace expert_catalog {

// One MoE layer as listed in the GGUF tensor directory.
struct expert_layer {
    uint32_t layer_index = 0;
    uint32_t expert_count = 0;
    // Byte size of one expert slice in each expert tensor of the layer.
    // For a fused tensor [n_embd, n_ff, n_expert] this is
    // tensor_size / expert_count. Separate gate and up tensors report the
    // same slice size in each field.
    uint64_t gate_up_bytes = 0;
    uint64_t down_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t up_bytes = 0;
    // Full tensor sizes for the layer.
    uint64_t gate_up_total_bytes = 0;
    uint64_t down_total_bytes = 0;
    uint64_t gate_total_bytes = 0;
    uint64_t up_total_bytes = 0;
    // Absolute file offset of the expert 0 plane in each tensor; 0 when
    // the tensor does not exist or has no sliceable expert axis.
    uint64_t gate_up_data_offset = 0;
    uint64_t down_data_offset = 0;
    uint64_t gate_data_offset = 0;
    uint64_t up_data_offset = 0;
    // Per-expert total, gate_up plus down when both exist.
    uint64_t expert_bytes() const {
        return gate_up_bytes + down_bytes + gate_bytes + up_bytes;
    }
};

struct catalog_summary {
    uint32_t layer_count = 0;
    uint32_t expert_count = 0;
    uint64_t experts_total_bytes = 0;
    std::string architecture;
};

// Format version for the placement JSON report.
constexpr uint32_t placement_protocol_version = 1;

// One expert-to-node assignment. This is bookkeeping only; it does not move
// tensors and is not connected to inference.
struct expert_placement {
    uint32_t layer_index = 0;
    uint32_t expert_index = 0;
    std::string node_id;
    uint64_t bytes = 0;
};

// Per-node totals derived from a placement table.
struct node_placement_summary {
    std::string node_id;
    uint32_t experts = 0;
    uint64_t bytes = 0;
};

class expert_catalog {
public:
    // Scan one GGUF file. Returns false when the file cannot be read or
    // contains no fused expert tensors. Metadata is fully read into memory;
    // tensor data is never loaded (no_alloc init).
    bool load_from_gguf(const std::string & path);

    bool valid() const {
        return !layers_.empty();
    }

    const catalog_summary & summary() const {
        return summary_;
    }

    const std::vector<expert_layer> & layers() const {
        return layers_;
    }

    // Layer lookup by index; nullptr when the layer is not an MoE layer.
    const expert_layer * find_layer(uint32_t layer_index) const;

    // Byte size of one expert slice in the given layer; 0 when unknown.
    uint64_t expert_bytes(uint32_t layer_index) const;

    // Compact JSON catalog report for diagnostics and registration messages.
    std::string json() const;

private:
    catalog_summary summary_;
    std::vector<expert_layer> layers_;
};

// Experimental expert-to-node placement. Assignments are added explicitly by
// the caller. The class validates layer and expert bounds and keeps totals
// per node. No network, thread, or inference side effects.
class expert_placement_table {
public:
    explicit expert_placement_table(const expert_catalog & catalog,
                                    size_t max_entries = 4096);

    // Assign one expert to a node. Re-assigning the same layer and expert
    // replaces the previous node. Returns false on out-of-bounds indices,
    // empty node ids, or when the table is full.
    bool assign(uint32_t layer_index, uint32_t expert_index,
                const std::string & node_id);

    // Look up the node holding one expert; empty string when unassigned.
    const std::string & node_for(uint32_t layer_index,
                                 uint32_t expert_index) const;

    // Remove one assignment; returns false when it does not exist.
    bool unassign(uint32_t layer_index, uint32_t expert_index);

    size_t size() const {
        return entries_.size();
    }

    size_t capacity() const {
        return max_entries_;
    }

    const std::vector<expert_placement> & entries() const {
        return entries_;
    }

    // Per-node expert counts and byte totals in deterministic first-seen
    // node order.
    std::vector<node_placement_summary> node_summaries() const;

    // Compact JSON placement report.
    std::string json() const;

private:
    const expert_catalog & catalog_;
    size_t max_entries_;
    std::vector<expert_placement> entries_;
};

// One loaded expert slice: the per-tensor planes read from the file.
struct loaded_expert {
    std::vector<uint8_t> gate_up;
    std::vector<uint8_t> down;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
};

// Experimental selective expert reader. Reads one expert's byte planes
// from the GGUF file at computed offsets. Synchronous and caller-driven:
// no threads, no caching, no inference integration. The offsets come from
// the catalog, which must have been built from the same file.
class expert_reader {
public:
    expert_reader() = default;
    ~expert_reader();

    expert_reader(const expert_reader &) = delete;
    expert_reader & operator=(const expert_reader &) = delete;

    // Open the file for slice reads. Returns false when the file cannot
    // be opened.
    bool open(const std::string & path);
    void close();
    bool valid() const;

    // Read one expert slice of the given layer into out. Returns false on
    // unknown layer, out-of-bounds expert, invalid reader, or short read.
    bool read_expert(const expert_catalog & catalog, uint32_t layer_index,
                     uint32_t expert_index, loaded_expert & out);

private:
    std::FILE * file_ = nullptr;
};

struct cache_stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_loaded = 0;
    uint64_t bytes_served = 0;
    // Hits over hits plus misses; 0.0 when no request was made.
    double hit_rate() const;
};

// Experimental LRU cache over expert slices. One reader is bound at
// construction. serve() returns a pointer to the resident planes: valid
// until the next serve, eviction, or clear. With zero capacity every
// request is a miss and the planes live in an internal scratch buffer.
// Synchronous, caller-driven, no threads, no prefetch, no inference
// integration.
class expert_cache {
public:
    expert_cache(const expert_catalog & catalog, expert_reader & reader,
                 uint64_t max_bytes,
                 size_t max_entries = 4096);

    // Serve one expert. On hit, out points at the cached planes. On miss,
    // the planes are read from the reader, cached, and evicted LRU as
    // needed. Returns false on unknown layer or expert, invalid reader,
    // or read failure. An expert larger than the whole capacity is served
    // without being cached.
    bool serve(uint32_t layer_index, uint32_t expert_index,
               const loaded_expert *& out);

    // Drop all resident entries. Statistics are cumulative and survive.
    void clear();

    size_t resident_entries() const;
    uint64_t resident_bytes() const;
    uint64_t capacity_bytes() const;
    const cache_stats & stats() const;

    // Compact JSON statistics report.
    std::string stats_json() const;

private:
    struct cache_entry {
        uint32_t layer_index = 0;
        uint32_t expert_index = 0;
        loaded_expert planes;
        uint64_t bytes = 0;
    };

    bool insert_and_serve(uint32_t layer_index,
                          uint32_t expert_index, const loaded_expert *& out);
    void evict_until_fits(uint64_t incoming_bytes);

    const expert_catalog & catalog_;
    expert_reader & reader_;
    uint64_t max_bytes_;
    size_t max_entries_;
    std::list<cache_entry> lru_;
    std::unordered_map<uint64_t, std::list<cache_entry>::iterator> index_;
    uint64_t resident_bytes_ = 0;
    loaded_expert scratch_;
    cache_stats stats_;
};

// Experimental single-layer MoE runner over the cache. The caller supplies
// the router decision (expert ids) and the activation input; the runner
// serves each needed expert from the cache, gathers the per-expert weight
// planes into a dense 3D tensor layout (expert e at e * plane bytes), and
// exposes the two ggml_mul_mat_id graphs (gate/up gather, down project).
// This mirrors what an inference integration would do per layer. No
// router, no tokenizer, no threads.
class expert_moe_runner {
public:
    expert_moe_runner(expert_cache & cache, uint32_t layer_index);

    // Serve the experts named by ids (n_ids values) into out. The out
    // pointers stay valid until the next gather or cache eviction, so the
    // caller must consume them before serving other experts.
    bool gather(const uint32_t * ids, size_t n_ids);

    // Number of unique experts served by the last gather, in serve order.
    size_t served_count() const;
    // The expert id served into the given slot.
    uint32_t served_id(size_t slot) const;

    const loaded_expert & planes(size_t slot) const;

    uint32_t layer_index() const;

private:
    expert_cache & cache_;
    uint32_t layer_;
    std::vector<loaded_expert> slots_;
    std::vector<uint32_t> served_;
};

} // namespace expert_catalog
