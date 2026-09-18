// Apple Silicon runtime telemetry and memory-tier metadata.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace apple_runtime {

enum class memory_tier : uint8_t {
    gpu,
    unified_ram,
    local_ssd,
    remote_ram,
    remote_ssd,
};

struct memory_tier_info {
    memory_tier tier;
    uint64_t capacity = 0;
    uint64_t available = 0;
    uint64_t latency_ns = 0;
    uint64_t bandwidth_bytes_per_second = 0;
    bool local = true;
};

struct system_telemetry {
    uint64_t physical_memory = 0;
    uint64_t free_memory = 0;
    uint64_t active_memory = 0;
    uint64_t wired_memory = 0;
    uint64_t compressed_memory = 0;
    uint64_t process_resident_memory = 0;
    uint64_t process_peak_resident_memory = 0;
    uint32_t logical_cpu_count = 0;
    bool apple_silicon = false;
    std::string chip;
};

constexpr size_t memory_tier_count = static_cast<size_t>(memory_tier::remote_ssd) + 1;

// These functions perform one bounded snapshot and do not start background work.
system_telemetry system_snapshot();
const char * memory_tier_name(memory_tier tier);

// Pressure thresholds as fractions of physical memory in [0, 1]. Wired and
// compressed ratios at or above a threshold raise the level; a free ratio
// strictly below the floor raises critical directly.
struct memory_pressure_config {
    double warning_wired_ratio = 0.66;
    double critical_wired_ratio = 0.80;
    double warning_compressed_ratio = 0.20;
    double critical_compressed_ratio = 0.32;
    double critical_free_ratio = 0.10;
};

enum class memory_pressure_level : uint8_t {
    normal,
    warning,
    critical,
};

struct memory_pressure_assessment {
    memory_pressure_level level = memory_pressure_level::normal;
    double wired_ratio = 0.0;
    double compressed_ratio = 0.0;
    double free_ratio = 0.0;
    double process_resident_ratio = 0.0;
};

// Classify one snapshot. Pure function of its inputs: no sysctl, no host
// calls, no threads. A zero-physical-memory snapshot classifies as normal.
memory_pressure_assessment assess_memory_pressure(const system_telemetry & telemetry,
                                                  const memory_pressure_config & config = {});

const char * memory_pressure_level_name(memory_pressure_level level);

// Caller-driven planned-reservation bookkeeping over the memory tiers. It
// tracks intent only: it allocates no real memory, starts nothing, and is
// not consulted by the inference path. Capacities are set explicitly per
// tier; reservations that would exceed a tier are rejected.
class memory_budget {
public:
    struct reservation {
        std::string label;
        memory_tier tier = memory_tier::unified_ram;
        uint64_t bytes = 0;
        uint64_t created_at_ms = 0;
    };

    explicit memory_budget(size_t max_reservations = 16);

    // Set one tier capacity. Returns false for an unknown tier.
    bool set_capacity(memory_tier tier, uint64_t bytes);
    // Reserve bytes on one tier. Returns false on empty label, zero bytes,
    // unknown tier, duplicate label, exhausted tier, or a full table.
    bool reserve(const std::string & label, memory_tier tier, uint64_t bytes, uint64_t now_ms = 0);
    // Release the first reservation with this label. Returns false when unknown.
    bool release(const std::string & label);

    uint64_t capacity(memory_tier tier) const;
    uint64_t reserved(memory_tier tier) const;
    uint64_t available(memory_tier tier) const;

    // Insertion-ordered reservation list.
    const std::vector<reservation> & reservations() const;
    size_t size() const;
    size_t max_reservations() const;

private:
    uint64_t capacity_[memory_tier_count] = {};
    uint64_t reserved_[memory_tier_count] = {};
    std::vector<reservation> reservations_;
    size_t max_reservations_;
};

} // namespace apple_runtime
