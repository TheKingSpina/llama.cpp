// Apple Silicon runtime telemetry and memory-tier metadata.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

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

// These functions perform one bounded snapshot and do not start background work.
system_telemetry system_snapshot();
const char * memory_tier_name(memory_tier tier);

} // namespace apple_runtime
