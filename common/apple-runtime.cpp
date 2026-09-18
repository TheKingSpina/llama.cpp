#include "apple-runtime.h"

#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace apple_runtime {

namespace {

uint64_t sysctl_u64(const char * name) {
#if defined(__APPLE__)
    uint64_t value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
#else
    (void) name;
    return 0;
#endif
}

std::string sysctl_string(const char * name) {
#if defined(__APPLE__)
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
#else
    (void) name;
    return {};
#endif
}

} // namespace

system_telemetry system_snapshot() {
    system_telemetry result;
    result.logical_cpu_count = std::thread::hardware_concurrency();
#if defined(__APPLE__)
    result.physical_memory = sysctl_u64("hw.memsize");
    result.chip = sysctl_string("machdep.cpu.brand_string");
    result.apple_silicon = sysctl_u64("hw.optional.arm64") != 0;

    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm = {};
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS) {
        const uint64_t page_size = 4096;
        result.free_memory = static_cast<uint64_t>(vm.free_count + vm.inactive_count) * page_size;
        result.active_memory = static_cast<uint64_t>(vm.active_count) * page_size;
        result.wired_memory = static_cast<uint64_t>(vm.wire_count) * page_size;
        result.compressed_memory = static_cast<uint64_t>(vm.compressor_page_count) * page_size;
    }

    task_vm_info_data_t task = {};
    mach_msg_type_number_t task_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&task), &task_count) == KERN_SUCCESS) {
        result.process_resident_memory = task.resident_size;
        result.process_peak_resident_memory = task.resident_size_peak;
    }
#endif
    return result;
}

const char * memory_tier_name(memory_tier tier) {
    switch (tier) {
        case memory_tier::gpu:        return "gpu";
        case memory_tier::unified_ram: return "unified_ram";
        case memory_tier::local_ssd:  return "local_ssd";
        case memory_tier::remote_ram: return "remote_ram";
        case memory_tier::remote_ssd: return "remote_ssd";
    }
    return "unknown";
}

memory_pressure_assessment assess_memory_pressure(const system_telemetry & telemetry,
                                                  const memory_pressure_config & config) {
    memory_pressure_assessment result;
    if (telemetry.physical_memory == 0) {
        return result;
    }
    const double physical = static_cast<double>(telemetry.physical_memory);
    result.wired_ratio = static_cast<double>(telemetry.wired_memory) / physical;
    result.compressed_ratio = static_cast<double>(telemetry.compressed_memory) / physical;
    result.free_ratio = static_cast<double>(telemetry.free_memory) / physical;
    result.process_resident_ratio = static_cast<double>(telemetry.process_resident_memory) / physical;

    if (result.wired_ratio >= config.critical_wired_ratio ||
        result.compressed_ratio >= config.critical_compressed_ratio ||
        result.free_ratio < config.critical_free_ratio) {
        result.level = memory_pressure_level::critical;
    } else if (result.wired_ratio >= config.warning_wired_ratio ||
               result.compressed_ratio >= config.warning_compressed_ratio) {
        result.level = memory_pressure_level::warning;
    }
    return result;
}

const char * memory_pressure_level_name(memory_pressure_level level) {
    switch (level) {
        case memory_pressure_level::normal:   return "normal";
        case memory_pressure_level::warning:  return "warning";
        case memory_pressure_level::critical: return "critical";
    }
    return "unknown";
}

memory_budget::memory_budget(size_t max_reservations) : max_reservations_(max_reservations) {}

bool memory_budget::set_capacity(memory_tier tier, uint64_t bytes) {
    if (static_cast<uint8_t>(tier) >= memory_tier_count) {
        return false;
    }
    capacity_[static_cast<size_t>(tier)] = bytes;
    return true;
}

bool memory_budget::reserve(const std::string & label, memory_tier tier, uint64_t bytes, uint64_t now_ms) {
    if (label.empty() || bytes == 0 || static_cast<uint8_t>(tier) >= memory_tier_count) {
        return false;
    }
    for (const reservation & existing : reservations_) {
        if (existing.label == label) {
            return false;
        }
    }
    const size_t index = static_cast<size_t>(tier);
    if (bytes > capacity_[index] - reserved_[index]) {
        return false;
    }
    if (reservations_.size() >= max_reservations_) {
        return false;
    }
    reservation entry;
    entry.label = label;
    entry.tier = tier;
    entry.bytes = bytes;
    entry.created_at_ms = now_ms;
    reservations_.push_back(std::move(entry));
    reserved_[index] += bytes;
    return true;
}

bool memory_budget::release(const std::string & label) {
    for (auto it = reservations_.begin(); it != reservations_.end(); ++it) {
        if (it->label == label) {
            reserved_[static_cast<size_t>(it->tier)] -= it->bytes;
            reservations_.erase(it);
            return true;
        }
    }
    return false;
}

uint64_t memory_budget::capacity(memory_tier tier) const {
    return static_cast<uint8_t>(tier) < memory_tier_count ? capacity_[static_cast<size_t>(tier)] : 0;
}

uint64_t memory_budget::reserved(memory_tier tier) const {
    return static_cast<uint8_t>(tier) < memory_tier_count ? reserved_[static_cast<size_t>(tier)] : 0;
}

uint64_t memory_budget::available(memory_tier tier) const {
    return capacity(tier) - reserved(tier);
}

const std::vector<memory_budget::reservation> & memory_budget::reservations() const {
    return reservations_;
}

size_t memory_budget::size() const {
    return reservations_.size();
}

size_t memory_budget::max_reservations() const {
    return max_reservations_;
}

} // namespace apple_runtime
