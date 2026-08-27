#include "apple-runtime.h"

#include <thread>

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

} // namespace apple_runtime
