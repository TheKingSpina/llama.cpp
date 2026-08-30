#include "node-runtime.h"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint16_t registration_type = 1;
constexpr uint16_t acknowledgement_type = 2;
constexpr uint16_t heartbeat_type = 4;
constexpr unsigned max_nodes = 8;

bool parse_port(const char * text, uint16_t & port) {
    unsigned value = 0;
    const char * end = text + std::char_traits<char>::length(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc() || result.ptr != end || value > 65535) {
        return false;
    }
    port = static_cast<uint16_t>(value);
    return true;
}

void usage(const char * name) {
    std::fprintf(stderr, "usage: %s [--bind ADDRESS] [--port PORT] [--once] [--monitor SECONDS] [--nodes COUNT]\n", name);
}

struct node_entry {
    node_runtime::tcp_transport transport;
    node_runtime::node_lifecycle lifecycle;
    node_runtime::registration_message registration;
    uint64_t heartbeats = 0;
    bool active = false;
};

// Receive one registration frame and reply by echoing it back, so the
// acknowledgement reflects the registering node, not the coordinator.
bool register_peer(node_runtime::tcp_transport & peer, node_entry & entry) {
    node_runtime::framed_message request;
    if (!node_runtime::receive_message(peer, request, registration_type, 5000)) {
        return false;
    }
    entry.registration = node_runtime::local_registration();
    if (!node_runtime::send_message(peer, acknowledgement_type,
                                    request.payload.data(), request.payload.size())) {
        return false;
    }
    entry.active = true;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    uint16_t port = 0;
    std::string bind_address = "127.0.0.1";
    bool once = false;
    unsigned monitor_seconds = 0;
    unsigned node_count = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--once") {
            once = true;
        } else if (argument == "--bind" && i + 1 < argc) {
            bind_address = argv[++i];
        } else if (argument == "--monitor" && i + 1 < argc) {
            try {
                monitor_seconds = std::stoul(argv[++i]);
            } catch (...) {
                usage(argv[0]);
                return 2;
            }
            if (monitor_seconds == 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--nodes" && i + 1 < argc) {
            try {
                node_count = std::stoul(argv[++i]);
            } catch (...) {
                usage(argv[0]);
                return 2;
            }
            if (node_count == 0 || node_count > max_nodes) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--port" && i + 1 < argc && parse_port(argv[++i], port)) {
            continue;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    node_runtime::tcp_transport listener;
    if (!listener.listen(port, bind_address)) {
        std::fprintf(stderr, "llama-node: failed to listen on %s\n", bind_address.c_str());
        return 1;
    }
    std::printf("llama-node listening on %s:%u, nodes=%u\n", bind_address.c_str(), listener.port(), node_count);
    std::fflush(stdout);

    std::vector<node_entry> nodes(node_count);
    unsigned registered = 0;
    for (unsigned i = 0; i < node_count; ++i) {
        node_runtime::tcp_transport peer = listener.accept(once ? 5000 : 30000);
        if (!peer.valid()) {
            break;
        }
        if (register_peer(peer, nodes[i])) {
            nodes[i].transport = std::move(peer);
            ++registered;
            std::printf("llama-node: node %u registered\n", i + 1);
            std::fflush(stdout);
        }
    }
    if (registered == 0) {
        return once ? 1 : 0;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(monitor_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        bool any_active = false;
        for (node_entry & entry : nodes) {
            if (!entry.active) {
                continue;
            }
            any_active = true;
            node_runtime::framed_message heartbeat_frame;
            // A receive timeout is not a disconnect; the lifecycle timeout decides.
            if (node_runtime::receive_message(entry.transport, heartbeat_frame, heartbeat_type, 1000) &&
                heartbeat_frame.payload.size() == sizeof(node_runtime::heartbeat_message)) {
                node_runtime::heartbeat_message heartbeat{};
                std::memcpy(&heartbeat, heartbeat_frame.payload.data(), sizeof(heartbeat));
                entry.lifecycle.observe_heartbeat(heartbeat, 1);
                if (!node_runtime::send_message(entry.transport, heartbeat_type, &heartbeat, sizeof(heartbeat))) {
                    entry.active = false;
                    entry.transport.close();
                    std::printf("llama-node: node send failed\n");
                    continue;
                }
                ++entry.heartbeats;
            } else if (entry.lifecycle.timed_out(1)) {
                entry.active = false;
                entry.transport.close();
                std::printf("llama-node: node timed out\n");
            }
        }
        if (!any_active) {
            break;
        }
    }
    for (const node_entry & entry : nodes) {
        if (entry.active) {
            std::printf("node %s: registered, %llu heartbeats, ok\n",
                        entry.registration.capabilities.node_id.c_str(),
                        static_cast<unsigned long long>(entry.heartbeats));
        }
    }
    // Machine-readable summary for the next distributed phase.
    std::printf("node_summary=[");
    bool first = true;
    for (const node_entry & entry : nodes) {
        if (!entry.active) {
            continue;
        }
        if (!first) {
            std::printf(",");
        }
        first = false;
        const node_runtime::capabilities & cap = entry.registration.capabilities;
        std::printf("{\"node_id\":\"%s\",\"chip\":\"%s\",\"physical_memory\":%llu,"
                    "\"logical_cpu_count\":%u,\"heartbeats\":%llu}",
                    cap.node_id.c_str(), cap.chip.c_str(),
                    static_cast<unsigned long long>(cap.physical_memory),
                    cap.logical_cpu_count,
                    static_cast<unsigned long long>(entry.heartbeats));
    }
    std::printf("]\n");
    for (node_entry & entry : nodes) {
        if (entry.active) {
            entry.lifecycle.request_shutdown();
            entry.lifecycle.mark_stopped();
            entry.transport.close();
        }
    }
    std::printf("llama-node heartbeat monitor completed\n");
    return 0;
}
