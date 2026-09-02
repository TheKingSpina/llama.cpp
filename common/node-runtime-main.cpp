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
    node_runtime::registration_message registration;
    bool active = false;
    // Index into the coordinator registry for this connection.
    std::string node_id;
};

// Receive one registration frame, record it in the persistent registry, and
// reply by echoing it back, so the acknowledgement reflects the registering
// node, not the coordinator.
bool register_peer(node_runtime::tcp_transport & peer, node_entry & entry,
                   node_runtime::node_registry & registry, uint64_t now_ms) {
    node_runtime::framed_message request;
    if (!node_runtime::receive_message(peer, request, registration_type, 5000)) {
        return false;
    }
    const std::string text(request.payload.begin(), request.payload.end());
    if (!node_runtime::registration_from_json(text, entry.registration)) {
        return false;
    }
    if (!registry.register_node(entry.registration.capabilities, now_ms)) {
        std::printf("llama-node: registry full, rejecting node %s\n",
                    entry.registration.capabilities.node_id.c_str());
        return false;
    }
    entry.node_id = entry.registration.capabilities.node_id;
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

    node_runtime::node_registry registry(node_count);
    std::vector<node_entry> nodes(node_count);
    unsigned registered = 0;
    uint64_t clock_ms = 0;
    for (unsigned i = 0; i < node_count; ++i) {
        node_runtime::tcp_transport peer = listener.accept(once ? 5000 : 30000);
        if (!peer.valid()) {
            break;
        }
        // Registration order gives each node a distinct logical timestamp; the
        // coordinator has no wall clock dependency in this loop.
        ++clock_ms;
        if (register_peer(peer, nodes[i], registry, clock_ms)) {
            nodes[i].transport = std::move(peer);
            ++registered;
            std::printf("llama-node: node %u registered (%s)\n", i + 1,
                        nodes[i].node_id.c_str());
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
                ++clock_ms;
                registry.observe_heartbeat(entry.node_id, heartbeat.sequence, clock_ms);
                if (!node_runtime::send_message(entry.transport, heartbeat_type, &heartbeat, sizeof(heartbeat))) {
                    entry.active = false;
                    entry.transport.close();
                    std::printf("llama-node: node send failed\n");
                    continue;
                }
            } else if (registry.expire_stale(++clock_ms, 5) > 0) {
                // Registry entries turn stale instead of being dropped; the
                // connection is closed but the node stays registered.
                entry.active = false;
                entry.transport.close();
                std::printf("llama-node: node timed out (kept in registry)\n");
            }
        }
        if (!any_active) {
            break;
        }
    }
    const std::vector<node_runtime::node_registry_entry> entries = registry.entries();
    for (const node_runtime::node_registry_entry & item : entries) {
        std::printf("node %s: state=%d, %llu heartbeats, reconnections=%llu\n",
                    item.capabilities.node_id.c_str(), static_cast<int>(item.state),
                    static_cast<unsigned long long>(item.heartbeats),
                    static_cast<unsigned long long>(item.reconnections));
    }
    // Machine-readable summary for the next distributed phase.
    std::printf("node_registry=%s\n", node_runtime::node_registry_json(registry).c_str());
    for (node_entry & entry : nodes) {
        if (entry.active) {
            entry.transport.close();
        }
    }
    std::printf("llama-node heartbeat monitor completed\n");
    return 0;
}
