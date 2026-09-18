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

node_entry * find_slot_by_node_id(std::vector<node_entry> & nodes, const std::string & node_id) {
    for (node_entry & entry : nodes) {
        if (entry.active && entry.node_id == node_id) {
            return &entry;
        }
    }
    return nullptr;
}

node_entry * find_free_slot(std::vector<node_entry> & nodes) {
    for (node_entry & entry : nodes) {
        if (!entry.active) {
            return &entry;
        }
    }
    return nullptr;
}

// Receive one registration frame, record it in the persistent registry, and
// reply by echoing it back, so the acknowledgement reflects the registering
// node, not the coordinator. A registration with a known node_id reuses its
// registry entry and takes over its slot; the superseded connection closes.
node_entry * register_peer(node_runtime::tcp_transport && peer, std::vector<node_entry> & nodes,
                           node_runtime::node_registry & registry, uint64_t now_ms) {
    node_runtime::framed_message request;
    if (!node_runtime::receive_message(peer, request, registration_type, 5000)) {
        return nullptr;
    }
    node_runtime::registration_message registration;
    const std::string text(request.payload.begin(), request.payload.end());
    if (!node_runtime::registration_from_json(text, registration)) {
        return nullptr;
    }
    if (!registry.register_node(registration.capabilities, now_ms)) {
        std::printf("llama-node: registry full, rejecting node %s\n",
                    registration.capabilities.node_id.c_str());
        return nullptr;
    }
    node_entry * slot = find_slot_by_node_id(nodes, registration.capabilities.node_id);
    if (!slot) {
        slot = find_free_slot(nodes);
    }
    if (!slot) {
        std::printf("llama-node: no free slot, rejecting node %s\n",
                    registration.capabilities.node_id.c_str());
        return nullptr;
    }
    if (slot->active) {
        slot->transport.close();
    }
    slot->transport = std::move(peer);
    slot->registration = registration;
    slot->node_id = registration.capabilities.node_id;
    slot->active = true;
    if (!node_runtime::send_message(slot->transport, acknowledgement_type,
                                    request.payload.data(), request.payload.size())) {
        slot->active = false;
        slot->transport.close();
        return nullptr;
    }
    return slot;
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
    uint64_t clock_ms = 0;
    const auto count_active = [&nodes]() {
        size_t count = 0;
        for (const node_entry & entry : nodes) {
            count += entry.active ? 1 : 0;
        }
        return count;
    };
    // Keep accepting until every slot has a live connection; a duplicate
    // registration with a known node_id replaces its previous connection and
    // reuses its slot, so the number of active slots still governs.
    while (count_active() < node_count) {
        node_runtime::tcp_transport peer = listener.accept(once ? 5000 : 30000);
        if (!peer.valid()) {
            break;
        }
        // Registration order gives each node a distinct logical timestamp; the
        // coordinator has no wall clock dependency in this loop.
        ++clock_ms;
        node_entry * slot = register_peer(std::move(peer), nodes, registry, clock_ms);
        if (slot) {
            std::printf("llama-node: node %s registered on slot %ld\n",
                        slot->node_id.c_str(), static_cast<long>(slot - nodes.data()));
            std::fflush(stdout);
        }
    }
    if (count_active() == 0) {
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
            if (node_runtime::receive_message(entry.transport, heartbeat_frame, 0, 1000)) {
                if (heartbeat_frame.type == heartbeat_type &&
                    heartbeat_frame.payload.size() == sizeof(node_runtime::heartbeat_message)) {
                    node_runtime::heartbeat_message heartbeat{};
                    std::memcpy(&heartbeat, heartbeat_frame.payload.data(), sizeof(heartbeat));
                    ++clock_ms;
                    registry.observe_heartbeat(entry.node_id, heartbeat.sequence, clock_ms);
                    if (!node_runtime::send_message(entry.transport, heartbeat_type, &heartbeat, sizeof(heartbeat))) {
                        entry.active = false;
                        entry.transport.close();
                        std::printf("llama-node: node send failed\n");
                    }
                } else if (heartbeat_frame.type == node_runtime::departure_message_type) {
                    // Graceful departure: the node announced a clean stop.
                    ++clock_ms;
                    registry.mark_departed(entry.node_id, clock_ms);
                    entry.active = false;
                    entry.transport.close();
                    std::printf("llama-node: node %s departed cleanly\n", entry.node_id.c_str());
                } else {
                    // Unknown frame type on a live link is a protocol violation.
                    entry.active = false;
                    entry.transport.close();
                    std::printf("llama-node: node %s sent unexpected frame type %u\n",
                                entry.node_id.c_str(), heartbeat_frame.type);
                }
            } else if (registry.expire_stale(++clock_ms, 5) > 0) {
                // Registry entries turn stale instead of being dropped; the
                // connection is closed but the node stays registered.
                entry.active = false;
                entry.transport.close();
                std::printf("llama-node: node timed out (kept in registry)\n");
            }
        }
        // Accept reconnections. When no link is live, wait up to 1s here so
        // the coordinator parks on accept instead of spinning until deadline.
        node_runtime::tcp_transport peer = listener.accept(any_active ? 0 : 1000);
        while (peer.valid()) {
            ++clock_ms;
            node_entry * slot = register_peer(std::move(peer), nodes, registry, clock_ms);
            if (slot) {
                std::printf("llama-node: node %s reconnected (slot %ld)\n",
                            slot->node_id.c_str(), static_cast<long>(slot - nodes.data()));
                std::fflush(stdout);
            }
            peer = listener.accept(0);
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
