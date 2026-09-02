#include "node-runtime.h"

#include <cstring>
#include <cstdint>
#include <arpa/inet.h>
#include <string>

namespace {
bool check(bool condition) {
    return condition;
}

struct synthetic_task_message {
    uint32_t protocol_version;
    uint32_t task_id;
    uint64_t compute_units;
};

struct synthetic_result_message {
    uint32_t protocol_version;
    uint32_t task_id;
    uint32_t worker_selected;
    uint64_t result;
};

struct wire_header {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t payload_size;
};

bool fault_connection(node_runtime::tcp_transport & listener, const wire_header & header,
                      size_t bytes, uint16_t expected_type, bool close_sender) {
    node_runtime::tcp_transport client;
    if (!check(client.connect("127.0.0.1", listener.port()))) {
        return false;
    }
    node_runtime::tcp_transport server = listener.accept();
    if (!check(server.valid()) || !check(client.send(&header, bytes))) {
        return false;
    }
    node_runtime::framed_message message;
    const bool received = node_runtime::receive_message(server, message, expected_type, 10);
    if (close_sender) {
        client.close();
    }
    return !received;
}
}

int main() {
    node_runtime::scheduler_task task{"task", 100.0, 1024, 10, 5};
    node_runtime::scheduler_candidate candidates[] = {
        {"slow", {}, 1.0, 2048, 0.1, 0.1},
        {"fast", {}, 2.0, 2048, 0.1, 0.1},
    };
    node_runtime::scheduler_score score;
    if (!check(node_runtime::scheduler_select_candidate(task, candidates, 2, &score) == 1) ||
        !check(score.candidate_id == "fast") ||
        !check(node_runtime::scheduler_select_candidate(task, nullptr, 0) == -1)) return 1;

    node_runtime::scheduler_candidate plan_candidates[] = {
        {"tie-b", {}, 1.0, 2048, 0.0, 0.0},
        {"infeasible", {}, 0.0, 512, 0.0, 0.0},
        {"tie-a", {}, 1.0, 2048, 0.0, 0.0},
    };
    const node_runtime::scheduler_plan_report plan =
        node_runtime::scheduler_plan(task, plan_candidates, 3);
    if (!check(plan.task_id == "task") || !check(plan.candidates.size() == 3) ||
        !check(plan.candidates[0].candidate_id == "tie-a") ||
        !check(plan.candidates[1].candidate_id == "tie-b") ||
        !check(plan.candidates[2].candidate_id == "infeasible") ||
        !check(plan.candidates[0].feasible) ||
        !check(!plan.candidates[2].feasible) ||
        !check(plan.candidates[0].compute_cost == 100.0) ||
        !check(plan.candidates[0].memory_cost == 0.5) ||
        !check(plan.candidates[0].network_cost == 0.0) ||
        !check(plan.candidates[0].storage_cost == 0.0)) return 1;

    node_runtime::registration_state registration = node_runtime::local_registration_state();
    if (!check(registration.message.protocol_version == 1) || !check(!registration.registered) ||
        !check(!node_runtime::registration_json(registration.message).empty())) {
        return 1;
    }

    node_runtime::tcp_transport listener;
    if (!check(listener.listen())) return 1;

    node_runtime::tcp_transport client;
    if (!check(client.connect("127.0.0.1", listener.port()))) return 1;
    node_runtime::tcp_transport server = listener.accept();
    if (!check(server.valid())) return 1;

    const node_runtime::registration_message worker_registration = node_runtime::local_registration();
    const std::string serialized_registration = node_runtime::registration_json(worker_registration);
    if (!check(node_runtime::send_message(client, 1, serialized_registration.data(), serialized_registration.size()))) return 1;
    node_runtime::framed_message received_registration;
    if (!check(node_runtime::receive_message(server, received_registration, 1, 1000)) ||
        !check(std::string(received_registration.payload.begin(), received_registration.payload.end()) == serialized_registration)) return 1;

    std::string oversized(node_runtime::message_max_payload + 1, 'x');
    if (!check(!node_runtime::send_message(client, 1, oversized.data(), oversized.size()))) return 1;
    if (!check(!node_runtime::send_message(client, 0, nullptr, 0))) return 1;

    const wire_header bad_version{htonl(node_runtime::message_magic), htons(999), htons(1), htonl(0)};
    if (!check(fault_connection(listener, bad_version, sizeof(bad_version), 1, true))) return 1;
    const wire_header valid_header{htonl(node_runtime::message_magic), htons(node_runtime::message_protocol_version), htons(1), htonl(0)};
    if (!check(fault_connection(listener, valid_header, sizeof(valid_header), 2, true))) return 1;
    if (!check(fault_connection(listener, valid_header, sizeof(valid_header) - 1, 1, true))) return 1;
    {
        node_runtime::tcp_transport timeout_client;
        if (!check(timeout_client.connect("127.0.0.1", listener.port()))) return 1;
        node_runtime::tcp_transport timeout_server = listener.accept();
        if (!check(timeout_server.valid())) return 1;
        node_runtime::framed_message timeout_message;
        if (!check(!node_runtime::receive_message(timeout_server, timeout_message, 1, 1))) return 1;
    }

    node_runtime::scheduler_candidate placement[] = {
        {"coordinator", {}, 1.0, 2048, 0.0, 0.0},
        {"worker", worker_registration.capabilities, 2.0, 2048, 0.001, 0.0},
    };
    node_runtime::scheduler_task distributed_task{"synthetic-task", 100.0, 1024, 0, 0};
    node_runtime::scheduler_score placement_score;
    if (!check(node_runtime::scheduler_select_candidate(distributed_task, placement, 2, &placement_score) == 1) ||
        !check(placement_score.candidate_id == "worker")) return 1;

    const synthetic_task_message task_message{1, 7, 100};
    if (!check(node_runtime::send_message(client, 2, &task_message, sizeof(task_message)))) return 1;
    node_runtime::framed_message received_task_frame;
    if (!check(node_runtime::receive_message(server, received_task_frame, 2, 1000)) ||
        !check(received_task_frame.payload.size() == sizeof(task_message))) return 1;
    synthetic_task_message received_task{};
    std::memcpy(&received_task, received_task_frame.payload.data(), sizeof(received_task));
    if (!check(received_task.protocol_version == 1) || !check(received_task.task_id == 7)) return 1;
    const synthetic_result_message result_message{1, received_task.task_id, 1, received_task.compute_units * 2};
    if (!check(node_runtime::send_message(server, 3, &result_message, sizeof(result_message)))) return 1;
    node_runtime::framed_message received_result_frame;
    if (!check(node_runtime::receive_message(client, received_result_frame, 3, 1000)) ||
        !check(received_result_frame.payload.size() == sizeof(result_message))) return 1;
    synthetic_result_message received_result{};
    std::memcpy(&received_result, received_result_frame.payload.data(), sizeof(received_result));
    if (!check(received_result.protocol_version == 1) ||
        !check(received_result.worker_selected == 1) || !check(received_result.result == 200)) return 1;

    node_runtime::set_registered(registration, true);
    if (!check(registration.registered)) return 1;
    node_runtime::node_lifecycle sender({10, 30});
    const node_runtime::heartbeat_message heartbeat = sender.heartbeat(100);
    if (!check(node_runtime::send_message(client, 4, &heartbeat, sizeof(heartbeat)))) return 1;

    node_runtime::framed_message received_heartbeat;
    if (!check(node_runtime::receive_message(server, received_heartbeat, 4, 1000)) ||
        !check(received_heartbeat.payload.size() == sizeof(heartbeat))) return 1;
    node_runtime::heartbeat_message received{};
    std::memcpy(&received, received_heartbeat.payload.data(), sizeof(received));
    if (
        !check(received.protocol_version == heartbeat.protocol_version) ||
        !check(received.sequence == heartbeat.sequence)) return 1;

    node_runtime::node_lifecycle lifecycle({10, 30});
    lifecycle.observe_heartbeat(received, 100);
    if (!check(lifecycle.state() == node_runtime::lifecycle_state::running) ||
        !check(lifecycle.heartbeat_due(110)) || !check(lifecycle.timed_out(130)) ||
        !check(lifecycle.state() == node_runtime::lifecycle_state::timed_out)) return 1;
    lifecycle.request_shutdown();
    if (!check(lifecycle.state() == node_runtime::lifecycle_state::timed_out)) return 1;
    lifecycle.heartbeat(140);
    lifecycle.request_shutdown();
    if (!check(lifecycle.state() == node_runtime::lifecycle_state::shutdown_requested)) return 1;
    lifecycle.mark_stopped();
    if (!check(lifecycle.state() == node_runtime::lifecycle_state::stopped)) return 1;
    node_runtime::node_lifecycle worker_lifecycle;
    worker_lifecycle.request_shutdown();
    worker_lifecycle.mark_stopped();
    if (!check(worker_lifecycle.state() == node_runtime::lifecycle_state::stopped)) return 1;

    const char payload[] = "node-runtime-selftest";
    if (!check(node_runtime::send_message(client, 5, payload, sizeof(payload)))) return 1;
    node_runtime::framed_message received_payload;
    if (!check(node_runtime::receive_message(server, received_payload, 5, 1000)) ||
        !check(received_payload.payload.size() == sizeof(payload)) ||
        !check(std::memcmp(payload, received_payload.payload.data(), sizeof(payload)) == 0)) return 1;

    node_runtime::node_registry registry(2);
    node_runtime::capabilities caps_a = worker_registration.capabilities;
    caps_a.node_id = "node-a";
    node_runtime::capabilities caps_b = worker_registration.capabilities;
    caps_b.node_id = "node-b";
    node_runtime::capabilities caps_c = worker_registration.capabilities;
    caps_c.node_id = "node-c";
    if (!check(registry.register_node(caps_a, 10)) || !check(registry.register_node(caps_b, 20)) ||
        !check(!registry.register_node(caps_c, 30)) || !check(registry.size() == 2)) return 1;

    registry.observe_heartbeat("node-a", 1, 15);
    registry.observe_heartbeat("node-a", 2, 25);
    registry.observe_heartbeat("node-b", 1, 20);
    registry.observe_heartbeat("unknown", 1, 25);
    registry.observe_heartbeat("node-a", 0, 25);
    const node_runtime::node_registry_entry * found_a = registry.find("node-a");
    if (!check(found_a != nullptr) || !check(found_a->heartbeats == 2) ||
        !check(found_a->heartbeat_sequence == 2) || !check(found_a->last_heartbeat_ms == 25) ||
        !check(found_a->state == node_runtime::lifecycle_state::running)) return 1;

    if (!check(registry.expire_stale(60, 30) == 2) ||
        !check(registry.count_state(node_runtime::lifecycle_state::timed_out) == 2) ||
        !check(registry.find("node-a")->state == node_runtime::lifecycle_state::timed_out)) return 1;
    // A heartbeat after expiry recovers the entry instead of re-registering.
    registry.observe_heartbeat("node-b", 2, 70);
    if (!check(registry.find("node-b")->state == node_runtime::lifecycle_state::running) ||
        !check(registry.count_state(node_runtime::lifecycle_state::timed_out) == 1)) return 1;

    // Re-registration with the same node_id reuses the entry.
    if (!check(registry.register_node(caps_a, 80)) ||
        !check(registry.size() == 2) ||
        !check(registry.find("node-a")->state == node_runtime::lifecycle_state::running) ||
        !check(registry.find("node-a")->reconnections == 1) ||
        !check(registry.find("node-a")->registered_at_ms == 80)) return 1;

    const std::string registry_json = node_runtime::node_registry_json(registry);
    if (!check(registry_json.find("\"node_id\":\"node-a\"") != std::string::npos) ||
        !check(registry_json.find("\"reconnections\":1") != std::string::npos)) return 1;

    // Wire-level reconnection: a second connection registering the same
    // node_id must reuse the registry entry and bump reconnections again,
    // mirroring what the coordinator does with a live reconnecting client.
    {
        node_runtime::tcp_transport reconnect_listener;
        if (!check(reconnect_listener.listen())) return 1;
        node_runtime::tcp_transport first_client;
        if (!check(first_client.connect("127.0.0.1", reconnect_listener.port()))) return 1;
        node_runtime::tcp_transport first_server = reconnect_listener.accept();
        if (!check(first_server.valid())) return 1;
        const std::string first_payload = node_runtime::registration_json(
            node_runtime::registration_message{1, caps_a});
        if (!check(node_runtime::send_message(first_client, 1, first_payload.data(), first_payload.size()))) return 1;
        node_runtime::framed_message first_request;
        if (!check(node_runtime::receive_message(first_server, first_request, 1, 1000))) return 1;

        node_runtime::node_registry reconnect_registry(2);
        if (!check(reconnect_registry.register_node(caps_a, 100))) return 1;
        reconnect_registry.observe_heartbeat("node-a", 1, 110);

        // The first link dies; the client reconnects with the same node_id.
        first_client.close();
        first_server.close();
        node_runtime::tcp_transport second_client;
        if (!check(second_client.connect("127.0.0.1", reconnect_listener.port()))) return 1;
        node_runtime::tcp_transport second_server = reconnect_listener.accept();
        if (!check(second_server.valid())) return 1;
        if (!check(node_runtime::send_message(second_client, 1, first_payload.data(), first_payload.size()))) return 1;
        node_runtime::framed_message second_request;
        if (!check(node_runtime::receive_message(second_server, second_request, 1, 1000))) return 1;
        node_runtime::registration_message second_registration;
        const std::string second_text(second_request.payload.begin(), second_request.payload.end());
        if (!check(node_runtime::registration_from_json(second_text, second_registration))) return 1;
        if (!check(reconnect_registry.register_node(second_registration.capabilities, 130))) return 1;
        if (!check(reconnect_registry.size() == 1) ||
            !check(reconnect_registry.find("node-a")->reconnections == 1) ||
            !check(reconnect_registry.find("node-a")->state == node_runtime::lifecycle_state::running)) return 1;
        // Heartbeats resume on the new link.
        reconnect_registry.observe_heartbeat("node-a", 2, 140);
        if (!check(reconnect_registry.find("node-a")->heartbeats == 2)) return 1;
    }

    // Heartbeats keep flowing for known nodes through the registry path.
    if (!check(node_runtime::send_message(client, 4, &heartbeat, sizeof(heartbeat)))) return 1;
    if (!check(node_runtime::receive_message(server, received_heartbeat, 4, 1000))) return 1;
    return 0;
}