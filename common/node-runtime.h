// Standalone node identity and capability reporting.

#pragma once

#include <cstdint>
#include <string>
#include <cstddef>
#include <vector>

namespace node_runtime {

struct capabilities {
    std::string node_id;
    std::string hostname;
    std::string chip;
    uint64_t physical_memory = 0;
    uint32_t logical_cpu_count = 0;
    bool apple_silicon = false;
    bool unified_memory = false;
};

// Artificial scheduler inputs. These types do not describe ggml tensors.
struct scheduler_task {
    std::string task_id;
    double compute_units = 0.0;
    uint64_t memory_bytes = 0;
    uint64_t network_bytes = 0;
    uint64_t storage_bytes = 0;
};

struct scheduler_candidate {
    std::string candidate_id;
    capabilities capabilities;
    double compute_capacity = 1.0;
    uint64_t available_memory_bytes = 0;
    double network_penalty = 0.0;
    double storage_penalty = 0.0;
};

struct scheduler_score {
    std::string candidate_id;
    double cost = 0.0;
    double compute_cost = 0.0;
    double memory_cost = 0.0;
    double network_cost = 0.0;
    double storage_cost = 0.0;
    bool feasible = false;
};

struct scheduler_plan_report {
    std::string task_id;
    std::vector<scheduler_score> candidates;
};

// Score and select artificial tasks only. Lower cost wins; ties use candidate_id.
scheduler_score scheduler_score_candidate(const scheduler_task & task,
                                           const scheduler_candidate & candidate);
// Return all candidates in deterministic order: feasible candidates first,
// then lower cost and candidate ID. This only evaluates artificial inputs.
scheduler_plan_report scheduler_plan(const scheduler_task & task,
                                     const scheduler_candidate * candidates,
                                     size_t candidate_count);
int scheduler_select_candidate(const scheduler_task & task,
                               const scheduler_candidate * candidates,
                               size_t candidate_count,
                               scheduler_score * selected_score = nullptr);

struct registration_message {
    uint32_t protocol_version = 1;
    capabilities capabilities;
};

struct registration_state {
    registration_message message;
    bool registered = false;
};

struct heartbeat_message {
    uint32_t protocol_version = 1;
    uint64_t sequence = 0;
    uint64_t sent_at_ms = 0;
};

enum class lifecycle_state : uint8_t {
    running,
    shutdown_requested,
    stopped,
    timed_out,
};

struct lifecycle_config {
    uint64_t heartbeat_interval_ms = 1000;
    uint64_t heartbeat_timeout_ms = 5000;
};

// Caller-driven node lifecycle state. Time is supplied by the caller so this
// type does not start a clock, thread, or event loop.
class node_lifecycle {
public:
    explicit node_lifecycle(lifecycle_config config = {});

    heartbeat_message heartbeat(uint64_t now_ms);
    void observe_heartbeat(const heartbeat_message & message, uint64_t now_ms);
    bool heartbeat_due(uint64_t now_ms) const;
    bool timed_out(uint64_t now_ms);
    void request_shutdown();
    void mark_stopped();

    lifecycle_state state() const;
    uint64_t last_heartbeat_ms() const;
    uint64_t heartbeat_sequence() const;

private:
    lifecycle_config config_;
    lifecycle_state state_ = lifecycle_state::running;
    uint64_t last_heartbeat_ms_ = 0;
    uint64_t next_heartbeat_ms_ = 0;
    uint64_t heartbeat_sequence_ = 0;
};

// Persistent coordinator-side registry. Entries survive disconnects: a node
// that reconnects with the same node_id reuses its entry, and timed-out nodes
// are marked stale instead of being removed. No threads or timers are used;
// the coordinator drives updates with caller-supplied timestamps.
struct node_registry_entry {
    capabilities capabilities;
    lifecycle_state state = lifecycle_state::running;
    uint64_t last_heartbeat_ms = 0;
    uint64_t heartbeat_sequence = 0;
    uint64_t heartbeats = 0;
    uint64_t registered_at_ms = 0;
    uint64_t reconnections = 0;
};

class node_registry {
public:
    explicit node_registry(size_t max_nodes = 8);

    // Register or re-register a node. Returns false when the registry is full
    // and the node_id is not already known.
    bool register_node(const capabilities & capabilities, uint64_t now_ms);
    // Record a received heartbeat for a known node.
    void observe_heartbeat(const std::string & node_id, uint64_t sequence, uint64_t now_ms);
    // Mark a node as deliberately stopped. Returns false when unknown.
    bool mark_departed(const std::string & node_id, uint64_t now_ms);
    // Mark nodes past the heartbeat timeout as timed_out. Returns how many
    // entries transitioned on this call.
    size_t expire_stale(uint64_t now_ms, uint64_t timeout_ms);
    // Find an entry by node id; nullptr when unknown.
    node_registry_entry * find(const std::string & node_id);
    const node_registry_entry * find(const std::string & node_id) const;

    std::vector<node_registry_entry> entries() const;
    size_t size() const;
    size_t capacity() const;
    // Count entries in the given lifecycle state.
    size_t count_state(lifecycle_state state) const;

private:
    size_t max_nodes_;
    std::vector<node_registry_entry> entries_;
};

// Compact JSON array report for diagnostics and the next distributed phase.
std::string node_registry_json(const node_registry & registry);

// Collect one bounded local snapshot. This does not start background work.
capabilities local_capabilities();

// Return a compact JSON report suitable for a registration or diagnostic message.
std::string capabilities_json(const capabilities & value);

// Create a local registration message without performing I/O.
registration_message local_registration();

// Create local registration state. The state changes only when explicitly updated.
registration_state local_registration_state();

void set_registered(registration_state & state, bool registered);

// Return a compact JSON registration message.
std::string registration_json(const registration_message & value);

// Parse a JSON registration message. Returns false on malformed input.
bool registration_from_json(const std::string & text, registration_message & out);

// Synchronous bounded TCP transport for local experiments. It binds only to
// loopback and does not create threads or own an event loop.
class tcp_transport {
public:
    tcp_transport() = default;
    ~tcp_transport();

    tcp_transport(const tcp_transport &) = delete;
    tcp_transport & operator=(const tcp_transport &) = delete;
    tcp_transport(tcp_transport && other) noexcept;
    tcp_transport & operator=(tcp_transport && other) noexcept;

    bool listen(uint16_t port = 0, const std::string & bind_address = "127.0.0.1");
    tcp_transport accept(int timeout_ms = -1) const;
    bool connect(const std::string & host, uint16_t port);
    bool send(const void * data, size_t size) const;
    // Receive up to capacity bytes. timeout_ms < 0 waits indefinitely.
    // A zero return means timeout, orderly close, or an error.
    size_t receive(void * data, size_t capacity, int timeout_ms = -1) const;
    void close();
    bool valid() const;
    uint16_t port() const;

private:
    explicit tcp_transport(int socket, uint16_t port);
    int socket_ = -1;
    uint16_t port_ = 0;
};

constexpr uint32_t message_magic = 0x4e52544d; // "NRTM"
constexpr uint16_t message_protocol_version = 1;
constexpr size_t message_max_payload = 4096;
constexpr uint16_t departure_message_type = 6;

struct framed_message {
    uint16_t type = 0;
    std::vector<uint8_t> payload;
};

// Send and receive one complete, versioned frame. The receiver validates the
// magic, version, type, and payload limit before allocating the payload.
bool send_message(const tcp_transport & transport, uint16_t type,
                  const void * payload, size_t payload_size,
                  size_t max_payload = message_max_payload);
bool receive_message(const tcp_transport & transport, framed_message & message,
                     uint16_t expected_type = 0, int timeout_ms = -1,
                     size_t max_payload = message_max_payload);

// Experimental chunked transfer for payloads larger than one frame. The
// sender emits one metadata frame followed by ordered data frames of at most
// chunk_bytes bytes; the receiver validates the sequence, sizes, and total
// before returning the assembled buffer. No threads, no background work.
// The peer must drain the stream concurrently (its own thread or process):
// a sequential send-then-receive over one connection deadlocks once bytes
// in flight exceed the socket buffers.
constexpr uint16_t chunk_message_type = 64;
constexpr size_t chunk_max_bytes = 1024 * 1024;
constexpr size_t chunk_max_total = 256 * 1024 * 1024;

bool send_chunked(const tcp_transport & transport, const void * data, size_t size,
                  size_t chunk_bytes = chunk_max_bytes);
bool receive_chunked(const tcp_transport & transport, std::vector<uint8_t> & out,
                     int timeout_ms = -1, size_t chunk_bytes = chunk_max_bytes);

} // namespace node_runtime