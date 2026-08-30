#include "node-runtime.h"

#include "apple-runtime.h"

#include <sstream>
#include <climits>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace node_runtime {

namespace {

double finite_nonnegative(double value) {
    return value >= 0.0 ? value : 0.0;
}

void close_socket(int socket) {
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(socket));
#else
    close(socket);
#endif
}

std::string hostname() {
#if defined(__APPLE__)
    size_t size = 0;
    if (sysctlbyname("kern.hostname", nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname("kern.hostname", value.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
#else
    return {};
#endif
}

std::string json_string(const std::string & value) {
    std::ostringstream result;
    result << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') {
            result << '\\' << c;
        } else if (c == '\n') {
            result << "\\n";
        } else if (c == '\r') {
            result << "\\r";
        } else if (c == '\t') {
            result << "\\t";
        } else {
            result << c;
        }
    }
    result << '"';
    return result.str();
}

} // namespace

scheduler_score scheduler_score_candidate(const scheduler_task & task,
                                           const scheduler_candidate & candidate) {
    scheduler_score result;
    result.candidate_id = candidate.candidate_id;
    result.feasible = candidate.compute_capacity > 0.0 &&
                      candidate.available_memory_bytes >= task.memory_bytes;
    if (!result.feasible) {
        return result;
    }
    result.compute_cost = finite_nonnegative(task.compute_units) / candidate.compute_capacity;
    result.memory_cost = candidate.available_memory_bytes == 0 ? 0.0 :
                         static_cast<double>(task.memory_bytes) /
                         static_cast<double>(candidate.available_memory_bytes);
    result.network_cost = finite_nonnegative(candidate.network_penalty) *
                          finite_nonnegative(static_cast<double>(task.network_bytes));
    result.storage_cost = finite_nonnegative(candidate.storage_penalty) *
                          finite_nonnegative(static_cast<double>(task.storage_bytes));
    result.cost = result.compute_cost + result.memory_cost + result.network_cost + result.storage_cost;
    return result;
}

scheduler_plan_report scheduler_plan(const scheduler_task & task,
                                     const scheduler_candidate * candidates,
                                     size_t candidate_count) {
    scheduler_plan_report report;
    report.task_id = task.task_id;
    if (candidates == nullptr) {
        return report;
    }
    report.candidates.reserve(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        report.candidates.push_back(scheduler_score_candidate(task, candidates[i]));
    }
    std::stable_sort(report.candidates.begin(), report.candidates.end(),
        [](const scheduler_score & left, const scheduler_score & right) {
            if (left.feasible != right.feasible) {
                return left.feasible;
            }
            if (left.feasible && left.cost != right.cost) {
                return left.cost < right.cost;
            }
            return left.candidate_id < right.candidate_id;
        });
    return report;
}

int scheduler_select_candidate(const scheduler_task & task,
                               const scheduler_candidate * candidates,
                               size_t candidate_count,
                               scheduler_score * selected_score) {
    if (candidates == nullptr || candidate_count == 0) {
        return -1;
    }
    int selected = -1;
    scheduler_score best;
    for (size_t i = 0; i < candidate_count; ++i) {
        const scheduler_score score = scheduler_score_candidate(task, candidates[i]);
        if (!score.feasible || (selected >= 0 &&
            (score.cost > best.cost || (score.cost == best.cost && score.candidate_id >= best.candidate_id)))) {
            continue;
        }
        selected = static_cast<int>(i);
        best = score;
    }
    if (selected_score != nullptr && selected >= 0) {
        *selected_score = best;
    }
    return selected;
}

capabilities local_capabilities() {
    const apple_runtime::system_telemetry telemetry = apple_runtime::system_snapshot();
    capabilities result;
    result.hostname = hostname();
    result.node_id = result.hostname;
    result.chip = telemetry.chip;
    result.physical_memory = telemetry.physical_memory;
    result.logical_cpu_count = telemetry.logical_cpu_count;
    result.apple_silicon = telemetry.apple_silicon;
    result.unified_memory = telemetry.apple_silicon;
    return result;
}

std::string capabilities_json(const capabilities & value) {
    std::ostringstream result;
    result << "{\"node_id\":" << json_string(value.node_id)
           << ",\"hostname\":" << json_string(value.hostname)
           << ",\"chip\":" << json_string(value.chip)
           << ",\"physical_memory\":" << value.physical_memory
           << ",\"logical_cpu_count\":" << value.logical_cpu_count
           << ",\"apple_silicon\":" << (value.apple_silicon ? "true" : "false")
           << ",\"unified_memory\":" << (value.unified_memory ? "true" : "false")
           << '}';
    return result.str();
}

registration_message local_registration() {
    registration_message result;
    result.capabilities = local_capabilities();
    return result;
}

registration_state local_registration_state() {
    registration_state result;
    result.message = local_registration();
    return result;
}

void set_registered(registration_state & state, bool registered) {
    state.registered = registered;
}

std::string registration_json(const registration_message & value) {
    std::ostringstream result;
    result << "{\"protocol_version\":" << value.protocol_version
           << ",\"capabilities\":" << capabilities_json(value.capabilities)
           << '}';
    return result.str();
}

node_lifecycle::node_lifecycle(lifecycle_config config) : config_(config) {}

heartbeat_message node_lifecycle::heartbeat(uint64_t now_ms) {
    heartbeat_message result;
    result.sequence = ++heartbeat_sequence_;
    result.sent_at_ms = now_ms;
    last_heartbeat_ms_ = now_ms;
    next_heartbeat_ms_ = now_ms + config_.heartbeat_interval_ms;
    if (state_ == lifecycle_state::timed_out) {
        state_ = lifecycle_state::running;
    }
    return result;
}

void node_lifecycle::observe_heartbeat(const heartbeat_message & message, uint64_t now_ms) {
    if (message.protocol_version != 1 || message.sequence == 0) {
        return;
    }
    last_heartbeat_ms_ = now_ms;
    if (state_ == lifecycle_state::timed_out) {
        state_ = lifecycle_state::running;
    }
}

bool node_lifecycle::heartbeat_due(uint64_t now_ms) const {
    return state_ == lifecycle_state::running && now_ms >= next_heartbeat_ms_;
}

bool node_lifecycle::timed_out(uint64_t now_ms) {
    if (state_ != lifecycle_state::running || last_heartbeat_ms_ == 0) {
        return false;
    }
    if (now_ms - last_heartbeat_ms_ >= config_.heartbeat_timeout_ms) {
        state_ = lifecycle_state::timed_out;
        return true;
    }
    return false;
}

void node_lifecycle::request_shutdown() {
    if (state_ == lifecycle_state::running) {
        state_ = lifecycle_state::shutdown_requested;
    }
}

void node_lifecycle::mark_stopped() {
    state_ = lifecycle_state::stopped;
}

lifecycle_state node_lifecycle::state() const { return state_; }
uint64_t node_lifecycle::last_heartbeat_ms() const { return last_heartbeat_ms_; }
uint64_t node_lifecycle::heartbeat_sequence() const { return heartbeat_sequence_; }

tcp_transport::tcp_transport(int socket, uint16_t port) : socket_(socket), port_(port) {}

tcp_transport::~tcp_transport() { close(); }

tcp_transport::tcp_transport(tcp_transport && other) noexcept : socket_(other.socket_), port_(other.port_) {
    other.socket_ = -1;
    other.port_ = 0;
}

tcp_transport & tcp_transport::operator=(tcp_transport && other) noexcept {
    if (this != &other) {
        close();
        socket_ = other.socket_;
        port_ = other.port_;
        other.socket_ = -1;
        other.port_ = 0;
    }
    return *this;
}

bool tcp_transport::listen(uint16_t port, const std::string & bind_address) {
    close();
    const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) {
        return false;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    if (bind_address == "localhost") {
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (::inet_pton(AF_INET, bind_address.c_str(), &address.sin_addr) != 1) {
        close_socket(socket);
        return false;
    }
    address.sin_port = htons(port);
    if (::bind(socket, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        ::listen(socket, 1) != 0) {
        close_socket(socket);
        return false;
    }
    socklen_t length = sizeof(address);
    if (::getsockname(socket, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
        close_socket(socket);
        return false;
    }
    socket_ = socket;
    port_ = ntohs(address.sin_port);
    return true;
}

tcp_transport tcp_transport::accept(int timeout_ms) const {
    if (!valid()) {
        return {};
    }
    if (timeout_ms >= 0) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(socket_, &read_set);
        timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (::select(socket_ + 1, &read_set, nullptr, nullptr, &timeout) <= 0) {
            return {};
        }
    }
    sockaddr_in address{};
    socklen_t length = sizeof(address);
    const int socket = ::accept(socket_, reinterpret_cast<sockaddr *>(&address), &length);
    return socket < 0 ? tcp_transport{} : tcp_transport(socket, port_);
}

bool tcp_transport::connect(const std::string & host, uint16_t port) {
    close();
    if (host != "localhost" && host != "127.0.0.1") {
        return false;
    }
    const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) {
        return false;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (::connect(socket, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        close_socket(socket);
        return false;
    }
    socket_ = socket;
    port_ = port;
    return true;
}

bool tcp_transport::send(const void * data, size_t size) const {
    if (!valid() || (size != 0 && data == nullptr)) {
        return false;
    }
    const char * bytes = static_cast<const char *>(data);
    while (size != 0) {
        const int sent = ::send(socket_, bytes, static_cast<int>(size), 0);
        if (sent <= 0) {
            return false;
        }
        bytes += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

size_t tcp_transport::receive(void * data, size_t capacity, int timeout_ms) const {
    if (!valid() || (capacity != 0 && data == nullptr) || capacity > static_cast<size_t>(INT32_MAX)) {
        return 0;
    }
    if (timeout_ms >= 0) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(socket_, &read_set);
        timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (::select(socket_ + 1, &read_set, nullptr, nullptr, &timeout) <= 0) {
            return 0;
        }
    }
    const int received = ::recv(socket_, static_cast<char *>(data), static_cast<int>(capacity), 0);
    return received > 0 ? static_cast<size_t>(received) : 0;
}

void tcp_transport::close() {
    if (valid()) {
        close_socket(socket_);
        socket_ = -1;
        port_ = 0;
    }
}

bool tcp_transport::valid() const { return socket_ >= 0; }

uint16_t tcp_transport::port() const { return port_; }

namespace {
bool receive_exact(const tcp_transport & transport, void * data, size_t size, int timeout_ms) {
    size_t offset = 0;
    while (offset < size) {
        const size_t received = transport.receive(static_cast<uint8_t *>(data) + offset,
                                                  size - offset, timeout_ms);
        if (received == 0) {
            return false;
        }
        offset += received;
    }
    return true;
}

struct wire_header {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t payload_size;
};
}

bool send_message(const tcp_transport & transport, uint16_t type,
                  const void * payload, size_t payload_size) {
    if (type == 0 || payload_size > message_max_payload ||
        (payload_size != 0 && payload == nullptr)) {
        return false;
    }
    wire_header header{htonl(message_magic), htons(message_protocol_version), htons(type),
                       htonl(static_cast<uint32_t>(payload_size))};
    return transport.send(&header, sizeof(header)) && transport.send(payload, payload_size);
}

bool receive_message(const tcp_transport & transport, framed_message & message,
                     uint16_t expected_type, int timeout_ms) {
    wire_header header{};
    if (!receive_exact(transport, &header, sizeof(header), timeout_ms) ||
        ntohl(header.magic) != message_magic ||
        ntohs(header.version) != message_protocol_version ||
        ntohs(header.type) == 0 ||
        (expected_type != 0 && ntohs(header.type) != expected_type) ||
        ntohl(header.payload_size) > message_max_payload) {
        return false;
    }
    const size_t size = ntohl(header.payload_size);
    std::vector<uint8_t> payload(size);
    if (!receive_exact(transport, payload.data(), size, timeout_ms)) {
        return false;
    }
    message.type = ntohs(header.type);
    message.payload = std::move(payload);
    return true;
}

} // namespace node_runtime