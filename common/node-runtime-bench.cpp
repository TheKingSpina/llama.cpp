#include "node-runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

bool parse_size(const char * value, size_t & result) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    size_t message_size = 1024;
    size_t message_count = 1000;
    if (argc > 3 || (argc > 1 && !parse_size(argv[1], message_size)) ||
        (argc > 2 && !parse_size(argv[2], message_count)) ||
        message_size > node_runtime::message_max_payload || message_count == 0) {
        std::fprintf(stderr, "usage: node-runtime-bench [message_bytes<=%zu] [message_count>0]\n",
                     node_runtime::message_max_payload);
        return 2;
    }

    node_runtime::tcp_transport listener;
    if (!listener.listen()) {
        std::fprintf(stderr, "failed to listen on loopback\n");
        return 1;
    }
    node_runtime::tcp_transport client;
    if (!client.connect("127.0.0.1", listener.port())) {
        std::fprintf(stderr, "failed to connect to loopback\n");
        return 1;
    }
    node_runtime::tcp_transport server = listener.accept(1000);
    if (!server.valid()) {
        std::fprintf(stderr, "failed to accept loopback connection\n");
        return 1;
    }

    std::vector<uint8_t> payload(message_size, 0x5a);
    node_runtime::framed_message received;
    double total_ms = 0.0;
    double minimum_ms = std::numeric_limits<double>::max();
    double maximum_ms = 0.0;
    for (size_t i = 0; i < message_count; ++i) {
        const auto start = clock_type::now();
        if (!node_runtime::send_message(client, 1, payload.data(), payload.size()) ||
            !node_runtime::receive_message(server, received, 1, 1000) ||
            received.payload != payload ||
            !node_runtime::send_message(server, 1, received.payload.data(), received.payload.size()) ||
            !node_runtime::receive_message(client, received, 1, 1000) ||
            received.payload != payload) {
            std::fprintf(stderr, "loopback exchange failed at message %zu\n", i);
            return 1;
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
        total_ms += elapsed_ms;
        minimum_ms = std::min(minimum_ms, elapsed_ms);
        maximum_ms = std::max(maximum_ms, elapsed_ms);
    }

    const double seconds = total_ms / 1000.0;
    const double round_trip_bytes = static_cast<double>(message_count) * message_size * 2.0;
    std::printf("message_bytes=%zu message_count=%zu\n", message_size, message_count);
    std::printf("round_trip_latency_ms_avg=%.3f min=%.3f max=%.3f\n",
                total_ms / message_count, minimum_ms, maximum_ms);
    std::printf("round_trip_throughput=%.2f messages/s %.2f MiB/s\n",
                message_count / seconds, round_trip_bytes / seconds / (1024.0 * 1024.0));
    return 0;
}