#include "node-runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
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
    size_t chunk_bytes = node_runtime::chunk_max_bytes;
    if (argc > 4 || (argc > 1 && !parse_size(argv[1], message_size)) ||
        (argc > 2 && !parse_size(argv[2], message_count)) ||
        (argc > 3 && !parse_size(argv[3], chunk_bytes)) ||
        message_size > node_runtime::chunk_max_total || message_count == 0 ||
        chunk_bytes == 0 || chunk_bytes > node_runtime::chunk_max_bytes) {
        std::fprintf(stderr, "usage: node-runtime-bench [message_bytes<=%zu] [message_count>0] [chunk_bytes<=%zu]\n",
                     node_runtime::chunk_max_total, node_runtime::chunk_max_bytes);
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
    bool server_ok = true;
    double total_ms = 0.0;
    double minimum_ms = std::numeric_limits<double>::max();
    double maximum_ms = 0.0;
    // The chunked path deadlocks when the same thread sends a full payload
    // larger than the socket buffers: the peer must drain concurrently. The
    // server leg runs on its own thread, as the two sides do in production.
    const bool threaded = message_size > node_runtime::message_max_payload;
    node_runtime::framed_message received;
    std::thread server_thread;
    if (threaded) {
        server_thread = std::thread([&]() {
            std::vector<uint8_t> big;
            for (size_t i = 0; i < message_count; ++i) {
                if (!node_runtime::receive_chunked(server, big, 1000) || big != payload ||
                    !node_runtime::send_chunked(server, big.data(), big.size(), chunk_bytes)) {
                    server_ok = false;
                    return;
                }
            }
        });
    }
    for (size_t i = 0; i < message_count; ++i) {
        const auto start = clock_type::now();
        bool exchanged;
        if (!threaded) {
            exchanged = node_runtime::send_message(client, 1, payload.data(), payload.size()) &&
                        node_runtime::receive_message(server, received, 1, 1000) &&
                        received.payload == payload &&
                        node_runtime::send_message(server, 1, received.payload.data(), received.payload.size()) &&
                        node_runtime::receive_message(client, received, 1, 1000) &&
                        received.payload == payload;
        } else {
            std::vector<uint8_t> echo;
            exchanged = node_runtime::send_chunked(client, payload.data(), payload.size(), chunk_bytes) &&
                        node_runtime::receive_chunked(client, echo, 1000) &&
                        echo == payload;
        }
        if (!exchanged) {
            std::fprintf(stderr, "loopback exchange failed at message %zu\n", i);
            return 1;
        }
        if (!server_ok) {
            std::fprintf(stderr, "server leg failed at message %zu\n", i);
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
    if (threaded) {
        server_thread.join();
        if (!server_ok) {
            std::fprintf(stderr, "server leg reported a failed exchange\n");
            return 1;
        }
    }
    return 0;
}