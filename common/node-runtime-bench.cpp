#include "node-runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct exchange_spec {
    size_t message_size  = 1024;
    size_t message_count = 1000;
    size_t chunk_bytes   = node_runtime::chunk_max_bytes;
};

struct exchange_stats {
    double total_ms   = 0.0;
    double minimum_ms = std::numeric_limits<double>::max();
    double maximum_ms = 0.0;
};

bool parse_size(const char * value, size_t & result) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return true;
}

bool parse_port(const char * value, uint16_t & result) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > 65535) {
        return false;
    }
    result = static_cast<uint16_t>(parsed);
    return true;
}

// True when the payload exceeds the framed-message limit, so the chunked
// path is required instead of send_message/receive_message.
bool needs_chunked(const exchange_spec & spec) {
    return spec.message_size > node_runtime::message_max_payload;
}

bool spec_valid(const exchange_spec & spec) {
    if (spec.message_size > node_runtime::chunk_max_total || spec.message_count == 0) {
        return false;
    }
    if (needs_chunked(spec)) {
        return spec.chunk_bytes > 0 && spec.chunk_bytes <= node_runtime::chunk_max_bytes;
    }
    return spec.chunk_bytes <= node_runtime::chunk_max_bytes;
}

void print_usage() {
    std::fprintf(stderr,
                 "usage: node-runtime-bench [message_bytes<=%zu] [message_count>0] [chunk_bytes<=%zu]\n"
                 "       node-runtime-bench --listen [--bind ADDRESS] PORT [message_count>0] [chunk_bytes 0..%zu]\n"
                 "       node-runtime-bench --connect HOST PORT [message_bytes<=%zu] [message_count>0] [chunk_bytes 0..%zu]\n"
                 "  message_bytes above %zu forces the chunked path, which needs chunk_bytes>0\n"
                 "  the client and the peer must agree on chunk_bytes and message_count\n",
                 node_runtime::chunk_max_total, node_runtime::chunk_max_bytes,
                 node_runtime::chunk_max_bytes, node_runtime::chunk_max_total,
                 node_runtime::chunk_max_bytes, node_runtime::message_max_payload);
}

void accumulate(exchange_stats & stats, double elapsed_ms) {
    stats.total_ms += elapsed_ms;
    stats.minimum_ms = std::min(stats.minimum_ms, elapsed_ms);
    stats.maximum_ms = std::max(stats.maximum_ms, elapsed_ms);
}

double elapsed_ms_since(const clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

// Framed round trip with the echo handled inline on a second transport.
// Only valid inside one process (loopback): the caller drives both sides.
bool framed_exchange(node_runtime::tcp_transport & client,
                     node_runtime::tcp_transport & server,
                     const std::vector<uint8_t> & payload) {
    node_runtime::framed_message received;
    return node_runtime::send_message(client, 1, payload.data(), payload.size()) &&
           node_runtime::receive_message(server, received, 1, 1000) &&
           received.payload == payload &&
           node_runtime::send_message(server, 1, received.payload.data(), received.payload.size()) &&
           node_runtime::receive_message(client, received, 1, 1000) &&
           received.payload == payload;
}

// Chunked round trip where the peer echoes the payload back. The transport
// carries the full round trip, so this works across two machines.
bool chunked_exchange(node_runtime::tcp_transport & client,
                      const exchange_spec & spec,
                      const std::vector<uint8_t> & payload) {
    std::vector<uint8_t> echo;
    return node_runtime::send_chunked(client, payload.data(), payload.size(), spec.chunk_bytes) &&
           node_runtime::receive_chunked(client, echo, 1000, spec.chunk_bytes) &&
           echo == payload;
}

// Framed round trip against a remote peer that echoes the payload back.
// Only valid across two machines: the peer runs the echo, not this side.
bool framed_exchange_remote(node_runtime::tcp_transport & client,
                            const std::vector<uint8_t> & payload) {
    node_runtime::framed_message received;
    return node_runtime::send_message(client, 1, payload.data(), payload.size()) &&
           node_runtime::receive_message(client, received, 1, 1000) &&
           received.payload == payload;
}

// Server leg used by both the loopback bench and --listen: receive each
// message and echo it back. chunk_bytes is the agreed chunk size; the client
// must use the same value so both sides accept each other's data frames. A
// chunk_bytes of 0 selects the framed path instead of the chunked one.
// Returns false on any transport failure.
bool run_echo_server(node_runtime::tcp_transport & server, size_t message_count, size_t chunk_bytes) {
    if (chunk_bytes == 0) {
        node_runtime::framed_message received;
        for (size_t i = 0; i < message_count; ++i) {
            if (!node_runtime::receive_message(server, received, 1, 1000) ||
                !node_runtime::send_message(server, 1, received.payload.data(), received.payload.size())) {
                return false;
            }
        }
        return true;
    }
    std::vector<uint8_t> big;
    for (size_t i = 0; i < message_count; ++i) {
        if (!node_runtime::receive_chunked(server, big, 1000, chunk_bytes) ||
            !node_runtime::send_chunked(server, big.data(), big.size(), chunk_bytes)) {
            return false;
        }
    }
    return true;
}

void print_stats(const exchange_spec & spec, const exchange_stats & stats) {
    const double seconds = stats.total_ms / 1000.0;
    const double round_trip_bytes = static_cast<double>(spec.message_count) * spec.message_size * 2.0;
    std::printf("message_bytes=%zu message_count=%zu\n", spec.message_size, spec.message_count);
    std::printf("round_trip_latency_ms_avg=%.3f min=%.3f max=%.3f\n",
                stats.total_ms / spec.message_count, stats.minimum_ms, stats.maximum_ms);
    std::printf("round_trip_throughput=%.2f messages/s %.2f MiB/s\n",
                spec.message_count / seconds, round_trip_bytes / seconds / (1024.0 * 1024.0));
}

// Single-process bench over loopback: one transport drives both legs.
int run_loopback(const exchange_spec & spec) {
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

    const std::vector<uint8_t> payload(spec.message_size, 0x5a);
    exchange_stats stats;
    if (!needs_chunked(spec)) {
        for (size_t i = 0; i < spec.message_count; ++i) {
            const auto start = clock_type::now();
            if (!framed_exchange(client, server, payload)) {
                std::fprintf(stderr, "loopback exchange failed at message %zu\n", i);
                return 1;
            }
            accumulate(stats, elapsed_ms_since(start));
        }
    } else {
        // The server leg must drain concurrently with the client send.
        bool server_ok = true;
        std::thread server_thread([&]() { server_ok = run_echo_server(server, spec.message_count, spec.chunk_bytes); });
        for (size_t i = 0; i < spec.message_count; ++i) {
            const auto start = clock_type::now();
            if (!chunked_exchange(client, spec, payload) || !server_ok) {
                std::fprintf(stderr, "loopback exchange failed at message %zu\n", i);
                return 1;
            }
            accumulate(stats, elapsed_ms_since(start));
        }
        server_thread.join();
    }
    print_stats(spec, stats);
    return 0;
}

// Accept connections on PORT and echo each message back. Reuses the listener
// so a single server can serve every battery size in sequence. chunk_bytes
// is the agreed chunk size the client also uses; 0 selects the framed path.
// The peer picks the payload size, the server echoes what it receives.
int run_listen(uint16_t port, const std::string & bind_address, size_t message_count, size_t chunk_bytes) {
    node_runtime::tcp_transport listener;
    if (!listener.listen(port, bind_address)) {
        std::fprintf(stderr, "failed to listen on %s port %u\n", bind_address.c_str(), port);
        return 1;
    }
    std::printf("listening on %s port=%u chunk_bytes=%zu\n", bind_address.c_str(), listener.port(), chunk_bytes);
    std::fflush(stdout);
    for (int connection = 0; connection < 64; ++connection) {
        node_runtime::tcp_transport server = listener.accept(30000);
        if (!server.valid()) {
            std::fprintf(stderr, "accept timed out, stopping\n");
            return 1;
        }
        std::printf("connection accepted index=%d\n", connection);
        std::fflush(stdout);
        if (!run_echo_server(server, message_count, chunk_bytes)) {
            std::fprintf(stderr, "server leg failed on connection %d\n", connection);
            return 1;
        }
        server.close();
        std::printf("connection closed index=%d\n", connection);
        std::fflush(stdout);
    }
    return 0;
}

// Connect to a peer, drive message_count round trips, print the stats. The
// peer runs the matching --listen as its echo server.
int run_connect(const std::string & host, uint16_t port, const exchange_spec & spec) {
    node_runtime::tcp_transport client;
    if (!client.connect(host, port)) {
        std::fprintf(stderr, "failed to connect to %s:%u\n", host.c_str(), port);
        return 1;
    }
    const std::vector<uint8_t> payload(spec.message_size, 0x5a);
    exchange_stats stats;
    for (size_t i = 0; i < spec.message_count; ++i) {
        const auto start = clock_type::now();
        const bool exchanged = needs_chunked(spec) ? chunked_exchange(client, spec, payload)
                                                   : framed_exchange_remote(client, payload);
        if (!exchanged) {
            std::fprintf(stderr, "exchange failed at message %zu\n", i);
            return 1;
        }
        accumulate(stats, elapsed_ms_since(start));
    }
    print_stats(spec, stats);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const bool want_listen = argc > 1 && std::strcmp(argv[1], "--listen") == 0;
    const bool want_connect = argc > 1 && std::strcmp(argv[1], "--connect") == 0;

    if (want_listen) {
        std::string bind_address = "127.0.0.1";
        int arg = 2;
        if (argc > 2 && std::strcmp(argv[2], "--bind") == 0) {
            if (argc < 4) {
                print_usage();
                return 2;
            }
            bind_address = argv[3];
            arg = 4;
        }
        if (argc <= arg) {
            print_usage();
            return 2;
        }
        uint16_t port = 0;
        size_t message_count = 1000;
        size_t chunk_bytes = 0;
        if (!parse_port(argv[arg], port)) {
            print_usage();
            return 2;
        }
        if (argc > arg + 1 && !parse_size(argv[arg + 1], message_count)) {
            print_usage();
            return 2;
        }
        if (argc > arg + 2 && !parse_size(argv[arg + 2], chunk_bytes)) {
            print_usage();
            return 2;
        }
        if (argc > arg + 3 || message_count == 0 ||
            (chunk_bytes != 0 && chunk_bytes > node_runtime::chunk_max_bytes)) {
            print_usage();
            return 2;
        }
        return run_listen(port, bind_address, message_count, chunk_bytes);
    }

    if (want_connect) {
        if (argc < 4) {
            print_usage();
            return 2;
        }
        uint16_t port = 0;
        exchange_spec spec;
        const std::string host = argv[2];
        if (!parse_port(argv[3], port)) {
            print_usage();
            return 2;
        }
        if (argc > 4 && !parse_size(argv[4], spec.message_size)) {
            print_usage();
            return 2;
        }
        if (argc > 5 && !parse_size(argv[5], spec.message_count)) {
            print_usage();
            return 2;
        }
        if (argc > 6 && !parse_size(argv[6], spec.chunk_bytes)) {
            print_usage();
            return 2;
        }
        if (argc > 7 || !spec_valid(spec)) {
            print_usage();
            return 2;
        }
        return run_connect(host, port, spec);
    }

    exchange_spec spec;
    if (argc > 4 || (argc > 1 && !parse_size(argv[1], spec.message_size)) ||
        (argc > 2 && !parse_size(argv[2], spec.message_count)) ||
        (argc > 3 && !parse_size(argv[3], spec.chunk_bytes)) ||
        !spec_valid(spec)) {
        print_usage();
        return 2;
    }
    return run_loopback(spec);
}