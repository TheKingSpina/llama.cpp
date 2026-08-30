#include "node-runtime.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>

namespace {
void usage(const char * name) {
    std::fprintf(stderr, "usage: %s HOST PORT [--monitor SECONDS]\n", name);
}
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 5) {
        usage(argv[0]);
        return 2;
    }

    unsigned port = 0;
    try {
        port = std::stoul(argv[2]);
    } catch (...) {
        usage(argv[0]);
        return 2;
    }
    if (port > 65535) {
        usage(argv[0]);
        return 2;
    }
    unsigned monitor_seconds = 0;
    if (argc == 5) {
        if (std::string(argv[3]) != "--monitor") {
            usage(argv[0]);
            return 2;
        }
        try {
            monitor_seconds = std::stoul(argv[4]);
        } catch (...) {
            usage(argv[0]);
            return 2;
        }
        if (monitor_seconds == 0) {
            usage(argv[0]);
            return 2;
        }
    }

    node_runtime::tcp_transport client;
    if (!client.connect(argv[1], static_cast<uint16_t>(port))) {
        std::fprintf(stderr, "node-runtime-client: connection failed\n");
        return 1;
    }

    const node_runtime::registration_message registration = node_runtime::local_registration();
    const std::string payload = node_runtime::registration_json(registration);
    if (!node_runtime::send_message(client, 1, payload.data(), payload.size())) {
        std::fprintf(stderr, "node-runtime-client: registration send failed\n");
        return 1;
    }

    node_runtime::framed_message acknowledgement;
    if (!node_runtime::receive_message(client, acknowledgement, 2, 5000)) {
        std::fprintf(stderr, "node-runtime-client: acknowledgement failed\n");
        return 1;
    }
    std::printf("registration_ack=%.*s\n", static_cast<int>(acknowledgement.payload.size()),
                acknowledgement.payload.data());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(monitor_seconds);
    uint64_t sequence = 0;
    do {
        const node_runtime::heartbeat_message heartbeat{1, ++sequence, sequence};
        if (!node_runtime::send_message(client, 4, &heartbeat, sizeof(heartbeat))) {
            std::fprintf(stderr, "node-runtime-client: heartbeat send failed\n");
            return 1;
        }
        node_runtime::framed_message heartbeat_ack;
        if (!node_runtime::receive_message(client, heartbeat_ack, 4, 5000) ||
            heartbeat_ack.payload.size() != sizeof(heartbeat)) {
            std::fprintf(stderr, "node-runtime-client: heartbeat acknowledgement failed\n");
            return 1;
        }
        std::printf("heartbeat_ack=ok sequence=%llu\n", static_cast<unsigned long long>(sequence));
        if (monitor_seconds == 0 || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        // One heartbeat per second keeps the link warm without flooding.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } while (true);
    return 0;
}
