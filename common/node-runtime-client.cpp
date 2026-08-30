#include "node-runtime.h"

#include <cstdio>
#include <string>

namespace {
void usage(const char * name) {
    std::fprintf(stderr, "usage: %s HOST PORT\n", name);
}
}

int main(int argc, char ** argv) {
    if (argc != 3) {
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
    return 0;
}
