#include "node-runtime.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
constexpr uint16_t registration_type = 1;
constexpr uint16_t acknowledgement_type = 2;

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
    std::fprintf(stderr, "usage: %s [--bind ADDRESS] [--port PORT] [--once]\n", name);
}
} // namespace

int main(int argc, char ** argv) {
    uint16_t port = 0;
    std::string bind_address = "127.0.0.1";
    bool once = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--once") {
            once = true;
        } else if (argument == "--bind" && i + 1 < argc) {
            bind_address = argv[++i];
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
    std::printf("llama-node listening on %s:%u\n", bind_address.c_str(), listener.port());
    std::fflush(stdout);

    node_runtime::node_lifecycle lifecycle;
    node_runtime::tcp_transport peer = listener.accept(once ? 5000 : 30000);
    if (!peer.valid()) {
        lifecycle.request_shutdown();
        lifecycle.mark_stopped();
        return once ? 1 : 0;
    }

    node_runtime::framed_message request;
    if (!node_runtime::receive_message(peer, request, registration_type, 5000)) {
        lifecycle.request_shutdown();
        lifecycle.mark_stopped();
        return 1;
    }

    node_runtime::registration_state registration = node_runtime::local_registration_state();
    node_runtime::set_registered(registration, true);
    const std::string reply = node_runtime::registration_json(registration.message);
    if (!node_runtime::send_message(peer, acknowledgement_type, reply.data(), reply.size())) {
        lifecycle.request_shutdown();
        lifecycle.mark_stopped();
        return 1;
    }

    lifecycle.observe_heartbeat(lifecycle.heartbeat(1), 1);
    lifecycle.request_shutdown();
    lifecycle.mark_stopped();
    std::printf("llama-node registration acknowledged; stopped\n");
    return 0;
}
