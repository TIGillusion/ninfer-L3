#include "product/logging/logging.h"
#include "product/logging/nonblocking_stderr.h"
#include "product/logging/startup_log.h"
#include "serve/fatal_signal.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <sys/mman.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    // FIRST: business threads must never be able to block on a stderr write.
    // Inside a container stderr is a pipe owned by the log driver; when the
    // consumer dies the pipe fills and every logging thread wedges holding
    // whatever lock it holds (observed: a 5-hour total deadlock, the stall
    // sentinel included - it died inside its own forensic log call). After
    // this call a full pipe merely drops the log line.
    (void)ninfer::product::install_nonblocking_stderr();

    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    // Attribute fatal memory signals before anything else runs: a corrupted
    // heap or a driver-level fault must leave a record (signal, fault address,
    // thread, backtrace) in the request log even when the stdout pipe is dead.
    ninfer::serve::fatal::install(
        options.request_log_jsonl.empty() ? nullptr : options.request_log_jsonl.c_str());

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        // Lock anonymous (heap / host-KV pool) mappings after warmup:
        // the decode orchestrator lives here. Named mappings (weights mmap,
        // driver shmem) are excluded - locking the 12.4GB weights mmap
        // OOM-locked the whole guest. We lock ranges 64MB-6GB with no path.
        {
            unsigned long locked_bytes = 0;
            int locked_ranges = 0;
            FILE *sm = fopen("/proc/self/smaps", "r");
            if (sm) {
                char buf[512];
                unsigned long rs = 0, rn = 0;
                long sz_kb = 0;
                int is_anon = 0;
                while (fgets(buf, sizeof(buf), sm)) {
                    unsigned long a = 0, b = 0;
                    char perms[8] = {0};
                    int used = 0;
                    if (sscanf(buf, "%lx-%lx %7s %*s %*s %*s %n", &a, &b, perms, &used) >= 3) {
                        // flush previous: lock if anon and 64MB..6GB
                        if (is_anon && sz_kb >= 65536 && (unsigned long)sz_kb <= 6291456) {
                            if (mlock((const void *)rs, (size_t)sz_kb * 1024) == 0) {
                                locked_bytes += (unsigned long)sz_kb * 1024;
                                locked_ranges++;
                            }
                        }
                        rs = a; rn = b;
                        // anon = nothing but whitespace after inode
                        is_anon = 1;
                        for (int i = used; buf[i] && buf[i] != '\n'; i++) {
                            if (buf[i] != ' ' && buf[i] != '\t') { is_anon = 0; break; }
                        }
                    } else if (strncmp(buf, "Size:", 5) == 0) {
                        sz_kb = strtol(buf + 5, NULL, 10);
                    } else if (strncmp(buf, "Name:", 5) == 0) {
                        const char *p = buf + 5;
                        while (*p == ' ' || *p == '\t') p++;
                        if (*p && *p != '\n') is_anon = 0;
                    }
                }
                fclose(sm);
                // flush last region
                if (is_anon && sz_kb >= 65536 && (unsigned long)sz_kb <= 6291456) {
                    if (mlock((const void *)rs, (size_t)sz_kb * 1024) == 0) {
                        locked_bytes += (unsigned long)sz_kb * 1024;
                        locked_ranges++;
                    }
                }
            }
            fprintf(stderr, "[serve] mlock-anon: locked %d ranges, %.2f GB pinned\n",
                    locked_ranges, locked_bytes / 1073741824.0);
            fflush(stderr);
        }
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
