#include "registry.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>

static std::string home_cache_dir() {
    const char * h = getenv("HOME");
    return std::string(h ? h : "/tmp") + "/.llama-mod/models";
}

static int cmd_pull(int argc, char ** argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: llama-mod pull <model-name>\n");
        return 1;
    }
    std::string name = argv[0];
    std::string cache = home_cache_dir();
    fprintf(stderr, "pulling %s...\n", name.c_str());
    if (registry_pull(name, cache)) {
        fprintf(stderr, "done: %s\n", cache_path(cache, name).c_str());
        return 0;
    }
    fprintf(stderr, "pull failed\n");
    return 1;
}

static int cmd_run(int argc, char ** argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: llama-mod run <model-name> [prompt]\n");
        return 1;
    }
    std::string name = argv[0];
    std::string cache = home_cache_dir();
    if (!cache_has(cache, name)) {
        fprintf(stderr, "model not cached. pull first: llama-mod pull %s\n", name.c_str());
        return 1;
    }
    std::string path = cache_path(cache, name);
    std::vector<char *> exec_args;
    exec_args.push_back((char *)"llama-cli");
    exec_args.push_back((char *)"-m");
    exec_args.push_back((char *)path.c_str());
    for (int i = 1; i < argc; i++) exec_args.push_back(argv[i]);
    exec_args.push_back(nullptr);
    execvp("llama-cli", exec_args.data());
    perror("execvp");
    return 1;
}

static int cmd_list(int argc, char ** /*argv*/) {
    (void)argc;
    std::string cache = home_cache_dir();
    auto entries = cache_list(cache);
    if (entries.empty()) {
        fprintf(stderr, "no cached models. use 'llama-mod pull <name>' first.\n");
        return 0;
    }
    printf("%-40s %10s  %s\n", "NAME", "SIZE", "SHA256");
    for (const auto & e : entries) {
        printf("%-40s %10.1fMB  %.12s...\n",
               e.name.c_str(),
               (float)e.size_bytes / 1e6,
               e.sha256.c_str());
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: llama-mod <pull|run|list> [args...]\n");
        return 1;
    }
    const char * cmd = argv[1];
    if (strcmp(cmd, "pull") == 0) return cmd_pull(argc - 2, argv + 2);
    if (strcmp(cmd, "run")  == 0) return cmd_run(argc - 2, argv + 2);
    if (strcmp(cmd, "list") == 0) return cmd_list(argc - 2, argv + 2);
    fprintf(stderr, "unknown command: %s\n", cmd);
    return 1;
}
