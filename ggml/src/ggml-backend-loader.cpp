#include "ggml-backend-plugin.h"

#include <dlfcn.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_BACKENDS_PATH "/usr/local/lib/ggml-backends"
#define MAX_PLUGINS 32

static struct {
    void * handle;
    ggml_backend_plugin_info info;
    ggml_backend_plugin_init_fn init_fn;
} g_plugins[MAX_PLUGINS];

static int g_n_plugins = 0;

int ggml_backend_loader_init(const char * path) {
    const char * dir = path ? path : GGML_BACKENDS_PATH;
    DIR * d = opendir(dir);
    if (!d) return 0;

    struct dirent * ent;
    g_n_plugins = 0;
    while ((ent = readdir(d)) != NULL && g_n_plugins < MAX_PLUGINS) {
        const char * dot = strrchr(ent->d_name, '.');
        if (!dot || strcmp(dot, ".so") != 0) continue;

        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);

        void * h = dlopen(full, RTLD_NOW);
        if (!h) continue;

        ggml_backend_plugin_init_fn init =
            (ggml_backend_plugin_init_fn)dlsym(h, "ggml_backend_plugin_init");
        if (!init) { dlclose(h); continue; }

        g_plugins[g_n_plugins].handle = h;
        g_plugins[g_n_plugins].init_fn = init;
        memset(&g_plugins[g_n_plugins].info, 0, sizeof(ggml_backend_plugin_info));

        if (init(&g_plugins[g_n_plugins].info) != 0) {
            dlclose(h);
            continue;
        }
        g_n_plugins++;
    }
    closedir(d);
    return g_n_plugins;
}

const ggml_backend_plugin_info * ggml_backend_loader_get_plugins(void) {
    static ggml_backend_plugin_info infos[MAX_PLUGINS];
    for (int i = 0; i < g_n_plugins; i++) infos[i] = g_plugins[i].info;
    return infos;
}

int ggml_backend_loader_count(void) {
    return g_n_plugins;
}

ggml_backend_t ggml_backend_loader_create(int idx, int device_idx) {
    if (idx < 0 || idx >= g_n_plugins) return NULL;
    if (device_idx < 0 || device_idx >= g_plugins[idx].info.n_devices) return NULL;
    if (g_plugins[idx].info.create_backend) {
        return g_plugins[idx].info.create_backend(device_idx);
    }
    return NULL;
}
