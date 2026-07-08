#ifndef LLAMA_MODHUB_REGISTRY_H
#define LLAMA_MODHUB_REGISTRY_H

#include <string>
#include <vector>
#include <cstdint>

struct model_registry_entry {
    std::string name;
    std::string repo;
    std::string filename;
    std::string url;
    std::string sha256;
    float       size_mb;
    std::string quant_type;
};

struct model_cache_entry {
    std::string name;
    std::string local_path;
    std::string sha256;
    int64_t     size_bytes;
    int64_t     cached_at;
};

std::vector<model_registry_entry> registry_list(const std::string & pattern);
model_registry_entry registry_find(const std::string & name);
bool registry_pull(const std::string & name, const std::string & dest_dir);

std::vector<model_cache_entry> cache_list(const std::string & cache_dir);
bool cache_has(const std::string & cache_dir, const std::string & name);
std::string cache_path(const std::string & cache_dir, const std::string & name);
bool cache_add(const std::string & cache_dir, const model_cache_entry & entry);
bool cache_remove(const std::string & cache_dir, const std::string & name);

#endif
