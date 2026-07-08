#include "modelfile.h"
#include <cstdio>
#include <cstring>
#include <string>

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static void parse_kv(const std::string & key, const std::string & val, const std::string & section, modelfile_config * out) {
    if (section == "model") {
        if (key == "name") out->name = val;
        else if (key == "base") out->base = val;
    } else if (section == "quantization") {
        if (key == "weight_type") out->quantization.weight_type = val;
        else if (key == "cache_type_k") out->quantization.cache_type_k = val;
        else if (key == "cache_type_v") out->quantization.cache_type_v = val;
    } else if (section == "inference") {
        if (key == "context_length") { try { out->inference.context_length = std::stoi(val); } catch (...) {} }
        else if (key == "n_gpu_layers") { try { out->inference.n_gpu_layers = std::stoi(val); } catch (...) {} }
        else if (key == "backend_preference") out->inference.backend_preference = val;
    } else if (section == "speculative") {
        if (key == "draft_model") out->speculative.draft_model = val;
        else if (key == "draft_ratio") { try { out->speculative.draft_ratio = std::stof(val); } catch (...) {} }
    } else if (section == "server") {
        if (key == "port") { try { out->server.port = std::stoi(val); } catch (...) {} }
        else if (key == "max_sequences") { try { out->server.max_sequences = std::stoi(val); } catch (...) {} }
    }
}

bool modelfile_parse(const char * path, modelfile_config * out) {
    FILE * f = fopen(path, "r");
    if (!f) return false;

    *out = modelfile_config();
    std::string section;
    char buf[1024];

    while (fgets(buf, sizeof(buf), f)) {
        std::string line = trim(buf);
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        parse_kv(key, val, section, out);
    }
    fclose(f);
    return true;
}
