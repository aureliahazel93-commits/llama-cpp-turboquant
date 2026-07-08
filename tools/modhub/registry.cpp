#include "registry.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/sha.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

static std::string manifest_path(const std::string & cache_dir) {
    return cache_dir + "/manifest.json";
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool write_file(const std::string & path, const std::string & content) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << content;
    return true;
}

static std::string sha256_hex(const std::string & data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((unsigned char *)data.c_str(), data.size(), hash);
    char buf[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        snprintf(buf + i * 2, 3, "%02x", hash[i]);
    }
    buf[64] = '\0';
    return buf;
}

static std::string sha256_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "";
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    char buf[8192];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        SHA256_Update(&ctx, buf, (size_t)f.gcount());
    }
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);
    char hex[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    hex[64] = '\0';
    return hex;
}

static std::string url_decode(const std::string & in) {
    std::string out;
    for (size_t i = 0; i < in.size(); i++) {
        if (in[i] == '%' && i + 2 < in.size()) {
            char hex[3] = { in[i+1], in[i+2], '\0' };
            out += (char)strtol(hex, nullptr, 16);
            i += 2;
        } else {
            out += in[i];
        }
    }
    return out;
}

static bool parse_url(const std::string & url, std::string & host, std::string & port, std::string & path) {
    if (url.size() < 8 || url.substr(0, 8) != "https://") {
        if (url.size() < 7 || url.substr(0, 7) != "http://") return false;
        port = "80";
        std::string rest = url.substr(7);
        auto slash = rest.find('/');
        if (slash == std::string::npos) { host = rest; path = "/"; }
        else { host = rest.substr(0, slash); path = rest.substr(slash); }
    } else {
        port = "443";
        std::string rest = url.substr(8);
        auto slash = rest.find('/');
        if (slash == std::string::npos) { host = rest; path = "/"; }
        else { host = rest.substr(0, slash); path = rest.substr(slash); }
    }
    if (!path.empty() && path[0] != '/') path = "/" + path;
    auto colon = host.find(':');
    if (colon != std::string::npos) {
        port = host.substr(colon + 1);
        host = host.substr(0, colon);
    }
    return true;
}

static int http_connect(const std::string & host, const std::string & port) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
    if (rc != 0) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);
    return fd;
}

static bool send_all(int fd, const char * buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

static std::string recv_all(int fd) {
    std::string out;
    char buf[8192];
    while (true) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        out.append(buf, (size_t)n);
    }
    return out;
}

struct http_response {
    int status = 0;
    std::string body;
    std::map<std::string, std::string> headers;
};

static http_response http_get(const std::string & url) {
    http_response resp;
    std::string host, port, path;
    if (!parse_url(url, host, port, path)) return resp;

    int fd = http_connect(host, port);
    if (fd < 0) return resp;

    std::string req = "GET " + path + " HTTP/1.1\r\n"
                      "Host: " + host + "\r\n"
                      "Connection: close\r\n"
                      "User-Agent: llama-mod/1.0\r\n"
                      "\r\n";
    if (!send_all(fd, req.c_str(), req.size())) { close(fd); return resp; }

    std::string raw = recv_all(fd);
    close(fd);

    auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return resp;
    std::string headers = raw.substr(0, hdr_end);
    resp.body = raw.substr(hdr_end + 4);

    auto first_line = headers.find("\r\n");
    std::string status_line = (first_line != std::string::npos)
        ? headers.substr(0, first_line) : headers;
    if (status_line.size() > 5) {
        resp.status = atoi(status_line.c_str() + 9);
    }

    std::istringstream hs(headers);
    std::string line;
    while (std::getline(hs, line)) {
        if (line.size() >= 2 && line[line.size()-1] == '\r') line.pop_back();
        auto colon = line.find(": ");
        if (colon != std::string::npos) {
            resp.headers[line.substr(0, colon)] = line.substr(colon + 2);
        }
    }
    return resp;
}

static std::string follow_redirect(const std::string & url) {
    std::string current = url;
    for (int i = 0; i < 5; i++) {
        auto resp = http_get(current);
        if (resp.status >= 300 && resp.status < 400) {
            auto it = resp.headers.find("location");
            if (it == resp.headers.end()) return "";
            std::string loc = it->second;
            if (loc.substr(0, 4) != "http") {
                std::string host, port, path;
                parse_url(current, host, port, path);
                current = "https://" + host + loc;
            } else {
                current = loc;
            }
        } else {
            return current;
        }
    }
    return "";
}

static std::string hf_resolve(const std::string & repo, const std::string & filename) {
    std::string url = "https://huggingface.co/" + repo + "/resolve/main/" + filename;
    std::string resolved = follow_redirect(url);
    return resolved.empty() ? url : resolved;
}

static bool download_file(const std::string & url, const std::string & dest) {
    std::string host, port, path;
    if (!parse_url(url, host, port, path)) return false;

    int fd = http_connect(host, port);
    if (fd < 0) return false;

    std::string req = "GET " + path + " HTTP/1.1\r\n"
                      "Host: " + host + "\r\n"
                      "Connection: close\r\n"
                      "User-Agent: llama-mod/1.0\r\n"
                      "\r\n";
    if (!send_all(fd, req.c_str(), req.size())) { close(fd); return false; }

    std::string raw = recv_all(fd);
    close(fd);

    auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return false;

    std::string header_section = raw.substr(0, hdr_end);
    std::string body = raw.substr(hdr_end + 4);

    auto first_nl = header_section.find("\r\n");
    std::string status_line = (first_nl != std::string::npos)
        ? header_section.substr(0, first_nl) : header_section;
    int status = 0;
    if (status_line.size() > 5) status = atoi(status_line.c_str() + 9);
    if (status != 200) {
        fprintf(stderr, "download failed: HTTP %d\n", status);
        return false;
    }

    FILE * fp = fopen(dest.c_str(), "wb");
    if (!fp) return false;
    fwrite(body.c_str(), 1, body.size(), fp);
    fclose(fp);
    return true;
}

static std::string trim(const std::string & s) {
    auto b = s.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\n\r");
    return s.substr(b, e - b + 1);
}

static std::string unquote(const std::string & s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

static std::string extract_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':')) pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
    std::string val;
    while (pos < json.size()) {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"':  val += '"'; break;
                case '\\': val += '\\'; break;
                case 'n':  val += '\n'; break;
                case 't':  val += '\t'; break;
                case '/':  val += '/'; break;
                default:   val += json[pos]; break;
            }
        } else if (json[pos] == '"') {
            break;
        } else {
            val += json[pos];
        }
        pos++;
    }
    return val;
}

static std::vector<std::string> split_array(const std::string & json) {
    std::vector<std::string> items;
    if (json.empty() || json[0] != '[') return items;
    int depth = 0;
    size_t start = 1;
    bool in_str = false;
    for (size_t i = 0; i < json.size(); i++) {
        char c = json[i];
        if (c == '"' && (i == 0 || json[i-1] != '\\')) in_str = !in_str;
        if (in_str) continue;
        if (c == '{') depth++;
        if (c == '}') depth--;
        if (c == ',' && depth == 0) {
            items.push_back(trim(json.substr(start, i - start)));
            start = i + 1;
        }
    }
    std::string last = trim(json.substr(start));
    if (!last.empty()) items.push_back(last);
    return items;
}

static std::string json_stringify_entry(const model_cache_entry & e) {
    std::string s = "{";
    s += "\"name\":\"" + e.name + "\",";
    s += "\"local_path\":\"" + e.local_path + "\",";
    s += "\"sha256\":\"" + e.sha256 + "\",";
    s += "\"size_bytes\":" + std::to_string(e.size_bytes) + ",";
    s += "\"cached_at\":" + std::to_string(e.cached_at);
    s += "}";
    return s;
}

static model_cache_entry parse_cache_entry(const std::string & json) {
    model_cache_entry e;
    e.size_bytes = 0;
    e.cached_at = 0;
    e.name = extract_string(json, "name");
    e.local_path = extract_string(json, "local_path");
    e.sha256 = extract_string(json, "sha256");
    std::string sb = extract_string(json, "size_bytes");
    if (!sb.empty()) e.size_bytes = atoll(sb.c_str());
    std::string ca = extract_string(json, "cached_at");
    if (!ca.empty()) e.cached_at = atoll(ca.c_str());
    return e;
}

static std::vector<model_cache_entry> parse_cache_manifest(const std::string & json) {
    std::vector<model_cache_entry> entries;
    auto items = split_array(json);
    for (const auto & item : items) {
        entries.push_back(parse_cache_entry(item));
    }
    return entries;
}

static std::string serialize_manifest(const std::vector<model_cache_entry> & entries) {
    std::string s = "[";
    for (size_t i = 0; i < entries.size(); i++) {
        if (i > 0) s += ",";
        s += json_stringify_entry(entries[i]);
    }
    s += "]";
    return s;
}

std::vector<model_cache_entry> cache_list(const std::string & cache_dir) {
    std::string mp = manifest_path(cache_dir);
    std::string content = read_file(mp);
    if (content.empty()) return {};
    return parse_cache_manifest(content);
}

bool cache_has(const std::string & cache_dir, const std::string & name) {
    auto entries = cache_list(cache_dir);
    for (const auto & e : entries) {
        if (e.name == name) return true;
    }
    return false;
}

std::string cache_path(const std::string & cache_dir, const std::string & name) {
    return cache_dir + "/" + name + ".gguf";
}

bool cache_add(const std::string & cache_dir, const model_cache_entry & entry) {
    fs::create_directories(cache_dir);
    auto entries = cache_list(cache_dir);
    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].name == entry.name) {
            entries[i] = entry;
            return write_file(manifest_path(cache_dir), serialize_manifest(entries));
        }
    }
    entries.push_back(entry);
    return write_file(manifest_path(cache_dir), serialize_manifest(entries));
}

bool cache_remove(const std::string & cache_dir, const std::string & name) {
    auto entries = cache_list(cache_dir);
    std::vector<model_cache_entry> updated;
    for (const auto & e : entries) {
        if (e.name != name) updated.push_back(e);
    }
    if (updated.size() == entries.size()) return false;
    std::string fp = cache_path(cache_dir, name);
    std::remove(fp.c_str());
    return write_file(manifest_path(cache_dir), serialize_manifest(updated));
}

std::vector<model_registry_entry> registry_list(const std::string & pattern) {
    std::vector<model_registry_entry> results;
    std::string url = "https://huggingface.co/api/models?search=" + pattern;
    auto resp = http_get(url);
    if (resp.status != 200) {
        fprintf(stderr, "registry query failed: HTTP %d\n", resp.status);
        return results;
    }
    auto items = split_array(resp.body);
    for (const auto & item : items) {
        model_registry_entry e;
        e.name = extract_string(item, "modelId");
        if (e.name.empty()) e.name = extract_string(item, "id");
        e.size_mb = 0;
        std::string size_s = extract_string(item, "size");
        if (!size_s.empty()) {
            auto sb = extract_string(item, "safetensors");
            if (!sb.empty()) e.size_mb = atof(sb.c_str()) / 1e6;
        }
        results.push_back(e);
    }
    return results;
}

model_registry_entry registry_find(const std::string & name) {
    model_registry_entry found;
    std::string url = "https://huggingface.co/api/models/" + name;
    auto resp = http_get(url);
    if (resp.status != 200) {
        fprintf(stderr, "registry find failed: HTTP %d\n", resp.status);
        return found;
    }
    found.name = extract_string(resp.body, "modelId");
    if (found.name.empty()) found.name = extract_string(resp.body, "id");
    auto siblings = split_array(extract_string(resp.body, "siblings") == ""
        ? "[]" : "[]");
    auto items = split_array(resp.body);
    auto idx = resp.body.find("\"siblings\"");
    if (idx != std::string::npos) {
        size_t arr_start = resp.body.find('[', idx);
        if (arr_start != std::string::npos) {
            int depth = 0;
            size_t arr_end = arr_start;
            for (size_t i = arr_start; i < resp.body.size(); i++) {
                if (resp.body[i] == '[') depth++;
                if (resp.body[i] == ']') depth--;
                if (depth == 0) { arr_end = i; break; }
            }
            std::string arr = resp.body.substr(arr_start, arr_end - arr_start + 1);
            auto sibs = split_array(arr);
            for (const auto & s : sibs) {
                std::string fname = extract_string(s, "rfilename");
                if (fname.find(".gguf") != std::string::npos) {
                    found.filename = fname;
                    found.url = "https://huggingface.co/" + found.name + "/resolve/main/" + fname;
                    found.sha256 = extract_string(s, "blobId");
                    break;
                }
            }
        }
    }
    return found;
}

bool registry_pull(const std::string & name, const std::string & dest_dir) {
    if (cache_has(dest_dir, name)) {
        fprintf(stderr, "already cached: %s\n", name.c_str());
        return true;
    }

    model_registry_entry entry = registry_find(name);
    if (entry.name.empty()) {
        fprintf(stderr, "model not found in registry: %s\n", name.c_str());
        return false;
    }
    if (entry.url.empty()) {
        fprintf(stderr, "no downloadable .gguf found for: %s\n", name.c_str());
        return false;
    }

    fs::create_directories(dest_dir);
    std::string dest = cache_path(dest_dir, name);
    fprintf(stderr, "resolving %s...\n", entry.url.c_str());
    std::string resolved = hf_resolve(entry.name, entry.filename);
    fprintf(stderr, "downloading %s...\n", resolved.c_str());

    if (!download_file(resolved, dest)) {
        fprintf(stderr, "download failed\n");
        std::remove(dest.c_str());
        return false;
    }

    fprintf(stderr, "verifying sha256...\n");
    std::string file_sha = sha256_file(dest);
    if (!entry.sha256.empty() && file_sha != entry.sha256) {
        fprintf(stderr, "sha256 mismatch: expected %s, got %s\n",
                entry.sha256.c_str(), file_sha.c_str());
        std::remove(dest.c_str());
        return false;
    }
    fprintf(stderr, "sha256 ok: %s\n", file_sha.c_str());

    model_cache_entry ce;
    ce.name = name;
    ce.local_path = dest;
    ce.sha256 = file_sha;
    ce.size_bytes = (int64_t)fs::file_size(dest);
    ce.cached_at = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return cache_add(dest_dir, ce);
}
