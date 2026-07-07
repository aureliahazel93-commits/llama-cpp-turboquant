#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>

#if defined(__linux__)
#include <fstream>
#include <sstream>
#endif

struct thermal_telemetry {
    float cpu_temp_c = 0.0f;
    float igpu_temp_c = 0.0f;
    float cpu_utilization = 0.0f;
    bool has_thermal = false;
};

#if defined(__linux__)
static float read_thermal_zone(const char * path) {
    std::ifstream f(path);
    if (!f.is_open()) return -1.0f;
    int temp_mc;
    if (f >> temp_mc) return temp_mc / 1000.0f;
    return -1.0f;
}

static thermal_telemetry read_linux_thermal(void) {
    thermal_telemetry t;

    float cpu_max = 0.0f;
    float igpu_max = 0.0f;
    bool found_cpu = false, found_igpu = false;

    for (int z = 0; z < 20; z++) {
        char type_path[256], temp_path[256];
        snprintf(type_path, sizeof(type_path),
            "/sys/class/thermal/thermal_zone%d/type", z);
        snprintf(temp_path, sizeof(temp_path),
            "/sys/class/thermal/thermal_zone%d/temp", z);

        std::ifstream tf(type_path);
        if (!tf.is_open()) break;

        std::string type_name;
        std::getline(tf, type_name);

        float temp = read_thermal_zone(temp_path);
        if (temp < 0.0f) continue;

        if (!found_cpu && (type_name.find("CPU") != std::string::npos ||
                           type_name.find("cpu") != std::string::npos ||
                           type_name.find("x86") != std::string::npos ||
                           type_name.find("ACPI") != std::string::npos)) {
            cpu_max = std::max(cpu_max, temp);
            found_cpu = true;
        }
        if (!found_igpu && (type_name.find("GPU") != std::string::npos ||
                            type_name.find("gpu") != std::string::npos ||
                            type_name.find("iGPU") != std::string::npos)) {
            igpu_max = std::max(igpu_max, temp);
            found_igpu = true;
        }
    }

    if (!found_cpu && !found_igpu) {
        for (int z = 0; z < 10; z++) {
            char temp_path[256];
            snprintf(temp_path, sizeof(temp_path),
                "/sys/class/thermal/thermal_zone%d/temp", z);
            float temp = read_thermal_zone(temp_path);
            if (temp < 0.0f) continue;
            if (z == 0) cpu_max = temp;
            if (z == 1) igpu_max = temp;
        }
    }

    t.cpu_temp_c = cpu_max;
    t.igpu_temp_c = igpu_max;

    if (found_cpu || found_igpu) t.has_thermal = true;

    return t;
}

static float read_cpu_utilization(void) {
    std::ifstream f("/proc/stat");
    if (!f.is_open()) return 0.0f;

    std::string line;
    if (!std::getline(f, line)) return 0.0f;
    if (line.substr(0, 4) != "cpu ") return 0.0f;

    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    if (sscanf(line.c_str() + 5, "%lu %lu %lu %lu %lu %lu %lu %lu",
               &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) != 8) {
        return 0.0f;
    }

    uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
    uint64_t busy = user + nice + system + irq + softirq;
    return (total > 0) ? (float)busy / (float)total : 0.0f;
}
#else
static thermal_telemetry read_linux_thermal(void) { return {}; }
static float read_cpu_utilization(void) { return 0.0f; }
#endif

struct thermal_balancer {
    float cpu_ratio = 0.5f;
    float cpu_ema_time = 0.0f;
    float igpu_ema_time = 0.0f;
    float kp = 0.05f;
    float ki = 0.01f;
    float kd = 0.0f;
    float integral = 0.0f;
    float prev_error = 0.0f;
    float thermal_threshold_c = 85.0f;
    float thermal_penalty = 0.1f;
    float ema_alpha = 0.9f;
    int layer_count = 0;

    void update(float cpu_time_ms, float igpu_time_ms) {
        layer_count++;

        cpu_ema_time = cpu_ema_time * ema_alpha + cpu_time_ms * (1.0f - ema_alpha);
        igpu_ema_time = igpu_ema_time * ema_alpha + igpu_time_ms * (1.0f - ema_alpha);

        if (igpu_ema_time < 0.001f && cpu_ema_time < 0.001f) return;

        float error = 0.0f;
        if (igpu_ema_time > 0.001f) {
            error = cpu_ema_time / igpu_ema_time - 1.0f;
        }

        integral += error * 0.01f;
        integral = std::max(-1.0f, std::min(1.0f, integral));

        float derivative = error - prev_error;
        prev_error = error;

        float pid = kp * error + ki * integral + kd * derivative;
        cpu_ratio -= pid * 0.1f;

        thermal_telemetry t = read_linux_thermal();
        if (t.has_thermal) {
            if (t.igpu_temp_c > thermal_threshold_c) {
                cpu_ratio += thermal_penalty;
            }
            if (t.cpu_temp_c > thermal_threshold_c) {
                cpu_ratio -= thermal_penalty;
            }
        }

        cpu_ratio = std::max(0.1f, std::min(0.9f, cpu_ratio));
    }

    float get_ratio() const { return cpu_ratio; }

    thermal_telemetry get_telemetry() const {
        thermal_telemetry t = read_linux_thermal();
        t.cpu_utilization = read_cpu_utilization();
        return t;
    }
};

static thermal_balancer g_balancer;

void hybrid_set_cpu_ratio(float ratio) {
    if (ratio < 0.1f) ratio = 0.1f;
    if (ratio > 0.9f) ratio = 0.9f;
    g_balancer.cpu_ratio = ratio;
}

float hybrid_get_cpu_ratio() {
    return g_balancer.get_ratio();
}

void hybrid_thermal_update(float cpu_time_ms, float igpu_time_ms) {
    g_balancer.update(cpu_time_ms, igpu_time_ms);
}

thermal_telemetry hybrid_thermal_read(void) {
    return g_balancer.get_telemetry();
}
