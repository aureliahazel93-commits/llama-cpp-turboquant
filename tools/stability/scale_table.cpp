#include "calibrator.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

int write_scale_table(
    const char * path,
    const std::vector<calibration_scale_entry> & entries) {

    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "scale_table: cannot open %s for writing\n", path);
        return 1;
    }

    for (const auto & entry : entries) {
        fprintf(f, "%s %s", entry.tensor_name.c_str(), entry.scale_type.c_str());
        for (size_t i = 0; i < entry.scales.size(); i++) {
            fprintf(f, " %.8g", entry.scales[i]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}

std::vector<calibration_scale_entry> read_scale_table(
    const char * path) {

    std::vector<calibration_scale_entry> entries;

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "scale_table: cannot open %s for reading\n", path);
        return entries;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        calibration_scale_entry entry;
        std::istringstream iss(line);

        if (!(iss >> entry.tensor_name >> entry.scale_type)) continue;

        float val;
        while (iss >> val) {
            entry.scales.push_back(val);
        }

        entries.push_back(std::move(entry));
    }

    return entries;
}
