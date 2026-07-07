#include "calibrator.h"
#include <cmath>
#include <algorithm>
#include <vector>

float refine_layer_scale(
    const float * ref_output,
    const float * quant_output,
    int n_elements,
    float current_scale,
    float search_min,
    float search_max,
    int grid_steps) {

    if (n_elements <= 0 || current_scale < 1e-8f || grid_steps <= 0) {
        return current_scale;
    }

    float best_scale = current_scale;
    float best_cos = -1.0f;
    float step = (search_max - search_min) / (float)grid_steps;

    for (int g = 0; g <= grid_steps; g++) {
        float s = search_min + g * step;
        float scale_ratio = s / current_scale;

        double dot = 0.0, mag_ref = 0.0, mag_quant = 0.0;
        for (int i = 0; i < n_elements; i++) {
            float rq = quant_output[i] * scale_ratio;
            dot      += (double)(ref_output[i] * rq);
            mag_ref  += (double)(ref_output[i] * ref_output[i]);
            mag_quant += (double)(rq * rq);
        }

        float cos_sim = 0.0f;
        if (mag_ref > 0.0 && mag_quant > 0.0) {
            cos_sim = (float)(dot / (sqrt(mag_ref) * sqrt(mag_quant)));
        }

        if (cos_sim > best_cos) {
            best_cos = cos_sim;
            best_scale = current_scale * s;
        }
    }

    return best_scale;
}
