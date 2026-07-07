#include "calibrator.h"
#include <cmath>

static const float aciq_alpha[] = {
    0.0f,
    0.0f,
    0.650f,
    0.900f,
    1.185f,
    1.450f,
    1.700f,
    1.850f,
    3.92403714f,
};

static const float aciq_gaussian_const = 0.5f * 0.35f * (1.0f + sqrtf((float)M_PI * logf(4.0f)));

float compute_aciq_scale(float absmax, int64_t n_elements, int n_bits) {
    if (absmax < 1e-8f || n_elements <= 0 || n_bits < 2 || n_bits > 8) {
        return 0.0f;
    }

    float std_dev = absmax * 2.0f * aciq_gaussian_const / sqrtf(2.0f * logf((float)n_elements));
    float clip = aciq_alpha[n_bits] * std_dev;

    if (clip < 1e-8f) return 0.0f;

    return ((1 << (n_bits - 1)) - 1) / clip;
}
