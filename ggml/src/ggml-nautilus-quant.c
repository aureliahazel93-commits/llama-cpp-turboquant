/*
 * NautilusQuant: KV cache compression via golden-ratio Givens rotation + PolarQuant
 * Based on: carlosfundora/NautilusQuant (golden angle theta_k = (2*pi/phi^2)*(k+1))
 *           Weyl 1916 equidistribution theorem: O(1/N) angular uniformity vs O(1/sqrt(N)) random
 *
 * Replaces TurboQuant's WHT rotation with deterministic golden-ratio Givens rotations.
 * Same block layout as turbo3_0 (norm + 2-bit indices + 1-bit signs = 14 bytes per 128 values).
 * Same 3+1 bit PolarQuant pipeline, same centroids, same norm-correction.
 * Key difference: rotation is orthogonal by construction (T^T*T = I), 1.9 KB LUT,
 * bit-identical every run, no PRNG, no state, maps 1:1 onto static-dataflow chips.
 */

#define _USE_MATH_DEFINES

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- constants ---------- */

#define NAUTILUS_D    128       /* rotation group size = head_dim */
#define NAUTILUS_PAIRS (NAUTILUS_D / 2)  /* 64 Givens pairs per layer */

/* Golden ratio and derived golden angle
 * phi = (1 + sqrt(5)) / 2 ≈ 1.618033988749895
 * theta_k = (2*pi / phi^2) * (k+1)
 *   phi^2 = phi + 1 ≈ 2.618033988749895
 *   2*pi / phi^2 ≈ 2.399963229728653 rad ≈ 137.507764050037854° */
#define PHI          1.61803398874989484820f
#define PHI_SQ       (PHI + 1.0f)              /* phi^2 = phi + 1 ≈ 2.61803 */
#define GOLDEN_BASE  (2.0f * (float)M_PI / PHI_SQ)  /* ≈ 2.399963 */

/* Same 3-bit Lloyd-Max centroids as turbo3_0 (for N(0, 1/128)) */
static const float NAUTILUS_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

/* ---------- golden-ratio Givens LUT ----------
 *
 * For each of the 64 pairs, we precompute cos(theta_k) and sin(theta_k)
 * for all 3 layers (base, base*phi, base*phi^2).
 * LUT size: 64 * 3 * 2 = 384 floats ≈ 1.5 KB (vs 8 MB for TurboQuant's d*d matrix)
 * Deterministic: computed once at init from phi and pi, no PRNG, no seed. */

static float nautilus_cos_l1[NAUTILUS_PAIRS];   /* layer 1: theta_k = GOLDEN_BASE * (k+1) */
static float nautilus_sin_l1[NAUTILUS_PAIRS];
static float nautilus_cos_l2[NAUTILUS_PAIRS];   /* layer 2: theta_k = GOLDEN_BASE * (k+1) * phi */
static float nautilus_sin_l2[NAUTILUS_PAIRS];
static float nautilus_cos_l3[NAUTILUS_PAIRS];   /* layer 3: theta_k = GOLDEN_BASE * (k+1) * phi^2 */
static float nautilus_sin_l3[NAUTILUS_PAIRS];
static int nautilus_rotation_initialized = 0;

static void nautilus_init_rotation(void) {
    if (nautilus_rotation_initialized) return;

    for (int k = 0; k < NAUTILUS_PAIRS; k++) {
        float t1 = GOLDEN_BASE * (k + 1);
        float t2 = t1 * PHI;
        float t3 = t2 * PHI;

        nautilus_cos_l1[k] = cosf(t1);
        nautilus_sin_l1[k] = sinf(t1);
        nautilus_cos_l2[k] = cosf(t2);
        nautilus_sin_l2[k] = sinf(t2);
        nautilus_cos_l3[k] = cosf(t3);
        nautilus_sin_l3[k] = sinf(t3);
    }

    nautilus_rotation_initialized = 1;
}

/* ---------- Givens rotation helpers ----------
 *
 * A Givens rotation G(i,j,theta) acts on a 2D subspace (elements i and j):
 *   [v_i]   [cos(theta)  -sin(theta)] [v_i]   [c*v_i - s*v_j]
 *   [v_j] = [sin(theta)   cos(theta)] [v_j] = [s*v_i + c*v_j]
 *
 * By construction: |G * v| = |v| for any v → T^T * T = I
 * dot(G*q, G*k) = dot(q, k) → attention scores preserved exactly.
 *
 * Layer 1: adjacent pairs (0,1), (2,3), (4,5), ...
 * Layer 2: shifted pairs (1,2), (3,4), (5,6), ...
 * Layer 3: butterfly stride (0,32), (1,33), (2,34), ...  stride = dim/4 = 32
 *   (only non-overlapping: for 128-dim, 32 pairs (0..31, 32..63))
 *
 * Full forward:  T = L3 * L2 * L1
 * Full inverse:  T^-1 = L1^T * L2^T * L3^T  (sine sign negated) */

static inline void givens_rotate_forward(float * v, int i, int j, float c, float s) {
    float vi = v[i];
    float vj = v[j];
    v[i] = c * vi - s * vj;
    v[j] = s * vi + c * vj;
}

static inline void givens_rotate_inverse(float * v, int i, int j, float c, float s) {
    /* Inverse Givens: negate sin (G^-1 = G^T = G(theta)^T = G(-theta)) */
    float vi = v[i];
    float vj = v[j];
    v[i] = c * vi + s * vj;
    v[j] = -s * vi + c * vj;
}

static void nautilus_forward(float * v, int dim) {
    const int pairs = dim / 2;
    const int stride = dim / 4;  /* = 32 for dim=128 */

    /* Layer 1: adjacent pairs (0,1), (2,3), ..., (126,127) */
    for (int k = 0; k < pairs; k++) {
        givens_rotate_forward(v, 2*k, 2*k + 1, nautilus_cos_l1[k], nautilus_sin_l1[k]);
    }

    /* Layer 2: shifted pairs (1,2), (3,4), ..., (125,126) */
    for (int k = 0; k < pairs - 1; k++) {
        givens_rotate_forward(v, 2*k + 1, 2*k + 2, nautilus_cos_l2[k], nautilus_sin_l2[k]);
    }

    /* Layer 3: butterfly pairs (k, k+stride) for k in 0..31 */
    for (int k = 0; k < stride; k++) {
        int i = k;
        int j = k + stride;
        givens_rotate_forward(v, i, j, nautilus_cos_l3[k], nautilus_sin_l3[k]);
    }
}

static void nautilus_inverse(float * v, int dim) {
    const int pairs = dim / 2;
    const int stride = dim / 4;

    /* Layer 3 inverse (negate sin) */
    for (int k = 0; k < stride; k++) {
        int i = k;
        int j = k + stride;
        givens_rotate_inverse(v, i, j, nautilus_cos_l3[k], nautilus_sin_l3[k]);
    }

    /* Layer 2 inverse (negate sin) */
    for (int k = 0; k < pairs - 1; k++) {
        givens_rotate_inverse(v, 2*k + 1, 2*k + 2, nautilus_cos_l2[k], nautilus_sin_l2[k]);
    }

    /* Layer 1 inverse (negate sin) */
    for (int k = 0; k < pairs; k++) {
        givens_rotate_inverse(v, 2*k, 2*k + 1, nautilus_cos_l1[k], nautilus_sin_l1[k]);
    }
}

/* ---------- nearest centroid (same as turbo3_0) ---------- */

static int nearest_centroid_nautilus3(float val) {
    /* 8 centroids, find nearest via midpoints (same thresholds as turbo3_0) */
    if (val < -0.154496f) return 0;
    if (val < -0.092804f) return 1;
    if (val < -0.044243f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.044243f) return 4;
    if (val <  0.092804f) return 5;
    if (val <  0.154496f) return 6;
    return 7;
}

/* ---------- quantize ---------- */

void quantize_row_nautilus3_0_ref(const float * GGML_RESTRICT x, block_nautilus3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % NAUTILUS_D == 0);
    nautilus_init_rotation();

    const int n_groups = k / NAUTILUS_D;

    for (int g = 0; g < n_groups; g++) {
        const float * src = x + g * NAUTILUS_D;
        block_nautilus3_0 * blk = &y[g];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        float buf[NAUTILUS_D];
        for (int j = 0; j < NAUTILUS_D; j++) {
            buf[j] = src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < NAUTILUS_D; j++) buf[j] *= inv_norm;

        /* 3. Forward golden-ratio Givens rotation (3 layers) */
        nautilus_forward(buf, NAUTILUS_D);

        /* 4. Quantize + pack (same layout as turbo3_0) */
        memset(blk->qs,   0, NAUTILUS_D / 4);
        memset(blk->signs, 0, NAUTILUS_D / 8);

        float recon_sq = 0.0f;
        for (int j = 0; j < NAUTILUS_D; j++) {
            int idx = nearest_centroid_nautilus3(buf[j]);
            blk->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) {
                blk->signs[j / 8] |= (1 << (j % 8));
            }
            recon_sq += NAUTILUS_CENTROIDS_3BIT[idx] * NAUTILUS_CENTROIDS_3BIT[idx];
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

/* ---------- dequantize ----------
 *
 * Unpack indices → dequantize centroids → inverse golden Givens rotation → rescale by norm.
 * The inverse rotation restores the original orientation so attention dot products work
 * on the same coordinate system as the original FP16 KV values. */

void dequantize_row_nautilus3_0(const block_nautilus3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % NAUTILUS_D == 0);
    nautilus_init_rotation();

    const int n_groups = k / NAUTILUS_D;

    for (int g = 0; g < n_groups; g++) {
        float norm = GGML_FP16_TO_FP32(x[g].norm);
        const block_nautilus3_0 * blk = &x[g];
        float buf[NAUTILUS_D];

        /* 1. Unpack + dequantize centroids */
        for (int j = 0; j < NAUTILUS_D; j++) {
            uint8_t low2 = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx  = low2 | (hi1 << 2);
            buf[j] = NAUTILUS_CENTROIDS_3BIT[idx];
        }

        /* 2. Inverse golden-ratio Givens rotation (3 layers, negated sin) */
        nautilus_inverse(buf, NAUTILUS_D);

        /* 3. Rescale by corrected norm */
        for (int j = 0; j < NAUTILUS_D; j++) {
            y[g * NAUTILUS_D + j] = buf[j] * norm;
        }
    }
}

/* ---------- bulk quantize ---------- */

size_t quantize_nautilus3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % NAUTILUS_D == 0);

    size_t row_size = (n_per_row / NAUTILUS_D) * sizeof(block_nautilus3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_nautilus3_0_ref(
            src + row * n_per_row,
            (block_nautilus3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
