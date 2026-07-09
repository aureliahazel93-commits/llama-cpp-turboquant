/*
 * Edge quantization formats: iso-edge and planar-edge
 *
 * Both use block_size=32 with per-half RMS scaling and uniform 4-bit grid.
 * Key optimization: vec_dot operates in the ROTATED DOMAIN (no inverse
 * rotation on read). Q is rotated inside vec_dot, amortized across all
 * KV positions. Dequant stays in rotated domain — same cost as Q4_0.
 *
 * iso-edge:    quaternion 4D rotation (8 groups of 4 elements per block)
 * planar-edge: Givens 2D rotation (16 pairs per block)
 *
 * Rotation preserves dot products: <RQ, RK> = <Q, K>
 */

#define _USE_MATH_DEFINES

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#define QK_EDGE 32

/* Uniform 4-bit grid: 16 levels uniformly spaced in [-1, 1] */
#define EDGE_4BIT_MAX 7.5f

/* ---------- Rotation constants ---------- */

static const float EDGE_COS[16] = {
    -0.6401948775f, 0.9876777670f, -0.1566164072f, 0.1675281983f,
    -0.0849015372f, -0.4444504570f, 0.7791700032f, 0.8544771479f,
    -0.8820599001f, 0.9825251974f, 0.1957806239f, -0.9994339505f,
    0.9861326540f, 0.3159545009f, -0.5883725236f, -0.9603962025f
};
static const float EDGE_SIN[16] = {
    -0.7682125480f, 0.1565012098f, 0.9876595066f, 0.9858672846f,
    -0.9963893461f, -0.8958034334f, -0.6268126563f, 0.5194889833f,
    0.4711372757f, 0.1861296229f, 0.9806477182f, -0.0336419184f,
    0.1659589972f, 0.9487743477f, -0.8085899990f, -0.2786379985f
};

static const float EDGE_QW[8] = {
    0.1839882820f, 0.3216364128f, 0.8630346533f, 0.3701115282f,
    -0.8812559863f, 0.3477010319f, -0.0978357045f, 0.5873644435f
};
static const float EDGE_QX[8] = {
    0.9792212364f, -0.8791538199f, -0.2377727311f, -0.5266550632f,
    -0.2041363468f, -0.6381082421f, -0.3334022431f, 0.3219073132f
};
static const float EDGE_QY[8] = {
    0.0311127163f, -0.1888759825f, 0.3721285740f, 0.2688341087f,
    -0.3080457626f, 0.4435399160f, 0.1297987954f, 0.5979764351f
};
static const float EDGE_QZ[8] = {
    -0.0794108380f, 0.2965880006f, 0.2452664647f, -0.7165054941f,
    0.2946592047f, -0.5245895602f, 0.9286675357f, 0.4402304797f
};

/* ---------- Shared helpers ---------- */

static inline float edge_rms16(const float * x) {
    float sq = 0.0f;
    for (int i = 0; i < 16; i++) sq += x[i] * x[i];
    return sqrtf(sq / 16.0f);
}

static inline uint8_t edge_quant4(float val, float scale) {
    float inv = (scale > 1e-10f) ? EDGE_4BIT_MAX / scale : 0.0f;
    int q = (int)roundf(val * inv + EDGE_4BIT_MAX);
    if (q < 0) q = 0;
    if (q > 15) q = 15;
    return (uint8_t)q;
}

static inline float edge_dequant4(uint8_t idx, float scale) {
    return ((float)idx - EDGE_4BIT_MAX) * (scale / EDGE_4BIT_MAX);
}

/* ===================================================================
 * Planar-edge: 2D Givens rotation
 * =================================================================== */

void quantize_row_planar_edge_ref(const float * GGML_RESTRICT x, block_edge_4bit * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_EDGE == 0);
    const int nb = (int)(k / QK_EDGE);

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * QK_EDGE;
        block_edge_4bit * blk = &y[b];

        float rot[QK_EDGE];
        for (int p = 0; p < 16; p++) {
            float v0 = src[2*p];
            float v1 = src[2*p + 1];
            rot[2*p]     = EDGE_COS[p] * v0 - EDGE_SIN[p] * v1;
            rot[2*p + 1] = EDGE_SIN[p] * v0 + EDGE_COS[p] * v1;
        }

        float d0 = edge_rms16(rot);
        float d1 = edge_rms16(rot + 16);

        blk->d0 = GGML_FP32_TO_FP16(d0);
        blk->d1 = GGML_FP32_TO_FP16(d1);

        memset(blk->qs, 0, QK_EDGE / 2);
        for (int j = 0; j < QK_EDGE; j++) {
            float d = (j < 16) ? d0 : d1;
            uint8_t q = edge_quant4(rot[j], d);
            blk->qs[j / 2] |= (uint8_t)(q << ((j % 2) * 4));
        }
    }
}

/* Dequant stays in rotated domain — NO inverse Givens.
 * The caller (flash attention) must have pre-rotated Q by the same Givens. */
void dequantize_row_planar_edge(const block_edge_4bit * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_EDGE == 0);
    const int nb = (int)(k / QK_EDGE);

    for (int b = 0; b < nb; b++) {
        float d0 = GGML_FP16_TO_FP32(x[b].d0);
        float d1 = GGML_FP16_TO_FP32(x[b].d1);
        float s0 = d0 / EDGE_4BIT_MAX;
        float s1 = d1 / EDGE_4BIT_MAX;

        for (int j = 0; j < 16; j++) {
            uint8_t idx = (x[b].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[b * QK_EDGE + j] = ((float)idx - EDGE_4BIT_MAX) * s0;
        }
        for (int j = 16; j < 32; j++) {
            uint8_t idx = (x[b].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[b * QK_EDGE + j] = ((float)idx - EDGE_4BIT_MAX) * s1;
        }
    }
}

/* vec_dot: rotates Q by Givens, dots with dequantized-rotated K.
 * <G(Q), G(K)> = <Q, K> since Givens is orthogonal. */
void ggml_vec_dot_planar_edge_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, size_t bx,
                                    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);

    const block_edge_4bit * GGML_RESTRICT x = (const block_edge_4bit *)vx;
    const block_q8_0 * GGML_RESTRICT yq = (const block_q8_0 *)vy;

    const int nb = n / QK_EDGE;
    const int nb_q8 = n / QK8_0;
    float sum = 0.0f;

    /* Dequantize Q from q8_0 to f32 */
    float qf[QK_EDGE * 16];
    int qoff = 0;
    for (int i = 0; i < nb_q8; i++) {
        float d = GGML_FP16_TO_FP32(yq[i].d);
        for (int j = 0; j < QK8_0 && qoff < n; j++) {
            qf[qoff++] = yq[i].qs[j] * d;
        }
    }

    for (int b = 0; b < nb; b++) {
        float d0 = GGML_FP16_TO_FP32(x[b].d0);
        float d1 = GGML_FP16_TO_FP32(x[b].d1);
        float s0 = d0 / EDGE_4BIT_MAX;
        float s1 = d1 / EDGE_4BIT_MAX;

        const float * q = qf + b * QK_EDGE;

        for (int p = 0; p < 16; p++) {
            float rq0 = EDGE_COS[p] * q[2*p] - EDGE_SIN[p] * q[2*p + 1];
            float rq1 = EDGE_SIN[p] * q[2*p] + EDGE_COS[p] * q[2*p + 1];

            int j0 = 2*p;
            int j1 = 2*p + 1;
            uint8_t i0 = (x[b].qs[j0 / 2] >> ((j0 % 2) * 4)) & 0xF;
            uint8_t i1 = (x[b].qs[j1 / 2] >> ((j1 % 2) * 4)) & 0xF;

            float rk0 = ((float)i0 - EDGE_4BIT_MAX) * s0;
            float rk1 = ((float)i1 - EDGE_4BIT_MAX) * ((j1 < 16) ? s0 : s1);

            sum += rq0 * rk0 + rq1 * rk1;
        }
    }

    *s = sum;
}

size_t quantize_planar_edge(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_EDGE == 0);
    size_t row_size = (n_per_row / QK_EDGE) * sizeof(block_edge_4bit);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar_edge_ref(
            src + row * n_per_row,
            (block_edge_4bit *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

/* ===================================================================
 * Iso-edge: quaternion 4D rotation
 * =================================================================== */

void quantize_row_iso_edge_ref(const float * GGML_RESTRICT x, block_edge_4bit * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_EDGE == 0);
    const int nb = (int)(k / QK_EDGE);

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * QK_EDGE;
        block_edge_4bit * blk = &y[b];

        float rot[QK_EDGE];
        for (int g = 0; g < 8; g++) {
            float v0 = src[4*g];
            float v1 = src[4*g + 1];
            float v2 = src[4*g + 2];
            float v3 = src[4*g + 3];
            float qw = EDGE_QW[g], qx = EDGE_QX[g], qy = EDGE_QY[g], qz = EDGE_QZ[g];
            rot[4*g]     = qw*v0 - qx*v1 - qy*v2 - qz*v3;
            rot[4*g + 1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
            rot[4*g + 2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
            rot[4*g + 3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
        }

        float d0 = edge_rms16(rot);
        float d1 = edge_rms16(rot + 16);

        blk->d0 = GGML_FP32_TO_FP16(d0);
        blk->d1 = GGML_FP32_TO_FP16(d1);

        memset(blk->qs, 0, QK_EDGE / 2);
        for (int j = 0; j < QK_EDGE; j++) {
            float d = (j < 16) ? d0 : d1;
            uint8_t q = edge_quant4(rot[j], d);
            blk->qs[j / 2] |= (uint8_t)(q << ((j % 2) * 4));
        }
    }
}

/* Dequant stays in rotated domain — NO inverse quaternion. */
void dequantize_row_iso_edge(const block_edge_4bit * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_EDGE == 0);
    const int nb = (int)(k / QK_EDGE);

    for (int b = 0; b < nb; b++) {
        float d0 = GGML_FP16_TO_FP32(x[b].d0);
        float d1 = GGML_FP16_TO_FP32(x[b].d1);
        float s0 = d0 / EDGE_4BIT_MAX;
        float s1 = d1 / EDGE_4BIT_MAX;

        for (int j = 0; j < 16; j++) {
            uint8_t idx = (x[b].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[b * QK_EDGE + j] = ((float)idx - EDGE_4BIT_MAX) * s0;
        }
        for (int j = 16; j < 32; j++) {
            uint8_t idx = (x[b].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[b * QK_EDGE + j] = ((float)idx - EDGE_4BIT_MAX) * s1;
        }
    }
}

/* vec_dot: rotates Q by quaternion, dots with dequantized-rotated K.
 * <quat(Q), quat(K)> = <Q, K> since quaternion rotation is orthogonal. */
void ggml_vec_dot_iso_edge_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, size_t bx,
                                 const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);

    const block_edge_4bit * GGML_RESTRICT x = (const block_edge_4bit *)vx;
    const block_q8_0 * GGML_RESTRICT yq = (const block_q8_0 *)vy;

    const int nb = n / QK_EDGE;
    const int nb_q8 = n / QK8_0;
    float sum = 0.0f;

    float qf[QK_EDGE * 16];
    int qoff = 0;
    for (int i = 0; i < nb_q8; i++) {
        float d = GGML_FP16_TO_FP32(yq[i].d);
        for (int j = 0; j < QK8_0 && qoff < n; j++) {
            qf[qoff++] = yq[i].qs[j] * d;
        }
    }

    for (int b = 0; b < nb; b++) {
        float d0 = GGML_FP16_TO_FP32(x[b].d0);
        float d1 = GGML_FP16_TO_FP32(x[b].d1);
        float s0 = d0 / EDGE_4BIT_MAX;
        float s1 = d1 / EDGE_4BIT_MAX;

        const float * q = qf + b * QK_EDGE;

        for (int g = 0; g < 8; g++) {
            float qw = EDGE_QW[g], qx = EDGE_QX[g], qy = EDGE_QY[g], qz = EDGE_QZ[g];
            float rq0 = qw*q[4*g]   - qx*q[4*g+1] - qy*q[4*g+2] - qz*q[4*g+3];
            float rq1 = qw*q[4*g+1] + qx*q[4*g]   + qy*q[4*g+3] - qz*q[4*g+2];
            float rq2 = qw*q[4*g+2] - qx*q[4*g+3] + qy*q[4*g]   + qz*q[4*g+1];
            float rq3 = qw*q[4*g+3] + qx*q[4*g+2] - qy*q[4*g+1] + qz*q[4*g];

            for (int c = 0; c < 4; c++) {
                int j = 4*g + c;
                uint8_t idx = (x[b].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
                float sk = (j < 16) ? s0 : s1;
                float rk = ((float)idx - EDGE_4BIT_MAX) * sk;
                float rq = (c == 0) ? rq0 : (c == 1) ? rq1 : (c == 2) ? rq2 : rq3;
                sum += rq * rk;
            }
        }
    }

    *s = sum;
}

size_t quantize_iso_edge(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_EDGE == 0);
    size_t row_size = (n_per_row / QK_EDGE) * sizeof(block_edge_4bit);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso_edge_ref(
            src + row * n_per_row,
            (block_edge_4bit *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
