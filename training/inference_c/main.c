/*
 * Tiny GPT inference engine — pure portable C, float math.
 * Phase 1: get the architecture right with floats.
 * Phase 2 (later): swap matmuls for fixed-point int8.
 *
 * Compile: clang -O2 -o tinygpt main.c -lm
 * Run:     ./tinygpt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "weights.h"
#define FFN_DIM (4 * N_EMBD)   // nanoGPT MLP hidden dim is 4 * n_embd

/* ============================================================
 * Activation buffers — these would live in SRAM on the MCU.
 * Keep them as globals so we can see the total footprint.
 * ============================================================ */
static float x[N_EMBD];                    // current token embedding
static float x_residual[N_EMBD];           // residual stream backup
static float x_norm[N_EMBD];               // post-layernorm
static float qkv[3 * N_EMBD];              // Q, K, V concatenated for current token
static float ffn_hidden[FFN_DIM];          // FFN intermediate
static float logits[VOCAB_SIZE];
static int8_t x_q[FFN_DIM];   // largest activation we'll quantize is FFN hidden

// KV cache: [layer][position][head][k_or_v][head_dim]
// Flattened: [N_LAYER][BLOCK_SIZE][2 * N_EMBD]
static float kv_cache[N_LAYER][BLOCK_SIZE][2 * N_EMBD];

// Attention scratch
static float attn_scores[BLOCK_SIZE];

/* ============================================================
 * Helpers: dequantize int8 weight to float on the fly.
 * Later we replace this with int8 MAC accumulation.
 * ============================================================ */
static inline float dq(int8_t q, float scale) {
    return (float)q * scale;
}

/* ============================================================
 * Quantize a float vector to int8 symmetric, returning the scale.
 * ============================================================ */
static float quantize_vec(const float *in, int8_t *out, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        for (int i = 0; i < n; i++) out[i] = 0;
        return 1.0f;
    }
    float scale = max_abs / 127.0f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(in[i] * inv_scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
    return scale;
}

/* ============================================================
 * True int8 x int8 -> int32 matvec.
 *   W is row-major (out_dim, in_dim), int8, scale W_scale
 *   in is int8[in_dim], scale in_scale
 *   out is float[out_dim]
 * Output is computed as: out[o] = (W_scale * in_scale) * sum(W[o,i] * in[i])
 * The sum runs entirely in int32. Only the final multiply touches float.
 * ============================================================ */
static void matvec_i8i8_pc(
    const int8_t *W, const float *W_scales,
    const int8_t *in, float in_scale,
    float *out,
    int out_dim, int in_dim)
{
    for (int o = 0; o < out_dim; o++) {
        int32_t acc = 0;
        const int8_t *row = W + o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            acc += (int32_t)row[i] * (int32_t)in[i];
        }
        out[o] = (float)acc * W_scales[o] * in_scale;
    }
}

/* ============================================================
 * RMS-style layernorm (nanoGPT uses LayerNorm without bias when bias=False)
 * Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight
 * ============================================================ */
/* ---- Fast math approximations (match MCU build) ---- */

static float fast_invsqrt(float x) {
    float xhalf = 0.5f * x;
    int32_t i;
    memcpy(&i, &x, 4);
    i = 0x5f3759df - (i >> 1);
    memcpy(&x, &i, 4);
    x *= (1.5f - xhalf * x * x);
    return x;
}

static float fast_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) return 3.4e38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1064866805.0f);
    return u.f;
}

static float fast_tanhf(float x) {
    if (x <= -3.0f) return -1.0f;
    if (x >=  3.0f) return  1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static void layernorm(float *out, const float *in, const float *weight, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)dim;

    float inv_std = fast_invsqrt(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) * inv_std * weight[i];
    }
}

/* ============================================================
 * GELU activation (nanoGPT uses approximate GELU)
 * tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * ============================================================ */
static inline float gelu(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + fast_tanhf(c * (x + 0.044715f * x * x * x)));
}

/* ============================================================
 * Softmax in-place over a length-n array
 * ============================================================ */
static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = fast_expf(x[i] - max);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/* ============================================================
 * Per-layer weight pointers (built from weights.h symbols)
 * ============================================================ */
typedef struct {
    const float *ln1_w;
    const int8_t *attn_qkv_w;  const float *attn_qkv_scales;
    const int8_t *attn_proj_w; const float *attn_proj_scales;
    const float *ln2_w;
    const int8_t *mlp_fc_w;    const float *mlp_fc_scales;
    const int8_t *mlp_proj_w;  const float *mlp_proj_scales;
} Layer;

static Layer layers[N_LAYER];

static void init_layers(void) {
    layers[0].ln1_w = ln1_w_0;
    layers[0].attn_qkv_w  = attn_qkv_w_0;  layers[0].attn_qkv_scales  = attn_qkv_w_0_scales;
    layers[0].attn_proj_w = attn_proj_w_0; layers[0].attn_proj_scales = attn_proj_w_0_scales;
    layers[0].ln2_w = ln2_w_0;
    layers[0].mlp_fc_w    = mlp_fc_w_0;    layers[0].mlp_fc_scales    = mlp_fc_w_0_scales;
    layers[0].mlp_proj_w  = mlp_proj_w_0;  layers[0].mlp_proj_scales  = mlp_proj_w_0_scales;

    layers[1].ln1_w = ln1_w_1;
    layers[1].attn_qkv_w  = attn_qkv_w_1;  layers[1].attn_qkv_scales  = attn_qkv_w_1_scales;
    layers[1].attn_proj_w = attn_proj_w_1; layers[1].attn_proj_scales = attn_proj_w_1_scales;
    layers[1].ln2_w = ln2_w_1;
    layers[1].mlp_fc_w    = mlp_fc_w_1;    layers[1].mlp_fc_scales    = mlp_fc_w_1_scales;
    layers[1].mlp_proj_w  = mlp_proj_w_1;  layers[1].mlp_proj_scales  = mlp_proj_w_1_scales;

    layers[2].ln1_w = ln1_w_2;
    layers[2].attn_qkv_w  = attn_qkv_w_2;  layers[2].attn_qkv_scales  = attn_qkv_w_2_scales;
    layers[2].attn_proj_w = attn_proj_w_2; layers[2].attn_proj_scales = attn_proj_w_2_scales;
    layers[2].ln2_w = ln2_w_2;
    layers[2].mlp_fc_w    = mlp_fc_w_2;    layers[2].mlp_fc_scales    = mlp_fc_w_2_scales;
    layers[2].mlp_proj_w  = mlp_proj_w_2;  layers[2].mlp_proj_scales  = mlp_proj_w_2_scales;
}

/* ============================================================
 * One transformer block forward pass for the current token.
 *   pos     = current token's position in the sequence (0..BLOCK_SIZE-1)
 *   layer_i = which layer
 * Modifies x in place: x = x + attn(ln1(x)) + mlp(ln2(x))
 * ============================================================ */
static void block_forward(int layer_i, int pos) {
    Layer *L = &layers[layer_i];

    /* ---- Self-attention ---- */
    memcpy(x_residual, x, sizeof(x));
    layernorm(x_norm, x, L->ln1_w, N_EMBD);

    // QKV projection: quantize x_norm, then int8 matmul
    float in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(L->attn_qkv_w, L->attn_qkv_scales, x_q, in_scale, qkv, 3 * N_EMBD, N_EMBD);

    // Split into Q, K, V
    float *q = qkv;
    float *k = qkv + N_EMBD;
    float *v = qkv + 2 * N_EMBD;

    // Store K and V for this position into the cache
    for (int i = 0; i < N_EMBD; i++) {
        kv_cache[layer_i][pos][i]           = k[i];
        kv_cache[layer_i][pos][N_EMBD + i]  = v[i];
    }

    // Multi-head attention
    // For each head: scores = q_head @ K_head^T / sqrt(head_dim), softmax, weighted sum of V
    float attn_out[N_EMBD] = {0};
    float scale = fast_invsqrt((float)HEAD_DIM);

    for (int h = 0; h < N_HEAD; h++) {
        float *q_h = q + h * HEAD_DIM;

        // Compute attention scores against all positions 0..pos
        for (int t = 0; t <= pos; t++) {
            float *k_t = &kv_cache[layer_i][t][h * HEAD_DIM];
            float dot = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) dot += q_h[i] * k_t[i];
            attn_scores[t] = dot * scale;
        }

        // Softmax over [0..pos]
        softmax(attn_scores, pos + 1);

        // Weighted sum of values
        for (int t = 0; t <= pos; t++) {
            float *v_t = &kv_cache[layer_i][t][N_EMBD + h * HEAD_DIM];
            float w = attn_scores[t];
            for (int i = 0; i < HEAD_DIM; i++) {
                attn_out[h * HEAD_DIM + i] += w * v_t[i];
            }
        }
    }

    // Output projection
    float attn_proj_out[N_EMBD];
    in_scale = quantize_vec(attn_out, x_q, N_EMBD);
    matvec_i8i8_pc(L->attn_proj_w, L->attn_proj_scales, x_q, in_scale, attn_proj_out, N_EMBD, N_EMBD);

    // Residual
    for (int i = 0; i < N_EMBD; i++) x[i] = x_residual[i] + attn_proj_out[i];

    /* ---- FFN ---- */
    memcpy(x_residual, x, sizeof(x));
    layernorm(x_norm, x, L->ln2_w, N_EMBD);

    // fc: quantize x_norm, then int8 matmul to FFN hidden
    in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(L->mlp_fc_w, L->mlp_fc_scales, x_q, in_scale, ffn_hidden, FFN_DIM, N_EMBD);    // GELU

    for (int i = 0; i < FFN_DIM; i++) ffn_hidden[i] = gelu(ffn_hidden[i]);
    // proj: quantize FFN hidden (post-GELU), then int8 matmul
    float ffn_proj_out[N_EMBD];
    in_scale = quantize_vec(ffn_hidden, x_q, FFN_DIM);
    matvec_i8i8_pc(L->mlp_proj_w, L->mlp_proj_scales, x_q, in_scale, ffn_proj_out, N_EMBD, FFN_DIM);

    for (int i = 0; i < N_EMBD; i++) x[i] = x_residual[i] + ffn_proj_out[i];
}

/* ============================================================
 * Forward pass for a single token at position `pos`.
 * Reads kv_cache for previous positions, writes for current.
 * Returns logits in `logits[]`.
 * ============================================================ */
static void forward(int token_id, int pos) {
    // Token embedding + position embedding
    for (int i = 0; i < N_EMBD; i++) {
        x[i] = (float)wte[token_id * N_EMBD + i] * wte_scales[token_id]
             + (float)wpe[pos       * N_EMBD + i] * wpe_scales[pos];
    }

    // Layer stack
    for (int l = 0; l < N_LAYER; l++) block_forward(l, pos);

    // Final layernorm
    layernorm(x_norm, x, lnf_w, N_EMBD);

    // Logits: quantize x_norm, int8 matmul against tied wte
    float lm_in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(wte, wte_scales, x_q, lm_in_scale, logits, VOCAB_SIZE, N_EMBD);
}

/* ============================================================
 * Sampling: temperature + top-k
 * ============================================================ */
static int sample(float temperature, int top_k) {
    if (temperature == 0.0f) {
        // Argmax
        int best = 0;
        for (int i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    // Apply temperature
    for (int i = 0; i < VOCAB_SIZE; i++) logits[i] /= temperature;

    // Top-k: zero out everything below the k-th largest
    if (top_k > 0 && top_k < VOCAB_SIZE) {
        float sorted[VOCAB_SIZE];
        memcpy(sorted, logits, sizeof(sorted));
        // Partial sort: just find the k-th largest via simple selection
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            for (int j = i + 1; j < VOCAB_SIZE; j++)
                if (sorted[j] > sorted[max_idx]) max_idx = j;
            float tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
        }
        float threshold = sorted[top_k - 1];
        for (int i = 0; i < VOCAB_SIZE; i++)
            if (logits[i] < threshold) logits[i] = -1e9f;
    }

    softmax(logits, VOCAB_SIZE);

    // Categorical sample
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cum += logits[i];
        if (r < cum) return i;
    }
    return VOCAB_SIZE - 1;
}

/* ============================================================
 * Main: generate N tokens starting from a newline
 * ============================================================ */
int main(int argc, char **argv) {
    init_layers();
    srand(42);  // match Python seed for reproducibility comparison

    int max_tokens = 100;
    float temperature = 1.0f;
    int top_k = 200;

    if (argc > 1) max_tokens = atoi(argv[1]);

    // Print config
    fprintf(stderr, "TinyGPT inference engine\n");
    fprintf(stderr, "  vocab=%d  layers=%d  heads=%d  embd=%d  ctx=%d\n",
            VOCAB_SIZE, N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE);
    fprintf(stderr, "  generating %d tokens\n\n", max_tokens);

    // Start with newline token (index 0 in our vocab)
    int token = 0;  // '\n'
    int pos = 0;

    putchar(vocab[token]);
    fflush(stdout);

    clock_t start = clock();
    for (int i = 0; i < max_tokens && pos < BLOCK_SIZE - 1; i++) {
        forward(token, pos);
        token = sample(temperature, top_k);
        putchar(vocab[token]);
        fflush(stdout);
        pos++;
    }
    clock_t end = clock();

    double secs = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "\n\n%.2f tok/s (%.1fms per token)\n",
            max_tokens / secs, secs * 1000.0 / max_tokens);
    return 0;
}