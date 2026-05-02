/*
 * hardgpt on the mspm0g3507 - a tiny character-level transformer running
 * on-chip. weights are int8 quantized and we keep an int8 kv cache so it
 * actually fits in sram. peripherals: 16x2 i2c lcd, three buttons
 * (generate / temp / seed), three status leds, and an active buzzer
 * driven through an npn transistor.
 */
 
#include "ti_msp_dl_config.h"
#include "weights.h"
#include "lcd.h"
#include <stdint.h>
#include <string.h>

#define FFN_DIM            (4 * N_EMBD)
#define MAX_GENERATE       300
#define FIXED_TEMPERATURE  0.8f

/* activation scratch buffers — live in sram, reused every step */
static float  x[N_EMBD];
static float  x_residual[N_EMBD];
static float  x_norm[N_EMBD];
static float  qkv[3 * N_EMBD];
static float  ffn_hidden[FFN_DIM];
static float  logits[VOCAB_SIZE];
static int8_t x_q[FFN_DIM];

/* kv cache stored as int8 with one scale per (layer, position) for k and v */
static int8_t kv_cache[N_LAYER][BLOCK_SIZE][2 * N_EMBD];
static float  kv_scales[N_LAYER][BLOCK_SIZE][2];
static float  attn_scores[BLOCK_SIZE];

/* probability of the most recently sampled token - drives the confidence led */
static float last_token_prob = 0.0f;

/* crude busy-wait. good enough for the human-scale delays we need here. */
static void delay_nops(uint32_t n) {
    volatile uint32_t d = n;
    while (d--) { __asm__("nop"); }
}

/* per-vector symmetric int8 quantization. returns the scale used. */
static float quantize_vec(const float *in, int8_t *out, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = in[i] < 0 ? -in[i] : in[i];
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        for (int i = 0; i < n; i++) out[i] = 0;
        return 1.0f;
    }
    float scale = max_abs * (1.0f / 127.0f);
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int v = (int)(in[i] * inv_scale + (in[i] >= 0 ? 0.5f : -0.5f));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
    return scale;
}

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

/* the famous quake 1/sqrt(x) trick, one newton-raphson pass for accuracy */
static float fast_invsqrt(float x) {
    float xhalf = 0.5f * x;
    int32_t i;
    memcpy(&i, &x, 4);
    i = 0x5f3759df - (i >> 1);
    memcpy(&x, &i, 4);
    x *= (1.5f - xhalf * x * x);
    return x;
}

/* schraudolph's bit-hack exp. plenty accurate once softmax normalizes it. */
static float fast_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) return 3.4e38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1064866805.0f);
    return u.f;
}

/* padé approximation of tanh — about 1% off in [-3, 3], dirt cheap */
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

static inline float gelu(float v) {
    const float c = 0.7978845608f;
    return 0.5f * v * (1.0f + fast_tanhf(c * (v + 0.044715f * v * v * v)));
}

static void softmax(float *v, int n) {
    float max = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max) max = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = fast_expf(v[i] - max);
        sum += v[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) v[i] *= inv_sum;
}

/* a layer's worth of pointers into the giant weights table */
typedef struct {
    const float  *ln1_w;
    const int8_t *attn_qkv_w;  const float *attn_qkv_scales;
    const int8_t *attn_proj_w; const float *attn_proj_scales;
    const float  *ln2_w;
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

static void block_forward(int layer_i, int pos) {
    Layer *L = &layers[layer_i];

    /* self-attention block: layernorm, project to q/k/v, attend, project out */
    memcpy(x_residual, x, sizeof(x));
    layernorm(x_norm, x, L->ln1_w, N_EMBD);

    float in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(L->attn_qkv_w, L->attn_qkv_scales, x_q, in_scale,
                   qkv, 3 * N_EMBD, N_EMBD);

    float *q = qkv;
    float *k = qkv + N_EMBD;
    float *v = qkv + 2 * N_EMBD;

    /* stash the new k and v for this position, quantized down to int8 */
    int8_t *k_dst = &kv_cache[layer_i][pos][0];
    int8_t *v_dst = &kv_cache[layer_i][pos][N_EMBD];
    kv_scales[layer_i][pos][0] = quantize_vec(k, k_dst, N_EMBD);
    kv_scales[layer_i][pos][1] = quantize_vec(v, v_dst, N_EMBD);

    float attn_out[N_EMBD] = {0};
    float attn_scale = fast_invsqrt((float)HEAD_DIM);

    for (int h = 0; h < N_HEAD; h++) {
        float *q_h = q + h * HEAD_DIM;
        for (int t = 0; t <= pos; t++) {
            const int8_t *k_t = &kv_cache[layer_i][t][h * HEAD_DIM];
            float k_scale = kv_scales[layer_i][t][0];
            float dot = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) {
                dot += q_h[i] * ((float)k_t[i] * k_scale);
            }
            attn_scores[t] = dot * attn_scale;
        }
        softmax(attn_scores, pos + 1);
        for (int t = 0; t <= pos; t++) {
            const int8_t *v_t = &kv_cache[layer_i][t][N_EMBD + h * HEAD_DIM];
            float v_scale = kv_scales[layer_i][t][1];
            float w = attn_scores[t];
            for (int i = 0; i < HEAD_DIM; i++) {
                attn_out[h * HEAD_DIM + i] += w * ((float)v_t[i] * v_scale);
            }
        }
    }

    float attn_proj_out[N_EMBD];
    in_scale = quantize_vec(attn_out, x_q, N_EMBD);
    matvec_i8i8_pc(L->attn_proj_w, L->attn_proj_scales, x_q, in_scale,
                   attn_proj_out, N_EMBD, N_EMBD);

    for (int i = 0; i < N_EMBD; i++) x[i] = x_residual[i] + attn_proj_out[i];

    /* feed-forward block: layernorm, expand 4x with gelu, project back */
    memcpy(x_residual, x, sizeof(x));
    layernorm(x_norm, x, L->ln2_w, N_EMBD);

    in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(L->mlp_fc_w, L->mlp_fc_scales, x_q, in_scale,
                   ffn_hidden, FFN_DIM, N_EMBD);

    for (int i = 0; i < FFN_DIM; i++) ffn_hidden[i] = gelu(ffn_hidden[i]);

    float ffn_proj_out[N_EMBD];
    in_scale = quantize_vec(ffn_hidden, x_q, FFN_DIM);
    matvec_i8i8_pc(L->mlp_proj_w, L->mlp_proj_scales, x_q, in_scale,
                   ffn_proj_out, N_EMBD, FFN_DIM);

    for (int i = 0; i < N_EMBD; i++) x[i] = x_residual[i] + ffn_proj_out[i];
}

static void forward(int token_id, int pos) {
    for (int i = 0; i < N_EMBD; i++) {
        x[i] = (float)wte[token_id * N_EMBD + i] * wte_scales[token_id]
             + (float)wpe[pos       * N_EMBD + i] * wpe_scales[pos];
    }
    for (int l = 0; l < N_LAYER; l++) block_forward(l, pos);
    layernorm(x_norm, x, lnf_w, N_EMBD);

    float lm_in_scale = quantize_vec(x_norm, x_q, N_EMBD);
    matvec_i8i8_pc(wte, wte_scales, x_q, lm_in_scale,
                   logits, VOCAB_SIZE, N_EMBD);
}

static void slide_kv_cache(void) {
    for (int l = 0; l < N_LAYER; l++) {
        for (int t = 0; t < BLOCK_SIZE - 1; t++) {
            for (int i = 0; i < 2 * N_EMBD; i++) {
                kv_cache[l][t][i] = kv_cache[l][t + 1][i];
            }
            kv_scales[l][t][0] = kv_scales[l][t + 1][0];
            kv_scales[l][t][1] = kv_scales[l][t + 1][1];
        }
    }
}

/* greedy pick — always take the highest-logit token */
static int sample_argmax(void) {
    int best = 0;
    for (int i = 1; i < VOCAB_SIZE; i++)
        if (logits[i] > logits[best]) best = i;
    last_token_prob = 1.0f;
    return best;
}

/* tiny linear-congruential rng — seedable from button presses */
static uint32_t rng_state = 12345;
static float rand_unit(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / (float)(1u << 24);
}

static int sample_temp(float temperature) {
    if (temperature <= 0.0f) return sample_argmax();
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < VOCAB_SIZE; i++) logits[i] *= inv_temp;
    softmax(logits, VOCAB_SIZE);
    float r = rand_unit();
    float cum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cum += logits[i];
        if (r < cum) {
            last_token_prob = logits[i];
            return i;
        }
    }
    last_token_prob = logits[VOCAB_SIZE - 1];
    return VOCAB_SIZE - 1;
}

/* buttons are active-low — pull-ups are configured in sysconfig */
static int btn_pressed(uint32_t pin_mask) {
    return !(DL_GPIO_readPins(GPIO_BUTTONS_PORT, pin_mask) & pin_mask);
}
static int btn_generate(void) { return btn_pressed(GPIO_BUTTONS_GENERATE_PIN); }
static int btn_temp(void)     { return btn_pressed(GPIO_BUTTONS_TEMP_PIN); }
static int btn_seed(void)     { return btn_pressed(GPIO_BUTTONS_SEED_PIN); }

/* spin until the user lets go of every button, plus a small debounce pause */
static void wait_release(void) {
    while (btn_generate() || btn_temp() || btn_seed()) { }
    delay_nops(200000);
}

/* active buzzer wired to pa22 through an npn switch — pin high turns it on */
static void buzzer_beep(uint32_t nops) {
    DL_GPIO_setPins(GPIO_BUZZER_PORT, GPIO_BUZZER_BUZZER_PIN);
    delay_nops(nops);
    DL_GPIO_clearPins(GPIO_BUZZER_PORT, GPIO_BUZZER_BUZZER_PIN);
}

static void leds_clear_all(void) {
    DL_GPIO_clearPins(GPIO_LEDS_PORT,
        GPIO_LEDS_USER_LED_1_PIN |
        GPIO_LEDS_USER_LED_2_PIN |
        GPIO_LEDS_USER_LED_3_PIN);
}

/* tiny "typewriter" wrapper around the lcd: tracks a 2-row shadow buffer
 * so we can scroll up when generation overflows the second row */
typedef struct {
    int  row;
    int  col;
    char line0[17];
    char line1[17];
} LcdView;

static void lcd_view_reset(LcdView *v) {
    v->row = 0;
    v->col = 0;
    memset(v->line0, 0, sizeof(v->line0));
    memset(v->line1, 0, sizeof(v->line1));
}

static void lcd_scroll_up(LcdView *v) {
    memcpy(v->line0, v->line1, 16);
    memset(v->line1, ' ', 16);
    v->line0[16] = 0;
    lcd_set_cursor(0, 0);
    lcd_puts(v->line0);
    lcd_set_cursor(1, 0);
    lcd_puts("                ");
    lcd_set_cursor(1, 0);
    v->col = 0;
}

static void lcd_advance_line(LcdView *v) {
    if (v->row == 0) {
        v->row = 1;
        v->col = 0;
        lcd_set_cursor(1, 0);
    } else {
        lcd_scroll_up(v);
    }
}

static void lcd_view_putchar(LcdView *v, char c) {
    if (c == '\n') {
        lcd_advance_line(v);
        return;
    }
    if (v->col >= 16) lcd_advance_line(v);
    char *target = (v->row == 0) ? v->line0 : v->line1;
    target[v->col] = c;
    lcd_putchar(c);
    v->col++;
}

static void lcd_view_puts(LcdView *v, const char *s) {
    while (*s) lcd_view_putchar(v, *s++);
}

/* seed options shown on the lcd. the first one feeds a newline so the model
 * starts a fresh utterance; the rest prime it with a shakespeare character. */
static const char        seed_chars[] = {'\n', 'K',  'R',     'L',    'H'    };
static const char *const seed_names[] = {"Random", "KING", "ROMEO", "LORD", "HENRY"};
#define NUM_SEEDS (sizeof(seed_chars) / sizeof(seed_chars[0]))

static void show_idle_screen(void) {
    lcd_clear();
    lcd_puts("HardGPT v1.0");
    lcd_set_cursor(1, 0);
    lcd_puts("Press Generate!");
}

static void show_seed_screen(int seed_idx) {
    lcd_clear();
    lcd_puts("Seed:");
    lcd_set_cursor(1, 0);
    lcd_puts(seed_names[seed_idx]);
}

static void show_temp_screen(float temp) {
    int whole = (int)temp;
    int frac  = (int)((temp - (float)whole) * 100.0f);
    if (frac < 0) frac = -frac;
    char tbuf[6];
    tbuf[0] = '0' + whole;
    tbuf[1] = '.';
    tbuf[2] = '0' + (frac / 10);
    tbuf[3] = '0' + (frac % 10);
    tbuf[4] = 0;

    lcd_clear();
    lcd_puts("Temperature:");
    lcd_set_cursor(1, 0);
    lcd_puts(tbuf);
}

/* look up the vocab index for a seed char; falls back to index 0 if missing */
static int seed_token_id(char seed_char) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (vocab[i] == seed_char) return i;
    }
    return 0;
}

int main(void)
{
    SYSCFG_DL_init();
    init_layers();
    lcd_init();

    /* short beep on boot so we know the board woke up */
    buzzer_beep(2400000);

    int seed_idx = 0;

    while (1) {
        /* idle: show the title and wait for input */
        show_idle_screen();
        leds_clear_all();

        for (;;) {
            if (btn_seed()) {
                seed_idx = (seed_idx + 1) % NUM_SEEDS;
                show_seed_screen(seed_idx);
                wait_release();
                delay_nops(2000000);
                show_idle_screen();
            }

            if (btn_temp()) {
                show_temp_screen(FIXED_TEMPERATURE);
                wait_release();
                delay_nops(1500000);
                show_idle_screen();
            }

            if (btn_generate()) {
                wait_release();
                break;
            }
        }

        /* generate: kick off a beep and start streaming tokens to the lcd */
        buzzer_beep(1600000);
        DL_GPIO_setPins(GPIO_LEDS_PORT, GPIO_LEDS_USER_LED_1_PIN);

        int token_id = seed_token_id(seed_chars[seed_idx]);
        int pos = 0;
        int generated = 0;
        int interrupted = 0;

        memset(kv_cache, 0, sizeof(kv_cache));
        memset(kv_scales, 0, sizeof(kv_scales));

        LcdView view;
        lcd_view_reset(&view);
        lcd_clear();

        /* purely cosmetic — print the seed name as a prefix; the model never sees it */
        lcd_view_puts(&view, seed_names[seed_idx]);
        lcd_view_putchar(&view, ':');
        lcd_view_putchar(&view, ' ');

        while (generated < MAX_GENERATE && !interrupted) {
            forward(token_id, pos);
            token_id = sample_temp(FIXED_TEMPERATURE);

            lcd_view_putchar(&view, vocab[token_id]);

            DL_GPIO_togglePins(GPIO_LEDS_PORT, GPIO_LEDS_USER_LED_2_PIN);
            if (last_token_prob > 0.5f)
                DL_GPIO_setPins(GPIO_LEDS_PORT, GPIO_LEDS_USER_LED_3_PIN);
            else
                DL_GPIO_clearPins(GPIO_LEDS_PORT, GPIO_LEDS_USER_LED_3_PIN);

            pos++;
            generated++;

            if (pos >= BLOCK_SIZE) {
                slide_kv_cache();
                pos = BLOCK_SIZE - 1;
            }

            if (btn_generate()) {
                interrupted = 1;
                wait_release();
            }
        }

        leds_clear_all();

        if (!interrupted) {
            buzzer_beep(1600000);
            delay_nops(1000000);
            buzzer_beep(1600000);
            delay_nops(8000000);
        }
    }
}
