/******************************************************************************
    Copyright 2026 anonimous <shkolnick-kun@gmail.com> and contributors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

    See the License for the specific language governing permissions
    and limitations under the License.
******************************************************************************/

/**
 * \file yars.c
 * \brief Implementation of the YARS resampler algorithm (f32, i32 and i16).
 */

#include "yars.h"

/*===========================================================================*/
/*                     Common constants for fixed‑point (Q8.24)              */
/*===========================================================================*/
#define _YARS_HALF_Q24 (1 << 23)
#define _YARS_ONE_Q24  (1 << 24)
#define _YARS_TWO_Q24  (1 << 25)

/*===========================================================================*/
/*                          Floating‑point (f32)                            */
/*===========================================================================*/

/**
 * \brief Kaiser window coefficients for 100 dB stopband (used by the default f32 configuration).
 * \internal
 */
YARS_CONST float _kaiser_100db_f32[] = {
    -3.9717498422e-01,
     2.3297753334e+00,
    -6.4596233368e+00,
     1.1202648163e+01,
    -1.3136844635e+01,
     1.0229041100e+01,
    -4.7674880028e+00,
     1.0000000000e+00
};

/**
 * \brief Default configuration (f32) – Kaiser window with 100 dB stopband, 79 taps.
 */
YARS_CONST yarsCfgF32St yars_f32_defaults = {
    .poly     = _kaiser_100db_f32,
    .fudge     = 0.914244920531081f,
    .window    = 2.5640861277e-02f,
    .ntaps     = 79,
    .npoly    = 8
};

/**
 * \brief Perform one resampling step (f32).
 * \param cfg      Configuration of the filter.
 * \param state    Resampler state.
 * \param input_cb Callback function that returns the next input sample.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample.
 */
float yars_f32_run(YARS_CONST yarsCfgF32St * cfg,
                   yarsStateF32St * state,
                   float (*input_cb)(void *),
                   void * cbarg)
{
    uint16_t ring_size = cfg->ntaps * YARS_MAX_RATIO;
    int32_t k;

    /* Read enough input samples */
    while (state->phase > 0.0f)
    {
        if (++state->index >= ring_size)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);
        state->phase -= 1.0f;
    }

    /* Compute adaptive filter parameters */
    float rinc = (state->freq_ratio > 1.0f) ? state->freq_ratio : 1.0f;
    float step = 1.0f / rinc;
    int32_t tmp = (cfg->ntaps - 1) / 2;
    int32_t d  =  roundf(tmp * rinc);
    float start_phase = (state->phase - (float)d) * step;

    /* Filtering */
    float out = 0.0f;
    int16_t j = state->index;

    for (k = 0; k <= 2 * d; k++)
    {
        float phase = start_phase + (float)k * step;
        out += yars_f32_weight(cfg, phase) / rinc * state->ring[j];

        if (--j < 0)
        {
            j = ring_size - 1;
        }
    }

    /* Clamp the frequency ratio (f_in / f_out) to avoid excessive decimation */
    if (state->freq_ratio < 2.0f / cfg->ntaps)
    {
        state->freq_ratio = 2.0f / cfg->ntaps;
    }

    /* Advance phase */
    state->phase += state->freq_ratio;

    return out;
}

/*===========================================================================*/
/*                      Fixed‑point (i32) implementation                    */
/*===========================================================================*/

/* Default configuration for i32 */
#define _KAISER_I32_100DB \
X(0,  -852926785, 31)     \
X(1,  1250788607, 29)     \
X(2, -1733991937, 28)     \
X(3,  1503593984, 27)     \
X(4, -1763197440, 27)     \
X(5,  1372918656, 27)     \
X(6, -1279762817, 28)     \
X(7,  1073741824, 30)

#define _KAISER_I32_ID(i) _KAISER_I32_ID_##i
typedef enum {
#define X(id, num, dig) _KAISER_I32_ID(id),
    _KAISER_I32_100DB
#undef X
    _KAISER_I32_ID_LIM
} _kaiserI32IdEn;

static YARS_CONST int32_t _kaiser_i32_100db[] = {
#define X(i, Ai, FDAi) Ai,
    _KAISER_I32_100DB
#undef X
};

static YARS_CONST int8_t _fd_kaiser_i32_100db[] = {
#define X(i, Ai, FDAi) FDAi,
    _KAISER_I32_100DB
#undef X
};

YARS_CONST yarsCfgI32St yars_i32_defaults = {
    .poly     = _kaiser_i32_100db,
    .fd_poly  = _fd_kaiser_i32_100db,
    .fudge     = 981663008,
    .window    = 55063330,
    .ntaps     = 79,
    .npoly    = 8,
    .fd_fudge  = 30,
    .fd_window = 31,
};

/**
 * \brief Perform one resampling step using fixed‑point arithmetic (i32).
 * \param cfg      Configuration of the filter (fixed‑point).
 * \param state    Resampler state (fixed‑point).
 * \param input_cb Callback that returns the next input sample as int32_t.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample as int32_t.
 */
int32_t yars_i32_run(YARS_CONST yarsCfgI32St * cfg,
                     yarsStateI32St * state,
                     int32_t (*input_cb)(void *),
                     void * cbarg)
{
    int32_t k;
    uint16_t ring_size = cfg->ntaps * YARS_MAX_RATIO;

    /* Read enough input samples while phase > 0 */
    while (state->phase > 0)
    {
        if (++state->index >= ring_size)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);  /* Q1.31 sample */
        state->phase -= _YARS_ONE_Q24;                /* subtract 1.0 */
    }

    /* Compute adaptive filter parameters in Q8.24 */
    int64_t rinc_q24 = (state->freq_ratio > _YARS_ONE_Q24) ? state->freq_ratio : _YARS_ONE_Q24;
    int64_t step_q24 = (((int64_t)1)<<48) / rinc_q24;

    int32_t d = mul_2pwr64(((int64_t)cfg->ntaps - 1) * rinc_q24, -25);
    int32_t start_phase = mul_2pwr64((state->phase - mul_2pwr64(d, 24)) * step_q24, -24);

    /* Filtering */
    int64_t acc = 0;
    int16_t j = state->index;

    for (k = 0; k <= 2 * d; k++)
    {
        int32_t phase = start_phase + k * step_q24;   /* Q8.24 */
        acc += _yars_i32_weight_calc(cfg, phase) * state->ring[j] / rinc_q24;  /* Q2.30 * Q1.31 / Q8.24 */
        if (--j < 0)
        {
            j = ring_size - 1;
        }
    }

    /* Clamp the frequency ratio (f_in / f_out) to avoid excessive decimation */
    int32_t min_freq_ratio = _YARS_TWO_Q24 / cfg->ntaps;  /* 2/ntaps in Q8.24 */
    if (state->freq_ratio < min_freq_ratio)
    {
        state->freq_ratio = min_freq_ratio;
    }

    /* Advance phase */
    state->phase += state->freq_ratio;

    /* Normalize by rinc and convert to Q1.31 */
    int64_t out = mul_2pwr64(acc, -6);

    if (out >= INT32_MAX)
    {
        out = INT32_MAX;
    }
    if (out <= INT32_MIN)
    {
        out = INT32_MIN;
    }

    return (int32_t)out;
}

/*===========================================================================*/
/*                      Fixed‑point (i16) implementation                    */
/*===========================================================================*/

/* ---- Kaiser window coefficients for i16 (100 dB) ---- */
#define _KAISER_I16_100DB \
X(0,  -13015, 15)     \
X(1,   19085, 13)     \
X(2,  -26459, 12)     \
X(3,   22943, 11)     \
X(4,  -26905, 11)     \
X(5,   20949, 11)     \
X(6,  -19528, 12)     \
X(7,   32767, 15)

/* Helper enum for coefficient identification */
#define _KAISER_I16_ID(i) _KAISER_I16_ID_##i
typedef enum {
#define X(id, num, dig) _KAISER_I16_ID(id),
    _KAISER_I16_100DB
#undef X
    _KAISER_I16_ID_LIM
} _kaiserI16IdEn;

/* Coefficient arrays and their fractional bits */
static YARS_CONST int16_t _kaiser_i16_100db_poly[] = {
#define X(i, Ai, FDAi) Ai,
    _KAISER_I16_100DB
#undef X
};

static YARS_CONST int8_t _fd_kaiser_i16_100db[] = {
#define X(i, Ai, FDAi) FDAi,
    _KAISER_I16_100DB
#undef X
};

/* Default configuration for i16 */
YARS_CONST yarsCfgI16St yars_i16_defaults = {
    .poly     = _kaiser_i16_100db_poly,
    .fd_poly  = _fd_kaiser_i16_100db,
    .fudge    = 981663008,      /* 0.9142449 in Q30 */
    .window   = 55063330,       /* 0.05128 in Q31   */
    .ntaps    = 79,
    .npoly    = 8,
    .fd_fudge = 30,
    .fd_window = 31
};

/**
 * \brief Perform one resampling step using fixed‑point arithmetic (i16).
 * \param cfg      Configuration of the filter (fixed‑point).
 * \param state    Resampler state (fixed‑point).
 * \param input_cb Callback that returns the next input sample as int16_t.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample as int16_t.
 */
int16_t yars_i16_run(YARS_CONST yarsCfgI16St * cfg,
                     yarsStateI16St * state,
                     int16_t (*input_cb)(void *),
                     void * cbarg)
{
    int32_t k;
    uint16_t ring_size = cfg->ntaps * YARS_MAX_RATIO;

    /* Read enough input samples while phase > 0 */
    while (state->phase > 0)
    {
        if (++state->index >= ring_size)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);  /* Q1.15 sample */
        state->phase -= _YARS_ONE_Q24;                /* subtract 1.0 */
    }

    /* Compute adaptive filter parameters in Q8.24 */
    int64_t rinc_q24 = (state->freq_ratio > _YARS_ONE_Q24) ? state->freq_ratio : _YARS_ONE_Q24;
    int64_t step_q24 = (((int64_t)1)<<48) / rinc_q24;

    int32_t d   = mul_2pwr64(((int64_t)cfg->ntaps - 1) * rinc_q24, -25);
    int32_t start_phase = mul_2pwr64((state->phase - mul_2pwr64(d, 24)) * step_q24, -24);

    /* Filtering */
    int32_t acc = 0;  /* Use 64-bit accumulator to avoid overflow */
    int16_t j = state->index;

    for (k = 0; k <= 2 * d; k++)
    {
        int32_t phase = start_phase + k * (int32_t)step_q24;   /* Q8.24 */
        int32_t delta = _yars_i16_weight_calc(cfg, phase) * state->ring[j]; /* Q1.15 * Q1.15 = Q.2.30*/
        acc += mul_2pwr64(delta, 24) / rinc_q24;  /* Q2.30 * Q8.24 / * Q8.24 */
        if (--j < 0)
        {
            j = ring_size - 1;
        }
    }

    /* Clamp the frequency ratio (f_in / f_out) to avoid excessive decimation */
    int32_t min_freq_ratio = _YARS_TWO_Q24 / cfg->ntaps;  /* 2/ntaps in Q8.24 */
    if (state->freq_ratio < min_freq_ratio)
    {
        state->freq_ratio = min_freq_ratio;
    }

    /* Advance phase */
    state->phase += state->freq_ratio;

    /* Normalize by rinc and convert to Q1.15 */
    int64_t out = mul_2pwr32(acc, -15); /* Q1.15 */;

    if (out >= INT16_MAX)
    {
        out = INT16_MAX;
    }
    if (out <= INT16_MIN)
    {
        out = INT16_MIN;
    }

    return (int16_t)out;
}
