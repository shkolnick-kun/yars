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
 * \brief Implementation of the YARS resampler algorithm (f32 and i32).
 */

#include "yars.h"

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
    /* Read enough input samples */
    while (state->phase > 0.0f)
    {
        if (++state->index >= cfg->ntaps)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);
        state->phase -= 1.0f;
    }

    /* Filtering */
    float   filter_phase = state->phase + 1.0f - 0.5f * (float)(cfg->ntaps - 1);
    float   out = 0.0f;
    int16_t j = state->index;

    for (int16_t i = cfg->ntaps; i > 0; i--)
    {
        out += yars_f32_weight(cfg, filter_phase) * state->ring[j];

        filter_phase += 1.0f;
        if (--j < 0)
        {
            j = cfg->ntaps - 1;
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
/*                      Fixed‑math (i32) implementation                     */
/*===========================================================================*/

/*Default configuration for i32*/
#define _KAISER_I32_100DB \
X(0,  -852926785, 31)    \
X(1,  1250788607, 29)    \
X(2, -1733991937, 28)    \
X(3,  1503593984, 27)    \
X(4, -1763197440, 27)    \
X(5,  1372918656, 27)    \
X(6, -1279762817, 28)    \
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
    const int32_t ONE_Q24 = 1 << 24;          /* 1.0 in Q24 */
    const int32_t HALF_Q24 = 1 << 23;         /* 0.5 in Q24 */

    /* Read enough input samples while phase > 0 */
    while (state->phase > 0)
    {
        if (++state->index >= cfg->ntaps)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);  /* Q31 sample */
        state->phase -= ONE_Q24;                      /* subtract 1.0 */
    }

    /* Filtering */
    int32_t filter_phase = state->phase + ONE_Q24 - ((cfg->ntaps - 1) * HALF_Q24);
    int64_t acc = 0;
    int16_t j = state->index;

    for (int16_t i = cfg->ntaps; i > 0; i--)
    {
        acc += _yars_i32_weight_calc(cfg, filter_phase) * state->ring[j]; /* Q61 */
        filter_phase += ONE_Q24;                               /* +1.0 */
        if (--j < 0)
        {
            j = cfg->ntaps - 1;
        }
    }

    /* Clamp the frequency ratio (f_in / f_out) to avoid excessive decimation */
    int32_t min_freq_ratio = (2 << 24) / cfg->ntaps;  /* 2/ntaps in Q24 */
    if (state->freq_ratio < min_freq_ratio)
    {
        state->freq_ratio = min_freq_ratio;
    }

    /* Advance phase */
    state->phase += state->freq_ratio;

    int64_t out = mul_2pwr32(acc, -30);

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
