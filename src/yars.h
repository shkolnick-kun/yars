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
 * \file yars.h
 * \brief Yet Another Resampler – a real‑time sample rate converter.
 *
 * \details This module implements fractional‑delay resampling using a windowed
 *          sinc filter approximated by polynomials. It supports arbitrary
 *          output‑to‑input frequency ratios.
 *          Both floating‑point (f32) and fixed‑point (i32) implementations
 *          are provided.
 */

#ifndef YARS_H
#define YARS_H

#include <stdint.h>
#include <math.h>
#include <stdio.h>

#ifndef YARS_CONST
#define YARS_CONST const
#endif /* YARS_CONST */

/*===========================================================================*/
/*                          Floating‑point (f32) API                         */
/*===========================================================================*/

/**
 * \brief Resampler state for floating‑point arithmetic.
 *
 * Contains the ring buffer and current processing parameters.
 */
typedef struct {
    float *  ring;       /**< Pointer to ring buffer memory (user‑allocated) */
    float    freq_ratio; /**< Input frequency divided by output frequency (f_in / f_out) */
    float    phase;      /**< Current output phase in input sample periods */
    uint16_t index;      /**< Index of the most recently written sample in the ring buffer */
} yarsStateF32St;

/**
 * \brief Declare a ring buffer for the f32 resampler.
 * \param buf Name of the buffer variable.
 * \param n   Number of samples (buffer length).
 */
#define YARS_F32_RING(buf, n) float buf[n]

/**
 * \brief Declare a ring buffer with the default size (79 taps).
 * \param buf Name of the buffer variable.
 */
#define YARS_F32_DEFAULT_RING(buf) YARS_F32_RING(buf, 79)

/**
 * \brief Initializer for a f32 resampler state.
 * \param buf Name of the ring buffer variable.
 * \param fr  Initial frequency ratio (f_in / f_out).
 */
#define YARS_F32_STATE_INITIALIZER(buf, fr) \
{                                           \
    .ring       = buf,                      \
    .freq_ratio = fr,                       \
    .phase      = 0.0f,                     \
    .index      = 0                         \
}

/**
 * \brief Resampler configuration for floating‑point.
 */
typedef struct {
    YARS_CONST float *  poly;    /**< Coefficients of the window polynomial */
    YARS_CONST float    fudge;    /**< Scaling factor (multiplier) for sinc correction */
    YARS_CONST float    window;   /**< Window scaling factor */
    YARS_CONST uint16_t ntaps;    /**< Number of filter taps */
    YARS_CONST uint8_t  npoly;   /**< Length of the window polynomial (number of coefficients) */
} yarsCfgF32St;

/**
 * \brief Default configuration for f32 (Kaiser window with 100 dB stopband, 79 taps).
 */
extern YARS_CONST yarsCfgF32St yars_f32_defaults;

/* --------------------------------------------------------------------------
 * Sinc approximation: sin(pi*x)/(pi*x)  (floating‑point)
 * -------------------------------------------------------------------------- */

/**
 * \def YARS_F32_SINC_FINE
 * \brief Fine sinc approximation (7 coefficients).
 * \hideinitializer
 */
#define YARS_F32_SINC_FINE() \
do {                         \
X( 1.2431410909e-04)         \
Y(-2.3109053328e-03)         \
Y( 2.6121352135e-02)         \
Y(-1.9074107598e-01)         \
Y( 8.1174018082e-01)         \
Y(-1.6449338599e+00)         \
Y( 9.9999999447e-01)         \
} while(0)

/**
 * \brief Evaluate the approximated sinc function (f32).
 * \param x Argument (typically within a few periods).
 * \return Approximated value of sin(pi*x)/(pi*x).
 */
static inline float yars_f32_sinc(float x)
{
    float ix  = ceilf(0.5f * (x - 1.0f));
    float fx  = x - 2.0f * ix;
    float fx2 = fx * fx;

    float out;
#define X(a) out = (a);
#define Y(a) out = out * fx2 + (a);
    YARS_F32_SINC_FINE();
#undef X
#undef Y

    if (0 != ix)
    {
        return out * fx / x;
    }
    return out;
}

/**
 * \brief Evaluate an even polynomial in f32.
 * \param p Pointer to coefficient array (highest degree first).
 * \param n Number of coefficients.
 * \param x Argument.
 * \return Value of the polynomial:
 *         p[0] * (x^2)^(n-1) + p[1] * (x^2)^(n-2) + ... + p[n-1].
 */
static inline float yars_f32_even_poly(YARS_CONST float *p, uint8_t n, float x)
{
    float out = *(p++);

    for (x *= x; n > 1; n--)
    {
        out = out * x + *(p++);
    }
    return out;
}

/**
 * \brief Compute a filter tap weight (f32) for a given phase.
 * \param cfg Pointer to the configuration.
 * \param x Phase (in input sample units) relative to the filter centre.
 * \return Weight value.
 */
static inline float yars_f32_weight(YARS_CONST yarsCfgF32St * cfg, float x)
{
    return yars_f32_sinc(x * cfg->fudge) *
           yars_f32_even_poly(cfg->poly, cfg->npoly, x * cfg->window) *
           cfg->fudge;
}

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
                   void * cbarg);

/*===========================================================================*/
/*                      Fixed‑point (i32) API                               */
/*===========================================================================*/

/**
 * \brief Resampler state for fixed‑point arithmetic.
 *
 * Contains ring buffer and current processing parameters in Q-format.
 * - ring: samples are stored in Q1.30 format (sign + 30 fractional bits).
 * - freq_ratio: phase increment per output sample (f_in / f_out) in Q8.24.
 * - phase: current output phase in input sample periods, Q8.24.
 * - index: index of most recently written sample in the ring buffer.
 */
typedef struct {
    int32_t *  ring;       /**< Pointer to ring buffer (user‑allocated) */
    int32_t    freq_ratio; /**< Phase increment (f_in / f_out) in Q8.24 */
    int32_t    phase;      /**< Current phase (Q8.24) */
    uint16_t   index;      /**< Index of last written sample */
} yarsStateI32St;

/**
 * \brief Declare a ring buffer for fixed‑point resampler.
 * \param buf Name of the buffer variable.
 * \param n   Number of samples (buffer length).
 */
#define YARS_I32_RING(buf, n) int32_t buf[n]

/**
 * \brief Declare a ring buffer with default size (79 taps) for fixed‑point.
 * \param buf Name of the buffer variable.
 */
#define YARS_I32_DEFAULT_RING(buf) YARS_I32_RING(buf, 79)

/**
 * \brief Initializer for a fixed‑point resampler state.
 * \param buf          Name of the ring buffer variable.
 * \param freq_ratio_q8_24   Phase increment in Q8.24 format (f_in / f_out scaled by 2^24).
 */
#define YARS_I32_STATE_INITIALIZER(buf, freq_ratio_q8_24) \
{                                                         \
    .ring  = buf,                                         \
    .freq_ratio = freq_ratio_q8_24,                       \
    .phase = 0,                                           \
    .index = 0                                            \
}

/*
Poly data format: XY(id, num, dig)
Where:
    id  - id of record
    num - numerator
    dig - number of fractional binary digits
*/

#define YARS_I32_SINC_FINE \
X(1,      266962, 31)     \
X(2,    -4962632, 31)     \
X(3,    56095176, 31)     \
X(4,  -409613342, 31)     \
X(5,  1743198764, 31)     \
X(6, -1766234284, 30)     \
X(7,  1073741818, 30)

/*---------------------------------------------------------------------------*/
#define YARS_SINC_I32_ID(i) YARS_SINC_I32_ID_##i

/*---------------------------------------------------------------------------*/
typedef enum {
#define X(id, num, dig) YARS_SINC_I32_ID(id),
    YARS_I32_SINC_FINE
#undef X
    YARS_SINC_I32_ID_LIM
} yarsSincI32IdEn;

/*---------------------------------------------------------------------------*/
typedef struct {
    /**< Coefficients of the window polynomial */
    YARS_CONST int32_t * poly;
    YARS_CONST int8_t  * fd_poly;

    YARS_CONST int32_t   fudge;    /**< Scaling factor (multiplier) for sinc correction */
    YARS_CONST int32_t   window;   /**< Window scaling factor */

    YARS_CONST uint16_t  ntaps;    /**< Number of filter taps */
    YARS_CONST uint8_t   npoly;   /**< Length of the window polynomial (number of coefficients) */

    /* Numbers of fractional binary digits */
    YARS_CONST int8_t    fd_fudge; /**< Fractional bits for fudge */
    YARS_CONST int8_t    fd_window; /**< Fractional bits for window */
} yarsCfgI32St;

extern YARS_CONST yarsCfgI32St yars_i32_defaults;

/*---------------------------------------------------------------------------*/
#define L32(n, i) (((int32_t)n) << (i))

/*---------------------------------------------------------------------------*/
static inline int64_t mul_2pwr(int64_t x, int8_t pwr)
{
    uint64_t ax  = (x >= 0         ) ? x              : -x;
    uint64_t sax = (pwr >= 0       ) ? ax << pwr      : ax >> (-pwr);
    uint64_t cax = (sax > INT64_MAX) ? INT64_MAX      : sax;
    return         (x >= 0         ) ? ((int64_t)cax) : -((int64_t)cax);
}

/*---------------------------------------------------------------------------*/
static inline int64_t _yars_i32_poly(int64_t x,
                                        int8_t  fdx,
                                        YARS_CONST int32_t *a,
                                        YARS_CONST  int8_t *fda,
                                        YARS_CONST uint8_t na)
{
    uint8_t i;
    int64_t result = a[0];
    for (i = 1; i < na; i++)
    {
        /*
                            result[i-1] * 2**fda[i]   x * 2**fda[i]
                            ----------------------- * -------------
                                  2**fda[i-1]            2**fdx

        result[i] = a[i] +  ---------------------------------------
                                            2**fda[i]


        or

        result[i] = a[i] + result[i-1] * x * 2**(fda[i] - fda[i - 1] - fdx)
        */
        result = a[i] + mul_2pwr(result * x, fda[i] - fda[i - 1] - fdx);
    }
    return result;
}

/*---------------------------------------------------------------------------*/
/*
input  is in Q24 format
output is in Q30 format
*/
static inline int64_t _yars_i32_sinc(int32_t x24)
{
    static YARS_CONST int32_t A[] = {
#define X(i, Ai, FDAi) Ai,
        YARS_I32_SINC_FINE
#undef X
    };

    static YARS_CONST int8_t FDA[] = {
#define X(i, Ai, FDAi) FDAi,
        YARS_I32_SINC_FINE
#undef X
    };

    int32_t ix   = 31  - (((L32(1, 30) - x24) - L32(1, 24)) >> 25);
    int64_t fx24 = x24 - (ix << 25);     /* fx24 is Q24 */
    int64_t fx2  = (fx24 * fx24)  >> 17; /* fx2  is Q31 */
    int64_t x    = ((int64_t)x24) << 7;  /* x    is Q31 */

    int64_t out  = _yars_i32_poly(fx2, 31, A, FDA, YARS_SINC_I32_ID_LIM);

    if (0 != ix)
    {
        int64_t fx31 = ((int64_t)fx24) << 7; /* fx31 is Q31 */
        out = (out * fx31 / x);
    }
    return out;
}

static inline int32_t yars_i32_sinc(int32_t x24)
{
    return _yars_i32_sinc(x24);
}

/*---------------------------------------------------------------------------*/
static inline int64_t _yars_i32_weight_calc(YARS_CONST yarsCfgI32St * cfg, int32_t x24)
{
    // Multiply x by fudge_real (fudge / 2^{fd_fudge})
    int64_t xs24 = mul_2pwr(((int64_t)x24) * cfg->fudge, -cfg->fd_fudge);
    int64_t sinc = _yars_i32_sinc(xs24);

    int64_t xw31 = mul_2pwr(((int64_t)x24) * ((int64_t)cfg->window), 7 - cfg->fd_window);
    int64_t fx2  = mul_2pwr(xw31 * xw31, -31);
    int64_t poly = _yars_i32_poly(fx2, 31, cfg->poly, cfg->fd_poly, cfg->npoly);

    // Multiply poly by fudge_real
    poly = mul_2pwr(poly * cfg->fudge, -cfg->fd_fudge);

    return mul_2pwr(sinc * poly, -cfg->fd_poly[cfg->npoly - 1]);
}

static inline int32_t yars_i32_weight(YARS_CONST yarsCfgI32St * cfg, int32_t x24)
{
    return _yars_i32_weight_calc(cfg, x24);
}

/**
 * \brief Perform one resampling step (i32).
 * \param cfg      Configuration of the filter (fixed‑point).
 * \param state    Resampler state (fixed‑point).
 * \param input_cb Callback that returns the next input sample as int32_t in Q1.30.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample as int32_t in Q1.30.
 */
int32_t yars_i32_run(YARS_CONST yarsCfgI32St * cfg,
                     yarsStateI32St * state,
                     int32_t (*input_cb)(void *),
                     void * cbarg);

#endif /* YARS_H */
