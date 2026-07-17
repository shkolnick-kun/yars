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
 * \brief Yet Another Resampler – a real‑time sample rate converter.
 *
 * \details This module implements fractional‑delay resampling using a windowed
 *          sinc filter approximated by polynomials. It supports arbitrary
 *          output‑to‑input frequency ratios.
 */

#ifndef YARS_H
#define YARS_H

#include <stdint.h>
#include <math.h>

#include <stdio.h>

#ifndef YARS_CONST
#define YARS_CONST const
#endif /* YARS_CONST */

/*
TODO: Перейти на freq_ratio = f_in/f_out по всей библиотеке!!!
*/

/**
 * \brief Resampler state (filter state).
 *
 * Contains the ring buffer and current processing parameters.
 */
typedef struct {
    float *  ring;       /**< Pointer to ring buffer memory (user‑allocated) */
    float    freq_ratio; /**< Input frequency divided by output frequency (f_in / f_out) */
    float    phase;      /**< Current output phase in input sample periods */
    uint16_t index;      /**< Index of the most recently written sample in the ring buffer */
} yarsStateSt;

/**
 * \brief Declare a ring buffer.
 * \param buf Name of the buffer variable.
 * \param n   Number of samples (buffer length).
 */
#define YARS_RING(buf, n) float buf[n]

/**
 * \brief Declare a ring buffer with the default size (79).
 * \param buf Name of the buffer variable.
 */
#define YARS_DEFAULT_RING(buf) YARS_RING(buf, 79)

/**
 * \brief Initializer for a resampler state.
 * \param buf Name of the ring buffer variable.
 * \param fr  Initial frequency ratio (f_in / f_out).
 */
#define YARS_STATE_INITIALIZER(buf, fr) \
{                                       \
    .ring       = buf,                  \
    .freq_ratio = fr,                   \
    .phase      = 0.0,                  \
    .index      = 0                     \
}

/**
 * \brief Resampler configuration (filter and approximation parameters).
 */
typedef struct {
    YARS_CONST float *  polly;  /**< Coefficients of the window polynomial */
    YARS_CONST float    fudge;  /**< Scaling factor for sinc correction */
    YARS_CONST float    window; /**< Window scaling factor */
    YARS_CONST uint16_t ntaps;  /**< Number of filter taps */
    YARS_CONST uint8_t  npolly; /**< Length of the window polynomial (number of coefficients) */
} yarsCfgSt;

/**
 * \brief Default configuration (Kaiser window with 100 dB stopband, 79 taps).
 */
extern YARS_CONST yarsCfgSt yars_defaults;

/* --------------------------------------------------------------------------
 * Sinc approximation: sin(pi*x)/(pi*x)
 * -------------------------------------------------------------------------- */

/**
 * \def YARS_SINC_COARSE
 * \brief Coarse sinc approximation (5 coefficients).
 * \hideinitializer
 */

#define YARS_SINC_COARSE()   \
do {                         \
X( 2.0372756182e-02)         \
Y(-1.8519742268e-01)         \
Y( 8.0937197953e-01)         \
Y(-1.6445423865e+00)         \
Y( 9.9998920619e-01)         \
} while(0)

/**
 * \def YARS_SINC_FINE
 * \brief Fine sinc approximation (7 coefficients).
 * \hideinitializer
 */
#define YARS_SINC_FINE() \
do {                     \
X( 1.2431410909e-04)     \
Y(-2.3109053328e-03)     \
Y( 2.6121352135e-02)     \
Y(-1.9074107598e-01)     \
Y( 8.1174018082e-01)     \
Y(-1.6449338599e+00)     \
Y( 9.9999999447e-01)     \
} while(0)

#ifdef YARS_USE_COARSE
#define YARS_SINC_POLLY YARS_SINC_COARSE
#else/*YARS_USE_COARSE*/
#define YARS_SINC_POLLY YARS_SINC_FINE
#endif/*YARS_USE_COARSE*/

/**
 * \brief Evaluate the approximated sinc function.
 * \param x Argument (typically within a few periods).
 * \return Approximated value of sin(pi*x)/(pi*x).
 *
 * \details The approximation uses the polynomial selected by the
 *          \c YARS_SINC_POLLY macro.
 */
static inline float yars_sinc(float x)
{
    float ix  = ceil(0.5f * (x - 1.0f));
    float fx  = x - 2.0f * ix;
    float fx2 = fx * fx;

    float out;
#define X(a) out = (a);
#define Y(a) out = out * fx2 + (a);
    YARS_SINC_POLLY();
#undef X
#undef Y

    if (0 != ix)
    {
        return out * fx / x;
    }
    return out;
}

/**
 * \brief Evaluate an even polynomial.
 * \param p Pointer to coefficient array (highest degree first).
 * \param n Number of coefficients.
 * \param x Argument.
 * \return Value of the polynomial:
 *         p[0] * (x^2)^(n-1) + p[1] * (x^2)^(n-2) + ... + p[n-1].
 *
 * \note Evaluation uses Horner's scheme.
 */
static inline float yars_even_polly(YARS_CONST float *p, uint8_t n, float x)
{
    float out = *(p++);

    for (x *= x; n > 1; n--)
    {
        out = out * x + *(p++);
    }
    return out;
}

/**
 * \brief Compute a filter tap weight for a given phase.
 * \param cfg Pointer to the configuration.
 * \param x Phase (in input sample units) relative to the filter centre.
 * \return Weight value.
 */
static inline float yars_weight(YARS_CONST yarsCfgSt * cfg, float x)
{
    return yars_sinc(x / cfg->fudge) *
           yars_even_polly(cfg->polly, cfg->npolly, x * cfg->window) /
           cfg->fudge;
}

/**
 * \brief Perform one resampling step.
 * \param cfg      Configuration of the filter.
 * \param state    Resampler state.
 * \param input_cb Callback function that returns the next input sample.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample.
 *
 * \details This function reads the required number of input samples
 *          (via the callback), computes a weighted sum of the ring buffer
 *          using the weights obtained from the configuration, and updates
 *          the phase for the next output sample.
 */
float yars_run(YARS_CONST yarsCfgSt * cfg,
               yarsStateSt * state,
               float (*input_cb)(void *),
               void * cbarg);

/*===========================================================================*/
/*                      Fixed math implementation                            */
/*===========================================================================*/

/**
 * \brief Resampler state for fixed-point arithmetic.
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
} yarsQnStateSt;

/**
 * \brief Declare a ring buffer for fixed-point resampler.
 * \param buf Name of the buffer variable.
 * \param n   Number of samples (buffer length).
 */
#define YARS_QN_RING(buf, n) int32_t buf[n]

/**
 * \brief Declare a ring buffer with default size (79 taps) for fixed-point.
 * \param buf Name of the buffer variable.
 */
#define YARS_QN_DEFAULT_RING(buf) YARS_QN_RING(buf, 79)

/**
 * \brief Initializer for a fixed-point resampler state.
 * \param buf          Name of the ring buffer variable.
 * \param freq_ratio_q8_24   Phase increment in Q8.24 format (f_in / f_out scaled by 2^24).
 */
#define YARS_QN_STATE_INITIALIZER(buf, freq_ratio_q8_24) \
{                                                        \
    .ring  = buf,                                        \
    .freq_ratio = freq_ratio_q8_24,                      \
    .phase = 0,                                          \
    .index = 0                                           \
}

/*
Poly data format: XY(id, num, dig)
Where:
    id  - id of record
    num - numerator
    dig - number of fractional binary digits
*/

#define YARS_SINC_QN_COARSE \
X(1,    43750160, 31)       \
X(2,  -397708437, 31)       \
X(3,  1738113091, 31)       \
X(4, -1765813942, 30)       \
X(5,  1073730234, 30)

#define YARS_SINC_QN_FINE \
X(1,      266962, 31)     \
X(2,    -4962632, 31)     \
X(3,    56095176, 31)     \
X(4,  -409613342, 31)     \
X(5,  1743198764, 31)     \
X(6, -1766234284, 30)     \
X(7,  1073741818, 30)


#ifdef YARS_USE_COARSE
#define YARS_SINC_QN_POLLY YARS_SINC_QN_COARSE
#else/*YARS_USE_COARSE*/
#define YARS_SINC_QN_POLLY YARS_SINC_QN_FINE
#endif/*YARS_USE_COARSE*/

/*---------------------------------------------------------------------------*/
#define YARS_SINC_ID(i) YARS_SINC_ID_##i

/*---------------------------------------------------------------------------*/
typedef enum {
#define X(id, num, dig) YARS_SINC_ID(id),
    YARS_SINC_QN_POLLY
#undef X
    YARS_SINC_QN_ID_LIM
} yarsSincIdEn;

/*---------------------------------------------------------------------------*/
typedef struct {
    /**< Coefficients of the window polynomial */
    YARS_CONST int32_t * polly;
    YARS_CONST int8_t  * fd_polly;

    YARS_CONST int32_t   fudge;  /**< Scaling factor for sinc correction*/
    YARS_CONST int32_t   window; /**< Window scaling factor */

    YARS_CONST uint16_t  ntaps;   /**< Number of filter taps */
    YARS_CONST uint8_t   npolly;  /**< Length of the window polynomial (number of coefficients) */

    /*Numbers of fractional binary digits*/
    YARS_CONST int8_t    fd_fudge; /**< Scaling factor for sinc correction*/
    YARS_CONST int8_t    fd_window; /**< Window scaling factor */
} yarsQnCfgSt;

extern YARS_CONST yarsQnCfgSt yars_qn_defaults;
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
static inline int64_t _poly_qn(int64_t x,
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
static inline int64_t _sinc_qn(int32_t x24)
{
    static YARS_CONST int32_t A[] = {
#define X(i, Ai, FDAi) Ai,
        YARS_SINC_QN_POLLY
#undef X
    };

    static YARS_CONST int8_t FDA[] = {
#define X(i, Ai, FDAi) FDAi,
        YARS_SINC_QN_POLLY
#undef X
    };

    int32_t ix   = 31  - (((L32(1, 30) - x24) - L32(1, 24)) >> 25);
    int64_t fx24 = x24 - (ix << 25);     /* fx24 is Q24 */
    int64_t fx2  = (fx24 * fx24)  >> 17; /* fx2  is Q31 */
    int64_t x    = ((int64_t)x24) << 7;  /* x    is Q31 */

    int64_t out  = _poly_qn(fx2, 31, A, FDA, YARS_SINC_QN_ID_LIM);

    if (0 != ix)
    {
        int64_t fx31 = ((int64_t)fx24) << 7; /* fx31 is Q31 */
        out = (out * fx31 / x);
    }
    return out;
}

static inline int32_t yars_sinc_qn(int32_t x24)
{
    return _sinc_qn(x24);
}

/*---------------------------------------------------------------------------*/
static inline int64_t _weight_qn(YARS_CONST yarsQnCfgSt * cfg, int32_t x24)
{
    int64_t xs24 = mul_2pwr(x24, cfg->fd_fudge) / cfg->fudge;
    int64_t sinc = _sinc_qn(xs24);

    int64_t xw31 = mul_2pwr(((int64_t)x24) * ((int64_t)cfg->window), 7 - cfg->fd_window);
    int64_t fx2  = mul_2pwr(xw31 * xw31, -31);
    int64_t poly = _poly_qn(fx2, 31, cfg->polly, cfg->fd_polly, cfg->npolly);

    poly = mul_2pwr(poly, cfg->fd_fudge) / cfg->fudge;

    return mul_2pwr(sinc * poly, -cfg->fd_polly[cfg->npolly - 1]);
}

static inline int32_t yars_weight_qn(YARS_CONST yarsQnCfgSt * cfg, int32_t x24)
{
    return _weight_qn(cfg, x24);
}

#endif /* YARS_H */
