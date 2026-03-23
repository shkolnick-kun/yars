/*******************************************************************************
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

#ifndef YARS_CONST
#define YARS_CONST const
#endif /* YARS_CONST */

/**
 * \brief Resampler state (filter state).
 *
 * Contains the ring buffer and current processing parameters.
 */
typedef struct {
    float *  ring;       /**< Pointer to ring buffer memory (user‑allocated) */
    float    freq_ratio; /**< Output frequency divided by input frequency (f_out / f_in) */
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
 * \param fr  Initial frequency ratio.
 */
#define YARS_STAE_INITIALIZER(buf, fr) \
{                                      \
    .ring       = buf,                 \
    .freq_ratio = fr,                  \
    .phase      = 0.0,                 \
    .index      = 0                    \
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
 * \def YARS_SINC_GROBE
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
#else/*YARS_SINC_POLLY*/
#define YARS_SINC_POLLY YARS_SINC_FINE
#endif/*YARS_SINC_POLLY*/

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
static inline float yasr_weight(YARS_CONST yarsCfgSt * cfg, float x)
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

#endif /* YARS_H */
