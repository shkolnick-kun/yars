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
 * \file yars.c
 * \brief Implementation of the YARS resampler algorithm.
 */

#include "yars.h"

/**
 * \brief Kaiser window coefficients for 100 dB stopband (used by the default configuration).
 * \internal
 */
YARS_CONST float _kaiser_100db[] = {
#ifdef YARS_USE_COARSE
    -3.9717492461e-01,
     2.3297750950e+00,
    -6.4596223831e+00,
     1.1202646255e+01,
    -1.3136842728e+01,
     1.0229040146e+01,
    -4.7674875259e+00,
     9.9999994040e-01
#else/*YARS_USE_COARSE*/
    -3.9717498422e-01,
     2.3297753334e+00,
    -6.4596233368e+00,
     1.1202648163e+01,
    -1.3136844635e+01,
     1.0229041100e+01,
    -4.7674880028e+00,
     1.0000000000e+00
#endif/*YARS_USE_COARSE*/
};

/**
 * \brief Default configuration (Kaiser window with 100 dB stopband, 79 taps).
 */
YARS_CONST yarsCfgSt yars_defaults = {
    .polly  = _kaiser_100db,
#ifdef YARS_USE_COARSE
    .fudge  = 1.0889013871e+00,
#else/*YARS_USE_COARSE*/
    .fudge  = 1.0937988033e+00,
#endif/*YARS_USE_COARSE*/
    .window = 2.5640861277e-02,
    .ntaps  = 79,
    .npolly = 8
};

/**
 * \brief Perform one resampling step.
 * \param cfg      Configuration of the filter.
 * \param state    Resampler state.
 * \param input_cb Callback function that returns the next input sample.
 * \param cbarg    Argument passed to the callback.
 * \return Next output sample.
 *
 * \details Implementation of the fractional‑delay resampling algorithm:
 *          - While the phase is positive, new input samples are fetched via the callback.
 *          - The weighted sum of the ring buffer is computed using weights that depend on the current phase shift.
 *          - The frequency ratio is clamped to a minimum value for stability.
 *          - The phase is updated for the next output sample.
 */
float yars_run(YARS_CONST yarsCfgSt * cfg,
               yarsStateSt * state,
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
        out += yasr_weight(cfg, filter_phase) * state->ring[j];

        filter_phase += 1.0f;
        if (--j < 0)
        {
            j = cfg->ntaps - 1;
        }
    }

    /* Clamp the frequency ratio */
    if (state->freq_ratio < 2.0f / cfg->ntaps)
    {
        state->freq_ratio = 2.0f / cfg->ntaps;
    }

    /* Advance phase */
    state->phase += 1.0f / state->freq_ratio;

    return out;
}
