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

#include "yars.h"

YARS_CONST float _kaiser_100db[] = {
    -3.9717498422e-01,
     2.3297753334e+00,
    -6.4596233368e+00,
     1.1202648163e+01,
    -1.3136844635e+01,
     1.0229041100e+01,
    -4.7674880028e+00,
     1.0000000000e+00
};

YARS_CONST yarsCfgSt yars_defaults = {
    .polly  = _kaiser_100db,
    .fudge  = 1.0937988033e+00,
    .window = 2.5640861277e-02,
    .ntaps  = 79,
    .npolly = 8
};

float yars_run(YARS_CONST yarsCfgSt * cfg, yarsStateSt * state, float (*input_cb)(void *), void * cbarg)
{
    /* Read sufficient amount of input signal points */
    while (state->phase > 0.0)
    {
        if (++state->index >= cfg->ntaps)
        {
            state->index = 0;
        }
        state->ring[state->index] = input_cb(cbarg);
        state->phase -= 1.0;
    }

    /* Filter input data */
    float   filter_phase = state->phase + 1.0 - 0.5 * (float)(cfg->ntaps - 1);
    float   out = 0;
    int16_t j = state->index;

    for (int16_t i = cfg->ntaps; i > 0; i--)
    {
        out += yasr_weight(cfg, filter_phase) * state->ring[j];

        filter_phase += 1.0;
        if (--j < 0)
        {
            j = cfg->ntaps - 1;
        }
    }

    /* Limit the ratio */
    if (state->freq_ratio < 2.0 / cfg->ntaps)
    {
        state->freq_ratio = 2.0 / cfg->ntaps;
    }

    /* Increase the phase */
    state->phase += 1.0 / state->freq_ratio;

    return out;
}
