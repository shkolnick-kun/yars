/*******************************************************************************
    Copyright 2021 anonimous <shkolnick-kun@gmail.com> and contributors.

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

#ifndef YARS_H
#define YARS_H

#include <stdint.h>
#include <math.h>

#ifndef YARS_CONST
#define YARS_CONST const
#endif/*YARS_CONST*/

typedef struct {
    float *  ring;       /*Ring buffer memory*/
    float    freq_ratio; /*f_out / f_in*/
    float    phase;      /*Output phase*/
    uint16_t index;      /*Ring buffer index*/
} yarsStateSt;

#define YARS_RING(buf, n) float buf[n]

#define YARS_DEFAULT_RING(buf) YARS_RING(buf, 79)

#define YARS_STAE_INITIALIZER(buf, fr) \
{                                      \
    .ring       = buf,                 \
    .freq_ratio = fr,                  \
    .phase      = 0.0,                 \
    .index      = 0                    \
}

typedef struct {
    YARS_CONST float *  polly;  /*Window pollynom coefficients*/
    YARS_CONST float    fudge;  /*Fudge factor*/
    YARS_CONST float    window; /*Window scale factor*/
    YARS_CONST uint16_t ntaps;  /*Number of filter taps*/
    YARS_CONST uint8_t  npolly; /*Length of window pollynom coefficients*/
} yarsCfgSt;

extern YARS_CONST yarsCfgSt yars_defaults;

/*
y = sin(PI * x) / (PI * x) approximation
*/

#define YARS_SINC_GROBE()    \
do {                         \
X( 2.0372393752e-02)         \
Y(-1.8519412803e-01)         \
Y( 8.0935758087e-01)         \
Y(-1.6445131302e+00)         \
Y( 9.9999900000e-01)         \
}while(0)
/*
Last value was hand tuned from:
Y( 9.9997911581e-01)
*/

#define YARS_SINC_FINE() \
do {                     \
X( 1.2431410909e-04)     \
Y(-2.3109053328e-03)     \
Y( 2.6121352135e-02)     \
Y(-1.9074107598e-01)     \
Y( 8.1174018082e-01)     \
Y(-1.6449338599e+00)     \
Y( 9.9999996750e-01)     \
}while(0)

/*
Last value was hand tuned from:
Y( 9.9999999447e-01)
*/

#ifndef YARS_SINC_POLLY
#define YARS_SINC_POLLY YARS_SINC_FINE
#endif/*YARS_SINC_POLLY*/

static inline float yars_sinc(float x)
{
    float ix  = ceil(0.5 * (x - 1.0));
    float fx  = x - 2.0 * ix;
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

static inline float yars_even_polly(YARS_CONST float *p, uint8_t n, float x)
{
    float out = *(p++);

    for (x *= x; n > 1; n--)
    {
        out = out * x + *(p++);
    }
    return out;
}

/*
Filter weight calculation
*/
static inline float yasr_weight(YARS_CONST yarsCfgSt * cfg, float x)
{
    return yars_sinc(x / cfg->fudge) * yars_even_polly(cfg->polly, cfg->npolly, x * cfg->window) / cfg->fudge;
}

float yars_run(YARS_CONST yarsCfgSt * cfg, yarsStateSt * state, float (*input_cb)(void *), void * cbarg);
#endif/*YARS_H*/
