# -*- coding: utf-8 -*-
"""
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
"""

#cython: language_level=3
#distutils: language=c

from libc cimport stdint
from libcpp cimport bool

ctypedef stdint.uint8_t  _uint8_t
ctypedef stdint.uint16_t _uint16_t
ctypedef stdint.uint32_t _uint32_t

cdef extern from "yars.c":

    ctypedef struct yarsStateSt:
        float *  ring       #/*Ring buffer memory*/
        float    freq_ratio #/*f_out / f_in*/
        float    phase      #/*Output phase*/
        _uint8_t index      #/*Ring buffer index*/

    ctypedef struct yarsCfgSt:
        float *   polly    #/*Window pollynom coefficients*/
        float     fudge  #/*Fudge factor*/
        float     window #/*Window scale factor*/
        _uint16_t ntaps #/*Number of filter taps*/
        _uint8_t  npolly #/*Length of window pollynom coefficients*/

    cdef yarsCfgSt yars_defaults;

    cdef float yars_sinc(float x)

    cdef float yars_even_poly(float *p, _uint8_t n, float x)

    cdef float yasr_weight(yarsCfgSt * cfg, float x)

    cdef float yars_run(yarsCfgSt * cfg, yarsStateSt * state,
                        float (*input_cb)(void *), void * cbarg)

#==============================================================================
import  numpy as np
cimport numpy as np

# We need traceback to print pythonic callback exceptions
import traceback as tb

#==============================================================================
def sinc(float x):
    return yars_sinc(x)

#==============================================================================
def weight(np.ndarray x, dict cfg=None):

    cdef yarsCfgSt _cfg = yars_defaults
    cdef np.ndarray  _polly
    cdef float [::1] v_polly

    if cfg:
        _cfg.fudge  = <float>(cfg['fudge'])
        _cfg.window = <float>(cfg['window'])
        _cfg.ntaps  = <_uint16_t>(cfg['ntaps'])
        _cfg.npolly = <_uint8_t>len(cfg['polly'])

        _polly     = cfg['polly'].astype(np.float32)
        v_polly    = _polly
        _cfg.polly = &v_polly[0]

    x32 = x.astype(np.float32)
    cdef float [::1] v_x = x32

    y32 = np.zeros_like(x32)
    cdef float [::1] v_y = y32

    for i in range(len(x)):
        v_y[i] = yasr_weight(&_cfg, v_x[i])

    return y32

#==============================================================================
cdef float _yars_input_cb(void * py_object):
    try:
        if not isinstance(<object>py_object, Resampler):
            raise ValueError('Invalid py_self type (must be subclass of yarsResampler)!')

        py_self = <Resampler>py_object

        callback = py_self._input_cb
        if not callable(callback):
            raise ValueError('callback must be callable!')

        return <float>(callback(py_self))

    except Exception as e:
        print(tb.format_exc())
        return 0.0

#==============================================================================
cdef class Resampler:

    cdef yarsStateSt state
    cdef yarsCfgSt cfg

    cdef np.ndarray  _polly
    cdef float [::1] v_polly

    cdef np.ndarray  _ring
    cdef float [::1] v_ring

    cdef object _input_cb

    #--------------------------------------------------------------------------
    def __init__(self, input_cb=None, ratio=1, cfg=None):

        if not cfg:
            self.cfg = yars_defaults
        else:
            self.cfg.fudge  = <float>(cfg['fudge'])
            self.cfg.window = <float>(cfg['window'])
            self.cfg.ntaps  = <_uint16_t>(cfg['ntaps'])
            self.cfg.npolly = <_uint8_t>len(cfg['polly'])

            self._polly     = cfg['polly'].astype(np.float32)
            self.v_polly    = self._polly
            self.cfg.polly  = &self.v_polly[0]

        self._input_cb = input_cb
        self._ring = np.zeros((self.cfg.ntaps,), dtype=np.float32)
        self.v_ring = self._ring

        self.state.ring = &self.v_ring[0]
        self.state.freq_ratio = ratio
        self.state.phase = 0.0
        self.state.index = 0

    #--------------------------------------------------------------------------
    def __call__(self):
        return yars_run(&self.cfg, &self.state, _yars_input_cb, <void *>self)

    #--------------------------------------------------------------------------
    @property
    def ntaps(self):
        return self.cfg.ntaps

    #--------------------------------------------------------------------------
    @property
    def ratio(self):
        return self.state.freq_ratio

    @ratio.setter
    def ratio(self, value):
        self.state.freq_ratio = value
