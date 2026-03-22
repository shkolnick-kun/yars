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

"""
Cython wrapper for the YARS (Yet Another Resampler) library.

This module provides high-performance sample rate conversion using a windowed
sinc filter. It supports arbitrary output-to-input frequency ratios and
integrates seamlessly with NumPy arrays.

Examples:
    >>> import numpy as np
    >>> from yarspy import Resampler
    >>> def source(resampler):
    ...     return np.sin(2 * np.pi * 440.0 * t)  # example
    >>> res = Resampler(input_cb=source, ratio=2.0)
    >>> output = res()  # get next resampled sample
"""

from libc cimport stdint
from libcpp cimport bool

ctypedef stdint.uint8_t  _uint8_t
ctypedef stdint.uint16_t _uint16_t
ctypedef stdint.uint32_t _uint32_t

cdef extern from "yars.c":
    ctypedef struct yarsStateSt:
        float *    ring       # Ring buffer memory
        float      freq_ratio # f_out / f_in
        float      phase      # Output phase
        _uint16_t  index      # Ring buffer index  # fixed: was _uint8_t

    ctypedef struct yarsCfgSt:
        float *    polly      # Window polynomial coefficients
        float      fudge      # Fudge factor
        float      window     # Window scale factor
        _uint16_t  ntaps      # Number of filter taps
        _uint8_t   npolly     # Length of window polynomial coefficients

    cdef yarsCfgSt yars_defaults

    cdef float yars_sinc(float x)

    cdef float yars_even_poly(float *p, _uint8_t n, float x)

    cdef float yasr_weight(yarsCfgSt * cfg, float x)

    cdef float yars_run(yarsCfgSt * cfg, yarsStateSt * state,
                        float (*input_cb)(void *), void * cbarg)

#==============================================================================
import numpy as np
cimport numpy as np
import traceback as tb

#==============================================================================
def sinc(float x):
    """
    Evaluate the approximated sinc function.

    This function returns sin(pi*x)/(pi*x) using a polynomial approximation
    defined by the YARS_SINC_POLLY macro in the C code.

    Args:
        x (float): Input argument.

    Returns:
        float: Approximated sinc value.
    """
    return yars_sinc(x)

#==============================================================================
def weight(np.ndarray x, dict cfg=None):
    """
    Compute filter weights for given phase values.

    Args:
        x (np.ndarray): Array of phase values (floating point).
        cfg (dict, optional): Configuration dictionary with keys:
            'fudge' (float),
            'window' (float),
            'ntaps' (int),
            'polly' (list or np.ndarray).
            If None, the default configuration is used.

    Returns:
        np.ndarray: Array of weights (same shape as x, dtype float32).
    """
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
    """
    C callback invoked by yars_run to obtain the next input sample.

    This function extracts the Python callback stored in the Resampler instance
    and calls it. Any exception raised is printed and a zero sample is returned.

    Args:
        py_object (void *): Pointer to a Python Resampler instance.

    Returns:
        float: Next input sample.
    """
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
    """
    Sample rate converter with fractional delay.

    This class implements a real‑time resampler using a windowed sinc filter.
    It maintains internal state and uses a user‑provided callback to fetch
    input samples on demand.

    Attributes:
        ntaps (int): Number of filter taps (read‑only).
        ratio (float): Current output‑to‑input frequency ratio. Can be changed
            dynamically during processing.

    Examples:
        >>> def source(res):
        ...     # Return a new input sample each call
        ...     return np.random.randn()
        >>> res = Resampler(input_cb=source, ratio=1.5)
        >>> for _ in range(10):
        ...     out = res()
        ...     print(out)
    """

    cdef yarsStateSt state
    cdef yarsCfgSt cfg

    cdef np.ndarray  _polly
    cdef float [::1] v_polly

    cdef np.ndarray  _ring
    cdef float [::1] v_ring

    cdef object _input_cb

    #--------------------------------------------------------------------------
    def __init__(self, input_cb=None, ratio=1, cfg=None):
        """
        Initialize the resampler.

        Args:
            input_cb (callable, optional): A callable that returns the next
                input sample as a float. The callable receives this Resampler
                instance as its only argument. Defaults to None, which will
                cause a ValueError when the resampler is called.
            ratio (float, optional): Initial frequency ratio (output / input).
                Defaults to 1.0.
            cfg (dict, optional): Custom configuration dictionary with the same
                keys as in `weight()`. If None, the default configuration
                (Kaiser 100 dB, 79 taps) is used.
        """
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
        """
        Compute the next output sample.

        This method advances the resampler by one output sample. It will
        internally fetch as many input samples as needed via the callback.

        Returns:
            float: Next output sample.

        Raises:
            ValueError: If no input callback was provided.
        """
        if self._input_cb is None:
            raise ValueError("No input callback provided.")
        return yars_run(&self.cfg, &self.state, _yars_input_cb, <void *>self)

    #--------------------------------------------------------------------------
    @property
    def ntaps(self):
        """
        int: Number of filter taps (read‑only).
        """
        return self.cfg.ntaps

    #--------------------------------------------------------------------------
    @property
    def ratio(self):
        """
        float: Current output‑to‑input frequency ratio.
        """
        return self.state.freq_ratio

    @ratio.setter
    def ratio(self, value):
        """
        Set the frequency ratio dynamically.

        Args:
            value (float): New ratio (must be positive).
        """
        self.state.freq_ratio = value
