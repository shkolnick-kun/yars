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

Three precision modes are available:
    - YARS_F32 : 32‑bit floating‑point
    - YARS_I32 : 32‑bit fixed‑point (Q1.30 data, Q8.24 phase)
    - YARS_I16 : 16‑bit fixed‑point (Q1.15 data, Q8.24 phase)

Examples:
    >>> import numpy as np
    >>> from yarspy import Resampler, YARS_F32
    >>> def source(resampler):
    ...     return np.sin(2 * np.pi * 440.0 * t)  # example
    >>> res = Resampler(input_cb=source, ratio=2.0, mode=YARS_F32)
    >>> output = res()  # get next resampled sample
"""

from libc cimport stdint
from libcpp cimport bool

ctypedef stdint.uint8_t  _uint8_t
ctypedef stdint.uint16_t _uint16_t
ctypedef stdint.uint32_t _uint32_t

ctypedef stdint.int8_t   _int8_t
ctypedef stdint.int16_t  _int16_t
ctypedef stdint.int32_t  _int32_t
ctypedef stdint.int64_t  _int64_t


# ============================================================================
#  Mode constants
# ============================================================================
YARS_F32 = 0
YARS_I32 = 1
YARS_I16 = 2

# Maximum ratio for ring buffer expansion (must match YARS_MAX_RATIO in yars.h)
YARS_MAX_RATIO = 10


# ============================================================================
#  C API imports from yars.c
# ============================================================================
cdef extern from "yars.c":
    # ---------- f32 ----------
    ctypedef struct yarsStateF32St:
        float *    ring
        float      freq_ratio
        float      phase
        _uint16_t  index

    ctypedef struct yarsCfgF32St:
        float *    poly
        float      fudge
        float      window
        _uint16_t  ntaps
        _uint8_t   npoly

    cdef yarsCfgF32St yars_f32_defaults

    cdef float yars_f32_sinc(float x)
    cdef float yars_f32_even_poly(float *p, _uint8_t n, float x)
    cdef float yars_f32_weight(yarsCfgF32St * cfg, float x)
    cdef float yars_f32_run(yarsCfgF32St * cfg, yarsStateF32St * state,
                            float (*input_cb)(void *), void * cbarg)

    # ---------- i32 ----------
    ctypedef struct yarsStateI32St:
        _int32_t * ring
        _int32_t   freq_ratio
        _int32_t   phase
        _uint16_t  index

    ctypedef struct yarsCfgI32St:
        _int32_t * poly
        _int8_t  * fd_poly
        _int32_t   fudge
        _int32_t   window
        _uint16_t  ntaps
        _uint8_t   npoly
        _int8_t    fd_fudge
        _int8_t    fd_window

    cdef yarsCfgI32St yars_i32_defaults

    cdef _int32_t yars_i32_sinc(_int32_t x24)
    cdef _int32_t yars_i32_weight(yarsCfgI32St * cfg, _int32_t x24)
    cdef _int32_t yars_i32_run(yarsCfgI32St * cfg, yarsStateI32St * state,
                               _int32_t (*input_cb)(void *), void * cbarg)

    # ---------- i16 ----------
    ctypedef struct yarsStateI16St:
        _int16_t * ring
        _int32_t   freq_ratio
        _int32_t   phase
        _uint16_t  index

    ctypedef struct yarsCfgI16St:
        _int16_t * poly
        _int8_t  * fd_poly
        _int32_t   fudge
        _int32_t   window
        _uint16_t  ntaps
        _uint8_t   npoly
        _int8_t    fd_fudge
        _int8_t    fd_window

    cdef yarsCfgI16St yars_i16_defaults

    cdef _int16_t yars_i16_sinc(_int32_t x24)
    cdef _int16_t yars_i16_weight(yarsCfgI16St * cfg, _int32_t x24)
    cdef _int16_t yars_i16_run(yarsCfgI16St * cfg, yarsStateI16St * state,
                               _int16_t (*input_cb)(void *), void * cbarg)


# ============================================================================
import numpy as np
cimport numpy as np
import traceback as tb


# ------------------------------------------------------------------------------
#  Helper functions for configuration handling
# ------------------------------------------------------------------------------

def _compute_fixed_value(float x, int max_bits=32):
    """
    Convert a floating-point number to Q-format with dynamic exponent.

    Args:
        x (float): Value to convert.
        max_bits (int): Maximum number of bits for the integer part
            (32 or 16).  If the scaled value would exceed the representable
            range, the exponent is increased (i.e. the value is scaled down).

    Returns:
        tuple: (integer_value, exponent)
    """
    cdef int fd = 31
    cdef float i = x
    while abs(i) > 0.4999:
        fd -= 1
        i /= 2.0
    # Compute scaled integer
    cdef _int64_t scaled = <_int64_t>(x * (2.0 ** fd))
    # Clamp to the allowed range if max_bits is given
    if max_bits == 16:
        if scaled > 32767:
            # Scale down further
            while scaled > 32767:
                fd -= 1
                scaled >>= 1
        elif scaled < -32768:
            while scaled < -32768:
                fd -= 1
                scaled >>= 1
    return <_int32_t>scaled, <_int8_t>fd


def convert_cfg_to_fixed(cfg, target='i32'):
    """
    Convert a floating-point configuration to fixed-point format.

    The input dictionary must contain the keys:
        'fudge' (float),
        'window' (float),
        'ntaps' (int),
        'poly' (iterable of floats).

    Args:
        cfg (dict): Floating-point configuration.
        target (str): Either 'i32' or 'i16'.

    Returns:
        dict: Dictionary with fields required by the corresponding
            yarsCfgI32St or yarsCfgI16St structure.
    """
    cdef float fudge = <float>cfg['fudge']
    cdef float window = <float>cfg['window']
    cdef int ntaps = <int>cfg['ntaps']
    cdef object poly_in = cfg['poly']

    # Convert poly to a flat list of floats
    try:
        poly_float = [float(v) for v in poly_in]
    except TypeError:
        raise TypeError("poly must be iterable of numbers")

    # Determine bit width for coefficients
    max_bits = 32 if target == 'i32' else 16

    # Convert fudge and window
    fudge_fixed, fd_fudge = _compute_fixed_value(fudge, max_bits=32)  # fudge is always 32-bit
    window_fixed, fd_window = _compute_fixed_value(window, max_bits=32)

    poly_fixed = []
    fd_poly = []
    for p in poly_float:
        fixed, exp = _compute_fixed_value(p, max_bits)
        poly_fixed.append(fixed)
        fd_poly.append(exp)

    return {
        'fudge': fudge_fixed,
        'fd_fudge': fd_fudge,
        'window': window_fixed,
        'fd_window': fd_window,
        'ntaps': ntaps,
        'npoly': len(poly_float),
        'poly': poly_fixed,
        'fd_poly': fd_poly
    }


def print_cfg(cfg, format='float', prefix='', name='my_cfg'):
    """
    Print configuration in the specified format.

    Args:
        cfg (dict): Dictionary with keys 'fudge', 'window', 'ntaps', 'poly'.
        format (str): One of:
            'float'   – floating-point values,
            'fixed'   – fixed-point values with exponents,
            'c_code'  – C initialization code for yarsCfgF32St and yarsCfgI32St structures.
        prefix (str): String added before each output line.
        name (str): Base name for variables in C code (default 'my_cfg').
    """
    if format == 'float':
        print(f"{prefix}fudge  = {cfg['fudge']}")
        print(f"{prefix}window = {cfg['window']}")
        print(f"{prefix}ntaps  = {cfg['ntaps']}")
        print(f"{prefix}npoly = {len(cfg['poly'])}")
        print(f"{prefix}poly  = {list(cfg['poly'])}")

    elif format == 'fixed':
        fixed = convert_cfg_to_fixed(cfg, target='i32')
        print(f"{prefix}fudge    = {fixed['fudge']}  (exp {fixed['fd_fudge']})")
        print(f"{prefix}window   = {fixed['window']}  (exp {fixed['fd_window']})")
        print(f"{prefix}ntaps    = {fixed['ntaps']}")
        print(f"{prefix}npoly   = {fixed['npoly']}")
        print(f"{prefix}poly    = {fixed['poly']}")
        print(f"{prefix}fd_poly = {fixed['fd_poly']}")

    elif format == 'c_code':
        fixed = convert_cfg_to_fixed(cfg, target='i32')
        # Prepare data
        poly_float = list(cfg['poly'])
        poly_fixed = fixed['poly']
        fd_poly = fixed['fd_poly']
        ntaps = cfg['ntaps']
        npoly = len(poly_float)
        fudge = cfg['fudge']
        window = cfg['window']
        fudge_fixed = fixed['fudge']
        fd_fudge = fixed['fd_fudge']
        window_fixed = fixed['window']
        fd_window = fixed['fd_window']

        # Build variable names
        float_arr_name = f"poly_float_{name}"
        fixed_arr_name = f"poly_fixed_{name}"
        fd_arr_name = f"fd_poly_{name}"
        float_cfg_name = f"cfg_float_{name}"
        fixed_cfg_name = f"cfg_fixed_{name}"

        # Output
        print(f"{prefix}// Configuration '{name}'")
        print(f"{prefix}static const float {float_arr_name}[] = {{{', '.join(map(str, poly_float))}}};")
        print(f"{prefix}static const int32_t {fixed_arr_name}[] = {{{', '.join(map(str, poly_fixed))}}};")
        print(f"{prefix}static const int8_t {fd_arr_name}[] = {{{', '.join(map(str, fd_poly))}}};")
        print(f"{prefix}")
        print(f"{prefix}static const yarsCfgF32St {float_cfg_name} = {{")
        print(f"{prefix}    .poly = {float_arr_name},")
        print(f"{prefix}    .fudge = {fudge}f,")
        print(f"{prefix}    .window = {window}f,")
        print(f"{prefix}    .ntaps = {ntaps},")
        print(f"{prefix}    .npoly = {npoly}")
        print(f"{prefix}}};")
        print(f"{prefix}")
        print(f"{prefix}static const yarsCfgI32St {fixed_cfg_name} = {{")
        print(f"{prefix}    .poly = {fixed_arr_name},")
        print(f"{prefix}    .fd_poly = {fd_arr_name},")
        print(f"{prefix}    .fudge = {fudge_fixed},")
        print(f"{prefix}    .window = {window_fixed},")
        print(f"{prefix}    .ntaps = {ntaps},")
        print(f"{prefix}    .npoly = {npoly},")
        print(f"{prefix}    .fd_fudge = {fd_fudge},")
        print(f"{prefix}    .fd_window = {fd_window}")
        print(f"{prefix}}};")
    else:
        raise ValueError("Unsupported format. Use 'float', 'fixed' or 'c_code'.")


# ============================================================================
#  Public functions
# ============================================================================

def sinc(np.ndarray x, int mode=YARS_F32):
    """
    Evaluate the approximated sinc function.

    This function returns sin(pi*x)/(pi*x) using a polynomial approximation
    defined by the YARS_F32_SINC_FINE macro in the C code.

    Args:
        x (np.ndarray): Input argument.
        mode (int): Precision mode, one of YARS_F32, YARS_I32, YARS_I16.
            Default is YARS_F32.

    Returns:
        np.ndarray: Approximated sinc values (float32).
    """
    x32 = x.astype(np.float32)
    cdef float [::1] v_x = x32

    y32 = np.zeros_like(x32)
    cdef float [::1] v_y = y32

    if mode == YARS_I32:
        for i in range(len(x)):
            v_y[i] = yars_i32_sinc(<_int32_t>int(v_x[i] * 2**24)) / 2**30
    elif mode == YARS_I16:
        for i in range(len(x)):
            v_y[i] = yars_i16_sinc(<_int32_t>int(v_x[i] * 2**24)) / 2**15
    else:
        for i in range(len(x)):
            v_y[i] = yars_f32_sinc(v_x[i])

    return y32


def weight(np.ndarray x, dict cfg=None, int mode=YARS_F32):
    """
    Compute filter weights for given phase values.

    Args:
        x (np.ndarray): Array of phase values (floating point).
        cfg (dict, optional): Configuration dictionary with keys:
            'fudge' (float) – scaling multiplier for sinc,
            'window' (float),
            'ntaps' (int),
            'poly' (list or np.ndarray).
            If None, the default configuration is used.
        mode (int): Precision mode, one of YARS_F32, YARS_I32, YARS_I16.
            Default is YARS_F32.

    Returns:
        np.ndarray: Array of weights (same shape as x, dtype float32).
    """
    cdef yarsCfgF32St   _cfg    = yars_f32_defaults
    cdef np.ndarray  _poly
    cdef float [::1] v_poly

    cdef yarsCfgI32St _cfg_i32 = yars_i32_defaults
    cdef np.ndarray  _poly_i32
    cdef _int32_t [::1] v_poly_i32
    cdef np.ndarray  _poly_fd_i32
    cdef _int8_t [::1] v_poly_fd_i32

    cdef yarsCfgI16St _cfg_i16 = yars_i16_defaults
    cdef np.ndarray  _poly_i16
    cdef _int16_t [::1] v_poly_i16
    cdef np.ndarray  _poly_fd_i16
    cdef _int8_t [::1] v_poly_fd_i16

    if cfg:
        # Float config
        _cfg.fudge    = <float>(cfg['fudge'])
        _cfg.window   = <float>(cfg['window'])
        _cfg.ntaps    = <_uint16_t>(cfg['ntaps'])
        _cfg.npoly   = <_uint8_t>len(cfg['poly'])

        _poly     = cfg['poly'].astype(np.float32)
        v_poly    = _poly
        _cfg.poly = &v_poly[0]

        # Fixed configs (i32 and i16)
        # Convert to fixed once and reuse for both if needed
        fixed_i32 = convert_cfg_to_fixed(cfg, target='i32')
        fixed_i16 = convert_cfg_to_fixed(cfg, target='i16')

        # i32
        _cfg_i32.ntaps  = _cfg.ntaps
        _cfg_i32.npoly = _cfg.npoly
        _cfg_i32.fudge = fixed_i32['fudge']
        _cfg_i32.fd_fudge = fixed_i32['fd_fudge']
        _cfg_i32.window = fixed_i32['window']
        _cfg_i32.fd_window = fixed_i32['fd_window']

        _poly_i32 = np.array(fixed_i32['poly'], dtype=np.int32)
        _poly_fd_i32 = np.array(fixed_i32['fd_poly'], dtype=np.int8)
        v_poly_i32 = _poly_i32
        v_poly_fd_i32 = _poly_fd_i32
        _cfg_i32.poly = &v_poly_i32[0]
        _cfg_i32.fd_poly = &v_poly_fd_i32[0]

        # i16
        _cfg_i16.ntaps  = _cfg.ntaps
        _cfg_i16.npoly = _cfg.npoly
        _cfg_i16.fudge = fixed_i16['fudge']
        _cfg_i16.fd_fudge = fixed_i16['fd_fudge']
        _cfg_i16.window = fixed_i16['window']
        _cfg_i16.fd_window = fixed_i16['fd_window']

        _poly_i16 = np.array(fixed_i16['poly'], dtype=np.int16)
        _poly_fd_i16 = np.array(fixed_i16['fd_poly'], dtype=np.int8)
        v_poly_i16 = _poly_i16
        v_poly_fd_i16 = _poly_fd_i16
        _cfg_i16.poly = &v_poly_i16[0]
        _cfg_i16.fd_poly = &v_poly_fd_i16[0]

    x32 = x.astype(np.float32)
    cdef float [::1] v_x = x32

    y32 = np.zeros_like(x32)
    cdef float [::1] v_y = y32

    if mode == YARS_I32:
        for i in range(len(x)):
            v_y[i] = yars_i32_weight(&_cfg_i32, <_int32_t>int(v_x[i] * (2 ** 24))) / (2 ** 30)
    elif mode == YARS_I16:
        for i in range(len(x)):
            v_y[i] = yars_i16_weight(&_cfg_i16, <_int32_t>int(v_x[i] * (2 ** 24))) / (2 ** 15)
    else:
        for i in range(len(x)):
            v_y[i] = yars_f32_weight(&_cfg, v_x[i])

    return y32


# ============================================================================
#  Callback adapters
# ============================================================================

cdef float _yars_input_cb_float(void * py_object) noexcept:
    """
    C callback invoked by yars_f32_run to obtain the next input sample.

    This function extracts the Python callback stored in the Resampler instance
    and calls it. Any exception raised is printed and a zero sample is returned.

    Args:
        py_object (void *): Pointer to a Python Resampler instance.

    Returns:
        float: Next input sample.
    """
    try:
        if not isinstance(<object>py_object, Resampler):
            raise ValueError('Invalid py_self type (must be subclass of Resampler)!')

        py_self = <Resampler>py_object

        callback = py_self._input_cb
        if not callable(callback):
            raise ValueError('callback must be callable!')

        return <float>(callback(py_self))

    except Exception as e:
        print(tb.format_exc())
        return 0.0


cdef _int32_t _yars_input_cb_i32(void * py_object) noexcept:
    """
    C callback invoked by yars_i32_run (fixed-point) to obtain the next input sample.

    Converts the Python float to Q1.30 format.

    Args:
        py_object (void *): Pointer to a Python Resampler instance.

    Returns:
        int32_t: Next input sample in Q1.30 format.
    """
    try:
        if not isinstance(<object>py_object, Resampler):
            raise ValueError('Invalid py_self type (must be subclass of Resampler)!')

        py_self = <Resampler>py_object

        callback = py_self._input_cb
        if not callable(callback):
            raise ValueError('callback must be callable!')

        val = <float>(callback(py_self))
        # Convert to Q1.30
        qn = <_int32_t>int(val * (2 ** 30))
        # Clip to range
        if qn > 0x7FFFFFFF:
            qn = 0x7FFFFFFF
        elif qn < -0x80000000:
            qn = -0x80000000
        return qn

    except Exception as e:
        print(tb.format_exc())
        return 0


cdef _int16_t _yars_input_cb_i16(void * py_object) noexcept:
    """
    C callback invoked by yars_i16_run to obtain the next input sample.

    Converts the Python float to Q1.15 format.

    Args:
        py_object (void *): Pointer to a Python Resampler instance.

    Returns:
        int16_t: Next input sample in Q1.15 format.
    """
    try:
        if not isinstance(<object>py_object, Resampler):
            raise ValueError('Invalid py_self type (must be subclass of Resampler)!')

        py_self = <Resampler>py_object

        callback = py_self._input_cb
        if not callable(callback):
            raise ValueError('callback must be callable!')

        val = <float>(callback(py_self))
        # Convert to Q1.15
        qn = <_int16_t>int(val * (2 ** 15))
        # Clip to range
        if qn > 32767:
            qn = 32767
        elif qn < -32768:
            qn = -32768
        return qn

    except Exception as e:
        print(tb.format_exc())
        return 0


# ============================================================================
#  Resampler class
# ============================================================================

cdef class Resampler:
    """
    Sample rate converter with fractional delay.

    This class implements a real‑time resampler using a windowed sinc filter.
    It maintains internal state and uses a user‑provided callback to fetch
    input samples on demand.

    Attributes:
        ntaps (int): Number of filter taps (read‑only).
        ratio (float): Current input‑to‑output frequency ratio (f_in / f_out).
            Can be changed dynamically during processing.

    Notes:
        The internal ring buffer size is automatically set to
        ntaps * YARS_MAX_RATIO to support adaptive filtering during
        decimation (see YARS_MAX_RATIO constant).

    Examples:
        >>> def source(res):
        ...     # Return a new input sample each call
        ...     return np.random.randn()
        >>> res = Resampler(input_cb=source, ratio=1.5, mode=YARS_F32)
        >>> for _ in range(10):
        ...     out = res()
        ...     print(out)
    """

    cdef yarsStateF32St state
    cdef yarsCfgF32St cfg

    cdef np.ndarray  _poly
    cdef float [::1] v_poly

    cdef np.ndarray  _ring
    cdef float [::1] v_ring

    cdef object _input_cb

    # Fixed-point fields
    cdef int _mode

    # i32
    cdef yarsStateI32St _i32_state
    cdef yarsCfgI32St _i32_cfg
    cdef np.ndarray _i32_ring
    cdef _int32_t [::1] _i32_ring_view
    cdef np.ndarray _i32_poly
    cdef _int32_t [::1] _i32_poly_view
    cdef np.ndarray _i32_fd_poly
    cdef _int8_t [::1] _i32_fd_poly_view

    # i16
    cdef yarsStateI16St _i16_state
    cdef yarsCfgI16St _i16_cfg
    cdef np.ndarray _i16_ring
    cdef _int16_t [::1] _i16_ring_view
    cdef np.ndarray _i16_poly
    cdef _int16_t [::1] _i16_poly_view
    cdef np.ndarray _i16_fd_poly
    cdef _int8_t [::1] _i16_fd_poly_view

    # --------------------------------------------------------------------------
    def __init__(self, input_cb=None, ratio=1.0, cfg=None, int mode=YARS_F32):
        """
        Initialize the resampler.

        Args:
            input_cb (callable, optional): A callable that returns the next
                input sample as a float. The callable receives this Resampler
                instance as its only argument. Defaults to None, which will
                cause a ValueError when the resampler is called.
            ratio (float, optional): Initial frequency ratio (f_in / f_out).
                Defaults to 1.0.
            cfg (dict, optional): Custom configuration dictionary with the same
                keys as in `weight()`. If None, the default configuration
                (Kaiser 100 dB, 79 taps) is used.  For fixed-point modes,
                the dictionary is converted automatically.
            mode (int): Precision mode, one of YARS_F32, YARS_I32, YARS_I16.
                Default is YARS_F32.
        """
        self._input_cb = input_cb
        self._mode = mode

        if mode == YARS_I32:
            # i32 fixed-point
            if cfg is not None:
                fixed = convert_cfg_to_fixed(cfg, target='i32')
                self._i32_cfg.ntaps = <_uint16_t>cfg['ntaps']
                self._i32_cfg.npoly = <_uint8_t>len(cfg['poly'])
                self._i32_cfg.fudge = fixed['fudge']
                self._i32_cfg.fd_fudge = fixed['fd_fudge']
                self._i32_cfg.window = fixed['window']
                self._i32_cfg.fd_window = fixed['fd_window']

                self._i32_poly = np.array(fixed['poly'], dtype=np.int32)
                self._i32_fd_poly = np.array(fixed['fd_poly'], dtype=np.int8)
                self._i32_poly_view = self._i32_poly
                self._i32_fd_poly_view = self._i32_fd_poly
                self._i32_cfg.poly = &self._i32_poly_view[0]
                self._i32_cfg.fd_poly = &self._i32_fd_poly_view[0]
            else:
                self._i32_cfg = yars_i32_defaults

            # Allocate enlarged ring buffer (ntaps * YARS_MAX_RATIO)
            self._i32_ring = np.zeros(self._i32_cfg.ntaps * YARS_MAX_RATIO, dtype=np.int32)
            self._i32_ring_view = self._i32_ring
            self._i32_state.ring = &self._i32_ring_view[0]
            self._i32_state.freq_ratio = <_int32_t>int(ratio * (2 ** 24))
            self._i32_state.phase = 0
            self._i32_state.index = 0

        elif mode == YARS_I16:
            # i16 fixed-point
            if cfg is not None:
                fixed = convert_cfg_to_fixed(cfg, target='i16')
                self._i16_cfg.ntaps = <_uint16_t>cfg['ntaps']
                self._i16_cfg.npoly = <_uint8_t>len(cfg['poly'])
                self._i16_cfg.fudge = fixed['fudge']
                self._i16_cfg.fd_fudge = fixed['fd_fudge']
                self._i16_cfg.window = fixed['window']
                self._i16_cfg.fd_window = fixed['fd_window']

                self._i16_poly = np.array(fixed['poly'], dtype=np.int16)
                self._i16_fd_poly = np.array(fixed['fd_poly'], dtype=np.int8)
                self._i16_poly_view = self._i16_poly
                self._i16_fd_poly_view = self._i16_fd_poly
                self._i16_cfg.poly = &self._i16_poly_view[0]
                self._i16_cfg.fd_poly = &self._i16_fd_poly_view[0]
            else:
                self._i16_cfg = yars_i16_defaults

            # Allocate enlarged ring buffer
            self._i16_ring = np.zeros(self._i16_cfg.ntaps * YARS_MAX_RATIO, dtype=np.int16)
            self._i16_ring_view = self._i16_ring
            self._i16_state.ring = &self._i16_ring_view[0]
            self._i16_state.freq_ratio = <_int32_t>int(ratio * (2 ** 24))
            self._i16_state.phase = 0
            self._i16_state.index = 0

        else:
            # Floating-point (default)
            if cfg is not None:
                self.cfg.fudge    = <float>(cfg['fudge'])
                self.cfg.window   = <float>(cfg['window'])
                self.cfg.ntaps    = <_uint16_t>(cfg['ntaps'])
                self.cfg.npoly   = <_uint8_t>len(cfg['poly'])

                self._poly = cfg['poly'].astype(np.float32)
                self.v_poly = self._poly
                self.cfg.poly = &self.v_poly[0]
            else:
                self.cfg = yars_f32_defaults

            # Allocate enlarged ring buffer
            self._ring = np.zeros(self.cfg.ntaps * YARS_MAX_RATIO, dtype=np.float32)
            self.v_ring = self._ring
            self.state.ring = &self.v_ring[0]
            self.state.freq_ratio = <float>ratio
            self.state.phase = 0.0
            self.state.index = 0

    # --------------------------------------------------------------------------
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

        if self._mode == YARS_I32:
            qn_out = yars_i32_run(&self._i32_cfg, &self._i32_state,
                                  _yars_input_cb_i32, <void *>self)
            return <float>qn_out / (2 ** 30)
        elif self._mode == YARS_I16:
            qn_out = yars_i16_run(&self._i16_cfg, &self._i16_state,
                                  _yars_input_cb_i16, <void *>self)
            return <float>qn_out / (2 ** 15)
        else:
            return yars_f32_run(&self.cfg, &self.state,
                                _yars_input_cb_float, <void *>self)

    # --------------------------------------------------------------------------
    @property
    def ntaps(self):
        """
        int: Number of filter taps (read‑only).
        """
        if self._mode == YARS_I32:
            return self._i32_cfg.ntaps
        elif self._mode == YARS_I16:
            return self._i16_cfg.ntaps
        else:
            return self.cfg.ntaps

    # --------------------------------------------------------------------------
    @property
    def ratio(self):
        """
        float: Current input‑to‑output frequency ratio (f_in / f_out).
        """
        if self._mode == YARS_I32:
            return <float>self._i32_state.freq_ratio / (2 ** 24)
        elif self._mode == YARS_I16:
            return <float>self._i16_state.freq_ratio / (2 ** 24)
        else:
            return self.state.freq_ratio

    @ratio.setter
    def ratio(self, value):
        """
        Set the frequency ratio dynamically.

        Args:
            value (float): New ratio (must be positive).
        """
        if self._mode == YARS_I32:
            self._i32_state.freq_ratio = <_int32_t>int(value * (2 ** 24))
        elif self._mode == YARS_I16:
            self._i16_state.freq_ratio = <_int32_t>int(value * (2 ** 24))
        else:
            self.state.freq_ratio = <float>value
