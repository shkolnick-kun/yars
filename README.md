# YARS – Yet Another ReSampler

A high‑performance sample rate converter with arbitrary output/input frequency ratio.
YARS implements a fractional‑delay filter based on a windowed sinc kernel, optimised for real‑time audio and general signal processing applications.
It is written in C and provides a convenient Cython wrapper for Python, integrating seamlessly with NumPy.

## Features

- **Arbitrary resampling ratio** – `f_out / f_in` can be any positive float (≥ `2 / ntaps`).
- **High‑quality filtering** – uses a Kaiser window with 100 dB stop‑band attenuation (default).
- **Polynomial approximation** – the Kaiser window is approximated by an even Chebyshev polynomial, reducing computational cost.
- **Low latency** – state‑ful design with a ring buffer and a callback‑driven input.
- **C and Python APIs** – the core is in C, while Python bindings are generated with Cython.
- **Filter design tools** – Python scripts (`make_filter.py`) allow custom filter design (taps, attenuation, polynomial degree).

## Requirements

- Python 3.6 or later
- C compiler (gcc, clang, or MSVC)
- [Cython](https://cython.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) (only needed for filter design scripts)
- [matplotlib](https://matplotlib.org/) (for plotting examples)
- [mpmath](https://mpmath.org/) (for high‑precision calculations in `approx.py`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shkolnick-kun/yars.git
   cd yars
   ```
2. Install the dependencies (preferably in a virtual environment):
   ```bash
   pip install cython numpy scipy matplotlib mpmath
   ```
   
## Usage examples
 * Usage with Python
    ```python
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    
    """
    Example: custom filter design with make_filter and usage in yarspy.Resampler.
    """
    
    import sys
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ----------------------------------------------------------------------
    # Ensure the correct paths are available
    # ----------------------------------------------------------------------
    # Assume the script is run from the repository root (or adjust paths accordingly)
    SRC_PATH = 'path_to_yars'   # Adjust it!
    
    # Add filter_design module
    sys.path.insert(0, os.path.join(SRC_PATH, 'filter_design'))
    from make_filter import make_filter, plot_filter
    
    # Add the Cython extension module
    sys.path.insert(0, os.path.join(SRC_PATH, 'src'))
    import yarspy
    
    # ----------------------------------------------------------------------
    # 1. Design a custom filter
    # ----------------------------------------------------------------------
    # Parameters
    taps = 127                 # filter length (must be odd)
    attenuation = 80.0         # desired stop‑band attenuation (dB)
    oversample = 1000          # oversampling factor for design (higher = more accurate)
    max_degree = 20            # maximum polynomial degree for window approximation
    tolerance = 1e-6           # allowed error in polynomial approximation
    
    # Run the filter design
    config, stop_atten, stop_start, minus3db, f_scaled, m = make_filter(
        taps=taps,
        attenuation=attenuation,
        oversample=oversample,
        max_polly_degree=max_degree,
        tolerance=tolerance
    )
    
    # ----------------------------------------------------------------------
    # 2. Inspect the results
    # ----------------------------------------------------------------------
    print("\n=== Designed Filter Configuration ===")
    print(f"ntaps        : {config['ntaps']}")
    print(f"fudge        : {config['fudge']:.10e}")
    print(f"window factor: {config['window']:.10e}")
    print(f"polly degree : {len(config['polly'])}")
    print(f"stop‑band attenuation achieved: {abs(stop_atten):.2f} dB")
    print(f"-3 dB frequency (normalised)  : {minus3db:.5f}")
    
    # Plot the frequency response (optional)
    target = 0.5 / oversample
    plot_filter(f_scaled / oversample, attenuation, stop_start, minus3db, target,
                filename="custom_filter_response.png")
    plt.show()
    
    # ----------------------------------------------------------------------
    # 3. Use the custom filter with the resampler
    # ----------------------------------------------------------------------
    # Input signal: 440 Hz sine wave at 8 kHz sample rate
    fs_in = 8000.0
    duration = 0.05                     # seconds
    t_in = np.linspace(0, duration, int(fs_in * duration), endpoint=False)
    x_in = np.sin(2 * np.pi * 440.0 * t_in)
    
    # Resampling ratio: output rate = 12 kHz (ratio = 1.5)
    ratio = 12000.0 / 8000.0
    
    # Callback that feeds the resampler
    idx = 0
    def input_cb(resampler):
        global idx
        if idx < len(x_in):
            s = x_in[idx]
            idx += 1
            return s
        else:
            return 0.0
    
    # Create the resampler with the custom configuration
    res = yarspy.Resampler(input_cb, ratio=ratio, cfg=config)
    
    # Generate output samples
    y_out = []
    while idx < len(x_in):
        y_out.append(res())
    
    y_out = np.array(y_out)
    
    # Time axis for output (based on output sample rate)
    fs_out = ratio * fs_in
    t_out = np.linspace(0, duration, len(y_out), endpoint=False)
    
    # Plot input vs output
    plt.figure(figsize=(10, 6))
    plt.plot(t_in, x_in, 'b-', label='Input (8 kHz)', alpha=0.7)
    plt.plot(t_out, y_out, 'r-', label='Resampled (12 kHz)', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Resampling with custom filter (127 taps, 80 dB stop‑band)')
    plt.grid(True)
    plt.show()
    ```

 * Usage with C
    ```C
    #include <stdio.h>
    #include <math.h>
    #include "yars.h"

    /* ------------------------------------------------------------
     * Input signal generator (440 Hz sine wave, f_in = 8000 Hz)
     * ------------------------------------------------------------ */
    typedef struct {
        double t;           /* current time (sec) */
        double dt;          /* input sample period (1/f_in) */
        double freq;        /* sine frequency (Hz) */
    } sine_gen_t;
    
    float input_callback(void *arg) {
        sine_gen_t *gen = (sine_gen_t *)arg;
        float sample = (float)sin(2.0 * M_PI * gen->freq * gen->t);
        gen->t += gen->dt;          /* advance time */
        return sample;
    }
    
    /* ------------------------------------------------------------
     * Main program
     * ------------------------------------------------------------ */
    int main(void) {
        /* Signal parameters */
        const double f_in  = 8000.0;     /* input sample rate (Hz) */
        const double f_out = 16000.0;    /* output sample rate (Hz) */
        const double ratio = f_out / f_in;  /* resampling ratio = 2.0 */
    
        /* Ring buffer — default size (79) */
        YARS_DEFAULT_RING(ring_buf);    /* declares array float ring_buf[79] */
    
        /* Initialise resampler state */
        yarsStateSt state = YARS_STAE_INITIALIZER(ring_buf, ratio);
    
        /* 440 Hz sine wave generator */
        sine_gen_t generator = {
            .t    = 0.0,
            .dt   = 1.0 / f_in,
            .freq = 440.0
        };
    
        /* Number of output samples to generate */
        const int output_samples = 200;
    
        /* Use default configuration (Kaiser window 100 dB, 79 taps) */
        const yarsCfgSt *cfg = &yars_defaults;
    
        /* Generate output samples */
        for (int i = 0; i < output_samples; ++i) {
            float out = yars_run(cfg, &state, input_callback, &generator);
            printf("%f\n", out);
        }
    
        return 0;
    }
    ```


