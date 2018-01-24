# -*- coding: utf-8 -*-

# import math
import warnings
import numpy as np
import scipy.signal as spsig
import pyfftw


def fft1d(s, fs=1, N=None, axis=0, WRAP=True):
    """
    Perform 1D FFT transform (using hanning window).
    @param: s - signal (1-D array (or python list)),
    @param: fs - sampling frequency (defalt is 1),
    @param: N - FFT point number (int),
    @param: axis - the axis along which to perform fft
                   (0: along column; 1: along row),
    @param: WRAP - whether to unwrap the phase angle.
    ----
    @return: f - frequency,
    @return: mag - magnitude of each frequency,
    @return: phase - phase angle of each frequency.
    """
    s = pyfftw.byte_align(s, dtype='float')

    if N is None:
        # N = 2**math.floor(math.log2(s.shape[0]))
        N = s.shape[0]

    if s.shape[0] < N:
        warnings.warn('WARNING: N > array length!!!')
        tmp = np.zeros(N)
        tmp[:s.shape[0]] = s
        s = tmp

    df = fs / N
    if N % 2 == 0:
        f = np.arange(N // 2 + 1) * df
    else:
        f = np.arange((N + 1) / 2) * df

    w = np.hanning(N)
    if len(s.shape) > 1:
        for i in range(s.shape[1]):
            s[:N, i] = w * spsig.detrend(s[:N, i], type='constant')
    else:
        s[:N] = w * spsig.detrend(s[:N], type='constant')

    fft = 2 * pyfftw.interfaces.numpy_fft.rfft(s[:N], N, axis=axis)
    mag = 2 * np.abs(fft) / N
    mag[0] /= 2  # amplitude of the constant component
    phase = np.angle(fft)

    if not WRAP:
        phase = np.unwrap(phase)

    return f, mag, phase
