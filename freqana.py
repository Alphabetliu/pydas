# -*- coding: utf-8 -*-

import warnings
import math
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
        N = s.shape[axis]

    if s.shape[axis] < N:
        warnings.warn('WARNING: N > array length!!!')
        tmp = np.zeros(N)
        tmp[:s.shape[axis]] = s
        s = tmp

    df = fs / N
    if N % 2 == 0:
        f = np.arange(N // 2 + 1) * df
    else:
        f = np.arange((N + 1) / 2) * df

    w = np.hanning(N)
    if len(s.shape) > 1:
        if axis == 0:
            for i in range(s.shape[1]):
                s[:N, i] = w * spsig.detrend(s[:N, i], type='constant')
            fft = 2 * pyfftw.interfaces.numpy_fft.rfft(s[:N, :], N, axis=axis)
        else:
            for i in range(s.shape[0]):
                s[i, :N] = w * spsig.detrend(s[i, :N], type='constant')
            fft = 2 * pyfftw.interfaces.numpy_fft.rfft(s[:, :N], N, axis=axis)
    else:
        s[:N] = w * spsig.detrend(s[:N], type='constant')
        fft = 2 * pyfftw.interfaces.numpy_fft.rfft(s[:N], N, axis=axis)

    mag = 2 * np.abs(fft) / N
    mag[0] /= 2  # amplitude of the constant component
    phase = np.angle(fft)

    if not WRAP:
        phase = np.unwrap(phase)

    return f, mag, phase


def psd(s, fs=1, N=None, axis=0, WRAP=True, smoothp=None):
    """
    Calculate 1D power spectrum (using hanning window).
    @param: s - signal (1-D array (or python list))
    @param: fs - sampling frequency (defalt is 1)
    @param: N - FFT point number (int)
    @param: axis - the axis along which to perform fft
                   (0: along column; 1: along row)
    @param: WRAP - whether to unwrap the phase angle
    @param: smoothp - number of points for smoothing (None or 0 or 1 for no smoothing)
    ----
    @return: f - frequency
    @return: psd - magnitude of each frequency
    @return: phase - phase angle of each frequency
    """
    f, mag, phase = fft1d(s, fs, N, axis, WRAP)
    df = f[-1] / (f.shape[0] - 1)
    S = 0.5 * mag**2 / df

    if smoothp not in [None, 0, 1]:
        S = smooth(S, smoothp, axis=axis)

    return f, S, phase


def fft1d_s(s, fs=1, N=None, axis=0, WRAP=None, smoothp=None):
    """
    Perform 1D FFT transform and smooth the spectrum (using hanning window).
    @param: s - signal (1-D array (or python list)),
    @param: fs - sampling frequency (defalt is 1),
    @param: N - FFT point number (int),
    @param: axis - the axis along which to perform fft
                   (0: along column; 1: along row),
    @param: WRAP - whether to unwrap the phase angle.
    @param: smoothp - number of points for smoothing (None or 0 or 1 for no smoothing)
    ----
    @return: f - frequency,
    @return: mag - magnitude of each frequency,
    @return: phase - phase angle of each frequency.
    """
    f, S, phase = psd(s, fs, N, axis, WRAP, smoothp)
    df = f[-1] / (f.shape[0] - 1)
    mag = np.sqrt(2 * S * df)

    return f, mag, phase


def smooth(x, wlen, win='hanning', axis=0):
    """
    Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    @param: x - the input signal
    @param: wlen - the dimension of the smoothing window; should be an odd integer
    @param: win - the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman';
            flat window will produce a moving average smoothing.
    @param: axis - the axis along which to perform smooth
                   (0: along column; 1: along row),
    ----
    @return: the smoothed signal x

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(wlen/2-1):-(wlen/2)] instead of just y.
    """

    # if x.ndim != 1:
    # raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.shape[axis] < wlen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if wlen < 3:
        return x
    if win not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if win == 'flat':  # moving average
        w = np.ones(wlen, 'd')
    else:
        w = eval('np.' + win + '(wlen)')

    if len(x.shape) > 1:
        y = np.empty_like(x)
        if axis == 0:
            for i in range(x.shape[1]):
                s = np.r_[x[wlen - 1:0:-1, i], x[:, i], x[-2:-(wlen + 1):-1, i]]
                y[:, i] = np.convolve(w / w.sum(), s, mode='same')[(wlen - 1):-(wlen - 1)]
        else:
            for i in range(x.shape[0]):
                s = np.r_[x[:, wlen - 1:0:-1], x[i, :], x[i, -2:-(wlen + 1):-1]]
                y[i, :] = np.convolve(w / w.sum(), s, mode='same')[(wlen - 1):-(wlen - 1)]
    else:
        s = np.r_[x[wlen - 1:0:-1], x, x[-2:-(wlen + 1):-1]]
        y = np.convolve(w / w.sum(), s, mode='same')[(wlen - 1):-(wlen - 1)]

    return y


def fftPeaks(mag, phase, fs=1, n=1, threshold=0.2, P=1):
    """
    Calculate the peak value of a fft frequency spectrum,
    including the amplitude and the phase angle.

    @param: mag - magnitude;
    @param: phase - phase angle;
    @param: fs - sampling frequence, default is 1;
    @param: n - number of peaks to return
    @param: threshold - (min value of peaks) / (the max peak value)
    @param: P - parameter, default is 1 (and values other than 1 are not supported
    currently).
    ----
    @return: fr - pear frequence
    @return: amp - peak amplitude
    @return: phi - peak phase angle
    """
    from pydas.statistics import cresti

    N = (mag.shape[0] - 1) * 2

    df = fs / N
    # f = np.arange(N / 2 + 1) * df

    peakid = cresti(mag, mph=threshold * np.max(mag), mpd=4, edge='rising')
    peak = mag[peakid]

    # sort the peaks by magnitude
    order = np.argsort(-peak)
    peakid = peakid[order]
    peak = peak[order]

    nPeaks = min(len(peakid), n)

    fr = np.zeros(nPeaks)
    amp = np.zeros(nPeaks)
    phi = np.zeros(nPeaks)

    for i in range(nPeaks):
        # correcting
        partial_r = ((P + 1 / 2) * mag[peakid[i]] * (mag[peakid[i] + 1] -
                                                     mag[peakid[i] - 1]) /
                     ((mag[peakid[i]] + mag[peakid[i] + 1])
             * (mag[peakid[i]] + mag[peakid[i] - 1])))
        fr[i] = (peakid[i] + partial_r) * df
        Dr = partial_r * (1 / (partial_r**2) - 1 / (partial_r**2 - 1)) *\
            math.sin(math.pi * partial_r)
        mr = partial_r - Dr / (2 * abs(Dr))
        amp[i] = math.pi * peak[i] / abs(Dr)
        phi[i] = phase[peakid[i]] - math.pi * (mr + 0.5)

    return fr, amp, phi
