# -*- coding: utf-8 -*-

import warnings
import numpy as np
import scipy.signal as spsig


def cresti(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False,
           valley=False):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    NOTE:
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.concatenate((dx, [0])) < 0) &
                       (np.concatenate(([0], dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.concatenate((dx, [0])) <= 0) &
                           (np.concatenate(([0], dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.concatenate((dx, [0])) < 0) &
                           (np.concatenate(([0], dx)) >= 0))[0]
    ind = np.unique(np.concatenate((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(
            np.concatenate((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.concatenate(
            ([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0),
            axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def statisticalana(t, series, h=0, mod='cw'):
    '''
    Return: significant values of crests, troughs, and double peaks,
            as well as mean level crossing period
    Parameters:
    -----------
    t: time, 1-d array
    series: data, 1-d array
    h: level, scalar
    mod: string, defines type of wave or crossing returned. Possible options are
        'dw' : downcrossing wave
        'uw' : upcrossing wave
        'cw' : crest wave
        'tw' : trough wave
        None : All crossings will be returned
    '''
    from wafo.objects import mat2timeseries
    from wafo.misc import findtc

    t = np.asarray(t, dtype='float')
    series = np.asarray(series, dtype='float')

    ts = mat2timeseries(np.stack((t, series), axis=-1))
    crestTroughID, crossID = findtc(ts.data, h, mod)

    ctp = series[crestTroughID] - h  # crest and trough point
    crests = ctp[ctp > 0]
    troughs = ctp[ctp <= 0]
    if series[crestTroughID[0]] < h:
        troughs = troughs[1:]
    n = min(crests.shape[0], troughs.shape[0])
    vpps = crests[:n] - troughs[:n]

    # mean level crossing period
    tLevelCrossing = t[crossID[::2]]

    # significant values (1/3 exceeding probability)
    ns = n // 3
    return np.mean(-np.sort(-crests)[:ns]) + h, np.mean(np.sort(troughs)[:ns]) + h,\
        np.mean(-np.sort(-vpps)[:ns]), np.mean(np.diff(tLevelCrossing))


def pExceed(s):
    """
    Calculate Cumulative Distribution of Exceedance Probability of s:
        P(s > X)
    Return: cumulative distribution of exceedance probability
    Parameters: an 1-d array of s
    """

    s = np.sort(s)
    p = 1 - 1 / s.shape[0] * np.arange(s.shape[0])

    return s, p
