#!/usr/bin/env python
# encoding: utf-8
"""
gammatone.py

Created by mi tian on 2016-02-15.
Copyright (c) 2016 Queen Mary University of London. All rights reserved.

Features extracted from the gammatonegram.

"""

import sys
import os
import numpy as np
from sklearn.decomposition import PCA

from .. import util
from .. import filters
from ..util.exceptions import ParameterError
from ..util import Deprecated, rename_kw

from ..core.time_frequency import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import logamplitude, _spectrogram
from ..core.gammatonegram import Gammatonegram, Filters
erb_filters = Filters()
gtgram = Gammatonegram()

__all__ = ['gammatone_cepstral_coeffecients', 'gammatone_contrast', 'gammatone_polynomial']


'''Functions for feature extraction.'''

def gammatone_cepstral_coeffecients(y, sr=44100, nfft=2048, hop_length=1024, nfilters=64, f_min=50, f_max=22050,\
		nCoeff=20, log=False):
	'''Gammatone cepstral coefficients (GTCC)
	Parameters:
	-----------

	Return:
	-----------
	GTCC : np.ndarray [shape=(nCoeff, t)]	
	'''

	gt_weights, gt = gtgram.fft_gtgram(y, sr, nfft, nfilters, f_min, nfft, hop_length, nfft)
	
	if log:
		gt = np.log(gt + 1e-5)

	gtcc = np.dot(filters.dct(nCoeff, gt.shape[0]), gt)

	return gtcc


def gammatone_contrast(y, sr=44100, nfft=2048, hop_length=1024, nfilters=64, f_min=50, f_max=22050,\
					n_bands=6, quantile=0.02, log=True):
	'''Gammatone contrast.
	Parameters:
	-----------

	Return:w
	-----------
	contrast : np.ndarray [shape=(n_bands + 1, t)]
		each row of contrast values corresponds to a given ERB frequency band
	'''

	gt_weights, gt = gtgram.fft_gtgram(y, sr, nfft, nfilters, f_min, nfft, hop_length, nfft)

	valley = np.zeros((n_bands, gt.shape[1]))
	peak = np.zeros_like(valley)
	bounds = np.zeros(n_bands + 2)
	
	erb_centre_freqs = erb_filters.centre_freqs(f_min, f_max)
	
	n_filters = len(erb_centre_freqs)
	
	bins_per_band = np.rint(float(n_filters) / n_bands)
	subbands = np.array(map(lambda x: x*bins_per_band, np.arange(0, n_bands+1))).astype(int)
	subbands[-1] = n_filters - 1
	gc = np.zeros((n_bands, gt.shape[1]))
	
	for i in xrange(n_bands):
		gc[i, :] = np.max(gt[subbands[i]:subbands[i+1], :], axis=0) - np.min(gt[subbands[i]:subbands[i+1], :], axis=0)
	
	
	if log:
		return np.log(gc + 1e-5)
	else:
		return gc


def gammatone_polynomial(y, sr=44100, nfft=2048, hop_length=1024, nfilters=64, f_min=50, f_max=22050,\
						order=2):

	gt_weights, gt = gtgram.fft_gtgram(y, sr, nfft, nfilters, f_min, nfft, hop_length, nfft)
	freq = erb_filters.centre_freqs(f_min, f_max)

	polycoeff = np.polyfit(freq, gt, order)

	return polycoeff

	
def main():
	pass


if __name__ == '__main__':
	main()

