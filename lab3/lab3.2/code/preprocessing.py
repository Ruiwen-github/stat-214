
import os
import sys
import numpy as np
import json
import joblib
import logging
from os.path import join, dirname
import pickle

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def lanczosfun(cutoff, x, window=3):
    """Helper function: windowed sinc filter for interpolation."""
    x = np.array(x) * cutoff  # scale time difference
    sinc = np.sinc(x)
    lanczos_window = np.sinc(x / window)
    return sinc * lanczos_window

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """
    Interpolates a 2D matrix over time using windowed sinc interpolation (Lanczos).

    Args:
        data: np.ndarray of shape [num_tokens, hidden_size]
        oldtime: list or np.array of shape [num_tokens] — timestamps for each token
        newtime: list or np.array of shape [num_TRs] — desired FMRI-aligned timepoints
        window: number of lobes to use in Lanczos filter (default 3)
        cutoff_mult: multiplier for cutoff frequency
        rectify: if True, interpolates positive and negative parts separately (rare)

    Returns:
        newdata: np.ndarray of shape [num_TRs, hidden_size]
    """
    data = np.array(data)
    oldtime = np.array(oldtime)
    newtime = np.array(newtime)

    # Determine the interpolation cutoff frequency based on TR resolution
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult

    # Build sinc kernel matrix [num_TRs, num_tokens]
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for i, t in enumerate(newtime):
        sincmat[i, :] = lanczosfun(cutoff, t - oldtime, window)

    # Apply filter to data
    if rectify:
        newdata = np.hstack([
            sincmat @ np.clip(data, -np.inf, 0),
            sincmat @ np.clip(data, 0, np.inf)
        ])
    else:
        newdata = sincmat @ data

    return newdata

def downsample_embeddings(stories, vectors, wordseqs):
    """
    Applies Lanczos interpolation to align token-level embeddings with FMRI TRs.

    Args:
        stories: List of story IDs (strings or ints)
        vectors: Dictionary {story: np.ndarray of shape (seq_len, hidden_size)}
        wordseqs: Dictionary-like object where wordseqs[story] has:
            - data_times: time of each token
            - tr_times: FMRI measurement times

    Returns:
        Dictionary {story: np.ndarray of shape (num_TRs, hidden_size)}
    """
    downsampled = {}

    for story in stories:
        data = vectors[story]  # shape (32, 256)
    
        # Ensure you have a matching data_times
        if len(wordseqs[story].data_times) != data.shape[0]:
            story_duration = wordseqs[story].tr_times[-1]
            wordseqs[story].data_times = np.linspace(0, story_duration, data.shape[0])

        # Now downsample
        old_times = wordseqs[story].data_times
        new_times = wordseqs[story].tr_times
        downsampled[story] = lanczosinterp2D(data, old_times, new_times, window=3)


        # Interpolate token-level data to TR-level alignment
        downsampled[story] = lanczosinterp2D(data, old_times, new_times, window=3)

    return downsampled



if __name__ == "__main__":
    raw_stories = pickle.load(open('raw_stories.pkl', 'rb'))
    