import numpy as np
import torch

import bandmat as bm
import bandmat.linalg as bla


def _build_win_mats(windows, num_frames):
    r"""For MLPG. Builds a window matrix of a given size for each window in a collection.

    Parameters
    ----------
    windows : list[window]
        Specifies the collection of windows as a sequence of `window` tuples with the structure `(l, u, win_coeff)`,
        where `l` and `u` are non-negative integers specifying the left and right extents of the window and
        `win_coeff` is an array specifying the window coefficients.
    num_frames : int
        Number of frames in the signal.

    Returns
    -------
    list[BandMat]
        Window matrices, one for each of the windows specified in `windows`. Each window matrix is a `frames` by
        `frames` Toeplitz matrix with lower bandwidth `l` and upper bandwidth `u`. The non-zero coefficients in each
        row of this Toeplitz matrix are given by `win_coeff`. The window matrices are stored as BandMats, i.e. using
        a banded representation.
    """
    win_mats = []
    for l, u, win_coeff in windows:
        assert l >= 0 and u >= 0
        assert len(win_coeff) == l + u + 1
        win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), num_frames)
        win_mat = bm.band_c_bm(u, l, win_coeffs).T
        win_mats.append(win_mat)

    return win_mats


def _build_poe(b_frames, tau_frames, win_mats, sdw=None):
    r"""For MLPG. Computes natural parameters for a Gaussian product-of-experts model.

    The natural parameters (b-value vector and precision matrix) are returned.
    The returned precision matrix is stored as a BandMat.

    Mathematically the b-value vector is given as:
        b = \sum_d \transpose{W_d} \tilde{b}_d

    and the precision matrix is given as:
        P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d

    where :math:`W_d` is the window matrix for window `d` as specified by an element of `win_mats`,
    :math:`\tilde{b}_d` is the sequence over time of b-value parameters for window `d` as given by a column of
    `b_frames`, and :math:`\tilde{\tau}_d` is the sequence over time of precision parameters for window `d` as given
    by a column of `tau_frames`.
    """
    if sdw is None:
        sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])
    num_windows = len(win_mats)
    frames = len(b_frames)
    assert np.shape(b_frames) == (frames, num_windows)
    assert np.shape(tau_frames) == (frames, num_windows)
    assert all([win_mat.l + win_mat.u <= sdw for win_mat in win_mats])

    b = np.zeros((frames,))
    prec = bm.zeros(sdw, sdw, frames)

    # Ensure inputs are of the correct type for bandmat.
    b_frames = b_frames.astype(np.float64)
    tau_frames = tau_frames.astype(np.float64)

    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                              diag=tau_frames[:, win_index])

    return b, prec


def MLPG(means, variances, windows=None, padding_size=100, seq_len=None):
    r"""Performs maximum-likelihood parameter generation.

    Parameters
    ----------
    means : np.ndarray, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
        Array of means for a single, or a batch of sequences.
    variances : np.ndarray, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim) or (feat_dim)
        Array of variances for a single, or a batch of sequences.
    windows : list[tuple[int, int, np.ndarray]]
        Windows describing the static/delta features included in the feature dimension of `means` and `variances`.
    padding_size : int
        Padding on either side of signal, used to handle smoothing at the boundaries.
    seq_len : array_like, shape (batch_size)
        Sequence lengths, necessary when a batch of sequences is given, as out-of-sequence frames will be all zeros.

    Returns
    -------
    most_probable_trajectory : np.ndarray, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
        The most probable trajectory, calculated by maximum-likelihood parameter generation.
    """
    # If inputs are torch.Tensor then convert to numpy.ndarry and convert back at the end of this function.
    device = None
    if isinstance(means, torch.Tensor):
        device = means.device
        means = means.detach().cpu().numpy()
    if isinstance(variances, torch.Tensor):
        if device is None:
            device = variances.device
        variances = variances.detach().cpu().numpy()
    if isinstance(seq_len, torch.Tensor):
        if device is None:
            device = seq_len.device
        seq_len = seq_len.detach().cpu().numpy()

    def _pad(sequence_feature, n=3):
        return np.concatenate(
            (np.repeat(sequence_feature[[0], :], n, axis=0),
             sequence_feature,
             np.repeat(sequence_feature[[-1], :], n, axis=0)),
            axis=0
        )

    if windows is None:
        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]

    if means.ndim == 2:  # Single sequence.
        means = means[np.newaxis, ...]
        using_batches = False
    else:  # Batch of sequences.
        using_batches = True

    batch_size = means.shape[0]
    num_frames = means.shape[1]
    num_windows = len(windows)
    feat_dim = means.shape[-1] // num_windows

    if seq_len is None:
        seq_len = [num_frames] * batch_size

    if variances.ndim == 2:  # Single sequence.
        variances = variances[None, ...]
    elif variances.ndim == 1:  # Global variance.
        one_batch_variances = np.repeat(variances[None, :], num_frames, axis=0)
        variances = np.repeat(one_batch_variances[None, :, :], batch_size, axis=0)

    # Index array that can be used to select feature dimension and its corresponding deltas.
    idx_base = np.arange(num_windows) * feat_dim

    most_probable_trajectory = np.zeros((batch_size, num_frames, feat_dim))

    for i in range(batch_size):
        # Crop using the sequence length, and add padding to act as a burn in.
        means_i = _pad(means[i, :seq_len[i]], n=padding_size)
        variances_i = _pad(variances[i, :seq_len[i]], n=padding_size)
        win_mats = _build_win_mats(windows, seq_len[i] + 2 * padding_size)

        for d in range(feat_dim):
            feat_mean = means_i[:, idx_base + d]
            feat_variance = variances_i[:, idx_base + d]

            feat_b = feat_mean / feat_variance
            feat_tau = 1.0 / feat_variance

            b, prec = _build_poe(feat_b, feat_tau, win_mats)
            feat_trajectory = bla.solveh(prec, b)

            most_probable_trajectory[i, :seq_len[i], d] = \
                feat_trajectory[padding_size:len(feat_trajectory)-padding_size]

    if not using_batches:
        most_probable_trajectory = most_probable_trajectory.squeeze(axis=0)

    # If the input had type torch.Tensor, then convert the output to the same type.
    if device is not None:
        most_probable_trajectory = torch.tensor(most_probable_trajectory).type(torch.float).to(device)

    return most_probable_trajectory

