"""NARX-related utilities."""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from functools import partial
from numbers import Integral

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import (
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils._array_api import (
    get_namespace_and_device,
    move_to,
    supported_float_dtypes,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted

from .._fastcan import FastCan
from .._lazyfastcan import LazyFastCan, _skip2valid
from .._refine import refine
from ..utils import mask_missing_values
from ._base import NARX, _prepare_poly_terms, _validate_session_sizes
from ._feature import (
    _make_a_time_shift_feature,
    make_poly_ids,
    make_time_shift_ids,
    tp2fd,
)


@validate_params(
    {
        "narx": [NARX],
        "term_space": [Interval(Integral, 1, None, closed="left")],
        "coef_space": [Interval(Integral, 1, None, closed="left")],
        "float_precision": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def print_narx(
    narx,
    term_space=20,
    coef_space=10,
    float_precision=3,
):
    """Print a NARX model as a Table which contains y idx, Term, and Coef.
    y idx is used to indicate which output (assuming it is a multi-output case)
    the term belongs to.

    Parameters
    ----------
    narx : NARX model
        The NARX model to be printed.

    term_space: int, default=20
        The space for the column of Term.

    coef_space: int, default=10
        The space for the column of Coef.

    float_precision: int, default=3
        The number of places after the decimal for Coef.

    Returns
    -------
    table : str
        The table of output index, terms, and coefficients of the NARX model.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from fastcan.narx import print_narx, NARX
    >>> X, y = load_diabetes(return_X_y=True)
    >>> print_narx(NARX().fit(X, y), term_space=10, coef_space=5, float_precision=0)
    | yid |   Term   |Coef |
    |-----|----------|-----|
    |  0  |Intercept | 152 |
    |  0  |  X[k,0]  | -10 |
    |  0  |  X[k,1]  |-240 |
    |  0  |  X[k,2]  | 520 |
    |  0  |  X[k,3]  | 324 |
    |  0  |  X[k,4]  |-792 |
    |  0  |  X[k,5]  | 477 |
    |  0  |  X[k,6]  | 101 |
    |  0  |  X[k,7]  | 177 |
    |  0  |  X[k,8]  | 751 |
    |  0  |  X[k,9]  | 68  |
    """
    check_is_fitted(narx)

    def _get_term_str(term_feat_ids, term_delay_ids):
        term_str = ""
        for _, (feat_id, delay_id) in enumerate(zip(term_feat_ids, term_delay_ids)):
            if -1 < feat_id < narx.n_features_in_:
                if delay_id == 0:
                    term_str += f"*X[k,{feat_id}]"
                else:
                    term_str += f"*X[k-{delay_id},{feat_id}]"
            elif feat_id >= narx.n_features_in_:
                term_str += f"*y_hat[k-{delay_id},{feat_id - narx.n_features_in_}]"
        return term_str[1:]

    yid_space = 5
    print(
        f"|{'yid':^{yid_space}}"
        + f"|{'Term':^{term_space}}"
        + f"|{'Coef':^{coef_space}}|"
    )
    print(f"|{'-' * yid_space}|{'-' * term_space}|{'-' * coef_space}|")
    for i in range(narx.n_outputs_):
        print(
            f"|{i:^{yid_space}}|"
            + f"{'Intercept':^{term_space}}|"
            + f"{narx.intercept_[i]:^{coef_space}.{float_precision}f}|"
        )
    for i, term_id in enumerate(zip(narx.feat_ids_, narx.delay_ids_)):
        print(
            f"|{narx.output_ids_[i]:^{yid_space}}|"
            + f"{_get_term_str(*term_id):^{term_space}}|"
            + f"{narx.coef_[i]:^{coef_space}.{float_precision}f}|"
        )


@validate_params(
    {
        "X": ["array-like", None],
        "y": ["array-like"],
        "n_terms_to_select": [
            Interval(Integral, 1, None, closed="left"),
            "array-like",
        ],
        "max_delay": [
            Interval(Integral, 0, None, closed="left"),
        ],
        "poly_degree": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "fit_intercept": ["boolean"],
        "session_sizes": [None, "array-like"],
        "max_candidates": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "include_zero_delay": [None, "array-like"],
        "static_indices": [None, "array-like"],
        "lazy": ["boolean"],
        "batch_size": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "shared_terms": ["boolean"],
        "refine_verbose": ["verbose"],
        "refine_drop": [
            None,
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"all"}),
        ],
        "refine_max_iter": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
    },
    prefer_skip_nested_validation=True,
)
def make_narx(
    X,
    y,
    n_terms_to_select,
    max_delay=1,
    poly_degree=1,
    *,
    fit_intercept=True,
    session_sizes=None,
    max_candidates=None,
    random_state=None,
    include_zero_delay=None,
    static_indices=None,
    lazy=False,
    batch_size=16,
    shared_terms=False,
    refine_verbose=1,
    refine_drop=None,
    refine_max_iter=None,
    **params,
):
    """Find `time_shift_ids`, `poly_ids`, `output_ids` for a NARX model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target vector or matrix.

    n_terms_to_select : int or array-like of shape (n_outputs,)
        The parameter is the absolute number of polynomial terms to select for
        each output. If `n_terms_to_select` is an integer, it is the
        same for all outputs.

    max_delay : int, default=1
        The maximum delay of time shift features.

    poly_degree : int, default=1
        The maximum degree of polynomial features.

    fit_intercept : bool, default=True
        Whether to fit the intercept. If set to False, intercept will be zeros.

    session_sizes : array-like of shape (n_sessions,), default=None
        The sizes of measurement sessions for time-series.
        The sum of session_sizes should be equal to n_samples.
        If None, the whole data is treated as one session.

        .. versionadded:: 0.5.0

    max_candidates : int, default=None
        Maximum number of candidate polynomial terms retained before selection.
        Randomly selected by reservoir sampling.
        If None, all candidates are considered.

    random_state : int or RandomState instance, default=None
        Used when `max_candidates` is not None to subsample candidate terms.
        See :term:`Glossary <random_state>` for details.

    include_zero_delay : {None, array-like} of shape (n_features,) default=None
        Whether to include the original (zero-delay) features.

    static_indices : {None, array-like} of shape (n_static_features,) default=None
        The indices of static features without time delay.

        .. note::
            If the corresponding include_zero_delay of the static features is False, the
            static feature will be excluded from candidate features.

    lazy : bool, default=False
        Whether to use LazyFastCan for selection. If False, FastCan will be used.

        .. versionadded:: 0.6.0

    batch_size : int, default=16
        The batch size of polynomial time shift terms generated at a time.
        Only available when `lazy` is True.

        .. versionadded:: 0.6.1

    shared_terms : bool, default=False
        Whether to select the polynomial terms jointly for all outputs, so
        that one term can be shared by multiple outputs. If True, terms are
        selected greedily to maximize the sum of the variance explained for
        each output, where every output is standardized to unit variance
        first, so that outputs with larger magnitude do not dominate the
        selection. Each selected term is booked to every output for which
        it is competitive, i.e. it explains a non-trivial amount of
        variance and is at least half as good as the best term still
        available for that output, and counts towards the
        `n_terms_to_select` budget of each of those outputs. As a
        result, the same term may appear in the models of several outputs
        (i.e. a shared term), while each output still gets exactly its own
        budget of terms. The selected terms, including the shared ones, are
        stored in the returned model and are exactly the features used at
        prediction time. Only available when `lazy` is False. When
        `shared_terms` is True, `refine_verbose`, `refine_drop`,
        `refine_max_iter` and `**params` are not used. If `y` has a single
        output, this parameter has no effect.

        .. versionadded:: 0.7.0

    refine_verbose : int, default=1
        The verbosity level of refine. Only available when `lazy` is False.

    refine_drop : int or "all", default=None
            The number of the selected features dropped for the consequent
            reselection. If `drop` is None, no refining will be performed.
            Only available when `lazy` is False.

    refine_max_iter : int, default=None
        The maximum number of valid iterations in the refining process.
        Only available when `lazy` is False.

    **params : dict
            Keyword arguments passed to
            `fastcan.FastCan`.
            Only available when `lazy` is False.

    Returns
    -------
    narx : NARX
        NARX instance.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from fastcan.narx import make_narx, print_narx
    >>> rng = np.random.default_rng(12345)
    >>> n_samples = 1000
    >>> max_delay = 3
    >>> e = rng.normal(0, 0.1, n_samples)
    >>> u0 = rng.uniform(0, 1, n_samples+max_delay) # input
    >>> u1 = rng.normal(0, 0.1, n_samples) # static input (i.e. no time shift)
    >>> y = np.zeros(n_samples+max_delay) # output
    >>> for i in range(max_delay, n_samples+max_delay):
    ...     y[i] = (0.5*y[i-1] + 0.3*u0[i]**2 + 2*u0[i-1]*u0[i-3] +
    ...             1.5*u0[i-2]*u1[i-max_delay] + 1)
    >>> y = y[max_delay:]+e
    >>> X = np.c_[u0[max_delay:], u1]
    >>> narx = make_narx(X=X,
    ...     y=y,
    ...     n_terms_to_select=4,
    ...     max_delay=3,
    ...     poly_degree=2,
    ...     static_indices=[1],
    ...     eta=True,
    ...     verbose=0,
    ...     refine_verbose=0,
    ...     refine_drop=1)
    >>> print(f"{mean_squared_error(y, narx.fit(X, y).predict(X)):.4f}")
    0.0289
    >>> print_narx(narx)
    | yid |        Term        |   Coef   |
    |-----|--------------------|----------|
    |  0  |     Intercept      |  1.050   |
    |  0  |    y_hat[k-1,0]    |  0.484   |
    |  0  |   X[k,0]*X[k,0]    |  0.306   |
    |  0  | X[k-1,0]*X[k-3,0]  |  2.000   |
    |  0  |  X[k-2,0]*X[k,1]   |  1.528   |

    For a multi-output system, e.g. a greenhouse where the temperature and
    the humidity are both driven by a common input signal, the terms can be
    selected jointly with `shared_terms=True`, so that one term can be
    shared by several outputs while each output still gets its own budget
    of terms:

    >>> rng = np.random.default_rng(12345)
    >>> n_samples = 1000
    >>> max_delay = 3
    >>> e0 = rng.normal(0, 0.1, n_samples)
    >>> e1 = rng.normal(0, 0.02, n_samples)
    >>> u = rng.uniform(0, 1, n_samples+max_delay) # common input signal
    >>> y0 = np.zeros(n_samples+max_delay) # temperature
    >>> y1 = np.zeros(n_samples+max_delay) # humidity
    >>> for i in range(max_delay, n_samples+max_delay):
    ...     y0[i] = 0.6*y0[i-1] + 8*u[i-1]**2 + 4*u[i-2] + 20
    ...     y1[i] = 0.5*y1[i-1] + 0.2*u[i-1]**2 + 0.1*u[i-3] + 0.5
    >>> y = np.c_[y0[max_delay:]+e0, y1[max_delay:]+e1]
    >>> X = u[max_delay:].reshape(-1, 1)
    >>> narx = make_narx(X, y, n_terms_to_select=[3, 3], max_delay=3,
    ...                  poly_degree=2, shared_terms=True).fit(X, y)
    >>> print_narx(narx)
    | yid |        Term        |   Coef   |
    |-----|--------------------|----------|
    |  0  |     Intercept      |  21.203  |
    |  1  |     Intercept      |  0.387   |
    |  0  |    y_hat[k-1,0]    |  0.580   |
    |  0  | X[k-1,0]*X[k-1,0]  |  8.019   |
    |  0  |X[k-2,0]*y_hat[k-3,0]|  0.067   |
    |  1  |    y_hat[k-1,0]    |  0.012   |
    |  1  | X[k-1,0]*X[k-1,0]  |  0.197   |
    |  1  |      X[k-3,0]      |  0.045   |
    """
    xp, _, device_ = get_namespace_and_device(y)
    y = check_array(
        y,
        dtype=supported_float_dtypes(xp, device_),
        ensure_2d=False,
        ensure_all_finite="allow-nan",
    )
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_samples, n_outputs = y.shape

    if X is None:
        xy_hstack = y
        n_features = 0
    else:
        X = move_to(X, xp=xp, device=device_)
        X = check_array(
            X,
            dtype=y.dtype,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
            ensure_min_features=0,
        )
        check_consistent_length(X, y)
        xy_hstack = xp.concat((X, y), axis=1)
        n_features = X.shape[1]

    if isinstance(n_terms_to_select, Integral):
        n_terms_to_select = np.full(n_outputs, n_terms_to_select, dtype=int)
    else:
        n_terms_to_select = column_or_1d(n_terms_to_select, dtype=int, warn=True)
        if len(n_terms_to_select) != n_outputs:
            raise ValueError(
                "The length of `n_terms_to_select` should be equal to "
                f"the number of outputs, {n_outputs}, but got "
                f"{len(n_terms_to_select)}."
            )
    if shared_terms and lazy:
        raise ValueError("`shared_terms` is not supported when `lazy` is True.")
    session_sizes_cumsum = _validate_session_sizes(session_sizes, n_samples)

    if include_zero_delay is None:
        _include_zero_delay = [True] * n_features + [False] * n_outputs
    else:
        _include_zero_delay = include_zero_delay + [False] * n_outputs

    time_shift_ids_all = make_time_shift_ids(
        n_features=xy_hstack.shape[1],
        max_delay=max_delay,
        include_zero_delay=_include_zero_delay,
    )

    if static_indices is None:
        static_indices = []

    time_shift_ids_all = np.delete(
        time_shift_ids_all,
        (
            np.isin(time_shift_ids_all[:, 0], static_indices)
            & (time_shift_ids_all[:, 1] > 0)
        ),
        0,
    )
    poly_ids_all = make_poly_ids(
        time_shift_ids_all.shape[0],
        poly_degree,
        max_poly=max_candidates,
        random_state=random_state,
    )

    if lazy:
        sample_mask = xp.ones(n_samples, dtype=xp.bool, device=device_)
        for start in session_sizes_cumsum[:-1]:
            sample_mask[start : start + max_delay] = False
        gen_terms = partial(
            _gen_poly_time_shift_terms,
            time_shift_ids=time_shift_ids_all,
            poly_ids=poly_ids_all,
            mode="constant",
            constant_values=0.0,
            batch_size=batch_size,
        )
        selected_poly_ids = []
        for i in range(n_outputs):
            csf = LazyFastCan(
                n_features_to_select=n_terms_to_select[i],
                feature_generator=gen_terms,
                sample_mask=sample_mask,
            ).fit(xy_hstack, y[:, i])
            selected_ids = csf.indices_
            selected_ids = xp.sort(selected_ids)
            selected_ids = move_to(selected_ids, xp=np, device="cpu")
            selected_poly_ids.append(poly_ids_all[selected_ids])
    else:
        poly_terms = _prepare_poly_terms(
            xy_hstack,
            time_shift_ids_all,
            poly_ids_all,
            session_sizes_cumsum,
            max_delay,
            xp=xp,
        )
        # Remove missing values
        poly_terms_masked, y_masked = mask_missing_values(poly_terms, y)
        poly_terms_masked = move_to(poly_terms_masked, xp=np, device="cpu")
        y_masked = move_to(y_masked, xp=np, device="cpu")

        if shared_terms and n_outputs > 1:
            selected_ids = _select_shared_terms(
                poly_terms_masked,
                y_masked,
                n_terms_to_select,
            )
            selected_poly_ids = [poly_ids_all[ids] for ids in selected_ids]
        else:
            selected_poly_ids = []
            for i in range(n_outputs):
                csf = FastCan(
                    n_terms_to_select[i],
                    **params,
                ).fit(poly_terms_masked, y_masked[:, i])
                if refine_drop is not None:
                    indices, _ = refine(
                        csf,
                        drop=refine_drop,
                        max_iter=refine_max_iter,
                        verbose=refine_verbose,
                    )
                    support = np.zeros(shape=poly_ids_all.shape[0], dtype=bool)
                    support[indices] = True
                else:
                    support = csf.get_support()
                selected_poly_ids.append(poly_ids_all[support])

    selected_poly_ids = np.vstack(selected_poly_ids)

    time_shift_ids = time_shift_ids_all[
        np.unique(selected_poly_ids[selected_poly_ids != -1]), :
    ]
    poly_ids = (
        rankdata(
            np.r_[[[-1] * poly_degree], selected_poly_ids],
            method="dense",
        ).reshape(-1, poly_degree)[1:]
        - 2
    )

    output_ids = [i for i in range(n_outputs) for _ in range(n_terms_to_select[i])]
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    return NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=fit_intercept,
    )


def _select_shared_terms(poly_terms, y, n_terms_to_select):
    """Select polynomial terms jointly for multiple outputs.

    Greedy forward selection where each selected term is booked to every
    output for which it is competitive, so one term can be shared by
    several outputs while counting towards the term budget of each of
    them.

    Parameters
    ----------
    poly_terms : ndarray of shape (n_samples, n_candidates)
        The candidate polynomial terms with missing values removed.

    y : ndarray of shape (n_samples, n_outputs)
        The targets with missing values removed.

    n_terms_to_select : ndarray of shape (n_outputs,)
        The number of terms to select for each output.

    Returns
    -------
    selected_ids : list of ndarray
        The indices of the selected candidate terms for each output, in the
        order they were booked. The i-th array has `n_terms_to_select`[i]
        elements.
    """
    n_samples, n_candidates = poly_terms.shape
    n_outputs = y.shape[1]
    budgets = np.asarray(n_terms_to_select, dtype=int).copy()
    for i, budget in enumerate(budgets):
        if budget > n_candidates:
            raise ValueError(
                f"`n_terms_to_select` for output {i} is {budget}, which is "
                f"larger than the number of candidate terms {n_candidates}."
            )

    # Standardize each output to unit variance, so that the joint gain of a
    # term is not dominated by the outputs with larger magnitude.
    y_centered = y - y.mean(axis=0)
    y_std = y_centered.std(axis=0)
    y_std[y_std == 0] = 1.0
    residuals = y_centered / y_std

    # Center and normalize the candidate terms. The gains are then measured
    # as the explained variance of the standardized outputs, while the
    # intercept is handled separately by the model.
    terms_centered = poly_terms - poly_terms.mean(axis=0)
    term_norms = np.linalg.norm(terms_centered, axis=0)
    term_norms[term_norms == 0] = 1.0
    terms_centered /= term_norms

    # A term is booked to an output only when it is competitive for that
    # output: it explains more variance than an irrelevant term would on
    # average, and it is at least half as good as the best term still
    # available for that output. As each output is standardized to unit
    # variance, its residual sum of squares is `n_samples`, and an
    # irrelevant term explains 1/`n_samples` of it, i.e. 1.0, on average.
    noise_floor = 1.0
    competitiveness = 0.5

    ortho_basis = [np.empty((n_samples, 0)) for _ in range(n_outputs)]
    booked = np.zeros((n_candidates, n_outputs), dtype=bool)
    selected_ids = [[] for _ in range(n_outputs)]

    def _book(term_id, output_id):
        """Book a term to an output and update its basis and residual."""
        basis = ortho_basis[output_id]
        direction = terms_centered[:, term_id]
        if basis.shape[1] > 0:
            direction = direction - basis @ (basis.T @ direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
            ortho_basis[output_id] = np.column_stack([basis, direction])
            residuals[:, output_id] -= direction * (direction @ residuals[:, output_id])
        booked[term_id, output_id] = True
        budgets[output_id] -= 1
        selected_ids[output_id].append(term_id)

    while np.any(budgets > 0):
        # Gains of the candidate terms for each output with remaining
        # budget, measured on the standardized residuals.
        gains = np.zeros((n_candidates, n_outputs))
        for i in range(n_outputs):
            if budgets[i] == 0:
                continue
            basis = ortho_basis[i]
            if basis.shape[1] == 0:
                ortho_terms = terms_centered
            else:
                ortho_terms = terms_centered - basis @ (basis.T @ terms_centered)
            ortho_norm2 = np.einsum("ij,ij->j", ortho_terms, ortho_terms)
            with np.errstate(divide="ignore", invalid="ignore"):
                gains[:, i] = np.where(
                    ortho_norm2 > 1e-12,
                    np.square(ortho_terms.T @ residuals[:, i]) / ortho_norm2,
                    0.0,
                )
        gains[booked] = 0.0
        term_id = np.argmax(gains.sum(axis=1))

        # Book the term to every output with remaining budget for which the
        # term is competitive (shared-term accounting).
        booked_any = False
        for i in range(n_outputs):
            if budgets[i] == 0 or booked[term_id, i]:
                continue
            if gains[term_id, i] > noise_floor and gains[term_id, i] >= (
                competitiveness * gains[:, i].max()
            ):
                _book(term_id, i)
                booked_any = True
        if not booked_any:
            # Guarantee progress: book the best remaining (term, output)
            # pair even when no gain beats the noise floor.
            remaining_gains = np.where(booked, -np.inf, gains)
            remaining_gains[:, budgets == 0] = -np.inf
            term_id, output_id = np.unravel_index(
                np.argmax(remaining_gains), remaining_gains.shape
            )
            _book(term_id, output_id)

    return [np.array(ids, dtype=int) for ids in selected_ids]


def _gen_poly_time_shift_terms(
    X, time_shift_ids, poly_ids, skip_indices=None, batch_size=16, **kwargs
):
    """Generate polynomial time shift terms."""
    xp, _, device_ = get_namespace_and_device(X)
    n_samples = X.shape[0]
    n_features = poly_ids.shape[0]
    valid_indices = _skip2valid(skip_indices, n_features, xp=xp, device=device_)

    n_valid = valid_indices.shape[0]
    for i in range(0, n_valid, batch_size):
        batch_idx = valid_indices[i : i + batch_size]
        batch_features = xp.ones(
            (n_samples, len(batch_idx)), dtype=X.dtype, device=device_
        )
        for j, feat_id in enumerate(batch_idx):
            for var_id in poly_ids[feat_id]:
                if var_id != -1:
                    time_shift_vars = _make_a_time_shift_feature(
                        X, time_shift_ids[var_id], **kwargs
                    )
                    batch_features[:, j] *= time_shift_vars
        yield batch_idx, batch_features
