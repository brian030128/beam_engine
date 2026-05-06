"""Diverse Beam Search wrapper.

Vijayakumar et al., 2016 (arXiv:1610.02424). DBS only changes the top-K
selection rule; the attention computation is delegated to whichever
attention-method ``beam_search`` callable you wrap.

Usage
-----

::

    from beam_engine.baselines import dbs, paged, tree
    from beam_engine.methods import bs_kernel, adaptive_pool

    # Pre-built per-method wrappers, all with default num_groups=4, λ=0.5.
    dbs.dbs_paged(...)          # paged + DBS
    dbs.dbs_bs_kernel(...)      # bs_kernel + DBS
    dbs.dbs_adaptive_pool(...)  # cascade kernel + DBS
    dbs.dbs_tree(...)           # tree-attention + DBS
    dbs.dbs_fasttree(...)       # fasttree + DBS
    dbs.dbs_mlca(...)           # non-fused cascade + DBS

    # Factory if you want custom (num_groups, λ):
    custom = dbs.wrap(paged.beam_search, num_groups=8, diversity_strength=1.0)

The default exported ``beam_search`` is ``dbs_paged`` (DBS over the paged
kernel) for backward compat with the prior ``methods["dbs"]`` entry.
"""

from __future__ import annotations

from typing import Callable

from ..decoding import dbs_strategy
from . import fasttree, mlca, paged, tree


def wrap(
    method_fn: Callable,
    *,
    num_groups: int = 4,
    diversity_strength: float = 0.5,
) -> Callable:
    """Return a ``beam_search`` callable that runs ``method_fn`` with the
    Hamming-diversity DBS strategy plumbed into its top-K selectors.

    ``method_fn`` must accept ``select_at_prefill`` and ``select_at_decode``
    keyword arguments (every method in this repo does).
    """
    pf, df = dbs_strategy(num_groups=num_groups, diversity_strength=diversity_strength)

    def _bs(*args, **kwargs):
        kwargs.setdefault("select_at_prefill", pf)
        kwargs.setdefault("select_at_decode", df)
        return method_fn(*args, **kwargs)

    _bs.__name__ = f"dbs_{getattr(method_fn, '__module__', 'method').rsplit('.', 1)[-1]}"
    _bs.__doc__ = (
        f"Diverse Beam Search (G={num_groups}, λ={diversity_strength}) over "
        f"{getattr(method_fn, '__module__', 'method')}."
    )
    return _bs


# Pre-built per-method DBS variants with default knobs.
dbs_paged = wrap(paged.beam_search)
dbs_tree = wrap(tree.beam_search)
dbs_fasttree = wrap(fasttree.beam_search)
dbs_mlca = wrap(mlca.beam_search)


def _lazy_dbs_adaptive_pool(*args, **kwargs):
    from ..methods.adaptive_pool import beam_search as _ap_bs
    return wrap(_ap_bs)(*args, **kwargs)


def _lazy_dbs_bs_kernel(*args, **kwargs):
    from ..methods.bs_kernel import beam_search as _bsk_bs
    return wrap(_bsk_bs)(*args, **kwargs)


dbs_adaptive_pool = _lazy_dbs_adaptive_pool
dbs_bs_kernel = _lazy_dbs_bs_kernel


# Backward-compat default: dbs.beam_search == dbs_paged.
beam_search = dbs_paged
