from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple


RANDOM_SEED = 2025


def build_pairs_rr(reals: List[Path], k_pairs: int = 20) -> List[Tuple[Path, Path]]:
    rng = random.Random(RANDOM_SEED)
    pairs: List[Tuple[Path, Path]] = []
    pool = reals[:]
    n = len(pool)
    if n < 2:
        return pairs
    for _ in range(k_pairs):
        a, b = rng.sample(pool, 2)
        pairs.append((a, b))
    return pairs


def build_pairs_cr(clones: List[Path], reals: List[Path], per_clone: int = 3) -> List[Tuple[Path, Path]]:
    rng = random.Random(RANDOM_SEED)
    pairs: List[Tuple[Path, Path]] = []
    for c in clones:
        m = min(per_clone, len(reals))
        targets = rng.sample(reals, m)
        pairs.extend((c, r) for r in targets)
    return pairs


def build_pairs_ci(clones: List[Path], impostors: List[Path], per_clone: int = 10) -> List[Tuple[Path, Path]]:
    rng = random.Random(RANDOM_SEED)
    pairs: List[Tuple[Path, Path]] = []
    for c in clones:
        m = min(per_clone, len(impostors))
        targets = rng.sample(impostors, m)
        pairs.extend((c, r) for r in targets)
    return pairs


