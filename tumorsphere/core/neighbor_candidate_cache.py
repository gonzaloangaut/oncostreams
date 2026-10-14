"""Transient, order-preserving candidate-neighbor cache for fixed-N batch dynamics.

Only cell *indices* are cached. The physics kernel always recomputes geometry,
relative positions, overlaps, forces, noise, and displacements each step.

A version is maintained for every possible center bucket. When a cell enters
or leaves a bucket, every center whose Moore neighborhood contains that bucket
is invalidated. This also catches changes to OrderedIndexSet insertion order.

The owning SpatialHashGrid excludes this cache from pickle/checkpoints.
"""
from __future__ import annotations


class OrderedNeighborCache:
    """Lazy ordered candidate lists invalidated by spatial-grid mutations."""

    def __init__(self):
        # index -> (center bucket key, version, ordered candidate tuple)
        self._entries = {}
        self._versions = {}
        self.hits = 0
        self.misses = 0

    def candidates(self, grid, index: int, position):
        """Return candidates in precisely the batch_numba reference order."""
        center = grid.get_hash_key(position)
        version = self._versions.get(center, 0)
        saved = self._entries.get(index)
        if saved is not None and saved[0] == center and saved[1] == version:
            self.hits += 1
            return saved[2]

        # Retain the original *two-part* ordering: lower indices first in
        # ascending order, then higher indices in grid insertion order.
        raw = [int(j) for j in grid.find_neighbors(position=position)
               if int(j) != index]
        earlier = sorted(j for j in raw if j < index)
        later = [j for j in raw if j > index]
        ordered = tuple(earlier + later)
        self._entries[index] = (center, version, ordered)
        self.misses += 1
        return ordered

    def modified_bucket(self, grid, position) -> None:
        """Invalidate centers affected by a successful bucket add/remove."""
        if not (grid.torus and grid.bounds is not None):
            raise RuntimeError('batch_cached_numba requires a periodic bounded grid')

        changed = grid.get_bucket_position(position)
        # grid.offsets contains all Moore-neighborhood offsets in 2D or 3D.
        # Deduplicate to remain correct even if offsets wrap onto one bucket.
        affected = {
            ((changed + offset) % grid.number_of_buckets).astype(int).tobytes()
            for offset in grid.offsets
        }
        for center in affected:
            self._versions[center] = self._versions.get(center, 0) + 1
