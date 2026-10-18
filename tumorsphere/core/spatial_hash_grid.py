"""Module that defines the SpatialHashGrid class."""

from collections import defaultdict
from collections.abc import Iterable, MutableSet
import warnings
from itertools import chain, product

import numpy as np

# from line_profiler import profile


class OrderedIndexSet(MutableSet):
    """Set-like collection with stable iteration order after pickle.

    Dictionary insertion order survives serialization. Membership, addition
    and removal retain expected average-case constant-time behavior.
    Iteration order is *insertion order*, unlike Python's builtin set; this
    numerical ordering change applies equally to cached and direct backends.
    """

    __slots__ = ("_members",)

    def __init__(self, iterable=()):
        self._members = dict.fromkeys(iterable)

    def __contains__(self, value):
        return value in self._members

    def __iter__(self):
        return iter(self._members)

    def __len__(self):
        return len(self._members)

    def add(self, value):
        self._members[value] = None

    def discard(self, value):
        self._members.pop(value, None)

    def remove(self, value):
        del self._members[value]

    def __repr__(self):
        return f"OrderedIndexSet({list(self._members)!r})"


class SpatialHashGrid:
    """Spatial hash for neighbor searches in two or three dimensions.

    Bucket coordinates are serialized as ``bytes`` keys. Each bucket
    stores cell indices in an ``OrderedIndexSet`` so iteration order is
    stable across checkpoints made with the current format. Transient
    candidate-neighbor caches are invalidated after bucket mutations
    and are not serialized.

    Attributes
    ----------
    hash_table : defaultdict[bytes, OrderedIndexSet]
        Spatial bucket membership, in insertion order.
    spatial_dimension : int
        Number of hashed dimensions: 2 (requires z=0) or 3.
    bounds : float or None
        Periodic box side when a bounded torus is used.
    cube_size : float
        Spatial bucket side length.
    torus : bool
        Whether bounded coordinates wrap periodically.
    offsets : np.ndarray
        Moore-neighborhood offsets for ``spatial_dimension`` dimensions.
    """

    def __init__(
        self,
        culture=None,  # type Culture, not declared to avoid circular imports
        bounds: float = None,
        cube_size: float = 2,
        torus: bool = True,
        spatial_dimension: int = 3,
    ) -> None:
        """Initialize the grid and its transient neighbor caches.

        Parameters
        ----------
        culture : Culture or None, optional
            Owning culture; generally set during ``Culture`` initialization.
        bounds : float or None, optional
            Side length of the bounded domain, or None for unbounded space.
        cube_size : float, optional
            Side length of a spatial bucket.
        torus : bool, optional
            Whether to wrap buckets across periodic boundaries.
        spatial_dimension : {2, 3}, optional
            Number of coordinates used for hashing. Two-dimensional grids
            require all stored positions to have z=0.
        """
        if (
            not isinstance(spatial_dimension, (int, np.integer))
            or isinstance(spatial_dimension, bool)
            or spatial_dimension not in (2, 3)
        ):
            raise ValueError("spatial_dimension must be 2 or 3.")

        self.spatial_dimension = int(spatial_dimension)

        self.culture = culture
        self.torus = torus
        self.bounds = bounds

        self.offsets = np.array(
            list(product(range(-1, 2), repeat=self.spatial_dimension))
        )

        self._offset_cache = {1: self.offsets}
        self.hash_table = defaultdict(OrderedIndexSet)
        self._neighbor_bucket_cache = {}
        self._batch_neighbor_cache = None
        # Both elongation caches are disposable, fixed-N accelerators.
        self._elongation_geometry_cache = None
        self._elongation_bucket_cache = None

        if not np.isfinite(cube_size) or cube_size <= 0:
            raise ValueError("cube_size must be finite and positive.")

        if self.torus and self.bounds is not None:
            if not np.isfinite(self.bounds) or self.bounds <= 0:
                raise ValueError(
                    "Periodic bounds must be finite and positive."
                )

            self.number_of_buckets = int(np.floor(self.bounds / cube_size))

            # Avoid repeated neighboring buckets after periodic wrapping
            if self.number_of_buckets < 3:
                raise ValueError(
                    "The periodic grid requires at least 3 buckets "
                    "per axis. Increase the box size or use a smaller "
                    "cube_size only if it still covers the full "
                    "interaction range."
                )

            # Fit an integer number of buckets into the periodic box
            self.cube_size = self.bounds / self.number_of_buckets

        else:
            self.cube_size = cube_size

    def __getstate__(self):
        """Exclude the neighbor cache from checkpoints."""
        state = self.__dict__.copy()
        state.pop("_neighbor_bucket_cache", None)
        state.pop("_batch_neighbor_cache", None)
        state.pop("_elongation_geometry_cache", None)
        state.pop("_elongation_bucket_cache", None)
        return state

    def __setstate__(self, state):
        """Restore the grid with an empty neighbor cache."""
        self.__dict__.update(state)

        # Checkpoints created before spatial_dimension existed
        # correspond to the original 3D grid.
        self.spatial_dimension = state.get("spatial_dimension", 3)

        # Existing checkpoints may contain defaultdict(set). The set iteration
        # order in those files was not guaranteed to match the live process.
        # Convert their *restored* order once, so future checkpoints are stable.
        # Exact continuation relative to a live pre-migration run is not
        # guaranteed for those legacy checkpoints.
        if (
            getattr(self.hash_table, "default_factory", None) is not OrderedIndexSet
            or any(
                not isinstance(bucket, OrderedIndexSet)
                for bucket in self.hash_table.values()
            )
        ):
            old_table = self.hash_table
            stable_table = defaultdict(OrderedIndexSet)
            for key, bucket in old_table.items():
                stable_table[key] = OrderedIndexSet(bucket)
            self.hash_table = stable_table
            warnings.warn(
                "Legacy SpatialHashGrid checkpoint migrated from builtin sets "
                "to ordered buckets. Its previous exact iteration order cannot "
                "be guaranteed; use new runs for bitwise checkpoint tests.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._neighbor_bucket_cache = {}
        # Candidate indices are rebuilt lazily after checkpoint restoration.
        self._batch_neighbor_cache = None
        # Both elongation caches are disposable, fixed-N accelerators.
        self._elongation_geometry_cache = None
        self._elongation_bucket_cache = None

    def get_bucket_position(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """Get the spatial bucket coordinates of a position."""

        if self.spatial_dimension == 2:
            if len(position) == 3 and position[2] != 0:
                raise ValueError(
                    "A 2D spatial grid requires z=0."
                )

            position = position[:2]

        return np.floor(position / self.cube_size).astype(int)

    def get_hash_key(self, position: np.ndarray) -> bytes:
        """Get the hash key of a position as a bytes object."""
        return self.get_bucket_position(position).tobytes()

    def add_cell_to_hash_table(
        self,
        cell_index: int,
        position: np.ndarray,
    ) -> None:
        """Add a cell to the hash table."""
        self.hash_table[self.get_hash_key(position)].add(cell_index)
        cache = getattr(self, "_batch_neighbor_cache", None)
        if cache is not None:
            cache.modified_bucket(self, position)
        elongation_cache = getattr(self, "_elongation_bucket_cache", None)
        if elongation_cache is not None:
            elongation_cache.clear()

    def remove_cell_from_hash_table(
        self,
        cell_index: int,
        position: np.ndarray,
    ) -> None:
        """Remove a cell from the hash table."""
        self.hash_table[self.get_hash_key(position)].remove(cell_index)
        cache = getattr(self, "_batch_neighbor_cache", None)
        if cache is not None:
            cache.modified_bucket(self, position)
        elongation_cache = getattr(self, "_elongation_bucket_cache", None)
        if elongation_cache is not None:
            elongation_cache.clear()

    def is_position_in_bounds(
        self,
        position: np.ndarray,
    ) -> bool:
        """Check if a position is in the bounds of the grid."""
        if self.bounds is None:
            return True
        return np.all(position >= 0) and np.all(position < self.bounds)

    def get_in_bounds_position(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Get the updated position within the bounds of the grid.

        Get the updated position within the bounds of the grid, given
        an old, out-of-bounds position.

        For a toroidal grid, positions wrap around the edges to the
        opposite side. For a bounded, non-toroidal grid, this method raises
        a NotImplementedError.

        Parameters
        ----------
        position : np.ndarray
            The original position of the cell.

        Returns
        -------
        np.ndarray
            The adjusted position within the bounds of the grid.

        Raises
        ------
        NotImplementedError
            If the grid is bounded and not toroidal.
        """
        if self.bounds is None:
            return position

        if self.torus:
            # Use modulus operation to wrap around for a toroidal grid.
            # This handles both positive and negative positions, ensuring
            # they wrap around correctly.
            return np.mod(position, self.bounds)
        else:
            # If the grid is not a torus and has defined bounds, raise a
            # NotImplementedError. This informs the user that behavior for
            # bounded, non-torus grids needs to be defined. I didn't
            # implement this because we could handle this case in a number
            # of ways: we could kill the cell, we could bounce it back,
            # we could reject the move/reproduction attempt, etc.
            raise NotImplementedError(
                "Non-torus grid boundary behavior is not yet implemented."
            )

    # @profile
    def find_neighbors(
        self,
        position: np.ndarray,
        bucket_radius: int = 1,
    ) -> Iterable:
        """Iterate over candidates in the neighboring spatial buckets.

        The returned cells are *candidates*, not necessarily interacting
        neighbors: overlap filtering is performed separately. Periodic
        wrapping deduplicates bucket coordinates. Iteration preserves the
        ordering of buckets and of indices within each bucket.

        Parameters
        ----------
        position : np.ndarray
            Position defining the central bucket.
        bucket_radius : int, optional
            Radius in bucket units. A value of one covers a 3x3 or 3x3x3
            Moore neighborhood in 2D or 3D, respectively.

        Returns
        -------
        Iterable[int]
            Candidate cell indices, including the reference cell if present.
        """
        if not isinstance(bucket_radius, (int, np.integer)) or bucket_radius < 0:
            raise ValueError("bucket_radius must be a non-negative integer.")

        bucket_radius = int(bucket_radius)

        if bucket_radius not in self._offset_cache:
            self._offset_cache[bucket_radius] = np.array(
                list(
                    product(
                        range(-bucket_radius, bucket_radius + 1),
                        repeat=self.spatial_dimension,
                    )
                ),
                dtype=int,
            )

        bucket = self.get_bucket_position(position)

        if self.bounds is not None and self.torus:
            # Cache key: central bucket and neighborhood radius.
            cache_key = (bucket.tobytes(), bucket_radius)

            neighbor_keys = self._neighbor_bucket_cache.get(cache_key)

            if neighbor_keys is None:
                # Compute neighboring buckets only on a cache miss.
                adj_buckets = np.mod(
                    bucket + self._offset_cache[bucket_radius],
                    self.number_of_buckets,
                )

                if self.number_of_buckets >= 2 * bucket_radius + 1:
                    # Preserve the lexicographic order of np.unique.
                    adj_buckets = adj_buckets[
                        np.lexsort(adj_buckets.T[::-1])
                    ]
                else:
                    # Remove duplicates when periodic wrapping requires it.
                    adj_buckets = np.unique(adj_buckets, axis=0)

                # Cache bucket keys, never their cell contents.
                neighbor_keys = tuple(
                    b.tobytes() for b in adj_buckets
                )

                self._neighbor_bucket_cache[cache_key] = neighbor_keys

            return chain.from_iterable(
                self.hash_table[key] for key in neighbor_keys
            )

        # Preserve the original behavior for non-periodic grids.
        adj_buckets = bucket + self._offset_cache[bucket_radius]

        return chain.from_iterable(
            self.hash_table[b.tobytes()] for b in adj_buckets
        )
