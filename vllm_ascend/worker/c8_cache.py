# SPDX-License-Identifier: Apache-2.0
"""Static C8 constants that must survive hybrid physical-page reassignment."""

from collections.abc import Callable
from typing import Any


class C8KVBlockZeroer:
    """Delegate zeroing, then restore constants before upstream block copies.

    Restoration intentionally runs even when the delegate has no zeroing
    metadata: hybrid state writes can still have overwritten the packet.
    Tensor views retain their original storage offset and page stride.
    """

    def __init__(self, delegate: Any, entries: list[tuple[Any, Any, int]], fill: Callable) -> None:
        self.delegate = delegate
        self.entries = entries
        self.fill = fill

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        # Reject the entire batch before any device write.
        for _, cache, ratio in self.entries:
            for block_id in block_ids:
                if block_id < 0 or block_id >= cache.shape[0] // ratio:
                    raise IndexError(f"C8 physical block ID out of range: {block_id}")
        if self.delegate is not None:
            self.delegate.zero_block_ids(block_ids)
        for value_scale, cache, ratio in self.entries:
            for block_id in block_ids:
                start = block_id * ratio
                self.fill(value_scale, cache[start : start + ratio])


def register_c8_scale_caches(
    caches: dict[str, Any], layers: dict[str, Any], num_blocks: int, backend_impl: type, fill: Callable
) -> list[tuple[Any, Any, int]]:
    """Fill only C8 V-scale views and retain the physical-to-kernel page ratio."""
    entries = []
    for name, cache in caches.items():
        layer = layers.get(name)
        if not isinstance(getattr(layer, "impl", None), backend_impl):
            continue
        value_cache = cache[3]
        if num_blocks <= 0 or value_cache.shape[0] % num_blocks:
            raise ValueError("C8 V-scale cache must contain whole physical pages")
        ratio = value_cache.shape[0] // num_blocks
        if ratio <= 0:
            raise ValueError("C8 physical pages must contain at least one kernel block")
        entries.append((layer.v_cache_scale, value_cache, ratio))
        fill(layer.v_cache_scale, value_cache)
    return entries
