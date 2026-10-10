# SPDX-License-Identifier: Apache-2.0
"""CPU packet-view regression harness; no vLLM/NPU initialization needed."""

import ast
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("c8_cache_lifecycle", ROOT / "vllm_ascend/worker/c8_cache.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
# Execute the real fill function, avoiding Ascend module import side effects.
TREE = ast.parse((ROOT / "vllm_ascend/attention/attention_c8_mxfp.py").read_text())
FUNC = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "fill_mxfp_v_scale_cache")
NAMESPACE = {"torch": torch}
exec(compile(ast.Module(body=[FUNC], type_ignores=[]), "attention_c8_mxfp.py", "exec"), NAMESPACE)
FILL = NAMESPACE["fill_mxfp_v_scale_cache"]


class C8Impl:
    pass


def packet():
    # Two 512-token kernel blocks per physical page; PA_NZ6D V scales
    # occupy a subsection of a packet containing other K/V/scale bytes.
    raw = torch.full((3 * 1408 + 7,), 99, dtype=torch.uint8)
    view = raw.as_strided((6, 1, 1, 8, 16, 2), (704, 256, 256, 32, 2, 1), 71)
    scale = torch.arange(110, 126, dtype=torch.uint8)
    layer = SimpleNamespace(impl=C8Impl(), v_cache_scale=scale)
    entries = MODULE.register_c8_scale_caches({"fa": (None, None, None, view)}, {"fa": layer}, 3, C8Impl, FILL)
    return raw, view, scale, entries


def test_fill_preserves_packet_other_sections_and_stride():
    raw, view, scale, entries = packet()
    assert entries[0][2] == 2
    assert view.storage_offset() == 71 and view.stride(0) == 704
    expected = scale.view(1, 1, 1, 16, 1).expand_as(view)
    assert torch.equal(view, expected)
    assert (raw[:71] == 99).all()
    assert (raw[327:775] == 99).all()


@pytest.mark.parametrize("has_meta", [True, False])
def test_reassigned_page_zero_restore_then_copy(has_meta):
    raw, view, scale, entries = packet()
    raw[2 * 704 : 4 * 704 + 71].zero_()  # previous hybrid state owner
    before = view.clone()
    events = []

    class Zeroer:
        _meta = object() if has_meta else None

        def zero_block_ids(self, ids):
            events.append("zero")
            if self._meta is not None:
                for block in ids:
                    view[block * 2 : block * 2 + 2].zero_()

    def fill(source, target):
        events.append("restore")
        FILL(source, target)

    zeroer = MODULE.C8KVBlockZeroer(Zeroer(), entries, fill)
    zeroer.zero_block_ids([1])
    assert torch.equal(view[2:4], scale.view(1, 1, 1, 16, 1).expand_as(view[2:4]))
    assert torch.equal(view[:2], before[:2]) and torch.equal(view[4:], before[4:])
    # Model upstream's following prefix copy: copied source constants must win.
    events.append("copy")
    view[2:4].fill_(133)
    assert events == ["zero", "restore", "copy"]
    assert (view[2:4] == 133).all()


@pytest.mark.parametrize("cache_state", ["uninitialized", "non_c8", "c8", "distinct_entries"])
def test_shutdown_releases_c8_views_before_parent_cleanup(cache_state):
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "shutdown")
    method.body = [node for node in method.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    references = []
    events = []

    class Parent:
        def shutdown(self):
            # Model the parent's remaining cache owners, then its collection
            # and allocator cleanup. The runner and wrapper stay alive.
            self.kv_caches.clear()
            assert all(reference() is None for reference in references)
            events.append("parent_cleanup")

    child = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {"Parent": Parent, "C8KVBlockZeroer": MODULE.C8KVBlockZeroer}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[child], type_ignores=[])), "model_runner.py", "exec"),
        namespace,
    )
    state = namespace["Runner"]()
    state.kv_caches = {}
    delegate = object()
    if cache_state != "uninitialized":
        state.kv_block_zeroer = delegate
    if cache_state in ("c8", "distinct_entries"):
        raw, view, scale, entries = packet()
        storage = raw.untyped_storage()
        references.extend((weakref.ref(view), weakref.ref(scale), weakref.ref(storage)))
        state.kv_caches["fa"] = (raw, view)
        state._c8_mxfp_v_scale_caches = entries
        wrapper_entries = list(entries) if cache_state == "distinct_entries" else entries
        wrapper = MODULE.C8KVBlockZeroer(delegate, wrapper_entries, FILL)
        state.kv_block_zeroer = wrapper
        del raw, view, scale, storage
    state.shutdown()
    assert events == ["parent_cleanup"]
    if cache_state != "uninitialized":
        assert state.kv_block_zeroer is delegate
    if cache_state in ("c8", "distinct_entries"):
        assert entries == [] and wrapper.entries == []
    state.shutdown()
    assert events == ["parent_cleanup", "parent_cleanup"]


def test_missing_zeroer_still_restores_and_empty_ids_noop():
    _, view, scale, entries = packet()
    view[4:6].zero_()
    zeroer = MODULE.C8KVBlockZeroer(None, entries, FILL)
    zeroer.zero_block_ids([])
    assert not view[4:6].any()
    zeroer.zero_block_ids([2])
    assert torch.equal(view[4:6], scale.view(1, 1, 1, 16, 1).expand_as(view[4:6]))
    with pytest.raises(IndexError):
        zeroer.zero_block_ids([3])


def test_non_c8_unchanged_and_invalid_page_ratio_rejected():
    raw, view, scale, _ = packet()
    before = raw.clone()
    assert MODULE.register_c8_scale_caches({"bf16": view}, {}, 3, C8Impl, FILL) == []
    assert torch.equal(raw, before)
    with pytest.raises(ValueError):
        MODULE.register_c8_scale_caches(
            {"fa": (None, None, None, view)},
            {"fa": SimpleNamespace(impl=C8Impl(), v_cache_scale=scale)},
            4,
            C8Impl,
            FILL,
        )


@pytest.mark.parametrize("block_ids", [[-1], [3], [0, 3]])
@pytest.mark.parametrize("has_delegate", [True, False])
def test_invalid_ids_rejected_before_any_write(block_ids, has_delegate):
    raw, _, _, entries = packet()
    before = raw.clone()
    events = []

    class Zeroer:
        def zero_block_ids(self, ids):
            events.append("zero")
            raw.zero_()

    def fill(source, target):
        events.append("restore")
        FILL(source, target)

    zeroer = MODULE.C8KVBlockZeroer(Zeroer() if has_delegate else None, entries, fill)
    with pytest.raises(IndexError):
        zeroer.zero_block_ids(block_ids)
    assert events == []
    assert torch.equal(raw, before)


def test_all_cache_entries_validated_before_any_write():
    raw, view, scale, entries = packet()
    before = raw.clone()
    events = []
    entries.append((scale, view[:2], 2))

    class Zeroer:
        def zero_block_ids(self, ids):
            events.append("zero")

    def fill(source, target):
        events.append("restore")
        FILL(source, target)

    with pytest.raises(IndexError):
        MODULE.C8KVBlockZeroer(Zeroer(), entries, fill).zero_block_ids([0, 1])
    assert events == []
    assert torch.equal(raw, before)


def test_worker_zero_meta_reinitialization_keeps_restore_wrapper():
    _, view, scale, entries = packet()
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_init_kv_zero_meta"
    )
    method.body = [node for node in method.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    class Zeroer:
        def __init__(self, device, pin):
            self._meta = None

        def init_meta(self, *args):
            self.args = args

        def zero_block_ids(self, ids):
            pass

    namespace = {
        "AscendKVBlockZeroer": Zeroer,
        "C8KVBlockZeroer": MODULE.C8KVBlockZeroer,
        "fill_mxfp_v_scale_cache": FILL,
        "is_pin_memory_available": lambda: False,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "model_runner.py", "exec"), namespace)
    state = SimpleNamespace(
        device="cpu",
        attn_groups=[],
        kernel_block_sizes=[512],
        cache_config=SimpleNamespace(cache_dtype="mxfp8"),
        compilation_config=SimpleNamespace(static_forward_context={}),
        _c8_mxfp_v_scale_caches=entries,
        kv_block_zeroer=MODULE.C8KVBlockZeroer(None, entries, FILL),
    )
    namespace["_init_kv_zero_meta"](state)
    assert isinstance(state.kv_block_zeroer, MODULE.C8KVBlockZeroer)
    assert isinstance(state.kv_block_zeroer.delegate, Zeroer)
    view[2:4].zero_()
    state.kv_block_zeroer.zero_block_ids([1])
    assert torch.equal(view[2:4], scale.view(1, 1, 1, 16, 1).expand_as(view[2:4]))
