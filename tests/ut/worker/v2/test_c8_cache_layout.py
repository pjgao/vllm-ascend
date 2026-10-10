# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the real MRV2 allocation functions and MXFP layout helpers.

AST loading avoids importing the NPU backend/plugin in a CPU-only process.
External vLLM configuration/spec types are small fixtures; function bodies,
packet offsets, dtype views, and allocation operations come from source.
This harness checks storage contracts, not NPU operators or upstream ABI.
"""

import ast
import math
import sys
import types
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class AttentionSpec:
    pass


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    block_size: int = 512
    num_kv_heads: int = 2
    head_size: int = 64
    head_size_v: int = 128
    dtype: object = torch.bfloat16
    state_content_bytes: int | None = None
    page_size_padded: int | None = None
    tokens_per_state: int = 1

    @property
    def num_heads(self):
        return self.num_kv_heads

    @property
    def unpadded_page_size_bytes(self):
        content = self.state_content_bytes
        if content is None:
            content = (self.head_size + self.head_size_v) * torch.empty((), dtype=self.dtype).element_size()
        return self.block_size * self.num_kv_heads * content

    @property
    def real_page_size_bytes(self):
        return self.unpadded_page_size_bytes

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.unpadded_page_size_bytes


@dataclass(frozen=True)
class MambaSpec:
    page_size_bytes: int


class UnsupportedSpec:
    pass


def load_functions(path, names, namespace, constants=False):
    tree = ast.parse(path.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in names
            or constants
            and isinstance(node, ast.Assign)
            and all(isinstance(target, ast.Name) and target.id.startswith("MXFP") for target in node.targets)
        ):
            body.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)


class TestC8CacheLayout(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        self.config = SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="mxfp8"),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
            model_config=SimpleNamespace(),
            kv_transfer_config=None,
        )
        self.layers = {}
        self.ns = dict(
            torch=torch,
            math=math,
            replace=replace,
            FullAttentionSpec=FullAttentionSpec,
            AttentionSpec=AttentionSpec,
            MambaSpec=MambaSpec,
            SlidingWindowSpec=UnsupportedSpec,
            UniformTypeKVCacheSpecs=UnsupportedSpec,
            AttentionLayerBase=UnsupportedSpec,
            MLAAttention=UnsupportedSpec,
            AscendSFAIndexerCacheSpec=UnsupportedSpec,
            AscendMLAAttentionSpec=UnsupportedSpec,
            AscendIndexerKPoolTailSpec=UnsupportedSpec,
            KVPPConfig=SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)),
            TURBOQUANT_CACHE_DTYPE="turboquant",
            get_current_vllm_config=lambda: self.config,
            get_layers_from_vllm_config=lambda *_: self.layers,
            is_c8_mxfp_kv_quant=lambda config: config.cache_config.cache_dtype == "mxfp8",
            kv_cache_dtype_str_to_dtype=lambda *_: torch.float8_e4m3fn,
            enable_sfa_dcp_replicated_indexer=lambda _: False,
            enable_sfa=lambda _: False,
            _is_dsv4_model=lambda _: False,
            is_deepseek_v41_cache=lambda _: False,
            is_hidden_state_cache_spec=lambda _: False,
            get_storage_block_size=lambda spec: spec.block_size,
            get_layerwise_reuse_config=lambda _: None,
            get_kv_cache_tensor_layers=lambda descriptor: descriptor.layers,
            logger=SimpleNamespace(debug=lambda *args: None),
        )
        layout = root / "vllm_ascend/attention/attention_c8_mxfp.py"
        names = {
            "validate_mxfp_k_scale_head_dim",
            "validate_mxfp_v_scale_block_size",
            "mxfp_kv_scale_groups",
            "mxfp_kv_block_scale_groups",
            "mxfp_k_scale_page_bytes",
            "mxfp_v_scale_page_bytes",
            "mxfp_k_scale_cache_shape",
            "mxfp_v_scale_cache_shape",
            "mxfp_packet_section_sizes",
            "mxfp_cache_spec",
            "mxfp_paged_cache_views",
            "mxfp_cache_views_for_spec",
        }
        load_functions(layout, names, self.ns, constants=True)
        load_functions(
            root / "vllm_ascend/worker/v2/attn_utils.py",
            {
                "get_kv_cache_spec",
                "_align_hybrid_attention_page_sizes",
                "_get_layer_kv_cache_specs",
                "_allocate_int8_cache_tensor",
                "_allocate_kv_cache",
                "_reshape_kv_cache_v2",
            },
            self.ns,
        )

    def configuration(self, spec, hybrid=False):
        blocks = 3
        page = spec.page_size_bytes
        groups = [SimpleNamespace(layer_names=["fa"], kv_cache_spec=spec)]
        descriptors = [
            SimpleNamespace(layers=["fa"], size=blocks * page, offset=0, layer_stride=blocks * page, block_stride=page)
        ]
        if hybrid:
            groups.append(SimpleNamespace(layer_names=["ssm"], kv_cache_spec=MambaSpec(page)))
            descriptors.append(
                SimpleNamespace(
                    layers=["ssm"], size=blocks * page, offset=0, layer_stride=blocks * page, block_stride=page
                )
            )
        return SimpleNamespace(num_blocks=blocks, kv_cache_groups=groups, kv_cache_tensors=descriptors)

    def reshape(self, spec, raw, alias=False):
        config = self.configuration(spec)
        group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["fa"], backend=object())
        return self.ns["_reshape_kv_cache_v2"](
            [group], {"fa": raw}, "mxfp8", [512], {"alias": "fa"} if alias else {}, config
        )

    def test_spec_budgets_four_byte_sections_without_changing_head_dims(self):
        spec = FullAttentionSpec()
        self.layers = {"fa": SimpleNamespace(get_kv_cache_spec=lambda _: spec)}
        module = types.ModuleType("vllm.model_executor.models.deepseek_v2")
        module.DeepseekV32IndexerCache = UnsupportedSpec
        with patch.dict(sys.modules, {module.__name__: module}):
            result = self.ns["get_kv_cache_spec"](self.config)["fa"]
        self.assertEqual((result.head_size, result.head_size_v), (64, 128))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)
        self.assertEqual(result.page_size_bytes, 512 * 2 * (64 + 128 + (64 + 128) // 32))
        self.config.cache_config.cache_dtype = "auto"
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertIs(self.ns["get_kv_cache_spec"](self.config)["fa"], spec)

    def test_standalone_allocation_is_one_packet_buffer(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec())
        raw = self.ns["_allocate_kv_cache"](self.configuration(spec), {}, torch.device("cpu"))["fa"]
        self.assertIsInstance(raw, torch.Tensor)
        self.assertEqual(raw.dtype, torch.int8)
        self.assertEqual(raw.numel(), 3 * spec.page_size_bytes)
        self.assertEqual(len(self.reshape(spec, raw)["fa"]), 4)

    def test_hybrid_allocation_overlays_one_backing(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec())
        raw = self.ns["_allocate_kv_cache"](self.configuration(spec, hybrid=True), {}, torch.device("cpu"))
        self.assertEqual(raw["fa"].untyped_storage().data_ptr(), raw["ssm"].untyped_storage().data_ptr())
        self.assertEqual(raw["fa"].untyped_storage().nbytes(), 3 * spec.page_size_bytes)
        self.assertEqual(len(self.reshape(spec, raw["fa"])["fa"]), 4)

    def test_shared_backing_preserves_nonzero_layer_offset(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec())
        config = self.configuration(spec, hybrid=True)
        layer_size = config.num_blocks * spec.page_size_bytes
        config.kv_cache_groups[0].layer_names.append("fa_second")
        config.kv_cache_tensors[0].layers.append("fa_second")
        for descriptor in config.kv_cache_tensors:
            descriptor.size = 2 * layer_size
        raw = self.ns["_allocate_kv_cache"](config, {}, torch.device("cpu"))
        self.assertEqual(raw["fa_second"].storage_offset(), layer_size)
        self.assertEqual(raw["fa"].untyped_storage().data_ptr(), raw["fa_second"].untyped_storage().data_ptr())
        group = SimpleNamespace(
            kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["fa", "fa_second"], backend=object()
        )
        caches = self.ns["_reshape_kv_cache_v2"]([group], raw, "mxfp8", [512], {}, config)
        self.assertEqual(caches["fa_second"][0].storage_offset(), layer_size)
        caches["fa_second"][3].fill_(37)
        self.assertTrue(torch.all(caches["fa"][3] == 0))

    def test_padded_virtual_blocks_and_absolute_storage_offset(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec(block_size=1024))
        spec = replace(spec, page_size_padded=spec.page_size_bytes + 8192)
        offset = 128
        storage = torch.full((3 * spec.page_size_bytes + offset + 128,), 7, dtype=torch.int8)
        raw = storage[offset:-128]
        caches = self.reshape(spec, raw, alias=True)
        k, v, ks, vs = caches["fa"]
        pitch = spec.page_size_bytes // 2
        sections = self.ns["mxfp_packet_section_sizes"](2, 512, 64, 128)
        self.assertIs(caches["alias"], caches["fa"])
        self.assertEqual([tensor.shape[0] for tensor in caches["fa"]], [6] * 4)
        self.assertEqual([tensor.stride(0) for tensor in caches["fa"]], [pitch] * 4)
        self.assertEqual(
            [tensor.storage_offset() for tensor in (k, ks, v, vs)],
            [offset, offset + sections[0], offset + sum(sections[:2]), offset + sum(sections[:3])],
        )
        self.assertTrue(all(not tensor.is_contiguous() for tensor in caches["fa"]))
        vs[2].fill_(93)
        self.assertTrue(torch.all(vs[2] == 93))
        self.assertTrue(torch.all(ks == 7))
        self.assertTrue(torch.all(storage[:offset] == 7))
        self.assertTrue(torch.all(storage[-128:] == 7))

    def test_split_raw_cache_is_rejected(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec())
        with self.assertRaisesRegex(ValueError, "one combined raw allocation"):
            self.reshape(spec, (torch.zeros(4), torch.zeros(4)))

    def test_incomplete_physical_pages_are_rejected(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec())
        with self.assertRaisesRegex(ValueError, "whole pages"):
            self.reshape(spec, torch.zeros(spec.page_size_bytes - 1, dtype=torch.int8))

    def test_nonintegral_kernel_ratio_is_rejected(self):
        spec = self.ns["mxfp_cache_spec"](FullAttentionSpec(block_size=768))
        with self.assertRaisesRegex(ValueError, "whole number of kernel blocks"):
            self.reshape(spec, torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8))


if __name__ == "__main__":
    unittest.main()
