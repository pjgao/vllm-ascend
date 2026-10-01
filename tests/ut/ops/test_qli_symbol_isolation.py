# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source identity checks; these do not replace an audit of installed binaries."""

import ast
import json
import re
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
CUSTOM = "vllm_quant_lightning_indexer_v2"


class TestQliSymbolIsolation(unittest.TestCase):
    def test_c4_routes_both_stages_to_cann(self):
        # Execute the real methods in isolation from device_op's Triton/NPU
        # imports. This checks dispatch and argument plumbing, not NPU math.
        path = ROOT / "vllm_ascend/device/device_op.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "A5DeviceAdaptor")
        methods = {"_load_cann_quant_lightning_indexer_ops", "indexer_select_post_process"}
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        module = ast.parse("from __future__ import annotations")
        module.body.append(cls)
        metadata, indices = object(), object()
        native = SimpleNamespace(
            quant_lightning_indexer_metadata=Mock(return_value=metadata),
            quant_lightning_indexer=Mock(return_value=(indices, None)),
        )
        importer = Mock(return_value=native)
        namespace = {"BaseDeviceAdaptor": object, "torch": torch, "import_module": importer, "lru_cache": lru_cache}
        exec(compile(module, str(path), "exec"), namespace)
        adaptor = namespace["A5DeviceAdaptor"]
        query_scale, key_scale, key, block_table = object(), object(), object(), object()
        result = adaptor.indexer_select_post_process(
            q_li=torch.empty(1, 32, 64),
            q_li_scale=query_scale,
            q_li_shape_ori=(1, 32, 128),
            weights=torch.ones(1, 32, dtype=torch.float16),
            kv_cache=(key, key_scale),
            indexer_k_cache_idx=0,
            indexer_scale_cache_idx=1,
            attn_metadata=SimpleNamespace(block_table=block_table),
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([4096], dtype=torch.int32),
            enable_sparse_li_c8=False,
            enable_sparse_li_c4=True,
            use_torch_npu_lightning_indexer=False,
        )
        self.assertIs(result, indices)
        importer.assert_called_once_with("cann_ops_transformer.ops")
        native.quant_lightning_indexer_metadata.assert_called_once()
        native.quant_lightning_indexer.assert_called_once()
        meta_args = native.quant_lightning_indexer_metadata.call_args.kwargs
        call = native.quant_lightning_indexer.call_args
        self.assertEqual(meta_args["quant_mode"], 5)
        self.assertEqual(call.kwargs["quant_mode"], 5)
        self.assertIs(call.kwargs["metadata"], metadata)
        self.assertIs(call.kwargs["block_table"], block_table)
        self.assertEqual(call.args[0].shape, (1, 32, 64))
        self.assertEqual(call.args[2].dtype, torch.float32)
        torch.testing.assert_close(call.kwargs["cu_seqlens_q"], torch.tensor([0, 1], dtype=torch.int32))

    def test_custom_sources_do_not_claim_native_identity(self):
        native = re.compile(r"(?<!Vllm)QuantLightningIndexerV2|(?<!vllm_)quant_lightning_indexer_v2")
        for directory in (CUSTOM, CUSTOM + "_metadata"):
            for path in (ROOT / "csrc/attention" / directory).rglob("*"):
                if path.suffix in {".h", ".cpp", ".json"} or path.name == "CMakeLists.txt":
                    with self.subTest(path=path.relative_to(ROOT)):
                        self.assertIsNone(native.search(path.read_text(encoding="utf-8")))

    def test_torch_calls_have_device_and_meta_registration(self):
        binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
        meta = (ROOT / "csrc/torch_binding_meta.cpp").read_text(encoding="utf-8")
        for name in ("npu_" + CUSTOM, "npu_" + CUSTOM + "_metadata", "npu_vllm_quant_lightning_indexer_v3"):
            self.assertIn('"' + name + "(", binding)
            self.assertIn('ops.impl("' + name + '", torch::kPrivateUse1', binding)
            self.assertTrue('ops.impl("' + name + '", torch::kMeta' in binding or 'ops.impl("' + name + '"' in meta)
        for path in (ROOT / "vllm_ascend").rglob("*.py"):
            for name in re.findall(
                r"torch\.ops\._C_ascend\.(npu_\w*quant_lightning_indexer_v[23]\w*)", path.read_text(encoding="utf-8")
            ):
                with self.subTest(path=path.relative_to(ROOT), name=name):
                    self.assertIn('"' + name + "(", binding)

    def test_metadata_both_api_copies_are_isolated(self):
        root = ROOT / "csrc/attention" / (CUSTOM + "_metadata")
        for relative in ("op_api", "op_host/op_api"):
            for extension in ("cpp", "h"):
                path = root / relative / ("aclnn_" + CUSTOM + "_metadata." + extension)
                source = path.read_text(encoding="utf-8")
                self.assertIn("aclnnVllmQuantLightningIndexerV2MetadataGetWorkspaceSize(", source)
                self.assertIn("aclnnVllmQuantLightningIndexerV2Metadata(", source)
        config = json.loads((root / "op_kernel_aicpu" / (CUSTOM + "_metadata_aicpu.json")).read_text())
        self.assertEqual(set(config), {"VllmQuantLightningIndexerV2Metadata"})

    def test_custom_adapter_uses_extended_abi_only(self):
        binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
        adapter = (ROOT / "csrc/attention" / CUSTOM / (CUSTOM + "_torch_adpt.h")).read_text(encoding="utf-8")
        self.assertNotIn("aclnnQuantLightningIndexerV2", binding + adapter)
        self.assertNotIn("EXEC_NPU_CMD(aclnnVllmQuantLightningIndexerV2,", binding)
        self.assertIn("EXEC_NPU_CMD(aclnnVllmQuantLightningIndexerV2,", adapter)
        self.assertIn("keyScaleStride0, sparseIndicesOut, sparseValuesOut, candidateTopkIndexOut", adapter)


if __name__ == "__main__":
    unittest.main()
