# QFA C8 adaptation to non-contiguous cache pages

This adaptation targets **ModelRunnerV1 only**. Select it explicitly with
`VLLM_USE_V2_MODEL_RUNNER=0` and `--kv-cache-dtype mxfp8`. ModelRunnerV2 has
no C8 allocation adaptation in this branch.

The source is AJF-cmd's `qfa_c8_paged` at
`42f8de515dd3b237013a2093696736f77c723e58`. Official `main` is merged at
`7df96048756856e4d30f29618129e9942ba6985c`, including
[PR #14340](https://github.com/vllm-project/vllm-ascend/pull/14340).
The upstream vLLM interface is the `ced6857afa` version used by that main.

## Integration

The QFA operators, query-scale planning, device sequence-length handling and
ModelSlim weight loading remain from the source branch. The adaptation is
limited to cache geometry and its lifecycle:

- Keep the real `head_size` and `head_size_v` in `FullAttentionSpec`. Account
  for E8M0 bytes through upstream `state_content_bytes`, preserving other
  spec fields. A one-head D=256 cache stores 528 bytes per token.
- Reuse main's combined raw allocation and hybrid planner. Build the four
  C8 views before generic attention reshaping. Remove the old requirement
  that the full K section equal the SSM state size.
- Use main's strided block-copy helper for K, V and both scales. Bitcast FP8
  payloads to INT8 for copying, avoiding unsupported FP8 stack/index ops.
  Stage sources before writes to support cyclic copies.
- Zero payload bytes using separate payload lengths and physical strides.
  Skip static V scales when clearing recycled pages. Fill those scales once
  before graph capture; an absent checkpoint scale defaults to E8M0 127.

There is no new environment variable, new runner implementation or V2 patch.
The cache shape helpers stay in `attention_c8_mxfp.py`.

## Physical pages and scale shapes

For kernel block size S, KV heads N and dimensions Dk/Dv, each kernel block
owns the following bytes:

```text
[K: N*S*Dk | K_scale: N*S*Dk/32 |
 V: N*S*Dv | V_scale: N*S*Dv/32 | padding]
```

The Python tuple order remains `(K, V, K_scale, V_scale)`. All four tensors
use the same dim-0 byte stride. Their inner dimensions are contiguous.
Every `as_strided` offset adds the raw slice's own storage offset exactly
once. Copying logical views requires no reconstruction of absolute offsets.

Let a scheduler block contain m kernel blocks. Its physical page size is P,
including padding, and every kernel view advances by `P / m`. P must divide
evenly by m, and the pitch must hold the complete kernel payload. The page
planner owns P; this adaptation does not increase it a second time.

For N=1, S=512 and Dk=Dv=256:

| Section | Byte offset | Payload bytes | Shape per cache |
| --- | ---: | ---: | --- |
| K | 0 | 131072 | `(blocks, 1, 8, 512, 32)` |
| K_scale | 131072 | 4096 | `(blocks, 1, 32, 4, 16, 2)` |
| V | 135168 | 131072 | `(blocks, 1, 8, 512, 32)` |
| V_scale | 266240 | 4096 | `(blocks, 1, 16, 8, 16, 2)` |

The unpadded pitch is **270336 bytes**. K/V are FP8 E4M3; scales are UINT8
storage bitcast to FP8 E8M0 at the QFA boundary. An INT8 scatter view is a
byte reinterpretation of FP8 data, not INT8 quantization with FP32 scales.

The PA_NZ scale proposal is retained:

```text
K_scale: [blocks, N, S/16, Dk/64, 16, 2]
V_scale: [blocks, N, Dv/16, S/64, 16, 2]
```

The physical PA_NZ K/V caches have five dimensions and scales have six.
Logical BBND K/V tensors have four dimensions; counting against those
logical shapes would make the scale tensors two dimensions larger.

For a future BBND operator, the compact equivalent scale shapes would be:

```text
K_scale: [blocks, N, S, Dk/64, 2]
V_scale: [blocks, N, S/64, Dv, 2]
```

Adding another S dimension to V_scale, as in
`[blocks, N, S, S/64, Dv, 2]`, repeats every scale S times and overallocates
by 512 for S=512. This branch continues to use the existing PA_NZ operator;
it does not add BBND or BNBD operator dispatch.

For the separate INT8/FP32 example with S=4, N=1 and D=128, a layout containing
only K's four FP32 scales has `[K:512 | K_scale:16 | V:512]`: byte stride
1040, V offset 528 and FP32 K_scale stride 260. If four FP32 V scales are
also appended, the stride becomes 1056, and the FP32 scale stride becomes
264. Token-interleaved K/V offsets such as 0/128/256 do not describe the
same contiguous K-then-V layout.

## Validation and limits

CPU regression coverage checks scale budgets, real dimensions, nonzero
storage offsets, padding, 1/2/8 kernel blocks per scheduler block, cyclic
copies, and clear ranges that preserve V scales and allocation guards.

The A5/CANN test
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_qfa_c8_paged_layout.py`
compares QFA output from strided scales against contiguous reference
caches for decode and prefill, then runs the real zeroing kernel. This test
must pass on the target operator delivery before claiming runtime support
for the scale strides. It has not been run in the Windows CPU workspace.
CPU tests do not prove CANN tiling, graph replay, model accuracy or throughput.

The existing QFA restrictions remain: causal attention, no sliding window,
no PCP/DCP or hamming sparse compression, and the source branch's Mooncake
restriction for PD transfer. Layout tests cover unequal K/V dimensions;
they do not establish QFA kernel support for unequal dimensions.
