# Codex Task: Clean Attention HLS Source Layout

## Objective

Refactor the current Attention HLS source tree into a clean, maintainable structure while preserving behavior, interfaces, numerical results, pragmas, and synthesis intent.

This task is **source organization and build cleanup only**.

Do not perform new architectural optimization in this pass.

---

## Target

- Tool: Vitis HLS 2021.2
- Device: `xcu250-figd2104-2L-e`
- Clock: `3.333 ns`
- Kernels:
  - `bert_qkv_kernel`
  - `bert_attn_core_kernel`
  - `bert_attn_out_norm_kernel`

Each kernel must synthesize independently and produce its own `csynth.rpt`.

---

## Required source layout

Use this structure:

```text
attention_hls/
├── include/
│   ├── bert_config.h
│   ├── bert_types.h
│   ├── bert_math.h
│   ├── bert_stream_utils.h
│   ├── bert_stage_tokens.h
│   └── bert_kernel_interfaces.h
│
├── kernels/
│   ├── bert_qkv_kernel.cpp
│   ├── bert_attn_core_kernel.cpp
│   └── bert_attn_out_norm_kernel.cpp
│
├── host/
│   └── host_stage_schedule.cpp
│
├── scripts/
│   └── run_vitis_attention_csynth.tcl
│
├── config/
│   └── system.cfg
│
├── reports/
│   ├── csynth_qkv.rpt
│   ├── csynth_attn_core.rpt
│   └── csynth_attn_out_norm.rpt
│
└── ATTENTION_FFN_SLR_ARCHITECTURE_HANDOFF.md
```

Do not keep duplicate files such as:

```text
file.cpp
file(1).cpp
old_file.cpp
new_file.cpp
final_file.cpp
```

Historical experiments may be moved to:

```text
archive/
```

They must not be part of the active Tcl build.

---

## Source organization rules

### 1. One top-level kernel per `.cpp`

Each active kernel must have exactly one top-level source file:

```text
kernels/bert_qkv_kernel.cpp
kernels/bert_attn_core_kernel.cpp
kernels/bert_attn_out_norm_kernel.cpp
```

Each file may contain:

- its top function;
- private `static` helper functions used only by that kernel;
- kernel-specific HLS pragmas;
- kernel-specific compile-time constants.

Do not put multiple top kernels into one giant `.cpp`.

Do not keep one monolithic `attention_stream.cpp` containing all QKV, Attention Core, and Output/Norm implementations.

Split its contents according to ownership:

```text
QKV projection helpers              -> bert_qkv_kernel.cpp
score/mask/softmax/context helpers  -> bert_attn_core_kernel.cpp
output projection/residual/LN       -> bert_attn_out_norm_kernel.cpp
```

### 2. Shared headers contain only shared code

Move only genuinely shared items into `include/`.

#### `bert_config.h`

Contains tensor sizes and compile-time configuration:

```cpp
SEQ_LEN
HIDDEN_SIZE
NUM_HEADS
HEAD_DIM
HEAD_PAR
GROUP_DIM
PACK_SIZE
PACKS
ATTN_PROJ_TILE_K
ATTN_PROJ_TILE_O
ATTN_OUT_TILE
```

#### `bert_types.h`

Contains shared HLS types:

```cpp
bus_t
hidden_stream_t
qkv_weight_bus_t
stage_token_t
```

#### `bert_math.h`

Contains small shared math helpers only:

```cpp
uint32_to_float
float_to_uint32
pack_bus16
unpack_bus16
fp32_sum4_tree
fp32_sum8_tree
fp32_dot4_tree
fp32_dot8_tree
```

Do not place large projection, softmax, layer-norm, or dataflow stages here.

#### `bert_stream_utils.h`

Contains small shared stream utilities only:

```cpp
duplicate_stream
store_hidden_stream
load_hidden_stream
```

Only keep a helper here when at least two kernels use it.

#### `bert_stage_tokens.h`

Contains:

```cpp
write_stage_token
wait_for_stage_token
```

#### `bert_kernel_interfaces.h`

Contains only public top-function declarations.

No large implementation should live in this header.

---

## Important implementation rule

Prefer this pattern:

```cpp
// one kernel source
#include "bert_kernel_interfaces.h"
#include "bert_math.h"

static void private_helper(...) {
    ...
}

extern "C" void bert_qkv_kernel(...) {
    ...
}
```

Do not use one large implementation header included by all kernels.

Do not include unused QKV code while synthesizing Attention Core.

Do not include unused Attention Core code while synthesizing Output/Norm.

Every kernel synthesis unit must contain only:

- shared lightweight headers;
- the selected kernel implementation;
- helpers reachable from that kernel.

---

## Existing file mapping

Refactor the current files as follows:

```text
bert_model.h
    -> split into:
       include/bert_config.h
       include/bert_types.h
       include/bert_math.h
       include/bert_stream_utils.h

bert_stage_kernels.h
    -> split into:
       include/bert_stage_tokens.h
       include/bert_kernel_interfaces.h

attention_stream.cpp
    -> split implementation into:
       kernels/bert_qkv_kernel.cpp
       kernels/bert_attn_core_kernel.cpp
       kernels/bert_attn_out_norm_kernel.cpp

bert_qkv_kernel.cpp
    -> keep top wrapper and merge QKV-only helpers into it

bert_attn_core_kernel.cpp
    -> keep top wrapper and merge Attention-Core-only helpers into it

bert_attn_out_norm_kernel.cpp
    -> keep top wrapper and merge Output/Norm-only helpers into it

host_stage_schedule.cpp
    -> move to host/

system.cfg
    -> move to config/
```

After migration, remove the active dependency on the original monolithic files.

---

## Preserve exactly

Do not change:

- public kernel names;
- public kernel argument order;
- AXI and AXIS interfaces;
- stream width;
- stream ordering;
- word counts;
- tensor dimensions;
- mathematical operations;
- weight layouts;
- storage bindings;
- partition/reshape factors;
- pipeline and unroll directives;
- dataflow structure;
- layer offsets;
- token synchronization semantics;
- `system.cfg` connectivity.

Do not silently enable or disable macros.

Kernel-specific macros must be applied only to the kernel that uses them.

For example:

```text
BERT_ATTN_INTERNAL_DATAFLOW
```

must not be added globally to QKV or Output/Norm unless their implementation actually requires it.

---

## Tcl requirements

Create:

```text
scripts/run_vitis_attention_csynth.tcl
```

The script must synthesize each kernel independently.

For each kernel:

1. Create a separate project.
2. Add only the required source file and include directory.
3. Set the correct top.
4. Use part `xcu250-figd2104-2L-e`.
5. Use clock `3.333 ns`.
6. Run only `csynth_design`.
7. Copy the generated report to a deterministic path.

Expected projects:

```text
build/proj_bert_qkv_kernel
build/proj_bert_attn_core_kernel
build/proj_bert_attn_out_norm_kernel
```

Expected reports:

```text
reports/csynth_qkv.rpt
reports/csynth_attn_core.rpt
reports/csynth_attn_out_norm.rpt
```

The script must support:

```bash
vitis_hls -f scripts/run_vitis_attention_csynth.tcl
```

and optionally one selected kernel:

```bash
vitis_hls -f scripts/run_vitis_attention_csynth.tcl bert_qkv_kernel
```

Do not compile host code in HLS projects.

Do not add every `.cpp` file to every project.

Suggested source mapping:

```tcl
bert_qkv_kernel:
    kernels/bert_qkv_kernel.cpp

bert_attn_core_kernel:
    kernels/bert_attn_core_kernel.cpp

bert_attn_out_norm_kernel:
    kernels/bert_attn_out_norm_kernel.cpp
```

Use:

```tcl
-cflags "-std=c++14 -I./include"
```

Add kernel-specific defines separately.

---

## Validation

Before refactoring, record the existing report values for each kernel:

- latency;
- interval;
- LUT;
- FF;
- BRAM;
- URAM;
- DSP;
- estimated clock.

After refactoring, run all three kernels again.

The refactor passes only when:

1. all three kernels complete `csynth_design`;
2. all three reports exist and are non-empty;
3. top-level interfaces are unchanged;
4. resource/latency results do not materially change;
5. no kernel includes unrelated implementation;
6. `system.cfg` still references valid kernel and port names.

Small report differences caused only by source hierarchy are acceptable, but any meaningful resource, latency, or II change must be explained and treated as a regression until proven otherwise.

---

## Cleanup policy

Delete only:

- duplicate active copies;
- obsolete generated project directories;
- root-level temporary reports;
- dead helpers proven unreachable;
- unused includes;
- superseded Tcl scripts after the new script passes.

Keep historical reports in:

```text
archive/reports/
```

Keep rejected optimization experiments in:

```text
archive/experiments/
```

Do not mix generated files with source files.

Add generated directories to `.gitignore`:

```gitignore
build/
*.log
.vitis_hls/
```

Do not ignore the curated reports in `reports/`.

---

## Execution order

1. Inspect all current source files and existing reports.
2. Build a call graph for each top kernel.
3. Identify shared helpers and kernel-private helpers.
4. Create the new directory structure.
5. Split the monolithic implementation.
6. Update includes and declarations.
7. Write the new Tcl script.
8. Synthesize QKV only.
9. Synthesize Attention Core only.
10. Synthesize Output/Norm only.
11. Compare new reports against the supplied baseline reports.
12. Verify `system.cfg`.
13. Remove duplicate active files only after all checks pass.

---

## Final response

Return:

### Changed files

List every created, moved, modified, and deleted file.

### Final tree

Print the active source tree.

### Kernel build mapping

Show which source files and defines are used for each kernel.

### Synthesis result

| Kernel | Status | Latency | II | LUT | FF | BRAM | URAM | DSP | Clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Compatibility checks

Confirm:

- kernel names unchanged;
- argument order unchanged;
- AXIS port names unchanged;
- `system.cfg` valid;
- reports generated.

Do not start architecture optimization after completing this cleanup.
