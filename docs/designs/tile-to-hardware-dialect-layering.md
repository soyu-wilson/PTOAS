# Tile → Hardware Dialect Layering (RFC)

Status: **Proposal / RFC**
Author: PTOAS compiler team
Scope: dialect architecture, `include/PTO/IR/*.td`, `lib/PTO/Transforms`, `tools/ptoas`, `docs/`

> ⚠️ This document proposes a dialect refactor and introduces a new
> **`tvec`** connecting dialect (scaffolding under `include/Tvec/`). The new
> dialect is intentionally **not yet wired into the top-level build / dialect
> registry** so the existing toolchain keeps building unchanged; the exact
> opt-in steps are listed in [§7 Rollout](#7-rollout-plan). None of the
> TableGen/C++ here has been compiled in the authoring environment — it is an
> RFC artifact to review and build against `llvmorg-19.1.7`.

---

## 1. Background: one dialect, two abstraction levels

Today the project ships a **single `pto` dialect** that mixes two
fundamentally different abstraction levels. This is documented in
`docs/designs/ptoas-tileop-expand-design.md` (§1.2), which explicitly calls
out *"两层粒度的 IR"* (two granularities of IR) living in the same dialect:

| Concern | "TileOp" level | "vPTO" level |
| --- | --- | --- |
| Example ops | `pto.tadd`, `pto.tmul`, `pto.tcvt`, `pto.tload` | `pto.vlds`, `pto.vadd`, `pto.vsts`, `pto.vcvt`, `pto.vdup` |
| Data type | `!pto.tile_buf<…>` (whole tile) | `!pto.vreg<64xf32>` (physical 256-byte register), `!pto.mask<b32>` |
| Semantics | one instruction = full tile semantics | explicit loops, explicit register width, explicit predication |
| Hardware coupling | **agnostic** (virtual tile) | **tied to the Da Vinci / Ascend NPU micro-ISA** |
| Loop structure | none (implicit) | explicit `pto.vecscope { scf.for { scf.for { … } } }` |
| TableGen home | `include/PTO/IR/PTOOps.td` | `include/PTO/IR/VPTOOps.td` |

The file split (`PTOOps.td` vs `VPTOOps.td`), the CLI flags
(`--pto-backend=vpto --emit-vpto`), and the pass names (`pto-infer-vpto-vecscope`,
`pto-validate-vpto-ir`) all already treat "vPTO" as a *de facto* separate layer
— but every op still lives under the same `pto` dialect name and C++ namespace
(`PTO_Op<"vadd">`, `TypeDef<PTO_Dialect, "VReg">`).

### 1.1 How the levels connect today

`pto-expand-tile-op` invokes the **TileLang Python DSL** templates in
`lib/TileOps/*.py` to lower a tile op directly into hardware-ISA vector ops.
For example `lib/TileOps/tadd_template.py`:

```python
@pto.vkernel(target="a5", op="pto.tadd")
def template_tadd(src0, src1, dst):
    for row in range(0, valid_rows, 1):
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
```

produces, after expand + inline + fold:

```mlir
pto.vecscope {
  scf.for %r = %c0 to %c16 step %c1 {
    scf.for %c = %c0 to %c64 step %c64 {
      %va = pto.vlds %vecA[%r, %c] : … -> !pto.vreg<64xf32>
      %vb = pto.vlds %vecB[%r, %c] : … -> !pto.vreg<64xf32>
      %vc = pto.vadd %va, %vb       : !pto.vreg<64xf32>, … -> !pto.vreg<64xf32>
      pto.vsts %vc, %vecC[%r, %c]   : !pto.vreg<64xf32>, …
    }
  }
}
```

**The jump from `pto.tadd` straight to the physical-register ISA happens in one
step.** There is no IR level in between where the loop nest exists but the
operations are still hardware-neutral. That missing middle level is exactly
where target-independent loop transformations — most importantly **loop
fusion** — belong.

## 2. Problem statement

In MLIR's design philosophy a dialect should model **one** abstraction level,
and lowering is a sequence of dialect-to-dialect conversions. Mixing the
virtual tile abstraction and the concrete hardware ISA in one dialect causes:

1. **No clean fusion surface.** We want to fuse the `scf.for` nests of, say,
   `tadd` followed by `tmul` so the intermediate tile never round-trips through
   memory. But today the only place those loops exist is *after* lowering to the
   physical ISA, where each value is a `!pto.vreg<64xf32>` bound to register
   pressure, `dist`/`part` encodings, and (soon) pipe/sync semantics. Proving
   fusion legal and rewiring intermediate values is far harder against the rigid
   ISA than against neutral SSA vectors.
2. **Premature commitment to register width.** `pto.vlds … -> !pto.vreg<64xf32>`
   bakes in the 256-byte / 64-lane physical width before any loop optimization
   runs, so fusion has to reconcile already-tiled loops instead of choosing one
   tiling for the fused body.
3. **Verifier / type confusion.** `!pto.tile_buf` and `!pto.vreg` constraints,
   `pto.*` tile verifiers and `pto.v*` ISA verifiers all share one dialect, so
   "is this op legal at this stage?" has no dialect-level answer (it is
   approximated by ad-hoc `pto-validate-vpto-ir`).
4. **Backend portability.** A second hardware backend would need its own ISA
   ops, but there is no neutral level to lower *from*; everything is entangled
   with the Da Vinci `vreg`.

## 3. Proposal: a three-level dialect stack

Separate the concerns into three dialects, lowered progressively:

```
┌──────────────────────────────────────────────────────────────────┐
│  pto   (TILE / virtual)            UNCHANGED                       │
│        pto.tadd / pto.tmul / pto.tcvt …   on  !pto.tile_buf        │
│        whole-tile, hardware-agnostic                              │
└──────────────────────────────────────────────────────────────────┘
            │  pto-expand-tile-op   (templates now emit `tvec`, not `vpto`)
            ▼
┌──────────────────────────────────────────────────────────────────┐
│  tvec  (TILED-VECTOR / machine-independent)        ★ NEW ★         │
│        scf.for nests + tvec.load / tvec.add / tvec.store           │
│        on  !tvec.vec<Nxf32>, !tvec.mask  (logical, width-free)     │
│        ── this is the loop-fusion playground ──                    │
└──────────────────────────────────────────────────────────────────┘
            │  tvec-fuse-loops        (loop / producer-consumer fusion)
            │  tvec-select-width      (pick physical tiling per target)
            │  tvec-lower-to-vpto     (bind to the concrete ISA)
            ▼
┌──────────────────────────────────────────────────────────────────┐
│  vpto  (HARDWARE ISA / Da Vinci)        split out of `pto`         │
│        vpto.vlds {dist} / vpto.vadd / vpto.vsts {dist}             │
│        on  !vpto.vreg<64xf32>, !vpto.mask<b32>, vpto.vecscope      │
│        pipes / sync / events                                       │
└──────────────────────────────────────────────────────────────────┘
            │  insert-sync, plan-memory, lower-to-emitc / llvm …
            ▼
        EmitC / LLVM  →  Da Vinci binary
```

Two dialect-level changes occupy the "connecting" region the task asks for:

- **`tvec` — brand-new** machine-independent tiled-vector dialect. This is the
  layer that did not exist before. Its whole reason to exist is to host the
  explicit loop nest *while the operations are still hardware-neutral*, so that
  loop fusion and width selection are clean, target-independent rewrites.
- **`vpto` — formalized** by splitting the existing `pto.v*` ops and
  `!pto.vreg` / `!pto.mask` types out of `pto` into their own dialect/namespace.
  Semantically unchanged; this just gives the *real hardware abstraction* its
  own dialect so the stage boundary is explicit. (See §6 for the migration; it
  is mechanical and can land after `tvec`.)

`pto` itself keeps its current role as the **virtual tile** dialect.

## 4. The `tvec` dialect

`tvec` = "tiled vector". It is the structured, machine-independent vectorized
form of a tile op: the loop nest is explicit (plain `scf.for`), but values are
**logical** vectors and masks with no physical width, no `dist`/`part` mode, no
register file, and no pipe/sync. Tile storage is referenced through `memref`
(exactly as the expanded form already does via `pto.tile_buf_addr`).

Scaffolding lives in `include/Tvec/IR/`:

| File | Contents |
| --- | --- |
| `TvecDialect.td` | dialect definition (`name = "tvec"`) |
| `TvecTypes.td` | `!tvec.vec<Nx T>`, `!tvec.mask<T>` |
| `TvecInterfaces.td` | `TvecTileAccessOpInterface`, `TvecElementwiseOpInterface` |
| `TvecOps.td` | `tvec.load/store/add/mul/…/broadcast/convert/make_mask` |
| `TvecDialect.h` | C++ aggregation header |
| `CMakeLists.txt` | tablegen targets (opt-in, see §7) |

### 4.1 Types

```mlir
!tvec.vec<64xf32>     // a logical vector strip of 64 f32 lanes.
                      //   N is the *logical* strip length chosen by the
                      //   producing template, NOT the physical register width.
!tvec.mask<f32>       // a logical predicate carrying only its element type;
                      //   it lowers to !vpto.mask<b32> once width is bound.
```

The element count `N` is a hint, not a hardware commitment. `tvec-select-width`
(§5) re-tiles `N` to the target register width and only then do we get
`!vpto.vreg<64xf32>`.

### 4.2 Representative ops

```mlir
// memref tile  ──load──▶  logical vector
%v = tvec.load %tile[%r, %c], %mask
        : memref<16x64xf32, #pto.address_space<vec>>, !tvec.mask<f32>
        -> !tvec.vec<64xf32>

// elementwise, pure, value semantics — no register, no pipe
%s = tvec.add %a, %b, %mask : !tvec.vec<64xf32>, !tvec.mask<f32>

// logical vector ──store──▶ memref tile (the only side-effecting op besides load)
tvec.store %s, %tile[%r, %c], %mask
        : !tvec.vec<64xf32>, memref<16x64xf32, #pto.address_space<vec>>, !tvec.mask<f32>

%m, %rem = tvec.make_mask %dtype, %remaining   // tail predication
%bc      = tvec.broadcast %scalar : f32 -> !tvec.vec<64xf32>
%cv      = tvec.convert %a {rnd, sat} : !tvec.vec<64xf32> -> !tvec.vec<128xf16>
```

The op set is intentionally a small, neutral superset of the `pto.v*` family
used by the templates (`vlds/vsts/vadd/vsub/vmul/vdiv/vmax/vmin/vand/vor/vdup/
vcvt/…`), minus everything hardware-specific (`dist`, `part`, granularity-typed
masks, `vecscope`, pipes).

### 4.3 Interfaces that make fusion cheap

Two op interfaces (`TvecInterfaces.td`) expose exactly what a loop-fusion pass
needs without hard-coding op names:

- **`TvecTileAccessOpInterface`** — implemented by `tvec.load` / `tvec.store`.
  Reports `getTile()` (the memref), `getRowIndex()`, `getColIndex()`, and
  `isWrite()`. A fusion pass uses this to build the read/write sets of two loop
  bodies and decide independence — instead of pattern-matching memref subscripts.
- **`TvecElementwiseOpInterface`** — implemented by the arithmetic ops. Marks an
  op as pure, per-lane, and safe to sink/merge across a fused loop boundary, and
  exposes its mask operand so the fuser can verify predicate compatibility.

Because every non-memory `tvec` op is `Pure` and operates on SSA values, the
classic blockers to ISA-level fusion (physical register aliasing,
`dist`-mode mismatches, pipe ordering) simply do not exist at this level.

## 5. Loop fusion — the centerpiece

The task specifically asks us to design "with loop fusion on the generated
`scf.for` in mind." `tvec` is built for it.

### 5.1 Why fuse at `tvec`, not at `vpto`

| | fuse at `tvec` (proposed) | fuse at `vpto` (today's only option) |
| --- | --- | --- |
| Intermediate value | SSA `!tvec.vec` — stays in a value (≈ register) when fused | `!pto.vreg<64xf32>` physical reg, + a memory round-trip of the whole intermediate tile |
| Legality check | data-flow + memref dependence on `TvecTileAccessOpInterface` | also must reason about `dist`/`part` encodings, pipes, sync events, `vecscope` semantics |
| Width | not yet fixed → choose one tiling for the fused body | already tiled to 64 lanes → must reconcile two fixed tilings |
| Target coupling | none — same pass works for any backend | entangled with Da Vinci ISA |

### 5.2 `tvec-fuse-loops` sketch

Input — two tile ops expanded independently (`tadd` then `tmul`, `dst1` feeds
`dst2`), each its own loop nest, with `dst1` materialized to memory in between:

```mlir
// from pto.tadd  (dst1 = a + b)
scf.for %r = … { scf.for %c = … {
  %va = tvec.load %A[%r,%c], %m
  %vb = tvec.load %B[%r,%c], %m
  %s  = tvec.add  %va, %vb, %m
  tvec.store %s, %D1[%r,%c], %m            // <- intermediate written to memory
}}
// from pto.tmul  (dst2 = dst1 * c)
scf.for %r = … { scf.for %c = … {
  %vd = tvec.load %D1[%r,%c], %m           // <- and read straight back
  %vc = tvec.load %C[%r,%c], %m
  %p  = tvec.mul  %vd, %vc, %m
  tvec.store %p, %D2[%r,%c], %m
}}
```

The pass:

1. **Groups candidate sibling loops** by identical iteration domain. We reuse
   the existing `IterationDomainInfo` / `IterationDomainClass` machinery in
   `include/PTO/Transforms/TileFusion/FusionAnalysis.h` (already used for
   tile-level fusion) so the domain-equality proof is shared, not reinvented.
2. **Builds read/write sets** via `TvecTileAccessOpInterface` and checks that
   the only producer→consumer coupling is the forwarded intermediate `%D1`
   (a true dependence we can satisfy by fusion), with no anti/output hazard on
   other tiles.
3. **Fuses the nests** and **forwards the SSA value**: the consumer's
   `tvec.load %D1` is replaced by the producer's `%s`, and the now-dead
   `tvec.store %s, %D1` / matching load are erased (DCE on `%D1`).

Output — one loop nest, intermediate stays in a register, two loads + one store
eliminated:

```mlir
scf.for %r = … { scf.for %c = … {
  %va = tvec.load %A[%r,%c], %m
  %vb = tvec.load %B[%r,%c], %m
  %s  = tvec.add  %va, %vb, %m            // %D1 never touches memory
  %vc = tvec.load %C[%r,%c], %m
  %p  = tvec.mul  %s, %vc, %m
  tvec.store %p, %D2[%r,%c], %m
}}
```

Only after this (and `tvec-select-width`) do we lower to `vpto`, so the fused,
width-selected body is what becomes `vpto.vlds/vadd/vmul/vsts` inside a single
`vpto.vecscope`. Sync insertion and memory planning then see the *already
fused* schedule.

### 5.3 Relationship to existing tile-level fusion

`pto-fusion-plan` / `pto-op-scheduling` (in `lib/PTO/Transforms/TileFusion/`)
fuse at the **tile** level — i.e. which `pto.tadd`/`pto.tmul` ops share an
iteration domain and may be scheduled together, *before* expansion. `tvec`
fusion is the **loop-level** realization of those decisions *after* expansion:
the tile-level planner decides *what* to fuse; `tvec-fuse-loops` actually merges
the `scf.for` bodies and forwards registers. The two are complementary, and the
shared `IterationDomainInfo` types keep their notion of "same domain" in sync.

## 6. Splitting `vpto` out of `pto` (migration)

Formalizing the hardware level is mechanical and can land independently of
`tvec`:

1. Move `PTO_VldsOp`, `PTO_VaddOp`, …, and the `VReg`/`Mask`/`Align` `TypeDef`s
   from `VPTOOps.td` / `VPTOTypeDefs.td` into a `VPTO_Dialect`
   (`name = "vpto"`, `cppNamespace = "::mlir::vpto"`), renaming the base class
   `PTO_Op<"vadd">` → `VPTO_Op<"vadd">`. Op spellings change `pto.vadd` →
   `vpto.vadd`.
2. Update the registry in `tools/ptoas/ptoas.cpp` (`registry.insert<…>`),
   `lib/CAPI`, and the Python bindings (`lib/Bindings/Python`).
3. Update `lib/PTO/Transforms/*` that build/match these ops (the VPTO* passes,
   `ExpandTileOp`, `PTOToEmitC`, the LLVM emitter) and the `.pto` tests /
   `FileCheck` lines (`pto.vlds` → `vpto.vlds`).

Because this is a large rename across the whole tree (and across the Python
templates' emitted spellings), it is staged **after** `tvec` lands; the doc
records it so the end-state is unambiguous. Until then `vpto` semantics
continue to live under the `pto.v*` spelling and `tvec-lower-to-vpto` simply
targets those existing ops.

## 7. Rollout plan

The new dialect is delivered as **self-contained scaffolding that does not
touch the existing build**. To keep `ninja` green, `include/Tvec/` is *not*
referenced by `include/CMakeLists.txt` yet.

Phased enablement:

1. **(this change)** Land the `tvec` TableGen + headers + RFC. No build wiring,
   no behavior change.
2. Wire the build: add `add_subdirectory(Tvec)` to `include/CMakeLists.txt`,
   add a `TvecDialect` C++ library under `lib/`, and `registry.insert<
   mlir::tvec::TvecDialect>()` in `tools/ptoas/ptoas.cpp`. Verify tablegen +
   link against `llvmorg-19.1.7`.
3. Retarget the TileLang templates' primitive emitters (`pto.vlds`/`pto.vadd`/
   `pto.vsts` → `tvec.load`/`tvec.add`/`tvec.store`) behind a
   `--emit-tvec` flag, keeping the direct-to-vpto path as default until parity.
4. Implement `tvec-fuse-loops`, `tvec-select-width`, `tvec-lower-to-vpto`; add
   lit tests under `test/lit/tvec/` (parse round-trip, a fusion before/after,
   width selection).
5. Flip the default pipeline to route through `tvec`; then do the `vpto` split
   (§6).

## 8. Cross-layer checklist (per `.claude/rules/cross-layer-sync.md`)

This RFC stage only adds the **ODS/dialect** layer (new `tvec` `.td` + headers)
plus **docs**, with no behavioral wiring, so no other layer changes yet. The
follow-up phases must keep these in sync:

- [ ] ODS: `tvec` ops print/parse; verifiers actionable. *(scaffolding added)*
- [ ] C++ IR: `TvecDialect` registration + verifiers. *(phase 2)*
- [ ] Lowering: `tvec-lower-to-vpto` handles every `tvec` op + masks/views. *(phase 4)*
- [ ] CLI: `--emit-tvec` gating in `tools/ptoas/ptoas.cpp`. *(phase 3)*
- [ ] Python: templates emit `tvec.*`; bindings if exposed. *(phase 3)*
- [ ] Docs/specs: this file + `docs/isa/` updates for the new level. *(this change)*
- [ ] Tests: `test/lit/tvec/` regression incl. a fusion case. *(phase 4)*

## 9. Alternatives considered

- **Keep one dialect, fuse at `vpto`.** Rejected: §5.1 — fusing against physical
  registers + pipes + `dist` modes is materially harder and target-specific, and
  it cannot remove the intermediate-tile memory round-trip cleanly.
- **Reuse the upstream `vector` dialect as the middle level.** Tempting, but the
  tile access pattern (predicated row/col strips over a `pto.tile_buf`-backed
  memref, tail masking via `make_mask`) and the need to share
  `IterationDomainInfo` with existing tile fusion make a small purpose-built
  `tvec` dialect a better fit than bending `vector` + `transfer_read/write`.
  `tvec` can still be lowered through `vector` later if desired.
- **One combined new dialect for both middle + ISA.** Rejected: it would
  re-create the very conflation we are removing; the machine-independent fusion
  surface must be a distinct level from the machine ISA.
