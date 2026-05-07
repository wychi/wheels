---
name: enable-kernel
description: Use when the user asks to enable, debug, or get a uTLX kernel running against the current `utlx_plugin` wheel. Triggers on phrases like "enable this kernel", "make this kernel work", "fix this tlx kernel", "why doesn't <kernel.py> compile", or when the user runs `python runner/runner.py kernels/<X>.py` and hits a failure. Walks through triage (kernel bug vs patch-layer bug), the established bridging workflow (match the failure against the patch catalog, bisect for the minimum patch set, decide patch-vs-rebuild for novel failures), and documents the outcome.
---

# Enable a uTLX Kernel

Goal: get the kernel at `<KERNEL>` to run end-to-end (or, failing that,
document precisely what's blocking it and where the proper fix belongs).

## Context to read first

1. `CLAUDE.md` — repo layout, build chain, known issues.
2. `runner/CLAUDE.md` — runner architecture, **patch catalog table**
   (canonical mapping from error symptom → patch that fixes it),
   diagnostic recipes.
3. `ENABLEMENT.md` — per-wheel-commit history. Find the section for the
   currently-installed wheel commit (or note it's missing — see
   workflow step 1).
4. `runner/tlx_patches.py` — patch implementations. Each docstring
   carries the bridge rationale and a `Retire when:` hint.
5. `runner/tlx_patches.toml` — per-wheel patch selection.
6. **Working reference kernels** — `kernels/tiny_gemm.py` (single-shot
   wgmma) and `kernels/hopper_ws.py` (warp-specialized + TMA). When in
   doubt about uTLX API usage, compare the user's kernel against these.

## Workflow

### 1. Identify the wheel

```bash
pip show utlx | awk '/^Version/ {print $2}'
```

If `ENABLEMENT.md` already has a section for this commit and a
`[utlx."<commit>"]` entry in `tlx_patches.toml` — start with that
patch list. If not, you're doing first-time enablement on this wheel.

### 2. Run the kernel as-is

```bash
source .venv/bin/activate
python runner/runner.py <KERNEL> 2>&1 | tail -40
```

Capture the failure. Then **triage** before patching.

### 3. Triage: kernel bug or patch-layer bug?

The patch layer is not the only thing that can be wrong. The kernel
itself might be buggy. Check these indicators **before** matching
against the patch catalog:

**Kernel bug indicators** (fix the kernel, not the patches):

- Python-level errors at the kernel's own line: `NameError`,
  `ImportError`, `SyntaxError`, `AttributeError` on a name *defined in
  the kernel*, off-by-one in shapes, wrong arg order to `tlx.*`.
- `assert` failure in the kernel itself (e.g. `assert M == K`).
- Compilation succeeds but the kernel's own validation check fails
  (e.g. `rel_err > 0.01`, `assert torch.allclose(...)`). That's
  numerical correctness, not bridging.
- Error mentions a `tlx.<api>(...)` call with arguments that don't
  match the signature in `utlx_plugin/<mem|mma>_ops.py`. Open the
  utlx source and verify the kernel is calling it correctly. Compare
  against `kernels/tiny_gemm.py` / `kernels/hopper_ws.py` as known-good
  references.
- Use of a uTLX feature the wheel doesn't implement (e.g. APIs that
  exist in some branches but not the installed wheel — grep
  `utlx_plugin/__init__.py`'s exports).

**Patch-layer bug indicators** (continue to step 4):

- Errors involving `triton._C.libtriton.*` symbols, `ir.builder` /
  `GluonOpBuilder` method names, MLIR pass names, or
  `incompatible function arguments` from pybind.
- Errors at compile-time (`make_ttir` / `make_ttgir` / etc.) that
  don't reference the kernel's own logic.
- Symptom matches a row in `runner/CLAUDE.md` → "Patch catalog".
- A previously-working kernel breaks after a wheel bump.

If kernel bug: fix it, re-run, repeat triage until you're either done
or hitting a patch-layer issue. Document the kernel fix in your
final report.

### 4. Activate or add the relevant patch

For catalog-known symptoms:
- If the patch exists but isn't active for this wheel, either declare
  `__tlx_patches__ = [...]` at the top of the kernel file (consulted
  via AST — kernel isn't executed for this lookup) or add a
  `[utlx."<commit>"]` entry to `tlx_patches.toml`.
- Re-run. Iterate.

### 5. Bisect to find the minimum patch set (recommended for new wheels)

For a clean record on a new wheel commit, follow the [Patch
re-evaluation playbook](../../../ENABLEMENT.md#patch-re-evaluation-playbook-run-on-every-new-wheel):
start with `__tlx_patches__ = []` and add patches additively, one per
failure, until you either reach success or a wall. This produces the
minimum required list and exposes which (if any) previous patches the
new wheel obsoleted.

### 6. Diagnose novel failures

If the error doesn't match any catalog entry, run the diagnostic
recipes in `runner/CLAUDE.md` ("Diagnostic recipe" section). Common
error shapes:

- `'ir.builder' object has no attribute 'create_X'` → op moved to
  `GluonOpBuilder`.
- `incompatible function arguments. Supported: 1. (...)` → binding
  signature drift.
- `Did you forget to add @triton.jit ?` → host-side call to a
  `@tl.builtin`.
- `... materialization ... that remained live after conversion` →
  conversion-pass leftover, usually a `tlx.*_layout` marker the C++
  pass doesn't lower.
- `... operand count (N) does not match ... operandSegmentSizes` →
  plugin C++ op constructor bug.

Then decide: **patch or rebuild?**
- **Add a patch** if: fix is ≲50 lines, Python-only, unblocks the
  kernel today.
- **Document as wheel-rebuild blocker** if: needs C++ (new pybind, new
  MLIR conversion pattern, op-builder fix), or the patch grows past
  ~100 lines, or patches start interacting in fragile ways.

When adding a patch:
- Append to `runner/tlx_patches.py` in registration order (apply
  order matters — see file header). Use `@register("<name>", default=True)`.
- Docstring: brief rationale + a `Retire when:` line stating what
  upstream change makes this patch obsolete.
- Add the new patch's name to `[utlx."<commit>"]` in
  `tlx_patches.toml`, AND a new row in `runner/CLAUDE.md` →
  "Patch catalog" with the verbatim error symptom.
- Classify into one of the follow-up tables in `ENABLEMENT.md` →
  "Patch Follow-ups" (`utlx-py` / `utlx-cpp` / `triton`) so future
  rebuilds know where the proper fix belongs.

### 7. Stop conditions

Stop and report (don't keep churning) when:

- Kernel passes validation. Go to step 8.
- You've hit a wall already documented in `ENABLEMENT.md` for the
  current wheel (e.g. the `tlx.release_layout` blocker for tiny_gemm).
  Don't try to bridge it again — confirm the symptom matches and point
  at the existing doc.
- A novel wall would require >50 lines of Python patching, or C++
  changes, or risky cross-patch interactions. Document the blocker as
  a new `## <commit>` section in `ENABLEMENT.md` (or extend the
  existing one) with: symptom, diagnosis, attempted approaches, the
  classification (`utlx-cpp` / `utlx-py`), and a key-files reference.

### 8. Validate + document

If the kernel passes:
- Run any explicit correctness check the kernel includes (e.g.
  `tiny_gemm` asserts `rel_err < 0.01`). Numerical correctness gates
  "passes", not just compile-success.
- If you bisected (step 5) and the wheel obsoleted any patches, set
  `default=False` on the retired ones in `tlx_patches.py` (don't
  delete — older wheels may still need them) and update their
  `Retire when:` lines.
- Write or update the `## <commit>` section in `ENABLEMENT.md` per
  the playbook (step 6 of [Patch re-evaluation
  playbook](../../../ENABLEMENT.md#patch-re-evaluation-playbook-run-on-every-new-wheel)):
  bisect order, minimum required list, not-exercised list, wheel-side
  delta vs the previous commit. If you fixed a kernel bug, note that
  separately from patch work.
- Commit in topical chunks. Suggested split:
  1. Kernel fixes if any (separate commit, separate from bridge work).
  2. Patch additions (`runner/tlx_patches.py` + matching `.toml` row +
     catalog row in `runner/CLAUDE.md`).
  3. Doc updates (`ENABLEMENT.md` per-wheel section, follow-up table
     edits).

## Hard rules (don't skip)

1. **Triage before patching.** A kernel bug looks like a patch bug if
   you don't check. Use step 3 — verify against `kernels/tiny_gemm.py`
   / `kernels/hopper_ws.py` and the utlx source signatures before
   touching `tlx_patches.py`.

2. **Read the catalog before adding a patch.** The symptom-to-patch
   table in `runner/CLAUDE.md` → "Patch catalog" maps verbatim error
   strings to the patch that fixes them. Most first-time failures are
   already covered — adding a new patch when one already exists wastes
   time and bloats the registry.

3. **Bisect, don't inherit.** When the wheel commit is new (no
   `[utlx."<commit>"]` entry in `runner/tlx_patches.toml`), do the
   additive bisect from `__tlx_patches__ = []`. Inheriting the
   previous commit's full list defeats the point of the rebuild —
   patches that were once needed may have been fixed in C++.

4. **Stop at known walls.** If you hit a failure already documented
   as a wheel-rebuild blocker in `ENABLEMENT.md` for the current
   commit, do **not** try to bridge it again. Confirm the symptom
   matches and stop. Report the existing doc link.

5. **Be honest about partial wins.** A patch that advances the failure
   point without fully unblocking the kernel is still useful — commit
   it with the trade-off in its docstring. Don't claim "fixed" when
   the kernel still fails downstream; say what it does and doesn't do.

6. **Classify every new patch.** When adding to `tlx_patches.py`:
   (a) add a row to the catalog in `runner/CLAUDE.md`,
   (b) add to a `[utlx."<commit>"]` entry in `tlx_patches.toml`,
   (c) classify into one of the follow-up tables in `ENABLEMENT.md`
       (`utlx-py` / `utlx-cpp` / `triton`) so the next wheel rebuild
       knows where the proper fix belongs.

## Quick reference

- **Which wheel am I on?** `pip show utlx | awk '/^Version/ {print $2}'`
- **Run the kernel:** `python runner/runner.py kernels/<X>.py 2>&1 | tail -40`
- **Patch catalog:** `runner/CLAUDE.md` → "Patch catalog"
- **Per-wheel history + bisect playbook:** `ENABLEMENT.md`
- **Patch implementations:** `runner/tlx_patches.py` (each docstring
  has a `Retire when:` hint)
- **Working reference kernels:** `kernels/tiny_gemm.py`,
  `kernels/hopper_ws.py`
- **utlx Python source (for verifying kernel API usage):**
  `~/oss/wheels/.venv/lib/python*/site-packages/utlx_plugin/`
