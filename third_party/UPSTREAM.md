# Upstream: HypoSpace

- Source: https://github.com/CTT-Pavilion/_HypoSpace
- Pinned commit: `c69e9318577b34b5b896996571aefd4ba6053f58` (2026-05-04)
- License: MIT (see the upstream repository; the vendored tree keeps the code unchanged)
- Paper: Chen et al., *HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation
  under Underdetermination and Sublinear Coverage Bounds*, ICML 2026 (arXiv:2510.15614)

Local modifications are kept out of this tree wherever possible. The only edits are the
`local` LLM adapter (`LocalLLM` in `*/modules/llm_interface.py`), `--output` support for the
boolean runner, and passing `HYPOSPACE_API_KEY` through to `LocalLLM`. They are captured as a
patch against the pinned commit:

```bash
git clone https://github.com/CTT-Pavilion/_HypoSpace /tmp/hs
git -C /tmp/hs checkout c69e9318577b34b5b896996571aefd4ba6053f58
git -C /tmp/hs apply "$PWD/patches/hypospace-local-llm.patch"
```

Do not edit files under `third_party/` directly: change the patch instead so the upstream diff
stays reviewable.
