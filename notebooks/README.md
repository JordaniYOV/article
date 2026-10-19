# Historical notebook factories

Generated `.ipynb` files and ZIP distributions were removed on 19 October 2026.
Training now uses the CLI documented in
[`gpu_runbook.md`](../linear_head_approach/gpu_runbook.md); benchmark inference
uses the API/UI.

The three `build_ibm_*.py` files are historical source factories, not runnable
notebooks. They remain for source-level reproducibility and offline factory
tests. Generating or running their old workflows requires explicitly restoring
the retired inputs; no notebook or dataset is downloaded automatically.

Likewise, `linear_head_approach/build_notebook.py` and `bundle.py` remain because
their bytes are included in existing training fingerprints. Cleanup does not
alter checkpoint identities or pretend those old distributions still exist.
