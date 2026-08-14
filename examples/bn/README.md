# BN example inputs

`pw.in` and `BORN` are a small four-atom BN example for exercising the workflow.

No MLIP model or pseudopotential file is included. Select a DeepMD-compatible model that supports both B and N, then run from a copied case directory:

```bash
cp -r examples/bn demo_case
cd demo_case
../run_dielectric.sh --model /absolute/path/to/model --relax-fixed-cell
```

These files demonstrate input formatting only. For quantitative work, verify that the BORN tensors, structure, atom order, MLIP applicability, supercell size, and displacement amplitude are mutually consistent.
