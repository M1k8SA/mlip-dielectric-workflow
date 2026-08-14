#!/usr/bin/env python3
"""Calculate MLIP forces used by the QE/phonopy dielectric workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms


def build_calculator(model: str | Path):
    """Create one DeepMD calculator that can be reused for all supercells."""
    from deepmd.calculator import DP

    return DP(model=str(Path(model).resolve()))


def phonopy_qe_cell_to_ase(cell: "PhonopyAtoms") -> "Atoms":
    """Convert a phonopy QE cell (Bohr internally) to an ASE cell (Angstrom)."""
    import numpy as np
    from ase import Atoms
    from ase.units import Bohr

    return Atoms(
        symbols=cell.symbols,
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        cell=np.asarray(cell.cell, dtype=float) * Bohr,
        pbc=True,
    )


def calculate_forces(atoms: "Atoms", calculator):
    """Return MLIP forces in ASE's native eV/Angstrom unit."""
    import numpy as np

    atoms.calc = calculator
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3):
        raise RuntimeError(
            f"MLIP returned forces with shape {forces.shape}; expected {(len(atoms), 3)}."
        )
    if not np.isfinite(forces).all():
        raise RuntimeError("MLIP returned NaN or infinite force values.")
    return forces


def ev_angstrom_to_ry_bohr(forces):
    """Convert force from eV/Angstrom to QE phonopy units, Ry/Bohr."""
    import numpy as np
    from ase.units import Bohr, Ry

    return np.asarray(forces, dtype=float) * Bohr / Ry


def write_forces(path: str | Path, forces) -> None:
    """Write an N x 3 force table."""
    import numpy as np

    np.savetxt(path, forces, fmt="%20.10f")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the MLIP forces of one complete QE pw.x input."
    )
    parser.add_argument("structure", nargs="?", default="pw.in")
    parser.add_argument(
        "--model",
        required=True,
        help="DeepMD-compatible MLIP model file; no model is selected by default.",
    )
    parser.add_argument("--output", default="forces.dat")
    parser.add_argument(
        "--unit",
        choices=("ry/bohr", "ev/angstrom"),
        default="ry/bohr",
        help="Output force unit (default: ry/bohr, as required by phonopy QE mode).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from ase.io import read

    atoms = read(args.structure, format="espresso-in")
    atoms.pbc = True
    calculator = build_calculator(args.model)
    forces = calculate_forces(atoms, calculator)
    if args.unit == "ry/bohr":
        forces = ev_angstrom_to_ry_bohr(forces)
    write_forces(args.output, forces)
    print(f"Wrote {len(atoms)} atomic forces to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
