#!/usr/bin/env python3
"""Relax a QE structure with a user-selected DeepMD MLIP model."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


def read_pseudopotential_labels(pw_input: str | Path) -> dict[str, str]:
    """Extract element -> UPF filename labels from the ATOMIC_SPECIES card.

    The UPF files are not opened by MLIP relaxation or by phonopy. ASE only
    needs these strings when serializing an espresso-in file.
    """
    from ase.io.espresso import label_to_symbol

    path = Path(pw_input)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text_without_comments = "\n".join(line.split("!", 1)[0] for line in lines)
    match = re.search(r"\bntyp\s*=\s*(\d+)", text_without_comments, re.IGNORECASE)
    if match is None:
        raise ValueError(f"ntyp was not found in QE input: {path}")
    ntyp = int(match.group(1))

    card_index = None
    for index, raw in enumerate(lines):
        clean = raw.split("!", 1)[0].strip()
        if clean and clean.split()[0].upper() == "ATOMIC_SPECIES":
            card_index = index
            break
    if card_index is None:
        raise ValueError(f"ATOMIC_SPECIES was not found in QE input: {path}")

    records: list[tuple[str, str, str]] = []
    for raw in lines[card_index + 1 :]:
        clean = raw.split("!", 1)[0].strip()
        if not clean:
            continue
        fields = clean.split()
        if len(fields) < 3:
            raise ValueError(f"Malformed ATOMIC_SPECIES line in {path}: {raw}")
        records.append((fields[0], fields[1], fields[2]))
        if len(records) == ntyp:
            break
    if len(records) != ntyp:
        raise ValueError(
            f"ATOMIC_SPECIES in {path} contains {len(records)} entries; expected {ntyp}."
        )

    pseudopotentials: dict[str, str] = {}
    for label, _mass, filename in records:
        symbol = label_to_symbol(label)
        previous = pseudopotentials.get(symbol)
        if previous is not None and previous != filename:
            print(
                "[relax] WARNING: multiple QE species labels map to element "
                f"{symbol}; using {previous!r} in the structure-only relaxed file."
            )
            continue
        pseudopotentials[symbol] = filename
    return pseudopotentials


def relax_structure(
    pw_input: str | Path,
    model: str | Path,
    output: str | Path,
    *,
    xyz_output: str | Path | None = None,
    trajectory: str | Path | None = None,
    logfile: str | Path | None = None,
    fmax: float = 0.005,
    max_steps: int = 500,
    relax_cell: bool = True,
    optimizer_name: str = "bfgs",
    timeout_seconds: float = 0.0,
) -> dict[str, float | int | bool | str]:
    """Run BFGS relaxation and return convergence diagnostics."""
    import numpy as np
    from ase.filters import ExpCellFilter
    from ase.io import read, write
    from ase.optimize import BFGS
    from ase.optimize.fire import FIRE
    from ase.optimize.lbfgs import LBFGS
    from ase.units import GPa
    from deepmd.calculator import DP

    pw_input = Path(pw_input).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pw_input.is_file():
        raise FileNotFoundError(f"QE input not found: {pw_input}")
    if not model.is_file():
        raise FileNotFoundError(f"MLIP model not found: {model}")
    if fmax <= 0:
        raise ValueError("fmax must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    optimizer_name = optimizer_name.lower()
    optimizer_classes = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}
    if optimizer_name not in optimizer_classes:
        raise ValueError("optimizer_name must be one of: bfgs, lbfgs, fire.")
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative.")

    pseudopotentials = read_pseudopotential_labels(pw_input)
    atoms = read(pw_input, format="espresso-in")
    atoms.pbc = True
    missing = sorted(set(atoms.get_chemical_symbols()) - set(pseudopotentials))
    if missing:
        raise ValueError(
            "No ATOMIC_SPECIES pseudopotential label was found for: "
            + ", ".join(missing)
        )

    initial_cell = np.asarray(atoms.cell.array, dtype=float).copy()
    initial_scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    initial_volume = float(atoms.get_volume())

    print(f"[relax] Input: {pw_input}", flush=True)
    print(f"[relax] Model: {model}", flush=True)
    print(
        f"[relax] {optimizer_name.upper()}: fmax={fmax:g} eV/Angstrom, "
        f"max_steps={max_steps}, relax_cell={relax_cell}, "
        f"timeout={timeout_seconds:g}s",
        flush=True,
    )
    print(
        "[relax] UPF names are copied as QE output labels only; no UPF file is read.",
        flush=True,
    )

    atoms.calc = DP(model=str(model))
    target = ExpCellFilter(atoms) if relax_cell else atoms
    optimizer = optimizer_classes[optimizer_name](
        target,
        trajectory=str(Path(trajectory).resolve()) if trajectory else None,
        logfile=str(Path(logfile).resolve()) if logfile else "-",
    )
    start_time = time.monotonic()

    def enforce_timeout() -> None:
        if timeout_seconds > 0 and time.monotonic() - start_time >= timeout_seconds:
            raise TimeoutError(
                f"Relaxation exceeded the {timeout_seconds:g} second time limit."
            )

    if timeout_seconds > 0:
        optimizer.attach(enforce_timeout, interval=1)

    run_error: Exception | None = None
    try:
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    except TimeoutError as exc:
        # Optimizer observers run between steps, so the current atoms are a valid
        # checkpoint and are written below before reporting failure.
        converged = False
        run_error = exc
    elapsed_seconds = time.monotonic() - start_time

    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    final_volume = float(atoms.get_volume())
    volume_change_percent = (final_volume / initial_volume - 1.0) * 100.0
    scaled_delta = atoms.get_scaled_positions(wrap=False) - initial_scaled
    scaled_delta -= np.rint(scaled_delta)
    max_internal_displacement = float(
        np.linalg.norm(scaled_delta @ atoms.cell.array, axis=1).max()
    )

    max_stress_gpa = 0.0
    if relax_cell:
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
        max_stress_gpa = float(np.abs(stress).max() / GPa)

    write(
        output,
        atoms,
        format="espresso-in",
        pseudopotentials=pseudopotentials,
        crystal_coordinates=True,
    )
    if xyz_output is not None:
        xyz_path = Path(xyz_output).expanduser().resolve()
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        write(xyz_path, atoms)

    # Guard against an invalid variable-cell result before phonopy consumes it.
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise RuntimeError("Relaxation produced NaN or infinite coordinates/cell values.")
    if final_volume <= 0 or abs(np.linalg.det(atoms.cell.array)) < 1e-8:
        raise RuntimeError("Relaxation produced a singular or non-positive-volume cell.")

    result: dict[str, float | int | bool | str] = {
        "converged": converged,
        "optimizer": optimizer_name,
        "steps": int(optimizer.nsteps),
        "elapsed_seconds": elapsed_seconds,
        "max_force_ev_a": max_force,
        "max_stress_gpa": max_stress_gpa,
        "initial_volume_a3": initial_volume,
        "final_volume_a3": final_volume,
        "volume_change_percent": volume_change_percent,
        "max_internal_displacement_a": max_internal_displacement,
    }
    print(
        f"[relax] Finished: converged={converged}, steps={optimizer.nsteps}, "
        f"elapsed={elapsed_seconds:.2f}s, "
        f"max atomic force={max_force:.6f} eV/Angstrom",
        flush=True,
    )
    if relax_cell:
        print(
            f"[relax] Volume change={volume_change_percent:+.4f}%, "
            f"max |stress|={max_stress_gpa:.4f} GPa",
            flush=True,
        )
    print(f"[relax] Wrote: {output}", flush=True)

    if run_error is not None:
        raise RuntimeError(f"{run_error} Last structure: {output}") from run_error
    if not converged:
        raise RuntimeError(
            f"{optimizer_name.upper()} did not converge within {max_steps} steps. "
            f"Last structure: {output}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="pw.in")
    parser.add_argument(
        "--model",
        required=True,
        help="DeepMD-compatible MLIP model file; no model is selected by default.",
    )
    parser.add_argument("--output", default="relaxed_pw.in")
    parser.add_argument("--xyz-output", default="relaxed.xyz")
    parser.add_argument("--trajectory", default="BFGS.traj")
    parser.add_argument("--logfile", default="BFGS.log")
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--optimizer", choices=("bfgs", "lbfgs", "fire"), default="bfgs"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Wall-time limit in seconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--fixed-cell",
        action="store_true",
        help="Relax atomic positions only and keep the lattice fixed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    relax_structure(
        args.input,
        args.model,
        args.output,
        xyz_output=args.xyz_output,
        trajectory=args.trajectory,
        logfile=args.logfile,
        fmax=args.fmax,
        max_steps=args.max_steps,
        relax_cell=not args.fixed_cell,
        optimizer_name=args.optimizer,
        timeout_seconds=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
