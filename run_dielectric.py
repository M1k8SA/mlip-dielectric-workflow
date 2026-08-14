#!/usr/bin/env python3
"""End-to-end pw.in + BORN + MLIP dielectric calculation workflow."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from calc_force import (
    build_calculator,
    calculate_forces,
    ev_angstrom_to_ry_bohr,
    phonopy_qe_cell_to_ase,
    write_forces,
)
from relax_mlip import relax_structure


FINAL_DATA = "epsil_ion.dat"
FINAL_LOG = "calc_epsil_parallel.out"


def status(message: str) -> None:
    print(f"[dielectric] {message}", flush=True)


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def require_dependencies() -> None:
    missing: list[str] = []
    for module in ("ase", "deepmd", "numpy", "phonopy"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
        except Exception as exc:
            raise RuntimeError(f"Failed to import {module}: {exc}") from exc
    if missing:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Install phonopy, ase, numpy, and deepmd-kit in the same environment."
        )


def find_phonopy_init() -> str:
    command = shutil.which("phonopy-init")
    if command:
        return command
    command = shutil.which("phonopy")
    if command:
        status("phonopy-init was not found; using the legacy phonopy setup command.")
        return command
    raise RuntimeError(
        "Neither 'phonopy-init' nor the legacy 'phonopy' command is on PATH."
    )


def find_epsil_script(explicit: str | None, base: Path) -> Path:
    if explicit:
        return require_file(Path(explicit), "dielectric post-processing script")

    candidates = [
        base / "calc_epsil_parallel.py",
        Path(__file__).resolve().parent / "calc_epsil_parallel.py",
        base.parent / "calc_epsil_parallel.py",
        base.parent.parent / "calc_epsil_parallel.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "calc_epsil_parallel.py was not found. Searched:\n  " + searched
    )


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare_work_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    # Delete only files produced by this workflow, never arbitrary user files.
    generated_names = {
        "BORN",
        "FORCE_SETS",
        "FORCE_CONSTANTS",
        "force_constants.hdf5",
        "phonopy_disp.yaml",
        "phonopy_params.yaml",
        "phonopy.yaml",
        FINAL_DATA,
        FINAL_LOG,
    }
    for child in path.iterdir():
        is_generated_supercell = (
            child.is_file()
            and child.suffix == ".in"
            and (child.stem == "supercell" or child.stem.startswith("supercell-"))
        )
        if child.is_file() and (child.name in generated_names or is_generated_supercell):
            child.unlink()

    forces_dir = path / "forces"
    if forces_dir.exists():
        if not forces_dir.is_dir():
            raise RuntimeError(f"Expected a directory but found a file: {forces_dir}")
        for child in forces_dir.iterdir():
            if child.is_file() and child.suffix == ".dat":
                child.unlink()
    else:
        forces_dir.mkdir()
    return path


def generate_displacements(
    command: str,
    pw_input: Path,
    work_dir: Path,
    dim: tuple[int, int, int],
    primitive_axes: str,
    amplitude: float | None,
) -> None:
    cmd = [
        command,
        "--qe",
        "-d",
        "--dim",
        *(str(value) for value in dim),
        "--pa",
        primitive_axes,
        "-c",
        str(pw_input),
    ]
    if amplitude is not None:
        cmd.extend(("--amplitude", str(amplitude)))
    status("Generating phonopy displacement supercells: " + " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, check=True)

    yaml_file = work_dir / "phonopy_disp.yaml"
    displaced = sorted(work_dir.glob("supercell-*.in"))
    if not yaml_file.is_file() or not displaced:
        raise RuntimeError(
            "phonopy finished but did not create phonopy_disp.yaml and displaced supercells."
        )
    status(f"Generated {len(displaced)} inequivalent displaced supercells.")


def relax_input_structure(
    pw_input: Path,
    model: Path | None,
    work_dir: Path,
    fmax: float,
    max_steps: int,
    relax_cell: bool,
    optimizer_name: str,
    timeout_seconds: float,
    failure_policy: str,
    reuse_relaxed: bool,
) -> Path:
    relax_dir = work_dir / "relaxation"
    if relax_dir.exists() and not relax_dir.is_dir():
        raise RuntimeError(f"Expected a directory but found a file: {relax_dir}")
    relax_dir.mkdir(exist_ok=True)
    paths = {
        "qe": relax_dir / "relaxed_pw.in",
        "xyz": relax_dir / "relaxed.xyz",
        "trajectory": relax_dir / "BFGS.traj",
        "log": relax_dir / "BFGS.log",
        "summary": relax_dir / "relaxation_summary.json",
    }
    if reuse_relaxed:
        if not paths["qe"].is_file():
            raise FileNotFoundError(
                f"--reuse-relaxed was requested but no checkpoint exists: {paths['qe']}"
            )
        from ase.io import read

        read(paths["qe"], format="espresso-in")
        status(f"Reusing previous relaxed structure: {paths['qe']}")
        return paths["qe"]

    if model is None:
        raise ValueError(
            "A relaxation model is required unless relaxation is skipped or reused."
        )

    for path in paths.values():
        if path.is_file():
            path.unlink()

    status("Relaxing pw.in with MLIP before generating phonopy displacements.")
    try:
        result = relax_structure(
            pw_input,
            model,
            paths["qe"],
            xyz_output=paths["xyz"],
            trajectory=paths["trajectory"],
            logfile=paths["log"],
            fmax=fmax,
            max_steps=max_steps,
            relax_cell=relax_cell,
            optimizer_name=optimizer_name,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError as exc:
        failure = {
            "converged": False,
            "error": str(exc),
            "failure_policy": failure_policy,
            "model": str(model),
        }
        paths["summary"].write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        status(f"WARNING: MLIP relaxation failed: {exc}")
        if failure_policy == "input":
            status("Relaxation failure policy: continuing with the original pw.in.")
            return pw_input
        if failure_policy == "last":
            if not paths["qe"].is_file():
                raise RuntimeError(
                    "Relaxation did not leave a readable last structure; cannot use "
                    "the 'last' failure policy."
                ) from exc
            from ase.io import read

            read(paths["qe"], format="espresso-in")
            status(
                "Relaxation failure policy: continuing with the last valid structure."
            )
            return paths["qe"]
        raise
    result["model"] = str(model)
    paths["summary"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status(
        "Relaxation converged in "
        f"{result['steps']} steps; volume change="
        f"{float(result['volume_change_percent']):+.4f}%."
    )
    status(
        "WARNING: The supplied BORN data will be reused for the relaxed structure. "
        "For quantitatively consistent results, BORN and electronic dielectric "
        "tensors should be recalculated when relaxation changes are not negligible."
    )
    return paths["qe"]


def write_force_sets(path: Path, dataset: dict, forces) -> None:
    import numpy as np

    if "first_atoms" not in dataset:
        raise RuntimeError(
            "Expected phonopy type-1 systematic displacements, but first_atoms is missing."
        )
    entries = dataset["first_atoms"]
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[0] != len(entries) or forces.shape[2] != 3:
        raise RuntimeError(
            f"Force array shape {forces.shape} is inconsistent with {len(entries)} displacements."
        )

    lines = [f"{forces.shape[1]}\n", f"{forces.shape[0]}\n", "\n"]
    for entry, force_table in zip(entries, forces, strict=True):
        displacement = np.asarray(entry["displacement"], dtype=float)
        if displacement.shape != (3,):
            raise RuntimeError(f"Invalid displacement vector shape: {displacement.shape}")
        lines.append(f"{int(entry['number']) + 1}\n")
        lines.append("%20.14f %20.14f %20.14f\n" % tuple(displacement))
        lines.extend(
            "%20.14f %20.14f %20.14f\n" % tuple(row) for row in force_table
        )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def calculate_displacement_forces(
    phonon,
    model: Path,
    work_dir: Path,
    subtract_reference: bool,
):
    import numpy as np

    displaced_cells = phonon.supercells_with_displacements
    if displaced_cells is None or not displaced_cells:
        raise RuntimeError("No displaced cells were loaded from phonopy_disp.yaml.")

    dataset = phonon.dataset
    if dataset is None or "first_atoms" not in dataset:
        raise RuntimeError("No type-1 displacement dataset was loaded from phonopy.")
    if len(dataset["first_atoms"]) != len(displaced_cells):
        raise RuntimeError(
            "phonopy displacement metadata and generated supercell counts do not match."
        )

    status(f"Loading MLIP model: {model}")
    calculator = build_calculator(model)

    reference = np.zeros((len(phonon.supercell), 3), dtype=float)
    if subtract_reference:
        status("Calculating the perfect-supercell residual force for subtraction.")
        reference_atoms = phonopy_qe_cell_to_ase(phonon.supercell)
        reference = calculate_forces(reference_atoms, calculator)
        write_forces(work_dir / "forces" / "reference_eV_A.dat", reference)
        status(
            "Maximum perfect-supercell residual force: "
            f"{np.linalg.norm(reference, axis=1).max():.6f} eV/Angstrom"
        )

    corrected_ev_a = []
    for index, cell in enumerate(displaced_cells, start=1):
        status(f"Calculating MLIP forces for displacement {index}/{len(displaced_cells)}.")
        atoms = phonopy_qe_cell_to_ase(cell)
        raw = calculate_forces(atoms, calculator)
        if raw.shape != reference.shape:
            raise RuntimeError(
                f"Displacement {index} has force shape {raw.shape}; expected {reference.shape}."
            )
        corrected = raw - reference
        # Match phonopy's QE force parser: remove the translational force drift.
        corrected = corrected - corrected.mean(axis=0)
        corrected_ev_a.append(corrected)
        write_forces(
            work_dir / "forces" / f"disp-{index:03d}_eV_A.dat", corrected
        )

    corrected_ev_a = np.asarray(corrected_ev_a, dtype=float)
    corrected_ry_bohr = ev_angstrom_to_ry_bohr(corrected_ev_a)
    for index, force_table in enumerate(corrected_ry_bohr, start=1):
        write_forces(
            work_dir / "forces" / f"disp-{index:03d}_Ry_Bohr.dat", force_table
        )
    return corrected_ry_bohr


def validate_born(path: Path, primitive_natom: int) -> None:
    import numpy as np

    try:
        values = np.loadtxt(path, skiprows=1, ndmin=2)
    except Exception as exc:
        raise ValueError(f"Failed to parse BORN file {path}: {exc}") from exc
    expected = (primitive_natom + 1, 9)
    if values.shape != expected:
        raise ValueError(
            f"BORN numeric data has shape {values.shape}; expected {expected} "
            "(one dielectric tensor plus one Born-charge tensor per primitive atom)."
        )
    charge_sum = values[1:].reshape(-1, 3, 3).sum(axis=0)
    max_sum = float(np.abs(charge_sum).max())
    if max_sum > 0.1:
        status(
            "WARNING: Born-charge acoustic sum is relatively large "
            f"(max |sum Z*| = {max_sum:.4f})."
        )


def validate_force_sets(work_dir: Path) -> None:
    import numpy as np
    import phonopy

    status("Validating FORCE_SETS and building force constants with phonopy.")
    with working_directory(work_dir):
        checked = phonopy.load(
            "phonopy_disp.yaml", force_sets_filename="FORCE_SETS"
        )
        checked.run_mesh([1, 1, 1], with_eigenvectors=False)
        mesh = getattr(checked, "mesh", None)
        if mesh is not None and hasattr(mesh, "frequencies"):
            frequencies = np.asarray(mesh.frequencies[0])
        else:
            frequencies = np.asarray(checked.get_mesh_dict()["frequencies"][0])
    if not np.isfinite(frequencies).all():
        raise RuntimeError("phonopy produced NaN or infinite Gamma-point frequencies.")
    status(
        "Gamma-point frequency range: "
        f"{frequencies.min():.6f} to {frequencies.max():.6f} THz"
    )
    if frequencies.min() < -0.1:
        status(
            "WARNING: Significant imaginary Gamma-point modes were found; "
            "inspect the structure, MLIP applicability, and supercell convergence."
        )


def run_dielectric_postprocess(
    script: Path,
    work_dir: Path,
    wavelength: str,
    relaxation: float,
    cores: int,
) -> tuple[Path, Path]:
    log_path = work_dir / FINAL_LOG
    cmd = [
        sys.executable,
        str(script),
        "-c",
        "BORN",
        "-l",
        wavelength,
        "-t",
        str(relaxation),
        "-n",
        str(cores),
    ]
    status("Running dielectric post-processing; full output goes to calc_epsil_parallel.out.")
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"calc_epsil_parallel.py failed with exit code {result.returncode}. "
            f"Last output lines:\n{tail}"
        )

    data_path = work_dir / FINAL_DATA
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise RuntimeError("Post-processing succeeded but epsil_ion.dat was not created.")
    return data_path, log_path


def validate_dielectric_data(path: Path, wavelength: str) -> None:
    import numpy as np

    data = np.loadtxt(path, ndmin=2)
    expected_rows = int(wavelength.split(",")[2])
    if data.shape != (expected_rows, 19):
        raise RuntimeError(
            f"{path.name} has shape {data.shape}; expected {(expected_rows, 19)}."
        )
    if not np.isfinite(data).all():
        raise RuntimeError(f"{path.name} contains NaN or infinite values.")


def publish_output(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def parse_wavelength(value: str) -> str:
    try:
        wmin_text, wmax_text, count_text = value.split(",")
        wmin, wmax, count = float(wmin_text), float(wmax_text), int(count_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use wmin,wmax,number, e.g. 2,16,100.") from exc
    if wmin <= 0 or wmax <= wmin or count < 2:
        raise argparse.ArgumentTypeError(
            "Wavelengths must satisfy 0 < wmin < wmax and number >= 2."
        )
    return f"{wmin:g},{wmax:g},{count}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate phonopy displacements from pw.in, calculate forces with a "
            "DeepMD MLIP, create FORCE_SETS, and calculate dielectric response."
        )
    )
    parser.add_argument("--input", default="pw.in", help="QE pw.x structure input.")
    parser.add_argument("--born", default="BORN", help="Existing phonopy BORN file.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Shared MLIP model for relaxation and displaced-supercell forces. "
            "There is no built-in default; specialized model options override it."
        ),
    )
    parser.add_argument(
        "--relax-model",
        default=None,
        help="MLIP model used only for relaxation; defaults to --model.",
    )
    parser.add_argument(
        "--force-model",
        default=None,
        help=(
            "MLIP model used for perfect/displaced-supercell forces and "
            "FORCE_SETS; defaults to --model."
        ),
    )
    parser.add_argument(
        "--skip-relax",
        action="store_true",
        help="Use pw.in directly without the workflow's MLIP relaxation stage.",
    )
    parser.add_argument("--relax-fmax", type=float, default=0.005)
    parser.add_argument("--relax-max-steps", type=int, default=500)
    parser.add_argument(
        "--relax-optimizer",
        choices=("bfgs", "lbfgs", "fire"),
        default="bfgs",
    )
    parser.add_argument(
        "--relax-timeout",
        type=float,
        default=0.0,
        help="Relaxation wall-time limit in seconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--on-relax-failure",
        choices=("stop", "input", "last"),
        default="stop",
        help="On relaxation failure: stop safely, use original input, or use last structure.",
    )
    parser.add_argument(
        "--reuse-relaxed",
        action="store_true",
        help="Reuse dielectric_work/relaxation/relaxed_pw.in without rerunning relaxation.",
    )
    parser.add_argument(
        "--relax-fixed-cell",
        action="store_true",
        help="Relax atomic positions but keep the pw.in lattice fixed.",
    )
    parser.add_argument("--dim", nargs=3, type=int, default=(2, 2, 2), metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--primitive-axes",
        default="P",
        help=(
            "phonopy primitive axes passed to --pa (default: P, preserving "
            "the pw.in cell and BORN atom order)."
        ),
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Displacement amplitude in Bohr; default lets phonopy choose for QE.",
    )
    parser.add_argument("--wavelength", type=parse_wavelength, default="2,16,100")
    parser.add_argument("--relaxation", type=float, default=0.002)
    parser.add_argument("--cores", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--work-dir", default="dielectric_work")
    parser.add_argument("--epsil-script", default=None)
    parser.add_argument(
        "--no-subtract-reference",
        action="store_true",
        help="Do not subtract perfect-supercell MLIP residual forces.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model is None and args.force_model is None:
        raise ValueError(
            "No force-constant MLIP model was specified. Use --model MODEL, or "
            "use --force-model MODEL when selecting separate models."
        )
    if (
        not args.skip_relax
        and not args.reuse_relaxed
        and args.model is None
        and args.relax_model is None
    ):
        raise ValueError(
            "No relaxation MLIP model was specified. Use --model MODEL, "
            "--relax-model MODEL, --skip-relax, or --reuse-relaxed."
        )
    if any(value < 1 for value in args.dim):
        raise ValueError("All --dim values must be positive integers.")
    if args.amplitude is not None and args.amplitude <= 0:
        raise ValueError("--amplitude must be positive.")
    if args.relax_fmax <= 0:
        raise ValueError("--relax-fmax must be positive.")
    if args.relax_max_steps < 1:
        raise ValueError("--relax-max-steps must be at least 1.")
    if args.relax_timeout < 0:
        raise ValueError("--relax-timeout must be non-negative.")
    if args.relaxation < 0:
        raise ValueError("--relaxation must be non-negative.")
    if args.cores < 1:
        raise ValueError("--cores must be at least 1.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    require_dependencies()

    base = Path.cwd().resolve()
    pw_input = require_file(Path(args.input), "pw.in")
    born = require_file(Path(args.born), "BORN file")
    force_model_arg = args.force_model or args.model
    if force_model_arg is None:
        raise ValueError(
            "No force-constant MLIP model was specified. Use --model MODEL, or "
            "use --force-model MODEL when selecting separate models."
        )
    force_model = require_file(Path(force_model_arg), "force-constant MLIP model")

    relax_model_arg = args.relax_model or args.model
    relax_model = (
        Path(relax_model_arg).expanduser().resolve()
        if relax_model_arg is not None
        else None
    )
    if not args.skip_relax and not args.reuse_relaxed:
        if relax_model is None:
            raise ValueError(
                "No relaxation MLIP model was specified. Use --model MODEL, "
                "--relax-model MODEL, --skip-relax, or --reuse-relaxed."
            )
        relax_model = require_file(relax_model, "relaxation MLIP model")
    epsil_script = find_epsil_script(args.epsil_script, base)
    phonopy_init = find_phonopy_init()
    work_dir_path = Path(args.work_dir).expanduser().resolve()
    if work_dir_path == base:
        raise ValueError("--work-dir must not be the input/output directory itself.")
    work_dir = prepare_work_directory(work_dir_path)

    status(f"Input: {pw_input}")
    status(f"Work directory: {work_dir}")
    status(f"Force-constant MLIP model: {force_model}")
    structure_input = pw_input
    if not args.skip_relax:
        if args.reuse_relaxed:
            status(
                "Relaxation model is not loaded because --reuse-relaxed was requested."
            )
        else:
            status(f"Relaxation MLIP model: {relax_model}")
        structure_input = relax_input_structure(
            pw_input,
            relax_model,
            work_dir,
            fmax=args.relax_fmax,
            max_steps=args.relax_max_steps,
            relax_cell=not args.relax_fixed_cell,
            optimizer_name=args.relax_optimizer,
            timeout_seconds=args.relax_timeout,
            failure_policy=args.on_relax_failure,
            reuse_relaxed=args.reuse_relaxed,
        )
    else:
        status("Skipping MLIP relaxation by user request.")
    generate_displacements(
        phonopy_init,
        structure_input,
        work_dir,
        tuple(args.dim),
        args.primitive_axes,
        args.amplitude,
    )

    import phonopy

    with working_directory(work_dir):
        phonon = phonopy.load("phonopy_disp.yaml")
    validate_born(born, len(phonon.primitive))
    shutil.copy2(born, work_dir / "BORN")

    forces = calculate_displacement_forces(
        phonon,
        force_model,
        work_dir,
        subtract_reference=not args.no_subtract_reference,
    )
    write_force_sets(work_dir / "FORCE_SETS", phonon.dataset, forces)
    status(f"Wrote phonopy FORCE_SETS: {work_dir / 'FORCE_SETS'}")

    validate_force_sets(work_dir)
    data_path, log_path = run_dielectric_postprocess(
        epsil_script,
        work_dir,
        args.wavelength,
        args.relaxation,
        args.cores,
    )
    validate_dielectric_data(data_path, args.wavelength)

    final_data = base / FINAL_DATA
    final_log = base / FINAL_LOG
    publish_output(data_path, final_data)
    publish_output(log_path, final_log)
    status(f"DONE: {final_data}")
    status(f"DONE: {final_log}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[dielectric] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
