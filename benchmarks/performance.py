#!/usr/bin/env python3
"""Reproducible fixed-N Oncostreams benchmarking and Python profiling.

From the repository root, after installing the package with the Numba extra::

    python benchmarks/performance.py benchmark
    python benchmarks/performance.py profile --backend numpy
    python benchmarks/performance.py profile --backend numba

Benchmark runs use separate Python processes for the two backends. They time
only Culture.simulate, following an unmeasured warmup, with no file output or
checkpoints. This compares the CURRENT reference (NumPy) and accelerated
(Numba) implementations, not a historical Git revision.

The profile mode saves cProfile stats from a separately instrumented run.
Numba compiled kernels are opaque to Python line-by-line profiling.
Outputs default to ``benchmarks/results/`` inside the Git repository.
Commit only selected reference summaries and readable profiles, not every run.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import math
import os
from pathlib import Path
from datetime import datetime
import platform
import pstats
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO / "benchmarks" / "results"

# Allow a clone to run the benchmark without relying on an editable install.
# Package dependencies (including optional Numba) must still be installed.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def optional_float(raw: str) -> float | None:
    """
    Convert a string to a float, or None if the string is "none" or "off".
    """
    if raw.lower() in ("none", "off"):
        return None
    return float(raw)


def add_physical_options(
    parser: argparse.ArgumentParser, *, include_repeats: bool = True
) -> None:
    """
    Define the physical and measurement options for the benchmark and
    profile commands.
    """
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    parser.add_argument("--delta-t", type=float, default=0.005)
    parser.add_argument("--deformation-period", type=float, default=0.05)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--initial-fraction", type=float, default=0.5)
    parser.add_argument("--eta", type=optional_float, default=0.033)
    parser.add_argument("--d-phi", type=optional_float, default=None)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--steps", type=int, default=150)
    if include_repeats:
        parser.add_argument(
            "--repeats", type=int, default=3,
            help="Independent timing repetitions, with identical seed (default: 3)",
        )
    else:
        parser.set_defaults(repeats=1)


def config_from_args(args: argparse.Namespace) -> dict:
    """
    Convert the command-line arguments to a configuration dictionary.
    """
    return {
        "n": args.n,
        "rho": args.rho,
        "seed": args.seed,
        "dimension": args.dimension,
        "delta_t": args.delta_t,
        "deformation_period": args.deformation_period,
        "attempts": args.attempts,
        "initial_fraction": args.initial_fraction,
        "eta": args.eta,
        "d_phi": args.d_phi,
        "warmup": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "aspect_ratio_max": 2.7,
        "cell_radius": 1.0,
        "force": {"name": "Anisotropic_Grosmann", "kRep": 2.5,
                  "bExp": 3.0, "lambda_core": 1.0, "shrinking": True},
        "overlap_threshold_ratio": "exp(-1)",
        "trabajo_final": True,
        "fixed_n": True,
        "movement": True,
        "periodic": True,
        "outputs": False,
        "checkpoints": False,
    }


def validate(args: argparse.Namespace) -> None:
    """
    Validate the command-line arguments for physical and measurement parameters.
    """
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    if not math.isfinite(args.rho) or not (0 < args.rho <= 1):
        raise ValueError("--rho must be in (0, 1]")
    if not math.isfinite(args.delta_t) or args.delta_t <= 0:
        raise ValueError("--delta-t must be positive and finite")
    ratio = args.deformation_period / args.delta_t
    if not math.isfinite(ratio) or round(ratio) < 1 or not math.isclose(
        ratio, round(ratio), rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError(
            "--deformation-period must be a positive integer multiple "
            "of --delta-t"
        )
    if args.attempts < 0 or args.warmup < 1 or args.steps < 1 or args.repeats < 1:
        raise ValueError("attempts>=0; warmup, steps and repeats must be positive")
    if args.warmup < round(ratio):
        raise ValueError(
            "--warmup must span at least one deformation interval, "
            "so the first scheduled deformation runs before timing"
        )
    # A periodic grid must contain at least three buckets along each axis.
    # See SpatialHashGrid's explicit validation of this geometric constraint.
    side = math.sqrt(args.n * math.pi / args.rho)
    cutoff_bucket_size = 2.0 * math.sqrt(2.7)
    if math.floor(side / cutoff_bucket_size) < 3:
        raise ValueError(
            "Box too small for the periodic spatial grid: increase --n "
            "or adjust --rho"
        )
    if not math.isfinite(args.initial_fraction) or not 0 <= args.initial_fraction <= 1:
        raise ValueError("--initial-fraction must be in [0, 1]")
    if any(x is not None and (not math.isfinite(x) or x < 0)
           for x in (args.eta, args.d_phi)):
        raise ValueError("noise strengths must be finite and nonnegative or 'none'")


def git_metadata() -> dict:
    """
    Return the current Git commit hash and whether the working tree is dirty.
    """
    def git(*params: str) -> str:
        """
        Run a Git command in the repository and return its stdout, or "unknown"
        if the command fails.
        """
        run = subprocess.run(
            ["git", "-C", str(REPO), *params], capture_output=True,
            text=True, check=False,
        )
        return run.stdout.strip() if run.returncode == 0 else "unknown"
    return {
        "commit": git("rev-parse", "HEAD"),
        # Ignore benchmark outputs when inspecting source-code cleanliness:
        # otherwise the current run itself makes the working tree "dirty".
        "dirty": bool(git(
            "status", "--porcelain", "--untracked-files=all", "--",
            ".", ":(exclude)benchmarks/results",
        )),
    }


def package_version(name: str) -> str | None:
    """
    Return the installed version of a package, or None if it is not installed.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment() -> dict:
    """
    Return the environment information.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": package_version("numba"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def sha_state(culture) -> str:
    """
    Compute a SHA-256 fingerprint of selected culture state variables.
    """
    hash_obj = hashlib.sha256()
    arrays = (
        culture.cell_positions,
        culture.cell_phies,
        np.fromiter((cell.aspect_ratio for cell in culture.cells),
                    dtype=np.float64, count=len(culture.cells)),
    )
    for array in arrays:
        hash_obj.update(np.asarray(array, dtype=np.float64).tobytes())
    # Positions alone do not identify a reproducible initial microstate:
    # the future trajectory also depends on the RNG state.
    hash_obj.update(repr(culture.rng.bit_generator.state).encode("utf-8"))
    return hash_obj.hexdigest()


def make_culture(args: argparse.Namespace, backend: str):
    """
    Create a Culture instance with the specified backend and parameters.
    """
    # Import the production API, not the tests' fixtures or benchmark probes.
    from tumorsphere.core.culture import Culture
    from tumorsphere.core.forces import Anisotropic_Grosmann
    from tumorsphere.core.output import OutputDemux
    from tumorsphere.core.spatial_hash_grid import SpatialHashGrid

    threshold = math.exp(-1)
    aspect_max = 2.7
    side = math.sqrt(args.n * math.pi / args.rho)
    grid = SpatialHashGrid(
        culture=None,
        bounds=side,
        cube_size=2 * math.sqrt(aspect_max * (-math.log(threshold))),
        torus=True,
        spatial_dimension=args.dimension,
    )
    force = Anisotropic_Grosmann(
        kRep=2.5, bExp=3.0,
        noise_eta=args.eta, d_phi=args.d_phi,
        shrinking=True, lambda_core=1.0,
    )
    return Culture(
        output=OutputDemux("performance", []),
        force=force, grid=grid, backend=backend,
        rng_seed=args.seed,
        initial_number_of_cells=args.n,
        initial_fraction_elongated=args.initial_fraction,
        reproduction=False, movement=True, deformation=True,
        stabilization_time=0,
        overlap_threshold_ratio=threshold,
        delta_t=args.delta_t,
        deformation_attempt_period=args.deformation_period,
        initial_aspect_ratio=1.0,
        aspect_ratio_max=aspect_max,
        cell_speed_max=1.0,
        trabajo_final=True,
        cell_max_def_attempts=args.attempts,
    )


def initial_and_warm(culture, warmup: int) -> tuple[str, str, float]:
    """
    Initialize the culture, compute its initial SHA-256 hash, perform a warmup
    simulation, and compute the warmed SHA-256 hash and the fraction of elongated
    cells after the warmup. 
    Return a tuple of (initial_hash, warmed_hash, fraction_elongated).
    """
    culture.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    starting_hash = sha_state(culture)
    culture.simulate(num_times=warmup, start_tic=0, checkpoint_path=None)
    warmed_hash = sha_state(culture)
    elongated = sum(not cell.is_round for cell in culture.cells)
    return starting_hash, warmed_hash, elongated / len(culture.cells)


def benchmark_worker(args: argparse.Namespace) -> dict:
    """
    Run the benchmark for the specified backend and return a dictionary of results.
    Time repeated runs; rebuild the same seeded culture for each repeat.
    """
    measures = []
    hashes = []
    warm_hashes = []
    warm_fractions = []
    for repeat in range(args.repeats):
        culture = make_culture(args, args.backend)
        initial_hash, warmed_hash, fraction = initial_and_warm(culture, args.warmup)
        hashes.append(initial_hash)
        warm_hashes.append(warmed_hash)
        warm_fractions.append(fraction)
        start = time.perf_counter()
        culture.simulate(
            num_times=args.warmup + args.steps,
            start_tic=args.warmup,
            checkpoint_path=None,
        )
        elapsed = time.perf_counter() - start
        measures.append(elapsed * 1000 / args.steps)
        print(f"{args.backend}, repetition {repeat + 1}/{args.repeats}: "
              f"{measures[-1]:.3f} ms/step", flush=True)
    if len(set(hashes)) != 1:
        raise RuntimeError("Inconsistent initial states across repetitions")
    if len(set(warm_hashes)) != 1:
        raise RuntimeError("Inconsistent warmed states across repetitions")
    return {
        "mode": "benchmark",
        "backend": args.backend,
        "git": git_metadata(),
        "config": config_from_args(args),
        "environment": environment(),
        "initial_microstate_sha256": hashes[0],
        "warmed_microstate_sha256": warm_hashes,
        "fraction_elongated_after_warmup": warm_fractions,
        "ms_per_step": measures,
        "median_ms_per_step": statistics.median(measures),
        "min_ms_per_step": min(measures),
        "max_ms_per_step": max(measures),
        "stdev_ms_per_step": (
            statistics.stdev(measures) if len(measures) > 1 else None
        ),
        "measured_at": datetime.now().astimezone().isoformat(),
    }


def profile_worker(args: argparse.Namespace) -> dict:
    """
    Run the simulation with cProfile instrumentation and return a dictionary 
    of results.
    """
    culture = make_culture(args, args.backend)
    initial_hash, warmed_hash, fraction = initial_and_warm(culture, args.warmup)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    try:
        culture.simulate(
            num_times=args.warmup + args.steps,
            start_tic=args.warmup,
            checkpoint_path=None,
        )
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - start
    path = Path(args.profile_stats).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(path))
    output = io.StringIO()
    output.write("cProfile: cumulative Python time\n")
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(35)
    output.write("\ncProfile: Python self time\n")
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats(
        "tottime"
    ).print_stats(35)
    output.write(
        "\nImportant: Numba nopython kernels are opaque here; "
        "the profiler does not attribute time to lines inside compiled kernels.\n"
        "This instrumented runtime must not be used as the benchmark timing.\n"
    )
    text_path = path.with_suffix(".txt")
    text_path.write_text(output.getvalue(), encoding="utf-8")
    print(f"{args.backend}: cProfile saved at {path}")
    print(f"{args.backend}: readable statistics at {text_path}")
    return {
        "mode": "profile",
        "backend": args.backend,
        "git": git_metadata(),
        "config": config_from_args(args),
        "environment": environment(),
        "initial_microstate_sha256": initial_hash,
        "warmed_microstate_sha256": warmed_hash,
        "fraction_elongated_after_warmup": fraction,
        "profile_seconds_instrumented": elapsed,
        "cprofile": str(path),
        "text": str(text_path),
        "measured_at": datetime.now().astimezone().isoformat(),
    }


def worker(args: argparse.Namespace) -> None:
    """
    Internal entry point for the child process that performs the benchmark or
    profile run. This function is not exposed to the public command-line interface.
    """
    validate(args)
    if args.mode == "benchmark":
        record = benchmark_worker(args)
    else:
        record = profile_worker(args)
    path = Path(args.out_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metadata: {path}")


def forward_args(args: argparse.Namespace) -> list[str]:
    """
    Convert the relevant command-line arguments to a list of strings for forwarding
    to the child process. Each argument is represented by its flag and value.
    """
    options = []
    mapping = (
        ("n", "--n"), ("rho", "--rho"), ("seed", "--seed"),
        ("dimension", "--dimension"), ("delta_t", "--delta-t"),
        ("deformation_period", "--deformation-period"),
        ("attempts", "--attempts"),
        ("initial_fraction", "--initial-fraction"),
        ("eta", "--eta"), ("d_phi", "--d-phi"),
        ("warmup", "--warmup"), ("steps", "--steps"),
        ("repeats", "--repeats"),
    )
    for attr, flag in mapping:
        value = getattr(args, attr)
        options.extend((flag, "none" if value is None else str(value)))
    return options


def run_child(args: argparse.Namespace, backend: str, mode: str,
              out_file: Path, profile_stats: Path | None = None) -> dict:
    """
    Run a child process with the specified backend and mode, forwarding the
    relevant command-line arguments. The child process will write its output to
    the specified out_file, and optionally write cProfile stats to profile_stats.
    """
    command = [
        sys.executable, str(Path(__file__).resolve()), "_worker",
        "--mode", mode, "--backend", backend,
        "--out-file", str(out_file), *forward_args(args),
    ]
    if profile_stats is not None:
        command.extend(("--profile-stats", str(profile_stats)))
    subprocess.run(command, cwd=REPO, check=True)
    return json.loads(out_file.read_text(encoding="utf-8"))


def result_directory(args: argparse.Namespace, mode: str) -> Path:
    """Create a unique run directory without overwriting earlier results."""
    config_bytes = json.dumps(
        config_from_args(args), sort_keys=True
    ).encode("utf-8")
    case_id = hashlib.sha256(config_bytes).hexdigest()[:10]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    directory = (
        args.output_dir.expanduser().resolve()
        / f"{mode}-{timestamp}-{os.getpid()}-{case_id}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def benchmark(args: argparse.Namespace) -> None:
    """
    Run the benchmark for both NumPy and Numba backends, compare their performance,
    and save the results to a summary JSON file. Validate that the initial microstates
    and configurations match across backends to ensure a fair comparison.
    """
    validate(args)
    directory = result_directory(args, "benchmark")
    records = {}
    for backend in ("numpy", "numba"):
        records[backend] = run_child(
            args, backend, "benchmark", directory / f"{backend}.json"
        )
    a, b = records["numpy"], records["numba"]
    if a["initial_microstate_sha256"] != b["initial_microstate_sha256"]:
        raise RuntimeError(
            "Initial microstates do not match; speedup is not comparable"
        )
    if a["config"] != b["config"]:
        raise RuntimeError("Physical or measurement configuration mismatch")
    if a["git"]["commit"] != b["git"]["commit"]:
        raise RuntimeError("Backends were benchmarked on different commits")
    if a["environment"] != b["environment"]:
        raise RuntimeError("Backends ran in different reported environments")
    speedup = a["median_ms_per_step"] / b["median_ms_per_step"]
    summary = {
        "comparison": "current numpy / current numba (same commit)",
        "speedup": speedup,
        "numpy_ms_per_step": a["median_ms_per_step"],
        "numba_ms_per_step": b["median_ms_per_step"],
        "initial_microstate_sha256": a["initial_microstate_sha256"],
        "same_warmed_microstate_bitwise": (
            a["warmed_microstate_sha256"] == b["warmed_microstate_sha256"]
        ),
        "warmup_fraction_elongated": {
            "numpy": a["fraction_elongated_after_warmup"][0],
            "numba": b["fraction_elongated_after_warmup"][0],
        },
        "config": a["config"],
        "git": a["git"],
        "environment": a["environment"],
        "note": (
            "Warmup and initial JIT compilation, initialization, file output "
            "and checkpoints are excluded from timed steps. First-use JIT "
            "for paths not exercised during warmup may still affect timing. "
            "Each timing repetition starts from the same seed. "
            "Warm states need not be bitwise identical across backends; "
            "trajectories can diverge numerically."
        ),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nCURRENT-BACKEND COMPARISON")
    print(f"NumPy: {a['median_ms_per_step']:.3f} ms/step")
    print(f"Numba: {b['median_ms_per_step']:.3f} ms/step")
    print(f"Speedup: {speedup:.2f}x (NumPy / Numba)")
    print(f"Repetitions per backend: {args.repeats}")
    print("Same initial microstate: yes")
    print(f"Same warmup microstate bitwise: "
          f"{summary['same_warmed_microstate_bitwise']}")
    print(
        "Elongated fraction after warmup: "
        f"NumPy={summary['warmup_fraction_elongated']['numpy']:.4f}, "
        f"Numba={summary['warmup_fraction_elongated']['numba']:.4f}"
    )
    if (
        summary["warmup_fraction_elongated"]["numpy"]
        != summary["warmup_fraction_elongated"]["numba"]
    ):
        print(
            "Caution: warmup populations differ; interpret the timing ratio "
            "as a workload-dependent comparison."
        )
    print(f"Saved: {directory / 'summary.json'}")


def profile(args: argparse.Namespace) -> None:
    """
    Run the simulation with cProfile instrumentation for the specified backend(s)
    and save the profiling results to JSON and .prof files. Validate that the
    initial microstates and configurations match across backends to ensure a fair
    comparison.
    """
    validate(args)
    directory = result_directory(args, "profile")
    backends = ("numpy", "numba") if args.backend == "both" else (args.backend,)
    for backend in backends:
        run_child(
            args, backend, "profile",
            directory / f"profile_{backend}.json",
            profile_stats=directory / f"profile_{backend}.prof",
        )
    print(f"Saved profiles: {directory}")


def main() -> None:
    """
    Command-line interface for benchmarking and profiling the Oncostreams simulation.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bm = commands.add_parser(
        "benchmark", help="Compare NumPy and Numba on the current code"
    )
    add_physical_options(bm)
    bm.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    pr = commands.add_parser(
        "profile", help="Save cProfile output for NumPy, Numba or both"
    )
    add_physical_options(pr, include_repeats=False)
    pr.add_argument(
        "--backend", choices=("numpy", "numba", "both"), default="both",
        help="Backend(s) to profile (default: both)",
    )
    pr.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    # Internal subprocess entry point is kept out of the public command list.
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        internal = argparse.ArgumentParser(prog="performance.py _worker")
        add_physical_options(internal)
        internal.add_argument(
            "--backend", choices=("numpy", "numba"), required=True
        )
        internal.add_argument("--mode", choices=("benchmark", "profile"), required=True)
        internal.add_argument("--out-file", required=True)
        internal.add_argument("--profile-stats", default=None)
        args = internal.parse_args(sys.argv[2:])
        if args.mode == "profile" and not args.profile_stats:
            internal.error("profile worker requires --profile-stats")
        if not (REPO / "tumorsphere" / "core" / "culture.py").is_file():
            internal.error(
                "Place this script in the repository's benchmarks/ directory"
            )
        worker(args)
        return

    args = parser.parse_args()
    if not (REPO / "tumorsphere" / "core" / "culture.py").is_file():
        parser.error("Place this script in the repository's benchmarks/ directory")
    try:
        if args.command == "benchmark":
            benchmark(args)
        elif args.command == "profile":
            profile(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
