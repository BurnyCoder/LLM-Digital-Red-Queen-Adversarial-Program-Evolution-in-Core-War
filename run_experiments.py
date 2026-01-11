#!/usr/bin/env python3
"""
Digital Red Queen (DRQ) - Experiment Runner

This script provides a unified interface to run all experiments from the paper:
"Digital Red Queen: Adversarial Program Evolution in Core War with LLMs"

Usage:
    python run_experiments.py --mode <mode> [options]

Modes:
    smoke_test      - Ultra-fast sanity check (1 round, 3 iterations, ~30 sec)
    quick_test      - Fast sanity check (3 rounds, 20 iterations, ~5 min)
    full_drq        - Full DRQ replication (20 rounds, 250 iterations, hours)
    static_opt      - Static optimization against single opponent (1 round)
    ablation_k1     - History ablation with K=1
    ablation_k3     - History ablation with K=3
    no_map_elites   - MAP-Elites ablation (single cell)
    eval            - Evaluate a warrior against human opponents
    visualize       - Visualize a battle between warriors
    batch_static    - Run static optimization against all human warriors
"""

import argparse
import random
import subprocess
import sys
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from glob import glob

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
HUMAN_WARRIORS_DIR = PROJECT_ROOT / "human_warriors"
COREWAR_DIR = PROJECT_ROOT / "corewar"
RESULTS_DIR = PROJECT_ROOT / "results"


def _setup_pickle_imports():
    """Setup imports needed for unpickling result files.

    Returns imported classes that need to be added to caller's globals.
    """
    # Add project paths
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Patch corewar imports (local corewar folder shadows the pip package)
    from corewar.corewar import MARS, Core, Warrior, redcode
    import corewar
    corewar.MARS = MARS
    corewar.Core = Core
    corewar.Warrior = Warrior
    corewar.redcode = redcode

    # Import classes needed for unpickling and inject into __main__
    from llm_corewar import GPTWarrior
    from drq import MapElites, Args
    from corewar_util import SimulationArgs

    # Inject into __main__ so pickle can find them
    import __main__
    __main__.GPTWarrior = GPTWarrior
    __main__.MapElites = MapElites
    __main__.Args = Args
    __main__.SimulationArgs = SimulationArgs


def format_warrior(w, indent=""):
    """Format all attributes of a GPTWarrior as a string."""
    lines = []
    lines.append(f"{indent}id: {w.id}")
    lines.append(f"{indent}parent_id: {w.parent_id}")
    lines.append(f"{indent}fitness: {w.fitness}")
    lines.append(f"{indent}bc: {w.bc}")
    if w.error:
        lines.append(f"{indent}error: {w.error}")
    if w.outputs:
        lines.append(f"{indent}outputs: {w.outputs}")
    if w.full_outputs:
        lines.append(f"{indent}full_outputs: {w.full_outputs}")
    lines.append(f"{indent}prompt:")
    for line in w.prompt.strip().split('\n'):
        lines.append(f"{indent}  {line}")
    lines.append(f"{indent}llm_response:")
    for line in w.llm_response.strip().split('\n'):
        lines.append(f"{indent}  {line}")
    if w.warrior:
        lines.append(f"{indent}warrior.name: {w.warrior.name}")
        lines.append(f"{indent}warrior.author: {w.warrior.author}")
        lines.append(f"{indent}warrior.start: {w.warrior.start}")
        lines.append(f"{indent}warrior.instructions: ({len(w.warrior.instructions)})")
        for idx, instr in enumerate(w.warrior.instructions):
            lines.append(f"{indent}  [{idx}] {instr}")
    return '\n'.join(lines)


def print_results_summary(results_dir):
    """Print summary of experiment results and save to readable/ subfolder."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Setup imports needed for unpickling
    _setup_pickle_imports()

    readable_dir = results_dir / 'readable'
    readable_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Results: {results_dir}")
    print(f"Saving to: {readable_dir}")
    print("=" * 60)

    # Args
    args_file = results_dir / 'args.pkl'
    if args_file.exists():
        with open(args_file, 'rb') as f:
            exp_args = pickle.load(f)
        lines = ["[Args]", ""]
        for key, val in vars(exp_args).items():
            if hasattr(val, '__dict__'):
                lines.append(f"{key}:")
                for k2, v2 in vars(val).items():
                    lines.append(f"  {k2}: {v2}")
            else:
                lines.append(f"{key}: {val}")
        content = '\n'.join(lines)
        print(f"\n{content}")
        (readable_dir / 'args.txt').write_text(content)

    # MAP-Elites
    me_file = results_dir / 'all_rounds_map_elites.pkl'
    if me_file.exists():
        with open(me_file, 'rb') as f:
            map_elites = pickle.load(f)
        lines = ["[MAP-Elites Archive]", ""]
        for round_num, me in map_elites.items():
            best = me.get_best()
            lines.append(f"Round {round_num}:")
            lines.append(f"  Archive cells: {len(me.archive)}")
            lines.append(f"  Fitness history: {me.fitness_history}")
            if best:
                lines.append(f"  Best fitness: {best.fitness:.4f}")
                lines.append(f"  Best BC: {best.bc}")
            lines.append(f"\n  All entries:")
            for bc, warrior in me.archive.items():
                lines.append(f"\n    --- BC {bc} ---")
                lines.append(format_warrior(warrior, indent="    "))
        content = '\n'.join(lines)
        print(f"\n{content}")
        (readable_dir / 'map_elites.txt').write_text(content)

    # Generations
    gen_file = results_dir / 'all_generations.pkl'
    if gen_file.exists():
        with open(gen_file, 'rb') as f:
            generations = pickle.load(f)
        lines = ["[Generations]", ""]
        for gen_idx, (op_type, batches) in enumerate(generations):
            total_warriors = sum(len(batch) for batch in batches)
            lines.append(f"\nGen {gen_idx} ({op_type}): {total_warriors} warriors")
            for batch_idx, batch in enumerate(batches):
                for w_idx, w in enumerate(batch):
                    lines.append(f"\n  --- Warrior {batch_idx}.{w_idx} ---")
                    lines.append(format_warrior(w, indent="  "))
        content = '\n'.join(lines)
        print(f"\n{content}")
        (readable_dir / 'generations.txt').write_text(content)

    # Timestamps
    ts_file = results_dir / 'timestamps.pkl'
    if ts_file.exists():
        with open(ts_file, 'rb') as f:
            timestamps = pickle.load(f)
        total_time = sum(t['dt'] for t in timestamps)
        lines = ["[Timestamps]", ""]
        lines.append(f"Total iterations: {len(timestamps)}")
        lines.append(f"Total time: {total_time:.2f}s")
        lines.append(f"\nAll entries:")
        for ts in timestamps:
            lines.append(f"  abs_iter={ts['abs_iter']}, round={ts['i_round']}, iter={ts['i_iter']}")
            lines.append(f"    dt={ts['dt']:.4f}s, rss={ts['rss']/1024/1024:.1f}MB, vms={ts['vms']/1024/1024:.1f}MB")
        content = '\n'.join(lines)
        print(f"\n{content}")
        (readable_dir / 'timestamps.txt').write_text(content)

    # Champion files
    champions = list(results_dir.glob('*_champion.red'))
    if champions:
        lines = ["[Champions]", ""]
        for champ in sorted(champions):
            lines.append(f"\n--- {champ.name} ---")
            for line in champ.read_text().strip().split('\n'):
                lines.append(line)
        content = '\n'.join(lines)
        print(f"\n{content}")
        (readable_dir / 'champions.txt').write_text(content)

    print(f"\n{'=' * 60}")
    print(f"Saved readable files to: {readable_dir}")
    print(f"{'=' * 60}")


def get_results_dir(mode, seed, save_dir=None, initial_opponent=None):
    """Get the results directory for a given mode and seed."""
    if save_dir:
        return Path(save_dir)

    if mode == "static_opt":
        opp_name = Path(initial_opponent).stem if initial_opponent else "imp"
        return RESULTS_DIR / "static_opt" / f"{opp_name}_seed{seed}"

    mode_dirs = {
        "smoke_test": "smoke_test",
        "quick_test": "quick_test",
        "full_drq": "full_drq",
        "ablation_k1": "ablation_k1",
        "ablation_k3": "ablation_k3",
        "no_map_elites": "no_map_elites",
    }

    if mode in mode_dirs:
        return RESULTS_DIR / mode_dirs[mode] / f"seed{seed}"

    return None


def check_setup():
    """Verify environment is properly configured."""
    # Check for .env file
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("WARNING: .env file not found. Create one with OPENAI_API_KEY=your_key")
        print(f"  Expected location: {env_file}")

    # Check for requirements
    try:
        import tyro
        import openai
        import pygame
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt && pip install -e corewar")
        sys.exit(1)


def get_base_drq_args(args):
    """Get base arguments common to all DRQ runs."""
    base_args = [
        f"--seed={args.seed}",
        f"--n_processes={args.n_processes}",
        f"--gpt_model={args.model}",
        f"--temperature={args.temperature}",
        "--simargs.rounds=20",
        "--simargs.size=8000",
        "--simargs.cycles=80000",
        "--simargs.processes=8000",
        "--simargs.length=100",
        "--simargs.distance=100",
        "--timeout=900",
        "--bc_axes=tsp,mc",
        "--warmup_with_init_opps=True",
        "--warmup_with_past_champs=True",
        "--n_init=8",
        "--n_mutate=1",
        "--fitness_threshold=0.8",
        "--sample_new_percent=0.1",
    ]
    # Add reasoning_effort if specified
    if args.reasoning_effort:
        base_args.append(f"--reasoning_effort={args.reasoning_effort}")
    return base_args


def run_smoke_test(args):
    """Smoke test - 1 round, 3 iterations (~30 sec)."""
    print("\n" + "="*60)
    print("Running SMOKE TEST (1 round, 3 iterations)")
    print("Fastest sanity check - just verifies everything works")
    print("="*60 + "\n")

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "smoke_test" / f"seed{args.seed}")
    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=1",
        "--n_iters=3",
        "--log_every=1",
        "--last_k_opps=1",
        "--n_init=2",
        "--resume=True",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_quick_test(args):
    """Quick test run - 3 rounds, 20 iterations."""
    print("\n" + "="*60)
    print("Running QUICK TEST (3 rounds, 20 iterations)")
    print("="*60 + "\n")

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "quick_test" / f"seed{args.seed}")
    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=3",
        "--n_iters=20",
        "--log_every=5",
        "--last_k_opps=3",
        "--resume=True",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_full_drq(args):
    """Full DRQ replication - 20 rounds, 250 iterations."""
    print("\n" + "="*60)
    print("Running FULL DRQ (20 rounds, 250 iterations)")
    print("This will take several hours!")
    print("="*60 + "\n")

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "full_drq" / f"seed{args.seed}")
    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=20",
        "--n_iters=250",
        "--log_every=20",
        "--last_k_opps=20",
        "--resume=True",
        f"--job_timeout={args.job_timeout}",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_static_optimization(args):
    """Static optimization - single round against one opponent."""
    print("\n" + "="*60)
    print("Running STATIC OPTIMIZATION (1 round, 1000 iterations)")
    print("="*60 + "\n")

    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")
    opp_name = Path(initial_opp).stem
    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "static_opt" / f"{opp_name}_seed{args.seed}")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=1",
        "--n_iters=1000",
        "--log_every=50",
        "--last_k_opps=1",
        "--resume=True",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_ablation(args, k):
    """History ablation experiment with specified K value."""
    print("\n" + "="*60)
    print(f"Running ABLATION K={k} (20 rounds, 250 iterations)")
    print("="*60 + "\n")

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / f"ablation_k{k}" / f"seed{args.seed}")
    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=20",
        "--n_iters=250",
        "--log_every=20",
        f"--last_k_opps={k}",
        "--resume=True",
        f"--job_timeout={args.job_timeout}",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_evaluation(args):
    """Evaluate a warrior against human opponents."""
    print("\n" + "="*60)
    print("Running EVALUATION")
    print("="*60 + "\n")

    if not args.warrior_path:
        print("ERROR: --warrior_path is required for evaluation mode")
        sys.exit(1)

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "eval")
    opponents_glob = args.opponents_glob or str(HUMAN_WARRIORS_DIR / "*.red")

    cmd = [
        sys.executable, str(SRC_DIR / "eval_warriors.py"),
        f"--seed={args.seed}",
        f"--warrior_path={args.warrior_path}",
        f"--opponents_path_glob={opponents_glob}",
        f"--n_processes={args.n_processes}",
        f"--save_dir={save_dir}",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def run_visualization(args):
    """Visualize a battle between warriors."""
    print("\n" + "="*60)
    print("Running VISUALIZATION")
    print("="*60 + "\n")

    warriors = args.warriors or [
        str(HUMAN_WARRIORS_DIR / "imp.red"),
        str(HUMAN_WARRIORS_DIR / "dwarf.red"),
    ]

    cmd = [
        sys.executable, "-m", "corewar.graphics",
        "--warriors", *warriors,
        "--rounds", "20",
        "--size", "8000",
        "--cycles", "80000",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(COREWAR_DIR))


def run_batch_static(args):
    """Run static optimization against all human warriors."""
    print("\n" + "="*60)
    print("Running BATCH STATIC OPTIMIZATION")
    print("This will run against all human warriors!")
    print("="*60 + "\n")

    human_warriors = sorted(glob(str(HUMAN_WARRIORS_DIR / "*.red")))
    print(f"Found {len(human_warriors)} human warriors\n")

    for i, warrior_path in enumerate(human_warriors):
        warrior_name = Path(warrior_path).stem
        print(f"\n[{i+1}/{len(human_warriors)}] Processing: {warrior_name}")

        save_dir = str(PROJECT_ROOT / "results" / "batch_static" / f"{warrior_name}_seed{args.seed}")

        # Skip if already completed
        if Path(save_dir).exists() and (Path(save_dir) / "round_000_champion.red").exists():
            print(f"  Skipping (already completed)")
            continue

        cmd = [
            sys.executable, str(SRC_DIR / "drq.py"),
            *get_base_drq_args(args),
            f"--save_dir={save_dir}",
            f"--initial_opps={warrior_path}",
            "--n_rounds=1",
            "--n_iters=1000",
            "--log_every=100",
            "--last_k_opps=1",
            "--resume=True",
        ]

        result = subprocess.run(cmd, cwd=str(SRC_DIR))
        if result.returncode != 0:
            print(f"  WARNING: Failed for {warrior_name}")


def run_map_elites_ablation(args):
    """Run DRQ without MAP-Elites (single cell variant)."""
    print("\n" + "="*60)
    print("Running MAP-ELITES ABLATION (single cell)")
    print("="*60 + "\n")

    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / "no_map_elites" / f"seed{args.seed}")
    initial_opp = args.initial_opponent or str(HUMAN_WARRIORS_DIR / "imp.red")

    cmd = [
        sys.executable, str(SRC_DIR / "drq.py"),
        *get_base_drq_args(args),
        f"--save_dir={save_dir}",
        f"--initial_opps={initial_opp}",
        "--n_rounds=10",
        "--n_iters=250",
        "--log_every=20",
        "--last_k_opps=10",
        "--single_cell=True",  # Disable MAP-Elites
        "--resume=True",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Digital Red Queen (DRQ) - Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        choices=[
            "smoke_test", "quick_test", "full_drq", "static_opt",
            "ablation_k1", "ablation_k3",
            "eval", "visualize", "batch_static",
            "no_map_elites"
        ],
        default="smoke_test",
        help="Experiment mode to run (default: smoke_test)"
    )

    # Common arguments
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--n_processes", "-p", type=int, default=20, help="Number of parallel processes")
    parser.add_argument("--model", default=os.environ.get("DRQ_MODEL", "gpt-5-nano"), help="LLM model to use (env: DRQ_MODEL)")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM temperature")
    parser.add_argument("--reasoning_effort", default=os.environ.get("DRQ_REASONING_EFFORT"),
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level for GPT-5 models (env: DRQ_REASONING_EFFORT)")
    parser.add_argument("--save_dir", help="Directory to save results")
    parser.add_argument("--job_timeout", type=int, default=36000, help="Job timeout in seconds")

    # Opponent arguments
    parser.add_argument("--initial_opponent", "-o", help="Initial opponent warrior file")
    parser.add_argument("--opponents_glob", help="Glob pattern for opponent warriors (eval mode)")

    # Evaluation arguments
    parser.add_argument("--warrior_path", "-w", help="Path to warrior to evaluate")

    # Visualization arguments
    parser.add_argument("--warriors", nargs="+", help="Warriors to visualize (2+ files)")

    # Skip setup check
    parser.add_argument("--skip_check", action="store_true", help="Skip setup verification")

    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"Using random seed: {args.seed}")

    # Verify setup
    if not args.skip_check:
        check_setup()

    # Run selected mode
    mode_handlers = {
        "smoke_test": run_smoke_test,
        "quick_test": run_quick_test,
        "full_drq": run_full_drq,
        "static_opt": run_static_optimization,
        "ablation_k1": lambda a: run_ablation(a, k=1),
        "ablation_k3": lambda a: run_ablation(a, k=3),
        "eval": run_evaluation,
        "visualize": run_visualization,
        "batch_static": run_batch_static,
        "no_map_elites": run_map_elites_ablation,
    }

    handler = mode_handlers[args.mode]
    result = handler(args)

    if result and result.returncode != 0:
        sys.exit(result.returncode)

    # Print results summary for modes that produce results
    results_dir = get_results_dir(args.mode, args.seed, args.save_dir, args.initial_opponent)
    if results_dir and results_dir.exists():
        print_results_summary(results_dir)


if __name__ == "__main__":
    main()
