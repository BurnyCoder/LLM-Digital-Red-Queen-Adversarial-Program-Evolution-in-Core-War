#!/usr/bin/env python
"""Script to read and inspect pickle result files from DRQ experiments."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Patch corewar imports (local corewar folder shadows the pip package)
from corewar.corewar import MARS, Core, Warrior, redcode
import corewar
corewar.MARS = MARS
corewar.Core = Core
corewar.Warrior = Warrior
corewar.redcode = redcode
from llm_corewar import GPTWarrior
from drq import MapElites, Args
from corewar_util import SimulationArgs
import pickle


def read_map_elites(filepath):
    """Read MAP-Elites archive pickle file.

    Returns: dict[round_num, MapElites]
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def read_generations(filepath):
    """Read generations pickle file.

    Returns: list[(operation_type, [[GPTWarrior]])]
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def read_timestamps(filepath):
    """Read timestamps pickle file.

    Returns: list[dict] with keys: abs_iter, i_round, i_iter, dt, rss, vms
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def read_args(filepath):
    """Read args pickle file.

    Returns: Args dataclass
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


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


def print_summary(results_dir):
    """Print summary to stdout and save to readable/ subfolder."""
    results_dir = Path(results_dir)
    readable_dir = results_dir / 'readable'
    readable_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Results: {results_dir}")
    print(f"Saving to: {readable_dir}")
    print("=" * 60)

    # Args
    args_file = results_dir / 'args.pkl'
    if args_file.exists():
        args = read_args(args_file)
        lines = ["[Args]", ""]
        for key, val in vars(args).items():
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
        map_elites = read_map_elites(me_file)
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
        generations = read_generations(gen_file)
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
        timestamps = read_timestamps(ts_file)
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read DRQ experiment results')
    parser.add_argument('results_dir', nargs='?', default='smoke_test/seed0',
                        help='Path to results directory')
    args = parser.parse_args()

    # Handle relative paths from results folder
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent / results_dir

    print_summary(results_dir)
