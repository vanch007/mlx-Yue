"""Transcribe recordings with native MLX SheetSage2 and MERT2."""
import argparse
import json
from pathlib import Path
import sys


def run(args):
    from ..measure import GPUExecution
    from ..transcription.pipeline import transcribe
    with GPUExecution(memory_budget_gib=args.memory_budget_gib) as guard:
        def cancelled():
            guard.check()
            return False

        result = transcribe(args.audio, args.output, model_path=args.model, base_model=args.base_model,
                            offline=args.offline, cache_dir=args.cache_dir, task=args.task,
                            preset=args.preset, max_seconds=args.max_seconds, cancelled=cancelled,
                            progress=None if args.quiet else lambda value: print(json.dumps(value), file=sys.stderr))
        guard.check()
    print(json.dumps({k: result[k] for k in ('status', 'backend', 'duration_seconds', 'truncated', 'elapsed_seconds')}, indent=2))
    if result['status'] != 'complete':
        raise RuntimeError(f"Transcription produced no usable ABC: {result.get('abc_error')}")
    return int(result['truncated'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='m-a-p/SheetSage2')
    parser.add_argument('--base-model', default='m-a-p/MERT-v2-FullSong')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--cache-dir', default='models/hf-cache')
    parser.add_argument('--task', choices=('full', 'melody-full', 'melody-vocal'), default='melody-full')
    parser.add_argument('--preset', choices=('default', 'paper'), default='default')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--memory-budget-gib', type=float, default=24)
    parser.add_argument('--quiet', action='store_true')
    try:
        return run(parser.parse_args())
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
