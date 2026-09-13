"""Use the pinned PGP research default on already-computed features, no API."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from surgical_agent.research.gate.pgp_default import load_default, predict_default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, help='JSON containing X and ordered feature_names')
    parser.add_argument('--output', type=Path, help='new output JSON; never overwrite')
    args = parser.parse_args()
    manifest, _ = load_default()
    if args.features is None:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--features requires a fresh --output')
    packet = json.loads(args.features.read_text('utf-8'))
    result = predict_default(packet['X'], packet['feature_names'])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump({'version': result['version'], 'actions': result['actions'].tolist(),
                   'scores': result['scores'].tolist(), 'api_calls': 0, 'deployable': False},
                  f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
