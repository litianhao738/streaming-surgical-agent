"""Offline five-head responses from the sealed Training collection (never GT labels)."""
from copy import deepcopy
import json
from scripts.run_pgp_pipeline import CachedBackend


class ProbeReplayBackend(CachedBackend):
    def __init__(self, original, collected):
        super().__init__(original)
        self.five_responses = deepcopy(collected['five_head_raw'])

    def five_head(self, seat, *args):
        if seat not in self.five_responses:
            raise ValueError('Replay requested an uncollected five-head seat: '+seat)
        return deepcopy(self.five_responses[seat])


def load_collection(folder, sha, read):
    receipt = read(folder/'receipt.json')
    if (receipt['state'] != 'PASS' or receipt['review_mode'] != 'five_head_probe'
            or sha(folder/'rows.jsonl') != receipt['rows_sha256']):
        raise ValueError('five-head Training collection is not sealed')
    rows = [json.loads(line) for line in (folder/'rows.jsonl').read_text('utf-8').splitlines()]
    indexed = {r['sample_id']:r for r in rows}
    if len(rows) != receipt['rows'] or len(indexed) != len(rows):
        raise ValueError('five-head cache is incomplete or duplicated')
    if any(r['source_split'] != 'Training' for r in rows):
        raise ValueError('Training replay cannot read Testing responses')
    return indexed, receipt['rows_sha256']
