"""Metrics on sealed predictions; caller alone supplies the common truth/masks."""
from collections import defaultdict
from statistics import mean, stdev

HEADS = ('instrument', 'verb', 'target', 'ivt', 'phase')


def evaluate(keys, before, after, truth):
    per_head = {}
    for h in HEADS:
        c = dict(n=0, tp=0, fp=0, fn=0, correct=0, C=0, F=0, H=0, D=0)
        for key in keys:
            t = truth[key]
            if not t['mask'][h]:
                continue
            gt, old, new = (set(t['gt'][h]), set(before[key][h]), set(after[key][h]))
            c['n'] += 1
            c['tp'] += len(new & gt)
            c['fp'] += len(new - gt)
            c['fn'] += len(gt - new)
            c['correct'] += new == gt
            c['C'] += old == gt
            c['F'] += old != gt and new == gt
            c['H'] += old == gt and new != gt
            c['D'] += old != new
        den = 2 * c['tp'] + c['fp'] + c['fn']
        c['f1'] = (200 * c['tp'] / den if den else 100.) if c['n'] else None
        c['precision'] = 100*c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None
        c['recall'] = 100*c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None
        c['accuracy'] = 100*c['correct']/c['n'] if c['n'] else None
        c['harm_percent'] = 100*c['H']/c['C'] if c['C'] else None
        c['net_gain_pp'] = 100*(c['F']-c['H'])/c['n'] if c['n'] else None
        assert c['F'] + c['H'] <= c['D']
        per_head[h] = c
    pooled = {k: sum(c[k] for c in per_head.values()) for k in ('n', 'C', 'F', 'H', 'D')}
    pooled['harm_percent'] = 100*pooled['H']/pooled['C'] if pooled['C'] else None
    pooled['net_gain_pp'] = 100*(pooled['F']-pooled['H'])/pooled['n'] if pooled['n'] else None
    return {'by_head': per_head, 'repair': pooled}


def summary_values(result):
    return {**{h+'_f1': result['by_head'][h]['f1'] for h in HEADS[:-1]},
            'phase_accuracy': result['by_head']['phase']['accuracy'],
            'harm_percent': result['repair']['harm_percent'],
            'net_gain_pp': result['repair']['net_gain_pp']}


def random_summary(results):
    rows = [summary_values(r) for r in results]
    def summarize(field):
        values = [r[field] for r in rows if r[field] is not None]
        return {'mean': mean(values) if values else None,
                'sd': stdev(values) if len(values) > 1 else None, 'n': len(values)}
    return {field: summarize(field) for field in rows[0]}
