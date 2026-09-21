"""Offline metrics and measured call-path costs; no API calls or budget changes."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

FX = Decimal('6.7065')
SOURCES = {
    'fx': 'https://www.investing.com/currencies/usd-cny-historical-data',
    'glm': 'https://docs.bigmodel.cn/cn/guide/start/pricing',
    'deepseek': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
    'qwen': 'https://help.aliyun.com/zh/model-studio/model-pricing',
}


def read(path):
    return json.loads(path.read_bytes())


def output_tokens(usage):
    prompt = int(usage['prompt_tokens'])
    completion = int(usage['completion_tokens'])
    return max(completion, int(usage.get('total_tokens', prompt+completion))-prompt)


def price_rmb(record, usage):
    """None means unknown, never zero. Published prices are estimates, not invoices."""
    if record['account'] == 'openrouter_usd':
        if usage.get('cost') is not None:
            return Decimal(str(usage['cost']))*FX, 'provider_reported'
        return None, 'unknown'
    if usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
        return None, 'unknown'
    p, c = int(usage['prompt_tokens']), output_tokens(usage)
    cached = int((usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0))
    if record['account'] == 'aliyun_cny':
        inp, out = ('0.2', '0.8') if p <= 32768 else ('0.6', '2.4') if p <= 262144 else ('1.2', '4.8')
        cost = Decimal(p)*Decimal(inp)+Decimal(c)*Decimal(out)
    elif record['account'] == 'glm_requests':
        cost = Decimal(p-cached)*Decimal('0.8')+Decimal(cached)*Decimal('0.23')+Decimal(c)*Decimal('2.8')
    elif record['account'] == 'deepseek_requests':
        cached = int(usage.get('prompt_cache_hit_tokens', cached))
        date = datetime.fromisoformat(record['started_utc'].replace('Z', '+00:00'))
        assert date.tzinfo is not None
        date = date.astimezone(timezone(timedelta(hours=8)))
        peak = date.weekday() < 5 and (9 <= date.hour < 12 or 14 <= date.hour < 18)
        cost = (Decimal(p-cached)+Decimal(cached)*Decimal('0.02')+Decimal(c)*4)*(2 if peak else 1)
    else:
        return None, 'unknown'
    return cost/Decimal(1000000), 'published_rate_estimate'


def summarize(out):
    core = out/'core'
    overview = read(out/'ablation_summary.json')
    assert read(core/'receipt.json')['state'] == 'PASS'
    records = {}
    for directory, _, files in os.walk(core/'targets'):
        if 'record.json' not in files:
            continue
        path = Path(directory)
        r = read(path/'record.json')
        response = read(path/'response.json') if (path/'response.json').exists() else {}
        body = response.get('body', response)
        usage = body.get('usage') or r.get('usage') or {}
        key = r['target'], r['stage']+'|'+r['seat']
        assert key not in records, 'Duplicate current request identity'
        cost, kind = price_rmb(r, usage)
        records[key] = dict(record=r, usage=usage, rmb=cost, cost_kind=kind)
    h0 = {key for key in records if key[1] == 'h0|base'}
    result = dict(status='COMPLETE_STRUCTURED_METRICS', scored_frames=overview['scored_frames'],
        pipeline_frames=overview['pipeline_frames'], fx_usd_cny=str(FX), fx_date='2026-09-16',
        pricing_sources=SOURCES, gate_on_open_rate=overview['gate_on_open_rate'], groups={},
        cost_definition='Observed costs of current collected calls consumed by each replay group, plus shared newly generated H0 (including warmup). Excludes historical imported H0, archived failed/retried attempts, GPU electricity and Report/Judge. Cached H0 adds no new API cost. Not independent paid runs.',
        accuracy_definition='Exact set-match accuracy per head; phase accuracy is categorical accuracy. Macro summaries average instrument, verb, target, ivt only.',
        missing_cost_policy='Missing provider usage/cost remains unknown; known_rmb is an observed subtotal, not an exact full bill.')
    for group, metric in overview['metrics'].items():
        predictions = read(out/'groups'/group/'continuous_predictions.json')
        consumed = set(h0)
        cached_h0 = 0
        for row in predictions:
            for call in row['call_keys']:
                key = row['key'], call
                if key not in records:
                    assert call == 'h0|base', 'Missing collected downstream call: '+str(key)
                    cached_h0 += 1
                else:
                    consumed.add(key)
        by_model = defaultdict(lambda: dict(requests=0, prompt_tokens=0, completion_tokens=0,
            usage_missing=0, cost_missing=0, known_rmb=Decimal(0), reserved_rmb_for_unknown=Decimal(0),
            unpriced_request_count=0))
        for key in sorted(consumed):
            item = records[key]
            r, usage = item['record'], item['usage']
            d = by_model[r['model']]
            d['requests'] += 1
            if usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
                d['prompt_tokens'] += int(usage['prompt_tokens'])
                d['completion_tokens'] += output_tokens(usage)
            else:
                d['usage_missing'] += 1
            if item['rmb'] is None:
                d['cost_missing'] += 1
                if r['account'] in ('openrouter_usd', 'aliyun_cny'):
                    d['reserved_rmb_for_unknown'] += Decimal(r['charge'])*(FX if r['account']=='openrouter_usd' else 1)
                else:
                    d['unpriced_request_count'] += 1
            else:
                d['known_rmb'] += item['rmb']
        totals = {k:sum(d[k] for d in by_model.values()) for k in next(iter(by_model.values()))}
        totals['reported_total_tokens'] = totals['prompt_tokens']+totals['completion_tokens']
        totals['known_plus_unknown_monetary_reservations_rmb'] = totals['known_rmb']+totals['reserved_rmb_for_unknown']
        heads = ('instrument','verb','target','ivt')
        result['groups'][group] = dict(metrics=metric,
            macro_four_head={name:sum(metric[h][field] for h in heads)/4 for name,field in
                            [('f1','f1'),('precision','precision'),('accuracy','accuracy')]},
            cost=totals, by_model=dict(by_model), cached_pipeline_h0=cached_h0,
            consumed_call_identities=[list(k) for k in sorted(consumed)])
    path = out/'ablation_metrics_cost_rmb.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    print(json.dumps({g:{'macro_four_head':v['macro_four_head'], 'metrics':v['metrics'], 'cost':v['cost']}
                      for g,v in result['groups'].items()}, ensure_ascii=False, default=str), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    summarize(p.parse_args().output.resolve())
