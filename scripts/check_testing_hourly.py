"""Read-only hourly pipeline checks. Save diagnostics; never restart or call a model."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_testing_half import snapshot, read


def running(out):
    command = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Cannot inspect Python processes')
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    return [r['ProcessId'] for r in rows
            if 'run_testing_half_complete.py' in (r.get('CommandLine') or '')
            and out.name.lower() in (r.get('CommandLine') or '').lower()]


def check(out, logs):
    value = snapshot(out)
    issues = []
    try:
        value['process_ids'] = running(out)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        value['process_ids'] = None
        issues.append('进程检查失败：'+str(exc))
    if value['state'] != 'COMPLETE':
        if value['process_ids'] == []:
            issues.append('没有运行中的 Pipeline 进程')
        if value['failure']:
            issues.append('停止记录：'+str(value['failure'].get('error_type', value['failure'])))
        quiet = value['last_completed_seconds_ago']
        if quiet is not None and quiet > 600:
            issues.append(f'已有 {quiet/60:.1f} 分钟没有新结果，请检查是否卡住')
        for name, ledger in value['budgets'].items():
            if ledger.get('read_error'):
                issues.append(name+' 账本读取失败：'+ledger['read_error'])
            for request in ledger['pending']:
                if (request.get('seconds') or 0) > 600:
                    issues.append(f"请求等待过久：{request['target']} / {request['seat']} / {request['seconds']:.0f} 秒")
    value['issues'] = issues
    value['checked_local_time'] = datetime.now().astimezone().isoformat(timespec='seconds')
    if issues:
        plan = read(out/'plan.json', {})
        core = Path(plan.get('core_output', str(out/'core')))
        candidates = sorted((core/'targets').rglob('record.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
        value['recent_request_errors'] = []
        for path in candidates:
            row = read(path, {})
            if row.get('error_type') or row.get('exception_type') or row.get('status') in ('FAILED', 'API_FAILED'):
                value['recent_request_errors'].append({'file': str(path), **{k: row.get(k) for k in
                    ('target', 'stage', 'seat', 'status', 'http_status', 'error_type', 'exception_type', 'error', 'provider_error')}})
    encoded = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = logs/'latest.tmp'
    temporary.write_text(encoded, encoding='utf-8')
    temporary.replace(logs/'latest.json')
    with (logs/'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    progress = value['progress']
    message = (f"[{value['checked_local_time']}] {'异常' if issues else '完成' if value['state']=='COMPLETE' else '正常'} | "
               f"核心 {progress['Gate+Tracker']['done']}/{progress['Gate+Tracker']['total']} | "
               f"报告 {progress['Reports H0/FULL']['done']}/{progress['Reports H0/FULL']['total']} | "
               f"评分 {progress['Report scoring']['done']}/{progress['Report scoring']['total']}")
    print(message, flush=True)
    if issues:
        details = message+'\n'+'\n'.join(issues)+'\n\n'+encoded
        (logs/'latest_error.txt').write_text(details, encoding='utf-8')
        with (logs/'alerts.log').open('a', encoding='utf-8') as stream:
            stream.write(message+'\n'+'\n'.join(issues)+'\n\n')
        print('\a'+'\n'.join(issues)+'\n错误已保存：'+str(logs/'latest_error.txt'), flush=True)
    return value['state']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_dir():
        parser.error('运行目录不存在')
    logs = out.with_name(out.name+'_hourly_monitor')
    logs.mkdir(exist_ok=True)
    print('启动时检查一次，此后每小时检查；Ctrl+C 只退出监视，不停止 Pipeline。', flush=True)
    try:
        while True:
            started = time.monotonic()
            try:
                state = check(out, logs)
            except Exception as exc:
                state = 'CHECK_FAILED'
                message = f'{datetime.now().isoformat()} 检查脚本异常：{type(exc).__name__}: {exc}'
                (logs/'latest_error.txt').write_text(message, encoding='utf-8')
                print(message, flush=True)
            if args.once or state == 'COMPLETE':
                break
            while time.monotonic()-started < 3600:
                time.sleep(min(60, 3600-(time.monotonic()-started)))
    except KeyboardInterrupt:
        print('监视已退出，Pipeline 不受影响。')


if __name__ == '__main__':
    main()
