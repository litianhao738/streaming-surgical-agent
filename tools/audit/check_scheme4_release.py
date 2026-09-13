"""Find repository Python dependencies missing from Git's index (read-only)."""
import ast
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def closure(entries):
    pending=list(entries); seen=set()
    def locate(module):
        path=Path(*module.split('.'))
        for base in (ROOT,ROOT/'src'):
            for p in (base/path.with_suffix('.py'),base/path/'__init__.py'):
                if p.is_file(): return p.relative_to(ROOT).as_posix()
    while pending:
        item=pending.pop()
        if item in seen: continue
        seen.add(item)
        file=ROOT/item; tree=ast.parse(file.read_text(encoding='utf-8-sig'))
        parts=Path(item).with_suffix('').parts
        if parts[0]=='src': parts=parts[1:]
        package=list(parts[:-1])
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import): modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                if node.module: prefix+=node.module.split('.')
                name='.'.join(prefix); modules=[name]+[name+'.'+a.name for a in node.names if a.name!='*']
            for module in modules:
                found=locate(module)
                if found and found not in seen: pending.append(found)
    return seen


if __name__=='__main__':
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    files=closure(['scripts/run_pipeline.py','scripts/run_tracker_scheme4_pipeline.py','tests/unit/test_scheme4_complete_pipeline.py',
                   'tests/unit/test_tracker_pipeline_v2.py'])
    for p in sorted(files-tracked): print(p)
