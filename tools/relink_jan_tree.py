"""Re-point the workspace links inside a moved Jan checkout.

    python tools/relink_jan_tree.py [path to the checkout]

yarn creates `node_modules/@janhq/*` in each workspace as symlinks with ABSOLUTE targets, so a
checkout that is moved (or that was built next to a sibling checkout) keeps resolving into the old
location. This rewrites every link under the checkout whose target lies in another Jan tree to the
same relative place in this one, as a relative link, so the next move needs nothing.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'src' / 'jan'
MARKERS = ('/src/jan/', '/src/jan-fastllm/')      # what a foreign Jan checkout's path looks like


def links(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if 'target' in dirnames:
            dirnames.remove('target')
        # a dangling directory link is reported among the files, and dangling is the usual case
        for name in list(dirnames) + filenames:
            p = Path(dirpath) / name
            # yarn on Windows makes junctions, which Python does not count as symlinks
            if p.is_symlink() or os.path.isjunction(p):
                if name in dirnames:
                    dirnames.remove(name)
                yield p


def main():
    fixed = kept = 0
    for link in links(JAN):
        target = os.readlink(link).replace('\\', '/')
        if target.startswith('/d/'):
            target = 'D:' + target[2:]
        low = target.lower()
        hit = next((m for m in MARKERS if m in low), None)
        if hit is None or low.startswith(str(JAN).replace('\\', '/').lower()):
            kept += 1
            continue
        rest = target[low.index(hit) + len(hit):]
        new_target = JAN / rest
        if not new_target.exists():
            print('no counterpart here, left alone:', link, '->', target)
            kept += 1
            continue
        os.rmdir(link) if link.is_dir() else os.remove(link)
        rel = os.path.relpath(new_target, link.parent)
        try:
            os.symlink(rel, link, target_is_directory=True)
            kind = 'symlink'
        except OSError:
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(new_target)], capture_output=True)
            kind = 'junction'
        ok = link.is_dir() and any(link.iterdir())
        print(f'{kind:8s} {link.relative_to(JAN)} -> {rel} {"ok" if ok else "BROKEN"}')
        fixed += 1
    print(f'relinked {fixed}, untouched {kept}')


if __name__ == '__main__':
    main()
