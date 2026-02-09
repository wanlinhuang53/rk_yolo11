import argparse
import os
from typing import Set

def _iter_images(root_dir: str, exts: Set[str]):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower().lstrip('.')
            if ext in exts:
                yield os.path.join(dirpath, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--exts', default='jpg,jpeg,png,bmp')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--posix', action='store_true')
    args = parser.parse_args()

    img_dir = os.path.abspath(args.img_dir)
    exts = {e.strip().lower() for e in args.exts.split(',') if e.strip()}

    imgs = sorted(_iter_images(img_dir, exts))
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in imgs:
            if args.posix:
                p = p.replace('\\', '/')
            f.write(p + '\n')

    print(f'Wrote {len(imgs)} image paths to: {out_path}')

if __name__ == '__main__':
    main()
