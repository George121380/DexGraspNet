import os
import sys
import argparse
import numpy as np


def collect_objects(in_root: str):
    objects = []
    if not os.path.isdir(in_root):
        return objects
    for name in sorted(os.listdir(in_root)):
        d = os.path.join(in_root, name)
        if os.path.isdir(d):
            # find a merged npy file pattern: <OBJ>_merged_*.npy
            cands = [f for f in os.listdir(d) if f.endswith('.npy') and f.startswith(name + '_merged_')]
            if len(cands) > 0:
                # choose the largest by number inside or just first sorted
                cands.sort()
                objects.append((name, os.path.join(d, cands[0])))
    return objects


def normalize_record(rec: dict) -> dict:
    # Ensure required keys exist and are python types
    out = {}
    out['scale'] = float(rec.get('scale', 1.0))
    # Copy qpos dicts directly (they already contain WRJT*, WRJR*, 20 joints)
    for k in ['qpos_left', 'qpos_right']:
        q = rec.get(k, {})
        # convert any numpy scalar to python float
        q_out = {}
        for kk, vv in q.items():
            if hasattr(vv, 'item'):
                try:
                    vv = vv.item()
                except Exception:
                    pass
            q_out[kk] = float(vv)
        out[k] = q_out
    return out


def convert_one(obj_name: str, npy_path: str, out_root: str):
    arr = np.load(npy_path, allow_pickle=True)
    data_list = []
    for i in range(len(arr)):
        rec = arr[i]
        if isinstance(rec, dict):
            data_list.append(normalize_record(rec))
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f'{obj_name}.npy')
    np.save(out_path, np.array(data_list, dtype=object), allow_pickle=True)
    print(f'[OK] {obj_name}: {npy_path} -> {out_path} (num={len(data_list)})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    args = parser.parse_args()

    objs = collect_objects(args.in_root)
    if not objs:
        print(f'[WARN] no objects found in {args.in_root}')
        sys.exit(0)
    os.makedirs(args.out_root, exist_ok=True)
    for name, npy_path in objs:
        convert_one(name, npy_path, args.out_root)


if __name__ == '__main__':
    main()
