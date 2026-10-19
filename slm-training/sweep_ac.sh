#!/usr/bin/env bash
# Clean thread sweep: warm the page cache first so every config is measured
# under identical conditions, then time real optimizer steps at each setting.
cd "$(dirname "$0")"
echo "warming page cache for train.bin/val.bin ..."
python -c "
import numpy as np
for f in ('train.bin','val.bin'):
    a = np.memmap(f, dtype=np.uint16, mode='r')
    s = int(a[::64].sum())   # touch pages across the whole file
    print(' ', f, len(a), 'tokens warmed')
"
for n in 4 6 8; do
  python train.py --benchmark 3 --threads "$n" 2>&1 | grep -E "Benchmark:" | sed "s/^/[threads=$n] /"
done
echo "SWEEP_DONE"
