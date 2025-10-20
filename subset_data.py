from datasets import load_from_disk, concatenate_datasets
from collections import defaultdict

DATA_PATH = "train_data_final"
OUT_PATH = "train_data_final_subset"
N_PER_COMBO = 200  # per (cell_type, leukemia_subtype)

ds = load_from_disk(DATA_PATH)
combos = defaultdict(list)

for i, row in enumerate(ds):
    key = (row.get("cell_type"), row.get("leukemia_subtype"))
    if all(key):  # skip if any is None
        combos[key].append(i)

indices = []
for key, idxs in combos.items():
    selected = idxs[:N_PER_COMBO] if len(idxs) >= N_PER_COMBO else idxs
    indices.extend(selected)

subset = ds.select(indices).shuffle(seed=42)
print(f"Final subset size: {subset.num_rows}")
subset.save_to_disk(OUT_PATH)
