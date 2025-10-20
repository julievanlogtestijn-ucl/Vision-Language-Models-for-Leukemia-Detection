from datasets import Dataset, Features, Value, Image
import pandas as pd
import os

DATA_CSV =          "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/dataset_detailed_final.csv"
LEUKEMIA_PATH =     "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/cropped_imgs_desc"
HEALTHY_PATH =      "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/pbc-dataset/PBC_dataset_split/PBC_dataset_split"

df = pd.read_csv(DATA_CSV)

#get subset for debugging
#df = df.sample(50, random_state=42)

def resolve_image_path(row):
    if row["leukemia_subtype"] == "Healthy":
        parts = row["image_filename"].split("_", 2)  # split into 3 parts: split, celltype, rest
        if len(parts) == 3:
            split_part, cell_type, rest = parts
            image_file = f"{rest}"
            return os.path.join(HEALTHY_PATH, split_part, cell_type, image_file)
        return os.path.join(HEALTHY_PATH, row["image_filename"])  # fallback
    else:
        return os.path.join(LEUKEMIA_PATH, row["image_filename"])

df['image'] = df.apply(resolve_image_path, axis=1)

df = df[~df['image_filename'].str.contains(r"\.DS_", regex=True)]
df = df.reset_index(drop=True)

features = Features({
    'image_filename': Value('string'),
    'description': Value('string'),
    'split_type': Value('string'),
    'leukemia_subtype': Value('string'),
    'image_path': Value('string'), 
    'desc_short': Value('string'),
    'desc_detailed': Value('string'),
    'desc_rich': Value('string'),
    'vqa_q1': Value('string'),
    'vqa_a1': Value('string'),
    'vqa_q2': Value('string'),
    'vqa_a2': Value('string'),
    'vqa_q3': Value('string'),
    'vqa_a3': Value('string'),
    'cell_type': Value('string'),
    'image': Image()
})

dataset = Dataset.from_pandas(df, features=features)

train_dataset = dataset.filter(lambda x: x['split_type'] == "train")
test_dataset = dataset.filter(lambda x: x['split_type'] == "test")

train_dataset.save_to_disk("train_data_detailed")
test_dataset.save_to_disk("test_data_detailed")

print("Datasets saved.")