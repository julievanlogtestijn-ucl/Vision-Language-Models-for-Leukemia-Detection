from datasets import Dataset, Features, Value, Image
import pandas as pd
import os

DATA_CSV =       "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/bloodcellatlas_data/bloodcellatlas.csv"
IMAGE_PATH =     "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/bloodcellatlas_data/images"

df = pd.read_csv(DATA_CSV)

df['image'] = df['image_filename'].apply(lambda x: os.path.join(IMAGE_PATH, x))

df = df[['image_filename', 'description', 'image']]

#get subset for debugging
#df = df.sample(50, random_state=42)

features = Features({
    'image_filename': Value('string'),
    'description': Value('string'),
    'image': Image()
})

dataset = Dataset.from_pandas(df, features=features)

dataset.save_to_disk("external_test_data")

print("Dataset saved.")
