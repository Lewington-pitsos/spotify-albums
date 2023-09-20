import json
import pandas as pd
import random

df = pd.read_csv('data/genres_v2.csv', low_memory=False)

ids = list(df['id'].values)

chosen = ids
# chosen = random.sample(ids, 50_000)

with open('data/40_000_ids.json', 'w') as f:
    json.dump(chosen, f)

