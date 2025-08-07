import pandas as pd
import minsearch
from tqdm.auto import tqdm
import json
import random


# Ingestion
def load_index(data_path="../data/data.csv"):
    df = pd.read_csv(data_path)

    df.head()

    df.columns

    index = minsearch.Index(
        text_fields = ['exercise_name', 'type_of_activity', 'type_of_equipment',
        'body_part', 'type', 'muscle_groups_activated', 'instructions'],
        keyword_fields = ['id']
    )

    documents = df.to_dict(orient='records')

    index.fit(documents)
    return index