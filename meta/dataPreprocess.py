import os
import sys
import pyprind
import numpy as np
import pandas as pd

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)

# Accumulate data in a list instead of using .append()
data = []

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                data.append([txt, labels[l]])
            pbar.update()

# Convert to DataFrame
df = pd.DataFrame(data, columns=['review', 'sentiment'])

# Shuffle and save
np.random.seed(0)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('movie_data.csv', index=False, encoding='utf-8')


