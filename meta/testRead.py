import pandas as pd

# Test
df = pd.read_csv('movie_data.csv', encoding='utf-8')
# the following column renaming is necessary on some computers:
df = df.rename(columns={"0": "review", "1": "sentiment"})
print(df.head(3))