import pandas as pd
import os

classes = ['concrete', 'bricks', 'metal', 'thatch']
files = pd.DataFrame()
for class_ in classes:
    for file in os.listdir('data/test/'+class_):
        files = files.append(pd.Series({'name': file, 'class': class_}), ignore_index=True)
files = files.sort_values(by='name')
print(files.head())