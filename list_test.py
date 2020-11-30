import pandas as pd
import os

classes = ['concrete', 'bricks', 'metal', 'thatch']
files = []
for class in classes:
    files = os.listdir