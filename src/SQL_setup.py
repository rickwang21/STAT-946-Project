import sqlite3
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
conn = sqlite3.connect('/home/rick/glove.db')
conn.text_factory = str

dataset = pd.read_csv('/home/rick/reduced.txt',header=0,delimiter=';',dtype=np.str)

print(dataset[1:20])

dataset.to_sql('main',conn,if_exists='append')

conn.close()
