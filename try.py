import pickle
import numpy as np

data = np.random.randint(0, 10000, 4500)

with open('hy.pkl','wb')as f:
    d = pickle.dumps(data)
    f.write(d)