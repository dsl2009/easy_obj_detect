from skimage import io
from dsl_data import utils
from matplotlib import pyplot as plt
import json
dd = {
      'name':'2',
      'timestamp':100000,
      'category':'s',
      'bbox': [1, 2, 3, 4],
      'score':0.15
                    }
x = []
x.append(dd)
print(json.dumps(x))
