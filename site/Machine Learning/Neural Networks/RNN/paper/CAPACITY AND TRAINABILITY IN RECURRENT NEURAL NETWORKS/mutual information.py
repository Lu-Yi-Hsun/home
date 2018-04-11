import numpy as np
from sklearn.metrics import mutual_info_score
import math
np.random.seed(42)
n=1
label_true = np.array([1,2,3,4,5,6,7,8])
label_predict = label_true
print(n*mutual_info_score(label_true, label_predict)/math.log(2))