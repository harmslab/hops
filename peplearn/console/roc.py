import numpy as np

from sklearn.metrics import roc_curve, auc
import pickle, sys
x = pickle.load(open(sys.argv[1],"rb"))

y_score = x._model.predict(x._test_features)
y_obs = x._obs.test_values
print("pct correct:",sum(y_score == y_obs)/len(y_obs))
print("pct test:",len(y_obs))

breaks = x._obs.breaks

num_classes = len(breaks) + 1

for i in range(num_classes):

    true_calc =   y_obs == i
    pred_calc = y_score == i

    a, b, c = roc_curve(true_calc,pred_calc)

    if i == 0:
        label = "       E <={:3}".format(breaks[0])
    elif i == (num_classes - 1):
        label = "       E > {:3}".format(breaks[-1])
    else:
        label = "{:3} <= E < {:3}".format(breaks[i-1],breaks[i])

    print(label,auc(a,b))

