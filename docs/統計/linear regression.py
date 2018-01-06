
import numpy as np
import statsmodels.api as sm

y =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

x = [
     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
     ]
x2=[

    [100,1,100,1,100,1,100,1,100,1,100,1,100,1,100,1,100,1,100,1,100,1,33]


]
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

print (reg_m(y, x).summary())



print (reg_m(y, x2).summary())