"""""
https://www.youtube.com/watch?v=tNa99PG8hR8
"""""

from sklearn.datasets import  load_iris
from sklearn import  tree
import  numpy as np
iris=load_iris()
test_idx=[0,50,100]

#training data

train_target=np.delete(iris.target,test_idx)

print(train_target)
train_date=np.delete(iris.data,test_idx,axis=0)

test_target=iris.target[test_idx]
test_data=iris.data[test_idx]


clf=tree.DecisionTreeClassifier()
clf.fit(train_date,train_target)

print(test_target)
print(clf.predict(test_data))


#print the  tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")