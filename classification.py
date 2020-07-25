import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import random

import graphviz

data = pd.read_csv("/home/philip/Desktop/heart.csv")
feature_names = list(data.columns[:-1])
results = data.values[:,-1]
samples = data.values[:,0:-1]

# decision tree with gini, entropy and varying max_depth
for criterion in ['gini','entropy']:
    for depth in range(1,11):
        treeclassifier = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        treeclassifier.fit(samples, results)
        
        tree.plot_tree(treeclassifier)
        dot_data = tree.export_graphviz(treeclassifier, out_file=None, feature_names=feature_names, class_names=['No', 'Yes'])
        graph = graphviz.Source(dot_data) 
        graph.render("heart") 
        #os.system("google-chrome /home/philip/heart.pdf")
        
        # test with original data
        ng_count = 0
        for d in data.values:
            predict = treeclassifier.predict([d[0:-1]])
            expected = d[-1]
            if predict[0] == expected:
                ok = 'OK'
            else:
                ok = 'NG'
                ng_count += 1
                #print(d, "=>", predict, ok)
                percentage = 100 / len(data.values) * ng_count
        print("Decision tree with criterion = ", criterion,
              "=> Depth = ", depth,
              "=> Total NG count =", ng_count,
              "out of", len(data.values), "(", round(percentage,1), "% )")

print()

# bayesian classification (varying degrees of alpha parameter)
bdict = { 'MultinomialNB': MultinomialNB,
          'BernoulliNB': BernoulliNB,
          'ComplementNB': ComplementNB,
          'CategoricalNB': CategoricalNB }
for bayes in bdict:
    r = 0
    while r < 1.0:
        r+=0.1
        gnb = bdict[bayes](alpha = r)
        gnb.fit(samples, results)
        ng_count = 0
        for d in data.values:
            predict = gnb.predict([d[0:-1]])
            expected = d[-1]
            if predict[0] == expected:
                ok = 'OK'
            else:
                ok = 'NG'
                ng_count += 1
                #print(d, "=>", predict, ok)
                percentage = 100 / len(data.values) * ng_count
        print(bayes, "=> range = ", round(r,1), "=> Total NG count =", ng_count,
                      "out of", len(data.values), "(", round(percentage,1), "% )")
                
print()

# support vector machines
ng_count=0
reg = svm.SVC()
reg.fit(samples, results)
for d in data.values:
    predict = reg.predict([d[0:-1]])
    expected = d[-1]
    if predict[0] == expected:
        ok = 'OK'
    else:
        ok = 'NG'
        ng_count += 1
        #print(d, "=>", predict, ok)
        percentage = 100 / len(data.values) * ng_count
print("SVC", "=> Total NG count =", ng_count,
      "out of", len(data.values), "(", round(percentage,1), "% )")

print()
# nearest neighbors
for algo in {'auto', 'ball_tree', 'kd_tree', 'brute'}:
    for n_neighbors in range(1,10):
        ng_count=0
        reg = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algo)
        reg.fit(samples, results)
        for d in data.values:
            predict = reg.predict([d[0:-1]])
            expected = d[-1]
            if predict[0] == expected:
                ok = 'OK'
            else:
                ok = 'NG'
                ng_count += 1
                #print(d, "=>", predict, ok)
                percentage = 100 / len(data.values) * ng_count
        print("NN => algo", algo, "=> neighbors = ", n_neighbors, "=> Total NG count =", ng_count,
              "out of", len(data.values), "(", round(percentage,1), "% )")
                
