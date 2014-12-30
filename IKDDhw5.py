# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pylab as P
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sklearn.svm as svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer


def main():
	df = pd.read_csv('train.csv', header = 0)
	df['Gender'] = 0
	df['Gender'] = df['Sex'].map({'female':1, 'male':0}).astype(int)
	df['FamilySize'] = df['SibSp'] + df['Parch']

	print df.info()

	# tmp = df['Name'].str.split(".")
	# for i, itera in enumerate(tmp) :
	# 	itera.pop(-1)
	# 	itera = str(itera[0]).split(" ")
	# 	tmp[i] = itera[-1]

	# df['Title_str'] = tmp
	# t = list(enumerate(np.unique(df['Title_str'])))
	# t_dict = {name: i for i, name in t}
	# df['Title'] = df.Title_str.map(lambda x: t_dict[x]).astype(int)
	# df['Title'] = df['Title'] * df['Gender']

	df = df.drop(['Fare','PassengerId','Name', 'Sex', 'Ticket', 'Cabin'], axis=1) 
	df = df.drop(['Age'], axis = 1)
	df = df.dropna()

	Ports = list(enumerate(np.unique(df['Embarked'])))
	Ports_dict = {name: i for i, name in Ports}
	df.Embarked = df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

	df_test = pd.read_csv('test.csv', header = 0)
	df_test['Gender'] = 0
	df_test['Gender'] = df_test['Sex'].map({'female':1, 'male':0}).astype(int)

	Ports = list(enumerate(np.unique(df_test['Embarked'])))
	Ports_dict = {name: i for i, name in Ports}
	df_test.Embarked = df_test.Embarked.map(lambda x : Ports_dict[x]).astype(int)

	df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
	df_test = df_test.drop(['Age','Fare','PassengerId','Name', 'Sex', 'Ticket', 'Cabin'], axis=1) 

	train_data = df.values
	test_data = df_test.values

	# build a classifier
	clf = RandomForestClassifier(n_estimators=100)

	# specify parameters and distributions to sample from
	param_grid = {"max_depth": [3, None],"max_features": [1,2,3],"min_samples_split": [1,2,3],"min_samples_leaf": [1, 3, 10],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}
    # run grid search
	grid_search = GridSearchCV(clf, param_grid=param_grid,verbose=3,scoring='precision',cv=10)
	grid_search.fit(train_data[0::,1::],train_data[0::,0])
	output = grid_search.predict(test_data)



	# print output

	fw = open('output.csv','w')
	string = "PassengerId,Survived\n"
	for j, itera in enumerate(output):
		string += str(j+892) + "," + str(int(itera)) + "\n"

	fw.write(string)
	fw.close()

	





if __name__ == '__main__':
	main()





