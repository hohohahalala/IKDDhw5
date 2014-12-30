# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.grid_search import GridSearchCV

def new_column(dataframe):
	dataframe['Gender'] = 0
	dataframe['Gender'] = dataframe['Sex'].map({'female':1, 'male':0}).astype(int)
	dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch']
	return dataframe

def clean_Embarked(dataframe):
	Ports = list(enumerate(np.unique(dataframe['Embarked'])))
	Ports_dict = {name: i for i, name in Ports}
	dataframe.Embarked = dataframe.Embarked.map(lambda x: Ports_dict[x]).astype(int)
	return dataframe

def main():
	df = pd.read_csv('train.csv', header = 0)
	df = new_column(df)
	df = df.drop(['Age', 'Fare', 'PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin'], axis=1) 
	df = df.dropna()
	df = clean_Embarked(df)

	################################################################################
	df_test = pd.read_csv('test.csv', header = 0)
	df_test = new_column(df_test)
	df_test = clean_Embarked(df_test)
	df_test = df_test.drop(['Age','Fare','PassengerId','Name', 'Sex', 'Ticket', 'Cabin'], axis=1) 

	################################################################################
	train_data = df.values
	test_data = df_test.values

	clf = RandomForestClassifier(n_estimators=300)
	param_grid = {"max_depth": [3, None],"max_features": [1,2,3],"min_samples_split": [1,2,3],"min_samples_leaf": [1, 3, 10],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}
	grid_search = GridSearchCV(clf, param_grid=param_grid,verbose=3,scoring='precision',cv=5)
	grid_search.fit(train_data[0::,1::],train_data[0::,0])
	output = grid_search.predict(test_data)

	fw = open('output.csv','w')
	string = "PassengerId,Survived\n"
	for i, itera in enumerate(output):
		string += str(i+892) + "," + str(int(itera)) + "\n"

	fw.write(string)
	fw.close()

	
if __name__ == '__main__':
	main()





