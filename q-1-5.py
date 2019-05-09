import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import OrderedDict

df = pd.read_csv("data.csv")
train = df.sample(frac=0.8, random_state=200)
validation = df.drop(train.index)
num_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
cols = num_cols + ['Work_accident' , 'promotion_last_5years' , 'sales' , 'salary']
splitPoints = []
error = []

def wholeEntropy(tdata):
	dfList = pd.value_counts(tdata['left'].values, sort=False).tolist()
	total = sum(dfList)
	if len(dfList) == 0 or len(dfList) == 1:
		return 0
	q = dfList[0]
	r = dfList[1]
	ratio_q = float(q)/total
	ratio_r = float(r)/total
	ent = -1*((ratio_q * np.log2(ratio_q)) + (ratio_r * np.log2(ratio_r)))
	return ent

def getEntropy(col, tdata):

	data = []

	for attrib in tdata[col].unique():
		dfList = pd.value_counts(tdata[tdata[col] == attrib]['left'].values, sort=False).tolist()
		total = sum(dfList)
		if len(dfList) == 1:
			data.append(0)
		else:
			q = dfList[0]
			r = dfList[1]
			ratio_q = float(q)/total
			ratio_r = float(r)/total
			entropy = -1*((ratio_q * np.log2(ratio_q)) + (ratio_r * np.log2(ratio_r)))
			data.append(entropy)

	dfCount = pd.value_counts(tdata[col].values).tolist()
	total = sum(dfCount)

	weightedEntropy = 0.0

	for i in range(0,len(dfCount)):
		weightedEntropy += ((float(dfCount[i])/total)*data[i])

	return weightedEntropy

def getNode(category, tdata):
	ent = wholeEntropy(tdata)
	gain = []
	entropy = []
	attrib = {}

	for cat in category:
		e = getEntropy(cat, tdata)
		entropy.append(e)
		gain.append(ent-e)
		attrib[ent-e] = cat

	return attrib[max(gain)]

def buildTree(cur, attribs, level, tree=None):

	if len(cur['left'].unique()) <= 1:
		leaf = { 'leaf' : cur['left'].unique()[0] }
		return leaf
	elif len(cur) == 0 or len(attribs) == 0:
		leaf = { 'leaf' : np.unique(cur['left'])[np.argmax(np.unique(cur['left'], return_counts = True)[1])] }
		return leaf

	node = getNode(attribs, cur)
	attribs.remove(node)

	if tree is None:
		tree = OrderedDict()
		tree[node] = {}
		tree['max'] = np.unique(cur['left'])[np.argmax(np.unique(cur['left'], return_counts = True)[1])]
		tree['level'] = level

	for attrib in cur[node].unique():
		sub = cur.where(cur[node] == attrib).dropna()
		tree[node][attrib] = buildTree(sub, attribs[:], level+1)

	return tree

def dfs(row, tree, level):
    try:
        if list(tree.keys())[0] == 'leaf':
            return tree['leaf']
        if tree['level'] == level:
        	return tree['max']
        val = row[list(tree.keys())[0]]
        return dfs(row, tree[list(tree.keys())[0]][val], level)
    except:
        return 0


def predict(tree, data, level):
	predicted = []

	for index, row in data.iterrows():
		predicted.append(dfs(row, tree, level))

	return predicted


def getStats(predictions, actual):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(actual)):
        if actual[i] == predictions[i] and actual[i] == 1:
            tp += 1
        elif actual[i] == predictions[i] and actual[i] == 0:
            tn += 1
        elif actual[i] != predictions[i] and actual[i] == 0:
            fp += 1
        else:
            fn += 1

    total = (float(tn) + float(tp) + float(fp) + float(fn))
    accuracy = (float(tn) + float(tp))/total
    print 'Accuracy: ', accuracy
    return 1-accuracy

def preProcess(data):
	global num_cols, splitPoints

	for col in num_cols:
		data.sort_values(col, inplace=True)

		unique_max = []

		for val in np.unique(data[col]):
			uni = data[data[col] == val]
			unique_max.append([val, np.unique(uni['left'])[np.argmax(np.unique(uni['left'], return_counts = True)[1])]])

		label = 0

		prev = unique_max[0]
		row = 1

		for cur in unique_max[1:]:
			if prev[1] != cur[1]:
				splitPoints.append([(prev[0]+cur[0])/2, label])
				label += 1
			prev = cur
			row += 1

		i = 0

		row = []

		for val in data[col]:
	 		if i < len(splitPoints) and val > splitPoints[i][0]:
	 			i += 1
	 		row.append(i)

	 	data.drop(col, axis=1)
	 	data[col] = pd.Series(row).values

	return data

def process_validation(data):
	global num_cols, splitPoints

	for col in num_cols:
		data.sort_values(col, inplace=True)

		i = 0

		row = []

		for val in data[col]:
	 		if i < len(splitPoints) and val > splitPoints[i][0]:
	 			i += 1
	 		row.append(i)

	 	data.drop(col, axis=1)
	 	data[col] = pd.Series(row).values

	return data


if __name__ == '__main__':

	train = preProcess(train)

	validation = process_validation(validation)

	model = buildTree(train, cols, 0)

	target = validation['left'].tolist()

	depth = range(9)

	for i in depth:
		print "Depth: ", i
		pred_label = predict(model, validation, i)
		error.append(getStats(pred_label, target))

	plt.plot(depth, error)
	plt.show()