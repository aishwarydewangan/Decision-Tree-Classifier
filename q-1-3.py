import pandas as pd
import numpy as np
from operator import itemgetter

df = pd.read_csv("data.csv")
train = df.sample(frac=0.8, random_state=200)
validation = df.drop(train.index)
num_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
cols = num_cols + ['Work_accident' , 'promotion_last_5years' , 'sales' , 'salary']
splitPoints = []

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

def wholeGiniIndex(tdata):
	dfList = pd.value_counts(tdata['left'].values, sort=False).tolist()
	total = sum(dfList)
	if len(dfList) == 0 or len(dfList) == 1:
		return 0
	q = dfList[0]
	r = dfList[1]
	ratio_q = float(q)/total
	ratio_r = float(r)/total
	g = 2*ratio_q*ratio_r
	return g

def getGiniIndex(col, tdata):
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
			g = 2*ratio_q*ratio_r
			data.append(g)

	dfCount = pd.value_counts(tdata[col].values).tolist()
	total = sum(dfCount)

	weightedGiniIndex = 0.0

	for i in range(0,len(dfCount)):
		weightedGiniIndex += ((float(dfCount[i])/total)*data[i])

	return weightedGiniIndex

def wholeMCR(tdata):
	dfList = pd.value_counts(tdata['left'].values, sort=False).tolist()
	total = sum(dfList)
	if len(dfList) == 0 or len(dfList) == 1:
		return 0
	q = dfList[0]
	r = dfList[1]
	ratio_q = float(q)/total
	ratio_r = float(r)/total
	m = min(ratio_q, ratio_r)
	return m

def getMCR(col, tdata):
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
			m = min(ratio_q, ratio_r)
			data.append(m)

	dfCount = pd.value_counts(tdata[col].values).tolist()
	total = sum(dfCount)

	weightedMCR = 0.0

	for i in range(0,len(dfCount)):
		weightedMCR += ((float(dfCount[i])/total)*data[i])

	return weightedMCR

def getNode(category, tdata, flag):
	ent = 0.0
	if flag == 1:
		ent = wholeEntropy(tdata)
	if flag == 2:
		ent = wholeGiniIndex(tdata)
	if flag == 3:
		ent = wholeMCR(tdata)
	gain = []
	entropy = []
	attrib = {}

	for cat in category:
		e = 0.0
		if flag == 1:
			e = getEntropy(cat, tdata)
		if flag == 2:
			e = getGiniIndex(cat, tdata)
		if flag == 3:
			e = getMCR(cat, tdata)
		entropy.append(e)
		gain.append(ent-e)
		attrib[ent-e] = cat

	return attrib[max(gain)]

def buildTree(cur, attribs, flag, tree=None):

	if len(cur['left'].unique()) <= 1:
		leaf = { 'leaf' : cur['left'].unique()[0] }
		return leaf
	elif len(cur) == 0 or len(attribs) == 0:
		leaf = { 'leaf' : np.unique(cur['left'])[np.argmax(np.unique(cur['left'], return_counts = True)[1])] }
		return leaf

	node = getNode(attribs, cur, flag)
	attribs.remove(node)

	if tree is None:
		tree = {}
		tree[node] = {}

	for attrib in cur[node].unique():
		sub = cur.where(cur[node] == attrib).dropna()
		tree[node][attrib] = buildTree(sub, attribs[:], flag)

	return tree

def dfs(query, tree):
    try:
        if list(tree.keys())[0] == 'leaf':
            return tree['leaf']
        val = query[list(tree.keys())[0]]
        return dfs(query, tree[list(tree.keys())[0]][val])
    except:
        return 0


def predict(tree, data):
	predicted = []

	for index, row in data.iterrows():
		predicted.append(dfs(row, tree))

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
    precision = float(tp)/(float(tp) + float(fp))
    recall = float(tp)/(float(tp) + float(fn))
    f1 = 2/((1/float(precision)) + (1/float(recall)))
    print 'True Positive: ', tp
    print 'True Negative: ', tn 
    print 'False Positive: ', fp
    print 'False Negative: ', fn
    print 'Accuracy: ', accuracy
    print 'Precision: ', precision
    print 'Recall: ', recall
    print 'F1 Score: ', f1

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

	target = validation['left'].tolist()

	flags = [1, 2, 3]

	impurity = ['Entropy', 'Gini Index', 'Misclassification Rate']

	for val in flags:
		print impurity[val-1] + ':'
		model = buildTree(train, cols, val)
		pred_label = predict(model, validation)
		getStats(pred_label, target)
		print ""
	