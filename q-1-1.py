import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
train = df.sample(frac=0.8, random_state=200)
validation = df.drop(train.index)
cat_col = ['salary', 'Work_accident', 'sales', 'promotion_last_5years']

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

def buildTree(cur, attribs, tree=None):

	if len(cur['left'].unique()) <= 1:
		leaf = { 'leaf' : cur['left'].unique()[0] }
		return leaf
	elif len(cur) == 0 or len(attribs) == 0:
		leaf = { 'leaf' : np.unique(cur['left'])[np.argmax(np.unique(cur['left'], return_counts = True)[1])] }
		return leaf

	node = getNode(attribs, cur)
	attribs.remove(node)

	if tree is None:
		tree = {}
		tree[node] = {}

	for attrib in cur[node].unique():
		sub = cur.where(cur[node] == attrib).dropna()
		tree[node][attrib] = buildTree(sub, attribs[:])

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

if __name__ == '__main__':
	model = buildTree(train, cat_col)

	pred_label = predict(model, validation)

	target = validation['left'].tolist()

	getStats(pred_label, target)