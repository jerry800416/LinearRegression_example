# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖時使用的字型為「微軟正黑體」，支援中文顯示
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示
from math import log
import operator


def createDataSet():
	'''
	函式：createDataSet()
	功能：建立訓練資料集與特徵名稱
	輸出：
	- dataSet：訓練資料（List of List）
	- labels：特徵名稱（List）
	'''
	dataSet = [[0, 0, 0, 0, 'no'],						
			   [0, 0, 0, 1, 'no'],
			   [0, 1, 0, 1, 'yes'],
			   [0, 1, 1, 0, 'yes'],
			   [0, 0, 0, 0, 'no'],
			   [1, 0, 0, 0, 'no'],
			   [1, 0, 0, 1, 'no'],
			   [1, 1, 1, 1, 'yes'],
			   [1, 0, 1, 2, 'yes'],
			   [1, 0, 1, 2, 'yes'],
			   [2, 0, 1, 2, 'yes'],
			   [2, 0, 1, 1, 'yes'],
			   [2, 1, 0, 1, 'yes'],
			   [2, 1, 0, 2, 'yes'],
			   [2, 0, 0, 0, 'no']]
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']		
	return dataSet, labels


def createTree(dataset, labels, featLabels):
	'''
	函式：createTree()
	功能：建立決策樹（遞迴）
	輸入：
	- dataset：資料集
	- labels：特徵名稱清單
	- featLabels：儲存使用過的特徵名稱
	輸出：
	- myTree：以字典結構表示的決策樹
	'''
	classList = [example[-1] for example in dataset]  # 取出所有類別（資料的最後一欄）
	if classList.count(classList[0]) == len(classList):  # 如果所有資料屬於同一類別，則直接回傳該類別（為葉節點）
		return classList[0]
	if len(dataset[0]) == 1:  # 如果只剩下一個特徵，無法再分裂 → 回傳票數最多的類別
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataset)  # 根據資訊增益選擇最佳分裂特徵
	bestFeatLabel = labels[bestFeat]
	featLabels.append(bestFeatLabel)
	myTree = {bestFeatLabel: {}}  # 建立樹的節點（字典形式）
	del labels[bestFeat]  # 將已用過的特徵從列表中刪除
	featValue = [example[bestFeat] for example in dataset]
	uniqueVals = set(featValue)  # 該特徵的所有唯一取值
	for value in uniqueVals:
		sublabels = labels[:]  # 複製特徵名稱列表
		# 遞迴建立子樹
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), sublabels, featLabels)
	return myTree


def majorityCnt(classList):
	'''
	函式：majorityCnt()
	功能：投票表決，回傳出現最多次的類別
	輸入：classList（所有類別標籤）
	輸出：出現次數最多的類別
	'''
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def chooseBestFeatureToSplit(dataset):
	'''
	函式：chooseBestFeatureToSplit()
	功能：選擇最佳特徵以進行分裂（根據資訊增益）
	輸入：dataset（資料集）
	輸出：最佳特徵的索引
	'''
	numFeatures = len(dataset[0]) - 1  # 不包含類別標籤
	baseEntropy = calcShannonEnt(dataset)  # 計算原始熵
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataset]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for val in uniqueVals:
			subDataSet = splitDataSet(dataset, i, val)
			prob = len(subDataSet) / float(len(dataset))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


def splitDataSet(dataset, axis, val):
	'''
	函式：splitDataSet()
	功能：根據指定特徵的取值，分割資料集
	輸入：
	- dataset：原始資料集
	- axis：要分割的特徵索引
	- val：特徵的值
	輸出：符合條件的子資料集（已去除該特徵）
	'''
	retDataSet = []
	for featVec in dataset:
		if featVec[axis] == val:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


def calcShannonEnt(dataset):
	'''
	函式：calcShannonEnt()
	功能：計算香農熵
	輸入：dataset（資料集）
	輸出：entropy（資訊不確定度）
	'''
	numExamples = len(dataset)
	labelCounts = {}
	for featVec in dataset:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts:
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numExamples
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


def getNumLeafs(myTree):
	'''
	函式：getNumLeafs()
	功能：計算決策樹的葉節點數量
	輸入：myTree（決策樹字典）
	輸出：葉節點數量
	'''
	numLeafs = 0
	firstStr = next(iter(myTree))
	secondDict = myTree[firstStr]
	for key in secondDict:
		if isinstance(secondDict[key], dict):
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs


def getTreeDepth(myTree):
	'''
	函式：getTreeDepth()
	功能：計算決策樹的最大深度
	輸入：myTree（決策樹字典）
	輸出：整棵樹的深度（最大路徑長）
	'''
	maxDepth = 0
	firstStr = next(iter(myTree))
	secondDict = myTree[firstStr]
	for key in secondDict:
		if isinstance(secondDict[key], dict):
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	'''
	函式：plotNode()
	功能：畫節點（含文字與箭頭）
	輸入：
	- nodeTxt：節點文字
	- centerPt：節點位置
	- parentPt：父節點位置
	- nodeType：節點樣式（判斷 or 葉節點）
	'''
	arrow_args = dict(arrowstyle="<-")
	font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=font)


def plotMidText(cntrPt, parentPt, txtString):
	'''
	函式：plotMidText()
	功能：畫連線中間的文字（分支條件）
	'''
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
	'''
	函式：plotTree()
	功能：遞迴畫出整棵樹
	'''
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")
	leafNode = dict(boxstyle="round4", fc="0.8")
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = next(iter(myTree))
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
	for key in secondDict:
		if isinstance(secondDict[key], dict):
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
	'''
	函式：createPlot()
	功能：初始化畫布並畫出整棵樹
	輸入：inTree（決策樹）
	'''
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5 / plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()


# ========================
# 主程式執行區
# ========================
if __name__ == '__main__':
	dataset, labels = createDataSet()
	featLabels = []  # 儲存使用過的特徵名稱
	myTree = createTree(dataset, labels, featLabels)
	createPlot(myTree)
