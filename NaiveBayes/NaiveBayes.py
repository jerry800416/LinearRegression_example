import numpy as np
import re 
import random

def textParse(input_string):
    """
    將輸入文字進行分詞（以非單字字元為分隔），並轉成小寫。

    參數：
    - input_string: 原始字串（例如一篇電子郵件內容）

    回傳：
    - 一個詞彙列表，每個元素為轉小寫的單詞
    """
    listofTokens = re.split(r'\W+', input_string)
    return [tok.lower() for tok in listofTokens if len(tok) > 2]  # 過濾掉過短的詞

def creatVocablist(doclist):
    """
    根據所有訓練文件建立詞彙表（不重複）

    參數：
    - doclist: 一個文件清單，每個文件是單詞的 list

    回傳：
    - 所有出現過的詞組成的詞彙表 list
    """
    vocabSet = set([])
    for document in doclist:
        vocabSet = vocabSet | set(document)  # 聯集（避免重複）
    return list(vocabSet)

def setOfWord2Vec(vocablist, inputSet):
    """
    將輸入的單詞集合轉換為詞彙表上的向量（0-1表示）

    參數：
    - vocablist: 詞彙表（不重複的單詞 list）
    - inputSet: 一篇文件中的單詞 list

    回傳：
    - 一個與詞彙表長度相同的 0-1 向量，表示該單詞是否出現
    """
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec

def trainNB(trainMat, trainClass):
    """
    訓練朴素貝葉斯分類器（使用二元詞集模型）

    參數：
    - trainMat: 訓練文件向量矩陣，每一行是一篇文件的向量
    - trainClass: 對應每篇文件的標籤（0為正常郵件，1為垃圾郵件）

    回傳：
    - p0Vec: 在類別 0 下，每個單詞的對數機率向量
    - p1Vec: 在類別 1 下，每個單詞的對數機率向量
    - p1: 類別 1 的先驗機率
    """
    numTrainDocs = len(trainMat)       # 訓練樣本數
    numWords = len(trainMat[0])        # 每個樣本的特徵數（詞彙表大小）
    p1 = sum(trainClass) / float(numTrainDocs)  # 類別為 1 的先驗機率

    # 初始化為 1 是拉普拉斯平滑
    p0Num = np.ones((numWords))  
    p1Num = np.ones((numWords))  
    p0Denom = 2.0  # 分母初始為 2，避免除以 0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainClass[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])

    # 使用對數避免下溢問題
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, p1

def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    """
    使用訓練好的朴素貝葉斯模型進行分類判斷

    參數：
    - wordVec: 測試樣本的向量表示
    - p0Vec: 類別 0 的對數機率向量
    - p1Vec: 類別 1 的對數機率向量
    - p1_class: 類別 1 的先驗機率

    回傳：
    - 分類結果：0 表示正常郵件，1 表示垃圾郵件
    """
    p1 = np.log(p1_class) + sum(wordVec * p1Vec)
    p0 = np.log(1.0 - p1_class) + sum(wordVec * p0Vec)
    if p0 > p1:
        return 0
    else:
        return 1

def spam():
    """
    主函數，執行垃圾郵件分類流程（訓練＋測試）

    步驟：
    1. 讀取 spam/ham 資料
    2. 建立詞彙表與向量表示
    3. 隨機挑選 10 筆測試，其餘作為訓練
    4. 訓練朴素貝葉斯分類器
    5. 測試並計算錯誤率
    """
    doclist = []
    classlist = []

    for i in range(1, 26):
        # 讀取垃圾郵件
        wordlist = textParse(open('NaiveBayes/email/spam/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read())
        doclist.append(wordlist)
        classlist.append(1)  # 1 表示垃圾郵件

        # 讀取正常郵件
        wordlist = textParse(open('NaiveBayes/email/ham/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read())
        doclist.append(wordlist)
        classlist.append(0)  # 0 表示正常郵件

    vocablist = creatVocablist(doclist)  # 建立詞彙表
    trainSet = list(range(50))  # 編號 0~49，共 50 筆資料
    testSet = []  # 測試集索引列表

    # 隨機抽取 10 筆作為測試集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])

    trainMat = []
    trainClass = []

    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])

    # 訓練模型
    p0Vec, p1Vec, p1 = trainNB(np.array(trainMat), np.array(trainClass))

    # 測試模型
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocablist, doclist[docIndex])
        if classifyNB(np.array(wordVec), p0Vec, p1Vec, p1) != classlist[docIndex]:
            errorCount += 1
    print('目前10個測試樣本中，錯誤數量：', errorCount)

if __name__ == '__main__':
    spam()
