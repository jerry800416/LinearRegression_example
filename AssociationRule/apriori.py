# -*- coding: UTF-8 -*-
'''
實作 Apriori 演算法，從交易資料中找出頻繁項目集，並產生關聯規則
'''


def loadDataSet():
    """
    載入樣本交易資料集。
    每筆交易為一個列表，裡面包含若干個商品（以整數表示）。
    
    Output:
        dataSet: List[List[int]]，範例交易資料集
    """
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


def createC1(dataSet):
    """
    建立候選 1-項目集（C1），即所有出現在交易資料中的單一商品，且不重複。

    Input:
        dataSet: List[List[int]]，原始交易資料集
    
    Output:
        C1: List[frozenset]，候選 1-項目集，每個元素為 frozenset 單項目集合
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # 將每個列表轉為不可變集合 frozenset


def scanD(D, CK, minSupport):
    """
    掃描交易資料 D，計算每個候選項目集 CK 的支持度（support），並篩選出滿足最小支持度的頻繁項目集。

    Input:
        D: List[set]，轉為集合形式的交易資料集
        CK: List[frozenset]，候選 k-項目集
        minSupport: float，最小支持度門檻
    
    Output:
        retlist: List[frozenset]，滿足支持度門檻的頻繁項目集
        supportData: Dict[frozenset → float]，所有候選項目集的支持度
    """
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):  # 若候選 can 是交易 tid 的子集，代表它出現在這筆交易中
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(list(D)))
    retlist = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retlist.insert(0, key)
        supportData[key] = support
    return retlist, supportData


def aprioriGen(LK, k):
    """
    由上一階段的頻繁 (k-1)-項目集 LK 產生候選 k-項目集 CK。

    Input:
        LK: List[frozenset]，頻繁 (k-1)-項目集
        k: int，目標項目集的長度
    
    Output:
        retlist: List[frozenset]，候選 k-項目集
    """
    retlist = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 只有前 k-2 個項目相同才可以合併
                retlist.append(LK[i] | LK[j])  # 聯集產生新的候選項目集
    return retlist


def apriori(dataSet, minSupport=0.5):
    """
    Apriori 主函數：找出所有滿足支持度的頻繁項目集。

    Input:
        dataSet: List[List[int]]，原始交易資料集
        minSupport: float，最小支持度門檻
    
    Output:
        L: List[List[frozenset]]，頻繁項目集的清單，L[k] 對應 k+1 項目集
        supportData: Dict[frozenset → float]，每個項目集對應的支持度
    """
    C1 = createC1(dataSet)  # 建立初始候選 1-項目集
    D = list(map(set, dataSet))  # 將資料轉為集合形式
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:  # 若上一層有頻繁項目集，則繼續生成下一階層
        CK = aprioriGen(L[k-2], k)
        LK, supK = scanD(D, CK, minSupport)
        supportData.update(supK)
        L.append(LK)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.6):
    """
    產生關聯規則，從頻繁項目集中找出所有信賴度（confidence）大於 minConf 的規則。

    Input:
        L: List[List[frozenset]]，頻繁項目集
        supportData: Dict[frozenset → float]，每個頻繁項目集的支持度
        minConf: float，最小信賴度門檻
    
    Output:
        rulelist: List[Tuple]，格式為 (前項, 後項, confidence)
    """
    rulelist = []
    for i in range(1, len(L)):  # 從 2-項目集開始產生規則
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  # 建立後件單元素集合
            rulessFromConseq(freqSet, H1, supportData, rulelist, minConf)
    return rulelist


def rulessFromConseq(freqSet, H, supportData, rulelist, minConf=0.6):
    """
    遞迴產生關聯規則的輔助函數。嘗試從頻繁項目集中產生更長的後件。

    Input:
        freqSet: frozenset，頻繁項目集
        H: List[frozenset]，可能的後件候選集合
        supportData: Dict，支持度資料
        rulelist: List，儲存產生的規則
        minConf: float，最小信賴度
    """
    m = len(H[0])
    while len(freqSet) > m:
        H = calConf(freqSet, H, supportData, rulelist, minConf)
        if len(H) > 1:
            H = aprioriGen(H, m + 1)  # 擴展後件集合的長度
            m += 1
        else:
            break


def calConf(freqSet, H, supportData, rulelist, minConf=0.6):
    """
    計算所有候選規則的信賴度，並篩選出滿足 minConf 的規則。

    Input:
        freqSet: frozenset，完整的頻繁項目集
        H: List[frozenset]，候選後件集合
        supportData: Dict，支持度資料
        rulelist: List，儲存輸出的規則
        minConf: float，最小信賴度
    
    Output:
        prunedH: List[frozenset]，保留下來的後件集合（可進一步生成更長規則）
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            rulelist.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH



if __name__ == '__main__':

    dataSet = loadDataSet()
    L, support = apriori(dataSet, minSupport=0.5)
    
    print('\n🔹 頻繁項目集：')
    for i, freq in enumerate(L):
        print(f'項數 {i+1}: {freq}')
    
    print('\n🔹 關聯規則（信賴度 ≥ 0.5）：')
    rules = generateRules(L, support, minConf=0.5)
