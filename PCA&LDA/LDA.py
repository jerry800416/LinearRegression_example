# -*- coding: utf-8 -*-
"""
📘 本程式說明：
本程式示範如何以 LDA（Linear Discriminant Analysis）實作降維流程，使用 iris 鳶尾花資料集。
過程包含：
1. 資料預處理與標準化
2. 類內/類間散度矩陣計算
3. 廣義特徵值分解
4. 手動與 sklearn 實作的 LDA 比較
5. 原始空間與降維後空間的視覺化呈現
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定中文字型為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False   # 正確顯示負號
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def plot_step_lda():
    """
    函數功能：視覺化原始資料在 X[0]-X[1] 平面上的分布
    輸入：無（直接使用全域變數 X, y）
    輸出：matplotlib 視覺化圖形
    """
    ax = plt.subplot(111)
    for label, marker, color in zip(range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(X[y == label, 0], X[y == label, 1],
                    marker=marker, color=color, alpha=0.5,
                    label=label_dict[label])
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.title('原始資料空間')
    plt.legend(loc='upper right', fancybox=True).get_frame().set_alpha(0.5)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_step_lda2():
    """
    函數功能：視覺化降維後 LDA 空間的分布（LD1 vs LD2）
    輸入：無（使用全域變數 X_lda, y）
    輸出：matplotlib 視覺化圖形
    """
    ax = plt.subplot(111)
    for label, marker, color in zip(range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1],
                    marker=marker, color=color, alpha=0.5,
                    label=label_dict[label])
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA 降維後空間')
    plt.legend(loc='upper right', fancybox=True).get_frame().set_alpha(0.5)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_scikit_lda(X, title):
    """
    函數功能：繪製 sklearn 計算的 LDA 結果
    輸入：
        X：降維後資料（2 維）
        title：圖形標題
    輸出：matplotlib 視覺化圖形
    """
    ax = plt.subplot(111)
    for label, marker, color in zip(range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(X[y == label, 0], X[y == label, 1] * -1,
                    marker=marker, color=color, alpha=0.5,
                    label=label_dict[label])
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
    plt.legend(loc='upper right', fancybox=True).get_frame().set_alpha(0.5)
    plt.grid()
    plt.tight_layout()
    plt.show()
    


if __name__ == '__main__':

    # ========================================
    # Step 1. 定義欄位與標籤對照表
    # ========================================
    feature_dict = {i: label for i, label in enumerate([
        'sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm'
    ])}

    label_dict = {i: label for i, label in zip(
        range(1, 4), ('Setosa', 'Versicolor', 'Virginica')
    )}

    # ========================================
    # Step 2. 讀取資料並指派欄位名稱
    # ========================================
    df = pd.read_csv('PCA&LDA/iris.data')
    df.columns = [feature_dict[i] for i in range(4)] + ['class label']
    print("這是資料前五筆:\n", df.head())

    # ========================================
    # Step 3. 分離特徵與標籤，並轉為數值編碼
    # ========================================
    X = df.iloc[:, 0:4].values
    y = df['class label'].values
    y = LabelEncoder().fit_transform(y) + 1  # 轉為 1~3 數值

    # ========================================
    # Step 4. 計算各類別的平均向量（類別中心）
    # ========================================
    mean_vectors = []
    np.set_printoptions(precision=4)  # 小數點顯示位數
    for cl in range(1, 4):
        mean_vec = np.mean(X[y == cl], axis=0)
        mean_vectors.append(mean_vec)
        print(f"類別 {cl} 的特徵均值向量:\n", mean_vec)

    # ========================================
    # Step 5. 類內散度矩陣 S_W 計算
    # ========================================
    S_W = np.zeros((4, 4))
    for cl, mv in zip(range(1, 4), mean_vectors):
        class_sc_mat = np.zeros((4, 4))
        for row in X[y == cl]:
            row, mv = row.reshape(4, 1), mv.reshape(4, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    print("類內散度矩陣 S_W:\n", S_W)

    # ========================================
    # Step 6. 類間散度矩陣 S_B 計算
    # ========================================
    overall_mean = np.mean(X, axis=0).reshape(4, 1)
    S_B = np.zeros((4, 4))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i + 1].shape[0]
        mean_vec = mean_vec.reshape(4, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    print("類間散度矩陣 S_B:\n", S_B)

    # ========================================
    # Step 7. 廣義特徵值分解 S_W^(-1) S_B
    # ========================================
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    for i in range(len(eig_vals)):
        print(f"\n特徵向量 {i+1}：\n", eig_vecs[:, i].real.reshape(4, 1))
        print(f"特徵值 {i+1}：{eig_vals[i].real:.4e}")

    # ========================================
    # Step 8. 特徵值排序與百分比貢獻
    # ========================================
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)

    print("特徵值排序（由大到小）:")
    for val, _ in eig_pairs:
        print(val)

    eigv_sum = sum(eig_vals)
    print("各特徵值的解釋變異百分比:")
    for i, pair in enumerate(eig_pairs):
        print(f"第 {i+1} 主成分：{(pair[0]/eigv_sum).real:.2%}")

    # ========================================
    # Step 9. 選取前 2 維 LDA 主成分
    # ========================================
    W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
    print("投影矩陣 W:\n", W.real)

    # ========================================
    # Step 10. 降維處理（X × W）
    # ========================================
    X_lda = X.dot(W)
    print("降維後的資料形狀:", X_lda.shape)

    # ========================================
    # Step 11. 函數：原始資料的前兩維視覺化
    # ========================================

    plot_step_lda()

    # ========================================
    # Step 12. 函數：LDA 降維後視覺化
    # ========================================

    plot_step_lda2()

    # ========================================
    # Step 13. 使用 scikit-learn 進行 LDA
    # ========================================
    sklearn_lda = LDA(n_components=2)
    X_lda_sklearn = sklearn_lda.fit_transform(X, y)

    # ========================================
    # Step 14. 函數：sklearn LDA 結果視覺化
    # ========================================

    plot_scikit_lda(X_lda_sklearn, title='sklearn 實作的 LDA 投影結果')
