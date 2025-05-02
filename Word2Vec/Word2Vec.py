# -*- coding: UTF-8 -*-
'''
Word2Vec Skip-Gram with Negative Sampling 模型完整實作（TensorFlow 1.x）
使用 text8 資料集訓練詞向量模型，並可透過 TensorBoard projector 可視化詞向量
模型架構使用 Skip-Gram，損失函數使用 NCE（Negative Sampling）
'''
from __future__ import absolute_import, division, print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # Python2/3 相容的迴圈函式
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 禁用 TensorFlow 2.x 的行為，使用 1.x 的 API
from tensorboard.plugins import projector  # TensorBoard 內嵌向量視覺化工具
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft JhengHei')  # 設定繪圖使用的字型為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 設定負號可以正確顯示


def maybe_download(filename, expected_bytes):
    """
    檢查是否已下載資料集，若無則下載，並檢查檔案大小是否一致
    輸入:
        - filename: 檔案名稱
        - expected_bytes: 預期大小（用來驗證是否正確下載）
    輸出:
        - local_filename: 實際儲存檔案路徑
    """
    url = 'http://mattmahoney.net/dc/'
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('已找到並驗證：', filename)
    else:
        raise Exception('檔案大小驗證失敗：' + local_filename)
    return local_filename


def read_data(filename):
    """
    解壓 zip 並將檔案轉為字詞 list
    輸入: 
        - filename: zip 檔案路徑
    輸出: 
        - data: 所有字詞的 list
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """
    將詞語轉成數值 ID，並建立詞頻字典
    輸入：
        - words: 詞語 list
        - n_words: 保留前 n_words 常見詞
    輸出：
        - data: 對應詞的 ID list
        - count: 詞頻統計列表
        - dictionary: 詞 → ID 映射
        - reverse_dictionary: ID → 詞 映射
    """
    count = [['UNK', -1]]  # UNK 用於替代罕見詞
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    """
    根據 Skip-Gram 概念，從語料資料中產生訓練 batch
    輸入：
        - batch_size: 每批資料筆數
        - num_skips: 每個中心詞產生幾個 context（通常為 2）
        - skip_window: 可選取的 context 範圍大小（左右詞數）
    輸出：
        - batch: 中心詞（輸入）
        - labels: 上下文詞（輸出）
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # 總範圍
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename):
    """
    使用 matplotlib 將嵌入向量視覺化（2D）
    輸入：
        - low_dim_embs: 經 TSNE 降維的向量
        - labels: 詞語標籤
        - filename: 輸出圖檔名稱
    """
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


if __name__ == '__main__':
    
    # -------------------------- 參數設定與資料夾準備 --------------------------
    # 指定 TensorBoard log 儲存位置（預設為當前目錄下的 log 資料夾）
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='TensorBoard 檔案儲存的目錄')
    FLAGS, unparsed = parser.parse_known_args()

    # 如果資料夾不存在則建立
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # -------------------------- Step 1: 下載 text8 語料 --------------------------
    filename = maybe_download('text8.zip', 31344016)
    # 讀取 zip 中的文本檔案
    vocabulary = read_data(filename)
    print('語料字數:', len(vocabulary))

    # -------------------------- Step 2: 建立詞典與數值化語料 --------------------------
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary  # 節省記憶體
    print('最常見詞：', count[:5])
    print('樣本資料：', data[:10], [reverse_dictionary[i] for i in data[:10]])
    data_index = 0  # 資料讀取位置指標

    # -------------------------- Step 3: 產生 Skip-Gram 訓練資料 --------------------------
    # 確認樣本輸出
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
  
    # -------------------------- Step 4: 建立 Skip-Gram + Negative Sampling 模型 --------------------------
    # 訓練參數
    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 2
    num_sampled = 64  # 負樣本數量

    # 驗證資料（用來觀察相似詞）
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # 詞向量矩陣
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # NCE（Negative Sampling）參數
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                        stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # NCE 損失函數（自動抽樣負樣本）
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # 將詞向量正規化以便於可視化
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    # -------------------------- Step 5: 執行訓練 --------------------------
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
        init.run()
        print('模型初始化完成')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict)
            average_loss += loss_val
            writer.add_summary(summary, step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Step', step, '平均損失:', average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = '與 %s 最相近的詞：' % valid_word
                    log_str += ', '.join(reverse_dictionary[nearest[k]] for k in range(top_k))
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()

        # 儲存 metadata
        with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

        # TensorBoard 可視化設定
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    # -------------------------- Step 6: 將嵌入視覺化為圖像 --------------------------
    filename = maybe_download('text8.zip', 31344016)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, 'Word2Vec/tsne.png')


