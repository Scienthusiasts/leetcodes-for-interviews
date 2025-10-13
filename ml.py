import numpy as np
import matplotlib.pyplot as plt
from collections import Counter





# KNN可优化的部分:
# 1.计算距离时采用kd树(1.选择方差最大的维度进行空间分割, 2.对左右子树递归构建, 3.从根节点开始, 根据查询点与分割超平面的距离, 递归搜索最近邻子树)
# 2.根据验证集自动搜索最优k值
# 3.采用加权投票输出结果(soft), 这样能输出概率

def knn(x, y, y_label, k):
    """
        x:       [2, ],  float
        y:       [n, 2], float
        y_label: [n]
    """
    x = x.reshape(1, 2)
    # 1.计算x与数据集中其余所有数据的距离(欧式距离, 余弦距离...)
    dist = np.sqrt(np.sum((y - x)**2, axis=1))
    # 2.根据k值取出距离最小的前k个数据对应的标签
    k_indices = np.argsort(dist)[:k]
    k_label = y_label[k_indices]
    # 3.出现次数最多的即作为x的标签
    pred, cnt_num = Counter(k_label).most_common(1)[0]
    return pred


def kmeans(x, k):
    """
        x:   [n, 2],  float
        k:   聚类中心数量   
    """
    # 1.随机选取k个聚类中心
    k_centers = np.random.randn(k,2)
    center_diff = 100
    while center_diff > 1e-4:
        # 2.将每条数据分配给距离最近的簇
        # 利用向量的平方和展开公式高效计算L2距离
        xx = np.sum(x * x, axis=1, keepdims=True)
        yy = np.sum(k_centers * k_centers, axis=1, keepdims=True)
        xy = x @ k_centers.T
        k_dists = np.sqrt(xx + yy.T - 2*xy)
        x_tmp_clusters = np.argmin(k_dists, axis=1)
        # 3.更新聚类中心
        center_diff = 0
        for i in range(k):
            i_x = x[x_tmp_clusters==i]
            new_center = i_x.mean(axis=0)
            center_diff += np.sqrt(np.sum((new_center - k_centers[i])**2))
            k_centers[i] = new_center
    # 4.当聚类中心不再变动, 认为收敛
    return x_tmp_clusters
    

    






if __name__ == '__main__':
    # 生成不同高斯分布的数据
    y1 = np.random.randn(50, 2)+np.array([2, -2])
    y1_label = np.zeros((50))
    y2 = np.random.randn(50, 2)+np.array([2, 2])
    y2_label = np.zeros((50)) + 1
    y3 = np.random.randn(50, 2)+np.array([0, -2])
    y3_label = np.zeros((50)) + 2
    y = np.concatenate([y1, y2, y3], axis=0)
    y_label = np.concatenate([y1_label, y2_label, y3_label], axis=0)
    plt.scatter(y[:,0], y[:,1], c=y_label)
    plt.show()


    '''knn'''
    # x = np.random.randn(2)+np.array([1, 1])
    # pred = knn(x, y, y_label, 10)
    # print(pred)

    '''kmeans'''
    y_pred_labels = kmeans(y, 6)
    plt.scatter(y[:,0], y[:,1], c=y_pred_labels)
    plt.show()
