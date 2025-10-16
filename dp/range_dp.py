from typing import List

# 区间DP
# 线性DP: 状态转移一般基于前缀或后缀进行转移(子问题是前缀或后缀)
# 区间DP: 状态转移会从小区间转移到大区间(子问题是小区间, 小区间可能在大区间中间)
# 通常是从两侧向内缩小问题规模
# 细节:递推时, 先解决子区间, 再解决大区间, 此时需要根据状态转移方程考虑遍历顺序是正序还是逆序






# 5. 最长回文子串(二维dp, 状态转移方程得找对) 
# https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=study-plan-v2&envId=top-100-liked
def longestPalindrome(self, s: str) -> str:
    # 初始化, cache[i][j]表示第i到第j个字符的子串是否是回文串
    n = len(s)
    cache = [[True] * n for _ in range(n)]
    max_huiwen_len = 1
    max_huiwen_s = ''
    # 这题得逆序才能用上之前的结果:
    for i in range(n-1, -1, -1):
        for j in range(n-1, i-1, -1):
            # 状态转移 (第i个字符等于第j个字符且第i+1到第j-1个字符串也是回文串)
            cache[i][j] = s[i]==s[j] and (i==j or cache[i+1][j-1])
            # 更新最大回文长度
            if cache[i][j]:
                if max_huiwen_len <= j-i+1:
                    max_huiwen_len = j-i+1 
                    max_huiwen_s = s[i:j+1]
    # print(cache)
    return max_huiwen_s


# 516. 最长回文子序列(上一题的改版, 注意遍历顺序和转态转移方程)
# https://leetcode.cn/problems/longest-palindromic-subsequence/description/
def longestPalindromeSubseq(s: str) -> int:
    n = len(s)
    # dp[i][j]表示i到j的子序列中的最长回文子序列长度
    dp = [[0] * n for _ in range(n)]

    for i in range(n-1, -1, -1):
        # 第二层的遍历顺序得用正向遍历, 否则无法用到之前的结果
        for j in range(i, n):
            # 只有一个字符的串一定是回文串, 长度为1
            if i == j:
                dp[i][j] = 1
            # 头尾相等的串是其去头去尾后的子序列里的最长回文子序列长度+2
            elif s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            # 头尾不相等的串的尺度是去头的子序列里的最长长度或去尾子序列里的最长长度
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    # print(dp)
    return dp[0][-1]        



# 1039. 多边形三角剖分的最低得分 https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/
def minScoreTriangulation(values: List[int]) -> int:
    n = len(values)
    inf = 10 ** 10
    # dp[i][j]表示顶点i到顶点j(连续)构成的多边形的最低得分
    # 只需要考虑上三角(i<=j的情况)
    dp = [[inf] * n for _ in range(n)]

    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if i==j-1:
                # 构成的多边形是一条线的情况
                dp[i][j] = 0
            elif i<j:
                res = inf
                # 确定一条边后，遍历所有顶点(遍历所有可能三角形)
                for k in range(i+1, j):
                    res = min(res, dp[i][k] + dp[k][j] + values[i]*values[k]*values[j])
                dp[i][j] = res
                
    return dp[0][n-1]



# 矩阵连乘的最小计算量
# 解析: https://developer.aliyun.com/article/1626301
def mat_mul(mat_size):
    n = len(mat_size)
    # dp[i][j]表示[i,j]的矩阵连乘的最小计算量
    dp = [[0]*(n) for _ in range(n)]

    for i in range(n-1, -1, -1):
        for j in range(i, n):
            if i < j:
                tmp = []
                # [i,j]的矩阵连乘的最小计算量可由[i, k], [k+1, j]的最小计算量推出, 然后加上A_ik@Ak+1j的计算量(遍历k)
                for k in range(i, j):
                    tmp.append(dp[i][k] + dp[k+1][j] + mat_size[i][0]*mat_size[k][1]*mat_size[j][1])
                dp[i][j] = min(tmp)

    return dp[0][-1]




if __name__ == '__main__':
    l = [1,3,1,4,1,5]
    print(minScoreTriangulation(l))

        