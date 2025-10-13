from typing import List




# 53. 最大子数组和
# 有负数, 得用一维动态规划(一维dp)(是不是也能用前缀和+hash?) 
# https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked
def maxSubArray(nums: List[int]) -> int:
    # cache[i]表示包含第i个元素的最大子数组和
    inf = -100000
    cache = [-inf] * (len(nums)+1)
    # 更新状态
    for i in range(len(nums)):
        cache[i+1] = max(cache[i] + nums[i], nums[i])
    
    return max(cache)




# 1143. 最长公共子序列(二维dp, 状态转移方程得找对) 
# https://leetcode.cn/problems/longest-common-subsequence/description/
def longestCommonSubsequence(text1: str, text2: str) -> int:
    # 初始化, cache[i][j]表示text1中[0, i-1]的子串与text2中[0, j-1]的子串的最长公共子序列 
    n, m = len(text1), len(text2)
    cache = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n):
        for j in range(m):
            # 3种情况: 当前字符相等时, 当前字符不等时(包含两个情况: 继续考虑两边分别减少一个字符时的最长公共子序列长度)
            cache[i+1][j+1] = max(int(text1[i] == text2[j]) + cache[i][j], cache[i][j+1], cache[i+1][j])
    # print(cache)
    return cache[-1][-1]




# 72. 编辑距离(除了初始化, 状态转移思路和上一题一样) 
# https://leetcode.cn/problems/edit-distance/description/
def minDistance(self, word1: str, word2: str) -> int:
    # 初始化, cache[i][j]表示word1中[0, i-1]的子串与word2中[0, j-1]的子串的编辑距离
    n, m = len(word1), len(word2)
    cache = [[0]*(m+1) for _ in range(n+1)]
    # 当一个字符为空另一个字符长度为i时，最短编辑距离就为i
    for i in range(n+1): cache[i][0] = i
    for i in range(m+1): cache[0][i] = i

    for i in range(n):
        for j in range(m):
            # 
            cache[i+1][j+1] = min(int(word1[i] != word2[j]) + cache[i][j], cache[i][j+1]+1, cache[i+1][j]+1)
    # print(cache)
    return cache[-1][-1]



# 300. 最长递增子序列(一维dp)
# https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2&envId=top-100-liked
def lengthOfLIS(nums: List[int]) -> int:
    # cache[i]表示当最长递增子序列长度为i时, 序列的最大值为多少
    inf = 10000
    cache = [-inf]

    for i in range(len(nums)):
        # 更新长度
        if nums[i] > cache[-1]:
            cache.append(nums[i])
        # 每次动态更新cache之前的值
        for j in range(len(cache)-1):
            if nums[i] > cache[j] and nums[i] < cache[j+1]:
                cache[j+1] = nums[i]
    # print(cache)
    return len(cache)-1



# 32. 最长有效括号(一维dp, 难在状态转移方程不好找) 
# https://leetcode.cn/problems/longest-valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked
def longestValidParentheses(s: str) -> int:
    # cache[i]表示包含第i-2个元素的最长有效括号的长度(前面多两个恒0元素是占位，防止还要多加条件判断避免越界)
    cache = [0] * (len(s)+2)
    
    for i in range(1, len(s)):
        # cache真实下标还要+2, 因为前面多了两个占位的
        I = i + 2
        if s[i] == ')':
            # 刚好和前面一个匹配的情况
            if s[i-1] == '(':
                # 之前的长度+当前匹配的一对
                cache[I] = cache[I-2] + 2
            # 匹配的部分还要再前面的情况
            elif s[i-1] == ')' and i - cache[I-1] - 1 >= 0 and s[i - cache[I-1] - 1] == '(':
                # 上一次的长度+当前匹配的一对+上上次(所有之前)的长度
                cache[I] = cache[I-1] + 2 + cache[I - cache[I-1] - 2]
    # print(cache)
    return max(cache)


# 1458. 两个子序列的最大点积
# https://leetcode.cn/problems/max-dot-product-of-two-subsequences/description/
def maxDotProduct(nums1: List[int], nums2: List[int]) -> int:
    inf = 10**10
    # dp[i][j]表示nums1从(0,i)索引的子序列与nums2从(0,j)的子序列的最大点积
    dp = [[-inf]*(len(nums2)+1) for _ in range(len(nums1)+1)]

    for i in range(len(nums1)):
        for j in range(len(nums2)):
            dot_ij = nums1[i] * nums2[j]
            dp[i+1][j+1] = max(dot_ij, dp[i][j]+dot_ij, dp[i+1][j], dp[i][j+1])
    
    return dp[-1][-1]






if __name__ == '__main__':
    print(maxDotProduct([-1,-1], [1,1]))