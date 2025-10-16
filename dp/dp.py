from typing import List, Optional

# DP三要素(个人总结)
# 怎么构造状态转移方程(即如何划分子问题)
# 怎么构造dp数组, dp[i][j]代表什么意思
# 如何遍历所有情况, 遍历顺序


class Solution:
    # 70.爬楼梯(一维dp, 简单)
    # https://leetcode.cn/problems/climbing-stairs/?envType=study-plan-v2&envId=top-100-liked
    def climbStairs(self, n: int) -> int:
        cache = []
        for i in range(1, n+1):
            if i == 1:
                cache.append(1)
            elif i == 2:
                cache.append(2)
            else:
                cache.append(cache[-1] + cache[-2])
    
        return cache[n-1]


    # 118. 杨辉三角(一维dp, 简单) 
    # https://leetcode.cn/problems/pascals-triangle/?envType=study-plan-v2&envId=top-100-liked
    def generate(self, numRows: int) -> List[List[int]]:
        cache = [[1]]
        for i in range(2, numRows+1):
            layer_cache = []
            for j in range(i):
                if j == 0 or j == len(cache[i-2]):
                    layer_cache.append(1)
                else:
                    layer_cache.append(cache[i-2][j-1] + cache[i-2][j])
            cache.append(layer_cache)
        return cache


    # 198. 打家劫舍(一维dp, 简单) 
    # https://leetcode.cn/problems/house-robber/description/
    def rob(self, nums: List[int]) -> int:
        cache = []
        for i in range(len(nums)):
            if i == 0:
                cache.append(nums[0])
            elif i == 1:
                cache.append(max(nums[0], nums[1]))
            else:
                cache.append(max(cache[-2]+nums[i], cache[-1]))
        
        return cache[-1]







    # 62. 不同路径(二维dp, 简单) 
    # https://leetcode.cn/problems/unique-paths/description/?envType=study-plan-v2&envId=top-100-liked
    def uniquePaths(self, m: int, n: int) -> int:
        # 初始化
        cache = [[0]*n for _ in range(m)]
        for i in range(n): cache[0][i] = 1
        for i in range(m): cache[i][0] = 1

        for i in range(1, m):
            for j in range(1, n):
                cache[i][j] = cache[i-1][j] + cache[i][j-1]

        return cache[-1][-1]


    # 64. 最小路径和(二维dp, 简单) 
    # https://leetcode.cn/problems/minimum-path-sum/description/
    def minPathSum(self, grid: List[List[int]]) -> int:
        # 初始化
        n, m = len(grid), len(grid[0])
        inf = 10 ** 10
        cache = [[inf]*(m+1) for _ in range(n+1)]
        cache[0][1] = cache[1][0] = 0
        
        for i in range(n):
            for j in range(m):
                cache[i+1][j+1] = grid[i][j] + min(cache[i][j+1], cache[i+1][j])
        # print(cache)
        return cache[-1][-1]



    


            














if __name__ == '__main__':
    sol = Solution()
    


