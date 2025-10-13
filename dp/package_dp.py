# 背包问题DP
# 动态规划问题
# dp:   适用于求最优解, 问题具有重叠子问题(存储子问题的解, 在后续递归或递推时直接记忆化搜索, 避免重复计算) 通常用于寻找最优解 (通常用的递推, 自底向上搜索)
# 回溯: 适用于需要穷举所有可能解的问题，如排列、组合、子集(通常就用的递归, 自顶向下搜索)

# 递推+记忆化搜索: 本质是构造一个cache数组存储已经遍历到的结果, 下次遍历(递推, 且一般可以用循环解决)到这个结果直接调用cache里的值即可
# 记忆化搜索 可以1:1翻译为 递推



# 背包问题型dp (给定背包容量, 物体体积, 物体价值, 求得选哪些物体装入背包, 不会超重, 并且价值最大), 最终返回最大价值
# cache是二维的(cache[i][j], i表示装几个物品, j表示背包容量(注意顺序不能错))
# 01背包:每种物品只能选一次
# 完全背包:每种物品可以无限重复选
# 恰好装满型背包: 根据选最少或选最多, cache的初始化值为inf或-inf

# 不同的背包题型，大致的框架是一样的，不一样的可能在于初始化条件和状态转移方程
# 遇到"总数"类型, 一般在状态转移使用相加，cache初始化为0即可 (518. 零钱兑换 II)
# 遇到"最少或最多"类型, 一般在状态转移使用min或max，cache初始化看是恰好装满型背包还是至少型背包 (322. 零钱兑换)

from typing import List



# 01背包 / 完全背包 模板 (区别在于每个物品的数量是否 只有一个 / 有无穷个)
def package(n: int, values: List[int], weights: List[int], is_01, full) -> int:
    """
        n:       背包容量
        values:  物品价值
        weights: 物品重量
        is_01:   是否是01背包/完全背包
        full:    是否恰好装满 / 无需装满背包
    """
    '''1. 初始化(物品个数+1)x(背包容量+1)大小的cache'''
    if full:
        # 恰好装满的初始化方法
        inf = 10**10
        cache = [[-inf]*(n+1) for _ in range(len(values)+1)]
        cache[0][0] = 0
    else:
        cache = [[0]*(n+1) for _ in range(len(values)+1)]

    '''2. 遍历子问题, 外层物品个数, 内层当前容量'''
    for i in range(len(values)):
        for j in range(n+1):
            # 下面就只用判断装还是不装的情况:
            if j < weights[i]:
                cache[i+1][j] = cache[i][j]
            else:
                if is_01:
                    # 01背包转移函数
                    cache[i+1][j] = max(cache[i][j], cache[i][j-weights[i]] + values[i]) 
                else:
                    # 完全背包转移函数
                    cache[i+1][j] = max(cache[i][j], cache[i+1][j-weights[i]] + values[i]) 
    print(cache)
    return cache[-1][-1] if cache[-1][-1] > 0 else 0





# 279. 完全平方数(恰好装满完全背包) 
# 背包容量为n, 物品个数为N, 求背包被完全平方数恰好装满的最少数量
# https://leetcode.cn/problems/perfect-squares/description/
def numSquares(n: int) -> int:
    # 初始化
    N = int(n**0.5) + 1
    inf = 10**10
    cache = [[inf] * (n+1) for _ in range(N+1)]
    cache[0][0] = 0
    # 遍历子问题
    for i in range(1, N+1):
        v = i*i
        for j in range(n+1):
            if j < v:
                cache[i][j] = cache[i-1][j]
            else:
                cache[i][j] = min(cache[i-1][j], cache[i][j-v]+1)
    # print(cache)
    return cache[-1][-1] if cache[-1][-1] < inf else 0


# 322. 零钱兑换(恰好装满完全背包) 
# https://leetcode.cn/problems/coin-change/description/
def coinChange(coins: List[int], amount: int) -> int:
    # 初始化 cache[i][w]表示对前i个物品，总金额为w时, 能放的最少硬币个数
    inf = 10 ** 10
    cache = [[inf] * (amount+1) for _ in range(len(coins)+1)]
    cache[0][0] = 0

    for i in range(len(coins)):
        for j in range(amount+1):
            if j < coins[i]:
                cache[i+1][j] = cache[i][j]
            else:
                cache[i+1][j] = min(cache[i][j], cache[i+1][j-coins[i]]+1)
    # print(cache)   
    return cache[-1][-1] if cache[-1][-1] < inf else -1


# 416. 分割等和子集(恰好装满01背包)
# https://leetcode.cn/problems/partition-equal-subset-sum/description/
def canPartition(nums: List[int]) -> bool:
    sum = 0
    for num in nums: sum += num
    if sum % 2 == 1: return False

    # 转化为恰好装满01背包问题
    inf = 10 ** 10
    n = sum // 2
    cache = [[-inf] * (n+1) for _ in range(len(nums)+1)]
    cache[0][0] = 0
    for i in range(len(nums)):
        for j in range(n+1):
            if j < nums[i]:
                cache[i+1][j] = cache[i][j]
            else:
                cache[i+1][j] = max(cache[i][j], cache[i][j-nums[i]]+nums[i])

    # print(cache)
    return True if cache[-1][-1] > 0 else False




# 494. 目标和(可以转化为类似01背包的范式，但是状态转移方程不一样)
# https://leetcode.cn/problems/target-sum/description/
def findTargetSumWays(nums: List[int], target: int) -> int:
    # 运算后所能达到的最大绝对值
    max_sum = sum([abs(num) for num in nums])
    # 无法构造的情况:
    if target < -max_sum:
        return 0

    # cache[i][j]表示当包含前i个数时，target=j-maxsum的表达式的数量
    # target的范围在[-max_sum, target]
    cache = [[0]*(max_sum+target+1) for _ in range(len(nums)+1)]
    cache[0][0] = 1
    # 转化题目为(负号=不放, 加号=放, 对此，初始状态是全部为负号, 因此nums需要x2)
    nums = [num*2 for num in nums]

    for i in range(len(nums)):
        for j in range(max_sum+target+1):
            # 放不下(没办法从更小的target得到当前状态, 即总和达不到这么小)
            if j-nums[i] < 0:
                cache[i+1][j] = cache[i][j]
            else:
                # cache[i][j](当前元素是负号) + cache[i][j-nums[i]](当前元素是正号)
                cache[i+1][j] = cache[i][j] + cache[i][j-nums[i]]

    return cache[-1][-1]





# 474. 一和零
# 本质是01背包, 只不过物品有两个容量属性, 两个都能装得下才装得下背包
# https://leetcode.cn/problems/ones-and-zeroes/description/
def findMaxForm(strs: List[str], m: int, n: int) -> int:
    # 统计01数量
    nums = []
    for str in strs:
        is_1, is_0 = 0, 0
        for s in str:
            if s == '0': is_0 += 1
            if s == '1': is_1 += 1
        nums.append([is_0, is_1])
    print(nums)
    
    # dp[i][j]表示当m=i-1, n=j-1时的最大子集长度
    # dp (k+1, m+1, n+1), dp[k][m][n]表示当包含前k个strs时, m, n的情况下的最大子集 
    k = len(strs)
    # 注意写法, 不要出现浅拷贝问题
    dp = [[[0 for _ in range(n+1)] for _ in range(m+1)] for _ in range(k+1)]

    # 遍历物品
    for i in range(k):
        # 遍历背包容量
        for j in range(m+1):
            for k in range(n+1):
                v0, v1 = nums[i][0], nums[i][1]
                # 装不下
                if j < v0 or k < v1:
                    dp[i+1][j][k] = dp[i][j][k]
                else:
                    dp[i+1][j][k] = max(dp[i][j][k], dp[i][j-v0][k-v1]+1)
    
    return dp[-1][-1][-1]














if __name__ == '__main__':
    # print(package(11, [1,6,18,22,28], [1,2,5,6,7], is_01=True, full=False))
    # print(package(10, [6,18,22,28], [2,5,6,7], is_01=True, full=True))
    print(package(4, [15, 20, 30], [1, 3, 4], is_01=False, full=True))