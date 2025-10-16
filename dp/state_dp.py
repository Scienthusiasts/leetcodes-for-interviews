from typing import List

# 状态机DP






# 152. 乘积最大子数组(一维dp) 
# 有正有负，所以维护两个memo, 一个存储包含当前值的最大值, 另一个存储包含当前值的最小值
# https://leetcode.cn/problems/maximum-product-subarray/?envType=study-plan-v2&envId=top-100-liked
def maxProduct(nums: List[int]) -> int:
    # 维护一个最大最小cache, cache[i]代表包含第i个元素的[0, i]序列中的乘积最大值
    min_cache, max_cache = [nums[0]], [nums[0]]

    for i in range(1, len(nums)):
        min_cache.append(min(min_cache[i-1] * nums[i], max_cache[i-1] * nums[i], nums[i]))
        max_cache.append(max(min_cache[i-1] * nums[i], max_cache[i-1] * nums[i], nums[i]))        
    # print(max_cache)
    return max(max_cache)



# 122. 买卖股票的最佳时机 II (状态机dp, 有几个状态就定义几个cache)
# https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/
def maxProfit(prices: List[int]) -> int:
    # 初始化(hold_state_cache[i]表示第i天若持有股票的最大利润, sell_state_cache表示第i天若不持有股票的最大利润)
    hold_state_cache = [0] * (len(prices)+1)
    sell_state_cache = [0] * (len(prices)+1)
    # 第0天不可能持有股票, 设置利润为-inf
    hold_state_cache[0] = -10 ** 10

    for i in range(1, len(prices)+1):
        # 第i天不持有股票的最大利润由 第i-1天持有股票的最大利润卖出股票 或者 第i-1天不持有股票且啥也不做 产生
        sell_state_cache[i] = max(hold_state_cache[i-1] + prices[i-1], sell_state_cache[i-1])
        # 第i天持有股票的最大利润由 第i-1天不持有股票的最大利润买入股票 或者 第i-1天持有股票且啥也不做 产生
        hold_state_cache[i] = max(sell_state_cache[i-1] - prices[i-1], hold_state_cache[i-1])

    # print(hold_state_cache)
    # print(sell_state_cache)
    return sell_state_cache[-1]


# 188. 买卖股票的最佳时机 IV (多了交易次数限制, 也是状态机dp(二维), 有几个状态就定义几个cache)
# https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/description/
def maxProfit(k: int, prices: List[int]) -> int:
    # 初始化(hold_state_cache[i][j]表示第i天若持有股票且已交易j次的最大利润, sell_state_cache[i][j]表示第i天若不持有股票且已交易j次的最大利润)
    hold_state_cache = [[0] * (k+1) for _ in range(len(prices)+1)]
    sell_state_cache = [[0] * (k+1) for _ in range(len(prices)+1)]
    # 第0天不可能持有股票, 设置利润为-inf
    inf = 10 ** 10
    for i in range(k+1): hold_state_cache[0][i] = -inf

    for i in range(1, len(prices)+1):
        for j in range(1, k+1):
            # 第i天不持有股票且交易j次的最大利润由 第i-1天持有股票且交易j次的最大利润卖出股票 或者 第i-1天不持有股票且交易j次且啥也不做 产生
            sell_state_cache[i][j] = max(hold_state_cache[i-1][j] + prices[i-1], sell_state_cache[i-1][j])
            # 第i天持有股票且交易j次的最大利润由 第i-1天不持有股票且交易j-1次的最大利润买入股票 或者 第i-1天持有股票且交易j次且啥也不做 产生
            hold_state_cache[i][j] = max(sell_state_cache[i-1][j-1] - prices[i-1], hold_state_cache[i-1][j])

    # print(hold_state_cache)
    # print(sell_state_cache)
    return sell_state_cache[-1][-1]