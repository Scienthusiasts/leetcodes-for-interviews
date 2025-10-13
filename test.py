import sys
from typing import List
from collections import defaultdict



# 老张爱好爬山，不过老张认为太过频繁的爬山对膝盖不太好。 
# 老张给自己定了一个规则，原则上只能每隔一天爬山一次，如果今天爬山了，
# 那么明天就休息一天不爬山了。 但老张认为凡事都有例外，所以他给了自己k次机会，
# 在昨天已经爬山的情况下，今天仍然连续爬山！ 换句话说就是老张每天最多爬山一次，
# 原则上如果昨天爬山了那么今天就不爬山，但有最多 k 次机会打破这一原则(注意也没说一定要隔一天爬一次)
# 输入描述：
# 第一行两个整数n和k，表示老张正在计划未来n天的爬山计划以及k次打破原则的机会。
# 第二行n个整数[a1, ...,an]，其中ai表示接下来第ai天如果进行爬山可以获得的愉悦值。
# 输出描述
# 输出一行一个整数，表示老张能在最佳爬山计划下获得的愉悦值之和。
# (状态机DP)
def climb(n, k, happiness:List) -> int:
    # cache_1[i][j]表示第i天且还有j次机会的前提下爬山能够获得的最大愉悦值(cache_0则表示不爬能够获得的最大愉悦值)
    # cache[i][j] = max(cache[i-2][j], cache[i-1][j+1]) + happiness[i]
    # i从下标2开始, j从下标1开始
    cache_1 = [[0] * (k+2) for _ in range(len(happiness)+2)]
    cache_0 = [[0] * (k+2) for _ in range(len(happiness)+2)]
    # 外层遍历天数, 内层遍历机会数
    for i in range(n):
        for j in range(k, -1, -1):
            # 今天爬                    前一天爬        前一天不爬
            cache_1[i+2][j+1] = max(cache_1[i+1][j], cache_0[i+1][j+1]) + happiness[i]
            # 今天不爬                  前一天爬        前一天不爬 
            cache_0[i+2][j+1] = max(cache_1[i+1][j+1], cache_0[i+1][j+1])
    # print(cache_0)
    # print(cache_1)
    return max(max(row) for row in cache_1+cache_0)


        
            
# 这天，小红薯在小红书上看到了一道每日一题之编程题，如下：
# [[引用开始]] 定义一个字符串是包裹字符串为：字符串的首字母等于最后一个字母。 求解一个字符串的全部子串中，有多少个不是包裹字符串。 [[引用结束]]
# 小红薯在评论区看到了这个问题的进阶版：
# 对于给定的字符串对于每一个前缀依次求解，它的全部非空子串中有多少个不是包裹字符串。小红薯觉得这个题目很有趣，所以她决定写一个程序，来解决这个问题。
# (hash)
def wrapStrNum(s:str) -> None:
    # cache[i] 表示到[0, i-1]的字符串的不是包裹字符串的个数
    cache = [0] * (len(s)+1)

    hash = defaultdict(int)
    for i in range(len(s)):
        cur_num = 0
        # 用hash代替循环:
        cur_num += i - hash[s[i]]
        cache[i+1] = cache[i] + cur_num
        print(cache[i+1])
        hash[s[i]] += 1


# 在小红书“品牌创意工坊”中，营销人员可以为直播和短视频活动创建定制化丝带AR特效，结合品牌ID与礼盒包装场景，实现动态丝带动画。
# 为了支撑亿级日活的前端渲染，后端需要在活动发布时预先计算并缓存所有可能的切割方案数，确保小程序组件和Web端秒级响应。
# 现有一根虚拟丝带长度为k，可以将其分割成若干段或保持一整段不动，但是每段长度只能取整数a、b或c中的一个，且不允许任何长度为a的段后面直接跟随长度为c的段。
# 请对所有长度k(1~n)，统计合法的切割方案数，供小红书前端组件批量加载与渲染。由于答案可能很大，请将答案对(10^9+7)取模后输出。顺序不同视为不同方案。
# 这题可以转化为背包问题, 但是注意，经典的背包问题模板的状态转移方程是(最大价值/装的最多的数量), 而这题是(方案总数)
# 并且这题方案总数是排列问题(装入背包次序不同但是物品一样也算不同的方案), 而经典背包是组合问题, 因此这题要先遍历容量再遍历物品

def maxCutNum(n, a, b, c) -> list:
    # dp[i][j] 表示使用前i种长度切割长度为j的丝带的方案数
    v = [a, b, c]
    dp = [[0] * (n + 1) for _ in range(len(v)+1)]
    
    # 初始化
    for i in range(4):
        dp[i][0] = 1
    
    # 先遍历容量，再遍历物品（这样才能考虑顺序）
    for j in range(n+1):
        for i in range(len(v)):
            if j < v[i]:
                dp[i+1][j] = dp[i][j]
            else:
                # 使用当前长度（注意这里要用dp[len(v)]，因为可以混合使用所有长度）
                dp[i+1][j] = dp[i][j] + dp[len(v)][j - v[i]]
    # 返回长度为1到n的方案数
    print(dp)



# 小明接融到了一款城市建设的游戏。在这个游戏玩家需要建造很多的设施。
# 其中一些设施可以提供电力，而另一些则会消耗电力，每种设施均只能建造至多一个。
# 目建造设施还需要花费一定的资金。如果某一时刻剩余电量(即所有发电设施产生的电力减去其他设施消耗的电力)恰好为1，则可以获得《高手电量》这一稀有成就。
# 小明现在希望获得这一成就，请你帮他计算，如何才能在花费最少资金的情况下达成这一成就

# 输入的第一行包含一个数n(1<=n<=3000)，表示小明可以建造n种不同的设施。
# 接下来的n行，每行包括两个整数ai, bi，表示建造第i种设施可以带来 ai的电力(如果 ai<0 则表示消耗电力)，但需花费bi的金额建造
# 如果无法做到剩余电量为1，则输出-1;如果可以，则输出所需花费的最小资金。

def minCostToAchievePower(n, facilities:List) -> int:
    # dp[i][j]表示使用前i种设施且电量为j时的最小花费
    offset = 3000  # 电量偏移量，防止负数索引
    dp = [[float('inf')] * (6001) for _ in range(n + 1)]
    dp[0][offset] = 0  # 初始状态，电量为0，花费为0

    for i in range(1, n + 1):
        power, cost = facilities[i - 1]
        for j in range(6001):
            # 建造第i种设施
            dp[i][j] = dp[i - 1][j]
            # 不建造/建造第i种设施
            if 0 <= j - power < 6001:
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - power] + cost)

    return dp[n][offset + 1] if dp[n][offset + 1] != float('inf') else -1



# 使用若干根长度不同的木棒, 请判断是否可以使用所有木棒恰好拼成一个正方形
# 输入样例
# 3
# 4 1 1 1 1
# 5 10 20 30 40 50
# 8 1 7 2 6 4 4 3 5
# 输出样例
# yes
# no
# yes
def can_form_square(l, idx):
    total_sum = sum(l)
    # 不能被4整除则组成不了正方形
    if total_sum % 4 != 0:
        print("no")
        return
     
    square_len = total_sum // 4
    max_len = max(l)
    # 最大棍子长度比正方形边长还长也组成不了正方形
    if max_len > square_len:
        print('no')
        return
    
    # 回溯
    group = [0] * 4
    if dfs(l, idx, group, square_len):
        print('yes')
    else:
        print('no')

    
def dfs(l, idx, group, square_len):
    # 遍历到最后一个棍子
    if idx >= len(l):
        # 恰好组成正方形:
        if all(i==square_len for i in group):
            return True
        else:
            return False
    
    # 回溯法遍历所有情况
    for i in range(4):
        if group[i] + l[idx] <= square_len:
            group[i] += l[idx]
            can_form = dfs(l, idx+1, group, square_len)
            group[i] -= l[idx]
            if can_form==False:
                return False
    return True




# 有一段长度为n的木材, 给定每种长度售卖价格，给出价值最大化的售卖方式(切割方式)和售卖价格:
# 例如, n=5, price=[1,4,5,7,8], 代表长度为1、2、3，4，5的木材售价分别为1，4，5，7，8、
# 那么最大售卖方式为[2, 3]或[1,2,2], 售卖价格为9
# (恰好装满完全背包, 额外多了需要输出最优选择记录)
def woodPackage(n, price):
    w = [i+1 for i in range(len(price))]
    # cache[i][j]表示当包含前i种售卖方式时，长度为j的木材能售出的最大价格
    inf = 10**10
    cache = [[-inf] * (n+1) for _ in range(len(price)+1)]
    cache[0][0] = 0
    # cutLen[i]表示木材长度为i时新增的切割长度
    cutLen = [-1] * (n+1)

    for i in range(len(price)):
        for j in range(n+1):
            # 不能切的情况:
            cache[i+1][j] = cache[i][j]
            # 能切的情况
            if j >= i:
                if cache[i+1][j-i-1]+price[i] > cache[i][j]:
                    cache[i+1][j] = cache[i+1][j-i-1]+price[i]
                    # 记录木材长度为j时允许的切割长度i+1
                    cutLen[j] = i+1

    # 回溯的找到最优切割长度记录:
    cut = []    
    while n > 0:
        cut.append(cutLen[n])
        n -= cutLen[n]
    return cut, cache[-1][-1]

    

# 767. 重构字符串 https://leetcode.cn/problems/reorganize-string/
# 贪心法, 每次都选和上一个元素不一样的数量最多的元素
def reorganizeString(s: str) -> str:

    if len(s) == 1:
        return s
    # 统计字频
    hash_dict = defaultdict(int)
    for c in s:
         hash_dict[c] += 1

    if len(hash_dict) == 1:
        return ''

    # key:按什么关键字排序, reverse=True:从大到小排序
    hash_dict = sorted(hash_dict.items(), key=lambda x: x[1], reverse=True)
    hash_dict = [list(v) for v in hash_dict]
    res = hash_dict[0][0]
    hash_dict[0][1] -= 1
    # 调整最大值顺序
    i = 1
    while i<len(hash_dict) and hash_dict[0][1] < hash_dict[i][1]:
        i+=1
    hash_dict[0], hash_dict[i-1] = hash_dict[i-1], hash_dict[0]
    for _ in range(1, len(s)):
        print(res)
        if hash_dict[0][0] != res[-1]:
            res += hash_dict[0][0]
            hash_dict[0][1] -= 1
            # 调整最大值顺序
            i = 1
            while i < len(hash_dict) and hash_dict[0][1] < hash_dict[i][1]:
                i+=1
            hash_dict[0], hash_dict[i-1] = hash_dict[i-1], hash_dict[0] 
        else:
            if hash_dict[1][1] == 0:
                return ''
            res += hash_dict[1][0]
            hash_dict[1][1] -= 1
            # 调整最大值顺序
            i = 2
            while i <len(hash_dict) and hash_dict[1][1] < hash_dict[i][1]:
                i+=1
            hash_dict[1], hash_dict[i-1] = hash_dict[i-1], hash_dict[1] 
    
    return res








if __name__ == '__main__':
    # 读取命令行输入
    # lines = sys.stdin.read().strip().splitlines()

    '''1'''
    # 5 1
    # 8 9 2 1 4
    # 7 1
    # 7 6 5 4 3 2 1
    # n, k = [int(i) for i in lines[0].split(' ')]
    # happiness = [int(i) for i in lines[1].split(' ')]
    # print(climb(n, k, happiness))

    '''2'''
    # n = int(lines[0])
    # s = lines[1]
    # wrapStrNum(s)

    '''3'''
    # N = int(lines[0])
    # for i in range(1, N+1):
    #     k, a, b, c = [int(j) for j in lines[i].split(' ')]
    #     maxCutNum(k, a, b, c)


    '''4'''
    # facilities = [[8,1], [-4, 2], [-2, 3], [-1, 4]]
    # n=len(facilities)
    # print(minCostToAchievePower(n, facilities))

    '''5'''
    # lines = sys.stdin.read().strip().splitlines()

    # for line in lines[1:]:
    #     line = [int(l) for l in line.split(' ')]
    #     l = line[1:]
    #     can_form_square(l, 0)

    '''6'''
    # print(woodPackage(5, [1,4,5,7,8]))


    '''7'''
    print(reorganizeString(s='aab'))






