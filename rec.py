from typing import List, Set

# 回溯专题 递归(原问题->子问题), 本质上是dfs
# 什么时候用回溯(当n较小时) (因为回溯本质还是暴力枚举, 时间复杂度高)

# 子集型回溯(每个结点选/不选)
# 子集型回溯有点像对一颗k叉树进行遍历, 当然, 这个树是抽象出来的, 实际可能并没有树
# 子集型回溯的几个要素: 记录的路径, 当前遍历到的位置(当前深度), 每次的分支逻辑 
# 子集型回溯如果没有重复元素，改用[二进制枚举(还没看)]可能写起来更快

# 组合型回溯
# 简单来说, 如果子集回溯是选所有情况, 那么组合型回溯就是在子集型回溯的基础上，增加某些判断逻辑再筛选出符合要求的即可

# 排列型回溯
# 每次根据数组大小都有n种情况(n个分支)
# 全排列 (和组合型回溯的区别在于排列型回溯元素一样顺序不一样也算一种)






# 46. 全排列(排列型回溯)
# https://leetcode.cn/problems/permutations/description/?envType=study-plan-v2&envId=top-100-liked
def permute(nums: List[int]) -> List[List[int]]:
    all_traj = []
    # 回溯就相当于dfs遍历一个虚拟的树
    def dfs(nums: Set[int], traj):
        if len(nums)==0:
            # 注意深拷贝
            all_traj.append(traj[:])
        else:
            # 类似n叉树的遍历
            for i in list(nums):
                nums.remove(i)
                traj.append(i)
                dfs(nums, traj)
                # 恢复原状
                nums.add(i)
                traj.pop(-1)

    dfs(set(nums), [])
    return all_traj

        


# 78. 子集(子集型回溯, 选或不许拿)
# https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked
def subsets(nums: List[int]) -> List[List[int]]:
    all_traj = []
    def dfs(nums, i, traj):
        if i == len(nums):
            all_traj.append(traj[:])
        else:
            # 不选当前元素
            dfs(nums, i+1, traj)
            # 选当前元素
            traj.append(nums[i])
            dfs(nums, i+1, traj)
            traj.pop(-1)
    
    dfs(nums, 0, [])
    return all_traj




# 17. 电话号码的字母组合(排列问题变种)
def letterCombinations(digits: str) -> List[str]:
    num_dict = {2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9:'wxyz'}
    all_traj = []
    def dfs(digits, i, traj):
        if i == len(digits):
            all_traj.append(traj)
        else:
            num = ord(digits[i]) - ord('0')
            for s in num_dict[num]:
                traj += s
                dfs(digits, i+1, traj)
                traj = traj[:-1]
        
    dfs(digits, 0, '')
    return all_traj





# 39. 组合总和(排列问题变种)
# https://leetcode.cn/problems/combination-sum/description/?envType=study-plan-v2&envId=top-100-liked
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    all_traj = []
    def dfs(nums, i, sum, traj):
        if sum == target:
            all_traj.append(traj[:])
        elif sum < target:
            # 每次都从i位置开始遍历, 既能保证重复选, 又能避免排列重复的问题:
            for j in range(i, len(nums)):
                sum += nums[j]
                traj.append(nums[j])
                dfs(nums, j, sum, traj)
                sum -= nums[j]
                traj.pop(-1)
        
    dfs(candidates, 0, 0, [])
    return all_traj



# 将nums中间插入(+, -, 或''表示和前面一个数拼在一起), 求得结果等于target的序列
def combineExp(nums, target):
    all_traj = []

    def dfs(nums, i, sum, traj):
        if i == len(nums) and target == sum:
            all_traj.append(traj[:])
        elif i<len(nums):
            # +号
            sum += nums[i]
            traj.append(nums[i])
            dfs(nums, i+1, sum, traj)
            traj.pop(-1)
            sum -= nums[i]
            # -号
            sum -= nums[i]
            traj.append(-nums[i])
            dfs(nums, i+1, sum, traj)
            traj.pop(-1)
            sum += nums[i]            
            # 啥都不加
            last_num = traj[-1]
            num = last_num*10 + nums[i] if last_num > 0 else last_num*10 - nums[i]
            sum +=(-last_num+num)
            traj.pop(-1)
            traj.append(num)
            dfs(nums, i+1, sum, traj)
            traj.pop(-1)
            traj.append(last_num)
            sum -=(-last_num+num)

    dfs(nums, 1, nums[0], [nums[0]])
    return all_traj
















if __name__ == '__main__':
    # print(permute([1,2,3]))
    # print(subsets([1,2,3]))
    # print(combinationSum([2,3,6,7], 7))
    # print(letterCombinations('235'))
    print(combineExp([1,1,1,1,1,1], 0))
