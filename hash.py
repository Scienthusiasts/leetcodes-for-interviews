from typing import List
import heapq
from collections import defaultdict


class Solution:

    # 1. 两数之和(哈希存当前结果, 后续直接查) 
    # 索引(这题nums无序, 如果有序则用双指针)
    # 要求返回的是下标
    # https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_hash = dict()
        for i in range(len(nums)):
            if target - nums[i] in nums_hash:
                return [nums_hash[target - nums[i]], i]
            else:
                nums_hash[nums[i]] = i
        

    # 49. 字母异位词分组
    # https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        # 为每个单词建立索引emb
        hash_list = [[0]*26 for _ in range(len(strs))]
        for i, str in enumerate(strs):
            for j in range(len(str)):
                c = ord(str[j]) - ord('a')
                hash_list[i][c] += 1

        # 根据hash进行字符串分组
        str_group = [[strs[0]]]
        hash_group = [hash_list[0]]
        for str, hash in zip(strs[1:], hash_list[1:]):
            i = 0
            match_flag = False
            # 匹配所有组
            while i < len(hash_group):
                if hash == hash_group[i]:
                    str_group[i].append(str)
                    match_flag = True
                    break
                i += 1
            # 所有组都没匹配上则新建一个组
            if match_flag == False:
                str_group.append([str])
                hash_group.append(hash)
        return str_group
    
    # 128. 最长连续序列 
    # https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked
    def longestConsecutive(self, nums: List[int]) -> int:
        # 生成hash索引
        hash = defaultdict()
        for i in range(len(nums)):
            hash[nums[i]] = 1
        # 将hash key从小到大排序
        hash = sorted(hash.items())

        # 找到最长连续序列
        lc_len = 0
        last_num = -1
        cur_len = 0
        for k, v in hash:
            if last_num + 1 == k:
                cur_len += 1
            else:
                cur_len = 1

            if cur_len > lc_len:
                lc_len = cur_len

            last_num = k
        
        return lc_len



    # 560. 和为 K 的子数组(前缀和类型+哈希)
    # https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 定义前缀和hash表, hash[i]表示前缀和为i(起点为0, 终点不确定)的个数
        pre_hash = defaultdict(int)
        # 注意初始化, 前缀和为0的个数有一个
        pre_hash[0] = 1
        pre_sum_k = 0
        cur_pre = 0
        # 更新前缀和
        for i in range(len(nums)):
            cur_pre += nums[i]
            # 核心:加入当前元素后, 包含当前元素的和为 K 的数组等价于:当前的前缀和-历史某个前缀和=k的个数
            pre_sum_k += pre_hash[cur_pre - k]
            pre_hash[cur_pre] += 1
        return pre_sum_k


    # 76. 最小覆盖子串(左右指针+哈希)(超时)
    # https://leetcode.cn/problems/minimum-window-substring/?envType=study-plan-v2&envId=top-100-liked
    def minWindow(self, s: str, t: str) -> str:
        if s == t: 
            return s
        # 建立t的字符统计hash
        s_hash = defaultdict(int)
        t_hash = defaultdict(int)
        for i in range(len(t)):
            t_hash[t[i]] += 1
        
        l, r = 0, 0
        min_len = 200000
        min_len_l, min_len_r = -1, -1
        while(r<len(s)):
            s_hash[s[r]] += 1

            # 更新左端点
            while (t_hash[s[l]]>0 and s_hash[s[l]] > t_hash[s[l]]) or t_hash[s[l]] == 0:
                s_hash[s[l]] -= 1
                l += 1
                if(l>r): return ""

            # 判断是否是覆盖子串(这里容易超时)
            case_flag = True
            for i in range(len(t)):
                if s_hash[t[i]] < t_hash[t[i]]:
                    case_flag = False
                    break

            # 更新最小覆盖子串
            if case_flag and r-l+1 < min_len:
                min_len = r-l+1
                min_len_l, min_len_r = l, r

            r += 1
        
        return s[min_len_l:min_len_r+1] 


    # 3. 无重复字符的最长子串(hash+快慢指针) 
    # https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/
    def lengthOfLongestSubstring(self, s: str) -> int:
        hash = defaultdict(int)
        l, r = 0, 0
        max_len = 0
        while(r<len(s)):
            hash[s[r]] += 1
            # 更新慢指针
            while(hash[s[r]] > 1 and l <= r):
                hash[s[l]] -= 1
                l += 1
            # 更新无重复字符的最长子串
            if r - l + 1 > max_len:
                max_len = r - l + 1
            # 更新快指针
            r += 1
        
        return max_len
    

    # 438. 找到字符串中所有字母异位词 
    # https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        if len(s) < len(p): return res
        # 初始化hash
        hash_s = [0]*26
        hash_p = [0]*26
        for i in range(len(p)):
            hash_p[ord(p[i]) - ord('a')] += 1
        for i in range(len(p)):
            hash_s[ord(s[i]) - ord('a')] += 1

        # 滑动窗口右移:
        for i in range(len(p), len(s)+1):
            if hash_s == hash_p:
                res.append(i - len(p))

            if i == len(s):break 
            
            hash_s[ord(s[i - len(p)]) - ord('a')] -= 1
            hash_s[ord(s[i]) - ord('a')] += 1
        
        return res

        
        








if __name__ == '__main__':
    sol = Solution()
    print(sol.findAnagrams(s = "ffacbaebabacb", p = "abc"))

