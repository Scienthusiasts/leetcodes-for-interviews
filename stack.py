from typing import List



class Solution:

    # 20. 有效的括号 （遇到左括号入栈, 右括号出栈, 且出栈一定对应入栈的那个括号,否则是无效的）
    # https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked
    def isValid(self, s: str) -> bool:
        match = {')':'(', ']':'[', '}':'{'}
        st = []
        for c in s:
            if c in ['(', '[', '{']:
                st.append(c)
            elif len(st) == 0 or match[c]!=st[-1]:
                return False
            elif match[c]==st[-1]:
                st.pop()
        
        if len(st) == 0:
            return True
        else:
            return False


    # 394. 字符串解码(用递归替代栈, 比较好写)
    # https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked
    def decodeString(self, s: str) -> str:
        decode_s, i = self.decodeStringRec(s, 0)
        return decode_s


    def decodeStringRec(self, s: str, i) -> str:
        decode_s = ''
        tmp_number = 0
        while i < len(s):
            # 只要判断4种情况, 字母, 数字, 左括号(递归), 右括号(结束递归)
            if s[i] >='a' and s[i] <='z':
                decode_s += s[i]
            elif s[i] >= '0' and s[i] <= '9':
                tmp_number = tmp_number * 10 + int(s[i])
            elif s[i] == '[':
                # i会跳转到匹配的']'出现的下标
                sub_str, i = self.decodeStringRec(s, i+1)
                decode_s += tmp_number * sub_str
                tmp_number = 0
            elif s[i] == ']':
                return decode_s, i
            i += 1
        return decode_s, i
        
        



    # 739. 每日温度(单调栈) 
    # https://leetcode.cn/problems/daily-temperatures/description/?envType=study-plan-v2&envId=top-100-liked
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        t_st = []
        res = [0] * len(temperatures)
        for i in range(len(temperatures)-1, -1, -1):
            # 栈不空则找到更高温度，栈空则没有更高温度
            if len(t_st) > 0:
                # pop掉之后比当前温度低的温度
                while(len(t_st) > 0 and temperatures[i] >= t_st[-1][1]):
                    t_st.pop()
                # 找到最近的下一次更高温度
                if len(t_st) > 0 and temperatures[i] < t_st[-1][1]:
                    res[i] = t_st[-1][0] - i
            # 当前温度进栈，栈里的元素从栈底到栈顶由大到小排列
            t_st.append([i, temperatures[i]])
                
        return res

                
    # 496. 下一个更大元素 I(单调栈)
    # https://leetcode.cn/problems/next-greater-element-i/
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nextMax = {}
        st = []
        for i in range(len(nums2)-1, -1, -1):
            while len(st)>0 and nums2[i] > st[-1]:
                st.pop()

            if len(st) > 0 and nums2[i] < st[-1]:
                nextMax[nums2[i]] = st[-1]

            if len(st) == 0:
                nextMax[nums2[i]] = -1
                
            st.append(nums2[i])
        
        res = []
        for num in nums1:
            res.append(nextMax[num])
        return res
                
                
    # 503. 下一个更大元素 II(循环数组，会有重复元素)  
    # https://leetcode.cn/problems/next-greater-element-ii/description/      
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        nextMax = []
        st = []
        # 循环数组只要复制两遍
        nums2 = nums + nums
        for i in range(len(nums2)-1, -1, -1):
            # 有重复元素, 这里得加等号
            while len(st)>0 and nums2[i] >= st[-1][0]:
                st.pop()
            # st[-1][1] - i != len(nums)的意思是比他大的元素不能是下一个循环中的自己
            if len(st) > 0 and nums2[i] < st[-1][0] and st[-1][1] - i != len(nums):
                nextMax.append(st[-1][0])

            else:
                nextMax.append(-1)
                
            st.append([nums2[i], i])

        return nextMax[::-1][:len(nums)]








if __name__ == '__main__':
    sol = Solution()
    # print(sol.decodeString('abc3[cd]xyz'))
    print(sol.nextGreaterElements([1,2,1]))


