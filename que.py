from typing import List
from collections import deque



# 239. 滑动窗口最大值(使用双端单调队列)
# https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked
def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    q = deque()
    res = []
    # 初始化第一个窗口
    for i in range(k):
        while len(q) and q[-1] < nums[i]:
            q.pop()
        q.append(nums[i])
    res.append(q[0])

    # 遍历剩余元素
    for i in range(k, len(nums)):
        while len(q) and q[-1] < nums[i]:
            q.pop()
        q.append(nums[i])
        # 当准备移除窗口的值正好是单调队列的队头元素, 则队列pop一次
        if nums[i-k] == q[0]:
            q.popleft()
        res.append(q[0])

    return res

                





if __name__ == '__main__':
    maxSlidingWindow([1,3,-1,-3,5,3,6,7], k=3)
        