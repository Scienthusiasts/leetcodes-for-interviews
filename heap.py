from typing import List
import heapq # 小根堆
from collections import defaultdict # 字典


class Solution:

    # 215. 数组中的第K个最大元素(大根堆, 使用python自带的heapq, 小根堆) 
    # https://leetcode.cn/problems/kth-largest-element-in-an-array/?envType=study-plan-v2&envId=top-100-liked
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 构建大根堆(heapq.heappush默认构建小根堆, 所以需要插入负数变成大根堆)
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        # 出堆, 第k个出堆的就是第k大元素
        max_num = 0
        for i in range(k):
            max_num = -heapq.heappop(heap)

        return max_num

    # 347. 前 K 个高频元素(大根堆) 
    # https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2&envId=top-100-liked
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 构造索引
        nums_freq = defaultdict(int)
        for num in nums:
            nums_freq[num] += 1
        # 构造堆
        heap = []
        for key in  nums_freq.keys():
            val = nums_freq[key]
            # 如果元素是一个元组heapq默认按元组的第一个值排序
            heapq.heappush(heap, [-val, key])
        # 出堆, 第k个出堆的就是第k大元素
        top_k = []
        for i in range(k):
            max_k = heapq.heappop(heap)
            top_k.append(max_k[1])
        return top_k
            

    # 239. 滑动窗口最大值(大根堆)  
    # 用优先队列(注意优先队列pop时先pop顶部元素, 因此这里需要调整pop的逻辑)
    # 相当于每次都插入大顶堆，堆的最顶部就是最大元素，窗口移动时就对堆进行调整(反正保证每次堆顶都是最大元素)
    # https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        # 建立大根堆
        heap = []
        for i in range(k):
            heapq.heappush(heap, [-nums[i], i])
        res.append(-heap[0][0])
        
        # 更新滑动窗口最大值(注意当最大值下标超过滑动窗口大小才pop)
        for i in range(k, len(nums)):
            heapq.heappush(heap, [-nums[i], i])
            # 当最大值下标超过滑动窗口大小pop
            while heap[0][1] < i-k+1:
                heapq.heappop(heap)
            res.append(-heap[0][0])
        return res





if __name__ == '__main__':
    sol = Solution()
    print(sol.maxSlidingWindow([9,10,9,-7,-4,-8,2,-6], 5))

