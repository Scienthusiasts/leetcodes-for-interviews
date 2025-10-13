from typing import List
import heapq
from collections import defaultdict


class Solution:
    # 35. 搜索插入位置(经典二分查找,) 
    # https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-100-liked
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1

        while(l<=r):
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                return mid
        
        # 没找到时, 返回l代表比其大的最小元素的下标, r代表比其小的最大元素的下标
        return l


    # 34. 在排序数组中查找元素的第一个和最后一个位置
    # https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 定义二分查找函数(变体, 找到后不直接返回, 而是找到边界)
        def bisearch(nums: List[int], target: int, lboard) -> int:
            l, r = 0, len(nums)-1
            while(l<=r):
                mid = (l + r) // 2
                if lboard:
                    if nums[mid] < target:
                        l = mid + 1
                    elif nums[mid] >= target:
                        r = mid - 1
                else:
                    if nums[mid] <= target:
                        l = mid + 1
                    elif nums[mid] > target:
                        r = mid - 1
            return l
        # 查找左右边界
        s = bisearch(nums, target, lboard=True)
        e = bisearch(nums, target, lboard=False)
        if s >= len(nums) or len(nums)==0 or nums[s]!=target:
            return [-1, -1]
        else:
            return [s, e-1]
        

    # 74. 搜索二维矩阵(可以用两次二分，但注意得先对列再对行) 
    # https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked
    # 这题可以只二分查找两次, 但是另一题不行
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 首先对列做二分查找
        l, r = 0, len(matrix)-1
        while(l<=r):
            mid = (l + r) // 2
            if matrix[mid][0] > target:
                r = mid - 1
            elif matrix[mid][0] < target:
                l = mid + 1
            else:
                return True
        
        row = r
        # 再对行查找
        l, r = 0, len(matrix[0])-1
        while(l<=r):
            mid = (l + r) // 2
            if matrix[row][mid] > target:
                r = mid - 1
            elif matrix[row][mid] < target:
                l = mid + 1
            else:
                return True
        
        return False
        


    # 153. 寻找旋转排序数组中的最小值 
    # https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:return nums[0]
        if len(nums) == 2:return min(nums[0], nums[1])

        right_value = nums[-1]
        l, r = 0, len(nums) - 1
        while(l<r):
            mid = (l + r) // 2
            if nums[mid] < nums[mid-1] and nums[mid] < nums[mid+1]:
                l = mid
                break
            # 如果当前元素比右端点元素大, 则最小元素一定在当前元素右侧
            elif nums[mid] > right_value:
                l = mid + 1
            # 如果当前元素比右端点元素小, 则最小元素一定在当前元素左侧
            elif nums[mid] < right_value:
                r = mid - 1
        
        return nums[l]


    # 33. 搜索旋转排序数组(同上一题,第一次寻找起点, 第二次执行常规二分查找)
    # https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
    def search(self, nums: List[int], target: int) -> int:
        
        offset = self.findMin(nums)
        l, r = 0, len(nums) - 1
        while(l<=r):
            mid = (l + r) // 2
            offset_mid = (mid + offset) % len(nums)
            if nums[offset_mid] < target:
                l = mid + 1
            elif nums[offset_mid] > target:
                r = mid - 1
            else:
                return offset_mid
            
        # l是其右索引的值, r是其左索引的值
        r = (r + offset) % len(nums)
        return -1 if nums[r] != target else r





if __name__ == '__main__':
    sol = Solution()
    
    # print(sol.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 60))
    # 4,5,6,7,0,1,2             1,2,4,5,6,7,0
    print(sol.findMin([4,5,6,7,0,1,2]))
