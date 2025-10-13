from typing import List



# 冒泡排序
def bubble_sort(l:List) -> List:
    n = len(l)

    for i in range(n, 0, -1):
        sorted = True
        for j in range(i-1):
            if (l[j]>l[j+1]):
                sorted=False
                l[j], l[j+1] = l[j+1], l[j]
        if sorted:
            break
    
    return l


# 快速排序
def quick_sort(nums:List, l:int, r:int) -> List:
    direction = False
    L, R = l, r

    if L >= R:
        return nums
    # 一趟快排(默认最左边元素为基准):
    while(l<r):
        if nums[l] > nums[r]:
            nums[l], nums[r] = nums[r], nums[l]
            direction = not direction

        if not direction:
            r -= 1 
        else:
            l += 1
    # 递归, 二分
    quick_sort(nums, L, l-1)
    quick_sort(nums, l+1, R)
    return nums
    

# 归并排序
def merge_sort(nums:List, l:int, r:int) -> List:

    L, R = l, r

    if l >= r:
        return nums
    
    mid = (l + r) // 2
    merge_sort(nums, l, mid)
    merge_sort(nums, mid+1, r)
    
    # 一趟归并:
    m = mid+1
    tmp = []
    while(l<=mid and m <= r):
        if nums[l] < nums[m]:
            tmp.append(nums[l])
            l += 1
        else:
            tmp.append(nums[m])
            m += 1     
    # 添加剩下的元素
    while l <= mid:
        tmp.append(nums[l])
        l += 1
    while m <= r:
        tmp.append(nums[m])
        m += 1
    nums[L:R+1] = tmp

    return nums

    
    


            
            


    


if __name__ == '__main__':
    nums = [5, 2, 8, 1,0,7,4,8,4,3, 9,5, 2, 4, 7, 1, 3, 2, 6]
    # print(bubble_sort(nums))
    # print(quick_sort(nums, 0, len(nums)-1))
    print(merge_sort(nums, 0, len(nums)-1))
