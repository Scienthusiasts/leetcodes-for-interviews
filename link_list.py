from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 创建链表
    def create_list(self, list:List):
        head = ListNode()
        p = head
        for i in list:
            p.next = ListNode(i)
            p = p.next
        return head.next
    
    # 打印链表
    def print_list(self, list: Optional[ListNode]):
        res = []
        while(list!=None):
            res.append(list.val)
            list = list.next
        print(res)
            

    # 21. 合并两个有序链表(递归)
    # https://leetcode.cn/problems/merge-two-sorted-lists/description/
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 == None:
            return list2
        elif list2 == None:
            return list1
        elif list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
        
        
    # 206. 反转链表(迭代, 三指针) 
    # https://leetcode.cn/problems/reverse-linked-list/description/
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 初始化三指针
        p = None
        q = head

        while(q!=None):
            r = q.next
            # 反转指针
            q.next = p
            # 右移
            p = q
            q = r
            if r!=None: r = r.next
        
        return p


    # 206. 反转链表(递归) 
    # https://leetcode.cn/problems/reverse-linked-list/description/
    def reverseListRec(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head.next == None:
            return head, head
        else:
            tail, new_head = self.reverseListRec(head.next)
            tail.next = head
            tail = head
            tail.next = None
            return tail, new_head


    # 24. 两两交换链表中的节点(递归思路清晰些)
    # https://leetcode.cn/problems/swap-nodes-in-pairs/?envType=study-plan-v2&envId=top-100-liked
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 结点只有一个或为空则不交换
        if head==None or head.next==None:
            return head
        # 交换
        else:
            tmp_node = head.next
            head.next = self.swapPairs(head.next.next)
            tmp_node.next = head
        
        return tmp_node













if __name__ == '__main__':
    sol = Solution()
    l1 = sol.create_list([1,2,3,4,5])

    _, l_reverse = sol.reverseListRec(l1)
    sol.print_list(l_reverse)

