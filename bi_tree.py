from typing import List, Optional, Dict
import heapq # 小根堆
from collections import defaultdict, deque # 字典, 队列
import time
from copy import deepcopy


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



class Solution:
    def __init__(self, ):
        self.res = []

    # 根据层次遍历结果生成二叉树
    def createTree(self, tree_list:List, idx=0) -> TreeNode:
        if idx >= len(tree_list) or tree_list[idx]==None:
            return None
        else:
            root = TreeNode(tree_list[idx])
            root.left = self.createTree(tree_list, 2*idx+1)
            root.right = self.createTree(tree_list, 2*idx+2)
            return root


    # 二叉树的前序遍历
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root != None:
            self.res.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)




    # 102. 二叉树的层序遍历(在循环前先记录队列中的元素数量就是每一层的数量) 
    # https://leetcode.cn/problems/binary-tree-level-order-traversal/description/
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        queue = deque()
        res = []
        if root==None: return []
        queue.append(root)

        # 开始遍历:
        while len(queue)!=0:
            # 在循环前先记录队列中的元素数量就是每一层的数量
            cur_layer_size = len(queue)
            cur_layer_res = []
            for _ in range(cur_layer_size):
                pop_node = queue.popleft()
                cur_layer_res.append(pop_node.val)
                if pop_node.left!=None:
                    queue.append(pop_node.left)
                if pop_node.right!=None:
                    queue.append(pop_node.right)
            res.append(cur_layer_res)
        return res
            
    


    # 108. 将有序数组转换为二叉搜索树 
    # https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        l, r = 0, len(nums)-1
        return self.sortedArrayToBSTRec(nums, l, r)
    
    # 题目要求是平衡二叉搜索树，但不一定就要按照平衡的构造法
    # 找到升序序列的中间元素, 以中序序列的方式构造二叉树，就是 平衡二叉搜索树
    def sortedArrayToBSTRec(self, nums: List[int], l, r) -> Optional[TreeNode]:

        node = None
        if (l<=r):
            mid = (l + r) // 2
            node = TreeNode(nums[mid])
            node.left = self.sortedArrayToBSTRec(nums, l, mid-1)
            node.right = self.sortedArrayToBSTRec(nums, mid+1, r)
        
        return node

    
    # 236. 二叉树的最近公共祖先(递归) 
    # https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 当前结点为空则返回None
        if root==None:
            return root
        # 当前结点就是p或q, 则当前节点一定是最近公共祖先
        if root==p or root==q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # 左右子树分别有pq, 则当前节点一定是最近公共祖先
        if left and right:
            return root
        # pq只在左子树或右子树,则最近公共祖先在左或右子树
        elif left:
            return left
        else:
            return right


    # 124. 二叉树中的最大路径和 (树形DP, 通常使用后序遍历)
    # https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-100-liked
    def maxPathSumRec(self, root: Optional[TreeNode], global_max=-10000) -> tuple[int, int]:
        # 空结点
        if not root:
            return -10000, -10000
        
        # 后序遍历递归
        left_max, l_global_max = self.maxPathSumRec(root.left)
        right_max, r_global_max = self.maxPathSumRec(root.right)
        # cur_max是包含当前节点的单边最大路径和(注意这里必须只能算单边路径, 否则后续输出的路径会有分叉)
        cur_max = max(left_max, right_max, 0) + root.val
        # global_max是以当前节点为根的最大路径和(路径里面不一定有当前节点)
        global_max = max(l_global_max, r_global_max, root.val + left_max + right_max, cur_max)
        return cur_max, global_max

    def maxPathSum(self, root: Optional[TreeNode]) -> int:    
        _, global_max = self.maxPathSumRec(root)
        return global_max


    # 437. 路径总和 III 
    # https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2&envId=top-100-liked
    # 类似560. 和为 K 的子数组(计算前缀和, 前序遍历)
    def pathSumRec(self, root, targetSum, hash, preSum) -> int:
        # 空节点不会产生路径
        if not root:
            return 0
        # preSum为包含当前节点的前缀和
        preSum += root.val
        totalNum = hash[preSum - targetSum]
        # 用hash记录前缀和
        hash[preSum] += 1
        totalNum += self.pathSumRec(root.left, targetSum, hash, preSum)
        totalNum += self.pathSumRec(root.right, targetSum, hash, preSum)
        # 恢复状态(字典共享内存空间, 递归修改会影响到上一层, 所以要回复)
        hash[preSum] -= 1
        return totalNum


    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        hash = defaultdict(int)
        hash[0] = 1
        res = self.pathSumRec(root, targetSum, hash, 0)
        return res






if __name__ == '__main__':
    sol = Solution()
    tree = sol.createTree([1,1,1,1,1])
    sol.preorderTraversal(tree)
    print(sol.res)
    print(sol.pathSum(tree, 0))



