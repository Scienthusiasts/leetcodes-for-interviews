from typing import List, Optional


# 树形DP (通常使用后序遍历, 递推)
# 递推: 自底向上, 先解决子问题, 再解决原问题
# 需要返回子问题的结果, 然后根据子问题的结果递推出原问题的结果


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right






# 543. 二叉树的直径 https://leetcode.cn/problems/diameter-of-binary-tree/description/
def diameterOfBinaryTreeRec(root: Optional[TreeNode], max_d) -> int:
    # 这里采用后续遍历(递推形式)
    if root:
        if not (root.left or root.right):
            return 0
        else:
            # l_max_d 表示左子树的最大直径(可能不包含当前节点), l_max表示左子树包含当前节点的最大直径(即深度)
            l_max_d, l_max = diameterOfBinaryTreeRec(root.left, max_d)
            r_max_d, r_max = diameterOfBinaryTreeRec(root.right, max_d)
            cur_depth = max(l_max, r_max) + 1
            max_d = max(l_max + r_max + 2, l_max_d, r_max_d)
            return max_d, cur_depth
    else:
        return -1

def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    max_d, cur_depth = diameterOfBinaryTreeRec(root, 0)
    return max_d





# 124. 二叉树中的最大路径和 (树形DP, 通常使用后序遍历, 递推)
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






# 337. 打家劫舍 III https://leetcode.cn/problems/house-robber-iii/description/
# 状态方程找对, 然后后续遍历递推
def robRec(root: Optional[TreeNode]) -> int:
    if root:
        l_steal, l_nosteal = robRec(root.left)
        r_steal, r_nosteal = robRec(root.right)
        # 状态转移方程:
        steal_max = l_nosteal+r_nosteal+root.val
        no_steal_max = max(l_steal, l_nosteal) + max(r_steal, r_nosteal)
        return steal_max, no_steal_max
    else:
        return 0, 0
    
def rob(root: Optional[TreeNode]) -> int:

    steal, nosteal = robRec(root)
    return max(steal, nosteal+root.val)
