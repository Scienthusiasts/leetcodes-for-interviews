from typing import List
import heapq
from collections import defaultdict


class Solution:

    # 200. 岛屿数量(DFS, 每到一块陆地就变成海, 统计dfs次数)
    # https://leetcode.cn/problems/number-of-islands/description/?envType=study-plan-v2&envId=top-100-liked
    # DFS
    def numIslandsDFS(self, grid: List[List[str]], x, y, n, m) -> int:
        if 0<=x<n and 0<=y<m and grid[x][y] == "1":
            grid[x][y] = "0"
            self.numIslandsDFS(grid, x+1, y, n, m)
            self.numIslandsDFS(grid, x, y+1, n, m)
            self.numIslandsDFS(grid, x-1, y, n, m)
            self.numIslandsDFS(grid, x, y-1, n, m)

    def numIslands(self, grid: List[List[str]]) -> int:
        num = 0
        n, m = len(grid), len(grid[0])
        # 遍历每一个grid, 每次进行dfs
        for i in range(n):
            for j in range(m):
                if grid[i][j] == "1":
                    num += 1
                    self.numIslandsDFS(grid, i, j, n, m)

        return num


    # 695. 岛屿最大面积(DFS)
    # https://leetcode.cn/problems/max-area-of-island/description/
    def maxAreaOfIslandRec(self, grid: List[List[int]], x, y, area):
        w, h = len(grid[0]), len(grid)
        if 0<=x<w and 0<=y<h and grid[y][x]>0:
            area += 1
            grid[y][x] = -1
            area = self.maxAreaOfIslandRec(grid, x+1, y, area)
            area = self.maxAreaOfIslandRec(grid, x-1, y, area)
            area = self.maxAreaOfIslandRec(grid, x, y+1, area)
            area = self.maxAreaOfIslandRec(grid, x, y-1, area)
        return area

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]>0:
                    # dfs
                    area = self.maxAreaOfIslandRec(grid, j, i, 0)
                    if max_area < area:
                        max_area = area
        return max_area


        







if __name__ == '__main__':
    sol = Solution()
    grid = [[1,1,0,0,0],
            [1,1,0,0,0],
            [0,0,0,1,1],
            [0,0,0,1,1]
            ]
    print(sol.maxAreaOfIsland(grid))
