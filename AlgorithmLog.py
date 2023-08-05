# Algorithm Logs

#===============================================================================
# date: 2023/07/29
#===============================================================================
class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        """"
        螺旋矩阵Ⅱ：
        这道题的难点就是如何改变旋转的方向, 与螺旋矩阵one类似, 采用一个数组保存四个方向(directions),
        然后设置初始方向为向右，所以启示坐标设置为(0,-1), 输出的矩阵初始化为全0, 更改前进方向的条件为:
        1.坐标在0到n-1之外;
        2.下一个坐标的矩阵值为0。
        从1循环到n**2+1输出矩阵就可得到答案。
        """

        matrix = [[0]*n for i in range(n)] # 初始化矩全为0
        directions = [(0,1),(1,0),(0,-1),(-1,0)] # 设置前进方向依次为右,下,左,上
        dir = 0 # 设置初始化方向
        i,j = 0,-1 # 设置初始坐标
        for key in range(1,n*n+1):
            di,dj = directions[dir%4] # 获得坐标改变量
            if not 0<=i+di<n or not 0<=j+dj<n or matrix[i+di][j+dj] != 0: # 判断下一坐标是否需要改变方向
                dir += 1 # 改变方向
                di,dj = directions[dir%4] # 获得新的坐标改变值
            i += di
            j += dj
            matrix[i][j] = key # 矩阵赋值
        return matrix

#===============================================================================
# date: 2023/07/30
#===============================================================================
from typing import Optional
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        旋转链表：
        这道题目要求向右旋转指定次数, 思路就是先将链表首尾相连, 然后根据次数移动现在相邻的首尾
        node, 最后将尾部节点指向None, 返回首节点即可。
        """
        
        # 空链表或者只有一个元素的链表直接返回头节点
        if not head or not head.next:
            return head
        # 利用pre记录链表首节点
        pre = head
        # 记录链表长度
        length = 1
        # 遍历链表找到尾节点，并计算链表长度
        while head.next:
            length += 1
            head = head.next
        # 首尾相连
        head.next = pre
        # 移动次数超过链表长度时取余
        k = k % length
        # 此时链表方向为向左, 用链表长度减去移动次数改变方向为向右(画图增加理解)
        k = length - k
        # 开始移动
        if k > 0:
            for i in range(k):
                head = pre
                pre = head.next
        # 移动结束，将尾部节点指向None
        head.next = None
        # 返回首节点
        return pre

#===============================================================================
# date: 2023/08/01
#===============================================================================
class Solution(object):
    def uniquePaths(self, m, n):
        """
        不同路径：
        这道题有很多解法，这里采用动态规划，先对每一行进行更新，然后对每一列进行更新：
        第一行只有一条路径，所以全为1，第一列同理全为1；
        其余位置只有有上一行或者上一列的位置达到，所以值等于二者之和dp[i] = dp[i-1] + dp[i]
        其中dp[i-1]为同行前一列的路径数，dp[i]为同列上一行的路径数；
        对所有列进行操作，返回dp最后一个数字即可。
        """
        # 一维空间，其大小为 n
        dp = [1] * n
        for i in range(1, m): # 对所有行进行更新
            for j in range(1, n): # 对一行中的所有位置进行更新
                # 等式右边的 dp[j]是上一次计算后的，加上左边的dp[j-1]即为当前结果
                dp[j] = dp[j] + dp[j - 1]
        return dp[-1]

#===============================================================================
# date: 2023/08/04
#===============================================================================
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        不同路径Ⅱ:
        思路和没有障碍相同，只是外加一个对当前点的判断，如果值为1就设为0，不能达到这里，
        不为1时更新路径总数: dp[j] = dp[j] + dp[j-1], 详情可ctrl+f搜索"不同路径"。
        """
        m = len(obstacleGrid) # 行
        n = len(obstacleGrid[0]) # 列
        # 动态规划第一个元素
        if obstacleGrid[0][0] == 1: # 动态规划第一个元素
            dp = [0]
        else:
            dp = [1]
        # 第一行的情况
        for i in range(1,n):
            if obstacleGrid[0][i] == 1:
                dp.append(0)
            else:
                dp.append(dp[-1])
        # 从第二行开始的情况
        for i in range(1,m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j == 0:
                    dp[j] = dp[j]
                else:
                    dp[j] = dp[j] + dp[j-1]
        return dp[-1]

#===============================================================================
# date: 2023/08/04
#===============================================================================
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        最小路径和：
        对于给出的grid中间的任意一个点，到达的最小路径为其前方点和上方点的最小路径加上
        该点的数字，所以可以使用动态规划更新标准为: dp[j] = min(dp[j],dp[j-1]) + grid[i][j]
        """
        # 获得网格信息
        r = len(grid)
        c = len(grid[0])
        #初始化dp
        dp = []
        dp.append(grid[0][0])
        for i in range(1,c):
            dp.append(grid[0][i] + dp[i-1])
        # 从第二行开始
        for i in range(1,r):
            # 第一个元素只能由上方元素得到
            dp[0] = dp[0] + grid[i][0]
            # 从第二列开始
            for j in range(1,c):
                # 更新dp
                dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
        return dp[-1]
