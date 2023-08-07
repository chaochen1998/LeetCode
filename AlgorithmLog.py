# Algorithm Logs

"""
the notes is above the coresponding code like this:

# notes
coresponding code

"""

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
# date: 2023/08/05
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

#===============================================================================
# date: 2023/08/06
#===============================================================================
class Solution:
    def simplifyPath(self, path: str) -> str:
        """
        简化路径：
        思路是通过split函数以'/'将path分开，然后依次判断每个字符串s：
        如果s在['.', '', '..']中，说明不是路径，所以不能加入到最终的输出结果res中，
        其实如果s等于'..', 还要进一步将res中的最后一个元素删除, 在res中还有元素的情况下,
        最后将res依次输出，加上'/'即可。
        """

        # 拆分路径
        tmp = path.split('/')
        # 不能加入到输出路径res中的元素
        target = ['.', '', '..']
        # 保存路径
        res = []
        # 开始遍历
        for s in tmp:
            # 可以加入输出路径的情况
            if s not in target:
                res.append(s)
                continue
            # 当输出路径res还有上一级时才返回上一级
            if s == '..' and len(res) > 0:
                res.pop()
        out = ''
        # 当res为空时表示为根目录
        if len(res) == 0:
            return '/'
        # 输出完整路径
        for s in res:
            out += '/'+s
        return out

#===============================================================================
# date: 2023/08/07
#===============================================================================

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        矩阵置零:
        开始使用了比较笨的办法，先遍历给0做标记，然后再遍历给有标记的所在行和列设置为0。
        改进是给零元素所在首行和首列做标记，然后单独设置标签表示首行是否存在零，第二次遍历时
        只需要判断该元素所在行头或列头是否为0。
        来自力扣-画图小匠
        """
        m = len(matrix)
        n = len(matrix[0])
        first_row = False   # 标记首行是否有0元素
        for i, row in enumerate(matrix):
            for j, item in enumerate(row):
                if i == 0 and item == 0:
                    first_row = True    # 首行出现0元素，用标志位标记
                elif item == 0:
                    matrix[i][0] = 0    # 非首行出现0元素，将对应的列首置为0，说明该列要置为0
                    matrix[0][j] = 0    # 将对应的行首置为0，说明该行要置为0
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                # 从最后一个元素反向遍历，避免行首和列首的信息被篡改
                if i == 0 and first_row:
                    matrix[i][j] = 0    # 首行元素是否置为0看标志位
                elif i != 0 and (matrix[i][0] == 0 or matrix[0][j] == 0):
                    matrix[i][j] = 0    # 非首行元素是否置为0看行首和列首是否为0
