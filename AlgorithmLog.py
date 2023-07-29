# Algorithm Logs

"""
date: 2023年7月29日 09点55分
"""
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