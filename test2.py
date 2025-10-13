import sys





def f(n,m,k,a):
    a = 'L' + a
    n = len(a)
    # 维护两个数组, 能否到达数组和剩余体力数组
    R = [0] * (n+1)
    K = [-1] * (n+1)
    K[0] = k
    R [0] = 1

    for i in range(n):
        # 当前位置根本到不了
        if not R[i]:
            break
        # 当前位置到得了且是陆地
        if a[i] == 'L':
            # 往前能跳到的都能reach
            for step in range(i+1, min(n+1, i+m+1)):
                R[step] = 1
                K[step] = max(k, K[step]) 
                # 能跳到目的, 提前结束
                if step == n:
                    return 1
        # 当前位置到得了且是水
        if a[i] == 'W':
            k -= 1
            # 如果没体力了当前格就到不了下一格(之前格仍然有可能到)
            if k >= 0:
                R[i+1] = 1
            K[i+1] = max(k, K[i+1]) 
    print(R)
    print(K)
    return R[-1]
    





if __name__ == '__main__':
    # f(6, 2, 15, 'LWLLCC') # n
    # f(6, 2, 0, 'LWLLLW') # y
    # f(6, 1, 1, 'LWLLLL') # y
    # f(6, 1, 1, 'LWLLWL') # n
    # f(6, 10, 0, 'CCCCCC') # y
    # f(7, 2, 1, "WLLCWLL") # n
    # f(6, 2, 0, "WLLLWLL") # y
    # f(6, 4, 1, "LWWCLL") # y
    # f(5, 4, 0, "LLLLL") # y
    # f(6, 1, 12, "WWWWWW") # y
    f(5, 5, 0, 'WCCCL')
