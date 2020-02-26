# kd_Tree
# Edited By ocean_waver
import numpy as np
import matplotlib.pyplot as plt


class Node(object):

    def __init__(self, mid, left, right, bound, flag, lchild=None, rchild=None, par=None,
                 l_bound=None, r_bound=None, side=-1):
        self.mid = mid
        self.left = left
        self.right = right
        self.bound = bound  # Dim * 2
        self.flag = flag
        self.lchild = lchild
        self.rchild = rchild
        self.par = par
        self.l_bound = l_bound
        self.r_bound = r_bound
        self.side = side


def find_median(a):
    # s = np.sort(a)
    arg_s = np.argsort(a)
    idx_mid = arg_s[len(arg_s) // 2]
    idx_left = np.array([arg_s[j] for j in range(0, len(arg_s) // 2)], dtype='int32')
    idx_right = np.array([arg_s[j] for j in range(len(arg_s) // 2 + 1, np.size(a))], dtype='int32')

    return idx_mid, idx_left, idx_right


def kd_tree_establish(root, points, dim):
    # print(root.mid)
    layer_flag = (root.flag + 1) % dim    # 确定分割点对应的分割线的维度

    if dim == 2:
        static_pos = points[root.mid, root.flag]
        if root.flag == 0:
            x_line = np.linspace(static_pos, static_pos, 10)
            y_line = np.linspace(root.bound[1, 0], root.bound[1, 1], 10)
        elif root.flag == 1:
            x_line = np.linspace(root.bound[0, 0], root.bound[0, 1], 10)
            y_line = np.linspace(static_pos, static_pos, 10)
        plt.plot(x_line, y_line, color='black')
        # plt.axis([0, 1, 0, 1])
        # plt.draw()
        # plt.pause(0.05)

    # new bound:
    root.l_bound = root.bound.copy()    # 先复制一份根节点边界(Note: need to use deep copy!)
    root.l_bound[root.flag, 1] = points[root.mid, root.flag]  # 改变特定边界的最大值，获取新边界
    root.r_bound = root.bound.copy()
    root.r_bound[root.flag, 0] = points[root.mid, root.flag]  # 改变特定边界的最小值，获取新边界

    if root.left.size > 0:
        # print('left : ', root.left)
        mid, left, right = find_median(points[root.left, layer_flag])
        mid, left, right = root.left[mid], root.left[left], root.left[right]

        left_node = Node(mid, left, right, root.l_bound, layer_flag)
        root.lchild = left_node
        left_node.par = root
        left_node.side = 0
        kd_tree_establish(left_node, points, dim)

    if root.right.size > 0:
        # print('right : ', root.right)
        mid, left, right = find_median(points[root.right, layer_flag])
        mid, left, right = root.right[mid], root.right[left], root.right[right]

        right_node = Node(mid, left, right, root.r_bound, layer_flag)
        root.rchild = right_node
        right_node.par = root
        right_node.side = 1
        kd_tree_establish(right_node, points, dim)


def distance(a, b, p):
    """
    Lp distance:
    input: a and b must have equal length
           p must be a positive integer, which decides the type of norm
    output: Lp distance of vector a-b"""
    try:
        vector = a - b
    except ValueError:
        print('Distance : input error !\n the coordinates have different length !')
    dis = np.power(np.sum(np.power(vector, p)), 1/p)
    return dis

# def search_other_branch(target, branch_node, points, dim):


def judge_cross(circle, branch, dim):
    """
    Judge if a sphere in dimension(dim) and the space of the other branch cross each other
    cross     : return 1
    not cross : return 0"""
    # print(circle, branch)
    count = 0
    for j in range(0, dim):
        if circle[j, 1] < branch[j, 0] or circle[j, 0] > branch[j, 1]:
            count = count + 1
    if count == 0:
        return 1    # cross
    else:
        return 0


if __name__ == '__main__':

    Dim = 2
    Num = 50
    Points = np.random.rand(Num, Dim)

    K = 4
    p = 2
    # Target = np.array([0.1, 0.9])
    Target = np.squeeze(np.random.rand(1, Dim))  # 这里只考虑一个目标点

    plt.scatter(Points[:, 0], Points[:, 1], color='blue')
    for i in range(0, Num):
        plt.text(Points[i, 0], Points[i, 1], str(i))

    '''# Test for find_median()
    idx_mid, idx_left, idx_right = find_median(Points[:, 0])
    print(Points[:, 0])
    print(Points[idx_mid, 0], idx_mid, idx_left, idx_right)'''

    # kdTree establish
    Mid, Left, Right = find_median(Points[:, 0])
    Bound = np.repeat(np.array([[0, 1]], dtype='float64'), Dim, axis=0)
    Root = Node(Mid, Left, Right, Bound, flag=0)
    # Root = Node(Bound, flag=0, dim=2)
    print('kdTree establish ...')
    kd_tree_establish(Root, Points, Dim)
    print('kdTree establish Done')

    # 定位初始搜索区域
    node = Root
    temp = Root
    side = 0    # 下降定位在终止时点所在的是左侧(side=0)还是右侧(side=1)
    while temp is not None:
        if Points[temp.mid, temp.flag] > Target[temp.flag]:    # 大于的情况
            node = temp
            temp = temp.lchild
            side = 0
        else:   # 包括小于和等于的情况
            node = temp
            temp = temp.rchild
            side = 1
    print('start node : ', node.mid, Points[node.mid])

    # 搜索最近邻点
    can_idx = np.array([], dtype='int32')
    can_dis = np.array([])

    temp = node
    while node is not None:
        # min_dis = distance(Target, Points[can_idx[-1]])
        search_flag = False
        temp_dis = distance(Target, Points[node.mid], p)

        if can_idx.size < K:    # 候选点列表未满
            can_idx = np.append(can_idx, node.mid)
            can_dis = np.append(can_dis, temp_dis)
        elif temp_dis < np.max(can_dis):
            can_idx[np.argmax(can_dis)] = node.mid
            can_dis[np.argmax(can_dis)] = temp_dis

        search_flag = False         # 查看另一支路是否为空
        if side == 0 and node.rchild is not None:
            branch_bound = node.rchild.bound
            branch_list = node.right
            search_flag = True
        elif side == 1 and node.lchild is not None:
            branch_bound = node.lchild.bound
            branch_list = node.left
            search_flag = True

        if search_flag is True:     # 开始判断和搜索另一侧的支路
            r = np.max(can_dis)
            temp_bound = np.array([[Target[i]-r, Target[i]+r] for i in range(0, Dim)])

            if judge_cross(temp_bound, branch_bound, Dim) == 1:     # 高维球与之路空间存在交叉

                for i in branch_list:
                    a_dis = distance(Target, Points[i], p)
                    if can_idx.size < K:            # 候选未满，直接添加
                        can_idx = np.append(can_idx, i)
                        can_dis = np.append(can_dis, a_dis)
                    elif a_dis < np.max(can_dis):   # 候选已满，更近者替换候选最远者
                        can_idx[np.argmax(can_dis)] = i
                        can_dis[np.argmax(can_dis)] = a_dis

        temp = node
        side = temp.side    # 更新上一个node的左右方位
        node = node.par

    sort_idx = np.argsort(can_dis)
    can_idx = can_idx[sort_idx]
    can_dis = can_dis[sort_idx]
    print('candidate_index : ', can_idx)
    print('candidate_distance : ', np.round(can_dis, 4))
    # print(Points)

    if Dim == 2:
        plt.scatter(Target[0], Target[1], c='cyan', s=30)
        for i in range(0, K):
            n = np.linspace(0, 2*3.14, 300)
            x = can_dis[i] * np.cos(n) + Target[0]
            y = can_dis[i] * np.sin(n) + Target[1]
            plt.plot(x, y, c='red')
            plt.axis([0, 1, 0, 1])
        plt.draw()
        plt.show()
