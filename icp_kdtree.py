from os import X_OK
from re import X
import numpy as np
import math
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class Param:
    def __init__(self):
        target_cloud = ""
        scan_cloud = ""
        indexes_temp = ""
        dists = ""


class Pose2D:
    def __init__(self):
        x_ = ""
        y_ = ""
        th_ = ""

def output_init_graph():
    initial_fig = plt.figure("kd_tree", figsize=(16, 9), dpi=120)
    ax_kd_tree = initial_fig.add_subplot(111)
    ax_kd_tree.plot(target_cloud[:, 0], target_cloud[:, 1], "ob")
    ax_kd_tree.plot(scan_cloud[:, 0], scan_cloud[:, 1], "og")
    for i in range(len(indexes_temp)):
        index = indexes_temp[i]
        ax_kd_tree.plot([target_cloud[index, 0], scan_cloud[i, 0]], [target_cloud[index, 1], scan_cloud[i, 1]], "-r")
    ax_kd_tree.grid()
    ax_kd_tree.set_aspect("equal")
    plt.show()


def gradient():
    #　点群同士の距離の総和、最近某探索
    ev, indexes_temp = kd_tree.query(scan_cloud_temp)

    evmin = ev
    delta_x, delta_y = 0, 0
    evold = 100000
    while abs(evold - ev) > evthere:
        evold = ev
        dEtx = (calcValue(temp_x + dd, tem_y, target, indexes_temp, scan) - ev) / dd
        dEty = (calcValue(temp_x, tem_y + dd, target, indexes_temp, scan) - ev) / dd

        dx = -kk * dEtx
        dy = -kk * dEty

        temp_x += dx
        tem_y += dy

        ev = calcValue(temp_x, tem_y, target, indexes_temp, scan)

        if ev < evmin:
            evmin = ev
            delta_x += dx
            delta_y += dy
    return evmin, delta_x, delta_y

def transpointcloud_zero(scan_cloud, zero_cloud)
    cloudmean = np.mean(scan_cloud, axis=1)
    print("mean:"cloudmean)
    zero_cloud = scan_cloud - cloudmean
    zerocloud_mean = np.mean(zero_cloud, axis=1)
    print("set0:"+ zerocloud_mean)

def transpointcloud(trans_pose, scan_cloud, trans_cloud):
    
    trans_cloud = scan_cloud + cloudmean

# 勾配計算時使用スコア計算
def calcValue(tx, ty, th):
    error = 0
    for i in range(len(indexes_temp)):
        index = indexes_temp[i]

        cx, cy = scan_cloud[i, 0], scan_cloud[i, 1]  # 現在のscan_cloud点群
        tar_x, tar_y = target_cloud[index, 0], target_cloud[index, 1]  # 参照点

        x = math.cos(th) * cx - math.sin(th) * cy + ty  # 回転
        y = math.sin(th) * cx + math.cos(th) * cy + tx

        edis = pow(x - tar_x, 2) + pow(y - tar_y, 2)  # スコア計算
        error += edis
    return error


if __name__ == "__main__":
    argv = sys.argv
    tar_cloud_path = argv[1]
    scan_cloud_path = argv[2]
    tar_df = pd.read_csv(tar_cloud_path)
    scan_df = pd.read_csv(scan_cloud_path)

    target_cloud = tar_df.to_numpy()
    scan_cloud = scan_df.to_numpy()
    del tar_df, scan_df

    # kd_tree
    kd_tree = KDTree(target_cloud)

    # scan点群をの平均値を(0,0)へ移動
    def transpointcloud_zero(scan_cloud, scan_cloud)

    # 初期化
    current_pose = Pose2D()
    pose_min = Pose2D()
    estPose = Pose2D()

    # 点群を初期位置に移動
    current_pose.x = float(input("initial_x"))
    current_pose.y = float(input("initial_y"))
    current_pose.th = float(input("initial_th"))
    pose_min = current_pose
    
    # transpointcloud(current_pose, scan_cloud, scan_cloud_temp)

    ev = 0
    evmin, evold = 10000, 10000
    evthere = 0.000001
    itr = 1  # iteration
    scan_points_num = scan_cloud.shape #配列数

    # ICP
    while abs(evold - ev) > evthere:
        if itr > 1:
            evold = ev
        # dists, indexes_temp = kd_tree.query(scan_cloud_temp)  # 最近傍探索をやり直す
        # ev = np.sum(dists**2) / scan_points_num
        # if itr == 1:
        #     output_init_graph()
        #     evold = ev  # 最初は最近傍探索したときの距離の総和

        new_pose = Pose2D()
        ev = gradient(scan_cloud, current_pose, new_pose)  # 勾配方向に直線的に進んでいった結果、その進んだ距離と誤差
        current_pose = new_pose
        transpointcloud(scan_cloud, current_pose)

        if ev < evmin: #前のスコアより低ければ最適解候補を更新
            pose_min = new_pose

            evmin = ev
        itr += 1

    estPose = pose_min #推定値
