from os import X_OK
from re import X
import copy
from xml.sax.handler import DTDHandler
import numpy as np
import math
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree
import scipy.linalg as linalg
from matplotlib import cm

dd, da, kk = 0.00001, 0.00001, 0.01
evthere = 0.000001
scan_points_num = 0
#graph initialize
frames_kdtree = []
kdtree_fig = plt.figure("kd_tree", figsize=(16, 9), dpi=120)
ax_kd_tree = kdtree_fig.add_subplot(111)
trj_fig = plt.figure("trjectory", figsize=(16, 9), dpi=120)
ax_trj = trj_fig.add_subplot(111)


class Pose2D:
    def __init__(self):
        self.x = ""
        self.y = ""
        self.th = ""

class Array2D:
    def __init__(self):
        self.x = np.empty((0,1))
        self.y = np.empty((0,1))
        self.th = np.empty((0,1))
        self.ev = np.empty((0,1))

def output_init_graph(target_cloud, scan_cloud, indexes_temp):
    vis0 = ax_kd_tree.plot(target_cloud[:, 0], target_cloud[:, 1], "ok")
    vis1 = ax_kd_tree.plot(scan_cloud[:, 0], scan_cloud[:, 1], "or")
    vis2 = []
    for i in range(len(indexes_temp)):
        index = indexes_temp[i]
        vis2_temp = ax_kd_tree.plot([target_cloud[index, 0], scan_cloud[i, 0]], [target_cloud[index, 1], scan_cloud[i, 1]], "-g")
        vis2.extend(vis2_temp)
    frames_kdtree.append(vis0 + vis1 + vis2)


def gradient(target_cloud, scan_cloud, init_pose):
    source_cloud = transpointcloud(scan_cloud, init_pose)
    t_ = copy.deepcopy(init_pose)

    #　点群同士の距離の総和、最近某探索
    dists, indexes_temp = kd_tree.query(source_cloud)

    #　アニメーション生成
    output_init_graph(target_cloud, source_cloud, indexes_temp)

    #最近傍探索時の誤差計算
    ev = np.sum(dists**2) / scan_points_num
    # print("search_ev",ev)
    # ev = calcValue(t_.x, t_.y, t_.th, target_cloud, scan_cloud, indexes_temp)
    # print("error_func_ev",ev)

    evmin = ev
    evold = 100000
    while abs(evold - ev) > evthere:
        evold = ev
        dEtx = func_tx(t_,target_cloud, scan_cloud,indexes_temp,ev)
        dEty = func_ty(t_,target_cloud, scan_cloud,indexes_temp,ev)
        dEth = func_tth(t_,target_cloud, scan_cloud,indexes_temp,ev)

        dx = -kk * dEtx
        dy = -kk * dEty
        dth = -kk * dEth

        t_.x += dx
        t_.y += dy
        t_.th += dth

        ev = calcValue(t_.x, t_.y, t_.th, target_cloud, scan_cloud,indexes_temp)
        # print("pass_ev",ev)

        if ev < evmin:
            evmin = ev
            txmin = copy.deepcopy(t_)
    return(txmin, evmin, indexes_temp)


def Newton(target_cloud, scan_cloud, init_pose):
    source_cloud = transpointcloud(scan_cloud, init_pose)
    t_ = copy.deepcopy(init_pose)

    #　点群同士の距離の総和、最近某探索
    dists, indexes_temp = kd_tree.query(source_cloud)

    #　アニメーション生成
    output_init_graph(target_cloud, source_cloud, indexes_temp)

    #最近傍探索時の誤差計算
    ev = np.sum(dists**2) / scan_points_num
    # print("search_ev",ev)
    # ev = calcValue(t_.x, t_.y, t_.th, target_cloud, scan_cloud, indexes_temp)
    # print("error_func_ev",ev)
    
    dEtx = func_tx(t_,target_cloud, scan_cloud,indexes_temp,ev)
    dEty = func_ty(t_,target_cloud, scan_cloud,indexes_temp,ev)
    dEth = func_tth(t_,target_cloud, scan_cloud,indexes_temp,ev)
    F = np.array([[dEtx],[dEty],[dEth]])

    dEtxtx = func_dEtxtx(t_,target_cloud, scan_cloud,indexes_temp,dEtx)
    dEtyty = func_ty(t_,target_cloud, scan_cloud,indexes_temp,dEty)
    dEtthtth = func_tth(t_,target_cloud, scan_cloud,indexes_temp,dEth)
    dEtxty = func_tx(t_,target_cloud, scan_cloud,indexes_temp,dEty)
    dEtxth = func_tx(t_,target_cloud, scan_cloud,indexes_temp,dEth)
    dEtyth = func_ty(t_,target_cloud, scan_cloud,indexes_temp,dEth)
    H = np.array([[dEtxtx,dEtxty,dEtxth],[dEtxty,dEtyty,dEtyth],[dEtxth,dEtyth,dEtthtth]])

    # LU分解でΔxを求める
    # LU = linalg.lu_factor(H) 
    # delta_pose = linalg.lu_solve(LU, -F)
    print("H",H)
    invH = np.linalg.inv(H)
    delta_pose = np.dot(H,-F)

    t_.x += delta_pose[0,0]
    t_.y += delta_pose[1,0]
    t_.th += delta_pose[2,0]
    evmin = calcValue(t_.x, t_.y, t_.th, target_cloud, scan_cloud,indexes_temp)
    print("tx",t_.x)
    txmin = copy.deepcopy(t_)
    return(txmin, evmin, indexes_temp)

def func_tx(t_,target_cloud, scan_cloud,indexes_temp,ev):
    return (calcValue(t_.x + dd, t_.y, t_.th,target_cloud, scan_cloud,indexes_temp) - ev) / da

def func_ty(t_,target_cloud, scan_cloud,indexes_temp,ev):
    return (calcValue(t_.x, t_.y + dd, t_.th,target_cloud, scan_cloud,indexes_temp) - ev) / da

def func_tth(t_,target_cloud, scan_cloud,indexes_temp,ev):
    return (calcValue(t_.x, t_.y, t_.th + da,target_cloud, scan_cloud,indexes_temp) - ev) / da

def func_dEtxtx(t_,target_cloud, scan_cloud,indexes_temp,dEtx):
    return (calcValue(t_.x + 2*dd, t_.y, t_.th,target_cloud, scan_cloud,indexes_temp) - )

# 勾配計算時使用スコア計算
def calcValue(tx, ty, th,target_cloud, source_cloud,indexes_temp):
    error = 0
    for i in range(len(indexes_temp)):
        index = indexes_temp[i]

        cx, cy = source_cloud[i, 0], source_cloud[i, 1]  # 現在のscan_cloud点群
        tar_x, tar_y = target_cloud[index, 0], target_cloud[index, 1]  # 参照点

        x = math.cos(th) * cx - math.sin(th) * cy + tx  # 回転
        y = math.sin(th) * cx + math.cos(th) * cy + ty

        edis = pow(x - tar_x, 2) + pow(y - tar_y, 2)  # スコア計算
        error += edis
    error = error/scan_points_num
    return(error)

def transpointcloud_zero(scan_cloud):
    cloudmean = np.mean(scan_cloud, axis=0)
    print("mean:",cloudmean)
    return(scan_cloud - cloudmean)
    # zerocloud_mean = np.mean(zero_cloud, axis=0)
    # print("set0:",zerocloud_mean)
    # print(zero_cloud)

def transpointcloud(scan_cloud, trans_pose):
    trans_cloud = np.empty((0,2))
    for i in range(len(scan_cloud)):
        cx, cy = scan_cloud[i, 0], scan_cloud[i, 1]
        tx, ty, tth =  trans_pose.x, trans_pose.y, trans_pose.th
        x = math.cos(tth) * cx - math.sin(tth) * cy + tx
        y = math.sin(tth) * cx + math.cos(tth) * cy + ty
        trans_cloud = np.append(trans_cloud, np.array([[x,y]]), axis=0)
    return(trans_cloud)

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
    scan_cloud = transpointcloud_zero(scan_cloud)

    # 初期化
    current_pose = Pose2D()
    pose_min = Pose2D()
    estPose = Pose2D()
    trj_array = Array2D() 

    # 点群を初期位置に移動
    mode = int(input("ICP/gradient:0, ICP/Newton:1 "))
    current_pose.x = float(input("initial_x: "))
    current_pose.y = float(input("initial_y: "))
    current_pose.th = float(input("initial_th: "))
    pose_min = current_pose
    trj_array.x = np.append(trj_array.x, np.array([[pose_min.x]]), axis=0)
    trj_array.y = np.append(trj_array.y, np.array([[pose_min.y]]), axis=0)
    trj_array.th = np.append(trj_array.th, np.array([[pose_min.th]]), axis=0)

    ev = 0
    evmin, evold = 10000, 10000

    itr = 1  # iteration
    scan_points_num = scan_cloud.shape[0] #配列数

    # ICP
    start_time = time.perf_counter()
    while abs(evold - ev) > evthere:
        if itr > 1:
            evold = ev

        new_pose = Pose2D()
        if mode == 0:
            new_pose ,ev , indexes_temp = gradient(target_cloud, scan_cloud, current_pose) # 勾配法
        elif mode == 1:
            new_pose ,ev , indexes_temp = Newton(target_cloud, scan_cloud, current_pose) # ニュートン法

        current_pose = new_pose

        if ev < evmin: #前のスコアより低ければ最適解候補を更新
            pose_min = new_pose
            evmin = ev
            print("trj_array.x",trj_array.x)
            print("pose_min.x",pose_min.x)
            trj_array.x = np.append(trj_array.x, np.array([[pose_min.x]]), axis=0)
            trj_array.y = np.append(trj_array.y, np.array([[pose_min.y]]), axis=0)
            trj_array.th = np.append(trj_array.th, np.array([[pose_min.th]]), axis=0)

        itr += 1
        if itr > 30:
            break

    end_time = time.perf_counter()
    exe_time = (end_time - start_time)*1000
    estPose = pose_min #推定値
    matched_cloud = transpointcloud(scan_cloud, estPose) #マッチングした点群
    output_init_graph(target_cloud, matched_cloud, indexes_temp) #マッチングしたときの点群をアニメーションに追加

    # 出力
    print("estimated pose:","x",estPose.x,"y",estPose.y,"z",estPose.th)
    print("iteration:",itr)
    print("exe_time:",exe_time,"[ms]")

    #アニメーション
    ax_kd_tree.set_xlabel('x [m]')
    ax_kd_tree.set_ylabel('y [m]')
    ax_kd_tree.grid()
    ax_kd_tree.set_aspect('equal')
    ani = animation.ArtistAnimation(kdtree_fig, frames_kdtree, interval=500, blit=True, repeat_delay=1000)
    ani.save('convergence_animation.gif')

    #軌跡
    width_offset = 0.02
    max_offset = 1.0
    points = int((max_offset/width_offset)*2 + 1)
    offset_array = Array2D() 
    for i in range(points):
        for j in range(points):
            offset_pose = Pose2D()
            offset_pose.x = estPose.x + width_offset * i - max_offset
            offset_pose.y = estPose.y + width_offset * j - max_offset
            offset_pose.th = 0
            offset_cloud = transpointcloud(scan_cloud, offset_pose)
            err_sum, indexes_dist = kd_tree.query(offset_cloud)
            err_sum_av = np.sum(err_sum) / scan_points_num
            offset_array.x = np.append(offset_array.x, np.array([[offset_pose.x]]), axis=0)
            offset_array.y = np.append(offset_array.y, np.array([[offset_pose.y]]), axis=0)
            offset_array.ev = np.append(offset_array.ev, np.array([[err_sum_av]]), axis=0)
    ex_len = len(offset_array.ev)
    length_tmp = int(np.sqrt(ex_len))
    X_dist = offset_array.x.reshape(length_tmp,length_tmp)
    Y_dist = offset_array.y.reshape(length_tmp,length_tmp)
    EX_dist = offset_array.ev.reshape(length_tmp,length_tmp)
    er_min = min(offset_array.ev)
    er_max = max(offset_array.ev)
    ax_hmap = ax_trj.pcolor(X_dist, Y_dist, EX_dist, cmap=cm.jet, vmin=er_min, vmax=er_max)
    ax_trj.plot(trj_array.x,trj_array.y,'or',linestyle='solid')
    plt.colorbar(ax_hmap, label='error average[m]')
    ax_trj.set_xlabel('x [m]')
    ax_trj.set_ylabel('y [m]')
    ax_trj.grid()
    ax_trj.set_aspect('equal')
    trj_fig.savefig("trj.png")

    plt.show()
