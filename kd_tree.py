import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

dd = 0.00001
kk = 0.00001
evthere = 0.000001


def calcValue(pose_x, pose_y, target, indexes_temp, scan):
    error = 0
    scan[:, 0] += pose_x # 原点まわりから指定座標へ txは座標
    scan[:, 1] += pose_y # 原点まわりから指定座標へ tyは座標
    for i in range(len(indexes_temp) > evthere):
        index = indexes_temp[i]
        edis = pow(scan[i, 0] - target[index, 0], 2) + pow(scan[i, 1] - target[index, 1], 2)
        error += edis
    return error


def gradient(ev, temp_x, tem_y, target, indexes_temp, scan):
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


if __name__ == "__main__":
    argv = sys.argv
    csv_dir = argv[1]
    result_df = pd.read_csv(csv_dir)
    sec_temp = result_df[".header.stamp.secs"]
    nsec_temp = result_df[".header.stamp.nsecs"]
    time_1_df = sec_temp + nsec_temp / 10**9
    time_2_df = time_1_df + 0.5
    linear_x_df = result_df[".twist.linear.x"]
    tar_df = pd.DataFrame()
    tar_df[0] = time_1_df
    tar_df[1] = linear_x_df
    target = tar_df.to_numpy()
    scan_df = pd.DataFrame()
    scan_df[0] = time_2_df
    scan_df[1] = linear_x_df
    scan = scan_df.to_numpy()
    del result_df

    # target = np.loadtxt('/home/megken/07_optimum/icp/twist.csv', delimiter=',', skiprows=1)
    # scan = np.loadtxt('/home/megken/07_optimum/icp/twist_offset.csv', delimiter=',', skiprows=1)

    # point cloud transform
    # scan[:, 0] += 0
    # scan[:, 1] +=0
    # target[:, 0] +=0
    # target[:, 1] +=0

    # kd_tree
    kd_tree = KDTree(target)
    dists, indexes_temp = kd_tree.query(scan)

    initial_fig= plt.figure("kd_tree", figsize=(16, 9), dpi=120)
    ax_kd_tree = initial_fig.add_subplot(111)
    ax_kd_tree.plot(target[:, 0], target[:, 1], "ob")
    ax_kd_tree.plot(scan[:, 0], scan[:, 1], "og")
    for i in range(len(indexes_temp)):
        index = indexes_temp[i]
        ax_kd_tree.plot([target[index, 0], scan[i, 0]], [target[index, 1], scan[i, 1]], "-r")

    est_x = 0
    est_y = 0
    evmin, evold = 100000, 100000
    ev, evopt = 0, 0
    itr = 1
    # ICP
    while abs(evold - evopt) > evthere:
        evold = evopt
        dists, indexes_temp = kd_tree.query(scan)
        ev = np.sum(dists)
        if i == 1:
            evold = ev
        evopt, delta_optx, delta_opty = gradient(ev, est_x, est_y, target, indexes_temp, scan)
        if evopt < evmin:
            evmin = evopt
            est_x += delta_optx
            est_y += delta_opty
            itr += 1
