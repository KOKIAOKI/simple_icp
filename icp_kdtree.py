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


trj_fig = plt.figure("Trjectory", figsize=(16, 9), dpi=120)
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


class ICPProcess:
    def __init__(self):
        self.scan_cloud = np.empty((0,3))
        self.source_cloud = np.empty((0,3))
        self.dd = 0.001
        self.da = 0.001
        self.kk = 0.01
        self.evthere = 0.000001

        self.init_fig = plt.figure("Initial pose")
        self.ax_init_fig = self.init_fig.add_subplot(111)

        self.frames_kdtree = []
        self.kdtree_fig = plt.figure("kd_tree", figsize=(16, 9), dpi=120)
        self.ax_kd_tree = self.kdtree_fig.add_subplot(111)
        self.ax_kd_tree.set_xlabel('x [m]')
        self.ax_kd_tree.set_ylabel('y [m]')
        self.ax_kd_tree.grid()
        self.ax_kd_tree.set_aspect('equal')


    # 点群平均値を(0,0)になるように、点群を移動
    def transpointcloud_zero(self, input_cloud):
        cloudmean = np.mean(input_cloud, axis=0)
        return(input_cloud - cloudmean)
    

    def transpointcloud(self, scan_cloud, trans_pose):
        trans_cloud = np.empty((0,2))
        for i in range(len(scan_cloud)):
            cx, cy = scan_cloud[i, 0], scan_cloud[i, 1]
            tx, ty, tth =  trans_pose.x, trans_pose.y, trans_pose.th
            x = math.cos(tth) * cx - math.sin(tth) * cy + tx
            y = math.sin(tth) * cx + math.cos(tth) * cy + ty
            trans_cloud = np.append(trans_cloud, np.array([[x,y]]), axis=0)
        return(trans_cloud)


    def setInputSource(self, input_cloud):
        self.scan_cloud = input_cloud
        self.scan_points_num = input_cloud.shape[0] #配列数


    def setInputTarget(self, input_cloud):
        self.target_cloud = input_cloud
        self.kd_tree = KDTree(self.target_cloud)# kd_tree


    def setMode(self, optmode):
        self.mode = optmode


    def getIndexes(self):
        return self.indexes_temp


    def getItr(self):
        return self.itr


    def ICP_scan_matching(self, current_pose):
        self.itr = 1
        ev = 0
        evmin, evold = 10000, 10000
        while abs(evold - ev) > self.evthere:
            if self.itr > 1:
                evold = ev

            new_pose = Pose2D()
            if self.mode == 0:
                new_pose, ev= self.gradient(current_pose) # 勾配法
            elif self.mode == 1:
                new_pose, ev= self.Newton(current_pose) # ニュートン法
            # elif self.mode == 2:
            #     new_pose ,ev= cg(target_cloud, scan_cloud, current_pose) # 共役勾配法

            current_pose = new_pose

            if ev < evmin: #前のスコアより低ければ最適解候補を更新
                pose_min = new_pose
                evmin = ev
                trj_array.x = np.append(trj_array.x, np.array([[pose_min.x]]), axis=0)
                trj_array.y = np.append(trj_array.y, np.array([[pose_min.y]]), axis=0)
                trj_array.th = np.append(trj_array.th, np.array([[pose_min.th]]), axis=0)
                
            if self.itr > 30:
                break
            self.itr += 1
        return pose_min
            

    # 勾配法
    def gradient(self, init_pose):
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        t_ = copy.deepcopy(init_pose)

        # 点群同士の距離の総和、最近某探索
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)

        # アニメーション生成
        self.output_anim_graph(self.target_cloud, self.source_cloud, self.indexes_temp)

        # 最近傍探索時の誤差計算
        ev = np.sum(dists**2) / self.scan_points_num
        evmin = ev
        evold = 100000
        print("ev",ev)
        while abs(evold - ev) > self.evthere:
            evold = ev

            Exdd, Eydd, Ethda = self.E_delta1(t_) #微小変位
            F = self.E_first_derivative(Exdd, Eydd, Ethda, ev) #勾配
            dx = -self.kk * F[0,0]
            dy = -self.kk * F[1,0]
            dth = -self.kk * F[2,0]

            t_.x += dx
            t_.y += dy
            t_.th += dth

            ev = self.calcValue(t_.x, t_.y, t_.th)

            if ev < evmin:
                evmin = ev
                txmin = copy.deepcopy(t_)
        return(txmin, evmin)

    # Newton法
    def Newton(self, init_pose):
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        t_ = copy.deepcopy(init_pose)

        # 点群同士の距離の総和、最近某探索
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)

        # アニメーション生成
        self.output_anim_graph(self.target_cloud, self.source_cloud, self.indexes_temp)

        # 最近傍探索時の誤差計算
        ev = np.sum(dists**2) / self.scan_points_num

        Exdd, Eydd, Ethda = self.E_delta1(t_) #微小変位
        F = self.E_first_derivative(Exdd, Eydd, Ethda, ev) #勾配

        Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd = self.E_delta2(t_) #微小変位
        H = self.E_second_derivative(Exdd, Eydd, Ethda, Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd, ev) #ヘシアン

        invH = np.linalg.inv(H)
        delta_pose = np.dot(invH,-F)

        t_.x += delta_pose[0,0]
        t_.y += delta_pose[1,0]
        t_.th += delta_pose[2,0]
        evmin = self.calcValue(t_.x, t_.y, t_.th)
        txmin = copy.deepcopy(t_)
        return(txmin, evmin)


    # 勾配計算用の微小変位
    def E_delta1(self, t_):
        Exdd = self.calcValue(t_.x + self.dd, t_.y, t_.th)
        Eydd = self.calcValue(t_.x, t_.y + self.dd, t_.th)
        Ethda = self.calcValue(t_.x, t_.y, t_.th + self.da)
        return (Exdd,Eydd,Ethda)


    # 勾配計算
    def E_first_derivative(self, Exdd, Eydd, Ethda, ev):
        dEtx = (Exdd - ev)/ self.dd
        dEty = (Eydd - ev)/ self.dd
        dEth = (Ethda - ev)/ self.da
        F = np.around(np.array([[dEtx],[dEty],[dEth]]),decimals=5)
        return F


    # ヘッセ行列計算用の微小変位
    def E_delta2(self, t_):
        Ex2dd = self.calcValue(t_.x + 2*self.dd, t_.y, t_.th)
        Ey2dd = self.calcValue(t_.x, t_.y + 2*self.dd, t_.th)
        Eth2da = self.calcValue(t_.x, t_.y, t_.th + 2*self.da)
        Exddydd = self.calcValue(t_.x + self.dd, t_.y + self.dd, t_.th)
        Exddthdd = self.calcValue(t_.x + self.dd, t_.y, t_.th + self.da)
        Eyddthdd = self.calcValue(t_.x, t_.y + self.dd, t_.th + self.da)
        return (Ex2dd,Ey2dd,Eth2da,Exddydd,Exddthdd,Eyddthdd)


    # ヘッセ行列計算
    def E_second_derivative(self, Exdd, Eydd, Ethda, Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd, ev):
        dEtxtx = (Ex2dd - 2*Exdd + ev) / pow(self.dd,2)
        dEtyty =  (Ey2dd - 2*Eydd + ev) / pow(self.dd,2)
        dEtthtth = (Eth2da - 2*Ethda + ev) / pow(self.da,2)
        dEtxty = (Exddydd - Eydd - Exdd + ev) / pow(self.dd,2)
        dEtxth = (Exddthdd - Ethda -Exdd + ev) / self.dd*self.da
        dEtyth = (Eyddthdd - Ethda - Eydd + ev) / self.dd*self.da
        H = np.around(np.array([[dEtxtx,dEtxty,dEtxth],[dEtxty,dEtyty,dEtyth],[dEtxth,dEtyth,dEtthtth]]),decimals=5)
        return H


    # 評価関数
    def calcValue(self, tx, ty, th):
        error = 0
        for i in range(len(self.indexes_temp)):
            index = self.indexes_temp[i]

            cx, cy = self.scan_cloud[i, 0], self.scan_cloud[i, 1]  # 現在のscan_cloud点群
            tar_x, tar_y = self.target_cloud[index, 0], self.target_cloud[index, 1]  # 参照点

            x = math.cos(th) * cx - math.sin(th) * cy + tx  # 回転, 並進
            y = math.sin(th) * cx + math.cos(th) * cy + ty

            edis = pow(x - tar_x, 2) + pow(y - tar_y, 2)  # スコア計算
            error += edis
        error = error/self.scan_points_num
        return(error)


    # 初期位置設定
    def init_pose(self, user_input_cloud, current_pose):
        self.output_init_graph(user_input_cloud)
        self.init_fig.show()
        print("<< Please set the initail pose >>")
        continue_init = 0
        while (continue_init == 0):
            current_pose.x = float(input("initial_x >> "))
            current_pose.y = float(input("initial_y >> "))
            current_pose.th = float(input("initial_theta >> "))
            self.ax_init_fig.cla()
            init_temp_cloud = self.transpointcloud(self.scan_cloud, current_pose)
            self.output_init_graph(init_temp_cloud)
            print(init_temp_cloud)
            self.init_fig.show()
            continue_init = int(input("Are you sure you want to conduct ICP from this pose? No:0 Yes:1 >>"))
        return current_pose


    # 初期値設定グラフ
    def output_init_graph(self, init_scan_cloud):
        self.ax_init_fig.set_title("Initial pose")
        self.ax_init_fig.plot(self.target_cloud[:, 0], self.target_cloud[:, 1], "ok")
        self.ax_init_fig.plot(init_scan_cloud[:, 0], init_scan_cloud[:, 1], "or")
        cloudmean = np.mean(init_scan_cloud, axis=0)
        self.ax_init_fig.plot(cloudmean[0],cloudmean[1],"om")
        self.ax_init_fig.text(cloudmean[0],cloudmean[1],"Average of the scan points")
        self.ax_init_fig.set_xlabel('x [m]')
        self.ax_init_fig.set_ylabel('y [m]')
        self.ax_init_fig.grid()
        self.ax_init_fig.set_aspect('equal')


    # アニメーショングラフ
    def output_anim_graph(self, target_cloud, scan_cloud, indexes_temp):
        vis0 = self.ax_kd_tree.plot(target_cloud[:, 0], target_cloud[:, 1], "ok")
        vis1 = self.ax_kd_tree.plot(scan_cloud[:, 0], scan_cloud[:, 1], "or")
        vis2 = []
        for i in range(len(indexes_temp)):
            index = indexes_temp[i]
            vis2_temp = self.ax_kd_tree.plot([target_cloud[index, 0], scan_cloud[i, 0]], [target_cloud[index, 1], scan_cloud[i, 1]], "-g")
            vis2.extend(vis2_temp)
        self.frames_kdtree.append(vis0 + vis1 + vis2)


if __name__ == "__main__":
    argv = sys.argv
    tar_cloud_path = argv[1]
    scan_cloud_path = argv[2]
    tar_df = pd.read_csv(tar_cloud_path)
    scan_df = pd.read_csv(scan_cloud_path)
    target_cloud = tar_df.to_numpy()
    user_input_cloud = scan_df.to_numpy()
    del tar_df, scan_df

    # 点群を初期位置に移動
    mode = int(input("[ ICP/gradient:0, ICP/Newton:1, ICP/CG:2 ] >> "))
    if mode == 0:
        output_name = "gradient"
    if mode == 1:
        output_name = "newton"
    if mode == 2:
        output_name = "CG"

    # ICPの基本プロセスのインスタンス化
    icp = ICPProcess()
    scan_cloud = icp.transpointcloud_zero(user_input_cloud) # scan点群をの平均値を(0,0)へ移動
    icp.setInputSource(scan_cloud) # スキャン点群を使いまわし用にセット
    icp.setInputTarget(target_cloud) # 地図点群を使いまわし用にセット
    icp.setMode(mode)

    # 初期化
    current_pose = Pose2D()
    trj_array = Array2D() 

    # 初期位置設定
    current_pose = icp.init_pose(user_input_cloud, current_pose)
    trj_array.x = np.append(trj_array.x, np.array([[current_pose.x]]), axis=0)
    trj_array.y = np.append(trj_array.y, np.array([[current_pose.y]]), axis=0)
    trj_array.th = np.append(trj_array.th, np.array([[current_pose.th]]), axis=0)

    # ICP
    est_Pose = Pose2D()
    start_time = time.perf_counter()
    print(current_pose.x)
    est_Pose = icp.ICP_scan_matching(current_pose)
    end_time = time.perf_counter()
    exe_time = (end_time - start_time)*1000
    matched_cloud = icp.transpointcloud(scan_cloud, est_Pose) #マッチングした点群
    indexes = icp.getIndexes()
    itr = icp.getItr()
    icp.output_anim_graph(target_cloud, matched_cloud, indexes) #マッチングしたときの点群をアニメーションに追加
    
    # 出力
    print("estimated pose:","x",est_Pose.x,"y",est_Pose.y,"theta",est_Pose.th)
    print("iteration:",itr)
    print("exe_time:",exe_time,"[ms]")

    ani = animation.ArtistAnimation(icp.kdtree_fig, icp.frames_kdtree, interval=500, blit=True, repeat_delay=1000)  #アニメーション
    # ani.save(output_name + 'convergence_animation.mp4')

    #軌跡
    # width_offset = 0.01
    # max_offset = 1.0
    # points = int((max_offset/width_offset)*2 + 1)
    # offset_array = Array2D() 
    # for i in range(points):
    #     for j in range(points):
    #         offset_pose = Pose2D()
    #         offset_pose.x = est_Pose.x + width_offset * i - max_offset
    #         offset_pose.y = est_Pose.y + width_offset * j - max_offset
    #         offset_pose.th = 0
    #         offset_cloud = transpointcloud(scan_cloud, offset_pose)
    #         err_sum, indexes_dist = kd_tree.query(offset_cloud)
    #         err_sum_av = np.sum(err_sum) / scan_points_num
    #         offset_array.x = np.append(offset_array.x, np.array([[offset_pose.x]]), axis=0)
    #         offset_array.y = np.append(offset_array.y, np.array([[offset_pose.y]]), axis=0)
    #         offset_array.ev = np.append(offset_array.ev, np.array([[err_sum_av]]), axis=0)
    # ex_len = len(offset_array.ev)
    # length_tmp = int(np.sqrt(ex_len))
    # X_dist = offset_array.x.reshape(length_tmp,length_tmp)
    # Y_dist = offset_array.y.reshape(length_tmp,length_tmp)
    # EX_dist = offset_array.ev.reshape(length_tmp,length_tmp)
    # er_min = min(offset_array.ev)
    # er_max = max(offset_array.ev)
    # ax_hmap = ax_trj.pcolor(X_dist, Y_dist, EX_dist, cmap=cm.jet, vmin=er_min, vmax=er_max)
    # ax_trj.plot(trj_array.x,trj_array.y,'or',linestyle='solid')
    # plt.colorbar(ax_hmap, label='error average[m]')
    # ax_trj.text(0.1,1.05, 'iteration: {} '.format(itr), fontsize=15, transform=ax_trj.transAxes)
    # ax_trj.text(0.1,1.01, 'execution time[ms]: {} '.format(round(exe_time,2)), fontsize=15, transform=ax_trj.transAxes)
    # ax_trj.set_xlabel('x [m]')
    # ax_trj.set_ylabel('y [m]')
    # ax_trj.grid()
    # ax_trj.set_aspect('equal')
    # trj_fig.savefig(output_name + "trj.png")

    plt.show()