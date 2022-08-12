import copy
import math
import sys
import time
from os import X_OK
from re import X
from xml.sax.handler import DTDHandler

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from matplotlib import cm
from scipy.spatial import KDTree


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

        self.m = np.empty((0,3))
        self.m_next = np.empty((0,3))

        self.init_fig = plt.figure("Initial pose")
        self.ax_init_fig = self.init_fig.add_subplot(111)

        self.frames_result = []
        self.result_fig = plt.figure("Result", figsize=(16, 9), dpi=120)
        self.ax_result = self.result_fig.add_subplot(121)
        self.ax_result.set_xlabel('x [m]')
        self.ax_result.set_ylabel('y [m]')
        self.ax_result.grid()
        self.ax_result.set_aspect('equal')


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


    def setMode(self, optmode, output_name):
        self.mode = optmode
        self.ax_result.set_title(output_name)


    def getIndexes(self):
        return self.indexes_temp


    def getItr(self):
        return self.itr


    def getEstPose(self):
        return self.pose_min


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
            elif self.mode == 2:
                new_pose ,ev= self.cg(current_pose) # 共役勾配法

            current_pose = new_pose

            if ev < evmin: #前のスコアより低ければ最適解候補を更新
                self.pose_min = new_pose
                evmin = ev
                trj_array.x = np.append(trj_array.x, np.array([[self.pose_min.x]]), axis=0)
                trj_array.y = np.append(trj_array.y, np.array([[self.pose_min.y]]), axis=0)
                trj_array.th = np.append(trj_array.th, np.array([[self.pose_min.th]]), axis=0)
                
            if self.itr > 29:
                break
            self.itr += 1
            

    # 勾配法
    def gradient(self, init_pose):
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        t_ = copy.deepcopy(init_pose)

        # 点群同士の距離の総和、最近傍探索
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)

        # アニメーション生成
        self.output_anim_graph(self.source_cloud)

        # 最近傍探索時の誤差計算
        ev = np.sum(dists**2) / self.scan_points_num
        evmin = ev
        evold = 100000
        while abs(evold - ev) > self.evthere:
            evold = ev

            Exdd, Eydd, Ethda = self.E_delta1(t_) # 微小変位
            F = self.E_first_derivative(Exdd, Eydd, Ethda, ev) # 勾配
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
        self.output_anim_graph(self.source_cloud)

        # 最近傍探索時の誤差計算
        ev = np.sum(dists**2) / self.scan_points_num

        Exdd, Eydd, Ethda = self.E_delta1(t_) # 微小変位
        F = self.E_first_derivative(Exdd, Eydd, Ethda, ev) # 勾配

        Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd = self.E_delta2(t_) # 微小変位
        H = self.E_second_derivative(Exdd, Eydd, Ethda, Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd, ev) # ヘシアン

        invH = np.linalg.inv(H)
        delta_pose = np.dot(invH,-F)

        t_.x += delta_pose[0,0]
        t_.y += delta_pose[1,0]
        t_.th += delta_pose[2,0]
        evmin = self.calcValue(t_.x, t_.y, t_.th)
        txmin = copy.deepcopy(t_)
        return(txmin, evmin)


    # 共役勾配法
    def cg(self, init_pose):
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        t_ = copy.deepcopy(init_pose)

        # 点群同士の距離の総和、最近某探索
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)

        # アニメーション生成
        self.output_anim_graph(self.source_cloud)

        # 最近傍探索時の誤差計算
        ev = np.sum(dists**2) / self.scan_points_num

        evmin = ev
        evold = 100000
        count_first = True
        while abs(evold - ev) > self.evthere:
            evold = ev

            Exdd, Eydd, Ethda = self.E_delta1(t_) # 微小変位
            F = self.E_first_derivative(Exdd, Eydd, Ethda, ev) # 勾配

            # iteration１回目は勾配方向を使う
            if self.itr == 1:
                if count_first == True:
                    self.m_next = F # はじめに求めた勾配方向を次の接線方向成分に使用する。
                    count_first = False
                dx = -self.kk * F[0,0]
                dy = -self.kk * F[1,0]
                dth = -self.kk * F[2,0]
            # iteration2回目以降は共役勾配ｍの方向へ進む
            else:
                Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd = self.E_delta2(t_) #微小変位
                H = self.E_second_derivative(Exdd, Eydd, Ethda, Ex2dd, Ey2dd, Eth2da, Exddydd, Exddthdd, Eyddthdd, ev) #ヘシアン
                if count_first == True:
                    alpha = - np.dot(self.m_next.T, np.dot(H,F)) / np.dot(self.m_next.T, np.dot(H, self.m_next))
                    self.m = F + alpha*self.m_next #勾配ベクトルと、接線ベクトルを足し合わせて、共役勾配方向を求める 
                    self.m_next = self.m #はじめに求めた共役勾配方向を次の接線方向成分に使用する。
                    count_first = False
                else:
                    alpha = - np.dot(self.m.T, np.dot(H,F)) / np.dot(self.m.T, np.dot(H, self.m))
                    self.m = F + alpha*self.m #勾配ベクトルと、接線ベクトルを足し合わせて、共役勾配方向を求める 
                dx = -self.kk * self.m[0,0]
                dy = -self.kk * self.m[1,0]
                dth = -self.kk * self.m[2,0]

            t_.x += dx
            t_.y += dy
            t_.th += dth

            ev = self.calcValue(t_.x, t_.y, t_.th)

            if ev < evmin:
                evmin = ev
                txmin = copy.deepcopy(t_)
            else:
                evmin = np.sum(dists**2) / self.scan_points_num
                txmin = copy.deepcopy(init_pose)

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
    def output_anim_graph(self ,scan_cloud):
        vis0 = self.ax_result.plot(self.target_cloud[:, 0], self.target_cloud[:, 1], "ok")
        vis1 = self.ax_result.plot(scan_cloud[:, 0], scan_cloud[:, 1], "or")
        vis2 = []
        for i in range(len(self.indexes_temp)):
            index = self.indexes_temp[i]
            vis2_temp = self.ax_result.plot([self.target_cloud[index, 0], scan_cloud[i, 0]], [self.target_cloud[index, 1], scan_cloud[i, 1]], "-g")
            vis2.extend(vis2_temp)
        self.frames_result.append(vis0 + vis1 + vis2)


    # 総当りの分布と軌跡のグラフ
    def trj_graph(self):
        
        ax_trj = self.result_fig.add_subplot(122)
        width_offset = 0.05
        max_offset = 1.0
        points = int((max_offset/width_offset)*2 + 1)
        offset_array = Array2D() 
        for i in range(points):
            for j in range(points):
                offset_pose = Pose2D()
                offset_pose.x = self.pose_min.x + width_offset * i - max_offset
                offset_pose.y = self.pose_min.y + width_offset * j - max_offset
                offset_pose.th = 0
                offset_cloud = self.transpointcloud(self.scan_cloud, offset_pose)
                err_sum, indexes_dist = self.kd_tree.query(offset_cloud)
                err_sum_av = np.sum(err_sum) / self.scan_points_num
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
        ax_trj.text(0.1,1.05, 'iteration: {} '.format(self.itr), fontsize=15, transform=ax_trj.transAxes)
        ax_trj.text(0.1,1.01, 'execution time[ms]: {} '.format(round(exe_time,2)), fontsize=15, transform=ax_trj.transAxes)
        ax_trj.set_xlabel('x [m]')
        ax_trj.set_ylabel('y [m]')
        ax_trj.grid()
        ax_trj.set_aspect('equal')
        

if __name__ == "__main__":
    argv = sys.argv
    tar_cloud_path = argv[1]
    scan_cloud_path = argv[2]
    target_cloud = np.loadtxt(tar_cloud_path, delimiter=',')
    user_input_cloud = np.loadtxt(scan_cloud_path, delimiter=',')

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
    icp.setMode(mode, output_name)

    # 初期化
    current_pose = Pose2D()
    trj_array = Array2D() 

    # 初期位置設定
    current_pose = icp.init_pose(user_input_cloud, current_pose)
    trj_array.x = np.append(trj_array.x, np.array([[current_pose.x]]), axis=0)
    trj_array.y = np.append(trj_array.y, np.array([[current_pose.y]]), axis=0)
    trj_array.th = np.append(trj_array.th, np.array([[current_pose.th]]), axis=0)

    # ICP
    start_time = time.perf_counter()
    icp.ICP_scan_matching(current_pose)
    end_time = time.perf_counter()
    exe_time = (end_time - start_time)*1000

    est_pose = icp.getEstPose()
    indexes = icp.getIndexes()
    itr = icp.getItr()

    matched_cloud = icp.transpointcloud(scan_cloud, est_pose) # マッチングした点群
    icp.output_anim_graph(matched_cloud) # マッチングしたときの点群をアニメーションに追加
    
    # 出力
    print("estimated pose:","x",est_pose.x,"y",est_pose.y,"theta",est_pose.th)
    print("iteration:",itr)
    print("exe_time:",exe_time,"[ms]")

    ani = animation.ArtistAnimation(icp.result_fig, icp.frames_result, interval=500, blit=True, repeat_delay=1000)  # アニメーション
    icp.trj_graph() # 軌跡のグラフ

    # グラフ保存
    file_name = "output_folder/" + output_name + "_animation"
    if sys.version_info < (3,7):
        ani.save(file_name + '.gif', writer="imagemagick")
    else:
        ani.save(file_name + '.mp4')
        ani.save(file_name + '.gif', writer="imagemagick")
    plt.show()
