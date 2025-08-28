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

        # Lazy-init for result figure (create after initial pose is confirmed)
        self.frames_result = []
        self.result_fig = None
        self.ax_result = None
        self.result_title = "Result"
        # Optimization algorithm: 'gn' (Gauss-Newton) or 'lm' (Levenberg-Marquardt)
        self.algorithm = "gn"
        # LM damping parameters
        self.lm_lambda = 1e-3
        self.lm_lambda_min = 1e-9
        self.lm_lambda_max = 1e9


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
        self.result_title = output_name
        if self.ax_result is not None:
            self.ax_result.set_title(output_name)

    def setAlgorithm(self, method_str):
        m = str(method_str).strip().lower()
        if m in ("lm", "levenberg-marquardt", "levenberg_marquardt"):
            self.algorithm = "lm"
        else:
            self.algorithm = "gn"


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
        # Create result axes if needed (after initial pose figure is closed)
        if self.ax_result is None:
            self.result_fig = plt.figure("Result", figsize=(16, 9), dpi=120)
            self.ax_result = self.result_fig.add_subplot(121)
        else:
            self.ax_result.cla()
        # Configure axes
        self.ax_result.set_title(self.result_title)
        self.ax_result.set_xlabel('x [m]')
        self.ax_result.set_ylabel('y [m]')
        self.ax_result.grid()
        self.ax_result.set_aspect('equal')
        while abs(evold - ev) > self.evthere:
            if self.itr > 1:
                evold = ev

            new_pose = Pose2D()
            # Optimization step
            if getattr(self, "algorithm", "gn") == "lm":
                new_pose, ev = self.levenberg_marquardt_se2(current_pose)
            else:
                new_pose, ev = self.gauss_newton_se2(current_pose)

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
            

    # Gauss-Newton step using se(2) analytic Jacobian (point-to-point)
    def gauss_newton_se2(self, init_pose):
        # 1) Transform scan by current pose
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        # 2) Nearest neighbors on target
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)
        # 3) Visualization for this iteration
        self.output_anim_graph(self.source_cloud)
        
        # 4) Build residual vector r and Jacobian J
        N = self.scan_points_num
        J = np.zeros((2 * N, 3))
        r = np.zeros((2 * N, 1))
        for i in range(N):
            xi, yi = self.source_cloud[i, 0], self.source_cloud[i, 1]
            idx = int(self.indexes_temp[i])
            qx, qy = self.target_cloud[idx, 0], self.target_cloud[idx, 1]
            # residual e_i = (Rp_i + t) - q_i
            r[2 * i, 0] = xi - qx
            r[2 * i + 1, 0] = yi - qy
            # Analytic Jacobian wrt left-multiplied twist [vx, vy, omega]
            # de/d[vx,vy,omega] = [[1, 0, -y_i], [0, 1, x_i]]
            J[2 * i, 0] = 1.0
            J[2 * i, 1] = 0.0
            J[2 * i, 2] = -yi
            J[2 * i + 1, 0] = 0.0
            J[2 * i + 1, 1] = 1.0
            J[2 * i + 1, 2] = xi
        
        # 5) Solve normal equations (with tiny damping for numerical stability)
        H = J.T @ J
        g = J.T @ r
        lam = 1e-8
        H_damped = H + lam * np.eye(3)
        try:
            delta = -np.linalg.solve(H_damped, g)  # [dvx, dvy, domega]
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(H_damped) @ g
        
        dvx = float(delta[0, 0])
        dvy = float(delta[1, 0])
        domega = float(delta[2, 0])
        
        # 6) Left-multiply update: T <- Exp(delta^) * T
        # For pose (x, y, th):
        #   R_delta = R(domega), t_delta = [dvx, dvy]
        #   x_new, y_new = R_delta @ [x, y] + t_delta; th_new = th + domega
        new_pose = copy.deepcopy(init_pose)
        c = math.cos(domega)
        s = math.sin(domega)
        x_new = c * new_pose.x - s * new_pose.y + dvx
        y_new = s * new_pose.x + c * new_pose.y + dvy
        th_new = new_pose.th + domega
        new_pose.x = x_new
        new_pose.y = y_new
        new_pose.th = th_new
        
        # 7) Evaluate new error (mean squared) after the update
        updated_cloud = self.transpointcloud(self.scan_cloud, new_pose)
        d2, self.indexes_temp = self.kd_tree.query(updated_cloud)
        evmin = np.sum(d2**2) / self.scan_points_num
        return new_pose, evmin

    # Levenberg-Marquardt step using se(2) analytic Jacobian (point-to-point)
    def levenberg_marquardt_se2(self, init_pose):
        # 1) Transform scan by current pose
        self.source_cloud = self.transpointcloud(self.scan_cloud, init_pose)
        # 2) Nearest neighbors on target
        dists, self.indexes_temp = self.kd_tree.query(self.source_cloud)
        # 3) Visualization for this iteration
        self.output_anim_graph(self.source_cloud)

        # 4) Build residual vector r and Jacobian J
        N = self.scan_points_num
        J = np.zeros((2 * N, 3))
        r = np.zeros((2 * N, 1))
        for i in range(N):
            xi, yi = self.source_cloud[i, 0], self.source_cloud[i, 1]
            idx = int(self.indexes_temp[i])
            qx, qy = self.target_cloud[idx, 0], self.target_cloud[idx, 1]
            # residual e_i = (Rp_i + t) - q_i
            r[2 * i, 0] = xi - qx
            r[2 * i + 1, 0] = yi - qy
            # Analytic Jacobian wrt left-multiplied twist [vx, vy, omega]
            J[2 * i, 0] = 1.0
            J[2 * i, 1] = 0.0
            J[2 * i, 2] = -yi
            J[2 * i + 1, 0] = 0.0
            J[2 * i + 1, 1] = 1.0
            J[2 * i + 1, 2] = xi

        # 5) Solve damped normal equations
        H = J.T @ J
        g = J.T @ r
        lam = getattr(self, "lm_lambda", 1e-3)
        H_damped = H + lam * np.eye(3)
        try:
            delta = -np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(H_damped) @ g

        dvx = float(delta[0, 0])
        dvy = float(delta[1, 0])
        domega = float(delta[2, 0])

        # 6) Left-multiply update: T <- Exp(delta^) * T
        new_pose = copy.deepcopy(init_pose)
        c = math.cos(domega)
        s = math.sin(domega)
        x_new = c * new_pose.x - s * new_pose.y + dvx
        y_new = s * new_pose.x + c * new_pose.y + dvy
        th_new = new_pose.th + domega
        new_pose.x = x_new
        new_pose.y = y_new
        new_pose.th = th_new

        # 7) Evaluate errors (mean squared)
        ev_curr = np.sum(dists**2) / N
        updated_cloud = self.transpointcloud(self.scan_cloud, new_pose)
        d2, self.indexes_temp = self.kd_tree.query(updated_cloud)
        ev_new = np.sum(d2**2) / N

        # 8) Adapt lambda and accept/reject step
        if ev_new < ev_curr:
            self.lm_lambda = max(lam * 0.5, getattr(self, "lm_lambda_min", 1e-9))
            return new_pose, ev_new
        else:
            self.lm_lambda = min(lam * 2.0, getattr(self, "lm_lambda_max", 1e9))
            return init_pose, ev_curr

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
        self.init_fig.canvas.draw_idle()
        plt.pause(0.001)
        print("<< Please set the initail pose >>")
        continue_init = 0
        while (continue_init == 0):
            current_pose.x = float(input("initial_x >> "))
            current_pose.y = float(input("initial_y >> "))
            current_pose.th = float(input("initial_theta >> "))
            self.ax_init_fig.cla()
            init_temp_cloud = self.transpointcloud(self.scan_cloud, current_pose)
            self.output_init_graph(init_temp_cloud)
            self.init_fig.canvas.draw_idle()
            plt.pause(0.001)
            continue_init = int(input("Are you sure you want to conduct ICP from this pose? No:0 Yes:1 >>"))
        # Close the initial pose figure to switch to the result view
        plt.close(self.init_fig)
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

    # 方式選択: Gauss-Newton or Levenberg-Marquardt
    method_in = input("Select optimization method [1: Gauss-Newton, 2: Levenberg-Marquardt] (default: 1) >> ").strip()
    if method_in in ("2", "lm", "LM", "Levenberg", "Levenberg-Marquardt", "levenberg", "levenberg_marquardt"):
        selected_method = "lm"
        output_name = "levenberg_marquardt"
    else:
        selected_method = "gn"
        output_name = "gauss_newton"

    # ICPの基本プロセスのインスタンス化
    icp = ICPProcess()
    scan_cloud = icp.transpointcloud_zero(user_input_cloud) # scan点群をの平均値を(0,0)へ移動
    icp.setInputSource(scan_cloud) # スキャン点群を使いまわし用にセット
    icp.setInputTarget(target_cloud) # 地図点群を使いまわし用にセット
    icp.setMode(3, output_name)  # modeは未使用、タイトル設定のために呼ぶ
    icp.setAlgorithm(selected_method)

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
