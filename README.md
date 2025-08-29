# simple_icp

## How to run ICP
Provide two CSV files (target.csv and scan.csv) with two columns: x in the first column and y in the second column. Sample data is in `1_test_dataset/` and `2_tunnel_dataset/`.
```
cd simple_icp
# create output folder for saved animations
mkdir -p output_folder

python3 icp_kdtree.py <target.csv> <scan.csv>
# example
python3 icp_kdtree.py 1_test_dataset/target_sample.csv 1_test_dataset/scan_sample.csv
```
## Operation example 
You can choose 2 optimization methods: Gauss-Newton or Levenberg-Marquardt.
Then, decide the initial pose while referring to the displayed graph. You can fix the initial pose.
```
Select optimization method [1: Gauss-Newton, 2: Levenberg-Marquardt] (default: 1) >> 1
<< Please set the initail pose >>
initial_x >> 7
initial_y >> 3.5
initial_theta >> 0
Are you sure you want to conduct ICP from this pose? No:0 Yes:1 >>1
```
![Initial_pose](https://user-images.githubusercontent.com/81670028/184363417-c18f45e8-35c3-4b47-aa3e-5811c61880a3.png)

Scan matching animation is saved in output_folder.
![newton_animation](https://user-images.githubusercontent.com/81670028/184362336-f6f5a0a5-c5d1-4a1f-85af-beba5ea6ca68.gif)