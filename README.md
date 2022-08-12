# simple_icp

## How to run ICP
Specify two types of csv (target.csv and scan.csv) containing x coordinates in the first row and y coordinates in the second row. Refer to some datasets.
```
cd simple_icp
python3 icp_kdtree.py <target.csv> <scan.csv>
```
## Operation example 
You can chose 3 types optimization methods.
Then, decide the initial pose while referring to the displayed graph. You can fixed the initial pose.
```
[ ICP/gradient:0, ICP/Newton:1, ICP/CG:2 ] >> 1
<< Please set the initail pose >>
initial_x >> 7
initial_y >> 3.5
initial_theta >> 0
Are you sure you want to conduct ICP from this pose? No:0 Yes:1 >>1
```
![Initial_pose](https://user-images.githubusercontent.com/81670028/184363417-c18f45e8-35c3-4b47-aa3e-5811c61880a3.png)

Scan matching animation is saved in output_folder.
![newton_animation](https://user-images.githubusercontent.com/81670028/184362336-f6f5a0a5-c5d1-4a1f-85af-beba5ea6ca68.gif)