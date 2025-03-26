# Can Real-to-Sim Approaches Capture Dynamic Fabric Behavior for Robotic Fabric Manipulation?

![image](https://github.com/user-attachments/assets/80cba701-7b98-4456-8080-027ecd31890a)

# Paper Abstract

This paper presents a rigorous evaluation of Real-to-Sim parameter estimation approaches for fabric manipulation in robotics. The study systematically assesses three state-of-the-art approaches, namely two differential pipelines and a data-driven approach. We also devise a novel physics-informed neural network approach for physics parameter estimation. These approaches are interfaced with two simulations across multiple Real-to-Sim scenarios (lifting, wind blowing, and stretching) for five different fabric types and evaluated on three unseen scenarios (folding, fling, and shaking). 

# Dataset

&#8226; Dataset can be downloaded at https://gla-my.sharepoint.com/:f:/g/personal/2649534r_student_gla_ac_uk/ElhpVzs8dTRPkxgEM4TtH3kBUrc24hZ7Wm5OVfyUIl-NBA?e=LChmsx

&#8226; Imaging Setup: Two ZED2i cameras were positioned at the front and top to capture synchronized RGB-D image sequences at 15 Hz. For the stretching scenario, an additional ZED2 camera was positioned above the fabric at 30 Hz to mitigate inaccuracies caused by camera tilting.

&#8226; Data Processing: SAM2 was used for fabric segmentation, ensuring that only the observed fabric was retained. Point cloud sequences were generated from the synchronized RGB and depth images based on the cameras' intrinsic and extrinsic parameters.

&#8226; Action Capture: A Rethink Robotics Baxter robot provided the action information. Baxter's zero-G mode was utilized for high-speed scenarios like shaking and fling, while predefined trajectories with linear interpolation ensured constant speed in other scenarios.

&#8226; Dataset Overview: The complete dataset comprises synchronized RGB-D images, point clouds, and action information, totaling around 50 GB.

# Real-to-Sim parameter estimation of fabrics
## Diffcloud
* Followed by Diffcloud setup, ensure that you are first inside the Docker container using the above step.
```
(diffsim_torch3d)# cd pysim
```
#### Lift
```
(diffsim_torch3d)/pysim# python pointcloud1.py
# Will produce a folder lift_exps_diffcloud containing visualizations
```
#### Wind
```
(diffsim_torch3d)/pysim# python real2simwind.py
# Will produce a folder wind_exps_diffcloud containing visualizations
```

## Physnet
#### Lift
```
cd ./Physnet
python main.py
# Will produce a folder figures containing embedding visualizations
```
#### Wind
```
cd ./Physnet
python main1.py
# Will produce a folder figures containing embedding visualizations
```

## PINN
#### Stretch for diffsim simulator
```
cd ./PINN
python pinngrey.py
```
#### Stretch for difftaichi simulator
```
cd ./PINN
python pinntaichi1.py
```

# Sim-to-Real evaluation on different scenarios
## For visualization and metrics
#### Fold
```
cd ./Sim2Real visualization and metrics
python foldsim2real.py
```
#### Fling
```
cd ./Sim2Real visualization and metrics
python flingsim2real2.py
```
#### Shake
```
cd ./Sim2Real visualization and metrics
python shakesim2real2.py
```
