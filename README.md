# Can Real-to-Sim Approaches Capture Dynamic Fabric Behavior for Robotic Fabric Manipulation?

![image](https://github.com/user-attachments/assets/80cba701-7b98-4456-8080-027ecd31890a)

# Paper Abstract

This paper presents a rigorous evaluation of Real-to-Sim parameter estimation approaches for fabric manipulation in robotics. The study systematically assesses three state-of-the-art approaches, namely two differential pipelines and a data-driven approach. We also devise a novel physics-informed neural network approach for physics parameter estimation. These approaches are interfaced with two simulations across multiple Real-to-Sim scenarios (lifting, wind blowing, and stretching) for five different fabric types and evaluated on three unseen scenarios (folding, fling, and shaking). 

# Dataset

&#8226;Dataset can be downloaded at 

Imaging Setup: Two ZED2i cameras were positioned at the front and top to capture synchronized RGB-D image sequences at 15 Hz. For the stretching scenario, an additional ZED2 camera was positioned above the fabric at 30 Hz to mitigate inaccuracies caused by camera tilting.

Data Processing: SAM2 was used for fabric segmentation, ensuring that only the observed fabric was retained. Point cloud sequences were generated from the synchronized RGB and depth images based on the cameras' intrinsic and extrinsic parameters.

Action Capture: A Rethink Robotics Baxter robot provided the action information. Baxter's zero-G mode was utilized for high-speed scenarios like shaking and fling, while predefined trajectories with linear interpolation ensured constant speed in other scenarios.

Dataset Overview: The complete dataset comprises synchronized RGB-D images, point clouds, and action information, totaling around 50 GB.
