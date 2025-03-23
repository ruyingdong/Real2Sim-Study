import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import open3d as o3d
import math
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize_pointcloud(points):
    """
    只将点云移动到质心，不进行缩放
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    return torch.from_numpy(centered_points).float().to(device)

# 存储第一帧的缩放参数
first_frame_params = None

def align_pointclouds(pcls, is_first_frame=False):
    """
    对齐多个点云到同一个参考系统，使用第一帧的参数。
    对于所有来自 OBJ 的点云（dirs 列表中前 5 个），均执行 x,y 坐标交换及 180° 旋转。
    """
    global first_frame_params
    aligned_pcls = []
    
    if is_first_frame:
        first_frame_params = []
        for pcl in pcls:
            points = pcl.cpu().numpy()
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
            first_frame_params.append((centroid, max_dist))
    
    for i, pcl in enumerate(pcls):
        points = pcl.cpu().numpy()
        if is_first_frame or first_frame_params is None:
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
        else:
            centroid, max_dist = first_frame_params[i]
        centered_points = points - centroid
        normalized_points = centered_points / max_dist
        
        # 对于所有 OBJ 数据（索引 < 5）均做坐标转换
        if i < 5:
            swapped_points = normalized_points.copy()
            # 交换 x 与 y
            swapped_points[:, 0] = normalized_points[:, 1]
            swapped_points[:, 1] = normalized_points[:, 0]
            theta = np.pi
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            normalized_points = np.dot(swapped_points, rotation_matrix.T)
        
        norm_pcl = torch.from_numpy(normalized_points).float().to(device)
        # 缩放并平移到目标位置
        norm_pcl = norm_pcl * 2
        norm_pcl = norm_pcl + torch.tensor([2.5, 1.75, 0.0]).to(device)
        aligned_pcls.append(norm_pcl)
    
    return aligned_pcls

def convert_objs_to_pcls(obj_dir, out_dir, num_points=5000):
    """将 OBJ 文件转换为点云文件"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.endswith('.obj')])
    print(f"Found {len(obj_files)} OBJ files in directory {obj_dir}")
    for obj_file in obj_files:
        try:
            if '_' in obj_file:
                frame_num = int(obj_file.split('_')[0])
            else:
                frame_num = int(''.join(filter(str.isdigit, obj_file[:4])))
            verts, faces, aux = load_obj(os.path.join(obj_dir, obj_file))
            mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
            sampled_points = sample_points_from_meshes(mesh, num_points)
            out_file = os.path.join(out_dir, f'{frame_num}.npy')
            np.save(out_file, sampled_points.squeeze(0).cpu().numpy())
            print(f"Processed {obj_file} -> {out_file}")
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            continue

def sample_pcd(pcd, num_points):
    """从 PCD 点云中采样指定数量的点"""
    points = np.asarray(pcd.points)
    if len(points) == num_points:
        return points
    elif len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]
    else:
        indices = np.random.choice(len(points), num_points, replace=True)
        return points[indices]

def load_pointcloud(file_path, num_points=None):
    """加载点云（支持 .npy, .txt, .pcd 格式）"""
    try:
        if file_path.endswith('.npy'):
            points = torch.from_numpy(np.load(file_path)).float().to(device)
        elif file_path.endswith('.txt'):
            points = np.loadtxt(file_path)
            if num_points is not None:
                if points.shape[0] > num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=False)
                    points = points[indices]
                elif points.shape[0] < num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=True)
                    points = points[indices]
            points = torch.from_numpy(points).float().to(device)
        elif file_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(file_path)
            if num_points is not None:
                points_np = sample_pcd(pcd, num_points)
                points = torch.from_numpy(points_np).float().to(device)
            else:
                points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        print(f"Loaded {file_path}: {points.shape[0]} points")
        return points
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def plot_all_views(pcls, labels, title=""):
    """
    将各个单独视图和一个叠加视图排列在一个网格中，所有子图大小一致。
    每行4个子图，总数 = 单独视图数 + 1（叠加图）。
    """
    num_individual = len(pcls)
    num_total = num_individual + 1
    num_cols = 5
    num_rows = math.ceil(num_total / num_cols)
    
    custom_colors = [
        (0.6, 0.0, 0.0, 1.0),   # Diffcloud(lifting)
        (0.0, 0.6, 0.0, 1.0),   # PINN on Diffsim(stretching)
        (0.0, 0.0, 0.6, 1.0),   # Physnet on Diffsim(lifting)
        (0.8, 0.4, 0.0, 1.0),   # Physnet on Diffsim(wind)
        (0.6, 0.0, 0.6, 1.0),   # Diffcloud(wind)
        (0.0, 0.6, 0.6, 1.0),   # Diffcp(lifting)
        (0.5, 0.5, 0.0, 1.0),   # PINN on Diffcp(stretching)
        (0.7, 0.7, 0.0, 1.0),   # New txt_dir3
        (0.0, 0.0, 0.0, 1.0)    # Real Point Cloud
    ]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows), subplot_kw={'projection': '3d'})
    canvas = FigureCanvas(fig)
    axes = np.array(axes).reshape(-1)

    for idx in range(num_individual):
        ax = axes[idx]
        pcl = pcls[idx]
        x, y, z = pcl.cpu().numpy().T
        ax.scatter(x, y, z, s=0.2, c=[custom_colors[idx]], label=labels[idx])
        ax.set_title(labels[idx])
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([-1, 5])
        ax.view_init(90, -90)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    ax_overlay = axes[num_individual]
    for idx in range(num_individual):
        pcl = pcls[idx]
        x, y, z = pcl.cpu().numpy().T
        ax_overlay.scatter(x, y, z, s=0.2, c=[custom_colors[idx]], label=labels[idx])
    ax_overlay.set_title("Overlaid")
    ax_overlay.set_xlim([0, 5])
    ax_overlay.set_ylim([0, 5])
    ax_overlay.set_zlim([-1, 5])
    ax_overlay.view_init(90, -90)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.set_zticks([])
    
    for j in range(num_total, len(axes)):
        axes[j].axis('off')
    
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def run_visualization(dirs, out_dir, total_steps, num_points):
    """直接生成视频"""
    video_writer = None
    last_frame = None
    
    for step in range(1, total_steps + 1):
        print(f"Processing step {step}")
        pcls = []
        for dir_path, format_type in dirs:
            file_path = os.path.join(dir_path, f'{step}.{format_type}')
            pcl = load_pointcloud(file_path, num_points if format_type in ['txt', 'pcd'] else None)
            if pcl is None: continue
            pcls.append(pcl)
        
        aligned_pcls = align_pointclouds(pcls, is_first_frame=(step == 1))
        frame = plot_all_views(aligned_pcls, custom_labels)
        
        if video_writer is None:
            h, w, _ = frame.shape
            video_writer = cv2.VideoWriter(
                os.path.join(out_dir, 'foldbrownnew1.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                25,  # 帧率
                (w, h)
            )
        
        video_writer.write(frame)
        last_frame = frame.copy()

    # 最后一帧停留3秒（25fps * 3s = 75帧）
    if last_frame is not None:
        for _ in range(75):
            video_writer.write(last_frame)
    
    if video_writer is not None:
        video_writer.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--txt_dir1', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/brown', 
                        help='First directory containing TXT point clouds')
    parser.add_argument('-t2', '--txt_dir2', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/brownfoldpinn', 
                        help='Second directory containing TXT point clouds')
    parser.add_argument('-t3', '--txt_dir3', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/fleekfoldpinn', 
                        help='Third directory containing TXT point clouds')
    parser.add_argument('-p', '--pcd_dir', type=str, 
                        default='/home/david/Downloads/baxter 1/baxter/fold_brown45/ruog1', 
                        help='Directory containing PCD point clouds')
    parser.add_argument('-o1', '--obj_dir1', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/PINN-stretching-diffsim/brownpinn1',
                        help='First directory containing OBJ files')
    parser.add_argument('-o2', '--obj_dir2', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Diffcloud - wind/diffsimwindbrown',
                        help='Second directory containing OBJ files')
    parser.add_argument('-o3', '--obj_dir3', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Physnet-lift/brownphysnet1 1/brownphysnet1',
                        help='Third directory containing OBJ files')
    parser.add_argument('-o4', '--obj_dir4', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Physnet-wind/brownphysnetfoldwindsceanrio',
                        help='Fourth directory containing OBJ files')
    parser.add_argument('-o5', '--obj_dir5', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Diffcloud-lift/browndiffsim',
                        help='Fifth directory containing OBJ files')
    parser.add_argument('-s', '--steps', type=int, default=118, 
                        help='Total simulation steps')
    parser.add_argument('--out_dir', type=str, default='allfold-redtrytry', 
                        help='Output directory')
    parser.add_argument('--points', type=int, default=10000, 
                        help='Number of points to sample from meshes (also used for TXT/PCD sampling)')
    
    args = parser.parse_args()
    
    # 自定义标签
    custom_labels = [
        'Diffcloud(lifting)',
        'PINN on Diffsim(stretching)',
        'Physnet on Diffsim(lifting)',
        'Physnet on Diffsim(wind)',
        'Diffcloud(wind)',
        'Diffcp(lifting)',
        'PINN on Diffcp(stretching)',
        'Physnet on Diffcp(lifting)',  # txt_dir3对应的标签
        'Real Point Cloud'
    ]
    
    # 转换OBJ文件
    obj_pcl_dir1 = os.path.join(args.out_dir, 'obj_pcls1')
    obj_pcl_dir2 = os.path.join(args.out_dir, 'obj_pcls2')
    obj_pcl_dir3 = os.path.join(args.out_dir, 'obj_pcls3')
    obj_pcl_dir4 = os.path.join(args.out_dir, 'obj_pcls4')
    obj_pcl_dir5 = os.path.join(args.out_dir, 'obj_pcls5')
    
    convert_objs_to_pcls(args.obj_dir1, obj_pcl_dir1, args.points)
    convert_objs_to_pcls(args.obj_dir2, obj_pcl_dir2, args.points)
    convert_objs_to_pcls(args.obj_dir3, obj_pcl_dir3, args.points)
    convert_objs_to_pcls(args.obj_dir4, obj_pcl_dir4, args.points)
    convert_objs_to_pcls(args.obj_dir5, obj_pcl_dir5, args.points)
    
    # 更新dirs顺序
    dirs = [
        (obj_pcl_dir1, 'npy'),
        (obj_pcl_dir2, 'npy'),
        (obj_pcl_dir3, 'npy'),
        (obj_pcl_dir4, 'npy'),
        (obj_pcl_dir5, 'npy'),
        (args.txt_dir1, 'txt'),
        (args.txt_dir2, 'txt'),
        (args.txt_dir3, 'txt'),
        (args.pcd_dir, 'pcd')
    ]
    
    run_visualization(dirs, args.out_dir, args.steps, args.points)
    print("Visualization completed.")