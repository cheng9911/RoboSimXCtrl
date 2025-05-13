import open3d as o3d
import numpy as np
import threading
import time
import queue

point_cloud_queue = queue.Queue()


def generate_fake_point_cloud():
    """
    生成有明显颜色区分的点云：红、绿、蓝点交替出现
    """
    while True:
        num_points = 3000
        xyz = np.random.rand(num_points, 3) * 2 - 1  # 点云位置 [-1, 1]

        # 制造颜色：红 绿 蓝 重复出现
        rgb = np.zeros((num_points, 3))
        rgb[::3] = [1, 0, 0]   # 红
        rgb[1::3] = [0, 1, 0] # 绿
        rgb[2::3] = [0, 0, 1] # 蓝

        point_cloud = np.hstack((xyz, rgb))
        point_cloud_queue.put(point_cloud)
        time.sleep(0.1)


def visualizer_loop():
    vis = o3d.visualization.Visualizer()
    vis.create_window("彩色点云")
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    while True:
        try:
            cloud = point_cloud_queue.get(timeout=1)
            points = cloud[:, :3]
            colors = cloud[:, 3:]

            # 明确设定为 float32
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
            # 更新点云
            vis.add_geometry(pcd)
            vis.run()
            

            # vis.update_geometry(pcd)
            # vis.poll_events()
            # vis.update_renderer()
        except queue.Empty:
            print("等待点云...")
            continue

    vis.destroy_window()


if __name__ == "__main__":
    threading.Thread(target=generate_fake_point_cloud, daemon=True).start()
    visualizer_loop()


# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# if __name__ == '__main__':
#     # 加载点云数据
#     points = np.random.rand(1000, 3)
    
#     # 注意，我们只需要一个点云，即points需要是[N, 3]
#     # points = points[0, :, :]  # 里面有多个点云 [5, 8192, 3]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])

#     # 获取高度范围
#     min_height = np.min(points[:, 2])
#     max_height = np.max(points[:, 2])

#     # 根据高度值计算归一化的值
#     normalized_heights = (points[:, 2] - min_height) / (max_height - min_height)

#     # 创建彩虹色映射
#     cmap = plt.cm.get_cmap('rainbow')

#     # 将归一化的高度值映射到彩虹色映射上
#     gradient_colors = cmap(normalized_heights)

#     # 将颜色数组赋值给点云对象
#     pcd.colors = o3d.utility.Vector3dVector(gradient_colors[:, :3])

#     # 显示点云
#     o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization",
#                                       point_show_normal=False, width=800, height=600)