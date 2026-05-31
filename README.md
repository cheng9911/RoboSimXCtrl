# RoboSimXCtrl

针对机器人在仿真环境中的运动控制器，实现运动学（Kinematics）、运动规划（Motion Planning）、碰撞检测（Collision Detection）及控制（Control）等功能。

## 功能模块

| 模块 | 说明 |
|------|------|
| `robot/` | 机器人模型定义（UR5e、KUKA IIWA14、Diana7），正/逆运动学 |
| `arm/` | 核心算法库：控制器、运动规划、几何碰撞、振动抑制 |
| `arm/controller/` | 控制器：PID、计算力矩、自适应、前馈 |
| `arm/motion_planning/` | 运动规划：轨迹规划、路径规划（RRT/RRT*）、笛卡尔规划 |
| `arm/geometry/` | 几何计算：碰撞检测（GJK）、旋转表示（SO3/SE3）、基本形状 |
| `arm/vibration_suppression/` | 振动抑制：输入整形（ZV、ZVD） |
| `assets/` | 机器人模型资源（URDF、MuJoCo XML、网格文件） |
| `utils/` | 工具函数（MuJoCo 辅助、机器人工具箱辅助） |
| `pyroboplan/` | 第三方运动规划库（vendored fork） |

## 项目结构

```
RoboSimXCtrl/
├── robot/                  # 机器人模型（UR5e、IIWA14、Diana）
│   ├── robot.py            # Robot 基类（DH 参数）
│   ├── ur5e.py             # UR5e (6-DOF)
│   ├── iiwa14.py           # KUKA IIWA14 (7-DOF)
│   ├── diana.py            # Diana7 (7-DOF, TracIK)
│   └── test/               # 测试/演示脚本
├── arm/                    # 核心算法包（pip 可安装）
│   ├── controller/         # 控制器实现
│   ├── motion_planning/    # 运动规划（轨迹/路径/RRT）
│   ├── geometry/           # 几何与碰撞检测
│   ├── compliance_control/ # 柔顺控制（导纳控制）
│   ├── vibration_suppression/ # 振动抑制
│   ├── constants/          # 数学常量
│   ├── interface/          # 策略模式接口
│   ├── interpolation/      # 插值器
│   ├── utils/              # 工具函数
│   └── assets/             # MuJoCo 机器人模型
├── assets/                 # URDF 机器人描述文件
├── utils/                  # 顶层工具（MuJoCo/RTB 辅助）
├── pyroboplan/             # 第三方运动规划库
├── requirements.txt        # Python 依赖
└── LICENSE                 # MIT 许可证
```

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n robosim python=3.11
conda activate robosim
```

### 2. 安装依赖

```bash
# 核心依赖
conda install pinocchio -c conda-forge
pip install -r requirements.txt

# 可视化（可选）
pip install meshcat panda3d_viewer
conda install gepetto-viewer-corba -c conda-forge

# 安装 arm 包
cd arm
pip install -e .
```

### 3. 安装 pyroboplan（可选）

```bash
# Python >= 3.10
pip install pyroboplan

# Python < 3.10：需从源码安装并修改版本限制
git clone https://github.com/sea-bass/pyroboplan.git
# 修改 pyproject.toml 中的 python 版本要求
pip install -e pyroboplan/
```

## 使用示例

```python
from robot import UR5e, IIWA14, Diana
from spatialmath import SE3

# 创建 UR5e 机器人
ur5e = UR5e()

# 正运动学
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T = ur5e.fkine(q)

# 逆运动学
q_sol = ur5e.ikine(T)
```

MuJoCo 仿真示例见 `robot/test/test_mujoco.py`。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
