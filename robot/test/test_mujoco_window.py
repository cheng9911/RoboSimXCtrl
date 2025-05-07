
import mujoco
import time

# 加载 MuJoCo 模型
model = mujoco.MjModel.from_xml_path("/home/sun/Documents/GitHub/RoboSimXCtrl/assets/mujoco/diana7/scene.xml")
data = mujoco.MjData(model)

# 离屏渲染测试
renderer = mujoco.Renderer(model, 800, 600)
renderer.update_scene(data)
rgb = renderer.render()
print(f"渲染结果尺寸: {rgb.shape}")

# 窗口模式测试
with mujoco.viewer.launch(model, data, headless=False) as v:
    for _ in range(100):
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)