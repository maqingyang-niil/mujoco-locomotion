from pathlib import Path

import mujoco


scene = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "anybotics_anymal_c"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

# 读取 12 个执行器各自对应的关节角度。
joint_ids = model.actuator_trnid[:, 0]
joint_positions = model.jnt_qposadr[joint_ids]
data.ctrl[:] = data.qpos[joint_positions]

print("执行器数量", model.nu)
print("初始机身高度", data.qpos[2])

for _ in range(100):
    mujoco.mj_step(model, data)

print("仿真时间", data.time)
print("步进后机身高度", data.qpos[2])