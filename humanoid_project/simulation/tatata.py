import pybullet as p, pybullet_data  # type: ignore

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("task2/humanoid.urdf", [0, 0, 1])
input("Press Enter to exit...")
