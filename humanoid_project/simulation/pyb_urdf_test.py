"""Simple PyBullet test script to load a humanoid URDF model."""

import pybullet as p, pybullet_data  # type: ignore

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("../assets/humanoid/humanoid.urdf", [0, 0, 1])
input("Press Enter to exit...")
