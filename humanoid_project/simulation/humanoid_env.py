# simulation/humanoid_env.py
import pybullet as p  # type: ignore
import pybullet_data  # type: ignore
import os


class HumanoidWalkEnv:
    def __init__(self, gui=True):
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load the humanoid URDF
        urdf_path = os.path.join("urdf", "humanoid.urdf")
        self.humanoid = p.loadURDF(urdf_path, basePosition=[0, 0, 1])
