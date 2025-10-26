"""Humanoid walking environment using PyBullet and Gymnasium."""

import pybullet as p  # type: ignore
import pybullet_data  # type: ignore
import numpy as np
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
import time


class HumanoidWalkEnv(gym.Env):
    """
    Custom environment to simulate a humanoid in PyBullet.
    """

    def __init__(self, render=True):
        super().__init__()

        self.render_mode = render
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground and humanoid model
        self.plane_id = p.loadURDF("plane.urdf")
        self.humanoid_id = p.loadURDF("task2/humanoid.urdf", [0, 0, 1.0])

        # Identify movable joints
        self.num_joints = p.getNumJoints(self.humanoid_id)
        self.joint_indices = [
            j
            for j in range(self.num_joints)
            if p.getJointInfo(self.humanoid_id, j)[2] != p.JOINT_FIXED
        ]

        # Define action & observation spaces
        n_act = len(self.joint_indices)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(n_act * 2,), dtype=np.float32
        )

    # ----------------------------------------------------------

    def reset(self, initial_pose=None):
        """Reset simulation. Optionally set humanoid to an initial pose θ_init."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.humanoid_id = p.loadURDF("task2/humanoid.urdf", [0, 0, 1.0])

        if initial_pose is not None:
            for i, j in enumerate(self.joint_indices):
                p.resetJointState(self.humanoid_id, j, initial_pose[i])

        obs = self._get_observation()
        return obs, {}

    # ----------------------------------------------------------

    def step(self, action):
        """Apply torques (actions) and step simulation."""
        for i, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=j,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * 50,  # scale torque
            )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # 240 Hz physics

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done = self._check_termination()
        info = {}

        return obs, reward, done, False, info

    # ----------------------------------------------------------

    def _get_observation(self):
        """Return joint positions and velocities."""
        joint_states = p.getJointStates(self.humanoid_id, self.joint_indices)
        q = np.array([s[0] for s in joint_states])
        qdot = np.array([s[1] for s in joint_states])
        return np.concatenate([q, qdot])

    def _compute_reward(self, obs):
        """Simple reward: encourage upright torso height."""
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        height = base_pos[2]
        return height  # higher → better

    def _check_termination(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        return base_pos[2] < 0.3  # if it falls

    # ----------------------------------------------------------

    def close(self):
        p.disconnect()


# ----------------------------------------------------------

if __name__ == "__main__":
    env = HumanoidWalkEnv(render=True)
    obs, _ = env.reset()
    print("Initial observation:", obs)

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            print("Humanoid fell! Resetting...")
            obs, _ = env.reset()
    env.close()
