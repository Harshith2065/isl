import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

robot = p.loadURDF("../assets/mekorama/mekorama.urdf", basePosition=[0, 0, 1])

num_joints = p.getNumJoints(robot)
print("Number of joints:", num_joints)

for i in range(num_joints):
    print(i, p.getJointInfo(robot, i)[1])

while True:
    p.stepSimulation()
    time.sleep(1 / 240)
