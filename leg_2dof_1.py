import pybullet as p
import pybullet_data
import time
import math
import os

# 1. 시뮬레이터 연결 및 환경 설정
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "leg_2dof.urdf")

# 본체가 떨어지지 않도록 공중에 고정 (스탠드에 매달린 상태)
startPos = [0, 0, 1.0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

print("2관절 로봇 다리 모터 제어 시작...")

# 2. 제어 루프
t = 0
while True:
    # ① 목표 각도 궤적 생성 (라디안 단위)
    hip_target_angle = 0.5 * math.sin(t * 2)         # 고관절: -0.5 ~ 0.5 라디안 스윙
    knee_target_angle = -0.5 + 0.5 * math.sin(t * 2)   # 무릎: -1.0 ~ 0.0 라디안 굽힘

    # ② 각 모터에 위치 제어 명령 인가 (내부적으로 PD 제어기를 통해 토크 연산됨)
    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target_angle)
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target_angle)

    p.stepSimulation()
    time.sleep(1./240.)
    t += 1./240.