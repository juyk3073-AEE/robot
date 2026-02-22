#관절 조절
import pybullet as p
import pybullet_data
import time
import os

# 1. 시뮬레이터 연결 및 환경 설정
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "leg_2dof.urdf")

# 본체를 공중에 고정
startPos = [0, 0, 1.0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

# 2. UI 슬라이더(Debug Parameter) 생성
# p.addUserDebugParameter("이름", 최솟값, 최댓값, 초기값)
hip_slider = p.addUserDebugParameter("Hip Pitch Angle", -1.57, 1.57, 0.0)
knee_slider = p.addUserDebugParameter("Knee Pitch Angle", -2.5, 0.0, 0.0)

print("수동 제어 시작: 우측 Params 패널의 슬라이더를 조작하십시오.")

# 3. 제어 루프
while True:
    # ① 슬라이더에서 현재 설정된 목표 각도(Target Position) 읽어오기
    hip_target = p.readUserDebugParameter(hip_slider)
    knee_target = p.readUserDebugParameter(knee_slider)

    # ② 모터에 목표 각도 인가 (내부 PID 제어기에 의한 위치 제어)
    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target)
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target)

    p.stepSimulation()
    time.sleep(1./240.)