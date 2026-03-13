import pybullet as p
import pybullet_data
import time
import os
import math

model_name= "blueprint_2leg_wheel_v1.urdf"

# 1. GUI 모드로 물리 엔진 연결
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. 바닥 로드 및 마찰력 설정
planeId = p.loadURDF("plane.urdf")
p.changeDynamics(planeId, -1, lateralFriction=1.5)

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, model_name)

# 공중에 살짝 띄워서 스폰
robotId = p.loadURDF(urdf_path, [0, 0, 0.7], useFixedBase=False)

joint_indices = {}
for i in range(p.getNumJoints(robotId)):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_indices[joint_name] = i
    p.changeDynamics(robotId, i, lateralFriction=1.5)

print("--- 역기구학(IK) 모드 적용 완료 ---")

# 4. GUI 슬라이더 생성
# 다리 길이 0.6m(30cm+30cm) + 바퀴 반지름(6cm) + 몸통 중심점 차이(5cm)
# 따라서 로봇의 키는 대략 0.26m(완전 굽힘) ~ 0.71m(완전 폄) 사이를 오갈 수 있습니다.
height_slider = p.addUserDebugParameter("Target Body Height (m)", 0.26, 0.71, 0.6)
wheel_vel_slider = p.addUserDebugParameter("Symmetric Wheel Velocity", -20.0, 20.0, 0)

while True:
    # 슬라이더에서 목표 '키(Height)' 읽기
    target_height = p.readUserDebugParameter(height_slider)
    wheel_vel_target = p.readUserDebugParameter(wheel_vel_slider)
    
    # [역기구학 계산]
    # D: 힙 관절부터 바퀴 중심까지의 직선 거리
    D = target_height - 0.11 # 몸통 오프셋(0.05) + 바퀴 반지름(0.06) 제외
    D = max(0.01, min(0.6, D)) # 에러 방지용 리미트 (다리 최대 길이 0.6m)
    
    # 제2 코사인 법칙을 이용한 무릎 각도 계산 (허벅지 0.3m, 종아리 0.3m)
    # L1^2 + L2^2 = 0.09 + 0.09 = 0.18
    cos_val = (D**2 - 0.18) / 0.18
    cos_val = max(-1.0, min(1.0, cos_val)) # 도메인 초과 방지
    
    knee_target = -math.acos(cos_val) # 무릎은 항상 뒤로 꺾이므로 음수(-) 적용
    hip_target = -knee_target / 2.0   # 발이 수직 아래를 유지하도록 힙 각도 자동 보정
    
    # 5. 모터 제어 하달
    p.setJointMotorControlArray(
        robotId, 
        jointIndices=[
            joint_indices['hip_left_joint'], joint_indices['hip_right_joint'],
            joint_indices['knee_left_joint'], joint_indices['knee_right_joint']
        ],
        controlMode=p.POSITION_CONTROL,
        targetPositions=[hip_target, hip_target, knee_target, knee_target],
        forces=[50, 50, 50, 50] 
    )
    
    p.setJointMotorControlArray(
        robotId,
        jointIndices=[
            joint_indices['wheel_left_joint'], joint_indices['wheel_right_joint']
        ],
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=[wheel_vel_target, wheel_vel_target],
        forces=[15, 15] 
    )
    
    p.stepSimulation()
    time.sleep(1./240.)