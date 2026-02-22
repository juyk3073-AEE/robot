# 가상 센서 데이터 추출 (상태 관측)
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
urdf_path = os.path.join(current_dir, "leg_2dof_4.urdf")

# 강화학습 환경 구성을 위해 본체 고정을 해제(useFixedBase=False)하고 공중에서 생성
startPos = [0, 0, 1.0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=False)

print("상태 관측 시작: 데이터 스트리밍을 터미널에서 확인하십시오.")

# 2. 데이터 수집 루프
while True:
    p.stepSimulation()

    # ① IMU 센서 모사: 본체의 위치 및 회전 데이터 파싱
    base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
    # 쿼터니언(Quaternion)을 직관적인 오일러 각(Euler Angle)으로 변환
    base_euler = p.getEulerFromQuaternion(base_orn) 
    
    # ② 자이로스코프 모사: 본체의 선속도 및 각속도 파싱
    base_vel, base_ang_vel = p.getBaseVelocity(robotId)

    # ③ 모터 엔코더 모사: 각 관절의 현재 상태 파싱 (0: Hip, 1: Knee)
    hip_state = p.getJointState(robotId, 0)
    knee_state = p.getJointState(robotId, 1)

    # ④ 관측 벡터(Observation Vector) 구성
    # 강화학습 모델의 입력(Input)으로 사용될 핵심 1D 배열 데이터
    pitch_angle = base_euler[1]      # Y축 회전 (앞뒤 기울기, rad)
    pitch_rate = base_ang_vel[1]     # Y축 각속도 (rad/s)
    hip_angle = hip_state[0]         # 고관절 현재 위치 (rad)
    knee_angle = knee_state[0]       # 무릎 현재 위치 (rad)

    print(f"Pitch: {pitch_angle:+.3f} | Pitch Rate: {pitch_rate:+.3f} | Hip: {hip_angle:+.3f} | Knee: {knee_angle:+.3f}")
    
    time.sleep(1./240.)