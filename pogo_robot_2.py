import pybullet as p
import pybullet_data
import time
import os

# 1. 시뮬레이터 연결 및 초기 설정
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "pogo.urdf")

pogoStartPos = [0, 0, 2.0] 
pogoStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
pogoId = p.loadURDF(urdf_path, pogoStartPos, pogoStartOrientation)

joint_index = 0

# 2. 전기/제어 파라미터 설정 (PD 제어기 게인)
k_p = 3000      # Proportional Gain (가상 스프링 상수, N/m)
k_d = 50        # Derivative Gain (가상 댐핑 계수, Ns/m)
target_pos = -0.05 # 관절 목표 위치 (모터가 유지하려는 길이)

# 관절의 내부 마찰력을 0으로 만들어 외부 토크 제어만 받도록 설정
p.setJointMotorControl2(pogoId, joint_index, p.VELOCITY_CONTROL, force=0)

# 3. 물리 연산 및 제어 루프
while True:
    # ① 센서 데이터 수집: 현재 관절의 위치(x)와 속도(x_dot) 측정
    joint_state = p.getJointState(pogoId, joint_index)
    current_pos = joint_state[0]
    current_vel = joint_state[1]

    # ② 제어기 연산: 목표 위치를 유지하기 위해 모터가 내야 할 힘 계산
    motor_force = k_p * (target_pos - current_pos) - k_d * current_vel

    # ③ 액추에이터 구동: 계산된 힘(Force/Torque)을 모터에 인가
    p.setJointMotorControl2(pogoId, joint_index, p.TORQUE_CONTROL, force=motor_force)

    # 시뮬레이션 1스텝 전진
    p.stepSimulation()
    time.sleep(1./240.)