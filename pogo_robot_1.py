import pybullet as p
import pybullet_data
import time

# 1. 시뮬레이터 연결 (GUI 창 유지)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81) # 중력 가속도

# 2. 바닥 및 로봇 로드
planeId = p.loadURDF("plane.urdf")

# 로봇을 공중(Z축 1.0m 높이)에서 생성
pogoStartPos = [0, 0, 1.0] 
pogoStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
pogoId = p.loadURDF("pogo.urdf", pogoStartPos, pogoStartOrientation)

# 3. 전기적 제어 초기 설정
# 관절(Joint)의 기본 마찰력을 0으로 설정하여 모터 제어만 받도록 초기화
joint_index = 0
p.setJointMotorControl2(pogoId, joint_index, p.VELOCITY_CONTROL, force=0)

print("시뮬레이션 루프 시작...")
# 4. 물리 연산 루프
while True:
    p.stepSimulation()
    time.sleep(1./240.) # 240Hz 주기로 물리 엔진 업데이트