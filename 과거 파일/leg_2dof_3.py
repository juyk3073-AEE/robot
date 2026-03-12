#역기구학 관절 조절
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

startPos = [0, 0, 1.0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

# 2. 역기구학 설정
# 말단 장치(End-Effector) 링크 인덱스 확인: 'calf'(종아리)는 두 번째 링크이므로 인덱스 1
END_EFFECTOR_INDEX = 2 

# 목표 X, Z 좌표를 입력받는 UI 슬라이더 생성 (Y는 0으로 고정)
# 다리 전체를 쭉 폈을 때 길이가 약 0.6m 이므로 Z값 범위를 -0.6 ~ -0.1로 설정
x_slider = p.addUserDebugParameter("Target X (Forward/Backward)", -0.4, 0.4, 0.0)
z_slider = p.addUserDebugParameter("Target Z (Height)", -0.6, -0.1, -0.4)

print("역기구학 제어 시작: 슬라이더로 발끝의 X, Z 좌표를 조작하십시오.")
# --- 2. 역기구학 설정 아래, 3. 제어 루프 진입 전(while True 위)에 추가 ---
marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 1])
marker_id = p.createMultiBody(baseVisualShapeIndex=marker_visual, basePosition=[0, 0, 0])

# 3. 제어 루프
while True:
    # ① 슬라이더에서 목표 좌표(X, Z) 읽기
    target_x = p.readUserDebugParameter(x_slider)
    target_z = p.readUserDebugParameter(z_slider)
    target_pos = [target_x, 0, target_z + 1.0]

    # 디버깅 마커의 위치를 슬라이더 목표 좌표(target_pos)로 실시간 업데이트
    p.resetBasePositionAndOrientation(marker_id, target_pos, [0, 0, 0, 1])

    # ② 역기구학(IK) 연산: 목표 좌표에 도달하기 위한 [고관절 각도, 무릎 각도] 계산
    joint_angles = p.calculateInverseKinematics(robotId, END_EFFECTOR_INDEX, target_pos)
    
    # ③ 계산된 각도를 각 모터에 인가
    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition=joint_angles[0])
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, targetPosition=joint_angles[1])

    p.stepSimulation()
    time.sleep(1./240.)

    