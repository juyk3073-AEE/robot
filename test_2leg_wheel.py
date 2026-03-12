import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# [설정] 경로 및 모델 이름 지정
WORK_DIR = r"C:\Users\juyk3\project\physical ai"
blueprint = "blueprint_2leg_wheel_v1.urdf"
save_name = "ppo_2leg_wheel_v2"
model_path = os.path.join(WORK_DIR, save_name)

# [핵심] 학습 파일에 의존하지 않는 테스트 전용 독립 환경 구성
class TestLeg2WheelEnv(gym.Env):
    def __init__(self, render=True):
        super(TestLeg2WheelEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = os.path.join(WORK_DIR, blueprint)
        self.robotId = p.loadURDF(self.urdf_path, [0, 0, 0.75], useFixedBase=False, physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240., numSubSteps=8, numSolverIterations=100, physicsClientId=self.client)
        
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(planeId, -1, lateralFriction=1.5, physicsClientId=self.client)

        self.robotId = p.loadURDF(self.urdf_path, [0, 0, 0.7], useFixedBase=False, physicsClientId=self.client)
        
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robotId, physicsClientId=self.client)):
            name = p.getJointInfo(self.robotId, i, physicsClientId=self.client)[1].decode('utf-8')
            self.joint_indices[name] = i
            p.changeDynamics(self.robotId, i, lateralFriction=1.5, physicsClientId=self.client)
            
        # [충돌 방지] 자유낙하 시 죽지 않도록 초기 50프레임 동안 자세 락(Lock)
        target_height = 0.55
        D = np.clip(target_height - 0.11, 0.01, 0.6)
        cos_val = np.clip((D**2 - 0.18) / 0.18, -1.0, 1.0)
        knee_target = -math.acos(cos_val)
        hip_target = -knee_target / 2.0
        
        p.setJointMotorControlArray(
            self.robotId, 
            jointIndices=[self.joint_indices['hip_left_joint'], self.joint_indices['hip_right_joint'], self.joint_indices['knee_left_joint'], self.joint_indices['knee_right_joint']],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[hip_target, hip_target, knee_target, knee_target],
            forces=[50, 50, 50, 50],
            physicsClientId=self.client
        )
        
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        return np.array([
            base_euler[1], base_ang_vel[1], 
            base_vel[0], base_pos[2], base_vel[2]
        ], dtype=np.float32)

    def step(self, action):
        # [테스트 전용] 학습 시 고정되었던 높이를 해제하여 사용자의 슬라이더(action[0]) 입력을 받음
        target_height = 0.485 + action[0] * 0.225
        wheel_vel = action[1] * 20.0
        
        D = np.clip(target_height - 0.11, 0.01, 0.6)
        cos_val = np.clip((D**2 - 0.18) / 0.18, -1.0, 1.0)
        knee_target = -math.acos(cos_val)
        hip_target = -knee_target / 2.0
        
        p.setJointMotorControlArray(
            self.robotId, 
            jointIndices=[self.joint_indices['hip_left_joint'], self.joint_indices['hip_right_joint'], self.joint_indices['knee_left_joint'], self.joint_indices['knee_right_joint']],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[hip_target, hip_target, knee_target, knee_target],
            forces=[50, 50, 50, 50],
            physicsClientId=self.client
        )
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=[self.joint_indices['wheel_left_joint'], self.joint_indices['wheel_right_joint']],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[wheel_vel, wheel_vel],
            forces=[15, 15],
            physicsClientId=self.client
        )
        
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_obs()
        pitch, x_vel, z_height = obs[0], obs[2], obs[3]
        base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)[1]
        roll = p.getEulerFromQuaternion(base_orn)[0]
        
        terminated = False
        reward = 1.0 
        
        # 사망 조건 (v2 기준 반영)
        if abs(pitch) > 1.5 or abs(roll) > 1.5 or z_height < 0.01:
            terminated = True
            reward = -20.0 
            
        return obs, reward, terminated, False, {}


# --- 메인 실행부 ---
if __name__ == "__main__":
    test_env = TestLeg2WheelEnv(render=True)
    obs, _ = test_env.reset()

    print(f"{model_path}.zip 모델을 불러옵니다...")
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
    else:
        print("모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        exit()

    # 1.0 입력 시 완벽한 수동 개입 모드 동작
    mode_slider = p.addUserDebugParameter("Manual Height Override (0=AI, 1=Manual)", 0, 1, 1)
    height_slider = p.addUserDebugParameter("Target Body Height (m)", 0.26, 0.71, 0.71)
    
    print("\nAI 제어 시작! 슬라이더를 1.0으로 맞추고 로봇의 높이를 제어하세요.")

while True:
        action, _states = model.predict(obs, deterministic=True)
        
        manual_mode = p.readUserDebugParameter(mode_slider)
        manual_height = p.readUserDebugParameter(height_slider)
        
        # 수동 모드(0.5 이상)일 경우 AI의 높이 명령을 슬라이더 값으로 변조
        if manual_mode >= 0.5:
            action[0] = (manual_height - 0.485) / 0.225
            
        obs, reward, done, _, _ = test_env.step(action)
        
        # [디버깅용 출력 코드 삽입 위치] step 연산 직후 센서 값 확인
        print(f"Pitch: {obs[0]:.2f}, Roll: {p.getEulerFromQuaternion(p.getBasePositionAndOrientation(test_env.robotId)[1])[0]:.2f}, Z: {obs[3]:.2f}")

        time.sleep(1./60.) 
        
        if done:
            print("로봇이 균형을 잃고 쓰러졌습니다. 리셋합니다.")
            obs, _ = test_env.reset()