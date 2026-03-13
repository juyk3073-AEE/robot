import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

WORK_DIR = r"C:\Users\juyk3\project\physical ai"
blueprint = "blueprint_2leg_wheel_v1.urdf"
save_name = "ppo_2leg_wheel_v3"
model_path = os.path.join(WORK_DIR, save_name)

class TestLeg2WheelEnv(gym.Env):
    def __init__(self, render=True):
        super(TestLeg2WheelEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = os.path.join(WORK_DIR, blueprint)
        self.robotId = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240., numSubSteps=8, numSolverIterations=100, physicsClientId=self.client)
        
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(planeId, -1, lateralFriction=1.5, physicsClientId=self.client)

        self.robotId = p.loadURDF(self.urdf_path, [0, 0, 0.72], useFixedBase=False, physicsClientId=self.client)
        
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robotId, physicsClientId=self.client)):
            name = p.getJointInfo(self.robotId, i, physicsClientId=self.client)[1].decode('utf-8')
            self.joint_indices[name] = i
            p.changeDynamics(self.robotId, i, lateralFriction=1.5, physicsClientId=self.client)
            
        target_height = 0.71
        D = np.clip(target_height - 0.11, 0.01, 0.6)
        cos_val = np.clip((D**2 - 0.18) / 0.18, -1.0, 1.0)
        knee_target = -math.acos(cos_val)
        hip_target = -knee_target / 2.0
        
        for j in ['hip_left_joint', 'hip_right_joint']: p.resetJointState(self.robotId, self.joint_indices[j], hip_target, physicsClientId=self.client)
        for j in ['knee_left_joint', 'knee_right_joint']: p.resetJointState(self.robotId, self.joint_indices[j], knee_target, physicsClientId=self.client)
        for j in ['wheel_left_joint', 'wheel_right_joint']: p.resetJointState(self.robotId, self.joint_indices[j], 0.0, targetVelocity=0.0, physicsClientId=self.client)

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
            targetVelocities=[0, 0],
            forces=[15, 15],
            physicsClientId=self.client
        )
        
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            pos, _ = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
            p.resetBasePositionAndOrientation(self.robotId, [0, 0, pos[2]], [0, 0, 0, 1], physicsClientId=self.client)
            p.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        return np.array([
            base_euler[1], base_ang_vel[1], 
            base_pos[0], base_vel[0], # 테스트 코드에도 X축 위치(base_pos[0])가 반드시 포함되어야 함
            base_pos[2], base_vel[2]
        ], dtype=np.float32)

    def step(self, action):
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
        
        if abs(pitch) > 1.5 or abs(roll) > 1.5 or z_height < 0.01:
            terminated = True
            reward = -20.0 
            
        return obs, reward, terminated, False, {}

if __name__ == "__main__":
    test_env = TestLeg2WheelEnv(render=True)
    obs, _ = test_env.reset()

    print(f"{model_path}.zip 모델을 불러옵니다...")
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
    else:
        print("모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        exit()

    mode_slider = p.addUserDebugParameter("Manual Height Override (0=AI, 1=Manual)", 0, 1, 1)
    height_slider = p.addUserDebugParameter("Target Body Height (m)", 0.26, 0.71, 0.71)

    print("\nAI 제어 시작! 슬라이더를 움직여보세요. (스무딩 필터 적용됨)")

    # [핵심] 현재 위치를 기억할 스무딩 변수 초기화 (초기 스폰 높이 0.71m 적용)
    current_smoothed_height = 0.71
    # 스무딩 계수: 1.0이면 즉각 이동, 0에 가까울수록 느리고 부드럽게 보간됨
    alpha = 0.03 

    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        manual_mode = p.readUserDebugParameter(mode_slider)
        raw_target_height = p.readUserDebugParameter(height_slider)
        
        # [핵심] 지수 이동 평균(EMA)을 이용한 높이 보간 연산
        current_smoothed_height += (raw_target_height - current_smoothed_height) * alpha
        
        if manual_mode >= 0.5:
            # 부드럽게 변환된 높이 값을 AI 액션으로 역산
            action[0] = (current_smoothed_height - 0.485) / 0.225
            
        obs, reward, done, _, _ = test_env.step(action)
        time.sleep(1./60.) 
        
        if done:
            print("로봇이 균형을 잃고 쓰러졌습니다. 리셋합니다.")
            obs, _ = test_env.reset()
            # 쓰러진 후 리셋 시 스무딩 변수도 원위치 복구
            current_smoothed_height = 0.71