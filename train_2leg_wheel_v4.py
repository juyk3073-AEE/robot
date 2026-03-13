# 다리 구부리고 중심잡기 & 현재 위치로 복귀(부족감쇄)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import math
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# [설정] 버전 v3 (동적 높이 변화 적응 학습)
WORK_DIR = r"C:\Users\juyk3\project\physical ai"
blueprint = "blueprint_2leg_wheel_v1.urdf"
save_name = "ppo_2leg_wheel_v4"

CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
LOG_DIR = os.path.join(WORK_DIR, "logs")

class Leg2WheelEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2WheelEnv, self).__init__()
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
            
        # 스폰 시에는 충격 방지를 위해 다리를 편 상태(0.71)로 시작
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
            
        # [추가] 에피소드 루프 제어용 변수
        self.current_step = 0
        # 사인파가 1.0 (높이 0.71m)부터 부드럽게 시작하도록 위상(Phase) 설정
        self.time_step = math.pi / 2.0 
        
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        return np.array([
            base_euler[1], base_ang_vel[1], 
            base_pos[0], base_vel[0], # [추가] base_pos[0] (X축 절대 위치 센서)
            base_pos[2], base_vel[2]
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.time_step += 0.02 # 주기 속도 조절 (약 5초마다 앉았다 일어남 반복)
        
        # [핵심] 사용자가 슬라이더를 흔드는 상황을 시스템적으로 구현
        sine_wave = math.sin(self.time_step)
        target_height = 0.485 + (sine_wave * 0.225) # 0.26m ~ 0.71m 강제 왕복
        
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
        
        # 1. 관측값 언패킹 변수 수정 (x_pos 추가)
        obs = self._get_obs()
        pitch, pitch_vel, x_pos, x_vel, z_height = obs[0], obs[1], obs[2], obs[3], obs[4]
        base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)[1]
        roll = p.getEulerFromQuaternion(base_orn)[0]
        
        terminated = False
        truncated = False
        
        # 2. 보상 함수 수정 (위치 복원력 추가)
        reward = 1.0 
        reward += np.exp(-1.0 * pitch**2) * 1.0  
        reward += np.exp(-2.0 * pitch_vel**2) * 2.0  
        # [핵심] X축 원점(0)에서 멀어질수록 강한 패널티 부여 (가상의 스프링 복원력 역할)
        reward += np.exp(-1.0 * x_pos**2) * 3.0  
        reward += np.exp(-3.0 * x_vel**2) * 2.0  
        reward -= 0.01 * np.sum(np.square(action))
        
        if abs(pitch) > 1.5 or abs(roll) > 1.5 or z_height < 0.01:
            terminated = True
            reward = -20.0 
            
        # [추가] 1000스텝(약 16초) 동안 스쿼트를 버티면 성공(타임아웃 리셋)
        if self.current_step >= 1000:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

def make_env(rank, seed=0):
    def _init():
        env = Leg2WheelEnv(render=False)
        env = Monitor(env, os.path.join(LOG_DIR, str(rank))) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    num_cpu = max(1, os.cpu_count() // 2)
    torch.set_num_threads(1) 
    
    print(f"[{num_cpu} 코어 할당] Phase 2: 스쿼트 주행(동적 높이 변화) 밸런싱 학습 시작.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_cpu, 
        save_path=CHECKPOINT_DIR,
        name_prefix=save_name
    )
    
    # 모델 저장명은 v3, 불러오는 과거 모델은 v2
    model_name = os.path.join(WORK_DIR, save_name)
    model_file = f"{model_name}.zip"
    
    # 전이 학습 (Transfer Learning)
    print("-> 관측 센서 차원 증가(6차원)로 인해 백지 상태에서 신규 학습을 시작합니다.")
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[128, 128])
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    
    try:
        # 난이도가 높으므로 학습량을 500,000 스텝(50만 번)으로 상향
        model.learn(total_timesteps=500000, reset_num_timesteps=False, callback=checkpoint_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨.")
    except KeyboardInterrupt:
        print("\n학습 중지됨.")