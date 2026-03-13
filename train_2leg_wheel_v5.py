import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import math
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

WORK_DIR = r"C:\Users\juyk3\project\physical ai"
blueprint = "blueprint_2leg_wheel_v1.urdf"
save_name = "ppo_2leg_wheel_v5"
pre_model = "ppo_2leg_wheel_v4.zip"
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
LOG_DIR = os.path.join(WORK_DIR, "logs")

class Leg2WheelEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2WheelEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 6차원 유지: pitch, pitch_vel, x_vel, z_height, z_vel, cmd_vel
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
            
        self.current_step = 0
        self.time_step = math.pi / 2.0 
        
        # [핵심] 에피소드 시작 시 랜덤 목표 속도 부여 (-3m/s ~ +3m/s)
        self.cmd_vel = random.uniform(-10.0, 10.0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        return np.array([
            base_euler[1], base_ang_vel[1], 
            self.cmd_vel, base_vel[0],  # [핵심] x_pos 자리에 정확히 cmd_vel을 삽입. x_vel은 4번째(인덱스3) 유지.
            base_pos[2], base_vel[2]    # z_height와 z_vel도 예전 인덱스(4, 5) 유지.
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.time_step += 0.02 
        
        # 주행 중에도 높이 변화(외란) 동시 극복 훈련 유지
        sine_wave = math.sin(self.time_step)
        target_height = 0.485 + (sine_wave * 0.225) 
        
        # 150스텝(약 2.5초)마다 AI에게 새로운 주행 속도를 요구하여 감감속/방향전환 훈련
        if self.current_step % 150 == 0:
            self.cmd_vel = random.uniform(-10.0, 10.0)
            
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
        
        # ... (step 내부 물리 시뮬레이션 연산 후) ...
        obs = self._get_obs()
        # obs[2]는 이제 cmd_vel이므로 제외하고, x_vel을 obs[3]으로, z_height를 obs[4]로 정확히 매칭
        pitch, pitch_vel, x_vel, z_height = obs[0], obs[1], obs[3], obs[4]
        base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)[1]
        roll = p.getEulerFromQuaternion(base_orn)[0]
        
        terminated = False
        truncated = False
        
        # [핵심] 속도 추종(Velocity Tracking) 보상 함수
        vel_error = x_vel - self.cmd_vel
        
        reward = 1.0 
        reward += np.exp(-1.0 * pitch**2) * 1.0  
        reward += np.exp(-2.0 * pitch_vel**2) * 2.0  
        # 현재 속도와 목표 속도의 오차가 0에 가까울수록 5.0점의 큰 보상 부여
        reward += np.exp(-2.0 * vel_error**2) * 5.0  
        reward -= 0.01 * np.sum(np.square(action))
        
        if abs(pitch) > 1.5 or abs(roll) > 1.5 or z_height < 0.01:
            terminated = True
            reward = -20.0 
            
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
    
    print(f"[{num_cpu} 코어 할당] Phase 3(v5): 전진 및 후진 주행 속도 추종 학습 시작.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_cpu, 
        save_path=CHECKPOINT_DIR,
        name_prefix=save_name
    )
    
    model_name = os.path.join(WORK_DIR, save_name)
    model_file = f"{model_name}.zip"
    v4_model_path = os.path.join(WORK_DIR, pre_model) # v4 경로 추가
    
    if os.path.exists(model_file):
        print(f"-> 기존 v5 모델({model_file})을 불러와서 이어서 학습합니다.")
        model = PPO.load(model_file, env=vec_env)
    elif os.path.exists(v4_model_path): # v4 연동 로직 추가
        print("-> [전이학습] v4 모델의 밸런싱 뇌를 바탕으로 v5(속도 추종) 학습을 시작합니다.")
        model = PPO.load(v4_model_path, env=vec_env)
    else:
        print("-> 기존 모델을 찾을 수 없어 백지 상태에서 신규 학습을 시작합니다.")
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[128, 128])
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)

    try: 
        model.learn(total_timesteps=500000, reset_num_timesteps=False, callback=checkpoint_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨. 종료합니다.")
    except KeyboardInterrupt:
        print("\n학습 중지됨.")