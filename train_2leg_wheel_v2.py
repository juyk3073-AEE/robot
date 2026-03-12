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

# [설정] 경로, 모델 및 저장명 지정 (버전 v2로 변경)
WORK_DIR = r"C:\Users\juyk3\project\physical ai"
blueprint = "blueprint_2leg_wheel_v1.urdf"
save_name = "ppo_2leg_wheel_v2"

CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
LOG_DIR = os.path.join(WORK_DIR, "logs")

class Leg2WheelEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2WheelEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
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

        self.robotId = p.loadURDF(self.urdf_path, [0, 0, 0.7], useFixedBase=False, physicsClientId=self.client)
        
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robotId, physicsClientId=self.client)):
            name = p.getJointInfo(self.robotId, i, physicsClientId=self.client)[1].decode('utf-8')
            self.joint_indices[name] = i
            p.changeDynamics(self.robotId, i, lateralFriction=1.5, physicsClientId=self.client)
            
        # [핵심 수정] 낙하하는 50프레임 동안 다리가 무너지지 않도록 0.55m 높이 자세로 모터 락(Lock)
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
        
        # 다리에 힘이 들어간 상태로 바닥에 안착할 때까지 대기
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
        # [수정됨] 커리큘럼 1단계: 높이(action[0]) 제어를 무시하고 0.55m로 영구 고정
        target_height = 0.55 
        # (참고) 2단계 학습 시 위 줄을 지우고 아래 줄의 주석을 해제하면 됩니다.
        # target_height = 0.485 + action[0] * 0.225 
        
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
        reward += np.exp(-5.0 * pitch**2) * 3.0 
        reward += np.exp(-2.0 * x_vel**2) 
        reward -= 0.01 * np.sum(np.square(action))
        
        if abs(pitch) > 0.6 or abs(roll) > 0.5 or z_height < 0.15:
            terminated = True
            reward = -20.0 
            
        return obs, reward, terminated, False, {}

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
    
    print(f"[{num_cpu} 코어 할당] Phase 1: 바퀴 밸런싱 집중 학습 시작.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_cpu, 
        save_path=CHECKPOINT_DIR,
        name_prefix=save_name
    )
    
    model_name = os.path.join(WORK_DIR, save_name)
    model_file = f"{model_name}.zip"
    
    if os.path.exists(model_file):
        print(f"-> {model_file} 로드하여 추가 학습")
        model = PPO.load(model_file, env=vec_env)
    else:
        print("-> 신규 학습 시작")
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[128, 128])
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    
    try:
        model.learn(total_timesteps=50000, reset_num_timesteps=False, callback=checkpoint_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨.")
    except KeyboardInterrupt:
        print("\n학습 중지됨.")