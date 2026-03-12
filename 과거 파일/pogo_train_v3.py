import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from typing import Callable

blueprint = "blueprint_leg_2dof_v2.urdf"

class Leg2DofEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2DofEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # [수정] 관측 공간을 7차원으로 정확히 명시 (Pitch, P속도, X속도, Z높이, Z속도, 힙, 무릎)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, blueprint)
        self.robotId = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240., numSubSteps=8, numSolverIterations=100, physicsClientId=self.client)
        
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(planeId, -1, lateralFriction=2.0, restitution=0.0, physicsClientId=self.client)

        startPos = [0, 0, 0.65] 
        self.robotId = p.loadURDF(self.urdf_path, startPos, useFixedBase=False, physicsClientId=self.client)
        
        p.resetJointState(self.robotId, 0, targetValue=0, physicsClientId=self.client)
        p.resetJointState(self.robotId, 1, targetValue=0, physicsClientId=self.client)
        
        for j in range(-1, p.getNumJoints(self.robotId, physicsClientId=self.client)):
            p.changeDynamics(self.robotId, j, lateralFriction=2.0, restitution=0.0, linearDamping=0.04, angularDamping=0.04, physicsClientId=self.client)
            
        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=0, force=50, physicsClientId=self.client)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=0, force=50, physicsClientId=self.client)
        
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        hip_state = p.getJointState(self.robotId, 0, physicsClientId=self.client)[0]
        knee_state = p.getJointState(self.robotId, 1, physicsClientId=self.client)[0]
        
        # [수정] 데이터 배열 순서 재정립 (크기 7)
        return np.array([
            base_euler[1],    # [0] Pitch
            base_ang_vel[1],  # [1] Pitch 각속도
            base_vel[0],      # [2] X축 속도 (앞뒤 밀림)
            base_pos[2],      # [3] Z축 높이
            base_vel[2],      # [4] Z축 속도
            hip_state,        # [5] 힙 상태
            knee_state        # [6] 무릎 상태
        ], dtype=np.float32)

    def step(self, action):
        hip_target = action[0] * 0.5  
        knee_target = (action[1] - 1.0) * 0.75  

        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target, force=30, physicsClientId=self.client)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target, force=30, physicsClientId=self.client)
        
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_obs()
        
        # [수정] 위 배열과 인덱스 동기화 완료
        pitch = obs[0]
        x_vel = obs[2]
        z_height = obs[3]
        
        base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)[1]
        roll = p.getEulerFromQuaternion(base_orn)[0]
        
        terminated = False
        reward = 1.0 
        
        reward += np.exp(-5.0 * pitch**2) * 3.0 
        reward += np.exp(-2.0 * x_vel**2) 
        reward -= 0.01 * np.sum(np.square(action))
        
        # 정상적인 Z축 높이를 참조하므로 시작하자마자 죽지 않음
        if abs(pitch) > 0.5 or abs(roll) > 0.5 or z_height < 0.25:
            terminated = True
            reward = -20.0 
            
        return obs, reward, terminated, False, {}

def make_env(rank, seed=0):
    def _init():
        env = Leg2DofEnv(render=False)
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    num_cpu = max(1, os.cpu_count() // 2)
    torch.set_num_threads(1) 
    
    print(f"[{num_cpu} 코어 할당] 버그 픽스 완료. 정상 학습을 시작합니다.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    eval_env = DummyVecEnv([make_env(99)])
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./best_model/',
        log_path='./logs/', 
        eval_freq=100000 // num_cpu,
        deterministic=True, 
        render=False
    )
    
    model_name = "ppo_leg_balancer_v4" 
    model_file = f"{model_name}.zip"
    
    if os.path.exists(model_file):
        print(f"-> {model_file} 로드하여 추가 학습")
        model = PPO.load(model_file, env=vec_env)
    else:
        print("-> 신규 학습 시작")
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[128, 128])
        model = PPO("MlpPolicy", vec_env, verbose=1, 
                    learning_rate=linear_schedule(0.0003), 
                    policy_kwargs=policy_kwargs)
    
    try:
        model.learn(total_timesteps=500000, reset_num_timesteps=False, callback=eval_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨.")
    except KeyboardInterrupt:
        print(f"\n종료됨. 최고 모델은 '.{model_file}'에 저장되어 있습니다.")