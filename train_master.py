import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

class Leg2DofEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2DofEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "leg_2dof_4.urdf")
        self.robotId = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSubSteps=8,
            numSolverIterations=100,
            physicsClientId=self.client
        )
        
        # [수정 1] leg_2dof_4.py와 동일하게 친숙한 plane.urdf 사용
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(planeId, -1, lateralFriction=2.0, restitution=0.0, physicsClientId=self.client)

        # 다리 전체 길이가 0.6m이므로, 지면에 살짝 닿을 듯한 0.65m에서 스폰
        startPos = [0, 0, 1] 
        self.robotId = p.loadURDF(self.urdf_path, startPos, useFixedBase=False, physicsClientId=self.client)
        
        # 관절 위치 강제 0점 정렬
        p.resetJointState(self.robotId, 0, targetValue=0, physicsClientId=self.client)
        p.resetJointState(self.robotId, 1, targetValue=0, physicsClientId=self.client)
        
        for j in range(-1, p.getNumJoints(self.robotId, physicsClientId=self.client)):
            p.changeDynamics(
                self.robotId, j, 
                lateralFriction=2.0,
                restitution=0.0,
                linearDamping=0.04,
                angularDamping=0.04,
                physicsClientId=self.client
            )
            
        # [수정 2] 핵심 버그 픽스: 대기 시간 동안 모터를 켜서 주저앉지 않고 꼿꼿이 버티게 만듦
        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=0, force=50, physicsClientId=self.client)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=0, force=50, physicsClientId=self.client)
        
        # 50 프레임만 대기하여 착지 충격만 흡수
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            
        return self._get_obs(), {}

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        hip_state = p.getJointState(self.robotId, 0, physicsClientId=self.client)[0]
        knee_state = p.getJointState(self.robotId, 1, physicsClientId=self.client)[0]
        
        return np.array([base_euler[1], base_ang_vel[1], hip_state, knee_state], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -0.3, 0.3)
        
        hip_target = action[0] * 1.57
        knee_target = -abs(action[1] * 2.5)
        
        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target, force=20, physicsClientId=self.client)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target, force=20, physicsClientId=self.client)
        
        p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_obs()
        pitch = obs[0]
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        roll = p.getEulerFromQuaternion(base_orn)[0]
        z_height = base_pos[2]
        
        terminated = False
        reward = 1.0 
        
        # 사망 조건
        if abs(pitch) > 0.5 or abs(roll) > 0.5 or z_height < 0.25:
            terminated = True
            reward = -20.0 
            
        reward -= abs(pitch) * 2.0 
        return obs, reward, terminated, False, {}

def make_env(rank, seed=0):
    def _init():
        env = Leg2DofEnv(render=False)
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    num_cpu = max(1, os.cpu_count() // 2)
    torch.set_num_threads(1) 
    
    print(f"[{num_cpu} 코어 할당] 안정화된 병렬 학습 시작.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_cpu, 
        save_path='./checkpoints/',
        name_prefix='rl_model'
    )
    
    model_name = "ppo_leg_balancer_v5"
    model_file = f"{model_name}.zip"
    
    if os.path.exists(model_file):
        print(f"-> {model_file} 로드하여 추가 학습")
        model = PPO.load(model_file, env=vec_env)
    else:
        print("-> 신규 학습 시작")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003)
    
    try:
        model.learn(total_timesteps=500, reset_num_timesteps=False, callback=checkpoint_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨.")
    except KeyboardInterrupt:
        print("\n종료됨. 체크포인트 확인 요망.")