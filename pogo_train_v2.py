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

blueprint = "blueprint_leg_2dof_v2.urdf"
class Leg2DofEnv(gym.Env):
    def __init__(self, render=False):
        super(Leg2DofEnv, self).__init__()
        # 액션: [-1, 1] 정규화 유지
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # [고도화 1] 관측 공간 확대 (크기 4 -> 6: Z축 높이, Z축 속도 추가)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, blueprint)
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
        
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(planeId, -1, lateralFriction=2.0, restitution=0.0, physicsClientId=self.client)

        startPos = [0, 0, 0.65] 
        self.robotId = p.loadURDF(self.urdf_path, startPos, useFixedBase=False, physicsClientId=self.client)
        
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
        
        # 배열 6개: Pitch, Pitch속도, Z높이, Z속도, 힙 각도, 무릎 각도
        return np.array([
            base_euler[1], base_ang_vel[1], 
            base_pos[2], base_vel[2], 
            hip_state, knee_state
        ], dtype=np.float32)

    def step(self, action):
        # [고도화 2] 액션 매핑 (Clipping 제거)
        # 신경망의 [-1, 1] 출력을 물리적 관절 범위로 스케일링
        hip_target = action[0] * 0.5  # -0.5 ~ +0.5 rad
        knee_target = (action[1] - 1.0) * 0.75  # 무릎은 뒤로만 접히도록 -1.5 ~ 0.0 rad

        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target, force=30, physicsClientId=self.client)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target, force=30, physicsClientId=self.client)
        
        # [고도화 3] Action Repeat (제어 주기 60Hz 모사)
        # 모터 명령 하달 후, 물리 엔진만 4프레임 전진시킴 (자연스러운 관성 움직임 유도)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_obs()
        pitch = obs[0]
        z_height = obs[2]
        
        base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)[1]
        roll = p.getEulerFromQuaternion(base_orn)[0]
        
        terminated = False
        
        # [고도화 4] 다중 보상 함수 설계
        reward = 1.0                              # 기본 생존 보상
        reward += np.exp(-5.0 * pitch**2)         # Pitch가 0(수직)에 가까울수록 기하급수적 보상 획득
        reward += z_height * 2.0                  # 주저앉지 않고 높은 고도를 유지할수록 가산점
        reward -= 0.1 * np.sum(np.square(action)) # 모터를 과도하게 튕기는 행위(에너지 낭비) 페널티
        
        # 사망 조건
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

if __name__ == "__main__":
    num_cpu = max(1, os.cpu_count() // 2)
    torch.set_num_threads(1) 
    
    print(f"[{num_cpu} 코어 할당] 강화된 보상/관측 기반 병렬 학습 시작.")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_cpu, 
        save_path='./checkpoints/',
        name_prefix='rl_model'
    )
    
    model_name = "ppo_leg_balancer_v3"
    model_file = f"{model_name}.zip"
    
    if os.path.exists(model_file):
        print(f"-> {model_file} 로드하여 추가 학습")
        model = PPO.load(model_file, env=vec_env)
    else:
        print("-> 신규 학습 시작")
        # 네트워크 크기 확장: 은닉층 [128, 128]로 늘려 복잡한 센서 데이터 처리력 강화
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[128, 128])
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    
    try:
        model.learn(total_timesteps=500000, reset_num_timesteps=False, callback=checkpoint_callback)
        model.save(model_name)
        print(f"\n최종 학습 완료! {model_file} 저장됨.")
    except KeyboardInterrupt:
        print("\n종료됨. 체크포인트 확인 요망.")