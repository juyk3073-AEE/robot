# Gymnasium 환경 래핑 (Environment Wrapping) 첫 학습
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import math

class Leg2DofEnv(gym.Env):
    def __init__(self, render=True):
        super(Leg2DofEnv, self).__init__()
        
        # 1. 입출력 규격(Space) 정의
        # 행동 공간 (Action Space): [-1.0, 1.0] 범위의 2D 벡터 (Hip, Knee 모터 제어 신호)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 상태 공간 (Observation Space): 무한대 범위의 4D 벡터 (Pitch, Pitch_rate, Hip_angle, Knee_angle)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # 2. PyBullet 물리 엔진 초기화
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "leg_2dof_4.urdf")
        self.robotId = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        startPos = [0, 0, 1.0] # 공중에서 시작
        self.robotId = p.loadURDF(self.urdf_path, startPos, useFixedBase=False, physicsClientId=self.client)
        
        # 모터 초기화 (외부 제어 모드)
        p.setJointMotorControl2(self.robotId, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.robotId, 1, p.VELOCITY_CONTROL, force=0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 가상 센서 데이터 추출 (상태 관측)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robotId, physicsClientId=self.client)
        
        hip_state = p.getJointState(self.robotId, 0, physicsClientId=self.client)[0]
        knee_state = p.getJointState(self.robotId, 1, physicsClientId=self.client)[0]
        
        # [Pitch 각도, Pitch 각속도, Hip 각도, Knee 각도]
        obs = np.array([base_euler[1], base_ang_vel[1], hip_state, knee_state], dtype=np.float32)
        return obs

    def step(self, action):
        # 1. 행동(Action) 인가: 정규화된 행동[-1, 1]을 실제 모터 각도로 매핑
        hip_target = action[0] * 1.57   # -1.57 ~ 1.57 rad
        knee_target = action[1] * 2.5   # -2.5 ~ 2.5 rad (음수 위주로 클리핑 필요)
        knee_target = -abs(knee_target) # 무릎은 항상 굽혀지는 방향으로만 제한
        
        p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=hip_target)
        p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=knee_target)
        
        p.stepSimulation(physicsClientId=self.client)
        
        # 2. 상태 관측
        obs = self._get_obs()
        pitch = obs[0]
        
        # 3. 보상 함수 (Reward Function) 및 종료 조건(Done)
        terminated = False
        reward = 1.0 # 살아있으면 매 스텝 기본 보상 1.0
        
        # 로봇이 +- 0.5 라디안(약 28도) 이상 기울어지면 넘어졌다고 판단
        if abs(pitch) > 0.5:
            terminated = True
            reward = -10.0 # 넘어지면 강력한 페널티
            
        # 수직을 잘 유지할수록 추가 보상 (목적 함수)
        reward -= abs(pitch) * 2.0 
        
        return obs, reward, terminated, False, {}

# 환경 테스트용 코드
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    import time

    # 1. 환경 검증
    # 학습 시에는 연산 속도를 극대화하기 위해 GUI 렌더링을 끕니다 (render=False)
    train_env = Leg2DofEnv(render=False)
    check_env(train_env) # 커스텀 환경이 표준 규격에 맞는지 자동 검사

    print("인공지능 제어기(PPO) 학습 시작... (약 1~3분 소요)")
    
    # 2. PPO 모델 정의 및 학습
    # MlpPolicy: 다층 퍼셉트론(신경망)을 제어기로 사용
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
    
    # 10만 번의 시뮬레이션 스텝 동안 넘어지고 일어서기를 반복하며 학습
    model.learn(total_timesteps=100000)
    
    # 학습된 가중치(제어 정책) 저장
    model.save("ppo_leg_balancer")
    print("학습 완료 및 모델 저장됨 (ppo_leg_balancer.zip)")

    # 3. 학습된 모델 시각적 테스트
    print("학습된 제어기 테스트 시작 (렌더링 켬)...")
    test_env = Leg2DofEnv(render=True)
    obs, _ = test_env.reset()

    while True:
        # 신경망이 현재 상태(obs)를 보고 최적의 모터 각도(action)를 예측
        action, _states = model.predict(obs, deterministic=True)
        
        # 예측된 행동을 환경에 인가
        obs, reward, done, _, _ = test_env.step(action)
        
        time.sleep(1./240.) # 실시간 관찰을 위한 프레임 지연
        
        if done:
            obs, _ = test_env.reset()