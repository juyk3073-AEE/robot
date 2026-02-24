from stable_baselines3 import PPO
import time

# 1. 의존성 업데이트: 패치된 통합 파이프라인에서 환경 로드
from train_master import Leg2DofEnv

test_env = Leg2DofEnv(render=True)
obs, _ = test_env.reset()

# 2. 파일 I/O 업데이트: V5 모델 로드
print("ppo_leg_balancer_v5.zip 모델을 불러옵니다...")
model = PPO.load("ppo_leg_balancer_v5")

print("AI 제어 시작!")
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    
    time.sleep(1./240.) 
    
    if done:
        obs, _ = test_env.reset()