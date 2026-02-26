from stable_baselines3 import PPO
import time
from pogo_train_v1 import Leg2DofEnv

zip_file = "ppo_leg_balancer_v3.zip"

test_env = Leg2DofEnv(render=True)
obs, _ = test_env.reset()

# 2. 파일 I/O 업데이트: V5 모델 로드
print(f"{zip_file} 모델을 불러옵니다...")
model = PPO.load(zip_file)

print("AI 제어 시작!")
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    
    time.sleep(1./240.) 
    
    if done:
        obs, _ = test_env.reset()