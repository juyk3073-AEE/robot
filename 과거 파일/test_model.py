from stable_baselines3 import PPO
import time

# [수정 1] 최신 파일명인 pogo_train_v2에서 환경 클래스(Env)를 가져옵니다.
from pogo_train_v2 import Leg2DofEnv

zip_file = "ppo_leg_balancer_v3.zip"

test_env = Leg2DofEnv(render=True)
obs, _ = test_env.reset()

print(f"{zip_file} 모델을 불러옵니다...")
model = PPO.load(zip_file)

print("AI 제어 시작!")
while True:
    # 학습된 AI가 현재 상태(obs)를 보고 행동(action)을 결정
    action, _states = model.predict(obs, deterministic=True)
    
    # 환경에 행동 적용
    obs, reward, done, _, _ = test_env.step(action)
    
    # [수정 2] 렌더링 속도 동기화
    # 학습 코드(pogo_train_v2)에서 1스텝당 물리연산을 4번(4/240초) 진행하므로,
    # 실제 시각화 지연 시간도 1/60초로 맞춰주어야 현실과 동일한 속도로 보입니다.
    time.sleep(1./60.) 
    
    if done:
        obs, _ = test_env.reset()