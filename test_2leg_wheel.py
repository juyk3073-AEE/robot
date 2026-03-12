import os
import time
import pybullet as p
from stable_baselines3 import PPO

# 학습 코드에서 만든 환경 클래스를 그대로 가져옵니다.
from train_2leg_wheel_v1 import Leg2WheelEnv

# [설정] 경로 및 불러올 모델 이름
WORK_DIR = r"C:\Users\juyk3\project\physical ai"
save_name = "ppo_2leg_wheel_v2"
model_path = os.path.join(WORK_DIR, save_name)

# 1. 렌더링이 켜진(GUI) 테스트 환경 생성
test_env = Leg2WheelEnv(render=True)
obs, _ = test_env.reset()

print(f"{model_path}.zip 모델을 불러옵니다...")
model = PPO.load(model_path)

# 2. GUI 슬라이더 생성
# AI의 결정을 무시하고 수동으로 높이를 제어할지 결정하는 스위치
mode_slider = p.addUserDebugParameter("Manual Height Override (0=AI, 1=Manual)", 0, 1, 1)
# 사용자가 지정할 목표 높이 (0.26m ~ 0.71m)
height_slider = p.addUserDebugParameter("Target Body Height (m)", 0.26, 0.71, 0.6)

print("\nAI 제어 시작! 슬라이더를 움직여 로봇의 높이를 강제로 낮추거나 높여보세요.")

while True:
    # 3. AI의 현재 상태에 따른 행동 예측
    # action[0] = 높이 제어, action[1] = 바퀴 속도 제어
    action, _states = model.predict(obs, deterministic=True)
    
    # 4. 사용자 입력값 읽기
    manual_mode = p.readUserDebugParameter(mode_slider)
    manual_height = p.readUserDebugParameter(height_slider)
    
    # [핵심] 수동 제어 모드(0.5 이상)일 경우 AI의 높이 명령을 해킹하여 덮어씌움
    if manual_mode >= 0.5:
        # 학습 환경의 수식 (target_height = 0.485 + action * 0.225)을 역산하여 action[0] 값을 변조
        action[0] = (manual_height - 0.485) / 0.225
        
    # 5. 변조된 액션을 환경에 적용
    obs, reward, done, _, _ = test_env.step(action)
    
    # 학습 코드의 Action Repeat(4프레임)에 맞춘 60Hz 현실 속도 동기화
    time.sleep(1./60.) 
    
    if done:
        print("로봇이 균형을 잃고 쓰러졌습니다. 에피소드를 리셋합니다.")
        obs, _ = test_env.reset()