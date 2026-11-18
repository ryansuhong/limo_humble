import os
import sys

# 매핑(slam_toolbox) 실행
#   python3 mapping.py
# 아래 명령과 같음:
#   ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true \
#     slam_params_file:=/home/hongsumin/ws_limo_humble/slam_limo.yaml
# 추가 인자를 주면 그대로 뒤에 붙음

# [수정] 이 파일과 같은 폴더의 slam_limo.yaml
SLAM_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slam_limo.yaml')
# 실제 LIMO에서는 export LIMO_REAL=1 을 먼저 하면 use_sim_time 기본값이 false 가 됨 (Gazebo는 그대로 true)
USE_SIM_TIME = 'false' if os.environ.get('LIMO_REAL') == '1' else 'true'

cmd = [
    'ros2', 'launch', 'slam_toolbox', 'online_async_launch.py',
    f'use_sim_time:={USE_SIM_TIME}',
    f'slam_params_file:={SLAM_PARAMS}',
] + sys.argv[1:]

# 현재 프로세스를 launch로 바꿔서 실행 (Ctrl+C로 바로 종료됨)
os.execvp(cmd[0], cmd)
