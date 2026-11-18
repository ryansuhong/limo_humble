import os
import sys

# 키보드 조종(teleop) 실행
#   python3 teleop_keyboard.py
# 아래 명령과 같음:
#   ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
# 추가 인자를 주면 그대로 뒤에 붙음 (예: python3 teleop_keyboard.py -p speed:=0.3)

cmd = [
    'ros2', 'run', 'teleop_twist_keyboard', 'teleop_twist_keyboard',
    '--ros-args', '-r', 'cmd_vel:=/cmd_vel',
] + sys.argv[1:]

# 현재 프로세스를 teleop으로 바꿔서 실행 (키보드 입력이 그대로 전달됨)
os.execvp(cmd[0], cmd)
