import os
import sys
import glob
import shutil
import time

# 매핑한 지도를 ws_limo_humble 에 저장 (매핑이 켜져 있는 상태에서 실행)
#   python3 map_saver.py 맵이름
#   python3 map_saver.py 맵이름 --fmt png     # 이미지 형식 지정 (기본 pgm)
# 아래 명령과 같음:
#   ros2 run nav2_map_server map_saver_cli -f /home/hongsumin/ws_limo_humble/맵이름
# 같은 이름의 지도가 이미 있으면 ws_limo_humble/map_backup 에 복사해 두고 저장
# (복사라서 저장에 실패해도 기존 지도는 그대로 남음)

# [수정] 이 파일이 있는 폴더 기준
WS_DIR = os.path.dirname(os.path.abspath(__file__))

args = [a for a in sys.argv[1:]]

if not args or args[0].startswith('-'):
    print('사용법: python3 map_saver.py 맵이름 [--fmt png]')
    sys.exit(1)

map_name = args[0]

if map_name.endswith('.yaml'):
    map_name = map_name[:-len('.yaml')]

target = os.path.join(WS_DIR, map_name)

existing = [
    p for p in glob.glob(target + '.*')
    if os.path.splitext(p)[1] in ('.yaml', '.pgm', '.png')
]

if existing:
    backup_dir = os.path.join(WS_DIR, 'map_backup')
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    for p in existing:
        base, ext = os.path.splitext(os.path.basename(p))
        moved = os.path.join(backup_dir, f'{base}_{stamp}{ext}')
        shutil.copy2(p, moved)
        print(f'기존 지도 백업: {p} -> {moved}')

cmd = [
    'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
    '-f', target,
] + args[1:]

print(f'지도 저장: {target}.yaml')
sys.stdout.flush()      # exec 전에 안내 문구 출력
os.execvp(cmd[0], cmd)
