import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction,
    Shutdown, TimerAction
)
from launch.substitutions import LaunchConfiguration

# 사용법 (ws_limo_humble 폴더에서):
#   ros2 launch localization.py map_name:=맵이름
# 지도 찾는 순서:
#   1) /home/hongsumin/ws_limo_humble/맵이름.yaml          (매핑 후 map_saver_cli로 저장한 곳)
#   2) /home/hongsumin/Raceline-Optimization/maps/맵이름.yaml
# map_name 에 .yaml 경로를 직접 줘도 됨 (예: map_name:=/경로/맵이름.yaml)

# [수정] 이 파일이 있는 폴더 기준 (어느 컴퓨터에 clone 해도 동작)
WS_DIR = os.path.dirname(os.path.abspath(__file__))
def _find_raceline():
    # [추가] Raceline-Optimization 위치: 환경변수 LIMO_RACELINE_DIR -> 이 폴더 옆(../Raceline-Optimization) -> 홈
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get('LIMO_RACELINE_DIR'),
              os.path.join(os.path.dirname(here), 'Raceline-Optimization'),
              os.path.expanduser('~/Raceline-Optimization')):
        if d and os.path.isdir(d):
            return d
    return os.path.join(os.path.dirname(here), 'Raceline-Optimization')


RACELINE_MAPS = os.path.join(_find_raceline(), 'maps')
# 실제 LIMO에서는 export LIMO_REAL=1 을 먼저 하면 use_sim_time 기본값이 false 가 됨 (Gazebo는 그대로 true)
USE_SIM_TIME = 'false' if os.environ.get('LIMO_REAL') == '1' else 'true'
PARAMS_FILE = os.path.join(WS_DIR, 'nav2_params_limo_tuned.yaml')


def is_map_yaml(path):
    # 지도 yaml인지 확인 (같은 이름의 다른 설정 yaml을 잘못 쓰지 않게)
    if not os.path.isfile(path):
        return False

    try:
        with open(path, 'r') as f:
            text = f.read()
    except OSError:
        return False

    return 'image:' in text and 'resolution:' in text


def find_map(map_name):
    if map_name.endswith('.yaml') or '/' in map_name:
        path = map_name if os.path.isabs(map_name) else os.path.join(WS_DIR, map_name)
        candidates = [path]
    else:
        candidates = [
            os.path.join(WS_DIR, f'{map_name}.yaml'),
            os.path.join(RACELINE_MAPS, f'{map_name}.yaml'),
        ]

    for path in candidates:
        if is_map_yaml(path):
            return path, candidates

    return None, candidates


def launch_setup(context):
    map_name = LaunchConfiguration('map_name').perform(context)
    map_path, candidates = find_map(map_name)

    if map_path is None:
        return [
            LogInfo(msg=(
                f'[localization] 지도를 찾을 수 없습니다: {map_name}\n'
                f'  확인한 위치: {", ".join(candidates)}'
            )),
            Shutdown(reason='map not found'),
        ]

    localization = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'nav2_bringup', 'localization_launch.py',
            f'map:={map_path}',
            f'params_file:={PARAMS_FILE}',
            f'use_sim_time:={USE_SIM_TIME}'
        ],
        output='screen'
    )

    initial_pose = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/initialpose',
                    'geometry_msgs/msg/PoseWithCovarianceStamped',
                    "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
                ],
                output='screen'
            )
        ]
    )

    return [
        LogInfo(msg=f'[localization] 지도: {map_path}'),
        localization,
        initial_pose,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='limo_ackermann_map',
            description='지도 이름 (ws_limo_humble 또는 Raceline-Optimization/maps 의 맵이름.yaml)'
        ),
        OpaqueFunction(function=launch_setup),
    ])
