import os
import math
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from tf2_ros import Buffer, TransformListener, TransformException


# [추가] /avoid_status (std_msgs/Float32MultiArray) 내용 순서
# pure_pursuit.py 의 STATUS_* 와 반드시 같아야 함
STATUS_OK = 0               # 1: 센서/위치 정상, 0: 대기 중
STATUS_AVOIDING = 1         # 1: 회피(LOCAL) 모드
STATUS_PASSED = 2           # 1: 경로 기준으로 장애물을 이미 지남
STATUS_LIMIT = 3            # LOCAL 속도 상한 (주변 여유 기준)
STATUS_LIMIT_PASSED = 4     # LOCAL 속도 상한 (장애물을 지난 상태 기준)
STATUS_GLOBAL_LIMIT = 5     # GLOBAL 속도 상한 (장애물 접근 감속, 위치 불량 감속)
STATUS_RECOVERY = 6         # 1: 비상정지 복구 중
STATUS_RECOVERY_V = 7
STATUS_RECOVERY_W = 8
STATUS_ROTATION_CLEAR = 9   # 1: 제자리 회전해도 안 닿음
STATUS_BLOCKED = 10         # 경로 위 장애물까지 거리 (로그용)
STATUS_LOC_ERROR = 11       # 스캔-지도 오차 (로그용)
STATUS_IN_GAP = 12          # 1: 틈 안에 들어와 정렬됨 -> 틈 통과 속도 허용
STATUS_SIZE = 13


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class ObstacleDWA(Node):

    def __init__(self):
        super().__init__('obstacle_dwa')

        self.global_path = None
        self.scan = None
        self.odom_time = None
        self.odom_pose = None
        self.scan_time = None

        # TF: map 좌표계에서 base_link 위치 조회
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

        self.map_frame = 'map'
        self.robot_frame = 'base_link'
        self.odom_frame = 'odom'

        # 실제 로봇 치수에 맞춰 조정 필요
        self.front_length = 0.20
        self.rear_length = 0.20
        self.half_width = 0.15
        self.safety_margin = 0.10          # 이보다 가까우면 추가 감점

        # [수정] 탈락 기준
        # - 옆/뒤: 차체에서 hard_margin 이내로 "지금보다 더" 가까워지면 탈락
        #   (이미 붙어 있으면 그 간격 유지는 허용 -> 박스 옆에서 영구 정지 방지)
        # - 앞: 차체 앞면 ~ front_margin 사이(차체 폭 안)에 점이 들어오면 탈락
        #   pure_pursuit.py의 emergency_gap(0.13)보다 커야
        #   DWA 경로를 따라가다 비상정지에 걸리지 않음
        # - [수정] 시간(1.5초) -> 거리 기준: hard_dist 안에서 걸리면 탈락,
        #   그 뒤에 걸리는 경로는 그 지점까지만 쓰고 감점
        #   (시간 기준이면 0.8m/s에서 1.2m 앞까지 탈락 -> 경로가 전부 사라졌음)
        self.hard_margin = 0.05
        self.front_margin = 0.15
        self.hard_dist = 0.25
        self.cut_penalty = 1.0

        # [수정] 박스에 바짝 붙어 지나가지 않도록 risk 범위/가중치 증가
        self.risk_range = 0.6              # 이보다 멀면 risk 감점 없음
        self.risk_weight = 1.0             # [수정] 2.0 -> 1.0 (틈 가운데로 가는 건 격자 경로가 담당)

        # [수정] LiDAR 위치는 TF(base_link -> scan frame)에서 읽음
        # LIMO: laser_link가 base_link보다 약 0.10m 앞
        # (기존: 0으로 가정 -> 모든 점이 실제보다 10cm 뒤에 찍힘)
        self.lidar_x = 0.0
        self.lidar_y = 0.0
        self.lidar_yaw = 0.0
        self.lidar_from_tf = False

        # Global Path 진행 방향 (pure_pursuit.py와 반드시 동일하게)
        self.path_direction = -1   # auto_direction=True면 시작 시 자동으로 덮어씀
        self.auto_direction = True
        self.direction_fixed = False

        # DWA 설정
        # [수정] 예측 속도: 0.15 고정 -> /odom 실제 속도 (min_speed ~ max_speed)
        # pure_pursuit.py LOCAL 속도가 0.15 ~ 0.8 가변이 되어 맞춤
        # [수정] 0.25 -> 0.5: 좁은 틈 통과 최저 속도 0.5 (pure_pursuit.py local_speed와 같게)
        self.min_speed = 0.48             # [수정] 0.5 -> 0.35 -> 0.42 -> 0.48 (0.5는 좁은 입구에서 비상정지 반복, run43/44)
        self.max_speed = 0.8               # pure_pursuit.py local_max_speed와 같게
        self.speed = self.min_speed
        self.odom_speed = 0.0
        # [수정] 4.0 -> 2.5: 회피 판단은 격자 경로 계획이 하므로
        # DWA는 짧게 보고 그 경로를 촘촘히 따라가는 게 좁은 틈 통과에 유리
        self.sim_time = 2.5
        self.max_horizon = 1.0             # [추가] 예측 거리 상한 (m)
        self.dt = 0.1
        self.target_distance = 1.5

        # [추가] 궤적이 제자리에서 빙글빙글 도는 원이 되지 않도록
        # 누적 회전각이 이 값을 넘으면 그 뒤로는 직진으로 예측 (90도)
        self.max_rollout_turn = math.pi / 2.0

        # [추가] 꺾었다가 펴는(S자) 후보: |omega| 이 값 이상인 원호에 대해 예측 길이의 s_curve_switch 지점에서
        # 직진 / 반대 회전으로 바꾼 궤적도 같이 평가
        self.s_curve_min_omega = 0.3
        self.s_curve_switch = 0.5

        self.omega_samples = [
            i * 0.1 for i in range(-15, 16)    # [수정] ±1.0 -> ±1.5
        ]

        # [추가] 현재 실제 각속도(/odom) 주변 후보만 우선 사용
        # (기존: 매 주기 -1~+1 전체에서 골라서 +0.8 -> -0.8로 뒤집힘)
        self.odom_omega = 0.0
        self.omega_window = 1.2            # [수정] 0.6 -> 1.2 (조향 변화 폭 확대)

        self.last_plan_ms = 0.0

        # [수정] 회피 방향 고정(원호 기준) 제거 -> 격자 경로 계획으로 대체
        # 기존: 장애물이 1.2m에 들어오는 순간 고른 원호 방향으로 고정
        #  -> 0.6m 예측으로는 어느 쪽이 뚫렸는지 모르고 골라서
        #     막힌 쪽(장애물 정중앙)으로 고정된 채 충돌
        # 변경: 로봇 주변 격자(스캔 장애물을 차체 폭만큼 부풀림)에서
        #  global path 앞 지점까지 최단 경로를 찾고, DWA는 그 경로를 따라감
        #  -> 실제로 지나갈 수 있는 틈으로만 경로가 생김
        self.grid_res = 0.05               # [수정] 0.08 -> 0.05 격자 크기 (m). 0.08이면 0.55m 틈이 매 주기 열렸다 닫혔다 함
        self.grid_half = 2.4               # 로봇 중심 ±2.4m
        self.plan_goal_dist = 2.2          # global path 이만큼 앞 지점을 목표로
        self.inflate = 0.20                # 장애물 부풀림 (차체 반폭 0.15 + 여유)
        # [수정] 0.5 -> 0.35: 0.5면 장애물 2번 앞에서 오른쪽 벽을 피해 격자 경로가 먼저 왼쪽으로 불룩했다가
        # 틈 쪽(오른쪽)으로 꺾여서 DWA가 직진만 고르다 틈 입구에서 급회전 -> 왼쪽 앞바퀴가 장애물 모서리에 걸림 (run53)
        # (좁은 틈에서는 기울기가 커져서 가운데로 가려는 힘은 오히려 강해짐)
        self.clear_pref = 0.35             # 장애물에서 이 거리 안이면 비용 증가
        self.clear_weight = 3.0
        self.prev_weight = 0.3             # 이전 경로에서 벗어나면 비용 증가 (좌우 뒤집힘 방지)
        # global path에서 멀어지면 비용 증가 -> 장애물을 지나면 바로 경로로 복귀
        # (없으면 격자 특성상 '옆으로 붙은 채 직진 후 끝에서 꺾는' 경로가 나와서
        #  이전 경로 선호와 겹쳐 복귀가 계속 미뤄졌음)
        self.path_pull = 1.0
        self.prev_plan = None
        # [수정] 1.0 -> 1.5: 틈 입구 쪽을 더 멀리 보고 회피 시작부터 미리 꺾음 (run53 재생: 첫 선택 omega 0 -> -0.2~-0.3)
        self.plan_target_dist = 1.5        # DWA 목표점: 계획 경로를 따라 이만큼 앞

        # 궤적이 기준 경로(격자 경로)에서 떨어진 평균 거리 감점
        self.path_weight = 2.0

        # =====================================
        # [이동] 장애물 감지 / 회피 판단 / 비상정지 (pure_pursuit.py 에서 옮김)
        # 결과는 /avoid_status 로 pure_pursuit.py 에 전달
        # =====================================

        self.global_speed = 1.2            # pure_pursuit.py global_speed와 같게 (접근 감속 계산용) [수정] 1.0 -> 1.2
        # [수정] 2.5 -> 3.0: 멀리서부터 천천히 감속 (라이다 점은 3.0m까지 사용)
        self.detect_dist = 3.0            # 장애물 감지 거리 (속도 1.0에서 미리 감속)
        # [수정] 1.2 -> 1.6: 감속하는 동안 회피 경로(사잇길 쪽)를 미리 따라가며 조금씩 꺾음
        # (기존: 1.2m까지 global path(장애물 정면)로 감속만 하다가 1.2m에서 급하게 꺾음)
        self.avoid_start_dist = 1.6       # 경로 위 장애물이 이 거리 안이면 회피 모드
        # 접근 감속(1.0 -> approach_speed)이 끝나는 거리. 회피 모드가 먼저 시작돼도 이 거리까지는 서서히 감속
        self.approach_end_dist = 1.2
        # slow_dist부터 avoid_start_dist까지 global_speed -> approach_speed로 감속
        self.slow_dist = 2.5
        # [수정] 장애물 접근 감속: 회피 모드(1.2m) 직전에 0.8 -> 0.3으로 급감속하던 것을
        # approach_start_dist(3.0m)부터 1.2m까지 1.0 -> approach_speed로 길게 나눠 천천히 감속
        # (slow_dist 는 '지남'/'복귀' 판단에도 쓰여서 감속 시작 거리만 따로 분리)
        # (시뮬레이션 비교: 3.0m는 한 바퀴 +3.7초, 2.0m는 감속이 급해져 충돌 -> 2.5m)
        self.approach_start_dist = 2.5
        # [수정] 0.35 -> 0.7: 장애물 앞 접근 감속은 최저 0.7까지만 (평소 1.0)
        # (0.35면 감속은 부드럽지만 장애물 앞을 오래 느리게 달려 한 바퀴 약 +2.5초)
        self.approach_speed = 0.6             # [수정] 0.7 -> 0.6 (요청)
        self.path_block_margin = 0.30     # 경로에서 이 거리 안의 점만 '경로 위 장애물'
        self.passed_skip = 0.2            # 장애물 지남 판정: 경로상 로봇보다 이만큼 앞부터 확인

        # 지도(/map)에 이미 있는 벽은 장애물로 보지 않음
        # 0.2: 장애물 1번 주변은 지도와 실제 벽이 최대 0.2m 어긋남
        # (트랙 장애물은 벽에서 0.36m 이상 떨어져 있음)
        self.map_wall_radius = 0.2
        self.wall_mask = None
        self.map_info = None
        self.map_dist = None

        self.avoiding = False
        self.obstacle_detected = False
        self.blocked_distance = float('inf')
        self.last_block_time = -float('inf')
        self.avoid_hold_time = 0.4        # 장애물 지난 뒤 GLOBAL 복귀까지 대기 [수정] 0.8 -> 0.4 (run63: 지난 뒤 0.8~1.3초 중 대부분이 대기)
        self.return_distance = 0.4        # 경로에서 이 거리 안이면 GLOBAL 복귀
        self.return_clearance = 0.45      # 복귀 시 앞쪽 새 장애물 최소 거리

        # LOCAL 속도 상한 (주변 여유 기준) - pure_pursuit.py local_speed/local_max_speed와 같게
        # [수정] 0.25 -> 0.5: 여유 기반 속도 상한의 바닥값 (틈 통과 0.5 ~ 0.7)
        # (시뮬레이션: 한 바퀴 61.5 -> 57.9초, 충돌 0. 문제가 있으면 0.25로 되돌리기)
        self.local_speed = 0.48           # [수정] 0.5 -> 0.35 -> 0.42 -> 0.48 (틈 통과 0.42 ~ 0.6, run64: 바퀴당 약 6초를 최저 0.35로 주행)
        # [추가] 장애물을 지나기 전(틈 통과) 속도 상한. 지난 뒤(ahead_only)는 local_max_speed 그대로
        # (틈 통과 0.5 ~ 0.6, 요청)
        self.gap_max_speed = 0.6
        self.local_max_speed = 0.8
        self.local_front_gain = 2.0        # 앞쪽 여유 1m당 속도 증가량
        self.local_side_gain = 3.0         # 차체 주변 여유 1m당 속도 증가량
        self.passed_front_min = 0.15      # 지난 상태: 앞쪽 여유가 이 값이면 최저 속도 (기본 0.25)
        self.passed_front_gain = 3.0      # 지난 상태: 앞쪽 여유 1m당 속도 증가 (기본 2.0)

        # 비상정지 영역 = 차체 폭 안 + 앞면에서 emergency_gap 이내
        # LiDAR(base_link 앞 0.10m)의 range_min이 0.20m라 차체 앞 약 0.10m 안은 안 보임
        # -> 그보다 바깥에서 걸려야 동작. front_margin(0.15)보다 작게
        self.emergency_gap = 0.13
        self.rotate_clearance = 0.24      # 제자리 회전 시 차체 모서리 반경(약 0.21m) + 여유

        # 비상정지 복구: 후진하면서 회피 경로(없으면 열린) 쪽으로 회전, 뒤가 막혔으면 제자리 회전
        self.backup_speed = 0.08          # 후진 속도 (m/s)
        self.backup_turn = 1.6            # 후진 중 회전 명령 (회전 보상 포함 값)
        self.rotate_speed = 2.0           # 뒤가 막혔을 때 제자리 회전 명령
        self.backup_duration = 1.0        # 한 번 걸리면 최대 이만큼 복구 동작 (s)
        self.backup_until = 0.0
        # 주의: LiDAR 시야가 ±120°라 바로 뒤는 거의 안 보임 -> 후진은 짧게 유지
        self.rear_check_dist = 0.20       # 뒤끝에서 이 거리 안에 물체 있으면 후진 금지
        self.escape_check_dist = 0.6      # 회전 방향 판단에 쓰는 전방 범위
        self.escape_path_dist = 0.5       # 회피 경로 방향 판단 거리
        # 복구 중 회피 경로 방향과 이 각도 안이면 조기 종료 (최소 recovery_min_time 유지)
        self.recovery_align_tol = math.radians(15)
        self.recovery_min_time = 0.3
        self.recovery_start = -float('inf')
        self.recovery_sign = 1            # +1: 왼쪽으로 회전, -1: 오른쪽
        self.recovery_end = -float('inf')
        self.recovery_keep_time = 3.0     # 회피 경로가 없을 때, 이 시간 안에 다시 걸리면 같은 방향 유지
        self.recovery_cmd = (0.0, 0.0)
        # 회피 중 DWA 경로가 계속 없으면 복구 동작
        self.local_stuck_time = 1.0
        self.no_path_since = None
        self.last_path_points = None      # 마지막으로 발행한 /local_path 점들

        # 위치 추정 감시: 스캔 점과 지도 벽 거리의 중앙값이 크면 AMCL 위치가 틀어진 것
        self.loc_error = 0.0
        self.loc_bad_dist = 0.20          # 중앙값이 이보다 크면 위치 불량
        self.loc_bad_time = 1.0           # 이 시간 이상 지속되면 감속
        self.loc_bad_since = None
        self.loc_bad = False
        self.loc_bad_speed = 0.3

        # [추가] 틈 안에 들어온 뒤 가속
        # 양옆이 가깝고(틈 안) 차체 앞길이 비어 있고 회피 경로와 방향이 맞으면
        # 옆 여유 기준 감속(최저 0.25) 대신 gap_speed까지 허용. 틈 입구 접근 감속은 그대로
        self.gap_speed = 0.5
        self.gap_side_dist = 0.15         # 차체 옆면에서 이 거리 안에 점이 있으면 틈 안 (장애물 옆을 지나는 중)
        self.gap_front_clear = 0.5        # 차체 앞면에서 이 거리까지 차체 폭 안에 점이 없어야 함
        self.gap_heading_tol = 0.44       # 회피 경로 방향 오차 (rad, 약 25도)
        self.gap_omega_max = 0.3          # 실제 회전 속도 (rad/s) 이하일 때만 (커브 중인 틈에서 가속하면 밀려나 끼임)
        self.in_gap = False

        # 데이터 유효 시간
        self.odom_timeout = 0.5
        self.scan_timeout = 0.5

        self.create_subscription(
            Path,
            '/global_path',
            self.on_global,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            qos_profile_sensor_data
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        # [추가] 격자 경로 계획에 지도 벽도 사용
        # 스캔만 쓰면 장애물에 가려 안 보이는 벽을 빈 공간으로 봐서
        # 장애물 2번 아래쪽 막다른 공간(틈 0.32m)으로 경로를 냈음
        # DWA 충돌 판정은 스캔만 사용 (위치가 틀어졌을 때 지도 벽으로 잘못 막히지 않게)
        self.map_points = None
        self.map_plan_range = 3.0          # 로봇 주변 이 거리 안의 지도 벽만 사용

        # [추가] 라이다 사각지대 기억: 라이다 시야(±120°) 밖인 뒷바퀴 옆/뒤는 스캔에 안 보여서
        # 틈을 빠져나오며 경로로 꺾을 때 뒷바퀴가 장애물 모서리에 걸렸음 (run27)
        # -> 최근 memory_time 동안 본 점 중 지금 시야 밖으로 나간 점만 기억해서 충돌 판정에 포함
        #    (시야 안의 점은 매번 새 스캔으로 갱신 -> 사라진 장애물은 바로 지워짐)
        self.memory_time = 2.0
        self.memory_range = 1.0            # 로봇 중심에서 이 거리 안의 점만 기억
        self.memory_voxel = 0.04           # 같은 칸(4cm)의 점은 최신 것 하나만
        self.memory = np.zeros((0, 3))     # (x, y, 본 시각)

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.on_map,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE
            )
        )

        self.pub = self.create_publisher(
            Path,
            '/local_path',
            10
        )

        # [추가] 회피 상태 -> pure_pursuit.py
        self.status_pub = self.create_publisher(
            Float32MultiArray,
            '/avoid_status',
            10
        )

        self.create_timer(0.1, self.plan)

        self.get_logger().info(
            'DWA 시작: /odom + TF(map->odom)'
        )

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_global(self, msg):
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warning(
                f'Global Path 좌표계 불일치: '
                f'{msg.header.frame_id}',
                throttle_duration_sec=2.0
            )
            self.global_path = None
            return

        self.global_path = msg

    def on_odom(self, msg):
        # /odom 메시지의 위치를 직접 사용 (TF보다 항상 최신)
        p = msg.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_of(msg.pose.pose.orientation))
        self.odom_omega = msg.twist.twist.angular.z
        self.odom_speed = msg.twist.twist.linear.x

        if msg.header.frame_id:
            self.odom_frame = msg.header.frame_id

        self.odom_time = self.now()

    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = self.now()

    def on_map(self, msg):
        # 점유 셀 중심 좌표 목록
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            info.height, info.width
        )
        iy, ix = np.nonzero(grid >= 50)

        # [이동] 지도 벽 판정용: 벽 셀을 map_wall_radius 만큼 넓힌 마스크
        # (pure_pursuit.py 의 is_map_wall 과 같은 정사각형 범위)
        r = max(1, int(self.map_wall_radius / info.resolution))
        self.wall_mask = binary_dilation(
            grid >= 50, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        )
        # [이동] 위치 감시용: 각 셀에서 가장 가까운 지도 벽까지 거리
        self.map_dist = distance_transform_edt(grid < 50) * info.resolution
        self.map_info = info

        first = self.map_points is None
        self.map_points = np.column_stack((
            info.origin.position.x + (ix + 0.5) * info.resolution,
            info.origin.position.y + (iy + 0.5) * info.resolution
        ))

        if first:
            self.get_logger().info(
                f'Map 수신: 벽 셀 {len(self.map_points)}개 (격자 경로 계획에 사용)'
            )

    def fix_direction(self, x, y, yaw):
        # 시작할 때 한 번만: 로봇이 바라보는 방향과
        # 가장 가까운 경로 구간의 방향을 비교해서 진행 방향 결정
        if self.direction_fixed or not self.auto_direction:
            return

        poses = self.global_path.poses
        n = len(poses)

        i = min(
            range(n),
            key=lambda k: math.hypot(
                poses[k].pose.position.x - x,
                poses[k].pose.position.y - y
            )
        )

        a = poses[i].pose.position
        b = poses[(i + 1) % n].pose.position

        dot = (
            math.cos(yaw) * (b.x - a.x)
            + math.sin(yaw) * (b.y - a.y)
        )

        self.path_direction = 1 if dot >= 0 else -1
        self.direction_fixed = True

        self.get_logger().info(
            f'경로 진행 방향 자동 결정: path_direction={self.path_direction}'
        )

    def robot_pose(self):
        # map 위치 = (map->odom TF) x (/odom 메시지 위치)
        # map->odom(AMCL)은 천천히 갱신되므로 나이 검사 없이 최신값 사용.
        # /odom의 신선도는 odom_timeout으로 이미 검사함.
        if self.odom_pose is None:
            return None

        try:
            m2o = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                Time()
            )

        except TransformException as e:
            self.get_logger().warning(
                f'map->{self.odom_frame} TF 조회 실패: {e}',
                throttle_duration_sec=1.0
            )
            return None

        mt = m2o.transform.translation
        myaw = yaw_of(m2o.transform.rotation)

        ox, oy, oyaw = self.odom_pose

        c = math.cos(myaw)
        s = math.sin(myaw)

        x = mt.x + c * ox - s * oy
        y = mt.y + s * ox + c * oy

        yaw = myaw + oyaw
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return x, y, yaw

    def update_lidar_offset(self, frame):
        if self.lidar_from_tf or not frame:
            return

        try:
            t = self.tf_buffer.lookup_transform(
                self.robot_frame,
                frame,
                Time()
            )

        except TransformException as e:
            self.get_logger().warning(
                f'{self.robot_frame}->{frame} TF 조회 실패, '
                f'LiDAR 위치 (0,0)으로 가정: {e}',
                throttle_duration_sec=2.0
            )
            return

        self.lidar_x = t.transform.translation.x
        self.lidar_y = t.transform.translation.y
        self.lidar_yaw = yaw_of(t.transform.rotation)
        self.lidar_from_tf = True

        self.get_logger().info(
            f'LiDAR 위치({frame}): x={self.lidar_x:.3f} '
            f'y={self.lidar_y:.3f} yaw={self.lidar_yaw:.3f}'
        )

    def obstacles_in_map(self, x, y, yaw):
        if self.scan is None:
            return []

        msg = self.scan

        self.update_lidar_offset(msg.header.frame_id)

        sx = (
            x
            + math.cos(yaw) * self.lidar_x
            - math.sin(yaw) * self.lidar_y
        )

        sy = (
            y
            + math.sin(yaw) * self.lidar_x
            + math.cos(yaw) * self.lidar_y
        )

        points = []

        for i, r in enumerate(msg.ranges):
            if i % 2:
                continue

            if not math.isfinite(r):
                continue

            if not (
                max(0.08, msg.range_min)
                < r
                < min(3.0, msg.range_max)
            ):
                continue

            a = (
                yaw
                + self.lidar_yaw
                + msg.angle_min
                + i * msg.angle_increment
            )

            points.append((
                sx + r * math.cos(a),
                sy + r * math.sin(a)
            ))

        return points

    def find_target(self, x, y):
        poses = self.global_path.poses
        n = len(poses)

        nearest = min(
            range(n),
            key=lambda i: math.hypot(
                poses[i].pose.position.x - x,
                poses[i].pose.position.y - y
            )
        )

        idx = nearest
        previous = poses[idx].pose.position
        distance = 0.0

        # 순환 경로를 가정
        for k in range(1, n):
            idx = (
                nearest + self.path_direction * k
            ) % n

            p = poses[idx].pose.position

            distance += math.hypot(
                p.x - previous.x,
                p.y - previous.y
            )

            previous = p

            if distance >= self.target_distance:
                break

        p = poses[idx].pose.position

        return p.x, p.y

    def ahead_path(self, x, y, length):
        # 가장 가까운 점부터 진행 방향으로 length만큼의 global path 점들
        poses = self.global_path.poses
        n = len(poses)

        nearest = min(
            range(n),
            key=lambda i: math.hypot(
                poses[i].pose.position.x - x,
                poses[i].pose.position.y - y
            )
        )

        previous = poses[nearest].pose.position
        points = [(previous.x, previous.y)]
        distance = 0.0

        for k in range(1, n):
            p = poses[
                (nearest + self.path_direction * k) % n
            ].pose.position

            distance += math.hypot(
                p.x - previous.x,
                p.y - previous.y
            )

            points.append((p.x, p.y))
            previous = p

            if distance >= length:
                break

        return np.asarray(points)

    def plan_local(self, x, y, obstacles):
        # [추가] 로봇 주변 격자에서 global path 앞 지점까지 최단 경로 (Dijkstra)
        # 반환: (K, 2) map 좌표 경로, 못 찾으면 None
        res = self.grid_res
        n = int(round(2.0 * self.grid_half / res))
        ox = x - self.grid_half
        oy = y - self.grid_half

        occ = np.zeros((n, n), dtype=bool)          # [행=y, 열=x]

        if len(obstacles):
            ix = np.floor((obstacles[:, 0] - ox) / res).astype(int)
            iy = np.floor((obstacles[:, 1] - oy) / res).astype(int)
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
            occ[iy[ok], ix[ok]] = True

        # 각 셀에서 가장 가까운 장애물까지 거리
        dist = distance_transform_edt(~occ) * res

        sj = int((x - ox) / res)
        si = int((y - oy) / res)

        # 이미 장애물에 가까이 있으면 부풀림을 줄여서 빠져나올 수 있게
        radius = min(
            self.inflate,
            max(dist[si, sj] - 0.02, 0.06)
        )

        free = dist >= radius
        free[si, sj] = True

        # 장애물에 가까울수록 비싸게 -> 틈의 가운데로 지나감
        cost = 1.0 + self.clear_weight * np.clip(
            (self.clear_pref - dist) / self.clear_pref, 0.0, 1.0
        )

        # global path에서 멀어질수록 비싸게
        global_ahead = self.ahead_path(x, y, self.plan_goal_dist)

        gj = np.floor((global_ahead[:, 0] - ox) / res).astype(int)
        gi = np.floor((global_ahead[:, 1] - oy) / res).astype(int)
        ok = (gj >= 0) & (gj < n) & (gi >= 0) & (gi < n)

        if ok.any():
            mask = np.zeros((n, n), dtype=bool)
            mask[gi[ok], gj[ok]] = True
            dg = distance_transform_edt(~mask) * res
            cost += self.path_pull * np.minimum(dg, 1.0)

        # 이전 경로에서 멀어질수록 비싸게 -> 매 주기 좌우가 뒤집히지 않음
        if self.prev_plan is not None:
            pj = np.floor((self.prev_plan[:, 0] - ox) / res).astype(int)
            pi = np.floor((self.prev_plan[:, 1] - oy) / res).astype(int)
            ok = (pj >= 0) & (pj < n) & (pi >= 0) & (pi < n)

            if ok.any():
                mask = np.zeros((n, n), dtype=bool)
                mask[pi[ok], pj[ok]] = True
                dp = distance_transform_edt(~mask) * res
                cost += self.prev_weight * np.minimum(dp, 0.6) / 0.6

        # 8방향 연결 그래프
        idx = np.arange(n * n).reshape(n, n)
        rows, cols, weights = [], [], []

        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = (
                slice(max(0, -di), n - max(0, di)),
                slice(max(0, -dj), n - max(0, dj))
            )
            b = (
                slice(max(0, -di) + di, n - max(0, di) + di),
                slice(max(0, -dj) + dj, n - max(0, dj) + dj)
            )

            m = free[a] & free[b]
            w = math.hypot(di, dj) * res * 0.5 * (cost[a] + cost[b])

            ia, ib, wm = idx[a][m], idx[b][m], w[m]
            rows += [ia, ib]
            cols += [ib, ia]
            weights += [wm, wm]

        graph = coo_matrix(
            (
                np.concatenate(weights),
                (np.concatenate(rows), np.concatenate(cols))
            ),
            shape=(n * n, n * n)
        ).tocsr()

        start = idx[si, sj]

        d, pred = dijkstra(
            graph,
            directed=True,
            indices=start,
            return_predecessors=True
        )

        # 목표: global path를 따라 가장 먼, 도달 가능한 점
        # (경로가 장애물을 관통하는 부분은 건너뜀)
        goal = None

        for gx, gy in global_ahead[::-1]:
            gj = int((gx - ox) / res)
            gi = int((gy - oy) / res)

            if (
                1 <= gi < n - 1
                and 1 <= gj < n - 1
                and np.isfinite(d[idx[gi, gj]])
            ):
                goal = idx[gi, gj]
                break

        if goal is None or goal == start:
            return None

        cells = [goal]

        while cells[-1] != start:
            c = pred[cells[-1]]

            if c < 0:
                return None

            cells.append(c)

        cells = np.asarray(cells[::-1])

        points = np.column_stack((
            ox + (cells % n + 0.5) * res,
            oy + (cells // n + 0.5) * res
        ))

        points[0] = (x, y)

        return points

    def point_along(self, path, length):
        # 경로를 따라 length만큼 간 지점
        travelled = 0.0

        for a, b in zip(path[:-1], path[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])

            if travelled + seg >= length and seg > 1e-9:
                t = (length - travelled) / seg
                return (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1])
                )

            travelled += seg

        return float(path[-1, 0]), float(path[-1, 1])

    def rollout(self, x, y, yaw, omega, speed, switch=None, omega2=0.0):
        # [추가] switch(0~1): 예측 길이의 이 비율 지점부터 omega2 로 바꿈 (꺾었다가 펴는 S자 후보)
        trajectory = [(x, y, yaw)]

        # [수정] 예측 길이 최대 max_horizon(m): 빠를 때 2m짜리 긴 원호는
        # 격자 경로에서 크게 벗어나 직진만 뽑혀서 늦게 꺾었음
        steps = int(
            min(self.sim_time, self.max_horizon / speed) / self.dt
        )
        turned = 0.0
        switch_step = None if switch is None else max(1, int(round(steps * switch)))

        for step in range(steps):
            if switch_step is not None and step == switch_step:
                omega = omega2
                turned = 0.0

            # [수정] 90도까지 돈 다음에는 직진으로 이어서 예측
            # -> 모든 궤적의 길이가 같아져서, 급회전 궤적이
            #    '짧아서 안 부딪히는' 착시로 뽑히지 않음
            if turned >= self.max_rollout_turn:
                omega = 0.0

            turned += abs(omega) * self.dt

            middle_yaw = yaw + omega * self.dt / 2.0

            x += (
                speed
                * math.cos(middle_yaw)
                * self.dt
            )

            y += (
                speed
                * math.sin(middle_yaw)
                * self.dt
            )

            yaw = wrap(yaw + omega * self.dt)

            trajectory.append((x, y, yaw))

        return trajectory

    def body_check(self, traj, obstacles):
        # numpy로 한 번에 계산 (파이썬 이중 반복문은 1회 0.6초 걸렸음)
        # traj: (T, 3) 자세 배열
        # 반환: 각 자세에서 차체까지 최소 거리 (T,),
        #       차체 앞 금지영역(front_margin)에 점이 있는지 (T,)
        T = len(traj)

        if len(obstacles) == 0:
            return (
                np.full(T, np.inf),
                np.zeros(T, dtype=bool)
            )

        tx = traj[:, 0:1]
        ty = traj[:, 1:2]
        c = np.cos(traj[:, 2:3])
        s = np.sin(traj[:, 2:3])

        dx = obstacles[:, 0][None, :] - tx          # (T, M)
        dy = obstacles[:, 1][None, :] - ty

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        outside_x = np.maximum(
            np.maximum(
                local_x - self.front_length,
                -self.rear_length - local_x
            ),
            0.0
        )

        outside_y = np.maximum(
            np.abs(local_y) - self.half_width,
            0.0
        )

        d = np.hypot(outside_x, outside_y).min(axis=1)

        front_hit = np.any(
            (local_x >= self.front_length)
            & (local_x < self.front_length + self.front_margin)
            & (np.abs(local_y) < self.half_width),
            axis=1
        )

        return d, front_hit

    def path_length(self, trajectory):
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(trajectory[:-1], trajectory[1:])
        )

    def evaluate(self, trajectory, obstacles, margin, speed):
        # 반환: (사용할 궤적 또는 None(탈락), 최소 거리, 잘린 비율, 문제 있음 여부)
        # 현재 위치(trajectory[0])는 제외
        traj = np.asarray(trajectory[1:])
        d, front_hit = self.body_check(traj, obstacles)

        bad = (d < margin) | front_hit

        if not bad.any():
            return trajectory, float(d.min()), 0.0, False

        k = int(np.argmax(bad))            # 처음 걸리는 스텝 (0부터)

        if (k + 1) * self.dt * speed <= self.hard_dist:
            return None, float(d[:k + 1].min()), 1.0, True

        # 먼 미래에 걸리면 그 직전까지만 사용 (탈락 대신 감점)
        return (
            trajectory[:k + 1],
            float(d[:k].min()),
            1.0 - k / len(traj),
            True
        )

    def choose_path(self, x, y, yaw, obstacles, all_obstacles):
        # [추가] 격자 경로를 기준 경로로 사용. 못 찾으면 global path
        plan = self.plan_local(x, y, all_obstacles)

        if plan is not None and len(plan) >= 2:
            ref = plan
            planned = True
        else:
            ref = self.ahead_path(
                x, y, self.speed * self.sim_time + self.target_distance
            )
            planned = False

        self.prev_plan = plan

        tx, ty = self.point_along(ref, self.plan_target_dist)

        start_distance = math.hypot(
            tx - x,
            ty - y
        )

        # [추가] 지금 이미 hard_margin보다 가까우면 그 간격까지는 허용
        # (지금보다 더 가까워지는 경로만 탈락)
        current, _ = self.body_check(
            np.array([[x, y, yaw]]),
            obstacles
        )

        margin = min(
            self.hard_margin,
            max(float(current[0]) - 0.01, 1e-6)
        )

        # [추가] 현재 속도 + 최저 속도 두 가지로 궤적 생성
        # 빠를 때 급하게 꺾어야 하면 저속의 촘촘한 궤적을 고를 수 있게
        # (pure_pursuit.py가 곡률/장애물 여유에 맞춰 실제로 감속함)
        speeds = [self.speed]

        if self.speed > self.min_speed + 0.05:
            speeds.append(self.min_speed)

        candidates = []

        for speed in speeds:
            for omega in self.omega_samples:
                trajectory = self.rollout(
                    x, y, yaw, omega, speed
                )

                kept, clear, cut, _ = self.evaluate(
                    trajectory,
                    obstacles,
                    margin,
                    speed
                )

                candidates.append((omega, kept, clear, cut))

                # [추가] 꺾었다가 펴는 후보 (앞 절반 omega, 뒤 절반 직진 / 반대로 되돌림)
                # (원호만으로는 '옆으로 비킨 뒤 똑바로'가 안 돼서, 충분히 비키는 원호는 반대쪽 벽에 닿아 탈락
                #  -> 장애물 2번 틈에서 장애물 쪽에 붙는 완만한 원호만 남아 왼쪽 앞바퀴가 걸림, run56)
                if abs(omega) >= self.s_curve_min_omega:
                    for omega2 in (0.0, -omega):
                        trajectory = self.rollout(
                            x, y, yaw, omega, speed,
                            switch=self.s_curve_switch, omega2=omega2
                        )
                        kept, clear, cut, _ = self.evaluate(
                            trajectory, obstacles, margin, speed
                        )
                        candidates.append((omega, kept, clear, cut))

        # [추가] 현재 각속도 주변 후보를 먼저 보고, 없으면 전체에서 선택
        window = [
            c for c in candidates
            if abs(c[0] - self.odom_omega) <= self.omega_window + 1e-6
        ]

        result = self.pick_best(
            window, start_distance, tx, ty, ref
        )
        in_window = True

        if result[0] is None:
            result = self.pick_best(
                candidates, start_distance, tx, ty, ref
            )
            in_window = False

        best_path, best_omega, best_score, best_clearance = result

        valid = sum(
            1 for c in candidates
            if c[1] is not None
        )

        if best_path is not None:
            self.get_logger().info(
                f'DWA 통과궤적={valid}/{len(candidates)} '
                f'예측속도={self.speed:.2f} '
                f'omega={best_omega:+.2f} '
                f'(현재 {self.odom_omega:+.2f}'
                f'{"" if in_window else ", 창 밖"}) '
                f'clearance={best_clearance:.2f} '
                f'여유기준={margin:.2f} '
                f'경로길이={self.path_length(best_path):.2f}m '
                f'기준경로={"격자" if planned else "global"} '
                f'score={best_score:.2f} '
                f'계산={self.last_plan_ms:.0f}ms'
            )

        elif planned:
            # [추가] DWA 원호가 전부 막혀도 격자 경로(차체 폭만큼 부풀려 계산한
            # 충돌 없는 경로)가 있으면 그걸 보냄. 좁은 틈 입구에 비스듬히 서 있을 때
            # 앞으로 가는 원호로는 각도를 못 고침 -> pure_pursuit.py가
            # 방향이 많이 틀어졌으면 제자리 회전 후 따라감
            # (기존: 빈 경로 -> 후진+회전 복구 반복, 후진 중에는 거의 안 돌아서 갇힘)
            best_path = [
                (
                    float(a[0]), float(a[1]),
                    math.atan2(b[1] - a[1], b[0] - a[0])
                )
                for a, b in zip(ref[:-1], ref[1:])
            ]
            best_path.append((
                float(ref[-1][0]), float(ref[-1][1]), best_path[-1][2]
            ))

            self.get_logger().warning(
                f'DWA 원호 전부 막힘 -> 격자 경로 사용 '
                f'({self.path_length(best_path):.2f}m)',
                throttle_duration_sec=0.5
            )

        return best_path

    def path_distances(self, points, path):
        # 각 점과 경로(꺾은선) 사이 최소 거리, points: (P, 2)
        if len(path) < 2:
            return np.hypot(
                points[:, 0] - path[0, 0],
                points[:, 1] - path[0, 1]
            )

        a = path[:-1]
        v = path[1:] - a
        l2 = np.maximum((v * v).sum(axis=1), 1e-9)

        wx = points[:, 0:1] - a[None, :, 0]          # (P, S)
        wy = points[:, 1:2] - a[None, :, 1]

        t = np.clip((wx * v[:, 0] + wy * v[:, 1]) / l2, 0.0, 1.0)

        return np.hypot(
            wx - t * v[:, 0],
            wy - t * v[:, 1]
        ).min(axis=1)

    def pick_best(self, candidates, start_distance, tx, ty, ref):
        best_path = None
        best_omega = 0.0
        best_score = -float('inf')
        best_clearance = 0.0

        for omega, trajectory, clear, cut in candidates:
            if trajectory is None:
                continue

            ex, ey, eyaw = trajectory[-1]

            progress = (
                start_distance
                - math.hypot(tx - ex, ty - ey)
            )

            desired_yaw = math.atan2(
                ty - ey,
                tx - ex
            )

            heading = (
                1.0
                - abs(wrap(desired_yaw - eyaw))
                / math.pi
            )

            # [수정] 1/(clear+0.12)는 가까운 벽 하나에 점수가 좌우되어
            # 목표와 반대쪽(넓은 쪽)으로 꺾음 -> 0~1 선형, risk_range 밖은 0
            risk = max(
                0.0,
                self.risk_range - clear
            ) / self.risk_range

            # 안전 여유보다 가까이 지나가면 추가 감점
            margin_penalty = max(
                0.0,
                self.safety_margin - clear
            ) * 10.0

            # [수정] 이전 선택값 대신 실제 각속도 기준
            smooth = abs(
                omega - self.odom_omega
            )

            # [수정] 끝점만이 아니라 궤적 전체가 기준 경로에서 떨어진 평균 거리
            # 잘린 궤적은 점이 몇 개뿐일 수 있으므로 시작점 다음부터 4개 간격 + 끝점
            rest = np.asarray(trajectory[1:])[:, :2]
            samples = np.vstack((rest[3::4], rest[-1:]))
            off_path = float(
                self.path_distances(samples, ref).mean()
            )

            score = (
                2.0 * progress
                + 0.4 * heading
                - self.path_weight * off_path
                - self.risk_weight * risk
                - 0.1 * smooth             # [수정] 0.2 -> 0.1 (조향 변화 허용)
                - margin_penalty
                - self.cut_penalty * cut
            )

            if score > best_score:
                best_score = score
                best_path = trajectory
                best_omega = omega
                best_clearance = clear

        return best_path, best_omega, best_score, best_clearance

    # =====================================
    # [이동] 장애물 감지 / 회피 판단 / 비상정지 복구 (pure_pursuit.py 에서 옮김)
    # 판단 기준과 값은 pure_pursuit.py 에 있던 것과 같음
    # =====================================

    def to_robot(self, pts, x, y, yaw):
        # 점들의 로봇 기준 (앞, 옆) 좌표
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return forward, lateral

    def segments_distance(self, pts, a, b):
        # 각 점에서 선분들(a[i] -> b[i]) 까지의 최소 거리
        if len(a) == 0 or len(pts) == 0:
            return np.full(len(pts), np.inf)

        v = b - a
        l2 = np.maximum((v * v).sum(axis=1), 1e-9)

        wx = pts[:, 0:1] - a[None, :, 0]
        wy = pts[:, 1:2] - a[None, :, 1]

        t = np.clip((wx * v[:, 0] + wy * v[:, 1]) / l2, 0.0, 1.0)

        return np.hypot(wx - t * v[:, 0], wy - t * v[:, 1]).min(axis=1)

    def new_obstacle_points(self, pts):
        # 지도 벽 근처(map_wall_radius)가 아닌 점 = 지도에 없는 새 장애물
        if self.wall_mask is None:
            self.get_logger().warning(
                '/map 미수신: 벽도 장애물로 판단 중',
                throttle_duration_sec=2.0
            )
            return pts

        if len(pts) == 0:
            return pts

        info = self.map_info
        res = info.resolution
        ix = np.floor((pts[:, 0] - info.origin.position.x) / res).astype(int)
        iy = np.floor((pts[:, 1] - info.origin.position.y) / res).astype(int)
        inside = (ix >= 0) & (ix < info.width) & (iy >= 0) & (iy < info.height)

        wall = np.zeros(len(pts), dtype=bool)
        wall[inside] = self.wall_mask[iy[inside], ix[inside]]

        return pts[~wall]

    def update_localization_check(self, pts):
        # 스캔 점 대부분은 지도 벽 위에 찍혀야 정상.
        # 중앙값을 쓰므로 새 장애물 점이 일부 섞여도 영향 적음
        if self.map_dist is None or len(pts) < 20:
            return

        # 빠르게 회전 중이면 판정 건너뜀 (스캔 시점과 현재 자세 차이로 오차가 커 보임)
        if abs(self.odom_omega) > 0.3:
            return

        info = self.map_info
        res = info.resolution
        h, w = self.map_dist.shape

        ix = ((pts[:, 0] - info.origin.position.x) / res).astype(int)
        iy = ((pts[:, 1] - info.origin.position.y) / res).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        d = np.full(len(pts), 1.0)
        d[inside] = self.map_dist[iy[inside], ix[inside]]

        self.loc_error = float(np.median(d))

        now = self.now()

        if self.loc_error > self.loc_bad_dist:
            if self.loc_bad_since is None:
                self.loc_bad_since = now
        else:
            self.loc_bad_since = None

        bad = (
            self.loc_bad_since is not None
            and now - self.loc_bad_since > self.loc_bad_time
        )

        if bad and not self.loc_bad:
            self.get_logger().error(
                f'위치 추정 불량: 스캔-지도 오차 {self.loc_error:.2f}m '
                f'-> {self.loc_bad_speed}m/s로 감속. '
                f'RViz 2D Pose Estimate로 위치를 다시 잡아주세요'
            )
        elif self.loc_bad and not bad:
            self.get_logger().info('위치 추정 회복')

        self.loc_bad = bad

    def detect_obstacle(self, x, y, yaw, new):
        # 로봇 앞쪽에서 global path 근처(path_block_margin)에 있는 새 장애물까지 거리
        ahead = self.ahead_path(x, y, self.detect_dist)
        blocked = float('inf')

        if len(new) and len(ahead) >= 2:
            forward, _ = self.to_robot(new, x, y, yaw)
            cand = new[forward > 0]

            if len(cand):
                d = self.segments_distance(cand, ahead[:-1], ahead[1:])
                near = cand[d < self.path_block_margin]

                if len(near):
                    blocked = float(np.min(np.hypot(near[:, 0] - x, near[:, 1] - y)))

        self.blocked_distance = blocked
        self.obstacle_detected = blocked <= self.avoid_start_dist

        if self.obstacle_detected:
            self.last_block_time = self.now()

    def obstacle_passed(self, x, y, new):
        # 경로 기준으로 장애물을 지났는지:
        # 경로를 따라 로봇보다 passed_skip 앞 ~ slow_dist 구간에
        # 새 장애물이 path_block_margin 안에 없으면 True
        ahead = self.ahead_path(x, y, self.detect_dist)

        if len(ahead) < 2 or len(new) == 0:
            return True

        a, b = ahead[:-1], ahead[1:]
        seg = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        start = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
        sel = (start + seg > self.passed_skip) & (start < self.slow_dist)

        if not sel.any():
            return True

        d = self.segments_distance(new, a[sel], b[sel])
        return not bool((d < self.path_block_margin).any())

    def can_return(self, x, y, yaw, new):
        if self.obstacle_detected:
            return False

        # 앞쪽 경로 위에 장애물이 아직 감속 거리 안에 있으면 복귀 안 함
        if self.blocked_distance < self.slow_dist:
            return False

        if self.now() - self.last_block_time < self.avoid_hold_time:
            return False

        ahead = self.ahead_path(x, y, 0.0)
        distance = math.hypot(ahead[0][0] - x, ahead[0][1] - y)

        if distance > self.return_distance:
            return False

        # 앞쪽 장애물만 봄 (이미 지나간 옆/뒤 장애물 때문에 복귀가 늦어지지 않게)
        if len(new):
            forward, _ = self.to_robot(new, x, y, yaw)
            front = new[forward > 0]

            if len(front) and np.min(np.hypot(front[:, 0] - x, front[:, 1] - y)) < self.return_clearance:
                return False

        return True

    def local_speed_limit(self, x, y, yaw, pts, ahead_only=False):
        # LOCAL 속도 상한
        # ahead_only=True: 장애물을 이미 지난 상태 -> 차체 앞면보다 앞에 있는 점만 봄
        # - 앞쪽(차체 폭 + 0.15) 장애물까지 여유가 0.25m면 최저, 멀수록 빠르게
        # - 차체 주변 가장 가까운 점까지 여유가 0.10m면 최저 (틈 통과 시 천천히)
        front = float('inf')
        near = float('inf')

        if len(pts):
            forward, lateral = self.to_robot(pts, x, y, yaw)

            m = (
                (forward > (self.front_length if ahead_only else 0.0))
                & (np.abs(lateral) < self.half_width + 0.15)
            )

            if m.any():
                front = float(np.min(forward[m] - self.front_length))

            m = forward > (self.front_length if ahead_only else -self.rear_length)

            if m.any():
                out_x = np.maximum(
                    np.maximum(forward[m] - self.front_length, -self.rear_length - forward[m]),
                    0.0
                )
                out_y = np.maximum(np.abs(lateral[m]) - self.half_width, 0.0)
                near = float(np.min(np.hypot(out_x, out_y)))

        # 장애물을 지난 상태면 앞쪽 여유 기준 완화
        front_min = self.passed_front_min if ahead_only else 0.25
        front_gain = self.passed_front_gain if ahead_only else self.local_front_gain

        return min(
            self.local_max_speed if ahead_only else self.gap_cap(),
            self.local_speed + front_gain * max(0.0, front - front_min),
            self.local_speed + self.local_side_gain * max(0.0, near - 0.10)
        )

    def gap_cap(self):
        # [추가] 장애물을 지나기 전 속도 상한: 접근 감속 구간(approach_end_dist 밖)에서는
        # 회피 모드여도 접근 감속 속도까지 허용 (일찍 회피 모드에 들어가도 0.6으로 뚝 떨어지지 않게)
        if self.blocked_distance > self.approach_end_dist:
            return max(self.gap_max_speed, self.obstacle_slow_speed())

        return self.gap_max_speed

    def obstacle_slow_speed(self):
        # 경로 위 장애물에 다가가면 slow_dist ~ avoid_start_dist 구간에서
        # global_speed -> approach_speed로 선형 감속
        b = self.blocked_distance

        if b >= self.approach_start_dist:
            return self.global_speed

        ratio = max(
            0.0,
            (b - self.approach_end_dist) / (self.approach_start_dist - self.approach_end_dist)
        )

        return self.approach_speed + (self.global_speed - self.approach_speed) * ratio

    def emergency_stop(self, x, y, yaw, pts):
        if len(pts) == 0:
            return False

        forward, lateral = self.to_robot(pts, x, y, yaw)

        return bool(np.any(
            (forward > 0)
            & (forward < self.front_length + self.emergency_gap)
            & (np.abs(lateral) < self.half_width)
        ))

    def rear_clear(self, x, y, yaw, pts):
        if len(pts) == 0:
            return True

        forward, lateral = self.to_robot(pts, x, y, yaw)

        return not bool(np.any(
            (forward > -(self.rear_length + self.rear_check_dist))
            & (forward < 0)
            & (np.abs(lateral) < self.half_width)
        ))

    def rotation_clear(self, x, y, pts):
        # 제자리 회전해도 차체 모서리가 안 닿는지
        if len(pts) == 0:
            return True

        return bool(np.all(np.hypot(pts[:, 0] - x, pts[:, 1] - y) > self.rotate_clearance))

    def local_path_error(self, x, y, yaw):
        # 회피 경로(마지막으로 발행한 /local_path)를 따라 escape_path_dist 앞 지점 방향과
        # 로봇 방향의 차이 (rad)
        path = self.last_path_points

        if path is None or len(path) < 2:
            return None

        nearest = int(np.argmin(np.hypot(path[:, 0] - x, path[:, 1] - y)))
        target = path[-1]
        travelled = 0.0

        for i in range(nearest + 1, len(path)):
            travelled += math.hypot(path[i, 0] - path[i - 1, 0], path[i, 1] - path[i - 1, 1])

            if travelled >= self.escape_path_dist:
                target = path[i]
                break

        return wrap(math.atan2(target[1] - y, target[0] - x) - yaw)

    def escape_sign(self, x, y, yaw, pts):
        # 회피 중이고 회피 경로가 있으면 그 경로가 향하는 쪽으로 회전
        if self.avoiding:
            error = self.local_path_error(x, y, yaw)

            if error is not None and abs(error) >= 0.1:
                return 1 if error > 0 else -1

        # 전방 가까운 점들이 좌/우 어느 쪽에 많은지 보고 적은 쪽(열린 쪽)으로 회전
        if len(pts):
            forward, lateral = self.to_robot(pts, x, y, yaw)
            m = (
                (forward > 0) & (forward < self.escape_check_dist)
                & (np.abs(lateral) < self.escape_check_dist)
            )

            if m.any():
                d = np.maximum(np.hypot(forward[m], lateral[m]), 0.05)
                weight = float(np.sum(np.where(lateral[m] >= 0, 1.0, -1.0) / d))

                if abs(weight) > 1.0:
                    return -1 if weight > 0 else 1

        # 좌우가 비슷하면 Global Path 쪽으로
        tx, ty = self.find_target(x, y)
        lateral = -math.sin(yaw) * (tx - x) + math.cos(yaw) * (ty - y)

        return 1 if lateral >= 0 else -1

    def update_recovery(self, x, y, yaw, pts, path_ok):
        # 비상정지/DWA 경로 없음 지속 -> 1초 복구 동작 (후진+회전 또는 제자리 회전)
        # 반환: 복구 중이면 (v, w), 아니면 None
        now = self.now()

        if self.avoiding and not path_ok:
            if self.no_path_since is None:
                self.no_path_since = now
        else:
            self.no_path_since = None

        local_stuck = (
            self.no_path_since is not None
            and now - self.no_path_since > self.local_stuck_time
        )

        emergency = self.emergency_stop(x, y, yaw, pts)

        if now >= self.backup_until:
            if emergency:
                reason = '비상정지'
            elif local_stuck:
                reason = 'DWA 경로 없음 지속'
            else:
                reason = None

            if reason is not None:
                # 회피 경로가 있으면 매번 그 방향으로 새로 계산
                # 회피 경로가 없을 때만 직전 방향 유지 (좌우 번갈아 돌며 제자리 방지)
                path_guided = self.avoiding and path_ok

                if (
                    path_guided
                    or now - self.recovery_end > self.recovery_keep_time
                ):
                    self.recovery_sign = self.escape_sign(x, y, yaw, pts)

                self.recovery_start = now
                self.backup_until = now + self.backup_duration
                self.recovery_end = self.backup_until
                self.no_path_since = None

                self.get_logger().warning(
                    f'{reason}: 복구 시작 '
                    f'({"왼쪽" if self.recovery_sign > 0 else "오른쪽"}으로 회전)'
                )

        # 복구 중 회피 경로 방향과 거의 맞으면(15° 이내) 최소 시간 뒤 바로 종료
        if (
            now < self.backup_until
            and now - self.recovery_start >= self.recovery_min_time
            and self.avoiding
            and path_ok
            and not emergency
        ):
            error = self.local_path_error(x, y, yaw)

            if error is not None and abs(error) < self.recovery_align_tol:
                self.backup_until = now
                self.recovery_end = now
                self.get_logger().info(
                    f'복구 조기 종료: 회피 경로 방향과 일치 ({math.degrees(error):+.0f}°)'
                )

        if now < self.backup_until:
            if self.rear_clear(x, y, yaw, pts):
                cmd = (-self.backup_speed, self.recovery_sign * self.backup_turn)
                action = '후진+회전'
            else:
                cmd = (0.0, self.recovery_sign * self.rotate_speed)
                action = '제자리 회전 (뒤 막힘)'

            self.get_logger().warning(
                f'복구 중: {action} (남은 {self.backup_until - now:.1f}s)',
                throttle_duration_sec=0.3
            )
            return cmd

        return None

    def publish_status(self, ok, passed=False, limit=0.0, limit_passed=0.0,
                       global_limit=0.0, recovery=None, rotation_clear=False,
                       in_gap=False):
        data = [0.0] * STATUS_SIZE
        data[STATUS_OK] = 1.0 if ok else 0.0
        data[STATUS_AVOIDING] = 1.0 if (ok and self.avoiding) else 0.0
        data[STATUS_PASSED] = 1.0 if passed else 0.0
        data[STATUS_LIMIT] = float(limit)
        data[STATUS_LIMIT_PASSED] = float(limit_passed)
        data[STATUS_GLOBAL_LIMIT] = float(global_limit)
        data[STATUS_RECOVERY] = 1.0 if recovery is not None else 0.0
        data[STATUS_RECOVERY_V] = float(recovery[0]) if recovery is not None else 0.0
        data[STATUS_RECOVERY_W] = float(recovery[1]) if recovery is not None else 0.0
        data[STATUS_ROTATION_CLEAR] = 1.0 if rotation_clear else 0.0
        data[STATUS_BLOCKED] = float(min(self.blocked_distance, 99.0))
        data[STATUS_LOC_ERROR] = float(self.loc_error)
        data[STATUS_IN_GAP] = 1.0 if (ok and in_gap) else 0.0

        msg = Float32MultiArray()
        msg.data = data
        self.status_pub.publish(msg)

    def update_blind_memory(self, x, y, yaw, obstacles):
        # [추가] 사각지대 기억 갱신, 지금 시야 밖에 있는 기억 점 (M, 2) 반환
        now = self.now()
        msg = self.scan
        lx = x + math.cos(yaw) * self.lidar_x - math.sin(yaw) * self.lidar_y
        ly = y + math.sin(yaw) * self.lidar_x + math.cos(yaw) * self.lidar_y

        def visible(p):
            # 라이다 시야각 안 (가장자리 0.05rad 여유) -> 새 스캔이 대신함
            a = np.arctan2(p[:, 1] - ly, p[:, 0] - lx) - (yaw + self.lidar_yaw)
            a = np.arctan2(np.sin(a), np.cos(a))
            return (a > msg.angle_min + 0.05) & (a < msg.angle_max - 0.05)

        mem = self.memory

        if len(mem):
            keep = (
                (now - mem[:, 2] <= self.memory_time)
                & (np.hypot(mem[:, 0] - x, mem[:, 1] - y) <= self.memory_range + 0.3)
            )
            mem = mem[keep]
            mem = mem[~visible(mem)]

        if len(obstacles):
            cur = obstacles[np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - y) <= self.memory_range]
            mem = np.vstack((mem, np.c_[cur, np.full(len(cur), now)]))

        if len(mem):
            order = np.argsort(-mem[:, 2])
            mem = mem[order]
            key = np.floor(mem[:, :2] / self.memory_voxel).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            mem = mem[idx]

        self.memory = mem

        if len(mem) == 0:
            return np.zeros((0, 2))

        return mem[~visible(mem), :2]

    def update_avoidance(self, x, y, yaw, obstacles, path_ok, blind=None):
        # 매 주기: 감지 -> 복구 -> 모드 전환 -> 속도 상한 -> /avoid_status 발행
        lx = x + math.cos(yaw) * self.lidar_x - math.sin(yaw) * self.lidar_y
        ly = y + math.sin(yaw) * self.lidar_x + math.cos(yaw) * self.lidar_y

        if len(obstacles):
            pts = obstacles[np.hypot(obstacles[:, 0] - lx, obstacles[:, 1] - ly) < self.detect_dist]
        else:
            pts = obstacles

        new = self.new_obstacle_points(pts)

        self.update_localization_check(pts)
        self.detect_obstacle(x, y, yaw, new)

        # [추가] 후진 가능/제자리 회전 가능 판단에는 사각지대 기억 점도 포함
        pts_all = pts if blind is None or len(blind) == 0 else np.vstack((pts, blind))

        recovery = self.update_recovery(x, y, yaw, pts_all, path_ok)

        # 복구 중에는 모드를 바꾸지 않음
        if recovery is None:
            if self.obstacle_detected and not self.avoiding:
                self.avoiding = True
                self.get_logger().warning(
                    f'LOCAL 전환: 장애물 {self.blocked_distance:.2f}m'
                )

            if self.avoiding and self.can_return(x, y, yaw, new):
                self.avoiding = False
                self.get_logger().info('GLOBAL 모드 복귀')

        global_limit = min(
            self.obstacle_slow_speed(),
            self.loc_bad_speed if self.loc_bad else self.global_speed
        )

        limit = self.local_speed_limit(x, y, yaw, pts, ahead_only=False)
        limit_passed = self.local_speed_limit(x, y, yaw, pts, ahead_only=True)

        # [추가] 틈 안에 들어와 정렬됐으면 gap_speed까지 허용
        in_gap = (
            recovery is None
            and self.avoiding
            and path_ok
            and self.gap_check(x, y, yaw, pts)
        )

        if in_gap != self.in_gap:
            self.get_logger().info(
                '틈 안 진입: 가속 허용' if in_gap else '틈 가속 해제'
            )
        self.in_gap = in_gap

        if in_gap:
            limit = max(limit, self.gap_speed)
            limit_passed = max(limit_passed, self.gap_speed)

        self.publish_status(
            True,
            passed=self.obstacle_passed(x, y, new),
            limit=limit,
            limit_passed=limit_passed,
            global_limit=global_limit,
            recovery=recovery,
            rotation_clear=self.rotation_clear(x, y, pts_all),
            in_gap=in_gap,
        )

    def gap_check(self, x, y, yaw, pts):
        # 틈 안에 들어와서 빠르게 지나가도 되는 상태인지
        if len(pts) == 0 or abs(self.odom_omega) > self.gap_omega_max:
            return False

        error = self.local_path_error(x, y, yaw)

        if error is None or abs(error) > self.gap_heading_tol:
            return False

        forward, lateral = self.to_robot(pts, x, y, yaw)

        # 차체 옆 (앞뒤 길이 범위) 가까운 점이 있어야 틈 안 (장애물 옆을 지나는 중)
        # -> 틈 입구로 향하는 중(장애물이 아직 앞에 있음)에는 해당 안 됨
        beside = (
            (forward > -self.rear_length)
            & (forward < self.front_length)
            & (np.abs(lateral) < self.half_width + self.gap_side_dist)
        )

        if not beside.any():
            return False

        # 차체 폭 안 앞길이 비어 있어야 함
        ahead = (
            (forward > 0.0)
            & (forward < self.front_length + self.gap_front_clear)
            & (np.abs(lateral) < self.half_width + 0.01)
        )

        return not bool(ahead.any())

    def publish_path(self, trajectory):
        msg = Path()

        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        if trajectory is not None:
            for x, y, yaw in trajectory:
                p = PoseStamped()
                p.header = msg.header

                p.pose.position.x = x
                p.pose.position.y = y

                p.pose.orientation.z = math.sin(yaw / 2.0)
                p.pose.orientation.w = math.cos(yaw / 2.0)

                msg.poses.append(p)

        self.pub.publish(msg)

    def plan(self):
        now = self.now()

        if (
            self.global_path is None
            or len(self.global_path.poses) < 2
        ):
            self.publish_path(None)
            self.publish_status(False)
            self.get_logger().warning(
                'DWA 대기: Global Path 없음',
                throttle_duration_sec=1.0
            )
            return

        if (
            self.odom_time is None
            or now - self.odom_time > self.odom_timeout
        ):
            self.publish_path(None)
            self.publish_status(False)
            self.get_logger().warning(
                'DWA 대기: /odom 없음 또는 오래됨',
                throttle_duration_sec=1.0
            )
            return

        if (
            self.scan_time is None
            or now - self.scan_time > self.scan_timeout
        ):
            self.publish_path(None)
            self.publish_status(False)
            self.get_logger().warning(
                'DWA 대기: /scan 없음 또는 오래됨',
                throttle_duration_sec=1.0
            )
            return

        pose = self.robot_pose()

        if pose is None:
            self.publish_path(None)
            self.publish_status(False)
            return

        x, y, yaw = pose

        self.fix_direction(x, y, yaw)

        # [추가] 실제 주행 속도로 궤적 예측
        self.speed = min(
            self.max_speed,
            max(self.min_speed, self.odom_speed)
        )

        t0 = time.perf_counter()

        obstacles = np.asarray(
            self.obstacles_in_map(x, y, yaw),
            dtype=float
        ).reshape(-1, 2)

        # [추가] 궤적이 닿을 수 있는 범위 밖의 점은 계산에서 제외
        reach = (
            self.speed * self.sim_time
            + max(self.front_length + self.front_margin,
                  self.rear_length, self.half_width)
            + 0.3
        )

        # [추가] 사각지대 기억 점 (시야 밖 뒷바퀴 옆/뒤)
        blind = self.update_blind_memory(x, y, yaw, obstacles)
        scan_obstacles = obstacles

        if len(blind):
            obstacles = np.vstack((obstacles, blind))

        near_obstacles = obstacles

        if len(obstacles):
            near = np.hypot(
                obstacles[:, 0] - x,
                obstacles[:, 1] - y
            ) <= reach
            near_obstacles = obstacles[near]

        # 격자 경로 계획은 3m 안 전체 점 + 지도 벽, DWA 충돌 계산은 가까운 스캔 점(+사각지대 기억)만
        plan_obstacles = obstacles

        if self.map_points is not None:
            m = self.map_points
            close = (
                (np.abs(m[:, 0] - x) <= self.map_plan_range)
                & (np.abs(m[:, 1] - y) <= self.map_plan_range)
            )
            plan_obstacles = np.vstack((obstacles, m[close]))

        # [추가] 계산 중 예외가 나도 노드가 죽지 않게 (빈 경로 발행 + 에러 로그)
        try:
            trajectory = self.choose_path(
                x, y, yaw, near_obstacles, plan_obstacles
            )

        except Exception:
            self.get_logger().error(
                'DWA 계산 오류:\n' + traceback.format_exc(),
                throttle_duration_sec=1.0
            )
            trajectory = None

        self.last_plan_ms = (time.perf_counter() - t0) * 1000.0

        if trajectory is None:
            self.get_logger().warning(
                'DWA: 충돌 없는 궤적 없음',
                throttle_duration_sec=0.5
            )

        self.publish_path(trajectory)

        self.last_path_points = (
            np.array([(p[0], p[1]) for p in trajectory])
            if trajectory is not None else None
        )

        # [이동] 장애물 감지 / 회피 판단 / 비상정지 -> /avoid_status (pure_pursuit.py 에서 옮김)
        try:
            self.update_avoidance(
                x, y, yaw, scan_obstacles, trajectory is not None and len(trajectory) >= 2,
                blind
            )

        except Exception:
            self.get_logger().error(
                '회피 판단 오류:\n' + traceback.format_exc(),
                throttle_duration_sec=1.0
            )
            self.publish_status(False)


def main(args=None):
    # [추가] use_sim_time 기본값 true: 명령에 따로 안 주면 자동으로 붙임
    # (python3 파일.py 만 쳐도 --ros-args -p use_sim_time:=true 와 같음)
    if args is None:
        args = list(sys.argv)

        if not any('use_sim_time' in a for a in args):
            # [수정] 실제 LIMO: export LIMO_REAL=1 이면 false (Gazebo 기본 true)
            sim = 'false' if os.environ.get('LIMO_REAL') == '1' else 'true'
            args += ['--ros-args', '-p', f'use_sim_time:={sim}']

    rclpy.init(args=args)

    node = ObstacleDWA()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
