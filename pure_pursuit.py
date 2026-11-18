import os
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from tf2_ros import Buffer, TransformListener, TransformException

# [수정] 장애물 감지 관련 코드는 모두 local_path_planner.py 로 옮김
# 이 노드는 /scan, /map 을 쓰지 않고 경로 추종만 함:
#   - GLOBAL 모드: /global_path 추종
#   - LOCAL 모드: /local_path (local_path_planner.py의 회피 경로) 추종
#   - 모드, 장애물 기반 속도 상한, 비상정지 복구 명령 등은
#     local_path_planner.py 가 /avoid_status 로 보내준 값을 그대로 따름
# local_path_planner.py 없이 실행하면 경고 후 global path만 추종 (장애물 기능 없음)

# /avoid_status (std_msgs/Float32MultiArray) 내용 순서
# local_path_planner.py 의 STATUS_* 와 반드시 같아야 함
STATUS_OK = 0               # 1: 센서/위치 정상, 0: 대기 중
STATUS_AVOIDING = 1         # 1: 회피(LOCAL) 모드
STATUS_PASSED = 2           # 1: 경로 기준으로 장애물을 이미 지남
STATUS_LIMIT = 3            # LOCAL 속도 상한 (주변 여유 기준)
STATUS_LIMIT_PASSED = 4     # LOCAL 속도 상한 (장애물을 지난 상태 기준)
STATUS_GLOBAL_LIMIT = 5     # GLOBAL 속도 상한 (장애물 접근 감속, 위치 불량 감속)
STATUS_RECOVERY = 6         # 1: 비상정지 복구 중 (아래 명령을 그대로 발행)
STATUS_RECOVERY_V = 7
STATUS_RECOVERY_W = 8
STATUS_ROTATION_CLEAR = 9   # 1: 제자리 회전해도 안 닿음
STATUS_BLOCKED = 10         # 경로 위 장애물까지 거리 (로그용)
STATUS_LOC_ERROR = 11       # 스캔-지도 오차 (로그용)
STATUS_IN_GAP = 12          # 1: 틈 안에 들어와 정렬됨 -> 틈 통과 속도 허용
STATUS_SIZE = 13


def yaw_of(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class SwitchingPurePursuit(Node):

    def __init__(self):
        super().__init__('pure_pursuit')

        self.global_path = None
        self.local_path = None

        self.odom_time = None
        self.odom_pose = None
        self.odom_omega = 0.0
        self.local_time = None

        # [추가] local_path_planner.py 가 보내는 회피 상태
        self.status = None
        self.status_time = None
        self.status_ever = False
        self.global_only_logged = False

        self.avoiding = False

        self.previous_angular = 0.0

        # =====================================
        # TF
        # =====================================

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

        self.map_frame = 'map'
        self.robot_frame = 'base_link'
        self.odom_frame = 'odom'

        # =====================================
        # Pure Pursuit 설정
        # =====================================

        # local_path_planner.py와 반드시 동일하게
        self.path_direction = -1   # auto_direction=True면 시작 시 자동으로 덮어씀
        self.auto_direction = True
        self.direction_fixed = False

        self.global_speed = 1.2            # [수정] 0.35 -> 1.0 -> 1.2 (최대 직진 속도, 커브는 turn_speed 0.9 그대로)
        # [수정] LOCAL(회피) 속도: 0.15 고정 -> 0.15 ~ 0.8 가변
        # 앞/옆 장애물이 멀면 빠르게, 가까우면(틈 통과 등) 느리게
        # (주변 여유에 따른 상한은 local_path_planner.py 가 계산해서 보내줌)
        # [수정] 0.25 -> 0.5: 회피 중 최저 속도 (local_path_planner.py local_speed/min_speed와 같게)
        self.local_speed = 0.48            # [수정] 0.5 -> 0.35 -> 0.42 -> 0.48 (local_path_planner.py local_speed/min_speed와 같게)
        # (run64: 장애물 옆을 지나는 동안 바퀴당 약 6초를 최저 0.35로 달림)
        self.local_max_speed = 0.8         # 최고 속도 (local_path_planner.py max_speed와 같게)
        self.min_global_speed = 0.15       # 감속해도 이 이하로는 안 내려감

        # [수정] GLOBAL lookahead를 속도에 비례하게
        # ld = base + gain * 현재속도 (0.6 ~ 1.1m)
        self.lookahead_base = 0.4
        self.lookahead_gain = 0.7
        self.global_lookahead_min = 0.6
        self.global_lookahead_max = 1.1
        self.global_lookahead = self.global_lookahead_min
        # [수정] LOCAL lookahead도 속도에 비례 (0.25 ~ 0.6m)
        self.local_lookahead_min = 0.25
        self.local_lookahead_max = 0.6
        self.local_lookahead = self.local_lookahead_min

        # [수정] 회전 강화: 최대 각속도 1.0 -> 1.8, 변화량 0.10 -> 0.25 (GLOBAL)
        # [수정] 3.0 -> 4.5: 커브 속도 0.8에서 가장 급한 커브(곡률 1.33)가
        # 회전 보상 4배 후 약 4.3 필요 -> 3.0이면 잘려서 덜 돎
        self.max_angular = 4.5                # 회전 보상 후 명령 상한
        self.max_angular_change = 1.0         # [수정] 0.5 -> 1.0 GLOBAL 모드용 (0.1초당, turn_gain 2배 증가에 맞춤)
        self.max_angular_change_local = 1.0   # [수정] 0.40 -> 1.0 LOCAL(회피) 모드용
        # [추가] LOCAL 회전 명령 저역 필터: 매 주기 새 값의 이 비율만 반영
        # (기록: 회피 중 0.1초마다 방향이 바뀌는 톱니 58회/분, 일반 주행의 3배)
        self.local_omega_filter = 0.5
        self.steer_sign = 1.0

        # [추가] GLOBAL: 방향 오차에 비례한 회전 (복귀 직후처럼 많이 틀어졌을 때
        # pure pursuit(ω = v·κ)만으로는 회전이 약함 -> 둘 중 큰 값 사용)
        self.heading_gain = 1.5

        # [수정] 속도 규칙 단순화: 회전 중이면 turn_speed, 아니면 직진 속도
        # (기존: 곡률/횡가속도/회전한계/방향오차 제한의 최솟값 -> 커브마다 들쭉날쭉)
        # GLOBAL 직진 1.0 / 커브 0.9, LOCAL 직진 최대 local_max_speed / 회전 중 0.3
        # [수정] 0.8 -> 0.9: 일반 주행 커브 속도. 시뮬레이션 한 바퀴 약 3초 단축, 커브 벽 여유 18~22cm 유지
        # (실제 주행에서 커브 바깥 벽에 붙으면 0.85 또는 0.8로 되돌리기)
        self.turn_speed = 0.9              # GLOBAL 커브/회전 중 주행 속도
        self.passed_turn_speed = 0.6       # [수정] 0.5 -> 0.6 회피 중 장애물을 지난 뒤 커브 속도 (turn_speed와 분리)
        # LOCAL(회피) 회전 중 속도: 0.5면 틈 입구에 도착하기 전에 옆으로 덜 빠져서
        # 0.5m 틈 통과 실패 (시뮬레이션 0.5: 4/5 통과, 0.3: 5/5 통과)
        # [수정] 0.4 -> 0.3으로 되돌림: 0.4는 시뮬레이션에선 괜찮았지만 실제 주행(13번째)에서
        # 장애물 2번 틈 입구에 0.38~0.40으로 들어가 회전이 늦게 들어오며 비상정지 4회
        # [수정] 0.35 -> 0.3으로 되돌림: 실제 주행(run20)에서 0.35로 장애물 2번 틈 입구에
        # 들어가며 회전이 늦게 먹다가 과회전 -> 3바퀴 모두 비상정지.
        # 틈 통과 속도는 대신 local_path_planner.py 틈 안 가속(gap_speed 0.5)으로 올림
        # [수정] 0.3 -> 0.7: 회피 중 회전 상한 (요청). 좁은 틈 감속은 local_path_planner.py 여유 기반 상한(최저 0.25)이 담당
        # (시뮬레이션 DWA: 한 바퀴 65.0 -> 61.5초, 충돌 0. 문제가 있으면 0.3으로 되돌리기)
        self.local_turn_speed = 0.6        # [수정] 0.7 -> 0.6 (요청)
        self.turn_kappa = 0.5              # 곡률(1/m) 이상이면 회전 (반경 2m 이하)
        # [추가] LOCAL(회피): 경로 곡률이 local_kappa_range[0] 이상이면 감속 시작,
        # [1] 이상이면 local_kappa_speed 까지 (사이는 비례)
        self.local_kappa_range = (0.3, 0.8)
        self.local_kappa_speed = 0.4
        self.turn_heading = 0.35           # 방향 오차(rad, 약 20°) 이상이면 회전
        # [수정] 0.5 -> 1.0: 짧은 직선마다 0.5 -> 1.0 -> 0.5로 가감속하던 것 줄임
        self.turn_hold_time = 1.0          # 회전 판정 후 유지 시간 (0.5 <-> 1.0 깜빡임 방지)
        self.turn_until = 0.0
        # [추가] 장애물을 지난 상태라도 목표점 곡률이 이보다 크면 회피 속도 유지
        # (장애물 1번처럼 지난 뒤 경로로 급히 돌아올 때 0.45로 꺾다 벽에 붙어 비상정지)
        self.passed_kappa = 1.0

        # [추가] 경로를 등지고 있을 때 제자리 회전 명령 (보상 포함 값)
        self.align_omega = 2.0
        self.align_preview = 2.5           # 경로 방향 맞출 때 보는 global path 길이
        # [추가] LOCAL: 목표 방향 오차가 이보다 크면 제자리 회전 먼저 (약 35°)
        # 좁은 틈 입구에 비스듬히 서면 전진 원호로는 각도를 못 고쳐서 갇혔음
        # [수정] 0.6(35°) -> 1.2(약 70°): 그보다 작은 오차는 멈추지 않고 천천히 움직이며 꺾음
        # (기록: 장애물 3번 진입 때 35~48° 오차에도 멈춰서 제자리 회전)
        # 제자리 회전해도 되는지(주변이 비었는지)는 local_path_planner.py 가 알려줌
        self.local_align_heading = 1.2
        # [수정] 1.5 -> 2.5: 앞쪽 2.5m가 곧을 때만 직진 속도로 가속
        # (실제 경로 기준 짧은 가속 구간 5개 -> 3개, 한 바퀴 약 +5초)
        self.curve_preview = 2.5           # 앞쪽 경로 곡률을 미리 보는 거리 (커브 전 감속)
        # [추가] GLOBAL(일반 주행) 전용 커브 판정 범위
        # 2.5m/1.0초는 커브 속도 0.5 시절 1.0<->0.5 가감속 반복을 줄이려던 값.
        # 커브 속도 0.8에서는 반복 폭이 작아서, 실제로 휘는 곳에서만 0.8로 달리도록 좁힘
        # (회피 중(LOCAL)은 curve_preview / turn_hold_time 그대로)
        self.global_curve_preview = 1.5
        self.global_turn_hold_time = 0.5

        # [추가] 회전 보상 배율 (실제 회전이 명령의 절반 정도라서)
        # 실제값이 명령(보상 전)보다 계속 작으면 올리고, 크면(과회전) 내리기
        # [수정] 2.0 -> 4.0: 기록 분석 결과 2배 보상 상태에서도 실제 회전이
        # 명령의 약 20% (원하는 회전의 약 40%)였음. 속도 0.2~0.6 모두 비슷
        self.turn_gain = 4.0
        # [수정] 0.5 -> 0.15: 저속(0.2~0.4)에서도 실제/명령 비율이 0.23으로
        # 고속과 같았음 -> 전진 중에는 거의 전부 적용
        self.turn_gain_speed = 0.15       # 이 속도 이상에서 turn_gain 전부 적용
        # [추가] GLOBAL(일반 주행) 회전 보상: 작은 회전 명령이 실제로 얼마나 먹는지 주행 중 추정
        # (기록 run25, 속도 0.9~1.0: 발행 명령 0.8 이하 -> 실제 약 80%, 그 이상 -> 약 25~30%.
        #  예전 기록에서는 작은 명령도 20~25%로 측정된 적 있음 -> 고정값 대신 추정)
        # 4배를 전부 적용하면 작은 회전이 잘 먹을 때 과회전 -> 커브 안쪽으로 최대 36cm 파고들어 충돌.
        # 발행 명령 global_turn_break 까지는 1/추정비율 배, 넘는 부분만 turn_gain 배
        # (회피 중(LOCAL)은 기존대로 전부 turn_gain 배)
        # [추가] LOCAL(회피) 회전 보상 = local_turn_offset + 배율 x 원하는 회전
        # 배율: 속도 0.35 이하 local_gain_slow, 0.55 이상 local_gain_fast (사이는 보간)
        # 원하는 회전이 local_offset_ramp 보다 작으면 시작값도 비례해서 줄임 (0 근처 좌우 떨림 방지)
        self.local_turn_offset = 0.6
        self.local_offset_ramp = 0.15
        self.local_gain_slow = 3.0
        self.local_gain_fast = 2.0
        self.local_gain_speeds = (0.35, 0.55)
        # [추가] 회피 중 실제 회전 피드백
        self.fb_min_turn = 0.15     # 원하는 회전이 이보다 작으면 적분 안 함 (rad/s, 보상 전)
        self.fb_ratio = 0.7         # 실제가 원하는 것의 이 비율보다 작으면 적분
        self.fb_ki = 2.0            # 적분 속도 (명령 증가량 / 초 / 모자란 rad/s)
        self.fb_int_max = 1.5       # 적분으로 더하는 최대 명령
        self.fb_over_band = 0.15    # 이만큼 넘치는 건 허용
        self.fb_kd = 1.5            # 넘친 rad/s 당 빼는 명령
        self.fb_counter = 0.5       # 반대 방향 명령 최대
        self.fb_int = 0.0
        self.fb_sign = 0.0
        self.fb_desired = None
        self.global_turn_break = 0.8       # [rad/s] 발행 명령 기준 선형 구간
        self.turn_ratio = 0.5              # 작은 명령의 실제/명령 비율 추정값 (시작값)
        self.turn_ratio_min = 0.25         # 추정 범위 (0.25 = 기존 4배 보상과 같음)
        self.turn_ratio_max = 1.0
        self.turn_ratio_forget = 0.98      # 0.1초마다 이전 데이터 반영 비율 (약 5초 기억)
        self.turn_ratio_delay = 0.3        # [s] 명령 -> 실제 회전 지연
        self.turn_ratio_num = 0.0
        self.turn_ratio_den = 0.0
        self.omega_history = []            # (시각, 발행한 각속도)

        self.curve_step = 0.2              # 곡률 계산용 샘플 간격

        # [수정] 가감속 제한 (0.1초당): 0.08/0.25 -> 0.05/0.12
        # 급가감속 시 Gazebo 4륜 스키드 바퀴가 미끄러져 odom이 틀어지고
        # AMCL 위치가 0.5m 이상 어긋났음 (벽이 장애물로 잡혀 회피 실패)
        self.accel_up = 0.05
        self.accel_down = 0.12
        self.previous_linear = 0.0

        # =====================================
        # 데이터 유효 시간
        # =====================================

        self.odom_timeout = 0.5
        self.local_timeout = 0.5
        self.status_timeout = 0.5

        # =====================================
        # ROS2
        # =====================================

        self.create_subscription(
            Path,
            '/global_path',
            self.on_global,
            10
        )

        self.create_subscription(
            Path,
            '/local_path',
            self.on_local,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Float32MultiArray,
            '/avoid_status',
            self.on_status,
            10
        )

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.create_timer(
            0.1,
            self.control
        )

        self.get_logger().info(
            'Pure Pursuit 시작: /odom + TF(map->odom), 회피 판단은 local_path_planner.py'
        )

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_global(self, msg):
        if msg.header.frame_id != self.map_frame:
            self.global_path = None
            self.get_logger().warning(
                'Global Path 좌표계가 map이 아님',
                throttle_duration_sec=1.0
            )
            return

        first = self.global_path is None
        self.global_path = msg

        if first:
            self.get_logger().info(
                f'Global Path 수신: {len(msg.poses)}개'
            )

    def on_local(self, msg):
        self.local_path = msg
        self.local_time = self.now()

    def on_odom(self, msg):
        # /odom 메시지의 위치를 직접 사용 (TF보다 항상 최신)
        p = msg.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_of(msg.pose.pose.orientation))
        self.odom_omega = msg.twist.twist.angular.z

        if msg.header.frame_id:
            self.odom_frame = msg.header.frame_id

        self.odom_time = self.now()

    def on_status(self, msg):
        if len(msg.data) < STATUS_SIZE:
            self.get_logger().warning(
                f'/avoid_status 길이 이상: {len(msg.data)}',
                throttle_duration_sec=2.0
            )
            return

        first = not self.status_ever
        self.status = list(msg.data)
        self.status_time = self.now()
        self.status_ever = True

        if first:
            self.get_logger().info('local_path_planner.py 회피 상태 수신 시작')

    def status_fresh(self):
        return (
            self.status is not None
            and self.status_time is not None
            and self.now() - self.status_time <= self.status_timeout
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

    def closest(self, poses, x, y):
        return min(
            range(len(poses)),
            key=lambda i: math.hypot(
                poses[i].pose.position.x - x,
                poses[i].pose.position.y - y
            )
        )

    def ahead_global_points(self, x, y, length):
        poses = self.global_path.poses
        n = len(poses)

        nearest = self.closest(poses, x, y)

        previous = poses[nearest].pose.position

        # [수정] 가장 가까운 점도 포함해서 시작
        points = [(previous.x, previous.y)]
        travelled = 0.0

        for k in range(1, n):
            i = (
                nearest + self.path_direction * k
            ) % n

            p = poses[i].pose.position

            travelled += math.hypot(
                p.x - previous.x,
                p.y - previous.y
            )

            points.append((p.x, p.y))
            previous = p

            if travelled >= length:
                break

        return points

    def local_fresh(self):
        return (
            self.local_path is not None
            and self.local_path.header.frame_id == self.map_frame
            and len(self.local_path.poses) >= 2
            and self.local_time is not None
            and self.now() - self.local_time
            <= self.local_timeout
        )

    def stop(self):
        self.pub.publish(Twist())
        self.previous_angular = 0.0
        self.previous_linear = 0.0

    def path_curvature_ahead(self, x, y, preview=None):
        # [추가] 앞쪽 curve_preview 거리 안 global path의 최대 곡률
        # curve_step 간격으로 샘플링해서 경로 점 잡음 영향 줄임
        poses = self.global_path.poses
        n = len(poses)

        nearest = self.closest(poses, x, y)
        previous = poses[nearest].pose.position

        samples = [(previous.x, previous.y)]
        step = 0.0
        total = 0.0

        for k in range(1, n):
            p = poses[
                (nearest + self.path_direction * k) % n
            ].pose.position

            d = math.hypot(
                p.x - previous.x,
                p.y - previous.y
            )

            step += d
            total += d
            previous = p

            if step >= self.curve_step:
                samples.append((p.x, p.y))
                step = 0.0

            if total >= (self.curve_preview if preview is None else preview):
                break

        headings = [
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(samples, samples[1:])
        ]

        kappa = 0.0

        for i in range(len(headings) - 2):
            kappa = max(
                kappa,
                abs(wrap(headings[i + 2] - headings[i]))
                / (2.0 * self.curve_step)
            )

        return kappa

    def align_to_path(self, x, y, yaw):
        # [추가] 경로 진행 방향을 등지고 있을 때 제자리 회전으로 방향 맞추기
        # 기준점: 가장 가까운 점에서 경로를 따라 global_lookahead_min 앞 (앞/뒤 상관없이)
        points = self.ahead_global_points(x, y, self.align_preview)
        tx, ty = points[-1]
        travelled = 0.0

        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            travelled += math.hypot(bx - ax, by - ay)

            if travelled >= self.global_lookahead_min:
                tx, ty = bx, by
                break

        error = wrap(math.atan2(ty - y, tx - x) - yaw)

        self.rotate_in_place(error, '경로 반대 방향')

    def align_to_local(self, x, y, yaw, rotation_clear):
        # [추가] LOCAL 경로가 옆/뒤쪽으로 나 있을 때 그 방향으로 제자리 회전
        # (격자 경로를 받았는데 로봇이 비스듬히 서 있는 경우 등)
        # 회전해도 되는지(rotation_clear)는 local_path_planner.py 가 판단
        # 회전했으면 True
        poses = self.local_path.poses

        if len(poses) < 2 or not rotation_clear:
            return False

        nearest = self.closest(poses, x, y)
        previous = poses[nearest].pose.position
        target = poses[-1].pose.position
        travelled = 0.0

        for i in range(nearest + 1, len(poses)):
            p = poses[i].pose.position
            travelled += math.hypot(p.x - previous.x, p.y - previous.y)
            previous = p

            if travelled >= self.local_lookahead:
                target = p
                break

        error = wrap(math.atan2(target.y - y, target.x - x) - yaw)
        self.rotate_in_place(error, '회피 경로 방향 맞춤')
        return True

    def rotate_in_place(self, error, reason):
        # [추가] 방향 오차 쪽으로 제자리 회전 (보상 포함 명령)
        cmd = Twist()
        cmd.angular.z = math.copysign(self.align_omega, error)
        self.pub.publish(cmd)

        self.previous_linear = 0.0
        self.previous_angular = cmd.angular.z

        self.get_logger().warning(
            f'{reason}: 제자리 회전 중 (방향 오차 {math.degrees(error):+.0f}°)',
            throttle_duration_sec=0.5
        )

    def is_turning(self, kappa, heading_error, path_kappa, hold=None):
        # [추가] 회전 판정: 목표점 곡률 / 앞쪽 경로 곡률 / 방향 오차 중 하나라도 크면
        # turn_hold_time 동안 회전 상태 유지
        now = self.now()

        if (
            abs(kappa) >= self.turn_kappa
            or path_kappa >= self.turn_kappa
            or abs(heading_error) >= self.turn_heading
        ):
            self.turn_until = now + (self.turn_hold_time if hold is None else hold)

        return now < self.turn_until

    def update_turn_ratio(self, speed):
        # [추가] turn_ratio_delay 전에 발행한 명령과 지금 실제 회전(/odom)으로
        # 작은 명령(선형 구간)의 실제/명령 비율을 최소제곱으로 추정
        if speed < 0.5 or not self.omega_history:
            return

        target = self.now() - self.turn_ratio_delay
        t, c = min(self.omega_history, key=lambda h: abs(h[0] - target))

        if abs(t - target) > 0.15 or not (0.1 <= abs(c) <= self.global_turn_break):
            return

        f = self.turn_ratio_forget
        self.turn_ratio_num = f * self.turn_ratio_num + c * self.odom_omega
        self.turn_ratio_den = f * self.turn_ratio_den + c * c

        if self.turn_ratio_den > 0.3:
            self.turn_ratio = max(
                self.turn_ratio_min,
                min(self.turn_ratio_max, self.turn_ratio_num / self.turn_ratio_den)
            )

    def stop_reason(self, reason):
        self.stop()

        self.get_logger().warning(
            f'정지 원인: {reason}',
            throttle_duration_sec=1.0
        )

    def lookahead(self, path, is_global, x, y, yaw):
        poses = path.poses

        if len(poses) < 2:
            return None

        nearest = self.closest(poses, x, y)

        distance = (
            self.global_lookahead
            if is_global
            else self.local_lookahead
        )

        if is_global:
            indices = [
                (nearest + self.path_direction * k)
                % len(poses)
                for k in range(1, len(poses))
            ]
        else:
            indices = range(
                nearest + 1,
                len(poses)
            )

        # 경로를 따라 너무 멀리(다른 구간까지) 찾지 않도록 제한
        travelled = 0.0
        previous = poses[nearest].pose.position

        for i in indices:
            p = poses[i].pose.position

            travelled += math.hypot(
                p.x - previous.x,
                p.y - previous.y
            )
            previous = p

            if travelled > distance + 1.5:
                break

            dx = p.x - x
            dy = p.y - y

            forward = (
                math.cos(yaw) * dx
                + math.sin(yaw) * dy
            )

            if (
                math.hypot(dx, dy) >= distance
                and forward > 0
            ):
                return p

        # Local 경로가 짧으면 마지막 전방점을 사용
        if not is_global:
            for i in range(len(poses) - 1, nearest, -1):
                p = poses[i].pose.position

                dx = p.x - x
                dy = p.y - y

                forward = (
                    math.cos(yaw) * dx
                    + math.sin(yaw) * dy
                )

                if forward > 0.05:
                    return p

        return None

    def control(self):
        now = self.now()

        # =====================================
        # 데이터 검사
        # =====================================

        if (
            self.global_path is None
            or len(self.global_path.poses) < 2
        ):
            self.stop_reason('Global Path 없음')
            return

        if self.odom_time is None:
            self.stop_reason('/odom 수신 안 됨')
            return

        odom_age = now - self.odom_time

        if odom_age > self.odom_timeout:
            self.stop_reason(
                f'/odom 오래됨 ({odom_age:.2f}s)'
            )
            return

        pose = self.robot_pose()

        if pose is None:
            self.stop_reason('로봇 위치 계산 실패 (map->odom TF 없음)')
            return

        x, y, yaw = pose

        self.fix_direction(x, y, yaw)

        # =====================================
        # local_path_planner.py 회피 상태
        # =====================================

        status = self.status if self.status_fresh() else None

        if status is None:
            if self.status_ever:
                # 실행 중이던 local_path_planner.py 가 멈춤 -> 안전하게 정지
                self.stop_reason('local_path_planner.py 회피 상태(/avoid_status) 끊김')
                return

            # 처음부터 없음 (Q1: pure_pursuit.py 단독 실행) -> global path만 추종
            if not self.global_only_logged:
                self.global_only_logged = True
                self.get_logger().info(
                    'local_path_planner.py 없이 실행: Global Path만 추종 (장애물 회피 없음)'
                )

        elif status[STATUS_OK] < 0.5:
            self.stop_reason('local_path_planner.py 대기 중 (센서/위치 확인 중)')
            return

        # [이동] 비상정지 복구: local_path_planner.py 가 정한 명령을 그대로 발행
        if status is not None and status[STATUS_RECOVERY] > 0.5:
            cmd = Twist()
            cmd.linear.x = status[STATUS_RECOVERY_V]
            cmd.angular.z = status[STATUS_RECOVERY_W]

            self.pub.publish(cmd)
            self.previous_angular = 0.0
            self.previous_linear = 0.0

            self.get_logger().warning(
                f'복구 중 (local_path_planner.py 명령): '
                f'v={cmd.linear.x:+.2f} w={cmd.angular.z:+.2f}',
                throttle_duration_sec=0.3
            )
            return

        # =====================================
        # Global / Local 전환 (local_path_planner.py 판단)
        # =====================================

        avoiding = status is not None and status[STATUS_AVOIDING] > 0.5

        if avoiding != self.avoiding:
            self.avoiding = avoiding

            if avoiding:
                self.get_logger().warning(
                    f'LOCAL 모드 (local_path_planner.py 판단, 장애물 '
                    f'{status[STATUS_BLOCKED]:.2f}m)'
                )
            else:
                self.get_logger().info('GLOBAL 모드 복귀')

        # =====================================
        # 사용할 경로 선택
        # =====================================

        if self.avoiding:
            if not self.local_fresh():
                self.stop_reason(
                    'DWA Local Path 없음/비어 있음/오래됨'
                )
                return

            path = self.local_path
            speed = self.local_speed
            is_global = False

        else:
            path = self.global_path
            speed = self.global_speed
            is_global = True

        rotation_clear = (
            status is not None
            and status[STATUS_ROTATION_CLEAR] > 0.5
        )

        # =====================================
        # Pure Pursuit
        # =====================================

        # [추가] 빠를수록 멀리 봄
        self.global_lookahead = min(
            self.global_lookahead_max,
            max(
                self.global_lookahead_min,
                self.lookahead_base
                + self.lookahead_gain * self.previous_linear
            )
        )

        self.local_lookahead = min(
            self.local_lookahead_max,
            max(
                self.local_lookahead_min,
                0.25 + 0.5 * self.previous_linear
            )
        )

        target = self.lookahead(
            path, is_global, x, y, yaw
        )

        if target is None:
            # [수정] GLOBAL에서 경로를 등지고 있으면 정지 대신 제자리 회전
            # (기존: 복구 후 경로 반대 방향을 보면 '전방 목표점 없음'으로 영구 정지)
            if is_global:
                self.align_to_path(x, y, yaw)
                return

            if self.align_to_local(x, y, yaw, rotation_clear):
                return

            self.stop_reason('전방 목표점 없음')
            return

        dx = target.x - x
        dy = target.y - y

        forward = (
            math.cos(yaw) * dx
            + math.sin(yaw) * dy
        )

        lateral = (
            -math.sin(yaw) * dx
            + math.cos(yaw) * dy
        )

        ld2 = dx * dx + dy * dy

        if forward <= 0 or ld2 < 0.0001:
            if is_global:
                self.align_to_path(x, y, yaw)
                return

            if self.align_to_local(x, y, yaw, rotation_clear):
                return

            self.stop_reason('목표점 위치 부적절')
            return

        kappa = 2.0 * lateral / ld2
        heading_error = math.atan2(lateral, forward)

        # [추가] LOCAL: 목표 방향이 많이 틀어졌으면 (주변이 비어 있을 때) 제자리 회전 먼저
        if (
            not is_global
            and abs(heading_error) > self.local_align_heading
            and rotation_clear
        ):
            self.rotate_in_place(heading_error, '회피 경로 방향 맞춤')
            return

        # [수정] 속도 결정: 회전 중이면 turn_speed, 아니면 직진 속도
        passed = False
        in_gap = False

        if is_global:
            # 앞쪽 경로 곡률도 보고 커브 들어가기 전에 미리 회전 판정
            turning = self.is_turning(
                kappa, heading_error,
                self.path_curvature_ahead(x, y, self.global_curve_preview),
                self.global_turn_hold_time
            )

            # 장애물 접근 감속 / 위치 불량 감속은 local_path_planner.py 가 계산한 상한
            global_limit = (
                status[STATUS_GLOBAL_LIMIT]
                if status is not None
                else self.global_speed
            )

            speed = max(
                self.min_global_speed,
                min(
                    self.turn_speed if turning else self.global_speed,
                    global_limit
                )
            )
        else:
            # LOCAL: 회전 중 local_turn_speed(0.3), 직선 최대 local_max_speed
            # 주변 여유 기반 감속은 local_path_planner.py 가 계산한 상한을 따름
            # [수정] 앞쪽 global path 곡률도 봄 (기존: DWA 경로 곡률만 봐서
            # DWA 경로가 잠깐 직선이면 커브 한가운데서 0.8로 가속했음)
            turning = self.is_turning(
                kappa, heading_error, self.path_curvature_ahead(x, y)
            )

            # [추가] 장애물을 이미 지남(local_path_planner.py 판단)
            # -> 회피 중이어도 커브 passed_turn_speed / 직선 local_max_speed
            # 단, 급하게 꺾어야 하면(곡률 passed_kappa 이상) 빠르게 가지 않음
            passed = (
                status[STATUS_PASSED] > 0.5
                and abs(kappa) < self.passed_kappa
            )

            # [추가] 틈 안에 들어와 정렬됨(local_path_planner.py 판단)
            # -> 커브 속도 제한 대신 local_path_planner.py 가 준 상한(gap_speed)까지
            in_gap = status[STATUS_IN_GAP] > 0.5

            if passed:
                cap = self.passed_turn_speed if turning else self.local_max_speed
                limit = status[STATUS_LIMIT_PASSED]
            else:
                cap = self.local_turn_speed if turning else self.local_max_speed
                limit = status[STATUS_LIMIT]

            # [추가] 회피 경로를 꺾어야 하는 정도(곡률)에 맞춰 미리 감속 (틈 안/지난 뒤 제외)
            # (run52~54: 0.6으로 달리면 회전 명령 1.2~2.0을 1초 넘게 줘도 실제 회전 0.1 정도.
            #  속도가 0.5 아래로 떨어지는 순간에야 1.6~2.4로 확 돌아서 과회전 + 틈 입구에서 앞바퀴 걸림)
            if not passed and not in_gap:
                k0, k1 = self.local_kappa_range
                r = min(1.0, max(0.0, (abs(kappa) - k0) / (k1 - k0)))
                cap = min(cap, cap - (cap - self.local_kappa_speed) * r)

            if in_gap:
                cap = self.local_max_speed

            speed = max(
                self.local_speed,
                min(limit, cap)
            )

        # [추가] 가감속 제한
        dv = max(
            -self.accel_down,
            min(self.accel_up, speed - self.previous_linear)
        )

        speed = self.previous_linear + dv
        self.previous_linear = speed

        omega = self.steer_sign * speed * kappa

        # [추가] GLOBAL: 방향 오차 비례 회전이 더 크면 그걸 사용
        if is_global:
            omega_heading = (
                self.steer_sign
                * self.heading_gain
                * heading_error
            )

            if abs(omega_heading) > abs(omega):
                omega = omega_heading

        # [추가] 회전 보상: 4륜 스키드 LIMO는 달리는 중 명령의 절반 정도만 돎
        # (로그: 명령 0.4~0.5 rad/s, v 0.9일 때 실제 0.21~0.26)
        # 속도 turn_gain_speed 이상에서 turn_gain배, 정지/저속에서는 1배에 가깝게
        blend = min(1.0, speed / self.turn_gain_speed)
        turn_gain = 1.0 + (self.turn_gain - 1.0) * blend

        if is_global:
            # [수정] 작은 회전은 추정 비율만큼만 보상, 선형 구간을 넘는 부분만 turn_gain 배
            self.update_turn_ratio(speed)
            gain_linear = 1.0 + (1.0 / self.turn_ratio - 1.0) * blend
            linear = self.global_turn_break / gain_linear    # 원하는 회전 기준 선형 구간
            a = abs(omega)
            boosted = (
                min(a, linear) * gain_linear
                + max(0.0, a - linear) * turn_gain
            )
            turn_gain = boosted / a if a > 1e-6 else gain_linear
            omega = math.copysign(boosted, omega)
        elif speed >= self.turn_gain_speed:
            # [수정] LOCAL 회전 보상: 4배 고정 -> 시작값 + 배율
            # (run49~52 기록, 속도 0.45~0.65: 발행 0.8 -> 실제 0.2, 1.85 -> 0.6, 2.6 -> 1.2, 3.75 -> 1.5
            #  = 실제 ~ 0.5 x (발행 - 0.6). 4배 고정이면 작은 회전은 거의 안 먹고 큰 회전은 1.7배 과회전
            #  -> 회피 시작 때 직진하다가 장애물 0.5m 앞에서 확 꺾임)
            a = abs(omega)
            low, high = self.local_gain_speeds
            r = min(1.0, max(0.0, (speed - low) / (high - low)))
            gain = self.local_gain_slow + (self.local_gain_fast - self.local_gain_slow) * r
            offset = self.local_turn_offset * min(1.0, a / self.local_offset_ramp)
            boosted = offset + gain * a

            # [추가] 실제 회전 피드백 (회피 중): 모자라면 명령을 점점 키움 (적분)
            # (run55 장애물 3번: 0.4 m/s에서 명령 1.8~2.2를 1초 줘도 실제 0.1~0.17 -> 갑자기 1.0~1.3으로 확 돎)
            sign = 1.0 if omega >= 0 else -1.0
            actual = self.odom_omega * sign             # 원하는 방향 기준 실제 회전

            if a < self.fb_min_turn or sign != self.fb_sign:
                self.fb_int = 0.0
            elif actual < a * self.fb_ratio:
                self.fb_int = min(self.fb_int_max, self.fb_int + self.fb_ki * 0.1 * (a - actual))
            elif actual > a:
                self.fb_int = max(0.0, self.fb_int - 2.0 * self.fb_ki * 0.1 * (actual - a))

            self.fb_sign = sign
            boosted += self.fb_int
            self.fb_desired = omega                      # 감쇠 판단용 (보상 전 원하는 회전)

            turn_gain = boosted / a if a > 1e-6 else gain
            omega = math.copysign(boosted, omega)
        else:
            omega *= turn_gain

        omega = max(
            -self.max_angular,
            min(self.max_angular, omega)
        )

        # [수정] 회피 중에는 각속도를 더 빨리 바꿀 수 있게
        max_change = (
            self.max_angular_change_local
            if self.avoiding
            else self.max_angular_change
        )

        delta = (
            omega - self.previous_angular
        )

        delta = max(
            -max_change,
            min(max_change, delta)
        )

        # [추가] 회피 중에는 저역 필터로 톱니 모양 흔들림 제거
        if self.avoiding:
            delta *= self.local_omega_filter

        omega = self.previous_angular + delta

        # [추가] 과회전 감쇠 (회피 중): 실제 회전이 원하는 회전보다 fb_over_band 이상 넘치면
        # 넘친 만큼 x fb_kd 를 명령에서 바로 뺌 (저역 필터 거치지 않음, 반대 방향은 최대 fb_counter 까지)
        if self.avoiding and not is_global and self.fb_desired is not None:
            excess = self.odom_omega - self.fb_desired
            # [수정] 원하는 회전과 같은 방향으로 넘칠 때만 (run57: 오른쪽으로 꺾어야 하는데 왼쪽으로 0.1 돌고 있으면
            #  이걸 '넘침'으로 보고 오른쪽 명령을 -0.5로 묶어 버려서 틈 안에서 방향을 못 고치고 걸림)
            same_dir = self.odom_omega * self.fb_desired > 0 or abs(self.fb_desired) < self.fb_min_turn
            if same_dir and self.odom_omega * excess > 0 and abs(excess) > self.fb_over_band:
                cut = self.fb_kd * (abs(excess) - self.fb_over_band)
                s_ = 1.0 if self.odom_omega > 0 else -1.0
                omega = s_ * max(-self.fb_counter, s_ * omega - cut)
                self.fb_int = 0.0

        if is_global:
            self.fb_int = 0.0

        self.fb_desired = None
        self.previous_angular = omega

        # [추가] 회전 반응 추정용 명령 기록 (최근 1초)
        now = self.now()
        self.omega_history.append((now, omega))
        self.omega_history = [h for h in self.omega_history if now - h[0] <= 1.0]

        # =====================================
        # cmd_vel 발행
        # =====================================

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega

        self.pub.publish(cmd)

        mode = 'GLOBAL' if is_global else 'LOCAL'
        blocked = status[STATUS_BLOCKED] if status is not None else float('inf')
        loc_error = status[STATUS_LOC_ERROR] if status is not None else 0.0

        self.get_logger().info(
            f'[{mode}] '
            f'v={speed:.2f} '
            f'omega={omega:+.2f}(보상x{turn_gain:.1f}, 실제 {self.odom_omega:+.2f}, 반응 {self.turn_ratio:.2f}) '
            f'회전={"Y" if turning else "N"}{" 지남" if passed else ""}{" 틈" if in_gap else ""} '
            f'ld={self.global_lookahead if is_global else self.local_lookahead:.2f} '
            f'blocked={blocked:.2f}m '
            f'위치오차={loc_error:.2f}m '
            f'odom_age={odom_age:.2f}s',
            throttle_duration_sec=0.5
        )


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

    node = SwitchingPurePursuit()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # ROS2 컨텍스트가 유효할 때만 정지 명령 발행
        if rclpy.ok():
            node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
