import sys
import os
import glob
import shutil
import subprocess
import time

# /home/hongsumin/Raceline-Optimization/global_path_maker.py 를 ws_limo_humble 에서 실행할 수 있게 한 버전
# - Raceline-Optimization 은 docker 컨테이너 안에만 있으므로 컨테이너 안에서 실행
# - 하위 스크립트/outputs 는 상대 경로라서 작업 폴더를 RACELINE_DIR 로 지정해 실행
# - ws_limo_humble 에 맵이름.yaml(+이미지)이 있으면 Raceline-Optimization/maps 로 먼저 복사
#   (매핑 후 map_saver_cli 로 ws_limo_humble 에 저장한 지도를 그대로 쓰기 위함)
# - [추가] 생성한 경로가 지도 벽에서 최소 거리(clearance) 이상 떨어지도록 확인/보정
# 사용법:
#   python3 global_path_maker.py 맵이름                     # global_path.csv 생성
#   python3 global_path_maker.py 맵이름 --publish           # 생성 후 global_path_publisher.py 실행
#   python3 global_path_maker.py 맵이름 --clearance=0.35    # 벽과의 최소 거리 지정 (기본 0.30m)

# [수정] 어느 컴퓨터에서든 쓰도록: 환경변수 LIMO_RACELINE_DIR 가 있으면 그 경로, 없으면 홈/Raceline-Optimization
def _find_raceline():
    # [추가] Raceline-Optimization 위치: 환경변수 LIMO_RACELINE_DIR -> 이 폴더 옆(../Raceline-Optimization) -> 홈
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get('LIMO_RACELINE_DIR'),
              os.path.join(os.path.dirname(here), 'Raceline-Optimization'),
              os.path.expanduser('~/Raceline-Optimization')):
        if d and os.path.isdir(d):
            return d
    return os.path.join(os.path.dirname(here), 'Raceline-Optimization')


RACELINE_DIR = _find_raceline()
WS_DIR = os.path.dirname(os.path.abspath(__file__))

# [추가] 경로 중심에서 지도 벽(점유 셀 중심)까지 최소 거리 [m]
# 차체 반폭 약 0.155m -> 0.30이면 차체 옆면과 벽 사이 약 0.12m
DEFAULT_CLEARANCE = 0.30

# [추가] main_globaltraj_f110.py 최소 곡률 최적화의 차량 폭(width_opt, params/f110.ini)
# 트랙 폭이 이보다 좁은 곳이 있으면 최적화가 "Problem not solvable"로 실패함
OPT_WIDTH = 0.9


def fix_narrow_track(track_csv, out_csv, curve_scale=0.9):
    # [추가] 최적화용 트랙 사본 생성 (원본 inputs/tracks/맵이름.csv 는 그대로 둠)
    # 1) 커브 안쪽 폭을 그 커브 반경 x curve_scale 이하로 자름
    #    (안쪽 폭이 반경보다 크면 법선끼리 교차 -> "normals are crossed" 오류.
    #     지도마다 폭이 조금씩 달라 가끔 실패했음)
    # 2) 트랙 폭이 OPT_WIDTH보다 좁은 점만 최적화가 풀리도록 폭을 살짝 늘림
    # 반환: 늘린 점 개수, 가장 좁은 폭, 커브에서 자른 점 개수
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    track = np.loadtxt(track_csv, delimiter=',', comments='#')

    # 중심선 곡률 (닫힌 경로, 주변 15점 평균으로 잡음 줄임)
    p = track[:, :2]
    d = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2.0
    dd = np.roll(p, -1, axis=0) - 2.0 * p + np.roll(p, 1, axis=0)
    kappa = (d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / np.maximum(np.hypot(d[:, 0], d[:, 1]) ** 3, 1e-9)
    kappa = uniform_filter1d(kappa, 15, mode='wrap')
    radius = 1.0 / np.maximum(np.abs(kappa), 1e-6)

    # 왼쪽으로 도는 커브(kappa>0)는 왼쪽 폭이 안쪽, 오른쪽 커브는 오른쪽 폭이 안쪽
    # (최소 OPT_WIDTH/2 + 0.01 은 남김: 그보다 좁히면 최적화가 안 풀림)
    # [추가] curve_scale=None 이면 커브 안쪽 폭을 자르지 않음
    limit = np.maximum(radius * (curve_scale if curve_scale else 1e9), OPT_WIDTH / 2.0 + 0.01)
    left_inner = kappa > 0
    clip_l = left_inner & (track[:, 3] > limit)
    clip_r = ~left_inner & (track[:, 2] > limit)
    track[clip_l, 3] = limit[clip_l]
    track[clip_r, 2] = limit[clip_r]

    total = track[:, 2] + track[:, 3]
    shift = np.maximum(0.0, (OPT_WIDTH - total) / 2.0 + 0.01)

    track[:, 2] += shift
    track[:, 3] += shift

    np.savetxt(out_csv, track, fmt='%0.4f', delimiter=',',
               header='x_m,y_m,w_tr_right_m,w_tr_left_m')

    return int((shift > 0).sum()), float(total.min()), int(clip_l.sum() + clip_r.sum())


def clip_crossing_normals(track_csv, out_csv, max_iter=30):
    # [추가] 법선이 실제로 교차하는 곳만 폭을 줄인 최적화용 트랙 생성
    # (커브 전체의 안쪽 폭을 반경 x 비율로 자르면 경로가 커브 안쪽을 못 써서 길어짐:
    #  예전 맵 기준 자르지 않음 43.8m / 반경 100% 48.1m / 90% 48.6m)
    # main_globaltraj_f110.py 와 같은 방법(params/f110.ini 의 스플라인 설정)으로 기준선을 다듬고,
    # 교차하는 법선 쌍마다 교차점 바로 앞까지만 그 쪽 폭을 줄임 -> 교차가 없어질 때까지 반복
    # 반환: 교차 없는 트랙을 만들었으면 줄인 점 개수, 실패하면 None
    import json
    import configparser
    import numpy as np
    import trajectory_planning_helpers as tph

    parser = configparser.ConfigParser()
    parser.read(os.path.join(RACELINE_DIR, 'params', 'f110.ini'))
    reg = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
    step = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))

    track = np.loadtxt(track_csv, delimiter=',', comments='#')
    changed = np.zeros(len(track), dtype=bool)
    horizon = 10

    for _ in range(max_iter):
        interp = tph.spline_approximation.spline_approximation(
            track=track, k_reg=reg['k_reg'], s_reg=reg['s_reg'],
            stepsize_prep=step['stepsize_prep'], stepsize_reg=step['stepsize_reg'], debug=False)
        closed = np.vstack((interp[:, :2], interp[0, :2]))
        normals = tph.calc_splines.calc_splines(path=closed)[3]

        n = len(interp)
        cuts = []          # (보간점 번호, 쪽(2=오른쪽, 3=왼쪽), 줄일 폭)

        for i in range(n):
            for k in range(-horizon, horizon + 1):
                if k == 0:
                    continue
                j = (i + k) % n
                a = np.column_stack((normals[i], -normals[j]))
                if abs(np.linalg.det(a)) < 1e-9:
                    continue
                li, lj = np.linalg.solve(a, interp[j, :2] - interp[i, :2])
                if -interp[i, 3] <= li <= interp[i, 2] and -interp[j, 3] <= lj <= interp[j, 2]:
                    for idx, lam in ((i, li), (j, lj)):
                        side = 2 if lam >= 0 else 3
                        cuts.append((idx, side, max(abs(lam) * 0.9, OPT_WIDTH / 2.0 + 0.01)))

        if not cuts:
            np.savetxt(out_csv, track, fmt='%0.4f', delimiter=',',
                       header='x_m,y_m,w_tr_right_m,w_tr_left_m')
            return int(changed.sum())

        # 보간점 -> 가장 가까운 원래 트랙 점(앞뒤 2점 포함)의 폭을 줄임
        for idx, side, width in cuts:
            d = np.hypot(track[:, 0] - interp[idx, 0], track[:, 1] - interp[idx, 1])
            k0 = int(np.argmin(d))
            for kk in range(k0 - 2, k0 + 3):
                kk %= len(track)
                if track[kk, side] > width:
                    track[kk, side] = width
                    changed[kk] = True

    return None


def load_wall_distance(yaml_path):
    # [추가] 지도 yaml + 이미지 -> 각 칸에서 가장 가까운 벽(점유 셀)까지 거리 [m]
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    info = {}

    with open(yaml_path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()

    image = info['image'].strip('"').strip("'")

    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(yaml_path), image)

    res = float(info['resolution'])
    origin = [float(v) for v in info['origin'].strip('[]').split(',')[:2]]
    occ_th = float(info.get('occupied_thresh', 0.65))
    negate = int(info.get('negate', 0))

    img = np.array(Image.open(image).convert('L'), dtype=float)
    img = np.flipud(img)    # 0행 = 지도 아래쪽(y 최소)

    occ_prob = img / 255.0 if negate else (255.0 - img) / 255.0
    wall = occ_prob > occ_th

    return distance_transform_edt(~wall) * res, res, origin


def wall_clearance(points, dist, res, origin):
    # [추가] 경로 점들의 벽까지 거리 (지도 밖은 0)
    import numpy as np

    h, w = dist.shape
    ix = np.floor((points[:, 0] - origin[0]) / res).astype(int)
    iy = np.floor((points[:, 1] - origin[1]) / res).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    d = np.zeros(len(points))
    d[inside] = dist[iy[inside], ix[inside]]
    return d


def push_from_walls(points, dist, res, origin, clearance, iterations=200):
    # [추가] 벽에 clearance보다 가까운 점을 벽 반대 방향(거리장 기울기)으로 밀고
    # 주변 점과 부드럽게 이어지도록 평활화. 닫힌 경로 기준
    import numpy as np

    gy, gx = np.gradient(dist)
    pts = points.copy()
    target = clearance + 0.03     # 점 사이 구간이 조금 더 가까울 수 있어서 여유

    for _ in range(iterations):
        d = wall_clearance(pts, dist, res, origin)
        low = d < target

        if not low.any():
            break

        gxv = wall_clearance(pts, gx, res, origin)
        gyv = wall_clearance(pts, gy, res, origin)
        norm = np.maximum(np.hypot(gxv, gyv), 1e-6)
        step = np.minimum(target - d, 0.05)[:, None]

        pts[low] += (np.c_[gxv, gyv] / norm[:, None] * step)[low]

        # 민 점은 그대로 두고, 주변 +-4점만 평활화해서 부드럽게 이어줌
        near = np.zeros(len(pts), dtype=bool)

        for k in range(-4, 5):
            near |= np.roll(low, k)

        near &= ~low
        smooth = 0.5 * pts + 0.25 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pts[near] = smooth[near]

    return pts


def check_and_fix_clearance(path_csv, yaml_path, clearance):
    # [추가] 생성된 경로(x,y,...)의 벽 거리 확인, 부족하면 밀어서 다시 저장
    import numpy as np

    rows = np.loadtxt(path_csv, delimiter=',', comments='#', ndmin=2)
    dist, res, origin = load_wall_distance(yaml_path)

    d = wall_clearance(rows[:, :2], dist, res, origin)
    k = int(np.argmin(d))
    print(f'벽까지 거리: 최소 {d.min():.2f}m ({rows[k, 0]:.2f}, {rows[k, 1]:.2f}), '
          f'중앙값 {np.median(d):.2f}m (기준 {clearance:.2f}m)')

    if d.min() >= clearance:
        return True

    print(f'기준보다 가까운 점 {int((d < clearance).sum())}개 -> 벽 반대쪽으로 보정')
    rows[:, :2] = push_from_walls(rows[:, :2], dist, res, origin, clearance)

    d = wall_clearance(rows[:, :2], dist, res, origin)
    k = int(np.argmin(d))
    print(f'보정 후 벽까지 거리: 최소 {d.min():.2f}m ({rows[k, 0]:.2f}, {rows[k, 1]:.2f})')

    np.savetxt(path_csv, rows, fmt='%.7f', delimiter=',')

    # 통로 폭 자체가 2*clearance보다 좁으면 끝까지 못 맞춤 (통로 가운데로 감)
    if d.min() < clearance - 0.03:
        print(f'경고: 통로가 좁아 {clearance:.2f}m를 다 지키지 못한 곳이 있습니다 '
              f'(최소 {d.min():.2f}m). 해당 위치를 RViz에서 확인하세요.')
        return False

    return True


def read_map_image(yaml_path):
    # 지도 yaml 의 image 항목 (없으면 None)
    try:
        with open(yaml_path, 'r') as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    if not any(line.strip().startswith('resolution:') for line in lines):
        return None

    for line in lines:
        if line.strip().startswith('image:'):
            return line.split(':', 1)[1].strip().strip('"').strip("'")

    return None


def same_file(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return False

    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        return fa.read() == fb.read()


def sync_map(map_name, ws_dir, maps_dir):
    # ws_dir 의 맵이름.yaml + 이미지를 maps_dir 로 복사
    # 같은 이름의 다른 지도가 이미 있으면 maps_dir/backup 으로 옮겨 둠
    # 반환: 복사했으면 True, ws_dir 에 지도가 없으면 False
    src_yaml = os.path.join(ws_dir, f'{map_name}.yaml')
    image = read_map_image(src_yaml)

    if image is None:
        return False

    src_img = image if os.path.isabs(image) else os.path.join(ws_dir, image)

    if not os.path.isfile(src_img):
        print(f'오류: {src_yaml} 의 이미지 파일이 없습니다: {src_img}')
        sys.exit(1)

    ext = os.path.splitext(src_img)[1]
    dst_yaml = os.path.join(maps_dir, f'{map_name}.yaml')
    dst_img = os.path.join(maps_dir, f'{map_name}{ext}')

    # 복사할 yaml 내용 (image 를 같은 폴더의 파일 이름으로)
    with open(src_yaml, 'r') as f:
        lines = f.read().splitlines()

    new_yaml = '\n'.join(
        f'image: {map_name}{ext}' if line.strip().startswith('image:') else line
        for line in lines
    ) + '\n'

    unchanged = (
        same_file(src_img, dst_img)
        and os.path.isfile(dst_yaml)
        and open(dst_yaml).read().strip() == new_yaml.strip()
    )

    # map_converter 는 .png 를 .pgm 보다 먼저 찾으므로 다른 확장자 이미지도 정리
    others = [
        os.path.join(maps_dir, f'{map_name}{e}')
        for e in ('.png', '.pgm')
        if e != ext
    ]

    if unchanged and not any(os.path.exists(p) for p in others):
        print(f'지도 동일: {dst_yaml} (복사 생략)')
        return True

    backup_dir = os.path.join(maps_dir, 'backup')
    stamp = time.strftime('%Y%m%d_%H%M%S')

    for path in [dst_yaml, dst_img] + others:
        if os.path.exists(path):
            os.makedirs(backup_dir, exist_ok=True)
            base, e = os.path.splitext(os.path.basename(path))
            moved = os.path.join(backup_dir, f'{base}_{stamp}{e}')
            n = 1

            while os.path.exists(moved):
                moved = os.path.join(backup_dir, f'{base}_{stamp}_{n}{e}')
                n += 1

            shutil.move(path, moved)
            print(f'기존 파일 백업: {path} -> {moved}')

    shutil.copy2(src_img, dst_img)

    with open(dst_yaml, 'w') as f:
        f.write(new_yaml)

    print(f'지도 복사: {src_yaml} -> {dst_yaml}')
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    publish = '--publish' in sys.argv[1:]
    clearance = DEFAULT_CLEARANCE

    for a in sys.argv[1:]:
        if a.startswith('--clearance='):
            clearance = float(a.split('=', 1)[1])

    if len(args) < 1:
        print('사용법: python3 global_path_maker.py 맵이름 [--publish] [--clearance=0.30]')
        sys.exit(1)

    if not os.path.isdir(RACELINE_DIR):
        print(f'오류: {RACELINE_DIR} 가 없습니다. docker 컨테이너 안에서 실행하세요.')
        sys.exit(1)

    map_name = args[0]
    maps_dir = os.path.join(RACELINE_DIR, 'maps')

    print(f'\n[0/3] 지도 확인: {map_name}')

    if not sync_map(map_name, WS_DIR, maps_dir):
        print(f'{WS_DIR} 에 {map_name}.yaml 없음 -> {maps_dir} 의 지도 사용')

    print(f'\n[1/3] Map Converter 실행: {map_name}')
    subprocess.run(
        ['python3', 'map_converter.py', map_name],
        cwd=RACELINE_DIR,
        check=True
    )

    # [추가] 트랙이 최적화 차량 폭(0.9m)보다 좁은 곳만 살짝 넓힌 사본으로 최적화
    # (기존: 좁은 곳이 하나라도 있으면 최적화 실패)
    track_csv = os.path.join(RACELINE_DIR, 'inputs', 'tracks', f'{map_name}.csv')
    opt_csv = os.path.join(RACELINE_DIR, 'inputs', 'tracks', f'{map_name}_maker.csv')
    print(f'\n[2/3] Raceline Optimization 실행: {map_name}')
    start = time.time()

    # [추가] "normals are crossed" 로 실패하면 커브 안쪽 폭을 더 줄여서 다시 시도
    # (그림 창은 띄우지 않음: 창이 뜨면 닫을 때까지 멈춰서 재시도가 안 됨)
    env = dict(os.environ, MPLBACKEND='Agg')

    # [수정] 먼저 커브 안쪽 폭을 자르지 않고(None) 시도 -> 실패할 때만 자름
    # (항상 0.9로 자르면 경로가 커브 바깥으로 돌아서 43.9m -> 49.0m로 길어졌음)
    # [수정] 순서: 자르지 않음 -> 교차하는 곳만 줄임('cross') -> 커브 전체 비율로 자름
    for curve_scale in (None, 'cross', 1.0, 0.95, 0.9, 0.7, 0.5):
        if curve_scale == 'cross':
            fix_narrow_track(track_csv, opt_csv, None)
            n_cross = clip_crossing_normals(opt_csv, opt_csv)
            if n_cross is None:
                print('최적화용 트랙: 교차하는 곳만 줄이기 실패 -> 다음 방법')
                continue
            print(f'최적화용 트랙: 법선이 교차하는 곳만 폭을 줄임 ({n_cross}개 점)')
        else:
            n_fixed, narrowest, n_clip = fix_narrow_track(track_csv, opt_csv, curve_scale)

            print('최적화용 트랙: '
                  + ('커브 안쪽 폭 그대로' if curve_scale is None else f'커브 안쪽 폭을 반경의 {curve_scale:.0%} 이하로 {n_clip}개 점 조정')
                  + (f', 최소 폭 {narrowest:.2f}m < {OPT_WIDTH}m 라 좁은 점 {n_fixed}개 보정' if n_fixed else ''))

        result = subprocess.run(
            ['python3', 'main_globaltraj_f110.py', '--map_name', map_name, '--map_path', opt_csv],
            cwd=RACELINE_DIR,
            check=False,
            env=env,
            capture_output=True,
            text=True
        )
        out = (result.stdout or '') + (result.stderr or '')
        print(out[-1500:])

        if 'normals are crossed' not in out:
            break

        print('-> 법선 교차로 실패: 커브 안쪽 폭을 더 줄여서 다시 시도')

    print('\n[3/3] Global Path 복사')

    # [수정] 이번에 새로 만든 결과만 사용
    # (기존: 최적화가 실패해도 예전 결과 중 최신 파일을 복사 -> 다른 지도의 경로가 쓰였음)
    files = [
        f for f in glob.glob(
            os.path.join(RACELINE_DIR, 'outputs', map_name, 'traj_race_cl-*.csv')
        )
        if os.path.getmtime(f) >= start - 1.0
    ]

    if not files:
        print('오류: Raceline Optimization이 실패해서 새 경로가 만들어지지 않았습니다.')
        print('      global_path.csv 는 바꾸지 않았습니다. 위 에러 메시지를 확인하세요.')
        sys.exit(1)

    latest_file = max(files, key=os.path.getmtime)
    destination = os.path.join(WS_DIR, 'global_path.csv')

    # [추가] 벽과의 거리 확인/보정 후 global_path.csv 로 저장
    tmp = destination + '.tmp'
    shutil.copy2(latest_file, tmp)

    yaml_path = os.path.join(maps_dir, f'{map_name}.yaml')
    ok = check_and_fix_clearance(tmp, yaml_path, clearance)
    os.replace(tmp, destination)

    print('\n완료!' if ok else '\n완료 (경고 확인 필요)')
    print(f'생성 파일: {latest_file}')
    print(f'Global Path: {destination}')

    if publish:
        print('\n[추가] global_path_publisher 실행 (Ctrl+C로 종료)')
        subprocess.run(['python3', os.path.join(WS_DIR, 'global_path_publisher.py')])


if __name__ == '__main__':
    main()
