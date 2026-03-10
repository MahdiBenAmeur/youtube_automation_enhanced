"""
2D Top-Down Driving Scene — YouTube Shorts (9:16)
Requires: pip install pygame
Controls: W/↑ Forward  S/↓ Backward  A/← Left  D/→ Right   R = new track
"""

import pygame
import math
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 405, 720
FPS          = 60
ROAD_WIDTH   = 76           # full road width in px
BARRIER_W    = 6            # thickness of each barrier strip
DASH_LEN     = 22           # length of one colour dash on barrier
MAX_ATTEMPTS = 40           # track-generation retries

# Colours
C_TERRAIN   = (55, 115, 48)
C_GRAVEL    = (158, 145, 125)
C_GRAVEL_D  = (142, 130, 110)
C_RED       = (205,  35,  35)
C_WHITE     = (235, 235, 235)
C_TRUNK     = ( 95,  65,  38)
C_CANOPY    = ( 32,  98,  32)
C_CANOPY_H  = ( 55, 130,  55)
C_CAR       = (220,  55,  55)
C_GLASS     = (155, 205, 255)
C_TYRE      = ( 25,  25,  25)
C_FINISH    = (255, 215,   0)

# ──────────────────────────────────────────────────────
#  SPLINE HELPERS  (Catmull-Rom)
# ──────────────────────────────────────────────────────
def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t; t3 = t2 * t
    x = 0.5*((2*p1[0]) + (-p0[0]+p2[0])*t +
              (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
              (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
    y = 0.5*((2*p1[1]) + (-p0[1]+p2[1])*t +
              (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
              (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    return (x, y)


def spline_points(waypoints: List[Tuple], steps_per_seg=32) -> List[Tuple]:
    """Catmull-Rom through all waypoints, with phantom endpoints."""
    if len(waypoints) < 2:
        return list(waypoints)
    pts = [
        (2*waypoints[0][0]-waypoints[1][0], 2*waypoints[0][1]-waypoints[1][1]),
        *waypoints,
        (2*waypoints[-1][0]-waypoints[-2][0], 2*waypoints[-1][1]-waypoints[-2][1]),
    ]
    result = []
    for i in range(1, len(pts) - 2):
        for s in range(steps_per_seg):
            result.append(catmull_rom(pts[i-1], pts[i], pts[i+1], pts[i+2], s/steps_per_seg))
    result.append(waypoints[-1])
    return result


# ──────────────────────────────────────────────────────
#  TRACK GENERATION  (waypoints → spline)
# ──────────────────────────────────────────────────────
def generate_waypoints() -> Optional[List[Tuple]]:
    margin  = ROAD_WIDTH + 12
    start_x = random.uniform(SCREEN_W*0.25, SCREEN_W*0.75)
    start_y = SCREEN_H - 55
    end_x   = random.uniform(SCREEN_W*0.25, SCREEN_W*0.75)
    end_y   = 65

    waypoints = [(start_x, start_y)]
    cx, cy   = start_x, start_y
    heading  = random.uniform(-0.25, 0.25)
    num_wp   = random.randint(10, 18)

    for step in range(num_wp):
        progress    = step / num_wp
        dx_e        = end_x - cx;  dy_e = end_y - cy
        dist_e      = math.hypot(dx_e, dy_e) or 1
        ideal_angle = math.atan2(dx_e, -dy_e)

        if dist_e < 95:
            waypoints.append((end_x, end_y))
            return waypoints if len(waypoints) >= 4 else None

        bias      = progress ** 1.4
        max_turn  = math.pi * 0.42 * (1 - bias * 0.5)
        delta     = random.uniform(-max_turn, max_turn)
        err       = ideal_angle - heading
        while err >  math.pi: err -= 2*math.pi
        while err < -math.pi: err += 2*math.pi
        delta    += err * bias * 0.55
        heading  += delta

        seg_len = random.uniform(62, 115)
        nx = cx + math.sin(heading) * seg_len
        ny = cy - math.cos(heading) * seg_len

        if nx < margin or nx > SCREEN_W - margin:
            heading = ideal_angle
            nx = max(margin, min(SCREEN_W - margin, nx))
        if ny > start_y + 20:
            continue

        too_close = any(math.hypot(nx-wx, ny-wy) < ROAD_WIDTH*1.15
                        for wx, wy in waypoints[:-3])
        if too_close:
            continue

        waypoints.append((nx, ny))
        cx, cy = nx, ny

    waypoints.append((end_x, end_y))
    return waypoints if len(waypoints) >= 4 else None


def get_track_spline() -> List[Tuple]:
    for attempt in range(MAX_ATTEMPTS):
        wps = generate_waypoints()
        if wps and len(wps) >= 4:
            pts = spline_points(wps, steps_per_seg=32)
            if len(pts) >= 8:
                print(f"Track ready  attempt={attempt+1}  "
                      f"wp={len(wps)}  pts={len(pts)}")
                return pts
    print("Fallback track")
    return spline_points([
        (SCREEN_W//2, SCREEN_H-55),
        (SCREEN_W//3, SCREEN_H*0.75),
        (SCREEN_W*2//3, SCREEN_H*0.50),
        (SCREEN_W//3, SCREEN_H*0.25),
        (SCREEN_W//2, 65),
    ], steps_per_seg=48)


# ──────────────────────────────────────────────────────
#  ROAD EDGE GEOMETRY
# ──────────────────────────────────────────────────────
def smooth_normals(pts):
    """Per-point inward-averaged normals for a polyline."""
    n = len(pts)
    normals = []
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
        elif i == n-1:
            dx, dy = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
        else:
            dx, dy = pts[i+1][0]-pts[i-1][0], pts[i+1][1]-pts[i-1][1]
        ln = math.hypot(dx, dy) or 1
        normals.append((-dy/ln, dx/ln))
    return normals


def build_road_edges(pts: List[Tuple], half=ROAD_WIDTH/2):
    norms = smooth_normals(pts)
    left  = [(pts[i][0]+norms[i][0]*half, pts[i][1]+norms[i][1]*half) for i in range(len(pts))]
    right = [(pts[i][0]-norms[i][0]*half, pts[i][1]-norms[i][1]*half) for i in range(len(pts))]
    return left, right


# ──────────────────────────────────────────────────────
#  CONTINUOUS BARRIER  (global distance → no seam gaps)
# ──────────────────────────────────────────────────────
def draw_continuous_barrier(surface, edge_pts: List[Tuple], inward_sign: float):
    """
    Walks the full edge polyline once, accumulating distance.
    inward_sign: +1 if edge is the left edge (inward = right), -1 for right edge.
    Draws a solid strip BARRIER_W wide with alternating R/W dashes.
    """
    n = len(edge_pts)
    if n < 2:
        return

    # Pre-compute inner offset points
    norms = smooth_normals(edge_pts)
    inner = [
        (edge_pts[i][0] + norms[i][0] * BARRIER_W * inward_sign,
         edge_pts[i][1] + norms[i][1] * BARRIER_W * inward_sign)
        for i in range(n)
    ]

    dist = 0.0
    for i in range(n - 1):
        ox0, oy0 = edge_pts[i];   ox1, oy1 = edge_pts[i+1]
        ix0, iy0 = inner[i];      ix1, iy1 = inner[i+1]
        sdx = ox1-ox0; sdy = oy1-oy0
        seg_len = math.hypot(sdx, sdy)
        if seg_len < 0.5:
            continue

        walked = 0.0
        while walked < seg_len:
            phase     = dist % (DASH_LEN * 2)
            colour    = C_RED if phase < DASH_LEN else C_WHITE
            remain    = DASH_LEN - (phase % DASH_LEN)
            step      = min(remain, seg_len - walked)

            t0 = walked / seg_len
            t1 = (walked + step) / seg_len

            A = (ox0+sdx*t0,  oy0+sdy*t0)
            B = (ox0+sdx*t1,  oy0+sdy*t1)
            C = (ix0+(ix1-ix0)*t1, iy0+(iy1-iy0)*t1)
            D = (ix0+(ix1-ix0)*t0, iy0+(iy1-iy0)*t0)

            pygame.draw.polygon(surface, colour, [A, B, C, D])

            walked += step
            dist   += step


# ──────────────────────────────────────────────────────
#  TREES
# ──────────────────────────────────────────────────────
def place_trees(left_edge, right_edge, count=150):
    CELL = 10
    road_cells = set()
    for ex, ey in left_edge + right_edge:
        r = int(ROAD_WIDTH // CELL) + 2
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                road_cells.add((int(ex//CELL)+dx, int(ey//CELL)+dy))

    trees = []
    attempts = 0
    while len(trees) < count and attempts < count * 25:
        attempts += 1
        tx = random.randint(8, SCREEN_W - 8)
        ty = random.randint(8, SCREEN_H - 8)
        if (int(tx//CELL), int(ty//CELL)) in road_cells:
            continue
        r = random.randint(8, 19)
        if all(math.hypot(tx-ox, ty-oy) > r+or_+4 for ox, oy, or_ in trees):
            trees.append((tx, ty, r))
    return trees


def draw_trees(surface, trees):
    for tx, ty, r in trees:
        pygame.draw.ellipse(surface, (35, 82, 28), (int(tx-r), int(ty+r//2), r*2, r//2+3))
        pygame.draw.rect(surface,    C_TRUNK,       (int(tx-2), int(ty), 4, r//2+2))
        pygame.draw.circle(surface,  C_CANOPY,      (int(tx), int(ty)),    r)
        pygame.draw.circle(surface,  C_CANOPY_H,    (int(tx-r//3), int(ty-r//3)), max(2, r//2))


# ──────────────────────────────────────────────────────
#  CAR PHYSICS  (modular — swap PlayerController for AI)
# ──────────────────────────────────────────────────────
@dataclass
class CarState:
    x: float; y: float
    angle:    float = 0.0
    speed:    float = 0.0
    steer:    float = 0.0
    drift_vx: float = 0.0
    drift_vy: float = 0.0


class PlayerController:
    """Human keyboard controller. Replace with AIController for autonomous driving."""
    def get_inputs(self, keys) -> dict:
        return {
            "forward":  bool(keys[pygame.K_w] or keys[pygame.K_UP]),
            "backward": bool(keys[pygame.K_s] or keys[pygame.K_DOWN]),
            "left":     bool(keys[pygame.K_a] or keys[pygame.K_LEFT]),
            "right":    bool(keys[pygame.K_d] or keys[pygame.K_RIGHT]),
        }


class CarPhysics:
    ACCEL       =  0.18
    BRAKE       =  0.24
    REVERSE_A   =  0.10
    MAX_FWD     =  4.0
    MAX_REV     = -1.6
    STEER_RATE  =  0.055
    STEER_DECAY =  0.80
    FRICTION    =  0.965
    DRIFT_GRIP  =  0.82
    DRIFT_BLEND =  0.14

    def update(self, s: CarState, inp: dict, dt: float) -> CarState:
        if inp["left"]:    s.steer -= self.STEER_RATE
        elif inp["right"]: s.steer += self.STEER_RATE
        else:               s.steer *= self.STEER_DECAY
        s.steer = max(-1.0, min(1.0, s.steer))

        if inp["forward"]:
            s.speed += self.ACCEL
        elif inp["backward"]:
            s.speed -= self.BRAKE if s.speed > 0.05 else -self.REVERSE_A
        else:
            s.speed *= self.FRICTION
        s.speed = max(self.MAX_REV, min(self.MAX_FWD, s.speed))

        s.angle += 0.038 * abs(s.speed) * s.steer

        tvx =  math.sin(s.angle) * s.speed
        tvy = -math.cos(s.angle) * s.speed
        s.drift_vx += (tvx - s.drift_vx) * self.DRIFT_GRIP
        s.drift_vy += (tvy - s.drift_vy) * self.DRIFT_GRIP

        if abs(s.speed) > 0.4:
            da = math.atan2(s.drift_vx, -s.drift_vy) - s.angle
            while da >  math.pi: da -= 2*math.pi
            while da < -math.pi: da += 2*math.pi
            s.angle += da * self.DRIFT_BLEND * abs(s.steer)

        s.x += s.drift_vx
        s.y += s.drift_vy
        return s


# ──────────────────────────────────────────────────────
#  COLLISION
# ──────────────────────────────────────────────────────
def point_on_road(px, py, pts: List[Tuple]) -> bool:
    threshold = (ROAD_WIDTH / 2 - 6) ** 2
    for i in range(0, len(pts)-1, 3):
        ax, ay = pts[i]; bx, by = pts[i+1]
        dx, dy = bx-ax, by-ay
        ln2 = dx*dx + dy*dy
        if ln2 < 1: continue
        t  = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / ln2))
        cx = ax+t*dx; cy = ay+t*dy
        if (px-cx)**2 + (py-cy)**2 <= threshold:
            return True
    return False


def clamp_to_road(s: CarState, pts, px, py) -> CarState:
    if not point_on_road(s.x, s.y, pts):
        s.x = px; s.y = py
        s.speed   *= -0.25
        s.drift_vx *= 0.15
        s.drift_vy *= 0.15
    return s


# ──────────────────────────────────────────────────────
#  CAR DRAWING
# ──────────────────────────────────────────────────────
def draw_car(surface, sx, sy, angle):
    CL, CW = 30, 15
    # Shadow
    sh_s = pygame.Surface((CL+6, CW+6), pygame.SRCALPHA)
    pygame.draw.rect(sh_s, (0,0,0,55), (0,0,CL+6,CW+6), border_radius=5)
    rot_sh = pygame.transform.rotate(sh_s, -math.degrees(angle)+90)
    surface.blit(rot_sh, rot_sh.get_rect(center=(int(sx)+3, int(sy)+3)))

    surf = pygame.Surface((CL, CW), pygame.SRCALPHA)
    pygame.draw.rect(surf, C_CAR,          (0, 0, CL, CW),        border_radius=4)
    pygame.draw.rect(surf, C_GLASS,        (CL//2+2, 2, CL//3, CW-4), border_radius=2)
    pygame.draw.rect(surf, (120,170,210),  (3, 2, CL//5, CW-4),   border_radius=2)
    for wx, wy, wl, wh in [(2,0,7,3),(2,CW-3,7,3),(CL-9,0,7,3),(CL-9,CW-3,7,3)]:
        pygame.draw.rect(surf, C_TYRE, (wx, wy, wl, wh), border_radius=1)

    rot  = pygame.transform.rotate(surf, -math.degrees(angle)+90)
    surface.blit(rot, rot.get_rect(center=(int(sx), int(sy))))


# ──────────────────────────────────────────────────────
#  START / FINISH MARKINGS
# ──────────────────────────────────────────────────────
def draw_start_line(surface, left, right):
    lx, ly = left[0]; rx, ry = right[0]
    pygame.draw.line(surface, C_WHITE, (int(lx), int(ly)), (int(rx), int(ry)), 5)


def draw_finish_line(surface, left, right):
    lx, ly = left[-1]; rx, ry = right[-1]
    dx, dy  = rx-lx, ry-ly
    steps   = max(1, int(math.hypot(dx,dy)/8))
    for i in range(steps):
        t0 = i/steps; t1 = (i+1)/steps
        c = C_FINISH if i%2==0 else (10,10,10)
        pygame.draw.line(surface, c,
                         (int(lx+dx*t0), int(ly+dy*t0)),
                         (int(lx+dx*t1), int(ly+dy*t1)), 8)


# ──────────────────────────────────────────────────────
#  WORLD SURFACE  (pre-rendered once per track)
# ──────────────────────────────────────────────────────
def build_world(pts, left_edge, right_edge, trees):
    all_x = [p[0] for p in pts]; all_y = [p[1] for p in pts]
    pad   = ROAD_WIDTH * 4
    wx0   = min(0, min(all_x)-pad); wy0 = min(0, min(all_y)-pad)
    wx1   = max(SCREEN_W, max(all_x)+pad)
    wy1   = max(SCREEN_H, max(all_y)+pad)
    W, H  = int(wx1-wx0), int(wy1-wy0)

    world = pygame.Surface((W, H))
    world.fill(C_TERRAIN)

    def sh(x, y): return (x-wx0, y-wy0)
    def sh_list(lst): return [sh(*p) for p in lst]

    s_pts  = sh_list(pts)
    s_left = sh_list(left_edge)
    s_right= sh_list(right_edge)
    s_tree = [(tx-wx0, ty-wy0, r) for tx,ty,r in trees]

    # Trees first (behind road)
    draw_trees(world, s_tree)

    # Road surface quads
    for i in range(len(s_pts)-1):
        poly   = [s_left[i], s_left[i+1], s_right[i+1], s_right[i]]
        colour = C_GRAVEL if i%2==0 else C_GRAVEL_D
        pygame.draw.polygon(world, colour, poly)

    # Centre dashes
    CDASH = 30; dist = 0.0
    for i in range(len(s_pts)-1):
        dx = s_pts[i+1][0]-s_pts[i][0]; dy = s_pts[i+1][1]-s_pts[i][1]
        slen = math.hypot(dx, dy)
        if slen < 0.5: continue
        walked = 0.0
        while walked < slen:
            t0   = walked/slen
            step = min(CDASH*0.5, slen-walked)
            t1   = (walked+step)/slen
            if int(dist/CDASH)%2 == 0:
                ax = s_pts[i][0]+dx*t0; ay = s_pts[i][1]+dy*t0
                bx = s_pts[i][0]+dx*t1; by = s_pts[i][1]+dy*t1
                pygame.draw.line(world, (192,180,158), (int(ax),int(ay)),(int(bx),int(by)), 2)
            walked += step; dist += step

    # Continuous barriers — left edge inward sign = +1, right edge = -1
    draw_continuous_barrier(world, s_left,  +1)
    draw_continuous_barrier(world, s_right, -1)

    # Start / Finish
    draw_start_line(world,  s_left, s_right)
    draw_finish_line(world, s_left, s_right)

    return world, wx0, wy0


# ──────────────────────────────────────────────────────
#  CAMERA
# ──────────────────────────────────────────────────────
class Camera:
    SMOOTH = 0.13
    def __init__(self, car_x, car_y):
        self.ox = car_x - SCREEN_W/2
        self.oy = car_y - SCREEN_H*0.62

    def update(self, car_x, car_y):
        self.ox += (car_x - SCREEN_W/2   - self.ox) * self.SMOOTH
        self.oy += (car_y - SCREEN_H*0.62 - self.oy) * self.SMOOTH


# ──────────────────────────────────────────────────────
#  GAME SETUP
# ──────────────────────────────────────────────────────
def new_game():
    pts           = get_track_spline()
    left, right   = build_road_edges(pts)
    trees         = place_trees(left, right, count=160)
    world, wx0, wy0 = build_world(pts, left, right, trees)

    # Car at the very first spline point, facing along the first segment
    sx, sy = pts[0]
    if len(pts) > 1:
        tx2 = pts[1][0]-pts[0][0]; ty2 = pts[1][1]-pts[0][1]
        init_angle = math.atan2(tx2, -ty2)
    else:
        init_angle = 0.0

    car    = CarState(x=sx, y=sy, angle=init_angle)
    camera = Camera(sx, sy)
    return pts, world, wx0, wy0, car, camera


# ──────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Top-Down Driver — YouTube Shorts")
    clock  = pygame.time.Clock()

    pts, world, wx0, wy0, car, camera = new_game()
    controller = PlayerController()
    physics    = CarPhysics()
    font       = pygame.font.SysFont("monospace", 14, bold=True)
    running    = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:    running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r:
                    pts, world, wx0, wy0, car, camera = new_game()

        keys   = pygame.key.get_pressed()
        inputs = controller.get_inputs(keys)
        px, py = car.x, car.y
        car    = physics.update(car, inputs, 1/FPS)
        car    = clamp_to_road(car, pts, px, py)
        camera.update(car.x, car.y)

        # Draw world
        screen.blit(world, (int(-(camera.ox+wx0)), int(-(camera.oy+wy0))))

        # Draw car in screen space
        draw_car(screen, car.x-camera.ox, car.y-camera.oy, car.angle)

        # HUD
        kmh  = abs(car.speed)*15
        shad = font.render(f"{kmh:.0f} km/h", True, (0,0,0))
        txt  = font.render(f"{kmh:.0f} km/h", True, (255,255,255))
        screen.blit(shad, (11,11)); screen.blit(txt, (10,10))
        hint = font.render("R = new track", True, (180,180,180))
        screen.blit(hint, (SCREEN_W-hint.get_width()-8, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()