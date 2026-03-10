"""
Arena Royale – YouTube Shorts Generator
========================================
Pure Python pipeline: simulation → frames (OpenCV) → audio (numpy/scipy) → mp4 (ffmpeg)

Usage:
    python arena_royale.py [--balls 8] [--output arena.mp4] [--preview]

Modes:
    (default)   Headless – simulate and encode to mp4, no window opened.
    --preview   Live window – watch the simulation in real time (no file written).
                Press Q to quit early.

Dependencies:
    pip install opencv-python numpy scipy tqdm
    ffmpeg must be on PATH
"""

import argparse
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────── CONFIG ───────────────────────────
CFG = {
    # Video
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "max_duration": 160,
    "slowmo_final": True,
    "slowmo_duration": 1.5,

    # Arena
    "arena_pad_x": 60,
    "arena_top": 500,
    "arena_bottom_pad": 500,

    # Balls
    "n_balls_min": 6,
    "n_balls_max": 12,
    "radius_min": 35,
    "radius_max": 42,
    "speed_min": 400.0,
    "speed_max": 800.0,
    "speed_absolute_min": 400.5,
    "speed_absolute_max": 2200.0,
    "restitution_wall": 0.92,
    "restitution_ball": 0.92,
    "hp": 150,

    # Bullets
    "fire_cooldown_min": 1,
    "fire_cooldown_max": 1.4,
    "bullet_damage_min": 8,
    "bullet_damage_max": 14,
    "bullet_speed": 15.0,
    "bullet_radius": 5,
    "bullet_lifetime": 90,
    "max_bullets": 200,
    "knockback": 2.5,

    # Power-ups
    "powerup_spawn_interval_min": 1.8,
    "powerup_spawn_interval_max": 3.5,
    "powerup_max": 3,
    "powerup_radius": 30,

    # Physics
    "substeps": 3,
    "jitter_interval": 1.2,
    "jitter_strength": 0.8,
    "corner_escape_time": 1.5,

    # Audio volumes (0-1)
    "vol_bounce": 0.4,
    "vol_collision": 0.5,
    "vol_hit": 0.35,
    "vol_powerup": 0.6,
    "vol_destroy": 0.7,
}
GF = 1.3

# Hard cap on ball radius — prevents balls from growing too large to move in the arena
# Arena inner width ≈ 1080 - 2*60 = 960 px. Cap at ~12% of that so 2 balls always fit side-by-side.
MAX_BALL_RADIUS = 115

# ─────────────────────────── COLORS ───────────────────────────
BALL_PALETTE = [
    (220,  60,  60),
    ( 60, 160, 220),
    ( 60, 210,  80),
    (230, 180,  30),
    (180,  60, 220),
    (240, 120,  30),
    ( 60, 210, 200),
    (240,  80, 160),
    (130, 220,  60),
    (220, 200,  60),
    (100, 100, 240),
    (240, 140, 140),
]

# ── NEW POWER-UP DEFINITIONS ──────────────────────────────────
POWERUP_TYPES = {
    "BH": {"color": (60,  20, 100), "label": "BLACK\nHOLE",   "duration": 3.0},  # Black Hole
    "MS": {"color": (255, 200,  40), "label": "MULTI\nSHOT",  "duration": 4.0},  # Multishot Burst
    "LB": {"color": ( 80, 240, 255), "label": "LASER\nBEAM",  "duration": 4.0},  # Laser Beam
    "CS": {"color": (200, 255, 140), "label": "CLONE\nSPLIT", "duration": 0},    # Clone Split (instant)
    "GM": {"color": (255, 140,  40), "label": "GIANT\nMODE",  "duration": 5.0},  # Giant Mode
    "PM": {"color": (160, 160, 255), "label": "PHASE\nMODE",  "duration": 4.0},  # Phase Mode
}

BG_COLOR      = (18, 18, 24)
WALL_COLOR    = (80, 90, 110)
HUD_BG        = BG_COLOR
TEXT_COLOR    = (220, 220, 230)
WINNER_COLOR  = (255, 215,   0)

# ─────────────────────────── DATA CLASSES ───────────────────────────
@dataclass
class Ball:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    base_radius: float      # original radius before Giant Mode
    mass: float
    color: Tuple[int,int,int]
    hp: float = 100.0
    max_hp: float = 100.0
    fire_cooldown: float = 0.0
    fire_rate: float = 1.0
    bullet_damage: float = 10.0
    active: bool = True
    flash_frames: int = 0
    buffs: dict = field(default_factory=dict)
    corner_frames: int = 0
    aim_jitter: float = 0.0
    bullet_scale: float = 1.0
    # Multishot state
    multishot_bullets_left: int = 0
    # Laser beam state
    laser_charge_frames: int = 0
    laser_active_frames: int = 0
    laser_dx: float = 0.0
    laser_dy: float = 0.0
    laser_cooldown: float = 0.0
    # Phase mode
    is_phased: bool = False
    # Giant mode bonus HP already applied
    giant_hp_bonus: float = 0.0

@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner_id: int
    color: Tuple[int,int,int]
    damage: float
    radius: float
    lifetime: int
    pierce: bool = False
    hits: int = 0

@dataclass
class PowerUp:
    x: float
    y: float
    kind: str
    frame_born: int
    active: bool = True

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int,int,int]
    life: int
    max_life: int
    radius: float

@dataclass
class BlackHole:
    """An active black-hole vortex in the arena."""
    x: float
    y: float
    frames_left: int
    max_frames: int
    strength: float = 600.0   # pull acceleration in px/s²

@dataclass
class LaserBeam:
    """A rendered laser beam owned by a ball."""
    owner_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    frames_left: int
    color: Tuple[int,int,int]

@dataclass
class AudioEvent:
    frame: int
    kind: str
    magnitude: float = 1.0

# ─────────────────────────── AUDIO SYNTHESIS ───────────────────────────
SAMPLE_RATE = 44100

def _fade(sig, attack=0.002, release=0.05):
    n = len(sig)
    a = int(attack * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    sig = sig.copy()
    if a > 0:
        sig[:a] *= np.linspace(0, 1, a)
    if r > 0 and r < n:
        sig[-r:] *= np.linspace(1, 0, r)
    return sig

def synth_bounce(mag=1.0):
    dur = 0.08
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freq = 280 + mag * 200
    sig = np.sin(2 * np.pi * freq * t) * np.exp(-t * 50)
    noise = np.random.randn(len(t)) * 0.15
    sig = (sig + noise) * np.exp(-t * 40)
    return _fade(sig * 0.6)

def synth_collision(mag=1.0):
    dur = 0.12
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freq = 140 + mag * 80
    sig = np.sin(2 * np.pi * freq * t) * np.exp(-t * 35)
    noise = np.random.randn(len(t)) * 0.3
    combined = (sig * 0.7 + noise * 0.3) * np.exp(-t * 30)
    return _fade(combined * 0.8)

def synth_hit(mag=1.0):
    dur = 0.06
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freq = 600 + mag * 400
    sig = np.sin(2 * np.pi * freq * t) * np.exp(-t * 80)
    return _fade(sig * 0.5)

def synth_powerup():
    dur = 0.3
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freqs = [440, 554, 659, 880]
    sig = np.zeros(len(t))
    step = len(t) // len(freqs)
    for i, f in enumerate(freqs):
        s = i * step
        e = min(s + step, len(t))
        tt = t[s:e] - t[s]
        chunk = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 20)
        sig[s:e] += chunk
    return _fade(sig * 0.7, release=0.08)

def synth_destroy():
    dur = 0.35
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    low   = np.sin(2 * np.pi * 60  * t) * np.exp(-t * 12)
    mid   = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 25)
    noise = np.random.randn(len(t)) * np.exp(-t * 15)
    sig = (low * 0.6 + mid * 0.4 + noise * 0.5)
    return _fade(sig * 0.9, release=0.1)

SYNTH_FN = {
    "bounce":    synth_bounce,
    "collision": synth_collision,
    "hit":       synth_hit,
    "powerup":   synth_powerup,
    "destroy":   synth_destroy,
}
VOL_KEY = {
    "bounce":    "vol_bounce",
    "collision": "vol_collision",
    "hit":       "vol_hit",
    "powerup":   "vol_powerup",
    "destroy":   "vol_destroy",
}

def build_audio(events: List[AudioEvent], total_frames: int, fps: int) -> np.ndarray:
    total_samples = int(total_frames / fps * SAMPLE_RATE) + SAMPLE_RATE
    mix = np.zeros(total_samples, dtype=np.float32)
    for ev in events:
        fn = SYNTH_FN.get(ev.kind)
        if fn is None:
            continue
        if ev.kind in ("powerup", "destroy"):
            chunk = fn()
        else:
            chunk = fn(np.clip(ev.magnitude, 0, 1))
        vol = CFG[VOL_KEY[ev.kind]]
        chunk = chunk * vol
        start = int(ev.frame / fps * SAMPLE_RATE)
        end   = start + len(chunk)
        if end > len(mix):
            chunk = chunk[:len(mix) - start]
            end   = len(mix)
        if start < len(mix):
            mix[start:end] += chunk
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix / peak * 0.95
    return mix

# ─────────────────────────── SIMULATION ───────────────────────────
class ArenaRoyale:
    def __init__(self, n_balls: Optional[int] = None):
        self.rng    = random.Random()
        self.np_rng = np.random.default_rng()

        W, H = CFG["width"], CFG["height"]
        self.ax1 = CFG["arena_pad_x"]
        self.ay1 = CFG["arena_top"]
        self.ax2 = W - CFG["arena_pad_x"]
        self.ay2 = H - CFG["arena_bottom_pad"]

        self.fps    = CFG["fps"]
        self.frame  = 0
        self.events: List[AudioEvent]  = []
        self.particles: List[Particle] = []
        self.bullets:   List[Bullet]   = []
        self.powerups:  List[PowerUp]  = []
        self.black_holes: List[BlackHole] = []
        self.laser_beams: List[LaserBeam] = []
        self.winner: Optional[Ball] = None
        self.winner_frame: int = 0
        self.confetti: List[Particle] = []

        self.next_powerup_frame = int(self.rng.uniform(
            CFG["powerup_spawn_interval_min"],
            CFG["powerup_spawn_interval_max"]) * self.fps)
        self.next_jitter_frame = int(CFG["jitter_interval"] * self.fps)

        nb     = n_balls or self.rng.randint(CFG["n_balls_min"], CFG["n_balls_max"])
        colors = self.rng.sample(BALL_PALETTE, min(nb, len(BALL_PALETTE)))
        while len(colors) < nb:
            colors.append(tuple(self.rng.randint(60, 240) for _ in range(3)))
        self.balls: List[Ball] = []
        self._spawn_balls(nb, colors)

    # ── Spawning ─────────────────────────────────────────────────────
    def _spawn_balls(self, nb, colors):
        margin = CFG["radius_max"] + 10
        placed = []
        for i in range(nb):
            r = self.rng.uniform(CFG["radius_min"], CFG["radius_max"])
            for _ in range(200):
                x = self.rng.uniform(self.ax1 + margin, self.ax2 - margin)
                y = self.rng.uniform(self.ay1 + margin, self.ay2 - margin)
                ok = all(math.hypot(x - p[0], y - p[1]) > r + p[2] + 6 for p in placed)
                if ok:
                    placed.append((x, y, r)); break
            else:
                x = self.rng.uniform(self.ax1 + margin, self.ax2 - margin)
                y = self.rng.uniform(self.ay1 + margin, self.ay2 - margin)
                placed.append((x, y, r))

            angle = self.rng.uniform(0, 2 * math.pi)
            spd   = self.rng.uniform(CFG["speed_min"], CFG["speed_max"])
            mass  = (r / CFG["radius_min"]) ** 2
            b = Ball(
                id=i, x=x, y=y,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                radius=r, base_radius=r, mass=mass,
                color=colors[i],
                hp=CFG["hp"], max_hp=CFG["hp"],
                fire_rate=self.rng.uniform(CFG["fire_cooldown_min"], CFG["fire_cooldown_max"]),
                bullet_damage=self.rng.uniform(CFG["bullet_damage_min"], CFG["bullet_damage_max"]),
                aim_jitter=self.rng.uniform(5, 20),
            )
            self.balls.append(b)

    def active_balls(self):
        return [b for b in self.balls if b.active]

    # ── Core physics ─────────────────────────────────────────────────
    def _physics_step(self, dt_sub):
        balls    = self.active_balls()
        ax1, ay1 = self.ax1, self.ay1
        ax2, ay2 = self.ax2, self.ay2
        rest_wall = CFG["restitution_wall"]
        rest_ball = CFG["restitution_ball"]
        vmin = CFG["speed_absolute_min"]
        vmax = CFG["speed_absolute_max"]

        # Black-hole gravity
        for bh in self.black_holes:
            for b in balls:
                dx = bh.x - b.x
                dy = bh.y - b.y
                dist = math.hypot(dx, dy)
                if dist < 5:
                    continue
                # Stronger pull the closer the ball
                pull = bh.strength * dt_sub / max(dist, 60)
                b.vx += (dx / dist) * pull
                b.vy += (dy / dist) * pull
            # Also pull bullets
            for blt in self.bullets:
                dx = bh.x - blt.x
                dy = bh.y - blt.y
                dist = math.hypot(dx, dy)
                if dist < 5:
                    continue
                pull = bh.strength * dt_sub * 2.5 / max(dist, 30)
                blt.vx += (dx / dist) * pull
                blt.vy += (dy / dist) * pull

        for b in balls:
            b.x += b.vx * dt_sub
            b.y += b.vy * dt_sub

            # Wall bounces (always apply, even when phased)
            if b.x - b.radius < ax1:
                b.x  = ax1 + b.radius
                b.vx = abs(b.vx) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vx)/vmax, 1)))
            if b.x + b.radius > ax2:
                b.x  = ax2 - b.radius
                b.vx = -abs(b.vx) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vx)/vmax, 1)))
            if b.y - b.radius < ay1:
                b.y  = ay1 + b.radius
                b.vy = abs(b.vy) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vy)/vmax, 1)))
            if b.y + b.radius > ay2:
                b.y  = ay2 - b.radius
                b.vy = -abs(b.vy) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vy)/vmax, 1)))

            spd = math.hypot(b.vx, b.vy)
            if spd < vmin:
                if spd < 0.01:
                    angle = self.rng.uniform(0, 2 * math.pi)
                    b.vx, b.vy = math.cos(angle) * vmin, math.sin(angle) * vmin
                else:
                    b.vx = b.vx / spd * vmin
                    b.vy = b.vy / spd * vmin
            elif spd > vmax:
                b.vx = b.vx / spd * vmax
                b.vy = b.vy / spd * vmax

        # Ball-ball collisions (skip phased balls)
        non_phased = [b for b in balls if not b.is_phased]
        for i in range(len(non_phased)):
            for j in range(i + 1, len(non_phased)):
                a, bb = non_phased[i], non_phased[j]
                dx, dy = bb.x - a.x, bb.y - a.y
                dist   = math.hypot(dx, dy)
                min_d  = a.radius + bb.radius
                if dist < min_d and dist > 0.001:
                    overlap = (min_d - dist) / 2
                    nx, ny  = dx / dist, dy / dist
                    a.x  -= nx * overlap; a.y  -= ny * overlap
                    bb.x += nx * overlap; bb.y += ny * overlap

                    dvx = bb.vx - a.vx
                    dvy = bb.vy - a.vy
                    dot = dvx * nx + dvy * ny
                    if dot < 0:
                        continue
                    m1, m2 = a.mass, bb.mass
                    imp    = (1 + rest_ball) * dot / (m1 + m2)

                    # Giant mode: increased knockback
                    gm_factor_a  = 1.6 if "GM" in a.buffs  else 1.0
                    gm_factor_bb = 1.6 if "GM" in bb.buffs else 1.0
                    a.vx  += imp * m2 * nx * gm_factor_bb
                    a.vy  += imp * m2 * ny * gm_factor_bb
                    bb.vx -= imp * m1 * nx * gm_factor_a
                    bb.vy -= imp * m1 * ny * gm_factor_a
                    self.events.append(AudioEvent(self.frame, "collision", min(abs(dot)/10, 1)))

    # ── Bullets ──────────────────────────────────────────────────────
    def _update_bullets(self):
        ax1, ay1, ax2, ay2 = self.ax1, self.ay1, self.ax2, self.ay2
        alive_bullets = []
        balls = self.active_balls()
        for blt in self.bullets:
            blt.x += blt.vx
            blt.y += blt.vy
            blt.lifetime -= 1
            if blt.lifetime <= 0:
                continue
            if not (ax1 < blt.x < ax2 and ay1 < blt.y < ay2):
                continue
            hit = False
            for ball in balls:
                if ball.id == blt.owner_id:
                    continue
                if ball.is_phased:        # phased balls can't be hit
                    continue
                dist = math.hypot(blt.x - ball.x, blt.y - ball.y)
                if dist < blt.radius + ball.radius:
                    ball.hp -= blt.damage
                    ball.flash_frames = 6
                    dx = ball.x - blt.x
                    dy = ball.y - blt.y
                    d  = math.hypot(dx, dy) or 1
                    ball.vx += (dx / d) * CFG["knockback"]
                    ball.vy += (dy / d) * CFG["knockback"]
                    self._spawn_hit_particles(blt.x, blt.y, ball.color)
                    self.events.append(AudioEvent(self.frame, "hit", min(blt.damage / 20, 1.0)))
                    blt.hits += 1
                    if blt.hits >= 2:
                        hit = True
                        break
                    hit = True
                    break
            if not hit:
                alive_bullets.append(blt)
        self.bullets = alive_bullets

    # ── Laser beams ──────────────────────────────────────────────────
    def _update_lasers(self):
        """Process laser beam activation and damage for balls with LB buff."""
        balls = self.active_balls()
        fps   = self.fps

        CHARGE_FRAMES  = int(0.5 * fps)
        ACTIVE_FRAMES  = int(0.7 * fps)
        LASER_COOLDOWN = 1.2            # seconds between laser shots

        for b in balls:
            if "LB" not in b.buffs:
                b.laser_charge_frames = 0
                b.laser_active_frames = 0
                continue

            b.laser_cooldown = max(0, b.laser_cooldown - 1 / fps)

            if b.laser_active_frames > 0:
                # Beam is firing — do continuous damage along the line
                b.laser_active_frames -= 1
                ax, ay = b.x, b.y
                bx2    = ax + b.laser_dx * 2000
                by2    = ay + b.laser_dy * 2000

                for target in balls:
                    if target.id == b.id or target.is_phased:
                        continue
                    # Distance from target centre to beam line segment
                    dist_line = self._point_to_segment_dist(
                        target.x, target.y, ax, ay, bx2, by2)
                    if dist_line < target.radius + 8:
                        dmg = 1.5  # damage per frame while in beam
                        target.hp -= dmg
                        target.flash_frames = 3

                # Register beam for rendering
                end_x = ax + b.laser_dx * 1500
                end_y = ay + b.laser_dy * 1500
                self.laser_beams.append(LaserBeam(
                    owner_id=b.id,
                    x1=ax, y1=ay, x2=end_x, y2=end_y,
                    frames_left=2,
                    color=b.color,
                ))

                if b.laser_active_frames == 0:
                    b.laser_cooldown = LASER_COOLDOWN

            elif b.laser_charge_frames > 0:
                b.laser_charge_frames -= 1
                if b.laser_charge_frames == 0:
                    b.laser_active_frames = ACTIVE_FRAMES
            else:
                # Ready to charge again?
                if b.laser_cooldown <= 0:
                    # Pick target direction
                    targets = [t for t in balls if t.id != b.id]
                    if targets:
                        tgt = min(targets, key=lambda t: math.hypot(t.x-b.x, t.y-b.y))
                        dx  = tgt.x - b.x; dy = tgt.y - b.y
                        d   = math.hypot(dx, dy) or 1
                        b.laser_dx = dx / d
                        b.laser_dy = dy / d
                        b.laser_charge_frames = CHARGE_FRAMES

        # Age existing beams
        self.laser_beams = [lb for lb in self.laser_beams if lb.frames_left > 0]
        for lb in self.laser_beams:
            lb.frames_left -= 1

    @staticmethod
    def _point_to_segment_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)))
        cx = ax + t*dx; cy = ay + t*dy
        return math.hypot(px - cx, py - cy)

    # ── Shooting ─────────────────────────────────────────────────────
    def _update_balls_shoot(self):
        balls = self.active_balls()
        for b in balls:
            b.fire_cooldown -= 1 / self.fps
            cooldown = b.fire_rate

            if b.fire_cooldown <= 0:
                b.fire_cooldown = cooldown
                if len(self.bullets) < CFG["max_bullets"] and len(balls) > 1:
                    targets = [t for t in balls if t.id != b.id]
                    target  = min(targets, key=lambda t: math.hypot(t.x - b.x, t.y - b.y))
                    dx, dy  = target.x - b.x, target.y - b.y
                    angle   = math.atan2(dy, dx)
                    jitter  = math.radians(self.rng.uniform(-b.aim_jitter, b.aim_jitter))
                    angle  += jitter
                    spd     = CFG["bullet_speed"]
                    blt_r   = CFG["bullet_radius"] * b.bullet_scale
                    blt_dmg = b.bullet_damage

                    # ── Giant Mode: bigger bullets ──
                    if "GM" in b.buffs:
                        blt_r   *= 1.7
                        blt_dmg *= 1.5

                    # ── Multishot Burst ──
                    if "MS" in b.buffs:
                        n_shots = self.rng.randint(6, 10)
                        spread  = math.pi * 2 / n_shots
                        for i in range(n_shots):
                            shot_angle = angle + spread * i
                            self.bullets.append(Bullet(
                                x=b.x + math.cos(shot_angle) * (b.radius + blt_r + 2),
                                y=b.y + math.sin(shot_angle) * (b.radius + blt_r + 2),
                                vx=math.cos(shot_angle) * spd,
                                vy=math.sin(shot_angle) * spd,
                                owner_id=b.id,
                                color=(255, 240, 80),
                                damage=blt_dmg * 0.8,
                                radius=blt_r,
                                lifetime=CFG["bullet_lifetime"],
                            ))
                        # Recoil
                        b.vx -= math.cos(angle) * 3.5
                        b.vy -= math.sin(angle) * 3.5
                        self._spawn_flash_particles(b.x, b.y, (255, 240, 80))
                    else:
                        self.bullets.append(Bullet(
                            x=b.x + math.cos(angle) * (b.radius + blt_r + 2),
                            y=b.y + math.sin(angle) * (b.radius + blt_r + 2),
                            vx=math.cos(angle) * spd,
                            vy=math.sin(angle) * spd,
                            owner_id=b.id,
                            color=b.color,
                            damage=blt_dmg,
                            radius=blt_r,
                            lifetime=CFG["bullet_lifetime"],
                        ))

    # ── Buffs / power-up durations ────────────────────────────────────
    def _update_buffs(self):
        for b in self.active_balls():
            to_remove = [k for k, v in b.buffs.items() if v <= 1]
            for k in to_remove:
                del b.buffs[k]
                # Giant Mode expiry: restore radius
                if k == "GM":
                    b.radius = b.base_radius
                    b.mass   = (b.radius / CFG["radius_min"]) ** 2
                    # Remove bonus HP
                    b.hp = max(1, b.hp - b.giant_hp_bonus)
                    b.max_hp -= b.giant_hp_bonus
                    b.giant_hp_bonus = 0
                # Phase Mode expiry: restore physics flag
                if k == "PM":
                    b.is_phased = False
            for k in list(b.buffs.keys()):
                b.buffs[k] -= 1

    # ── Black-hole decay ─────────────────────────────────────────────
    def _update_black_holes(self):
        alive = []
        for bh in self.black_holes:
            bh.frames_left -= 1
            if bh.frames_left > 0:
                alive.append(bh)
        self.black_holes = alive

    # ── Power-up spawning & collection ───────────────────────────────
    def _update_powerups(self):
        if self.frame >= self.next_powerup_frame:
            active_pu = [p for p in self.powerups if p.active]
            if len(active_pu) < CFG["powerup_max"]:
                margin = CFG["powerup_radius"] + 10
                px   = self.rng.uniform(self.ax1 + margin, self.ax2 - margin)
                py   = self.rng.uniform(self.ay1 + margin, self.ay2 - margin)
                kind = self.rng.choice(list(POWERUP_TYPES.keys()))
                self.powerups.append(PowerUp(x=px, y=py, kind=kind, frame_born=self.frame))
            interval = self.rng.uniform(CFG["powerup_spawn_interval_min"],
                                        CFG["powerup_spawn_interval_max"])
            self.next_powerup_frame = self.frame + int(interval * self.fps)

        for pu in self.powerups:
            if not pu.active:
                continue
            for b in self.active_balls():
                dist = math.hypot(b.x - pu.x, b.y - pu.y)
                if dist < b.radius + CFG["powerup_radius"]:
                    pu.active = False
                    self._apply_powerup(b, pu.kind, pu.x, pu.y)
                    break

    def _apply_powerup(self, b: Ball, kind: str, px: float, py: float):
        pdata      = POWERUP_TYPES[kind]
        dur_frames = int(pdata["duration"] * self.fps)
        color      = pdata["color"]

        if kind == "BH":
            # Black Hole — spawn at power-up location
            bh = BlackHole(
                x=px, y=py,
                frames_left=dur_frames,
                max_frames=dur_frames,
            )
            self.black_holes.append(bh)

        elif kind == "MS":
            b.buffs["MS"] = dur_frames

        elif kind == "LB":
            b.buffs["LB"]   = dur_frames
            b.laser_cooldown = 0

        elif kind == "CS":
            # Clone Split — instant: spawn 2 clones
            self._clone_ball(b, n_clones=2)

        elif kind == "GM":
            b.buffs["GM"] = dur_frames
            # Grow radius (capped so the giant ball still fits in the arena)
            b.base_radius = b.radius
            new_r = b.radius * 2.0
            arena_w = self.ax2 - self.ax1
            arena_h = self.ay2 - self.ay1
            max_r = min(MAX_BALL_RADIUS, arena_w / 4 - 10, arena_h / 4 - 10)
            b.radius = min(new_r, max_r)
            b.mass   = (b.radius / CFG["radius_min"]) ** 2
            # Grant bonus HP
            bonus = 40
            b.hp     += bonus
            b.max_hp += bonus
            b.giant_hp_bonus = bonus

        elif kind == "PM":
            b.buffs["PM"] = dur_frames
            b.is_phased   = True

        self._spawn_powerup_particles(px, py, color)
        self.events.append(AudioEvent(self.frame, "powerup"))

    def _clone_ball(self, original: Ball, n_clones: int = 2):
        for i in range(n_clones):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd   = math.hypot(original.vx, original.vy) or CFG["speed_min"]
            # Lighter colour
            lc = tuple(min(255, int(v * 1.35 + 40)) for v in original.color)
            clone_r  = original.radius * 0.75
            clone_hp = original.hp * 0.5

            offset_dist = original.radius + clone_r + 8
            cx = original.x + math.cos(angle) * offset_dist
            cy = original.y + math.sin(angle) * offset_dist
            cx = max(self.ax1 + clone_r, min(self.ax2 - clone_r, cx))
            cy = max(self.ay1 + clone_r, min(self.ay2 - clone_r, cy))

            clone = Ball(
                id=len(self.balls),
                x=cx, y=cy,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                radius=clone_r, base_radius=clone_r,
                mass=(clone_r / CFG["radius_min"]) ** 2,
                color=lc,
                hp=clone_hp, max_hp=original.max_hp * 0.5,
                fire_rate=original.fire_rate,
                bullet_damage=original.bullet_damage * 0.7,
                aim_jitter=original.aim_jitter,
            )
            self.balls.append(clone)
        self._spawn_flash_particles(original.x, original.y, (200, 255, 140))

    # ── Deaths ────────────────────────────────────────────────────────
    def _check_deaths(self):
        deaths = 0
        for b in self.active_balls():
            if b.hp <= 0:
                b.active = False
                self._spawn_explosion(b.x, b.y, b.color, b.radius)
                self.events.append(AudioEvent(self.frame, "destroy"))
                deaths += 1
        if deaths:
            self._apply_death_growth(deaths)

    def _apply_death_growth(self, death_count: int):
        growth_factor = GF ** death_count
        arena_w = self.ax2 - self.ax1
        arena_h = self.ay2 - self.ay1
        # Cap: ball must fit with at least 10 px margin on each side AND
        # two of the largest balls must still be able to sit next to each other.
        # We derive the cap dynamically so it stays valid regardless of arena size.
        max_r = min(MAX_BALL_RADIUS,
                    arena_w / 4 - 10,   # two balls wide with breathing room
                    arena_h / 4 - 10)

        for b in self.active_balls():
            if "GM" not in b.buffs:
                b.radius = min(b.radius * growth_factor, max_r)
            b.base_radius = min(b.base_radius * growth_factor, max_r)
            b.mass = (b.radius / CFG["radius_min"]) ** 2
            # Clamp bullet scale so bullets don't become comically huge either
            b.bullet_scale = min(b.bullet_scale * growth_factor, 4.0)
        for blt in self.bullets:
            blt.radius = min(blt.radius * growth_factor, 30)

    # ── Jitter & corner escape ────────────────────────────────────────
    def _apply_jitter(self):
        if self.frame >= self.next_jitter_frame:
            for b in self.active_balls():
                angle = self.rng.uniform(0, 2 * math.pi)
                s = CFG["jitter_strength"]
                b.vx += math.cos(angle) * s
                b.vy += math.sin(angle) * s
            self.next_jitter_frame = self.frame + int(CFG["jitter_interval"] * self.fps)

    def _check_corner_escape(self):
        margin = 60
        for b in self.active_balls():
            in_corner = (
                (b.x < self.ax1 + margin or b.x > self.ax2 - margin) and
                (b.y < self.ay1 + margin or b.y > self.ay2 - margin)
            )
            if in_corner:
                b.corner_frames += 1
                if b.corner_frames > CFG["corner_escape_time"] * self.fps:
                    cx = (self.ax1 + self.ax2) / 2
                    cy = (self.ay1 + self.ay2) / 2
                    dx, dy = cx - b.x, cy - b.y
                    d = math.hypot(dx, dy) or 1
                    boost = CFG["speed_min"] * 1.5
                    b.vx += dx / d * boost
                    b.vy += dy / d * boost
                    b.corner_frames = 0
            else:
                b.corner_frames = 0

    # ── Main step ────────────────────────────────────────────────────
    def step(self):
        self.frame += 1
        dt_sub = 1.0 / self.fps / CFG["substeps"]
        for _ in range(CFG["substeps"]):
            self._physics_step(dt_sub)
        self._apply_jitter()
        self._check_corner_escape()
        self._update_buffs()
        self._update_balls_shoot()
        self._update_bullets()
        self._update_lasers()
        self._update_powerups()
        self._update_black_holes()
        self._check_deaths()
        self._update_particles()

        alive = self.active_balls()
        if len(alive) == 1 and self.winner is None:
            self.winner = alive[0]
            self.winner_frame = self.frame
            self._spawn_confetti()

    # ── Particle helpers ──────────────────────────────────────────────
    def _spawn_hit_particles(self, x, y, color):
        for _ in range(8):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(1.5, 5)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=12, max_life=12, radius=2.5)
            self.particles.append(p)

    def _spawn_flash_particles(self, x, y, color):
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(3, 10)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=18, max_life=18, radius=4)
            self.particles.append(p)

    def _spawn_powerup_particles(self, x, y, color):
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd   = self.rng.uniform(2, 7)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=25, max_life=25, radius=3.5)
            self.particles.append(p)

    def _spawn_explosion(self, x, y, color, radius):
        for _ in range(50):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd   = self.rng.uniform(2, radius * 0.5)
            life  = self.rng.randint(20, 40)
            r     = self.rng.uniform(3, 8)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=life, max_life=life, radius=r)
            self.particles.append(p)
        p = Particle(x=x, y=y, vx=0, vy=0, color=(255,255,255),
                     life=8, max_life=8, radius=radius*1.5)
        self.particles.append(p)

    def _spawn_confetti(self):
        W = CFG["width"]
        for _ in range(120):
            c = self.rng.choice(BALL_PALETTE)
            p = Particle(
                x=self.rng.uniform(0, W),
                y=self.rng.uniform(0, CFG["arena_top"]),
                vx=self.rng.uniform(-2, 2),
                vy=self.rng.uniform(2, 8),
                color=c, life=90, max_life=90,
                radius=self.rng.uniform(4, 9)
            )
            self.confetti.append(p)

    def _update_particles(self):
        alive = []
        for p in self.particles:
            p.x += p.vx; p.y += p.vy
            p.vy += 0.15
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.particles = alive

        alive = []
        for p in self.confetti:
            p.x += p.vx; p.y += p.vy
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.confetti = alive

    # ─────────────────────────── RENDERING ───────────────────────────
    def render_frame(self, img: np.ndarray):
        W, H = CFG["width"], CFG["height"]
        ax1, ay1, ax2, ay2 = self.ax1, self.ay1, self.ax2, self.ay2

        img[:] = BG_COLOR

        # Arena walls
        cv2.rectangle(img, (ax1, ay1), (ax2, ay2), WALL_COLOR, 3)
        L, T = 25, 5
        for cx, cy in [(ax1, ay1), (ax2, ay1), (ax1, ay2), (ax2, ay2)]:
            dx = 1 if cx == ax1 else -1
            dy = 1 if cy == ay1 else -1
            cv2.line(img, (cx, cy), (cx + dx*L, cy), (160,170,200), T)
            cv2.line(img, (cx, cy), (cx, cy + dy*L), (160,170,200), T)

        # ── Black Holes ─────────────────────────────────────────────
        for bh in self.black_holes:
            ratio   = bh.frames_left / bh.max_frames
            max_r   = 80
            cur_r   = max(8, int(max_r * ratio))
            cx_, cy_ = int(bh.x), int(bh.y)

            # Dark outer glow
            for ring_r in range(cur_r + 40, cur_r - 1, -6):
                alpha_val = max(0, min(255, int(80 * (1 - ring_r / (cur_r + 40)))))
                overlay = img.copy()
                cv2.circle(overlay, (cx_, cy_), ring_r, (80, 10, 120), -1)
                img[:] = cv2.addWeighted(img, 0.92, overlay, 0.08, 0)

            cv2.circle(img, (cx_, cy_), cur_r, (20, 5, 40), -1)
            cv2.circle(img, (cx_, cy_), cur_r, (140, 60, 200), 3)

            # Spiral orbit particles
            t = self.frame * 0.15
            for k in range(8):
                theta  = t + k * (math.pi * 2 / 8)
                orbit_r = cur_r * 1.5
                px_ = int(cx_ + math.cos(theta) * orbit_r)
                py_ = int(cy_ + math.sin(theta) * orbit_r)
                cv2.circle(img, (px_, py_), 4, (180, 80, 255), -1)

        # ── Power-ups ────────────────────────────────────────────────
        for pu in self.powerups:
            if not pu.active:
                continue
            pdata = POWERUP_TYPES[pu.kind]
            pulse = 0.7 + 0.3 * math.sin(self.frame * 0.15)
            r     = int(CFG["powerup_radius"] * pulse)
            c     = pdata["color"]
            cv2.circle(img, (int(pu.x), int(pu.y)), r+6,
                       tuple(int(v*0.4) for v in c), -1)
            cv2.circle(img, (int(pu.x), int(pu.y)), r, c, -1)
            # Two-letter label
            short = pu.kind
            cv2.putText(img, short, (int(pu.x)-12, int(pu.y)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        # ── Particles ────────────────────────────────────────────────
        for p in self.particles:
            alpha = p.life / p.max_life
            r = max(1, int(p.radius * alpha))
            c = tuple(int(v*alpha) for v in p.color)
            cv2.circle(img, (int(p.x), int(p.y)), r, c, -1)

        # ── Confetti ─────────────────────────────────────────────────
        for p in self.confetti:
            alpha = p.life / p.max_life
            c = tuple(int(v*alpha) for v in p.color)
            rr = int(p.radius); rh = max(1, rr//2)
            cv2.rectangle(img,
                (int(p.x)-rr, int(p.y)-rh),
                (int(p.x)+rr, int(p.y)+rh), c, -1)

        # ── Laser beams ──────────────────────────────────────────────
        for lb in self.laser_beams:
            # Thick glowing beam
            cv2.line(img, (int(lb.x1), int(lb.y1)), (int(lb.x2), int(lb.y2)),
                     (255, 255, 255), 6, cv2.LINE_AA)
            cv2.line(img, (int(lb.x1), int(lb.y1)), (int(lb.x2), int(lb.y2)),
                     lb.color, 3, cv2.LINE_AA)
            # Short afterimage
            mid_x = int((lb.x1 + lb.x2) / 2)
            mid_y = int((lb.y1 + lb.y2) / 2)
            cv2.line(img, (int(lb.x1), int(lb.y1)), (mid_x, mid_y),
                     (200, 240, 255), 1, cv2.LINE_AA)

        # ── Bullets ──────────────────────────────────────────────────
        for blt in self.bullets:
            cv2.circle(img, (int(blt.x), int(blt.y)), int(blt.radius), (255,255,220), -1)

        # ── Balls ────────────────────────────────────────────────────
        for b in self.active_balls():
            flash = b.flash_frames > 0
            if flash:
                b.flash_frames -= 1
                draw_color = (255, 255, 255)
            else:
                draw_color = b.color

            bx, by = int(b.x), int(b.y)
            br     = int(b.radius)

            # Phase Mode: semi-transparent + glitch lines
            if b.is_phased:
                overlay = img.copy()
                cv2.circle(overlay, (bx, by), br, draw_color, -1)
                img[:] = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
                # Glitch lines
                for _ in range(3):
                    gx1 = bx + self.rng.randint(-br, br)
                    gy1 = by + self.rng.randint(-br, br)
                    gx2 = gx1 + self.rng.randint(-20, 20)
                    gy2 = gy1 + self.rng.randint(-5, 5)
                    cv2.line(img, (gx1, gy1), (gx2, gy2), (160, 160, 255), 2)
            else:
                # Giant Mode aura
                if "GM" in b.buffs:
                    pulse = int(8 + 4 * math.sin(self.frame * 0.25))
                    cv2.circle(img, (bx, by), br + pulse + 6, (255, 160, 40), 3)
                    cv2.circle(img, (bx, by), br + pulse,     (255, 200, 80), 2)

                # Laser charge ring
                if b.laser_charge_frames > 0:
                    pulse = int(b.radius * 0.3 * math.sin(self.frame * 0.4))
                    cv2.circle(img, (bx, by), br + 10 + pulse, (80, 240, 255), 3)

                cv2.circle(img, (bx, by), br, draw_color, -1)

            # Highlight
            hx = int(b.x - b.radius * 0.3)
            hy = int(b.y - b.radius * 0.3)
            hr = max(3, int(b.radius * 0.25))
            hl = tuple(min(255, int(v*1.4+60)) for v in draw_color)
            cv2.circle(img, (hx, hy), hr, hl, -1)

            # HP bar
            bar_w  = int(b.radius * 2.2)
            bar_h  = 6
            bar_bx = int(b.x - bar_w / 2)
            bar_by = int(b.y - b.radius - 14)
            cv2.rectangle(img, (bar_bx, bar_by), (bar_bx+bar_w, bar_by+bar_h), (40,40,40), -1)
            filled   = int(bar_w * max(0, b.hp / b.max_hp))
            hp_color = self._hp_color(b.hp / b.max_hp)
            cv2.rectangle(img, (bar_bx, bar_by), (bar_bx+filled, bar_by+bar_h), hp_color, -1)

            # Ball number
            cv2.putText(img, str(b.id+1),
                        (int(b.x)-6, int(b.y)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            # Buff icons
            icon_labels = {
                "BH": "BH", "MS": "MS", "LB": "LB",
                "CS": "CS", "GM": "GM", "PM": "PM",
            }
            for ki, k in enumerate(b.buffs.keys()):
                ic_x = int(b.x - b.radius + ki * 20)
                ic_y = int(b.y + b.radius + 14)
                ic_c = POWERUP_TYPES[k]["color"]
                cv2.circle(img, (ic_x, ic_y), 9, ic_c, -1)
                cv2.putText(img, icon_labels.get(k, k), (ic_x-7, ic_y+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        # Winner overlay
        if self.winner:
            wf = self.frame - self.winner_frame
            if wf < self.fps * 4:
                self._draw_winner(img, wf)

        # HUD
        self._draw_hud(img)

    def _hp_color(self, ratio):
        if ratio > 0.5:   return (50, 200, 80)
        elif ratio > 0.25: return (240, 180, 30)
        else:              return (220, 50, 50)

    def _draw_hud(self, img):
        W = CFG["width"]
        cv2.rectangle(img, (0, 0), (W, CFG["arena_top"]-10), HUD_BG, -1)
        cv2.putText(img, "ARENA ROYALE",
                    (W//2-160, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,215,0), 2, cv2.LINE_AA)

        n = len(self.balls)
        bar_total_w = min(W-40, n*90)
        bar_w  = bar_total_w // max(n, 1)
        start_x = (W - bar_total_w) // 2
        y = 80
        for i, b in enumerate(self.balls):
            bx = start_x + i*bar_w + 4
            cv2.circle(img, (bx+12, y+14), 12, b.color, -1)
            cv2.rectangle(img, (bx+27, y+8), (bx+bar_w-8, y+20), (40,40,40), -1)
            if b.active:
                ratio  = max(0, b.hp/b.max_hp)
                filled = int((bar_w-35)*ratio)
                cv2.rectangle(img, (bx+27, y+8), (bx+27+filled, y+20),
                              self._hp_color(ratio), -1)
            else:
                cv2.line(img, (bx+27, y+8),    (bx+bar_w-8, y+20), (80,80,80), 2)
                cv2.line(img, (bx+bar_w-8, y+8),(bx+27, y+20),      (80,80,80), 2)

    def _draw_winner(self, img, wf):
        W, H = CFG["width"], CFG["height"]
        b     = self.winner
        alpha = min(1.0, wf / (self.fps * 0.5))

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0,0,0), -1)
        img[:] = cv2.addWeighted(img, 1-alpha*0.45, overlay, alpha*0.45, 0)

        cx, cy = W//2, H//2 - 80
        pts = np.array([
            [cx-50, cy+30], [cx-30, cy-20], [cx, cy+10],
            [cx+30, cy-20], [cx+50, cy+30],
        ], np.int32)
        cv2.fillPoly(img, [pts], (255,215,0))
        cv2.rectangle(img, (cx-50, cy+30), (cx+50, cy+45), (255,215,0), -1)

        cv2.putText(img, "WINNER!", (W//2-130, H//2+20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, WINNER_COLOR, 4, cv2.LINE_AA)
        cv2.putText(img, f"BALL {b.id+1}", (W//2-90, H//2+90),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, b.color, 3, cv2.LINE_AA)


# ─────────────────────────── VIDEO PIPELINE ───────────────────────────
def _write_wav(path, data_int16: np.ndarray, sr: int):
    num_samples = len(data_int16)
    data_bytes  = data_int16.tobytes()
    data_size   = len(data_bytes)
    byte_rate   = sr * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)


def run_headless(n_balls: Optional[int], output: str):
    print(f"\n  Arena Royale [HEADLESS] | output={output}\n")
    sim = ArenaRoyale(n_balls=n_balls)

    fps      = CFG["fps"]
    W, H     = CFG["width"], CFG["height"]
    max_frames = CFG["max_duration"] * fps
    slowmo   = CFG["slowmo_final"]
    slowmo_dur = int(CFG["slowmo_duration"] * fps)

    frames_main      = []
    winner_frame_idx = None

    print("  Simulating + rendering frames ...")
    img = np.zeros((H, W, 3), dtype=np.uint8)

    for f in tqdm(range(max_frames), unit="frame"):
        sim.step()
        frame_img = img.copy()
        sim.render_frame(frame_img)
        frames_main.append(frame_img.copy())

        if sim.winner is not None and winner_frame_idx is None:
            winner_frame_idx = f
        if sim.winner is not None and (f - winner_frame_idx) >= fps * 3:
            break

    total_main = len(frames_main)

    slowmo_frames = []
    if slowmo and winner_frame_idx is not None:
        print("  Building slow-motion segment ...")
        pre_frames = frames_main[max(0, winner_frame_idx - slowmo_dur):winner_frame_idx + 1]
        for pf in pre_frames:
            pf_marked = pf.copy()
            cv2.putText(pf_marked, "SLOW MOTION",
                        (W//2-140, H - CFG["arena_bottom_pad"] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,220,0), 2, cv2.LINE_AA)
            slowmo_frames.append(pf_marked)
            slowmo_frames.append(pf_marked)

    all_frames   = frames_main + slowmo_frames
    total_frames = len(all_frames)
    print(f"   Total frames: {total_frames}  ({total_frames/fps:.1f}s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video = os.path.join(tmpdir, "raw.mp4")
        audio_wav = os.path.join(tmpdir, "audio.wav")

        print("  Writing video frames ...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_video, fourcc, fps, (W, H))
        for frame_img in tqdm(all_frames, unit="frame"):
            writer.write(frame_img)
        writer.release()

        print("  Synthesising audio ...")
        audio = build_audio(sim.events, total_main, fps)
        target_samples = int(total_frames / fps * SAMPLE_RATE) + 1024
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]
        audio_int16 = (audio * 32767).astype(np.int16)
        _write_wav(audio_wav, audio_int16, SAMPLE_RATE)

        print("  Muxing audio + video ...")
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_video, "-i", audio_wav,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg stderr:", result.stderr[-2000:])
            print("  ffmpeg failed; saving video-only fallback.")
            import shutil
            shutil.copy(raw_video, output)
        else:
            print(f"\n  Done!  {output}")


def run_preview(n_balls: Optional[int]):
    DISPLAY_SCALE = 0.4
    print("\n  Arena Royale [PREVIEW]")
    print("  Window controls:  Q = quit\n")

    sim = ArenaRoyale(n_balls=n_balls)
    fps = CFG["fps"]
    W, H = CFG["width"], CFG["height"]
    dW, dH = int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE)
    max_frames     = CFG["max_duration"] * fps
    frame_delay_ms = max(1, int(1000 / fps))

    img = np.zeros((H, W, 3), dtype=np.uint8)
    winner_frame_idx = None

    win_name = "Arena Royale  |  Q to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, dW, dH)

    for f in range(max_frames):
        t_start = time.perf_counter()

        sim.step()
        sim.render_frame(img)

        display = cv2.resize(img, (dW, dH), interpolation=cv2.INTER_LINEAR)
        cv2.putText(display, "PREVIEW MODE  |  press Q to quit",
                    (10, dH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 130), 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)

        if sim.winner is not None and winner_frame_idx is None:
            winner_frame_idx = f
        if sim.winner is not None and (f - winner_frame_idx) >= fps * 3:
            break

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        wait_ms    = max(1, frame_delay_ms - elapsed_ms)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == 27:
            print("  Quit by user.")
            break

    if sim.winner:
        print(f"\n  Winner: Ball {sim.winner.id + 1}!")
        cv2.waitKey(2000)

    cv2.destroyAllWindows()
    print("  Preview ended.")


# ─────────────────────────── ENTRY POINT ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arena Royale - Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Power-ups
---------
  BH  Black Hole    – spawns a gravity vortex that pulls balls & bullets
  MS  Multishot     – next shots fire 6-10 radial bullets with recoil
  LB  Laser Beam    – charges then fires a persistent damaging beam
  CS  Clone Split   – instantly splits into 2 smaller clones
  GM  Giant Mode    – 2x size, bigger bullets, extra HP, strong knockback
  PM  Phase Mode    – passes through walls and balls, can't be hit

Modes
-----
  (default)    Headless – simulate and encode to mp4.
  --preview    Live window. Press Q or Esc to quit.

Examples
--------
  python arena_royale.py
  python arena_royale.py --balls 10 --output round1.mp4
  python arena_royale.py --preview --balls 8
        """
    )
    parser.add_argument("--balls",   type=int, default=None)
    parser.add_argument("--output",  type=str, default="arena_royale.mp4")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.preview:
        run_preview(n_balls=args.balls)
    else:
        run_headless(n_balls=args.balls, output=args.output)