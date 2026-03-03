"""
Arena Royale – YouTube Shorts Generator
========================================
Pure Python pipeline: simulation → frames (OpenCV) → audio (numpy/scipy) → mp4 (ffmpeg)

Usage:
    python arena_royale.py [--seed 42] [--balls 8] [--output arena.mp4] [--preview]

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
    "max_duration": 160,      # seconds (hard cap)
    "slowmo_final": True,    # slow-mo on final kill
    "slowmo_duration": 1.5,  # seconds of slowmo at end

    # Arena
    "arena_pad_x": 60,       # px from edge
    "arena_top": 500,        # px from top (HUD space)
    "arena_bottom_pad": 500,

    # Balls
    "n_balls_min": 6,
    "n_balls_max": 12,
    "radius_min": 600,
    "radius_max": 820,
    "speed_min": 400.0,        # px/frame
    "speed_max": 800.0,
    "speed_absolute_min": 400.5,
    "speed_absolute_max": 2200.0,
    "restitution_wall": 0.92,
    "restitution_ball": 0.92,
    "hp": 100,

    # Bullets
    "fire_cooldown_min": 1,  # seconds
    "fire_cooldown_max": 1.4,
    "bullet_damage_min": 8,
    "bullet_damage_max": 14,
    "bullet_speed": 15.0,      # px/frame
    "bullet_radius": 5,
    "bullet_lifetime": 90,     # frames
    "max_bullets": 200,
    "knockback": 2.5,

    # Power-ups
    "powerup_spawn_interval_min": 1.8,  # seconds
    "powerup_spawn_interval_max": 3.5,
    "powerup_max": 3,
    "powerup_radius": 30,

    # Physics
    "substeps": 3,
    "jitter_interval": 1.2,   # seconds between random impulses
    "jitter_strength": 0.8,
    "corner_escape_time": 1.5, # seconds before nudging cornered ball

    # Audio volumes (0-1)
    "vol_bounce": 0.4,
    "vol_collision": 0.5,
    "vol_hit": 0.35,
    "vol_powerup": 0.6,
    "vol_destroy": 0.7,
}

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

POWERUP_TYPES = {
    "F": {"color": (255, 80,  80), "label": "RAPID",   "duration": 5.0},
    "S": {"color": ( 80,160, 255), "label": "SHIELD",  "duration": 5.0},
    "B": {"color": (255,200,  40), "label": "BIG GUN", "duration": 4.0},
    "V": {"color": ( 80,255,120),  "label": "SPEED",   "duration": 4.0},
    "H": {"color": (255, 80, 200), "label": "HEAL",    "duration": 0},
    "P": {"color": (200, 80, 255), "label": "PIERCE",  "duration": 4.0},
}

BG_COLOR      = (18, 18, 24)
WALL_COLOR    = (80, 90, 110)
HUD_BG        = (10, 10, 16)
TEXT_COLOR    = (220, 220, 230)
WINNER_COLOR  = (255, 215,  0)

# ─────────────────────────── DATA CLASSES ───────────────────────────
@dataclass
class Ball:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
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
    low  = np.sin(2 * np.pi * 60  * t) * np.exp(-t * 12)
    mid  = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 25)
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
        end = start + len(chunk)
        if end > len(mix):
            chunk = chunk[:len(mix) - start]
            end = len(mix)
        if start < len(mix):
            mix[start:end] += chunk
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix / peak * 0.95
    return mix

# ─────────────────────────── SIMULATION ───────────────────────────
class ArenaRoyale:
    def __init__(self, seed: int, n_balls: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        W, H = CFG["width"], CFG["height"]
        self.ax1 = CFG["arena_pad_x"]
        self.ay1 = CFG["arena_top"]
        self.ax2 = W - CFG["arena_pad_x"]
        self.ay2 = H - CFG["arena_bottom_pad"]

        self.fps = CFG["fps"]
        self.frame = 0
        self.events: List[AudioEvent] = []
        self.particles: List[Particle] = []
        self.bullets: List[Bullet] = []
        self.powerups: List[PowerUp] = []
        self.winner: Optional[Ball] = None
        self.winner_frame: int = 0
        self.confetti: List[Particle] = []

        self.next_powerup_frame = int(self.rng.uniform(
            CFG["powerup_spawn_interval_min"],
            CFG["powerup_spawn_interval_max"]) * self.fps)
        self.next_jitter_frame = int(CFG["jitter_interval"] * self.fps)

        nb = n_balls or self.rng.randint(CFG["n_balls_min"], CFG["n_balls_max"])
        colors = self.rng.sample(BALL_PALETTE, min(nb, len(BALL_PALETTE)))
        while len(colors) < nb:
            colors.append(tuple(self.rng.randint(60, 240) for _ in range(3)))
        self.balls: List[Ball] = []
        self._spawn_balls(nb, colors)

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
                    placed.append((x, y, r))
                    break
            else:
                x = self.rng.uniform(self.ax1 + margin, self.ax2 - margin)
                y = self.rng.uniform(self.ay1 + margin, self.ay2 - margin)
                placed.append((x, y, r))

            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(CFG["speed_min"], CFG["speed_max"])
            mass = (r / CFG["radius_min"]) ** 2
            b = Ball(
                id=i, x=x, y=y,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                radius=r, mass=mass,
                color=colors[i],
                hp=CFG["hp"], max_hp=CFG["hp"],
                fire_rate=self.rng.uniform(CFG["fire_cooldown_min"], CFG["fire_cooldown_max"]),
                bullet_damage=self.rng.uniform(CFG["bullet_damage_min"], CFG["bullet_damage_max"]),
                aim_jitter=self.rng.uniform(5, 20),
            )
            self.balls.append(b)

    def active_balls(self):
        return [b for b in self.balls if b.active]

    def _physics_step(self, dt_sub):
        balls = self.active_balls()
        ax1, ay1, ax2, ay2 = self.ax1, self.ay1, self.ax2, self.ay2
        rest_wall = CFG["restitution_wall"]
        rest_ball = CFG["restitution_ball"]
        vmin = CFG["speed_absolute_min"]
        vmax = CFG["speed_absolute_max"]

        for b in balls:
            b.x += b.vx * dt_sub
            b.y += b.vy * dt_sub

            if b.x - b.radius < ax1:
                b.x = ax1 + b.radius
                b.vx = abs(b.vx) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vx)/vmax, 1)))
            if b.x + b.radius > ax2:
                b.x = ax2 - b.radius
                b.vx = -abs(b.vx) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vx)/vmax, 1)))
            if b.y - b.radius < ay1:
                b.y = ay1 + b.radius
                b.vy = abs(b.vy) * rest_wall
                self.events.append(AudioEvent(self.frame, "bounce", min(abs(b.vy)/vmax, 1)))
            if b.y + b.radius > ay2:
                b.y = ay2 - b.radius
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

        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, bb = balls[i], balls[j]
                dx, dy = bb.x - a.x, bb.y - a.y
                dist = math.hypot(dx, dy)
                min_dist = a.radius + bb.radius
                if dist < min_dist and dist > 0.001:
                    overlap = (min_dist - dist) / 2
                    nx, ny = dx / dist, dy / dist
                    a.x -= nx * overlap
                    a.y -= ny * overlap
                    bb.x += nx * overlap
                    bb.y += ny * overlap

                    dvx = bb.vx - a.vx
                    dvy = bb.vy - a.vy
                    dot = dvx * nx + dvy * ny
                    if dot < 0:
                        continue
                    m1, m2 = a.mass, bb.mass
                    imp = (1 + rest_ball) * dot / (m1 + m2)
                    a.vx  += imp * m2 * nx
                    a.vy  += imp * m2 * ny
                    bb.vx -= imp * m1 * nx
                    bb.vy -= imp * m1 * ny
                    self.events.append(AudioEvent(self.frame, "collision", min(abs(dot)/10, 1)))

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
                dist = math.hypot(blt.x - ball.x, blt.y - ball.y)
                if dist < blt.radius + ball.radius:
                    dmg = blt.damage
                    if "S" in ball.buffs:
                        dmg = max(0, dmg - 30)
                    ball.hp -= dmg
                    ball.flash_frames = 6
                    dx = ball.x - blt.x
                    dy = ball.y - blt.y
                    d = math.hypot(dx, dy) or 1
                    ball.vx += (dx / d) * CFG["knockback"]
                    ball.vy += (dy / d) * CFG["knockback"]
                    self._spawn_hit_particles(blt.x, blt.y, ball.color)
                    self.events.append(AudioEvent(self.frame, "hit", min(dmg / 20, 1.0)))
                    blt.hits += 1
                    if not blt.pierce or blt.hits >= 2:
                        hit = True
                        break
            if not hit:
                alive_bullets.append(blt)
        self.bullets = alive_bullets

    def _update_balls_shoot(self):
        balls = self.active_balls()
        for b in balls:
            b.fire_cooldown -= 1 / self.fps
            cooldown = b.fire_rate
            if "F" in b.buffs:
                cooldown *= 0.35
            if b.fire_cooldown <= 0:
                b.fire_cooldown = cooldown
                if len(self.bullets) < CFG["max_bullets"] and len(balls) > 1:
                    targets = [t for t in balls if t.id != b.id]
                    target = min(targets, key=lambda t: math.hypot(t.x - b.x, t.y - b.y))
                    dx, dy = target.x - b.x, target.y - b.y
                    angle = math.atan2(dy, dx)
                    jitter = math.radians(self.rng.uniform(-b.aim_jitter, b.aim_jitter))
                    angle += jitter
                    spd = CFG["bullet_speed"]
                    blt_r = CFG["bullet_radius"]
                    blt_dmg = b.bullet_damage
                    pierce = False
                    if "B" in b.buffs:
                        blt_r *= 1.8
                        blt_dmg *= 1.6
                    if "P" in b.buffs:
                        pierce = True
                    blt = Bullet(
                        x=b.x + math.cos(angle) * (b.radius + blt_r + 2),
                        y=b.y + math.sin(angle) * (b.radius + blt_r + 2),
                        vx=math.cos(angle) * spd,
                        vy=math.sin(angle) * spd,
                        owner_id=b.id,
                        color=b.color,
                        damage=blt_dmg,
                        radius=blt_r,
                        lifetime=CFG["bullet_lifetime"],
                        pierce=pierce,
                    )
                    self.bullets.append(blt)

    def _update_buffs(self):
        for b in self.active_balls():
            to_remove = [k for k, v in b.buffs.items() if v <= 1]
            for k in to_remove:
                del b.buffs[k]
            for k in list(b.buffs.keys()):
                b.buffs[k] -= 1

    def _update_powerups(self):
        if self.frame >= self.next_powerup_frame:
            active_pu = [p for p in self.powerups if p.active]
            if len(active_pu) < CFG["powerup_max"]:
                margin = CFG["powerup_radius"] + 10
                px = self.rng.uniform(self.ax1 + margin, self.ax2 - margin)
                py = self.rng.uniform(self.ay1 + margin, self.ay2 - margin)
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
                    pdata = POWERUP_TYPES[pu.kind]
                    dur_frames = int(pdata["duration"] * self.fps)
                    if pu.kind == "H":
                        b.hp = min(b.hp + 25, b.max_hp)
                    elif pu.kind == "V":
                        spd = math.hypot(b.vx, b.vy)
                        factor = 1.35
                        if spd > 0.01:
                            b.vx = b.vx / spd * spd * factor
                            b.vy = b.vy / spd * spd * factor
                        b.buffs["V"] = dur_frames
                    else:
                        b.buffs[pu.kind] = dur_frames
                    self._spawn_powerup_particles(pu.x, pu.y, pdata["color"])
                    self.events.append(AudioEvent(self.frame, "powerup"))
                    break

    def _check_deaths(self):
        for b in self.active_balls():
            if b.hp <= 0:
                b.active = False
                self._spawn_explosion(b.x, b.y, b.color, b.radius)
                self.events.append(AudioEvent(self.frame, "destroy"))

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

    def _update_buffs_speed(self):
        for b in self.active_balls():
            if "V" in b.buffs:
                spd = math.hypot(b.vx, b.vy)
                target = CFG["speed_min"] * 1.8
                if spd < target and spd > 0.01:
                    b.vx = b.vx / spd * target
                    b.vy = b.vy / spd * target

    def step(self):
        self.frame += 1
        dt_sub = 1.0 / self.fps / CFG["substeps"]
        for _ in range(CFG["substeps"]):
            self._physics_step(dt_sub)
        self._apply_jitter()
        self._check_corner_escape()
        self._update_buffs()
        self._update_buffs_speed()
        self._update_balls_shoot()
        self._update_bullets()
        self._update_powerups()
        self._check_deaths()
        self._update_particles()

        alive = self.active_balls()
        if len(alive) == 1 and self.winner is None:
            self.winner = alive[0]
            self.winner_frame = self.frame
            self._spawn_confetti()

    def _spawn_hit_particles(self, x, y, color):
        for _ in range(8):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(1.5, 5)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=12, max_life=12, radius=2.5)
            self.particles.append(p)

    def _spawn_powerup_particles(self, x, y, color):
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(2, 7)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=25, max_life=25, radius=3.5)
            self.particles.append(p)

    def _spawn_explosion(self, x, y, color, radius):
        for _ in range(50):
            angle = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(2, radius * 0.5)
            life = self.rng.randint(20, 40)
            r = self.rng.uniform(3, 8)
            p = Particle(x=x, y=y,
                         vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                         color=color, life=life, max_life=life, radius=r)
            self.particles.append(p)
        # Flash ring
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
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.15
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        self.particles = alive

        alive = []
        for p in self.confetti:
            p.x += p.vx
            p.y += p.vy
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

        # Power-ups
        for pu in self.powerups:
            if not pu.active:
                continue
            pdata = POWERUP_TYPES[pu.kind]
            pulse = 0.7 + 0.3 * math.sin(self.frame * 0.15)
            r = int(CFG["powerup_radius"] * pulse)
            c = pdata["color"]
            cv2.circle(img, (int(pu.x), int(pu.y)), r+6,
                       tuple(int(v*0.4) for v in c), -1)
            cv2.circle(img, (int(pu.x), int(pu.y)), r, c, -1)
            cv2.putText(img, pu.kind, (int(pu.x)-7, int(pu.y)+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Particles
        for p in self.particles:
            alpha = p.life / p.max_life
            r = max(1, int(p.radius * alpha))
            c = tuple(int(v*alpha) for v in p.color)
            cv2.circle(img, (int(p.x), int(p.y)), r, c, -1)

        # Confetti
        for p in self.confetti:
            alpha = p.life / p.max_life
            c = tuple(int(v*alpha) for v in p.color)
            rr = int(p.radius)
            rh = max(1, rr//2)
            cv2.rectangle(img,
                (int(p.x)-rr, int(p.y)-rh),
                (int(p.x)+rr, int(p.y)+rh), c, -1)

        # Bullets
        for blt in self.bullets:
            cv2.circle(img, (int(blt.x), int(blt.y)), int(blt.radius), (255,255,220), -1)

        # Balls
        for b in self.active_balls():
            flash = b.flash_frames > 0
            if flash:
                b.flash_frames -= 1
                draw_color = (255,255,255)
            else:
                draw_color = b.color

            if "S" in b.buffs:
                pulse = int(5 + 3*math.sin(self.frame*0.2))
                cv2.circle(img, (int(b.x), int(b.y)), int(b.radius)+pulse+4, (80,160,255), 2)

            cv2.circle(img, (int(b.x), int(b.y)), int(b.radius), draw_color, -1)
            hx = int(b.x - b.radius*0.3)
            hy = int(b.y - b.radius*0.3)
            hr = max(3, int(b.radius*0.25))
            hl_color = tuple(min(255, int(v*1.4+60)) for v in draw_color)
            cv2.circle(img, (hx, hy), hr, hl_color, -1)

            # HP bar
            bar_w = int(b.radius*2.2)
            bar_h = 6
            bx = int(b.x - bar_w/2)
            by = int(b.y - b.radius - 14)
            cv2.rectangle(img, (bx, by), (bx+bar_w, by+bar_h), (40,40,40), -1)
            filled = int(bar_w * max(0, b.hp/b.max_hp))
            hp_color = self._hp_color(b.hp/b.max_hp)
            cv2.rectangle(img, (bx, by), (bx+filled, by+bar_h), hp_color, -1)

            # Ball number
            cv2.putText(img, str(b.id+1),
                        (int(b.x)-6, int(b.y)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            # Buff icons
            for ki, k in enumerate(b.buffs.keys()):
                ic_x = int(b.x - b.radius + ki*18)
                ic_y = int(b.y + b.radius + 12)
                ic_c = POWERUP_TYPES[k]["color"]
                cv2.circle(img, (ic_x, ic_y), 8, ic_c, -1)
                cv2.putText(img, k, (ic_x-5, ic_y+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

        # Winner overlay
        if self.winner:
            wf = self.frame - self.winner_frame
            if wf < self.fps * 4:
                self._draw_winner(img, wf)

        # HUD
        self._draw_hud(img)

        # Seed watermark
        cv2.putText(img, f"SEED: {self.seed}",
                    (ax1+5, ay2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,90,110), 1, cv2.LINE_AA)

    def _hp_color(self, ratio):
        if ratio > 0.5:
            return (50, 200, 80)
        elif ratio > 0.25:
            return (240, 180, 30)
        else:
            return (220, 50, 50)

    def _draw_hud(self, img):
        W = CFG["width"]
        cv2.rectangle(img, (0, 0), (W, CFG["arena_top"]-10), HUD_BG, -1)
        cv2.putText(img, "ARENA ROYALE",
                    (W//2-160, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,215,0), 2, cv2.LINE_AA)

        n = len(self.balls)
        bar_total_w = min(W-40, n*90)
        bar_w = bar_total_w // max(n, 1)
        start_x = (W - bar_total_w) // 2
        y = 80
        for i, b in enumerate(self.balls):
            bx = start_x + i*bar_w + 4
            cv2.circle(img, (bx+12, y+14), 12, b.color, -1)
            cv2.rectangle(img, (bx+27, y+8), (bx+bar_w-8, y+20), (40,40,40), -1)
            if b.active:
                ratio = max(0, b.hp/b.max_hp)
                filled = int((bar_w-35)*ratio)
                cv2.rectangle(img, (bx+27, y+8), (bx+27+filled, y+20),
                              self._hp_color(ratio), -1)
            else:
                cv2.line(img, (bx+27, y+8), (bx+bar_w-8, y+20), (80,80,80), 2)
                cv2.line(img, (bx+bar_w-8, y+8), (bx+27, y+20), (80,80,80), 2)

    def _draw_winner(self, img, wf):
        W, H = CFG["width"], CFG["height"]
        b = self.winner
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
    data_bytes = data_int16.tobytes()
    data_size = len(data_bytes)
    byte_rate = sr * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))    # PCM
        f.write(struct.pack("<H", 1))    # mono
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))    # block align
        f.write(struct.pack("<H", 16))   # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)


def run_headless(seed: int, n_balls: Optional[int], output: str):
    """Simulate and encode to mp4. No window is opened."""
    print(f"\n  Arena Royale [HEADLESS] | seed={seed} | output={output}\n")
    sim = ArenaRoyale(seed=seed, n_balls=n_balls)

    fps = CFG["fps"]
    W, H = CFG["width"], CFG["height"]
    max_frames = CFG["max_duration"] * fps
    slowmo = CFG["slowmo_final"]
    slowmo_dur = int(CFG["slowmo_duration"] * fps)

    frames_main = []
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

    # Slow-mo: duplicate frames before winner
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

    all_frames = frames_main + slowmo_frames
    total_frames = len(all_frames)
    print(f"   Total frames: {total_frames}  ({total_frames/fps:.1f}s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video = os.path.join(tmpdir, "raw.mp4")
        audio_wav  = os.path.join(tmpdir, "audio.wav")

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
            "-i", raw_video,
            "-i", audio_wav,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
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


def run_preview(seed: int, n_balls: Optional[int]):
    """
    Live preview mode – opens an OpenCV window and plays the simulation
    at real-time speed. No file is written. Press Q to quit early.

    The window is scaled down to fit most screens (the full 1080x1920
    canvas is rendered internally but displayed at 40% size by default).
    """
    DISPLAY_SCALE = 0.4   # change this if your screen is larger/smaller

    print(f"\n  Arena Royale [PREVIEW] | seed={seed}")
    print("  Window controls:  Q = quit\n")

    sim = ArenaRoyale(seed=seed, n_balls=n_balls)

    fps = CFG["fps"]
    W, H = CFG["width"], CFG["height"]
    dW, dH = int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE)
    max_frames = CFG["max_duration"] * fps
    frame_delay_ms = max(1, int(1000 / fps))

    img = np.zeros((H, W, 3), dtype=np.uint8)
    winner_frame_idx = None

    win_name = f"Arena Royale  |  seed={seed}  |  Q to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, dW, dH)

    for f in range(max_frames):
        t_start = time.perf_counter()

        sim.step()
        sim.render_frame(img)

        # Scale down for display
        display = cv2.resize(img, (dW, dH), interpolation=cv2.INTER_LINEAR)

        # Overlay mode label
        cv2.putText(display, "PREVIEW MODE  |  press Q to quit",
                    (10, dH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 130), 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)

        # Track winner for early exit
        if sim.winner is not None and winner_frame_idx is None:
            winner_frame_idx = f
        if sim.winner is not None and (f - winner_frame_idx) >= fps * 3:
            break

        # Pace to real-time; Q quits
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        wait_ms = max(1, frame_delay_ms - elapsed_ms)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == 27:   # Q or Esc
            print("  Quit by user.")
            break

    # Hold winner screen for 2 s then close
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
Modes
-----
  (default)    Headless – simulate and encode to mp4, no window opened.
               Fast: renders every frame to memory then muxes with ffmpeg.

  --preview    Live window – watch the simulation in real time.
               No file is written. Press Q or Esc to quit early.
               The window is shown at 40% scale (edit DISPLAY_SCALE in
               run_preview() to change this).

Examples
--------
  python arena_royale.py --seed 42
  python arena_royale.py --seed 42 --balls 10 --output round1.mp4
  python arena_royale.py --preview --seed 42
  python arena_royale.py --preview --balls 8
        """
    )
    parser.add_argument("--seed",    type=int,  default=int(time.time()) % 100000,
                        help="RNG seed (default: time-based)")
    parser.add_argument("--balls",   type=int,  default=None,
                        help="Number of balls (overrides random, e.g. 8)")
    parser.add_argument("--output",  type=str,  default="arena_royale.mp4",
                        help="Output mp4 filename (headless mode only)")
    parser.add_argument("--preview", action="store_true",
                        help="Open a live window instead of encoding to file")
    args = parser.parse_args()

    if args.preview:
        run_preview(seed=args.seed, n_balls=args.balls)
    else:
        run_headless(seed=args.seed, n_balls=args.balls, output=args.output)