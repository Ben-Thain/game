# LAST LIGHT
# First-person survival shooter. Stay alive as long as you can.
#
# Install:
#     pip install ursina
# Run:
#     python last_light.py
#
# One file, no asset downloads. Textures, the sky, and every sound are
# generated when the game starts. Best time and mouse sensitivity are
# saved next to this script in last_light_save.json.

from __future__ import annotations

import array
import heapq
import json
import math
import random
import sys
import tempfile
import wave
from pathlib import Path

from panda3d.core import AntialiasAttrib, SamplerState
from PIL import Image

from ursina import *
from ursina import application
from ursina.shaders import unlit_shader
from ursina.ursinamath import lerp_exponential_decay as damp


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CELL = 4.0
WALL_H = 3.85
PLAYER_RADIUS = 0.34
STAND_HEIGHT = 1.74
CROUCH_HEIGHT = 1.12
WALK_SPEED = 5.55
SPRINT_SPEED = 8.75
CROUCH_SPEED = 3.05
GRAVITY = 28.0
JUMP_SPEED = 8.15
HIP_FOV = 90
ADS_FOV = 68
SPRINT_FOV = 98
MAG_SIZE = 30
RESERVE_START = 90
RESERVE_MAX = 180
FIRE_INTERVAL = 0.1
RELOAD_TIME = 1.62
RIFLE_DAMAGE = 26
HEADSHOT_MULT = 1.75
AMMO_PICKUP = 36
HEALTH_PICKUP = 40
SAVE_PATH = Path(__file__).resolve().with_name('last_light_save.json')

THREAT_NAMES = ('CALM', 'HUNT', 'SIEGE', 'OVERRUN')
THREAT_AT = (0, 18, 42, 75)
SPAWN_INTERVAL = (4.6, 3.5, 2.7, 2.05)
SPAWN_CAP = (4, 7, 10, 13)
SPAWN_WEIGHTS = (
    (('chaser', 1),),
    (('chaser', 3), ('shooter', 2)),
    (('chaser', 3), ('shooter', 3), ('brute', 2)),
    (('chaser', 2), ('shooter', 3), ('brute', 3)),
)

ARCHETYPES = {
    'chaser': dict(hp=72, speed=6.35, sight=26, bulk=1.0, alert=0.72,
                   body=(92, 42, 40), eye=(255, 64, 48)),
    'shooter': dict(hp=56, speed=4.15, sight=32, bulk=0.94, alert=1.18,
                    body=(42, 58, 68), eye=(90, 220, 255)),
    'brute': dict(hp=168, speed=2.85, sight=20, bulk=1.34, alert=0.52,
                  body=(74, 54, 40), eye=(255, 150, 48)),
}

MAP_TEXT = (
"################################\n"
"#S......#..............#......S#\n"
"#....*..#..............#..*....#\n"
"#..c....#...YYYYYYYY...#....c..#\n"
"#.......+...YYYYYYYY...+.......#\n"
"#....*..#...YYYYYYYY...#..*....#\n"
"#.......#...YYYYYYYY...#.......#\n"
"#.......#...YYYYYYYY...#.......#\n"
"#...c...#......*.......#...c...#\n"
"#.......#..............#.......#\n"
"########+######..######+########\n"
"#..............................#\n"
"#.....#..................#.....#\n"
"#..H..#...*..........*...#..A..#\n"
"#.....+.........P........+.....#\n"
"#.....#..o..c......c..o..#.....#\n"
"#.....#..................#.....#\n"
"######+#######....#######+######\n"
"#...H......................A...#\n"
"#.........*....c.....*.........#\n"
"#S............................S#\n"
"################################\n"
)

FOG_COLOR = color.rgb(0.025, 0.03, 0.048)
ACCENT = color.rgb32(255, 148, 64)
RATE = 22050

game = None
player = None
nav = None
level_root = None
sounds = None
probe = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ui_text(parent, text, px=18, **kwargs):
    # With the UI film fitted to the window, Text scale 1 is about 18 px at 720p.
    kwargs.setdefault('origin', (0, 0))
    kwargs.setdefault('z', -1)
    return Text(parent=parent, text=text, scale=max(0.45, px / 18), **kwargs)


def fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(seconds) % 60
    tenth = int((seconds - int(seconds)) * 10)
    return f'{minutes:02d}:{secs:02d}.{tenth}'


def norm(vector):
    length = vector.length()
    if length < 1e-5:
        return Vec3(0, 0, 0)
    return Vec3(vector.x / length, vector.y / length, vector.z / length)


def approach(current, target, amount):
    if current < target:
        return min(target, current + amount)
    return max(target, current - amount)


def clamp01(value):
    return clamp(value, 0, 1)


def frame_dt():
    return min(getattr(time, 'dt', 1 / 60) or 1 / 60, 1 / 30)


def threat_index(elapsed):
    index = 0
    for i, mark in enumerate(THREAT_AT):
        if elapsed >= mark:
            index = i
    return index


def weighted_kind(index):
    pairs = SPAWN_WEIGHTS[index]
    total = sum(weight for _, weight in pairs)
    roll = random.uniform(0, total)
    upto = 0
    for kind, weight in pairs:
        upto += weight
        if roll <= upto:
            return kind
    return pairs[-1][0]


def level_blocked(origin, direction, dist):
    if dist <= 0.001 or level_root is None:
        return False
    hit = raycast(origin, direction, distance=dist, traverse_target=level_root)
    return bool(hit.hit)


def line_of_sight(start, end):
    delta = end - start
    dist = delta.length()
    if dist < 0.08:
        return True
    hit = raycast(start, delta, distance=max(0.01, dist - 0.12), traverse_target=level_root)
    return not hit.hit


def victim_from(entity):
    if entity is None:
        return None
    if getattr(entity, 'is_enemy', False) and not getattr(entity, '_gone', False):
        return entity
    owner = getattr(entity, 'owner', None)
    if owner is not None and getattr(owner, 'is_enemy', False) and not getattr(owner, '_gone', False):
        return owner
    return None


def in_front_of_camera(pos):
    return camera.forward.dot(pos - camera.world_position) > 0.15


def world_to_ui(pos):
    probe.position = pos
    point = probe.screen_position
    return Vec2(point.x, point.y)


def angle_from_player(world):
    delta = Vec3(world.x - player.x, 0, world.z - player.z)
    local = Vec3(*player.getRelativeVector(scene, delta))
    return math.atan2(local.x, local.z)


def safe_destroy(entity):
    if entity is None or getattr(entity, '_gone', False):
        return
    entity._gone = True
    for child in list(getattr(entity, 'children', []) or []):
        safe_destroy(child)
    try:
        if not getattr(entity, 'eternal', False):
            destroy(entity)
    except Exception:
        pass


def set_mouse_locked(value):
    try:
        mouse.locked = value
    except Exception:
        mouse._locked = bool(value)


def retire(entity):
    if entity is None or getattr(entity, '_retired', False) or getattr(entity, '_gone', False):
        return
    entity._retired = True
    try:
        entity.enabled = False
    except Exception:
        pass
    if game is None:
        safe_destroy(entity)
        return
    if getattr(entity, 'is_projectile', False) and entity in game.projectiles:
        game.projectiles.remove(entity)
    if getattr(entity, 'is_enemy', False) and entity in game.enemies:
        game.enemies.remove(entity)
    if entity in game.decals:
        game.decals.remove(entity)
    game.trash.append(entity)


# ---------------------------------------------------------------------------
# Procedural audio
# ---------------------------------------------------------------------------

def _write_wav(path, samples):
    peak = max((abs(sample) for sample in samples), default=1.0)
    scale = 0.92 / peak if peak > 0.92 else 1.0
    frames = array.array('h', (int(clamp(sample * scale, -1, 1) * 32767) for sample in samples))
    with wave.open(str(path), 'w') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(frames.tobytes())
    return path


def _mix(*tracks):
    length = max(len(track) for track in tracks)
    mixed = [0.0] * length
    for track in tracks:
        for i, sample in enumerate(track):
            mixed[i] += sample
    return mixed


def _silence(duration):
    return [0.0] * int(duration * RATE)


def _tone(freq, duration, amp=1.0, decay=None, slide=0.0):
    count = int(duration * RATE)
    out = []
    for i in range(count):
        t = i / RATE
        envelope = math.exp(-t / decay) if decay else 1.0
        f = freq + slide * t
        out.append(math.sin(2 * math.pi * f * t) * amp * envelope)
    return out


def _noise(duration, amp, decay, rng, color_mix=0.0):
    count = int(duration * RATE)
    out = []
    hold = 0.0
    for i in range(count):
        t = i / RATE
        white = rng.uniform(-1, 1)
        hold = hold * color_mix + white * (1 - color_mix)
        out.append(hold * amp * math.exp(-t / decay))
    return out


def _pad(samples, delay):
    return _silence(delay) + samples


class SoundPool:
    def __init__(self, path, voices):
        self.voices = []
        self.index = 0
        for _ in range(voices):
            try:
                audio = Audio(Path(path), autoplay=False, loop=False, volume=1)
                self.voices.append(audio)
            except Exception as exc:
                print('audio voice skipped:', exc)
                break

    def play(self, volume=1.0, pitch=1.0):
        if not self.voices:
            return
        voice = self.voices[self.index % len(self.voices)]
        self.index += 1
        try:
            voice.volume = volume
            voice.pitch = max(0.2, pitch)
            voice.play()
        except Exception:
            pass


class SoundBoard:
    def __init__(self):
        self.pools = {}
        self.ambient = None

    def add(self, name, path, voices=4):
        self.pools[name] = SoundPool(path, voices)

    def play(self, name, volume=1.0, pitch=1.0):
        pool = self.pools.get(name)
        if pool:
            pool.play(volume, pitch)


def build_sounds():
    rng = random.Random(11)
    folder = Path(tempfile.mkdtemp(prefix='lastlight_audio_'))
    board = SoundBoard()
    recipes = {
        'shot': _mix(
            _noise(0.15, 0.85, 0.028, rng, 0.15),
            _tone(78, 0.16, 0.85, 0.035),
            _tone(160, 0.07, 0.28, 0.02),
        ),
        'dry': _mix(_tone(1400, 0.03, 0.3, 0.008), _noise(0.02, 0.2, 0.006, rng)),
        'reload': _mix(
            _pad(_noise(0.04, 0.55, 0.01, rng), 0.0),
            _pad(_tone(420, 0.05, 0.25, 0.012), 0.12),
            _pad(_noise(0.06, 0.7, 0.012, rng), 0.28),
            _pad(_tone(180, 0.08, 0.35, 0.02), 0.3),
        ),
        'step': _noise(0.07, 0.55, 0.016, rng, 0.55),
        'alert': _tone(520, 0.22, 0.4, 0.12, slide=900),
        'hurt': _mix(_noise(0.18, 0.45, 0.05, rng), _tone(196, 0.16, 0.35, 0.06), _tone(311, 0.14, 0.2, 0.05)),
        'death': _mix(_tone(440, 0.55, 0.4, 0.22, slide=-520), _tone(220, 0.6, 0.25, 0.25, slide=-180)),
        'enemy_shot': _mix(_noise(0.09, 0.55, 0.02, rng), _tone(240, 0.08, 0.35, 0.02)),
        'melee': _mix(_noise(0.12, 0.45, 0.04, rng, 0.4), _tone(90, 0.1, 0.3, 0.04)),
        'pickup': _mix(_tone(880, 0.08, 0.3, 0.04), _pad(_tone(1320, 0.12, 0.25, 0.06), 0.07)),
        'tick': _tone(1600, 0.045, 0.28, 0.012),
        'rumble': _mix(_tone(48, 0.45, 0.6, 0.16), _noise(0.4, 0.25, 0.14, rng, 0.8)),
        'ui': _tone(740, 0.07, 0.28, 0.03),
    }
    for name, samples in recipes.items():
        path = _write_wav(folder / f'{name}.wav', samples)
        voices = 6 if name in ('shot', 'step', 'tick') else 3
        board.add(name, path, voices)
    wind = _mix(_tone(54, 2.4, 0.22), _tone(81, 2.4, 0.08), _noise(2.4, 0.05, 8.0, rng, 0.85))
    wind_path = _write_wav(folder / 'wind.wav', wind)
    try:
        board.ambient = Audio(wind_path, autoplay=True, loop=True, volume=0.08)
    except Exception as exc:
        print('ambient skipped:', exc)
        board.ambient = None
    return board


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------

def _noise_image(size, seed, painter):
    image = Image.new('RGB', (size, size))
    pixels = image.load()
    rng = random.Random(seed)
    coarse_n = 8
    coarse = [[rng.random() for _ in range(coarse_n + 1)] for _ in range(coarse_n + 1)]
    for y in range(size):
        for x in range(size):
            fx = x / size * coarse_n
            fy = y / size * coarse_n
            x0 = int(fx)
            y0 = int(fy)
            tx = fx - x0
            ty = fy - y0
            top = coarse[y0][x0] * (1 - tx) + coarse[y0][x0 + 1] * tx
            bot = coarse[y0 + 1][x0] * (1 - tx) + coarse[y0 + 1][x0 + 1] * tx
            n = top * (1 - ty) + bot * ty
            r, g, b = painter(x, y, n, rng)
            pixels[x, y] = (int(clamp(r, 0, 255)), int(clamp(g, 0, 255)), int(clamp(b, 0, 255)))
    return image


def _finish(image):
    tex = Texture(image, filtering='mipmap')
    tex._texture.setWrapU(SamplerState.WM_repeat)
    tex._texture.setWrapV(SamplerState.WM_repeat)
    tex._texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    tex._texture.setMagfilter(SamplerState.FT_linear)
    return tex


def build_textures():
    def concrete(x, y, n, rng, base, spread):
        v = base + (n - 0.5) * spread
        if rng.random() < 0.03:
            v += rng.uniform(-18, 10)
        if x % 64 == 0 or y % 64 == 0:
            v -= 10
        return v, v * 1.01, v * 1.04

    floor_img = _noise_image(128, 3, lambda x, y, n, rng: concrete(x, y, n, rng, 78, 28))
    wall_img = _noise_image(128, 9, lambda x, y, n, rng: concrete(x, y, n, rng, 58, 22))
    yard_img = _noise_image(128, 5, lambda x, y, n, rng: (
        52 + (n - 0.5) * 20,
        58 + (n - 0.5) * 18,
        66 + (n - 0.5) * 16,
    ))

    def crate_pixel(x, y, n, rng):
        stripe = 42 if (x // 10) % 2 == 0 else 28
        v = 108 + (n - 0.5) * 24 - stripe * 0.15
        return v, v * 0.62, v * 0.32

    crate_img = _noise_image(64, 4, crate_pixel)
    vig = Image.new('RGBA', (256, 256))
    vig_px = vig.load()
    center = 127.5
    for y in range(256):
        for x in range(256):
            dx = (x - center) / center
            dy = (y - center) / center
            radius = math.sqrt(dx * dx + dy * dy)
            alpha = clamp01((radius - 0.42) / 0.75) ** 1.6
            vig_px[x, y] = (255, 255, 255, int(alpha * 255))
    sky = Image.new('RGB', (8, 8), (6, 8, 14))
    return {
        'floor': _finish(floor_img),
        'wall': _finish(wall_img),
        'yard': _finish(yard_img),
        'crate': _finish(crate_img),
        'vignette': Texture(vig),
        'sky': _finish(sky),
    }


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class NavGrid:
    BLOCKED = set('#co')

    def __init__(self, rows):
        self.rows_text = rows
        self.rows = len(rows)
        self.cols = len(rows[0])
        if any(len(row) != self.cols for row in rows):
            lengths = [len(row) for row in rows]
            raise ValueError(f'map rows must be rectangular, got {lengths}')
        self.spawns = []
        self.player_cell = (1, 1)
        self.pickups = []
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                if char == 'S':
                    self.spawns.append((c, r))
                elif char == 'P':
                    self.player_cell = (c, r)
                elif char in ('H', 'A'):
                    self.pickups.append((char, c, r))
        self.player_start = self.center(*self.player_cell)

    def char(self, col, row):
        if col < 0 or row < 0 or col >= self.cols or row >= self.rows:
            return '#'
        return self.rows_text[row][col]

    def walkable(self, col, row):
        return self.char(col, row) not in self.BLOCKED

    def center(self, col, row):
        x = (col - self.cols / 2 + 0.5) * CELL
        z = (row - self.rows / 2 + 0.5) * CELL
        return Vec3(x, 0, z)

    def world_to_cell(self, x, z):
        col = math.floor(x / CELL + self.cols / 2)
        row = math.floor(z / CELL + self.rows / 2)
        return int(col), int(row)

    def nearest(self, cell):
        col, row = cell
        if self.walkable(col, row):
            return (col, row)
        for radius in range(1, 10):
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if max(abs(dc), abs(dr)) != radius:
                        continue
                    nc, nr = col + dc, row + dr
                    if self.walkable(nc, nr):
                        return (nc, nr)
        return None

    def cells_of(self, chars):
        found = []
        for r, row in enumerate(self.rows_text):
            for c, char in enumerate(row):
                if char in chars:
                    found.append((c, r))
        return found

    def neighbors(self, cell):
        col, row = cell
        for dc, dr, cost in (
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ):
            nc, nr = col + dc, row + dr
            if not self.walkable(nc, nr):
                continue
            if dc != 0 and dr != 0:
                if not self.walkable(col + dc, row) or not self.walkable(col, row + dr):
                    continue
            yield (nc, nr), cost

    def astar(self, start, goal):
        if start == goal:
            return [start]
        if start is None or goal is None:
            return []
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, start))
        came = {}
        score = {start: 0}
        closed = set()
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came:
                    current = came[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)
            for neighbor, cost in self.neighbors(current):
                tentative = score[current] + cost
                if tentative >= score.get(neighbor, 1e18):
                    continue
                score[neighbor] = tentative
                came[neighbor] = current
                counter += 1
                heuristic = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                heapq.heappush(frontier, (tentative + heuristic, counter, neighbor))
        return []

    def route(self, start_pos, goal_pos):
        start = self.nearest(self.world_to_cell(start_pos.x, start_pos.z))
        goal = self.nearest(self.world_to_cell(goal_pos.x, goal_pos.z))
        cells = self.astar(start, goal)
        return [self.center(col, row) for col, row in cells]


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------

def _merge(mask):
    rows = len(mask)
    cols = len(mask[0])
    used = [[False] * cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c] or used[r][c]:
                continue
            c2 = c
            while c2 + 1 < cols and mask[r][c2 + 1] and not used[r][c2 + 1]:
                c2 += 1
            r2 = r
            while r2 + 1 < rows and all(mask[r2 + 1][cc] and not used[r2 + 1][cc] for cc in range(c, c2 + 1)):
                r2 += 1
            for rr in range(r, r2 + 1):
                for cc in range(c, c2 + 1):
                    used[rr][cc] = True
            rects.append((c, r, c2, r2))
    return rects


def _box(c0, r0, c1, r1, y_center, height, texture, tint, tag):
    width = (c1 - c0 + 1) * CELL
    depth = (r1 - r0 + 1) * CELL
    cx = ((c0 + c1) / 2 - nav.cols / 2 + 0.5) * CELL
    cz = ((r0 + r1) / 2 - nav.rows / 2 + 0.5) * CELL
    entity = Entity(
        parent=level_root, model='cube', position=(cx, y_center, cz),
        scale=(width, height, depth), texture=texture, color=tint, collider='box',
    )
    entity.texture_scale = (max(1.0, width / 2.2), max(1.0, depth / 2.2))
    entity.is_level = True
    entity.name = tag
    return entity


def build_level(textures):
    global level_root
    level_root = Entity(name='level')
    floor = Entity(
        parent=level_root, model='cube', position=(0, -0.12, 0),
        scale=(nav.cols * CELL, 0.24, nav.rows * CELL),
        texture=textures['floor'], color=color.rgb(1.15, 1.18, 1.22), collider='box',
    )
    floor.texture_scale = (nav.cols * 0.85, nav.rows * 0.85)
    floor.is_level = True

    wall_mask = [[nav.char(c, r) == '#' for c in range(nav.cols)] for r in range(nav.rows)]
    ceil_mask = [[nav.char(c, r) not in '#Y' for c in range(nav.cols)] for r in range(nav.rows)]
    yard_mask = [[nav.char(c, r) == 'Y' for c in range(nav.cols)] for r in range(nav.rows)]

    for c0, r0, c1, r1 in _merge(wall_mask):
        wall = _box(c0, r0, c1, r1, WALL_H / 2, WALL_H, textures['wall'], color.rgb(0.95, 0.98, 1.05), 'wall')
        wall.texture_scale = (max(1, wall.scale_x / 2.4), max(1, WALL_H / 2.2))
    for c0, r0, c1, r1 in _merge(ceil_mask):
        slab = _box(
            c0, r0, c1, r1, WALL_H + 0.12, 0.24,
            textures['wall'], color.rgb(0.72, 0.75, 0.82), 'ceiling',
        )
        slab.scale = (slab.scale_x + 0.06, slab.scale_y, slab.scale_z + 0.06)
    for c0, r0, c1, r1 in _merge(yard_mask):
        pad = _box(c0, r0, c1, r1, 0.015, 0.03, textures['yard'], color.rgb(0.7, 0.74, 0.8), 'yard')
        pad.texture_scale = (max(1, pad.scale_x / 3), max(1, pad.scale_z / 3))

    lights = []
    for r, row in enumerate(nav.rows_text):
        for c, char in enumerate(row):
            pos = nav.center(c, r)
            if char == 'c':
                crate = Entity(
                    parent=level_root, model='cube', position=(pos.x, 0.52, pos.z),
                    scale=(1.7, 1.04, 1.7), texture=textures['crate'],
                    color=color.rgb(0.9, 0.78, 0.62), collider='box',
                )
                crate.texture_scale = (2, 2)
                crate.is_level = True
            elif char == 'o':
                pillar = Entity(
                    parent=level_root, model=Cylinder(resolution=10, start=0, height=1),
                    position=(pos.x, 0, pos.z), scale=(0.9, WALL_H, 0.9), texture=textures['wall'],
                    color=color.rgb(0.55, 0.58, 0.64), collider='box',
                )
                pillar.is_level = True
            elif char == '+':
                lintel = Entity(
                    parent=level_root, model='cube', position=(pos.x, WALL_H - 0.18, pos.z),
                    scale=(CELL * 0.72, 0.07, 0.16), color=ACCENT, shader=unlit_shader,
                )
                lintel.name = 'lintel'
            elif char == '*':
                fixture = Entity(
                    parent=level_root, model='cube', position=(pos.x, WALL_H - 0.08, pos.z),
                    scale=(1.1, 0.05, 0.28), color=color.rgb32(255, 220, 185), shader=unlit_shader,
                )
                fixture.name = 'fixture'
                glow = Entity(
                    parent=level_root, model='cube', position=(pos.x, WALL_H - 0.22, pos.z),
                    scale=(0.7, 0.02, 0.16), color=color.rgb32(255, 196, 140), shader=unlit_shader,
                )
                lights.append((pos.x, pos.z, fixture))
    return lights


# ---------------------------------------------------------------------------
# Short-lived effects
# ---------------------------------------------------------------------------

class Spark(Entity):
    def __init__(self, pos, velocity, tint):
        super().__init__(model='sphere', position=pos, scale=0.07, color=tint, shader=unlit_shader)
        self.vel = Vec3(velocity)
        self.life = 0.16
        self.is_fx = True

    def update(self):
        if getattr(self, '_retired', False) or game.state != 'playing':
            return
        dt = frame_dt()
        self.position += self.vel * dt
        self.life -= dt
        self.scale = max(0.001, 0.08 * (self.life / 0.16))
        if self.life <= 0:
            retire(self)


class Tracer(Entity):
    def __init__(self, start, end, tint):
        super().__init__(model='cube', color=tint, shader=unlit_shader)
        self.is_fx = True
        self.position = (start + end) * 0.5
        self.look_at(end)
        self.scale = (0.015, 0.015, max(0.05, (end - start).length()))
        self.life = 0.04

    def update(self):
        self.life -= time.dt
        if self.life <= 0:
            retire(self)


class Shell(Entity):
    def __init__(self, pos, velocity):
        super().__init__(
            model='cube', position=pos, scale=(0.016, 0.016, 0.032),
            color=color.rgb32(212, 176, 78), shader=unlit_shader,
            rotation=(random.random() * 80, random.random() * 180, 0),
        )
        self.vel = Vec3(velocity)
        self.life = 2.4
        self.is_fx = True

    def update(self):
        if game.state != 'playing':
            return
        dt = frame_dt()
        self.vel.y -= 14 * dt
        self.position += self.vel * dt
        if self.y < 0.04:
            self.y = 0.04
            self.vel.y *= -0.28
            self.vel.x *= 0.55
            self.vel.z *= 0.55
        self.rotation_x += 520 * dt
        self.life -= dt
        if self.life <= 0:
            retire(self)


class DamageNumber(Entity):
    def __init__(self, pos, text, tint, scale):
        super().__init__(parent=camera.ui)
        self.is_fx = True
        self.world = Vec3(pos)
        self.life = 0.7
        self.label = ui_text(self, text, px=scale, color=tint)

    def update(self):
        if game.state not in ('playing', 'paused'):
            retire(self)
            return
        dt = frame_dt()
        self.life -= dt
        self.world.y += 0.85 * dt
        if self.life <= 0 or not in_front_of_camera(self.world):
            retire(self)
            return
        self.position = world_to_ui(self.world)
        tint = self.label.color
        self.label.color = color.rgba(tint.r, tint.g, tint.b, clamp01(self.life * 1.6))


def burst(pos, tint, count=4):
    for _ in range(count):
        velocity = Vec3(random.uniform(-1.4, 1.4), random.uniform(0.2, 2.2), random.uniform(-1.4, 1.4))
        Spark(pos, velocity, tint)


def add_decal(pos, normal):
    decal = Entity(
        parent=level_root, model='quad', position=pos + normal * 0.025,
        scale=random.uniform(0.09, 0.16), color=color.rgba32(18, 16, 14, 210), shader=unlit_shader,
    )
    decal.look_at(pos + normal * 3)
    decal.life = 14
    decal.is_fx = True
    game.decals.append(decal)
    if len(game.decals) > 28:
        retire(game.decals.pop(0))


# ---------------------------------------------------------------------------
# Weapon and player
# ---------------------------------------------------------------------------

class Rifle:
    def __init__(self, owner):
        self.owner = owner
        self.mag = MAG_SIZE
        self.reserve = RESERVE_START
        self.cooldown = 0
        self.dry_lock = 0
        self.reloading = False
        self.reload_t = 0
        self.ads = False
        self.ads_blend = 0
        self.bloom = 0
        self.gun_kick = 0
        self.sway = Vec3(0, 0, 0)
        self.flash_t = 0
        self.viewmodel = Entity(parent=camera, position=(0.46, -0.4, 0.86))
        metal = color.rgb32(34, 36, 40)
        dark = color.rgb32(18, 19, 22)
        edge = color.rgb32(168, 132, 86)
        self.parts = []

        def part(pos, scale, tint, rot=(0, 0, 0)):
            entity = Entity(
                parent=self.viewmodel, model='cube', position=pos, scale=scale,
                color=tint, shader=unlit_shader, rotation=rot,
            )
            self.parts.append(entity)
            return entity

        part((0, 0, 0.02), (0.055, 0.07, 0.34), metal)
        part((0, 0.038, 0.34), (0.022, 0.024, 0.46), dark)
        part((0, -0.01, -0.22), (0.045, 0.055, 0.2), metal)
        part((0, -0.08, -0.02), (0.03, 0.1, 0.04), dark, rot=(18, 0, 0))
        self.mag = part((0, -0.09, 0.05), (0.036, 0.11, 0.055), color.rgb32(22, 24, 28))
        self.mag_home = self.mag.y
        part((0, 0.055, 0.04), (0.012, 0.02, 0.08), edge)
        part((0, 0.05, 0.52), (0.008, 0.018, 0.012), color.rgb32(255, 196, 120))
        self.muzzle = Entity(parent=self.viewmodel, position=(0, 0.038, 0.58))
        self.flash = Entity(
            parent=self.muzzle, model='sphere', scale=0.045,
            color=color.rgb32(255, 220, 170), shader=unlit_shader, enabled=False,
        )
        self.last_hit = ''

    def reset(self):
        self.mag_count = MAG_SIZE
        self.reserve = RESERVE_START
        self.cooldown = 0.18
        self.dry_lock = 0
        self.reloading = False
        self.reload_t = 0
        self.ads = False
        self.bloom = 0
        self.gun_kick = 0

    @property
    def mag_rounds(self):
        return self.mag_count

    def spread_now(self):
        speed = math.hypot(self.owner.vel_x, self.owner.vel_z)
        move = (speed / SPRINT_SPEED) * 0.02
        base = 0.0022 + self.bloom + move
        if self.owner.crouching and self.owner.grounded:
            base *= 0.72
        if self.ads:
            base *= 0.42
        if not self.owner.grounded:
            base += 0.01
        return base

    def reload(self):
        if self.reloading or self.mag_count >= MAG_SIZE or self.reserve <= 0:
            return
        self.reloading = True
        self.reload_t = RELOAD_TIME
        sounds.play('reload', 0.62, pitch=random.uniform(0.97, 1.03))

    def try_fire(self):
        if game.state != 'playing' or self.cooldown > 0 or self.reloading:
            return False
        if self.mag_count <= 0:
            if self.reserve > 0:
                self.reload()
            elif self.dry_lock <= 0:
                sounds.play('dry', 0.45, pitch=random.uniform(0.96, 1.04))
                self.dry_lock = 0.28
            return False
        self.mag_count -= 1
        self.cooldown = FIRE_INTERVAL
        self._fire()
        return True

    def _fire(self):
        spread = self.spread_now()
        direction = camera.forward + camera.right * random.uniform(-spread, spread) + camera.up * random.uniform(-spread, spread)
        direction = norm(direction)
        muzzle = self.muzzle.world_position
        hit = raycast(camera.world_position, direction, distance=80, ignore=[player])
        self.bloom = min(0.045, self.bloom + 0.011)
        self.gun_kick = 0.075
        self.flash_t = 0.045
        self.owner.add_recoil(1.05)
        sounds.play('shot', 0.72, pitch=random.uniform(0.96, 1.04))
        game.noise_pos = Vec3(player.x, 0, player.z)
        game.noise_t = 0.32
        if player.muzzle_light:
            player.muzzle_light.color = color.rgb(1.6, 1.15, 0.7)
            player.muzzle_light_t = 0.045
        end = camera.world_position + direction * 42
        self.last_hit = 'none'
        if hit.hit:
            end = hit.world_point
            victim = victim_from(hit.entity)
            if victim is not None and not victim.dead:
                headshot = hit.world_point.y >= victim.y + 1.48 * victim.bulk
                victim.take_damage(RIFLE_DAMAGE, headshot, hit.world_point + Vec3(0, 0.15, 0))
                burst(hit.world_point, color.rgb32(255, 70, 54), 5)
                self.last_hit = victim.kind
                game.hud.mark_hit()
            else:
                normal = hit.world_normal if hit.world_normal is not None else Vec3(0, 1, 0)
                burst(hit.world_point, color.rgb32(190, 186, 176), 3)
                if normal.length() > 0.1:
                    add_decal(hit.world_point, norm(normal))
                self.last_hit = getattr(hit.entity, 'name', 'world')
        Tracer(muzzle, end, color.rgba32(255, 220, 170, 230))
        shell_pos = muzzle + camera.right * 0.14 + camera.up * 0.02
        shell_vel = camera.right * 2.8 + camera.up * 2.2 - camera.forward * 0.25
        shell_vel += Vec3(random.uniform(-0.15, 0.15), random.uniform(0, 0.25), random.uniform(-0.15, 0.15))
        Shell(shell_pos, shell_vel)

    def update(self, dt):
        self.cooldown = max(0, self.cooldown - dt)
        self.dry_lock = max(0, self.dry_lock - dt)
        self.bloom = damp(self.bloom, 0, 9, dt)
        self.gun_kick = damp(self.gun_kick, 0, 18, dt)
        self.ads = bool(held_keys['right mouse']) and not self.reloading
        self.ads_blend = damp(self.ads_blend, 1 if self.ads else 0, 12, dt)
        if self.reloading:
            self.reload_t -= dt
            if self.reload_t <= 0:
                take = min(MAG_SIZE - self.mag_count, self.reserve)
                self.mag_count += take
                self.reserve -= take
                self.reloading = False
                self.reload_t = 0
        if held_keys['left mouse']:
            self.try_fire()
        progress = 0
        if self.reloading:
            progress = 1 - (self.reload_t / RELOAD_TIME)
        hip = Vec3(0.34, -0.36, 0.86)
        ads = Vec3(0.0, -0.292, 0.58)
        base = lerp(hip, ads, self.ads_blend)
        mx = mouse.velocity[0] if mouse.locked else 0
        my = mouse.velocity[1] if mouse.locked else 0
        target_sway = Vec3(clamp(-mx * 0.55, -0.055, 0.055), clamp(my * 0.4, -0.04, 0.04), 0)
        self.sway = damp(self.sway, target_sway, 9, dt)
        reload_rx, reload_ry, reload_rz, mag_drop = self._reload_pose(progress)
        bob = math.sin(self.owner.phase) * 0.01 if self.owner.grounded else 0
        self.viewmodel.position = base + self.sway + Vec3(0, bob, -self.gun_kick)
        self.viewmodel.rotation = (
            reload_rx + self.gun_kick * 90 + (-self.owner.recoil) * 0.35,
            reload_ry + self.sway.x * -40,
            reload_rz + self.owner.roll * 0.35,
        )
        self.mag.y = self.mag_home - mag_drop
        self.flash.enabled = self.flash_t > 0
        self.flash_t = max(0, self.flash_t - dt)

    def _reload_pose(self, progress):
        if progress <= 0:
            return 0, 0, 0, 0
        if progress < 0.32:
            k = progress / 0.32
            return 32 * k, 10 * k, 16 * k, 0.16 * k
        if progress < 0.62:
            return 32, 10, 16, 0.16
        k = (progress - 0.62) / 0.38
        return 32 * (1 - k), 10 * (1 - k), 16 * (1 - k), 0.16 * (1 - k)


class Player(Entity):
    def __init__(self):
        super().__init__(position=nav.player_start)
        self.pivot = Entity(parent=self, y=STAND_HEIGHT - 0.12)
        self.shake_pivot = Entity(parent=self.pivot)
        self.vel_x = 0
        self.vel_z = 0
        self.vel_y = 0
        self.hp = 100
        self.max_hp = 100
        self.iframe = 0
        self.hurt_flash = 0
        self.shake = 0
        self.stamina = 100
        self.sprint_lock = False
        self.stam_delay = 0
        self.sprinting = False
        self.grounded = False
        self.coyote = 0
        self.jump_buffer = 0
        self.crouching = False
        self.current_height = STAND_HEIGHT
        self.land_dip = 0
        self.phase = 0
        self.step_accum = 0
        self.yaw = 0
        self.pitch = 3
        self.recoil = 0
        self.yaw_kick = 0
        self.roll = 0
        self.bob_x = 0
        self.bob_y = 0
        self.look_enabled = True
        self.light_on = True
        self.muzzle_light_t = 0
        camera.parent = self.shake_pivot
        camera.position = Vec3(0, 0, 0)
        camera.rotation = Vec3(0, 0, 0)
        camera.fov = HIP_FOV
        camera.clip_plane_near = 0.045
        camera.clip_plane_far = 130
        self.rifle = Rifle(self)
        self.spot = SpotLight(parent=camera, color=color.rgb(1.35, 1.28, 1.15), rotation=(0, 0, 0))
        self.spot._light.setAttenuation((1, 0.035, 0.008))
        self.spot._light.setExponent(28)
        lens = self.spot._light.getLens()
        lens.setNearFar(0.12, 30)
        lens.setFov(46)
        self.beam = Entity(
            parent=camera, model=Cone(resolution=14, radius=0.5, height=1),
            position=(0, -0.05, 1.15), rotation_x=90,
            scale=(0.42, 2.1, 0.42), color=color.rgba32(190, 210, 255, 18), shader=unlit_shader,
        )
        self.muzzle_light = PointLight(parent=camera, position=(0.15, -0.12, 0.8), color=color.black)
        self.muzzle_light._light.setAttenuation((1, 0.2, 1.4))
        self.apply_light()

    def attach_camera(self):
        camera.parent = self.shake_pivot
        camera.position = Vec3(0, 0, 0)
        camera.rotation = Vec3(0, 0, 0)
        camera.fov = HIP_FOV

    def detach_camera(self):
        camera.parent = scene

    def apply_light(self):
        active = self.light_on and game is not None and game.state == 'playing'
        self.spot.color = color.rgb(1.45, 1.36, 1.2) if active else color.black
        self.beam.enabled = active

    def add_recoil(self, kick):
        # Most of the kick stays on the gun. A little climbs the view so you can still track.
        self.pitch -= kick * 0.11
        self.pitch = clamp(self.pitch, -87, 87)
        self.recoil -= kick * 0.55
        self.yaw_kick += random.uniform(-kick * 0.1, kick * 0.1)

    def revive(self, pos):
        self.position = Vec3(pos.x, 0.02, pos.z)
        self.vel_x = self.vel_z = self.vel_y = 0
        self.hp = self.max_hp
        self.iframe = 0
        self.hurt_flash = 0
        self.shake = 0
        self.stamina = 100
        self.sprint_lock = False
        self.stam_delay = 0
        self.sprinting = False
        self.grounded = False
        self.coyote = 0.12
        self.jump_buffer = 0
        self.crouching = False
        self.current_height = STAND_HEIGHT
        self.land_dip = 0
        self.phase = 0
        self.step_accum = 0
        self.recoil = 0
        self.yaw_kick = 0
        self.roll = 0
        self.pitch = 3
        self.look_at(self.position + Vec3(0, 0, -1))
        self.yaw = self.rotation_y
        self.rotation_x = 0
        self.rotation_z = 0
        self.rifle.reset()
        self.light_on = True
        self.viewmodel_enabled(True)
        self.attach_camera()
        self.apply_light()

    def viewmodel_enabled(self, enabled):
        self.rifle.viewmodel.enabled = enabled

    def hurt(self, amount, source=None):
        if game.state != 'playing' or self.iframe > 0 or self.hp <= 0:
            return
        self.hp = max(0, self.hp - amount)
        self.iframe = 0.16
        self.hurt_flash = min(1.0, 0.38 + amount / 70)
        self.shake = min(1.5, self.shake + amount / 48)
        sounds.play('hurt', 0.7, pitch=random.uniform(0.94, 1.06))
        if source is not None:
            game.hud.ping(source)
        if self.hp <= 0:
            game.on_death()

    def input(self, key):
        if game.state != 'playing':
            return
        if key == 'space':
            self.jump_buffer = 0.12
        elif key == 'r':
            self.rifle.reload()
        elif key == 'f':
            self.light_on = not self.light_on
            self.apply_light()

    def update(self):
        if game is None or game.state != 'playing':
            return
        dt = frame_dt()
        self._look(dt)
        self._move(dt)
        self._bob(dt)
        self.rifle.update(dt)
        target_fov = ADS_FOV if self.rifle.ads else (SPRINT_FOV if self.sprinting else HIP_FOV)
        camera.fov = damp(camera.fov, target_fov, 8, dt)
        self.shake = max(0, self.shake - dt * 2.3)
        if self.shake > 0.02:
            mag = self.shake * 0.05
            self.shake_pivot.position = (random.uniform(-mag, mag), random.uniform(-mag, mag) * 0.65, 0)
        else:
            self.shake_pivot.position = (0, 0, 0)
        self.iframe = max(0, self.iframe - dt)
        self.hurt_flash = max(0, self.hurt_flash - dt * 1.5)
        self.muzzle_light_t = max(0, self.muzzle_light_t - dt)
        if self.muzzle_light_t <= 0 and self.muzzle_light is not None:
            self.muzzle_light.color = color.black

    def _look(self, dt):
        if self.look_enabled and mouse.locked:
            sens = game.sensitivity
            self.yaw += mouse.velocity[0] * sens
            self.pitch -= mouse.velocity[1] * sens
            self.pitch = clamp(self.pitch, -87, 87)
        self.recoil = damp(self.recoil, 0, 9.5, dt)
        self.yaw_kick = damp(self.yaw_kick, 0, 8, dt)
        self.rotation_y = self.yaw + self.yaw_kick
        self.rotation_x = 0
        self.rotation_z = 0
        self.pivot.rotation_x = self.pitch + self.recoil
        self.pivot.y = self.current_height - 0.12

    def _wish(self):
        wish = Vec3(0, 0, 0)
        forward = Vec3(self.forward.x, 0, self.forward.z)
        right = Vec3(self.right.x, 0, self.right.z)
        if held_keys['w']:
            wish += forward
        if held_keys['s']:
            wish -= forward
        if held_keys['d']:
            wish += right
        if held_keys['a']:
            wish -= right
        return norm(wish)

    def _move(self, dt):
        wish = self._wish()
        moving = wish.length() > 0.1
        crouch_held = bool(held_keys['left control'] or held_keys['right control'] or held_keys['c'])
        if not crouch_held:
            head = self.world_position + Vec3(0, CROUCH_HEIGHT, 0)
            if level_blocked(head, Vec3(0, 1, 0), STAND_HEIGHT - CROUCH_HEIGHT):
                crouch_held = True
        self.crouching = crouch_held
        target_h = CROUCH_HEIGHT if self.crouching else STAND_HEIGHT
        self.current_height = damp(self.current_height, target_h, 14, dt)
        shift = bool(held_keys['left shift'] or held_keys['right shift'])
        self.sprinting = (
            moving and shift and self.grounded and not self.crouching
            and not self.sprint_lock and self.stamina > 0 and not self.rifle.ads
        )
        if self.sprinting:
            self.stamina = max(0, self.stamina - 28 * dt)
            self.stam_delay = 0.4
            if self.stamina <= 0:
                self.sprint_lock = True
        else:
            self.stam_delay = max(0, self.stam_delay - dt)
            if self.stam_delay <= 0:
                self.stamina = min(100, self.stamina + 22 * dt)
            if self.stamina >= 18:
                self.sprint_lock = False
        speed = CROUCH_SPEED if self.crouching else (SPRINT_SPEED if self.sprinting else WALK_SPEED)
        if self.rifle.ads:
            speed *= 0.64
        accel = 64 * dt if self.grounded else 18 * dt
        self.vel_x = approach(self.vel_x, wish.x * speed, accel)
        self.vel_z = approach(self.vel_z, wish.z * speed, accel)
        if self.grounded and not moving:
            self.vel_x = damp(self.vel_x, 0, 12, dt)
            self.vel_z = damp(self.vel_z, 0, 12, dt)
        if not self._try_axis(self.vel_x * dt, 0):
            self.vel_x = 0
        if not self._try_axis(0, self.vel_z * dt):
            self.vel_z = 0
        self._vertical(dt)
        speed_now = math.hypot(self.vel_x, self.vel_z)
        if self.grounded and speed_now > 1.1:
            self.step_accum += speed_now * dt
            stride = 2.55 if self.sprinting else 2.05
            if self.step_accum >= stride:
                self.step_accum = 0
                sounds.play('step', 0.46 if self.sprinting else 0.28, pitch=random.uniform(0.9, 1.12))
        self.jump_buffer = max(0, self.jump_buffer - dt)
        if self.jump_buffer > 0 and (self.grounded or self.coyote > 0) and self.vel_y <= 0.2:
            self.vel_y = JUMP_SPEED
            self.grounded = False
            self.coyote = 0
            self.jump_buffer = 0

    def _try_axis(self, dx, dz):
        step = Vec3(dx, 0, dz)
        dist = step.length()
        if dist < 1e-7:
            return True
        direction = step / dist
        reach = dist + PLAYER_RADIUS
        low = self.world_position + Vec3(0, 0.4, 0)
        high = self.world_position + Vec3(0, max(0.95, self.current_height - 0.2), 0)
        if level_blocked(low, direction, reach) or level_blocked(high, direction, reach):
            return False
        hit = raycast(low, direction, distance=reach, ignore=[self])
        if hit.hit:
            victim = victim_from(hit.entity)
            if victim is not None and not victim.dead and victim.spawning <= 0:
                victim.knock += Vec3(direction.x, 0, direction.z) * 5
                self.position += step * 0.32
                return False
            if getattr(hit.entity, 'is_level', False):
                return False
        self.position += step
        return True

    def _vertical(self, dt):
        self.vel_y -= GRAVITY * dt
        if self.vel_y > 0:
            head = self.world_position + Vec3(0, self.current_height, 0)
            if level_blocked(head, Vec3(0, 1, 0), self.vel_y * dt + 0.05):
                self.vel_y = 0
            else:
                self.y += self.vel_y * dt
            if self.grounded:
                self.coyote = 0.1
            self.grounded = False
            self.coyote = max(0, self.coyote - dt)
            return
        origin = self.world_position + Vec3(0, 0.18, 0)
        reach = 0.36 if self.grounded else max(0.22, -self.vel_y * dt + 0.22)
        hit = raycast(origin, Vec3(0, -1, 0), distance=reach, traverse_target=level_root)
        if hit.hit and hit.world_normal.y > 0.45:
            fall = -self.vel_y
            self.y = hit.world_point.y
            if not self.grounded and fall > 4.2:
                self.land_dip = clamp(fall * 0.011, 0.02, 0.09)
                sounds.play('step', 0.5, pitch=random.uniform(0.75, 0.9))
            self.vel_y = 0
            self.grounded = True
            self.coyote = 0.11
        else:
            was = self.grounded
            self.y += self.vel_y * dt
            self.grounded = False
            if was:
                self.coyote = 0.11
            else:
                self.coyote = max(0, self.coyote - dt)

    def _bob(self, dt):
        strafe = float(held_keys['d']) - float(held_keys['a'])
        self.roll = damp(self.roll, -strafe * (0.7 if self.rifle.ads else 2.0), 8, dt)
        speed = math.hypot(self.vel_x, self.vel_z)
        if self.grounded and speed > 0.7:
            freq = 11.0 if self.sprinting else 8.4
            self.phase += dt * freq * (speed / WALK_SPEED)
            amp = 0.048 if self.sprinting else 0.03
            target_y = math.sin(self.phase) * amp
            target_x = math.cos(self.phase * 0.5) * 0.016
        else:
            target_y = 0
            target_x = 0
        self.bob_x = damp(self.bob_x, target_x, 10, dt)
        self.bob_y = damp(self.bob_y, target_y, 10, dt)
        self.land_dip = damp(self.land_dip, 0, 7, dt)
        camera.position = (self.bob_x, self.bob_y - self.land_dip, 0)
        camera.rotation = (0, 0, self.roll)


# ---------------------------------------------------------------------------
# Enemies
# ---------------------------------------------------------------------------

class Projectile(Entity):
    def __init__(self, pos, direction, damage):
        super().__init__(
            model='sphere', position=pos, scale=0.16,
            color=color.rgb32(255, 176, 70), shader=unlit_shader,
        )
        self.direction = norm(direction)
        self.speed = 17.5
        self.damage = damage
        self.life = 2.4
        self.is_projectile = True
        game.projectiles.append(self)
        glow = Entity(parent=self, model='sphere', scale=1.8, color=color.rgba32(255, 140, 40, 90), shader=unlit_shader)
        glow.name = 'glow'

    def update(self):
        if getattr(self, '_retired', False) or game.state != 'playing':
            return
        dt = frame_dt()
        dist = self.speed * dt
        start = Vec3(self.world_position)
        hit = raycast(start, self.direction, distance=dist + 0.05, traverse_target=level_root)
        travel = dist if not hit.hit else max(0, hit.distance - 0.04)
        self.position += self.direction * travel
        end = Vec3(self.world_position)
        if _segment_hits_player(start, end):
            player.hurt(self.damage, self)
            burst(end, color.rgb32(255, 170, 80), 4)
            retire(self)
            return
        if hit.hit:
            burst(hit.world_point, color.rgb32(255, 186, 90), 3)
            retire(self)
            return
        self.life -= dt
        if self.life <= 0:
            retire(self)


def _capsule_contains(point):
    feet = player.y + 0.15
    top = player.y + player.current_height * 0.92
    cy = clamp(point.y, feet, top)
    dx = point.x - player.x
    dy = point.y - cy
    dz = point.z - player.z
    return dx * dx + dy * dy + dz * dz < 0.42 ** 2


def _segment_hits_player(start, end):
    for t in (0.0, 0.33, 0.66, 1.0):
        point = lerp(start, end, t)
        if _capsule_contains(point):
            return True
    return False


class Enemy(Entity):
    def __init__(self, kind, pos, hp_mul=1, speed_mul=1):
        spec = ARCHETYPES[kind]
        super().__init__(position=Vec3(pos.x, 0, pos.z), name=f'enemy_{kind}')
        self.kind = kind
        self.bulk = spec['bulk']
        self.speed = spec['speed'] * speed_mul
        self.sight = spec['sight']
        self.max_hp = spec['hp'] * hp_mul
        self.hp = self.max_hp
        self.body_color = color.rgb32(*spec['body'])
        self.eye_color = color.rgb32(*spec['eye'])
        self.alert_pitch = spec['alert']
        self.is_enemy = True
        self.dead = False
        self.spawning = 0.45
        self.see = False
        self.alerted = False
        self.lost_t = 0
        self.search_wait = 0
        self.last_known = None
        self.path = []
        self.wp = 0
        self.goal = None
        self.path_cd = random.uniform(0, 0.2)
        self.attack_cd = random.uniform(0.2, 0.6)
        self.knock = Vec3(0, 0, 0)
        self.stuck = 0
        self.last_pos = Vec3(self.position)
        self.anim_speed = 0
        self.phase = random.random() * 6
        self.swinging = False
        self.swing_t = 0
        self.swing_windup = 0.3
        self.swing_hit = False
        self.mode = 'approach'
        self.timer = 0
        self.charge_dir = Vec3(0, 0, 1)
        self.charge_hit = False
        self.strafe_dir = random.choice((-1, 1))
        self.strafe_t = random.uniform(0.4, 1.2)
        self.flash_t = 0
        self._build_rig()
        self.collider = BoxCollider(
            self,
            center=Vec3(0, 0.86 * self.bulk, 0),
            size=Vec3(0.66 * self.bulk, 1.72 * self.bulk, 0.5 * self.bulk),
        )
        game.enemies.append(self)
        burst(self.position + Vec3(0, 1.0, 0), self.eye_color, 6)

    def _build_rig(self):
        b = self.bulk
        leg = color.rgb(self.body_color.r * 0.55, self.body_color.g * 0.55, self.body_color.b * 0.55)
        self.body = Entity(
            parent=self, model='cube', position=(0, 0.98 * b, 0),
            scale=(0.62 * b, 0.78 * b, 0.36 * b), color=self.body_color,
        )
        self.head = Entity(
            parent=self, model='sphere', position=(0, 1.52 * b, 0.04 * b),
            scale=0.32 * b, color=color.rgb(self.body_color.r * 0.85, self.body_color.g * 0.85, self.body_color.b * 0.85),
        )
        self.head.owner = self
        self.eye_l = Entity(
            parent=self, model='sphere', position=(-0.07 * b, 1.56 * b, 0.16 * b),
            scale=0.055 * b, color=self.eye_color, shader=unlit_shader,
        )
        self.eye_r = duplicate(self.eye_l, x=0.07 * b)
        self.leg_l = Entity(
            parent=self, model='cube', position=(-0.14 * b, 0.78 * b, 0), origin_y=0.5,
            scale=(0.16 * b, 0.78 * b, 0.16 * b), color=leg,
        )
        self.leg_r = duplicate(self.leg_l, x=0.14 * b)
        self.arm_l = Entity(
            parent=self, model='cube', position=(-0.38 * b, 1.22 * b, 0), origin_y=0.5,
            scale=(0.12 * b, 0.62 * b, 0.12 * b), color=leg,
        )
        self.arm_r = duplicate(self.arm_l, x=0.38 * b)
        if self.kind == 'brute':
            Entity(
                parent=self, model='cube', position=(0, 1.18 * b, 0),
                scale=(0.92 * b, 0.28 * b, 0.42 * b), color=self.body_color,
            )
        elif self.kind == 'shooter':
            Entity(
                parent=self, model='cube', position=(0.34 * b, 1.02 * b, 0.22 * b),
                scale=(0.05 * b, 0.05 * b, 0.46 * b), color=color.rgb(0.12, 0.13, 0.15),
            )
        self.tell = Entity(
            parent=self, model=Circle(resolution=28, radius=0.5, mode='line', thickness=3),
            rotation_x=90, y=0.045, scale=1.1, color=self.eye_color, shader=unlit_shader, enabled=False,
        )
        self.beacon = Entity(
            parent=self, model='cube', position=(0, 1.35 * b, 0),
            scale=(0.055 * b, 2.5, 0.055 * b), color=self.eye_color, shader=unlit_shader,
        )
        self.bar_bg = Entity(
            model='quad', scale=(0.74, 0.055), color=color.rgba32(0, 0, 0, 180),
            billboard=True, shader=unlit_shader, enabled=False,
        )
        self.bar = Entity(
            parent=self.bar_bg, model='quad', origin=(-0.5, 0), position=(-0.5, 0, -0.01),
            scale=(1, 0.62), color=self.eye_color, shader=unlit_shader,
        )
        self.bar_bg.enabled = False

    def take_damage(self, amount, headshot, hit_pos):
        if self.dead:
            return
        dealt = amount * (HEADSHOT_MULT if headshot else 1)
        self.hp -= dealt
        self.flash_t = 0.07
        self.body.color = color.rgb(1.0, 0.86, 0.8)
        if self.bar_bg:
            self.bar_bg.enabled = True
        tint = color.rgb32(255, 214, 90) if headshot else color.rgb32(245, 245, 245)
        DamageNumber(hit_pos, str(int(round(dealt))), tint, 26 if headshot else 20)
        sounds.play('tick', 0.28, pitch=1.45 if headshot else 1.12)
        away = Vec3(self.x - player.x, 0, self.z - player.z)
        if away.length() > 0.05:
            self.knock += norm(away) * (0.45 if self.kind == 'brute' else 1.35)
        self.last_known = Vec3(player.x, 0, player.z)
        self.alerted = True
        if self.hp <= 0:
            self.die()

    def die(self):
        if self.dead:
            return
        self.dead = True
        self.collision = False
        if self in game.enemies:
            game.enemies.remove(self)
        game.kills += 1
        game.hud.toast(f'{self.kind.upper()} DOWN')
        sounds.play('tick', 0.4, pitch=0.8)
        self.body.color = self.body_color
        self.tell.enabled = False
        self.beacon.enabled = False
        if self.bar_bg:
            retire(self.bar_bg)
            self.bar_bg = None
            self.bar = None
        self.death_t = 0

    def update(self):
        if getattr(self, '_retired', False):
            return
        dt = frame_dt()
        if self.dead:
            self._death(dt)
            return
        if game.state != 'playing':
            return
        if self.spawning > 0:
            self.spawning -= dt
            arrived = 1 - clamp(self.spawning / 0.45, 0, 1)
            self.body.color = lerp(color.rgb(0.05, 0.05, 0.06), self.body_color, arrived)
            self.beacon.enabled = True
            self.beacon.scale_y = 0.4 + arrived * 2.2
            if self.spawning <= 0:
                self.body.color = self.body_color
                self.beacon.enabled = False
            return
        self.beacon.enabled = False
        self.attack_cd = max(0, self.attack_cd - dt)
        self._perceive(dt)
        self._act(dt)
        self._animate(dt)
        self._bar()
        self._unstick(dt)

    def _death(self, dt):
        self.death_t += dt
        self.rotation_z = damp(self.rotation_z, 0, 10, dt)
        self.rotation_x = damp(self.rotation_x, 82, 6, dt)
        self.y = max(-0.45, self.y - dt * 0.45)
        if self.death_t > 1.35:
            retire(self)

    def _perceive(self, dt):
        eye = self.position + Vec3(0, 1.4 * self.bulk, 0)
        target = player.position + Vec3(0, max(0.75, player.current_height * 0.66), 0)
        dist = distance_xz(self, player)
        self.see = dist <= self.sight and line_of_sight(eye, target)
        if self.see:
            self.last_known = Vec3(player.x, 0, player.z)
            self.search_wait = 0
            self.lost_t = 0
            if not self.alerted:
                self.alerted = True
                sounds.play('alert', 0.42, pitch=self.alert_pitch * random.uniform(0.94, 1.06))
        else:
            if self.alerted:
                self.lost_t += dt
                if self.lost_t > 4.5:
                    self.alerted = False
            if game.noise_t > 0 and distance_xz(self.position, game.noise_pos) < 30:
                self.last_known = Vec3(game.noise_pos.x, 0, game.noise_pos.z)
                if not self.alerted:
                    self.alerted = True
                    sounds.play('alert', 0.28, pitch=self.alert_pitch)
            if self.last_known is not None and distance_xz(self, self.last_known) < 1.25:
                self.search_wait += dt
                if self.search_wait > 0.8:
                    self.last_known = Vec3(player.x, 0, player.z)
                    self.search_wait = 0
        self.path_cd -= dt
        if self.path_cd <= 0:
            self.path_cd = 0.28
            goal = Vec3(player.x, 0, player.z) if self.see or self.last_known is None else self.last_known
            self.goal = goal
            if self not in game.path_queue:
                game.path_queue.append(self)

    def _wish(self, direct):
        if direct:
            return Vec3(player.x - self.x, 0, player.z - self.z)
        if self.path:
            while self.wp < len(self.path) - 1 and distance_xz(self.position, self.path[self.wp]) < 1.05:
                self.wp += 1
            target = self.path[min(self.wp, len(self.path) - 1)]
            return Vec3(target.x - self.x, 0, target.z - self.z)
        if self.goal is not None:
            return Vec3(self.goal.x - self.x, 0, self.goal.z - self.z)
        return Vec3(player.x - self.x, 0, player.z - self.z)

    def _move(self, wish, speed, dt, separate=True):
        wish = Vec3(wish.x, 0, wish.z)
        if separate:
            for other in game.enemies:
                if other is self or other.dead:
                    continue
                delta = Vec3(self.x - other.x, 0, self.z - other.z)
                dist = delta.length()
                if 0.001 < dist < 1.4:
                    wish += (delta / dist) * (1.4 - dist) * 2.1
        wish += self.knock
        self.knock = damp(self.knock, Vec3(0, 0, 0), 6.5, dt)
        if wish.length() < 0.08:
            self.anim_speed = 0
            self._face(Vec3(player.x - self.x, 0, player.z - self.z))
            return False
        direction = norm(wish)
        velocity = direction * speed
        blocked = False
        for axis in ('x', 'z'):
            step = Vec3(velocity.x * dt if axis == 'x' else 0, 0, velocity.z * dt if axis == 'z' else 0)
            if step.length() < 1e-6:
                continue
            if self._blocked(step):
                blocked = True
                if axis == 'x':
                    velocity.x = 0
                else:
                    velocity.z = 0
            else:
                self.position += step
        self.y = 0
        self.anim_speed = math.hypot(velocity.x, velocity.z)
        face = velocity if self.anim_speed > 0.3 else Vec3(player.x - self.x, 0, player.z - self.z)
        self._face(face)
        return blocked

    def _blocked(self, step):
        dist = step.length()
        if dist < 1e-6:
            return False
        origin = self.world_position + Vec3(0, 0.7 * self.bulk, 0)
        return level_blocked(origin, step / dist, dist + 0.36 * self.bulk)

    def _face(self, vector):
        if abs(vector.x) + abs(vector.z) < 0.02:
            return
        self.look_at(self.position + Vec3(vector.x, 0, vector.z))
        self.rotation_x = 0
        self.rotation_z = 0

    def _act(self, dt):
        if self.kind == 'shooter':
            self._act_shooter(dt)
        elif self.kind == 'brute':
            self._act_brute(dt)
        else:
            self._act_chaser(dt)

    def _act_chaser(self, dt):
        dist = distance_xz(self, player)
        if self.swinging or (self.see and dist < 1.7 and self.attack_cd <= 0):
            self._melee(dt, 15, 1.95, 0.38, 0.92)
            return
        wish = self._wish(self.see and dist < 12)
        if self.see and dist < 1.35:
            wish = Vec3(0, 0, 0)
        self._move(wish, self.speed, dt)

    def _melee(self, dt, damage, reach, windup, cooldown):
        if not self.swinging:
            self.swinging = True
            self.swing_t = windup
            self.swing_windup = windup
            self.swing_hit = False
            sounds.play('melee', 0.5, pitch=random.uniform(0.86, 1.08))
            return
        self.swing_t -= dt
        toward = norm(Vec3(player.x - self.x, 0, player.z - self.z))
        self._move(toward, self.speed * 0.9, dt, separate=False)
        if (
            not self.swing_hit and self.swing_t <= self.swing_windup * 0.42
            and distance_xz(self, player) <= reach and abs(player.y - self.y) < 1.5
            and line_of_sight(self.position + Vec3(0, 1.1, 0), player.position + Vec3(0, 1.0, 0))
        ):
            player.hurt(damage, self)
            self.swing_hit = True
        if self.swing_t <= 0:
            self.swinging = False
            self.attack_cd = cooldown

    def _act_shooter(self, dt):
        dist = distance_xz(self, player)
        self.strafe_t -= dt
        if self.strafe_t <= 0:
            self.strafe_dir *= -1
            self.strafe_t = random.uniform(0.75, 1.45)
        if self.see and dist < 20:
            toward = norm(Vec3(player.x - self.x, 0, player.z - self.z))
            side = Vec3(-toward.z, 0, toward.x) * self.strafe_dir
            wish = side
            if dist > 13.5:
                wish += toward * 0.85
            elif dist < 7.2:
                wish -= toward
            self._move(wish, self.speed, dt)
        else:
            self._move(self._wish(False), self.speed, dt)
        if self.see and self.attack_cd <= 0 and dist < 24:
            origin = self.position + Vec3(0, 1.32 * self.bulk, 0)
            aim = player.position + Vec3(0, player.current_height * 0.62, 0)
            direction = aim - origin
            spread = 0.05
            direction += Vec3(
                random.uniform(-spread, spread),
                random.uniform(-spread, spread) * 0.35,
                random.uniform(-spread, spread),
            )
            direction = norm(direction)
            Projectile(origin + direction * 0.7, direction, 11)
            sounds.play('enemy_shot', 0.46, pitch=random.uniform(0.94, 1.08))
            self.attack_cd = random.uniform(1.15, 1.5)

    def _act_brute(self, dt):
        dist = distance_xz(self, player)
        if self.mode == 'approach':
            if self.see and dist < 8.2 and self.attack_cd <= 0:
                self.mode = 'windup'
                self.timer = 0.74
                self.charge_dir = norm(Vec3(player.x - self.x, 0, player.z - self.z))
                sounds.play('rumble', 0.72, pitch=random.uniform(0.85, 1.0))
                return
            self._move(self._wish(self.see and dist < 12), self.speed, dt)
        elif self.mode == 'windup':
            self.timer -= dt
            desired = norm(Vec3(player.x - self.x, 0, player.z - self.z))
            self.charge_dir = norm(self.charge_dir * 0.82 + desired * 0.18)
            self._face(self.charge_dir)
            self.body.color = lerp(self.body_color, self.eye_color, 0.7)
            if self.timer <= 0:
                self.mode = 'charge'
                self.timer = 0.48
                self.charge_hit = False
                self.charge_dir = norm(Vec3(player.x - self.x, 0, player.z - self.z))
        elif self.mode == 'charge':
            self.timer -= dt
            blocked = self._move(self.charge_dir, 12.4, dt, separate=False)
            if not self.charge_hit and dist < 1.5 and abs(player.y - self.y) < 1.6:
                player.hurt(32, self)
                self.charge_hit = True
            if self.timer <= 0 or blocked:
                self.mode = 'recover'
                self.timer = 0.85
                self.attack_cd = 3.3
                self.body.color = self.body_color
        else:
            self.timer -= dt
            self._move(self._wish(False), self.speed * 0.4, dt)
            if self.timer <= 0:
                self.mode = 'approach'

    def _animate(self, dt):
        if self.flash_t > 0:
            self.flash_t -= dt
            if self.flash_t <= 0 and not (self.kind == 'brute' and self.mode == 'windup'):
                self.body.color = self.body_color
        if self.anim_speed > 0.35 and not self.swinging:
            swing = math.sin(self.phase) * (22 if self.kind != 'brute' else 16)
            self.phase += dt * (7.5 if self.kind != 'brute' else 5.5) * (self.anim_speed / 4.5)
        else:
            swing = math.sin(time.time() * 2 + self.phase) * 2
        self.leg_l.rotation_x = swing
        self.leg_r.rotation_x = -swing
        self.arm_l.rotation_x = -swing * 0.55
        self.arm_r.rotation_x = swing * 0.55
        if self.swinging:
            k = 1 - self.swing_t / max(0.01, self.swing_windup)
            punch = -75 if k > 0.5 else -25 * k
            self.arm_l.rotation_x = punch
            self.arm_r.rotation_x = punch
        if self.kind == 'brute' and self.mode in ('windup', 'charge'):
            self.arm_l.rotation_x = -55
            self.arm_r.rotation_x = -55
        pulse = 1 + 0.1 * math.sin(time.time() * (8 if self.see else 3))
        self.eye_l.scale = 0.055 * self.bulk * pulse
        self.eye_r.scale = 0.055 * self.bulk * pulse
        warning = self.swinging or (self.kind == 'brute' and self.mode in ('windup', 'charge'))
        self.tell.enabled = warning
        if self.swinging:
            k = 1 - self.swing_t / max(0.01, self.swing_windup)
            self.tell.scale = lerp(0.55, 2.15, clamp(k, 0, 1))
            self.tell.color = self.eye_color
        elif self.kind == 'brute' and self.mode == 'windup':
            k = 1 - self.timer / 0.74
            self.tell.scale = lerp(0.7, 2.8, clamp(k, 0, 1))
            self.tell.color = color.rgb32(255, 90, 40)
        elif self.kind == 'brute' and self.mode == 'charge':
            self.tell.scale = 1.35
            self.tell.color = color.rgb32(255, 210, 120)

    def _bar(self):
        if not self.bar_bg or not self.bar_bg.enabled:
            return
        self.bar_bg.world_position = self.world_position + Vec3(0, 1.2 * self.bulk + 1.05, 0)
        self.bar.scale_x = clamp(self.hp / self.max_hp, 0, 1)

    def _unstick(self, dt):
        moved = distance_xz(self.position, self.last_pos)
        if moved < 0.025 and self.mode == 'approach' and not self.swinging:
            self.stuck += dt
        else:
            self.stuck = 0
        self.last_pos = Vec3(self.position)
        if self.stuck > 1.05:
            self.stuck = 0
            self.path = []
            self.path_cd = 0
            side = norm(Vec3(random.uniform(-1, 1), 0, random.uniform(-1, 1)))
            if side.length() > 0 and not self._blocked(side * 0.45):
                self.position += side * 0.4
                self.y = 0


class Pickup(Entity):
    def __init__(self, kind, pos):
        super().__init__(position=Vec3(pos.x, 0.55, pos.z))
        self.kind = kind
        self.ready = True
        self.timer = 0
        self.spin = random.random() * 6
        if kind == 'H':
            self.tint = color.rgb32(80, 220, 140)
            Entity(parent=self, model='cube', scale=(0.46, 0.12, 0.12), color=self.tint, shader=unlit_shader)
            Entity(parent=self, model='cube', scale=(0.12, 0.46, 0.12), color=self.tint, shader=unlit_shader)
        else:
            self.tint = color.rgb32(240, 196, 70)
            Entity(parent=self, model='cube', scale=(0.28, 0.18, 0.4), color=self.tint, shader=unlit_shader)
            Entity(parent=self, model='cube', position=(0, 0.16, 0), scale=(0.12, 0.12, 0.12), color=color.white, shader=unlit_shader)
        self.base = Entity(
            parent=self, model='cube', position=(0, -0.42, 0), scale=(0.55, 0.04, 0.55),
            color=color.rgba32(10, 12, 16, 180), shader=unlit_shader,
        )

    def reset(self):
        self.ready = True
        self.timer = 0
        self.visible = True

    def update(self):
        dt = frame_dt()
        self.spin += dt * 2.2
        self.y = 0.62 + math.sin(self.spin) * 0.06
        self.rotation_y += 50 * dt
        if not self.ready:
            self.visible = False
            if game.state == 'playing':
                self.timer -= dt
                if self.timer <= 0:
                    self.ready = True
                    self.visible = True
                    burst(self.world_position, self.tint, 3)
            return
        self.visible = True
        if game.state != 'playing':
            return
        if distance_xz(self, player) > 1.35:
            return
        if self.kind == 'H':
            if player.hp >= player.max_hp:
                return
            player.hp = min(player.max_hp, player.hp + HEALTH_PICKUP)
            DamageNumber(self.world_position + Vec3(0, 0.45, 0), f'+{HEALTH_PICKUP}', self.tint, 22)
        else:
            if player.rifle.reserve >= RESERVE_MAX:
                return
            player.rifle.reserve = min(RESERVE_MAX, player.rifle.reserve + AMMO_PICKUP)
            DamageNumber(self.world_position + Vec3(0, 0.45, 0), f'+{AMMO_PICKUP}', self.tint, 22)
        sounds.play('pickup', 0.55, pitch=1.05 if self.kind == 'H' else 0.9)
        self.ready = False
        self.timer = 18 if self.kind == 'H' else 14
        self.visible = False


# ---------------------------------------------------------------------------
# HUD, menus, session
# ---------------------------------------------------------------------------

class Pinger:
    def __init__(self, parent):
        self.ent = Entity(
            parent=parent, model='quad', scale=(0.012, 0.03),
            color=color.rgba(1, 0.2, 0.16, 0), shader=unlit_shader, enabled=False, z=-2,
        )
        self.life = 0
        self.source = None
        self.fallback = Vec3(0, 0, 0)

    def ping(self, source):
        self.life = 0.85
        self.source = source
        self.fallback = Vec3(getattr(source, 'world_position', source.position))
        self.ent.enabled = True

    def refresh(self, dt):
        if self.life <= 0:
            self.ent.enabled = False
            return
        self.life -= dt
        source = self.source
        if source is not None and not getattr(source, '_gone', False):
            pos = source.world_position
        else:
            pos = self.fallback
        angle = angle_from_player(pos)
        radius = 0.155
        self.ent.position = (math.sin(angle) * radius, math.cos(angle) * radius)
        self.ent.rotation_z = -math.degrees(angle)
        self.ent.color = color.rgba(1, 0.25, 0.18, clamp01(self.life))


def _button(parent, text, y, callback):
    button = Button(
        parent=parent, text=text, position=(0, y), scale=(0.3, 0.05), z=-2,
        color=color.rgba32(42, 56, 76, 245),
        text_color=color.rgb32(236, 238, 242),
        highlight_color=color.rgba32(86, 118, 150, 255),
        pressed_color=color.rgba32(16, 20, 28, 255),
        text_size=0.72,
    )
    button.on_click = callback
    return button


class HUD:
    def __init__(self, textures):
        self.root = Entity(parent=camera.ui, enabled=False)
        self.menu = Entity(parent=camera.ui)
        self.pause = Entity(parent=camera.ui, enabled=False)
        self.death = Entity(parent=camera.ui, enabled=False)
        self.vignette = Entity(
            parent=camera.ui, model='quad', texture=textures['vignette'],
            color=color.rgba(0, 0, 0, 0.35), shader=unlit_shader, z=1, scale=(1.8, 1),
        )
        self._play_widgets()
        self._menu_widgets()
        self._pause_widgets()
        self._death_widgets()
        self.hit_t = 0
        self.toasts = []
        self.pingers = [Pinger(self.root) for _ in range(4)]

    def _play_widgets(self):
        self.hp_bg = Entity(parent=self.root, model='quad', origin=(-0.5, 0), scale=(0.4, 0.02), color=color.rgba32(0, 0, 0, 170), shader=unlit_shader, z=-1)
        self.hp_fill = Entity(parent=self.hp_bg, model='quad', origin=(-0.5, 0), position=(-0.5, 0, -0.01), scale=(1, 0.7), color=color.rgb32(196, 52, 46), shader=unlit_shader)
        self.st_bg = Entity(parent=self.root, model='quad', origin=(-0.5, 0), scale=(0.4, 0.008), color=color.rgba32(0, 0, 0, 140), shader=unlit_shader, z=-1)
        self.st_fill = Entity(parent=self.st_bg, model='quad', origin=(-0.5, 0), position=(-0.5, 0, -0.01), scale=(1, 0.8), color=color.rgb32(220, 186, 78), shader=unlit_shader)
        self.hp_label = ui_text(self.root, 'HEALTH', px=13, color=color.rgb32(180, 186, 194), origin=(-0.5, 0))
        self.clock = ui_text(self.root, '00:00.0', px=28, color=color.rgb32(240, 242, 245))
        self.threat = ui_text(self.root, 'CALM', px=14, color=color.rgb32(120, 200, 140))
        self.kills = ui_text(self.root, 'KILLS  0', px=15, color=color.rgb32(210, 214, 220), origin=(-0.5, 0))
        self.ammo = ui_text(self.root, '30  |  090', px=26, color=color.rgb32(240, 242, 245), origin=(0.5, 0))
        self.ammo_label = ui_text(self.root, 'AMMO', px=12, color=color.rgb32(170, 176, 184), origin=(0.5, 0))
        self.rl_bg = Entity(parent=self.root, model='quad', origin=(0.5, 0), scale=(0.18, 0.007), color=color.rgba32(0, 0, 0, 160), shader=unlit_shader, z=-1, enabled=False)
        self.rl_fill = Entity(parent=self.rl_bg, model='quad', origin=(0.5, 0), position=(0.5, 0, -0.01), scale=(1, 0.7), color=color.rgb32(240, 196, 90), shader=unlit_shader)
        self.hint = ui_text(self.root, '', px=15, color=color.rgba(1, 1, 1, 0.85))
        self.bars = []
        for _ in range(4):
            self.bars.append(Entity(parent=self.root, model='quad', color=color.rgba(1, 0.96, 0.92, 0.9), shader=unlit_shader, z=-2))
        self.dot = Entity(parent=self.root, model='quad', scale=0.003, color=color.rgba(1, 1, 1, 0.8), shader=unlit_shader, z=-2)
        self.hit_bars = []
        for angle in (45, 135, 225, 315):
            rad = math.radians(angle)
            ent = Entity(
                parent=self.root, model='quad', rotation_z=angle, scale=(0.012, 0.0022),
                position=(math.cos(rad) * 0.028, math.sin(rad) * 0.028),
                color=color.white, shader=unlit_shader, enabled=False, z=-2,
            )
            self.hit_bars.append(ent)

    def _menu_widgets(self):
        Entity(parent=self.menu, model='quad', scale=(0.46, 0.68), color=color.rgba32(8, 10, 16, 188), shader=unlit_shader, z=2)
        Entity(parent=self.menu, model='quad', scale=(0.18, 0.003), y=0.175, color=ACCENT, shader=unlit_shader, z=-3)
        ui_text(self.menu, 'LAST LIGHT', px=46, y=0.23, color=color.rgb32(244, 246, 248))
        ui_text(self.menu, 'SURVIVE THE HUNT', px=15, y=0.155, color=ACCENT)
        self.best = ui_text(self.menu, 'BEST  --:--.-', px=14, y=0.105, color=color.rgb32(190, 196, 204))
        _button(self.menu, 'PLAY', 0.03, lambda: game.start_run())
        ui_text(self.menu, 'SENSITIVITY', px=13, y=-0.055, color=color.rgb32(186, 192, 200))
        self.slider = Slider(
            parent=self.menu, min=18, max=90, default=48, step=1, dynamic=True,
            text='', y=-0.11, x=0.05, z=-2, scale=0.58,
        )
        self.slider.knob.text_entity.scale = 0.6
        self.slider.on_value_changed = self._on_sens
        _button(self.menu, 'QUIT', -0.2, application.quit)
        ui_text(
            self.menu, px=12, y=-0.29,
            color=color.rgb32(176, 182, 190),
            text='WASD move    SHIFT sprint    CTRL crouch    SPACE jump\nLMB fire    RMB aim    R reload    F flashlight    ESC pause',
        )

    def _pause_widgets(self):
        Entity(parent=self.pause, model='quad', scale=(0.42, 0.48), color=color.rgba32(6, 8, 12, 210), shader=unlit_shader, z=2)
        ui_text(self.pause, 'PAUSED', px=40, y=0.14, color=color.white)
        _button(self.pause, 'RESUME', 0.02, lambda: game.resume())
        _button(self.pause, 'RESTART', -0.08, lambda: game.start_run())
        _button(self.pause, 'MENU', -0.18, lambda: game.show_menu())

    def _death_widgets(self):
        Entity(parent=self.death, model='quad', scale=(0.46, 0.56), color=color.rgba32(12, 6, 8, 214), shader=unlit_shader, z=2)
        ui_text(self.death, 'YOU DIED', px=42, y=0.18, color=color.rgb32(230, 70, 60))
        self.death_time = ui_text(self.death, '', px=28, y=0.08, color=color.white)
        self.death_kills = ui_text(self.death, '', px=16, y=0.02, color=color.rgb32(210, 214, 220))
        self.death_best = ui_text(self.death, '', px=15, y=-0.03, color=ACCENT)
        _button(self.death, 'RESTART', -0.14, lambda: game.start_run())
        _button(self.death, 'MENU', -0.24, lambda: game.show_menu())

    def _on_sens(self):
        if game is None:
            return
        game.sensitivity = self.slider.value
        game.save()

    def mark_hit(self):
        self.hit_t = 0.12

    def ping(self, source):
        slot = min(self.pingers, key=lambda item: item.life)
        slot.ping(source)

    def toast(self, text):
        label = ui_text(self.root, text, px=15, origin=(-0.5, 0), color=color.rgb32(230, 232, 236))
        self.toasts.append([label, 2.1])
        if len(self.toasts) > 4:
            old, _ = self.toasts.pop(0)
            destroy(old)

    def show_menu(self):
        self.menu.enabled = True
        self.pause.enabled = False
        self.death.enabled = False
        self.root.enabled = False
        if game.best > 0:
            self.best.text = f'BEST  {fmt_time(game.best)}'
        else:
            self.best.text = 'BEST  --:--.-'
        self.slider.value = game.sensitivity

    def show_play(self):
        self.menu.enabled = False
        self.pause.enabled = False
        self.death.enabled = False
        self.root.enabled = True

    def show_pause(self):
        self.pause.enabled = True
        self.menu.enabled = False
        self.death.enabled = False
        self.root.enabled = True

    def show_death(self, is_best):
        self.death.enabled = True
        self.pause.enabled = False
        self.menu.enabled = False
        self.root.enabled = True
        self.death_time.text = f'SURVIVED  {fmt_time(game.time)}'
        self.death_kills.text = f'KILLS  {game.kills}'
        extra = '   NEW BEST' if is_best else ''
        self.death_best.text = f'BEST  {fmt_time(game.best)}{extra}'

    def refresh(self):
        dt = frame_dt()
        aspect = window.aspect_ratio
        self.vignette.scale = (aspect, 1)
        state = game.state
        playing = state == 'playing'
        low = 0
        if player and player.hp < 36 and playing:
            pulse = math.sin(time.time() * (7 if player.hp < 18 else 4)) ** 2
            low = 0.18 + pulse * (0.28 if player.hp < 18 else 0.12)
        flash = player.hurt_flash if player else 0
        if state == 'menu':
            self.vignette.color = color.rgba(0, 0, 0, 0.22)
        elif state == 'dead':
            self.vignette.color = color.rgba(0.45, 0.02, 0.02, 0.55)
        else:
            if flash > 0.02 or low > 0.02:
                alpha = clamp(flash * 0.85 + low, 0, 0.88)
                self.vignette.color = color.rgba(0.75, 0.04, 0.02, alpha)
            else:
                self.vignette.color = color.rgba(0, 0, 0, 0.32)
        if not self.root.enabled or player is None:
            return
        bl = window.bottom_left
        br = window.bottom_right
        self.hp_bg.position = (bl.x + 0.08, bl.y + 0.07)
        self.st_bg.position = (bl.x + 0.08, bl.y + 0.1)
        self.hp_label.position = (bl.x + 0.08, bl.y + 0.125)
        self.ammo.position = (br.x - 0.08, br.y + 0.08)
        self.ammo_label.position = (br.x - 0.08, br.y + 0.125)
        self.clock.position = (0, 0.45)
        self.threat.position = (0, 0.4)
        self.kills.position = (window.top_left.x + 0.06, window.top_left.y - 0.06)
        self.hint.position = (0, -0.28)
        frac = player.hp / player.max_hp
        self.hp_fill.scale_x = max(0.001, frac)
        self.hp_fill.color = color.rgb32(210, 54, 46) if frac > 0.3 else color.rgb32(255, 80, 60)
        self.st_fill.scale_x = max(0.001, player.stamina / 100)
        index = threat_index(game.time if playing else 0)
        self.clock.text = fmt_time(game.time)
        self.threat.text = THREAT_NAMES[index]
        self.threat.color = (color.rgb32(120, 196, 140), color.rgb32(230, 196, 80), color.rgb32(232, 120, 48), color.rgb32(220, 64, 52))[index]
        self.kills.text = f'KILLS  {game.kills}'
        rifle = player.rifle
        self.rl_bg.position = (br.x - 0.08, br.y + 0.048)
        self.rl_bg.enabled = playing and rifle.reloading
        if rifle.reloading:
            self.rl_fill.scale_x = max(0.001, 1 - rifle.reload_t / RELOAD_TIME)
        if rifle.reloading:
            self.ammo.text = 'RELOADING'
            self.ammo.color = color.rgb32(240, 196, 90)
        elif rifle.mag_rounds <= 0 and rifle.reserve <= 0:
            self.ammo.text = 'EMPTY'
            self.ammo.color = color.rgb32(230, 70, 60)
        else:
            self.ammo.text = f'{rifle.mag_rounds:02d}  |  {rifle.reserve:03d}'
            self.ammo.color = color.rgb32(240, 242, 245) if rifle.mag_rounds > 5 else color.rgb32(230, 90, 70)
        show_hint = playing and (game.contact > 0 or game.time < 6.5)
        self.hint.enabled = show_hint
        if game.contact > 0:
            self.hint.text = f'CONTACT IN {max(0, math.ceil(game.contact))}\nWASD   SHIFT   CTRL   SPACE\nLMB FIRE   RMB AIM   R RELOAD   F LIGHT'
        else:
            self.hint.text = 'WASD   SHIFT   CTRL   SPACE\nLMB FIRE   RMB AIM   R RELOAD   F LIGHT'
        if show_hint:
            fade = 1 if game.time < 5 else clamp01((6.5 - game.time) / 1.5)
            self.hint.color = color.rgba(1, 1, 1, 0.88 * fade)
        aimed = False
        if playing:
            center = raycast(camera.world_position, camera.forward, distance=80, ignore=[player])
            aimed = victim_from(center.entity) is not None if center.hit else False
        gap = 0.007 + rifle.spread_now() * 1.35
        length = 0.016
        thick = 0.0021
        col = color.rgba(1, 0.35, 0.28, 0.95) if aimed else color.rgba(1, 0.96, 0.92, 0.88)
        specs = (
            ((0, gap + length / 2), (thick, length)),
            ((0, -(gap + length / 2)), (thick, length)),
            ((-(gap + length / 2), 0), (length, thick)),
            ((gap + length / 2, 0), (length, thick)),
        )
        for bar, (pos, scale) in zip(self.bars, specs):
            bar.position = pos
            bar.scale = scale
            bar.color = col
            bar.enabled = playing
        self.dot.enabled = playing
        self.hit_t = max(0, self.hit_t - dt)
        for bar in self.hit_bars:
            bar.enabled = playing and self.hit_t > 0
            bar.color = color.rgba(1, 1, 1, clamp01(self.hit_t * 10))
        for pinger in self.pingers:
            if playing:
                pinger.refresh(dt)
            else:
                pinger.ent.enabled = False
        alive = []
        y = window.top_left.y - 0.1
        x = window.top_left.x + 0.06
        for label, life in self.toasts:
            life -= dt
            if life <= 0:
                destroy(label)
                continue
            label.position = (x, y)
            label.color = color.rgba(0.92, 0.93, 0.95, clamp01(life))
            y -= 0.035
            alive.append([label, life])
        self.toasts = alive


class Game:
    def __init__(self):
        self.state = 'menu'
        self.time = 0
        self.kills = 0
        self.best = 0
        self.sensitivity = 48
        self.contact = 0
        self.spawn_timer = 0
        self.noise_t = 0
        self.noise_pos = Vec3(0, 0, 0)
        self.enemies = []
        self.projectiles = []
        self.decals = []
        self.path_queue = []
        self.trash = []
        self.pickups = []
        self.flicker = None
        self.flicker_base = (1.15, 0.9, 0.68)
        self.hud = None
        self.menu_pivot = None
        self.menu_arm = None
        self.menu_focus = Vec3(0, 1.4, 0)
        self._load()

    def _load(self):
        try:
            data = json.loads(SAVE_PATH.read_text())
            self.best = float(data.get('best', 0))
            self.sensitivity = float(data.get('sensitivity', 48))
        except Exception:
            self.best = 0
            self.sensitivity = 48

    def save(self):
        try:
            SAVE_PATH.write_text(json.dumps({'best': round(self.best, 2), 'sensitivity': self.sensitivity}))
        except Exception:
            pass

    def boot_camera(self, focus):
        self.menu_focus = focus
        self.menu_pivot = Entity(position=focus)
        self.menu_arm = Entity(parent=self.menu_pivot, position=(2.6, 14.8, 5.4))

    def tick(self):
        self.flush()
        dt = frame_dt()
        self._pump_paths()
        if self.state == 'menu':
            self.menu_pivot.rotation_y += dt * 7
            camera.world_position = self.menu_arm.world_position
            camera.look_at(self.menu_focus)
            camera.fov = damp(camera.fov, 74, 3, dt)
        elif self.state == 'playing':
            self.time += dt
            self.noise_t = max(0, self.noise_t - dt)
            if self.contact > 0:
                self.contact -= dt
                if self.contact <= 0:
                    self.spawn_timer = 0.15
            else:
                self._spawn(dt)
            self._flicker_light()
            self._decay_decals(dt)
        if self.hud:
            self.hud.refresh()
        self._ambient()

    def _ambient(self):
        if not sounds or not sounds.ambient:
            return
        if self.state == 'playing':
            index = threat_index(self.time)
            sounds.ambient.volume = 0.07 + index * 0.015
            sounds.ambient.pitch = 0.82 + index * 0.06
        else:
            sounds.ambient.volume = 0.05
            sounds.ambient.pitch = 0.78

    def _flicker_light(self):
        if self.flicker is None:
            return
        wave_amp = 0.84 + 0.16 * math.sin(time.time() * 8.0)
        if random.random() < 0.01:
            wave_amp *= 0.5
        r, g, b = self.flicker_base
        self.flicker.color = color.rgb(r * wave_amp, g * wave_amp, b * wave_amp)

    def _decay_decals(self, dt):
        keep = []
        for decal in self.decals:
            decal.life -= dt
            if decal.life <= 0:
                retire(decal)
            else:
                keep.append(decal)
        self.decals = keep

    def _pump_paths(self):
        if not self.path_queue or nav is None:
            return
        enemy = self.path_queue.pop(0)
        if enemy is None or enemy.dead or getattr(enemy, '_gone', False) or enemy.goal is None:
            return
        enemy.path = nav.route(enemy.position, enemy.goal)
        enemy.wp = 0

    def _spawn(self, dt):
        self.spawn_timer -= dt
        if self.spawn_timer > 0:
            return
        index = threat_index(self.time)
        if len(self.enemies) >= SPAWN_CAP[index]:
            self.spawn_timer = 0.5
            return
        pos = self._pick_spawn()
        if pos is None:
            self.spawn_timer = 0.8
            return
        kind = weighted_kind(index)
        hp_mul = 1 + min(0.42, self.time * 0.0045)
        speed_mul = 1 + min(0.26, self.time * 0.0032)
        Enemy(kind, pos, hp_mul=hp_mul, speed_mul=speed_mul)
        self.spawn_timer = SPAWN_INTERVAL[index] * random.uniform(0.85, 1.15)

    def _pick_spawn(self):
        ranked = []
        for cell in nav.spawns:
            pos = nav.center(*cell)
            dist = distance_xz(pos, player.position)
            visible = line_of_sight(pos + Vec3(0, 1.4, 0), player.position + Vec3(0, 1.3, 0))
            ranked.append((visible, dist, pos))
        hidden = [item for item in ranked if not item[0] and item[1] > 12]
        pool = hidden or [item for item in ranked if item[1] > 10] or ranked
        if not pool:
            return None
        pool.sort(key=lambda item: item[1], reverse=True)
        return random.choice(pool[:max(1, len(pool) // 2)])[2]

    def flush(self):
        for entity in self.trash:
            safe_destroy(entity)
        self.trash.clear()

    def clear_transient(self):
        for entity in list(scene.entities):
            if getattr(entity, 'is_enemy', False) or getattr(entity, 'is_projectile', False) or getattr(entity, 'is_fx', False):
                retire(entity)
        self.enemies.clear()
        self.projectiles.clear()
        self.decals.clear()
        self.path_queue.clear()
        self.flush()

    def start_run(self):
        self.clear_transient()
        self.state = 'playing'
        self.time = 0
        self.kills = 0
        self.contact = 2.6
        self.spawn_timer = 99
        self.noise_t = 0
        for pickup in self.pickups:
            pickup.reset()
        player.revive(nav.player_start)
        player.apply_light()
        set_mouse_locked(True)
        self.hud.show_play()
        sounds.play('ui', 0.45, pitch=1.1)

    def pause(self):
        if self.state != 'playing':
            return
        self.state = 'paused'
        set_mouse_locked(False)
        player.viewmodel_enabled(False)
        player.apply_light()
        self.hud.show_pause()

    def resume(self):
        if self.state != 'paused':
            return
        self.state = 'playing'
        set_mouse_locked(True)
        player.rifle.cooldown = max(player.rifle.cooldown, 0.12)
        player.viewmodel_enabled(True)
        player.attach_camera()
        player.apply_light()
        self.hud.show_play()

    def show_menu(self):
        self.clear_transient()
        self.state = 'menu'
        set_mouse_locked(False)
        player.detach_camera()
        player.viewmodel_enabled(False)
        player.apply_light()
        self.hud.show_menu()
        self.save()

    def on_death(self):
        if self.state == 'dead':
            return
        self.state = 'dead'
        sounds.play('death', 0.75)
        set_mouse_locked(False)
        player.viewmodel_enabled(False)
        player.apply_light()
        is_best = self.time > self.best
        if is_best:
            self.best = self.time
        self.save()
        self.hud.show_death(is_best)


def update():
    if game is not None:
        game.tick()


def input(key):
    if game is None:
        return
    if key == 'escape':
        if game.state == 'playing':
            game.pause()
        elif game.state == 'paused':
            game.resume()
        elif game.state == 'dead':
            game.show_menu()
    elif key == 'enter':
        if game.state in ('menu', 'dead'):
            game.start_run()


def _place_lights(spots):
    # A handful of real lights. The rest of the fixtures are unlit strips.
    chosen = spots[::2][:7]
    flicker = None
    for i, (x, z, _fixture) in enumerate(chosen):
        light = PointLight(position=(x, WALL_H - 0.55, z), color=color.rgb(1.5, 1.2, 0.9))
        light._light.setAttenuation((1, 0.09, 0.04))
        if i == 0:
            flicker = light
    return flicker


def _yard_focus():
    cells = nav.cells_of('Y')
    acc = Vec3(0, 0, 0)
    for col, row in cells:
        acc += nav.center(col, row)
    acc /= max(1, len(cells))
    acc.y = 1.35
    return acc


def run_smoke(app):
    route = nav.route(nav.player_start, nav.center(*nav.spawns[0]))
    assert len(route) > 2, route
    for _ in range(4):
        app.step()
    _shot(app, '/tmp/lastlight_menu.png')
    game.start_run()
    player.look_enabled = False
    held_keys['w'] = 1
    start = Vec3(player.position)
    for _ in range(90):
        app.step()
    held_keys['w'] = 0
    moved = distance_xz(player.position, start)
    assert -0.05 <= player.y <= 0.4, player.y
    assert moved > 0.6, moved
    ahead = player.position + norm(Vec3(player.forward.x, 0, player.forward.z)) * 5
    cell = nav.nearest(nav.world_to_cell(ahead.x, ahead.z))
    pos = nav.center(*cell)
    enemy = Enemy('chaser', pos, hp_mul=1, speed_mul=0)
    enemy.spawning = 0
    player.look_at(Vec3(pos.x, player.y, pos.z))
    player.yaw = player.rotation_y
    player.pitch = 0
    player.pivot.rotation_x = 0
    app.step()
    before = enemy.hp
    for _ in range(8):
        player.rifle.cooldown = 0
        player.rifle.try_fire()
        app.step()
    assert enemy.hp < before or enemy.dead, (enemy.hp, before, player.rifle.last_hit, pos, player.position)
    _shot(app, '/tmp/lastlight_game.png')
    # Chase: an enemy across the map should receive a path.
    far = nav.center(*nav.spawns[0])
    hunter = Enemy('shooter', far, hp_mul=1, speed_mul=1)
    hunter.spawning = 0
    hunter.see = False
    hunter.last_known = Vec3(player.x, 0, player.z)
    hunter.goal = Vec3(player.x, 0, player.z)
    hunter.path = nav.route(hunter.position, hunter.goal)
    assert len(hunter.path) > 1, (hunter.path, hunter.position, hunter.goal)
    game.time = 9
    player.hurt(1000, None)
    assert game.state == 'dead'
    assert game.best >= 9
    app.step()
    _shot(app, '/tmp/lastlight_death.png')
    game.start_run()
    assert game.state == 'playing' and player.hp == 100 and player.rifle.mag_rounds == MAG_SIZE
    # Front indicator sits above the crosshair when the threat is ahead.
    front = player.position + Vec3(player.forward.x, 0, player.forward.z) * 4
    ang = angle_from_player(front)
    assert abs(ang) < 0.35, ang
    print('SMOKE OK', 'moved', round(moved, 2), 'hit', player.rifle.last_hit, 'path', len(hunter.path))


def _shot(app, path):
    try:
        saved = app.screenshot(namePrefix=path, defaultFilename=0)
        print('saved', saved)
    except Exception as exc:
        print('screenshot failed', path, exc)


def main():
    global game, player, nav, sounds, probe, SAVE_PATH
    smoke = '--smoke' in sys.argv
    if smoke:
        SAVE_PATH = Path(tempfile.gettempdir()) / 'last_light_smoke.json'
        if SAVE_PATH.exists():
            SAVE_PATH.unlink()
    sounds = build_sounds()
    Audio.volume_multiplier = 0.7
    app = Ursina(
        title='LAST LIGHT',
        size=(1280, 720),
        fullscreen=False,
        borderless=False,
        vsync=not smoke,
        development_mode=False,
        editor_ui_enabled=False,
        show_ursina_splash=False,
        window_type='offscreen' if smoke else 'onscreen',
    )
    window.color = color.rgb(0.02, 0.024, 0.04)
    # Offscreen smoke never fires the aspect-ratio event, so the UI lens stays at 1x1
    # and every label is blown up. Fit it the same way a real window does.
    aspect = window.aspect_ratio
    if not aspect or aspect < 0.3 or aspect > 5:
        aspect = 16 / 9
    camera.ui_lens.set_film_size(20 * aspect, 20)
    window.exit_button.visible = False
    window.exit_button.enabled = False
    window.fps_counter.enabled = False
    try:
        render.setAntialias(AntialiasAttrib.MAuto)
    except Exception:
        pass
    try:
        render.setShaderAuto()
    except Exception as exc:
        print('shader auto unavailable:', exc)

    rows = [line for line in MAP_TEXT.strip().split('\n') if line]
    nav = NavGrid(rows)
    textures = build_textures()
    Sky(texture=textures['sky'], color=color.rgb(0.11, 0.13, 0.18))
    scene.fog_color = FOG_COLOR
    scene.fog_density = (26, 88)
    AmbientLight(color=color.rgb(0.34, 0.37, 0.44))
    moon = DirectionalLight(shadows=False, color=color.rgb(0.95, 1.0, 1.15))
    moon.look_at(Vec3(-0.35, -1, -0.25))
    spots = build_level(textures)
    probe = Entity(eternal=True, name='probe')
    game = Game()
    game.flicker = _place_lights(spots)
    player = Player()
    for kind, col, row in nav.pickups:
        game.pickups.append(Pickup(kind, nav.center(col, row)))
    game.hud = HUD(textures)
    game.boot_camera(_yard_focus())
    game.hud.slider.value = game.sensitivity
    game.show_menu()
    if smoke:
        try:
            run_smoke(app)
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
    app.run()


if __name__ == '__main__':
    main()
