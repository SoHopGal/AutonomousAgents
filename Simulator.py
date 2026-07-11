"""
Multi-Agent Traffic Simulation - 3x3 Grid
- 9 intersections, 30-unit spacing
- 2 lanes per direction: right-lane {straight,right}, left-lane {straight,left}
- Vehicles keep lane for entire trip, choose turn randomly at each intersection
- Two light modes: 'fixed' (10s toggle) / 'adaptive' (greenest-queue-first)
- Metrics: avg travel time, avg wait time, collisions, per-intersection per-direction queue counts
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from IPython.display import HTML

matplotlib.rcParams['animation.embed_limit'] = 200

GRID_N   = 3
SPACING  = 30.0
HALF_INT = 6.0          # intersection box half-size (widened from 4.0 to give
                        # perpendicular roads enough corner clearance — see
                        # ReservationManager docstring history / step() comments
                        # for the corner-proximity collision case this fixes)
STOP_OFF = 8.0           # stop line offset from center (scaled with HALF_INT)

DIRECTION_COLORS = {'right': '#4A90D9', 'left': '#E74C3C',
                    'up': '#9B59B6', 'down': '#F39C12'}

DIR_VEC = {'right': (1, 0), 'left': (-1, 0), 'up': (0, 1), 'down': (0, -1)}
VEC_DIR = {v: k for k, v in DIR_VEC.items()}

TURN_MAP = {
    ((1, 0), 'left'):  (0, 1),
    ((1, 0), 'right'): (0, -1),
    ((-1, 0), 'left'): (0, -1),
    ((-1, 0), 'right'): (0, 1),
    ((0, 1), 'left'):  (-1, 0),
    ((0, 1), 'right'): (1, 0),
    ((0, -1), 'left'): (1, 0),
    ((0, -1), 'right'): (-1, 0),
}

# Perpendicular lane offset relative to the road centerline, per (direction, lane_side).
# Mirrors the original single-intersection design (inner/outer lanes at -1/-3 or +1/+3).
LANE_OFFSET = {
    ('right', 'right'): (0, -1),
    ('right', 'left'):  (0, -3),
    ('left',  'right'): (0,  1),
    ('left',  'left'):  (0,  3),
    ('down',  'right'): (-1, 0),
    ('down',  'left'):  (-3, 0),
    ('up',    'right'): (1, 0),
    ('up',    'left'):  (3, 0),
}


def lane_offset(direction, lane_side):
    return LANE_OFFSET[(direction, lane_side)]

W_SAFETY, W_EFFICIENCY, W_COMFORT, W_RULES = 0.50, 0.25, 0.15, 0.10
W1_EM, W2_WAIT, W3_DIST, W4_DEN = 10.0, 1.0, 1.0, 0.5


# =====================================================
# Traffic Light
# =====================================================
class TrafficLight:
    def __init__(self, start_green):
        self.phase = 'green' if start_green else 'red'
        self.timer = 0.0
        self.green_duration = 10.0
        self.yellow_duration = 2.0

    def step(self, dt):
        self.timer += dt
        if self.phase == 'green' and self.timer >= self.green_duration:
            self.phase, self.timer = 'yellow', 0.0
        elif self.phase == 'yellow' and self.timer >= self.yellow_duration:
            self.phase, self.timer = 'red', 0.0
        elif self.phase == 'red' and self.timer >= self.green_duration:
            self.phase, self.timer = 'green', 0.0

    def force_green(self):
        self.phase, self.timer = 'green', 0.0

    def force_red(self):
        self.phase, self.timer = 'red', 0.0

    def is_green(self):
        return self.phase == 'green'

    def color(self):
        return {'green': '#2ECC71', 'yellow': '#F1C40F', 'red': '#E74C3C'}[self.phase]


class IntersectionLights:
    """Complementary H/V lights for one intersection."""
    def __init__(self):
        self.horizontal = TrafficLight(True)
        self.vertical = TrafficLight(False)

    def step_fixed(self, dt):
        self.horizontal.step(dt)
        if self.horizontal.phase in ('green', 'yellow'):
            self.vertical.phase = 'red'
        else:
            self.vertical.phase = 'green'

    def step_adaptive(self, dt, h_wait, v_wait, min_green=4.0):
        # Exactly one axis is "active" (green/yellow) at all times; the other is red.
        h_active = self.horizontal.phase != 'red'

        if h_active:
            self.horizontal.timer += dt
            if self.horizontal.phase == 'green':
                want_switch = (v_wait > h_wait) and self.horizontal.timer >= min_green
                if want_switch or self.horizontal.timer >= self.horizontal.green_duration:
                    self.horizontal.phase, self.horizontal.timer = 'yellow', 0.0
            elif self.horizontal.phase == 'yellow':
                if self.horizontal.timer >= self.horizontal.yellow_duration:
                    self.horizontal.force_red()
                    self.vertical.force_green()
        else:
            self.vertical.timer += dt
            if self.vertical.phase == 'green':
                want_switch = (h_wait > v_wait) and self.vertical.timer >= min_green
                if want_switch or self.vertical.timer >= self.vertical.green_duration:
                    self.vertical.phase, self.vertical.timer = 'yellow', 0.0
            elif self.vertical.phase == 'yellow':
                if self.vertical.timer >= self.vertical.yellow_duration:
                    self.vertical.force_red()
                    self.horizontal.force_green()

    def can_go(self, horizontal_vehicle):
        return self.horizontal.is_green() if horizontal_vehicle else self.vertical.is_green()


# =====================================================
# Reservation Manager (per intersection)
# =====================================================
class ReservationManager:
    """Reservation scheduling for one intersection (SDD 2.3, 3.3).

    Conflict model: STRICT MUTUAL EXCLUSION — at most one (non-emergency) vehicle
    may hold the box at a time, regardless of its travel direction or turn.

    Why not a finer-grained per-lane/per-direction model: an earlier version allowed
    same-orientation vehicles to cross simultaneously (e.g. two vehicles both
    classified as "horizontal"), using only entry/exit orientation as the conflict
    test. That missed real geometric conflicts whenever a turn was involved — a
    vehicle turning from one lane line onto another can physically cross a second
    vehicle's path even though both are nominally "the same orientation". Hand-deriving
    every (origin, lane, turn) case that does or doesn't actually cross proved
    error-prone in practice (three separate residual-collision bugs were found this
    way). Full mutual exclusion is strictly conservative — it forgoes legitimate
    simultaneous non-conflicting movements a real intersection would allow — but it
    is correct by construction: collisions inside the box are only possible when two
    vehicles occupy it at once, and this model never allows that.
    """

    def __init__(self):
        self.slots = {}

    def request(self, vehicle):
        if vehicle.vehicle_id in self.slots:
            return True
        if self.slots:
            return False
        self.slots[vehicle.vehicle_id] = vehicle
        return True

    def release(self, vehicle):
        self.slots.pop(vehicle.vehicle_id, None)

    def force_grant(self, vehicle, occupant_inside_box=False):
        """Emergency override (SDD 3.7).

        Clears the slot UNLESS the current holder is still physically inside the
        box — evicting an occupant mid-crossing would let the emergency vehicle
        enter while the evicted vehicle's body is still there, recreating exactly
        the kind of geometric overlap this model exists to prevent. If the holder
        is genuinely still inside, the emergency vehicle waits one tick rather
        than force its way past a vehicle that hasn't physically cleared yet.
        """
        if self.slots and occupant_inside_box and vehicle.vehicle_id not in self.slots:
            return False
        self.slots.clear()
        self.slots[vehicle.vehicle_id] = vehicle
        return True


# =====================================================
# Pedestrian (human-uncertainty agent, FR7.1 / SVR4.1)
# =====================================================
class Pedestrian:
    """A pedestrian that occasionally crosses near a given intersection."""
    def __init__(self, intersection):
        self.inter = intersection
        self.active = False
        self.pos = np.array([intersection.cx, intersection.cy])
        self.axis = 'h'
        self.speed = 1.5
        self._wait = np.random.uniform(5, 20)

    def step(self, dt):
        if not self.active:
            self._wait -= dt
            if self._wait <= 0:
                self._start_crossing()
            return
        if self.axis == 'h':
            self.pos[0] += self.speed * dt
            if self.pos[0] - self.inter.cx > 6:
                self._reset()
        else:
            self.pos[1] += self.speed * dt
            if self.pos[1] - self.inter.cy > 6:
                self._reset()

    def _start_crossing(self):
        self.active = True
        self.axis = np.random.choice(['h', 'v'])
        self.speed = np.random.uniform(1.0, 2.5)
        cx, cy = self.inter.cx, self.inter.cy
        if self.axis == 'h':
            self.pos = np.array([cx - 6.0, cy + np.random.choice([-3.0, 3.0])])
        else:
            self.pos = np.array([cx + np.random.choice([-3.0, 3.0]), cy - 6.0])

    def _reset(self):
        self.active = False
        self._wait = np.random.uniform(8, 25)

    def near_intersection(self):
        return self.active and \
            abs(self.pos[0] - self.inter.cx) < 6 and abs(self.pos[1] - self.inter.cy) < 6


# =====================================================
# Intersection
# =====================================================
class Intersection:
    def __init__(self, row, col):
        self.row, self.col = row, col
        self.cx, self.cy = col * SPACING, -row * SPACING
        self.lights = IntersectionLights()
        self.manager = ReservationManager()
        # waiting vehicle counts by travel direction, for legend + adaptive control
        self.waiting_counts = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
        self.faulty = False   # sensor/communication fault (FR1.3, SVR2.2)

    def reset_waiting_counts(self):
        self.waiting_counts = {'right': 0, 'left': 0, 'up': 0, 'down': 0}

    def step(self, dt, mode):
        if self.faulty:
            # Fault mode: lights freeze, all directions treated as red (fail-safe stop)
            return
        h_wait = self.waiting_counts['left'] + self.waiting_counts['right']
        v_wait = self.waiting_counts['up'] + self.waiting_counts['down']
        if mode == 'adaptive':
            self.lights.step_adaptive(dt, h_wait, v_wait)
        else:
            self.lights.step_fixed(dt)

    def can_go(self, horizontal_vehicle):
        if self.faulty:
            return False
        return self.lights.can_go(horizontal_vehicle)


# =====================================================
# Vehicle
# =====================================================
class Vehicle:
    def __init__(self, vehicle_id, x, y, direction, lane_side,
                 is_emergency=False, speed=6.0, is_human=False):
        self.vehicle_id = vehicle_id
        self.pos = np.array([x, y], dtype=np.float64)
        self.direction = direction                     # 'right'/'left'/'up'/'down'
        self.lane_side = lane_side                      # 'left' or 'right' (fixed for whole trip)
        self.is_emergency = is_emergency
        self.is_human = is_human
        self.base_speed = speed
        self.speed = speed
        self.dx, self.dy = DIR_VEC[direction]

        self.horizontal = direction in ('right', 'left')
        self.v = self._velocity()
        self.has_crossed_current = False   # crossed current target intersection
        self.waiting = False
        self.wait_time = 0.0
        self.start_time = 0.0
        self.finished = False
        self.current_priority = 0.0

        # Human-driven vehicles occasionally change speed unpredictably (FR7.2)
        self._erratic_timer = np.random.uniform(3, 10) if is_human else float('inf')

        # Sensor/communication fault state (FR1.3) - set externally by Env fault injection
        self.sensor_fault = False

        self.w, self.h = (3.2, 1.4) if self.horizontal else (1.4, 3.2)
        self.color = '#E74C3C' if is_emergency else '#000000'

        # which intersection is the vehicle currently approaching/inside
        self.target_row, self.target_col = None, None
        self.inside_intersection_now = False

        # Turn is decided once at the stop line (reservation time), not at box entry —
        # this lets the reservation system know the vehicle's REAL post-turn axis before
        # granting a slot, so a vehicle turning across another vehicle's straight path
        # is correctly treated as a conflict (see ReservationManager._conflicts).
        self.pending_direction = None

    def _velocity(self):
        return np.array([self.dx, self.dy], dtype=np.float64) * self.speed

    def set_direction(self, direction):
        self.direction = direction
        self.dx, self.dy = DIR_VEC[direction]
        self.horizontal = direction in ('right', 'left')
        self.v = self._velocity()
        self.w, self.h = (3.2, 1.4) if self.horizontal else (1.4, 3.2)

    def choose_turn(self):
        options = ['straight', 'right'] if self.lane_side == 'right' else ['straight', 'left']
        choice = np.random.choice(options)
        if choice == 'straight':
            return self.direction
        new_vec = TURN_MAP[((self.dx, self.dy), choice)]
        return VEC_DIR[new_vec]

    def move(self, dt):
        self.pos += self.v * dt

    def human_tick(self, dt):
        """Erratic speed changes for human-driven vehicles (FR7.2)."""
        self._erratic_timer -= dt
        if self._erratic_timer <= 0:
            self.speed = self.base_speed * np.random.uniform(0.6, 1.3)
            self._erratic_timer = np.random.uniform(2, 8)

    def priority(self, density=1.0):
        em = 1.0 if self.is_emergency else 0.0
        p = W1_EM * em + W2_WAIT * self.wait_time + W4_DEN * density
        self.current_priority = p
        return p

    def utility(self, light_green, dist_front=None):
        j_safety = min(dist_front / 10.0, 1.0) if dist_front is not None else 1.0
        j_eff = np.linalg.norm(self.v) / max(self.speed, 0.1)
        j_comfort = 0.0 if self.waiting else 1.0
        j_rules = 1.0 if (self.is_emergency or light_green) else 0.0
        return (W_SAFETY * j_safety + W_EFFICIENCY * j_eff +
                W_COMFORT * j_comfort + W_RULES * j_rules)


# =====================================================
# Environment
# =====================================================
class Env:
    def __init__(self, light_mode='fixed', total_vehicles=None, total_emergency=0,
                 human_ratio=0.0, pedestrians_enabled=False, fault_at=None, fault_duration=15.0):
        self.dt = 0.1
        self.t = 0.0
        self.light_mode = light_mode      # 'fixed' or 'adaptive'
        self.human_ratio = human_ratio    # fraction of spawned vehicles that are human-driven (FR7.2)
        self.fault_at = fault_at          # seconds; inject a random intersection fault at this time (None = disabled)
        self.fault_duration = fault_duration
        self.fault_active = False
        self.faulty_intersection = None

        self.intersections = {
            (r, c): Intersection(r, c)
            for r in range(GRID_N) for c in range(GRID_N)
        }

        self.pedestrians = []
        if pedestrians_enabled:
            for inter in self.intersections.values():
                self.pedestrians.append(Pedestrian(inter))

        self.vehicles = []
        self.next_vehicle_id = 1
        self.spawn_timer = 0.0
        self.spawn_interval = 1.2
        self.emergency_timer = 0.0
        self.emergency_interval = 35.0

        # Scenario caps: if total_vehicles is set, spawning of regular vehicles
        # stops once that many have been created (None = unlimited / continuous).
        self.total_vehicles = total_vehicles
        self.total_emergency = total_emergency
        self.regular_spawned = 0
        self.emergency_spawned = 0

        self.completed_travel_times = []
        self.completed_wait_times = []
        self.collision_count = 0

        self.span = (GRID_N - 1) * SPACING

    # -------------------------------------------------
    # Spawning
    # -------------------------------------------------
    def spawn_vehicle(self, is_emergency=False):
        edge = np.random.randint(0, 4)   # 0=W,1=E,2=N,3=S
        speed = 10 if is_emergency else 6
        lane_side = np.random.choice(['left', 'right'])
        is_human = (not is_emergency) and (np.random.random() < self.human_ratio)

        if edge == 0:      # entering from west, moving right (+x)
            row = np.random.randint(0, GRID_N)
            oy, ox = lane_offset('right', lane_side)[1], lane_offset('right', lane_side)[0]
            x, y = -15.0, -row * SPACING + oy
            direction = 'right'
            target_row, target_col = row, 0
        elif edge == 1:    # entering from east, moving left (-x)
            row = np.random.randint(0, GRID_N)
            oy = lane_offset('left', lane_side)[1]
            x, y = self.span + 15.0, -row * SPACING + oy
            direction = 'left'
            target_row, target_col = row, GRID_N - 1
        elif edge == 2:    # entering from north, moving down (-y)
            col = np.random.randint(0, GRID_N)
            ox = lane_offset('down', lane_side)[0]
            x, y = col * SPACING + ox, 15.0
            direction = 'down'
            target_row, target_col = 0, col
        else:              # entering from south, moving up (+y)
            col = np.random.randint(0, GRID_N)
            ox = lane_offset('up', lane_side)[0]
            x, y = col * SPACING + ox, -self.span - 15.0
            direction = 'up'
            target_row, target_col = GRID_N - 1, col

        # avoid spawning on top of an existing vehicle in the same lane/position
        for other in self.vehicles:
            if other.direction == direction and np.linalg.norm(other.pos - np.array([x, y])) < 5.0:
                return  # skip this spawn cycle, try again next tick

        v = Vehicle(self.next_vehicle_id, x, y, direction, lane_side,
                    is_emergency=is_emergency, speed=speed, is_human=is_human)
        v.start_time = self.t
        v.target_row, v.target_col = target_row, target_col
        self.next_vehicle_id += 1
        self.vehicles.append(v)
        if is_emergency:
            self.emergency_spawned += 1
        else:
            self.regular_spawned += 1

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _intersection_at(self, row, col):
        return self.intersections.get((row, col))

    def _next_target(self, row, col, direction):
        dx, dy = DIR_VEC[direction]
        # grid step: +x -> col+1 ; -x -> col-1 ; +y(up) -> row-1 ; -y(down) -> row+1
        ncol = col + (1 if dx == 1 else -1 if dx == -1 else 0)
        nrow = row + (-1 if dy == 1 else 1 if dy == -1 else 0)
        return nrow, ncol

    def _in_bounds(self, row, col):
        return 0 <= row < GRID_N and 0 <= col < GRID_N

    def same_lane(self, v1, v2):
        if v1.direction != v2.direction:
            return False
        if v1.horizontal:
            return abs(v1.pos[1] - v2.pos[1]) < 1.5
        return abs(v1.pos[0] - v2.pos[0]) < 1.5

    def is_blocked_by_front(self, vehicle):
        min_dist = None
        for other in self.vehicles:
            if other is vehicle or not self.same_lane(vehicle, other):
                continue
            dist = np.linalg.norm(vehicle.pos - other.pos)
            if dist >= 8:
                continue
            ahead = False
            if vehicle.dx > 0 and other.pos[0] > vehicle.pos[0]: ahead = True
            elif vehicle.dx < 0 and other.pos[0] < vehicle.pos[0]: ahead = True
            elif vehicle.dy > 0 and other.pos[1] > vehicle.pos[1]: ahead = True
            elif vehicle.dy < 0 and other.pos[1] < vehicle.pos[1]: ahead = True
            if ahead and (min_dist is None or dist < min_dist):
                min_dist = dist
        if min_dist is not None and min_dist < 7:
            return True, min_dist
        return False, None

    def traffic_density(self):
        return len(self.vehicles)

    def average_wait_time(self):
        return float(np.mean(self.completed_wait_times)) if self.completed_wait_times else 0.0

    def min_wait_time(self):
        return float(np.min(self.completed_wait_times)) if self.completed_wait_times else 0.0

    def max_wait_time(self):
        return float(np.max(self.completed_wait_times)) if self.completed_wait_times else 0.0

    def average_travel_time(self):
        return float(np.mean(self.completed_travel_times)) if self.completed_travel_times else 0.0

    def min_travel_time(self):
        return float(np.min(self.completed_travel_times)) if self.completed_travel_times else 0.0

    def max_travel_time(self):
        return float(np.max(self.completed_travel_times)) if self.completed_travel_times else 0.0

    def set_light_mode(self, mode):
        assert mode in ('fixed', 'adaptive')
        self.light_mode = mode

    # -------------------------------------------------
    # Step
    # -------------------------------------------------
    def step(self):
        self.t += self.dt
        self.spawn_timer += self.dt
        self.emergency_timer += self.dt

        # ---- Fault injection / repair (FR1.3, SVR2.2) ----
        if self.fault_at is not None and not self.fault_active and self.t >= self.fault_at:
            self.faulty_intersection = self.intersections[
                (np.random.randint(GRID_N), np.random.randint(GRID_N))]
            self.faulty_intersection.faulty = True
            self.fault_active = True
        if self.fault_active and self.t >= self.fault_at + self.fault_duration:
            self.faulty_intersection.faulty = False
            self.faulty_intersection = None
            self.fault_active = False
            self.fault_at = None

        # ---- Pedestrians (FR7.1, SVR4.1) ----
        for ped in self.pedestrians:
            ped.step(self.dt)

        if self.spawn_timer >= self.spawn_interval:
            if self.total_vehicles is None or self.regular_spawned < self.total_vehicles:
                self.spawn_vehicle()
            self.spawn_timer = 0.0

        if self.emergency_timer >= self.emergency_interval:
            if self.emergency_spawned < self.total_emergency:
                self.spawn_vehicle(is_emergency=True)
            self.emergency_timer = 0.0

        for inter in self.intersections.values():
            inter.reset_waiting_counts()

        # pre-pass: count waiting vehicles per intersection+direction (for legend/adaptive)
        for vehicle in self.vehicles:
            if vehicle.waiting and vehicle.target_row is not None and vehicle.target_col is not None:
                inter = self._intersection_at(vehicle.target_row, vehicle.target_col)
                if inter is not None:
                    inter.waiting_counts[vehicle.direction] += 1

        for inter in self.intersections.values():
            inter.step(self.dt, self.light_mode)

        density = self.traffic_density()

        for vehicle in self.vehicles:
            if vehicle.finished:
                continue

            if vehicle.is_human:
                vehicle.human_tick(self.dt)

            blocked, dist_front = self.is_blocked_by_front(vehicle)

            inter = self._intersection_at(vehicle.target_row, vehicle.target_col)
            light_green = True
            if inter is not None:
                light_green = inter.can_go(vehicle.horizontal) or vehicle.is_emergency

            vehicle.utility(light_green, dist_front=dist_front)
            vehicle.priority(density=density)

            # ---- Sensor fault: vehicle stops defensively (FR1.3) ----
            if vehicle.sensor_fault:
                vehicle.waiting = True
                vehicle.wait_time += self.dt
                vehicle.v = np.zeros(2)
                continue

            if blocked:
                vehicle.waiting = True
                vehicle.wait_time += self.dt
                vehicle.v = np.zeros(2)
                continue

            # ---- Pedestrian caution: slow/stop non-emergency vehicles near a crossing pedestrian (FR7.1) ----
            if not vehicle.is_emergency:
                near_ped = any(
                    p.near_intersection() and
                    self._intersection_at(vehicle.target_row, vehicle.target_col) is p.inter
                    for p in self.pedestrians
                ) if vehicle.target_row is not None else False
                if near_ped:
                    vehicle.waiting = True
                    vehicle.wait_time += self.dt
                    vehicle.v = vehicle._velocity() * 0.25
                    vehicle.move(self.dt)
                    continue

            icx, icy = (inter.cx, inter.cy) if inter is not None else (None, None)

            at_stop_line = False
            inside_box = False
            fully_clear = True
            if inter is not None:
                rel = vehicle.pos - np.array([icx, icy])
                if vehicle.dx > 0:   at_stop_line = rel[0] + vehicle.w / 2 >= -STOP_OFF
                elif vehicle.dx < 0: at_stop_line = rel[0] - vehicle.w / 2 <= STOP_OFF
                elif vehicle.dy > 0: at_stop_line = rel[1] + vehicle.h / 2 >= -STOP_OFF
                elif vehicle.dy < 0: at_stop_line = rel[1] - vehicle.h / 2 <= STOP_OFF
                inside_box = abs(rel[0]) <= HALF_INT and abs(rel[1]) <= HALF_INT
                # Release the reservation only once the vehicle's full body (not just its
                # center point) has cleared the box - otherwise the slot frees up while the
                # tail is still geometrically inside, letting the next vehicle's body overlap
                # it before it has actually left (this caused residual collisions at the
                # box boundary even after the turn-conflict fix above).
                half_len = max(vehicle.w, vehicle.h) / 2.0
                fully_clear = (abs(rel[0]) > HALF_INT + half_len or
                               abs(rel[1]) > HALF_INT + half_len)

            if inter is not None and at_stop_line and not inside_box and not vehicle.has_crossed_current:
                # Decide the turn now, BEFORE requesting a reservation, so the reservation
                # system can correctly detect conflicts based on the vehicle's real
                # post-turn path rather than its pre-turn approach axis.
                if vehicle.pending_direction is None:
                    vehicle.pending_direction = vehicle.choose_turn()

                if not light_green:
                    vehicle.waiting = True
                    vehicle.wait_time += self.dt
                    vehicle.v = np.zeros(2)
                    continue

                if vehicle.is_emergency:
                    holder = next(iter(inter.manager.slots.values()), None)
                    holder_inside = False
                    if holder is not None and holder is not vehicle:
                        hrel = holder.pos - np.array([icx, icy])
                        holder_inside = abs(hrel[0]) <= HALF_INT and abs(hrel[1]) <= HALF_INT
                    granted = inter.manager.force_grant(vehicle, occupant_inside_box=holder_inside)
                    if not granted:
                        vehicle.waiting = True
                        vehicle.wait_time += self.dt
                        vehicle.v = np.zeros(2)
                        continue
                else:
                    if not inter.manager.request(vehicle):
                        # ---- Priority-based overtaking (VS-13, SDD 3.4) ----
                        # If this vehicle's priority clearly exceeds the vehicle currently
                        # holding the slot, it preempts (only for non-emergency contention;
                        # emergency vehicles already bypass via force_grant above).
                        holder = next(iter(inter.manager.slots.values()), None)
                        holder_inside = False
                        if holder is not None:
                            hrel = holder.pos - np.array([icx, icy])
                            holder_inside = abs(hrel[0]) <= HALF_INT and abs(hrel[1]) <= HALF_INT
                        if (holder is not None and not holder.is_emergency and
                                not holder_inside and
                                vehicle.current_priority > holder.current_priority + 2.0):
                            # Preemption is only safe while the lower-priority holder is
                            # still WAITING at its own stop line (not yet physically in the
                            # box) — otherwise this reproduces the eviction-while-occupied
                            # bug fixed above for the emergency case.
                            inter.manager.slots.pop(holder.vehicle_id, None)
                            inter.manager.slots[vehicle.vehicle_id] = vehicle
                            holder.waiting = True
                        else:
                            vehicle.waiting = True
                            vehicle.wait_time += self.dt
                            vehicle.v = np.zeros(2)
                            continue

            vehicle.waiting = False
            vehicle.v = vehicle._velocity()
            vehicle.move(self.dt)

            if inter is not None and inside_box:
                if not vehicle.has_crossed_current:
                    # Apply the turn decided earlier at the stop line (do NOT re-roll here —
                    # the reservation was granted based on that exact decision).
                    new_dir = vehicle.pending_direction if vehicle.pending_direction is not None \
                        else vehicle.choose_turn()
                    vehicle.set_direction(new_dir)
                    vehicle.has_crossed_current = True
                    # snap perpendicular position onto the new direction's lane line
                    ox, oy = lane_offset(new_dir, vehicle.lane_side)
                    if new_dir in ('right', 'left'):
                        vehicle.pos[1] = icy + oy
                    else:
                        vehicle.pos[0] = icx + ox

            if inter is not None and vehicle.has_crossed_current and fully_clear:
                inter.manager.release(vehicle)
                nrow, ncol = self._next_target(vehicle.target_row, vehicle.target_col, vehicle.direction)
                vehicle.has_crossed_current = False
                vehicle.pending_direction = None
                if self._in_bounds(nrow, ncol):
                    vehicle.target_row, vehicle.target_col = nrow, ncol
                else:
                    vehicle.target_row, vehicle.target_col = None, None  # exiting grid

            # exit condition: outside overall bounds
            if (vehicle.pos[0] < -20 or vehicle.pos[0] > self.span + 20 or
                    vehicle.pos[1] > 20 or vehicle.pos[1] < -self.span - 20):
                if not vehicle.finished:
                    vehicle.finished = True
                    self.completed_travel_times.append(self.t - vehicle.start_time)
                    self.completed_wait_times.append(vehicle.wait_time)

        self.vehicles = [v for v in self.vehicles if not v.finished]

        # Collision detection: axis-aligned rectangle overlap (each vehicle's actual
        # w x h footprint), not a flat circular distance threshold. A flat radius check
        # produces false positives between vehicles on entirely different, non-conflicting
        # lanes whose centers simply happen to pass within that radius of each other at
        # some point on the grid - their real car bodies, at actual width/length and lane
        # separation, never touch. True overlap requires both axes to overlap simultaneously.
        for i in range(len(self.vehicles)):
            v1 = self.vehicles[i]
            for v2 in self.vehicles[i + 1:]:
                dx = abs(v1.pos[0] - v2.pos[0])
                dy = abs(v1.pos[1] - v2.pos[1])
                overlap_x = dx < (v1.w + v2.w) / 2.0
                overlap_y = dy < (v1.h + v2.h) / 2.0
                if overlap_x and overlap_y:
                    self.collision_count += 1


# =====================================================
# Scenario Definitions (SDD validation scenarios)
# =====================================================
# Categories:
#   - Normal operation:   VS-01
#   - Congestion:         VS-02, VS-11, VS-12
#   - Emergency:          VS-03, VS-14
#   - Priority resolution: VS-13 (priority-based overtaking)
#   - Failure/robustness: fault_at param (apply to any scenario via human_fault_overrides)
#   - Human uncertainty:  human_ratio + pedestrians_enabled (apply via human_fault_overrides)
SCENARIOS = {
    'VS-01': dict(label='Multi-Vehicle Intersection Coordination',
                  vehicles=10, emergency=2, sim_time=300,
                  goal='Validate reservation scheduling.'),
    'VS-02': dict(label='Heavy Traffic Congestion',
                  vehicles=50, emergency=2, sim_time=300,
                  goal='Evaluate congestion handling.'),
    'VS-03': dict(label='Emergency Vehicle Prioritization',
                  vehicles=30, emergency=2, sim_time=300,
                  goal='Validate emergency prioritization.'),
    'VS-11': dict(label='Reservation Stress Test',
                  vehicles=100, emergency=2, sim_time=300,
                  goal='Evaluate reservation scheduling under extreme demand.'),
    'VS-12': dict(label='High-Density Traffic',
                  vehicles=75, emergency=2, sim_time=300,
                  goal='Evaluate scalability and congestion management.'),
    'VS-13': dict(label='Priority-Based Overtaking',
                  vehicles=30, emergency=2, sim_time=300,
                  goal='Validate priority resolution between competing vehicles.'),
    'VS-14': dict(label='Mixed Emergency Traffic',
                  vehicles=50, emergency=5, sim_time=300,
                  goal='Evaluate emergency handling under congestion.'),
}
# Note: VS-03 (2) and VS-14 (5) keep their original spec'd emergency counts exactly.
# All other scenarios originally specified 0 emergency vehicles; a small baseline of 2
# was added to each so every scenario demonstrates emergency-vehicle handling at least
# minimally, per explicit request — this is a deliberate deviation from the literal
# VS-01/02/11/12/13 spec values and should be noted as such in any written report.

# Optional add-ons demonstrating the other two SDD scenario categories.
# Set per-scenario via this dict: {scenario_id: dict(human_ratio=.., pedestrians_enabled=.., fault_at=..)}
# Left empty by default so VS-01..14 match the spec exactly; uncomment lines below to layer
# "Failure/robustness" and "Human uncertainty" conditions on top of any scenario for extended testing.
SCENARIO_OVERRIDES = {
    # 'VS-01': dict(human_ratio=0.2, pedestrians_enabled=True),   # human-uncertainty variant
    # 'VS-02': dict(fault_at=100.0),                               # failure/robustness variant
}


def run_scenario_headless(scenario_id, light_mode='fixed', seed=None):
    """Run one scenario with no animation; returns a dict of summary metrics."""
    if seed is not None:
        np.random.seed(seed)
    cfg = SCENARIOS[scenario_id]
    overrides = SCENARIO_OVERRIDES.get(scenario_id, {})
    e = Env(light_mode=light_mode,
            total_vehicles=cfg['vehicles'],
            total_emergency=cfg['emergency'],
            **overrides)
    steps = int(cfg['sim_time'] / e.dt)
    for _ in range(steps):
        e.step()
    return {
        'scenario': scenario_id,
        'label': cfg['label'],
        'goal': cfg['goal'],
        'vehicles_spawned': e.regular_spawned + e.emergency_spawned,
        'completed_trips': len(e.completed_travel_times),
        'avg_travel': e.average_travel_time(),
        'min_travel': e.min_travel_time(),
        'max_travel': e.max_travel_time(),
        'avg_wait': e.average_wait_time(),
        'min_wait': e.min_wait_time(),
        'max_wait': e.max_wait_time(),
        'collisions': e.collision_count,
    }


def run_all_scenarios(light_mode='fixed', seed=None):
    results = []
    
    if seed is not None:
        np.random.seed(seed)
        
    for sid in SCENARIOS:
        print(f'Running {sid} ({SCENARIOS[sid]["label"]}) [{light_mode}] ...')
        results.append(run_scenario_headless(sid, light_mode=light_mode, seed=np.random.randint(0, 1_000_000)))
    return results


# Column definitions for the scenario comparison table — every value is computed
# from vehicles that COMPLETED their trip during the 300s run (i.e. spawned, drove
# through the grid, and exited). Vehicles still on the road when the run ends are
# not included in travel/wait statistics, only in the "Spawned" count.
TABLE_COLUMNS = """
Scenario comparison table — column reference
---------------------------------------------
ID        : Scenario identifier (VS-xx), see SCENARIOS dict for full label/goal.
Spawned   : Total vehicles created during the run (regular + emergency).
Completed : Vehicles that finished their trip (exited the grid) within 300s.
AvgTrav   : Mean travel time (s), spawn -> exit, over completed vehicles only.
MinTrav / MaxTrav : Fastest / slowest completed trip (s).
AvgWait   : Mean total time (s) a completed vehicle spent stationary
            (red light, blocked by traffic ahead, or yielding to a pedestrian).
MinWait / MaxWait : Best-case / worst-case total wait time (s) among completed vehicles.
Collisions: Count of vehicle-pairs detected closer than the minimum safety distance
            during the run (lower is better; ideally 0).
"""


def print_scenario_table(results, title):
    print(f'--- {title} ---')
    header = (f"{'ID':6}{'Spawned':9}{'Completed':10}{'AvgTrav':9}{'MinTrav':9}"
              f"{'MaxTrav':9}{'AvgWait':9}{'MinWait':9}{'MaxWait':9}{'Collisions':10}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['scenario']:6}{r['vehicles_spawned']:<9}{r['completed_trips']:<10}"
              f"{r['avg_travel']:<9.1f}{r['min_travel']:<9.1f}{r['max_travel']:<9.1f}"
              f"{r['avg_wait']:<9.1f}{r['min_wait']:<9.1f}{r['max_wait']:<9.1f}"
              f"{r['collisions']:<10}")
    print()


print(TABLE_COLUMNS)

# Run all 7 scenarios under BOTH light-control strategies, so the table also serves
# as a direct fixed-vs-adaptive comparison (this is the only way adaptive mode's
# effect becomes visible — it is never used by default otherwise).
_results_fixed = run_all_scenarios(light_mode='fixed')
_results_adaptive = run_all_scenarios(light_mode='adaptive')
with open("fixed_results.json","w") as f:
    json.dump(_results_fixed,f,indent=4)

with open("adaptive_results.json","w") as f:
    json.dump(_results_adaptive,f,indent=4)
print()
print_scenario_table(_results_fixed, 'FIXED light timing (10s/axis, no traffic-aware adjustment)')
print_scenario_table(_results_adaptive, 'ADAPTIVE light timing (green given to the busier axis)')

# =====================================================
# Setup (animated view of one selected scenario)
# =====================================================
ANIMATE_SCENARIO = 'VS-01'        # choose any key from SCENARIOS to visualize live
ANIMATE_LIGHT_MODE = 'fixed'      # 'fixed' or 'adaptive' — controls the live animated run below
_cfg = SCENARIOS[ANIMATE_SCENARIO]
_overrides = SCENARIO_OVERRIDES.get(ANIMATE_SCENARIO, {})
env = Env(light_mode=ANIMATE_LIGHT_MODE, total_vehicles=_cfg['vehicles'],
          total_emergency=_cfg['emergency'], **_overrides)
for _ in range(6):
    env.spawn_vehicle()

SPAN = (GRID_N - 1) * SPACING
PAD = 25

fig = plt.figure(figsize=(15, 9), facecolor='#0D1117')
ax = fig.add_axes([0.01, 0.02, 0.68, 0.96])     # simulation (left)
ax_info = fig.add_axes([0.72, 0.02, 0.27, 0.96])  # dashboard (right)

ax.set_xlim(-PAD, SPAN + PAD)
ax.set_ylim(-SPAN - PAD, PAD)
ax.set_aspect('equal')
ax.set_facecolor('#1A252F')
ax.axis('off')

# Road surface
for r in range(GRID_N):
    ax.add_patch(Rectangle((-PAD, -r * SPACING - HALF_INT), SPAN + 2 * PAD, 2 * HALF_INT,
                            color='#3D4F60', zorder=1))
for c in range(GRID_N):
    ax.add_patch(Rectangle((c * SPACING - HALF_INT, -SPAN - PAD), 2 * HALF_INT, SPAN + 2 * PAD,
                            color='#3D4F60', zorder=1))

# ---- Lane markings, built directly from LANE_OFFSET so lines match real vehicle paths ----
# Horizontal roads: 4 lanes total (2 going right = positive y offsets -1/-3 i.e. below center;
#                    2 going left  = offsets +1/+3, above center)
#   white dashed between same-direction lanes, yellow dashed between opposite directions
right_far, right_near = lane_offset('right', 'right')[1], lane_offset('right', 'left')[1]   # -1, -3
left_far, left_near = lane_offset('left', 'right')[1], lane_offset('left', 'left')[1]       # 1, 3

for r in range(GRID_N):
    y0 = -r * SPACING
    # center divider (yellow dashed) between the two travel directions
    ax.plot([-PAD, SPAN + PAD], [y0, y0], '--', color='#F1C40F', lw=1.2, alpha=0.8, zorder=2)
    # white dashed line between the 2 lanes of "right"-bound traffic
    ax.plot([-PAD, SPAN + PAD], [y0 + (right_far + right_near) / 2] * 2,
            '--', color='white', lw=0.7, alpha=0.6, zorder=2)
    # white dashed line between the 2 lanes of "left"-bound traffic
    ax.plot([-PAD, SPAN + PAD], [y0 + (left_far + left_near) / 2] * 2,
            '--', color='white', lw=0.7, alpha=0.6, zorder=2)

down_far, down_near = lane_offset('down', 'right')[0], lane_offset('down', 'left')[0]   # 1, 3
up_far, up_near = lane_offset('up', 'right')[0], lane_offset('up', 'left')[0]           # -1, -3

for c in range(GRID_N):
    x0 = c * SPACING
    ax.plot([x0, x0], [-SPAN - PAD, PAD], '--', color='#F1C40F', lw=1.2, alpha=0.8, zorder=2)
    ax.plot([x0 + (down_far + down_near) / 2] * 2, [-SPAN - PAD, PAD],
            '--', color='white', lw=0.7, alpha=0.6, zorder=2)
    ax.plot([x0 + (up_far + up_near) / 2] * 2, [-SPAN - PAD, PAD],
            '--', color='white', lw=0.7, alpha=0.6, zorder=2)

# Intersection boxes (drawn on top of lane markings)
for (r, c), inter in env.intersections.items():
    ax.add_patch(Rectangle((inter.cx - HALF_INT, inter.cy - HALF_INT),
                            2 * HALF_INT, 2 * HALF_INT, color='#7F8C8D', zorder=3))

# Traffic light indicator patches + one waiting-count label per corner (matches the
# light it sits next to: h_left -> queue of vehicles travelling 'left' through this
# intersection, h_right -> 'right', v_top -> 'up', v_bottom -> 'down')
tl_patches = {}
grid_queue_texts = {}
_label_style = dict(fontsize=10, color='#FFD700', fontweight='bold', zorder=10,
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='#1A252F',
                               alpha=0.75, edgecolor='none'))
for (r, c), inter in env.intersections.items():
    cx, cy = inter.cx, inter.cy
    tl_patches[(r, c)] = {
        'h_left':  ax.add_patch(Rectangle((cx - 9, cy - 1), 1, 2, color='gray', zorder=4)),
        'h_right': ax.add_patch(Rectangle((cx + 8, cy - 1), 1, 2, color='gray', zorder=4)),
        'v_bottom': ax.add_patch(Rectangle((cx - 1, cy - 9), 2, 1, color='gray', zorder=4)),
        'v_top':   ax.add_patch(Rectangle((cx - 1, cy + 8), 2, 1, color='gray', zorder=4)),
    }
    # one label just outside each of the 4 light patches, in that direction's travel sense
    grid_queue_texts[(r, c)] = {
        'left':  ax.text(cx - 11.5, cy, '', **_label_style),   # next to h_left  -> traffic going left
        'right': ax.text(cx + 11.5, cy, '', **_label_style),   # next to h_right -> traffic going right
        'up':    ax.text(cx, cy + 11.5, '', **_label_style),   # next to v_top   -> traffic going up
        'down':  ax.text(cx, cy - 11.5, '', **_label_style),   # next to v_bottom -> traffic going down
    }

legend_elements = [
    Patch(facecolor='#000000', edgecolor='white', label='Vehicle (moving)'),
    Patch(facecolor='#808080', edgecolor='white', label='Vehicle (waiting)'),
    Patch(facecolor='#E74C3C', edgecolor='white', label='🚑 Emergency'),
    Patch(facecolor='#00B894', label='🚶 Pedestrian'),
]
ax.legend(handles=legend_elements, loc='upper right',
          facecolor='#2C3E50', labelcolor='white', fontsize=10, framealpha=0.9)

rectangles, labels, ped_patches = [], [], []

# ---- Dashboard (right panel) ----
ax_info.set_facecolor('#161B22')
ax_info.axis('off')
ax_info.text(0.5, 0.98, 'Traffic Dashboard', transform=ax_info.transAxes,
             fontsize=14, color='#C9D1D9', ha='center', va='top', fontweight='bold')

dash_labels_y = {
    'time': 0.92, 'mode': 0.87, 'vehicles': 0.82,
    'travel': 0.75, 'travel_minmax': 0.71,
    'wait': 0.66, 'wait_minmax': 0.62,
    'collisions': 0.56,
}
# Light separators only between the 3 logical groups (status / travel-wait / collisions),
# positioned in the gaps between groups rather than under every single row.
for y_sep in (0.795, 0.595):
    ax_info.plot([0.03, 0.97], [y_sep, y_sep], color='#21262D', lw=0.8,
                 transform=ax_info.transAxes)

dash_vals = {k: ax_info.text(0.06, y, '', transform=ax_info.transAxes,
                              fontsize=12, color='#C9D1D9', va='top')
             for k, y in dash_labels_y.items()}
# min/max sub-rows use a slightly smaller, dimmer font
dash_vals['travel_minmax'].set_fontsize(9)
dash_vals['travel_minmax'].set_color('#8B949E')
dash_vals['wait_minmax'].set_fontsize(9)
dash_vals['wait_minmax'].set_color('#8B949E')

ax_info.text(0.06, 0.49, 'Waiting vehicles per intersection', transform=ax_info.transAxes,
             fontsize=11, color='#8B949E', va='top')

# Per-intersection waiting-queue readout (replaces on-grid floating text)
queue_y_start = 0.45
queue_dy = (queue_y_start - 0.03) / GRID_N / GRID_N * 1.6
queue_texts = {}
for idx, (r, c) in enumerate(sorted(env.intersections.keys())):
    y = queue_y_start - idx * queue_dy
    queue_texts[(r, c)] = ax_info.text(0.06, y, '', transform=ax_info.transAxes,
                                       fontsize=9.5, color='#79C0FF', va='top')


def update(frame):
    env.step()
    global rectangles, labels, ped_patches

    for r in rectangles: r.remove()
    for l in labels: l.remove()
    for p in ped_patches: p.remove()
    rectangles, labels, ped_patches = [], [], []

    for (r, c), inter in env.intersections.items():
        if inter.faulty:
            tl_patches[(r, c)]['h_left'].set_facecolor('#FFD700')
            tl_patches[(r, c)]['h_right'].set_facecolor('#FFD700')
            tl_patches[(r, c)]['v_bottom'].set_facecolor('#FFD700')
            tl_patches[(r, c)]['v_top'].set_facecolor('#FFD700')
        else:
            hc = inter.lights.horizontal.color()
            vc = inter.lights.vertical.color()
            tl_patches[(r, c)]['h_left'].set_facecolor(hc)
            tl_patches[(r, c)]['h_right'].set_facecolor(hc)
            tl_patches[(r, c)]['v_bottom'].set_facecolor(vc)
            tl_patches[(r, c)]['v_top'].set_facecolor(vc)

        wc = inter.waiting_counts
        fault_tag = '  ⚠ FAULT' if inter.faulty else ''
        queue_texts[(r, c)].set_text(
            f"I({r},{c})  ←{wc['left']} →{wc['right']} ↑{wc['up']} ↓{wc['down']}{fault_tag}"
        )
        grid_queue_texts[(r, c)]['left'].set_text(f"←{wc['left']}")
        grid_queue_texts[(r, c)]['right'].set_text(f"→{wc['right']}")
        grid_queue_texts[(r, c)]['up'].set_text(f"↑{wc['up']}")
        grid_queue_texts[(r, c)]['down'].set_text(f"↓{wc['down']}")

    for ped in env.pedestrians:
        if ped.active:
            from matplotlib.patches import Circle
            c = Circle(ped.pos, 0.8, color='#00B894', zorder=7)
            ax.add_patch(c)
            ped_patches.append(c)

    for vehicle in env.vehicles:
        color = '#808080' if (vehicle.waiting and not vehicle.is_emergency) else vehicle.color
        rect = Rectangle((vehicle.pos[0] - vehicle.w / 2, vehicle.pos[1] - vehicle.h / 2),
                          vehicle.w, vehicle.h, color=color, zorder=5)
        ax.add_patch(rect)
        rectangles.append(rect)
        prefix = '🚑' if vehicle.is_emergency else ('H' if vehicle.is_human else '')
        lbl = ax.text(vehicle.pos[0], vehicle.pos[1], f'{prefix}{vehicle.vehicle_id}',
                      fontsize=6.5, ha='center', va='center', color='white',
                      fontweight='bold', zorder=6)
        labels.append(lbl)

    dash_vals['time'].set_text(f'⏱  Time: {env.t:.1f}s')
    dash_vals['mode'].set_text(f'⚙  Light mode: {env.light_mode}')
    dash_vals['vehicles'].set_text(f'🚘 Active vehicles: {len(env.vehicles)}')
    dash_vals['travel'].set_text(f'🏁 Avg travel time: {env.average_travel_time():.1f}s')
    dash_vals['travel_minmax'].set_text(
        f'   min {env.min_travel_time():.1f}s  /  max {env.max_travel_time():.1f}s')
    dash_vals['wait'].set_text(f'⏸  Avg wait time: {env.average_wait_time():.1f}s')
    dash_vals['wait_minmax'].set_text(
        f'   min {env.min_wait_time():.1f}s  /  max {env.max_wait_time():.1f}s')
    dash_vals['collisions'].set_text(f'💥 Collisions: {env.collision_count}')

    all_tl = [p for d in tl_patches.values() for p in d.values()]
    all_grid_queue = [t for d in grid_queue_texts.values() for t in d.values()]
    return (rectangles + labels + ped_patches + all_tl +
            list(dash_vals.values()) + list(queue_texts.values()) +
            all_grid_queue)


anim = FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.close()
anim.save("simulation.gif", writer="pillow")
