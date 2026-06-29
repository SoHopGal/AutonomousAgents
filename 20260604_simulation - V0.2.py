import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from IPython.display import HTML

matplotlib.rcParams['animation.embed_limit'] = 100

# =====================================
# Traffic Light
# =====================================
class TrafficLight:

    def __init__(self, axis):
        """
        axis: 'horizontal' or 'vertical'
        Green for horizontal means horizontal vehicles can go.
        """
        self.axis = axis
        self.state = 'green' if axis == 'horizontal' else 'red'
        self.timer = 0.0
        self.green_duration = 8.0
        self.red_duration = 8.0
        self.yellow_duration = 2.0
        self.phase = 'green'  # green / yellow / red

    def step(self, dt):
        self.timer += dt

        if self.phase == 'green':
            if self.timer >= self.green_duration:
                self.phase = 'yellow'
                self.timer = 0.0

        elif self.phase == 'yellow':
            if self.timer >= self.yellow_duration:
                self.phase = 'red'
                self.timer = 0.0

        elif self.phase == 'red':
            if self.timer >= self.red_duration:
                self.phase = 'green'
                self.timer = 0.0

        self.state = self.phase

    def is_green(self):
        return self.phase == 'green'

    def color_rgb(self):
        if self.phase == 'green':
            return '#2ECC71'
        elif self.phase == 'yellow':
            return '#F1C40F'
        else:
            return '#E74C3C'


class IntersectionTrafficLights:
    """
    Two complementary lights: one for horizontal, one for vertical.
    When horizontal is green, vertical is red and vice versa.
    """

    def __init__(self):
        self.horizontal = TrafficLight('horizontal')
        self.vertical = TrafficLight('vertical')
        # Start opposite
        self.vertical.phase = 'red'
        self.vertical.timer = 0.0

    def step(self, dt):
        self.horizontal.step(dt)

        # Vertical is always opposite to horizontal
        if self.horizontal.phase == 'green':
            self.vertical.phase = 'red'
        elif self.horizontal.phase == 'yellow':
            self.vertical.phase = 'red'
        elif self.horizontal.phase == 'red':
            self.vertical.phase = 'green'

    def can_go(self, vehicle):
        if vehicle.is_emergency:
            return True
        if vehicle.horizontal:
            return self.horizontal.is_green()
        else:
            return self.vertical.is_green()


# =====================================
# Reservation Manager
# =====================================
class ReservationManager:

    def __init__(self):
        self.slots = {}

    def request(self, vehicle):
        for vid, slot in self.slots.items():
            if vid == vehicle.vehicle_id:
                return True
            if self._conflicts(vehicle, slot):
                return False
        self.slots[vehicle.vehicle_id] = vehicle
        return True

    def _conflicts(self, v1, v2):
        return v1.horizontal != v2.horizontal

    def release(self, vehicle):
        self.slots.pop(vehicle.vehicle_id, None)

    def force_grant(self, vehicle):
        """Emergency override — clear conflicting slots."""
        to_remove = [
            vid for vid, v in self.slots.items()
            if self._conflicts(vehicle, v)
        ]
        for vid in to_remove:
            self.slots.pop(vid, None)
        self.slots[vehicle.vehicle_id] = vehicle


# =====================================
# Vehicle
# =====================================
DIRECTION_COLORS = {
    'right': '#4A90D9',
    'left':  '#E74C3C',
    'up':    '#9B59B6',
    'down':  '#F39C12',
}

# Weights from SDD unified utility framework
W_SAFETY     = 0.50
W_EFFICIENCY = 0.25
W_COMFORT    = 0.15
W_RULES      = 0.10

# Priority weights from SDD priority function
W1_EMERGENCY = 10.0
W2_WAITING   = 1.0
W3_DISTANCE  = 1.0
W4_DENSITY   = 0.5


class Vehicle:

    def __init__(self, vehicle_id, x, y, vx, vy, horizontal,
                 is_emergency=False):

        self.vehicle_id   = vehicle_id
        self.pos          = np.array([x, y], dtype=np.float64)
        self.v            = np.array([vx, vy], dtype=np.float64)
        self.original_v   = self.v.copy()
        self.horizontal   = horizontal
        self.has_crossed  = False
        self.wait_time    = 0.0
        self.waiting      = False
        self.is_emergency = is_emergency
        self.speed        = np.linalg.norm(self.original_v)

        if horizontal:
            self.w, self.h = 4, 2
        else:
            self.w, self.h = 2, 4

        if vx > 0:
            self.direction = 'right'
        elif vx < 0:
            self.direction = 'left'
        elif vy > 0:
            self.direction = 'up'
        else:
            self.direction = 'down'

        if is_emergency:
            self.color = '#FFFFFF'
        else:
            self.color = DIRECTION_COLORS[self.direction]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def inside_intersection(self):
        return -4 <= self.pos[0] <= 4 and -4 <= self.pos[1] <= 4

    def reached_stop_line(self):
        if self.original_v[0] > 0:
            return self.pos[0] + self.w / 2 >= -6
        if self.original_v[0] < 0:
            return self.pos[0] - self.w / 2 <= 6
        if self.original_v[1] > 0:
            return self.pos[1] + self.h / 2 >= -6
        if self.original_v[1] < 0:
            return self.pos[1] - self.h / 2 <= 6
        return False

    def distance_to_intersection(self):
        """Approximate distance to intersection center."""
        return np.linalg.norm(self.pos)

    def move(self, dt):
        self.pos += self.v * dt

    # ------------------------------------------------------------------
    # SDD: Priority function
    # Priority = w1*Emergency + w2*WaitingTime + w3*(1/Distance) + w4*Density
    # ------------------------------------------------------------------
    def compute_priority(self, traffic_density=1.0):
        emergency_flag = 1.0 if self.is_emergency else 0.0
        dist = max(self.distance_to_intersection(), 0.1)
        priority = (
            W1_EMERGENCY * emergency_flag
            + W2_WAITING  * self.wait_time
            + W3_DISTANCE * (1.0 / dist)
            + W4_DENSITY  * traffic_density
        )
        return priority

    # ------------------------------------------------------------------
    # SDD: Unified utility function
    # U = wsafety*Jsafety + wefficiency*Jefficiency +
    #     wcomfort*Jcomfort + wrules*Jrules
    # ------------------------------------------------------------------
    def compute_utility(self, nearby_vehicles, traffic_light_green,
                        dist_to_front=None):

        # Jsafety: 1 if no close vehicle ahead, else decreases
        if dist_to_front is not None:
            j_safety = min(dist_to_front / 10.0, 1.0)
        else:
            j_safety = 1.0

        # Jefficiency: 1 if moving at full speed, 0 if stopped
        current_speed = np.linalg.norm(self.v)
        j_efficiency = current_speed / max(self.speed, 0.1)

        # Jcomfort: penalize waiting (waiting = discomfort)
        j_comfort = 0.0 if self.waiting else 1.0

        # Jrules: 1 if green light (or emergency), 0 if red
        if self.is_emergency:
            j_rules = 1.0
        else:
            j_rules = 1.0 if traffic_light_green else 0.0

        utility = (
            W_SAFETY     * j_safety
            + W_EFFICIENCY * j_efficiency
            + W_COMFORT    * j_comfort
            + W_RULES      * j_rules
        )

        return utility, {
            'safety':     round(j_safety, 2),
            'efficiency': round(j_efficiency, 2),
            'comfort':    round(j_comfort, 2),
            'rules':      round(j_rules, 2),
            'total':      round(utility, 2),
        }


# =====================================
# Environment
# =====================================
class Env:

    def __init__(self):
        self.dt              = 0.1
        self.t               = 0.0
        self.manager         = ReservationManager()
        self.traffic_lights  = IntersectionTrafficLights()
        self.vehicles        = []
        self.next_vehicle_id = 1
        self.spawn_timer     = 0.0
        self.spawn_interval  = 2.5
        self.emergency_timer = 0.0
        self.emergency_interval = 30.0  # emergency vehicle every 30s

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
    def spawn_vehicle(self, is_emergency=False):
        lane = np.random.randint(0, 8)
        speed = 10 if is_emergency else 6

        configs = [
            (-45, -1,  speed,  0,    True),
            (-45, -3,  speed,  0,    True),
            ( 45,  1, -speed,  0,    True),
            ( 45,  3, -speed,  0,    True),
            (  1, -45,  0,  speed,   False),
            (  3, -45,  0,  speed,   False),
            ( -1,  45,  0, -speed,   False),
            ( -3,  45,  0, -speed,   False),
        ]

        x, y, vx, vy, horizontal = configs[lane]
        self.add_vehicle(x, y, vx, vy, horizontal, is_emergency)

    def add_vehicle(self, x, y, vx, vy, horizontal, is_emergency=False):
        v = Vehicle(
            self.next_vehicle_id,
            x, y, vx, vy, horizontal,
            is_emergency=is_emergency
        )
        self.next_vehicle_id += 1
        self.vehicles.append(v)

    # ------------------------------------------------------------------
    # Lane helpers
    # ------------------------------------------------------------------
    def same_lane(self, v1, v2):
        same_dir = np.allclose(v1.original_v, v2.original_v, atol=0.1)
        if not same_dir:
            return False
        if v1.horizontal:
            return abs(v1.pos[1] - v2.pos[1]) < 1.5
        else:
            return abs(v1.pos[0] - v2.pos[0]) < 1.5

    def is_blocked_by_front(self, vehicle):
        """Return (blocked, distance_to_front_vehicle)."""
        min_dist = None
        for other in self.vehicles:
            if vehicle is other:
                continue
            if not self.same_lane(vehicle, other):
                continue
            dist = np.linalg.norm(vehicle.pos - other.pos)
            if dist >= 8:
                continue
            # Check that other is *ahead*
            ahead = False
            if vehicle.original_v[0] > 0 and other.pos[0] > vehicle.pos[0]:
                ahead = True
            elif vehicle.original_v[0] < 0 and other.pos[0] < vehicle.pos[0]:
                ahead = True
            elif vehicle.original_v[1] > 0 and other.pos[1] > vehicle.pos[1]:
                ahead = True
            elif vehicle.original_v[1] < 0 and other.pos[1] < vehicle.pos[1]:
                ahead = True
            if ahead:
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        if min_dist is not None and min_dist < 7:
            return True, min_dist
        return False, None

    def traffic_density(self):
        return len(self.vehicles)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self):
        self.t += self.dt
        self.spawn_timer     += self.dt
        self.emergency_timer += self.dt

        if self.spawn_timer >= self.spawn_interval:
            self.spawn_vehicle()
            self.spawn_timer = 0.0

        if self.emergency_timer >= self.emergency_interval:
            self.spawn_vehicle(is_emergency=True)
            self.emergency_timer = 0.0

        self.traffic_lights.step(self.dt)

        density = self.traffic_density()

        for vehicle in self.vehicles:

            blocked, dist_front = self.is_blocked_by_front(vehicle)

            # Compute utility & priority every step (agent self-reasoning)
            light_green = self.traffic_lights.can_go(vehicle)
            vehicle.compute_utility(
                self.vehicles,
                light_green,
                dist_to_front=dist_front
            )
            vehicle.compute_priority(traffic_density=density)

            if blocked:
                vehicle.waiting = True
                vehicle.wait_time += self.dt
                vehicle.v = np.zeros(2)
                continue

            if (
                not vehicle.has_crossed
                and vehicle.reached_stop_line()
                and not vehicle.inside_intersection()
            ):
                # Red light check
                if not self.traffic_lights.can_go(vehicle):
                    vehicle.waiting = True
                    vehicle.wait_time += self.dt
                    vehicle.v = np.zeros(2)
                    continue

                # Reservation
                if vehicle.is_emergency:
                    self.manager.force_grant(vehicle)
                else:
                    granted = self.manager.request(vehicle)
                    if not granted:
                        vehicle.waiting = True
                        vehicle.wait_time += self.dt
                        vehicle.v = np.zeros(2)
                        continue

            vehicle.waiting = False
            vehicle.v = vehicle.original_v.copy()
            vehicle.move(self.dt)

            if vehicle.inside_intersection():
                vehicle.has_crossed = True

            if vehicle.has_crossed and not vehicle.inside_intersection():
                self.manager.release(vehicle)

        self.vehicles = [
            v for v in self.vehicles
            if abs(v.pos[0]) < 55 and abs(v.pos[1]) < 55
        ]


# =====================================
# Setup
# =====================================
env = Env()
for _ in range(4):
    env.spawn_vehicle()


# =====================================
# Figure
# =====================================
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-48, 48)
ax.set_ylim(-48, 48)
ax.set_aspect('equal')
ax.set_facecolor('#1A252F')
fig.patch.set_facecolor('#1A252F')

# Roads
ax.axhspan(-4, 4, color='#5D6D7E', zorder=1)
ax.axvspan(-4, 4, color='#5D6D7E', zorder=1)
ax.add_patch(Rectangle((-4, -4), 8, 8, color='#7F8C8D', zorder=2))

# Lane markings
for y in [-2, 0, 2]:
    ax.plot([-48, -4], [y, y],
            '--' if y == 0 else '-',
            color='#F1C40F' if y == 0 else 'white',
            lw=1 if y == 0 else 0.5,
            alpha=0.6, zorder=3)
    ax.plot([4, 48], [y, y],
            '--' if y == 0 else '-',
            color='#F1C40F' if y == 0 else 'white',
            lw=1 if y == 0 else 0.5,
            alpha=0.6, zorder=3)

for x in [-2, 0, 2]:
    ax.plot([x, x], [-48, -4],
            '--' if x == 0 else '-',
            color='#F1C40F' if x == 0 else 'white',
            lw=1 if x == 0 else 0.5,
            alpha=0.6, zorder=3)
    ax.plot([x, x], [4, 48],
            '--' if x == 0 else '-',
            color='#F1C40F' if x == 0 else 'white',
            lw=1 if x == 0 else 0.5,
            alpha=0.6, zorder=3)

# Stop lines
stop_lines = [
    ([-6, -6], [-4, 4]),
    ([6, 6],   [-4, 4]),
    ([-4, 4],  [-6, -6]),
    ([-4, 4],  [6, 6]),
]
for xl, yl in stop_lines:
    ax.plot(xl, yl, color='white', lw=2.5, zorder=3)

# Traffic light patches (visual indicators)
tl_patches = {
    'h_left':  ax.add_patch(Rectangle((-7, -1), 1, 2,
                             color='gray', zorder=4)),
    'h_right': ax.add_patch(Rectangle((6, -1),  1, 2,
                             color='gray', zorder=4)),
    'v_bottom':ax.add_patch(Rectangle((-1, -7), 2, 1,
                             color='gray', zorder=4)),
    'v_top':   ax.add_patch(Rectangle((-1,  6), 2, 1,
                             color='gray', zorder=4)),
}

# Legend
legend_elements = [
    Patch(facecolor=DIRECTION_COLORS['right'], label='→ Right'),
    Patch(facecolor=DIRECTION_COLORS['left'],  label='← Left'),
    Patch(facecolor=DIRECTION_COLORS['up'],    label='↑ Up'),
    Patch(facecolor=DIRECTION_COLORS['down'],  label='↓ Down'),
    Patch(facecolor='#FFFFFF',                 label='🚑 Emergency'),
    Patch(facecolor='#FF6B6B',                 label='⏸ Waiting'),
]
legend = ax.legend(
    handles=legend_elements, loc='upper right',
    facecolor='#2C3E50', labelcolor='white', fontsize=8
)

# Text overlays
time_text  = ax.text(-46, 44, '', fontsize=10, color='white', zorder=10)
count_text = ax.text(-46, 41, '', fontsize=10, color='white', zorder=10)
light_text = ax.text(-46, 38, '', fontsize=10, color='white', zorder=10)

rectangles = []
labels     = []


# =====================================
# Animation
# =====================================
def update(frame):
    env.step()

    global rectangles, labels

    for r in rectangles:
        r.remove()
    for l in labels:
        l.remove()
    rectangles = []
    labels     = []

    # Update traffic light colors
    h_color = env.traffic_lights.horizontal.color_rgb()
    v_color = env.traffic_lights.vertical.color_rgb()
    tl_patches['h_left'].set_facecolor(h_color)
    tl_patches['h_right'].set_facecolor(h_color)
    tl_patches['v_bottom'].set_facecolor(v_color)
    tl_patches['v_top'].set_facecolor(v_color)

    for vehicle in env.vehicles:
        color = '#FF6B6B' if vehicle.waiting else vehicle.color

        rect = Rectangle(
            (vehicle.pos[0] - vehicle.w / 2,
             vehicle.pos[1] - vehicle.h / 2),
            vehicle.w, vehicle.h,
            color=color, zorder=5
        )
        ax.add_patch(rect)
        rectangles.append(rect)

        prefix = '🚑' if vehicle.is_emergency else ''
        lbl = ax.text(
            vehicle.pos[0], vehicle.pos[1],
            f'{prefix}{vehicle.vehicle_id}',
            fontsize=6, ha='center', va='center',
            color='white', fontweight='bold', zorder=6
        )
        labels.append(lbl)

    h_state = env.traffic_lights.horizontal.phase.upper()
    v_state = env.traffic_lights.vertical.phase.upper()

    time_text.set_text(f'Time: {env.t:.1f}s')
    count_text.set_text(f'Vehicles: {len(env.vehicles)}')
    light_text.set_text(
        f'H-light: {h_state}   V-light: {v_state}'
    )

    return (rectangles + labels +
            [time_text, count_text, light_text] +
            list(tl_patches.values()))


anim = FuncAnimation(
    fig, update,
    frames=400,
    interval=50,
    blit=True
)

plt.close()
HTML(anim.to_jshtml())