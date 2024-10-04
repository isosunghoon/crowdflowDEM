import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.path import Path
import math
import time
import sys
import json

class Vec(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype='float64').view(cls)
        return obj
    
    def dist(self, other):
        return np.sqrt((self[0]-other[0])**2+(self[1]-other[1])**2)
    
    def rotate(self, angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.dot(rotation_matrix, self).flatten()

class Human:
    def __init__(self, pos, direction, color, speed, radius, mass):
        self.pos = Vec(pos)
        self.direction = Vec(direction)
        self.vel = self.direction * speed
        self.force = Vec((0, 0))

        self.neighbor_list = []

        self.speed = speed
        self.radius = radius
        self.mass = mass
        self.color = color
        self.index = len(humans)
        self.patch = patches.Circle(self.pos, radius=simulation['s_radius'], facecolor=self.color, edgecolor = 'black')
        ax.add_patch(self.patch)

class Obstacle:
    def __init__(self, points):
        self.points = points
        self.index = len(obstacles)
        self.patch = patches.Polygon(self.points, closed=True, linewidth=1, edgecolor='black', facecolor='grey')
        ax.add_patch(self.patch)
    
    def point_to_segment_distance(self, px, py, ax, ay, bx, by):
        ab = np.array([bx - ax, by - ay])
        ap = np.array([px - ax, py - ay])
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        closest_point = np.array([ax, ay]) + t * ab
        return closest_point - np.array([px, py])

    def point_to_polygon_distance(self, pos, xy):
        px, py = pos
        min_distance = float('inf')
        min_point = []
        for i in range(len(self.points)):
            ax, ay = self.points[i]
            bx, by = self.points[(i + 1) % len(self.points)]
            point = self.point_to_segment_distance(px, py, ax, ay, bx, by)
            if min_distance > np.linalg.norm(point):
                min_distance = np.linalg.norm(point)
                min_point = point
        if xy:
            return min_point
        else:
            return min_distance

    def is_point_in_polygon(self, pos):
        path = Path(self.points)
        return path.contains_point(pos)

    def distance(self, pos, xy = False):
        if self.is_point_in_polygon(pos):
            if xy:
                return [np.sign(np.random.random()-0.5)*1e-6,np.sign(np.random.random()-0.5)*1e-6]
            else:
                return 0.0
        else:
            return self.point_to_polygon_distance(pos, xy)

def add_human(t):
    if t <= 1000:
        s = np.random.normal(loc=simulation['speed_mean'], scale=simulation['speed_var'])
        if t % 40 == 0:
            h = Human(pos = (np.random.randint(240, 360), 600), direction = (0,-1), speed = s, color = '#179BAE', radius = simulation['radius'], mass = simulation['mass'])
            humans.append(h)
        if t % 40 == 1:
            h = Human(pos = (600, (np.random.randint(240, 360))), direction = (-1,0), speed = s, color = '#4158A6', radius = simulation['radius'], mass = simulation['mass'])
            humans.append(h)
        if t % 40 == 2:
            p = np.random.randint(-90,90)
            h = Human(pos = (max(p,0),-min(p,0)), direction = (1/2**0.5,1/2**0.5), speed = s, color = '#FF8343', radius = simulation['radius'], mass = simulation['mass'])            
            humans.append(h)
        if t % 40 == 3:
            p = np.random.randint(-90,90)
            h = Human(pos = (max(p,0),600+min(p,0)), direction = (1/2**0.5,-1/2**0.5), speed = s, color = '#F1DEC6', radius = simulation['radius'], mass = simulation['mass'])
            humans.append(h)

def find_neighbor():
    #LCM
    k                = 1.004
    domain_dimension = Vec((0.,0.))
    point_min        = Vec(( 1000. , 1000.))
    point_max        = Vec((-1000. ,-1000.))
    radius_max       = 0.

    def update_domain():
        nonlocal k, domain_dimension, point_min, point_max, radius_max
        radius_max = -1.
        for human in humans:
            if (human.radius > radius_max): radius_max = human.radius
        for human in humans:
            if (human.pos[0]  < point_min[0]) : point_min[0] = human.pos[0]
            if (human.pos[1]  < point_min[1]) : point_min[1] = human.pos[1]
            if (human.pos[0]  > point_max[0]) : point_max[0] = human.pos[0]
            if (human.pos[1]  > point_max[1]) : point_max[1] = human.pos[1]
            if (human.radius > radius_max) : radius_max = human.radius
        domain_dimension = point_max - point_min

    def compute_colliding_pair(expand_ratio=1.):
        nonlocal k, domain_dimension, point_min, point_max, radius_max
        update_domain()
        radius_max *= expand_ratio
        alpha = 2*k*radius_max
        C = math.floor(domain_dimension[0]/alpha)+1
        R = math.floor(domain_dimension[1]/alpha)+1
        grid =  [[[] for i in range(R+2)] for j in range(C+2)]
        
        for human in humans:
            c = math.floor(C*(human.pos[0] - point_min[0]) / (C*alpha)) + 1
            r = math.floor(R*(human.pos[1] - point_min[1]) / (R*alpha)) + 1
            grid[c][r].append(human.index)
        pair_list = []
        for c in range(1,C+1):
            for r in range(1,R+1):
                for dc in range(-1,2):
                    for gr1 in grid[c][r]:
                        for gr2 in grid[c+dc][r+1]:
                            pair_list.append((gr1,gr2))
                for gr1 in grid[c][r]:
                    for gr2 in grid[c+1][r]:
                        pair_list.append((gr1,gr2))
                for i,gr1 in enumerate(grid[c][r]):
                    for j in range(i+1, len(grid[c][r])):
                        pair_list.append((gr1, grid[c][r][j]))
        return pair_list

    # human-human
    for (n1, n2) in compute_colliding_pair():
        if n1==n2:
            continue
        human1, human2 = humans[n1], humans[n2]
        d = human1.pos.dist(human2.pos)
        if d<human1.radius + human2.radius:
            collided_human_pair.append((n1,n2))
        if (d - human2.radius)/human1.radius < simulation['eye']:
            human1.neighbor_list.append({'type':'human','index':human2.index})
        if (d - human1.radius)/human2.radius < simulation['eye']:
            human2.neighbor_list.append({'type':'human','index':human1.index})

    # human-obstacle
    for obs in obstacles:
        for human in humans:
            d = obs.distance(human.pos)
            if d < human.radius:
                collided_human_obs.append({'human':human.index,'obs':obs.index})
            if d < simulation['eye']*human.radius:
                human.neighbor_list.append({'type':'obs','index':obs.index})
m = 0
def human_will():
    def f1(arg, dist):
        return np.sign(arg+1e-6) / (1 + abs(arg)*simulation['alpha']) / dist / dist
    def f2(dist):
        return np.sign(dist) * (np.pi/2 - 1/(simulation['beta']*abs(dist) + 2/np.pi))
    
    for human in humans:
        point = 0
        optimal_arg = np.arctan2(human.direction[1],human.direction[0])

        for n in human.neighbor_list:
            if n['type'] == 'human':
                try:
                    neighbor = humans[n['index']]
                    x, y = (neighbor.pos-human.pos)[0], (neighbor.pos-human.pos)[1]
                    arg, dist = np.arctan2(y, x) - optimal_arg, (x**2+y**2)**(0.5)
                    if abs(arg)<simulation['angle']*np.pi/180:
                        if neighbor.direction.dist(human.direction) < 1:
                            point -= f1(arg, dist)/simulation['same_penalty']
                        else:
                            point -= f1(arg, dist)
                except:
                    pass
            if n['type'] == 'obs':
                neighbor = obstacles[n['index']]
                x, y = neighbor.distance(human.pos, xy = True)
                arg, dist = np.arctan2(y, x) - optimal_arg, (x**2+y**2)**(0.5)
                if abs(arg)<simulation['angle']*np.pi/180:
                    point -= f1(arg, dist)*simulation['obstacle_advantage']
        
        # global m
        # if abs(point)>m:
        #     print(point)
        #     m = abs(point)
        
        arg = f2(point)
        if abs(arg)<simulation['min_arg']*np.pi/180: arg=0
        will_dir = human.speed * human.direction.rotate(arg) - human.vel
        human.force += will_dir * simulation['will_coef']

def human_collision():
    for (n1, n2) in collided_human_pair:
        human1 = humans[n1]
        human2 = humans[n2]
        rel_pos = human2.pos - human1.pos
        delta = -human1.pos.dist(human2.pos) + human1.radius + human2.radius
        if (delta >0.):
            K = simulation['stiffness']
            K2 = simulation['restitution']
            normal = rel_pos/human1.pos.dist(human2.pos)
            force1 = normal * delta * K
            human1.force -= force1
            human2.force += force1
            M  = (human1.mass*human2.mass)/(human1.mass+human2.mass)
            C  = 2.*(1./math.sqrt(1. + math.pow(math.pi/math.log(K2), 2)))*math.sqrt(K*M)
            V  = (human2.vel - human1.vel) * normal
            force2 = C * V * normal
            human1.force += force2
            human2.force -= force2

def obstacle_collision():
    for t in collided_human_obs:
        human = humans[t['human']]
        obs = obstacles[t['obs']]
        x, y = obs.distance(human.pos, xy = True)
        r = (x**2+y**2)**(0.5)
        K = simulation['stiffness']
        K2 = simulation['restitution']
        delta = -r + human.radius
        normal = Vec((x/r,y/r))
        force1 = normal * delta * K
        human.force -= force1
        M  = human.mass
        C  = 2.*(1./math.sqrt(1. + math.pow(math.pi/math.log(K2), 2)))*math.sqrt(K*M)
        V  = - human.vel * normal
        force2 = C * V * normal
        human.force += force2

def update():
    global collided_human_pair, collided_human_obs
    collided_human_pair = []
    collided_human_obs = []
    for human in humans:
        dt = simulation['dt']
        acc = human.force/human.mass
        human.vel += acc * (dt/2)
        human.pos += human.vel * dt + 0.5*acc*(dt**2)
        human.force = Vec((0, 0))
        human.neighbor_list = []
        human.patch.center = human.pos

        out_of_screen = max(-human.pos[0],-human.pos[1],human.pos[0]-simulation['w'],human.pos[1]-simulation['h'])
        if out_of_screen>simulation['padding']:
            humans.remove(human)

    for i, human in enumerate(humans):
        human.index = i

def loop(t):
    global debug_var
    if t%100==0:
        print(f"frame {t} done, time {time.time()-debug_var}")
        debug_var = time.time()
    add_human(t)
    find_neighbor()
    if t % simulation['will_rate'] == 0:
        human_will()
    human_collision()
    obstacle_collision()
    update()
    return [x.patch for x in humans]+[x.patch for x in obstacles]

info = []
with open(sys.argv[1], 'r') as file:
    info = json.load(file)

humans, obstacles, collided_human_pair, collided_human_obs = [], [], [], []
simulation = {'eye':info['eye'],'alpha':info['alpha'], 'beta':info['beta'], 'will_coef':info['will_coef'], 'stiffness':info['stiffness'], 'restitution':info['restitution'], #hyperparameter
              'frame':4000, 'dt':0.004, 'w': 600, 'h': 600, 'radius':10, 'mass':500, 'speed_mean':info['speed_mean'], 'speed_var':info['speed_var'], 'padding':10, #environment variable
              'human_rate':10,'will_rate':info['will_rate'], 'same_penalty':info['same_penalty'], 'obstacle_advantage':info['obstacle_advantage'],'angle':info['angle'],'s_radius':info['s_radius'],'min_arg':info['min_arg'] #modification
            }
debug_var = time.time()

fig, ax = plt.subplots()
ax.set_xlim(0, simulation['w'])
ax.set_ylim(0, simulation['h'])
ax.set_aspect('equal')

obstacles.append(Obstacle([Vec((100,0)),Vec((230,0)),Vec((230,130))]))
obstacles.append(Obstacle([Vec((100,600)),Vec((230,600)),Vec((230,470))]))
obstacles.append(Obstacle([Vec((500,0)),Vec((370,0)),Vec((370,130))]))
obstacles.append(Obstacle([Vec((500,600)),Vec((370,600)),Vec((370,470))]))
obstacles.append(Obstacle([Vec((0,100)),Vec((0,230)),Vec((130,230))]))
obstacles.append(Obstacle([Vec((600,100)),Vec((600,230)),Vec((470,230))]))
obstacles.append(Obstacle([Vec((0,500)),Vec((0,370)),Vec((130,370))]))
obstacles.append(Obstacle([Vec((600,500)),Vec((600,370)),Vec((470,370))]))

ani = animation.FuncAnimation(fig, loop, frames=range(1,simulation['frame']+1), blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=round(1/simulation['dt']), bitrate=1800)
ani.save(f'./res/{sys.argv[0].split('.')[0]}_{''.join(sys.argv[1].split('.')[:-1]).split('/')[-1]}.mp4', writer=writer, dpi = 400)