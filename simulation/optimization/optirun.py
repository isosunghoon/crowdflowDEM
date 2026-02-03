import json
import os
import math
import numpy as np
from matplotlib.path import Path
import sys
import pandas as pd

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

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Vec):
            return obj.tolist()
        if isinstance(obj, Human):
            return obj.to_dict()
        if isinstance(obj, Obstacle):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

class Human:
    def __init__(self, pos, direction, color, speed, radius, mass, realid):
        self.pos = Vec(pos)
        self.direction = Vec(direction)
        self.vel = self.direction * speed
        self.force = Vec((0, 0))

        self.neighbor_list = []

        self.speed = speed
        self.radius = radius
        self.mass = mass
        self.color = color
        self.index = len(simulation['humans'])
        self.realid = realid
    
    def to_dict(self):
        return self.__dict__

class Obstacle:
    def __init__(self, points):
        self.points = points
        self.index = len(simulation['obstacles'])
    
    def to_dict(self):
        return self.__dict__
    
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


def init():
    # simulation['obstacles'].append(Obstacle([Vec((0,0)),Vec((50,0)),Vec((50,900)),Vec((0,900))]))
    # simulation['obstacles'].append(Obstacle([Vec((350,0)),Vec((400,0)),Vec((400,900)),Vec((350,900))]))
    simulation['df'] = pd.read_csv('../realworlddata_match/processed_tracks.csv')
    simulation['df'].sort_values(by='Starting Frame', inplace=True)
    simulation['humancount'] = 0

    os.makedirs(f'res/{simulation['name']}', exist_ok=True)
    os.makedirs(f'res/{simulation['name']}/dem', exist_ok=True)
    with open(f'./res/{simulation['name']}/dem/init.dem', 'w') as file:
        json.dump(simulation, file, indent=4, cls=CustomEncoder)

def add_human():
    while simulation['humancount']<len(simulation['df']) and t == 10*simulation['df']['Starting Frame'].iloc[simulation['humancount']]:
        v = eval(simulation['df'].iloc[simulation['humancount']]['Mean Velocity'])
        p = eval(simulation['df'].iloc[simulation['humancount']]['Initial Position'])
        r = (v[0]**2+v[1]**2)**0.5
        i = simulation['df'].iloc[simulation['humancount']]['Person ID']
        h = Human(realid = i, pos = p, direction = (v[0]/r,v[1]/r), speed = r, color = '#77E4C8', radius = simulation['radius'], mass = simulation['mass'])
        simulation['humans'].append(h)
        simulation['humancount'] += 1

    # if t <= 500:
    #     if t % 10 == 0:
    #         h = Human(pos = (np.random.randint(60, 340), 0), direction = (0,1), speed = 100, color = '#77E4C8', radius = simulation['radius'], mass = simulation['mass'])
    #         simulation['humans'].append(h)
    #     if t % 10 == 1:
    #         h = Human(pos = (np.random.randint(60, 340), 900), direction = (0,-1), speed = 100, color = '#478CCF', radius = simulation['radius'], mass = simulation['mass'])
    #         simulation['humans'].append(h)

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
        for human in simulation['humans']:
            if (human.radius > radius_max): radius_max = human.radius
        for human in simulation['humans']:
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
        
        for human in simulation['humans']:
            c = math.floor(C*(human.pos[0] - point_min[0]) / (C*alpha)) + 1
            r = math.floor(R*(human.pos[1] - point_min[1]) / ( R*alpha)) + 1
            grid[c][r].append(human)
        pair_list =[]
        for c in range(1,C+1):
            for r in range(1,R+1):
                for dc in range(-1,2):
                    for gr1 in grid[c][r]:
                        for gr2 in grid[c+dc][r+1]:
                            pair_list.append((gr1.index,gr2.index))
                for gr1 in grid[c][r]:
                    for gr2 in grid[c+1][r]:
                        pair_list.append((gr1.index,gr2.index))
                for i,gr1 in enumerate(grid[c][r]):
                    for j in range(i+1, len(grid[c][r])):
                        pair_list.append((gr1.index, grid[c][r][j].index))
        return pair_list

    # human-human
    for (n1, n2) in compute_colliding_pair():
        if n1==n2:
            continue
        human1, human2 = simulation['humans'][n1], simulation['humans'][n2]
        d = human1.pos.dist(human2.pos)
        if d<human1.radius + human2.radius:
            simulation['collided_human_pair'].append((n1,n2))
        else:
            if (d - human2.radius)/human1.radius < simulation['eye']:
                human1.neighbor_list.append({'type':'human','index':human2.index})
            if (d - human1.radius)/human2.radius < simulation['eye']:
                human2.neighbor_list.append({'type':'human','index':human1.index})

    # human-obstacle
    for obs in simulation['obstacles']:
        for human in simulation['humans']:
            d = obs.distance(human.pos)
            if d < human.radius:
                simulation['collided_human_obs'].append({'human':human.index,'obs':obs.index})
            elif d < simulation['eye']*human.radius:
                human.neighbor_list.append({'type':'obs','index':obs.index})

def human_will():
    def f1(arg, dist):
        return np.sign(arg+np.sign(np.random.random()-0.5)*1e-6) / (1 + abs(arg)*simulation['alpha']) / dist / dist
    def f2(dist):
        return np.sign(dist) * (np.pi/2 - 1/(simulation['beta']*abs(dist) + 2/np.pi))
    
    point = 0
    for human in simulation['humans']:
        optimal_arg = np.arctan2(human.direction[1],human.direction[0])

        for n in human.neighbor_list:
            if n['type'] == 'human':
                neighbor = simulation['humans'][n['index']]
                x, y = (neighbor.pos-human.pos)[0], (neighbor.pos-human.pos)[1]
                arg, dist = np.arctan2(y, x) - optimal_arg, (x**2+y**2)**(0.5)
                if abs(arg)<np.pi/2:
                    point -= f1(arg, dist)
            if n['type'] == 'obs':
                neighbor = simulation['obstacles'][n['index']]
                x, y = neighbor.distance(human.pos, xy = True)
                arg, dist = np.arctan2(y, x) - optimal_arg, (x**2+y**2)**(0.5)
                if abs(arg)<np.pi/2:
                    point -= f1(arg, dist)
        arg = f2(point)
        will_dir = human.speed * human.direction.rotate(arg) - human.vel
        human.force += will_dir * simulation['will_coef']

def human_collision():
    for (n1, n2) in simulation['collided_human_pair']:
        human1 = simulation['humans'][n1]
        human2 = simulation['humans'][n2]
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
    for t in simulation['collided_human_obs']:
        human = simulation['humans'][t['human']]
        obs = simulation['obstacles'][t['obs']]
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
    simulation['collided_human_pair'] = []
    simulation['collided_human_obs'] = []
    for human in simulation['humans']:
        dt = simulation['dt']
        acc = human.force/human.mass
        human.vel += acc * (dt/2)
        human.pos += human.vel * dt + 0.5*acc*(dt**2)
        human.force = 0
        human.neighbor_list = []

def save():
    with open(f'./res/{simulation['name']}/dem/{t}.dem', 'w') as file:
        json.dump(simulation, file, indent=4, cls=CustomEncoder)
    

def loop():
    if t%100==0:
        print(t)
    t += 1
    add_human()
    find_neighbor()
    if t%5==0:
        human_will()
    human_collision()
    obstacle_collision()
    save()
    update()

simulation = {'humans':[], 'obstacles':[], 'collided_human_pair':[], 'collided_human_obs':[], 't': 0, #variable
              'eye':float(sys.argv[3]), 'alpha':float(sys.argv[1]), 'beta':float(sys.argv[2]), 'will_coef':float(sys.argv[5]), 'stiffness':float(sys.argv[4]), 'restitution':float(sys.argv[6]), #hyperparameter
              'frame':5100, 'dt':0.004, 'w': 1200, 'h': 738, 'radius':10, 'mass':1, 'speed_mean':100, 'speed_var':30, #environment variable
              'name':'optimizetemp'}

hyp = {}

init()
for _ in range(simulation['frame']):
    loop()
# os.system("python draw.py")