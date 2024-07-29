import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import json

name = 'temp'
with open(f'./res/{name}/init.dem', 'r') as file:
    simulation = json.load(file)
fig, ax = plt.subplots()
ax.set_xlim(0, simulation['w'])
ax.set_ylim(0, simulation['h'])
ax.set_aspect('equal')

human_li, obs_li = [], []
for obs in simulation['obstacles']:
    temp = patches.Polygon(obs['points'], closed=True, linewidth=1, edgecolor='black', facecolor='grey')
    ax.add_patch(temp)
    obs_li.append(temp)

def loop(frame):
    if frame % (simulation['frame']//100) == 0:
        print(f'{frame/(simulation['frame']//100)}% done')
    with open(f'./res/{name}/{frame}.dem', 'r') as file:
        data = json.load(file)
    for idx, human in enumerate(data['humans']):
        if idx <len(human_li):
            human_li[idx].center = human['pos']
        else:
            temp = patches.Circle(human['pos'], radius=human['radius'], facecolor=human['color'], edgecolor = 'black')
            ax.add_patch(temp)
            human_li.append(temp)
    return human_li + obs_li

ani = animation.FuncAnimation(fig, loop, frames=range(1,simulation['frame']+1), blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, bitrate=1800)
ani.save(f'./res/{name}/result.mp4', writer=writer, dpi = 400)