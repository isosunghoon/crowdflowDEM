import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
import pickle

def save_checkpoint(result, filename='checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def load_checkpoint(filename='checkpoint.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        return result
    return None

def dtw():
    def extract_track_segment(track, start_frame, num_frames):
        end_frame = start_frame + num_frames
        segment = [(frame, pos) for frame, pos in track if start_frame <= frame < end_frame]
        return segment

    def compare_tracks_dtw(track1, track2):
        positions1 = [pos for _, pos in track1]
        positions2 = [pos for _, pos in track2]
        
        distance, _ = fastdtw(positions1, positions2, dist=euclidean)
        return distance

    def compare_all_tracks(tracks1, tracks2, start_frame, num_frames):
        dtw_distances = {}
        for person_id in tracks1.keys():
            if person_id in tracks2:
                track1 = extract_track_segment(tracks1[person_id], start_frame, num_frames)
                track2 = extract_track_segment(tracks2[person_id], start_frame, num_frames)
                track2 = track2[:len(track1)]
                # if person_id == 54:
                #     print(track1)
                #     print(track2)
                dtw_distance = compare_tracks_dtw(track1, track2)
                dtw_distances[person_id] = dtw_distance
        return dtw_distances

    with open(f'./tracks.txt', 'r') as f:
        data = f.readlines()

    tracks = {}

    for line in data:
        line = list(map(int, line.split()))
        if line[0] not in tracks.keys():
            tracks[line[0]] = []
        tracks[line[0]].append((line[1]+1, (line[2],line[3]-100)))

    name = 'optimizetemp'
    start_frame = 0
    num_frames = 510

    with open(f'./res/{name}/data.dtw', 'r') as f:
        data = f.readlines()

    tracks2 = {}

    for line in data:
        line = list(map(int, line.split()))
        if line[0] not in tracks2.keys():
            tracks2[line[0]] = []
        tracks2[line[0]].append((line[1], tuple(line[2:])))

    mean_lasting_frame = 136
    dtw_distances = compare_all_tracks(tracks, tracks2, start_frame, num_frames)
    m = sum(dtw_distances.values())/len(dtw_distances)/mean_lasting_frame
    return m

param_space = [
    Real(1, 1e5, name='alpha'),
    Real(1, 1e5, name='beta'),
    Real(1, 10, name='chi'),
    Real(1, 1e6, name='fp'),
    Real(1, 1000, name='fm'),
    Real(0.01, 0.7, name='k')
]

def evaluate_model(param1, param2, param3, param4, param5, param6):
    os.system(f"python optirun.py {param1} {param2} {param3} {param4} {param5} {param6}")
    os.system(f"python gendata.py optimizetemp")
    return dtw()
x = 0
@use_named_args(param_space)
def objective(**params):
    global x
    x += 1
    dtw_score = evaluate_model(params['alpha'], params['beta'], params['chi'], 
                              params['fp'], params['fm'], params['k'])
    print(f"iteration {x}")
    return dtw_score

checkpoint_file = 'checkpoint.pkl'
result = gp_minimize(objective, param_space, n_calls=50, random_state=0)
save_checkpoint(result, checkpoint_file)

# Best parameters found
best_params = result.x
best_score = result.fun

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_score}")

plot_convergence(result)
plt.tight_layout()
plt.savefig('convergence.png')
plot_evaluations(result)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('evaluation.png')
plot_objective(result)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('objective.png')
