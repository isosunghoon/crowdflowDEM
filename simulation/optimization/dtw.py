from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
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

    name = sys.argv[1]
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