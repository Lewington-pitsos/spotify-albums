import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
from concurrent.futures import ThreadPoolExecutor
import shutil

def save_idx_track(*args, idx):
    final = save_track(*args)

    if idx % 100 == 0:
        print('finished with', idx)

    return final

def save_track(sp, track_id, base_dir):
    t = sp.track(track_id)
    img_url = t['album']['images'][0]['url']
    
    track_dir = base_dir + '/' + track_id
    # Create directory with track_id as name
    if not os.path.exists(track_dir):
        os.makedirs(track_dir)
    
    # Download and save the album image
    img_response = requests.get(img_url, stream=True)
    img_response.raise_for_status()
    with open(os.path.join(track_dir, 'album_image.jpg'), 'wb') as img_file:
        img_response.raw.decode_content = True
        shutil.copyfileobj(img_response.raw, img_file)
    
    # Save the track data as JSON
    with open(os.path.join(track_dir, 'track_data.json' ), 'w') as json_file:
        json.dump(t, json_file, indent=4)

with open('credentials/spotify.json') as f:
    cred = json.load(f)

scope = "user-read-recently-played"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cred['client_id'], client_secret= cred['secret'], redirect_uri='https://example.com/callback', scope=scope))

with open('data/40_000_ids.json') as f:
    chosen = json.load(f)

data_dir = 'data/tracks2'

downloaded = os.listdir(data_dir)

chosen = [track_id for track_id in chosen if track_id not in downloaded]

print('downloading', len(chosen), 'tracks')

for i, track_id in enumerate(chosen):
    save_idx_track(sp, track_id, data_dir, idx=i)

# with ThreadPoolExecutor(max_workers=5) as executor:

#     futures = [executor.submit(save_idx_track, sp, track_id, data_dir, idx=i) for i, track_id in enumerate(chosen)]

#     results = [f.result() for f in futures]