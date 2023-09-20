import os

base_dir = 'data/tracks2/'

track_dirs = [base_dir + f for f in os.listdir(base_dir)]

for t in track_dirs:
    if len(os.listdir(t)) != 2:
        print(t)
        os.rmdir(t)