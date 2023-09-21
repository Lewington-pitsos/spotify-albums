import os
import timm
from constants import *
import json
import torchvision.transforms as tfms
import torch
from torch.utils.data import Dataset
from PIL import Image

MEAN_STD = (
    [0.40494787, 0.3638653,  0.36222387], 
    [0.3520901,  0.33091326, 0.32863148]
)

def _load_img(img_path):
    image = Image.open(img_path)
    n_channels = len(image.getbands())
    if n_channels in [1, 4]:
        image = image.convert('RGB')
    return image

class TrackDataset(Dataset):
    def __init__(self, tracks, model_name, model, size=32, mean_std=MEAN_STD, augment=False):
        self.tracks = tracks
        if model_name in TIMM_MODELS:
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            self.transform = transforms 
        else:
            if augment:
                base_tfms = [
                    tfms.RandomHorizontalFlip(),
                    tfms.RandomRotation(10),
                ]
            else:
                base_tfms = []
                
            base_tfms.extend([
                tfms.ToTensor(),
                tfms.Resize((size, size), antialias=True),
                tfms.Normalize(
                    mean=mean_std[0],
                    std=mean_std[1]
                )
            ])
            
            self.transform = tfms.Compose(base_tfms)


    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        img_path = track[ALBUM_IMG_KEY]
        image = _load_img(img_path)

        image = self.transform(image)
        return image, torch.tensor([track['popularity'] / 100]).to(torch.float32)


class InputIndependent(Dataset):
    def __init__(self, tracks):
        self.tracks = tracks
        self.transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Resize((32, 32)),
        ])
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        img_path = track[ALBUM_IMG_KEY]
        image = _load_img(img_path)

        image = self.transform(image)

        return torch.full_like(image, 0.5).to(torch.float32), torch.tensor([track['popularity'] / 100]).to(torch.float32)

def build_dataloader(model_name, model, tracks, batch_size, input_independent=False, img_size=32, augment=False, shuffle=False):
    if model_name in ['resnet-18', 'resnet-50']:
        mean_std = (
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    else:
        mean_std = MEAN_STD

    print("mean std", mean_std)

    if input_independent:
        dataset = InputIndependent(tracks, img_size)
    else:
        dataset = TrackDataset(tracks, model_name, model, img_size, mean_std, augment)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def get_track_dirs():
    track_dirs = [BASE_DIR + f for f in os.listdir(BASE_DIR)]
    return track_dirs


def load_data(track_dir):
    with open(track_dir + '/track_data.json') as json_file:
        data = json.load(json_file)

    data[ALBUM_IMG_KEY] = track_dir + '/album_image.jpg'

    return data

def load_tracks():
    return [load_data(track_dir) for track_dir in get_track_dirs()]
