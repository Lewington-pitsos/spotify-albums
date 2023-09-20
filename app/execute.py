import uuid
import torch
import fire
import random
import datetime
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from constants import *
from dataload import load_tracks, build_dataloader
from models import build_model
from training import train_basic_conv
import wandb

def random_predictions(tracks):
    return [random.random() for _ in tracks]

def basic_conv(
        trn, 
        tst, 
        model, 
        batch_size=4, 
        input_independent=False, 
        epochs=2, 
        one_batch=False, 
        img_size=32, 
        lr=1e-3, 
        bias=None, 
        freeze=False,
        hidden_head_size=None, **kwargs):
    
    m = build_model(model, bias, freeze, hidden_head_size)

    if one_batch:
        trn = trn[:batch_size]

    train_basic_conv(model, m, trn, batch_size, input_independent, epochs, img_size, lr)

    with torch.no_grad():
        predictions = []

        dataloader = build_dataloader(model, m, tst, batch_size, input_independent=input_independent, img_size=img_size)

        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = m(inputs)
            for prediction in outputs:
                predictions.append(prediction.item())
    
        return predictions

def make_predictions(trn, tst, run):
    if run['model'] == 'random':
        return random_predictions(tst)
    elif run['model'] == 'constant':
        return [run['constant_value']] * len(tst)
    elif run['strategy'] == 'cnn':
        return basic_conv(
            trn, 
            tst, 
            **run
        )
    else:
        raise Exception('unknown model: ' + run['model'])

def calculate_metrics(tracks, predictions):
    gt = [track['popularity'] / 100 for track in tracks]
    
    return {
        'mae': mean_absolute_error(gt, predictions),
        'mse': mean_squared_error(gt, predictions),
        'rmse': np.sqrt(mean_squared_error(gt, predictions))
    }

def load_unique_album_tracks():
    data = load_tracks()

    album_set = set()

    wanted = []
    for track in data:
        if track['album']['id'] not in album_set:
            album_set.add(track['album']['id'])
            wanted.append(track)

    return wanted

def perform_run(run):
    wandb.init(project='popularity', entity="lewington", name=run['name'] + "_" + str(uuid.uuid4()), config=run)
    data = load_unique_album_tracks()

    print('number of unique tracks:', len(data))

    trn = data[:-500]
    tst = data[-500:]

    trn_predictions = make_predictions(trn, tst, run)
    metrics = calculate_metrics(tst, trn_predictions)

    sample_predictions = []

    wandb.log(metrics)

    metrics['sample_predictions'] = sample_predictions

    wandb.finish()
    return metrics

def main(filename):
    with open(filename) as json_file:
        runs = json.load(json_file)

    for run in runs['runs']:
        perform_run(run)

if __name__ == '__main__':
    random.seed(42)
    fire.Fire(main)