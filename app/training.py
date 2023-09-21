import wandb
from dataload import build_dataloader
import torch.optim as optim
import torch.nn as nn
import torch
import os
from constants import device
import uuid
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_basic_conv(model_name, model, tracks, tst_tracks, batch_size=4, input_independent=False, epochs=2, img_size=32, lr=1e-3, early_stopping=False, augment=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataloader = build_dataloader(model_name, model, tracks, batch_size, input_independent, img_size, augment, shuffle=True)

    model = model.to(device)

    scores = []
    model_paths = []
    for epoch in range(epochs):  
        running_loss = 0.0
        count = 0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            if i == 0:
                image_list = []
                for i in range(inputs.shape[0])[:10]:
                    true_label = labels[i].cpu().detach().numpy().tolist()
                    predicted_output = outputs[i].cpu().detach().numpy().tolist()
                    caption = "True Label: {}, Predicted: {}".format(true_label, predicted_output)
                    image_list.append(
                        wandb.Image(
                            inputs[i].cpu().detach().numpy().transpose(1, 2, 0),
                            caption=caption
                        )
                    )

                # Log the entire list of images along with their corresponding labels and outputs
                wandb.log({
                    "images": image_list,
                    'batch_labels': labels.cpu().detach().numpy(),
                    'batch_outputs': outputs.cpu().detach().numpy()
                })
            loss = criterion(outputs, labels)
            
            wandb.log({'epoch': epoch, 'loss': loss})
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
            if i % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] mse loss: {running_loss / count:.3f}')
                running_loss = 0.0
                count = 0
        if count > 0:
            print(f'[{epoch + 1}, {i + 1:5d}] mse loss: {running_loss / count:.3f}')

        es_steps = 29
        if early_stopping:
            predictions = test_set_preds(model_name, model, tst_tracks, batch_size, img_size)
            new_mse = calculate_metrics(tst_tracks, predictions)['mse']
            if len(scores) > es_steps - 1 and new_mse > max(scores):
                print(f'stopping at epoch {i}, since mse {round(new_mse, 5)} is greater than previous mse: {[round(s, 5) for s in scores]}')

                index_of_lowest = scores.index(min(scores))
                model = torch.load(model_paths[index_of_lowest])

                for p in model_paths:
                    os.remove(p)

                return model
            
            if len(scores) == es_steps:
                to_destroy = model_paths.pop(0)
                os.remove(to_destroy)
                scores.pop(0)

            model_path =  'data/' + str(uuid.uuid4()) + '.pth'
            torch.save(model, model_path)
            model_paths.append(model_path)
            scores.append(new_mse)

    for p in model_paths:
        os.remove(p)
    
    return model


def calculate_metrics(tracks, predictions):
    gt = [track['popularity'] / 100 for track in tracks]
    
    return {
        'mae': mean_absolute_error(gt, predictions),
        'mse': mean_squared_error(gt, predictions),
        'rmse': np.sqrt(mean_squared_error(gt, predictions))
    }

def test_set_preds(model_name, m, tst, batch_size, img_size):
    with torch.no_grad():
        predictions = []

        dataloader = build_dataloader(model_name, m, tst, batch_size, input_independent=False, img_size=img_size, shuffle=False)

        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = m(inputs)
            for prediction in outputs:
                predictions.append(prediction.item())
    
        return predictions
