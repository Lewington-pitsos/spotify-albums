import wandb
from dataload import build_dataloader
import torch.optim as optim
import torch.nn as nn
from constants import device

def train_basic_conv(model_name, model, tracks, batch_size=4, input_independent=False, epochs=2, img_size=32, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    dataloader = build_dataloader(model_name, model, tracks, batch_size, input_independent, img_size)

    model = model.to(device)

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
                for i in range(inputs.shape[0]):
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
