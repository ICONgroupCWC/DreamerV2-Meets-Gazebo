import torch
from tqdm.auto import tqdm
import numpy as np


def cc_train(encoder, train_dataloader, loss_fn, optimizer, device):

    encoder.train()
    running_loss = 0.0
    for batch_idx, (channels_t, _, ind) in enumerate(train_dataloader):

        optimizer.zero_grad()

        embeddings = encoder(channels_t.cfloat().to(device))

        loss = loss_fn(channels_t, embeddings, ind)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return np.array(running_loss) / len(train_dataloader)
 
def power_train(net, data_loader, optimizer, device):

    net.train()
    loss_fn=torch.nn.MSELoss()
    running_loss = 0.0
    for (chart, power) in data_loader:

        optimizer.zero_grad()

        preds = net(chart.to(device))

        loss = loss_fn(power.unsqueeze(-1).to(device), preds)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return np.array(running_loss) / len(data_loader)



def jepa_train(
    encoder,
    predictor,
    target_encoder,
    train_data_loader,
    loss_fn,
    optimizer,
    momentum_step,
    device,
):
    losses = {"train": 0.0, "val": 0.0}

    encoder.train()
    predictor.train()
    print("Start_training")
    for csi, act in tqdm(train_data_loader, leave=False, desc="Training"):

        optimizer.zero_grad()

        csi = csi.to(device)
        act = act.to(device)

        # forward target
        with torch.no_grad():
            targets = target_encoder(
                torch.flatten(csi, start_dim=0, end_dim=1).cfloat()
            ).view(csi.shape[0], csi.shape[1], -1)
        # targets = encoder(torch.flatten(csi, start_dim=0, end_dim=1).cfloat()).view(csi.shape[0], csi.shape[1], -1)

        # forward context
        # print(csi[:, 0, ...].cfloat().shape)
        context = encoder(csi[:, 0, ...].cfloat())
        predictions, _ = predictor(act.float())
        predictions = predictions + context.unsqueeze(dim=1) #predictions.cumsum(1)
        # print(predictions.shape, targets.shape)
        loss = loss_fn(targets[:, 1:, :], predictions)
        loss.backward()
        optimizer.step()

        losses["train"] += loss.item()

        # momentum update target encoder
        with torch.no_grad():
            for param_q, param_k in zip(
                encoder.parameters(), target_encoder.parameters()
            ):
                param_k.data.mul_(momentum_step).add_(
                    (1.0 - momentum_step) * param_q.detach().data
                )
            target_encoder.encoder.bn1.running_mean.copy_(
                encoder.encoder.bn1.running_mean
            )
            target_encoder.encoder.bn1.running_var.copy_(
                encoder.encoder.bn1.running_var
            )
            target_encoder.encoder.bn2.running_mean.copy_(
                encoder.encoder.bn2.running_mean
            )
            target_encoder.encoder.bn2.running_var.copy_(
                encoder.encoder.bn2.running_var
            )
            target_encoder.encoder.bn3.running_mean.copy_(
                encoder.encoder.bn3.running_mean
            )
            target_encoder.encoder.bn3.running_var.copy_(
                encoder.encoder.bn3.running_var
            )
            target_encoder.encoder.bn4.running_mean.copy_(
                encoder.encoder.bn4.running_mean
            )
            target_encoder.encoder.bn4.running_var.copy_(
                encoder.encoder.bn4.running_var
            )
            target_encoder.encoder.bn5.running_mean.copy_(
                encoder.encoder.bn5.running_mean
            )
            target_encoder.encoder.bn5.running_var.copy_(
                encoder.encoder.bn5.running_var
            )

    return losses
