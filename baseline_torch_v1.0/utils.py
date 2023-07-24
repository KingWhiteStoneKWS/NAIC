import torch.utils.data as Data
import torch

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def get_dataset_iter(features, label, device, batch_size=128, shuffle=True, validation_split=0.1):
    size = len(features)
    val_len = int(size*validation_split)
    lengths = [size-val_len, val_len]
    features_tensor = torch.tensor(features, device=device)
    label_tensor = torch.tensor(label, device=device)
    print(features_tensor.shape, label_tensor.shape)
    dataset = Data.TensorDataset(features_tensor, label_tensor)
    train_dataset, val_dataset = Data.random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
    train_dataset_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataset_iter = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataset_iter, val_dataset_iter

def get_tensor(features, device):
    features_tensor = torch.tensor(features, device=device)
    return features_tensor 


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.to(torch.float64), y.to(torch.float64))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def verify(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.to(torch.float64), y.to(torch.float64)).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

def inference(tensor, model):
    model.eval()
    with torch.no_grad():
        pred = model(tensor)
    return pred.cpu().numpy()


