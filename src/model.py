import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 64), 
    nn.ReLU(),
    nn.Linear(64, 3)
)


def train_model(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


x_data = torch.tensor([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],

    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=torch.float32)

y_data = torch.tensor([
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
], dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data, y_data),
    batch_size=1,
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, optimizer, criterion, 20)

torch.save(model.state_dict(), "model.pt")
