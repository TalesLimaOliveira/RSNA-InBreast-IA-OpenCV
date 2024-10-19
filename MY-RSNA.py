



transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Criação dos datasets para treinamento, validação e teste
train_dataset = CustomDataset(dataframe=train_df, transform=transform)
val_dataset = CustomDataset(dataframe=val_df, transform=transform)
test_dataset = CustomDataset(dataframe=test_df, transform=transform)

# Criação dos DataLoaders para treinamento, validação e teste
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verificação dos dados retornados pelo DataLoader
def verificar_dataloader(dataloader, loader_name):
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"{loader_name} - Batch {batch_idx}:")
        print(f"  Imagens: {images.shape}")
        print(f"  Labels: {labels.shape}")
        if batch_idx == 0:
            break  # Verifica apenas o primeiro batch

verificar_dataloader(train_loader, "Treinamento")
verificar_dataloader(val_loader, "Validação")
verificar_dataloader(test_loader, "Teste")

# Definir o modelo PyTorch e treinamento
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(416 * 416 * 3, 128)  # Camada totalmente conectada
        self.fc2 = nn.Linear(128, 1)  # Camada de saída

    def forward(self, x):
        x = x.view(-1, 416 * 416 * 3)  # Achata a imagem
        x = torch.relu(self.fc1(x))    # Primeira camada com ReLU
        x = torch.sigmoid(self.fc2(x)) # Camada de saída com sigmoid
        return x

# Verificar se a GPU está disponível e mover o modelo para a GPU
model = SimpleNN().to(device)

print("Parâmetros do modelo:")
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Monitoramento de uso da GPU
print(f"Memória total disponível na GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Memória usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memória livre: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print(train_df.head())
print(train_df['cancer'].value_counts())
print(train_df['image file path'].value_counts())

# Teste do modelo com os dados de teste
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        print(f"Test Loss: {loss.item():.4f}")
