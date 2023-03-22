import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import csv
import os

# CSV reader
# Directory where the CSV files are located
directory = "desktop/ML/GRNNtest"

# Loop through files
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        
        with open(directory + filename, 'r') as csv_file:
            graph_df = pd.read_csv(csv_file)

#  PyTorch DataLoader (CSV)
class GraphDataset(Dataset):
    def __init__(self, graph_df):
        self.graph_df = graph_df

    def __len__(self):
        return len(self.graph_df)

    def __getitem__(self, idx):
        src_node = self.graph_df.iloc[idx]['source_node']
        tgt_node = self.graph_df.iloc[idx]['target_node']
        return src_node, tgt_node

graph_dataset = GraphDataset(graph_df)
graph_dataloader = DataLoader(graph_dataset, batch_size=64, shuffle=True)


# parameters
input_dim = 2
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
num_epochs = 100

# GRNN model
class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, x):
        a = torch.cat((h, x), dim=1)
        h = nn.ReLU(self.W(a))
        y = nn.ReLU(self.U(h))
        return y

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), learning_rate)


# Train
model = GRNN(input_dim, hidden_dim, output_dim)
for epoch in range(num_epochs):
    for i, batch in enumerate(graph_dataloader):
        src_nodes, tgt_nodes = batch
        optimizer.zero_grad()
        output = model(src_nodes, tgt_nodes)
        loss = criterion(output)
        loss.backward()
        optimizer.step()


# Test ######################################################################
test_dataset = GraphDataset(graph_df)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_dataloader:
        src_nodes, tgt_nodes = batch
        output = model(src_nodes, tgt_nodes)
        test_loss += criterion(output).item()

test_loss /= len(test_dataloader.dataset)
print(f'Test MSE: {test_loss:.3f}')
