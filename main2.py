import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import classification_report, accuracy_score, f1_score

# --- Set seeds and device ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    # Enable benchmark for faster training when possible
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Label and path definitions ---
LABEL_MAPPING = {
    'adware': 0,
    'benign': 1,
    'downloader': 2,
    'trojan': 3,
    'addisplay': 4  # This class will be treated as unseen during training
}

# Set file paths (modify if needed)
graph_data_path = 'malnet_graphs_tiny/malnet-graphs-tiny'
split_info_path = 'split_info_tiny/split_info_tiny/type'


# --- Split load helper ---
def load_split(file_name):
    split_file = os.path.join(split_info_path, file_name)
    with open(split_file, 'r') as f:
        return [line.strip() for line in f]


train_graphs = load_split('train.txt')
val_graphs = load_split('val.txt')
test_graphs = load_split('test.txt')


# --- Node feature calculation ---
@torch.no_grad()
def calculate_node_features(G_nx, num_nodes):
    # Degree features
    degrees = torch.tensor([d for _, d in G_nx.degree()], dtype=torch.float32)
    # Clustering coefficient on undirected graph
    G_undirected = G_nx.to_undirected()
    clustering_dict = nx.clustering(G_undirected)
    clustering_values = torch.tensor([clustering_dict[i] for i in range(num_nodes)], dtype=torch.float32)
    # Betweenness centrality: approximate if graph is large
    if num_nodes > 500:
        betweenness_dict = nx.betweenness_centrality(G_nx, k=min(50, num_nodes))
    else:
        betweenness_dict = nx.betweenness_centrality(G_nx)
    betweenness_values = torch.tensor([betweenness_dict[i] for i in range(num_nodes)], dtype=torch.float32)
    # Eigenvector centrality with fallback
    try:
        eigenvector_dict = nx.eigenvector_centrality(G_nx, max_iter=100)
    except Exception as e:
        eigenvector_dict = {i: 1.0 / num_nodes for i in range(num_nodes)}
    eigenvector_values = torch.tensor([eigenvector_dict[i] for i in range(num_nodes)], dtype=torch.float32)
    return torch.stack([degrees, clustering_values, betweenness_values, eigenvector_values], dim=1)


# --- Graph loading function with improved error handling ---
def load_graphs(graph_list, desc="Loading graphs"):
    data_list = []
    with tqdm(total=len(graph_list), desc=desc, position=0, leave=True) as pbar:
        for graph_entry in graph_list:
            # Expecting "class/filename" structure
            category_folder, filename = os.path.split(graph_entry)
            class_name = category_folder.split('/')[0]
            if class_name not in LABEL_MAPPING:
                pbar.update(1)
                continue

            graph_path = os.path.join(graph_data_path, category_folder, f"{filename}.edgelist")
            if not os.path.exists(graph_path):
                pbar.update(1)
                continue

            try:
                # Efficient edge list reading
                with open(graph_path, 'r') as f:
                    edge_list = [tuple(map(int, line.strip().split()))
                                 for line in f if line.strip() and not line.startswith('#')]
                if not edge_list:
                    pbar.update(1)
                    continue

                edge_index = torch.tensor(edge_list).t().contiguous()
                num_nodes = int(edge_index.max().item()) + 1

                # Build networkx graph and calculate node features
                G_nx = nx.DiGraph()
                G_nx.add_nodes_from(range(num_nodes))
                G_nx.add_edges_from(edge_list)
                x = calculate_node_features(G_nx, num_nodes)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor([LABEL_MAPPING[class_name]], dtype=torch.long)
                )
                data_list.append(data)
            except Exception as e:
                print(f"[Warning] Error processing {graph_path}: {str(e)}")
            finally:
                pbar.update(1)
    return data_list


# --- Load datasets ---
print("Loading datasets...")
train_data_raw = load_graphs(train_graphs, desc="Training graphs")
val_data_raw = load_graphs(val_graphs, desc="Validation graphs")
test_data_raw = load_graphs(test_graphs, desc="Test graphs")

# Filter out the unseen class ('addisplay') from training data
unseen_label = LABEL_MAPPING['addisplay']
train_data = [d for d in train_data_raw if d.y.item() != unseen_label]
val_data = val_data_raw
test_data = test_data_raw


# --- Attribute matrix module ---
class AttributeMatrix(torch.nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # For now, semantics are randomly initialized. Replace with semantic embeddings if available.
        self.attributes = torch.nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, indices=None):
        # Normalize attribute vectors to unit length to stabilize cosine similarity comparisons.
        attr = F.normalize(self.attributes, p=2, dim=1)
        return attr if indices is None else attr[indices]


# --- Zero-Shot GNN Model ---
class ZSLGNN(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, embed_dim=64):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim * 2, embed_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        # Process batch if available; otherwise, create a dummy batch tensor.
        batch = data.batch.to(device) if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long,
                                                                                 device=device)
        x = self.dropout(F.elu(self.conv1(x, edge_index)))
        x = self.dropout(F.elu(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out


# --- Set up model, optimizer, scheduler, and attribute matrix ---
model = ZSLGNN().to(device)
attribute_matrix = AttributeMatrix(num_classes=len(LABEL_MAPPING), embed_dim=64).to(device)
optimizer = AdamW(list(model.parameters()) + list(attribute_matrix.parameters()), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.0001)

# --- Data loaders ---
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)


# --- Training function ---
def train(model, loader, optimizer, scheduler, attribute_matrix):
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        embeddings = model(data)

        # Ensure targets are correctly shaped and normalized
        y_labels = data.y.squeeze() if data.y.dim() > 1 else data.y
        targets = attribute_matrix(y_labels)

        # Normalize both model embeddings and targets for robust cosine similarity computation
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        targets_norm = targets  # Already normalized in AttributeMatrix

        # Loss: cosine distance plus L2 regularization penalty on embeddings
        loss = (1 - F.cosine_similarity(embeddings_norm, targets_norm, dim=1)).mean()
        loss += 0.01 * torch.norm(embeddings, p=2, dim=1).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)


# --- Evaluation function ---
@torch.no_grad()
def evaluate(model, loader, attribute_matrix):
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        embeddings = model(data)
        y_labels = data.y.squeeze() if data.y.dim() > 1 else data.y

        # Normalize embeddings and attribute matrix
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        attr_norm = attribute_matrix()  # Entire normalized attribute matrix

        # Compute cosine similarity between each sample embedding and each class attribute
        sims = torch.mm(embeddings_norm, attr_norm.t())
        preds = sims.argmax(dim=1)
        y_true.extend(y_labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(classification_report(y_true, y_pred))
    return accuracy, f1


# --- Main training loop ---
if __name__ == "__main__":
    print(f"Using device: {device}")

    EPOCHS = 15
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, scheduler, attribute_matrix)
        print(f"Epoch {epoch + 1}/{EPOCHS}: Loss = {train_loss:.4f}")

        # Evaluate on validation set every 3 epochs
        if (epoch + 1) % 3 == 0:
            val_acc, val_f1 = evaluate(model, val_loader, attribute_matrix)
            print(f"Validation Accuracy: {val_acc:.4f}, Macro F1: {val_f1:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'attribute_matrix_state_dict': attribute_matrix.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, 'best_model.pth')

    print("\nFinal Test Evaluation:")
    test_acc, test_f1 = evaluate(model, test_loader, attribute_matrix)
    print(f"Test Accuracy: {test_acc:.4f}, Macro F1: {test_f1:.4f}")
