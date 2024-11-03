import time
import pandas as pd
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
import os

warnings.filterwarnings('ignore')


class NetworkTrafficDataset:
    def __init__(self, train_path, test_path, sample_size=10000):
        self.train_path = train_path
        self.test_path = test_path
        self.sample_size = sample_size

    def preprocess_data(self):
        # Load datasets
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Sample data for faster processing
        if self.sample_size:
            train_df = train_df.sample(n=min(self.sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(self.sample_size // 2, len(test_df)), random_state=42)

        # Select only important features
        important_features = ['proto', 'service', 'state', 'sbytes', 'dbytes',
                              'sttl', 'dttl', 'rate', 'sload', 'dload']

        # Handle categorical variables
        categorical_cols = ['proto', 'service', 'state']
        self.label_encoders = {}

        # Combine datasets temporarily for encoding
        all_data = pd.concat([train_df, test_df], axis=0)

        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            all_data[col] = all_data[col].fillna('-')
            self.label_encoders[col].fit(all_data[col])
            train_df[col] = self.label_encoders[col].transform(train_df[col].fillna('-'))
            test_df[col] = self.label_encoders[col].transform(test_df[col].fillna('-'))

        # Prepare features and labels
        X_train = train_df[important_features]
        y_train = train_df['label']
        X_test = test_df[important_features]
        y_test = test_df['label']

        # Scale numerical features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def create_graph(self, features, labels, k=3):
        print("Converting to tensors...")
        x = torch.FloatTensor(features)
        y = torch.LongTensor(labels)

        print("Creating edges...")
        edge_index = []
        protocol_col = 0  # Assuming protocol is the first column

        # Create protocol groups
        protocol_groups = {}
        for i in range(len(features)):
            proto = features[i, protocol_col]
            if proto not in protocol_groups:
                protocol_groups[proto] = []
            protocol_groups[proto].append(i)

        # Connect nodes within same protocol group
        for proto_group in protocol_groups.values():
            if len(proto_group) > 1:
                for i in range(len(proto_group)):
                    # Connect to next k nodes in same protocol group
                    for j in range(i + 1, min(i + k + 1, len(proto_group))):
                        edge_index.append([proto_group[i], proto_group[j]])
                        edge_index.append([proto_group[j], proto_group[i]])

        edge_index = torch.LongTensor(edge_index).t()
        print(f"Graph created with {len(edge_index[0])} edges")

        return Data(x=x, edge_index=edge_index, y=y)


class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MalwareDetector:
    def create_visualizations(self, train_losses, train_accs, test_accs):
        # Plot training metrics
        plt.figure(figsize=(12, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(test_accs, label='Test')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

    def __init__(self, train_path, test_path, sample_size=10000):
        self.dataset = NetworkTrafficDataset(train_path, test_path, sample_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def evaluate(self, data):
        """Evaluate model performance on given data"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.to(self.device))
            pred = out.argmax(dim=1)
            correct = (pred == data.y.to(self.device)).sum()
            return correct.item() / len(data.y)

    def visualize_graph(self, data, num_nodes=100):
        """Visualize network structure"""
        print("Creating graph visualization...")

        # Create subset of data
        x = data.x[:num_nodes]
        y = data.y[:num_nodes]
        mask = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
        edge_index = data.edge_index[:, mask]

        # Convert to networkx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Create visualization
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=1 / np.sqrt(num_nodes))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_color=['red' if label == 1 else 'blue' for label in y],
                               node_size=100,
                               alpha=0.7)

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)

        plt.title("Network Traffic Graph Structure\nRed: Malicious, Blue: Normal")
        plt.axis('off')
        plt.savefig('graph_structure.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_embeddings(self):
        """Visualize node embeddings using t-SNE"""
        print("Creating embedding visualization...")
        self.model.eval()

        with torch.no_grad():
            # Get embeddings from first GCN layer
            embeddings = self.model.conv1(self.train_data.x.to(self.device),
                                          self.train_data.edge_index.to(self.device))
            embeddings = embeddings.cpu().numpy()

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Create visualization
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                  c=self.train_data.y.numpy(),
                                  cmap='coolwarm',
                                  alpha=0.6)

            plt.colorbar(scatter, label='Class (0: Normal, 1: Attack)')
            plt.title("Node Embeddings Visualization (t-SNE)")
            plt.xlabel("t-SNE dimension 1")
            plt.ylabel("t-SNE dimension 2")
            plt.savefig('embeddings.png', dpi=300, bbox_inches='tight')
            plt.close()

    def compare_with_baseline(self, X_train, X_test, y_train, y_test):
        """Compare with traditional ML baseline"""
        print("\nComparing with baseline model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_accuracy = rf_model.score(X_test, y_test)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        return rf_accuracy

    def monitor_performance(self):
        """Monitor resource usage"""
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return memory_usage

    def create_synthetic_attacks(self, num_samples=100):
        """Create synthetic attack patterns"""
        # Create anomalous patterns by modifying existing attack samples
        attack_samples = self.train_data.x[self.train_data.y == 1]
        if len(attack_samples) > 0:
            synthetic_samples = attack_samples[:num_samples].clone()
            # Modify features to create new patterns
            noise = torch.randn_like(synthetic_samples) * 0.1
            synthetic_samples += noise
            return synthetic_samples
        return None

    def train(self, epochs=20):
        # Track metrics and performance
        train_losses = []
        train_accs = []
        test_accs = []
        start_time = time.time()
        initial_memory = self.monitor_performance()

        # Preprocess data
        X_train, X_test, y_train, y_test = self.dataset.preprocess_data()
        print("Data preprocessed successfully")

        # Compare with baseline
        baseline_acc = self.compare_with_baseline(X_train, X_test, y_train, y_test)

        # Create graphs
        print("Creating graphs...")
        self.train_data = self.dataset.create_graph(X_train, y_train)
        self.test_data = self.dataset.create_graph(X_test, y_test)

        # Initialize model
        self.model = SimpleGNN(X_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Training loop
        print("\nStarting training...")
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.train_data.to(self.device))
            loss = F.nll_loss(out, self.train_data.y.to(self.device))
            loss.backward()
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            train_acc = self.evaluate(self.train_data)
            test_acc = self.evaluate(self.test_data)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        # Final performance metrics
        training_time = time.time() - start_time
        final_memory = self.monitor_performance()
        memory_usage = final_memory - initial_memory

        print("\nPerformance Metrics:")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Memory Usage: {memory_usage:.2f} MB")
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"GNN Final Accuracy: {test_acc:.4f}")

        # Create visualizations
        self.visualize_graph(self.train_data)
        self.visualize_embeddings()
        self.create_visualizations(train_losses, train_accs, test_accs)

        # Test on synthetic attacks
        synthetic_samples = self.create_synthetic_attacks()
        if synthetic_samples is not None:
            self.test_synthetic_attacks(synthetic_samples)

        # Save model
        self.save_model("Third_run/malware_detector.pt")

        # Final evaluation
        print("\nFinal Evaluation:")
        self.evaluate_detailed()

    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoders': self.dataset.label_encoders,
            'scaler': self.dataset.scaler
        }, path)
        print(f"\nModel saved to {path}")

    def evaluate_detailed(self):
        """Print detailed evaluation metrics"""
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            out = self.model(self.test_data.to(self.device))
            pred = out.argmax(dim=1).cpu()
            y_true = self.test_data.y.cpu()

            print("\nDetailed Classification Report:")
            print(classification_report(y_true, pred, target_names=['Normal', 'Attack']))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, pred)
            print(cm)

    def test_synthetic_attacks(self, synthetic_samples):
        """Test model on synthetic attack patterns"""
        self.model.eval()
        with torch.no_grad():
            # Create a smaller graph for synthetic samples
            num_synthetic = len(synthetic_samples)

            # Create edges for synthetic samples
            edge_index = []
            for i in range(num_synthetic):
                # Connect each node to its neighbors
                for j in range(max(0, i - 2), min(num_synthetic, i + 3)):
                    if i != j:
                        edge_index.append([i, j])
                        edge_index.append([j, i])

            edge_index = torch.LongTensor(edge_index).t()

            # Create synthetic data object
            synthetic_data = Data(
                x=synthetic_samples.to(self.device),
                edge_index=edge_index.to(self.device)
            )

            # Get predictions
            out = self.model(synthetic_data)
            predictions = out.argmax(dim=1)

            # Calculate detection rate
            detection_rate = (predictions == 1).float().mean()
            print(f"\nZero-day Detection Rate: {detection_rate:.4f}")


def main():
    torch.manual_seed(42)
    print("Starting Zero-day Malware Detection POC...")

    detector = MalwareDetector(
        train_path='UNSW_NB15_training-set.csv',
        test_path='UNSW_NB15_testing-set.csv',
        sample_size=10000
    )

    detector.train(epochs=20)

    print("\nVisualization files saved:")
    print("- graph_structure.png")
    print("- embeddings.png")
    print("- training_metrics.png")


if __name__ == "__main__":
    main()
