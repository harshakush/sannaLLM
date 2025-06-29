import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class NeuralNetworkVisualizer:
    def __init__(self, model, word_to_idx, idx_to_word, output_dir="visualizations"):
        self.model = model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_all(self, sample_words=['the', 'quick', 'brown']):
        visualizations = [
            ("Model Architecture", self.visualize_architecture),
            ("Embedding Weights", lambda: self.visualize_embeddings(num_words=30)),
            ("Weight Distributions", self.visualize_weight_distributions),
            ("Activation Flow", lambda: self.visualize_activations(sample_words)),
            ("Word Importance", lambda: self.visualize_word_importance(sample_words)),
            ("Network Graph", self.visualize_network_graph),
            ("Training Dynamics", self.visualize_training_dynamics),
            ("Prediction Analysis", lambda: self.visualize_predictions(sample_words))
        ]
        for name, viz_func in visualizations:
            try:
                print(f"  {name}...")
                viz_func()
            except Exception as e:
                print(f"  Error in {name}: {e}")
        print("All visualizations completed!")

    def visualize_architecture(self):
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        embedding_dim = self.model.embedding.embedding_dim
        vocab_size = self.model.embedding.num_embeddings
        ngram = self.model.ngram
        hidden_dim = self.model.fc1.out_features
        output_dim = self.model.fc2.out_features
        total_params = sum(p.numel() for p in self.model.parameters())
        layers_info = [
            ("Input Layer", f"{ngram} word indices", ngram, 0.08, "Input tokens"),
            ("Embedding Layer", f"{vocab_size} → {embedding_dim}d", embedding_dim, 0.22, f"Word embeddings\n{vocab_size:,} × {embedding_dim} = {vocab_size*embedding_dim:,} params"),
            ("Concatenation", f"Flatten to {ngram}×{embedding_dim}d", ngram * embedding_dim, 0.36, "Concatenate embeddings"),
            ("Hidden Layer", f"FC + ReLU → {hidden_dim}d", hidden_dim, 0.5, f"Dense layer\n{ngram*embedding_dim:,} × {hidden_dim:,} = {ngram*embedding_dim*hidden_dim:,} params"),
            ("Output Layer", f"FC → {output_dim}d", output_dim, 0.64, f"Classification\n{hidden_dim:,} × {output_dim:,} = {hidden_dim*output_dim:,} params"),
            ("Softmax", f"Probability distribution", output_dim, 0.78, "Next word probabilities")
        ]
        colors = ['#E3F2FD', '#E8F5E8', '#FFF3E0', '#FCE4EC', '#F3E5F5', '#E0F2F1']
        for i, (name, desc, size, x_pos, detail) in enumerate(layers_info):
            box_height = min(max(0.12, np.log10(size + 1) * 0.05), 0.25)
            box_width = 0.1
            rect = FancyBboxPatch(
                (x_pos - box_width/2, 0.5 - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.015",
                facecolor=colors[i],
                edgecolor='#333333',
                linewidth=2.5
            )
            ax.add_patch(rect)
            ax.text(x_pos, 0.5 + 0.02, name, ha='center', va='center', fontweight='bold', fontsize=11, color='#333333')
            ax.text(x_pos, 0.5 - 0.02, desc, ha='center', va='center', fontsize=9, color='#555555')
            ax.text(x_pos, 0.5 - box_height/2 - 0.08, detail, ha='center', va='center', fontsize=7, style='italic', color='#666666')
            if i < len(layers_info) - 1:
                next_x = layers_info[i+1][3]
                arrow_y = 0.5
                ax.annotate('', xy=(next_x - box_width/2 - 0.01, arrow_y), 
                           xytext=(x_pos + box_width/2 + 0.01, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
        title = f'N-gram Language Model Architecture\n'
        title += f'Vocabulary: {vocab_size:,} | N-gram: {ngram} | Hidden Units: {hidden_dim:,} | Total Parameters: {total_params:,}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=30, color='#333333')
        param_text = f"Parameter Breakdown:\n"
        param_text += f"• Embeddings: {vocab_size*embedding_dim:,} ({(vocab_size*embedding_dim/total_params)*100:.1f}%)\n"
        param_text += f"• Hidden Layer: {ngram*embedding_dim*hidden_dim + hidden_dim:,} ({((ngram*embedding_dim*hidden_dim + hidden_dim)/total_params)*100:.1f}%)\n"
        param_text += f"• Output Layer: {hidden_dim*output_dim + output_dim:,} ({((hidden_dim*output_dim + output_dim)/total_params)*100:.1f}%)"
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0.15, 0.85)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_architecture.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_embeddings(self, num_words=50):
        embedding_weights = self.model.embedding.weight.data.cpu().numpy()
        vocab_size = min(num_words, len(self.idx_to_word))
        selected_weights = embedding_weights[:vocab_size, :]
        selected_words = [self.idx_to_word[i] for i in range(vocab_size)]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        im1 = ax1.imshow(selected_weights, cmap='RdBu_r', aspect='auto')
        ax1.set_yticks(range(0, vocab_size, max(1, vocab_size//20)))
        ax1.set_yticklabels([selected_words[i] for i in range(0, vocab_size, max(1, vocab_size//20))], fontsize=8)
        ax1.set_xlabel('Embedding Dimensions')
        ax1.set_ylabel('Vocabulary Words')
        ax1.set_title('Embedding Weights Heatmap')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        norms = np.linalg.norm(selected_weights, axis=1)
        bars = ax2.bar(range(len(norms)), norms, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Word Index')
        ax2.set_ylabel('Embedding Norm')
        ax2.set_title('Embedding Vector Magnitudes')
        ax2.grid(True, alpha=0.3)
        top_indices = np.argsort(norms)[-5:]
        for idx in top_indices:
            bars[idx].set_color('orange')
            ax2.text(idx, norms[idx] + 0.1, selected_words[idx], rotation=45, ha='left', fontsize=8)
        dim_vars = np.var(selected_weights, axis=0)
        ax3.plot(dim_vars, 'g-', linewidth=2)
        ax3.fill_between(range(len(dim_vars)), dim_vars, alpha=0.3, color='green')
        ax3.set_xlabel('Embedding Dimension')
        ax3.set_ylabel('Variance Across Words')
        ax3.set_title('Embedding Dimension Importance')
        ax3.grid(True, alpha=0.3)
        subset_size = min(15, vocab_size)
        subset_weights = selected_weights[:subset_size]
        similarity_matrix = np.corrcoef(subset_weights)
        im4 = ax4.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(subset_size))
        ax4.set_yticks(range(subset_size))
        ax4.set_xticklabels(selected_words[:subset_size], rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(selected_words[:subset_size], fontsize=8)
        ax4.set_title('Word Similarity Matrix (Subset)')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/embedding_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_weight_distributions(self):
        from scipy import stats
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        layers = [
            ('Embedding Weights', self.model.embedding.weight.data.cpu().numpy().flatten()),
            ('FC1 Weights', self.model.fc1.weight.data.cpu().numpy().flatten()),
            ('FC1 Bias', self.model.fc1.bias.data.cpu().numpy().flatten()),
            ('FC2 Weights', self.model.fc2.weight.data.cpu().numpy().flatten()),
            ('FC2 Bias', self.model.fc2.bias.data.cpu().numpy().flatten())
        ]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink']
        for i, (layer_name, weights) in enumerate(layers):
            ax = axes[i]
            ax.hist(weights, bins=50, alpha=0.7, color=colors[i], density=True, edgecolor='black')
            kde = stats.gaussian_kde(weights)
            x_range = np.linspace(weights.min(), weights.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            mean_val = np.mean(weights)
            std_val = np.std(weights)
            median_val = np.median(weights)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_val:.4f}')
            ax.set_title(f'{layer_name}\nShape: {self._get_layer_shape(layer_name)}', fontweight='bold')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        ax = axes[-1]
        ax.axis('off')
        stats_text = "Weight Statistics Summary:\n\n"
        for layer_name, weights in layers:
            stats_text += f"{layer_name}:\n"
            stats_text += f"  Mean: {np.mean(weights):.4f}\n"
            stats_text += f"  Std:  {np.std(weights):.4f}\n"
            stats_text += f"  Min:  {np.min(weights):.4f}\n"
            stats_text += f"  Max:  {np.max(weights):.4f}\n\n"
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.suptitle('Weight Distributions Across All Layers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/weight_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _get_layer_shape(self, layer_name):
        if 'Embedding' in layer_name:
            return f"{self.model.embedding.num_embeddings} × {self.model.embedding.embedding_dim}"
        elif 'FC1 Weight' in layer_name:
            return f"{self.model.fc1.in_features} × {self.model.fc1.out_features}"
        elif 'FC1 Bias' in layer_name:
            return f"{self.model.fc1.out_features}"
        elif 'FC2 Weight' in layer_name:
            return f"{self.model.fc2.in_features} × {self.model.fc2.out_features}"
        elif 'FC2 Bias' in layer_name:
            return f"{self.model.fc2.out_features}"
        return "Unknown"

    def visualize_activations(self, sample_words):
        self.model.eval()
        input_tensor = torch.tensor([[self.word_to_idx.get(w, self.word_to_idx['<unk>']) for w in sample_words]], dtype=torch.long)
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        hooks = []
        hooks.append(self.model.embedding.register_forward_hook(get_activation('embedding')))
        hooks.append(self.model.fc1.register_forward_hook(get_activation('fc1')))
        hooks.append(self.model.relu.register_forward_hook(get_activation('relu')))
        hooks.append(self.model.fc2.register_forward_hook(get_activation('fc2')))
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        for hook in hooks:
            hook.remove()
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(sample_words)), [1]*len(sample_words), color=['lightblue', 'lightgreen', 'lightcoral'])
        ax1.set_xticks(range(len(sample_words)))
        ax1.set_xticklabels(sample_words, fontsize=12, fontweight='bold')
        ax1.set_title('Input Words', fontweight='bold')
        ax1.set_ylabel('Active')
        ax2 = fig.add_subplot(gs[0, 1:])
        embedding_data = activations['embedding'][0]
        im2 = ax2.imshow(embedding_data.T, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(sample_words)))
        ax2.set_xticklabels(sample_words)
        ax2.set_ylabel('Embedding Dimensions')
        ax2.set_title(f'Embedding Activations ({embedding_data.shape[1]} dimensions)')
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        ax3 = fig.add_subplot(gs[1, :2])
        fc1_data = activations['fc1'][0]
        ax3.bar(range(len(fc1_data)), fc1_data, alpha=0.7, color='lightcoral')
        ax3.set_title(f'FC1 Layer Activations ({len(fc1_data)} neurons)')
        ax3.set_xlabel('Neuron Index')
        ax3.set_ylabel('Activation Value')
        ax3.grid(True, alpha=0.3)
        ax4 = fig.add_subplot(gs[1, 2:])
        relu_data = activations['relu'][0]
        ax4.bar(range(len(relu_data)), relu_data, alpha=0.7, color='lightgreen')
        ax4.set_title(f'ReLU Activations ({len(relu_data)} neurons)')
        ax4.set_xlabel('Neuron Index')
        ax4.set_ylabel('Activation Value')
        ax4.grid(True, alpha=0.3)
        ax5 = fig.add_subplot(gs[2, :2])
        probs = probabilities[0].cpu().numpy()
        top_indices = np.argsort(probs)[-10:][::-1]
        top_probs = probs[top_indices]
        top_words = [self.idx_to_word[i] for i in top_indices]
        bars = ax5.bar(range(len(top_probs)), top_probs, color='gold', alpha=0.8)
        ax5.set_xticks(range(len(top_words)))
        ax5.set_xticklabels(top_words, rotation=45, ha='right')
        ax5.set_title('Top 10 Predicted Words')
        ax5.set_ylabel('Probability')
        for bar, prob in zip(bars, top_probs):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        stats_text = f"Activation Statistics for: '{' '.join(sample_words)}'\n\n"
        stats_text += f"Embedding Layer:\n"
        stats_text += f"  Shape: {embedding_data.shape}\n"
        stats_text += f"  Mean: {np.mean(embedding_data):.4f}\n"
        stats_text += f"  Std: {np.std(embedding_data):.4f}\n\n"
        stats_text += f"Hidden Layer (FC1):\n"
        stats_text += f"  Active neurons: {np.sum(fc1_data > 0)}/{len(fc1_data)}\n"
        stats_text += f"  Mean activation: {np.mean(fc1_data):.4f}\n"
        stats_text += f"  Max activation: {np.max(fc1_data):.4f}\n\n"
        stats_text += f"ReLU Layer:\n"
        stats_text += f"  Active neurons: {np.sum(relu_data > 0)}/{len(relu_data)}\n"
        stats_text += f"  Sparsity: {(1 - np.sum(relu_data > 0)/len(relu_data))*100:.1f}%\n\n"
        stats_text += f"Output Layer:\n"
        stats_text += f"  Top prediction: '{top_words[0]}' ({top_probs[0]:.3f})\n"
        stats_text += f"  Entropy: {-np.sum(probs * np.log(probs + 1e-10)):.3f}"
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.suptitle(f'Neural Network Activation Flow\nInput: "{" ".join(sample_words)}"', 
                    fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/activation_flow.png", dpi=300, bbox_inches='tight')
        plt.show()
        return activations

    def visualize_word_importance(self, sample_words):
        self.model.eval()
        input_tensor = torch.tensor([[self.word_to_idx.get(w, self.word_to_idx['<unk>']) for w in sample_words]], dtype=torch.long)
        with torch.no_grad():
            embeddings = self.model.embedding(input_tensor)
            embedding_norms = torch.norm(embeddings, dim=2)
            attention_weights = torch.softmax(embedding_norms, dim=1)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_words)))
        bars = ax1.bar(range(len(sample_words)), attention_weights[0].cpu().numpy(), color=colors)
        ax1.set_xticks(range(len(sample_words)))
        ax1.set_xticklabels(sample_words, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Importance Weight')
        ax1.set_title('Word Importance (Based on Embedding Magnitudes)')
        ax1.grid(True, alpha=0.3)
        for bar, weight in zip(bars, attention_weights[0].cpu().numpy()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        embedding_flat = embeddings[0].cpu().numpy()
        similarity_matrix = np.corrcoef(embedding_flat)
        im2 = ax2.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(sample_words)))
        ax2.set_yticks(range(len(sample_words)))
        ax2.set_xticklabels(sample_words, fontsize=12)
        ax2.set_yticklabels(sample_words, fontsize=12)
        ax2.set_title('Word Embedding Similarity Matrix')
        for i in range(len(sample_words)):
            for j in range(len(sample_words)):
                ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', fontweight='bold',
                        color='white' if abs(similarity_matrix[i, j]) > 0.5 else 'black')
        plt.colorbar(im2, ax=ax2)
        if embedding_flat.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedding_2d = pca.fit_transform(embedding_flat)
            ax3.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                       c=colors, s=200, alpha=0.8, edgecolors='black')
            for i, word in enumerate(sample_words):
                ax3.annotate(word, (embedding_2d[i, 0], embedding_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            ax3.set_title(f'Embedding Vectors (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        else:
            ax3.scatter(embedding_flat[:, 0], embedding_flat[:, 1], 
                       c=colors, s=200, alpha=0.8, edgecolors='black')
            for i, word in enumerate(sample_words):
                ax3.annotate(word, (embedding_flat[i, 0], embedding_flat[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            ax3.set_title('Embedding Vectors (2D)')
            ax3.set_xlabel('Dimension 1')
            ax3.set_ylabel('Dimension 2')
        ax3.grid(True, alpha=0.3)
        ax4.axis('off')
        importance_text = f"Word Importance Analysis\n"
        importance_text += f"Input: '{' '.join(sample_words)}'\n\n"
        for i, (word, weight) in enumerate(zip(sample_words, attention_weights[0].cpu().numpy())):
            importance_text += f"{i+1}. {word}: {weight:.4f} ({weight/attention_weights[0].sum()*100:.1f}%)\n"
        importance_text += f"\nInterpretation:\n"
        importance_text += f"• Higher weights indicate more 'important' words\n"
        importance_text += f"• Based on embedding vector magnitudes\n"
        importance_text += f"• Most important: '{sample_words[attention_weights[0].argmax()]}'\n"
        importance_text += f"• Least important: '{sample_words[attention_weights[0].argmin()]}'"
        ax4.text(0.05, 0.95, importance_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/word_importance.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_network_graph(self):
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Skipping network graph.")
            return
        G = nx.DiGraph()
        embedding_dim = self.model.embedding.embedding_dim
        hidden_dim = self.model.fc1.out_features
        output_dim = self.model.fc2.out_features
        layers = [
            ('Input', {'layer': 0, 'size': self.model.ngram, 'color': '#E3F2FD', 'type': 'input'}),
            ('Embedding', {'layer': 1, 'size': embedding_dim, 'color': '#E8F5E8', 'type': 'embedding'}),
            ('Flatten', {'layer': 2, 'size': self.model.ngram * embedding_dim, 'color': '#FFF3E0', 'type': 'reshape'}),
            ('FC1', {'layer': 3, 'size': hidden_dim, 'color': '#FCE4EC', 'type': 'dense'}),
            ('ReLU', {'layer': 4, 'size': hidden_dim, 'color': '#F3E5F5', 'type': 'activation'}),
            ('FC2', {'layer': 5, 'size': output_dim, 'color': '#E0F2F1', 'type': 'dense'}),
            ('Softmax', {'layer': 6, 'size': output_dim, 'color': '#FFF8E1', 'type': 'activation'})
        ]
        for name, attrs in layers:
            G.add_node(name, **attrs)
        edges = [
            ('Input', 'Embedding', {'weight': 1.0, 'type': 'lookup'}),
            ('Embedding', 'Flatten', {'weight': 1.0, 'type': 'reshape'}),
            ('Flatten', 'FC1', {'weight': 2.0, 'type': 'linear'}),
            ('FC1', 'ReLU', {'weight': 1.0, 'type': 'activation'}),
            ('ReLU', 'FC2', {'weight': 2.0, 'type': 'linear'}),
            ('FC2', 'Softmax', {'weight': 1.0, 'type': 'activation'})
        ]
        for src, dst, attrs in edges:
            G.add_edge(src, dst, **attrs)
        pos = {}
        layer_spacing = 2.0
        for node in G.nodes():
            layer = G.nodes[node]['layer']
            pos[node] = (layer * layer_spacing, 0)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        for node in G.nodes():
            x, y = pos[node]
            size = G.nodes[node]['size']
            color = G.nodes[node]['color']
            node_type = G.nodes[node]['type']
            display_size = min(max(np.sqrt(size) * 50, 500), 2000)
            circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontweight='bold', fontsize=10)
            ax.text(x, y-0.6, f'Size: {size}', ha='center', va='center', fontsize=8, style='italic')
            ax.text(x, y+0.6, node_type.title(), ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        for edge in G.edges():
            src, dst = edge
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            edge_type = G.edges[edge]['type']
            weight = G.edges[edge]['weight']
            ax.annotate('', xy=(x2-0.3, y2), xytext=(x1+0.3, y1),
                       arrowprops=dict(arrowstyle='->', lw=weight*2, color='darkblue'))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 + 0.2
            ax.text(mid_x, mid_y, edge_type, ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='lightblue', alpha=0.7))
        ax.set_xlim(-1, len(layers) * layer_spacing)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        total_params = sum(p.numel() for p in self.model.parameters())
        title = f'Neural Network Graph Representation\n'
        title += f'Total Parameters: {total_params:,} | Layers: {len(layers)} | Connections: {len(edges)}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/network_graph.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_training_dynamics(self, training_log_path=None):
        import os
        if training_log_path and os.path.exists(training_log_path):
            import pandas as pd
            df = pd.read_csv(training_log_path)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.plot(df['epoch'], df['perplexity'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Perplexity')
            ax2.set_title('Model Perplexity')
            ax2.grid(True, alpha=0.3)
            ax3.semilogy(df['epoch'], df['learning_rate'], 'purple', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate (log scale)')
            ax3.set_title('Learning Rate Schedule')
            ax3.grid(True, alpha=0.3)
            ax4.plot(df['epoch'], df['epoch_time'], 'orange', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Training Time per Epoch')
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/training_dynamics.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            self._create_placeholder_training_plot()

    def _create_placeholder_training_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Training Dynamics Visualization\n\nTo see actual training curves:\n1. Train your model\n2. Save training log as CSV\n3. Pass log path to visualize_training_dynamics()',
               ha='center', va='center', fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Training Dynamics (Placeholder)', fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_dynamics_placeholder.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_predictions(self, sample_words, top_k=15):
        self.model.eval()
        input_tensor = torch.tensor([[self.word_to_idx.get(w, self.word_to_idx['<unk>']) for w in sample_words]], dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]
        top_words = [self.idx_to_word[i] for i in top_indices]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_words)))
        bars = ax1.barh(range(len(top_words)), top_probs, color=colors)
        ax1.set_yticks(range(len(top_words)))
        ax1.set_yticklabels(top_words, fontsize=10)
        ax1.set_xlabel('Probability')
        ax1.set_title(f'Top {top_k} Predictions for "{" ".join(sample_words)}"')
        ax1.grid(True, alpha=0.3)
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{prob:.4f}', va='center', fontsize=8)
        sample_size = min(100, len(probabilities))
        sample_indices = np.random.choice(len(probabilities), sample_size, replace=False)
        sample_probs = probabilities[sample_indices]
        ax2.hist(sample_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(sample_probs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sample_probs):.6f}')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Probability Distribution (Sample of {sample_size} words)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        sorted_probs = np.sort(probabilities)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        ax3.plot(range(1, len(cumulative_probs)+1), cumulative_probs, 'b-', linewidth=2)
        ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% mass')
        ax3.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% mass')
        ax3.set_xlabel('Word Rank')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, min(1000, len(cumulative_probs)))
        ax4.axis('off')
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        top1_prob = top_probs[0]
        top5_mass = np.sum(top_probs[:5])
        top10_mass = np.sum(top_probs[:10])
        words_for_50_percent = np.sum(cumulative_probs <= 0.5) + 1
        words_for_90_percent = np.sum(cumulative_probs <= 0.9) + 1
        stats_text = f"Prediction Statistics\n"
        stats_text += f"Input: '{' '.join(sample_words)}'\n\n"
        stats_text += f"Top Prediction: '{top_words[0]}' ({top1_prob:.4f})\n\n"
        stats_text += f"Probability Mass:\n"
        stats_text += f"• Top 1: {top1_prob:.4f} ({top1_prob*100:.2f}%)\n"
        stats_text += f"• Top 5: {top5_mass:.4f} ({top5_mass*100:.2f}%)\n"
        stats_text += f"• Top 10: {top10_mass:.4f} ({top10_mass*100:.2f}%)\n\n"
        stats_text += f"Distribution Properties:\n"
        stats_text += f"• Entropy: {entropy:.3f}\n"
        stats_text += f"• Perplexity: {np.exp(entropy):.1f}\n"
        stats_text += f"• 50% mass in top {words_for_50_percent} words\n"
        stats_text += f"• 90% mass in top {words_for_90_percent} words\n\n"
        stats_text += f"Model Confidence: {'High' if top1_prob > 0.5 else 'Medium' if top1_prob > 0.1 else 'Low'}"
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/prediction_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
