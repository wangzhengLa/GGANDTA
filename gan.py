import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gan_datahelper import DataSet,smile_to_graph


def pad_or_truncate_features(features_list, max_atoms=32):
    """
    Processing molecular features into fixed-size 3D tensors
    Input: List of (num_atoms, 78) arrays
    Output: (num_molecules, 32, 78) array
    """
    padded_features = []

    for features in features_list:
        if features is None:
            # 处理空特征
            features = np.zeros((max_atoms, 78))
        else:
            features = np.array(features)
            if len(features.shape) != 2:
                print(f"警告: 特征形状异常: {features.shape}")
                continue

            num_atoms = features.shape[0]
            if num_atoms > max_atoms:
                # 截断过大的分子
                features = features[:max_atoms]
            else:
                # 填充小分子
                padding = np.zeros((max_atoms - num_atoms, features.shape[1]))
                features = np.vstack([features, padding])

        padded_features.append(features)

    return np.array(padded_features)  # Shape: (num_molecules, 32, 78)


def pad_or_truncate_sequence(sequences, max_seq_len=1000):
    """
    Pad or truncate protein sequences to a fixed length.
    Input: List of protein sequences
    Output: (num_proteins, max_seq_len) array
    """
    padded_sequences = []

    for seq in sequences:
        if len(seq) > max_seq_len:
            padded_seq = seq[:max_seq_len]
        else:
            padded_seq = np.pad(seq,
                                (0, max_seq_len - len(seq)),
                                mode='constant',
                                constant_values=0)
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)  # Shape: (num_proteins, max_seq_len)


def pretrain_protein_gan(protein_data, device, epochs=100, batch_size=64):
    """Pretrain Protein GAN with improved training process"""
    print("Preprocessing protein sequences...")
    padded_proteins = pad_or_truncate_sequence(protein_data, max_seq_len=1000)

    # Normalize input data to [-1, 1] range to match tanh output
    padded_proteins = (padded_proteins * 2) - 1

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(padded_proteins))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gan = ProteinGAN(seq_len=1000).to(device)

    # Reduced learning rate and modified beta values
    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    print(f"Starting GAN training with {len(padded_proteins)} sequences...")

    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0

        for real_seqs in data_loader:
            real_seqs = real_seqs[0].to(device)
            batch_size = real_seqs.size(0)

            # Train discriminator
            optimizer_d.zero_grad()

            # Add noise to labels for label smoothing
            label_real = torch.ones(batch_size, 1).to(device) * 0.9  # Smooth real labels to 0.9
            label_fake = torch.zeros(batch_size, 1).to(device) * 0.1  # Smooth fake labels to 0.1

            # Add noise to real samples
            real_noise = torch.randn_like(real_seqs) * 0.1
            real_seqs_noisy = real_seqs + real_noise

            # Generate fake samples
            z = torch.randn(batch_size, 100).to(device)
            fake_seqs = gan.generate(z)

            # Discriminator loss
            d_real = gan.discriminate(real_seqs_noisy)
            d_fake = gan.discriminate(fake_seqs.detach())

            d_loss_real = F.binary_cross_entropy(d_real, label_real)
            d_loss_fake = F.binary_cross_entropy(d_fake, label_fake)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()

            # Generate new fake samples
            z = torch.randn(batch_size, 100).to(device)
            fake_seqs = gan.generate(z)
            g_fake = gan.discriminate(fake_seqs)

            # Use real labels for generator loss
            g_loss = F.binary_cross_entropy(g_fake, label_real)

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}')

    return gan


def pretrain_drug_gan(drug_data, device, epochs=100, batch_size=512):
    """Pretrain Drug GAN with improved training process"""
    padded_features = pad_or_truncate_features(drug_data)

    gan = DrugGAN(num_atoms=32, atom_dim=78).to(device)

    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    discriminator_params = []
    for component in gan.discriminator.values():
        discriminator_params.extend(component.parameters())
    optimizer_d = torch.optim.Adam(discriminator_params, lr=0.0004, betas=(0.5, 0.999))

    edge_index = []
    for i in range(32):
        for j in range(32):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).t().contiguous().to(device)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(padded_features))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=epochs,
        eta_min=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d,
        T_max=epochs,
        eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    def batch_discriminate(x, edge_index, chunk_size=64):
        features = []
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i:i + chunk_size]
            chunk_features = []
            for curr_x in chunk:
                # GCN feature extraction
                for gcn_layer in gan.discriminator['gcn_layers']:
                    curr_x = F.relu(gcn_layer(curr_x, edge_index))
                    curr_x = F.dropout(curr_x, p=0.3, training=gan.training)
                # Global pooling
                curr_x = torch.mean(curr_x, dim=0)
                chunk_features.append(curr_x)
            features.extend(chunk_features)
        return gan.discriminator['fc_layers'](torch.stack(features))

    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0

        print(f'\nEpoch [{epoch + 1}/{epochs}]')

        for real_mols in data_loader:
            real_mols = real_mols[0].to(device, non_blocking=True)
            curr_batch_size = real_mols.size(0)

            optimizer_d.zero_grad(set_to_none=True)

            label_real = torch.ones(curr_batch_size, 1).to(device) * 0.9
            label_fake = torch.zeros(curr_batch_size, 1).to(device) * 0.1

            with torch.cuda.amp.autocast():
                z = torch.randn(curr_batch_size, 100).to(device)
                fake_mols = gan.generate(z)

                d_real_logits = batch_discriminate(real_mols, edge_index)
                d_fake_logits = batch_discriminate(fake_mols.detach(), edge_index)

                d_loss_real = criterion(d_real_logits, label_real)
                d_loss_fake = criterion(d_fake_logits, label_fake)
                d_loss = d_loss_real + d_loss_fake

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)

            if num_batches % 2 == 0:
                optimizer_g.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    g_fake_logits = batch_discriminate(fake_mols, edge_index)
                    g_loss = criterion(g_fake_logits, label_real)

                scaler.scale(g_loss).backward()
                scaler.step(optimizer_g)

                total_g_loss += g_loss.item()

                if num_batches % 10 == 0:
                    print(f'Batch [{num_batches}]: D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

            scaler.update()
            total_d_loss += d_loss.item()
            num_batches += 1

            scheduler_d.step()
            scheduler_g.step()

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / (num_batches // 2)

        print(f'Epoch [{epoch + 1}/{epochs}] completed - '
              f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, '
              f'LR_d: {scheduler_d.get_last_lr()[0]:.6f}, '
              f'LR_g: {scheduler_g.get_last_lr()[0]:.6f}')

        torch.cuda.empty_cache()

    return gan


class ProteinGAN(nn.Module):
    def __init__(self, seq_len=1000):
        super(ProteinGAN, self).__init__()


        self.seq_len = seq_len
        conv1_size = seq_len // 2  # stride=2
        conv2_size = conv1_size // 2  # stride=2
        conv3_size = conv2_size // 2  # stride=2
        self.final_size = conv3_size * 128  # 128

        # Generator - Made deeper but with smaller layers
        self.generator = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256 * 5),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256 * 5),
            nn.Dropout(0.3),
            # Reshape to (batch_size, 256, 5) happens in generate()
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=4, padding=0),
            nn.Tanh()
        )

        # Discriminator - Made weaker with fewer parameters
        self.discriminator = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Second conv block
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Third conv block
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(self.final_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )


    def generate(self, z):
        # Process through initial dense layers
        x = self.generator[0:6](z)  # Process until before reshape
        x = x.view(-1, 256, 5)  # Reshape to (batch_size, channels, length)
        # Process through conv layers
        for layer in self.generator[6:]:
            x = layer(x)
        return x

    def discriminate(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if x.size(2) != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        return self.discriminator(x)



class DrugGAN(nn.Module):
    def __init__(self, num_atoms=32, atom_dim=78):
        super(DrugGAN, self).__init__()
        self.num_atoms = num_atoms
        self.atom_dim = atom_dim

        self.generator = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, num_atoms * atom_dim),
            nn.Tanh()
        )


        self.discriminator = nn.ModuleDict({
            'gcn_layers': nn.ModuleList([
                GCNConv(atom_dim, 780),
                GCNConv(780, 1024),
            ]),

            'fc_layers': nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        })


    def generate(self, batch_size):
        z = torch.randn(batch_size.size(0), 100).to(next(self.parameters()).device)
        return self.generator(z).view(batch_size.size(0), self.num_atoms, self.atom_dim)

    def discriminate(self, x, edge_index):
        # GCN layers
        h = x
        for gcn_layer in self.discriminator['gcn_layers']:
            h = gcn_layer(h, edge_index)
            h = F.relu(h)

        # Global pooling
        h = torch.mean(h, dim=0)

        # FC layers
        return self.discriminator['fc_layers'](h)

def gan_smile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")


    # Load data
    dataset = DataSet(1000, 100)
    drugs_train, proteins_train = dataset.parse_data()

    print("\nConverting SMILES to graphs...")
    drug_train_graphs = []
    for i, smile in enumerate(drugs_train):
        if isinstance(smile, str):
            node_features, edge_index = smile_to_graph(smile)
            if node_features is not None:
                drug_train_graphs.append((node_features, edge_index))
            if (i + 1) % 1000 == 0:
                print(f"处理了 {i + 1}/{len(drugs_train)} 个分子")

    drug_features = [g[0] for g in drug_train_graphs]
    protein_gan = pretrain_protein_gan(proteins_train, device, 500)

    drug_gan = pretrain_drug_gan(drug_features, device, 500)
    return drug_gan,protein_gan
