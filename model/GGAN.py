import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool as gap, global_max_pool as gmp

from gan import gan_smile

drug_gan,protein_gan=gan_smile()
class MolGraphBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(MolGraphBlock, self).__init__()
        self.gatv2_layer1 = GATv2Conv(input_dim, output_dim, heads=10, dropout=dropout_rate)
        self.gatv2_layer2 = GATv2Conv(output_dim * 10, output_dim * 10, dropout=dropout_rate)
        self.gan_smiles = drug_gan.discriminator['gcn_layers'][-1]  # 256x512
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gatv2_layer1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gatv2_layer2(x, edge_index)
        x = self.gan_smiles(x, edge_index)
        x = self.activation(x)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        return x

class ProSequenceBlock(nn.Module):
    def __init__(self, xt_features, embed_dim, n_filters, output_dim):
        super(ProSequenceBlock, self).__init__()
        self.embedding = nn.Embedding(xt_features + 1, embed_dim)
        self.conv_protein = nn.Convd(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.bilstm = nn.LSTM(embed_dim, 64, 1, dropout=0.2, bidirectional=True)
        self.gan_protein = protein_gan.discriminator[-12]  # 32 x64
        self.fc_protein = nn.Linear(64 * 60, output_dim)

    def forward(self, target):
        embedded_xt = self.embedding(target)
        embedded_xt, _ = self.bilstm(embedded_xt)
        conv_xt = self.conv_protein(embedded_xt)
        xt_gan = self.gan_protein(conv_xt)  # 512x64x60
        xt = xt_gan.view(xt_gan.size(0), -1)
        return self.fc_protein(xt)

class Combined(nn.Module):
    def __init__(self, input_dim, output_size, dropout_rate):
        super(Combined, self).__init__()
        self.fc_com1 = nn.Linear(input_dim, 1024)
        self.fc_com2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc_com1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_com2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output_layer(x)

class GGANModel(nn.Module):
    def __init__(self, output_size=1, xd_features=78, xt_features=25,
                 filter_count=32, embedding_dim=128, dense_output_dim=128, dropout_rate=0.2):
        super(GGANModel, self).__init__()
        self.drug_block = MolGraphBlock(xd_features, xd_features, dropout_rate)
        self.protein_block = ProSequenceBlock(xt_features, embedding_dim, filter_count, dense_output_dim)
        self.combined_block = Combined(2176, output_size, dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        drug_features = self.drug_block(x, edge_index, batch)
        protein_features = self.protein_block(target)

        combined_features = torch.cat((drug_features, protein_features), 1)
        return self.combined_block(combined_features)