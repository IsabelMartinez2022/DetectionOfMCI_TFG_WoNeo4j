import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.norm import BatchNorm

# Definición del modelo HeteroGNN mejorado
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Capa lineal inicial para proyectar las características de los nodos 'subject' y 'region' a hidden_channels
        self.lin_subject = torch.nn.Linear(3, hidden_channels)  # Los sujetos tienen 3 características (sexo, edad, diagnóstico)
        self.lin_region = torch.nn.Linear(hidden_channels, hidden_channels)  # Proyección de las características de los nodos 'region'

        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                # Aseguramos que ambas convoluciones usen hidden_channels como tamaño de entrada y salida
                ('subject', 'has_region', 'region'): gnn.GraphConv(hidden_channels, hidden_channels),
                ('region', 'rev_has_region', 'subject'): gnn.GraphConv(hidden_channels, hidden_channels),
                ('region', 'is_functionally_connected', 'region'): gnn.GATConv(hidden_channels, hidden_channels, heads=1)
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))

        # Capas lineales finales para producir las salidas
        self.lin_output_subject = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_output_region = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        # Proyectar las características del nodo 'subject' y 'region' a hidden_channels
        x_dict['subject'] = self.lin_subject(x_dict['subject'])
        x_dict['region'] = self.lin_region(x_dict['region'])
        
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict['subject'] = norm(x_dict['subject'])  # Normalización de nodos 'subject'
            x_dict['region'] = norm(x_dict['region'])  # Normalización de nodos 'region'
            x_dict['subject'] = F.relu(x_dict['subject'])
            x_dict['region'] = F.relu(x_dict['region'])
            x_dict['subject'] = F.dropout(x_dict['subject'], p=self.dropout, training=self.training)
            x_dict['region'] = F.dropout(x_dict['region'], p=self.dropout, training=self.training)
            
        return {
            'subject': self.lin_output_subject(x_dict['subject']),
            'region': self.lin_output_region(x_dict['region'])
        }
