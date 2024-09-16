# model.py
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn

# Definición del modelo HeteroGNN
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        # Configuración de capas convolucionales para grafo heterogéneo
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                # Paso de mensajes de 'subject' a 'region' (has_region)
                ('subject', 'has_region', 'region'): gnn.SAGEConv((-1, -1), hidden_channels),
                # Paso de mensajes de 'region' a 'subject' (rev_has_region)
                ('region', 'rev_has_region', 'subject'): gnn.SAGEConv((-1, -1), hidden_channels),
                # Paso de mensajes entre 'region' (is_functionally_connected)
                ('region', 'is_functionally_connected', 'region'): gnn.GCNConv(-1, hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)
        
        # Capas lineales finales para producir las salidas
        self.lin_subject = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_region = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Aplicar las capas convolucionales y procesar los datos
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # Retornar las salidas tanto para 'subject' como para 'region'
        return {
            'subject': self.lin_subject(x_dict['subject']),
            'region': self.lin_region(x_dict['region'])
        }
