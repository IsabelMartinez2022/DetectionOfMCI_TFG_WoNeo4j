import sys
import os
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from src.model3 import HeteroGNN, pretrain_model

# Verificación de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapeo de valores categóricos
sex_map = {'M': 0, 'F': 1}
diagnosis_map = {'CN': 0, 'MCI': 1}

# Rutas a los archivos CSV en la carpeta 'data'
data_dir = '../data'
has_region_path = os.path.join(data_dir, 'has_region.csv')
is_connected_to_CN_path = os.path.join(data_dir, 'is_connected_to_CN.csv')
is_connected_to_MCI_path = os.path.join(data_dir, 'is_connected_to_MCI.csv')
is_functionally_connected_path = os.path.join(data_dir, 'is_functionally_connected.csv')
regions_path = os.path.join(data_dir, 'regions.csv')
subjects_path = os.path.join(data_dir, 'subjects.csv')

# Cargar CSVs
has_region = pd.read_csv(has_region_path)
is_connected_to_CN = pd.read_csv(is_connected_to_CN_path)
is_connected_to_MCI = pd.read_csv(is_connected_to_MCI_path)
is_functionally_connected = pd.read_csv(is_functionally_connected_path)
regions = pd.read_csv(regions_path)
subjects = pd.read_csv(subjects_path)

# Preprocesar los datos de los sujetos
subjects['sex'] = subjects['sex'].map(sex_map)
subjects['diagnosis'] = subjects['diagnosis'].map(diagnosis_map)

# Crear el grafo heterogéneo
data = HeteroData()

# Agregar las características de nodos para 'region'
data['region'].x = torch.eye(len(regions))

# Agregar características de nodos para 'subject' (sexo, edad, diagnóstico)
data['subject'].x = torch.tensor(subjects[['sex', 'age', 'diagnosis']].values, dtype=torch.float)

# Convertir listas a arrays de NumPy antes de convertirlas a tensores
subject_id_codes = has_region['subject_id'].astype('category').cat.codes.values
region_ids = has_region['region_id'].values - 1

# Asegurarse de que los índices de los bordes se convierten en arrays de NumPy
edge_index_subject_region = np.array([subject_id_codes, region_ids], dtype=np.int64)
data['subject', 'has_region', 'region'].edge_index = torch.tensor(edge_index_subject_region, dtype=torch.long)

# Agregar las conexiones entre regiones (is_functionally_connected)
edges_region = np.array([is_functionally_connected['Region1'] - 1, is_functionally_connected['Region2'] - 1])
data['region', 'is_functionally_connected', 'region'].edge_index = torch.tensor(edges_region, dtype=torch.long)

# Añadir la relación inversa entre 'region' y 'subject' para que los nodos de 'subject' reciban mensajes
data['region', 'rev_has_region', 'subject'].edge_index = data['subject', 'has_region', 'region'].edge_index.flip(0)

# Definir el modelo, optimizador y parámetros de entrenamiento
hidden_channels = 64
out_channels = 2
num_layers = 2
model = HeteroGNN(hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Lista para guardar las pérdidas (losses)
losses = []
all_predictions = []
all_targets = []

# Entrenar el modelo y guardar los pesos
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    target = torch.randint(0, 2, (len(data['subject'].x),))  # Target ficticio para demostración
    loss = F.cross_entropy(out['subject'], target)
    loss.backward()
    optimizer.step()
    
    # Guardar la pérdida actual
    losses.append(loss.item())
    
    # Predecir etiquetas
    predictions = out['subject'].argmax(dim=1)
    all_predictions.extend(predictions.cpu().numpy())
    all_targets.extend(target.cpu().numpy())
    
    # Mostrar progreso
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Guardar los pesos del modelo
model_file = 'trained_heteroGNN_model.pth'
torch.save(model.state_dict(), model_file)
print(f"Modelo guardado en {model_file}")

# Guardar la gráfica de la pérdida
plt.figure()
plt.plot(range(1, epochs + 1), losses, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')  # Guardar la visualización en un archivo
print("Gráfica de pérdida guardada como 'training_loss.png'")

# Generar la matriz de confusión
cm = confusion_matrix(all_targets, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'MCI'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión")
plt.savefig('confusion_matrix.png')
print("Matriz de confusión guardada como 'confusion_matrix.png'")

# Reporte de clasificación (precisión, recall, F1-score)
classification_rep = classification_report(all_targets, all_predictions, target_names=['CN', 'MCI'])
print("Reporte de clasificación:\n", classification_rep)

# Guardar el reporte de clasificación en un archivo de texto
with open("classification_report.txt", "w") as f:
    f.write(classification_rep)
print("Reporte de clasificación guardado como 'classification_report.txt'")
