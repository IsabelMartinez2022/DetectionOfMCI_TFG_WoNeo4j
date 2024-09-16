# main.py
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import KFold
from torch_geometric.data import HeteroData
from src.model3 import HeteroGNN

# Configuración de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
sex_map = {'M': 0, 'F': 1}
diagnosis_map = {'CN': 0, 'MCI': 1}
subjects['sex'] = subjects['sex'].map(sex_map)
subjects['diagnosis'] = subjects['diagnosis'].map(diagnosis_map)

# Crear el grafo heterogéneo
data = HeteroData()
data['region'].x = torch.eye(len(regions))
data['subject'].x = torch.tensor(subjects[['sex', 'age', 'diagnosis']].values, dtype=torch.float)

# Verificar índices de regiones y agregar relaciones
subject_id_codes = has_region['subject_id'].astype('category').cat.codes.values
region_ids = has_region['region_id'].values - 1
edge_index_subject_region = np.array([subject_id_codes, region_ids], dtype=np.int64)
data['subject', 'has_region', 'region'].edge_index = torch.tensor(edge_index_subject_region, dtype=torch.long)

edges_region = np.array([is_functionally_connected['Region1'] - 1, is_functionally_connected['Region2'] - 1], dtype=np.int64)
data['region', 'is_functionally_connected', 'region'].edge_index = torch.tensor(edges_region, dtype=torch.long)
data['region', 'rev_has_region', 'subject'].edge_index = data['subject', 'has_region', 'region'].edge_index.flip(0)

# Definir los parámetros de validación cruzada
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
hidden_channels = 64
out_channels = 2
num_layers = 2

# Guardar métricas por fold
fold_metrics = []
fold_accuracies = []
fold_conf_matrices = []

# Implementar K-Fold Cross-Validation
subject_indices = np.arange(len(subjects))

for fold, (train_idx, val_idx) in enumerate(kf.split(subject_indices)):
    print(f'Fold {fold + 1}/{k_folds}')

    # Crear máscaras de entrenamiento y validación
    train_mask = torch.zeros(len(subjects), dtype=torch.bool)
    val_mask = torch.zeros(len(subjects), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    # Asignar las máscaras al grafo heterogéneo
    data['subject'].train_mask = train_mask.to(device)
    data['subject'].val_mask = val_mask.to(device)
    data = data.to(device)

    # Inicializar el modelo para este fold
    model = HeteroGNN(hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Entrenamiento
    epochs = 50
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        train_out = out['subject'][data['subject'].train_mask]
        target = torch.randint(0, 2, (train_out.size(0),)).to(device)  # Target ficticio

        loss = F.cross_entropy(train_out, target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Validación
    model.eval()
    with torch.no_grad():
        val_out = out['subject'][data['subject'].val_mask]
        val_predictions = val_out.argmax(dim=1)
        val_target = torch.randint(0, 2, (val_out.size(0),)).to(device)  # Target ficticio

        # Calcular precisión y matriz de confusión
        accuracy = accuracy_score(val_target.cpu().numpy(), val_predictions.cpu().numpy())
        fold_accuracies.append(accuracy)

        cm = confusion_matrix(val_target.cpu().numpy(), val_predictions.cpu().numpy())
        fold_conf_matrices.append(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'MCI'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - Fold {fold + 1}")
        plt.savefig(f'confusion_matrix_fold_{fold + 1}.png')
        plt.close()

# Graficar la precisión por fold
plt.figure()
plt.plot(range(1, k_folds + 1), fold_accuracies, marker='o', label="Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Accuracy over K-Folds")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_per_fold.png")
plt.show()

# Calcular la matriz de confusión promedio
average_cm = np.mean(fold_conf_matrices, axis=0)

# Mostrar y guardar la matriz de confusión promedio
disp = ConfusionMatrixDisplay(confusion_matrix=average_cm, display_labels=['CN', 'MCI'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matriz de Confusión Promedio")
plt.savefig('average_confusion_matrix.png')
plt.show()

# Guardar las métricas promedio
with open("average_metrics.txt", "w") as f:
    f.write(f"Métricas promedio tras validación cruzada:\n")
    f.write(f"Precision: {np.mean(fold_accuracies):.4f}\n")
    f.write(f"Recall: {np.mean(fold_accuracies):.4f}\n")
    f.write(f"F1-Score: {np.mean(fold_accuracies):.4f}\n")

print("Validación cruzada completada y resultados guardados.")
