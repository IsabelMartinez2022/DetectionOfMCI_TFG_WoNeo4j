import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from torch_geometric.data import HeteroData
from model import HeteroGNN

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuración de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurar logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Rutas a los archivos CSV en la carpeta 'data'
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
subjects_path = os.path.join(data_dir, 'subjects.csv')
has_region_path = os.path.join(data_dir, 'has_region.csv')
is_functionally_connected_path = os.path.join(data_dir, 'is_functionally_connected.csv')
regions_path = os.path.join(data_dir, 'regions.csv')

# Cargar CSVs
subjects = pd.read_csv(subjects_path)
has_region = pd.read_csv(has_region_path)
is_functionally_connected = pd.read_csv(is_functionally_connected_path)
regions = pd.read_csv(regions_path)

# Preprocesar los datos de los sujetos
sex_map = {'M': 0, 'F': 1}
diagnosis_map = {'CN': 0, 'MCI': 1}
subjects['sex'] = subjects['sex'].map(sex_map)
subjects['diagnosis'] = subjects['diagnosis'].map(diagnosis_map)

# Crear el grafo heterogéneo
data = HeteroData()

# Cambiar la inicialización de las características de 'region' para que coincidan con hidden_channels
hidden_channels = 64
data['region'].x = torch.randn(len(regions), hidden_channels)  # Inicialización aleatoria con el tamaño correcto

# Proyectar las características de los nodos 'subject'
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
out_channels = 2
num_layers = 2

# Guardar las métricas por fold
fold_accuracies = []
fold_f1_scores = []
fold_precisions = []
fold_recalls = []
fold_conf_matrices = []
fold_losses = []
lr_values = []

# Implementar K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(subjects)):
    print(f'Fold {fold + 1}/{k_folds}')
    logging.info(f'Starting Fold {fold + 1}/{k_folds}')

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
    
    # Optimizer con weight_decay para regularización
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    # Scheduler para reducir el learning rate dinámicamente
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        train_out = out['subject'][data['subject'].train_mask]
        target = torch.randint(0, 2, (train_out.size(0),)).to(device)

        loss = F.cross_entropy(train_out, target)
        loss.backward()
        optimizer.step()
        
        # Guardar el learning rate actual
        lr_values.append(optimizer.param_groups[0]['lr'])

        epoch_train_losses.append(loss.item())

        # Validación
        model.eval()
        with torch.no_grad():
            val_out = out['subject'][data['subject'].val_mask]
            val_predictions = val_out.argmax(dim=1)
            val_target = torch.randint(0, 2, (val_out.size(0),)).to(device)

            val_loss = F.cross_entropy(val_out, val_target)
            epoch_val_losses.append(val_loss.item())

            scheduler.step(val_loss)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Guardar el mejor modelo
                torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')
            else:
                patience_counter += 1

            # Aplicar early stopping si no mejora la pérdida de validación
            if patience_counter >= patience:
                print(f"Early stopping en la epoch {epoch+1}")
                break

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    # Cargar el mejor modelo guardado
    model.load_state_dict(torch.load(f'model_fold_{fold+1}.pth'))
    model.eval()

    # Calcular métricas
    accuracy = accuracy_score(val_target.cpu().numpy(), val_predictions.cpu().numpy())
    f1 = f1_score(val_target.cpu().numpy(), val_predictions.cpu().numpy())
    precision = precision_score(val_target.cpu().numpy(), val_predictions.cpu().numpy())
    recall = recall_score(val_target.cpu().numpy(), val_predictions.cpu().numpy())

    fold_accuracies.append(accuracy)
    fold_f1_scores.append(f1)
    fold_precisions.append(precision)
    fold_recalls.append(recall)

    cm = confusion_matrix(val_target.cpu().numpy(), val_predictions.cpu().numpy())
    fold_conf_matrices.append(cm)

    # Mostrar y guardar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'MCI'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión - Fold {fold + 1}")
    plt.savefig(f'confusion_matrix_fold_{fold + 1}.png')
    plt.close()

    # Guardar pérdidas y métricas de cada fold
    fold_losses.append((epoch_train_losses, epoch_val_losses))

# Graficar pérdidas de entrenamiento y validación
plt.figure()
for i, (train_losses, val_losses) in enumerate(fold_losses):
    plt.plot(range(len(train_losses)), train_losses, label=f'Train Fold {i+1}')
    plt.plot(range(len(val_losses)), val_losses, label=f'Val Fold {i+1}', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss.png')
plt.show()

# Graficar métricas por fold
plt.figure()
plt.plot(range(1, k_folds + 1), fold_accuracies, marker='o', label="Accuracy per Fold")
plt.plot(range(1, k_folds + 1), fold_f1_scores, marker='x', label="F1-Score per Fold")
plt.plot(range(1, k_folds + 1), fold_precisions, marker='s', label="Precision per Fold")
plt.plot(range(1, k_folds + 1), fold_recalls, marker='d', label="Recall per Fold")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Metrics over K-Folds")
plt.legend()
plt.grid(True)
plt.savefig("metrics_per_fold.png")
plt.show()

# Graficar el learning rate durante el entrenamiento
plt.figure()
plt.plot(range(len(lr_values)), lr_values)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate over Training Steps")
plt.grid(True)
plt.savefig("learning_rate_over_steps.png")
plt.show()

# Calcular y mostrar la matriz de confusión promedio
average_cm = np.mean(fold_conf_matrices, axis=0)

# Mostrar y guardar la matriz de confusión promedio
disp = ConfusionMatrixDisplay(confusion_matrix=average_cm, display_labels=['CN', 'MCI'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matriz de Confusión Promedio")
plt.savefig('average_confusion_matrix.png')
plt.show()

# Guardar las métricas promedio
with open("average_metrics.txt", "w") as f:
    avg_accuracy = np.mean(fold_accuracies)
    avg_f1 = np.mean(fold_f1_scores)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)

    f.write(f"Métricas promedio tras validación cruzada:\n")
    f.write(f"Precision: {avg_precision:.4f}\n")
    f.write(f"Recall: {avg_recall:.4f}\n")
    f.write(f"F1-Score: {avg_f1:.4f}\n")
    f.write(f"Accuracy: {avg_accuracy:.4f}\n")

print("Validación cruzada completada y resultados guardados.")

# Graficar la curva ROC para cada fold
plt.figure()
for i, (train_idx, val_idx) in enumerate(kf.split(subjects)):
    val_out = out['subject'][data['subject'].val_mask]
    val_predictions = val_out.argmax(dim=1).cpu().numpy()
    val_target = torch.randint(0, 2, (val_out.size(0),)).cpu().numpy()

    fpr, tpr, _ = roc_curve(val_target, val_predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Fold')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Graficar la curva Precision-Recall para cada fold
plt.figure()
for i, (train_idx, val_idx) in enumerate(kf.split(subjects)):
    val_out = out['subject'][data['subject'].val_mask]
    val_predictions = val_out.argmax(dim=1).cpu().numpy()
    val_target = torch.randint(0, 2, (val_out.size(0),)).cpu().numpy()

    precision, recall, _ = precision_recall_curve(val_target, val_predictions)
    plt.plot(recall, precision, lw=2, label=f'Precision-Recall fold {i+1}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Fold')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.show()

print("Visualización de curvas ROC y Precision-Recall completadas.")
