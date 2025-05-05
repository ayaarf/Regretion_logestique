import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# Exemple de dataset simple
data = pd.DataFrame({
    'Amount': [1000, 1500, 1200, 300],
    'VATRate': [20, 15, 10, 5],
    'SupplierPrice': [950, 1400, 1100, 280],
    'PaymentStatus': [0, 1, 0, 1]  # 0 = Paid, 1 = Unpaid
})

# Ajout de log_amount
data['log_amount'] = np.log1p(data['Amount'])

# Variables d’entrée et cible
X = data[['Amount', 'VATRate', 'SupplierPrice', 'PaymentStatus', 'log_amount']]
y = [1, 0, 1, 0]  # 1 = Approved, 0 = Not Approved

# Entraînement du modèle
model = LogisticRegression()
model.fit(X, y)

# Sauvegarde du modèle compatible
with open('Amount_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Nouveau modèle Amount_model.pkl généré avec succès.")
