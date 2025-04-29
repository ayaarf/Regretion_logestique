# 📊 Analyse des Transactions Journalières - Monoprix France

Une application interactive construite avec **Streamlit** pour :
- Visualiser les ventes journalières,
- Filtrer les transactions par date,
- Exporter les données filtrées,


---

## 🚀 Fonctionnalités

### 🔍 Exploration des données
- Connexion à une base de données SQL Server.
- Extraction de données transactionnelles (`TransactionDate`, `Amount`, `PaymentMethod`).
- Affichage des statistiques descriptives.
- Courbe des ventes journalières via Seaborn.

### 📅 Filtrage dynamique
- Sélection d'une plage de dates.
- Affichage des transactions correspondantes.
- Téléchargement en `.csv`.

### 📈 Prédiction des ventes
- Utilisation du modèle **Prophet**.
- Prévision du chiffre d'affaires futur pour X jours.
- Téléchargement des résultats prédits.

---

## ⚙️ Technologies utilisées

- `Streamlit` : Interface web interactive
- `Pandas` : Manipulation des données
- `Matplotlib` & `Seaborn` : Visualisation
- `Prophet` : Modèle de prévision temporelle
- `pyodbc` : Connexion à SQL Server

---

## 🛠️ Pré-requis

- Python ≥ 3.8
- Une base de données SQL Server avec une table `SA_Sales_Transactions`
- Bibliothèques à installer :

```bash
pip install streamlit pandas matplotlib seaborn prophet pyodbc
