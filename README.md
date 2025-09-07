# 🔍 Clustering Analysis Dashboard (End-to-End ML Project)

## App Link:
https://huggingface.co/spaces/Sharmilamalaiyarasan/clustering-dashboard
## 📌 Project Overview

This project is an End-to-End Machine Learning application that performs clustering analysis on datasets using multiple algorithms.

It provides a Gradio-powered interactive dashboard where users can log in, choose a clustering algorithm, evaluate results, and visualize clusters.

It covers the complete ML lifecycle:

📂 Data preprocessing – Scaling input data with StandardScaler

⚙️ Clustering algorithms – KMeans, Hierarchical, and DBSCAN

📊 Evaluation metrics – Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score

📈 Visualization – Elbow Method & PCA-based 2D cluster visualization

🔑 Authentication – Simple login system before accessing the dashboard

## 🚀 Features

✅ Multiple clustering algorithms (KMeans, Hierarchical, DBSCAN)
✅ Automatic evaluation with 3 cluster metrics
✅ Elbow Method for optimal K visualization
✅ PCA-based cluster scatterplot
✅ Secure login before analysis
✅ User-friendly Gradio web app

## 🛠️ Tech Stack

Python 🐍

Scikit-learn → Clustering models & metrics

Pandas / NumPy → Data handling

Matplotlib / Seaborn / Plotly → Visualization

Gradio → Interactive dashboard frontend

## 📂 Project Structure
Clustering_App/

├── app.py                 
├── data.csv               
├── requirements.txt       
└── README.md             

## ⚙️ Installation & Setup

2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the Gradio app

python app.py

## 📊 Example Workflow

Login with username & password

Choose clustering algorithm (KMeans, Hierarchical, DBSCAN)

Adjust parameters (clusters, eps, min_samples)

Run clustering → Get metrics + elbow plot + PCA visualization

## 📊 Evaluation Metrics

Silhouette Score → Measures how well clusters are separated

Davies-Bouldin Score → Lower is better (compact & separated clusters)

Calinski-Harabasz Score → Higher is better (well-defined clusters)

## 🔐 Authentication

Default users (can be extended to DB):

admin / password

user / cluster123

## 🎯 Future Enhancements

Upload custom datasets

Export clustering reports (PDF/CSV)

Role-based access (Admin vs User)

More clustering algorithms (Gaussian Mixture, OPTICS, etc.)
