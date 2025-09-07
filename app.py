import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from PIL import Image
import io

users = {"admin": "password", "user": "cluster123"}

def load_data():
    df = pd.read_csv("data.csv")
    X = df.values
    scaler = StandardScaler()
    return df, scaler.fit_transform(X)

df, X_scaled = load_data()

def run_clustering(algorithm, n_clusters, eps, min_samples):
    if algorithm == "KMeans":
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
    elif algorithm == "Hierarchical":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_scaled)
    else: 
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
    else:
        sil, db, ch = np.nan, np.nan, np.nan

    pca = PCA(n_components=2).fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6,4))
    sc = ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap="viridis", s=50, edgecolor='k')
    ax.set_title(f"{algorithm} Clustering")
    plt.colorbar(sc)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
    plt.close()
    img = Image.open(buf)

    results_df = pd.DataFrame({
        "Algorithm": [algorithm],
        "Clusters": [len(set(labels))],
        "Silhouette": [sil],
        "Davies-Bouldin": [db],
        "Calinski-Harabasz": [ch]
    })

    return results_df, img

def login(username, password):
    if username in users and users[username] == password:
        return gr.update(visible=False), gr.update(visible=True), f"✅ Welcome {username}!"
    else:
        return gr.update(visible=True), gr.update(visible=False), "❌ Wrong credentials!"

with gr.Blocks(css="""
    body {background-color:#f0f2f5; font-family:Arial, sans-serif;}
    .login-box {background:white; padding:30px; border-radius:12px; box-shadow:0 8px 25px rgba(0,0,0,0.1);}
    .login-btn {background-color:#4CAF50; color:white; font-weight:bold;}
    .header {color:#333; text-align:center; margin-bottom:20px;}
""") as app:

    with gr.Row():
        gr.Markdown("") 
        with gr.Column(scale=1, min_width=350):
            with gr.Group(visible=True) as login_group:
                gr.Markdown("## 🔑 Login", elem_classes="header")
                user = gr.Textbox(label="Username", placeholder="Enter your username")
                pwd = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                btn = gr.Button("Login", elem_classes="login-btn")
                status = gr.Markdown()
        gr.Markdown("") 

    with gr.Row(visible=False) as main_group:
        with gr.Column(scale=1, min_width=650):
            gr.Markdown("## 📊 Clustering Dashboard", elem_classes="header")
            algo = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], value="KMeans", label="Algorithm")
            k = gr.Slider(2, 10, step=1, value=3, label="n_clusters")
            eps = gr.Slider(0.1, 2.0, step=0.1, value=0.5, label="DBSCAN eps")
            min_s = gr.Slider(2, 20, step=1, value=5, label="DBSCAN min_samples")
            run = gr.Button("Run")
            table = gr.Dataframe()
            plot = gr.Image()

            run.click(run_clustering, [algo, k, eps, min_s], [table, plot])

    btn.click(login, [user, pwd], [login_group, main_group, status])

app.launch()
