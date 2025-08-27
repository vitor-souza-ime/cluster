# * *   *   *   *   *   *   *   *   *   *   *   *   *   *
# *               Agglomerative Clustering              *
# * *   *   *   *   *   *   *   *   *   *   *   *   *   *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data  # Atributos

# Aplicar Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = model.fit_predict(X)

# Criar um DataFrame para facilitar a visualização
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels

# Adicionar os nomes das espécies reais para comparação
df['Species'] = [iris.target_names[i] for i in iris.target]

# Mapear clusters para espécies (baseado na maioria)
import numpy as np
cluster_to_species = {}
for cluster in [0, 1, 2]:
    cluster_mask = labels == cluster
    species_in_cluster = iris.target[cluster_mask]
    most_common_species = np.bincount(species_in_cluster).argmax()
    cluster_to_species[cluster] = iris.target_names[most_common_species]

# Criar coluna com nomes das espécies baseado nos clusters
df['Cluster_Species'] = [f"Cluster {cluster} ({cluster_to_species[cluster]})" for cluster in labels]

# Visualizar os resultados com Seaborn - SÉPALAS
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='Cluster_Species', palette='tab10', data=df)
plt.title("Agglomerative Clustering no Conjunto de Dados Iris")
plt.xlabel("Comprimento da Sépala (cm)")
plt.ylabel("Largura da Sépala (cm)")
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()  # ← Adicionado plt.show() aqui

#coolwarm, viridis, plasma,
# Visualizar os resultados com Seaborn - PÉTALAS
plt.figure(figsize=(10, 6))
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='Cluster_Species', palette='tab10', data=df)
plt.title("Agglomerative Clustering - Pétalas")
plt.xlabel("Comprimento da Pétala (cm)")
plt.ylabel("Largura da Pétala (cm)")
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()  # ← Este já estava aqui
