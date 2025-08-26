import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# Carregar o Iris dataset
iris = load_iris()
X = iris.data

# Aplicar Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, metric='euclidean')
labels = model.fit_predict(X)

# Criar DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels

# Plot com sépalas
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='Cluster',
                palette='viridis', data=df, legend=False)
plt.title("Agglomerative Clustering no Conjunto de Dados Iris")
plt.xlabel("Comprimento da Sépala (cm)")
plt.ylabel("Largura da Sépala (cm)")

# Legenda manual mostrando nomes
import matplotlib.patches as mpatches
legend_labels = {0: "0 = Setosa", 1: "1 = Versicolor", 2: "2 = Virginica"}
patches = [mpatches.Patch(color=sns.color_palette('viridis', 3)[i], label=legend_labels[i]) for i in range(3)]
plt.legend(handles=patches, title="Clusters")
plt.show()

# Plot com pétalas
plt.figure(figsize=(10, 6))
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='Cluster',
                palette='viridis', data=df, legend=False)
plt.title("Agglomerative Clustering - Pétalas")
plt.xlabel("Comprimento da Pétala (cm)")
plt.ylabel("Largura da Pétala (cm)")
plt.legend(handles=patches, title="Clusters")
plt.show()
