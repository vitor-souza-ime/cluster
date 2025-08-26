# Agglomerative Clustering com o Dataset Iris

Este repositório contém um exemplo de aplicação de **Clustering Aglomerativo Hierárquico** utilizando o conjunto de dados clássico **Iris**. O projeto realiza a segmentação não supervisionada das espécies de Iris com base em características morfométricas de sépalas e pétalas.

## Repositório
[https://github.com/vitor-souza-ime/cluster](https://github.com/vitor-souza-ime/cluster)

## Arquivos
- `main.py` – Script principal em Python que realiza o clustering e gera gráficos de visualização.

## Descrição do Projeto
O algoritmo de **Agglomerative Clustering** é aplicado para identificar padrões naturais nos dados de Iris sem utilizar os rótulos das espécies. São criados dois gráficos com **Seaborn**:
1. **Gráfico de sépalas**: comprimento vs largura da sépala.
2. **Gráfico de pétalas**: comprimento vs largura da pétala.

As legendas indicam os clusters encontrados pelo modelo e a correspondência aproximada com as espécies:
- 0 = Setosa  
- 1 = Versicolor  
- 2 = Virginica  

## Tecnologias Utilizadas
- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn

## Como Executar
1. Clone o repositório:
```bash
git clone https://github.com/vitor-souza-ime/cluster.git
````

2. Instale as dependências (recomendado criar um virtual environment):

```bash
pip install pandas seaborn matplotlib scikit-learn
```

3. Execute o script:

```bash
python main.py
```

4. O script irá gerar dois gráficos mostrando os clusters para sépalas e pétalas.

## Referência do Dataset

O conjunto de dados **Iris** é originalmente do estudo de Fisher (1936) e está disponível na biblioteca `scikit-learn`.

