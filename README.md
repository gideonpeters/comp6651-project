## COMP6651 - WINTER 2025 COURSE PROJECT

This repository contains the implementation, analysis, and experimentation of several well-known clustering algorithms as part of the **COMP 6651: Algorithm Design Techniques** course at Concordia University (Winter 2025).

## 📁 Directory Structure

```
├── datasets/
│   ├── AI_index_db.csv
│   ├── earthquakes_column_descriptors.txt
│   ├── earthquakes.csv
│   └── iris.csv

├── scripts/
│   ├── 01-dbscan.ipynb
│   ├── 02-optics.ipynb
│   ├── 03-gmm.ipynb
│   ├── 04-kmeans.ipynb
│   ├── 05-kmedoids.ipynb
│   ├── 06-affinity.ipynb
│   ├── 07-mean-shift.ipynb
│   ├── 08-birch.ipynb
│   └── 09-agglomerative.ipynb
```


## 📌 Project Objectives

The goal of this project is to:
- Understand, implement, and analyze the complexity of various clustering algorithms.
- Evaluate their performance using multiple metrics.
- Experiment with real-world datasets and compare custom implementations with Scikit-learn equivalents.

## 🧠 Implemented Clustering Algorithms

The following clustering techniques are implemented and analyzed:

- 📈 **k-means**
- 💎 **k-medoids**
- 🧠 **Gaussian Mixture Model (GMM)**
- 🌐 **DBSCAN**
- 📏 **OPTICS**
- 🌲 **BIRCH**
- 💬 **Affinity Propagation**
- 🔄 **Mean Shift**
- 🧩 **Agglomerative Hierarchical Clustering**

## 📐 Evaluation Metrics

All clustering algorithms are evaluated using the following metrics:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index (Variance Ratio Criterion)
- Adjusted Rand Index (ARI)
- Mutual Information (MI)
- Split
- Diameter

## 📊 Datasets Used

- **Iris Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset)
- **AI Global Index Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/katerynameleshenko/ai-index)
- **Global Earthquake Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/shreyasur965/recent-earthquakes)

These datasets are stored in the `datasets/` directory and are used in Part II of the project.

## ⚙️ Features

- Feature scaling and dissimilarity coefficient computations (Euclidean, mixed types).
- Complexity analysis of each algorithm.
- Visual and statistical comparison with Scikit-learn implementations.
- Experimentation with feature selection and metric effectiveness.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/gideonpeters/comp6651-project.git
   cd comp6651-project
   ```

2. Install dependencies (if applicable):
    ```bash
    pip install -r requirements.txt
    ```

3. Run algorithm scripts using .ipynb code blocks and in the right sequence.

4. View output and results in respective directories or generated plots.

## 📄 Reports
Detailed academic-style reports for Part I and Part II are provided (PDF format), discussing:

- Algorithm analysis and complexity
- Evaluation metrics and theoretical insights
- Experiments and comparative performance
- Team contributions and references

## 📜 Notes
- No input parameters are hardcoded; all data is read from input files.
- All AI-generated content (if any) will be disclosed clearly in the appendix sections of the reports.

## 👥 Team Contributions

| Team Member             | Contributions |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Gideon Peters**        | - Literature review on clustering algorithms.<br>- Implemented k-means, k-medoids, and GMM algorithms.<br>- Conducted complexity analysis for centroid-based clustering methods.<br>- Wrote sections on k-means accuracy and improvements. |
| **Bhoomi**               | - Implemented DBSCAN, OPTICS, and Affinity Propagation.<br>- Conducted complexity analysis for density-based clustering methods.<br>- Analyzed clustering evaluation metrics and provided mathematical definitions.<br>- Wrote sections on metrics appropriateness. |
| **Anjolaoluwa Lasekan** | - Implemented hierarchical clustering (BIRCH, Agglomerative).<br>- Prepared experiments and dataset preprocessing.<br>- Evaluated results using silhouette score, Davies-Bouldin Index, and ARI.<br>- Compiled the final report and formatted the document in LaTeX. |
| **Masum Newaz**         | - Highlighted affinity propagation.<br>- Worked on mean shift clustering algorithm.<br>- Compiled the final report and formatted the document in LaTeX. |

> All team members contributed equally to discussions, debugging, and reviewing the report. The project was a collaborative effort, and responsibilities were distributed at random.


Feel free to fork, explore, and adapt this project to learn more about clustering techniques and algorithmic design in machine learning!