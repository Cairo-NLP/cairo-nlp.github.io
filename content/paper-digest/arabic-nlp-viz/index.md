---
title: "Arabic NLP Papers Explorer"
date: 2025-11-10
draft: false
---

### Interactive UMAP visualization of research papers in Arabic NLP conferences (2023–2025)

{{< rawhtml >}}
<div style="border: 2px solid #4a5568; 
            border-radius: 8px; 
            padding: 30px; 
            margin: 40px 0;
            text-align: center;
            background: #ffffff;">
    <div style="display: flex; 
                justify-content: center; 
                align-items: center; 
                gap: 30px; 
                flex-wrap: wrap;">
        
        <!-- Interactive Map Button -->
        <a href="https://cairo-nlp.github.io/paper-map/" 
           target="_blank"
           style="display: inline-flex;
                  align-items: center;
                  padding: 16px 32px; 
                  background: #2d3748;
                  color: white; 
                  text-decoration: none; 
                  border-radius: 6px; 
                  font-size: 16px; 
                  font-weight: 600;
                  transition: all 0.2s ease;
                  cursor: pointer;
                  border: 2px solid #2d3748;"
           onmouseover="this.style.background='#4a5568'; this.style.borderColor='#4a5568';"
           onmouseout="this.style.background='#2d3748'; this.style.borderColor='#2d3748';">
            <span style="margin-right: 8px;">🗺️</span>
            Go to Interactive Map
            <span style="margin-left: 8px;">→</span>
        </a>
        
        <span style="color: #718096; font-weight: 500; font-size: 16px;">
            OR
        </span>
        
        <!-- Continue Reading Button -->
        <a href="" 
           style="display: inline-flex;
                  align-items: center;
                  padding: 16px 32px; 
                  background: transparent;
                  color: #2d3748; 
                  text-decoration: none; 
                  border: 2px solid #2d3748;
                  border-radius: 6px; 
                  font-size: 16px; 
                  font-weight: 600;
                  transition: all 0.2s ease;
                  cursor: pointer;"
           onmouseover="this.style.background='#f7fafc'; this.style.borderColor='#4a5568';"
           onmouseout="this.style.background='transparent'; this.style.borderColor='#2d3748';">
            <span style="margin-right: 8px;">📖</span>
            Continue to Read the Code
            <span style="margin-left: 8px;">↓</span>
        </a>
    </div>
    
    <p style="margin-top: 20px; 
              color: #718096; 
              font-size: 14px;">
    </p>
</div>
{{< /rawhtml >}}

The **ArabicNLP** conference series began in **2023** with the first edition, [**ArabicNLP 2023 @ EMNLP 2023**](https://www.sigarab.org/), organized by **SIGARAB** (the ACL Special Interest Group on Arabic NLP) and co-located with **EMNLP 2023** in *Singapore* (December 7 2023).  
- **Main conference:** 80 submissions · 38 accepted (47%)  
- **Shared tasks:** 5 overview papers · 48 system papers  

The **second edition**, **ArabicNLP 2024**, took place on **August 16 2024** in *Bangkok, Thailand*, co-located with **ACL 2024**.  
- **Main conference:** 68 submissions · 31 accepted (42.6%)  
- **Shared tasks:** 8 overview papers · 71 system papers  

The **third edition**, **ArabicNLP 2025**, is scheduled for **November 8–9 2025** in *Suzhou, China*, co-located with **EMNLP 2025**.  
- **Accepted papers:** 40 main · 139 shared-task  
- (Submission totals not yet announced)

---

### Goal of This Notebook

To build an **interactive 2D visualization** of all published papers across the three ArabicNLP editions to explore the published work interactively, with clusters, and LLM-generated cluster names. We will do that via:

- download all papers across the three editions
- embedd the abstract of each paper
- cluster the embeddings using K-means
- name the clusters using LLM
- build 2D-visualization of the embeddings after dimension reduction using UMAP.
---

### Code

All needed code is in [Arabic-NLP-Papers-visualization reposiroty](https://github.com/Cairo-NLP/Arabic-NLP-Papers-visualization)


---

### Credits

This work is **inspired by and adapted from** [**Jay Alammar**](https://jalammar.github.io/) and his visualization of **NeurIPS 2025** papers, featured [here](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual).


## 1. Download and Collect ArabicNLP Papers

### 1.1 Download pdfs and metadata

In this step, we will download all papers from the three ArabicNLP conference editions.  
You can browse the official proceedings [here](https://aclanthology.org/venues/arabicnlp/).

We’ll use the `paper_crawler.py` script to fetch the paper metadata and PDFs.  
Run the script using the `runner.py` helper with the following commands:

```bash
# ArabicNLP 2025
python3 runner.py --base-url "https://aclanthology.org/2025.arabicnlp-main.{id}" --year 2025 --start 1 --end 40 --delay 1.0

# ArabicNLP 2024
python3 runner.py --base-url "https://aclanthology.org/2024.arabicnlp-1.{id}" --year 2024 --start 1 --end 107 --delay 10.0

# ArabicNLP 2023
python3 runner.py --base-url "https://aclanthology.org/2023.arabicnlp-1.{id}" --year 2023 --start 1 --end 91 --delay 10.0


### 1.2 Import and collect metadata in dataframe


```python
import os
import requests
import umap
from pathlib import Path
import json

from typing import Any
from paper_craweler import PaperCrawler
from embedder import get_embedding
import altair as alt
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

```

    /Users/ahmed/Desktop/arabic-nlp-2025-viz/.venv/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
def load_metadata_dir(directory: str) -> pd.DataFrame:
    files = os.listdir(directory)
    data = []
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                data.append(metadata)
    df = pd.DataFrame(data)
    return df

def classify_title(title: str, shared_tasks: list) -> str:
    if " at " in title:
        return "shared_task"
    if any(name in title for name in shared_tasks):
        return "shared_task"
    return "main"

shared_tasks_2024_name = [
    'AraFinNLP',
    'FIGNEWS',
    'ArAIEval',
    'StanceEval2024',
    'WojoodNER',
    'ArabicNLU',
    'NADI',
    'KSAA-CAD'
]

shared_tasks_2023_name = [
    'WojoodNER',
    'NADI',
    'KSAA-RD',
    'Qur\'an QA',
    'ArAIEval'
]
```


```python
# load the 2025 metadata papers and create a dataframe

df_2025 = load_metadata_dir("metadata_2025/")
df_2025['year'] = 2025
df_2025['type'] = 'main'

# load the 2025 shared task metadata papers and create a dataframe
df_2025_shared = load_metadata_dir("metadata_2025_shared_tasks/")
df_2025_shared['year'] = 2025
df_2025_shared['type'] = 'shared_task'

# load the 2024 metadata papers and create a dataframe
df_2024 = load_metadata_dir("metadata_2024/")
df_2024['year'] = 2024
df_2024["type"] = df_2024["title"].apply(classify_title, args=(shared_tasks_2024_name,))

# load the 2023 metadata papers and create a dataframe
df_2023 = load_metadata_dir("metadata_2023/")
df_2023['year'] = 2023
df_2023["type"] = df_2023["title"].apply(classify_title, args=(shared_tasks_2023_name,))

# combine all dataframes
df = pd.concat([df_2023, df_2024, df_2025, df_2025_shared], ignore_index=True)

df.shape

```




    (328, 13)



## 2. Embed the Abstracts

After collecting the metadata for all papers (including their abstracts),  
we generate vector representations using OpenAI’s **`text-embedding-3-small`** model.

This is done through the `get_embedding()` function defined in the `embedder.py` script.


```python
# loop over all rows, send abstract (or title) to embedder and store the embedding in a list

def create_embedding_array(df: pd.DataFrame) -> Any:
    embeddings = []
    for index, row in df.iterrows():
        title = row['title']
        abstract = row['abstract']
        text = f"{abstract}"
        if not abstract or abstract.strip() == "":
            text = title
            print(f"Using title for embedding at index {index}")
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

# check if embeddings_2023_2024_2025_abstracts_only.npy exists, if yes load it, if not create it and save it
if Path("embeddings_2023_2024_2025_abstracts_only.npy").exists():
    embeddings_array = np.load("embeddings_2023_2024_2025_abstracts_only.npy", allow_pickle=True)
else:
    embeddings = create_embedding_array(df)
    embeddings_array = np.array(embeddings)
    np.save("embeddings_2023_2024_2025_abstracts_only.npy", embeddings_array)

embeddings_array.shape
```




    (328, 1536)



### 3. UMAP for Dimensionality Reduction

For visualization, we apply **UMAP (Uniform Manifold Approximation and Projection)** to reduce the embedding space into two dimensions. This makes it easier to explore the relationships between papers interactively in the upcoming visualization.  

The hyperparameters in UMAP are only lightly tuned here. Feel free to experiment with values such as:
- **`n_neighbors`** (controls local vs. global structure)
- **`min_dist`** (controls how tightly points are packed)
- **`metric`** (defines the distance function, e.g., *cosine* or *euclidean*)

We use **2 components** (`n_components=2`) to produce a compact 2D projection suitable for visual exploration on the interactive plot.



```python
# 1. Generate UMAP embeddings with optimized parameters
print("Generating UMAP embeddings...")
reducer = umap.UMAP(
    n_neighbors=50,
    n_components=2,
    min_dist=0.1,
    metric='cosine',
)
umap_coords = reducer.fit_transform(embeddings_array)

# Add UMAP coordinates to dataframe
df['x'] = umap_coords[:, 0]
df['y'] = umap_coords[:, 1]
```

    Generating UMAP embeddings...

### 4. Cluster the Embeddings Using K-Means

After reducing the embeddings with UMAP, we apply **K-Means** clustering to group similar papers together based on their embedding representations.  

The number of clusters is a flexible parameter, you are encouraged to experiment with different. You can also use the **elbow method** or **silhouette score** to help determine an appropriate number of clusters.



```python
# 2. Perform clustering with dynamic cluster count
print("Clustering papers...")
# Determine optimal number of clusters based on data size
n_clusters = 5

kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans_model.fit_predict(embeddings_array)
df['cluster'] = clusters
```

    Clustering papers...


### 5. Extract Keywords for Each Cluster

To make each cluster more interpretable, we extract **representative keywords** from the papers assigned to it.

We first aggregate the text (here we use the **abstract** field) of all papers within the same cluster into a single document per cluster. Then we:

1. Build a **bag-of-words representation** using `CountVectorizer`, including unigrams and bigrams and filtering out very rare terms.
2. Apply **Class-based TF–IDF (c-TF-IDF)** via `ClassTfidfTransformer` to compute terms that are particularly characteristic of each cluster.
3. For every cluster, select the **top keywords** based on their c-TF-IDF scores and store:
   - A human-readable **cluster label** that includes the top 2–3 keywords.
   - A full list of cluster **keywords** to be used later by llm for clusters naming.

These labels and keyword lists are then added back to the main dataframe so they can be used in tooltips and legends in the interactive UMAP plot.



```python
# 3. Extract keywords for each cluster
print("Extracting cluster keywords...")
# Combine title and abstract for better keyword extraction
documents = df['abstract'].fillna('')

# Create documents dataframe for clustering
docs_df = pd.DataFrame({
    "Document": documents,
    "ID": range(len(documents)),
    "Topic": clusters
})

# Group documents by topic
documents_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

# Extract keywords using TF-IDF
count_vectorizer = CountVectorizer(
    stop_words="english",
    max_features=200,
    ngram_range=(1, 2),  # Include bigrams
    min_df=2
)
count = count_vectorizer.fit(documents_per_topic.Document)
count_matrix = count.transform(documents_per_topic.Document)
words = count.get_feature_names_out()

# Apply c-TF-IDF
ctfidf = ClassTfidfTransformer().fit_transform(count_matrix).toarray()

# Get top keywords for each cluster
words_per_cluster = {}
all_words_per_cluster = {}
for label in documents_per_topic.Topic:
    top_indices = ctfidf[label].argsort()[-10:][::-1]  # Top 8 keywords
    top_words = [words[idx] for idx in top_indices]
    words_per_cluster[label] = top_words

    all_indices = ctfidf[label].argsort()[::-1]  # All keywords sorted
    all_top_words = [words[idx] for idx in all_indices]
    all_words_per_cluster[label] = all_top_words

# Create cluster labels with top 3 keywords
cluster_labels = {}
for cluster_id, keywords in words_per_cluster.items():
    cluster_labels[cluster_id] = f"Cluster {cluster_id}: {', '.join(keywords[:3])}"

df['cluster_label'] = df['cluster'].map(cluster_labels)
df['keywords'] = df['cluster'].map(lambda x: ', '.join(words_per_cluster[x]))

```

    Extracting cluster keywords...


### 6. Name the Clusters Using GPT-5

Here we experiment with automatically assigning more descriptive names to each cluster using GPT-5.  

The process works as follows:

1. **Collect representative titles**: for every cluster, we gather the titles of all papers belonging to it.
2. **Compose a structured prompt**: the prompt includes both:
   - The cluster’s most relevant keywords (from the c-TF-IDF step), and  
   - The titles of representative papers in that cluster.
3. **Query GPT-5** — we send the prompt to GPT-5 via a custom helper (`llm_call_openai`) asking it to produce a concise, meaningful label (maximum 5 words) that summarizes the main research direction of the cluster.
4. **Cache the results** — to avoid redundant API calls, the generated labels are stored locally in a JSON file (`cluster_labels_llm.json`). If the file already exists, it is loaded and reused; otherwise, it is created and saved after generation.


```python
# get the titles of papers in each cluster
titles_in_cluster = {}
for cluster_id in range(n_clusters):
    titles_in_cluster[cluster_id] = df[df['cluster'] == cluster_id]['title'].tolist()   
```


```python
# name the cluster using LLM
from utils import llm_call_openai

# check if cluster_labels_llm.json exists, if yes load it, if not create it and save it
if Path("cluster_labels_llm.json").exists():
    with open("cluster_labels_llm.json", "r", encoding="utf-8") as f:
        cluster_labels_llm = json.load(f)
else:
    cluster_labels_llm = {}

    for i in range(n_clusters):
        titles = titles_in_cluster[i]
        titles_text = '; '.join(titles)
        prompt = f"""Provide a descriptive label for the given Arabic NLP research paper cluster.\n
        The label will be used as legend in a visualization of Arabic NLP research trends from 2023 to 2025.\n
        Make sure the label is concise (max 5 words), informative, and captures the main research directions of the cluster.\n
        You cant say theme is shared tasks or main track, you need to focus on research topics.\n
        Use abbreviations where appropriate.\n
        Cluster keywords: {', '.join(all_words_per_cluster[i])}.\n
        Titles of representative papers: {titles_text}.\n
        Respond exactly in this format: LABEL"""
        response = llm_call_openai(prompt, model='gpt-5')
        cluster_labels_llm[i] = response
        print(f"Cluster {i} LLM response: {cluster_labels_llm[i]}")

        with open("cluster_labels_llm.json", "w", encoding="utf-8") as f:
            json.dump(cluster_labels_llm, f, ensure_ascii=False, indent=4)



```

    backend used openai - gpt-5
    Cluster 0 LLM response: Arabic NLP: Disinfo, QA, MT
    backend used openai - gpt-5
    Cluster 1 LLM response: Arabic LLMs Dialects Bias Evaluation
    backend used openai - gpt-5
    Cluster 2 LLM response: Arabic Multitask Multimodal LLM Benchmarks
    backend used openai - gpt-5
    Cluster 3 LLM response: Arabic Dialects, NER & Misinformation
    backend used openai - gpt-5
    Cluster 4 LLM response: Dialects, Safety, Readability, RAG-QA



```python
# add cluster_labels_llm to dataframe
df['cluster_label_llm'] = df['cluster'].map(lambda x: cluster_labels_llm.get(x))
```

# 7. Create Visualization


```python
print("Step 4: Creating bright visualization...")

# Year filter
df['year'] = df['year'].astype(str)
year_dropdown = alt.binding_select(
    options=[None] + sorted(df['year'].unique().tolist()),
    labels=['All'] + sorted(df['year'].unique().tolist()),
    name='Year'
)
year_select = alt.selection_point(
    fields=['year'],
    bind=year_dropdown,
    empty='all'
)

# Type filter
type_dropdown = alt.binding_select(
    options=[None] + sorted(df['type'].unique().tolist()),
    labels=['All'] + sorted(df['type'].unique().tolist()),
    name='Type'
)
type_select = alt.selection_point(
    fields=['type'],
    bind=type_dropdown,
    empty='all'
)

# Legend selection for topics
selection = alt.selection_point(fields=['cluster_label_llm'], bind='legend', toggle='true')
'keywords'

# Main chart with bright theme
chart = alt.Chart(df).mark_circle(
    size=40, 
    stroke='#666', 
    strokeWidth=1, 
    opacity=0.3
).encode(
    x=alt.X('x',
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),
    y=alt.Y('y',
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),
    href='url:N',
    color=alt.Color('cluster_label_llm:N',
        legend=alt.Legend(
            columns=1, 
            symbolLimit=0, 
            labelFontSize=13,
            title='Research Topics',
            titleFontSize=14,
            labelLimit=0
        )
    ),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    size=alt.condition(selection, alt.value(100), alt.value(40)),
    tooltip=[
        alt.Tooltip('title:N', title='Title'),
        alt.Tooltip('authors:N', title='Authors'),
        alt.Tooltip('year:N', title='Year'),
        alt.Tooltip('type:N', title='Type'),
        #alt.Tooltip('conference:N', title='Conference'),
        alt.Tooltip('cluster_label_llm:N', title='Cluster')
    ]
).add_params(
    selection,
    year_select,
    type_select
).transform_filter(
    year_select
).transform_filter(
    type_select
).properties(
    width=800,
    height=500,
    title={
        'text': f'Arabic NLP Papers: {len(df)} Research Papers ({df["year"].min()}-{df["year"].max()})',
        'fontSize': 18,
        'font': 'system-ui, -apple-system, sans-serif',
        'color': '#000000'
    }
).configure_legend(
    labelLimit=0,
    labelFontSize=12,
    titleFontSize=13,
    strokeColor='#ddd',
    fillColor='#000000',
    padding=10,
    cornerRadius=5,
    labelColor='black',   # 👈 add this
    titleColor='black'    # 👈 and this
).configure_view(
    strokeWidth=0
).configure(
    background="#FDF7F0"
).interactive()

# open links in a new tab
chart['usermeta'] = {
    "embedOptions": {
        "loader": {"target": "_blank"}
    }
}

chart
```

    Step 4: Creating bright visualization...

{{< rawhtml >}}
<iframe src="/workflow/umap_viz.html" 
        width="100%" 
        height="600px" 
        frameborder="0">
</iframe>
{{< /rawhtml >}}
