import numpy as np
import pandas as pd

import sys
#https://www.kaggle.com/datasets/inversion/sentence-transformers-222
sys.path.append('../input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2')

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/llm-detect-ai-ensemble/train1.csv')
df2 = df[df['label']==0].reset_index(drop=True)
df = df[df['label']==1].reset_index(drop=True)

vector1 = model.encode(list(df["text"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=False)
vector2 = model.encode(list(df2["text"]), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=False)

from sklearn.manifold import TSNE

tsne = TSNE()

show_df = pd.concat([df, df2], ignore_index=True) # 两部分数据抽前1w条来可视化
show_emb = np.concatenate([vector1, vector2])
X_embedded = tsne.fit_transform(show_emb)
import seaborn as sns

df2.shape

show_df.loc[:df.shape[0], "src"] = "LLM"
show_df.loc[df.shape[0]:(df.shape[0]+df2.shape[0]), "src"] = "student"

sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=show_df['src'], legend='full', )
