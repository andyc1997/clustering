import numpy as np
import pandas as pd
import torch

from model import MiniBatchKMeans, LloydKMeans, MacQueenKMeans, HartiganWongKMeans


#region read iris data
X_tr = pd.read_csv(r'..\pca\iris.csv', sep=',')
labels_true = X_tr['Species']
X_tr = X_tr.drop([X_tr.columns[0], X_tr.columns[5]], axis=1)
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtypes}\n')

#endregion


#region preprocessing
X_tr = np.array((X_tr - X_tr.mean())/X_tr.std())
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtype}\n')
X_tr = torch.Tensor(X_tr)

#endregion


#region train model
# model = MiniBatchKMeans(max_epoch=500, k=3, batch_size=32, sparse=False)
# model.fit(X_tr)
# labels = model.update_center(X_tr)
# df_cluster = pd.DataFrame({'labels_pred': labels.numpy(), 'labels_true':labels_true})
# print(pd.crosstab(df_cluster['labels_pred'], df_cluster['labels_true']))

model = LloydKMeans(k=3)
model.fit(X_tr)
df_cluster = pd.DataFrame({'labels_pred': model.label.numpy(), 'labels_true':labels_true})
print(pd.crosstab(df_cluster['labels_pred'], df_cluster['labels_true']))

model = HartiganWongKMeans(k=3, max_iter=30)
model.fit(X_tr)
df_cluster = pd.DataFrame({'labels_pred': model.label.numpy(), 'labels_true':labels_true})
print(pd.crosstab(df_cluster['labels_pred'], df_cluster['labels_true']))
#endregion