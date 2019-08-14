
聚类分析：是一种将研究对象分为相对同质的群组的统计分析技术

将观测对象的群体按照相似性和相异性进行不同群组的划分,划分后每个群组内部各对象相似度很高, 而不同群组之间的对象彼此相异度很高

聚类分析后会产生一组集合,主要用于降维

K均值算法实现逻辑：
K均值算法需要输入待聚类的数据和欲聚类的簇数K
1、随机生成k个初始点作为质心
2、将数据集中的数据按照距离质心的远近分到各个簇中
3、将各个簇中的数据求平均值,作为新的质心,重复上一步,直到所有的簇不再改变


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
from sklearn.datasets.samples_generator import make_blobs
#make_blobs 聚类数据生成器

x,y_ture = make_blobs(n_samples= 300,
                     centers= 4,
                     cluster_std= 0.5,
                     random_state= 0)
#n_samples : 待生成的样本总数
#centers : 类别数
#cluster_std : 每个类别的方差,如多类数据不同方差,可设置为[1.0,3.0]这里针对2类数据
#random_state : 随机数种子
# x → 生成数据值 , y → 生成数据对应的类别标签
print(x[:5])
print(y_ture[:5])
print('-------------------------------------')

#plt.figure(figsize = (16,6))
#plt.scatter(x[:,0],x[:,1],s = 10 , alpha = 0.6)
#plt.grid()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)
centroids = kmeans.cluster_centers_

plt.figure(figsize = (16,6))
plt.scatter(x[:,0],x[:,1],c = y_kmeans,cmap = 'Dark2',s = 50,alpha = 0.5,marker = 'x')
plt.scatter(centroids[:,0],centroids[:,1],c = [0,1,2,3],cmap = 'Dark2',s = 70,marker = 'o')
plt.title('K-means 300 points\n')
plt.xlabel('value1')
plt.ylabel('value2')
plt.grid()
```

    [[ 1.03992529  1.92991009]
     [-1.38609104  7.48059603]
     [ 1.12538917  4.96698028]
     [-1.05688956  7.81833888]
     [ 1.4020041   1.726729  ]]
    [1 3 0 3 1]
    -------------------------------------



![png](output_3_1.png)

