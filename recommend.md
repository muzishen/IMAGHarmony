## Recommendation method
----
### 名词解释
>**召回**：从海量的item中挑选出一小部分作为候选集后，送到排序层用复杂的模型做精排。![](https://pic4.zhimg.com/v2-2ab55f5bab9bad92ab62491bd807d563_r.jpg)

>**冷启动**：冷启动指的是当推荐系统面临一些新的或没有足够历史数据的情况下，如何进行推荐。这些情况包括：
>- 新用户冷启动：当新用户注册或访问系统时，系统没有足够的历史数据来了解他们的兴趣和喜好。
>- 新物品冷启动：当新物品被引入系统中，没有足够的历史交互数据来了解用户对这些物品的喜好。
>- 新领域冷启动：当推荐系统扩展到新的领域或业务领域，系统需要适应新的数据和情境。
>- 冷启动问题通常更具挑战性，因为缺乏历史数据，传统的协同过滤方法可能不够有效。 

>**热启动**：热启动是指推荐系统已经积累了足够的历史数据，可以有效地为用户生成个性化推荐的情况。在热启动阶段，推荐系统可以依赖用户的历史行为和交互数据来进行推荐，因为系统已经了解了用户的兴趣和喜好。
热启动通常比冷启动更容易，因为有足够的数据来支持推荐算法的运行。
### 1. Collaborative Filtering(协同过滤，CF)  
#### 基于近邻数据统计（记忆）的CF
>&emsp;核心思想：“物以类聚、人以群分”
>&emsp;矩阵表示：
>$$
>\left[
>\begin{matrix}
>R11 &R12 &\cdots &R1m \\
>R21 &R22 &\cdots &R2m \\
>\vdots &\vdots &\ddots &\vdots\\
>Rn1 &Rn2 &\cdots &Rnm\\ 
>\end{matrix}
>\right]
>$$
>&emsp;其中，n表示用户的个数，m表示物品的个数，如果某个用户对某个标的物未产生行为，值为0。其中行向量代表某个用户对所有标的物的评分向量，列向量代表所有用户对某个标的物的评分向量。
>&emsp;行向量之间的相似度就是用户之间的相似度，列向量之间的相似度就是标的物之间的相似度。
>#### 1.1 基于用户的协同过滤
>&emsp;原理：计算用户之间的相似度
>#### 1.2 基于物品的协同过滤
>&emsp;原理：计算物品之间的相似度
>#### 1.3 计算相似度
>&emsp;余弦相似度:$sim(v1,v2)=\displaystyle\frac{v1\cdot v2}{\parallel v1 \parallel \ast \parallel v2 \parallel} $  
>#### 1.4 计算用户对未评价物品的感兴趣程度
>##### &emsp;基于用户：
>&emsp;$score(u,s)=\sum_{u_i \in U}{score(u_i,s)\ast sim(u,u_i)}$
>&emsp;其中U是与该用户最相似的用户集合(我们可以基于用户相似度找到与某用户最相似的K个用户)
>##### &emsp;基于物品：
>&emsp;$score(u,s)=\sum_{s_i \in S}{score(u,s_i)\ast sim(s,s_i)}$
>&emsp;其中S是所有用户操作过的标的物的列表  
#### 基于模型的CF
>&emsp;核心思想：基于模型的协同过滤推荐就是基于样本的用户喜好信息，训练一个推荐模型，然后根据实时的用户喜好的信息进行预测，计算推荐。  
>&emsp;常用方法：关联算法、聚类算法、分类算法、回归算法、**矩阵算法**、神经网络
### 2、Factorization Machine（因子分解机，FM）
### 3、Matrix Factorization (矩阵分解，MF)
>&emsp;矩阵分解，实际上就是把原来的大矩阵分解成两个小矩阵的乘积，在实际推荐计算时不再使用大矩阵，而是使用分解得到的连个小矩阵。按照矩阵分解的原理，我们会发现原来$m \times n$会分解成$ m \times k$和$ k \times n$的两个小矩阵，这里多出来的k维向量（K的值自己设定），也就是隐因子向量。![](https://img-blog.csdnimg.cn/20201106133936468.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NDE1NzU4,size_16,color_FFFFFF,t_70#pic_center)  
>目标函数：$ \min\limits_{X,Y} \sum\limits_{r_{ui\neq0}}{(r_{ui}-X_u^TY_i)^2} +\lambda(\sum\limits_{u} {\parallel X_u \parallel}^2_2+\sum\limits_{i} {\parallel Y_i \parallel}^2_2)$
```python
#加载数据
import pandas as pd
import numpy as np
rating = pd.read_csv("./ml-latest-small/ratings.csv",sep=",")
num_user = np.max(rating["userId"])
num_movie = np.max(rating["movieId"])
print(num_user,num_movie,len(rating))

#搭建模型
import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras import Model
import keras.backend as K
from keras.layers import Embedding,Reshape,Input,Dot
K.clear_session()
def Recmand_model(num_user,num_movie,k):
    input_uer = Input(shape=[None,],dtype="int32")
    model_uer = Embedding(num_user+1,k,input_length = 1)(input_uer)
    model_uer = Reshape((k,))(model_uer)

    input_movie = Input(shape=[None,],dtype="int32")
    model_movie  = Embedding(num_movie+1,k,input_length = 1)(input_movie)
    model_movie = Reshape((k,))(model_movie)

    out = Dot(1)([model_uer,model_movie])
    model = Model(inputs=[input_uer,input_movie], outputs=out)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    return model

#模型训练
model = Recmand_model(num_user,num_movie,100)
train_user = rating["userId"].values
train_movie = rating["movieId"].values
train_x = [train_user,train_movie]
train_y = rating["rating"].values
model.fit(train_x,train_y,batch_size = 100,epochs =10)

#预测测试
a=np.array([1])
b=np.array([1])
test=[a,b]
model.predict(test)
```
### 4、Neural Collaborative Filtering(神经协同过滤，NCF)
### 5、ConvNCF
### 6、GCF