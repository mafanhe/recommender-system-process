## 四、深度学习在个性化推荐中的应用

### 结论
得益于深度学习强大的表示能力，目前深度学习在推荐系统中需要对用户与物品进行表示学习的任务中有着不错的表现，但优势不如图像与文本那么显著[1]。
### 深度学习与分布式表示简介

    深度学习的概念源于人工神经网络的研究。深度学习通过组合低层特征形成更加抽象的高层表示属性类别或特征，以发现数据的有效表示，而这种使用相对较短、稠密的向量表示叫做分布式特征表示（也可以称为嵌入式表示）。本部分主要对于目前使用较广的一些学习算法进行一个简单的回顾。
    
    首先介绍一些浅层的分布式表示模型。目前在文本领域，浅层分布式表示模型得到了广泛的使用，例如word2vec、GloVec、fasttext等 [2]。与传统词袋模型对比，词嵌入模型可以将词或者其他信息单元（例如短语、句子和文档等）映射到一个低维的隐含空间。在这个隐含空间中，每个信息单元的表示都是稠密的特征向量。词嵌入表示模型的基本思想实际还是上来自于传统的“Distributional semantics”[3]，概括起来讲就是当前词的语义与其相邻的背景词紧密相关。因此，词嵌入的建模方法就是利用嵌入式表示来构建当前词和背景词之间的语义关联。相比多层神经网络，词嵌入模型的训练过程非常高效，而且实践效果很好、可解释性也不错，因此得到了广泛的应用 

    对应于神经网络模型，最为常见的模型包括多层感知器、卷积神经网络、循环神经网络、递归神经网络等 [4]。多层感知器主要利用多层神经元结构来构建复杂的非线性特征变换，输入可以为提取得到的多种特征，输出可以为目标任务的标签或者数值，本质上可以构建一种复杂的非线性变换；卷积神经网络可以直接部署在多层感知器上，感知器的输入特征很有可能是不定长或者有序的，通过多个卷积层和子采样层，最终得到一个固定长度的向量。循环神经网络是用来对于时序序列建模的常用模型，刻画隐含状态的关联性，可以捕捉到整个序列的数据特征。针对简单的循环神经网络存在长期依赖问题（“消失的导数”），不能有效利用长间隔的历史信息，两个改进的模型是长短时记忆神经网络（LSTM） 和基于门机制的循环单元（GRU）。递归神经网络根据一个外部给定的拓扑结构，不断递归得到一个序列的表示，循环神经网络可以被认为是一种简化的递归神经网络。

### 应用
#### 1.相似匹配

- 1.1.嵌入式表示模型
    通过行为信息构建用户和物品（或者其他背景信息）的嵌入式表示，使得用户与物品的嵌入式表示分布在同一个隐含向量空间，进而可以计算两个实体之间的相似性。
     
     > 很多推荐任务，本质可以转换为相关度排序问题，因此嵌入式表示模型是一种适合的候选方法。一般来说，浅层的嵌入式表示模型的训练非常高效，因此在大规模数据集合上有效性和复杂度都能达到不错的效果。
     
     在[5]中，嵌入式表示被应用到了产品推荐中，给定一个当前待推荐的产品，其对应的生成背景（context）为用户和上一个交易的产品集合，利用这些背景信息对应的嵌入式表示向量可以形成一个背景向量，刻画了用户偏好和局部购买信息的依赖关系。然后基于该背景向量，生成当前待推荐的产品。经推导，这种模型与传统的矩阵分解模型具有很强的理论联系。在[6]中，Zhao等人使用doc2vec模型来同时学习用户和物品的序列特征表示，然后将其用在基于特征的推荐框架中，引入的嵌入式特征可以在一定程度上改进推荐效果。在[7]中，嵌入式表示模型被用来进行地点推荐，其基本框架就是刻画一个地理位置的条件生成概率，考虑了包括用户、轨迹、临近的地点、类别、时间、区域等因素。


- 1.2.语义匹配模型
     
    [8]深度结构化语义模型（Deep Structured Semantic Models，简称为DSSM）是基于多层神经网络模型搭建的广义语义匹配模型 。其本质上可以实现两种信息实体的语义匹配。基本思想是设置两个映射通路，两个映射通路负责将两种信息实体映射到同一个隐含空间，在这个隐含空间，两种信息实体可以同时进行表示，进一步利用匹配函数进行相似度的刻画。
    ![avatar](recommend_intro/DSSM.PNG)
    如图展示了一个DSSM的通用示意图，其中Q表示一个Query，D表示一个Document，对应到推荐系统里面的用户和物品。通过级联的深度神经网络模型的映射与变换，最终Query和Document在同一个隐含空间得到了表示，可以使用余弦相似度进行计算。DSSM最初主要用在信息检索领域，用来刻画文档和查询之间的相似度。
    
    [9]随后被用在推荐系统中：一端对应着用户信息，另外一端对应着物品信息 。以DSSM为主的这些工作的基本出发点实际上和浅层嵌入式表示模型非常相似，能够探索用户和物品两种不同的实体在同一个隐含空间内的相似性。其中一个较为关键的地方，就是如何能够融入任务特定的信息(例如物品内容信息)以及模型配置（例如可以使用简单多层神经网络模型或者卷积神经网络模型），从而获得理想的结果。

#### 2.评分预测
- 2.1.基于用户的原始评分（或者反馈）来挖掘深度的数据模式特征（神经网络矩阵分解）
    
    [10]限制玻尔兹曼机进行评分预测。
    ![avatar](recommend_intro/限制玻尔兹曼机.PNG)
    如图所示，其所使用的模型具有一个两层的类二部图结构，其中用户层为隐含层 (h)，可见层为用户的评分信息 (V)，通过非线性关联两层上的数据信息。其中隐含层为二元变量，而用户评分信息被刻画为多项式分布变量。建立用户隐含表示信息以及其评分信息的联合能量函数，然后进行相应的参数求解。该方法的一个主要问题是连接隐含层和评分层的权重参数规模过大（对于大数据集合），也就是权重矩阵W。
        
    [11]优化计算的改进，作者进一步提出使用将W分解为两个低秩矩阵，减小参数规模。不过实验效果表明所提出的方法并没有比基于矩阵分解的方法具有显著的改进，而且参数求解使用较为费时的近似算法。
        
    [12]优化改进，Zheng 等人提出使用Neural Autoregressive Distribution Estimator来改进上述问题，该方法不需要显式对于二元隐含变量进行推理，减少了模型复杂度，并且使用排序代价函数来进行参数最优化。实验表明所提出的方法能够取得非常好的效果。
        
    [13]Wu等人使用去噪自动编码模型（Denoising Autoencoder）进行top-N物品推荐，其输入为加入噪声的对于物品的偏好（采纳为1，否则为0），输出为用户对于物品的原始评分，通过学习非线性映射关系来进行物品预测。
    ![avatar](recommend_intro/去噪自动编码模型.PNG)
    如图所示，用户可见的评分数据通过加上噪音后进入输入层，然后通过非线性映射形成隐含层，再由隐含层经映射后重构评分数据。注意，该模型中加入了用户偏好表示(User Node)和偏置表示(Bias Node)。
        
    [14]Devooght提出将协同过滤方法可以看作时间序列的预测问题。
    ![avatar](recommend_intro/CFRNN.PNG)

    作者提出，传统基于协同过滤的推荐方法，无论基于何种特征，都没有考虑用户历史行为的时间属性，只是将历史行为中的每个item统一考虑。这样处理带来的最大问题在于推荐系统无法分析用户喜好的变化情况，从而给出更符合用户现阶段喜好的推荐结果。那么，如果基于协同过滤“由过去，看未来”的思想，如果将该问题视作序列预测问题，一方面可以更好的分析用户的兴趣爱好的变化情况给出更好的推荐结果，另一方面也可以将在时序预测问题中广泛使用的RNN深度网络模型引入到推荐系统中。
        
    [15]NCF 作者提出一种通用的神经网络协同过滤框架，通过用神经网络结构多层感知机去学习用户-项目之间交互函数替代传统的矩阵分解中的内积运算，从而从数据中学习任意函数（非线性）。
    ![avatar](recommend_intro/CFDNN.PNG)

    并提出了两种NCF实例：基于线性核的GMF（广义矩阵分解），基于非线性核的MLP。并且将GMF与MLP融合，使他们相互强化。（tf model zoo）
    ![avatar](recommend_intro/CFDNN2.PNG)


- 2.2. 深度神经网络模型当做特征变换模块（内容embedding->矩阵分解）

    [16]Wang等人关注推荐系统中的一个重要问题：带有文本信息的评分预测（如博客文章等）。传统解决方法通常联合使用主题模型与矩阵分解（Collaborative Topic Modeling）。[16]中的主要想法就是替换掉主题模型，使用Stacked Denoising Autoencoders进行文本特征与评分预测中的数据特征相融合。
    
    在[17]中，Oord等人主要解决音乐推荐系统中的冷启动问题。通常来说，冷启动问题包括两个方面，新用户和新物品，这里主要考虑新物品。传统矩阵分解的推荐算法通过将评分分解为两个低秩向量来进行预测，也就是$\hat r_{i,j} =\vec{u_{i}}\cdot \vec{v_{j}}$，其中$\hat r_{i,j}$ 为用户i对于物品j 的预测评分,$\vec{u_{i}}$ 和$\vec{v_{j}}$ 是两个K维的向量，分别代表用户和物品的隐含表示。基本想法是从音乐的音频数据中提取到相关的特征$\vec{x_{j}}$ ，然后将这些音乐自身的数据特征映射为通过矩阵分解学习得到的隐含向量,也就是学习一个函数f，使之达到$f(\vec{x_{j} })\rightarrow \vec{v_{j} }$。通过学习这样的变换函数，当新音乐来到时，可以通过提取其自身的音频特征来得到其隐含向量，而不必要求使用用户数据来训练$\vec{v_{j}}$ 。得到$\vec{v_{j}}$ 的预测值之后，从而可以使用传统矩阵分解的方法来计算待推荐用户与新物品直接的相似性。
    
    与[17]非常相似，Wang等人在[18]中使用深度信念网络(Deep Belief Network)进行音频数据特征变换，不同的是同时保留两种表示，第一种表示从方法中得到的数据表示，而第二部分则对应基于内容方法得到的数据表示，最后两部分表示分别做点积，用来拟合最后的评分结果。
    
    这三种方法都是将传统协同过滤的矩阵分解方法与神经网络模型相结合的途径。


#### 3.排序
   Deep CTR [https://mp.weixin.qq.com/s/xWqpIHHISSkO97O_fKkb6A]

![avatar](recommend_intro/DeepCTR.PNG)

3.1. 总结（结论先行）


>1. FM 其实是对嵌入特征进行两两内积实现特征二阶组合；FNN 在 FM 基础上引入了 MLP； 
2. DeepFM 通过联合训练、嵌入特征共享来兼顾 FM 部分与 MLP 部分不同的特征组合机制； 3. NFM、PNN 则是通过改造向量积的方式来延迟FM的实现过程，在其中添加非线性成分来提升模型表现力； 
4. AFM 更进一步，直接通过子网络来对嵌入向量的两两逐元素乘积进行加权求和，以实现不同组合的差异化，也是一种延迟 FM 实现的方式； 
5. DCN 则是将 FM 进行高阶特征组合的方向上进行推广，并结合 MLP 的全连接式的高阶特征组合机制； 
6. Wide&Deep 是兼容手工特征组合与 MLP 的特征组合方式，是许多模型的基础框架； 
7. Deep Cross 是引入残差网络机制的前馈神经网络，给高维的 MLP 特征组合增加了低维的特征组合形式，启发了 DCN； 
8. DIN 则是对用户侧的某历史特征和广告侧的同领域特征进行组合，组合成的权重反过来重新影响用户侧的该领域各历史特征的求和过程； 
9. 多任务视角则是更加宏观的思路，结合不同任务（而不仅是同任务的不同模型）对特征的组合过程，以提高模型的泛化能力。
    
3.2. DNN

    深度排序模型( embedding-神经网络),embedding+MLP 是对于分领域离散特征进行深度学习 CTR 预估的通用框架。深度学习在特征组合挖掘（特征学习）方面具有很大的优势。比如以 CNN 为代表的深度网络主要用于图像、语音等稠密特征上的学习，以 W2V、RNN 为代表的深度网络主要用于文本的同质化、序列化高维稀疏特征的学习。CTR 预估的主要场景是对离散且有具体领域的特征进行学习，所以其深度网络结构也不同于 CNN 与 RNN。
    
![avatar](recommend_intro/embedding_mlp.PNG)  

    embedding+MLP 的过程如下： 

    1. 对不同领域的 one-hot 特征进行嵌入（embedding），使其降维成低维度稠密特征。 
    2. 然后将这些特征向量拼接（concatenate）成一个隐含层。 
    3. 之后再不断堆叠全连接层，也就是多层感知机（Multilayer Perceptron, MLP，有时也叫作前馈神经网络）。 
    4. 最终输出预测的点击率。 

    
3.3. Wide & Deep Network(连续特征->交叉特征+LR、离散特征->onehot->DNN)

![avatar](recommend_intro/WideAndDeepNetwork.PNG)

    Google 在 2016 年提出的宽度与深度模型（Wide&Deep）在深度学习 CTR 预估模型中占有非常重要的位置，它奠定了之后基于深度学习的广告点击率预估模型的框架。 Wide&Deep将深度模型与线性模型进行联合训练，二者的结果求和输出为最终点击率。其计算图如下： 

3.4. DeepFM

    在Wide & Deep Network基础上进行的改进，DeepFM的Wide部分是 FM

3.5. Deep & Cross Network（特征->cross netword+LR、DNN）
    
    Ruoxi Wang 等在 2017 提出的深度与交叉神经网络（Deep & Cross Network，DCN）借鉴了FM的特征点击交叉。DCN 的计算图如下：

![avatar](recommend_intro/DeepAndCrossNetwork.PNG)

    DCN 的特点如下：
    1. Deep 部分就是普通的 MLP 网络，主要是全连接。 
    2. 与 DeepFM 类似，DCN 是由 embedding + MLP 部分与 cross 部分进行联合训练的。Cross 部分是对 FM 部分的推广。 
    3. Cross 部分的公式如下：
    4. 可以证明，cross 网络是 FM 的过程在高阶特征组合的推广。完全的证明需要一些公式推导，感兴趣的同学可以直接参考原论文的附录。
    5. 而用简单的公式证明可以得到一个很重要的结论：只有两层且第一层与最后一层权重参数相等时的 Cross 网络与简化版 FM 等价。
    6. 此处对应简化版的 FM 视角是将拼接好的稠密向量作为输入向量，且不做领域方面的区分（但产生这些稠密向量的过程是考虑领域信息的，相对全特征维度的全连接层减少了大量参数，可以视作稀疏链接思想的体现）。而且之后进行 embedding 权重矩阵 W 只有一列——是退化成列向量的情形。
    7. 与 MLP 网络相比，Cross 部分在增加高阶特征组合的同时减少了参数的个数，并省去了非线性激活函数
    
3.6. DIN [Deep Interest Network]对同领域历史信息引入注意力机制的MLP
   
    以上神经网络对同领域离散特征的处理基本是将其嵌入后直接求和，这在一般情况下没太大问题。但其实可以做得更加精细。
    由 Bahdanau et al. (2015) 引入的现代注意力机制，本质上是加权平均（权重是模型根据数据学习出来的），其在机器翻译上应用得非常成功。受注意力机制的启发，Guorui Zhou 等在 2017 年提出了深度兴趣网络（Deep Interest Network，DIN）。DIN 主要关注用户在同一领域的历史行为特征，如浏览了多个商家、多个商品等。DIN 可以对这些特征分配不同的权重进行求和。其网络结构图如下：
 
![avatar](recommend_intro/DeepInterestNetwork.PNG)  
 
    1. 此处采用原论文的结构图，表示起来更清晰。
    2. DIN 考虑对同一领域的历史特征进行加权求和，以加强其感兴趣的特征的影响。
    3. 用户的每个领域的历史特征权重则由该历史特征及其对应备选广告特征通过一个子网络得到。即用户历史浏览的商户特征与当前浏览商户特征对应，历史浏览的商品特征与当前浏览商品特征对应。
    4. 权重子网络主要包括特征之间的元素级别的乘法、加法和全连接等操作。
    5. AFM 也引入了注意力机制。但是 AFM 是将注意力机制与 FM 同领域特征求和之后进行结合，DIN 直接是将注意力机制与同领域特征求和之前进行结合。

3.7. FM -> FNN -> NFM -> PNN -> AFM
    
    LR:
![avatar](recommend_intro/LR.PNG)  
    
    FM：
![avatar](recommend_intro/FM2.PNG)  
    
    FNN：FM隐向量 + 拼接 + MLP
![avatar](recommend_intro/FNN.PNG)  

    
    NFM：FM隐向量 + 特征交叉（逐元素向量乘法）+ 求和 + MLP
![avatar](recommend_intro/NFM.PNG)  
 
    PNN：与NFM类似，特征交叉法采用了向量积的方法 + 拼接 + mlp
![avatar](recommend_intro/PNN.PNG)  

    AFM：基于NFM的改进，通过在逐元素乘法之后形成的向量进行加权求和（Attention Net），去除了MLP部分直接接一个softmax
![avatar](recommend_intro/AFM.PNG)  

3.8. 多任务学习：同时学习多个任务
![avatar](recommend_intro/MultiTask.PNG)  

    - 完全共享网络层的参数
    - 只共享embedding层参数
    
#### 4.序列预测
    循环神经网络（刻画隐含状态的关联性，可以捕捉到整个序列的数据特征）
    
    [19]Hidasi等人使用循环神经网络进行基于session的推荐，该工作是对于RNN的一个直接应用。
    
    [20]Brébisson等人使用神经网络模型进行解决2015年的ECML/PKDD 数据挑战题目“出租车下一地点预测”，取得了该比赛第一名。在[20]中，作者对于多种多层感知器模型以及循环神经网络模型进行对比，最后发现基于改进后的多层感知器模型取得了最好的效果，比结构化的循环神经网络的效果还要好。
    
    在[21]中，Yang等人同时结合RNN及其变种GRU模型来分别刻画用户运动轨迹的长短期行为模式，通过实验验证，在“next location”推荐任务中取得了不错的效果。如图5所示，给定一个用户生成的轨迹序列，在预测下一个地点时，直接临近的短期访问背景和较远的长期访问背景都同时被刻画。
    
    (加入用户信息，内容信息，上下文信息的RNN模型)
    TODO
    此外还有一些基于RNN的优化模型[https://zhuanlan.zhihu.com/p/30720579]
    
    
    - GRU4REC[22]，使用GRU单元
    - GRU4REC+item features[23]，加入内容特征
    - GRU4REC+sampling+Dwell Time[24], 将用户在session中item上的停留时间长短考虑进去
    - Hierachical RNN[25],一种层次化的RNN模型，相比之前的工作，可以刻画session中用户个人的兴趣变化，做用户个性化的session推荐。
    - GRU4REC+KNN[26], 将session 中的RNN模型，与KNN方法结合起来，能够提高推荐的效果。
    - Improvenment GRU4REC[27]，基于GRU4REC的训练优化
    - GRU + attention[28]，加入attention机制
    
### 原因：
- 原始的用户物品二维矩阵框架（基于协同，矩阵分解）不能完全刻画复杂的推荐任务。
- 数据采集维度不够，特征太稀疏，影响用户的上下文环境过于复杂

### 展望
- 结构化神经网络RNN
- 深度强化学习

### 参考
[[1].深度学习在推荐算法上的应用进展](https://zhuanlan.zhihu.com/p/26237106)

[2] Tomas Mikolov. Using Neural Networks for Modeling and Representing Natural Languages. COLING (Tutorials) 2014: 3-4

[3] Daoud Clarke. A Context-Theoretic Framework for Compositionality in Distributional Semantics. Computational Linguistics 38(1): 41-71 (2012)

[4] Ian Goodfellow, Yoshua Bengio and Aaron Courville. Deep Learning. Book. The MIT press.2016.

[5] Pengfei Wang, Jiafeng Guo, Yanyan Lan, Jun Xu, Shengxian Wan, Xueqi Cheng. Learning Hierarchical Representation Model for NextBasket Recommendation. SIGIR 2015: 403-412

[6] Wayne Xin Zhao, Sui Li, Yulan He, Edward Y. Chang, Ji-Rong Wen, Xiaoming Li. Connecting Social Media to E-Commerce: Cold-Start Product Recommendation Using Microblogging Information. IEEE Trans. Knowl. Data Eng. 28(5): 1147-1159 (2016)

[7] Ningnan Zhou Wayne Xin Zhao, Xiao Zhang, Ji-Rong Wen, Shan Wang.A General Multi-Context Embedding Model For Mining Human Trajectory Data. IEEE Trans. Knowl. Data Eng. :Online first, 2016.

[8] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry P. Heck. Learning deep structured semantic models for web search using clickthrough data. CIKM 2013: 2333-2338

[9] Ali Mamdouh Elkahky, Yang Song, Xiaodong He. A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems. WWW 2015: 278-288

[10] Ruslan Salakhutdinov, Andriy Mnih, Geoffrey E. Hinton. Restricted Boltzmann machines for collaborative filtering. ICML 2007: 791-798

[11] Ruslan Salakhutdinov, Andriy Mnih. Probabilistic Matrix Factorization. NIPS 2007: 1257-1264

[12] Yin Zheng, Bangsheng Tang, Wenkui Ding, Hanning Zhou. A Neural Autoregressive Approach to Collaborative Filtering. CoRR abs/1605.09477 (2016)

[13] Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester. Collaborative Denoising Auto-Encoders for Top-N Recommender Systems. WSDM 2016: 153-162

[14]Devooght R, Bersini H. Collaborative filtering with recurrent neural networks[J]. arXiv preprint arXiv:1608.07400, 2016.

[15]He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017: 173-182.

[16] Hao Wang, Naiyan Wang, Dit-Yan Yeung. Collaborative Deep Learning for Recommender Systems. KDD 2015: 1235-1244

[17] Aäron Van Den Oord, Sander Dieleman, Benjamin Schrauwen. Deep content-based music recommendation. NIPS 2013: 2643-2651

[18] Xinxi Wang, Ye Wang. Improving Content-based and Hybrid Music Recommendation using Deep Learning. ACM Multimedia 2014: 627-636

[19] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. Session-based Recommendations with Recurrent Neural Networks. CoRR abs/1511.06939 (2015)

[20] Alexandre de Brébisson, Étienne Simon, Alex Auvolat, Pascal Vincent, Yoshua Bengio. Artificial Neural Networks Applied to Taxi Destination Prediction. DC@PKDD/ECML 2015

[21] Cheng Yang, Maosong Sun, Wayne Xin Zhao, Zhiyuan Liu. A Neural Network Approach to Joint Modeling Social Networks and Mobile Trajectories. arXiv:1606.08154 (2016)

[22] Session-based recommendations with recurrent neural networks. (ICLR 2016)

[23] Parallel Recurrent Neural Network Architectures for Feature-rich Session-based
Recommendations. (RecSys 2016)

[24] Incorporating Dwell Time in Session-Based Recommendatons with Recurrent Neural Networks. (RecSys 2017)

[25] Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks. (RecSys 2017)

[26] When Recurrent Neural Networks meet the Neighborhood for Session-Based
Recommendation. (RecSys 2017)

[27] Improved Recurrent Neural Networks for Session-based Recommendations. (DLRS 2016)

[28] Li J, Ren P, Chen Z, et al. Neural attentive session-based recommendation[C]//Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017: 1419-1428.