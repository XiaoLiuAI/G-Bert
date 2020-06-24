# 预训练

## 数据

### 数据格式
预训练脚本`run_pretraining.py`读取数据文件(`data-multi-visit.pkl`, `data-single-visit.pkl`)和id文件
(`train-id.txt`，`eval-id.txt`，`test-id.txt`). id对应的是患者，数据文件是pandas的dataframe，里面的
`SUBJECT_ID`列对应患者id，一行为一次就诊，数据列包括药品，诊断和操作代码. `data-multi-visit.pkl`包含的是
一个人有多次就诊的数据，`data-single-visit.pkl`包含的是一个人只有一次就诊的数据.

预训练数据生成`DataSet`返回元素为
```python
X = [
['cls']+['diagnosis code'],
['cls']+['drug code']
]

y1 = [0, 1]  # one-hot 诊断
y2 = [0, 1]  # one-hot 药品
```
`X`是进行了替换`[MASK]`的,这一次没有使用操作代码


## 数据预处理

## 模型
`predictive_models.py`

# 模型
1. bert_models
2. graph_models
3. predictive_models

## 类型继承
- bert_models.py
    - PreTrainedBertModel
    - BERT(PreTrainedBertModel)
        - embedding = FuseEmbeddings
    
- predictive_models.py
    - GBERT_Pretrain(PreTrainedBertModel)
        - bert = BERT
    - GBERT_Predict(PreTrainedBertModel)
    
- graph_models.py
    - OntologyEmbedding
        - graph generation
        - g = GATConv
    - ConcatEmbeddings
        - rx_embedding = OntologyEmbedding
        - dx_embedding = OntologyEmbedding
    - FuseEmbeddings
        - ontology_embedding = ConcatEmbeddings 
    
*Question 1: 是不是每次embedding取数的时候都会启动GATConv？*

Answer: 是

*Question 2: pytorch_geometric包和GATConv的实现细节以及代码逻辑*

pytorch_geometric 实现了 message passing 的机制. 原理是把neighbor的信息aggregate到中心node上.
基础类MessagePassing实现了整个meta算法,使用者自己定义message 和 update 两个函数.

`message` 接受两个node和一个edge(即三元组),定义neighbor传导给node的信息.

`update` 定义怎么根据node本身以及传导的信息来更新node. 返回更新值,但是并不对embedding本身直接进行更新.
