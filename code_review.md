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

