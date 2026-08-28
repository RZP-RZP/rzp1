# 冲压板材成形数据集说明
## 数据集说明
本数据集参考公开冲压工艺学术数据集的数据分布构造仿真样例数据集，用于课程设计Demo开发。
完整公开学术数据集参考冲压成形相关SCI论文附属数据，本仓库使用仿真样例保证链接不会失效。

## 字段说明
### 输入工艺特征
- sheet_thickness：板材厚度(mm)
- blank_holder_force：压边力(kN)
- stamping_speed：冲压速度(mm/s)
- friction_coeff：摩擦系数
- die_gap：模具间隙(mm)

### 输出标签
- crack：是否开裂 0=未开裂，1=开裂（分类标签）
- wrinkle：是否起皱 0=无起皱，1=起皱（分类标签）
- springback：回弹量mm，回归预测标签

## 目录说明
1. raw：原始样例数据集 stamping_raw_sample.csv
2. processed：预处理之后数据集，包含训练集 stamping_train.csv、测试集 stamping_test.csv

## 使用注意
数据仅用于课程设计学习，不用于工业生产。
