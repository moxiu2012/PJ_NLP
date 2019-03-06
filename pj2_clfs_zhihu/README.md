 ### 知乎话题多标签分类

 #### 0、任务类型
 * 多标签分类
 * 使用pytorch训练话题多标签分类模型

 #### 1、数据集
 * 知乎看山杯的话题多标签数据，采样了一部分数据

 #### 2、项目结构：
 * config.py文件主要包含参数：文件路径、训练参数、模型参数
 * sample_data.py文件对数据集进行采样。
 * pre_data.py文件主要负责数据清洗、处理成模型输入形式、简化embedding矩阵等。
 * dataset.py文件主要是将数据处理成批量输入的迭代器。
 * model.py文件构建模型
 * train.py文件负责模型训练和验证集的验证过程。
 * test.py文件负责测试集的预测过程。

 #### 3、运行顺序： 
 * sample_data.py -> pre_data.py -> train.py -> test.py
