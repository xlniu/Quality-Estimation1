# Quality-Estimation1
机器翻译子任务-翻译质量评价-复现 WMT2018 阿里论文结果<br>

## 简介
翻译质量评价（Quality Estimation,QE）是机器翻译领域中的一个子任务，大致可分为 Sentence-level QE，Word-level QE，Phrase-level QE，详情可参考WMT(workshop machine translation)比赛官网 http://www.statmt.org/wmt17/quality-estimation-task.html 。 __本项目针对 Sentence-level QE，试图复现论文 [“Bilingual Expert” Can Find Translation Errors](https://arxiv.org/pdf/1807.09433.pdf) 的实验结果。__ 上述论文的开源代码如下：https://github.com/lovecambi/qebrain ，本人受服务器驱动限制，装不了高版本的tensorflow， __基于transformer开源代码 https://github.com/Kyubyong/transformer 重写了论文中的专家模型和qe模型。__ 由于 wmt18-qe 的测试集标签没有公布，本项目仅在 wmt17-qe 数据集上进行实验。<br>
<br>
## 实验环境
python3<br>
tensorflow == 1.2.0<br>
<br>
## 实验步骤
**1、准备数据**<br>
用于训练专家模型的数据,其数据来源是 WMT17 Translation task、WMT17 qe task中sentence-level task中训练集的数据（src+pe）。数据统计信息见下表:

|Dataset|Sentences|
|:---|:--------|
|Europarl v7|1,920,209|
|Common Crawl corpus|2,399,123|
|News Commentary v12|268,328|
|Rapid corpus of EU press releases|1,314,689|
|UFAL Medical Corpus|2,660,314|
|Khresmoi development data|500|
|Sentence-level QE en-de smt|23,000|
|Sentence-level QE de-en smt|25,000|
|total|8,611,163|
|filter|8,449,133|

注：过滤后的数据总量为8,449,133<br>

**2、数据预处理**<br>
**tokenize**(./preprocess/token.sh);<br>
**lower**(./preprocess/lower.sh);<br>
**filter**(./preprocess/data_filter_merge.ipynb, we filtered the source and target sentence with length <= 70 and the length ratio between 1/3 to 3);<br>
**merge**(./preprocess/data_filter_merge.ipynb，将所有数据集按照语言分别合并);<br>

**3、词表生成**<br>
分别生成源端和目标端的词表，生成后的词表按照词频排序，代码见：./prepro.py，运行代码前要先设置词表大小;<br>

**4、预训练专家模型**<br>
设置exp_hyperparams.py中的参数，运行expert_model.py;<br>

**5、联合训练专家模型和qe模型**<br>
设置qe_hyperparams.py中的参数，运行qe_model.py;<br>
使用 Sentence-level QE en-de smt 训练en-de模型，使用 Sentence-level QE de-en smt 训练de-en模型;<br>
<br>
## 实验结果
|Data|Pearson’s|vocab_size|batch|steps|
|:---|:---|:---|:---|:---|
|test 2017 en-de|0.5974|30k|128|500k|
|state of the art(Single)|0.6837|120k|-|-|
|test 2017 de-en|0.6745|30k|128|410k|
|state of the art(Single)|0.7099|120k|-|-|

注：<br>
1、state of the art 参考论文：[“Bilingual Expert” Can Find Translation Errors](https://arxiv.org/pdf/1807.09433.pdf) ；<br>
2、表中的batch和steps都是说的训练专家模型的情况，steps指训练了多少步（每一步是一个batch）；<br>
3、从我的观察来看，基本上是专家模型被训练的steps越多，后续再联合训练qe模型而得到的结果就越好，但是随着专家模型被训练步数的不断增多（比如达到350k以上），最终的结果会出现波动；<br>
4、训练专家模型和联合训练qe模型的日志文件都放在了 ./nohup 文件夹下；<br>
