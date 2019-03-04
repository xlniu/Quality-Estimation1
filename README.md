# Quality-Estimation1
机器翻译子任务-翻译质量评价<br>

## 简介
翻译质量评价（Quality Estimation,QE）是机器翻译领域中的一个子任务，大致可分为 Sentence-level QE，Word-level QE，Phrase-level QE，详情可参考WMT(workshop machine translation)比赛官网 http://www.statmt.org/wmt17/quality-estimation-task.html 。本项目针对 Sentence-level QE，试图复现论文[“Bilingual Expert” Can Find Translation Errors](https://arxiv.org/pdf/1807.09433.pdf)的实验结果。上述论文的开源代码如下：https://github.com/lovecambi/qebrain ，本人受服务器驱动限制，装不了高版本的tensorflow，基于transformer开源代码https://github.com/Kyubyong/transformer 重写了论文中的专家模型和qe模型。由于 wmt18-qe 的测试集标签没有公布，本项目仅在 wmt17-qe 数据集上进行实验。

## 实验需要的包
tensorflow == 1.2.0;<br>
