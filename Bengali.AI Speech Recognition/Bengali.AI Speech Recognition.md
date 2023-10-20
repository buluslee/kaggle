# kaggle竞赛Bengali.AI Speech Recognition比赛 首战 金牌

**比赛链接** 

<https://www.kaggle.com/competitions/kaggle-llm-science-exam/overview>

## 一、比赛背景


**比赛目标**

本次比赛的目的是从未分发的录音中识别孟加拉语语音。您将构建一个基于第一个大规模众包 （MaCro） 孟加拉语语音数据集训练的模型，其中包含来自印度和孟加拉国的 ~1,200 人的 24,000 小时数据。测试集包含来自训练中不存在的 17 个不同域的样本。

您的努力可以使用第一个孟加拉语分发外语音识别数据集来改进孟加拉语语音识别。此外，您提交的内容将是孟加拉语的首批开源语音识别方法之一。

**上下文**

孟加拉语是世界上使用最多的语言之一，全球约有340.74亿母语和第二语言使用者。随之而来的是方言和韵律特征（声音组合）的多样性。例如，孟加拉语的穆斯林宗教布道通常以与常规讲话明显不同的节奏和音调进行。即使对于商用语音识别方法，这种“转变”也可能具有挑战性（孟加拉语的谷歌语音API对于孟加拉语宗教布道的单词错误率为<>%）。

目前孟加拉语没有强大的开源语音识别模型，尽管你的数据科学技能肯定可以帮助改变这一点。特别是，分布外泛化是一个常见的机器学习问题。当测试和训练数据相似时，它们是分布中的。为了解释孟加拉语的多样性，本次竞赛的数据故意不分发，挑战是改善结果。

竞赛主办方 Bengali.AI 是一项非营利性社区倡议，致力于加速孟加拉语（当地称为孟加拉语）的语言技术研究。Bengali.AI 通过社区驱动的收集活动众包大规模数据集，并通过研究竞赛为其数据集提供众包解决方案。孟加拉.AI语双管齐下的方法的所有结果，包括数据集和训练模型，都是开源的，供公众使用。

您在本次竞赛中的工作可能会对世界上最流行但资源匮乏的语言之一的语音识别改进产生影响。您还可以为解决语音识别的主要挑战之一（分布外泛化）提供急需的推动力。

## 二、评估指标

提交的内容按[平均单词错误率](https://en.wikipedia.org/wiki/Word_error_rate)进行评估，步骤如下：

- 1.WER 是为测试集中的每个实例计算的。
- 2.WER在域内取平均值，由句子中的单词数加权。
- 3.域平均值的（未加权）平均值是最终分数。

此 Python 代码计算指标：

```python
import jiwer  # you may need to install this library

def mean_wer(solution, submission):
    joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
    domain_scores = joined.groupby('domain').apply(
        # note that jiwer.wer computes a weighted average wer by default when given lists of strings
        lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
    )
    return domain_scores.mean()

assert (solution.columns == ['id', 'domain', 'sentence']).all()
assert (submission.columns == ['id',' sentence']).all()
```
## 三、数据集

**数据描述**

数据集描述
比赛数据集包括大约1200小时的孟加拉语演讲录音。您的目标是转录与训练集相关的不分发的语音录音。

请注意，这是一个代码竞赛，其中实际的测试集是隐藏的。在此公共版本中，我们以正确的格式提供了一些示例数据，以帮助您编写解决方案。完整的测试集在近 20 个 MP8000 音频文件中包含大约 3 小时的语音。测试集中的所有文件在一个通道中以 32k 的采样率、48k 的比特率进行编码。测试集注释使用 bnUnicodeNormalizer 进行规范化。

有关数据集的详细信息，请参阅数据集论文：https://arxiv.org/abs/2305.09688


**文件和字段说明**
- train/训练集，包括数千个MP3格式的记录。
- test/测试集包括来自 18 个领域的自发语音记录，其中 17 个域与训练集无关。专用测试集中可能存在不在公共测试集中的域。
- examples/每个测试集域的示例记录。您可能会发现这些示例记录有助于创建对域变化具有鲁棒性的模型。这些是具有代表性的记录，它们都不存在在测试集中。
   - train.csv 训练集的句子标签。
   - id此实例的唯一标识符。对应于 train/ 中的文件。{id}.mp3
   - sentence录音的纯文本转录。您的目标是预测测试集中每个记录的这些句子。
   - split无论是还是.拆分中的注释已经过手动检查和更正，而拆分中的注释仅通过算法进行了清理。样本通常具有比样本更高质量的注释，但其他方面来自相同的分布。
- sample_submission.csv格式正确的示例提交文件。有关更多详细信息，请参阅评估页面。
- 
## 四、比赛思路与实现

**模型选择**

我们最开始使用的语音模型 是whisper，使用的训练数据是比赛方公布的900k条训练集，训练代码和推理代码都是使用的比赛方公布的入门笔记本， 第一次提交的分数是 cv 0.21 LB 0.51。

在后面我们发现公布笔记本使用的模型都是Wav2vec2 CTC model，分数都普遍的高。

最终我们选择的预训练模型是Wav2vec2 CTC model，因为评论区都对官方数据集的质量提出了质疑所以之后我们使用了OOD数据集。

**比赛思路**

训练代码 https://www.kaggle.com/code/takanashihumbert/bengali-sr-wav2vec-v1-bengali-training

推理代码 https://www.kaggle.com/code/renyiwei/single-best-model-0-373?scriptVersionId=146392108

对[@umongsain](https://www.kaggle.com/datasets/umongsain/common-voice-13-bengali-normalized)筛选的常见语音数据集进行微调  公共LB：0.439 -> 0.428

在Openslr 37上进行微调导致：公共LB0.428 -> 0.414

将奥斯卡语料库合并到 ngram 中

通过音频进行数据增强。

此[存储库](https://github.com/xashru/punctuation-restoration)中的 XLM-Roberta-大型配置。

用于解码超参数的 Optuna 搜索，如本[笔记本](https://www.kaggle.com/code/royalacecat/lb-0-442-the-best-decoding-parameters)所示。

### 训练的参数

**环境配置为**

    GPU 40G显存A100
    CUDA 11.7
    内存 64G
    python 3.8.5
    
    使用该代码训练模型最低GPU显存为32G


**训练参数**
```python
model.freeze_feature_extractor()
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    group_by_length=False,
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=12,
    evaluation_strategy="steps",
    save_strategy="steps",
    max_steps=20000, # you can change to "num_train_epochs"
    fp16=True,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=500,
    learning_rate=1e-5,
    warmup_steps=2000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    prediction_loss_only=False,
    auto_find_batch_size=True,
    report_to="none"
)
```

- 使用gradient_accumlation_steps(模拟更大的批量大小)
- 使用fp16(加速推理时间)
- greater_is_better为False表示指标越低模型越好
- 使用metric_for_best_model指定评价指标为wer
- 使用cosine学习率(加快模型收敛)
- 冻结模型编码层(减少了训练的权重)


**数据处理**
定义一个数据生成器用来处理加载数据
```python
import torch
import librosa
import os

class BengaliSRTestDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        audio_paths: list[str],
        sampling_rate: int
    ):
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        
    def __len__(self,):
        return len(self.audio_paths)
    
    def __getitem__(self, index: int):
        audio_path = self.audio_paths[index]
        sr = self.sampling_rate
        
        w = librosa.load(audio_path, sr=sr, mono=False)[0]
        return w
```
**评估指标**
评估指标我们使用的是比赛方要求的wer
```python
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = 100*wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
```
**训练结果**

训练代码

```python
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    #compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)
trainer.train()
trainer.save_model(output_dir)
```

训练成绩

![image](https://github.com/buluslee/kaggle/assets/93359778/56f24a08-054b-4182-807e-5473fcd1992d)

看图，训练到8000步时，cv最好为0.19(打印时乘了个100)，训练的时间大约为12小时。

因为本次比赛要求笔记本运行的时间不能超过9小时，所以我们采取了在本地训练好模型，然后上传kaggle平台，直接加载模型进行推理，这样节省了kaggle每周的gpu时间，以及代码运行的时间。

想要获得好的成绩，我们的推理代码一样很重要

### 推理部分

标点修复代码，具体的模型参数和设置根据实际需求和数据集进行了配置。模型基于RoBERTa模型，可以用于处理文本数据中的标点符号修复任务。
```python
from transformers import XLMRobertaModel,XLMRobertaTokenizer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 有了xformers后可能不需要这句

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}

MODELS = {
    'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer, 1024, 'roberta'),
}

import torch.nn as nn
import torch

class DeepPunctuation(nn.Module):
    def __init__(self, model, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
#         self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
#         has to modest the former line a bit to fit with the offline requirement
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained(model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x

roberta_model = '/kaggle/input/xlm-roberta-large'
roberta_pretrained_model = 'xlm-roberta-large'

cuda = True
lstm_dim = -1
use_crf = False
language = 'bn'
#in_file = '/kaggle/input/punctuation-model/punctuation-restoration-master/punctuation-restoration-master/data/test_bn.txt' #测试文件路径
weight_path = '/kaggle/input/last-shot/weights_4_finetuned.pt' #这是权重路径 原项目说这已经是微调后的结果
sequence_length = 256
# out_file = '/kaggle/working/out/test_bn_out.txt'
device = torch.device('cuda' if (cuda and torch.cuda.is_available()) else 'cpu')

if use_crf:
    deep_punctuation = DeepPunctuationCRF(roberta_model, roberta_pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
else:
    deep_punctuation = DeepPunctuation(roberta_model, roberta_pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
deep_punctuation.to(device)
```

之前的代码扩展,此代码段的主要目的是定义了一个函数，它接受原始文本，通过加载的模型进行标点恢复，并返回已恢复标点的文本。这可以用于对文本进行自动标点处理。
```python
from transformers import RobertaModel,RobertaTokenizer


#'roberta-large': (RobertaModel, RobertaTokenizer, 1024, 'roberta'),

tokenizer = MODELS[roberta_pretrained_model][1].from_pretrained(roberta_model)
token_style = MODELS[roberta_pretrained_model][3]
deep_punctuation.load_state_dict(torch.load(weight_path),strict=False)
deep_punctuation.eval()

import re
def inference_string(text:str):

    #print('Original text:')
    #print(text)
    text = re.sub(r"[,:\-–.!;?।\"“”]", '', text)
    words_original_case = text.split()
    words = text.lower().split()

    word_pos = 0
    sequence_len = sequence_length
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
    if language != 'en':
        punctuation_map[2] = '।'

    while word_pos < len(words):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0]

        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

        x = torch.tensor(x).reshape(1,-1)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask).reshape(1,-1)
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)
        
        
        with torch.no_grad():
            if use_crf:
                y = torch.zeros(x.shape[0])
                y_predict = deep_punctuation(x, attn_mask, y)
                y_predict = y_predict.view(-1)
            else:
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
    #print('Punctuated text')
    #print(result)
    return result
```
通过模型执行批量数据上的推理，并将推理结果进行标点恢复，最后将标点修复的句子添加到pred_sentence_list中.
```python
with torch.no_grad():
    for batch in tqdm(test_loader):
        x = batch["input_values"]
#         print(x.dtype)
        x = x.to(device, non_blocking=True)
#         denoised_audio = sampling(net, x)
#         denoised_audio = denoised_audio.squeeze(1).squeeze(1)
        
#         print(denoised_audio.dtype)
    
        with torch.cuda.amp.autocast(True):
            y = model(x).logits
        y = y.detach().cpu()
        
        for l in y:  
            sentence = processor_with_lm.decode(l.numpy(), beam_width=1024, alpha = 0.4, beta = 0.05040179005591594).text
            sentence = inference_string(sentence)
            pred_sentence_list.append(sentence)
```
定义了一个名为postprocess的函数，用于对文本句子进行后处理。主要目的是确保生成的句子具有适当的标点符号和格式。

```python
bnorm = Normalizer()

def postprocess(sentence):
    period_set = set([".", "?", "!", "।"])
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    try:
        if sentence[-1] not in period_set:
            sentence+="।"
    except:
        # print(sentence)
        sentence = "।"
    return sentence
```

最后生成csv提交kaggle平台

```python

pp_pred_sentence_list = [
    postprocess(s) for s in tqdm(pred_sentence_list)]

test["sentence"] = pp_pred_sentence_list

test.to_csv("submission.csv", index=False)

print(test.head())
```
## 五、总结
我们一共是2队参加比赛
    一队的最高成绩为 公榜0.372 私榜0.449 排名11 金牌
   ![image](https://github.com/buluslee/kaggle/assets/93359778/d4cc6220-e727-4f3d-951e-5cf8f66b5257)
    二队由于设备与时间问题，所以没有获得良好的成绩
