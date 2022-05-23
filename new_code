from fastNLP.io import ChnSentiCorpLoader
from fastNLP.io import ChnSentiCorpPipe
from fastNLP.embeddings import BertEmbedding
from fastNLP import Tester
from fastNLP import AccuracyMetric
import torch
import torch.nn as nn
from fastNLP.modules import LSTM
from fastNLP import RandomSampler
from fastNLP import DataSetIter
from fastNLP import Trainer
from fastNLP.core.utils import _move_dict_value_to_device
import matplotlib.pyplot as plt
from fastNLP import Tester
from fastNLP import AccuracyMetric

# 读取数据集，分为训练集、验证集、测试集
loader = ChnSentiCorpLoader()
data_bundle = loader.load(
    '/home/chen/Documents/AIProject/LiXuan/data/chn_senti_corp')

# 数据预处理，对文本进行分字，然后将其与标签转化为字典
pipe = ChnSentiCorpPipe()
data_bundle = pipe.process(data_bundle)
traindata = data_bundle.get_dataset('train')
devdata = data_bundle.get_dataset('dev')
testdata = data_bundle.get_dataset('test')

device = 1 if torch.cuda.is_available() else 'cpu'
sampler = RandomSampler()

# 加载预中文训练BERT
char_vocab = data_bundle.get_vocab('chars')
bert_embed = BertEmbedding(char_vocab,
                           model_dir_or_name=r'/home/chen/Documents/AIProject/LiXuan/model/pretrain_model/bert-chinese-wwm',
                           auto_truncate=True, requires_grad=True)


# 构建BERTBiLSTM模型
class BERTBiLSTM(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=400, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = embed
        self.lstm = LSTM(self.embed.embedding_dim, hidden_size=hidden_size // 2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, chars, seq_len):
        chars = self.embed(chars)
        outputs, _ = self.lstm(chars, seq_len)
        outputs = self.dropout_layer(outputs)
        outputs, _ = torch.max(outputs, dim=1)
        outputs = self.fc(outputs)
        return {'pred': outputs}


# 实例化模型
my_Model = BERTBiLSTM(bert_embed, len(data_bundle.get_vocab('target')))
metric = AccuracyMetric()


# 训练过程
class myTrain():
    def __init__(self, epoch, batch_size, model, train_data, dev_data):
        self.epoch = epoch
        self.batch_size = batch_size
        self.model = model.to(device='cuda:1' if torch.cuda.is_available() else 'cpu')
        self.train_data = train_data
        self.dev_data = dev_data

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        lossfunc = torch.nn.CrossEntropyLoss()
        train_sampler = RandomSampler()
        train_batch = DataSetIter(batch_size=self.batch_size, dataset=self.train_data, sampler=train_sampler)
        print("-" * 5 + "start training" + "-" * 5)
        printloss = []
        tag = True
        for i in range(self.epoch):
            loss_list = []
            for batch_x, batch_y in train_batch:
                _move_dict_value_to_device(batch_x, batch_y, device=torch.device('cuda:1'))
                optimizer.zero_grad()
                output = self.model(batch_x['chars'], batch_x['seq_len'])
                loss = lossfunc(output['pred'], batch_y['target'])
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                if tag:
                    printloss.append(loss)
                    tag = False
            sumloss = sum(loss_list) / len(loss_list)
            printloss.append(sumloss)


# 开始训练
my_train = myTrain(epoch=10, batch_size=6, model=my_Model, train_data=traindata, dev_data=devdata)
my_train.train()

# 保存模型
torch.save(my_Model, r'/home/chen/Documents/AIProject/LiXuan/model/save_model/BERTBiLSTM.pt')

# 测试模型
tester = Tester(data=data_bundle.get_dataset('test'), model=my_Model,
                metrics=metric, batch_size=6, device=device)
tester.test()
