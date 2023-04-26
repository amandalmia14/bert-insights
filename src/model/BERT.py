import torch
import torch.nn as nn

from transformers import BertModel


class BERT(nn.Module):

    def __init__(self, model_path, hidden_size):
        super(BERT, self).__init__()
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.bert_model = BertModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)

        self.label_num = 2

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, self.label_num)

    def forward(self, bert_ids, bert_mask):
        outputs = self.bert_model(input_ids=bert_ids, attention_mask=bert_mask)
        pooler_output = outputs['pooler_output']

        x = self.dense(pooler_output)
        x = torch.tanh(x)
        x = self.dropout(x)
        fc_output = self.fc(x)

        return fc_output, outputs
