#!/usr/bin/env python
# coding: utf-8

# In[6]:


from transformers import SiglipVisionModel, SiglipVisionConfig, Trainer, TrainingArguments
from torch.nn import functional as F
from torch import nn
from einops import rearrange
import torch


# In[7]:


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        # Triplet Loss
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        return loss

class SiglipEecoder(nn.Module):
    def __init__(self, config, device, margin):
        super().__init__()
        self.model = SiglipVisionModel(config=config).to(device)
        self.device = device
        self.lossfn = TripletLoss(margin)

    def get_embedding(self, data):
        data = data.to(self.device)
        embs = self.model(data, return_dict=True).pooler_output
        return embs

    def forward(self, anchor, pos, neg):
        data = torch.cat([anchor, pos, neg], dim=0).to(self.device)
        embs = self.model(data, return_dict=True).pooler_output
        anchor_emb, pos_emb, neg_emb = rearrange(embs, "(n b) d -> n b d", n=3).unbind(0)
        if self.training:
            loss = self.lossfn(anchor_emb, pos_emb, neg_emb)
            return {"loss":loss}
        else:
            return {"anchor_emb":anchor_emb, "pos_emb":pos_emb, "neg_emb":neg_emb}


# In[8]:


from Utils.FontData import *
from torchvision import transforms
from torch.utils.data import DataLoader

transformer = transforms.Compose([transforms.ToTensor()])

def collate_fn(batch):
    anchor, pos, neg = zip(*batch)
    anchor = torch.stack([transformer(img) for img in anchor])
    pos = torch.stack([transformer(img) for img in pos])
    neg = torch.stack([transformer(img) for img in neg])
    return {"anchor":anchor, "pos":pos, "neg":neg}

fonts_root = "/home/qba/Data/Project/DeepLearning/FontDream/data/font/中文"
char_set_path = "/home/qba/Data/Project/DeepLearning/FontDream/data/common-char-level-1.txt"
data = FontsDataset(fonts_root, char_set_path,(64,64))
dataloader = DataLoader(data, batch_size=4, shuffle=True, collate_fn=collate_fn)

x = next(iter(dataloader))


# In[9]:


config = SiglipVisionConfig(
    image_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=128,
    intermediate_size=256,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_channels=1)
model = SiglipEecoder(config, "cuda", margin=0.2)

training_args = TrainingArguments(learning_rate=3e-5,
                                  weight_decay=0.01,
                                  num_train_epochs=5,
                                  logging_steps=500,
                                  save_total_limit=3,
                                  report_to="tensorboard",
                                  output_dir="./results",
                                  per_device_train_batch_size=4,
                                  metric_for_best_model="loss"
                                  )
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=data,
                  data_collator=collate_fn
                  )


# In[10]:


trainer.train()


# In[ ]:




