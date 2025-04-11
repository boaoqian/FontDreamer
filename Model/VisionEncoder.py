from torchvision import transforms
from transformers import SiglipVisionModel, SiglipVisionConfig
from torch.nn import functional as F
from torch import nn
from einops import rearrange
import torch

DEF_CONFIG = SiglipVisionConfig(
    image_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=128,
    intermediate_size=256,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_channels=1)

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
    def __init__(self, config=DEF_CONFIG, device="cuda", margin=0.2):
        super().__init__()
        self.model = SiglipVisionModel(config=config).to(device)
        self.device = device
        self.lossfn = TripletLoss(margin)
        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize(config.image_size)])


    def get_embedding(self, data):
        data = data.to(self.device)
        embs = self.model(data, return_dict=True).pooler_output
        return embs

    def get_distance(self, img1, img2):
        emb1 = self.get_embedding(self.transformer(img1).to(self.device).unsqueeze(0))
        emb2 = self.get_embedding(self.transformer(img2).to(self.device).unsqueeze(0))
        dist = F.pairwise_distance(emb1, emb2)
        return dist

    def forward(self, anchor, pos, neg):
        data = torch.cat([anchor, pos, neg], dim=0).to(self.device)
        embs = self.model(data, return_dict=True).pooler_output
        anchor_emb, pos_emb, neg_emb = rearrange(embs, "(n b) d -> n b d", n=3).unbind(0)
        if self.training:
            loss = self.lossfn(anchor_emb, pos_emb, neg_emb)
            return {"loss":loss}
        else:
            return {"anchor_emb":anchor_emb, "pos_emb":pos_emb, "neg_emb":neg_emb}

