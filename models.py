import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertForSequenceClassification

class investorConferenceAnalyzer(nn.Module):
    """Sentiment analysis classifier for investor conference context.
    
    Based on ALBERT.

    Args:
        pce_model (str): Pretrained model loaded for ALBERT.
        num_labels: (int):

    Return:
        log_p: (torch.Tensor): Predicted log probability of each labels.
    """
    def __init__(self, pce_model, num_labels):
        super(investorConferenceAnalyzer, self).__init__()
        self.classifier = AlbertForSequenceClassification.from_pretrained(pce_model, num_labels=num_labels)
    
    def forward(self, input_idxs, token_type_idxs, attention_masks):
        logits = self.classifier(input_ids=input_idxs,
                                 attention_mask=attention_masks,
                                 token_type_ids=token_type_idxs)[0]
        log_p = F.log_softmax(logits, dim=-1)                  
        
        return log_p
