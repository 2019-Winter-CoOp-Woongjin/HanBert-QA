#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, PreTrainedModel

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels # 2

        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels) # (768,2)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # start position : (batch_size, 1)
        # end position : (batch_size, 1)
        # output[0]: The last hidden-state is the first element of the output tuple
        # output[2]: Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) 
        # \ of shape (batch_size, sequence_length, hidden_size).
        
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        
        # train
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

