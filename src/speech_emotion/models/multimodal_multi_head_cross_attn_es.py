import pandas as pd 
from datasets import DatasetDict, features
from datasets import load_dataset, Audio, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Model, Wav2Vec2BertForSequenceClassification, Wav2Vec2BertModel
from transformers import AutoTokenizer, BertModel
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()

_HIDDEN_STATES_START_POSITION = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("f1")
metric_name = "f_macro"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(classification_report(labels, predictions, digits=4))
    precision = metric1.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f_score = metric3.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    f_macro = metric3.compute(predictions=predictions, references=labels, average='macro')["f1"]
    return {"precision": precision, "recall": recall, "f_score": f_score, "f_macro": f_macro}


def preprocess_function(examples, feature_extractor):
    audio_arrayas = [x["array"] for x in examples["audio"]]
    # feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    inputs = feature_extractor(
        audio_arrayas, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    

    return inputs
        
# Head_size = 768
# embedding = 768 
        
class MultiModalCrossAttention(nn.Module):
    def __init__(self, dim, dropout: int = 0.3, qk_norm: bool = True):
        super(MultiModalCrossAttention, self).__init__()
        self.dim = dim
        self.qk_norm = qk_norm

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Embedding audio -> Embedding bert 
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # Embedding bert -> Embedding audio 
        self.Wq_reverse = nn.Linear(dim, dim)
        self.Wk_reverse = nn.Linear(dim, dim)
        self.Wv_reverse = nn.Linear(dim, dim)

        self.linear_out = nn.Linear(2 * dim, dim)

    def forward(self, hidden_bert, hidden_wav2vec): 
        Qcross = self.Wq(hidden_bert)
        Kcross = self.Wk(hidden_wav2vec)
        Vcross = self.Wv(hidden_wav2vec)

        if self.qk_norm:
            # Normalize Qcross and Kcross
            Qcross = self.norm(Qcross)
            Kcross = self.norm(Kcross)
        else:
            pass

        with torch.backends.cuda.sdp_kernel(enable_math=True):
            attn_weights = F.scaled_dot_product_attention(Qcross, Kcross, Vcross)
            # dropout
            attn_weights = self.dropout(attn_weights)

        Hcross = attn_weights + Vcross

        Qcross_reverse = self.Wq_reverse(hidden_wav2vec)
        Kcross_reverse = self.Wk_reverse(hidden_bert)
        Vcross_reverse = self.Wv_reverse(hidden_bert)

        with torch.backends.cuda.sdp_kernel(enable_math=True):
            attn_weights_reverse = F.scaled_dot_product_attention(
                Qcross_reverse, Kcross_reverse, Vcross_reverse
            )
            # dropout
            attn_weights_reverse = self.dropout(attn_weights_reverse)

        Hcross_reverse = attn_weights_reverse + Vcross_reverse

        # Concatenate the results
        output = torch.cat((Hcross, Hcross_reverse), dim=-1)
        output = self.linear_out(output)
        return output
        
        
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([MultiModalCrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd*num_heads, n_embd)
        self.dropout = nn.Dropout(0.3)

    def forward(self, hidden_bert, hidden_wav2vec):
        out = torch.cat([h.forward(hidden_bert, hidden_wav2vec) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_extra_dims):
        super().__init__()
        # total_dims = config.hidden_size+num_extra_dims
        # total_dims = 768 + num_extra_dims 
        total_dims = 768
        self.dense = nn.Linear(total_dims, total_dims)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)
        self.layernorm = nn.LayerNorm(total_dims)

    def forward(self, features, **kwargs):
        # x = self.layernorm(features) 
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class CustomAudioClassificationAttn(Wav2Vec2BertForSequenceClassification):    
    
    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.attn_model = MultiModalCrossAttention(768)
        self.mult_attn_model = MultiHeadAttention(8, 768, 768)
        
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.classifier = ClassificationHead(config, num_extra_dims)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(        
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        sentence_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            
        # pooled_output = torch.cat((pooled_output, sentence_embedding), dim=1)  
        # attn_output = self.attn_model.forward(sentence_embedding, pooled_output)
        attn_output = self.mult_attn_model.forward(sentence_embedding, pooled_output)
        
        logits = self.classifier(attn_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    