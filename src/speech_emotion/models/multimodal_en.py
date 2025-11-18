import pandas as pd
from datasets import DatasetDict, features
from datasets import load_from_disk, Audio, Dataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertForSequenceClassification,
    Wav2Vec2BertModel,
)
from transformers import AutoTokenizer, BertModel
import evaluate
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

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
    precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f_score = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    f_macro = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"precision": precision, "recall": recall, "f_score": f_score, "f_macro": f_macro}


def preprocess_function(examples, feature_extractor):
    audio_arrayas = [x["array"] for x in examples["audio"]]

    inputs = feature_extractor(
        audio_arrayas,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True,
    )
    return inputs


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_extra_dims, t_cls):
        super().__init__()
        if t_cls == "mean":
            total_dims = 768
        elif t_cls == "concat":
            total_dims = 768 + num_extra_dims
        self.dense = nn.Linear(total_dims, total_dims)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)
        self.layernorm = nn.LayerNorm(total_dims)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomAudioClassificationMean(Wav2Vec2BertForSequenceClassification):
    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1
        self.classifier = ClassificationHead(config, num_extra_dims, "mean")
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
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

        pooled_output = torch.mean(torch.stack((pooled_output, sentence_embedding), 1), 1)
        logits = self.classifier(pooled_output)

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


class CustomAudioClassificationConcat(Wav2Vec2BertForSequenceClassification):
    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1
        self.classifier = ClassificationHead(config, num_extra_dims, "concat")
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
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

        # concat audio + texto
        pooled_output = torch.cat((pooled_output, sentence_embedding), dim=1)
        logits = self.classifier(pooled_output)

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
