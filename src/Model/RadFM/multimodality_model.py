from torch import nn
from transformers.models.llama import LlamaForCausalLM
from .my_embedding_layer import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import numpy as np


class MultiLLaMAForCausalLM(nn.Module):

    def __init__(self,
                 lang_model_path,
                 lora_r=64,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 lora_bias="none",
                 torch_dtype=torch.bfloat16,
                 bits=4):
        super(MultiLLaMAForCausalLM, self).__init__()
        self.lang_model = LlamaForCausalLM.from_pretrained(
            lang_model_path,
            # torch_dtype=torch_dtype,
            # load_in_8_bit=True,
        )

        # self.lang_model.gradient_checkpointing_enable()

        # if hasattr(self.lang_model, "enable_input_require_grads"):
        #     self.lang_model.enable_input_require_grads()
        # else:

        # def make_inputs_require_grad(module, input, output):
        #     output.requires_grad_(True)

        # self.lang_model.get_input_embeddings().register_forward_hook(
        #     make_inputs_require_grad)
        #
        # self.lang_model.enable_input_require_grads()

        # self.lang_model.requires_grad_(False)

        # self.lang_model.requires_grad_(False)
        self.embedding_layer = MyEmbedding()

        # self.embedding_layer.requires_grad_(False)
        self.embedding_layer.weight = self.lang_model.get_input_embeddings(
        ).weight
        self.hidden_dim = 5120
        self.voc_size = 32000

    def forward(self, lang_x, vision_x, attention_mask, labels, loss_reweight,
                key_words_query):
        if labels.shape == lang_x.shape:
            self.embedding_layer.flag = 'Text'
            # lang_x = lang_x.to(vision_x.dtype)
            # lang_x = lang_x + torch.zeros(1, dtype=lang_x.dtype, device=lang_x.device, requires_grad=True)
            # vision_x = vision_x + torch.zeros(1, dtype=vision_x.dtype, device=vision_x.device, requires_grad=True)
            # input_embedding = checkpoint(self.embedding_layer, lang_x, vision_x)

            print('lang_x', lang_x.dtype, lang_x.shape)
            print('vision_x', vision_x.dtype, vision_x.shape)

            input_embedding, loss_match = self.embedding_layer(
                lang_x, vision_x, key_words_query)  # ,loss_matching

            print('after embedding layer: ')
            print('input_embedding: ', input_embedding.shape,
                  input_embedding.dtype)
            print('attn mask: ', attention_mask.shape, attention_mask.dtype)
            print('labels: ', labels.shape, labels.dtype)

            output = self.lang_model(inputs_embeds=input_embedding,
                                     attention_mask=attention_mask,
                                     labels=labels)
            logits = output['logits']
            print('*' * 100)
            print('logits: ', logits.dtype, logits.shape)
            print('loss_reweight: ', loss_reweight.dtype, loss_reweight.shape)
            print('*' * 100)

            loss_reg = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(
                    shift_logits.device)
                print('*' * 100)
                print('shift_logits: ', shift_logits.dtype, shift_logits.shape)
                print('shift_labels: ', shift_labels.dtype, shift_labels.shape)
                print('*' * 100)

                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight *
                                     loss_reg) / torch.sum(shift_loss_reweight)
            loss = loss_reg

            print('*' * 100)
            print('inside multimodality model:')
            print('loss: ', loss)
            print('*' * 100)

            if loss_match != None:
                loss = 0.8 * loss + 0.2 * loss_match

            logits = output['logits'][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            Acc = torch.sum(
                torch.all(torch.logical_or(predictions == labels,
                                           labels == -100),
                          dim=-1))
            Accuracy = Acc / total

            return dict(
                # loss_reg = loss_reg,
                # loss_matching = loss_matching,
                logits=Accuracy,
                loss=output['loss'],
            )
        ### useless for now ignore the folowing codes ###
        # if labels.shape == vision_x.shape:
        #    self.embedding_layer.flag = 'Seg'
        #    input_embedding = self.embedding_layer(lang_x, vision_x)

    def generate(self, lang_x, vision_x):
        self.embedding_layer.flag = 'Text'
        with torch.no_grad():
            input_embedding, _ = self.embedding_layer(lang_x, vision_x)
            generation = self.lang_model.generate(
                inputs_embeds=input_embedding, max_new_tokens=200, top_k=50)
        return generation
