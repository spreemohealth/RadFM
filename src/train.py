# import tqdm as tqdm
import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Dataset.multi_dataset import multi_dataset, MultidatasetBigrad
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
from datasets import load_metric
from Dataset.multi_dataset_test_for_close import multi_dataset_close
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import LlamaTokenizer
import evaluate
from functools import partial
from Dataset.dataset.internal3d import Internal3DDataset, DfForDlDataset, All_Combi_Dataset

rouge_score = evaluate.load("rouge")

tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/team_blackhole/kawshik/Language_files/")


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    # print(eval_preds.predictions.shape)
    # print(eval_preds.predictions)
    # print(tokenizer)
    # print(tokenizer.additional_special_tokens)
    # preds = tokenizer.batch_decode(eval_preds.predictions)

    preds = tokenizer.batch_decode(
        np.where(eval_preds.predictions == -100,
                 np.zeros(eval_preds.predictions.shape),
                 eval_preds.predictions))

    preds = [
        pred.replace(tokenizer.decode(np.zeros((1))), "") for pred in preds
    ]

    labels = tokenizer.batch_decode(
        np.where(eval_preds.label_ids == -100,
                 np.zeros(eval_preds.label_ids.shape), eval_preds.label_ids))

    labels = [
        label.replace(tokenizer.decode(np.zeros((1))), "") for label in labels
    ]
    # for i,pred in enumerate(preds):
    #     print('*'*100)
    #     print('pred: ',pred)
    #     print('-'*100)
    #     print("labels: ",labels[i].replace(tokenizer.decode(np.zeros((1))),""))
    #     print('-'*100)
    #     input("enter to continue")

    result = rouge_score.compute(predictions=preds,
                                 references=labels,
                                 use_stemmer=True)

    # print(ACCs)
    result = {key: value for key, value in result.items()}

    return result


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/mnt/team_blackhole/kawshik/Language_files/")
    tokenizer_path: str = field(
        default="/mnt/team_blackhole/kawshik/Language_files/",
        metadata={"help": "Path to the tokenizer data."})
    model_ckpt_load_dir: str = field(
        default='/mnt/team_s3_synced/msandora/RadFM/pytorch_model.bin'),
    param_groups_to_train: str = field(default='[lora]')


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    qtype: Optional[str] = field(default=None)
    max_seq: Optional[int] = field(default=1280)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=4)
    output_dir: Optional[str] = field(default="./Results/BLIP_overfit/")
    cache_dir: Optional[str] = field(default=None)
    lr: str = field(default="[2e-4,5e-5]")
    optim: str = field(default="adamw_torch")


@dataclass
class DataCollator_orig(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, attention_masks, labels, loss_reweight, key_words_query = tuple(
            [instance[key] for instance in instances]
            for key in ('vision_x', 'lang_x', 'attention_mask', 'labels',
                        'loss_reweight', 'key_words_query'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks],
                                    dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight],
                                  dim=0)
        # print('lang shapes: ')
        # print(lang_xs.shape, attention_masks.shape, labels.shape,
        #       loss_reweight.shape)

        # target_H = 512
        # target_W = 512
        target_H = 256
        target_W = 256
        target_D = 4
        MAX_D = 0

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [
            torch.nn.functional.interpolate(s,
                                            size=(target_H, target_W,
                                                  target_D)) for s in vision_xs
        ]

        vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs,
                                                    batch_first=True,
                                                    padding_value=0)
        # print('vision shapes: ')
        # print(vision_xs.shape, vision_xs.dtype)
        return dict(lang_x=lang_xs,
                    vision_x=vision_xs,
                    attention_mask=attention_masks,
                    labels=labels,
                    loss_reweight=loss_reweight,
                    key_words_query=key_words_query)


@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, attention_masks, labels, loss_reweight, key_words_query = tuple(
            [instance[key] for instance in instances]
            for key in ('vision_x', 'lang_x', 'attention_mask', 'labels',
                        'loss_reweight', 'key_words_query'))

        max_len = max([len(x_i) for x_i in lang_xs])

        # print('lang_xs: ', lang_xs)

        lang_xs = torch.cat([
            torch.cat([lang_xi,
                       torch.zeros(max_len - len(lang_xi)).long()
                       ]).unsqueeze(0) for lang_xi in lang_xs
        ],
                            dim=0)
        attention_masks = torch.cat([
            torch.cat([am_i, torch.zeros(max_len - len(am_i)).long()
                       ]).unsqueeze(0) for am_i in attention_masks
        ],
                                    dim=0)
        labels = torch.cat([
            torch.cat([label_i,
                       torch.zeros(max_len - len(label_i)).long()
                       ]).unsqueeze(0) for label_i in labels
        ],
                           dim=0)
        loss_reweight = torch.cat([
            torch.cat([lr_i, torch.zeros(max_len - len(lr_i))]).unsqueeze(0)
            for lr_i in loss_reweight
        ],
                                  dim=0)
        # print('lang shapes: ')
        # print(lang_xs.shape, attention_masks.shape, labels.shape,
        #       loss_reweight.shape)

        # target_H = 512
        # target_W = 512
        target_H = 256
        target_W = 256
        target_D = 4
        MAX_D = 0

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [
            torch.nn.functional.interpolate(s,
                                            size=(target_H, target_W,
                                                  target_D)) for s in vision_xs
        ]

        vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs,
                                                    batch_first=True,
                                                    padding_value=0)
        # print('vision shapes: ')
        # print(vision_xs.shape, vision_xs.dtype)
        return dict(lang_x=lang_xs,
                    vision_x=vision_xs,
                    attention_mask=attention_masks,
                    labels=labels,
                    loss_reweight=loss_reweight,
                    key_words_query=key_words_query)


def get_preds(logits, labels):
    out = logits[0]
    # print(logits[0].shape, logits[0].dtype)
    # print(logits[1].shape, logits[1].dtype)
    # print(logits.shape, logits.dtype)
    return out.max(dim=-1)[1]


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.data_sampler = My_DistributedBatchSampler

    print("Setup Data")

    all_combi_df_path = "/mnt/team_s3_synced/kawshik/knee_15k_all_combinations_QA_v1.pkl"

    internal_dataset = partial(All_Combi_Dataset,
                               all_combi_df_path=all_combi_df_path,
                               qtype=data_args.qtype)

    Train_dataset = MultidatasetBigrad(
        text_tokenizer=model_args.tokenizer_path,
        max_seq=data_args.max_seq,
        dataset_base=All_Combi_Dataset(all_combi_df_path, 'train'),
        split='train')

    for i in Train_dataset:

        print(i['question'])
        print(i['lang_x'])
        print(Train_dataset.text_tokenizer.convert_ids_to_tokens(i['lang_x']))

        print('*' * 50)
        break

    Eval_dataset = MultidatasetBigrad(text_tokenizer=model_args.tokenizer_path,
                                      max_seq=data_args.max_seq,
                                      dataset_base=All_Combi_Dataset(
                                          all_combi_df_path, 'validation'),
                                      split='validation')

    global tokenizer
    tokenizer = Eval_dataset.text_tokenizer

    print('*' * 100)

    import time
    start = time.time()
    for b_i, b in enumerate(Train_dataset):
        #         if b_i%100==0:
        #             print(b_i/len(Train_dataset))

        #             print('time: ',(time.time() - start) / (b_i+1))
        for key in b:

            print(key, type(b[key]), (b[key].shape if str(type(
                b[key])) == "<class 'torch.Tensor'>" else len(b[key])))
        break

    print('*' * 100)

    print("Setup Model")

    model = MultiLLaMAForCausalLM(lang_model_path=model_args.lang_encoder_path,
                                  torch_dtype=torch.bfloat16,
                                  bits=16)

    print("Load pretrained ckpt ")

    ckpt = torch.load(
        model_args.model_ckpt_load_dir, map_location='cpu'
    )  # Please dowloud our checkpoint from huggingface and Decompress the original zip file first

    model.load_state_dict(ckpt, strict=False)

    print("add Lora adapters")

    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.05
    lora_bias = "none"
    bits = 16

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_names(model.lang_model),
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )

    if bits == 16:
        # if training_args.bf16:
        model.to(torch.bfloat16)
    # if training_args.fp16:
    #     model.to(torch.float16)

    # rank0_print("Adding LoRA adapters...")

    # self.lang_model.config.use_cache = False

    model.lang_model = get_peft_model(model.lang_model, lora_config)

    # if bits in [4, 8]:

    #     self.lang_model.config.torch_dtype = torch_dtype
    #     self.lang_model = prepare_model_for_kbit_training(
    #         self.lang_model, use_gradient_checkpointing=True)

    print("freezing everything except lora")

    total = 0

    # param_groups_to_train = ['lora', 'embedding_layer']

    param_groups_to_train = eval(model_args.param_groups_to_train)

    lora_params = []
    finetune_params = []
    for n, p in model.named_parameters():
        if "lora" in n or "embedding_layer" in n:
            if "lora" in n:
                lora_params.append(p)
                if "lora" in param_groups_to_train:
                    p.requires_grad = True
                    total += p.numel()
                else:
                    p.requires_grad = False
            else:
                finetune_params.append(p)
                if "embedding_layer" in param_groups_to_train:
                    p.requires_grad = True
                    total += p.numel()
                else:
                    p.requires_grad = False

        else:
            p.requires_grad = False

    print('*' * 100, '\n', 'num_params:', total)
    print("Model Setup done")

    print('*' * 100)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    print('*' * 100)

    # print(model.lang_model)
    # tin = torch.rand(16, 256, 5120)
    # tout = model.lang_model.to('cuda')(inputs_embeds=tin.to('cuda'),
    #                                    attention_mask=torch.ones(
    #                                        16, 256).to('cuda'))
    # print(tin.shape, tout.shape)

    print('*' * 100)
    print(type(training_args.lr), type(eval(training_args.lr)))
    print('*' * 100)

    lrs = eval(training_args.lr)

    optimizer_grouped_parameters = []
    if "lora" in param_groups_to_train:
        optimizer_grouped_parameters.append({
            "params": lora_params,
            "lr": lrs[0],
            "weight_decay": 1e-4,
        })

    if "embedding_layer" in param_groups_to_train:
        optimizer_grouped_parameters.append({
            "params": finetune_params,
            "lr": lrs[1],
            "weight_decay": 5e-5,
        })

    optimizer = transformers.AdamW(params=optimizer_grouped_parameters)

    trainer = Trainer(model=model,
                      train_dataset=Train_dataset,
                      eval_dataset=Eval_dataset,
                      args=training_args,
                      optimizers=(optimizer, None),
                      data_collator=DataCollator(),
                      compute_metrics=compute_metrics,
                      preprocess_logits_for_metrics=get_preds)

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
