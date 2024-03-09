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
from transformers import GenerationConfig
import evaluate
from functools import partial
from Dataset.dataset.internal3d import Internal3DDataset, DfForDlDataset, All_Combi_Dataset


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
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs, axis=-1)}


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/mnt/team_blackhole/kawshik/Language_files/")
    tokenizer_path: str = field(
        default="/mnt/team_blackhole/kawshik/Language_files/",
        metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=4)
    output_dir: Optional[str] = field(default="./Results/BLIP_overfit/")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pathology_choice: str = field(default=None)


#dict_keys(['acl', 'cartilage - lateral compartment', 'cartilage - medial compartment', 'cartilage - patellofemoral compartment', 'extensor alignment', 'joint fluid', 'lcl', 'lateral meniscus', 'mcl', 'marrow/bone', 'medial meniscus', 'pcl', 'patellar tendon', 'popliteal cyst', 'quad tendon', 'synovium', 'intraarticular body'])


@dataclass
class DataCollator(object):

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


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.data_sampler = My_DistributedBatchSampler

    print(training_args.pathology_choice)

    print("Setup Data")

    #     image_df_path = "/mnt/team_blackhole/kawshik/df_for_dl_no_crop.pkl"
    #     sep_qa_path = "/mnt/team_blackhole/kawshik/seq_qa.pkl"
    #     report_qa_path = "/mnt/team_blackhole/kawshik/60k_internal_data_reports_w_sections_and_segments_v2.pkl"

    # internal_dataset = partial(DfForDlDataset,
    #                            image_df_path=image_df_path,
    #                            sep_qa_path=sep_qa_path,
    #                            report_qa_path=report_qa_path,
    #                            pathology_choice=pathology_choice)

    all_combi_df_path = "/mnt/team_s3_synced/kawshik/knee_15k_all_combinations_QA.pkl"
    qtype = 'pathology_severity'
    split = 'train'
    sample_num = 10

    # internal_dataset =

    Train_dataset = MultidatasetBigrad(
        text_tokenizer=model_args.tokenizer_path,
        max_seq=1024,
        split='train',
        mode='eval',
        dataset_base=All_Combi_Dataset(all_combi_df_path,
                                       'train',
                                       qtype,
                                       sample_num=sample_num),
        pathology_choice=training_args.pathology_choice)

    for i in Train_dataset:

        print(i['question'])
        print(i['lang_x'])
        print(Train_dataset.text_tokenizer.convert_ids_to_tokens(i['lang_x']))
        print('*' * 50)

    Eval_dataset = MultidatasetBigrad(
        text_tokenizer=model_args.tokenizer_path,
        max_seq=1024,
        split='validation',
        mode='eval',
        dataset_base=All_Combi_Dataset(all_combi_df_path, 'validation', qtype),
        pathology_choice=training_args.pathology_choice)

    Test_dataset = MultidatasetBigrad(
        text_tokenizer=model_args.tokenizer_path,
        max_seq=1024,
        split='test',
        mode='eval',
        dataset_base=All_Combi_Dataset(all_combi_df_path, 'test', qtype),
        pathology_choice=training_args.pathology_choice)

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

    #     print("add Lora adapters")

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

    ckpt = torch.load(
        # '/mnt/team_s3_synced/kawshik/radfm_ckpts/severity/checkpoint-1034/pytorch_model.bin', # all finetune
        # "/mnt/team_s3_synced/kawshik/radfm_ckpts/wo_lora/checkpoint-950/pytorch_model.bin", # pretrain
        # "/mnt/team_s3_synced/kawshik/radfm_ckpts/finetune_lora_severity/only_lora/b64_2e-4/checkpoint-3385/pytorch_model.bin",  # finetune only lora
        "/mnt/team_s3_synced/kawshik/radfm_ckpts/finetune_lora_severity/all_combinations_in_each_epoch/checkpoint-3104/pytorch_model.bin",  # finetune severity only lora - all combinations
        map_location='cpu'
    )  # Please dowloud our checkpoint from huggingface and Decompress the original zip file first

    model.load_state_dict(ckpt)

    # if bits in [4, 8]:

    #     self.lang_model.config.torch_dtype = torch_dtype
    #     self.lang_model = prepare_model_for_kbit_training(
    #         self.lang_model, use_gradient_checkpointing=True)

    # print('*' * 100, '\n', 'num_params:', total)
    print("Model Setup done")

    model.to(torch.bfloat16)
    model = model.to('cuda')
    model.eval()
    # if bits == 16:
    # if training_args.bf16:

    # print(model.lang_model)
    # tin = torch.rand(16, 256, 5120)
    # tout = model.lang_model.to('cuda')(inputs_embeds=tin.to('cuda'),
    #                                    attention_mask=torch.ones(
    #                                        16, 256).to('cuda'))
    # print(tin.shape, tout.shape)

    # print('*' * 100)

    c_i = 0

    #### 128 max len for task 3

    config_type = 'beam_search'
    # config_type = 'topk'
    # config_type = 'contrastive_search' # 'beam_search'

    count_num = 19245

    penalty_alpha = 0
    top_k = 1
    num_beams = 1
    do_sample = False
    temperature = 1.0
    top_p = 1.0
    if config_type == 'greedy':
        num_beams = 1
        do_sample = False
    elif config_type == 'topk':
        do_sample = True
        temperature = 0.5
        top_k = 50
        top_p = 1.0
    elif config_type == 'beam_search':
        num_beams = 5
        do_sample = False
    elif config_type == 'contrastive_search':
        penalty_alpha = 0.5
        top_k = 5

    generation_config = GenerationConfig(
        # max_new_tokens=1024,
        max_new_tokens=32,
        min_new_tokens=8,
        num_beams=num_beams,
        early_stopping=False,
        use_cache=True,
        do_sample=do_sample,
        # do_sample=True,
        penalty_alpha=penalty_alpha,
        # top_k=10,
        temperature=temperature,
        top_k=top_k,
        # top_p=1.0,
        repetition_penalty=1.1,
        # repetition_penalty=1.0,
        # temperature=0.5,
        # penalty_alpha=0.6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # pad_token_id=0,
    )

    rouge_score = evaluate.load("rouge")

    outs = []

    if split == 'test':
        dataset_to_use = Test_dataset
    elif split == 'validation':
        dataset_to_use = Eval_dataset
    else:
        dataset_to_use = Train_dataset

    for instance in tqdm.tqdm(dataset_to_use):
        if instance is None:
            continue

        study_id = instance['study_id']
        vision_x = instance['vision_x']
        vision_x = torch.nn.functional.interpolate(vision_x,
                                                   size=(256, 256, 24)).to(
                                                       torch.bfloat16)

        lang_x = instance['lang_x']
        attention_mask = instance['attention_mask']
        # print(vision_x.shape, lang_x.shape, attention_mask.shape)

        with torch.no_grad():
            generation = model.generate(
                lang_x.unsqueeze(0).to('cuda'),
                vision_x.unsqueeze(0).to('cuda'),
                generation_config=generation_config).squeeze(0)
            # print('generation: ', generation)

        generated_texts = Test_dataset.text_tokenizer.decode(
            generation).strip()

        # print('*'*100)
        # print('question: ', instance['question'])
        # print('-'*25)
        # print('answer: ', instance['answer'])
        # print('-'*25)
        # print('prediction: ', generated_texts)
        # print('*'*100)

        result = rouge_score.compute(predictions=[generated_texts],
                                     references=[instance['answer']],
                                     use_stemmer=True)
        # print(result)
        # print('*'*100)

        # print(ACCs)
        result = {key: value for key, value in result.items()}
        # print(result)
        #         print('*'*100)

        base_dict = {
            "study_id": study_id,
            "question": instance['question'],
            "answer": instance['answer'],
            "pred": generated_texts,
            "qtype": instance["qtype"],
            "pathology": instance["pathology"],
        }

        # print(base_dict)
        base_dict.update(result)
        # print(base_dict)
        outs.append(base_dict)

        # input("enter to continue")

        c_i += 1

        if c_i > count_num:
            break

        # {
        #     'vision_x': vision_x,
        #     'lang_x': lang_x,
        #     'attention_mask': attention_mask,
        #     'labels': labels,
        #     'loss_reweight': reweight_tensor,
        #     'key_words_query': emphasize_words
        # }

    import pandas as pd
    outs = pd.DataFrame(outs)
    for key in result:
        print(key, np.mean(outs[key].tolist()), np.std(outs[key].tolist()),
              np.min(outs[key].tolist()), np.max(outs[key].tolist()))

    # outs.to_pickle(f"sample_outputs_report_findings_{config_type}_{count_num}{'_pathology-' + training_args.pathology_choice if training_args.pathology_choice is not None else ''}.pkl")
    outs.to_pickle(
        f"all_outputs_{split + '-' + str(sample_num) if sample_num is not None else split}_report_findings_{config_type}_{count_num}{'_pathology-' + training_args.pathology_choice if training_args.pathology_choice is not None else ''}.pkl"
    )


if __name__ == "__main__":
    main()