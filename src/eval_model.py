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

    print("Setup Data")
    Train_dataset = MultidatasetBigrad(
        text_tokenizer=model_args.tokenizer_path,
        max_seq=1024,
        split='train',
        mode='eval')
    Eval_dataset = MultidatasetBigrad(text_tokenizer=model_args.tokenizer_path,
                                      max_seq=1024,
                                      split='validation',
                                      mode='eval')
    Test_dataset = MultidatasetBigrad(text_tokenizer=model_args.tokenizer_path,
                                      max_seq=1024,
                                      split='validation',
                                      mode='eval')

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

    ckpt = torch.load(
        '/mnt/team_blackhole/kawshik/radfm_ckpts/checkpoint-250/pytorch_model.bin',
        map_location='cpu'
    )  # Please dowloud our checkpoint from huggingface and Decompress the original zip file first

    model.load_state_dict(ckpt)

    # if bits in [4, 8]:

    #     self.lang_model.config.torch_dtype = torch_dtype
    #     self.lang_model = prepare_model_for_kbit_training(
    #         self.lang_model, use_gradient_checkpointing=True)

    # print('*' * 100, '\n', 'num_params:', total)
    print("Model Setup done")

    model = model.to('cuda')
    model.eval()

    # print(model.lang_model)
    # tin = torch.rand(16, 256, 5120)
    # tout = model.lang_model.to('cuda')(inputs_embeds=tin.to('cuda'),
    #                                    attention_mask=torch.ones(
    #                                        16, 256).to('cuda'))
    # print(tin.shape, tout.shape)

    # print('*' * 100)

    c_i = 0
    for instance in tqdm.tqdm(Train_dataset):
        vision_x = instance['vision_x']
        vision_x = torch.nn.functional.interpolate(vision_x,
                                                   size=(256, 256, 24))

        lang_x = instance['lang_x']
        attention_mask = instance['attention_mask']
        print(vision_x.shape, lang_x.shape, attention_mask.shape)

        generation = model.generate(
            lang_x.unsqueeze(0).to('cuda'),
            vision_x.unsqueeze(0).to('cuda')).squeeze(0)
        print('generation: ', generation)
        generated_texts = Test_dataset.text_tokenizer.decode(
            generation, skip_special_tokens=True)

        print('*' * 100)
        print('question: ',
              Test_dataset.text_tokenizer.decode(instance['lang_x']))
        # print('
        print('-' * 25)
        print('answer: ', instance['answer'])
        print('-' * 25)
        print('prediction: ',
              Test_dataset.text_tokenizer.convert_ids_to_tokens(generation))
        print('*' * 100)

        input("enter to continue")

        c_i += 1

        # {
        #     'vision_x': vision_x,
        #     'lang_x': lang_x,
        #     'attention_mask': attention_mask,
        #     'labels': labels,
        #     'loss_reweight': reweight_tensor,
        #     'key_words_query': emphasize_words
        # }

    # trainer.train()
    # trainer.save_state()


if __name__ == "__main__":
    main()