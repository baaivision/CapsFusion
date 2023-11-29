import os
import json
import logging
from typing import Dict
from tqdm import tqdm
import time
from omegaconf import OmegaConf

import torch
import transformers
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

config = OmegaConf.load("config.yaml")


class CapsFusion(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        logging.info(f'Number of Samples: {len(self.data_list)}')

        def wrap(ori_caption, synthetic_caption):
            sample = f"Please merge and refine the information from the two given sentences. " \
                       f"Sentence 1 provides detailed real-world knowledge, " \
                       f"yet it suffers from flaws in sentence structure and grammar. " \
                       f"Sentence 2 exhibits nice sentence structure, " \
                       f"but lacking in-depth real-world details and may contain false information. " \
                       f"Please combine them into a new sentence, " \
                       f"ensuring a well-structured sentence while retaining the detailed real-world information provided in Sentence 1. " \
                       f"Avoid simply concatenating the sentences.\n\n" \
                       f"Sentence 1: {ori_caption}\n" \
                       f"Sentence 2: {synthetic_caption}\n" \
                       f"New Sentence:"
            return sample
        
        def input_text(ori, syn):
            total_len = len(tokenizer(wrap(ori, syn)).input_ids)
            if total_len >= 256:
                ori = tokenizer.decode(tokenizer(ori, add_special_tokens=False).input_ids[:-(total_len - 256)])
            return wrap(ori, syn)

        self.input = [
            input_text(example['laion_2b'].strip(), example['laion_coco'].strip()) for example in self.data_list
        ]

    def __getitem__(self, item):
        return self.data_list[item], self.input[item]

    def __len__(self):
        return len(self.data_list) // 10


def collate_fn(batch):
    data_list = []
    input_list = []
    for data, inpt in batch:
        data_list.append(data)
        input_list.append(inpt)

    return data_list, input_list


def build_lcoco_loader(tokenizer):
    dataset = CapsFusion(config.data_file, tokenizer)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=distributed_sampler,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        

def main():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

    # flash attention
    try:
        from llama_flash_attn_monkey_patch_xops import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        print("Monkey patched llama attention with flash attention...")
    except ImportError:
        print("Could not monkey patch llama attention with flash attention...")

    # initialize and load model and tokenizer
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
    ).bfloat16().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # dataloader
    SAVE_PATH = config.save_path
    dataloader = build_lcoco_loader(tokenizer)
    logging.info(f'Building DataLoader Done!!! This Dataloader Length: {len(dataloader)}')

    # inference
    return_list = []
    with torch.no_grad():
        for data, inputs in tqdm(dataloader, disable=not dist.get_rank() == 0):
            input_tokens = tokenizer(
                inputs,
                padding="max_length",
                return_tensors="pt",
                max_length=config.text_input_max_length,
                truncation=True,
            ).to(device)

            input_ids = input_tokens.input_ids
            attention_mask = input_tokens.attention_mask

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=config.num_beams,
                max_new_tokens=config.max_new_tokens,
            )

            output_text = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:], skip_special_tokens=True
            )

            for d, answer in zip(data, output_text):
                d["capsfusion"] = answer

            return_list.extend(data)
            
    save_file = f'{SAVE_PATH}/result_{dist.get_rank()}.json'
    logging.info(f"Save to: {save_file}")
    with open(save_file, 'w') as f:
        json.dump(return_list, f)

    # waiting for possible process asynchronization
    # logging.info(f"Starting Sleeping and Waiting for Other Processes")
    # time.sleep(1800)

    logging.info('Finished')

if __name__ == "__main__":
    main()