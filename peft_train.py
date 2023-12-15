from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from peft import LoraConfig,get_peft_model,TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import evaluate

from datasets_process import Seq2SeqLMAndSeqClsDatasets

datasets_map = {
    'SEQ_2_SEQ_LM':Seq2SeqLMAndSeqClsDatasets(),
    'SEQ_CLS':Seq2SeqLMAndSeqClsDatasets()
}

########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)




# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=64)
    per_device_eval_batch_size: Optional[int] = field(default=64)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    max_length: Optional[int] = field(default=128)
    model_name: Optional[str] = field(
        default="tiiuae/falcon-7b-instruct", #tiiuae/falcon-40b-instruct"/ tiiuae/falcon-7b tiiuae/falcon-7b-instruct "ybelkada/falcon-7b-sharded-bf16"
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading."},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models."},
    )
    need_hyperparameters_search: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether need hyperparameters search."},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    epochs: Optional[int] = field(
        default=5,
        metadata={"help": "number of epochs."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables load_in_8bit Default False."},
    )

    enable_peft: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables peft training."},
    )
    use_cpu: Optional[bool] = field(
        default=False,
        metadata={"help": "Weather use cpu."},
    )

    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Weather push_to_hub."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    output_model_path: str = field(
        default="output/falcon-7b_lora",
        metadata={"help": "model path for persist."},
    )
    model_type: str = field(
        default="CAUSAL_LM", #CAUSAL_LM,SEQ_CLS,SEQ_2_SEQ_LM
        metadata={"help": "model path for persist."},
    )
    num_labels: Optional[int] = field(
        default=2,
        metadata={"help": "number of labels for classification."},
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    epochs: int = field(default=5, metadata={"help": "number of epochs."})
    save_steps: int = field(default=100, metadata={"help": "save_steps."})
    save_total_limit: int = field(default=8, metadata={"help": "save_steps."})

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # device_map = {"": 0}
    device_map = "auto"
    model = None
    lora_type = TaskType.SEQ_CLS


    if  'CAUSAL_LM' in args.model_type:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config,
            device_map=device_map, trust_remote_code=True,
        )

        model.lm_head = CastOutputToFloat(model.lm_head)

        lora_type = TaskType.CAUSAL_LM

        target_modules = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],  # , "word_embeddings", "lm_head"],

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            # task_type="CAUSAL_LM",
            task_type=lora_type,
            target_modules=target_modules,
        )

        print('use AutoModelForCausalLM load llms model.')
    elif 'SEQ_CLS' in args.model_type:
        from transformers import BertForSequenceClassification,AutoModelForSequenceClassification
        if 'bert' in args.model_name:
            model = BertForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=args.num_labels
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=args.num_labels
            )
        lora_type = TaskType.SEQ_CLS
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            # task_type="CAUSAL_LM",
            task_type=lora_type,
        )
        print('use AutoModelForSequenceClassification load bert model.')
    elif 'SEQ_2_SEQ_LM' in args.model_type:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name
            ,load_in_8bit=args.load_in_8bit
        )
        lora_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q", "v"]  # , "word_embeddings", "lm_head"],

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            # task_type="CAUSAL_LM",
            task_type=lora_type,
            target_modules=target_modules,
        )

        print('use AutoModelForSeq2SeqLM load  model.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if 'bert' not in script_args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()



    if args.enable_peft:
        if args.load_in_8bit:
            '''
            Casts all the non int8 modules to full precision (fp32) for stability
            Add a forward_hook to the input embedding layer to enable gradient computation of the input hidden states
            Enable gradient checkpointing for more memory-efficient training
            '''
            from peft import prepare_model_for_int8_training
            model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    return model, peft_config, tokenizer



def compute_metrics_seq_cls(eval_pred):
    logits, labels = eval_pred

    if len(labels.shape) >1:
        # 提取预测的类别
        predictions = np.argmax(logits[0], axis=-1)

        # 筛选出有效的标签和对应的预测
        valid_indices = labels != -100
        valid_labels = labels[valid_indices]
        valid_predictions = predictions[valid_indices]
    else:
        if task != "stsb":
            valid_predictions = np.argmax(logits, axis=1)
        else:
            valid_predictions = logits[:, 0]
        valid_labels = labels

    # 计算指标
    accuracy = accuracy_score(valid_labels, valid_predictions)
    precision = precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
    recall = recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
    f1 = f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0)

    dict1={
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    # dict2 = metric.compute(predictions=valid_predictions, references=labels)
    # merged_dict = {**dict1, **dict2}
    # return merged_dict
    return dict1


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    print(logits.shape)  # 应该是二维的，例如 (batch_size, num_classes)
    print(labels.shape)  # 通常是一维的，例如 (batch_size,)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_datasets(script_args,tokenizer):
    return datasets_map[script_args.model_type].load_datasets(script_args=script_args,tokenizer=tokenizer)

if __name__ == '__main__':

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # metric = evaluate.load("accuracy")

    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    print("tokenizer padding setting:", tokenizer.pad_token)
    model.config.use_cache = False

    train_dataset,eval_dataset,dataset,task = get_datasets(script_args,tokenizer)

    trainer = None
    training_arguments = None

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"



    if  'CAUSAL_LM' in script_args.model_type:
        training_arguments = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            optim=script_args.optim,
            save_steps=script_args.save_steps,
            logging_steps=script_args.logging_steps,
            learning_rate=script_args.learning_rate,
            fp16=script_args.fp16,
            bf16=script_args.bf16,
            max_grad_norm=script_args.max_grad_norm,
            max_steps=script_args.max_steps,
            warmup_ratio=script_args.warmup_ratio,
            group_by_length=script_args.group_by_length,
            lr_scheduler_type=script_args.lr_scheduler_type,
            report_to="none",
            use_cpu=script_args.use_cpu,
            # load_best_model_at_end=True,
            # metric_for_best_model=metric_name
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=script_args.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=script_args.packing,
        )
    # elif 'SEQ_2_SEQ_LM' in script_args.model_type:
    else:

        training_arguments = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            logging_steps=script_args.logging_steps,
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            # auto_find_batch_size=True,
            num_train_epochs=script_args.epochs,
            save_steps=script_args.save_steps,
            save_total_limit=script_args.save_total_limit,
            group_by_length=script_args.group_by_length,
            use_cpu=script_args.use_cpu,
            report_to="none",
            # load_best_model_at_end=True,
            # metric_for_best_model=metric_name,
            # push_to_hub=True,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_seq_cls,
        )

    # else:
    #     training_args = TrainingArguments(
    #         output_dir="./results",
    #         evaluation_strategy="epoch",
    #         logging_steps=script_args.logging_steps,
    #         learning_rate=script_args.learning_rate,
    #         per_device_train_batch_size=script_args.per_device_train_batch_size,
    #         gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    #         auto_find_batch_size=True,
    #         num_train_epochs=script_args.epochs,
    #         save_steps=script_args.save_steps,
    #         save_total_limit=script_args.save_total_limit,
    #         report_to = "none",
    #     )
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         compute_metrics=compute_metrics,
    #     )
    model.config.use_cache = False



    for name, module in trainer.model.named_modules():
        if isinstance(module, LoraLayer):
            if script_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if script_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    trainer.train()


    if script_args.need_hyperparameters_search:
        # train_dataset = train_dataset["train"].shard(index=1, num_shards=10)

        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

        print(best_run)

        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()

    # Save trained model
    trainer.model.save_pretrained(script_args.output_model_path)

    import os
    if script_args.push_to_hub:
        from huggingface_hub import notebook_login
        notebook_login()
        model.push_to_hub("wbj/{}-{}-lora".format(script_args.model_name,os.path.basename(script_args.output_model_path)), use_auth_token=True)