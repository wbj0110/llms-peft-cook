from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from datasets import Features, ClassLabel,Dataset

task_to_keys = {
    "financial_phrasebank": ("sentence", None),
    "imdb-truncated": ("text", None),
    "imdb": ("text", None),
    "tweet_eval_irony": ("text", None),
    "tweet_eval_stance_abortion": ("text", None),
    "emotion": ("text", None),
    "ag_news": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "auditor_sentiment": ("sentence", None),

}

class DatasetsFactory:
    def __init__(self):
        pass

    def load_datasets(self,script_args,tokenizer):
        pass



class Seq2SeqLMAndSeqClsDatasets(DatasetsFactory):
    def load_datasets(self,script_args,tokenizer):
        # dataset = load_dataset(script_args.dataset_name, split="train")
        if 'tweet_eval' in script_args.dataset_name:
            if 'tweet_eval_irony' in script_args.dataset_name:
                dataset = load_dataset(script_args.dataset_name,'irony')
            elif  'tweet_eval_stance_abortion' in script_args.dataset_name:
                dataset = load_dataset(script_args.dataset_name,'abortion')
        elif 'auditor' in script_args.dataset_name:
            if 'auditor_sentiment' in script_args.dataset_name:

                dataset = load_dataset(script_args.dataset_name, 'sentiment')
        elif 'imdb-truncated' in  script_args.dataset_name:

            dataset = load_dataset(script_args.dataset_name)

        else:
            dataset = load_dataset(script_args.dataset_name)
             # dataset = dataset["train"].train_test_split(test_size=0.1)
        sentence1_key, sentence2_key = 'text',None
        task = None
        if 'financial_phrasebank' in script_args.dataset_name:
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=55, stratify_by_column="label")
            dataset["validation"] = dataset["test"]
            del dataset["test"]
            task = 'financial_phrasebank'
            sentence1_key, sentence2_key = task_to_keys[task]

        elif 'imdb-truncated' in  script_args.dataset_name:
            task = 'imdb-truncated'
            sentence1_key, sentence2_key = task_to_keys[task]

            original_features = dataset['train'].features
            print("imdb-truncated原始数据集特征:", original_features)

            num_classes = script_args.num_labels  #
            class_label = ClassLabel(num_classes=num_classes, names=[str(i) for i in range(num_classes)])

            # 更新数据集的特征类型
            new_features = original_features.copy()
            new_features['label'] = class_label
            dataset = dataset.cast(new_features)

            # 检查更新后的数据集结构
            updated_features = dataset['train'].features
            print("imdb-truncated更新后数据集特征:", updated_features)
        elif 'imdb' in  script_args.dataset_name:
            dataset["validation"] = dataset["test"]
            del dataset["test"]
            task = 'imdb'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'cola' in  script_args.dataset_name:
            task = 'cola'
            sentence1_key, sentence2_key = task_to_keys[task]
            dataset["validation"] = dataset["test"]
            del dataset["test"]
        elif 'mnli' in  script_args.dataset_name:
            task = 'mnli'
            sentence1_key, sentence2_key = task_to_keys[task]
            dataset["validation"] = dataset["validation_matched"]
            del dataset["validation_matched"]
        elif 'mnli-mm' in  script_args.dataset_name:
            task = 'mnli-mm'
            sentence1_key, sentence2_key = task_to_keys[task]
            dataset["validation"] = dataset["validation_mismatched"]
            del dataset["validation_mismatched"]
        elif 'mrpc' in  script_args.dataset_name:
            task = 'mrpc'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'qnli' in  script_args.dataset_name:
            task = 'qnli'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'qqp' in  script_args.dataset_name:
            task = 'qqp'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'rte' in  script_args.dataset_name:
            task = 'rte'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'sst2' in  script_args.dataset_name:
            task = 'sst2'
            sentence1_key, sentence2_key = task_to_keys[task]
            # dataset["train"] = dataset["test"]
            del dataset["test"]
        elif 'stsb' in  script_args.dataset_name:
            task = 'stsb'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'wnli' in  script_args.dataset_name:
            task = 'wnli'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'tweet_eval_irony' in  script_args.dataset_name:
            task = 'tweet_eval_irony'
            sentence1_key, sentence2_key = task_to_keys[task]
        elif 'ag_news' in  script_args.dataset_name:
            task = 'ag_news'
            sentence1_key, sentence2_key = task_to_keys[task]
            dataset = dataset["test"].train_test_split(test_size=0.2, shuffle=True,seed=52, stratify_by_column="label")
            dataset["validation"] = dataset["test"]
            del dataset["test"]
        elif 'auditor_sentiment' in script_args.dataset_name:
            task = 'auditor_sentiment'
            sentence1_key, sentence2_key = task_to_keys[task]
            dataset["validation"] = dataset["test"]
            del dataset["test"]

            original_features = dataset['train'].features
            print("auditor_sentiment原始数据集特征:", original_features)

            num_classes = script_args.num_labels  #
            class_label = ClassLabel(num_classes=num_classes, names=[str(i) for i in range(num_classes)])

            # 更新数据集的特征类型
            new_features = original_features.copy()
            new_features['label'] = class_label
            dataset = dataset.cast(new_features)

            # 检查更新后的数据集结构
            updated_features = dataset['train'].features
            print("auditor_sentiment更新后数据集特征:", updated_features)


        else:
            sentence1_key, sentence2_key = "text",None
        #
        # if task in ['imdb','tweet_eval_irony','ag_news','sst2']:
        #     range_size = min(6000, len(dataset))
        #     eval_range_size = min(2000, len(dataset["validation"]))
        #     dataset["train"] = dataset["train"].shuffle(seed=15).select(range(range_size))
        #     dataset["validation"] = dataset["validation"].shuffle(seed=15).select(range(eval_range_size))

        max_length = script_args.max_length
        if sentence2_key is None:
            print(f"Sentence: {dataset['train'][0][sentence1_key]}")
        else:
            print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
            print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

        # label_map = {}
        # for i in range(script_args.num_labels):
        #     label_map[i] = str(i)

        if 'bert' in script_args.model_name:
            def preprocess_function(examples):

                if sentence2_key is None:
                    return tokenizer(examples[sentence1_key], max_length=max_length, padding="max_length",truncation=True)
                return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
            processed_datasets = dataset.map(preprocess_function, batched=True)

        else:
            # if 'auditor_sentiment' in script_args.dataset_name:
            #     classes = dataset["train"].features["label"].names
            # else:
            classes = dataset["train"].features["label"].names
            num_labels = len(classes)
            print('number of labels:{}'.format(num_labels))
            def dataset_process(example):
                x =  {"text_label": [classes[label] for label in example["label"]]}
                return x

            dataset = dataset.map(
                dataset_process,
                batched=True,
                num_proc=1,
            )

            # data preprocessing
            '''
            apply some pre-processing of the input data, the labels needs to be pre-processed, 
            the tokens corresponding to pad_token_id needs to be set to -100 
            so that the CrossEntropy loss associated with the model will correctly ignore these tokens.
            '''

            label_column = "text_label"
            text_column =sentence1_key
            def preprocess_function(examples):
                inputs = examples[text_column]
                targets = examples[label_column]
                model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True,
                                         return_tensors="pt")

                labels = tokenizer(targets, max_length=num_labels, padding="max_length", truncation=True,
                                   return_tensors="pt")
                labels = labels["input_ids"]
                labels[labels == tokenizer.pad_token_id] = -100
                model_inputs["labels"] = labels
                return model_inputs

            processed_datasets = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        # if task in ['financial_phrasebank','imdb','tweet_eval_irony','ag_news','sst2']:
        #     range_size = min(1000, len(train_dataset))
        #     eval_range_size = min(1000, len(eval_dataset))
        #     train_dataset = train_dataset.shuffle(seed=55).select(range(range_size))
        #     eval_dataset = eval_dataset.shuffle(seed=55).select(range(eval_range_size))
        return train_dataset, eval_dataset, dataset,task



