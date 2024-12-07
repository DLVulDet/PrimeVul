# PrimeVul: Vulnerability Detection with Code Language Models: How Far Are We?

<p align="left">
    <a href="https://arxiv.org/abs/2403.18624"><img src="https://img.shields.io/badge/arXiv-2403.18624-b31b1b.svg?style=for-the-badge">
    <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge">
</p>

<p align="left">
    📜 &nbsp;<a href="#-overview">Overview</a>
    | 📚&nbsp;<a href="#-primevul-dataset">Dataset</a>
    | 💻&nbsp;<a href="#-experiments">Experiments</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
</p>


* (09/05/24) 🌟 We released [PrimeVul-v0.1](https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK?usp=drive_link) to include more metadata for vulnerabilities. Check what's new!
* (07/02/24) 🎉 Our paper has been accepted to [ICSE 2025](https://conf.researchr.org/home/icse-2025) during the first submission cycle!
* (06/14/24) 🚀 PrimeVul has been included to evaluate [Gemini-1.5](https://arxiv.org/abs/2403.05530) for vulnerability detection!
* (03/27/24) We released our paper, data, and code for experiments.

## 📜 Overview

PrimeVul is a new dataset for vulnerability detection, aiming to train and evaluate code language models in the realistic vulnerability detection settings.

### Better Data

* ✨ **Diverse and Sufficient Data**: ~7k vulnerable functions and ~229k benign functions from real-world C/C++ projects, covering 140+ CWEs.
* ✨ **Accurate Labels**: Novel labeling techniques achieve human-level labeling accuracy, up to 3 $\times$ more accurate than existing automatic labeling approaches.
* ✨ **Minimal Contamination**: Thorough data de-duplication and chronological data splits minimizes the data contamination. 

### Better Evaluation

* ✨ **Realistic Tradeoff w/ VD-Score**: Measure risks of missing security flaws when not overwhelming developers with false alarms.
* ✨ **Exposing Models' Weaknesses w/ Paired Samples**: Analyzing models' capabilities in capturing subtle vulnerable patterns and distinguishing the vulnerable code from its patch.

## 📚 PrimeVul Dataset

### 💡 **\[Latest Release\]** **v0.1**

<p>
  <a href="https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK?usp=drive_link"><img src="https://img.shields.io/badge/Drive-PrimeVul_V0.1-ff8811.svg?style=for-the-badge&logo=googledrive&logoColor=white"></a>
</p>

#### 🌟 What's New?

To facilitate the future research and evaluation using PrimeVul, we retrieve more metadata, so that users could customize their training and evaluation (e.g., including more contexts) according to thier needs. Specifically, we have retrieve these key attributes:

* ⭐ **Commit Metadata**: We add (1) project URL, (2) commit URL, and (3) commit message correspnding to PrimeVul vulnerabilities. As a result, users could easily locate the original commit corresponding to the sample and retrieve project-level contexts.
* ⭐ **Vulnerability Metadata**: We add metadata for the vulnerabilities: (1) CVE description (2) NVD link to the vulnerability. With these information, users could perform in-dpeth analysis (manually or programmatically) for included vulnerabilities
* ⭐ **File-level Metadata**: We retrieve the file-level information for functions in PrimeVul. Specifically, we include (1) file name that the function belongs to, (2) the file's relative path in the project (3) the location of the function in the original file (4) a copy of the whole file. With these information, users could play with the file-level context of PrimeVul samples offline.

__How to Use__: While __commit and vulnerability metadata__ are directly saved as part of the json object, __file-level metadata__ is saved separately in `file_info.json` and `file_contents/`. Using the `func_hash` in the dataset, the file information can be found in `file_info.json`, which also provides the path to the local copy of the whole file.


> [!NOTE]
> 
> PrimeVul is a dataset that __combines and reconstructs__ existing vulnerability detection datasets with more accurate labels and thorough evaluation. However, not all datasets provide sufficient resources to retrieve the metadata (e.g., some samples do not originally have CWE type or CVE numbers). Therefore, we could not retrieve the same metadata for every sample in PrimeVul. In PrimeVul-v0.1, we only include vulnerabilities that we successfully retrieved their metadata. For the full set of samples that we orignally used in the paper, please refer to the original release below.

### **\[Original Release\]**

<p>
  <a href="https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY?usp=drive_link"><img src="https://img.shields.io/badge/Drive-PrimeVul-ff8811.svg?style=for-the-badge&logo=googledrive&logoColor=white"></a>
</p>


## 💻 Experiments

### Install Dependencies

```conda env create -f environment.yml```

### 📖 Open-source Code LMs (< 7B)

#### Standard Binary Classification

```sh
cd os_expr;

PROJECT="primevul_cls"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

#### Binary Classification + Class Weights

```sh
cd os_expr;

WEIGHT=30
PROJECT="primevul_cls_weights_${WEIGHT}"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --class_weight \
    --vul_weight $WEIGHT \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

#### Binary Classification + Contrastive Learning

```sh
cd os_expr;

PROJECT="primevul_cls_clr"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    <--clr_mask> \ # Add this parameter to enable CA-CLR
    --clr_temp 1.0 \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --epoch 10 \
    --block_size 512 \
    --group_size 32 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
cd ..;
```

### 📖 Open-source Code LMs (7B)

For 7B models, since it requires more expensive computation, we implement the training differently fromm < 7B models

#### Set up
- Install [Huggingface Accelerate](https://huggingface.co/docs/accelerate/en/index).
- To train CodeGen2.5, we need transformers==4.33.0. Otherwise, there will be an error while loading the tokenizer. See this [issue](https://github.com/salesforce/CodeGen/issues/82) for further details. 
- To train StarCoder2, please use the newest transformers.
- Cconfigure Accelerate. Run `accelerate config` and choose the configuration based on your need. The configuration we used is shown in ``os_expr/default_config.yaml``.

#### Standard Binary Classification with Parallel Training
```sh
PROJECT="parallel_primevul_cls"
TYPE=<MODEL_TYPE>
MODEL=<HUGGINGFACE_MODEL>
TOKENIZER=<HUGGINGFACE_TOKENIZER>
OUTPUT_DIR=../output/
accelerate launch run_ft_accelerator.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=<PATH_TO_primevul_train.jsonl> \
    --eval_data_file=<PATH_TO_primevul_valid.jsonl> \
    --test_data_file=<PATH_TO_primevul_test.jsonl> \
    --gradient_accumulation_steps 4 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

### 🤖 OpenAI Models

#### Standard Binary Classification
```sh
cd openai_expr;
MODEL=gpt-3.5-turbo-0125 # gpt-4-0125-preview
PROMPT_STRATEGY="std_cls";
python run_prompting.py \
    --model $MODEL \
    --prompt_strategy $PROMPT_STRATEGY \
    --data_path <PATH_TO_primevul_test_paired.jsonl> \
    --output_folder ../output_dir \
    --temperature 0.0 \
    --max_gen_length 1024 \
    --seed 1337 \
    --logprobs \
    --fewshot_eg
cd ..;
```

#### Chain-of-thought
```sh
cd openai_expr;
MODEL=gpt-3.5-turbo-0125 # gpt-4-0125-preview
PROMPT_STRATEGY=cot;
python run_prompting.py \
    --model $MODEL \
    --prompt_strategy $PROMPT_STRATEGY \
    --data_path <PATH_TO_primevul_test_paired.jsonl> \
    --output_folder ../output_dir \
    --temperature 0.0 \
    --max_gen_length 1024 \
    --seed 1337
cd ..;
```

### ⏲️ Calculate Vulnerability Detection Score (VD-S)

```sh
python calc_vd_score.py \
    --pred_file <OUTPUT_DIR>/predictions.txt
    --test_file <PATH_TO_primevul_test.jsonl>
```

## 📝 Citation

```bibtex
@article{ding2024primevul,
  title={Vulnerability Detection with Code Language Models: How Far Are We?}, 
  author={Yangruibo Ding and Yanjun Fu and Omniyyah Ibrahim and Chawin Sitawarin and Xinyun Chen and Basel Alomair and David Wagner and Baishakhi Ray and Yizheng Chen},
  journal={arXiv preprint arXiv:2403.18624},
  year={2024}
}
```
