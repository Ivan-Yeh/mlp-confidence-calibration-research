# Under Construction 🏗️

This project is still being developed... 🔨

# Language Model Evaluation Pipeline

## Huggingface Downloader Instructions

To download a model from Huggingface, run:

`python3 hf_downloader.py [Huggingface token] [Huggingface model id]`

## Simple QA

### Setup

Ensure `./tmp` exists before running as evaluation results will be stored there. 

The evaluation requires an environment variable `HF_TOKEN` that holds your Huggingface Access Token. To set up a persistent environment variable:

Append `export HF_TOKEN="[YOUR TOKEN]"` to `~/.bashrc`,
and then run `source ~/.bashrc` to apply changes.

### Run Simple QA with a Huggingface Model

run: `python -m simple-evals.my_evals --model [Huggingface model id] [--examples 5]`

e.g. 

`python -m simple-evals.my_evals --model meta-llama/Llama-3.2-3B-Instruct --examples 3` 

`python -m simple-evals.my_evals --model meta-llama/Llama-3.1-8B-Instruct --examples 3`