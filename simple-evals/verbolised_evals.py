import json
import argparse
import pandas as pd
from . import common
from .mmlu_eval import MMLUEval
from .sampler.hf_sampler import HFSampler
import torch
import transformers
from huggingface_hub import snapshot_download, login
import os
from .verbalised_conf import vanilla_prompt, cot_prompt, self_probing_prompt, multi_step_prompt, top_k_prompt
from .ece import ece_equal_width, ece_equal_weight

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--list-prompting", action="store_true", help="List available prompting strategies")
    parser.add_argument("--prompting", type=str, help="Select a prompting strategy to calculate ECE")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    models_ls = ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
    prompting_ls = ["Vanilla", "CoT", "Self-Probing", "Multi-Step", "Top-K"]

    if args.list_models:
        print("Available models:")
        for model_name in models_ls:
            print(f" - {model_name}")
        return
    
    if args.list_prompting:
        print("Available prompting strategies: ")
        for strategy in prompting_ls:
            print(f" - {strategy}")
        return

    if args.model:
        if args.model not in models_ls:
            print(f"Error: Model '{args.model}' not found.")
            return
        
    if args.prompting:
        if args.prompting not in prompting_ls:
            print(f"Error: Prompting strategy '{args.prompting}' not found.")
            return
        
    # login(token=os.environ["HF_TOKEN"])
    with open('./api.json', 'r') as f:
        key = json.load(f)["HF_TOKEN"]
        login(token=key)
    local_dir = snapshot_download(repo_id=args.model)
    pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, device_map="auto", model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    system_msg = None
    match args.prompting:
        case "Vanilla": system_msg = vanilla_prompt()
        case "CoT": system_msg = cot_prompt()
        case "Self-Probing": system_msg = self_probing_prompt()
        case "Multi-Step": system_msg = multi_step_prompt()
        case "Top-K": system_msg = top_k_prompt()

    models = {args.model: HFSampler(pipeline=pipeline, terminators=terminators, model=args.model, system_message=system_msg, max_tokens=2048),}

    # grading_sampler = HFSampler(pipeline, terminators, model=args.model, system_message=system_msg, max_tokens=2048)
    # equality_checker = HFSampler(pipeline, terminators, model=args.model, system_message=system_msg, max_tokens=2048)

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            
    evals = {
        eval_name: get_evals(eval_name, args.debug)
        # for eval_name in ["simpleqa"]
        for eval_name in ["mmlu"]
    }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name.split("/")[-1]}"
            report_filename = f"./tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"./tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics

if __name__ == "__main__":
    main()