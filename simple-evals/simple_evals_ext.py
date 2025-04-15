import json
import argparse
import pandas as pd
from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .simpleqa_eval import SimpleQAEval
# from .sampler.chat_completion_sampler import (
#     OPENAI_SYSTEM_MESSAGE_API,
#     OPENAI_SYSTEM_MESSAGE_CHATGPT,
#     ChatCompletionSampler,
# )
# from .sampler.o_chat_completion_sampler import OChatCompletionSampler
# from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS

# HF Extension:
from .sampler.hf_sampler import HFSamplerPipeline, HFSamplerTokeniser
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import transformers
from huggingface_hub import snapshot_download, login
import os

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--list-tests", action="store_true", help="List available tests (MMLU by default)"
    )
    parser.add_argument(
        "--list-confidence", action="store_true", help="List available confidence extraction methods (verbalised by default)"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--test", type=str, help="Select a test by name")
    parser.add_argument("--confidence", type=str, help="Select a confidence extraction method by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    models = {
        # chatgpt models:
        # "gpt-4o-2024-11-20_assistant": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-2024-11-20_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "o1": OChatCompletionSampler(
        #     model="o1",
        # ),
        # "o1-preview": OChatCompletionSampler(
        #     model="o1-preview",
        # ),
        # "o1-mini": OChatCompletionSampler(
        #     model="o1-mini",
        # ),
        # # Default == Medium
        # "o3-mini": OChatCompletionSampler(
        #     model="o3-mini",
        # ),
        # "o3-mini_high": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="high",
        # ),
        # "o3-mini_low": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="low",
        # ),
        # "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        # ),
        # "gpt-4o_assistant": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
        #     model="gpt-4o-mini-2024-07-18",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4.5-preview-2025-02-27": ChatCompletionSampler(
        #     model="gpt-4.5-preview-2025-02-27",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ), 
        # # claude models:
        # "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
        #     model="claude-3-opus-20240229",
        #     system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        # ),
        # HF models: None by default; only constructed when chosen
        "meta-llama/Llama-3.2-3B-Instruct": None,
        "meta-llama/Llama-3.1-8B-Instruct": None
    }

    


    tests = ["simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop", "humaneval"]
    if args.list_tests:
        print("Available tests:")
        for r in tests:
            print(f" - {r}")
        return
    if args.test:
        if args.test not in tests:
            print(f"Error: Test '{args.test}' not found.")
            return
        else:
            tests = [args.test]
    else:
        tests = ["mmlu"]

    verbal_sampler = ["verbal-cot", "verbal-vanilla"]
    logits_sampler = ["empirical-semantic", "single-generation"]
    confidence = verbal_sampler + logits_sampler
    if args.list_confidence:
        print("Available confidence extraction methods:")
        for r in confidence:
            print(f" - {r}")
        return
    if args.confidence:
        if args.confidence not in confidence:
            print(f"Error: Test '{args.confidence}' not found.")
            return
        else:
            confidence = [args.confidence]
    else:
        confidence = ["single-generation"]

    login(token=os.environ["HF_TOKEN"])
    hf_models = [k for k, v in models.items() if v == None]
    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return
    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        elif args.model in hf_models:
            # setup HF model
            
            local_dir = snapshot_download(repo_id=args.model)
            if confidence[0] in logits_sampler:
                model = LlamaForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                tokeniser = AutoTokenizer.from_pretrained(local_dir)
                models = {args.model: HFSamplerTokeniser(model=model, tokeniser=tokeniser, model_name=args.model, max_new_tokens=256)}
            else:
                pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
                terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                models = {args.model: HFSamplerPipeline(pipeline=pipeline, terminators=terminators, model_name=args.model, max_tokens=256)}
        else:
            models = {args.model: models[args.model]}

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples, confidence_type=confidence[0])
            case "math":
                pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
                terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                equality_checker = HFSamplerPipeline(pipeline=pipeline, terminators=terminators, model_name=args.model, max_tokens=1024)
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples, confidence_type=confidence[0]
                )
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
                terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                grading_sampler = HFSamplerPipeline(pipeline, terminators, model_name=args.model, max_tokens=1024)
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in tests
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
            with open(report_filename, "w", encoding="utf-8") as fh:
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
            {"eval_name": eval_name, "model_name": model_name.split("/")[-1], "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
