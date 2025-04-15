"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import time
import pandas
from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from .confidence_extractor import *
from .ece import ece_equal_width, ece_equal_weight

class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 1,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        confidence_type = "single-generation"
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.confidence_type = confidence_type[0] if isinstance(confidence_type, list) else confidence_type
        self.outputs: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence'])
        self.ece_df: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence', 'score'])

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict) + " Response starting with the word 'Answer:' followed by only one of ABCD and the option", role="user"
                )
            ]
            print(format_multichoice_question(choices_dict))

            # prepare (response_text, extracted_answer, confidence, score, new_ece_row, new_output_row) with different confidence types
            #---------------------------------------------------------------------------------------------------------------------
            sampling = 2

            match self.confidence_type:
                
                case "empirical-semantic":
                    response_with_conf = [sampler(prompt_messages) for _ in range(sampling)]
                    extracted_answers = [gpqa_regex_extract_response(text[0]) for text in response_with_conf]
                    response_texts, lnll_lst, labels = get_mcq_clusters(response_with_conf, "gpqa")
                    response_text, confidence = empirical_semantic_confidence(lnll_lst, response_texts, labels)
                    extracted_answer = gpqa_regex_extract_response(response_text)
                    score = 1.0 if extracted_answer == correct_answer else 0.0
                    new_output_row = pandas.DataFrame({'prompt': [prompt_messages] * sampling, 'question': [format_multichoice_question(choices_dict)] * sampling, 'answer': [correct_answer] * sampling, 'response_raw': response_texts, 'response_extracted': extracted_answers, 'confidence': lnll_lst})
                    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(choices_dict)], 'answer':[correct_answer], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})
                    
                case "single-generation":
                    response_with_conf = sampler(prompt_messages)
                    response_text, confidence = response_with_conf 
                    extracted_answer = gpqa_regex_extract_response(response_text)
                    score = 1.0 if extracted_answer == correct_answer else 0.0
                    new_output_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(choices_dict)], 'answer':[correct_answer], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence]})
                    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(choices_dict)], 'answer':[correct_answer], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})
                
                case "verbal-vanilla": 
                    response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = gpqa_vanilla_confidence(sampler, prompt_messages, row)

                case "verbal-cot": 
                    response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = gpqa_cot_confidence(sampler, prompt_messages, row)
                    
                case _:
                    raise Exception(f"Unrecognized confidence type: {self.confidence_type}")
                
            self.ece_df = pandas.concat([self.ece_df, new_ece_row], ignore_index=True)
            self.outputs = pandas.concat([self.outputs, new_output_row], ignore_index=True)
            #---------------------------------------------------------------------------------------------------------------------
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
                confidence=confidence,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = common.map_with_progress(fn, self.examples)
        self.outputs.to_csv(f"tmp/{sampler.model_name}_gpqa_{self.confidence_type}_outputs.csv")
        self.ece_df.to_csv(f"tmp/{sampler.model_name}_gpqa_{self.confidence_type}_unprocessed_ece.csv")
        print(self.outputs)
        print(ece_equal_weight(self.ece_df, 10, f"tmp/{sampler.model_name}_gpqa_{self.confidence_type}_equal_weight_ece.csv"))
        print(ece_equal_width(self.ece_df, 10, f"tmp/{sampler.model_name}_gpqa_{self.confidence_type}_equal_width_ece.csv"))
        return common.aggregate_results(results)
