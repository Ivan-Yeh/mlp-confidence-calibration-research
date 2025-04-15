"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import pandas

from . import common
from .common import (
    HTML_JINJA,
    format_multichoice_question
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .ece import ece_equal_width, ece_equal_weight
from .semantic_confidence import *
from .confidence_extractor import *
import time

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(self, num_examples: int | None = None, language: str = "EN-US", confidence_type = "single-generation"):
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.confidence_type = confidence_type[0] if isinstance(confidence_type, list) else confidence_type
        self.outputs: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence'])
        self.ece_df: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence', 'score'])


    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row) + " Response starting with the word 'Answer:' followed by only one of ABCD and the option", role="user"
                )
            ]
            print(format_multichoice_question(row))
            response_text = ""
            extracted_answer = ""
            confidence = ""

            # Call confidence extracting function & return (response_text, extracted_answer, confidence, score, new_ece_row, new_output_row)
            #---------------------------------------------------------------------------------------------------------------------
            sampling = 2

            match self.confidence_type:
                
                case "single-generation":
                    response_with_conf = sampler(prompt_messages)
                    response_text, confidence = response_with_conf 
                    extracted_answer = mmlu_regex_extract_response(response_text)
                    score = 1.0 if extracted_answer == row["Answer"] else 0.0
                    new_output_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(row)], 'answer':[row["Answer"]], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence]})
                    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(row)], 'answer':[row["Answer"]], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})

                case "empirical-semantic":
                    response_with_conf = [sampler(prompt_messages) for _ in range(sampling)]
                    extracted_answers = [mmlu_regex_extract_response(text[0]) for text in response_with_conf]
                    response_texts, lnll_lst, labels = get_mcq_clusters(response_with_conf, "mmlu")
                    response_text, confidence = empirical_semantic_confidence(lnll_lst, response_texts, labels)
                    extracted_answer = mmlu_regex_extract_response(response_text)
                    score = 1.0 if extracted_answer == row["Answer"] else 0.0
                    new_output_row = pandas.DataFrame({'prompt': [prompt_messages] * sampling, 'question': [format_multichoice_question(row)] * sampling, 'answer': [row["Answer"]] * sampling, 'response_raw': response_texts, 'response_extracted': extracted_answers, 'confidence': lnll_lst})
                    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [format_multichoice_question(row)], 'answer':[row["Answer"]], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})
                
                case "verbal-vanilla": 
                    response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = mmlu_vanilla_confidence(sampler, prompt_messages, row)
                
                case "verbal-cot": 
                    response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = mmlu_cot_confidence(sampler, prompt_messages, row)
                    
                case _:
                    raise Exception(f"Unrecognized confidence type: {self.confidence_type}")
                
            self.ece_df = pandas.concat([self.ece_df, new_ece_row], ignore_index=True)
            self.outputs = pandas.concat([self.outputs, new_output_row], ignore_index=True)
        #---------------------------------------------------------------------------------------------------------------------
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                confidence=confidence,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )
        
        results = common.map_with_progress(fn, self.examples)
        self.outputs.to_csv(f"tmp/{sampler.model_name}_mmlu_{self.confidence_type}_outputs.csv")
        self.ece_df.to_csv(f"tmp/{sampler.model_name}_mmlu_{self.confidence_type}_unprocessed_ece.csv")
        print(self.outputs)
        print(ece_equal_weight(self.ece_df, 10, f"tmp/{sampler.model_name}_mmlu_{self.confidence_type}_equal_weight_ece.csv"))
        print(ece_equal_width(self.ece_df, 10, f"tmp/{sampler.model_name}_mmlu_{self.confidence_type}_equal_width_ece.csv"))

        return common.aggregate_results(results)
