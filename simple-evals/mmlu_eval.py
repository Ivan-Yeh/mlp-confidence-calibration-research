"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
import time

import pandas

from . import common
from .common import (
    HTML_JINJA,
    format_multichoice_question
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .confidence_extractor import mmlu_vanilla_confidence, mmlu_cot_confidence
from .ece import ece_equal_width, ece_equal_weight

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
    def __init__(self, num_examples: int | None = None, language: str = "EN-US", confidence_type: str = "CoT"):
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.confidence_type = confidence_type
        self.outputs: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence'])
        self.ece_df: pandas.DataFrame = pandas.DataFrame(columns=['prompt', 'question', 'answer', 'response_raw', 'response_extracted', 'confidence', 'score'])

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]

            match self.confidence_type:
                case "verbal-vanilla": response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = mmlu_vanilla_confidence(sampler, prompt_messages, row)
                case "verbal-cot": response_text, extracted_answer, confidence, score, new_ece_row, new_output_row = mmlu_cot_confidence(sampler, prompt_messages, row)

            self.ece_df = pandas.concat([self.ece_df, new_ece_row], ignore_index=True)
            self.outputs = pandas.concat([self.outputs, new_output_row], ignore_index=True)

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
        
        # Calculate ECE
        ece_equal_weight(self.ece_df)
        ece_equal_width(self.ece_df)

        self.outputs.to_csv(f"tmp/{sampler.model_name.split('/')[-1]}_mmlu_{self.confidence_type}_outputs_{time.time()}.csv")
        
        return common.aggregate_results(results)
