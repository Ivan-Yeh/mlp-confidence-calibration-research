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
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

from .ece import ece_equal_width, ece_equal_weight
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from .semantic_confidence import single_generation_confidence, empirical_semantic_confidence, likelihood_based_semantic_confidence, mean_likelihood_based_semantic_confidence, bayesian_semantic_confidence

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
    def __init__(self, num_examples: int | None = None, language: str = "EN-US"):
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.ece_dfs: list[pandas.DataFrame] = [pandas.DataFrame(columns=['question', 'answer', 'predicted_answer', 'confidence', 'accuracy'])] * 5

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row) + " Respond starting with the word 'Answer:' followed by only one of ABCD and the option", role="user"
                )
            ]

            print(format_multichoice_question(row))
            response_sample = [sampler(prompt_messages) for _ in range(20)]
            confidence_methods: list[function] = [single_generation_confidence, 
                                                                      empirical_semantic_confidence, 
                                                                      likelihood_based_semantic_confidence, 
                                                                      mean_likelihood_based_semantic_confidence, 
                                                                      bayesian_semantic_confidence]
            model_name="all-MiniLM-L6-v2"
            distance_threshold = 0.15
            lnll_lst = [(x)[1] for x in response_sample]
            response_list = [x[0] for x in response_sample]
            embeddings = SentenceTransformer(model_name).encode(response_list)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
            labels = clustering.fit_predict(embeddings)

            for i, method in enumerate(confidence_methods):
                response_text, confidence = method(lnll_lst, response_list, labels)
                
                response_text = normalize_response(response_text)
                print(response_text)
                extracted_answer = None
                for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                    match = re.search(regex, response_text)
                    if match:
                        extracted_answer = normalize_extracted_answer(match.group(1))
                        break
                score = 1.0 if extracted_answer == row["Answer"] else 0.0

                new_row = pandas.DataFrame({"question": [format_multichoice_question(row)], 
                                            "answer": [row["Answer"]], 
                                            "predicted_answer": [extracted_answer], 
                                            "confidence": [confidence], 
                                            "accuracy": [score]})
                print(extracted_answer)
                self.ece_dfs[i] = pandas.concat([self.ece_dfs[i], new_row], ignore_index=True)
            
            
            response_text, confidence = bayesian_semantic_confidence(lnll_lst, response_list, labels)
            response_text = normalize_response(response_text)
            extracted_answer = None
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                match = re.search(regex, response_text)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )
        
        results = common.map_with_progress(fn, self.examples)

        print(self.ece_dfs)
        print("##################")
        print("TEST RESULTS") 
        with open("tmp/mmlu_results.txt", "w") as f:
            for i, name in enumerate(["SGC", "ESC", "LSC", "MLSC", "BSC"]):
                print("Accuracy", name, self.ece_dfs[i]["accuracy"].mean() ,file=f)
                print("Accuracy", name, self.ece_dfs[i]["accuracy"].mean())
                self.ece_dfs[i].to_csv(f"tmp/ece_{name}.csv", index=False)
        print("##################")
        print()
        for i, name in enumerate(["SGC", "ESC", "LSC", "MLSC", "BSC"]):
            print("EQUAL WEIGHT ECE", name)
            print(ece_equal_weight(self.ece_dfs[i], file_path=f"tmp/equal_weight_ece_{name}.csv"))
            print("EQUAL WIDTH ECE", name)
            print(ece_equal_width(self.ece_dfs[i], file_path=f"tmp/equal_width_ece_{name}.csv"))
            print("##################")
            print()
        return common.aggregate_results(results)
