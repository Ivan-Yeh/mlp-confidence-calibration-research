import re
from .semantic_confidence import *
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
import pandas


mcq_regex_extractors = {"mmlu": mmlu_regex_extract_response, "gpqa": gpqa_regex_extract_response}

## all functions to return a tuple (response_text, extracted_answer, confidence)
def single_generation_mcq_confidence_extractor(sampler: SamplerBase, prompt_messages, question, answer_key, test="mmlu") -> tuple:
    response_with_conf = sampler(prompt_messages)
    response_text, confidence = response_with_conf 
    extracted_answer = mcq_regex_extractors[test](response_text)
    score = 1.0 if extracted_answer == answer_key else 0.0

    new_output_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence]})
    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})

    return response_text, extracted_answer, confidence, score, new_ece_row, new_output_row



def empirical_semantic_mcq_confidence_extractor(sampler: SamplerBase, prompt_messages, question, answer_key, sampling = 20, test = "mmlu") -> tuple:
    response_with_conf = [sampler(prompt_messages) for _ in range(sampling)]
    extracted_answers = [mcq_regex_extractors[test](text[0]) for text in response_with_conf]
    response_texts, lnll_lst, labels = get_mcq_clusters(response_with_conf, test)
    response_text, confidence = empirical_semantic_confidence(lnll_lst, response_texts, labels)
    extracted_answer = mcq_regex_extractors[test](response_text)
    score = 1.0 if extracted_answer == answer_key else 0.0

    new_output_row = pandas.DataFrame({'prompt': [prompt_messages] * sampling, 'question': [question] * sampling, 'answer': [answer_key] * sampling, 'response_raw': response_texts, 'response_extracted': extracted_answers, 'confidence': lnll_lst})
    new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})

    return response_text, extracted_answer, confidence, score, new_ece_row, new_output_row