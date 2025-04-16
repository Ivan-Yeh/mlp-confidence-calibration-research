import re
from .semantic_confidence import *
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
import pandas
from .sampler.other_samplers import *
from .common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    ANSWER_PATTERN_MULTICHOICE,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)


# mcq_regex_extractors = {"mmlu": mmlu_regex_extract_response, "gpqa": gpqa_regex_extract_response}

# ## all functions to return a tuple (response_text, extracted_answer, confidence)
# def single_generation_mcq_confidence_extractor(sampler: SamplerBase, prompt_messages, question, answer_key, test="mmlu") -> tuple:
#     response_with_conf = sampler(prompt_messages)
#     response_text, confidence = response_with_conf 
#     extracted_answer = mcq_regex_extractors[test](response_text)
#     score = 1.0 if extracted_answer == answer_key else 0.0

#     new_output_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence]})
#     new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})

#     return response_text, extracted_answer, confidence, score, new_ece_row, new_output_row



# def empirical_semantic_mcq_confidence_extractor(sampler: SamplerBase, prompt_messages, question, answer_key, sampling = 20, test = "mmlu") -> tuple:
#     response_with_conf = [sampler(prompt_messages) for _ in range(sampling)]
#     extracted_answers = [mcq_regex_extractors[test](text[0]) for text in response_with_conf]
#     response_texts, lnll_lst, labels = get_mcq_clusters(response_with_conf, test)
#     response_text, confidence = empirical_semantic_confidence(lnll_lst, response_texts, labels)
#     extracted_answer = mcq_regex_extractors[test](response_text)
#     score = 1.0 if extracted_answer == answer_key else 0.0

#     new_output_row = pandas.DataFrame({'prompt': [prompt_messages] * sampling, 'question': [question] * sampling, 'answer': [answer_key] * sampling, 'response_raw': response_texts, 'response_extracted': extracted_answers, 'confidence': lnll_lst})
#     new_ece_row = pandas.DataFrame({'prompt': [prompt_messages], 'question': [question], 'answer':[answer_key], 'response_raw': [response_text], 'response_extracted': [extracted_answer], 'confidence': [confidence], 'score': [score]})

#     return response_text, extracted_answer, confidence, score, new_ece_row, new_output_row


VANILLA_PROMPT = """Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
“‘Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%”’
Only the answer and confidence, don’t give me the explanation.
Question:[Specific Question Here]
Now, please answer this question and provide your confidence level."""

COT_PROMPT = """Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
“‘Explanation: [insert step-by-step analysis here]
Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%”’
Only give me the reply according to this format, don’t give me any other words.
Question:[Specific Question Here]
Now, please answer this question and provide your confidence level. Let’s think it step by step.
"""

def regex_extract_reponse(response_text) -> str | None:
    response_text = normalize_response(response_text)
    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    return extracted_answer

def mmlu_vanilla_confidence(sampler: SamplerBase, prompt_messages: str, row) -> tuple:
    sampler.system_message = VANILLA_PROMPT

    response_text, _, log_probs = sampler(prompt_messages)
    response_text = normalize_response(response_text)

    confidence = 0.0
    ans_and_conf = response_text.split('\n')[0].replace("%", "").replace(")", ":").replace(",", ":")
    try:
        confidence = ans_and_conf.split(':')[-1]
        confidence = float(confidence)/100
        if confidence > 1:
            confidence = 0.0
    except:
        confidence = 0.0

    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    if not extracted_answer: # 'Answer' is not given in the response
        for line in response_text.split('\n'):
            answer_candidate = line.replace(':', ')').split(')')[0]
            if len(answer_candidate) == 1: #only one answer is provided in the response
                extracted_answer = answer_candidate

    score = 1.0 if extracted_answer == row["Answer"] else 0.0

    new_row = pandas.DataFrame({"prompt": [prompt_messages], "question": [format_multichoice_question(row)], 
        "answer": [row.get("Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "logprobs": [log_probs], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row, new_row.drop(columns=["score"]))
    
def mmlu_cot_confidence(sampler: SamplerBase, prompt_messages: str, row) -> tuple:
    sampler.system_message = COT_PROMPT

    response_text, _, log_probs = sampler(prompt_messages)
    response_text = normalize_response(response_text)
    confidence = 0.0
    ans_and_conf = response_text.split('\n')[-1].split(':')[-1].strip(' ').replace("%", "").replace(',', ":").replace(')', ":").replace(' ', ':')
    try:
        confidence = ans_and_conf.split(':')[-1]
        confidence = float(confidence)/100
        if confidence > 1:
            confidence = 0.0
    except:
        if len(confidence) > 1 and confidence[1:].isnumeric() and confidence[0].isalpha():
            confidence = float(confidence[1:])/100
            if confidence > 1:
                confidence = 0.0
        else:
            confidence = 0.0

    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    if not extracted_answer: # 'Answer' is not given in the response
        for line in response_text.split('\n'):
            answer_candidate = line.replace(':', ')').split(')')[0]
            if len(answer_candidate) == 1: #only one answer is provided in the response
                extracted_answer = answer_candidate

    score = 1.0 if extracted_answer == row["Answer"] else 0.0

    new_row = pandas.DataFrame({"prompt": [prompt_messages], "question": [format_multichoice_question(row)], 
        "answer": [row.get("Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "logprobs": [log_probs], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row, new_row)

def gpqa_vanilla_confidence(sampler: SamplerBase, prompt_messages: str, row, correct_answer) -> tuple:
    sampler.system_message = VANILLA_PROMPT

    response_text, _, log_probs = sampler(prompt_messages)
    response_text = normalize_response(response_text)

    confidence = 0.0
    ans_and_conf = response_text.split('\n')[0].replace("%", "").replace(")", ":").replace(",", ":")
    try:
        confidence = ans_and_conf.split(':')[-1]
        confidence = float(confidence)/100
        if confidence > 1:
            confidence = 0.0
    except:
        confidence = 0.0

    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    if not extracted_answer: # 'Answer' is not given in the response
        for line in response_text.split('\n'):
            answer_candidate = line.replace(':', ')').split(')')[0]
            if len(answer_candidate) == 1: #only one answer is provided in the response
                extracted_answer = answer_candidate

    score = 1.0 if extracted_answer == correct_answer else 0.0

    new_row = pandas.DataFrame({"prompt": [prompt_messages], "question": [row["Question"]], 
        "answer": [row.get("Correct Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "logprobs": [log_probs], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row,  new_row.drop(columns=["score"]))

def gpqa_cot_confidence(sampler: SamplerBase, prompt_messages: str, row, correct_answer) -> tuple:
    sampler.system_message = COT_PROMPT
    response_text, _, log_probs = sampler(prompt_messages)
    response_text = normalize_response(response_text)

    confidence = 0.0
    ans_and_conf = response_text.split('\n')[-1].split(':')[-1].strip(' ').replace("%", "").replace(',', ":").replace(')', ":").replace(' ', ':')
    try:
        confidence = ans_and_conf.split(':')[-1]
        confidence = float(confidence)/100
        if confidence > 1:
            confidence = 0.0
    except:
        if len(confidence) > 1 and confidence[1:].isnumeric() and confidence[0].isalpha():
            confidence = float(confidence[1:])/100
            if confidence > 1:
                confidence = 0.0
        else:
            confidence = 0.0

    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    if not extracted_answer: # 'Answer' is not given in the response
        for line in response_text.split('\n'):
            answer_candidate = line.replace(':', ')').split(')')[0]
            if len(answer_candidate) == 1: #only one answer is provided in the response
                extracted_answer = answer_candidate

    score = 1.0 if extracted_answer == correct_answer else 0.0

    new_row = pandas.DataFrame({"prompt": [prompt_messages], "question": [row["Question"]], 
        "answer": [row.get("Correct Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "logprobs": [log_probs], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row,  new_row.drop(columns=["score"]))
