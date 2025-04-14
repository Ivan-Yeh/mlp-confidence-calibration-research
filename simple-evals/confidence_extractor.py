from .sampler.hf_sampler import HFSampler
from .common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
import re
import pandas

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

def vanilla_confidence(sampler: HFSampler, prompt_messages: str, row) -> tuple:
    sampler.system_message = VANILLA_PROMPT

    response_text = normalize_response(sampler(prompt_messages))

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

    new_row = pandas.DataFrame({"question": [format_multichoice_question(row)], 
        "answer": [row.get("Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row, new_row)
    

def cot_confidence(sampler: HFSampler, prompt_messages: str, row) -> tuple:
    sampler.system_message = COT_PROMPT

    response_text = normalize_response(sampler(prompt_messages))

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

    new_row = pandas.DataFrame({"question": [format_multichoice_question(row)], 
        "answer": [row.get("Answer")], "response_raw": [response_text], "response_extracted": [extracted_answer], "confidence": [confidence], "score": [score]})

    return (response_text, extracted_answer, confidence, score, new_row, new_row)