def vanilla_prompt():
    return """Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
“‘Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%”’
Only the answer and confidence, don’t give me the explanation.
Question:[Specific Question Here]
Now, please answer this question and provide your confidence level."""

def cot_prompt():
    return """Read the question, analyze step by step, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
“‘Explanation: [insert step-by-step analysis here]
Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%”’
Only give me the reply according to this format, don’t give me any other words.
Question:[Specific Question Here]
Now, please answer this question and provide your confidence level. Let’s think it step by step.
"""
# def self_probing_prompt():
#     return """Question: [The specific question]
# Possible Answer: [The answer candidates]
# Q: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:
# “‘Confidence: [the probability of answer {{answer}} to be correct, not the one you think correct, please only include the numerical number]”’
# """

# def multi_step_prompt():
#     return """Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then derive your final answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.
# Use the following format to answer:
# “‘Step 1: [Your reasoning], Confidence: [ONLY the confidence valu that this step is correct]%
# ...
# Step K: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%
# Final Answer and Overall Confidence (0-100): [ONLY the answer type; not a complete sentence], [Your confidence value]%”’"""

# def top_k_prompt():
#     return """Provide your k best guesses and the probability that each is correct (0% to 100%) for the following question. Give ONLY the task output description of your guesses and probabilities, no other words or explanation. For example:
# G1: <ONLY the task output description of first most likely guess; not a complete sentence, just the guess!> P1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!>
# ...
# Gk: <ONLY the task output description of k-th most likely guess> Pk: <ONLY the probability that Gk is correct, without any extra commentary whatsoever; just the probability!>
# """

def vanilla_confidence(response_text: str):
    ans_and_conf = response_text.split('\n')[0].replace("%", "").replace(")", ":").replace(",", ":")
    try:
        conf = ans_and_conf.split(':')[-1]
        conf = float(conf)/100
        if conf <= 1:
            return conf
        return 0.0
    except:
        return 0.0

def cot_confidence(response_text: str):
    ans_and_conf = response_text.split('\n')[-1].split(':')[-1].strip(' ').replace("%", "").replace(',', ":").replace(')', ":").replace(' ', ':')
    try:
        conf = ans_and_conf.split(':')[-1]
        conf = float(conf)/100
        if conf <= 1:
            return conf
        return 0.0
    except:
        if len(conf) > 1 and conf[1:].isnumeric() and conf[0].isalpha():
            conf = float(conf[1:])/100
            if conf <= 1:
                return conf
        return 0.0