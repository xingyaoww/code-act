'''
This script is adopted from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py
AND https://github.com/THUDM/AgentTuning/blob/main/eval_general/eval_gsm8k_tgi.py
'''
import re
import os
import time
import openai
import numpy as np
import traceback
import argparse
import jsonlines
import datetime
from datasets import load_from_disk, load_dataset
import json
import multiprocessing
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

Q1 = '''Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?'''
A1 = '''Let's think step by step.
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20'''

Q2 = '''Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?'''
A2 = '''Let's think step by step.
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817'''

Q3 = '''Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?'''
A3 = '''Let's think step by step.
The total ratio representing the slices of pizza that Buzz bought is 5+8=13.
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices.
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28.
The answer is 28'''

Q4 = '''Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?'''
A4 = '''Let's think step by step.
He makes 20*40=$800 per week.
He used to make 16*25=$400 per week.
So his raise was 800-400=$400 per week.
So he makes 400*52=$20,800 per year more.
The answer is 20800'''

Q5 = '''Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?'''
A5 = '''Let's think step by step.
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats.
Each student will receive 80 / 20 = 4 sweet treats.
The answer is 4'''

Q6 = '''Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot's vehicles in all?'''
A6 = '''Let's think step by step.
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale.
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot's vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84'''

Q7 = '''Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?'''
A7 = '''Let's think step by step.
Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters.
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes.
When she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts.
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes.
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing.
The answer is 15'''

Q8 = '''Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?'''
A8 = '''Let's think step by step.
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350'''

shots = [(Q1, A1), (Q2, A2), (Q3, A3), (Q4, A4),
         (Q5, A5), (Q6, A6), (Q7, A7), (Q8, A8)]


def doc_to_messages(doc):
    ret = [
        {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Please provide a detailed step-by-step solution for the following grade school math question. "},
        {"role": "assistant", "content": "Ok."},
    ]
    # Add few shots examples
    for q, a in shots:
        ret.append({"role": "user", "content": q})
        ret.append({"role": "assistant", "content": a})
    
    # Add the question
    ret.append({"role": "user", "content": f"\nQuestion: {doc['question']}"})
    return ret


def clean_answer(text):
    text = text.split("Question:")[0]
    return text


def generate_sample(messages, model) -> str:
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
                n=1,
                stop=["Question:"],
                api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_KEY"),
            )
            resp_msg = response["choices"][0]["message"]
            assert resp_msg["role"] == "assistant"
            return clean_answer(resp_msg["content"])
        except Exception as e:
            traceback.print_exc()
            time.sleep(1)
            continue


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r'\d+', completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HF checkpoint.')
    parser.add_argument("-f", "--sample-input-file", type=str, default="data/eval/gsm8k")
    parser.add_argument("-o", "--sample-output-file",
                        type=str, default="gsm8k_res.jsonl")
    parser.add_argument("-m", "--model", type=str,
                        default="code-act-agent", help="Model name on OpenAI API")
    args = parser.parse_args()

    dataset = load_from_disk(args.sample_input_file)
    test = dataset["test"]

    def process_doc(doc, args):
        messages = doc_to_messages(doc)
        completion = generate_sample(messages, args.model)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        return doc, acc

    f_output = jsonlines.Writer(
        open(args.sample_output_file, 'w', encoding='utf-8'))
    acc_res = []
    executor = ProcessPoolExecutor(max_workers=16)
    partial_process_doc = partial(process_doc, args=args)

    with multiprocessing.Pool(processes=16) as pool:
        for doc, acc in tqdm(executor.map(partial_process_doc, test), total=len(test)):
            f_output.write(doc)
            acc_res.append(acc)
    f_output.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(args.sample_output_file + f".metrics.json", "w") as f:
        result = {}
        result["args"] = vars(args)
        result["acc"] = np.mean(acc_res)
        result["timestamp"] = timestamp
        result["acc_res"] = acc_res
        json.dump(result, f)
