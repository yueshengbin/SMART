import openai
import pandas as pd
import os
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
import jsonlines
import random
import time
from requests.exceptions import ConnectionError, Timeout, RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


KNOWLEDGE_INSTRUCTIONS = {
    "nq": "Please answer the following questions using the shortest possible response. For example, if the question asks 'What is the capital of France?'', you can simply reply with 'Paris'.",
    "fever": "Determine whether the following statement is true or false.",
    "wow": "You have been provided with a chat history between two agents, separated by new lines. Generate a response that is informative and engaging based on the latest message."}

PROMPT_DICT = {
    "multi": (
        "Given a multiple choice question, your task is to rewrite it into a better search query for a retrieval engine to answer the given question from some web document (e.g., Wikipedia),"
        " and state the most central search topic in the question. The rewrite process can not carry your own preferences for option answers.\n\n "
        "##\nQuestion: A student standing near a campfire feels warmer as the fire grows. Which process most likely transfers heat from the campfire to the student?\nA: conduction\nB: convection\nC: radiation\nD: transformation\n\n"
        "Search Intent: How does heat transfer from a campfire to a person standing nearby\n\n"
        "##\nQuestion: Stars are\nA: warm lights that float\nB: made out of nitrate\nC: great balls of gas burning billions of miles away\nD: lights in the sky\n\n"
        "Search Intent: What are stars made of \n\n"
        "##\nQuestion: The density of a red clay brick block is about 2000 kg/m^3. the density of air is 1 kg/m^3. which of the following has the least mass?\nA: 2 m^3 of brick\nB: 4 m^3 of brick\nC: 6000 m^3 of air\nD: 10,000 m^3 of air\n\n"
        "Search Intent: How to calculate the mass of an object given its density \n\n"
        "##\nQuestion: {input}\n\n"
        "Search Intent:"
    ),
    "hotpot": (
        "Given a ambiguous question, decompose it into one or more sub intents to help better retrieve the answer from external document on the web (e.g., Wikipedia). "
        "Split them with ';'. If it takes multiple hops to complete, use # to represent the answer to the previous question\n "
        "##\nQuestion: When is Pan Shu's husband's birthday?\n\n"
        "Search Intent: Panshu's husband; #1 birthday\n\n"
        "##\nQuestion: Californian rock band Lit recorded A Place in the Sun in 1995, but what's their best known song?\n\n"
        "Search Intent: Californian rock band Lit's most famous and popular songs\n\n"
        "##\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\n\n"
        "Search Intent: Arthur's Magazine publication year; First for Women publication year\n\n"
        "##\nQuestion: {input}\n\n"
        "Search Intent:"
    ),

    "NQ": (
        "Given a question, provide knowledge search intent to help better retrieve the answer from external document on the web (e.g., Wikipedia). Split the intent with ';' and write an explanation.\n "
        "##\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\n\n"
        "Search Intent: Arthur's Magazine publication year; First for Women publication year\n\n"
        "##\nQuestion: What is the legal age of marriage, without parental consent or other authorization, in Nebraska?\n\n"
        "Search Intent:  legal age of marriage in Nebraska without parental consent\n\n"
        "##\nQuestion: Californian rock band Lit recorded A Place in the Sun in 1995, but what's their best known song?\n\n"
        "Search Intent: Californian rock band Lit's most famous and popular songs\n\n"
        "##\nQuestion: {input}\n\n"
        "Search Intent:"
    ),

    "self": (
        "Given a instruction, provide clarified knowledge search intent to help better retrieve the answer from external document on the web (e.g., Wikipedia). "
        "If there are different intents, split them with ';'.\n "
        "##\nInstruction: Write a response that appropriately completes the request.\n\nInstruction:\nName some nations with a monarchy government.?\n\n"
        "Search Intent: nations with a monarchy government\n\n"
        "##\nInstruction: What are the most important values in life?\n\n"
        "Search Intent: The most important values in life\n\n"
        "##\nInstruction: Tell me two advantages of using AI assistants.\n\n"
        "Search Intent: Advantages of Artificial Intelligence Assistants\n\n"
        "##\nInstruction: Task: Come up with 5 example datasets that demonstrate the use of natural language processing.\n<|Input|>: <No input>\n\n"
        "Search Intent: natural language processing example dataset\n\n"
        "##\nInstruction: {instruction}\n\n"
        "Search Intent:"
    ),
    "asqa": (
        "Given a question, provide knowledge search intent to help better retrieve the answer from external document on the web (e.g., Wikipedia). and write an explanation.\n "
        "##\nQuestion: Who won the 2016 ncaa football national championship?\n\n"
        "Search Intent: the 2016 ncaa football national championship Winner\n\n"
        "##\nQuestion: Californian rock band Lit recorded A Place in the Sun in 1995, but what's their best known song?\n\n"
        "Search Intent: Californian rock band Lit's most famous and popular songs\n\n"
        "##\nQuestion: {input}\n\n"
        "Search Intent:"
    ),
    "wow": (
        "Given a question, answer and chat history separated by new lines, provide a knowledge search intent for the question to help better obtain answers from external documents on the web (e.g., Wikipedia). "
        "The intent needs to consider important and necessary contextual information from the history so that it can be fully understood.\n "
        "##\nHistory: What can you tell me about Gary Cherone?\nGary Francis Caine Cherone is an American rock singer and songwriter, known for his work as the lead vocalist of Extreme and for his short stint for Van Halen.\nDid Gary Cherone sing well?\nYes, Gary Cherone is also known for his work as the lead vocalist of the Boston rock group Extreme.\nWhat significant fact can you tell me about Gary Cherone that you liked?\nI like that Gary Cherone remained in contact and on good terms with Van Halen.\nWhat did Gary Cherone do after Van Halen?\nAfter his departure from Van Halen, Gary Cherone returned to Boston and put together a new project, Tribe of Judah.\n\n"
        "Question: Did they release any albums during that time frame?\n\n"
        "Answer: After Gary Cherone, Eddie Van Halen recovered from his hip surgery in November 1999, and no official statements were made by Van Halen and no music was released.\n\n"
        "Search Intent: Any album released by Eddie Van Halen after Gary Cherone left\n\n"
        "##\nHistory: Where does Call of the Dead take place\nIt takes place in a desolate area of the Siberian tundra next to the frozen ruins of a broken cargo ship and a old Soviet lighthouse.\nWhat is Call ForThe Dead's theme\nThe players are once again are tasked with surviving the never-ending onslaught of the Zombie hordes, while also dealing with a new, dangerous threat.\n\n"
        "Question: What is the genre\n\n"
        "Answer: The genre is crime, spy novel.\n\n"
        "Search Intent: the genre of Call For The Dead\n\n"
        "##\nHistory: {history}\n\n"
        "Question: {question}\n\n"
        "Answer: {output}\n\n"
        "Search Intent:"
    ),
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def chat_gpt(args, gen_content):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE"))
    max_retries = 3  # Set the number of retries
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": gen_content}]
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0.0
            )
            response_message = response.choices[0].message.content
            return response_message

        except (ConnectionError, Timeout) as e:
            print(f"Network error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            if attempt == max_retries - 1:
                raise  # Re-raise the last exception if all retries fail

        except RequestException as e:
            # Handle other types of requests exceptions
            print(f"An error occurred: {e}.")
            raise  # Re-raise the exception and exit the function
        except Exception as e:
            print(e)
    # Optional: Return a default message or handle the failed attempts
    return "Unable to get a response after several attempts."



def process_item(args, example, idx, result_list):
    input = PROMPT_DICT["wow"].format_map(example)
    results = chat_gpt(args, input)

    if idx % 20 == 0:
        print("Idx: {}".format(idx))
        print("Input: {}".format(example["instruction"]))
        print("Output: {}".format(example["output"]))
        print("Intent: {})".format(results))
        print("======================================================================\n")

    data = {"id": example['id'],
            "dataset_name": example["dataset_name"],
            "instruction": example["instruction"],
            "input": example["input"],
            'intent': results,
            'output': example['output'],
            'retrieval': [],
            }
    result_list.append(data)
    if idx % 20 == 0:
        print("saved output at {}".format(args.output_file_name + "_tmp"))
        with open(args.output_file_name + "_tmp", "w") as outfile:
            json.dump(result_list, outfile, indent=4, ensure_ascii=False)
    return data





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--multi_retrieval', action="store_true")
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    print(f"Used Model:{args.model_name}")
    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)

    task_types = Counter([item["dataset_name"]
                          for item in examples if "dataset_name" in item])

    print(Counter(task_types))
    futures=[]
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, example in tqdm(enumerate(examples)):
            future = executor.submit(process_item, args, example, idx, result_list)
            futures.append(future)
    print(f"Total list:{len(result_list)}")
    with open(args.output_file_name, "w", encoding='utf-8') as outfile:
            result = json.dumps(result_list, ensure_ascii=False, indent=4)
            outfile.write(result)


if __name__ == "__main__":
    main()
