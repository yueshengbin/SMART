import openai
import pandas as pd
import os
import argparse
import json
from collections import Counter
import jsonlines
import random



def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_name', type=str)
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    examples = []

    for filename in os.listdir(args.input_folder_name):
        if filename.endswith(".json"):
            file_path = os.path.join(args.input_folder_name, filename)
            examples += json.load(open(file_path))


    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)

    for idx, item in enumerate(examples):
        text = []
        fact = []
        re_fact = []
        for idx, context in enumerate(item["retrieval"]):
            id = str(idx + 1)
            if "title" in context:
                if context["title"]+"\n" in context["text"]:
                    text.append("["+id+"]" + " " + context["text"])
                else:
                    context["text"] = context["title"] + "\n" + context["text"]
                    text.append("[" + id + "]" + " " + context["text"])
                if context["title"]+"\n" in context["fact"]:
                    pass
                else:
                    context["fact"] = context["title"] + "\n" + context["fact"]

            else:
                text.append("[" + id + "]" + " " + context["text"])

            if context['relevant'] == "ture":
                re_fact.append("["+id+"]")
                fact.append("[Relevant]"+": "+"["+id+"]" + " " + context["fact"])

            elif context["relevant"] == "false":
                fact.append("[Irrelevant]" + ": " + "["+id+"]" + " Lacking Supporting Facts")

        all_text = "\n".join(text)
        all_fact = "\n".join(fact)
        all_fact_cite = " ".join(re_fact)
        if "[" in all_fact_cite:
            Cite="\n" + "[Cite]: " + all_fact_cite
        else:
            Cite = ""

        item['text'] = "<retrieval>" + all_text + "</retrieval>" + "\n\n"
        item['filter'] = "<|Locator|>:\n" + all_fact + "</eol>" + "\n\n"
        item['output'] = "<|Generator|>:\n" + item['output']
        item['intent'] = "<|Reconstructor|>:\nSearch("+ item['rewrite'] +")"+ "</eor>" + "\n\n"
        output =  item['intent']+ item['text'] + item['filter']+item['output']+Cite

        if item["input"] == "":
            item["instruction"] = item["instruction"] + "</eoi>" + "\n\n"
        else:
            item["input"] = item["input"] + "</eoi>" + "\n\n"

        data = {
                "id": item['id'],
                "dataset_name": item["dataset_name"],
                "instruction": item['instruction'],
                "input":  item['input'],
                'output': output+"</eog>",
                        }
        result_list.append(data)
    task_types = Counter([item["dataset_name"]
                          for item in result_list if "dataset_name" in item])

    print(Counter(task_types))
    with open(args.output_file_name, "w") as outfile:
        print(len(result_list))
        json.dump(result_list, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
