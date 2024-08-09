import jsonlines
import json
import copy
import re

PROMPT_DICT = {
    "rag_prompt_no_input": (
        "<|Instruction|>:\n{instruction}</eoi>\n\n"
    ),
    "intent_prompt_no_input": (
        "<|Instruction|>:\n{instruction}</eoi>\n\n<|Intent|>:\n"
    ),
    "REPLUG_prompt_no_input": (
        "[INST]Knowledge: {paragraph}\n\nQuestion: {instruction}\n\nAnswer: [/INST]"
    ),
    "rrg_prompt_input": (
        "<|Instruction|>:\n{instruction}</eoi>\n\n<paragraph>{paragraph}</paragraph>\n\n<|Assistant|>:\n"
    ),
    "rgg_prompt_input": (
        "<|Reference|>:\n{paragraph}\n\n<|Instruction|>:\n{instruction}\n\n<|Input|>:\n{input}\n\n<|Assistant|>:\n"
    ),
    "rrg_prompt_no_input": (
        "<|Instruction|>:\n{instruction}</eoi>\n\n<|Assistant|>:\n"
    ),
    "rrg_prompt_no_input_retrieval": (
        "<|Reference|>:\n{paragraph}\n\n<|Instruction|>:\n{instruction}\n\n<|Assistant|>:\n"
    ),
    "alpaca_prompt_input": (
        "<|Instruction|>:\n{instruction}\n\n<|Assistant|>:\n"
    ),
    "short_prompt_input": (
        "<|Instruction|>:\n{instruction}\n\n<|Generator|>:\n"
    ),
    "alpaca_no_input_retrieval": (
        "<|Instruction|>:\n{instruction}\n\nParagraph:\n{paragraph}\n\n<|Assistant|>:\n"
    ),
    "prompt_input": (
        "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n"
    ),
    "Mistral_prompt_no_input": (
        "<s>[INST] Instruction:\n{instruction}\n\nResponse:\n [/INST]"
    ),

    "prompt_no_input": (
        "Instruction:\n{instruction}\n\nResponse:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Based on the provided Paragraphs, rite a response that appropriately completes the request.\n\n"
        "Paragraph:\n{paragraph}\n\nInstruction:\n{instruction}\n\nResponse:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]Below is an instruction that describes a task. "
        "Based on the provided Paragraphs, rite a response that appropriately completes the request.\n\n"
        "Paragraph:\n{paragraph}\n\nInstruction:\n{instruction}\n\n}[/INST]"
    ),
    # "llama_chat_prompt_retrieval": (
    #     "[INST]{paragraph}\n{instruction}[/INST]"
    # ),
    "vicuna_prompt": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        " USER: {instruction} "
        "ASSISTANT: "
    ),
    "Mistral_prompt": (
        "[INST]{instruction}[/INST] "
    ),
    "INTERACT_prompt": (
        "[INST]Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided "
        "search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic "
        "tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least "
        "one document and at most three documents in each sentence. If multiple documents support the sentence, "
        "only cite a minimum sufficient subset of the documents. \n\n"
        "{paragraph}\n\nQuestion: {instruction}\n\nAnswer:[/INST]"
    ),
    "1INTERACT_prompt": (
        "[INST]Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided "
        "search results and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual "
        "claim.\nYou are provided summaries/snippets of the search results. You can use \"Search: key words\" to "
        "check the most relevant document's full text and use \"Output:\" to output a sentence in the answer. In the "
        "answer, cite properly by using [1][2][3]. Cite at least one document and at most three documents in each "
        "sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the "
        "documents. Use \"End\" to end the generation. \n\n "
        "{paragraph}\n\nQuestion: {instruction}\n\nAnswer:[/INST]"
    ),

    "agent_prompt_1": (
        "Given the follow passage and the question, extract a useful span from the passage that can answer the question. "
        "Resolve all the coreference issues to make the extracted span understandable standalone. If the passage is not helpful for answering the question, return irrelevant. "
        "\n\nParagraph: {paragraph}\n\nQuestion: {instruction}\n\nExtracted span:"
    ),
    "agent_prompt_2": (
        "USER: {instruction}\n"
        "ASSISTANT: "
    ),
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "arc_c": "Given some answer candidates, choose the best answer choice.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instfance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = item["instruction"] + choices
    # processed_instruction = "Instruction:\n"+ instruction + "\nInput:\n" + item["instruction"] + choices

    # processed_instruction = instruction + "\n" + item["instruction"] + choices

    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower()  in ["yes", "no"]:
        final_output = "true" if output == "yes" else "no"
    if task == "fever" and output in ["correct", "incorrect"]:
        final_output = "true" if output == "correct" else "no"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    # if task == "fever" and "true." in output.lower():
    #     final_output = "true"
    # if task == "fever" and "false." in output.lower() :
    #     final_output = "false"
    # if task == "arc_c" and "best answer choice is" in output:
    #     final_output = output.split("best answer choice is")[1]
    # if task == "arc_c" and "correct answer is" in output:
    #     final_output = output.split("correct answer is")[1]
    if final_output is None:
        return output
    else:
        return final_output
