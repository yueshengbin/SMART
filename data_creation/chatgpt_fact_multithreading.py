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


PROMPT_DICT = {
    "multi_choice": (
        "Given a question，answer and external passage, your job is to determine if the passage is relevant to the question and can provides useful information to obtan the answer. "
        "If the passage meets this requirement, respond with [Relevant] and extract a useful span from the passage as supporting fact."
        "The extracted spans consist of complete sentences to make the extracted span understandable standalone. \n\n"
        "###\nQuestion:What are the frameworks of analysis within which terrorism has been considered (as of 2020)?\nA: Competition between larger nations has resulted in some countries actively supporting terrorist groups to undermine the strength of rival states. Terrorist networks are extended patronage clubs maintained and paid for by their donor states and are conceptualised as being like state actors, to be dealt with using military force.\nB: Globalization has enabled the internationalization of terrorist activities by opening up their operational space, although coordination is still managed from a geographical base. This suggests that terrorist groups are nationally structured which means that terrorism cannot be considered in terms of a war to be defeated militarily without having serious implications on the indigenous population.\nC: Terrorism can be viewed as a problem to be resolved by military means (war on terrorism), by normal police techniques (terrorism as crime), or as a medical problem with underlying causes and symptoms (terrorism as disease).\nD: Terrorism is viewed as a criminal problem. The criminalization of terrorism has two important implications. Firstly, it suggests that terrorism can be eradicated - terrorists can be caught and brought to trial by normal judicial proceedings thereby removing the threat from society - and secondly, it suggests that preventative crime techniques are applicable to prevent its development.\n\n"
        "Answer: C: Terrorism can be viewed as a problem to be resolved by military means (war on terrorism), by normal police techniques (terrorism as crime), or as a medical problem with underlying causes and symptoms (terrorism as disease).\n\n"
        "Passage: discipline that is primarily concerned with looking at acts of terrorism by non-state actors. This is a very state-centric perspective which has a limited set of assumptions and narratives about the nature and cause of terrorism. Moreover, this becomes accepted as the general consensus at the macro, meso and micro levels of government and institutions, and is reflected in policy and the way the mainstream view terrorism. Traditional terrorism studies is also largely concerned with \"problem solving theory\", which looks at the world \"with the prevailing social and power relationships and the institutions into which they are organised, as the\n\n"
        "Rating: [Irrelevant]\n\n"
        "Extracted span: None.\n\n"
        "###\nQuestion:The following are features of Alzheimer's disease except:\nA: short-term memory loss.\nB: confusion.\nC: poor attention.\nD: drowsiness.\n\n"
        "Answer: D: drowsiness.\n\n"
        "Passage: most common symptoms of Alzheimer's disease are short-term memory loss and word-finding difficulties. People with Alzheimer's disease also have trouble with visual-spatial areas (for example, they may begin to get lost often), reasoning, judgment, and insight. Insight refers to whether or not the person realizes he/she has memory problems. Common early symptoms of Alzheimer's include repetition, getting lost, difficulties keeping track of bills, problems with cooking especially new or complicated meals, forgetting to take medication, and word-finding problems. The part of the brain most affected by Alzheimer's is the hippocampus. Other parts of the brain that show shrinking (atrophy) include\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: most common symptoms of Alzheimer's disease are short-term memory loss and word-finding difficulties. People with Alzheimer's disease also have trouble with visual-spatial areas (for example, they may begin to get lost often), reasoning, judgment, and insight. Insight refers to whether or not the person realizes he/she has memory problems. Common early symptoms of Alzheimer's include repetition, getting lost, difficulties keeping track of bills, problems with cooking especially new or complicated meals, forgetting to take medication, and word-finding problems. \n\n"
        "###\nQuestion: {input}\n\n"
        "Answer: {output}\n\n"
        "Passage: {text}\n\n"
        "Rating:"
    ),

    "alpaca": (
        "Given a instruction, answer, and external knowledge passage, your job is to determine whether the passage is relevant to the question and can provide useful information for obtaining the answer. "
        "If the passage meets this requirement, respond with [Relevant] and extract useful spans from the passage."
        "The extracted spans consist of complete sentences to make the extracted span understandable standalone. \n\n"
        "###\nInstruction: Write a response that appropriately completes the request.\n\nInstruction:\nName two physical conditions that could lead to arthritis.\n\n"
        "Answer: There are several physical conditions that could contribute to the development of arthritis, including:\n\n1. Joint injury: Damage to a joint from an injury or trauma can cause changes in the joint that lead to the development of arthritis over time. This is why athletes and people who engage in demanding physical activities are often at an increased risk of developing arthritis in the affected joints.\n\n2. Obesity: Carrying excess body weight puts extra strain on weight-bearing joints such as the knees and hips. Over time, this can cause damage to the joint tissue and cartilage, which can increase the likelihood of developing arthritis.\n\n"
        "Passage: obesity, heredity, joint injury or stress, and aging. Cartilage in the knee may begin to break down after sustained stress from prolonged standing or walking, leaving the bones of the knee rubbing against each other. Persons who work in a place that applies repetitive stress on the knees are at a high risk of developing this condition. Bone deformities increase the risk for osteoarthritis of the knee since the joints are already malformed and may contain defective cartilage. Having gout, rheumatoid arthritis, Paget's disease of bone or septic arthritis may increase a person's risk of developing osteoarthritis. It is believed\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: obesity, heredity, joint injury or stress, and aging. Cartilage in the knee may begin to break down after sustained stress from prolonged standing or walking, leaving the bones of the knee rubbing against each other. Persons who work in a place that applies repetitive stress on the knees are at a high risk of developing this condition. Bone deformities increase the risk for osteoarthritis of the knee since the joints are already malformed and may contain defective cartilage. Having gout, rheumatoid arthritis, Paget's disease of bone or septic arthritis may increase a person's risk of developing osteoarthritis. \n\n"
        "###\nInstruction: Tell me two advantages of using AI assistants.\n\n"
        "Answer: Two advantages of using AI assistants are (1) they can reduce manual labor, and (2) they can improve customer service by providing instant responses. AI assistants can also be programmed to perform more complex tasks, such as natural language processing and conversational AI. This allows organizations to interact with customers more effectively and allows them to automate more tasks.\n\n"
        "Passage: artificial intelligence or AI. The customer benefits of AI is the feel for chatting with a live agent through improved speech technologies while giving customers the self-service benefit. Another example of automated customer service is by touch-tone phone, which usually involves a main menu, and the use of the keypad as options (i.e. \"Press 1 for English, Press 2 for Spanish\", etc.) However, in the Internet era, a challenge has been to maintain and/or enhance the personal experience while making use of the efficiencies of online commerce. \"Online customers are literally invisible to you (and you to them), so it's\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: artificial intelligence or AI. The customer benefits of AI is the feel for chatting with a live agent through improved speech technologies while giving customers the self-service benefit. \n\n"
        "###\nInstruction: Rewrite the sentence using gender-neutral language.\nOutput should be a single sentence.\n\nA salesman is giving a presentation.\n\n"
        "Answer: A salesperson is giving a presentation.\n\n"
        "Passage: trade-off between inclusiveness and wordiness. As a result of campaigns by advocates of feminist language modification, many job advertisements are now formulated so as to explicitly include a grammatically male and a female word (\"Informatiker oder Informatikerin\"). The option of repeating all terms in two gender forms is considered clumsy, and in the singular requires adjectives, articles, and pronouns to also be stated twice. As an alternative, the use of slashes or parenthesis is commonplace, as in \"Informatiker/-in\", but this is considered visually ungainly and there is no consensus on how it is pronounced. Recently, another approach is to use\n\n"
        "Rating: [Irrelevant]\n\n"
        "Extracted span: None\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Answer: {output}\n\n"
        "Passage: {text}\n\n"
        "Rating:"
    ),

    "fever": (
        "Given a fact-checking question，answer and external passage, your job is to determine if the passage is relevant to the question and can provides fact spans to obtan the answer. "
        "If the passage meets this requirement, respond with [Relevant] and extract a useful span from the passage as supporting fact to answer the question."
        "The extracted spans consist of complete sentences to make the extracted span understandable standalone. "
        "If the passage does not help answer the question, return [Irrelevant].\n\n"
        "###\nQuestion: Paul Newman won an Academy Award.\n\n"
        "Answer: true\n\n"
        "Passage: Actress category. Best Actor winner Paul Newman was the fourth actor to have been nominated for portraying the same character in two different films, having previously earned a nomination for his role as \"Fast Eddie\" Felson in 1961's \"The Hustler\". By virtue of his victory in the Best Actor category, Newman and wife Joanne Woodward, who won Best Actress for her performance in 1957's \"The Three Faces of Eve\", became the second married couple to win acting Oscars. \"\" and \"Down and Out in America\"s joint win in the Best Documentary Feature category marked the fourth occurrence of a tie\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: By virtue of his victory in the Best Actor category, Newman and wife Joanne Woodward, who won Best Actress for her performance in 1957's \"The Three Faces of Eve\", became the second married couple to win acting Oscars. \n\n"
        "###\nQuestion: The Catalyst is by an American bluegrass band.\n\n"
        "Answer: false\n\n"
        "Passage:The Catalyst Fire is the second studio album by Australian progressive rock band Dead Letter Circus. It featured new band members Tom Skerlj and Clint Vincent after founding member and guitarist Rob Maric left the band at the end of 2012. In 2011, Tom Skerlj was added to the band as a second guitarist and keyboardist and the band began working on writing new material, hinting that they would follow a similar process to the preceding album in that completion of the songwriting would occur while the band was recording the album; \"We don't ever write whole\n\n"
        "Rating: [Irrelevant]\n\n"
        "Extracted span: None.\n\n"
        "###\nQuestion: {input}\n\n"
        "Answer: {output}\n\n"
        "Passage: {text}\n\n"
        "Rating:"
    ),

    "nq": (
        "Given a question，answer and external knowledge passage, your job is to determine whether the passage is relevant to the question and can provide useful information for obtaining the answer. "
        "If the passage meets this requirement, respond with [Relevant] and extract a useful spans from the passage as supporting fact to answer the question."
        "The extracted spans consist of complete sentences to make the extracted span understandable standalone. "
        "If the passage does not help answer the question, return [Irrelevant].\n\n"
        "###\nQuestion: when did k9 first appear in doctor who ?\n\n"
        "Answer: 1977\n\n"
        "Passage: K9 is a robot dog acquired by \"Doctor Who\"s title character in the 1977 serial \"The Invisible Enemy\". The first two incarnations of the character travelled alongside the Fourth Doctor (portrayed by Tom Baker) until 1981. In these stories, K9 proved useful for the powerful laser weapon concealed in his nose, his encyclopaedic knowledge and his vast computer intelligence. By 1981, each of the two models of K9 which travelled alongside the Doctor had been left with one of the Doctor's female companions. The character subsequently transitioned into spin-off territory. Producers hoped K9's popularity with children would launch the series \n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: K9 is a robot dog acquired by \"Doctor Who\"'s title character in the 1977 serial \"The Invisible Enemy\".\n\n"
        "###\nQuestion: when did k9 first appear in doctor who ?\n\n"
        "Answer: 1977\n\n"
        "Passage: Jaeger in the year 5000. K9 subsequently travelled with the Fourth Doctor (Tom Baker) and Leela (Louise Jameson) as a companion of the Doctor in his adventures in time and space until \"The Invasion of Time\" (1978). In this serial, K9 decides to remain on the Doctor's home planet of Gallifrey with Leela. Immediately afterwards, \"Doctor Who\" would introduce a second incarnation of K9, played by the same prop; the last scene of \"The Invasion of Time\" shows the Doctor unpacking a box labeled \"K9 Mk II\". Although the first incarnation of K9 does not appear again in televised \"Doctor\n\n"
        "Rating: [Irrelevant]\n\n"
        "Extracted span: None.\n"
        "###\nQuestion: {input}\n\n"
        "Answer: {output}\n\n"
        "Passage: {text}\n\n"
        "Rating:"
    ),

    "wow": (
        "Given a question，answer and external passage, your job is to determine if the passage is relevant to the question and can provides useful information to obtan the answer. "
        "If the passage meets this requirement, respond with [Relevant] and extract a useful span from the passage as supporting fact to answer the question."
        "The extracted spans consist of complete sentences to make the extracted span understandable standalone. "
        "If the passage does not help answer the question, return [Irrelevant].\n\n"
        "###\nQuestion: Notable person with the surname Dawson\n\n"
        "Answer: Abraham Dawson (1816–1884) was an Irish-Canadian Anglican cleric.\n\n"
        "Passage: Abraham Dawson Abraham Dawson was an Irish-Canadian Anglican cleric. He was also a very prominent member of the Orange Order in Canada and member of a Canadian political family. He was born in Killyman, Co. Tyrone, on 29 July 1816. As a Christian preacher, he was based in a variety of locations throughout Ireland, including Knockmanaul, Turin, Athlone, Manorhamilton, Sligo, Strabane and Newtownstewart, before emigrating to Canada West in 1864. He had about twelve children, including George Walker Wesley Dawson, who became an MP. He was the Grand Chaplain of the Grand Orange Lodge of Canada in 1874. He died on 12 May 1884 in Plevna, Ontario. Abraham Dawson Abraham Dawson was an Irish-Canadian Anglican cleric.\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: Abraham Dawson Abraham Dawson was an Irish-Canadian Anglican cleric. He was also a very prominent member of the Orange Order in Canada and member of a Canadian political family. He was born in Killyman, Co. Tyrone, on 29 July 1816. As a Christian preacher, he was based in a variety of locations throughout Ireland, including Knockmanaul, Turin, Athlone, Manorhamilton, Sligo, Strabane and Newtownstewart, before emigrating to Canada West in 1864. He had about twelve children, including George Walker Wesley Dawson, who became an MP. He was the Grand Chaplain of the Grand Orange Lodge of Canada in 1874. He died on 12 May 1884 in Plevna, Ontario. Abraham Dawson Abraham Dawson was an Irish-Canadian Anglican cleric.\n\n"
        "###\nQuestion: Elevation album by Anggun international success\n\n"
        "Answer: In France, Elevation debuted at number 36 on the French Albums Chart.\n\n"
        "Passage: Roman Catholic priest Aeneas McDonell Dawson. He died in Ottawa in 1902, virtually forgotten. Simon James Dawson Simon James Dawson (June 13, 1818 – October 30, 1902) was a Canadian civil engineer and politician. Born in Redhaven, Banffshire, Scotland, Dawson emigrated to Canada as a young man and began his career as an engineer. In 1857, as a member of a Canadian government expedition, he surveyed a line of road from Prince Arthur’s Landing (later Port Arthur, now part of Thunder Bay, Ontario) to Fort Garry and further explored that area in 1858 and 1859. His report greatly stimulated Canadian\n\n"
        "Rating: [Relevant]\n\n"
        "Extracted span: None.\n\n"
        "###\nQuestion: {input}\n\n"
        "Answer: {output}\n\n"
        "Passage: {text}\n\n"
        "Rating:"
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
                temperature=0.2,
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


def postprocess(raw_output):
    if "[Irrelevant]" in raw_output:
        score = "[Irrelevant]"
        fact = "None"
        return score, fact
    if "\nExtracted span:" in raw_output:
        score = raw_output.split("\n\nExtracted span:")[0]
        fact = raw_output.split("\nExtracted span:")[1]
        return score, fact
    else:
        return "", ""


def process_input(example, multi_retrieval=False):
    if multi_retrieval is False:
        return PROMPT_DICT["alpaca"].format_map(example)
    else:
        if "sent_idx" not in example or example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_DICT["multi_no_preceding"].format_map(example)
        else:
            return PROMPT_DICT["alpaca"].format_map(example)


def process_item(args, example, idx, result_list):
    reivial = []
    for context in example['retrieval']:
        example['text'] = context['text']
        input = PROMPT_DICT["multi_choice"].format_map(example)
        results = chat_gpt(args, input)
        score, fact = postprocess(results)
        context['text'] = context['title'] + "\n" + example['text']
        if fact == "None":
            context['fact'] = fact
        else:
            context['fact'] = fact
        context['relevant'] = "true" if score == "[Relevant]" else "false"
        reivial.append(context)

        if idx % 20 == 0:
            print("Input: {}".format(example["instruction"]))
            print("Output: {}".format(example["output"]))
            print("Evidence: {}".format(example["text"]))
            print("Score: {}".format(score))
            print("Fact: {}".format(fact))
            print("======================================================================\n")

    data = {"id": example['id'],
            "dataset_name": example["dataset_name"],
            "instruction": example["instruction"],
            "input": example["input"],
            'intent': example['intent'],
            'output': example['output'],
            "retrieval": reivial
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
