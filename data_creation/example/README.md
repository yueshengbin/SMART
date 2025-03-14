## How to Construct data

We collect long trajectory data. Use the following script to collect data.

- Step 1 [Reconstructing intent](chatgpt_intent_multithreading.py)

```sh
python chatgpt_intent_multithreading.py \
    --input_file path_to_input_file \
    --output_file_name path_to_output_file \
    --model_name open_ai_model_name \
```

- Step 2 [Retrieval knowledge](passage_retrieval.py)

```sh
python passage_retrieval.py \
--model_name_or_path facebook/contriever-msmarco \
--passages psgs_w100.tsv \
--passages_embeddings "wikipedia_embeddings/*" \
--dataset input_data_path \
--output_dir output_data_path \
--n_docs 3
```

- Step 3 [Fact Locating](chatgpt_fact_multithreading.py)

```sh
python chatgpt_fact_multithreading.py \
    --input_file path_to_input_file \
    --output_file_name path_to_output_file \
    --model_name open_ai_model_name \
```

- Step 4 [Combine data and Insert token](long_data_combine.py)

The file is a `json` or `jsonl` file containing a list of entries. Each entry consists of 

```py
{
    "instruction": str, # input instruction 
    "input": str, # input
    "intent": str, # knowledge query intent
    "output": str, # model response
    "retrieval": [
                    {
                        "title": str,  # retrieval document title
                        "text": str,   # retrieval relevant content
                        "fact": str,  # fact in relevant relevant
                        "relevant": str  # true or false
                    }
    ] 
}
```

Combine the data and insert the trajectory header and end token.

```sh
python long_data_combine.py \
    --input_folder_name path_to_folder_name \
    --output_file_name path_to_output_file \
```