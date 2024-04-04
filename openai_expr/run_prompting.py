import argparse
import os
import time
import json
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

import tiktoken

# add your OpenAI API key here
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Count the number of tokens used by a list of messages, 
    and truncate the messages if the number of tokens exceeds the limit.
    Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    if model == "gpt-3.5-turbo-0125":
        max_tokens = 16385 - max_gen_length
    elif model == "gpt-4-0125-preview":
        max_tokens = 128000 - max_gen_length
    else:
        max_tokens = 4096 - max_gen_length
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3

    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    trunc_messages = []
    for message in messages:
        tm = {}
        num_tokens += tokens_per_message
        for key, value in message.items():
            encoded_value = encoding.encode(value)
            num_tokens += len(encoded_value)
            if num_tokens > max_tokens:
                print(f"Truncating message: {value[:100]}...")
                tm[key] = encoding.decode(encoded_value[:max_tokens - num_tokens])
                break
            else:
                tm[key] = value
        trunc_messages.append(tm)
    return trunc_messages


# get completion from an OpenAI chat model
def get_openai_chat(
    prompt,
    args
):
    if args.fewshot_eg:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["prompt"]}
        ]
    else:
        # select the correct in-context learning prompt based on the task
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
            ]
    
    # count the number of tokens in the prompt
    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)
    # get response from OpenAI
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
            seed=args.seed,
            logprobs=args.logprobs,
            top_logprobs=5 if args.logprobs else NOT_GIVEN,
            )
        response_content = response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs.content[0].top_logprobs if args.logprobs else None
        # map the token to the prob
        log_prob_mapping = {}
        if response_logprobs:
            for topl in response_logprobs:
                log_prob_mapping[topl.token] = topl.logprob
        # the below could be used to verify the system fingerprint and ensure the system is the same
        # system_fingrprint = response.system_fingerprint
        # print(system_fingrprint)
        
        # if the API is unstable, consider sleeping for a short period of time after each request
        # time.sleep(0.2)
        return response_content, log_prob_mapping

    # when encounter RateLimit or Connection Error, sleep for 5 or specified seconds and try again
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(
            prompt,
            args,
        )
    # when encounter bad request errors, print the error message and return None
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        prompts.append(p)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125", choices=["gpt-3.5-turbo-0125", "gpt-4-0125-preview"], help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()

    output_file = os.path.join(args.output_folder, f"{args.model}_{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}.jsonl")
    if args.prompt_strategy == "std_cls":
            inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} {args.data} prompts ...")
        for p in tqdm(prompts):
            response, logprobs = get_openai_chat(p, args)
            if logprobs:
                p["logprobs"] = logprobs
                print(logprobs)
            if response is None:
                response = "ERROR"
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    main()