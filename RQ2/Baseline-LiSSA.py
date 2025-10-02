import json
import time
import re

import requests
import argparse
import sys

import logging
from anyio import current_time
from pyexpat.errors import messages

import pandas as pd
from sklearn.model_selection import KFold

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ======================
# Model settings
# ======================
# Model settings
MODEL_SETTINGS = {
    "Claude 3.5 Sonnet": {"max_tokens": 20000, "temperature": 0, "top_p": 0.95, "top_k": 50}
}



# ======================
# Available models
# ======================
MODELS = {
    "1": {"name": "Claude 3.5 Sonnet", "id": "anthropic.claude-3-5-sonnet-20240620-v1:0"}
}


# ======================
# Bedrock Proxy
# ======================
class BedrockProxy:
    def __init__(self, api_url):
        self.api_url = api_url

    def __getattr__(self, name):
        def method(**kwargs):
            response = requests.post(self.api_url, json={
                'bedrock_method': name,
                'bedrock_params': kwargs
            }, stream=True)
            return response
        return method


class BedrockClient:
    def __init__(self, proxy):
        self.proxy = proxy

    def invoke_model_with_response_stream(self, *args, **kwargs):
        return self.proxy.invoke_model_with_response_stream(*args, **kwargs)


# ======================
# Prompt formatting
# ======================
def format_claude_prompt(history, new_prompt, params):
    messages = []
    if new_prompt:
        messages.append({"role": "user", "content": new_prompt})
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        **params
    }



# ======================
# Response processing
# ======================
def process_claude_response(response):
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "type" in chunk:
                    if chunk["type"] == "content_block_delta" and "text" in chunk.get("delta", {}):
                        yield chunk["delta"]["text"]
                    elif chunk["type"] == "message_delta" and "stop_reason" in chunk.get("delta", {}):
                        yield {"stop_reason": chunk["delta"]["stop_reason"]}
                    if "amazon-bedrock-invocationMetrics" in chunk:
                        metrics = chunk["amazon-bedrock-invocationMetrics"]
                        yield {"usage": {"input_tokens": metrics["inputTokenCount"], "output_tokens": metrics["outputTokenCount"]}}
            except Exception as e:
                yield {"error": str(e)}


# ======================
# Generate response
# ======================
def generate_response(bedrock_client, prompt, model_id, history):
    model_name = next(model["name"] for model in MODELS.values() if model["id"] == model_id)
    if "claude" in model_id.lower():
        messages = format_claude_prompt(history, prompt, MODEL_SETTINGS[model_name])
        process_response = process_claude_response
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")

    body = json.dumps(messages)
    try:
        response = bedrock_client.invoke_model_with_response_stream(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        if response.status_code != 200:
            error_content = response.json()
            return f"Error: {error_content.get('error', 'Unknown error occurred')}"

        full_response = ""
        stop_reason = None
        usage = None
        for chunk in process_response(response):
            if isinstance(chunk, str):
                full_response = chunk
            elif isinstance(chunk, dict):
                if "stop_reason" in chunk:
                    stop_reason = chunk["stop_reason"]
                if "usage" in chunk:
                    usage = chunk["usage"]
                if "error" in chunk:
                    print(f"\nError: {chunk['error']}")

        return full_response, stop_reason, usage
    except Exception as e:
        return f"Error: {str(e)}", None, None


# ======================
# Prompt creator
# ======================
def make_prompt(stake, sys):

    # KISS prompt
    prompt = (
        f"Question: Here are two parts of software development artifacts."
        f"Stakeholder Requirement: {stake}"
        f"System Requirement: {sys}"
        f"Are they related?"
        f"Answer with 'yes' or 'no'."
    )

    # # CoT prompt
    # prompt = (f"Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>."
    #           f"Stakeholder Requirement: {stake}"
    #           f"System Requirement: {sys}"
    # )
    return prompt


# ======================
# Load dataset
# ======================
def load_data():
    df = pd.read_csv("pairs.csv")
    print(f"We obtained {len(df)} stakeholder, system requirement pairs.")
    return df


# ======================
# Main logic
# ======================
def main(prompt, number):
    parser = argparse.ArgumentParser(description="AI Model Interaction CLI")
    parser.add_argument("--prompt", nargs="?", default=prompt, help="The prompt to send to the AI model")
    parser.add_argument("--model", choices=MODELS.keys(), default=number, help="Select the model to use")
    args = parser.parse_args()

    config = {
        "PROXY_URL": "xxxx"   # EC2 bedrock proxy URL requires VPN
    }
    bedrock = BedrockProxy(f"{config['PROXY_URL']}/bedrock")
    bedrock_client = BedrockClient(bedrock)

    selected_model = MODELS[args.model]

    if args.prompt:
        try:
            response, stop_reason, usage = generate_response(bedrock_client, args.prompt, selected_model["id"], history=[])
            print("Answer:", response)
            if stop_reason:
                print(f"\nStop reason: {stop_reason}")
            if usage:
                print(f"Input tokens: {usage['input_tokens']}, Output tokens: {usage['output_tokens']}")
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    else:
        history = []
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == "quit":
                break
            try:
                response, stop_reason, usage = generate_response(bedrock_client, prompt, selected_model["id"], history)
                print("\nResponse:", response)
                if stop_reason:
                    print(f"\nStop reason: {stop_reason}")
                if usage:
                    print(f"Input tokens: {usage['input_tokens']}, Output tokens: {usage['output_tokens']}")
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"Error: {str(e)}")
                return None


if __name__ == "__main__":
    current_time_str = time.strftime("%H:%M:%S", time.localtime())
    print("Current time:", current_time_str)

    df = load_data()
    results = []

    modelID = "1"  # we use the same model with TVR
    for _, row in df.iterrows():
        prompt = make_prompt(row["stakeholder"], row["system"])
        response = main(prompt, modelID)

        systemID = row["systemID"]
        stakeholderID = row["stakeholderID"]
        label = row["label"]

        result = {
            "systemID": systemID,
            "stakeholderID": stakeholderID,
            "label": label,
            "response": response
        }
        results.append(result)

    with open(f"{modelID}-LiSSA.json", "w") as outfile:
        json.dump(results, outfile)
