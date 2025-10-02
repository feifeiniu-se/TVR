import json
import time
import re

import requests
import argparse
import pandas as pd



# ======================
# Model settings
# ======================
# Model settings
MODEL_SETTINGS = {
    "Claude 3.5 Sonnet": {"max_tokens": 20000, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Claude 3 Sonnet": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Claude 2": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Llama 3 8B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Llama 3 70B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Claude 3 Opus": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Claude Instant": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Claude 3 Haiku": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Llama 3 1.0 70B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Llama 3 1.0 8B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Llama 3 2.0 1B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Llama 3 2.0 3B": {"max_gen_len": 2048, "temperature": 0, "top_p": 0.95},
    "Mistral 7B": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Mixtral 8x7B": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Mistral Large 2402": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Mistral Large 2407": {"max_tokens": 4096, "temperature": 0, "top_p": 0.95, "top_k": 50},
    "Amazon Titan Text Premier": {"maxTokenCount": 3072, "temperature": 0, "topP": 0.95},
    "Amazon Titan Text Express": {"maxTokenCount": 3072, "temperature": 0, "topP": 0.95},
    "Amazon Titan Text Lite": {"maxTokenCount": 3072, "temperature": 0, "topP": 0.95},
}



# ======================
# Available models
# ======================
MODELS = {
    "1": {"name": "Claude 3.5 Sonnet", "id": "anthropic.claude-3-5-sonnet-20240620-v1:0"},
    "2": {"name": "Claude 3 Sonnet", "id": "anthropic.claude-3-sonnet-20240229-v1:0"},
    "3": {"name": "Claude 2", "id": "anthropic.claude-v2"},
    "4": {"name": "Llama 3 8B", "id": "meta.llama3-8b-instruct-v1:0"},
    "5": {"name": "Llama 3 70B", "id": "meta.llama3-70b-instruct-v1:0"},
    "6": {"name": "Claude 3 Opus", "id": "anthropic.claude-3-opus-20240229-v1:0"},
    "7": {"name": "Claude Instant", "id": "anthropic.claude-instant-v1:0"},
    "8": {"name": "Claude 3 Haiku", "id": "anthropic.claude-3-haiku-20240307-v1:0"},
    "9": {"name": "Llama 3 1.0 70B", "id": "meta.llama3-70b-instruct-v1:0"},
    "10": {"name": "Llama 3 1.0 8B", "id": "meta.llama3-8b-instruct-v1:0"},
    "11": {"name": "Llama 3 2.0 1B", "id": "meta.llama3-2-8b-instruct-v0:2"},
    "12": {"name": "Llama 3 2.0 3B", "id": "meta.llama3-2-70b-instruct-v0:2"},
    "13": {"name": "Mistral 7B", "id": "mistral.mistral-7b-instruct-v0:2"},
    "14": {"name": "Mixtral 8x7B", "id": "mistral.mixtral-8x7b-instruct-v0:1"},
    "15": {"name": "Mistral Large 2402", "id": "mistral.mistral-large-2402-v1:0"},
    "16": {"name": "Mistral Large 2407", "id": "mistral.mistral-large-2407-v1:0"},
    "17": {"name": "Amazon Titan Text Premier", "id": "amazon.titan-text-premier-v1:0"},
    "18": {"name": "Amazon Titan Text Express", "id": "amazon.titan-text-express-v1:0"},
    "19": {"name": "Amazon Titan Text Lite", "id": "amazon.titan-text-lite-v1"},
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


def format_llama_prompt(history, new_prompt, params):
    prompt = "You are a helpful AI assistant. Provide concise and relevant answers.\n\n"
    prompt += f"Human: {new_prompt}\nAssistant:"
    return {
        "prompt": prompt,
        **params,
        "stop": ["\nHuman:", "\n\nHuman:"]
    }


def format_mistral_prompt(history, new_prompt, params):
    prompt = f"Human: {new_prompt}\nAssistant:"
    return {
        "prompt": prompt,
        "max_tokens": params.get("max_tokens", 4096),
        "stop": params.get("stop", ["\nHuman:", "\n\nHuman:"]),
        "temperature": params.get("temperature", 0),
        "top_p": params.get("top_p", 0.95),
        "top_k": params.get("top_k", 50)
    }


def format_titan_prompt(history, new_prompt, params):
    prompt = new_prompt
    top_p = float(params.get("top_p", 0.95))
    top_p = max(min(top_p, 1.0), 1e-8)
    return {
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": float(params.get("temperature", 0)),
            "topP": top_p,
            "maxTokenCount": int(params.get("maxTokenCount", 4096)),
            "stopSequences": params.get("stop_sequences", [])
        }
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


def process_llama_response(response):
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "generation" in chunk:
                    full_response += chunk["generation"]
                if "stop_reason" in chunk:
                    if chunk["stop_reason"] is not None:
                        yield full_response
                    yield {"stop_reason": chunk["stop_reason"]}
                if "amazon-bedrock-invocationMetrics" in chunk:
                    metrics = chunk["amazon-bedrock-invocationMetrics"]
                    yield {"usage": {"input_tokens": metrics["inputTokenCount"], "output_tokens": metrics["outputTokenCount"]}}
            except Exception as e:
                yield {"error": str(e)}
    if full_response:
        yield full_response


def process_mistral_response(response):
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "outputs" in chunk:
                    for output in chunk["outputs"]:
                        if "text" in output:
                            yield output["text"]
                        if "stop_reason" in output:
                            yield {"stop_reason": output["stop_reason"]}
                if "amazon-bedrock-invocationMetrics" in chunk:
                    metrics = chunk["amazon-bedrock-invocationMetrics"]
                    yield {"usage": {"input_tokens": metrics["inputTokenCount"], "output_tokens": metrics["outputTokenCount"]}}
            except Exception as e:
                yield {"error": str(e)}


def process_titan_response(response):
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "outputText" in chunk:
                    yield chunk["outputText"]
                if "completionReason" in chunk:
                    yield {"stop_reason": chunk["completionReason"]}
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
    elif "llama" in model_id.lower():
        messages = format_llama_prompt(history, prompt, MODEL_SETTINGS[model_name])
        process_response = process_llama_response
    elif "mistral" in model_id.lower():
        messages = format_mistral_prompt(history, prompt, MODEL_SETTINGS[model_name])
        process_response = process_mistral_response
    elif "amazon" in model_id.lower():
        messages = format_titan_prompt(history, prompt, MODEL_SETTINGS[model_name])
        process_response = process_titan_response
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
    # 这里可以切换不同的 Prompt 模式
    prompt = (
        f"Please check if the message or signal from the stakeholder requirement is correctly covered by the system requirement. Please only focus on verifying the message or signal mentioned, without considering other parts of the requirement. \n"
        f"Stakeholder Requirement: <stakeholder>{stake}</stakeholder> and System Requirement: <system>{sys}</system>.\n"
        f"Let's think step by step and respond with only 'Yes' or 'No'."
    )
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

    modelIDs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]   # 可以切换模型
    df = load_data()
    results = []

    for modelID in modelIDs:
        print(f"\nmodelID: {modelID}")
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

        with open(f"{modelID}-CoT.json", "w") as outfile:
            json.dump(results, outfile)
