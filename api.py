##  python api.py --trt_engine_path model/ --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path model/ --data_dir dataset/ --port 8081

import time

#import gradio as gr
import argparse
import pdfkit
from flask import jsonify
import uuid
from trt_llama_api import TrtLlmAPI #llama_index does not currently support TRT-LLM. The trt_llama_api.py file defines a llama_index compatible interface for TRT-LLM.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage
from flask import Flask, Response, request, jsonify
from utils import messages_to_prompt, completion_to_prompt, ChatMessage, MessageRole, DEFAULT_SYSTEM_PROMPT

# Create an argument parser
parser = argparse.ArgumentParser(description='NVIDIA Chatbot Parameters')

# Add arguments
parser.add_argument('--trt_engine_path', type=str, required=True,
                    help="Path to the TensorRT engine.", default="")
parser.add_argument('--trt_engine_name', type=str, required=True,
                    help="Name of the TensorRT engine.", default="")
parser.add_argument('--tokenizer_dir_path', type=str, required=True,
                    help="Directory path for the tokenizer.", default="")
parser.add_argument('--embedded_model', type=str,
                    help="Name or path of the embedded model. Defaults to 'sentence-transformers/all-MiniLM-L6-v2' if "
                         "not provided.",
                    default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--data_dir', type=str, required=False,
                    help="Directory path for data.", default="./dataset")
parser.add_argument('--verbose', type=bool, required=False,
                    help="Enable verbose logging.", default=False)

parser.add_argument("--host", type=str, help="Set the ip address to listen.(default: 127.0.0.1)", default='127.0.0.1')
parser.add_argument("--port", type=int, help="Set the port to listen.(default: 8081)", default=8081)
parser.add_argument("--max_output_tokens", type=int, help="Maximum output tokens.(default: 1024)", default=1024)
parser.add_argument("--max_input_tokens", type=int, help="Maximum input tokens.(default: 3900)", default=3900)

app = Flask(__name__)

# Parse the arguments
args = parser.parse_args()

# Use the provided arguments
trt_engine_path = args.trt_engine_path
trt_engine_name = args.trt_engine_name
tokenizer_dir_path = args.tokenizer_dir_path
embedded_model = args.embedded_model
data_dir = args.data_dir
verbose = args.verbose
host = args.host
port = args.port
no_system_prompt = False

# create trt_llm engine object
llm = TrtLlmAPI(
    model_path=trt_engine_path,
    engine_name=trt_engine_name,
    tokenizer_dir=tokenizer_dir_path,
    temperature=0.1,
    max_new_tokens=args.max_output_tokens,
    context_window=args.max_input_tokens,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False
)

# create embeddings model object
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embedded_model))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# load the vectorstore index
faiss_storage = FaissEmbeddingStorage(data_dir=data_dir)
query_engine = faiss_storage.get_query_engine()

def make_resData(data, chat=False, promptToken=[]):
    resData = {
        "id": f"chatcmpl-{str(uuid.uuid4())}" if (chat) else f"cmpl-{str(uuid.uuid4())}",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA",
        "usage": {
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "total_tokens": data["prompt_tokens"] + data["completion_tokens"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken
    if (chat):
        # only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    else:
        # only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    return resData

# chat function to trigger inference
def chatbot(query, history):
    if verbose:
        start_time = time.time()
        response = faiss_storage.get_query_engine().query(query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference e2e time    : {elapsed_time:.2f} seconds \n")

    else:
        response = faiss_storage.get_query_engine().query(query)  
    # return str(response)
    thisdict = dict(truncated=False,
                prompt_tokens=2048,
                completion_tokens=2048,
                content=str(response),
                stopped=False,
                slot_id=1,
                stop=True)

    resData = make_resData(thisdict, chat=True)
    return jsonify(resData)

def is_present(json, key):
    try:
        buf = json[key]
    except KeyError:
        return False
    if json[key] == None:
        return False
    return True

@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    assert request.headers.get('Content-Type') == 'application/json'
    body = request.get_json()
    stream = False
    temperature = 1.0
    if (is_present(body, "stream")):
        stream = body["stream"]
    if (is_present(body, "temperature")):
        temperature = body["temperature"]
    formatted = False
    if verbose:
        print("/chat/completions called with stream=" + str(stream))

    prompt = ""
    if "messages" in body:
        messages = []
        for item in body["messages"]:
            chat = ChatMessage()
            if "role" in item:
                if item["role"] == 'system':
                    chat.role = MessageRole.SYSTEM
                elif item["role"] == 'user':
                    chat.role = MessageRole.USER
                elif item["role"] == 'assistant':
                    chat.role = MessageRole.ASSISTANT
                elif item["role"] == 'function':
                    chat.role = MessageRole.FUNCTION
                else:
                    print("Missing handling role in message:" + item["role"])
            else:
                print("Missing role in message")

            chat.content = item["content"]
            messages.append(chat)

        system_prompt = ""
        if not no_system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        prompt = messages_to_prompt(messages, system_prompt)

        formatted = True
    elif "prompt" in body:
        prompt = body["prompt"]

    if verbose:
        print("INPUT SIZE: " + str(len(prompt)))
        print("INPUT: " + prompt)

    return chatbot(prompt, '')

@app.route('/v1/gen_dataset', methods=['POST'])
def generate_new_dataset():
    assert request.headers.get('Content-Type') == 'application/json'
    body = request.get_json()
    url = "https://google.com"
    if (is_present(body, "url")):
        url = body["url"]
    print(url)
    pdfkit.from_url(url,'dataset/content.pdf')

    # load the vectorstore index
    faiss_storage.regenerate_index()

    return 'ok'

if __name__ == '__main__':
    app.run(host, port=port, debug=True, use_reloader=False, threaded=False)

