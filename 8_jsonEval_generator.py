# %% [markdown]
# # JSON Evaluation Pipeline

# %%
# !pip install transformers
# !pip install torch
# !pip install pandas
# !pip3 install torch torchvision torchaudio
# !pip install ipywidgets
# !pip install bitsandbytes
# !pip install accelerate

# %%
import transformers
import torch
import tqdm
import json

# %% [markdown]
# ## Model Import

# %%
## define run name
run_name = "finalTraining_v1"
# run_name = "MLPC-2048-StarCoderBase7B"

# define model for tokenizer
model_name = "codellama/CodeLlama-7b-hf"
# model_name = "bigcode/starcoderbase-7b"

# dataset import folder
export_folder = "./dataset/" + run_name + "/"

# model save path
model_save_path = "./models/" + run_name + "/"

# model checkpoint path
model_checkpoint_path = "./checkpoints/" + run_name + "/"

# %%
## Test loading model and inference with that model

# load quantization config for 4bit quantization -> must be same as training
quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# load model from model_save_path with quantization config
model = transformers.AutoModelForCausalLM.from_pretrained(model_save_path, quantization_config=quantization_config, low_cpu_mem_usage=True)

# optional: load model untrained
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage=True)

# optional: load model unquantized and untrained
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

# optional: load model from checkpoint
# model = transformers.AutoModelForCausalLM.from_pretrained("./output/bigRun/checkpoint-1000", quantization_config=quantization_config, low_cpu_mem_usage=True)

# %%
# load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# add pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# %% [markdown]
# ## Generator

# %%
def generate_nitrox_json(model, tokenizer, prompt, use_custom_eos=False, custom_eos_token="}", max_length=3000, confidence_threshold=0.01):
    """ Generate code completions for a given prompt using the model and tokenizer.
    """


    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Initialize the output as the input
    output = input_ids
    
    # Loop until the end token is generated or counter is at max_length
    for i in tqdm.tqdm(range(max_length)):
        # Predict the probabilities of the next token
        with torch.no_grad():
            outputs = model(output)
        predictions = outputs.logits[:, -1, :]
        probabilities = torch.nn.functional.softmax(predictions, dim=-1)

        # Get the token with the highest probability
        max_prob, max_token_id = torch.max(probabilities, dim=-1)

        # Check if the confidence is over the threshold
        if max_prob.item() < confidence_threshold:
            break

        # Append the token to the output
        output = torch.cat([output, max_token_id.unsqueeze(0)], dim=-1)

        if len(output[0]) > 3 + len(custom_eos_token):
            evtl_end = tokenizer.decode(output[0][-3:], skip_special_tokens=True)
            if use_custom_eos:
                if custom_eos_token in evtl_end:
                    break
            # check for <EOS> in evtl_end
            if "<EOS>" in evtl_end:
                break
        
        # decode every 1000 iterations and print output
        if len(output[0]) % 50 == 0:
            # print(tokenizer.decode(output[0], skip_special_tokens=True))
            print("Length of output: ", len(output[0]))
            print("Max prob: ", max_prob.item())
            print("Max token: ", max_token_id.item())
            print("Counter: ", i)
            print("")

        
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # delete <START> from generated text
    generated_text = generated_text.replace("<START>", "")

    print(generated_text)

    # cleanup
    del output
    del input_ids
    torch.cuda.empty_cache()

    return generated_text

# %%
def generate_nitrox_json_fast_hope(model, tokenizer, prompt, custom_eos_token="<END>", max_length=4000):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    end_token_id = tokenizer.encode(custom_eos_token, add_special_tokens=False)[0]

    output = model.generate(input_ids, eos_token_id=end_token_id, temperature=0.1, max_length=max_length)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # delete <START> from generated text
    generated_text = generated_text.replace("<START>", "")
    # print(generated_text)
    
    # cleanup
    del output
    del input_ids
    torch.cuda.empty_cache()

    return generated_text

# %%
# test generate_nitrox_json_fast_hope
# prompt_test = '<START> { "@class" : "nitrox.dlc.mirror.model.EnumModel"'
# test_output = generate_nitrox_json_fast_hope(model, tokenizer, prompt_test)

# %%
def generate_multi_prompts(prompts, model, tokenizer, use_custom_eos=True, custom_eos_token='"valueObject" : true}', 
max_length=6000, confidence_threshold=0.01, export_path="gen_json"):
    counter = 0
    for prompt in tqdm.tqdm(prompts):
        # generated_text = generate_nitrox_json(model, tokenizer, prompt, use_custom_eos=use_custom_eos, custom_eos_token=custom_eos_token, max_length=max_length, confidence_threshold=confidence_threshold)
        generated_text = generate_nitrox_json_fast_hope(model, tokenizer, prompt, custom_eos_token=custom_eos_token, max_length=max_length)

        # save generated text as file named after the model Type
        # with open(export_path + "/" + str(counter) + "_" + prompt.split("nitrox.dlc.mirror.model.")[1].split(",")[0].replace("\"","") + ".json", "w") as f:
        with open(export_path + "/" + str(counter) + ".json", "w") as f:
            f.write(generated_text)
        counter += 1

# %% [markdown]
# ## Generate from Prompts

# %% [markdown]
# Generate JSON to max token length.

# %%
prompts = [
    '<START>',
    '<START> { ',
    '<START> { "',
    '<START> { "@class" ',
    '<START> { "@class" : ',
    '<START> { "@class" : "nitrox.',
    '<START> { "@class" : "nitrox.dlc.',
    '<START> { "@class" : "nitrox.dlc.mirror.',
    '<START> { "@class" : "nitrox.dlc.mirror.model.',
    '',
]
# prompts = [
#     '<START> { "@class" : "nitrox.dlc.mirror.model.EntityModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.ValueObjectModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.AggregateRootModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.IdentityModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.EnumModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.DomainServiceModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.RepositoryModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.ApplicationServiceModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.DomainEventModel"',
#     '<START> { "@class" : "nitrox.dlc.mirror.model.DomainCommandModel"',
# ]

# %%
# multiply prompts for more prompts -> prompt = 10 * prompt
prompts = 5 * prompts

len(prompts)

# %%
# generate generate_multi_prompts
generate_multi_prompts(prompts, model, tokenizer, use_custom_eos=False, custom_eos_token="<END>", max_length=4000, confidence_threshold=0.01, export_path="gen_json")

# %% [markdown]
# ## Complete JSONs

# %%
def complete_json(json_file):
    """
    clean and close the json 
    """

    # 1. check the end of the JSON and delete the last uncomplete key/value pair. (Delete till the first komma ",")
    json_file = json_file[:json_file.rfind(",")]

    # 2. create a array with the unclosed brackets in the json -> move sequential through the json and add when brackets are open and delete when they are closed. ((, {, [) -> (,), }, ])
    open_brackets = []
    closed_brackets = []
    for char in json_file:
        if char in ["(", "{", "["]:
            open_brackets.append(char)
        if char in [")", "}", "]"]:
            closed_brackets.append(char)
    
    # find the missing brackets by comparing the two arrays
    brackets = []
    for bracket in open_brackets:
        if bracket == "(":
            if ")" not in closed_brackets:
                brackets.append(bracket)
            else:
                closed_brackets.remove(")")
        elif bracket == "{":
            if "}" not in closed_brackets:
                brackets.append(bracket)
            else:
                closed_brackets.remove("}")
        elif bracket == "[":
            if "]" not in closed_brackets:
                brackets.append(bracket)
            else:
                closed_brackets.remove("]")
    
    print(brackets)
    # 3. add the missing brackets from the array to the end of the json
    for bracket in brackets:
        if bracket == "(":
            json_file += ")"
        elif bracket == "{":
            json_file += "}"
        elif bracket == "[":
            json_file += "]"
    
    return json_file

# %%
# read all json_files in gen_json folder
import os

json_files = []
for file in os.listdir("gen_json"):
    if file.endswith(".json"):
        with open("gen_json/" + file, "r") as f:
            json_files.append(f.read())

# %%
# complete all jsons
completed_jsons = []
for json_file in json_files:
    completed_jsons.append(complete_json(json_file))

# %%
# save all jsons to gen_json/completed folder
for i in range(len(completed_jsons)):
    with open("gen_json/completed/" + str(i) + ".json", "w") as f:
        f.write(complete_json(completed_jsons[i]))

# %% [markdown]
# ## Parse completed JSONs

# %%
import json
import os

# %%
# test test_json_comp with json.loads and look for errors
def test_json_function(test_json):
    try:
        json.loads(test_json)
        return "No error"
    except Exception as e:
        return e
    

# %%
test_json = ' { "@class" : "nitrox.dlc.mirror.model.ApplicationServiceModel" , "typeName" : "VAR_typeName" }'

result = test_json_function(test_json)
result

# %%
# read all jsons in gen_json/completed folder

jsons_to_parse = []
for file in os.listdir("gen_json/completed"):
    if file.endswith(".json"):
        with open("gen_json/completed/" + file, "r") as f:
            jsons_to_parse.append(f.read())

# %%
# test all jsons
results = {}
for number, json_file in enumerate(jsons_to_parse):
    # append results to results dict
    results[number] = test_json_function(json_file)

# save results to results.txt
with open("gen_json/completed/results.txt", "w") as f:
    for key in results:
        f.write(str(key) + ": " + str(results[key]) + "\n")

# %%



