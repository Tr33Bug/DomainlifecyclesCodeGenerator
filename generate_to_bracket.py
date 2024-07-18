import transformers
import torch
import torch.nn.functional as F


### PROMPT ###
prompt = "<START>{ \"@class\" : \"nitrox.dlc.mirror.model.FieldModel\","
###




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



# load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# add pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})



input_ids = tokenizer.encode(prompt, return_tensors='pt')
input_ids = input_ids.to('cuda')

end_token_id = tokenizer.encode("}", add_special_tokens=False)[0]



# generate token to end token
# output = model.generate(input_ids, eos_token_id=end_token_id, temperature=0.1, max_length=300)

# Initialize the sequence with the prompt
sequence = input_ids

counter = 0
while True:
    # Generate the next token
    next_token_logits = model(sequence).logits[:, -1, :]
    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

    # Add the next token to the sequence
    output = torch.cat([sequence, next_token], dim=-1)

    # Increment the counter
    counter += 1

    # Stop if the "}" token is generated or 300 tokens are reached
    if next_token.item() == end_token_id or counter >= 300:
        break
    print(next_token)
print(output)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


# delete start token "<START>" from text
generated_text = generated_text.replace("<START>", "")


with open('generated_json_bracket.json', 'w') as f:
        generated_text_clean = generated_text.replace("<START>", "")
        f.write(generated_text_clean)




