"""
Run the CPO training script with the following command with some example arguments.
In general, the optimal configuration for CPO will be similar to that of DPO:

python cpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-cpo" \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import torch
from trl import CPOConfig, CPOTrainer, ModelConfig, ScriptArguments, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, CPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    example_prompts = [
        "Describe a beautiful sunrise in detail.",
        "Write something positive about teamwork.",
        "What makes a garden a pleasant place to relax?",
        "Describe the feeling of achieving a goal.",
        "Explain why regular exercise is beneficial in a cheerful tone.",
    ]
    print("=== Model output before RLHF ===")
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nOutput: {output_text}\n")
    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # train and save the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    print("=== Model output after RLHF ===")
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nOutput: {output_text}\n")
