import shutil
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
python ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    example_prompts = [
        "Describe a beautiful sunrise in detail.",
        "Write something positive about teamwork.",
        "What makes a garden a pleasant place to relax?",
        "Describe the feeling of achieving a goal.",
        "Explain why regular exercise is beneficial in a cheerful tone.",
    ]
    print("=== RLHF前模型输出 ===")
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)
        with torch.no_grad():
            output_ids = policy.generate(**inputs)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nOutput: {output_text}\n")

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()

    print("=== RLHF后模型输出 ===")
    rlhf_policy = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir, trust_remote_code=model_args.trust_remote_code
    ).to(policy.device)
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(rlhf_policy.device)
        with torch.no_grad():
            output_ids = rlhf_policy.generate(**inputs, max_new_tokens=50)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nOutput: {output_text}\n")
