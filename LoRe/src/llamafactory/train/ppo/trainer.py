# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import wandb
import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import json
import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from ...model import load_model, load_tokenizer
from ...extras.callbacks import FixValueHeadModelCallback, LogCallback
from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm
from ...data import process_dataset, get_dataset
from datasets import Dataset
from ...utils import utils

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        dataset: "Dataset",
        data_collator: "DataCollatorWithPadding",
        simulator

    ):
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        ppo_config.accelerator_kwargs["kwargs_handlers"] = [
            DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
        ]
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.simulator = simulator
        self.current_device = get_current_device()  # patch for deepspeed training
        self.processor = processor

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, FixValueHeadModelCallback)

        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        self.is_chatglm_model = getattr(unwrapped_model.config, "model_type", None) == "chatglm"

        self.amp_context = torch.autocast(self.current_device.type, dtype=self.model_args.compute_dtype)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)  # 多少个batch
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(count_parameters(self.model)[0]))

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)


        # play simulator


        self.simulator.load_simulator()
        self.simulator.play()
        wandb.init(
            # Set the project where this run will be logged
            project="basic-intro",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment{self.simulator.config['provider_decision_policy']}",
            # Track hyperparameters and run metadata
            config={
                "mode": "click_based",
                "epochs": self.simulator.config['round'],
                "finetune": self.simulator.config['finetune'],
                "lr": 1.0e-5,
                "gradient_accumulation_steps": 8,
                **self.simulator.config
            })
        # print(f'max step:{max_steps}')

        for step in tqdm(range(self.simulator.config['round']), disable=not self.is_local_process_zero(), desc="Epoch Processing..."):
            # try:
            #     batch = next(dataiter)
            # except StopIteration:
            #     dataiter = iter(self.dataloader)
            #     batch = next(dataiter)

            # simulator.
            # self.simulator.round()
            # Get inputs
            self.simulator.update_round(step)

            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            tokenizer_module = load_tokenizer(self.model_args)
            if self.simulator.config['provider_decision_policy'] not in ['random', 'LBR', 'CFD']:
                analyze_query_text = self.simulator.construct_analyze_prompts()
                # with open('/home/xiaopeng_ye/experiment/Agent4Fairness/LLaMA-Factory/data/fairagent.json', 'w', encoding='utf-8') as file:
                #     file.write(json.dumps(analyze_query_text, indent=2))  # 将json数组转化为文本字符,indent代表缩进两个字符
                with open('/home/chenming/mywork/study/RS/LLaMA-Factory/data/fairagent.json', 'w', encoding='utf-8') as file:
                    file.write(json.dumps(analyze_query_text, indent=2))  # 将json数组转化为文本字符,indent代表缩进两个字符
                # query_dataset = Dataset.from_list(query_text)

                # tokenizer = tokenizer_module["tokenizer"]
                # print(f'query_dataset:{query_dataset}')
                analyse_batches = get_dataset(model_args=self.model_args,
                                      data_args=self.data_args,
                                      training_args=self.args,
                                      stage="ppo",
                                      **tokenizer_module)
                # for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                analyze_responses_list = []
                analyze_prompt_list = [single_dict['instruction'] for single_dict in analyze_query_text]
                for single_batch in tqdm(analyse_batches, desc='analyze_batch_processing..'):  # 一个single batch代表providers
                    # print(f'batch:{single_batch}')
                    single_batch = {k: torch.tensor([v]).to('cuda') for k, v in single_batch.items()}
                    mini_batch_queries, mini_batch_responses = self.get_inputs(
                        single_batch  # [idx : idx + self.config.mini_batch_size]
                    )
                    responses_ids = [
                        output_ids for output_ids in mini_batch_responses
                    ]
                    responses_text = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)[0]
                    analyze_responses_list.append(responses_text)


                    queries_ids = [
                        output_ids for output_ids in mini_batch_queries
                    ]
                    queries_text = self.tokenizer.batch_decode(queries_ids, skip_special_tokens=True)[0]

                    # print(f'responses_text:{responses_text}')
                    # print(f'queries_text:{queries_text}')
            else:
                analyze_responses_list = None
                analyze_prompt_list = None
            generate_query_text, choose_genre = self.simulator.construct_generate_prompts(analyze_responses_list,
                                                                            analyze_prompt_list)
            # with open('/home/xiaopeng_ye/experiment/Agent4Fairness/LLaMA-Factory/data/fairagent.json', 'w', encoding='utf-8') as file:
            #     file.write(json.dumps(generate_query_text, indent=2))  # 将json数组转化为文本字符,indent代表缩进两个字符
            with open('/home/chenming/mywork/study/RS/LLaMA-Factory/data/fairagent.json', 'w', encoding='utf-8') as file:
                file.write(json.dumps(generate_query_text, indent=2))  # 将json数组转化为文本字符,indent代表缩进两个字符
            generate_batches = get_dataset(model_args=self.model_args,
                                           data_args=self.data_args,
                                           training_args=self.args,
                                           stage="ppo",
                                           **tokenizer_module)
            for single_batch in tqdm(generate_batches, desc='generate_batch_processing..'):  # 一个single batch代表每个provider
                # print(f'batch:{single_batch}')
                retries = 0
                while retries < 5:
                    try:
                        single_batch_cuda = {k: torch.tensor([v]).to('cuda') for k, v in single_batch.items()}
                        mini_batch_queries, mini_batch_responses = self.get_inputs(
                            single_batch_cuda  # [idx : idx + self.config.mini_batch_size]
                        )
                        self.upload_item_to_simulator(mini_batch_queries, mini_batch_responses, choose_genre)
                        break
                    except Exception as e:
                        print(f"Error occurred: {e}, retrying...")
                        retries += 1


                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                # rewards.extend(mini_batch_rewards)

            provider_click_dict, rewards, total_click, ctr_feedbacks = self.simulator.get_user_feedbacks()
            # rewards.extend(mini_batch_rewards)
            genre_count = self.simulator.get_genre_item_count()
            # provider_click_dict = {str(i):c for i,c in enumerate(clicks)}
            wandb.log({
                       "total_rewards": sum(rewards),
                       "total_click": total_click,
                        **genre_count,
                        **ctr_feedbacks,
                        **provider_click_dict
                      }
                      )
            import pandas as pd
            if self.simulator.round_cnt == 1:
                df = pd.DataFrame([provider_click_dict])
            else:
                df = pd.concat([df, pd.DataFrame([provider_click_dict])], ignore_index=True)
            # df.to_csv(f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/Bandwagon_effect/provider_dict.csv')
            df.to_csv(f'/home/chenming/mywork/study/RS/LLaMA-Factory/src/llamafactory/figures/Bandwagon_effect/provider_dict.csv')
            rewards = [torch.tensor(x).type(torch.float64).to('cuda') for x in rewards]
            if self.simulator.config['finetune'] == True:
            # Run PPO step
                import random
                batch_size = self.config.batch_size
                indices = random.sample(range(len(rewards)), batch_size)
                select_queries = [queries[i] for i in indices]
                select_responses = [responses[i] for i in indices]
                select_rewards = [rewards[i] for i in indices]

                self.model.train()
                stats = self.step(select_queries, select_responses, select_rewards)#self.step(queries, responses, rewards)
                self.tokenizer.padding_side = "left"  # restore padding side
                loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
                reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

                # if self.config.log_with is not None:
                #     try:
                #         batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                #         batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                #         self.log_stats(stats, batch, rewards)
                #     except Exception:
                #         logger.warning("Failed to save stats due to unknown errors.")

                self.state.global_step += 1
                self.log_callback.on_step_end(self.args, self.state, self.control)

                if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                    logs = dict(
                        loss=round(loss_meter.avg, 4),
                        reward=round(reward_meter.avg, 4),
                        learning_rate=stats["ppo/learning_rate"],
                        epoch=round(step / max_steps, 2), #steps_in_epoch
                    )
                    tqdm.write(str(logs))
                    logs["step"] = step
                    self.state.log_history.append(logs)
                    self.log_callback.on_log(self.args, self.state, self.control)
                    loss_meter.reset()
                    reward_meter.reset()

                if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                    self.save_model(
                        os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                    )
                    self.save_callback.on_save(
                        self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )
        wandb.finish()


    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimzer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, "torch.Tensor"]) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        r"""
        Generates model's responses given queries.
        """
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model = self.accelerator.unwrap_model(self.model)  # issue in trl v0.8.6
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()

        response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1  # allow empty response
            else:
                response_length = response_index[-1].item() + 1
            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right
        # print(f'provider queries:{queries}')
        # print(f'provider responses:{responses}')
        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List["torch.Tensor"],
        responses: List["torch.Tensor"],  # response
    ) -> List["torch.Tensor"]:
        r"""
        Computes scores using given reward model.

        Both inputs and outputs are put on CPU.
        """

        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            return get_rewards_from_server(self.reward_model, messages)
        # print(queries)
        # print(responses)
        batch = self.prepare_model_inputs(queries, responses)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        # print(f'batch:{batch}')
        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            _, _, values = reward_model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        # print(f'values:{values}')
        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        if self.is_chatglm_model:  # assume same architecture
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_indexes = (batch["input_ids"][i] != self.tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            rewards.append(values[i, end_index].float().detach().cpu())  # use fp32 type

        return rewards

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs)

            if self.is_chatglm_model:
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        if self.processor is not None and self.args.should_save:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)


    def upload_item_to_simulator(self, queries, responses, choose_genre):

        # token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]

        # token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
        queries_ids = [
            output_ids for output_ids in queries
        ]
        queries_text = self.tokenizer.batch_decode(queries_ids, skip_special_tokens=True)[0]
        # print(f'query_text:{queries_text}')
        provider_name = self.extract_provider_name(queries_text)


        responses_ids = [
            output_ids for output_ids in responses
        ]

        responses_text = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)[0]
        # print(f'massages:{messages}')

        # print(f'responses_text:{responses_text}')
        # if responses_text.find('STOP') != -1 or responses_text.find('Stop') != -1 or responses_text.find('stop') != -1:
        #     print(f'Stop Creating')
        #     self.simulator.upload_item('', '', '', provider_name)
        # else:
        item_name, item_genre, item_tags, item_description = utils.response_to_item(responses_text, choose_genre)
        self.simulator.upload_item(item_name, item_genre, item_tags, item_description, provider_name)


    def extract_provider_name(self, messages):
        provider_names = [provider['name'] for provider in self.simulator.data.providers.values()]
        # print(f'provider names:{provider_names}')
        for provider_name in provider_names:
            if messages.find(provider_name) != -1:
                return provider_name

        raise ValueError('Provider Not Found.')
