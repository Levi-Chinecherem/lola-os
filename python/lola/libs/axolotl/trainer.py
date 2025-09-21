# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import os
from pathlib import Path
import yaml
import subprocess
from contextlib import asynccontextmanager
import time
from dataclasses import dataclass, asdict
from enum import Enum
from unittest.mock import Mock
import uuid

# Third-party
try:
    import wandb
    import datasets
    from transformers import TrainingArguments
    import torch
except ImportError:
    raise ImportError("Axolotl training dependencies missing. Run 'poetry add axolotl wandb datasets transformers torch'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.libs.prometheus.exporter import get_lola_prometheus
from lola.libs.wandb.logger import get_wandb_logger
from sentry_sdk import capture_exception, start_transaction

"""
File: Axolotl fine-tuning integration for LOLA OS Model Garden.
Purpose: Provides automated fine-tuning workflows for open-weight models using 
         Axolotl's configuration-driven training with LOLA-specific data 
         preparation, evaluation, and deployment hooks.
How: Generates Axolotl YAML configs from LOLA specifications; manages dataset 
     preparation, training jobs, evaluation loops, and model export; integrates 
     with W&B for experiment tracking and Hugging Face for model sharing.
Why: Enables continuous agent improvement through automated fine-tuning loops, 
     creating the "Model Garden" where agents get smarter over time using 
     real interaction data while maintaining developer sovereignty.
Full Path: lola-os/python/lola/libs/axolotl/trainer.py
"""

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning job."""
    base_model: str = "meta-llama/Llama-2-7b-hf"
    model_type: str = "llama"
    tokenizer_type: str = "LlamaTokenizer"
    
    # Training parameters
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    
    # Dataset
    train_dataset: str = "lola_training_data.jsonl"
    eval_dataset: Optional[str] = "lola_eval_data.jsonl"
    dataset_prepared_path: Optional[str] = None
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    max_steps: int = -1
    
    # Output
    output_dir: str = "./lola_finetuned_models"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    
    # Hardware
    per_device_train_batch_size: Optional[int] = None
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False
    
    # Experiment tracking
    wandb_project: str = "lola-model-garden"
    wandb_run_name: Optional[str] = None
    report_to: List[str] = None

class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LolaAxolotlTrainer:
    """LolaAxolotlTrainer: Automated fine-tuning for LOLA Model Garden.
    Does NOT execute training directly—generates configs and manages jobs."""

    DEFAULT_CONFIG_TEMPLATE = """
base_model: {{ base_model }}
model_type: {{ model_type }}
tokenizer_type: {{ tokenizer_type }}

load_in_8bit: false
load_in_4bit: false

datasets:
  - path: {{ train_dataset }}
    type: alpaca  # Assuming alpaca format
    formatting: prompt

{% if eval_dataset %}
  - path: {{ eval_dataset }}
    type: alpaca
    formatting: prompt
    eval_split: test
{% endif %}

sequence_len: 2048
sample_packing: true

{% if use_lora %}
lora_r: {{ lora_r }}
lora_alpha: {{ lora_alpha }}
lora_dropout: {{ lora_dropout }}
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
{% endif %}

micro_batch_size: {{ micro_batch_size }}
gradient_accumulation_steps: {{ gradient_accumulation_steps }}
num_epochs: {{ num_epochs }}
learning_rate: {{ learning_rate }}
lr_scheduler: {{ lr_scheduler }}
warmup_steps: {{ warmup_steps }}

{% if eval_steps %}
eval_steps: {{ eval_steps }}
{% endif %}

save_steps: {{ save_steps }}
logging_steps: {{ logging_steps }}
max_steps: {{ max_steps }}

output_dir: {{ output_dir }}
save_strategy: {{ save_strategy }}
load_best_model_at_end: {{ load_best_model_at_end }}

{% if gradient_checkpointing %}
gradient_checkpointing: {{ gradient_checkpointing }}
{% endif %}

{% if fp16 %}
fp16: {{ fp16 }}
{% endif %}

{% if bf16 %}
bf16: {{ bf16 }}
{% endif %}

{% if wandb_project %}
wandb_project: {{ wandb_project }}
{% if wandb_run_name %}
wandb_run_name: {{ wandb_run_name }}
{% endif %}
report_to: {{ report_to | default(['wandb']) }}
{% endif %}
"""

    def __init__(self):
        """
        Initializes Axolotl trainer with LOLA configuration.
        Does Not: Load models—lazy loading during training.
        """
        config = get_config()
        self.enabled = config.get("model_garden_enabled", False)
        self.output_base_dir = config.get("model_garden_output_dir", "./lola_models")
        self.max_parallel_jobs = config.get("model_garden_max_jobs", 1)
        self.auto_evaluate = config.get("model_garden_auto_eval", True)
        
        # Observability
        self.prometheus = get_lola_prometheus()
        self.wandb_logger = get_wandb_logger()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # Job management
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        
        Path(self.output_base_dir).mkdir(exist_ok=True)
        
        logger.info(f"Axolotl trainer initialized: {self.output_base_dir}")

    async def fine_tune(
        self, 
        config: FineTuneConfig, 
        dataset_path: Optional[str] = None,
        eval_dataset_path: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Starts fine-tuning job with Axolotl.
        Args:
            config: Fine-tuning configuration.
            dataset_path: Path to training dataset (JSONL/CSV).
            eval_dataset_path: Path to evaluation dataset.
            run_name: Optional run name for tracking.
        Returns:
            Job ID for tracking.
        """
        if not self.enabled:
            raise RuntimeError("Model Garden not enabled in configuration")

        job_id = str(uuid.uuid4())
        run_name = run_name or f"lola-finetune-{int(time.time())}"
        
        try:
            # Start metrics transaction
            with start_transaction(
                op="model_garden.finetune",
                name=f"Fine-tune {config.base_model}"
            ) as transaction:
                
                # Prepare dataset
                if dataset_path:
                    await self._prepare_dataset(dataset_path, config, job_id)
                
                if eval_dataset_path:
                    await self._prepare_eval_dataset(eval_dataset_path, config, job_id)
                
                # Generate Axolotl config
                axolotl_config = self._generate_axolotl_config(config, run_name, job_id)
                
                # Start W&B run
                wandb_run = self.wandb_logger.start_run(
                    project="lola-model-garden",
                    name=run_name,
                    config=asdict(config)
                )
                
                # Launch training job
                job_status = await self._launch_training_job(
                    axolotl_config, 
                    config.output_dir,
                    job_id,
                    wandb_run
                )
                
                # Record completion
                self.completed_jobs[job_id] = {
                    "config": config,
                    "status": job_status,
                    "run_name": run_name,
                    "wandb_run": wandb_run,
                    "timestamp": time.time(),
                    "output_dir": config.output_dir
                }
                
                logger.info(f"Fine-tuning job {job_id} completed: {job_status}")
                return job_id
                
        except Exception as exc:
            self._handle_error(exc, f"fine-tune {config.base_model}")
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            raise

    async def _prepare_dataset(self, dataset_path: str, config: FineTuneConfig, 
                              job_id: str) -> None:
        """
        Prepares training dataset in Axolotl format.
        Args:
            dataset_path: Path to raw dataset.
            config: Training configuration.
            job_id: Job identifier.
        """
        try:
            # Load dataset
            if dataset_path.endswith('.jsonl'):
                dataset = datasets.load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith(('.csv', '.tsv')):
                dataset = datasets.load_dataset('csv', data_files=dataset_path, split='train')
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
            
            # LOLA-specific data preparation
            def format_example(example):
                # Assuming alpaca format: instruction, input, output
                if 'instruction' in example and 'output' in example:
                    return {
                        "instruction": example['instruction'],
                        "input": example.get('input', ''),
                        "output": example['output']
                    }
                else:
                    # Generic chat format
                    return {
                        "instruction": "Respond to the user query.",
                        "input": example.get('query', example.get('prompt', '')),
                        "output": example.get('response', example.get('completion', ''))
                    }
            
            formatted_dataset = dataset.map(format_example)
            
            # Save in Axolotl format
            output_path = Path(config.output_dir) / f"{job_id}_training_data.jsonl"
            formatted_dataset.to_json(output_path, orient='records', lines=True)
            
            config.train_dataset = str(output_path)
            logger.info(f"Training dataset prepared: {len(formatted_dataset)} examples")
            
        except Exception as exc:
            self._handle_error(exc, f"dataset preparation {dataset_path}")
            raise

    async def _prepare_eval_dataset(self, dataset_path: str, config: FineTuneConfig, 
                                   job_id: str) -> None:
        """
        Prepares evaluation dataset.
        """
        try:
            # Similar to training dataset but smaller
            if dataset_path.endswith('.jsonl'):
                dataset = datasets.load_dataset('json', data_files=dataset_path, split='train')
            else:
                raise ValueError(f"Unsupported eval format: {dataset_path}")
            
            # Take first 100 examples for evaluation
            eval_dataset = dataset.select(range(min(100, len(dataset))))
            
            output_path = Path(config.output_dir) / f"{job_id}_eval_data.jsonl"
            eval_dataset.to_json(output_path, orient='records', lines=True)
            
            config.eval_dataset = str(output_path)
            logger.info(f"Evaluation dataset prepared: {len(eval_dataset)} examples")
            
        except Exception as exc:
            self._handle_error(exc, f"eval dataset preparation {dataset_path}")
            raise

    def _generate_axolotl_config(self, config: FineTuneConfig, run_name: str, 
                                job_id: str) -> Dict[str, Any]:
        """
        Generates Axolotl YAML configuration from LOLA config.
        Args:
            config: LOLA fine-tune configuration.
            run_name: W&B run name.
            job_id: Job identifier.
        Returns:
            Axolotl configuration dictionary.
        """
        try:
            # Create output directory
            output_dir = Path(config.output_dir) / f"{job_id}_{run_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate config from template
            config_dict = asdict(config)
            config_dict.update({
                "wandb_run_name": run_name,
                "report_to": ["wandb"]
            })
            
            axolotl_config = yaml.safe_load(self.DEFAULT_CONFIG_TEMPLATE)
            # Update with LOLA config values
            for key, value in config_dict.items():
                if key in axolotl_config:
                    axolotl_config[key] = value
            
            # Save config
            config_path = output_dir / "axolotl_config.yml"
            with open(config_path, "w") as f:
                yaml.dump(axolotl_config, f)
            
            logger.debug(f"Axolotl config generated: {config_path}")
            return axolotl_config
            
        except Exception as exc:
            self._handle_error(exc, "config generation")
            raise

    async def _launch_training_job(self, axolotl_config: Dict[str, Any], 
                                  output_dir: str, job_id: str, 
                                  wandb_run) -> str:
        """
        Launches Axolotl training job.
        Args:
            axolotl_config: Generated Axolotl configuration.
            output_dir: Output directory.
            job_id: Job ID.
            wandb_run: W&B run instance.
        Returns:
            Job status.
        """
        try:
            # Start Prometheus transaction
            with self.prometheus.start_agent_run(
                Mock(), "model_finetuning"
            ) as timing:
                
                # Launch training process
                cmd = [
                    "accelerate", "launch",
                    "--config_file", "default_axolotl_config.yaml",  # Axolotl default
                    "axolotl.cli.train",
                    str(Path(output_dir) / "axolotl_config.yml")
                ]
                
                # Start process
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Monitor process
                self.active_jobs[job_id] = {
                    "process": process,
                    "start_time": time.time(),
                    "wandb_run": wandb_run,
                    "status": TrainingStatus.RUNNING
                }
                
                # Stream output to W&B
                async for line in process.stdout:
                    decoded_line = line.decode().strip()
                    if decoded_line:
                        wandb_run.log({"training_log": decoded_line})
                        logger.debug(f"Training [{job_id}]: {decoded_line[:100]}...")
                
                # Wait for completion
                return_code = await process.wait()
                
                duration = time.time() - self.active_jobs[job_id]["start_time"]
                
                if return_code == 0:
                    status = TrainingStatus.COMPLETED
                    logger.info(f"Training job {job_id} completed in {duration:.1f}s")
                else:
                    status = TrainingStatus.FAILED
                    logger.error(f"Training job {job_id} failed with code {return_code}")
                
                # Cleanup
                del self.active_jobs[job_id]
                self.completed_jobs[job_id]["status"] = status
                self.completed_jobs[job_id]["duration"] = duration
                
                return status.value
                
        except Exception as exc:
            status = TrainingStatus.FAILED
            self._handle_error(exc, f"training job {job_id}")
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            return status.value

    async def evaluate_model(self, model_path: str, eval_dataset: str, 
                           job_id: str) -> Dict[str, Any]:
        """
        Evaluates fine-tuned model.
        Args:
            model_path: Path to fine-tuned model.
            eval_dataset: Evaluation dataset path.
            job_id: Job identifier.
        Returns:
            Evaluation metrics dictionary.
        """
        if not self.auto_evaluate:
            logger.info("Model evaluation disabled")
            return {"status": "skipped"}

        try:
            # Load evaluation dataset
            dataset = datasets.load_dataset('json', data_files=eval_dataset, split='train')
            
            # Simple evaluation: perplexity and sample generation
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            metrics = {
                "perplexity": self._calculate_perplexity(model, tokenizer, dataset),
                "generation_samples": self._generate_samples(model, tokenizer, dataset[:3]),
                "eval_dataset_size": len(dataset)
            }
            
            # Log to W&B
            if job_id in self.completed_jobs:
                wandb_run = self.completed_jobs[job_id].get("wandb_run")
                if wandb_run:
                    wandb_run.log(metrics)
                    wandb_run.finish()
            
            logger.info(f"Model evaluation completed for job {job_id}: perplexity {metrics['perplexity']:.3f}")
            return metrics
            
        except Exception as exc:
            self._handle_error(exc, f"model evaluation {job_id}")
            return {"status": "failed", "error": str(exc)}

    def _calculate_perplexity(self, model, tokenizer, dataset) -> float:
        """
        Calculates model perplexity on evaluation dataset.
        """
        try:
            model.eval()
            total_loss = 0
            total_tokens = 0
            
            for example in dataset:
                inputs = tokenizer(
                    example['input'] + example['output'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].numel()
                    total_tokens += inputs["input_ids"].numel()
            
            perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
            return perplexity.item()
            
        except Exception as exc:
            logger.warning(f"Perplexity calculation failed: {str(exc)}")
            return float('inf')

    def _generate_samples(self, model, tokenizer, examples) -> List[str]:
        """
        Generates sample outputs from evaluation examples.
        """
        try:
            model.eval()
            samples = []
            
            for example in examples:
                inputs = tokenizer(example['input'], return_tensors="pt", truncation=True, max_length=256)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prompt_end = generated.find(example['output'][:20])  # Find where prompt ends
                if prompt_end != -1:
                    generated = generated[prompt_end:]
                
                samples.append(generated.strip())
            
            return samples
            
        except Exception as exc:
            logger.warning(f"Sample generation failed: {str(exc)}")
            return ["Error generating samples"]

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Error handling for training operations.
        """
        logger.error(f"Axolotl trainer {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets status of training job.
        Args:
            job_id: Job identifier.
        Returns:
            Job status dictionary or None if not found.
        """
        if job_id in self.active_jobs:
            return {
                "status": self.active_jobs[job_id]["status"].value,
                "progress": "running",
                "start_time": self.active_jobs[job_id]["start_time"]
            }
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            return None


# Global trainer instance
_lola_axolotl_trainer = None

def get_axolotl_trainer() -> LolaAxolotlTrainer:
    """Singleton Axolotl trainer instance."""
    global _lola_axolotl_trainer
    if _lola_axolotl_trainer is None:
        _lola_axolotl_trainer = LolaAxolotlTrainer()
    return _lola_axolotl_trainer

__all__ = [
    "FineTuneConfig",
    "TrainingStatus",
    "LolaAxolotlTrainer",
    "get_axolotl_trainer"
]