import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import datetime

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator, DeepSpeedPlugin
from functools import partial
from util.logutil import init_logger, get_logger
from PIL import Image

# ========== Init Accelerator ==========
deep_plugin = DeepSpeedPlugin(
    zero_stage=3,
    gradient_accumulation_steps=2,
    zero3_save_16bit_model=True,
    offload_optimizer_device="cpu",
    offload_param_device="cpu"
)
accelerator = Accelerator(deepspeed_plugin=deep_plugin)
device = accelerator.device

# ========== Logging ==========
output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()

# ========== Dataset ==========
class DiscriminatorDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def debug_tensor_devices(batch, model=None):
    print("=" * 40)
    print("🔍 Tensor device check:")
    model_device = next(model.parameters()).device if model is not None else torch.cuda.current_device()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:20}: {str(v.shape):20} on {v.device} {'✅' if v.device == model_device else '❌'}")
    print("Model expected device:", model_device)
    print("=" * 40)

# ========== Collate Function ==========
def collate_fn(batch, processor, device):
    questions = [item['question'] for item in batch]
    image_paths = [item['image_paths'] for item in batch]
    labels = [item['label'] for item in batch]

    # 加载图片
    flat_images = []
    for paths in image_paths:
        for path in paths:
            img = Image.open(path).convert('RGB')
            flat_images.append(img)

    inputs = processor(
        text=questions,
        images=flat_images,
        padding=True,
        return_tensors='pt'
    )

    def move_to_device(x):
        if isinstance(x, dict):
            return {k: move_to_device(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [move_to_device(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(move_to_device(v) for v in x)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return x

    inputs = move_to_device(inputs)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return inputs, labels

# ========== Discriminator Model ==========
class DiscriminatorModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classification_head = nn.Linear(hidden_size, 2)
        self.device = next(base_model.parameters()).device

    def forward(self, **batch):
        debug_tensor_devices(batch, model=self.base_model)
        outputs = self.base_model(**batch, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        cls_hidden = last_hidden_state[:, 0, :]  # [B, H]
        logits = self.classification_head(cls_hidden)
        return logits

# ========== Training Loop ==========
def train():
    if accelerator.is_local_main_process:
        logger.info("Loading model...")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cpu()

    model = DiscriminatorModel(base_model)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=96 * 28 * 28,
        max_pixels=160 * 28 * 28,
        padding_side="right"
    )

    dataset = DiscriminatorDataset("d_train.json")
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    optimizer = AdamW(model.parameters(), lr=1e-5)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model.train()

    num_epochs = 2
    for epoch in range(num_epochs):
        for step, (inputs, labels) in enumerate(train_loader):
            with accelerator.accumulate(model):
                logits = model(**inputs)
                loss = F.cross_entropy(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_local_main_process:
                    logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.6f}")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(output_dir, f"checkpoint-epoch{epoch+1}"),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )

if __name__ == "__main__":
    train()
