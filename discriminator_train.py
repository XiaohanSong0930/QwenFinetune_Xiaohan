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
    offload_param_device=None
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
# def collate_fn(batch, processor, device):
#     from transformers import PreTrainedTokenizer
#     #questions = [item['question'] for item in batch]
#     # newly added 5.16
#     messages_batch = [
#         [
#             {"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": q}]}
#         ]
#         for q, path in zip(questions, image_paths)
#     ]
#     questions = [processor.apply_chat_template(m, tokenize=False) for m in messages_batch]
#     #end of newly added
#     image_paths = [item['image_paths'] for item in batch]
#     labels = [item['label'] for item in batch]

#     # 加载图片
#     flat_images = []
#     for paths in image_paths:
#         for path in paths:
#             img = Image.open(path).convert('RGB')
#             flat_images.append(img)

#     inputs = processor(
#         text=questions,
#         images=flat_images,
#         padding=True,
#         return_tensors='pt'
#     )

#     def move_to_device(x):
#         if isinstance(x, dict):
#             return {k: move_to_device(v) for k, v in x.items()}
#         elif isinstance(x, list):
#             return [move_to_device(v) for v in x]
#         elif isinstance(x, tuple):
#             return tuple(move_to_device(v) for v in x)
#         elif isinstance(x, torch.Tensor):
#             return x.to(device)
#         else:
#             return x

#     inputs = move_to_device(inputs)
#     labels = torch.tensor(labels, dtype=torch.long, device=device)

#     return inputs, labels

# newly added 5.16 (a new version of def collate_fn())
from PIL import Image

def collate_fn(batch, processor, device):
    texts = []
    all_images = []
    labels = []

    for item in batch:
        question = item['question']
        reasoning = item.get('reasoning', '')
        cot_instruction = item.get('cot_instruction', '')
        final_prompt = f"{question}\n\n{cot_instruction}\n\nReasoning: {reasoning}"
        image_paths = item['image_paths']
        label = item['label']

        # === 构建多图的对话格式 ===
        contents = []
        for _ in image_paths:
            contents.append({"type": "image", "image": "<image>"})  # Placeholder for vision tokens
        contents.append({"type": "text", "text": final_prompt})
        messages = [{"role": "user", "content": contents}]
        text = processor.apply_chat_template(messages, tokenize=False)

        texts.append(text)

        # === 加载所有图片 ===
        images = [Image.open(path).convert("RGB") for path in image_paths]
        all_images.append(images)
        labels.append(label)

    # === 调用 Processor ===
    inputs = processor(
        text=texts,
        images=all_images,   # 每个元素是一组图像
        padding=True,
        return_tensors="pt"
    )

    # === 移动到设备 ===
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return inputs, labels
# end of newly added

# ========== Discriminator Model ==========
class DiscriminatorModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classification_head = nn.Linear(hidden_size, 2)
        self.device = next(base_model.parameters()).device

    # def forward(self, **batch):
    #     debug_tensor_devices(batch, model=self.base_model)
    #     outputs = self.base_model(**batch, output_hidden_states=True)
    #     last_hidden_state = outputs.hidden_states[-1]
    #     cls_hidden = last_hidden_state[:, 0, :]  # [B, H]
    #     logits = self.classification_head(cls_hidden)
    #     return logits
    
    #newly added 5.16 (new version of def forward())
    def forward(self, **batch):
        outputs = self.base_model(**batch, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # ✅ 正确地获取最后一层 hidden state
        cls_hidden = last_hidden[:, 0, :]        # 通常第一 token 是 <bos>
        logits = self.classification_head(cls_hidden)
        return logits

    # end of newly added

def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved. 
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    I hope this bug will be fixed soon and I can remove this function then.
    '''
    
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")

# ========== Training Loop ==========
def train():
    # ---------- Accelerator / DeepSpeed ----------
    global accelerator, device

    # ---------- 日志 & 输出目录 ----------
    output_dir = f"train_output/{datetime.datetime.now():%Y%m%d%H%M%S}/"
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        init_logger(output_dir)
        logger = get_logger()
        logger.info(f"💡 训练输出目录：{output_dir}")

    # ---------- 模型 & Processor ----------
    base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    if accelerator.is_local_main_process:
        logger.info("🔄 加载基座模型与 Processor ...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(
        base_model_name,
        min_pixels=96 * 28 * 28,
        max_pixels=160 * 28 * 28,
        padding_side="right",
    )

    # ---------- 构建判别器包装 ----------
    model = DiscriminatorModel(base_model)

    # ---------- 数据 ----------
    dataset = DiscriminatorDataset("d_train_small.json")
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device),
    )

    # ---------- 优化器 ----------
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # ---------- 加速准备 ----------
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    model.train()

    # ---------- 训练循环 ----------
    num_epochs = 2
    for epoch in range(num_epochs):
        for step, (inputs, labels) in enumerate(train_loader):
            with accelerator.accumulate(model):
                logits = model(**inputs)
                loss = F.cross_entropy(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_local_main_process and step % 10 == 0:
                    logger.info(
                        f"[Ep {epoch+1}/{num_epochs}] "
                        f"Step {step:04d} | loss {loss.item():.6f}"
                    )

        # ---------- 保存 ----------
        accelerator.wait_for_everyone()

        ckpt_dir      = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
        base_ckpt_dir = os.path.join(ckpt_dir, "base_model")
        if accelerator.is_main_process:
            os.makedirs(base_ckpt_dir, exist_ok=True)

        if accelerator.is_main_process:
            logger.info("------ start gathering & saving ------")

            # ① ZeRO-3 分片一次性 all-gather 到 rank-0 CPU
            state_dict = accelerator.get_state_dict(model)          # ← 关键，只拷一次

            logger.info("------ allgather done ------")

            # ② 直接写单文件 .bin ；写完即得完整权重
            unwrapped_model = accelerator.unwrap_model(model)       # DeepSpeedEngine --> nn.Module
            
            logger.info("------ unwrap_model done ------")

            unwrapped_model.save_pretrained(
                base_ckpt_dir,
                is_main_process=True,
                save_function=accelerator.save,                      # 让 DeepSpeed 封装来写
                state_dict=state_dict,                               # **已在 CPU，无需再搬**
                safe_serialization=False,                            # 写 .bin 最快；若想要 .safetensors 改 True
                max_shard_size="20GB",                               # 合并成一个大文件
            )
            logger.info("--- base model save done ---")

            # ③ 保存分类头
            torch.save(
                unwrapped_model.classification_head.state_dict(),
                os.path.join(ckpt_dir, "classifier_head.pt")
            )
            logger.info("✅ classifier head saved")

            # ④ 最后一轮额外保存 processor & chat_template
            if epoch == num_epochs - 1:
                processor.save_pretrained(base_ckpt_dir)
                write_chat_template(processor, base_ckpt_dir)
                logger.info("✅ Processor & Chat-template saved")





if __name__ == "__main__":
    train()
