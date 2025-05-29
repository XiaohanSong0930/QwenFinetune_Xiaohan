import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

BASE_MODEL_DIR = "train_output/20250528024158/checkpoint-epoch2/base_model"
CLASSIFIER_PATH = "train_output/20250528024158/checkpoint-epoch2/classifier_head.pt"

sample = {
      "image_paths": [
        "train_data/real_slices/0-HM2VCdrC0$1.png",
        "train_data/real_slices/0-HM2VCdrC0$2.png",
        "train_data/real_slices/0-HM2VCdrC0$3.png",
        "train_data/real_slices/0-HM2VCdrC0$4.png",
        "train_data/real_slices/0-HM2VCdrC0$5.png",
        "train_data/real_slices/0-HM2VCdrC0$6.png",
        "train_data/real_slices/0-HM2VCdrC0$7.png",
        "train_data/real_slices/0-HM2VCdrC0$8.png"
      ],
      "question": "Transcript: 00:00:01.240 --> 00:00:04.950 God what why are you here wouldn't 00:00:04.950 --> 00:00:04.960 God what why are you here wouldn't 00:00:04.960 --> 00:00:06.910 God what why are you here wouldn't return my phone calls what do you want 00:00:06.910 --> 00:00:06.920 return my phone calls what do you want 00:00:06.920 --> 00:00:09.829 return my phone calls what do you want you want another picture your paper Jane 00:00:09.829 --> 00:00:09.839 you want another picture your paper Jane 00:00:09.839 --> 00:00:12.830 you want another picture your paper Jane I sorry please you used me to get ahead 00:00:12.830 --> 00:00:12.840 I sorry please you used me to get ahead 00:00:12.840 --> 00:00:15.590 I sorry please you used me to get ahead in your career be a man and admit it or 00:00:15.590 --> 00:00:15.600 in your career be a man and admit it or 00:00:15.600 --> 00:00:17.550 in your career be a man and admit it or or don't but please please don't pretend 00:00:17.550 --> 00:00:17.560 or don't but please please don't pretend 00:00:17.560 --> 00:00:20.429 or don't but please please don't pretend that you give a you just please let 00:00:20.429 --> 00:00:20.439 that you give a you just please let 00:00:20.439 --> 00:00:23.269 that you give a you just please let me explain no it doesn't matter I just 00:00:23.269 --> 00:00:23.279 me explain no it doesn't matter I just 00:00:23.279 --> 00:00:25.189 me explain no it doesn't matter I just destroyed my life and I didn't need your 00:00:25.189 --> 00:00:25.199 destroyed my life and I didn't need your 00:00:25.199 --> 00:00:27.670 destroyed my life and I didn't need your help to do it 00:00:27.670 --> 00:00:27.680 help to do it 00:00:27.680 --> 00:00:30.549 help to do it great finally I saw what you did there 00:00:30.549 --> 00:00:30.559 great finally I saw what you did there 00:00:30.559 --> 00:00:31.389 great finally I saw what you did there and you know what I thought it was 00:00:31.389 --> 00:00:31.399 and you know what I thought it was 00:00:31.399 --> 00:00:33.950 and you know what I thought it was amazing was it absolutely certifiably 00:00:33.950 --> 00:00:33.960 amazing was it absolutely certifiably 00:00:33.960 --> 00:00:35.790 amazing was it absolutely certifiably nuts yes it was but you did something 00:00:35.790 --> 00:00:35.800 nuts yes it was but you did something 00:00:35.800 --> 00:00:37.830 nuts yes it was but you did something Jane for the first time you were not 00:00:37.830 --> 00:00:37.840 Jane for the first time you were not 00:00:37.840 --> 00:00:40.029 Jane for the first time you were not just the perfect bridesmaid stop just 00:00:40.029 --> 00:00:40.039 just the perfect bridesmaid stop just 00:00:40.039 --> 00:00:41.549 just the perfect bridesmaid stop just please I'm I'm not doing this with you 00:00:41.549 --> 00:00:41.559 please I'm I'm not doing this with you 00:00:41.559 --> 00:00:42.830 please I'm I'm not doing this with you again I don't even know why I'm sitting 00:00:42.830 --> 00:00:42.840 again I don't even know why I'm sitting 00:00:42.840 --> 00:00:44.110 again I don't even know why I'm sitting here talking to you you know what let me 00:00:44.110 --> 00:00:44.120 here talking to you you know what let me 00:00:44.120 --> 00:00:47.310 here talking to you you know what let me tell look here listen to me do you want 00:00:47.310 --> 00:00:47.320 tell look here listen to me do you want 00:00:47.320 --> 00:00:48.950 tell look here listen to me do you want to know the real reason why I came here 00:00:48.950 --> 00:00:48.960 to know the real reason why I came here 00:00:48.960 --> 00:00:50.830 to know the real reason why I came here tonight because I knew this was going to 00:00:50.830 --> 00:00:50.840 tonight because I knew this was going to 00:00:50.840 --> 00:00:52.670 tonight because I knew this was going to be hard for you and for the first time 00:00:52.670 --> 00:00:52.680 be hard for you and for the first time 00:00:52.680 --> 00:00:54.270 be hard for you and for the first time in a really long time I wanted to be 00:00:54.270 --> 00:00:54.280 in a really long time I wanted to be 00:00:54.280 --> 00:00:56.990 in a really long time I wanted to be there for somebody yeah 00:00:56.990 --> 00:00:57.000 there for somebody yeah 00:00:57.000 --> 00:00:59.270 there for somebody yeah all right I I I messed up I did I'm 00:00:59.270 --> 00:00:59.280 all right I I I messed up I did I'm. **End of transcript** Combine images and transcript, answer this question: What is the man's attitude during the conversation?. Options: A) The man is contrite and so takes the woman's barbs with no complaint, knowing that she is justified in her anger.; B) The man is trying to sell a car to the woman.; C) He feels like he is being attacked during the argument.; D) The man is dismissive of the woman's concerns and starts talking over her.. Answer with a letter A or B or C or D.",
      "user_info": "",
      "cot_instruction": "Let's do this step by step.",
      "reasoning": "The man is seen apologizing multiple times and admitting his mistakes, indicating a contrite attitude. He also seems to be trying to understand the woman's perspective and apologize for past actions.",
      "label": 1
}


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classification_head = torch.nn.Linear(hidden_size, 2)

    def forward(self, **batch):
        outputs = self.base_model(**batch, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        cls_hidden = last_hidden[:, 0, :]
        return self.classification_head(cls_hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR)

model = DiscriminatorModel(base_model)
model.classification_head.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
model.to(device)
model.eval()


images = [Image.open(path).convert("RGB") for path in sample["image_paths"]]

final_prompt = f"{sample['question']}\n\n{sample.get('cot_instruction', '')}\n\nReasoning: {sample.get('reasoning', '')}"
contents = [{"type": "image", "image": "<image>"} for _ in images]
contents.append({"type": "text", "text": final_prompt})
messages = [{"role": "user", "content": contents}]
text = processor.apply_chat_template(messages, tokenize=False)

inputs = processor(
    text=[text],
    images=[images],
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs)
    pred = torch.argmax(logits, dim=-1).item()

print(f"Predicted Label: {pred} ({'real' if pred == 1 else 'fake'})")
