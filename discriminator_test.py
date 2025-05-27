import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

BASE_MODEL_DIR = "train_output/20250525153739/checkpoint-epoch2/base_model"
CLASSIFIER_PATH = "train_output/20250525153739/checkpoint-epoch2/classifier_head.pt"

sample = {
    "image_paths": [
      "train_data/real_slices/0ajln01OfXs$1.png",
      "train_data/real_slices/0ajln01OfXs$2.png",
      "train_data/real_slices/0ajln01OfXs$3.png",
      "train_data/real_slices/0ajln01OfXs$4.png",
      "train_data/real_slices/0ajln01OfXs$5.png",
      "train_data/real_slices/0ajln01OfXs$6.png",
      "train_data/real_slices/0ajln01OfXs$7.png",
      "train_data/real_slices/0ajln01OfXs$8.png"
    ],
    "question": "Transcript: 00:00:01.390 --> 00:00:01.400 think I got it hey Susan how's your job 00:00:01.400 --> 00:00:03.509 think I got it hey Susan how's your job at Michael K going oh they offered me a 00:00:03.509 --> 00:00:03.519 at Michael K going oh they offered me a 00:00:03.519 --> 00:00:05.990 at Michael K going oh they offered me a full-time position in accessories design 00:00:05.990 --> 00:00:06.000 full-time position in accessories design 00:00:06.000 --> 00:00:08.350 full-time position in accessories design that's awesome I know yeah when things 00:00:08.350 --> 00:00:08.360 that's awesome I know yeah when things 00:00:08.360 --> 00:00:09.709 that's awesome I know yeah when things weren't taking off I thought Jim and I 00:00:09.709 --> 00:00:09.719 weren't taking off I thought Jim and I 00:00:09.719 --> 00:00:11.350 weren't taking off I thought Jim and I were just going to start having babies 00:00:11.350 --> 00:00:11.360 were just going to start having babies 00:00:11.360 --> 00:00:13.389 were just going to start having babies but Hannah really encouraged me to keep 00:00:13.389 --> 00:00:13.399 but Hannah really encouraged me to keep 00:00:13.399 --> 00:00:15.990 but Hannah really encouraged me to keep going whenever I think of giving up I 00:00:15.990 --> 00:00:16.000 going whenever I think of giving up I 00:00:16.000 --> 00:00:17.830 going whenever I think of giving up I think of Hannah I mean I didn't 00:00:17.830 --> 00:00:17.840 think of Hannah I mean I didn't 00:00:17.840 --> 00:00:19.350 think of Hannah I mean I didn't sacrifice everything just to get married 00:00:19.350 --> 00:00:19.360 sacrifice everything just to get married 00:00:19.360 --> 00:00:21.150 sacrifice everything just to get married and have kids you know I came here for 00:00:21.150 --> 00:00:21.160 and have kids you know I came here for 00:00:21.160 --> 00:00:22.830 and have kids you know I came here for my dreams and I take that really 00:00:22.830 --> 00:00:22.840 my dreams and I take that really 00:00:22.840 --> 00:00:25.470 my dreams and I take that really seriously yeah wait a minute are you 00:00:25.470 --> 00:00:25.480 seriously yeah wait a minute are you 00:00:25.480 --> 00:00:27.270 seriously yeah wait a minute are you guys saying that you don't think I take 00:00:27.270 --> 00:00:27.280 guys saying that you don't think I take 00:00:27.280 --> 00:00:28.869 guys saying that you don't think I take my dream seriously because I got 00:00:28.869 --> 00:00:28.879 my dream seriously because I got 00:00:28.879 --> 00:00:30.590 my dream seriously because I got pregnant that's not what she's saying 00:00:30.590 --> 00:00:30.600 pregnant that's not what she's saying 00:00:30.600 --> 00:00:32.590 pregnant that's not what she's saying that's exactly what she just said you 00:00:32.590 --> 00:00:32.600 that's exactly what she just said you 00:00:32.600 --> 00:00:34.310 that's exactly what she just said you did get give up Project Runway they 00:00:34.310 --> 00:00:34.320 did get give up Project Runway they 00:00:34.320 --> 00:00:36.509 did get give up Project Runway they pushed the show 6 months and your due 00:00:36.509 --> 00:00:36.519 pushed the show 6 months and your due 00:00:36.519 --> 00:00:38.070 pushed the show 6 months and your due date landed smack in the middle of the 00:00:38.070 --> 00:00:38.080 date landed smack in the middle of the 00:00:38.080 --> 00:00:40.350 date landed smack in the middle of the show so it did kind of get in the way 00:00:40.350 --> 00:00:40.360 show so it did kind of get in the way 00:00:40.360 --> 00:00:42.429 show so it did kind of get in the way look you don't have to get mad okay it's 00:00:42.429 --> 00:00:42.439 look you don't have to get mad okay it's 00:00:42.439 --> 00:00:44.749 look you don't have to get mad okay it's just a different decision 00:00:44.749 --> 00:00:44.759 just a different decision 00:00:44.759 --> 00:00:48.789 just a different decision alen is that your ex oh 00:00:48.789 --> 00:00:48.799 alen is that your ex oh 00:00:48.799 --> 00:00:55.789 alen is that your ex oh who runs in a tank what a douche 00:00:55.789 --> 00:00:55.799. **End of transcript** Combine images and transcript, answer this question: Why does the man start running towards the pregnant woman?. Options: A) He wants to ask her for directions.; B) He is trying to catch his dog who ran towards the pregnant woman.; C) She dropped something, and he wants to return it.; D) He realizes that she is his ex-girlfriend.. Answer with a letter A or B or C or D.",
    "user_info": "",
    "cot_instruction": "Let's do this step by step.",
    "reasoning": "The transcript mentions that the man starts running towards the pregnant woman after she says \"alén is that your ex.\" This suggests that he recognizes her as his ex-girlfriend."
  }


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.classification_head = torch.nn.Linear(base_model.config.hidden_size, 2)

    def forward(self, **batch):
        outputs = self.base_model(**batch, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        cls_hidden = last_hidden[:, 0, :]
        return self.classification_head(cls_hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR)

model = DiscriminatorModel(base_model)
model.classification_head.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
model.to(device)
model.eval()


images = [Image.open(path).convert("RGB") for path in sample["image_paths"]]
question = sample["question"]

messages = [{
    "role": "user",
    "content": [{"type": "image", "image": "<image>"} for _ in images] + [{"type": "text", "text": question}]
}]
text = processor.apply_chat_template(messages, tokenize=False)

inputs = processor(
    text=[text],
    images=[images],
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    logits = model(**inputs)
    pred = torch.argmax(logits, dim=-1).item()

print(f"Predicted Label: {pred} ({'fake' if pred == 1 else 'real'})")
