from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import infer_auto_device_map
import torch,json

# default: Load the model on the available device(s)
device = torch.device("cuda:0")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="bfloat16"
).to(device)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",min_pixels=96*28*28, max_pixels=160*28*28, padding_side="left")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test_data/4.png",
            },
            {"type": "text", "text": "描述一下这个图片"},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test_data/5.png",
            },
            {"type": "text", "text": "描述一下这个图片"},
        ],
    }
]

messages3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "test_data/1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "描述一下这个视频"},
        ],
    }
]


# messages = [messages1, messages2, messages3]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# if isinstance(text, str):
#     text = [text]
# inputs = processor(
#     text=text,
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids = generated_ids.to(device)

# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)


# 读取新的 test_data 格式
with open("test_80.json", "r") as f:
    test_data = json.load(f)

correct = 0
total = 0
results = []

for i, item in enumerate(test_data):
    messages = item["messages"]
    #correct_answer = item.get("answer", "").strip().upper()  # A/B/C/D

    print(f"\n=== Inference for Sample {i+1} ===")

    # 构造文本 prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 图像/视频处理
    image_inputs, video_inputs = process_vision_info([messages])

    # 模型输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    # 模型生成
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    # 提取回答中的字母选项（只保留A/B/C/D）
    output_text = output_text.strip().strip(";").strip(".")
    len_output = len(output_text)
    # answer = ""
    # if output_text == item['A'][:len_output]:
    #     answer = "A"
    # elif output_text == item['B'][:len_output]:
    #     answer = "B"
    # elif output_text == item['C'][:len_output]:
    #     answer = "C"
    # elif output_text == item['D'][:len_output]:
    #     answer = "D"
    # else:
    #     print(output_text)
    # answer = output_text
    is_correct = 0
    # if answer != "" and answer in correct_answer:
    #     is_correct = 1
    #     correct += 1
    # total += 1
    # accuracy = correct/total
    if item['answer_idx'] == 0:
        answer = "A"
    elif item['answer_idx'] == 1:
        answer = "B"
    elif item['answer_idx'] == 2:
        answer = "C"
    elif item['answer_idx'] == 3:
        answer = "D"
    if answer in output_text:
        is_correct = 1
        correct += 1
    total += 1
    accuracy = correct/total




    # 记录结果
    results.append({
        "input": messages,
        "model_output": output_text,
        
        "correct_answer": item['answer'],
        "is_correct": is_correct
    })
    print(f"output is:{output_text}; correct answer is: {answer}, is_correct:{is_correct}, accuracy: {accuracy:.2%}")

# 总体正确率
accuracy = correct / total if total > 0 else 0
print(f"\n=== Final Accuracy: {accuracy:.2%} ({correct}/{total}) ===")

# 可选保存
with open("test_qtv_outputs_scored.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

