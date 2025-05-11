from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pprint import pprint

model_dir = "train_output/20250404113605/checkpoint-epoch2"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="bfloat16", device_map={"":0}
)


# default processer
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=96*28*28, max_pixels=160*28*28, padding_side="left")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

import json
with open("test_data/test_qtva.json") as f:
    test_data = json.load(f)

all_messages = [item["messages"] for item in test_data]

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

# Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )







# messages = [messages1, messages2, messages3]

# texts = [
#     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#     for msg in all_messages
# ]

# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=texts,
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# pprint(output_text)


import json

# # 加载你的测试数据（test_qtv.json）
# with open("test_data/test_qtv.json", "r") as f:
#     test_data = json.load(f)

# # 取出每一条 messages
# all_messages = [item["messages"] for item in test_data]

# results = []

# for i, messages in enumerate(all_messages):
#     print(f"\n=== Inference for Sample {i+1} ===")
    
#     # 构造文本 prompt
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     # 图像/视频处理（传入 messages 列表）
#     image_inputs, video_inputs = process_vision_info([messages])

#     # 处理成模型输入格式
#     inputs = processor(
#         text=[text],  # 注意包装成列表
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     ).to("cuda:0")

#     # 推理生成
#     generated_ids = model.generate(**inputs, max_new_tokens=128)

#     # 截掉 prompt 部分，仅保留模型生成的内容
#     generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
#     output_text = processor.decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )

#     print(f"Output: {output_text}")
#     results.append({"input": messages, "output": output_text})

# # 可选：保存结果为 json
# with open("test_qtv_outputs.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)


# 读取新的 test_data 格式
with open("test_data/test_qtva.json", "r") as f:
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
    generated_ids = model.generate(**inputs, max_new_tokens=16)
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    # 提取回答中的字母选项（只保留A/B/C/D）
    output_text = output_text.strip().strip(";").strip(".")
    len_output = len(output_text)
    answer = ""
    print(output_text)
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
    is_correct = 0
    if output_text != "" and output_text in item['answer']:
        is_correct = 1
        correct += 1
    # if output_text in item['answer']:
    #     is_correct = 1
    #     correct += 1
    total += 1
    accuracy = correct/total



    # 记录结果
    results.append({
        "input": messages,
        "model_output": output_text,
        "correct_answer": item['answer'],
        "is_correct": is_correct
    })
    print(f"output is:{output_text}; correct answer is {item['answer']}, is_correct:{is_correct}, accuracy: {accuracy:.2%}")

# 总体正确率
accuracy = correct / total if total > 0 else 0
print(f"\n=== Final Accuracy: {accuracy:.2%} ({correct}/{total}) ===")

# 可选保存
with open("test_qtv_outputs_scored.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
