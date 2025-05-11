# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from accelerate import infer_auto_device_map
# import torch, json, re

# # Load the model on GPU
# device = torch.device("cuda:0")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16
# ).to(device)

# # Default processor
# processor = AutoProcessor.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     min_pixels=96*28*28,
#     max_pixels=160*28*28,
#     padding_side="left"
# )

# # 读取 test 数据
# with open("test_small.json", "r") as f:
#     test_data = json.load(f)

# correct = 0
# total = 0
# results = []
# reasoning_log = []  # 新增：记录推理内容

# for i, item in enumerate(test_data):
#     messages = item["messages"]

#     print(f"\n=== Inference for Sample {i+1} ===")

#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info([messages])

#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     ).to(device)

#     generated_ids = model.generate(**inputs, max_new_tokens=128)  # 适当加长输出以容纳完整推理内容
#     generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
#     output_text = processor.decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True
#     ).strip()

#     # 判断正确答案
#     if item['answer_idx'] == 0:
#         answer = "A"
#     elif item['answer_idx'] == 1:
#         answer = "B"
#     elif item['answer_idx'] == 2:
#         answer = "C"
#     elif item['answer_idx'] == 3:
#         answer = "D"

#     is_correct = 0
#     if answer in output_text:
#         is_correct = 1
#         correct += 1

#         # 如果答对了，从输出中提取推理部分并记录
#         reasoning_match = re.search(r"My reasoning:(.*?)(My answer:|$)", output_text, re.IGNORECASE | re.DOTALL)
#         if reasoning_match:
#             reasoning_text = reasoning_match.group(1).strip()
#         else:
#             reasoning_text = "[未能提取推理内容]"

#         reasoning_log.append({
#             "question_id": i + 1,
#             "reasoning": reasoning_text
#         })

#     total += 1
#     accuracy = correct / total

#     results.append({
#         "input": messages,
#         "model_output": output_text,
#         "correct_answer": item['answer'],
#         "is_correct": is_correct
#     })

#     print(f"output is:{output_text}; correct answer is: {answer}, is_correct:{is_correct}, accuracy: {accuracy:.2%}")

# # 输出整体准确率
# print(f"\n=== Final Accuracy: {accuracy:.2%} ({correct}/{total}) ===")

# # 保存全部结果
# with open("test_qtv_outputs_scored.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# # 保存正确样本的编号和推理内容
# with open("correct_reasoning_5000.json", "w", encoding="utf-8") as f:
#     json.dump(reasoning_log, f, indent=2, ensure_ascii=False)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import infer_auto_device_map
import torch, json, re

# Load the model on GPU
device = torch.device("cuda:4")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16
).to(device)

# Default processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=96*28*28,
    max_pixels=160*28*28,
    padding_side="left"
)

# 读取 test 数据
with open("test_small_filtered.json", "r") as f:
    test_data = json.load(f)

correct = 0
total = 0
results = []
reasoning_log = []         # 记录答对的推理
wrong_reasoning_log = []   # 记录答错的推理

for i, item in enumerate(test_data):
    messages = item["messages"]

    print(f"\n=== Inference for Sample {i+1} ===")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    # 判断正确答案
    answer = ["A", "B", "C", "D"][item['answer_idx']]

    is_correct = 0
    if answer in output_text:
        is_correct = 1
        correct += 1

    # 提取推理内容
    reasoning_match = re.search(r"My reasoning:(.*?)(My answer:|$)", output_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
    else:
        reasoning_text = "[未能提取推理内容]"

    if is_correct:
        reasoning_log.append({
            "question_id": i + 1,
            "reasoning": reasoning_text
        })
    else:
        wrong_reasoning_log.append({
            "question_id": i + 1,
            "reasoning": reasoning_text
        })

    total += 1
    accuracy = correct / total

    results.append({
        "input": messages,
        "model_output": output_text,
        "correct_answer": item['answer'],
        "is_correct": is_correct
    })

    print(f"output is:{output_text}; correct answer is: {answer}, is_correct:{is_correct}, accuracy: {accuracy:.2%}")

# 输出整体准确率
print(f"\n=== Final Accuracy: {accuracy:.2%} ({correct}/{total}) ===")

# 保存全部结果
with open("test_qtv_outputs_scored.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 保存答对的推理内容
with open("correct_reasoning_5000_new.json", "w", encoding="utf-8") as f:
    json.dump(reasoning_log, f, indent=2, ensure_ascii=False)

# 保存答错的推理内容
with open("wrong_reasoning_5000.json", "w", encoding="utf-8") as f:
    json.dump(wrong_reasoning_log, f, indent=2, ensure_ascii=False)
