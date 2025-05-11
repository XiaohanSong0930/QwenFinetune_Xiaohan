import json
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 设置路径
model_dir = "train_output/20250331192026/"
json_path = "test_data/test_qtva.json"
output_path = "test_qtv_outputs.json"

# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map={"": 0}
)
processor = AutoProcessor.from_pretrained(
    model_dir,
    min_pixels=96 * 28 * 28,
    max_pixels=160 * 28 * 28,
    padding_side="left"
)
model.eval()

# 读取测试数据
with open(json_path, "r") as f:
    test_data = json.load(f)

results = []
correct = 0

# 推理 loop
for i, item in enumerate(test_data):
    messages = item["messages"]
    gt_answer = item.get("answer", "N/A")  # 正确答案

    print(f"\n=== Inference for Sample {i+1} ===")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 加强提示，鼓励回答为 A/B/C/D
    if not text.strip().endswith("Answer with a letter A, B, C or D."):
        text += "\nAnswer with a letter A, B, C or D."

    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info([messages])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # 提取选项 A/B/C/D
    match = re.search(r"\b([ABCD])\b", output_text.strip(), re.IGNORECASE)
    pred = match.group(1).upper() if match else "N/A"

    is_correct = pred == gt_answer
    correct += int(is_correct)

    print(f"Prediction: {pred} | Ground Truth: {gt_answer} | Correct: {is_correct}")
    print(f"Model Output: {output_text}")

    results.append({
        "input": messages,
        "output": output_text,
        "predicted_answer": pred,
        "ground_truth": gt_answer,
        "correct": is_correct
    })

# 准确率统计
accuracy = correct / len(test_data)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}% ({correct}/{len(test_data)})")

# 保存预测结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✅ Results saved to {output_path}")
