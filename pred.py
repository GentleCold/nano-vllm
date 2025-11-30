import argparse
import json
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm import LLM, SamplingParams


def load_qasper_dataset(data_dir):
    file_path = os.path.join(data_dir, "qasper.jsonl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    print(f"加载数据集: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading qasper"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}, 行内容: {line[:100]}...")
    
    print(f"成功加载 {len(data)} 条数据")
    return data

def get_pred(data, fout):
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/zwt/model/models/Qwen/Qwen3-8B/", trust_remote_code=True
    )

    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    prompts = []

    # 准备所有prompt
    for item in tqdm(data, desc="准备prompt"):
        
        question = item["input"].strip()
        context = item.get("context", "").strip()
        gold_answers = item.get("answers", [])

        choice_A = gold_answers[0] if len(gold_answers) > 0 else "Not mentioned in the context."
        choice_B = "This option is not supported by the context."
        choice_C = "This option contradicts the information in the document."
        choice_D = "None of the above."

        text = f"""Read the text and answer the question.
        <text>
        {context}
        </text>"""

        task = f"""Which is the correct choice to this question: {question}
        Choices:
        (A) {choice_A}
        (B) {choice_B}
        (C) {choice_C}
        (D) {choice_D}

        Respond with only one letter (A/B/C/D):"""

        # 构建完整的prompt
        prompt = text + "\n\n" + task

        max_len = 40000
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

        data_len = len(tokenizer(text).input_ids)
        prompt_len = len(tokenizer(prompt).input_ids)
        prompts.append({"text": prompt,
                       "task_start": data_len,  # task 从 data_len 开始
                       "prompt_len": prompt_len})
        
    # 批量生成
    print("开始预测...")
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=1))

    # 处理结果
    correct = 0
    for output, item in zip(outputs, data):
        response = output["text"].strip() if output["text"] else ""

        result_item = {
            "pred": response,
            "judge": response == 'A'
        }
        if response == "A":
            correct += 1
        
        fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        fout.flush()
    
    accuracy = correct / len(outputs)
    print(f"Accuracy:{accuracy}\n")

def main():
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"参数: {args}")
    
    out_file = os.path.join(args.save_dir, f"{args.model}.jsonl")
    try:
        data_all = load_qasper_dataset("/home/ld/data/data")
    except FileNotFoundError as e:
        return
    
    print(f"总数据量: {len(data_all)}")
    
    test_data = data_all[:args.test_samples]
    print(f"测试样本数: {len(test_data)}")
    
    with open(out_file, "w", encoding="utf-8") as fout:
        get_pred(test_data, fout)
    
    print(f"完成！结果保存至: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集评估脚本")
    parser.add_argument("--save_dir", "-s", type=str, default="results", help="结果保存目录")
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B", help="模型名称")
    parser.add_argument("--test_samples", type=int, default=200, help="测试样本数")
    
    args = parser.parse_args()
    main()