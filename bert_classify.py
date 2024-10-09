import os
import random
import torch
import time
import sys
import subprocess
import logging

log = logging.getLogger('bert_classify')
log.setLevel(logging.INFO)

try:
    import torch_npu
    print("torch npu available:", torch_npu.npu.is_available())
    from torch_npu.contrib import transfer_to_npu
    print("torch cuda available:", torch.cuda.is_available())
except Exception as e:
    print("import torch_npu not available:", e)

try:
    from extract_ict_qa import extract_common_sec_qa, extract_reformat_sec_event_disposal, parse_script_qa, \
        extract_nl2flinksql_qa, extract_sec_event_qa, parse_nl2sql_qa, parse_verify_disposal_qa
    from parse_cissp import extract_cissp
except Exception as e:
    print("import extract data not available:", e)

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertModel


# bge_model_name = "BAAI/bge-large-zh-v1.5"
bge_model_name = "./bge-large-zh-v1.5"
bert_model_name = './bert-base-uncased'

model_path = "bge_text_classifier.pt"
# model_path = "bge_text_classifier_20240520.pt"
need_force_retrain = True
# need_force_retrain = False
trained_epochs = 1
# Define label constants
LABELS = {
    "COMMON_SEC_QA": 0, # 通用安全问答
    # "OTHER_QA": 1, # 其他问题，不便归类的放在这里
    # "TRACING_QA": 3, # 安全事件调查
    # "SCRIPT_QA": 1, # 攻击脚本研判
    "NL2FLINKSQL_QA": 1, # 自然语言转Flink SQL
    "VERIFY_DISPOSAL_QA": 2  # 安全事件研判和处置建议
}

number_class = len(LABELS)

class TextClassifier:
    def __init__(self, model_name=bge_model_name, num_labels=number_class, need_freeze=False, force_retrain=need_force_retrain):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        # model_path = 'saved_model.pth'
        if os.path.exists(model_path) and not force_retrain:
            # Load saved model
            self.tokenizer, self.model = self.load_model(model_path, model_name, num_labels)
        else:
            # 初始化新模型
            self.tokenizer = BertTokenizer.from_pretrained(model_name, from_local_files=True)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # 设置GPU
            self.model.to(self.device)

        if need_freeze:
            # 冻结预训练模型的所有参数
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            # 创建优化器，注意这里我们只传递requires_grad=True的参数
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=2e-5)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5)


    def train(self, train_dataset, test_dataset, epochs=4, batch_size=64):
        dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print('-' * 10)
            total_loss = 0
            self.model.train()
            for step, batch in enumerate(dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                self.model.zero_grad()
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_train_loss = total_loss / len(dataloader)
            print(f"Average training loss: {avg_train_loss:.2f}")
            # Evaluate model after each epoch
            test_accuracy, test_loss = self.evaluate(test_dataset, batch_size=batch_size)
            print(f"Test Accuracy after epoch {epoch + 1}: {test_accuracy:.4f} {test_loss:.4f}")


    @staticmethod
    def load_model(file_path=model_path, model_name=bge_model_name, num_labels=number_class):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("test model embedding:", "bert.embeddings.position_ids" in torch.load(file_path).keys())
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # model.load_state_dict(torch.load(file_path))
        model.load_state_dict(torch.load(file_path), strict=False)
        print("test model embedding:", "bert.embeddings.position_ids" in torch.load(file_path).keys())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("loaded_model device:", device)
        model.to(device)
        return tokenizer, model

    def evaluate(self, dataset, batch_size=64):
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        total_loss = 0
        correct = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(logits, b_labels)
                predictions = torch.argmax(logits, dim=1)
                total_loss += loss.item() * b_labels.size(0)
                total_samples += b_labels.size(0)
                correct += (predictions == b_labels).sum().item()

        average_loss = total_loss / total_samples
        accuracy = correct / total_samples
        return accuracy, average_loss

    def predict(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            probabilities = torch.softmax(logits, dim=1).tolist()[0]
        return [predicted_class_id, probabilities]

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        try:
            import moxing as mox
            mox.file.copy_parallel(file_path, f"s3://bucket-sec-llm-wl/bert_classify/saved_model/{file_path}")
            print("copy model OK.")
        except Exception as e:
            print("copy model failed:", e)

    def prepare_data(self, texts, labels, max_length=64):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        return TensorDataset(input_ids, attention_masks, labels)

# 定义训练数据
# 假设的数据集
def get_fake_train_test_data():
    # 使用示例数据
    trained_texts = ["如何配置防火墙产品？", "How to configure firewall product?", "EDR product performance?",
                     "This event is attack successful?", "What is SQL injection?"]
    trained_labels = [1, 1, 1, 0, 0]  # 1代表通用场景，0代表其他场景
    test_texts = ["How to configure firewall?", "防火墙的产品配置是如何做？", "Give me as SQL injection attack example?"]
    test_labels = [1, 1, 0]
    return trained_texts, trained_labels, test_texts, test_labels


def get_real_train_test_data():
    # Extract data from different QA sources
    qa_data = {
        "COMMON_SEC_QA":  extract_common_sec_qa() + extract_sec_event_qa(("./LLM自动标注语料/通用安全问答",), reformat_type="no_type"), #  (cissp + 数通L1 + GPT生成)
        # "OTHER_QA": extract_sec_event_qa(("./LLM自动标注语料/其他", ), reformat_type="no_type"),
        # "TRACING_QA": extract_sec_event_qa(("./LLM自动标注语料/安全事件调查", "./人工生成语料/安全事件溯源"), reformat_type="no_type"),
        # "SCRIPT_QA": parse_script_qa(),
        "NL2FLINKSQL_QA": extract_nl2flinksql_qa(),
        "VERIFY_DISPOSAL_QA": extract_sec_event_qa(("./人工生成语料/安全事件研判与处置",), reformat_type="no_type") + parse_verify_disposal_qa() + parse_script_qa()
    }

    # Shuffle the data for each QA source
    for qa_type, qa_list in qa_data.items():
        random.shuffle(qa_list)

    # Combine all data and labels
    all_texts = [text for qa_type, qa_list in qa_data.items() for text in qa_list]
    all_labels = [label for qa_type, qa_list in qa_data.items() for label in [LABELS[qa_type]] * len(qa_list)]

    # 将数据和标签一起打乱
    data = list(zip(all_texts, all_labels))
    random.shuffle(data)
    all_texts, all_labels = zip(*data)

    # 按照9:1的比例拆分训练集和测试集
    train_size = int(len(all_texts) * 0.9)
    test_size = len(all_texts) - train_size
    print(f"train size: {train_size}, test size: {test_size}")

    trained_texts = list(all_texts[:train_size])
    trained_labels = list(all_labels[:train_size])
    test_texts = list(all_texts[train_size:])
    test_labels = list(all_labels[train_size:])

    return trained_texts, trained_labels, test_texts, test_labels


def train():
    # trained_texts, trained_labels, test_texts, test_labels = get_real_train_test_data()
    trained_texts, trained_labels, test_texts, test_labels = get_fake_train_test_data()
    classifier = TextClassifier(num_labels=len(LABELS), force_retrain=need_force_retrain)
    train_dataset = classifier.prepare_data(trained_texts, trained_labels)
    test_dataset = classifier.prepare_data(test_texts, test_labels)

    start_time = time.time()
    classifier.train(train_dataset, test_dataset, epochs=trained_epochs)
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time:.2f} seconds. epochs: {trained_epochs}")

    classifier.save_model(model_path)
    test_accuracy, test_loss = classifier.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f} {test_loss:.4f}")


classifier = TextClassifier()
def predict(text):
    # 加载模型和分词器
    # 创建分类器实例
    # 使用加载的模型和分词器进行预测
    prediction = classifier.predict(text)
    # 获取标签映射
    label_mapping = {v: k for k, v in LABELS.items()}
    predicted_label = label_mapping[prediction[0]] if prediction[0] in label_mapping else "Unknown"
    if predicted_label == 'VERIFY_DISPOSAL_QA':
        if "/bin/bash" in text or "/bin/sh" in text or "脚本" in text:
            return "SCRIPT_QA", prediction
    # if predicted_label == 'COMMON_SEC_QA':
    if text.isdigit():
        return "VERIFY_DISPOSAL_QA", prediction
    log.info(f"Predicted class for '{text}': {predicted_label}, with probabilities: {prediction}")
    return predicted_label, prediction


def singe_test():
    # 预测
    test_text = "现代网络环境面临哪些安全威胁？我们应该如何保护组织的网络安全？"
    test_text = "是否有可能这次攻击是由内部人员发动的"
    test_text = '针对DDoS攻击(DDoS)-当前连接耗尽攻击(Concurrent Connections Flood)事件/告警，智慧体，你能帮我识别出DDoS攻击中使用的攻击工具或方法吗？'
    test_text = '你能帮我识别出DDoS攻击中使用的攻击工具或方法吗？'
    test_text = '我要采取哪些行动来应对XX安全问题？'
    predict(test_text)


def input_test():
    label_mapping = {v: k for k, v in LABELS.items()}
    while True:
        test_text = input("请输入问题（输入q退出）: ")
        if test_text.lower() == 'q':
            break
        predicted_label, prediction = predict(test_text)
        log.info(f"Predicted class for '{test_text}': {predicted_label} {prediction}")

def mannual_test():
    # 加载模型
    # 处理特定文件
    process_manual_file("./人工生成语料/解决措施.txt")
    process_manual_file("./L2_data/NL2FlinkSQL_qa.txt")

    # 处理其他数据集
    process_json_file("./L1_data/cissp_评测.jsonl", "cissp评测")
    # process_json_file("./L1_data/cissp_微调.jsonl", tokenizer, loaded_model, "cissp微调")
    process_json_file(None,"script qa", parser=parse_script_qa)
    process_json_file(None,"nl2sql qa", parser=parse_nl2sql_qa)


def process_manual_file(filename):
    if os.path.isfile(filename):
        print(f"Processing {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            process_lines(lines, filename)


def process_json_file(jsonl_file_path, dataset_type, parser=None):
    if parser:
        lines = parser()
    else:
        lines = extract_cissp(jsonl_file_path)
    process_lines(lines, dataset_type)


def process_lines(lines, label, topK=100):
    start_time = time.time()
    cnt = len(lines)
    for i, line in enumerate(lines[:topK]):
        line = line.strip()
        _, prediction = predict(line)
        print(f"Predicted class for {label} '{i}': {prediction}")
    end_time = time.time()
    print(f"Time taken for processing {label}: {end_time - start_time:.2f} seconds. lines count: {cnt}")

if __name__ == "__main__":
    print("*"*88)
    train()
    print("finished training model.")
    print("*"*88)
    # mannual_test()
    singe_test()
    # input_test()
