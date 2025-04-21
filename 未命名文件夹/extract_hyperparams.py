import os
import re

# 定义输出文件目录
output_dir = "/Users/yipengli/Desktop/559/514_Final"

# 模型和文件名
models = [
    {"name": "Artificial Neural Network (Binary)", "file": "ann_binary_output.txt", "param_name": "hidden_layer_sizes"},
    {"name": "Artificial Neural Network (Multi-Class)", "file": "ann_multi_output.txt", "param_name": "hidden_layer_sizes"},
    {"name": "Random Forest (Binary)", "file": "rf_binary_output.txt", "param_name": "n_estimators"},
    {"name": "Random Forest (Multi-Class)", "file": "rf_multi_output.txt", "param_name": "n_estimators"},
    {"name": "Naïve Bayes (Binary)", "file": "nb_binary_output.txt", "param_name": "var_smoothing"},
    {"name": "Naïve Bayes (Multi-Class)", "file": "nb_multi_output.txt", "param_name": "var_smoothing"},
    {"name": "Support Vector Machine (Binary)", "file": "svm_binary_output.txt", "param_name": "C"},
    {"name": "Support Vector Machine (Multi-Class)", "file": "svm_multi_output.txt", "param_name": "C"},
]

# 正则表达式，匹配超参数和准确率
# 匹配：hidden_layer_sizes = (50, 50), Accuracy = 0.8909 或 var_smoothing = 1e-09, Accuracy = 0.8500
pattern = re.compile(r"(\w+\s*=\s*[^\n,]+?),\s*Accuracy\s*=\s*(\d+\.\d+)")

# 存储结果
results = []

for model in models:
    file_path = os.path.join(output_dir, model["file"])
    model_name = model["name"]
    param_name = model["param_name"]
    
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # 提取所有超参数和准确率
        param_scores = []
        for line in lines:
            match = pattern.search(line)
            if match:
                param_value = match.group(1).strip()
                accuracy = float(match.group(2))
                param_scores.append((param_value, accuracy))
        
        # 找到最佳超参数
        if param_scores:
            best_param, best_accuracy = max(param_scores, key=lambda x: x[1])
            result = {
                "model": model_name,
                "best_param": best_param,
                "best_accuracy": best_accuracy,
                "reason": f"{best_param} achieved the best 5-fold cross-validation accuracy of {best_accuracy:.4f}. Other values were tested but resulted in lower accuracy."
            }
            results.append(result)
        else:
            print(f"警告：未在 {model['file']} 中找到交叉验证结果，可能格式不匹配")
            print(f"文件内容：{[line.strip() for line in lines[:5]]}")
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"处理 {model['file']} 时出错：{e}")

# 输出 Q2.3 答案
print("\n=== Q2.3 Pick your best hyperparameter values ===")
print("1. k-NN:")
print("   - Best hyperparameter value: k = 3")
print("   - Reason: k = 3 achieved the best cross-validation accuracy of 86.17%. Increasing k beyond 3 did not further improve accuracy.")
print("2. Decision Tree:")
print("   - Best hyperparameter value: max_depth = 5")
print("   - Reason: max_depth = 5 achieved the best cross-validation accuracy of 88.81%. Increasing depth beyond 5 did not further improve accuracy, and smaller values underperformed.")
for i, result in enumerate(results, start=3):
    print(f"{i}. {result['model']}:")
    print(f"   - Best hyperparameter value: {result['best_param']}")
    print(f"   - Reason: {result['reason']}")