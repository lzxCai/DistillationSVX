# 文件名: verify_mismatch.py
import torch as th
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt

# 模拟MLPPredictionBlock的行为
def simulate_mlp_prediction_block(features, labels):
    """模拟MLPPredictionBlock的内部排序行为"""
    print("Original label:", labels)
    
    # MLPPredictionBlock内部的排序操作
    sorted_indices = th.argsort(labels)
    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]
    print("Sorted label:", sorted_labels)

    # 模拟预测 - 假设预测值接近排序后的标签
    predictions = sorted_labels * 0.95  # 简单模拟，预测值接近真实标签的95%
    print("Predictions (corresponding to sorted labels):", predictions)

    return predictions, sorted_labels

# 模拟当前代码的损失计算方式
def current_loss_calculation(predictions, original_labels):
    """模拟当前代码中错误的损失计算方式"""
    loss_fn = nn.L1Loss()
    loss = loss_fn(predictions, original_labels)
    print("Current loss calculation (current code):", loss.item())
    print("Differences for each value:", th.abs(predictions - original_labels))
    return loss

# 模拟正确的损失计算方式
def correct_loss_calculation(predictions, sorted_labels):
    """模拟修复后的正确损失计算方式"""
    loss_fn = nn.L1Loss()
    loss = loss_fn(predictions, sorted_labels)
    print("Correct loss calculation (should be used):", loss.item())
    print("Differences for each value:", th.abs(predictions - sorted_labels))
    return loss

# 生成一些测试数据
def generate_test_data(num_samples=3):
    """生成测试数据"""
    # 随机生成一些值表示特征和标签
    features = th.rand(num_samples, 5)  # 假设每个节点有5个特征
    labels = th.tensor([414.58, 132.45, 523.03])  # 使用您示例中的类似值
    return features, labels

# 主函数
def main():
    print("=== 验证标签错位问题 ===\n")
    
    # 生成测试数据
    features, labels = generate_test_data()
    
    # 模拟MLPPredictionBlock的行为
    predictions, sorted_labels = simulate_mlp_prediction_block(features, labels)
    
    # 计算当前错误的损失
    current_loss = current_loss_calculation(predictions, labels)
    
    # 计算正确的损失
    correct_loss = correct_loss_calculation(predictions, sorted_labels)
    
    # 对比两种损失的差异
    print("\n损失差异:", (current_loss - correct_loss).item())
    
    # 可视化对比
    plt.figure(figsize=(12, 6))
    
    # 错误匹配
    plt.subplot(1, 2, 1)
    plt.title("Current code: Incorrect matching")
    for i in range(len(labels)):
        plt.plot([i, i], [labels[i].item(), predictions[i].item()], 'r-')
        plt.text(i-0.1, labels[i].item(), f"{labels[i].item():.1f}")
        plt.text(i-0.1, predictions[i].item(), f"{predictions[i].item():.1f}")
    plt.scatter(range(len(labels)), labels.numpy(), label='Original Labels')
    plt.scatter(range(len(labels)), predictions.numpy(), label='Predictions')
    plt.legend()
    plt.grid(True)
    
    # 正确匹配
    plt.subplot(1, 2, 2)
    plt.title("After repair: Correct match")
    for i in range(len(sorted_labels)):
        plt.plot([i, i], [sorted_labels[i].item(), predictions[i].item()], 'g-')
        plt.text(i-0.1, sorted_labels[i].item(), f"{sorted_labels[i].item():.1f}")
        plt.text(i-0.1, predictions[i].item(), f"{predictions[i].item():.1f}")
    plt.scatter(range(len(sorted_labels)), sorted_labels.numpy(), label='Sorted Labels')
    plt.scatter(range(len(sorted_labels)), predictions.numpy(), label='Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('label_mismatch_visualization.png')
    plt.show()

if __name__ == "__main__":
    main()