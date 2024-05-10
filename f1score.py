import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

max_student_id = 900
max_teacher_id = 1600



def gettrue_labels():
    with open('./data/1.txt', 'r') as file:
        # 读取文件内容
        lines = file.readlines()

    # 初始化空列表来存储结果
    result = []

    # 遍历每一行数据
    for line in lines:
        # 将每一行数据按制表符进行分割，并转换成整数类型
        data = tuple(map(int, line.strip().split('\t')))
        # 将每一行数据添加到结果列表中
        result.append(data)

    # 找出学生和教师ID的最大值


    # 初始化一个二维数组，根据最大学生和教师ID确定大小，初始值都为0
    ratings_array = np.zeros((max_student_id, max_teacher_id))

    # 创建一个字典来存储每个学生对每个教师的评分
    ratings_dict = {(student_id, teacher_id): rating for student_id, teacher_id, rating in result}

    # 填充二维数组
    for i in range(max_student_id):
        for j in range(max_teacher_id):
            rating = ratings_dict.get((i + 1, j + 1), 0)  # 如果评分不存在，则默认为0
            ratings_array[i, j] = rating

    return ratings_array

def getpredicted_probs():
    with open('predictions.txt', 'r') as file:
        lines = file.readlines()

    # 初始化一个空的二维数组
    array_2d = []

    # 将每行的一维数组转换成列表并添加到二维数组中
    for line in lines:
        # 去掉换行符并使用 eval 将字符串转换成列表
        array_1d = eval(line.strip())
        array_2d.append(array_1d)

    # 计算数组的行数和列数
    rows = len(array_2d)
    cols = len(array_2d[0])

    # 如果需要填充到 800 行 1600 列，可以进行相应的填充
    # 假设填充的值为 0
    desired_rows = 900
    desired_cols = 1600

    # 填充行
    while len(array_2d) < desired_rows:
        array_2d.append([0] * cols)

    # 填充列
    for i in range(rows):
        while len(array_2d[i]) < desired_cols:
            array_2d[i].append(0)

    return array_2d

def getoptimal_threshold(y_true,y_prob):
    auc_scores = []
    for class_idx in range(6):
        y_true_class = (y_true == class_idx).astype(int)

        y_prob_class = y_prob
        fpr, tpr, thresholds = roc_curve(y_true_class.flatten(), y_prob_class.flatten())
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    # 找到平均AUC最高的类别
    best_class_idx = np.argmax(auc_scores)

    # 绘制ROC曲线
    y_true_best_class = (y_true == best_class_idx).astype(int).flatten()
    y_prob_best_class = y_prob.flatten()
    fpr, tpr, thresholds = roc_curve(y_true_best_class, y_prob_best_class)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    # 找到最佳阈值
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    print("Optimal Threshold:", optimal_threshold)
    return optimal_threshold

def getscore(y_true,y_pre,optimal_threshold):
    a = optimal_threshold
    predicted_binary = (y_pre > a).astype(int)
    actual_binary = (y_true > a).astype(int)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(actual_binary.ravel(), predicted_binary.ravel())

    print("Confusion Matrix:")
    print(conf_matrix)

    precision = conf_matrix[0][0] / (conf_matrix[1][0] + conf_matrix[0][0])

    # 计算 Recall
    recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])

    # 计算 F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)


if __name__ == '__main__':


    # true_labels = np.array(gettrue_labels())  # 真实标签，假设有800个学生和1600个教师
    # predicted_probs = np.array(getpredicted_probs())  # 模型的预测概率，假设为随机概率

    true_labels = gettrue_labels()
    predicted_probs = np.array(getpredicted_probs())
    getscore(true_labels,predicted_probs,getoptimal_threshold(true_labels,predicted_probs))


