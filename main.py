import sys

import ast_transition_similarity
import metrics_similarity
import test

sys.path.extend(['.', '..'])

# 两两比较 Threshold = 0.290791, tpf = 75.839% , fpr = 14.427%
# 使用KMeans聚类，横向为原所在的组，纵向为聚类后所在的组，值为计数
# [12, 481, 5, 0, 0]
# [4, 0, 489, 2, 0]
# [354, 68, 4, 70, 1]
# [81, 353, 22, 44, 0]
# [481, 15, 4, 0, 0]

if __name__ == "__main__":
    # 计算使用metrics的roc曲线
    print("**** Metrics ****")
    metrics_list = test.load_pairwise_compare_data(metrics_similarity.MetricsParser)
    print("Threshold = %f,TPR = %f, FPR = %f, AUC=%f" % test.similarity_roc(metrics_similarity.MetricsParser, metrics_list, True, True))
    # 计算使用transition矩阵的roc曲线
    print("**** Transition ****")
    metrics_list = test.load_pairwise_compare_data(ast_transition_similarity.AstTransitionParser, False)
    print("Threshold = %f,TPR = %f, FPR = %f, AUC=%f" % test.similarity_roc(ast_transition_similarity.AstTransitionParser, metrics_list, True, True))
