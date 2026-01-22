# NEU-Machine-Learning-Homework
李天牧的作业

本项目是机器学习课程的期末实践成果。针对医疗决策中“黑盒模型”不可解释的痛点，我们构建了一个基于 XGBoost 的糖尿病预测模型，并深入集成了 SHAP (SHapley Additive exPlanations) 解释框架。
本项目的设计灵感来源于 Lundberg & Lee (2017) 的经典论文 《A Unified Approach to Interpreting Model Predictions》。我们不仅关注预测的准确性，更通过博弈论归因理论实现了从数据处理到个体辅助诊断的完整可解释性闭环。

本项目生成了两类关键图像，用于验证模型的逻辑：
1. 全局摘要图 (Global Summary Plot)
展示特征对全量样本的影响分布。
解读： 血糖 (Glucose) 越高，SHAP 值越大（右移），风险越高，符合临床共识。
2. 个体归因瀑布图 (Local Waterfall Plot)
展示针对特定患者 (Patient #0) 的逻辑解构。
解读： 清晰展示了年轻（Age）带来的保护性偏置如何被肥胖（BMI）带来的风险推力所抵消。

本项目的理论基础源于以下论文：
Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NIPS 2017).
