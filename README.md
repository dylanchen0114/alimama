Online Feature Documentation: [https://docs.google.com/spreadsheets/d/1sDYOzEW6zWBI-8w7ZVGktaklyh9J760tTMV3OE5hY3A/](https://docs.google.com/spreadsheets/d/1sDYOzEW6zWBI-8w7ZVGktaklyh9J760tTMV3OE5hY3A/)

Before after

Timestamp or sample weight

Probability



关于样本前后分布有较大差异，导致train、valid、test的score趋势不一致的解决方法：

1. 用前5天 + 后2天来构造特征，用后2天来作CV（CV的方法？）来保证CV score与线上test的一致性；

   **还需确认预测线上test时，特征的做法**

   缺点：前5天的大量数据样本没有使用，但这5天确实受节日影响分布非常不同

   ​

2. 用前5天 + 后2天来构造特征，不去除前5天样本，告诉模型前5天与后2天的差异

   加样本权重、加binary特征？

   ​

3. 用前5天、后2天分别构造模型，对test做出blending的预测









