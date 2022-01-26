### 1.Deep Multi-Fidelity Gaussian Process Prediction

##### 1.1Brief Background

In daily life, we often encounter situations where it is difficult to obtain high-fidelity data during the training phase of some data predictions, such as changes in the earth’s atmosphere and heat conduction data. But for these experiments, we can use artificially simulated physical models to obtain a large amount of data. But these artificially simulated data often have a gap with the real situation, that is, low-fidelity data. How to train a model with a large number of low-fidelity data obtained by simulation combined with very few high-precision high-fidelity data so that the model has a high correct rate of prediction for high-fidelity data has become a problem worthy of research.

##### 1.2 Method of Experiment

###### 1.2.1Deep Multi-fidelity Gaussian Processes

We mainly use this structure to conduct joint training on high and low fidelity data in order to obtain information between high and low fidelity information.
$$
x \left[\begin{array}{c}f_{1}(h) \\f_{2}(h)\end{array}\right] \sim \mathcal{G} \mathcal{P}\left(\left[\begin{array}{l}0 \\0\end{array}\right],\left[\begin{array}{cc}k_{1}\left(h, h^{\prime}\right) & \rho k_{1}\left(h, h^{\prime}\right) \\\rho k_{1}\left(h, h^{\prime}\right) & \rho^{2} k_{1}\left(h, h^{\prime}\right)+k_{2}\left(h, h^{\prime}\right)\end{array}\right]\right)
$$
Where
$$
x  \longmapsto h:=h(x) \longmapsto\left[\begin{array}{l}f_{1}(h(x)) \\f_{2}(h(x))\end{array}\right]\\\\f_{1}(h(x)):establish\ a\ low\ fidelity\ system \\f_{2}(h(x)):establish\ a\ high\ fidelity\ system \\\mathcal{G} \mathcal{P}：Gaussian\ Processes\\h(x)：Arbitrarily\ determined\ parameter\ data\ conversion
$$
But in the follow-up research, I kept trying and making mistakes, transforming h(x) into a multilayer neural network.
$$
h(x):=\left(h^{L} \circ \ldots \circ h^{1}\right)(x)
$$
where
$$
h^{\ell}(z)=\sigma^{\ell}\left(w^{\ell} z+b^{\ell}\right)
$$

###### **1.2.2Forecasting Process**

$$
\left[\begin{array}{c}f_{1}(h) \\f_{2}(h)\end{array}\right] \sim \mathcal{G} \mathcal{P}\left(\left[\begin{array}{l}0 \\0\end{array}\right],\left[\begin{array}{cc}k_{11}\left(h, h^{\prime}\right) & k_{12}\left(h, h^{\prime}\right) \\k_{21}\left(h, h^{\prime}\right) & k_{22}\left(h, h^{\prime}\right)\end{array}\right]\right)\\where：k_{11} \equiv k_{1}\\k_{12} \equiv k_{21} \equiv \rho k_{1}\\k_{22} \equiv \rho^{2} k_{1}+k_{2}
$$

This can be used to obtain the prediction distribution of the replacement model of the high-fidelity system at the new test point x∗:
$$
p\left(f_{2}\left(h\left(x_{*}\right)\right) \mid x_{*}, \boldsymbol{x}_{\mathbf{1}}, \boldsymbol{f}_{1}, \boldsymbol{x}_{\mathbf{2}}, \boldsymbol{f}_{2}\right)
$$
Introduce a new test point x∗ and find the joint density:
$$
\left[\begin{array}{c}f_{2}\left(h\left(x_{*}\right)\right) \\\boldsymbol{f}_{1} \\\boldsymbol{f}_{2}\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}0 \\\mathbf{0} \\\mathbf{0}\end{array}\right],\left[\begin{array}{ccc}k_{22}\left(h_{*}, h_{*}\right) & k_{21}\left(h_{*}, \boldsymbol{h}_{1}\right) & k_{22}\left(h_{*}, \boldsymbol{h}_{2}\right) \\k_{12}\left(\boldsymbol{h}_{1}, h_{*}\right) & k_{11}\left(\boldsymbol{h}_{1}, \boldsymbol{h}_{1}\right) & k_{12}\left(\boldsymbol{h}_{1}, \boldsymbol{h}_{2}\right) \\k_{22}\left(\boldsymbol{h}_{2}, h_{*}\right) & k_{21}\left(\boldsymbol{h}_{2}, \boldsymbol{h}_{1}\right) & k_{22}\left(\boldsymbol{h}_{2}, \boldsymbol{h}_{2}\right)\end{array}\right]\right)\\where\  h_{*}=h(x_{*}),\boldsymbol{h}_{1}=h(\boldsymbol{x}_{1}),\boldsymbol{h}_{2}=h(\boldsymbol{x}_{2})
$$
Introduce a new test point x∗ and find the joint density:
$$
p\left(f_{2}\left(h\left(x_{*}\right)\right) \mid x_{*}, \boldsymbol{x}_{1}, \boldsymbol{f}_{1}, \boldsymbol{x}_{2}, \boldsymbol{f}_{2}\right)=\mathcal{N}\left(K_{*} K^{-1} \boldsymbol{f}, k_{22}\left(h_{*}, h_{*}\right)-K_{*} K^{-1} K_{*}^{T}\right)
$$
where:
$$
\begin{array}{l}\boldsymbol{f}:=\left[\begin{array}{ll}\boldsymbol{f}_{1} \\\boldsymbol{f}_{2}\end{array}\right] \\K_{*}:=\left[\begin{array}{ll}k_{21}\left(h_{*}, \boldsymbol{h}_{1}\right) & k_{22}\left(h_{*}, \boldsymbol{h}_{2}\right)\end{array}\right] \\K:=\left[\begin{array}{ll}k_{11}\left(\boldsymbol{h}_{1}, \boldsymbol{h}_{1}\right) & k_{12}\left(\boldsymbol{h}_{1}, \boldsymbol{h}_{2}\right) \\k_{21}\left(\boldsymbol{h}_{2}, \boldsymbol{h}_{1}\right) & k_{22}\left(\boldsymbol{h}_{2}, \boldsymbol{h}_{2}\right)\end{array}\right]\end{array}
$$

###### 1.2.3 Project Framework

![图片 1](/Users/sunluzhe/Desktop/图片 1.png)

##### 1.3Training Effect Display

In my four versions of the improved code, evaluated using the R2 score, the accuracy of the predictions was eventually improved from a negative number to 0.62. All predictions fell within acceptable confidence intervals.

**version1**

<img src="/Users/sunluzhe/Library/Application Support/typora-user-images/image-20210922173138439.png" alt="image-20210922173138439" style="zoom:33%;" />

**version4**

<img src="/Users/sunluzhe/Library/Application Support/typora-user-images/image-20210922173410665.png" alt="image-20210922173410665" style="zoom:33%;" />

