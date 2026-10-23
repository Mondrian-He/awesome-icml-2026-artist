# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 113 of 197

## 目录 (Table of Contents)

1. [Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](#paper-3697)
2. [PDAgent: An LLM-Driven Autonomous Agent Framework Towards *In Silico* Protein Design via Directed Mutation](#paper-3698)
3. [A Diffusive Classification Loss for Learning Energy-based Generative Models](#paper-3699)
4. [Primal-Spectral Generative Modeling: Fast Analytical Generation via Pseudoinverse Lévy Inversion](#paper-3700)
5. [DenseSteer: Steering Small Language Models towards Dense Math Reasoning](#paper-3701)
6. [LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](#paper-3702)
7. [ReaForest: Fostering Generative Video Reasoning for Spatial Planning](#paper-3703)
8. [Must All Negatives Be Pushed Away Equally? Uncertainty-Aware Cross-View Geo-Localization via Normal Inverse Gamma Distribution](#paper-3704)
9. [Calibrating Conservatism for Scalable Oversight](#paper-3705)
10. [The Oversight Game: Learning to Cooperatively Balance an AI Agent’s Safety and Autonomy](#paper-3706)
11. [SoftBinary Coding: A New Information-Theoretic Paradigm for Neural Compression via Fast Channel Simulation](#paper-3707)
12. [The Latent Color Subspace: Emergent Order in High-Dimensional Chaos](#paper-3708)
13. [Tail Annealing for Heavy-Tailed Flow Matching](#paper-3709)
14. [Allocating Variance to Maximize Expectation](#paper-3710)
15. [Aggregate Models, Not Explanations: Improving Feature Importance Estimation](#paper-3711)
16. [Revisiting the Volume Hypothesis](#paper-3712)
17. [DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](#paper-3713)
18. [Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality](#paper-3714)
19. [Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](#paper-3715)
20. [Benchmarking and Improving Fine-Grained Text-to-Image Alignment via Paired Reinforcement Learning](#paper-3716)
21. [Adaptive Multi-Round Allocation with Stochastic Arrivals](#paper-3717)
22. [Difference-Aware Decision Learning for Multimodal Image Fusion](#paper-3718)
23. [Trajectory-Level Speculative Decoding for Diffusion Language Models](#paper-3719)
24. [ECCO: Evidence-Driven Causal Reasoning for Compiler Optimization](#paper-3720)
25. [Problem Distributions as Tasks: Repurposing Meta Learning for Generative Combinatorial Optimization towards Multi-task Pretraining and Adaptation](#paper-3721)
26. [Beyond Instance-Level Self-Supervision in 3D Multi-Modal Medical Imaging](#paper-3722)
27. [Knowing Bias, Doing Better: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement](#paper-3723)
28. [ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management](#paper-3724)
29. [Detecting and Filtering Unsafe Training Data via Data Attribution with Denoised Representation](#paper-3725)
30. [Temporal-aware Flow Matching for Video Generation with Temporally Coherent Motion](#paper-3726)
31. [Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](#paper-3727)
32. [Optimal Transport for LLM Reward Modeling from Noisy Feedback](#paper-3728)
33. [AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning](#paper-3729)

---


<a id="paper-3697"></a>

## Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation

### Images

![452329f16c2f569acb0e3cf7f1db679b4bf6b95bb7b173136b87071347acab81.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/452329f16c2f569acb0e3cf7f1db679b4bf6b95bb7b173136b87071347acab81.jpg)

![4e1629cdcc666f7794fbd0e4be79b69a8696ff95ebf7b04d9918d1765a0766c2.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/4e1629cdcc666f7794fbd0e4be79b69a8696ff95ebf7b04d9918d1765a0766c2.jpg)

![65d7232c6e04760a8e22a07f0847d1c04ede4b039c837a6d41c49b3e712725bf.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/65d7232c6e04760a8e22a07f0847d1c04ede4b039c837a6d41c49b3e712725bf.jpg)

![762990f791d499c4baa67bfca9d53007ed3c855d2b6dad6081ffacb6b0639693.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/762990f791d499c4baa67bfca9d53007ed3c855d2b6dad6081ffacb6b0639693.jpg)

![9f903ab825c78500e4e6eec27b64896f36e54d5ae3e2fa8169e454a6913f6cb9.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/9f903ab825c78500e4e6eec27b64896f36e54d5ae3e2fa8169e454a6913f6cb9.jpg)

![b5e150a3738320c0365bf3479e88cedc893e544bf1ee42a9e1ce226101672454.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/b5e150a3738320c0365bf3479e88cedc893e544bf1ee42a9e1ce226101672454.jpg)

![d2e0c3f796e9d1ffee66a97fb98c4a3aaa4d7b61b77d3a9efd2be27d6461845b.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/d2e0c3f796e9d1ffee66a97fb98c4a3aaa4d7b61b77d3a9efd2be27d6461845b.jpg)

![fd5c7d7dd3d41e74b7ee69abe5c02b7792a554f7bfd9f7a43c9940a7ef9b997d.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/images/fd5c7d7dd3d41e74b7ee69abe5c02b7792a554f7bfd9f7a43c9940a7ef9b997d.jpg)

### Tables

![05652c7ba28a55ee8345f2cf6dd99c82628945a87c1df794e7c25a977a4ad58f.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/05652c7ba28a55ee8345f2cf6dd99c82628945a87c1df794e7c25a977a4ad58f.jpg)

![1cda86537d5c57ffb09b6593211967e0ecedd9c3fd742db1383b77d9764d16ed.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/1cda86537d5c57ffb09b6593211967e0ecedd9c3fd742db1383b77d9764d16ed.jpg)

![1f091960d4fe0ee5d4ba1b68d9fd3b48624e861e78c0e8c83cee2973b1608213.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/1f091960d4fe0ee5d4ba1b68d9fd3b48624e861e78c0e8c83cee2973b1608213.jpg)

![264236d360cad9500361dd16acc8b7ef508f7387deaef08a43bbfb56c016f1ac.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/264236d360cad9500361dd16acc8b7ef508f7387deaef08a43bbfb56c016f1ac.jpg)

![311758016880ae9ae43bec43e0d584ef6da99c57c26d5beb6b8ed0544f3c9c08.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/311758016880ae9ae43bec43e0d584ef6da99c57c26d5beb6b8ed0544f3c9c08.jpg)

![3266d43baf28a461d5e718e046dc023db5d1936d67cfddf87cc8b748cd263a26.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/3266d43baf28a461d5e718e046dc023db5d1936d67cfddf87cc8b748cd263a26.jpg)

![4962db2474e0704e1a72b828a10759a650a53f9a9819cbe5d452829fc6d97a0d.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/4962db2474e0704e1a72b828a10759a650a53f9a9819cbe5d452829fc6d97a0d.jpg)

![644c410bce6c6591174fdf33d5592b8e910d6591bdaa4d4c43f7775edd1f9b78.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/644c410bce6c6591174fdf33d5592b8e910d6591bdaa4d4c43f7775edd1f9b78.jpg)

![64575f7d56d2acdfd66e56813c60496c61fea9e2889dc38707dd16ef4499b760.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/64575f7d56d2acdfd66e56813c60496c61fea9e2889dc38707dd16ef4499b760.jpg)

![65dbcbef265441ed004cecd971d53e13b24915ea6660e39d51848a6eddd7617f.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/65dbcbef265441ed004cecd971d53e13b24915ea6660e39d51848a6eddd7617f.jpg)

![7147d2b43f4042c5e1e7c1bd31022fb67e467cc44d81ffeeb1301fc540936cb6.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/7147d2b43f4042c5e1e7c1bd31022fb67e467cc44d81ffeeb1301fc540936cb6.jpg)

![79c0f9d588da38421a7440a50b326e2de6a0fee5f6c00b244fb20b8c2ca74b47.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/79c0f9d588da38421a7440a50b326e2de6a0fee5f6c00b244fb20b8c2ca74b47.jpg)

![7dce2bfbb9f8143dbc61da6d2fe3485a60d35f179561a9087fa1e33075054ddb.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/7dce2bfbb9f8143dbc61da6d2fe3485a60d35f179561a9087fa1e33075054ddb.jpg)

![8cc59c9e0b33f3a82a634dbb1512ffa9e0638a8a7bfeb3af454426bc4de31471.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/8cc59c9e0b33f3a82a634dbb1512ffa9e0638a8a7bfeb3af454426bc4de31471.jpg)

![91abba337fc5db33d3f06ef83645de51c925927cec4baf9db7d45aa43873ae3a.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/91abba337fc5db33d3f06ef83645de51c925927cec4baf9db7d45aa43873ae3a.jpg)

![94230b7e4470bbe9bcf73de32ddb16cf438bb57ff17f9b8fff5a745a5fd5e2a3.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/94230b7e4470bbe9bcf73de32ddb16cf438bb57ff17f9b8fff5a745a5fd5e2a3.jpg)

![9589d2281a0d5e3d2f4864d2a3b8949c423b414d2771e93d23f6bf5e009e019d.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/9589d2281a0d5e3d2f4864d2a3b8949c423b414d2771e93d23f6bf5e009e019d.jpg)

![9680fac7de8a47372fc49b7d25e72595fb4dd69b8fea4ac03061d5d2fcc9c4cc.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/9680fac7de8a47372fc49b7d25e72595fb4dd69b8fea4ac03061d5d2fcc9c4cc.jpg)

![9c34f8aa2334d9131e0309f5543d447160b696347ebc58fa97ce373d5bfce82f.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/9c34f8aa2334d9131e0309f5543d447160b696347ebc58fa97ce373d5bfce82f.jpg)

![9c51eeca7e0374c7cfda3336341769e9fa2061249569771d7e7fda485643ac77.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/9c51eeca7e0374c7cfda3336341769e9fa2061249569771d7e7fda485643ac77.jpg)

![a43d4e216b8438257b9731785fd0a40957e2363515539b60b6ca0c62e9b61ff0.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/a43d4e216b8438257b9731785fd0a40957e2363515539b60b6ca0c62e9b61ff0.jpg)

![a9450e312b5abf6d2e8bf0d050467bd4fdcf8adcded3aff186524125b6587fb5.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/a9450e312b5abf6d2e8bf0d050467bd4fdcf8adcded3aff186524125b6587fb5.jpg)

![b60b6ab2ef8601c558ae12d015a8ad0b72a3348233a9e5f5b94bf0d3a8bab5b3.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/b60b6ab2ef8601c558ae12d015a8ad0b72a3348233a9e5f5b94bf0d3a8bab5b3.jpg)

![bc906e7e126f1b65ca23413fcd2f1489baeccf50064dbb4f4c7443541f5586e9.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/bc906e7e126f1b65ca23413fcd2f1489baeccf50064dbb4f4c7443541f5586e9.jpg)

![bf7e0715ad58c3006dee5a737aba3bd3262499e8888eaddb65af5667604ec47b.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/bf7e0715ad58c3006dee5a737aba3bd3262499e8888eaddb65af5667604ec47b.jpg)

![c988174b2e39cb129139d129da7d4991b96c94209dfc9e9fc7f728eea6f4d497.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/c988174b2e39cb129139d129da7d4991b96c94209dfc9e9fc7f728eea6f4d497.jpg)

![d677d3e6051bcb179c890e39e27ec24efb0ca30b1629c0f1c489f5240fd4f1b0.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/d677d3e6051bcb179c890e39e27ec24efb0ca30b1629c0f1c489f5240fd4f1b0.jpg)

![dc8f77919a255fe86ae9a77bab6302a96e93f42baf3ec54aa14dd22296575541.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/dc8f77919a255fe86ae9a77bab6302a96e93f42baf3ec54aa14dd22296575541.jpg)

![de3437ce03c0ecabed773eda5dbfa6020e709625ac1affbee2d836ab041f9be5.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/de3437ce03c0ecabed773eda5dbfa6020e709625ac1affbee2d836ab041f9be5.jpg)

![fb674f3b3f2dc38264013ec314ff2c1afa5ae6665fd994e0cddc6b005841c930.jpg](../icml_results/3736_MoRGen_%20Mixture-of-Resolutions%20Generative%20Forecasting%20for%20Irregularly%20Sampled%20Medical%20Time-Series%20Da/tables/fb674f3b3f2dc38264013ec314ff2c1afa5ae6665fd994e0cddc6b005841c930.jpg)

## Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation


### Images

![0ae29e73814df58df0b9bd9f8ad0d337fd601ac15d4b15cbdbbc4102d81a2039.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/0ae29e73814df58df0b9bd9f8ad0d337fd601ac15d4b15cbdbbc4102d81a2039.jpg)

![49b9db63ff0d9e4543adb89e7b69c8a82ac0e517a454539d3bd312d5f5b27bed.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/49b9db63ff0d9e4543adb89e7b69c8a82ac0e517a454539d3bd312d5f5b27bed.jpg)

![7eefb1bf29711f9d16300f4af4e00d40394c48d942fe7bfcd0718d4e3c979ccd.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/7eefb1bf29711f9d16300f4af4e00d40394c48d942fe7bfcd0718d4e3c979ccd.jpg)

![ce680281d9e61d3aff87cf0b3e1dc4ae5f8401d791d46696d45ef342fdce0830.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/ce680281d9e61d3aff87cf0b3e1dc4ae5f8401d791d46696d45ef342fdce0830.jpg)

![d8d6f66547d9754d82e2ffa69da537a33a0126905ca1f0dc7770e68b208f6ec8.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/d8d6f66547d9754d82e2ffa69da537a33a0126905ca1f0dc7770e68b208f6ec8.jpg)

![db286d5d5f553f51659b61ab91a6976bcdb38cce645ce54cb84b607edbdcd15e.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/db286d5d5f553f51659b61ab91a6976bcdb38cce645ce54cb84b607edbdcd15e.jpg)

![fd919ece295dc0df690762899b06d31da868aaadd20a9dd27dac0e4796d330c9.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/images/fd919ece295dc0df690762899b06d31da868aaadd20a9dd27dac0e4796d330c9.jpg)

### Tables

![03074a2e9b793635d7d8b7aed01cff04663150c60cf0c77fba64fc6380ae6530.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/03074a2e9b793635d7d8b7aed01cff04663150c60cf0c77fba64fc6380ae6530.jpg)

![0b5d5a73d472b1dbb2099e64815a7ca9c0bbdf40cd1d2ebf8bbd31a439baa3e7.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/0b5d5a73d472b1dbb2099e64815a7ca9c0bbdf40cd1d2ebf8bbd31a439baa3e7.jpg)

![152713f8152610ae0f7ad0caee2f7961775936fd5d237d0b33ba694a2295e6e2.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/152713f8152610ae0f7ad0caee2f7961775936fd5d237d0b33ba694a2295e6e2.jpg)

![3d82c1a6e20d0a8a923f421de2e4948278a435567917643dd5de0e4023532fef.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/3d82c1a6e20d0a8a923f421de2e4948278a435567917643dd5de0e4023532fef.jpg)

![5f28266b12acac4476ac50e1cb15e738dcb9943f7635b9f1a8c27ad392595b44.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/5f28266b12acac4476ac50e1cb15e738dcb9943f7635b9f1a8c27ad392595b44.jpg)

![69c7198a9633178eb8cb536fecf4f58c654cf1262cd07240b0aa654c72734802.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/69c7198a9633178eb8cb536fecf4f58c654cf1262cd07240b0aa654c72734802.jpg)

![81fcbc41d23054f5e5571a4947d15ca56dc4d7f8ef2096c62ebd1e06dba4d418.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/81fcbc41d23054f5e5571a4947d15ca56dc4d7f8ef2096c62ebd1e06dba4d418.jpg)

![9db40e769cac5ecb664fbbd6812f3917943db901623dc5ba3a8729513ccd1dfa.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/9db40e769cac5ecb664fbbd6812f3917943db901623dc5ba3a8729513ccd1dfa.jpg)

![ab75718349d4b8d975f78ec4a61d42dcbbbeeab628f168c6ce1abe6ff1ab078e.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/ab75718349d4b8d975f78ec4a61d42dcbbbeeab628f168c6ce1abe6ff1ab078e.jpg)

![acee1d9011d941c042dc161616639715f5b0e15d08f65ba07a833a4fb0aa6720.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/acee1d9011d941c042dc161616639715f5b0e15d08f65ba07a833a4fb0aa6720.jpg)

![bae0a50319fd6f5705af229cbe9c83c73849ce8510b3eb4efb50d1fa3f355695.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/bae0a50319fd6f5705af229cbe9c83c73849ce8510b3eb4efb50d1fa3f355695.jpg)

![c9e8c034ff6cd5c98680fc4b4e5d7f97f91cce750e5c34a14f50d39e95fa3310.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/c9e8c034ff6cd5c98680fc4b4e5d7f97f91cce750e5c34a14f50d39e95fa3310.jpg)

![e668573aecf43d285497768cf90dea4abfd57d154cec54652ebb03739733d192.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/e668573aecf43d285497768cf90dea4abfd57d154cec54652ebb03739733d192.jpg)

![e9403f00f37d7fab516dd58009f65a2ff7ec8e538a4a4ed98b796db5071f9648.jpg](../icml_results/3737_Learning%20to%20Extrapolate%20to%20New%20Tasks_%20A%20Relational%20Approach%20to%20Task%20Extrapolation/tables/e9403f00f37d7fab516dd58009f65a2ff7ec8e538a4a4ed98b796db5071f9648.jpg)

<a id="paper-3698"></a>

## PDAgent: An LLM-Driven Autonomous Agent Framework Towards *In Silico* Protein Design via Directed Mutation


### Images

![13f51b1b303b9f2d18840dd2e2c9b9c65b37353bb788881bd62fbf43173679ee.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/images/13f51b1b303b9f2d18840dd2e2c9b9c65b37353bb788881bd62fbf43173679ee.jpg)

![9071c6ed61f5530948e61892db5f574de7227e0615e6e3ae1b640e3ad0155ea2.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/images/9071c6ed61f5530948e61892db5f574de7227e0615e6e3ae1b640e3ad0155ea2.jpg)

![973b374478c5d0f17e0af48a34d442ceec720e4bf19397a761603dc4f1a9f125.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/images/973b374478c5d0f17e0af48a34d442ceec720e4bf19397a761603dc4f1a9f125.jpg)

![e8e6d7f8a6aaf0823e9134a05071a2d7a571ad4e1d9465b8a64c1392bcae17f6.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/images/e8e6d7f8a6aaf0823e9134a05071a2d7a571ad4e1d9465b8a64c1392bcae17f6.jpg)

### Tables

![0c5171fa58b60853fc45f66b20dcb6b54e7c7bbb80bd6a1a6cb7090adde9a729.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/0c5171fa58b60853fc45f66b20dcb6b54e7c7bbb80bd6a1a6cb7090adde9a729.jpg)

![159ae42ee6b0dc9ca522e32602973d94337ad485db8ac088f2c2e53e84cb435a.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/159ae42ee6b0dc9ca522e32602973d94337ad485db8ac088f2c2e53e84cb435a.jpg)

![311185bc390e0d510b3220f0c501b9ac49f8f89f3eb7d49eed80336b98398c78.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/311185bc390e0d510b3220f0c501b9ac49f8f89f3eb7d49eed80336b98398c78.jpg)

![421333d9b6ed7ebc00989e336697cd89f2dac2a25d41b277882a9062badb3330.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/421333d9b6ed7ebc00989e336697cd89f2dac2a25d41b277882a9062badb3330.jpg)

![54cba639e87a23968eca670485d58f788978c86a96f2b9b596dc2c292b822a6b.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/54cba639e87a23968eca670485d58f788978c86a96f2b9b596dc2c292b822a6b.jpg)

![5c32046842c1c9f82b4ed960944a194d72372170bae412e275032c284207b326.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/5c32046842c1c9f82b4ed960944a194d72372170bae412e275032c284207b326.jpg)

![6280fcbf305cec761feca6fd304e596dcee8afc257a4642bce471c4cb002cf0a.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/6280fcbf305cec761feca6fd304e596dcee8afc257a4642bce471c4cb002cf0a.jpg)

![65b62c9ce13a0e0dafedd8785184adce2c8b6df61825aa346a49f52e52920a63.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/65b62c9ce13a0e0dafedd8785184adce2c8b6df61825aa346a49f52e52920a63.jpg)

![6f6204fdf06970cba4c3f0e4fc89c53f0f88000bdb9c5cbad3a2cad9ca84e16e.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/6f6204fdf06970cba4c3f0e4fc89c53f0f88000bdb9c5cbad3a2cad9ca84e16e.jpg)

![77765ed2b1b1267c10db49aab8bb0e13ccd70c0be7492e3989516d042ecef14a.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/77765ed2b1b1267c10db49aab8bb0e13ccd70c0be7492e3989516d042ecef14a.jpg)

![9749ec480f70a395e45f21f63a081a31d741433e338d488dc88a92bdbdd03f48.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/9749ec480f70a395e45f21f63a081a31d741433e338d488dc88a92bdbdd03f48.jpg)

![a0395e4ed285c33d6125c4f9f2ffa049bea115a57ff8bafe15ae90dcaf4cc9b2.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/a0395e4ed285c33d6125c4f9f2ffa049bea115a57ff8bafe15ae90dcaf4cc9b2.jpg)

![b21328a5cbc09a46acfd1a4be05e09c3c985944d27d51118bc020d9887a2adec.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/b21328a5cbc09a46acfd1a4be05e09c3c985944d27d51118bc020d9887a2adec.jpg)

![b38764a227c02db585f3ae71b4417124f5e0b9cf63b1a76ac8b6acc51d906aaa.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/b38764a227c02db585f3ae71b4417124f5e0b9cf63b1a76ac8b6acc51d906aaa.jpg)

![b38d4e4fcde4e4e11440a9ce1c2638dc9e058b1629ffd265d2745f2f7fef2be7.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/b38d4e4fcde4e4e11440a9ce1c2638dc9e058b1629ffd265d2745f2f7fef2be7.jpg)

![c053686d247ec95f529c02948417b3a663c7bdaa1ae6dfddcc39585fc2f296bc.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/c053686d247ec95f529c02948417b3a663c7bdaa1ae6dfddcc39585fc2f296bc.jpg)

![e16f1a5ea45e4cd2f81c5496bfd541550f8b1efac98fd00cd21b872358d1080c.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/e16f1a5ea45e4cd2f81c5496bfd541550f8b1efac98fd00cd21b872358d1080c.jpg)

![e8cf1a995a4c90dc3de4319df319393e9097a5fa75e0eec7ec205b4f08fc6df3.jpg](../icml_results/3738_PDAgent_%20An%20LLM-Driven%20Autonomous%20Agent%20Framework%20Towards%20_In%20Silico_%20Protein%20Design%20via%20Directed%20Mu/tables/e8cf1a995a4c90dc3de4319df319393e9097a5fa75e0eec7ec205b4f08fc6df3.jpg)

<a id="paper-3699"></a>

## A Diffusive Classification Loss for Learning Energy-based Generative Models


### Images

![28d987e6d1084bb6bf399a1f440ec0186207959abee7c8d8d351ff857b47d7b5.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/28d987e6d1084bb6bf399a1f440ec0186207959abee7c8d8d351ff857b47d7b5.jpg)

![309813ac910441db7b0e899344c0e55d69f4ab04eceee22cde8f8febd94c6f4a.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/309813ac910441db7b0e899344c0e55d69f4ab04eceee22cde8f8febd94c6f4a.jpg)

![4295bb91b43bd0a606a8d29d053d9bb6b2aade5c80201bc72bf9c68aa6f88986.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/4295bb91b43bd0a606a8d29d053d9bb6b2aade5c80201bc72bf9c68aa6f88986.jpg)

![4c1d45feb3fa703c3def7d13a23e9fcdf150bf0f77997c6ec876ef9504577891.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/4c1d45feb3fa703c3def7d13a23e9fcdf150bf0f77997c6ec876ef9504577891.jpg)

![63611668dd04e24dac87526c68a397df55ce36a05d138138c3789067c7185aa3.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/63611668dd04e24dac87526c68a397df55ce36a05d138138c3789067c7185aa3.jpg)

![70b2421be924c4c34ed683443ede012e1ebea571902e8038cda3af9c66825d49.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/70b2421be924c4c34ed683443ede012e1ebea571902e8038cda3af9c66825d49.jpg)

![9e4551789a2078331ce8fef0b337df9b4fa34a34c1b94075be31b158522c74be.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/9e4551789a2078331ce8fef0b337df9b4fa34a34c1b94075be31b158522c74be.jpg)

![bd7b98bca1ca39a026dc6a916696c34defe688de83395fb530e81635a005cf7c.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/bd7b98bca1ca39a026dc6a916696c34defe688de83395fb530e81635a005cf7c.jpg)

![da962a557d8fb3c8e05e6df3e24be8ffca4f46a34ead82f0c30fb44273840605.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/images/da962a557d8fb3c8e05e6df3e24be8ffca4f46a34ead82f0c30fb44273840605.jpg)

### Tables

![03fd3efbea03e648f4014f12bcbac212f1922c21ad7526ed2162a9cdee200288.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/03fd3efbea03e648f4014f12bcbac212f1922c21ad7526ed2162a9cdee200288.jpg)

![05a6efa6f0c29e18508727fc7f42b554754292fef92ad11e83f456ac01de4638.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/05a6efa6f0c29e18508727fc7f42b554754292fef92ad11e83f456ac01de4638.jpg)

![6a11214680622d539f6f2bb09a2426e9de7feae95a65bf393b85839fbbc159d2.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/6a11214680622d539f6f2bb09a2426e9de7feae95a65bf393b85839fbbc159d2.jpg)

![9b3426e2e1a496af425ff421ce224c4b2463bf58423ef52fb0efae387507f39d.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/9b3426e2e1a496af425ff421ce224c4b2463bf58423ef52fb0efae387507f39d.jpg)

![9b80fddcfb2f27fa5a8e3aeaff07ef2ecb69f36413bd3529e731bc64293ecb15.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/9b80fddcfb2f27fa5a8e3aeaff07ef2ecb69f36413bd3529e731bc64293ecb15.jpg)

![a3587529f10b85df914868736344963acca6557fda4ec431f6d0df0d89c5bb77.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/a3587529f10b85df914868736344963acca6557fda4ec431f6d0df0d89c5bb77.jpg)

![ac56241b06f6817c71686952196e6dbf591823ca3b0dcbc807e1709fb0c0162e.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/ac56241b06f6817c71686952196e6dbf591823ca3b0dcbc807e1709fb0c0162e.jpg)

![b1c60fba66aec8478b6dfbd918f65be87b395f9d74daca65372853d6fc3e1c33.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/b1c60fba66aec8478b6dfbd918f65be87b395f9d74daca65372853d6fc3e1c33.jpg)

![dc4ef9a26199ed42dbf8c14cd3e53d8ab5047ae7b4d50752fe78f1c33680ca7e.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/dc4ef9a26199ed42dbf8c14cd3e53d8ab5047ae7b4d50752fe78f1c33680ca7e.jpg)

![dfcf0ac18fec64f62807714a4e2a94a905b73d962640ba534a7e148783edcb65.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/dfcf0ac18fec64f62807714a4e2a94a905b73d962640ba534a7e148783edcb65.jpg)

![f4731ee39a7dbe7deaa477859af4c67f4fdefbc4dc0c90b09808a024505ab166.jpg](../icml_results/3739_A%20Diffusive%20Classification%20Loss%20for%20Learning%20Energy-based%20Generative%20Models/tables/f4731ee39a7dbe7deaa477859af4c67f4fdefbc4dc0c90b09808a024505ab166.jpg)

<a id="paper-3700"></a>

## Primal-Spectral Generative Modeling: Fast Analytical Generation via Pseudoinverse Lévy Inversion


### Images

![10b7ec827f984814ca468685ca1f1b7fdf1c731330258c5ed4307770d5148bc6.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/10b7ec827f984814ca468685ca1f1b7fdf1c731330258c5ed4307770d5148bc6.jpg)

![19a3416856b07e9f288f0f99208b17619fc98f1723979b930caafa44e83c27e0.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/19a3416856b07e9f288f0f99208b17619fc98f1723979b930caafa44e83c27e0.jpg)

![1ec993a5ff8d045cd1149d2446fd3f20d8001d4280b20a329e74128dbe1adfd5.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/1ec993a5ff8d045cd1149d2446fd3f20d8001d4280b20a329e74128dbe1adfd5.jpg)

![2ceecaf40fdd33572b84b1f647c4d576f88cce8b7d073e28bcc7b6c31b45445e.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/2ceecaf40fdd33572b84b1f647c4d576f88cce8b7d073e28bcc7b6c31b45445e.jpg)

![31923249887bcbd8e1eb6108db99e154986bd41e248c3950443b14a4dd36c891.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/31923249887bcbd8e1eb6108db99e154986bd41e248c3950443b14a4dd36c891.jpg)

![32ee491494dfad25b5caa6a63212a34e05da44641911694cf07bc9b8581ce8c8.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/32ee491494dfad25b5caa6a63212a34e05da44641911694cf07bc9b8581ce8c8.jpg)

![3595fa273eea880b25f9618f7879984220a09828f8dfe5acf604b3c45d8b2fb3.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/3595fa273eea880b25f9618f7879984220a09828f8dfe5acf604b3c45d8b2fb3.jpg)

![3ab0e36b07b20664d8ca4687c9e50be36b5cae98922af0053a98c4c4db4fea82.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/3ab0e36b07b20664d8ca4687c9e50be36b5cae98922af0053a98c4c4db4fea82.jpg)

![3e42e378ba2aa7751864c64a86f6c6184d5a2cc26cd1cba9375df80d394b5b7a.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/3e42e378ba2aa7751864c64a86f6c6184d5a2cc26cd1cba9375df80d394b5b7a.jpg)

![597d2fa7372155973287c97f19f1c92c4e7f4b1f94b9364b914d84bbd989c1f1.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/597d2fa7372155973287c97f19f1c92c4e7f4b1f94b9364b914d84bbd989c1f1.jpg)

![6d5251c2d428d7768af3fa3d53631edb3b70351d85b95f4f6606574eb429a7b9.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/6d5251c2d428d7768af3fa3d53631edb3b70351d85b95f4f6606574eb429a7b9.jpg)

![815e6c1dcdea5758317b99c2500f4fec2380a84a062b01cdfeaad62c47ffa620.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/815e6c1dcdea5758317b99c2500f4fec2380a84a062b01cdfeaad62c47ffa620.jpg)

![82c52d32fb30d2e4b71070c9d9512dc88fa7b236f93687d1f7ac60a97d9a6d71.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/82c52d32fb30d2e4b71070c9d9512dc88fa7b236f93687d1f7ac60a97d9a6d71.jpg)

![8a21d9966fc7caa56caeec9235be906694dc97dd174da2330e078f40debc9aee.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/images/8a21d9966fc7caa56caeec9235be906694dc97dd174da2330e078f40debc9aee.jpg)

### Tables

![0729709f37c40ceac7f70e61c89cf1cc87dc313150d34d2a24d061e113a02465.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/0729709f37c40ceac7f70e61c89cf1cc87dc313150d34d2a24d061e113a02465.jpg)

![925f2b45c1721203fa9b556e33f36a54eae305bf1dc7a52d5d1fbf9af0b5259f.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/925f2b45c1721203fa9b556e33f36a54eae305bf1dc7a52d5d1fbf9af0b5259f.jpg)

![93122e0cbc99a77806d0362204669b9fbe54e2bfb97f39cd13d72d5a5aeef3d7.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/93122e0cbc99a77806d0362204669b9fbe54e2bfb97f39cd13d72d5a5aeef3d7.jpg)

![b176f3c8da2a9f9e9b77a7f9aa0055f6dc9a2bb7c8cfd13b3c5a4c741e3061c2.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/b176f3c8da2a9f9e9b77a7f9aa0055f6dc9a2bb7c8cfd13b3c5a4c741e3061c2.jpg)

![b929d21b0bc709891ec3abc7ba7b148870ce3aa8de363134e3bae4a5e8b59fa4.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/b929d21b0bc709891ec3abc7ba7b148870ce3aa8de363134e3bae4a5e8b59fa4.jpg)

![dbd335d2399abad252046cf758662a954ba05e7403468e4552f7b2a1cbac14c1.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/dbd335d2399abad252046cf758662a954ba05e7403468e4552f7b2a1cbac14c1.jpg)

![e4ee0ca48d66f0e36d87c083911f42b68828a7ce99db35e80accd59f3dd27f5c.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/e4ee0ca48d66f0e36d87c083911f42b68828a7ce99db35e80accd59f3dd27f5c.jpg)

![e83c287e7768246f7ba77cf45e7e8b84ad287a6092d58eadfe29e42ba19550e1.jpg](../icml_results/3740_Primal-Spectral%20Generative%20Modeling_%20Fast%20Analytical%20Generation%20via%20Pseudoinverse%20L%C3%A9vy%20Inversion/tables/e83c287e7768246f7ba77cf45e7e8b84ad287a6092d58eadfe29e42ba19550e1.jpg)

<a id="paper-3701"></a>

## DenseSteer: Steering Small Language Models towards Dense Math Reasoning


### Images

![07a3537970844a2d758b8703323c356b9972aa5938452b0c6ed490308ca018fd.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/07a3537970844a2d758b8703323c356b9972aa5938452b0c6ed490308ca018fd.jpg)

![3aefa16567639f12856b913363dcce1a1451a83d002787aa1ec01e4b45b71e8b.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/3aefa16567639f12856b913363dcce1a1451a83d002787aa1ec01e4b45b71e8b.jpg)

![4a62a09c75b525bff3fffb355b0f9056c95c0f01defb341e2616166f54f32a58.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/4a62a09c75b525bff3fffb355b0f9056c95c0f01defb341e2616166f54f32a58.jpg)

![4f8e69db2dabb3fc6d5e070287a50c054f9d3979f43a1966847920f3ac6f8ee5.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/4f8e69db2dabb3fc6d5e070287a50c054f9d3979f43a1966847920f3ac6f8ee5.jpg)

![554b7329997afad672399e586ad893022ccc53b3e75dea3bf091152f272fa773.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/554b7329997afad672399e586ad893022ccc53b3e75dea3bf091152f272fa773.jpg)

![97295d05212b29c1f12855c1d924595b2c7e1b9791472caebb5c1561538144b9.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/97295d05212b29c1f12855c1d924595b2c7e1b9791472caebb5c1561538144b9.jpg)

![b5b286291333e303c14eaee30fb055d644ff6d8885b9c928efcf063bfb200c9c.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/b5b286291333e303c14eaee30fb055d644ff6d8885b9c928efcf063bfb200c9c.jpg)

![ecc69b315fdbdd5b54f89263ae180cf60af5dcd1c47a70118e9c8386ff8e502d.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/ecc69b315fdbdd5b54f89263ae180cf60af5dcd1c47a70118e9c8386ff8e502d.jpg)

![f1775b6b58f74d4ba229d73686c6d188d2aec8769ef23a001ea7f52b30f26094.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/f1775b6b58f74d4ba229d73686c6d188d2aec8769ef23a001ea7f52b30f26094.jpg)

![fc8508352c76569c24033d13fa78f9126616dd098247c45fc4ef5954b4c1c8f1.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/images/fc8508352c76569c24033d13fa78f9126616dd098247c45fc4ef5954b4c1c8f1.jpg)

### Tables

![177629e8f8a4dc3e9dd75a0abc24305f47b3575ed44ea585d1ff1aa7d621298a.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/177629e8f8a4dc3e9dd75a0abc24305f47b3575ed44ea585d1ff1aa7d621298a.jpg)

![2620f12f9d780fd673b586e91d6dce68e0c28a1267f475ce1049a89520541b37.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/2620f12f9d780fd673b586e91d6dce68e0c28a1267f475ce1049a89520541b37.jpg)

![2aefba03220eb85b80963f1c38ac866a267725e49de536edc4e67848a65a135f.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/2aefba03220eb85b80963f1c38ac866a267725e49de536edc4e67848a65a135f.jpg)

![6e09e92fabecaf874b4fd7ebe4c4743db11e806e052f7db0be09549f7a737b4b.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/6e09e92fabecaf874b4fd7ebe4c4743db11e806e052f7db0be09549f7a737b4b.jpg)

![8f85c23bf45bfc2f97a5201997d07bbc23cbe348905828681799094f69f31c92.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/8f85c23bf45bfc2f97a5201997d07bbc23cbe348905828681799094f69f31c92.jpg)

![91dc92aca8d3208368ba3be68230b253abfd9bbbc3ab0f7c4f1af86ad9980c6c.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/91dc92aca8d3208368ba3be68230b253abfd9bbbc3ab0f7c4f1af86ad9980c6c.jpg)

![cf86ccb28d0aad9be7bb519f3864d403a183b961601d3e29222f6c7fd11955fc.jpg](../icml_results/3741_DenseSteer_%20Steering%20Small%20Language%20Models%20towards%20Dense%20Math%20Reasoning/tables/cf86ccb28d0aad9be7bb519f3864d403a183b961601d3e29222f6c7fd11955fc.jpg)

<a id="paper-3702"></a>

## LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws


### Images

![13556f060db586f36bbb0362df7e57001059a79289e5b727170143b3fd4eeaac.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/13556f060db586f36bbb0362df7e57001059a79289e5b727170143b3fd4eeaac.jpg)

![33108c01343651d4fd73d381c5293da03a28415e38a8e16738ea715829d22222.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/33108c01343651d4fd73d381c5293da03a28415e38a8e16738ea715829d22222.jpg)

![508227b611cdd038666420a061efbb6370cbd1a1ff36a21e6b15b7d529b949ab.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/508227b611cdd038666420a061efbb6370cbd1a1ff36a21e6b15b7d529b949ab.jpg)

![5c37b38be06431d40f03255c6719ff4e58d27189689a1838f2b513bd8b0b3b88.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/5c37b38be06431d40f03255c6719ff4e58d27189689a1838f2b513bd8b0b3b88.jpg)

![5f18b21bf600857de71bb3266bd0bb4176e27c6eb0c0c2c47189dc45aeb9c111.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/5f18b21bf600857de71bb3266bd0bb4176e27c6eb0c0c2c47189dc45aeb9c111.jpg)

![665e2fb45e0f9a8cea8488a54f0ad82ab13309fd0bc3881284a422612d1102c7.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/665e2fb45e0f9a8cea8488a54f0ad82ab13309fd0bc3881284a422612d1102c7.jpg)

![754ecc72e7158b7473344df1b547ba3887498f9e21ae362dafb5a7566867a56b.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/754ecc72e7158b7473344df1b547ba3887498f9e21ae362dafb5a7566867a56b.jpg)

![7dacb58ecb5ee21c104b76628f8e413ae17c726b8f26dac2d265312a8184a677.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/7dacb58ecb5ee21c104b76628f8e413ae17c726b8f26dac2d265312a8184a677.jpg)

![85dcec25ab4764e7dc8d62d3f0c35086766dc1e4d6722eb187ba50d10fa9e28e.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/85dcec25ab4764e7dc8d62d3f0c35086766dc1e4d6722eb187ba50d10fa9e28e.jpg)

![8d0ea3aec666c1a99411412947b02852661336f43c8866a3d960fba14644e24f.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/8d0ea3aec666c1a99411412947b02852661336f43c8866a3d960fba14644e24f.jpg)

![a52d8e6c53224661b8860fc235aa0668d5fdf9ce0f377346c7f57a6ffbc51b03.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/a52d8e6c53224661b8860fc235aa0668d5fdf9ce0f377346c7f57a6ffbc51b03.jpg)

![ab37c3669227e97639dff18cbd51e0626abcbc9dcf58c81eb2cad5594f7562a6.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/ab37c3669227e97639dff18cbd51e0626abcbc9dcf58c81eb2cad5594f7562a6.jpg)

![af2496bfc9a1525842a40d33e63ab999e84331851f5a7fd77792328e49405320.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/af2496bfc9a1525842a40d33e63ab999e84331851f5a7fd77792328e49405320.jpg)

![cad5b6ca59aef3bf5529cad6514e02ec036773d04796748a82b73c15627e6c4c.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/cad5b6ca59aef3bf5529cad6514e02ec036773d04796748a82b73c15627e6c4c.jpg)

![cda37edc4b74e5276e27cb440a2e57a8d72bcee209df592b5e65b9bf31b06501.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/cda37edc4b74e5276e27cb440a2e57a8d72bcee209df592b5e65b9bf31b06501.jpg)

![dc6e3f7ced04c8109fa945b6575a2efb5d5ab009558d4dba0ee67d323326a57c.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/images/dc6e3f7ced04c8109fa945b6575a2efb5d5ab009558d4dba0ee67d323326a57c.jpg)

### Tables

![11504fbebfd77956efc806e11a06a30cca669060aa73d1d87a04ca9aaf0cac02.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/11504fbebfd77956efc806e11a06a30cca669060aa73d1d87a04ca9aaf0cac02.jpg)

![35dae3db6d1f0f0f6c501d9d0d85d04533d30aa1b62f329d5f5a640ac0e0b84f.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/35dae3db6d1f0f0f6c501d9d0d85d04533d30aa1b62f329d5f5a640ac0e0b84f.jpg)

![3831e09782d862a81f3bb6378467cfcc521c657ac67e32fa06fb6141aa0a1234.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/3831e09782d862a81f3bb6378467cfcc521c657ac67e32fa06fb6141aa0a1234.jpg)

![62e1353441528f9ee0ef0592fcad02b693a58e7a6b0bbb8f3fbf416359731105.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/62e1353441528f9ee0ef0592fcad02b693a58e7a6b0bbb8f3fbf416359731105.jpg)

![75ee1a26ae739b491ce18db82f47c5e54fcbfb1d7fd55ea4349140b966428a1b.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/75ee1a26ae739b491ce18db82f47c5e54fcbfb1d7fd55ea4349140b966428a1b.jpg)

![7ed462640a061ef85aa1b8d3a13ee527c00258353d3bc9e4dba91cd3e9b224f9.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/7ed462640a061ef85aa1b8d3a13ee527c00258353d3bc9e4dba91cd3e9b224f9.jpg)

![87edb0acf6d50ecd021294e81ecd56f1c9cd9dc2f97f26d3e11509696dbb50a1.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/87edb0acf6d50ecd021294e81ecd56f1c9cd9dc2f97f26d3e11509696dbb50a1.jpg)

![9862cf8ad63926e33a1e62b8c9bc3f72525a7cf017e0cc1871c3de2da9893044.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/9862cf8ad63926e33a1e62b8c9bc3f72525a7cf017e0cc1871c3de2da9893044.jpg)

![ae2848bffd439b83323823ec0df181d15bf4aee75e9bf1f27e909722f6ca57bd.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/ae2848bffd439b83323823ec0df181d15bf4aee75e9bf1f27e909722f6ca57bd.jpg)

![aef56bd969b7a6e14a10f521bff8f684ceead1865b6ef7abddd1d9b08acf54ae.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/aef56bd969b7a6e14a10f521bff8f684ceead1865b6ef7abddd1d9b08acf54ae.jpg)

![e213cd0bfa5db06cb67ba4a985a7f131d77487c26302f59d3281f7ec16fd1451.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/e213cd0bfa5db06cb67ba4a985a7f131d77487c26302f59d3281f7ec16fd1451.jpg)

![e3ae2ea31b6f78c3ed568f1d747d9f2877c72731fdda1d03a89e74a0483cfd32.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/e3ae2ea31b6f78c3ed568f1d747d9f2877c72731fdda1d03a89e74a0483cfd32.jpg)

![e89a9bd820f04fc90e1ec4c778d75ede4b783e01412405dfc9f801e9df6f1b20.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/e89a9bd820f04fc90e1ec4c778d75ede4b783e01412405dfc9f801e9df6f1b20.jpg)

![eff576891b8e29cce8e1dbfe5324f36182c638d4ae1fd8abd5be87301a18199a.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/eff576891b8e29cce8e1dbfe5324f36182c638d4ae1fd8abd5be87301a18199a.jpg)

![f01e2893620f88a737853b2c03749c42d09e234364b7824ef282f67c90c1833e.jpg](../icml_results/3742_LLMs%20as%20Noisy%20Channels_%20A%20Shannon%20Perspective%20on%20Model%20Capacity%20and%20Scaling%20Laws/tables/f01e2893620f88a737853b2c03749c42d09e234364b7824ef282f67c90c1833e.jpg)

<a id="paper-3703"></a>

## ReaForest: Fostering Generative Video Reasoning for Spatial Planning


### Images

![03e127cc156bee61ef75bea722c82dcf950ffe636da346016b55ecf40f869efc.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/03e127cc156bee61ef75bea722c82dcf950ffe636da346016b55ecf40f869efc.jpg)

![1f17acc5dab4a6a9bd6cf7c5a81fd2d3541876965038c3de19cbbbaafd8751a7.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/1f17acc5dab4a6a9bd6cf7c5a81fd2d3541876965038c3de19cbbbaafd8751a7.jpg)

![66904423601463d0ffc108e6824786e4eb4ed8d072ad66e72e1abeeaa5615f96.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/66904423601463d0ffc108e6824786e4eb4ed8d072ad66e72e1abeeaa5615f96.jpg)

![6a269282a4c9c3313088910bf4031ac287caa349f4e5adbffcd2722681967cb2.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/6a269282a4c9c3313088910bf4031ac287caa349f4e5adbffcd2722681967cb2.jpg)

![815ae3e22fce8fa6d778b9560851d5f50d6bfe0492e269bb529902b3bee4e7d7.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/815ae3e22fce8fa6d778b9560851d5f50d6bfe0492e269bb529902b3bee4e7d7.jpg)

![92682516edb6c63d9edfd64c3520bb9a298f8ef5d1cc85d109399bb3c1295990.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/92682516edb6c63d9edfd64c3520bb9a298f8ef5d1cc85d109399bb3c1295990.jpg)

![a72d15df25583d3cc8c7c78602a1f8fa3a482eae2b69c248483fd05e06ea062b.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/a72d15df25583d3cc8c7c78602a1f8fa3a482eae2b69c248483fd05e06ea062b.jpg)

![bdc3338a7067d529fe7053a8f5e5b4e386fbdb4a0a421d8e1d109d60a1d67a7f.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/bdc3338a7067d529fe7053a8f5e5b4e386fbdb4a0a421d8e1d109d60a1d67a7f.jpg)

![c69fcd8684708314a8a45b058da9450b9c770273491af93cf4b4786fcb785574.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/c69fcd8684708314a8a45b058da9450b9c770273491af93cf4b4786fcb785574.jpg)

![c6c4a275abc591a5139f5fb7867d9974c370adeec9980b0a187054c03ddaa23c.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/c6c4a275abc591a5139f5fb7867d9974c370adeec9980b0a187054c03ddaa23c.jpg)

![d46544db78aa59b0c140733039fdee6944ea047eef7b2011c6f6345d8f6beba0.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/d46544db78aa59b0c140733039fdee6944ea047eef7b2011c6f6345d8f6beba0.jpg)

![e668d0feea8da47eb4a9239197659480fd80c0c3f76740933e3f97f4a8027a93.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/e668d0feea8da47eb4a9239197659480fd80c0c3f76740933e3f97f4a8027a93.jpg)

![f6b4b1d93dd89400408ddb2f810e95e1514a1778f8acd50c9cbb8e1cacd1b021.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/images/f6b4b1d93dd89400408ddb2f810e95e1514a1778f8acd50c9cbb8e1cacd1b021.jpg)

### Tables

![2246d580127114f55dc85abd75f241f5fcc420c2a343a10e6910d8aa5caaaf7d.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/2246d580127114f55dc85abd75f241f5fcc420c2a343a10e6910d8aa5caaaf7d.jpg)

![401463dad04b0351a0a491a4567546e4a7db55c7891ba9f568e087862826c875.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/401463dad04b0351a0a491a4567546e4a7db55c7891ba9f568e087862826c875.jpg)

![4800c34b90925a2edf1cf1c4d39ea7a535d2d7275d04d132e17295267e809663.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/4800c34b90925a2edf1cf1c4d39ea7a535d2d7275d04d132e17295267e809663.jpg)

![764d68af47447bbce911d6e8f6f3da74d7fe31a7645ba76a2915bea0f98d87f1.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/764d68af47447bbce911d6e8f6f3da74d7fe31a7645ba76a2915bea0f98d87f1.jpg)

![853677e58ed7662157eed1454ebed3d3b7e4ba0c7b3638fc25a71b48bb294963.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/853677e58ed7662157eed1454ebed3d3b7e4ba0c7b3638fc25a71b48bb294963.jpg)

![949dfc19abd72497fb133ad58ab2f6a37a1b8c3645833cde92b87f6ed43fb094.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/949dfc19abd72497fb133ad58ab2f6a37a1b8c3645833cde92b87f6ed43fb094.jpg)

![b72b2ef88cbf84fcbdc2fcf87451a6bb994ecd24b5db51ab2de374b74b92a1e9.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/b72b2ef88cbf84fcbdc2fcf87451a6bb994ecd24b5db51ab2de374b74b92a1e9.jpg)

![e675d0f58b267e7a6999b9383cac6e3635e5be23b1288d125cb761adbd1af4be.jpg](../icml_results/3743_ReaForest_%20Fostering%20Generative%20Video%20Reasoning%20for%20Spatial%20Planning/tables/e675d0f58b267e7a6999b9383cac6e3635e5be23b1288d125cb761adbd1af4be.jpg)

<a id="paper-3704"></a>

## Must All Negatives Be Pushed Away Equally? Uncertainty-Aware Cross-View Geo-Localization via Normal Inverse Gamma Distribution


### Images

![17ea07ceba4800f57356bb0b89d54620cbe4ddd5dc398471c31f9d111922a613.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/17ea07ceba4800f57356bb0b89d54620cbe4ddd5dc398471c31f9d111922a613.jpg)

![3d925c312f9e538075b83d1af8f14d8bd19d7ec93bdfbf67850f703c052afd59.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/3d925c312f9e538075b83d1af8f14d8bd19d7ec93bdfbf67850f703c052afd59.jpg)

![43998a5ee5cbab751fc282462b1199b3c620c85b2587e5af8f4f06b4e878b0a1.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/43998a5ee5cbab751fc282462b1199b3c620c85b2587e5af8f4f06b4e878b0a1.jpg)

![45ce93edadc5305508a0a4b95bc615feff11be70e6eacdb4d61dac23e2700523.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/45ce93edadc5305508a0a4b95bc615feff11be70e6eacdb4d61dac23e2700523.jpg)

![5ff7fdd437211a466739879ac4c63eef31a44c25c010d8804406db6820d6a0ba.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/5ff7fdd437211a466739879ac4c63eef31a44c25c010d8804406db6820d6a0ba.jpg)

![d26f4c405a4088104a893cf8d838a84dada25b0b435634ed2b7aa400c124d2fb.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/d26f4c405a4088104a893cf8d838a84dada25b0b435634ed2b7aa400c124d2fb.jpg)

![ddb9b0758acf14a8e793f8dbb4ffa99920b2f6173777ed08b262f5abfb202fb2.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/ddb9b0758acf14a8e793f8dbb4ffa99920b2f6173777ed08b262f5abfb202fb2.jpg)

![edbcabd312942330fe0a25ba638fd0f76916ba05ef299ce44ec6ab5b922bdfed.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/images/edbcabd312942330fe0a25ba638fd0f76916ba05ef299ce44ec6ab5b922bdfed.jpg)

### Tables

![2af88ff883997027a469f6db156353ff9e5ca7a8cf67a3433fc1df19b32acf1b.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/2af88ff883997027a469f6db156353ff9e5ca7a8cf67a3433fc1df19b32acf1b.jpg)

![435984ccac94bc4871642d09615caa129638f154e9a13832cd05daa8ab2fc6ba.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/435984ccac94bc4871642d09615caa129638f154e9a13832cd05daa8ab2fc6ba.jpg)

![4975596fa7f2d879aad539a85053a2ddcc27ecad80a0f17fcb61cede93e21a66.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/4975596fa7f2d879aad539a85053a2ddcc27ecad80a0f17fcb61cede93e21a66.jpg)

![4d957aae36401629d1ea263da5ac6f42c7cf9ef6c823291464b6afef09071497.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/4d957aae36401629d1ea263da5ac6f42c7cf9ef6c823291464b6afef09071497.jpg)

![63983466072a3c720562febe952e8bab08995a730dd72c8d9d23f48c8edaed05.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/63983466072a3c720562febe952e8bab08995a730dd72c8d9d23f48c8edaed05.jpg)

![9bad4a89c86adf29b0b32953e876e40a529cc3f4cd74589399fc84ad1d72b6fa.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/9bad4a89c86adf29b0b32953e876e40a529cc3f4cd74589399fc84ad1d72b6fa.jpg)

![9c76242a4cfb2880fc940ebe9e8a8b56b53a36c15e47cc55a8578def6d0b80e5.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/9c76242a4cfb2880fc940ebe9e8a8b56b53a36c15e47cc55a8578def6d0b80e5.jpg)

![ad906e00a5b1aa129110c755ab2bbf9630f0823dfc167c6afd4cf49867fe2440.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/ad906e00a5b1aa129110c755ab2bbf9630f0823dfc167c6afd4cf49867fe2440.jpg)

![b4d424613f2e7cc2a48a7ceef9c4255d088b24110985608886f461a72069b199.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/b4d424613f2e7cc2a48a7ceef9c4255d088b24110985608886f461a72069b199.jpg)

![d28503655f5698e09288896e4133d2f1e4bdeafc7bf6042041ecd8736cf93ad2.jpg](../icml_results/3744_Must%20All%20Negatives%20Be%20Pushed%20Away%20Equally_%20Uncertainty-Aware%20Cross-View%20Geo-Localization%20via%20Normal%20/tables/d28503655f5698e09288896e4133d2f1e4bdeafc7bf6042041ecd8736cf93ad2.jpg)

<a id="paper-3705"></a>

## Calibrating Conservatism for Scalable Oversight


### Images

![0325f3ecfcfae90ad11263671160abd62ffe1f870a075fe4c2bc1a8e914decdf.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/images/0325f3ecfcfae90ad11263671160abd62ffe1f870a075fe4c2bc1a8e914decdf.jpg)

![0f6d1a6d5e09ac7e40f615501590b7a1af0d23aeafec9e426be2d4667743e06d.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/images/0f6d1a6d5e09ac7e40f615501590b7a1af0d23aeafec9e426be2d4667743e06d.jpg)

### Tables

![00730d9fb70ae1b372538aced309bb0403fd987d7ada576751efead8433822da.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/00730d9fb70ae1b372538aced309bb0403fd987d7ada576751efead8433822da.jpg)

![3149549e007486834d3d71bdc2a65daa52e5db384cf08a43804067b0b40d61de.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/3149549e007486834d3d71bdc2a65daa52e5db384cf08a43804067b0b40d61de.jpg)

![322fb3deff023bdf33827086a23de6f1b3d5049efe6bdc2b42ae362e7863653d.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/322fb3deff023bdf33827086a23de6f1b3d5049efe6bdc2b42ae362e7863653d.jpg)

![4b984fb395722bb4ee21e1e528680e46267c6a1e4550e4b6c45b22f125711d41.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/4b984fb395722bb4ee21e1e528680e46267c6a1e4550e4b6c45b22f125711d41.jpg)

![59903de4aba632ac00a7d4237bf83044e3869f2c4c0ece359047b390e90055b4.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/59903de4aba632ac00a7d4237bf83044e3869f2c4c0ece359047b390e90055b4.jpg)

![825346e5527a43dfba6d5c18ff44818919e94c4b32024de05baa9a9d011a79c9.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/825346e5527a43dfba6d5c18ff44818919e94c4b32024de05baa9a9d011a79c9.jpg)

![98ea210372f5f892d99b09c1402805cb073f9c61f4f8e77df36b9dd2d6ab759e.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/98ea210372f5f892d99b09c1402805cb073f9c61f4f8e77df36b9dd2d6ab759e.jpg)

![e965072dd9d925ca5c1a32b7a163a0135991eacb37ab965ba6e4d3eef801c579.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/e965072dd9d925ca5c1a32b7a163a0135991eacb37ab965ba6e4d3eef801c579.jpg)

![f44f84701adab90c6a9b4fcb51ccc5b4c2df33885caf3301ede4b118cf4d4952.jpg](../icml_results/3745_Calibrating%20Conservatism%20for%20Scalable%20Oversight/tables/f44f84701adab90c6a9b4fcb51ccc5b4c2df33885caf3301ede4b118cf4d4952.jpg)

<a id="paper-3706"></a>

## The Oversight Game: Learning to Cooperatively Balance an AI Agent’s Safety and Autonomy


### Images

![1b005016a0e2a5a6a227464ea9fd800a4ceb549fedc25dcdfc8f8ef88f112811.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/1b005016a0e2a5a6a227464ea9fd800a4ceb549fedc25dcdfc8f8ef88f112811.jpg)

![2ee2d2687184cba8e23755cc34fc10a24ab48fa2c9f457af2f919b3cbe7201b0.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/2ee2d2687184cba8e23755cc34fc10a24ab48fa2c9f457af2f919b3cbe7201b0.jpg)

![3fc6aaafbdd3770f4a061a90f5bf484936312bcbf8cc6bbdb021ea4ddc3907e7.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/3fc6aaafbdd3770f4a061a90f5bf484936312bcbf8cc6bbdb021ea4ddc3907e7.jpg)

![4e2147dcc53b4ee952d86c0079bc2c98e1afc5f65cc4b8ea5e0672fc55fd134c.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/4e2147dcc53b4ee952d86c0079bc2c98e1afc5f65cc4b8ea5e0672fc55fd134c.jpg)

![4f579dcce0b0b46f837c5de57371bc306900a5bb460b4a68533c3ccec3721ded.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/4f579dcce0b0b46f837c5de57371bc306900a5bb460b4a68533c3ccec3721ded.jpg)

![a26e270edcd52877494af16a7c063e1fbdd20539b749117103f94ccba0215ea2.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/a26e270edcd52877494af16a7c063e1fbdd20539b749117103f94ccba0215ea2.jpg)

![bd73131eee2c8a753b2945f821f9e2c4e5ca7309890fde403d2d45cedcea41e2.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/bd73131eee2c8a753b2945f821f9e2c4e5ca7309890fde403d2d45cedcea41e2.jpg)

![c10b2595349e431ae11e9820030a1b81b10504afc8fad2c483858581048764b1.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/images/c10b2595349e431ae11e9820030a1b81b10504afc8fad2c483858581048764b1.jpg)

### Tables

![51730bd15ab10d267a1cd54ea68aff366e12f7aba862a6bbcbab2a2098644580.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/tables/51730bd15ab10d267a1cd54ea68aff366e12f7aba862a6bbcbab2a2098644580.jpg)

![5aa5f7582aea6c769655a90aecac484508b41293d8f16d78d6c275bc64e66d85.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/tables/5aa5f7582aea6c769655a90aecac484508b41293d8f16d78d6c275bc64e66d85.jpg)

![693e4e5a0a042a12a347b2f02a43a7e22ec2ac6cee3f675de50a879acf7876ba.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/tables/693e4e5a0a042a12a347b2f02a43a7e22ec2ac6cee3f675de50a879acf7876ba.jpg)

![7e12a0c0d7a03c0be95b4888378d6b35b27095166622886d31d0ec4df0a60ac0.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/tables/7e12a0c0d7a03c0be95b4888378d6b35b27095166622886d31d0ec4df0a60ac0.jpg)

![aad160c890d373ed3bd25dfee03e986bc4d696ed38c27d17908c8a297a8db783.jpg](../icml_results/3746_The%20Oversight%20Game_%20Learning%20to%20Cooperatively%20Balance%20an%20AI%20Agent%E2%80%99s%20Safety%20and%20Autonomy/tables/aad160c890d373ed3bd25dfee03e986bc4d696ed38c27d17908c8a297a8db783.jpg)

<a id="paper-3707"></a>

## SoftBinary Coding: A New Information-Theoretic Paradigm for Neural Compression via Fast Channel Simulation


### Images

![07ec3f93d459348c5bbf9b3b822021552363b97985a5a60025a72cb43b1c07c3.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/07ec3f93d459348c5bbf9b3b822021552363b97985a5a60025a72cb43b1c07c3.jpg)

![29580804ae146e2ab7b57cec0fd1602da1d8fb01e170f9db35b441becc31d1ed.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/29580804ae146e2ab7b57cec0fd1602da1d8fb01e170f9db35b441becc31d1ed.jpg)

![31a3eea0b5f2cc1cec57d9dc21d0c46a20c1471ce94ffcc503d8dd5a3347fa99.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/31a3eea0b5f2cc1cec57d9dc21d0c46a20c1471ce94ffcc503d8dd5a3347fa99.jpg)

![412bae265ad1a4565daea6bbddb27df9c3c442cdf15425f866d255dadf855742.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/412bae265ad1a4565daea6bbddb27df9c3c442cdf15425f866d255dadf855742.jpg)

![43fe0fcd7ae2facb472ab460128c34d4398ebbb57d0c23cb483f73ac0b9e8da4.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/43fe0fcd7ae2facb472ab460128c34d4398ebbb57d0c23cb483f73ac0b9e8da4.jpg)

![82393f7d95cee2c259e206be1e0d7aac5ea3be7b54d6dab25ea18005b9ec2bbf.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/82393f7d95cee2c259e206be1e0d7aac5ea3be7b54d6dab25ea18005b9ec2bbf.jpg)

![9c2d4d81966416c9da295921641ecc4c190b7a1f7d0e60bb14b25a61a410e7f7.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/9c2d4d81966416c9da295921641ecc4c190b7a1f7d0e60bb14b25a61a410e7f7.jpg)

![c693057d44233307e1be45a62e74be27a72d5a4b266263fd03078c69ec22cc12.jpg](../icml_results/3747_SoftBinary%20Coding_%20A%20New%20Information-Theoretic%20Paradigm%20for%20Neural%20Compression%20via%20Fast%20Channel%20Simu/images/c693057d44233307e1be45a62e74be27a72d5a4b266263fd03078c69ec22cc12.jpg)

<a id="paper-3708"></a>

## The Latent Color Subspace: Emergent Order in High-Dimensional Chaos


### Images

![00ca2f61ec0e99b9098c26b3ce43f8ae6cecbe71fd17d9a4a3ee673a446d5064.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/00ca2f61ec0e99b9098c26b3ce43f8ae6cecbe71fd17d9a4a3ee673a446d5064.jpg)

![3169e13e8d94786ef16b1feb4fcd4cf3f1e006ab29c0eafb2f908c091f5d1ea6.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/3169e13e8d94786ef16b1feb4fcd4cf3f1e006ab29c0eafb2f908c091f5d1ea6.jpg)

![33b78802c966f72c8b89f7559cd1b26e1cfe2e79b50283037209fd1bf5ec296c.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/33b78802c966f72c8b89f7559cd1b26e1cfe2e79b50283037209fd1bf5ec296c.jpg)

![471d677bd9fb9112c33b048f3b94c698ed620fa6461c9a3fe37a8bb9e591bd8a.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/471d677bd9fb9112c33b048f3b94c698ed620fa6461c9a3fe37a8bb9e591bd8a.jpg)

![4f50bf9d913f97944ccbd620a0f2ef22048e7d9378dca43d3b80eb8b98daa621.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/4f50bf9d913f97944ccbd620a0f2ef22048e7d9378dca43d3b80eb8b98daa621.jpg)

![5062f7b02b23714ca18180c0b32deed3c504cfbc25da79356ab0442e12590145.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/5062f7b02b23714ca18180c0b32deed3c504cfbc25da79356ab0442e12590145.jpg)

![572bb8039e0dbb5cc0ef2bf48ea5699ea41f298cdaf25f0acbad28318a9e7c83.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/572bb8039e0dbb5cc0ef2bf48ea5699ea41f298cdaf25f0acbad28318a9e7c83.jpg)

![5cfe5021cd3f21c740ceb713fbb942aec2ec5ba8ac54ac1b5c1b1c2cfc3f0935.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/5cfe5021cd3f21c740ceb713fbb942aec2ec5ba8ac54ac1b5c1b1c2cfc3f0935.jpg)

![6dc40ab6592e675e1ee8144cd1b9c200866e23c3b879ddca064cc1fa97dab319.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/6dc40ab6592e675e1ee8144cd1b9c200866e23c3b879ddca064cc1fa97dab319.jpg)

![71858c7126d1438473018fb47ef9b8b6ff48b15afdd1c1b3b78cfd48f5377194.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/71858c7126d1438473018fb47ef9b8b6ff48b15afdd1c1b3b78cfd48f5377194.jpg)

![7b0ba36aa2e5ff92fbce0acd609ea7b19048e980c3dad4ee02e38f12fd261c0b.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/7b0ba36aa2e5ff92fbce0acd609ea7b19048e980c3dad4ee02e38f12fd261c0b.jpg)

![a404e2e659f3b07904143afe69b26fd0d42bb10886c013b9cf81a26d282202fd.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/a404e2e659f3b07904143afe69b26fd0d42bb10886c013b9cf81a26d282202fd.jpg)

![be5122f229b7000c57abcc2b0c2e7fe1b56941817276e933000ecdce9cabc5f3.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/be5122f229b7000c57abcc2b0c2e7fe1b56941817276e933000ecdce9cabc5f3.jpg)

![bef24eede1570dd662c84fca1d5caf0c8ba3b1ef0b58327b56a98ffd59b8a52d.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/bef24eede1570dd662c84fca1d5caf0c8ba3b1ef0b58327b56a98ffd59b8a52d.jpg)

![bfa6806bc5d0cff0fe5f9668ad40e3a42df37b12e0d642411bf54f44285cd5aa.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/bfa6806bc5d0cff0fe5f9668ad40e3a42df37b12e0d642411bf54f44285cd5aa.jpg)

![c8c49c68807856fa4cf4ad7ad6a4791ea319a9cfee2cb97d47f157aece48d492.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/c8c49c68807856fa4cf4ad7ad6a4791ea319a9cfee2cb97d47f157aece48d492.jpg)

![ec1a1457a7a438113a6b151e4a6296f4cedd70ea92e1472718c2a7f7b3d7e5d9.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/images/ec1a1457a7a438113a6b151e4a6296f4cedd70ea92e1472718c2a7f7b3d7e5d9.jpg)

### Tables

![0358317ddda9ce475a777ae64f94c1f8f9a5d819dd2ac8a4cb0a2828a13f8afc.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/0358317ddda9ce475a777ae64f94c1f8f9a5d819dd2ac8a4cb0a2828a13f8afc.jpg)

![1b18c869d7573c0c3a5577c2d0188d9fa249881c211974cf8371f7038f1b9305.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/1b18c869d7573c0c3a5577c2d0188d9fa249881c211974cf8371f7038f1b9305.jpg)

![2244401b655fd7f0dea8c8d22b4473e82397da076a6678373b151bde90e90b26.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/2244401b655fd7f0dea8c8d22b4473e82397da076a6678373b151bde90e90b26.jpg)

![2d208718dfad715c44e7b24a82e1482c6dc58ff31d32ef568eb4a2b2c09fa517.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/2d208718dfad715c44e7b24a82e1482c6dc58ff31d32ef568eb4a2b2c09fa517.jpg)

![6b420add39e0d7fca4b95348967e5a19e9864342e6f202d4f87e8b0d5edd236f.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/6b420add39e0d7fca4b95348967e5a19e9864342e6f202d4f87e8b0d5edd236f.jpg)

![8a3391dd99ecc5a4458a85e2e2cb60ce548625110c8f9f8691ab38dc77da2575.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/8a3391dd99ecc5a4458a85e2e2cb60ce548625110c8f9f8691ab38dc77da2575.jpg)

![a8a5da51eaffb772388a0a4136c056c585961f77938d2dc77bf526e7550a5ef3.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/a8a5da51eaffb772388a0a4136c056c585961f77938d2dc77bf526e7550a5ef3.jpg)

![d8fc91f99d303e4896432a714b54b6cf56d46e1cb3aacfa81ecca9c65c4da71b.jpg](../icml_results/3748_The%20Latent%20Color%20Subspace_%20Emergent%20Order%20in%20High-Dimensional%20Chaos/tables/d8fc91f99d303e4896432a714b54b6cf56d46e1cb3aacfa81ecca9c65c4da71b.jpg)

<a id="paper-3709"></a>

## Tail Annealing for Heavy-Tailed Flow Matching


### Tables

![0ec7695dfae388eec5932bf5b4c023673f6a510e207e252c664515231dde2002.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/0ec7695dfae388eec5932bf5b4c023673f6a510e207e252c664515231dde2002.jpg)

![3f5fd1ed9d92b77266b1c9a4d5bba9f137ebdb2ac13e0acf6a03a8191e993839.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/3f5fd1ed9d92b77266b1c9a4d5bba9f137ebdb2ac13e0acf6a03a8191e993839.jpg)

![4d0f2f68fd02930f9b571da7ee088ca3ea2579d992832256f1249d3d9934fc50.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/4d0f2f68fd02930f9b571da7ee088ca3ea2579d992832256f1249d3d9934fc50.jpg)

![5adf23497c7a0e5685e5443fa5c5b1f768b2034918f856b7e4445aa9ff0a7286.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/5adf23497c7a0e5685e5443fa5c5b1f768b2034918f856b7e4445aa9ff0a7286.jpg)

![5bc7fb38923c352f3f58a83094cf36a39f69210c8e0d8ddb62fb3c2b937d1612.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/5bc7fb38923c352f3f58a83094cf36a39f69210c8e0d8ddb62fb3c2b937d1612.jpg)

![61a6881476c42ddca29154689e4d6dc5e0abdbc6104ebf8c0f4c5e2525b9f6db.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/61a6881476c42ddca29154689e4d6dc5e0abdbc6104ebf8c0f4c5e2525b9f6db.jpg)

![7ac48b49744fb77c741db44e7540cf506defe337b111f6dcf3a0bfa4bb66695a.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/7ac48b49744fb77c741db44e7540cf506defe337b111f6dcf3a0bfa4bb66695a.jpg)

![837cee07738bb16f562efb2de2f120e93a774e2170a4962976dc5cd400d870ac.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/837cee07738bb16f562efb2de2f120e93a774e2170a4962976dc5cd400d870ac.jpg)

![8cd76700bc6d32be50bf1a7f8f7d7b30e99f5f2b57cd281bafe3a28d5304d85d.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/8cd76700bc6d32be50bf1a7f8f7d7b30e99f5f2b57cd281bafe3a28d5304d85d.jpg)

![93c65f265ddd0dc8c54356adb6a55dea32024ca689917f9bb11170bfbc31c314.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/93c65f265ddd0dc8c54356adb6a55dea32024ca689917f9bb11170bfbc31c314.jpg)

![acd30fe55aa2a6c0a0dce1a2fb8758db7b52daab037181e1f3e3975ef6d493d4.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/acd30fe55aa2a6c0a0dce1a2fb8758db7b52daab037181e1f3e3975ef6d493d4.jpg)

![b08073238e9b52bb62b7447f3059aa6b6db7230c891e9113ab36fb0f07abf2d7.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/b08073238e9b52bb62b7447f3059aa6b6db7230c891e9113ab36fb0f07abf2d7.jpg)

![d169d5096ddd7834ba42b80a295beb4d167b12625f4133796803213a1ee6ad4e.jpg](../icml_results/3749_Tail%20Annealing%20for%20Heavy-Tailed%20Flow%20Matching/tables/d169d5096ddd7834ba42b80a295beb4d167b12625f4133796803213a1ee6ad4e.jpg)

<a id="paper-3710"></a>

## Allocating Variance to Maximize Expectation


### Images

![54db83d94354649f7cfe25398b31e991197fb96e073b452521f0df3c22bf38b3.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/images/54db83d94354649f7cfe25398b31e991197fb96e073b452521f0df3c22bf38b3.jpg)

![ec8b4dccecccbb3ec23fdd85490b19b899d563cdb111ad25ffdc9176e2262228.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/images/ec8b4dccecccbb3ec23fdd85490b19b899d563cdb111ad25ffdc9176e2262228.jpg)

### Tables

![0c288a129e32350e55eccd482cf8e168aa989f5bc33e83b8a7671ca2d24378bd.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/tables/0c288a129e32350e55eccd482cf8e168aa989f5bc33e83b8a7671ca2d24378bd.jpg)

![28018833d47a9cc6da311d8e9d54178f55791fafa0afc7ab4b955f0984846c37.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/tables/28018833d47a9cc6da311d8e9d54178f55791fafa0afc7ab4b955f0984846c37.jpg)

![663b0a414c1fa3e0e83e4613278a1c396e38cef03ba9c8136199f8fe0ed0120f.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/tables/663b0a414c1fa3e0e83e4613278a1c396e38cef03ba9c8136199f8fe0ed0120f.jpg)

![a27fbce89c57715d0b140ce7c028080d50b791793534cf86cf628b32221fd195.jpg](../icml_results/3750_Allocating%20Variance%20to%20Maximize%20Expectation/tables/a27fbce89c57715d0b140ce7c028080d50b791793534cf86cf628b32221fd195.jpg)

<a id="paper-3711"></a>

## Aggregate Models, Not Explanations: Improving Feature Importance Estimation


### Images

![1d47ffaf4d6e16eac03afc4b3ec39e5ef626c25b549132aad765cdfe97324c73.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/1d47ffaf4d6e16eac03afc4b3ec39e5ef626c25b549132aad765cdfe97324c73.jpg)

![3ba8e316a3048270995eb1b5405316d2743f939fc09ea070bac968de56cd8a45.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/3ba8e316a3048270995eb1b5405316d2743f939fc09ea070bac968de56cd8a45.jpg)

![4469d6aabb827e6d240cd01d6cb1814e9acb83a5f770aa8bbaf1edcd7807efc7.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/4469d6aabb827e6d240cd01d6cb1814e9acb83a5f770aa8bbaf1edcd7807efc7.jpg)

![461fceba559b262f1185faf4d006acf5ad0a414974e4fafd23f9d39e352506ad.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/461fceba559b262f1185faf4d006acf5ad0a414974e4fafd23f9d39e352506ad.jpg)

![525ac1f125accc6920fff7a746d3acae3ff49aff5092bdeb21b44e1daed10447.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/525ac1f125accc6920fff7a746d3acae3ff49aff5092bdeb21b44e1daed10447.jpg)

![539d3f7244f27bf03712b1fe4c6d94453335a411a1bafd628600bc13a4964a10.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/539d3f7244f27bf03712b1fe4c6d94453335a411a1bafd628600bc13a4964a10.jpg)

![635ee26f1b74495f290b3818ca0f8c9367c944b6c77ff5ce24165880221f4fed.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/635ee26f1b74495f290b3818ca0f8c9367c944b6c77ff5ce24165880221f4fed.jpg)

![77e4b78a97510c395d86a729b904e855ddbcbd2e78e55d33a6974d786d0bf146.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/77e4b78a97510c395d86a729b904e855ddbcbd2e78e55d33a6974d786d0bf146.jpg)

![7879e751bd428940fab3c76f0a13054b40fad228ff9857f66df3b5c31f2339c5.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/7879e751bd428940fab3c76f0a13054b40fad228ff9857f66df3b5c31f2339c5.jpg)

![873df8ac3e0bf2fc70b24fd03f33a53d5b4334a699df45380a2e5ca7212663a8.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/873df8ac3e0bf2fc70b24fd03f33a53d5b4334a699df45380a2e5ca7212663a8.jpg)

![890d4c809a627bd2d081248141026fd143ae26db19290a8330abfeedccd45720.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/890d4c809a627bd2d081248141026fd143ae26db19290a8330abfeedccd45720.jpg)

![a1649a52d106f692e8df856a6a860b7718f3133f507e738f61f9371eb3e1a3c8.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/a1649a52d106f692e8df856a6a860b7718f3133f507e738f61f9371eb3e1a3c8.jpg)

![bddc796bfc43da264343faada00a41fcf8185e81e9fa0e2a6e68edf91174282c.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/bddc796bfc43da264343faada00a41fcf8185e81e9fa0e2a6e68edf91174282c.jpg)

![cb6c8405903c52d4449b4ad8fc82c5075d20313012fc73339c721021d4e8f278.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/cb6c8405903c52d4449b4ad8fc82c5075d20313012fc73339c721021d4e8f278.jpg)

![cf15b77d2019c07ec7513c4d510fe6a6ea0150dfcdbb3fdfb92f373ba10a9db0.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/cf15b77d2019c07ec7513c4d510fe6a6ea0150dfcdbb3fdfb92f373ba10a9db0.jpg)

![cf4025d1fb2ff908c2c3dd04eae8fe80e6ea4380beb1b4f5fded03fccfd81e2b.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/cf4025d1fb2ff908c2c3dd04eae8fe80e6ea4380beb1b4f5fded03fccfd81e2b.jpg)

![d997daad213618b761acce076d5a86c4da71cdf28213445b1e3f79d447d621a7.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/d997daad213618b761acce076d5a86c4da71cdf28213445b1e3f79d447d621a7.jpg)

![dbfb4b8d7d5371bbd54f93eba29dad3b856bfcb68294d01ec900c32240faa0cd.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/dbfb4b8d7d5371bbd54f93eba29dad3b856bfcb68294d01ec900c32240faa0cd.jpg)

![f0079a14e92936a88646432597682abbf50c7ba5ad5153ef287ad319e5bdef5f.jpg](../icml_results/3751_Aggregate%20Models%2C%20Not%20Explanations_%20Improving%20Feature%20Importance%20Estimation/images/f0079a14e92936a88646432597682abbf50c7ba5ad5153ef287ad319e5bdef5f.jpg)

<a id="paper-3712"></a>

## Revisiting the Volume Hypothesis


### Images

![06db1afb9f3d656486ce74925151f66dc5ade6390c86b94d14d7570a9327b676.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/images/06db1afb9f3d656486ce74925151f66dc5ade6390c86b94d14d7570a9327b676.jpg)

![4b213c1be02e0b63be594880d23e8f8af31ae45a9ac61f086f09912b31333777.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/images/4b213c1be02e0b63be594880d23e8f8af31ae45a9ac61f086f09912b31333777.jpg)

![57a4ade3fef8b5fe73992b885b4c6a2c851eabbafb60739d1feb4b5a31094adf.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/images/57a4ade3fef8b5fe73992b885b4c6a2c851eabbafb60739d1feb4b5a31094adf.jpg)

![7053d12dc38b683d79b835a021211e9b68f67c4a9b8aa4d038ef7b5f2d1ee8e9.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/images/7053d12dc38b683d79b835a021211e9b68f67c4a9b8aa4d038ef7b5f2d1ee8e9.jpg)

![fbc24a61c4b0ef7c0fb8e20a50b32cc3e49bb8e9df169f7be94859bc191286b4.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/images/fbc24a61c4b0ef7c0fb8e20a50b32cc3e49bb8e9df169f7be94859bc191286b4.jpg)

### Tables

![0efd08eebe32efb34222d37c4e4f278df80652596ac9a41719602535392d7c2e.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/0efd08eebe32efb34222d37c4e4f278df80652596ac9a41719602535392d7c2e.jpg)

![3586064b0eaad84f1f38db64eeb7347e3cca78742136197c0ea64f66a13419e5.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/3586064b0eaad84f1f38db64eeb7347e3cca78742136197c0ea64f66a13419e5.jpg)

![4a12dff496263f4ef2a84748ae4f159a9a402fbba65d996c191f2e219be23c1e.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/4a12dff496263f4ef2a84748ae4f159a9a402fbba65d996c191f2e219be23c1e.jpg)

![50774b60123458b4ae4f9fa28ecc4ddf081370323c0063cfa48cc72c7e6c3b48.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/50774b60123458b4ae4f9fa28ecc4ddf081370323c0063cfa48cc72c7e6c3b48.jpg)

![946b5dcb3bc6cf728e2ab7aa1d70bf1b56cdffd276cb5790309b47695a8636cd.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/946b5dcb3bc6cf728e2ab7aa1d70bf1b56cdffd276cb5790309b47695a8636cd.jpg)

![9b2d5aed1f91046c96732a8d27d629383ffa3007b7a922b76a0df1ecc7aa51b4.jpg](../icml_results/3752_Revisiting%20the%20Volume%20Hypothesis/tables/9b2d5aed1f91046c96732a8d27d629383ffa3007b7a922b76a0df1ecc7aa51b4.jpg)

<a id="paper-3713"></a>

## DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers


### Images

![06bd2d0cc4c9edd67392d65d1989e017371f09feda09294eabb117f24054993f.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/images/06bd2d0cc4c9edd67392d65d1989e017371f09feda09294eabb117f24054993f.jpg)

![1103160d6dc7868c3e81945612b55ddb67845b1339127febef694c56f9219264.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/images/1103160d6dc7868c3e81945612b55ddb67845b1339127febef694c56f9219264.jpg)

![9072e60fd0760bb9c9079b9b74a948ec9115b3a645827b13075dee95ccac3ff0.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/images/9072e60fd0760bb9c9079b9b74a948ec9115b3a645827b13075dee95ccac3ff0.jpg)

![b8eec3ee14ac34e8bb0330aacbd6d8c8f37a5d2e51a861a785c555d22cc1eb55.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/images/b8eec3ee14ac34e8bb0330aacbd6d8c8f37a5d2e51a861a785c555d22cc1eb55.jpg)

![da0b94a759ccd7f8461c91f223106ffbcd3257df90a92faeb6b622fd615fd231.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/images/da0b94a759ccd7f8461c91f223106ffbcd3257df90a92faeb6b622fd615fd231.jpg)

### Tables

![100df96a80576355dd9f1581538e17eb83d333653ad568cbac27d330da552162.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/100df96a80576355dd9f1581538e17eb83d333653ad568cbac27d330da552162.jpg)

![18ee78bdc5eb4d89d44aba118f96a8e0b5289c9d1a8bfcb0eb15bbd0e784a11f.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/18ee78bdc5eb4d89d44aba118f96a8e0b5289c9d1a8bfcb0eb15bbd0e784a11f.jpg)

![1933294d421751de59dcdca68dc9eea3e465854e299379d500d073609981c0d8.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/1933294d421751de59dcdca68dc9eea3e465854e299379d500d073609981c0d8.jpg)

![2c3f12dfef99993f07b78c1c18f4a2504af1f538c3ff2c2046ebbb7311b70dfb.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/2c3f12dfef99993f07b78c1c18f4a2504af1f538c3ff2c2046ebbb7311b70dfb.jpg)

![2d0eb7b3e11df0d09b2c5dc77d202f71e8cd24b540d914c2ebb21a24e0a27ce5.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/2d0eb7b3e11df0d09b2c5dc77d202f71e8cd24b540d914c2ebb21a24e0a27ce5.jpg)

![3f749d5f62c1f0d74e19291d44ec35aab90cc6f7151928ddb505f9cd7d990e00.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/3f749d5f62c1f0d74e19291d44ec35aab90cc6f7151928ddb505f9cd7d990e00.jpg)

![4efd15e8d91fd4c6900889c65bb5c82a2f4a4be9fefd77b83a7cfe761cbaad91.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/4efd15e8d91fd4c6900889c65bb5c82a2f4a4be9fefd77b83a7cfe761cbaad91.jpg)

![8146f1a5135e32cc9636955fe25a6894a221646e906f2be7a2ea8865bd49c919.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/8146f1a5135e32cc9636955fe25a6894a221646e906f2be7a2ea8865bd49c919.jpg)

![8c81f0cfd3429ec69c1613f56817e458b95dce24ca3c7b2fbaa27b6de6762938.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/8c81f0cfd3429ec69c1613f56817e458b95dce24ca3c7b2fbaa27b6de6762938.jpg)

![a7adfb1c6ab3ca92afcd6fbea6558be8a0059089724af6a44d37a86b5ad42487.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/a7adfb1c6ab3ca92afcd6fbea6558be8a0059089724af6a44d37a86b5ad42487.jpg)

![d3b60a206aa4a648a5e7ff2a489cacab652e1bc2d7a4dc8d020ce3a982dc7ae9.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/d3b60a206aa4a648a5e7ff2a489cacab652e1bc2d7a4dc8d020ce3a982dc7ae9.jpg)

![e7b26849cb5db41e37ee94bddae6b67a6355d2aaa17fe744826c6ecca0c338ff.jpg](../icml_results/3753_DisjunctiveNet_%20Neural%20Symbolic%20Learning%20via%20Differentiable%20Convexified%20Optimization%20Layers/tables/e7b26849cb5db41e37ee94bddae6b67a6355d2aaa17fe744826c6ecca0c338ff.jpg)

<a id="paper-3714"></a>

## Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality


### Images

![137a0ffb67a3b259ca3d9d784e2de37657836b93748a807ee93c9cc2f50e2cd8.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/137a0ffb67a3b259ca3d9d784e2de37657836b93748a807ee93c9cc2f50e2cd8.jpg)

![152003e3c63e1707d80813d4c7e964ac8d9f8261a44e94a2a8e45b1c53502fbc.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/152003e3c63e1707d80813d4c7e964ac8d9f8261a44e94a2a8e45b1c53502fbc.jpg)

![1708cd9b7a446c38046d4598a74a89753f7116f24a57307b9db7ddc888723efd.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/1708cd9b7a446c38046d4598a74a89753f7116f24a57307b9db7ddc888723efd.jpg)

![1b238155956e5d5b26614ea19cc9a8392e4fb60bd8c2e920434862b9c732eefd.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/1b238155956e5d5b26614ea19cc9a8392e4fb60bd8c2e920434862b9c732eefd.jpg)

![23a2ef1b52a0bfd223c73c6c9e3591fc08e01fecd376f03782519578fe983601.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/23a2ef1b52a0bfd223c73c6c9e3591fc08e01fecd376f03782519578fe983601.jpg)

![3bdc8e1d3d064456047531d3a762257bcd067adaceedcf4fcc678d5754d34488.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/3bdc8e1d3d064456047531d3a762257bcd067adaceedcf4fcc678d5754d34488.jpg)

![786c49418faf523307fafa4401f2d4f551615c41c361a43081cbd1145bc2ad09.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/786c49418faf523307fafa4401f2d4f551615c41c361a43081cbd1145bc2ad09.jpg)

![78f68c208698b922acf0f6f9fe0f664b62232bcdd39047032d9c13c722f3b324.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/78f68c208698b922acf0f6f9fe0f664b62232bcdd39047032d9c13c722f3b324.jpg)

![8c6077d3c94090d69942acaae51211425741dde971c9f0981bbcb13fd9f185aa.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/8c6077d3c94090d69942acaae51211425741dde971c9f0981bbcb13fd9f185aa.jpg)

![8fe9d8e8198101291f21b0ecf8f4c94b7dada95e2bf418e4a18794f915f31e9a.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/8fe9d8e8198101291f21b0ecf8f4c94b7dada95e2bf418e4a18794f915f31e9a.jpg)

![d490553ae979d0934a8462454358f6d4ecfd718cf2754735c64979fa2ae00ef2.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/d490553ae979d0934a8462454358f6d4ecfd718cf2754735c64979fa2ae00ef2.jpg)

![fbbe4b98a1c8b4bac8a4d2f45591a6b8c50e1ee4518251654103be5f540d40eb.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/images/fbbe4b98a1c8b4bac8a4d2f45591a6b8c50e1ee4518251654103be5f540d40eb.jpg)

### Tables

![4f8c3cbd1838e732596e73aec07d7e74f1704f1475afce9cbb65b3536a04ae09.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/tables/4f8c3cbd1838e732596e73aec07d7e74f1704f1475afce9cbb65b3536a04ae09.jpg)

![63dcce56a1538151117129faa3c6ff1499fcc71a2f17d0d02591ca7359828942.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/tables/63dcce56a1538151117129faa3c6ff1499fcc71a2f17d0d02591ca7359828942.jpg)

![86b65eb8705d6e07581eceb4d9d6bb6e1583eccce48c23df50089c429e4eb0e8.jpg](../icml_results/3754_Reinforcement%20Learning%20for%20Reachability_%20Guaranteeing%20Asymptotic%20Optimality/tables/86b65eb8705d6e07581eceb4d9d6bb6e1583eccce48c23df50089c429e4eb0e8.jpg)

<a id="paper-3715"></a>

## Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models


### Images

![010d1ac5bad25e085857b3f008a0d97fec1e30281006f8e6341980556390ad2a.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/010d1ac5bad25e085857b3f008a0d97fec1e30281006f8e6341980556390ad2a.jpg)

![04acb9350755148cdb05c570ba2edcc4fc4e12dcbaa962e7b29c6235a4851319.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/04acb9350755148cdb05c570ba2edcc4fc4e12dcbaa962e7b29c6235a4851319.jpg)

![0d525835bf995301c19a98c06f11c4119959fbf0f24c471fc1a952516978c4e8.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/0d525835bf995301c19a98c06f11c4119959fbf0f24c471fc1a952516978c4e8.jpg)

![228926cc5005cf4a990b290bb0971379981c51f811b5cba625981a07b5138cbd.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/228926cc5005cf4a990b290bb0971379981c51f811b5cba625981a07b5138cbd.jpg)

![25a0f9309191f9ca40623c201e256260ac7749e412c54b694f44e32e85b017e1.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/25a0f9309191f9ca40623c201e256260ac7749e412c54b694f44e32e85b017e1.jpg)

![26cdd6803385b457cd1fc1d980b7ee21f6bb8b499b705dbf60e337eba4bf268e.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/26cdd6803385b457cd1fc1d980b7ee21f6bb8b499b705dbf60e337eba4bf268e.jpg)

![42696fed54fe91d96dc7ee02d0cef30dbf8f75e4d1f268dacaf0cad222b40617.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/42696fed54fe91d96dc7ee02d0cef30dbf8f75e4d1f268dacaf0cad222b40617.jpg)

![62b03b3bcbf4fbf76cc7af0a37fc79c10d3433b8fbe14da92a470d7fd1fc1ff6.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/62b03b3bcbf4fbf76cc7af0a37fc79c10d3433b8fbe14da92a470d7fd1fc1ff6.jpg)

![7d5496762753c5750e24bc5041bc8e3ea40f9cba86d1e9bfe69d857c00ad6dd4.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/7d5496762753c5750e24bc5041bc8e3ea40f9cba86d1e9bfe69d857c00ad6dd4.jpg)

![8159dd3e63040b2d570d685821f94cd4b3c9c1566f34a2fa473e21b1f55cdc96.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/8159dd3e63040b2d570d685821f94cd4b3c9c1566f34a2fa473e21b1f55cdc96.jpg)

![8c89ab64b0832a0b5bbef6b6ba310de51be6ccbbb463a9486464cdfa5148d22c.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/8c89ab64b0832a0b5bbef6b6ba310de51be6ccbbb463a9486464cdfa5148d22c.jpg)

![97adf9385557eae1628808113be05309d43fe3c26f71fdbd946c1b0ba1d6eeea.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/97adf9385557eae1628808113be05309d43fe3c26f71fdbd946c1b0ba1d6eeea.jpg)

![9d70bb86a29345a0a7db61f04880987a19b59eb7eae476293618ccb71c875793.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/9d70bb86a29345a0a7db61f04880987a19b59eb7eae476293618ccb71c875793.jpg)

![b747c19f9092309d4d57e6ca24060c197bf6635b49c96dccc740c5e15ced05e4.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/b747c19f9092309d4d57e6ca24060c197bf6635b49c96dccc740c5e15ced05e4.jpg)

![b8f9dc3e5d268ac079e0f3675b4d0c0926dd476bc73c6e0fcde9d4b068239e39.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/b8f9dc3e5d268ac079e0f3675b4d0c0926dd476bc73c6e0fcde9d4b068239e39.jpg)

![bde5e71fe6ba34ce042252f9e4afc97b68fd8ea882f048109af3ab955a322c26.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/bde5e71fe6ba34ce042252f9e4afc97b68fd8ea882f048109af3ab955a322c26.jpg)

![d74363cc544c480913fa608fc3f51db181f3c452e5709e39ef3c0f067a600508.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/d74363cc544c480913fa608fc3f51db181f3c452e5709e39ef3c0f067a600508.jpg)

![eb067d7f5017d9f380428d4f8391f516d4ded3c9c9d0ffedefb05bef2876609d.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/images/eb067d7f5017d9f380428d4f8391f516d4ded3c9c9d0ffedefb05bef2876609d.jpg)

### Tables

![175e38120884d1e1dab3359b39fd47fa9cb0592f9d319380d6f7c172a580bc3c.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/175e38120884d1e1dab3359b39fd47fa9cb0592f9d319380d6f7c172a580bc3c.jpg)

![1e8c076c571274dc15570057bfe2537c9315037cb82c6638a184c5e5c1236471.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/1e8c076c571274dc15570057bfe2537c9315037cb82c6638a184c5e5c1236471.jpg)

![20663487ef283676c221d88f00417fb5816551294a6df6bd6a83d8b24ab8492e.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/20663487ef283676c221d88f00417fb5816551294a6df6bd6a83d8b24ab8492e.jpg)

![23f0fd00380347a02affff9dffbc00d0fa993eda32cc22969cc46c159bba3fd0.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/23f0fd00380347a02affff9dffbc00d0fa993eda32cc22969cc46c159bba3fd0.jpg)

![263c03d5401ec6cd66536687fcf89d7b0fe7117a408952ff6d1b2dec7f32a213.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/263c03d5401ec6cd66536687fcf89d7b0fe7117a408952ff6d1b2dec7f32a213.jpg)

![3293ce4260c7e64b868ec4727f460ad7d6f8c83cb877363c7387dfe284b4d625.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/3293ce4260c7e64b868ec4727f460ad7d6f8c83cb877363c7387dfe284b4d625.jpg)

![3784c46637cec21a82ec24c9c3183d2845f9b60020e8bba1c874989f689e9ad0.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/3784c46637cec21a82ec24c9c3183d2845f9b60020e8bba1c874989f689e9ad0.jpg)

![38b2f11d05ed47c17f1bc8bf827169b507fb752edd7236d9802726fbd25f42ba.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/38b2f11d05ed47c17f1bc8bf827169b507fb752edd7236d9802726fbd25f42ba.jpg)

![3932d0da67771c9e76ba074fc13ac15b4929a028cf0f7ef715b3650ada153b54.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/3932d0da67771c9e76ba074fc13ac15b4929a028cf0f7ef715b3650ada153b54.jpg)

![3ea6c23df91cb7dc69c843fb5d0263a9501afc8695cb0d1cb25efc88ea4ab3ab.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/3ea6c23df91cb7dc69c843fb5d0263a9501afc8695cb0d1cb25efc88ea4ab3ab.jpg)

![3fc9dd3d55dc1ae8da800a3843cb9826dca2273691a8d056166d68221125764e.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/3fc9dd3d55dc1ae8da800a3843cb9826dca2273691a8d056166d68221125764e.jpg)

![4012ad1044b530aebe8b104c033de6758e383614e18ca19c5fe6817e0c620515.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/4012ad1044b530aebe8b104c033de6758e383614e18ca19c5fe6817e0c620515.jpg)

![5a0e1535113c4f2b9c849af58bd5a4c26935b932ae5c4a492e35f7fcbb6afb02.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/5a0e1535113c4f2b9c849af58bd5a4c26935b932ae5c4a492e35f7fcbb6afb02.jpg)

![5c2ae7b4bfa3f930783f9631fea82a9b2b77a1b193d8b131938bf414795bebe2.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/5c2ae7b4bfa3f930783f9631fea82a9b2b77a1b193d8b131938bf414795bebe2.jpg)

![6405d8c22a00fa17931782a5271eb348b134c3a0a11dfcd883192dd73795b176.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/6405d8c22a00fa17931782a5271eb348b134c3a0a11dfcd883192dd73795b176.jpg)

![6cd293e5786f4586b44403613424ef57a9145d8e7473a023ad17417a70d2d8d8.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/6cd293e5786f4586b44403613424ef57a9145d8e7473a023ad17417a70d2d8d8.jpg)

![825eaf9e00939a62a139fb9f5226b176311064ca5b4d47be88ef19a86555205a.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/825eaf9e00939a62a139fb9f5226b176311064ca5b4d47be88ef19a86555205a.jpg)

![87116ba55c0f2a95ddbcc6dc9c078247946c6d8c9a10aa286f5ae3764ac55ee2.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/87116ba55c0f2a95ddbcc6dc9c078247946c6d8c9a10aa286f5ae3764ac55ee2.jpg)

![88b37282828caec25179ebfc7dcaf5ee5c12db41b43df1c46fdd7361425f86cd.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/88b37282828caec25179ebfc7dcaf5ee5c12db41b43df1c46fdd7361425f86cd.jpg)

![8a0da522964f2aa9bee48a12eb18713236d47b8e799f3236690246ff5f9cdd18.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/8a0da522964f2aa9bee48a12eb18713236d47b8e799f3236690246ff5f9cdd18.jpg)

![a25e7bbde6f7841be314b64fedb8c7b1695c5f8a5ced202138626e07b2d31bc1.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/a25e7bbde6f7841be314b64fedb8c7b1695c5f8a5ced202138626e07b2d31bc1.jpg)

![a61ea704897aff14a8631d0a407192b0117d9f243b29dddaf865289138eed73a.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/a61ea704897aff14a8631d0a407192b0117d9f243b29dddaf865289138eed73a.jpg)

![d175319dd0a53dfc700894b1f6ad2e2321c5dc5d5a99983cf1bda31cc56f37a1.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/d175319dd0a53dfc700894b1f6ad2e2321c5dc5d5a99983cf1bda31cc56f37a1.jpg)

![e1806f1723cbd421dd703fdf04989c15989e07f0cd672c1858da4de05da877a3.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/e1806f1723cbd421dd703fdf04989c15989e07f0cd672c1858da4de05da877a3.jpg)

![f09e2e8ff3dd6e734b8dc2c010646b115e72da9396277fd414b3c15a1c6148f2.jpg](../icml_results/3755_Scalable%20Single-Cell%20Gene%20Expression%20Generation%20with%20Latent%20Diffusion%20Models/tables/f09e2e8ff3dd6e734b8dc2c010646b115e72da9396277fd414b3c15a1c6148f2.jpg)

<a id="paper-3716"></a>

## Benchmarking and Improving Fine-Grained Text-to-Image Alignment via Paired Reinforcement Learning


### Images

![14badb3b55cb3de38a99fca5dfd6cad6106185efc59ccf875d140b018670daa6.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/14badb3b55cb3de38a99fca5dfd6cad6106185efc59ccf875d140b018670daa6.jpg)

![2303540b97389d1ac8e9fb8808e293df62bde14147910200839548bb5d216911.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/2303540b97389d1ac8e9fb8808e293df62bde14147910200839548bb5d216911.jpg)

![2b05f6518a2d1704d5c0d9c4ecaed6bb4234d4c5e90012960218bbde6472bef6.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/2b05f6518a2d1704d5c0d9c4ecaed6bb4234d4c5e90012960218bbde6472bef6.jpg)

![4044015e0a1ef492c760fba8430002011329d71aacef8bca4f24a585d18bbb89.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/4044015e0a1ef492c760fba8430002011329d71aacef8bca4f24a585d18bbb89.jpg)

![43463ca42ba755d2e8cb43c234b3b710a52ad8aaa7f7b8c0e85d0394665c458e.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/43463ca42ba755d2e8cb43c234b3b710a52ad8aaa7f7b8c0e85d0394665c458e.jpg)

![542cbebb297e980b0749fee9963cc610fa2f872321bf2eff984905578d6cdc13.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/542cbebb297e980b0749fee9963cc610fa2f872321bf2eff984905578d6cdc13.jpg)

![7ec1611036d1d901ab49f36346aa0c5adf843c9d453bd00d4a41b90e14db5d86.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/7ec1611036d1d901ab49f36346aa0c5adf843c9d453bd00d4a41b90e14db5d86.jpg)

![b12b32aa4914e82a7fadad5189537b4f4708e303d25879f4cd994e50e5476303.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/b12b32aa4914e82a7fadad5189537b4f4708e303d25879f4cd994e50e5476303.jpg)

![c96aaa4c0fa488f8d4e62ae6d047a2d05fa4abce725d7d71a0e968702e63acbe.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/c96aaa4c0fa488f8d4e62ae6d047a2d05fa4abce725d7d71a0e968702e63acbe.jpg)

![edce824e6a611557e4d0fa8c3b47fbf279bd373d59f2119192f21dc3883c9770.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/images/edce824e6a611557e4d0fa8c3b47fbf279bd373d59f2119192f21dc3883c9770.jpg)

### Tables

![74d512879166de8799bdc83ef4b7ab9170be6063da3e2cf628ae4076a4398812.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/tables/74d512879166de8799bdc83ef4b7ab9170be6063da3e2cf628ae4076a4398812.jpg)

![d8fc3d0b1acd88ae5a3d70100b9cdf0af306014520aa1f170553f23cde6342c7.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/tables/d8fc3d0b1acd88ae5a3d70100b9cdf0af306014520aa1f170553f23cde6342c7.jpg)

![eb8a37ae7b7ef837837231f6fe106cdc1e2ec943350d504f41e80a197768b0da.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/tables/eb8a37ae7b7ef837837231f6fe106cdc1e2ec943350d504f41e80a197768b0da.jpg)

![f9b14aa1e3e4787a90fab4993329b9dde9c2aefe4ff181ade909e9c4fcfae15b.jpg](../icml_results/3756_Benchmarking%20and%20Improving%20Fine-Grained%20Text-to-Image%20Alignment%20via%20Paired%20Reinforcement%20Learning/tables/f9b14aa1e3e4787a90fab4993329b9dde9c2aefe4ff181ade909e9c4fcfae15b.jpg)

<a id="paper-3717"></a>

## Adaptive Multi-Round Allocation with Stochastic Arrivals


### Images

![1149ec1a3c4ded2f341fbd342be91db78718bea8609cc14d228f688b1496a3ad.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/1149ec1a3c4ded2f341fbd342be91db78718bea8609cc14d228f688b1496a3ad.jpg)

![16999fcf32ecef7d0f1ca4ce9edd67023b9e16a6fc48a4fb3b091afd16d69bd6.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/16999fcf32ecef7d0f1ca4ce9edd67023b9e16a6fc48a4fb3b091afd16d69bd6.jpg)

![1f3e3175c373bf990a9f550f210a1be06bfd97ec67e104173e14ef9395a38d5e.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/1f3e3175c373bf990a9f550f210a1be06bfd97ec67e104173e14ef9395a38d5e.jpg)

![2698590ffbb54b0edf26f2635b35db0171143047369ce4117ceac584a27d15e4.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/2698590ffbb54b0edf26f2635b35db0171143047369ce4117ceac584a27d15e4.jpg)

![277a71bccb9cf6330f6040a607fdbba51e49a5fabebb1ece4c0dddd1b5332466.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/277a71bccb9cf6330f6040a607fdbba51e49a5fabebb1ece4c0dddd1b5332466.jpg)

![3b803a1b49094c3b7d44bb5366292f06aa7027c00021d8225814c182777d7d6e.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/3b803a1b49094c3b7d44bb5366292f06aa7027c00021d8225814c182777d7d6e.jpg)

![42123ba4a6483966ff5b3194f3b7478e7abfc38d93d030676ba507d1c59d10b1.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/42123ba4a6483966ff5b3194f3b7478e7abfc38d93d030676ba507d1c59d10b1.jpg)

![60f1a40a99c18fee03d7e18ba0056566b86d844ef4827a5e6b878a398eb0b990.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/60f1a40a99c18fee03d7e18ba0056566b86d844ef4827a5e6b878a398eb0b990.jpg)

![8ccb27eeed8d70fe7d367ceb01eec4249804817d51312abc9d53d8b7e04bce62.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/8ccb27eeed8d70fe7d367ceb01eec4249804817d51312abc9d53d8b7e04bce62.jpg)

![943170a7212b32f9f9eed00ce9efda760009eb9a29efa92933205530f1ed1f81.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/943170a7212b32f9f9eed00ce9efda760009eb9a29efa92933205530f1ed1f81.jpg)

![98afc23c5c8a4133a0b8ef338e7c364f87f6a939df10e35640f0d2e334186e27.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/98afc23c5c8a4133a0b8ef338e7c364f87f6a939df10e35640f0d2e334186e27.jpg)

![dbecb4de9798a1b43a28eb41c6eacf3efcee3a9fc167097ed34ce11e63336749.jpg](../icml_results/3757_Adaptive%20Multi-Round%20Allocation%20with%20Stochastic%20Arrivals/images/dbecb4de9798a1b43a28eb41c6eacf3efcee3a9fc167097ed34ce11e63336749.jpg)

<a id="paper-3718"></a>

## Difference-Aware Decision Learning for Multimodal Image Fusion


### Images

![203c65a9d4844ddb5777e9cdc356a2d6784a3ef75edf170479fa848e55c115b1.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/203c65a9d4844ddb5777e9cdc356a2d6784a3ef75edf170479fa848e55c115b1.jpg)

![8566643b37bb5b6c7dc73ec0985b4cb353dafa1ef297d0d6e01f5de8a4e3343c.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/8566643b37bb5b6c7dc73ec0985b4cb353dafa1ef297d0d6e01f5de8a4e3343c.jpg)

![8f9bd8f670f1dab8ea0047a64746f2674ef183575d171fdc23605010328f5a69.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/8f9bd8f670f1dab8ea0047a64746f2674ef183575d171fdc23605010328f5a69.jpg)

![9adf39c8d1d35b5ebc2a0b9934a6d14aed16a292961be1ba617725b4c288c688.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/9adf39c8d1d35b5ebc2a0b9934a6d14aed16a292961be1ba617725b4c288c688.jpg)

![b90a0340501792736f1cfecbb20632e6d36eed0b7491d570b4622040167be0c3.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/b90a0340501792736f1cfecbb20632e6d36eed0b7491d570b4622040167be0c3.jpg)

![be02e7cd8eafd3b4c99e51d1371cd45ef6572393086a57b7fa1ad417e4582c7a.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/be02e7cd8eafd3b4c99e51d1371cd45ef6572393086a57b7fa1ad417e4582c7a.jpg)

![cca0efa23c1d728d00027ec75768129caf8a2cb7fd00cbadd3d6621f5e69b8ad.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/images/cca0efa23c1d728d00027ec75768129caf8a2cb7fd00cbadd3d6621f5e69b8ad.jpg)

### Tables

![1552bd29573fc3c570ffc42de01f110f798e713f89e7e2c60e019799e56bce49.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/1552bd29573fc3c570ffc42de01f110f798e713f89e7e2c60e019799e56bce49.jpg)

![240f3f3e8a2b94c5d6bbf46cd495ea43b40a7c067094700c23247f8495bfde50.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/240f3f3e8a2b94c5d6bbf46cd495ea43b40a7c067094700c23247f8495bfde50.jpg)

![81f80e29c0d48b75beffd3289542ce52d0f6529b3b75f27345c82cda3d9e3857.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/81f80e29c0d48b75beffd3289542ce52d0f6529b3b75f27345c82cda3d9e3857.jpg)

![a02b9be87900031c67f3388b7cdfca6e743c4889f6efe6f68c943e64a49551fd.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/a02b9be87900031c67f3388b7cdfca6e743c4889f6efe6f68c943e64a49551fd.jpg)

![c2ae6cefb4d0fb1a11a59fd02ac322dcf57c3a05762b3dcd68125f9cd6069c0b.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/c2ae6cefb4d0fb1a11a59fd02ac322dcf57c3a05762b3dcd68125f9cd6069c0b.jpg)

![c2ed180bab649ff449da9d839e02140f53a9f5400c5cfee5bc6560d239c00447.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/c2ed180bab649ff449da9d839e02140f53a9f5400c5cfee5bc6560d239c00447.jpg)

![e21f9cfb51ff2130f7dd1d3568abc8304ac8e2581d9689ac13f7f0d614e1af38.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/e21f9cfb51ff2130f7dd1d3568abc8304ac8e2581d9689ac13f7f0d614e1af38.jpg)

![f24e4199c88efadb4b1a993f89b6b508550e95591202c6365b311f7caaa14797.jpg](../icml_results/3758_Difference-Aware%20Decision%20Learning%20for%20Multimodal%20Image%20Fusion/tables/f24e4199c88efadb4b1a993f89b6b508550e95591202c6365b311f7caaa14797.jpg)

<a id="paper-3719"></a>

## Trajectory-Level Speculative Decoding for Diffusion Language Models


### Images

![1795f0013d238406f3f0fda6fce5959056f66636b9e3169eb3796f24e9d59991.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/1795f0013d238406f3f0fda6fce5959056f66636b9e3169eb3796f24e9d59991.jpg)

![55a6a1918aa2331f249ad0f014753249a907ee4588ea3345e27809afd935fc6e.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/55a6a1918aa2331f249ad0f014753249a907ee4588ea3345e27809afd935fc6e.jpg)

![6cc8470b2d4847c66b7f9e67d3a16b4a361616467fa43c43141ee351a56ee01c.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/6cc8470b2d4847c66b7f9e67d3a16b4a361616467fa43c43141ee351a56ee01c.jpg)

![92405b1684c5655a3759bb91f7e4f253fc6e9f1ac54ce25ea9e2bd850287ca7d.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/92405b1684c5655a3759bb91f7e4f253fc6e9f1ac54ce25ea9e2bd850287ca7d.jpg)

![9e93aef1ac99f9f2554bbda0830886609cb921d7d77c47df210a22fefec87af0.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/9e93aef1ac99f9f2554bbda0830886609cb921d7d77c47df210a22fefec87af0.jpg)

![ac9c76c0fb2b1d7d780649b722c95051ba9ad6e73da76ce76042d0417928f87d.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/images/ac9c76c0fb2b1d7d780649b722c95051ba9ad6e73da76ce76042d0417928f87d.jpg)

### Tables

![027ad64bc0b591ff4e24bc18b421320ebcd8ac733bdb0a81fd95bb122c771ad2.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/027ad64bc0b591ff4e24bc18b421320ebcd8ac733bdb0a81fd95bb122c771ad2.jpg)

![053c4acf128b0bc635c36c9c685e2786de3e0fe18c6c6fd507725bcf9751e23f.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/053c4acf128b0bc635c36c9c685e2786de3e0fe18c6c6fd507725bcf9751e23f.jpg)

![0b60f425b9e2db7d7142a2dafa896fa7f833cc381669131127b1a78affc9225a.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/0b60f425b9e2db7d7142a2dafa896fa7f833cc381669131127b1a78affc9225a.jpg)

![0fc4ad2a50dd8e1c1b3b03e2bd737627cab7f6a80011c4683f32dddbbf5027ca.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/0fc4ad2a50dd8e1c1b3b03e2bd737627cab7f6a80011c4683f32dddbbf5027ca.jpg)

![1334b4df5a4802ef0420e17ad5d8bc39f499f165bd29cb9f00d21e423bdafdc6.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/1334b4df5a4802ef0420e17ad5d8bc39f499f165bd29cb9f00d21e423bdafdc6.jpg)

![3d8f42221aa0d2bfd97ffe2b69e5572e4ea31b691e8ed0aed89bbeec4ad0c019.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/3d8f42221aa0d2bfd97ffe2b69e5572e4ea31b691e8ed0aed89bbeec4ad0c019.jpg)

![41f0b8272e8cc8ed7f2d55d53ec1288f0337cdeaf4ffb16abda3863d90bb4cc7.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/41f0b8272e8cc8ed7f2d55d53ec1288f0337cdeaf4ffb16abda3863d90bb4cc7.jpg)

![6500539200bbeedad4146586dcaa0c7c8007b6426bb38eaea58aee8e7f5406a9.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/6500539200bbeedad4146586dcaa0c7c8007b6426bb38eaea58aee8e7f5406a9.jpg)

![6764d74d7146da66058cb1909ad7e9d2745afd05e5423230ea7e24621bda23ab.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/6764d74d7146da66058cb1909ad7e9d2745afd05e5423230ea7e24621bda23ab.jpg)

![7904d3092895522f5fe6b8d7a8a0bc22217ed8c6576d64ed6d7f0c28736b44d7.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/7904d3092895522f5fe6b8d7a8a0bc22217ed8c6576d64ed6d7f0c28736b44d7.jpg)

![a854d5fd7e9375203db043c54e0a3bf1c377fa7d86eb79c943b28211e64aed95.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/a854d5fd7e9375203db043c54e0a3bf1c377fa7d86eb79c943b28211e64aed95.jpg)

![d4b8d41a3725bb8f8a21300332df808bcf8b1de1d405519d58c71e94cb9b8995.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/d4b8d41a3725bb8f8a21300332df808bcf8b1de1d405519d58c71e94cb9b8995.jpg)

![f3404b9fbe4fecfc36a9f974fdf9dc4e046796cd05c9743b10b96acf3af6ef44.jpg](../icml_results/3759_Trajectory-Level%20Speculative%20Decoding%20for%20Diffusion%20Language%20Models/tables/f3404b9fbe4fecfc36a9f974fdf9dc4e046796cd05c9743b10b96acf3af6ef44.jpg)

<a id="paper-3720"></a>

## ECCO: Evidence-Driven Causal Reasoning for Compiler Optimization


### Images

![29a71f110fce8668d30d2d4473b3ca497f652e336ac10bd7903b1fd2b2c1e5eb.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/images/29a71f110fce8668d30d2d4473b3ca497f652e336ac10bd7903b1fd2b2c1e5eb.jpg)

![818708b78969deb6fc119648f13700cb3e6fb47897875d4f62a3ae64904026c0.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/images/818708b78969deb6fc119648f13700cb3e6fb47897875d4f62a3ae64904026c0.jpg)

![909e358891cb6bd3c842c5d9f8234caea99eb1d0b3da2600a15125f42509ddcf.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/images/909e358891cb6bd3c842c5d9f8234caea99eb1d0b3da2600a15125f42509ddcf.jpg)

![d1a1bda7b79dd1b96738280550fae410b75782aa5eed19953995488eeefb3328.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/images/d1a1bda7b79dd1b96738280550fae410b75782aa5eed19953995488eeefb3328.jpg)

### Tables

![00f0654e87813df977bd6c83a46cd893a2f06d90e34709a0acd421e509d3bdfd.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/00f0654e87813df977bd6c83a46cd893a2f06d90e34709a0acd421e509d3bdfd.jpg)

![22d5a6a76ed919217164b97438503b1ffdca6bc4cdf80b38eb39e632bf6601e5.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/22d5a6a76ed919217164b97438503b1ffdca6bc4cdf80b38eb39e632bf6601e5.jpg)

![4d2cf1d38b2f3091eeab71ec17ae9d6463649171982d2987217c72aff19ae613.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/4d2cf1d38b2f3091eeab71ec17ae9d6463649171982d2987217c72aff19ae613.jpg)

![8dd8cfc279dd077b9a58789e50d86e621da98f76533c5ade21be4ca589811de1.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/8dd8cfc279dd077b9a58789e50d86e621da98f76533c5ade21be4ca589811de1.jpg)

![9394d3acaf28ad8cdf6cf8fcfa8a0b97bb27dc43ab2b0a093769f3cd30302297.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/9394d3acaf28ad8cdf6cf8fcfa8a0b97bb27dc43ab2b0a093769f3cd30302297.jpg)

![a45caddab4019f5d2779df5f229f143d408c343afba5aba460fa8ed75b5d3655.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/a45caddab4019f5d2779df5f229f143d408c343afba5aba460fa8ed75b5d3655.jpg)

![b9f5ff475748399ac2ce941b6b17e0eb27a79d92b87c6c728aad27f14ca56f5e.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/b9f5ff475748399ac2ce941b6b17e0eb27a79d92b87c6c728aad27f14ca56f5e.jpg)

![c397f67adb231fee7766abf73d889746ddb04b95af341f4c555b4972d423ed21.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/c397f67adb231fee7766abf73d889746ddb04b95af341f4c555b4972d423ed21.jpg)

![cad7d297894b3070972c537aaa90c9c1bddfd3b3aecf08a4ed425d684d37bdb9.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/cad7d297894b3070972c537aaa90c9c1bddfd3b3aecf08a4ed425d684d37bdb9.jpg)

![d66ef7e197f0e7292c2a53329873e4a2d40e0319dc0afa601d499f7474a6f155.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/d66ef7e197f0e7292c2a53329873e4a2d40e0319dc0afa601d499f7474a6f155.jpg)

![ddb260d5d1326d12caa147888b32e527d7d6bbde8a7072be5d2290727f4eea3a.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/ddb260d5d1326d12caa147888b32e527d7d6bbde8a7072be5d2290727f4eea3a.jpg)

![deabcaf0258680734aad5215d301496e0c9ba88012827ed961b5c79d63e1254d.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/deabcaf0258680734aad5215d301496e0c9ba88012827ed961b5c79d63e1254d.jpg)

![efa9eb66eebd872b63a202830340c020ef833fadde5e6b32f68a681dc9ffe3e8.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/efa9eb66eebd872b63a202830340c020ef833fadde5e6b32f68a681dc9ffe3e8.jpg)

![f7ad6deec1ccefd5dac19df96b1adeb0e2c23540c449d0dbdcbfa8b9aa322b09.jpg](../icml_results/3760_ECCO_%20Evidence-Driven%20Causal%20Reasoning%20for%20Compiler%20Optimization/tables/f7ad6deec1ccefd5dac19df96b1adeb0e2c23540c449d0dbdcbfa8b9aa322b09.jpg)

<a id="paper-3721"></a>

## Problem Distributions as Tasks: Repurposing Meta Learning for Generative Combinatorial Optimization towards Multi-task Pretraining and Adaptation


### Images

![4fbd373bbc1022d5af38a78372038fd05a2d407f034d97c749a9f61aaab8be90.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/images/4fbd373bbc1022d5af38a78372038fd05a2d407f034d97c749a9f61aaab8be90.jpg)

![4fbdda8aabd7f2aafaf91c04c2a64b7b561e1a14efe023efef37810f77b6b7f1.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/images/4fbdda8aabd7f2aafaf91c04c2a64b7b561e1a14efe023efef37810f77b6b7f1.jpg)

![b6f9ddfa31119d3b1c083259c3c1684ab650dfb0679e3320c99425ea04094185.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/images/b6f9ddfa31119d3b1c083259c3c1684ab650dfb0679e3320c99425ea04094185.jpg)

![e9e88579dcbd8438e82731a163521561d77dcea2a12927a6458e2c9389a38910.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/images/e9e88579dcbd8438e82731a163521561d77dcea2a12927a6458e2c9389a38910.jpg)

![f7e73aa8bd837cad852cda58c0c788f99677ab14c56a291e4f277b82b878b105.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/images/f7e73aa8bd837cad852cda58c0c788f99677ab14c56a291e4f277b82b878b105.jpg)

### Tables

![001bc07c36e73c75b3aa576222a5613a3b017022c755da308b7ae39814c4ef45.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/001bc07c36e73c75b3aa576222a5613a3b017022c755da308b7ae39814c4ef45.jpg)

![0a6406542d4eaf40b80ab61b7a31df5b997968ac42a0dd0b81f1a3417b5b3210.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/0a6406542d4eaf40b80ab61b7a31df5b997968ac42a0dd0b81f1a3417b5b3210.jpg)

![17a10f900f2e40706880ee6c97324419a3a40466e87d24efc8c2ff60d721bde7.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/17a10f900f2e40706880ee6c97324419a3a40466e87d24efc8c2ff60d721bde7.jpg)

![2972cbc93a90637b3b3cc74b4b7d78821293e1882da031ed9d72fb23c3cd4ef1.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/2972cbc93a90637b3b3cc74b4b7d78821293e1882da031ed9d72fb23c3cd4ef1.jpg)

![30163886f4d9e4a856957d329ae2e7229c946659a0e795b61aa1f3ee4770f666.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/30163886f4d9e4a856957d329ae2e7229c946659a0e795b61aa1f3ee4770f666.jpg)

![313b7e15b76cf726915b5f2e23aa85fbd0e3480f02117d2544de17b7d6579467.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/313b7e15b76cf726915b5f2e23aa85fbd0e3480f02117d2544de17b7d6579467.jpg)

![32497e1a45554ad2a923e275e16499f0fae9bc64522b3e50aeb4b44703e4118c.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/32497e1a45554ad2a923e275e16499f0fae9bc64522b3e50aeb4b44703e4118c.jpg)

![340ccd48e69c48a31b7ac50fd2e45b26a5d36cbc34c63d3707e09dfff30bf091.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/340ccd48e69c48a31b7ac50fd2e45b26a5d36cbc34c63d3707e09dfff30bf091.jpg)

![3e4208c2c1eecdfe8db92a06b7edb160c674fdc1e9cd5e245fc4e72b74f8a27f.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/3e4208c2c1eecdfe8db92a06b7edb160c674fdc1e9cd5e245fc4e72b74f8a27f.jpg)

![40ec2e8a1c9062a45a94faa311914fbc57b4a79b49b44eaea3d9e4cbf42547e5.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/40ec2e8a1c9062a45a94faa311914fbc57b4a79b49b44eaea3d9e4cbf42547e5.jpg)

![43b9fbbc437b033e77164d06eb73abebc1febe64bbe310fac8cf126ec5bbf289.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/43b9fbbc437b033e77164d06eb73abebc1febe64bbe310fac8cf126ec5bbf289.jpg)

![4a86e88afd57f5af65e1e88e86e8e2a31fcd76303535304b836de979060b411f.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/4a86e88afd57f5af65e1e88e86e8e2a31fcd76303535304b836de979060b411f.jpg)

![4b76bbc8d6152e2b8ea8e45ca5234329d7918afd63bef9965b4e5b1461198d71.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/4b76bbc8d6152e2b8ea8e45ca5234329d7918afd63bef9965b4e5b1461198d71.jpg)

![4da38fc2456521e2fc7c210d784fa56e62e49ec663ad738c6c99e296d93b1738.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/4da38fc2456521e2fc7c210d784fa56e62e49ec663ad738c6c99e296d93b1738.jpg)

![4f9f754be0839b8d917a8696e95fc1a85c60cf86b414aa3f4d41ebec28223cd5.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/4f9f754be0839b8d917a8696e95fc1a85c60cf86b414aa3f4d41ebec28223cd5.jpg)

![560f33f6ee9bf9af627ff2a3d50f8e1a3c0be5006fbd22a92a72a0a07d184115.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/560f33f6ee9bf9af627ff2a3d50f8e1a3c0be5006fbd22a92a72a0a07d184115.jpg)

![73d48ea9e126ab9df05ec581d19be2c3ae9283c24aa9fe35763d1acdd3e0e90c.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/73d48ea9e126ab9df05ec581d19be2c3ae9283c24aa9fe35763d1acdd3e0e90c.jpg)

![7a737cc9e414751495b0a090e6d96621ca74eea67d498c67c502574bc4d67fd4.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/7a737cc9e414751495b0a090e6d96621ca74eea67d498c67c502574bc4d67fd4.jpg)

![81eaaaf977d4702b4b53448ac3f3ed7b48ab577d5c4b4395c749d2ab14f4a4df.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/81eaaaf977d4702b4b53448ac3f3ed7b48ab577d5c4b4395c749d2ab14f4a4df.jpg)

![897423f818a8ffc572728477e5d801afb2a6844627f639ff30471816fe37b522.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/897423f818a8ffc572728477e5d801afb2a6844627f639ff30471816fe37b522.jpg)

![8999dcebb49216315409c0b98a06f5e77813d666259e485c71a26c226d937472.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/8999dcebb49216315409c0b98a06f5e77813d666259e485c71a26c226d937472.jpg)

![8bdcc0ab1317377d1c4db54aa895043fa8c641fb4fd56cb2f55db742ce1d9320.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/8bdcc0ab1317377d1c4db54aa895043fa8c641fb4fd56cb2f55db742ce1d9320.jpg)

![8c12f099b45ca0492d7ba2e0cb39a441527f221a9ebf887aee90ce5faf240e5f.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/8c12f099b45ca0492d7ba2e0cb39a441527f221a9ebf887aee90ce5faf240e5f.jpg)

![8cba4476743c8854eddb3ea27cca86bb2669b1320cb4b4f3213428398e784c4e.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/8cba4476743c8854eddb3ea27cca86bb2669b1320cb4b4f3213428398e784c4e.jpg)

![9504f6dcd8a9d101a8ad67505580cd45436fb41afe28036a012e07d938d9e4bb.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/9504f6dcd8a9d101a8ad67505580cd45436fb41afe28036a012e07d938d9e4bb.jpg)

![9750914dae5117a4ff603fcf7608f323e67d2a03dd87d1fd8dee7e455e9469fc.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/9750914dae5117a4ff603fcf7608f323e67d2a03dd87d1fd8dee7e455e9469fc.jpg)

![9b045e5dafdbba5efe9638a9a0c6274b6f3dc6a32de4ef128c35e1df7c17061d.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/9b045e5dafdbba5efe9638a9a0c6274b6f3dc6a32de4ef128c35e1df7c17061d.jpg)

![9c6a4718f251381d752b7f70b5e82b2b0791851a4243e531ef5948b3978671fb.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/9c6a4718f251381d752b7f70b5e82b2b0791851a4243e531ef5948b3978671fb.jpg)

![ae8a92ee2da45a8d7fcbcfae57763d1fa907d1c81e15f6c57f7e77c8f5881460.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/ae8a92ee2da45a8d7fcbcfae57763d1fa907d1c81e15f6c57f7e77c8f5881460.jpg)

![be7ec62e5cb9c0c394af83a71c35becb06b74adafaa2b01fa1e2ca271ce2d365.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/be7ec62e5cb9c0c394af83a71c35becb06b74adafaa2b01fa1e2ca271ce2d365.jpg)

![bfc670bad9ca78c5b50db1f7c3cdeca95a61ece952fa9678c74318cf95e856b4.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/bfc670bad9ca78c5b50db1f7c3cdeca95a61ece952fa9678c74318cf95e856b4.jpg)

![c152656ee018533f02a9638a5c1e6025b306679110b3ed3de90d17fa9078e0c2.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/c152656ee018533f02a9638a5c1e6025b306679110b3ed3de90d17fa9078e0c2.jpg)

![cf125cdb6556363bf1ac66307cec23910e9d7b11c986d017b5845383eff11f66.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/cf125cdb6556363bf1ac66307cec23910e9d7b11c986d017b5845383eff11f66.jpg)

![d110132059782f9b6b7ce874c79e312091dc756567cd9f2707e951d34245f5d9.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/d110132059782f9b6b7ce874c79e312091dc756567cd9f2707e951d34245f5d9.jpg)

![d380972733caf616eba290519b222623c2e1fd1c59cf2153589256c6b11fc780.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/d380972733caf616eba290519b222623c2e1fd1c59cf2153589256c6b11fc780.jpg)

![dd33a2f9dcd8f8a23aa0d52bb93dde54f16614f30c1d8f3829a23a06a7cc7102.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/dd33a2f9dcd8f8a23aa0d52bb93dde54f16614f30c1d8f3829a23a06a7cc7102.jpg)

![e8864fac7933503fb002972a95e37af1335c1f31c61b29b61d947f03886f6439.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/e8864fac7933503fb002972a95e37af1335c1f31c61b29b61d947f03886f6439.jpg)

![f7598f3d9cee0580f1501cda9b751ae9f86ecb9bacea80a2e1ea984717926045.jpg](../icml_results/3761_Problem%20Distributions%20as%20Tasks_%20Repurposing%20Meta%20Learning%20for%20Generative%20Combinatorial%20Optimization%20/tables/f7598f3d9cee0580f1501cda9b751ae9f86ecb9bacea80a2e1ea984717926045.jpg)

<a id="paper-3722"></a>

## Beyond Instance-Level Self-Supervision in 3D Multi-Modal Medical Imaging


### Images

![516d6f7ee64305a85d2f560b0ef8086d711e578f497d75014ad494d506154865.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/516d6f7ee64305a85d2f560b0ef8086d711e578f497d75014ad494d506154865.jpg)

![64fc5819dc75668354e0960266bcd702922004f1b8bfe3d8203bddbec3ff8b64.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/64fc5819dc75668354e0960266bcd702922004f1b8bfe3d8203bddbec3ff8b64.jpg)

![90c74c779f9e490bbe61c1bd20f47a1d74c8e8381cc73a2dc89a6f87e9553ef6.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/90c74c779f9e490bbe61c1bd20f47a1d74c8e8381cc73a2dc89a6f87e9553ef6.jpg)

![a873f59b4a5ab4b27840b8be738e796d7dfa1876f032f9fe1b800cf4fb1a667e.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/a873f59b4a5ab4b27840b8be738e796d7dfa1876f032f9fe1b800cf4fb1a667e.jpg)

![ab32a8d3ac743f0af88901257a57c27b1fa7f31814e42f056d4c96ec6e71a5fa.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/ab32a8d3ac743f0af88901257a57c27b1fa7f31814e42f056d4c96ec6e71a5fa.jpg)

![b316d30a406af5613fc68a405dd854ce746514e5dfd229d037b3e17723509584.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/b316d30a406af5613fc68a405dd854ce746514e5dfd229d037b3e17723509584.jpg)

![c696b23ed659af9f212a079d45b3c13c13815229ef1a1848e38d6f836442cc77.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/c696b23ed659af9f212a079d45b3c13c13815229ef1a1848e38d6f836442cc77.jpg)

![d6c373d06324950aa6d3bdbe00d58d85d768b268e00637982d57d3c077b32d81.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/images/d6c373d06324950aa6d3bdbe00d58d85d768b268e00637982d57d3c077b32d81.jpg)

### Tables

![057893ef385d5a5f2e6e77f05fd28d4c306da8a3f8bae5b252d720cd333c1f2e.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/057893ef385d5a5f2e6e77f05fd28d4c306da8a3f8bae5b252d720cd333c1f2e.jpg)

![14eae30ea215f0d8a3fb102c1b02cd61d0279f90cbac7032281a9eab63544ab5.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/14eae30ea215f0d8a3fb102c1b02cd61d0279f90cbac7032281a9eab63544ab5.jpg)

![1c18616892a724f01586e6356fbf843ca72e8d8822e2c7ba4ea4b1d9371150b8.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/1c18616892a724f01586e6356fbf843ca72e8d8822e2c7ba4ea4b1d9371150b8.jpg)

![539873ae7f7f6ac1294badb088e58d38cc4e7f8709a9c1c279dfce6180351178.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/539873ae7f7f6ac1294badb088e58d38cc4e7f8709a9c1c279dfce6180351178.jpg)

![58051476d9170646f5241ee8e752294b8a75ec0ec0735f2d8908d96778fabb07.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/58051476d9170646f5241ee8e752294b8a75ec0ec0735f2d8908d96778fabb07.jpg)

![7195ee86e9a7307ba4c39730a005b713141f4ef614fc42373ea3a40b7d1d90ce.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/7195ee86e9a7307ba4c39730a005b713141f4ef614fc42373ea3a40b7d1d90ce.jpg)

![98cb6e248509c97dd9047afe47f2d73c68228cba358459ba04a703cb3ca220d1.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/98cb6e248509c97dd9047afe47f2d73c68228cba358459ba04a703cb3ca220d1.jpg)

![99f256a1a93c5ba5fb340883c0ac0206def1f8b57a4ff96b71561ebc57cdcd27.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/99f256a1a93c5ba5fb340883c0ac0206def1f8b57a4ff96b71561ebc57cdcd27.jpg)

![a905f1703bb2e32c32f38760a0f74fd327bc380d5d760ef0a81bb9d5c98c3989.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/a905f1703bb2e32c32f38760a0f74fd327bc380d5d760ef0a81bb9d5c98c3989.jpg)

![b4dd548ab78a3fa9125385ed01ae0e5d9902deffa0d90151f8011e16eb0e8d13.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/b4dd548ab78a3fa9125385ed01ae0e5d9902deffa0d90151f8011e16eb0e8d13.jpg)

![f91ba5c13c13b917982bff5a9a3c52ff41f32f32c4d14b033e9ae70168371feb.jpg](../icml_results/3762_Beyond%20Instance-Level%20Self-Supervision%20in%203D%20Multi-Modal%20Medical%20Imaging/tables/f91ba5c13c13b917982bff5a9a3c52ff41f32f32c4d14b033e9ae70168371feb.jpg)

<a id="paper-3723"></a>

## Knowing Bias, Doing Better: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement


### Images

![4b9cdcd43584da1fc12aa90ec0017474e5a9799b48d61654adbb69ef60aa7ba3.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/images/4b9cdcd43584da1fc12aa90ec0017474e5a9799b48d61654adbb69ef60aa7ba3.jpg)

![6a3705c3529c53a97526fca770f4b082062baf00d5671edec9b41f58fc3d3478.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/images/6a3705c3529c53a97526fca770f4b082062baf00d5671edec9b41f58fc3d3478.jpg)

![93e2f8836b2323a29a34d8d34ecc81412284b195dc8bcf0410ab96d48ed2d1aa.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/images/93e2f8836b2323a29a34d8d34ecc81412284b195dc8bcf0410ab96d48ed2d1aa.jpg)

![b20270a35e5814fb7ca1be81631d5b74aca318b4b1cf2a6de5b1a073f7c624c7.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/images/b20270a35e5814fb7ca1be81631d5b74aca318b4b1cf2a6de5b1a073f7c624c7.jpg)

![f94963c73af591018417cf33d66d153dc4dea79803005359b18a776c7651bc3c.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/images/f94963c73af591018417cf33d66d153dc4dea79803005359b18a776c7651bc3c.jpg)

### Tables

![0f2e1d211bc7a1bc47291ac565fd776f3fb27f3b8c48ec3f5ed17e5019c0ceac.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/0f2e1d211bc7a1bc47291ac565fd776f3fb27f3b8c48ec3f5ed17e5019c0ceac.jpg)

![0fe1cf2203e1101e9cb30e8637ef485446ee191e93693d9cc37abf6f7eb9e7e5.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/0fe1cf2203e1101e9cb30e8637ef485446ee191e93693d9cc37abf6f7eb9e7e5.jpg)

![111b39d45d9bd880dfd3b9d7e270a039aa769f1a0e5d99d7fcce9f279b2b24bd.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/111b39d45d9bd880dfd3b9d7e270a039aa769f1a0e5d99d7fcce9f279b2b24bd.jpg)

![19cbe0adc3d7b5ade760c8f8e3614f8828e0e7dab9373bd9845dd7fdb27c4616.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/19cbe0adc3d7b5ade760c8f8e3614f8828e0e7dab9373bd9845dd7fdb27c4616.jpg)

![23f38e3fe668d724677790c55b405afcca215c215d3a0b7f60b6e130133a3038.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/23f38e3fe668d724677790c55b405afcca215c215d3a0b7f60b6e130133a3038.jpg)

![30ab9f194a7eecb3b5d2f72eee767d9fa279a6ec68b8e9a1091b3269b64ec422.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/30ab9f194a7eecb3b5d2f72eee767d9fa279a6ec68b8e9a1091b3269b64ec422.jpg)

![58b92c190e442bb642031a7ac2ac26a0d8d40846a966d5c29bb64b3042c44040.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/58b92c190e442bb642031a7ac2ac26a0d8d40846a966d5c29bb64b3042c44040.jpg)

![64c0275b0a2913c517ed546cd3ef74bff422662b5b87f33894c7c90cb686195f.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/64c0275b0a2913c517ed546cd3ef74bff422662b5b87f33894c7c90cb686195f.jpg)

![866505edba69332a03e63c5980aa778b2a89000b3b76051f50492c5c30dd064a.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/866505edba69332a03e63c5980aa778b2a89000b3b76051f50492c5c30dd064a.jpg)

![87fedc096aa1b09e320d8fa7534bf6bb9d99a85d117eb964f1aa62612eb6eb9f.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/87fedc096aa1b09e320d8fa7534bf6bb9d99a85d117eb964f1aa62612eb6eb9f.jpg)

![96e35231fb32b978570d2e71fed83f9e9d3c1912b8ae19830145bd3a261a2972.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/96e35231fb32b978570d2e71fed83f9e9d3c1912b8ae19830145bd3a261a2972.jpg)

![a8cd84b3b9d512fa6ad248b8d3b28bd2ce7cef8a717d98414ae7cd4dab59edb8.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/a8cd84b3b9d512fa6ad248b8d3b28bd2ce7cef8a717d98414ae7cd4dab59edb8.jpg)

![b541f302f6ad5a81cb39ba6d35b88a7a791115353170b440041bec329a4717e0.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/b541f302f6ad5a81cb39ba6d35b88a7a791115353170b440041bec329a4717e0.jpg)

![d165983dcbb8b262e8dfec93dee294b90563c52df0c267184fed124c653155c1.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/d165983dcbb8b262e8dfec93dee294b90563c52df0c267184fed124c653155c1.jpg)

![d954a9a67081690a1b6c40b01e4298a9cdcc6d8d8bc1a79eb2678b2ab79baedf.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/d954a9a67081690a1b6c40b01e4298a9cdcc6d8d8bc1a79eb2678b2ab79baedf.jpg)

![e00fc74fdd633b58a306c84413bf5013b10c388f7132b0e996ab77d359e1afc9.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/e00fc74fdd633b58a306c84413bf5013b10c388f7132b0e996ab77d359e1afc9.jpg)

![eac6a8e3c6fe46256ea26dbf1d46963162926240a1700d9c68a25918faa6cd5e.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/eac6a8e3c6fe46256ea26dbf1d46963162926240a1700d9c68a25918faa6cd5e.jpg)

![f63fc5703fa0a2e56062fb292cb9974df8eced0e86c86a5dad63d57d64afb1b5.jpg](../icml_results/3763_Knowing%20Bias%2C%20Doing%20Better_%20Mitigating%20Social%20Bias%20in%20LLMs%20via%20Know-Bias%20Neuron%20Enhancement/tables/f63fc5703fa0a2e56062fb292cb9974df8eced0e86c86a5dad63d57d64afb1b5.jpg)

<a id="paper-3724"></a>

## ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management


### Images

![01a8a8ebae9da9be0fb329cf408e5a9140766c56c057fa0701a8f88e5dec21fa.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/01a8a8ebae9da9be0fb329cf408e5a9140766c56c057fa0701a8f88e5dec21fa.jpg)

![03833c35a7a11febac3bc94809a9dd4fa1dd8e44ee3b7207798b38e2058fc659.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/03833c35a7a11febac3bc94809a9dd4fa1dd8e44ee3b7207798b38e2058fc659.jpg)

![215fac7bf943161f75629a85d099567f8754b82764b32e2d53b6252d156da989.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/215fac7bf943161f75629a85d099567f8754b82764b32e2d53b6252d156da989.jpg)

![25cf0f82d663b09fe47c4f79f6c8b4fa611056a23e6b9a4c36ffa557afa89a3c.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/25cf0f82d663b09fe47c4f79f6c8b4fa611056a23e6b9a4c36ffa557afa89a3c.jpg)

![2ad5dc0bc38ca806be13e515fb86f24c86470d16a0186ecbc75318e0f0baf97a.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/2ad5dc0bc38ca806be13e515fb86f24c86470d16a0186ecbc75318e0f0baf97a.jpg)

![2f9ad31d3e834f0525855bfa6b32ad2f12fb1bb98cc87ac80f4f827f10304336.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/2f9ad31d3e834f0525855bfa6b32ad2f12fb1bb98cc87ac80f4f827f10304336.jpg)

![33344a17920da0537192c6a5a9cb3af20bb7e49517ebb4feeed6d4001a55832a.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/33344a17920da0537192c6a5a9cb3af20bb7e49517ebb4feeed6d4001a55832a.jpg)

![38c6eac4752fdc01b8a7c2dc4f9bd6980c716beb781fb7cebbe92d29791e95b8.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/38c6eac4752fdc01b8a7c2dc4f9bd6980c716beb781fb7cebbe92d29791e95b8.jpg)

![45d945103c517799c2a71a464829818526140452daebb92c739a8abca696fb0d.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/45d945103c517799c2a71a464829818526140452daebb92c739a8abca696fb0d.jpg)

![5dccb80734fb6d4d4653a5898f897f58837bc961b655380e0ea5131787a23941.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/5dccb80734fb6d4d4653a5898f897f58837bc961b655380e0ea5131787a23941.jpg)

![61acac8133feaacaecd72c66957d24f84fe7efde37116c6fb49c4d70ab7a2c07.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/61acac8133feaacaecd72c66957d24f84fe7efde37116c6fb49c4d70ab7a2c07.jpg)

![734942a7fd8b05d5d3fdccbf9f008a790cd2b9cdcd0181a40e76752ec76fa7bc.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/734942a7fd8b05d5d3fdccbf9f008a790cd2b9cdcd0181a40e76752ec76fa7bc.jpg)

![7a4be972b73bd552e05f3d7d9cacdea37eb38cc46dc04b28c19a806e0666af0e.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/7a4be972b73bd552e05f3d7d9cacdea37eb38cc46dc04b28c19a806e0666af0e.jpg)

![af41cd8e79ee9cbbf98426aba1086131e9016ad0d146fcd7ff9524f2314d2a3e.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/af41cd8e79ee9cbbf98426aba1086131e9016ad0d146fcd7ff9524f2314d2a3e.jpg)

![bcd33f646447b93fed43cd737de473d86532fe96d70556ec80127b16addf5892.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/bcd33f646447b93fed43cd737de473d86532fe96d70556ec80127b16addf5892.jpg)

![c8c4bf40e3707fc6cbfd1cae5ad50d194ac61a7c12e0a96378c481e6aba474ce.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/c8c4bf40e3707fc6cbfd1cae5ad50d194ac61a7c12e0a96378c481e6aba474ce.jpg)

![dd2cef7a7c4f9877e3e3ae0f22a8180fa406277f4867dbc55b0ae36f5ea2db98.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/images/dd2cef7a7c4f9877e3e3ae0f22a8180fa406277f4867dbc55b0ae36f5ea2db98.jpg)

### Tables

![229fa9bdd7329f6d2b436f81e9d89c751c29ba7456d0c7e40c0e70de99aa808f.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/tables/229fa9bdd7329f6d2b436f81e9d89c751c29ba7456d0c7e40c0e70de99aa808f.jpg)

![50f019e20fab54203c80022e21bc1d39ce6c45c7e922c69fc2122f46502d5ec2.jpg](../icml_results/3764_ScaleSim_%20Serving%20Large-Scale%20Multi-Agent%20Simulation%20with%20Invocation%20Distance-Based%20Memory%20Managemen/tables/50f019e20fab54203c80022e21bc1d39ce6c45c7e922c69fc2122f46502d5ec2.jpg)

<a id="paper-3725"></a>

## Detecting and Filtering Unsafe Training Data via Data Attribution with Denoised Representation


### Images

![033d650751210d40176d8f50225c869711366c199ecfd04908aadec30de697eb.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/images/033d650751210d40176d8f50225c869711366c199ecfd04908aadec30de697eb.jpg)

![43dd54ad37821e86194b0defd1509a31ad1d38df691c1edbf49de8c2b9d97156.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/images/43dd54ad37821e86194b0defd1509a31ad1d38df691c1edbf49de8c2b9d97156.jpg)

![9a4b81d95f1323144fc7251f6b75ebdbe93b2ca174584485a39c98210eee9edc.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/images/9a4b81d95f1323144fc7251f6b75ebdbe93b2ca174584485a39c98210eee9edc.jpg)

![d60ec93b065cca436334cbd13170fa184e0f51493c50a6ff2021b326905dd6e4.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/images/d60ec93b065cca436334cbd13170fa184e0f51493c50a6ff2021b326905dd6e4.jpg)

### Tables

![13cc9a50a438e03d9f46368436b0d95d61dc20cdf11c4d67c10b79152c370e01.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/13cc9a50a438e03d9f46368436b0d95d61dc20cdf11c4d67c10b79152c370e01.jpg)

![14ad859e1eed89b021fd2d98f3cfbbca2eee46899ae8de5869adcd49266f90f0.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/14ad859e1eed89b021fd2d98f3cfbbca2eee46899ae8de5869adcd49266f90f0.jpg)

![3390bbbaae25cfddbc53c341cc7a72dabd66999ef0e9f56f71e7779bf187365a.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/3390bbbaae25cfddbc53c341cc7a72dabd66999ef0e9f56f71e7779bf187365a.jpg)

![56d115bd6e645de42edec5bbc2332c1e8d68195eb8594788e449526e847aad86.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/56d115bd6e645de42edec5bbc2332c1e8d68195eb8594788e449526e847aad86.jpg)

![76df3fd0103fc5072d3d380fdf688a3ab795fc1f05d1107ff10751b076e9915c.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/76df3fd0103fc5072d3d380fdf688a3ab795fc1f05d1107ff10751b076e9915c.jpg)

![78446d5c59c79d7ea14a3d68fe03a8316e8b97700239138eae7b70d43709daef.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/78446d5c59c79d7ea14a3d68fe03a8316e8b97700239138eae7b70d43709daef.jpg)

![9cd886b7efd3c894665e0e16070b7a422d1b67dd8d29bddaaf4e8cf6b426d5de.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/9cd886b7efd3c894665e0e16070b7a422d1b67dd8d29bddaaf4e8cf6b426d5de.jpg)

![b8e70c71a56446eaacf35f0c25c7c83e7c65a40de28643436f360cbed68b88b5.jpg](../icml_results/3765_Detecting%20and%20Filtering%20Unsafe%20Training%20Data%20via%20Data%20Attribution%20with%20Denoised%20Representation/tables/b8e70c71a56446eaacf35f0c25c7c83e7c65a40de28643436f360cbed68b88b5.jpg)

<a id="paper-3726"></a>

## Temporal-aware Flow Matching for Video Generation with Temporally Coherent Motion


### Images

![0034496b66557ed1368aecd2fdf257767cf6a96f7db46858e9f4a0f662eb676a.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/0034496b66557ed1368aecd2fdf257767cf6a96f7db46858e9f4a0f662eb676a.jpg)

![04c51963bb085dc0f88fe6e5996ecd215f16b5271f486826e3ada2ecbd7beebf.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/04c51963bb085dc0f88fe6e5996ecd215f16b5271f486826e3ada2ecbd7beebf.jpg)

![2afb68ffcff6bd483f8f263d5965a1a9c7975e403f71161be54b8bdbd63a1525.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/2afb68ffcff6bd483f8f263d5965a1a9c7975e403f71161be54b8bdbd63a1525.jpg)

![437e3a4213cbe1a61cc7578f6ed88ecf604e720148f8c8838f4f6e522486608c.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/437e3a4213cbe1a61cc7578f6ed88ecf604e720148f8c8838f4f6e522486608c.jpg)

![4dca6d286be3c94607af8627058b43b915087c1915fcdaf6577d84de566a1de5.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/4dca6d286be3c94607af8627058b43b915087c1915fcdaf6577d84de566a1de5.jpg)

![51ea52045748ff40dd77f51aee5c08bfb4fa898a437a9d190c1bfe5b5d0bb521.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/51ea52045748ff40dd77f51aee5c08bfb4fa898a437a9d190c1bfe5b5d0bb521.jpg)

![537251da9bfbfc81efde6764eed75816b7da9669f48f63cabdb539d50efb7f17.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/537251da9bfbfc81efde6764eed75816b7da9669f48f63cabdb539d50efb7f17.jpg)

![af3ac24e7cecc3b7157959f7d258c19e82d5f6c33d33ad9b51f4b68e72a96758.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/af3ac24e7cecc3b7157959f7d258c19e82d5f6c33d33ad9b51f4b68e72a96758.jpg)

![d9c14e5686a4d23122464c2a80038ba1d0b1892bff1db21d98002b72bdfd4224.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/images/d9c14e5686a4d23122464c2a80038ba1d0b1892bff1db21d98002b72bdfd4224.jpg)

### Tables

![16d9ec46a306a2c7395afc3d9d40e73f22997246ba5073d98493a2ad51be3dd0.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/16d9ec46a306a2c7395afc3d9d40e73f22997246ba5073d98493a2ad51be3dd0.jpg)

![1ef540b422a1b5a14386f7c7df0dbd142ad520d3543fecb468a8195fb151eec3.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/1ef540b422a1b5a14386f7c7df0dbd142ad520d3543fecb468a8195fb151eec3.jpg)

![20833ab9cc09782248e48d0dbe446c572202b6336b1bc065d58a40de969e7445.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/20833ab9cc09782248e48d0dbe446c572202b6336b1bc065d58a40de969e7445.jpg)

![335b4ea94fc2a253175a932f28ce78eb7211ecc335d054a0c1b16c21be5ed38a.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/335b4ea94fc2a253175a932f28ce78eb7211ecc335d054a0c1b16c21be5ed38a.jpg)

![3c35d02f43271cdf3a38b848873b197c839eb29e00c3304969bd1d9a93047aa3.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/3c35d02f43271cdf3a38b848873b197c839eb29e00c3304969bd1d9a93047aa3.jpg)

![5d52af2175a3abc0be7465d187497cd42f47949611a290752afc2c5d8d9ea4a5.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/5d52af2175a3abc0be7465d187497cd42f47949611a290752afc2c5d8d9ea4a5.jpg)

![6604ca7b0b794e757b7a6fbd7f99708392088760292f635fc00ae69b1bd913d7.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/6604ca7b0b794e757b7a6fbd7f99708392088760292f635fc00ae69b1bd913d7.jpg)

![75fea1117810fa144fb8629668d93e8dfa99b01d0454bb133d0689d76b4b7844.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/75fea1117810fa144fb8629668d93e8dfa99b01d0454bb133d0689d76b4b7844.jpg)

![b98ef6e1680ebd76519c1217ac6923e363485639d6cd986b1aa9a560ebaec0a1.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/b98ef6e1680ebd76519c1217ac6923e363485639d6cd986b1aa9a560ebaec0a1.jpg)

![be17301689df5d2611f7696547d78d0afa8a235ae870396fe3a53434bde1c7de.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/be17301689df5d2611f7696547d78d0afa8a235ae870396fe3a53434bde1c7de.jpg)

![d4013e4a73d236151e329ffd48364e12610a9d4dddd0175cf2a07bfbd77318d0.jpg](../icml_results/3766_Temporal-aware%20Flow%20Matching%20for%20Video%20Generation%20with%20Temporally%20Coherent%20Motion/tables/d4013e4a73d236151e329ffd48364e12610a9d4dddd0175cf2a07bfbd77318d0.jpg)

<a id="paper-3727"></a>

## Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning


### Images

![4b9b29bd0796cf99377b6daec829ce5af09e8a0dd865543305532a9949c813d9.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/4b9b29bd0796cf99377b6daec829ce5af09e8a0dd865543305532a9949c813d9.jpg)

![743a8b1b6f89a2c599d8bee294531604aa4c931a338c4dba0f99f91fb26e3b8c.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/743a8b1b6f89a2c599d8bee294531604aa4c931a338c4dba0f99f91fb26e3b8c.jpg)

![9269289b2cb30f6f86127713730d1cd85ff61cd3ff3012022daff458ad9268e2.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/9269289b2cb30f6f86127713730d1cd85ff61cd3ff3012022daff458ad9268e2.jpg)

![adb89e24ad8df9799f6b572d7d25a54d246479bacc9605705d990e96ae1e6d53.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/adb89e24ad8df9799f6b572d7d25a54d246479bacc9605705d990e96ae1e6d53.jpg)

![b681d50656fa228755f2d331f82f03236146777a79eae30e7770fc43dab62129.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/b681d50656fa228755f2d331f82f03236146777a79eae30e7770fc43dab62129.jpg)

![bafb6442e73c3a15333eee94c0afed4a34432d71f7c33702f3b18a4df832a5d0.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/bafb6442e73c3a15333eee94c0afed4a34432d71f7c33702f3b18a4df832a5d0.jpg)

![d16186afdd39a1e2a7876dbace988f164c5348e523e2160ee5a0490c2444e223.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/d16186afdd39a1e2a7876dbace988f164c5348e523e2160ee5a0490c2444e223.jpg)

![d5d4435deed3c2487a5a38df1d2ffdf13ff071ed93064ab51f44cf2dfb177058.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/images/d5d4435deed3c2487a5a38df1d2ffdf13ff071ed93064ab51f44cf2dfb177058.jpg)

### Tables

![33a886e203e7f6dfd024db7bfb5d58596a5817c218de9869ecf3453ed8b21113.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/tables/33a886e203e7f6dfd024db7bfb5d58596a5817c218de9869ecf3453ed8b21113.jpg)

![a6220bc8bc968efa8eb8266b39509cd7ac20e8662328fc5d25dbe14a55c19707.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/tables/a6220bc8bc968efa8eb8266b39509cd7ac20e8662328fc5d25dbe14a55c19707.jpg)

![c062adc69366cf755183df5c2ccfbe928919c2d42b043b76d0e7c4e5c96c26c5.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/tables/c062adc69366cf755183df5c2ccfbe928919c2d42b043b76d0e7c4e5c96c26c5.jpg)

![f756c724f3f9217c31b6bc2b9622cbe461d0b6e0dd0320e6fa1261de626f49da.jpg](../icml_results/3767_Stochastic%20Minimum-Cost%20Reach-Avoid%20Reinforcement%20Learning/tables/f756c724f3f9217c31b6bc2b9622cbe461d0b6e0dd0320e6fa1261de626f49da.jpg)

<a id="paper-3728"></a>

## Optimal Transport for LLM Reward Modeling from Noisy Feedback


### Images

![045209ab4c0068cc36f9a529f508b57be2024b26c1d96c78dda4b98e734f5159.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/045209ab4c0068cc36f9a529f508b57be2024b26c1d96c78dda4b98e734f5159.jpg)

![0f454b2a52e191d87f00f2c3108d83fc4ceb2ecc89015942ad539022b4b13661.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/0f454b2a52e191d87f00f2c3108d83fc4ceb2ecc89015942ad539022b4b13661.jpg)

![4d161d6dcf8d01b4639ab6b80b9098628a5f9fee1d41bfc33a468d167cdd302c.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/4d161d6dcf8d01b4639ab6b80b9098628a5f9fee1d41bfc33a468d167cdd302c.jpg)

![8943289cdf4d92809991076542edefa07160f5520015c2df3eb208ed8fe7ad30.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/8943289cdf4d92809991076542edefa07160f5520015c2df3eb208ed8fe7ad30.jpg)

![8ed8da1f047cfb0ec39b3a889b7afefa715755aa0bcb2b43d26304b7169b9df6.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/8ed8da1f047cfb0ec39b3a889b7afefa715755aa0bcb2b43d26304b7169b9df6.jpg)

![96b7d844c9f401b64d84fd034c166ec080c376bc27e50f8e3dc3a472ab4ba49c.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/96b7d844c9f401b64d84fd034c166ec080c376bc27e50f8e3dc3a472ab4ba49c.jpg)

![a68ba8e1bff8e227e7cae8439f0ddfe5898db0dd07e029ddc875d3313faa9704.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/a68ba8e1bff8e227e7cae8439f0ddfe5898db0dd07e029ddc875d3313faa9704.jpg)

![cd903877aa4b2623c057ce58bb4e2d560f819b884a42c6ffaf76eb5870a1d1e6.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/images/cd903877aa4b2623c057ce58bb4e2d560f819b884a42c6ffaf76eb5870a1d1e6.jpg)

### Tables

![092959dddd782797a76132b07623b3f957968b7ef44c5f4f519ec74a38a797d9.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/092959dddd782797a76132b07623b3f957968b7ef44c5f4f519ec74a38a797d9.jpg)

![1ed13cb132c5ae667117554da157115d4cb61ee50bf8f685dde471f185c5bddf.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/1ed13cb132c5ae667117554da157115d4cb61ee50bf8f685dde471f185c5bddf.jpg)

![493b0c23febe703aa63bcc71a4566fa86a1bcd2e0f16db29943309a7222e8a9d.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/493b0c23febe703aa63bcc71a4566fa86a1bcd2e0f16db29943309a7222e8a9d.jpg)

![75c1596ac35899db5ddd64aa7013bd822a27cff87a2295e038af5aa432dab1f0.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/75c1596ac35899db5ddd64aa7013bd822a27cff87a2295e038af5aa432dab1f0.jpg)

![7f5da8f8e895d5e82d1609626eb2e596f8a4ee8541047afa868d592784082c0f.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/7f5da8f8e895d5e82d1609626eb2e596f8a4ee8541047afa868d592784082c0f.jpg)

![c4ec53ad68a3fd32bfdb66bd8a5a483d62bf0e38af33bb7816655eaa68f405e6.jpg](../icml_results/3768_Optimal%20Transport%20for%20LLM%20Reward%20Modeling%20from%20Noisy%20Feedback/tables/c4ec53ad68a3fd32bfdb66bd8a5a483d62bf0e38af33bb7816655eaa68f405e6.jpg)

<a id="paper-3729"></a>

## AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning

### Images

![11b680e7dca05d9981e774e0e5658fe6ce230db6d3dbc2b4a9aa6d4876238e16.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/11b680e7dca05d9981e774e0e5658fe6ce230db6d3dbc2b4a9aa6d4876238e16.jpg)

![12820af21c99f5b04618c4231c9d322fe28120079f89b1f3f01924052a2ccbb1.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/12820af21c99f5b04618c4231c9d322fe28120079f89b1f3f01924052a2ccbb1.jpg)

![2dfedfbd49946ad8854a4e5bcfe660cb09dc436e2a44305bbc1c14368f46de63.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/2dfedfbd49946ad8854a4e5bcfe660cb09dc436e2a44305bbc1c14368f46de63.jpg)

![334132bfe2006fa0178e04021177bd8e06ac61d11ac15db94908fb510c552437.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/334132bfe2006fa0178e04021177bd8e06ac61d11ac15db94908fb510c552437.jpg)

![3533e9510164d5c861f3caaeb2eb14a65afa247bc9a19c81d271bdad667652e0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/3533e9510164d5c861f3caaeb2eb14a65afa247bc9a19c81d271bdad667652e0.jpg)

![5355c2456bcf9019310d2e32a9d1b1fe1a9fae01643bb35a9d902fbde27a7946.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5355c2456bcf9019310d2e32a9d1b1fe1a9fae01643bb35a9d902fbde27a7946.jpg)

![5a19a925bd804cc050ffe2d0d1abb933c84950c294b346e4f9137db961c94d8a.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5a19a925bd804cc050ffe2d0d1abb933c84950c294b346e4f9137db961c94d8a.jpg)

![5dda6551ea30467bc87558d3f1eaad6d1094a2d54a1e98859db233e683f9c921.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5dda6551ea30467bc87558d3f1eaad6d1094a2d54a1e98859db233e683f9c921.jpg)

![c0d3918db461446f7571b5d83760ad589bc655747cdccbce56e6154f77507d3c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/c0d3918db461446f7571b5d83760ad589bc655747cdccbce56e6154f77507d3c.jpg)

![d0592b6eac86719056976ce3930599ad7b318d3c18d4e4cecb3527fc67232b98.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/d0592b6eac86719056976ce3930599ad7b318d3c18d4e4cecb3527fc67232b98.jpg)

![d364c2c4d5cadc08395f669610f1074c8b7e73da00aa000e958c116923092091.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/d364c2c4d5cadc08395f669610f1074c8b7e73da00aa000e958c116923092091.jpg)

![e5ff4e186bc9808c4a692b31b50fa1b4009c48854c8f1ab3b3553d730e83dd1a.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/e5ff4e186bc9808c4a692b31b50fa1b4009c48854c8f1ab3b3553d730e83dd1a.jpg)

### Tables

![05910b3efee3a677ec4de965497af3dd85fa92dca68192be64fbadc141309e13.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/05910b3efee3a677ec4de965497af3dd85fa92dca68192be64fbadc141309e13.jpg)

![1f731284cb9b52524a881365a323fb189896b12db9ae4884d6b0827db349c97e.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/1f731284cb9b52524a881365a323fb189896b12db9ae4884d6b0827db349c97e.jpg)

![1fb21ced81a0b4125488008657e008fd5cef5e6a7991fe6d8c988e58816bdbef.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/1fb21ced81a0b4125488008657e008fd5cef5e6a7991fe6d8c988e58816bdbef.jpg)

![248845af73f48cb753a3bbcf5ad0be3824a3e1e1281c7e14035cbd0626adecb4.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/248845af73f48cb753a3bbcf5ad0be3824a3e1e1281c7e14035cbd0626adecb4.jpg)

![27c6817d18c8191560faa65a02232b3a8c4e84edb4cee8967698b38d8cc211c0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/27c6817d18c8191560faa65a02232b3a8c4e84edb4cee8967698b38d8cc211c0.jpg)

![32bfe5009d7d474748bb534dd4a66648c717db527956ba08a4a4fc0f2e9138a0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/32bfe5009d7d474748bb534dd4a66648c717db527956ba08a4a4fc0f2e9138a0.jpg)

![54f9c0eb758f834a91da517bf1a5c58033894c5a42a0d4b335d75f5dc01cff4e.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/54f9c0eb758f834a91da517bf1a5c58033894c5a42a0d4b335d75f5dc01cff4e.jpg)

![6e35fbd26c12e28832c3d182313ba2418b8f9a748393f5f9a5ce898ad738c58c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/6e35fbd26c12e28832c3d182313ba2418b8f9a748393f5f9a5ce898ad738c58c.jpg)

![73812515953af23d58bfcae2f47e12cac5c963cf0c41ee0bc044434fcc6ec33d.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/73812515953af23d58bfcae2f47e12cac5c963cf0c41ee0bc044434fcc6ec33d.jpg)

![9478d012a1d70b82a19c9510c7ae89a23664009bfdbf1fbcd9595558b14698e6.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/9478d012a1d70b82a19c9510c7ae89a23664009bfdbf1fbcd9595558b14698e6.jpg)

![c55b5a51f1fe3a9597a01a714dfdc61baa8ce588983c31fa9e0b0b41f827be34.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/c55b5a51f1fe3a9597a01a714dfdc61baa8ce588983c31fa9e0b0b41f827be34.jpg)

![d7983fa5f676375bc9071372f19e3af329d9c09bc91cbee839e66eb6c4f22bc1.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/d7983fa5f676375bc9071372f19e3af329d9c09bc91cbee839e66eb6c4f22bc1.jpg)

![da90c31cdea07261286dd64d0c2e4a263ea3abf88c37eb3560b248cf3bafe00c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/da90c31cdea07261286dd64d0c2e4a263ea3abf88c37eb3560b248cf3bafe00c.jpg)

![dcf347e6693d64eb5447f00119313e77325895bda13726a1a77f2a611b546b70.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/dcf347e6693d64eb5447f00119313e77325895bda13726a1a77f2a611b546b70.jpg)

![dd0e2e5c49ddc1a11a3f003587f3181316d9a5110015ce49a40d9801e62a33f2.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/dd0e2e5c49ddc1a11a3f003587f3181316d9a5110015ce49a40d9801e62a33f2.jpg)

![f1f8e9eedd05f861bc8643c5e8633cdc9cc0773e0ff210bae3d282c7bbfada74.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/f1f8e9eedd05f861bc8643c5e8633cdc9cc0773e0ff210bae3d282c7bbfada74.jpg)
