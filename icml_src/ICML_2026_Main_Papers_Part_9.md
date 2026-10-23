# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 9 of 197

## 目录 (Table of Contents)

1. [Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink](#paper-0265)
2. [LaST$_0$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model](#paper-0266)
3. [Midtraining Bridges Pretraining and Posttraining Distributions](#paper-0267)
4. [Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](#paper-0268)
5. [From Denoising to De-Channeling: Integrating Physical Channel Priors into Diffusion Models for Radio Signal Understanding](#paper-0269)
6. [TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance](#paper-0270)
7. [Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning](#paper-0271)
8. [Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection](#paper-0272)
9. [TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention](#paper-0273)
10. [Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models](#paper-0274)
11. [Towards the Explainability of Temporal Graph Networks via Memory Backtracking and Topological Attribution](#paper-0275)
12. [Balancing Understanding and Generation in Discrete Diffusion Models](#paper-0276)
13. [Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving](#paper-0277)
14. [Learning Biophysical Models of Large-Scale Multineuronal Data To Enable Precise Neurostimulation](#paper-0278)
15. [Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion](#paper-0279)
16. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](#paper-0280)
17. [Interpretable Functional Koopman Learning with Non-Markovian Closure for Spatiotemporal Systems](#paper-0281)
18. [Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching](#paper-0282)
19. [FedARC: Anchor-Guided Residual Compensation for Data and Model Heterogeneous Federated Learning](#paper-0283)
20. [Ratio-Variance Regularized Policy Optimization](#paper-0284)
21. [CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability](#paper-0285)
22. [Symmetry Reveals Layerwise Dynamics: How Transformers Perform In-Context Classification](#paper-0286)
23. [Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning](#paper-0287)
24. [Faster Activation Functions at the Edge for Post-Training Speedups](#paper-0288)
25. [Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model](#paper-0289)
26. [Breaking the Reversal Curse in Autoregressive Language Models via Identity Bridge](#paper-0290)
27. [What Makes Value Learning Efficient in Residual Reinforcement Learning?](#paper-0291)
28. [Dual-Latent Memory Routing for Vision-Language Reasoning](#paper-0292)
29. [ScaleMoE: Mixture-of-Experts for Scalable Continuous Control in Actor-Critic Reinforcement Learning](#paper-0293)
30. [MACKO: Sparse matrix-vector multiplication for low sparsity](#paper-0294)
31. [SWING: Unlocking Implicit Graph Representations for Graph Random Features](#paper-0295)
32. [Quantifying Frontier LLM Capabilities for Container Sandbox Escape](#paper-0296)
33. [Benchmarking at the Edge of Comprehension](#paper-0297)

---


<a id="paper-0265"></a>

## Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink

### Images

![14e7c6ef9d459b14dd6d6baba7a93b08d9e1bd75cd2613decb8570f6e8ac3e7f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/14e7c6ef9d459b14dd6d6baba7a93b08d9e1bd75cd2613decb8570f6e8ac3e7f.jpg)

![3e36c034ce412f1322ef582da94595b30e9fd644797a25f1f908ada5428c1176.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/3e36c034ce412f1322ef582da94595b30e9fd644797a25f1f908ada5428c1176.jpg)

![4dd246e4cb4a2bb37aceb09da20eabd3d9de7cd57a0c699747faaf43c517f82d.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/4dd246e4cb4a2bb37aceb09da20eabd3d9de7cd57a0c699747faaf43c517f82d.jpg)

![7b89bd493612ff2c5e09a5d51546ead28e9ef26ef85a082e8a8a49c3102fb3b5.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/7b89bd493612ff2c5e09a5d51546ead28e9ef26ef85a082e8a8a49c3102fb3b5.jpg)

![8177d992a91a5b1fa14762ba459d4d7ffc8d4fff3eeecfed6aa0fa1f55b03355.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/8177d992a91a5b1fa14762ba459d4d7ffc8d4fff3eeecfed6aa0fa1f55b03355.jpg)

![a6430daf0b7ced99229151545931f572ddc3a2943f08f235659b1cda406687a4.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/a6430daf0b7ced99229151545931f572ddc3a2943f08f235659b1cda406687a4.jpg)

![ba267c6a75a9ad1b937fec2861a392cb404cb771da771953cb5eae1ce6f9aa36.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/ba267c6a75a9ad1b937fec2861a392cb404cb771da771953cb5eae1ce6f9aa36.jpg)

![c3e60463f8d2c80323e1fa647f82d4a02c71b8fb6a4f9ac67ddee130863a945d.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/c3e60463f8d2c80323e1fa647f82d4a02c71b8fb6a4f9ac67ddee130863a945d.jpg)

![cb56352b8587936f23f1e8cad1cb2e0a0ac7a8ec2196fc72070ec67cd817383e.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/cb56352b8587936f23f1e8cad1cb2e0a0ac7a8ec2196fc72070ec67cd817383e.jpg)

![d932e475842d683200a93fd6ec91b5d922de294c178813b4f5fe480472fb61e7.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/d932e475842d683200a93fd6ec91b5d922de294c178813b4f5fe480472fb61e7.jpg)

![e7a1824278a1f6ccaf91d483039a3ba8680301e66ec442ce650d151bca77552f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/e7a1824278a1f6ccaf91d483039a3ba8680301e66ec442ce650d151bca77552f.jpg)

![fe7a75410ec52414d85af672095710e0603721303d837011e785480e4801d46a.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/fe7a75410ec52414d85af672095710e0603721303d837011e785480e4801d46a.jpg)

### Tables

![22adf03773a2d225a7b4d23bc85021bdd19283ae00f9264c89a14204b49dfb19.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/22adf03773a2d225a7b4d23bc85021bdd19283ae00f9264c89a14204b49dfb19.jpg)

![29e80fca1b73400877d84adf591ded0cd7d258fa1650f510c2c23cd1dd02196f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/29e80fca1b73400877d84adf591ded0cd7d258fa1650f510c2c23cd1dd02196f.jpg)

![958029f29eede0901fe80f6479a12631650a9c23eda9a96d7e5e94c16a02a703.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/958029f29eede0901fe80f6479a12631650a9c23eda9a96d7e5e94c16a02a703.jpg)

![97d0585e8471e938ee014d9d992f94d9031520fd345628cb57cd18b4f4ddda96.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/97d0585e8471e938ee014d9d992f94d9031520fd345628cb57cd18b4f4ddda96.jpg)

## Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink


### Images

![1fc4a79cec0fef97987069b334077fd0633be2d8b725006090bd96c9967793b1.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/1fc4a79cec0fef97987069b334077fd0633be2d8b725006090bd96c9967793b1.jpg)

![3b91f2fae457c7c6f684059b88a18fa43858428ea5616c090accbd895cb720d8.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/3b91f2fae457c7c6f684059b88a18fa43858428ea5616c090accbd895cb720d8.jpg)

![47b14a26c15e424acba87944a2722cd1ed45e4d2b413d8d9f931306a6ab1c248.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/47b14a26c15e424acba87944a2722cd1ed45e4d2b413d8d9f931306a6ab1c248.jpg)

![481be600152407f62b668695c6221b39a54366fb1bfee04c8e066f20a7c7ea11.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/481be600152407f62b668695c6221b39a54366fb1bfee04c8e066f20a7c7ea11.jpg)

![4bc3875c724c2758603c4ffe0ccd8199e85b176922e52726b098f4c332c51150.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/4bc3875c724c2758603c4ffe0ccd8199e85b176922e52726b098f4c332c51150.jpg)

![4f1213db14e1db9f6ae6b7599977da36453641fb42fba9ee6c1c0f7bbef8c26d.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/4f1213db14e1db9f6ae6b7599977da36453641fb42fba9ee6c1c0f7bbef8c26d.jpg)

![70fe8a8cf8072542f6f4de7c6c11cdd6dfcd01c13f4247d164b19dc31502deb2.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/70fe8a8cf8072542f6f4de7c6c11cdd6dfcd01c13f4247d164b19dc31502deb2.jpg)

![8261b319da13df1367ec2ebb7f274d033a0f7e4a768c2af61b579222037154c5.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/8261b319da13df1367ec2ebb7f274d033a0f7e4a768c2af61b579222037154c5.jpg)

![970f935a57c364a25115eb0966fe897eaca39df02759cbcbdf25f7f25234a61d.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/970f935a57c364a25115eb0966fe897eaca39df02759cbcbdf25f7f25234a61d.jpg)

![e4230ed0c98e321b3b640a1cfdc9b8039ec3d5241ac5376084a4beb6ebd6f74b.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/images/e4230ed0c98e321b3b640a1cfdc9b8039ec3d5241ac5376084a4beb6ebd6f74b.jpg)

### Tables

![0ca3f65bac54ae3119fe00b9ad2325c10bddd66df2b9d8a5176e367a56a9ba72.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/0ca3f65bac54ae3119fe00b9ad2325c10bddd66df2b9d8a5176e367a56a9ba72.jpg)

![13e3fa68e9f5a2316308952ed844780698e11a9432577118e186c96c199836c4.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/13e3fa68e9f5a2316308952ed844780698e11a9432577118e186c96c199836c4.jpg)

![3ee47eec3f15833a85403c9e8561e92044ac8a00ae8de113f11289466dce8de5.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/3ee47eec3f15833a85403c9e8561e92044ac8a00ae8de113f11289466dce8de5.jpg)

![4190aebaf74188875c2b4df83b88e52a112140691ee48cf0f7d2c9ffd6a92144.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/4190aebaf74188875c2b4df83b88e52a112140691ee48cf0f7d2c9ffd6a92144.jpg)

![4a61861b2199be6d6d11f1067a158da6ce6cbfc652cfeaac5bc5316327157ab7.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/4a61861b2199be6d6d11f1067a158da6ce6cbfc652cfeaac5bc5316327157ab7.jpg)

![4b28dbf5f501f90cbd7d5787caa3d7236b147ed9108d49bb51b9ed94c859d4ea.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/4b28dbf5f501f90cbd7d5787caa3d7236b147ed9108d49bb51b9ed94c859d4ea.jpg)

![4c8497f561a4494842508c8b94951e4f700623442ef5b4109675109f42e031de.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/4c8497f561a4494842508c8b94951e4f700623442ef5b4109675109f42e031de.jpg)

![5edef0851d0c764b2270d52fa0ab574bfedaafdae9442259cff61f817d5e1129.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/5edef0851d0c764b2270d52fa0ab574bfedaafdae9442259cff61f817d5e1129.jpg)

![8e66163617bac8319f4a1cc404bc564a23f057dc99d073df64fb36875a2c3ad7.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/8e66163617bac8319f4a1cc404bc564a23f057dc99d073df64fb36875a2c3ad7.jpg)

![9cde957b912b4e7c13e64a41dff2396c2405f0c0254f69a04d0d97326b894c62.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/9cde957b912b4e7c13e64a41dff2396c2405f0c0254f69a04d0d97326b894c62.jpg)

![a5a571911300790b1c8931622dc501923cc330d26cd6a7a72debdd982660616d.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/a5a571911300790b1c8931622dc501923cc330d26cd6a7a72debdd982660616d.jpg)

![acb0d0194c3a205e4927ee71406cad9bd72a2ff7e508432983894a39119cfb75.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/acb0d0194c3a205e4927ee71406cad9bd72a2ff7e508432983894a39119cfb75.jpg)

![b881be96bc0a45fa8c371f02e15f23769fff03b4d9758f139e3075f5abc4af10.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/b881be96bc0a45fa8c371f02e15f23769fff03b4d9758f139e3075f5abc4af10.jpg)

![c8644a48b53cb43ce0db6eeed54ec7a70add7ed7c2353e0bccbaccf177d333c7.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/c8644a48b53cb43ce0db6eeed54ec7a70add7ed7c2353e0bccbaccf177d333c7.jpg)

![cd58d442244321d7324e931644886c9e77fae5e63a4bd0e98b3bc068d432110e.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/cd58d442244321d7324e931644886c9e77fae5e63a4bd0e98b3bc068d432110e.jpg)

![cf4c99c3d67abad9a73480d7dd610d2ac3908cf1b1d933c1e6e8b94c125efee2.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/cf4c99c3d67abad9a73480d7dd610d2ac3908cf1b1d933c1e6e8b94c125efee2.jpg)

![cf510ae53026577bc4c52c9573658541abc65aee4411e42c63b95f07b3d3fcbb.jpg](../icml_results/270_Surgery_%20Mitigating%20Harmful%20Fine-Tuning%20for%20Large%20Language%20Models%20via%20Attention%20Sink/tables/cf510ae53026577bc4c52c9573658541abc65aee4411e42c63b95f07b3d3fcbb.jpg)

<a id="paper-0266"></a>

## LaST$_0$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model


### Images

![102ed1f6ebfb403fc0ff619c8b7dabe327b7bd522557bf0804eddffe11016c10.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/102ed1f6ebfb403fc0ff619c8b7dabe327b7bd522557bf0804eddffe11016c10.jpg)

![112386cb2e6957d6b25fbf53fc00addd69551eaff214f1a64960b433f94c9503.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/112386cb2e6957d6b25fbf53fc00addd69551eaff214f1a64960b433f94c9503.jpg)

![16cf2369891e03f6226ef5e95a94f98bc34c5446733d4df4c0bb63891aa9a02e.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/16cf2369891e03f6226ef5e95a94f98bc34c5446733d4df4c0bb63891aa9a02e.jpg)

![1da9087e7e698968c848509ae238cd3ec3942fbd089f7b5d6328cfe04391b4b8.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/1da9087e7e698968c848509ae238cd3ec3942fbd089f7b5d6328cfe04391b4b8.jpg)

![2b1130991d0f71c9f44d1e3864b1ce022b54cb1159b74ff54548e2c0e15d08d2.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/2b1130991d0f71c9f44d1e3864b1ce022b54cb1159b74ff54548e2c0e15d08d2.jpg)

![5dd4e936852580b88e8fad5ad7be321e9a3b1a31ad2d77f808000ed51e751021.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/5dd4e936852580b88e8fad5ad7be321e9a3b1a31ad2d77f808000ed51e751021.jpg)

![86df13b4ac6c8fc7c885d5274521417c8d175dc1f9d3c6b6187b32e48eb80e47.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/86df13b4ac6c8fc7c885d5274521417c8d175dc1f9d3c6b6187b32e48eb80e47.jpg)

![9045451162b9fc9d379c11c85626b3c3d055a60f6214804307697039a972aa29.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/9045451162b9fc9d379c11c85626b3c3d055a60f6214804307697039a972aa29.jpg)

![99b826b6510eee3b62d284831f0e00bbd698d99b15e3182dd857438ee75e2d2c.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/99b826b6510eee3b62d284831f0e00bbd698d99b15e3182dd857438ee75e2d2c.jpg)

![c1d26c6631caea8f0e3b9d5bad03683a4f8bce2bf9c52fc4a2583f92cba5f048.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/c1d26c6631caea8f0e3b9d5bad03683a4f8bce2bf9c52fc4a2583f92cba5f048.jpg)

![c7c117e357beffa1195aec1d87c18f09d3406f995306e43b008ce1b7e0302560.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/c7c117e357beffa1195aec1d87c18f09d3406f995306e43b008ce1b7e0302560.jpg)

![df382c281f4dfb5dad47afe8db4ad87a6c30b6db783ae9eb29f0990e8dd8594b.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/df382c281f4dfb5dad47afe8db4ad87a6c30b6db783ae9eb29f0990e8dd8594b.jpg)

![ea8ad0da641e3a49314a1636e5331210e6dc8a2f26b8bc085fc8cbb4b328b3ee.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/images/ea8ad0da641e3a49314a1636e5331210e6dc8a2f26b8bc085fc8cbb4b328b3ee.jpg)

### Tables

![11eb8550dd7a539bd31c11b33cb2fce0365bd3f71ab3b4de25aa5a4d3c0453aa.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/11eb8550dd7a539bd31c11b33cb2fce0365bd3f71ab3b4de25aa5a4d3c0453aa.jpg)

![18e407a8c521f17e1ef7e305a415e7f3f66b2c71dd5a182319e0a9cedcf8ce43.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/18e407a8c521f17e1ef7e305a415e7f3f66b2c71dd5a182319e0a9cedcf8ce43.jpg)

![45f754dd78b0668e2061e22af717f68e7cbe260a3c0584c59673ac5ce86a9b84.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/45f754dd78b0668e2061e22af717f68e7cbe260a3c0584c59673ac5ce86a9b84.jpg)

![5f96bfa345fc6190d63af039f057b21ca4b886d92af8ffde911f0eab30ec544c.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/5f96bfa345fc6190d63af039f057b21ca4b886d92af8ffde911f0eab30ec544c.jpg)

![7c95c04597504c5c4a318c6997a848b22100e9ac02fe32b52a9c1add8f4a9d96.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/7c95c04597504c5c4a318c6997a848b22100e9ac02fe32b52a9c1add8f4a9d96.jpg)

![82535012607863df1ea71d26cbca6443d26e8df38699dd341beb48754d79666b.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/82535012607863df1ea71d26cbca6443d26e8df38699dd341beb48754d79666b.jpg)

![b798ec6b7ea84cd27d77810f68f09e050ffc3442cf979c0c057cf2b25b648b7a.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/b798ec6b7ea84cd27d77810f68f09e050ffc3442cf979c0c057cf2b25b648b7a.jpg)

![df6c1cd433ac70475c7ff3f3fae84ad169224a1bfd7736c7fedab42e5cb8b671.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/df6c1cd433ac70475c7ff3f3fae84ad169224a1bfd7736c7fedab42e5cb8b671.jpg)

![e7418b0a7ce719d7d9a5a81de616929c4107b655a45be67e668e8a782828f9ef.jpg](../icml_results/271_LaST%24_0%24_%20Latent%20Spatio-Temporal%20Chain-of-Thought%20for%20Robotic%20Vision-Language-Action%20Model/tables/e7418b0a7ce719d7d9a5a81de616929c4107b655a45be67e668e8a782828f9ef.jpg)

<a id="paper-0267"></a>

## Midtraining Bridges Pretraining and Posttraining Distributions


### Images

![0a9bd71a77da41e898c95f8ac14df2532d60990bdc6a4b438713791de047884c.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/0a9bd71a77da41e898c95f8ac14df2532d60990bdc6a4b438713791de047884c.jpg)

![0e07702e934f2b41ff9963d7d258c4250fc6f610091e70666debe705bbbd0f35.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/0e07702e934f2b41ff9963d7d258c4250fc6f610091e70666debe705bbbd0f35.jpg)

![157be700e9d3c30eb6e6d994e4d1bf478826a75563e1394b531cd523a09ca6a0.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/157be700e9d3c30eb6e6d994e4d1bf478826a75563e1394b531cd523a09ca6a0.jpg)

![177f6d03e0d44c899c748c9a429719a46f500ca0184eb3f2d3f28a8de91b5b7a.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/177f6d03e0d44c899c748c9a429719a46f500ca0184eb3f2d3f28a8de91b5b7a.jpg)

![3a14b043dd63f725e2e4eb8690c0321ebcdba9d7fa370d42f3047add9faea2d2.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/3a14b043dd63f725e2e4eb8690c0321ebcdba9d7fa370d42f3047add9faea2d2.jpg)

![73ea0685349e5a581f3c2a540ac896756c218b3055fab65f305ee3021edc67db.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/73ea0685349e5a581f3c2a540ac896756c218b3055fab65f305ee3021edc67db.jpg)

![74a2c8e05fdc2ed528489d196827dee08cb05a6559b2021c34f90f699cdc9acd.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/74a2c8e05fdc2ed528489d196827dee08cb05a6559b2021c34f90f699cdc9acd.jpg)

![816b877bc382526a82e449708084babbde6c016855acdae838641be1297f1f3d.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/816b877bc382526a82e449708084babbde6c016855acdae838641be1297f1f3d.jpg)

![832e6a8844f689ba1b23416b744957e3d97d650d2265653d28c6c21f3c70aa40.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/832e6a8844f689ba1b23416b744957e3d97d650d2265653d28c6c21f3c70aa40.jpg)

![8f016e6039a91a978b0fda009fcdd85a33b24ee918e8850b0436b22ff7142111.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/8f016e6039a91a978b0fda009fcdd85a33b24ee918e8850b0436b22ff7142111.jpg)

![9725108c06536b735f6d39185d820721b842fcc4428be0ddc59b08418e037970.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/9725108c06536b735f6d39185d820721b842fcc4428be0ddc59b08418e037970.jpg)

![9d8ef75abbea38c405749f632aa3242de73cc3c85c0188214620e7309a05480e.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/9d8ef75abbea38c405749f632aa3242de73cc3c85c0188214620e7309a05480e.jpg)

![9e6bf8c3841b294995c86c2e58610091dcf0b35e08e6368551598963adbaabbf.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/9e6bf8c3841b294995c86c2e58610091dcf0b35e08e6368551598963adbaabbf.jpg)

![a8f4209160ec6014a3e26d478a7f3121c994f93d9edc240bba908d605e65d78c.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/a8f4209160ec6014a3e26d478a7f3121c994f93d9edc240bba908d605e65d78c.jpg)

![afb17725e2015f946d7ec322797277884613ae3d2a47a09557e3853f3f98131a.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/afb17725e2015f946d7ec322797277884613ae3d2a47a09557e3853f3f98131a.jpg)

![b8d9e14211a5cbb5f31fd6651297e1dc5ea85260e0547d06fb0dc176408cbefa.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/b8d9e14211a5cbb5f31fd6651297e1dc5ea85260e0547d06fb0dc176408cbefa.jpg)

![e4dafeb564c28a8a7367cabed1a2a08ced48be2e3b3aec674e91e6af9fecd864.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/e4dafeb564c28a8a7367cabed1a2a08ced48be2e3b3aec674e91e6af9fecd864.jpg)

![e513e5823e6cfdbc4be7c68ff04ea6d55986de1174936e301467bb2424ff302e.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/e513e5823e6cfdbc4be7c68ff04ea6d55986de1174936e301467bb2424ff302e.jpg)

![f089b94c167920891313f98028e42d8bd12f457060a7b77b81e7b876ddc8f2f0.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/images/f089b94c167920891313f98028e42d8bd12f457060a7b77b81e7b876ddc8f2f0.jpg)

### Tables

![296dbb342dd4f88f0e7f3c21f6f3230d160488a5f29e6b83f83e6ee614028723.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/296dbb342dd4f88f0e7f3c21f6f3230d160488a5f29e6b83f83e6ee614028723.jpg)

![4904632d4c6a108780f92de4e0fc6c0019c2a5384aafcb5626567aa1ee1adf39.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/4904632d4c6a108780f92de4e0fc6c0019c2a5384aafcb5626567aa1ee1adf39.jpg)

![771a76cc0c6e77f2ef32833708f37aa45f1d838524b396b4bd37c9bd131b0131.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/771a76cc0c6e77f2ef32833708f37aa45f1d838524b396b4bd37c9bd131b0131.jpg)

![7cddec8d7e7ee7bf627284fe6d6edfab4ce85bd006b553b1f8a2fc6516e35f56.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/7cddec8d7e7ee7bf627284fe6d6edfab4ce85bd006b553b1f8a2fc6516e35f56.jpg)

![8f26cb9f2aaf71c2bfc571263a45326ca339f2b5987c2f1b9e1957f39bc6bb3d.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/8f26cb9f2aaf71c2bfc571263a45326ca339f2b5987c2f1b9e1957f39bc6bb3d.jpg)

![99c684298e7a44af223a46358e707f3a18545d4e5fc47a84f2945b2eb4251dc7.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/99c684298e7a44af223a46358e707f3a18545d4e5fc47a84f2945b2eb4251dc7.jpg)

![a10e6924cf21f6095d8f9913b8a02ac219853c501ae8228483fcc32031bda047.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/a10e6924cf21f6095d8f9913b8a02ac219853c501ae8228483fcc32031bda047.jpg)

![f35471697db7e977bcd6ace376f705be47de7ff2eecbf43ec9663c68ab3e6d82.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/f35471697db7e977bcd6ace376f705be47de7ff2eecbf43ec9663c68ab3e6d82.jpg)

![f3fa888c610e55b2ef7750dedbece31fae8ee572355a13fd9134911f230543cb.jpg](../icml_results/272_Midtraining%20Bridges%20Pretraining%20and%20Posttraining%20Distributions/tables/f3fa888c610e55b2ef7750dedbece31fae8ee572355a13fd9134911f230543cb.jpg)

<a id="paper-0268"></a>

## Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation


### Images

![14d8cbe52530f1a7bfce25070769036f816bffbd9144a346ed80d7f37e4696ae.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/14d8cbe52530f1a7bfce25070769036f816bffbd9144a346ed80d7f37e4696ae.jpg)

![25a234fc019f10249c6c2264be1fe3b1ec3c7490697e56c51628621a858c4a13.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/25a234fc019f10249c6c2264be1fe3b1ec3c7490697e56c51628621a858c4a13.jpg)

![356ecbdad86155f7c60e8bbfb456d095628670b37e22f555ad5ec09ae3471d94.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/356ecbdad86155f7c60e8bbfb456d095628670b37e22f555ad5ec09ae3471d94.jpg)

![3f43e8ce150c5557028ca11c964fddb275c4c544d9c39caf8a1881c1cad586b8.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/3f43e8ce150c5557028ca11c964fddb275c4c544d9c39caf8a1881c1cad586b8.jpg)

![4463e52f6d9f77a60932c41189a1ffc50e7db9cfb8067ffd9476654afc6f1be6.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/4463e52f6d9f77a60932c41189a1ffc50e7db9cfb8067ffd9476654afc6f1be6.jpg)

![472ff8840c749d31e6d1ea4d5b881bfbb305c0664634293b8918d3a17410c188.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/472ff8840c749d31e6d1ea4d5b881bfbb305c0664634293b8918d3a17410c188.jpg)

![5455d76beae59bf1abd33c5c5341fd40c915d61c3f823979676f7ce84dce26d4.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/5455d76beae59bf1abd33c5c5341fd40c915d61c3f823979676f7ce84dce26d4.jpg)

![5e45f83061270a5b163667b096bcfc8937db3b9fe31230647f6063ea88993bfd.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/5e45f83061270a5b163667b096bcfc8937db3b9fe31230647f6063ea88993bfd.jpg)

![612c578da009435a54c5a228a31365e795756a11d61d40dbcc9c6f788350bbd1.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/612c578da009435a54c5a228a31365e795756a11d61d40dbcc9c6f788350bbd1.jpg)

![695b01563c95e951cc7e110ddae88dd85b9ce3737dc4583f7a662a6737258d4d.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/695b01563c95e951cc7e110ddae88dd85b9ce3737dc4583f7a662a6737258d4d.jpg)

![7faf8118c518bfddcbf32ba996b30d8f774285116799f081c7260fa0b8404cdb.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/7faf8118c518bfddcbf32ba996b30d8f774285116799f081c7260fa0b8404cdb.jpg)

![826d3569005f3e81b3b3ece974b9bb9ab841c0bfa46fbc577b41e0136d22b187.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/826d3569005f3e81b3b3ece974b9bb9ab841c0bfa46fbc577b41e0136d22b187.jpg)

![91570f3150c3e758a1c337dd2f3d4c0dd87a71ccc92cadda045058a867646444.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/91570f3150c3e758a1c337dd2f3d4c0dd87a71ccc92cadda045058a867646444.jpg)

![bb0999132a9d00efd227dcf311aff91be08ac7e725b04d48f58eb041ce86605d.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/bb0999132a9d00efd227dcf311aff91be08ac7e725b04d48f58eb041ce86605d.jpg)

![be00b5e40b6ee99b2caadce7690adcb9aeb4b914924804d54fc484ef9a24fb95.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/be00b5e40b6ee99b2caadce7690adcb9aeb4b914924804d54fc484ef9a24fb95.jpg)

![d4f49b81673032876f6afe424410f7148f4262b7809be0d69d4cd015e0c62e5d.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/d4f49b81673032876f6afe424410f7148f4262b7809be0d69d4cd015e0c62e5d.jpg)

![d82cd67d72f92be367bc788e1f4f4959aad17390936b80712f453c9e520b4a61.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/d82cd67d72f92be367bc788e1f4f4959aad17390936b80712f453c9e520b4a61.jpg)

![e7770105b5357d985d989e49c8c615f63e5335f431902e11bf4b11c0b7823b35.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/e7770105b5357d985d989e49c8c615f63e5335f431902e11bf4b11c0b7823b35.jpg)

![eb5c61e656fa88e80fe485ac079ec0d1df878e2150d01673b6b92c1c635e0f01.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/eb5c61e656fa88e80fe485ac079ec0d1df878e2150d01673b6b92c1c635e0f01.jpg)

![eba924c27c5576c1764673ea09f180d17ca27306b2b5b59635d82aebcd29694f.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/eba924c27c5576c1764673ea09f180d17ca27306b2b5b59635d82aebcd29694f.jpg)

![f0f2faffe8bbff641bda8737e05935af9e40960a822362806fa689353f372b8c.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/images/f0f2faffe8bbff641bda8737e05935af9e40960a822362806fa689353f372b8c.jpg)

### Tables

![0a6dba1144567a0944b8aa71db8b968223155f2750bbcd41afeda157e56e2cf8.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/0a6dba1144567a0944b8aa71db8b968223155f2750bbcd41afeda157e56e2cf8.jpg)

![1457df87a2b99c07178028489e84877fc78b6ac9e6fa63b79266d24ec5dddcc9.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/1457df87a2b99c07178028489e84877fc78b6ac9e6fa63b79266d24ec5dddcc9.jpg)

![17437d47a49acb4bbdb214547ae25b426a9d4520ad79255ceeeb008e84f714b3.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/17437d47a49acb4bbdb214547ae25b426a9d4520ad79255ceeeb008e84f714b3.jpg)

![24152722a6451de08f4436501e4776eb73dd3dfccd7e2040d2f6c007c7a2503e.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/24152722a6451de08f4436501e4776eb73dd3dfccd7e2040d2f6c007c7a2503e.jpg)

![36e7982b46411cb3f8e3539a8664ef1d581bdc02c451f478c2a6e6f62c5c582e.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/36e7982b46411cb3f8e3539a8664ef1d581bdc02c451f478c2a6e6f62c5c582e.jpg)

![4c4fb1e3eb19dee306fbe7fd397c6c77db5fa32202c5ef2e654d027c19b3c038.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/4c4fb1e3eb19dee306fbe7fd397c6c77db5fa32202c5ef2e654d027c19b3c038.jpg)

![4e7522acb7e57fc1947e0709defd840d3d0f615e3c892866bf84512d548234b3.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/4e7522acb7e57fc1947e0709defd840d3d0f615e3c892866bf84512d548234b3.jpg)

![7da8be8943841c93d163a02ab7ece85098fccb5bf7512097cfacf171580da9ed.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/7da8be8943841c93d163a02ab7ece85098fccb5bf7512097cfacf171580da9ed.jpg)

![912e9b3bcdf5db056e96abf98b0630cad973346c0ce6405df4173ce9058d34ff.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/912e9b3bcdf5db056e96abf98b0630cad973346c0ce6405df4173ce9058d34ff.jpg)

![b01d6627a9fefbfcc29892ccf57035ae519d308a998518e4c2c2a1babe7d5e9d.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/b01d6627a9fefbfcc29892ccf57035ae519d308a998518e4c2c2a1babe7d5e9d.jpg)

![b541829e174d194b5283da92f9e5f65a192c9b796625a95a26f3919e5fc9d07b.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/b541829e174d194b5283da92f9e5f65a192c9b796625a95a26f3919e5fc9d07b.jpg)

![bd777e01fb22a525a3b451b55f62b7fdffa0e69e0107d20b1ac418644e9393e8.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/bd777e01fb22a525a3b451b55f62b7fdffa0e69e0107d20b1ac418644e9393e8.jpg)

![bf46438e14837b8d9b1fec042261f9b99c7b6a0a25ba950422a4f359b68bca69.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/bf46438e14837b8d9b1fec042261f9b99c7b6a0a25ba950422a4f359b68bca69.jpg)

![c0115966f0a2750cfe0f0477679e088ba68cebdcdf32bc58c4d43da46d532dcb.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/c0115966f0a2750cfe0f0477679e088ba68cebdcdf32bc58c4d43da46d532dcb.jpg)

![c2cdacaf2b8844ac26c353ad14794d158fec5ac7bcc60ec2022a2a61ff804ddf.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/c2cdacaf2b8844ac26c353ad14794d158fec5ac7bcc60ec2022a2a61ff804ddf.jpg)

![c45e025b8f570180c6c87acd64bef9a0090a474af82280f0aa48b7bf6d8b2092.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/c45e025b8f570180c6c87acd64bef9a0090a474af82280f0aa48b7bf6d8b2092.jpg)

![c5a67ec49ccd60bcf530a26924c3a084c0eb78d65c69fc85399da8ea93c5cdd5.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/c5a67ec49ccd60bcf530a26924c3a084c0eb78d65c69fc85399da8ea93c5cdd5.jpg)

![e1dbbc4325b1a90075d7b8fe852d7dfdaead8fce0db7f3259fae411f1ff84a5e.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/e1dbbc4325b1a90075d7b8fe852d7dfdaead8fce0db7f3259fae411f1ff84a5e.jpg)

![e65b354abe557000873c0f70824b293a492927c9490ed0a180733598b5e60b02.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/e65b354abe557000873c0f70824b293a492927c9490ed0a180733598b5e60b02.jpg)

![ecd1bf62a9c3847ce02b7557f9e368002dc188ff97ebbfa28f9bd827ae417615.jpg](../icml_results/273_Iterative%20Refinement%20Neural%20Operators%20are%20Learned%20Fixed-Point%20Solvers_%20A%20Principled%20Approach%20to%20Spec/tables/ecd1bf62a9c3847ce02b7557f9e368002dc188ff97ebbfa28f9bd827ae417615.jpg)

<a id="paper-0269"></a>

## From Denoising to De-Channeling: Integrating Physical Channel Priors into Diffusion Models for Radio Signal Understanding


### Images

![16f0cc9ffdcfc2e0d7c6cb93e19650cc139212fdc19430483b26005b606a9339.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/16f0cc9ffdcfc2e0d7c6cb93e19650cc139212fdc19430483b26005b606a9339.jpg)

![2320ad0f3675a85c4d4112c0998fc1946b454d9890f6b99d9fdb9fdcdf4dc779.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/2320ad0f3675a85c4d4112c0998fc1946b454d9890f6b99d9fdb9fdcdf4dc779.jpg)

![4133e67aa172c6b0d91183cd47a30b6d825a08a34ccfd53556e182f247d9deac.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/4133e67aa172c6b0d91183cd47a30b6d825a08a34ccfd53556e182f247d9deac.jpg)

![48852c2a46247f300b3c9d519aa8289087e571728cf66953524204184a626f90.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/48852c2a46247f300b3c9d519aa8289087e571728cf66953524204184a626f90.jpg)

![8c78ea320aab1233d9438b4df5ce583d9706694d4c8354f59ce6f36121dd67fc.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/8c78ea320aab1233d9438b4df5ce583d9706694d4c8354f59ce6f36121dd67fc.jpg)

![8fefc8f265fadf36029ec7cc9723d9a9a0e854c68656ec0b1cf9daa043379aee.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/8fefc8f265fadf36029ec7cc9723d9a9a0e854c68656ec0b1cf9daa043379aee.jpg)

![9dfa1fbced93574cf3a32bef1df05581d1b4aafe3737e82e5dd2402852a8468e.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/9dfa1fbced93574cf3a32bef1df05581d1b4aafe3737e82e5dd2402852a8468e.jpg)

![b02ad3a589be685d32886299d4c009268f1a17cfab448f0845d667bd799eff45.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/b02ad3a589be685d32886299d4c009268f1a17cfab448f0845d667bd799eff45.jpg)

![b3f63d802c61ea2ec2736c8f4cdf94a07bed8c922ec9d1a0943775a316b6fb9b.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/b3f63d802c61ea2ec2736c8f4cdf94a07bed8c922ec9d1a0943775a316b6fb9b.jpg)

![dd2b49c58abd5bffe3d91dd9be42edb1e2dd3b1ffb5c8d1761cc8a6a3fd0aca4.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/images/dd2b49c58abd5bffe3d91dd9be42edb1e2dd3b1ffb5c8d1761cc8a6a3fd0aca4.jpg)

### Tables

![10f0a585dd2236e74616ebfce118c3a2a02669245b4d1a12560225ef745229c6.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/10f0a585dd2236e74616ebfce118c3a2a02669245b4d1a12560225ef745229c6.jpg)

![5035eb5d82c01f1100c96b2e271313db8e71b6fcf18bf82712235b6925151aef.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/5035eb5d82c01f1100c96b2e271313db8e71b6fcf18bf82712235b6925151aef.jpg)

![623b2aac1960a963a54828f6d44cd3f48ee554996bef4d81b833c8a83a2dafbe.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/623b2aac1960a963a54828f6d44cd3f48ee554996bef4d81b833c8a83a2dafbe.jpg)

![9a0f80d9b1638e417633f44531433cbb28c1b1b988c0fa8548073f5830414991.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/9a0f80d9b1638e417633f44531433cbb28c1b1b988c0fa8548073f5830414991.jpg)

![9c163cd489a0e18d0a298e3e2b4c1050a2f8a4ae97c1f4570c58b46850b99857.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/9c163cd489a0e18d0a298e3e2b4c1050a2f8a4ae97c1f4570c58b46850b99857.jpg)

![a44423137bf24a397e97ff4e1e0503262a2b6cbe0c46ba4adfa643f174250661.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/a44423137bf24a397e97ff4e1e0503262a2b6cbe0c46ba4adfa643f174250661.jpg)

![a44fd6f5a3aa60f80f3052dfcc10f296c702e769f2c0a48ebc2bde04a0032a53.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/a44fd6f5a3aa60f80f3052dfcc10f296c702e769f2c0a48ebc2bde04a0032a53.jpg)

![f7a4da393cb595060e085c2d58bfaa5f700d336830d68f6aadb6c56084155b8e.jpg](../icml_results/274_From%20Denoising%20to%20De-Channeling_%20Integrating%20Physical%20Channel%20Priors%20into%20Diffusion%20Models%20for%20Radio/tables/f7a4da393cb595060e085c2d58bfaa5f700d336830d68f6aadb6c56084155b8e.jpg)

<a id="paper-0270"></a>

## TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance


### Images

![0ac3562d01c63b3f714982eaf6400a929aa9f629af16ef57ea6b05f9ef910b0a.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/0ac3562d01c63b3f714982eaf6400a929aa9f629af16ef57ea6b05f9ef910b0a.jpg)

![0ffd4fa1c571857159407f070c82410a0bfa9bb52594ce9d0036d2f3d57f1569.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/0ffd4fa1c571857159407f070c82410a0bfa9bb52594ce9d0036d2f3d57f1569.jpg)

![11fd409df8bfd444dfed86a35f7dcc467d8a2bccde68cce27801c6d025811319.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/11fd409df8bfd444dfed86a35f7dcc467d8a2bccde68cce27801c6d025811319.jpg)

![1416aac162fe5283b726182a70ca99277d09c8bcfbd0b05b12dd0a09e53fc283.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/1416aac162fe5283b726182a70ca99277d09c8bcfbd0b05b12dd0a09e53fc283.jpg)

![22d6ffda2a5e61fde794dcf0171c25690606d0e23f3022281699b51d7ef5e1ce.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/22d6ffda2a5e61fde794dcf0171c25690606d0e23f3022281699b51d7ef5e1ce.jpg)

![273fa393bebee55767f00ac717c1cc5dc73179cbd52298f82aba88e92214f6d5.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/273fa393bebee55767f00ac717c1cc5dc73179cbd52298f82aba88e92214f6d5.jpg)

![44ddcd4ba253d1b80ae0c5660b335014c59c1b4f39ebdcb3d0daf8d926a6097b.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/44ddcd4ba253d1b80ae0c5660b335014c59c1b4f39ebdcb3d0daf8d926a6097b.jpg)

![44e4997e12c36fb6a958b45ac09156d35804a6da9fcde1fdc87a2d311986c961.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/44e4997e12c36fb6a958b45ac09156d35804a6da9fcde1fdc87a2d311986c961.jpg)

![45a22e98e0a1d1f6a71a1ddea0757391b87e0a6f5242fb8f088aa448aaf27817.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/45a22e98e0a1d1f6a71a1ddea0757391b87e0a6f5242fb8f088aa448aaf27817.jpg)

![584d9b91864b21284c07665246496a3e7ad0bdcf981a4bf0ce05679873d5d7d7.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/584d9b91864b21284c07665246496a3e7ad0bdcf981a4bf0ce05679873d5d7d7.jpg)

![6a846143148abfe9cef8ebc290c078538513524cda6e712d65deab826332cae0.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/6a846143148abfe9cef8ebc290c078538513524cda6e712d65deab826332cae0.jpg)

![6cae6af95a4252b0e1af7e67784989f1f1b814a04261c76a1595b61e3d8a8c16.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/6cae6af95a4252b0e1af7e67784989f1f1b814a04261c76a1595b61e3d8a8c16.jpg)

![a03e55b6e9ae968a8f666a99af6bd5b258d04958935c335de45f07e64cf15989.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/a03e55b6e9ae968a8f666a99af6bd5b258d04958935c335de45f07e64cf15989.jpg)

![b0af84ec3cfdb02835b5a6f7f7407fc93ba57f8f159845b0c15136544f2aa2dd.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/b0af84ec3cfdb02835b5a6f7f7407fc93ba57f8f159845b0c15136544f2aa2dd.jpg)

![b4e825c6c8849bc6e0a272c544cc6b245fa65d44fb3d78b2696c18c98229dbdb.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/b4e825c6c8849bc6e0a272c544cc6b245fa65d44fb3d78b2696c18c98229dbdb.jpg)

![cd422dd324028c9512efc2af8417e8337de8c8a1ac33f26c59499efca66805d2.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/cd422dd324028c9512efc2af8417e8337de8c8a1ac33f26c59499efca66805d2.jpg)

![d9cf213543f104688e9dbf2251b28a38758ff067728c6c46f2def047ac53c30b.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/d9cf213543f104688e9dbf2251b28a38758ff067728c6c46f2def047ac53c30b.jpg)

![dcf31fbc14e998c9440e84e36931991c602f6ad218b6356ed1030b64853e1718.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/dcf31fbc14e998c9440e84e36931991c602f6ad218b6356ed1030b64853e1718.jpg)

![feb993305ec0615128e58869237a7f976c7b0fbf6294fd0c34c1f21a5fa623ca.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/feb993305ec0615128e58869237a7f976c7b0fbf6294fd0c34c1f21a5fa623ca.jpg)

![ff50a62bb7174f898dd5cad1873926805560f7ba7b0b5aa97d245712813fe58b.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/images/ff50a62bb7174f898dd5cad1873926805560f7ba7b0b5aa97d245712813fe58b.jpg)

### Tables

![933c4be7101e9578ca292327003beef472f1a8b99adfab9ef94d608fdecb6a7a.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/tables/933c4be7101e9578ca292327003beef472f1a8b99adfab9ef94d608fdecb6a7a.jpg)

![97933b6d397f4abccedc6445eba9d2987d64699ffbec73ccb542b3abf5fcab84.jpg](../icml_results/275_TimeRewarder_%20Learning%20Dense%20Reward%20from%20Passive%20Videos%20via%20Frame-wise%20Temporal%20Distance/tables/97933b6d397f4abccedc6445eba9d2987d64699ffbec73ccb542b3abf5fcab84.jpg)

<a id="paper-0271"></a>

## Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning


### Images

![06dadfd4c24114d0af57f4aa2133bc5b46b573409be35dee7ee2128ddc3384cb.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/06dadfd4c24114d0af57f4aa2133bc5b46b573409be35dee7ee2128ddc3384cb.jpg)

![2f42cf624fc89678d5b69196d67732aab735776075d8aa319b015e933013c966.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/2f42cf624fc89678d5b69196d67732aab735776075d8aa319b015e933013c966.jpg)

![3a537a5ee13f43cfcb07cc3b581adf1625c39a32a050d91b3e9791771d029cc0.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/3a537a5ee13f43cfcb07cc3b581adf1625c39a32a050d91b3e9791771d029cc0.jpg)

![5a0e417913c8c6394ec62f00d5fcf7b8d5c337759ae393663846adf2c11a6b3e.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/5a0e417913c8c6394ec62f00d5fcf7b8d5c337759ae393663846adf2c11a6b3e.jpg)

![618a2a1504f5aa998378f48b9881684360c52453bb36a2b3e170fc576ab2569f.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/618a2a1504f5aa998378f48b9881684360c52453bb36a2b3e170fc576ab2569f.jpg)

![7ece3fffc26b165bcec9b0c19d908f8a49c187b2ea63e2620cf1ba52c9f41170.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/7ece3fffc26b165bcec9b0c19d908f8a49c187b2ea63e2620cf1ba52c9f41170.jpg)

![7ee07d05947ac8f865539ab6d97eb4671469ea52f91be1d5c533db7c765ca92a.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/7ee07d05947ac8f865539ab6d97eb4671469ea52f91be1d5c533db7c765ca92a.jpg)

![849e6b0bd9b0f3a6ed71d461ba172e51365e990966ff73e4b2ea669221713044.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/849e6b0bd9b0f3a6ed71d461ba172e51365e990966ff73e4b2ea669221713044.jpg)

![8e2bd3552ff5f8ffe7e62ae88525fd0b4f2641880a6daf8561a6583c41c6854f.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/8e2bd3552ff5f8ffe7e62ae88525fd0b4f2641880a6daf8561a6583c41c6854f.jpg)

![a5c63dd5e119e2525beaba31eeb204d3372f082576603ef96d33755a5e01af26.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/a5c63dd5e119e2525beaba31eeb204d3372f082576603ef96d33755a5e01af26.jpg)

![c74c069ebadffa9e6934100da165de3542743b45851538b940f5f7a6e6f4913a.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/c74c069ebadffa9e6934100da165de3542743b45851538b940f5f7a6e6f4913a.jpg)

![ef14a4e66ce2b019d03c26f8ebe2392e48d88c7c36c2180c9e4261055646bcfa.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/ef14a4e66ce2b019d03c26f8ebe2392e48d88c7c36c2180c9e4261055646bcfa.jpg)

![efa2a877c960e009616c24c16951d6e2ceb08f81c96cd909829767ea0cafd149.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/images/efa2a877c960e009616c24c16951d6e2ceb08f81c96cd909829767ea0cafd149.jpg)

### Tables

![206def3bd99b66b333e73fbdd442846e1b7202e5675510fb9f5bac41ea76de8a.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/tables/206def3bd99b66b333e73fbdd442846e1b7202e5675510fb9f5bac41ea76de8a.jpg)

![37693f6a4fd8568fdb96c65bdb9bbc0b86cfd4ad96fb8f68a26327246bbf7858.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/tables/37693f6a4fd8568fdb96c65bdb9bbc0b86cfd4ad96fb8f68a26327246bbf7858.jpg)

![58b707497353b19c82121e57273afbed167c1523003b77c259d01d3d9b2861d4.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/tables/58b707497353b19c82121e57273afbed167c1523003b77c259d01d3d9b2861d4.jpg)

![8a1618719082c7978a1aef7d8364faeab90e9e1719a34b81c819dae75025afdf.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/tables/8a1618719082c7978a1aef7d8364faeab90e9e1719a34b81c819dae75025afdf.jpg)

![ec5b966dd2490dbc79e52109b9fd58d32673de6cb73b8c9a57ffd4cee663fa84.jpg](../icml_results/276_Agent0-VL_%20Exploring%20Self-Evolving%20Agent%20for%20Tool-Integrated%20Vision-Language%20Reasoning/tables/ec5b966dd2490dbc79e52109b9fd58d32673de6cb73b8c9a57ffd4cee663fa84.jpg)

<a id="paper-0272"></a>

## Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection


### Images

![05244ccbc82dfa019fdaee15a654cbf0ecb27cb88a5c487012272f5824fad215.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/05244ccbc82dfa019fdaee15a654cbf0ecb27cb88a5c487012272f5824fad215.jpg)

![2f3d72f98e119225d5a29977acad231b0b1ef2812ebc15eccecfe287e4b40fac.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/2f3d72f98e119225d5a29977acad231b0b1ef2812ebc15eccecfe287e4b40fac.jpg)

![33fba8695d8b7bd06af3ed661f71932563bf27bf986b4feea91248df626f0748.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/33fba8695d8b7bd06af3ed661f71932563bf27bf986b4feea91248df626f0748.jpg)

![4af81464343fa859a07a092d7fa2b175e0abaef0ec11012591b7be15e3f102d0.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/4af81464343fa859a07a092d7fa2b175e0abaef0ec11012591b7be15e3f102d0.jpg)

![4b1b98301d6cf903754fbd4959ed34a8784899e170f0c983ae84ecde83940bd5.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/4b1b98301d6cf903754fbd4959ed34a8784899e170f0c983ae84ecde83940bd5.jpg)

![504a728dc67eff6494fb1c78dee0357661ec2bff046e14571088c8cb1e1a7fe8.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/504a728dc67eff6494fb1c78dee0357661ec2bff046e14571088c8cb1e1a7fe8.jpg)

![7134366c8b637eb89997b2cb9b0d537fe1df6e0ef82082d085c1c25a62be576b.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/7134366c8b637eb89997b2cb9b0d537fe1df6e0ef82082d085c1c25a62be576b.jpg)

![83d46066d3d092e189e73bfc915800266a819589eeb792881c1eef9ff8ee63d0.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/83d46066d3d092e189e73bfc915800266a819589eeb792881c1eef9ff8ee63d0.jpg)

![9d1c5dbbca082ca84872f4c31386f56c3ffd3b15b53dab926dd95ba13bf876b1.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/9d1c5dbbca082ca84872f4c31386f56c3ffd3b15b53dab926dd95ba13bf876b1.jpg)

![ccc452f6e1852c1d00c1fc77ed34cb6a82c546188683176acfea2dbae16e8544.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/ccc452f6e1852c1d00c1fc77ed34cb6a82c546188683176acfea2dbae16e8544.jpg)

![d5214a4b6252f6ba77fb19be11e924c7171032a603a296861e8155a24620cca0.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/d5214a4b6252f6ba77fb19be11e924c7171032a603a296861e8155a24620cca0.jpg)

![d8e2794f454d051639852c85f2e8cf6f74de2514013b070e1810d0f0e7f834d9.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/d8e2794f454d051639852c85f2e8cf6f74de2514013b070e1810d0f0e7f834d9.jpg)

![e2206c5119f1216f797a41f64ce7335a0eef4cecddd3517ed048c47e1ad0a110.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/e2206c5119f1216f797a41f64ce7335a0eef4cecddd3517ed048c47e1ad0a110.jpg)

![f97ad963e59b24a073cf86a159ef8cb74e07bccc83166e3f53cc9b03b9bc6803.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/f97ad963e59b24a073cf86a159ef8cb74e07bccc83166e3f53cc9b03b9bc6803.jpg)

![feaeced21aaf90b1a6805e5312b8741080b9a59a0d3ca87966a306a61b7f9408.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/images/feaeced21aaf90b1a6805e5312b8741080b9a59a0d3ca87966a306a61b7f9408.jpg)

### Tables

![0aa7c160ac2fdffb73ae1000116d1017907eddac9502e9b8a91720a593a925c3.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/0aa7c160ac2fdffb73ae1000116d1017907eddac9502e9b8a91720a593a925c3.jpg)

![35a818cea5c3bddf9b862a410ae75b9207276c0743afa99fa0dd5f503adf8e55.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/35a818cea5c3bddf9b862a410ae75b9207276c0743afa99fa0dd5f503adf8e55.jpg)

![47be2264fbb759a1e8305e2161a6ce35ec778ec9232900a1f78941f0b6e84b88.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/47be2264fbb759a1e8305e2161a6ce35ec778ec9232900a1f78941f0b6e84b88.jpg)

![9defd6aa5e57e67cde27a71ead5dd9da427978a5ae4fd20107a9a4784511fa14.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/9defd6aa5e57e67cde27a71ead5dd9da427978a5ae4fd20107a9a4784511fa14.jpg)

![b0f80709a34ed6002a275cd5376f808d9656f98696f8f7c84e40fec045b3cecc.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/b0f80709a34ed6002a275cd5376f808d9656f98696f8f7c84e40fec045b3cecc.jpg)

![d6cb403668ad737bc30495cc851ee65592764a46796f9b129ad68357c0354463.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/d6cb403668ad737bc30495cc851ee65592764a46796f9b129ad68357c0354463.jpg)

![d84a52715327f1e56d91adb138359fb8d99a74df6bf2a87dabd41fa76b5dd0c8.jpg](../icml_results/277_Diffuse%20to%20Detect_%20Bi-Level%20Sample%20Rebalancing%20with%20Pseudo-Label%20Diffusion%20for%20Point-Supervised%20Infr/tables/d84a52715327f1e56d91adb138359fb8d99a74df6bf2a87dabd41fa76b5dd0c8.jpg)

<a id="paper-0273"></a>

## TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention


### Images

![0ea79f0a06ba82cef379d761e30e6ce24d514ff3901e963d2d545a78c08ada2c.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/0ea79f0a06ba82cef379d761e30e6ce24d514ff3901e963d2d545a78c08ada2c.jpg)

![114c0bb60b92b6696e17740e68add2decc5a0736cc2b7b08d7fbb06617e36dad.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/114c0bb60b92b6696e17740e68add2decc5a0736cc2b7b08d7fbb06617e36dad.jpg)

![1d37d4ccbd92b6b26d7325bc47ed80ef1e517780b42be4c60ed25d47db4fef0c.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/1d37d4ccbd92b6b26d7325bc47ed80ef1e517780b42be4c60ed25d47db4fef0c.jpg)

![3546b47a92df1d425176d71e254d08b2090ab09af88c2f9d08e15813ae8f238d.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/3546b47a92df1d425176d71e254d08b2090ab09af88c2f9d08e15813ae8f238d.jpg)

![3cee697bf201e3a9ce3f9e445d83b3b48b16953904f38a2bc644fba7b33adcb4.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/3cee697bf201e3a9ce3f9e445d83b3b48b16953904f38a2bc644fba7b33adcb4.jpg)

![3dd39d741664cbfebffc61698e25716d0340a3838d3623643c80266cd8d59d64.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/3dd39d741664cbfebffc61698e25716d0340a3838d3623643c80266cd8d59d64.jpg)

![4bda814e5f1e3e441c754e17b2b195cf549f395691bdee8704d6483434f1c5bc.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/4bda814e5f1e3e441c754e17b2b195cf549f395691bdee8704d6483434f1c5bc.jpg)

![bb2d56d2b0c9d217332d411e9d468455d253eb0ada8ab245fcab13f30e46d201.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/bb2d56d2b0c9d217332d411e9d468455d253eb0ada8ab245fcab13f30e46d201.jpg)

![bdc13032a54f0b0aa72048d2ed2038a3e237e65907ab4486c36302cd69a4a748.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/images/bdc13032a54f0b0aa72048d2ed2038a3e237e65907ab4486c36302cd69a4a748.jpg)

### Tables

![0569b3d275093bb71dee1bfa643bb34548f4e2804694460fa064d470a1a6ab2c.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/tables/0569b3d275093bb71dee1bfa643bb34548f4e2804694460fa064d470a1a6ab2c.jpg)

![27e1bf6656b5eeb7754fdff7352bb5319891a24dac5402f70333404b3cae2261.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/tables/27e1bf6656b5eeb7754fdff7352bb5319891a24dac5402f70333404b3cae2261.jpg)

![55106cdb0e6b7f9e77997f8f17503368c3c6efd5f7a442bd8b254b80e2e88b76.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/tables/55106cdb0e6b7f9e77997f8f17503368c3c6efd5f7a442bd8b254b80e2e88b76.jpg)

![d49a2d02f90537d779b9054c5a2d4d7d9b6f0f43fac3a8caaff1e50e6085945c.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/tables/d49a2d02f90537d779b9054c5a2d4d7d9b6f0f43fac3a8caaff1e50e6085945c.jpg)

![d5b2e5035d59f286859bc0df7bf747e83c6acfe639db6c2cfe090974a78ace11.jpg](../icml_results/278_TabSwift_%20An%20Efficient%20Tabular%20Foundation%20Model%20with%20Row-Wise%20Attention/tables/d5b2e5035d59f286859bc0df7bf747e83c6acfe639db6c2cfe090974a78ace11.jpg)

<a id="paper-0274"></a>

## Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models


### Images

![2449c0d41721e603ce62e7af6cd5dfab93d48d64481bfefa4e3b333ae5c2cd06.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/images/2449c0d41721e603ce62e7af6cd5dfab93d48d64481bfefa4e3b333ae5c2cd06.jpg)

![420d5409937cfb38e260ff8bbabc2b3bc5224b52fa70ef044e0f91e2df2f5311.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/images/420d5409937cfb38e260ff8bbabc2b3bc5224b52fa70ef044e0f91e2df2f5311.jpg)

![72129ee04184747c028c9e0428bc54103a3f3da62c61fc18cc431ef9fefca5dd.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/images/72129ee04184747c028c9e0428bc54103a3f3da62c61fc18cc431ef9fefca5dd.jpg)

![901ede6232d01559d76b56ed400a9f6210e40bb1caf777e5c8141c080c5a6d4c.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/images/901ede6232d01559d76b56ed400a9f6210e40bb1caf777e5c8141c080c5a6d4c.jpg)

### Tables

![0e463006b5f00700f389c11555ddcbed1b23e38d7e07329e0a2acf9e24326e1a.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/0e463006b5f00700f389c11555ddcbed1b23e38d7e07329e0a2acf9e24326e1a.jpg)

![10d679361d2981833f524a522098bcb0ae41468043b7b503a33744f7d53d1581.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/10d679361d2981833f524a522098bcb0ae41468043b7b503a33744f7d53d1581.jpg)

![13fa306e37732e84118fceafa95d3024933c0ba3721ad296e7d8504cc3c13d36.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/13fa306e37732e84118fceafa95d3024933c0ba3721ad296e7d8504cc3c13d36.jpg)

![459f10efcf3e2f98fdb90a32fa35ef9067874665128f79e7ef5c9d4d2b957ffd.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/459f10efcf3e2f98fdb90a32fa35ef9067874665128f79e7ef5c9d4d2b957ffd.jpg)

![74fedc17b435867b052e113e9dd566de928efadebd4c3fe3d35ec17c497ff6c7.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/74fedc17b435867b052e113e9dd566de928efadebd4c3fe3d35ec17c497ff6c7.jpg)

![b771824d42d7ff8af817e3e62747747de8ad9aa02a0e7341ede85756ae74521b.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/b771824d42d7ff8af817e3e62747747de8ad9aa02a0e7341ede85756ae74521b.jpg)

![bc13f6bb8f8580362b13e9caf944df71e9f7285a5af8a7c32da027c6577d4b81.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/bc13f6bb8f8580362b13e9caf944df71e9f7285a5af8a7c32da027c6577d4b81.jpg)

![c007897c58fc09e0dc7c03d7b0e18f132a18353b99eec15a0a75b78944a4d311.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/c007897c58fc09e0dc7c03d7b0e18f132a18353b99eec15a0a75b78944a4d311.jpg)

![c098e01f384e0a552f2a22462a7a7d6d5d2678703630a2c9364be3c68c54be86.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/c098e01f384e0a552f2a22462a7a7d6d5d2678703630a2c9364be3c68c54be86.jpg)

![d3ffcee9397bbc3ec0c779245ea32cc5b4ef2699492e143b925f8fb935534891.jpg](../icml_results/279_Error%20Propagation%20Mechanisms%20and%20Compensation%20Strategies%20for%20Quantized%20Diffusion%20Models/tables/d3ffcee9397bbc3ec0c779245ea32cc5b4ef2699492e143b925f8fb935534891.jpg)

<a id="paper-0275"></a>

## Towards the Explainability of Temporal Graph Networks via Memory Backtracking and Topological Attribution


### Images

![182da65fca8c87921c035fe2bc888ed207b0d498ca7fffdd4baec2348d4b4b20.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/182da65fca8c87921c035fe2bc888ed207b0d498ca7fffdd4baec2348d4b4b20.jpg)

![1c47a763c402f19429cadf4398051158c3d566890db4b1afafad78c37ed36de7.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/1c47a763c402f19429cadf4398051158c3d566890db4b1afafad78c37ed36de7.jpg)

![2089913dafae955fc47ffa99f3269c7b1a46a7353e470c90e0b4b0aea8f12357.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/2089913dafae955fc47ffa99f3269c7b1a46a7353e470c90e0b4b0aea8f12357.jpg)

![22805fd4429d40f244d7b2f21b377c33b3d658652dc309853a35f3476beaebb9.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/22805fd4429d40f244d7b2f21b377c33b3d658652dc309853a35f3476beaebb9.jpg)

![2d8fd2c3311f507cb8560a6e9fffb5c53f855ba54dee9b794a4454cde403d1df.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/2d8fd2c3311f507cb8560a6e9fffb5c53f855ba54dee9b794a4454cde403d1df.jpg)

![33c3b2d7e2358b353c9cabd18a367cb2fc8da8b2572414888b41daff1218e371.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/33c3b2d7e2358b353c9cabd18a367cb2fc8da8b2572414888b41daff1218e371.jpg)

![7cffaa894abd6ad9cee5d0e86d505dec82b6a75f0a462e58a9d55a19e41ddd2a.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/7cffaa894abd6ad9cee5d0e86d505dec82b6a75f0a462e58a9d55a19e41ddd2a.jpg)

![8f6d3076ed9650e5f7b0ce859bd555c9f8806fbb77834a6e8353cf46f841eacb.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/8f6d3076ed9650e5f7b0ce859bd555c9f8806fbb77834a6e8353cf46f841eacb.jpg)

![9a6d6786226e02c026c8f7a4cf9d3c2dff297b149d75987f38a748c9b83adefa.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/9a6d6786226e02c026c8f7a4cf9d3c2dff297b149d75987f38a748c9b83adefa.jpg)

![9ae836c22faf9e93d9cc862f5207bd8083e46347224e36cc2ab3572c5f5ecdba.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/9ae836c22faf9e93d9cc862f5207bd8083e46347224e36cc2ab3572c5f5ecdba.jpg)

![a7bf0e6b7829e85a3cfe3f59aba405b35819345e5f347ea182eb8ed0d68d72f9.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/a7bf0e6b7829e85a3cfe3f59aba405b35819345e5f347ea182eb8ed0d68d72f9.jpg)

![afa3c11507e1fa7b5c19784bc31341112593ab8b3cc2199381e228355f245214.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/afa3c11507e1fa7b5c19784bc31341112593ab8b3cc2199381e228355f245214.jpg)

![c98d6d8affbd707e752574e3da7d31120c6ae412ddeb545a2f2fb4500b07d157.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/c98d6d8affbd707e752574e3da7d31120c6ae412ddeb545a2f2fb4500b07d157.jpg)

![cb5d41a354b5a232d84fbfea43bf7c9f24741ed4169c42550ecacc2956987269.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/cb5d41a354b5a232d84fbfea43bf7c9f24741ed4169c42550ecacc2956987269.jpg)

![d50d51564533d5755857ae299ab5adc3c9557e4d07788263ee4db8ebf01b335e.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/d50d51564533d5755857ae299ab5adc3c9557e4d07788263ee4db8ebf01b335e.jpg)

![d92e06a808ca17906ed25385ed1381628e2839d2a649b996f7c16d23b1e9f3da.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/d92e06a808ca17906ed25385ed1381628e2839d2a649b996f7c16d23b1e9f3da.jpg)

![e98d6bd72f8c1c17f27e22a72a61ab568642400be3e962b1f9092ad909c6bc9f.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/images/e98d6bd72f8c1c17f27e22a72a61ab568642400be3e962b1f9092ad909c6bc9f.jpg)

### Tables

![08d60d4ef85654d7acca219da358e71c6bef0f6e31891ddaa64c83b096295e85.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/08d60d4ef85654d7acca219da358e71c6bef0f6e31891ddaa64c83b096295e85.jpg)

![134eb63b5fdd5af1dcdd91c67256821a962bbb49597767ddc7681e2a0753a0e7.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/134eb63b5fdd5af1dcdd91c67256821a962bbb49597767ddc7681e2a0753a0e7.jpg)

![330ac589277af8353435223e018f179387a5a0c6e440368c5d839fb6df5cc5fd.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/330ac589277af8353435223e018f179387a5a0c6e440368c5d839fb6df5cc5fd.jpg)

![3943a6d2d2ae3ce062e41896345105d6f256af386ec098efbf4c2be7cda8a10b.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/3943a6d2d2ae3ce062e41896345105d6f256af386ec098efbf4c2be7cda8a10b.jpg)

![41900ed0b1f6caaee33d73e6bc042db423b3c17cc833160c5014e3060341ba53.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/41900ed0b1f6caaee33d73e6bc042db423b3c17cc833160c5014e3060341ba53.jpg)

![56ba2406d23830a1bd7d953fd7521904747142ae34e9b941ef6a6819003839d4.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/56ba2406d23830a1bd7d953fd7521904747142ae34e9b941ef6a6819003839d4.jpg)

![c2b0652a88eba322798944d4426287269ea4ae8c58327507c3d4f229f980f970.jpg](../icml_results/280_Towards%20the%20Explainability%20of%20Temporal%20Graph%20Networks%20via%20Memory%20Backtracking%20and%20Topological%20Attrib/tables/c2b0652a88eba322798944d4426287269ea4ae8c58327507c3d4f229f980f970.jpg)

<a id="paper-0276"></a>

## Balancing Understanding and Generation in Discrete Diffusion Models


### Images

![06535e8158a1f29fa4ca780fd984345eebb0a6940a53401a781970980a1a1b71.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/06535e8158a1f29fa4ca780fd984345eebb0a6940a53401a781970980a1a1b71.jpg)

![33ae7a8aeeaf00e9ed235716d979ec955974ce35cab002a9ecf9cf876c2a998e.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/33ae7a8aeeaf00e9ed235716d979ec955974ce35cab002a9ecf9cf876c2a998e.jpg)

![3c1c416985bb48bfaa1edc26fd68620aa6d3bc553117c4477c24bb7fbb93c957.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/3c1c416985bb48bfaa1edc26fd68620aa6d3bc553117c4477c24bb7fbb93c957.jpg)

![3dd5dce6fe378c332456153c5d2a1d4372189f486ac92cd36a3b8e3b60049936.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/3dd5dce6fe378c332456153c5d2a1d4372189f486ac92cd36a3b8e3b60049936.jpg)

![457ed08a918e06e9f8271eec9dc28da535578213f80df0fcb2144a4cd2ac3755.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/457ed08a918e06e9f8271eec9dc28da535578213f80df0fcb2144a4cd2ac3755.jpg)

![4b108d8d87e367067094f09a84f7c80d73fb577b3348b88f6441838b9afbe0f4.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/4b108d8d87e367067094f09a84f7c80d73fb577b3348b88f6441838b9afbe0f4.jpg)

![880cb14a91af8420158a759b3476f16e9a1905a53f4198f93bb9eeb28acef014.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/880cb14a91af8420158a759b3476f16e9a1905a53f4198f93bb9eeb28acef014.jpg)

![8afe44d043560989874cfc761181c1da031bd055d092cd2e7fb0b79e2eb18539.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/8afe44d043560989874cfc761181c1da031bd055d092cd2e7fb0b79e2eb18539.jpg)

![a4e14df0f8d80456e69787c408162a178322f5d7d2e74b4a9fb49a547c312191.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/a4e14df0f8d80456e69787c408162a178322f5d7d2e74b4a9fb49a547c312191.jpg)

![ee35fe54a509b391a9b5244e86108573301e0017088b0181e9973760c2f1e171.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/images/ee35fe54a509b391a9b5244e86108573301e0017088b0181e9973760c2f1e171.jpg)

### Tables

![03415bd04a3f333cb5eb5d0cde90e9c73e1817b1c537915f708037eb341389d4.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/03415bd04a3f333cb5eb5d0cde90e9c73e1817b1c537915f708037eb341389d4.jpg)

![078e06e7d6c68808b225df2c97d06e81ca5470233277bf71868d727f01cd5db8.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/078e06e7d6c68808b225df2c97d06e81ca5470233277bf71868d727f01cd5db8.jpg)

![07f2d29313b16361e231252681c458f3472433829712476144cc3f5d117e0e0f.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/07f2d29313b16361e231252681c458f3472433829712476144cc3f5d117e0e0f.jpg)

![1312a05972d28b20e30cf84e363adbdc6b7f6f983539c30e3a03d8c92ae934a0.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/1312a05972d28b20e30cf84e363adbdc6b7f6f983539c30e3a03d8c92ae934a0.jpg)

![1a355328bee5a9d575dda3cbf6d07d30c9592594bac6e625807559ba3a1ee9c1.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/1a355328bee5a9d575dda3cbf6d07d30c9592594bac6e625807559ba3a1ee9c1.jpg)

![1c91dda6b91ed8273b992f68c40a14931f743c00c3ecfb0bba2aae3cf7c24588.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/1c91dda6b91ed8273b992f68c40a14931f743c00c3ecfb0bba2aae3cf7c24588.jpg)

![28761fa9afa8e9917bcb1eb4732702a56b383af7a7d3946f3f15a7f52851bf86.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/28761fa9afa8e9917bcb1eb4732702a56b383af7a7d3946f3f15a7f52851bf86.jpg)

![4c37a6e8f853bf702684305ad42b7d9c1cb9abd8b81430469e41bfffc180c6d2.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/4c37a6e8f853bf702684305ad42b7d9c1cb9abd8b81430469e41bfffc180c6d2.jpg)

![55d1b79e9168de93171253671e2b6fab9ba3fce2899ddcb12a18c78f40589b07.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/55d1b79e9168de93171253671e2b6fab9ba3fce2899ddcb12a18c78f40589b07.jpg)

![6b4b0a263adf10c0f1a475fe6572b6cd0435c3cdfe9042dd3bce27465ea0cd13.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/6b4b0a263adf10c0f1a475fe6572b6cd0435c3cdfe9042dd3bce27465ea0cd13.jpg)

![6d0a78b661df061a82d4dbbda890aaf81d525e53969cfc8db8bdd1b97031e40a.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/6d0a78b661df061a82d4dbbda890aaf81d525e53969cfc8db8bdd1b97031e40a.jpg)

![7db548e5225bddf568766c83e98afcbadac288b2f91a723e52e64a162f07ef73.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/7db548e5225bddf568766c83e98afcbadac288b2f91a723e52e64a162f07ef73.jpg)

![8235d9893edcefaab29821c98cff1182ddb58b9734b3e20c79c05324705363a7.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/8235d9893edcefaab29821c98cff1182ddb58b9734b3e20c79c05324705363a7.jpg)

![8565f69a6dc4ae3ce90ec8ccf9127a49b109f8e5add5ac1a031ac7077a43f9a2.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/8565f69a6dc4ae3ce90ec8ccf9127a49b109f8e5add5ac1a031ac7077a43f9a2.jpg)

![8b7eeebcc26731cb04e43fa5eaceaf6bbc42bb378a572746bafd30a36b762a71.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/8b7eeebcc26731cb04e43fa5eaceaf6bbc42bb378a572746bafd30a36b762a71.jpg)

![8f0758b85d8eb67dd4522d577735a3611c6ce4f1fb2655d9c249464cfac64a3d.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/8f0758b85d8eb67dd4522d577735a3611c6ce4f1fb2655d9c249464cfac64a3d.jpg)

![aa281600bb71ea0af1c9411e1af170783a44ad19aeb5c82c1913bd68e1cf4cd2.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/aa281600bb71ea0af1c9411e1af170783a44ad19aeb5c82c1913bd68e1cf4cd2.jpg)

![ca4cf7af1b85c5edae1cb153c596a3d476f2956035fac8b245dbbf4a658a0b23.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/ca4cf7af1b85c5edae1cb153c596a3d476f2956035fac8b245dbbf4a658a0b23.jpg)

![d9ef6d78d6dee9fe55048022a2b1654f4fa6bbdd194e8e0ef6c155e6e0e2b984.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/d9ef6d78d6dee9fe55048022a2b1654f4fa6bbdd194e8e0ef6c155e6e0e2b984.jpg)

![dee8de823d5c3ad55fa0a8da1b97531b2e1466a84abbe342cb3a2e4cf038d8b6.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/dee8de823d5c3ad55fa0a8da1b97531b2e1466a84abbe342cb3a2e4cf038d8b6.jpg)

![f0cc2b18833db37b4151514f701da0e28cbce57304c37c82e3c6ed45335a8b89.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/f0cc2b18833db37b4151514f701da0e28cbce57304c37c82e3c6ed45335a8b89.jpg)

![f619f06b97abbe0c80ad332ee3c3740eeff2da6a8f54ac1da77fd1abf5cce545.jpg](../icml_results/281_Balancing%20Understanding%20and%20Generation%20in%20Discrete%20Diffusion%20Models/tables/f619f06b97abbe0c80ad332ee3c3740eeff2da6a8f54ac1da77fd1abf5cce545.jpg)

<a id="paper-0277"></a>

## Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving


### Images

![063cad6277ab83573528c4d1bcf82e4b69e64dfc66a3fa7e5f58b03aed56aa2d.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/images/063cad6277ab83573528c4d1bcf82e4b69e64dfc66a3fa7e5f58b03aed56aa2d.jpg)

![1b62a4eefdd90223e9721786bea08b73c2255ba535f058315235cfc5b3c6cb15.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/images/1b62a4eefdd90223e9721786bea08b73c2255ba535f058315235cfc5b3c6cb15.jpg)

![7b859398e7c9a422eacc0b4e53fd0f418bd33a524a318e7f0326c51af2025405.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/images/7b859398e7c9a422eacc0b4e53fd0f418bd33a524a318e7f0326c51af2025405.jpg)

![a33dc693896cbadc0692f1b76fec1249c07948ecc83c1f9a1c61be445f1a8f36.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/images/a33dc693896cbadc0692f1b76fec1249c07948ecc83c1f9a1c61be445f1a8f36.jpg)

### Tables

![089ec5a8a7e42dcabad6ab26d5a2c864c00b0af2701644a3159a88ac82bdfb19.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/089ec5a8a7e42dcabad6ab26d5a2c864c00b0af2701644a3159a88ac82bdfb19.jpg)

![2a48036b667d7f117f4e9bdc9cb89b0e4fe96e0ce9aa6efcb0e4e5106c1b02c3.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/2a48036b667d7f117f4e9bdc9cb89b0e4fe96e0ce9aa6efcb0e4e5106c1b02c3.jpg)

![5b7ddcdea779e553d42671250272e2629dd53cb126bb5bddafcc03d5281e40d6.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/5b7ddcdea779e553d42671250272e2629dd53cb126bb5bddafcc03d5281e40d6.jpg)

![78b584a3a5c061c36e17e8418e4714901d99e0e7b7d1971326e6a57b9a361ea8.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/78b584a3a5c061c36e17e8418e4714901d99e0e7b7d1971326e6a57b9a361ea8.jpg)

![78ec97bb1bc424eda5649ae6eea1a38685e69e19bc1b3d1b1fa6dc15a3c53447.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/78ec97bb1bc424eda5649ae6eea1a38685e69e19bc1b3d1b1fa6dc15a3c53447.jpg)

![97400149322dc42edc5126c857ab4754da2ee6436047bebe0edc286fd6efa30c.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/97400149322dc42edc5126c857ab4754da2ee6436047bebe0edc286fd6efa30c.jpg)

![9efa812d731f13b92f158499e2dd7e197b80aa4bf45a2811bc53f1f78eefb6ae.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/9efa812d731f13b92f158499e2dd7e197b80aa4bf45a2811bc53f1f78eefb6ae.jpg)

![a8e529bd90e714f8fd96903a434450aa5d9434ec457db1f4a227c03c546afe47.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/a8e529bd90e714f8fd96903a434450aa5d9434ec457db1f4a227c03c546afe47.jpg)

![d5f930861079602e85d4d9ac7b4c6522cc601dffa135492e3bf4753201691eea.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/d5f930861079602e85d4d9ac7b4c6522cc601dffa135492e3bf4753201691eea.jpg)

![f1e08b90b96ff761dceebf85242a7c29d0b8cd8cd2040a530eacef6b67ffc130.jpg](../icml_results/282_Beyond%20Theorem%20Proving_%20Formulation%2C%20Framework%20and%20Benchmark%20for%20Formal%20Problem-Solving/tables/f1e08b90b96ff761dceebf85242a7c29d0b8cd8cd2040a530eacef6b67ffc130.jpg)

<a id="paper-0278"></a>

## Learning Biophysical Models of Large-Scale Multineuronal Data To Enable Precise Neurostimulation


### Images

![096c0023b83dd5016ddad5920537a905e119954f6dfe4ccb0dd118add02112b9.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/096c0023b83dd5016ddad5920537a905e119954f6dfe4ccb0dd118add02112b9.jpg)

![1db2f5350f5c1b46ebb779e2505b85c7e286790bc3ed408850680ad0762a1879.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/1db2f5350f5c1b46ebb779e2505b85c7e286790bc3ed408850680ad0762a1879.jpg)

![2737df183f59bd48ceba020efe4b60a9d35061d46d5d8553ba497feed7d1f6c0.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/2737df183f59bd48ceba020efe4b60a9d35061d46d5d8553ba497feed7d1f6c0.jpg)

![46088e68e6593542d6b822fa1ceced1414efb85f4186c381f49e15fe1997ccf8.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/46088e68e6593542d6b822fa1ceced1414efb85f4186c381f49e15fe1997ccf8.jpg)

![47364808fae780a44178d80911c4076b979fcb4f8298de3d7a6d9c1aabb8537d.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/47364808fae780a44178d80911c4076b979fcb4f8298de3d7a6d9c1aabb8537d.jpg)

![547c0d062204bc94db3f84c6e692c3e32551d53c586a1423167729777f522e3c.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/547c0d062204bc94db3f84c6e692c3e32551d53c586a1423167729777f522e3c.jpg)

![607341ff7f162e6bcbbbe28b4604df093b3683d17504ff77c7288982d0bdade6.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/607341ff7f162e6bcbbbe28b4604df093b3683d17504ff77c7288982d0bdade6.jpg)

![71005e1075b7b7bda0506b5948d9b479d857a0934c4a3c2d0dec5fad31dbdd91.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/71005e1075b7b7bda0506b5948d9b479d857a0934c4a3c2d0dec5fad31dbdd91.jpg)

![7f1199134ec8ca189326735c92f84eb452e0e9191d34a26f94415ff6a1495245.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/7f1199134ec8ca189326735c92f84eb452e0e9191d34a26f94415ff6a1495245.jpg)

![a381a08c482167ed0f996b71698d013f9d84d4c648074833227bdcda0ac5d12f.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/a381a08c482167ed0f996b71698d013f9d84d4c648074833227bdcda0ac5d12f.jpg)

![a68837ff8340e18369eb8adb3763d4b54238d0e80d337eae13f03149966018f9.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/a68837ff8340e18369eb8adb3763d4b54238d0e80d337eae13f03149966018f9.jpg)

![ab3ca980ced0d11d8f103df8f423894ebdb44443f1b3c8361ed8a81894e695e0.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/ab3ca980ced0d11d8f103df8f423894ebdb44443f1b3c8361ed8a81894e695e0.jpg)

![b5753b8e0a38dc52aab435c0b4f3d7f191c7bd66259682c3a9da580d3e92eee0.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/b5753b8e0a38dc52aab435c0b4f3d7f191c7bd66259682c3a9da580d3e92eee0.jpg)

![b73d306b92fcb64ee6c128c1749defbdcd516fc5c7441681f5af047dd9e98be7.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/b73d306b92fcb64ee6c128c1749defbdcd516fc5c7441681f5af047dd9e98be7.jpg)

![b96bdec8bbd981248b6ac2e2646281cfc023621a555d93d9b8802c4d19fa032d.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/b96bdec8bbd981248b6ac2e2646281cfc023621a555d93d9b8802c4d19fa032d.jpg)

![c17c61e1c04d73e0d91fa3e316e692957fed5665ca3b2f09a27073176de2bcb1.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/c17c61e1c04d73e0d91fa3e316e692957fed5665ca3b2f09a27073176de2bcb1.jpg)

![c25a67c47a9c2ae4106edc732d88d4742b0c10d187b99472678545836d33309c.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/c25a67c47a9c2ae4106edc732d88d4742b0c10d187b99472678545836d33309c.jpg)

![c4181c10572ff9386708b1e4a1bcee9db9b6c4d40b1051a278e44f63598c088e.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/c4181c10572ff9386708b1e4a1bcee9db9b6c4d40b1051a278e44f63598c088e.jpg)

![d55d920fa0c22ee39906abce74dd4afbd2f7f1d7cbb212a056f2af83571641e8.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/d55d920fa0c22ee39906abce74dd4afbd2f7f1d7cbb212a056f2af83571641e8.jpg)

![d7edbaf397c5b3e255ab0cc80b856d0eb32428c719c88de3b81b1387ac2086b8.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/d7edbaf397c5b3e255ab0cc80b856d0eb32428c719c88de3b81b1387ac2086b8.jpg)

![fefc3dd49f7a1231c736e55da9cd130ffc51408a017b146c51338dcd6a756544.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/images/fefc3dd49f7a1231c736e55da9cd130ffc51408a017b146c51338dcd6a756544.jpg)

### Tables

![0e771f83916ccef90d590b7e2acfdea5c832ffe747719b2beec9b597db6f914e.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/0e771f83916ccef90d590b7e2acfdea5c832ffe747719b2beec9b597db6f914e.jpg)

![1a441a97731baea9fc2d1bf5e83bbc56fc3dbe986e281ac9b966a584668299a3.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/1a441a97731baea9fc2d1bf5e83bbc56fc3dbe986e281ac9b966a584668299a3.jpg)

![2f1b47d49f14a5276c005919cc226134ccaeeb8f7fece1ac11cfff570d078fc4.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/2f1b47d49f14a5276c005919cc226134ccaeeb8f7fece1ac11cfff570d078fc4.jpg)

![3a9a6a376ae39a2001483add3db44190e8c149a55680db85cb7ade5d96d0eb86.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/3a9a6a376ae39a2001483add3db44190e8c149a55680db85cb7ade5d96d0eb86.jpg)

![4a07c47c507334afd20c77d434dc0755866833883a4a40e6eaa9c23e12fdd666.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/4a07c47c507334afd20c77d434dc0755866833883a4a40e6eaa9c23e12fdd666.jpg)

![5ef3ed668020efc1f0d5178a11026613bb1ad18ae3e7fb8a335f03b8b14a1183.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/5ef3ed668020efc1f0d5178a11026613bb1ad18ae3e7fb8a335f03b8b14a1183.jpg)

![62d49e92010d86fab76401f5d1378a871cb590e89758e4b36a0e0f072d505658.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/62d49e92010d86fab76401f5d1378a871cb590e89758e4b36a0e0f072d505658.jpg)

![7b7615f8466657d55bbdff740f78f41e3023a0418e92f842d2c816eafc78fdd2.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/7b7615f8466657d55bbdff740f78f41e3023a0418e92f842d2c816eafc78fdd2.jpg)

![803b2c2efab3f32f52417cd388f79ca5297542b3cd9b2233372edcb9fbb8e636.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/803b2c2efab3f32f52417cd388f79ca5297542b3cd9b2233372edcb9fbb8e636.jpg)

![a7848ef2fd680b22c429b60923c91f134372dbef1f8deb5018f87eea0f6e6580.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/a7848ef2fd680b22c429b60923c91f134372dbef1f8deb5018f87eea0f6e6580.jpg)

![ab60242c8e0dcb758c2602c610d00c069df3d921072fe48167716b715d9508fd.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/ab60242c8e0dcb758c2602c610d00c069df3d921072fe48167716b715d9508fd.jpg)

![bb2ca7fd3d6f1945720d83de4903b3a566e7582f6d4b06e084025507bafc54f1.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/bb2ca7fd3d6f1945720d83de4903b3a566e7582f6d4b06e084025507bafc54f1.jpg)

![c49392361f9b5d92ce587026c03201c07643cdcb20de4f5ee3830fa5ddd4edd5.jpg](../icml_results/283_Learning%20Biophysical%20Models%20of%20Large-Scale%20Multineuronal%20Data%20To%20Enable%20Precise%20Neurostimulation/tables/c49392361f9b5d92ce587026c03201c07643cdcb20de4f5ee3830fa5ddd4edd5.jpg)

<a id="paper-0279"></a>

## Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion


### Images

![001828c32d99e21503ddeef205c4e3a30228a3fedf9c494822de4970b63323a0.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/001828c32d99e21503ddeef205c4e3a30228a3fedf9c494822de4970b63323a0.jpg)

![1b093ef82e96316466a91f3c9d13fd972c3566d91c8f51641a5fba3a76530189.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/1b093ef82e96316466a91f3c9d13fd972c3566d91c8f51641a5fba3a76530189.jpg)

![1b6ec0aa8e6e8187a56c8ecbfdf55afc85d5f34a33caeab779b532a6117fb5c6.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/1b6ec0aa8e6e8187a56c8ecbfdf55afc85d5f34a33caeab779b532a6117fb5c6.jpg)

![2f397f323cdcf366d6cabbdfc7d823e79c940e88f5314dfaf4ee79ee444e3ec5.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/2f397f323cdcf366d6cabbdfc7d823e79c940e88f5314dfaf4ee79ee444e3ec5.jpg)

![3cd35f5be262d999fa8a2adb770a26ce63d45faac7c8c0b806fe8301f770fe46.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/3cd35f5be262d999fa8a2adb770a26ce63d45faac7c8c0b806fe8301f770fe46.jpg)

![4a1bdf66296d1857ef58944f36921fee2321febc2d854c91eae9db43994ed36e.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/4a1bdf66296d1857ef58944f36921fee2321febc2d854c91eae9db43994ed36e.jpg)

![56906d35713b6011d33cca810274f5ab02578ade8bd9d08c9530ef6f69093812.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/56906d35713b6011d33cca810274f5ab02578ade8bd9d08c9530ef6f69093812.jpg)

![5d1ed0ac8cb7e947065c8286b92ace581b7b7ee5b9211bea2a69d7886c4716ac.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/5d1ed0ac8cb7e947065c8286b92ace581b7b7ee5b9211bea2a69d7886c4716ac.jpg)

![5ef1ec1c3dc484e72a3a0197855a536e8ac98b48fbbb08ba811bbd6b51bc0086.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/5ef1ec1c3dc484e72a3a0197855a536e8ac98b48fbbb08ba811bbd6b51bc0086.jpg)

![60fa7641140c5b268d71dfc2b7d545b483e24e3a7a74b3dae1aa81a2d614344b.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/60fa7641140c5b268d71dfc2b7d545b483e24e3a7a74b3dae1aa81a2d614344b.jpg)

![714cf4196911066320e78869b0001e52206330aa099f9f5b587b533bd0f2cbd3.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/714cf4196911066320e78869b0001e52206330aa099f9f5b587b533bd0f2cbd3.jpg)

![83102c52123b1f6d6a61b488b9bc73f52445da8f774c2ddbaae55d4a1df10eb1.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/83102c52123b1f6d6a61b488b9bc73f52445da8f774c2ddbaae55d4a1df10eb1.jpg)

![9f882a5fbda44553f3bb3c0d90eceb09a6937be2e5ab733b60b8d04dfdbc2e3b.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/9f882a5fbda44553f3bb3c0d90eceb09a6937be2e5ab733b60b8d04dfdbc2e3b.jpg)

![a01966de41cf7e6f01f74b2370682b9a4260dee6b8a4948c0f2f7ebb8eed29d8.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a01966de41cf7e6f01f74b2370682b9a4260dee6b8a4948c0f2f7ebb8eed29d8.jpg)

![a09efcff0ba3df6d5e58a40c1a351132fd22c5fc7563b230892c293dd51b010b.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a09efcff0ba3df6d5e58a40c1a351132fd22c5fc7563b230892c293dd51b010b.jpg)

![a30f1fbe511b8b53fe840575664f5f20a25dfa2028fcd6e3575b6cca72b38252.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a30f1fbe511b8b53fe840575664f5f20a25dfa2028fcd6e3575b6cca72b38252.jpg)

![a31b95481ff1253a351f85fc1bd2de5110edb32ec1108089628e4fc0971cceed.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a31b95481ff1253a351f85fc1bd2de5110edb32ec1108089628e4fc0971cceed.jpg)

![a49b70f138d2043ef27b6e2527e051ec5ec98a9aa1ea92c24049e758fd256b4f.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a49b70f138d2043ef27b6e2527e051ec5ec98a9aa1ea92c24049e758fd256b4f.jpg)

![a6aaf749cb8325b2a0f27c06369ab369f367b6b8ca0d1c4c468214e662794eb1.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/a6aaf749cb8325b2a0f27c06369ab369f367b6b8ca0d1c4c468214e662794eb1.jpg)

![c9a65bd72b61c4a4426c1002d9b2800b817ea850a44c31037dd3700b0ce8d5d0.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/c9a65bd72b61c4a4426c1002d9b2800b817ea850a44c31037dd3700b0ce8d5d0.jpg)

![cf72144acfaa61021887810eb1e30fd820f5d664d58a1f602de090b2c3c54296.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/cf72144acfaa61021887810eb1e30fd820f5d664d58a1f602de090b2c3c54296.jpg)

![d62712c676cf9b0d7a6605f18ec895263efa404171f4265f8c83e6bea899f6fc.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/d62712c676cf9b0d7a6605f18ec895263efa404171f4265f8c83e6bea899f6fc.jpg)

![d9087ba18d3258d5952edfa36fc6e715747987c160a1a8ca9b314c2a412d7657.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/d9087ba18d3258d5952edfa36fc6e715747987c160a1a8ca9b314c2a412d7657.jpg)

![ef21b57e5cdee484f066b36801290ed3e9edd317a19b8635d3e6f8d6a2a0f1bb.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/ef21b57e5cdee484f066b36801290ed3e9edd317a19b8635d3e6f8d6a2a0f1bb.jpg)

![f49d44d999294de0a834f3b10e810cc022ac1489659036c8108efca9588e7463.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/f49d44d999294de0a834f3b10e810cc022ac1489659036c8108efca9588e7463.jpg)

![fcd86a3f6794a454b3a45d74bbf8b0b4034d253222f4bfc27f6bb51abc9a553a.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/images/fcd86a3f6794a454b3a45d74bbf8b0b4034d253222f4bfc27f6bb51abc9a553a.jpg)

### Tables

![07580d3d0ed41b757def74c7cffc0e48e667e4fa60672a906b1803b6220b3ccf.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/07580d3d0ed41b757def74c7cffc0e48e667e4fa60672a906b1803b6220b3ccf.jpg)

![0cfb10965272bd70fd02e884d1e7a74411cc083b0a3502d2880fbee68b7cf769.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/0cfb10965272bd70fd02e884d1e7a74411cc083b0a3502d2880fbee68b7cf769.jpg)

![153092d7b77d18bf45f82b47f7bd78396d7a42318bc52b36ad5efad625b1eca3.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/153092d7b77d18bf45f82b47f7bd78396d7a42318bc52b36ad5efad625b1eca3.jpg)

![1d289562df74e25860f7a48790e320310f8c4593459632207ffb0f0a7586229f.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/1d289562df74e25860f7a48790e320310f8c4593459632207ffb0f0a7586229f.jpg)

![3b61ca0fedaaa0b08b3c6a5916f55a36d4d69df0aef738fef55bfa589ad8b361.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/3b61ca0fedaaa0b08b3c6a5916f55a36d4d69df0aef738fef55bfa589ad8b361.jpg)

![514d9ab4ac5b70c4e996793552e0b1af608ee5224a882a5352a95caa25b7ae75.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/514d9ab4ac5b70c4e996793552e0b1af608ee5224a882a5352a95caa25b7ae75.jpg)

![72caa69bb0831f575228e7317923d709b01b57f1e8596788167a33c5dedebd3c.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/72caa69bb0831f575228e7317923d709b01b57f1e8596788167a33c5dedebd3c.jpg)

![78181c79632ad35ab14a29d1517aacb75ac125e2ad416b47bb6603f213754d64.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/78181c79632ad35ab14a29d1517aacb75ac125e2ad416b47bb6603f213754d64.jpg)

![88ac6788bdd5c6120344cd4214cf26aadcf61b2304fe8281313a7557613f05a6.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/88ac6788bdd5c6120344cd4214cf26aadcf61b2304fe8281313a7557613f05a6.jpg)

![ac3c9e54e1d52e2817c663609042deffc73b770171665dcffee21c3d98040fef.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/ac3c9e54e1d52e2817c663609042deffc73b770171665dcffee21c3d98040fef.jpg)

![afea766b1d59a6791bd2e80108905f0b76b125b1573173d1585cd5571a134bfb.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/afea766b1d59a6791bd2e80108905f0b76b125b1573173d1585cd5571a134bfb.jpg)

![b787eab7008df76f4c59296be98fa5fc388c67295a29cb411db86565bc933565.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/b787eab7008df76f4c59296be98fa5fc388c67295a29cb411db86565bc933565.jpg)

![bd21e42a14e88da0111d5c22aa9df73354d294a58803bc18fcc7442a08997826.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/bd21e42a14e88da0111d5c22aa9df73354d294a58803bc18fcc7442a08997826.jpg)

![c5adef054fdf19625966781dcd9185514e81704924c311ae6cd507ed8c151380.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/c5adef054fdf19625966781dcd9185514e81704924c311ae6cd507ed8c151380.jpg)

![cfbfcfaaba107f3d33c49b7910b6ccb74ba4ba8d31e9f488dc688e98d4149c86.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/cfbfcfaaba107f3d33c49b7910b6ccb74ba4ba8d31e9f488dc688e98d4149c86.jpg)

![d030c9ad233f32c56ed6dee832da33b0f8962f265b4cd48765121f6fd1f89397.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/d030c9ad233f32c56ed6dee832da33b0f8962f265b4cd48765121f6fd1f89397.jpg)

![de36e9b30c0cb9b17440647945238f55190344ad2caf40c7a58f2248be954bdc.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/de36e9b30c0cb9b17440647945238f55190344ad2caf40c7a58f2248be954bdc.jpg)

![e34717ffb7dd414bec42a6c05f29674997084c91de2b5ec77c582392eb6e0c1f.jpg](../icml_results/284_Mind-Omni_%20A%20Unified%20Multi-Task%20Framework%20for%20Brain-Vision-Language%20Modeling%20via%20Discrete%20Diffusion/tables/e34717ffb7dd414bec42a6c05f29674997084c91de2b5ec77c582392eb6e0c1f.jpg)

<a id="paper-0280"></a>

## The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models


### Images

![07c9ef583463eb1a62cebdc4e6c2ef5595148d9dfc4aecabf56fa5b6c6173b14.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/07c9ef583463eb1a62cebdc4e6c2ef5595148d9dfc4aecabf56fa5b6c6173b14.jpg)

![0a7e688579c69e106b9f8480f4c75dd294a6c3091e1c8c029acb29beee2075ad.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/0a7e688579c69e106b9f8480f4c75dd294a6c3091e1c8c029acb29beee2075ad.jpg)

![0f4f63eae473b55c91f57e644dd20f28e4a63f583c8c9e091aacdf50b0573c37.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/0f4f63eae473b55c91f57e644dd20f28e4a63f583c8c9e091aacdf50b0573c37.jpg)

![0f50741e33d84139289f4d29f121f1de92bb2a3e473651095c411082ad55b2c9.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/0f50741e33d84139289f4d29f121f1de92bb2a3e473651095c411082ad55b2c9.jpg)

![12cce811bcdf4460b44b965da14eeda48683386517c8acb75387881da41d258a.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/12cce811bcdf4460b44b965da14eeda48683386517c8acb75387881da41d258a.jpg)

![130bd25fc8093b5e606d32c5d07a9d94e214d893a40611e96cd93079fee16a38.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/130bd25fc8093b5e606d32c5d07a9d94e214d893a40611e96cd93079fee16a38.jpg)

![1c399eb65e05df081bfad2705b368fd46784bb1f4af03f2e946b6ec562f7721a.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/1c399eb65e05df081bfad2705b368fd46784bb1f4af03f2e946b6ec562f7721a.jpg)

![1e9cdd0f8e0d689c2b0070307c917e3d32a78872fcd159d86b2b304821d13c1c.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/1e9cdd0f8e0d689c2b0070307c917e3d32a78872fcd159d86b2b304821d13c1c.jpg)

![227b32ee179538cf75b56526453eb6baf6f6391b48f88b6eb6808759110d8c90.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/227b32ee179538cf75b56526453eb6baf6f6391b48f88b6eb6808759110d8c90.jpg)

![27190d67afce8e2cb6dc86588b6e7cadf62c92fc594870d1b244a427e167e09b.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/27190d67afce8e2cb6dc86588b6e7cadf62c92fc594870d1b244a427e167e09b.jpg)

![2ae9c428dbce1e4c7ff62272d0d790b42a1f60c6c059be9e369a5879fd80f1b3.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/2ae9c428dbce1e4c7ff62272d0d790b42a1f60c6c059be9e369a5879fd80f1b3.jpg)

![2bdbbf2777f9f8525e2ba9a1700c07d33f5df2da01dcbf9bc98462391c92a28a.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/2bdbbf2777f9f8525e2ba9a1700c07d33f5df2da01dcbf9bc98462391c92a28a.jpg)

![36c7d3504c023c012c9ce349cda74313c507d6bf94412f7dd7fa44e718fc5706.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/36c7d3504c023c012c9ce349cda74313c507d6bf94412f7dd7fa44e718fc5706.jpg)

![3b1cf763de9ead467e3c113a2ec269272e01c19d9db4c4990e79544ff2a49d3d.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/3b1cf763de9ead467e3c113a2ec269272e01c19d9db4c4990e79544ff2a49d3d.jpg)

![4399c94500ae1a503a9de7b45af1f342a7d5dbfec56f6fc058c387604f794177.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/4399c94500ae1a503a9de7b45af1f342a7d5dbfec56f6fc058c387604f794177.jpg)

![4546107ec752afba72a151fb30dab6548901ef381bb34de74473aea099baff49.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/4546107ec752afba72a151fb30dab6548901ef381bb34de74473aea099baff49.jpg)

![672168eccdee277f8137122a1b1aecadaa356637d79e6e1b945e39b3c3b8b512.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/672168eccdee277f8137122a1b1aecadaa356637d79e6e1b945e39b3c3b8b512.jpg)

![68af5291f51f17d527b5be67be8f6a1dc13072d90a7a7131d6f28663f1eb4a89.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/68af5291f51f17d527b5be67be8f6a1dc13072d90a7a7131d6f28663f1eb4a89.jpg)

![6ccb968a6d1ec85c407aacf6a25c005932b303703ee2c000b9ec095d1d00cb21.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/6ccb968a6d1ec85c407aacf6a25c005932b303703ee2c000b9ec095d1d00cb21.jpg)

![8deaeb02d0618d314cc5343d1282b9399f8fb5541b8e0716099867aeb1458906.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/8deaeb02d0618d314cc5343d1282b9399f8fb5541b8e0716099867aeb1458906.jpg)

![8ea857407aa1aac76e387aaa52468c46c058844ac110cbff3cc075d65cff66b7.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/8ea857407aa1aac76e387aaa52468c46c058844ac110cbff3cc075d65cff66b7.jpg)

![903c550c397aaba3044d53e053bd9347e11d5c3c690dca04d764ae68ff554153.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/903c550c397aaba3044d53e053bd9347e11d5c3c690dca04d764ae68ff554153.jpg)

![925571e80cc06cb04d2bc846c9ad7bcc96dfbc91f5b3bdfce54955be8e0a818b.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/925571e80cc06cb04d2bc846c9ad7bcc96dfbc91f5b3bdfce54955be8e0a818b.jpg)

![928f040f170340b6fac55325b6839c9f8a055bc64c567630e8c22dac7dd7788e.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/928f040f170340b6fac55325b6839c9f8a055bc64c567630e8c22dac7dd7788e.jpg)

![a599112eb19cb2be81e58783e8a5e0829dbca7ee5a4faba89de77d5289619ea0.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/a599112eb19cb2be81e58783e8a5e0829dbca7ee5a4faba89de77d5289619ea0.jpg)

![abe8b9b6bbabbcb6d45367f71519e61748e0c9bf7e674a0daca69d89e279f38c.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/abe8b9b6bbabbcb6d45367f71519e61748e0c9bf7e674a0daca69d89e279f38c.jpg)

![ba71873b67f90cd2d4c86b0dfdc321f0fb2b809cd0dd0945fd72c072b9d7d08d.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/ba71873b67f90cd2d4c86b0dfdc321f0fb2b809cd0dd0945fd72c072b9d7d08d.jpg)

![beee37e85b7964e9a089d1b1fa4b241e9d538971af60f37fde8df0970bbd896b.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/beee37e85b7964e9a089d1b1fa4b241e9d538971af60f37fde8df0970bbd896b.jpg)

![c654a2e2bc917ddc8e12a200fac39b135ccd4b7a8107a97e3d117128c28f8e5d.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/c654a2e2bc917ddc8e12a200fac39b135ccd4b7a8107a97e3d117128c28f8e5d.jpg)

![d2c38041abb8091ef039b7b1b9e690e4ff4256d749eaf092aa5f3e48a69c2cca.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/d2c38041abb8091ef039b7b1b9e690e4ff4256d749eaf092aa5f3e48a69c2cca.jpg)

![dc4e6095119150f30098881f083164b3d85a4afb98c552a010d8ece373e02f8b.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/dc4e6095119150f30098881f083164b3d85a4afb98c552a010d8ece373e02f8b.jpg)

![dc518da8b370577e7645c1a7e74e4af6096a561c1c7e537a7da0633f2e98a572.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/dc518da8b370577e7645c1a7e74e4af6096a561c1c7e537a7da0633f2e98a572.jpg)

![e122e59b39caae0450ab818847acf599dba162bc7286e74d0b73c9c81570bab7.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/e122e59b39caae0450ab818847acf599dba162bc7286e74d0b73c9c81570bab7.jpg)

![fa5730044adb65a04afd6829be7ebcb821d9c9351fc2f5faee419873eaea1f41.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/fa5730044adb65a04afd6829be7ebcb821d9c9351fc2f5faee419873eaea1f41.jpg)

![fac0043dd98f1d806dc8ff7fa1ff4223508337e02ca351786295a66ecf9a6b8a.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/fac0043dd98f1d806dc8ff7fa1ff4223508337e02ca351786295a66ecf9a6b8a.jpg)

![fdb5a9b53ea4f2c3cc2c07feed27e68910e2ec66d3787cd62d426cd416dd4014.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/images/fdb5a9b53ea4f2c3cc2c07feed27e68910e2ec66d3787cd62d426cd416dd4014.jpg)

### Tables

![04d743fad47e18a20d86dcae3b9c20611accf38ae85653378a8b9bac902ca898.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/04d743fad47e18a20d86dcae3b9c20611accf38ae85653378a8b9bac902ca898.jpg)

![20936c57e18fa195d6daa25006506a1789d48b8c02d9e4560e8fa287a2bae8c0.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/20936c57e18fa195d6daa25006506a1789d48b8c02d9e4560e8fa287a2bae8c0.jpg)

![270af6e637a1696dfe7a1db0b28673b90140bb95435b4a1f55163e0cf1586a20.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/270af6e637a1696dfe7a1db0b28673b90140bb95435b4a1f55163e0cf1586a20.jpg)

![3097b6f120ee2c9f75831ef2558a702d5398140dec973e6023eebd4b40f8bd31.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/3097b6f120ee2c9f75831ef2558a702d5398140dec973e6023eebd4b40f8bd31.jpg)

![3b45a996ce843cfbe24997e5a45b85acd88cf938f0929a95599ccd9ce766098c.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/3b45a996ce843cfbe24997e5a45b85acd88cf938f0929a95599ccd9ce766098c.jpg)

![47e0e94c931c4ea1b726f26ffe05d7634424372fc4dff6525f07ff0b48a2c42f.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/47e0e94c931c4ea1b726f26ffe05d7634424372fc4dff6525f07ff0b48a2c42f.jpg)

![4dd2c91d70e8d61a0623b14b453d4a7345a5d8a5cb52f20690486911fdb3bf18.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/4dd2c91d70e8d61a0623b14b453d4a7345a5d8a5cb52f20690486911fdb3bf18.jpg)

![5614b13d3a01bf47a0517ae2262b3163fd84096f1a239ea988ce0bb4ab43e3e5.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/5614b13d3a01bf47a0517ae2262b3163fd84096f1a239ea988ce0bb4ab43e3e5.jpg)

![6135cabf0b774e2dc6649cbd02e3fa6a6088a217a750774e155d1f2495dce32e.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/6135cabf0b774e2dc6649cbd02e3fa6a6088a217a750774e155d1f2495dce32e.jpg)

![77d7579666aebbc1851db4524ba467217e14d4e8a7efb00737b800f039de406f.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/77d7579666aebbc1851db4524ba467217e14d4e8a7efb00737b800f039de406f.jpg)

![a663be67b0f045808a4b9d429163913ff6ed9df92947d274bbce73c1ab92ae6f.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/a663be67b0f045808a4b9d429163913ff6ed9df92947d274bbce73c1ab92ae6f.jpg)

![c10b1be584781e56c71a2a42475e533e3da810cbfa6798a9a5542e24d9c83e81.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/c10b1be584781e56c71a2a42475e533e3da810cbfa6798a9a5542e24d9c83e81.jpg)

![c46a53a53c01057c946a3093224846b6349beaea7082a7a3a377608ff75cc904.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/c46a53a53c01057c946a3093224846b6349beaea7082a7a3a377608ff75cc904.jpg)

![c5ba1a9d199d28388951508db60163422dec47b3d979db885221209f02bbc3d4.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/c5ba1a9d199d28388951508db60163422dec47b3d979db885221209f02bbc3d4.jpg)

![d3de5fede8da0446056eb272b0e440a5c46c71fa2bdcc3d93b6994a20aaa8d4e.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/d3de5fede8da0446056eb272b0e440a5c46c71fa2bdcc3d93b6994a20aaa8d4e.jpg)

![da58274c5aef18bc22ad7c859b2ad64993e7e45150ca78e4a3b200ce8060ec98.jpg](../icml_results/285_The%20Assistant%20Axis_%20Situating%20and%20Stabilizing%20the%20Default%20Persona%20of%20Language%20Models/tables/da58274c5aef18bc22ad7c859b2ad64993e7e45150ca78e4a3b200ce8060ec98.jpg)

<a id="paper-0281"></a>

## Interpretable Functional Koopman Learning with Non-Markovian Closure for Spatiotemporal Systems


### Images

![2fee1d8cf241f261822b40f8808b03b6205c66a925e272faa670d7e74313796b.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/2fee1d8cf241f261822b40f8808b03b6205c66a925e272faa670d7e74313796b.jpg)

![331bf6613349cd18ac7f89a6cababeba24ccf016634731ddb3de7facc00ac493.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/331bf6613349cd18ac7f89a6cababeba24ccf016634731ddb3de7facc00ac493.jpg)

![36f6fc8409ed2d4ba7a7a4c5000fff6608aa0b5b1e2d1e0f52ca66a4b3973454.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/36f6fc8409ed2d4ba7a7a4c5000fff6608aa0b5b1e2d1e0f52ca66a4b3973454.jpg)

![40487d8237e94f1d14714eadaf4ab0fc1a8fe1b0960ecd463e0080a806b79b76.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/40487d8237e94f1d14714eadaf4ab0fc1a8fe1b0960ecd463e0080a806b79b76.jpg)

![5307936a41c31ad5b980d7544890d64fec8572a830e148c5c62c7c691cd06068.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/5307936a41c31ad5b980d7544890d64fec8572a830e148c5c62c7c691cd06068.jpg)

![681357a782668dbd3af072c643121d19f5af7c38f15637cdcf3db1dcef7e7193.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/681357a782668dbd3af072c643121d19f5af7c38f15637cdcf3db1dcef7e7193.jpg)

![68b27c253e3186b33e1d227cc7acff9726e4b5303f49f7e750d99ab1995c9c7f.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/68b27c253e3186b33e1d227cc7acff9726e4b5303f49f7e750d99ab1995c9c7f.jpg)

![6f708f3b5f842c60c3d3e1499bef348378f4ac1dbace801a64230f5981315d63.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/6f708f3b5f842c60c3d3e1499bef348378f4ac1dbace801a64230f5981315d63.jpg)

![728ffcb5210c53638991cb142738ce40de5514a32b88c5cac6795bfe276115be.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/728ffcb5210c53638991cb142738ce40de5514a32b88c5cac6795bfe276115be.jpg)

![74395f37e9a682b99b3009bb8ffc3c845cf1f45e15b0b98efbf2b9a53e2a2e61.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/74395f37e9a682b99b3009bb8ffc3c845cf1f45e15b0b98efbf2b9a53e2a2e61.jpg)

![7d46840b7c4d7b951b2163dca1648103689fff13425a139e8f3b1fee75e699ee.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/7d46840b7c4d7b951b2163dca1648103689fff13425a139e8f3b1fee75e699ee.jpg)

![8e8bee66f5fadb5ad2284b36c916b7fe0c9c0bf1c86c3cf2b0ac9fe8e3e66fb3.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/8e8bee66f5fadb5ad2284b36c916b7fe0c9c0bf1c86c3cf2b0ac9fe8e3e66fb3.jpg)

![905903df85e2a35deef51ae8ccb8ab3b31b592260e27710e670f58dbd9c319a4.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/905903df85e2a35deef51ae8ccb8ab3b31b592260e27710e670f58dbd9c319a4.jpg)

![9cdfa01c5945f4857f32e036c16477d5fa889e0e2bf4be1747e2336bfeea5922.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/9cdfa01c5945f4857f32e036c16477d5fa889e0e2bf4be1747e2336bfeea5922.jpg)

![d3b879a8815a7de4800f56be0df1c3683655c2a1506ac7dd6ea09eff0523630a.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/d3b879a8815a7de4800f56be0df1c3683655c2a1506ac7dd6ea09eff0523630a.jpg)

![f6c80199bc861612cff686fac877f11dec1c2ba65c71ef625f203c39ec70e266.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/f6c80199bc861612cff686fac877f11dec1c2ba65c71ef625f203c39ec70e266.jpg)

![fde6673a715159af143b6cd3fcf66f2ce9a59d0642382789fed7469e2d045bd9.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/images/fde6673a715159af143b6cd3fcf66f2ce9a59d0642382789fed7469e2d045bd9.jpg)

### Tables

![15ab9f497cce6a06c591f8b81060a39fe65486bf6cec07805223201219a35513.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/15ab9f497cce6a06c591f8b81060a39fe65486bf6cec07805223201219a35513.jpg)

![527e2bd39fac4232ca4739b880c87be6fec5afd3f72cd616d2eaae9de37f83b1.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/527e2bd39fac4232ca4739b880c87be6fec5afd3f72cd616d2eaae9de37f83b1.jpg)

![55b7ff0c9602232fdc58830daf6ef139f2bb6ab28200f1093a43e0711844c90e.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/55b7ff0c9602232fdc58830daf6ef139f2bb6ab28200f1093a43e0711844c90e.jpg)

![72f4ad00cd368339b8028938f356845059c518ec4e5dc362d738bc6fff273cad.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/72f4ad00cd368339b8028938f356845059c518ec4e5dc362d738bc6fff273cad.jpg)

![7c98a3cbb80a90c5b6900b8201994ad848b03174e207d438e43b6705eaa7691a.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/7c98a3cbb80a90c5b6900b8201994ad848b03174e207d438e43b6705eaa7691a.jpg)

![7da8a379a036d090aa0232e139f642f3e1721b7f5c60cba94bd6234f01fb9cd9.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/7da8a379a036d090aa0232e139f642f3e1721b7f5c60cba94bd6234f01fb9cd9.jpg)

![84cb11811a536ef1617a03f1ee04768edce3f0bbe72e5724bce9883906b7e260.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/84cb11811a536ef1617a03f1ee04768edce3f0bbe72e5724bce9883906b7e260.jpg)

![8ee5c9b4a4b8990f5dc08c2a1e3272035fa6f6f5905bcad4f9e06d827f670b2b.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/8ee5c9b4a4b8990f5dc08c2a1e3272035fa6f6f5905bcad4f9e06d827f670b2b.jpg)

![90b1177c79f2dc66c46d43b6e7d6902df6508c3732b9b2b83554a8f09b1bad64.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/90b1177c79f2dc66c46d43b6e7d6902df6508c3732b9b2b83554a8f09b1bad64.jpg)

![b23ae2fb6915d93b2714b29276c929f0e1031ad9db75d373d3952c306dd5bb65.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/b23ae2fb6915d93b2714b29276c929f0e1031ad9db75d373d3952c306dd5bb65.jpg)

![b520d115c1c32f73b4021a441ad919d689fa805d31d54e7a5eb73401301ee356.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/b520d115c1c32f73b4021a441ad919d689fa805d31d54e7a5eb73401301ee356.jpg)

![d893f34ab1de08d719a10ec80dd70ea6e8fb95091a636de0b934eadc306a1f48.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/d893f34ab1de08d719a10ec80dd70ea6e8fb95091a636de0b934eadc306a1f48.jpg)

![e42101cd1cbafad3ebdab3e8a9818f2fef9b6d308e44ea14bd6d3b7cc7f9e885.jpg](../icml_results/286_Interpretable%20Functional%20Koopman%20Learning%20with%20Non-Markovian%20Closure%20for%20Spatiotemporal%20Systems/tables/e42101cd1cbafad3ebdab3e8a9818f2fef9b6d308e44ea14bd6d3b7cc7f9e885.jpg)

<a id="paper-0282"></a>

## Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching


### Images

![709aec7152a72d4c1ea6be2c1fff4fe1196de5d71bed2a483835b8b2e3223ff0.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/images/709aec7152a72d4c1ea6be2c1fff4fe1196de5d71bed2a483835b8b2e3223ff0.jpg)

![dde627eacf5af25767b85edfc585c6939902dc16d91c0247e7f450c1c1e60730.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/images/dde627eacf5af25767b85edfc585c6939902dc16d91c0247e7f450c1c1e60730.jpg)

![e1f873ef47065367bab63e2b17e108b46e34a35f75ffed1275daa4a1d19a6e30.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/images/e1f873ef47065367bab63e2b17e108b46e34a35f75ffed1275daa4a1d19a6e30.jpg)

### Tables

![039b75ea1e237f480232db51eff5b6cfb0e18109e7491bf412296b508532b4dd.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/039b75ea1e237f480232db51eff5b6cfb0e18109e7491bf412296b508532b4dd.jpg)

![205bfd6fee3df86dfe3775b1465badd0a4eebd1c10308c6de2831468fd185b6b.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/205bfd6fee3df86dfe3775b1465badd0a4eebd1c10308c6de2831468fd185b6b.jpg)

![26346f09a1dd5e07cf1123bba0a9ddf72b9e943dace23d30d073b8dcbd902af5.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/26346f09a1dd5e07cf1123bba0a9ddf72b9e943dace23d30d073b8dcbd902af5.jpg)

![2d24608992b69071dad5492dc2e86fa544b367ee8a53453c232bdb65ec64a8f9.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/2d24608992b69071dad5492dc2e86fa544b367ee8a53453c232bdb65ec64a8f9.jpg)

![72890fbe0ec7b2e9f1aaefe77f1ef7ba946679ce6715aaddc38fd50859389df1.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/72890fbe0ec7b2e9f1aaefe77f1ef7ba946679ce6715aaddc38fd50859389df1.jpg)

![7c2dc4e41ddc80a1a1eb5fb0ca3a19f9a8dbe34afc7cecef881b0647306921ac.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/7c2dc4e41ddc80a1a1eb5fb0ca3a19f9a8dbe34afc7cecef881b0647306921ac.jpg)

![84c2472e564452f8b8ca35150fa5881c9fa7c1148d08f73a7fa1c5728cb8a48f.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/84c2472e564452f8b8ca35150fa5881c9fa7c1148d08f73a7fa1c5728cb8a48f.jpg)

![8b28898dfcf1f7725eb7d8f7268a5d8e730464e316d07d908b0918894408c282.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/8b28898dfcf1f7725eb7d8f7268a5d8e730464e316d07d908b0918894408c282.jpg)

![aa14116a92cc1c2965c9e1a2b2a93516e9cf91cb0171e911e772c662ed9d85a5.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/aa14116a92cc1c2965c9e1a2b2a93516e9cf91cb0171e911e772c662ed9d85a5.jpg)

![b0e3fd8f1ab0803790600b95593be0681b9affa8f4a6fac8771301f01f817289.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/b0e3fd8f1ab0803790600b95593be0681b9affa8f4a6fac8771301f01f817289.jpg)

![d093f1e5c543aaa9397ead9eb4a32591124d72751de5e9fc3b3c4a588ec1c6d0.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/d093f1e5c543aaa9397ead9eb4a32591124d72751de5e9fc3b3c4a588ec1c6d0.jpg)

![f0f4e7604ba1204b8ee388185a041850db442f800ab0c1555748f87082fa84c6.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/f0f4e7604ba1204b8ee388185a041850db442f800ab0c1555748f87082fa84c6.jpg)

![fb7257ef03f1da5588a066e0ed4289e15fede12507077cc924aebd957487336b.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/fb7257ef03f1da5588a066e0ed4289e15fede12507077cc924aebd957487336b.jpg)

![fd8aad2f294d95b21ac7fb17e2bea01618b77f2423c9aa74c8067e88aec2e234.jpg](../icml_results/287_Dynamic%20Stratified%20Contrastive%20Learning%20with%20Upstream%20Augmentation%20for%20MILP%20Branching/tables/fd8aad2f294d95b21ac7fb17e2bea01618b77f2423c9aa74c8067e88aec2e234.jpg)

<a id="paper-0283"></a>

## FedARC: Anchor-Guided Residual Compensation for Data and Model Heterogeneous Federated Learning


### Images

![26366a82e4f4a79588cf6cd059172b67c66ed1986934912ff43837d19648f59b.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/26366a82e4f4a79588cf6cd059172b67c66ed1986934912ff43837d19648f59b.jpg)

![33f0e3a9a5be76ff16b2ad4fe4032e24cd056340e14a18b36c2f20b59bb2d1d8.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/33f0e3a9a5be76ff16b2ad4fe4032e24cd056340e14a18b36c2f20b59bb2d1d8.jpg)

![4604e29d58afc71350ce39fd7213b9781c78640c34be2f548ce39fa71fcbc55e.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/4604e29d58afc71350ce39fd7213b9781c78640c34be2f548ce39fa71fcbc55e.jpg)

![4966e87fa17f723fc997bc1ba9981f44e88e465d3b9fe387b9db064f27ad4909.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/4966e87fa17f723fc997bc1ba9981f44e88e465d3b9fe387b9db064f27ad4909.jpg)

![51b9cf6d885c517496846f27fdb35f04428c00c207cd08728cc811963e8a8929.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/51b9cf6d885c517496846f27fdb35f04428c00c207cd08728cc811963e8a8929.jpg)

![69c216948f84abb84494959ae104c09b150877ffcfd8c0feb0ca6848591c6fea.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/69c216948f84abb84494959ae104c09b150877ffcfd8c0feb0ca6848591c6fea.jpg)

![6ab15ed593fc5685f49d3982aab39c828042a8d144a20f51e28dbf21fd6feb81.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/6ab15ed593fc5685f49d3982aab39c828042a8d144a20f51e28dbf21fd6feb81.jpg)

![6c4082d427de72d3cf72fda5ac6a9961b0f05da35ed8cee06a4091a4f38a0856.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/6c4082d427de72d3cf72fda5ac6a9961b0f05da35ed8cee06a4091a4f38a0856.jpg)

![a7f0074371bfc0ef477159c88ebcbf8c2c2f0da4a43acd4dd26b00c05656acb0.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/a7f0074371bfc0ef477159c88ebcbf8c2c2f0da4a43acd4dd26b00c05656acb0.jpg)

![c7e4a0f4f032d82c3973d17516cd41a42624d26608bbda551089076561717792.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/c7e4a0f4f032d82c3973d17516cd41a42624d26608bbda551089076561717792.jpg)

![d1a6ca836c978d1627fbf61f99faad5d84429fdd34c06f36b056fd3269519f1b.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/d1a6ca836c978d1627fbf61f99faad5d84429fdd34c06f36b056fd3269519f1b.jpg)

![dbab60cd6d24adb4bc0ab7821173dfd9fabe53ee22e110da4035d951c5c369ae.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/dbab60cd6d24adb4bc0ab7821173dfd9fabe53ee22e110da4035d951c5c369ae.jpg)

![f6677538ec0de041944f45b0ef3c6407d13cdbcc13ed2f2a272a2d0c68fdb6ed.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/f6677538ec0de041944f45b0ef3c6407d13cdbcc13ed2f2a272a2d0c68fdb6ed.jpg)

![fc5643857af8796079f4d969fab1dbc91031f6fbae8f545bdcb5c3f0577ed95c.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/fc5643857af8796079f4d969fab1dbc91031f6fbae8f545bdcb5c3f0577ed95c.jpg)

![fdb9ba3c6a1369a027ea32e0fea4288bada5ef085055e9fd69cb929b2da133b7.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/images/fdb9ba3c6a1369a027ea32e0fea4288bada5ef085055e9fd69cb929b2da133b7.jpg)

### Tables

![019fc2ee0b0e24fee21612b650c2ac759f606685008149b1aa2325f015e6e539.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/019fc2ee0b0e24fee21612b650c2ac759f606685008149b1aa2325f015e6e539.jpg)

![116fa6e220766e373b9daaf1836ce6474b8894a492abd448350636f4c4b55ee6.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/116fa6e220766e373b9daaf1836ce6474b8894a492abd448350636f4c4b55ee6.jpg)

![1808f618a32c7b2d89e9229d9ecb400ad8c316665758d2aebbffdde2dffd226a.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/1808f618a32c7b2d89e9229d9ecb400ad8c316665758d2aebbffdde2dffd226a.jpg)

![2901aa1aebcf5108759830b87aaa764d81f9810744c6fb26e375ea6ce0db060a.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/2901aa1aebcf5108759830b87aaa764d81f9810744c6fb26e375ea6ce0db060a.jpg)

![4880f757ea8cacaf38b375ab44538adc073fe2320cb2dac989dc1851ee65cb3f.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/4880f757ea8cacaf38b375ab44538adc073fe2320cb2dac989dc1851ee65cb3f.jpg)

![4f7c0b26db5990161bd2e9eaa2589b4c285efae34aeddd6afd6ddccbaad85172.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/4f7c0b26db5990161bd2e9eaa2589b4c285efae34aeddd6afd6ddccbaad85172.jpg)

![61cc7779fc2cf60e05533402b6c09e5df632179c275b2e1a9ef3f0900f9907d2.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/61cc7779fc2cf60e05533402b6c09e5df632179c275b2e1a9ef3f0900f9907d2.jpg)

![63dc4d20bd75473f784b2f87e5d8729b6cd59252214a3bff014d469504444f9d.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/63dc4d20bd75473f784b2f87e5d8729b6cd59252214a3bff014d469504444f9d.jpg)

![7990db7b2b57d063e2b17d490be37729d14fde87076d7a326d436f980be472e2.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/7990db7b2b57d063e2b17d490be37729d14fde87076d7a326d436f980be472e2.jpg)

![91360f268a1842bb5a041a6a175aca31034e3a3dadb2ebd0bc99d18c9afde565.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/91360f268a1842bb5a041a6a175aca31034e3a3dadb2ebd0bc99d18c9afde565.jpg)

![ab9cbf0da9cf1db511bd44196b41dc4138d061f3269bdaa7af4905b0ec4c7db6.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/ab9cbf0da9cf1db511bd44196b41dc4138d061f3269bdaa7af4905b0ec4c7db6.jpg)

![ae56ec6ff1e626935cc44264730b259e57da872b667d5525f0313069e12384a5.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/ae56ec6ff1e626935cc44264730b259e57da872b667d5525f0313069e12384a5.jpg)

![cc4c2735d86c9ae2d29c25b324973c09d420aa2352d9424e132b883bd7b19507.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/cc4c2735d86c9ae2d29c25b324973c09d420aa2352d9424e132b883bd7b19507.jpg)

![d4eff71669cd331e7e757cbd95df42e422696e71353e4217847d4586bb270d1f.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/d4eff71669cd331e7e757cbd95df42e422696e71353e4217847d4586bb270d1f.jpg)

![dcd5476c6b2e3b166d37cbff537fc7a14e1dff6ea4b73f4cfa48cc2c1457fa6b.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/dcd5476c6b2e3b166d37cbff537fc7a14e1dff6ea4b73f4cfa48cc2c1457fa6b.jpg)

![f7dde99452ac0d603750bb47a53f6da8d817053754cff99598f699f501328fbe.jpg](../icml_results/288_FedARC_%20Anchor-Guided%20Residual%20Compensation%20for%20Data%20and%20Model%20Heterogeneous%20Federated%20Learning/tables/f7dde99452ac0d603750bb47a53f6da8d817053754cff99598f699f501328fbe.jpg)

<a id="paper-0284"></a>

## Ratio-Variance Regularized Policy Optimization


### Images

![4461351e1d9e962414fe90d3dc52e2113dea9c615ae4e5143c78cc49a4bcc6f7.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/4461351e1d9e962414fe90d3dc52e2113dea9c615ae4e5143c78cc49a4bcc6f7.jpg)

![513e12fd86e9eb5b17611603b64fffa4650c230f48dcd38d42f44e338157df01.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/513e12fd86e9eb5b17611603b64fffa4650c230f48dcd38d42f44e338157df01.jpg)

![643f469f3cbca096636eae8c3d2d054dbbe34c38ce94ae598bb5362a16e3aae5.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/643f469f3cbca096636eae8c3d2d054dbbe34c38ce94ae598bb5362a16e3aae5.jpg)

![65716d3f587f0c829a79ba681b0a32e7bb469167748b1f47428fdad3639d688d.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/65716d3f587f0c829a79ba681b0a32e7bb469167748b1f47428fdad3639d688d.jpg)

![772258d52c8a66c38f28cf66b237d66339c2dc4d7ea8675066d08bd023bdfeea.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/772258d52c8a66c38f28cf66b237d66339c2dc4d7ea8675066d08bd023bdfeea.jpg)

![806c9228633401bd6f939a0fc348d4730420b892ef2e079d6fe100a503d669e9.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/806c9228633401bd6f939a0fc348d4730420b892ef2e079d6fe100a503d669e9.jpg)

![a1546136b8fad4e4ef094305ddddb656730033d8f27bcc8f4f4ab3c1c1b24103.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/a1546136b8fad4e4ef094305ddddb656730033d8f27bcc8f4f4ab3c1c1b24103.jpg)

![b2d2b45e77761afb13d5ee1504a4f2194dbbe11dd9a438feb47e444555580907.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/b2d2b45e77761afb13d5ee1504a4f2194dbbe11dd9a438feb47e444555580907.jpg)

![b7d0da885f1806289dd0db278d0dfc7202cff75e75f39462a80233be66e0e392.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/b7d0da885f1806289dd0db278d0dfc7202cff75e75f39462a80233be66e0e392.jpg)

![bbaad3f4f6f521834082c3934b85c196a7e5ba40584767593300b84d43772b04.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/bbaad3f4f6f521834082c3934b85c196a7e5ba40584767593300b84d43772b04.jpg)

![be61822a67909d1b3919b44958a44b4525c8013a9eec4c4207621f3737b9c94b.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/be61822a67909d1b3919b44958a44b4525c8013a9eec4c4207621f3737b9c94b.jpg)

![ddc8d0a45a49a43dff8817e9802d31a78ed03c5118293638596f6f4ea5217aee.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/images/ddc8d0a45a49a43dff8817e9802d31a78ed03c5118293638596f6f4ea5217aee.jpg)

### Tables

![13a013bf9fc6d8a3719868fa7d0d8416e696aa09c45d7d88a1e7e36af6a108f7.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/tables/13a013bf9fc6d8a3719868fa7d0d8416e696aa09c45d7d88a1e7e36af6a108f7.jpg)

![9227522995fd9236dc39fdbbe3635e265a75afb0574bb00f14c152fc7289fc4c.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/tables/9227522995fd9236dc39fdbbe3635e265a75afb0574bb00f14c152fc7289fc4c.jpg)

![9b7c276df741a3885b15818f4dd857b4c24377311b566abf154a628579d8bf2a.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/tables/9b7c276df741a3885b15818f4dd857b4c24377311b566abf154a628579d8bf2a.jpg)

![b00d1860296273d580b1a36141bf3dc24f591673c0efe5ee500d5554d27a7704.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/tables/b00d1860296273d580b1a36141bf3dc24f591673c0efe5ee500d5554d27a7704.jpg)

![b85c6d6dbac04701401cedd311345bbc76134c16093019f22bdc90d8a6f2582a.jpg](../icml_results/289_Ratio-Variance%20Regularized%20Policy%20Optimization/tables/b85c6d6dbac04701401cedd311345bbc76134c16093019f22bdc90d8a6f2582a.jpg)

<a id="paper-0285"></a>

## CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability


### Images

![044acf9954480eeb518c3c0feb9e0de1d434097e0b27902744cba1fe757fc689.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/044acf9954480eeb518c3c0feb9e0de1d434097e0b27902744cba1fe757fc689.jpg)

![1df0e4e197f53a297bb90ed847a25a881386c5da40e90e27f878da77fb73fa8d.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/1df0e4e197f53a297bb90ed847a25a881386c5da40e90e27f878da77fb73fa8d.jpg)

![35220c8356cdf2f856bf64aeba7b39ef6d70240402a185982104b3bcdafa19a6.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/35220c8356cdf2f856bf64aeba7b39ef6d70240402a185982104b3bcdafa19a6.jpg)

![4f77d30c2df8b069c3f8ff35c73a71c956751590de4931f2e4c77c27b09f914f.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/4f77d30c2df8b069c3f8ff35c73a71c956751590de4931f2e4c77c27b09f914f.jpg)

![84989bf560bb80ec51f2842952a11c54215aedc47b2a15d0ec496402f5a7fc80.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/84989bf560bb80ec51f2842952a11c54215aedc47b2a15d0ec496402f5a7fc80.jpg)

![9dda2a26f8e9d182c13227031867804f812f463571c5b08bb50f91bbecd5a38b.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/9dda2a26f8e9d182c13227031867804f812f463571c5b08bb50f91bbecd5a38b.jpg)

![d87a87669f556eb8d0bff1686259ce11b7a5560cee303908b1b987dfccf9e9c2.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/d87a87669f556eb8d0bff1686259ce11b7a5560cee303908b1b987dfccf9e9c2.jpg)

![e4519d15062e75ef098baf06f8d3a4bc0ccefbdeca02e8d53ec632e17dabad19.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/images/e4519d15062e75ef098baf06f8d3a4bc0ccefbdeca02e8d53ec632e17dabad19.jpg)

### Tables

![03939985983a0bf03a21f2cb9b36054ec768446d23823e190516185ab43e99a5.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/03939985983a0bf03a21f2cb9b36054ec768446d23823e190516185ab43e99a5.jpg)

![13eafcbdae7c04cee6bd5a9700f84badb061ab1a221e5c79968ae2eafd8845a4.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/13eafcbdae7c04cee6bd5a9700f84badb061ab1a221e5c79968ae2eafd8845a4.jpg)

![394fbdc783434091e09955064de6cab8f7894e50bfabff9d166484cb26cbd991.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/394fbdc783434091e09955064de6cab8f7894e50bfabff9d166484cb26cbd991.jpg)

![3efc8641196e52562087215a28b9034796e52fb619f2178687affd0eaa126665.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/3efc8641196e52562087215a28b9034796e52fb619f2178687affd0eaa126665.jpg)

![67bd3329e74ef1155a73dcc0acdd4b646706413adf2371355893c296cb30108d.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/67bd3329e74ef1155a73dcc0acdd4b646706413adf2371355893c296cb30108d.jpg)

![7af1703b5271f22b351b1eed86907fb9398bb2c4b99f83ed28e94ecf8d237179.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/7af1703b5271f22b351b1eed86907fb9398bb2c4b99f83ed28e94ecf8d237179.jpg)

![a0ed5dbc80e611c220ded26e1e367dd258cb0f3a7c99b70fe0255ff88a9caf0f.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/a0ed5dbc80e611c220ded26e1e367dd258cb0f3a7c99b70fe0255ff88a9caf0f.jpg)

![c953024834ddde39dbbb55329f2aba490813acda8bb8680f253afd8f89ac1b76.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/c953024834ddde39dbbb55329f2aba490813acda8bb8680f253afd8f89ac1b76.jpg)

![ca5e772f83319b2bac6145c610c4b3a825ccc4b9c37b982f821fbd7b317a62c8.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/ca5e772f83319b2bac6145c610c4b3a825ccc4b9c37b982f821fbd7b317a62c8.jpg)

![ed35b9b1e8929788d1136fdc64bdebd0f692a14102b8bdf051105079f207130f.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/ed35b9b1e8929788d1136fdc64bdebd0f692a14102b8bdf051105079f207130f.jpg)

![ff47bca88b08ccd90e497cc390015d0e760a5b3cd9e2dec93d2e54f733051478.jpg](../icml_results/290_CVE-Factory_%20Scaling%20Expert-Level%20Agentic%20Tasks%20for%20Code%20Security%20Vulnerability/tables/ff47bca88b08ccd90e497cc390015d0e760a5b3cd9e2dec93d2e54f733051478.jpg)

<a id="paper-0286"></a>

## Symmetry Reveals Layerwise Dynamics: How Transformers Perform In-Context Classification


### Images

![03fd598cf9f53c0e34633879ec0f825f6cbb171a53937a50ed947fdb0d66a1ac.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/03fd598cf9f53c0e34633879ec0f825f6cbb171a53937a50ed947fdb0d66a1ac.jpg)

![0971cde09a680d0c1bae9f434100551ca7ba3494a28f128d22dc5bdeb8dae1ab.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/0971cde09a680d0c1bae9f434100551ca7ba3494a28f128d22dc5bdeb8dae1ab.jpg)

![09f4ff9e095004277693fe62d2c466dec87051ad477fd87c1826e5a5306c37ea.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/09f4ff9e095004277693fe62d2c466dec87051ad477fd87c1826e5a5306c37ea.jpg)

![0eef99a0d748be16adcaa34bb968751a3a200e30ef15a74482d130df57760119.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/0eef99a0d748be16adcaa34bb968751a3a200e30ef15a74482d130df57760119.jpg)

![1f1ba5e7108b4378b0a7ba5fcc3704fd493df5452ae7d4daae131d52b9e8670b.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/1f1ba5e7108b4378b0a7ba5fcc3704fd493df5452ae7d4daae131d52b9e8670b.jpg)

![2eedfd30c93b81238cdf065fcbbe61d7633a572f0b855ba14174fe75e446eeab.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/2eedfd30c93b81238cdf065fcbbe61d7633a572f0b855ba14174fe75e446eeab.jpg)

![30421f7f28288c2cfa92a97dc51dc76c3ebc281ee554c1a99119a8e399f26abc.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/30421f7f28288c2cfa92a97dc51dc76c3ebc281ee554c1a99119a8e399f26abc.jpg)

![3401fb4a93b8fc9cc1b5ba358e177598737407833f5c74a39a2aec2303da7e15.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/3401fb4a93b8fc9cc1b5ba358e177598737407833f5c74a39a2aec2303da7e15.jpg)

![3a07ca5d1c336e90c77ffb43d1e6914c521b63f9daecd41e2a0d222f83df2e6e.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/3a07ca5d1c336e90c77ffb43d1e6914c521b63f9daecd41e2a0d222f83df2e6e.jpg)

![3f5a43ba0242b7885dd4daac6e394c9adcef67b4313f705668140f946d0eeed1.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/3f5a43ba0242b7885dd4daac6e394c9adcef67b4313f705668140f946d0eeed1.jpg)

![4accad1294227f77cbdbd089841e2931d77aaf17354b9b4ce7f59047ab9249ba.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/4accad1294227f77cbdbd089841e2931d77aaf17354b9b4ce7f59047ab9249ba.jpg)

![70f28538e8e5b0e947badeb8547078cc7c0ed1d745bd3b419be9d68452dfc750.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/70f28538e8e5b0e947badeb8547078cc7c0ed1d745bd3b419be9d68452dfc750.jpg)

![89df69e30fa512c78428b16187cc6afe0c0a7d682c54c8af299290c46e98a216.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/89df69e30fa512c78428b16187cc6afe0c0a7d682c54c8af299290c46e98a216.jpg)

![8a4c452ea8daa9b812a3fc8483b1d97e8a20b8196059e0a9ac6cd550f77feeb1.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/8a4c452ea8daa9b812a3fc8483b1d97e8a20b8196059e0a9ac6cd550f77feeb1.jpg)

![92ea6b8d079564e48a9ad6aa8cf23eb18db6e08204b615f6966eeb68b8b79cd0.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/92ea6b8d079564e48a9ad6aa8cf23eb18db6e08204b615f6966eeb68b8b79cd0.jpg)

![9364023cb8fbbca5d477381c357447a5415a2c3542d6f0fedef10aea266136f3.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/9364023cb8fbbca5d477381c357447a5415a2c3542d6f0fedef10aea266136f3.jpg)

![9aa7b24a5a7f4e9fcd386af98f902f2c2696cbd7a7704bb5bef2b4e13d9fd4c5.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/9aa7b24a5a7f4e9fcd386af98f902f2c2696cbd7a7704bb5bef2b4e13d9fd4c5.jpg)

![9c6a55856c137f92f418176241b8eb71f26ada7fb19af5b1e537f007478e4417.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/9c6a55856c137f92f418176241b8eb71f26ada7fb19af5b1e537f007478e4417.jpg)

![be09e1d59a7cf711380243add3b28ce21426b57c72a542f11e6e88db9473eb8e.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/be09e1d59a7cf711380243add3b28ce21426b57c72a542f11e6e88db9473eb8e.jpg)

![c11dc8c15f6028e0b709194d3f0a4c949db01b97bf938d99532af5640d3548cb.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/c11dc8c15f6028e0b709194d3f0a4c949db01b97bf938d99532af5640d3548cb.jpg)

![c74013fe0dc809551539d121a6e3b21d9cdd2485c15914de25bcfc062f048008.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/c74013fe0dc809551539d121a6e3b21d9cdd2485c15914de25bcfc062f048008.jpg)

![d1523d1dec55ce4135a8fdb292f8a5bb8265be48d9fa10f522332402fb9071af.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/d1523d1dec55ce4135a8fdb292f8a5bb8265be48d9fa10f522332402fb9071af.jpg)

![d88102bcfe03409b4e50213f58f60b683702ed83679adfcd4380556d948aae75.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/d88102bcfe03409b4e50213f58f60b683702ed83679adfcd4380556d948aae75.jpg)

![e5fa4ecf505aa22274e322467785012fd1ecf69ff4db74e91c3300660e42bc05.jpg](../icml_results/291_Symmetry%20Reveals%20Layerwise%20Dynamics_%20How%20Transformers%20Perform%20In-Context%20Classification/images/e5fa4ecf505aa22274e322467785012fd1ecf69ff4db74e91c3300660e42bc05.jpg)

<a id="paper-0287"></a>

## Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning


### Images

![0deb95853b892320b7255f686b5389df4879c240130dc723097bc7cde4a18b68.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/0deb95853b892320b7255f686b5389df4879c240130dc723097bc7cde4a18b68.jpg)

![22d13a578adb7bbff83a0c094518664cb27e5f78c1fecfb64d2f0d630a8e672a.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/22d13a578adb7bbff83a0c094518664cb27e5f78c1fecfb64d2f0d630a8e672a.jpg)

![2ad4c08f50b7cf02d6f2e7e1aaefe629df7bcf9e8cb69018fab5bc4d3652e055.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/2ad4c08f50b7cf02d6f2e7e1aaefe629df7bcf9e8cb69018fab5bc4d3652e055.jpg)

![2e21926def7ced3148850f8b4c2f4340721ff1684e482a0be58ca193e4a9d229.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/2e21926def7ced3148850f8b4c2f4340721ff1684e482a0be58ca193e4a9d229.jpg)

![34d6f6fa728575556b2315ce15318d40efdceb71da5494b7a88ce94108f54edb.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/34d6f6fa728575556b2315ce15318d40efdceb71da5494b7a88ce94108f54edb.jpg)

![36965304444ba6cef2f59f64ba16480dd74f8a5446305ea78cf38e3d04332436.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/36965304444ba6cef2f59f64ba16480dd74f8a5446305ea78cf38e3d04332436.jpg)

![44d3153979bd4aafc7f5537fa90ab98ea962ae76a25d230fa4d5003afa447bd9.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/44d3153979bd4aafc7f5537fa90ab98ea962ae76a25d230fa4d5003afa447bd9.jpg)

![50c33e82d03b69d02452011d3fe4baaf0b061954d4934ac8426f62c1e3d6ec4f.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/50c33e82d03b69d02452011d3fe4baaf0b061954d4934ac8426f62c1e3d6ec4f.jpg)

![655a50c11d46bb34e79c302ad7bc3c166ec3d18f2ee0a1a06795ac0a0a7a8d58.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/655a50c11d46bb34e79c302ad7bc3c166ec3d18f2ee0a1a06795ac0a0a7a8d58.jpg)

![6603a18f61d6a3bcb5b18db9bb242f6f5292f0558a1e0abe582f29b8b9951430.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/6603a18f61d6a3bcb5b18db9bb242f6f5292f0558a1e0abe582f29b8b9951430.jpg)

![6efc90f60bb96c13c70a384200787e6b0efd75b89564ae3e2a67c7e7d47eca7d.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/6efc90f60bb96c13c70a384200787e6b0efd75b89564ae3e2a67c7e7d47eca7d.jpg)

![7c44b15887dcfe53b68360226846cd6ae0e09529b708f5fc4a1f0d39e838dd7c.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/7c44b15887dcfe53b68360226846cd6ae0e09529b708f5fc4a1f0d39e838dd7c.jpg)

![7c9b0fcbb046819251e45db3abcdbb0f8bac09d7a3634a26d005149ee67bfde5.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/7c9b0fcbb046819251e45db3abcdbb0f8bac09d7a3634a26d005149ee67bfde5.jpg)

![908d5d756dee49ee59059085d562031cb73ae135e3ed4a8e16a14665304ab223.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/908d5d756dee49ee59059085d562031cb73ae135e3ed4a8e16a14665304ab223.jpg)

![b03e8098d208271523a83213393fa96323c24b807300e546aa378d7922d95629.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/b03e8098d208271523a83213393fa96323c24b807300e546aa378d7922d95629.jpg)

![be8cfc32c6b6f1691922354b68b4d190006caa444c0fb89341a7af7a26e7f51e.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/be8cfc32c6b6f1691922354b68b4d190006caa444c0fb89341a7af7a26e7f51e.jpg)

![beb6fca346767e4fa49c993620cd0faf05c4d24c4ec15e431965fe3a2cdfbfd8.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/beb6fca346767e4fa49c993620cd0faf05c4d24c4ec15e431965fe3a2cdfbfd8.jpg)

![da78afe112049591c9e08be298246495b1592b3541652d39f70d174de945e227.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/da78afe112049591c9e08be298246495b1592b3541652d39f70d174de945e227.jpg)

![e63fe9e463b04afbd0d467ddf9a8e129c7e95d6f68c8151e5ad5ddc0da260e6a.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/images/e63fe9e463b04afbd0d467ddf9a8e129c7e95d6f68c8151e5ad5ddc0da260e6a.jpg)

### Tables

![1521c8d4fbdf8e91956a6004c78f72d2f7c0f1bcc9e55001d9a210ee155d3416.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/1521c8d4fbdf8e91956a6004c78f72d2f7c0f1bcc9e55001d9a210ee155d3416.jpg)

![2ea0a4889ce53e1d7e3cf48bd6df6002f959b65eb70d4791c650cd16e59ac312.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/2ea0a4889ce53e1d7e3cf48bd6df6002f959b65eb70d4791c650cd16e59ac312.jpg)

![3ee9ef42050cdd54e6579f0a1100560d5469a877474ee6af8788a0613ae716e0.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/3ee9ef42050cdd54e6579f0a1100560d5469a877474ee6af8788a0613ae716e0.jpg)

![40d318e83f3c905dd4bc6fba81b7e30fa68e6dc71878207f6df3b8e3f9e40b13.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/40d318e83f3c905dd4bc6fba81b7e30fa68e6dc71878207f6df3b8e3f9e40b13.jpg)

![4a14c1cf809883e62e62a31a4ac6001234fd384ec36ea2bf146e9b8bbac6bbca.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/4a14c1cf809883e62e62a31a4ac6001234fd384ec36ea2bf146e9b8bbac6bbca.jpg)

![604cef183c68dcd305dba5f4641b45722bbbe37e124bb8b2c0d48d3f3bffa313.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/604cef183c68dcd305dba5f4641b45722bbbe37e124bb8b2c0d48d3f3bffa313.jpg)

![66d38d14ab740560b2dc6da989f7386f6fd1751aaa16d37e2f93f826f4738cd1.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/66d38d14ab740560b2dc6da989f7386f6fd1751aaa16d37e2f93f826f4738cd1.jpg)

![720d04fb7f105b6f732d90eccb563ff84adba684e51a2297e01cdff93d353e4e.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/720d04fb7f105b6f732d90eccb563ff84adba684e51a2297e01cdff93d353e4e.jpg)

![d45d9d83a057cf73919df9e2727fab8fd44ab47ab517c87025efb90d0a4c9eb6.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/d45d9d83a057cf73919df9e2727fab8fd44ab47ab517c87025efb90d0a4c9eb6.jpg)

![d6c3d4d91150bc41663ea8a83243d8fb8b59e1dd7299e3ff397c7c2cdf855e50.jpg](../icml_results/292_Video-Based%20Optimal%20Transport%20for%20Feedback-Efficient%20Offline%20Preference-Based%20Reinforcement%20Learning/tables/d6c3d4d91150bc41663ea8a83243d8fb8b59e1dd7299e3ff397c7c2cdf855e50.jpg)

<a id="paper-0288"></a>

## Faster Activation Functions at the Edge for Post-Training Speedups


### Images

![191adac757eb93a266b03d31b18d84e63358f3b3c23c95f15f5856ffc2c126be.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/191adac757eb93a266b03d31b18d84e63358f3b3c23c95f15f5856ffc2c126be.jpg)

![3a72fec6f00e458116e033f10e77b25029beb5baf4644f3f3e47f046c164df28.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/3a72fec6f00e458116e033f10e77b25029beb5baf4644f3f3e47f046c164df28.jpg)

![3a864bda889f5526b386b9a9746dd9166e2fe6cd9ba97b806c7b4db75eb43bf8.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/3a864bda889f5526b386b9a9746dd9166e2fe6cd9ba97b806c7b4db75eb43bf8.jpg)

![3ca8b735dc7920cce96643a6022616de7d15175a3b3e0c9834615b0ce10a46b6.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/3ca8b735dc7920cce96643a6022616de7d15175a3b3e0c9834615b0ce10a46b6.jpg)

![4643f6be385f23b9ec5a5b49a7d15e51383d0bf9fd678bea10c0ce526721caed.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/4643f6be385f23b9ec5a5b49a7d15e51383d0bf9fd678bea10c0ce526721caed.jpg)

![482b304d4662564e3e3c3732dc3040907b9c3056a22366e5701d352a18a64311.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/482b304d4662564e3e3c3732dc3040907b9c3056a22366e5701d352a18a64311.jpg)

![81064206bd768e2066dad27f4d6c3e81c36f09a89a6ba64c3c167d054a1dc083.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/81064206bd768e2066dad27f4d6c3e81c36f09a89a6ba64c3c167d054a1dc083.jpg)

![b37204966ce0b137c6b37768e88aa0782c13ff2af839354d3bbe2b3a35f8c651.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/b37204966ce0b137c6b37768e88aa0782c13ff2af839354d3bbe2b3a35f8c651.jpg)

![bcf5ce9e0743addb758cf82bce54409c2ee9e5e84167660e0feb3ea4b6dd67bc.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/images/bcf5ce9e0743addb758cf82bce54409c2ee9e5e84167660e0feb3ea4b6dd67bc.jpg)

### Tables

![423986971aa5b9d1574d51757b99d5f69a802b981a8e2b1041d8523befb659ce.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/tables/423986971aa5b9d1574d51757b99d5f69a802b981a8e2b1041d8523befb659ce.jpg)

![52c41e9956868aad4723b5393a4a6881496d9a564bc14b383ad1503f0773cba8.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/tables/52c41e9956868aad4723b5393a4a6881496d9a564bc14b383ad1503f0773cba8.jpg)

![5bf6de7147727f7cae9c92c24fec489fa8eb712923ff37700cde7f76ce8cd543.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/tables/5bf6de7147727f7cae9c92c24fec489fa8eb712923ff37700cde7f76ce8cd543.jpg)

![b6f3a5af3b9d6679966829d40cf1fea0c68291ee431bc9472ba52b9a2adc1302.jpg](../icml_results/293_Faster%20Activation%20Functions%20at%20the%20Edge%20for%20Post-Training%20Speedups/tables/b6f3a5af3b9d6679966829d40cf1fea0c68291ee431bc9472ba52b9a2adc1302.jpg)

<a id="paper-0289"></a>

## Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model


### Images

![0c6ae2c66eb128d72be9df3e4c9105f0fc262bac6fa46612bfa05b4bd6dd0dc5.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/0c6ae2c66eb128d72be9df3e4c9105f0fc262bac6fa46612bfa05b4bd6dd0dc5.jpg)

![232b755a2ed37e326a45f326b00392a4c135a7a5650727be6c67120a033dfc11.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/232b755a2ed37e326a45f326b00392a4c135a7a5650727be6c67120a033dfc11.jpg)

![28d9db5cea74ef0aab8b2f5483cc8b97162375791c60675471125bd2e0fdeac9.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/28d9db5cea74ef0aab8b2f5483cc8b97162375791c60675471125bd2e0fdeac9.jpg)

![30463ebc51e22f04dab7a51d82c65421cd757d3a6f1b8720b891f8ab89d6fb78.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/30463ebc51e22f04dab7a51d82c65421cd757d3a6f1b8720b891f8ab89d6fb78.jpg)

![3759c0f7c40ba6b8aaa0a2b31bba34459be31eeb7f3650111f5fd260cd5f1931.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/3759c0f7c40ba6b8aaa0a2b31bba34459be31eeb7f3650111f5fd260cd5f1931.jpg)

![50ec22f2a67d7347b8883c558fd4cd7dc13453eda5ab547458ce799a0f2b6811.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/50ec22f2a67d7347b8883c558fd4cd7dc13453eda5ab547458ce799a0f2b6811.jpg)

![59d8e97f2e2ae007f571a71cacd588aa5ce6393432a47b6e1617111810d8dff8.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/59d8e97f2e2ae007f571a71cacd588aa5ce6393432a47b6e1617111810d8dff8.jpg)

![7167804b2b4456f2b69d1f5f3f3b9f1d0531c8323fc57ac67252b3705d128c17.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/7167804b2b4456f2b69d1f5f3f3b9f1d0531c8323fc57ac67252b3705d128c17.jpg)

![71ba48a4af3d7f943c0e32bf87f78869d535cfe635d34fad1f9312eec53d1552.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/71ba48a4af3d7f943c0e32bf87f78869d535cfe635d34fad1f9312eec53d1552.jpg)

![79faddc8a7be1693cc9b49e8b7a18b5a0d255989a688b006c4da8cb733474687.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/79faddc8a7be1693cc9b49e8b7a18b5a0d255989a688b006c4da8cb733474687.jpg)

![7ab382a84686dbd4cde29247af9c0d784f1edf32457632d0ea08c7ff6b8c3055.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/7ab382a84686dbd4cde29247af9c0d784f1edf32457632d0ea08c7ff6b8c3055.jpg)

![7c73a52d7c8cf9e6fd93b12472531d6cc903424517681bcc4345ae700171f0a9.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/7c73a52d7c8cf9e6fd93b12472531d6cc903424517681bcc4345ae700171f0a9.jpg)

![8feb5dd3f067cce44e5920c1c0885778b3130944909f62d73f60ba4164b94695.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/8feb5dd3f067cce44e5920c1c0885778b3130944909f62d73f60ba4164b94695.jpg)

![9ac1e7e0266d9673dc32d32674f4c9edc69e9c21f691d43f63482f9752fd124a.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/9ac1e7e0266d9673dc32d32674f4c9edc69e9c21f691d43f63482f9752fd124a.jpg)

![a20ab8dff42f6f65f47d19ec427ae072068c2f73da8dbcdc91f11ccd73011065.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/a20ab8dff42f6f65f47d19ec427ae072068c2f73da8dbcdc91f11ccd73011065.jpg)

![a4783e11d85aedf87026160101828700efaec9e9c5059a06b2fa43f32bf0e7be.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/a4783e11d85aedf87026160101828700efaec9e9c5059a06b2fa43f32bf0e7be.jpg)

![c142a1d57726b9bb67c0e68c475ec2dc73764896f8f2454c259b82fcd08c071f.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/c142a1d57726b9bb67c0e68c475ec2dc73764896f8f2454c259b82fcd08c071f.jpg)

![ce9d44cd53f805bd9d8533d0c0a992cd047110b6decc48d7d1d304ff47d67290.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/ce9d44cd53f805bd9d8533d0c0a992cd047110b6decc48d7d1d304ff47d67290.jpg)

![d2779baf6f5bfc1582f50fb19a01a9aa969fd67b68f07060083e18610296f59a.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/d2779baf6f5bfc1582f50fb19a01a9aa969fd67b68f07060083e18610296f59a.jpg)

![de66dedc72a3579c1819d8b1df0e2072b5f8ea23691a51298b345f9f7bcd9f9f.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/de66dedc72a3579c1819d8b1df0e2072b5f8ea23691a51298b345f9f7bcd9f9f.jpg)

![fb266ac0fd00cb2b19bc9f51d2dbdb212604cc241b7a5141ed91abb6256c24fd.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/images/fb266ac0fd00cb2b19bc9f51d2dbdb212604cc241b7a5141ed91abb6256c24fd.jpg)

### Tables

![3cf1c2be27685b4859f28210649481152e445b9cf004c62da9b08de4b701e139.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/tables/3cf1c2be27685b4859f28210649481152e445b9cf004c62da9b08de4b701e139.jpg)

![6f6f5c88e8262910cc8cd9567014e2f1a4537756a3ff8e70adce4ea11629c121.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/tables/6f6f5c88e8262910cc8cd9567014e2f1a4537756a3ff8e70adce4ea11629c121.jpg)

![9464025aec7d88a1984f340c3afeea29d4f80f5201e8d2c3555109222c6e52c9.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/tables/9464025aec7d88a1984f340c3afeea29d4f80f5201e8d2c3555109222c6e52c9.jpg)

![fe79339f330dc1ebd340a51a6bc447048e9dd03ac3b2875b57afff5d28e5b780.jpg](../icml_results/294_Flex-Forcing_%20Towards%20a%20Unified%20Autoregressive%20and%20Bidirectional%20Video%20Diffusion%20Model/tables/fe79339f330dc1ebd340a51a6bc447048e9dd03ac3b2875b57afff5d28e5b780.jpg)

<a id="paper-0290"></a>

## Breaking the Reversal Curse in Autoregressive Language Models via Identity Bridge


### Images

![0b997684e03c09dcf546af2efab836f4990cbed984b0bf1ed242ba2958cff87b.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/0b997684e03c09dcf546af2efab836f4990cbed984b0bf1ed242ba2958cff87b.jpg)

![2cc1ae359d5d4eea383c154bdc4dd8f29172addb46b8829f2bd3996cc4b95f6d.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/2cc1ae359d5d4eea383c154bdc4dd8f29172addb46b8829f2bd3996cc4b95f6d.jpg)

![34fd14ca188d481374a773504b196f8deedca7b2f28f7cab18b856cd3c7a5c1b.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/34fd14ca188d481374a773504b196f8deedca7b2f28f7cab18b856cd3c7a5c1b.jpg)

![8db598edf878eaf7e9837187bcc89c379537b24ba07be0a318bba34ba56fad76.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/8db598edf878eaf7e9837187bcc89c379537b24ba07be0a318bba34ba56fad76.jpg)

![902fbbf6e1fe55248b4405fe51361401ba4e9163a1dd5033784669f1318cec6b.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/902fbbf6e1fe55248b4405fe51361401ba4e9163a1dd5033784669f1318cec6b.jpg)

![94b7a0488c1dfe54c547e4f077eabdaa3fe9f49da102b2086f5988bf425c87f4.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/94b7a0488c1dfe54c547e4f077eabdaa3fe9f49da102b2086f5988bf425c87f4.jpg)

![a60b6b6b4875f1a79f54e775f4d6f6f6fa2c28ceb6647bcd8d9d646c4cd9fbfd.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/a60b6b6b4875f1a79f54e775f4d6f6f6fa2c28ceb6647bcd8d9d646c4cd9fbfd.jpg)

![b547b8fe05e61dca5d8ecf71d504e9659cb61600c259c7bbf32edda39f959e5c.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/b547b8fe05e61dca5d8ecf71d504e9659cb61600c259c7bbf32edda39f959e5c.jpg)

![c69fc506c96b25070a9a573a6c0c3508116f95885a0c9d067782e7017cfa2e4a.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/c69fc506c96b25070a9a573a6c0c3508116f95885a0c9d067782e7017cfa2e4a.jpg)

![d514b0ad47ad3bdf1a7310c32523c02e0a4ea85e03f11f554d598345aaf03543.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/d514b0ad47ad3bdf1a7310c32523c02e0a4ea85e03f11f554d598345aaf03543.jpg)

![dfffc7db2e0e516fbca9527127b9a919d4a2ceacc6250107b8917d66bde2f78d.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/images/dfffc7db2e0e516fbca9527127b9a919d4a2ceacc6250107b8917d66bde2f78d.jpg)

### Tables

![9b0948e753189ac144164a92b369e5150a1ddf430e1acc4b0f9ad92b7792e271.jpg](../icml_results/295_Breaking%20the%20Reversal%20Curse%20in%20Autoregressive%20Language%20Models%20via%20Identity%20Bridge/tables/9b0948e753189ac144164a92b369e5150a1ddf430e1acc4b0f9ad92b7792e271.jpg)

<a id="paper-0291"></a>

## What Makes Value Learning Efficient in Residual Reinforcement Learning?


### Images

![0f6f44a7e960aa00cfb798fada84e09913f90565abb928bddfd915a8edfe4397.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/0f6f44a7e960aa00cfb798fada84e09913f90565abb928bddfd915a8edfe4397.jpg)

![0f821da98b6f4dd6dc32f1b76be1df87175b2cfb946a9cc0b599c6406a0d9c31.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/0f821da98b6f4dd6dc32f1b76be1df87175b2cfb946a9cc0b599c6406a0d9c31.jpg)

![134dcc383540465ccb702a8fc6cce6c0b5e90c3e9e421a21d45cce19bca64b07.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/134dcc383540465ccb702a8fc6cce6c0b5e90c3e9e421a21d45cce19bca64b07.jpg)

![148c632b54f024309b4e5c7f7c0a464b1070ae357aab9ec6f322d034f2c2331b.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/148c632b54f024309b4e5c7f7c0a464b1070ae357aab9ec6f322d034f2c2331b.jpg)

![1524d4f885052020d92107bd42dcf61adc5abc3f34aceaa15b12532e3345b6fe.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/1524d4f885052020d92107bd42dcf61adc5abc3f34aceaa15b12532e3345b6fe.jpg)

![1d899e134af82e7f630969e9ea3b136f6dfbf23788d231e1de7d56b719b2ca47.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/1d899e134af82e7f630969e9ea3b136f6dfbf23788d231e1de7d56b719b2ca47.jpg)

![2c22edb8dd3657654ee2522798f162ca0332343e7ec7086733c71b1efd97dcdd.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/2c22edb8dd3657654ee2522798f162ca0332343e7ec7086733c71b1efd97dcdd.jpg)

![2d58482c6d323cdf069845aee0550bb8f062fb91e034b11fe9e2205ad67892e7.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/2d58482c6d323cdf069845aee0550bb8f062fb91e034b11fe9e2205ad67892e7.jpg)

![3a92345dd3788187757b113f596725a0b9584818ea550d2908a514e6bad17800.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/3a92345dd3788187757b113f596725a0b9584818ea550d2908a514e6bad17800.jpg)

![3b862d1aef8051f41daae417f7c0654e37df69a414f9fd089f7b53d439833854.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/3b862d1aef8051f41daae417f7c0654e37df69a414f9fd089f7b53d439833854.jpg)

![44c6a4a4d7312d6feb82ee2429e76304591b08f3d31e434adf6b4f12d4cdf2a6.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/44c6a4a4d7312d6feb82ee2429e76304591b08f3d31e434adf6b4f12d4cdf2a6.jpg)

![543f7a37b32a9a362596cf219424754b4f39211f5678555755c8e801bfc119c8.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/543f7a37b32a9a362596cf219424754b4f39211f5678555755c8e801bfc119c8.jpg)

![5fc98f343672ecbda6ad435e8eb47b2c5a98592eb918b2d267b5a13370caefec.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/5fc98f343672ecbda6ad435e8eb47b2c5a98592eb918b2d267b5a13370caefec.jpg)

![65199db384127ac912294372f80a4933839c8763c3743dacd342cd28acb3d67b.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/65199db384127ac912294372f80a4933839c8763c3743dacd342cd28acb3d67b.jpg)

![653cc7e7433c037d5ceebeb348c6c6c32d05992906bba27b3a0dc86b14ad9822.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/653cc7e7433c037d5ceebeb348c6c6c32d05992906bba27b3a0dc86b14ad9822.jpg)

![672e2aa7096413fb082ce4f0829088ba21e1f37c34c61dcd6a5c718aff49f556.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/672e2aa7096413fb082ce4f0829088ba21e1f37c34c61dcd6a5c718aff49f556.jpg)

![6b5f90cd13e9af773eebc57b8c575cd8de48b572b23759b93549eb22ac742a48.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/6b5f90cd13e9af773eebc57b8c575cd8de48b572b23759b93549eb22ac742a48.jpg)

![78d967741996614cf69453654736257b5b0e9f64a4da0887266078d047f6d511.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/78d967741996614cf69453654736257b5b0e9f64a4da0887266078d047f6d511.jpg)

![845ebd49ebc6ec9d32db68c533b46fcab710e367356b0d4ce763077873d16449.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/845ebd49ebc6ec9d32db68c533b46fcab710e367356b0d4ce763077873d16449.jpg)

![892372ce3945e4e3945cc30236f93eda3efd66ad9ebd85970f94adec877a156d.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/892372ce3945e4e3945cc30236f93eda3efd66ad9ebd85970f94adec877a156d.jpg)

![8dbc68823f9f3013f509054a86ad9aabae10680f5cd786849136b468a467690c.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/8dbc68823f9f3013f509054a86ad9aabae10680f5cd786849136b468a467690c.jpg)

![9ab19c4a02b22a85ddfc9c6311973db5b0c03a2f906e49e55886e9d1533fd7e1.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/9ab19c4a02b22a85ddfc9c6311973db5b0c03a2f906e49e55886e9d1533fd7e1.jpg)

![af098c1b999c737b142a08893ddbf65d917bf6ff9fb2c9839b4ef59ef0b5c831.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/af098c1b999c737b142a08893ddbf65d917bf6ff9fb2c9839b4ef59ef0b5c831.jpg)

![b7297ab534d027a66d3d5833e1cb0f24201acce8489448c9d6b52421b3636431.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/b7297ab534d027a66d3d5833e1cb0f24201acce8489448c9d6b52421b3636431.jpg)

![c473a900fbb641b06d6e8c0abb3614a4356f5040d4df1ea2ffd519cf59ee7441.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/c473a900fbb641b06d6e8c0abb3614a4356f5040d4df1ea2ffd519cf59ee7441.jpg)

![dbc474edc54182764c4a4ce9d4765086b7045ea9206e6ff5afccfbc0da96931e.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/dbc474edc54182764c4a4ce9d4765086b7045ea9206e6ff5afccfbc0da96931e.jpg)

![e1ec4f93d29c27b77925509b3c043e81fa1bda4c32c6ed0109d0975a20c4bac2.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/e1ec4f93d29c27b77925509b3c043e81fa1bda4c32c6ed0109d0975a20c4bac2.jpg)

![e30cb4ed787780d74a2d6a88ac44e151124c055b616610c933de8d3091c745f5.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/e30cb4ed787780d74a2d6a88ac44e151124c055b616610c933de8d3091c745f5.jpg)

![f33f13d89ecba851d78fbb137b4506fa16143c0b242d2c2ca9764c05d62f6d9e.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/images/f33f13d89ecba851d78fbb137b4506fa16143c0b242d2c2ca9764c05d62f6d9e.jpg)

### Tables

![2612f3b00ff035c16077e9a8bea40a8db23d081983e57ebd1bdac31c2c58b96e.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/2612f3b00ff035c16077e9a8bea40a8db23d081983e57ebd1bdac31c2c58b96e.jpg)

![3b2c32e36d7d416d9f4c606842dc40277d86341468c7e5eeb614053c13f4c193.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/3b2c32e36d7d416d9f4c606842dc40277d86341468c7e5eeb614053c13f4c193.jpg)

![586f31ff7f14b77f5d1e3abf41e235423c8656cd25b0080fbd46385223531825.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/586f31ff7f14b77f5d1e3abf41e235423c8656cd25b0080fbd46385223531825.jpg)

![696f27e3021843d4fb47095cc2328a1c05305260e49e436199b208856f88e3cc.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/696f27e3021843d4fb47095cc2328a1c05305260e49e436199b208856f88e3cc.jpg)

![b7eb69bd00e1940ffc3d014ccfd931693faa9b1183b72eca31b4844a2e5ae96a.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/b7eb69bd00e1940ffc3d014ccfd931693faa9b1183b72eca31b4844a2e5ae96a.jpg)

![c21190f236eab7eb74e6c7f7da3bf159826715603ebbba472ce1ca9613e0c4ef.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/c21190f236eab7eb74e6c7f7da3bf159826715603ebbba472ce1ca9613e0c4ef.jpg)

![cb37e913f9a15b20c9ca82acf73428f8927b57b295dad5f44b13ded76711bbdb.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/cb37e913f9a15b20c9ca82acf73428f8927b57b295dad5f44b13ded76711bbdb.jpg)

![dc2a2e17472623657fee93ec1abdaf770df3fea6d35bcbe6e3e6bae3ceb3a668.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/dc2a2e17472623657fee93ec1abdaf770df3fea6d35bcbe6e3e6bae3ceb3a668.jpg)

![f9f536167f7b115e4fada0c46c14aaf08417148c9c13f245f9af99b4a41d4724.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/f9f536167f7b115e4fada0c46c14aaf08417148c9c13f245f9af99b4a41d4724.jpg)

![fd52fb1fab811b560502985f9c0ffee19b15af06878cde87a5ffea18fce3a0da.jpg](../icml_results/296_What%20Makes%20Value%20Learning%20Efficient%20in%20Residual%20Reinforcement%20Learning_/tables/fd52fb1fab811b560502985f9c0ffee19b15af06878cde87a5ffea18fce3a0da.jpg)

<a id="paper-0292"></a>

## Dual-Latent Memory Routing for Vision-Language Reasoning


### Images

![1550f92292400119ade1046f97beb29cd7d57836b98a7f62ed0561dadcbe1f07.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/1550f92292400119ade1046f97beb29cd7d57836b98a7f62ed0561dadcbe1f07.jpg)

![42c9c1aaea6ddc5b273c7b522eda81d2e9e5ab28465e1ef5fb1a1d4d773360cc.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/42c9c1aaea6ddc5b273c7b522eda81d2e9e5ab28465e1ef5fb1a1d4d773360cc.jpg)

![7600740ee91cd84a15ba5d945d508829fa893795202191f9896743cf4d33e846.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/7600740ee91cd84a15ba5d945d508829fa893795202191f9896743cf4d33e846.jpg)

![a24b026887d23beda4e07379a7e2fce7b099056c8b05e9ab0d7160f6911b432e.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/a24b026887d23beda4e07379a7e2fce7b099056c8b05e9ab0d7160f6911b432e.jpg)

![bbe949f16a4657dfb22229d4b5f79ad47016e17f04ffda2443f98e5513d046d2.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/bbe949f16a4657dfb22229d4b5f79ad47016e17f04ffda2443f98e5513d046d2.jpg)

![c65fb5fb81c4921dbbe9b4a876b77408c6c2bd4a3978ee0860200a9ea90b656b.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/c65fb5fb81c4921dbbe9b4a876b77408c6c2bd4a3978ee0860200a9ea90b656b.jpg)

![e7c01c0adb158f9d01f419dca77c6560171e5d57f4450ea8f55e94e0777f3eb4.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/images/e7c01c0adb158f9d01f419dca77c6560171e5d57f4450ea8f55e94e0777f3eb4.jpg)

### Tables

![055216f70339602a55da28b52171c048d7f8c3ccd81115a44772dbdca2effb3c.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/055216f70339602a55da28b52171c048d7f8c3ccd81115a44772dbdca2effb3c.jpg)

![13fe5963cf61877de80125cccab8c5c768ec8082f98fda399dd0ee01d58d3d08.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/13fe5963cf61877de80125cccab8c5c768ec8082f98fda399dd0ee01d58d3d08.jpg)

![2142461b2f9043f2b460f782a431f7de1676dc70c88d9dda46cf6373e62fc708.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/2142461b2f9043f2b460f782a431f7de1676dc70c88d9dda46cf6373e62fc708.jpg)

![2f08cdc54293c1a13fb15d31107138e4cb8e664ca8241a11320e63e7f81d1cbe.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/2f08cdc54293c1a13fb15d31107138e4cb8e664ca8241a11320e63e7f81d1cbe.jpg)

![311feca17a2f4740e5bf8dbe1b32fb153a142abfd0ebba942c525fe6017ae5ea.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/311feca17a2f4740e5bf8dbe1b32fb153a142abfd0ebba942c525fe6017ae5ea.jpg)

![50bee4251311abecc746da59f26efbce5c88986427535f06a35a618b2c8752c9.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/50bee4251311abecc746da59f26efbce5c88986427535f06a35a618b2c8752c9.jpg)

![9db95b7d0561555686ae5d1456334aec4af5fd5c49ee5f297725c5dcb5b49e87.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/9db95b7d0561555686ae5d1456334aec4af5fd5c49ee5f297725c5dcb5b49e87.jpg)

![a9fc54e4b38ea8d2ab5a7c31c85fb8a7186050ea70f276157121be590baa971f.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/a9fc54e4b38ea8d2ab5a7c31c85fb8a7186050ea70f276157121be590baa971f.jpg)

![bc58c3363f5596bc703cef1fb70bc6b71739f2a64fee6c74c44a0d2f36588ce3.jpg](../icml_results/297_Dual-Latent%20Memory%20Routing%20for%20Vision-Language%20Reasoning/tables/bc58c3363f5596bc703cef1fb70bc6b71739f2a64fee6c74c44a0d2f36588ce3.jpg)

<a id="paper-0293"></a>

## ScaleMoE: Mixture-of-Experts for Scalable Continuous Control in Actor-Critic Reinforcement Learning


### Images

![16f063533ba6198e01e58f2fb00bfae18ca5cf53faa996b255f4e326eff207b8.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/16f063533ba6198e01e58f2fb00bfae18ca5cf53faa996b255f4e326eff207b8.jpg)

![176fea166a0b4178cc126d2634e61925a66328315f3560c52699fe21063dc2a8.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/176fea166a0b4178cc126d2634e61925a66328315f3560c52699fe21063dc2a8.jpg)

![1f8141ce74a64c81720a4898e60a9529fbd31dc649f9f8cacd76492287a1c263.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/1f8141ce74a64c81720a4898e60a9529fbd31dc649f9f8cacd76492287a1c263.jpg)

![249e8d016f74ef804f7da76a933a3beac0ee94f2f39f3af140a4f9d54c00602f.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/249e8d016f74ef804f7da76a933a3beac0ee94f2f39f3af140a4f9d54c00602f.jpg)

![37748974366f741da56a18f4153d9c004fcc8e86b49b4a074bc258d25cd362cf.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/37748974366f741da56a18f4153d9c004fcc8e86b49b4a074bc258d25cd362cf.jpg)

![37b30bd827baa10d25489d11800b420db757803e2cadfb5191c12f5f74dd46ec.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/37b30bd827baa10d25489d11800b420db757803e2cadfb5191c12f5f74dd46ec.jpg)

![37bfe77b2297c82f26c52ab16437265b11d158075019e7f7bfa9f34808944817.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/37bfe77b2297c82f26c52ab16437265b11d158075019e7f7bfa9f34808944817.jpg)

![387ba34e5caf1882c58ec8e56140901f5b2022164a21b3e0e914fab08b575cae.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/387ba34e5caf1882c58ec8e56140901f5b2022164a21b3e0e914fab08b575cae.jpg)

![4277596f7582cf021ddd948b327657d2c382f9d3c279163f5d2c5e579d4b14ec.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/4277596f7582cf021ddd948b327657d2c382f9d3c279163f5d2c5e579d4b14ec.jpg)

![43f0c2feaf3ca00c88075d1bd28326a2122023aa6eaeafb4b2aa781fbbf3cdef.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/43f0c2feaf3ca00c88075d1bd28326a2122023aa6eaeafb4b2aa781fbbf3cdef.jpg)

![537ba1cdc5e0b47bb0b8a0fa58c2c52f52d186e38ae34d8872328b45bfd14a8d.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/537ba1cdc5e0b47bb0b8a0fa58c2c52f52d186e38ae34d8872328b45bfd14a8d.jpg)

![54eecc7e5c0c804dcadf16b81d0d01cb5c49d4e6bc8fd39c73feb370d6787c54.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/54eecc7e5c0c804dcadf16b81d0d01cb5c49d4e6bc8fd39c73feb370d6787c54.jpg)

![5f0b68e9779e209f5c37e85a9f159411ff691ceff3a9240b212c6f78d03d10ff.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/5f0b68e9779e209f5c37e85a9f159411ff691ceff3a9240b212c6f78d03d10ff.jpg)

![64bb2a29676308d7bb544c0ac8e05dc54475489e62c6ca10fb4db953e5cff468.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/64bb2a29676308d7bb544c0ac8e05dc54475489e62c6ca10fb4db953e5cff468.jpg)

![687f12746cdc9491f83923acea72ddb33a2ebe1f19593f64a97421f37e69a2db.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/687f12746cdc9491f83923acea72ddb33a2ebe1f19593f64a97421f37e69a2db.jpg)

![720852cfbd24ec270d3f9b69cc43a4992558df04f6ec651d58c348c504a22eff.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/720852cfbd24ec270d3f9b69cc43a4992558df04f6ec651d58c348c504a22eff.jpg)

![90a1035361178431aefe0dd79888fc326c4704e7822a5e949455917f10f41b1c.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/90a1035361178431aefe0dd79888fc326c4704e7822a5e949455917f10f41b1c.jpg)

![9cbe047f5ba30ae617c132d953fcb58f1ae31613b1da79828b4814a693155f91.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/9cbe047f5ba30ae617c132d953fcb58f1ae31613b1da79828b4814a693155f91.jpg)

![9edd6f2166d669628403c96135120bf36c207d9465ffe3747de852522aaeb4fd.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/9edd6f2166d669628403c96135120bf36c207d9465ffe3747de852522aaeb4fd.jpg)

![b0730458cbfec0d19574f07b05473bb40e51f1fa030ad0bc436f480362d69ca6.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/b0730458cbfec0d19574f07b05473bb40e51f1fa030ad0bc436f480362d69ca6.jpg)

![b82980c1aff7fd88387f39b108fcfcdb81ed1154fa96b8e06ddf9b47c41308d6.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/b82980c1aff7fd88387f39b108fcfcdb81ed1154fa96b8e06ddf9b47c41308d6.jpg)

![b938bb2b6a4620e053938542e8d7bd69da1e9ebe255ab68f67e4e2e704f58ead.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/b938bb2b6a4620e053938542e8d7bd69da1e9ebe255ab68f67e4e2e704f58ead.jpg)

![c4186741e27aa004c3253db41ff197755946b68c69145fccad827d877d50cea1.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/c4186741e27aa004c3253db41ff197755946b68c69145fccad827d877d50cea1.jpg)

![c908b104eda2fe33ba4d2b5d96395e2e32af7f9c1de0a37683b3c8a6283d4be4.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/c908b104eda2fe33ba4d2b5d96395e2e32af7f9c1de0a37683b3c8a6283d4be4.jpg)

![d02c041d1bb8743176edcbfe9624a8e00839a2039a9e0db1ec9379bad387a107.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/d02c041d1bb8743176edcbfe9624a8e00839a2039a9e0db1ec9379bad387a107.jpg)

![d2212d7905d4b3f49f40cd0cac529cb610101f1087ab23409f56631d13c740b2.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/d2212d7905d4b3f49f40cd0cac529cb610101f1087ab23409f56631d13c740b2.jpg)

![d5a92b8216b0332ab7bb4649dbf13a66b71c4b6e2a88cbf89aa6c43d493f5307.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/d5a92b8216b0332ab7bb4649dbf13a66b71c4b6e2a88cbf89aa6c43d493f5307.jpg)

![d894d1d8d6b105444384aa267257a33d12463baaef32e4f7d6ac18d87d3d56c2.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/d894d1d8d6b105444384aa267257a33d12463baaef32e4f7d6ac18d87d3d56c2.jpg)

![d90ddd00c3ee3f4d83037db1a0f894fa2081d5fdedb5d2a0bbdcfb0a5d804343.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/d90ddd00c3ee3f4d83037db1a0f894fa2081d5fdedb5d2a0bbdcfb0a5d804343.jpg)

![ec55867b8dd60766e2008faf3091b87fd1acba4c655522d44dccdc119b3395a8.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/ec55867b8dd60766e2008faf3091b87fd1acba4c655522d44dccdc119b3395a8.jpg)

![f18e493fbecc68761d5a032a97f51f21ebd2252d36dc05a78634b3cb81386b70.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/f18e493fbecc68761d5a032a97f51f21ebd2252d36dc05a78634b3cb81386b70.jpg)

![f450438209d34d3068d6040401b8bda96fb4bf1357f2c4e8f414947b4953cc76.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/f450438209d34d3068d6040401b8bda96fb4bf1357f2c4e8f414947b4953cc76.jpg)

![f5a19b2aeea5081421e8ab3a3aeff224c2b26f7d524a31b63f8ace1226b017a8.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/f5a19b2aeea5081421e8ab3a3aeff224c2b26f7d524a31b63f8ace1226b017a8.jpg)

![f5b28d8f5da62bee32960ebfbd3684a548b368fdeef34ef3ceafcba600d16692.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/f5b28d8f5da62bee32960ebfbd3684a548b368fdeef34ef3ceafcba600d16692.jpg)

![f9a8fcd861cd5104ec0d4b89882cf413bd13749c5b84c125a3f1ec51fbb3624f.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/f9a8fcd861cd5104ec0d4b89882cf413bd13749c5b84c125a3f1ec51fbb3624f.jpg)

![fcb7648da96c0d23dec37a0f941f24b52f13e9ee4a22f546ada1b78fc1ee774a.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/fcb7648da96c0d23dec37a0f941f24b52f13e9ee4a22f546ada1b78fc1ee774a.jpg)

![fd4759fd3320e42967b9d8afc5cb8592c2044f5d0d9538da5cc9b8263b93cb60.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/images/fd4759fd3320e42967b9d8afc5cb8592c2044f5d0d9538da5cc9b8263b93cb60.jpg)

### Tables

![383d5b2412da0b1c713984e6bee708129ec1d36b9481bcb00a2c81a06e940ef8.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/383d5b2412da0b1c713984e6bee708129ec1d36b9481bcb00a2c81a06e940ef8.jpg)

![6123826a51d0b7060ec3790c235db4c1d00ceeb206a755c552f7b97d60769fac.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/6123826a51d0b7060ec3790c235db4c1d00ceeb206a755c552f7b97d60769fac.jpg)

![afe5183db7658d9eabbb646c2e16eac023db13d3a5c25f856f820c65eb0cb594.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/afe5183db7658d9eabbb646c2e16eac023db13d3a5c25f856f820c65eb0cb594.jpg)

![d391df97608ef1b0c3f465056000c349a7cb9a3f71d83e5bbda7e72dc31163eb.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/d391df97608ef1b0c3f465056000c349a7cb9a3f71d83e5bbda7e72dc31163eb.jpg)

![df7f32fc05049f427dcb70d1610e9284886188c040d5cd88e5c103209ea264b0.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/df7f32fc05049f427dcb70d1610e9284886188c040d5cd88e5c103209ea264b0.jpg)

![e5df03aca6c8718026dbd31a2b686747ce390d3e5934e9c7449c8a1b770f6d09.jpg](../icml_results/298_ScaleMoE_%20Mixture-of-Experts%20for%20Scalable%20Continuous%20Control%20in%20Actor-Critic%20Reinforcement%20Learning/tables/e5df03aca6c8718026dbd31a2b686747ce390d3e5934e9c7449c8a1b770f6d09.jpg)

<a id="paper-0294"></a>

## MACKO: Sparse matrix-vector multiplication for low sparsity


### Images

![043feaa3eceb7abbd2f1fdcb6fea8e0094139ace3944a40146c6db608b53c19c.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/043feaa3eceb7abbd2f1fdcb6fea8e0094139ace3944a40146c6db608b53c19c.jpg)

![2ceecf6b9553c4230dd1506f5ad5b4cde2bfc867009dc98d36394ec1361a81ca.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/2ceecf6b9553c4230dd1506f5ad5b4cde2bfc867009dc98d36394ec1361a81ca.jpg)

![3e4d72ee643587097674a329e700092ad143cdbf3397591336b3a1ba33792b56.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/3e4d72ee643587097674a329e700092ad143cdbf3397591336b3a1ba33792b56.jpg)

![593ab99de9e7871756ce2b428449604fa3f21a82567a95db1f3dff971f279b09.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/593ab99de9e7871756ce2b428449604fa3f21a82567a95db1f3dff971f279b09.jpg)

![642e33dd3e674fba092f3888355296ad7d4540d0a1db2b1f1bff9a740e22f363.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/642e33dd3e674fba092f3888355296ad7d4540d0a1db2b1f1bff9a740e22f363.jpg)

![9047dee320bb3955b9ce70f7003ea2d13f457df3d4806eb12a93bbc18ebfdf37.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/9047dee320bb3955b9ce70f7003ea2d13f457df3d4806eb12a93bbc18ebfdf37.jpg)

![ac549771584cdc20d807e7dac1665d812e0ff96c0e14e46bb6517aa3368a5a02.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/ac549771584cdc20d807e7dac1665d812e0ff96c0e14e46bb6517aa3368a5a02.jpg)

![d1ad3230d1fcc50dbce665a65584d1142acdbc46ab02d7044f92a475c181bc56.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/d1ad3230d1fcc50dbce665a65584d1142acdbc46ab02d7044f92a475c181bc56.jpg)

![d3a19a7fb0ce2207a094b3084c4468e8fa641c69f627da021365c4b8018d9e7c.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/d3a19a7fb0ce2207a094b3084c4468e8fa641c69f627da021365c4b8018d9e7c.jpg)

![d8dd67448e064a35e756a330ff1131cd96793ea98cd86e7082b0b279fc2c648d.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/images/d8dd67448e064a35e756a330ff1131cd96793ea98cd86e7082b0b279fc2c648d.jpg)

### Tables

![90c3a44a7ae5fa9faffd1e67264296a9315e4559b103052fa47dd3b090c55873.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/tables/90c3a44a7ae5fa9faffd1e67264296a9315e4559b103052fa47dd3b090c55873.jpg)

![e26a1781cf7737aa7625a923afeed70a46cf89a2c6fc5edc982539403e564ac6.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/tables/e26a1781cf7737aa7625a923afeed70a46cf89a2c6fc5edc982539403e564ac6.jpg)

![f613a24c8f94be486ed1352f5b33888402547e4ad7631d5fae188a3812fce872.jpg](../icml_results/299_MACKO_%20Sparse%20matrix-vector%20multiplication%20for%20low%20sparsity/tables/f613a24c8f94be486ed1352f5b33888402547e4ad7631d5fae188a3812fce872.jpg)

<a id="paper-0295"></a>

## SWING: Unlocking Implicit Graph Representations for Graph Random Features


### Images

![12e8c0d0024b13aab253e849e71d2429724d8f1aabb6aa1a8bd93e1e189e87d5.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/12e8c0d0024b13aab253e849e71d2429724d8f1aabb6aa1a8bd93e1e189e87d5.jpg)

![143b0105446739becc7408be26bdae59c6eee3472611e86e1ef8a2fb60afe70f.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/143b0105446739becc7408be26bdae59c6eee3472611e86e1ef8a2fb60afe70f.jpg)

![1ea564dc8e760b7c825ca66c7620a8fe71e0648d7c662fcb7cfe9b66cba6eb2b.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/1ea564dc8e760b7c825ca66c7620a8fe71e0648d7c662fcb7cfe9b66cba6eb2b.jpg)

![22cbbebf00180aaeb8f558d97e6ec1ead084a86c02446ebb63059ab94e272ca5.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/22cbbebf00180aaeb8f558d97e6ec1ead084a86c02446ebb63059ab94e272ca5.jpg)

![2d835d7257f7542ee06805770989ecb52d1fee8075b60ca78f85ae1db95e1847.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/2d835d7257f7542ee06805770989ecb52d1fee8075b60ca78f85ae1db95e1847.jpg)

![5085ca44e4b4c99296c9514576d18eef451b424b0834d329bf72001931207952.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/5085ca44e4b4c99296c9514576d18eef451b424b0834d329bf72001931207952.jpg)

![5cdfffe5e7ba9751b795221ada85dfda5abf8b35170340b86b3ad995f7638ec5.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/5cdfffe5e7ba9751b795221ada85dfda5abf8b35170340b86b3ad995f7638ec5.jpg)

![66f49d24d2824a756ffd7ae5093e40c7f99f46182e15428d84c82862ede9449c.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/66f49d24d2824a756ffd7ae5093e40c7f99f46182e15428d84c82862ede9449c.jpg)

![9d16cbb144fa53e1fa3a6fbb234a990ee90e23de546ccb3d108eee3a4c3033b3.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/9d16cbb144fa53e1fa3a6fbb234a990ee90e23de546ccb3d108eee3a4c3033b3.jpg)

![b43b881a8566bd9dc356135d0e4485ebc01380b6c03368e9e4ebdc4a6eb5a996.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/b43b881a8566bd9dc356135d0e4485ebc01380b6c03368e9e4ebdc4a6eb5a996.jpg)

![f4187959a37c43e20824a3b76f358e6764f3cc99911f75c9e739cf2616738ce7.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/images/f4187959a37c43e20824a3b76f358e6764f3cc99911f75c9e739cf2616738ce7.jpg)

### Tables

![bca6f95efb435a7a5a515ccf7dd2aa6e58a68e725d6b59fc5f2dba80e48db658.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/tables/bca6f95efb435a7a5a515ccf7dd2aa6e58a68e725d6b59fc5f2dba80e48db658.jpg)

![ec98c0367c6f1836148e7db69b6ff1868337c44f27e6667aaa6fd8556fd3ba69.jpg](../icml_results/300_SWING_%20Unlocking%20Implicit%20Graph%20Representations%20for%20Graph%20Random%20Features/tables/ec98c0367c6f1836148e7db69b6ff1868337c44f27e6667aaa6fd8556fd3ba69.jpg)

<a id="paper-0296"></a>

## Quantifying Frontier LLM Capabilities for Container Sandbox Escape


### Images

![1c92eb839213af40f1107a72d74fc2605d8135778d1aed2adfb34951f720c37c.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/1c92eb839213af40f1107a72d74fc2605d8135778d1aed2adfb34951f720c37c.jpg)

![20f528d84fcc080ff78d353b5700b0128c3c66dbe374c27119713da5a4d88934.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/20f528d84fcc080ff78d353b5700b0128c3c66dbe374c27119713da5a4d88934.jpg)

![22efa0a97fffda722655f7374e1cbbe1a3f09ba69eac9bcd7e7a9f7b57c0facb.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/22efa0a97fffda722655f7374e1cbbe1a3f09ba69eac9bcd7e7a9f7b57c0facb.jpg)

![2d3cf8d29b62f98445a5d92254068b1590a184ffab9b96d821adde05b727d4bf.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/2d3cf8d29b62f98445a5d92254068b1590a184ffab9b96d821adde05b727d4bf.jpg)

![46dea370d53358f4ef07949b4ada3685a3fc02d17010f589d31a8e7efdd0ac02.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/46dea370d53358f4ef07949b4ada3685a3fc02d17010f589d31a8e7efdd0ac02.jpg)

![49b4f91d77260281ba1a35a419907bc7e95fc18fd675343e9e9beda75430d2af.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/49b4f91d77260281ba1a35a419907bc7e95fc18fd675343e9e9beda75430d2af.jpg)

![4f87c6dd1f025b1deeea340cc44829778de8768b7757e261d599bbd44c06d42d.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/4f87c6dd1f025b1deeea340cc44829778de8768b7757e261d599bbd44c06d42d.jpg)

![50d149bf83a36d86ab8a0f7156885309bea19a3b62a9b4b5e3368e01a8ce05dd.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/50d149bf83a36d86ab8a0f7156885309bea19a3b62a9b4b5e3368e01a8ce05dd.jpg)

![5b3069af6c65a4353a6d9cc6755ed1aab7601b45708810e3dfab61e887c67c42.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/5b3069af6c65a4353a6d9cc6755ed1aab7601b45708810e3dfab61e887c67c42.jpg)

![69605a043a9e20af3a923d50d4ba7b84afdbf4d8e3bc8d690aba7f316379ac01.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/69605a043a9e20af3a923d50d4ba7b84afdbf4d8e3bc8d690aba7f316379ac01.jpg)

![a977e21a9e27c1d6daeea263f4015f1e7da367024c965334ea62639cbde8c43f.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/a977e21a9e27c1d6daeea263f4015f1e7da367024c965334ea62639cbde8c43f.jpg)

![bc905c8ef941fd1be1d5f369a0c015b24efb33d38cfba27a02a48ce9e74a467c.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/bc905c8ef941fd1be1d5f369a0c015b24efb33d38cfba27a02a48ce9e74a467c.jpg)

![ee36a1a4761751a9a3d4992ec43823e4570131aa79374121d08cf288a2e98ab0.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/images/ee36a1a4761751a9a3d4992ec43823e4570131aa79374121d08cf288a2e98ab0.jpg)

### Tables

![383d7a618c7adfab6e385ba495466ef427e5cb364bd5288ea4c406b8ad0207e9.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/383d7a618c7adfab6e385ba495466ef427e5cb364bd5288ea4c406b8ad0207e9.jpg)

![3cb90272e365e75b3b13e7b8f8f7e5d5ba79be17fa612e239cfa6f883976214d.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/3cb90272e365e75b3b13e7b8f8f7e5d5ba79be17fa612e239cfa6f883976214d.jpg)

![5bebd7a5de7290f40da2b5a3a03c1d03bca661867358e49866ee6744124c48d8.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/5bebd7a5de7290f40da2b5a3a03c1d03bca661867358e49866ee6744124c48d8.jpg)

![6f014853e1f2fcdde564f3a40533e1e1545634aef25952469b121035c107c73d.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/6f014853e1f2fcdde564f3a40533e1e1545634aef25952469b121035c107c73d.jpg)

![7091f4925dfb2ec2fffbc1202cd3e588ff5fe051052cff85182025bec0828dd8.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/7091f4925dfb2ec2fffbc1202cd3e588ff5fe051052cff85182025bec0828dd8.jpg)

![74e25d1b2f9f92397aed3a883c93a2082f726e0800c10686c11224d94607b825.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/74e25d1b2f9f92397aed3a883c93a2082f726e0800c10686c11224d94607b825.jpg)

![926dd441010428b8f29bc9078f2791bcb43c59dd2be4daf1e81676593d9a4e44.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/926dd441010428b8f29bc9078f2791bcb43c59dd2be4daf1e81676593d9a4e44.jpg)

![999c03440b3d9840a50a6f85ab05caecaabfc1e6432d48edb4aa5ebb47410746.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/999c03440b3d9840a50a6f85ab05caecaabfc1e6432d48edb4aa5ebb47410746.jpg)

![ac071957fba5a8f61bf6e62be66c9bd5a0ac961b71b27ab95c620f9bc67380a5.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/ac071957fba5a8f61bf6e62be66c9bd5a0ac961b71b27ab95c620f9bc67380a5.jpg)

![b5a6b770c7ac114e2f99b48e8d0153a6515b1df9deb1dc6e8d4ba2498af33363.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/b5a6b770c7ac114e2f99b48e8d0153a6515b1df9deb1dc6e8d4ba2498af33363.jpg)

![b91750c4c4c1717e22e2d138bcf7a35a03ae2e445ca3830faf65d78b66866671.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/b91750c4c4c1717e22e2d138bcf7a35a03ae2e445ca3830faf65d78b66866671.jpg)

![ba71b868590f7ba36f4d06792a39ec73b32c471849a47b7586d0de41b9141d5f.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/ba71b868590f7ba36f4d06792a39ec73b32c471849a47b7586d0de41b9141d5f.jpg)

![bb5d8054b97786edcb267b1e0ca56cae4d78e3158e5710e7d0fd48d6e6aa57a4.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/bb5d8054b97786edcb267b1e0ca56cae4d78e3158e5710e7d0fd48d6e6aa57a4.jpg)

![cbea15e7658beb09897227ad86c4de41a28778bedc0aef5b043c13fe9fc8dd7a.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/cbea15e7658beb09897227ad86c4de41a28778bedc0aef5b043c13fe9fc8dd7a.jpg)

![da7b2837acbf18d5c5856d8fd61002ca3af7bd3e08365d10c8d613d6aae3b432.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/da7b2837acbf18d5c5856d8fd61002ca3af7bd3e08365d10c8d613d6aae3b432.jpg)

![fbd9c6b9df1799036d755fdc356084c3b1410d805d8bef6de8bcc633524f32fd.jpg](../icml_results/301_Quantifying%20Frontier%20LLM%20Capabilities%20for%20Container%20Sandbox%20Escape/tables/fbd9c6b9df1799036d755fdc356084c3b1410d805d8bef6de8bcc633524f32fd.jpg)

<a id="paper-0297"></a>

## Benchmarking at the Edge of Comprehension

### Images

![004ec7e7a26a1f3b1b6b2d1af87629ba1a7ec7742a9908df29662674c4a5ee05.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/004ec7e7a26a1f3b1b6b2d1af87629ba1a7ec7742a9908df29662674c4a5ee05.jpg)

![18746a0b030f0ad2013b62680cfda987a301fd850b7c05abab239e25785f45b9.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/18746a0b030f0ad2013b62680cfda987a301fd850b7c05abab239e25785f45b9.jpg)

![1951bdf4c48433503c3e5ea5ae9dd4c3dc95eaf6706aa2fc82e729196f06da77.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/1951bdf4c48433503c3e5ea5ae9dd4c3dc95eaf6706aa2fc82e729196f06da77.jpg)

![375acb25c6d03a5bde8b96c4be83a3e18a5eb046a009916ee2f912db72effca2.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/375acb25c6d03a5bde8b96c4be83a3e18a5eb046a009916ee2f912db72effca2.jpg)

![8c7d475cca4aec9b1b8b050a06492a3474fde2f92333d3036adaab016ec231a5.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/8c7d475cca4aec9b1b8b050a06492a3474fde2f92333d3036adaab016ec231a5.jpg)

![b1df34d397f53d13fbc78689743423ed44b890bad9038e552fe9ec4c0e2f27b9.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/b1df34d397f53d13fbc78689743423ed44b890bad9038e552fe9ec4c0e2f27b9.jpg)

![ba61024b33f2bcf0b865c2523b451425d9ef562ee7838f6b3d62826e3c68c05a.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/ba61024b33f2bcf0b865c2523b451425d9ef562ee7838f6b3d62826e3c68c05a.jpg)

![ceb8670055ef281b0864d2e9f0adf97ff44c1a95815d74f7441eccb007d8291b.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/ceb8670055ef281b0864d2e9f0adf97ff44c1a95815d74f7441eccb007d8291b.jpg)

![d0960d60c67f8b0cab0c9365fe12ab26b3e47b465823de45380eddc601c22ceb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/d0960d60c67f8b0cab0c9365fe12ab26b3e47b465823de45380eddc601c22ceb.jpg)

### Tables

![022131925469533eccf8c461cb9e6630612fa44e626d4974b0ffc40bed47096b.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/022131925469533eccf8c461cb9e6630612fa44e626d4974b0ffc40bed47096b.jpg)

![13594ec21552532815121155bb6e987d2f7f963195f0ce70162eb6d40e317afb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/13594ec21552532815121155bb6e987d2f7f963195f0ce70162eb6d40e317afb.jpg)

![3f49be4f3710085fbd5c4a3e3eae0cc6b730458673885eea5e14899f6b9d4c6e.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/3f49be4f3710085fbd5c4a3e3eae0cc6b730458673885eea5e14899f6b9d4c6e.jpg)

![4502ec05ce0d7d50a5f70fd89a2f60e7784557c8756ecf4f62774bff57f030cb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/4502ec05ce0d7d50a5f70fd89a2f60e7784557c8756ecf4f62774bff57f030cb.jpg)

![6394c348edcfbe010707d4ae08f10880d294b49bcacc85bcadd5f8b434ebf540.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/6394c348edcfbe010707d4ae08f10880d294b49bcacc85bcadd5f8b434ebf540.jpg)

![73ab345184383b3f4bfd8150cb57615e81a345f16686bd069d169d8b3d68757d.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/73ab345184383b3f4bfd8150cb57615e81a345f16686bd069d169d8b3d68757d.jpg)

![7520907135681e56e45cedc97ee6b352240acd9288b0cf190e370cfe60e0a6aa.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/7520907135681e56e45cedc97ee6b352240acd9288b0cf190e370cfe60e0a6aa.jpg)

![7a8e224dbfb3b1e37f6f7ca29f17946d4ff3c6841582ce01874a75eafee10bb5.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/7a8e224dbfb3b1e37f6f7ca29f17946d4ff3c6841582ce01874a75eafee10bb5.jpg)

![a274d990744902aa502b278d0274f2fa5b7f77152fff282fb6537928029dd363.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/a274d990744902aa502b278d0274f2fa5b7f77152fff282fb6537928029dd363.jpg)

![bb0245047aedac5a3950ecc5f7cdfc5c6f0888467cd66c389ba7483f6f7dc7c6.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/bb0245047aedac5a3950ecc5f7cdfc5c6f0888467cd66c389ba7483f6f7dc7c6.jpg)

![d05d3e9e88ba687ca6f7997fb1eb97dc48394899c642488a4769368a691bc962.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/d05d3e9e88ba687ca6f7997fb1eb97dc48394899c642488a4769368a691bc962.jpg)

![e1c716c9cf79f4ca9bf07c9e21377557a36d0b9edd51df453154c7ed56a03d1a.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/e1c716c9cf79f4ca9bf07c9e21377557a36d0b9edd51df453154c7ed56a03d1a.jpg)

![f7136db808069c49ff7c2135633ea3dda4ede41b1969a61c295d063329c14da3.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/f7136db808069c49ff7c2135633ea3dda4ede41b1969a61c295d063329c14da3.jpg)
