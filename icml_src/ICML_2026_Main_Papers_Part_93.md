# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 93 of 197

## 目录 (Table of Contents)

1. [Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees](#paper-3037)
2. [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](#paper-3038)
3. [A Task-centric Theory for Iterative Self-Improvement with Easy-to-Hard Curricula](#paper-3039)
4. [Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models](#paper-3040)
5. [Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases](#paper-3041)
6. [Test-Time Training with KV Binding Is Secretly Linear Attention](#paper-3042)
7. [FedCDWA: Decoupled Federated Prototype Distillation with Hierarchical Wasserstein Aggregation](#paper-3043)
8. [TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](#paper-3044)
9. [Intra-Modal Neighbors Never Lie: Rectifying Inter-Modal Noisy Correspondence via Graph-Based Intra-Modal Reasoning](#paper-3045)
10. [BRIDGE: Predicting Human Task Completion Time From Model Performance](#paper-3046)
11. [From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers](#paper-3047)
12. [Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](#paper-3048)
13. [Minimizing Upper Confidence Bounds: A Data-Driven Framework for Stochastic Programming](#paper-3049)
14. [NeurVLA: Unleashing Failure-Handling Capability of Vision-Language-Action Models via Neural-Symbolic Reasoning](#paper-3050)
15. [The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](#paper-3051)
16. [PrivCode++: Latent-Conditioned Differentially Private Code Generation for Comprehensive Guarantees](#paper-3052)
17. [ViEEG: Hierarchical Visual Neural Representation for EEG Brain Decoding](#paper-3053)
18. [LieStoNet: Learning Lie Symmetries from Spatiotemporal Data for Stochastic Dynamical Systems](#paper-3054)
19. [TGV-KV: Text-Grounded KV Eviction for Vision-Language Models](#paper-3055)
20. [GeoSense: Internalizing Geometric Necessity Perception for Multimodal Reasoning](#paper-3056)
21. [$R^3$DAO: Reactive Recovery and Reconstruction for Long-horizon Data Agent Orchestration](#paper-3057)
22. [Unsupervised Camouflaged Object Detection with Dual-Eigenvector Spectral Pseudo-Labeling and Contrastive Refinement](#paper-3058)
23. [Beyond Additive Decompositions: Interpretability Through Separability](#paper-3059)
24. [From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](#paper-3060)
25. [Dual-stage Contrastive Learning-enhanced Multi-view Variational Clustering](#paper-3061)
26. [CoPE: A Framework for Optimizing Coordination between Planning and Execution in LLM-based Agents](#paper-3062)
27. [E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs](#paper-3063)
28. [Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers](#paper-3064)
29. [HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling](#paper-3065)
30. [AIR: Post-training Data Selection for Reasoning via Attention Head Influence](#paper-3066)
31. [DocOS: Towards Proactive Document-Guided Actions in GUI Agents](#paper-3067)
32. [GAE: Unleashing Physical Potential of VLM with Generalizable Action Expert](#paper-3068)
33. [ConvexBench: Can LLMs Recognize Convex Functions?](#paper-3069)

---


<a id="paper-3037"></a>

## Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees

### Images

![45aee35a0d2c7e21edec6e2eee46084f14b6fcdcf11f2e53438cfeebd8abbf1c.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/45aee35a0d2c7e21edec6e2eee46084f14b6fcdcf11f2e53438cfeebd8abbf1c.jpg)

![5687e7e97612b52572ac8d1ebcdc2bf21197d3bb6d52c847bc5ee482d1934333.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/5687e7e97612b52572ac8d1ebcdc2bf21197d3bb6d52c847bc5ee482d1934333.jpg)

![5f48e42345ef89919e91f2d3741e677525bc2a4d098ea086576c68eaa9c7fb52.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/5f48e42345ef89919e91f2d3741e677525bc2a4d098ea086576c68eaa9c7fb52.jpg)

![649787dce535c2419c8896091cdbb152dcafdae87cee4fa1e303bbfe86643fba.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/649787dce535c2419c8896091cdbb152dcafdae87cee4fa1e303bbfe86643fba.jpg)

![aeecd140602c310135c362c1afc81fb80afdb79747166e4f811720b031994631.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/aeecd140602c310135c362c1afc81fb80afdb79747166e4f811720b031994631.jpg)

![b1eadae4ffcd9fc292f97f441baa5e986a01d1a1dc973feb1f34a398624a2849.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/b1eadae4ffcd9fc292f97f441baa5e986a01d1a1dc973feb1f34a398624a2849.jpg)

![b9b9759b9af38f2c5a1a46cfd7af58d2044b57b27a46913d6d5d0887683bf203.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/b9b9759b9af38f2c5a1a46cfd7af58d2044b57b27a46913d6d5d0887683bf203.jpg)

![e6f07b2fa201a7d77e3c9e287c6193a0226dcacdfbc0fec2724a0fd4bb455673.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/e6f07b2fa201a7d77e3c9e287c6193a0226dcacdfbc0fec2724a0fd4bb455673.jpg)

![f458d56973e16fe6dd9c373c3ee736a519b66267948b6508f2cf771c5e433a7b.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/images/f458d56973e16fe6dd9c373c3ee736a519b66267948b6508f2cf771c5e433a7b.jpg)

### Tables

![0a696a66909a5ac68b456e544ac31d88da6d05cd27beb656f740983db3e126ca.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/0a696a66909a5ac68b456e544ac31d88da6d05cd27beb656f740983db3e126ca.jpg)

![162f0de846f4ae6e4a523c68ab6b23afd24ffb15f2c8be3a6328479ba55253a9.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/162f0de846f4ae6e4a523c68ab6b23afd24ffb15f2c8be3a6328479ba55253a9.jpg)

![2a472e5a7e287eceba2aaf4cad5d71fccfe248ae3e0860a59ecb3890df419ab2.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/2a472e5a7e287eceba2aaf4cad5d71fccfe248ae3e0860a59ecb3890df419ab2.jpg)

![3e1d9258e57c241a839b38cf3130f5e395c50b3a3a1ec9484b7e7ebbd9965d7e.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/3e1d9258e57c241a839b38cf3130f5e395c50b3a3a1ec9484b7e7ebbd9965d7e.jpg)

![501edbecbccd2206c27b343e9fe8bfe232ad121fd5f4eed7b8886683e60ea1c9.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/501edbecbccd2206c27b343e9fe8bfe232ad121fd5f4eed7b8886683e60ea1c9.jpg)

![6fd0f9f778ec7dce6a48f523146755eda5ff8883a89c329d0f5c5e15a4776fd0.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/6fd0f9f778ec7dce6a48f523146755eda5ff8883a89c329d0f5c5e15a4776fd0.jpg)

![7489f0594d8579c10cf000dd63f390f2663453c9660338ab5b2288a97438fa27.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/7489f0594d8579c10cf000dd63f390f2663453c9660338ab5b2288a97438fa27.jpg)

![8942f4df519aed1dd74b10eb8c16ee22f9de0e1d8c8bb643a32908097a53aa12.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/8942f4df519aed1dd74b10eb8c16ee22f9de0e1d8c8bb643a32908097a53aa12.jpg)

![a2aa1b8e1943fd86f4e378be9097644e6b7ff04778ad478fe2dd054b567c53d5.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/a2aa1b8e1943fd86f4e378be9097644e6b7ff04778ad478fe2dd054b567c53d5.jpg)

![dff1caa28f29390671b8d22d919ef22ea8a2c5b4e4de4264e9d9d049bcf87976.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/dff1caa28f29390671b8d22d919ef22ea8a2c5b4e4de4264e9d9d049bcf87976.jpg)

![e6e4be5a237dcf6681609f341d57647a59e150605e1ccc181d14e54c01fa230e.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/e6e4be5a237dcf6681609f341d57647a59e150605e1ccc181d14e54c01fa230e.jpg)

![e9a01b0f907fdac21f0d2a505aff4a7b31b9b3ed3a50754cc7a1a199b5e5d741.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/e9a01b0f907fdac21f0d2a505aff4a7b31b9b3ed3a50754cc7a1a199b5e5d741.jpg)

![ed986c82f1853c1d56a29b4771793de727d418c8d5ea08041903b2e78c9b92dc.jpg](../icml_results/3075_Learning%20Graph%20Foundation%20Models%20on%20Riemannian%20Graph-of-Graphs/tables/ed986c82f1853c1d56a29b4771793de727d418c8d5ea08041903b2e78c9b92dc.jpg)

## Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees


### Images

![2fdbb303d523c9c7f5e53552dff0d32acb10747a8fccb498d3b1f981400164fe.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/images/2fdbb303d523c9c7f5e53552dff0d32acb10747a8fccb498d3b1f981400164fe.jpg)

![6eebdb1fde8d190b25f1132545e93be87cf375820b32a5ddcfa8a02b2dfa09e3.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/images/6eebdb1fde8d190b25f1132545e93be87cf375820b32a5ddcfa8a02b2dfa09e3.jpg)

![96b9af2db1cd5dbfab98ca783d9bc7026da9fddcb596d9f21cbbff827c2ba432.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/images/96b9af2db1cd5dbfab98ca783d9bc7026da9fddcb596d9f21cbbff827c2ba432.jpg)

![b3a20fbd078b63607931311468ec67c61ec9cb90911c69e660dd372f20178dd3.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/images/b3a20fbd078b63607931311468ec67c61ec9cb90911c69e660dd372f20178dd3.jpg)

![f0902001c89f6fa5659a4ac0e86482f5e842af978341685be0a3243f71c059da.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/images/f0902001c89f6fa5659a4ac0e86482f5e842af978341685be0a3243f71c059da.jpg)

### Tables

![11bb7c7b00dac630131af45c2cd712bb33918852774782792a3c546ac3b3acaa.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/11bb7c7b00dac630131af45c2cd712bb33918852774782792a3c546ac3b3acaa.jpg)

![13142d9acc94c645b1b2281961fabe8549a578691a3cb4c1ca7856912618f71f.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/13142d9acc94c645b1b2281961fabe8549a578691a3cb4c1ca7856912618f71f.jpg)

![13f36365a199f457099b213bb3a99f7cd16d0fadcdd04071ac43d1f4f0b36129.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/13f36365a199f457099b213bb3a99f7cd16d0fadcdd04071ac43d1f4f0b36129.jpg)

![1e48f38290bbdf8cb3cc8c9e5a322a301eedbe35ed3d8f8e69a63007a4311270.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/1e48f38290bbdf8cb3cc8c9e5a322a301eedbe35ed3d8f8e69a63007a4311270.jpg)

![574a47194afd579d452733be10a5dba16e63748ef9f14e8114bea3680b078c6d.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/574a47194afd579d452733be10a5dba16e63748ef9f14e8114bea3680b078c6d.jpg)

![64b265a5b917a7929fb4d13700c5f58f820c480c77e21708b0648b9fffb3676c.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/64b265a5b917a7929fb4d13700c5f58f820c480c77e21708b0648b9fffb3676c.jpg)

![83e510ddd9a7d125cf53bdd0dad7cb8d482129acaf89a1dbdf45ad8a1434ea84.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/83e510ddd9a7d125cf53bdd0dad7cb8d482129acaf89a1dbdf45ad8a1434ea84.jpg)

![8b86bb6f96381c8eac33fed21820ccabeaf04c8d4ce1b310d3629ba3dadc73c8.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/8b86bb6f96381c8eac33fed21820ccabeaf04c8d4ce1b310d3629ba3dadc73c8.jpg)

![a4373d7a0426d8a3e7a1dfdeef7c3658c7be1226361d3ed150d7c4a1cc577227.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/a4373d7a0426d8a3e7a1dfdeef7c3658c7be1226361d3ed150d7c4a1cc577227.jpg)

![a6b3643d42ba54143d999344902afee5ab0e4872c75a66907f6aba4e24aa0446.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/a6b3643d42ba54143d999344902afee5ab0e4872c75a66907f6aba4e24aa0446.jpg)

![aa1ca014647c90a7cc77370156117a0edcd772f85ad19df5d402e74758390a6c.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/aa1ca014647c90a7cc77370156117a0edcd772f85ad19df5d402e74758390a6c.jpg)

![b161e73115f86c4804fc65407e566049f2c936c2bf07f21ffe175ba473b78b34.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/b161e73115f86c4804fc65407e566049f2c936c2bf07f21ffe175ba473b78b34.jpg)

![c0d2ea271d2e1fe9cf00d42a95543e0e8e06e36a97efb6f0efd3c34e0cddfc84.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/c0d2ea271d2e1fe9cf00d42a95543e0e8e06e36a97efb6f0efd3c34e0cddfc84.jpg)

![fbdce747bf428a8b7be607ca9524f84840a386279e77acc61a1b79a7cf20a7a5.jpg](../icml_results/3076_Decompose%2C%20Structure%2C%20and%20Repair_%20A%20Neuro-Symbolic%20Framework%20for%20Autoformalization%20via%20Operator%20Tree/tables/fbdce747bf428a8b7be607ca9524f84840a386279e77acc61a1b79a7cf20a7a5.jpg)

<a id="paper-3038"></a>

## GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization


### Images

![00f55692ed144636b4e3d108870623c9b8f6b619509cef054200a7e4bff55bf8.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/00f55692ed144636b4e3d108870623c9b8f6b619509cef054200a7e4bff55bf8.jpg)

![2fab89d3e93291ec264e70fac607e7ca07c1920ac28521d8b3a065449d8fdb8f.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/2fab89d3e93291ec264e70fac607e7ca07c1920ac28521d8b3a065449d8fdb8f.jpg)

![430dd1c2af8597917bd6b326ce20d09269786e3b30ac56b6da6e37a286c9d517.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/430dd1c2af8597917bd6b326ce20d09269786e3b30ac56b6da6e37a286c9d517.jpg)

![55a07b92694be087fa4696f3c45017369f6ef4eeece7e1a412bf1d756dd82c8e.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/55a07b92694be087fa4696f3c45017369f6ef4eeece7e1a412bf1d756dd82c8e.jpg)

![70a40541e3ccf9feec206dff3ad0405063df266455e6324fc2cfb933db26b99c.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/70a40541e3ccf9feec206dff3ad0405063df266455e6324fc2cfb933db26b99c.jpg)

![7775c1f066305d2e50d4bcd6ebb820e98e7306b5b19739b068128e6f8c895fc8.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/7775c1f066305d2e50d4bcd6ebb820e98e7306b5b19739b068128e6f8c895fc8.jpg)

![789582ccae753649f80f159dec0fe3d13630f5a675b21de03922f6e20cbf25ce.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/789582ccae753649f80f159dec0fe3d13630f5a675b21de03922f6e20cbf25ce.jpg)

![9be6b7de40834cb1f0e6408abb0cd8c3bb72a6957a649e777d022f6eac1ce068.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/9be6b7de40834cb1f0e6408abb0cd8c3bb72a6957a649e777d022f6eac1ce068.jpg)

![c506108d66d84cb149d1ecf5324dd18720b282da909ba2c3893904d0bf5fbe0b.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/c506108d66d84cb149d1ecf5324dd18720b282da909ba2c3893904d0bf5fbe0b.jpg)

![c75a05fa73a8883602a153ff0970c57380ca36deed7bb6957ebbdb0a1d804233.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/c75a05fa73a8883602a153ff0970c57380ca36deed7bb6957ebbdb0a1d804233.jpg)

![c952583777bf137471ef6a88168af0ce9a5b119353ba17719becc0f3c6a27659.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/c952583777bf137471ef6a88168af0ce9a5b119353ba17719becc0f3c6a27659.jpg)

![d1af774f0f6947a2ac4ea0111150240ec4af4ea471244b7cae58850679a64765.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/images/d1af774f0f6947a2ac4ea0111150240ec4af4ea471244b7cae58850679a64765.jpg)

### Tables

![0abe56447ecd7c4eb1d10df3461c037c3d632973683b2fd0c35e4138c74c797a.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/0abe56447ecd7c4eb1d10df3461c037c3d632973683b2fd0c35e4138c74c797a.jpg)

![0f1e857243677cde5f0a2a84590bc36c7483ba81e8e75b1c980ce2730f1866bd.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/0f1e857243677cde5f0a2a84590bc36c7483ba81e8e75b1c980ce2730f1866bd.jpg)

![30a854074e88220fc5c80d63a442575f0fdc7c1f6032c94074d0f4e6acb77160.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/30a854074e88220fc5c80d63a442575f0fdc7c1f6032c94074d0f4e6acb77160.jpg)

![695fd4ee5b62cbc901a171c31230268cb918f0e288f4352e6b6292ff3205393b.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/695fd4ee5b62cbc901a171c31230268cb918f0e288f4352e6b6292ff3205393b.jpg)

![80c11081b6054a420b3981372a7870ac88d6baf98bd1f179125c8af3c36d969d.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/80c11081b6054a420b3981372a7870ac88d6baf98bd1f179125c8af3c36d969d.jpg)

![8e346813a276fcaaecc978c1b8f8e25e4a10ee74d9e60e9788e4a0a358064d1b.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/8e346813a276fcaaecc978c1b8f8e25e4a10ee74d9e60e9788e4a0a358064d1b.jpg)

![941c0c3078437574d570d7b95893d3e0a8a55f5991dd03a2f9e64f1ff56ec9e6.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/941c0c3078437574d570d7b95893d3e0a8a55f5991dd03a2f9e64f1ff56ec9e6.jpg)

![b0ed78a9400b18e06b584df3ef147a88e4c284414258c9fd65f9b1ef5e8e0cbd.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/b0ed78a9400b18e06b584df3ef147a88e4c284414258c9fd65f9b1ef5e8e0cbd.jpg)

![daa30ba56e579129c11ee772fda909379fdbf4389c81d90056cec8385018599d.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/daa30ba56e579129c11ee772fda909379fdbf4389c81d90056cec8385018599d.jpg)

![de51a5055b77ad120a9515fcdc37e253a6e0772bca34b15d4d5ca3629126f8ed.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/de51a5055b77ad120a9515fcdc37e253a6e0772bca34b15d4d5ca3629126f8ed.jpg)

![e5eb4cfd9f8ed9507cf86377edb215501e3d67dcaa33e73c5da51b0c5d2795c1.jpg](../icml_results/3077_GDPO_%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization/tables/e5eb4cfd9f8ed9507cf86377edb215501e3d67dcaa33e73c5da51b0c5d2795c1.jpg)

<a id="paper-3039"></a>

## A Task-centric Theory for Iterative Self-Improvement with Easy-to-Hard Curricula


### Images

![1a8005e87eb455986ee1a32f97f7c55c2f133de501520d171093fc35eb15080e.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/1a8005e87eb455986ee1a32f97f7c55c2f133de501520d171093fc35eb15080e.jpg)

![234ba7019f235f8a36acca93a11c725071b1770cdf67a4c9428c6c7374bd5136.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/234ba7019f235f8a36acca93a11c725071b1770cdf67a4c9428c6c7374bd5136.jpg)

![2aa55d6d09b3465a7dbaec73a9667f423a60946b4472b7a5557b72b2a685faa1.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/2aa55d6d09b3465a7dbaec73a9667f423a60946b4472b7a5557b72b2a685faa1.jpg)

![4fcc832e0b23f40934cff27228444902c50e6febfde98fc059f3be184e683d7c.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/4fcc832e0b23f40934cff27228444902c50e6febfde98fc059f3be184e683d7c.jpg)

![b81c6c2584c7369b08c6c81936239d9b0d33e3de99cf9e31215166b78140e0c0.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/b81c6c2584c7369b08c6c81936239d9b0d33e3de99cf9e31215166b78140e0c0.jpg)

![d51b7b2b4984f9246eb33ec1c9a607c9aeefccd37f6ecec76df77ff14f2a35d7.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/images/d51b7b2b4984f9246eb33ec1c9a607c9aeefccd37f6ecec76df77ff14f2a35d7.jpg)

### Tables

![0e044bf79f6a8296e55b3b2f7572848f142e3fda47fc4e7a80f4c0b32d8c2f86.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/0e044bf79f6a8296e55b3b2f7572848f142e3fda47fc4e7a80f4c0b32d8c2f86.jpg)

![17806a25b87e6b71a5035844983eee75b037902f4cd63fb9afbf9f3a18c7fe54.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/17806a25b87e6b71a5035844983eee75b037902f4cd63fb9afbf9f3a18c7fe54.jpg)

![6b74ba97084dcc4720b57741d8ceace99e210c791a999a46f097543fa5f535c2.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/6b74ba97084dcc4720b57741d8ceace99e210c791a999a46f097543fa5f535c2.jpg)

![72079b86a91feb694d8682913991fa409e62357941fc52a6da5aec342281296d.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/72079b86a91feb694d8682913991fa409e62357941fc52a6da5aec342281296d.jpg)

![73bd26a6d16a342c5434166c0d3694a55b54503cc2738b88b1bc2ae5ffb75532.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/73bd26a6d16a342c5434166c0d3694a55b54503cc2738b88b1bc2ae5ffb75532.jpg)

![82a3714f66a84973c07a438d56cf1306daac03b337a77c9358deae501af58df6.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/82a3714f66a84973c07a438d56cf1306daac03b337a77c9358deae501af58df6.jpg)

![895eeab3e3fdb0e09ff7b6892828115019cb2017791decd62b6f55265bd61a08.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/895eeab3e3fdb0e09ff7b6892828115019cb2017791decd62b6f55265bd61a08.jpg)

![a0ec712cd91f26bd6c22897190a3e4c5b547c32e26c85bfc1119a71e151b992d.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/a0ec712cd91f26bd6c22897190a3e4c5b547c32e26c85bfc1119a71e151b992d.jpg)

![b00891128e80c1923c052a678a3755026178e1a21437bbbac05d8671ac18b1fc.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/b00891128e80c1923c052a678a3755026178e1a21437bbbac05d8671ac18b1fc.jpg)

![c0b9548995a7a90381bc5355906ac7cfe8cbc690036dda253888ca8b0314a9de.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/c0b9548995a7a90381bc5355906ac7cfe8cbc690036dda253888ca8b0314a9de.jpg)

![cbd8c33612cae4d1c80ffcc89131678f25f550e5c41b5d9495617069511e1e07.jpg](../icml_results/3078_A%20Task-centric%20Theory%20for%20Iterative%20Self-Improvement%20with%20Easy-to-Hard%20Curricula/tables/cbd8c33612cae4d1c80ffcc89131678f25f550e5c41b5d9495617069511e1e07.jpg)

<a id="paper-3040"></a>

## Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models


### Images

![237e8b90be84afea9254f809231cd9c87dbcc2328146b214004dcde419f97b96.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/237e8b90be84afea9254f809231cd9c87dbcc2328146b214004dcde419f97b96.jpg)

![28f2857afd1a33e1bfd9d5faea01e913b723deffa739088d949f515069228ad5.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/28f2857afd1a33e1bfd9d5faea01e913b723deffa739088d949f515069228ad5.jpg)

![29d976b4b12ea426abc54a9f0055393a733b491276cabc86fd79cd2bea4070d9.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/29d976b4b12ea426abc54a9f0055393a733b491276cabc86fd79cd2bea4070d9.jpg)

![2b7596813c47db6a4ffddc0cafef2045e2264910fa7ec333b0d6687ffa2c3925.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/2b7596813c47db6a4ffddc0cafef2045e2264910fa7ec333b0d6687ffa2c3925.jpg)

![337f9d88294c15cc526c666f5749117b6cfa19548a8ab266d4f6d74459c431cc.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/337f9d88294c15cc526c666f5749117b6cfa19548a8ab266d4f6d74459c431cc.jpg)

![345250e617a16feb0e814c9e6e51634dad267bfd283b6eb72c2b6f7263ec9a73.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/345250e617a16feb0e814c9e6e51634dad267bfd283b6eb72c2b6f7263ec9a73.jpg)

![3e4f26af34b8d1434347867e7cbf7561db4dcf346b357e943986636953f00a09.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/3e4f26af34b8d1434347867e7cbf7561db4dcf346b357e943986636953f00a09.jpg)

![532a55e3caa0080e8bf6b6e8b36ef4b9a16984b9d1c5a2774decd68d73306517.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/532a55e3caa0080e8bf6b6e8b36ef4b9a16984b9d1c5a2774decd68d73306517.jpg)

![6ef9f7dd9798ee02772bc4f82517623043ffec345814a2349d1583bc85c6dbed.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/6ef9f7dd9798ee02772bc4f82517623043ffec345814a2349d1583bc85c6dbed.jpg)

![b9d17d36290dd0294f12fd8a175d6a01a26ef48b385557546923b1d8d288375e.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/images/b9d17d36290dd0294f12fd8a175d6a01a26ef48b385557546923b1d8d288375e.jpg)

### Tables

![114a39cb7f46b0722e9d388f8eb74e020a3ec245c5137eba86e1f0d8e3c6cab1.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/114a39cb7f46b0722e9d388f8eb74e020a3ec245c5137eba86e1f0d8e3c6cab1.jpg)

![16002b261e55aca1556bcf95fa9e1c73d0005302b1694de4ccba8f968455815b.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/16002b261e55aca1556bcf95fa9e1c73d0005302b1694de4ccba8f968455815b.jpg)

![21b6e1300cf1bb0fb3abfe5b25f89ddf224e95f1e125b82a710121dba2610c06.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/21b6e1300cf1bb0fb3abfe5b25f89ddf224e95f1e125b82a710121dba2610c06.jpg)

![297807cdfae78fe6e68235ff62e1d9f64c586346b586023a3b4539dd9abc965f.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/297807cdfae78fe6e68235ff62e1d9f64c586346b586023a3b4539dd9abc965f.jpg)

![3ee608e926ddd82ac934a7acf3bc8d056644df395a6436f1a1f675050878204b.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/3ee608e926ddd82ac934a7acf3bc8d056644df395a6436f1a1f675050878204b.jpg)

![5271dd1a25d8628f07c0bb2eb901913124c2553984e094fa450eaf4fa3d1eb71.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/5271dd1a25d8628f07c0bb2eb901913124c2553984e094fa450eaf4fa3d1eb71.jpg)

![6ccb5cc8930b0e064edf4649ead497aa745d04b6c38ecf8a0e7065b0ba5fc75c.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/6ccb5cc8930b0e064edf4649ead497aa745d04b6c38ecf8a0e7065b0ba5fc75c.jpg)

![6dc2201dd9f731777b60c91ca6a0ecf156471e742b47f6c42ed5d31378f32d1e.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/6dc2201dd9f731777b60c91ca6a0ecf156471e742b47f6c42ed5d31378f32d1e.jpg)

![81449f90f811a6b6b45cdeed342708274d010ccc5d28300cf4de5ffff796d99c.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/81449f90f811a6b6b45cdeed342708274d010ccc5d28300cf4de5ffff796d99c.jpg)

![93ec448f83bdb521ce1d43cfd79a81325998c616461fb81ae0d86b8c88050db1.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/93ec448f83bdb521ce1d43cfd79a81325998c616461fb81ae0d86b8c88050db1.jpg)

![970bcd4e0124014e7e906d4deda131f10bec299d849f9e722243733968026b52.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/970bcd4e0124014e7e906d4deda131f10bec299d849f9e722243733968026b52.jpg)

![a3c965d94b2b7eb35218f9518e2fccde0c91380fa2ca17eff028b4d3ef57c8de.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/a3c965d94b2b7eb35218f9518e2fccde0c91380fa2ca17eff028b4d3ef57c8de.jpg)

![c72df29d52ee358cdf65dcc95a4185150ed3caeae3be9e72975ba66735ab83f1.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/c72df29d52ee358cdf65dcc95a4185150ed3caeae3be9e72975ba66735ab83f1.jpg)

![c94f361238d4582e8d91cdefecdb07ec776a0038705eef12df311bafa877961d.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/c94f361238d4582e8d91cdefecdb07ec776a0038705eef12df311bafa877961d.jpg)

![cc917dec5e0f37407555cd0c6ac4d436472e9cbe10874eb4a1fa84f3bbde6f18.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/cc917dec5e0f37407555cd0c6ac4d436472e9cbe10874eb4a1fa84f3bbde6f18.jpg)

![db29375052d1a690a6e3fa35101c9e832ca40d03a314793351aea40317a9cecb.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/db29375052d1a690a6e3fa35101c9e832ca40d03a314793351aea40317a9cecb.jpg)

![ebd9922a7e91ecd1ee4929754612bbdd5abee6c8185684af8bbbf35fc409a656.jpg](../icml_results/3079_Self-Calibrated%20Consistency%20can%20Fight%20Back%20for%20Adversarial%20Robustness%20in%20Vision-Language%20Models/tables/ebd9922a7e91ecd1ee4929754612bbdd5abee6c8185684af8bbbf35fc409a656.jpg)

<a id="paper-3041"></a>

## Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases


### Images

![0440e045b59777b2f789d9e746af5b37f8a36d6fd7ee44ee168368b6eee06363.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/0440e045b59777b2f789d9e746af5b37f8a36d6fd7ee44ee168368b6eee06363.jpg)

![14e6a434bc11837faeb2f01dd1694857349af788f4a043b16fb3117bfdbf1ce6.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/14e6a434bc11837faeb2f01dd1694857349af788f4a043b16fb3117bfdbf1ce6.jpg)

![1c22a9dc1cb8425a31f5bb2a7e91bed0e01c5c055da5e430a9ef54c0186c73f5.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/1c22a9dc1cb8425a31f5bb2a7e91bed0e01c5c055da5e430a9ef54c0186c73f5.jpg)

![2118d2e7f0008a6624a15c918457e10e7ac7316df2a9656860059ad902f46bd3.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/2118d2e7f0008a6624a15c918457e10e7ac7316df2a9656860059ad902f46bd3.jpg)

![2585dfce4121e5064334d598c0961fbd31c98c39463c968128dcbf5bc95a1512.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/2585dfce4121e5064334d598c0961fbd31c98c39463c968128dcbf5bc95a1512.jpg)

![2e6e0b43a59aedf1a81a4dc4a9f8d646c54e269667544186405ec62da6824324.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/2e6e0b43a59aedf1a81a4dc4a9f8d646c54e269667544186405ec62da6824324.jpg)

![465f10ea6bb55de3a00b36caeec07066e0e4e30e5d62a496604fee6f07f3cec8.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/465f10ea6bb55de3a00b36caeec07066e0e4e30e5d62a496604fee6f07f3cec8.jpg)

![4d2cd08283d9a91cec76c170ee2e8b68dc8c0f77106888a9ba14d718fdad7d35.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/4d2cd08283d9a91cec76c170ee2e8b68dc8c0f77106888a9ba14d718fdad7d35.jpg)

![4eda2614317955e95c2f82211b8a31f0536782a3b55e0e876feb513e692077f7.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/4eda2614317955e95c2f82211b8a31f0536782a3b55e0e876feb513e692077f7.jpg)

![4ee85f2d4224fd68fd47fdf7dcbcff42b018eab8ad806686f33765a29c9586a1.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/4ee85f2d4224fd68fd47fdf7dcbcff42b018eab8ad806686f33765a29c9586a1.jpg)

![5021a4aa25a520451439dee4a091b0b6c8fef4218d1aec09dfeb95ac476b058d.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/5021a4aa25a520451439dee4a091b0b6c8fef4218d1aec09dfeb95ac476b058d.jpg)

![56ca4db2c08c7ea0d3579d21a7c76e6a137211ecc908dc13dc86cdfcdab06375.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/56ca4db2c08c7ea0d3579d21a7c76e6a137211ecc908dc13dc86cdfcdab06375.jpg)

![5c022a7423556b53816d1b120486912878f93beebe4cb40730b06090984474c7.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/5c022a7423556b53816d1b120486912878f93beebe4cb40730b06090984474c7.jpg)

![6661cc469719cc55ac4977caf033e0b0fa649f4822ac74491461376c7e5ae5aa.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/6661cc469719cc55ac4977caf033e0b0fa649f4822ac74491461376c7e5ae5aa.jpg)

![6a860dbc13d101446568b72d856a37417547984ea1cedac2291d0ea61ceead1e.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/6a860dbc13d101446568b72d856a37417547984ea1cedac2291d0ea61ceead1e.jpg)

![76ddc8e7d5040157fb67acc200c07f701a3872ac7d4b956a677fd00ce2ea7753.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/76ddc8e7d5040157fb67acc200c07f701a3872ac7d4b956a677fd00ce2ea7753.jpg)

![7da7ebe1dbd5b2bb4786cc4ba5d31d7e0d643d9a5c9309d83bb4a2f60bbdc664.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/7da7ebe1dbd5b2bb4786cc4ba5d31d7e0d643d9a5c9309d83bb4a2f60bbdc664.jpg)

![8f51cf9728397083e52a482b16a3214a13a5771fb46e1ef26735ec0b5887638a.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/8f51cf9728397083e52a482b16a3214a13a5771fb46e1ef26735ec0b5887638a.jpg)

![963943f6c6887b8c737dfb2b2766b215a44f75d80c4d0e5d4ea7f1aeb0429e7f.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/963943f6c6887b8c737dfb2b2766b215a44f75d80c4d0e5d4ea7f1aeb0429e7f.jpg)

![a26757095db499ee129cb192c6dfee710049dbc56d554349e158bd9486c67e1f.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/a26757095db499ee129cb192c6dfee710049dbc56d554349e158bd9486c67e1f.jpg)

![a3a815a0b8697e7490e0e1d1ceb55f08c1b21ed54ccf96abdd4de1855cc7f18f.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/a3a815a0b8697e7490e0e1d1ceb55f08c1b21ed54ccf96abdd4de1855cc7f18f.jpg)

![bb8dfb32a9a32fb05b668c04e92c41ecc70701114685e2421925e2a5123a2ad4.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/bb8dfb32a9a32fb05b668c04e92c41ecc70701114685e2421925e2a5123a2ad4.jpg)

![c489286f2dd1b8507db065fa52c5e274b1f2530cf77d5556d0f408cf0491574b.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/c489286f2dd1b8507db065fa52c5e274b1f2530cf77d5556d0f408cf0491574b.jpg)

![ca1384f67391c438c74ff1ef5a7402826c22b6b2c90d53cd5fbafe70643a9189.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/ca1384f67391c438c74ff1ef5a7402826c22b6b2c90d53cd5fbafe70643a9189.jpg)

![cadaa94b51a5223d2b2816837309cf0c59f74ed34721ef67e129f7a2589e1f1b.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/cadaa94b51a5223d2b2816837309cf0c59f74ed34721ef67e129f7a2589e1f1b.jpg)

![e1cdf59890307eeb16c4821a25dc635198e1e162be7c2c90ca0d77089c7e7383.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/e1cdf59890307eeb16c4821a25dc635198e1e162be7c2c90ca0d77089c7e7383.jpg)

![eb38df0bc229a387a281aedf86ae459167c85260dad65f41f61e85f56a00c21b.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/eb38df0bc229a387a281aedf86ae459167c85260dad65f41f61e85f56a00c21b.jpg)

![f01a6099ce9d09ca09ad281f139875021b5f8732e99bd94bdc519b47d3db8d63.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/f01a6099ce9d09ca09ad281f139875021b5f8732e99bd94bdc519b47d3db8d63.jpg)

![f115218a0b79175ba920f2782b25cfe5d62142aa259b433185fe3215a6fae329.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/f115218a0b79175ba920f2782b25cfe5d62142aa259b433185fe3215a6fae329.jpg)

![f47c364d789aa12f866afdaab1df54bfb8a20bedd2307cea26da458c10c061fb.jpg](../icml_results/3080_Less%20Data%2C%20Faster%20Training_%20repeating%20smaller%20datasets%20speeds%20up%20learning%20via%20sampling%20biases/images/f47c364d789aa12f866afdaab1df54bfb8a20bedd2307cea26da458c10c061fb.jpg)

<a id="paper-3042"></a>

## Test-Time Training with KV Binding Is Secretly Linear Attention


### Images

![9cb0f214f53b51847039d5b69d99820e71b9f7169aeb16094d6bfe5b6dbf62fa.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/images/9cb0f214f53b51847039d5b69d99820e71b9f7169aeb16094d6bfe5b6dbf62fa.jpg)

![a7b0741281b0130fc49cb26e873a492e7fc2daaf493ea572facc925068efea3a.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/images/a7b0741281b0130fc49cb26e873a492e7fc2daaf493ea572facc925068efea3a.jpg)

![b857f58402ed4d3e43ae5f27c3e3061fa6ab22ec8032c4d1e82d061a03e05b9e.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/images/b857f58402ed4d3e43ae5f27c3e3061fa6ab22ec8032c4d1e82d061a03e05b9e.jpg)

![cdc79698dd16beefc2a69810d3af4379572dd6d8623d8b7ab236ca62304794ee.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/images/cdc79698dd16beefc2a69810d3af4379572dd6d8623d8b7ab236ca62304794ee.jpg)

### Tables

![5214e9a92dec9e06d2b5f4eb91015169e0a0589d3b3d2bcb1cc9fae13bec62d0.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/tables/5214e9a92dec9e06d2b5f4eb91015169e0a0589d3b3d2bcb1cc9fae13bec62d0.jpg)

![667ccb8ccb64becfcf8827ef690a8114f24dfbcebbb542d84528c7f70c76e224.jpg](../icml_results/3081_Test-Time%20Training%20with%20KV%20Binding%20Is%20Secretly%20Linear%20Attention/tables/667ccb8ccb64becfcf8827ef690a8114f24dfbcebbb542d84528c7f70c76e224.jpg)

<a id="paper-3043"></a>

## FedCDWA: Decoupled Federated Prototype Distillation with Hierarchical Wasserstein Aggregation


### Images

![ad473d7d2b1b469056f329eb63dca795d473f734cb8baea2f63027097b71967a.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/images/ad473d7d2b1b469056f329eb63dca795d473f734cb8baea2f63027097b71967a.jpg)

![b1fc6a97a05a5122d90fed1f2d4aa15ffc2c06479ed2aa8a5c38a8950a5c6476.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/images/b1fc6a97a05a5122d90fed1f2d4aa15ffc2c06479ed2aa8a5c38a8950a5c6476.jpg)

### Tables

![14ab49527271440dee4265fc9a4022d850ebb5a2cc6bacc3e81c3d2f861dbd09.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/14ab49527271440dee4265fc9a4022d850ebb5a2cc6bacc3e81c3d2f861dbd09.jpg)

![153ef9761b9aa3967e42bdc6c508cb1ce0b47d7bfd578ab308e162655ae4676b.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/153ef9761b9aa3967e42bdc6c508cb1ce0b47d7bfd578ab308e162655ae4676b.jpg)

![50327e3b787fc90b9e469c01630769e2deafe80edab3117c2f12f2ec8d2b7cb8.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/50327e3b787fc90b9e469c01630769e2deafe80edab3117c2f12f2ec8d2b7cb8.jpg)

![5752b85c15e455dae0e483bc045c9a71072ea7631eee8612551436b45a6264a1.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/5752b85c15e455dae0e483bc045c9a71072ea7631eee8612551436b45a6264a1.jpg)

![6210098809b2a5a8fdf1d24d4073514c1310401347f8a3343078bfbc3a2ddfc6.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/6210098809b2a5a8fdf1d24d4073514c1310401347f8a3343078bfbc3a2ddfc6.jpg)

![667a81658c6aad94bc971128f090b7cd2a773afcce3add0a9f0e19b56c84b426.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/667a81658c6aad94bc971128f090b7cd2a773afcce3add0a9f0e19b56c84b426.jpg)

![71b6b50b4cbd2e8d775e282fe1a6e2a0aeee5c42ba34f2766e9d69e898bb632f.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/71b6b50b4cbd2e8d775e282fe1a6e2a0aeee5c42ba34f2766e9d69e898bb632f.jpg)

![92bbc117edbc4929f4cbd091503134b63f4e8b4774190ed82c5bba13f6254226.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/92bbc117edbc4929f4cbd091503134b63f4e8b4774190ed82c5bba13f6254226.jpg)

![aaa69b843598f07a7a36e571b569f2bc72fe9850a8bcfdc2fb84c3c5c0d91bd1.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/aaa69b843598f07a7a36e571b569f2bc72fe9850a8bcfdc2fb84c3c5c0d91bd1.jpg)

![ca784c1daf87c5497587e7696260217ffb2db32436bac68a0fcb5b382b8b70b7.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/ca784c1daf87c5497587e7696260217ffb2db32436bac68a0fcb5b382b8b70b7.jpg)

![d03c6d538322c68eab3b697be04f7828bdb4a668221fc6b285a80d270ba70b3d.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/d03c6d538322c68eab3b697be04f7828bdb4a668221fc6b285a80d270ba70b3d.jpg)

![d043ba196659a54a9ab0cc0fe505d7bb32a1ae526711103b759547d7a4b5b693.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/d043ba196659a54a9ab0cc0fe505d7bb32a1ae526711103b759547d7a4b5b693.jpg)

![d896554730d19c8a7cd2766cd509b1490985b6b43ad3f08cadaa8e8de900d2b0.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/d896554730d19c8a7cd2766cd509b1490985b6b43ad3f08cadaa8e8de900d2b0.jpg)

![e58d415f3226be4e6b0b1978f956c46cfc33984edd481a1312e1eb5ecf138de3.jpg](../icml_results/3082_FedCDWA_%20Decoupled%20Federated%20Prototype%20Distillation%20with%20Hierarchical%20Wasserstein%20Aggregation/tables/e58d415f3226be4e6b0b1978f956c46cfc33984edd481a1312e1eb5ecf138de3.jpg)

<a id="paper-3044"></a>

## TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning


### Images

![06dd0fd3079423219fda30a30b97b2f8d6d786303e568697b6e7412943a0a5a5.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/06dd0fd3079423219fda30a30b97b2f8d6d786303e568697b6e7412943a0a5a5.jpg)

![438c6f790a1948158a5c7342f47acc23870d485bdea24bb47b816cc6d3e11afa.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/438c6f790a1948158a5c7342f47acc23870d485bdea24bb47b816cc6d3e11afa.jpg)

![53a3b7b4886bb15192682030839aecf4d2c2f9d0b4795b77c4eb83cff2381212.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/53a3b7b4886bb15192682030839aecf4d2c2f9d0b4795b77c4eb83cff2381212.jpg)

![54f1edf39ef4d48a0bdc30fbf6b706b6bdbba6e43a14ab5d1361a7a5a78cc7f5.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/54f1edf39ef4d48a0bdc30fbf6b706b6bdbba6e43a14ab5d1361a7a5a78cc7f5.jpg)

![66969c12b302c560ca8a22378f748bc25e14b5b0127f59f4dd4c014d2adb2161.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/66969c12b302c560ca8a22378f748bc25e14b5b0127f59f4dd4c014d2adb2161.jpg)

![926f735674ff2e98a4c4c85d91bae15fa054ab1cb8fec6929dabf75f4a3f92aa.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/926f735674ff2e98a4c4c85d91bae15fa054ab1cb8fec6929dabf75f4a3f92aa.jpg)

![a12c823a126a229097da1204e9db35804d75a5b11c8f5b7e0b1cf3562f9bce98.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/a12c823a126a229097da1204e9db35804d75a5b11c8f5b7e0b1cf3562f9bce98.jpg)

![afc18910108961a9c0233ecfbf5f529c6a23ddd9a761f76fbcb9c0a536348d18.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/afc18910108961a9c0233ecfbf5f529c6a23ddd9a761f76fbcb9c0a536348d18.jpg)

![e43efebc13d2e20392d651cecaa8a4613746eb8e83b67bc72e42b528b88debfe.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/e43efebc13d2e20392d651cecaa8a4613746eb8e83b67bc72e42b528b88debfe.jpg)

![fdaca0e49f4f6972a57c6d15225dbddefd3f660d841af2476e67f6beb60734ca.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/images/fdaca0e49f4f6972a57c6d15225dbddefd3f660d841af2476e67f6beb60734ca.jpg)

### Tables

![3101c709c7c9348f0f0af2b3dc264542c72a2af4f1f02c79b5c18844a4d10790.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/3101c709c7c9348f0f0af2b3dc264542c72a2af4f1f02c79b5c18844a4d10790.jpg)

![426d009ca303cf84a6ef88b578dfebbd58a34aea3d34def552648d1148630681.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/426d009ca303cf84a6ef88b578dfebbd58a34aea3d34def552648d1148630681.jpg)

![4605cbe4a5a027a7eab3f93d1e9ba50b99272614fce85ccb034ef5494aaaf6f3.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/4605cbe4a5a027a7eab3f93d1e9ba50b99272614fce85ccb034ef5494aaaf6f3.jpg)

![4b74f1aa5adf2113cbc2eb430f2614312bf44ccb10d6045255465ea52a07d7fa.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/4b74f1aa5adf2113cbc2eb430f2614312bf44ccb10d6045255465ea52a07d7fa.jpg)

![7512e7708ba7e20d11abbead057c568b1e3ac64204e67f9e6ed0e4c04feb70c6.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/7512e7708ba7e20d11abbead057c568b1e3ac64204e67f9e6ed0e4c04feb70c6.jpg)

![8c741b670d46f581413e529cf400cf235ffb38f58e255b7f396b36fcbda1165a.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/8c741b670d46f581413e529cf400cf235ffb38f58e255b7f396b36fcbda1165a.jpg)

![90d2350df692f6859fcb7af0e97526d9b4ea8d3d3746b11e3d07ba16c252aa69.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/90d2350df692f6859fcb7af0e97526d9b4ea8d3d3746b11e3d07ba16c252aa69.jpg)

![922ed716810b3c404fd59e44ba014b8b798ebc77836924cf399b43c8edcc4553.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/922ed716810b3c404fd59e44ba014b8b798ebc77836924cf399b43c8edcc4553.jpg)

![ac66d08bbd360c9bccecd48e19845b56b19f115c24acdc2917699ea0c5c8003f.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/ac66d08bbd360c9bccecd48e19845b56b19f115c24acdc2917699ea0c5c8003f.jpg)

![e079134dd71a50f190aba70964d802cd48886fc251652cd4f73132ec024d01eb.jpg](../icml_results/3083_TCAP_%20Tri-Component%20Attention%20Profiling%20for%20Unsupervised%20Backdoor%20Detection%20in%20MLLM%20Fine-Tuning/tables/e079134dd71a50f190aba70964d802cd48886fc251652cd4f73132ec024d01eb.jpg)

<a id="paper-3045"></a>

## Intra-Modal Neighbors Never Lie: Rectifying Inter-Modal Noisy Correspondence via Graph-Based Intra-Modal Reasoning


### Images

![216b061b6a90d90775f18648ca78b0b78926fc87fe82b025cafd972281a96e14.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/images/216b061b6a90d90775f18648ca78b0b78926fc87fe82b025cafd972281a96e14.jpg)

![8ffa295f07734c31c1fc572d82396fa031fbc5a04302a04be582a3e3aa78abe9.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/images/8ffa295f07734c31c1fc572d82396fa031fbc5a04302a04be582a3e3aa78abe9.jpg)

![927f2c03a198cb5bcfd91ed6df2314c7c97b5eb48e423afd80b5605ac319adcb.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/images/927f2c03a198cb5bcfd91ed6df2314c7c97b5eb48e423afd80b5605ac319adcb.jpg)

![d5741a258adb1e0cfbb59d9ef625cd03cc78606ce6e470883970f43f1680c9cf.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/images/d5741a258adb1e0cfbb59d9ef625cd03cc78606ce6e470883970f43f1680c9cf.jpg)

### Tables

![0e979ef09256122587a637b46368466a75bd7bb4bc5efc3c1bf27973b710f8dd.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/0e979ef09256122587a637b46368466a75bd7bb4bc5efc3c1bf27973b710f8dd.jpg)

![132411d8b2485336d518716a08bf8a57b27cf909581a5bd22f67ce33585075cc.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/132411d8b2485336d518716a08bf8a57b27cf909581a5bd22f67ce33585075cc.jpg)

![3647d3b31f1242215ad5cc10b303c250888de94eddaca8879d80cdc7d93f3fc4.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/3647d3b31f1242215ad5cc10b303c250888de94eddaca8879d80cdc7d93f3fc4.jpg)

![739b14a8ebfefe2aede32a3db9c94e1f891135ac375fd4caecfc7affa60a4bbf.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/739b14a8ebfefe2aede32a3db9c94e1f891135ac375fd4caecfc7affa60a4bbf.jpg)

![793315f7686a6b6cc59fb1a480886527b635e9719129fe30d8bb9ad3bb768a23.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/793315f7686a6b6cc59fb1a480886527b635e9719129fe30d8bb9ad3bb768a23.jpg)

![7d9908e05762809b8f26a1e899ba8b8ca505d15805b3a1adb56275a153de6630.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/7d9908e05762809b8f26a1e899ba8b8ca505d15805b3a1adb56275a153de6630.jpg)

![c4d84d0160bb8fec5bfe760438409be1bde34f2a1c7739d3605292c28a339d9c.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/c4d84d0160bb8fec5bfe760438409be1bde34f2a1c7739d3605292c28a339d9c.jpg)

![d0f95deefcaf7f4e5bb6466de1173dd02fede4fc198a36a6e1f56dfb72f69953.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/d0f95deefcaf7f4e5bb6466de1173dd02fede4fc198a36a6e1f56dfb72f69953.jpg)

![efa72431ef7fca337d175a3037cf4f05d6d9ece83b9fbfcf2c3ee80f7c01b1db.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/efa72431ef7fca337d175a3037cf4f05d6d9ece83b9fbfcf2c3ee80f7c01b1db.jpg)

![feb3c4bfd95d4a115fe2f0e1627af6b72fa6fc63153af31feea2b7407f91f107.jpg](../icml_results/3084_Intra-Modal%20Neighbors%20Never%20Lie_%20Rectifying%20Inter-Modal%20Noisy%20Correspondence%20via%20Graph-Based%20Intra-M/tables/feb3c4bfd95d4a115fe2f0e1627af6b72fa6fc63153af31feea2b7407f91f107.jpg)

<a id="paper-3046"></a>

## BRIDGE: Predicting Human Task Completion Time From Model Performance


### Images

![015ea658ec07befcd13849e1dd16e81be228641ae0c17d8bc49afd88803a7c60.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/015ea658ec07befcd13849e1dd16e81be228641ae0c17d8bc49afd88803a7c60.jpg)

![2fe9a429807e462e386b80d817352ae7d7f56d524d2d05f3da58bdccd38069be.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/2fe9a429807e462e386b80d817352ae7d7f56d524d2d05f3da58bdccd38069be.jpg)

![4ba789103b80da7b345f4fdf069318390f34a47f5061caee8130f5ca935f5bb0.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/4ba789103b80da7b345f4fdf069318390f34a47f5061caee8130f5ca935f5bb0.jpg)

![5805e164dafe87f0ca77923fe83fc452ec303c41e5e40adbd110d6427772d3dd.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/5805e164dafe87f0ca77923fe83fc452ec303c41e5e40adbd110d6427772d3dd.jpg)

![6c453f38f98a0faedc0d0fc2c8bb15ae8947b5e726567bf4eddb1631e47e1916.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/6c453f38f98a0faedc0d0fc2c8bb15ae8947b5e726567bf4eddb1631e47e1916.jpg)

![81cc401234dbaaf03c2251b7e882eafc44fc757156faf4198c214d7b5be72a59.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/81cc401234dbaaf03c2251b7e882eafc44fc757156faf4198c214d7b5be72a59.jpg)

![95a7cf25b3a132a31bde596fd1bdddf7a3a2db96fd10ec7e10d6090442bb32a2.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/95a7cf25b3a132a31bde596fd1bdddf7a3a2db96fd10ec7e10d6090442bb32a2.jpg)

![b073bc5058139f640efe542f1210c2b39130f97e29f15c30b8306fc77e210d7e.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/b073bc5058139f640efe542f1210c2b39130f97e29f15c30b8306fc77e210d7e.jpg)

![b32f30dd6dc0f47c522408dd2adc2758daadf845fd63f6370a98d37b62228810.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/b32f30dd6dc0f47c522408dd2adc2758daadf845fd63f6370a98d37b62228810.jpg)

![b6ea7b92147948b3bfe9197c73fcf771f9cffea68ce2c5764ff8ba51523fea23.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/b6ea7b92147948b3bfe9197c73fcf771f9cffea68ce2c5764ff8ba51523fea23.jpg)

![ba885df27ac5317cfbbc2246a2a8f843a02cc3d166b5e9d38bb7bd70c44739af.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/ba885df27ac5317cfbbc2246a2a8f843a02cc3d166b5e9d38bb7bd70c44739af.jpg)

![cff0b0cd94205aaa6baf75537c30609226baf0160979d253ff38a257985710af.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/images/cff0b0cd94205aaa6baf75537c30609226baf0160979d253ff38a257985710af.jpg)

### Tables

![374940aeacee3c86e333f767f64da9127ecae9078c7f7b5b8e1b682e75f79970.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/tables/374940aeacee3c86e333f767f64da9127ecae9078c7f7b5b8e1b682e75f79970.jpg)

![66af5ccd02ef69a7c04d1547cb1eb793913b3c9f5706f1041944da6973dc0aba.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/tables/66af5ccd02ef69a7c04d1547cb1eb793913b3c9f5706f1041944da6973dc0aba.jpg)

![b9f1f15a6708a5e315d60f05c61a33ecae5f4108bca56f77982f4d4e911175fd.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/tables/b9f1f15a6708a5e315d60f05c61a33ecae5f4108bca56f77982f4d4e911175fd.jpg)

![f2891eef38da612083736cb5bcb09a4f229e3dfb5b627f793bbb75842f858e05.jpg](../icml_results/3085_BRIDGE_%20Predicting%20Human%20Task%20Completion%20Time%20From%20Model%20Performance/tables/f2891eef38da612083736cb5bcb09a4f229e3dfb5b627f793bbb75842f858e05.jpg)

<a id="paper-3047"></a>

## From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers


### Images

![23d060a31536b92a5177ae6ec8fec586e62a15afaf88ffce457329f4e4ac9378.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/23d060a31536b92a5177ae6ec8fec586e62a15afaf88ffce457329f4e4ac9378.jpg)

![42f95dae7e920dd7092114f077f2f5eb890945a1cd56b5bdd63afc5c4b4b1161.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/42f95dae7e920dd7092114f077f2f5eb890945a1cd56b5bdd63afc5c4b4b1161.jpg)

![4ca35fd704dd49269b066c17d3a81567867951796dfa3afba51c42151c3d3be1.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/4ca35fd704dd49269b066c17d3a81567867951796dfa3afba51c42151c3d3be1.jpg)

![4e95b0ddc3c39d25f642f58ac1e0ab8cac6d9ea20a913e7b1155c974860cb335.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/4e95b0ddc3c39d25f642f58ac1e0ab8cac6d9ea20a913e7b1155c974860cb335.jpg)

![6b57eb528006c9bcc7c716e371da3b0a746e539e9af980bd2807c5ce230306c9.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/6b57eb528006c9bcc7c716e371da3b0a746e539e9af980bd2807c5ce230306c9.jpg)

![76f2f24aeca5ca7e13b8bfd8cfa0e08df89101894fba9fca9a8761ea260f05f7.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/76f2f24aeca5ca7e13b8bfd8cfa0e08df89101894fba9fca9a8761ea260f05f7.jpg)

![7d5e27ac152e71592ca560c439fb59a8e481384e6f754beb747f5cbfcdceda90.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/7d5e27ac152e71592ca560c439fb59a8e481384e6f754beb747f5cbfcdceda90.jpg)

![89912f4f2158a76ed1b831ab6c183d84525896a9f3436487c7bd2a76a02cffa9.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/89912f4f2158a76ed1b831ab6c183d84525896a9f3436487c7bd2a76a02cffa9.jpg)

![af2b85cf89358fcabef912a62272c1cd3bed441e0f54cd23095dc0029e80e503.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/af2b85cf89358fcabef912a62272c1cd3bed441e0f54cd23095dc0029e80e503.jpg)

![f10569155a1a2589d916c699fb17433c1b08e921220115ae12c9a8c0f328780b.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/f10569155a1a2589d916c699fb17433c1b08e921220115ae12c9a8c0f328780b.jpg)

![f68df82be062a991cfef7337e87bf26b84ebff39c9c0dcd0627e24a460f9b39c.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/f68df82be062a991cfef7337e87bf26b84ebff39c9c0dcd0627e24a460f9b39c.jpg)

![f8f931170d316841e208a8d32752da7f5c8b701562134b381fb806026c52a32f.jpg](../icml_results/3086_From%20Kepler%20to%20Newton_%20Inductive%20Biases%20Guide%20Learned%20World%20Models%20in%20Transformers/images/f8f931170d316841e208a8d32752da7f5c8b701562134b381fb806026c52a32f.jpg)

<a id="paper-3048"></a>

## Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners


### Images

![024bbb12c88020f6e9434a32947426ffdd24ad6277460c98c112cbcc46c0a144.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/024bbb12c88020f6e9434a32947426ffdd24ad6277460c98c112cbcc46c0a144.jpg)

![1d3f6186001a19ab3363d33058b713886cbbca45c26469ec819b7cc150d1f800.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/1d3f6186001a19ab3363d33058b713886cbbca45c26469ec819b7cc150d1f800.jpg)

![222d561c684274003e8019bc9ee543609a98b6c443cef0e8e14c28106c2c6d91.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/222d561c684274003e8019bc9ee543609a98b6c443cef0e8e14c28106c2c6d91.jpg)

![22bf2f6bea1c282564504b5c0f5614c2a806da1949467097f23d4a93a280b33b.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/22bf2f6bea1c282564504b5c0f5614c2a806da1949467097f23d4a93a280b33b.jpg)

![28c58c8c44a48703345aa2657a5de64b7e3cc35784e461d9c9e42acc4302f600.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/28c58c8c44a48703345aa2657a5de64b7e3cc35784e461d9c9e42acc4302f600.jpg)

![33fa72ac9d03bc45e7fe84894a39a1208126869d12cfebf9659e3cfab43ed7bf.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/33fa72ac9d03bc45e7fe84894a39a1208126869d12cfebf9659e3cfab43ed7bf.jpg)

![379fa68ff74f09540b44e6b455fdd0e55d1600358d793b38b246d6d8b22f261c.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/379fa68ff74f09540b44e6b455fdd0e55d1600358d793b38b246d6d8b22f261c.jpg)

![509d8afa8db50658041babffeb91005d3a0a88f734951d2ab6ef0d460098b23d.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/509d8afa8db50658041babffeb91005d3a0a88f734951d2ab6ef0d460098b23d.jpg)

![519ff4086346d17d7a58380895b41f690cfb4dd7acf9841c2cc57b450cc0f8e8.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/519ff4086346d17d7a58380895b41f690cfb4dd7acf9841c2cc57b450cc0f8e8.jpg)

![55f50b9534c39b54870b819e4c1697393cd855f349bf012f647c56ce95e9d0ac.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/55f50b9534c39b54870b819e4c1697393cd855f349bf012f647c56ce95e9d0ac.jpg)

![599f12b9f87df3449dde50c6d7ea466a18db6f73c8fe89a2ead8b72ce0d3154b.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/599f12b9f87df3449dde50c6d7ea466a18db6f73c8fe89a2ead8b72ce0d3154b.jpg)

![63091b3011b43e7407091d43b10b290247653d5e4594130f5227a8ba72a2d84f.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/63091b3011b43e7407091d43b10b290247653d5e4594130f5227a8ba72a2d84f.jpg)

![63a7cd2db828b1aec78107d3722be13b89593e29eb16c6989bb3973b7c522219.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/63a7cd2db828b1aec78107d3722be13b89593e29eb16c6989bb3973b7c522219.jpg)

![6efe9bd80f5bc0f13f813efeb3aa7f59cd007d4a6d1d73aed69ddb381fdfe50c.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/6efe9bd80f5bc0f13f813efeb3aa7f59cd007d4a6d1d73aed69ddb381fdfe50c.jpg)

![731be3942f18a577595abfdd3b760aa92265723b7ece688fce469515cae3d68a.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/731be3942f18a577595abfdd3b760aa92265723b7ece688fce469515cae3d68a.jpg)

![84b82add1a89eb3c172e16f64907e0d9266d2709e9ec6e3cdcad9f8126cb3189.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/84b82add1a89eb3c172e16f64907e0d9266d2709e9ec6e3cdcad9f8126cb3189.jpg)

![8b9a0a89c3e5bf29cb0c33b09a6396885e6dd6ad6c13ed31f8497e531cfbc3a4.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/8b9a0a89c3e5bf29cb0c33b09a6396885e6dd6ad6c13ed31f8497e531cfbc3a4.jpg)

![a5819d5b6db028b8cbcdcb8278418c378740fe0521a76570a8ab6f7c868cbd57.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/a5819d5b6db028b8cbcdcb8278418c378740fe0521a76570a8ab6f7c868cbd57.jpg)

![a9379b96378a8923d123cf0966806ece72006d2d35bf0eae89c9d70cb0833c03.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/a9379b96378a8923d123cf0966806ece72006d2d35bf0eae89c9d70cb0833c03.jpg)

![b09a61857660b15d18c35a772769fc4d4753004fcfa44f40805e5dad727a373a.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/b09a61857660b15d18c35a772769fc4d4753004fcfa44f40805e5dad727a373a.jpg)

![b9f22364963cddb3eb07ce8b967f7b6011b1f753e588db01ac06ed69e590815a.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/b9f22364963cddb3eb07ce8b967f7b6011b1f753e588db01ac06ed69e590815a.jpg)

![c6f4ddcb9f934d712d95127f4a0bce005430108a846100317d6ddf5bf800bbdc.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/c6f4ddcb9f934d712d95127f4a0bce005430108a846100317d6ddf5bf800bbdc.jpg)

![d7168b8af3bdd6867e618ad66adbff08722c0f0a14ec0b3f0a1af8a13588656d.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/d7168b8af3bdd6867e618ad66adbff08722c0f0a14ec0b3f0a1af8a13588656d.jpg)

![f0dd0b83d557d0e2df7b1ea3e4e5ebf1f0e14c0e27bedbebc6d394412036f7ea.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/images/f0dd0b83d557d0e2df7b1ea3e4e5ebf1f0e14c0e27bedbebc6d394412036f7ea.jpg)

### Tables

![0a1e785e24668a3e6df3c876161907774476d8384e4c7a210a0e89040eb48e9b.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/tables/0a1e785e24668a3e6df3c876161907774476d8384e4c7a210a0e89040eb48e9b.jpg)

![532d9953a4f4ec20a4c9221a16a342b9ef8339d0aea56aa47c84d01d7dd963f4.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/tables/532d9953a4f4ec20a4c9221a16a342b9ef8339d0aea56aa47c84d01d7dd963f4.jpg)

![794a01ac445913640f66e8383890747bc3adecd6ee6a56cbd806714d1f7ee53d.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/tables/794a01ac445913640f66e8383890747bc3adecd6ee6a56cbd806714d1f7ee53d.jpg)

![e24e8e787f4604ebc14090e0754081d794cede45dffd2408e06eb5705f01cbf8.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/tables/e24e8e787f4604ebc14090e0754081d794cede45dffd2408e06eb5705f01cbf8.jpg)

![e629eb95434df1ba4b735378be866a35e221ed0a4a0aff5fc6f19973affd81f1.jpg](../icml_results/3087_Breaking%20Dual%20Bottlenecks_%20Evolving%20Unified%20Multimodal%20Models%20into%20Self-Adaptive%20Interleaved%20Visual%20/tables/e629eb95434df1ba4b735378be866a35e221ed0a4a0aff5fc6f19973affd81f1.jpg)

<a id="paper-3049"></a>

## Minimizing Upper Confidence Bounds: A Data-Driven Framework for Stochastic Programming


### Images

![507c9bf537f9c2c141ce232ff14a680b81370a49a2f97b924ae1eb8abf64ea2c.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/507c9bf537f9c2c141ce232ff14a680b81370a49a2f97b924ae1eb8abf64ea2c.jpg)

![9a1560bc05c0f13680d0cead0d8e231d9c4dc6cd215dcb6a7391a18e2863ad9c.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/9a1560bc05c0f13680d0cead0d8e231d9c4dc6cd215dcb6a7391a18e2863ad9c.jpg)

![9dbecb6dd5bc316ad79fb20e1c7c0282b869bb904fa02c36c60f6bf15ba5245f.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/9dbecb6dd5bc316ad79fb20e1c7c0282b869bb904fa02c36c60f6bf15ba5245f.jpg)

![a41c826b2e5bd8e45a5776b10941fd0acadf61c164b380c22bd029403e8aee0d.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/a41c826b2e5bd8e45a5776b10941fd0acadf61c164b380c22bd029403e8aee0d.jpg)

![df1aff4ff78a3d08a009a7fb795696fd7fc930b8d3516a42122a5d96221eb5ac.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/df1aff4ff78a3d08a009a7fb795696fd7fc930b8d3516a42122a5d96221eb5ac.jpg)

![e369dc1cc69c0fb22a8700ff86a24143f4af11be9e12148f75c65f3e672cae37.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/images/e369dc1cc69c0fb22a8700ff86a24143f4af11be9e12148f75c65f3e672cae37.jpg)

### Tables

![12f7d9f907594bc442bb87f8f4726279ccc0e02204145610e76c7ef91bb6fb7e.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/tables/12f7d9f907594bc442bb87f8f4726279ccc0e02204145610e76c7ef91bb6fb7e.jpg)

![56c96670db903b30e1ce2ae76aa094ad7c7cabe70681d34edf88d6f79aaa7788.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/tables/56c96670db903b30e1ce2ae76aa094ad7c7cabe70681d34edf88d6f79aaa7788.jpg)

![a9af4bf27e19ab35ef045e8b6dc8801e06775b7e9b1a3400c9bed0cd2701cc5e.jpg](../icml_results/3088_Minimizing%20Upper%20Confidence%20Bounds_%20A%20Data-Driven%20Framework%20for%20Stochastic%20Programming/tables/a9af4bf27e19ab35ef045e8b6dc8801e06775b7e9b1a3400c9bed0cd2701cc5e.jpg)

<a id="paper-3050"></a>

## NeurVLA: Unleashing Failure-Handling Capability of Vision-Language-Action Models via Neural-Symbolic Reasoning


### Images

![5051cd592e1da17d420b07fcb622f74f924e9dc45b8549c964f753b416fb60ab.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/5051cd592e1da17d420b07fcb622f74f924e9dc45b8549c964f753b416fb60ab.jpg)

![76f69b1ca2c768a2d15870a97299df05393290dd713a59517883bc524fcfc002.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/76f69b1ca2c768a2d15870a97299df05393290dd713a59517883bc524fcfc002.jpg)

![790678dd4ff61495318a2dae35384878a44d6c8c4afc257716f1c956d0f32e29.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/790678dd4ff61495318a2dae35384878a44d6c8c4afc257716f1c956d0f32e29.jpg)

![8bbe9613f3a5c0d201eb38a90ac32434c275c059ab0bac67cb05d2962b8dac54.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/8bbe9613f3a5c0d201eb38a90ac32434c275c059ab0bac67cb05d2962b8dac54.jpg)

![92a395cd3b0965c178d829fa941ee104f0737519706049e16bb2e5b3d9d9aa40.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/92a395cd3b0965c178d829fa941ee104f0737519706049e16bb2e5b3d9d9aa40.jpg)

![95bb0a63ca916ff2d1a9bd17689ee971e0c4f28977dd096ad2e52f07ed1e6799.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/95bb0a63ca916ff2d1a9bd17689ee971e0c4f28977dd096ad2e52f07ed1e6799.jpg)

![a005559478cfaba1245e1cd32d2c62c09048dd444bca38e5ca33471ed1d97dbd.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/a005559478cfaba1245e1cd32d2c62c09048dd444bca38e5ca33471ed1d97dbd.jpg)

![f1d0e3d57bc251f41c86f5c88b2c1b8a7ed845828462f530899e95c927b38443.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/images/f1d0e3d57bc251f41c86f5c88b2c1b8a7ed845828462f530899e95c927b38443.jpg)

### Tables

![27744648653dcd621c1fb2372bb90fa440a106efd287b0e99ee7ced0eed3d8af.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/tables/27744648653dcd621c1fb2372bb90fa440a106efd287b0e99ee7ced0eed3d8af.jpg)

![3481a1a18e239eb3854c703f1c2ccf532f9f011be40d6ef84494e32cd50ee94c.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/tables/3481a1a18e239eb3854c703f1c2ccf532f9f011be40d6ef84494e32cd50ee94c.jpg)

![39a063b46ddfb4fe0c8d1d0be099f5822f89b409fcc249c324f83c86244ee820.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/tables/39a063b46ddfb4fe0c8d1d0be099f5822f89b409fcc249c324f83c86244ee820.jpg)

![43f6151e0edd2a8f93083deb693b9e018fbb06949ead6576672366f001df1fb9.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/tables/43f6151e0edd2a8f93083deb693b9e018fbb06949ead6576672366f001df1fb9.jpg)

![b0fa4184c66325962edc4a94c1cc6f4eabf40c74ba28f11cb6948675da9b021f.jpg](../icml_results/3089_NeurVLA_%20Unleashing%20Failure-Handling%20Capability%20of%20Vision-Language-Action%20Models%20via%20Neural-Symbolic/tables/b0fa4184c66325962edc4a94c1cc6f4eabf40c74ba28f11cb6948675da9b021f.jpg)

<a id="paper-3051"></a>

## The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design


### Images

![08437fa13d343a24b4ff80ec5da4aada039366700cd4887b147262b6a36c59f4.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/08437fa13d343a24b4ff80ec5da4aada039366700cd4887b147262b6a36c59f4.jpg)

![08ae8926d5a0ea1dec86687aede8a968250a1f092d07e9bcb95cba87c002cce2.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/08ae8926d5a0ea1dec86687aede8a968250a1f092d07e9bcb95cba87c002cce2.jpg)

![53141261ff5fa6fd93ca1ed0527cd19d9c0a53ca40188ddfce74ed1c8653bd8f.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/53141261ff5fa6fd93ca1ed0527cd19d9c0a53ca40188ddfce74ed1c8653bd8f.jpg)

![658d69dd388f699100faacba92952aaa00a9d326e547cc64cf8937af07984ad6.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/658d69dd388f699100faacba92952aaa00a9d326e547cc64cf8937af07984ad6.jpg)

![6b634696e460860fdfdff0e7e3aafe8078c97e138c50020d0d3b072755a3abae.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/6b634696e460860fdfdff0e7e3aafe8078c97e138c50020d0d3b072755a3abae.jpg)

![6f85b4db601ad56b8b45eeaf433b04230943f404a7a500040bf244736c69e677.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/6f85b4db601ad56b8b45eeaf433b04230943f404a7a500040bf244736c69e677.jpg)

![8aad1578ded594c4e06314a1b7302f9a5b24651f6ce55bd22256cf126b09d725.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/8aad1578ded594c4e06314a1b7302f9a5b24651f6ce55bd22256cf126b09d725.jpg)

![958b5b8d83ccaa5e958f59fc851ce214ddbea315fd35b776cd7dbb938cb95340.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/958b5b8d83ccaa5e958f59fc851ce214ddbea315fd35b776cd7dbb938cb95340.jpg)

![98ae0fed682656f9684890f2fdf954271a463ed89113e336a51f4d7476335d4b.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/98ae0fed682656f9684890f2fdf954271a463ed89113e336a51f4d7476335d4b.jpg)

![ced23328dbf14fe8dd30bee329965a4cf019763e671dde5c8724b49aa1ef19ea.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/ced23328dbf14fe8dd30bee329965a4cf019763e671dde5c8724b49aa1ef19ea.jpg)

![d182a2f585504514f6efb9cd72fa0c09696a6778c7af8153b628859ee9419ee3.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/d182a2f585504514f6efb9cd72fa0c09696a6778c7af8153b628859ee9419ee3.jpg)

![d230a26789d24d18a02d1214a1c1e47d800669a70250caa399038186f1157926.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/d230a26789d24d18a02d1214a1c1e47d800669a70250caa399038186f1157926.jpg)

![d972ed13bf2197556bb0eebe583fa21655f980707ede6e4dadb7bd2de4b27600.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/d972ed13bf2197556bb0eebe583fa21655f980707ede6e4dadb7bd2de4b27600.jpg)

![f941b42b595ab62b58de57c02c05c7c55ea9960b675fe834a2ad99978a3b965e.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/images/f941b42b595ab62b58de57c02c05c7c55ea9960b675fe834a2ad99978a3b965e.jpg)

### Tables

![0529781ef645132e952e05ea2a231fad5fafba305d2b311b46f1e53bc619bb11.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/0529781ef645132e952e05ea2a231fad5fafba305d2b311b46f1e53bc619bb11.jpg)

![08e52768c8b4c6b16a4004ff8a234b61d823257ebea9412bdce21031fa074253.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/08e52768c8b4c6b16a4004ff8a234b61d823257ebea9412bdce21031fa074253.jpg)

![9514a46064259c61052e3b62dd18efd97b3f332b3e413404e52d9892554ea60d.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/9514a46064259c61052e3b62dd18efd97b3f332b3e413404e52d9892554ea60d.jpg)

![98d43617d8a506b96e13455ffc3c5ae40686fb5b4785f1713ba8b62a7a1029be.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/98d43617d8a506b96e13455ffc3c5ae40686fb5b4785f1713ba8b62a7a1029be.jpg)

![aaa5d6f8e7428eae58b0c5e3a2100296aec4a43d4a067e64cc58578bac9f3683.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/aaa5d6f8e7428eae58b0c5e3a2100296aec4a43d4a067e64cc58578bac9f3683.jpg)

![ced29928b2313ec384bba81620b13200c34b1456d3c42ddc8753cdfc54fe1a78.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/ced29928b2313ec384bba81620b13200c34b1456d3c42ddc8753cdfc54fe1a78.jpg)

![dd6f8eb0efed546dea3dc77ce4dc41c9c36bb9ca968f9a224bb96dd9db66119b.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/dd6f8eb0efed546dea3dc77ce4dc41c9c36bb9ca968f9a224bb96dd9db66119b.jpg)

![e100e8548f549ac2efe1141cf3db111b6c218ce97c744316e3defeb4294b9fc5.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/e100e8548f549ac2efe1141cf3db111b6c218ce97c744316e3defeb4294b9fc5.jpg)

![f61f01cce224f2e7080919e99ad36e2924ae9bc9d75b8a37be9434a454b1dedc.jpg](../icml_results/3090_The%20Perceptual%20Bandwidth%20Bottleneck%20in%20Vision-Language%20Models_%20Active%20Visual%20Reasoning%20via%20Sequentia/tables/f61f01cce224f2e7080919e99ad36e2924ae9bc9d75b8a37be9434a454b1dedc.jpg)

<a id="paper-3052"></a>

## PrivCode++: Latent-Conditioned Differentially Private Code Generation for Comprehensive Guarantees


### Images

![46f7b4cc1c9054ff69af0c027c5ed47f76010bed1697055c3d1b6b56d34f2b7d.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/46f7b4cc1c9054ff69af0c027c5ed47f76010bed1697055c3d1b6b56d34f2b7d.jpg)

![56c8e74b285deddce1049e15e287ce949c12781db0fe2b83941dd06f79a2a574.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/56c8e74b285deddce1049e15e287ce949c12781db0fe2b83941dd06f79a2a574.jpg)

![804b8ff3b38cefd494aef591dc2b1ec1d6e46120499ec78348892c2ea7831d16.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/804b8ff3b38cefd494aef591dc2b1ec1d6e46120499ec78348892c2ea7831d16.jpg)

![86fcc5cd084d0f5d63d541333fe7f887630877b174b49647c56d2fa5d9d29c4d.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/86fcc5cd084d0f5d63d541333fe7f887630877b174b49647c56d2fa5d9d29c4d.jpg)

![9aea9f45fccb7949e1dec7f56132187907c3528fdf208c54bd6fc82ad183dec6.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/9aea9f45fccb7949e1dec7f56132187907c3528fdf208c54bd6fc82ad183dec6.jpg)

![9f6f407e90ad1b6fbbaf58130ed8d843d5df6f955b050fe3958fbebd43420c10.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/images/9f6f407e90ad1b6fbbaf58130ed8d843d5df6f955b050fe3958fbebd43420c10.jpg)

### Tables

![30783201ed87767d3a5df2f9a7e6e9760a4e89270f086875ac1c96ab527d2708.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/30783201ed87767d3a5df2f9a7e6e9760a4e89270f086875ac1c96ab527d2708.jpg)

![38016ecdc8ef57efcd88262a262626eb4822ae595d6d5288006cb7a812beb96d.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/38016ecdc8ef57efcd88262a262626eb4822ae595d6d5288006cb7a812beb96d.jpg)

![4c70ca8c2f0faab3456dbd1733c5ce4655d641ffb14212c65c4d0f383e7218de.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/4c70ca8c2f0faab3456dbd1733c5ce4655d641ffb14212c65c4d0f383e7218de.jpg)

![5fd9956701bacc546aa309e97c634777abffd3e75baec2cb15519d6c3382cdb2.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/5fd9956701bacc546aa309e97c634777abffd3e75baec2cb15519d6c3382cdb2.jpg)

![8c64ad46acd63690bd9374de3b05e7bf3f4052d598edc73e2ccc9c2ab1f8a433.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/8c64ad46acd63690bd9374de3b05e7bf3f4052d598edc73e2ccc9c2ab1f8a433.jpg)

![8dc9af63a2131a145aaf514aea691023a7dda68fce604144a617567c50fffa72.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/8dc9af63a2131a145aaf514aea691023a7dda68fce604144a617567c50fffa72.jpg)

![97169d2c4bc79a49c0626339efb98d58bab647c5ca357deead85cb008adeb21f.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/97169d2c4bc79a49c0626339efb98d58bab647c5ca357deead85cb008adeb21f.jpg)

![c1b4d566712f0d5a45b8630cd607a2eb1b7b264d95ecd323c8dc5815764bfb29.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/c1b4d566712f0d5a45b8630cd607a2eb1b7b264d95ecd323c8dc5815764bfb29.jpg)

![d18737a83b7f1451bdc003906b8ce8633330b891521c0b6bb67c8a738743993e.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/d18737a83b7f1451bdc003906b8ce8633330b891521c0b6bb67c8a738743993e.jpg)

![e44e5ee59357b59b174bfcf8cbcb5e97bc6dcfa23d6f65820676fdd6fb4832a9.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/e44e5ee59357b59b174bfcf8cbcb5e97bc6dcfa23d6f65820676fdd6fb4832a9.jpg)

![e7c9e905c2900d1df9e1f0b7dab4ed439646425444495276bd7a0bc040171936.jpg](../icml_results/3091_PrivCode%2B%2B_%20Latent-Conditioned%20Differentially%20Private%20Code%20Generation%20for%20Comprehensive%20Guarantees/tables/e7c9e905c2900d1df9e1f0b7dab4ed439646425444495276bd7a0bc040171936.jpg)

<a id="paper-3053"></a>

## ViEEG: Hierarchical Visual Neural Representation for EEG Brain Decoding


### Images

![0b6b2979240bd8704f356041bc920b72d31a276c9cdaafbe6c9872c465fac75b.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/0b6b2979240bd8704f356041bc920b72d31a276c9cdaafbe6c9872c465fac75b.jpg)

![121ebe94f0c68b60b28170b3e360d1f22e09d302087a2540a99b96552f46aa29.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/121ebe94f0c68b60b28170b3e360d1f22e09d302087a2540a99b96552f46aa29.jpg)

![2dfa5bceac79128ff171de0e53443389a10536b10e80a00fe49faf0c1af3b47e.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/2dfa5bceac79128ff171de0e53443389a10536b10e80a00fe49faf0c1af3b47e.jpg)

![32358a44c87ecdac3b682446a95b2350fca9d28a999b2e4d9f6630cb6bbe7bec.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/32358a44c87ecdac3b682446a95b2350fca9d28a999b2e4d9f6630cb6bbe7bec.jpg)

![344e338442a041cf7f21da46d598e5f332070e02461029baacf5827e1d9ae117.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/344e338442a041cf7f21da46d598e5f332070e02461029baacf5827e1d9ae117.jpg)

![3f3635461d875dc26b8b6c4b149b261bbde624f3c546bf6d43ae7c03e3f67911.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/3f3635461d875dc26b8b6c4b149b261bbde624f3c546bf6d43ae7c03e3f67911.jpg)

![40e56f2296a296893ffff7fc76ee6b89209324f43115668c58a95136af90f14b.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/40e56f2296a296893ffff7fc76ee6b89209324f43115668c58a95136af90f14b.jpg)

![47a0e1d18d5f115f13cc10e81c479ca3dc02898e913baa0d8a1c00222f73c6c9.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/47a0e1d18d5f115f13cc10e81c479ca3dc02898e913baa0d8a1c00222f73c6c9.jpg)

![6222c9bf82b9b04f48778d5054b5620036c9b0c057c36bf09789d9aeb54ce142.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/6222c9bf82b9b04f48778d5054b5620036c9b0c057c36bf09789d9aeb54ce142.jpg)

![66069bc6806b07fc7b3558ce6eb9ecd4df7b35825530db33e2047ae2649089b0.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/66069bc6806b07fc7b3558ce6eb9ecd4df7b35825530db33e2047ae2649089b0.jpg)

![7fbbc06f085178a01a3b7a66372598b6c181c5b685fff853219e60fab3e08c29.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/7fbbc06f085178a01a3b7a66372598b6c181c5b685fff853219e60fab3e08c29.jpg)

![80089f7c3151910898977e76628873f5146458feca6e9509e9494b9479f5fa52.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/80089f7c3151910898977e76628873f5146458feca6e9509e9494b9479f5fa52.jpg)

![8f2f58086f79f76a0b6a22b462f0a5211ae25af2696d8d0d1fda91333344d762.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/8f2f58086f79f76a0b6a22b462f0a5211ae25af2696d8d0d1fda91333344d762.jpg)

![936d138dd3815986acd815fc60c3181609d70df8a9a54eb46aefe1f65ea2dba7.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/936d138dd3815986acd815fc60c3181609d70df8a9a54eb46aefe1f65ea2dba7.jpg)

![a04806d1c319c04a6eeb2ccda13d3ce4a8b6419d604180b9b8d1d9a4a07dfb03.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/a04806d1c319c04a6eeb2ccda13d3ce4a8b6419d604180b9b8d1d9a4a07dfb03.jpg)

![ac835b09031507b4a7bf0e36eb008b3b06e8f35e0519a64a985ef70ace1a8e16.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/ac835b09031507b4a7bf0e36eb008b3b06e8f35e0519a64a985ef70ace1a8e16.jpg)

![f23cf634bb7b70c7a74f5e12863337480673e7742576d2716fbb12efeb3e28ec.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/f23cf634bb7b70c7a74f5e12863337480673e7742576d2716fbb12efeb3e28ec.jpg)

![f9212f87338bf4b6d7564d98532d1f7dfa4ddcbb498ea128b7a2cfc3710bd922.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/images/f9212f87338bf4b6d7564d98532d1f7dfa4ddcbb498ea128b7a2cfc3710bd922.jpg)

### Tables

![05bcf3c51b7f6826b7ee421e7392d366046d89e15bbd474d2e4c0c238531b103.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/05bcf3c51b7f6826b7ee421e7392d366046d89e15bbd474d2e4c0c238531b103.jpg)

![27625782c8f7338fb568ff6dce6bdc91d98f9a9b0f9dcd86bbd5685532ed58b9.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/27625782c8f7338fb568ff6dce6bdc91d98f9a9b0f9dcd86bbd5685532ed58b9.jpg)

![679fa1b8b24b32f6d22382abd5e212b4b73af2ac254f0fb60f6fc7859ace7d00.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/679fa1b8b24b32f6d22382abd5e212b4b73af2ac254f0fb60f6fc7859ace7d00.jpg)

![8e831babc54c603dcea414956521214c3d3e9cfc11b40e847b8b5610794e4f72.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/8e831babc54c603dcea414956521214c3d3e9cfc11b40e847b8b5610794e4f72.jpg)

![940a2cf39acfc3cec951c56b193ccb61fa4c8da3657c26bc475cd33c195db9e4.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/940a2cf39acfc3cec951c56b193ccb61fa4c8da3657c26bc475cd33c195db9e4.jpg)

![d3d2cacf1ad0ca07fb29291fde672225bcd91706bf13b52a50377692d459755c.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/d3d2cacf1ad0ca07fb29291fde672225bcd91706bf13b52a50377692d459755c.jpg)

![dc0a6de043ba8d8ef8784f5cd0b80edcd30685a8168639031ae3760a3906aeb2.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/dc0a6de043ba8d8ef8784f5cd0b80edcd30685a8168639031ae3760a3906aeb2.jpg)

![dd3eb0e705842d80fcdeef6b970a644b0d1af2bc90f7a5b3ec355b0d1773fc58.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/dd3eb0e705842d80fcdeef6b970a644b0d1af2bc90f7a5b3ec355b0d1773fc58.jpg)

![e24f17c7e65c91d9b7731c127de8ad91c67ca264cf22081fe9112d3c46d09c33.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/e24f17c7e65c91d9b7731c127de8ad91c67ca264cf22081fe9112d3c46d09c33.jpg)

![f9c96bbe0252dc44af34864475ea90e9e7055d795db3ff2896a9a304310437b5.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/f9c96bbe0252dc44af34864475ea90e9e7055d795db3ff2896a9a304310437b5.jpg)

![fda20aa613f39e767ca8852a0ae669577912b43cd69f30f9da689ac233fcecdb.jpg](../icml_results/3092_ViEEG_%20Hierarchical%20Visual%20Neural%20Representation%20for%20EEG%20Brain%20Decoding/tables/fda20aa613f39e767ca8852a0ae669577912b43cd69f30f9da689ac233fcecdb.jpg)

<a id="paper-3054"></a>

## LieStoNet: Learning Lie Symmetries from Spatiotemporal Data for Stochastic Dynamical Systems


### Images

![07dd2e589bcf0f5d7b9b0384f62c83393ede49dfb20eb4e9407c5dde91ef89e3.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/07dd2e589bcf0f5d7b9b0384f62c83393ede49dfb20eb4e9407c5dde91ef89e3.jpg)

![0c7e27fcf47d660e00d04e863a6ca0da38a3a291977a94a4558527e6a6ad0b18.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/0c7e27fcf47d660e00d04e863a6ca0da38a3a291977a94a4558527e6a6ad0b18.jpg)

![3f94ec3ce6afe119f36d696896fc0ac26f128eb45c5b19a99e4aaadecb182f13.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/3f94ec3ce6afe119f36d696896fc0ac26f128eb45c5b19a99e4aaadecb182f13.jpg)

![4ba9e608a49b8065ed5fac959eb9f7551a765272fe98e8fa9c453c2cd9c30528.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/4ba9e608a49b8065ed5fac959eb9f7551a765272fe98e8fa9c453c2cd9c30528.jpg)

![8ab35c2d3d8f35f0830378a8bf62d79f0311a551537022b2afbc34ca50196f48.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/8ab35c2d3d8f35f0830378a8bf62d79f0311a551537022b2afbc34ca50196f48.jpg)

![bb8228d3f2a5527c59dd1dab707a5451a5d886d201d8dcc1f54a0bc020cd4ea0.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/bb8228d3f2a5527c59dd1dab707a5451a5d886d201d8dcc1f54a0bc020cd4ea0.jpg)

![c02f90f956d3808f8bce94aac73365a13d4b4313b40ff1768f3561401fc101ce.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/c02f90f956d3808f8bce94aac73365a13d4b4313b40ff1768f3561401fc101ce.jpg)

![c35eddd9e1e7955cd69e4685cc321cb68ef749e5bc9753896ca72697a42dffa5.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/images/c35eddd9e1e7955cd69e4685cc321cb68ef749e5bc9753896ca72697a42dffa5.jpg)

### Tables

![1c0beda8c2578d94bbddf62478453297b740638727bb27554b1685f56eafcf3b.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/1c0beda8c2578d94bbddf62478453297b740638727bb27554b1685f56eafcf3b.jpg)

![247b35fec70cbbdbc1e47f4ea585ac75fd16d3c72d1c83e114fa51412357b669.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/247b35fec70cbbdbc1e47f4ea585ac75fd16d3c72d1c83e114fa51412357b669.jpg)

![2895dda93d2b371277c0354b406de09ae94115c939b0d3b85d0840dd3436f18f.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/2895dda93d2b371277c0354b406de09ae94115c939b0d3b85d0840dd3436f18f.jpg)

![2dc14690cc51d9796435988957ca3afb115b2661b0e6f28f015dae78c5d2948a.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/2dc14690cc51d9796435988957ca3afb115b2661b0e6f28f015dae78c5d2948a.jpg)

![3611d51677ca75b3581628a4caff34656aae51a11f7684bd9d62288cdfb1b5de.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/3611d51677ca75b3581628a4caff34656aae51a11f7684bd9d62288cdfb1b5de.jpg)

![3f9389d303112ecaf70b0ae70700448726446796325fc1a3b918a631bef4b77d.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/3f9389d303112ecaf70b0ae70700448726446796325fc1a3b918a631bef4b77d.jpg)

![41be066c5dd2efbf6ea6e151276feb0f72f28b269597be860d32927c5508bf62.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/41be066c5dd2efbf6ea6e151276feb0f72f28b269597be860d32927c5508bf62.jpg)

![6cfc7a9b9aceffe734c21669b16dd9a45004159fd10744fe8de08d70142fb87b.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/6cfc7a9b9aceffe734c21669b16dd9a45004159fd10744fe8de08d70142fb87b.jpg)

![73044f400b68b4425c4a1cd46c131705e118d5cc94da2cf0b604dbbdc64e98b8.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/73044f400b68b4425c4a1cd46c131705e118d5cc94da2cf0b604dbbdc64e98b8.jpg)

![8f9931ba5c2c9652f6f49ff0982abffa81938cefa8923ac0e443fa29540a07aa.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/8f9931ba5c2c9652f6f49ff0982abffa81938cefa8923ac0e443fa29540a07aa.jpg)

![98565316c0d771bafe53a8b5f550f74e0ea2259e7ece02cba21f2958c9796c9f.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/98565316c0d771bafe53a8b5f550f74e0ea2259e7ece02cba21f2958c9796c9f.jpg)

![9edf9d6106e5515b451e4e3602a66860261099ef2025375b6ef144376591ea24.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/9edf9d6106e5515b451e4e3602a66860261099ef2025375b6ef144376591ea24.jpg)

![a024b7a6ca63ee55656c2eab0fec082b092bb3e070db7cc16c3330bbc14b3685.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/a024b7a6ca63ee55656c2eab0fec082b092bb3e070db7cc16c3330bbc14b3685.jpg)

![ba112d6b43043f4bcf8d4d51b392472204b4ca66e22a614e7615c463834a13ac.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/ba112d6b43043f4bcf8d4d51b392472204b4ca66e22a614e7615c463834a13ac.jpg)

![c0cc4a026641b1a2437cd0285c412ce8e66702e2ce5d14a44043fd1d2be047b1.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/c0cc4a026641b1a2437cd0285c412ce8e66702e2ce5d14a44043fd1d2be047b1.jpg)

![dfaa8ad757a9b4cbc1abb431796e6d5fb6ed095a038bcca9f08b6ece302749cf.jpg](../icml_results/3093_LieStoNet_%20Learning%20Lie%20Symmetries%20from%20Spatiotemporal%20Data%20for%20Stochastic%20Dynamical%20Systems/tables/dfaa8ad757a9b4cbc1abb431796e6d5fb6ed095a038bcca9f08b6ece302749cf.jpg)

<a id="paper-3055"></a>

## TGV-KV: Text-Grounded KV Eviction for Vision-Language Models


### Images

![1fbc667944eedd9e12d3becc45d604548ea6fe110a495cde8fb09507864ae358.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/1fbc667944eedd9e12d3becc45d604548ea6fe110a495cde8fb09507864ae358.jpg)

![28bb6eba5119010e77dea0007364f874a0374ab431fb9f29d8559bcf89e4d3d4.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/28bb6eba5119010e77dea0007364f874a0374ab431fb9f29d8559bcf89e4d3d4.jpg)

![29bc424fae0f6bc1112ae618000f4da1ec698135be2885f210327430f40695f4.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/29bc424fae0f6bc1112ae618000f4da1ec698135be2885f210327430f40695f4.jpg)

![4925d3ad3e8f581363fae4d0680b43013ebf869f6fe0627874a17d76b702f85a.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/4925d3ad3e8f581363fae4d0680b43013ebf869f6fe0627874a17d76b702f85a.jpg)

![5bc162afc294471b9113e3507ecef6534ad84b14f4b24f8d13878c54545391b0.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/5bc162afc294471b9113e3507ecef6534ad84b14f4b24f8d13878c54545391b0.jpg)

![d66b98a2eafeb38d56e32c3b955fed4aefea3b089cd79b647a6c88f413e4f853.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/d66b98a2eafeb38d56e32c3b955fed4aefea3b089cd79b647a6c88f413e4f853.jpg)

![f25dc1da19d7477caf9ecc508cf6fb5f7cf3fe3ddb55567902fab6125fc04a8b.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/f25dc1da19d7477caf9ecc508cf6fb5f7cf3fe3ddb55567902fab6125fc04a8b.jpg)

![fd00b92aa6965fb5584eab600cf9f44afa5d412e243288783c3e38936936228c.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/images/fd00b92aa6965fb5584eab600cf9f44afa5d412e243288783c3e38936936228c.jpg)

### Tables

![034e69151fe81c36cb331935546acac69e0a8b53cfa5e4129869827743c136e3.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/034e69151fe81c36cb331935546acac69e0a8b53cfa5e4129869827743c136e3.jpg)

![17148a30851a47b149cceab14ad528f020c15ed7f4f48b87136a4e108dd48977.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/17148a30851a47b149cceab14ad528f020c15ed7f4f48b87136a4e108dd48977.jpg)

![1dda62b7a180c98a4de7f9a7106e374e3671930c64912cd36845ed12d1af0bf1.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/1dda62b7a180c98a4de7f9a7106e374e3671930c64912cd36845ed12d1af0bf1.jpg)

![8f6ac38db9ffa7e0fa6d14b6628c69c329968e09d560a470e7c9868397fb18e0.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/8f6ac38db9ffa7e0fa6d14b6628c69c329968e09d560a470e7c9868397fb18e0.jpg)

![9264ab40fbc8734c976ce8869b6481780e5fdebb53a0457309538f04ee331bd9.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/9264ab40fbc8734c976ce8869b6481780e5fdebb53a0457309538f04ee331bd9.jpg)

![92972cf32207ef08721445e1042b89c39bb197d0e09a22d084f25b99d40ad078.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/92972cf32207ef08721445e1042b89c39bb197d0e09a22d084f25b99d40ad078.jpg)

![9600f01b61903d89822001926811de78fd410198e6e5d273f04efd9eacf50905.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/9600f01b61903d89822001926811de78fd410198e6e5d273f04efd9eacf50905.jpg)

![a68707c6d6a7ec6b777bf2300c2f16358df1eeed43c8cb729a1a7426643f1557.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/a68707c6d6a7ec6b777bf2300c2f16358df1eeed43c8cb729a1a7426643f1557.jpg)

![c41f0730a18ad7ba40bab01b677786cf81291e5eb46e6d2399dc6d9b5a0f9468.jpg](../icml_results/3094_TGV-KV_%20Text-Grounded%20KV%20Eviction%20for%20Vision-Language%20Models/tables/c41f0730a18ad7ba40bab01b677786cf81291e5eb46e6d2399dc6d9b5a0f9468.jpg)

<a id="paper-3056"></a>

## GeoSense: Internalizing Geometric Necessity Perception for Multimodal Reasoning


### Images

![36c16ba6bba23ef87f6a1f7cb96744d439dc80dbcee12778c939b7455777dd2c.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/images/36c16ba6bba23ef87f6a1f7cb96744d439dc80dbcee12778c939b7455777dd2c.jpg)

![480efbf430cacf475852ba47b1b23a6c94c761c2feb0703d1c1170d2c94cda50.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/images/480efbf430cacf475852ba47b1b23a6c94c761c2feb0703d1c1170d2c94cda50.jpg)

![6a150b91ce6221f9897a344b4dd236a62963bb50d500d2ea1c96d98ecbf33cbd.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/images/6a150b91ce6221f9897a344b4dd236a62963bb50d500d2ea1c96d98ecbf33cbd.jpg)

![7223af60485b84e5073b5da3e4fe17ea2abc5b45169323504c1a890cbc653253.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/images/7223af60485b84e5073b5da3e4fe17ea2abc5b45169323504c1a890cbc653253.jpg)

![e8b1c1e928a5820dcad98f3a307e2e445274276006b17d25de6138702aadb887.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/images/e8b1c1e928a5820dcad98f3a307e2e445274276006b17d25de6138702aadb887.jpg)

### Tables

![1e24004e8f962c961ec5b79b3409f619ae0bd77890ea5527ac9b22decc90a6a5.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/tables/1e24004e8f962c961ec5b79b3409f619ae0bd77890ea5527ac9b22decc90a6a5.jpg)

![2de725229f870a10bfc12007508cd3a960d0959fa89a1736585f523c7083721e.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/tables/2de725229f870a10bfc12007508cd3a960d0959fa89a1736585f523c7083721e.jpg)

![2f50161fe4afff37ddc3aa9fc3246ece4e97c796784e39ade7d3ce03f2fe1565.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/tables/2f50161fe4afff37ddc3aa9fc3246ece4e97c796784e39ade7d3ce03f2fe1565.jpg)

![68d5ceca0201948d73cd3566dee4f2d4df5468b387acc1cbb90e1def18036e64.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/tables/68d5ceca0201948d73cd3566dee4f2d4df5468b387acc1cbb90e1def18036e64.jpg)

![83c2d96c6800a8bdf642f047b9da1fc5d113a12c2842252b71e00374dd87e280.jpg](../icml_results/3095_GeoSense_%20Internalizing%20Geometric%20Necessity%20Perception%20for%20Multimodal%20Reasoning/tables/83c2d96c6800a8bdf642f047b9da1fc5d113a12c2842252b71e00374dd87e280.jpg)

<a id="paper-3057"></a>

## $R^3$DAO: Reactive Recovery and Reconstruction for Long-horizon Data Agent Orchestration


### Images

![7163d894e2614af46ee3560a2b001a5804a91559c176cf8e5f7c6aba45def7a6.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/images/7163d894e2614af46ee3560a2b001a5804a91559c176cf8e5f7c6aba45def7a6.jpg)

![9749fb1eb3d175579328f721a0577dd456965f5e9c9912501a8930ff0fb3c511.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/images/9749fb1eb3d175579328f721a0577dd456965f5e9c9912501a8930ff0fb3c511.jpg)

![bb233eff8d5d42cc464f8a973bc7a75651b1b102e93057e647ecc4d189fcc032.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/images/bb233eff8d5d42cc464f8a973bc7a75651b1b102e93057e647ecc4d189fcc032.jpg)

![d192cbe1397685bb343d904b53a3928d7ab86ef8183bb9d696cd1ff1f85550ff.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/images/d192cbe1397685bb343d904b53a3928d7ab86ef8183bb9d696cd1ff1f85550ff.jpg)

![eccf5e44bff26567affab7c741133cb63a6b172cfb2335c697fa46347c5fbddb.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/images/eccf5e44bff26567affab7c741133cb63a6b172cfb2335c697fa46347c5fbddb.jpg)

### Tables

![3311ddf25867e48b5598b06a466028345c201b8197596478d26dcc56beb1c459.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/3311ddf25867e48b5598b06a466028345c201b8197596478d26dcc56beb1c459.jpg)

![5518c673b8fc54c32cbbf9c851069476575c4566afb8304680bcaae3d319c0ad.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/5518c673b8fc54c32cbbf9c851069476575c4566afb8304680bcaae3d319c0ad.jpg)

![66eb89c15b19846a95be71b36927399d499762e698105e11b7921b702ff63288.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/66eb89c15b19846a95be71b36927399d499762e698105e11b7921b702ff63288.jpg)

![97c38d82916e8076d5bbeef7fa2c063f372231264a4fefb9d5a02ff35b15cd0b.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/97c38d82916e8076d5bbeef7fa2c063f372231264a4fefb9d5a02ff35b15cd0b.jpg)

![a3d67377920a1a1f64e67a694302b45bf8b35b0753c2a05bb59739921481c545.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/a3d67377920a1a1f64e67a694302b45bf8b35b0753c2a05bb59739921481c545.jpg)

![c908c7a28f989c1ee1440d8390e55f9d730130cfb0b6b82e87a61f6ecc8ec7ef.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/c908c7a28f989c1ee1440d8390e55f9d730130cfb0b6b82e87a61f6ecc8ec7ef.jpg)

![ce5a9fb54634fbedb4777c26ca2650b8acaa772d511ce6e65efb46600b4154f9.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/ce5a9fb54634fbedb4777c26ca2650b8acaa772d511ce6e65efb46600b4154f9.jpg)

![df7742d68f6dd7154394b3fcb81fec6cadaa6c6a04145da66cf8c2f8e03b0f0f.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/df7742d68f6dd7154394b3fcb81fec6cadaa6c6a04145da66cf8c2f8e03b0f0f.jpg)

![fba6a80bb05e68522a0bd07ada6c0633fbc2b20f239a0202be748d31820fd165.jpg](../icml_results/3096_%24R%5E3%24DAO_%20Reactive%20Recovery%20and%20Reconstruction%20for%20Long-horizon%20Data%20Agent%20Orchestration/tables/fba6a80bb05e68522a0bd07ada6c0633fbc2b20f239a0202be748d31820fd165.jpg)

<a id="paper-3058"></a>

## Unsupervised Camouflaged Object Detection with Dual-Eigenvector Spectral Pseudo-Labeling and Contrastive Refinement


### Images

![11667932a5bff40893f667426d654ecd168eb71fd395ca441d33689643b7b37d.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/11667932a5bff40893f667426d654ecd168eb71fd395ca441d33689643b7b37d.jpg)

![43e5094af56fd081352ac87c5671a46f9fe11433cfbb8603f612b661b18ad58f.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/43e5094af56fd081352ac87c5671a46f9fe11433cfbb8603f612b661b18ad58f.jpg)

![7baff521626de9f42c06aa88bbccbf4ce369721fbae96b6ef9cfd7d5958f28c5.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/7baff521626de9f42c06aa88bbccbf4ce369721fbae96b6ef9cfd7d5958f28c5.jpg)

![7f9bf1e15e6049453e6b25401ae8f7e867f50030af36aca48fff068df391b765.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/7f9bf1e15e6049453e6b25401ae8f7e867f50030af36aca48fff068df391b765.jpg)

![c465385e1bac0c8620454c28a090591a0bcac87719ae3d6cc208a331932e1304.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/c465385e1bac0c8620454c28a090591a0bcac87719ae3d6cc208a331932e1304.jpg)

![d049ad29c07989c1e08a4db5818a5dcd9cc5f0fe84e274c51d270ec1dc39bf6a.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/d049ad29c07989c1e08a4db5818a5dcd9cc5f0fe84e274c51d270ec1dc39bf6a.jpg)

![e3f3e3f1613e7847a3be35c100808dff62971b4a609842b71a690be49479feff.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/e3f3e3f1613e7847a3be35c100808dff62971b4a609842b71a690be49479feff.jpg)

![ff381f367139802e6e9ddd5bab555e537989a3124f169bd7bbdb9d9e9ef8c0ac.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/images/ff381f367139802e6e9ddd5bab555e537989a3124f169bd7bbdb9d9e9ef8c0ac.jpg)

### Tables

![0827537db92f9085af9ebd92ed5f293d4be2d9c38c24397d370d5d4eef3077d7.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/0827537db92f9085af9ebd92ed5f293d4be2d9c38c24397d370d5d4eef3077d7.jpg)

![104e182406d8c5daf5cdf57ba448d0f79148c585e901f8bd3109ab763a8413fe.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/104e182406d8c5daf5cdf57ba448d0f79148c585e901f8bd3109ab763a8413fe.jpg)

![268482d1a53d6b74a92d1723f15c9735ce2b84e99bc434ec42a1dcc7c1178f1c.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/268482d1a53d6b74a92d1723f15c9735ce2b84e99bc434ec42a1dcc7c1178f1c.jpg)

![30bd486959bfff9ded5035689cb28c7f4a9e66b7dea9c3c7acbdcd4f4b998f76.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/30bd486959bfff9ded5035689cb28c7f4a9e66b7dea9c3c7acbdcd4f4b998f76.jpg)

![58acc264458172c2c82e044a0024acaefabc3f4a2389737c43d77a712834f152.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/58acc264458172c2c82e044a0024acaefabc3f4a2389737c43d77a712834f152.jpg)

![5eceec251056b1f8de77f8bfb2216c74d7917fd6f2c2d3b8af7a07ff3ad343eb.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/5eceec251056b1f8de77f8bfb2216c74d7917fd6f2c2d3b8af7a07ff3ad343eb.jpg)

![a50e390c4eac26d03b80e13642cbf2358bb79b58161a79adf68b71a338227456.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/a50e390c4eac26d03b80e13642cbf2358bb79b58161a79adf68b71a338227456.jpg)

![a6f03b70f0dc8b8d9488f20549edbf26eeeed93495e9a61cdc2c1e232c7a0db2.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/a6f03b70f0dc8b8d9488f20549edbf26eeeed93495e9a61cdc2c1e232c7a0db2.jpg)

![b1495b75e1b3d95efd0e18be355f74565bcb7f884322198034e311b12a058b08.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/b1495b75e1b3d95efd0e18be355f74565bcb7f884322198034e311b12a058b08.jpg)

![b1e8fda230c0d846e5df577e43e30ff7590d7e589d583cba0565c9e30bbb6e03.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/b1e8fda230c0d846e5df577e43e30ff7590d7e589d583cba0565c9e30bbb6e03.jpg)

![b73e6e0093d6bf84274b9cf5f9581df0338d386ca01ca4371dec9acec0391e2b.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/b73e6e0093d6bf84274b9cf5f9581df0338d386ca01ca4371dec9acec0391e2b.jpg)

![b815bf812393ed923c8f763f51ae2ebed07c63c3eea4fd43b8bf00150cf48d29.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/b815bf812393ed923c8f763f51ae2ebed07c63c3eea4fd43b8bf00150cf48d29.jpg)

![c4f25a35e8fa82a75eae8b6e94deda5cd7f0b854ec8f8e74f12946b4a8e4d77c.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/c4f25a35e8fa82a75eae8b6e94deda5cd7f0b854ec8f8e74f12946b4a8e4d77c.jpg)

![e88ae84b168e751113239562f1816f7371d206316c7ac3ca40b521cec1d8d9df.jpg](../icml_results/3097_Unsupervised%20Camouflaged%20Object%20Detection%20with%20Dual-Eigenvector%20Spectral%20Pseudo-Labeling%20and%20Contras/tables/e88ae84b168e751113239562f1816f7371d206316c7ac3ca40b521cec1d8d9df.jpg)

<a id="paper-3059"></a>

## Beyond Additive Decompositions: Interpretability Through Separability


### Images

![0eadf58d5900d495856713f22970bd5073112faaf0e004473bfbff8c7f0e70e2.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/0eadf58d5900d495856713f22970bd5073112faaf0e004473bfbff8c7f0e70e2.jpg)

![1090855f8ff428d728401bb3ef1968d415dbd030da428b9f213b6b224dfec67f.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/1090855f8ff428d728401bb3ef1968d415dbd030da428b9f213b6b224dfec67f.jpg)

![199339ae39c5a88046f4f1713c8c2ebadace2e48d5e795401a771f5bc0687d97.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/199339ae39c5a88046f4f1713c8c2ebadace2e48d5e795401a771f5bc0687d97.jpg)

![31e8d744bacb120ec6c6a14528caf671a73905626ff711df6650bcf8d9a246c6.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/31e8d744bacb120ec6c6a14528caf671a73905626ff711df6650bcf8d9a246c6.jpg)

![3633937dde24845de3be582e90c2829b8612b8f8a5ac82f63f5c581ef8a2e7c0.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/3633937dde24845de3be582e90c2829b8612b8f8a5ac82f63f5c581ef8a2e7c0.jpg)

![40cb706a9f3ac591518ca99e1219cfff42225e5c57fa94a123df5fcfc1641d6f.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/40cb706a9f3ac591518ca99e1219cfff42225e5c57fa94a123df5fcfc1641d6f.jpg)

![55ef8e32d46d1055b20772681f1aa42672f1cdc552e6e53503e786f48b789846.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/55ef8e32d46d1055b20772681f1aa42672f1cdc552e6e53503e786f48b789846.jpg)

![5637371b67b3413775c4c99e8ff5f8e8bdab91b03c788b93868cebfa82202505.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/5637371b67b3413775c4c99e8ff5f8e8bdab91b03c788b93868cebfa82202505.jpg)

![574940104ac741d88f2a949c11573dae919a02663ec2dead180360658c48ca32.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/574940104ac741d88f2a949c11573dae919a02663ec2dead180360658c48ca32.jpg)

![641b275fc4a832f7af11baef0b84e3e0fa0315b8d322ab0b4c992a97afa623e3.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/641b275fc4a832f7af11baef0b84e3e0fa0315b8d322ab0b4c992a97afa623e3.jpg)

![64f87dfe18eec18d1281404624677b0c0a5a3658e3ac423e6fd29e3053e976ca.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/64f87dfe18eec18d1281404624677b0c0a5a3658e3ac423e6fd29e3053e976ca.jpg)

![7b461e4ed96faa3d8bbb361b20bfff76e653d373e10bd0ef8ece6d4c955e82e0.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/7b461e4ed96faa3d8bbb361b20bfff76e653d373e10bd0ef8ece6d4c955e82e0.jpg)

![7fa8cc63bc6bbd1ee67a16b738c8419642af55d0d82a013836a7737f98aeccd5.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/7fa8cc63bc6bbd1ee67a16b738c8419642af55d0d82a013836a7737f98aeccd5.jpg)

![8a823936b5b4a80f8134c1275a12ada8a44ace37cac203c32175e30770572a46.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/8a823936b5b4a80f8134c1275a12ada8a44ace37cac203c32175e30770572a46.jpg)

![cd3c8eff512cd92339b91d9d39dbac26c55c1e21a633c07c054a6c32787d3c89.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/cd3c8eff512cd92339b91d9d39dbac26c55c1e21a633c07c054a6c32787d3c89.jpg)

![d0dc1d3d1c2e113ad0d83baf82855982a259554135c1c9d09d4693ea731b4cb2.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/d0dc1d3d1c2e113ad0d83baf82855982a259554135c1c9d09d4693ea731b4cb2.jpg)

![f44e70e8421612933c831eaf18bf8624e0b02ada55be5e8828a90c7da2310208.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/images/f44e70e8421612933c831eaf18bf8624e0b02ada55be5e8828a90c7da2310208.jpg)

### Tables

![1839618b88aa1c2f443be08c8566dc787bd21b961988c9d91c4b0f8e56d34664.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/1839618b88aa1c2f443be08c8566dc787bd21b961988c9d91c4b0f8e56d34664.jpg)

![3817f1d9c7f249a8ab5ad34a7ce8b4f1ef0fcde15479e4b5253c0a8513c787f2.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/3817f1d9c7f249a8ab5ad34a7ce8b4f1ef0fcde15479e4b5253c0a8513c787f2.jpg)

![4983e9bfd1953f4406515fde11c7387cd1d86a075dd916ddd1421b3544106e25.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/4983e9bfd1953f4406515fde11c7387cd1d86a075dd916ddd1421b3544106e25.jpg)

![5c679425daea537ebd6954ef3904e3e5d8ed328f9a41e7a8091cb7c9adbe9069.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/5c679425daea537ebd6954ef3904e3e5d8ed328f9a41e7a8091cb7c9adbe9069.jpg)

![77b8b290502c0e8162b82b0b100881916b2fd7b54fded68ee596bb7db763496b.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/77b8b290502c0e8162b82b0b100881916b2fd7b54fded68ee596bb7db763496b.jpg)

![7c60f5fb02978cb2c311e06a78d911d7367bb9195a5daf101ec33a0fe4fedd14.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/7c60f5fb02978cb2c311e06a78d911d7367bb9195a5daf101ec33a0fe4fedd14.jpg)

![82b8affcb9cbc2541c79e0a4d698df803cd08769983d73c0f9c193727dbb152b.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/82b8affcb9cbc2541c79e0a4d698df803cd08769983d73c0f9c193727dbb152b.jpg)

![9b2f6d9a7d63ff2effb834dc9ec0d2515285068029c9f097a0a98a6c01272193.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/9b2f6d9a7d63ff2effb834dc9ec0d2515285068029c9f097a0a98a6c01272193.jpg)

![a10a8129434268331180348933ca262889ee88cb23542d5fe7ac0b7dc04d3a9d.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/a10a8129434268331180348933ca262889ee88cb23542d5fe7ac0b7dc04d3a9d.jpg)

![e30f83e1513fb88802517b18b3721a0dbbc9d06d7a181a97349f80645b693fc0.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/e30f83e1513fb88802517b18b3721a0dbbc9d06d7a181a97349f80645b693fc0.jpg)

![ed1a3451b268a31ac728cf5c29b26e421a01497a4d671fd72d80c1b61261a2da.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/ed1a3451b268a31ac728cf5c29b26e421a01497a4d671fd72d80c1b61261a2da.jpg)

![ef13c09731365c22edf7f9e3eff0d979785f014f0211e32928273df18a5ccd06.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/ef13c09731365c22edf7f9e3eff0d979785f014f0211e32928273df18a5ccd06.jpg)

![fc7a59e4f3298433b6958e9883494dad625ae7b9c546c18a5fd981a7a0d57c40.jpg](../icml_results/3098_Beyond%20Additive%20Decompositions_%20Interpretability%20Through%20Separability/tables/fc7a59e4f3298433b6958e9883494dad625ae7b9c546c18a5fd981a7a0d57c40.jpg)

<a id="paper-3060"></a>

## From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing


### Images

![251590350db492c488d60e29152c6cc929315f40083f28991ed99e12876635d1.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/images/251590350db492c488d60e29152c6cc929315f40083f28991ed99e12876635d1.jpg)

![2cfc529c802bc0f1ef4968396f2f95bd5fcfb91d2fa08a72662f22b2712ea760.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/images/2cfc529c802bc0f1ef4968396f2f95bd5fcfb91d2fa08a72662f22b2712ea760.jpg)

![65b344b090f934fdc5da0f3a902b50938a68f38d66bcb26e887c2cd04916104c.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/images/65b344b090f934fdc5da0f3a902b50938a68f38d66bcb26e887c2cd04916104c.jpg)

### Tables

![158c220308a89877822e3f9c1ca86603fec763c15fc9f3f10c1c81067b132528.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/158c220308a89877822e3f9c1ca86603fec763c15fc9f3f10c1c81067b132528.jpg)

![4b220bb462452c85b9a7dee97d849b67ddc530042531efe2b5ce1eff05a08c1a.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/4b220bb462452c85b9a7dee97d849b67ddc530042531efe2b5ce1eff05a08c1a.jpg)

![6adfbf5641bc3bd4316e46823b474ec7904e911f55f9d5412bc9ab2d3be81dc9.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/6adfbf5641bc3bd4316e46823b474ec7904e911f55f9d5412bc9ab2d3be81dc9.jpg)

![7e45559a0f32e493ccac8d064c167bda75326c4801da80e8a2ba54a66533c91a.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/7e45559a0f32e493ccac8d064c167bda75326c4801da80e8a2ba54a66533c91a.jpg)

![a1aac858174b17c4da0c310d9a960f19dabac306844946f434062dd8eacb87b0.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/a1aac858174b17c4da0c310d9a960f19dabac306844946f434062dd8eacb87b0.jpg)

![b727663afef9143460cb37eb1817e5114723f6c56138394e3d40f2f72f9acda2.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/b727663afef9143460cb37eb1817e5114723f6c56138394e3d40f2f72f9acda2.jpg)

![b7ddaad873687b000cceb405259f1c4579ae2ae5031fef2b55fff2efcfff1836.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/b7ddaad873687b000cceb405259f1c4579ae2ae5031fef2b55fff2efcfff1836.jpg)

![cf8cb63189d26c31b8554d9e91cb3a23b2d8dc057ae0004f70efdbb2b372a66c.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/cf8cb63189d26c31b8554d9e91cb3a23b2d8dc057ae0004f70efdbb2b372a66c.jpg)

![d1d7ce9b61c78e94cd81fa5423b16d4b19b1d5428fa861d6528129290ebe534f.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/d1d7ce9b61c78e94cd81fa5423b16d4b19b1d5428fa861d6528129290ebe534f.jpg)

![d4811523f03d84d75f3da10cfdbe54cbb5c6ad623f74314498d49c6cd95431d6.jpg](../icml_results/3099_From%20Backward%20Spreading%20to%20Forward%20Replay_%20Revisiting%20Target%20Construction%20in%20LLM%20Parameter%20Editing/tables/d4811523f03d84d75f3da10cfdbe54cbb5c6ad623f74314498d49c6cd95431d6.jpg)

<a id="paper-3061"></a>

## Dual-stage Contrastive Learning-enhanced Multi-view Variational Clustering


### Images

![13a12756db64ca65edd478f210f7d706adb85594d9d0763767a87855666dea41.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/13a12756db64ca65edd478f210f7d706adb85594d9d0763767a87855666dea41.jpg)

![27e963cb162c11f3bf3c332b83ef3f546709a9f5be391895fc0ab0df96afa60c.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/27e963cb162c11f3bf3c332b83ef3f546709a9f5be391895fc0ab0df96afa60c.jpg)

![2d92e309a825a4528f501c5041df943b958457c97c8375bab8d2c86f293b2233.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/2d92e309a825a4528f501c5041df943b958457c97c8375bab8d2c86f293b2233.jpg)

![5a5bdc7e81b9b90c694d73ccb1c98be13822095f7ab55c0cb16086490f41341a.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/5a5bdc7e81b9b90c694d73ccb1c98be13822095f7ab55c0cb16086490f41341a.jpg)

![5ab9d4440fbe4d0b1fee096d6cc60cbb4b6070d0e29d25081dd606838d122795.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/5ab9d4440fbe4d0b1fee096d6cc60cbb4b6070d0e29d25081dd606838d122795.jpg)

![5c79496bfd9a45504aa12ccc89c16ab891dc417e258193d17c4bf00ebfed54f9.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/5c79496bfd9a45504aa12ccc89c16ab891dc417e258193d17c4bf00ebfed54f9.jpg)

![5ff249b027c137df4d3b3784071ab772247ca6c37871b102080e6971cfccdfec.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/5ff249b027c137df4d3b3784071ab772247ca6c37871b102080e6971cfccdfec.jpg)

![6377907f58d6793af879722c9ae8338f1a4eb6b56af3594ff9c49c363de3381c.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/6377907f58d6793af879722c9ae8338f1a4eb6b56af3594ff9c49c363de3381c.jpg)

![d11ddcf92d43d3a7e0dd6fbca1b94bdcc159e662ea708a560b0728e5e4ed5cc6.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/d11ddcf92d43d3a7e0dd6fbca1b94bdcc159e662ea708a560b0728e5e4ed5cc6.jpg)

![dd5e09027bea702aaad822b9f9916e82d58c7aca3d66d1adb2d49670a9aeab3d.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/dd5e09027bea702aaad822b9f9916e82d58c7aca3d66d1adb2d49670a9aeab3d.jpg)

![faca34e99a2ed0b1f065d6352743190a9b4e82490db1d3b9a39769876539f088.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/faca34e99a2ed0b1f065d6352743190a9b4e82490db1d3b9a39769876539f088.jpg)

![fd4cf72ce1d5b99aa16eed170c5f20c2dddb798216b247792761d07a18610930.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/images/fd4cf72ce1d5b99aa16eed170c5f20c2dddb798216b247792761d07a18610930.jpg)

### Tables

![090ee870ab5efd67b208b25c4178681c10a6a167271e8282447321fb0d3d40b4.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/090ee870ab5efd67b208b25c4178681c10a6a167271e8282447321fb0d3d40b4.jpg)

![158e374294c5f3e8064bc676e5bc51cd1d53d258e5f7331be4e76a291b63afe0.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/158e374294c5f3e8064bc676e5bc51cd1d53d258e5f7331be4e76a291b63afe0.jpg)

![167940b6300abc01fd931b2ee0fbf9b7071d0096c695fc6ea21a8217323b85e8.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/167940b6300abc01fd931b2ee0fbf9b7071d0096c695fc6ea21a8217323b85e8.jpg)

![266da1070a1266ba1a3fe2e3a9ea93456e5f626a6611291e64da6ec43d06e516.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/266da1070a1266ba1a3fe2e3a9ea93456e5f626a6611291e64da6ec43d06e516.jpg)

![97425b33f8e558c36450bcbd470658df7ed6bff9733e373f2e8ed1aaec552f7e.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/97425b33f8e558c36450bcbd470658df7ed6bff9733e373f2e8ed1aaec552f7e.jpg)

![b2ce43ac1940f997110a9ddddc132669f4aa4adf9eb9b68ab4cecafd404e6c78.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/b2ce43ac1940f997110a9ddddc132669f4aa4adf9eb9b68ab4cecafd404e6c78.jpg)

![b5e33ca0dafbca88bd61e24b6cc68e15a40b4deb75be1ecacdd0f39db383b328.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/b5e33ca0dafbca88bd61e24b6cc68e15a40b4deb75be1ecacdd0f39db383b328.jpg)

![e29bc367b70f6d8b95813de586b2d7f579706ad4fa30cfab915b9328037d687d.jpg](../icml_results/3100_Dual-stage%20Contrastive%20Learning-enhanced%20Multi-view%20Variational%20Clustering/tables/e29bc367b70f6d8b95813de586b2d7f579706ad4fa30cfab915b9328037d687d.jpg)

<a id="paper-3062"></a>

## CoPE: A Framework for Optimizing Coordination between Planning and Execution in LLM-based Agents


### Images

![111c94cfc348595d054bff2908415a7459232392f8cdf6f3398eb65976ebed59.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/111c94cfc348595d054bff2908415a7459232392f8cdf6f3398eb65976ebed59.jpg)

![35219cc70b5397611acd184bc83f95c1e7c4737ac36e312985a199a4e6421034.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/35219cc70b5397611acd184bc83f95c1e7c4737ac36e312985a199a4e6421034.jpg)

![3775f64cb4a081a6d2008aee2c97cf3365a8f6781c5246606dec1311ecc96d87.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/3775f64cb4a081a6d2008aee2c97cf3365a8f6781c5246606dec1311ecc96d87.jpg)

![4071f1f67c48bb9251b7248160e4fee46b30881edfffbb93709e70c4ea094189.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/4071f1f67c48bb9251b7248160e4fee46b30881edfffbb93709e70c4ea094189.jpg)

![52e7363beba80932125e14475bbedc3e68fd39baab2e1fc4da5a40f9f57c31ed.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/52e7363beba80932125e14475bbedc3e68fd39baab2e1fc4da5a40f9f57c31ed.jpg)

![8ee6a2a5d3ab686280c58ce0d185a6b054cdba193ed621cc4dd0abd1180adc06.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/8ee6a2a5d3ab686280c58ce0d185a6b054cdba193ed621cc4dd0abd1180adc06.jpg)

![9b5dbe79cbcb66caea8600702a0ae7f0a116b8a744246c9f90fd8a4e667fcd16.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/9b5dbe79cbcb66caea8600702a0ae7f0a116b8a744246c9f90fd8a4e667fcd16.jpg)

![acf69699472b3f0a5eb4ff6ca4b34f411e95b2dfec378e4274fb9e9c5205ada1.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/acf69699472b3f0a5eb4ff6ca4b34f411e95b2dfec378e4274fb9e9c5205ada1.jpg)

![c3f71ca601d469a70dcc5852ef1cda09a2dd2c732021f0392dca2caca9960870.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/c3f71ca601d469a70dcc5852ef1cda09a2dd2c732021f0392dca2caca9960870.jpg)

![d5c672f11a6f974758c3a16fefa52897b79e5b8e5b12fa5228367b6113289fad.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/images/d5c672f11a6f974758c3a16fefa52897b79e5b8e5b12fa5228367b6113289fad.jpg)

### Tables

![0390dccd5260e5f7439b0113714e1303a00572ef94a0ba09ef52ad0ea5ff24da.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/0390dccd5260e5f7439b0113714e1303a00572ef94a0ba09ef52ad0ea5ff24da.jpg)

![0b83ef957b54503576a7f57b6007068b88f5fb91a03ba3dfa7f1fe77557c9e39.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/0b83ef957b54503576a7f57b6007068b88f5fb91a03ba3dfa7f1fe77557c9e39.jpg)

![1b52a759c41768fdc93fa5a3f1c794101c24ffe8682816686c2b005b2fe1e4f8.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/1b52a759c41768fdc93fa5a3f1c794101c24ffe8682816686c2b005b2fe1e4f8.jpg)

![1c534eec8b16e3ff0fb0c60dd19cf99b7f11f6a17517c1ea2f3008401f1dd801.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/1c534eec8b16e3ff0fb0c60dd19cf99b7f11f6a17517c1ea2f3008401f1dd801.jpg)

![564c9f9114369dd36ed623720bd0a67920338e0b984c141295cb06720a95c266.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/564c9f9114369dd36ed623720bd0a67920338e0b984c141295cb06720a95c266.jpg)

![69f52382ccc9ad05e02e976e1dcb2a8faca94a22122f979165d35b0769b8cabf.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/69f52382ccc9ad05e02e976e1dcb2a8faca94a22122f979165d35b0769b8cabf.jpg)

![7892b97ea8eabb30d0d3e4e05071fd77964746733ab6749d0161941ee2363fe7.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/7892b97ea8eabb30d0d3e4e05071fd77964746733ab6749d0161941ee2363fe7.jpg)

![97148c66f3e155a306ba76c5e0dfdbffd67d6afbfb36ea6d2e2d1baf566d038f.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/97148c66f3e155a306ba76c5e0dfdbffd67d6afbfb36ea6d2e2d1baf566d038f.jpg)

![a32a9e923a4f306bde24e08dd569798d0ace5e0e5772eb5206698047892805d0.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/a32a9e923a4f306bde24e08dd569798d0ace5e0e5772eb5206698047892805d0.jpg)

![aaddd14330207c4e016b00d04f894c99eb9f6265a3666456c7cc5532cd98aae7.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/aaddd14330207c4e016b00d04f894c99eb9f6265a3666456c7cc5532cd98aae7.jpg)

![ad4f2c86f0b5e49587acfbdf938fecf88dcc5290bc9419ac3c65f72dad434cfd.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/ad4f2c86f0b5e49587acfbdf938fecf88dcc5290bc9419ac3c65f72dad434cfd.jpg)

![d0123741d0eab392339089fb7049244bff102bef9798e21b3d654b50b75c5bfe.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/d0123741d0eab392339089fb7049244bff102bef9798e21b3d654b50b75c5bfe.jpg)

![de69d8ed169c9e792a420a30caba5ab0d4f614638af716c4909a3df471c67422.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/de69d8ed169c9e792a420a30caba5ab0d4f614638af716c4909a3df471c67422.jpg)

![f70cd9b8398582b881bc9ca29a0688d7009b5165b1c228c2349655bbd783d973.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/f70cd9b8398582b881bc9ca29a0688d7009b5165b1c228c2349655bbd783d973.jpg)

![feed35420b8fe5cae9811df00ff2d16daf50977531c48025b6dbf835c20c28e8.jpg](../icml_results/3101_CoPE_%20A%20Framework%20for%20Optimizing%20Coordination%20between%20Planning%20and%20Execution%20in%20LLM-based%20Agents/tables/feed35420b8fe5cae9811df00ff2d16daf50977531c48025b6dbf835c20c28e8.jpg)

<a id="paper-3063"></a>

## E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs


### Images

![1df8cad3a7da0894730d1ed6900d4142787a1a43e1afbd99123bd36e5053ad63.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/1df8cad3a7da0894730d1ed6900d4142787a1a43e1afbd99123bd36e5053ad63.jpg)

![5060177a4bb08413fa30aeca8c540a2caafb4d9924424e717f51ddfea2a246aa.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/5060177a4bb08413fa30aeca8c540a2caafb4d9924424e717f51ddfea2a246aa.jpg)

![53875399ba2ef37dc9343e74f9740961b2774e4ad3224ac1dc9918cc4ec3ab12.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/53875399ba2ef37dc9343e74f9740961b2774e4ad3224ac1dc9918cc4ec3ab12.jpg)

![5ce7ddbdc017f2025df455da36fecfa723ba7f90503986cecd1a9b481591bdd5.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/5ce7ddbdc017f2025df455da36fecfa723ba7f90503986cecd1a9b481591bdd5.jpg)

![81445842b667dec05642ec19e98b521eb900d2a57be8c0e333bee70f1963ae92.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/81445842b667dec05642ec19e98b521eb900d2a57be8c0e333bee70f1963ae92.jpg)

![b0ccbca9a1d2e6acea871d79ffab8e6ace1aeab970937b4cc5c79ffb520ff03b.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/b0ccbca9a1d2e6acea871d79ffab8e6ace1aeab970937b4cc5c79ffb520ff03b.jpg)

![f58fd7053e5876eee703fdb19db6cba61dcea3238cabe1e47b50d092920a4137.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/images/f58fd7053e5876eee703fdb19db6cba61dcea3238cabe1e47b50d092920a4137.jpg)

### Tables

![2b4661ee2436bbac5e4dc3e94b3cf5bf2e0cbec6ab620dd0b6d1b9ebfb070133.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/tables/2b4661ee2436bbac5e4dc3e94b3cf5bf2e0cbec6ab620dd0b6d1b9ebfb070133.jpg)

![449448a6c6edd764125886587ee6976e0650d144a0008bba19a6c3afe0a6facb.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/tables/449448a6c6edd764125886587ee6976e0650d144a0008bba19a6c3afe0a6facb.jpg)

![81bb43b3056b6b9b388fff05bb85ba8a959c43a7f0b14c5060eb6aa7d9c8c397.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/tables/81bb43b3056b6b9b388fff05bb85ba8a959c43a7f0b14c5060eb6aa7d9c8c397.jpg)

![90ddbba593e5cb3565faac6940f4705e8aa39ff3b21f6beb9f5aa33cb0f364e4.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/tables/90ddbba593e5cb3565faac6940f4705e8aa39ff3b21f6beb9f5aa33cb0f364e4.jpg)

![d3eb45ab5cf02a77d1eeb3ec8a9619ec455d62df3476c3b2b126e8bb1b81a5fc.jpg](../icml_results/3102_E-VAds_%20An%20E-commerce%20Short%20Videos%20Understanding%20Benchmark%20for%20MLLMs/tables/d3eb45ab5cf02a77d1eeb3ec8a9619ec455d62df3476c3b2b126e8bb1b81a5fc.jpg)

<a id="paper-3064"></a>

## Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers


### Images

![033af937724a52ddaf759af89a92a31117697a8a3a204a76c08924f6d9c7cff3.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/033af937724a52ddaf759af89a92a31117697a8a3a204a76c08924f6d9c7cff3.jpg)

![0fddb8622e09399d92df02796c162c8dceaf67ab7d6d2fdb11c1ec196f0e10a0.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/0fddb8622e09399d92df02796c162c8dceaf67ab7d6d2fdb11c1ec196f0e10a0.jpg)

![2090f0288a198f7a1bf7e2c12e4e2f91d56b2fc064ff3e7ce4ae5376aee2c797.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/2090f0288a198f7a1bf7e2c12e4e2f91d56b2fc064ff3e7ce4ae5376aee2c797.jpg)

![3eb6642ddbe390ac64c9a6b832f0bae2cd5fbcb727b651e874c4866e920d23a1.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/3eb6642ddbe390ac64c9a6b832f0bae2cd5fbcb727b651e874c4866e920d23a1.jpg)

![40ecf143cd019b30893710123b292de18e49e83f9137175f5e43216b68fa2649.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/40ecf143cd019b30893710123b292de18e49e83f9137175f5e43216b68fa2649.jpg)

![51be406c436b8b1d32e86c3b8f2bb0ebb29e208d15fbcb067553ca01dfb62355.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/51be406c436b8b1d32e86c3b8f2bb0ebb29e208d15fbcb067553ca01dfb62355.jpg)

![54aaf27c81008297ea736b4d6772fb603993770fd43ba3aa43c05f7a7600996c.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/54aaf27c81008297ea736b4d6772fb603993770fd43ba3aa43c05f7a7600996c.jpg)

![58b07d4b04d69615e90a6233f79c14be46f1dc64cb105d0aaaf327d99228415f.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/58b07d4b04d69615e90a6233f79c14be46f1dc64cb105d0aaaf327d99228415f.jpg)

![616a4a429842d274c4623356e462f85b8064c935483f86002df0d55a12d9ec1c.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/616a4a429842d274c4623356e462f85b8064c935483f86002df0d55a12d9ec1c.jpg)

![7f516a1a10bde77052f91c8f756b1fb488af71bc3b9307bb939cde56ed0a6670.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/7f516a1a10bde77052f91c8f756b1fb488af71bc3b9307bb939cde56ed0a6670.jpg)

![9381d7a064a6361ab656ce6f5964354e3f78e1ca3aaeae2c1f1faffca2e02cfd.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/9381d7a064a6361ab656ce6f5964354e3f78e1ca3aaeae2c1f1faffca2e02cfd.jpg)

![a358242782b1316299d61f4e82fad4919caa1f97c2ec9753e4c4262e6e724a8a.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/a358242782b1316299d61f4e82fad4919caa1f97c2ec9753e4c4262e6e724a8a.jpg)

![b199264d4c86f5398d95ac1e1663786231c19a86a653a0a6c24529417db659e2.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/b199264d4c86f5398d95ac1e1663786231c19a86a653a0a6c24529417db659e2.jpg)

![cdbdd78eb7708a7b233f083b393902c5e4178e0b244e17831c707ce8feec0220.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/cdbdd78eb7708a7b233f083b393902c5e4178e0b244e17831c707ce8feec0220.jpg)

![cdc8d7ada34bc8ffe8d371fbe42cc775a460c6fa6fbb25add96de2ebc24715ec.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/cdc8d7ada34bc8ffe8d371fbe42cc775a460c6fa6fbb25add96de2ebc24715ec.jpg)

![dff698a49e1573cf94fe5a8ba26928ebc54c89a8e67f2a1e6da4bd3f63deca8c.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/dff698a49e1573cf94fe5a8ba26928ebc54c89a8e67f2a1e6da4bd3f63deca8c.jpg)

![ea95a6a713693946a96127fa7accbf8dcfc24faf166811873ca4655fa0ca5cee.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/ea95a6a713693946a96127fa7accbf8dcfc24faf166811873ca4655fa0ca5cee.jpg)

![eec85671caf2c0bb98437467f6f5ecbebd9850436e39f86fef7856be057459a5.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/eec85671caf2c0bb98437467f6f5ecbebd9850436e39f86fef7856be057459a5.jpg)

![f7302b6a2ed2f396770104ef3f4cc784ff48d52d5dc6207b468516b136109be0.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/f7302b6a2ed2f396770104ef3f4cc784ff48d52d5dc6207b468516b136109be0.jpg)

![f9d3c81d55f8b9a46e30265118ab1d0f75f0bf3ae4b7e02a608886422ad8a20a.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/images/f9d3c81d55f8b9a46e30265118ab1d0f75f0bf3ae4b7e02a608886422ad8a20a.jpg)

### Tables

![060fdef2a4d158bada60b3d26d8dd28996e9e648c6f352a875933628b496f385.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/060fdef2a4d158bada60b3d26d8dd28996e9e648c6f352a875933628b496f385.jpg)

![2c8815b3342ebbcb02247eff4925357e8b04cdd7e09dc8b5e8c5fc1d95570ad0.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/2c8815b3342ebbcb02247eff4925357e8b04cdd7e09dc8b5e8c5fc1d95570ad0.jpg)

![2cf1dccfaf464718d7d5fdf440547cdf5c2eb1f2cf27014be78e111ba6f7579d.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/2cf1dccfaf464718d7d5fdf440547cdf5c2eb1f2cf27014be78e111ba6f7579d.jpg)

![479362eeb6126426d99e35cb2e1038d0313f1e3efa55234c1fd0f24e328cc845.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/479362eeb6126426d99e35cb2e1038d0313f1e3efa55234c1fd0f24e328cc845.jpg)

![a766f6ba53ad3d252c8e91bfc626f8261f935acf0b266f155501ebeb796dda21.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/a766f6ba53ad3d252c8e91bfc626f8261f935acf0b266f155501ebeb796dda21.jpg)

![b14d58445fe45a6b32412d406f331575bf925fc73527c1a3f76f98e95c5c1ef6.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/b14d58445fe45a6b32412d406f331575bf925fc73527c1a3f76f98e95c5c1ef6.jpg)

![b1d25e725f132124ba9b6c6329fd23b6f9ccc9a174b4119920e912e54ae8bff3.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/b1d25e725f132124ba9b6c6329fd23b6f9ccc9a174b4119920e912e54ae8bff3.jpg)

![b43ecae0bcfc199b14de6c3dd90f5330ea15123ea1707f472b5d8d36bd1a588e.jpg](../icml_results/3103_Mixture%20of%20Distributions%20Matters_%20Dynamic%20Sparse%20Attention%20for%20Efficient%20Video%20Diffusion%20Transformer/tables/b43ecae0bcfc199b14de6c3dd90f5330ea15123ea1707f472b5d8d36bd1a588e.jpg)

<a id="paper-3065"></a>

## HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling


### Images

![11e85403f9ff0fb3b3ef683172b497cd4992cdd985b20ec358f5461399780593.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/11e85403f9ff0fb3b3ef683172b497cd4992cdd985b20ec358f5461399780593.jpg)

![22a75715c95a4334ce9832cf75d2decb5976fd267f52ccb5262286178af51581.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/22a75715c95a4334ce9832cf75d2decb5976fd267f52ccb5262286178af51581.jpg)

![2534e16a818f45862375eabbe59ce7ec33397027932745c1212b23446c2fa734.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/2534e16a818f45862375eabbe59ce7ec33397027932745c1212b23446c2fa734.jpg)

![31f000245deb3733846dee82a26ea7c1343b169740c2245f77c5c058e8207121.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/31f000245deb3733846dee82a26ea7c1343b169740c2245f77c5c058e8207121.jpg)

![4b67c18b0b905024e5a6cf744baf28161ba825474e750812d272b0d07f6251c3.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/4b67c18b0b905024e5a6cf744baf28161ba825474e750812d272b0d07f6251c3.jpg)

![64b33124f443873691ad1e5a20edd9471dd553a9a73a364252f842fbcab72d8b.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/64b33124f443873691ad1e5a20edd9471dd553a9a73a364252f842fbcab72d8b.jpg)

![7115ec2b6e65e26a4da54ad466f1d5d61ed9104795ada5291a0502acb50dcbc3.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/7115ec2b6e65e26a4da54ad466f1d5d61ed9104795ada5291a0502acb50dcbc3.jpg)

![72deae8d8ab313aba84ce36edc46a4842ed871bacb26fad36398531f78104ad3.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/72deae8d8ab313aba84ce36edc46a4842ed871bacb26fad36398531f78104ad3.jpg)

![771e0b2f16b3fb3e395e9f8a9a8654ecb4889024d226f8ede4303ff3a39acadb.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/771e0b2f16b3fb3e395e9f8a9a8654ecb4889024d226f8ede4303ff3a39acadb.jpg)

![969cd38bcd93bbae00d9dedff934199af98af22df867998def14b2da24691fed.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/969cd38bcd93bbae00d9dedff934199af98af22df867998def14b2da24691fed.jpg)

![a248bd149cfd6bb6744e07a9cc41099b3fe7896f286791701d8beea32d0c839c.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/a248bd149cfd6bb6744e07a9cc41099b3fe7896f286791701d8beea32d0c839c.jpg)

![a2dbb461f6569b640a98927583fbe52f231d74da23074903923db18e1d678a7c.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/a2dbb461f6569b640a98927583fbe52f231d74da23074903923db18e1d678a7c.jpg)

![e5483e6f7b37eb940ec90b10bcc4882f2a48becc922a152b589f90516b8f830f.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/images/e5483e6f7b37eb940ec90b10bcc4882f2a48becc922a152b589f90516b8f830f.jpg)

### Tables

![0f514b62ec1a42251ad720d8787ec9de3e886baa945f9315e576759df7371578.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/0f514b62ec1a42251ad720d8787ec9de3e886baa945f9315e576759df7371578.jpg)

![1b579fe0ce7f8ca86a21d904039ba7c577245b4c26c65c20d8789aca5f7e4049.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/1b579fe0ce7f8ca86a21d904039ba7c577245b4c26c65c20d8789aca5f7e4049.jpg)

![29c97b1bd1ff5d6493db05109b08ccfb1f04b96cb8f0d9a9b8ecc92245f12c95.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/29c97b1bd1ff5d6493db05109b08ccfb1f04b96cb8f0d9a9b8ecc92245f12c95.jpg)

![353f41f7243a01e3d6f3ee7345c75141135e30e62060aed28d61f0d0bdd223b7.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/353f41f7243a01e3d6f3ee7345c75141135e30e62060aed28d61f0d0bdd223b7.jpg)

![4ee43b08463aacd90b9c7bd786cda2c2c1d2b2d7641e695adf908d984200d169.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/4ee43b08463aacd90b9c7bd786cda2c2c1d2b2d7641e695adf908d984200d169.jpg)

![594755cb827b3a7ff776b47745578a44f6a8bc304c7084c8417927610c2856d2.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/594755cb827b3a7ff776b47745578a44f6a8bc304c7084c8417927610c2856d2.jpg)

![a5bddfd35601b1b9df04165bc0b53f4d2222be3c68497d0a3b5fe54c8c7bf467.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/a5bddfd35601b1b9df04165bc0b53f4d2222be3c68497d0a3b5fe54c8c7bf467.jpg)

![ad84628a38cd2be920a72d54bd8376a6899f848363d949e7e6fa82b9dadaedbf.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/ad84628a38cd2be920a72d54bd8376a6899f848363d949e7e6fa82b9dadaedbf.jpg)

![bb034baf08af6d39fbcb4c4b00bea671c2d2ab5e6ea98f34bb32a6ef06850452.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/bb034baf08af6d39fbcb4c4b00bea671c2d2ab5e6ea98f34bb32a6ef06850452.jpg)

![bfa247a4ff0244589e964a6cd0011de71bc1da34f46345243187445c4f1a7442.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/bfa247a4ff0244589e964a6cd0011de71bc1da34f46345243187445c4f1a7442.jpg)

![d6a4665da1ca793db986c42f4f54cfb787536880f1eba16d6b5b94b6aa8d7657.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/d6a4665da1ca793db986c42f4f54cfb787536880f1eba16d6b5b94b6aa8d7657.jpg)

![eeb3fbfe6a0df946a4b5ce93811ab672d7de32548e4e681d3f5ee74109bf50fb.jpg](../icml_results/3104_HiDe_%20Rethinking%20The%20Zoom-IN%20method%20in%20High%20Resolution%20MLLMs%20via%20Hierarchical%20Decoupling/tables/eeb3fbfe6a0df946a4b5ce93811ab672d7de32548e4e681d3f5ee74109bf50fb.jpg)

<a id="paper-3066"></a>

## AIR: Post-training Data Selection for Reasoning via Attention Head Influence


### Images

![279073d2091c8d0425a0d30c4b9701225da2e8bf05b45385dfb029eaf6551344.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/images/279073d2091c8d0425a0d30c4b9701225da2e8bf05b45385dfb029eaf6551344.jpg)

![60910ff1d825f041d1c10c71b0ee073ae140fec6da13f82a972d7d00856ca27e.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/images/60910ff1d825f041d1c10c71b0ee073ae140fec6da13f82a972d7d00856ca27e.jpg)

![c43c77f001e2867532b952e3c6bfe6a2c5ee08f4cfb4e9f1bf9acbc9d3150498.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/images/c43c77f001e2867532b952e3c6bfe6a2c5ee08f4cfb4e9f1bf9acbc9d3150498.jpg)

### Tables

![165577ab565ffa6a4c62784100040847711d2a662954858c7c1f8d4a03698390.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/165577ab565ffa6a4c62784100040847711d2a662954858c7c1f8d4a03698390.jpg)

![19e99d134bb47e5e060cd86f0f5b504ef910e948ac39b4f25a8a3cb210443b02.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/19e99d134bb47e5e060cd86f0f5b504ef910e948ac39b4f25a8a3cb210443b02.jpg)

![2b6b30ffbf68a5f846822d1ca6b1876dec8ad33ed650fa44ef68971365892832.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/2b6b30ffbf68a5f846822d1ca6b1876dec8ad33ed650fa44ef68971365892832.jpg)

![305d1c122513f79915e498a23e92cb17b23b6e63394d52bc45bca4b3f159eac0.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/305d1c122513f79915e498a23e92cb17b23b6e63394d52bc45bca4b3f159eac0.jpg)

![66b427450908a712ee7b33e62702c00f4ca81fcc9d453ea556e013d7ec0ed39d.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/66b427450908a712ee7b33e62702c00f4ca81fcc9d453ea556e013d7ec0ed39d.jpg)

![79f981107598af8b1b89f3dd1435339e31ec76a3be0fe0e86fc9475711cd6c0d.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/79f981107598af8b1b89f3dd1435339e31ec76a3be0fe0e86fc9475711cd6c0d.jpg)

![7c2d951560ba64d8e271f2c7c6362a62dfb6438682a33efb1c1609a1ef2f1f54.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/7c2d951560ba64d8e271f2c7c6362a62dfb6438682a33efb1c1609a1ef2f1f54.jpg)

![b061155ed84d58a90ff4cca828cdbb7e46ed6dc972a59bcc3dc95ff99265525c.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/b061155ed84d58a90ff4cca828cdbb7e46ed6dc972a59bcc3dc95ff99265525c.jpg)

![b11085c290cb0541abe5c04189abd3efd7a3381d5b2dd231e65145d67ddcc4c4.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/b11085c290cb0541abe5c04189abd3efd7a3381d5b2dd231e65145d67ddcc4c4.jpg)

![cc835d479455a4118419a51b1a0a58e54c24b8d45cf7d23177c4b27db9756eb2.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/cc835d479455a4118419a51b1a0a58e54c24b8d45cf7d23177c4b27db9756eb2.jpg)

![d5edfb91ff15912e19cd50edb2473cacef09cf1e48bf219ec9911a038bf2deaa.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/d5edfb91ff15912e19cd50edb2473cacef09cf1e48bf219ec9911a038bf2deaa.jpg)

![d788803121a2eff7b64056a41faa866b95a4f516f183c88ce68aeb3ecb8aa7b9.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/d788803121a2eff7b64056a41faa866b95a4f516f183c88ce68aeb3ecb8aa7b9.jpg)

![eb1e5921b1fb58f4ac1fbf1098eb56b0a8aafea6fd857f8399964c243f2b2473.jpg](../icml_results/3105_AIR_%20Post-training%20Data%20Selection%20for%20Reasoning%20via%20Attention%20Head%20Influence/tables/eb1e5921b1fb58f4ac1fbf1098eb56b0a8aafea6fd857f8399964c243f2b2473.jpg)

<a id="paper-3067"></a>

## DocOS: Towards Proactive Document-Guided Actions in GUI Agents


### Images

![30d25103303ac6bce64078eedb6d795a53a057a5125e7b8601f653488ee99dba.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/30d25103303ac6bce64078eedb6d795a53a057a5125e7b8601f653488ee99dba.jpg)

![31d7775b672f0f547c15c20c55a806a857dd8b70f78754b92f5ae3719c2c97fb.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/31d7775b672f0f547c15c20c55a806a857dd8b70f78754b92f5ae3719c2c97fb.jpg)

![3db2b6aef4df3cb59764a71a4cb01a6f5053ad26bec9c309bd1ef57b7e92a9d4.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/3db2b6aef4df3cb59764a71a4cb01a6f5053ad26bec9c309bd1ef57b7e92a9d4.jpg)

![46cd524c217d2d1190118c32fad83de23e8ecd0e83f51b85576a9fdc56331295.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/46cd524c217d2d1190118c32fad83de23e8ecd0e83f51b85576a9fdc56331295.jpg)

![4755dd23e210c39d4634a424960d8237511f9d28fff97c623b4b850c93defdcd.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/4755dd23e210c39d4634a424960d8237511f9d28fff97c623b4b850c93defdcd.jpg)

![52e053294adc434220a19c65a8bee0d630e6f9c1ec3ff910b2f1824af1120afa.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/52e053294adc434220a19c65a8bee0d630e6f9c1ec3ff910b2f1824af1120afa.jpg)

![59f8619a11e7324c2067cb3242564aff9b2dc98d9ed515bc79fe8896b7b24188.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/59f8619a11e7324c2067cb3242564aff9b2dc98d9ed515bc79fe8896b7b24188.jpg)

![69010dfff467673563f05c73a9b014fbd020b6c8b3c4966bd262acf3faaa6cfd.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/69010dfff467673563f05c73a9b014fbd020b6c8b3c4966bd262acf3faaa6cfd.jpg)

![6a38be4b9adde091d7d5231b8b59f6403c43776fc6c1d187457d8352eca45481.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/6a38be4b9adde091d7d5231b8b59f6403c43776fc6c1d187457d8352eca45481.jpg)

![6f036b6cb6800726432abe553489fbd2575a34515c62fe1484c7390c8d680679.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/6f036b6cb6800726432abe553489fbd2575a34515c62fe1484c7390c8d680679.jpg)

![721d0f891f40601f96040abab988aa4664f6f9dc94f5e64014bc8226ca302e01.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/721d0f891f40601f96040abab988aa4664f6f9dc94f5e64014bc8226ca302e01.jpg)

![7c739a63b9e0bf79476a1fe060ab9f48e18f755b95a01b224cb4ebe653f0ea2f.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/7c739a63b9e0bf79476a1fe060ab9f48e18f755b95a01b224cb4ebe653f0ea2f.jpg)

![92583c71452c16f19702b63c281b45e5577d517a63a0a6a5f13d4d9018d85479.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/92583c71452c16f19702b63c281b45e5577d517a63a0a6a5f13d4d9018d85479.jpg)

![947a148dbb01cb36385cd8e651e3535e543873bb84b96b1bad17d4290e4c3f27.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/947a148dbb01cb36385cd8e651e3535e543873bb84b96b1bad17d4290e4c3f27.jpg)

![b02285df8afcb018fb75db437c8e7016455ded12709ef4aa0068c6f70dc6c580.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/b02285df8afcb018fb75db437c8e7016455ded12709ef4aa0068c6f70dc6c580.jpg)

![b2430377e0be2fabbe5860ea89e979987044ec87fd99f1f3cd127300f339b515.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/b2430377e0be2fabbe5860ea89e979987044ec87fd99f1f3cd127300f339b515.jpg)

![b5808d26c904f1025fe9dc38578212eb20f73c7f682c9e2cd0eff14ef2f616c9.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/b5808d26c904f1025fe9dc38578212eb20f73c7f682c9e2cd0eff14ef2f616c9.jpg)

![bb7a067dfe5888de087509f0e4687f08e80da3ba1175496df91b67dc183e0fc6.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/bb7a067dfe5888de087509f0e4687f08e80da3ba1175496df91b67dc183e0fc6.jpg)

![c7a5102921cd5c8eda3c2bed3bc03a5e9419bd8a62725a2def00974ea8166ae4.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/c7a5102921cd5c8eda3c2bed3bc03a5e9419bd8a62725a2def00974ea8166ae4.jpg)

![de8a650942645d1d7d6699cbc18d517da21858331165307b4d9b3fe5d0bc82ee.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/de8a650942645d1d7d6699cbc18d517da21858331165307b4d9b3fe5d0bc82ee.jpg)

![e9b9259d2e92f1a637d55e8da692853816f0ede5478567f236c8b8b14f5677dd.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/e9b9259d2e92f1a637d55e8da692853816f0ede5478567f236c8b8b14f5677dd.jpg)

![eeb28ea641e1a5984b07487292dfe4fee63afe6cc1fb7d5edf3624757708efea.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/eeb28ea641e1a5984b07487292dfe4fee63afe6cc1fb7d5edf3624757708efea.jpg)

![f13c7bb728d2d1bfddcf0e79a6f7b54119b247a4f85b6125c2e9825ffd20eb7a.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/f13c7bb728d2d1bfddcf0e79a6f7b54119b247a4f85b6125c2e9825ffd20eb7a.jpg)

![fd1efa7cc20f6c358690dcb306ee6c340888fb5bc7d343f55d40aeec742c3f22.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/images/fd1efa7cc20f6c358690dcb306ee6c340888fb5bc7d343f55d40aeec742c3f22.jpg)

### Tables

![06ddd1512fb4bb68a7538e6cc9bd95d4b324e186a5b02e39db6c93d6be99bc18.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/06ddd1512fb4bb68a7538e6cc9bd95d4b324e186a5b02e39db6c93d6be99bc18.jpg)

![08568db211f4b61eabc039f44e63931df046a4a134b2f38abf337bd8f2e739f7.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/08568db211f4b61eabc039f44e63931df046a4a134b2f38abf337bd8f2e739f7.jpg)

![1cf1c37a22a58d86f0c6a5ee5d03730aea1e354489c69940e55cbd625314dac0.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/1cf1c37a22a58d86f0c6a5ee5d03730aea1e354489c69940e55cbd625314dac0.jpg)

![2f3a7d3814db91ee8e00925166742deab03e4a0a38ccc82ab5993a6b5f150695.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/2f3a7d3814db91ee8e00925166742deab03e4a0a38ccc82ab5993a6b5f150695.jpg)

![4b2526a7782b4f015afee5ebfd1fc71b0a0a28f3136e5a59234444a678c089ab.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/4b2526a7782b4f015afee5ebfd1fc71b0a0a28f3136e5a59234444a678c089ab.jpg)

![694112ed15a3e3ab9bd4b7070500918755921e35618cd35c38e8afc1ab76187e.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/694112ed15a3e3ab9bd4b7070500918755921e35618cd35c38e8afc1ab76187e.jpg)

![7991e7c8549e0f3aab08bf2f0390fafa1728302dba89d4121344b2429dd2bd57.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/7991e7c8549e0f3aab08bf2f0390fafa1728302dba89d4121344b2429dd2bd57.jpg)

![bb0028f9cf3a70fbdf3a9d509cb87645fe730aed41fd928d2710abdd7046094c.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/bb0028f9cf3a70fbdf3a9d509cb87645fe730aed41fd928d2710abdd7046094c.jpg)

![e0737e0dcb23e6b960545f6f67193c09e57110d074afcb78270bbf5113e3354a.jpg](../icml_results/3106_DocOS_%20Towards%20Proactive%20Document-Guided%20Actions%20in%20GUI%20Agents/tables/e0737e0dcb23e6b960545f6f67193c09e57110d074afcb78270bbf5113e3354a.jpg)

<a id="paper-3068"></a>

## GAE: Unleashing Physical Potential of VLM with Generalizable Action Expert


### Images

![1e9327c1e9725a0069d1ec1ca33cd3f6f0e49864edd8f0faf417a4596926e915.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/1e9327c1e9725a0069d1ec1ca33cd3f6f0e49864edd8f0faf417a4596926e915.jpg)

![2c5626a0dd5d2dac450503edb59514907d9e01139d550dcc5657db7d35532016.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/2c5626a0dd5d2dac450503edb59514907d9e01139d550dcc5657db7d35532016.jpg)

![4f3b639c9c1cf5666c03cd81f2ff5663a6610e769c15f9919ad6f71574ef0f49.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/4f3b639c9c1cf5666c03cd81f2ff5663a6610e769c15f9919ad6f71574ef0f49.jpg)

![524376338560b61d9fd180c49896575aae2a4dd383c0b8cbf7cd02ef03b21a16.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/524376338560b61d9fd180c49896575aae2a4dd383c0b8cbf7cd02ef03b21a16.jpg)

![54dc11d950c6ca8e59b10b9a9944916def6d4703631b0cfa434f0eff4beb1d82.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/54dc11d950c6ca8e59b10b9a9944916def6d4703631b0cfa434f0eff4beb1d82.jpg)

![5c82e5aab8abf2c87d89a344423b7ca8698882d93970496734f888ab01df116e.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/5c82e5aab8abf2c87d89a344423b7ca8698882d93970496734f888ab01df116e.jpg)

![5fba342c8238fed57eb1d853001cef0a645465d8befb77f62c46548be001feb4.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/5fba342c8238fed57eb1d853001cef0a645465d8befb77f62c46548be001feb4.jpg)

![6b4fadf1253f90e2bc1d40b1fe040542dbcf2775b7bddf45ffde3eeb6f5e3667.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/6b4fadf1253f90e2bc1d40b1fe040542dbcf2775b7bddf45ffde3eeb6f5e3667.jpg)

![858b18a01926039598b3d04ea3af5696dcf0bdf618c32ba560c2c1d13e782f76.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/858b18a01926039598b3d04ea3af5696dcf0bdf618c32ba560c2c1d13e782f76.jpg)

![9145d4fc8813a5b5ec62c70f09e5bd779c71ea1fab59687177bdcc566bce731b.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/9145d4fc8813a5b5ec62c70f09e5bd779c71ea1fab59687177bdcc566bce731b.jpg)

![ab8bea1669219e066a0f44b337d350d841642f20b1d0078e62a585b0a2c8a04f.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/ab8bea1669219e066a0f44b337d350d841642f20b1d0078e62a585b0a2c8a04f.jpg)

![c1add5a03ff13bc99e4e3ddfdb4eb702792c0df449e815cbcacac81a0120e96e.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/c1add5a03ff13bc99e4e3ddfdb4eb702792c0df449e815cbcacac81a0120e96e.jpg)

![f6347ce741a32c343d3317491d99724505b910b0cf312ce1d6fa6a4a9a4d261c.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/f6347ce741a32c343d3317491d99724505b910b0cf312ce1d6fa6a4a9a4d261c.jpg)

![f653bf22d47f4e24841301205482c9dd645ed44e7bdd278214a20bce9db0991f.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/images/f653bf22d47f4e24841301205482c9dd645ed44e7bdd278214a20bce9db0991f.jpg)

### Tables

![15d40414ccadd8ee87fff6143c87e257bda8ed75861cc512c1f7405d166b20cf.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/15d40414ccadd8ee87fff6143c87e257bda8ed75861cc512c1f7405d166b20cf.jpg)

![41df546c23e20e77c2724afdfa09294f0aaf64bc59c6d17fdba6dfbd3145d3bf.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/41df546c23e20e77c2724afdfa09294f0aaf64bc59c6d17fdba6dfbd3145d3bf.jpg)

![45d3da89666b66faadabbc5ccb8f51f6dc113b300ede519d4cf995233cfbdd69.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/45d3da89666b66faadabbc5ccb8f51f6dc113b300ede519d4cf995233cfbdd69.jpg)

![4c6b951c8cc0e4270a8b81ee20ae046bbc1ee5919600878496f72a98c14c1649.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/4c6b951c8cc0e4270a8b81ee20ae046bbc1ee5919600878496f72a98c14c1649.jpg)

![501c9c095b2230e6f754ce3fdefa5714f68592ea4d9c586e4c10395e7ccf1773.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/501c9c095b2230e6f754ce3fdefa5714f68592ea4d9c586e4c10395e7ccf1773.jpg)

![5149fc3c20d799143e25521dbf2dd39920f335f39af9dda1da4bab1db366c89b.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/5149fc3c20d799143e25521dbf2dd39920f335f39af9dda1da4bab1db366c89b.jpg)

![54073b53d7eeb01e55c49784913174d3c034d6e841a5c041aa1b529083f4168d.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/54073b53d7eeb01e55c49784913174d3c034d6e841a5c041aa1b529083f4168d.jpg)

![55f223ff32d776fdea61da22c990ffee49df0ecd80526fef1ded68d68cb605bb.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/55f223ff32d776fdea61da22c990ffee49df0ecd80526fef1ded68d68cb605bb.jpg)

![61f7da2f86a479cde089fc5c8147942fde6e895fd8f14c86342e1cfb9416e9de.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/61f7da2f86a479cde089fc5c8147942fde6e895fd8f14c86342e1cfb9416e9de.jpg)

![62f1460f99600f55ea409648cef01cdd8c4b54b05cb60ef7ae2a3f4a43c6bd04.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/62f1460f99600f55ea409648cef01cdd8c4b54b05cb60ef7ae2a3f4a43c6bd04.jpg)

![6b74e039478b441cbd2b9c74f1671f161163c04a61350e52dae16cc5879d7066.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/6b74e039478b441cbd2b9c74f1671f161163c04a61350e52dae16cc5879d7066.jpg)

![768c507c91d0836242c180c08f28490ba9ac829e2308de6b5b11a227baace3ec.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/768c507c91d0836242c180c08f28490ba9ac829e2308de6b5b11a227baace3ec.jpg)

![7a7bf6c7ea6e737b60e25efd5a80a80392f144ec1a3d485e5097b26a3ae68cb1.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/7a7bf6c7ea6e737b60e25efd5a80a80392f144ec1a3d485e5097b26a3ae68cb1.jpg)

![84b171795343b355080ff2bc2a6947f2a3cd95ba454a46585ae686c5b377b81c.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/84b171795343b355080ff2bc2a6947f2a3cd95ba454a46585ae686c5b377b81c.jpg)

![90d0d7559d8361d974ac6acf7a2fc01337a2668a6ded062a9c274451b8287cf1.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/90d0d7559d8361d974ac6acf7a2fc01337a2668a6ded062a9c274451b8287cf1.jpg)

![a9aab063afc62c8441f6cf28945f7ce9cab20fa86951a0bb58bf7962e6e062c4.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/a9aab063afc62c8441f6cf28945f7ce9cab20fa86951a0bb58bf7962e6e062c4.jpg)

![bf0ad42bcd0217d949515bc07f3b461c7093ced5eb7ba12e1e7024486cb4636b.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/bf0ad42bcd0217d949515bc07f3b461c7093ced5eb7ba12e1e7024486cb4636b.jpg)

![c257499ca62def95c89a34e6f57214fcc51d856c1fa4d09d38880cce3be45d0f.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/c257499ca62def95c89a34e6f57214fcc51d856c1fa4d09d38880cce3be45d0f.jpg)

![ca5f604e8d6bfa63008d10836e456cfd70404f1105348a4a8442085aacd2b19c.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/ca5f604e8d6bfa63008d10836e456cfd70404f1105348a4a8442085aacd2b19c.jpg)

![e4729866a0b0a66838b81d34a8b6ccf1434b39ba03aea7f0f6e6d7af99b1d245.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/e4729866a0b0a66838b81d34a8b6ccf1434b39ba03aea7f0f6e6d7af99b1d245.jpg)

![e9e64091cb3741b22855f6b9f9a9655a768b52f87571a048dc38adf790517508.jpg](../icml_results/3107_GAE_%20Unleashing%20Physical%20Potential%20of%20VLM%20with%20Generalizable%20Action%20Expert/tables/e9e64091cb3741b22855f6b9f9a9655a768b52f87571a048dc38adf790517508.jpg)

<a id="paper-3069"></a>

## ConvexBench: Can LLMs Recognize Convex Functions?

### Images

![12eb43ce26385996a7ccec2c707430e618d819496d1f371a4cb4401325392b3d.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/12eb43ce26385996a7ccec2c707430e618d819496d1f371a4cb4401325392b3d.jpg)

![2f3a942948d7187bde505b3cdc6a4f066f80bfcc47c6f2f906e7e597eccd3d6b.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/2f3a942948d7187bde505b3cdc6a4f066f80bfcc47c6f2f906e7e597eccd3d6b.jpg)

![51818bea5a5aaeb42223b2e8ccb207ec02e5a9a6c511d2a1d7c31940bc727b54.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/51818bea5a5aaeb42223b2e8ccb207ec02e5a9a6c511d2a1d7c31940bc727b54.jpg)

![54bb9a552efe9f791568a485cecfa408361ab6ae1007568f860fb7b8a34efb0f.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/54bb9a552efe9f791568a485cecfa408361ab6ae1007568f860fb7b8a34efb0f.jpg)

![9ef37ced9a570ddb5a4dd5a6dcae08cb6a6f355b473f6cbc8e7e6fbd6b523df7.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/9ef37ced9a570ddb5a4dd5a6dcae08cb6a6f355b473f6cbc8e7e6fbd6b523df7.jpg)

![b5e7af41d1ac8f9d0e61a02d43830a6ad0ebfc1ec041656b2da30719bb4faecf.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/b5e7af41d1ac8f9d0e61a02d43830a6ad0ebfc1ec041656b2da30719bb4faecf.jpg)

![d73869c9d3ed8670b4ee2f7160edaca8df1eba5448af05912fcdf296a33c7a6f.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/d73869c9d3ed8670b4ee2f7160edaca8df1eba5448af05912fcdf296a33c7a6f.jpg)

### Tables

![7b14e95b0c3e24d2b32b6b62b5bca6810d27a408e529ad12660be16003d6a38e.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/7b14e95b0c3e24d2b32b6b62b5bca6810d27a408e529ad12660be16003d6a38e.jpg)

![c79f0b0ef365c3733e82f69b18311c03865c2f03fb41625e7533f666f40bf433.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/c79f0b0ef365c3733e82f69b18311c03865c2f03fb41625e7533f666f40bf433.jpg)

![da3425dd094dfb6f6a4290158a80867246d78e8201e444640cbea2b9e54568e2.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/da3425dd094dfb6f6a4290158a80867246d78e8201e444640cbea2b9e54568e2.jpg)

![db9469235ac6edc5599271347a7cb994697e1539a03313facfae2029afbc7ce9.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/db9469235ac6edc5599271347a7cb994697e1539a03313facfae2029afbc7ce9.jpg)

![e37ddd8e51ee19e8b4703ffc8629c79f02539a3881a8dc4d711b6a3a76c2b93a.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/e37ddd8e51ee19e8b4703ffc8629c79f02539a3881a8dc4d711b6a3a76c2b93a.jpg)
