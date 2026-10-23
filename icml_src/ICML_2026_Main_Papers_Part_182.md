# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 182 of 197

## 目录 (Table of Contents)

1. [iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](#paper-5974)
2. [Grounding LLMs in Scientific Discovery via Embodied Actions](#paper-5975)
3. [FedGain: Toward Negative-Gain-Free Client Collaboration in Federated Learning](#paper-5976)
4. [The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training](#paper-5977)
5. [RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](#paper-5978)
6. [Transform Trained Transformer for Accelerating Native 4K Video Generation](#paper-5979)
7. [Cycle-of-Science: Reliable Reasoning through Counterfactual Verification for Agent Decision Making](#paper-5980)
8. [Unveiling And Addressing Dimensional Collapse In Vector Quantization Models Via Codebook Regularization](#paper-5981)
9. [Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR](#paper-5982)
10. [GRAPE: Let GRPO Supervise Query Rewriting by Ranking for Retrieval](#paper-5983)
11. [Learnability-Driven Knowledge Assimilation for Class-Incremental Semantic Segmentation](#paper-5984)
12. [Think in Latent, Explain in Language: Self-Explainable Latent Reasoning](#paper-5985)
13. [Population-Free Pareto Tracking for Sample-Efficient Multi-Policy MORL](#paper-5986)
14. [Learning GUI Grounding with Spatial Reasoning from Visual Feedback](#paper-5987)
15. [LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](#paper-5988)
16. [LitReview Arena: Evaluating Literature Review Agents with Battle-style Peer Review Platform](#paper-5989)
17. [WinDeskGround: A Benchmark for Robust GUI Grounding in Complex Multi-Window Desktop Environments](#paper-5990)
18. [When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs](#paper-5991)
19. [Syntax vs. Semantics: How Transformers Learn Deep Dependencies](#paper-5992)
20. [Nonparametric Data Attribution for Diffusion Models](#paper-5993)
21. [What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code](#paper-5994)
22. [CameraNoise: Enabling Faithful Camera Control in Video Diffusion through Geometry-Flow-Guided Noise Warping](#paper-5995)
23. [GHOST: Geometry-Guided Hallucination of Opaque Surface Textures](#paper-5996)
24. [MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation](#paper-5997)
25. [When Is Rank-1 Enough? Geometry-Guided Initialization for Parameter-Efficient Fine-Tuning](#paper-5998)
26. [From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data](#paper-5999)
27. [DiP-G: Discrete Prompting for Graph Neural Networks](#paper-6000)
28. [Shift-Dependent Asymmetry: Orthogonal Inverse Low-Rank Adaptation for Federated Medical Segmentation](#paper-6001)
29. [Rh-3DGS: Robust Open-Vocabulary Scene Understanding via Riemannian Huber Distillation and Manifold-Aware Sampling](#paper-6002)
30. [Learning Sparse Visual Representations via Spatial-Semantic Factorization](#paper-6003)
31. [RECAST: Model Reconstruction via Counterfactual-Aware Wasserstein Geometry under Limited Data](#paper-6004)
32. [CAffNet: Hard Constraint-Affine Neural Networks](#paper-6005)
33. [Rethinking Low-Confidence Pseudo Labels: Influence-Aware Semi-Supervised Fine-Tuning for Hyperspectral Change Detection](#paper-6006)

---


<a id="paper-5974"></a>

## iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning

### Images

![1277ebbe25a2d360690d5a89ca1576bae542591ee749c78a0c1f742f4ae48371.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/1277ebbe25a2d360690d5a89ca1576bae542591ee749c78a0c1f742f4ae48371.jpg)

![27ce868e9b690c615b5422a07f679bb6fca2d53041ba8b7139ac90fa9fc947c8.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/27ce868e9b690c615b5422a07f679bb6fca2d53041ba8b7139ac90fa9fc947c8.jpg)

![3b70dad6c56e17b9ba73405cce3767cf1f5440d662e02ebc8f95cb1e241af06b.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/3b70dad6c56e17b9ba73405cce3767cf1f5440d662e02ebc8f95cb1e241af06b.jpg)

![59e2cc1eb849155aa61b5e2d4486298cf79ff35f845c9c069d99f6f24a3f4a7c.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/59e2cc1eb849155aa61b5e2d4486298cf79ff35f845c9c069d99f6f24a3f4a7c.jpg)

![5f1d5ca7513ad84ff6399644b4184219becc9756d8f6df75419fde3deaa321c2.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/5f1d5ca7513ad84ff6399644b4184219becc9756d8f6df75419fde3deaa321c2.jpg)

![a369c2b1bb8cf9a04d098a325e95da1f3d16b5ba5546beef6de44bd9831a9a4b.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/a369c2b1bb8cf9a04d098a325e95da1f3d16b5ba5546beef6de44bd9831a9a4b.jpg)

![e818a511227e3026c85720ea93a39420d02f0695a3b70245c5fa44836cbecff0.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/e818a511227e3026c85720ea93a39420d02f0695a3b70245c5fa44836cbecff0.jpg)

![ec2e381c6db33690cf049ca1736937a92947688710eb872ba8e81a12852d1285.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/ec2e381c6db33690cf049ca1736937a92947688710eb872ba8e81a12852d1285.jpg)

![f38373df7e10b259789d25e5b5b7d358a2eb3920e4d804836cffd6afe6a9f64f.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/f38373df7e10b259789d25e5b5b7d358a2eb3920e4d804836cffd6afe6a9f64f.jpg)

![f9fbe15420ec4e07f1b1c670452db14c129853450f3d569364855f801dc79a50.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/f9fbe15420ec4e07f1b1c670452db14c129853450f3d569364855f801dc79a50.jpg)

### Tables

![148d16c4a1cc1f62d4d3a634de783d461c6c582d88afa1b60c643021f0483f7e.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/148d16c4a1cc1f62d4d3a634de783d461c6c582d88afa1b60c643021f0483f7e.jpg)

![1953703f3afa2ae79d889448b5dbc44a0399243cffec8242b311c0f032001b71.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/1953703f3afa2ae79d889448b5dbc44a0399243cffec8242b311c0f032001b71.jpg)

![1a0569586097bff84f8d8687dae001e9abd6c60f48bee5a4d2ce2799b0cb8f75.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/1a0569586097bff84f8d8687dae001e9abd6c60f48bee5a4d2ce2799b0cb8f75.jpg)

![3eb1c8cb9107c272bb3f7c1eea60df34d6844e534fce10b6d27e063bbb46f0c6.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/3eb1c8cb9107c272bb3f7c1eea60df34d6844e534fce10b6d27e063bbb46f0c6.jpg)

![66ead100288996a57cd05bd8205bf892ecbc89761cc1a44d29722be4aaaf4cfa.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/66ead100288996a57cd05bd8205bf892ecbc89761cc1a44d29722be4aaaf4cfa.jpg)

![becc5a993a5514bca34d2374c7d340a98a56082e5fba0487442d84d42080f18d.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/becc5a993a5514bca34d2374c7d340a98a56082e5fba0487442d84d42080f18d.jpg)

![cfbcefb0a86569b834c3d02b0f03e9056bc7d8d56cb67b164095eaeddab3d9f1.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/cfbcefb0a86569b834c3d02b0f03e9056bc7d8d56cb67b164095eaeddab3d9f1.jpg)

![d33f374008de6a1c1c7b3a7cfe6d60a07cba817b24e5f1013b802b18fb0195a4.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/d33f374008de6a1c1c7b3a7cfe6d60a07cba817b24e5f1013b802b18fb0195a4.jpg)

![e75382a31f500460afd3b0462dbec9983cc9f567ed5ea9fcb3d89aa6a0dcf8ce.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/e75382a31f500460afd3b0462dbec9983cc9f567ed5ea9fcb3d89aa6a0dcf8ce.jpg)

## iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning


### Images

![123b5907f130aaf5ecad888e03686fd7293ffe0686cde7ede19df8f3fd286b99.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/123b5907f130aaf5ecad888e03686fd7293ffe0686cde7ede19df8f3fd286b99.jpg)

![33608cd4496edca9110e114dbceb9515e2af7b9de8b3d04fb0a553a766cdd352.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/33608cd4496edca9110e114dbceb9515e2af7b9de8b3d04fb0a553a766cdd352.jpg)

![7c45637e039a4b746d84f8375bb5bcfb270abb5e86f9efa92aa33edbb77ac17f.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/7c45637e039a4b746d84f8375bb5bcfb270abb5e86f9efa92aa33edbb77ac17f.jpg)

![8574b16a8143408a38ed4c73715244233df3f089f27021912cfc4796bfd91ab6.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/8574b16a8143408a38ed4c73715244233df3f089f27021912cfc4796bfd91ab6.jpg)

![f0fc83bbce76eff39296f194d95d6e0f09a5ecdbb57c588655c9e68702ace5d2.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/f0fc83bbce76eff39296f194d95d6e0f09a5ecdbb57c588655c9e68702ace5d2.jpg)

![fc91429a6bb127904ab1197c53b502544256ea998a504b2256a88571620b82f5.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/images/fc91429a6bb127904ab1197c53b502544256ea998a504b2256a88571620b82f5.jpg)

### Tables

![15293cfe2be1e19a1e0dad809804a31945ea9c95a2710db922e5d193b11dd77d.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/15293cfe2be1e19a1e0dad809804a31945ea9c95a2710db922e5d193b11dd77d.jpg)

![2f4abf363a4a84f2ef6f3aab5372a6f2cecc5ee6cdd786662b7dbc90ce62d398.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/2f4abf363a4a84f2ef6f3aab5372a6f2cecc5ee6cdd786662b7dbc90ce62d398.jpg)

![3e4e683f267b90f3a9911dc8fbeff22c6aa1f13fa7b2b844e7f7ecb87009d8c8.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/3e4e683f267b90f3a9911dc8fbeff22c6aa1f13fa7b2b844e7f7ecb87009d8c8.jpg)

![46a0525ff95be026bb160a21b75561f37338e866a48074e982f6e8d204764418.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/46a0525ff95be026bb160a21b75561f37338e866a48074e982f6e8d204764418.jpg)

![4937383c5bebab96adae8b4aee3a358943f04f50546a2e115419d23fea6040ba.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/4937383c5bebab96adae8b4aee3a358943f04f50546a2e115419d23fea6040ba.jpg)

![58b9c284fd5b6bae05fe4fd44b8d25e3bb5e35fb6daa01320ca686245fb5ed1b.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/58b9c284fd5b6bae05fe4fd44b8d25e3bb5e35fb6daa01320ca686245fb5ed1b.jpg)

![6053527eb71ac8822a5804c04cd37e49be56635457ae7a91fb16b1ec67c4d923.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/6053527eb71ac8822a5804c04cd37e49be56635457ae7a91fb16b1ec67c4d923.jpg)

![91015fe9891c95c82b5e5db9a4217196ab9d1189d9d1d640bc4a1bcff109bd2e.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/91015fe9891c95c82b5e5db9a4217196ab9d1189d9d1d640bc4a1bcff109bd2e.jpg)

![9e2a01610325d2df0d7a39cc9cc91be3fd75901c59ee1c57a04c32f17af06224.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/9e2a01610325d2df0d7a39cc9cc91be3fd75901c59ee1c57a04c32f17af06224.jpg)

![9f5513cdc3af8e0b6e3e51e9ddb1d12fe19ed49f05c681a73c40080a4a75ca85.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/9f5513cdc3af8e0b6e3e51e9ddb1d12fe19ed49f05c681a73c40080a4a75ca85.jpg)

![a79484796649d24812da1613cc6aa84c3cab24831595d40e8662c7baafb20a9a.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/a79484796649d24812da1613cc6aa84c3cab24831595d40e8662c7baafb20a9a.jpg)

![cff0a0f29d1fb85c7ac1c33cd0af8c15d249f45f24302de7fdb67df1eda0390a.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/cff0a0f29d1fb85c7ac1c33cd0af8c15d249f45f24302de7fdb67df1eda0390a.jpg)

![d40df2d9b680b2e2dfee5132b8baefe63ccfc1af7c7237e6e16f41ed03e2b775.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/d40df2d9b680b2e2dfee5132b8baefe63ccfc1af7c7237e6e16f41ed03e2b775.jpg)

![e876f7214de45b95cd11ffd43d6e9bd05c4675cbd0c9e4f035890a883592943b.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/e876f7214de45b95cd11ffd43d6e9bd05c4675cbd0c9e4f035890a883592943b.jpg)

![ee731f58276fae2a91430d72bbefbe025df097b4dee50fd82194a64caab8b835.jpg](../icml_results/6023_iVGR_%20Internalizing%20Visually%20Grounded%20Reasoning%20for%20MLLMs%20with%20Reinforcement%20Learning/tables/ee731f58276fae2a91430d72bbefbe025df097b4dee50fd82194a64caab8b835.jpg)

<a id="paper-5975"></a>

## Grounding LLMs in Scientific Discovery via Embodied Actions


### Images

![03af793386b5f9a298dc6c8305c6adf4958fc47600140dad31779827238cca2c.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/03af793386b5f9a298dc6c8305c6adf4958fc47600140dad31779827238cca2c.jpg)

![0b3753c4303b99b39e6f11a32b2b6578fdb6e6e9fa38dfdb8c0df894521f14f2.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/0b3753c4303b99b39e6f11a32b2b6578fdb6e6e9fa38dfdb8c0df894521f14f2.jpg)

![2604ca4967b485952e1fba6efeaf3a317d52ebf0aee3df1ec01eac2ef9b2a712.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/2604ca4967b485952e1fba6efeaf3a317d52ebf0aee3df1ec01eac2ef9b2a712.jpg)

![389f3566c874e9dd004eed337e42ca35465561898f98a787dd34ab2aff154edb.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/389f3566c874e9dd004eed337e42ca35465561898f98a787dd34ab2aff154edb.jpg)

![62152b169d95082601236082d569cb78249bd5fafc8d979126c78ef82ee4c362.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/62152b169d95082601236082d569cb78249bd5fafc8d979126c78ef82ee4c362.jpg)

![705b4e0d39eadb324e3ad0e7e6930b66110c8e39380dda147f88489117c89c87.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/705b4e0d39eadb324e3ad0e7e6930b66110c8e39380dda147f88489117c89c87.jpg)

![7daf4101f535cc1b8abff3dea70d9b28c29af7cfbee77573c376d0f88713b458.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/7daf4101f535cc1b8abff3dea70d9b28c29af7cfbee77573c376d0f88713b458.jpg)

![96168a0344d0e9a7abcdb064b76a2217d3afbfb65ae7b35d0f4b2264b6d11cda.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/96168a0344d0e9a7abcdb064b76a2217d3afbfb65ae7b35d0f4b2264b6d11cda.jpg)

![996294a7667279817d0bed617d1a40264764050bbaa125d26e576a774a90b373.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/996294a7667279817d0bed617d1a40264764050bbaa125d26e576a774a90b373.jpg)

![ba60bb16b8ae6110119be5bc016e19a4abf48a04ca88a5ea0f91b6a8e909df43.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/ba60bb16b8ae6110119be5bc016e19a4abf48a04ca88a5ea0f91b6a8e909df43.jpg)

![ea49aae23c8906c87c858f7759ba0fdab9a37088a4cc931f05f1118cef592873.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/images/ea49aae23c8906c87c858f7759ba0fdab9a37088a4cc931f05f1118cef592873.jpg)

### Tables

![25c4e84ca8ab382b48ac2802e335feab07dc2d2c4af92b7f452cd3dd3724cf7c.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/25c4e84ca8ab382b48ac2802e335feab07dc2d2c4af92b7f452cd3dd3724cf7c.jpg)

![34ce133039fe87cf6064eef62d96122c26cbd29542ccb74dfb9b1200b562f860.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/34ce133039fe87cf6064eef62d96122c26cbd29542ccb74dfb9b1200b562f860.jpg)

![378d84fe467907fa3ff3d7eab18e4ab2825aa639ddaeb28eaceacc5d6b831ede.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/378d84fe467907fa3ff3d7eab18e4ab2825aa639ddaeb28eaceacc5d6b831ede.jpg)

![3dca9818d0950261b728bff30f30e9d13b2bfe6e709c4b38f18e6a2bd86fcc02.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/3dca9818d0950261b728bff30f30e9d13b2bfe6e709c4b38f18e6a2bd86fcc02.jpg)

![43effdbf707135cae2ee764364dd97ac2700c9b582fd8603fbdaab63670e1477.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/43effdbf707135cae2ee764364dd97ac2700c9b582fd8603fbdaab63670e1477.jpg)

![5167785b71cc9975819e2a0c921d760d3aa635b568c110215ea5675b72d27e68.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/5167785b71cc9975819e2a0c921d760d3aa635b568c110215ea5675b72d27e68.jpg)

![61ef91a09b05670e88bb3272651c639811884b441af984d5568faeb05f487a7b.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/61ef91a09b05670e88bb3272651c639811884b441af984d5568faeb05f487a7b.jpg)

![697db6311d612882b5f629b36184a15679a752f4fdb9502a8b80c86266b492b2.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/697db6311d612882b5f629b36184a15679a752f4fdb9502a8b80c86266b492b2.jpg)

![78f48c015a3e34b0f33f39cc879ae00d5a2c579a8112a743f148eb6bfe42774f.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/78f48c015a3e34b0f33f39cc879ae00d5a2c579a8112a743f148eb6bfe42774f.jpg)

![92939b1cf9619dbeecb64f3964b4116b676ccb4bdabcb4bf9289011970996b06.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/92939b1cf9619dbeecb64f3964b4116b676ccb4bdabcb4bf9289011970996b06.jpg)

![ae647eff04e916468982aa02df9b249719057581488faffd9616c8b8fc7a7103.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/ae647eff04e916468982aa02df9b249719057581488faffd9616c8b8fc7a7103.jpg)

![e7417d086debc687fbbbc151d3da56fe32120e7d71536440d53e35359fa81151.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/e7417d086debc687fbbbc151d3da56fe32120e7d71536440d53e35359fa81151.jpg)

![e81ed6b34e6ddae22ee3b0f98b3cb0a23d89ac1e0200d362886d9c39d2c84e75.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/e81ed6b34e6ddae22ee3b0f98b3cb0a23d89ac1e0200d362886d9c39d2c84e75.jpg)

![fda2e8bcf43ab49922a1219a07b36db44155b25c82b6bc89055001ec15bdfd8f.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/fda2e8bcf43ab49922a1219a07b36db44155b25c82b6bc89055001ec15bdfd8f.jpg)

![ffa46eb229c9d7157dc42d210d87b514a40ff26a974ee61f97dff895509ecb9d.jpg](../icml_results/6024_Grounding%20LLMs%20in%20Scientific%20Discovery%20via%20Embodied%20Actions/tables/ffa46eb229c9d7157dc42d210d87b514a40ff26a974ee61f97dff895509ecb9d.jpg)

<a id="paper-5976"></a>

## FedGain: Toward Negative-Gain-Free Client Collaboration in Federated Learning


### Images

![0742d5333cb3abce527cc6eed319a07dd8f727c54a3dc0a188ff57a7e48ce435.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/0742d5333cb3abce527cc6eed319a07dd8f727c54a3dc0a188ff57a7e48ce435.jpg)

![4c78756ba5c5ba294f1b8b2ce31f0fa4c66219d24f11193ffa9eee2221bd35f8.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/4c78756ba5c5ba294f1b8b2ce31f0fa4c66219d24f11193ffa9eee2221bd35f8.jpg)

![806d3957bc9ac451934878faed8b94b06c11349c423cb009b7860aca2a2b7631.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/806d3957bc9ac451934878faed8b94b06c11349c423cb009b7860aca2a2b7631.jpg)

![9fff6a6880eaf89af29b5bd6a37ee66d7f1c02e1ecb272fe5e11aa8419eef5bf.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/9fff6a6880eaf89af29b5bd6a37ee66d7f1c02e1ecb272fe5e11aa8419eef5bf.jpg)

![c9fe9ffd5c356b4cdf76ad1fc45317e8e1b70c04c354277e891b892b6c939018.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/c9fe9ffd5c356b4cdf76ad1fc45317e8e1b70c04c354277e891b892b6c939018.jpg)

![ddb70af5cb8266d6b5c755bd3c1fbc2e569516b099db0c1bfac1e4b0b31ff5b7.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/ddb70af5cb8266d6b5c755bd3c1fbc2e569516b099db0c1bfac1e4b0b31ff5b7.jpg)

![f7564eeb03c4a4442ddde0e137dea9518944079c3885a29d0d47eb126dc63554.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/images/f7564eeb03c4a4442ddde0e137dea9518944079c3885a29d0d47eb126dc63554.jpg)

### Tables

![0d7519873ee82089ffadb4bde26e50edfe25e1b0372222e865b29f4470ae6d37.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/0d7519873ee82089ffadb4bde26e50edfe25e1b0372222e865b29f4470ae6d37.jpg)

![225629c541d3bb02ae0a91a43af8c12d0ef0f4dac164f1e42fd7985b9ee754f0.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/225629c541d3bb02ae0a91a43af8c12d0ef0f4dac164f1e42fd7985b9ee754f0.jpg)

![3d458613f1d53bdb704cb552ab96e178ac7381d5cc73343903b25679d37cd030.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/3d458613f1d53bdb704cb552ab96e178ac7381d5cc73343903b25679d37cd030.jpg)

![43d422b346f7ed4c05e9878ee4cbe11ff7797f972862a7cd28bd6edf937b1ebf.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/43d422b346f7ed4c05e9878ee4cbe11ff7797f972862a7cd28bd6edf937b1ebf.jpg)

![6c68a034d14a4d0afde11b448b766ec73f5c6e3b34660ee56b6ad2b2a64fdb61.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/6c68a034d14a4d0afde11b448b766ec73f5c6e3b34660ee56b6ad2b2a64fdb61.jpg)

![6e8a39121ea07aff58a1cf95dd294a8440eb0d612feffd44e66fa1ca49d42aac.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/6e8a39121ea07aff58a1cf95dd294a8440eb0d612feffd44e66fa1ca49d42aac.jpg)

![81e837b81f7989429ec0e11d4344fe3f409786077457ee3491bdede5cf78fdc6.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/81e837b81f7989429ec0e11d4344fe3f409786077457ee3491bdede5cf78fdc6.jpg)

![a302b75ea03273b99ba090e57022a7eda556a373da9b83fbc703e7d3c16e44a8.jpg](../icml_results/6025_FedGain_%20Toward%20Negative-Gain-Free%20Client%20Collaboration%20in%20Federated%20Learning/tables/a302b75ea03273b99ba090e57022a7eda556a373da9b83fbc703e7d3c16e44a8.jpg)

<a id="paper-5977"></a>

## The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training


### Images

![0eb5cb591e7a937ee8ee5046b0128f093b757601d75ad5b06ad4e9c49116fb00.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/images/0eb5cb591e7a937ee8ee5046b0128f093b757601d75ad5b06ad4e9c49116fb00.jpg)

![41c1518c40188e37aa2aadae9378f829cb6db98088396225f520b0f42c2aef03.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/images/41c1518c40188e37aa2aadae9378f829cb6db98088396225f520b0f42c2aef03.jpg)

![8e0c716bf53ef3427a3913fdfd476fbd264f1bb5aa415d011d71d48b5f8003ef.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/images/8e0c716bf53ef3427a3913fdfd476fbd264f1bb5aa415d011d71d48b5f8003ef.jpg)

![a6f5e3fa87c330e3879ccc9b032b594ba0617e321c219d3959ba387f2c0d2edf.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/images/a6f5e3fa87c330e3879ccc9b032b594ba0617e321c219d3959ba387f2c0d2edf.jpg)

![cf022a5dedbdf442fb37efd2181ea8d3b30a848bdbee369bd2738c45c48eb953.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/images/cf022a5dedbdf442fb37efd2181ea8d3b30a848bdbee369bd2738c45c48eb953.jpg)

### Tables

![36cfc36f36d1d9494102f2dec320e54399156abe1dbcb4b271cbd20aa320f95a.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/tables/36cfc36f36d1d9494102f2dec320e54399156abe1dbcb4b271cbd20aa320f95a.jpg)

![d0f89078ff8b0786d2f20b510c420300795b6d2235cd5fa09fb9f497f06dc683.jpg](../icml_results/6026_The%20Stability%20of%20Singular%20Distribution_%20A%20Spectral%20Perspective%20on%20the%20Two-Phase%20Dynamics%20of%20Language/tables/d0f89078ff8b0786d2f20b510c420300795b6d2235cd5fa09fb9f497f06dc683.jpg)

<a id="paper-5978"></a>

## RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation


### Images

![454f2deef26e0d4736f6a5016bc2d8232649ae118726719dc3a2c95775e67e73.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/454f2deef26e0d4736f6a5016bc2d8232649ae118726719dc3a2c95775e67e73.jpg)

![620ad9b7f25c8d3a2457298723c2e70c675b63f340107559a74b23506c61485a.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/620ad9b7f25c8d3a2457298723c2e70c675b63f340107559a74b23506c61485a.jpg)

![6b8fcb75402ed5c74a7443bd8d283970a072dff7b662729bed0e80d6b1f1ad6f.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/6b8fcb75402ed5c74a7443bd8d283970a072dff7b662729bed0e80d6b1f1ad6f.jpg)

![d33a8a8ce6c744eeaa0ddb3c0dd9f539069a1d7ff48750764fb38c35f45e873e.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/d33a8a8ce6c744eeaa0ddb3c0dd9f539069a1d7ff48750764fb38c35f45e873e.jpg)

![dfcffcd692e959d6b6b2da3869872c33194d0eebbefe223e3bb1c40a7c475d46.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/dfcffcd692e959d6b6b2da3869872c33194d0eebbefe223e3bb1c40a7c475d46.jpg)

![f77b61ec9eecd09b1bf2496503502f225d9681ea630321131f8b1c2d3f382294.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/f77b61ec9eecd09b1bf2496503502f225d9681ea630321131f8b1c2d3f382294.jpg)

![f9970408cba6778013cee901125b1b5204bab0ea9ab36f7b8e657ecd50e37555.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/images/f9970408cba6778013cee901125b1b5204bab0ea9ab36f7b8e657ecd50e37555.jpg)

### Tables

![021a367c523c46f2286f33dbca1cf98da06cf7d1e2c25a419be3a646e938bc76.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/021a367c523c46f2286f33dbca1cf98da06cf7d1e2c25a419be3a646e938bc76.jpg)

![0b777a002ef1ba52b5c27ff59625a9203dc497a714c0f8775ced8d11a2e5ae08.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/0b777a002ef1ba52b5c27ff59625a9203dc497a714c0f8775ced8d11a2e5ae08.jpg)

![30be416d26d9a1e5c03f311d86236c8d7f345441a1e7e8193a799369c83c879b.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/30be416d26d9a1e5c03f311d86236c8d7f345441a1e7e8193a799369c83c879b.jpg)

![54f2859d64d65ffee6ad5b81bc19d24b9b8ccbe83f4cd0df93393ae71fc950d2.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/54f2859d64d65ffee6ad5b81bc19d24b9b8ccbe83f4cd0df93393ae71fc950d2.jpg)

![698d5e5a8862ce448be20da0ecb12d16d0f319952d686e2e3cb5fd4f9c2f2e8c.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/698d5e5a8862ce448be20da0ecb12d16d0f319952d686e2e3cb5fd4f9c2f2e8c.jpg)

![8d08f8b4fbe8531ea2ab5a98227f29062ee80cd0d493ea64cd494cd042ac5393.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/8d08f8b4fbe8531ea2ab5a98227f29062ee80cd0d493ea64cd494cd042ac5393.jpg)

![be11cc354d10d1be104bb272a5b0a439c11cc2a80f243484b498b3bf94f792d8.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/be11cc354d10d1be104bb272a5b0a439c11cc2a80f243484b498b3bf94f792d8.jpg)

![f827511fd2344ca83fbea4cd2bfdc6ac888637f0937fdcb42ed2ed4abf9e9af1.jpg](../icml_results/6027_RADAR_%20Redundancy-Aware%20Diffusion%20for%20Multi-Agent%20Communication%20Structure%20Generation/tables/f827511fd2344ca83fbea4cd2bfdc6ac888637f0937fdcb42ed2ed4abf9e9af1.jpg)

<a id="paper-5979"></a>

## Transform Trained Transformer for Accelerating Native 4K Video Generation


### Images

![1bbe3c8b51f2798d2a187ebc175cbf786407b11dc2ff70f0ce5945141713ce6b.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/1bbe3c8b51f2798d2a187ebc175cbf786407b11dc2ff70f0ce5945141713ce6b.jpg)

![1cb83189946d31581024d6372d7306c3e31c978bd2c5465a2189c2156a7bd34a.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/1cb83189946d31581024d6372d7306c3e31c978bd2c5465a2189c2156a7bd34a.jpg)

![2b52f7727e7283ba8a8a7cbb62314846b4cbe06eb5a2fa2d660403abb6c97ac9.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/2b52f7727e7283ba8a8a7cbb62314846b4cbe06eb5a2fa2d660403abb6c97ac9.jpg)

![300426e00d16fc37f4e553dc741be369980f79d97754d6363b5ffa7a67cfeb96.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/300426e00d16fc37f4e553dc741be369980f79d97754d6363b5ffa7a67cfeb96.jpg)

![4da9bff9c8a557a46af0b544a327a4075304e8b284ad96c4124c0ae8c9772282.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/4da9bff9c8a557a46af0b544a327a4075304e8b284ad96c4124c0ae8c9772282.jpg)

![764fc23a519438dcdfe574a45aed83f53c326b35ed39fd970c9a2c5c9ef2b5c2.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/764fc23a519438dcdfe574a45aed83f53c326b35ed39fd970c9a2c5c9ef2b5c2.jpg)

![8f69d61e853fc4477bc4084ce379a418326274ffb41d93a4b46cf4f6e8347f83.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/8f69d61e853fc4477bc4084ce379a418326274ffb41d93a4b46cf4f6e8347f83.jpg)

![fcd60213f5699ee9ed470a18f0f14de90abb7423661e4f12ea1e1570cd97f620.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/images/fcd60213f5699ee9ed470a18f0f14de90abb7423661e4f12ea1e1570cd97f620.jpg)

### Tables

![1e341bf7f1a97b10dee61f80d4ba0cc319c35f1fdf971a2c8bde7ade0be345b2.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/1e341bf7f1a97b10dee61f80d4ba0cc319c35f1fdf971a2c8bde7ade0be345b2.jpg)

![46ecf5278ad229bbb1ce5ec9413f8fef1ac465b0f65f35be2aaddf82adf9f74e.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/46ecf5278ad229bbb1ce5ec9413f8fef1ac465b0f65f35be2aaddf82adf9f74e.jpg)

![76fdec61bd703fd481343447f6962b07034f03cdc0145a8dc3d27f12a6b1240e.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/76fdec61bd703fd481343447f6962b07034f03cdc0145a8dc3d27f12a6b1240e.jpg)

![7eff63ef07aa4b152e9cfae80b14b4da28e1f90bbc238be2684c856d76e3cd33.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/7eff63ef07aa4b152e9cfae80b14b4da28e1f90bbc238be2684c856d76e3cd33.jpg)

![a7029171ee8413856e034762c0a672c02eb1fec863b6e5c8d588b8c4ddea6576.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/a7029171ee8413856e034762c0a672c02eb1fec863b6e5c8d588b8c4ddea6576.jpg)

![ae2dd79c3c1f37a8b00d11cdeab8e94ea445b41d3e3dcdd4b260cf1f3de8c25c.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/ae2dd79c3c1f37a8b00d11cdeab8e94ea445b41d3e3dcdd4b260cf1f3de8c25c.jpg)

![ba2060cce8167530c1f49e60edb15d9189c33ea239ae9a4dbd4890a50060718f.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/ba2060cce8167530c1f49e60edb15d9189c33ea239ae9a4dbd4890a50060718f.jpg)

![ce2d9687ae99b0ceb1046880fd437f7a801d743669a65d9368406c2cd7aba038.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/ce2d9687ae99b0ceb1046880fd437f7a801d743669a65d9368406c2cd7aba038.jpg)

![d9d457109f5ea17e86a882a0217918b556f110ac81093dd8d6cc0b929b9e15fb.jpg](../icml_results/6028_Transform%20Trained%20Transformer%20for%20Accelerating%20Native%204K%20Video%20Generation/tables/d9d457109f5ea17e86a882a0217918b556f110ac81093dd8d6cc0b929b9e15fb.jpg)

<a id="paper-5980"></a>

## Cycle-of-Science: Reliable Reasoning through Counterfactual Verification for Agent Decision Making


### Images

![08bfb94a037e64956faea08234b2b6edcc10d3a8bca0c80fafd01c3965294c91.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/08bfb94a037e64956faea08234b2b6edcc10d3a8bca0c80fafd01c3965294c91.jpg)

![26ee822f6214e6877698af0bc62341c3ee2f6903afffd5ea7566bad6e1533bc9.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/26ee822f6214e6877698af0bc62341c3ee2f6903afffd5ea7566bad6e1533bc9.jpg)

![3c4539fe5383a3772b55400b57815e991ec386560f584543f1c7e7e09dffee58.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/3c4539fe5383a3772b55400b57815e991ec386560f584543f1c7e7e09dffee58.jpg)

![43bb2724438ab4e4aa5099fa77cffd116e0e3f747c93eafd551b4214db8c1c1b.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/43bb2724438ab4e4aa5099fa77cffd116e0e3f747c93eafd551b4214db8c1c1b.jpg)

![68c5bb201c94a637fcca7847b09f799fe9b224ec94e34712f9e89a6608cce9a4.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/68c5bb201c94a637fcca7847b09f799fe9b224ec94e34712f9e89a6608cce9a4.jpg)

![a2fc55c3e1c21797835620975859307e34ca357f7a6f22cddf2d734bb595c098.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/a2fc55c3e1c21797835620975859307e34ca357f7a6f22cddf2d734bb595c098.jpg)

![ecc0c081b9976deb0fc49791aed6330128b4b4711afc0c3f1d9d74b205997e95.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/images/ecc0c081b9976deb0fc49791aed6330128b4b4711afc0c3f1d9d74b205997e95.jpg)

### Tables

![0042045d732c51ede5351232f2b607b8b3883640726401220f700c7c6f739402.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/0042045d732c51ede5351232f2b607b8b3883640726401220f700c7c6f739402.jpg)

![0533afe0fea9c2376324522562aa90faa09c2caba8c86939682800c4129cf8f1.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/0533afe0fea9c2376324522562aa90faa09c2caba8c86939682800c4129cf8f1.jpg)

![0d7ca3d04b89d4239e2f41a30d9d46bfca9900ad5ac3c48b21802a3e9193bbed.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/0d7ca3d04b89d4239e2f41a30d9d46bfca9900ad5ac3c48b21802a3e9193bbed.jpg)

![0ead3872669b4f67b0ca3e335f92d76825c2e05d9efff0a97c41b462dcf0c5ad.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/0ead3872669b4f67b0ca3e335f92d76825c2e05d9efff0a97c41b462dcf0c5ad.jpg)

![263faf4c00ee115b21bc6605a95b1371170625b0cb122b8b40a64ab39c04933f.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/263faf4c00ee115b21bc6605a95b1371170625b0cb122b8b40a64ab39c04933f.jpg)

![397d1063a938a6546bca2a817845b3fee71e73c097c7f96b5da191ae77a7c891.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/397d1063a938a6546bca2a817845b3fee71e73c097c7f96b5da191ae77a7c891.jpg)

![561d5e364aaebf7d68eb8279716d4e19c101ba2effa187b9d99d61343d79d63b.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/561d5e364aaebf7d68eb8279716d4e19c101ba2effa187b9d99d61343d79d63b.jpg)

![59f4d654473a862b8f2bd01c70f8e1d4210ef97b5734e90b949fb169f9f8dd51.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/59f4d654473a862b8f2bd01c70f8e1d4210ef97b5734e90b949fb169f9f8dd51.jpg)

![5f2a15abad04c77e9f438ba937926ea4bd97c036e8f29f36bbd0c882b40f55a1.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/5f2a15abad04c77e9f438ba937926ea4bd97c036e8f29f36bbd0c882b40f55a1.jpg)

![66a5683a6b5b7b02b1ac96ab03236babd6df5c530b00e77f66b0576de6e8de62.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/66a5683a6b5b7b02b1ac96ab03236babd6df5c530b00e77f66b0576de6e8de62.jpg)

![86672225065bc4304c4821458a6c7db905c925a1a5d03e89897a55d06a183cf1.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/86672225065bc4304c4821458a6c7db905c925a1a5d03e89897a55d06a183cf1.jpg)

![8ed988d9c8fa5ee9beb2e71d2dfa06bdb7df4181a9f0ab710fec5805364ef572.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/8ed988d9c8fa5ee9beb2e71d2dfa06bdb7df4181a9f0ab710fec5805364ef572.jpg)

![9c0ffd0a194d453ea2e7a4c6d3f96fc98dc21aa9552bec4177563771cd30e4d9.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/9c0ffd0a194d453ea2e7a4c6d3f96fc98dc21aa9552bec4177563771cd30e4d9.jpg)

![a1105c8ef75575bac68f99ce39b79187600d18674375478c7b6730852b02aa9a.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/a1105c8ef75575bac68f99ce39b79187600d18674375478c7b6730852b02aa9a.jpg)

![b2aa97019b55ed39244c198208da358ce191bdf8bfb492a4ea2e28ea5e1acd49.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/b2aa97019b55ed39244c198208da358ce191bdf8bfb492a4ea2e28ea5e1acd49.jpg)

![c40eb4b5a5c601c063eed5eceeb69816a6ca4f8c938e5bf11e20cb249f396732.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/c40eb4b5a5c601c063eed5eceeb69816a6ca4f8c938e5bf11e20cb249f396732.jpg)

![cf9adb4a71fee5922c16f3b8c48a84fc5cee29edefdeb3a5341088ef7d01b648.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/cf9adb4a71fee5922c16f3b8c48a84fc5cee29edefdeb3a5341088ef7d01b648.jpg)

![f413676fbc08e145fa94b5c3f517efa7749ca2b032ab036d08fdf5cbaad9e546.jpg](../icml_results/6029_Cycle-of-Science_%20Reliable%20Reasoning%20through%20Counterfactual%20Verification%20for%20Agent%20Decision%20Making/tables/f413676fbc08e145fa94b5c3f517efa7749ca2b032ab036d08fdf5cbaad9e546.jpg)

<a id="paper-5981"></a>

## Unveiling And Addressing Dimensional Collapse In Vector Quantization Models Via Codebook Regularization


### Images

![3762fbd01bbc6198ad2c2a6196e72e84b00fea194ec87f6ebb9910fad3b50285.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/images/3762fbd01bbc6198ad2c2a6196e72e84b00fea194ec87f6ebb9910fad3b50285.jpg)

![b2965872a70bd791aab987c28a003dd94908b3df1d65fa49339c6884a5efd86c.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/images/b2965872a70bd791aab987c28a003dd94908b3df1d65fa49339c6884a5efd86c.jpg)

![bd39af34e760df584a41571a2fab3bac5f04d1d2a96c66e8ee979ab50eb2ea7e.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/images/bd39af34e760df584a41571a2fab3bac5f04d1d2a96c66e8ee979ab50eb2ea7e.jpg)

### Tables

![2d28420536a0806bd45649f7716b4a5e80a6887a1c2e65627b320d772374b82b.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/2d28420536a0806bd45649f7716b4a5e80a6887a1c2e65627b320d772374b82b.jpg)

![36f49c7c951eb62504115b39d28da8a46d87a4c2e3e0f40d73546c17fae463ee.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/36f49c7c951eb62504115b39d28da8a46d87a4c2e3e0f40d73546c17fae463ee.jpg)

![4daa5f010cb9cd464af20ad21bf9cd3160d54a7280d539284a15b4af0755679d.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/4daa5f010cb9cd464af20ad21bf9cd3160d54a7280d539284a15b4af0755679d.jpg)

![5d4ae89ee1c476c66f476248c9c8330c9dcb43965660d02ae357c115f9c73f19.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/5d4ae89ee1c476c66f476248c9c8330c9dcb43965660d02ae357c115f9c73f19.jpg)

![801aa58cc79cc6eda8e5b30c23e651817dfe461e28ff274b9ca8ba969dcf4a6a.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/801aa58cc79cc6eda8e5b30c23e651817dfe461e28ff274b9ca8ba969dcf4a6a.jpg)

![942ac1de122440241f591c389f58b09c3b8bba77c3a980f000379d9ed0f0499f.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/942ac1de122440241f591c389f58b09c3b8bba77c3a980f000379d9ed0f0499f.jpg)

![962dc84a37953f644d820b4559fbc77d90080aa675b9f243ce94ca554f5c13ca.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/962dc84a37953f644d820b4559fbc77d90080aa675b9f243ce94ca554f5c13ca.jpg)

![97823dd2a7abea1871a08fae0d29b869ff733cf9915af889353a20c2ac2aec68.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/97823dd2a7abea1871a08fae0d29b869ff733cf9915af889353a20c2ac2aec68.jpg)

![f9e2f18b75da9f6efe0fa96637e8af744acc6da0cd2f673a6daf2c85b647214b.jpg](../icml_results/6030_Unveiling%20And%20Addressing%20Dimensional%20Collapse%20In%20Vector%20Quantization%20Models%20Via%20Codebook%20Regularizat/tables/f9e2f18b75da9f6efe0fa96637e8af744acc6da0cd2f673a6daf2c85b647214b.jpg)

<a id="paper-5982"></a>

## Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR


### Images

![0db3897158d7b8e9c5e6e7412eb2dda5acf27fce3d4b4bd211a825633acfce72.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/0db3897158d7b8e9c5e6e7412eb2dda5acf27fce3d4b4bd211a825633acfce72.jpg)

![1a003d1ce273d0d65fde01297340b059db91a82c7de6f67f7b5c4c919f882de3.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/1a003d1ce273d0d65fde01297340b059db91a82c7de6f67f7b5c4c919f882de3.jpg)

![3edab6987a3799df8c49c8e9cc0ccf4b623570bfff46fd56c3c4cc3397a3d815.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/3edab6987a3799df8c49c8e9cc0ccf4b623570bfff46fd56c3c4cc3397a3d815.jpg)

![680206165b4eb2534c3f3c57c45ec9d1aeb181709e1f2c186c5b98a6de8e37a8.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/680206165b4eb2534c3f3c57c45ec9d1aeb181709e1f2c186c5b98a6de8e37a8.jpg)

![77169ce6a274a00547503600985725e463789b29755b30308277de5062129354.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/77169ce6a274a00547503600985725e463789b29755b30308277de5062129354.jpg)

![7846803542d18762ceb14a1b44791fe86537599662817ac60770fe408f9e1738.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/7846803542d18762ceb14a1b44791fe86537599662817ac60770fe408f9e1738.jpg)

![80cf7a4e5562531601e8e4de542b8e54261c97e7cf45faf83907e8539a0eceab.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/80cf7a4e5562531601e8e4de542b8e54261c97e7cf45faf83907e8539a0eceab.jpg)

![a3f89db72fce25c4e76342b1392c812a62a61812da2ca7905ee3913c957a6735.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/a3f89db72fce25c4e76342b1392c812a62a61812da2ca7905ee3913c957a6735.jpg)

![aea5438da494464995c3e968655f9074dcfa899db0a52c59b21db38d2e8810ee.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/aea5438da494464995c3e968655f9074dcfa899db0a52c59b21db38d2e8810ee.jpg)

![b67b9b0ca50e7bf8f67b6568cef290fb69507d5607b43c22eafea9b5ebb28c58.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/b67b9b0ca50e7bf8f67b6568cef290fb69507d5607b43c22eafea9b5ebb28c58.jpg)

![ebcf5a0ab9efb20a3567c589ecad273d99cc6b14d8ff2aaf730b9c9e12e4655e.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/images/ebcf5a0ab9efb20a3567c589ecad273d99cc6b14d8ff2aaf730b9c9e12e4655e.jpg)

### Tables

![37bcd291e8de4e3c47ddae61e51880e863d0f413ea7fbc63bf0b956e855d8823.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/37bcd291e8de4e3c47ddae61e51880e863d0f413ea7fbc63bf0b956e855d8823.jpg)

![39250963fe4c9e1730da79d280951b23e5c29a6b5292b1f1376398208c2a1f5e.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/39250963fe4c9e1730da79d280951b23e5c29a6b5292b1f1376398208c2a1f5e.jpg)

![4308916dcfc43064ee78c2c1d48b59a9a39015a3a61962eb1c97d3b5d135ae52.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/4308916dcfc43064ee78c2c1d48b59a9a39015a3a61962eb1c97d3b5d135ae52.jpg)

![4d4ab6f03a6d9b2ccb52310f9b3cb0cce157f27d7ba968a5f01cdaf9f9739e6f.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/4d4ab6f03a6d9b2ccb52310f9b3cb0cce157f27d7ba968a5f01cdaf9f9739e6f.jpg)

![514914f5c9948c95d18b9b79feeecbb2a4e75dc5629bec9dd5d515c44c2df6a7.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/514914f5c9948c95d18b9b79feeecbb2a4e75dc5629bec9dd5d515c44c2df6a7.jpg)

![a2daa9a702408963c7da545b638bad0622fa1fc6fd507c8067c96b641cbb998f.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/a2daa9a702408963c7da545b638bad0622fa1fc6fd507c8067c96b641cbb998f.jpg)

![ba1eb064f0bdc6098c02d3c364237651378a376a5bac816d62aacb3fae0d8b0d.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/ba1eb064f0bdc6098c02d3c364237651378a376a5bac816d62aacb3fae0d8b0d.jpg)

![fafe9de7116816aff77f280db8cf0f1f461fb7f95633db17ad16f733360f8a68.jpg](../icml_results/6031_Geometry-Preserving%20Orthonormal%20Initialization%20for%20Low-Rank%20Adaptation%20in%20RLVR/tables/fafe9de7116816aff77f280db8cf0f1f461fb7f95633db17ad16f733360f8a68.jpg)

<a id="paper-5983"></a>

## GRAPE: Let GRPO Supervise Query Rewriting by Ranking for Retrieval


### Images

![39a122efa3d10d74302d4ce351f0416f51fc0c130fdc955feb89be61530d9695.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/39a122efa3d10d74302d4ce351f0416f51fc0c130fdc955feb89be61530d9695.jpg)

![82f135870aeb29dccbe86f6aa7cdffcd7fc2a317e826ec683e93ee0c0d08e7aa.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/82f135870aeb29dccbe86f6aa7cdffcd7fc2a317e826ec683e93ee0c0d08e7aa.jpg)

![9537afcfb5f70518f16650ef3e00fa91b39d548caa85945499dd509f7ce13240.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/9537afcfb5f70518f16650ef3e00fa91b39d548caa85945499dd509f7ce13240.jpg)

![9708c7f16c68538c91eecec7379cfe9702d710e881b40554f185d91117a2cb34.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/9708c7f16c68538c91eecec7379cfe9702d710e881b40554f185d91117a2cb34.jpg)

![a6d4622d35b8675a2633cc73ffcdeaa2cbd3ee450b32c277c6fee3473870357a.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/a6d4622d35b8675a2633cc73ffcdeaa2cbd3ee450b32c277c6fee3473870357a.jpg)

![d9897fd9bcd3a9544b6141089a2c4a3a510213a31b921d3d31c2f192fe76afde.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/d9897fd9bcd3a9544b6141089a2c4a3a510213a31b921d3d31c2f192fe76afde.jpg)

![e8f337be438b79aaed55d66550b5b1614c6b163dcd31c32ff73573dd7448677d.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/images/e8f337be438b79aaed55d66550b5b1614c6b163dcd31c32ff73573dd7448677d.jpg)

### Tables

![1264467cd55d8fca3e24929bb08ec775e30d5284a472ff66b1d0f0c80bb140a0.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/1264467cd55d8fca3e24929bb08ec775e30d5284a472ff66b1d0f0c80bb140a0.jpg)

![1ae657dfa5fcc80505639359a7cc86bfd9a79e3e21454eb7bee85bdb1011f656.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/1ae657dfa5fcc80505639359a7cc86bfd9a79e3e21454eb7bee85bdb1011f656.jpg)

![bc59e76bdf2b316df75442e796f075013b7a25c80a240ea8fecc7ff0ca1bc293.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/bc59e76bdf2b316df75442e796f075013b7a25c80a240ea8fecc7ff0ca1bc293.jpg)

![c3c41f9324ea5b203a8f72c5c122e55b981eeec37dd7cb6b8cf5e42eaa36188f.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/c3c41f9324ea5b203a8f72c5c122e55b981eeec37dd7cb6b8cf5e42eaa36188f.jpg)

![dc8f92fbe06dce70bea352d2d50b190a334b45d3e78e8aa02455c36bee37e6fa.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/dc8f92fbe06dce70bea352d2d50b190a334b45d3e78e8aa02455c36bee37e6fa.jpg)

![e4a304527dd03dbbfa371c541eb952ef2704fa4d56484094235b7b5cb4d936df.jpg](../icml_results/6032_GRAPE_%20Let%20GRPO%20Supervise%20Query%20Rewriting%20by%20Ranking%20for%20Retrieval/tables/e4a304527dd03dbbfa371c541eb952ef2704fa4d56484094235b7b5cb4d936df.jpg)

<a id="paper-5984"></a>

## Learnability-Driven Knowledge Assimilation for Class-Incremental Semantic Segmentation


### Images

![26c090fb8490c05b799f95300c2a26a5806c6f183cff0aa5e202d5b6468ca5c0.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/26c090fb8490c05b799f95300c2a26a5806c6f183cff0aa5e202d5b6468ca5c0.jpg)

![29c8283c8018b6ff70beb4af9450d6e38b74dc0acd47c872181d6aa34ddb31cd.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/29c8283c8018b6ff70beb4af9450d6e38b74dc0acd47c872181d6aa34ddb31cd.jpg)

![2a147e81604d11ab6c49cf84dbdeb82abfd743a75ef81f0108c81c57ea555ba9.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/2a147e81604d11ab6c49cf84dbdeb82abfd743a75ef81f0108c81c57ea555ba9.jpg)

![2ec46c4817e9202b79134a0aaf6471ea64d80fdff23a920a5763902bea510095.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/2ec46c4817e9202b79134a0aaf6471ea64d80fdff23a920a5763902bea510095.jpg)

![4a59214695b457b5d9cf3f82eaf4e4c417a712bcd60035dd345657c12da31d91.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/4a59214695b457b5d9cf3f82eaf4e4c417a712bcd60035dd345657c12da31d91.jpg)

![63914c4ac8578d604a4bce13a0f70e8b7c243ed04599d0cab152a2da5c9bc54e.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/63914c4ac8578d604a4bce13a0f70e8b7c243ed04599d0cab152a2da5c9bc54e.jpg)

![6a56a2396d86ce981e9a926e7fadbdfbc86fdce4830a74a99183f2f2d4cf9920.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/6a56a2396d86ce981e9a926e7fadbdfbc86fdce4830a74a99183f2f2d4cf9920.jpg)

![7337db7102ea1cd556b8ce4ace2c16a79da3c4d414f3ae017f6f05ffa12378f8.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/7337db7102ea1cd556b8ce4ace2c16a79da3c4d414f3ae017f6f05ffa12378f8.jpg)

![7acb3b64cd8af9c1bbaea9510a5fafc6e6e27097d394e9330cad1c7e2ba18445.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/7acb3b64cd8af9c1bbaea9510a5fafc6e6e27097d394e9330cad1c7e2ba18445.jpg)

![c043eb2d3f001c7ea4069068399139eb12e9d4675ec879badfc1e2eb83409df0.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/c043eb2d3f001c7ea4069068399139eb12e9d4675ec879badfc1e2eb83409df0.jpg)

![c044eedf3c63f6837d4ffb1f0c2a729d91a5fb7e8338ffe7e6c4d3fb8f6eee6e.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/c044eedf3c63f6837d4ffb1f0c2a729d91a5fb7e8338ffe7e6c4d3fb8f6eee6e.jpg)

![efd616e926aab429e8f5e309fe7df19d8545aceeac7a525c5802d017d9d9d5d7.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/images/efd616e926aab429e8f5e309fe7df19d8545aceeac7a525c5802d017d9d9d5d7.jpg)

### Tables

![01eab0d6bec205626043a8f0fe63cb06921604359d894639253cd8b1b5b3c249.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/01eab0d6bec205626043a8f0fe63cb06921604359d894639253cd8b1b5b3c249.jpg)

![12633e54b28d78a4c33553998e29ec270f96158e252fb10039cd33d425ed8b2c.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/12633e54b28d78a4c33553998e29ec270f96158e252fb10039cd33d425ed8b2c.jpg)

![1eb0d7fe4306aa3d752bbc67063009cf447edde0a2d4712e333225904ea5b386.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/1eb0d7fe4306aa3d752bbc67063009cf447edde0a2d4712e333225904ea5b386.jpg)

![221451978fd99e3a6e6dcc941347767295d87d9986099794dd8c2b3beac13a0c.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/221451978fd99e3a6e6dcc941347767295d87d9986099794dd8c2b3beac13a0c.jpg)

![23514f0ff6101ee2f7ede6a041eb4fcf18f50f9f92554d631272723817322d72.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/23514f0ff6101ee2f7ede6a041eb4fcf18f50f9f92554d631272723817322d72.jpg)

![38dec5d0142c3ca739336681bf87051c71e0ab613f077a557482b52d56706ffb.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/38dec5d0142c3ca739336681bf87051c71e0ab613f077a557482b52d56706ffb.jpg)

![3fad44a401a4025381c2b4e4c0df163733e90b9b6684bbadbf04f8dec601c071.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/3fad44a401a4025381c2b4e4c0df163733e90b9b6684bbadbf04f8dec601c071.jpg)

![5b20b364df1f3c8ce992732474532d60437948b9b4c9309054dd88ca6f9ac749.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/5b20b364df1f3c8ce992732474532d60437948b9b4c9309054dd88ca6f9ac749.jpg)

![74d23ceeab3cf64a3a556706b83936cf64105606a1d044cfcc0a87db9aa7b38b.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/74d23ceeab3cf64a3a556706b83936cf64105606a1d044cfcc0a87db9aa7b38b.jpg)

![9be1d089a53a3f89dccd4fe07d9ed6a68186969632fb684723f51ac0cff46136.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/9be1d089a53a3f89dccd4fe07d9ed6a68186969632fb684723f51ac0cff46136.jpg)

![d28f05119e39616a5ebf5852260e972d260ccdec63fa9f1bb87c5f28998f3a3a.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/d28f05119e39616a5ebf5852260e972d260ccdec63fa9f1bb87c5f28998f3a3a.jpg)

![eb23b74418797acc066359c33254f5b5093be47da850355d5f51dcc3876c6458.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/eb23b74418797acc066359c33254f5b5093be47da850355d5f51dcc3876c6458.jpg)

![feec04c23ff5b1b1838946f24292f19422f18c1219d5a990a136377a9776910c.jpg](../icml_results/6033_Learnability-Driven%20Knowledge%20Assimilation%20for%20Class-Incremental%20Semantic%20Segmentation/tables/feec04c23ff5b1b1838946f24292f19422f18c1219d5a990a136377a9776910c.jpg)

<a id="paper-5985"></a>

## Think in Latent, Explain in Language: Self-Explainable Latent Reasoning


### Images

![11f87d8d89af5ebacae5a7e42a0d35c0463991fe10d42bfdadd0fddf383b5a8f.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/11f87d8d89af5ebacae5a7e42a0d35c0463991fe10d42bfdadd0fddf383b5a8f.jpg)

![18954117efa780c8a1462de19186855f4ea99bc74a214dda0450c3588b8ec4b4.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/18954117efa780c8a1462de19186855f4ea99bc74a214dda0450c3588b8ec4b4.jpg)

![2f45af504a9c046b45a2e447d0c20abb5f7a6721070dd08a95a1e1fb409663bf.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/2f45af504a9c046b45a2e447d0c20abb5f7a6721070dd08a95a1e1fb409663bf.jpg)

![5bfa02c76354b7ef2f7ebbeaf9ff56530934cd1fa3f3b38d1d1339993941322c.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/5bfa02c76354b7ef2f7ebbeaf9ff56530934cd1fa3f3b38d1d1339993941322c.jpg)

![7f90c822aad17e0cd5a1058a42cf7fc5527922479231d1cfbc04e569eb277f58.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/7f90c822aad17e0cd5a1058a42cf7fc5527922479231d1cfbc04e569eb277f58.jpg)

![8b9c6dfa144fcaa2766298ccec52bc5d25a61f29eeba791e26c1579e685e7149.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/8b9c6dfa144fcaa2766298ccec52bc5d25a61f29eeba791e26c1579e685e7149.jpg)

![a328699b0d9ed4c813d1d16d0ab472d814908f70632c24f806bd8be6a209ecc3.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/a328699b0d9ed4c813d1d16d0ab472d814908f70632c24f806bd8be6a209ecc3.jpg)

![b6efb933701cd266abfaccb7b2bc874c52a3efa26b10d397a6366c0c04838840.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/b6efb933701cd266abfaccb7b2bc874c52a3efa26b10d397a6366c0c04838840.jpg)

![d751b689a143f87fec4829acc059b1082e443afebbfe8783a2966e09e6f96eb0.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/images/d751b689a143f87fec4829acc059b1082e443afebbfe8783a2966e09e6f96eb0.jpg)

### Tables

![1ea4156fb7e81d1b7b9c302c769e55b8279b749ec3329e7182b6a9ab36b463e0.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/1ea4156fb7e81d1b7b9c302c769e55b8279b749ec3329e7182b6a9ab36b463e0.jpg)

![244116b34888523063ce738b8ef2767924151dec31f47ac10b6a7f2f80a289f1.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/244116b34888523063ce738b8ef2767924151dec31f47ac10b6a7f2f80a289f1.jpg)

![31b5702387eabe6c2179078918613db6c265194bd0637a3d27a90048ebf5628f.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/31b5702387eabe6c2179078918613db6c265194bd0637a3d27a90048ebf5628f.jpg)

![376021fbaebe9b703e7559313f1f9009f9fd5f982c5b06ef420f7cb30536b6db.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/376021fbaebe9b703e7559313f1f9009f9fd5f982c5b06ef420f7cb30536b6db.jpg)

![65dca0bb2082e773c689ef0a2bc67e7943a9b593a0630848447733b39125fb02.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/65dca0bb2082e773c689ef0a2bc67e7943a9b593a0630848447733b39125fb02.jpg)

![67d20360d7706c9919684b72fba536b880ff8cf851c82b6287dd18eb9414fc3e.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/67d20360d7706c9919684b72fba536b880ff8cf851c82b6287dd18eb9414fc3e.jpg)

![8265844b2535cafe1fc35ca4a3da5d370e6b79aafa79dc0adbddac00694ef7e9.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/8265844b2535cafe1fc35ca4a3da5d370e6b79aafa79dc0adbddac00694ef7e9.jpg)

![9f2e2be8440ebc94ac1874a8551d9784b116248376a0599748e6c178cf4189f5.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/9f2e2be8440ebc94ac1874a8551d9784b116248376a0599748e6c178cf4189f5.jpg)

![a0858e71e5b222c1e12774a26da4b0bcac911e125ae9b3c4a25b6f78e25ce140.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/a0858e71e5b222c1e12774a26da4b0bcac911e125ae9b3c4a25b6f78e25ce140.jpg)

![a5630ead8512e93351681eb5c9bb8a03110417bd0264756e5e374c09f5683941.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/a5630ead8512e93351681eb5c9bb8a03110417bd0264756e5e374c09f5683941.jpg)

![ae056fe771013e3b61fde5c6c45aa6a6b58d914834515b05527214d2b5ea4d72.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/ae056fe771013e3b61fde5c6c45aa6a6b58d914834515b05527214d2b5ea4d72.jpg)

![cb44d7d7b0368173ca2bbb0696a0b2a06eae532179a22c8b878ba711dbba370e.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/cb44d7d7b0368173ca2bbb0696a0b2a06eae532179a22c8b878ba711dbba370e.jpg)

![da5b46a337c9f38998bf70fedee1f0db11bfa558c143501cb3255b24d6589e63.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/da5b46a337c9f38998bf70fedee1f0db11bfa558c143501cb3255b24d6589e63.jpg)

![e92243d36c40793a39652957647c1b809ab396ad14cfd5621a998415aff7ff8b.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/e92243d36c40793a39652957647c1b809ab396ad14cfd5621a998415aff7ff8b.jpg)

![ea3f6065d7804c6a02a43a379c1b74d1be427174fb4f52dee8528e576b1912cd.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/ea3f6065d7804c6a02a43a379c1b74d1be427174fb4f52dee8528e576b1912cd.jpg)

![ea587ec55249c47a32400a78cbaa60e97d7c03448383e3af1d2a6ee4b1976afc.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/ea587ec55249c47a32400a78cbaa60e97d7c03448383e3af1d2a6ee4b1976afc.jpg)

![ece4ead3dd1e87d672c8d62db8f06264dafbee42ad24f36698b50f0f95405a7f.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/ece4ead3dd1e87d672c8d62db8f06264dafbee42ad24f36698b50f0f95405a7f.jpg)

![ef0646ea62f259ee8a17f95bef41554cf4eecc12cb8b49bdf24fe8afb0e7b788.jpg](../icml_results/6034_Think%20in%20Latent%2C%20Explain%20in%20Language_%20Self-Explainable%20Latent%20Reasoning/tables/ef0646ea62f259ee8a17f95bef41554cf4eecc12cb8b49bdf24fe8afb0e7b788.jpg)

<a id="paper-5986"></a>

## Population-Free Pareto Tracking for Sample-Efficient Multi-Policy MORL


### Images

![32e207377817a0fe5ed23a139ae36fe2607783757f9060d72aa3b9962d596143.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/32e207377817a0fe5ed23a139ae36fe2607783757f9060d72aa3b9962d596143.jpg)

![59cbf7daf574eb54df5e728a437f7372bcc9daa23ab38c0be687779741f5b10d.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/59cbf7daf574eb54df5e728a437f7372bcc9daa23ab38c0be687779741f5b10d.jpg)

![5e15e368271141ed15201a0bd3bf845e1cb2a5386e244b156acac30a65fb99d1.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/5e15e368271141ed15201a0bd3bf845e1cb2a5386e244b156acac30a65fb99d1.jpg)

![68e2bcb39f8a40beca98e51a6c8fe8de01503590a3004c8d410ee81573d7c106.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/68e2bcb39f8a40beca98e51a6c8fe8de01503590a3004c8d410ee81573d7c106.jpg)

![72264f564dea881985223751ca092ea7221509e69d16227a778024ae39907a1e.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/72264f564dea881985223751ca092ea7221509e69d16227a778024ae39907a1e.jpg)

![76d689c24be30b1a0d3e6633791f88c2dbe99f0d8bb3e600b8c76f31981abf8e.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/76d689c24be30b1a0d3e6633791f88c2dbe99f0d8bb3e600b8c76f31981abf8e.jpg)

![98cbd897541573eaf5683c1a929348daef9f739e9d3b7d5966586afe3ff4b7b7.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/98cbd897541573eaf5683c1a929348daef9f739e9d3b7d5966586afe3ff4b7b7.jpg)

![9cc7cdb98637b83c587bc34b053fcfebe5c30a26b81108c8e555dde759ffb01b.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/9cc7cdb98637b83c587bc34b053fcfebe5c30a26b81108c8e555dde759ffb01b.jpg)

![aae8b603e5ea973041ef89d90a9186e076ce7660384a64b2f76b528cc93e120a.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/aae8b603e5ea973041ef89d90a9186e076ce7660384a64b2f76b528cc93e120a.jpg)

![e2fa6693523651c53c46e236c0c4d84d6a51191a5886f5005e0f9374a38b559e.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/images/e2fa6693523651c53c46e236c0c4d84d6a51191a5886f5005e0f9374a38b559e.jpg)

### Tables

![076a37e980b2525e8b5f341f7b52f861e0995a67c60eb10be5b3dd6bed051815.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/076a37e980b2525e8b5f341f7b52f861e0995a67c60eb10be5b3dd6bed051815.jpg)

![14e35f57725f40ea42e471b508da8d215e803def099783f2282cc96a30112e63.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/14e35f57725f40ea42e471b508da8d215e803def099783f2282cc96a30112e63.jpg)

![3188c4396558f8513f6ff41aa023e5cd1cdf51a43ccc30f5e75c7b8193571108.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/3188c4396558f8513f6ff41aa023e5cd1cdf51a43ccc30f5e75c7b8193571108.jpg)

![32afc88b5c16bab0ed762ebdc1b9168033caa2b85a4e98f7edd57ca9f097decc.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/32afc88b5c16bab0ed762ebdc1b9168033caa2b85a4e98f7edd57ca9f097decc.jpg)

![59986e6a47b3a2002f3832dd028f9c2c2f57357c016d97da15c2dd33794f52db.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/59986e6a47b3a2002f3832dd028f9c2c2f57357c016d97da15c2dd33794f52db.jpg)

![5bb88a1e400ef4fa6350a529e8d8dd4964bbe4146fd7495b0ff08a9302d5db7c.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/5bb88a1e400ef4fa6350a529e8d8dd4964bbe4146fd7495b0ff08a9302d5db7c.jpg)

![5d6cec4b250a94fc36dd4450f1802bedad6c0b2b4e6137060e35f5b35dcaf3a2.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/5d6cec4b250a94fc36dd4450f1802bedad6c0b2b4e6137060e35f5b35dcaf3a2.jpg)

![6696cf63a07368484da35860a6c7d4673e63df02c6b542639efde14c5f1389d3.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/6696cf63a07368484da35860a6c7d4673e63df02c6b542639efde14c5f1389d3.jpg)

![691fa171cb3acaa873673363d913e97e35edc63c35f06a231fde5f6ac34c7c0c.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/691fa171cb3acaa873673363d913e97e35edc63c35f06a231fde5f6ac34c7c0c.jpg)

![6c63ba6d87d20d41736c3e109cd7449a71779eb5982ebc6b6735414189e9a421.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/6c63ba6d87d20d41736c3e109cd7449a71779eb5982ebc6b6735414189e9a421.jpg)

![7404e586e2cc55760bccbc45ff32bd4c7383fd3127d120c10412266383cc2c59.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/7404e586e2cc55760bccbc45ff32bd4c7383fd3127d120c10412266383cc2c59.jpg)

![7d2f43a255922d536ecf37686ceae309e842a40d7d889dab2bc4241860f66172.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/7d2f43a255922d536ecf37686ceae309e842a40d7d889dab2bc4241860f66172.jpg)

![7df9d1a6c00c1dc8c94f1d6ea95d76a5cb22646ef1560d59b10d882491053e67.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/7df9d1a6c00c1dc8c94f1d6ea95d76a5cb22646ef1560d59b10d882491053e67.jpg)

![8d64a7c267bf4cb52a89842c348a6fe2608cff327f9ab10607e8f20987585fc0.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/8d64a7c267bf4cb52a89842c348a6fe2608cff327f9ab10607e8f20987585fc0.jpg)

![a509685386ce23fb4c445cb17b5a582ea27e3137dd8252b8bc680c8b5dac4298.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/a509685386ce23fb4c445cb17b5a582ea27e3137dd8252b8bc680c8b5dac4298.jpg)

![b128994d7dabd58b00360a0e910c0ea74737831eaebc8a9a55eb9a787193fe4f.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/b128994d7dabd58b00360a0e910c0ea74737831eaebc8a9a55eb9a787193fe4f.jpg)

![e602f3987893e53adb857a46e0f13c043dc5aeefbc177e9dbb2b8d51d1b826d2.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/e602f3987893e53adb857a46e0f13c043dc5aeefbc177e9dbb2b8d51d1b826d2.jpg)

![f52f4e2ae17ed0e99ba0e861e11a5a84f6007907e9ceca1b89aa11b10828acd0.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/f52f4e2ae17ed0e99ba0e861e11a5a84f6007907e9ceca1b89aa11b10828acd0.jpg)

![ff34dda61e1ca6252c005fed5e704bbe284577fc6084610f1825a18a7ca7c60b.jpg](../icml_results/6035_Population-Free%20Pareto%20Tracking%20for%20Sample-Efficient%20Multi-Policy%20MORL/tables/ff34dda61e1ca6252c005fed5e704bbe284577fc6084610f1825a18a7ca7c60b.jpg)

<a id="paper-5987"></a>

## Learning GUI Grounding with Spatial Reasoning from Visual Feedback


### Images

![01874d2ef3b6ab06d7ea8bb2c30f37aff09401d8ee8fa989fa4a834cf3aa2a95.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/01874d2ef3b6ab06d7ea8bb2c30f37aff09401d8ee8fa989fa4a834cf3aa2a95.jpg)

![0c29b4ed20e160495f30d158f94dbbd971900a89e1b079353da5368a3b63bfb2.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/0c29b4ed20e160495f30d158f94dbbd971900a89e1b079353da5368a3b63bfb2.jpg)

![0d258b31ac3a2fe736e40f1021b2f41d826fa860e70d70a360af3ddebf20f338.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/0d258b31ac3a2fe736e40f1021b2f41d826fa860e70d70a360af3ddebf20f338.jpg)

![2a319708bfb3bb77a5751de217688a5fcfb5c06aa45b28dd6ebf1f7d09448fe1.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/2a319708bfb3bb77a5751de217688a5fcfb5c06aa45b28dd6ebf1f7d09448fe1.jpg)

![4a36f974d5a2e92ab65bdc9b03784fe9a50348aafc43b6ac3728b60531bf27fe.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/4a36f974d5a2e92ab65bdc9b03784fe9a50348aafc43b6ac3728b60531bf27fe.jpg)

![4e9fc94db1a60e5811a2ec26ac87619bb0e762165628e93da7aa1606855d3842.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/4e9fc94db1a60e5811a2ec26ac87619bb0e762165628e93da7aa1606855d3842.jpg)

![596150a2582aa1ac0f1ce7dac4829e3e25410ffab73cc2b4b7ac3ec39558a389.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/596150a2582aa1ac0f1ce7dac4829e3e25410ffab73cc2b4b7ac3ec39558a389.jpg)

![6d1cc5da6abd17de1295d6c60a33a9ec5e448b5128dc15b4cd4b626a64a4a310.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/6d1cc5da6abd17de1295d6c60a33a9ec5e448b5128dc15b4cd4b626a64a4a310.jpg)

![726313ba71875b5b915ce85f03db073896d99da8d51d5be67e8675b1e87c53f9.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/726313ba71875b5b915ce85f03db073896d99da8d51d5be67e8675b1e87c53f9.jpg)

![792917353123f34c8f370e1281c34586d334082a2a2fe433fc31944b3e265c9a.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/792917353123f34c8f370e1281c34586d334082a2a2fe433fc31944b3e265c9a.jpg)

![8cbe0825e478bdb31238e4a95f12b781c8304e3e6ea9730f494e76ae6d9c68a7.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/8cbe0825e478bdb31238e4a95f12b781c8304e3e6ea9730f494e76ae6d9c68a7.jpg)

![9aa2df8a4a97411138b1f612bb3f22f0dae2e6489b0c6fe61dea4a05efb74b8b.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/9aa2df8a4a97411138b1f612bb3f22f0dae2e6489b0c6fe61dea4a05efb74b8b.jpg)

![a62a0092dcbd72688644a5a4dc87cbc954bd58717b4541871a4585ad9fddbb77.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/a62a0092dcbd72688644a5a4dc87cbc954bd58717b4541871a4585ad9fddbb77.jpg)

![acfb707afb658edf3b646f03685962be7551ea80668c2b185b6b778837b07fff.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/acfb707afb658edf3b646f03685962be7551ea80668c2b185b6b778837b07fff.jpg)

![b104ca5cea716781a0ec6e522a6236f34b199393e83563bc8054afc379a836f0.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/b104ca5cea716781a0ec6e522a6236f34b199393e83563bc8054afc379a836f0.jpg)

![de450014602324d039b83518c8ecb964cba38e5430163b8f5a2efb95199b193b.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/de450014602324d039b83518c8ecb964cba38e5430163b8f5a2efb95199b193b.jpg)

![de52767869e384e7dce0783b73a10d584e5103cbade88b6d06c34f158ad01f15.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/de52767869e384e7dce0783b73a10d584e5103cbade88b6d06c34f158ad01f15.jpg)

![e6733f175fced70dda1acfa63c7bf31ff07cfda32faf95f4a1ac7b8d70509390.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/e6733f175fced70dda1acfa63c7bf31ff07cfda32faf95f4a1ac7b8d70509390.jpg)

![f9e4ee38286e8b61f88dfe0cd9254adc4ac949491eef783aef2b5b93efb26e90.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/images/f9e4ee38286e8b61f88dfe0cd9254adc4ac949491eef783aef2b5b93efb26e90.jpg)

### Tables

![1be6053da744799b3c66c021f30de35c0908738941fe583900a6849b91aa4e2f.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/1be6053da744799b3c66c021f30de35c0908738941fe583900a6849b91aa4e2f.jpg)

![216513507272e4a4bcbdecd223e416f421456d91f6ce4ab4b7d0336c6a103e30.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/216513507272e4a4bcbdecd223e416f421456d91f6ce4ab4b7d0336c6a103e30.jpg)

![24cb14b7602d4cff83edfb6f14c38115ad82b8dde226911c34bca088a118e0be.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/24cb14b7602d4cff83edfb6f14c38115ad82b8dde226911c34bca088a118e0be.jpg)

![33796a9f1ab7ac97ed290b8def49fcf7b023333b9d3152adfeab8b5b0f3b187c.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/33796a9f1ab7ac97ed290b8def49fcf7b023333b9d3152adfeab8b5b0f3b187c.jpg)

![41ec8fd20038dcde21377f65ff8b3e0a68f8027045e106f5bd042154f7037c5c.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/41ec8fd20038dcde21377f65ff8b3e0a68f8027045e106f5bd042154f7037c5c.jpg)

![90ff92fd2d25c0f35078c23df48d714e0dc08c95310026152c602fdbb67211a9.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/90ff92fd2d25c0f35078c23df48d714e0dc08c95310026152c602fdbb67211a9.jpg)

![b6bfe9f3eca95c59d6bcf9ed09b0fd4127b633610cc2174a66b5a3db0d162513.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/b6bfe9f3eca95c59d6bcf9ed09b0fd4127b633610cc2174a66b5a3db0d162513.jpg)

![b99140a63623e844858a0f4782b14a826a1d657097950486bd25075d2ec5069c.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/b99140a63623e844858a0f4782b14a826a1d657097950486bd25075d2ec5069c.jpg)

![c79cd65a917d540696b00aa5cb46e6953d914c3136aa4d978c6e1ac356516ee5.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/c79cd65a917d540696b00aa5cb46e6953d914c3136aa4d978c6e1ac356516ee5.jpg)

![d03c8657b1d329c2e54a015215a574570ff30e80be049d6290e5b8cd9320db28.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/d03c8657b1d329c2e54a015215a574570ff30e80be049d6290e5b8cd9320db28.jpg)

![ecff1ccfed337950ed011541e7c6fc5ab08aaa8fc0252257047ff79236929296.jpg](../icml_results/6036_Learning%20GUI%20Grounding%20with%20Spatial%20Reasoning%20from%20Visual%20Feedback/tables/ecff1ccfed337950ed011541e7c6fc5ab08aaa8fc0252257047ff79236929296.jpg)

<a id="paper-5988"></a>

## LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts


### Images

![07b8961119b17edb9e3638ae7fc57444e7209c95d14da957aae729a4b0a58599.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/07b8961119b17edb9e3638ae7fc57444e7209c95d14da957aae729a4b0a58599.jpg)

![1da3ef61ab4a4e421bf01de1a9fd0c5bd76f31120a5743fddf6abd1d70e2c53d.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/1da3ef61ab4a4e421bf01de1a9fd0c5bd76f31120a5743fddf6abd1d70e2c53d.jpg)

![2af71fe4d7606ce7e99f7e5e24219185d797d4d706c72bfaef35ebb1ac60d3a5.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/2af71fe4d7606ce7e99f7e5e24219185d797d4d706c72bfaef35ebb1ac60d3a5.jpg)

![53e00b6d8f4a17e159d369e1713e090dfa74ecd81a497d3da5d51012ea2e3f61.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/53e00b6d8f4a17e159d369e1713e090dfa74ecd81a497d3da5d51012ea2e3f61.jpg)

![5dc7d9a8a82895e15c602d2d1c372aee0715ba0a5bdb15a6ccae82ceb33e6151.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/5dc7d9a8a82895e15c602d2d1c372aee0715ba0a5bdb15a6ccae82ceb33e6151.jpg)

![66371eab36f7e9d9859f15f90bd6b18ce35275c72a8d33300338384367f4d021.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/66371eab36f7e9d9859f15f90bd6b18ce35275c72a8d33300338384367f4d021.jpg)

![70e15d911b8aa5d5b548c0358eeb71af6789a4d8964dd7296f2d600b2e0715f4.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/70e15d911b8aa5d5b548c0358eeb71af6789a4d8964dd7296f2d600b2e0715f4.jpg)

![7d3437f125bfad8829d0b8c4044b0609c60f0cb44528f4d2bf1b423a1f31bdbe.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/7d3437f125bfad8829d0b8c4044b0609c60f0cb44528f4d2bf1b423a1f31bdbe.jpg)

![7e6963b83565f74d83793a390db035c66ed8de0e5363d218376be87ba6dab4a2.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/7e6963b83565f74d83793a390db035c66ed8de0e5363d218376be87ba6dab4a2.jpg)

![83fcf2a3dc714f14b1c51b89fe1e17288cd727db0ccd9af792ed9e0d3150ecf6.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/83fcf2a3dc714f14b1c51b89fe1e17288cd727db0ccd9af792ed9e0d3150ecf6.jpg)

![94d7e91959abbba66e254410ce5b222fe6c3c3dd675121353d6d4ec7548e24b0.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/94d7e91959abbba66e254410ce5b222fe6c3c3dd675121353d6d4ec7548e24b0.jpg)

![c240e1ffe0c10b484f6dc8effb1f41d2bdf5218e1ceb398ab4f30fdfc593aa7c.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/c240e1ffe0c10b484f6dc8effb1f41d2bdf5218e1ceb398ab4f30fdfc593aa7c.jpg)

![c3483ed4f22a34bf93a7b1959ff89262e1cd1bff9b35a815bd7004eb29c3e254.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/c3483ed4f22a34bf93a7b1959ff89262e1cd1bff9b35a815bd7004eb29c3e254.jpg)

![cd0676ed82e0c4d048f619e3a15389c2e8b409aeffbcdb47fa86812d68b01d28.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/cd0676ed82e0c4d048f619e3a15389c2e8b409aeffbcdb47fa86812d68b01d28.jpg)

![cdcbfeed9d4afcdbd4f725bf463f7f668b514de2ade9699b94414cd930fe0f1d.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/cdcbfeed9d4afcdbd4f725bf463f7f668b514de2ade9699b94414cd930fe0f1d.jpg)

![f07056d60b6ad2279a5299fae66379963fc1b5bb4b3e6deb99c5ad8ae8b4f6a4.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/images/f07056d60b6ad2279a5299fae66379963fc1b5bb4b3e6deb99c5ad8ae8b4f6a4.jpg)

### Tables

![118793f01c4f0deab593cccc19b940894510cbe1168f8f07a779fe61f605f24b.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/118793f01c4f0deab593cccc19b940894510cbe1168f8f07a779fe61f605f24b.jpg)

![1a11ac17a96ddbe754f564fbcba1d5b69f250ccae77341a59c8c04f7cdfce349.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/1a11ac17a96ddbe754f564fbcba1d5b69f250ccae77341a59c8c04f7cdfce349.jpg)

![22ed7429cea93c52283e85f7fbd55ff9126e398d86f00cd07b06c5bad4881aa0.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/22ed7429cea93c52283e85f7fbd55ff9126e398d86f00cd07b06c5bad4881aa0.jpg)

![44f8b9e5621464e947f3bef676993e2fe578bf388582322af9f631bad448fa1f.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/44f8b9e5621464e947f3bef676993e2fe578bf388582322af9f631bad448fa1f.jpg)

![6e0dd5430ab8945c7c5beee1e732e42cc3075e19ce3ba60bce444c4ef50fa203.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/6e0dd5430ab8945c7c5beee1e732e42cc3075e19ce3ba60bce444c4ef50fa203.jpg)

![770d150c04b8029e01234358f456629576746ad6974f5272c3d1e2b4f038f058.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/770d150c04b8029e01234358f456629576746ad6974f5272c3d1e2b4f038f058.jpg)

![7e9270b9ecdbd43888933565e516a32636ce3883bab72df6d987ae9d62db4253.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/7e9270b9ecdbd43888933565e516a32636ce3883bab72df6d987ae9d62db4253.jpg)

![83a8cd2ac2e5fafb5de484a7c9e773a1e8aeba8bc666389be0911b6a19702875.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/83a8cd2ac2e5fafb5de484a7c9e773a1e8aeba8bc666389be0911b6a19702875.jpg)

![8695cbff611fcca2abb3cd28a35d19945c3068a3cde5095fd6991ef276af517d.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/8695cbff611fcca2abb3cd28a35d19945c3068a3cde5095fd6991ef276af517d.jpg)

![c158cb8902ac572efcb691a4ecc976222852f8a032c5a215cea6f347ff9c120f.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/c158cb8902ac572efcb691a4ecc976222852f8a032c5a215cea6f347ff9c120f.jpg)

![cb06de23e221eb70e9a80058fdcf90a2dfe537c9ea7a0271a1e60aa1fa212729.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/cb06de23e221eb70e9a80058fdcf90a2dfe537c9ea7a0271a1e60aa1fa212729.jpg)

![d0fc687996decb1090d3f7bd6e4a2562307107c7e4b8e8019ba186a63a39ca08.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/d0fc687996decb1090d3f7bd6e4a2562307107c7e4b8e8019ba186a63a39ca08.jpg)

![de56ca506fc43ac5d16af81c73b9222b99d9fe32aebd4542f8a21fc48c9f1939.jpg](../icml_results/6037_LUVE%20_%20Latent-Cascaded%20Ultra-High-Resolution%20Video%20Generation%20with%20Dual%20Frequency%20Experts/tables/de56ca506fc43ac5d16af81c73b9222b99d9fe32aebd4542f8a21fc48c9f1939.jpg)

<a id="paper-5989"></a>

## LitReview Arena: Evaluating Literature Review Agents with Battle-style Peer Review Platform


### Images

![5939e17b0f8da02f9e2e3087b07ff09171278ea814982aae7ca3a0968a77f4cc.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/images/5939e17b0f8da02f9e2e3087b07ff09171278ea814982aae7ca3a0968a77f4cc.jpg)

![e1f753b82a5ce86692762f3277190939e20319253b931e7eccd17d9273ec7f91.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/images/e1f753b82a5ce86692762f3277190939e20319253b931e7eccd17d9273ec7f91.jpg)

### Tables

![0bac1d24c6319214e6fa363b5839df0f217007178181502630f104786c2a15c6.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/0bac1d24c6319214e6fa363b5839df0f217007178181502630f104786c2a15c6.jpg)

![3c8f13e87399ba4a6fc908e7e5b5d77a16d28d9fc3c35067babc6b671472693d.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/3c8f13e87399ba4a6fc908e7e5b5d77a16d28d9fc3c35067babc6b671472693d.jpg)

![49a35cdfdba5687c6f224a727419957dd89558b42f8bbe049d4a47310bf31f15.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/49a35cdfdba5687c6f224a727419957dd89558b42f8bbe049d4a47310bf31f15.jpg)

![b98fe30fb469c82a3ecbcf9dc46a8324db533b686f1f66e83688c7f6d1062ae9.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/b98fe30fb469c82a3ecbcf9dc46a8324db533b686f1f66e83688c7f6d1062ae9.jpg)

![cca9bff82396fcbd25ebeb5cb643f8fa9cc89621a91e435b907099981fdc0b8f.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/cca9bff82396fcbd25ebeb5cb643f8fa9cc89621a91e435b907099981fdc0b8f.jpg)

![e87f8bf3d045f0c07adf215938963fdf76bb3ba855761fa1b2d28b93d24883ef.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/e87f8bf3d045f0c07adf215938963fdf76bb3ba855761fa1b2d28b93d24883ef.jpg)

![f293002002b20b4eef8e4dd2cec1d3f9167f9862db4efe65318722254b11a65d.jpg](../icml_results/6038_LitReview%20Arena_%20Evaluating%20Literature%20Review%20Agents%20with%20Battle-style%20Peer%20Review%20Platform/tables/f293002002b20b4eef8e4dd2cec1d3f9167f9862db4efe65318722254b11a65d.jpg)

<a id="paper-5990"></a>

## WinDeskGround: A Benchmark for Robust GUI Grounding in Complex Multi-Window Desktop Environments


### Images

![0c1eef4da5d4a3a1b1bc091dd313d7a0be88ac1d7297eaf854475fcce25df851.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/0c1eef4da5d4a3a1b1bc091dd313d7a0be88ac1d7297eaf854475fcce25df851.jpg)

![143a105d636dfad20bf2d227aeb2368dfcb46fcc4e9fa20738efec6da28cbc86.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/143a105d636dfad20bf2d227aeb2368dfcb46fcc4e9fa20738efec6da28cbc86.jpg)

![31a2f968786c226676fbe9ed315e0db55a6406e70e9fbbe70abb71969631f53d.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/31a2f968786c226676fbe9ed315e0db55a6406e70e9fbbe70abb71969631f53d.jpg)

![60809437f7558069051a0219851b7e5f25394f28efa3f5df40985e734bb85670.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/60809437f7558069051a0219851b7e5f25394f28efa3f5df40985e734bb85670.jpg)

![db951ce69d21168d2bdc84411194e16715fde1c7a5ff769f75bf09d6eeaf05db.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/db951ce69d21168d2bdc84411194e16715fde1c7a5ff769f75bf09d6eeaf05db.jpg)

![f74608133bec8b5c0671c0d2fd9317452e772c9a44c6cf6e27d0787c01e54cc8.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/images/f74608133bec8b5c0671c0d2fd9317452e772c9a44c6cf6e27d0787c01e54cc8.jpg)

### Tables

![44e6c86e31101af98eea854dbe37a668cb91ff0234a519daf6e978ab6458d757.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/tables/44e6c86e31101af98eea854dbe37a668cb91ff0234a519daf6e978ab6458d757.jpg)

![82b4f5fdaff6d6725a072ef58f53facb2c7fe4969a61c6a9d0a42f15b938f79c.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/tables/82b4f5fdaff6d6725a072ef58f53facb2c7fe4969a61c6a9d0a42f15b938f79c.jpg)

![9f688017be61276c0ecb183aea889ae832dbfbdea557c9aab4a91947627cb919.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/tables/9f688017be61276c0ecb183aea889ae832dbfbdea557c9aab4a91947627cb919.jpg)

![d9171f67baaf4062997da88f8aeed92afd1215e0d848baa04798824502763f03.jpg](../icml_results/6039_WinDeskGround_%20A%20Benchmark%20for%20Robust%20GUI%20Grounding%20in%20Complex%20Multi-Window%20Desktop%20Environments/tables/d9171f67baaf4062997da88f8aeed92afd1215e0d848baa04798824502763f03.jpg)

<a id="paper-5991"></a>

## When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs


### Images

![11dbe375366b5bdc9e5fae7d3281f184ab55addd01bfcef97c2c4fb25ae84631.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/11dbe375366b5bdc9e5fae7d3281f184ab55addd01bfcef97c2c4fb25ae84631.jpg)

![12436b30c723c79e3e2255b3fb8959e5fded8fd0b6650caa1b02a4b5786ee39d.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/12436b30c723c79e3e2255b3fb8959e5fded8fd0b6650caa1b02a4b5786ee39d.jpg)

![142c8473e99b0c6483d40e1004e5cee7ad31f9461214fd7590f720741c067bd5.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/142c8473e99b0c6483d40e1004e5cee7ad31f9461214fd7590f720741c067bd5.jpg)

![18401a1e30e36882d07c1d0b497642c54565363ffb86faccaab0c1c576e7f86c.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/18401a1e30e36882d07c1d0b497642c54565363ffb86faccaab0c1c576e7f86c.jpg)

![1b8bfb55ddccd4d58572d166d881d14eefecb3738e1ceb5b85933a9a6f06df38.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/1b8bfb55ddccd4d58572d166d881d14eefecb3738e1ceb5b85933a9a6f06df38.jpg)

![445f20b629061876a0de7bb80d77c23bd582418845f911840ca7f504f234090b.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/445f20b629061876a0de7bb80d77c23bd582418845f911840ca7f504f234090b.jpg)

![4b0a48129be8d18164ca86ffa78c28f3e5fc0ee902755eba54aeecaf391eb5da.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/4b0a48129be8d18164ca86ffa78c28f3e5fc0ee902755eba54aeecaf391eb5da.jpg)

![509335603222ca624eac1648b0423ab6b6a6d1e50fa6f067f5c2de266f81d04f.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/509335603222ca624eac1648b0423ab6b6a6d1e50fa6f067f5c2de266f81d04f.jpg)

![888a3c0f8f1ce2b179f4e007b25d1f3863316f66a288966b8f9ed095e9ec71f3.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/888a3c0f8f1ce2b179f4e007b25d1f3863316f66a288966b8f9ed095e9ec71f3.jpg)

![9c2d5c7d61fbf5e23638541fd7d5fe2be4fa85b783b961c528a85ddf10b7fa36.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/9c2d5c7d61fbf5e23638541fd7d5fe2be4fa85b783b961c528a85ddf10b7fa36.jpg)

![a462093a8d5db9bac65203cc121e8d87070a53edfb8265598b9465824c681a87.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/a462093a8d5db9bac65203cc121e8d87070a53edfb8265598b9465824c681a87.jpg)

![b227b0aefe44922bdd214078f606510e96a1fca199a1c1fe28c241d2cbc55a52.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/b227b0aefe44922bdd214078f606510e96a1fca199a1c1fe28c241d2cbc55a52.jpg)

![b62eb0e61c959edf040706eb5bdcfb4969f89d4011c1b284424b52eba79db350.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/b62eb0e61c959edf040706eb5bdcfb4969f89d4011c1b284424b52eba79db350.jpg)

![b7f1e4d91c102ec431f87ab88f173ae24792a33192ef0770cc3f446f0496213d.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/b7f1e4d91c102ec431f87ab88f173ae24792a33192ef0770cc3f446f0496213d.jpg)

![ca3282af8fa818f99c7aa14ae51fa53beafc1f209b2fe7c06eec493364b1c58f.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/ca3282af8fa818f99c7aa14ae51fa53beafc1f209b2fe7c06eec493364b1c58f.jpg)

![d4bb5f08b27dc5b3e2b81ab227f9beea5fb9a8a1c7b6a3cb79d562646ada724e.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/d4bb5f08b27dc5b3e2b81ab227f9beea5fb9a8a1c7b6a3cb79d562646ada724e.jpg)

![f8eabd08479dd96aabe9e0e635f0b26481f853065c8d889cc33bbf1c6af9635b.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/f8eabd08479dd96aabe9e0e635f0b26481f853065c8d889cc33bbf1c6af9635b.jpg)

![f92da31ab9e321998fea522240f3d30528bdbd9883d1caa124fdf9040187d24d.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/images/f92da31ab9e321998fea522240f3d30528bdbd9883d1caa124fdf9040187d24d.jpg)

### Tables

![191501cced7b8a738f3640845a2fb55a7ec9e4dd01ead2931036ae55c276d2af.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/191501cced7b8a738f3640845a2fb55a7ec9e4dd01ead2931036ae55c276d2af.jpg)

![612db5099c4e7d5ff22e1bc415ce4ac9c8b58cd17f1777c3708bcc4f8089d7cb.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/612db5099c4e7d5ff22e1bc415ce4ac9c8b58cd17f1777c3708bcc4f8089d7cb.jpg)

![86c31c30e7053ef45d26da39751eb89b8b4e0d809b0f891c17a91231b403e2f0.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/86c31c30e7053ef45d26da39751eb89b8b4e0d809b0f891c17a91231b403e2f0.jpg)

![8d8a0579b9735fa44ee230e588fad0908f4030e1041a37a99ac6bb43e50a0a9f.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/8d8a0579b9735fa44ee230e588fad0908f4030e1041a37a99ac6bb43e50a0a9f.jpg)

![a40e7a322b0f24770a3a46d29d0e228b10d6296a71e55bd01bcdb87a5e905d9c.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/a40e7a322b0f24770a3a46d29d0e228b10d6296a71e55bd01bcdb87a5e905d9c.jpg)

![a9ba4dd9e2e06001b911e332cee01e6707e9ccddf496090360e88b26b4676d42.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/a9ba4dd9e2e06001b911e332cee01e6707e9ccddf496090360e88b26b4676d42.jpg)

![cc2ab67a519f5e9ee48b4bbe38da5028afba5fe853217b3637a249b04603e3b1.jpg](../icml_results/6040_When%20RAG%20Hurts_%20Diagnosing%20and%20Mitigating%20Attention%20Distraction%20in%20Retrieval-Augmented%20LVLMs/tables/cc2ab67a519f5e9ee48b4bbe38da5028afba5fe853217b3637a249b04603e3b1.jpg)

<a id="paper-5992"></a>

## Syntax vs. Semantics: How Transformers Learn Deep Dependencies


### Images

![0e6447ece974cddbc860f80094babc60cf650339295f20d1790584dd3ee2f42b.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/0e6447ece974cddbc860f80094babc60cf650339295f20d1790584dd3ee2f42b.jpg)

![4cd08086ac7f381a792ab31d7faa9de78797330abdba6b49714226ecead61776.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/4cd08086ac7f381a792ab31d7faa9de78797330abdba6b49714226ecead61776.jpg)

![5b642ac1fae2425375ff68ffbf4554f6403ce28f677104109fc0210cd9b89524.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/5b642ac1fae2425375ff68ffbf4554f6403ce28f677104109fc0210cd9b89524.jpg)

![5d269f37ac4a395975074f188c97e01f55660593535cebc7bf6996a252997d8b.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/5d269f37ac4a395975074f188c97e01f55660593535cebc7bf6996a252997d8b.jpg)

![8cdcccc5970e5d0317935e331c3a0a6696f66f41882466d75618419a0bbbead6.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/8cdcccc5970e5d0317935e331c3a0a6696f66f41882466d75618419a0bbbead6.jpg)

![9aeda9021be35d73cfa5a03df6ed3f6349b5c6d269bdb4544e6554bb6198c842.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/9aeda9021be35d73cfa5a03df6ed3f6349b5c6d269bdb4544e6554bb6198c842.jpg)

![9cab8199b1f0cc8283b4b499093d188a6ce31f0b21a66056a117628cc61519ed.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/9cab8199b1f0cc8283b4b499093d188a6ce31f0b21a66056a117628cc61519ed.jpg)

![9dced903afb3ee99360b3128a3b822114132154eae69b718bd99a9aab561fb6d.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/9dced903afb3ee99360b3128a3b822114132154eae69b718bd99a9aab561fb6d.jpg)

![a9244c5c159d7506d8d92cb011f391c5c744a54883a32746fbf2e8d3cddd4e1f.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/a9244c5c159d7506d8d92cb011f391c5c744a54883a32746fbf2e8d3cddd4e1f.jpg)

![e38de4febeb4915d29a6ec259e0a8b493237b8e4697cdbeb9e5eedf44acf1c4e.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/e38de4febeb4915d29a6ec259e0a8b493237b8e4697cdbeb9e5eedf44acf1c4e.jpg)

![e87474b2f7076451859c88ca60cfd1323983ae14b91df169e73c47aafa70d125.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/e87474b2f7076451859c88ca60cfd1323983ae14b91df169e73c47aafa70d125.jpg)

![e890f98400c59a967ca59d2048c6e275ab82aaf99dfcc3854d3382f5c3d5d5c8.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/e890f98400c59a967ca59d2048c6e275ab82aaf99dfcc3854d3382f5c3d5d5c8.jpg)

![ee67566420b697d48c188a4b37b5872b4be6d14e78de731fdaa5234f34ec902a.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/ee67566420b697d48c188a4b37b5872b4be6d14e78de731fdaa5234f34ec902a.jpg)

![fba686ff9ae893163c736ae25a96b156cb69c44aa12937edddea7f9427dbb13c.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/images/fba686ff9ae893163c736ae25a96b156cb69c44aa12937edddea7f9427dbb13c.jpg)

### Tables

![1383035f01a34c8f6280fa237869948d3c2e94934be31af100c01df06c87a1d4.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/1383035f01a34c8f6280fa237869948d3c2e94934be31af100c01df06c87a1d4.jpg)

![52458fc500cb70407e3e740c14a7d0f539d3e6da060851e28dd48aea30a04de2.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/52458fc500cb70407e3e740c14a7d0f539d3e6da060851e28dd48aea30a04de2.jpg)

![59fa2c59d7ce6d78f478b938a7dd305a20baf12d77edc71004be04a44e7b774e.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/59fa2c59d7ce6d78f478b938a7dd305a20baf12d77edc71004be04a44e7b774e.jpg)

![7022ac0325c69d1aa3faddd34ef816f8c1cf95b7da7a224be62a8c35be27163c.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/7022ac0325c69d1aa3faddd34ef816f8c1cf95b7da7a224be62a8c35be27163c.jpg)

![7ae37bab49067481866c204d2654f49a76fb62d7b3ed2f61d22a16c5e4b94b25.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/7ae37bab49067481866c204d2654f49a76fb62d7b3ed2f61d22a16c5e4b94b25.jpg)

![822317cec99f7008d07d28c776b1bed45e8ed28024fdaa65716f4fdf35936187.jpg](../icml_results/6041_Syntax%20vs.%20Semantics_%20How%20Transformers%20Learn%20Deep%20Dependencies/tables/822317cec99f7008d07d28c776b1bed45e8ed28024fdaa65716f4fdf35936187.jpg)

<a id="paper-5993"></a>

## Nonparametric Data Attribution for Diffusion Models


### Images

![146ec09a5b51593308793ef8fa0d9bba795309aaddb01fccd0e97ec85d9a4119.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/146ec09a5b51593308793ef8fa0d9bba795309aaddb01fccd0e97ec85d9a4119.jpg)

![2ddfdc8cf77f56c70be1689a70c189fd26ab9b6f4a74b463ca8eef509795a154.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/2ddfdc8cf77f56c70be1689a70c189fd26ab9b6f4a74b463ca8eef509795a154.jpg)

![4354c2bde6bb6ad7a89b4788051daa2c8a8828617f5b09ad86f19fcb53eaf53b.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/4354c2bde6bb6ad7a89b4788051daa2c8a8828617f5b09ad86f19fcb53eaf53b.jpg)

![49078a6fbbbeb4d1b15b48e6b6bb83c273e06fa904b8d52a24da2af69286fdbb.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/49078a6fbbbeb4d1b15b48e6b6bb83c273e06fa904b8d52a24da2af69286fdbb.jpg)

![4b2d374d50c1d0e3729f61d4d87588ca182fbd731a37b3094cd3e1d28e707a9a.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/4b2d374d50c1d0e3729f61d4d87588ca182fbd731a37b3094cd3e1d28e707a9a.jpg)

![58f752c1ce4f0c31c9f6e5308bc50f1aa18c86243ca6c56d0b7631de1b598b82.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/58f752c1ce4f0c31c9f6e5308bc50f1aa18c86243ca6c56d0b7631de1b598b82.jpg)

![67b186324d43fc690cc9cffd72f1eea201ff274a9fba511c77f814fe46baf93b.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/67b186324d43fc690cc9cffd72f1eea201ff274a9fba511c77f814fe46baf93b.jpg)

![7d031165dfd838ab27192615a09ac5b30cf101bed0d8c3d0c7e527801151997b.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/7d031165dfd838ab27192615a09ac5b30cf101bed0d8c3d0c7e527801151997b.jpg)

![9815dca128d3fe7ed2af02f5397150741683312016d20f5fb79fe2bfff106a4d.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/9815dca128d3fe7ed2af02f5397150741683312016d20f5fb79fe2bfff106a4d.jpg)

![9af85659b1ab81b5f8e9ee8d49b3deccb2e790a7178aa0bcac86350e6b7f2ee9.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/9af85659b1ab81b5f8e9ee8d49b3deccb2e790a7178aa0bcac86350e6b7f2ee9.jpg)

![a86f98a440cd71896c47619370dbb24192a06ae6d37d10b854e003e368afe6dc.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/a86f98a440cd71896c47619370dbb24192a06ae6d37d10b854e003e368afe6dc.jpg)

![bd53e8039800be311349ecab18c9446324d7a31556107cb7a1bebb60611e467d.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/bd53e8039800be311349ecab18c9446324d7a31556107cb7a1bebb60611e467d.jpg)

![e370412f7e0bd150f3ed532ced5f84ddfb7d5118339e87454ef6b0baed1727b3.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/e370412f7e0bd150f3ed532ced5f84ddfb7d5118339e87454ef6b0baed1727b3.jpg)

![e7460d937fb111bcb47e0eb6aab38f28fe62e195ebfff1ac8edd4f0c678d0cff.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/e7460d937fb111bcb47e0eb6aab38f28fe62e195ebfff1ac8edd4f0c678d0cff.jpg)

![e95369a5fc533b638ce45e2225f601aedcc9776c43e81eaa016ba84b8dce4f8d.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/e95369a5fc533b638ce45e2225f601aedcc9776c43e81eaa016ba84b8dce4f8d.jpg)

![ef6be9e9f637ae7541890d8cb0f70abbbec876933a5cd1a35710f33a3fe0baf1.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/images/ef6be9e9f637ae7541890d8cb0f70abbbec876933a5cd1a35710f33a3fe0baf1.jpg)

### Tables

![52b169fc9a11a9a4481d3a0a022bdec2788be17321409246417986737a1a5798.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/52b169fc9a11a9a4481d3a0a022bdec2788be17321409246417986737a1a5798.jpg)

![56e25025811c442492fc423266dbb7236599e9e231dfdad85dbad393c7742e6a.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/56e25025811c442492fc423266dbb7236599e9e231dfdad85dbad393c7742e6a.jpg)

![683dd59d107d3d354139b6d19e07f2c24dbb8609941a1ca6a316bdc5b92c85af.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/683dd59d107d3d354139b6d19e07f2c24dbb8609941a1ca6a316bdc5b92c85af.jpg)

![6aedfd9a39d73f4f71b40981d1851da89faee5945aa8f5ea1c9ed05a2af846df.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/6aedfd9a39d73f4f71b40981d1851da89faee5945aa8f5ea1c9ed05a2af846df.jpg)

![9ee9a50235bb670c599f8c7032399fa99bca51948af999e9486f3e432af68a1a.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/9ee9a50235bb670c599f8c7032399fa99bca51948af999e9486f3e432af68a1a.jpg)

![f22978301df7ce78a5f4abd1adfadc0f9ad7fdc8dfee68992365a6160a7fc05a.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/f22978301df7ce78a5f4abd1adfadc0f9ad7fdc8dfee68992365a6160a7fc05a.jpg)

![f82b0254341ea7b1f3c162e691752d0bb9850289d2e3791c6133102f1d5a38fe.jpg](../icml_results/6042_Nonparametric%20Data%20Attribution%20for%20Diffusion%20Models/tables/f82b0254341ea7b1f3c162e691752d0bb9850289d2e3791c6133102f1d5a38fe.jpg)

<a id="paper-5994"></a>

## What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code


### Images

![32ff60ffeac5ae0a3441e7c92494f577e44cb15da98a51f766efcbf13e63ff3f.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/32ff60ffeac5ae0a3441e7c92494f577e44cb15da98a51f766efcbf13e63ff3f.jpg)

![3a9a810027374e24789885ed173edaee4f192a4a7521215b3a261fe96fe5ba44.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/3a9a810027374e24789885ed173edaee4f192a4a7521215b3a261fe96fe5ba44.jpg)

![44f08da08e9a97d3600f0d0b594b02c879cbb9f289f14e6d4e31ea52e16ad295.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/44f08da08e9a97d3600f0d0b594b02c879cbb9f289f14e6d4e31ea52e16ad295.jpg)

![5cb73f4c02561f1927f63d9623daa0cc4256bcce21d64c6823607ab0c6c738b7.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/5cb73f4c02561f1927f63d9623daa0cc4256bcce21d64c6823607ab0c6c738b7.jpg)

![64e61af02244bfe89625a1949dbb553df388da81ee0ba753207f952ef59a4cfb.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/64e61af02244bfe89625a1949dbb553df388da81ee0ba753207f952ef59a4cfb.jpg)

![7b84c7d203197e85b873ca576e5bf54320bb0fcdc5afe09d51bbcee6373e632f.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/7b84c7d203197e85b873ca576e5bf54320bb0fcdc5afe09d51bbcee6373e632f.jpg)

![b4317b7ce5008f4600d33ed1e648c35f287e71429eee9628b8c948fa5e95b011.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/b4317b7ce5008f4600d33ed1e648c35f287e71429eee9628b8c948fa5e95b011.jpg)

![d349e15dcc0f9c58831d0997cecd4687ed764b5fbf8c82dea84177ce82bafd3c.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/d349e15dcc0f9c58831d0997cecd4687ed764b5fbf8c82dea84177ce82bafd3c.jpg)

![d7c4b61ce2cced6d3030c049297a18d1fd022c32aa3754b5b17397791efde24f.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/d7c4b61ce2cced6d3030c049297a18d1fd022c32aa3754b5b17397791efde24f.jpg)

![ed3276c162df70e56c20941a9cde4eb099045db2f1b7e4d03d4ba51b8c466521.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/images/ed3276c162df70e56c20941a9cde4eb099045db2f1b7e4d03d4ba51b8c466521.jpg)

### Tables

![2679ef849809c6fdbf550300cda149148facbe41892a8af13e75e232f07efb70.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/2679ef849809c6fdbf550300cda149148facbe41892a8af13e75e232f07efb70.jpg)

![58153a655afbfc5d04c76dff9dc32b1359de0194484264544ce1d5fe99aa46d2.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/58153a655afbfc5d04c76dff9dc32b1359de0194484264544ce1d5fe99aa46d2.jpg)

![904d3ddd24d2ad8e97d5accf251b764fe4fc5d3a4a72e4ab4bfb8c5bf8bc6ff0.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/904d3ddd24d2ad8e97d5accf251b764fe4fc5d3a4a72e4ab4bfb8c5bf8bc6ff0.jpg)

![99718edb000e8cf3e3a4bb4ef5218546d57e8ea6818bb0fe6438c497c700a9bf.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/99718edb000e8cf3e3a4bb4ef5218546d57e8ea6818bb0fe6438c497c700a9bf.jpg)

![aa960006e8870e9c7ed58ea869fa3dcab9d1b212533f5ad5148e8468ec9f04f5.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/aa960006e8870e9c7ed58ea869fa3dcab9d1b212533f5ad5148e8468ec9f04f5.jpg)

![b54c8f0ffcce7dd2c08204e20b129a853a22fc3daae74755083463ae1e02fbf6.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/b54c8f0ffcce7dd2c08204e20b129a853a22fc3daae74755083463ae1e02fbf6.jpg)

![de83f238e49ce7a5aa9ce579f3d44495a5ae4c14ffab3a650eaecd5ab0eff413.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/de83f238e49ce7a5aa9ce579f3d44495a5ae4c14ffab3a650eaecd5ab0eff413.jpg)

![e3e5bf567a907a9e702c4f0effe29528c43211413fd7cb07ee2685051bfba906.jpg](../icml_results/6043_What%20Really%20Improves%20Mathematical%20Reasoning_%20Structured%20Reasoning%20Signals%20Beyond%20Pure%20Code/tables/e3e5bf567a907a9e702c4f0effe29528c43211413fd7cb07ee2685051bfba906.jpg)

<a id="paper-5995"></a>

## CameraNoise: Enabling Faithful Camera Control in Video Diffusion through Geometry-Flow-Guided Noise Warping


### Images

![026a2d6a47c480ac707874ba8eabf0443779d6c551f3ba9b7e4960ed79de0c96.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/026a2d6a47c480ac707874ba8eabf0443779d6c551f3ba9b7e4960ed79de0c96.jpg)

![0911f67de8c402169a1e61ec711d6f9c8243aae6527e2fcda1ecb6b3258171a3.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/0911f67de8c402169a1e61ec711d6f9c8243aae6527e2fcda1ecb6b3258171a3.jpg)

![23ef59ac1a4659a2e4f785dec462a41416bdde27eabba1a6c960f5704e53d976.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/23ef59ac1a4659a2e4f785dec462a41416bdde27eabba1a6c960f5704e53d976.jpg)

![2a26b305240b1e19de4b259fe32600b177c24c9ccb54dbc46a296f60bb6bf5f9.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/2a26b305240b1e19de4b259fe32600b177c24c9ccb54dbc46a296f60bb6bf5f9.jpg)

![2f483017ec83297ddfabe15836ca09d57184f79d95b26fe6ad9389a8e6071171.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/2f483017ec83297ddfabe15836ca09d57184f79d95b26fe6ad9389a8e6071171.jpg)

![33978012fd7944addd60daa0f913c8c5829cd4430cf2be23db2cb85729db7aa4.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/33978012fd7944addd60daa0f913c8c5829cd4430cf2be23db2cb85729db7aa4.jpg)

![3a92650f19da4d72d276fd0ae602c9c99f2002c50e3e49e396c7fcb24579f08b.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/3a92650f19da4d72d276fd0ae602c9c99f2002c50e3e49e396c7fcb24579f08b.jpg)

![42dee23b56f4439f908dbaa60e2de97f0e845c09109b5f23c3d4af3a78e44917.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/42dee23b56f4439f908dbaa60e2de97f0e845c09109b5f23c3d4af3a78e44917.jpg)

![460347cbab9d710a8f1f92bd8af57fe7ad8f75d4f3925ada3450121e745a6599.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/460347cbab9d710a8f1f92bd8af57fe7ad8f75d4f3925ada3450121e745a6599.jpg)

![4a82e0d9fcea1331f3f55d73c5917a0a04cde4714558a67a0e41a92bc1f4aff9.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/4a82e0d9fcea1331f3f55d73c5917a0a04cde4714558a67a0e41a92bc1f4aff9.jpg)

![5776f677f8721fec718dfb0da7daa81240c3d28d755b96b6160d6590acfd9ef0.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/5776f677f8721fec718dfb0da7daa81240c3d28d755b96b6160d6590acfd9ef0.jpg)

![6ab02eb9d1db881e0fff19803b3b1ea294a82c120e3c1b4e720595aa4bf43fc5.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/6ab02eb9d1db881e0fff19803b3b1ea294a82c120e3c1b4e720595aa4bf43fc5.jpg)

![6ada8747b8cf2c285ef9ffb4c05111863585c4920a36122a4ee8a008511bf452.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/6ada8747b8cf2c285ef9ffb4c05111863585c4920a36122a4ee8a008511bf452.jpg)

![6e6a296a00fd00a9d780489d1b691915b31e20c4c406a99aa997cedd6396d319.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/6e6a296a00fd00a9d780489d1b691915b31e20c4c406a99aa997cedd6396d319.jpg)

![71ce601f8132c23cd20415e60a49817d0e4b1059dd595af4604391743bcda1d6.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/71ce601f8132c23cd20415e60a49817d0e4b1059dd595af4604391743bcda1d6.jpg)

![83ecd41e2ae2fd872c74444ea022b522eb18610adcc97de1b40a4c7b959e2c5f.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/83ecd41e2ae2fd872c74444ea022b522eb18610adcc97de1b40a4c7b959e2c5f.jpg)

![a692884d436c0b0d840cfa5e2f8166bd29d67f61173b70b733121df235afe598.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/a692884d436c0b0d840cfa5e2f8166bd29d67f61173b70b733121df235afe598.jpg)

![a71c0ac0e103c5e3c22cfb5991441cf535cd343989a1e8dd297bd4785dfd9d42.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/a71c0ac0e103c5e3c22cfb5991441cf535cd343989a1e8dd297bd4785dfd9d42.jpg)

![a74718359d1bd8059de3f226e024d28f48e6128f1f5a983eb63852629f58386f.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/a74718359d1bd8059de3f226e024d28f48e6128f1f5a983eb63852629f58386f.jpg)

![aeac406e0f97b519f11ed44cca01291f6a9fe52b48b6189fd629087d24426876.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/aeac406e0f97b519f11ed44cca01291f6a9fe52b48b6189fd629087d24426876.jpg)

![b499fab09d1cf5fbb29a48fd94776bf342b8f2e650cdf42474af1a121fc97907.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/b499fab09d1cf5fbb29a48fd94776bf342b8f2e650cdf42474af1a121fc97907.jpg)

![e508c3fcc9460f633b1e0218f1be418336fd7bbf0b4d87481d75dfb59e1f391d.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/e508c3fcc9460f633b1e0218f1be418336fd7bbf0b4d87481d75dfb59e1f391d.jpg)

![e59d5a8a407476c769f7c080515a5f0ab7e59abdb6df90675c5f1f9fdcf70fd7.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/e59d5a8a407476c769f7c080515a5f0ab7e59abdb6df90675c5f1f9fdcf70fd7.jpg)

![ecc534030319cb48f31a9b643da982a40727979d547c88789bab0ee372edff63.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/ecc534030319cb48f31a9b643da982a40727979d547c88789bab0ee372edff63.jpg)

![ef8e626ae8c0481f7862feb6678acd1c18bf54353713e7eebd66dcedf2ae071e.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/ef8e626ae8c0481f7862feb6678acd1c18bf54353713e7eebd66dcedf2ae071e.jpg)

![f13bebf16eafb448616efe147d4797a31253d3c2d320cbf17981fbf2dedcb330.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/f13bebf16eafb448616efe147d4797a31253d3c2d320cbf17981fbf2dedcb330.jpg)

![fb6a2cbee72840ae8e92b42f0f8c4dd8a4d7187e3f4b947124fd57c7a253e1b7.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/images/fb6a2cbee72840ae8e92b42f0f8c4dd8a4d7187e3f4b947124fd57c7a253e1b7.jpg)

### Tables

![222a3261d78c62e30306bc6ea6eab3056b4fef6c57964e1718c951bf5c7ae6de.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/222a3261d78c62e30306bc6ea6eab3056b4fef6c57964e1718c951bf5c7ae6de.jpg)

![2d67e646914120d57348c36a76da65ae576ac1d2ddaf8c55d56180d9a0e07af4.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/2d67e646914120d57348c36a76da65ae576ac1d2ddaf8c55d56180d9a0e07af4.jpg)

![32c27e271bc53531f691ab7d710b16cf84181d14d3f18004136035648ccdd8b7.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/32c27e271bc53531f691ab7d710b16cf84181d14d3f18004136035648ccdd8b7.jpg)

![3f80b7ad1fe3e8c4fd1f06f967eb407e13ef7763f470caf4d42c8190b97fae77.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/3f80b7ad1fe3e8c4fd1f06f967eb407e13ef7763f470caf4d42c8190b97fae77.jpg)

![64230fa6c35996a80bc6b05d6ac5623ce5cb85a7cc27f0791137ea6aa595fd26.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/64230fa6c35996a80bc6b05d6ac5623ce5cb85a7cc27f0791137ea6aa595fd26.jpg)

![67a85a9663ce70114f922886304cdf682241c41a9858fa43f9854fae8a97af3f.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/67a85a9663ce70114f922886304cdf682241c41a9858fa43f9854fae8a97af3f.jpg)

![6d467d1a81da1804a5b81198ca5f130e80a7b510ac86418d4dc98407447294e5.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/6d467d1a81da1804a5b81198ca5f130e80a7b510ac86418d4dc98407447294e5.jpg)

![a1313e3993cfeea8c28df6347e7bd0c5fa4f24fa3ca220d8c208020d561bda96.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/a1313e3993cfeea8c28df6347e7bd0c5fa4f24fa3ca220d8c208020d561bda96.jpg)

![e01b4971307f4b54e19ff58f86058667954d7a7fcbf11b50b829fab45a4792fd.jpg](../icml_results/6044_CameraNoise_%20Enabling%20Faithful%20Camera%20Control%20in%20Video%20Diffusion%20through%20Geometry-Flow-Guided%20Noise%20/tables/e01b4971307f4b54e19ff58f86058667954d7a7fcbf11b50b829fab45a4792fd.jpg)

<a id="paper-5996"></a>

## GHOST: Geometry-Guided Hallucination of Opaque Surface Textures


### Images

![1c47b91c281af5bab14cc63db16fcca2ecaa56fb302dbb2fc269dc844a701ce9.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/1c47b91c281af5bab14cc63db16fcca2ecaa56fb302dbb2fc269dc844a701ce9.jpg)

![52a714da2cf4eca32594a035895b80747b82e6463cbc90f07fa7b7a8f26ba557.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/52a714da2cf4eca32594a035895b80747b82e6463cbc90f07fa7b7a8f26ba557.jpg)

![7f891fe5cc653a4f62006d66d0656525ea959bceb1341284cf95c2812d1c8b39.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/7f891fe5cc653a4f62006d66d0656525ea959bceb1341284cf95c2812d1c8b39.jpg)

![a09a7a5c6b77eeac94f11192df12e88f77168dbf401e8ec59177e8a1c5292234.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/a09a7a5c6b77eeac94f11192df12e88f77168dbf401e8ec59177e8a1c5292234.jpg)

![ba7d42eb81db3a01fec9abc3f8ded57e046c2a2cf3cf4e87ba43d8b8ea7bb673.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/ba7d42eb81db3a01fec9abc3f8ded57e046c2a2cf3cf4e87ba43d8b8ea7bb673.jpg)

![bf0ab3095ee7679a7a129241aea15cb6d11d0286d51dddb69d08b34f254fada4.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/bf0ab3095ee7679a7a129241aea15cb6d11d0286d51dddb69d08b34f254fada4.jpg)

![e0027cbc49c0d2794510a697dc910f00d70a35b9205b16ee1bdf6b872a3e2e81.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/images/e0027cbc49c0d2794510a697dc910f00d70a35b9205b16ee1bdf6b872a3e2e81.jpg)

### Tables

![0c6431c33e571a2b302a2739d5b4db1d9fe2f7ac60a0eb51c8759fea1db11e7a.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/0c6431c33e571a2b302a2739d5b4db1d9fe2f7ac60a0eb51c8759fea1db11e7a.jpg)

![26bfd5045801e8f46ec2b13aed21f06d10c614e3787dc71e8f93cd6b8d4535e9.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/26bfd5045801e8f46ec2b13aed21f06d10c614e3787dc71e8f93cd6b8d4535e9.jpg)

![39f644543046d35a134712254b9897530777d7982b0d963aa00a58dec332040d.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/39f644543046d35a134712254b9897530777d7982b0d963aa00a58dec332040d.jpg)

![5a038b672901a9245be20fd6f9a8ba7eed0dd529640b441de8634ac66431f791.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/5a038b672901a9245be20fd6f9a8ba7eed0dd529640b441de8634ac66431f791.jpg)

![98cf14f95bd241d34b17f65f7e8750930b39c4542ce49de8fb7f65f0accd14fc.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/98cf14f95bd241d34b17f65f7e8750930b39c4542ce49de8fb7f65f0accd14fc.jpg)

![b4460ef723bab8402c335bb5f2c2c89358fc6e5bef00ade701e2bdd74a3338c6.jpg](../icml_results/6045_GHOST_%20Geometry-Guided%20Hallucination%20of%20Opaque%20Surface%20Textures/tables/b4460ef723bab8402c335bb5f2c2c89358fc6e5bef00ade701e2bdd74a3338c6.jpg)

<a id="paper-5997"></a>

## MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation


### Images

![30af52efd9572cc842d0b369e0adc11f2fb1a3d2c0e145196ca3922a07bfe85b.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/30af52efd9572cc842d0b369e0adc11f2fb1a3d2c0e145196ca3922a07bfe85b.jpg)

![3300b851a1b5999a4dbe09f37508bd051a0f077a48ca112c948e04e90be75ddf.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/3300b851a1b5999a4dbe09f37508bd051a0f077a48ca112c948e04e90be75ddf.jpg)

![39884d5a6f2dabc2490385515bc7d6fdf624247eca21546650f591eb6a570da4.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/39884d5a6f2dabc2490385515bc7d6fdf624247eca21546650f591eb6a570da4.jpg)

![af178c2a5dc75eb818ba4af57cb8f9e80c3dc462444c7da38815876db399595e.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/af178c2a5dc75eb818ba4af57cb8f9e80c3dc462444c7da38815876db399595e.jpg)

![cb5d035536ef080646e5731af0361b8bf65c873b4d26d41de10255213fe7956c.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/cb5d035536ef080646e5731af0361b8bf65c873b4d26d41de10255213fe7956c.jpg)

![f7ec154e3905d4f8232840c4d1106a18cf59d4e28c3a6a912270f838255c6542.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/images/f7ec154e3905d4f8232840c4d1106a18cf59d4e28c3a6a912270f838255c6542.jpg)

### Tables

![060feea77eb971f31b36fde79fa3e484ef9d79ee4b6460c0c946af03dac7f966.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/060feea77eb971f31b36fde79fa3e484ef9d79ee4b6460c0c946af03dac7f966.jpg)

![0e14361d21124e378f86d9bf361da1f5594617409234b20df8c643b7543139d4.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/0e14361d21124e378f86d9bf361da1f5594617409234b20df8c643b7543139d4.jpg)

![0e9769bdb8d7d2b89684bbb98e9d4f69ebee5704c29e5a46f93e35f19dfc03dd.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/0e9769bdb8d7d2b89684bbb98e9d4f69ebee5704c29e5a46f93e35f19dfc03dd.jpg)

![13e4ae361db665381098c94d94cd899672793beb10967df952f586fd93a2e74f.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/13e4ae361db665381098c94d94cd899672793beb10967df952f586fd93a2e74f.jpg)

![510f1e8ea40803247ff84211267f4294ec7fbc260cd7582e425731a02d06b569.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/510f1e8ea40803247ff84211267f4294ec7fbc260cd7582e425731a02d06b569.jpg)

![614ba8900b53d78eb870522939a24cea35c6f502882a1c7f64908905be0651f0.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/614ba8900b53d78eb870522939a24cea35c6f502882a1c7f64908905be0651f0.jpg)

![7427661178844c13e5ecfa0ad4c8b4783a07e85a4040dcb4b41daf314158a52f.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/7427661178844c13e5ecfa0ad4c8b4783a07e85a4040dcb4b41daf314158a52f.jpg)

![772869402d00e205db9656e74d34b18589aeba010ec6edfd9984c3b786a9a248.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/772869402d00e205db9656e74d34b18589aeba010ec6edfd9984c3b786a9a248.jpg)

![a6c37c5b2d67dffe1bd57d934d69d15d99b625598dededf583cfc4632f72c3aa.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/a6c37c5b2d67dffe1bd57d934d69d15d99b625598dededf583cfc4632f72c3aa.jpg)

![a6ccfd0b362bb7a5ead9ea40354ae0aa8c4be5726b53b8245cce5e91595beeed.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/a6ccfd0b362bb7a5ead9ea40354ae0aa8c4be5726b53b8245cce5e91595beeed.jpg)

![e06c0a37a67453f5dcae577377c12885ffd3b232339c894d6af4b71cb01170f4.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/e06c0a37a67453f5dcae577377c12885ffd3b232339c894d6af4b71cb01170f4.jpg)

![f4941cde537c3116bdb6d96f03bf196331e8986a2634f9e62f5992c0eb264575.jpg](../icml_results/6046_MedCoG_%20Maximizing%20LLM%20Inference%20Density%20in%20Medical%20Reasoning%20via%20Meta-Cognitive%20Regulation/tables/f4941cde537c3116bdb6d96f03bf196331e8986a2634f9e62f5992c0eb264575.jpg)

<a id="paper-5998"></a>

## When Is Rank-1 Enough? Geometry-Guided Initialization for Parameter-Efficient Fine-Tuning


### Images

![53099653173e074d0315c9ad629ba74aa04a23deba48ae51989a67b1d6a41c9a.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/images/53099653173e074d0315c9ad629ba74aa04a23deba48ae51989a67b1d6a41c9a.jpg)

![a0a9cecd2d59b388a61e0bc204111169e8e2ead851b9ab2fc60c73582589d9db.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/images/a0a9cecd2d59b388a61e0bc204111169e8e2ead851b9ab2fc60c73582589d9db.jpg)

![a7950211d3971becf686e6b2f3360c45b8cceef8ef7032c910ebb087b6209765.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/images/a7950211d3971becf686e6b2f3360c45b8cceef8ef7032c910ebb087b6209765.jpg)

![ee32333a3f16cd7367a6222367b3b28e62d92e0cce6b42587a4f294a41ceb918.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/images/ee32333a3f16cd7367a6222367b3b28e62d92e0cce6b42587a4f294a41ceb918.jpg)

![ff9f2868c121a7494d1978afbc27fff11f2a2c5ac942af220e11198d6c8df657.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/images/ff9f2868c121a7494d1978afbc27fff11f2a2c5ac942af220e11198d6c8df657.jpg)

### Tables

![0d9eb6f17732a7974ad74fd0844c601f0e21a95c9c600451b4445c683eb6fc9c.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/0d9eb6f17732a7974ad74fd0844c601f0e21a95c9c600451b4445c683eb6fc9c.jpg)

![338415a388f89952ecbea1f8b1fd0b5ac0f726f100b72085f0f162542dad9791.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/338415a388f89952ecbea1f8b1fd0b5ac0f726f100b72085f0f162542dad9791.jpg)

![3822230e50c273bdb94e042fb770ba41ade630cc5fe03d75232980b5d06df5b5.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/3822230e50c273bdb94e042fb770ba41ade630cc5fe03d75232980b5d06df5b5.jpg)

![423761deee19cc4ed9d69c2f77d0a82ec9680b0dc19a1467d20c7e0d1351757d.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/423761deee19cc4ed9d69c2f77d0a82ec9680b0dc19a1467d20c7e0d1351757d.jpg)

![4897724da10565d0d76b18186f3f09367fe8e8f73989b2ec6d0e2b78c66cf5c5.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/4897724da10565d0d76b18186f3f09367fe8e8f73989b2ec6d0e2b78c66cf5c5.jpg)

![94c1665007a04506bc37bdd74ec47101b3fc5f65051660e15d3b9a97ec137f56.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/94c1665007a04506bc37bdd74ec47101b3fc5f65051660e15d3b9a97ec137f56.jpg)

![a72ec4e7dcbdf5183c994de67a784d4d99e088efc6e9f0252df856feb4d71e33.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/a72ec4e7dcbdf5183c994de67a784d4d99e088efc6e9f0252df856feb4d71e33.jpg)

![b4fe578b1565dc350031433ab7d8e332f5dc04b129bdde4beb0a0f4af94aa755.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/b4fe578b1565dc350031433ab7d8e332f5dc04b129bdde4beb0a0f4af94aa755.jpg)

![c4b119fabfd654e38884ee48ae630a30c8c4209b96c029821d930edd32874e7f.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/c4b119fabfd654e38884ee48ae630a30c8c4209b96c029821d930edd32874e7f.jpg)

![c51c5c21b758c039a9652d73a29e03fe5a1010c73bc6b0dc9812b2571518e0dc.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/c51c5c21b758c039a9652d73a29e03fe5a1010c73bc6b0dc9812b2571518e0dc.jpg)

![c7728ee4ec259a1b2b753144bd44ec0ace12c36fa309c6d7cc4287eec922f11d.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/c7728ee4ec259a1b2b753144bd44ec0ace12c36fa309c6d7cc4287eec922f11d.jpg)

![ce6bc53decbd90b3a35e5590d9d35eceacf9d00c6833b6fb3cb0a848282d7714.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/ce6bc53decbd90b3a35e5590d9d35eceacf9d00c6833b6fb3cb0a848282d7714.jpg)

![d2ec17897d3ca2cf821870e6a607d38eac15bb96c19ed726bce7192f5b7daba0.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/d2ec17897d3ca2cf821870e6a607d38eac15bb96c19ed726bce7192f5b7daba0.jpg)

![e48272aad16812857408f0ba1763ad950ac85f950d1da376a6ecf2f4386cdf6e.jpg](../icml_results/6047_When%20Is%20Rank-1%20Enough_%20Geometry-Guided%20Initialization%20for%20Parameter-Efficient%20Fine-Tuning/tables/e48272aad16812857408f0ba1763ad950ac85f950d1da376a6ecf2f4386cdf6e.jpg)

<a id="paper-5999"></a>

## From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data


### Images

![0c1b26edef6046b6171feef4c5fcff27603be85c84980cdaf2b9b8accf4d5d25.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/0c1b26edef6046b6171feef4c5fcff27603be85c84980cdaf2b9b8accf4d5d25.jpg)

![33198a126c44635c932aaf18cee46eaac96b487cb721c2c71174b280fab2d27c.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/33198a126c44635c932aaf18cee46eaac96b487cb721c2c71174b280fab2d27c.jpg)

![39948689ede96c33ed6996690f00e160ac2c28a55011947461fec086060f0797.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/39948689ede96c33ed6996690f00e160ac2c28a55011947461fec086060f0797.jpg)

![4b561e711ba2abe8f0ca9c5de1b8d08fc4c21d952d6d552292bd8a536e62a702.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/4b561e711ba2abe8f0ca9c5de1b8d08fc4c21d952d6d552292bd8a536e62a702.jpg)

![572406b7fab8c6a8fa367df551a52dcf8ce393cb069098012d0cd26eb39037b7.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/572406b7fab8c6a8fa367df551a52dcf8ce393cb069098012d0cd26eb39037b7.jpg)

![8bf9cc23c93bee2ff1ce843dc00f5875f43fefeaa4e566ee2925ee176dbdb5a9.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/8bf9cc23c93bee2ff1ce843dc00f5875f43fefeaa4e566ee2925ee176dbdb5a9.jpg)

![949a011f2e462e0ce945b6886fbe52046ab1eb044188d9cbdf40bc39647e4259.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/949a011f2e462e0ce945b6886fbe52046ab1eb044188d9cbdf40bc39647e4259.jpg)

![bfa4177311b7539e1264b635d9c06fbf0e9fdd395fb6ca28b9e043b8acafa6d6.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/bfa4177311b7539e1264b635d9c06fbf0e9fdd395fb6ca28b9e043b8acafa6d6.jpg)

![c249cd124b2ca42fb06390f8c21052661170a3e67b6811102372a34b5e90252f.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/c249cd124b2ca42fb06390f8c21052661170a3e67b6811102372a34b5e90252f.jpg)

![d8cf5cd9cdf73aa47f1464297cd270f833d6da1deba09b56ffaf1cdfb8d1da73.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/d8cf5cd9cdf73aa47f1464297cd270f833d6da1deba09b56ffaf1cdfb8d1da73.jpg)

![d91b4a72f925b5166913d8f9271000226e204f98175e72a7b210ae2a53da33b8.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/d91b4a72f925b5166913d8f9271000226e204f98175e72a7b210ae2a53da33b8.jpg)

![e4354462d03766edd7cf1400f9bef97f2133e5f083b965c75636b011c01fd41f.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/e4354462d03766edd7cf1400f9bef97f2133e5f083b965c75636b011c01fd41f.jpg)

![eb505c35956eb58ac6708644d047a5222327bce7a1de68a018ab123c22fa7c98.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/eb505c35956eb58ac6708644d047a5222327bce7a1de68a018ab123c22fa7c98.jpg)

![ed97bb45633d5d476b9982d4ea871a5256dfc476d7dfc1b94261f572d23adc2f.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/images/ed97bb45633d5d476b9982d4ea871a5256dfc476d7dfc1b94261f572d23adc2f.jpg)

### Tables

![125fed1779918dc62d365d348e4f40cacfdb5053c4cb9cc36145f2a892c1a497.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/125fed1779918dc62d365d348e4f40cacfdb5053c4cb9cc36145f2a892c1a497.jpg)

![34e6e14dcaa4fc3837a8bd22c9f9ee13a0e46ff5e50eb1d1dec5cefa94d5d35d.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/34e6e14dcaa4fc3837a8bd22c9f9ee13a0e46ff5e50eb1d1dec5cefa94d5d35d.jpg)

![46883350482c340d583cb7565b76c30fb86c60354b9e827e878125f44e861f92.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/46883350482c340d583cb7565b76c30fb86c60354b9e827e878125f44e861f92.jpg)

![6a085923a992f84b5b869d5a98f0805ed585ab97204c21363ac8287d2fae99e4.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/6a085923a992f84b5b869d5a98f0805ed585ab97204c21363ac8287d2fae99e4.jpg)

![72692d47874f8451fb59b76e6808f36d6e34bfdb913fd02afd8e6336e6737f76.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/72692d47874f8451fb59b76e6808f36d6e34bfdb913fd02afd8e6336e6737f76.jpg)

![7a7854065db523fde29339668f7828bd4b648ccd1011ffbc0ee20766435e6c0a.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/7a7854065db523fde29339668f7828bd4b648ccd1011ffbc0ee20766435e6c0a.jpg)

![971e576e50b4e8d813ba0d7665e61c648d9b61fc4f853ad8cf26abf1a4ac38f3.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/971e576e50b4e8d813ba0d7665e61c648d9b61fc4f853ad8cf26abf1a4ac38f3.jpg)

![cae9cbefd8317433f7cebedeb58e966f10031a3d50b924079871d657c645ae6e.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/cae9cbefd8317433f7cebedeb58e966f10031a3d50b924079871d657c645ae6e.jpg)

![d18ef381ea46fd00ecefa0f1c6ef79012e6fd54bde854c58bc5188fb142b5afe.jpg](../icml_results/6048_From%20Extrinsic%20to%20Intrinsic_%20Geodesic-Guided%20Representation%20Learning%20for%203D%20Geometric%20Data/tables/d18ef381ea46fd00ecefa0f1c6ef79012e6fd54bde854c58bc5188fb142b5afe.jpg)

<a id="paper-6000"></a>

## DiP-G: Discrete Prompting for Graph Neural Networks


### Images

![1ec9d62eacff0bfcdc95a4d6fe4eaed4040300134f89b605dc543c634576efe6.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/images/1ec9d62eacff0bfcdc95a4d6fe4eaed4040300134f89b605dc543c634576efe6.jpg)

![423c8c30db213d6ecebe7d8a5fd1f8bbd2c4aa6a0f116629be0e3d0173bf0a49.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/images/423c8c30db213d6ecebe7d8a5fd1f8bbd2c4aa6a0f116629be0e3d0173bf0a49.jpg)

![4484d43c57c6cfa04faa1798164a1c128cc75ea70c4d114272452d3bcec96e4d.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/images/4484d43c57c6cfa04faa1798164a1c128cc75ea70c4d114272452d3bcec96e4d.jpg)

![ba8daa0d2a0581651bb722f0f2c44c5852fa01ff028d4feac2bf410d5c7d61a5.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/images/ba8daa0d2a0581651bb722f0f2c44c5852fa01ff028d4feac2bf410d5c7d61a5.jpg)

![e0c4e095995e95a3fae14beeac1cbc89bebd8297b54edf2c1523ddfb820de23e.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/images/e0c4e095995e95a3fae14beeac1cbc89bebd8297b54edf2c1523ddfb820de23e.jpg)

### Tables

![2db9ccf07c2289ce11e697b35783c64237fb476164bb167ab973ef6e9c397049.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/2db9ccf07c2289ce11e697b35783c64237fb476164bb167ab973ef6e9c397049.jpg)

![811531882bda7ded999e352ec4cb03a54a64c9cf073df9143403f7cd844fa3fd.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/811531882bda7ded999e352ec4cb03a54a64c9cf073df9143403f7cd844fa3fd.jpg)

![a0b0ecb6b4f5eb258554b6edf67a1ca951936db5061acf2e160022cb0e06903c.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/a0b0ecb6b4f5eb258554b6edf67a1ca951936db5061acf2e160022cb0e06903c.jpg)

![bb615846e5078524026cc67c4a2b8a0df6b189e2ee4cadc4898b84c123136907.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/bb615846e5078524026cc67c4a2b8a0df6b189e2ee4cadc4898b84c123136907.jpg)

![c10c94ae2ad2ad74910aee70dbf6882edaf87fa42e4d388420025bc2e9c8ab6f.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/c10c94ae2ad2ad74910aee70dbf6882edaf87fa42e4d388420025bc2e9c8ab6f.jpg)

![eb61853c0edf50109db266af9434aff88c629d83386b94a0a71657a4b463703b.jpg](../icml_results/6049_DiP-G_%20Discrete%20Prompting%20for%20Graph%20Neural%20Networks/tables/eb61853c0edf50109db266af9434aff88c629d83386b94a0a71657a4b463703b.jpg)

<a id="paper-6001"></a>

## Shift-Dependent Asymmetry: Orthogonal Inverse Low-Rank Adaptation for Federated Medical Segmentation


### Images

![24d0822017b29d5c1095d3c2baf28b6c37911c857e2e55322648a1dc901948ff.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/images/24d0822017b29d5c1095d3c2baf28b6c37911c857e2e55322648a1dc901948ff.jpg)

![46b83210c17ffbbeb9fca42cfdfe9dd9daa8ae2ae1fe5f57a265f2c9e2490f56.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/images/46b83210c17ffbbeb9fca42cfdfe9dd9daa8ae2ae1fe5f57a265f2c9e2490f56.jpg)

![7119e1d339953573bebdbaaee2de176c9c2d42280018f26796e40b4e9cbd0c0f.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/images/7119e1d339953573bebdbaaee2de176c9c2d42280018f26796e40b4e9cbd0c0f.jpg)

![fb3b3b6c6d8ea72b0071238209fdcd49d7790e13a5b024c6465a4b500037d644.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/images/fb3b3b6c6d8ea72b0071238209fdcd49d7790e13a5b024c6465a4b500037d644.jpg)

![ffd39dd9662831d493a029a7c3d3da453b24de493908543c1f6e852f3afbe235.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/images/ffd39dd9662831d493a029a7c3d3da453b24de493908543c1f6e852f3afbe235.jpg)

### Tables

![24934bdf39b5d35f67295342e1b6b61d429c186e59a30b0a18fbc3e5927e21ea.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/tables/24934bdf39b5d35f67295342e1b6b61d429c186e59a30b0a18fbc3e5927e21ea.jpg)

![7afb626b29e352a1db8a3a1e8ed1cfb1f42232b112f16ac6165b998209afb57a.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/tables/7afb626b29e352a1db8a3a1e8ed1cfb1f42232b112f16ac6165b998209afb57a.jpg)

![ce10475dac7c50aa31d700b77e0f53086a0e02e2ab284479f1ff086c15d2dc52.jpg](../icml_results/6050_Shift-Dependent%20Asymmetry_%20Orthogonal%20Inverse%20Low-Rank%20Adaptation%20for%20Federated%20Medical%20Segmentation/tables/ce10475dac7c50aa31d700b77e0f53086a0e02e2ab284479f1ff086c15d2dc52.jpg)

<a id="paper-6002"></a>

## Rh-3DGS: Robust Open-Vocabulary Scene Understanding via Riemannian Huber Distillation and Manifold-Aware Sampling


### Images

![27acc14369ad2028f988aa34752e64a4e4098678a2f2f530b96308b400598fc5.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/27acc14369ad2028f988aa34752e64a4e4098678a2f2f530b96308b400598fc5.jpg)

![3e066f4b750e9b7e1ad59012e2815a43e9a640948aae2c889c89f81aec43ea1d.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/3e066f4b750e9b7e1ad59012e2815a43e9a640948aae2c889c89f81aec43ea1d.jpg)

![42cf50240f0fb073445bd154935ff20d8ef6837f92a23aa1d800ff5ec03a5838.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/42cf50240f0fb073445bd154935ff20d8ef6837f92a23aa1d800ff5ec03a5838.jpg)

![531766889245767be263fba56f5f6bb56babf69957e332d51f32fd3f2f25df30.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/531766889245767be263fba56f5f6bb56babf69957e332d51f32fd3f2f25df30.jpg)

![75446ec91bae1cd6b796483c17a53aef91eeec963ed566235917f41a0af1f2f9.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/75446ec91bae1cd6b796483c17a53aef91eeec963ed566235917f41a0af1f2f9.jpg)

![95065f7ca7e8f336900e90684080b429947accab3664bbb2f90e7427aaa22021.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/95065f7ca7e8f336900e90684080b429947accab3664bbb2f90e7427aaa22021.jpg)

![a52ea581f3efd61da4267b93bd01fe2d1c0717ab0dfc1fd3b5f2290a6f2fef2e.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/a52ea581f3efd61da4267b93bd01fe2d1c0717ab0dfc1fd3b5f2290a6f2fef2e.jpg)

![c52fd67cb513cf6cf1d24ad8f29915c7d0caeab3057a2d21e16430c2e210c509.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/c52fd67cb513cf6cf1d24ad8f29915c7d0caeab3057a2d21e16430c2e210c509.jpg)

![ed7008b4de3f3ef5003aa1ddc30352e8a91fd2fc8e7708c1f5b89d1ed3fb7e2b.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/images/ed7008b4de3f3ef5003aa1ddc30352e8a91fd2fc8e7708c1f5b89d1ed3fb7e2b.jpg)

### Tables

![19a1eba5e45f6f48643beff67aeae33115312b5e0753b10302ea3972c02e0eb7.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/19a1eba5e45f6f48643beff67aeae33115312b5e0753b10302ea3972c02e0eb7.jpg)

![20bdbb811beb4f753163b37f66a8eae00c060a6013605d73e5c3f82ede11d4c4.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/20bdbb811beb4f753163b37f66a8eae00c060a6013605d73e5c3f82ede11d4c4.jpg)

![2ac4bb28d2fe53d8ac80ce92aaf609cc8a227fa1c04a830daa091b2efbebf2ee.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/2ac4bb28d2fe53d8ac80ce92aaf609cc8a227fa1c04a830daa091b2efbebf2ee.jpg)

![4ce37d22c8497ff316e6152eac760db8b2eae4938fd1828255de7e64d6892f8b.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/4ce37d22c8497ff316e6152eac760db8b2eae4938fd1828255de7e64d6892f8b.jpg)

![abaf9c898eadd511151b2f6783772f459c4749306f83eec99541d5e00e99e7e1.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/abaf9c898eadd511151b2f6783772f459c4749306f83eec99541d5e00e99e7e1.jpg)

![b176d936479021590d09dcfb16c1bfd742648a90350c50cd8ad99a240cef9a23.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/b176d936479021590d09dcfb16c1bfd742648a90350c50cd8ad99a240cef9a23.jpg)

![bbb7bde18f39ababb756c83401e34d2852d6ff1950b74f4375046d63ea8a62ba.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/bbb7bde18f39ababb756c83401e34d2852d6ff1950b74f4375046d63ea8a62ba.jpg)

![c02df299eab2d217e9ddfca3f19744a1202d9aba9cac97c968f2223365c3ec5b.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/c02df299eab2d217e9ddfca3f19744a1202d9aba9cac97c968f2223365c3ec5b.jpg)

![d048c6d2a71127daaa325c8ecdc5e07dafb2d7e2e112531a34da8f668eba2dc5.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/d048c6d2a71127daaa325c8ecdc5e07dafb2d7e2e112531a34da8f668eba2dc5.jpg)

![db719431f776562881b84c9a578ed43c5bd9cafa4c7351b97d0ff1d98c6fe63f.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/db719431f776562881b84c9a578ed43c5bd9cafa4c7351b97d0ff1d98c6fe63f.jpg)

![e6a0fdedfbabb98504182e098cba0ba5c040eccdb8a326bdc5f4cfc42ff9f8d0.jpg](../icml_results/6051_Rh-3DGS_%20Robust%20Open-Vocabulary%20Scene%20Understanding%20via%20Riemannian%20Huber%20Distillation%20and%20Manifold-A/tables/e6a0fdedfbabb98504182e098cba0ba5c040eccdb8a326bdc5f4cfc42ff9f8d0.jpg)

<a id="paper-6003"></a>

## Learning Sparse Visual Representations via Spatial-Semantic Factorization


### Images

![6eb830fee2e1e0616d2d6fc9499be5aa1f6a9580ea506ab4645f632b1c7b2f98.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/images/6eb830fee2e1e0616d2d6fc9499be5aa1f6a9580ea506ab4645f632b1c7b2f98.jpg)

![73905db50d9ea0b440b47e004cc65c220c2aaef10141bb3f6398354b27fae15a.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/images/73905db50d9ea0b440b47e004cc65c220c2aaef10141bb3f6398354b27fae15a.jpg)

![aa0b336e1cd97f1de8f5f9fe1fb27fd8e530a39fa11c5331ad93507a64cb7223.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/images/aa0b336e1cd97f1de8f5f9fe1fb27fd8e530a39fa11c5331ad93507a64cb7223.jpg)

![b26a12801a99529d7030b42a0edf0c1671b7bc6bb253b2b65c5136eec93bccea.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/images/b26a12801a99529d7030b42a0edf0c1671b7bc6bb253b2b65c5136eec93bccea.jpg)

![d51cfba43381eff4e910d5574af12f21e6d1124a2d9268fe13fe0293437d5342.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/images/d51cfba43381eff4e910d5574af12f21e6d1124a2d9268fe13fe0293437d5342.jpg)

### Tables

![0943d99f54941c9480887db95430d38c49c5653c0d888ef5e2f11b695a6d9b4f.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/0943d99f54941c9480887db95430d38c49c5653c0d888ef5e2f11b695a6d9b4f.jpg)

![22364be8c5834335c408759260551cbb4cfcdc025ff7c6f75b53ad1db479eb3c.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/22364be8c5834335c408759260551cbb4cfcdc025ff7c6f75b53ad1db479eb3c.jpg)

![368a181878a17977e44c4442e919d400683bf1413b09d82cc16656f527a91254.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/368a181878a17977e44c4442e919d400683bf1413b09d82cc16656f527a91254.jpg)

![4eaa831d04762a84108fc24421676fa45df8c6293d236b8e41ef961a93aa5a3c.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/4eaa831d04762a84108fc24421676fa45df8c6293d236b8e41ef961a93aa5a3c.jpg)

![554bcc93c05accc2aba861391fa862a4c21cf5a8aa21794e7636d086a2855299.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/554bcc93c05accc2aba861391fa862a4c21cf5a8aa21794e7636d086a2855299.jpg)

![a2bfc732f98466f3719b06bec90fdffe26098ac35c0740345c522951211938a0.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/a2bfc732f98466f3719b06bec90fdffe26098ac35c0740345c522951211938a0.jpg)

![a8ccb367d7cfdc991ccf83da4a1cbd1ff3f21820bc19eebe0cea3408c059f6eb.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/a8ccb367d7cfdc991ccf83da4a1cbd1ff3f21820bc19eebe0cea3408c059f6eb.jpg)

![a9ddc856167fd1908c8fc993853aec5e01578fb11b39e846fea0532ad842d28e.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/a9ddc856167fd1908c8fc993853aec5e01578fb11b39e846fea0532ad842d28e.jpg)

![abb55de4355d2af4df15661e26a3fe32458b5b59b48606e30f0a8365b75bc15f.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/abb55de4355d2af4df15661e26a3fe32458b5b59b48606e30f0a8365b75bc15f.jpg)

![dfea9cccbfaea46c8c149d0d99410757d0216fc87b093ca8034722ce6b6094f3.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/dfea9cccbfaea46c8c149d0d99410757d0216fc87b093ca8034722ce6b6094f3.jpg)

![e6f0b04a9aa0f820fdd8b16d8fbc9ec13a07fdec963be7ed495f23175863bb39.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/e6f0b04a9aa0f820fdd8b16d8fbc9ec13a07fdec963be7ed495f23175863bb39.jpg)

![eca58d6a3d5853dc41f11c22e21bec5e3a9ad4205795f98dfe9461e1141b2e37.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/eca58d6a3d5853dc41f11c22e21bec5e3a9ad4205795f98dfe9461e1141b2e37.jpg)

![ef6ac2ae56d616b12e1b83dfeb7bb64152fd590a734587a0ce169f46b4c82be2.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/ef6ac2ae56d616b12e1b83dfeb7bb64152fd590a734587a0ce169f46b4c82be2.jpg)

![faa6be2dbe2001cb00db670974e3f175954cc567a0e7c8ae42460e0529083792.jpg](../icml_results/6052_Learning%20Sparse%20Visual%20Representations%20via%20Spatial-Semantic%20Factorization/tables/faa6be2dbe2001cb00db670974e3f175954cc567a0e7c8ae42460e0529083792.jpg)

<a id="paper-6004"></a>

## RECAST: Model Reconstruction via Counterfactual-Aware Wasserstein Geometry under Limited Data


### Images

![1dce888258902098c249fc43678f883699288910c37be3cf31cbfc8e1afd9827.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/1dce888258902098c249fc43678f883699288910c37be3cf31cbfc8e1afd9827.jpg)

![1ff6f5351786b06853e741ba94b82cb91199c0b014f3e0b2f1064ca9ec4c98c6.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/1ff6f5351786b06853e741ba94b82cb91199c0b014f3e0b2f1064ca9ec4c98c6.jpg)

![31f227ba479b98dd830134aef1befc5a283ac53bac095f049e82c2c29a89025a.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/31f227ba479b98dd830134aef1befc5a283ac53bac095f049e82c2c29a89025a.jpg)

![34b1de7912157679a08435bb628504c828d8288ea745f777e6dff4809be12498.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/34b1de7912157679a08435bb628504c828d8288ea745f777e6dff4809be12498.jpg)

![70bd6189bf6b6a5d80c816d098157fea0207fec3c1b4a5c29f2c81391938ba56.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/70bd6189bf6b6a5d80c816d098157fea0207fec3c1b4a5c29f2c81391938ba56.jpg)

![737d831dabd3356afda6b5307b847c049ad28b05a1b83aa25b7865f17a5f34a3.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/737d831dabd3356afda6b5307b847c049ad28b05a1b83aa25b7865f17a5f34a3.jpg)

![9cf5397be552a7251a184cb848f57480e7e6a448472d9889507cd1e1381a2f1c.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/9cf5397be552a7251a184cb848f57480e7e6a448472d9889507cd1e1381a2f1c.jpg)

![b3dc9fb24a2573c436a77a96ae2c375bc6aeab7c46157a62fcdd5792b9ba5491.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/b3dc9fb24a2573c436a77a96ae2c375bc6aeab7c46157a62fcdd5792b9ba5491.jpg)

![caf1fb15c52c06c27b64cd738d71cdf5688214c51459f3688156c675f48e163e.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/caf1fb15c52c06c27b64cd738d71cdf5688214c51459f3688156c675f48e163e.jpg)

![f4859fd1a950d2f495f657727c60562d91a73388d63ca407738bdeae850f7972.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/f4859fd1a950d2f495f657727c60562d91a73388d63ca407738bdeae850f7972.jpg)

![fd62b7d99e5698b477211fd5d34cfdf1646c4be353b620942923612819d86124.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/images/fd62b7d99e5698b477211fd5d34cfdf1646c4be353b620942923612819d86124.jpg)

### Tables

![3225c4e842f3cc97d701b2fe732a18fb1920816d5a2536e85476aab8556ee7ef.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/3225c4e842f3cc97d701b2fe732a18fb1920816d5a2536e85476aab8556ee7ef.jpg)

![8554f6bb1234b6fa9b2291f24305407c100559c214427ba536236c01344ae425.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/8554f6bb1234b6fa9b2291f24305407c100559c214427ba536236c01344ae425.jpg)

![a667a1fc1977c3c32309bf0114b56466d549cd1e32a0284a1096802889e00486.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/a667a1fc1977c3c32309bf0114b56466d549cd1e32a0284a1096802889e00486.jpg)

![b1999dfe128ab40c14c60eda5e1a0df03b3a06d4495dc613a0dc46401fe2d937.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/b1999dfe128ab40c14c60eda5e1a0df03b3a06d4495dc613a0dc46401fe2d937.jpg)

![bc7add879eb1a41a37324e2eb764e86409138c51196b5ac490938d00403f3608.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/bc7add879eb1a41a37324e2eb764e86409138c51196b5ac490938d00403f3608.jpg)

![f89e9e9e607a1da12e78cfa45642c9875f48a3deefae1b05ea71e14dcd23e00a.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/f89e9e9e607a1da12e78cfa45642c9875f48a3deefae1b05ea71e14dcd23e00a.jpg)

![fb30a0edecc485ce27a79ffd065c35fd3581768a15e3588aa83e86a5109c0b35.jpg](../icml_results/6053_RECAST_%20Model%20Reconstruction%20via%20Counterfactual-Aware%20Wasserstein%20Geometry%20under%20Limited%20Data/tables/fb30a0edecc485ce27a79ffd065c35fd3581768a15e3588aa83e86a5109c0b35.jpg)

<a id="paper-6005"></a>

## CAffNet: Hard Constraint-Affine Neural Networks


### Images

![074c230fed0c42e93117304f4621f47dca0f89dfd185dcb22aa1d462265fdf8f.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/074c230fed0c42e93117304f4621f47dca0f89dfd185dcb22aa1d462265fdf8f.jpg)

![33b2110311e429b8b829babb0c16b576f466bbcecf01d62dffade1025690d417.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/33b2110311e429b8b829babb0c16b576f466bbcecf01d62dffade1025690d417.jpg)

![33bb2356293c54c44aaaa7b094c7b58832d85a35a4385920d3923c035afec111.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/33bb2356293c54c44aaaa7b094c7b58832d85a35a4385920d3923c035afec111.jpg)

![35339df30f498d3b5feb30b12c730204123f02054622378952ccfbf31d240e2d.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/35339df30f498d3b5feb30b12c730204123f02054622378952ccfbf31d240e2d.jpg)

![3effe975b683f0046bd975bdb963906bc2fa0b3f53b42742c714df2a6f1cf0d5.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/3effe975b683f0046bd975bdb963906bc2fa0b3f53b42742c714df2a6f1cf0d5.jpg)

![78cd33e5b309c27e24f38f93e4a05e86e7994228016e3b140d4f270aa2b384e3.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/78cd33e5b309c27e24f38f93e4a05e86e7994228016e3b140d4f270aa2b384e3.jpg)

![9c2cdf5974c2cb49b8dad7b3fd36eb20da598787102b66e3c77ae9304c4e5e0b.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/9c2cdf5974c2cb49b8dad7b3fd36eb20da598787102b66e3c77ae9304c4e5e0b.jpg)

![e9ca08d7342ea02231566c6dc46da67c2e1c702e48e2096ee262d71357635dc2.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/images/e9ca08d7342ea02231566c6dc46da67c2e1c702e48e2096ee262d71357635dc2.jpg)

### Tables

![207d1f5db3171e3395013f26a1f0905745dfaf59d40795ba5c8b3fc72e88b11c.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/tables/207d1f5db3171e3395013f26a1f0905745dfaf59d40795ba5c8b3fc72e88b11c.jpg)

![4bf91ca17433e0f1feff6d131c4b3c7e7fb6cee9667585aea299dfc829029c67.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/tables/4bf91ca17433e0f1feff6d131c4b3c7e7fb6cee9667585aea299dfc829029c67.jpg)

![5081e954c2e7d3999e17c9456f480109a6ea74e8acf08ea0b8d469d799d65b57.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/tables/5081e954c2e7d3999e17c9456f480109a6ea74e8acf08ea0b8d469d799d65b57.jpg)

![593bd35d132aee897d5f32919207e943ee87ae8313a4975b0f11345924e8fcdd.jpg](../icml_results/6054_CAffNet_%20Hard%20Constraint-Affine%20Neural%20Networks/tables/593bd35d132aee897d5f32919207e943ee87ae8313a4975b0f11345924e8fcdd.jpg)

<a id="paper-6006"></a>

## Rethinking Low-Confidence Pseudo Labels: Influence-Aware Semi-Supervised Fine-Tuning for Hyperspectral Change Detection

### Images

![006b30250d471e619220066770cc6caa9154b33ef924a837acadae73bd2281c7.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/006b30250d471e619220066770cc6caa9154b33ef924a837acadae73bd2281c7.jpg)

![692cfa6b9b72c547e1854b4fb95d252c6a1d47ead11151fec4c22e5f1f7dfa5b.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/692cfa6b9b72c547e1854b4fb95d252c6a1d47ead11151fec4c22e5f1f7dfa5b.jpg)

![6cab311595c90a3b16e4402f9eb275fde9ab51b479dffdfeb1b7287a897a2691.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/6cab311595c90a3b16e4402f9eb275fde9ab51b479dffdfeb1b7287a897a2691.jpg)

![7483fc45873f268b30b2663ed9b865fc9ffbf7bd4fb7377b2bea29e01fa6d87f.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/7483fc45873f268b30b2663ed9b865fc9ffbf7bd4fb7377b2bea29e01fa6d87f.jpg)

![7e7d7938ec6058798e335ed5f789e1e7086e64c218bd0c63930a3b3f6e81cbbf.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/7e7d7938ec6058798e335ed5f789e1e7086e64c218bd0c63930a3b3f6e81cbbf.jpg)

![8702742e6bef249824c0bf83d627067640570a464e3f199f51a35d78935e51a4.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/8702742e6bef249824c0bf83d627067640570a464e3f199f51a35d78935e51a4.jpg)

![9f375310f8a38dbd74195de3f37a51f958a58e4ed7d8ece4fb5c7a1cccba5800.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/9f375310f8a38dbd74195de3f37a51f958a58e4ed7d8ece4fb5c7a1cccba5800.jpg)

![ae17599ce8fef4b458925a7caeda85129e5609a33256e1dff211efa2da349469.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/ae17599ce8fef4b458925a7caeda85129e5609a33256e1dff211efa2da349469.jpg)

![c62f2a4588fcf28052547b91ded9cb4774e75bdafb3684229a4bb49df735f9bb.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/c62f2a4588fcf28052547b91ded9cb4774e75bdafb3684229a4bb49df735f9bb.jpg)

![f906df4d115182c1ee095667d68e1d8be73b9cab7ed01ea18958c20ff83e3e00.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/f906df4d115182c1ee095667d68e1d8be73b9cab7ed01ea18958c20ff83e3e00.jpg)

### Tables

![021076733f25ad9854bbd97b46ae406867da0b25729c3e0acb50e70eb0331ead.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/021076733f25ad9854bbd97b46ae406867da0b25729c3e0acb50e70eb0331ead.jpg)

![4ac3b54dea157090da0b689151ad02ad08c8b139d07522992f41557700897717.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/4ac3b54dea157090da0b689151ad02ad08c8b139d07522992f41557700897717.jpg)

![5b83b04101bb372fa47f69603d351e9de43b526234f5f3f8d0aba5d18e5b0ea8.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/5b83b04101bb372fa47f69603d351e9de43b526234f5f3f8d0aba5d18e5b0ea8.jpg)

![8ad00627ebfb01f2326cdd07734647b1f5a3b0df56e8d7e2bb1d8666d3b9976f.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/8ad00627ebfb01f2326cdd07734647b1f5a3b0df56e8d7e2bb1d8666d3b9976f.jpg)

![91d25c9ffaa085af5a2c1ce108d5442248a2c13454e54fccb76f2b255a57112d.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/91d25c9ffaa085af5a2c1ce108d5442248a2c13454e54fccb76f2b255a57112d.jpg)

![961cec77729ca3aff2a47f2aefb8fd43d44cd4c9af71a7776a4560e35d9f436c.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/961cec77729ca3aff2a47f2aefb8fd43d44cd4c9af71a7776a4560e35d9f436c.jpg)

![c2d1268b5d6e6ab81e1b30fc102bc650acf717546225dd2ee1b361638496c1c0.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/c2d1268b5d6e6ab81e1b30fc102bc650acf717546225dd2ee1b361638496c1c0.jpg)

![e02452acc7195615d4dbdd4c19def0ceb8355283f73ee0c26cfde8f6509dae1b.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/e02452acc7195615d4dbdd4c19def0ceb8355283f73ee0c26cfde8f6509dae1b.jpg)

![e40ea90c4106c38b85269c6cb4303f163ce1b89451b2532e8134914fd45c5c29.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/e40ea90c4106c38b85269c6cb4303f163ce1b89451b2532e8134914fd45c5c29.jpg)
