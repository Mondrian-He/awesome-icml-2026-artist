# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 127 of 197

## 目录 (Table of Contents)

1. [Fast and Scalable Analytical Diffusion](#paper-4159)
2. [COGNOS: Universal Enhancement for Time Series Anomaly Detection via Constrained Gaussian-Noise Optimization and Smoothing](#paper-4160)
3. [DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving](#paper-4161)
4. [One Batch Is Enough: A Unified Dataset Condensation Framework for General Time Series Analysis](#paper-4162)
5. [The Pareto-optimal Trade-off between Regret and Statistical Inference in Linear Stochastic Bandits under Safety Constraints](#paper-4163)
6. [Spurious Rewards: Rethinking Training Signals in RLVR](#paper-4164)
7. [MonoScale: Scaling Multi-Agent System with Monotonic Improvement](#paper-4165)
8. [LiveFigure: Generating Editable Scientific Illustration with VLM Agents](#paper-4166)
9. [Block Rotation is All You Need for MXFP4 Quantization](#paper-4167)
10. [SciNet: Evaluating AI Agents in Relation-Aware Scientific Literature Retrieval](#paper-4168)
11. [Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](#paper-4169)
12. [LIVEditor-14B: Lightning Unified Video Editing via In-Context Sparse Attention](#paper-4170)
13. [CONGA:Confidence-and-Gradient-Aware Learning Rate Schedule for Test Time Adaptation](#paper-4171)
14. [How RLHF Amplifies Sycophancy](#paper-4172)
15. [Intentional Updates for Streaming Reinforcement Learning](#paper-4173)
16. [Who’s in Charge? Disempowerment Patterns in Real-World LLM Usage](#paper-4174)
17. [RubricRobustness: Evaluating the Sensitivity of Rubrics-Based Benchmarks to Simple Perturbations](#paper-4175)
18. [The Realignment Problem: When Right becomes Wrong in LLMs](#paper-4176)
19. [Ariadne’s Thread of LipSync: Unraveling Forgeries via Inconsistency between Lip Motions and Head Poses](#paper-4177)
20. [Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices](#paper-4178)
21. [Laplacian Representations for Decision-Time Planning](#paper-4179)
22. [MMClima: A Framework for Multimodal Climate Science Data and Evaluation](#paper-4180)
23. [DF-LoGiT: Data-Free Logic-Gated Backdoor Attacks in Vision Transformers](#paper-4181)
24. [CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks](#paper-4182)
25. [IDRBench: Understanding the Capability of Large Language Models on Interdisciplinary Research](#paper-4183)
26. [Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](#paper-4184)
27. [Breaking the Scale Barrier: One-Shot Knowledge Transfer via Frequency Transform](#paper-4185)
28. [Rényi Diffusion Models](#paper-4186)
29. [ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling](#paper-4187)
30. [Training-Trajectory-Aware Token Selection](#paper-4188)
31. [TRIP-Bench: A Benchmark for Long-Horizon Interactive Agents in Real-World Scenarios](#paper-4189)
32. [SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space](#paper-4190)
33. [TEFormer: Structured Bidirectional Temporal Enhancement Modeling in Spiking Transformers](#paper-4191)

---


<a id="paper-4159"></a>

## Fast and Scalable Analytical Diffusion

### Images

![b4109edbad0160b81730939f0dc93ca72df694f3acabe9298e9abc4f40adca9f.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/b4109edbad0160b81730939f0dc93ca72df694f3acabe9298e9abc4f40adca9f.jpg)

![b45a8a5d4aa64a4efc9a07b45dfef2d22ab619a70b4648f4a3341e3cb9f922e3.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/b45a8a5d4aa64a4efc9a07b45dfef2d22ab619a70b4648f4a3341e3cb9f922e3.jpg)

![bf287b4e79e81ed3727fb8a047a02a38bbdb22c52128b8537250abe3740287d9.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/bf287b4e79e81ed3727fb8a047a02a38bbdb22c52128b8537250abe3740287d9.jpg)

![f2aa47f53ebb4920142efb23b6080a1130be4458d0e5af58122d37273cab0245.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/f2aa47f53ebb4920142efb23b6080a1130be4458d0e5af58122d37273cab0245.jpg)

![f341cc195b2e1b29cd6119b9f97e4e701fbb3aebadeccc92671296806cc94a9c.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/f341cc195b2e1b29cd6119b9f97e4e701fbb3aebadeccc92671296806cc94a9c.jpg)

![f6b0e52f30b89722036ff84cb65062cc4dc6f28cf0dee1485d82a3f54b6cf04f.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/images/f6b0e52f30b89722036ff84cb65062cc4dc6f28cf0dee1485d82a3f54b6cf04f.jpg)

### Tables

![32a9ccf7d4ba4e88b3c5445cd7eb3794ae877b47dcbf39146fb833d9088b92b9.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/32a9ccf7d4ba4e88b3c5445cd7eb3794ae877b47dcbf39146fb833d9088b92b9.jpg)

![3951ba30a707f4f5ef8ad8e7484444dfb0573de15fb303f695386acb9b7a7b48.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/3951ba30a707f4f5ef8ad8e7484444dfb0573de15fb303f695386acb9b7a7b48.jpg)

![4d235811dc50a1a4c69958645834a97e11b25aca21035b327a7ccd6c874cce10.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/4d235811dc50a1a4c69958645834a97e11b25aca21035b327a7ccd6c874cce10.jpg)

![6ce700d2a23f7f2812e85d57d14424d8413798d01273da9c329b012a8e35256f.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/6ce700d2a23f7f2812e85d57d14424d8413798d01273da9c329b012a8e35256f.jpg)

![6d060f4a15caeb64f3b3322aa4ebc6b22913d8f44f6e5ebc5b90fae41e853d30.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/6d060f4a15caeb64f3b3322aa4ebc6b22913d8f44f6e5ebc5b90fae41e853d30.jpg)

![b041497d3f9882f1e902dcc31227cbb83d6f35473d7ee315a1d178069b2014fb.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/b041497d3f9882f1e902dcc31227cbb83d6f35473d7ee315a1d178069b2014fb.jpg)

![c00b2c3f0b40240358b9e1cec87cb20d3824caebed46240fe990a1da97aeb5b7.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/c00b2c3f0b40240358b9e1cec87cb20d3824caebed46240fe990a1da97aeb5b7.jpg)

![c988e29c12a4d4b9d5341f63bfcf567bcdeaa8c0ade26fa52ca3a3719a870f76.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/c988e29c12a4d4b9d5341f63bfcf567bcdeaa8c0ade26fa52ca3a3719a870f76.jpg)

![c9db960c99973be4da16b8dc0f0f41379487492ba87f1de2a0744feb4b734f9c.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/c9db960c99973be4da16b8dc0f0f41379487492ba87f1de2a0744feb4b734f9c.jpg)

![cd86ed3e013111e716d87edafc90714dd989ffa2222799eb30fdf7cd5f83c0b5.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/cd86ed3e013111e716d87edafc90714dd989ffa2222799eb30fdf7cd5f83c0b5.jpg)

![d27c5b6b9eb083b7c2aba39fcad9f4c9b42decbb14e71420f2feaf465238920b.jpg](../icml_results/4200_Training%E2%80%93Inference%20Consistent%20Segmented%20Execution%20for%20Long-Context%20LLMs/tables/d27c5b6b9eb083b7c2aba39fcad9f4c9b42decbb14e71420f2feaf465238920b.jpg)

## Fast and Scalable Analytical Diffusion


### Images

![4066b4b5ed464cc5eb256c6173afbe157ec47a22187aba93b0d38dacddc80931.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/4066b4b5ed464cc5eb256c6173afbe157ec47a22187aba93b0d38dacddc80931.jpg)

![671d69d398ea630ed98ae08e07a59ab43d9072aa3f60ccd8d4a3586d8bd5e76e.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/671d69d398ea630ed98ae08e07a59ab43d9072aa3f60ccd8d4a3586d8bd5e76e.jpg)

![a5bc8fb2a75f9ed19ea2ef3a2db22d7a67b703ff440804421af8945155d528a7.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/a5bc8fb2a75f9ed19ea2ef3a2db22d7a67b703ff440804421af8945155d528a7.jpg)

![c9f02e170bb5a2f4f2564f542999082cc84da946c55c3d7e8c05a8206e7a62c2.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/c9f02e170bb5a2f4f2564f542999082cc84da946c55c3d7e8c05a8206e7a62c2.jpg)

![d2155af17169397aeb64bccf897f1ac8bba66b43154025d44df2b1b2ef27ada1.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/d2155af17169397aeb64bccf897f1ac8bba66b43154025d44df2b1b2ef27ada1.jpg)

![f940ac9beddd48c01112191289aab6af8c11fee27119e78db96d7555e96b6739.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/images/f940ac9beddd48c01112191289aab6af8c11fee27119e78db96d7555e96b6739.jpg)

### Tables

![5c59c4bbcb03e23b81f7e8af9d4164f12ad33b600b1a533dbbf7bc79c79b1382.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/5c59c4bbcb03e23b81f7e8af9d4164f12ad33b600b1a533dbbf7bc79c79b1382.jpg)

![60f7f5d64f850c356f12ed912f6540658d82cfcad9782fb1c675726a110db471.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/60f7f5d64f850c356f12ed912f6540658d82cfcad9782fb1c675726a110db471.jpg)

![8aff0a5c4657e2c1a0916f2eb110b10bafca95e0ce38d480a9410746859dd2b0.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/8aff0a5c4657e2c1a0916f2eb110b10bafca95e0ce38d480a9410746859dd2b0.jpg)

![8eba80981d055c22b676b418fc86e9282336794cc8e2c482370e87e7a295bace.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/8eba80981d055c22b676b418fc86e9282336794cc8e2c482370e87e7a295bace.jpg)

![8f0a7e13e1d9e841c255bc5affa3fdadbf42539a99b8dc71f4e00a09eb488f1f.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/8f0a7e13e1d9e841c255bc5affa3fdadbf42539a99b8dc71f4e00a09eb488f1f.jpg)

![a7bbbb64146bf43ccd47b78dcb6e84b9967ddf42c1efb2997509afbdc181d64e.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/a7bbbb64146bf43ccd47b78dcb6e84b9967ddf42c1efb2997509afbdc181d64e.jpg)

![b540336cfd3092b8c60e3380f87b51cb3570a9b61e2c805086271c1673cf2119.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/b540336cfd3092b8c60e3380f87b51cb3570a9b61e2c805086271c1673cf2119.jpg)

![c5fc077734070433de555b3eb31ad8a86d14c51aed7ee1a5af0ce3e4112cc3bb.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/c5fc077734070433de555b3eb31ad8a86d14c51aed7ee1a5af0ce3e4112cc3bb.jpg)

![cb738ad1591aefe747b3c3807daf44c9c4681d7d14e31ffa579e40aea0aab692.jpg](../icml_results/4201_Fast%20and%20Scalable%20Analytical%20Diffusion/tables/cb738ad1591aefe747b3c3807daf44c9c4681d7d14e31ffa579e40aea0aab692.jpg)

<a id="paper-4160"></a>

## COGNOS: Universal Enhancement for Time Series Anomaly Detection via Constrained Gaussian-Noise Optimization and Smoothing


### Images

![312061ef03bf802dfc25a7a3c5e35936019ad201d87566045ba6c8e48eb077a6.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/312061ef03bf802dfc25a7a3c5e35936019ad201d87566045ba6c8e48eb077a6.jpg)

![3a8581668c303c61c6be0ebcb7b1d56479d8a8866d1c2915ee35e18fffc415f3.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/3a8581668c303c61c6be0ebcb7b1d56479d8a8866d1c2915ee35e18fffc415f3.jpg)

![43638b00ef101ee9cccb4a19601026050d57f2df5e2d6ea9e44341dc8ccbe925.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/43638b00ef101ee9cccb4a19601026050d57f2df5e2d6ea9e44341dc8ccbe925.jpg)

![7ac08226fe4c458caf517ba611aee4e7e30bdb6a5b03c9835f10b1e66d8434c2.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/7ac08226fe4c458caf517ba611aee4e7e30bdb6a5b03c9835f10b1e66d8434c2.jpg)

![b74c36f77b3b572f9f03af2c1da08e8947d727b56cd7db7be28e0110d6f489cc.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/b74c36f77b3b572f9f03af2c1da08e8947d727b56cd7db7be28e0110d6f489cc.jpg)

![bb6b69d1ef380efdfc302df0d7b7ababeeb0070f292b3d6821394f776ed504bb.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/bb6b69d1ef380efdfc302df0d7b7ababeeb0070f292b3d6821394f776ed504bb.jpg)

![bfdbec0e9b25c27e91dcaa951138c24b841d3d74c4eef1fdb0e6bd2f0f15c3ca.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/bfdbec0e9b25c27e91dcaa951138c24b841d3d74c4eef1fdb0e6bd2f0f15c3ca.jpg)

![ea5986c3be056c063fed8fe4b0db2f76fb9e088983d913a00953a963264f076a.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/ea5986c3be056c063fed8fe4b0db2f76fb9e088983d913a00953a963264f076a.jpg)

![f22a2c25aa962f96642d7d5812c76a8d26ab465791aa0060500bbb6c3d28f11b.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/f22a2c25aa962f96642d7d5812c76a8d26ab465791aa0060500bbb6c3d28f11b.jpg)

![f4b583a017a497d0168aca057ff2a34c19797b8493a46633b9971bb94919d513.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/images/f4b583a017a497d0168aca057ff2a34c19797b8493a46633b9971bb94919d513.jpg)

### Tables

![022d9590d7d68ee2a48f6b36cac20916b818076274749749d17386c041943b57.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/022d9590d7d68ee2a48f6b36cac20916b818076274749749d17386c041943b57.jpg)

![1761be78e4a8e924ad12f1d7c47b80bfca1c013d5b1417a648a124fbebf02669.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/1761be78e4a8e924ad12f1d7c47b80bfca1c013d5b1417a648a124fbebf02669.jpg)

![1ae69c47dafcabcb4e5bbcbfbfd746d14f2d04aaabda4c778003a75aa690b84c.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/1ae69c47dafcabcb4e5bbcbfbfd746d14f2d04aaabda4c778003a75aa690b84c.jpg)

![26e31d474795b54a8e6768350d25fe3f07df4238a253efd8ea782edecfd2182c.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/26e31d474795b54a8e6768350d25fe3f07df4238a253efd8ea782edecfd2182c.jpg)

![36001083b1a5419a5e3b1cc974b32f8ed38c6263a3da230678625259139d9e10.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/36001083b1a5419a5e3b1cc974b32f8ed38c6263a3da230678625259139d9e10.jpg)

![813bfab810915cdab23a390ed802654a95c40a9112f67c9e98968c05753ad990.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/813bfab810915cdab23a390ed802654a95c40a9112f67c9e98968c05753ad990.jpg)

![87b4889654586ff9879129297a994a1e34f954918a768d5b3e4871db80fe82c2.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/87b4889654586ff9879129297a994a1e34f954918a768d5b3e4871db80fe82c2.jpg)

![9299c3d4ec5c7be0ef8b914da528ebcddf532db73f209d567c43bf8dd2e14234.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/9299c3d4ec5c7be0ef8b914da528ebcddf532db73f209d567c43bf8dd2e14234.jpg)

![d78439c2f0e7038fb6a3672f8fa3fe4547f798f13fd7345939bed41226aa3e60.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/d78439c2f0e7038fb6a3672f8fa3fe4547f798f13fd7345939bed41226aa3e60.jpg)

![f83b27ebf670f03c3da8df65ea5363510e5f464c66a70fd613642630a4933a86.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/f83b27ebf670f03c3da8df65ea5363510e5f464c66a70fd613642630a4933a86.jpg)

![fc28e2bb58a5c5c707c6523c42178f40e7e9a7313920ffad73803417024a53bd.jpg](../icml_results/4202_COGNOS_%20Universal%20Enhancement%20for%20Time%20Series%20Anomaly%20Detection%20via%20Constrained%20Gaussian-Noise%20Optim/tables/fc28e2bb58a5c5c707c6523c42178f40e7e9a7313920ffad73803417024a53bd.jpg)

<a id="paper-4161"></a>

## DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving


### Images

![059d174aaae59a31bc14d0807af2da2e5fcc62626a0ee8720bdcf2f87a043c5b.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/059d174aaae59a31bc14d0807af2da2e5fcc62626a0ee8720bdcf2f87a043c5b.jpg)

![13718f004645e6658f6b5f8f94b64ee07db305f443e11087526a47b3b66db7e0.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/13718f004645e6658f6b5f8f94b64ee07db305f443e11087526a47b3b66db7e0.jpg)

![1611a6509d65db5cb55b9ff9274fd1b019c8a49e8642899fdbf940d5acdb93d7.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/1611a6509d65db5cb55b9ff9274fd1b019c8a49e8642899fdbf940d5acdb93d7.jpg)

![2aef4a8d3a0005df3fef48abdb7251e773fbbccd5a3742b7bdf0672d097dcb81.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/2aef4a8d3a0005df3fef48abdb7251e773fbbccd5a3742b7bdf0672d097dcb81.jpg)

![2e9036a08503e1d73693ac3819985794a084bb66aaeedb3dd8efba29a3f83f09.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/2e9036a08503e1d73693ac3819985794a084bb66aaeedb3dd8efba29a3f83f09.jpg)

![31de8506407731a086cee46f83e29ddfaa4acbdeb3955d9c307255eaea2174b6.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/31de8506407731a086cee46f83e29ddfaa4acbdeb3955d9c307255eaea2174b6.jpg)

![47c58dfc8fcde1bd93e7dcf50274ffa04e84f548073b4e9a91ac2c5fcad4793e.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/47c58dfc8fcde1bd93e7dcf50274ffa04e84f548073b4e9a91ac2c5fcad4793e.jpg)

![6ea2131ab21ea8fd6e02ed765eea6c01b149d4b0cbd8cac153698e937f6f596e.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/6ea2131ab21ea8fd6e02ed765eea6c01b149d4b0cbd8cac153698e937f6f596e.jpg)

![cdf42603a0018fa6f5b8cbb53c8c6c650a9b7be73f6794e8be6e2dcb2ec71b86.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/cdf42603a0018fa6f5b8cbb53c8c6c650a9b7be73f6794e8be6e2dcb2ec71b86.jpg)

![e4f298c60686e09109fb9a2f39cc679adfcb8892d3239fdd616bd7e55d62e50c.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/images/e4f298c60686e09109fb9a2f39cc679adfcb8892d3239fdd616bd7e55d62e50c.jpg)

### Tables

![0dd658bf53a271bab006b47beeee0ed8af2b1564411f1d92b64fadc8283ae284.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/0dd658bf53a271bab006b47beeee0ed8af2b1564411f1d92b64fadc8283ae284.jpg)

![226cb2a16f3a5c2110ed4fa51a8b148ef6b86141317d99adb68088362a50aec3.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/226cb2a16f3a5c2110ed4fa51a8b148ef6b86141317d99adb68088362a50aec3.jpg)

![39375c5d066d22dbb4c95e739e73e5ab2c67b63c98169fbdfdf31b646aad6223.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/39375c5d066d22dbb4c95e739e73e5ab2c67b63c98169fbdfdf31b646aad6223.jpg)

![569f922f1081c5d381af05d2a85f5314ce4cb1563c60073092dfa9e49b1ecda7.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/569f922f1081c5d381af05d2a85f5314ce4cb1563c60073092dfa9e49b1ecda7.jpg)

![60011492d189ad88bb88ce0ff0ace378f365b15e778ad56b49f0321ea84619d6.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/60011492d189ad88bb88ce0ff0ace378f365b15e778ad56b49f0321ea84619d6.jpg)

![761a79c0a595fc0728a8f449f2c7c9d4dd4cc9afa38ef751605c799095a9d2d9.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/761a79c0a595fc0728a8f449f2c7c9d4dd4cc9afa38ef751605c799095a9d2d9.jpg)

![b3fd9c94e626baedc5614ae1052efbf645deb87fb9d20f6af7a2743f1c3580ea.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/b3fd9c94e626baedc5614ae1052efbf645deb87fb9d20f6af7a2743f1c3580ea.jpg)

![ea3a4f252850e819030b527c96b46d9f254dca3316142e00de77d387b97a5186.jpg](../icml_results/4203_DynVLA_%20Learning%20World%20Dynamics%20for%20Action%20Reasoning%20in%20Autonomous%20Driving/tables/ea3a4f252850e819030b527c96b46d9f254dca3316142e00de77d387b97a5186.jpg)

<a id="paper-4162"></a>

## One Batch Is Enough: A Unified Dataset Condensation Framework for General Time Series Analysis


### Images

![0c025ceb6a0aae9d2df56feecefd99d64aafeee25556346df4b7f83d3a8b173f.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/0c025ceb6a0aae9d2df56feecefd99d64aafeee25556346df4b7f83d3a8b173f.jpg)

![14c94f0e75371e5631451dd583bdd937a232b71ba257fc4089dfafcc9e02b34e.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/14c94f0e75371e5631451dd583bdd937a232b71ba257fc4089dfafcc9e02b34e.jpg)

![32998b6f924a0110ba4f4417ede59a06f07c312c96e895b2ea09f069cb4de708.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/32998b6f924a0110ba4f4417ede59a06f07c312c96e895b2ea09f069cb4de708.jpg)

![3fffa0fb6f6c57970b8eac2c173bc2e1f2ea03a53bf47db8169fbe733a96e778.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/3fffa0fb6f6c57970b8eac2c173bc2e1f2ea03a53bf47db8169fbe733a96e778.jpg)

![515fb9df1b78ea9776fb3e94059d3cba1851cadefb427c80e72d7f368d6f0b14.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/515fb9df1b78ea9776fb3e94059d3cba1851cadefb427c80e72d7f368d6f0b14.jpg)

![6471150d2bd3f3152a2d49966f0655af9a3e03233af7fe1f999cac4abc1698ba.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/6471150d2bd3f3152a2d49966f0655af9a3e03233af7fe1f999cac4abc1698ba.jpg)

![67f7de1d569fa09bfde0cbee6393ce42e0a99f4555e1ec5619fa9d464cf38eac.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/67f7de1d569fa09bfde0cbee6393ce42e0a99f4555e1ec5619fa9d464cf38eac.jpg)

![9cb146195db3763e65c515732cb46feb027a86ee729d6eeaf26c243385cb2ae2.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/9cb146195db3763e65c515732cb46feb027a86ee729d6eeaf26c243385cb2ae2.jpg)

![9ff4248dbd163771eefad841840d19cdd126a6ef7a44931f119100e7cfff288b.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/9ff4248dbd163771eefad841840d19cdd126a6ef7a44931f119100e7cfff288b.jpg)

![bfe5f3c35dbfe40e6ecc5ff1b70418974f4ebfea02150843d31a99844de9e438.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/bfe5f3c35dbfe40e6ecc5ff1b70418974f4ebfea02150843d31a99844de9e438.jpg)

![e24ab468eba663458ac82b21750daf607cdecba5212f8ff41cdc4e3417fcba39.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/e24ab468eba663458ac82b21750daf607cdecba5212f8ff41cdc4e3417fcba39.jpg)

![ea413048a6b1a12c221f7b2f59f44016533734010aa79fd3f9dcf89b3d3a2c7e.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/images/ea413048a6b1a12c221f7b2f59f44016533734010aa79fd3f9dcf89b3d3a2c7e.jpg)

### Tables

![1de7cd4e34b8a694fc300ac0152a895059c001694fe3f56a023217e830c3e5e8.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/1de7cd4e34b8a694fc300ac0152a895059c001694fe3f56a023217e830c3e5e8.jpg)

![39e605f171a17f955aef498f7ed2396ca3d8ac08e277f3fd716f22cba9733221.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/39e605f171a17f955aef498f7ed2396ca3d8ac08e277f3fd716f22cba9733221.jpg)

![3d46a7627ea398bbea4ae5e4dec536dd4c7a84a562ca015d2a9e2cb5a84e0258.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/3d46a7627ea398bbea4ae5e4dec536dd4c7a84a562ca015d2a9e2cb5a84e0258.jpg)

![6b0f76c8606e9d29836f22a961af27c4719b80380b6b21b0089cf3180cf96d98.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/6b0f76c8606e9d29836f22a961af27c4719b80380b6b21b0089cf3180cf96d98.jpg)

![6cb332fbd98bcdf45c29dc0adfe37d659b77dc6ee5fc1413fc6dc7518e69d4f2.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/6cb332fbd98bcdf45c29dc0adfe37d659b77dc6ee5fc1413fc6dc7518e69d4f2.jpg)

![7824375df1ca832a17053c65ab4a2673cac6e83b8b037b8bba86539a0f48776c.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/7824375df1ca832a17053c65ab4a2673cac6e83b8b037b8bba86539a0f48776c.jpg)

![86f2b3fa80baf837374de48b127da3320d37b7c542b2550a99db95f79227585e.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/86f2b3fa80baf837374de48b127da3320d37b7c542b2550a99db95f79227585e.jpg)

![b1212488a4ec8b51e9678f09c5da5df5dd58a5f92669f2c75ce3a6ecbfc7eaf1.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/b1212488a4ec8b51e9678f09c5da5df5dd58a5f92669f2c75ce3a6ecbfc7eaf1.jpg)

![bacc16d0bbd1b41f7e5de6a30146203297f48db79984c24a97ad2f347f4f2f80.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/bacc16d0bbd1b41f7e5de6a30146203297f48db79984c24a97ad2f347f4f2f80.jpg)

![c5dd98772d69f9e9b376084a116fa86969a12ac9df1faff664ae74860eb97ab7.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/c5dd98772d69f9e9b376084a116fa86969a12ac9df1faff664ae74860eb97ab7.jpg)

![d252f43951813cb4beff14f255e632fc7741ba7f2f9535cdccb8c057093c6345.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/d252f43951813cb4beff14f255e632fc7741ba7f2f9535cdccb8c057093c6345.jpg)

![efb47e4577c328c01ac67507f0a0390a11edd84f00b58ec5d3fe49e3aff4ecdc.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/efb47e4577c328c01ac67507f0a0390a11edd84f00b58ec5d3fe49e3aff4ecdc.jpg)

![fae1a6392ff26f2a008f9d9d731f5308d39b15dd1200fd152dca188cc72e5748.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/fae1a6392ff26f2a008f9d9d731f5308d39b15dd1200fd152dca188cc72e5748.jpg)

![fe67e1b5d3abf9fc2b27b5028b9ca591f436d10a63dff7723214309da6eda0d6.jpg](../icml_results/4204_One%20Batch%20Is%20Enough_%20A%20Unified%20Dataset%20Condensation%20Framework%20for%20General%20Time%20Series%20Analysis/tables/fe67e1b5d3abf9fc2b27b5028b9ca591f436d10a63dff7723214309da6eda0d6.jpg)

<a id="paper-4163"></a>

## The Pareto-optimal Trade-off between Regret and Statistical Inference in Linear Stochastic Bandits under Safety Constraints


### Images

![11ea2a5d04768d0509c46165ffcb95970ef3374f35b6f19bf98c17d9623696d4.jpg](../icml_results/4205_The%20Pareto-optimal%20Trade-off%20between%20Regret%20and%20Statistical%20Inference%20in%20Linear%20Stochastic%20Bandits%20u/images/11ea2a5d04768d0509c46165ffcb95970ef3374f35b6f19bf98c17d9623696d4.jpg)

![cf0f99bd34b667b5e0fe78e38e57548d762951d30d116ebdcd60e6d9c0a0ae0b.jpg](../icml_results/4205_The%20Pareto-optimal%20Trade-off%20between%20Regret%20and%20Statistical%20Inference%20in%20Linear%20Stochastic%20Bandits%20u/images/cf0f99bd34b667b5e0fe78e38e57548d762951d30d116ebdcd60e6d9c0a0ae0b.jpg)

### Tables

![81df097a9515d456756453cfc7964d444d5d0d5850d7bb98c61be7709025d8f5.jpg](../icml_results/4205_The%20Pareto-optimal%20Trade-off%20between%20Regret%20and%20Statistical%20Inference%20in%20Linear%20Stochastic%20Bandits%20u/tables/81df097a9515d456756453cfc7964d444d5d0d5850d7bb98c61be7709025d8f5.jpg)

<a id="paper-4164"></a>

## Spurious Rewards: Rethinking Training Signals in RLVR


### Images

![0db76751afe2543bdf87b46731944192d2b2793909346dd4b62e30b1ba736e04.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/0db76751afe2543bdf87b46731944192d2b2793909346dd4b62e30b1ba736e04.jpg)

![11363a166053b5ce1295b2d8ed0f195f77237b89a9aae35de732ce426240f01c.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/11363a166053b5ce1295b2d8ed0f195f77237b89a9aae35de732ce426240f01c.jpg)

![16723f3ee02a75a48273132b65a1aa37c7f87233e4aa3812d923556f352642a4.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/16723f3ee02a75a48273132b65a1aa37c7f87233e4aa3812d923556f352642a4.jpg)

![2d2268f41dde6cfdede1831c73b426cb260e1b3900f505b6fc5f26f1ddc67c00.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/2d2268f41dde6cfdede1831c73b426cb260e1b3900f505b6fc5f26f1ddc67c00.jpg)

![3142dcadd4606e016a17b685108c59a2b6019faa2c839222430d220a79f58371.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/3142dcadd4606e016a17b685108c59a2b6019faa2c839222430d220a79f58371.jpg)

![429d0f0d48aca8507e10c1def3eb14c09b856c6a156e60d3c5518b91e730e9b1.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/429d0f0d48aca8507e10c1def3eb14c09b856c6a156e60d3c5518b91e730e9b1.jpg)

![4a54063e1d30063b594dac37efcf9432aa06aaa2f270ec4010eafc65bcaadb2a.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/4a54063e1d30063b594dac37efcf9432aa06aaa2f270ec4010eafc65bcaadb2a.jpg)

![5810698322b9135980356033a597d1a30c81d13a18d5aa4a29180703a03fd3f1.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/5810698322b9135980356033a597d1a30c81d13a18d5aa4a29180703a03fd3f1.jpg)

![6ceccfa6f6382311e3c0e5ed47758d9957ca325c1ac0fd5f70352465cfca80d9.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/6ceccfa6f6382311e3c0e5ed47758d9957ca325c1ac0fd5f70352465cfca80d9.jpg)

![6f49105983fffa621dada10e2f96ed36df730a55926de169bd89c37b963fcaa2.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/6f49105983fffa621dada10e2f96ed36df730a55926de169bd89c37b963fcaa2.jpg)

![7addaa5c06572f6088de0d3d4a160bd92f5edd7443ac6710477f267ebd6fd5ec.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/7addaa5c06572f6088de0d3d4a160bd92f5edd7443ac6710477f267ebd6fd5ec.jpg)

![8cd6e796e2157e94c75500f0be1235acaacd910b5ce8e3a07b936477f7e6c52e.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/8cd6e796e2157e94c75500f0be1235acaacd910b5ce8e3a07b936477f7e6c52e.jpg)

![9949631f7ed2977a899b688088fecb9ed8ec65283a19887b711d151e7a8b86b2.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/9949631f7ed2977a899b688088fecb9ed8ec65283a19887b711d151e7a8b86b2.jpg)

![a55bf82bbacdc31c92d1a59ae9c33873b272fe00b6f4e8cf57ab6c999e4d4641.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/a55bf82bbacdc31c92d1a59ae9c33873b272fe00b6f4e8cf57ab6c999e4d4641.jpg)

![a5e5ca712ee95eb2fb8e383448d8cb943e4cb01b700b32463e09df8233233603.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/a5e5ca712ee95eb2fb8e383448d8cb943e4cb01b700b32463e09df8233233603.jpg)

![aa7226da5b4bfb6f02e2bae1639c61b30208b2cca3bce3f85cf0b1b305deb9fb.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/aa7226da5b4bfb6f02e2bae1639c61b30208b2cca3bce3f85cf0b1b305deb9fb.jpg)

![ab7881fc5e50f7244fb1b5f922346d4b8451cc8d4ec0c8677369412f4fd5e355.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/ab7881fc5e50f7244fb1b5f922346d4b8451cc8d4ec0c8677369412f4fd5e355.jpg)

![ac8f886e9a91da71f8df0881a17a4938f11f09c163258b9599b9cfbf5081fbb1.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/ac8f886e9a91da71f8df0881a17a4938f11f09c163258b9599b9cfbf5081fbb1.jpg)

![af8d9319f7dc237c84f3ff0acc06822ba90d71428d5e4cee4b9393eb6f64f508.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/af8d9319f7dc237c84f3ff0acc06822ba90d71428d5e4cee4b9393eb6f64f508.jpg)

![ba6764ca308d16ebdd9e117430d708d317f77f31edae611292dae05b357887f8.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/ba6764ca308d16ebdd9e117430d708d317f77f31edae611292dae05b357887f8.jpg)

![c60c62e596dd50ba992248cfbcd6b7f80aaef37bc24e969702f30f2c70134b15.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/c60c62e596dd50ba992248cfbcd6b7f80aaef37bc24e969702f30f2c70134b15.jpg)

![da6cb8ab5bd327957359ebe984e70f5db4cafdd616ea4df8fc8d5a567eb2273e.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/da6cb8ab5bd327957359ebe984e70f5db4cafdd616ea4df8fc8d5a567eb2273e.jpg)

![dd567b9b51226786e75a009e1f661aa243c29ecc52a75320c2f234d9dcb48c2d.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/dd567b9b51226786e75a009e1f661aa243c29ecc52a75320c2f234d9dcb48c2d.jpg)

![e0f73301f31822ded76eb621e67d856da2a56064e76a502dbe55ed83351e0005.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/e0f73301f31822ded76eb621e67d856da2a56064e76a502dbe55ed83351e0005.jpg)

![f6c61b04b32eea0b767ce64058ece7eb753d67ad484d023bec68c00879dae970.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/images/f6c61b04b32eea0b767ce64058ece7eb753d67ad484d023bec68c00879dae970.jpg)

### Tables

![4b359d8c472c5059c01d132a279ca94f2ba6cae0c5388ef9c6f4be7e933c22a6.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/tables/4b359d8c472c5059c01d132a279ca94f2ba6cae0c5388ef9c6f4be7e933c22a6.jpg)

![a3f1b1379cfa4e09ba6159bd9b6a6231dbbf082f011c253b66f8d1ee7a6bd83f.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/tables/a3f1b1379cfa4e09ba6159bd9b6a6231dbbf082f011c253b66f8d1ee7a6bd83f.jpg)

![c7af764244079a560581121a9773a35a93373ef8c1d9487f9f7bd5a985a59a6c.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/tables/c7af764244079a560581121a9773a35a93373ef8c1d9487f9f7bd5a985a59a6c.jpg)

![ce69de3ea03cb69bfc53c111b06f796bd27c1b29d68a42349200681de9675c28.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/tables/ce69de3ea03cb69bfc53c111b06f796bd27c1b29d68a42349200681de9675c28.jpg)

![efa2da3a90f4e352196a0733114592388c556cc84a429ae3e0700fe4a0dff78b.jpg](../icml_results/4206_Spurious%20Rewards_%20Rethinking%20Training%20Signals%20in%20RLVR/tables/efa2da3a90f4e352196a0733114592388c556cc84a429ae3e0700fe4a0dff78b.jpg)

<a id="paper-4165"></a>

## MonoScale: Scaling Multi-Agent System with Monotonic Improvement


### Images

![49363839424e097984158c2640d6c77dfd49cea56f0f7341559156adf49c10db.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/images/49363839424e097984158c2640d6c77dfd49cea56f0f7341559156adf49c10db.jpg)

![4f3dab377c611f5db0ded84d9be9dd0ee4789f3acb515998d59a2b8cd4c908b5.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/images/4f3dab377c611f5db0ded84d9be9dd0ee4789f3acb515998d59a2b8cd4c908b5.jpg)

![5e54a8a71996393a8d6f7eb43eebafd49dabbc36447d8eff7aa0aa25a1d7da53.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/images/5e54a8a71996393a8d6f7eb43eebafd49dabbc36447d8eff7aa0aa25a1d7da53.jpg)

![62a7260d58a994bc519c740ccc7e6f1a88465fb5b1a1b4204787caf2031ae514.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/images/62a7260d58a994bc519c740ccc7e6f1a88465fb5b1a1b4204787caf2031ae514.jpg)

![98b972c44823e1407068bce625e2791d770d19f4aeb685a47ad1e3296bc5314b.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/images/98b972c44823e1407068bce625e2791d770d19f4aeb685a47ad1e3296bc5314b.jpg)

### Tables

![348615c1ff05e92ac000a5afb09d73d175dc2869f01bc9501c45c324b427153f.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/348615c1ff05e92ac000a5afb09d73d175dc2869f01bc9501c45c324b427153f.jpg)

![6862327b05f48f5a5930c62f31411d4d5d7de51a8f257e4bf4e10b02a6cccb34.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/6862327b05f48f5a5930c62f31411d4d5d7de51a8f257e4bf4e10b02a6cccb34.jpg)

![7088ba047f735d8676c0bdc556b2c46ee18dd685234d20a113ca0679eb2f273e.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/7088ba047f735d8676c0bdc556b2c46ee18dd685234d20a113ca0679eb2f273e.jpg)

![74df336b40cd98ac3d2035bdf389fae17ead52747c183a72c0f547da5b2a9dbb.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/74df336b40cd98ac3d2035bdf389fae17ead52747c183a72c0f547da5b2a9dbb.jpg)

![a05a5faa14cb7a611d25019bf7c4ed7d998cefbf9e03c3d5aa67203fc7802b51.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/a05a5faa14cb7a611d25019bf7c4ed7d998cefbf9e03c3d5aa67203fc7802b51.jpg)

![b061702e0266e511ffe36f9dee962bb5a332b2f5aa273913f924e689086aaba4.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/b061702e0266e511ffe36f9dee962bb5a332b2f5aa273913f924e689086aaba4.jpg)

![ce7ef20de50d8e5265d5276a211b2d96efda920cee1f796b7698cdf0e6be9624.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/ce7ef20de50d8e5265d5276a211b2d96efda920cee1f796b7698cdf0e6be9624.jpg)

![d3fb1dd8e93d563e53b2644b7332ab7a725e5bc686d0b9b6c01ef2572ea9290b.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/d3fb1dd8e93d563e53b2644b7332ab7a725e5bc686d0b9b6c01ef2572ea9290b.jpg)

![dc24068919d9f62547a781bbcb775ddf03dccacd52f58aa365999595d190c28e.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/dc24068919d9f62547a781bbcb775ddf03dccacd52f58aa365999595d190c28e.jpg)

![f924814f2104b4db4e617aaddc5f3ef8582bf5feb80f254f4719ee6f9f5ac328.jpg](../icml_results/4207_MonoScale_%20Scaling%20Multi-Agent%20System%20with%20Monotonic%20Improvement/tables/f924814f2104b4db4e617aaddc5f3ef8582bf5feb80f254f4719ee6f9f5ac328.jpg)

<a id="paper-4166"></a>

## LiveFigure: Generating Editable Scientific Illustration with VLM Agents


### Images

![20777355f1bb8eaebcb563573f2b33f895a4c2d4154c1945ea46ff422f34e2cf.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/20777355f1bb8eaebcb563573f2b33f895a4c2d4154c1945ea46ff422f34e2cf.jpg)

![2380cde12e2256bb690ee78c04a35182a6d08714fd7c750ab548ca8ba098c518.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/2380cde12e2256bb690ee78c04a35182a6d08714fd7c750ab548ca8ba098c518.jpg)

![2aad6720ff043565272ebb225a014fc0393ed847295353bde77c3c7b2c4112dc.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/2aad6720ff043565272ebb225a014fc0393ed847295353bde77c3c7b2c4112dc.jpg)

![3fa179c69cd048cb58f5777b32aac2083aa6dc72d24b516e6bd962a5ced65484.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/3fa179c69cd048cb58f5777b32aac2083aa6dc72d24b516e6bd962a5ced65484.jpg)

![4af58f2773e8060b59f34e8b9e403fbeb477b9ffb2b55a289fe20b7ee11812a3.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/4af58f2773e8060b59f34e8b9e403fbeb477b9ffb2b55a289fe20b7ee11812a3.jpg)

![4c879c442237ac53fb4a4759000702e0aabaf190ee490c915f043cceaed640d3.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/4c879c442237ac53fb4a4759000702e0aabaf190ee490c915f043cceaed640d3.jpg)

![4d7627395b1f9a1c7775bcb15b9389941b5c028e69ecde9c2cf42110971d885e.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/4d7627395b1f9a1c7775bcb15b9389941b5c028e69ecde9c2cf42110971d885e.jpg)

![5b99097fc282c646d2f1719f2eb764a066b067efa99752008ab835f53137fd1f.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/5b99097fc282c646d2f1719f2eb764a066b067efa99752008ab835f53137fd1f.jpg)

![805b7755878ecd63b4e7a5a2741f746af60fff409c1ef791d3c0466512836942.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/805b7755878ecd63b4e7a5a2741f746af60fff409c1ef791d3c0466512836942.jpg)

![82abcdec2100d0182efec81c54f6b64274b1ba3cf0273b5c8cc508b089b1ce76.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/82abcdec2100d0182efec81c54f6b64274b1ba3cf0273b5c8cc508b089b1ce76.jpg)

![871bdb1d0aeb289ad0f1ccddfc97b007c6683d78de8cbe28a2c7fe4a25b4be1a.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/871bdb1d0aeb289ad0f1ccddfc97b007c6683d78de8cbe28a2c7fe4a25b4be1a.jpg)

![945b1dc7994ed22d10968bd70da6b643e46f200305a34d40ccff3f53165bbad4.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/945b1dc7994ed22d10968bd70da6b643e46f200305a34d40ccff3f53165bbad4.jpg)

![951de2b35cc72805e1fa4e19381f53a443b7453e4b40593474438232c2a80c73.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/951de2b35cc72805e1fa4e19381f53a443b7453e4b40593474438232c2a80c73.jpg)

![d4f3bef77639d9507ab31adaa63a482acc529c470583f387c6f92783f6e72407.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/images/d4f3bef77639d9507ab31adaa63a482acc529c470583f387c6f92783f6e72407.jpg)

### Tables

![042c1f1fab483a3bcdfe345b77a50a3b10afa85208ecc10caa8c651d78f6b4b9.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/tables/042c1f1fab483a3bcdfe345b77a50a3b10afa85208ecc10caa8c651d78f6b4b9.jpg)

![663820214e780351933566e59b18a9edff51f70347a6c08b242cc7b90c00f50a.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/tables/663820214e780351933566e59b18a9edff51f70347a6c08b242cc7b90c00f50a.jpg)

![93267c9e601f7d4247ae2a4aac023f26354feecf2284ff67c9404d6c9c9d19a3.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/tables/93267c9e601f7d4247ae2a4aac023f26354feecf2284ff67c9404d6c9c9d19a3.jpg)

![afe8a9d563f55899b5057421ecbef4a8e6a2a6347190d3294ef09e145de67709.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/tables/afe8a9d563f55899b5057421ecbef4a8e6a2a6347190d3294ef09e145de67709.jpg)

![d2f6ecf27123c64f1a995d89059fee0635e857fe45257ed59d60e92b47ca504c.jpg](../icml_results/4208_LiveFigure_%20Generating%20Editable%20Scientific%20Illustration%20with%20VLM%20Agents/tables/d2f6ecf27123c64f1a995d89059fee0635e857fe45257ed59d60e92b47ca504c.jpg)

<a id="paper-4167"></a>

## Block Rotation is All You Need for MXFP4 Quantization


### Images

![08a19d8641352cc2afea4cf0eafc61cf12f0b2667876a65a4e2e938ea5f33a09.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/08a19d8641352cc2afea4cf0eafc61cf12f0b2667876a65a4e2e938ea5f33a09.jpg)

![0cc0e21324fcdb4e3fe67be8142ab3146421c6ddb6b86f2b8c29f155c1f25861.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/0cc0e21324fcdb4e3fe67be8142ab3146421c6ddb6b86f2b8c29f155c1f25861.jpg)

![172bbbe148b289cd85bd448187cbbad62931dcc652db73b943b5042cf0ffd923.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/172bbbe148b289cd85bd448187cbbad62931dcc652db73b943b5042cf0ffd923.jpg)

![1b96c7e6ac22eda868426aba08bc10b448c0e84ee00b3a47cf521440aff86bf5.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/1b96c7e6ac22eda868426aba08bc10b448c0e84ee00b3a47cf521440aff86bf5.jpg)

![48fdad0bbd7a0bc817a52714e73998d34088e85f3a35f7eb231cee32b4150e6b.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/48fdad0bbd7a0bc817a52714e73998d34088e85f3a35f7eb231cee32b4150e6b.jpg)

![60268ae06c2536b408851008772c68a42078e5adfb34ea4ad2d26942627e5f3a.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/60268ae06c2536b408851008772c68a42078e5adfb34ea4ad2d26942627e5f3a.jpg)

![a863933661a465161eb6a9d532793479dd9f61356f31e0d2af856c5a1e807fd5.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/a863933661a465161eb6a9d532793479dd9f61356f31e0d2af856c5a1e807fd5.jpg)

![aca2210eae6fa86813cc28335803061b316756036be319f2d3f9a6e101539d6e.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/aca2210eae6fa86813cc28335803061b316756036be319f2d3f9a6e101539d6e.jpg)

![de347de5f36798efaa26beedb296c0568a82a909768a22cd0d19f78d405fdb6f.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/de347de5f36798efaa26beedb296c0568a82a909768a22cd0d19f78d405fdb6f.jpg)

![f4022bd95f530cc0f788318b52d37585c9dddb1722afb180c0d4c4c249519d69.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/images/f4022bd95f530cc0f788318b52d37585c9dddb1722afb180c0d4c4c249519d69.jpg)

### Tables

![0e44cce93326c766e0c8e486d0cf20ed1f8185b3d38c9a712016402fcd47f7bf.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/0e44cce93326c766e0c8e486d0cf20ed1f8185b3d38c9a712016402fcd47f7bf.jpg)

![0ed782bf2cde4dc8ca5209f8130ce4f65ed62f5471424c1a27db92e9fc3281f1.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/0ed782bf2cde4dc8ca5209f8130ce4f65ed62f5471424c1a27db92e9fc3281f1.jpg)

![14885e69c259a3b4cec7d98767ec1e201d541f3871b6a9372c874316a0a83d03.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/14885e69c259a3b4cec7d98767ec1e201d541f3871b6a9372c874316a0a83d03.jpg)

![207d54d266a3fe21f93a0adc4f414e0524fe7f84677f7839526c7b1a5aafbcd2.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/207d54d266a3fe21f93a0adc4f414e0524fe7f84677f7839526c7b1a5aafbcd2.jpg)

![402bea63445f644c4a0593373f6c283635c00a01ce923fca7179e7135e44b3c2.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/402bea63445f644c4a0593373f6c283635c00a01ce923fca7179e7135e44b3c2.jpg)

![445528e345c3c9dc0ccc20d46ff1cb2ccca3ce6808088dad23db501e5a7e227f.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/445528e345c3c9dc0ccc20d46ff1cb2ccca3ce6808088dad23db501e5a7e227f.jpg)

![44fa8847df05a8a8b9a4627e66248c7daa31e9c4538bf0bf80926dcd6a9d6546.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/44fa8847df05a8a8b9a4627e66248c7daa31e9c4538bf0bf80926dcd6a9d6546.jpg)

![4d97bac27e147c0a85c84586437e055a392ad906ddb5f50321939a7d39b891a7.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/4d97bac27e147c0a85c84586437e055a392ad906ddb5f50321939a7d39b891a7.jpg)

![5275fe62e26ff09297d9eb0000fa1fcfdd2e9fea7ed5a8944f6243d0281fa4b1.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/5275fe62e26ff09297d9eb0000fa1fcfdd2e9fea7ed5a8944f6243d0281fa4b1.jpg)

![69359fe77c3898589e3f31b8a8ac58fd8cb4e5048458ba0acb9f9825f932e1bf.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/69359fe77c3898589e3f31b8a8ac58fd8cb4e5048458ba0acb9f9825f932e1bf.jpg)

![6ff6f3a24e5a7911dbcf653894e3b373571cb07ec516e3edeb1f0575e4537912.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/6ff6f3a24e5a7911dbcf653894e3b373571cb07ec516e3edeb1f0575e4537912.jpg)

![72e0a419ee5ab442a0f33f1e3ab4da4a3f1c28dca24e40f24e21664f4cbf41a3.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/72e0a419ee5ab442a0f33f1e3ab4da4a3f1c28dca24e40f24e21664f4cbf41a3.jpg)

![79e3abfca9075eb5ca0c39651d8c7f326282e24ac4b276d2762a390dce42b34b.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/79e3abfca9075eb5ca0c39651d8c7f326282e24ac4b276d2762a390dce42b34b.jpg)

![87f75b89b4ac9fa29954f1ea02dc76ebeb27588532ae48363daeb249c60155eb.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/87f75b89b4ac9fa29954f1ea02dc76ebeb27588532ae48363daeb249c60155eb.jpg)

![90c187f68afe3ad1c58188b4451903a771d302096896246a0729acbc3f38dfcd.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/90c187f68afe3ad1c58188b4451903a771d302096896246a0729acbc3f38dfcd.jpg)

![9a88fa2df6c76432522e92e9bc3c1a7d90a68b88fc0f4ddc0b3ee67b1ec93560.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/9a88fa2df6c76432522e92e9bc3c1a7d90a68b88fc0f4ddc0b3ee67b1ec93560.jpg)

![aa9a6844a8afe7a2f36de632a796a98e7b792583d59edff2a7517f4ddceee524.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/aa9a6844a8afe7a2f36de632a796a98e7b792583d59edff2a7517f4ddceee524.jpg)

![b17def1cb0e8c1af10fedd31d33ca589be9d4d4f436941564d9aeaa9227007c7.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/b17def1cb0e8c1af10fedd31d33ca589be9d4d4f436941564d9aeaa9227007c7.jpg)

![b86c9d7808fe6d0da021a30cb3c10041bfb89c1ed5bc9c95ff27c962906edb68.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/b86c9d7808fe6d0da021a30cb3c10041bfb89c1ed5bc9c95ff27c962906edb68.jpg)

![b8743e809c40971cf687582fc5f31e2729d65f4f6c9c6220ee95b75b88426d37.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/b8743e809c40971cf687582fc5f31e2729d65f4f6c9c6220ee95b75b88426d37.jpg)

![e13ce51f477657f935bebb2e718b09cf20e98fd859e66d094501e461cf29fcca.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/e13ce51f477657f935bebb2e718b09cf20e98fd859e66d094501e461cf29fcca.jpg)

![f98a34c3380b629e071c4afd2c2724640a2fb83f2f2563767166533567c86937.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/f98a34c3380b629e071c4afd2c2724640a2fb83f2f2563767166533567c86937.jpg)

![f9f90db0cb2f1068b968f77df63cf2e7fcb096cc834ad11065e7cc7c9fcfa2b6.jpg](../icml_results/4209_Block%20Rotation%20is%20All%20You%20Need%20for%20MXFP4%20Quantization/tables/f9f90db0cb2f1068b968f77df63cf2e7fcb096cc834ad11065e7cc7c9fcfa2b6.jpg)

<a id="paper-4168"></a>

## SciNet: Evaluating AI Agents in Relation-Aware Scientific Literature Retrieval


### Images

![282dccbb8ed3cb87e0dd4f95f590d601b2ece33b6f69cbe97222977c80b8b3ce.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/282dccbb8ed3cb87e0dd4f95f590d601b2ece33b6f69cbe97222977c80b8b3ce.jpg)

![28f821ab801594a99c94a6010994947dca03275c574b78c86828a0d7262d7d95.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/28f821ab801594a99c94a6010994947dca03275c574b78c86828a0d7262d7d95.jpg)

![3272e31cf9b9db8e6909e96d6b2c00baf6c92443b4bfb4ab73911957a7943649.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/3272e31cf9b9db8e6909e96d6b2c00baf6c92443b4bfb4ab73911957a7943649.jpg)

![a0b121db2b9e514f4b09aa8e40cd055e061bdfa8077180205ff4c6f49f5ac8cb.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/a0b121db2b9e514f4b09aa8e40cd055e061bdfa8077180205ff4c6f49f5ac8cb.jpg)

![c6da4efa33d03705e2ce8459f2b4e322fd5a9f5a140aaac3737628e49f802a65.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/c6da4efa33d03705e2ce8459f2b4e322fd5a9f5a140aaac3737628e49f802a65.jpg)

![fb34384f11e47f11b5464d0d18d24af9778f7b766c3bc7eb2d9d667b813ab097.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/fb34384f11e47f11b5464d0d18d24af9778f7b766c3bc7eb2d9d667b813ab097.jpg)

![fe64348898c04a5c2a068c7cebda19908148c4547d26d9aa93338dbd03d6a338.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/images/fe64348898c04a5c2a068c7cebda19908148c4547d26d9aa93338dbd03d6a338.jpg)

### Tables

![189cb1204d78c901098ca9edbc1b5e54c05b85d8a88cf2edd82a1eba348dcc32.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/189cb1204d78c901098ca9edbc1b5e54c05b85d8a88cf2edd82a1eba348dcc32.jpg)

![2d7aa4e97d85c7ab83cf208bb51ea63e1fab71591f7522fa847f9db0b72db2e7.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/2d7aa4e97d85c7ab83cf208bb51ea63e1fab71591f7522fa847f9db0b72db2e7.jpg)

![55fc2d536bccb8de1a1f377c6e4760e78a921238fa9274b8ad834abe53f70fef.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/55fc2d536bccb8de1a1f377c6e4760e78a921238fa9274b8ad834abe53f70fef.jpg)

![6afe03b75cc8a4a1f6603b92e6d3cf61ca83a6c33ad9926407225c47a7e7e243.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/6afe03b75cc8a4a1f6603b92e6d3cf61ca83a6c33ad9926407225c47a7e7e243.jpg)

![73a0053f5ba42738b5a0ae682cd8b721202661a53258950e7d15b07a930f243e.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/73a0053f5ba42738b5a0ae682cd8b721202661a53258950e7d15b07a930f243e.jpg)

![8f34d363d3657f38243536a9df7c494b717e2a2a338f768036aa0545005ceea4.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/8f34d363d3657f38243536a9df7c494b717e2a2a338f768036aa0545005ceea4.jpg)

![db85152860db0ecc11c2a4c3bcdd710ce659ee2c4ca0f59c608d061a19da605a.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/db85152860db0ecc11c2a4c3bcdd710ce659ee2c4ca0f59c608d061a19da605a.jpg)

![e9b543cfa391c7a29b5896c784e18694298aa31d9ae90ef4eb138ea297272b87.jpg](../icml_results/4210_SciNet_%20Evaluating%20AI%20Agents%20in%20Relation-Aware%20Scientific%20Literature%20Retrieval/tables/e9b543cfa391c7a29b5896c784e18694298aa31d9ae90ef4eb138ea297272b87.jpg)

<a id="paper-4169"></a>

## Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks


### Images

![1574ee4615d9c3689dd79f903521de65d8be37afd07b271204c436b5846049ba.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/images/1574ee4615d9c3689dd79f903521de65d8be37afd07b271204c436b5846049ba.jpg)

![3824748c87c28e788105a6aaf4cf1b490932c1419de532fc709534b68a544b32.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/images/3824748c87c28e788105a6aaf4cf1b490932c1419de532fc709534b68a544b32.jpg)

![75ed640a465fda737fc5723d82a11c7c4c0a55c993d3e693fa652d5a62567843.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/images/75ed640a465fda737fc5723d82a11c7c4c0a55c993d3e693fa652d5a62567843.jpg)

![b15ea9cd3108d180dc09b7cabf6e3f62350ca25f52ea5fd52a87e08bff350343.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/images/b15ea9cd3108d180dc09b7cabf6e3f62350ca25f52ea5fd52a87e08bff350343.jpg)

### Tables

![59728404c1a8fc86a91d84e1879c3bd325181185ab12c524cb507fc89fc0ab3a.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/tables/59728404c1a8fc86a91d84e1879c3bd325181185ab12c524cb507fc89fc0ab3a.jpg)

![8e5e8b426811defc267c881e03828059679bceb9f1d5edacbb9f50b3d00fe3c7.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/tables/8e5e8b426811defc267c881e03828059679bceb9f1d5edacbb9f50b3d00fe3c7.jpg)

![dff246f514659c774ebd87244166c7683a9758b82a857c5b51c6755e38004317.jpg](../icml_results/4211_Lifting%20Traces%20to%20Logic_%20Programmatic%20Skill%20Induction%20with%20Neuro-Symbolic%20Learning%20for%20Long-Horizon%20/tables/dff246f514659c774ebd87244166c7683a9758b82a857c5b51c6755e38004317.jpg)

<a id="paper-4170"></a>

## LIVEditor-14B: Lightning Unified Video Editing via In-Context Sparse Attention


### Images

![18da5252e583e36570a3330930fb41020ad598005baf0b8cbb38941753576046.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/18da5252e583e36570a3330930fb41020ad598005baf0b8cbb38941753576046.jpg)

![4bbaf177a62be0edded0bda331e597f8cf9fe45c007ce13b4d9a93677060917e.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/4bbaf177a62be0edded0bda331e597f8cf9fe45c007ce13b4d9a93677060917e.jpg)

![4ee623b19e9906e52547a68d9ca2086dd928c56278f29a3b65c03d04a432b9e7.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/4ee623b19e9906e52547a68d9ca2086dd928c56278f29a3b65c03d04a432b9e7.jpg)

![5b32b77b0f0a08ca57d669453e01c3392c70daa5e12d7d02d6d0325f5c90cf90.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/5b32b77b0f0a08ca57d669453e01c3392c70daa5e12d7d02d6d0325f5c90cf90.jpg)

![6d77b6e333e711e31637023926583ed1c036ed054f613c0f274b2c1dcc853ef5.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/6d77b6e333e711e31637023926583ed1c036ed054f613c0f274b2c1dcc853ef5.jpg)

![793bec01ae2ab2ac88fe784619d1b3f0601b6ae8cb8a5d97b7d5950a8ca865d4.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/793bec01ae2ab2ac88fe784619d1b3f0601b6ae8cb8a5d97b7d5950a8ca865d4.jpg)

![7abfa8fd9eae3cecb26edca57cdfa191d8e464ad7ea1857603db9cf91eea9e3a.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/7abfa8fd9eae3cecb26edca57cdfa191d8e464ad7ea1857603db9cf91eea9e3a.jpg)

![80fa6535e779bae308bef0cef8a710df0d247800b0d64e1ea49d7232dc082d12.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/80fa6535e779bae308bef0cef8a710df0d247800b0d64e1ea49d7232dc082d12.jpg)

![854b103dc95fefe82e40713d8736bcc70969b9315fb87ad8dbd13517670ce990.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/854b103dc95fefe82e40713d8736bcc70969b9315fb87ad8dbd13517670ce990.jpg)

![a0783d40667539ee3cc3a6e8b5ce851dd422bc627d9b312a2be1eeecde71f93b.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/a0783d40667539ee3cc3a6e8b5ce851dd422bc627d9b312a2be1eeecde71f93b.jpg)

![a90fea7ad658855aa4709e11dfe4ff27a8bf4a99062f7a2933757bc84bad5fae.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/a90fea7ad658855aa4709e11dfe4ff27a8bf4a99062f7a2933757bc84bad5fae.jpg)

![ba2c4c8885c67a2829069a6ca566e506f498fd2218bbab97c1367230e632f393.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/ba2c4c8885c67a2829069a6ca566e506f498fd2218bbab97c1367230e632f393.jpg)

![d63efc5e8d26555755876975528ec020c3d7d25a19d676a1d85aea475406d8f7.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/d63efc5e8d26555755876975528ec020c3d7d25a19d676a1d85aea475406d8f7.jpg)

![da8364a78aa36b64926617f472bdf43c315d485a4ccae0240fee4f9208fefa08.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/da8364a78aa36b64926617f472bdf43c315d485a4ccae0240fee4f9208fefa08.jpg)

![e083ca681e8e86443d05181f35dadfadc17383e78a24873e14d69dd43335f977.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/e083ca681e8e86443d05181f35dadfadc17383e78a24873e14d69dd43335f977.jpg)

![e807a342ad2d1ecdc82371fef6f70075f48c7a4153b62beab13c9391d2d712c2.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/e807a342ad2d1ecdc82371fef6f70075f48c7a4153b62beab13c9391d2d712c2.jpg)

![f0637f816f78e4f04a0b12eed185993861b73e927e2008a9a0ef2ecd03fae5a9.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/images/f0637f816f78e4f04a0b12eed185993861b73e927e2008a9a0ef2ecd03fae5a9.jpg)

### Tables

![0923d0cdcf1596227154e025488cba7ba500c9a69cdfd6066223a12ca6930599.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/0923d0cdcf1596227154e025488cba7ba500c9a69cdfd6066223a12ca6930599.jpg)

![1d8560922e4293eef148cfcf9de87a87f8e2a684fb40884b5aa327fb1ba9eddd.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/1d8560922e4293eef148cfcf9de87a87f8e2a684fb40884b5aa327fb1ba9eddd.jpg)

![2130f9ea1e4000634bd2574d58addcc2ca17eccd0ab9e1d035ce4e57283efcd3.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/2130f9ea1e4000634bd2574d58addcc2ca17eccd0ab9e1d035ce4e57283efcd3.jpg)

![6258c220917091f229754463bbcd7654f7f4f225462881cf10aa4a866dfe76a2.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/6258c220917091f229754463bbcd7654f7f4f225462881cf10aa4a866dfe76a2.jpg)

![8a45df7dae669bdd2559c206f1e467c3c11aed598e77a2f2d001f47d9ad07070.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/8a45df7dae669bdd2559c206f1e467c3c11aed598e77a2f2d001f47d9ad07070.jpg)

![9249db882f4764390c298a1d9a6c51c877e114869d4d3f3dc4597df0e54ed573.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/9249db882f4764390c298a1d9a6c51c877e114869d4d3f3dc4597df0e54ed573.jpg)

![9f3b21fe6ccb9bf874e8c7ba81957f926c47d00bb969266f9897fb328d4a151c.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/9f3b21fe6ccb9bf874e8c7ba81957f926c47d00bb969266f9897fb328d4a151c.jpg)

![a5b965f276da3a5f47138073751513571e646f4e356c88074dbb209306d87787.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/a5b965f276da3a5f47138073751513571e646f4e356c88074dbb209306d87787.jpg)

![b5d9fd0765317f49e38ac8c214f53a345bd5e706170286033ae989849fd73459.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/b5d9fd0765317f49e38ac8c214f53a345bd5e706170286033ae989849fd73459.jpg)

![c2a994544b11c17f8e96f6999a8e9e62d2cdd636238ff2915bca92ad81c5ef5a.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/c2a994544b11c17f8e96f6999a8e9e62d2cdd636238ff2915bca92ad81c5ef5a.jpg)

![ccff1d972b7b4e12bd57ed4d8e73b8849ab3b0d84fe3df14ba28a42a81ee9889.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/ccff1d972b7b4e12bd57ed4d8e73b8849ab3b0d84fe3df14ba28a42a81ee9889.jpg)

![cd0c6c7d6b6adcde7c2614b185b535bb82aa11d5dbce84158c12b5e09a6a6181.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/cd0c6c7d6b6adcde7c2614b185b535bb82aa11d5dbce84158c12b5e09a6a6181.jpg)

![d5f6ebd7118543b8ddf185b0bbe74fd670649def5c499d0f7261083c8198b540.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/d5f6ebd7118543b8ddf185b0bbe74fd670649def5c499d0f7261083c8198b540.jpg)

![e05d09fa3ec322d7d5dae999c7734b9ed176fa4401dcc049919fcbe5e8bcec0e.jpg](../icml_results/4212_LIVEditor-14B_%20Lightning%20Unified%20Video%20Editing%20via%20In-Context%20Sparse%20Attention/tables/e05d09fa3ec322d7d5dae999c7734b9ed176fa4401dcc049919fcbe5e8bcec0e.jpg)

<a id="paper-4171"></a>

## CONGA:Confidence-and-Gradient-Aware Learning Rate Schedule for Test Time Adaptation


### Images

![1d60a56e845af1eb18b9d3d7e26cb2d27496e6f14eebd283d6e9699f964fd341.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/1d60a56e845af1eb18b9d3d7e26cb2d27496e6f14eebd283d6e9699f964fd341.jpg)

![28cc9498f9d926af594eb9fcd417e943edb289a91cbab074e9cf701e21a2644f.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/28cc9498f9d926af594eb9fcd417e943edb289a91cbab074e9cf701e21a2644f.jpg)

![330a47156e4ce4e4ee5f41cb152700c21efce9c1c80326a8206e2c0d8da9639f.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/330a47156e4ce4e4ee5f41cb152700c21efce9c1c80326a8206e2c0d8da9639f.jpg)

![42e1df6946a73aaf372b5146c974dc3c7523441cffff38093f668058ebc7c156.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/42e1df6946a73aaf372b5146c974dc3c7523441cffff38093f668058ebc7c156.jpg)

![57f9ad9a5e2090f8215d2a0ce15594cf71d589c3ca307b515aa124495fdf7a03.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/57f9ad9a5e2090f8215d2a0ce15594cf71d589c3ca307b515aa124495fdf7a03.jpg)

![a766c901bcd4a8799227d06c55c51def191460adcb97e71f408102df50616a18.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/images/a766c901bcd4a8799227d06c55c51def191460adcb97e71f408102df50616a18.jpg)

### Tables

![1ee4d21f641e92dc3452480c824af5a6b4f6de62c7266624143b923d6d8f4e2f.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/1ee4d21f641e92dc3452480c824af5a6b4f6de62c7266624143b923d6d8f4e2f.jpg)

![229307bae912c492f9cd6377100eee87a401b4bedee15a028009a6f91146f153.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/229307bae912c492f9cd6377100eee87a401b4bedee15a028009a6f91146f153.jpg)

![2da8e85ca30355edca356170ea71649d5f5af5d2524b81b7ba2064b6e2ccc4fc.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/2da8e85ca30355edca356170ea71649d5f5af5d2524b81b7ba2064b6e2ccc4fc.jpg)

![4707f9543a40132ecccc22968937149275cd1eaa9acd61b40f54bbef304b5608.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/4707f9543a40132ecccc22968937149275cd1eaa9acd61b40f54bbef304b5608.jpg)

![47ccaaae7dc94b46303801838ed048f0bafbdd28e9e557eefb61b0b4b8063249.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/47ccaaae7dc94b46303801838ed048f0bafbdd28e9e557eefb61b0b4b8063249.jpg)

![4df7baf95929c0001b06ebb02f38c3ca7313477145efd339d8901bf61c08086a.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/4df7baf95929c0001b06ebb02f38c3ca7313477145efd339d8901bf61c08086a.jpg)

![5cb9b16a661bc574ee4846575c0e8e339f1eb4e38b7064e8c981ed9828ddee89.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/5cb9b16a661bc574ee4846575c0e8e339f1eb4e38b7064e8c981ed9828ddee89.jpg)

![7457bc04b2bb19e46c38fdf9e96ae85f5897a4ac5153f523804abc86e5b059e3.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/7457bc04b2bb19e46c38fdf9e96ae85f5897a4ac5153f523804abc86e5b059e3.jpg)

![7f24b482f0d161acbb8e0d2f45f1b04ddd5e5e899420987117144da2ea9796a6.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/7f24b482f0d161acbb8e0d2f45f1b04ddd5e5e899420987117144da2ea9796a6.jpg)

![875c77691b0b0fd84080f30f569289af5b81e621b62534cb9d9eb367c359d299.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/875c77691b0b0fd84080f30f569289af5b81e621b62534cb9d9eb367c359d299.jpg)

![8cfaf7cdf895a2462e8ddb9de6565fdc936dcfdc69c2ea93fbd8d861d0433de7.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/8cfaf7cdf895a2462e8ddb9de6565fdc936dcfdc69c2ea93fbd8d861d0433de7.jpg)

![a021d26ddae112cb0d4de4930fa34d9c060cce3c898f5725eeb2718bc33f8e92.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/a021d26ddae112cb0d4de4930fa34d9c060cce3c898f5725eeb2718bc33f8e92.jpg)

![a60e01f700b7d297b4f39c54e1497a65263fbe1db019780febeb2a49f5994234.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/a60e01f700b7d297b4f39c54e1497a65263fbe1db019780febeb2a49f5994234.jpg)

![b6d529bf416501ae64c71b28dbd235b1d67a799d67c042ebb4b2f418e4a61b8b.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/b6d529bf416501ae64c71b28dbd235b1d67a799d67c042ebb4b2f418e4a61b8b.jpg)

![cc395befe1f995d1259f3d0436f0e6d6ca04ca3e577eb6126895c55e04e5f57f.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/cc395befe1f995d1259f3d0436f0e6d6ca04ca3e577eb6126895c55e04e5f57f.jpg)

![d817e1474747420bf9e92fab3a902d0fb9fc8b49b09c017dce00a498ac8c156a.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/d817e1474747420bf9e92fab3a902d0fb9fc8b49b09c017dce00a498ac8c156a.jpg)

![f12e721722f410e7251f38b733dfe2aad92e9610060467a480cc744944608dd0.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/f12e721722f410e7251f38b733dfe2aad92e9610060467a480cc744944608dd0.jpg)

![f8d1662adf8d82adf67a8746edca0098f37b11e504c6db7a97931fa6c509e6c6.jpg](../icml_results/4213_CONGA_Confidence-and-Gradient-Aware%20Learning%20Rate%20Schedule%20for%20Test%20Time%20Adaptation/tables/f8d1662adf8d82adf67a8746edca0098f37b11e504c6db7a97931fa6c509e6c6.jpg)

<a id="paper-4172"></a>

## How RLHF Amplifies Sycophancy


### Images

![349fd63595c78a542798a8ab6dd32cd59cb0243edfbefc6e937dfbaaa101fbdd.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/images/349fd63595c78a542798a8ab6dd32cd59cb0243edfbefc6e937dfbaaa101fbdd.jpg)

![3a5065b28e101ba8240df43e5b1e3a038f81ef9e017f0492136a41f716da342a.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/images/3a5065b28e101ba8240df43e5b1e3a038f81ef9e017f0492136a41f716da342a.jpg)

![d83389a8efaf33a5a9940e44893c17de615e3f54acea7507d29fad97c0047b70.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/images/d83389a8efaf33a5a9940e44893c17de615e3f54acea7507d29fad97c0047b70.jpg)

![f39b8908e93f247a4b7552b93ad8ea5718e9d9d182eb4ddd377fc684c6f36cb9.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/images/f39b8908e93f247a4b7552b93ad8ea5718e9d9d182eb4ddd377fc684c6f36cb9.jpg)

![f7529f44dd2c8517921b361ad103d146790f8bf7484b28a60ccc76c63ff6b490.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/images/f7529f44dd2c8517921b361ad103d146790f8bf7484b28a60ccc76c63ff6b490.jpg)

### Tables

![76354490314efb5e2c76640fe8f5c97a3c571fd5e80c0ce463bbe2d1823f8869.jpg](../icml_results/4214_How%20RLHF%20Amplifies%20Sycophancy/tables/76354490314efb5e2c76640fe8f5c97a3c571fd5e80c0ce463bbe2d1823f8869.jpg)

<a id="paper-4173"></a>

## Intentional Updates for Streaming Reinforcement Learning


### Images

![08fd546932ff858b2a71fa4565c59a30a1a0b26626a3aa764f06e9c5a5a22ef1.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/08fd546932ff858b2a71fa4565c59a30a1a0b26626a3aa764f06e9c5a5a22ef1.jpg)

![0f66ebfe1e008b4f136c901a1019cf7e00ba77cfcca676dbc93e34d84d10fe91.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/0f66ebfe1e008b4f136c901a1019cf7e00ba77cfcca676dbc93e34d84d10fe91.jpg)

![467a397c1ead8bf3a5e431699e35f6865dace495534a74ebb5ab541b319d0f85.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/467a397c1ead8bf3a5e431699e35f6865dace495534a74ebb5ab541b319d0f85.jpg)

![c2595e58443a3b757ba0e17014299afb60e5772ae343620474d3c4c992c875aa.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/c2595e58443a3b757ba0e17014299afb60e5772ae343620474d3c4c992c875aa.jpg)

![ca3011a37766937a3843f76c6b6085326b7e2b545e388f956a19a22d7e0d090b.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/ca3011a37766937a3843f76c6b6085326b7e2b545e388f956a19a22d7e0d090b.jpg)

![d187458a139a5a08448a4220d176bf99c3a7e406dd1510c283e74e5dc53f3ab9.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/d187458a139a5a08448a4220d176bf99c3a7e406dd1510c283e74e5dc53f3ab9.jpg)

![d35a144707e52ebe58ebaab28a46a6063bbfb3d77e502018b2224f924aa7af0d.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/d35a144707e52ebe58ebaab28a46a6063bbfb3d77e502018b2224f924aa7af0d.jpg)

![f7a77a042317f267bbcbcfdd14f899de3f8e2c5f3483be37aae1e198949c5e4e.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/images/f7a77a042317f267bbcbcfdd14f899de3f8e2c5f3483be37aae1e198949c5e4e.jpg)

### Tables

![1618856186cd6c636fdf6c02053d8ca0fd9223aea46556b1e4e0e5e55c8a5ebd.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/1618856186cd6c636fdf6c02053d8ca0fd9223aea46556b1e4e0e5e55c8a5ebd.jpg)

![33e3910e27c04cf8e895288124ab5b36c9306dcfd5dbe79475c06a77fa71e6f6.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/33e3910e27c04cf8e895288124ab5b36c9306dcfd5dbe79475c06a77fa71e6f6.jpg)

![65f7bb3ff36cb1ac259cdc40b3c9673176e1353f750ec4da0ceb35516d1f55c5.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/65f7bb3ff36cb1ac259cdc40b3c9673176e1353f750ec4da0ceb35516d1f55c5.jpg)

![920e501b2ad0ba5f48e268f4c23dc7ecb74c99b134e1c62fc71c1ebe06778c52.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/920e501b2ad0ba5f48e268f4c23dc7ecb74c99b134e1c62fc71c1ebe06778c52.jpg)

![9e9c8807ae626c15ce673710a715c752d510a9c2bf587e7ee94b48d508cd3a9e.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/9e9c8807ae626c15ce673710a715c752d510a9c2bf587e7ee94b48d508cd3a9e.jpg)

![a13c6a99643524db8bc4d1a4126bbe1408b2baa186f7086b22ce6a861e8fc07d.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/a13c6a99643524db8bc4d1a4126bbe1408b2baa186f7086b22ce6a861e8fc07d.jpg)

![ebe8041112be4716f3f9cd247409a63ae5f9a60854a977560b30966df41dae87.jpg](../icml_results/4215_Intentional%20Updates%20for%20Streaming%20Reinforcement%20Learning/tables/ebe8041112be4716f3f9cd247409a63ae5f9a60854a977560b30966df41dae87.jpg)

<a id="paper-4174"></a>

## Who’s in Charge? Disempowerment Patterns in Real-World LLM Usage


### Images

![1531c580fca1964cc7519d122b8ccd196f17918910e2754c6befe2a31137492f.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/1531c580fca1964cc7519d122b8ccd196f17918910e2754c6befe2a31137492f.jpg)

![307bf28afb73c841f5cda69d45ec563615838da08696fd36b37cacdb686132e7.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/307bf28afb73c841f5cda69d45ec563615838da08696fd36b37cacdb686132e7.jpg)

![4464ca295b76a971c339fee2e73053977c9d0050bf75783d039f11ebb2aba9de.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/4464ca295b76a971c339fee2e73053977c9d0050bf75783d039f11ebb2aba9de.jpg)

![57b6bd9597f14cd3e569df51d28d2c4b61c07dd0f52e9acca1a9e7c95d386f88.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/57b6bd9597f14cd3e569df51d28d2c4b61c07dd0f52e9acca1a9e7c95d386f88.jpg)

![580a81cbc8305559c5327448c1b5446c0f7dac007fb4b2ffba721fc7bb933b99.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/580a81cbc8305559c5327448c1b5446c0f7dac007fb4b2ffba721fc7bb933b99.jpg)

![62ea4997087c3c90323af66632334a1923d27bbaf9578bcb59f9df4a69335b40.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/62ea4997087c3c90323af66632334a1923d27bbaf9578bcb59f9df4a69335b40.jpg)

![766de64a431a1c428f4bd66d5bf7171c0920d68cd2899d556975a5afc8b6ac9d.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/766de64a431a1c428f4bd66d5bf7171c0920d68cd2899d556975a5afc8b6ac9d.jpg)

![790575dc208d558b5c89cad4e44f6d206a69d1f7d6700b2375595d4f914d9666.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/790575dc208d558b5c89cad4e44f6d206a69d1f7d6700b2375595d4f914d9666.jpg)

![80d9edc7fdbd997d0a1f9da9a8108b5a8bc352db3b35723d8c1b17666433fd41.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/80d9edc7fdbd997d0a1f9da9a8108b5a8bc352db3b35723d8c1b17666433fd41.jpg)

![955bfef87e81db57d0f3d379cd5674eaebfdbdd277703a44da7677715dc34db3.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/955bfef87e81db57d0f3d379cd5674eaebfdbdd277703a44da7677715dc34db3.jpg)

![982691ffd6bf63b61cda8503fc2af3d776a2307f59ed2f4d13c2e3b5d09725ba.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/982691ffd6bf63b61cda8503fc2af3d776a2307f59ed2f4d13c2e3b5d09725ba.jpg)

![b2881d458ace7c25702b49438160c3d21c3d8f563cea82ef8476d388fbdb5e84.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/b2881d458ace7c25702b49438160c3d21c3d8f563cea82ef8476d388fbdb5e84.jpg)

![b92b67d1c5e1880878b8718a18cb3e6f5d0f62b5f3b099dba859cd5a23c65faf.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/b92b67d1c5e1880878b8718a18cb3e6f5d0f62b5f3b099dba859cd5a23c65faf.jpg)

![b93187c2523902c545b1ca820a6eb97fae787055b461ef712b466c63425f9ef1.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/b93187c2523902c545b1ca820a6eb97fae787055b461ef712b466c63425f9ef1.jpg)

![b9d4773b7a49d359a8431a08c64cd1ea942c5e140e11293418e6a8d2191f9433.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/b9d4773b7a49d359a8431a08c64cd1ea942c5e140e11293418e6a8d2191f9433.jpg)

![cc7332b9c3c3197b5abd22edd552e689d581c25a240bf989c4ff45333bb1d2f4.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/cc7332b9c3c3197b5abd22edd552e689d581c25a240bf989c4ff45333bb1d2f4.jpg)

![cdf6b05045adee9060f1351c4c951aeed8206644c487b8738b5e4ed4b4f1108a.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/cdf6b05045adee9060f1351c4c951aeed8206644c487b8738b5e4ed4b4f1108a.jpg)

![d6f94ea93462bcc50033e1b52818e181eb501120bf5c31b5b1c33328cb23831d.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/d6f94ea93462bcc50033e1b52818e181eb501120bf5c31b5b1c33328cb23831d.jpg)

![df4298e36104c5b3b750600f01bab3504b0284c82c852631eb9b501564da327e.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/df4298e36104c5b3b750600f01bab3504b0284c82c852631eb9b501564da327e.jpg)

![f7ec55e0deb297bc5663e18312bcc98d3f4fc9cd1b82b7db6543cf975d495643.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/f7ec55e0deb297bc5663e18312bcc98d3f4fc9cd1b82b7db6543cf975d495643.jpg)

![fd43deb8035bcdd1be5dd986e7ab758817833ec5b834eb96ce5b0ee979174108.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/fd43deb8035bcdd1be5dd986e7ab758817833ec5b834eb96ce5b0ee979174108.jpg)

![fe92fd660da43e760277bbb4eb027d8c550226935c76ee280b4af35ee0e77999.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/images/fe92fd660da43e760277bbb4eb027d8c550226935c76ee280b4af35ee0e77999.jpg)

### Tables

![49eb860db7ddb07fde1859527982b17a9211624f8d7410d95acb354bccf4ee6e.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/49eb860db7ddb07fde1859527982b17a9211624f8d7410d95acb354bccf4ee6e.jpg)

![6882cf654bf38cb17866829107dfb766387a1d312e457cfefe95a5f75b840640.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/6882cf654bf38cb17866829107dfb766387a1d312e457cfefe95a5f75b840640.jpg)

![68fec89ad7800954584270d3c36f9f7d4820a753e0798c643ba5148a52c6a82a.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/68fec89ad7800954584270d3c36f9f7d4820a753e0798c643ba5148a52c6a82a.jpg)

![69952c494cf0a2a05fee5f26b2b60ad3fbc798d45b148b74fcc0766387c6f890.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/69952c494cf0a2a05fee5f26b2b60ad3fbc798d45b148b74fcc0766387c6f890.jpg)

![713aef023946bde872017e47bb787874efa15a05838bc31f45f0076e9a6a383d.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/713aef023946bde872017e47bb787874efa15a05838bc31f45f0076e9a6a383d.jpg)

![72a70bf28b3944213b4a81d70bddae7e768282a0f9e39b7d30d0c1d28919db3c.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/72a70bf28b3944213b4a81d70bddae7e768282a0f9e39b7d30d0c1d28919db3c.jpg)

![9c45a4110b5ed6e06f67ee4066a87c6da23da2a0ad1586c8478d1eafbd7c72a1.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/9c45a4110b5ed6e06f67ee4066a87c6da23da2a0ad1586c8478d1eafbd7c72a1.jpg)

![be7b82e95138cc20de058f00c9e457c07230b2322fc5f269fa7959e512da86c7.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/be7b82e95138cc20de058f00c9e457c07230b2322fc5f269fa7959e512da86c7.jpg)

![c54cf9ceee7b859949b596a941c25bcab642cf231bca7bba6dc71ba31d7edf12.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/c54cf9ceee7b859949b596a941c25bcab642cf231bca7bba6dc71ba31d7edf12.jpg)

![c5917ae26d0bafb5a93bdfdb92595a99b39b31da2faeb46c9ce5bd34c6c91a93.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/c5917ae26d0bafb5a93bdfdb92595a99b39b31da2faeb46c9ce5bd34c6c91a93.jpg)

![c9df75194a556d9288d892988197e68a5084fa8aa52215f579bfcf0b83ec2907.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/c9df75194a556d9288d892988197e68a5084fa8aa52215f579bfcf0b83ec2907.jpg)

![d006d512e43344faf22eda63d39bb0dd9797db5957cfd38ff90da02b8ed9ab0f.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/d006d512e43344faf22eda63d39bb0dd9797db5957cfd38ff90da02b8ed9ab0f.jpg)

![d43bb15df17361b8a54ef7246c82969a902772f14a15320d180d0f7259a7febd.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/d43bb15df17361b8a54ef7246c82969a902772f14a15320d180d0f7259a7febd.jpg)

![def1ee2cc68f8f4532758d1ff8d688ca3fb8cf9b49cc71ba925b200909c26420.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/def1ee2cc68f8f4532758d1ff8d688ca3fb8cf9b49cc71ba925b200909c26420.jpg)

![ea243f48deee77ae9f8d364f3b3599239d251e5b8564a4de26dae7fe94b2c3f4.jpg](../icml_results/4216_Who%E2%80%99s%20in%20Charge_%20Disempowerment%20Patterns%20in%20Real-World%20LLM%20Usage/tables/ea243f48deee77ae9f8d364f3b3599239d251e5b8564a4de26dae7fe94b2c3f4.jpg)

<a id="paper-4175"></a>

## RubricRobustness: Evaluating the Sensitivity of Rubrics-Based Benchmarks to Simple Perturbations


### Images

![0569ac48456abd05b2b0c8d042d6d6c41d30ada13c20963f1b3223b203b37535.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/images/0569ac48456abd05b2b0c8d042d6d6c41d30ada13c20963f1b3223b203b37535.jpg)

![72000788bf8f356b9a13bb7f106f8fb5ae4b9976edbe0b2b7f44342f8254dab6.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/images/72000788bf8f356b9a13bb7f106f8fb5ae4b9976edbe0b2b7f44342f8254dab6.jpg)

![9e5dd0b2c7b00f16e0a508f285e829b5ed9ebe4d02e7334778fa111e665d13e0.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/images/9e5dd0b2c7b00f16e0a508f285e829b5ed9ebe4d02e7334778fa111e665d13e0.jpg)

### Tables

![7578f4c8a9d435e74ca30856adfcb4aa6f02a1ec2c66e12aa96451c1d4720d2f.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/tables/7578f4c8a9d435e74ca30856adfcb4aa6f02a1ec2c66e12aa96451c1d4720d2f.jpg)

![7c273a5ec0e88d41e3e5d132221aae6907db8b407c519680619579db6ac2a7da.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/tables/7c273a5ec0e88d41e3e5d132221aae6907db8b407c519680619579db6ac2a7da.jpg)

![bd67798fb4f438b1ebe80552f079b944fd78dab8f3f75d53829d4583035748c6.jpg](../icml_results/4217_RubricRobustness_%20Evaluating%20the%20Sensitivity%20of%20Rubrics-Based%20Benchmarks%20to%20Simple%20Perturbations/tables/bd67798fb4f438b1ebe80552f079b944fd78dab8f3f75d53829d4583035748c6.jpg)

<a id="paper-4176"></a>

## The Realignment Problem: When Right becomes Wrong in LLMs


### Images

![35e013b2a1f80f5327f3f642cf57e20b751441b1cfe36c8a194f9f88fbfbe69a.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/images/35e013b2a1f80f5327f3f642cf57e20b751441b1cfe36c8a194f9f88fbfbe69a.jpg)

![be9bb373aa40a49d4477869606af35595e0a54d3255c233471f2a4604ca09819.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/images/be9bb373aa40a49d4477869606af35595e0a54d3255c233471f2a4604ca09819.jpg)

![e4bc0cf5263b5237d1299073417e2b01d174dd82f9e2c1e1b5d9528b49a3bcbe.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/images/e4bc0cf5263b5237d1299073417e2b01d174dd82f9e2c1e1b5d9528b49a3bcbe.jpg)

### Tables

![0c46dcf7917d45f5b29144f3dd0d3c10567dabacb27d3d4fa7c77f51aa1d98f0.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/0c46dcf7917d45f5b29144f3dd0d3c10567dabacb27d3d4fa7c77f51aa1d98f0.jpg)

![0fb659fc3c90e98b1297590cb3eb487dea4ae7d5a0d805ee7eed90b3158d68f0.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/0fb659fc3c90e98b1297590cb3eb487dea4ae7d5a0d805ee7eed90b3158d68f0.jpg)

![1732c5940343121fe9c428c3d5ec007f5c87b537b81036351c1c5ca3063a4b0a.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/1732c5940343121fe9c428c3d5ec007f5c87b537b81036351c1c5ca3063a4b0a.jpg)

![2d0f3af6ff3707e456b7f0adb80947399698d3e9d119bb00c109fdf73fa9a69b.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/2d0f3af6ff3707e456b7f0adb80947399698d3e9d119bb00c109fdf73fa9a69b.jpg)

![2e7d710e13a353c05edf79d9bc8fd57a1422e8eb6210f955d5d046adde4b3d34.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/2e7d710e13a353c05edf79d9bc8fd57a1422e8eb6210f955d5d046adde4b3d34.jpg)

![557849edc49d090af924cb87d806b088626c0a3bf405615cfd0483cf6e3c5f56.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/557849edc49d090af924cb87d806b088626c0a3bf405615cfd0483cf6e3c5f56.jpg)

![77f407b45369918329c6154d7af153827f9be4787f1f6ef4e967004da849a08a.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/77f407b45369918329c6154d7af153827f9be4787f1f6ef4e967004da849a08a.jpg)

![ae60d0efe19a984252e1e2729df94fdfeb056d90744b986b94ec88e390381473.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/ae60d0efe19a984252e1e2729df94fdfeb056d90744b986b94ec88e390381473.jpg)

![b8aa92051bd0605d474004418f36b965b65f06f594cc3a26c883c344ead03756.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/b8aa92051bd0605d474004418f36b965b65f06f594cc3a26c883c344ead03756.jpg)

![bfff393fdd8ad31a70ffbd4f2ec0d014eeb8b3dbc64188dadf441089293b7c48.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/bfff393fdd8ad31a70ffbd4f2ec0d014eeb8b3dbc64188dadf441089293b7c48.jpg)

![c762d7a97173d61bf9d4aeba64d65b9c03289510dc86d17d6d4a2daddd36e405.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/c762d7a97173d61bf9d4aeba64d65b9c03289510dc86d17d6d4a2daddd36e405.jpg)

![cfeacc9c400f9755c31a2e0f5a9ced019dd490bcc07996b1b3c00b99d01d0781.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/cfeacc9c400f9755c31a2e0f5a9ced019dd490bcc07996b1b3c00b99d01d0781.jpg)

![d54eb156fca56c9c27fbefa28449097d7030231cb591f5b269d3c5d45e9411b7.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/d54eb156fca56c9c27fbefa28449097d7030231cb591f5b269d3c5d45e9411b7.jpg)

![e0d7459573e327e7fac59a51d95fef7fdc10cfe0725839529c5a93c13bacc9b9.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/e0d7459573e327e7fac59a51d95fef7fdc10cfe0725839529c5a93c13bacc9b9.jpg)

![e45cac1157849eca2e1c556174707e1e8c76eb08762f84fabe0143ac2c9b61a2.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/e45cac1157849eca2e1c556174707e1e8c76eb08762f84fabe0143ac2c9b61a2.jpg)

![f075a0c1e697c3a41365f90459f3355d46d3d6ae9abcf883149e8a3975f945ce.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/f075a0c1e697c3a41365f90459f3355d46d3d6ae9abcf883149e8a3975f945ce.jpg)

![fe2a1c271fcaa3af447c1f9d517713c15f6786ff3627d6105ac3350bc6e06f82.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/fe2a1c271fcaa3af447c1f9d517713c15f6786ff3627d6105ac3350bc6e06f82.jpg)

![feb47d1576781a731887b1a8c28bddf2f90ba8c4de6ef641ebd40e9e27769f27.jpg](../icml_results/4218_The%20Realignment%20Problem_%20When%20Right%20becomes%20Wrong%20in%20LLMs/tables/feb47d1576781a731887b1a8c28bddf2f90ba8c4de6ef641ebd40e9e27769f27.jpg)

<a id="paper-4177"></a>

## Ariadne’s Thread of LipSync: Unraveling Forgeries via Inconsistency between Lip Motions and Head Poses


### Images

![118a9ac95b09833bf95afc09911561b331f78b480b705174c1c44433b68ebad7.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/118a9ac95b09833bf95afc09911561b331f78b480b705174c1c44433b68ebad7.jpg)

![222b4f973b978e5386a81fb3932b1d2ee817508ab5d1ca2268ca153caa8307f0.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/222b4f973b978e5386a81fb3932b1d2ee817508ab5d1ca2268ca153caa8307f0.jpg)

![2d1009fc7c70e9180fad7904557e238c242d59f4b645a8bf2d019a40d5b3974d.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/2d1009fc7c70e9180fad7904557e238c242d59f4b645a8bf2d019a40d5b3974d.jpg)

![38f500677c9cdbf9dae65e28d97d61cbf4826b945c94c5ab333d0ee671e43bba.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/38f500677c9cdbf9dae65e28d97d61cbf4826b945c94c5ab333d0ee671e43bba.jpg)

![4ccc5b834615e7643f343eca7eec0ecbf68a636bf78cffce2b27ec71c07d84b5.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/4ccc5b834615e7643f343eca7eec0ecbf68a636bf78cffce2b27ec71c07d84b5.jpg)

![552a72044331eae7bc36aaa2f90217ebb37173cfcd262f6cd50fc891f2065e04.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/552a72044331eae7bc36aaa2f90217ebb37173cfcd262f6cd50fc891f2065e04.jpg)

![639553cc38e57b45fe69e73b8801430ea2253b9aeac5003a260c9e54ea5170ef.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/639553cc38e57b45fe69e73b8801430ea2253b9aeac5003a260c9e54ea5170ef.jpg)

![79b4099c00355af8c95663b9499d0257c7562b6b63e41cc1459e1837f7ed93ab.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/79b4099c00355af8c95663b9499d0257c7562b6b63e41cc1459e1837f7ed93ab.jpg)

![7d9e71f2072f78578f391b6271811039bf7d7e686a4f1954a023849ff32cc090.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/7d9e71f2072f78578f391b6271811039bf7d7e686a4f1954a023849ff32cc090.jpg)

![8bbed290c5f96bc6effbcc32602a1d71169503c2c40d3a727430c998318ad8f0.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/8bbed290c5f96bc6effbcc32602a1d71169503c2c40d3a727430c998318ad8f0.jpg)

![b2f981c9a3b440867666d2596dd9542a0899fbab5a12e61b2197507cac3fedf5.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/b2f981c9a3b440867666d2596dd9542a0899fbab5a12e61b2197507cac3fedf5.jpg)

![b387a45c8373262dfef46ca67e8951fbd9e761733cd0325e7473ccc4d5573ccf.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/b387a45c8373262dfef46ca67e8951fbd9e761733cd0325e7473ccc4d5573ccf.jpg)

![c6e61ff77372a5a91a1cc36db6719efba54dff3ee1c30383cbe391fac08cb1fa.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/c6e61ff77372a5a91a1cc36db6719efba54dff3ee1c30383cbe391fac08cb1fa.jpg)

![c738d083fc517d566e6ad49a8558c0703fa8d801db6fc07d510524858fe218e2.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/c738d083fc517d566e6ad49a8558c0703fa8d801db6fc07d510524858fe218e2.jpg)

![fe8a27957bff308c0f0764be75f628f27d53202f97afdb50a65e9eb513f8b4f9.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/images/fe8a27957bff308c0f0764be75f628f27d53202f97afdb50a65e9eb513f8b4f9.jpg)

### Tables

![05388114176203c6a92bc05c0a20cf9cbdf415f55a1c425799eb2a3e362abfec.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/05388114176203c6a92bc05c0a20cf9cbdf415f55a1c425799eb2a3e362abfec.jpg)

![146997880bbdfbb355d5adb623a60719b935a7e0db1eed5fa61a10749a2020a0.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/146997880bbdfbb355d5adb623a60719b935a7e0db1eed5fa61a10749a2020a0.jpg)

![1bd048791cd33f6d5fd73a10b426896d2fae6ef38e1bb61c5d3e9d68098ff2af.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/1bd048791cd33f6d5fd73a10b426896d2fae6ef38e1bb61c5d3e9d68098ff2af.jpg)

![23a07dea52356f02ce65c55de3cac643734a651e7fbdca821e9600d67bd85f76.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/23a07dea52356f02ce65c55de3cac643734a651e7fbdca821e9600d67bd85f76.jpg)

![2fdc8a21bd3ce6580d000a2a4b34517614cfcbb6ea804ed6a541c86dec6bc463.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/2fdc8a21bd3ce6580d000a2a4b34517614cfcbb6ea804ed6a541c86dec6bc463.jpg)

![37c2278f431947f868483614c07641fc186a72f8581a142055b0b19cac29a96a.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/37c2278f431947f868483614c07641fc186a72f8581a142055b0b19cac29a96a.jpg)

![385be4c3577362f684c97f5272aed895432330f23ad455b5f454c45ad35f7dd3.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/385be4c3577362f684c97f5272aed895432330f23ad455b5f454c45ad35f7dd3.jpg)

![491ba4ac64cd5b79e04feb98a8d674a234cb78d00208f07c62d465dc97fafb04.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/491ba4ac64cd5b79e04feb98a8d674a234cb78d00208f07c62d465dc97fafb04.jpg)

![61fdeabf027e86717610d25631709e86a5d47ce45ddc797ee534ac13affcbbff.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/61fdeabf027e86717610d25631709e86a5d47ce45ddc797ee534ac13affcbbff.jpg)

![79de78563dd52e3d0fe18863ecc6add4a8da27a132d62632027e3a16b3a3c340.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/79de78563dd52e3d0fe18863ecc6add4a8da27a132d62632027e3a16b3a3c340.jpg)

![8351665d5ea7a5eadaa0a3180535f7f0618b7eec903cba90fc07fe1a3ab1942a.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/8351665d5ea7a5eadaa0a3180535f7f0618b7eec903cba90fc07fe1a3ab1942a.jpg)

![ba5f6747f7ca655145a8e5415b43e3ec3527fcdd050a033b5c9596bb96aae9d7.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/ba5f6747f7ca655145a8e5415b43e3ec3527fcdd050a033b5c9596bb96aae9d7.jpg)

![cce4b170a162b23ee84a8e19aa2e6eb09b23fa999750c7cd765251605d90307b.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/cce4b170a162b23ee84a8e19aa2e6eb09b23fa999750c7cd765251605d90307b.jpg)

![d2ce8844fade02f80c5eb846f6621f0293a8bc63bf351e4280fcf8d4c0b54732.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/d2ce8844fade02f80c5eb846f6621f0293a8bc63bf351e4280fcf8d4c0b54732.jpg)

![e3f80d0876e7f6d202df18b0a88ea5f065127d122f24b349cc05788b75916dd1.jpg](../icml_results/4219_Ariadne%E2%80%99s%20Thread%20of%20LipSync_%20Unraveling%20Forgeries%20via%20Inconsistency%20between%20Lip%20Motions%20and%20Head%20Pos/tables/e3f80d0876e7f6d202df18b0a88ea5f065127d122f24b349cc05788b75916dd1.jpg)

<a id="paper-4178"></a>

## Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices


### Images

![09a9b2fdca51dcdb30b5162109fc375252c6af6340d7eec10b34046c75c6a5d7.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/09a9b2fdca51dcdb30b5162109fc375252c6af6340d7eec10b34046c75c6a5d7.jpg)

![287a51c2aa9eb6fec8af42428104cb0aa7f40719c1c2c1f97624fe8af1a38ded.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/287a51c2aa9eb6fec8af42428104cb0aa7f40719c1c2c1f97624fe8af1a38ded.jpg)

![35688acea904ef2ecb313ca9e22a57c1f355984f23f05f8ab1c0f7b55d285162.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/35688acea904ef2ecb313ca9e22a57c1f355984f23f05f8ab1c0f7b55d285162.jpg)

![458bc9bd657ab5bcdeba7b93e2bce0300d4a39b37df545c94e23032b43550bd2.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/458bc9bd657ab5bcdeba7b93e2bce0300d4a39b37df545c94e23032b43550bd2.jpg)

![4fc79823c1abaa6f48c9e39420c330aa8a8ed39285a329d5c047272a2697f37d.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/4fc79823c1abaa6f48c9e39420c330aa8a8ed39285a329d5c047272a2697f37d.jpg)

![73effd2dd0c7819d2ff1e14a4fb484a2af0f57b182df82c5377871f608b3b6fa.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/73effd2dd0c7819d2ff1e14a4fb484a2af0f57b182df82c5377871f608b3b6fa.jpg)

![86670152c69ac5829160f1e684a6813e6114183da77b5842cab3a1c8890b829d.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/86670152c69ac5829160f1e684a6813e6114183da77b5842cab3a1c8890b829d.jpg)

![d987f897bab2358023d6be9ea76374b0506ab6ce45a83fba190e253f7e65a5a6.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/images/d987f897bab2358023d6be9ea76374b0506ab6ce45a83fba190e253f7e65a5a6.jpg)

### Tables

![0550a7cf4b4428a91c968087f48c4d28ce5437e8b38b590a2d8f7467be5691ed.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/tables/0550a7cf4b4428a91c968087f48c4d28ce5437e8b38b590a2d8f7467be5691ed.jpg)

![4df08eceea21551b4f8ba04bf4731cbc297f51946d0e29059cec0d262003c576.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/tables/4df08eceea21551b4f8ba04bf4731cbc297f51946d0e29059cec0d262003c576.jpg)

![c844117e3985ae217e87aa7346cf0130d131dc3096bc10d4163fe8dcaf2e03c0.jpg](../icml_results/4220_Spik4lite_%20Refactoring%20Neuromorphic%20Sparsity%20for%20Efficient%20Spiking%20Neural%20Networks%20on%20Commodity%20Edge/tables/c844117e3985ae217e87aa7346cf0130d131dc3096bc10d4163fe8dcaf2e03c0.jpg)

<a id="paper-4179"></a>

## Laplacian Representations for Decision-Time Planning


### Images

![07e3f117b5aa310d5c45e1917a6efbc7f9ad9a4d9caa81fdc6f7abb52a47a3b2.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/07e3f117b5aa310d5c45e1917a6efbc7f9ad9a4d9caa81fdc6f7abb52a47a3b2.jpg)

![235ca3a1477e9096ab78fca710ddbc70b93b18f25f03596f663ba15dcc7a9868.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/235ca3a1477e9096ab78fca710ddbc70b93b18f25f03596f663ba15dcc7a9868.jpg)

![2774a7968084a3b60c5fb84b35590c3ab8e9c9fb77d603a3b64cd3d0db957110.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/2774a7968084a3b60c5fb84b35590c3ab8e9c9fb77d603a3b64cd3d0db957110.jpg)

![3319051db71eb2ffca2d30abcacca46501c132bcb45a64c60c775947397a02c3.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/3319051db71eb2ffca2d30abcacca46501c132bcb45a64c60c775947397a02c3.jpg)

![37f255fdc05cc3b0abd59f5a96b4a8837845ea80db320c7c569e2befe2892224.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/37f255fdc05cc3b0abd59f5a96b4a8837845ea80db320c7c569e2befe2892224.jpg)

![535d5d40f983e52654bc00ec69655c4506b0a7065d5b68159e8feae9d0330d29.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/535d5d40f983e52654bc00ec69655c4506b0a7065d5b68159e8feae9d0330d29.jpg)

![6ecd1ffefd2c05f8bb89fd603462322acee7df75744dbc4a8b9bb5d982039835.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/6ecd1ffefd2c05f8bb89fd603462322acee7df75744dbc4a8b9bb5d982039835.jpg)

![b754623bc1dfdb62c8ccca718781b7689ef687c8d1c07c12df8d2625256eaef4.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/b754623bc1dfdb62c8ccca718781b7689ef687c8d1c07c12df8d2625256eaef4.jpg)

![fe2c119ab8d04b27edc26816d5412468bc66750453ea86890668df596ee41fd7.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/images/fe2c119ab8d04b27edc26816d5412468bc66750453ea86890668df596ee41fd7.jpg)

### Tables

![04242f46ee21447a5ab1159d40edc0ecf52ea877e200fa868c42948f964935eb.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/04242f46ee21447a5ab1159d40edc0ecf52ea877e200fa868c42948f964935eb.jpg)

![3b5b6f6ad53d40a6565b07dff4e5fa38daa2d66bd3ae6efdc02ca138e1499dbc.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/3b5b6f6ad53d40a6565b07dff4e5fa38daa2d66bd3ae6efdc02ca138e1499dbc.jpg)

![4b8c2bb78a6ff8e5f5722605f5ea4b94e2b41ec12c9f842cb2715ad99e595a09.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/4b8c2bb78a6ff8e5f5722605f5ea4b94e2b41ec12c9f842cb2715ad99e595a09.jpg)

![5e9c959932d92f621a158599123aa01cd671299abb5362fef24c8b557e5d2407.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/5e9c959932d92f621a158599123aa01cd671299abb5362fef24c8b557e5d2407.jpg)

![63f63d82a6710daa849d3ca51d12648f5bb5095e544a1579f78838cfd533ae6c.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/63f63d82a6710daa849d3ca51d12648f5bb5095e544a1579f78838cfd533ae6c.jpg)

![68702e5991970ea9a84246bd12b942e0c2723f4de8afc75cad1eb2014a1fb5fc.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/68702e5991970ea9a84246bd12b942e0c2723f4de8afc75cad1eb2014a1fb5fc.jpg)

![7072cb8af8b97d23a295b24e62049700251df71fa35d812ccb73f84f84c0280c.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/7072cb8af8b97d23a295b24e62049700251df71fa35d812ccb73f84f84c0280c.jpg)

![7637e4df116ad92f171ca06e1339e2440951cc40ea6fc3c64e3c30d40d317ef3.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/7637e4df116ad92f171ca06e1339e2440951cc40ea6fc3c64e3c30d40d317ef3.jpg)

![7fae1d762a0a33df62cc97944fb6092109d6c641377f49ce12cd6f5bd5d27f5f.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/7fae1d762a0a33df62cc97944fb6092109d6c641377f49ce12cd6f5bd5d27f5f.jpg)

![8b4f396936d6d91a020980d0030d0b6aad4f103f66794eb711ca2d97977378c9.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/8b4f396936d6d91a020980d0030d0b6aad4f103f66794eb711ca2d97977378c9.jpg)

![a9dbbdfb3d70dfa67eeb9f423cc1f786eb78520358f518e8bcbb029f61c1b8f1.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/a9dbbdfb3d70dfa67eeb9f423cc1f786eb78520358f518e8bcbb029f61c1b8f1.jpg)

![ae07f29bb3d7867bf6abd7e15797452137e1c529fd203e864c53ebfc9aa3e140.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/ae07f29bb3d7867bf6abd7e15797452137e1c529fd203e864c53ebfc9aa3e140.jpg)

![b62402c6fb1bc8572f13b2d9e9dc47d8e9004d16856f07c769a70ceff1bd759c.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/b62402c6fb1bc8572f13b2d9e9dc47d8e9004d16856f07c769a70ceff1bd759c.jpg)

![be29039c2c385c2f1d32367fe1218773e72ede2d2334863ed1b1e7f02ea63675.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/be29039c2c385c2f1d32367fe1218773e72ede2d2334863ed1b1e7f02ea63675.jpg)

![c7acf9564e49551cce0b7e66163e83cc5af8c0f2aeeb8c514170a3c0129ca423.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/c7acf9564e49551cce0b7e66163e83cc5af8c0f2aeeb8c514170a3c0129ca423.jpg)

![cf00678b0dc54ead409289bc040d8235ba2ceea3e3a76f494fba6c6a4011d7c3.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/cf00678b0dc54ead409289bc040d8235ba2ceea3e3a76f494fba6c6a4011d7c3.jpg)

![d9fdd3cfe293c49b9c9cbfe8ff3dd3d43e5e3e1dd2537cf41426263459367e7a.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/d9fdd3cfe293c49b9c9cbfe8ff3dd3d43e5e3e1dd2537cf41426263459367e7a.jpg)

![e5e2e957b9a3a6028be5a658ac5de47f94e06342c886da0ed5e93af67228bf61.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/e5e2e957b9a3a6028be5a658ac5de47f94e06342c886da0ed5e93af67228bf61.jpg)

![e7ae99eb7829497c00e0293393b86252211fb7c8719da37ba41ec5e906da1db8.jpg](../icml_results/4221_Laplacian%20Representations%20for%20Decision-Time%20Planning/tables/e7ae99eb7829497c00e0293393b86252211fb7c8719da37ba41ec5e906da1db8.jpg)

<a id="paper-4180"></a>

## MMClima: A Framework for Multimodal Climate Science Data and Evaluation


### Images

![0b84cda6d352bcb5aee2b63d53cc07ba5942c0914155fa2b91e9ba672df33c9b.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/0b84cda6d352bcb5aee2b63d53cc07ba5942c0914155fa2b91e9ba672df33c9b.jpg)

![28bede13c4a51d0f4c8f1e2c7c39d0e91ed0c02d33681553a97e5e0e77cba058.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/28bede13c4a51d0f4c8f1e2c7c39d0e91ed0c02d33681553a97e5e0e77cba058.jpg)

![32cd882b4cd3472184f860ac8c33af8f3fe1838a4d70ffd870a136203e70cca9.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/32cd882b4cd3472184f860ac8c33af8f3fe1838a4d70ffd870a136203e70cca9.jpg)

![6e5b69e6cdd96d1833578d384c2e68fe84cce3271a1f0866a027deef4f8a5a68.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/6e5b69e6cdd96d1833578d384c2e68fe84cce3271a1f0866a027deef4f8a5a68.jpg)

![8f97d1e4358ab4c21fc79c3ac7bfc4e3d459007887bee53f3bcba21d886b9bce.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/8f97d1e4358ab4c21fc79c3ac7bfc4e3d459007887bee53f3bcba21d886b9bce.jpg)

![d16393a0185a207e364e3dcf614203af7c51cc82fd0a8b81e3ff9a69641557cb.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/d16393a0185a207e364e3dcf614203af7c51cc82fd0a8b81e3ff9a69641557cb.jpg)

![d186a92dcb1cbc2b2be54a54d2cb468f0307d08227f2cc59f041f8cc81a8714f.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/d186a92dcb1cbc2b2be54a54d2cb468f0307d08227f2cc59f041f8cc81a8714f.jpg)

![d9dba150a165f923926ade765ecae9285947cc186d3c564287a77e53c119fc03.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/d9dba150a165f923926ade765ecae9285947cc186d3c564287a77e53c119fc03.jpg)

![e2689a39dd3f58e604db42b62f582a743bf01d5daeeaba85548b9573f48f9e6b.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/e2689a39dd3f58e604db42b62f582a743bf01d5daeeaba85548b9573f48f9e6b.jpg)

![f2a03f462c561d7efde8ceaea194528c3e68163006d1b8f9fae37dab558178e4.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/f2a03f462c561d7efde8ceaea194528c3e68163006d1b8f9fae37dab558178e4.jpg)

![f3166c6b3e237bce3e7b29ba355f654eae1f1762843be0f391c44114d05fbcac.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/images/f3166c6b3e237bce3e7b29ba355f654eae1f1762843be0f391c44114d05fbcac.jpg)

### Tables

![06da35f84af7a7a03090443585db10c452d6ec709e9465cffe363627bafabd65.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/06da35f84af7a7a03090443585db10c452d6ec709e9465cffe363627bafabd65.jpg)

![129556cf7b76ac0695bcab0cd5b3e1790da632533536f37247c1585e4fef87e1.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/129556cf7b76ac0695bcab0cd5b3e1790da632533536f37247c1585e4fef87e1.jpg)

![2743f97bccd89258b5eba3c372c856725dcf1c8ef2438cf4e9b9d6e6c2a109ed.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/2743f97bccd89258b5eba3c372c856725dcf1c8ef2438cf4e9b9d6e6c2a109ed.jpg)

![4d9f78d10c3d54e52aa8f440bb5573443c6a1a7198d2e98d6d70c95a3e78c757.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/4d9f78d10c3d54e52aa8f440bb5573443c6a1a7198d2e98d6d70c95a3e78c757.jpg)

![4e5633bc938933f0a35462c548c2b66a50b411537502720df38430883f9f0af4.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/4e5633bc938933f0a35462c548c2b66a50b411537502720df38430883f9f0af4.jpg)

![4e84dd250e36ea4ccb2e7d4a0a9f5a652294faeae42fe7d24348a7f4ca1175c9.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/4e84dd250e36ea4ccb2e7d4a0a9f5a652294faeae42fe7d24348a7f4ca1175c9.jpg)

![717d06654de77a5e381eb49b2d8d1b449d23604c4f23c85d171d20748c675ca6.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/717d06654de77a5e381eb49b2d8d1b449d23604c4f23c85d171d20748c675ca6.jpg)

![98afc7d313091d4892ddafdcc3c3fba11fd27f38fb245c00d6e7143ab7448078.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/98afc7d313091d4892ddafdcc3c3fba11fd27f38fb245c00d6e7143ab7448078.jpg)

![af3c0e19bffc64ceabebedffe3a05b4ef5799fa98430a06edcf6fc26c223fdc7.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/af3c0e19bffc64ceabebedffe3a05b4ef5799fa98430a06edcf6fc26c223fdc7.jpg)

![cdb897b03742a035e97c60e540af31afba7a5d0955af33b6dd7856e59b52e73b.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/cdb897b03742a035e97c60e540af31afba7a5d0955af33b6dd7856e59b52e73b.jpg)

![d82d80e3aaae3c79916bca5fc4c3fddcb1e2b51c55cc40d938e6f131a1a41cd3.jpg](../icml_results/4222_MMClima_%20A%20Framework%20for%20Multimodal%20Climate%20Science%20Data%20and%20Evaluation/tables/d82d80e3aaae3c79916bca5fc4c3fddcb1e2b51c55cc40d938e6f131a1a41cd3.jpg)

<a id="paper-4181"></a>

## DF-LoGiT: Data-Free Logic-Gated Backdoor Attacks in Vision Transformers


### Images

![0d76e2a9dad4f0613f8ef671127ff38fb4ca0e39bd1de7cf6e5074de30a1a4cc.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/0d76e2a9dad4f0613f8ef671127ff38fb4ca0e39bd1de7cf6e5074de30a1a4cc.jpg)

![150fb1c3501cb1eaa1310c14d34d23372877d7a27123aa5d40144279f9d95b81.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/150fb1c3501cb1eaa1310c14d34d23372877d7a27123aa5d40144279f9d95b81.jpg)

![20dc4ea252bc6d091eb9956e004a349d4b99ea56ae250244d45fb93d9867b5c5.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/20dc4ea252bc6d091eb9956e004a349d4b99ea56ae250244d45fb93d9867b5c5.jpg)

![2d2810a1c8a4b7fcf72d94e72621921c278c43d40528b536c5d6554dc2335bf5.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/2d2810a1c8a4b7fcf72d94e72621921c278c43d40528b536c5d6554dc2335bf5.jpg)

![4e918fdb18e1cb16f50a193f21c71d5e07ec8eb5f1ace2a48917828f38ba87af.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/4e918fdb18e1cb16f50a193f21c71d5e07ec8eb5f1ace2a48917828f38ba87af.jpg)

![57537aefc48fe28003bfe40da25cad431917fe9744bc47244f77c4e61814ff5e.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/57537aefc48fe28003bfe40da25cad431917fe9744bc47244f77c4e61814ff5e.jpg)

![7386a777fd3f80d780338f3dfac82578f8ab7527b4a1b934e7bc45b91432aaf1.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/7386a777fd3f80d780338f3dfac82578f8ab7527b4a1b934e7bc45b91432aaf1.jpg)

![b03dd0029d4fb479426de63db99959dc1daec4acdca673220ecf590c08b0f36d.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/b03dd0029d4fb479426de63db99959dc1daec4acdca673220ecf590c08b0f36d.jpg)

![c7f7f331630fd3daaf23a62135d7a83da333eb425d0d19113377b37af914fd4e.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/c7f7f331630fd3daaf23a62135d7a83da333eb425d0d19113377b37af914fd4e.jpg)

![fe2c63488f0d1e364762045481603a70f6a7cfe0a23db8027525c49be7929457.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/images/fe2c63488f0d1e364762045481603a70f6a7cfe0a23db8027525c49be7929457.jpg)

### Tables

![2f9b648ab8d307d0ff379f3ff6b6077aaf2bd543ca7e9eaa5a0be246d5875981.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/2f9b648ab8d307d0ff379f3ff6b6077aaf2bd543ca7e9eaa5a0be246d5875981.jpg)

![602e48a0fc392925f1ef7bf972b842860dd9f81345a468bb1c8a4f0da2cc188d.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/602e48a0fc392925f1ef7bf972b842860dd9f81345a468bb1c8a4f0da2cc188d.jpg)

![ab2172ea12d37d29a632001cceb32786e477d956130dec59c337a1dc6a1aec66.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/ab2172ea12d37d29a632001cceb32786e477d956130dec59c337a1dc6a1aec66.jpg)

![b59450c650f474f9e8939f13d29559548d57595359bbb748accfac06e847245d.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/b59450c650f474f9e8939f13d29559548d57595359bbb748accfac06e847245d.jpg)

![eba892fdcc3c540587a47c2ac3b75651d295a91206ac59ea90a37f0b2b9d189b.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/eba892fdcc3c540587a47c2ac3b75651d295a91206ac59ea90a37f0b2b9d189b.jpg)

![ed31d2cbc967c804197eadcf937d0299a952e3cd95325c1a86611b2cdb055b29.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/ed31d2cbc967c804197eadcf937d0299a952e3cd95325c1a86611b2cdb055b29.jpg)

![f1129946c57ff7108d3d7859f2e54ec565d5f51337ea6ef82dcdad95b1b8ed70.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/f1129946c57ff7108d3d7859f2e54ec565d5f51337ea6ef82dcdad95b1b8ed70.jpg)

![f3b1e795a0fff0e34784a184d23d10f8f3995a012d791a1f57b65882407b3e98.jpg](../icml_results/4223_DF-LoGiT_%20Data-Free%20Logic-Gated%20Backdoor%20Attacks%20in%20Vision%20Transformers/tables/f3b1e795a0fff0e34784a184d23d10f8f3995a012d791a1f57b65882407b3e98.jpg)

<a id="paper-4182"></a>

## CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks


### Images

![a47e4dd3437868e75013b6e0face2eb37c3e16e92a8acb64755520b8f4cf3ae5.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/images/a47e4dd3437868e75013b6e0face2eb37c3e16e92a8acb64755520b8f4cf3ae5.jpg)

### Tables

![4267e4907a6c015a0b46d7495b9cb55d8b2b49df75ac47d823d41152c152ec67.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/4267e4907a6c015a0b46d7495b9cb55d8b2b49df75ac47d823d41152c152ec67.jpg)

![4f481ccfafb5629e026ebd562bcd541dca04dd3bec1144043f788b73a712d73c.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/4f481ccfafb5629e026ebd562bcd541dca04dd3bec1144043f788b73a712d73c.jpg)

![6fe77eedb976f8fda10a4830cd3e799caea6c8143be1a1c429b4728c5593a274.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/6fe77eedb976f8fda10a4830cd3e799caea6c8143be1a1c429b4728c5593a274.jpg)

![77bbc47a42e5d2da4c1a7b01c922fa89930fac9836f25788e2aeb554e5d7d51d.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/77bbc47a42e5d2da4c1a7b01c922fa89930fac9836f25788e2aeb554e5d7d51d.jpg)

![9311261faf1d1abc3f107b413f765cff6a65b421179705df8a47686600f33bb4.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/9311261faf1d1abc3f107b413f765cff6a65b421179705df8a47686600f33bb4.jpg)

![a3ce6b49d6839c0414556303e608c9cba3513016e62f79a56c4d0960203700b6.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/a3ce6b49d6839c0414556303e608c9cba3513016e62f79a56c4d0960203700b6.jpg)

![c5fdac9be81b9bc15962b0250ab20fdde1679ebc5fa28dfa871c38f32aae9add.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/c5fdac9be81b9bc15962b0250ab20fdde1679ebc5fa28dfa871c38f32aae9add.jpg)

![cca5aba8cf3a55a195166eb1c2c7f2811b9f9bfb1cf056b238b9281685a549e7.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/cca5aba8cf3a55a195166eb1c2c7f2811b9f9bfb1cf056b238b9281685a549e7.jpg)

![cf6efe91f4036bf411af341a98179ff545ee937ed7c4771b086ad3b20cb8dc64.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/cf6efe91f4036bf411af341a98179ff545ee937ed7c4771b086ad3b20cb8dc64.jpg)

![edce5cc450d20169b7d48c6b6e1bd96160f3c85ed00f7bf83649cc1314d1794b.jpg](../icml_results/4224_CREDIT_%20Certified%20Ownership%20Verification%20of%20Deep%20Neural%20Networks%20Against%20Model%20Extraction%20Attacks/tables/edce5cc450d20169b7d48c6b6e1bd96160f3c85ed00f7bf83649cc1314d1794b.jpg)

<a id="paper-4183"></a>

## IDRBench: Understanding the Capability of Large Language Models on Interdisciplinary Research


### Images

![0841fc50867b168473682927b5e01438dc70ad890fac97195495c3a7195b4656.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/0841fc50867b168473682927b5e01438dc70ad890fac97195495c3a7195b4656.jpg)

![366660fa7f29c21c5fd00ea9e5b69f42f3ace21d8b327bc6ee9ba69630fc2698.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/366660fa7f29c21c5fd00ea9e5b69f42f3ace21d8b327bc6ee9ba69630fc2698.jpg)

![59152ce917885e88bcdc652458ac09ff7c05b3f35e06c59fd7e7e938dbfcc03b.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/59152ce917885e88bcdc652458ac09ff7c05b3f35e06c59fd7e7e938dbfcc03b.jpg)

![98a167b39a8b0ebfab529d9f81ef23e594c01d6de9415b6d1aae0f984b95f020.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/98a167b39a8b0ebfab529d9f81ef23e594c01d6de9415b6d1aae0f984b95f020.jpg)

![a410209350651d49152b3a69b2b166d21763ddc6cb8b8a2c88d5af934b152d58.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/a410209350651d49152b3a69b2b166d21763ddc6cb8b8a2c88d5af934b152d58.jpg)

![ac12f20a3a5b297834c6daacc9819809885fabb29278daffabfae0b9a2878e02.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/ac12f20a3a5b297834c6daacc9819809885fabb29278daffabfae0b9a2878e02.jpg)

![ad5800ead3df0ed177b20c9d4f24751e4d2b037ea55bffa86754c64a0839ca48.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/ad5800ead3df0ed177b20c9d4f24751e4d2b037ea55bffa86754c64a0839ca48.jpg)

![b449e91320a9259fecc5edbc6202f39d1ecf8c85b7a4cda60a983fb8a1899b99.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/b449e91320a9259fecc5edbc6202f39d1ecf8c85b7a4cda60a983fb8a1899b99.jpg)

![d4193c1804f9140a76d3769ad6be17609c17b8e2226909260eedf699dccb962a.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/d4193c1804f9140a76d3769ad6be17609c17b8e2226909260eedf699dccb962a.jpg)

![d87c4487101ce8c726a4105b0658ba68f467f8f7a52926c9e24318cae76d48e2.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/d87c4487101ce8c726a4105b0658ba68f467f8f7a52926c9e24318cae76d48e2.jpg)

![e7f1a85bbbcfcf787ec3e81f5a4f92f0a349d0030ac7677064e6ca24f252b3ec.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/e7f1a85bbbcfcf787ec3e81f5a4f92f0a349d0030ac7677064e6ca24f252b3ec.jpg)

![fda869524fd5d7db0ed9b66a05f64a430e0c323d864925b9170b7b8906cfb574.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/fda869524fd5d7db0ed9b66a05f64a430e0c323d864925b9170b7b8906cfb574.jpg)

![ff24f3dc0be7dc6709f2b676c5bf065a9c2e3960540d758f1d9b62e7132326b9.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/images/ff24f3dc0be7dc6709f2b676c5bf065a9c2e3960540d758f1d9b62e7132326b9.jpg)

### Tables

![487b034a6087791749cd38398f9c702bb2bb2c4cc08aaa12378fca10672aea8e.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/487b034a6087791749cd38398f9c702bb2bb2c4cc08aaa12378fca10672aea8e.jpg)

![5ac5b256f721986b45866285ad290d0010346db0c81584e772dfe263118cfa32.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/5ac5b256f721986b45866285ad290d0010346db0c81584e772dfe263118cfa32.jpg)

![5f1f2fe02f20d0d0b0e78910526324b012ffc55a44979fac68b5705733382ffa.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/5f1f2fe02f20d0d0b0e78910526324b012ffc55a44979fac68b5705733382ffa.jpg)

![7d81e02c6df910d539b92c8d7fb4cbb4a50e00a6093589934a48aab101cd3484.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/7d81e02c6df910d539b92c8d7fb4cbb4a50e00a6093589934a48aab101cd3484.jpg)

![8d8496328d766d9ade63ad474658b9f533aaa9b62b583cebf2c58f86327cce01.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/8d8496328d766d9ade63ad474658b9f533aaa9b62b583cebf2c58f86327cce01.jpg)

![a899af966c13e9574130dd57aada10214cba67dadf198a455815d923ed93dd6b.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/a899af966c13e9574130dd57aada10214cba67dadf198a455815d923ed93dd6b.jpg)

![b9409e6fcf0d4cfa16eb99768743019643eb5e42a7c9389bdc64824072ef2020.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/b9409e6fcf0d4cfa16eb99768743019643eb5e42a7c9389bdc64824072ef2020.jpg)

![d20b2241c435988c64dc29912356862c312544d5b5619b0571cc66da0c2dc697.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/d20b2241c435988c64dc29912356862c312544d5b5619b0571cc66da0c2dc697.jpg)

![f19832bd2cb291fab858fb743088549ca920ba9dfecd1ecfe545a6c9fa264c2d.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/f19832bd2cb291fab858fb743088549ca920ba9dfecd1ecfe545a6c9fa264c2d.jpg)

![fd2a82c17535d59cfabf3e415198a8382705f70598f3ac655bc9caa32c1afda0.jpg](../icml_results/4225_IDRBench_%20Understanding%20the%20Capability%20of%20Large%20Language%20Models%20on%20Interdisciplinary%20Research/tables/fd2a82c17535d59cfabf3e415198a8382705f70598f3ac655bc9caa32c1afda0.jpg)

<a id="paper-4184"></a>

## Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation


### Images

![037b8aca9e747fdc94520b49811bcb384336c5a9aad6a566d81575bd4c37fa9e.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/037b8aca9e747fdc94520b49811bcb384336c5a9aad6a566d81575bd4c37fa9e.jpg)

![1902d2344164aef5a85303cf79a3f8850bb9aabe8083b56b3f145eb8a25da484.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/1902d2344164aef5a85303cf79a3f8850bb9aabe8083b56b3f145eb8a25da484.jpg)

![378a368adfe3667f05d52b7ef7b28363b0a60c980d70af8a0102d181c9cb8d83.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/378a368adfe3667f05d52b7ef7b28363b0a60c980d70af8a0102d181c9cb8d83.jpg)

![4c4829df62a42f1d566609c819abd4bd9b59b99960947065f71935a49084ae07.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/4c4829df62a42f1d566609c819abd4bd9b59b99960947065f71935a49084ae07.jpg)

![5f0538b4448c7717e1eddfc93a0e6cd24026e6e029433887a34daf9e8435ccef.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/5f0538b4448c7717e1eddfc93a0e6cd24026e6e029433887a34daf9e8435ccef.jpg)

![a1ead45e6b26ca5c9beae26482f2bf265cbfb64833ea2d1716bfe81cd86f900a.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/a1ead45e6b26ca5c9beae26482f2bf265cbfb64833ea2d1716bfe81cd86f900a.jpg)

![a74fa374d0973145ba4d8d8f1b7e13bb89d96d543e26fafa7691621fc8e4316c.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/a74fa374d0973145ba4d8d8f1b7e13bb89d96d543e26fafa7691621fc8e4316c.jpg)

![bd91acdd27aa2e082e900dc34c25b50b84199061168cff5d7e806c4efabee737.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/bd91acdd27aa2e082e900dc34c25b50b84199061168cff5d7e806c4efabee737.jpg)

![d14d84816d5c20da9e59bd6d9d6b0bd46e8945c4d235eb61a3d2e44a70fa0320.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/d14d84816d5c20da9e59bd6d9d6b0bd46e8945c4d235eb61a3d2e44a70fa0320.jpg)

![e8179ba1ba0cbc27a143f333ab6f52c203f3d2ac5b186fcbd9513b5ddf05741f.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/e8179ba1ba0cbc27a143f333ab6f52c203f3d2ac5b186fcbd9513b5ddf05741f.jpg)

![e9ed25d7d08336e9e61cc42df9c317ca0456dd2cf927af1cbe153b77c5859eda.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/images/e9ed25d7d08336e9e61cc42df9c317ca0456dd2cf927af1cbe153b77c5859eda.jpg)

### Tables

![063626c9b5b7bb6c16c6460bd48e0f7e40f4a442682a70d8e88ae883d3830772.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/063626c9b5b7bb6c16c6460bd48e0f7e40f4a442682a70d8e88ae883d3830772.jpg)

![1fa945e9f50076a4747ac20adcbebbab144fbdae14006e50c3b28cebac43a67e.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/1fa945e9f50076a4747ac20adcbebbab144fbdae14006e50c3b28cebac43a67e.jpg)

![2d66cdca29e3556d8eeb15a2629c7d4e6dc78739e1b909ff82beb3bbbcc56142.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/2d66cdca29e3556d8eeb15a2629c7d4e6dc78739e1b909ff82beb3bbbcc56142.jpg)

![3c5514d9afecc3b7f602d36ee3ea40b79a32ec9445050e2208fe5a8cf9c4dc99.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/3c5514d9afecc3b7f602d36ee3ea40b79a32ec9445050e2208fe5a8cf9c4dc99.jpg)

![485ad2d6dc22782c1a93aeae3fdda6ad7d653c5fc45b1c7a429fbd1e2e4a069d.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/485ad2d6dc22782c1a93aeae3fdda6ad7d653c5fc45b1c7a429fbd1e2e4a069d.jpg)

![7aedf9d89b3d100659c6d41dfe7a4537400338a29b06475e448ca330b3ceccac.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/7aedf9d89b3d100659c6d41dfe7a4537400338a29b06475e448ca330b3ceccac.jpg)

![7fb01006749f947a8ab612656635d193d00f2ba3d90dd0e742da6f9861a3a0a1.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/7fb01006749f947a8ab612656635d193d00f2ba3d90dd0e742da6f9861a3a0a1.jpg)

![82be9dee3e5b8867c09291584aaafb9e6a08ea5f7581ac4e0acd64c65d6208a9.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/82be9dee3e5b8867c09291584aaafb9e6a08ea5f7581ac4e0acd64c65d6208a9.jpg)

![8504620c835b73e1dd8a1c29e9e88d1705c1c347f91f232439b2b4b108be25a6.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/8504620c835b73e1dd8a1c29e9e88d1705c1c347f91f232439b2b4b108be25a6.jpg)

![8e309b08807cd2033100cd138b403c37460137e38f378e827e1ef92f981f33e9.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/8e309b08807cd2033100cd138b403c37460137e38f378e827e1ef92f981f33e9.jpg)

![a05ee7c3da9fbe9020de304ff8364f2f5c6748bdeedbfb4d9fb7ff17db70c2ed.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/a05ee7c3da9fbe9020de304ff8364f2f5c6748bdeedbfb4d9fb7ff17db70c2ed.jpg)

![baa15a7d5fd6124520cff5d36a403bf6cb8d8009a9624cff978439302cd32fde.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/baa15a7d5fd6124520cff5d36a403bf6cb8d8009a9624cff978439302cd32fde.jpg)

![da2a2ed19e086084962f9ef00d38ec13ee161675fff2b034bef52484f3063b3c.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/da2a2ed19e086084962f9ef00d38ec13ee161675fff2b034bef52484f3063b3c.jpg)

![f91d8d9a151b37b22e6e1f493c6595ee49c78b77e9129de2b54cb26c36850522.jpg](../icml_results/4226_Plan%20in%20Sandbox%2C%20Navigate%20in%20Open%20Worlds_%20Learning%20Physics-Grounded%20Abstracted%20Experience%20for%20Embodi/tables/f91d8d9a151b37b22e6e1f493c6595ee49c78b77e9129de2b54cb26c36850522.jpg)

<a id="paper-4185"></a>

## Breaking the Scale Barrier: One-Shot Knowledge Transfer via Frequency Transform


### Images

![10ba1933ecb69949ca6b0a7fd7e2b0f2f2da3e415d4bba71d88b7bb625dbd69b.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/10ba1933ecb69949ca6b0a7fd7e2b0f2f2da3e415d4bba71d88b7bb625dbd69b.jpg)

![2d3cd23106e939cbdb82288b92467c138cdb31b7670c81017e83c1c1ce8f083d.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/2d3cd23106e939cbdb82288b92467c138cdb31b7670c81017e83c1c1ce8f083d.jpg)

![5464cad9c5b5b2a559a9d5bd639eb7b9a18416eeba835767ef6d498ae7a3be43.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/5464cad9c5b5b2a559a9d5bd639eb7b9a18416eeba835767ef6d498ae7a3be43.jpg)

![60886555392cee4106e8beb1fd863f1f3d15df05b18f99c0ab418f3d9f3488ca.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/60886555392cee4106e8beb1fd863f1f3d15df05b18f99c0ab418f3d9f3488ca.jpg)

![8ad4fcfbf7bd52e17bf960232f445dc46ebd03bcf933e5f6968de5054016d8b9.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/8ad4fcfbf7bd52e17bf960232f445dc46ebd03bcf933e5f6968de5054016d8b9.jpg)

![9683300d32b453e81673e77abea5e9b2676762ef86f99e6ae993a24423a326e1.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/9683300d32b453e81673e77abea5e9b2676762ef86f99e6ae993a24423a326e1.jpg)

![a260e6b1ba77e3b0d522f38fa4e676efcb01be583dc5b2147e669909221a627a.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/a260e6b1ba77e3b0d522f38fa4e676efcb01be583dc5b2147e669909221a627a.jpg)

![f3cc770a2ff501f686162afe3696e333536da554794a23d3ef9e26d423259b3c.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/images/f3cc770a2ff501f686162afe3696e333536da554794a23d3ef9e26d423259b3c.jpg)

### Tables

![11f82fe60f3688631fa987b5791e77e6cbc5739e553eee1501f2310bf14225b3.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/11f82fe60f3688631fa987b5791e77e6cbc5739e553eee1501f2310bf14225b3.jpg)

![160f0b77734dd0aa97ec4fcbcc8bb88fd0f8bbc15d2ae539ee0265da3a8288de.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/160f0b77734dd0aa97ec4fcbcc8bb88fd0f8bbc15d2ae539ee0265da3a8288de.jpg)

![1ebaaf9677be670b84dd5f585d2a3345e0fd1a1945e244adf9b5cfb0e07c2009.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/1ebaaf9677be670b84dd5f585d2a3345e0fd1a1945e244adf9b5cfb0e07c2009.jpg)

![2318138d1a3c8b04bfca1e35060764c88d61de67d536708270918f1bbda43534.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/2318138d1a3c8b04bfca1e35060764c88d61de67d536708270918f1bbda43534.jpg)

![27276062de28475c48a327b5411ce4118fe745250c70f105b53193cbb9619368.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/27276062de28475c48a327b5411ce4118fe745250c70f105b53193cbb9619368.jpg)

![27c6989258b5ca6c196963485fa73462dc17d04473cd27507746a5b172f3efa8.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/27c6989258b5ca6c196963485fa73462dc17d04473cd27507746a5b172f3efa8.jpg)

![3f1370861b89c8d3c62d2308f7811c6ba19ee2a495049271e309d81c1b077968.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/3f1370861b89c8d3c62d2308f7811c6ba19ee2a495049271e309d81c1b077968.jpg)

![55ac962390b330f15ccab8c2572e7092d7eb8a019aa35f88b1f56a4cc4871ac6.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/55ac962390b330f15ccab8c2572e7092d7eb8a019aa35f88b1f56a4cc4871ac6.jpg)

![56467cb69b2a487f2377f22fa33a78c45ae8815551e16b2f1ac0b30b3d8f8c8d.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/56467cb69b2a487f2377f22fa33a78c45ae8815551e16b2f1ac0b30b3d8f8c8d.jpg)

![6ce58df5fe46058d59a76a511766a581177b94c58e10f9ac8173af2df2845081.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/6ce58df5fe46058d59a76a511766a581177b94c58e10f9ac8173af2df2845081.jpg)

![73df7be9672c4f238c34bd46bd3d58d77d5a5de287acb8f74a810ca80dc04210.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/73df7be9672c4f238c34bd46bd3d58d77d5a5de287acb8f74a810ca80dc04210.jpg)

![c410b1699463718f088ae4f15519af77f067ad8d3e6de7421b17fea90fd10c79.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/c410b1699463718f088ae4f15519af77f067ad8d3e6de7421b17fea90fd10c79.jpg)

![c6397427d4e1b3535eebb0d411fa4dd4b21aa075a5ab53c0871103b13520933f.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/c6397427d4e1b3535eebb0d411fa4dd4b21aa075a5ab53c0871103b13520933f.jpg)

![d342387e55ab2042388544a4272581fee787f0e92fba3c2fd6f2666f9ee5060c.jpg](../icml_results/4227_Breaking%20the%20Scale%20Barrier_%20One-Shot%20Knowledge%20Transfer%20via%20Frequency%20Transform/tables/d342387e55ab2042388544a4272581fee787f0e92fba3c2fd6f2666f9ee5060c.jpg)

<a id="paper-4186"></a>

## Rényi Diffusion Models


### Images

![0f64df0dfeb0f54b3aa81557ae8aa4aed9fc5e54a1fbfc5b10111fd4020c85fc.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/0f64df0dfeb0f54b3aa81557ae8aa4aed9fc5e54a1fbfc5b10111fd4020c85fc.jpg)

![466de19f665f8d26afa8b9f154434b976ed4f48e24f7bdee6cb1e1e2f0b7d39b.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/466de19f665f8d26afa8b9f154434b976ed4f48e24f7bdee6cb1e1e2f0b7d39b.jpg)

![4b8524d2439a6a70c8224b3f3ac0cee162c1b1bb4e3c1360294478af44d495ac.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/4b8524d2439a6a70c8224b3f3ac0cee162c1b1bb4e3c1360294478af44d495ac.jpg)

![96373c2e1c085c967390c4b3f8f1ee8574a929fa4e70cc9a9352153b668bdab0.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/96373c2e1c085c967390c4b3f8f1ee8574a929fa4e70cc9a9352153b668bdab0.jpg)

![b6d1e643d8700469cf2706063919567f95ef6072c2945f7cd66a6ce4f4cfb910.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/b6d1e643d8700469cf2706063919567f95ef6072c2945f7cd66a6ce4f4cfb910.jpg)

![c6de0fa7a48bcae4fa4e78e6f7b92a718c6e063f83e9ef6648a7c2903fdf87df.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/c6de0fa7a48bcae4fa4e78e6f7b92a718c6e063f83e9ef6648a7c2903fdf87df.jpg)

![e42ceadef36a33d92267a3bf6c2c4d6aa1930a9c8bf00aeb12503f67c7e621f3.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/e42ceadef36a33d92267a3bf6c2c4d6aa1930a9c8bf00aeb12503f67c7e621f3.jpg)

![feb011ca2446ed9ad0cabf721207d905da501ed59e3a37291ae77452858b8756.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/images/feb011ca2446ed9ad0cabf721207d905da501ed59e3a37291ae77452858b8756.jpg)

### Tables

![0450ac8dfce1629fb49e761bcaf08843a80633972fa25534d4523db201ca74f0.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/0450ac8dfce1629fb49e761bcaf08843a80633972fa25534d4523db201ca74f0.jpg)

![19b37053a24950ac2e47e8d2c61ac368e1ea27dd7e02b5c54de40944257b1b37.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/19b37053a24950ac2e47e8d2c61ac368e1ea27dd7e02b5c54de40944257b1b37.jpg)

![2bc944ebd1203867761fab070bcc5ce4f6e5414c0a5037f838d751341942167b.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/2bc944ebd1203867761fab070bcc5ce4f6e5414c0a5037f838d751341942167b.jpg)

![4b82aafec9b21ec8f251c49985561b408796d43fc623a6200adc178609f86b1b.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/4b82aafec9b21ec8f251c49985561b408796d43fc623a6200adc178609f86b1b.jpg)

![5190174c8db5a1688a460ba0636d9c4e1d4f8015000580577c3c56ca7b3771dc.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/5190174c8db5a1688a460ba0636d9c4e1d4f8015000580577c3c56ca7b3771dc.jpg)

![91564d91cd65157faf9f87eac6d45d961891a8a45f66e7dfbc670fe0965bd6f3.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/91564d91cd65157faf9f87eac6d45d961891a8a45f66e7dfbc670fe0965bd6f3.jpg)

![c20259d81deabb78fe08bc92775926053831f17d14a1d142b4a8845cbfd38490.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/c20259d81deabb78fe08bc92775926053831f17d14a1d142b4a8845cbfd38490.jpg)

![c20a38f7e5427489ad4bb7d5ab2424d2a5942ca8a02f47e85baff885da037e37.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/c20a38f7e5427489ad4bb7d5ab2424d2a5942ca8a02f47e85baff885da037e37.jpg)

![eb950907d4597082857d0c188bc57727a8c98fa8f2a536c5f01239186e5780f8.jpg](../icml_results/4228_R%C3%A9nyi%20Diffusion%20Models/tables/eb950907d4597082857d0c188bc57727a8c98fa8f2a536c5f01239186e5780f8.jpg)

<a id="paper-4187"></a>

## ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling


### Images

![03c27958d8007d26e1afc2649fc3007720a8bdb13c6c403bc6d839403a0cd747.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/03c27958d8007d26e1afc2649fc3007720a8bdb13c6c403bc6d839403a0cd747.jpg)

![0a37e2ab7dcb1bcc6649a2643ba139623d4bbf953be368313db67e8c652f4bae.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/0a37e2ab7dcb1bcc6649a2643ba139623d4bbf953be368313db67e8c652f4bae.jpg)

![1330c24300f06b970cf838e07d02cba319d3d36708c910089e5cac7a459515af.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/1330c24300f06b970cf838e07d02cba319d3d36708c910089e5cac7a459515af.jpg)

![22ca3a949e952154887936a4199f9b7fe4c93029d378253114d1a7702301dcef.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/22ca3a949e952154887936a4199f9b7fe4c93029d378253114d1a7702301dcef.jpg)

![4123e362f81531147d32bf6d0e63c250e0bbc3e228d5e9193e64b5c5032d9410.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/4123e362f81531147d32bf6d0e63c250e0bbc3e228d5e9193e64b5c5032d9410.jpg)

![67f86acc48e4db779856aeca000a4aa4444b4f27b7ecf2e12963cfbb61b596d1.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/67f86acc48e4db779856aeca000a4aa4444b4f27b7ecf2e12963cfbb61b596d1.jpg)

![6f4124eda9ebabbefcfdaa7068e58be4950473d985beb46b667e34fe18c2a655.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/6f4124eda9ebabbefcfdaa7068e58be4950473d985beb46b667e34fe18c2a655.jpg)

![8e557ba7a6c3fdbbc5244e91f15d1d003b6b2cc58a5f2176ec31910f848e6321.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/8e557ba7a6c3fdbbc5244e91f15d1d003b6b2cc58a5f2176ec31910f848e6321.jpg)

![ae18d0b1ffeccdaa123d3ce9084e0b81fe66b8e2190ecbf8984f2beecb20e3e3.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/ae18d0b1ffeccdaa123d3ce9084e0b81fe66b8e2190ecbf8984f2beecb20e3e3.jpg)

![b5b7b201da00c5ec52c1ef91765a489631449b0002214617a4cb26beb214641e.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/images/b5b7b201da00c5ec52c1ef91765a489631449b0002214617a4cb26beb214641e.jpg)

### Tables

![01c97f2a11dc83efc531c97b1be3d5c98e41ae189860a45058eeb7ed471af037.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/01c97f2a11dc83efc531c97b1be3d5c98e41ae189860a45058eeb7ed471af037.jpg)

![040da8f7c9ac4447085a5cd4e3717567a7a6a599e55ee77b3058a36d64a55a10.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/040da8f7c9ac4447085a5cd4e3717567a7a6a599e55ee77b3058a36d64a55a10.jpg)

![09fbdb630198d47374df61b8d30099507f6375978205b8df7c5a30a5fb8a62aa.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/09fbdb630198d47374df61b8d30099507f6375978205b8df7c5a30a5fb8a62aa.jpg)

![0a56560b5d4b25bbcc3ab985c8823088b50e2196f701f7f1ae08ec003da8a2e7.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/0a56560b5d4b25bbcc3ab985c8823088b50e2196f701f7f1ae08ec003da8a2e7.jpg)

![1342062398a09be7a815147c3ccfdba447cb3b218179b6aecad1baafb8ed57ee.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/1342062398a09be7a815147c3ccfdba447cb3b218179b6aecad1baafb8ed57ee.jpg)

![13bd9f10cd4b53c01f560d5b08c535d476522a5b656a4a0e1c47a2213e8641f8.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/13bd9f10cd4b53c01f560d5b08c535d476522a5b656a4a0e1c47a2213e8641f8.jpg)

![15e27bdcc18f39299d9fd5e052603adce0e81390c2d5437c0672f725825e2f2c.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/15e27bdcc18f39299d9fd5e052603adce0e81390c2d5437c0672f725825e2f2c.jpg)

![1cdcf28771eebe4a04258139886e442b63564d7bdf4544581229cbdf7f71d9ce.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/1cdcf28771eebe4a04258139886e442b63564d7bdf4544581229cbdf7f71d9ce.jpg)

![1d75b7db6bbc4ea37741aa13fd8a2dcb7ec20b7a1307291993e210db3ad9ddbd.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/1d75b7db6bbc4ea37741aa13fd8a2dcb7ec20b7a1307291993e210db3ad9ddbd.jpg)

![230e821a6e67a0419a4e2054d864cfa6ffe17d9fc5702a21b004ebacd3c37965.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/230e821a6e67a0419a4e2054d864cfa6ffe17d9fc5702a21b004ebacd3c37965.jpg)

![238893ee8cdf4302afd54341e9ff714fa2aa5b93f31f1cb2889636a33533fc0c.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/238893ee8cdf4302afd54341e9ff714fa2aa5b93f31f1cb2889636a33533fc0c.jpg)

![2515c16041aa3f20a2029f36dff1a9e7944302a6ca6c3f94eee5be4ae12dadfd.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/2515c16041aa3f20a2029f36dff1a9e7944302a6ca6c3f94eee5be4ae12dadfd.jpg)

![25c052d33e656ee74422c46273aae6664ed39c3e6cdede15593ccbce046481a9.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/25c052d33e656ee74422c46273aae6664ed39c3e6cdede15593ccbce046481a9.jpg)

![2b574ca80f675417de79a855febf1234b648ebd032c24151a6c7529ec009a608.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/2b574ca80f675417de79a855febf1234b648ebd032c24151a6c7529ec009a608.jpg)

![2ca4a85cf7dd7838268f342d1fd83d74a8bfc442f3b3bc82dd1b390f1757675c.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/2ca4a85cf7dd7838268f342d1fd83d74a8bfc442f3b3bc82dd1b390f1757675c.jpg)

![340fbb542b4c6968ee8ca6e730370481b89ace2dd93276b5fec4ee42104e47fc.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/340fbb542b4c6968ee8ca6e730370481b89ace2dd93276b5fec4ee42104e47fc.jpg)

![353e385989e92ab0539ca41f582d092d1aa92b202cb35e18c49a0bdb6cf02127.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/353e385989e92ab0539ca41f582d092d1aa92b202cb35e18c49a0bdb6cf02127.jpg)

![393ade35602b88c8fff8e4d4094b64f58571cbaf0026c55019f45908be0b4432.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/393ade35602b88c8fff8e4d4094b64f58571cbaf0026c55019f45908be0b4432.jpg)

![40284569be8d076b08943b209cc7c71de1ff647da2caa465ba47bca2a2ea99c2.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/40284569be8d076b08943b209cc7c71de1ff647da2caa465ba47bca2a2ea99c2.jpg)

![41aa0bf749d94da085d8bca7223f09c7fd61feef0086d7274921c01fe798f1fc.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/41aa0bf749d94da085d8bca7223f09c7fd61feef0086d7274921c01fe798f1fc.jpg)

![43a0ddac2b59a01edb83ca3f520e76031d943a5d54575838ea0da2a65e5f4861.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/43a0ddac2b59a01edb83ca3f520e76031d943a5d54575838ea0da2a65e5f4861.jpg)

![617732428dcf0a47794d18be0da8d636c853c86022c7ec3186458fa9ec097fa0.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/617732428dcf0a47794d18be0da8d636c853c86022c7ec3186458fa9ec097fa0.jpg)

![664f94a08662e145b5d953f2b220b08e49517f78453bd099f8c7c6678e16c927.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/664f94a08662e145b5d953f2b220b08e49517f78453bd099f8c7c6678e16c927.jpg)

![6afd3fe4456f91d6b583176ffbccc36a58b0535229e5ef00728483ceae5a3a74.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/6afd3fe4456f91d6b583176ffbccc36a58b0535229e5ef00728483ceae5a3a74.jpg)

![733d89610a9e069ee3fc097a84cabb981d56c605efa9532251ba4313551ad3f4.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/733d89610a9e069ee3fc097a84cabb981d56c605efa9532251ba4313551ad3f4.jpg)

![7d42775b8cce5d47cd1fc32f73c3ea6a30f439f83183d2e97fe4794f3d1a2d4b.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/7d42775b8cce5d47cd1fc32f73c3ea6a30f439f83183d2e97fe4794f3d1a2d4b.jpg)

![7ef1590e9682fb578ba1b0e30b799a8bae153dea7d20fb8c380c3d58be3a17a8.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/7ef1590e9682fb578ba1b0e30b799a8bae153dea7d20fb8c380c3d58be3a17a8.jpg)

![85a96d450ddac63639086d90e9363bc09af05cd7eaa090e31c75090299f04502.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/85a96d450ddac63639086d90e9363bc09af05cd7eaa090e31c75090299f04502.jpg)

![aaac52b3fc0f2093200145ad0afac914d7f7df282220ec70c138391e486c5862.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/aaac52b3fc0f2093200145ad0afac914d7f7df282220ec70c138391e486c5862.jpg)

![abe27ee9421f2440fa2d73ed31c334683f1a1a360d88a0dee1b72a4f77d1e2bb.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/abe27ee9421f2440fa2d73ed31c334683f1a1a360d88a0dee1b72a4f77d1e2bb.jpg)

![b9aa68e30c853fb5b75706fb422d881018069be0931197f8be3f282a41cfea2b.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/b9aa68e30c853fb5b75706fb422d881018069be0931197f8be3f282a41cfea2b.jpg)

![c760468cc9695b0314433390aeee7359b7212b285f997a9873566ab0a3efe2eb.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/c760468cc9695b0314433390aeee7359b7212b285f997a9873566ab0a3efe2eb.jpg)

![cb76de66935fb996988af50a6f9ce4d619a389b95ec1d934118b5945c7e2a2e8.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/cb76de66935fb996988af50a6f9ce4d619a389b95ec1d934118b5945c7e2a2e8.jpg)

![cfaaabc3da962010aa34e8d7252ace5f4ac3f5fa114779cecc55e5d9a31b450e.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/cfaaabc3da962010aa34e8d7252ace5f4ac3f5fa114779cecc55e5d9a31b450e.jpg)

![d7f0368ee360b923813a2ad03de522220da27794532f616b0daf9a4453faaf98.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/d7f0368ee360b923813a2ad03de522220da27794532f616b0daf9a4453faaf98.jpg)

![d84d6629b00c10a735883de01b4a83b0b4377d2d85e30ef9af0b63352b10bed3.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/d84d6629b00c10a735883de01b4a83b0b4377d2d85e30ef9af0b63352b10bed3.jpg)

![de80c630655d0e702f987cdb760ce4ea96bc20dfc5bd59dd6339a21f4b70dacb.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/de80c630655d0e702f987cdb760ce4ea96bc20dfc5bd59dd6339a21f4b70dacb.jpg)

![e471ca7e48e37ce58b3f5e57a4d5ce3dd93018bfe0fc94d1fdb66fde4e7aa601.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/e471ca7e48e37ce58b3f5e57a4d5ce3dd93018bfe0fc94d1fdb66fde4e7aa601.jpg)

![e9d313ff72b3f79f56d42784d15267cee8bd3aef2d50c0149868aacd2ceeba8a.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/e9d313ff72b3f79f56d42784d15267cee8bd3aef2d50c0149868aacd2ceeba8a.jpg)

![f4a750dd6dc5d6e300ffda30979c49dd357fe41162748e196f54dbee3f7cef53.jpg](../icml_results/4229_ProOPF_%20Benchmarking%20and%20Improving%20LLMs%20for%20Professional-Grade%20Power%20Systems%20Optimization%20Modeling/tables/f4a750dd6dc5d6e300ffda30979c49dd357fe41162748e196f54dbee3f7cef53.jpg)

<a id="paper-4188"></a>

## Training-Trajectory-Aware Token Selection


### Images

![055bc155967184a76331a8514ed2f2e6ae1145728de216bd766885bc7fe39d83.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/055bc155967184a76331a8514ed2f2e6ae1145728de216bd766885bc7fe39d83.jpg)

![22ffecb5043342447caecdb8841db8862cebddc34aebfcd6f4a9d1a61f1d6d0c.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/22ffecb5043342447caecdb8841db8862cebddc34aebfcd6f4a9d1a61f1d6d0c.jpg)

![2c0c722734be4003f191c05adacb8f017798885b98e08c82a18c8fa254d4f3bf.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/2c0c722734be4003f191c05adacb8f017798885b98e08c82a18c8fa254d4f3bf.jpg)

![31b2d4bc06dbbc1cd0b9a598c1074f787bcd77fe21a8538800cd74ae712a7a13.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/31b2d4bc06dbbc1cd0b9a598c1074f787bcd77fe21a8538800cd74ae712a7a13.jpg)

![57a67415012bdb98d6218ee2562de8de06ebbc099579c0b9a0e1b75a112279ba.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/57a67415012bdb98d6218ee2562de8de06ebbc099579c0b9a0e1b75a112279ba.jpg)

![59b56cc12e45b02b3894d69d0cb8c9a54b1217d156863aa3ddb70650f47bf5d5.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/59b56cc12e45b02b3894d69d0cb8c9a54b1217d156863aa3ddb70650f47bf5d5.jpg)

![70daac336db5d69700b2dba29224b977937430b22eb3e929092547d5e6b8d1e9.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/70daac336db5d69700b2dba29224b977937430b22eb3e929092547d5e6b8d1e9.jpg)

![a4f6653422819e33e909ee8055af7a94496099cabc444ffcda8757c14616f988.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/a4f6653422819e33e909ee8055af7a94496099cabc444ffcda8757c14616f988.jpg)

![ab9715c1e28a8d2a38a6a5cea9d0a0e5879dbe6bda1d1a9e01f6e45ed4b86fcd.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/ab9715c1e28a8d2a38a6a5cea9d0a0e5879dbe6bda1d1a9e01f6e45ed4b86fcd.jpg)

![ab97c5e1e33a1714b7727d5d6b6808b5e340ebc4997e43634fd0bf9ad59ad023.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/ab97c5e1e33a1714b7727d5d6b6808b5e340ebc4997e43634fd0bf9ad59ad023.jpg)

![ae4698de3176094f0c15faa6824feb2e5dd50c50b5d508545a122a6a77bf832e.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/ae4698de3176094f0c15faa6824feb2e5dd50c50b5d508545a122a6a77bf832e.jpg)

![ba7f0b79967c3a2577a7e2914b09ad3957e4b14611f02877fddedde8cdd2fcbf.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/ba7f0b79967c3a2577a7e2914b09ad3957e4b14611f02877fddedde8cdd2fcbf.jpg)

![c7f8c3b1acc87d84fdc77d6728545187d2895c474d9cbac763cb9f65d6073837.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/c7f8c3b1acc87d84fdc77d6728545187d2895c474d9cbac763cb9f65d6073837.jpg)

![d939f617cb77e62e4d2b43afc6521e26de2365a1ca112dded48534a6f68805db.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/images/d939f617cb77e62e4d2b43afc6521e26de2365a1ca112dded48534a6f68805db.jpg)

### Tables

![035939f6164dee4d6054023a575fc6da90ab1a48c335554ecea15f9917282e05.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/035939f6164dee4d6054023a575fc6da90ab1a48c335554ecea15f9917282e05.jpg)

![0646c99df225b421f29fee57b19a9bfdcaca98f37d22ba206cf4ee9be6032e08.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/0646c99df225b421f29fee57b19a9bfdcaca98f37d22ba206cf4ee9be6032e08.jpg)

![1077536ed53eca9940e1f5ec37c51b3d7e2cb8594d46e0936a9333a909f506b0.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/1077536ed53eca9940e1f5ec37c51b3d7e2cb8594d46e0936a9333a909f506b0.jpg)

![10fee88470f62080ac04e9978c7193ee6d569a96971873268072557a44994787.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/10fee88470f62080ac04e9978c7193ee6d569a96971873268072557a44994787.jpg)

![261df58b6e9195bb7174c27aea58935c503323fc366e7507415d14ece37ba0e5.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/261df58b6e9195bb7174c27aea58935c503323fc366e7507415d14ece37ba0e5.jpg)

![2aedbe71475d980ee512f3d4c041bba50c3e208318c5ce06c5466d97687a6ada.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/2aedbe71475d980ee512f3d4c041bba50c3e208318c5ce06c5466d97687a6ada.jpg)

![347448f0ed12662b8cb94e7f592e653a405c42c923c2fe6414f2bcd84bc2d74f.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/347448f0ed12662b8cb94e7f592e653a405c42c923c2fe6414f2bcd84bc2d74f.jpg)

![47323ccdff79d41699426e509a76912bad41341cf3dd3b11025d6be09ce4b3dc.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/47323ccdff79d41699426e509a76912bad41341cf3dd3b11025d6be09ce4b3dc.jpg)

![4b8e4d70eec1a44a09b943954be39762176c716ec57acc8762961faa642eaa6b.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/4b8e4d70eec1a44a09b943954be39762176c716ec57acc8762961faa642eaa6b.jpg)

![52c74cc5f536afcdd99f6ff5009cfcea56d3c70f6d338f95fc9afff4cab8104e.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/52c74cc5f536afcdd99f6ff5009cfcea56d3c70f6d338f95fc9afff4cab8104e.jpg)

![6d3e4cbfcafb7341ef78d0ebaf4cf8e036a7951ed4e6ecbdec94095e65be6fe9.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/6d3e4cbfcafb7341ef78d0ebaf4cf8e036a7951ed4e6ecbdec94095e65be6fe9.jpg)

![7f41f00e9b87ac5f551bb493dafcd2c1701862d1bb3f011c3935c023ad1e4ce3.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/7f41f00e9b87ac5f551bb493dafcd2c1701862d1bb3f011c3935c023ad1e4ce3.jpg)

![91c02168f0cf0175281d5925295289462e3f39e4ab081c567561c9717407cd1b.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/91c02168f0cf0175281d5925295289462e3f39e4ab081c567561c9717407cd1b.jpg)

![9c7720102570c9826cd578a1c8631e74a4bcac1f682c4c1f29a05a26d400f4f0.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/9c7720102570c9826cd578a1c8631e74a4bcac1f682c4c1f29a05a26d400f4f0.jpg)

![a37c59e8b8b0401a51245b43fafd77bc6b99daee7146979107b79e46bfc95334.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/a37c59e8b8b0401a51245b43fafd77bc6b99daee7146979107b79e46bfc95334.jpg)

![c7b73412711ee1a4d6fe1d2f2467f93886755183b2cc0caa0e0f49ebb4a74212.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/c7b73412711ee1a4d6fe1d2f2467f93886755183b2cc0caa0e0f49ebb4a74212.jpg)

![d7540ca85040b4ca6f37e2bd683993da5f8da066bf0b4a59b3542af3db1ead77.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/d7540ca85040b4ca6f37e2bd683993da5f8da066bf0b4a59b3542af3db1ead77.jpg)

![f2d9af0030ebc7df8e0f090df1c0a4dc71a1a38235f291e35fda362b68aa2079.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/f2d9af0030ebc7df8e0f090df1c0a4dc71a1a38235f291e35fda362b68aa2079.jpg)

![f41b827ba3c34fa41c6325ba7c6e65cadac6a95c580f5d5e9851990de7a4b276.jpg](../icml_results/4230_Training-Trajectory-Aware%20Token%20Selection/tables/f41b827ba3c34fa41c6325ba7c6e65cadac6a95c580f5d5e9851990de7a4b276.jpg)

<a id="paper-4189"></a>

## TRIP-Bench: A Benchmark for Long-Horizon Interactive Agents in Real-World Scenarios


### Images

![35d3908903b184665f134546e1bba527c54deea6cc1ffca8e9f715c0ff829c49.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/35d3908903b184665f134546e1bba527c54deea6cc1ffca8e9f715c0ff829c49.jpg)

![3e0b954c766e061d1eba7d6c091f734cbf341b0a8cae40e536ac731857360244.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/3e0b954c766e061d1eba7d6c091f734cbf341b0a8cae40e536ac731857360244.jpg)

![64dbf17c53c363b570b7d410a5979feccfe015ced403a1b0d1c3456e203377e3.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/64dbf17c53c363b570b7d410a5979feccfe015ced403a1b0d1c3456e203377e3.jpg)

![6a50f80988d14da17cf8ae1c91d2e3191683903db714de064f9d3e997a59cabc.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/6a50f80988d14da17cf8ae1c91d2e3191683903db714de064f9d3e997a59cabc.jpg)

![abf9cfeea0d33405393eed882e01ba497765b46df1877d2f9b4428ccf25c0fc5.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/abf9cfeea0d33405393eed882e01ba497765b46df1877d2f9b4428ccf25c0fc5.jpg)

![e2ab7aa0783d955f41681328a1b6d3150f83e4cae0eec07d6c5b925901f453c1.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/e2ab7aa0783d955f41681328a1b6d3150f83e4cae0eec07d6c5b925901f453c1.jpg)

![f5eafc67e4a068b5f087c1784fa0348777a6d1f13c8090eab1e67d9998f366b2.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/images/f5eafc67e4a068b5f087c1784fa0348777a6d1f13c8090eab1e67d9998f366b2.jpg)

### Tables

![08fd02d74dcf5e6a0005173c86a754f74b3c4dbf530e46b09864a7810cca8baa.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/08fd02d74dcf5e6a0005173c86a754f74b3c4dbf530e46b09864a7810cca8baa.jpg)

![09d6f581414e9c9c23f0f2aaca38bea6ab510d9f52809bc004886202c6e8071a.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/09d6f581414e9c9c23f0f2aaca38bea6ab510d9f52809bc004886202c6e8071a.jpg)

![0bb40499da2b2ceae148ebccf17e8a276a9f4284277ea252b1e8b26a26b7c9e4.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/0bb40499da2b2ceae148ebccf17e8a276a9f4284277ea252b1e8b26a26b7c9e4.jpg)

![a13d8be86a27da861a1ece12ec7aa529111abc23b6a6f640251141a481c98957.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/a13d8be86a27da861a1ece12ec7aa529111abc23b6a6f640251141a481c98957.jpg)

![cf08ca7555bd371c14f35aebc02a44c4702f206e301e01675e8fdfef2d6d1364.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/cf08ca7555bd371c14f35aebc02a44c4702f206e301e01675e8fdfef2d6d1364.jpg)

![f08bb00136fe40f3ba8a420b2a4f7b54006cfd44f259e6e9ee19c6a2cfb65629.jpg](../icml_results/4231_TRIP-Bench_%20A%20Benchmark%20for%20Long-Horizon%20Interactive%20Agents%20in%20Real-World%20Scenarios/tables/f08bb00136fe40f3ba8a420b2a4f7b54006cfd44f259e6e9ee19c6a2cfb65629.jpg)

<a id="paper-4190"></a>

## SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space


### Images

![05f40a39047771b4488431711a4c91441b9bf18e308087324ea2ffb4560647f7.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/05f40a39047771b4488431711a4c91441b9bf18e308087324ea2ffb4560647f7.jpg)

![0ce57a4c1704a128ae596002a58825170efa60261c254746305a9de3c792f076.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/0ce57a4c1704a128ae596002a58825170efa60261c254746305a9de3c792f076.jpg)

![281aea44bed415033ce003f6fc2a2671809817b016075ca774ef7877b0960ae4.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/281aea44bed415033ce003f6fc2a2671809817b016075ca774ef7877b0960ae4.jpg)

![330618aa7cbf5c67e31b07d1c1f1237a44be9ac41305a272d09b6dfc8b84b980.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/330618aa7cbf5c67e31b07d1c1f1237a44be9ac41305a272d09b6dfc8b84b980.jpg)

![3d59f0c0e4999d6df2c1ee7ff26ec2640b904539b233bae1ab37160edfb48052.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/3d59f0c0e4999d6df2c1ee7ff26ec2640b904539b233bae1ab37160edfb48052.jpg)

![57b5b0d14be48e9bd1f2497bf1005f24ad205bbba684a6644b25b8fe943ba5fa.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/57b5b0d14be48e9bd1f2497bf1005f24ad205bbba684a6644b25b8fe943ba5fa.jpg)

![6643a339df13722faf1f3f0277fad2a662fc8456d0578f72fddf09e4f446a4c5.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/6643a339df13722faf1f3f0277fad2a662fc8456d0578f72fddf09e4f446a4c5.jpg)

![7b2c374296547960e0a1e49fcbb12fa46ed708c9333119ea4f270909d0611e85.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/7b2c374296547960e0a1e49fcbb12fa46ed708c9333119ea4f270909d0611e85.jpg)

![a28f3a8415751d9d28dfb9088a0e042600e15a10436bca04c3ed3dd4f9c028ee.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/a28f3a8415751d9d28dfb9088a0e042600e15a10436bca04c3ed3dd4f9c028ee.jpg)

![f6eae5647dfc4616c2c870d823e96daaeb8b68c8bf2a408570635e1f5bb442a3.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/images/f6eae5647dfc4616c2c870d823e96daaeb8b68c8bf2a408570635e1f5bb442a3.jpg)

### Tables

![248f6a86bb3fd9899c4124d57f7970be0b72fe7a9699d89d70aa2d68f3a1f5a6.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/248f6a86bb3fd9899c4124d57f7970be0b72fe7a9699d89d70aa2d68f3a1f5a6.jpg)

![448936abd8388b998ea576d741f36c8e74313a14ea508ee6e95b0142a5a5fc58.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/448936abd8388b998ea576d741f36c8e74313a14ea508ee6e95b0142a5a5fc58.jpg)

![44de336c7e6fc12279a2fe146b9dac223268ae0b1f549308ac321a34e2308e27.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/44de336c7e6fc12279a2fe146b9dac223268ae0b1f549308ac321a34e2308e27.jpg)

![5bca8b891e36b20c445312a8059076375493e8aeeaa237ac28e5bf97040a7f5f.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/5bca8b891e36b20c445312a8059076375493e8aeeaa237ac28e5bf97040a7f5f.jpg)

![63182720732a7a96c62b09f76560208e81d6a044651ad0ca7b9e95304febdb47.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/63182720732a7a96c62b09f76560208e81d6a044651ad0ca7b9e95304febdb47.jpg)

![67cb05baccc5064ff0ed74618a4e5e8a9b31d0aaa11b59071fd8ce9ec02f0e0d.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/67cb05baccc5064ff0ed74618a4e5e8a9b31d0aaa11b59071fd8ce9ec02f0e0d.jpg)

![694c6004141fbf45b4a4189c0f1e5f526f135e8dc98e93b6d166ac4239847f85.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/694c6004141fbf45b4a4189c0f1e5f526f135e8dc98e93b6d166ac4239847f85.jpg)

![6af96f2169f5b4f1876916c5a4eda19853945805d41a906781ffac5ed1eb0b87.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/6af96f2169f5b4f1876916c5a4eda19853945805d41a906781ffac5ed1eb0b87.jpg)

![8d3fdb8acc6ff3b0730a3e9a929b0855576ff9297b929e3668566b646f9d9a80.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/8d3fdb8acc6ff3b0730a3e9a929b0855576ff9297b929e3668566b646f9d9a80.jpg)

![994b3a2f311a9226d4310d2014a0f3332f70858f0988b4e2328481ebe78290d6.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/994b3a2f311a9226d4310d2014a0f3332f70858f0988b4e2328481ebe78290d6.jpg)

![9b8c777fd8ec48201ccf0bf0a2d01264741e89a4d7bdffdaf982df4c1f669d4b.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/9b8c777fd8ec48201ccf0bf0a2d01264741e89a4d7bdffdaf982df4c1f669d4b.jpg)

![abe96a15d203000f90a6c4a213437e1ee30d9b778d893c34b2d667b5f310e707.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/abe96a15d203000f90a6c4a213437e1ee30d9b778d893c34b2d667b5f310e707.jpg)

![d438513ce1403f0ec382a16b77992913be075a9983222214de9de3c74a938e03.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/d438513ce1403f0ec382a16b77992913be075a9983222214de9de3c74a938e03.jpg)

![d6a3780e4085ef4832d17350012f612851ea8b0372bff0c482ccbd894dbc1f7f.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/d6a3780e4085ef4832d17350012f612851ea8b0372bff0c482ccbd894dbc1f7f.jpg)

![df55c0a5bc114f06703c12407b51e697a3ce7369319867eec10fffeb8afd884f.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/df55c0a5bc114f06703c12407b51e697a3ce7369319867eec10fffeb8afd884f.jpg)

![e002fa74ad37c8edaef7b560a8bb39cc34e963081a8421d04c3c7b625d2841a4.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/e002fa74ad37c8edaef7b560a8bb39cc34e963081a8421d04c3c7b625d2841a4.jpg)

![e32aa7415eb2ba4f5543027e9d33f1d6768e6555775d8964cf851807fcaea45e.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/e32aa7415eb2ba4f5543027e9d33f1d6768e6555775d8964cf851807fcaea45e.jpg)

![ef736e5acad8380791c8359b685016741cd3ba33ef0ff4e6e4bc9478ef8ef68c.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/ef736e5acad8380791c8359b685016741cd3ba33ef0ff4e6e4bc9478ef8ef68c.jpg)

![f19382bb0943124d8bf7c2ae871a20af776129a428c39dc9c9827e2acc1d8bc3.jpg](../icml_results/4232_SSA_%20Sparse%20Sparse%20Attention%20by%20Aligning%20Full%20and%20Sparse%20Attention%20Outputs%20in%20Feature%20Space/tables/f19382bb0943124d8bf7c2ae871a20af776129a428c39dc9c9827e2acc1d8bc3.jpg)

<a id="paper-4191"></a>

## TEFormer: Structured Bidirectional Temporal Enhancement Modeling in Spiking Transformers

### Images

![7df5e54b8eded7ea3c9ed5520c0f0bd14f4bb9b7d5778ba5ece66b93802542a5.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/images/7df5e54b8eded7ea3c9ed5520c0f0bd14f4bb9b7d5778ba5ece66b93802542a5.jpg)

![c1cc9b8a795494d7830c179f88b368e0bd69effec2d68f83256dc4dd551ba8fc.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/images/c1cc9b8a795494d7830c179f88b368e0bd69effec2d68f83256dc4dd551ba8fc.jpg)

### Tables

![097b3d2ca4765166a2a2541d3292cf9f1f1a297cfecf560d3ac968f482bdf315.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/097b3d2ca4765166a2a2541d3292cf9f1f1a297cfecf560d3ac968f482bdf315.jpg)

![134284ed5ca2e815682c744a22a56b38a9a3e44e77c2481615ac2f02e0e22725.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/134284ed5ca2e815682c744a22a56b38a9a3e44e77c2481615ac2f02e0e22725.jpg)

![4c4929c040804c12161bebd2e76e53b961dd2727f806792e93b1c3632b3b1f0d.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/4c4929c040804c12161bebd2e76e53b961dd2727f806792e93b1c3632b3b1f0d.jpg)

![4dbec8a1dc52a93f1f494856cdecf027b663a0ff0babfea6a8c7d9ee4859b2ba.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/4dbec8a1dc52a93f1f494856cdecf027b663a0ff0babfea6a8c7d9ee4859b2ba.jpg)

![812154be487b525e78e580fe8d6c72fef94edc60655966935018c885df19255d.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/812154be487b525e78e580fe8d6c72fef94edc60655966935018c885df19255d.jpg)

![885c4a63c1fa1f09a393490ae991ed81202ab1610aed3be4f8ba2795723654d5.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/885c4a63c1fa1f09a393490ae991ed81202ab1610aed3be4f8ba2795723654d5.jpg)

![8e4545705e7a5db3e7b37c0643e36a9eb3803343ed2c5f70215c6105cfa1b9fd.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/8e4545705e7a5db3e7b37c0643e36a9eb3803343ed2c5f70215c6105cfa1b9fd.jpg)

![a289b49408d28f096955e50d96e1f7c1db882a307b85edde4bfed858d6b4176d.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/a289b49408d28f096955e50d96e1f7c1db882a307b85edde4bfed858d6b4176d.jpg)

![acb968f048b6591bc960f377407e9bbf8ce6ddbbef28716abda850679a6116e0.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/acb968f048b6591bc960f377407e9bbf8ce6ddbbef28716abda850679a6116e0.jpg)

![caa3e56f2875c97a3ca6f470ece54d52613031e879b0e1c29166763cb2bf7fc7.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/caa3e56f2875c97a3ca6f470ece54d52613031e879b0e1c29166763cb2bf7fc7.jpg)

![ef687bb8f29c679647250d4a68d36879afe1e83d1e9053e314129b4f200203ca.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/ef687bb8f29c679647250d4a68d36879afe1e83d1e9053e314129b4f200203ca.jpg)

![f8313290e4120f9378c5a18a272ef61cefad832120f837b71ef750ac09192f12.jpg](../icml_results/4233_TEFormer_%20Structured%20Bidirectional%20Temporal%20Enhancement%20Modeling%20in%20Spiking%20Transformers/tables/f8313290e4120f9378c5a18a272ef61cefad832120f837b71ef750ac09192f12.jpg)
