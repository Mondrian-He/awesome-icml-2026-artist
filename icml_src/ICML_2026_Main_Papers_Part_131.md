# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 131 of 197

## 目录 (Table of Contents)

1. [Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver](#paper-4291)
2. [SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives](#paper-4292)
3. [Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control](#paper-4293)
4. [Variational Learning of Disentangled Representations](#paper-4294)
5. [Calibrating Generative Models to Distributional Constraints](#paper-4295)
6. [Causally Evaluating the Learnability of Formal Language Tasks](#paper-4296)
7. [Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](#paper-4297)
8. [Scalable and Interpretable Representation Alignment with Ordinal Similarity](#paper-4298)
9. [CountsDiff: A diffusion model on the natural numbers for generation and imputation of count-based data](#paper-4299)
10. [Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](#paper-4300)
11. [TT-Sparse: Learning Sparse Rule Models with Differentiable Truth Tables](#paper-4301)
12. [Process Reward Agents for Steering Knowledge-Intensive Reasoning](#paper-4302)
13. [Accelerated and Stable Convergence with Anchored Generalized Optimistic Method](#paper-4303)
14. [Neural Quantum States in Mixed Precision](#paper-4304)
15. [WaveSSM: Multiscale State-Space Models for Non-stationary Signal Attention](#paper-4305)
16. [Decision Tree Learning on Product Spaces](#paper-4306)
17. [Distilling Linearized Behavior into Non-linear Fine-Tuning for Effective Task Arithmetic](#paper-4307)
18. [Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?](#paper-4308)
19. [Decoupling Variance and Scale-Invariant Updates in Adaptive Gradient Descent for Unified Vector and Matrix Optimization](#paper-4309)
20. [EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors](#paper-4310)
21. [Expanding the Capabilities of Reinforcement Learning via Text Feedback](#paper-4311)
22. [When Diffusion Language Models Hesitate: Detecting and Correcting Visual Hallucinations via Confidence Fluctuation](#paper-4312)
23. [EnsembleVLA: Ensemble Learning for Vision-Language Action Models](#paper-4313)
24. [RNA-FM: Flow-Matching Generative Model for Genome-wide RNA-Seq Prediction](#paper-4314)
25. [Estimating the Empowerment of Language Model Agents](#paper-4315)
26. [LoBCD-GW: A Fast and Data-Dependent Algorithm for Computing Gromov-Wasserstein Distance via Localized Block Coordinate Descent](#paper-4316)
27. [Kinematics-Driven Gaussian Shape Deformation for Blurry Monocular Dynamic Scenes](#paper-4317)
28. [R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](#paper-4318)
29. [GO-PRE: Goal-Oriented Next-Best-View Selection via Predictive Rendering Entropy for Active 3D Reconstruction](#paper-4319)
30. [The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting](#paper-4320)
31. [MIMO-LP: A Multi-Input Multi-Output Framework for Subgraph-based Link Prediction](#paper-4321)
32. [MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](#paper-4322)
33. [Expo-GS: Exposure-Aware Signed Distance Function in Gaussian Splatting for High Dynamic Range](#paper-4323)

---


<a id="paper-4291"></a>

## Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver

### Images

![00d1e192d04d9363b51f92425e96cfd1e406f11dac1788a678fded1e6b40604e.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/00d1e192d04d9363b51f92425e96cfd1e406f11dac1788a678fded1e6b40604e.jpg)

![44cbe28cd116f299da47db0d54574e502cc025ef25ee89ecf59fc0d7eba24228.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/44cbe28cd116f299da47db0d54574e502cc025ef25ee89ecf59fc0d7eba24228.jpg)

![6475131c95ee182fa4c5dd805f919a4e395bb8660d679d140ae7d730233bb360.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/6475131c95ee182fa4c5dd805f919a4e395bb8660d679d140ae7d730233bb360.jpg)

![67c30222366be74704a7825d46386ceba0a57dc029a95bf2f41cce4483b11848.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/67c30222366be74704a7825d46386ceba0a57dc029a95bf2f41cce4483b11848.jpg)

![6daba154ce7da7722de6100c8a1dfec6d027eaed1fc8ef5e722f1ceb3606a36d.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/6daba154ce7da7722de6100c8a1dfec6d027eaed1fc8ef5e722f1ceb3606a36d.jpg)

![76648469e8f6aabdbf112305e90f570b44d1ba6a7b0bea42f6c34a38e375f144.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/76648469e8f6aabdbf112305e90f570b44d1ba6a7b0bea42f6c34a38e375f144.jpg)

![945e472f414a0be016d74af38a7db583171456cf8f060033b7c066cf306b7e79.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/945e472f414a0be016d74af38a7db583171456cf8f060033b7c066cf306b7e79.jpg)

![a036fa0a2b2d92bffeae75e8f0d5f243dab0d6f4be1ec2fb6837438663ade5da.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/a036fa0a2b2d92bffeae75e8f0d5f243dab0d6f4be1ec2fb6837438663ade5da.jpg)

### Tables

![050375c0b331156bcf7e97f09131320a69e79e657396c82b0a1c6cac2a2e46ba.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/050375c0b331156bcf7e97f09131320a69e79e657396c82b0a1c6cac2a2e46ba.jpg)

![09dc70edadfeaa5bf1fe019f541229ea72b976def329ad4157d27679ee321462.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/09dc70edadfeaa5bf1fe019f541229ea72b976def329ad4157d27679ee321462.jpg)

![396a45c7584b74314b83123d90beca3be55f1d33f3041c409e856fbf84597875.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/396a45c7584b74314b83123d90beca3be55f1d33f3041c409e856fbf84597875.jpg)

![430f6bcaf93d9828c5c821bf64bf6be3ed7fa3d69edfe092b3a5186d92fba0e9.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/430f6bcaf93d9828c5c821bf64bf6be3ed7fa3d69edfe092b3a5186d92fba0e9.jpg)

![6baa4354547c6334ea283edb01f6966880107b8f39aff90679d69436c5f1f686.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/6baa4354547c6334ea283edb01f6966880107b8f39aff90679d69436c5f1f686.jpg)

![7f09726ef2a4cb767beec6febc2c05e5e41ec3e34a5ae7ffc5eecf3937be8de9.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/7f09726ef2a4cb767beec6febc2c05e5e41ec3e34a5ae7ffc5eecf3937be8de9.jpg)

![c0756a8fe9c7a7fc7de4eeb3beca6fc6820d05c673aab4b090f65a72690d70c7.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/c0756a8fe9c7a7fc7de4eeb3beca6fc6820d05c673aab4b090f65a72690d70c7.jpg)

![c09cc9daf69f8fbb0afc5ae67edc933784b83895b4c491a6c158a9b4bb7efbe7.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/c09cc9daf69f8fbb0afc5ae67edc933784b83895b4c491a6c158a9b4bb7efbe7.jpg)

## Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver


### Images

![062f1b1be30347d3c9ec3e9ed1330dbd2b95cd85ccf23d9e1b6b88f1d5542809.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/062f1b1be30347d3c9ec3e9ed1330dbd2b95cd85ccf23d9e1b6b88f1d5542809.jpg)

![0ac66ded8f04691e1e4c571b4f4e67ffb5d1b6d09a9b80d93c39a2a3d4704700.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/0ac66ded8f04691e1e4c571b4f4e67ffb5d1b6d09a9b80d93c39a2a3d4704700.jpg)

![26546e8d8a375380818d48afd5b92a609d3fe1f49db757fdb71739346b74120c.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/26546e8d8a375380818d48afd5b92a609d3fe1f49db757fdb71739346b74120c.jpg)

![4f77464bf70073ba0b0c860fe49fb32ff2c8bb79434d675a4cda3beb56e281e7.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/4f77464bf70073ba0b0c860fe49fb32ff2c8bb79434d675a4cda3beb56e281e7.jpg)

![8d2b66ea4a3c9d185cc1289381e0f19aa5cb1f4ff73b007f857771474b8aec85.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/8d2b66ea4a3c9d185cc1289381e0f19aa5cb1f4ff73b007f857771474b8aec85.jpg)

![cedacf83c9e7027aac98e53663d888fe167ac402ee326f23fc355fa2b8e79670.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/cedacf83c9e7027aac98e53663d888fe167ac402ee326f23fc355fa2b8e79670.jpg)

![d6d33541ad94680159d0bb7219391b7780e22c06bedaf33d7e36b6d6b96f47b6.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/d6d33541ad94680159d0bb7219391b7780e22c06bedaf33d7e36b6d6b96f47b6.jpg)

![dc5c78ffdf0a23d5328fb42a20c70cc5fca6e3c39584ee606418b0a30758ea18.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/dc5c78ffdf0a23d5328fb42a20c70cc5fca6e3c39584ee606418b0a30758ea18.jpg)

![e8f8e1a7d0481621551ac8d6752cd782489d89ad8c4b890f583ef6ae8f26e778.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/e8f8e1a7d0481621551ac8d6752cd782489d89ad8c4b890f583ef6ae8f26e778.jpg)

![eec2ecdb55c2fa547347496957ea282f9367c96eb1465b0072ffe67bded960d7.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/eec2ecdb55c2fa547347496957ea282f9367c96eb1465b0072ffe67bded960d7.jpg)

![fe690a04135789e6cb59db3c23a446405458d2f15912b57c23541e573aa36904.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/images/fe690a04135789e6cb59db3c23a446405458d2f15912b57c23541e573aa36904.jpg)

### Tables

![0e669a24034ce1b6a5cd10d9cf0c11c18a2649fa4f09198dba75b0b988a46551.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/0e669a24034ce1b6a5cd10d9cf0c11c18a2649fa4f09198dba75b0b988a46551.jpg)

![2eb995d830dbc8034b84ab1646dceda26d6e80dcaaa3f219194d121c64119e22.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/2eb995d830dbc8034b84ab1646dceda26d6e80dcaaa3f219194d121c64119e22.jpg)

![3bb9a29f87d516c09e58d1013527c6b5df2e3846c2986baaa37b10d64967fb62.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/3bb9a29f87d516c09e58d1013527c6b5df2e3846c2986baaa37b10d64967fb62.jpg)

![3fa0eeae954279421884161cfb552a1e77151b69570f3a61276fb7025eb261b4.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/3fa0eeae954279421884161cfb552a1e77151b69570f3a61276fb7025eb261b4.jpg)

![3ffd611dce35f5f2022bf33382a62d548292591f845eba726ba2b173fb1a4227.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/3ffd611dce35f5f2022bf33382a62d548292591f845eba726ba2b173fb1a4227.jpg)

![4ddabb4503dc20cd0a5fdcb716db0c68cd0f98b1761f312adefd73e7612c9be0.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/4ddabb4503dc20cd0a5fdcb716db0c68cd0f98b1761f312adefd73e7612c9be0.jpg)

![9bc146bb7c7636b6c46e3d761e17c22bf0920790fc3e4358a5abe2eee2262030.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/9bc146bb7c7636b6c46e3d761e17c22bf0920790fc3e4358a5abe2eee2262030.jpg)

![dc36cbcdb71f4d6043c7b7dd344c9ea402b146e447b9fbc10be1ff0ff6bc1e5e.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/dc36cbcdb71f4d6043c7b7dd344c9ea402b146e447b9fbc10be1ff0ff6bc1e5e.jpg)

![e70f6779f699e9ba4b6a37c7be4fdb628827c8bc62505e31796dff467c04bd24.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/e70f6779f699e9ba4b6a37c7be4fdb628827c8bc62505e31796dff467c04bd24.jpg)

![ea050b8d21490d185deace2bee400d3a3b26ca21bef47be6b631f2354dd5701d.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/ea050b8d21490d185deace2bee400d3a3b26ca21bef47be6b631f2354dd5701d.jpg)

![eb2b5ce3b9f538ea496f3f04ea174f321c7a0c0fb60f050a31305fa04a59b086.jpg](../icml_results/4335_Neural-HSS_%20Hierarchical%20Semi-Separable%20Neural%20PDE%20Solver/tables/eb2b5ce3b9f538ea496f3f04ea174f321c7a0c0fb60f050a31305fa04a59b086.jpg)

<a id="paper-4292"></a>

## SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives


### Images

![0464e649c96de58630b2bb603a1c0c93222ff04441def6c5f280fc01a94b9738.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/0464e649c96de58630b2bb603a1c0c93222ff04441def6c5f280fc01a94b9738.jpg)

![05550c2f0c5c5170b5fb36f190e08387d5693cc6f6569f45f9c19af59ad46359.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/05550c2f0c5c5170b5fb36f190e08387d5693cc6f6569f45f9c19af59ad46359.jpg)

![0f96f2cd2cc2804459ca010b7ec5f263a4b95eebee9ba93a03ab646de0c9b4ed.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/0f96f2cd2cc2804459ca010b7ec5f263a4b95eebee9ba93a03ab646de0c9b4ed.jpg)

![20a527fb3bd577ffa0a171a036c7fb968368fd5efa6610a1dd8e9e4a41cbdedb.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/20a527fb3bd577ffa0a171a036c7fb968368fd5efa6610a1dd8e9e4a41cbdedb.jpg)

![3409cf585867e1e08250f679a94919e52521fce36c2254897481cabbeba85c72.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/3409cf585867e1e08250f679a94919e52521fce36c2254897481cabbeba85c72.jpg)

![3d367f7ee009eb58654d3fcbd369b533122edc582d490124c74c31b29c45a4d0.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/3d367f7ee009eb58654d3fcbd369b533122edc582d490124c74c31b29c45a4d0.jpg)

![5c1413aee57ea779149262bdcd7c656f50eb21d4cc273bef862844ca4bf15e13.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/5c1413aee57ea779149262bdcd7c656f50eb21d4cc273bef862844ca4bf15e13.jpg)

![6cf3ef6366c9f4247663538e72d576ec704546d3c2a026ccef9b8385c73145a0.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/6cf3ef6366c9f4247663538e72d576ec704546d3c2a026ccef9b8385c73145a0.jpg)

![70c866cf00e85189f54fe71ec7295d33f89f73da3314bba9e39e7b9cfcae6a9d.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/70c866cf00e85189f54fe71ec7295d33f89f73da3314bba9e39e7b9cfcae6a9d.jpg)

![852700798f3d944f12917b5b9ec39b3e5ded0582917e784594fa3d28deabf188.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/852700798f3d944f12917b5b9ec39b3e5ded0582917e784594fa3d28deabf188.jpg)

![8c224c8d6029db023cb441064c2d7f46a273d500ca51ccc83f2b84adc2e933fd.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/8c224c8d6029db023cb441064c2d7f46a273d500ca51ccc83f2b84adc2e933fd.jpg)

![a0cb82278a9bb020cc3060c77002ae8e0bbf82d4ddeb0a4a18ccaca8ba7aa2bc.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/a0cb82278a9bb020cc3060c77002ae8e0bbf82d4ddeb0a4a18ccaca8ba7aa2bc.jpg)

![b10a994fb577cec670644a3c73508169693065475e01b92c1099d8d0126cb0e2.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/b10a994fb577cec670644a3c73508169693065475e01b92c1099d8d0126cb0e2.jpg)

![d828fd36f039c311145285433dfefb3164643c851f7a918bb79ce62353a1a680.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/d828fd36f039c311145285433dfefb3164643c851f7a918bb79ce62353a1a680.jpg)

![dad4b648e010e696cbc68c2476b1ab868e2e2af686db100b2a8b4a17de49af61.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/dad4b648e010e696cbc68c2476b1ab868e2e2af686db100b2a8b4a17de49af61.jpg)

![eadd5632b7d5b52a67bd300d183602ed43cefbca8635c4b7d9d30df377e65edf.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/eadd5632b7d5b52a67bd300d183602ed43cefbca8635c4b7d9d30df377e65edf.jpg)

![faf3be8843e31f2b6d49556479f879ecd685a28c051cb88801a08d43fbdd09e1.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/images/faf3be8843e31f2b6d49556479f879ecd685a28c051cb88801a08d43fbdd09e1.jpg)

### Tables

![0025955e691d389471aa3e1914c91dea9182da495e7c6fe6022b151ff73b73c3.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/0025955e691d389471aa3e1914c91dea9182da495e7c6fe6022b151ff73b73c3.jpg)

![0582cafa9597f7b3be13b134e108fec05ce06e4fa768d1eac617b3e90b8345ae.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/0582cafa9597f7b3be13b134e108fec05ce06e4fa768d1eac617b3e90b8345ae.jpg)

![0c7d762c19c1e9f7e5fb315ed3f206bf45676f6c842f04ffc1fe0cd591f8f782.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/0c7d762c19c1e9f7e5fb315ed3f206bf45676f6c842f04ffc1fe0cd591f8f782.jpg)

![0d71ff6781f5f4a84a8e06764f41f7ee7aae9c2592e2ee611140f65eddaff6ea.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/0d71ff6781f5f4a84a8e06764f41f7ee7aae9c2592e2ee611140f65eddaff6ea.jpg)

![1092a778d817d6f2ef0cf5f7130c4371b10d5e0036920bb7ac74f74a97e961a0.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/1092a778d817d6f2ef0cf5f7130c4371b10d5e0036920bb7ac74f74a97e961a0.jpg)

![121b8b5253262cef79423b0565250c1d0e6c6b2a5b9e71584ed8220dac193a0d.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/121b8b5253262cef79423b0565250c1d0e6c6b2a5b9e71584ed8220dac193a0d.jpg)

![15f12b0e2dfbf746ee5be49b77f902e4ab241e4d515698aebf2920fe57615705.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/15f12b0e2dfbf746ee5be49b77f902e4ab241e4d515698aebf2920fe57615705.jpg)

![1bd971d8ff73c4d8c1767cdea50900ba7399dfe3fe0472afc5019c30ff52de04.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/1bd971d8ff73c4d8c1767cdea50900ba7399dfe3fe0472afc5019c30ff52de04.jpg)

![1d6e79cd0188802a2112ac069ebe814b3ba13adeae66fd2ebb5e4ef5e55a927a.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/1d6e79cd0188802a2112ac069ebe814b3ba13adeae66fd2ebb5e4ef5e55a927a.jpg)

![1db6099bac9970429cd4b1b756fd1aab73fc7cda4c17523727b9cf00c8d72a19.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/1db6099bac9970429cd4b1b756fd1aab73fc7cda4c17523727b9cf00c8d72a19.jpg)

![1e5eae2c537a1e434e89fdf2812d1ae1091a20840249e99dc8c009117f676f7a.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/1e5eae2c537a1e434e89fdf2812d1ae1091a20840249e99dc8c009117f676f7a.jpg)

![21e6608629241f93e8779d8413410538790927668e924ed4d0e131669f06b14d.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/21e6608629241f93e8779d8413410538790927668e924ed4d0e131669f06b14d.jpg)

![2cfbf8fd882df58dcf9d988e30402e7bcf488250db0b33951e341b969d5c4d86.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/2cfbf8fd882df58dcf9d988e30402e7bcf488250db0b33951e341b969d5c4d86.jpg)

![3605172886d9d8a003a85b1b59f2da8205c5c50ba012398642b1d67f6ad5fc82.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/3605172886d9d8a003a85b1b59f2da8205c5c50ba012398642b1d67f6ad5fc82.jpg)

![36d1e112c8f41e6b952971a77dcbaed1fcb5cdb89ecdee7eb13985aefc5c12e5.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/36d1e112c8f41e6b952971a77dcbaed1fcb5cdb89ecdee7eb13985aefc5c12e5.jpg)

![3aedf984351260b8450a317070d8e63c198852f7403ec33764847c38b74fad8a.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/3aedf984351260b8450a317070d8e63c198852f7403ec33764847c38b74fad8a.jpg)

![46915a8e85da143289ef5da68011b4aebd3f3d827d4e89a3cbec73a7500a8e01.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/46915a8e85da143289ef5da68011b4aebd3f3d827d4e89a3cbec73a7500a8e01.jpg)

![4891e2f7005fcfa32a9075a917ad7816cacb45a2465adf467905f0abdcc31a1f.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/4891e2f7005fcfa32a9075a917ad7816cacb45a2465adf467905f0abdcc31a1f.jpg)

![4fd9e8b49d7b4d86d922903457f2f3eb727b0fdb987fa3041a99e5116137235b.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/4fd9e8b49d7b4d86d922903457f2f3eb727b0fdb987fa3041a99e5116137235b.jpg)

![5240143b46a23c0a4f2a65ca8c12a892357ba316275f27278a72404a87066296.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/5240143b46a23c0a4f2a65ca8c12a892357ba316275f27278a72404a87066296.jpg)

![5320a9a8a2c5bd117b333d3b0410774658189f53815a678433a1d5c7d8d85b7c.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/5320a9a8a2c5bd117b333d3b0410774658189f53815a678433a1d5c7d8d85b7c.jpg)

![54284466ba1d9c5ab2660c19a55718a003f0eeb91294541d5fd43d3f411319de.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/54284466ba1d9c5ab2660c19a55718a003f0eeb91294541d5fd43d3f411319de.jpg)

![6486fed8aa53b5d47487bd9b3050d7d0d759b4de75833d1b4aabc2b1b6b5171c.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/6486fed8aa53b5d47487bd9b3050d7d0d759b4de75833d1b4aabc2b1b6b5171c.jpg)

![79136c43b11442fa1dc423d21d95bf43f69f2eea1e9e98b6f57ec2241a4ef382.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/79136c43b11442fa1dc423d21d95bf43f69f2eea1e9e98b6f57ec2241a4ef382.jpg)

![8296a7bf175f86addda3551b688741859c388aa00d0e4cb5094c8438501f9cee.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/8296a7bf175f86addda3551b688741859c388aa00d0e4cb5094c8438501f9cee.jpg)

![83dab1d4cb8034b14408c913973535035d8cb5283b4874aee4b06a89cf2b33dc.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/83dab1d4cb8034b14408c913973535035d8cb5283b4874aee4b06a89cf2b33dc.jpg)

![87f6a377819582bb614a85f31d9710706730833e0d65e303a5571d04bbdf023f.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/87f6a377819582bb614a85f31d9710706730833e0d65e303a5571d04bbdf023f.jpg)

![889c367eb0fb96b91448cb0300a1e2e829414d52e622546fdfcea5fd8407c82f.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/889c367eb0fb96b91448cb0300a1e2e829414d52e622546fdfcea5fd8407c82f.jpg)

![8c6c89e2bc355b0fafe2a33ebd1c082ba9f36d948cee5ed8dd3cdc4df9db82f2.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/8c6c89e2bc355b0fafe2a33ebd1c082ba9f36d948cee5ed8dd3cdc4df9db82f2.jpg)

![8fb8046b828cacc2d8e18b94b4f31664df010b0af2cd48f5b8621601a2bff288.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/8fb8046b828cacc2d8e18b94b4f31664df010b0af2cd48f5b8621601a2bff288.jpg)

![9082a623e1f1b3958c09b9221835f97e68b5da9900c52535b7964198ace4d12b.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/9082a623e1f1b3958c09b9221835f97e68b5da9900c52535b7964198ace4d12b.jpg)

![9253272cd4089f777e00168b5974665da1a06d8fa4dc54998359d72693c6c880.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/9253272cd4089f777e00168b5974665da1a06d8fa4dc54998359d72693c6c880.jpg)

![a5a46cdd13021dcbb60fd7828fa30e455c29931b84b7d86855ff154b36e85232.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/a5a46cdd13021dcbb60fd7828fa30e455c29931b84b7d86855ff154b36e85232.jpg)

![b1b164f6101a43639250188410e602e85451daee06e24bb00f1ceffc5247dcb6.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b1b164f6101a43639250188410e602e85451daee06e24bb00f1ceffc5247dcb6.jpg)

![b22c7936df5efa5a1f81d81f169adf888d8848dd695bdf3d567ef8c5079ef362.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b22c7936df5efa5a1f81d81f169adf888d8848dd695bdf3d567ef8c5079ef362.jpg)

![b3932e71363d5c136066d3ed97521216ed5f2dd6d26aab6f4912f4a17bae29e6.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b3932e71363d5c136066d3ed97521216ed5f2dd6d26aab6f4912f4a17bae29e6.jpg)

![b6916615f415cae5c9c0d074a70fe0450de127bd0b89ddf4b8a6b2e728fc14e2.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b6916615f415cae5c9c0d074a70fe0450de127bd0b89ddf4b8a6b2e728fc14e2.jpg)

![b93807db7bb75c84b41ff39606a57dd45ec1764b3be1cb38b7fd990cb3719239.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b93807db7bb75c84b41ff39606a57dd45ec1764b3be1cb38b7fd990cb3719239.jpg)

![b9ba7a8ffc041d6f4901f089eb3d5e30c1838af7809b1551304ee4f28f30baf6.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/b9ba7a8ffc041d6f4901f089eb3d5e30c1838af7809b1551304ee4f28f30baf6.jpg)

![c1e1bc69303f70ba52af28c9d8afb1e13b0435b515d417254b25dd6ad033d4f4.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/c1e1bc69303f70ba52af28c9d8afb1e13b0435b515d417254b25dd6ad033d4f4.jpg)

![c60fcfa864e1ba6a4bf4883ab1e8826f517c80a680fd8652954186dd3dce24e7.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/c60fcfa864e1ba6a4bf4883ab1e8826f517c80a680fd8652954186dd3dce24e7.jpg)

![c66a7d1f3e3cb557eef327f90626dfad45c8a773ef42060e0bda0328d5b258f5.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/c66a7d1f3e3cb557eef327f90626dfad45c8a773ef42060e0bda0328d5b258f5.jpg)

![cd9e974d335f242b6d836568d4c464e9f34be76e004b63eecd98611687c9daef.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/cd9e974d335f242b6d836568d4c464e9f34be76e004b63eecd98611687c9daef.jpg)

![d31d5849bcb8ad7d9ae7aafb8f96f83a0c9bdb6754955fb4629f4d658b31b6e7.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/d31d5849bcb8ad7d9ae7aafb8f96f83a0c9bdb6754955fb4629f4d658b31b6e7.jpg)

![d320857578317be5984e6628c5f39e3bec285c33062609728d673dbe0c0f776b.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/d320857578317be5984e6628c5f39e3bec285c33062609728d673dbe0c0f776b.jpg)

![d3a7b20fe9da056b7203f1bf81791120f095f9df9b47a05596c7e45850b9367e.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/d3a7b20fe9da056b7203f1bf81791120f095f9df9b47a05596c7e45850b9367e.jpg)

![d9915ce111ab341b6b720c9ae42f9cf986c9ad8cf13582c5bd269352871d0ba9.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/d9915ce111ab341b6b720c9ae42f9cf986c9ad8cf13582c5bd269352871d0ba9.jpg)

![e5a77738f77826d4a2f67f82c1364cf4ee7f90835658853eeaec180cf72ff87a.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/e5a77738f77826d4a2f67f82c1364cf4ee7f90835658853eeaec180cf72ff87a.jpg)

![ec03fdc4c5ea6322b0769b7e42101f0840b8928ad3df00798fba670443080980.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/ec03fdc4c5ea6322b0769b7e42101f0840b8928ad3df00798fba670443080980.jpg)

![f5184a8e1fcc88394f9313d4d0c07e07b30e5f3baa4ed84b42c87c40cb6bcfce.jpg](../icml_results/4336_SteeringSafety_%20Benchmarking%20Representation%20Steering%20in%20LLMs%20Across%20Safety%20Perspectives/tables/f5184a8e1fcc88394f9313d4d0c07e07b30e5f3baa4ed84b42c87c40cb6bcfce.jpg)

<a id="paper-4293"></a>

## Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control


### Images

![15aa3847137640f57f76dc791c278563ea338d392b4c656ed04a0e47401e1cb4.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/15aa3847137640f57f76dc791c278563ea338d392b4c656ed04a0e47401e1cb4.jpg)

![245c7e1c87897ffe1735984e3e9b31539f838bbbc05dfbb3100989941129359d.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/245c7e1c87897ffe1735984e3e9b31539f838bbbc05dfbb3100989941129359d.jpg)

![24c323250f7fdc3941cc26002eb53bf3e862d2bcb7355e46089b1a984c0487a6.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/24c323250f7fdc3941cc26002eb53bf3e862d2bcb7355e46089b1a984c0487a6.jpg)

![2bc2f9b9d5306d3b3dde5d31164cc9e7079716da82c466c029116ff96bbcfee9.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/2bc2f9b9d5306d3b3dde5d31164cc9e7079716da82c466c029116ff96bbcfee9.jpg)

![3e49b011d94f2fd5938f14575ec7ce9d57ae31b92e167374df34e549b59c1661.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/3e49b011d94f2fd5938f14575ec7ce9d57ae31b92e167374df34e549b59c1661.jpg)

![457f29bbdccc99c34af93a3ab2ddfc61b2ed4d0d13875f611a70875421140e5a.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/457f29bbdccc99c34af93a3ab2ddfc61b2ed4d0d13875f611a70875421140e5a.jpg)

![4933f2be77b6dd7023700d1c31c225a041f10fc86d34f87a14fd6dcc5a5410f3.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/4933f2be77b6dd7023700d1c31c225a041f10fc86d34f87a14fd6dcc5a5410f3.jpg)

![4e0952a35fe7d488ffb08987416f8ef6bb6b94c21f9841a147d041fb369b08be.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/4e0952a35fe7d488ffb08987416f8ef6bb6b94c21f9841a147d041fb369b08be.jpg)

![53f7a0aa72f97837e7f4ee7018b0e1f096f1fbfe090be445b98f25aa40b9da90.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/53f7a0aa72f97837e7f4ee7018b0e1f096f1fbfe090be445b98f25aa40b9da90.jpg)

![58fb50dbb8e438fa009a2ac3603847d938b40584f8c40ce29183c3c9a5b422e6.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/58fb50dbb8e438fa009a2ac3603847d938b40584f8c40ce29183c3c9a5b422e6.jpg)

![63f589f0634e97980e14d866f232cab6163f8a2c9105528d7c6ea76996efd7f3.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/63f589f0634e97980e14d866f232cab6163f8a2c9105528d7c6ea76996efd7f3.jpg)

![664374a3f6343b46242b423ba380e15d39ded3bf70e7e9a041534b3c3d6443bb.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/664374a3f6343b46242b423ba380e15d39ded3bf70e7e9a041534b3c3d6443bb.jpg)

![767dbeef746014e0b9666761bf02e65e874d97511835fe07e2fbba630487c6ce.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/767dbeef746014e0b9666761bf02e65e874d97511835fe07e2fbba630487c6ce.jpg)

![8671b6b3f6d54d2778f10eb50e07b37340a2250e4a87d6396a05d2ec6ef8c485.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/8671b6b3f6d54d2778f10eb50e07b37340a2250e4a87d6396a05d2ec6ef8c485.jpg)

![9125691378dd219021e846cec0613240d298c65991e46e1a66459b22a42fd379.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/9125691378dd219021e846cec0613240d298c65991e46e1a66459b22a42fd379.jpg)

![9cb3b920f3db2be42a029b9cb782111a4db9cc0484b4748080bcb6faad190082.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/9cb3b920f3db2be42a029b9cb782111a4db9cc0484b4748080bcb6faad190082.jpg)

![a4373301c11a8562583665a79fd265148cbff6f75756bc3e39b69d3abcef786f.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/a4373301c11a8562583665a79fd265148cbff6f75756bc3e39b69d3abcef786f.jpg)

![affb54764db3829b84f78fc723424f8bf3b2ca679635be4b4eb0198f47ba3b65.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/affb54764db3829b84f78fc723424f8bf3b2ca679635be4b4eb0198f47ba3b65.jpg)

![b5b7c1b1f282062c00b0352f4be6cefa271fd64b330c5dec5e08217982c01e00.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/b5b7c1b1f282062c00b0352f4be6cefa271fd64b330c5dec5e08217982c01e00.jpg)

![bc302ffbefa097f146f3a4278e69d1843df72b114cb98d93877bfeb2fa49ce86.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/bc302ffbefa097f146f3a4278e69d1843df72b114cb98d93877bfeb2fa49ce86.jpg)

![d039404dc33328b69f400e5ae88dd3fa2ff540febc6bb5af2dbd6f1e765842f7.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/d039404dc33328b69f400e5ae88dd3fa2ff540febc6bb5af2dbd6f1e765842f7.jpg)

![d131e35a6d8789c26ed5938e8ffca09fcd078fbb51f463b7435cee4e017a4de0.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/d131e35a6d8789c26ed5938e8ffca09fcd078fbb51f463b7435cee4e017a4de0.jpg)

![d893798ca5913d1acd9c179163133e8675819fa0a2c052528b86b9ad52e52cbc.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/d893798ca5913d1acd9c179163133e8675819fa0a2c052528b86b9ad52e52cbc.jpg)

![deb625385ba246e6d097821d3a1a287e1b5e3b92f037a0025c5012642a59d35d.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/deb625385ba246e6d097821d3a1a287e1b5e3b92f037a0025c5012642a59d35d.jpg)

![fa75bad1e4a184059aed3ebf1e294ed6e7d084a0cd9be375116e0636262a8c6f.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/images/fa75bad1e4a184059aed3ebf1e294ed6e7d084a0cd9be375116e0636262a8c6f.jpg)

### Tables

![101744deb07aff02c50da4b0bf8264c73e46da6d5adfde2bd1db1ee44e04b93c.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/101744deb07aff02c50da4b0bf8264c73e46da6d5adfde2bd1db1ee44e04b93c.jpg)

![44da127ff9ba824cff72924f832f2c514b52ee6ecf757bdd109f0e69667fe8f8.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/44da127ff9ba824cff72924f832f2c514b52ee6ecf757bdd109f0e69667fe8f8.jpg)

![4ccdd45c12f1d6773a39ed4bccdc0c3ad0df64b7333941244af6f48b134021e3.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/4ccdd45c12f1d6773a39ed4bccdc0c3ad0df64b7333941244af6f48b134021e3.jpg)

![6131bd630d8f0761f17946d2d8d04d34c3901c826ea91658921b429ad5380631.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/6131bd630d8f0761f17946d2d8d04d34c3901c826ea91658921b429ad5380631.jpg)

![63177c3eeaa3c206adb98f19f9d797ed3822c1a00de3adcb1693befd9ea23707.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/63177c3eeaa3c206adb98f19f9d797ed3822c1a00de3adcb1693befd9ea23707.jpg)

![67d8699b8a693f92a33c3cb7c407d0ddc04e90a72dbe54d25f8290a32f1df807.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/67d8699b8a693f92a33c3cb7c407d0ddc04e90a72dbe54d25f8290a32f1df807.jpg)

![690bb5f60b3263770aeae7d6027b4df4e5d3475424f5d2f01dedc29d79c4d264.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/690bb5f60b3263770aeae7d6027b4df4e5d3475424f5d2f01dedc29d79c4d264.jpg)

![72b061a5d6e7ef307d5bc64e9dbdf20a382473f2f24606e1d0db4da91225b43e.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/72b061a5d6e7ef307d5bc64e9dbdf20a382473f2f24606e1d0db4da91225b43e.jpg)

![812cb65f66b2d07e98c6197ac6e420256d0761bb7daf4e398cf711cdac745c34.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/812cb65f66b2d07e98c6197ac6e420256d0761bb7daf4e398cf711cdac745c34.jpg)

![8ad19d856176c8ecb9fa34570962e3d690bdfae5ce622f9051f706a2edd33657.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/8ad19d856176c8ecb9fa34570962e3d690bdfae5ce622f9051f706a2edd33657.jpg)

![8aeeb57681c57c2e2639550d21c69cce8ab6ce8590f46f58350c7983b4748212.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/8aeeb57681c57c2e2639550d21c69cce8ab6ce8590f46f58350c7983b4748212.jpg)

![a5e887037bcb65c6a1ab438397795077a996e0da80cab20e985153dbba2d328e.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/a5e887037bcb65c6a1ab438397795077a996e0da80cab20e985153dbba2d328e.jpg)

![ae6baf9e0702810319b4e03e810b73d1941a69ac8429f89b7a06d3b1837b9f04.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/ae6baf9e0702810319b4e03e810b73d1941a69ac8429f89b7a06d3b1837b9f04.jpg)

![c5063a88df8bc486d65428226d8c50e5680d1e50deab7fdff4449fa790907d0a.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/c5063a88df8bc486d65428226d8c50e5680d1e50deab7fdff4449fa790907d0a.jpg)

![dfb17ab0c1c771dbb9493a06d3c114d4f49d92e9c4360ccd2a041f2ecdd8742f.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/dfb17ab0c1c771dbb9493a06d3c114d4f49d92e9c4360ccd2a041f2ecdd8742f.jpg)

![f4a048aba4e5a3c8eb4d9167a313a0e310ef2f4c1d67d22962fe8cd6b16e39a5.jpg](../icml_results/4337_Local%20Linearity%20of%20LLMs%20Enables%20Activation%20Steering%20via%20Model-Based%20Linear%20Optimal%20Control/tables/f4a048aba4e5a3c8eb4d9167a313a0e310ef2f4c1d67d22962fe8cd6b16e39a5.jpg)

<a id="paper-4294"></a>

## Variational Learning of Disentangled Representations


### Images

![000d04d305300973ed06b6d40ad00fb0fc3428808a2ce899165f823401981a46.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/000d04d305300973ed06b6d40ad00fb0fc3428808a2ce899165f823401981a46.jpg)

![0fd443bf5ca3b44fa97e6d73cee441e46443e1ec370d29cf6b037239df142e18.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/0fd443bf5ca3b44fa97e6d73cee441e46443e1ec370d29cf6b037239df142e18.jpg)

![4c7cfb426dbf91853c31d209341b1e3aabee2e5ff4c4b0adcc05d3dc0950ad7a.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/4c7cfb426dbf91853c31d209341b1e3aabee2e5ff4c4b0adcc05d3dc0950ad7a.jpg)

![570a1af65766aab24143e5b8370dc77aed7fc0884e783039f36dd869f60becba.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/570a1af65766aab24143e5b8370dc77aed7fc0884e783039f36dd869f60becba.jpg)

![58c2e134147d284ebf7ba356f8b0c5d4db2cc8f88f2d5c8cd0d4b13ccbcd27e1.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/58c2e134147d284ebf7ba356f8b0c5d4db2cc8f88f2d5c8cd0d4b13ccbcd27e1.jpg)

![5a20aef425fa67b3a95c692a86f8d1ca15d52c8fe8a2eb0e705ccf4bd5fe8b67.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/5a20aef425fa67b3a95c692a86f8d1ca15d52c8fe8a2eb0e705ccf4bd5fe8b67.jpg)

![752b629025922af80d5941a2c7e5abf004209e20aa6db8032d5042a75ec6571a.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/752b629025922af80d5941a2c7e5abf004209e20aa6db8032d5042a75ec6571a.jpg)

![7e52cabdf656a6634203f1c08620b21a31f58c2a2c5409d4187dbeca035f7436.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/7e52cabdf656a6634203f1c08620b21a31f58c2a2c5409d4187dbeca035f7436.jpg)

![89af699a391f364da1bf43ccadd7d1d46c2677f3a6e884e81fbda4be37bdbc66.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/89af699a391f364da1bf43ccadd7d1d46c2677f3a6e884e81fbda4be37bdbc66.jpg)

![a1eeef7b4449d4091922cdac9d9856e75206becc689615c478c9854e66f93e03.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/a1eeef7b4449d4091922cdac9d9856e75206becc689615c478c9854e66f93e03.jpg)

![b204f399b2fd671800d724f83966ffcb54da45887f27e7b3ab89fac5da7fd5b9.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/b204f399b2fd671800d724f83966ffcb54da45887f27e7b3ab89fac5da7fd5b9.jpg)

![e6f1bd7bcd6264b9263bd942c4730a9d8c407fecbe42184410c0f429e35fc17a.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/e6f1bd7bcd6264b9263bd942c4730a9d8c407fecbe42184410c0f429e35fc17a.jpg)

![e70ca5cdf85f6dc6c0a1b47949c678c38dc0d4555f2e46eea109b857084f69b2.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/e70ca5cdf85f6dc6c0a1b47949c678c38dc0d4555f2e46eea109b857084f69b2.jpg)

![f36a6a5284cfe98da7ec971de5990e62cfbe225393b43a80703758ab9051e5a8.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/images/f36a6a5284cfe98da7ec971de5990e62cfbe225393b43a80703758ab9051e5a8.jpg)

### Tables

![0104582b6079be820ee223c900a7d6c7c28e1cbdb9e2368fbaf763465a59eb88.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/0104582b6079be820ee223c900a7d6c7c28e1cbdb9e2368fbaf763465a59eb88.jpg)

![061fb111989fcd59f7a019444342bf9e1ab51b418639e0ed703a7b518fb8f1eb.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/061fb111989fcd59f7a019444342bf9e1ab51b418639e0ed703a7b518fb8f1eb.jpg)

![19a367a8cf4dd6353aef70ba4303d405c57c41e48c1542c60618db94a8a76a2c.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/19a367a8cf4dd6353aef70ba4303d405c57c41e48c1542c60618db94a8a76a2c.jpg)

![2c8550e69facf5d5b527f8b185bbe3a58b7a0a4ba03782076669c207a11b5b8a.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/2c8550e69facf5d5b527f8b185bbe3a58b7a0a4ba03782076669c207a11b5b8a.jpg)

![318228b683cdece1440c0d3c38c75d9e5d127e029f71d432958d92b0ba9d4e55.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/318228b683cdece1440c0d3c38c75d9e5d127e029f71d432958d92b0ba9d4e55.jpg)

![35c2c4a35a96be53a2922ceab3b86a4beb4d525f801c64ea23438b62e42f3278.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/35c2c4a35a96be53a2922ceab3b86a4beb4d525f801c64ea23438b62e42f3278.jpg)

![36d57264eb2c211d7fabb6400c4fa9b7521e38aa2bbb60066f8c1e6b354e4e81.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/36d57264eb2c211d7fabb6400c4fa9b7521e38aa2bbb60066f8c1e6b354e4e81.jpg)

![390155ce382d7cec14ad0355f81edc8a40e21c58bd2d941bdccd0f2dfd48373a.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/390155ce382d7cec14ad0355f81edc8a40e21c58bd2d941bdccd0f2dfd48373a.jpg)

![485758a2cc6d074ff852ba439eeafca205c313a7924e105e596a0cd7b64454bc.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/485758a2cc6d074ff852ba439eeafca205c313a7924e105e596a0cd7b64454bc.jpg)

![517f94f5e16eec3b5ff7a3e1ecb88ade854811b78194ae8860ea3bc5a4ccc294.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/517f94f5e16eec3b5ff7a3e1ecb88ade854811b78194ae8860ea3bc5a4ccc294.jpg)

![558cde131a4375dde58a9429871796e842909fe33c35e5679386afe921ad2081.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/558cde131a4375dde58a9429871796e842909fe33c35e5679386afe921ad2081.jpg)

![55eaddba49efb305d2c26490e093ee806b597c5b9b74ec805e7f23139dbb3f0b.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/55eaddba49efb305d2c26490e093ee806b597c5b9b74ec805e7f23139dbb3f0b.jpg)

![55f26eb2aef073e18223653ea40a7fcdad3620ec6c74c893e3dc50b4bc869273.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/55f26eb2aef073e18223653ea40a7fcdad3620ec6c74c893e3dc50b4bc869273.jpg)

![5eca5e09e7ba102a2f37dab9991cc385376b576dfa5834c49b72da93c6d6b70f.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/5eca5e09e7ba102a2f37dab9991cc385376b576dfa5834c49b72da93c6d6b70f.jpg)

![6806ab7eae7664762f401aa29978fd89b3c2a6026af7b31503461d6541bd0f1b.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/6806ab7eae7664762f401aa29978fd89b3c2a6026af7b31503461d6541bd0f1b.jpg)

![6ae1e49244585381e924ba4b1e19ccf6b0f778babc3189918c3da9e0725e8293.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/6ae1e49244585381e924ba4b1e19ccf6b0f778babc3189918c3da9e0725e8293.jpg)

![82cae783c0750b12a9cff231a817f537f2a1d898106059cbb7e1ac6e5037213c.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/82cae783c0750b12a9cff231a817f537f2a1d898106059cbb7e1ac6e5037213c.jpg)

![854c43dcb0bc1d406920d64b481bfd4e7272046aab583efb26b19a65d3b6f9a8.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/854c43dcb0bc1d406920d64b481bfd4e7272046aab583efb26b19a65d3b6f9a8.jpg)

![8cf53111a811947b43667396bedcf46e1679295a4f8417949cd1c09e318f64b1.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/8cf53111a811947b43667396bedcf46e1679295a4f8417949cd1c09e318f64b1.jpg)

![a03590a416d481c0d455141aa489002ebf387d852e62b0d2d99a82caa349c050.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/a03590a416d481c0d455141aa489002ebf387d852e62b0d2d99a82caa349c050.jpg)

![a3f7472bb60745539b405d9c7f669af27cda61a94872345c6cc531490c668be3.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/a3f7472bb60745539b405d9c7f669af27cda61a94872345c6cc531490c668be3.jpg)

![af67128c382ec08d2ea1ed1979c36eba0c13f5111fd6e48705600d7687f91209.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/af67128c382ec08d2ea1ed1979c36eba0c13f5111fd6e48705600d7687f91209.jpg)

![c98f17fef244e6cd0534694253a154295ce7ab1701ca1abf9f5b3578b0fbf491.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/c98f17fef244e6cd0534694253a154295ce7ab1701ca1abf9f5b3578b0fbf491.jpg)

![cd8136c7d798d133b86379485fa0094a9d0ad7816ceb80db6a0ba47f86e368f3.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/cd8136c7d798d133b86379485fa0094a9d0ad7816ceb80db6a0ba47f86e368f3.jpg)

![d7dcfc8a136daa14d0107a48d8cbcc8c9cb69dfc9dbb0c4c4f6ccdc71eb56eb7.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/d7dcfc8a136daa14d0107a48d8cbcc8c9cb69dfc9dbb0c4c4f6ccdc71eb56eb7.jpg)

![ef341ebd085b26fa6658b121deec390d20ae96c53bbb593d51843b7e65f7bb15.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/ef341ebd085b26fa6658b121deec390d20ae96c53bbb593d51843b7e65f7bb15.jpg)

![f6f5440d20383629a1d452cbb0dd5113511c89c26125b20363ebcc8b635891ae.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/f6f5440d20383629a1d452cbb0dd5113511c89c26125b20363ebcc8b635891ae.jpg)

![f8719780191864ce948bfca18e4555380b31beebb404a33ede990e7eadfc5c45.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/f8719780191864ce948bfca18e4555380b31beebb404a33ede990e7eadfc5c45.jpg)

![fa9f38c31575d7cf3ec19c0d07aca410c663a71af5dd0dae09c4195646b60c31.jpg](../icml_results/4338_Variational%20Learning%20of%20Disentangled%20Representations/tables/fa9f38c31575d7cf3ec19c0d07aca410c663a71af5dd0dae09c4195646b60c31.jpg)

<a id="paper-4295"></a>

## Calibrating Generative Models to Distributional Constraints


### Images

![17dc26846c24f6931ca19d99d0c50026fae5116ae4be50891da1364eec30c34d.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/17dc26846c24f6931ca19d99d0c50026fae5116ae4be50891da1364eec30c34d.jpg)

![1f4e65b2f8152e72dc4eabfc8fb058ec52e8caf6a05f10072cf0f82badffd549.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/1f4e65b2f8152e72dc4eabfc8fb058ec52e8caf6a05f10072cf0f82badffd549.jpg)

![2f1d6508388ed4632b50ab8e8a46ebabf53d0731d9fef194e161e677e5a05387.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/2f1d6508388ed4632b50ab8e8a46ebabf53d0731d9fef194e161e677e5a05387.jpg)

![4699baaa72106699b006c396db6f37987382573a3fbb15c90d533d5b61da6c39.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/4699baaa72106699b006c396db6f37987382573a3fbb15c90d533d5b61da6c39.jpg)

![5d99958c4cc6afa3d4a71b240e8749dde19e2c32c50a7dca1d825ca10605861d.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/5d99958c4cc6afa3d4a71b240e8749dde19e2c32c50a7dca1d825ca10605861d.jpg)

![8be8c8e53f18daa1636e98cadf18ad53ce7950d461e746705425e7a331b99e9d.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/8be8c8e53f18daa1636e98cadf18ad53ce7950d461e746705425e7a331b99e9d.jpg)

![9299dd9a0daefdec494dc6b94be7330d737d37349f9d39e44208d0587c37701c.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/9299dd9a0daefdec494dc6b94be7330d737d37349f9d39e44208d0587c37701c.jpg)

![b772ab01735acd3d6bbaadaa0f50ddda0e4e3e1a3eccdac6280a0fa9dc1f9bd5.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/b772ab01735acd3d6bbaadaa0f50ddda0e4e3e1a3eccdac6280a0fa9dc1f9bd5.jpg)

![c5734363210496cab2f1b383f63051a95498cc601779bd03d0216336812ec963.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/c5734363210496cab2f1b383f63051a95498cc601779bd03d0216336812ec963.jpg)

![e839d90f7147187dfaab0029eda8643f4335ec8b4c16c29015570b89c48cfae5.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/e839d90f7147187dfaab0029eda8643f4335ec8b4c16c29015570b89c48cfae5.jpg)

![f188cb9b3aba38c67c2045d4f21a77f62d6846013a99fcf4e91023b92c293e11.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/f188cb9b3aba38c67c2045d4f21a77f62d6846013a99fcf4e91023b92c293e11.jpg)

![f288e3825f41f2cdc3c10b26300b44fd1b0e8701c6bb79ca43c742ca3333fdcf.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/f288e3825f41f2cdc3c10b26300b44fd1b0e8701c6bb79ca43c742ca3333fdcf.jpg)

![f9cbc4b4c4c0a4f095ba06db6c59606cf5e7ca5a3eb00c210de0a1fe6d2cd97a.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/images/f9cbc4b4c4c0a4f095ba06db6c59606cf5e7ca5a3eb00c210de0a1fe6d2cd97a.jpg)

### Tables

![360699ba47f9efa5063ca7e6274c191a94d3122865a48630b7d1b0966bd947ba.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/tables/360699ba47f9efa5063ca7e6274c191a94d3122865a48630b7d1b0966bd947ba.jpg)

![50171d0ddb4ac61b6eb92458e049fddbdccb2a097295f834cdefebfb0289588d.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/tables/50171d0ddb4ac61b6eb92458e049fddbdccb2a097295f834cdefebfb0289588d.jpg)

![8b31560166a72131ab7969053e00ce7f75c67086e031f0c182ecc6d3e8ea6e41.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/tables/8b31560166a72131ab7969053e00ce7f75c67086e031f0c182ecc6d3e8ea6e41.jpg)

![8dd3d099eb52d2a41170c9d9aee7d048c0834bef53c16598775fa1bbc81d771a.jpg](../icml_results/4339_Calibrating%20Generative%20Models%20to%20Distributional%20Constraints/tables/8dd3d099eb52d2a41170c9d9aee7d048c0834bef53c16598775fa1bbc81d771a.jpg)

<a id="paper-4296"></a>

## Causally Evaluating the Learnability of Formal Language Tasks


### Images

![069ae8afb734cd7286f4267ee12749a27c95ded854028f19797808802112ae3c.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/069ae8afb734cd7286f4267ee12749a27c95ded854028f19797808802112ae3c.jpg)

![0b05dc2cbbcb73584bb57bdb9e6814f7445a976165275d1be39a6de11f2b2832.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/0b05dc2cbbcb73584bb57bdb9e6814f7445a976165275d1be39a6de11f2b2832.jpg)

![2d497f7b2722ac0f57981e071b65c1710e3aba0debd980ac6e551dbbab7d0202.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/2d497f7b2722ac0f57981e071b65c1710e3aba0debd980ac6e551dbbab7d0202.jpg)

![869c7f6628363459c7af04acaa106c96de058abf91d336115a19e9f37562a9df.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/869c7f6628363459c7af04acaa106c96de058abf91d336115a19e9f37562a9df.jpg)

![b1c5eb4e8a963a5b3a8fd04746c716d5d5fe032ffb18681718412b21a5fdfd17.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/b1c5eb4e8a963a5b3a8fd04746c716d5d5fe032ffb18681718412b21a5fdfd17.jpg)

![f6b62be5c28f4d0359e17f5ef550c8dfcb905f3075c434b916c681f061972e6b.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/f6b62be5c28f4d0359e17f5ef550c8dfcb905f3075c434b916c681f061972e6b.jpg)

![fda470b0470853e33a80d34d9ddaf11003256bd30b6dc9d95fbdfb7ff0d9642d.jpg](../icml_results/4340_Causally%20Evaluating%20the%20Learnability%20of%20Formal%20Language%20Tasks/images/fda470b0470853e33a80d34d9ddaf11003256bd30b6dc9d95fbdfb7ff0d9642d.jpg)

<a id="paper-4297"></a>

## Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation


### Images

![02a5531c4cacc606da239ff7615b6182abfb2f0b888bf9a1588e8d6be56d220c.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/02a5531c4cacc606da239ff7615b6182abfb2f0b888bf9a1588e8d6be56d220c.jpg)

![127d63c82a7e857526432a4674d2060a8a2c564185f52ac0b689578b407f20ce.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/127d63c82a7e857526432a4674d2060a8a2c564185f52ac0b689578b407f20ce.jpg)

![1559d5fa81eb6ef4a6eb73fcda2fdcd8a6682655beda7d19f8fce7077020f8e1.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/1559d5fa81eb6ef4a6eb73fcda2fdcd8a6682655beda7d19f8fce7077020f8e1.jpg)

![3ab71e0ce230b9e891fdfa308fa9b5f3c2a329825e1d3372d8f0721925732c58.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/3ab71e0ce230b9e891fdfa308fa9b5f3c2a329825e1d3372d8f0721925732c58.jpg)

![688fc3bb0053bcb27512accaf2e1626cbb936e3c2635951eba829153475252cc.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/688fc3bb0053bcb27512accaf2e1626cbb936e3c2635951eba829153475252cc.jpg)

![6cb6c42728fe6c0fcbeea01477454f779b60124d120e88df26bafbc657fef165.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/6cb6c42728fe6c0fcbeea01477454f779b60124d120e88df26bafbc657fef165.jpg)

![73a0bb9774fb7420cafe5358b77f55b3a91f1fc814ed308294802afd5d881d2b.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/73a0bb9774fb7420cafe5358b77f55b3a91f1fc814ed308294802afd5d881d2b.jpg)

![74a306167377e152c17b32e9a83aff49f712b7eb9af5cb803295ff96f5ea553d.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/74a306167377e152c17b32e9a83aff49f712b7eb9af5cb803295ff96f5ea553d.jpg)

![7cced7c26f874185456b0e9f47d6e5bc9dcbb99da0883da770225f4c5f3190ab.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/7cced7c26f874185456b0e9f47d6e5bc9dcbb99da0883da770225f4c5f3190ab.jpg)

![7ea2733aab5368374e1bd75123a0124bf2d0be8a89dfd9adee1d43066dd55bb5.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/7ea2733aab5368374e1bd75123a0124bf2d0be8a89dfd9adee1d43066dd55bb5.jpg)

![84d0071e6e1151b5c538b4fad5d6499faff668398edee36d603345088a784dcc.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/84d0071e6e1151b5c538b4fad5d6499faff668398edee36d603345088a784dcc.jpg)

![b3732b577a425dcab9a3aa2f15c5fa0844c717667ac9b614a006a1d4723fc585.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/b3732b577a425dcab9a3aa2f15c5fa0844c717667ac9b614a006a1d4723fc585.jpg)

![b6a5e6f479dead5ca3de106a384d13754960e11ca5e9a778f32b028d5d7d2dd0.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/b6a5e6f479dead5ca3de106a384d13754960e11ca5e9a778f32b028d5d7d2dd0.jpg)

![bffc7ecc4b424617060cde8154b1d1dbc4f14045670e0e4e3e787d15b96a545d.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/bffc7ecc4b424617060cde8154b1d1dbc4f14045670e0e4e3e787d15b96a545d.jpg)

![c109a38cc5a6f0cf0f65ed4d3536295c14e6cc7bb309c9ecf0527ec0a3f6beaa.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/c109a38cc5a6f0cf0f65ed4d3536295c14e6cc7bb309c9ecf0527ec0a3f6beaa.jpg)

![c358d772fb301abc1efb8a27700ed88f9d5936d02107115a1f940422a0088424.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/c358d772fb301abc1efb8a27700ed88f9d5936d02107115a1f940422a0088424.jpg)

![c4c11787c23c4b8a68039cdb31c19da2997a27c431ecec987019ca2e829c3834.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/c4c11787c23c4b8a68039cdb31c19da2997a27c431ecec987019ca2e829c3834.jpg)

![d481889e2b960890d5e9c78fdef340974cdc8c2703fa7151c99f29da61c4132f.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/d481889e2b960890d5e9c78fdef340974cdc8c2703fa7151c99f29da61c4132f.jpg)

![de39eb6b58e7d5c9cfb220ca7713f271d2c840368a6723f545d77ed7d25d7b0a.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/de39eb6b58e7d5c9cfb220ca7713f271d2c840368a6723f545d77ed7d25d7b0a.jpg)

![f9c721631b08279f36b07beee2bc52276310bed6bd95f3cc29e43f8b90521e5c.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/images/f9c721631b08279f36b07beee2bc52276310bed6bd95f3cc29e43f8b90521e5c.jpg)

### Tables

![2c33a7baaa7dd36f36b0b541b21c76e18f040dc8fe39330555c6bdb1039c5eac.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/2c33a7baaa7dd36f36b0b541b21c76e18f040dc8fe39330555c6bdb1039c5eac.jpg)

![2d38aef992e23d64d4f68c06595eab276bf2b46bb06625599f8ac7b5df876a09.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/2d38aef992e23d64d4f68c06595eab276bf2b46bb06625599f8ac7b5df876a09.jpg)

![2feec54cd455e99b0f69d55168cf762d2c16f85e1b8635e4926e620afc07ede8.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/2feec54cd455e99b0f69d55168cf762d2c16f85e1b8635e4926e620afc07ede8.jpg)

![30a92264fe4829c270d39d033464e1e1c1e60a63595ddf551491ecb9b1183487.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/30a92264fe4829c270d39d033464e1e1c1e60a63595ddf551491ecb9b1183487.jpg)

![35e52cf70a44df21fe21b3e4fe0f4cbddbb9d63e363e32bb30edf1a550bb27ac.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/35e52cf70a44df21fe21b3e4fe0f4cbddbb9d63e363e32bb30edf1a550bb27ac.jpg)

![3817d26e755649c15c24ccc85381718d4a67a26e047b88fadc897f288dd13a92.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/3817d26e755649c15c24ccc85381718d4a67a26e047b88fadc897f288dd13a92.jpg)

![40e3b4f4aedc7280adbb6c822309d505a686129971218bd14d82bb972b1c3c69.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/40e3b4f4aedc7280adbb6c822309d505a686129971218bd14d82bb972b1c3c69.jpg)

![489339783099d08e00522bd5aa813935ffb79696b119b59b1ec83e837ce27ba0.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/489339783099d08e00522bd5aa813935ffb79696b119b59b1ec83e837ce27ba0.jpg)

![51cc81118ddad4d89dd9f5b6125bd2a4f268ef4256c7eaa7b4eea657c9801544.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/51cc81118ddad4d89dd9f5b6125bd2a4f268ef4256c7eaa7b4eea657c9801544.jpg)

![57057d7c7b91a6bcdd36303f0105fcfc9b32a586ffdded51a0f359f32e080df8.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/57057d7c7b91a6bcdd36303f0105fcfc9b32a586ffdded51a0f359f32e080df8.jpg)

![8e3b76c7d98694b2ae1353a39552195e5c660d7a9b66368f406e142b1b1765ea.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/8e3b76c7d98694b2ae1353a39552195e5c660d7a9b66368f406e142b1b1765ea.jpg)

![d7bd190f0bc176c1345d5b8c77327c97d8b7c1f4648087836fd9c180b6949d0e.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/d7bd190f0bc176c1345d5b8c77327c97d8b7c1f4648087836fd9c180b6949d0e.jpg)

![f2e9d2e15f6654daacb1688a76f96e0eefbf20932555f6f589b86b19cb2dde96.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/f2e9d2e15f6654daacb1688a76f96e0eefbf20932555f6f589b86b19cb2dde96.jpg)

![f7735a41127befeb92062e2d3ed75fbcb4f5aa8ed535c75595bc0641fedea796.jpg](../icml_results/4341_Speculative%20Coupled%20Decoding%20for%20Training-Free%20Lossless%20Acceleration%20of%20Autoregressive%20Visual%20Genera/tables/f7735a41127befeb92062e2d3ed75fbcb4f5aa8ed535c75595bc0641fedea796.jpg)

<a id="paper-4298"></a>

## Scalable and Interpretable Representation Alignment with Ordinal Similarity


### Images

![081787234b82745cc2a8e8745a6c19faaeba5d206cbb50c8543388c0eefae38e.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/081787234b82745cc2a8e8745a6c19faaeba5d206cbb50c8543388c0eefae38e.jpg)

![2c2ee8bf19058d50ff2116d807127de9dec07cb4fbbc90142ecf26b1c3091fdc.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/2c2ee8bf19058d50ff2116d807127de9dec07cb4fbbc90142ecf26b1c3091fdc.jpg)

![2f0bd6ad66720d76ec6f409f782f048f6f7aaf841fa993bb8aef36605808b8df.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/2f0bd6ad66720d76ec6f409f782f048f6f7aaf841fa993bb8aef36605808b8df.jpg)

![46d2d9b7421815792629bedc50bea126e3b8a2415bcdb7935fc4c1082363f612.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/46d2d9b7421815792629bedc50bea126e3b8a2415bcdb7935fc4c1082363f612.jpg)

![70e269a62340b903e14817c8a3e1e26e3ebd8a98269f9f2bee388eec46993bec.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/70e269a62340b903e14817c8a3e1e26e3ebd8a98269f9f2bee388eec46993bec.jpg)

![9f17423a7d46e5139b33f5325ac5fee077b641ccc73fdf7c89e5c62df439e678.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/9f17423a7d46e5139b33f5325ac5fee077b641ccc73fdf7c89e5c62df439e678.jpg)

![b94b2b66cac274b1d2122b2ab141ef9ede32430773bee2fd57d86a14f546c435.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/b94b2b66cac274b1d2122b2ab141ef9ede32430773bee2fd57d86a14f546c435.jpg)

![dce8e1ad5c6717d5f6713ca4866493cd3f531aee0d5291b1679542b75a8e4853.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/dce8e1ad5c6717d5f6713ca4866493cd3f531aee0d5291b1679542b75a8e4853.jpg)

![dd1f638565282d272e838ecea92301754f3e556f7daa45b0913a64d2ec525f88.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/dd1f638565282d272e838ecea92301754f3e556f7daa45b0913a64d2ec525f88.jpg)

![e9a7cb2488497df92f90ae5df276221cbe8499c3823b2f911564606bc0f9e1e0.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/e9a7cb2488497df92f90ae5df276221cbe8499c3823b2f911564606bc0f9e1e0.jpg)

![fda8810dd1872d436dda9963d63eb708497a8daccb68ab6bc0e648a6b7ec6ab1.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/images/fda8810dd1872d436dda9963d63eb708497a8daccb68ab6bc0e648a6b7ec6ab1.jpg)

### Tables

![187c14d7f02ec7aaa64f2076465c1f6e7fffebaf4c4ee0b92c3fa6d6add86597.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/tables/187c14d7f02ec7aaa64f2076465c1f6e7fffebaf4c4ee0b92c3fa6d6add86597.jpg)

![6b6f628edc2a8af4221cbeb123545cebaa6feae0a21bc353d65c56264f1ca81d.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/tables/6b6f628edc2a8af4221cbeb123545cebaa6feae0a21bc353d65c56264f1ca81d.jpg)

![b0821e507d3c1d8b356e2840b26cb6a2a461ac04ad0e05ef12c596b9b442e810.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/tables/b0821e507d3c1d8b356e2840b26cb6a2a461ac04ad0e05ef12c596b9b442e810.jpg)

![e16694515dc477721a7b790e0171d5664a61be65810b91416ef437c0af5d387b.jpg](../icml_results/4342_Scalable%20and%20Interpretable%20Representation%20Alignment%20with%20Ordinal%20Similarity/tables/e16694515dc477721a7b790e0171d5664a61be65810b91416ef437c0af5d387b.jpg)

<a id="paper-4299"></a>

## CountsDiff: A diffusion model on the natural numbers for generation and imputation of count-based data


### Images

![13bbfd19ea156caa97bb16cf60443e5a3cf4e5a6bee4079d6c738a32c0d16144.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/13bbfd19ea156caa97bb16cf60443e5a3cf4e5a6bee4079d6c738a32c0d16144.jpg)

![1a7a1552009822b6be89a41eff2cdb80ac2dacbfad8eeecb5725d7d3469d07b3.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/1a7a1552009822b6be89a41eff2cdb80ac2dacbfad8eeecb5725d7d3469d07b3.jpg)

![25129bf4649a2f4a58449390ebb20da8b1a3c00a2f7824954bba569a637f5ead.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/25129bf4649a2f4a58449390ebb20da8b1a3c00a2f7824954bba569a637f5ead.jpg)

![6c63748c0b97dad717b7bd2a44b8894d74badad596b7c1a412094a587397f84e.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/6c63748c0b97dad717b7bd2a44b8894d74badad596b7c1a412094a587397f84e.jpg)

![7cd4e787439691ccf18e4597fcbe4ff8c703575fcb3fd496f337989984028011.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/7cd4e787439691ccf18e4597fcbe4ff8c703575fcb3fd496f337989984028011.jpg)

![83a38646d339930879f7d7b1f5d37505419a55920fbc200f6874c8925e6050cb.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/83a38646d339930879f7d7b1f5d37505419a55920fbc200f6874c8925e6050cb.jpg)

![97a57cad7c0cddc5c5843f0e9ee5b5734018388831341629029db32a88ebf2b7.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/97a57cad7c0cddc5c5843f0e9ee5b5734018388831341629029db32a88ebf2b7.jpg)

![9abe73462bbf5219c15581e63f60c14986f00a4ccab7a5d89a7117865e0d1532.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/9abe73462bbf5219c15581e63f60c14986f00a4ccab7a5d89a7117865e0d1532.jpg)

![9fef69a2e0eb42d96808698484c3ec1cc11e309ee96d693447d997ea22b63066.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/9fef69a2e0eb42d96808698484c3ec1cc11e309ee96d693447d997ea22b63066.jpg)

![affd87ceefe3c2c30334be8af01ca00da78ca7a258dbce3eedf54655db70cb0c.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/affd87ceefe3c2c30334be8af01ca00da78ca7a258dbce3eedf54655db70cb0c.jpg)

![b4c02f5444be0d1c00337db656fcad27fe018f5cd752ff742cef86f8259ba4e7.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/b4c02f5444be0d1c00337db656fcad27fe018f5cd752ff742cef86f8259ba4e7.jpg)

![c61b1e4b33a741f1e50ede4bd491df8f5731c6377aa95bbb32750fc78c287c6a.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/c61b1e4b33a741f1e50ede4bd491df8f5731c6377aa95bbb32750fc78c287c6a.jpg)

![d6cfb4edf140b986b68f1e75dc631422fbc8d7d895c93d853d11d981a8dcf8b0.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/images/d6cfb4edf140b986b68f1e75dc631422fbc8d7d895c93d853d11d981a8dcf8b0.jpg)

### Tables

![05f02a68c841785e9c54dd20ba55c6af3e5ac752be10beae9992e357b96c7fc2.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/05f02a68c841785e9c54dd20ba55c6af3e5ac752be10beae9992e357b96c7fc2.jpg)

![105d6f7cbedd1ad3a6edc3f49b3bfb9e8b88bdbae1b746fafc715f7170cd560a.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/105d6f7cbedd1ad3a6edc3f49b3bfb9e8b88bdbae1b746fafc715f7170cd560a.jpg)

![2b7efee2ef1d1f077cc834025a6b8e206b17b84e74bb089cf2f70e89ba43a0e2.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/2b7efee2ef1d1f077cc834025a6b8e206b17b84e74bb089cf2f70e89ba43a0e2.jpg)

![3a1913e0bb8f716b5c93f000c876e95e9eaa28901ee14bf1e5df173605ab3ec0.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/3a1913e0bb8f716b5c93f000c876e95e9eaa28901ee14bf1e5df173605ab3ec0.jpg)

![3fd00a0525810b5075f7cb29a737def91e691adaab2bf9e0398dfff565ec9f4b.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/3fd00a0525810b5075f7cb29a737def91e691adaab2bf9e0398dfff565ec9f4b.jpg)

![4d0d0a979883615ce0f7636b8f720e2c52d58ecf47119c018ca7362b85c76e0a.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/4d0d0a979883615ce0f7636b8f720e2c52d58ecf47119c018ca7362b85c76e0a.jpg)

![4e6f31235dd9d9f58c15808605aaf0c6d1da019ed02d294d2f2cc246a393818e.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/4e6f31235dd9d9f58c15808605aaf0c6d1da019ed02d294d2f2cc246a393818e.jpg)

![572a0eac8b5d8905a7b59aa865d0a29efca0a2b6b0d3048ea30e17e3465bc75a.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/572a0eac8b5d8905a7b59aa865d0a29efca0a2b6b0d3048ea30e17e3465bc75a.jpg)

![6f9d4bc750f7b5457d090c6a5047f14e0d760d1d0dd0384f42ff8b38bcedb065.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/6f9d4bc750f7b5457d090c6a5047f14e0d760d1d0dd0384f42ff8b38bcedb065.jpg)

![7751a39c83d87db264fe58aae741c2beb20895a6bd28cff36667337c77d467ae.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/7751a39c83d87db264fe58aae741c2beb20895a6bd28cff36667337c77d467ae.jpg)

![8485f51e91c6ba7b690c7101fd0565edd90104faab5b9175cb04fc0800a75e43.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/8485f51e91c6ba7b690c7101fd0565edd90104faab5b9175cb04fc0800a75e43.jpg)

![902f1dbf06b55bf47afda8127cad9081c61be3ec0d39c67fc52eff3b197ff541.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/902f1dbf06b55bf47afda8127cad9081c61be3ec0d39c67fc52eff3b197ff541.jpg)

![9423d342b80210ce8bec3afeb5cd78da210a64aa893c7bb83dc345706ea737be.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/9423d342b80210ce8bec3afeb5cd78da210a64aa893c7bb83dc345706ea737be.jpg)

![a1dd35f0810f248cc82ada95f5e175f084d74ef2c4f2f69f00b94c9a9e8a304f.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/a1dd35f0810f248cc82ada95f5e175f084d74ef2c4f2f69f00b94c9a9e8a304f.jpg)

![a252ca4ec264b1ae40d845464fb0fc3e6f9504734e66854107a2abbd64b75cc0.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/a252ca4ec264b1ae40d845464fb0fc3e6f9504734e66854107a2abbd64b75cc0.jpg)

![bef8f4a5e0bed97fd3ebe53057953e7ebdc23ee37e3c7cdbdac2bd3315f25b93.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/bef8f4a5e0bed97fd3ebe53057953e7ebdc23ee37e3c7cdbdac2bd3315f25b93.jpg)

![c094610716e233a94d72e4b6b960066b11124ec490623cb1b71bc326236ba44f.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/c094610716e233a94d72e4b6b960066b11124ec490623cb1b71bc326236ba44f.jpg)

![c53f853e5f75f67b39238283241c08086a801da106004eac22639512dbbd06f8.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/c53f853e5f75f67b39238283241c08086a801da106004eac22639512dbbd06f8.jpg)

![d9574a3384f3fe7b0c324f972eafc66d8f269b7019cad876f561193a04616c1a.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/d9574a3384f3fe7b0c324f972eafc66d8f269b7019cad876f561193a04616c1a.jpg)

![e04f62521b3365deee353d91cda12854b45ab8235ffac2e7d8cfd03a1859f95f.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/e04f62521b3365deee353d91cda12854b45ab8235ffac2e7d8cfd03a1859f95f.jpg)

![e4ef47a00f8f17e1f7b6e4d44958b6550ebc11206e655b72f52cfb27644a5dcb.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/e4ef47a00f8f17e1f7b6e4d44958b6550ebc11206e655b72f52cfb27644a5dcb.jpg)

![f13b8ef40c5d5146845429046bcb67e530b73d1705dba2cdc2fdfc19138adf93.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/f13b8ef40c5d5146845429046bcb67e530b73d1705dba2cdc2fdfc19138adf93.jpg)

![f5c6ec31191f6bb162ba4869117a8b4dd99e63fa12c0efcd66b73c23baaa1e5f.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/f5c6ec31191f6bb162ba4869117a8b4dd99e63fa12c0efcd66b73c23baaa1e5f.jpg)

![f81703c9b06dd1d88ecdc4e1ae7130796fc6e5036fccc80df9fcd22009b20729.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/f81703c9b06dd1d88ecdc4e1ae7130796fc6e5036fccc80df9fcd22009b20729.jpg)

![fbff0c51ec2da5330db8885fde0e8129396957c67fd71283fdf95fbdda9109fe.jpg](../icml_results/4343_CountsDiff_%20A%20diffusion%20model%20on%20the%20natural%20numbers%20for%20generation%20and%20imputation%20of%20count-based%20da/tables/fbff0c51ec2da5330db8885fde0e8129396957c67fd71283fdf95fbdda9109fe.jpg)

<a id="paper-4300"></a>

## Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions


### Images

![0514a665b7fb2bcb9a8161f506aefd614bed99d87afda76a221fd8c08b429036.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/0514a665b7fb2bcb9a8161f506aefd614bed99d87afda76a221fd8c08b429036.jpg)

![061ffad10f9584498b4f5ad4a9d911cbac91614d233f3e621f64382485718b38.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/061ffad10f9584498b4f5ad4a9d911cbac91614d233f3e621f64382485718b38.jpg)

![12ca1d837abc7efb458b6aecf93c9b64bb9e527574a8f896121d284159479e3d.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/12ca1d837abc7efb458b6aecf93c9b64bb9e527574a8f896121d284159479e3d.jpg)

![13fb649f276ba9143e2496f094fe3b9477e798a41733542894df1eb94b699a97.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/13fb649f276ba9143e2496f094fe3b9477e798a41733542894df1eb94b699a97.jpg)

![2181f39197be5904e3cf041c4c4d4656ec953c1db465f3f547151503550e673b.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/2181f39197be5904e3cf041c4c4d4656ec953c1db465f3f547151503550e673b.jpg)

![2514eb043bf86925464f60bc7f9b154c45b779444b9a5e672b96eed9f19de9b0.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/2514eb043bf86925464f60bc7f9b154c45b779444b9a5e672b96eed9f19de9b0.jpg)

![26464a4f9b66bfd0d292a0e9ba7a6bf4e36452e1834f4731a51d17e7ac8fad04.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/26464a4f9b66bfd0d292a0e9ba7a6bf4e36452e1834f4731a51d17e7ac8fad04.jpg)

![2846cd5811c01560fca375e8e0126bc448ca41e104ab5bee84db5f36b8a683e6.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/2846cd5811c01560fca375e8e0126bc448ca41e104ab5bee84db5f36b8a683e6.jpg)

![39e183eb0540665a1ed09326fddb6e56bf6bbb4bb9a5e0444f4f46cb4cad3edb.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/39e183eb0540665a1ed09326fddb6e56bf6bbb4bb9a5e0444f4f46cb4cad3edb.jpg)

![3cd247e759af47bc802c65a7a52b2c3016118da9a4a5acb298b2edd80f78c2a6.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/3cd247e759af47bc802c65a7a52b2c3016118da9a4a5acb298b2edd80f78c2a6.jpg)

![4201ece8687d70b92e66f754fe49cd551f24914a12a7d071a89ab7b3931d9cf0.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/4201ece8687d70b92e66f754fe49cd551f24914a12a7d071a89ab7b3931d9cf0.jpg)

![45e85384602417b45529147c964443f988da9d185f11f4062233c560c031e43e.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/45e85384602417b45529147c964443f988da9d185f11f4062233c560c031e43e.jpg)

![4e4d8309b7b02b2c8e267a0c24b5b8c208cd652bf14004eae9cacedd1da73bbc.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/4e4d8309b7b02b2c8e267a0c24b5b8c208cd652bf14004eae9cacedd1da73bbc.jpg)

![4e8e92f2f6757de254da49c844a693914fa207ab4c24663ac29c98d06de94c5c.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/4e8e92f2f6757de254da49c844a693914fa207ab4c24663ac29c98d06de94c5c.jpg)

![59d74e771831dd1d935ba9b0175a78147082fad5a9617971554f7d89dfc80ba4.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/59d74e771831dd1d935ba9b0175a78147082fad5a9617971554f7d89dfc80ba4.jpg)

![5e78b54b022677a24ec5b5f2f3897729a5739ecb5466a0a9f26fd220d9b14fd9.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/5e78b54b022677a24ec5b5f2f3897729a5739ecb5466a0a9f26fd220d9b14fd9.jpg)

![61a54489b7ee78c0900bcdd0238788e9f6be7537599745f09de8435a7a98b07a.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/61a54489b7ee78c0900bcdd0238788e9f6be7537599745f09de8435a7a98b07a.jpg)

![6437779f886ab4d743b28bf9cd6b6163d7d45704317aed407865a86b85820cde.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/6437779f886ab4d743b28bf9cd6b6163d7d45704317aed407865a86b85820cde.jpg)

![648ece787e7ba60d03cd3d4d8b1c09f02efd4db636e39c1bca5d22fbac70b4e0.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/648ece787e7ba60d03cd3d4d8b1c09f02efd4db636e39c1bca5d22fbac70b4e0.jpg)

![658d3a96015ecd7f030c4db4f91c6a9069605f615a48b99ed487e99e190fb5eb.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/658d3a96015ecd7f030c4db4f91c6a9069605f615a48b99ed487e99e190fb5eb.jpg)

![6bcd8f5bd764d12e0464ae628811e3c9358407fa6630ec5ef453d501890ff393.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/6bcd8f5bd764d12e0464ae628811e3c9358407fa6630ec5ef453d501890ff393.jpg)

![7be4994208367b0ae9f1da4cbceb04e9885fa519ff3f4778837ea1834f21a036.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/7be4994208367b0ae9f1da4cbceb04e9885fa519ff3f4778837ea1834f21a036.jpg)

![814c9c2ac4259e207f223db8a0b27393ce89a785ca8fa1f9dc350b435a189604.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/814c9c2ac4259e207f223db8a0b27393ce89a785ca8fa1f9dc350b435a189604.jpg)

![82d172c3ea5c68074b937fceb79b4ccb10e4e3b536b580709a0efb8511a74a2b.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/82d172c3ea5c68074b937fceb79b4ccb10e4e3b536b580709a0efb8511a74a2b.jpg)

![947402bade00a9fef9f466d82e5516e0e7eb8e356bde63cf8379ea77ff2150e2.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/947402bade00a9fef9f466d82e5516e0e7eb8e356bde63cf8379ea77ff2150e2.jpg)

![956bf8a7e0cfd103a894cdc08b3a9928274ff9822ea5bcf26311ff1f425a9c68.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/956bf8a7e0cfd103a894cdc08b3a9928274ff9822ea5bcf26311ff1f425a9c68.jpg)

![99816786ec42e497866b32c60b44d614429a6e0569361b8d9dfa552e54b97256.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/99816786ec42e497866b32c60b44d614429a6e0569361b8d9dfa552e54b97256.jpg)

![a1ac60a12e11e038307bfcdd1cf6dcb930ae9d0d109c9b7d748769b305f53e3b.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/a1ac60a12e11e038307bfcdd1cf6dcb930ae9d0d109c9b7d748769b305f53e3b.jpg)

![aa128ced24227766e1c3c179d5f9b5f828eae9979a32a0c8b961e6d98938e55c.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/aa128ced24227766e1c3c179d5f9b5f828eae9979a32a0c8b961e6d98938e55c.jpg)

![b24f8257af8978da608ac49d2c43781a1b48219a1da97a9d8a10709f654fe565.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/b24f8257af8978da608ac49d2c43781a1b48219a1da97a9d8a10709f654fe565.jpg)

![b353fadb6e09efdc3b303ef7f3792ce87cac674ce3434f64172e5ee3ebc5ecda.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/b353fadb6e09efdc3b303ef7f3792ce87cac674ce3434f64172e5ee3ebc5ecda.jpg)

![c45706231665e8a96a7f6ac283a3a5a85ed9eec7c6079f4d3625c086fa9c11cf.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/c45706231665e8a96a7f6ac283a3a5a85ed9eec7c6079f4d3625c086fa9c11cf.jpg)

![cf99a39b8235da1bca2d2dc54095c6787470889b64ac55ce54e2e8c2ecfc417d.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/cf99a39b8235da1bca2d2dc54095c6787470889b64ac55ce54e2e8c2ecfc417d.jpg)

![d155b418a7ac984549cc37b4cb1d7108a4634e62f7b10eba955397fad143e7b8.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/d155b418a7ac984549cc37b4cb1d7108a4634e62f7b10eba955397fad143e7b8.jpg)

![d6f35beeb67586a08c73e38fce8e92ac20779a3eb655a80c0b42bd635de96064.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/d6f35beeb67586a08c73e38fce8e92ac20779a3eb655a80c0b42bd635de96064.jpg)

![d8316617de07ac4bbe76c38ed01255a7c1bcbe3fafaa3cb69e34b65ab448fdff.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/d8316617de07ac4bbe76c38ed01255a7c1bcbe3fafaa3cb69e34b65ab448fdff.jpg)

![dce0b605350e8b6c48ae559c988558b56174b9a3c91f94854b76d76423c9fd71.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/dce0b605350e8b6c48ae559c988558b56174b9a3c91f94854b76d76423c9fd71.jpg)

![e720d52f49aca61790fb3c48badef935056b0c92c04bc48192dbf7b80ebcd9bc.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/e720d52f49aca61790fb3c48badef935056b0c92c04bc48192dbf7b80ebcd9bc.jpg)

![ef7d87d0b024b26c14574b4cbed605397614a79475e6f71d75c83ae15a873198.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/ef7d87d0b024b26c14574b4cbed605397614a79475e6f71d75c83ae15a873198.jpg)

![f48186f9a17cb1884f37bb04afe6b83a106074905a0531293aed010eeb6ed1b4.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/f48186f9a17cb1884f37bb04afe6b83a106074905a0531293aed010eeb6ed1b4.jpg)

![f745838c99e53e3ff1d2d787a41c39937584abaa603ead5e306914032a7c618d.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/images/f745838c99e53e3ff1d2d787a41c39937584abaa603ead5e306914032a7c618d.jpg)

### Tables

![03a56c343e5df16f10eebe70a845d1b97ef0fefb598598a7e5fb0a73bb1c3fc4.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/03a56c343e5df16f10eebe70a845d1b97ef0fefb598598a7e5fb0a73bb1c3fc4.jpg)

![2c0a6b2f871d955c4490dc07ef2f7dd5d26beddedf3944bdb1102dd2bb0c09a5.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/2c0a6b2f871d955c4490dc07ef2f7dd5d26beddedf3944bdb1102dd2bb0c09a5.jpg)

![4591e1e147c0cf9b3950ab9e5b34be61242abfa4029e751d5752e55b85cda908.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/4591e1e147c0cf9b3950ab9e5b34be61242abfa4029e751d5752e55b85cda908.jpg)

![49174268d3b2bb3dabf232cda47ccf2d9cebe7c7c6c428856d383d20addfc3d6.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/49174268d3b2bb3dabf232cda47ccf2d9cebe7c7c6c428856d383d20addfc3d6.jpg)

![494361ee6465fe855d210081197d1a70b812bd9d2e87b4651164896d2729ee8f.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/494361ee6465fe855d210081197d1a70b812bd9d2e87b4651164896d2729ee8f.jpg)

![5b568a1f2b0e392c3c8115bdcc072e730752b9479977e93d2b973c37031bb051.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/5b568a1f2b0e392c3c8115bdcc072e730752b9479977e93d2b973c37031bb051.jpg)

![5dfcfeceba3177f38a198d326ea07b844a7058e6740442b99d58ad51506d0bda.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/5dfcfeceba3177f38a198d326ea07b844a7058e6740442b99d58ad51506d0bda.jpg)

![8c6dbf86e0edcaf1941142f9280e9929d37b7c0f02f829709aa40ec5721666cf.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/8c6dbf86e0edcaf1941142f9280e9929d37b7c0f02f829709aa40ec5721666cf.jpg)

![ad8c23bb60757e754b2acc808957a7c8047ba7bdf7be86799ca764d4f8163355.jpg](../icml_results/4344_Auditing%20Sybil_%20Explaining%20Deep%20Lung%20Cancer%20Risk%20Prediction%20Through%20Generative%20Interventional%20Attrib/tables/ad8c23bb60757e754b2acc808957a7c8047ba7bdf7be86799ca764d4f8163355.jpg)

<a id="paper-4301"></a>

## TT-Sparse: Learning Sparse Rule Models with Differentiable Truth Tables


### Images

![16bb8faaabc54b1bff28272295aec7d1d96b1fe4c28f4968f3f610f2d22bd67a.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/16bb8faaabc54b1bff28272295aec7d1d96b1fe4c28f4968f3f610f2d22bd67a.jpg)

![219caa95ba3110de2c08fc1afc43251e9bf1a09daf677d2293b16f8ebff416e1.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/219caa95ba3110de2c08fc1afc43251e9bf1a09daf677d2293b16f8ebff416e1.jpg)

![2242e6dbe652523ac90046ceacebd2bd5faf9255745ddc727a5cf6a31ce834f5.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/2242e6dbe652523ac90046ceacebd2bd5faf9255745ddc727a5cf6a31ce834f5.jpg)

![28d6f2f1cea137a7b61bf40282b623502f4701f562035907dceff5c14323b0fb.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/28d6f2f1cea137a7b61bf40282b623502f4701f562035907dceff5c14323b0fb.jpg)

![2ff0f99a0e2810d0933861904115f2a005ad18bd7d06f7c7c3e82f833672cf99.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/2ff0f99a0e2810d0933861904115f2a005ad18bd7d06f7c7c3e82f833672cf99.jpg)

![3f5c6d71bc57f063006f22935aa108680e16cc02e2b413ca629443d8122df0fa.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/3f5c6d71bc57f063006f22935aa108680e16cc02e2b413ca629443d8122df0fa.jpg)

![4659bcbd2cf9d8845813e56dd7fd82d9717cc249b6c141c8750a1a9c3a878913.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/4659bcbd2cf9d8845813e56dd7fd82d9717cc249b6c141c8750a1a9c3a878913.jpg)

![4c12ff2de6b01fd15cdaac20682aa44c87f20ea2c8c72bd002458323d78ca99d.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/4c12ff2de6b01fd15cdaac20682aa44c87f20ea2c8c72bd002458323d78ca99d.jpg)

![51844d49e7afcad1ec374a712cdb418d6852ede806b4fe3500023e845510e9ed.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/51844d49e7afcad1ec374a712cdb418d6852ede806b4fe3500023e845510e9ed.jpg)

![54c3370edbcc6906fbc8c0c128b3a9eda275402840da265f9de1e3b855960f84.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/54c3370edbcc6906fbc8c0c128b3a9eda275402840da265f9de1e3b855960f84.jpg)

![5aba54dcbd4f20ef44f7e65ba253867ce33502268082ba7cae050b6f8d43f9fb.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/5aba54dcbd4f20ef44f7e65ba253867ce33502268082ba7cae050b6f8d43f9fb.jpg)

![5e73505d5760e7a14d203bbaf5b2606df7fb0828b5c6375dede6657574938024.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/5e73505d5760e7a14d203bbaf5b2606df7fb0828b5c6375dede6657574938024.jpg)

![5e9293e2c4ce19a2adbe1259edecf3b80b91884783ad751c11332945bb0fb326.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/5e9293e2c4ce19a2adbe1259edecf3b80b91884783ad751c11332945bb0fb326.jpg)

![68fc23ee1759fe4d95c8813c55e84e8da84f15fcb0c10c30c3007310c864c052.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/68fc23ee1759fe4d95c8813c55e84e8da84f15fcb0c10c30c3007310c864c052.jpg)

![69be780e29d57b580f3e81a77ea69fdb1bff86ec91be41af155fd8eb089c36aa.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/69be780e29d57b580f3e81a77ea69fdb1bff86ec91be41af155fd8eb089c36aa.jpg)

![6cf7f34a7a293a79cf32571890918c7b1684e303cce5966e5e184b8b0c8f83ce.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/6cf7f34a7a293a79cf32571890918c7b1684e303cce5966e5e184b8b0c8f83ce.jpg)

![6f668f262a2e048b33ddc12bfcb6cba38c202d48f6d76119f6d43fa2947912e9.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/6f668f262a2e048b33ddc12bfcb6cba38c202d48f6d76119f6d43fa2947912e9.jpg)

![79cd02243423bfebbb0cdb5435bdcfca2cf797e0e1e757fc2251e4fa136ae673.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/79cd02243423bfebbb0cdb5435bdcfca2cf797e0e1e757fc2251e4fa136ae673.jpg)

![847d4a559c11811cb6363da0aed601f06e8befa129cb4426fe2caa0bbc8632bd.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/847d4a559c11811cb6363da0aed601f06e8befa129cb4426fe2caa0bbc8632bd.jpg)

![87a1a1d927f73b6f89b5e1a90c887319163dc848066783f564ade0c1c23459dc.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/87a1a1d927f73b6f89b5e1a90c887319163dc848066783f564ade0c1c23459dc.jpg)

![8dbb4f7ea36746e36f4c05b5d50dba0ee2898e2be2cd98ef59f69ad48c83da77.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/8dbb4f7ea36746e36f4c05b5d50dba0ee2898e2be2cd98ef59f69ad48c83da77.jpg)

![9172abdfb4b73278601f649654088862cf41e0dbd156bee82856feeb867a3737.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/9172abdfb4b73278601f649654088862cf41e0dbd156bee82856feeb867a3737.jpg)

![92aa9c1a6f8ee5f0bbb9772d8ef08b2dda2633aa5ddbe86340fc47446f2dab09.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/92aa9c1a6f8ee5f0bbb9772d8ef08b2dda2633aa5ddbe86340fc47446f2dab09.jpg)

![963d75500243aa1522789e374f6136f729c1ecd0a05c0ee52575b1ec8d558551.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/963d75500243aa1522789e374f6136f729c1ecd0a05c0ee52575b1ec8d558551.jpg)

![964cfa78fd443ba227fa324d50dfabb5469222a7a47d7795875cf95640ad70ed.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/964cfa78fd443ba227fa324d50dfabb5469222a7a47d7795875cf95640ad70ed.jpg)

![9fc9e1f14a74bddaca44e487b6ef37befa404cbf0153833408522941937a7057.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/9fc9e1f14a74bddaca44e487b6ef37befa404cbf0153833408522941937a7057.jpg)

![a5cb9cec08e742099cc80ef952075b273ac700eafe97818bc1ad5ed836158790.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/a5cb9cec08e742099cc80ef952075b273ac700eafe97818bc1ad5ed836158790.jpg)

![a91caeadae610d60f0e57d88b3978bd244298322bc0a2191eb7392fe9b24b802.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/a91caeadae610d60f0e57d88b3978bd244298322bc0a2191eb7392fe9b24b802.jpg)

![ad297e330a54b61ea7deabeffb228d71f19477c9b6365b4f69a4436f666c90d7.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/ad297e330a54b61ea7deabeffb228d71f19477c9b6365b4f69a4436f666c90d7.jpg)

![b72f1bff0a22fe87dc075a943a6e2a899ed4c8eccfe4b6aefd181a24ea9a942c.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/b72f1bff0a22fe87dc075a943a6e2a899ed4c8eccfe4b6aefd181a24ea9a942c.jpg)

![beae1ef695a1558b257d4421080afd4611033408d98b5eea7a2dea70457a7f78.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/beae1ef695a1558b257d4421080afd4611033408d98b5eea7a2dea70457a7f78.jpg)

![bff994fbcf780ccb5f91bff474ae2868219a0866ff3159bd0878de51db913ed4.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/bff994fbcf780ccb5f91bff474ae2868219a0866ff3159bd0878de51db913ed4.jpg)

![d0a81d56a6053ca575e265172c7b64cff244abc7e0fb8cc2fe136d75afc82b9c.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/d0a81d56a6053ca575e265172c7b64cff244abc7e0fb8cc2fe136d75afc82b9c.jpg)

![db75047fdc9d4a95728ece15ceb40416120d6039f4c5181dc7dc8e46332aebec.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/db75047fdc9d4a95728ece15ceb40416120d6039f4c5181dc7dc8e46332aebec.jpg)

![e4e5e4b1a321d2069f2cd503e1e41e07769384e2afcf66584ade810ec30d791d.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/e4e5e4b1a321d2069f2cd503e1e41e07769384e2afcf66584ade810ec30d791d.jpg)

![ea9f5ea45ba9408e9543dcaad57cd87aac6edebd50aa67629cf1de9fe9acd584.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/ea9f5ea45ba9408e9543dcaad57cd87aac6edebd50aa67629cf1de9fe9acd584.jpg)

![ede3e4dd979f5e4717165a9280a6502471abdb9330503c60ca6295136834a8e8.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/ede3e4dd979f5e4717165a9280a6502471abdb9330503c60ca6295136834a8e8.jpg)

![f0f7faaa318a473b685afe5c1892384e809cf92307ffa9b89824ca39c494fcd0.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/f0f7faaa318a473b685afe5c1892384e809cf92307ffa9b89824ca39c494fcd0.jpg)

![f724018b72bada5b2d553ec7ddf2b55373bee9626e7fb6dadd7d6369c4d04c4a.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/f724018b72bada5b2d553ec7ddf2b55373bee9626e7fb6dadd7d6369c4d04c4a.jpg)

![f791d91f801c742ce689867957a405ff6f92dc28a614d925c05d3e8e9427733f.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/f791d91f801c742ce689867957a405ff6f92dc28a614d925c05d3e8e9427733f.jpg)

![fc13a1e57cb7ba3b92e270185af41c181e7b993b92de301e6cff25d26228a25a.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/images/fc13a1e57cb7ba3b92e270185af41c181e7b993b92de301e6cff25d26228a25a.jpg)

### Tables

![365dd1c091a56d2fdf3d35b1fd7d1489efb110fdddf1f2d5a44e82784379de37.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/365dd1c091a56d2fdf3d35b1fd7d1489efb110fdddf1f2d5a44e82784379de37.jpg)

![65e8354f4bc318e29a0cd97a999e0ad6690329bbc8370b49fed0417e792442b0.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/65e8354f4bc318e29a0cd97a999e0ad6690329bbc8370b49fed0417e792442b0.jpg)

![71a0b50d627d22f0eb96a6c02b18e9507d6449df2d8b5a87ada9b5ffe1e525d0.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/71a0b50d627d22f0eb96a6c02b18e9507d6449df2d8b5a87ada9b5ffe1e525d0.jpg)

![80a5b44bf63ee79b56d0db28b2bc825826aef06d9de35b26b0b1227bb4b28c97.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/80a5b44bf63ee79b56d0db28b2bc825826aef06d9de35b26b0b1227bb4b28c97.jpg)

![94fe203830c20b4fcca638c40331cc8d9f39a9afe8b7aa510b2e452e5a449cba.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/94fe203830c20b4fcca638c40331cc8d9f39a9afe8b7aa510b2e452e5a449cba.jpg)

![a731c824338c3214973b5a54c0369a1716f4937cbfc73ebcc76245a7c27e0026.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/a731c824338c3214973b5a54c0369a1716f4937cbfc73ebcc76245a7c27e0026.jpg)

![ee0995afb9887d0a3ec7b469b6a399f925c71f0359240245084c4736b1d40ad6.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/ee0995afb9887d0a3ec7b469b6a399f925c71f0359240245084c4736b1d40ad6.jpg)

![fe419e07a957f0d5f5187505b1b142b7b5be1a9765c5ead3e20fc5abf4af1d5b.jpg](../icml_results/4345_TT-Sparse_%20Learning%20Sparse%20Rule%20Models%20with%20Differentiable%20Truth%20Tables/tables/fe419e07a957f0d5f5187505b1b142b7b5be1a9765c5ead3e20fc5abf4af1d5b.jpg)

<a id="paper-4302"></a>

## Process Reward Agents for Steering Knowledge-Intensive Reasoning


### Images

![2fa312001c35e8d154589c7997f89ade6dea7d58dac13c2526dd9359f6f23399.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/2fa312001c35e8d154589c7997f89ade6dea7d58dac13c2526dd9359f6f23399.jpg)

![648979120746b64ad40ad1af3a3915b51b27ac4354dc4efc73fa77705c5f50db.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/648979120746b64ad40ad1af3a3915b51b27ac4354dc4efc73fa77705c5f50db.jpg)

![b494eed6b3ef658747f39a0c5aada8eb34987713226d964b58039e6bb76a6090.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/b494eed6b3ef658747f39a0c5aada8eb34987713226d964b58039e6bb76a6090.jpg)

![d2def8c36c0dc369e4ed1a21d6a053c7b50bcf6144314003bb377a4357753ff1.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/d2def8c36c0dc369e4ed1a21d6a053c7b50bcf6144314003bb377a4357753ff1.jpg)

![e189d57b631784ef5d5e9220db3dd5c3d7581fda89eed38745a64b7835ecf697.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/e189d57b631784ef5d5e9220db3dd5c3d7581fda89eed38745a64b7835ecf697.jpg)

![f1acce6a639aa133eb63a2f69573c23d0bce58a82f373a0f58bbf2643831ffba.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/f1acce6a639aa133eb63a2f69573c23d0bce58a82f373a0f58bbf2643831ffba.jpg)

![facda29ef5be3b88227da6a1b53b61b33f68d3021b2f1031ebb89720646e7f9c.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/images/facda29ef5be3b88227da6a1b53b61b33f68d3021b2f1031ebb89720646e7f9c.jpg)

### Tables

![09d950fade8e4d0059c364715e9a6680e6fd37f26790d8ac35ce59a097a24959.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/tables/09d950fade8e4d0059c364715e9a6680e6fd37f26790d8ac35ce59a097a24959.jpg)

![1d87fca73e225f5fb81dde37fa2c7e94acf2243c560093faecdb59e0e50a5ded.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/tables/1d87fca73e225f5fb81dde37fa2c7e94acf2243c560093faecdb59e0e50a5ded.jpg)

![21381d630f89f9d326c054314f3c8438dae897877a3d1510161954bff3208a5a.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/tables/21381d630f89f9d326c054314f3c8438dae897877a3d1510161954bff3208a5a.jpg)

![be9a08a997fc4904b4ea2aa4a203f7ef33b42ecb1eb95063712f5fccc9c76375.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/tables/be9a08a997fc4904b4ea2aa4a203f7ef33b42ecb1eb95063712f5fccc9c76375.jpg)

![eb0606180c1ec087586398c608cca51ad4cd7d297e63991eabffd1d4340e5846.jpg](../icml_results/4346_Process%20Reward%20Agents%20for%20Steering%20Knowledge-Intensive%20Reasoning/tables/eb0606180c1ec087586398c608cca51ad4cd7d297e63991eabffd1d4340e5846.jpg)

<a id="paper-4303"></a>

## Accelerated and Stable Convergence with Anchored Generalized Optimistic Method


### Images

![27f8f89f84e13cd9e8d3c3948be0e223d33c74245510b09b8de1f676361c5add.jpg](../icml_results/4347_Accelerated%20and%20Stable%20Convergence%20with%20Anchored%20Generalized%20Optimistic%20Method/images/27f8f89f84e13cd9e8d3c3948be0e223d33c74245510b09b8de1f676361c5add.jpg)

![5fe022501bcc0848930825b7f9b0fafc92b854725fb519010f76d74dce757433.jpg](../icml_results/4347_Accelerated%20and%20Stable%20Convergence%20with%20Anchored%20Generalized%20Optimistic%20Method/images/5fe022501bcc0848930825b7f9b0fafc92b854725fb519010f76d74dce757433.jpg)

![82ac95febdfac981c73a68ff365b81fe71396599bbea66604c7b9de0c2bde976.jpg](../icml_results/4347_Accelerated%20and%20Stable%20Convergence%20with%20Anchored%20Generalized%20Optimistic%20Method/images/82ac95febdfac981c73a68ff365b81fe71396599bbea66604c7b9de0c2bde976.jpg)

### Tables

![1b086390d28f6cd95f3b3084a969bd63d84694a82936e0f59fb6f9b4a3962fcd.jpg](../icml_results/4347_Accelerated%20and%20Stable%20Convergence%20with%20Anchored%20Generalized%20Optimistic%20Method/tables/1b086390d28f6cd95f3b3084a969bd63d84694a82936e0f59fb6f9b4a3962fcd.jpg)

<a id="paper-4304"></a>

## Neural Quantum States in Mixed Precision


### Images

![05c944ff8c36574b2d2f7e1c3c69db5826c350a5315ae0f53270dd757eae0144.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/05c944ff8c36574b2d2f7e1c3c69db5826c350a5315ae0f53270dd757eae0144.jpg)

![1ee58f4aa4f2d50bbf88659546e32d0dc1200473fc6caad7c4ee75e39d43685d.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/1ee58f4aa4f2d50bbf88659546e32d0dc1200473fc6caad7c4ee75e39d43685d.jpg)

![497413ed8a31826ac5288a207d949555fdeadbee0819412c211fadeedd8165cc.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/497413ed8a31826ac5288a207d949555fdeadbee0819412c211fadeedd8165cc.jpg)

![5f221b7be6a2df152d71a2da6ce7e753e268eeed38137f85bbc032fc609a005f.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/5f221b7be6a2df152d71a2da6ce7e753e268eeed38137f85bbc032fc609a005f.jpg)

![6562127cff289bc46510e61a91673a314333b25ce7eadcf8bc260329e9c2f40e.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/6562127cff289bc46510e61a91673a314333b25ce7eadcf8bc260329e9c2f40e.jpg)

![7ed6403b2599bd5f812a87733ea21e2d41f4132c159135c51791baa0a77a2102.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/7ed6403b2599bd5f812a87733ea21e2d41f4132c159135c51791baa0a77a2102.jpg)

![9b255e2743287f3bd39de2917fcb4fcfaac511d130e8d04fa2cd18e65c403734.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/9b255e2743287f3bd39de2917fcb4fcfaac511d130e8d04fa2cd18e65c403734.jpg)

![b4605e02d0446702eb72e9b728b04164b7e0bdd41d9368c225faea4ffe78982c.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/b4605e02d0446702eb72e9b728b04164b7e0bdd41d9368c225faea4ffe78982c.jpg)

![c095cd66b54619e62c28fa9856668dbc974c894ef660b30436bf11e7db6b5a5b.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/c095cd66b54619e62c28fa9856668dbc974c894ef660b30436bf11e7db6b5a5b.jpg)

![c756772925581f9960a3ee8f927ff6cb46434c73535414ee6b9cbc7d966cfc63.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/c756772925581f9960a3ee8f927ff6cb46434c73535414ee6b9cbc7d966cfc63.jpg)

![cbb172da1da9c8ac9e63fdb7c5c2f3590c4d978e4ffdf61bdc3620a50feee8f1.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/cbb172da1da9c8ac9e63fdb7c5c2f3590c4d978e4ffdf61bdc3620a50feee8f1.jpg)

![fb1fb74493fc17f335694fb4d529bfde22ca96d1b8d89e8bf165b9997ce07955.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/images/fb1fb74493fc17f335694fb4d529bfde22ca96d1b8d89e8bf165b9997ce07955.jpg)

### Tables

![482f7509c09d633a0ba4af4347638a76a0febdfe91f2e6710218375dfc6a1b45.jpg](../icml_results/4348_Neural%20Quantum%20States%20in%20Mixed%20Precision/tables/482f7509c09d633a0ba4af4347638a76a0febdfe91f2e6710218375dfc6a1b45.jpg)

<a id="paper-4305"></a>

## WaveSSM: Multiscale State-Space Models for Non-stationary Signal Attention


### Images

![071d5b562e7d7ebec4e53374175011a770aeb1f8bbec29d5d9a2cf2540453587.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/071d5b562e7d7ebec4e53374175011a770aeb1f8bbec29d5d9a2cf2540453587.jpg)

![0a26cb565e5990c658ceec1f4599dd7cdcf9714f4967d4b0dd7b1aef1d31033f.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/0a26cb565e5990c658ceec1f4599dd7cdcf9714f4967d4b0dd7b1aef1d31033f.jpg)

![0c1d2c427db7ba36e9a86547a9b8fd31b261bde73a4f620bd8dd50ab86eed612.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/0c1d2c427db7ba36e9a86547a9b8fd31b261bde73a4f620bd8dd50ab86eed612.jpg)

![18a848a677b5c5cd06892b89f86313e4113d129771abe0b184c9bc4ce7f7f004.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/18a848a677b5c5cd06892b89f86313e4113d129771abe0b184c9bc4ce7f7f004.jpg)

![1fb0c16944a82354bcec3746f217dbbc23d5e7be3dc9f345f9edf335e5ec3b79.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/1fb0c16944a82354bcec3746f217dbbc23d5e7be3dc9f345f9edf335e5ec3b79.jpg)

![266a6ff7eb6b3497747fb9fb89946e60c254bf25794b6a46beca711fd172d0da.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/266a6ff7eb6b3497747fb9fb89946e60c254bf25794b6a46beca711fd172d0da.jpg)

![28f9ed3aa04db668e9f6bef22bc54676bd24daf9507be149a44ce6ff12a2440a.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/28f9ed3aa04db668e9f6bef22bc54676bd24daf9507be149a44ce6ff12a2440a.jpg)

![29fb1b9d46a02bf0cf5accbbf0c599a36791cd8081b429e6a78a20a40df97f13.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/29fb1b9d46a02bf0cf5accbbf0c599a36791cd8081b429e6a78a20a40df97f13.jpg)

![3dfe5ca9accde33e11a05645352ba8a0e74f1363a943502e225506a0ca6f1417.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/3dfe5ca9accde33e11a05645352ba8a0e74f1363a943502e225506a0ca6f1417.jpg)

![3e48d0d75d0369348d5cdc55f9b622f322826c9b1a16dcd76bee6c8ebb136129.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/3e48d0d75d0369348d5cdc55f9b622f322826c9b1a16dcd76bee6c8ebb136129.jpg)

![49225f9548133464efcb0c43296f4cc841dfb10f4b485ff7627d2670c5232154.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/49225f9548133464efcb0c43296f4cc841dfb10f4b485ff7627d2670c5232154.jpg)

![56a89f2b81d82ae638ebf6557dd23b74d97336bcbaf41ff2106f21eaf8650a8a.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/56a89f2b81d82ae638ebf6557dd23b74d97336bcbaf41ff2106f21eaf8650a8a.jpg)

![6739cde83df7dfa3b45e6e6210ea13ec59d8a3f2a7a7c0379093723bced73fc1.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/6739cde83df7dfa3b45e6e6210ea13ec59d8a3f2a7a7c0379093723bced73fc1.jpg)

![8168b795ce5fdf9f30e7881ad6266b8d4fb7a3a86b8feb2af201632eed095b57.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/8168b795ce5fdf9f30e7881ad6266b8d4fb7a3a86b8feb2af201632eed095b57.jpg)

![a4a1bc19bdc39f9d81f47cc26e7655f0ff54ed5d3e8bb61c50471458bf8e7c86.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/a4a1bc19bdc39f9d81f47cc26e7655f0ff54ed5d3e8bb61c50471458bf8e7c86.jpg)

![aa7d84a75cf692c14cbe678cedff55a47698149237ab487ad375c5ae047f2677.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/aa7d84a75cf692c14cbe678cedff55a47698149237ab487ad375c5ae047f2677.jpg)

![e0e6ade63a462e5555cf99c8ce9805689020dac17f7d706b12af1cb18c1eefc4.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/e0e6ade63a462e5555cf99c8ce9805689020dac17f7d706b12af1cb18c1eefc4.jpg)

![e8dba2127a01bc880ecdd1595037570983cc64b0a1e8df321c7d9609aafcf71c.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/images/e8dba2127a01bc880ecdd1595037570983cc64b0a1e8df321c7d9609aafcf71c.jpg)

### Tables

![0a240c7478bc2651d582d5cc0d5036a13181495653fc5f3d958ffb99dff8fdd6.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/0a240c7478bc2651d582d5cc0d5036a13181495653fc5f3d958ffb99dff8fdd6.jpg)

![0c7f6a5c818b0169fb2f2496201a52e48093155ebe6f813b2ebea0232139bb17.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/0c7f6a5c818b0169fb2f2496201a52e48093155ebe6f813b2ebea0232139bb17.jpg)

![15b5b20d8e15a75fc893aae4f69ab8d5af0b57cd71c4aa34ee3e35d86e08dd6a.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/15b5b20d8e15a75fc893aae4f69ab8d5af0b57cd71c4aa34ee3e35d86e08dd6a.jpg)

![27692a1dadefeae21d539071b23853ea0cee06996efb9156d5e155f95ce4a9b9.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/27692a1dadefeae21d539071b23853ea0cee06996efb9156d5e155f95ce4a9b9.jpg)

![29a619fc0ccefce2fad180881c740d7e3aeca6c202177eeb9dbe1f3058ebaf37.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/29a619fc0ccefce2fad180881c740d7e3aeca6c202177eeb9dbe1f3058ebaf37.jpg)

![3799aeaf71d6ef9a799810267218c129649ff4092b7dd4c188d0376c4addfbbf.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/3799aeaf71d6ef9a799810267218c129649ff4092b7dd4c188d0376c4addfbbf.jpg)

![5dc805fa51603b8a03d16324d4a16ec40fd28a5fd3fe125dfcfab56e2b53b9d0.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/5dc805fa51603b8a03d16324d4a16ec40fd28a5fd3fe125dfcfab56e2b53b9d0.jpg)

![7078bfaacd2c0b24ce70bf0e3183244922a6a4736add2acaa1d19f45ada37557.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/7078bfaacd2c0b24ce70bf0e3183244922a6a4736add2acaa1d19f45ada37557.jpg)

![7657c11808762e76cbc7cdcbea2bd0a46f1465cc5d13b67338e20fcb8532783d.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/7657c11808762e76cbc7cdcbea2bd0a46f1465cc5d13b67338e20fcb8532783d.jpg)

![79c7d78611b83772fd2947329854df62e5661e287bfd21305aac7eefe1367d49.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/79c7d78611b83772fd2947329854df62e5661e287bfd21305aac7eefe1367d49.jpg)

![c4fa03623b61a132d16a7d4e659046d1d0f4d77dea4f0c9f5f449a6f58cee81c.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/c4fa03623b61a132d16a7d4e659046d1d0f4d77dea4f0c9f5f449a6f58cee81c.jpg)

![ce55340e2bfca8047ebe45ef66b9cca36b79ede534f18c446416321d0bc67a09.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/ce55340e2bfca8047ebe45ef66b9cca36b79ede534f18c446416321d0bc67a09.jpg)

![d40f1884d9ee2cd2a41d9d142b49926dbe3d833b7b04cb7f3242f68acc5f2f6e.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/d40f1884d9ee2cd2a41d9d142b49926dbe3d833b7b04cb7f3242f68acc5f2f6e.jpg)

![f59116fe12494015a8a27873cc08820aad0f09023f4ef71aff6429f63d4190cc.jpg](../icml_results/4349_WaveSSM_%20Multiscale%20State-Space%20Models%20for%20Non-stationary%20Signal%20Attention/tables/f59116fe12494015a8a27873cc08820aad0f09023f4ef71aff6429f63d4190cc.jpg)

<a id="paper-4306"></a>

## Decision Tree Learning on Product Spaces


### Images

![84c710b61ba7a461797b0ef7bd538cd52599094aa16b64399ddbb3c2d77d123a.jpg](../icml_results/4350_Decision%20Tree%20Learning%20on%20Product%20Spaces/images/84c710b61ba7a461797b0ef7bd538cd52599094aa16b64399ddbb3c2d77d123a.jpg)

![b34d8063f023d52763a22baed822518589c1e3f6a74f920c79b07a2466986104.jpg](../icml_results/4350_Decision%20Tree%20Learning%20on%20Product%20Spaces/images/b34d8063f023d52763a22baed822518589c1e3f6a74f920c79b07a2466986104.jpg)

![bffecf765b0164e5756c6ebea299efcff6862383dd31f61d73a6d9be076f9bfd.jpg](../icml_results/4350_Decision%20Tree%20Learning%20on%20Product%20Spaces/images/bffecf765b0164e5756c6ebea299efcff6862383dd31f61d73a6d9be076f9bfd.jpg)

<a id="paper-4307"></a>

## Distilling Linearized Behavior into Non-linear Fine-Tuning for Effective Task Arithmetic


### Images

![0073c5e5fff2632bb7715f13ee6e5bc7d9ace98fa785f1b0d5bb893545a461a0.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/0073c5e5fff2632bb7715f13ee6e5bc7d9ace98fa785f1b0d5bb893545a461a0.jpg)

![05f18aead5bdf3acb29a8d929b8131065b6e683f2f9f9d9f939beea660f5fd68.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/05f18aead5bdf3acb29a8d929b8131065b6e683f2f9f9d9f939beea660f5fd68.jpg)

![40aa3a8c165c529858223182811edc5d5a603cea97d400407c4a7018ba7ba1d1.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/40aa3a8c165c529858223182811edc5d5a603cea97d400407c4a7018ba7ba1d1.jpg)

![441336529217e28390eb0924c056b838fae3d1f5bc95c9c632ccae1d38e36e64.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/441336529217e28390eb0924c056b838fae3d1f5bc95c9c632ccae1d38e36e64.jpg)

![4909418f921871057e1327b069123c1f690f3d5e98aef315f81b7776ec3b4fa6.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/4909418f921871057e1327b069123c1f690f3d5e98aef315f81b7776ec3b4fa6.jpg)

![4975cd5025461f6bb8040726b6e8f27464d21dd21ffec9ddbee39201105c312e.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/4975cd5025461f6bb8040726b6e8f27464d21dd21ffec9ddbee39201105c312e.jpg)

![635393be406c51b78dc2ac65d309c6be6fce0bd963678ec2d6291fe266c563d9.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/635393be406c51b78dc2ac65d309c6be6fce0bd963678ec2d6291fe266c563d9.jpg)

![90b156688e1b84de52777d3d4698144d39b2afb2c789dc40811cde4347cd89ba.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/90b156688e1b84de52777d3d4698144d39b2afb2c789dc40811cde4347cd89ba.jpg)

![978c8880ddd863edc4859478433ae8fe035ec88d03e28d1b8984e66604dcfc32.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/978c8880ddd863edc4859478433ae8fe035ec88d03e28d1b8984e66604dcfc32.jpg)

![a9d5c4eccf251dfd329dfffcb548bb4bb7f0856287e75f02da86d52395c13c20.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/a9d5c4eccf251dfd329dfffcb548bb4bb7f0856287e75f02da86d52395c13c20.jpg)

![af93de62024043278a1a86091778c1d0626c457b1b17535a95d9b44a98fa73cf.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/af93de62024043278a1a86091778c1d0626c457b1b17535a95d9b44a98fa73cf.jpg)

![b3650b60235da5007e48c5394a7ac1fbd4ae10ec03c4370c7737206a62f054a9.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/b3650b60235da5007e48c5394a7ac1fbd4ae10ec03c4370c7737206a62f054a9.jpg)

![b8d960f0facd38dc2fa1c6ff8ad3a909280ac14e2136fdf206b6aa07a6c8cbb6.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/b8d960f0facd38dc2fa1c6ff8ad3a909280ac14e2136fdf206b6aa07a6c8cbb6.jpg)

![becb84f8ef57c9e8e6440c7f4b86f7f1be8438a7339f42ae4b13178ed36d4814.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/becb84f8ef57c9e8e6440c7f4b86f7f1be8438a7339f42ae4b13178ed36d4814.jpg)

![c19f8ae2648210ea612f3be596284822b6cfa75ab234d2118494325049535b4c.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/c19f8ae2648210ea612f3be596284822b6cfa75ab234d2118494325049535b4c.jpg)

![c65a3ed935de9620933f26acf593091406b69d27229cadfcd0f7806221f13fbc.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/c65a3ed935de9620933f26acf593091406b69d27229cadfcd0f7806221f13fbc.jpg)

![e68362da2633cd088024afe955b6c180d69d5ccfed5fbcc3f634344ea49c3813.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/e68362da2633cd088024afe955b6c180d69d5ccfed5fbcc3f634344ea49c3813.jpg)

![e794f7016106d00186d18a8855f5dc47daafcd993e66e76315d8ce653ebc79ae.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/images/e794f7016106d00186d18a8855f5dc47daafcd993e66e76315d8ce653ebc79ae.jpg)

### Tables

![1f4d8a50e037d4dd9fda4646cea87863b4f910b53fc21e9df2ed5344770b5699.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/1f4d8a50e037d4dd9fda4646cea87863b4f910b53fc21e9df2ed5344770b5699.jpg)

![436818549715c9eb646abb360bcc63c2e5489ec33d9c777c23db33f76a273123.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/436818549715c9eb646abb360bcc63c2e5489ec33d9c777c23db33f76a273123.jpg)

![8577c4bbe4007b3aafebf063be91998a7e6c5991193b10e40a90b3c4ddf26f03.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/8577c4bbe4007b3aafebf063be91998a7e6c5991193b10e40a90b3c4ddf26f03.jpg)

![a1e0ab594b205dadf8c4e2b275989df2f9409e0e1568f3aac9735ab11c296dd3.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/a1e0ab594b205dadf8c4e2b275989df2f9409e0e1568f3aac9735ab11c296dd3.jpg)

![a5b46ef5b5a17f5b1b71761f8ecd7ea0fea185e7f15c6baa0ce6995ec94339a6.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/a5b46ef5b5a17f5b1b71761f8ecd7ea0fea185e7f15c6baa0ce6995ec94339a6.jpg)

![c3d0fa55ef214a804916633a788b0c91ae3b7a8ca95e14f1c49bd953dd2c298f.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/c3d0fa55ef214a804916633a788b0c91ae3b7a8ca95e14f1c49bd953dd2c298f.jpg)

![d6020aa03bf3463e7abf830b6460f8d6ab4be17f675f65234539b61a63507389.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/d6020aa03bf3463e7abf830b6460f8d6ab4be17f675f65234539b61a63507389.jpg)

![ec419a2925871d92d696350ebd006c98fffd26a54183ad7b21c2a4e941c18a2f.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/ec419a2925871d92d696350ebd006c98fffd26a54183ad7b21c2a4e941c18a2f.jpg)

![f2c8ee34e89f0743e40a33abfd9e3599782f54530c51f7d8322f659734a75248.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/f2c8ee34e89f0743e40a33abfd9e3599782f54530c51f7d8322f659734a75248.jpg)

![fddadebdea8c1ebb2a7ade131ef976a7024295cac3ad4552fde67bc0dac6e36c.jpg](../icml_results/4351_Distilling%20Linearized%20Behavior%20into%20Non-linear%20Fine-Tuning%20for%20Effective%20Task%20Arithmetic/tables/fddadebdea8c1ebb2a7ade131ef976a7024295cac3ad4552fde67bc0dac6e36c.jpg)

<a id="paper-4308"></a>

## Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?


### Images

![1c639006434290d12cbb5e5417e57f9ec19a6cec6baf34b42966fe661acc6bfe.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/1c639006434290d12cbb5e5417e57f9ec19a6cec6baf34b42966fe661acc6bfe.jpg)

![36408c288319244b108385a43ae682da080c3ce2e8a071102372838055ddb626.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/36408c288319244b108385a43ae682da080c3ce2e8a071102372838055ddb626.jpg)

![4053d74f8c572ac9a09badb7024fde48a3edd987caf7e7ceb97f610e0de064b8.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/4053d74f8c572ac9a09badb7024fde48a3edd987caf7e7ceb97f610e0de064b8.jpg)

![4dcda0065848404501ddccbcbf28172132962cf62d61e83eca9aa97b0022103b.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/4dcda0065848404501ddccbcbf28172132962cf62d61e83eca9aa97b0022103b.jpg)

![5e35ebb988bd2e88ec4c8a44ca5b4fdf8a79107cb9a8702b05499c346ea1df45.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/5e35ebb988bd2e88ec4c8a44ca5b4fdf8a79107cb9a8702b05499c346ea1df45.jpg)

![618a04936f5e394c0d757a472ce55d3f73961fbb3d4f34caa5f6571bf79761f9.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/618a04936f5e394c0d757a472ce55d3f73961fbb3d4f34caa5f6571bf79761f9.jpg)

![b5b5075c72ae054128a5e3c0e45b4c4e9ce2b1d113ed5afb4d438bf90aa4a71a.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/b5b5075c72ae054128a5e3c0e45b4c4e9ce2b1d113ed5afb4d438bf90aa4a71a.jpg)

![c08bb67c067f6e2e02b2c3903a5611f88e25c5a60d484537333153a3ac061588.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/c08bb67c067f6e2e02b2c3903a5611f88e25c5a60d484537333153a3ac061588.jpg)

![cb505d7399f255135beea677e4b1e2c41124658806f1d2cb44388b69f1d26f5b.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/cb505d7399f255135beea677e4b1e2c41124658806f1d2cb44388b69f1d26f5b.jpg)

![cd969448990ebc335d359e9383a4a42560f8eef1a842ff34e317037f946b89ca.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/cd969448990ebc335d359e9383a4a42560f8eef1a842ff34e317037f946b89ca.jpg)

![e5cd7b026a68805da97918029f317ac630f0595399559d09fa055bed0412b0fb.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/e5cd7b026a68805da97918029f317ac630f0595399559d09fa055bed0412b0fb.jpg)

![e8c6be497a366dcdfd858275453f63831292dd4bee47ad523ed3909b46ce2117.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/e8c6be497a366dcdfd858275453f63831292dd4bee47ad523ed3909b46ce2117.jpg)

![efe5e370b45b26c542e89dc2174f8e43ddd31f8a049b60a3356095a79c981e43.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/efe5e370b45b26c542e89dc2174f8e43ddd31f8a049b60a3356095a79c981e43.jpg)

![f320ba1dabe3708ec144eece7186adaf90afb7d812af1195ebfbe4ee67b4656e.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/images/f320ba1dabe3708ec144eece7186adaf90afb7d812af1195ebfbe4ee67b4656e.jpg)

### Tables

![0c34b2f019abc9e30afa20b55e9d2e5e63b4c4e6a78018d40262274ba11c4705.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/0c34b2f019abc9e30afa20b55e9d2e5e63b4c4e6a78018d40262274ba11c4705.jpg)

![1a08557f20c85636156692aa3de1e9279110dfd8ae907a9772df23db7baf58fb.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/1a08557f20c85636156692aa3de1e9279110dfd8ae907a9772df23db7baf58fb.jpg)

![1ab0cfa5ddb49141caaf521aee6e27d7f386904f8e4c5ff4dce3f1b08ae5143d.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/1ab0cfa5ddb49141caaf521aee6e27d7f386904f8e4c5ff4dce3f1b08ae5143d.jpg)

![33a7af2e839acebcf814a9f8a0faadd3814bc06b8f728dde31e74aa417515e26.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/33a7af2e839acebcf814a9f8a0faadd3814bc06b8f728dde31e74aa417515e26.jpg)

![3bf902d63840c6278074cb2b38ac58b32a8e29d0700a53a4177641d12c730aca.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/3bf902d63840c6278074cb2b38ac58b32a8e29d0700a53a4177641d12c730aca.jpg)

![51794178e201a18dc5dcdc4d81cf277f126d4ae5c8996723b15abbda4dfd8bc2.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/51794178e201a18dc5dcdc4d81cf277f126d4ae5c8996723b15abbda4dfd8bc2.jpg)

![54f2d3d8060d9f756ecde584499caa0bee516c67d715f38f1c34585b45858008.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/54f2d3d8060d9f756ecde584499caa0bee516c67d715f38f1c34585b45858008.jpg)

![75c8bedb79c49887ef7621323af71fae389e0bdd90dfea7e0d5c8df948394dd3.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/75c8bedb79c49887ef7621323af71fae389e0bdd90dfea7e0d5c8df948394dd3.jpg)

![a9fe223dda058fa395ae1bfc4c24ff24b3224ddab7412657be3d93f3a7376ca7.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/a9fe223dda058fa395ae1bfc4c24ff24b3224ddab7412657be3d93f3a7376ca7.jpg)

![b3013d5ecd4483e115b42ae6d972ebbb9abcdce4820feaccc76b8d0b105efd6e.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/b3013d5ecd4483e115b42ae6d972ebbb9abcdce4820feaccc76b8d0b105efd6e.jpg)

![bb9414614378a021ac42e34d6d4a8c2fbf435db57971c7fec6913db5cfaedf6d.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/bb9414614378a021ac42e34d6d4a8c2fbf435db57971c7fec6913db5cfaedf6d.jpg)

![d64e28c469d408fa5bfb553ecdb92e980a21bd1763dd9259dfd86c1a4f0e5374.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/d64e28c469d408fa5bfb553ecdb92e980a21bd1763dd9259dfd86c1a4f0e5374.jpg)

![e6ea6d0a329da5af3bf283c1853d9f4c32c1a52461b52f542f83c9971aa5320e.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/e6ea6d0a329da5af3bf283c1853d9f4c32c1a52461b52f542f83c9971aa5320e.jpg)

![e8fceb8ffec99f2c4f687c0229e6d2683ba7ed4f679c5ff588b4c829014f2b58.jpg](../icml_results/4352_Can%20Microcanonical%20Langevin%20Dynamics%20Leverage%20Mini-Batch%20Gradient%20Noise_/tables/e8fceb8ffec99f2c4f687c0229e6d2683ba7ed4f679c5ff588b4c829014f2b58.jpg)

<a id="paper-4309"></a>

## Decoupling Variance and Scale-Invariant Updates in Adaptive Gradient Descent for Unified Vector and Matrix Optimization


### Images

![08b4dab039429fb8135ca04cf8432dc48bac4af23f41000a7891f85304f3a503.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/08b4dab039429fb8135ca04cf8432dc48bac4af23f41000a7891f85304f3a503.jpg)

![118f08bc5ef375570021d6493b9ed3932c079e5456a6c0752761a645aa11b42c.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/118f08bc5ef375570021d6493b9ed3932c079e5456a6c0752761a645aa11b42c.jpg)

![12238cdf7e95b39334061e70504e380362ac87d25cc7c99588cdbddbad3fe17c.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/12238cdf7e95b39334061e70504e380362ac87d25cc7c99588cdbddbad3fe17c.jpg)

![285287e972a5a4185271b6bb4ff2d7a921d44cf27aabca88cee0c398c3e26f92.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/285287e972a5a4185271b6bb4ff2d7a921d44cf27aabca88cee0c398c3e26f92.jpg)

![5f21db4e2abc02977181c42247d285328bac0f84380e3a076e16a3739617d3ee.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/5f21db4e2abc02977181c42247d285328bac0f84380e3a076e16a3739617d3ee.jpg)

![61136a6dc554187ec9b0da0f11d85d09e28d57fd6a8c5afbc69cb1a6b9e4ef57.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/61136a6dc554187ec9b0da0f11d85d09e28d57fd6a8c5afbc69cb1a6b9e4ef57.jpg)

![a94477b7f0890e26345bc7fa036ef2cdeae3759e150f8744b75b2f54c519bc82.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/a94477b7f0890e26345bc7fa036ef2cdeae3759e150f8744b75b2f54c519bc82.jpg)

![c75c1157f706e3ed16b3ea0c85636ea90c30c8f489d04f91b8e885725c8efe9d.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/c75c1157f706e3ed16b3ea0c85636ea90c30c8f489d04f91b8e885725c8efe9d.jpg)

![e5f3691c765891d98bd35c0d8caf2378557ef3e8d15a026bcc1875ce48f19130.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/images/e5f3691c765891d98bd35c0d8caf2378557ef3e8d15a026bcc1875ce48f19130.jpg)

### Tables

![06beddf4f77e46edbbc6dc8d631a524b1000764a78fe23af6914885d1e7dd1d8.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/06beddf4f77e46edbbc6dc8d631a524b1000764a78fe23af6914885d1e7dd1d8.jpg)

![0e2b06dc0bfdfab008d797abda4e2ef6d021b61d89eda4d80cc9837a49d41aad.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/0e2b06dc0bfdfab008d797abda4e2ef6d021b61d89eda4d80cc9837a49d41aad.jpg)

![40e5f8d0f906521c7b046c1289794002f263a0821f06eb90e2188651a1f76802.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/40e5f8d0f906521c7b046c1289794002f263a0821f06eb90e2188651a1f76802.jpg)

![4198487c8881d14c91c2741366e402b48333b8fda13330c2072b3f398b6849b1.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/4198487c8881d14c91c2741366e402b48333b8fda13330c2072b3f398b6849b1.jpg)

![619e7b182e423f0e53c4988f56652ca6a0b9b057f05d6efdb68c3fda9818743c.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/619e7b182e423f0e53c4988f56652ca6a0b9b057f05d6efdb68c3fda9818743c.jpg)

![64e29132a9dfd83fe333139356233bca7a4ba1ea03850e78019d5b9f19c08418.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/64e29132a9dfd83fe333139356233bca7a4ba1ea03850e78019d5b9f19c08418.jpg)

![9eb4fd63cba81ecadec98ac91217122080d840e38356647b6dcc527d27581171.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/9eb4fd63cba81ecadec98ac91217122080d840e38356647b6dcc527d27581171.jpg)

![b63718432c6a4edb77c567817a126a3d0b2720f4779fabacb3e0174175590460.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/b63718432c6a4edb77c567817a126a3d0b2720f4779fabacb3e0174175590460.jpg)

![bd8a7af2360fe6bc5b425e01650f513d099b115d8d9519e4f4e06be88ed892e0.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/bd8a7af2360fe6bc5b425e01650f513d099b115d8d9519e4f4e06be88ed892e0.jpg)

![f3e62dc942cdc39de92dc6cf284a6f97e49e37398e723edee86e5db77bcb9129.jpg](../icml_results/4353_Decoupling%20Variance%20and%20Scale-Invariant%20Updates%20in%20Adaptive%20Gradient%20Descent%20for%20Unified%20Vector%20and%20/tables/f3e62dc942cdc39de92dc6cf284a6f97e49e37398e723edee86e5db77bcb9129.jpg)

<a id="paper-4310"></a>

## EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors


### Images

![65d0299dae87b605447f63f46d816d85650a9b39cc4190f8adc8b131eb4c25d6.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/65d0299dae87b605447f63f46d816d85650a9b39cc4190f8adc8b131eb4c25d6.jpg)

![82c7b413e9a369734758cda8b98b756b278a8f46c5ad28d811104aa8628c5e57.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/82c7b413e9a369734758cda8b98b756b278a8f46c5ad28d811104aa8628c5e57.jpg)

![a122def1c3b8164289a6bf704873c756e5ef7ab9e0d64f94c94b6570d1151e8c.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/a122def1c3b8164289a6bf704873c756e5ef7ab9e0d64f94c94b6570d1151e8c.jpg)

![a2845b1033d8983ee4de7bf7f48546bd5a2aae7863e5a94314709f4cd0a2c1cd.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/a2845b1033d8983ee4de7bf7f48546bd5a2aae7863e5a94314709f4cd0a2c1cd.jpg)

![aeee151aaaba2e3d92dc42d814e588c00033a5750de269524d6468d3c045696d.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/aeee151aaaba2e3d92dc42d814e588c00033a5750de269524d6468d3c045696d.jpg)

![cbc95d593fb66efacd8e78bc493cee505b16499f2497b31deec302c0eb06ba0a.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/cbc95d593fb66efacd8e78bc493cee505b16499f2497b31deec302c0eb06ba0a.jpg)

![e311f481a3342dbfccc23df7cd61506ad28fdf4a2b3e948b631ec0bc7925323f.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/e311f481a3342dbfccc23df7cd61506ad28fdf4a2b3e948b631ec0bc7925323f.jpg)

![e5639db4f7d83944ea7d19a46ca026ece91dc69f2c25d2f681cda8e347416611.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/images/e5639db4f7d83944ea7d19a46ca026ece91dc69f2c25d2f681cda8e347416611.jpg)

### Tables

![0c8ef53769badc1b4275a71de4af2e6aa6c132755df1ef859273eaff4fbcdf15.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/0c8ef53769badc1b4275a71de4af2e6aa6c132755df1ef859273eaff4fbcdf15.jpg)

![0f84ecdd984fbdb3f2aa207f71215845a5c36dd07f696587b3dec363bcfadc36.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/0f84ecdd984fbdb3f2aa207f71215845a5c36dd07f696587b3dec363bcfadc36.jpg)

![2e4a46310499312ce7cdd3cffbd39ae22bfcdd5f8cee4f8b6187bae01b83d3ba.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/2e4a46310499312ce7cdd3cffbd39ae22bfcdd5f8cee4f8b6187bae01b83d3ba.jpg)

![9331ccc75e93aa2b5a7ea9d4b477e1432b9c17b7896027669e8bf5c167a2dbb1.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/9331ccc75e93aa2b5a7ea9d4b477e1432b9c17b7896027669e8bf5c167a2dbb1.jpg)

![960daefbae7e85f32f9eb60b9dfff22d9b58e71152bd5f79276058e41bd07331.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/960daefbae7e85f32f9eb60b9dfff22d9b58e71152bd5f79276058e41bd07331.jpg)

![aa17037e0322f15c64962942423621244a27be5bc53e1034c7155c8f7a07dd1e.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/aa17037e0322f15c64962942423621244a27be5bc53e1034c7155c8f7a07dd1e.jpg)

![b6fde938e73913298f60ea168d58ceb846d77c022659c0294eec97cecb05b3a9.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/b6fde938e73913298f60ea168d58ceb846d77c022659c0294eec97cecb05b3a9.jpg)

![cacf36f70e96d21ac1a5665c4cbe85c8cf9c8a3862e0162d60e4ad035b70bb3c.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/cacf36f70e96d21ac1a5665c4cbe85c8cf9c8a3862e0162d60e4ad035b70bb3c.jpg)

![e1f286d07feb287035703a0f0782a2494dc3071a50616a47594043c058e7af6a.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/e1f286d07feb287035703a0f0782a2494dc3071a50616a47594043c058e7af6a.jpg)

![fa7fafa4a080a0d92b8dd9579892a1ad9b7935e9efd9300330841e394764578b.jpg](../icml_results/4354_EnerGS_%20Energy-Based%20Gaussian%20Splatting%20with%20Partial%20Geometric%20Priors/tables/fa7fafa4a080a0d92b8dd9579892a1ad9b7935e9efd9300330841e394764578b.jpg)

<a id="paper-4311"></a>

## Expanding the Capabilities of Reinforcement Learning via Text Feedback


### Images

![01da44a357d229a47d8261f7e4c07bdb592b59666c950aca9deca81d6334402a.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/images/01da44a357d229a47d8261f7e4c07bdb592b59666c950aca9deca81d6334402a.jpg)

![9602155f855a88bdd54738de902078b7eb7d815ad861997295aed0b74b2b3b82.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/images/9602155f855a88bdd54738de902078b7eb7d815ad861997295aed0b74b2b3b82.jpg)

![e9416a683e620bbe3bbbf5858667b6a92669260267bdf683f3e3ddc973a7753f.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/images/e9416a683e620bbe3bbbf5858667b6a92669260267bdf683f3e3ddc973a7753f.jpg)

![e99099bb1f58873d9e4783e922ad6e377da32e42d1d08b881885d6243cd308c2.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/images/e99099bb1f58873d9e4783e922ad6e377da32e42d1d08b881885d6243cd308c2.jpg)

![fddd079bc23c5abd909e096e12b06c0d5c584aac21a6a0a3cf17cb95ac9bf135.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/images/fddd079bc23c5abd909e096e12b06c0d5c584aac21a6a0a3cf17cb95ac9bf135.jpg)

### Tables

![00723c86b240462352b6ddb3a2b8f7dc47c0368f59464c510e95e8066cf4c812.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/tables/00723c86b240462352b6ddb3a2b8f7dc47c0368f59464c510e95e8066cf4c812.jpg)

![661203cdc0d93027cf7cf397bc33b2943b71fcaca681849ff37928704b7c7910.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/tables/661203cdc0d93027cf7cf397bc33b2943b71fcaca681849ff37928704b7c7910.jpg)

![8c983831d04594d1ab0618b9cc910d8c637e980e85135793438ecf13947b5953.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/tables/8c983831d04594d1ab0618b9cc910d8c637e980e85135793438ecf13947b5953.jpg)

![b9cbc4335faa82c0a0aa92d24e62b930ecb430d76be9bed0682eb2897866687d.jpg](../icml_results/4355_Expanding%20the%20Capabilities%20of%20Reinforcement%20Learning%20via%20Text%20Feedback/tables/b9cbc4335faa82c0a0aa92d24e62b930ecb430d76be9bed0682eb2897866687d.jpg)

<a id="paper-4312"></a>

## When Diffusion Language Models Hesitate: Detecting and Correcting Visual Hallucinations via Confidence Fluctuation


### Images

![218ab46004c5702c98280cfc27b16c9f856b05cceb98e3338495a8be216f7723.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/218ab46004c5702c98280cfc27b16c9f856b05cceb98e3338495a8be216f7723.jpg)

![36842af0d44ed1ac7b6bd67dea9cf774d20c20f50b4b908816623e8b77d6fa2d.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/36842af0d44ed1ac7b6bd67dea9cf774d20c20f50b4b908816623e8b77d6fa2d.jpg)

![62c9a4ee344f1f5b422cd5fd32a7a3551f443ef11395d970f52bbd0e61b10a5f.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/62c9a4ee344f1f5b422cd5fd32a7a3551f443ef11395d970f52bbd0e61b10a5f.jpg)

![7e1d4b5c1c39003bd66926a6685565d828fa728c0bb49ae80cc93bee03f8587b.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/7e1d4b5c1c39003bd66926a6685565d828fa728c0bb49ae80cc93bee03f8587b.jpg)

![82cf13fa7229aaf84dbf818e8afd6f3a51b64fee96199896a9da1b02f2c04cd5.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/82cf13fa7229aaf84dbf818e8afd6f3a51b64fee96199896a9da1b02f2c04cd5.jpg)

![875d1ef9a251c46bfefde57a3e2960f16e0efa95540982fa603cc128472c1282.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/875d1ef9a251c46bfefde57a3e2960f16e0efa95540982fa603cc128472c1282.jpg)

![8cfa59a3773cbe1e237f40e773892e396ccfc9a285dc0c3f6618d1b22072bb7c.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/8cfa59a3773cbe1e237f40e773892e396ccfc9a285dc0c3f6618d1b22072bb7c.jpg)

![90fa3e7eedafdece70c10d0b0bbd629b9df05c896e3ea9daf61770bca1c1f1fc.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/90fa3e7eedafdece70c10d0b0bbd629b9df05c896e3ea9daf61770bca1c1f1fc.jpg)

![b7c770745a4267173770990b1c13c46bd5209762d1693d35fda0c35ffa463d1b.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/b7c770745a4267173770990b1c13c46bd5209762d1693d35fda0c35ffa463d1b.jpg)

![cf0754c18c2d52a2d1e1c141034973338f68f29524621d1b5ce0a68a2a6a87ad.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/cf0754c18c2d52a2d1e1c141034973338f68f29524621d1b5ce0a68a2a6a87ad.jpg)

![d5e18e2c46b54cff72e993c9304ba96de922dd5a51ce2af0a8493295ce0ccfb2.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/images/d5e18e2c46b54cff72e993c9304ba96de922dd5a51ce2af0a8493295ce0ccfb2.jpg)

### Tables

![15647b3f7948d4d5182fe509f1727670a4293989347f2ef6d5f323685051b84f.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/15647b3f7948d4d5182fe509f1727670a4293989347f2ef6d5f323685051b84f.jpg)

![254f2b3912ae22ccfcd950bc8e01d5c7d962e39aafc1be3ed355b5bff9d9c8f4.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/254f2b3912ae22ccfcd950bc8e01d5c7d962e39aafc1be3ed355b5bff9d9c8f4.jpg)

![3566bf16a544323e12fa2a2e5c782d55defc5bad776b0fd612a578680ed5799c.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/3566bf16a544323e12fa2a2e5c782d55defc5bad776b0fd612a578680ed5799c.jpg)

![4143fb0e10d7798523e79fd687cc9ee68643428684d3371110334d77f773b8ae.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/4143fb0e10d7798523e79fd687cc9ee68643428684d3371110334d77f773b8ae.jpg)

![43144a6566350c2efdadc016ee12b21ab4be06711a9922b041beddf402ee5238.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/43144a6566350c2efdadc016ee12b21ab4be06711a9922b041beddf402ee5238.jpg)

![5787351bddcfd7de638fde3d4d4eac791b9955a306da324c6098f0ce5d07524a.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/5787351bddcfd7de638fde3d4d4eac791b9955a306da324c6098f0ce5d07524a.jpg)

![84c14c4f7143ad1c6268d7f9b4305e3752d662fa55d5aae2faed940bab7efa44.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/84c14c4f7143ad1c6268d7f9b4305e3752d662fa55d5aae2faed940bab7efa44.jpg)

![86bd8dc39b274e8bac6c7695101f7c79f19b2f9ed1bff1b33b2944477ba33448.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/86bd8dc39b274e8bac6c7695101f7c79f19b2f9ed1bff1b33b2944477ba33448.jpg)

![9ffcd11e1822a2ec17f35671a0cc342a9064ece46dbb8c7246d29bf76572321c.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/9ffcd11e1822a2ec17f35671a0cc342a9064ece46dbb8c7246d29bf76572321c.jpg)

![aba602ccf44e46d9d6000a9399a4195656086a477ec2ec6371eb23a558cfb543.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/aba602ccf44e46d9d6000a9399a4195656086a477ec2ec6371eb23a558cfb543.jpg)

![b10e92d48b84d678cb7501288c9dfb8d21fb95adaa98e35eaf0006288fe4c63a.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/b10e92d48b84d678cb7501288c9dfb8d21fb95adaa98e35eaf0006288fe4c63a.jpg)

![b305b8b2be1660246fc84f2360eaa72422663901d29bb786938072948265dbd5.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/b305b8b2be1660246fc84f2360eaa72422663901d29bb786938072948265dbd5.jpg)

![ca5f4052e14696e508a8cc005407f38aeea9b9e8a06b6a0336d2be188d3d85af.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/ca5f4052e14696e508a8cc005407f38aeea9b9e8a06b6a0336d2be188d3d85af.jpg)

![f5e9a96ee7802d4d9050bd88d3cde65b76bd21f481194e940984ebeec923ffc4.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/f5e9a96ee7802d4d9050bd88d3cde65b76bd21f481194e940984ebeec923ffc4.jpg)

![f6d2e953de87ef6df9492149f4b2da542be68eb7ac0e6d14454f9cc1cf4f28bc.jpg](../icml_results/4356_When%20Diffusion%20Language%20Models%20Hesitate_%20Detecting%20and%20Correcting%20Visual%20Hallucinations%20via%20Confiden/tables/f6d2e953de87ef6df9492149f4b2da542be68eb7ac0e6d14454f9cc1cf4f28bc.jpg)

<a id="paper-4313"></a>

## EnsembleVLA: Ensemble Learning for Vision-Language Action Models


### Images

![0475e4c2eb7fd1af2ee62999673b0ad6090bfb789b368b8c1bb5d03ff438059e.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/0475e4c2eb7fd1af2ee62999673b0ad6090bfb789b368b8c1bb5d03ff438059e.jpg)

![165872ebfff4249a8066600ca38417a1f5b5b8011a14270ce396f5afc5433e34.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/165872ebfff4249a8066600ca38417a1f5b5b8011a14270ce396f5afc5433e34.jpg)

![1817001dc1e05da91ea0fbea2ce2d10cee8ecc92fc3e094d91a6e750ca0c9d3c.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/1817001dc1e05da91ea0fbea2ce2d10cee8ecc92fc3e094d91a6e750ca0c9d3c.jpg)

![415d5c4b06958efdd1df3b846bfb00a4c0888df814ee2eaf9c6aa72965436a55.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/415d5c4b06958efdd1df3b846bfb00a4c0888df814ee2eaf9c6aa72965436a55.jpg)

![6cdfe4facb3c7a0f2fc5978f9d9b86f90ae34bd3e246cfd830309db93cc65a71.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/6cdfe4facb3c7a0f2fc5978f9d9b86f90ae34bd3e246cfd830309db93cc65a71.jpg)

![791b03feb1b55c1b7458a13e403597176b38a4128ff77655529467a2d8b48d99.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/791b03feb1b55c1b7458a13e403597176b38a4128ff77655529467a2d8b48d99.jpg)

![9a9424e9ee287c847b4360696d6518ead08b537504bb9e19fdc2488608d06bd0.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/9a9424e9ee287c847b4360696d6518ead08b537504bb9e19fdc2488608d06bd0.jpg)

![b9937fc82c4818c21f013af2fe99ee53f4c51f161c50747013d69ab52e6a21be.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/b9937fc82c4818c21f013af2fe99ee53f4c51f161c50747013d69ab52e6a21be.jpg)

![ebead73d22683a2123033eb85d712f92a3daa1723efd8e03efaccb8a8f56bc2c.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/images/ebead73d22683a2123033eb85d712f92a3daa1723efd8e03efaccb8a8f56bc2c.jpg)

### Tables

![56cbb492d62880208664be52144602c35cc90d79920b831a3e2cef07b8a6e836.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/tables/56cbb492d62880208664be52144602c35cc90d79920b831a3e2cef07b8a6e836.jpg)

![99d1de73e78a8b7559671b72cc960bc314fa433c87880ece1de922e3e04bad9b.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/tables/99d1de73e78a8b7559671b72cc960bc314fa433c87880ece1de922e3e04bad9b.jpg)

![d20be92a3f2675e1ae8b44df34cb752acdb20d77233a64d399a0c19bd6b35b8c.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/tables/d20be92a3f2675e1ae8b44df34cb752acdb20d77233a64d399a0c19bd6b35b8c.jpg)

![d4e5b3bce91b53ab033ad2c83caa84cf212cc5e30717108046446bfce23dac94.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/tables/d4e5b3bce91b53ab033ad2c83caa84cf212cc5e30717108046446bfce23dac94.jpg)

![f29f195347f43f3d90a5d8e4862fb14dd3bcd6392d8d1de1a3647df01018f185.jpg](../icml_results/4357_EnsembleVLA_%20Ensemble%20Learning%20for%20Vision-Language%20Action%20Models/tables/f29f195347f43f3d90a5d8e4862fb14dd3bcd6392d8d1de1a3647df01018f185.jpg)

<a id="paper-4314"></a>

## RNA-FM: Flow-Matching Generative Model for Genome-wide RNA-Seq Prediction


### Images

![34e510c8c63667443cde93fc4af118db6de1e9dc2e639c3935ae3da9699ec4b5.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/images/34e510c8c63667443cde93fc4af118db6de1e9dc2e639c3935ae3da9699ec4b5.jpg)

![6e2b1b54d1952fdd04bccdbb554b9c531046f352fcaae339ad2af6cd3258e22b.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/images/6e2b1b54d1952fdd04bccdbb554b9c531046f352fcaae339ad2af6cd3258e22b.jpg)

![6febe6dea82eba71bb9c0356e7a08a6ca4e8e3c195c38f12000965092efce3a7.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/images/6febe6dea82eba71bb9c0356e7a08a6ca4e8e3c195c38f12000965092efce3a7.jpg)

### Tables

![0f260da3fca251821ac61612223255d70e67e4e6329e3890d3ff6c6901705557.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/0f260da3fca251821ac61612223255d70e67e4e6329e3890d3ff6c6901705557.jpg)

![11f4ab4bf9d869736e138083f9885298cd11510bfd04154da18494a9ed269f1e.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/11f4ab4bf9d869736e138083f9885298cd11510bfd04154da18494a9ed269f1e.jpg)

![24ba7d18c6ab58fd24f052b32356ac0502f011a46dd02f7ed19d54e677314e55.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/24ba7d18c6ab58fd24f052b32356ac0502f011a46dd02f7ed19d54e677314e55.jpg)

![5cdeff4a051cdb4ccda7b6555c348d6354224e6554255bfec973c90755ecf829.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/5cdeff4a051cdb4ccda7b6555c348d6354224e6554255bfec973c90755ecf829.jpg)

![8bbf7189b37b3d93d743e6360c7211729101ab383bc114c8f171005bf2997541.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/8bbf7189b37b3d93d743e6360c7211729101ab383bc114c8f171005bf2997541.jpg)

![91ac8e3a7e5440c8b373c7e71b8508da0b15975846edb027d9209a29ae0ca83b.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/91ac8e3a7e5440c8b373c7e71b8508da0b15975846edb027d9209a29ae0ca83b.jpg)

![9eea8a2f6298093694de7ee9b85774f3b20258d3b0cea38ae554a74006caffe7.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/9eea8a2f6298093694de7ee9b85774f3b20258d3b0cea38ae554a74006caffe7.jpg)

![a06c1d7055c44b50d021d65c29735d966f16a6ee08ae336d9482fe004c28713c.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/a06c1d7055c44b50d021d65c29735d966f16a6ee08ae336d9482fe004c28713c.jpg)

![a83b8a12b8b4df53985211af21157fd74ea68278abb8adae52de42f530287f91.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/a83b8a12b8b4df53985211af21157fd74ea68278abb8adae52de42f530287f91.jpg)

![aa99703a9d9363e86f4a1155aca2d337d125b869810f1361638ae35a989f8034.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/aa99703a9d9363e86f4a1155aca2d337d125b869810f1361638ae35a989f8034.jpg)

![c30ee23d10f330fcc76a22d13aff2f0bf3ccdfa088b1863b5b7e91fdbdee194e.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/c30ee23d10f330fcc76a22d13aff2f0bf3ccdfa088b1863b5b7e91fdbdee194e.jpg)

![c3b49815bb461c07aa0005377b3b10d945e27bf8428a1315a2d801e3fe26d197.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/c3b49815bb461c07aa0005377b3b10d945e27bf8428a1315a2d801e3fe26d197.jpg)

![dde05f29f4bdbc315614a29e346559ceb879c8f157ad061bd8a40cd0569c2dbc.jpg](../icml_results/4358_RNA-FM_%20Flow-Matching%20Generative%20Model%20for%20Genome-wide%20RNA-Seq%20Prediction/tables/dde05f29f4bdbc315614a29e346559ceb879c8f157ad061bd8a40cd0569c2dbc.jpg)

<a id="paper-4315"></a>

## Estimating the Empowerment of Language Model Agents


### Images

![04679135eaa0819e46e8bce109645f107698d90fd6e3989abc2358dda45727e6.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/04679135eaa0819e46e8bce109645f107698d90fd6e3989abc2358dda45727e6.jpg)

![245fb516e004f79a545d88bb09b690ef0a85a8881768bf79838ea554ff7df071.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/245fb516e004f79a545d88bb09b690ef0a85a8881768bf79838ea554ff7df071.jpg)

![373e2313bc699a163502e35dec5529917c59a4fd02b88bd7967589e10781f000.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/373e2313bc699a163502e35dec5529917c59a4fd02b88bd7967589e10781f000.jpg)

![65a5fb4a42a7a515b52e2a0b86d8f72e4932f3a167a25debde6476f9b5ad64e3.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/65a5fb4a42a7a515b52e2a0b86d8f72e4932f3a167a25debde6476f9b5ad64e3.jpg)

![6c1b3fa7432be44b5d930c61d4665297ed6d3d678bacdf04f98d2655e9b5ce16.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/6c1b3fa7432be44b5d930c61d4665297ed6d3d678bacdf04f98d2655e9b5ce16.jpg)

![75c76c67e15592ea0e29cd3e68bdc39eec943d66c5d17113d9f9962436773543.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/75c76c67e15592ea0e29cd3e68bdc39eec943d66c5d17113d9f9962436773543.jpg)

![7a5e7b544579c71422e92104ba0e0bceac46604ce08ac271917894aa0fff1638.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/7a5e7b544579c71422e92104ba0e0bceac46604ce08ac271917894aa0fff1638.jpg)

![7c2e5d89a09e44435be1ca31f8ca7f8d6d070c8208de192f1f3a48fff10bde14.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/7c2e5d89a09e44435be1ca31f8ca7f8d6d070c8208de192f1f3a48fff10bde14.jpg)

![85fe3af9ec3e8c6e0a160b9bdb126f0d219a260ab5d63fdf7d43efa69ec64337.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/85fe3af9ec3e8c6e0a160b9bdb126f0d219a260ab5d63fdf7d43efa69ec64337.jpg)

![9bd921e2559480bc5eb590b37f7ef95d7279ac4abf861b8ed67ddfb0f9747e2c.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/9bd921e2559480bc5eb590b37f7ef95d7279ac4abf861b8ed67ddfb0f9747e2c.jpg)

![a1ba1d82441d7bdb119353b83344af511f922bf14e78966ad8990a7d8d780d4b.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/a1ba1d82441d7bdb119353b83344af511f922bf14e78966ad8990a7d8d780d4b.jpg)

![a6709ddb1688dd3d1a43e6d3c22599baeb2bd07b3903dff6ad568002c06f1d61.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/a6709ddb1688dd3d1a43e6d3c22599baeb2bd07b3903dff6ad568002c06f1d61.jpg)

![a7ec66c4de4aedd94b555359b8f61d2f9b31c10ac413fd8359753b5be13edd8e.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/a7ec66c4de4aedd94b555359b8f61d2f9b31c10ac413fd8359753b5be13edd8e.jpg)

![b61f956087de1e6e968c60a725311b497a4665c432fb2347814acd662486e74f.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/b61f956087de1e6e968c60a725311b497a4665c432fb2347814acd662486e74f.jpg)

![bdf145a1e79e3588a5c03df496bf2c7b6176d7f5ca4c9b4b49c5c6b8249d9d55.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/bdf145a1e79e3588a5c03df496bf2c7b6176d7f5ca4c9b4b49c5c6b8249d9d55.jpg)

![d5a215467b75ad6ba585372dc1572ccfa5610dca6d789d881997809d6a7e5f9a.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/d5a215467b75ad6ba585372dc1572ccfa5610dca6d789d881997809d6a7e5f9a.jpg)

![d7325fee8b25d2eb24acb81ac9e5427bd592b4077fca16ec5f51e7ee39bf584d.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/d7325fee8b25d2eb24acb81ac9e5427bd592b4077fca16ec5f51e7ee39bf584d.jpg)

![dd7e804504161407191a71f89d52ce082082d2bf6c2d67818e4341464b7e35fd.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/dd7e804504161407191a71f89d52ce082082d2bf6c2d67818e4341464b7e35fd.jpg)

![e7ef2a67bfbf423aa0058bfdd68c0a32a4cb8fbbc746819423ef288bd108fbe9.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/e7ef2a67bfbf423aa0058bfdd68c0a32a4cb8fbbc746819423ef288bd108fbe9.jpg)

![ec3e9f47c2680f61a60bc56bcb5aa0fd20de8ce7e5961b2f4181d70eab4f0c64.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/ec3e9f47c2680f61a60bc56bcb5aa0fd20de8ce7e5961b2f4181d70eab4f0c64.jpg)

![f9459ff521da9243dcc824a67688a0e392e5ed136a38cc93b74190ce8646d20a.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/f9459ff521da9243dcc824a67688a0e392e5ed136a38cc93b74190ce8646d20a.jpg)

![f94b4b94d8dc9c1ce93b808f6864cdab94920610146d643845fbffb2dc53721f.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/images/f94b4b94d8dc9c1ce93b808f6864cdab94920610146d643845fbffb2dc53721f.jpg)

### Tables

![2c4cae7c976c55826d81f01457e1887f44d78e7e90bbda2655bf80210ab16e40.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/2c4cae7c976c55826d81f01457e1887f44d78e7e90bbda2655bf80210ab16e40.jpg)

![52a350120d80117653a0b9aae24eaccef730977e272017df29bf53a5a6189153.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/52a350120d80117653a0b9aae24eaccef730977e272017df29bf53a5a6189153.jpg)

![6f386d4d6245c1d7c781591c28c1f80d5be33130c631f601525d94dbc7cd34f4.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/6f386d4d6245c1d7c781591c28c1f80d5be33130c631f601525d94dbc7cd34f4.jpg)

![75e4d40f54a86d3dff6fb1abfe1cafd25a18371f8b45db7317ea9f09472c72b1.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/75e4d40f54a86d3dff6fb1abfe1cafd25a18371f8b45db7317ea9f09472c72b1.jpg)

![775b18ae1135149ce3beedd6a003c4f5fdde59dc613b81d7651d75ab43f4735c.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/775b18ae1135149ce3beedd6a003c4f5fdde59dc613b81d7651d75ab43f4735c.jpg)

![7de7dbada20822fee4468c6b49b9b63ecaee28da7818e7ec54189bdc71065dd3.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/7de7dbada20822fee4468c6b49b9b63ecaee28da7818e7ec54189bdc71065dd3.jpg)

![bf0f6329a5c407571ebbec87fe28246d7a9ae77adff4728e376238ccf140de51.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/bf0f6329a5c407571ebbec87fe28246d7a9ae77adff4728e376238ccf140de51.jpg)

![cfc00d57e1a2d47389991262f8a40d27803557471f839f88e5f89d8b21a9697d.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/cfc00d57e1a2d47389991262f8a40d27803557471f839f88e5f89d8b21a9697d.jpg)

![d8be5f785b52772c813a2b25fcfec15a636f8fd01060f51a76d7ecbc27af3227.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/d8be5f785b52772c813a2b25fcfec15a636f8fd01060f51a76d7ecbc27af3227.jpg)

![f5e648a66245774abadb1b19aefcd3bff8f66aef93a036c8a95aa00ff1781d12.jpg](../icml_results/4359_Estimating%20the%20Empowerment%20of%20Language%20Model%20Agents/tables/f5e648a66245774abadb1b19aefcd3bff8f66aef93a036c8a95aa00ff1781d12.jpg)

<a id="paper-4316"></a>

## LoBCD-GW: A Fast and Data-Dependent Algorithm for Computing Gromov-Wasserstein Distance via Localized Block Coordinate Descent


### Images

![2ccc533643dbdf8d76b32209f09a3370f92d5c6e7b0b2fc51c4b7edd7f79c811.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/images/2ccc533643dbdf8d76b32209f09a3370f92d5c6e7b0b2fc51c4b7edd7f79c811.jpg)

![301e0c978c5dd18aa935134c0353ae20113a43d728622c06e00a865fa6d25e21.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/images/301e0c978c5dd18aa935134c0353ae20113a43d728622c06e00a865fa6d25e21.jpg)

![49dcfd17ca08c5cf930a7746ddbf77e39b69ea406e0059213d13615b6e88bbd4.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/images/49dcfd17ca08c5cf930a7746ddbf77e39b69ea406e0059213d13615b6e88bbd4.jpg)

![590adaee521646f8bcb919d5d4d0460a9a02f8242de44747f5f37430ab6adcec.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/images/590adaee521646f8bcb919d5d4d0460a9a02f8242de44747f5f37430ab6adcec.jpg)

![ffe6efec618b394f9f0407b153015ec2d70795ddca192c644b7b046660d9b1f6.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/images/ffe6efec618b394f9f0407b153015ec2d70795ddca192c644b7b046660d9b1f6.jpg)

### Tables

![034d54584c185c5566fb8fd28eaa78bb50fa6f0afccb5b69d80dfb04cff4acb6.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/034d54584c185c5566fb8fd28eaa78bb50fa6f0afccb5b69d80dfb04cff4acb6.jpg)

![120f465c88a193f25bbc288cc2ec7abc575c0f8594c0a5cac6ed109f4e03499a.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/120f465c88a193f25bbc288cc2ec7abc575c0f8594c0a5cac6ed109f4e03499a.jpg)

![341a62edd0294e0c773ee6f8483fba300ef5ff33549ed941b624ba51e7525b8f.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/341a62edd0294e0c773ee6f8483fba300ef5ff33549ed941b624ba51e7525b8f.jpg)

![558c1fed27785c9d49f328667d917cb1647fd9b59ab37492ba13f5d61918c716.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/558c1fed27785c9d49f328667d917cb1647fd9b59ab37492ba13f5d61918c716.jpg)

![9d27ca2ba34102f8a02fde5097e13e8213f01c3c0e3ae0cebf785ecae54a2a28.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/9d27ca2ba34102f8a02fde5097e13e8213f01c3c0e3ae0cebf785ecae54a2a28.jpg)

![aa80aeff8dfb450e8bb4eef235a472d1b1fb87dc058af35690f7e6bf36d105ff.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/aa80aeff8dfb450e8bb4eef235a472d1b1fb87dc058af35690f7e6bf36d105ff.jpg)

![ae4ff08e37e4ba75901c80a2d028f9a6d9c0c0560540055a5eafad758e427a1e.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/ae4ff08e37e4ba75901c80a2d028f9a6d9c0c0560540055a5eafad758e427a1e.jpg)

![cdc7686409a8f0c4c66c6a5d41e3a1841431db1f8b95cd1b7f8af584c16617d3.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/cdc7686409a8f0c4c66c6a5d41e3a1841431db1f8b95cd1b7f8af584c16617d3.jpg)

![e54aef055b4009d91949a1cdb680c4d2e0ad8f113abc502f5d0959713f66de2d.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/e54aef055b4009d91949a1cdb680c4d2e0ad8f113abc502f5d0959713f66de2d.jpg)

![e8ba4c20f44f9868880bafdcfaed6b82ad398ed6f3de161bddddf9b55aae2717.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/e8ba4c20f44f9868880bafdcfaed6b82ad398ed6f3de161bddddf9b55aae2717.jpg)

![e93885c1a9c9393477bf05624c7ea6d252eac33a3d39185a534404718c2dd8ab.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/e93885c1a9c9393477bf05624c7ea6d252eac33a3d39185a534404718c2dd8ab.jpg)

![f4da946fe619605ad83a4973c278a3e0caf76ab0d1b2bc8219554ad502bff884.jpg](../icml_results/4360_LoBCD-GW_%20A%20Fast%20and%20Data-Dependent%20Algorithm%20for%20Computing%20Gromov-Wasserstein%20Distance%20via%20Localize/tables/f4da946fe619605ad83a4973c278a3e0caf76ab0d1b2bc8219554ad502bff884.jpg)

<a id="paper-4317"></a>

## Kinematics-Driven Gaussian Shape Deformation for Blurry Monocular Dynamic Scenes


### Images

![042918edd367ea3cabf23efe8d2021c956af7a80b0c6065828d63eed357115b4.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/042918edd367ea3cabf23efe8d2021c956af7a80b0c6065828d63eed357115b4.jpg)

![41aa8ad2c9dd88e98359768cac87e2b92173d8d6f3e27311ebc3fc4fe9a363a0.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/41aa8ad2c9dd88e98359768cac87e2b92173d8d6f3e27311ebc3fc4fe9a363a0.jpg)

![68d6cc1280b241eaa9641532e7b214756d3498be407270ba769175635011e771.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/68d6cc1280b241eaa9641532e7b214756d3498be407270ba769175635011e771.jpg)

![69260e14899b1ca2cb0b06d0997fd512125c536581f504abfd091aede5e70db1.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/69260e14899b1ca2cb0b06d0997fd512125c536581f504abfd091aede5e70db1.jpg)

![71a2f43d64498eaeffa3ae983b6066e71b4b3af00a3dc000e998cf97b1a32585.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/71a2f43d64498eaeffa3ae983b6066e71b4b3af00a3dc000e998cf97b1a32585.jpg)

![71b62fa7f2160fb6e8625beb94e8f4b3c2ad26b40460686319cda135e3ee1d4c.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/71b62fa7f2160fb6e8625beb94e8f4b3c2ad26b40460686319cda135e3ee1d4c.jpg)

![a2ce6ae84ba5262732d9d677f29c4917f3e52d21f98934f506dd3f78c1f8963f.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/a2ce6ae84ba5262732d9d677f29c4917f3e52d21f98934f506dd3f78c1f8963f.jpg)

![b93037598404251fbb666a6acdcf4f5e1c98beceee97afa19c8a68a235617be5.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/b93037598404251fbb666a6acdcf4f5e1c98beceee97afa19c8a68a235617be5.jpg)

![c677c39d88783d24241f3cd271a0e941727884d4203b591e4ea4977351532fe9.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/images/c677c39d88783d24241f3cd271a0e941727884d4203b591e4ea4977351532fe9.jpg)

### Tables

![157193fba9dcc64b41d99b34345da3fc6de1865ccb1eb5e5fb14f2abb9f1a2b7.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/157193fba9dcc64b41d99b34345da3fc6de1865ccb1eb5e5fb14f2abb9f1a2b7.jpg)

![1b883dc0a041112b734a5162a76e54afa59642132df0b79a4cf4520887d8aa86.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/1b883dc0a041112b734a5162a76e54afa59642132df0b79a4cf4520887d8aa86.jpg)

![33c83d89b2ee4abb654f8d5f777ff2d68c7c6d9505f6f6cba6b8319ace23a0d0.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/33c83d89b2ee4abb654f8d5f777ff2d68c7c6d9505f6f6cba6b8319ace23a0d0.jpg)

![5ae0e915f0d29a49419c9d5e91787e566215d983e24760d681b7620fbe6d030a.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/5ae0e915f0d29a49419c9d5e91787e566215d983e24760d681b7620fbe6d030a.jpg)

![7f12a51e604dcc190cc825feb33a7201ba380dad4e6be1a8e0f2124335b41c93.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/7f12a51e604dcc190cc825feb33a7201ba380dad4e6be1a8e0f2124335b41c93.jpg)

![9d8c4115b9f081c919187256d20c274daa63c25a21588f33d7c0c51ab4daeef4.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/9d8c4115b9f081c919187256d20c274daa63c25a21588f33d7c0c51ab4daeef4.jpg)

![a402d9468b97fb022ec7cc7a4139728a35e1f4a64ad07850096f2bd30e5c5b28.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/a402d9468b97fb022ec7cc7a4139728a35e1f4a64ad07850096f2bd30e5c5b28.jpg)

![a9e65664cfd0c41e9d195f18a3e063ab117a77cc377a1fec715513efb2955e6c.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/a9e65664cfd0c41e9d195f18a3e063ab117a77cc377a1fec715513efb2955e6c.jpg)

![aa8fb1090b5846d468c7e7a44f391bfdf4eacce1f9d59410b0648808cba6c8f3.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/aa8fb1090b5846d468c7e7a44f391bfdf4eacce1f9d59410b0648808cba6c8f3.jpg)

![af4ece5041a2c2cddb9f8102c5082b83fabe1d8f65c894125f924dde93f390fb.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/af4ece5041a2c2cddb9f8102c5082b83fabe1d8f65c894125f924dde93f390fb.jpg)

![b5efa156baa386ec1376465c845a644a98d45df4f8b3954e89c761c38f5c14d2.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/b5efa156baa386ec1376465c845a644a98d45df4f8b3954e89c761c38f5c14d2.jpg)

![c3a49216d385ca0ea7a8660c6cde3df1fc51ca05ecfef2c75cc99e8ec804f0b3.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/c3a49216d385ca0ea7a8660c6cde3df1fc51ca05ecfef2c75cc99e8ec804f0b3.jpg)

![ee2fe00fbf450f3a61f89f962e6066aab6cb4f829f49d127a11907f2997c712b.jpg](../icml_results/4361_Kinematics-Driven%20Gaussian%20Shape%20Deformation%20for%20Blurry%20Monocular%20Dynamic%20Scenes/tables/ee2fe00fbf450f3a61f89f962e6066aab6cb4f829f49d127a11907f2997c712b.jpg)

<a id="paper-4318"></a>

## R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning


### Images

![088452aa6d9fd987b157df705604dfedf2874c579de0be07c18830b1b191ef5e.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/088452aa6d9fd987b157df705604dfedf2874c579de0be07c18830b1b191ef5e.jpg)

![11bf59cccf3d650e5e91ea4b56abb61bdd24e756c226eb53c1810ce2da231d43.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/11bf59cccf3d650e5e91ea4b56abb61bdd24e756c226eb53c1810ce2da231d43.jpg)

![26afcadd3b8f28aa9263af71b7cb7919e86463bc3c7af08c35f88158ed8ca8b1.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/26afcadd3b8f28aa9263af71b7cb7919e86463bc3c7af08c35f88158ed8ca8b1.jpg)

![304ec8d4ef487c4586d7c7ee243fe2831ad8175fbd1830e220b4c6a9ac22b25d.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/304ec8d4ef487c4586d7c7ee243fe2831ad8175fbd1830e220b4c6a9ac22b25d.jpg)

![459b0b4cadfbf709b8cc7805c8cfb61f5fa1f87b1721b35110b18d2d18039cae.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/459b0b4cadfbf709b8cc7805c8cfb61f5fa1f87b1721b35110b18d2d18039cae.jpg)

![4987476af20b279200da208772e2ccf2ecaad086d67ff0aaba66c9423bbdc12a.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/4987476af20b279200da208772e2ccf2ecaad086d67ff0aaba66c9423bbdc12a.jpg)

![54aa9d575acac9c0fb7bb3ceab839ca526f8bae2d2f3f573455df7f1eca342d4.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/54aa9d575acac9c0fb7bb3ceab839ca526f8bae2d2f3f573455df7f1eca342d4.jpg)

![732ff09d9fcbcde616ef627646e743a3d9f5146fe8ca4138f6fd5068277775d7.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/732ff09d9fcbcde616ef627646e743a3d9f5146fe8ca4138f6fd5068277775d7.jpg)

![7ddc01a2dfeb506823d235bf09bb130784de3aebd3f34de67d91d400b0160a06.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/7ddc01a2dfeb506823d235bf09bb130784de3aebd3f34de67d91d400b0160a06.jpg)

![7e41628514cac3dc8c414d43f1f3c08f2093265b197d88d452aa2171517f1977.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/7e41628514cac3dc8c414d43f1f3c08f2093265b197d88d452aa2171517f1977.jpg)

![db0f5b83c32528d651f2d76bf3d88d307df10b33ab2e5eba775009e00ff01837.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/db0f5b83c32528d651f2d76bf3d88d307df10b33ab2e5eba775009e00ff01837.jpg)

![dea961d30d678d50bbdff98b436697d3b387dd0a55254760d8d16abed57fd781.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/dea961d30d678d50bbdff98b436697d3b387dd0a55254760d8d16abed57fd781.jpg)

![ee616d030188a2730b31e50af27aea1cdcec2e90f20bcf2a276abb5fcecd5ee6.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/ee616d030188a2730b31e50af27aea1cdcec2e90f20bcf2a276abb5fcecd5ee6.jpg)

![f2cef0107b07c24e0737c243fb27c8d4a8b91b60475c20e5cfeb4ab70090fbd8.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/f2cef0107b07c24e0737c243fb27c8d4a8b91b60475c20e5cfeb4ab70090fbd8.jpg)

![fd89fdc59ed1516c49f9b0b7308f76336b901511e16bc8342b33d9160ccfec81.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/images/fd89fdc59ed1516c49f9b0b7308f76336b901511e16bc8342b33d9160ccfec81.jpg)

### Tables

![0f11498754c7f3646c69cb81bb531551d74ba61708a8322e285baf62f8539fed.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/0f11498754c7f3646c69cb81bb531551d74ba61708a8322e285baf62f8539fed.jpg)

![1a4fbbeaa89f388004d4a01ef3ac1219c9143c8cd0f6bd9dfbe780769e6bcab3.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/1a4fbbeaa89f388004d4a01ef3ac1219c9143c8cd0f6bd9dfbe780769e6bcab3.jpg)

![203175300f74043c875c5101b564a3ef7e65ec5a6f821ec2d12ceee3a2a493f8.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/203175300f74043c875c5101b564a3ef7e65ec5a6f821ec2d12ceee3a2a493f8.jpg)

![2c357ff90988ac7db5d1ac1936da40b0dc077d26d26db67cbe64b3f5d5adee88.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/2c357ff90988ac7db5d1ac1936da40b0dc077d26d26db67cbe64b3f5d5adee88.jpg)

![31dd0575c0e776668374c9f2b7c3941ab01262e9088fe6801647db5fb2d35ae7.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/31dd0575c0e776668374c9f2b7c3941ab01262e9088fe6801647db5fb2d35ae7.jpg)

![44cadb6003951e75d2ba0a92cbea91091925b263d363b2509b5a6f73d672f466.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/44cadb6003951e75d2ba0a92cbea91091925b263d363b2509b5a6f73d672f466.jpg)

![4f075e75cd06097742f29ca146636c275c581bbaadc8702db3a7e1f19de9fe56.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/4f075e75cd06097742f29ca146636c275c581bbaadc8702db3a7e1f19de9fe56.jpg)

![7f8d51a75368815d2c6948a442d85563a9add5ae30b97b44584265f92f8ef207.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/7f8d51a75368815d2c6948a442d85563a9add5ae30b97b44584265f92f8ef207.jpg)

![9b63c5b3a642fa4e6104a16bc09c363f5ed05fb2d7850f7ce79a869f3500934b.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/9b63c5b3a642fa4e6104a16bc09c363f5ed05fb2d7850f7ce79a869f3500934b.jpg)

![a846743cdd97d2afaf5ca2902522cde985bf6008250b876ab967c0c47633596c.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/a846743cdd97d2afaf5ca2902522cde985bf6008250b876ab967c0c47633596c.jpg)

![e4215abde850e2816a22cecf36b6a3fdaed1b91c7acdd5a802dc3b7c7d8161d5.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/e4215abde850e2816a22cecf36b6a3fdaed1b91c7acdd5a802dc3b7c7d8161d5.jpg)

![e665c6218d6aa138e48686745b19c99ead627f9b181dafc7095705ed131406b3.jpg](../icml_results/4362_R2R2_%20Robust%20Representation%20for%20Intensive%20Experience%20Reuse%20via%20Redundancy%20Reduction%20in%20Self-Predicti/tables/e665c6218d6aa138e48686745b19c99ead627f9b181dafc7095705ed131406b3.jpg)

<a id="paper-4319"></a>

## GO-PRE: Goal-Oriented Next-Best-View Selection via Predictive Rendering Entropy for Active 3D Reconstruction


### Images

![1db374ff7077a86e1a116886e316bf40a293d8bd2666478363d1030ed4713d73.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/1db374ff7077a86e1a116886e316bf40a293d8bd2666478363d1030ed4713d73.jpg)

![3b24974885c208d7da1de74014d2cd37b66e1a0c0fe48879d1d71d014e1df9c0.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/3b24974885c208d7da1de74014d2cd37b66e1a0c0fe48879d1d71d014e1df9c0.jpg)

![9555636be11cb605fd3d97949bd3f6a031b423fc63d0efc036fbcacc20cf11dd.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/9555636be11cb605fd3d97949bd3f6a031b423fc63d0efc036fbcacc20cf11dd.jpg)

![a3bd7f25208793476b92503699e72e09d8b80b57f3b16a14c24fe24092631742.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/a3bd7f25208793476b92503699e72e09d8b80b57f3b16a14c24fe24092631742.jpg)

![af96172486157e4968e49444dc46f84e771ae4cdcb55635125673dc33529b57a.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/af96172486157e4968e49444dc46f84e771ae4cdcb55635125673dc33529b57a.jpg)

![b73775eefd09659ffa3ffef97949b7779a141c9f3dfeb7d0882d1318db1be8cd.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/b73775eefd09659ffa3ffef97949b7779a141c9f3dfeb7d0882d1318db1be8cd.jpg)

![ed609979de7ad1de207e41f829a617552685aa4a7e26a2b642bff600cd5e157d.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/images/ed609979de7ad1de207e41f829a617552685aa4a7e26a2b642bff600cd5e157d.jpg)

### Tables

![0c9a445c95420aa9290cb5d5a58b3b9f419b2a71ecbf193ca58b16baab8ee351.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/0c9a445c95420aa9290cb5d5a58b3b9f419b2a71ecbf193ca58b16baab8ee351.jpg)

![11d95b4ec70e6ecf7b527588eb3e89141590c25c42b499a0eb5d4aae35075bf6.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/11d95b4ec70e6ecf7b527588eb3e89141590c25c42b499a0eb5d4aae35075bf6.jpg)

![8ddb0a55e3fff6dc597a5316c123061273b86406ea9cd39b912cbf0c85314904.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/8ddb0a55e3fff6dc597a5316c123061273b86406ea9cd39b912cbf0c85314904.jpg)

![a78c3ecf4b0d5d5b494911b59be455dd1f0bc92d66026fccfaedeab37d474877.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/a78c3ecf4b0d5d5b494911b59be455dd1f0bc92d66026fccfaedeab37d474877.jpg)

![f634dde6267e6473acd727e18b2f2bbcdd5d388bce761cb87fe414c82f564156.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/f634dde6267e6473acd727e18b2f2bbcdd5d388bce761cb87fe414c82f564156.jpg)

![fcc50142d83becab303dade4623366d95ceb67fde4680da77ace63066c8c0da3.jpg](../icml_results/4363_GO-PRE_%20Goal-Oriented%20Next-Best-View%20Selection%20via%20Predictive%20Rendering%20Entropy%20for%20Active%203D%20Recons/tables/fcc50142d83becab303dade4623366d95ceb67fde4680da77ace63066c8c0da3.jpg)

<a id="paper-4320"></a>

## The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting


### Images

![0ab355fc2b7075caad974906856174639f7d852b09f1cfdb13686d507f76ec94.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/images/0ab355fc2b7075caad974906856174639f7d852b09f1cfdb13686d507f76ec94.jpg)

![4eb1817fb57e4e3e2d483150b2d00e49c03031611434915047a2c3b520daf9f8.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/images/4eb1817fb57e4e3e2d483150b2d00e49c03031611434915047a2c3b520daf9f8.jpg)

![63ba71afac378930fd59de687c974229b89f6197724d28ef319fc7b9519b7e65.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/images/63ba71afac378930fd59de687c974229b89f6197724d28ef319fc7b9519b7e65.jpg)

![bfcb83ffb624886aa9c4daefe757807d2c2072ffb45927c16a8407e312cea617.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/images/bfcb83ffb624886aa9c4daefe757807d2c2072ffb45927c16a8407e312cea617.jpg)

### Tables

![000e09e2c81b9cf55032d9767c4ac3c99f79b74e68cede8b94f4311f53a3218c.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/000e09e2c81b9cf55032d9767c4ac3c99f79b74e68cede8b94f4311f53a3218c.jpg)

![2c810210c279eb24e049fbf38e93eb4b32bf2955c5fea492cbdfbb862c2a610c.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/2c810210c279eb24e049fbf38e93eb4b32bf2955c5fea492cbdfbb862c2a610c.jpg)

![2caac3e528ed71c46a599c92eff2777f780200cda8f07abfb2664c5bb9b51601.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/2caac3e528ed71c46a599c92eff2777f780200cda8f07abfb2664c5bb9b51601.jpg)

![2dbd9bb4d16153d0d61fabc6b280d42d1e977f4d9800dc779f2b9b325452037d.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/2dbd9bb4d16153d0d61fabc6b280d42d1e977f4d9800dc779f2b9b325452037d.jpg)

![a1dd79013a2f84eb822bc3435554ac5bd744f5112ad0bec889a7db640070af16.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/a1dd79013a2f84eb822bc3435554ac5bd744f5112ad0bec889a7db640070af16.jpg)

![a5281b59b79d346f59236484176c8f3f72a8f70dbb8bc4d308f9179392b352ca.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/a5281b59b79d346f59236484176c8f3f72a8f70dbb8bc4d308f9179392b352ca.jpg)

![dc9a8f4ca5118722d4f40ed2a03333da8cdae55b0500cf05cf6b330bcc409621.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/dc9a8f4ca5118722d4f40ed2a03333da8cdae55b0500cf05cf6b330bcc409621.jpg)

![f0946061aac2589b15990a7c86208b98217bb7895f48a48470a814e4316ff51c.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/f0946061aac2589b15990a7c86208b98217bb7895f48a48470a814e4316ff51c.jpg)

![f5a22098d60db3004b44af2b0265e04ca70458eb55e066cb77d65472f34be9ef.jpg](../icml_results/4364_The%20Label%20Horizon%20Paradox_%20Rethinking%20Supervision%20Targets%20in%20Financial%20Forecasting/tables/f5a22098d60db3004b44af2b0265e04ca70458eb55e066cb77d65472f34be9ef.jpg)

<a id="paper-4321"></a>

## MIMO-LP: A Multi-Input Multi-Output Framework for Subgraph-based Link Prediction


### Images

![235cfab3c36e77fcc30a680174137ae623e675a6f47077a6eb3dc6d8c9287d0b.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/235cfab3c36e77fcc30a680174137ae623e675a6f47077a6eb3dc6d8c9287d0b.jpg)

![7640f0dbfe20e8dee66c1de8c61b0044a0a7c1bcaba7e9e5897be6fd5d2893be.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/7640f0dbfe20e8dee66c1de8c61b0044a0a7c1bcaba7e9e5897be6fd5d2893be.jpg)

![8f18bd0421db6992c945162cc629c2afa45c296aadde75e8ca0250ac8eb6185a.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/8f18bd0421db6992c945162cc629c2afa45c296aadde75e8ca0250ac8eb6185a.jpg)

![968f515e72516f306bb2f7a245b0f58def39439925679c59736f0add0c6b7836.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/968f515e72516f306bb2f7a245b0f58def39439925679c59736f0add0c6b7836.jpg)

![ab6f1a21d1154cc691d6d32d2d030c36ae1a6237d0ed05be273f69ad3eb9da9f.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/ab6f1a21d1154cc691d6d32d2d030c36ae1a6237d0ed05be273f69ad3eb9da9f.jpg)

![c2bef8710f881d054ae8a7f41db1b5ed1710ccd89afcf9189895f7ae06b06327.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/c2bef8710f881d054ae8a7f41db1b5ed1710ccd89afcf9189895f7ae06b06327.jpg)

![e672c5ddb1d4937d0aba5d8844d74d5538c34c3e79c154fe862f5f6ec6b92b37.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/e672c5ddb1d4937d0aba5d8844d74d5538c34c3e79c154fe862f5f6ec6b92b37.jpg)

![f2fcc6982568482e29f7b9cd01e41495707a54ed784d380ef530874bbb49f680.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/f2fcc6982568482e29f7b9cd01e41495707a54ed784d380ef530874bbb49f680.jpg)

![f790d3f598eee209665d66c0a33be00d91bbdf6cf3935a0566fc34c78caf8b18.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/images/f790d3f598eee209665d66c0a33be00d91bbdf6cf3935a0566fc34c78caf8b18.jpg)

### Tables

![1cb927d5d668f528c55c6808a9bdfb9aea64de64bfad53d349cf84e763c42c35.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/1cb927d5d668f528c55c6808a9bdfb9aea64de64bfad53d349cf84e763c42c35.jpg)

![2ac05e9fc0c2b3e94b0646e896f155eff3ad5998c7f482218ef27903f3cf4c2d.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/2ac05e9fc0c2b3e94b0646e896f155eff3ad5998c7f482218ef27903f3cf4c2d.jpg)

![367caca02357c3361cc58f7add78a683a8001ceedd3073d215b39da96d40844f.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/367caca02357c3361cc58f7add78a683a8001ceedd3073d215b39da96d40844f.jpg)

![4b54fe70b3b5cff82d25b94b8e6575c4fd489f32329ebbd83199624830d310f3.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/4b54fe70b3b5cff82d25b94b8e6575c4fd489f32329ebbd83199624830d310f3.jpg)

![905246ffa29b9d5cc89f5eab73db1333518293f9e50b429b399897b2e7f88238.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/905246ffa29b9d5cc89f5eab73db1333518293f9e50b429b399897b2e7f88238.jpg)

![939be24093bab26e102eeb2516dfaa1851016009c7be648f646043fe777af76c.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/939be24093bab26e102eeb2516dfaa1851016009c7be648f646043fe777af76c.jpg)

![fe9462a5a5db0aafb25ce834728502f4efec1c63e14199f1356020cefff2772e.jpg](../icml_results/4365_MIMO-LP_%20A%20Multi-Input%20Multi-Output%20Framework%20for%20Subgraph-based%20Link%20Prediction/tables/fe9462a5a5db0aafb25ce834728502f4efec1c63e14199f1356020cefff2772e.jpg)

<a id="paper-4322"></a>

## MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety


### Images

![023f90a290aa62451eb807e46d8699ed84400555d8d15d6b549bbfca02f85432.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/023f90a290aa62451eb807e46d8699ed84400555d8d15d6b549bbfca02f85432.jpg)

![0afed581e70bb6025b15b2cbf7837ad6c22d163545e0486c02d4ad9f31cb0831.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/0afed581e70bb6025b15b2cbf7837ad6c22d163545e0486c02d4ad9f31cb0831.jpg)

![193e1efcf6b7df6fb6134304354e98320fa53c24346f702d4d7be3246004c57c.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/193e1efcf6b7df6fb6134304354e98320fa53c24346f702d4d7be3246004c57c.jpg)

![1e9dba51ee0f8403f0453fa88dbbcdf3088dbff28e694d4cba478f6b644c9927.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/1e9dba51ee0f8403f0453fa88dbbcdf3088dbff28e694d4cba478f6b644c9927.jpg)

![41503261e3442461709a0dd9e52f4f96963cf64170676960de0359584777aaa3.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/41503261e3442461709a0dd9e52f4f96963cf64170676960de0359584777aaa3.jpg)

![4696fb9deda1c8a045f134a32b689701252afecefecf6771f8352e2839e6607d.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/4696fb9deda1c8a045f134a32b689701252afecefecf6771f8352e2839e6607d.jpg)

![4df2836261c7c2fa5d669e29d4c7cbb669e868b280ce7f590a853b7721be5b5e.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/4df2836261c7c2fa5d669e29d4c7cbb669e868b280ce7f590a853b7721be5b5e.jpg)

![aa4919d76cfb8da70acb0ac50451728fad6926fb08ecc62a16b0db144992fdf0.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/aa4919d76cfb8da70acb0ac50451728fad6926fb08ecc62a16b0db144992fdf0.jpg)

![c42acf6453b400d7134678b111e6903054d7de38a59931907abcbc16051cdd9e.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/c42acf6453b400d7134678b111e6903054d7de38a59931907abcbc16051cdd9e.jpg)

![d7494fd17d20aeb60dec73aa16db8628aaaa07dbc42904a6c3e2425634474e0d.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/d7494fd17d20aeb60dec73aa16db8628aaaa07dbc42904a6c3e2425634474e0d.jpg)

![f9594d9e0013dc9c3da93d0b0ff6ff71f5def4faaa04fa7220ea7e356a8b29dd.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/images/f9594d9e0013dc9c3da93d0b0ff6ff71f5def4faaa04fa7220ea7e356a8b29dd.jpg)

### Tables

![0c6c0133d66bae97ac441578ee9e4eb5ae7b502df6c4ca4ae5c808480905e5f0.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/0c6c0133d66bae97ac441578ee9e4eb5ae7b502df6c4ca4ae5c808480905e5f0.jpg)

![10c455f1da662f313a1e48fcf5f5da9e270c77108d4c9c82250777c3767d4f20.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/10c455f1da662f313a1e48fcf5f5da9e270c77108d4c9c82250777c3767d4f20.jpg)

![408c869133ff642bda27522af16d9cd012a13238720015da8e9d99c5be7fd8f5.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/408c869133ff642bda27522af16d9cd012a13238720015da8e9d99c5be7fd8f5.jpg)

![5cff5690a2957f650d8a25ccbeb0cf3c72292f9227781b1386c43ee948e4a3d9.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/5cff5690a2957f650d8a25ccbeb0cf3c72292f9227781b1386c43ee948e4a3d9.jpg)

![60994c8c8959705e577e6d0b42095253780dfe1473e4dfcd4cb9faf89c542c83.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/60994c8c8959705e577e6d0b42095253780dfe1473e4dfcd4cb9faf89c542c83.jpg)

![654b748660c2fd7320b5a8972253400dd432ec297851c43b616076b58d137a0c.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/654b748660c2fd7320b5a8972253400dd432ec297851c43b616076b58d137a0c.jpg)

![69c1db789581fe6b27460c76ef51035bcc54ae65345f25672cba2361e13a79fa.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/69c1db789581fe6b27460c76ef51035bcc54ae65345f25672cba2361e13a79fa.jpg)

![87e9bdcf2f5ef2920b48c49f55d87e17870f0d97e3fbbc83c355c373b04bdaf9.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/87e9bdcf2f5ef2920b48c49f55d87e17870f0d97e3fbbc83c355c373b04bdaf9.jpg)

![8ebff3da71118a1013f2e0e2cd85e72273bc00e8f20fbea72e3dd7f08520d489.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/8ebff3da71118a1013f2e0e2cd85e72273bc00e8f20fbea72e3dd7f08520d489.jpg)

![954547d199310db3e1b44e44bd820349429f1191023b7c580795724151897b21.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/954547d199310db3e1b44e44bd820349429f1191023b7c580795724151897b21.jpg)

![96f3aab2ce057e47e2bd65b4113ea414a47bdd6b120d49eed572f7e8880cbced.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/96f3aab2ce057e47e2bd65b4113ea414a47bdd6b120d49eed572f7e8880cbced.jpg)

![997e47b3e5f4a7ba92df549efa5ede567d3f62d89d6b6bfc4afc55c2c874c874.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/997e47b3e5f4a7ba92df549efa5ede567d3f62d89d6b6bfc4afc55c2c874c874.jpg)

![a067fc5daab87b1d393583de53935759d2ece6085fb50506c78ff44b139d0e37.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/a067fc5daab87b1d393583de53935759d2ece6085fb50506c78ff44b139d0e37.jpg)

![a8d49a1f79e3f7b09b9cc7523a0e784d54b9fb14f10a21d81bb705c764d6642c.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/a8d49a1f79e3f7b09b9cc7523a0e784d54b9fb14f10a21d81bb705c764d6642c.jpg)

![afd9e0ffb0f4a6179b68b1b64204ce1708c9f64597545ce7de9827d9aa66b9ec.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/afd9e0ffb0f4a6179b68b1b64204ce1708c9f64597545ce7de9827d9aa66b9ec.jpg)

![bfe453170a2f7625e2379702d1fd10355fa434881b8b0f8a125904bd73d0ce38.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/bfe453170a2f7625e2379702d1fd10355fa434881b8b0f8a125904bd73d0ce38.jpg)

![ca1effb574fa338668a878019fac72bda232e70cda3d144b902808d2d5943b68.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/ca1effb574fa338668a878019fac72bda232e70cda3d144b902808d2d5943b68.jpg)

![e25888e4e166a6f8e6b1feaedccf9ee53d1467aa7e1ab232a202130fbc0902ca.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/e25888e4e166a6f8e6b1feaedccf9ee53d1467aa7e1ab232a202130fbc0902ca.jpg)

![e41dd85cedfd1b9b24f06c4b52a8c5f2410327bfa6e7fe36ce95063da4c61f94.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/e41dd85cedfd1b9b24f06c4b52a8c5f2410327bfa6e7fe36ce95063da4c61f94.jpg)

![e6b36f97fd3bfe8d90c92f430330b83fe863e9dd86ac4581de81d83268812eba.jpg](../icml_results/4366_MultiBreak_%20A%20Scalable%20and%20Diverse%20Multi-turn%20Jailbreak%20Benchmark%20for%20Evaluating%20LLM%20Safety/tables/e6b36f97fd3bfe8d90c92f430330b83fe863e9dd86ac4581de81d83268812eba.jpg)

<a id="paper-4323"></a>

## Expo-GS: Exposure-Aware Signed Distance Function in Gaussian Splatting for High Dynamic Range

### Images

![45bb1fab0e1728651692012e08ee65bcf731183bc6c9ebc05eec72f54da043c9.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/45bb1fab0e1728651692012e08ee65bcf731183bc6c9ebc05eec72f54da043c9.jpg)

![5997f00b246982e2c8e5d6a71068d5d48b6ae167f34d7be745ec138fc6a95d16.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/5997f00b246982e2c8e5d6a71068d5d48b6ae167f34d7be745ec138fc6a95d16.jpg)

![5bf9fd76ba53d4dcbfb2ca21f9cef33925cb9b4b9cdca28c81510f44f3ca7a74.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/5bf9fd76ba53d4dcbfb2ca21f9cef33925cb9b4b9cdca28c81510f44f3ca7a74.jpg)

![6bcc4f622c1b121c9aa5516442ba11bac70f522d7aa3ac053de49834518f2cd0.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/6bcc4f622c1b121c9aa5516442ba11bac70f522d7aa3ac053de49834518f2cd0.jpg)

![8c22cb580d99d755f30f1b416a7fbde659c94e47b9355c09d61d0d2e76667988.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/8c22cb580d99d755f30f1b416a7fbde659c94e47b9355c09d61d0d2e76667988.jpg)

![994d7108f65703a7529799beedd41f453028e1d0e2735f50f9c21bd7798d8b49.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/994d7108f65703a7529799beedd41f453028e1d0e2735f50f9c21bd7798d8b49.jpg)

![9d5ef2901d8e56864e7e7ebb1945377abd9717dd735c69c48e3b7964fdef3a78.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/9d5ef2901d8e56864e7e7ebb1945377abd9717dd735c69c48e3b7964fdef3a78.jpg)

![b40250ecdd4a12375cfd5a07021ee1bf3ab5d765c673c4d39dbc4c090efa2cec.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/b40250ecdd4a12375cfd5a07021ee1bf3ab5d765c673c4d39dbc4c090efa2cec.jpg)

![e33f635d7358428e926122b0836caee0ebcc53bf31e6790b14a0003ec37ed2e9.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/e33f635d7358428e926122b0836caee0ebcc53bf31e6790b14a0003ec37ed2e9.jpg)

![e9be9fb0d0d3431da01fc44596e4c132f691ed03286b7ca939940a522a8346a0.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/e9be9fb0d0d3431da01fc44596e4c132f691ed03286b7ca939940a522a8346a0.jpg)

![f9bd75d33c853d719b303bf81d8d71c281d5255bac3110e5dc7ed1a50d2139f3.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/f9bd75d33c853d719b303bf81d8d71c281d5255bac3110e5dc7ed1a50d2139f3.jpg)

![fd4a7d35d43a6d0d7bac8c56ec53de3726fa98ae8d553d1eab406edea2e1f2c1.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/images/fd4a7d35d43a6d0d7bac8c56ec53de3726fa98ae8d553d1eab406edea2e1f2c1.jpg)

### Tables

![04462675bb93d3c1b4b64f611d0f5c24ad17eb1072914385500c0b8a351e9f8d.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/04462675bb93d3c1b4b64f611d0f5c24ad17eb1072914385500c0b8a351e9f8d.jpg)

![145f64374bdbb2f38da428beb3229d9255a7fa4e47c6a49abd367e1b6a07b05e.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/145f64374bdbb2f38da428beb3229d9255a7fa4e47c6a49abd367e1b6a07b05e.jpg)

![36271a74d4be4e8f63d0bf9beec20836a5f77ed8d8e65d46d64142711d0134b1.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/36271a74d4be4e8f63d0bf9beec20836a5f77ed8d8e65d46d64142711d0134b1.jpg)

![36837191ccc5f69c57a34d266759398bb96c4c08e36f178dab1c9b7e0b65129e.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/36837191ccc5f69c57a34d266759398bb96c4c08e36f178dab1c9b7e0b65129e.jpg)

![498a1855a8bba938efc5537e33d828c6b41a25c6e81aeba36b80de5fd1d2de46.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/498a1855a8bba938efc5537e33d828c6b41a25c6e81aeba36b80de5fd1d2de46.jpg)

![4addca0ccb6f82a59bfb22ee7ce8bbbabe898f4a1fc87ff34e93945d0e5fff90.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/4addca0ccb6f82a59bfb22ee7ce8bbbabe898f4a1fc87ff34e93945d0e5fff90.jpg)

![4ed1e9846e3a5c21fe9e0230355f963d31d04df3d99d05858c26239851a7a33a.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/4ed1e9846e3a5c21fe9e0230355f963d31d04df3d99d05858c26239851a7a33a.jpg)

![6db1f51fd4305267b8cf2fce9aa56928263a7c0111bf44b063b0bb4fb68cce67.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/6db1f51fd4305267b8cf2fce9aa56928263a7c0111bf44b063b0bb4fb68cce67.jpg)

![82e1ab0575eb21a606604e8ded59cce35ea817325277d2839c71ea26a6fd2ab8.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/82e1ab0575eb21a606604e8ded59cce35ea817325277d2839c71ea26a6fd2ab8.jpg)

![96eca428e4f5cb28ca79c5b6cb7a74a0495f787435bc7f09398f679820c4c1f5.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/96eca428e4f5cb28ca79c5b6cb7a74a0495f787435bc7f09398f679820c4c1f5.jpg)

![aca843a3c4364990af1ef0a5d5a9ba41b388b5ede77bc74723dcc3b110e231c1.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/aca843a3c4364990af1ef0a5d5a9ba41b388b5ede77bc74723dcc3b110e231c1.jpg)

![b7d7799d2977f9c2f0a2678630f42da52df4bf756552801dc6de12d10530a87d.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/b7d7799d2977f9c2f0a2678630f42da52df4bf756552801dc6de12d10530a87d.jpg)

![c8da37f1c738fa42f8ee6e162f713e6ef129596211a0126d6745f75bddccb758.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/c8da37f1c738fa42f8ee6e162f713e6ef129596211a0126d6745f75bddccb758.jpg)

![cabb2ea1c57ce079ed298c0b82a981aa618ac8e5f0f8ef481d41f29eb2342271.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/cabb2ea1c57ce079ed298c0b82a981aa618ac8e5f0f8ef481d41f29eb2342271.jpg)

![f7853dfc12283d0571fc88115b5607a8529259bf5355ff7be5d701c73700ddf9.jpg](../icml_results/4367_Expo-GS_%20Exposure-Aware%20Signed%20Distance%20Function%20in%20Gaussian%20Splatting%20for%20High%20Dynamic%20Range/tables/f7853dfc12283d0571fc88115b5607a8529259bf5355ff7be5d701c73700ddf9.jpg)
