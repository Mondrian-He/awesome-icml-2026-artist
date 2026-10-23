# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 75 of 197

## 目录 (Table of Contents)

1. [Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents](#paper-2443)
2. [AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking](#paper-2444)
3. [Transfer Learning in High-dimensional Ising Models](#paper-2445)
4. [CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution](#paper-2446)
5. [StretchTime: Adaptive Time Series Forecasting via Symplectic Attention](#paper-2447)
6. [Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards](#paper-2448)
7. [Temporal Preference Optimization for Unsupervised Retrieval](#paper-2449)
8. [DNACHUNKER: Learnable Tokenization for DNA Language Models](#paper-2450)
9. [EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](#paper-2451)
10. [Identifiable Token Correspondence for World Models](#paper-2452)
11. [Persona-Pruner: Sculpting Lightweight Models for Role-Playing](#paper-2453)
12. [Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](#paper-2454)
13. [SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models](#paper-2455)
14. [Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression](#paper-2456)
15. [ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](#paper-2457)
16. [TRACE: Toulmin-based Reasoning Assessment through Constructive Elements for LLM CoT Evaluation](#paper-2458)
17. [ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](#paper-2459)
18. [Memory as Dynamics: Learning Reliability-Guided Predictive Models for Online Video Perception](#paper-2460)
19. [Coverage Improvement and Fast Convergence of On-policy Preference Learning](#paper-2461)
20. [Equivariant Neural Networks for General Linear Symmetries on Lie Algebras](#paper-2462)
21. [Dywave: Event-Aligned Dynamic Tokenization for Heterogeneous IoT Sensing Signals](#paper-2463)
22. [Representational Curvature Modulates Behavioral Uncertainty in Large Language Models](#paper-2464)
23. [Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks](#paper-2465)
24. [Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking](#paper-2466)
25. [Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation](#paper-2467)
26. [Tucker Attention: A generalization of approximate attention mechanisms](#paper-2468)
27. [Skewness-Robust Causal Discovery in Location-Scale Noise Models](#paper-2469)
28. [Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)](#paper-2470)
29. [Tractable Expected Information Gains for Exponential Family Posteriors](#paper-2471)
30. [Beyond the Bellman Recursion: A Pontryagin-Guided Framework for Non-Exponential Discounting](#paper-2472)
31. [From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](#paper-2473)
32. [Geometric Embedding Alignment via Curvature Matching in Transfer Learning](#paper-2474)
33. [Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks](#paper-2475)

---


<a id="paper-2443"></a>

## Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents

### Images

![12ae1f9c923d9276b70e71675885cfd6af0a8f1e8d5112de19cccfde70706c52.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/12ae1f9c923d9276b70e71675885cfd6af0a8f1e8d5112de19cccfde70706c52.jpg)

![33a686d70aa4e100abd10268d6f766f430c0c9896434a7f3ae15cfe01e2fe046.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/33a686d70aa4e100abd10268d6f766f430c0c9896434a7f3ae15cfe01e2fe046.jpg)

![5696cb7fea6d4230a132496b72e6bc8ce594aa19445ccbd52d7f52993b4db58b.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/5696cb7fea6d4230a132496b72e6bc8ce594aa19445ccbd52d7f52993b4db58b.jpg)

![6147b2d2f95d5b489293542a846632ca93ff3689c28bd2ea141ada20320bc554.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/6147b2d2f95d5b489293542a846632ca93ff3689c28bd2ea141ada20320bc554.jpg)

![6df567517b424b6346460ad711bd89590fdc37a765b6cba9b82e0a9fb4115a1b.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/6df567517b424b6346460ad711bd89590fdc37a765b6cba9b82e0a9fb4115a1b.jpg)

![7f3f2128ac4817833a2514d786a87a53849e26d3761e1efca9f083d3027a659a.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/7f3f2128ac4817833a2514d786a87a53849e26d3761e1efca9f083d3027a659a.jpg)

![c6a19e33993c33a852a5b34f5067369dda1e2e6505128b8e66f1a7d4aea4dfd6.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/c6a19e33993c33a852a5b34f5067369dda1e2e6505128b8e66f1a7d4aea4dfd6.jpg)

![e29377a04587757e3d2b4a80abee72863fb8a16a31444d3c489bc5b27dda2c71.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/images/e29377a04587757e3d2b4a80abee72863fb8a16a31444d3c489bc5b27dda2c71.jpg)

### Tables

![07b5a447ad86ffdd15a0285befc1d2ce3303bfe404724b9f434703e2e8116e39.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/07b5a447ad86ffdd15a0285befc1d2ce3303bfe404724b9f434703e2e8116e39.jpg)

![0e8c7eb46c7188567ea19083975dd27575757cb58f3b128acfce707800df54de.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/0e8c7eb46c7188567ea19083975dd27575757cb58f3b128acfce707800df54de.jpg)

![2d67b2719944af051cd0987b39a3555260f4d7460e44585b680d42963abe33bf.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/2d67b2719944af051cd0987b39a3555260f4d7460e44585b680d42963abe33bf.jpg)

![563120d1cc3deeff765b8ed4fe65addca57c98031afa4110bb01ca92f308210e.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/563120d1cc3deeff765b8ed4fe65addca57c98031afa4110bb01ca92f308210e.jpg)

![66895b09c3e66cdf42192331cd9260a69f627286c0271dafb8a9bd123380562c.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/66895b09c3e66cdf42192331cd9260a69f627286c0271dafb8a9bd123380562c.jpg)

![80371e28f986162a81293ab65f716895f2e4bd4bf5f1c15f0d2f7634eec072ed.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/80371e28f986162a81293ab65f716895f2e4bd4bf5f1c15f0d2f7634eec072ed.jpg)

![8cf9eddcb375407f748cedfe35e1507dc091b4e47a00fc75d22c83ee04fe9ba5.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/8cf9eddcb375407f748cedfe35e1507dc091b4e47a00fc75d22c83ee04fe9ba5.jpg)

![9b49e18781b4c596cefb611a3795e6a379cb949b1c3091c49f9cbcd849aa361b.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/9b49e18781b4c596cefb611a3795e6a379cb949b1c3091c49f9cbcd849aa361b.jpg)

![a0f8407186026a737cd1d976d438caae3e7f7b381f21a7300673f42f0ce70b57.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/a0f8407186026a737cd1d976d438caae3e7f7b381f21a7300673f42f0ce70b57.jpg)

![b6dc27f13f3445f654d64bf7617a43da3c3d27389308f6425b78819ca0050da6.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/b6dc27f13f3445f654d64bf7617a43da3c3d27389308f6425b78819ca0050da6.jpg)

![c7e0bc296c34be2e4734563074f1542dec7347950abf8d0fe927747e39bf7465.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/c7e0bc296c34be2e4734563074f1542dec7347950abf8d0fe927747e39bf7465.jpg)

![cbc53f45e0220f89b18e4bed29f35d301259b75f69c58f059be0332b0bd8701d.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/cbc53f45e0220f89b18e4bed29f35d301259b75f69c58f059be0332b0bd8701d.jpg)

![fd4083f9174f46ca44c8c220d3d02cd31e270bbea7a131b2a8b1761274f53ff9.jpg](../icml_results/2472_GSRQ_%20Gain-Shape%20Residual%20Quantization%20for%20Sub-1-bit%20KV%20Cache/tables/fd4083f9174f46ca44c8c220d3d02cd31e270bbea7a131b2a8b1761274f53ff9.jpg)

## Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents


### Images

![1e9592d5f7c4f284dc90d017e5b5ed67f3fb0a31df25e9fef7a2d49be7bfb582.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/1e9592d5f7c4f284dc90d017e5b5ed67f3fb0a31df25e9fef7a2d49be7bfb582.jpg)

![211d9644481668e569959d4f97619890411f70431a8bb9c186023aecffdb9466.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/211d9644481668e569959d4f97619890411f70431a8bb9c186023aecffdb9466.jpg)

![29381ae0779324fda34d787ac765d6e408a16a114d2c70b6cb761294d4d484d2.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/29381ae0779324fda34d787ac765d6e408a16a114d2c70b6cb761294d4d484d2.jpg)

![2cade8d6be4cefa5f25136c99fb9ee5d69103a76a54e713584d15ff980b92e55.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/2cade8d6be4cefa5f25136c99fb9ee5d69103a76a54e713584d15ff980b92e55.jpg)

![68a4e84e7050ca7f65253a3f776234e181f39346a5dfa57c6d24e85ce3b25a0b.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/68a4e84e7050ca7f65253a3f776234e181f39346a5dfa57c6d24e85ce3b25a0b.jpg)

![c0042688935b0f43d8e1ebf57af4b211817fc870dd950c39417988d8c6dd2a93.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/c0042688935b0f43d8e1ebf57af4b211817fc870dd950c39417988d8c6dd2a93.jpg)

![debb4246ca9a2bb41180237f9e7b5b35eb1d855a1efa9a685a4e0ac1993a9dd3.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/images/debb4246ca9a2bb41180237f9e7b5b35eb1d855a1efa9a685a4e0ac1993a9dd3.jpg)

### Tables

![03dc2e42b34704cdb0f134ba55dc6a9270c24088c14da857922f64d8d6a4c861.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/03dc2e42b34704cdb0f134ba55dc6a9270c24088c14da857922f64d8d6a4c861.jpg)

![2d7eaaffa4e3aa55e36a8e69d799dae560e73d3ad8da91f2f2d31696a7772fbb.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/2d7eaaffa4e3aa55e36a8e69d799dae560e73d3ad8da91f2f2d31696a7772fbb.jpg)

![5c8a4e126ce68459d89cc8442f524662b90ec84855199042bac2aba21873165b.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/5c8a4e126ce68459d89cc8442f524662b90ec84855199042bac2aba21873165b.jpg)

![70b61b6b4496ad9b0fcb0885b2919951c2de9897c958be689a023f68c92bfd34.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/70b61b6b4496ad9b0fcb0885b2919951c2de9897c958be689a023f68c92bfd34.jpg)

![86ac98a0d83f7329602931c543fbd6e80d522519e3ac988f690d51e724bb8df1.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/86ac98a0d83f7329602931c543fbd6e80d522519e3ac988f690d51e724bb8df1.jpg)

![90782fbb2ee9499dd7a6c96cf4ce124496cbc4730d47b7a831f5ffc9ba3c6ede.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/90782fbb2ee9499dd7a6c96cf4ce124496cbc4730d47b7a831f5ffc9ba3c6ede.jpg)

![9098a0b3553c614e094dfda00a03cdca93615ea6cd2a7be3230c20b151cbf896.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/9098a0b3553c614e094dfda00a03cdca93615ea6cd2a7be3230c20b151cbf896.jpg)

![b3c6742be8c6c8c0b49038342196c45c95d34b23168cd074dc30b12b632a8b6b.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/b3c6742be8c6c8c0b49038342196c45c95d34b23168cd074dc30b12b632a8b6b.jpg)

![be5bc97b15911fff2db67da69ac0915803ecab3bc5e2f06ee8313e463277dc38.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/be5bc97b15911fff2db67da69ac0915803ecab3bc5e2f06ee8313e463277dc38.jpg)

![c2a238ad898f3793f60f5cdb4df17217adf5c0024ab7adfe8832361af23985b6.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/c2a238ad898f3793f60f5cdb4df17217adf5c0024ab7adfe8832361af23985b6.jpg)

![d3e4d4ab1c09d03ac18de370606ba88c9c6626540229c72cdf45665b3e342cb2.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/d3e4d4ab1c09d03ac18de370606ba88c9c6626540229c72cdf45665b3e342cb2.jpg)

![fbe0f65ae099906c7fe8b84a090478ee96f14b512e60b881a41d98fef5c7004c.jpg](../icml_results/2473_Beyond%20the%20Final%20Answer_%20Evaluating%20the%20Reasoning%20Trajectories%20of%20Tool-Augmented%20Agents/tables/fbe0f65ae099906c7fe8b84a090478ee96f14b512e60b881a41d98fef5c7004c.jpg)

<a id="paper-2444"></a>

## AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking


### Images

![03002208d869935d1864cad8a5866e1b226f9d267d7d63b2edde134172875003.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/03002208d869935d1864cad8a5866e1b226f9d267d7d63b2edde134172875003.jpg)

![0a22f052adc56a89722fe8e3ddb28a9a7d46144e728154c0a4999786180a421c.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/0a22f052adc56a89722fe8e3ddb28a9a7d46144e728154c0a4999786180a421c.jpg)

![2b74af728d73c2040c4955acd55777d049366f8973773c0f5c1553d4a7be7415.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/2b74af728d73c2040c4955acd55777d049366f8973773c0f5c1553d4a7be7415.jpg)

![40eee11df95be08559012edc1f716acb91636f393fc900e850d20f0f0fc26b00.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/40eee11df95be08559012edc1f716acb91636f393fc900e850d20f0f0fc26b00.jpg)

![416d91b31836739b19791d70fbbb22c638091a25ef2876e4de39f99c66369601.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/416d91b31836739b19791d70fbbb22c638091a25ef2876e4de39f99c66369601.jpg)

![48080f793baf7454bc24176721908a4b5990569e1d706916e30cbc5f461a9d16.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/48080f793baf7454bc24176721908a4b5990569e1d706916e30cbc5f461a9d16.jpg)

![5385dfbc54b287b35e8ff86ff061ae571bdfee19e85fbe00aa8c84588add7aaf.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/5385dfbc54b287b35e8ff86ff061ae571bdfee19e85fbe00aa8c84588add7aaf.jpg)

![7c559edd992ad13e7f8c85bbc0c89f8487b419a17be58c63b1baebb29678deda.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/7c559edd992ad13e7f8c85bbc0c89f8487b419a17be58c63b1baebb29678deda.jpg)

![919ea9244acdfde926a02cad91fd5508535f75ee24bfddf05dfe168112ccfd2a.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/919ea9244acdfde926a02cad91fd5508535f75ee24bfddf05dfe168112ccfd2a.jpg)

![c58a7d163501ba631b99a9eb9d09e05c5303ffd3ae203b63da2c3f343b333650.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/c58a7d163501ba631b99a9eb9d09e05c5303ffd3ae203b63da2c3f343b333650.jpg)

![ce9f4ff2b543db1b97f9027b8e9f90424ca7bfe19b586ac1770e71e188c9e787.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/ce9f4ff2b543db1b97f9027b8e9f90424ca7bfe19b586ac1770e71e188c9e787.jpg)

![d1714f102e3c94d4c767ae0f4b446cce4d7eabb5d9d4afa849a073381d12838a.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/d1714f102e3c94d4c767ae0f4b446cce4d7eabb5d9d4afa849a073381d12838a.jpg)

![dc88a5baa9fdd7b9e7d1e1663f4be91cadefaeca9452a625d343644d0b2f4389.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/dc88a5baa9fdd7b9e7d1e1663f4be91cadefaeca9452a625d343644d0b2f4389.jpg)

![e614c756bce190faa8680ef34c5e0dd282336e2eb2f27635660cf267eac025f8.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/images/e614c756bce190faa8680ef34c5e0dd282336e2eb2f27635660cf267eac025f8.jpg)

### Tables

![04c75b50cc5c6288cc710996dcfe7cab9b8d1d23cb7418a455f45d1a42f97030.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/04c75b50cc5c6288cc710996dcfe7cab9b8d1d23cb7418a455f45d1a42f97030.jpg)

![11d4802c1c94f088736dd6aba27f68a2f4f6685e4d34e127fffaf11e8ca54bd5.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/11d4802c1c94f088736dd6aba27f68a2f4f6685e4d34e127fffaf11e8ca54bd5.jpg)

![39fc57ace0f1f544ff558ff96bcdcd3eb2e45a6c3636e3c553fad46bce980d62.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/39fc57ace0f1f544ff558ff96bcdcd3eb2e45a6c3636e3c553fad46bce980d62.jpg)

![48712399c591edf483793fc6a525323eb8aaa5dbd756e5072e373e593bfad780.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/48712399c591edf483793fc6a525323eb8aaa5dbd756e5072e373e593bfad780.jpg)

![4caec80de6849ffa5d2909ac3bd1eafc499ca82c787dea0b56a741224bd57077.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/4caec80de6849ffa5d2909ac3bd1eafc499ca82c787dea0b56a741224bd57077.jpg)

![554cd92c3774fbe6433fbcbebfae2e7b9e2be74ee1bd94b6687332e2620db71d.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/554cd92c3774fbe6433fbcbebfae2e7b9e2be74ee1bd94b6687332e2620db71d.jpg)

![65f80f7cf5b721f7132f742de08cc363a8d156d6046aa9e7c6790c2c2cebf79f.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/65f80f7cf5b721f7132f742de08cc363a8d156d6046aa9e7c6790c2c2cebf79f.jpg)

![716be4e6d28977ae0937f7016f67430434b782240ebf2ff69135fb2c6a528a4f.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/716be4e6d28977ae0937f7016f67430434b782240ebf2ff69135fb2c6a528a4f.jpg)

![75eae6b9b262f00ed06bfa479c1ef768fd6e738bcc254be4e1b20c149efaee28.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/75eae6b9b262f00ed06bfa479c1ef768fd6e738bcc254be4e1b20c149efaee28.jpg)

![786ea2e5db8e92401e42ed872569b5466cc4024480201d72c543135844fcc63a.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/786ea2e5db8e92401e42ed872569b5466cc4024480201d72c543135844fcc63a.jpg)

![817d7066703f504a31d5167c47f3191f4de0a73a9008f59ade3379e5ea576970.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/817d7066703f504a31d5167c47f3191f4de0a73a9008f59ade3379e5ea576970.jpg)

![848066776b53891ab636dd4ab2a99e517d400b7370eabbeceaf7f5702be4e620.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/848066776b53891ab636dd4ab2a99e517d400b7370eabbeceaf7f5702be4e620.jpg)

![862dd30e5c051d00293239c7045c040c669c6c58cf526ef87fa5eaee1efadd67.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/862dd30e5c051d00293239c7045c040c669c6c58cf526ef87fa5eaee1efadd67.jpg)

![8af95a8714142c85db3791421fc148f89dbb037e1ef27ec4bbfc14551378635b.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/8af95a8714142c85db3791421fc148f89dbb037e1ef27ec4bbfc14551378635b.jpg)

![b227fcfd8176fad4469642fdd4fce4e563922dca907aecd2b1b7436fe87f6aaf.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/b227fcfd8176fad4469642fdd4fce4e563922dca907aecd2b1b7436fe87f6aaf.jpg)

![c90ed709e1ce052bb1461593ba5fbf172b0324d0ffa98c1b6caf9eac06d3184b.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/c90ed709e1ce052bb1461593ba5fbf172b0324d0ffa98c1b6caf9eac06d3184b.jpg)

![e3806ebad18438ac1ff55068d3f4b89642ca754dd5f6939e4b41008200d67f88.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/e3806ebad18438ac1ff55068d3f4b89642ca754dd5f6939e4b41008200d67f88.jpg)

![f73ee90cf92405bb72f03de2449808482aa43253c4268d7b71bedd3a359d7e92.jpg](../icml_results/2474_AugMask_%20Training%20Diffusion%20Models%20on%20Incomplete%20Tabular%20Data%20via%20Stochastic%20Augmentation%20and%20Maskin/tables/f73ee90cf92405bb72f03de2449808482aa43253c4268d7b71bedd3a359d7e92.jpg)

<a id="paper-2445"></a>

## Transfer Learning in High-dimensional Ising Models


### Images

![14e24e86bf4b5dd9fdae446077652ed75088f910e469d8a0c7ad98e5f283f6ab.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/14e24e86bf4b5dd9fdae446077652ed75088f910e469d8a0c7ad98e5f283f6ab.jpg)

![5570f9150110720b15858bc87bb601573d57180efa9654d2b09b7a9276d58f99.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/5570f9150110720b15858bc87bb601573d57180efa9654d2b09b7a9276d58f99.jpg)

![5f25518c508c4e19f16f12b3f244a826891b7af208bcc909f0888b0372e42851.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/5f25518c508c4e19f16f12b3f244a826891b7af208bcc909f0888b0372e42851.jpg)

![b3cbc6a4c223fc6689f126153866b2cec5bc24bafe997c091c3d66e2196519de.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/b3cbc6a4c223fc6689f126153866b2cec5bc24bafe997c091c3d66e2196519de.jpg)

![d6fe42f6d46ee04288308f6e8395c170280b56f2776e8b0569fa365742badb02.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/d6fe42f6d46ee04288308f6e8395c170280b56f2776e8b0569fa365742badb02.jpg)

![f39e95eafc3c9cd23ed75df93977583e4df31f2418c3e0149bb44e99c41f3586.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/f39e95eafc3c9cd23ed75df93977583e4df31f2418c3e0149bb44e99c41f3586.jpg)

![fd3ac02cb9d92930a4c1c0e0225708411f4cfa089b111878c3a371dd3929f5ea.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/fd3ac02cb9d92930a4c1c0e0225708411f4cfa089b111878c3a371dd3929f5ea.jpg)

![fe23ad80dd8e430009670450c2e4b1089dcd98e69d4bb0a37aaaecf0d392d153.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/images/fe23ad80dd8e430009670450c2e4b1089dcd98e69d4bb0a37aaaecf0d392d153.jpg)

### Tables

![6c618024ef17960cb5282b0a4d00692f254ebf71e97ca15649c95547a3ec2e51.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/tables/6c618024ef17960cb5282b0a4d00692f254ebf71e97ca15649c95547a3ec2e51.jpg)

![8e7168622a10e85e68ed834f3fd49ba57962e15f94875b2435282ad5751f366c.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/tables/8e7168622a10e85e68ed834f3fd49ba57962e15f94875b2435282ad5751f366c.jpg)

![9f6cd65be1a037663304193fec08ff4f7f2e64c66bdbcdda8180bf49a278b62c.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/tables/9f6cd65be1a037663304193fec08ff4f7f2e64c66bdbcdda8180bf49a278b62c.jpg)

![ccd52194438b81f2eb737be61cde643156d96d54cfcfd07c52d5e1cfb7aeab98.jpg](../icml_results/2475_Transfer%20Learning%20in%20High-dimensional%20Ising%20Models/tables/ccd52194438b81f2eb737be61cde643156d96d54cfcfd07c52d5e1cfb7aeab98.jpg)

<a id="paper-2446"></a>

## CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution


### Images

![1b4a6299655392310fc1e4a8b2a4bca5a953e620653a2300d9d67bf5cac4bf6b.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/1b4a6299655392310fc1e4a8b2a4bca5a953e620653a2300d9d67bf5cac4bf6b.jpg)

![29f9f3f74dae37d73f6117ecab15ce14a9eef9ef7c8f0f598f4d9e5eade6f10d.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/29f9f3f74dae37d73f6117ecab15ce14a9eef9ef7c8f0f598f4d9e5eade6f10d.jpg)

![2f59d9a53d1a87ec6fac19db0c99c4350f7cf91d2a0c296cf52d14cfb32691c3.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/2f59d9a53d1a87ec6fac19db0c99c4350f7cf91d2a0c296cf52d14cfb32691c3.jpg)

![4556dcebce0e82ce85d8b75dc87f83e87ec8c110db6358b9b00097d175bdb369.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/4556dcebce0e82ce85d8b75dc87f83e87ec8c110db6358b9b00097d175bdb369.jpg)

![6bc4ea4e95a6894fef770f5241b747e9c904d489e4158e178f261ec18aa7517a.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/6bc4ea4e95a6894fef770f5241b747e9c904d489e4158e178f261ec18aa7517a.jpg)

![951c7042648a01a56e66eb8c721fa612d6374f772612a088401bd31737d5e5ee.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/951c7042648a01a56e66eb8c721fa612d6374f772612a088401bd31737d5e5ee.jpg)

![be0ac29b80a5b7801209fcb9353fc0f35904167f7058622894da8c7bf01b7e4f.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/be0ac29b80a5b7801209fcb9353fc0f35904167f7058622894da8c7bf01b7e4f.jpg)

![be35c6b9274a40fd6c047e7d364cda75e581c7e48d6a7aaf3377518b1147af13.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/images/be35c6b9274a40fd6c047e7d364cda75e581c7e48d6a7aaf3377518b1147af13.jpg)

### Tables

![17334a87c1c7ce117ba3d0fa4bb27ae4cf624d69507e3e1f0496704028648604.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/tables/17334a87c1c7ce117ba3d0fa4bb27ae4cf624d69507e3e1f0496704028648604.jpg)

![43512fc2b3fdddf47f82a0336b54878519fe911493839f90dd208f940af2707e.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/tables/43512fc2b3fdddf47f82a0336b54878519fe911493839f90dd208f940af2707e.jpg)

![6604bd7af3730bdf4be71ebf7c0f8968143acfb7804bb595709cec11bff7ab23.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/tables/6604bd7af3730bdf4be71ebf7c0f8968143acfb7804bb595709cec11bff7ab23.jpg)

![7bf1355f5a19a696aa4ec86a7133a89b7cf655292d095f3675eb6e19333c53d4.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/tables/7bf1355f5a19a696aa4ec86a7133a89b7cf655292d095f3675eb6e19333c53d4.jpg)

![d4157c900de20aa810331b9de1d2055534ec95899a3803101c3e1edecc409dd6.jpg](../icml_results/2476_CausalArmor_%20Efficient%20Indirect%20Prompt%20Injection%20Guardrails%20via%20Causal%20Attribution/tables/d4157c900de20aa810331b9de1d2055534ec95899a3803101c3e1edecc409dd6.jpg)

<a id="paper-2447"></a>

## StretchTime: Adaptive Time Series Forecasting via Symplectic Attention


### Images

![251c1817a594a472d9a7ce2e244e35a61db3a21231a86e6a86f3158464ca28d1.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/images/251c1817a594a472d9a7ce2e244e35a61db3a21231a86e6a86f3158464ca28d1.jpg)

![ee67410b4a139acbb6785d33296a7fc185a8251a8abb5a14774fc225c625d5cc.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/images/ee67410b4a139acbb6785d33296a7fc185a8251a8abb5a14774fc225c625d5cc.jpg)

![eef971a69cb9447f91a079dd92457809dde54585ed5b81c1978797180b4934c3.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/images/eef971a69cb9447f91a079dd92457809dde54585ed5b81c1978797180b4934c3.jpg)

### Tables

![1de045beaaf73d56923ae35339747ea3aba9b74aa7da9467ce9eccad14be0636.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/1de045beaaf73d56923ae35339747ea3aba9b74aa7da9467ce9eccad14be0636.jpg)

![24c2b1c1cbd6c7f36f732b4facce4cd014ac2e28d9f24c7e37f6780f8b33a042.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/24c2b1c1cbd6c7f36f732b4facce4cd014ac2e28d9f24c7e37f6780f8b33a042.jpg)

![4060a57898cdc63ab428f08efcb79e22cd1266aa3bc7ce0e5fd6abb4dd2c55f2.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/4060a57898cdc63ab428f08efcb79e22cd1266aa3bc7ce0e5fd6abb4dd2c55f2.jpg)

![408389018e335a5ebfe2a418bd249f6b4a45442aea3f225b71caddc8ad77f4b2.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/408389018e335a5ebfe2a418bd249f6b4a45442aea3f225b71caddc8ad77f4b2.jpg)

![7e68ec66751f9fdb1cde73bc2101e515e848bf4816b5a3a0c4f760f23c808641.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/7e68ec66751f9fdb1cde73bc2101e515e848bf4816b5a3a0c4f760f23c808641.jpg)

![c427191b6e713dadadbed6b76074b194cd34812ba9b636daae95940f4aeb6b13.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/c427191b6e713dadadbed6b76074b194cd34812ba9b636daae95940f4aeb6b13.jpg)

![e9c12be2460ce2840606720e59439b70345f5140f5e24f500fa782acfd7f09bd.jpg](../icml_results/2477_StretchTime_%20Adaptive%20Time%20Series%20Forecasting%20via%20Symplectic%20Attention/tables/e9c12be2460ce2840606720e59439b70345f5140f5e24f500fa782acfd7f09bd.jpg)

<a id="paper-2448"></a>

## Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards


### Images

![09b1c6221343b7678a96ed4c242e80eea74e1d120a1d2a33ec20e39c26510742.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/09b1c6221343b7678a96ed4c242e80eea74e1d120a1d2a33ec20e39c26510742.jpg)

![2ff28e204bd82c42bbc60ed322b7ebf3c22dd38d3962b4607afb99aff465af68.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/2ff28e204bd82c42bbc60ed322b7ebf3c22dd38d3962b4607afb99aff465af68.jpg)

![3ad512a6610e6df3f008020a8e7765072a133e90b840d3a8b42b8fbebf8a1878.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/3ad512a6610e6df3f008020a8e7765072a133e90b840d3a8b42b8fbebf8a1878.jpg)

![4083d71c36a57ff7efa35143e05a3bc8199396c62898cf47342dcd1f6bd415ea.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/4083d71c36a57ff7efa35143e05a3bc8199396c62898cf47342dcd1f6bd415ea.jpg)

![897b13b52429aa2a5e7671d6bdfbbd2d6535689ab0bc6bd92c2fefab907e5e29.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/897b13b52429aa2a5e7671d6bdfbbd2d6535689ab0bc6bd92c2fefab907e5e29.jpg)

![b1c640db385433d313fa2171b85e8171624dedbfd5864746d5ba550070a963fd.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/b1c640db385433d313fa2171b85e8171624dedbfd5864746d5ba550070a963fd.jpg)

![de4eec28945f2ecea098b464dbe1c5a95137f29fcdded1da816822322d7c95c1.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/de4eec28945f2ecea098b464dbe1c5a95137f29fcdded1da816822322d7c95c1.jpg)

![f88645e5b33477b83fa0680d06941b3b0fb52d2f6f50b922d524ee71bd875927.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/images/f88645e5b33477b83fa0680d06941b3b0fb52d2f6f50b922d524ee71bd875927.jpg)

### Tables

![1da916046f705ce7d2cae5d994a89911d8f9742dc21af0db4f083cbc48737b4f.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/1da916046f705ce7d2cae5d994a89911d8f9742dc21af0db4f083cbc48737b4f.jpg)

![256ab300821f5f3483eaf5744eee67f13d302844eb56f159930d576530dcc4e9.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/256ab300821f5f3483eaf5744eee67f13d302844eb56f159930d576530dcc4e9.jpg)

![4231981b9fc7d22b1cf6bf123dddafa79459c2ff77216a0e5ff401463b44c558.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/4231981b9fc7d22b1cf6bf123dddafa79459c2ff77216a0e5ff401463b44c558.jpg)

![8ca8c74246dd0086907278f671c6cf351be00b6a9c02c8d7332b51f7124dbd0b.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/8ca8c74246dd0086907278f671c6cf351be00b6a9c02c8d7332b51f7124dbd0b.jpg)

![bf2bf25f7d8c2fa9bdf02e9a3b741954e78183289aee6dcda1f5ac0993feda79.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/bf2bf25f7d8c2fa9bdf02e9a3b741954e78183289aee6dcda1f5ac0993feda79.jpg)

![c38f75ee235abc87878e38da0532d9e2903a9ab9c1b5a4f9949bdbab5e741a24.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/c38f75ee235abc87878e38da0532d9e2903a9ab9c1b5a4f9949bdbab5e741a24.jpg)

![df5460dba63a1512c73bb5d2efe9dadcffabe94429085c3fd721d290e9bed62d.jpg](../icml_results/2478_Discounted%20Beta%E2%80%93Bernoulli%20Reward%20Estimation%20for%20Sample-Efficient%20Reinforcement%20Learning%20with%20Verifia/tables/df5460dba63a1512c73bb5d2efe9dadcffabe94429085c3fd721d290e9bed62d.jpg)

<a id="paper-2449"></a>

## Temporal Preference Optimization for Unsupervised Retrieval


### Images

![31007f593cbc63e862663d5ee4fc353dffcc6428b9c135337f24c7d2a6f8ab59.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/31007f593cbc63e862663d5ee4fc353dffcc6428b9c135337f24c7d2a6f8ab59.jpg)

![3f2d5c6cb55709d6648e4c3b1bf166147a81e6ca82897e90b66daf25fa4f5269.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/3f2d5c6cb55709d6648e4c3b1bf166147a81e6ca82897e90b66daf25fa4f5269.jpg)

![45d2f46857088533360a443d12058b532abdbb4620d9cb49a6178cf338f0e9f8.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/45d2f46857088533360a443d12058b532abdbb4620d9cb49a6178cf338f0e9f8.jpg)

![786dfef172e8783e377c3b3d8e309a2d2d02699c2d406c11de150eae235925ca.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/786dfef172e8783e377c3b3d8e309a2d2d02699c2d406c11de150eae235925ca.jpg)

![7ce9cfc0c5f51ee5a2db9ea2adec5fd9ddcf15b497c50b591546d29958058c74.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/7ce9cfc0c5f51ee5a2db9ea2adec5fd9ddcf15b497c50b591546d29958058c74.jpg)

![89992f9401fd204becc748d8410e6bdae5a8d6f40a38403ec9fd10dcae757751.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/89992f9401fd204becc748d8410e6bdae5a8d6f40a38403ec9fd10dcae757751.jpg)

![906571862223e3bc2f3576cb124445dbb26bca855dcd09f09e4a5089992e3ad5.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/906571862223e3bc2f3576cb124445dbb26bca855dcd09f09e4a5089992e3ad5.jpg)

![90d452d140c03aaba6c286fdc79e7f3a23c2cdfd0e4f001d609388b479824033.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/90d452d140c03aaba6c286fdc79e7f3a23c2cdfd0e4f001d609388b479824033.jpg)

![d22562cce0a8b979024f0f0b172ff0b6ead969e2c47d364103b8bdb69a6876b8.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/d22562cce0a8b979024f0f0b172ff0b6ead969e2c47d364103b8bdb69a6876b8.jpg)

![fb745bb41bf258347095965fc0ef63485efc41ece2d62946afff53a417075f1f.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/images/fb745bb41bf258347095965fc0ef63485efc41ece2d62946afff53a417075f1f.jpg)

### Tables

![0460b71f5435175f8691dc11e268496fd7f87850d4883607b949f2ba7261b1dd.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/0460b71f5435175f8691dc11e268496fd7f87850d4883607b949f2ba7261b1dd.jpg)

![2f9aef6a9c04def61736e2a06cd847c8bf595077e2366f7ee20f8e05f146833f.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/2f9aef6a9c04def61736e2a06cd847c8bf595077e2366f7ee20f8e05f146833f.jpg)

![332067ad198bf22aaef9714f9787a9f5b2a5f76c103f94eedffd6e1f7714dc0e.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/332067ad198bf22aaef9714f9787a9f5b2a5f76c103f94eedffd6e1f7714dc0e.jpg)

![3b7e4d9cbe9b92898c6539fd7f38e71581b9aebe0637ba62528fa697edb96570.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/3b7e4d9cbe9b92898c6539fd7f38e71581b9aebe0637ba62528fa697edb96570.jpg)

![4047b5a7bafbf31bddb98c4d277500a0c78f62b25b39d60bbb670042eb583612.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/4047b5a7bafbf31bddb98c4d277500a0c78f62b25b39d60bbb670042eb583612.jpg)

![42404da962be69eebf64245db4f59cbab151cafce7db74fb9db5ec6563518588.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/42404da962be69eebf64245db4f59cbab151cafce7db74fb9db5ec6563518588.jpg)

![4dc44e9be25cd961c813f9415ce0294a278324c89a8cdcc07b5ae8c3fefc25dc.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/4dc44e9be25cd961c813f9415ce0294a278324c89a8cdcc07b5ae8c3fefc25dc.jpg)

![4e7487d77be79d802a325377136dea20bf5f6212928b66ee1bea36a82fdbfe38.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/4e7487d77be79d802a325377136dea20bf5f6212928b66ee1bea36a82fdbfe38.jpg)

![6c84d0bda15dba049071cbe480475f08f8739b8687a2744f5c2c4c27710ed246.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/6c84d0bda15dba049071cbe480475f08f8739b8687a2744f5c2c4c27710ed246.jpg)

![6d87a085ff356e0ab5b8b9ea8a5d89109d4d71d9aad444e804ddae7563f4105e.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/6d87a085ff356e0ab5b8b9ea8a5d89109d4d71d9aad444e804ddae7563f4105e.jpg)

![6e46d0bd85236ce5613a003cae8ba76f8533dd7f527c8ee21ebf1a4c4dbe2fd3.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/6e46d0bd85236ce5613a003cae8ba76f8533dd7f527c8ee21ebf1a4c4dbe2fd3.jpg)

![6f1ea3e69712b732a69b0a93c78e8464b2d951f1014583e188bb879f9b4dbfe1.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/6f1ea3e69712b732a69b0a93c78e8464b2d951f1014583e188bb879f9b4dbfe1.jpg)

![74044d2ae259922c5d247d165ff437836ccd5b771ca873b8c9614be1888ff446.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/74044d2ae259922c5d247d165ff437836ccd5b771ca873b8c9614be1888ff446.jpg)

![7aa91ca0d52d97ecaff8e458a5a906ce7992c9bc7ff81b4948c24c484fe8e039.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/7aa91ca0d52d97ecaff8e458a5a906ce7992c9bc7ff81b4948c24c484fe8e039.jpg)

![80813ae838c89766cf7d88014d8f3e836d11670b284060dcc08c856a8a28b43b.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/80813ae838c89766cf7d88014d8f3e836d11670b284060dcc08c856a8a28b43b.jpg)

![88f7e4b4b3a71e059574c65c4075590bf310ce529c439c9e141951fbc77fc0a1.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/88f7e4b4b3a71e059574c65c4075590bf310ce529c439c9e141951fbc77fc0a1.jpg)

![8e552ff9d279a563afa799815ac72d2488e159fb54a18cd96028aae54ccde58d.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/8e552ff9d279a563afa799815ac72d2488e159fb54a18cd96028aae54ccde58d.jpg)

![98d275e2b95cfa323178c7a13157d0c4128109c4ef0abb050d25c5c8242044e6.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/98d275e2b95cfa323178c7a13157d0c4128109c4ef0abb050d25c5c8242044e6.jpg)

![9bed59f602051f4ab458c78fae0f1db1f154cc2c0be02d1c285d322ca941a571.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/9bed59f602051f4ab458c78fae0f1db1f154cc2c0be02d1c285d322ca941a571.jpg)

![9bf8ca5b9a35ad63e06b4b7fc14821b5ff7f0ed020520c18e03bf079158f316b.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/9bf8ca5b9a35ad63e06b4b7fc14821b5ff7f0ed020520c18e03bf079158f316b.jpg)

![9dc32ab43513a2076dff5ed4677bb99835e4aeac1f367bc54862f9acc5bc8cf1.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/9dc32ab43513a2076dff5ed4677bb99835e4aeac1f367bc54862f9acc5bc8cf1.jpg)

![bdc5a328591bcb5f1b2fe90911d21c10380f2b30b08488339b934414bbb71013.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/bdc5a328591bcb5f1b2fe90911d21c10380f2b30b08488339b934414bbb71013.jpg)

![c0afb50b2efdbeb28c6a2b13392f3dff53e0beea048581010b72826d8fe1e4db.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/c0afb50b2efdbeb28c6a2b13392f3dff53e0beea048581010b72826d8fe1e4db.jpg)

![d4ed32ae8deed67420f05ff5da4d3645a55bd6a116081e5f4cdea377233215a4.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/d4ed32ae8deed67420f05ff5da4d3645a55bd6a116081e5f4cdea377233215a4.jpg)

![ee125e5f79143ba80df5bdfe0c393070dac89ca8164d32cf7e9b54f63fc7a68e.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/ee125e5f79143ba80df5bdfe0c393070dac89ca8164d32cf7e9b54f63fc7a68e.jpg)

![eed11a7cb53553e43fa538cb934b406a936ea5faa0a2a5ae853a7d32900624c5.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/eed11a7cb53553e43fa538cb934b406a936ea5faa0a2a5ae853a7d32900624c5.jpg)

![fbd8ee5cf202fbae6d198f619d1b0876fa0db6445172a0d7589f5a2c6847c309.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/fbd8ee5cf202fbae6d198f619d1b0876fa0db6445172a0d7589f5a2c6847c309.jpg)

![feb1fad9af30a638824a51e26b49487930115b7168c6f4906a69d36f86937804.jpg](../icml_results/2479_Temporal%20Preference%20Optimization%20for%20Unsupervised%20Retrieval/tables/feb1fad9af30a638824a51e26b49487930115b7168c6f4906a69d36f86937804.jpg)

<a id="paper-2450"></a>

## DNACHUNKER: Learnable Tokenization for DNA Language Models


### Images

![096381bcb8d460c7ca9a63bd4e04a876163553f3a13dddb51d67332e07a6a0d8.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/images/096381bcb8d460c7ca9a63bd4e04a876163553f3a13dddb51d67332e07a6a0d8.jpg)

![19b52771020b94267363dd44d3063a5f450e9a571e3a0d1a7781a8a3884fb78d.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/images/19b52771020b94267363dd44d3063a5f450e9a571e3a0d1a7781a8a3884fb78d.jpg)

![2db3958ada25b1c94257923096d177074d65970bc6da780f55dbf50f81dab351.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/images/2db3958ada25b1c94257923096d177074d65970bc6da780f55dbf50f81dab351.jpg)

![6543bd434e1709ca32a284f283ccaa925c481aeeef995a07d076bd32d5549d10.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/images/6543bd434e1709ca32a284f283ccaa925c481aeeef995a07d076bd32d5549d10.jpg)

![ac2a747471a398ef38fe84867d5f35016d831eae892bc42f91abc2292b2dcf0a.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/images/ac2a747471a398ef38fe84867d5f35016d831eae892bc42f91abc2292b2dcf0a.jpg)

### Tables

![3e7721986f4ab91e97a12b7d2d688a2cfe2c8ca80e55ae7a867cb571d3fdd647.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/3e7721986f4ab91e97a12b7d2d688a2cfe2c8ca80e55ae7a867cb571d3fdd647.jpg)

![62548ee3b14f56ac4c5d291f11b76bae63451cb50f30d8730ae615c110f2544d.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/62548ee3b14f56ac4c5d291f11b76bae63451cb50f30d8730ae615c110f2544d.jpg)

![67090eae9ec47074c90a16e5b280d1431186275ea5d79f3567b9113e58779122.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/67090eae9ec47074c90a16e5b280d1431186275ea5d79f3567b9113e58779122.jpg)

![752b5d29f15578beed6933033d35ee08086054cd0fa9302b178b33034f159088.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/752b5d29f15578beed6933033d35ee08086054cd0fa9302b178b33034f159088.jpg)

![92de67d71db001e262806874301c63b31a62aa953904f6fb0dfa300af51593fb.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/92de67d71db001e262806874301c63b31a62aa953904f6fb0dfa300af51593fb.jpg)

![b5a31c77949741541f945afd79748edd0ef580e92cdf695fc5ab3ab718f756d1.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/b5a31c77949741541f945afd79748edd0ef580e92cdf695fc5ab3ab718f756d1.jpg)

![c0ad3c90341c029655ef34a7b082549ab1e3b9559d0dec99912591f5ba9faa53.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/c0ad3c90341c029655ef34a7b082549ab1e3b9559d0dec99912591f5ba9faa53.jpg)

![c13436b0c379b7fc9242873c81eecc4daec6d256640343df3e14fcabe11fed16.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/c13436b0c379b7fc9242873c81eecc4daec6d256640343df3e14fcabe11fed16.jpg)

![c455648da7c5f5322b590d865ef590d745cfcf5c3b16a6a808d05157e54f18b2.jpg](../icml_results/2480_DNACHUNKER_%20Learnable%20Tokenization%20for%20DNA%20Language%20Models/tables/c455648da7c5f5322b590d865ef590d745cfcf5c3b16a6a808d05157e54f18b2.jpg)

<a id="paper-2451"></a>

## EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs


### Images

![062cd15616c4a5f0ab93968e99cc5127005fec9a44e01b4a5163847d57978b18.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/062cd15616c4a5f0ab93968e99cc5127005fec9a44e01b4a5163847d57978b18.jpg)

![084e8bea9b9af64a0f82d08259c6c55cd8ad59b17f39b9f9c372d23288c882c0.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/084e8bea9b9af64a0f82d08259c6c55cd8ad59b17f39b9f9c372d23288c882c0.jpg)

![0a04f26ba1a0bca1f59200aad583be359605b56a79413704f2c2e0ca4f6d7a06.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/0a04f26ba1a0bca1f59200aad583be359605b56a79413704f2c2e0ca4f6d7a06.jpg)

![162d8ae6b01288a506b73106f3840a47010b0c5475cef17490a73294751d1149.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/162d8ae6b01288a506b73106f3840a47010b0c5475cef17490a73294751d1149.jpg)

![18df1d156553cf20f48f3e60aa00e6d513f6d559962334d4969400421882beec.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/18df1d156553cf20f48f3e60aa00e6d513f6d559962334d4969400421882beec.jpg)

![769f5e5821d2e5858778b11ff3b3fbb677d2c4cdab49d314ba4e445f62749152.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/769f5e5821d2e5858778b11ff3b3fbb677d2c4cdab49d314ba4e445f62749152.jpg)

![81cbab0ac6f5dc36834d9b297b662cb9490bc9c99c6d85c3611703492df1f49f.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/81cbab0ac6f5dc36834d9b297b662cb9490bc9c99c6d85c3611703492df1f49f.jpg)

![845762617f0057e8638283788390180c1ee09f056f85f4a0b5bcebe086a01ef4.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/845762617f0057e8638283788390180c1ee09f056f85f4a0b5bcebe086a01ef4.jpg)

![96314355254ee770a7e9a7a543dfb60948e9c96b32b16b6cc4cbbbc38eef9072.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/96314355254ee770a7e9a7a543dfb60948e9c96b32b16b6cc4cbbbc38eef9072.jpg)

![9d664e4e0d8b302ebbaefaccafe050555046d7ee80a5ace7997a70c9f3ab2b91.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/9d664e4e0d8b302ebbaefaccafe050555046d7ee80a5ace7997a70c9f3ab2b91.jpg)

![a1ee746adda371b6c50b759e417de5dd2922a16f42573eee5e0f9fe707e2f574.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/a1ee746adda371b6c50b759e417de5dd2922a16f42573eee5e0f9fe707e2f574.jpg)

![a5da0ce0004e639b7841cdc4d23861e6fe683033756f0cbdbb275e2e6e58b299.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/a5da0ce0004e639b7841cdc4d23861e6fe683033756f0cbdbb275e2e6e58b299.jpg)

![a73f62c87cc7a3c910f94950623dc9a10612183cecd1b43e966c9ddfe3ae0806.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/a73f62c87cc7a3c910f94950623dc9a10612183cecd1b43e966c9ddfe3ae0806.jpg)

![c4a8ad33c0ae4d8c419d5ac30cf42d4969c23f444c7347901db0c9c3883239c1.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/c4a8ad33c0ae4d8c419d5ac30cf42d4969c23f444c7347901db0c9c3883239c1.jpg)

![d23ac93dcb600ee11e25cf8aed46511115f09682e690289d7e41e6f0e801dce5.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/d23ac93dcb600ee11e25cf8aed46511115f09682e690289d7e41e6f0e801dce5.jpg)

![db6344ef35189a3bfb18a5e1fe5ce06b67e1999136abc9d354d9a3359d786d93.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/db6344ef35189a3bfb18a5e1fe5ce06b67e1999136abc9d354d9a3359d786d93.jpg)

![e147544a8204e6b5559382af280e4cb923801c984e2044d491c2a21c4c5ed9db.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/e147544a8204e6b5559382af280e4cb923801c984e2044d491c2a21c4c5ed9db.jpg)

![e5780956bc25b57eb6f37f0b101a2c8409ed44f61fddbb1cdde0f4d5bad24b60.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/e5780956bc25b57eb6f37f0b101a2c8409ed44f61fddbb1cdde0f4d5bad24b60.jpg)

![edfd7fa97cc4ed5ee8f776171dedc3f4500423e4ff22d19756ae58b76e532dd1.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/edfd7fa97cc4ed5ee8f776171dedc3f4500423e4ff22d19756ae58b76e532dd1.jpg)

![fb1c65cb1d511d494f7264821fa170ebaeb0cf06dc39bc1b928d58ad12007a8b.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/images/fb1c65cb1d511d494f7264821fa170ebaeb0cf06dc39bc1b928d58ad12007a8b.jpg)

### Tables

![37be7a29c4c0d4a758923da534db9846fa8ae0192523a719b0aaae6420475ce0.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/tables/37be7a29c4c0d4a758923da534db9846fa8ae0192523a719b0aaae6420475ce0.jpg)

![b800fb4dfc017bef64b7c4e2a466eaad1507a2c045f1b4123215cfab0e81f1ad.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/tables/b800fb4dfc017bef64b7c4e2a466eaad1507a2c045f1b4123215cfab0e81f1ad.jpg)

![c244799fe8760571ddf6b7c452476b85f16ca1a2084775f673ba68ad9b63a706.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/tables/c244799fe8760571ddf6b7c452476b85f16ca1a2084775f673ba68ad9b63a706.jpg)

![e686d6d5de86c784084d7f3883f0371275186f394cc1a6017c3228975eae457a.jpg](../icml_results/2481_EqGINO_%20Equivariant%20Geometry-Informed%20Fourier%20Neural%20Operators%20for%203D%20PDEs/tables/e686d6d5de86c784084d7f3883f0371275186f394cc1a6017c3228975eae457a.jpg)

<a id="paper-2452"></a>

## Identifiable Token Correspondence for World Models


### Images

![197620938d6fcb93c93708cad024fa50bc15cd2c497ea2d484b7bd010e042f43.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/197620938d6fcb93c93708cad024fa50bc15cd2c497ea2d484b7bd010e042f43.jpg)

![2fb7e3eadd5c4512bedc8763788aa091b6aa216cbc60e587421672234acc96d6.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/2fb7e3eadd5c4512bedc8763788aa091b6aa216cbc60e587421672234acc96d6.jpg)

![5d57556038ab5050cb9a8cb5bbfcc7793e4e406f9cc0001159653c97b0414520.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/5d57556038ab5050cb9a8cb5bbfcc7793e4e406f9cc0001159653c97b0414520.jpg)

![9111185a60a904f94b56130d01b4192e734f4542372dba2e7fa4d676345021fe.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/9111185a60a904f94b56130d01b4192e734f4542372dba2e7fa4d676345021fe.jpg)

![ab5ae54763b436f7d69c0e23cdbdf66e5403d541b0cc9aa6be59ddc272279a50.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/ab5ae54763b436f7d69c0e23cdbdf66e5403d541b0cc9aa6be59ddc272279a50.jpg)

![d28f7789ea4ad8e688f745a2f45c2918c4d8b2ef06377071290630e102d51b43.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/d28f7789ea4ad8e688f745a2f45c2918c4d8b2ef06377071290630e102d51b43.jpg)

![df125c0613f7f7db273c4584c7c3303154e61918332c02a10157f61cc22648cf.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/images/df125c0613f7f7db273c4584c7c3303154e61918332c02a10157f61cc22648cf.jpg)

### Tables

![03d379d7d6daf17ae3f45853dbd07661c8edf3925d9e341f0b0a2ad1630fee75.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/03d379d7d6daf17ae3f45853dbd07661c8edf3925d9e341f0b0a2ad1630fee75.jpg)

![05707137dc5d0aeecb68e60cc8ae86d8902012cd7ed2b3dc48d6888a691a5b29.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/05707137dc5d0aeecb68e60cc8ae86d8902012cd7ed2b3dc48d6888a691a5b29.jpg)

![11276f7c38ffe0893197c85589cc06a3dcf76f8083eff9485cdd0ca4c09352af.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/11276f7c38ffe0893197c85589cc06a3dcf76f8083eff9485cdd0ca4c09352af.jpg)

![12a881385494fd3fc676604726f9d7ff8bf257d25a1889883a68850f409634c9.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/12a881385494fd3fc676604726f9d7ff8bf257d25a1889883a68850f409634c9.jpg)

![253d2ce67e027bd6cd1c570b5f2fcbdd825a481d5db5eaa70cd8fdc9460739a3.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/253d2ce67e027bd6cd1c570b5f2fcbdd825a481d5db5eaa70cd8fdc9460739a3.jpg)

![2b2526495a4a51536c70a0c4d74ffe3778ed7c156951dc26bb73eed3ecdecd2a.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/2b2526495a4a51536c70a0c4d74ffe3778ed7c156951dc26bb73eed3ecdecd2a.jpg)

![348758f8000a96450544318fcb87d8f5db8eb21dc2739b4191044aef621511bd.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/348758f8000a96450544318fcb87d8f5db8eb21dc2739b4191044aef621511bd.jpg)

![46c46071cc425ab24fc99333b4428bb97c4a6fa6685c291ddda81438b40e0f51.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/46c46071cc425ab24fc99333b4428bb97c4a6fa6685c291ddda81438b40e0f51.jpg)

![4aa3b934865d0e5094da2d44ae8956386522315dad1e040bc620d1c3bed64b0b.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/4aa3b934865d0e5094da2d44ae8956386522315dad1e040bc620d1c3bed64b0b.jpg)

![5f953d68d458ed1a9e99a17faf108fc1ed8ba99512a9dd2f99431c11108e6fa7.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/5f953d68d458ed1a9e99a17faf108fc1ed8ba99512a9dd2f99431c11108e6fa7.jpg)

![77c2c745ed22439fb1be0d70698899ca7ba656d7cc866be2e30c9446e62a5357.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/77c2c745ed22439fb1be0d70698899ca7ba656d7cc866be2e30c9446e62a5357.jpg)

![80ba79ddae1ffbb36c1faac501056c842b738d991b1972e6993777b945608092.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/80ba79ddae1ffbb36c1faac501056c842b738d991b1972e6993777b945608092.jpg)

![a9547180ae2c9f81ce2a1fb831b5a6186a61244bf92d5be324a3699bc6c17df2.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/a9547180ae2c9f81ce2a1fb831b5a6186a61244bf92d5be324a3699bc6c17df2.jpg)

![bcb6aea4d17b8dbf97126dc83ec4d4aff605aacb216047dfc8e313129af730eb.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/bcb6aea4d17b8dbf97126dc83ec4d4aff605aacb216047dfc8e313129af730eb.jpg)

![c66d6a0bfcca951cdb2436c46d6a696eb54405190a1799e89443ef8ea2fd422c.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/c66d6a0bfcca951cdb2436c46d6a696eb54405190a1799e89443ef8ea2fd422c.jpg)

![c73684884bb4595a8f9e4e959a719ddbd8aeacf38e1cff9ab9b165fc9f683e4a.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/c73684884bb4595a8f9e4e959a719ddbd8aeacf38e1cff9ab9b165fc9f683e4a.jpg)

![db3d554df36bb11420db177356c27a1f10783b4418f7d56c0271fb681adbaad1.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/db3d554df36bb11420db177356c27a1f10783b4418f7d56c0271fb681adbaad1.jpg)

![dfd878b72b5f4c15b35638e0ee1cfa63da2bb6effc16a58269435a59e0825eea.jpg](../icml_results/2482_Identifiable%20Token%20Correspondence%20for%20World%20Models/tables/dfd878b72b5f4c15b35638e0ee1cfa63da2bb6effc16a58269435a59e0825eea.jpg)

<a id="paper-2453"></a>

## Persona-Pruner: Sculpting Lightweight Models for Role-Playing


### Images

![10264fe3b86efdf28c2c8e5045f66551bf6dcbf5d3375b2594233094dc1799e7.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/images/10264fe3b86efdf28c2c8e5045f66551bf6dcbf5d3375b2594233094dc1799e7.jpg)

![21deff12048479129851e2726bac2f4615774d8bfe2db63df8f9f64892360211.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/images/21deff12048479129851e2726bac2f4615774d8bfe2db63df8f9f64892360211.jpg)

![32cd6eb91203e6bff8a3a831cac9162f1d0960694fecda99011044a27cba2a20.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/images/32cd6eb91203e6bff8a3a831cac9162f1d0960694fecda99011044a27cba2a20.jpg)

![737a5f13ac4d67cb8cc8ae45db61160ac9ce4bc28c6ab48acb7f1bfb713bfdfb.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/images/737a5f13ac4d67cb8cc8ae45db61160ac9ce4bc28c6ab48acb7f1bfb713bfdfb.jpg)

![e55d4403735ac62edc5dacec651b69b8b6c5e61450288cb8e0682c1ed214d47e.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/images/e55d4403735ac62edc5dacec651b69b8b6c5e61450288cb8e0682c1ed214d47e.jpg)

### Tables

![21f89bd158da5cef47839d996b4739259497c180a40f38d925865b63054a7c5e.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/21f89bd158da5cef47839d996b4739259497c180a40f38d925865b63054a7c5e.jpg)

![33923b4f7ef414a3a475343c8b10d3c88ab72aee4ddf73d1277a86957f6c9272.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/33923b4f7ef414a3a475343c8b10d3c88ab72aee4ddf73d1277a86957f6c9272.jpg)

![665d49ad08dfe75d3dc6c38c548680901db025fa73406e9fee7b8d85c90a644a.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/665d49ad08dfe75d3dc6c38c548680901db025fa73406e9fee7b8d85c90a644a.jpg)

![680ba990806cedaf1a6e87eb445357002c831ab36117fbe31069fd06d2e8500b.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/680ba990806cedaf1a6e87eb445357002c831ab36117fbe31069fd06d2e8500b.jpg)

![6ef16711ce788f4f0e48113359f820c788b4311973e5c6caed94d41140f33cb4.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/6ef16711ce788f4f0e48113359f820c788b4311973e5c6caed94d41140f33cb4.jpg)

![72d96632f517660cc1109c277dc88a8b0aa9e15688d40f0656774ab388845eaa.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/72d96632f517660cc1109c277dc88a8b0aa9e15688d40f0656774ab388845eaa.jpg)

![79059af2bad691191286db0f00ecef683a0391cc61cba16190010440a4df43ac.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/79059af2bad691191286db0f00ecef683a0391cc61cba16190010440a4df43ac.jpg)

![7e884622891b05ee9d8e310204e0dac50177690574c61ac0176c81397c46279f.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/7e884622891b05ee9d8e310204e0dac50177690574c61ac0176c81397c46279f.jpg)

![8efdfff310315c0b0f72a5e8bc5a70ab207b4fd06edff60dafe9e6f485c05384.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/8efdfff310315c0b0f72a5e8bc5a70ab207b4fd06edff60dafe9e6f485c05384.jpg)

![b5976101329cb1d5b489918449798ecd510d77f3670bb9fae0bbf66629152a12.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/b5976101329cb1d5b489918449798ecd510d77f3670bb9fae0bbf66629152a12.jpg)

![bb10d78bee2949bca03fb5b77c0b8f70f0b706324aadbe493beac8cc25f49a6b.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/bb10d78bee2949bca03fb5b77c0b8f70f0b706324aadbe493beac8cc25f49a6b.jpg)

![c6a95d21c150ba5acd68537d19b3628656bc1d362ed5293c108e4be7b1931a85.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/c6a95d21c150ba5acd68537d19b3628656bc1d362ed5293c108e4be7b1931a85.jpg)

![cb4d3c527ee13dd1a2f149142abaed144f57c8426e9a313d778b12e91c461865.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/cb4d3c527ee13dd1a2f149142abaed144f57c8426e9a313d778b12e91c461865.jpg)

![ce9ca4d25da165dc6e508c8a392f5784ae547b5113dcee9f174fd35dbd99ca9c.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/ce9ca4d25da165dc6e508c8a392f5784ae547b5113dcee9f174fd35dbd99ca9c.jpg)

![d68d9945413d484300dc370d9880ea71dcb204a00df50a100bc8af99a6549525.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/d68d9945413d484300dc370d9880ea71dcb204a00df50a100bc8af99a6549525.jpg)

![da8821ebf75b0f2f580f48a5dd4b0aaacd1f842c844e8a03fff7727d10cc98ba.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/da8821ebf75b0f2f580f48a5dd4b0aaacd1f842c844e8a03fff7727d10cc98ba.jpg)

![e27f3d9b46fa3c5e6ef6e667d2db27cc3e0bd375559ac3696717d51bca354f18.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/e27f3d9b46fa3c5e6ef6e667d2db27cc3e0bd375559ac3696717d51bca354f18.jpg)

![fddc21c4128d651b21b07b13d6f8ed6976ab05ed3e3be35917823f9e5da96c76.jpg](../icml_results/2483_Persona-Pruner_%20Sculpting%20Lightweight%20Models%20for%20Role-Playing/tables/fddc21c4128d651b21b07b13d6f8ed6976ab05ed3e3be35917823f9e5da96c76.jpg)

<a id="paper-2454"></a>

## Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents


### Images

![06d1e583862b596a8ed5f6d905c02370141daf5f51cd00abba1cfc4547dfa7c6.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/06d1e583862b596a8ed5f6d905c02370141daf5f51cd00abba1cfc4547dfa7c6.jpg)

![4d895d292dab36599f7c441ae35637811fff167249b933b66c25dff93515079d.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/4d895d292dab36599f7c441ae35637811fff167249b933b66c25dff93515079d.jpg)

![675c758b9e09cdab15b6c017a2d8f651bf84775e354b6d830bc37c0fdd274a34.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/675c758b9e09cdab15b6c017a2d8f651bf84775e354b6d830bc37c0fdd274a34.jpg)

![7a2ddc62056119bad8c8d14c4e579830cc93a7f1506c6b375eec831eb397012a.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/7a2ddc62056119bad8c8d14c4e579830cc93a7f1506c6b375eec831eb397012a.jpg)

![8b5bd32b929d9085d295b9512ec389a2afef57daa40de04e1f13b8701f2e54ff.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/8b5bd32b929d9085d295b9512ec389a2afef57daa40de04e1f13b8701f2e54ff.jpg)

![b09e5c7574df5db46747f5cc2a9cabf7f86f2d6e54ad476cb86535b7176f168c.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/b09e5c7574df5db46747f5cc2a9cabf7f86f2d6e54ad476cb86535b7176f168c.jpg)

![bdd01639b9335a45e6c201e6c83c000993164608d89f3bc7d81d8869dd4ba424.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/bdd01639b9335a45e6c201e6c83c000993164608d89f3bc7d81d8869dd4ba424.jpg)

![c2ebdf9bb9a586f3eb8db9f96b3ae4421d2c22a5388e943c5c1d93d91e2a3d01.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/c2ebdf9bb9a586f3eb8db9f96b3ae4421d2c22a5388e943c5c1d93d91e2a3d01.jpg)

![cb00811751197672dbc5e2de1b08eaebe0f6d442de3c0b881c26cf1dd2eefb9b.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/cb00811751197672dbc5e2de1b08eaebe0f6d442de3c0b881c26cf1dd2eefb9b.jpg)

![cb61ba9817724623782b4c0d01c1eb7dc738afc57a281f37385a6f700eb1a652.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/cb61ba9817724623782b4c0d01c1eb7dc738afc57a281f37385a6f700eb1a652.jpg)

![f96b93ed91a90605b81f64e164a36c226405b0a2935e5f8182f15bdee2d16027.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/f96b93ed91a90605b81f64e164a36c226405b0a2935e5f8182f15bdee2d16027.jpg)

![fb201883a5524316d4b91060c40afe426f8fbf480a4b9d434ce2bef6ce8b1df9.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/images/fb201883a5524316d4b91060c40afe426f8fbf480a4b9d434ce2bef6ce8b1df9.jpg)

### Tables

![033308a5bc881cdd1dddeeccac5ba9a69fb7b9fdf8e82e014c9a09d60d9a97cc.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/033308a5bc881cdd1dddeeccac5ba9a69fb7b9fdf8e82e014c9a09d60d9a97cc.jpg)

![0a89459bb3ed0fb2af86c93f66c94ba1dfbd095e5377fdfd4fc4db80f0e492cd.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/0a89459bb3ed0fb2af86c93f66c94ba1dfbd095e5377fdfd4fc4db80f0e492cd.jpg)

![4ffbbbad66e1757c8f27668d7a8e1ba5cdd9a13be4366e59c366db890e90e33c.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/4ffbbbad66e1757c8f27668d7a8e1ba5cdd9a13be4366e59c366db890e90e33c.jpg)

![6358f89398a39a008c85eb02457c2d2b62f49fd5a9f7547cdab51783dceddfbe.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/6358f89398a39a008c85eb02457c2d2b62f49fd5a9f7547cdab51783dceddfbe.jpg)

![82fa2dc5e6b48ac53805530a3740a13b36d16ba8d2acc61f891fb520c4210ad6.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/82fa2dc5e6b48ac53805530a3740a13b36d16ba8d2acc61f891fb520c4210ad6.jpg)

![f023827f3088db2cf4bf4a269b383efbce4fcb60564dae027aeebdaa419cca30.jpg](../icml_results/2484_Diversity%20Over%20Frequency_%20Rethinking%20Tool%20Use%20in%20Visual%20Chain-of-Thought%20Agents/tables/f023827f3088db2cf4bf4a269b383efbce4fcb60564dae027aeebdaa419cca30.jpg)

<a id="paper-2455"></a>

## SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models


### Images

![125291ff7c09291cbd7aa90f5babfe4c2def4bc1305e50c71f35e8568c350b00.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/125291ff7c09291cbd7aa90f5babfe4c2def4bc1305e50c71f35e8568c350b00.jpg)

![231354926aa97d8799844206f7991164b5511a560dfab2aedf941b1906449b14.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/231354926aa97d8799844206f7991164b5511a560dfab2aedf941b1906449b14.jpg)

![24b46b60763d6276289f3fda94942376e8de01424a3cd18d65d8b9b291ce1915.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/24b46b60763d6276289f3fda94942376e8de01424a3cd18d65d8b9b291ce1915.jpg)

![2ef7fd8e0678080f58e665b1270a64f35f8fca46b2c4a027e1209a749c723816.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/2ef7fd8e0678080f58e665b1270a64f35f8fca46b2c4a027e1209a749c723816.jpg)

![3deda2b701be439063e37878622b4da7ab8090fc0b4b7f57a80ef1f1c54cc436.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/3deda2b701be439063e37878622b4da7ab8090fc0b4b7f57a80ef1f1c54cc436.jpg)

![520de25edcc3ac849561c3399b65212e307f1ee92a9f6ada931022fdf2a7c0a7.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/520de25edcc3ac849561c3399b65212e307f1ee92a9f6ada931022fdf2a7c0a7.jpg)

![53127501f75eba60970fcd86309000d683abf19a23ea29a9ed852d37d6884890.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/53127501f75eba60970fcd86309000d683abf19a23ea29a9ed852d37d6884890.jpg)

![60d77cf6983171f110923eaccc9a23dc226c23765d3e2ca941ee1f5995828886.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/60d77cf6983171f110923eaccc9a23dc226c23765d3e2ca941ee1f5995828886.jpg)

![912ac92e4cc774de38dac3b26847db79e78d231ce4eb6b222edbca341363fe12.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/912ac92e4cc774de38dac3b26847db79e78d231ce4eb6b222edbca341363fe12.jpg)

![e9ff113b8bd253c54b3e9b8ecc79391e2ad67097c83a99323b4d2103dc853c65.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/images/e9ff113b8bd253c54b3e9b8ecc79391e2ad67097c83a99323b4d2103dc853c65.jpg)

### Tables

![127a5be526835a5ea5392a2f3f543491a63e67a4d1b936eab5aa84d3d09eb241.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/127a5be526835a5ea5392a2f3f543491a63e67a4d1b936eab5aa84d3d09eb241.jpg)

![26f4b125397e445de205a764c79c26c4ae156014d029df291814f9c8684027ee.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/26f4b125397e445de205a764c79c26c4ae156014d029df291814f9c8684027ee.jpg)

![2764c0fec0272473d4878d7bc77dad4aac70a4ccf8d9bce07decf32ded741193.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/2764c0fec0272473d4878d7bc77dad4aac70a4ccf8d9bce07decf32ded741193.jpg)

![2f09d4936ecdbf159e77b80ba6b4b8a69662d5b11b970eef51a084d210f62d05.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/2f09d4936ecdbf159e77b80ba6b4b8a69662d5b11b970eef51a084d210f62d05.jpg)

![44a0c47788a85d95a6db0dfa51c84761aa13b19ac6444fd17836fb15ae162947.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/44a0c47788a85d95a6db0dfa51c84761aa13b19ac6444fd17836fb15ae162947.jpg)

![578ec94bbfe328ae2afa3da7341c665ca07372adc29960cae97b8bdba77cbcfc.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/578ec94bbfe328ae2afa3da7341c665ca07372adc29960cae97b8bdba77cbcfc.jpg)

![7c5d278aa51281a2412c3cdec851e45a365623408af4b8ff7cf87890fe462698.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/7c5d278aa51281a2412c3cdec851e45a365623408af4b8ff7cf87890fe462698.jpg)

![7f706c8596c9d9c0257392f1cd4863d93072ac653f0a3194770f18dadcf7881f.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/7f706c8596c9d9c0257392f1cd4863d93072ac653f0a3194770f18dadcf7881f.jpg)

![82dd5f00808c5034818306962e4b57823417c437a54139d0eb41bb1d9d1c8f8c.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/82dd5f00808c5034818306962e4b57823417c437a54139d0eb41bb1d9d1c8f8c.jpg)

![89775a650bbb6b8b34aa21cecb537d9b22e14f2fc6082e546828844b116159f0.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/89775a650bbb6b8b34aa21cecb537d9b22e14f2fc6082e546828844b116159f0.jpg)

![8dbb2eb1e1583e1a5b8515bfd9101a8d7af66348694436d302d80a7bd982f13d.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/8dbb2eb1e1583e1a5b8515bfd9101a8d7af66348694436d302d80a7bd982f13d.jpg)

![a0b2622dea5ae18f67c38f40d84a941046357e1cda0b41072b39fa50570896be.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/a0b2622dea5ae18f67c38f40d84a941046357e1cda0b41072b39fa50570896be.jpg)

![b756cb2a0cce867fdd0c69602394499156d148ba35c996fe4deae661d4af8648.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/b756cb2a0cce867fdd0c69602394499156d148ba35c996fe4deae661d4af8648.jpg)

![bdbc957f49524de23f864107747ae2d907c7bbf168c6a6c5da86881f3aa0f8ec.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/bdbc957f49524de23f864107747ae2d907c7bbf168c6a6c5da86881f3aa0f8ec.jpg)

![d22932920e71cd4b4a052cc070725cdf8dbca04502601b5adc7d21b4e4429812.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/d22932920e71cd4b4a052cc070725cdf8dbca04502601b5adc7d21b4e4429812.jpg)

![e870b7865b573e047d7d3e27c433f82e1558ae876e5c0f2a52f22a5d87feb6e6.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/e870b7865b573e047d7d3e27c433f82e1558ae876e5c0f2a52f22a5d87feb6e6.jpg)

![f5427e338d4c0d172d0aad54f64319226ce4db0191a98ec51ac2543741609eef.jpg](../icml_results/2485_SS-TPT_%20Stability%20and%20Suitability-Guided%20Test-Time%20Prompt%20Tuning%20for%20Adversarially%20Robust%20Vision-Lan/tables/f5427e338d4c0d172d0aad54f64319226ce4db0191a98ec51ac2543741609eef.jpg)

<a id="paper-2456"></a>

## Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression


### Tables

![0fc5f46220db71b971a0c5e4248707df746b207d2c6759fbd153bcb8505f62a8.jpg](../icml_results/2486_Estimating%20Continuous%20Treatment%20Effects%20with%20Two-Stage%20Kernel%20Ridge%20Regression/tables/0fc5f46220db71b971a0c5e4248707df746b207d2c6759fbd153bcb8505f62a8.jpg)

![58108ce8a247ef2b9de1d6fce788974af163d759fa126901c9e0b063c4d52522.jpg](../icml_results/2486_Estimating%20Continuous%20Treatment%20Effects%20with%20Two-Stage%20Kernel%20Ridge%20Regression/tables/58108ce8a247ef2b9de1d6fce788974af163d759fa126901c9e0b063c4d52522.jpg)

![6c73442c7a5b41fd9af90739c7433acd0f88342e6490dbccd23e8f5f8d7db287.jpg](../icml_results/2486_Estimating%20Continuous%20Treatment%20Effects%20with%20Two-Stage%20Kernel%20Ridge%20Regression/tables/6c73442c7a5b41fd9af90739c7433acd0f88342e6490dbccd23e8f5f8d7db287.jpg)

![9fea975e9a8c1c8b3b632c39681f119606d5a51657b8640f998584214a80ad8b.jpg](../icml_results/2486_Estimating%20Continuous%20Treatment%20Effects%20with%20Two-Stage%20Kernel%20Ridge%20Regression/tables/9fea975e9a8c1c8b3b632c39681f119606d5a51657b8640f998584214a80ad8b.jpg)

<a id="paper-2457"></a>

## ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation


### Images

![304b34f90770033d7c2774356d8b022952f954c5c85de42961034f96f618481d.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/images/304b34f90770033d7c2774356d8b022952f954c5c85de42961034f96f618481d.jpg)

![5ed1f861dfceff768526a34d4d4cf8db0298081df9aa1dc65363ffb60236421c.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/images/5ed1f861dfceff768526a34d4d4cf8db0298081df9aa1dc65363ffb60236421c.jpg)

![97f348ebd49fbc67072d481cf7572574dba13a0d3da3bddf7f2ac51f69c5d35d.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/images/97f348ebd49fbc67072d481cf7572574dba13a0d3da3bddf7f2ac51f69c5d35d.jpg)

![be4418d31cfc2e4e47625f56bfe2d19b055fd72d5a7b3cd2f01661f0f08577d6.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/images/be4418d31cfc2e4e47625f56bfe2d19b055fd72d5a7b3cd2f01661f0f08577d6.jpg)

![e69cfd6f52d6ff58a847685635f602c3bbc3948deb706075ed283f392fd115f9.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/images/e69cfd6f52d6ff58a847685635f602c3bbc3948deb706075ed283f392fd115f9.jpg)

### Tables

![48bd53ddf492bb263017c1fe78cb4880eb619aec793c54c9db63c79bb1ef8e72.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/48bd53ddf492bb263017c1fe78cb4880eb619aec793c54c9db63c79bb1ef8e72.jpg)

![4afb7560194ac548ed18b6853aec4829f2aea49d2496ed05001c0a99771b876d.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/4afb7560194ac548ed18b6853aec4829f2aea49d2496ed05001c0a99771b876d.jpg)

![81f03e43bb8f07ebe9190a8a9d335dbdf7e3bbde63b46cb872da2794320eb915.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/81f03e43bb8f07ebe9190a8a9d335dbdf7e3bbde63b46cb872da2794320eb915.jpg)

![acf0a64e63f8e4497f93f355cbc263d5a188b98892c678523a3b4ce569616a3f.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/acf0a64e63f8e4497f93f355cbc263d5a188b98892c678523a3b4ce569616a3f.jpg)

![b6a635ad1bfc1011c2645e5aedf26f0bc66f3ecfc91d0e0671687deb4a4c7819.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/b6a635ad1bfc1011c2645e5aedf26f0bc66f3ecfc91d0e0671687deb4a4c7819.jpg)

![cd888aa39791af804bb4363f7b4ca4e86c99d6bac14accdbd68c71ef0e92fd8a.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/cd888aa39791af804bb4363f7b4ca4e86c99d6bac14accdbd68c71ef0e92fd8a.jpg)

![dbc77be28cd845ef4f1f91368e72667e523c6813299fdfda7258c0feaefd2dfd.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/dbc77be28cd845ef4f1f91368e72667e523c6813299fdfda7258c0feaefd2dfd.jpg)

![e21f1760e16082b1034b07bb0db1e436e02aed6a7ae06bd212d1df43d592df66.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/e21f1760e16082b1034b07bb0db1e436e02aed6a7ae06bd212d1df43d592df66.jpg)

![e7c1641c6e1c187fd4111af5b2081d08553edb80e6193efebc98edefd558a1f3.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/e7c1641c6e1c187fd4111af5b2081d08553edb80e6193efebc98edefd558a1f3.jpg)

![f6c2ba6b7e9e6558bbf8abf194d9af73af8f5a4f06c8cfe5c5fbba871d6a2de1.jpg](../icml_results/2487_ReSpinQuant_%20Efficient%20Layer-Wise%20LLM%20Quantization%20via%20Subspace%20Residual%20Rotation%20Approximation/tables/f6c2ba6b7e9e6558bbf8abf194d9af73af8f5a4f06c8cfe5c5fbba871d6a2de1.jpg)

<a id="paper-2458"></a>

## TRACE: Toulmin-based Reasoning Assessment through Constructive Elements for LLM CoT Evaluation


### Images

![2b0a477acfc1c7c393a4a5bfeaac33eee5f1bad26c251e91141d575cae78005f.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/2b0a477acfc1c7c393a4a5bfeaac33eee5f1bad26c251e91141d575cae78005f.jpg)

![37f237b18378ef92aaf225f2b7b7b25d7183f9377b888d08c8e9a884754893a1.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/37f237b18378ef92aaf225f2b7b7b25d7183f9377b888d08c8e9a884754893a1.jpg)

![803983f571e08e342725dffedc6b9a691bfd8fc87d6feee48bc55e9f0cfd8a5a.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/803983f571e08e342725dffedc6b9a691bfd8fc87d6feee48bc55e9f0cfd8a5a.jpg)

![bcd76d5f50e5131688fb9309e4bdbba97ee964b76bbd03f5271b8f01eb850544.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/bcd76d5f50e5131688fb9309e4bdbba97ee964b76bbd03f5271b8f01eb850544.jpg)

![d5758129322138a1a95c5bb5aea8252b30e91e2d83e585891ae9132165dc657e.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/d5758129322138a1a95c5bb5aea8252b30e91e2d83e585891ae9132165dc657e.jpg)

![e4d05cbadece763a78d1c8c371437377a9c911045d9720006611fe98919836eb.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/e4d05cbadece763a78d1c8c371437377a9c911045d9720006611fe98919836eb.jpg)

![fffbc93e0a66d63a29f6b527784b6bee52d50b432933f1b81b1c1af804150c10.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/images/fffbc93e0a66d63a29f6b527784b6bee52d50b432933f1b81b1c1af804150c10.jpg)

### Tables

![09dbb5c62f18b40d9af21b2ef1b359a28cdcd49e14df9f9042303a7e3a76cce6.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/09dbb5c62f18b40d9af21b2ef1b359a28cdcd49e14df9f9042303a7e3a76cce6.jpg)

![11cf2f208bccd12fef23ddf1ae347cd0d392f6f63d370603321422daf5b999f0.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/11cf2f208bccd12fef23ddf1ae347cd0d392f6f63d370603321422daf5b999f0.jpg)

![2e4a712438fa9d4da13788f5cc566a5e27ea859c32d5ed2be9c48ca683673afa.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/2e4a712438fa9d4da13788f5cc566a5e27ea859c32d5ed2be9c48ca683673afa.jpg)

![34be796e6c7ddf00b13fdef65a78f31d6d8f0bf830aa3b5dd82a4b854f0b9708.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/34be796e6c7ddf00b13fdef65a78f31d6d8f0bf830aa3b5dd82a4b854f0b9708.jpg)

![537bc70427055f0fd22f3e5ada52e95f100964dd163132a9c91e3f148e696fb1.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/537bc70427055f0fd22f3e5ada52e95f100964dd163132a9c91e3f148e696fb1.jpg)

![67680429011589d4372f681e75d27bcacb9f8c07c479d317d1107882619610be.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/67680429011589d4372f681e75d27bcacb9f8c07c479d317d1107882619610be.jpg)

![78f40532571a64333026329b5341d5e5a840dea04edabdf33b4b602de65c6e93.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/78f40532571a64333026329b5341d5e5a840dea04edabdf33b4b602de65c6e93.jpg)

![978964b485ffb6f28b0cfe4b5ca093e99771f9534a4a8732484e6de45e7eede8.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/978964b485ffb6f28b0cfe4b5ca093e99771f9534a4a8732484e6de45e7eede8.jpg)

![a963f23c911ad22bb2b924920f2a3f96f02823cb5629f276246e5e9e625575a4.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/a963f23c911ad22bb2b924920f2a3f96f02823cb5629f276246e5e9e625575a4.jpg)

![addec5b22fd7f76dbe960e19931dcce7eb727abc7ec6195974e27c8649333c97.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/addec5b22fd7f76dbe960e19931dcce7eb727abc7ec6195974e27c8649333c97.jpg)

![ca7a80261e81b201819c1da4aae94e478031d917525ffed930fc4f9fe3abea33.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/ca7a80261e81b201819c1da4aae94e478031d917525ffed930fc4f9fe3abea33.jpg)

![cae2b87f5b6ecb03465f6c4ca5e3b2890af93724a564a0453d2db269b3f95d91.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/cae2b87f5b6ecb03465f6c4ca5e3b2890af93724a564a0453d2db269b3f95d91.jpg)

![d399d254391bf224cc35f68669deb3f2855c4234e8f33132f06e1b3fa2c1c7a6.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/d399d254391bf224cc35f68669deb3f2855c4234e8f33132f06e1b3fa2c1c7a6.jpg)

![da7bdf630d4695112e7475285c46b98372f2e6e67511fad231b1a4870504cfb0.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/da7bdf630d4695112e7475285c46b98372f2e6e67511fad231b1a4870504cfb0.jpg)

![f8dd5ae539052120ed8eccd3fc6c476f1f80ef38aefe86edef85293b28825693.jpg](../icml_results/2488_TRACE_%20Toulmin-based%20Reasoning%20Assessment%20through%20Constructive%20Elements%20for%20LLM%20CoT%20Evaluation/tables/f8dd5ae539052120ed8eccd3fc6c476f1f80ef38aefe86edef85293b28825693.jpg)

<a id="paper-2459"></a>

## ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin


### Images

![12bced46eee53edd5250fc807de1654651a29b986fe16fd5a11d365771f4207f.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/12bced46eee53edd5250fc807de1654651a29b986fe16fd5a11d365771f4207f.jpg)

![1928ae44c1a4d83eda940fe5728003bc64f0f63d7938c2f78cdeac0a224ecb86.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/1928ae44c1a4d83eda940fe5728003bc64f0f63d7938c2f78cdeac0a224ecb86.jpg)

![2a1a496d031d35974536bbae6b1181b3e27b14fc5c2a11aa8040d65119ef9f8d.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/2a1a496d031d35974536bbae6b1181b3e27b14fc5c2a11aa8040d65119ef9f8d.jpg)

![573724dcfd729767e7eaa8398baacac27c1b9708d1e47435e176819e14768902.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/573724dcfd729767e7eaa8398baacac27c1b9708d1e47435e176819e14768902.jpg)

![807d415a8c2eba2a0f14e806cd5f1cc17ce83f8d53433f978a15f832a00ec05d.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/807d415a8c2eba2a0f14e806cd5f1cc17ce83f8d53433f978a15f832a00ec05d.jpg)

![a86d90984e58d523da7350912a3233f2abef5535d26dcfa3cce8dadb6b932b7c.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/a86d90984e58d523da7350912a3233f2abef5535d26dcfa3cce8dadb6b932b7c.jpg)

![bef085954bbe9243c05c4bf79fe8eaf560862c9e0fd41940833af63df455eab5.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/bef085954bbe9243c05c4bf79fe8eaf560862c9e0fd41940833af63df455eab5.jpg)

![e3247196733b537e48d3e5049826780ab79111c2c9807e1b99881c8a6c20da90.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/images/e3247196733b537e48d3e5049826780ab79111c2c9807e1b99881c8a6c20da90.jpg)

### Tables

![141b91e07dbcedbe382a7b4ecd86004b68db74d4223d4893a8e5e7f24c59b5fc.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/141b91e07dbcedbe382a7b4ecd86004b68db74d4223d4893a8e5e7f24c59b5fc.jpg)

![2bccf823edebd25e70387cf12ae48a006a661bddbb3a1593c1d5a91ab70810ea.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/2bccf823edebd25e70387cf12ae48a006a661bddbb3a1593c1d5a91ab70810ea.jpg)

![3cda83b97b4ddc498f7f4b00781148f14c07223de3c25dd1f1ca73734f3943fc.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/3cda83b97b4ddc498f7f4b00781148f14c07223de3c25dd1f1ca73734f3943fc.jpg)

![481bc1c89bd257b0403927097ed1a6821bffa733105c4451f8ec7f4eac2607de.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/481bc1c89bd257b0403927097ed1a6821bffa733105c4451f8ec7f4eac2607de.jpg)

![638d980a52bd9349d2147446bf41e461907875ba0671a1196ab0d96003a94efd.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/638d980a52bd9349d2147446bf41e461907875ba0671a1196ab0d96003a94efd.jpg)

![71fdee94fe118996506e756c5a7956ae916d09c50438532b409f41fcb104a57a.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/71fdee94fe118996506e756c5a7956ae916d09c50438532b409f41fcb104a57a.jpg)

![81775e14bc6fe5af85e6b370ce2ce6b1054a66b675b77663b89475112e982de6.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/81775e14bc6fe5af85e6b370ce2ce6b1054a66b675b77663b89475112e982de6.jpg)

![a4ce8a41e10a02f0dd377983c676d3664df6c52c2fcb2c03033a701545330a28.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/a4ce8a41e10a02f0dd377983c676d3664df6c52c2fcb2c03033a701545330a28.jpg)

![b0331c2afc56dbb43dd382a7c25eb52fb79ba10365fa6419b8f4872bea678d2d.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/b0331c2afc56dbb43dd382a7c25eb52fb79ba10365fa6419b8f4872bea678d2d.jpg)

![caa6e506df5723d38b0e7b0f47e55bad0ba587a571100b4bb7f2f1e0f1f53025.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/caa6e506df5723d38b0e7b0f47e55bad0ba587a571100b4bb7f2f1e0f1f53025.jpg)

![efb969f2091a49e4de31c9541fcd0e9e35ed7583ecdeacd6e22407b9f91cc90b.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/efb969f2091a49e4de31c9541fcd0e9e35ed7583ecdeacd6e22407b9f91cc90b.jpg)

![f82a799eeb2916e7f9c58110786f4134b8d26abf07f0bc4a0ffb5e0f240567e4.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/f82a799eeb2916e7f9c58110786f4134b8d26abf07f0bc4a0ffb5e0f240567e4.jpg)

![fe1ce9b09252375bd7218e790edba4ab5db1559ff410278181e8ddbba5d7010e.jpg](../icml_results/2489_ArcVQ-VAE_%20A%20Spherical%20Vector%20Quantization%20Framework%20with%20ArcCosine%20Additive%20Margin/tables/fe1ce9b09252375bd7218e790edba4ab5db1559ff410278181e8ddbba5d7010e.jpg)

<a id="paper-2460"></a>

## Memory as Dynamics: Learning Reliability-Guided Predictive Models for Online Video Perception


### Images

![06f1a58053550b169682563c0fb1637ee52eee21a7d5eee243bbba8396a26cc4.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/06f1a58053550b169682563c0fb1637ee52eee21a7d5eee243bbba8396a26cc4.jpg)

![145c13dbb36d3e76d3783ab7156e52f6b1a05a393a1271f935dd7db3d1c3d847.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/145c13dbb36d3e76d3783ab7156e52f6b1a05a393a1271f935dd7db3d1c3d847.jpg)

![41d3225720264f64e9f9a470d2ce5f97b6e38dee1fa05e7614379981a99801bb.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/41d3225720264f64e9f9a470d2ce5f97b6e38dee1fa05e7614379981a99801bb.jpg)

![66b067b00d09c11011510efd760f74ebdd6053e0635cb57c4ae81d49ee69e4c1.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/66b067b00d09c11011510efd760f74ebdd6053e0635cb57c4ae81d49ee69e4c1.jpg)

![67198118de5d18d51f027eec397390aa0ad1e28d6e84f2833b2c2f711746e5aa.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/67198118de5d18d51f027eec397390aa0ad1e28d6e84f2833b2c2f711746e5aa.jpg)

![7d67643c5ff4bb96b4725b564159f3c0638f138b714f98965e51aa8d492f5d06.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/7d67643c5ff4bb96b4725b564159f3c0638f138b714f98965e51aa8d492f5d06.jpg)

![d7a6ccf30e247b0b0aee379e6b7a5798ee148406fcbbaf6ebb3698ae37392bdd.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/images/d7a6ccf30e247b0b0aee379e6b7a5798ee148406fcbbaf6ebb3698ae37392bdd.jpg)

### Tables

![164161badf8716088140708a8d65fff1a398cc108f33e4b78dd882ea49aaaa8c.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/164161badf8716088140708a8d65fff1a398cc108f33e4b78dd882ea49aaaa8c.jpg)

![263176dc9cd0bdd9db8756280f7a6d6c8f582d3b9a3bddf8153ff9a7437ffef3.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/263176dc9cd0bdd9db8756280f7a6d6c8f582d3b9a3bddf8153ff9a7437ffef3.jpg)

![2f1dd911b1789ab49bb35c5647645dd8143290fc6c398899c7753af7d6f7ed40.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/2f1dd911b1789ab49bb35c5647645dd8143290fc6c398899c7753af7d6f7ed40.jpg)

![3bd5e4b760166844d27763f9f6e292619866f1a8327554b23c37cd763ffacaa9.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/3bd5e4b760166844d27763f9f6e292619866f1a8327554b23c37cd763ffacaa9.jpg)

![6eba3043a5dcb258b69680ddd6e64250102cf9157c92540258db50799c1e1eaa.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/6eba3043a5dcb258b69680ddd6e64250102cf9157c92540258db50799c1e1eaa.jpg)

![81923315f4d63f879d6ffbb354eba924fe932fd834e92b73d21be25af09e9c49.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/81923315f4d63f879d6ffbb354eba924fe932fd834e92b73d21be25af09e9c49.jpg)

![840fd250bb1976fa573a38e87fdcdd0426f700029e562df24e04018818c647fc.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/840fd250bb1976fa573a38e87fdcdd0426f700029e562df24e04018818c647fc.jpg)

![8bfe7516e2dadc01edf1d3fc6e144c0afc583bca9caab3b014b8175ce7fd3b96.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/8bfe7516e2dadc01edf1d3fc6e144c0afc583bca9caab3b014b8175ce7fd3b96.jpg)

![90755215d23209d7021baeb1dcc84efceadb0452f812dd2537edec589fb2720b.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/90755215d23209d7021baeb1dcc84efceadb0452f812dd2537edec589fb2720b.jpg)

![9573abf4a1b22b9d8c36d3d7fd4f224847bee06ea5c257064a068059ed3f1c61.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/9573abf4a1b22b9d8c36d3d7fd4f224847bee06ea5c257064a068059ed3f1c61.jpg)

![a2c6b492c723b5ea0f6912a38af5a34d7557a6405240c678624b85bb8cb1152d.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/a2c6b492c723b5ea0f6912a38af5a34d7557a6405240c678624b85bb8cb1152d.jpg)

![c02cca2e344cfca04cd84111b85ee32b8cc80f2028deb081825e54ff47f2e6f8.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/c02cca2e344cfca04cd84111b85ee32b8cc80f2028deb081825e54ff47f2e6f8.jpg)

![eb5b7ff6f47daac973e1d4bfc96b1406749fee6e7c53b45bdbcc888e017094ee.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/eb5b7ff6f47daac973e1d4bfc96b1406749fee6e7c53b45bdbcc888e017094ee.jpg)

![ec6079f5ac5cb42ce43e46d9821638ec430f7398c174de764a9b2ba4f1b7eec8.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/ec6079f5ac5cb42ce43e46d9821638ec430f7398c174de764a9b2ba4f1b7eec8.jpg)

![ee68992d939bdf94f1554672a43e04661e1a6380581dd2b6d88924574944d97f.jpg](../icml_results/2490_Memory%20as%20Dynamics_%20Learning%20Reliability-Guided%20Predictive%20Models%20for%20Online%20Video%20Perception/tables/ee68992d939bdf94f1554672a43e04661e1a6380581dd2b6d88924574944d97f.jpg)

<a id="paper-2461"></a>

## Coverage Improvement and Fast Convergence of On-policy Preference Learning


### Images

![5d1e8a0ef1dca4954390cbf8bf8b8ab88433968dc2cc0a289f368f2b6a7fa4ae.jpg](../icml_results/2491_Coverage%20Improvement%20and%20Fast%20Convergence%20of%20On-policy%20Preference%20Learning/images/5d1e8a0ef1dca4954390cbf8bf8b8ab88433968dc2cc0a289f368f2b6a7fa4ae.jpg)

### Tables

![cae94550ff233cec380d7daaeb6763ce890b122ac0f4255a4b2b61b135813db3.jpg](../icml_results/2491_Coverage%20Improvement%20and%20Fast%20Convergence%20of%20On-policy%20Preference%20Learning/tables/cae94550ff233cec380d7daaeb6763ce890b122ac0f4255a4b2b61b135813db3.jpg)

![ce8f79bdef7c8a091d0909346e61e261955c83d2140cc0d19cd3da693a4ddad3.jpg](../icml_results/2491_Coverage%20Improvement%20and%20Fast%20Convergence%20of%20On-policy%20Preference%20Learning/tables/ce8f79bdef7c8a091d0909346e61e261955c83d2140cc0d19cd3da693a4ddad3.jpg)

<a id="paper-2462"></a>

## Equivariant Neural Networks for General Linear Symmetries on Lie Algebras


### Images

![08682194c571ee53a63674086692acf20c6b8cd4e4b8aebc8cb4dcdacf17c9a8.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/08682194c571ee53a63674086692acf20c6b8cd4e4b8aebc8cb4dcdacf17c9a8.jpg)

![1382fb2fd63a78481516711e793c5476fc1b0ba2da9498b46cf98bc8f3e4e2a2.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/1382fb2fd63a78481516711e793c5476fc1b0ba2da9498b46cf98bc8f3e4e2a2.jpg)

![5965755b14cb44f8cdb3aac839e95195c02db304dafcdc702521e07e8ad2e417.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/5965755b14cb44f8cdb3aac839e95195c02db304dafcdc702521e07e8ad2e417.jpg)

![7e756e191123c076984e184585f357accc08888b2175aa22c7a10cd31655852b.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/7e756e191123c076984e184585f357accc08888b2175aa22c7a10cd31655852b.jpg)

![a56e7bb50ef9c7dc5a4a6f4f74ca550e4e3a9185af5df1fa2e6f38afce41ac7b.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/a56e7bb50ef9c7dc5a4a6f4f74ca550e4e3a9185af5df1fa2e6f38afce41ac7b.jpg)

![ba44fdd79b316852ff8a0298eb2efc5f25b9945a68e863a00addb713f3ecd3f1.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/ba44fdd79b316852ff8a0298eb2efc5f25b9945a68e863a00addb713f3ecd3f1.jpg)

![e7dec8399c559969645828feec0bbf1fbd0772671097d680175d90fdf5859511.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/e7dec8399c559969645828feec0bbf1fbd0772671097d680175d90fdf5859511.jpg)

![e8398dbd49c4c83ce3f41a866423fcfef17287a36f3a83661f7d33d6a51f3e9c.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/e8398dbd49c4c83ce3f41a866423fcfef17287a36f3a83661f7d33d6a51f3e9c.jpg)

![f0d83b34906b6521842ec8b41d80bbb70a7573d9d652093c7caa3b20303b6c77.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/images/f0d83b34906b6521842ec8b41d80bbb70a7573d9d652093c7caa3b20303b6c77.jpg)

### Tables

![0c20f1d9adcac8254c4c4551369b29bbeb3054f499553a665adc241c06b043ae.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/0c20f1d9adcac8254c4c4551369b29bbeb3054f499553a665adc241c06b043ae.jpg)

![1412250b610d4cd3f921df981a62012ff4208be85a969040fe3f051728242208.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/1412250b610d4cd3f921df981a62012ff4208be85a969040fe3f051728242208.jpg)

![15b8caff784dcb50860aea0d034609480df5b87900279d47be246e5ac9597eb3.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/15b8caff784dcb50860aea0d034609480df5b87900279d47be246e5ac9597eb3.jpg)

![461e3f413ab359b364b8738512ade06c270b2fd83c3cbf56025d80c2fbf19d7e.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/461e3f413ab359b364b8738512ade06c270b2fd83c3cbf56025d80c2fbf19d7e.jpg)

![58be604a3282fd19ef3e067c85664046827f78897647d81a5ab36758721c86e0.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/58be604a3282fd19ef3e067c85664046827f78897647d81a5ab36758721c86e0.jpg)

![625ab198971f67d2a880bbd8cb554da158a4d59db92713d63be3ed1d4824f781.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/625ab198971f67d2a880bbd8cb554da158a4d59db92713d63be3ed1d4824f781.jpg)

![6dfd41089a0b470e4382f78e92d5e0390c54c9d063d9327b6fe57ef5af66b2d1.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/6dfd41089a0b470e4382f78e92d5e0390c54c9d063d9327b6fe57ef5af66b2d1.jpg)

![6e08f0390bb3450b765189ba600914929e933bba5a6ee42088c65e54161b7562.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/6e08f0390bb3450b765189ba600914929e933bba5a6ee42088c65e54161b7562.jpg)

![75e3d976f6ec62572e0053d7219e5efecedcd3fd3f082b28b4568131095dd31c.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/75e3d976f6ec62572e0053d7219e5efecedcd3fd3f082b28b4568131095dd31c.jpg)

![90b5c95e036e768d5e5fe0b058b635a804ecec814eb49cc9f3e3ac989ccb5fd3.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/90b5c95e036e768d5e5fe0b058b635a804ecec814eb49cc9f3e3ac989ccb5fd3.jpg)

![a87e95b77568f01d77da9f7fb502f48acb99328805848ba1d5fce6df6d42fe54.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/a87e95b77568f01d77da9f7fb502f48acb99328805848ba1d5fce6df6d42fe54.jpg)

![ba64ce2e8f36d1ba6e1d94f722d3b89152ab65d4f66f34c26aacfb8fa9781765.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/ba64ce2e8f36d1ba6e1d94f722d3b89152ab65d4f66f34c26aacfb8fa9781765.jpg)

![d60ea5312ba44d9a356042d9c9e6a1826ba39829f4650fd9086df057faa1805f.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/d60ea5312ba44d9a356042d9c9e6a1826ba39829f4650fd9086df057faa1805f.jpg)

![db1d1f82f73e0ccb4e42eefe4c59667580991cdcda5afe5de61ae1f198f52fbd.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/db1d1f82f73e0ccb4e42eefe4c59667580991cdcda5afe5de61ae1f198f52fbd.jpg)

![dbe659df2cac19f87e4244593ff3d92ffbb7ac2c25b8328f8e2d1bb3ff27419c.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/dbe659df2cac19f87e4244593ff3d92ffbb7ac2c25b8328f8e2d1bb3ff27419c.jpg)

![ef8ba5f4089d70eadea3f9b4c813fa2fdc58713f4cf85e598d363f4ca33a6eaf.jpg](../icml_results/2492_Equivariant%20Neural%20Networks%20for%20General%20Linear%20Symmetries%20on%20Lie%20Algebras/tables/ef8ba5f4089d70eadea3f9b4c813fa2fdc58713f4cf85e598d363f4ca33a6eaf.jpg)

<a id="paper-2463"></a>

## Dywave: Event-Aligned Dynamic Tokenization for Heterogeneous IoT Sensing Signals


### Images

![0655a57fe3542754af227dcb4ee55a01696701f04c496bb6738123aeb9f7c36b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/0655a57fe3542754af227dcb4ee55a01696701f04c496bb6738123aeb9f7c36b.jpg)

![1997e2615dfaa240b646c590ca715c6fc11f85f342018f67401616fd6772ed31.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/1997e2615dfaa240b646c590ca715c6fc11f85f342018f67401616fd6772ed31.jpg)

![19b8ef0834244cd85d7ae5801ce234be0ad82a68b44eddfc2b2ce1fd4f0d0dc0.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/19b8ef0834244cd85d7ae5801ce234be0ad82a68b44eddfc2b2ce1fd4f0d0dc0.jpg)

![1b1eccf8aaee02fd0e8665ea4a3cac2776b64bf8cd679a17ee66f37a41ca9aa2.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/1b1eccf8aaee02fd0e8665ea4a3cac2776b64bf8cd679a17ee66f37a41ca9aa2.jpg)

![1b6a4638a63bdac2f24cfbaee1bddaaebeaa93c3eddea771641a94852dd8d929.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/1b6a4638a63bdac2f24cfbaee1bddaaebeaa93c3eddea771641a94852dd8d929.jpg)

![28f89f28d9ec98afa919b5ec202dd8030f7f0d1b1b78da625b33fdd3e88d873c.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/28f89f28d9ec98afa919b5ec202dd8030f7f0d1b1b78da625b33fdd3e88d873c.jpg)

![33664dbc786a36f13c75d02f2c6276971171dcf30fc6ea32eae47f53f9f9f46b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/33664dbc786a36f13c75d02f2c6276971171dcf30fc6ea32eae47f53f9f9f46b.jpg)

![367a94dc6129433c91a139ca09b5b43eabebbb505681c9f8dd47ab29508d53ca.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/367a94dc6129433c91a139ca09b5b43eabebbb505681c9f8dd47ab29508d53ca.jpg)

![369c88ed99000e1651b6bc9832e3e77a4be471c8da5db1bf2011b693c325dfe5.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/369c88ed99000e1651b6bc9832e3e77a4be471c8da5db1bf2011b693c325dfe5.jpg)

![43580c545c4ddabe8335a3337e89f14b82786beb7567f9ea65d2c68393a3b27b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/43580c545c4ddabe8335a3337e89f14b82786beb7567f9ea65d2c68393a3b27b.jpg)

![7f7d7f548ba11290974049f0d3456f204793c206e22711dfc72d5f3c7ccdc4b4.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/7f7d7f548ba11290974049f0d3456f204793c206e22711dfc72d5f3c7ccdc4b4.jpg)

![8748deb0a843323a04fc93310e7a6ce91cf85429acbb4d5f4fc94e5141ced7ee.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/8748deb0a843323a04fc93310e7a6ce91cf85429acbb4d5f4fc94e5141ced7ee.jpg)

![8db2fecfeadeff71cdfb884490446e31c18ef2332d25e0133a2c5bed55ebd712.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/8db2fecfeadeff71cdfb884490446e31c18ef2332d25e0133a2c5bed55ebd712.jpg)

![8e6fe8d01cb53f58b78358e0669aa1f012b238eb4d5edb6bb6919bea32a21c04.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/8e6fe8d01cb53f58b78358e0669aa1f012b238eb4d5edb6bb6919bea32a21c04.jpg)

![af5b5d48618d03fb96192c31629a8275569ed52190daf1fa6e6f8874a8a6e828.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/af5b5d48618d03fb96192c31629a8275569ed52190daf1fa6e6f8874a8a6e828.jpg)

![b1dcf2a0c182ceb7bd53f2be781c79624970cebd7b5fd6b50b343d1e3a64094b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/b1dcf2a0c182ceb7bd53f2be781c79624970cebd7b5fd6b50b343d1e3a64094b.jpg)

![c87f2483a3cc3ba5db3033f9fb3b20acb8ca2916ad6fd2ae774b7ec3c1ee612b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/c87f2483a3cc3ba5db3033f9fb3b20acb8ca2916ad6fd2ae774b7ec3c1ee612b.jpg)

![d1f574aeb02010e0935428f5d4c5e7e430d57c158b8a1b4e51b49deb058722e0.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/d1f574aeb02010e0935428f5d4c5e7e430d57c158b8a1b4e51b49deb058722e0.jpg)

![dfdf7631f03b4bb6ba985d9841138b3abf71fae3623df4353b98f591d65fc666.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/images/dfdf7631f03b4bb6ba985d9841138b3abf71fae3623df4353b98f591d65fc666.jpg)

### Tables

![102bc46e183b8e7fdf84d64a5a3b71095d90089d46896d4426514411278b5b3d.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/102bc46e183b8e7fdf84d64a5a3b71095d90089d46896d4426514411278b5b3d.jpg)

![5bdef8472dbc39bee27388189bd00247971ed955ff8996d9a1043bf9d843f053.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/5bdef8472dbc39bee27388189bd00247971ed955ff8996d9a1043bf9d843f053.jpg)

![96512e34fbe70aa1ca8fb6e115baa7527884076c47c1eb837bf707b1d821a109.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/96512e34fbe70aa1ca8fb6e115baa7527884076c47c1eb837bf707b1d821a109.jpg)

![b804ff5df00227b7faa1592afd3f37d662e58e9c5af83ba327943313e3c06a59.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/b804ff5df00227b7faa1592afd3f37d662e58e9c5af83ba327943313e3c06a59.jpg)

![cb39cbb78c70b202464181c4b1ebb2d33df4099e7cd59e241b66133538464b4b.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/cb39cbb78c70b202464181c4b1ebb2d33df4099e7cd59e241b66133538464b4b.jpg)

![edcad3371d2a92dc1f49e991562dccb8b0420cceae9f11492b23591d878fe143.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/edcad3371d2a92dc1f49e991562dccb8b0420cceae9f11492b23591d878fe143.jpg)

![fffbee551886caa44bc64ec69c1ef139780edc61e7d82436e7d3f736f8ec29de.jpg](../icml_results/2493_Dywave_%20Event-Aligned%20Dynamic%20Tokenization%20for%20Heterogeneous%20IoT%20Sensing%20Signals/tables/fffbee551886caa44bc64ec69c1ef139780edc61e7d82436e7d3f736f8ec29de.jpg)

<a id="paper-2464"></a>

## Representational Curvature Modulates Behavioral Uncertainty in Large Language Models


### Images

![05248f7f8c0578ae7f4723d1169daa40e78999a4d44565790ed5103668bbd254.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/05248f7f8c0578ae7f4723d1169daa40e78999a4d44565790ed5103668bbd254.jpg)

![214caf7b234aa39e6aaa448db4a4c7ad1b07acc6a12e4b399e6d9f23ebdd7f11.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/214caf7b234aa39e6aaa448db4a4c7ad1b07acc6a12e4b399e6d9f23ebdd7f11.jpg)

![447e883e5ab3d2ce1c72e3c19774e4d9a24300d223b713ef4ca72a6779b2b99d.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/447e883e5ab3d2ce1c72e3c19774e4d9a24300d223b713ef4ca72a6779b2b99d.jpg)

![4510a022de93281773fc25d5ff8bb2146c975aa5d1d65e06d04ff05e4332fbf7.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/4510a022de93281773fc25d5ff8bb2146c975aa5d1d65e06d04ff05e4332fbf7.jpg)

![6d16d4d54d7bf449753181ec7cd6f887b98bc09166fb7fb4e1d70a03ba417aa4.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/6d16d4d54d7bf449753181ec7cd6f887b98bc09166fb7fb4e1d70a03ba417aa4.jpg)

![864630ff507b9dc26893de91cb56b9e8426cad47e7f6065894c4bc365818b426.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/864630ff507b9dc26893de91cb56b9e8426cad47e7f6065894c4bc365818b426.jpg)

![9b534f1e278006f8c7d1c361a7ce9dba67be19d30ebb298675e08ea0ca1fe2b1.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/9b534f1e278006f8c7d1c361a7ce9dba67be19d30ebb298675e08ea0ca1fe2b1.jpg)

![9ff2fcc748f399a9a6e6935915d30850f50e2bdad8776d0a17ae2dd307fe7062.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/9ff2fcc748f399a9a6e6935915d30850f50e2bdad8776d0a17ae2dd307fe7062.jpg)

![c768afb8555d8a7761f0ef87d4cc4163dbe66be9fc676ef978b571793289af5d.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/c768afb8555d8a7761f0ef87d4cc4163dbe66be9fc676ef978b571793289af5d.jpg)

![cf6a11f90404e0e639a00aeec835e02e078a5e13040f83d9b5fcced845e01570.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/cf6a11f90404e0e639a00aeec835e02e078a5e13040f83d9b5fcced845e01570.jpg)

![ef5f17f956c7100599e59219830eceb6fa241a8f4ca99a00d5aec604f1914482.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/images/ef5f17f956c7100599e59219830eceb6fa241a8f4ca99a00d5aec604f1914482.jpg)

### Tables

![ccb1204cba922f287722c2bba88e905a45cda24a08c7a67d1f9b70e9bd9708d7.jpg](../icml_results/2494_Representational%20Curvature%20Modulates%20Behavioral%20Uncertainty%20in%20Large%20Language%20Models/tables/ccb1204cba922f287722c2bba88e905a45cda24a08c7a67d1f9b70e9bd9708d7.jpg)

<a id="paper-2465"></a>

## Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks


### Images

![25c6ea074ca016f64094333e35fa2b8ba24fe3b1fed9aa8329992c45d99dbdc8.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/images/25c6ea074ca016f64094333e35fa2b8ba24fe3b1fed9aa8329992c45d99dbdc8.jpg)

![8a194e24528c6713b07312acd9659388950e27629643c6657f43ed6f3de650ca.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/images/8a194e24528c6713b07312acd9659388950e27629643c6657f43ed6f3de650ca.jpg)

![a140caac4be7b5c774b31b5faa5798d361f9ded18ad811fb67aaa96f9ac467f1.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/images/a140caac4be7b5c774b31b5faa5798d361f9ded18ad811fb67aaa96f9ac467f1.jpg)

### Tables

![4bd48631977764e533a904c0d9e55753cbb3c163c7a1173bfb4c11c2d629beac.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/4bd48631977764e533a904c0d9e55753cbb3c163c7a1173bfb4c11c2d629beac.jpg)

![4e27c44e39620aba1060f2a3355a9270936a14622e0ea410bbfd546221052e39.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/4e27c44e39620aba1060f2a3355a9270936a14622e0ea410bbfd546221052e39.jpg)

![610a4442d7809c2f669f35e7647c30340e4bffca78ed1cb7665dc008bc4118c5.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/610a4442d7809c2f669f35e7647c30340e4bffca78ed1cb7665dc008bc4118c5.jpg)

![66386c5425cb6b671e101dc034cf223e3d124eb9b07f9997ce78c1bb785d3b58.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/66386c5425cb6b671e101dc034cf223e3d124eb9b07f9997ce78c1bb785d3b58.jpg)

![740b34bfb60b0e982bbeaa816b41dde61402accd306a5e2e0add0de542b8a618.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/740b34bfb60b0e982bbeaa816b41dde61402accd306a5e2e0add0de542b8a618.jpg)

![99c6364e49d364f6150d6b6fdf870dd1d5df91647b1bfcdd21162da76341ea8b.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/99c6364e49d364f6150d6b6fdf870dd1d5df91647b1bfcdd21162da76341ea8b.jpg)

![c777d77b500c03d5d8bbf8c041c60dc21d340c1c0575a1be0dcec59a3ca44b83.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/c777d77b500c03d5d8bbf8c041c60dc21d340c1c0575a1be0dcec59a3ca44b83.jpg)

![cd07672ed5a575be0ab70063b9603d2da6bd5c6cefca1abbafb458fb7f651602.jpg](../icml_results/2495_Dataset%20Distillation%20Efficiently%20Encodes%20Low-Dimensional%20Representations%20from%20Gradient-Based%20Learnin/tables/cd07672ed5a575be0ab70063b9603d2da6bd5c6cefca1abbafb458fb7f651602.jpg)

<a id="paper-2466"></a>

## Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking


### Images

![290b2ee26d8a8b6061a44302e570b2b1cd50241b9bf05cb187e0b0a0e805d8ed.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/290b2ee26d8a8b6061a44302e570b2b1cd50241b9bf05cb187e0b0a0e805d8ed.jpg)

![306445edce551147152a3867c59e7de3abe97355e7afbd870cb4465e2a38a0d4.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/306445edce551147152a3867c59e7de3abe97355e7afbd870cb4465e2a38a0d4.jpg)

![9886e6eb8ae80c33fb0a7d415d7a5683b347355988217d5e6caaf69a250936b3.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/9886e6eb8ae80c33fb0a7d415d7a5683b347355988217d5e6caaf69a250936b3.jpg)

![af5768aeee9ca588f2074ce198ae8df5336702d3d2c7031af5e74c30f746a513.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/af5768aeee9ca588f2074ce198ae8df5336702d3d2c7031af5e74c30f746a513.jpg)

![b43607fb472979a6ddd324493f6c9cd135afba2d17630738526d8bab22a18c57.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/b43607fb472979a6ddd324493f6c9cd135afba2d17630738526d8bab22a18c57.jpg)

![ebb84df6aa47e72523a2bbf4922e79a9fe3720e4c2121e822e4ebd43a2ca366d.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/ebb84df6aa47e72523a2bbf4922e79a9fe3720e4c2121e822e4ebd43a2ca366d.jpg)

![f3967dc0cfb4aab9c87d8e8894941af72f3f5ab224071a534945205af61090f8.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/images/f3967dc0cfb4aab9c87d8e8894941af72f3f5ab224071a534945205af61090f8.jpg)

### Tables

![88a0f4b26c63476f018c5ff49ef9e879023f4ba8c2f63312667ae013e874e00d.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/88a0f4b26c63476f018c5ff49ef9e879023f4ba8c2f63312667ae013e874e00d.jpg)

![bf53fdd84c334741af7ac87a610d281dc74fc69953ecdf2f562c4a4a876c47e0.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/bf53fdd84c334741af7ac87a610d281dc74fc69953ecdf2f562c4a4a876c47e0.jpg)

![d4eff3deedebb0e3e2bae263c8cc28884431be3e655d3e70d53124e6bb2831db.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/d4eff3deedebb0e3e2bae263c8cc28884431be3e655d3e70d53124e6bb2831db.jpg)

![e6813d1f7b50cfd7db07c3404ce490914876aa86f508f9cf6bb2b6e6e668698b.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/e6813d1f7b50cfd7db07c3404ce490914876aa86f508f9cf6bb2b6e6e668698b.jpg)

![f9f84dbd4e9de600df5db49f9f6014e649ba7cad4befad7031cf53eabd4e7a3d.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/f9f84dbd4e9de600df5db49f9f6014e649ba7cad4befad7031cf53eabd4e7a3d.jpg)

![fbc0f78761266ea48b57fa75894bf4862f099da47984e7ec4571ca3183ad75b5.jpg](../icml_results/2496_Credit-assigned%20Policy%20Gradient%20for%20Early%20Stage%20Retrieval%20in%20Two-stage%20Ranking/tables/fbc0f78761266ea48b57fa75894bf4862f099da47984e7ec4571ca3183ad75b5.jpg)

<a id="paper-2467"></a>

## Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation


### Images

![0f2caab712e00fa9442d0d1a6d4a58cf56f998687da2af50e562980bfe5ae688.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/0f2caab712e00fa9442d0d1a6d4a58cf56f998687da2af50e562980bfe5ae688.jpg)

![2d41613a6468037da852c63b8ca0650603ca70df322bfc811260bb111cefcd13.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/2d41613a6468037da852c63b8ca0650603ca70df322bfc811260bb111cefcd13.jpg)

![310f904d7d6558bf570d339a12274bc602b9190688b85ff1fec720728fd02efd.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/310f904d7d6558bf570d339a12274bc602b9190688b85ff1fec720728fd02efd.jpg)

![4bd01e0542b82034dd3f024ef661d3faedf407b01e749fa0ed2fc0f5f3ea21e8.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/4bd01e0542b82034dd3f024ef661d3faedf407b01e749fa0ed2fc0f5f3ea21e8.jpg)

![62ff0eb94efa83f6a824bc537164031b16675af81486559b7043325667276c1a.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/62ff0eb94efa83f6a824bc537164031b16675af81486559b7043325667276c1a.jpg)

![864ecf082c5595922bd61b55e035c9ed3cb276876a7138125f28e982feb15d4d.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/864ecf082c5595922bd61b55e035c9ed3cb276876a7138125f28e982feb15d4d.jpg)

![8e644500c527a67c15b298fee6d9c1d508096cb47478f0f8d3a523687b4ef5ef.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/8e644500c527a67c15b298fee6d9c1d508096cb47478f0f8d3a523687b4ef5ef.jpg)

![d123a4578b6b29fc4c3206caf4cc286043c67e34e0de131fe364da41d4818c06.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/d123a4578b6b29fc4c3206caf4cc286043c67e34e0de131fe364da41d4818c06.jpg)

![d2b2927ef785318973979776019befac4186715cb5c2ce9afa7c2944d6987100.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/d2b2927ef785318973979776019befac4186715cb5c2ce9afa7c2944d6987100.jpg)

![e8cdf65ff7c22418cc19a4c1110395ef18fab0140066a28fd6b2f16844b2ec8d.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/e8cdf65ff7c22418cc19a4c1110395ef18fab0140066a28fd6b2f16844b2ec8d.jpg)

![eea846afd8c64bdb0c20f45fa8e7bf7b76443b43964ddd63b5e2b1d2ddc19466.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/eea846afd8c64bdb0c20f45fa8e7bf7b76443b43964ddd63b5e2b1d2ddc19466.jpg)

![fc95beba22889891033bcde0b67d7c7d5f544ab0d9e2757cf9d3e66b2027cda1.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/images/fc95beba22889891033bcde0b67d7c7d5f544ab0d9e2757cf9d3e66b2027cda1.jpg)

### Tables

![148e319e90305f4491b14d5b9665eb173e8a84c69e238c09bd27dfd96ee6704a.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/148e319e90305f4491b14d5b9665eb173e8a84c69e238c09bd27dfd96ee6704a.jpg)

![41596e8d326d010b666971d76780dd5401e76f45907c65cbb9e6c03102c41a93.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/41596e8d326d010b666971d76780dd5401e76f45907c65cbb9e6c03102c41a93.jpg)

![7ce95338b7250af00309f76453c026b59a993513f12d04958b05fe75e3b61eb6.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/7ce95338b7250af00309f76453c026b59a993513f12d04958b05fe75e3b61eb6.jpg)

![832933bea9c4ed8c3ad1083691ca858eb642d3fd66e6110b36f3cd5bf503fa71.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/832933bea9c4ed8c3ad1083691ca858eb642d3fd66e6110b36f3cd5bf503fa71.jpg)

![8434f3495f4bfe31ca4ecc050662f2771b80f5e5574a92e6ede47316483afeaf.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/8434f3495f4bfe31ca4ecc050662f2771b80f5e5574a92e6ede47316483afeaf.jpg)

![851fe95e2857943f5fe25637ab7a4fc4084694841e1bfaf0efbd73ae23665115.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/851fe95e2857943f5fe25637ab7a4fc4084694841e1bfaf0efbd73ae23665115.jpg)

![bb92e2836c5684b74cbe2893aa288dabb79ae86cfbd9447c814c6c39b3f90c3d.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/bb92e2836c5684b74cbe2893aa288dabb79ae86cfbd9447c814c6c39b3f90c3d.jpg)

![c4f087ea7c8d7125cb1ef9c40f239993a2ccbb766a449fce96c55c968d4c1d46.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/c4f087ea7c8d7125cb1ef9c40f239993a2ccbb766a449fce96c55c968d4c1d46.jpg)

![dfb4041ef799d0b24754e1aa6b706a51ca7f987ca2fc2acbad35ec535854c8fe.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/dfb4041ef799d0b24754e1aa6b706a51ca7f987ca2fc2acbad35ec535854c8fe.jpg)

![f7016052560b5c8fc6cc3ed2eaf03cc330276dceef6607353cba3b66a43ed596.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/f7016052560b5c8fc6cc3ed2eaf03cc330276dceef6607353cba3b66a43ed596.jpg)

![fae4a930c547205293348907f7576df1cbd4d5692d6a744c7a37eddc9d98f558.jpg](../icml_results/2497_Coverage%2C%20Not%20Averages_%20Semantic%20Stratification%20for%20Trustworthy%20Retrieval%20Evaluation/tables/fae4a930c547205293348907f7576df1cbd4d5692d6a744c7a37eddc9d98f558.jpg)

<a id="paper-2468"></a>

## Tucker Attention: A generalization of approximate attention mechanisms


### Images

![028acf582128ff66de2d97c8381fc638131fbef3856d95aff8bdefe30ac637f9.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/028acf582128ff66de2d97c8381fc638131fbef3856d95aff8bdefe30ac637f9.jpg)

![0f96330b42f6f4b2f24dda8eed4db1cdeeb53661cdea41cd43070630106ee267.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/0f96330b42f6f4b2f24dda8eed4db1cdeeb53661cdea41cd43070630106ee267.jpg)

![103db2f2da650c4ecb39f246fa78c4119e1603fa2df24ed76487aeef4bd0e26d.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/103db2f2da650c4ecb39f246fa78c4119e1603fa2df24ed76487aeef4bd0e26d.jpg)

![4c4db24f0a2629e40c2d7503de9592bc8eddfbb9254055c11a675ec3fcefe488.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/4c4db24f0a2629e40c2d7503de9592bc8eddfbb9254055c11a675ec3fcefe488.jpg)

![51db7b6a54aeebb9c2a3393708cb0da1d11ed847b651bd92fb494960521a8edb.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/51db7b6a54aeebb9c2a3393708cb0da1d11ed847b651bd92fb494960521a8edb.jpg)

![5c1de03068f279f4aa7db859ae33364bce68d1a9f23c65d06695eeeb7e2247a0.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/5c1de03068f279f4aa7db859ae33364bce68d1a9f23c65d06695eeeb7e2247a0.jpg)

![739d91b006fb152f4b46ac9369b175dfa1b3924dc15ece0c605fe01480673219.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/739d91b006fb152f4b46ac9369b175dfa1b3924dc15ece0c605fe01480673219.jpg)

![90054953fe9af09e853cef6ed00de62c84779f1188d7287bfc94893508e056f3.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/90054953fe9af09e853cef6ed00de62c84779f1188d7287bfc94893508e056f3.jpg)

![a929196420e6aa402db611a52634fbdb09a44390afe356402813ec986c304ecf.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/a929196420e6aa402db611a52634fbdb09a44390afe356402813ec986c304ecf.jpg)

![d353fa7040bfb6aa5b666c2c05cfdc2878078329ae25ac28a6626684c723fcbc.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/images/d353fa7040bfb6aa5b666c2c05cfdc2878078329ae25ac28a6626684c723fcbc.jpg)

### Tables

![011d9dde1d948728015f8d5a81f54d02e408700a28f132a5a3641b26a81a6e62.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/011d9dde1d948728015f8d5a81f54d02e408700a28f132a5a3641b26a81a6e62.jpg)

![080f66b7bf7b2da8d1d79dc4fcc0ea333f856c3c3e3f3b256ac0d88883278cf8.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/080f66b7bf7b2da8d1d79dc4fcc0ea333f856c3c3e3f3b256ac0d88883278cf8.jpg)

![090ca8f0027f0519be0a86112bdeab823c89c1084f704d2645756af5427b72bf.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/090ca8f0027f0519be0a86112bdeab823c89c1084f704d2645756af5427b72bf.jpg)

![a56c19dc0e912bf35e99ebd6f1fda2d7068e704f22dbe2ff3427db78f887a979.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/a56c19dc0e912bf35e99ebd6f1fda2d7068e704f22dbe2ff3427db78f887a979.jpg)

![e84710dbd9345186d1816ca0e44b46296e7afd479e953b794fb3e1e75b522ea3.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/e84710dbd9345186d1816ca0e44b46296e7afd479e953b794fb3e1e75b522ea3.jpg)

![f098a1b6fb21b0679043b225f0e28a65627a46d9ec4120ed6fd8d67c071d9f15.jpg](../icml_results/2498_Tucker%20Attention_%20A%20generalization%20of%20approximate%20attention%20mechanisms/tables/f098a1b6fb21b0679043b225f0e28a65627a46d9ec4120ed6fd8d67c071d9f15.jpg)

<a id="paper-2469"></a>

## Skewness-Robust Causal Discovery in Location-Scale Noise Models


### Images

![21f2d53b97ee40d0abad13127af9093bda1131132060ac3bea7d9df2ce98e330.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/21f2d53b97ee40d0abad13127af9093bda1131132060ac3bea7d9df2ce98e330.jpg)

![2c37be3cb47a81a4d3a23bfd47bfd9dcf25c9bce03980894ad99bc76691d4f76.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/2c37be3cb47a81a4d3a23bfd47bfd9dcf25c9bce03980894ad99bc76691d4f76.jpg)

![4423b98fd58958f55dffd47aba43f29a570f04580a2c9754f4b338180bc0c153.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/4423b98fd58958f55dffd47aba43f29a570f04580a2c9754f4b338180bc0c153.jpg)

![61de04f1760f2b2ab652fe79023e32397c91367344d59ee549dda5f1ade9a6fd.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/61de04f1760f2b2ab652fe79023e32397c91367344d59ee549dda5f1ade9a6fd.jpg)

![6853809e991cd678cba918ac130dc271e7aa2c22ae7e5e99e9932b26d5285045.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/6853809e991cd678cba918ac130dc271e7aa2c22ae7e5e99e9932b26d5285045.jpg)

![d11dd79934870d421fb2044f16f8657db00f64869c67e257b1299659cc320841.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/d11dd79934870d421fb2044f16f8657db00f64869c67e257b1299659cc320841.jpg)

![d7d3fde77a886fd91b6fe83a6f13e0099837daa3af8f88646e9890fd1451e049.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/d7d3fde77a886fd91b6fe83a6f13e0099837daa3af8f88646e9890fd1451e049.jpg)

![f521a84cc9e04a8f529d3a6b8b687f17a2337625bd28d2cc62408fc434dbcd4f.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/f521a84cc9e04a8f529d3a6b8b687f17a2337625bd28d2cc62408fc434dbcd4f.jpg)

![fb0ce49340de90e88498f5d9add6f6e94f9972952b34118bc8bb616298cf26d4.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/images/fb0ce49340de90e88498f5d9add6f6e94f9972952b34118bc8bb616298cf26d4.jpg)

### Tables

![0027a526676146bb22f4635488ed27eb195cf2cfc1ea029f5a343dca2c3811ed.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/0027a526676146bb22f4635488ed27eb195cf2cfc1ea029f5a343dca2c3811ed.jpg)

![08300ae6624493e524fedcec888636f91f84db0df4865a22bc114e5d8fd514ab.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/08300ae6624493e524fedcec888636f91f84db0df4865a22bc114e5d8fd514ab.jpg)

![3476b4d409f0abdcc086b572db24e69521da74631ca5cf5e0dcd0d7e08bea3a3.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/3476b4d409f0abdcc086b572db24e69521da74631ca5cf5e0dcd0d7e08bea3a3.jpg)

![38aa8551fa8c0f9fdf9ecd8617e8e6f302457775e42f5f3f8fb078996d75f63f.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/38aa8551fa8c0f9fdf9ecd8617e8e6f302457775e42f5f3f8fb078996d75f63f.jpg)

![454a7ac43509fb3a12e0b52fe867b62ff0884b714adf03213c05d7ebd26f35e8.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/454a7ac43509fb3a12e0b52fe867b62ff0884b714adf03213c05d7ebd26f35e8.jpg)

![55fe722dce19d8fcb65bd7e79397360624c8e6dbac1d5a476201c5d1addf4db4.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/55fe722dce19d8fcb65bd7e79397360624c8e6dbac1d5a476201c5d1addf4db4.jpg)

![6e9650f09f1c19feb4760d67c08bc8b74300a3b1bb4d29ba1c040f05dcb10925.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/6e9650f09f1c19feb4760d67c08bc8b74300a3b1bb4d29ba1c040f05dcb10925.jpg)

![714d4ee85c93012e939f911e731a7d1f16b5cc07a83d553d2f239d6bfd6a65d7.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/714d4ee85c93012e939f911e731a7d1f16b5cc07a83d553d2f239d6bfd6a65d7.jpg)

![71e116389d31bfe7ae3aa330c2c20a738fa9074f70b1a0c3b8e16c881f03e4a5.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/71e116389d31bfe7ae3aa330c2c20a738fa9074f70b1a0c3b8e16c881f03e4a5.jpg)

![971211b7e596f153dfcd1b4f16ee3fbf3899b08277deb159897644259f7e4104.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/971211b7e596f153dfcd1b4f16ee3fbf3899b08277deb159897644259f7e4104.jpg)

![b277ca846fc9d4980277cf397c25ab637bed28735abb686bfa9f1fb4d58c90d0.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/b277ca846fc9d4980277cf397c25ab637bed28735abb686bfa9f1fb4d58c90d0.jpg)

![c53948bc31942553e176aadea813ca05c2604c5bf984b9030beffae2c7951318.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/c53948bc31942553e176aadea813ca05c2604c5bf984b9030beffae2c7951318.jpg)

![d21cfcb7571a7e55568fa1b5e33f1d2a2dade2594aed00690701f8fbbb846b7d.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/d21cfcb7571a7e55568fa1b5e33f1d2a2dade2594aed00690701f8fbbb846b7d.jpg)

![df4fb358bfcd5509b3ca7f2153a910fe3169f3da7ad795f0692ba06d62865920.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/df4fb358bfcd5509b3ca7f2153a910fe3169f3da7ad795f0692ba06d62865920.jpg)

![f18090760999ec57c199cf00d1c75f44ea5e2ece6f83d94f6acedfabbb4322c1.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/f18090760999ec57c199cf00d1c75f44ea5e2ece6f83d94f6acedfabbb4322c1.jpg)

![f6cdd48a78ee78727a13514c3a5bcf496db1a68d16a446c410cceb2f21dce7bd.jpg](../icml_results/2499_Skewness-Robust%20Causal%20Discovery%20in%20Location-Scale%20Noise%20Models/tables/f6cdd48a78ee78727a13514c3a5bcf496db1a68d16a446c410cceb2f21dce7bd.jpg)

<a id="paper-2470"></a>

## Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)


### Images

![023831ddd3be23c8661314fb159087cbace52482844df88e683fcd04993105c6.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/023831ddd3be23c8661314fb159087cbace52482844df88e683fcd04993105c6.jpg)

![0b0d5422e713b05c09c110e888c9934191f10d204cb96f2e062d1820b4cc413d.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/0b0d5422e713b05c09c110e888c9934191f10d204cb96f2e062d1820b4cc413d.jpg)

![217074e6fe8100e7782ddf6019b49fb31d4faf66658e853270e4440df2b97282.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/217074e6fe8100e7782ddf6019b49fb31d4faf66658e853270e4440df2b97282.jpg)

![554d964bca97ce3eef4149dc429bf4423e9d6dbda73c700648703e87b3f1ddd0.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/554d964bca97ce3eef4149dc429bf4423e9d6dbda73c700648703e87b3f1ddd0.jpg)

![5be2a6cdfcd8ab2937fc63dc6acb88f2690b1e657beb40af8b1285f4451b3b67.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/5be2a6cdfcd8ab2937fc63dc6acb88f2690b1e657beb40af8b1285f4451b3b67.jpg)

![8bf097a28a743e1b39a7410ea9c368183b2725cafac223f09e3a1e07e77c270f.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/8bf097a28a743e1b39a7410ea9c368183b2725cafac223f09e3a1e07e77c270f.jpg)

![8dafbd681e0e1916cf5b15b69b05316c53c9e75837e12729d691ddf2f86a1d29.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/8dafbd681e0e1916cf5b15b69b05316c53c9e75837e12729d691ddf2f86a1d29.jpg)

![934531841f7e1a46a257ec7beee18adafd4cb5be8b8b45e1ae418264d51affca.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/934531841f7e1a46a257ec7beee18adafd4cb5be8b8b45e1ae418264d51affca.jpg)

![9db4a24763e7ac0755cc43122358977ecd9d01a67d1e5286175095bbeba0da83.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/9db4a24763e7ac0755cc43122358977ecd9d01a67d1e5286175095bbeba0da83.jpg)

![acf30e6e37fe926b222962e085ce1158b25e83af3fbb500376fdfe682090add8.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/acf30e6e37fe926b222962e085ce1158b25e83af3fbb500376fdfe682090add8.jpg)

![fee723ef7c1c2a60798ba54b241a85c466e7dd68d4be9e58c7942866a03b2ffa.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/images/fee723ef7c1c2a60798ba54b241a85c466e7dd68d4be9e58c7942866a03b2ffa.jpg)

### Tables

![0c1aa2df609f6b4621a8a3388e8e4f228d99ad357967dd0cbaef7c2f1980b0d5.jpg](../icml_results/2500_Reinforcement%20Learning%20for%20Tool-Calling%20Agents%20in%20Fast%20Healthcare%20Interoperability%20Resources%20%28FHIR%29/tables/0c1aa2df609f6b4621a8a3388e8e4f228d99ad357967dd0cbaef7c2f1980b0d5.jpg)

<a id="paper-2471"></a>

## Tractable Expected Information Gains for Exponential Family Posteriors


### Images

![03347e1a6e626d1e959d419a546daa681ad7274f111e6db20317d6f5ec8e1260.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/03347e1a6e626d1e959d419a546daa681ad7274f111e6db20317d6f5ec8e1260.jpg)

![0b038242a81630fd919bdef57bc5148aeecabae8f40874b74fbbd47b7eac076f.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/0b038242a81630fd919bdef57bc5148aeecabae8f40874b74fbbd47b7eac076f.jpg)

![3b3d1ea82c1635992386110fb604f82bf5e3be468ce910cbb59320c70d447eed.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/3b3d1ea82c1635992386110fb604f82bf5e3be468ce910cbb59320c70d447eed.jpg)

![79ef3e0775adf9473e58921d23b55cc6c5c30487bb0717df0297cddb1ddb2233.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/79ef3e0775adf9473e58921d23b55cc6c5c30487bb0717df0297cddb1ddb2233.jpg)

![8022b2830724b56621b45c85fcd01036f5df679deec40909c2035d9bd5519470.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/8022b2830724b56621b45c85fcd01036f5df679deec40909c2035d9bd5519470.jpg)

![9003e8c926c29994d53cc3fdeb912e3624fb8c033bf7740a765babc9220c4ae0.jpg](../icml_results/2501_Tractable%20Expected%20Information%20Gains%20for%20Exponential%20Family%20Posteriors/images/9003e8c926c29994d53cc3fdeb912e3624fb8c033bf7740a765babc9220c4ae0.jpg)

<a id="paper-2472"></a>

## Beyond the Bellman Recursion: A Pontryagin-Guided Framework for Non-Exponential Discounting


### Images

![2a98eda3e9d3410ca57bc9e2c86a249bb49d861334abc83c7d76a4ead601b46f.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/2a98eda3e9d3410ca57bc9e2c86a249bb49d861334abc83c7d76a4ead601b46f.jpg)

![5b73524b4d01624c1961f9738e714131cdc37ceba58d4db526aa13e4553e12b2.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/5b73524b4d01624c1961f9738e714131cdc37ceba58d4db526aa13e4553e12b2.jpg)

![5c46952c6dc66aeb692c2e4691ccf7749d93d4aaab5ba45097f1debe8c5f66d8.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/5c46952c6dc66aeb692c2e4691ccf7749d93d4aaab5ba45097f1debe8c5f66d8.jpg)

![7477218f926216414121e0a1cf96eb5f6d1a7b81e3c7d00f164fe53b0a591965.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/7477218f926216414121e0a1cf96eb5f6d1a7b81e3c7d00f164fe53b0a591965.jpg)

![76550a2b0ed182c11e1cfaf2672a43122980bfe1d56a354d53d2398e898a5bcb.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/76550a2b0ed182c11e1cfaf2672a43122980bfe1d56a354d53d2398e898a5bcb.jpg)

![b976f20ed124de5c7db74203d116855c1a4466562d298d1a3be0eccbac34b2b1.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/b976f20ed124de5c7db74203d116855c1a4466562d298d1a3be0eccbac34b2b1.jpg)

![fe7fc10786ee7c37828ed4b213ac79c3871792b0f950eefe315c4c3752f23656.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/images/fe7fc10786ee7c37828ed4b213ac79c3871792b0f950eefe315c4c3752f23656.jpg)

### Tables

![1a7a3bf2f719085c7dcb3d4b028707acc81233bcdae992735e9d8cc8166d8391.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/tables/1a7a3bf2f719085c7dcb3d4b028707acc81233bcdae992735e9d8cc8166d8391.jpg)

![354688c6fc0a98021e43211f54b3f302e544545cb6185547546eca465838ed99.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/tables/354688c6fc0a98021e43211f54b3f302e544545cb6185547546eca465838ed99.jpg)

![60e4a4d721a3b39b4cdf532e1eda81e43f601e8425398c67d44560f6a0f764af.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/tables/60e4a4d721a3b39b4cdf532e1eda81e43f601e8425398c67d44560f6a0f764af.jpg)

![6f62461ed98e6d1c14d3a848663368e9e7c662fca5e66bb8c6981f5cf862c0ec.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/tables/6f62461ed98e6d1c14d3a848663368e9e7c662fca5e66bb8c6981f5cf862c0ec.jpg)

![7934baff166e7e516954915a7d4e1561cc5328d3c373f6f02f6292a82e4680ef.jpg](../icml_results/2502_Beyond%20the%20Bellman%20Recursion_%20A%20Pontryagin-Guided%20Framework%20for%20Non-Exponential%20Discounting/tables/7934baff166e7e516954915a7d4e1561cc5328d3c373f6f02f6292a82e4680ef.jpg)

<a id="paper-2473"></a>

## From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents


### Images

![0580d37d8f7c41ca28eb52ca56f3a62551f98ab215eb20efdc19e253b7ccd0c7.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/0580d37d8f7c41ca28eb52ca56f3a62551f98ab215eb20efdc19e253b7ccd0c7.jpg)

![81ed21bd8284f5dec3973541c13ce27db26049cec5158d055e0849affcce36a7.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/81ed21bd8284f5dec3973541c13ce27db26049cec5158d055e0849affcce36a7.jpg)

![921b3b9430d66891363b5d98ff405e9e91458f8975e88fa69c048405da4dadc6.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/921b3b9430d66891363b5d98ff405e9e91458f8975e88fa69c048405da4dadc6.jpg)

![93dac8d2486a5f489bd3cdc7ef038923b0d9110b007022b885855e8759c3cdb9.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/93dac8d2486a5f489bd3cdc7ef038923b0d9110b007022b885855e8759c3cdb9.jpg)

![e8bf4c6c4a60b2d78dbb552771268a34bd3f551192adac327ccd05262e0720be.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/e8bf4c6c4a60b2d78dbb552771268a34bd3f551192adac327ccd05262e0720be.jpg)

![fcb006b766dd6d9c03ee4f586913d19bcacc61b6511ece07f347e5f49581a2a3.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/images/fcb006b766dd6d9c03ee4f586913d19bcacc61b6511ece07f347e5f49581a2a3.jpg)

### Tables

![0dc1d6911843331d40b5ea06cf940aa26a28da66523ef7215c17b2bf1552328e.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/tables/0dc1d6911843331d40b5ea06cf940aa26a28da66523ef7215c17b2bf1552328e.jpg)

![128d7e455d061fba5a54688c7f0e54618ff2d2de794fea73dd3620639d851804.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/tables/128d7e455d061fba5a54688c7f0e54618ff2d2de794fea73dd3620639d851804.jpg)

![6af0a06eaace1b42eb76c0ea355d5fbff81fe141a32fd25197c950d24d6b46a1.jpg](../icml_results/2503_From%20Weak%20Cues%20to%20Real%20Identities_%20Evaluating%20Inference-Driven%20De-Anonymization%20in%20LLM%20Agents/tables/6af0a06eaace1b42eb76c0ea355d5fbff81fe141a32fd25197c950d24d6b46a1.jpg)

<a id="paper-2474"></a>

## Geometric Embedding Alignment via Curvature Matching in Transfer Learning


### Images

![2d7e8aa0266c6a4347499068f97e17fc765bfb03b2f84c2a2ba3229fc2f2e1a1.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/2d7e8aa0266c6a4347499068f97e17fc765bfb03b2f84c2a2ba3229fc2f2e1a1.jpg)

![35f916abba5e9bf86c62304a1c36ba96d89296f6d7fed5b2ad27b0f84f8f64e3.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/35f916abba5e9bf86c62304a1c36ba96d89296f6d7fed5b2ad27b0f84f8f64e3.jpg)

![50e194eabc27eb6f44ff64d2a682d7ff6293c6aef9cafad2cc026271f2f1c3be.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/50e194eabc27eb6f44ff64d2a682d7ff6293c6aef9cafad2cc026271f2f1c3be.jpg)

![533e02c9ef230d6d1647fe20e3d02dde19d234c4854f255ca0eecfaa97cfaa4a.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/533e02c9ef230d6d1647fe20e3d02dde19d234c4854f255ca0eecfaa97cfaa4a.jpg)

![76c97bce6971988c2d07afa7cda4d301dd35c7c03676d9900b08f5d8a5eb6b30.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/76c97bce6971988c2d07afa7cda4d301dd35c7c03676d9900b08f5d8a5eb6b30.jpg)

![8a90e5e53358a2f3abd86ce70713fafbaf8a6a307adc7dbb25fdb9f6e05d1cbe.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/8a90e5e53358a2f3abd86ce70713fafbaf8a6a307adc7dbb25fdb9f6e05d1cbe.jpg)

![a3cd9a3a6b5373054334d07a551d2df639d78061e35d2420f930055b27e6592e.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/a3cd9a3a6b5373054334d07a551d2df639d78061e35d2420f930055b27e6592e.jpg)

![f76381597ac546a144354023b8102be1ee811752745b033ac9519ac713effd2b.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/images/f76381597ac546a144354023b8102be1ee811752745b033ac9519ac713effd2b.jpg)

### Tables

![3675df3d7b5d38ef97b0cf1f298b88bc598077f650794ff5433d55f751e3b615.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/3675df3d7b5d38ef97b0cf1f298b88bc598077f650794ff5433d55f751e3b615.jpg)

![3d524c27bb8a17dea678e016b92f6bd8dda47d7c26cac20372c95ce128a49327.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/3d524c27bb8a17dea678e016b92f6bd8dda47d7c26cac20372c95ce128a49327.jpg)

![4ad2ee5beee8866e0b9aefeba054843b263ee7ae69fae1d84ca85b7ffae7bebd.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/4ad2ee5beee8866e0b9aefeba054843b263ee7ae69fae1d84ca85b7ffae7bebd.jpg)

![5290894140e2bc3bbeb686fd7d9c89dfb80e588a12cf2fbce5fba8226956f002.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/5290894140e2bc3bbeb686fd7d9c89dfb80e588a12cf2fbce5fba8226956f002.jpg)

![6a731f6b31b412e841dbae380dc22cd5587bddeb311871ad82da81d4047337cf.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/6a731f6b31b412e841dbae380dc22cd5587bddeb311871ad82da81d4047337cf.jpg)

![6f8700a7dc4d51372cd5cae73600ac9dc4d5bb25bd18275e110044aa66a8b62b.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/6f8700a7dc4d51372cd5cae73600ac9dc4d5bb25bd18275e110044aa66a8b62b.jpg)

![719bb82cd0aa0193b644da136f06478baa4c5d95e55c26e7f87f319c10ab62fd.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/719bb82cd0aa0193b644da136f06478baa4c5d95e55c26e7f87f319c10ab62fd.jpg)

![8ac9cafac7c6ebb0dbd609d1b8e77ad5bf9b378bc3065f706b69d8aa610ffed4.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/8ac9cafac7c6ebb0dbd609d1b8e77ad5bf9b378bc3065f706b69d8aa610ffed4.jpg)

![98b75475bc0cbd402456abad772263418568a082c91f68d75d5ba4e9ae60b0e9.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/98b75475bc0cbd402456abad772263418568a082c91f68d75d5ba4e9ae60b0e9.jpg)

![b6e594818e5073e13174886f83c4dbe6ca2a74980d40a1d3c86f1699a6660a6f.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/b6e594818e5073e13174886f83c4dbe6ca2a74980d40a1d3c86f1699a6660a6f.jpg)

![f66c297ea768e05c3f0674dcd3e6cd9f18b1e69704284ec6e718c6b0fb79016e.jpg](../icml_results/2504_Geometric%20Embedding%20Alignment%20via%20Curvature%20Matching%20in%20Transfer%20Learning/tables/f66c297ea768e05c3f0674dcd3e6cd9f18b1e69704284ec6e718c6b0fb79016e.jpg)

<a id="paper-2475"></a>

## Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks

### Tables

![1359e647619574f9fdcbecbdf41e1e61740ed56c79410c4024b499cbcc83dbd6.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/1359e647619574f9fdcbecbdf41e1e61740ed56c79410c4024b499cbcc83dbd6.jpg)

![13873268d399abaf2723d9d41d0ee41e2994abc7e99521a35fd57cdd34205726.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/13873268d399abaf2723d9d41d0ee41e2994abc7e99521a35fd57cdd34205726.jpg)

![44a37db26c7faa1b7ebfaf78c3657cd75202486fa046140e294e0924eff08fea.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/44a37db26c7faa1b7ebfaf78c3657cd75202486fa046140e294e0924eff08fea.jpg)

![485954148bfecdb8f7c96b73763392d7d5cb91d2a3758c1d41dbb05597e5e330.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/485954148bfecdb8f7c96b73763392d7d5cb91d2a3758c1d41dbb05597e5e330.jpg)

![6ddfa93e5b63f2643cb18ccf26370bebbdfdb4ded7a7de4621a8e13f68ff5286.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/6ddfa93e5b63f2643cb18ccf26370bebbdfdb4ded7a7de4621a8e13f68ff5286.jpg)

![7ed37733ee3b43cd24dd8a6ec81e9cf541cdd17a807f5b61303a669db6cc1ce4.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/7ed37733ee3b43cd24dd8a6ec81e9cf541cdd17a807f5b61303a669db6cc1ce4.jpg)

![94cf6abd2bb8c79dbadc60d948ee36e708dad73ca4b3e047725bcb072a395896.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/94cf6abd2bb8c79dbadc60d948ee36e708dad73ca4b3e047725bcb072a395896.jpg)

![ddfad474a3d28bb0c404a90308472d9aa0777171831eff14530f33fc36a387b1.jpg](../icml_results/2505_Riemannian%20Diffusion%20Models%20on%20General%20Manifolds%20via%20Physics-Informed%20Neural%20Networks/tables/ddfad474a3d28bb0c404a90308472d9aa0777171831eff14530f33fc36a387b1.jpg)
