# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 85 of 197

## 目录 (Table of Contents)

1. [WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](#paper-2773)
2. [Layer-wise Gradient Disentanglement: Decoupling Semantics and Preferences in Direct Preference Optimization](#paper-2774)
3. [AdverMCTS: Combating Pseudo-Correctness in Code Generation via Adversarial Monte Carlo Tree Search](#paper-2775)
4. [Towards One-for-All Anomaly Detection for Tabular Data](#paper-2776)
5. [Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion](#paper-2777)
6. [Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning](#paper-2778)
7. [How Can Mamba Learn In Context with Outliers and Generalize Provably?](#paper-2779)
8. [Constructing Industrial-Scale Optimization Modeling Benchmark](#paper-2780)
9. [GeoDM: Geometry-aware Distribution Matching for Dataset Distillation](#paper-2781)
10. [Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models](#paper-2782)
11. [Predicting Large Model Test Losses with a Noisy Quadratic System](#paper-2783)
12. [Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](#paper-2784)
13. [A Minimax Approach for Optimal Intervention Policy Learning with Two-Stage Outcomes](#paper-2785)
14. [Transformers Learn the Optimal DDPM Denoiser for Multi-Token GMMs](#paper-2786)
15. [CryoACE: An Atom-centric Framework for Accurate and Automated Model Building in Cryo-EM](#paper-2787)
16. [OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](#paper-2788)
17. [Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](#paper-2789)
18. [OptProver: Bridging Olympiad and Optimization through Continual Training in Formal Theorem Proving](#paper-2790)
19. [Causal Structure Learning for Sparse Matrix Fill-in Reduction](#paper-2791)
20. [PCGS: Deblurring 3D Gaussian Splatting with Patch Comparison](#paper-2792)
21. [Solving Spatial-Spectral Fusion with Latent Spectral Operators](#paper-2793)
22. [BVS: Bayesian Visual Search with Multimodal Large Language Model for Fine-grained Perception](#paper-2794)
23. [When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](#paper-2795)
24. [Energy-Structured Low-Rank Adaptation for Continual Learning](#paper-2796)
25. [Intrinsic Gradient Suppression for Label-Noise Prompt Tuning in Vision–Language Models](#paper-2797)
26. [Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](#paper-2798)
27. [Reasoning as an Attack Surface: Adaptive Evolutionary CoT Jailbreaks for LLMs](#paper-2799)
28. [AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing](#paper-2800)
29. [A Studentized Spherical Harmonics–Based Nonparametric Two-Sample Test for Compositional and Directional Data](#paper-2801)
30. [Safe Reinforcement Learning with Preference-based Constraint Inference](#paper-2802)
31. [SURF: Separation via Unsupervised Remixing Flow](#paper-2803)
32. [Towards Parameter-Free Temporal Difference Learning](#paper-2804)
33. [SHAP-Guided Kernel Actor-Critic for Explainable Reinforcement Learning](#paper-2805)

---


<a id="paper-2773"></a>

## WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points

### Images

![3ea0c5a6f5a76a90af89c453b83dd476ccd843b3faed2d26de7cf232bec5e477.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/3ea0c5a6f5a76a90af89c453b83dd476ccd843b3faed2d26de7cf232bec5e477.jpg)

![3f80e0865a3aed1cdee6879992877242071a49e682e60e035cee298dc7409cfe.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/3f80e0865a3aed1cdee6879992877242071a49e682e60e035cee298dc7409cfe.jpg)

![40e7fa88d3a8b0d86c5b6f19fdb704e96da5ae9c9d74f37eb6be46a6fe36acbd.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/40e7fa88d3a8b0d86c5b6f19fdb704e96da5ae9c9d74f37eb6be46a6fe36acbd.jpg)

![41f93ee87b2a54ae43db77a176bb66ffdfe2591ee6ae586f25899e57bccde58c.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/41f93ee87b2a54ae43db77a176bb66ffdfe2591ee6ae586f25899e57bccde58c.jpg)

![46a25f0a132a391c24dd27346da07aec9daeffe79f9ec19b8e508641c39677a1.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/46a25f0a132a391c24dd27346da07aec9daeffe79f9ec19b8e508641c39677a1.jpg)

![5a6d1ad555a0d2a8e5144fecb9e2ca335ed67d6f1e300b6e5277464cfb4da76b.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/5a6d1ad555a0d2a8e5144fecb9e2ca335ed67d6f1e300b6e5277464cfb4da76b.jpg)

![62c93a3f373e7d65023651a88c0da0f868a2a0960022b73f84d4233fb8cf724b.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/62c93a3f373e7d65023651a88c0da0f868a2a0960022b73f84d4233fb8cf724b.jpg)

![8e32929c9f6a9a93f26a3a2437675b7ec0fddd729b5d81beda348eb8e84c2092.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/8e32929c9f6a9a93f26a3a2437675b7ec0fddd729b5d81beda348eb8e84c2092.jpg)

![9230ab84b000904eb800e8b51a6ac0d26cbae4f3a0a2f19bf613b70219fa857a.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/9230ab84b000904eb800e8b51a6ac0d26cbae4f3a0a2f19bf613b70219fa857a.jpg)

![bef63a73c5933b33c145ee42723b5a3f2b694b338e79a7c90cf44bd0a012823a.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/bef63a73c5933b33c145ee42723b5a3f2b694b338e79a7c90cf44bd0a012823a.jpg)

![e49345f86fabd2d1e67dd2a74129f667d992dd381d103adbf68b7d05b30e016b.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/images/e49345f86fabd2d1e67dd2a74129f667d992dd381d103adbf68b7d05b30e016b.jpg)

### Tables

![2d6ce09a15d509988b106a8fa6c0e58b14682aa58cc98161e0f78b0e38651246.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/tables/2d6ce09a15d509988b106a8fa6c0e58b14682aa58cc98161e0f78b0e38651246.jpg)

![2e0eafa57f9357b3d30a701599d0494913bb516a309e9f52b8f0f8ae8ab368ff.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/tables/2e0eafa57f9357b3d30a701599d0494913bb516a309e9f52b8f0f8ae8ab368ff.jpg)

![5e3598cc1d160782ef258db9eea2d9c964e56124ed63a566d383afcff99e0375.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/tables/5e3598cc1d160782ef258db9eea2d9c964e56124ed63a566d383afcff99e0375.jpg)

![a55435b6b8b1dcaf8ff126e4ec12236d51a2eace48e25ba1e0f57b89054f7b33.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/tables/a55435b6b8b1dcaf8ff126e4ec12236d51a2eace48e25ba1e0f57b89054f7b33.jpg)

![a8ad1d66313bd27fa35210808a284a2af69c357b9b96c76246831bb44745c118.jpg](../icml_results/2808_Not%20All%20Prefills%20Are%20Equal_%20PPD%20Disaggregation%20for%20Multi-turn%20LLM%20Serving/tables/a8ad1d66313bd27fa35210808a284a2af69c357b9b96c76246831bb44745c118.jpg)

## WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points


### Images

![06b041a6ba84af0d9ec4c6dbfb3e7cf5162837b80e21518728e7970a7917db34.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/06b041a6ba84af0d9ec4c6dbfb3e7cf5162837b80e21518728e7970a7917db34.jpg)

![08b126085cbdce522f43497133c026ae891a1bc8619264aed4ca9a8fed8e34f5.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/08b126085cbdce522f43497133c026ae891a1bc8619264aed4ca9a8fed8e34f5.jpg)

![21f37d3f0afbca24b78f4faa18a3e8fe888796228b147bd839642a1a3272c578.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/21f37d3f0afbca24b78f4faa18a3e8fe888796228b147bd839642a1a3272c578.jpg)

![2c0da317f16f4409fe2c3cb1cbe286de2af776c20b5c96d631c44c97d6eba0d6.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/2c0da317f16f4409fe2c3cb1cbe286de2af776c20b5c96d631c44c97d6eba0d6.jpg)

![36f881fc71ccab44ee99f5a066969cbf5f609348af680b126519cd2a45fc2d03.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/36f881fc71ccab44ee99f5a066969cbf5f609348af680b126519cd2a45fc2d03.jpg)

![73620293e56e55371547ebbeadd5dd3b03287aed5f6a13493909ecca48f8230e.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/73620293e56e55371547ebbeadd5dd3b03287aed5f6a13493909ecca48f8230e.jpg)

![8d5e98908332d02c310126eeaac7b754fa88b87fcd0bd5a2d993ebb0c8a4b186.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/8d5e98908332d02c310126eeaac7b754fa88b87fcd0bd5a2d993ebb0c8a4b186.jpg)

![bed8404413ac52d33221f59417847313bab4346a571787837d01048caf4d6d45.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/bed8404413ac52d33221f59417847313bab4346a571787837d01048caf4d6d45.jpg)

![d279bda02e59f1ec050ab7af54cdd83accd689ee91ba443533068094bb9e48f0.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/d279bda02e59f1ec050ab7af54cdd83accd689ee91ba443533068094bb9e48f0.jpg)

![d3edc4eb127fef522e2be9404530857d619f27887fc11e6e96b9db6d23cf36c2.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/d3edc4eb127fef522e2be9404530857d619f27887fc11e6e96b9db6d23cf36c2.jpg)

![d66f905b7b9bc345c04fed4367b24e39141965b8cfc10e465a4f9f84560181a9.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/images/d66f905b7b9bc345c04fed4367b24e39141965b8cfc10e465a4f9f84560181a9.jpg)

### Tables

![0ca80add41930bcc2b1f1895d8cd062e5c8c3982f7fb064792cb660733c44cd0.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/0ca80add41930bcc2b1f1895d8cd062e5c8c3982f7fb064792cb660733c44cd0.jpg)

![0e52eb28d0afbd1b631e400eaf64d3b98fccf32c0c93c3754c79b06972cc9fde.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/0e52eb28d0afbd1b631e400eaf64d3b98fccf32c0c93c3754c79b06972cc9fde.jpg)

![1a39094a3f5853aa25083af444f5eb2f647d7aea8d0326dd73dff585a48ca410.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/1a39094a3f5853aa25083af444f5eb2f647d7aea8d0326dd73dff585a48ca410.jpg)

![2df5e66d9932b330113fd8f0043e47c1c605a0a42872004ff1207cce82e45afe.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/2df5e66d9932b330113fd8f0043e47c1c605a0a42872004ff1207cce82e45afe.jpg)

![4da659a2f591f71199132a068521dc5775ddd95d838a12f13cb7130ca45ab30d.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/4da659a2f591f71199132a068521dc5775ddd95d838a12f13cb7130ca45ab30d.jpg)

![5bf536dd24bd4044721561ca3e534e8e2eab3e6c8b86d71f6959a6583031d9f5.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/5bf536dd24bd4044721561ca3e534e8e2eab3e6c8b86d71f6959a6583031d9f5.jpg)

![873618cbea6274507e956ca530d7c44422735fae5d55a73ad811927afa4cb274.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/873618cbea6274507e956ca530d7c44422735fae5d55a73ad811927afa4cb274.jpg)

![8f095e91eebec81b5d453317ef066f8cb91684706c4ab9222c59e037856352e8.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/8f095e91eebec81b5d453317ef066f8cb91684706c4ab9222c59e037856352e8.jpg)

![93e161aea37c309b1e4cceb8bc7a6d430d6dd8b7670c333ed07fa7f98f1636f8.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/93e161aea37c309b1e4cceb8bc7a6d430d6dd8b7670c333ed07fa7f98f1636f8.jpg)

![973e7e10e5b7082630422c5b8935f2dd8635dcbb494ad51b13482e0d466a288b.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/973e7e10e5b7082630422c5b8935f2dd8635dcbb494ad51b13482e0d466a288b.jpg)

![a08230a72138e43081715a1f6c9c5a5f149e40893b74976c23ad01b1c0fc5cf3.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/a08230a72138e43081715a1f6c9c5a5f149e40893b74976c23ad01b1c0fc5cf3.jpg)

![ac06199fec8f4150d347287a0dc4452d4bef0e1ecd390227c670a7d737e325aa.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/ac06199fec8f4150d347287a0dc4452d4bef0e1ecd390227c670a7d737e325aa.jpg)

![ad93b19d8753eb5db0f8857422e1ec203976dddc5b6c94fef95c74d148dfa077.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/ad93b19d8753eb5db0f8857422e1ec203976dddc5b6c94fef95c74d148dfa077.jpg)

![b1f76442cc5b594bf63e9c185ae6a34a7450c73ba0dcbf12ffae92365f787d65.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/b1f76442cc5b594bf63e9c185ae6a34a7450c73ba0dcbf12ffae92365f787d65.jpg)

![b2c73eb6ad796e3e7014a0ccd22853229dfe89e652e11eb04bdaa12632e75cb4.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/b2c73eb6ad796e3e7014a0ccd22853229dfe89e652e11eb04bdaa12632e75cb4.jpg)

![ea9c92d89df4e2e8fbd594aa8320e0a01d54917df3d6d44336c363842ae5518f.jpg](../icml_results/2809_WinQ_%20Accelerating%20Quantization-Aware%20Training%20of%20Language%20Models%20Around%20Saddle%20Points/tables/ea9c92d89df4e2e8fbd594aa8320e0a01d54917df3d6d44336c363842ae5518f.jpg)

<a id="paper-2774"></a>

## Layer-wise Gradient Disentanglement: Decoupling Semantics and Preferences in Direct Preference Optimization


### Images

![4b99ecadd5c026dd91924ef09d89471d10b8a632cb1e25ab4a1ff0c5df798303.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/4b99ecadd5c026dd91924ef09d89471d10b8a632cb1e25ab4a1ff0c5df798303.jpg)

![4c5d3dd804902d86fb91b6c0906a360f2d4b722c7087b35c6b04a1b287912f21.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/4c5d3dd804902d86fb91b6c0906a360f2d4b722c7087b35c6b04a1b287912f21.jpg)

![6fdf6efbe4a805ed744753390c07adf231cfb9bea7766b1d28f6822bb600dce8.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/6fdf6efbe4a805ed744753390c07adf231cfb9bea7766b1d28f6822bb600dce8.jpg)

![78f4addbed4796d480172cd0c8949815815f3b6cbe6c1d4fb789b671ac4ce347.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/78f4addbed4796d480172cd0c8949815815f3b6cbe6c1d4fb789b671ac4ce347.jpg)

![b3b525cb9ff3da534268d34e3296aa4c4dc0e6ac2eb48a4651f06f918fbc9ded.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/b3b525cb9ff3da534268d34e3296aa4c4dc0e6ac2eb48a4651f06f918fbc9ded.jpg)

![c2ceb8f47d0482de0d106fe1b6ee966c9c50947167a47038cffdfd8995cf7373.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/c2ceb8f47d0482de0d106fe1b6ee966c9c50947167a47038cffdfd8995cf7373.jpg)

![c64445ef4ef6e5cc0bdf4484d16db8dbded680e04b927e42566441b9d3cae57d.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/c64445ef4ef6e5cc0bdf4484d16db8dbded680e04b927e42566441b9d3cae57d.jpg)

![d28bb076fe496f820c016ea80f61e89ee38d84d2393f10c80bd0dd45da8124dd.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/images/d28bb076fe496f820c016ea80f61e89ee38d84d2393f10c80bd0dd45da8124dd.jpg)

### Tables

![0df787999b3a1cc55954d5aef8a0b29945f16fd2cc687873d18f150947f75b58.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/0df787999b3a1cc55954d5aef8a0b29945f16fd2cc687873d18f150947f75b58.jpg)

![305db79cfc21a2394e81c145923c56731a876bade7b284a9d21c44a9d3b9aa46.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/305db79cfc21a2394e81c145923c56731a876bade7b284a9d21c44a9d3b9aa46.jpg)

![33b7a3ce3ff776844521686e9667192e3ce8f58ed71b0d98d158da96817c06cb.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/33b7a3ce3ff776844521686e9667192e3ce8f58ed71b0d98d158da96817c06cb.jpg)

![340969266ec118f87e84102865b73f05b76d71b7c9c6897d109ee648bc38f9ba.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/340969266ec118f87e84102865b73f05b76d71b7c9c6897d109ee648bc38f9ba.jpg)

![361ceef31b8ebd4bc4f8daf2f9e3486d3241a17299fbd24811e69eb6f0f28ac4.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/361ceef31b8ebd4bc4f8daf2f9e3486d3241a17299fbd24811e69eb6f0f28ac4.jpg)

![4e6a07027d7679352849fccb0a8bbaea589ca8b43f4e93cae79dfd7a19e3048e.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/4e6a07027d7679352849fccb0a8bbaea589ca8b43f4e93cae79dfd7a19e3048e.jpg)

![56ab4c1e6bffd1e6a3d02e2285845367f2e1e614a580a0cc28b25b831e9502d3.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/56ab4c1e6bffd1e6a3d02e2285845367f2e1e614a580a0cc28b25b831e9502d3.jpg)

![829e1c3caaac062a5c94c9135e4b171161d069e3cc26ee9a9eef6501999855b3.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/829e1c3caaac062a5c94c9135e4b171161d069e3cc26ee9a9eef6501999855b3.jpg)

![968ad463c8b3d16f86817491d49ea9e8943c8fb941932ab8b897d5543b9ccab6.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/968ad463c8b3d16f86817491d49ea9e8943c8fb941932ab8b897d5543b9ccab6.jpg)

![9a188f39118c2fb0d997f76a7a5a28973d584dddd4bbef76b53a915cfd8083ce.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/9a188f39118c2fb0d997f76a7a5a28973d584dddd4bbef76b53a915cfd8083ce.jpg)

![9a29226983cf9500f72c118957bba56e84b7ac894ff2510052e036028b5a3448.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/9a29226983cf9500f72c118957bba56e84b7ac894ff2510052e036028b5a3448.jpg)

![9e16aff8e871b123f6a7c8cd4a87c41c90e94aeca84a597cec0982f886ef87ea.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/9e16aff8e871b123f6a7c8cd4a87c41c90e94aeca84a597cec0982f886ef87ea.jpg)

![cdca0c256b0a80a77840be8aefe75b05228a7144c729c211a3797ef04bdc8c8f.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/cdca0c256b0a80a77840be8aefe75b05228a7144c729c211a3797ef04bdc8c8f.jpg)

![e555ae2bb35a412c2eb7e6b980e940e2f9aa3dd045577067f90f1f2a0fa20f31.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/e555ae2bb35a412c2eb7e6b980e940e2f9aa3dd045577067f90f1f2a0fa20f31.jpg)

![fbbe3ad0d407923a5eafe37f3f77ff3847cc5b3e53b073d6a3935abd83e1ae53.jpg](../icml_results/2810_Layer-wise%20Gradient%20Disentanglement_%20Decoupling%20Semantics%20and%20Preferences%20in%20Direct%20Preference%20Optim/tables/fbbe3ad0d407923a5eafe37f3f77ff3847cc5b3e53b073d6a3935abd83e1ae53.jpg)

<a id="paper-2775"></a>

## AdverMCTS: Combating Pseudo-Correctness in Code Generation via Adversarial Monte Carlo Tree Search


### Images

![190d1fc244ffcaada58becd30d3eef0d5d97ad1b64458f88a4674056e78dd13f.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/190d1fc244ffcaada58becd30d3eef0d5d97ad1b64458f88a4674056e78dd13f.jpg)

![1d8e70f2f0c06bcb27579a96f73b90b45539e133da42653fb6b11702b5b568b9.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/1d8e70f2f0c06bcb27579a96f73b90b45539e133da42653fb6b11702b5b568b9.jpg)

![3c4e6f067b95e30ab12fc61355c82c0c85a90ff70534c59b6fb59c24c73faeae.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/3c4e6f067b95e30ab12fc61355c82c0c85a90ff70534c59b6fb59c24c73faeae.jpg)

![5e61c30db8dc8a94fd2779e9d231c9c6a9cba121e192f6cfae2faf6ccb83b863.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/5e61c30db8dc8a94fd2779e9d231c9c6a9cba121e192f6cfae2faf6ccb83b863.jpg)

![998f311a3a0bd002cc2895105b498489c285d8a4a557ab0c8edac87050adad61.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/998f311a3a0bd002cc2895105b498489c285d8a4a557ab0c8edac87050adad61.jpg)

![ad5fe17e3332a22cc6abe92262086e5211728f28d1fe84c5380d3fd4f8d48de4.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/ad5fe17e3332a22cc6abe92262086e5211728f28d1fe84c5380d3fd4f8d48de4.jpg)

![baa0e5e5b3c1f66476403de51cb72d3c4fe86c84d4e3656e3fd8a8c5bd3d92e9.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/baa0e5e5b3c1f66476403de51cb72d3c4fe86c84d4e3656e3fd8a8c5bd3d92e9.jpg)

![c3a8b95458ca2f8bef7b65ae81d2eb44e70e65ae43f48947a8da8953302bdc92.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/c3a8b95458ca2f8bef7b65ae81d2eb44e70e65ae43f48947a8da8953302bdc92.jpg)

![fa95c234a05e9288289456d243f7dee915d91d7eb6aa512e5867581a2fc21668.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/images/fa95c234a05e9288289456d243f7dee915d91d7eb6aa512e5867581a2fc21668.jpg)

### Tables

![2b37a14c82510a8ecfc0f332acd17351199c5aaa7bc7f0a2c3d7395b0697f4d2.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/tables/2b37a14c82510a8ecfc0f332acd17351199c5aaa7bc7f0a2c3d7395b0697f4d2.jpg)

![67f662218d450f1a72f09d9075107b28df014b81b8ab4edd04d660b683469b9a.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/tables/67f662218d450f1a72f09d9075107b28df014b81b8ab4edd04d660b683469b9a.jpg)

![7f6f00b234ad385c4608010f523e2b3619b764fc359f4aa9777828cb5be96475.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/tables/7f6f00b234ad385c4608010f523e2b3619b764fc359f4aa9777828cb5be96475.jpg)

![d94e5007df773fa7e207b0ec18dfb7778efc2ff98f6387be1aecf609f8d5e8ab.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/tables/d94e5007df773fa7e207b0ec18dfb7778efc2ff98f6387be1aecf609f8d5e8ab.jpg)

![f4291a4305e5fdd26f2367551f1c13a161a840eef07cb96ef4fdf0ec13a0a9bc.jpg](../icml_results/2811_AdverMCTS_%20Combating%20Pseudo-Correctness%20in%20Code%20Generation%20via%20Adversarial%20Monte%20Carlo%20Tree%20Search/tables/f4291a4305e5fdd26f2367551f1c13a161a840eef07cb96ef4fdf0ec13a0a9bc.jpg)

<a id="paper-2776"></a>

## Towards One-for-All Anomaly Detection for Tabular Data


### Images

![0f4e13a49a79c39523c4ecb4ba13a6619bdf47c5f3cebbd12250434857584016.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/0f4e13a49a79c39523c4ecb4ba13a6619bdf47c5f3cebbd12250434857584016.jpg)

![11164bc1a112f43258342008f015a503f0a30e1a95400c92952af51979d3d206.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/11164bc1a112f43258342008f015a503f0a30e1a95400c92952af51979d3d206.jpg)

![13afca5a2474e4f2dac41d04877b5f8df71ff3475fff5c4e60cb93a8ef765284.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/13afca5a2474e4f2dac41d04877b5f8df71ff3475fff5c4e60cb93a8ef765284.jpg)

![1c03fa1eacbe688ba7b3cf0263cbeb9bd035fa8763ce021181921fc992851065.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/1c03fa1eacbe688ba7b3cf0263cbeb9bd035fa8763ce021181921fc992851065.jpg)

![1d45b0eb782fd881096e3334107304aeec76b4d11f1e1f6c431483e3b84f89fa.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/1d45b0eb782fd881096e3334107304aeec76b4d11f1e1f6c431483e3b84f89fa.jpg)

![59786bbc60885d5d64d2f330edb2d0a06e950559543cb2d173770ab6003fe434.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/59786bbc60885d5d64d2f330edb2d0a06e950559543cb2d173770ab6003fe434.jpg)

![5a9ebc48acba240101135017be60c389beb1851b85d7a28fa13b65786bd6b097.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/5a9ebc48acba240101135017be60c389beb1851b85d7a28fa13b65786bd6b097.jpg)

![90ac6d218262c44b9c94961315ca596ac7f5a3774f12d035e439d06e09859c94.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/90ac6d218262c44b9c94961315ca596ac7f5a3774f12d035e439d06e09859c94.jpg)

![c298631e6675fc33d5267f5309009bfda6e42c4beb32de3825bc43426f6a3e05.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/c298631e6675fc33d5267f5309009bfda6e42c4beb32de3825bc43426f6a3e05.jpg)

![cb588a6de77fbddf64a7afc761846890f3fe82393541234ff8f5cb8d474b7538.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/images/cb588a6de77fbddf64a7afc761846890f3fe82393541234ff8f5cb8d474b7538.jpg)

### Tables

![1544d3ac21f9d4ff70d44df017b472c3d5c6ec84aa20688a42641cf964555ef2.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/1544d3ac21f9d4ff70d44df017b472c3d5c6ec84aa20688a42641cf964555ef2.jpg)

![1d5aa8925464f21164b17aec47d2e205ae15ae0d60c500ae8981c79925ff6783.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/1d5aa8925464f21164b17aec47d2e205ae15ae0d60c500ae8981c79925ff6783.jpg)

![3597534509ce44f698879728b6170d523080db1b8790287ccba55f8757a99d46.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/3597534509ce44f698879728b6170d523080db1b8790287ccba55f8757a99d46.jpg)

![ba0471728a2d94bc83a2a9806effc4b9fff84588bc42a4e68bbcb56ecb8588f2.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/ba0471728a2d94bc83a2a9806effc4b9fff84588bc42a4e68bbcb56ecb8588f2.jpg)

![bae49e8b0d2e114c96339011a70fd8363128536660f47000a4bf40c078096e74.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/bae49e8b0d2e114c96339011a70fd8363128536660f47000a4bf40c078096e74.jpg)

![d3fbe82abc208031851c0d04b3d166d7ae735b990bbf0de72b510a4e9769624c.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/d3fbe82abc208031851c0d04b3d166d7ae735b990bbf0de72b510a4e9769624c.jpg)

![dc96bc4a1d4b5f2962d7034117359fab3188977415eec8b332170c8ba04d4da3.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/dc96bc4a1d4b5f2962d7034117359fab3188977415eec8b332170c8ba04d4da3.jpg)

![dd886ad9ad8425eae60204cc56e02a9b48dc7f95e1c3829f17277110cf6820c7.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/dd886ad9ad8425eae60204cc56e02a9b48dc7f95e1c3829f17277110cf6820c7.jpg)

![f07313fd9bd2f03b980b3ca516b81efc046a675d0510c941dd7c434f4858adf7.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/f07313fd9bd2f03b980b3ca516b81efc046a675d0510c941dd7c434f4858adf7.jpg)

![f7fc6f230bc149ecb0c03019743f822dc102117836e01b7d2ed46ae6cfae5ee9.jpg](../icml_results/2812_Towards%20One-for-All%20Anomaly%20Detection%20for%20Tabular%20Data/tables/f7fc6f230bc149ecb0c03019743f822dc102117836e01b7d2ed46ae6cfae5ee9.jpg)

<a id="paper-2777"></a>

## Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion


### Images

![1171dccf3e80bd3ef95897df4eaf3fe5e85679857a74fc244c139b4e471bd235.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/1171dccf3e80bd3ef95897df4eaf3fe5e85679857a74fc244c139b4e471bd235.jpg)

![120d16f603d14534b6e1c8e5e1d9ce8cb3b043aa511535438809b86f709809aa.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/120d16f603d14534b6e1c8e5e1d9ce8cb3b043aa511535438809b86f709809aa.jpg)

![2620ea524e59376666eeb7f2ca2856fc6c0eddfe91f400bb4017470c570a832d.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/2620ea524e59376666eeb7f2ca2856fc6c0eddfe91f400bb4017470c570a832d.jpg)

![4e8dadb5dfe385fd67bc3dd10be1479dd9972c4d838757ac47673af8d03630bc.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/4e8dadb5dfe385fd67bc3dd10be1479dd9972c4d838757ac47673af8d03630bc.jpg)

![5542cbc62f55a14292922ddff97151d88f8fbc23c720c37f7d33489f518ac600.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/5542cbc62f55a14292922ddff97151d88f8fbc23c720c37f7d33489f518ac600.jpg)

![743be340ec3cf94415e5a501d50f00238ba3ac1f50fa77204d1ecacd4da28bd1.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/743be340ec3cf94415e5a501d50f00238ba3ac1f50fa77204d1ecacd4da28bd1.jpg)

![a7781213480e426a751bcef2e7838ffac33e9547222016dcb29452701d24d73a.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/a7781213480e426a751bcef2e7838ffac33e9547222016dcb29452701d24d73a.jpg)

![ae3ee68b8d4d2d70409bc9bf7c4eaa41c9404f6f0715076ff4417560b99796f9.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/ae3ee68b8d4d2d70409bc9bf7c4eaa41c9404f6f0715076ff4417560b99796f9.jpg)

![b92915ec3c744017d34bbb75bc9bd721693d8d656cb01f7f48c8ed5577d2457a.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/b92915ec3c744017d34bbb75bc9bd721693d8d656cb01f7f48c8ed5577d2457a.jpg)

![c5aed101acfe577956d6317bb046d1a5afa7587d51a4993a4f487f9a2f172779.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/c5aed101acfe577956d6317bb046d1a5afa7587d51a4993a4f487f9a2f172779.jpg)

![d02633bd626901721bd9f3e67d545983bee86111bca8433cc532250507c57191.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/d02633bd626901721bd9f3e67d545983bee86111bca8433cc532250507c57191.jpg)

![d5694a39b4a4a8f28735e6d35ee204214bc58009a2e3cd811e114ccbff6a84ef.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/d5694a39b4a4a8f28735e6d35ee204214bc58009a2e3cd811e114ccbff6a84ef.jpg)

![da60114f842886a90281fdcf871ca0236b436b2f71676078e3b3a5d93edfc538.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/da60114f842886a90281fdcf871ca0236b436b2f71676078e3b3a5d93edfc538.jpg)

![e0b2149075443e9a074e36772546f3acdb40a49564403b8d27782936c1c63330.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/e0b2149075443e9a074e36772546f3acdb40a49564403b8d27782936c1c63330.jpg)

![e4634a4b7c03630ece7a87a873a1b1bdbf0dd70a930f0dddc17f8376148db304.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/e4634a4b7c03630ece7a87a873a1b1bdbf0dd70a930f0dddc17f8376148db304.jpg)

![fa4c9af597986f622f928bb3262aa204bf55a63a8f65e31b94539fdcd9658aba.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/images/fa4c9af597986f622f928bb3262aa204bf55a63a8f65e31b94539fdcd9658aba.jpg)

### Tables

![13eefae7361aa06ce04159ab9b4e9e0494c2c510bbf46c414d39cda5fd5a3d36.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/13eefae7361aa06ce04159ab9b4e9e0494c2c510bbf46c414d39cda5fd5a3d36.jpg)

![22622b34b534cdb0667f1eaf668598b676eeba882a1e224513701dc27f886c7b.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/22622b34b534cdb0667f1eaf668598b676eeba882a1e224513701dc27f886c7b.jpg)

![2e0ee9f2edc33fd5b034c7700be5d2cff6c7605cd77e0b1153895aecba334b82.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/2e0ee9f2edc33fd5b034c7700be5d2cff6c7605cd77e0b1153895aecba334b82.jpg)

![3003f5676ce2e8345bb21b033f8fc43428dbc740915e2b8133cd73ba50041330.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/3003f5676ce2e8345bb21b033f8fc43428dbc740915e2b8133cd73ba50041330.jpg)

![6a0b2d96fad7042aee055c442e8a66cf16b7942bb826027d0fad105abd0f5227.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/6a0b2d96fad7042aee055c442e8a66cf16b7942bb826027d0fad105abd0f5227.jpg)

![7e1f00c596335f394e9f2acd0d05cd4ae3077a84a9948d01ff0c99bd5118fbb9.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/7e1f00c596335f394e9f2acd0d05cd4ae3077a84a9948d01ff0c99bd5118fbb9.jpg)

![a265fa3c5079660a972580006643256ab9a3c165022a83227ab4d92508791657.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/a265fa3c5079660a972580006643256ab9a3c165022a83227ab4d92508791657.jpg)

![d744819210ae7f30fd45635c087c8cfc6c018dc2f12bd5a348cb597749de5f52.jpg](../icml_results/2813_Omni-Diffusion_%20Unified%20Multimodal%20Understanding%20and%20Generation%20with%20Masked%20Discrete%20Diffusion/tables/d744819210ae7f30fd45635c087c8cfc6c018dc2f12bd5a348cb597749de5f52.jpg)

<a id="paper-2778"></a>

## Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning


### Images

![19e57c7d33ef2ecd14320d132183ee6286915ab6852590acfe5325b561a454b0.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/19e57c7d33ef2ecd14320d132183ee6286915ab6852590acfe5325b561a454b0.jpg)

![5c28060c1c6aea2580bb158c7137caa18861fe1abe706134148e1194c3f0b49f.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/5c28060c1c6aea2580bb158c7137caa18861fe1abe706134148e1194c3f0b49f.jpg)

![892b06f07b15a4f1afc1c5261615a50cb7657c5b5f387f60fad575239fe86054.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/892b06f07b15a4f1afc1c5261615a50cb7657c5b5f387f60fad575239fe86054.jpg)

![905260c12173c3edb6e245eea9fabae66763142affafa65c0a30344ccf085013.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/905260c12173c3edb6e245eea9fabae66763142affafa65c0a30344ccf085013.jpg)

![979b6bf48c02b73f8f975c4180052636f3160eede87ae270ca56466c92352d7c.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/979b6bf48c02b73f8f975c4180052636f3160eede87ae270ca56466c92352d7c.jpg)

![b3fefa9c5cc2254f41a6b2fb2a79178d277e8a0df49f82a37cb1cfeca42f7af7.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/images/b3fefa9c5cc2254f41a6b2fb2a79178d277e8a0df49f82a37cb1cfeca42f7af7.jpg)

### Tables

![187a7b60c01bd5972e16ac373b0fee325d4f595a70411d4b5aa7815e46e7db41.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/187a7b60c01bd5972e16ac373b0fee325d4f595a70411d4b5aa7815e46e7db41.jpg)

![261d4bd0ba94ee4d2aeb4d18549c6e4556d04a8490539efb45877fb20e283cd1.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/261d4bd0ba94ee4d2aeb4d18549c6e4556d04a8490539efb45877fb20e283cd1.jpg)

![2dc73116d0ab58b29904b1387c7fbf4b3c1499eeece2f654de8e421576a1a7c6.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/2dc73116d0ab58b29904b1387c7fbf4b3c1499eeece2f654de8e421576a1a7c6.jpg)

![49e37105d6f4605a2e2a2cf539a68752c491eab000e2bbeae5fb7d197f92c537.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/49e37105d6f4605a2e2a2cf539a68752c491eab000e2bbeae5fb7d197f92c537.jpg)

![7d1e5ba85183e32c1705bdf9a47aef8407e737f9d202ae6843a47f8c70410642.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/7d1e5ba85183e32c1705bdf9a47aef8407e737f9d202ae6843a47f8c70410642.jpg)

![adcb1403bf8687d962b3984c94e7ba3fab6f55a7ce27a12453db6a9edf1bdfec.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/adcb1403bf8687d962b3984c94e7ba3fab6f55a7ce27a12453db6a9edf1bdfec.jpg)

![da9b331ad8a9784480e2dd2fbd0ddb95fea9588833c4bd11dc8ad1329a8d2315.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/da9b331ad8a9784480e2dd2fbd0ddb95fea9588833c4bd11dc8ad1329a8d2315.jpg)

![e77344f7a2e30ca89c7c922d4d9d174af48153ceb477feb584caae0dcc8e2a8e.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/e77344f7a2e30ca89c7c922d4d9d174af48153ceb477feb584caae0dcc8e2a8e.jpg)

![f10847b2f1497cc4b61a6cf540e4c408b52226fef6d76285a22fa9b85512568e.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/f10847b2f1497cc4b61a6cf540e4c408b52226fef6d76285a22fa9b85512568e.jpg)

![fc922cbb8b7dc92ba5a43b0f1fb49ac2c4b4edca205b4fe53122e73dcd0dd761.jpg](../icml_results/2814_Tackling%20Length%20Inflation%20Without%20Trade-offs_%20Group%20Relative%20Reward%20Rescaling%20for%20Reinforcement%20Lear/tables/fc922cbb8b7dc92ba5a43b0f1fb49ac2c4b4edca205b4fe53122e73dcd0dd761.jpg)

<a id="paper-2779"></a>

## How Can Mamba Learn In Context with Outliers and Generalize Provably?


### Images

![3580c84eba822f1e3b5ff5bc18a8477345be1048515cd0555399746e8f241327.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/images/3580c84eba822f1e3b5ff5bc18a8477345be1048515cd0555399746e8f241327.jpg)

![44a46218e54bb135ee0b7644a426586f3d6c12d2347e39ef062feaa1cecd344b.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/images/44a46218e54bb135ee0b7644a426586f3d6c12d2347e39ef062feaa1cecd344b.jpg)

![71a07f3b22d2cc000702bcd6d135ffb7e1d3210f1845df930dce3494e2673479.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/images/71a07f3b22d2cc000702bcd6d135ffb7e1d3210f1845df930dce3494e2673479.jpg)

![7bd49ffdc731b9d78ffa6a4b881bfb17f84bd6476b1fc0e3ded342dd3a5c6cbb.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/images/7bd49ffdc731b9d78ffa6a4b881bfb17f84bd6476b1fc0e3ded342dd3a5c6cbb.jpg)

![99a6c418b3b9df56b477c375b7ceb6789450ffd2de16328280aae35d9c3206c2.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/images/99a6c418b3b9df56b477c375b7ceb6789450ffd2de16328280aae35d9c3206c2.jpg)

### Tables

![11e3a82b40af85316f256e6a9621a34334a40870286784bd2bab11ceabd3eedd.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/11e3a82b40af85316f256e6a9621a34334a40870286784bd2bab11ceabd3eedd.jpg)

![2a67a7609418389da98b4ceca0acf310bb583a93b0137a964680d6240a33cbad.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/2a67a7609418389da98b4ceca0acf310bb583a93b0137a964680d6240a33cbad.jpg)

![38134c83874e8f7b072a4a2ef783bd4ce887d4be73dc24efea43905d8e28f5a4.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/38134c83874e8f7b072a4a2ef783bd4ce887d4be73dc24efea43905d8e28f5a4.jpg)

![6eef6e989d940722f3a380b3f5035840bd8079b1091cc96f657e2b8c495506ee.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/6eef6e989d940722f3a380b3f5035840bd8079b1091cc96f657e2b8c495506ee.jpg)

![8c9e276ff6dc8c1c2fd48accd0b8a769cafbe4aca6c04a2335aac53785ac87ed.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/8c9e276ff6dc8c1c2fd48accd0b8a769cafbe4aca6c04a2335aac53785ac87ed.jpg)

![eef09ce4a7e098c2a7161a41a3df8e085c32ab6b7e071ee603b378c104e9de85.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/eef09ce4a7e098c2a7161a41a3df8e085c32ab6b7e071ee603b378c104e9de85.jpg)

![f33e24240d0d80036cdc5b59f16bc156710d951c03380f922d741e1e8df35ff8.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/f33e24240d0d80036cdc5b59f16bc156710d951c03380f922d741e1e8df35ff8.jpg)

![fce672ceaafc5dfc7f26ac830c232313fda84dbbe6948288a1c146ed1dad1d20.jpg](../icml_results/2815_How%20Can%20Mamba%20Learn%20In%20Context%20with%20Outliers%20and%20Generalize%20Provably_/tables/fce672ceaafc5dfc7f26ac830c232313fda84dbbe6948288a1c146ed1dad1d20.jpg)

<a id="paper-2780"></a>

## Constructing Industrial-Scale Optimization Modeling Benchmark


### Images

![024b5c8b595eb384b8ef8a3a83afe11900a027a84f65b9107c64356990911781.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/024b5c8b595eb384b8ef8a3a83afe11900a027a84f65b9107c64356990911781.jpg)

![0a59c65d5a60a1cfc0a3bc005b558e28ded712ee207e96138f1c9a2a1a4b2c87.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/0a59c65d5a60a1cfc0a3bc005b558e28ded712ee207e96138f1c9a2a1a4b2c87.jpg)

![1179f7dec1ac9cf7d0b48269f2123488f155a523b81367b5502da87f26286c43.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/1179f7dec1ac9cf7d0b48269f2123488f155a523b81367b5502da87f26286c43.jpg)

![1ae12d8fb73a5eb493b1a0e52c846c10e30d2007c5e978de0f9ea8a19eeea0c6.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/1ae12d8fb73a5eb493b1a0e52c846c10e30d2007c5e978de0f9ea8a19eeea0c6.jpg)

![1b890e7af87487407324b1b78ab19914f91b8bfa78539ae4542be7a35ac211d6.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/1b890e7af87487407324b1b78ab19914f91b8bfa78539ae4542be7a35ac211d6.jpg)

![267c3bdc50533000c4f95ec46f13b90638af173d5891b7d02f0753250219eff4.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/267c3bdc50533000c4f95ec46f13b90638af173d5891b7d02f0753250219eff4.jpg)

![2752be57a76f3054825b42e9961e4568b5a30198dd7ca2cb08a6b6cf0e128dda.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/2752be57a76f3054825b42e9961e4568b5a30198dd7ca2cb08a6b6cf0e128dda.jpg)

![2ec7aa13f0d0678596b9eee5e2c2078798fea00c63afc9f897173a1539fcf15b.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/2ec7aa13f0d0678596b9eee5e2c2078798fea00c63afc9f897173a1539fcf15b.jpg)

![3397cb0e1ec6484ef2a5801026839176b469fcd5e2eb08a1d006d4bdbe3518bf.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/3397cb0e1ec6484ef2a5801026839176b469fcd5e2eb08a1d006d4bdbe3518bf.jpg)

![4f11019bece935ad62bad8eba3c5fbb621d1224ab1dace0a8b891f64ac487d2c.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/4f11019bece935ad62bad8eba3c5fbb621d1224ab1dace0a8b891f64ac487d2c.jpg)

![6047ec695535bc98ab2140946e705d778358f3b930dad66156900205a8043e33.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/6047ec695535bc98ab2140946e705d778358f3b930dad66156900205a8043e33.jpg)

![7f5c140440927b1b1890693fdf6ccf9d2a7c568ddd279180546cc8c1b8677723.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/7f5c140440927b1b1890693fdf6ccf9d2a7c568ddd279180546cc8c1b8677723.jpg)

![8a146ddeff06c261f39a8f004fe0df8f37b608b34065654ac92bb5f7361eb840.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/8a146ddeff06c261f39a8f004fe0df8f37b608b34065654ac92bb5f7361eb840.jpg)

![8ad1a437e15e15e76d9b3742e4aaee444be2beb4433729c1bb48013b7cfa6c35.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/8ad1a437e15e15e76d9b3742e4aaee444be2beb4433729c1bb48013b7cfa6c35.jpg)

![8f8d9784530663e50fce62ee0d295cb214ec1ea8405be0b7f8ae479710a99c32.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/8f8d9784530663e50fce62ee0d295cb214ec1ea8405be0b7f8ae479710a99c32.jpg)

![9aa0c7655fd5a66ae0ec7156c6e98bf74188a725e154563e870b9287f910de59.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/9aa0c7655fd5a66ae0ec7156c6e98bf74188a725e154563e870b9287f910de59.jpg)

![b0294f9e12f702e47d75a2983e210a7d1f97e4d88ce92479e05096599e1ac0ad.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/b0294f9e12f702e47d75a2983e210a7d1f97e4d88ce92479e05096599e1ac0ad.jpg)

![c3880dc342d1960ce5b78234cd4cd9a71959e6ac3ba59b9c4b86064ed3bee990.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/c3880dc342d1960ce5b78234cd4cd9a71959e6ac3ba59b9c4b86064ed3bee990.jpg)

![d309a0dc4580691df6c21442f06e72f238f3321f2d97a54da257219d24064231.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/d309a0dc4580691df6c21442f06e72f238f3321f2d97a54da257219d24064231.jpg)

![d58244c820d11af12be19699cb4a936ddfe61f854d6b6e8b47cdb87644b87aee.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/d58244c820d11af12be19699cb4a936ddfe61f854d6b6e8b47cdb87644b87aee.jpg)

![f7ac1afa5122e55f55dc55812196f1d34592bf03aee00a8483fc16c8a925176c.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/images/f7ac1afa5122e55f55dc55812196f1d34592bf03aee00a8483fc16c8a925176c.jpg)

### Tables

![09c3b03f2cf69403f7e7f128fd41492a4d7f836310010040f719d21154cfe37d.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/09c3b03f2cf69403f7e7f128fd41492a4d7f836310010040f719d21154cfe37d.jpg)

![103f22d39c098094473c11dcc24b17f95d90c0b74ca2a26c999faf8c7c98181e.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/103f22d39c098094473c11dcc24b17f95d90c0b74ca2a26c999faf8c7c98181e.jpg)

![13a6d1925c7cb0373e013e4b03badcbac7c52b3be8618b90b50a8579bdd09ff7.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/13a6d1925c7cb0373e013e4b03badcbac7c52b3be8618b90b50a8579bdd09ff7.jpg)

![731dd01cdec8c7811aa14850f1bed8320ecb7b8aeb900712b2102e59cdc6497a.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/731dd01cdec8c7811aa14850f1bed8320ecb7b8aeb900712b2102e59cdc6497a.jpg)

![8425c22662175c65cb6090f640eae9bd65e1cbff2e6676e1441d16e592d49ab2.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/8425c22662175c65cb6090f640eae9bd65e1cbff2e6676e1441d16e592d49ab2.jpg)

![8567f4249226e0b10c689ac5ad5f33e80e41cb0e62424f685e757cfcb3451945.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/8567f4249226e0b10c689ac5ad5f33e80e41cb0e62424f685e757cfcb3451945.jpg)

![d6765ff3159f7eb94612a6966240d85eccb662fb748e70e4e01d93105995b9d6.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/d6765ff3159f7eb94612a6966240d85eccb662fb748e70e4e01d93105995b9d6.jpg)

![dbadea2fb6888f8848f80a0a21edac9b783346693b19db4863f8c8dfa7a7c579.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/dbadea2fb6888f8848f80a0a21edac9b783346693b19db4863f8c8dfa7a7c579.jpg)

![e3e2789b96c25a8ab12689085e14d1474e93a15a930431d4e7ee1a03ac3dd79b.jpg](../icml_results/2817_Constructing%20Industrial-Scale%20Optimization%20Modeling%20Benchmark/tables/e3e2789b96c25a8ab12689085e14d1474e93a15a930431d4e7ee1a03ac3dd79b.jpg)

<a id="paper-2781"></a>

## GeoDM: Geometry-aware Distribution Matching for Dataset Distillation


### Images

![2249dfc0e385ae2ce2b3a554de06365923de6bfda51a6da04643027114898119.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/images/2249dfc0e385ae2ce2b3a554de06365923de6bfda51a6da04643027114898119.jpg)

![237c371bed002d7903a0ef2cee13dc698c6743a833e5322c008dc0706fa92633.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/images/237c371bed002d7903a0ef2cee13dc698c6743a833e5322c008dc0706fa92633.jpg)

![dba5086adedbcdc9d1dcaeeb662d471b41bca689fb16bb59e7c9c291a27f017a.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/images/dba5086adedbcdc9d1dcaeeb662d471b41bca689fb16bb59e7c9c291a27f017a.jpg)

![fb0663ea11c54dbb7e828ad3ca056e92a7d8ec5ce2eee66b435e39d26b8ec876.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/images/fb0663ea11c54dbb7e828ad3ca056e92a7d8ec5ce2eee66b435e39d26b8ec876.jpg)

### Tables

![08af21cc6acc2d1ed24fd9310d8473792ea7f7ffaed9e42572280f1fed8f0b00.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/08af21cc6acc2d1ed24fd9310d8473792ea7f7ffaed9e42572280f1fed8f0b00.jpg)

![0f3cb7215324a75038459d2ea7eb3ea8d7e9ec8f93a4075824ac07fab54c7ce4.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/0f3cb7215324a75038459d2ea7eb3ea8d7e9ec8f93a4075824ac07fab54c7ce4.jpg)

![206b0f9ecbc5eb182fd86f845f9423ee014051529e792924d38aff480a425442.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/206b0f9ecbc5eb182fd86f845f9423ee014051529e792924d38aff480a425442.jpg)

![4adc857ae5f3237e942655dfc993b92232692021f9ae4a2afbece6d081b34b3d.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/4adc857ae5f3237e942655dfc993b92232692021f9ae4a2afbece6d081b34b3d.jpg)

![5495ff24e806722eb54e78be6eeeabf494db77b32001227c470a29965c93d26b.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/5495ff24e806722eb54e78be6eeeabf494db77b32001227c470a29965c93d26b.jpg)

![5acc19a7b74eb9cf6c0b34c329cf1adbe5f4d8754d5a784c91c9731d792c1989.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/5acc19a7b74eb9cf6c0b34c329cf1adbe5f4d8754d5a784c91c9731d792c1989.jpg)

![74d95a9b012721fef36f623aefc5d6001543db0e687ca3a6480a6c991f88292c.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/74d95a9b012721fef36f623aefc5d6001543db0e687ca3a6480a6c991f88292c.jpg)

![7930c3f3f59f4dd022bed98a03e24cc3b3977a2b9f9928ecbc211ff7efd6a0c4.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/7930c3f3f59f4dd022bed98a03e24cc3b3977a2b9f9928ecbc211ff7efd6a0c4.jpg)

![b1f948541aeff750dfa6862970f090f5c316b537e61bb0e8ddeb867b0674cdfa.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/b1f948541aeff750dfa6862970f090f5c316b537e61bb0e8ddeb867b0674cdfa.jpg)

![bfe7ed58141a1553b3b598cd8d0bb79ab7860f945cd67048355eee67590bd050.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/bfe7ed58141a1553b3b598cd8d0bb79ab7860f945cd67048355eee67590bd050.jpg)

![c9bce4d3360b34ff09cb8f83850b0120843bd44443d8a5e48b1add1b28fd8e74.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/c9bce4d3360b34ff09cb8f83850b0120843bd44443d8a5e48b1add1b28fd8e74.jpg)

![cdd10c1830603198efb6e1b4228f15b8880a950a61d67c4aed1758aff33570b2.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/cdd10c1830603198efb6e1b4228f15b8880a950a61d67c4aed1758aff33570b2.jpg)

![d5e7a8bca9a1f924869ede09587bd0187897f6bd08ab2db04776c61cfe46db7a.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/d5e7a8bca9a1f924869ede09587bd0187897f6bd08ab2db04776c61cfe46db7a.jpg)

![da6877373d3f3e1b30f6110963150d449c55216189067d6a1437bf55723e1dc6.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/da6877373d3f3e1b30f6110963150d449c55216189067d6a1437bf55723e1dc6.jpg)

![e18b811af843a6da96e791a60640cc326459b8d9edb8a5252a1e880d294be3e3.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/e18b811af843a6da96e791a60640cc326459b8d9edb8a5252a1e880d294be3e3.jpg)

![ee4b02ca8766d9b9c7cea96130ec24c41ef19b22610b3d9df068896766fb56b6.jpg](../icml_results/2818_GeoDM_%20Geometry-aware%20Distribution%20Matching%20for%20Dataset%20Distillation/tables/ee4b02ca8766d9b9c7cea96130ec24c41ef19b22610b3d9df068896766fb56b6.jpg)

<a id="paper-2782"></a>

## Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models


### Images

![3de5412e0393bbbe092790f0df00d92d2c435e0eceb75f63ed719abef28532d0.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/images/3de5412e0393bbbe092790f0df00d92d2c435e0eceb75f63ed719abef28532d0.jpg)

![8a220a0d7e60a681508a5cf6ab52aa6e61f6fece7717dbb01d8642f22ba7f2b0.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/images/8a220a0d7e60a681508a5cf6ab52aa6e61f6fece7717dbb01d8642f22ba7f2b0.jpg)

![b8d51a63325642d9b00ae4c17c14b1644f6c46bce6534ee1710683cdd614a70a.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/images/b8d51a63325642d9b00ae4c17c14b1644f6c46bce6534ee1710683cdd614a70a.jpg)

![ba564a71b611cf86fb2312973611b27a4f58275401fcdd7122860c6645291837.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/images/ba564a71b611cf86fb2312973611b27a4f58275401fcdd7122860c6645291837.jpg)

![e8e63610fed3c297da0e6370387fcce3ceef633fd75335d6430ab29390291d05.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/images/e8e63610fed3c297da0e6370387fcce3ceef633fd75335d6430ab29390291d05.jpg)

### Tables

![06098133834cc74a137f99f5c1831fe5eccb6bad1128c581d29ed757b7b81c06.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/tables/06098133834cc74a137f99f5c1831fe5eccb6bad1128c581d29ed757b7b81c06.jpg)

![a23d3025b581245824813293a6d57a85df3ba191d361166a765a8ada71bd1aad.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/tables/a23d3025b581245824813293a6d57a85df3ba191d361166a765a8ada71bd1aad.jpg)

![cc2ea4010403d9c401f6d466b16158ed7069b86caba2553fa85187ab2ca85dd5.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/tables/cc2ea4010403d9c401f6d466b16158ed7069b86caba2553fa85187ab2ca85dd5.jpg)

![d3c96f8baee0aff9746a62aca744b6c70c7f74e60a4e8e28531bc6dd9c0976b6.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/tables/d3c96f8baee0aff9746a62aca744b6c70c7f74e60a4e8e28531bc6dd9c0976b6.jpg)

![e81b146caf547ac6f9d681e6543c22e90258bdfb5072a135f4a171590ecf0fd3.jpg](../icml_results/2819_Gradients%20with%20Respect%20to%20Semantics%20Preserving%20Embeddings%20Tell%20the%20Uncertainty%20of%20Large%20Language%20Mod/tables/e81b146caf547ac6f9d681e6543c22e90258bdfb5072a135f4a171590ecf0fd3.jpg)

<a id="paper-2783"></a>

## Predicting Large Model Test Losses with a Noisy Quadratic System


### Images

![4e0c73aca79e6a15eeb0731fdfdb9d0d846ab6ff0176f645e7d8cd93d6ceb3ae.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/4e0c73aca79e6a15eeb0731fdfdb9d0d846ab6ff0176f645e7d8cd93d6ceb3ae.jpg)

![93261b808dbece1545085c60e9ab1e2e49c7ea3c2a557851fb2805ca4b7a0fb1.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/93261b808dbece1545085c60e9ab1e2e49c7ea3c2a557851fb2805ca4b7a0fb1.jpg)

![96062341d907e3555fe31546a3f8b1601fe661da0eb8a58c72b29a548cffa0cd.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/96062341d907e3555fe31546a3f8b1601fe661da0eb8a58c72b29a548cffa0cd.jpg)

![b249855f233afdf0b1726f5fd7c11d2346231203cbd1c98232f96803c432b620.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/b249855f233afdf0b1726f5fd7c11d2346231203cbd1c98232f96803c432b620.jpg)

![b55740dfcb33f339882bab77472fff6b6c9646ae9f1bce2e5f443df156a0615f.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/b55740dfcb33f339882bab77472fff6b6c9646ae9f1bce2e5f443df156a0615f.jpg)

![be2b3373162e5b6ecd32144597c1082a19cb159829f80599c58ec72ce806d9f5.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/be2b3373162e5b6ecd32144597c1082a19cb159829f80599c58ec72ce806d9f5.jpg)

![ec7a8a316f4599e82b4a512f0c47a23aa8b742ed45d4ad379dd2108ed5c4ff1e.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/ec7a8a316f4599e82b4a512f0c47a23aa8b742ed45d4ad379dd2108ed5c4ff1e.jpg)

![f53e00891f76b9a365af61ea91aba197f002adef03708c634d4ef36842f2754c.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/f53e00891f76b9a365af61ea91aba197f002adef03708c634d4ef36842f2754c.jpg)

![fc87795539898eafed6b971675cc134ba118fd72044f89f5bb458e9534aa467d.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/images/fc87795539898eafed6b971675cc134ba118fd72044f89f5bb458e9534aa467d.jpg)

### Tables

![59cab3086321673f219545c0da1986a70fc17f27a1ddb584d434ce748bcbde0d.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/tables/59cab3086321673f219545c0da1986a70fc17f27a1ddb584d434ce748bcbde0d.jpg)

![6299b9a337fa7c001a0bf020fb7328b66f13bc44a44a7d6dcbf37e224f536901.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/tables/6299b9a337fa7c001a0bf020fb7328b66f13bc44a44a7d6dcbf37e224f536901.jpg)

![7d85ca8e364cdcd9c5a1129c69ef8e49014275d80049d20ffa4d9915949e24fd.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/tables/7d85ca8e364cdcd9c5a1129c69ef8e49014275d80049d20ffa4d9915949e24fd.jpg)

![c3a9f37d799486c96b9c42ad676ba4c00ad2ed127558bb97342b2d09d43c1d83.jpg](../icml_results/2820_Predicting%20Large%20Model%20Test%20Losses%20with%20a%20Noisy%20Quadratic%20System/tables/c3a9f37d799486c96b9c42ad676ba4c00ad2ed127558bb97342b2d09d43c1d83.jpg)

<a id="paper-2784"></a>

## Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning


### Images

![00076201e46acbf5893ec17577cc448d66bef2d414e2e3e865206ea7811f3c6c.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/images/00076201e46acbf5893ec17577cc448d66bef2d414e2e3e865206ea7811f3c6c.jpg)

![808201028090b55d08c645809d5b59c9df6d1a509c2e988b77ff0cc97614b937.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/images/808201028090b55d08c645809d5b59c9df6d1a509c2e988b77ff0cc97614b937.jpg)

![bc9036202098fed337f776fca93d1df2549408499dbf121e4d48a79b1e3047e7.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/images/bc9036202098fed337f776fca93d1df2549408499dbf121e4d48a79b1e3047e7.jpg)

### Tables

![647240a4d2b1d2235894a04eec94db88424674e9963373c70c83f7c833c7cf78.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/647240a4d2b1d2235894a04eec94db88424674e9963373c70c83f7c833c7cf78.jpg)

![725af8a71e988cdf19e0428180ba223b9ab4638efd508111760d1e66f3537e2e.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/725af8a71e988cdf19e0428180ba223b9ab4638efd508111760d1e66f3537e2e.jpg)

![7e5a97c7ef1a7a49f0caca7d4f32633b558361990c6a7181b834630131dbc7b4.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/7e5a97c7ef1a7a49f0caca7d4f32633b558361990c6a7181b834630131dbc7b4.jpg)

![886529bb339e07fc8713b724962724e97f570b2c55b9333135cee03634f3d213.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/886529bb339e07fc8713b724962724e97f570b2c55b9333135cee03634f3d213.jpg)

![a1014e468e38df08cf00c5c38beeb518019e5e003783eed5142e233a13cafd1a.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/a1014e468e38df08cf00c5c38beeb518019e5e003783eed5142e233a13cafd1a.jpg)

![ce0090c2515fd6889f9690857ec46a40e2a5c841fa7cef71c61a72b477ef54c8.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/ce0090c2515fd6889f9690857ec46a40e2a5c841fa7cef71c61a72b477ef54c8.jpg)

![ce117ec0ae5abda3b442a1c2390403b57b11ff49a6fa8b1288ad85e02281400a.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/ce117ec0ae5abda3b442a1c2390403b57b11ff49a6fa8b1288ad85e02281400a.jpg)

![e79edbcdf4e3e8e428c4fe07435a75849dffd9c178d0b8d4d527cae48ffbf5ec.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/e79edbcdf4e3e8e428c4fe07435a75849dffd9c178d0b8d4d527cae48ffbf5ec.jpg)

![ef64e2001ab8cdbf048295a930ce05814198a7e744546d75a47a6ea04a66bc86.jpg](../icml_results/2821_Vulnerable%20Agent%20Identification%20in%20Large-Scale%20Multi-Agent%20Reinforcement%20Learning/tables/ef64e2001ab8cdbf048295a930ce05814198a7e744546d75a47a6ea04a66bc86.jpg)

<a id="paper-2785"></a>

## A Minimax Approach for Optimal Intervention Policy Learning with Two-Stage Outcomes


### Images

![54c086db66d7eb5e6e7669b39afc11f8aa8d45ebd743829ecda91074276fb03b.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/54c086db66d7eb5e6e7669b39afc11f8aa8d45ebd743829ecda91074276fb03b.jpg)

![6c430b74d971e1341365f2b84018c4cb550e69df942c696d7526233515e97e30.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/6c430b74d971e1341365f2b84018c4cb550e69df942c696d7526233515e97e30.jpg)

![96e9a4b4390d63558982c1cbe161e28de76974b82a75a67c1a2fbfe16cf36142.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/96e9a4b4390d63558982c1cbe161e28de76974b82a75a67c1a2fbfe16cf36142.jpg)

![ab61fce2e8f978062a862225f3dc57b6bf2a66747e479365d02ec2259a51db3a.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/ab61fce2e8f978062a862225f3dc57b6bf2a66747e479365d02ec2259a51db3a.jpg)

![babe133308dd554908551b14a5a16df122b0a23f4c691dbbb62c2920b4fd9959.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/babe133308dd554908551b14a5a16df122b0a23f4c691dbbb62c2920b4fd9959.jpg)

![bd4712f6a2bff5e737eae1b09ae27c101a88e959e0161ba66fc98dec164a0569.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/bd4712f6a2bff5e737eae1b09ae27c101a88e959e0161ba66fc98dec164a0569.jpg)

![be8956e8c67f8e7f549d0cbf23662073a5a156eeb7df188821603bf2b043476f.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/be8956e8c67f8e7f549d0cbf23662073a5a156eeb7df188821603bf2b043476f.jpg)

![edf4a72b54b2e7fd4d05f1ac3d0cd6549f586c61ccf1a8722b2d8ce662b2ee2c.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/images/edf4a72b54b2e7fd4d05f1ac3d0cd6549f586c61ccf1a8722b2d8ce662b2ee2c.jpg)

### Tables

![1723741c57a5d9dd10aea40c16ec574cbd47bfcdead7d37aee16e40ea00d047f.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/1723741c57a5d9dd10aea40c16ec574cbd47bfcdead7d37aee16e40ea00d047f.jpg)

![1ce128e863ab5ea9f31867d081ab1471c343aafa159df516e715827c6b1a7f7d.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/1ce128e863ab5ea9f31867d081ab1471c343aafa159df516e715827c6b1a7f7d.jpg)

![30883cc0fb0b01b79377bb8f94968d6300de5df6531c59d5da004286fc68e762.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/30883cc0fb0b01b79377bb8f94968d6300de5df6531c59d5da004286fc68e762.jpg)

![31ab572830e44c49b178d61748e44898e0c84530e7da6632ad2b66d9931ab9a2.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/31ab572830e44c49b178d61748e44898e0c84530e7da6632ad2b66d9931ab9a2.jpg)

![31c616206f2fad675a316e69d7dc8f38eb181190f1b0ce835eda3370e6568c5d.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/31c616206f2fad675a316e69d7dc8f38eb181190f1b0ce835eda3370e6568c5d.jpg)

![3ebf13206695aa0e85239e31b6f614d20ede175aabdafa169de774f0a8603386.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/3ebf13206695aa0e85239e31b6f614d20ede175aabdafa169de774f0a8603386.jpg)

![7f4b506970991d485a89375212cde1fbcb0839f86b217bfae076205188fbf337.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/7f4b506970991d485a89375212cde1fbcb0839f86b217bfae076205188fbf337.jpg)

![8181107ad56f92ab48fc34dff321e63fa37780987719ab214698f415bedfd58c.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/8181107ad56f92ab48fc34dff321e63fa37780987719ab214698f415bedfd58c.jpg)

![adf9409ff79eb0090c1297a11f0d2d7d492c3ac22afa78d701d8d174f87d8f35.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/adf9409ff79eb0090c1297a11f0d2d7d492c3ac22afa78d701d8d174f87d8f35.jpg)

![df8f3287c910a1b2b7d5bfa28fdac7ae6128f41b8fa26fa2d41e950bcc83a644.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/df8f3287c910a1b2b7d5bfa28fdac7ae6128f41b8fa26fa2d41e950bcc83a644.jpg)

![e22a06739b4df4aa25e45c2955df650a3061a2fe6c8c6d58a38f2a68826f112a.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/e22a06739b4df4aa25e45c2955df650a3061a2fe6c8c6d58a38f2a68826f112a.jpg)

![f46671993e4318a7397bebbc33a2015dd6e4defbfde50847b33ec076e3cbaac7.jpg](../icml_results/2822_A%20Minimax%20Approach%20for%20Optimal%20Intervention%20Policy%20Learning%20with%20Two-Stage%20Outcomes/tables/f46671993e4318a7397bebbc33a2015dd6e4defbfde50847b33ec076e3cbaac7.jpg)

<a id="paper-2786"></a>

## Transformers Learn the Optimal DDPM Denoiser for Multi-Token GMMs


### Images

![22068784614656289431a1cb0a1e693d1bba02da2557676d0d10aa5b60a073aa.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/images/22068784614656289431a1cb0a1e693d1bba02da2557676d0d10aa5b60a073aa.jpg)

![3613919ec700b69b36196a28a41e743181ceb8a709766467463055dfba48fd7b.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/images/3613919ec700b69b36196a28a41e743181ceb8a709766467463055dfba48fd7b.jpg)

![3f559125df770ab84add8c1f813a860eb0ee4de809a433297626ac77224f066f.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/images/3f559125df770ab84add8c1f813a860eb0ee4de809a433297626ac77224f066f.jpg)

![c721611961c136c6a2f09df9df0a69f2931ac9192a772b27ff3b5b933a3b8082.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/images/c721611961c136c6a2f09df9df0a69f2931ac9192a772b27ff3b5b933a3b8082.jpg)

![d2272d84f299b2b4548a9c3b5fa3e638c035943aa4a1ad950a5f70d8e2742e34.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/images/d2272d84f299b2b4548a9c3b5fa3e638c035943aa4a1ad950a5f70d8e2742e34.jpg)

### Tables

![a7d5f98a4c924ee73513fea7a15f207754e17921c75f429a2bfedc6e47b9a01b.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/tables/a7d5f98a4c924ee73513fea7a15f207754e17921c75f429a2bfedc6e47b9a01b.jpg)

![b538a34eb83a7cab52fe405b78f0f6547a37e98a2e850094c1637566f0487723.jpg](../icml_results/2823_Transformers%20Learn%20the%20Optimal%20DDPM%20Denoiser%20for%20Multi-Token%20GMMs/tables/b538a34eb83a7cab52fe405b78f0f6547a37e98a2e850094c1637566f0487723.jpg)

<a id="paper-2787"></a>

## CryoACE: An Atom-centric Framework for Accurate and Automated Model Building in Cryo-EM


### Images

![262063a9de71cf7c6b5e8b282f5be9c49101c3cea91a601baa6d43042089f3a1.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/262063a9de71cf7c6b5e8b282f5be9c49101c3cea91a601baa6d43042089f3a1.jpg)

![4023e7c8c56680e5da440db97e7ef720d5fb7266da9e8251f27fdebc67919dfa.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/4023e7c8c56680e5da440db97e7ef720d5fb7266da9e8251f27fdebc67919dfa.jpg)

![a1cf34096a9212106c78ab7abafe33e253d63c2b9ec23f25bcc942dbc4d129ec.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/a1cf34096a9212106c78ab7abafe33e253d63c2b9ec23f25bcc942dbc4d129ec.jpg)

![c9008244f51356472708835a2707a64fae0d3b478dbdae14db1d45d8820a0b1c.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/c9008244f51356472708835a2707a64fae0d3b478dbdae14db1d45d8820a0b1c.jpg)

![ed65896058d0f29845cf9094afb66ac9ebba06112609cb97d8ac115bc9ee49ef.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/ed65896058d0f29845cf9094afb66ac9ebba06112609cb97d8ac115bc9ee49ef.jpg)

![f7d1d48948c59182d59db8113906f22874b1aad6e1cb0c01bd9c8170a23d566f.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/images/f7d1d48948c59182d59db8113906f22874b1aad6e1cb0c01bd9c8170a23d566f.jpg)

### Tables

![0777a081ff66a61d29140499b6a19633106071fc110a46842c33ce177ad0a529.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/0777a081ff66a61d29140499b6a19633106071fc110a46842c33ce177ad0a529.jpg)

![0b30e3a8f5e8c6a3363d5b4141069ff0c2510684d629163d358fc2498d9ccf60.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/0b30e3a8f5e8c6a3363d5b4141069ff0c2510684d629163d358fc2498d9ccf60.jpg)

![1118fe911fc0c074a502a5d3bf6c23e01bdd0eeb856e76a723b662686f26d98a.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/1118fe911fc0c074a502a5d3bf6c23e01bdd0eeb856e76a723b662686f26d98a.jpg)

![1e5a71a7540232e57e620110f978d3c4d4608946577244bbe6734f9561d7dd0d.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/1e5a71a7540232e57e620110f978d3c4d4608946577244bbe6734f9561d7dd0d.jpg)

![5e2779d02e8a4947cea9d140bdf52360c055d55f625c4ee20be47f077273a7cd.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/5e2779d02e8a4947cea9d140bdf52360c055d55f625c4ee20be47f077273a7cd.jpg)

![70f411daa6f2d607ab7b3de59b19bef59b9069450c955b88f95eff97ab412255.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/70f411daa6f2d607ab7b3de59b19bef59b9069450c955b88f95eff97ab412255.jpg)

![dc242cc95c37da43d7ebe851d86d63ce433df7016f30c7d1e1aa3019bae465a4.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/dc242cc95c37da43d7ebe851d86d63ce433df7016f30c7d1e1aa3019bae465a4.jpg)

![f62206d93aa7c5fff0d45636557f34e9617097c400c637325dc96d36ae095a20.jpg](../icml_results/2824_CryoACE_%20An%20Atom-centric%20Framework%20for%20Accurate%20and%20Automated%20Model%20Building%20in%20Cryo-EM/tables/f62206d93aa7c5fff0d45636557f34e9617097c400c637325dc96d36ae095a20.jpg)

<a id="paper-2788"></a>

## OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents


### Images

![29adc72933e2ffe40dd89a8c56abbc5fcbd4a3c9f3ed148ff3f061bdb618880e.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/images/29adc72933e2ffe40dd89a8c56abbc5fcbd4a3c9f3ed148ff3f061bdb618880e.jpg)

![749b44d8456adcc6550c65fcf34b11883aa51616d29f2160e36d82d55e7162d7.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/images/749b44d8456adcc6550c65fcf34b11883aa51616d29f2160e36d82d55e7162d7.jpg)

### Tables

![1cd57bfd5ed8bdcbacc0c09f9a7536b30149844b16e0836e871ceef36cb6a816.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/1cd57bfd5ed8bdcbacc0c09f9a7536b30149844b16e0836e871ceef36cb6a816.jpg)

![2b82c954607ecefe31fcd448450e7da36e2ab47893715081502434abff8c97f6.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/2b82c954607ecefe31fcd448450e7da36e2ab47893715081502434abff8c97f6.jpg)

![2c350f190ccd600c8ce97ab5eb768a4991f11c25e49dcb61ea896164fdd9f88e.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/2c350f190ccd600c8ce97ab5eb768a4991f11c25e49dcb61ea896164fdd9f88e.jpg)

![402fa3e2e43eabc44cc167452d4f8c1a632910bba0c7098438460595df7ed1f1.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/402fa3e2e43eabc44cc167452d4f8c1a632910bba0c7098438460595df7ed1f1.jpg)

![43af7d9d501d44a882e9f2379bfc4b27d32da4b7cdef559ffbcd9d979e60a57e.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/43af7d9d501d44a882e9f2379bfc4b27d32da4b7cdef559ffbcd9d979e60a57e.jpg)

![54a832214a8a9e6fd0b65e9dce516698e5424e71b15fd0396d8bde6549f52978.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/54a832214a8a9e6fd0b65e9dce516698e5424e71b15fd0396d8bde6549f52978.jpg)

![7ba8dc79885d1c1ad7495e26c3f0ce0ae0d31cc75dbf23b2bf37696b8729a75d.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/7ba8dc79885d1c1ad7495e26c3f0ce0ae0d31cc75dbf23b2bf37696b8729a75d.jpg)

![8169d1e455cc0ffc7d387297cc2f13a91f13c2a6957f94b1404c344961f19325.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/8169d1e455cc0ffc7d387297cc2f13a91f13c2a6957f94b1404c344961f19325.jpg)

![9342c014bd579ef758d48a2fcf26c498bba7c34fdaece2eadfdb147b52af20c7.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/9342c014bd579ef758d48a2fcf26c498bba7c34fdaece2eadfdb147b52af20c7.jpg)

![9c50c5c354e677179d09b2665d7641b2c149eec9782705b0c6e377b92deb8fe4.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/9c50c5c354e677179d09b2665d7641b2c149eec9782705b0c6e377b92deb8fe4.jpg)

![9e6e5e683917ca4da0bc6f34e2925a48832e9b2e52859dbce0158a104fa57803.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/9e6e5e683917ca4da0bc6f34e2925a48832e9b2e52859dbce0158a104fa57803.jpg)

![a22e53802f8d8178be22106368420ff5f9effdfaca37bb4fcc2129c75967109e.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/a22e53802f8d8178be22106368420ff5f9effdfaca37bb4fcc2129c75967109e.jpg)

![af43f51fb53bf8adde48ca59f37b29db45e3fc123bd7d5db14015c4ad4ae7dc0.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/af43f51fb53bf8adde48ca59f37b29db45e3fc123bd7d5db14015c4ad4ae7dc0.jpg)

![b44cf66828bebcd57dcec6e216ce940ef0e792d668e33896eb764eb9e1652a63.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/b44cf66828bebcd57dcec6e216ce940ef0e792d668e33896eb764eb9e1652a63.jpg)

![b61d15509dcb21bd8fa2a657dbb8adcaafe3e80ceb351c23768601a637b8ac44.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/b61d15509dcb21bd8fa2a657dbb8adcaafe3e80ceb351c23768601a637b8ac44.jpg)

![f056234f62d2e268cb3456e187b9bc6780cab14ae04efb907b64b5694d3c53de.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/f056234f62d2e268cb3456e187b9bc6780cab14ae04efb907b64b5694d3c53de.jpg)

![f0df62aa0aba7789bc9e388d399bbb06c776f0b780995fcecdd37335047fa977.jpg](../icml_results/2825_OTora_%20A%20Unified%20Red%20Teaming%20Framework%20for%20Reasoning-Level%20Denial-of-Service%20in%20LLM%20Agents/tables/f0df62aa0aba7789bc9e388d399bbb06c776f0b780995fcecdd37335047fa977.jpg)

<a id="paper-2789"></a>

## Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs


### Images

![63e3a188c976f308a0cf84f8f05f0d529b7f49cb5d02cd341cae838fbfa83f50.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/images/63e3a188c976f308a0cf84f8f05f0d529b7f49cb5d02cd341cae838fbfa83f50.jpg)

![9d9e8ed579e8a191eedfd91709d9df362ef8ed453efa7ce4fcaa344594521efb.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/images/9d9e8ed579e8a191eedfd91709d9df362ef8ed453efa7ce4fcaa344594521efb.jpg)

![f737ecd63fa80ffb273d49da516799b4f91b35c6929ad3b8def0197de77f5798.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/images/f737ecd63fa80ffb273d49da516799b4f91b35c6929ad3b8def0197de77f5798.jpg)

### Tables

![32e4cd587d0a54ab78bbcd3aa8bf3f14e12372f06fabb5d1a1da16e4d3778d7f.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/32e4cd587d0a54ab78bbcd3aa8bf3f14e12372f06fabb5d1a1da16e4d3778d7f.jpg)

![341fa3c93adddfb156f549880c684207d1a05d2738b52f0910c0fd0423f3ea69.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/341fa3c93adddfb156f549880c684207d1a05d2738b52f0910c0fd0423f3ea69.jpg)

![690fd9b26fb0eb48e9bb7564978891510b8c971c45e34ae17ac51df7ce9a6c87.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/690fd9b26fb0eb48e9bb7564978891510b8c971c45e34ae17ac51df7ce9a6c87.jpg)

![7fabd0ca88cb9c0aa37bbc610afde3ecc766f83f21a8c66d5543dedb55afd8af.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/7fabd0ca88cb9c0aa37bbc610afde3ecc766f83f21a8c66d5543dedb55afd8af.jpg)

![90c54c9594309d161815d3f4ad5dcc0fc5048b394636d323a952e49d2c1753fd.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/90c54c9594309d161815d3f4ad5dcc0fc5048b394636d323a952e49d2c1753fd.jpg)

![a89f6fc0de12fc5c289a3181ffd7142b5134da2fa1bd0f60747a1a1ec2dd49a1.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/a89f6fc0de12fc5c289a3181ffd7142b5134da2fa1bd0f60747a1a1ec2dd49a1.jpg)

![fa18c955da25241456022a37fa6ae6576f45aa271543b678e9f215651b2b3cf0.jpg](../icml_results/2826_Systematic%20Failures%20in%20Collective%20Reasoning%20under%20Distributed%20Information%20in%20Multi-Agent%20LLMs/tables/fa18c955da25241456022a37fa6ae6576f45aa271543b678e9f215651b2b3cf0.jpg)

<a id="paper-2790"></a>

## OptProver: Bridging Olympiad and Optimization through Continual Training in Formal Theorem Proving


### Images

![425028261226901fca49523d86becb3a57549e02e90907851f83fd14b530c828.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/425028261226901fca49523d86becb3a57549e02e90907851f83fd14b530c828.jpg)

![c75abb54280ac76c911b17074a34e25f0545af0834f7c4e6785b8c91fb12248d.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/c75abb54280ac76c911b17074a34e25f0545af0834f7c4e6785b8c91fb12248d.jpg)

![e7676055c5648f0503126f4e53b45153bac07a5e339a674028f72ece84bd74f0.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/e7676055c5648f0503126f4e53b45153bac07a5e339a674028f72ece84bd74f0.jpg)

![ef96e4200c05987926a5c44d56070158e692697bbc8eba80373aa68705e10136.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/ef96e4200c05987926a5c44d56070158e692697bbc8eba80373aa68705e10136.jpg)

![f331b778b2bb12e4d7773b3b7a5dbef9f839dcb7972c5b460ea5f73281958863.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/f331b778b2bb12e4d7773b3b7a5dbef9f839dcb7972c5b460ea5f73281958863.jpg)

![f662d70c7568e15eb1f819735518a95167366642b38b9ff8fdbfb32c2ec7ea0e.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/images/f662d70c7568e15eb1f819735518a95167366642b38b9ff8fdbfb32c2ec7ea0e.jpg)

### Tables

![1bbba32de5073f629a017e8abac27fbe3e303370f7182ec776de63650c537313.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/tables/1bbba32de5073f629a017e8abac27fbe3e303370f7182ec776de63650c537313.jpg)

![98430eed39df34e72784c78da3e29cb715fe13a5df0704a4c3a7254bdb5db021.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/tables/98430eed39df34e72784c78da3e29cb715fe13a5df0704a4c3a7254bdb5db021.jpg)

![9a5ba4f4b24a6f5bdf37e5715e4906eae48e606197726f2181603a7f922cc22a.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/tables/9a5ba4f4b24a6f5bdf37e5715e4906eae48e606197726f2181603a7f922cc22a.jpg)

![a526c594314f2b6efba0eef80a75827ba30d608c949085489c7ec05bfdf4c790.jpg](../icml_results/2827_OptProver_%20Bridging%20Olympiad%20and%20Optimization%20through%20Continual%20Training%20in%20Formal%20Theorem%20Proving/tables/a526c594314f2b6efba0eef80a75827ba30d608c949085489c7ec05bfdf4c790.jpg)

<a id="paper-2791"></a>

## Causal Structure Learning for Sparse Matrix Fill-in Reduction


### Images

![3432ab1682dcbf354cfb4959eb93c144c4a93d8de0b8335cf45a31ce0ab2dbe9.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/3432ab1682dcbf354cfb4959eb93c144c4a93d8de0b8335cf45a31ce0ab2dbe9.jpg)

![36163858f5c350bf67f520341376010b074c90d2b65b99f4a34652a99613a6d8.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/36163858f5c350bf67f520341376010b074c90d2b65b99f4a34652a99613a6d8.jpg)

![845a7bec454a25c09f4aa9ab1d2c12f21223a0d898ad1b364a1dadb013d0638d.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/845a7bec454a25c09f4aa9ab1d2c12f21223a0d898ad1b364a1dadb013d0638d.jpg)

![992451d4b9c421067a3f4118ef6b3c4a6a3541d1556589413b7b4d9a8a752228.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/992451d4b9c421067a3f4118ef6b3c4a6a3541d1556589413b7b4d9a8a752228.jpg)

![ae912f86493ff510b9773278b3fa5ed852fada1a7f9bee61ac58beff8cc4f3ae.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/ae912f86493ff510b9773278b3fa5ed852fada1a7f9bee61ac58beff8cc4f3ae.jpg)

![c534cd03556a55cd29572df043da3ec7db5b0a21d7e6560d6d7e8b97a1a2841d.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/c534cd03556a55cd29572df043da3ec7db5b0a21d7e6560d6d7e8b97a1a2841d.jpg)

![d845938340978176c5c1e86f5985c78a1377c897a8aaac5a2642caa39cbe23d4.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/images/d845938340978176c5c1e86f5985c78a1377c897a8aaac5a2642caa39cbe23d4.jpg)

### Tables

![172f032fbde7af945e3ebd6dd8293416f9a179b26a261c830f4029ad309aabf5.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/tables/172f032fbde7af945e3ebd6dd8293416f9a179b26a261c830f4029ad309aabf5.jpg)

![2a059f6d4ca352796818879a413034715ce3dafd0f1aaf6d738c08090c1f6822.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/tables/2a059f6d4ca352796818879a413034715ce3dafd0f1aaf6d738c08090c1f6822.jpg)

![bb0ab5e83fe084bb666373b8299f5f99a8f2e975871f91c986f6e88870cf36a3.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/tables/bb0ab5e83fe084bb666373b8299f5f99a8f2e975871f91c986f6e88870cf36a3.jpg)

![c23469876b8e3caaf1398e50784284ce1d78def179bb457faa5f114bcf79a17f.jpg](../icml_results/2828_Causal%20Structure%20Learning%20for%20Sparse%20Matrix%20Fill-in%20Reduction/tables/c23469876b8e3caaf1398e50784284ce1d78def179bb457faa5f114bcf79a17f.jpg)

<a id="paper-2792"></a>

## PCGS: Deblurring 3D Gaussian Splatting with Patch Comparison


### Images

![0e99dc4b990eb49df11f4c83d91a3e6e24a3469fc9728f42bfacd34f688f947c.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/0e99dc4b990eb49df11f4c83d91a3e6e24a3469fc9728f42bfacd34f688f947c.jpg)

![4ca859e1dadfecec772879515c610b28df7f7bc28575b0750acfad4a0fb6cef2.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/4ca859e1dadfecec772879515c610b28df7f7bc28575b0750acfad4a0fb6cef2.jpg)

![5d2e194c738cecdfc09168d98d194ce261c4819123a0c2507b886d6a09b8e51c.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/5d2e194c738cecdfc09168d98d194ce261c4819123a0c2507b886d6a09b8e51c.jpg)

![5f4e7a54d9affb93c26e185ec39f33465f9bde4586613a5406212135ab4316df.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/5f4e7a54d9affb93c26e185ec39f33465f9bde4586613a5406212135ab4316df.jpg)

![ad6e22330b1b907ef7affed19e1911d8d8b3611e5db8ccb0b2c131bd9cc6f48d.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/ad6e22330b1b907ef7affed19e1911d8d8b3611e5db8ccb0b2c131bd9cc6f48d.jpg)

![b89218770f577672799689536dcf68b9692a74bd1e5f3a7919e02e07949fc6b6.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/b89218770f577672799689536dcf68b9692a74bd1e5f3a7919e02e07949fc6b6.jpg)

![bcf63eaaa92fc85acc131cbd972cb906c4eeef336a656470835dc7d72e5725b2.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/bcf63eaaa92fc85acc131cbd972cb906c4eeef336a656470835dc7d72e5725b2.jpg)

![bfc04a9f5f6bbcc45ed56f2a1767a66cfa0f9d8d750d93c559600faa007d1cc8.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/bfc04a9f5f6bbcc45ed56f2a1767a66cfa0f9d8d750d93c559600faa007d1cc8.jpg)

![d2039fd905c9784d76cb4876fd0d8b0ae2005472eded092951cada454b5ffa9a.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/d2039fd905c9784d76cb4876fd0d8b0ae2005472eded092951cada454b5ffa9a.jpg)

![dcf8b7f77dbd02e2c7701a10c4c3d249d5c64fffe7744b825c93a91949edd89c.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/images/dcf8b7f77dbd02e2c7701a10c4c3d249d5c64fffe7744b825c93a91949edd89c.jpg)

### Tables

![011e124615e65c357d23ff5a25395d0a5cdc584c06ca70518eef35cf2a996bbb.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/011e124615e65c357d23ff5a25395d0a5cdc584c06ca70518eef35cf2a996bbb.jpg)

![316a2c98b5c42530b89ed31166a29b7042980dcf479c113cb6ece1943d4c0d15.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/316a2c98b5c42530b89ed31166a29b7042980dcf479c113cb6ece1943d4c0d15.jpg)

![41a58f898f8861267340693ad00d62edbe0d21d19d642bca359b644bbb9baf9d.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/41a58f898f8861267340693ad00d62edbe0d21d19d642bca359b644bbb9baf9d.jpg)

![493a549249a14061b36768bac2beaf19c0ddeb95d55afe436aa663cd33b1994c.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/493a549249a14061b36768bac2beaf19c0ddeb95d55afe436aa663cd33b1994c.jpg)

![746a02ff35f3593db0b659960b443af44bd7fecb9d58907983870f311fc7df63.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/746a02ff35f3593db0b659960b443af44bd7fecb9d58907983870f311fc7df63.jpg)

![81d21343f6dd9e6c2af227f42a27aa91c5b7aacba32a5e3a3e12dd85216895fd.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/81d21343f6dd9e6c2af227f42a27aa91c5b7aacba32a5e3a3e12dd85216895fd.jpg)

![88244fc4d30baa229a8ecea96fb462484b7986ce5506b16f2fa2ccec36897b1c.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/88244fc4d30baa229a8ecea96fb462484b7986ce5506b16f2fa2ccec36897b1c.jpg)

![a3696940d2ee3254b04da887ed48c0462b05cb55ce2a910272a7b223ea882b79.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/a3696940d2ee3254b04da887ed48c0462b05cb55ce2a910272a7b223ea882b79.jpg)

![b2746c8a74f1356ec5ea40d0f40eeb9ed24172d0dbc12c849f0b305f38fdb6b1.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/b2746c8a74f1356ec5ea40d0f40eeb9ed24172d0dbc12c849f0b305f38fdb6b1.jpg)

![b9f9d23c87054eeac41e1af9de2dc0d0f62119b074bc9823a6bd655b54946621.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/b9f9d23c87054eeac41e1af9de2dc0d0f62119b074bc9823a6bd655b54946621.jpg)

![e1c0c7b463255b4304af2e380cb6c97f96c0225f593fd141cc5828e8e1f210ef.jpg](../icml_results/2829_PCGS_%20Deblurring%203D%20Gaussian%20Splatting%20with%20Patch%20Comparison/tables/e1c0c7b463255b4304af2e380cb6c97f96c0225f593fd141cc5828e8e1f210ef.jpg)

<a id="paper-2793"></a>

## Solving Spatial-Spectral Fusion with Latent Spectral Operators


### Images

![07f4a0ec52a7e6306ab96a7b0d26fbed5159c3d52065c4030eb63ff57bdaf879.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/07f4a0ec52a7e6306ab96a7b0d26fbed5159c3d52065c4030eb63ff57bdaf879.jpg)

![217f29cfd3b8056d8767f80d5b5f300a1d5fa8411430ad1549b47d4fbba1f851.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/217f29cfd3b8056d8767f80d5b5f300a1d5fa8411430ad1549b47d4fbba1f851.jpg)

![67bafd821860038ba8d991e62467644654d13a57deff2bd03d1fd5310d602129.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/67bafd821860038ba8d991e62467644654d13a57deff2bd03d1fd5310d602129.jpg)

![6d91e8b5af3483a3bb3c3e1e866795b6b6b585a7a0d979b1bcc4e25b55bb5e94.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/6d91e8b5af3483a3bb3c3e1e866795b6b6b585a7a0d979b1bcc4e25b55bb5e94.jpg)

![73f7d61bc5a2a6f21127066ddaf58a1f264062ea7e0ba4bd29034902bd0398ff.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/73f7d61bc5a2a6f21127066ddaf58a1f264062ea7e0ba4bd29034902bd0398ff.jpg)

![85030c51ee54baeba375bbef526ff6edf76d6d0c9f502302ce06473a36771a4c.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/85030c51ee54baeba375bbef526ff6edf76d6d0c9f502302ce06473a36771a4c.jpg)

![872bcdb156775054fd15e8be1a86e36ea4547607d25b5199780a4716b312547c.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/872bcdb156775054fd15e8be1a86e36ea4547607d25b5199780a4716b312547c.jpg)

![9ee5b3fda3b6ea08c2b363cadeff45f81df95bead824c9233dce54275cf8a47e.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/9ee5b3fda3b6ea08c2b363cadeff45f81df95bead824c9233dce54275cf8a47e.jpg)

![a1675ba8e34fe1526e1ec104056a8d5ecc9b7e3d802687e6ff45e038aaa39bff.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/a1675ba8e34fe1526e1ec104056a8d5ecc9b7e3d802687e6ff45e038aaa39bff.jpg)

![b4991bc0c7f4379268ce198e027ab6afc909f3961fed2e8a01115761ae35ab8b.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/b4991bc0c7f4379268ce198e027ab6afc909f3961fed2e8a01115761ae35ab8b.jpg)

![bc1a14d41fad552de27b3081fd55009d59d2357328940dbdd5319dd3d838a563.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/bc1a14d41fad552de27b3081fd55009d59d2357328940dbdd5319dd3d838a563.jpg)

![ff43967e202f24e49a56c382d81d5a3025664bca2ac3e8b43a9dbca11ee25a53.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/ff43967e202f24e49a56c382d81d5a3025664bca2ac3e8b43a9dbca11ee25a53.jpg)

![ff81f38f7d0e9704a970012909f1d5aa21f238d955a3e543717a935b087486b5.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/images/ff81f38f7d0e9704a970012909f1d5aa21f238d955a3e543717a935b087486b5.jpg)

### Tables

![04ea424a638c1125638b901da03215507f314306520913139b93349ebf03c55d.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/04ea424a638c1125638b901da03215507f314306520913139b93349ebf03c55d.jpg)

![067b9898ed88ad21f82b59199280a78b7072e7385ab74118e9b980c93554c4ff.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/067b9898ed88ad21f82b59199280a78b7072e7385ab74118e9b980c93554c4ff.jpg)

![072297aef584ec1b7a6e3bb6698b92769aeebb8442e06fa639a6adb8ae7c6ac7.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/072297aef584ec1b7a6e3bb6698b92769aeebb8442e06fa639a6adb8ae7c6ac7.jpg)

![1a8e659b91334438841dd964cbaa8c886ced875d82f39ade437117a143c1f855.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/1a8e659b91334438841dd964cbaa8c886ced875d82f39ade437117a143c1f855.jpg)

![2d0821930455d924a35e10eeb201932438bfc48cc7fab8483d9ece4671fb8fb6.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/2d0821930455d924a35e10eeb201932438bfc48cc7fab8483d9ece4671fb8fb6.jpg)

![2efcd98928fe25c568dec532782a8e9593ee8fd23c33a54f0182eee92d0c94bc.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/2efcd98928fe25c568dec532782a8e9593ee8fd23c33a54f0182eee92d0c94bc.jpg)

![45537c332d719ef41cfac18f5f42c7d9ace5efd737e5e138dee8b296bb4d2463.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/45537c332d719ef41cfac18f5f42c7d9ace5efd737e5e138dee8b296bb4d2463.jpg)

![64900c32246bfedcaa50ec7381d27c69f9dd2ba6725f74320d1e75ad88324e3b.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/64900c32246bfedcaa50ec7381d27c69f9dd2ba6725f74320d1e75ad88324e3b.jpg)

![95d0161ced987e05c98cccaed18e03cb3a2b01761596945577fa73d148660f77.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/95d0161ced987e05c98cccaed18e03cb3a2b01761596945577fa73d148660f77.jpg)

![9d2bc2ac5549c2cf7350025663910a4e9a003d64606eb4bb96c6d94c26e73f82.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/9d2bc2ac5549c2cf7350025663910a4e9a003d64606eb4bb96c6d94c26e73f82.jpg)

![aa5962a1a9ead6b3d24b6fa684ddc9b6a5a040ac45ede0310ef24ce4980f6926.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/aa5962a1a9ead6b3d24b6fa684ddc9b6a5a040ac45ede0310ef24ce4980f6926.jpg)

![c42ba8dfd87d2a22163d4c30e19afce8ae68e4ea13afce2716ad3840f017a9dd.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/c42ba8dfd87d2a22163d4c30e19afce8ae68e4ea13afce2716ad3840f017a9dd.jpg)

![efd9a85896a3ebca9d57974c8cf594309cce513e9a50736b13edb03b7aaa4eba.jpg](../icml_results/2830_Solving%20Spatial-Spectral%20Fusion%20with%20Latent%20Spectral%20Operators/tables/efd9a85896a3ebca9d57974c8cf594309cce513e9a50736b13edb03b7aaa4eba.jpg)

<a id="paper-2794"></a>

## BVS: Bayesian Visual Search with Multimodal Large Language Model for Fine-grained Perception


### Images

![2577ca42df31b51450205306d5c36765e896d3d0b359f9d90b68e549793bf707.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/images/2577ca42df31b51450205306d5c36765e896d3d0b359f9d90b68e549793bf707.jpg)

![4820586874e3f33584d56576dfa2799a6391cc7b0297b10bf647e4591cbf42e3.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/images/4820586874e3f33584d56576dfa2799a6391cc7b0297b10bf647e4591cbf42e3.jpg)

![b34580db8644585a148e1f94386b23b9060fb9c3ae68ac63c82977b1bbfb2793.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/images/b34580db8644585a148e1f94386b23b9060fb9c3ae68ac63c82977b1bbfb2793.jpg)

![b74f0e3dc502e254539f1b77eb8db619e768d4d7d04626fa42365dd834460ce0.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/images/b74f0e3dc502e254539f1b77eb8db619e768d4d7d04626fa42365dd834460ce0.jpg)

![eb819f9a3d2188c7ba3eee57de6d1ed051b98a2c2d2887b109418b02ff8b1911.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/images/eb819f9a3d2188c7ba3eee57de6d1ed051b98a2c2d2887b109418b02ff8b1911.jpg)

### Tables

![3995abebaf0f8ca7fe5568170f878adfeccde802dc07ddbcd427ddd9d4996584.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/tables/3995abebaf0f8ca7fe5568170f878adfeccde802dc07ddbcd427ddd9d4996584.jpg)

![5ed760907b88e42dca37950ed6d62726331943a35a0ba84acd538d57f2e5a080.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/tables/5ed760907b88e42dca37950ed6d62726331943a35a0ba84acd538d57f2e5a080.jpg)

![69e45c93945c6b16bec9ed274c5de31f5c261ae98b3e76fbafcfbc2beea208a6.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/tables/69e45c93945c6b16bec9ed274c5de31f5c261ae98b3e76fbafcfbc2beea208a6.jpg)

![6d42a34c908d9de74b501da38ffd393107bc23037c3f74f19f1c5d72bcce2c3e.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/tables/6d42a34c908d9de74b501da38ffd393107bc23037c3f74f19f1c5d72bcce2c3e.jpg)

![eb5d4ac5e1a766599721dfa331a5d08aa9ca4c3e9374f0e1b73968b9e6bf2b28.jpg](../icml_results/2831_BVS_%20Bayesian%20Visual%20Search%20with%20Multimodal%20Large%20Language%20Model%20for%20Fine-grained%20Perception/tables/eb5d4ac5e1a766599721dfa331a5d08aa9ca4c3e9374f0e1b73968b9e6bf2b28.jpg)

<a id="paper-2795"></a>

## When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging


### Images

![552d3a03061595ae69ae26a9ecb61746578db5c8a6d82edc3b62a04496ad2b2f.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/552d3a03061595ae69ae26a9ecb61746578db5c8a6d82edc3b62a04496ad2b2f.jpg)

![5f58c326efabaab54b2b997f0f6f234341d995c05125868dd7c4180459a8b877.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/5f58c326efabaab54b2b997f0f6f234341d995c05125868dd7c4180459a8b877.jpg)

![8e3a1f23a44897c273e62f5660c1b3450659becbc8a5b2b7b86ecceb19598177.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/8e3a1f23a44897c273e62f5660c1b3450659becbc8a5b2b7b86ecceb19598177.jpg)

![a483a910a3df1857774a387e7b8b503b70d60e8f2c0aea4322a3e85ea1dfd02a.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/a483a910a3df1857774a387e7b8b503b70d60e8f2c0aea4322a3e85ea1dfd02a.jpg)

![d2049184a879795c71480bf68d182970614c0ae9791415d79225ac6479164582.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/d2049184a879795c71480bf68d182970614c0ae9791415d79225ac6479164582.jpg)

![e5b596adfabe8a77fe27016bcd78b200d3dbc875877486df69d4007f3215ecce.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/images/e5b596adfabe8a77fe27016bcd78b200d3dbc875877486df69d4007f3215ecce.jpg)

### Tables

![1eeb414c52f6921186d821a651263f554bd7942a8df6033ca12ab7473991e64f.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/1eeb414c52f6921186d821a651263f554bd7942a8df6033ca12ab7473991e64f.jpg)

![2805f4b44f5c64a381b08fd4fdfe38e7e321e777ab7b8069b2fb3bddda0b38e8.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/2805f4b44f5c64a381b08fd4fdfe38e7e321e777ab7b8069b2fb3bddda0b38e8.jpg)

![640afb16476da4b71084d0a52981da842f0c05f25af258b2446fe4b9d92e071d.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/640afb16476da4b71084d0a52981da842f0c05f25af258b2446fe4b9d92e071d.jpg)

![6d622abf25e1f15e8a23afdf563bac37b14597952e8222d19b15ec773658d09b.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/6d622abf25e1f15e8a23afdf563bac37b14597952e8222d19b15ec773658d09b.jpg)

![777aa447c126db51eb73fa46c10e7fea1ebd128d1fc84c8085530f63c91b36bc.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/777aa447c126db51eb73fa46c10e7fea1ebd128d1fc84c8085530f63c91b36bc.jpg)

![80bfba3e9aeb0fffe26016d5c79129c7b35c860b36e9ba15aa4dec85620aba8b.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/80bfba3e9aeb0fffe26016d5c79129c7b35c860b36e9ba15aa4dec85620aba8b.jpg)

![965fc4cd361e6733daf3dc1a38ba394262ba5e184098740495eb805ca65e2c61.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/965fc4cd361e6733daf3dc1a38ba394262ba5e184098740495eb805ca65e2c61.jpg)

![bab62cf201679e5f3955e2637bc24dd2572221f5437726a794cf1f4e6fbf44be.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/bab62cf201679e5f3955e2637bc24dd2572221f5437726a794cf1f4e6fbf44be.jpg)

![bc538e9adbabb4ba3bc4799aeae673957a789107ad2c1e750a185a3b16e3fe83.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/bc538e9adbabb4ba3bc4799aeae673957a789107ad2c1e750a185a3b16e3fe83.jpg)

![cd205c6b89d3aaeed3389c4a8ae7d41e6acd4808a06f82eba73f7faa8cd44243.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/cd205c6b89d3aaeed3389c4a8ae7d41e6acd4808a06f82eba73f7faa8cd44243.jpg)

![d321a881df0aa90d4e805546e7fa0163a30fa223ab3487a828003948e6168866.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/d321a881df0aa90d4e805546e7fa0163a30fa223ab3487a828003948e6168866.jpg)

![d839a352d2ae4b238d11579fac3d4060c8350e3050b8ae1ac53ffb32218e2879.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/d839a352d2ae4b238d11579fac3d4060c8350e3050b8ae1ac53ffb32218e2879.jpg)

![da1bc004438b5c2a905101f35924433d320d6c0760b0ea971bac2595935a76c1.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/da1bc004438b5c2a905101f35924433d320d6c0760b0ea971bac2595935a76c1.jpg)

![ec802de317d4cd17b36c38fe0c314250191a655606b3f04f41beba43080427f5.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/ec802de317d4cd17b36c38fe0c314250191a655606b3f04f41beba43080427f5.jpg)

![f2ebf7d0f78661b0e79ec99dde1e9d9d18e6e8dafcca73f1cecaebba87e517c2.jpg](../icml_results/2832_When%20Shared%20Knowledge%20Hurts_%20Spectral%20Over-Accumulation%20in%20Model%20Merging/tables/f2ebf7d0f78661b0e79ec99dde1e9d9d18e6e8dafcca73f1cecaebba87e517c2.jpg)

<a id="paper-2796"></a>

## Energy-Structured Low-Rank Adaptation for Continual Learning


### Images

![4119879eda4319c3463659a6fd567c8a80de4cc15183a829a930a2ca1c0f913b.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/4119879eda4319c3463659a6fd567c8a80de4cc15183a829a930a2ca1c0f913b.jpg)

![832178850d7687e7047033101505ea99cd228146e3b950646a3323efffbda184.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/832178850d7687e7047033101505ea99cd228146e3b950646a3323efffbda184.jpg)

![c323a11ab0a130824c99f04a53ffc1106be7b4d95e2a5b365d32160ad3129900.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/c323a11ab0a130824c99f04a53ffc1106be7b4d95e2a5b365d32160ad3129900.jpg)

![d4462a653524e6d75baed8465f3cc6a18082f1c27ed6879d994751a93a988d2c.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/d4462a653524e6d75baed8465f3cc6a18082f1c27ed6879d994751a93a988d2c.jpg)

![db892ad8a9158187b911b1f9ff18452ccbf2998e9c4794e7e19de5ad1a901607.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/db892ad8a9158187b911b1f9ff18452ccbf2998e9c4794e7e19de5ad1a901607.jpg)

![fa58b0f80108daf28422178f1475985d5236e135b5cea76af0a4a31ad00efdec.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/fa58b0f80108daf28422178f1475985d5236e135b5cea76af0a4a31ad00efdec.jpg)

![fc39c7014b312b7c0815dfd4880bc2ac6ea027da3447f33ba52c68e67da475db.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/fc39c7014b312b7c0815dfd4880bc2ac6ea027da3447f33ba52c68e67da475db.jpg)

### Tables

![39b6197044d5fa4242d3dc7956edac55574cfed0d9f238781ee06fe49f714942.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/39b6197044d5fa4242d3dc7956edac55574cfed0d9f238781ee06fe49f714942.jpg)

![729838fbbd8426d814c17c5470273d2b68b3611fbc5ce476538054436480c342.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/729838fbbd8426d814c17c5470273d2b68b3611fbc5ce476538054436480c342.jpg)

![9fc1f0607a9f6474ef2587390991fae691376750dd815ee63c6ec868eba28f82.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/9fc1f0607a9f6474ef2587390991fae691376750dd815ee63c6ec868eba28f82.jpg)

![a1ecc233db6d928c01ae68459acae7cea403e5cc08a7db3909218b2688247954.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/a1ecc233db6d928c01ae68459acae7cea403e5cc08a7db3909218b2688247954.jpg)

![cf4ae58b5e657b129a1132c5a19ad939a3b947b03bf39cd2cf308a732e61150c.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/cf4ae58b5e657b129a1132c5a19ad939a3b947b03bf39cd2cf308a732e61150c.jpg)

![d50396f13ee1e314fa581a9fe89e47b906a3f03a9db0fd241c3949257ce77207.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/d50396f13ee1e314fa581a9fe89e47b906a3f03a9db0fd241c3949257ce77207.jpg)

![e02a3c9586dc2880654d2a44b8669424fc903a91dd9bc484a3397d6afb153e3a.jpg](../icml_results/2833_Energy-Structured%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/e02a3c9586dc2880654d2a44b8669424fc903a91dd9bc484a3397d6afb153e3a.jpg)

<a id="paper-2797"></a>

## Intrinsic Gradient Suppression for Label-Noise Prompt Tuning in Vision–Language Models


### Images

![051ffe312be0a858a7d2ede71535a3595e076ca7c94152fdfafdc7cbd37bc7f7.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/051ffe312be0a858a7d2ede71535a3595e076ca7c94152fdfafdc7cbd37bc7f7.jpg)

![660bd34313adb0cab4ef8efe57a6b7fbe48f3620819da9f835660a09c1b337c7.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/660bd34313adb0cab4ef8efe57a6b7fbe48f3620819da9f835660a09c1b337c7.jpg)

![7ff618ccc3928ba6c533d876cc0ce02ee9b08a47dd3d83222abc36ae5d378796.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/7ff618ccc3928ba6c533d876cc0ce02ee9b08a47dd3d83222abc36ae5d378796.jpg)

![c80e758ad3b13f22e9e9f8b41e356d6fe68d76f8fdae489fbed28d31ad21339f.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/c80e758ad3b13f22e9e9f8b41e356d6fe68d76f8fdae489fbed28d31ad21339f.jpg)

![d9fed9fc53a486fef28ba3a112c3294b832955eb7fc196baadcd6b901b00ef7f.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/d9fed9fc53a486fef28ba3a112c3294b832955eb7fc196baadcd6b901b00ef7f.jpg)

![e6b1efa8942354576e09ed90dd56bf97c8c009411733e5362b34c445674ed9ca.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/e6b1efa8942354576e09ed90dd56bf97c8c009411733e5362b34c445674ed9ca.jpg)

![e6ba2b9f3eb77148408880c903b837dc01afbe7dd350f6a4f2731c156645b5cd.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/e6ba2b9f3eb77148408880c903b837dc01afbe7dd350f6a4f2731c156645b5cd.jpg)

![e7a4e852baec449d51c0b1a74e7e923cdd3ee1f2c050d5e5d0f6df2436ac13e7.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/e7a4e852baec449d51c0b1a74e7e923cdd3ee1f2c050d5e5d0f6df2436ac13e7.jpg)

![fbab5f1cf0fe45589f52a6d06d07c4b7c65636b13c7079c4fef1f29b9256df7a.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/images/fbab5f1cf0fe45589f52a6d06d07c4b7c65636b13c7079c4fef1f29b9256df7a.jpg)

### Tables

![294d4644668f51d36e77c53f6071f7289b223ec0cb9b3af7c04de16be320c1aa.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/tables/294d4644668f51d36e77c53f6071f7289b223ec0cb9b3af7c04de16be320c1aa.jpg)

![2bcb868bae1402a99ab8d27982ac23489150c04ced57534336b2501358c99aae.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/tables/2bcb868bae1402a99ab8d27982ac23489150c04ced57534336b2501358c99aae.jpg)

![467940e0ff3679b387127c489eb352010ce7378249121eff3bd91512108545e3.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/tables/467940e0ff3679b387127c489eb352010ce7378249121eff3bd91512108545e3.jpg)

![76cec6256cd7fd600576a8d43d70921ec784efde747eb6a0a24a5b1985fad34a.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/tables/76cec6256cd7fd600576a8d43d70921ec784efde747eb6a0a24a5b1985fad34a.jpg)

![cc185622291e4e1e4f13a8c891ce84c6fa03840812e11dbe0015b41900842da3.jpg](../icml_results/2834_Intrinsic%20Gradient%20Suppression%20for%20Label-Noise%20Prompt%20Tuning%20in%20Vision%E2%80%93Language%20Models/tables/cc185622291e4e1e4f13a8c891ce84c6fa03840812e11dbe0015b41900842da3.jpg)

<a id="paper-2798"></a>

## Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?


### Images

![14ca02ced57a116d3a2f243165eb22cc1c4ec33ceef424115ac71641503acbc4.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/14ca02ced57a116d3a2f243165eb22cc1c4ec33ceef424115ac71641503acbc4.jpg)

![426843575e929a30a773f6fe853d2693b6feb4b32f66011b060be66d35f0b4de.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/426843575e929a30a773f6fe853d2693b6feb4b32f66011b060be66d35f0b4de.jpg)

![d17063683d8ef86dd3500d16ea1e14ead94a39edc4b326fbff48ce6f2f210d10.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/d17063683d8ef86dd3500d16ea1e14ead94a39edc4b326fbff48ce6f2f210d10.jpg)

![e30ae43235f6c9a7c34bab1f3b30e895d43721a0c923a0f937fa54ce90f230e7.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/e30ae43235f6c9a7c34bab1f3b30e895d43721a0c923a0f937fa54ce90f230e7.jpg)

![e34994db2acb3f5c427c10b5d0aef6ae25a1179477bf9286ed03e22915bba40d.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/e34994db2acb3f5c427c10b5d0aef6ae25a1179477bf9286ed03e22915bba40d.jpg)

![f95d0598dff8901297f06b7c30ad291df75e11d00081a0a34518ef4af129fcdf.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/images/f95d0598dff8901297f06b7c30ad291df75e11d00081a0a34518ef4af129fcdf.jpg)

### Tables

![36c4448bb63ea8fc9cd3afd5097227318ee0e62604925a3280034559dc4f895b.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/36c4448bb63ea8fc9cd3afd5097227318ee0e62604925a3280034559dc4f895b.jpg)

![65b7ed1b4c11eaf9636a39a6ebbd63c742ec6b0ed37dc1f33df008b0f965c7b5.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/65b7ed1b4c11eaf9636a39a6ebbd63c742ec6b0ed37dc1f33df008b0f965c7b5.jpg)

![af41c2432ae6591a55c8c5b4b13bb5a818bdb8f5198c4d4cd7ad1b29c6e8bd0e.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/af41c2432ae6591a55c8c5b4b13bb5a818bdb8f5198c4d4cd7ad1b29c6e8bd0e.jpg)

![b87251aba383854af02a551187a21d495a5cfd6e29eca40da4cd969f1c803682.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/b87251aba383854af02a551187a21d495a5cfd6e29eca40da4cd969f1c803682.jpg)

![d2fd3b872becddcb6e07bf059a79f0e800b772c1bacee0c862f042a82cf67b64.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/d2fd3b872becddcb6e07bf059a79f0e800b772c1bacee0c862f042a82cf67b64.jpg)

![e1dcfdd71b7c7ccde85313d109243d646f95bb714d3f2bf59ae9ed10195abcea.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/e1dcfdd71b7c7ccde85313d109243d646f95bb714d3f2bf59ae9ed10195abcea.jpg)

![f537ecde1864e8930cce9c1362aff535f966a1d3f986adde107a2b5314107364.jpg](../icml_results/2835_Are%20We%20Overconfident%20in%20Models%20and%20Results%20for%20Semi-Supervised%203D%20Medical%20Image%20Segmentation_/tables/f537ecde1864e8930cce9c1362aff535f966a1d3f986adde107a2b5314107364.jpg)

<a id="paper-2799"></a>

## Reasoning as an Attack Surface: Adaptive Evolutionary CoT Jailbreaks for LLMs


### Images

![7645576a137d1c763a0655289d7bdb59038376fe7732bbd91a7e3f331832dc74.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/images/7645576a137d1c763a0655289d7bdb59038376fe7732bbd91a7e3f331832dc74.jpg)

![b28a2f2f45e7843a107a2d0daa33819cc9d4aeaa7afbd8587879dc9f027221ed.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/images/b28a2f2f45e7843a107a2d0daa33819cc9d4aeaa7afbd8587879dc9f027221ed.jpg)

### Tables

![1ceb33567e251076733d2cd9bdc79872b4f0ad64044cd197aec9fb4ee69caf28.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/1ceb33567e251076733d2cd9bdc79872b4f0ad64044cd197aec9fb4ee69caf28.jpg)

![280f0e52920b6c711a481d912f2629074a5a88d2dcc6665df2b399de36331367.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/280f0e52920b6c711a481d912f2629074a5a88d2dcc6665df2b399de36331367.jpg)

![34c538dce4c4412de42c7f5a76428154615d9fffc786fb7dc0133f2bdbcd6453.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/34c538dce4c4412de42c7f5a76428154615d9fffc786fb7dc0133f2bdbcd6453.jpg)

![3a45abfa22119378c6daf30311094ade59637b34960138e35f917b6f06ae05ea.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/3a45abfa22119378c6daf30311094ade59637b34960138e35f917b6f06ae05ea.jpg)

![3ab0146b64f6ae14ecc3735fbbea1dc364563b07cd71ca289d5a681aedc0c9de.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/3ab0146b64f6ae14ecc3735fbbea1dc364563b07cd71ca289d5a681aedc0c9de.jpg)

![3e3a99b520113e1d192ddfd41135363cb2060bb91c0884f666f71b3a47d04d0f.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/3e3a99b520113e1d192ddfd41135363cb2060bb91c0884f666f71b3a47d04d0f.jpg)

![4a270370baef21a8ac67b7895d8b0d723e9170177431d351d9bcbbbba47e69df.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/4a270370baef21a8ac67b7895d8b0d723e9170177431d351d9bcbbbba47e69df.jpg)

![50e2c4146b0b842e80bb0aa80afe63efa1bbeedafb06ced6d1b2407dea1f9e84.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/50e2c4146b0b842e80bb0aa80afe63efa1bbeedafb06ced6d1b2407dea1f9e84.jpg)

![564261ab57f7a8c7d3cc688b36b6386e37e110d54c7f5f67f6b1ee65ce652b9a.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/564261ab57f7a8c7d3cc688b36b6386e37e110d54c7f5f67f6b1ee65ce652b9a.jpg)

![567fd16c814d693481e38c8fc0f9b6528cb738ea8520ff8a65ff3c3d58934f9c.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/567fd16c814d693481e38c8fc0f9b6528cb738ea8520ff8a65ff3c3d58934f9c.jpg)

![5af6dff6d01a016906e46e60e507c0eb09c02bdf59f1002724ffd43810a8e96d.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/5af6dff6d01a016906e46e60e507c0eb09c02bdf59f1002724ffd43810a8e96d.jpg)

![727b7419b8ed654b51d0c9861acefea2951638707039cb820883b0145c2ccfb9.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/727b7419b8ed654b51d0c9861acefea2951638707039cb820883b0145c2ccfb9.jpg)

![783bc67e263ea7de896e73ec358654ee4cac8a732e5449088671dfc9ba89775a.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/783bc67e263ea7de896e73ec358654ee4cac8a732e5449088671dfc9ba89775a.jpg)

![852e38c7cffa2e15fa41fca349b58eb5edca5fc444363fd6deae04c07d86f8bb.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/852e38c7cffa2e15fa41fca349b58eb5edca5fc444363fd6deae04c07d86f8bb.jpg)

![973c5d53579033e1b347ca7ca8f301e3ed5ab2f202bc307aa1426b038363fd6c.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/973c5d53579033e1b347ca7ca8f301e3ed5ab2f202bc307aa1426b038363fd6c.jpg)

![a9f0505ac6c85cd70cd52ca06190f9ca16b2697a92b3543c4934436ba92f93c3.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/a9f0505ac6c85cd70cd52ca06190f9ca16b2697a92b3543c4934436ba92f93c3.jpg)

![abfbbf1096423624eaa7aba6029546d690eca0dd349cb1ba92fc0ac4ea337bdf.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/abfbbf1096423624eaa7aba6029546d690eca0dd349cb1ba92fc0ac4ea337bdf.jpg)

![ad78e8a2992b9ed582a9291dac035a3af699d19e629207f6107d7948fbd4ce61.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/ad78e8a2992b9ed582a9291dac035a3af699d19e629207f6107d7948fbd4ce61.jpg)

![ddd6f1169b5ff45973cc171f2554c25a9848bf9d77b263819985f7755f20a62a.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/ddd6f1169b5ff45973cc171f2554c25a9848bf9d77b263819985f7755f20a62a.jpg)

![fae0bc46203833c7d2505a428713a2f7d460d0cfc252559dcec3406256923632.jpg](../icml_results/2836_Reasoning%20as%20an%20Attack%20Surface_%20Adaptive%20Evolutionary%20CoT%20Jailbreaks%20for%20LLMs/tables/fae0bc46203833c7d2505a428713a2f7d460d0cfc252559dcec3406256923632.jpg)

<a id="paper-2800"></a>

## AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing


### Images

![0d142f0991ca9cc46f6377024faf6492b862a0dc9abe1033f4372ed870aef9f1.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/0d142f0991ca9cc46f6377024faf6492b862a0dc9abe1033f4372ed870aef9f1.jpg)

![2e2940d5e979aae504dcbfd33d3f329e8ebbbfa832cd253bc8cebf7761a6502e.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/2e2940d5e979aae504dcbfd33d3f329e8ebbbfa832cd253bc8cebf7761a6502e.jpg)

![4169156be216b3e0aa057658e4716520e13a76368bca89c71f28b7467a6a3ac9.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/4169156be216b3e0aa057658e4716520e13a76368bca89c71f28b7467a6a3ac9.jpg)

![60b4e1f488f013e6f292393a3f7a6da7939549c772db1e0c156ea5a09c468a1f.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/60b4e1f488f013e6f292393a3f7a6da7939549c772db1e0c156ea5a09c468a1f.jpg)

![7ddf7fe6f7d50a63a34b73e407c6728e17099793a1d9e3d4bec442a5488b3792.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/7ddf7fe6f7d50a63a34b73e407c6728e17099793a1d9e3d4bec442a5488b3792.jpg)

![7f4183aa603f874e8cf0875a4d2be274f438c76e421a24b0a18a856be7146256.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/7f4183aa603f874e8cf0875a4d2be274f438c76e421a24b0a18a856be7146256.jpg)

![7ffda2ad45304820fa1e24764725c91c21bd278fd1d07b2ce5f17cf3383c9fa2.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/7ffda2ad45304820fa1e24764725c91c21bd278fd1d07b2ce5f17cf3383c9fa2.jpg)

![ea0de0a04d3c6672b838a3a6fdd6a2beee15b4294a6e6cb29a99ba2900208ff6.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/images/ea0de0a04d3c6672b838a3a6fdd6a2beee15b4294a6e6cb29a99ba2900208ff6.jpg)

### Tables

![040437bfa0c93961ea1079f08c8001dcc6fb46969e5fe37cc7e7d3386daedb76.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/040437bfa0c93961ea1079f08c8001dcc6fb46969e5fe37cc7e7d3386daedb76.jpg)

![1629c35f92184771e226e06f6ac63fe230dad40192c492fab372a26cf24c5570.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/1629c35f92184771e226e06f6ac63fe230dad40192c492fab372a26cf24c5570.jpg)

![30cb0488e62a38db4721e00b5fd102179508351ef414645ff5450ddf9b902710.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/30cb0488e62a38db4721e00b5fd102179508351ef414645ff5450ddf9b902710.jpg)

![33a69b7dda45f9b3af217abf7a01920bebfb3dcff1718181780604d1deb33cea.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/33a69b7dda45f9b3af217abf7a01920bebfb3dcff1718181780604d1deb33cea.jpg)

![3cf97a348965cffef2afc7d69f2155181e3fe5bef08f62dd570d91eebe0f3477.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/3cf97a348965cffef2afc7d69f2155181e3fe5bef08f62dd570d91eebe0f3477.jpg)

![4ab72a848806caec2ece44b93b8d2158e220b1d46459eca0977539df31f5e90d.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/4ab72a848806caec2ece44b93b8d2158e220b1d46459eca0977539df31f5e90d.jpg)

![730deb1ea381317ce37f5f6c2a196785b2336cbefe9961f723946d233de95cc9.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/730deb1ea381317ce37f5f6c2a196785b2336cbefe9961f723946d233de95cc9.jpg)

![7ac956c5527e708f055ffcb60299785530603c7191ebe40690b5b6a4a1dfc955.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/7ac956c5527e708f055ffcb60299785530603c7191ebe40690b5b6a4a1dfc955.jpg)

![9c84c3f3e36eb0e1b5d567bb915990ea1de5e3bbfb598068fafc3a811e4afe9e.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/9c84c3f3e36eb0e1b5d567bb915990ea1de5e3bbfb598068fafc3a811e4afe9e.jpg)

![a52378e74403832cd16d25719f4562c4e46ead141b51ea9188984720e86c8028.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/a52378e74403832cd16d25719f4562c4e46ead141b51ea9188984720e86c8028.jpg)

![b393e5b63d4646422e8b8788ad14df6f78995e6f19e7feb0242fd7c3d5501346.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/b393e5b63d4646422e8b8788ad14df6f78995e6f19e7feb0242fd7c3d5501346.jpg)

![e06b5d56bcc15d363db9c96ddb06e5d4380a7989fa1dc66d8f979fefb2ae1857.jpg](../icml_results/2837_AliMark_%20Enhancing%20Robustness%20of%20Sentence-Level%20Watermarking%20Against%20Text%20Paraphrasing/tables/e06b5d56bcc15d363db9c96ddb06e5d4380a7989fa1dc66d8f979fefb2ae1857.jpg)

<a id="paper-2801"></a>

## A Studentized Spherical Harmonics–Based Nonparametric Two-Sample Test for Compositional and Directional Data


### Images

![140a51e1bf7116c64856386aea9cc824dbe511f590e583a90215ae79b9005c13.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/140a51e1bf7116c64856386aea9cc824dbe511f590e583a90215ae79b9005c13.jpg)

![2f6837a40ae07c0901ac7f969471de2999d42ab1ef35302e801b27b1a70eb062.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/2f6837a40ae07c0901ac7f969471de2999d42ab1ef35302e801b27b1a70eb062.jpg)

![4b633c003bcf8073f7b9a5e9cd04002c9be7df0da731b55dfec53a5de432ae15.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/4b633c003bcf8073f7b9a5e9cd04002c9be7df0da731b55dfec53a5de432ae15.jpg)

![66281180690be4c6c6822667773957eb51686dfd56cdf5bf3fc477c10b6abb0d.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/66281180690be4c6c6822667773957eb51686dfd56cdf5bf3fc477c10b6abb0d.jpg)

![890f69e0fa4d3d16fcad0a6e5e205b3fe5ac5a5c0223b9b1c61f87473693a3e1.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/890f69e0fa4d3d16fcad0a6e5e205b3fe5ac5a5c0223b9b1c61f87473693a3e1.jpg)

![c34c448c2886b8f00b81f6d1ba4bee607384d8cbb97a0115c4f891cdfb68697f.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/images/c34c448c2886b8f00b81f6d1ba4bee607384d8cbb97a0115c4f891cdfb68697f.jpg)

### Tables

![1c21f6e0f61b5f6eb498954c84a7299fe4e188c1d0f7e802bf8b313e4a37e86d.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/tables/1c21f6e0f61b5f6eb498954c84a7299fe4e188c1d0f7e802bf8b313e4a37e86d.jpg)

![40b0398a4a9e53a280de77d0118b141053152ac51d16a5c3c9d0f025536d18c5.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/tables/40b0398a4a9e53a280de77d0118b141053152ac51d16a5c3c9d0f025536d18c5.jpg)

![76dfd74b03f95728f4c47c248b2e817d62cc28da4586d49cd4d20a9405569ad1.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/tables/76dfd74b03f95728f4c47c248b2e817d62cc28da4586d49cd4d20a9405569ad1.jpg)

![7c3829dd5d39eafb706ded31cc4585b98cbffd5e073862d33e35dc3859ea8ce8.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/tables/7c3829dd5d39eafb706ded31cc4585b98cbffd5e073862d33e35dc3859ea8ce8.jpg)

![c0dfedf2930d632467623f8997f7a873052cf25625de79059b51e1939349f768.jpg](../icml_results/2838_A%20Studentized%20Spherical%20Harmonics%E2%80%93Based%20Nonparametric%20Two-Sample%20Test%20for%20Compositional%20and%20Directio/tables/c0dfedf2930d632467623f8997f7a873052cf25625de79059b51e1939349f768.jpg)

<a id="paper-2802"></a>

## Safe Reinforcement Learning with Preference-based Constraint Inference


### Images

![0e3bc7a2bed19fa20ceac9712fbb254f6e2e61a565b21678ef6a58e0ca298455.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/0e3bc7a2bed19fa20ceac9712fbb254f6e2e61a565b21678ef6a58e0ca298455.jpg)

![2276c60b89dbd4a944621cd445225a0ef820f576456bc3ad5d99a32a03f01c8b.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/2276c60b89dbd4a944621cd445225a0ef820f576456bc3ad5d99a32a03f01c8b.jpg)

![364bda61497b59ee865ffa9c594cf22cdebc6fee8e9bb16f946eed064c8a5bec.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/364bda61497b59ee865ffa9c594cf22cdebc6fee8e9bb16f946eed064c8a5bec.jpg)

![464cd816f49053f9b7d1b16f08232dd9dba4b42f1428d4e51e6cf3490e3f57fb.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/464cd816f49053f9b7d1b16f08232dd9dba4b42f1428d4e51e6cf3490e3f57fb.jpg)

![9c8d72392dfd0591b22e58617fc8296c907f975e07fdf25936e6418c3e8bb9c2.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/9c8d72392dfd0591b22e58617fc8296c907f975e07fdf25936e6418c3e8bb9c2.jpg)

![b08d1c3885e6f71d56907f0beff8f40e9943230100abd2ca132d4be19767223a.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/b08d1c3885e6f71d56907f0beff8f40e9943230100abd2ca132d4be19767223a.jpg)

![baa64714eb88047a441511e767602751f6f6195f5701160a4cd106ea3d6a29b9.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/baa64714eb88047a441511e767602751f6f6195f5701160a4cd106ea3d6a29b9.jpg)

![ccd92a9d71f249ff67900a398cabe6cd605a773ad14e952d115b740bcad5c866.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/images/ccd92a9d71f249ff67900a398cabe6cd605a773ad14e952d115b740bcad5c866.jpg)

### Tables

![01a2b78701f24dbd81870595d299b566ccec7102198351972074ce631e16ddbe.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/01a2b78701f24dbd81870595d299b566ccec7102198351972074ce631e16ddbe.jpg)

![16567572a6f0c9b3344ef9d748d09a18bf23f49e5cecfe77fc736afadd54603f.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/16567572a6f0c9b3344ef9d748d09a18bf23f49e5cecfe77fc736afadd54603f.jpg)

![4050fcc8844533a418e8067dc142f1ae1864be51d4264af6ba14537617d86cbf.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/4050fcc8844533a418e8067dc142f1ae1864be51d4264af6ba14537617d86cbf.jpg)

![482eed4194936aa9d2d5cfa09e596019983627e0cd7b3872e0ceea094784f553.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/482eed4194936aa9d2d5cfa09e596019983627e0cd7b3872e0ceea094784f553.jpg)

![4e62679431ca00815b9cf879ece3ed24534e2aec22e78f2cf5bb479b01eae0a3.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/4e62679431ca00815b9cf879ece3ed24534e2aec22e78f2cf5bb479b01eae0a3.jpg)

![62ab11f8ccf04dee935fa5f19e7cd9e8e870f73a45946fdede25599d7c83d391.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/62ab11f8ccf04dee935fa5f19e7cd9e8e870f73a45946fdede25599d7c83d391.jpg)

![6704d2fb5be1434225a36b37e7458dbd4cc1d098c7df948e4a012bc46c9f866f.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/6704d2fb5be1434225a36b37e7458dbd4cc1d098c7df948e4a012bc46c9f866f.jpg)

![69fa72b8d2a66a8fa3a9ac849a339348271e009536756182f2db6e438b4c881d.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/69fa72b8d2a66a8fa3a9ac849a339348271e009536756182f2db6e438b4c881d.jpg)

![7ed04e74b5bad964f41d4efcb1de540368a03ee0ca501bd20039cb81b8198e8a.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/7ed04e74b5bad964f41d4efcb1de540368a03ee0ca501bd20039cb81b8198e8a.jpg)

![85258f993a90eb7c85630c25cbd21af9702605714f18abf09828c5dc57648cd9.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/85258f993a90eb7c85630c25cbd21af9702605714f18abf09828c5dc57648cd9.jpg)

![87a271158c3157d33c9e3d42aeb81fbe8bcb100b46582f00d2831262e9fcd7d6.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/87a271158c3157d33c9e3d42aeb81fbe8bcb100b46582f00d2831262e9fcd7d6.jpg)

![89866f7b99e7b5c8fe00fc16310d38d1b4382917190be3e72a2591cfb037d834.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/89866f7b99e7b5c8fe00fc16310d38d1b4382917190be3e72a2591cfb037d834.jpg)

![8f6e6fbb000f151a5fc24c42cf02a29c2301b0196bca22f6e1db1a662eab82cc.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/8f6e6fbb000f151a5fc24c42cf02a29c2301b0196bca22f6e1db1a662eab82cc.jpg)

![d1ad782985f7ebb9f22f2a0a5641e2acd9abc9e7492876e2e8a14e562e864ada.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/d1ad782985f7ebb9f22f2a0a5641e2acd9abc9e7492876e2e8a14e562e864ada.jpg)

![f8eb59ec22f6fa92d62f4bf6df5984c122707de05ed0c0a7fc87585f7e637df4.jpg](../icml_results/2839_Safe%20Reinforcement%20Learning%20with%20Preference-based%20Constraint%20Inference/tables/f8eb59ec22f6fa92d62f4bf6df5984c122707de05ed0c0a7fc87585f7e637df4.jpg)

<a id="paper-2803"></a>

## SURF: Separation via Unsupervised Remixing Flow


### Images

![20cc0213eb96192ab6c11980db4c1db33c69b34ddcc9c4fb061b59e69baece1b.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/20cc0213eb96192ab6c11980db4c1db33c69b34ddcc9c4fb061b59e69baece1b.jpg)

![4bb9d711dbd0ab7d369dfe4845db5f289fb376ee53c96581e4dde43752dd7836.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/4bb9d711dbd0ab7d369dfe4845db5f289fb376ee53c96581e4dde43752dd7836.jpg)

![86cdc13f0ffb349de2ec08853942dcafc07e36f5bbf7762140cbd2a4668555ba.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/86cdc13f0ffb349de2ec08853942dcafc07e36f5bbf7762140cbd2a4668555ba.jpg)

![a70feaf98ee11f1421f08ad950641227748cfade45e7d6092d0daaca64efd89a.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/a70feaf98ee11f1421f08ad950641227748cfade45e7d6092d0daaca64efd89a.jpg)

![aa1c780424dbf8c84c63dc5cb50e3040563279765328e06dcfdc6456f9d7b971.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/aa1c780424dbf8c84c63dc5cb50e3040563279765328e06dcfdc6456f9d7b971.jpg)

![b0978893ee321d59b2df0459697acd181c17e0f9a56a62535704fbd1b9a4a596.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/b0978893ee321d59b2df0459697acd181c17e0f9a56a62535704fbd1b9a4a596.jpg)

![e104f20bca7b51d240ddcbcc4eb4c61c5e7762374885aeb0c75dc888d1fbea89.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/images/e104f20bca7b51d240ddcbcc4eb4c61c5e7762374885aeb0c75dc888d1fbea89.jpg)

### Tables

![04c8735af330ba810ec1c0d9164831e8c4d8ceb663597a30042b320f0386c562.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/04c8735af330ba810ec1c0d9164831e8c4d8ceb663597a30042b320f0386c562.jpg)

![2fde0ec0985152a78b8b82c598d2ebbd8b73175c74b90eca092d503ed774cafd.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/2fde0ec0985152a78b8b82c598d2ebbd8b73175c74b90eca092d503ed774cafd.jpg)

![4f05c53d51109171f4f9198cf6a1e35127f24a200ded36f3523de18385361439.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/4f05c53d51109171f4f9198cf6a1e35127f24a200ded36f3523de18385361439.jpg)

![56d1efd295496cdf858a6b45dfdabae042ec12591c5daa28e4a621c715652491.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/56d1efd295496cdf858a6b45dfdabae042ec12591c5daa28e4a621c715652491.jpg)

![64da87848b094ca8204d647ba5c3d1b300ef0247d9b79e41c2a623560788a2c3.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/64da87848b094ca8204d647ba5c3d1b300ef0247d9b79e41c2a623560788a2c3.jpg)

![7994cf4d70e69d835c1847b9032fabed936cdae8df258437c580ee445aa7b441.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/7994cf4d70e69d835c1847b9032fabed936cdae8df258437c580ee445aa7b441.jpg)

![b36edd9f3612bf87ef3c6842471a6a3220fd111ae26a8513a23d937ecac05217.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/b36edd9f3612bf87ef3c6842471a6a3220fd111ae26a8513a23d937ecac05217.jpg)

![c5c832ddfacdf119e6f2b4cda0bdb8730c2fb8f8ce0f5b6254c18c99a372a427.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/c5c832ddfacdf119e6f2b4cda0bdb8730c2fb8f8ce0f5b6254c18c99a372a427.jpg)

![e6a911a3b08e5af6102e9b878b083bb3f4b147efa74060fba68386d46339460b.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/e6a911a3b08e5af6102e9b878b083bb3f4b147efa74060fba68386d46339460b.jpg)

![ee14acd56e61f9859e9d8bae2ba35d4529cbe2c29b3bafab6162aaed3be85b05.jpg](../icml_results/2840_SURF_%20Separation%20via%20Unsupervised%20Remixing%20Flow/tables/ee14acd56e61f9859e9d8bae2ba35d4529cbe2c29b3bafab6162aaed3be85b05.jpg)

<a id="paper-2804"></a>

## Towards Parameter-Free Temporal Difference Learning


### Images

![a4abbc285de25ee8fc7afd552477e25ef5c53ed6b9d5b0809cdf74cacff868d5.jpg](../icml_results/2841_Towards%20Parameter-Free%20Temporal%20Difference%20Learning/images/a4abbc285de25ee8fc7afd552477e25ef5c53ed6b9d5b0809cdf74cacff868d5.jpg)

![dce3286d0fbe324fdfad1605ca6d8eca810413186d0be271ab2e2c11e9718e68.jpg](../icml_results/2841_Towards%20Parameter-Free%20Temporal%20Difference%20Learning/images/dce3286d0fbe324fdfad1605ca6d8eca810413186d0be271ab2e2c11e9718e68.jpg)

### Tables

![682b226ee752935496e5e2d75ed6714a56d54becfaa59fcb811979bd8ab4defb.jpg](../icml_results/2841_Towards%20Parameter-Free%20Temporal%20Difference%20Learning/tables/682b226ee752935496e5e2d75ed6714a56d54becfaa59fcb811979bd8ab4defb.jpg)

<a id="paper-2805"></a>

## SHAP-Guided Kernel Actor-Critic for Explainable Reinforcement Learning

### Images

![02411fe9efa51fd4d0f1162a8a507a6edd57e2ebae71a8949e09857150c436d0.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/02411fe9efa51fd4d0f1162a8a507a6edd57e2ebae71a8949e09857150c436d0.jpg)

![265f7bb628099f2fc0064dc175c892cc7d107d293d2ab0a0bdb9b8ad94df46de.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/265f7bb628099f2fc0064dc175c892cc7d107d293d2ab0a0bdb9b8ad94df46de.jpg)

![30a50c7dd773a4fddc7bcc508a45d9a1045a869d77100d200829b2694088655e.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/30a50c7dd773a4fddc7bcc508a45d9a1045a869d77100d200829b2694088655e.jpg)

![4ee8bea456f0cdd8d2a563de25d8554daa00fcf195fcb9f2d20e542ee2ac6fc9.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/4ee8bea456f0cdd8d2a563de25d8554daa00fcf195fcb9f2d20e542ee2ac6fc9.jpg)

![60f788f6487aa3224884ae00698e117af217e385b12a716527706a6a464596c5.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/60f788f6487aa3224884ae00698e117af217e385b12a716527706a6a464596c5.jpg)

![76437b344fadee97402d94f9415978498bd0890ba236569e2b61c53ddd0c4aee.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/76437b344fadee97402d94f9415978498bd0890ba236569e2b61c53ddd0c4aee.jpg)

![94c3c7615aa5802fdeb9f8e0a9648597f6ec52b1cfa77eeae1e600dc4cea5513.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/94c3c7615aa5802fdeb9f8e0a9648597f6ec52b1cfa77eeae1e600dc4cea5513.jpg)

![a58db4f7c729ebe60524ef8bde9e62b550a139de55b316ffea75efffffd706e8.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/a58db4f7c729ebe60524ef8bde9e62b550a139de55b316ffea75efffffd706e8.jpg)

![abe65a193e80db7fc80f3a1f825b3ada801858c20f2d58b2b54b34317be9ca56.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/abe65a193e80db7fc80f3a1f825b3ada801858c20f2d58b2b54b34317be9ca56.jpg)

![bce6c7fcb7875eea0e4b797358c8db9ae3ddf98550deb7b8c6bf3b303b252a6b.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/bce6c7fcb7875eea0e4b797358c8db9ae3ddf98550deb7b8c6bf3b303b252a6b.jpg)

![f01ec8a7b962d8d6270c09d63a2cd873b57540af0a0fab97fc6056fa6c6db675.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/f01ec8a7b962d8d6270c09d63a2cd873b57540af0a0fab97fc6056fa6c6db675.jpg)

![f6c1642c8288b0e27a5a2d800fe56d51d71c4f0d519a2c43447fbc693e29fee3.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/f6c1642c8288b0e27a5a2d800fe56d51d71c4f0d519a2c43447fbc693e29fee3.jpg)

### Tables

![0a9df3ba5175d0484af180746baf746a052e2372a92824c9751240ee28f247f1.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/0a9df3ba5175d0484af180746baf746a052e2372a92824c9751240ee28f247f1.jpg)

![4bc534a291577d5c2ddc1246cee13f21b0eed4be4084127310939eb734db29b8.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/4bc534a291577d5c2ddc1246cee13f21b0eed4be4084127310939eb734db29b8.jpg)

![4be235addd46f76af697117bffb804852115a22daa9a45866d2266efd6705fdc.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/4be235addd46f76af697117bffb804852115a22daa9a45866d2266efd6705fdc.jpg)

![5170f7302018d13fad8ac5d712cb5456534eaeab7cd4138ac8715f9bdd9c0f21.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/5170f7302018d13fad8ac5d712cb5456534eaeab7cd4138ac8715f9bdd9c0f21.jpg)

![51a0154fb43d62d091cb2a2bbbeb8fcf1b6b31b173ff7fc14e6dcd55651b568d.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/51a0154fb43d62d091cb2a2bbbeb8fcf1b6b31b173ff7fc14e6dcd55651b568d.jpg)

![61b63fb28ffab00f79ce38ae258c1b4343a0b9e9a72b6ad63a4eebfbdb16a938.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/61b63fb28ffab00f79ce38ae258c1b4343a0b9e9a72b6ad63a4eebfbdb16a938.jpg)

![67c230d3679bda63006d2db5244cfaa5d64ed5d7593cea32d4ed0d9b9b76c204.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/67c230d3679bda63006d2db5244cfaa5d64ed5d7593cea32d4ed0d9b9b76c204.jpg)

![9a26ba235318735f4e730e09b1a7b34f736b15b6eeafab96c392b46679da6dbd.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/9a26ba235318735f4e730e09b1a7b34f736b15b6eeafab96c392b46679da6dbd.jpg)

![e4a261a03f87021ceb434e0166c5f12878bf2d117cf149356e3ba6c40aba5f8a.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/e4a261a03f87021ceb434e0166c5f12878bf2d117cf149356e3ba6c40aba5f8a.jpg)
