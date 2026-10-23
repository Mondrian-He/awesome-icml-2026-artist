# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 109 of 197

## 目录 (Table of Contents)

1. [WS-GRPO: Weakly-Supervised Group-Relative Policy Optimization for Rollout-Efficient Reasoning](#paper-3565)
2. [LAPRAS : Learning-Augmented PRivate Answering for linear query Streams.](#paper-3566)
3. [Scaling Laws and Architectural Frontiers in Metagenomic Foundation Models](#paper-3567)
4. [Terminal Dimension Reduction for Time Series with Applications](#paper-3568)
5. [Cluster-Aware Causal Mixer for Online Anomaly Detection in Multivariate Time Series](#paper-3569)
6. [GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](#paper-3570)
7. [Discovering Scaling Exponents with Physics-Informed Müntz-Szász Networks](#paper-3571)
8. [Interleaved Selective State Space Models for Efficient WiFi-Based 3D Multi-Person Pose Estimation](#paper-3572)
9. [ITSPACE: Monotone Gaussian Optimal Transport Updates](#paper-3573)
10. [Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models](#paper-3574)
11. [Hierarchical Causal Abduction: A Foundation Framework for Explainable Model Predictive Control](#paper-3575)
12. [How does Chain of Thought decompose complex tasks?](#paper-3576)
13. [Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent](#paper-3577)
14. [TRIM: Token-wise Attention-Derived Saliency for Data-Efficient Instruction Tuning](#paper-3578)
15. [Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories](#paper-3579)
16. [Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](#paper-3580)
17. [Interpreting Genomic Language Models using Sparse Autoencoders](#paper-3581)
18. [Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining](#paper-3582)
19. [Aitchison Embeddings for Learning Compositional Graph Representations](#paper-3583)
20. [FOAM: Frequency and Operator-Error Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo](#paper-3584)
21. [Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data](#paper-3585)
22. [Causal-JEPA: Learning World Models through Object-Level Latent Masking](#paper-3586)
23. [The Invisible Lottery: How Subtle Cues Steer Algorithm Choice in LLM Code Generation](#paper-3587)
24. [Telescope: Improving Zero Shot Detection of LLM Generated Content By Measuring Token Repetition Probability](#paper-3588)
25. [One Probe Won’t Catch Them All: Towards Targeted Deception Detection](#paper-3589)
26. [Learning Dynamic Stability Landscapes in Synchronization Networks](#paper-3590)
27. [Adaptive Node Feature Selection for Graph Neural Networks](#paper-3591)
28. [A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn’t)](#paper-3592)
29. [Achieving Logarithmic Regret in KL-Regularized Zero-Sum Markov Games](#paper-3593)
30. [CADFit: Precise Mesh-to-CAD Program Generation with Hybrid Optimization](#paper-3594)
31. [LAMP: Data-Efficient Linear Affine Weight-Space Models for Parameter-Controlled 3D Shape Generation and Extrapolation](#paper-3595)
32. [Universal Representation of Generalized Convex Functions and their Gradients](#paper-3596)
33. [Toward Identifiable Sparse Autoencoders](#paper-3597)

---


<a id="paper-3565"></a>

## WS-GRPO: Weakly-Supervised Group-Relative Policy Optimization for Rollout-Efficient Reasoning

### Images

![2a857d7ee0a40426864a9a3c23db61c00e0f190f49b0bfeff194b72f9e92da75.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/2a857d7ee0a40426864a9a3c23db61c00e0f190f49b0bfeff194b72f9e92da75.jpg)

![6932deab912b205e15423a228f930b4455cf4be8bc3a67ce078fd19075226e71.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/6932deab912b205e15423a228f930b4455cf4be8bc3a67ce078fd19075226e71.jpg)

![98c360d93cc4b0ad339728895a6e0ccf1ff32c475f794f2d2930d9a1a9baef29.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/98c360d93cc4b0ad339728895a6e0ccf1ff32c475f794f2d2930d9a1a9baef29.jpg)

![a37f8719b8d4a5295c5e7a736e8d37062593992ae967cf9125e959527363bc4c.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/a37f8719b8d4a5295c5e7a736e8d37062593992ae967cf9125e959527363bc4c.jpg)

![ac42c15b8d404b5d189be53afb6bd85504a2ca67dffaa5cb5fa85b6d3889d8c7.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/ac42c15b8d404b5d189be53afb6bd85504a2ca67dffaa5cb5fa85b6d3889d8c7.jpg)

![b0587e155b4babb566c822700ef22630815b5cb27b8fe0de4d5ed9628aaf8e4e.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/b0587e155b4babb566c822700ef22630815b5cb27b8fe0de4d5ed9628aaf8e4e.jpg)

![cd2729d8f8cf1f7699236e17c69274a0104ad11206c6705be9243e4144949419.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/cd2729d8f8cf1f7699236e17c69274a0104ad11206c6705be9243e4144949419.jpg)

![d07b58891f876c7de73406b54e14b1c8cc5700b75b30f474f1a388f5ef6342d4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/d07b58891f876c7de73406b54e14b1c8cc5700b75b30f474f1a388f5ef6342d4.jpg)

![e0c7bba5cf4c28d58d66aa710bfdab3db445249bafb48eaeaadfa1a2bdc8203b.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/e0c7bba5cf4c28d58d66aa710bfdab3db445249bafb48eaeaadfa1a2bdc8203b.jpg)

![e2a31192cf4d62ce70d704ed1298397c85171cf819f17eab5b848dd3b4bb32dc.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/e2a31192cf4d62ce70d704ed1298397c85171cf819f17eab5b848dd3b4bb32dc.jpg)

### Tables

![0dcdb8a2d04a5ab22e5bbb69845b4dc23d2716e841acccc2ea626ad23e278c10.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/0dcdb8a2d04a5ab22e5bbb69845b4dc23d2716e841acccc2ea626ad23e278c10.jpg)

![2658302e66e85d572d43c5f20cc64d61a714b50c299ac5bed8b9e1849a8a7df0.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/2658302e66e85d572d43c5f20cc64d61a714b50c299ac5bed8b9e1849a8a7df0.jpg)

![30451beb988f5632e3b89f0761fad7b0973e5955c4706c51d25cd8e618db91c4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/30451beb988f5632e3b89f0761fad7b0973e5955c4706c51d25cd8e618db91c4.jpg)

![3205fe5fcf4944c4e84aa708e8c0c069fe29958428167a1c5553fb4f2e8d3e5a.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/3205fe5fcf4944c4e84aa708e8c0c069fe29958428167a1c5553fb4f2e8d3e5a.jpg)

![4195996f0b03dc356e8414722e5e7278593556fb47553c05daa4a752533ec0a0.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4195996f0b03dc356e8414722e5e7278593556fb47553c05daa4a752533ec0a0.jpg)

![4cb346f478ea87b9caa82636cd445831b55b4d632489494483e7409e1fee630b.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4cb346f478ea87b9caa82636cd445831b55b4d632489494483e7409e1fee630b.jpg)

![4d6c92bef348c671fd8ecb6d9ef48453ac91205508571a285db67d8a107ca147.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4d6c92bef348c671fd8ecb6d9ef48453ac91205508571a285db67d8a107ca147.jpg)

![554d0cd0ba275af1dc8c734ccfd1277a4bd192c8a3065f9310b020ec93b65726.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/554d0cd0ba275af1dc8c734ccfd1277a4bd192c8a3065f9310b020ec93b65726.jpg)

![64276e3842d0c5011ea44f13b614b5d26074f4d4dfa8a8f79985f86b53f0b70e.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/64276e3842d0c5011ea44f13b614b5d26074f4d4dfa8a8f79985f86b53f0b70e.jpg)

![6fa31e5fb339fc0abfa6ca245e40bde1f4555d0e27a4bb5d93f1538f2bcc7ae2.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/6fa31e5fb339fc0abfa6ca245e40bde1f4555d0e27a4bb5d93f1538f2bcc7ae2.jpg)

![7113a649f47dda577fe18bc9c3733dca7d1fc85bcd91da6d5f49a19f91570498.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/7113a649f47dda577fe18bc9c3733dca7d1fc85bcd91da6d5f49a19f91570498.jpg)

![7b0aa4602d31e0185512f94a32739a27ab508e13867f86210affd7f61b0998bf.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/7b0aa4602d31e0185512f94a32739a27ab508e13867f86210affd7f61b0998bf.jpg)

![9bc0991cc9b2e859f704a8747f44e2643ebf07a9b4aa403db1d4b7b68bd83ddc.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/9bc0991cc9b2e859f704a8747f44e2643ebf07a9b4aa403db1d4b7b68bd83ddc.jpg)

![9f2891d62ee3175a0ba0d71aa586705d747bdbb9eb11c9ef90c804c731c19687.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/9f2891d62ee3175a0ba0d71aa586705d747bdbb9eb11c9ef90c804c731c19687.jpg)

![abf2e366b2f37299f8ffa004bbe957fce2fdb32bfa3d0ce8dee9e7d58dc09df9.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/abf2e366b2f37299f8ffa004bbe957fce2fdb32bfa3d0ce8dee9e7d58dc09df9.jpg)

![acb0c78abcef03e17b788e13096e66882b5b37a4ae32eeda54a0850c54b770df.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/acb0c78abcef03e17b788e13096e66882b5b37a4ae32eeda54a0850c54b770df.jpg)

![acc624e8d1a30c3f65579663838b33484f92d5f6cab6811999f44ab443f1f564.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/acc624e8d1a30c3f65579663838b33484f92d5f6cab6811999f44ab443f1f564.jpg)

![b03ff99d3f78e63f359d4e5deafef83c7955977dcfedc9cb67fae1ae85c51543.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b03ff99d3f78e63f359d4e5deafef83c7955977dcfedc9cb67fae1ae85c51543.jpg)

![b0ad46a6cd79bac3450a20f75692ec115ad2ee43180708cccd9ce133535ac196.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b0ad46a6cd79bac3450a20f75692ec115ad2ee43180708cccd9ce133535ac196.jpg)

![b123b3f18f168b6a040344b9a3aadf9f6582986f109cd927fe6ff7d81c2521e8.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b123b3f18f168b6a040344b9a3aadf9f6582986f109cd927fe6ff7d81c2521e8.jpg)

![b5574c39e81841a5f4f1f95902ccd4230ec2d4de34130195a95d7fec93cdd3ff.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b5574c39e81841a5f4f1f95902ccd4230ec2d4de34130195a95d7fec93cdd3ff.jpg)

![bfb03a9ab1b3b5af532f82f87b24fef4dbdea663ed05500db50e28d370d1b1b4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/bfb03a9ab1b3b5af532f82f87b24fef4dbdea663ed05500db50e28d370d1b1b4.jpg)

![c5f8190db7d0ad50afd93f4325f0ec9001059dba78fcc405b90220b915cde50c.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/c5f8190db7d0ad50afd93f4325f0ec9001059dba78fcc405b90220b915cde50c.jpg)

![d2d81989eb882c6db8a6c7658130d9ee6c6df34bb255c4ca80416df7816015c6.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/d2d81989eb882c6db8a6c7658130d9ee6c6df34bb255c4ca80416df7816015c6.jpg)

![ec2044678b90b7a3abb1e9939d9523e6a3e06d6293bb2d0f66a4862745854c40.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/ec2044678b90b7a3abb1e9939d9523e6a3e06d6293bb2d0f66a4862745854c40.jpg)

![f5caa8992afa8e9bf30f9d14a26193f1c0b7cb06ccb470ff5f93b3b1f9a36d3a.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/f5caa8992afa8e9bf30f9d14a26193f1c0b7cb06ccb470ff5f93b3b1f9a36d3a.jpg)

![f9bbac40e42acaa0e18b7f7c25dcd61c8c64d5bceadc709619dc9290af0bb219.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/f9bbac40e42acaa0e18b7f7c25dcd61c8c64d5bceadc709619dc9290af0bb219.jpg)

## WS-GRPO: Weakly-Supervised Group-Relative Policy Optimization for Rollout-Efficient Reasoning


### Images

![44a1c17b28a0adbcfc32908b03357e743b51a89097f902fbed482d6393bf5b07.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/44a1c17b28a0adbcfc32908b03357e743b51a89097f902fbed482d6393bf5b07.jpg)

![65d058f936511a368fff895b7295fd75d4840d5ea8a2f202ddabba2a49498e28.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/65d058f936511a368fff895b7295fd75d4840d5ea8a2f202ddabba2a49498e28.jpg)

![8863c80622df1e4cc627df408786a282c849e6f0993591b7fecd1931ef8aeb79.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/8863c80622df1e4cc627df408786a282c849e6f0993591b7fecd1931ef8aeb79.jpg)

![9319b0af6725808e6419504e6d571dde69500cd420396466829b73dad4f9dc6b.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/9319b0af6725808e6419504e6d571dde69500cd420396466829b73dad4f9dc6b.jpg)

![9dd3d8cc928b0d41a37193aabd08d6a20e82cc5dab8ba73b1e14ea9c3b7f2eeb.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/9dd3d8cc928b0d41a37193aabd08d6a20e82cc5dab8ba73b1e14ea9c3b7f2eeb.jpg)

![a87dd81e3189787f3a1c0fe687172e34722acc2d485c52980499096099c5bed1.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/a87dd81e3189787f3a1c0fe687172e34722acc2d485c52980499096099c5bed1.jpg)

![a9f5c14f8b484e160d7fec889355e4a7f96d1329291fa3f3a9e9465f4437ef9d.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/a9f5c14f8b484e160d7fec889355e4a7f96d1329291fa3f3a9e9465f4437ef9d.jpg)

![cb3bf14afde973f53add9970cc57e804e4be3da46e85cc7650670b4cba3e6abc.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/cb3bf14afde973f53add9970cc57e804e4be3da46e85cc7650670b4cba3e6abc.jpg)

![de700b861ae0a2a5847931aa9f8448e3615d6c206f526032ed05d490fc533af6.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/de700b861ae0a2a5847931aa9f8448e3615d6c206f526032ed05d490fc533af6.jpg)

![df22bce955cbc0b9965f26f74f61cb9304a094594dba4b8182d2285e0f9e2560.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/images/df22bce955cbc0b9965f26f74f61cb9304a094594dba4b8182d2285e0f9e2560.jpg)

### Tables

![19bca8e933f7579c583d18fb9bb86075f8b948c894bd835ca9b127219ec79c2f.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/19bca8e933f7579c583d18fb9bb86075f8b948c894bd835ca9b127219ec79c2f.jpg)

![692f5cb4cf8894b8d65ba93c002534d83dc39576e91539a5791a2b17cce10ef8.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/692f5cb4cf8894b8d65ba93c002534d83dc39576e91539a5791a2b17cce10ef8.jpg)

![ad909b4f26c4bf9306a9a58bef6e25f2897d678fca1a9dbd5e34e63967889b5f.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/ad909b4f26c4bf9306a9a58bef6e25f2897d678fca1a9dbd5e34e63967889b5f.jpg)

![b501a4bef081136726065a647b67f1a23e07def1ad4a61cf4ba360e4759ab813.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/b501a4bef081136726065a647b67f1a23e07def1ad4a61cf4ba360e4759ab813.jpg)

![c25bfa8c63f06882310babd241d7e1178794b8b2025e1e5b0e88b16061a5c5b2.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/c25bfa8c63f06882310babd241d7e1178794b8b2025e1e5b0e88b16061a5c5b2.jpg)

![e17999e2f3ba9c2e0eb16799d9ad47c1acbfb8acfb23e345c2339b1b05f9deba.jpg](../icml_results/3605_WS-GRPO_%20Weakly-Supervised%20Group-Relative%20Policy%20Optimization%20for%20Rollout-Efficient%20Reasoning/tables/e17999e2f3ba9c2e0eb16799d9ad47c1acbfb8acfb23e345c2339b1b05f9deba.jpg)

<a id="paper-3566"></a>

## LAPRAS : Learning-Augmented PRivate Answering for linear query Streams.


### Images

![44c9d347d26a209200419f10b723f87bcf2479ae696f2bef7dba95c7a384fcb6.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./images/44c9d347d26a209200419f10b723f87bcf2479ae696f2bef7dba95c7a384fcb6.jpg)

![bde2d021e87f690f67b77de825ee390a4a17c99a0e2539fe550e8ae2260b8d54.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./images/bde2d021e87f690f67b77de825ee390a4a17c99a0e2539fe550e8ae2260b8d54.jpg)

### Tables

![0727de6401adc0c78987b784a1b01b01ca24e92843ed09cdfae6f4ff4349b0e4.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/0727de6401adc0c78987b784a1b01b01ca24e92843ed09cdfae6f4ff4349b0e4.jpg)

![839d242144f81d5d431bfdce1b613187fac2bf731247206c041009567ca0980e.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/839d242144f81d5d431bfdce1b613187fac2bf731247206c041009567ca0980e.jpg)

![84e302b4b714d3bf7a0b0bfa6888670e2f9d426a0028ca5342b8a467d68b65ff.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/84e302b4b714d3bf7a0b0bfa6888670e2f9d426a0028ca5342b8a467d68b65ff.jpg)

![b000d9e7d521c9e0368df394685ab5427a892542865f082f6773ca490d02b0b0.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/b000d9e7d521c9e0368df394685ab5427a892542865f082f6773ca490d02b0b0.jpg)

![c02db4c7683dbfa8af01a1391b1ffc2bf5732681871c0074def7621f44f56bc1.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/c02db4c7683dbfa8af01a1391b1ffc2bf5732681871c0074def7621f44f56bc1.jpg)

![d2f2ebae6459f54c2b238cd50d666f3fd1b822607fd1ab1d44ce0b0c1b690f1e.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/d2f2ebae6459f54c2b238cd50d666f3fd1b822607fd1ab1d44ce0b0c1b690f1e.jpg)

![d5c4376b60a15cc48c03936e9551d30f8a6df3b748999f81618fa87377eddbef.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/d5c4376b60a15cc48c03936e9551d30f8a6df3b748999f81618fa87377eddbef.jpg)

![e54895c4dc91da207208f2b190a651b85ac421c8b04550a61f5d754297e75dff.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/e54895c4dc91da207208f2b190a651b85ac421c8b04550a61f5d754297e75dff.jpg)

![f232a411a5b87779a3b4c89aa1cbf01fedd93095a6826eb10d12263d62e4a16a.jpg](../icml_results/3606_LAPRAS%20_%20Learning-Augmented%20PRivate%20Answering%20for%20linear%20query%20Streams./tables/f232a411a5b87779a3b4c89aa1cbf01fedd93095a6826eb10d12263d62e4a16a.jpg)

<a id="paper-3567"></a>

## Scaling Laws and Architectural Frontiers in Metagenomic Foundation Models


### Images

![14da0fcc53c00fac7b3ecad4591b87d2f3822ec60213ce78d28e03618a5d9fd8.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/14da0fcc53c00fac7b3ecad4591b87d2f3822ec60213ce78d28e03618a5d9fd8.jpg)

![2a60522d85078adc2768bff59ea61bef3d469ae8b18165bd8c882bf6bb4df2a6.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/2a60522d85078adc2768bff59ea61bef3d469ae8b18165bd8c882bf6bb4df2a6.jpg)

![40c31cbdf00078343b7c072d0fbd01e10e7bc90bb4ae98c889cfea293a18b796.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/40c31cbdf00078343b7c072d0fbd01e10e7bc90bb4ae98c889cfea293a18b796.jpg)

![5e43c525c321a1e32555e6c8298e508e792fb15e487a88bb83a86ee33783b7c1.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/5e43c525c321a1e32555e6c8298e508e792fb15e487a88bb83a86ee33783b7c1.jpg)

![83aad00d8acccc739765cce51f896ac4aebcd8c3cd2d90718b9a5220119b74ad.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/83aad00d8acccc739765cce51f896ac4aebcd8c3cd2d90718b9a5220119b74ad.jpg)

![93b1027d26b1eb9d57448cae97a2ec84c4f73418d61e6d004a88bc9116928f4b.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/93b1027d26b1eb9d57448cae97a2ec84c4f73418d61e6d004a88bc9116928f4b.jpg)

![a83328358e13a9d8057b2b272e2b1a5d26d067a0c49ec99824cf3ea89fb11064.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/a83328358e13a9d8057b2b272e2b1a5d26d067a0c49ec99824cf3ea89fb11064.jpg)

![c3ab7cca828b863fd119dbae8fddefc5449fb6a7ac67510676f4593ac11b48b0.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/c3ab7cca828b863fd119dbae8fddefc5449fb6a7ac67510676f4593ac11b48b0.jpg)

![c8e8a3be4d43642cf4e71c48571c5db2424fa476fc323de07e4aa372367d1d9f.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/c8e8a3be4d43642cf4e71c48571c5db2424fa476fc323de07e4aa372367d1d9f.jpg)

![cbac17ee9b8531317f51d588ab0d6ec7edcdb4e4f72fd7cd80a7973746cb6e05.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/cbac17ee9b8531317f51d588ab0d6ec7edcdb4e4f72fd7cd80a7973746cb6e05.jpg)

![d5f9183bfedb0fed9647e1cba2299485c95c2d01038b09ff73f7c22fcdc77d7a.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/d5f9183bfedb0fed9647e1cba2299485c95c2d01038b09ff73f7c22fcdc77d7a.jpg)

![da4aed130156291e74eafbd9c787a0361063e9c204afc7ed7759ea6b70cdfe43.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/da4aed130156291e74eafbd9c787a0361063e9c204afc7ed7759ea6b70cdfe43.jpg)

![ed11247169fbda6063d33411b3ceb550eca827f8342baab986e5e8f1fb43ec3b.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/ed11247169fbda6063d33411b3ceb550eca827f8342baab986e5e8f1fb43ec3b.jpg)

![f7b1831dddf6ea11a223c81866c2b38b9329a9448b263af6171921d486cd5f9c.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/f7b1831dddf6ea11a223c81866c2b38b9329a9448b263af6171921d486cd5f9c.jpg)

![f9ffcc3abb326387744a1f6719ff3bb1d3ca14cd9ccdf23517bc5dc48b0c8902.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/images/f9ffcc3abb326387744a1f6719ff3bb1d3ca14cd9ccdf23517bc5dc48b0c8902.jpg)

### Tables

![043ab1e03785c4b980b58c2f104f97dd32a049b81d5eaa0014b7417469db0204.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/043ab1e03785c4b980b58c2f104f97dd32a049b81d5eaa0014b7417469db0204.jpg)

![3b7b6e88db0baa2ef8c8ee0f4ab817bca7b90d67036eee563d2fd52cfd0c0fe4.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/3b7b6e88db0baa2ef8c8ee0f4ab817bca7b90d67036eee563d2fd52cfd0c0fe4.jpg)

![3d5d44b9214a7793ec28e0e9d0bdc37226a23ad644cf6466e005405a9551b808.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/3d5d44b9214a7793ec28e0e9d0bdc37226a23ad644cf6466e005405a9551b808.jpg)

![aba1268eca924d70a9a10595017c5121bc61cb50691b2aa84e0a2237be18b845.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/aba1268eca924d70a9a10595017c5121bc61cb50691b2aa84e0a2237be18b845.jpg)

![c95b8e1941f180ccbd5be4fda20324f023a9d299ad3a8c557a176220b83579fc.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/c95b8e1941f180ccbd5be4fda20324f023a9d299ad3a8c557a176220b83579fc.jpg)

![d4dc3269b86ef9949f2ad792a8ec21fee73cfa415a9e4d8b89d3f07725bc71f8.jpg](../icml_results/3607_Scaling%20Laws%20and%20Architectural%20Frontiers%20in%20Metagenomic%20Foundation%20Models/tables/d4dc3269b86ef9949f2ad792a8ec21fee73cfa415a9e4d8b89d3f07725bc71f8.jpg)

<a id="paper-3568"></a>

## Terminal Dimension Reduction for Time Series with Applications


### Images

![2897c213656c2ea69c569c95c653d9db3b50f4389163fd763fdf11bb778f8cb9.jpg](../icml_results/3608_Terminal%20Dimension%20Reduction%20for%20Time%20Series%20with%20Applications/images/2897c213656c2ea69c569c95c653d9db3b50f4389163fd763fdf11bb778f8cb9.jpg)

![2cf7bf7c06ded8a48c988a33c8870cc70925d4f77cea47d682311004d39dc2ae.jpg](../icml_results/3608_Terminal%20Dimension%20Reduction%20for%20Time%20Series%20with%20Applications/images/2cf7bf7c06ded8a48c988a33c8870cc70925d4f77cea47d682311004d39dc2ae.jpg)

![9076a4e9839a4f67d153f7f7977d2efe17a1761c5a4db4f83435e43c188ba529.jpg](../icml_results/3608_Terminal%20Dimension%20Reduction%20for%20Time%20Series%20with%20Applications/images/9076a4e9839a4f67d153f7f7977d2efe17a1761c5a4db4f83435e43c188ba529.jpg)

![eb2bcf1e54ecb64ef5ed1e48c609537e93446911b0a0fd2ea3b27e5c29632afe.jpg](../icml_results/3608_Terminal%20Dimension%20Reduction%20for%20Time%20Series%20with%20Applications/images/eb2bcf1e54ecb64ef5ed1e48c609537e93446911b0a0fd2ea3b27e5c29632afe.jpg)

### Tables

![890f5416f3abb24ea6c502fa016a1c75c1cb617c0475288760032326adbd30cb.jpg](../icml_results/3608_Terminal%20Dimension%20Reduction%20for%20Time%20Series%20with%20Applications/tables/890f5416f3abb24ea6c502fa016a1c75c1cb617c0475288760032326adbd30cb.jpg)

<a id="paper-3569"></a>

## Cluster-Aware Causal Mixer for Online Anomaly Detection in Multivariate Time Series


### Images

![336d22d14106e9c003e2c52d40fa04c5eb88ddf32a0c750a935c5475116d3bd1.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/336d22d14106e9c003e2c52d40fa04c5eb88ddf32a0c750a935c5475116d3bd1.jpg)

![3dd7bbf911e63d5d06bb91e1556cc50647662bb0596c4189f82d72756f2a52e2.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/3dd7bbf911e63d5d06bb91e1556cc50647662bb0596c4189f82d72756f2a52e2.jpg)

![4ebb8ad65a61b46624cd39157540590824f15ae8d9dcdb0504641adbe42a18d8.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/4ebb8ad65a61b46624cd39157540590824f15ae8d9dcdb0504641adbe42a18d8.jpg)

![740811aa94dd0b419e08b8bb8d44125c98505e53c713ce0aa8a034d39e4c8258.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/740811aa94dd0b419e08b8bb8d44125c98505e53c713ce0aa8a034d39e4c8258.jpg)

![7f118dbeedda00c890432f0b2a8236fa2e744d005fa7bf8ac9883918c141f062.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/7f118dbeedda00c890432f0b2a8236fa2e744d005fa7bf8ac9883918c141f062.jpg)

![83a207996a286b1f555045de8e780480be78239437d6f30bc180363e12550121.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/83a207996a286b1f555045de8e780480be78239437d6f30bc180363e12550121.jpg)

![87a0211af8094f18b0aed09b9f191b328c6cad37b4db5a53bcd02dfaa9ad1fc8.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/87a0211af8094f18b0aed09b9f191b328c6cad37b4db5a53bcd02dfaa9ad1fc8.jpg)

![a71d19b30980fc53e4a8366eaed3ca8645feb1a127f00f3238f213e2c06fa66b.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/a71d19b30980fc53e4a8366eaed3ca8645feb1a127f00f3238f213e2c06fa66b.jpg)

![be08845e26c55e34a478484acd1790bbcc97863b54909b95e23a4586a8917c30.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/be08845e26c55e34a478484acd1790bbcc97863b54909b95e23a4586a8917c30.jpg)

![c939f1e72683b5f3323c4fed4f8d8a131d50b5fb87ce72e87243686b231c9050.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/c939f1e72683b5f3323c4fed4f8d8a131d50b5fb87ce72e87243686b231c9050.jpg)

![d04704125c8ec5e6d67776884c9884a1a4effa017b402b7dca156f5fedc7937d.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/d04704125c8ec5e6d67776884c9884a1a4effa017b402b7dca156f5fedc7937d.jpg)

![d16878e6ef93ad8fc738650e29b146efb0d96065d58174fdb95a8dff5d5c280a.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/d16878e6ef93ad8fc738650e29b146efb0d96065d58174fdb95a8dff5d5c280a.jpg)

![d2b126e1342d543ac242df6790f6192551fb66786f84e7bd2cbd8d1fb7d7a784.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/d2b126e1342d543ac242df6790f6192551fb66786f84e7bd2cbd8d1fb7d7a784.jpg)

![d65d4920d8216d1d14430977f72237dd30c2629696712813793320c418fdeb27.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/d65d4920d8216d1d14430977f72237dd30c2629696712813793320c418fdeb27.jpg)

![fe100cad7daef60fc0cfb149164ed196e4d2192d5454274157a4429a66b5d55a.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/images/fe100cad7daef60fc0cfb149164ed196e4d2192d5454274157a4429a66b5d55a.jpg)

### Tables

![176f682d0f1f39b20be47fc5824ea0dc3808405f4454efd4a6c5424c0dc7bd33.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/176f682d0f1f39b20be47fc5824ea0dc3808405f4454efd4a6c5424c0dc7bd33.jpg)

![1be458242c8fc067cb22c6d61a0b47408ffebeebdc828508233780901730adc6.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/1be458242c8fc067cb22c6d61a0b47408ffebeebdc828508233780901730adc6.jpg)

![1f9e13acef330548c881187c039cccd70c7ff29e4b6c0525160838c54cc70132.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/1f9e13acef330548c881187c039cccd70c7ff29e4b6c0525160838c54cc70132.jpg)

![2b3d4b082807a9c49a289fecd7f7d8e1f03cb6d583175ad4100407006bb35402.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/2b3d4b082807a9c49a289fecd7f7d8e1f03cb6d583175ad4100407006bb35402.jpg)

![49246301c3626d9550ea480500f04d074aef9033ce0460739428f828d790f101.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/49246301c3626d9550ea480500f04d074aef9033ce0460739428f828d790f101.jpg)

![734d867f9d3ff3decf5bdabf4fff5efdd1ab5b128e63c72c0f87f501248e0026.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/734d867f9d3ff3decf5bdabf4fff5efdd1ab5b128e63c72c0f87f501248e0026.jpg)

![81ca2ddef176652787b231b459cfa18a98c6265484e690fb5774541e39f14cac.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/81ca2ddef176652787b231b459cfa18a98c6265484e690fb5774541e39f14cac.jpg)

![8f86c1e12d7ffcde4aa81d99e69dd66532a2266538d70cfcb5f38ef555faf46a.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/8f86c1e12d7ffcde4aa81d99e69dd66532a2266538d70cfcb5f38ef555faf46a.jpg)

![b7bc0b3303852ec7201d736d80a8c64a49757ef51d165f548cd004117d1f7171.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/b7bc0b3303852ec7201d736d80a8c64a49757ef51d165f548cd004117d1f7171.jpg)

![bdf604bafcef8f6f83889874487333bb3d8071268525781b70c40034063e197b.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/bdf604bafcef8f6f83889874487333bb3d8071268525781b70c40034063e197b.jpg)

![d4a1c1e91c91aaa63bebcebe4ea0facbde94637153942c37f6416d4c55cbeca8.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/d4a1c1e91c91aaa63bebcebe4ea0facbde94637153942c37f6416d4c55cbeca8.jpg)

![d4ecb85e096bdb892576b6ed910c54e9da6db9e35b146d0af54d93c9ad75530b.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/d4ecb85e096bdb892576b6ed910c54e9da6db9e35b146d0af54d93c9ad75530b.jpg)

![dc56d117efa20549b2391f383d898d48263206cb60848b7f45e4c774ef9ae829.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/dc56d117efa20549b2391f383d898d48263206cb60848b7f45e4c774ef9ae829.jpg)

![e5e432efc33cf9558fec8e1111d1b48ec93292866473737e5f1ab6751b0beb31.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/e5e432efc33cf9558fec8e1111d1b48ec93292866473737e5f1ab6751b0beb31.jpg)

![f58f27504554b488eeba63c0660c492e34618b007b8ef5481d7ce731f47f081c.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/f58f27504554b488eeba63c0660c492e34618b007b8ef5481d7ce731f47f081c.jpg)

![fbd989689297be66de25e668ec142301b730d8c77eecec4ebff5a2f8d5f462a7.jpg](../icml_results/3609_Cluster-Aware%20Causal%20Mixer%20for%20Online%20Anomaly%20Detection%20in%20Multivariate%20Time%20Series/tables/fbd989689297be66de25e668ec142301b730d8c77eecec4ebff5a2f8d5f462a7.jpg)

<a id="paper-3570"></a>

## GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning


### Images

![35c2f340cd5748f3b8f431e480d9cd828472c665e0058eb9d0981380ddcc48e7.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/images/35c2f340cd5748f3b8f431e480d9cd828472c665e0058eb9d0981380ddcc48e7.jpg)

![557e434e1d2e9173619dfde45fa3d27d68eac0ac533f4f136c08b1bd371fe9ff.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/images/557e434e1d2e9173619dfde45fa3d27d68eac0ac533f4f136c08b1bd371fe9ff.jpg)

![786cc58df011f708503d10835fc5c3d811a96ae382f8f913081bd35fa4a3f225.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/images/786cc58df011f708503d10835fc5c3d811a96ae382f8f913081bd35fa4a3f225.jpg)

![8cb00064fc812ba78880dfe1a850ddd2c2d5c8feb6349cb163fba381d9b556d7.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/images/8cb00064fc812ba78880dfe1a850ddd2c2d5c8feb6349cb163fba381d9b556d7.jpg)

![9c5f015d4be73ae62f053458d21e63fe5ac74d54df4d227e236af008f18e51ae.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/images/9c5f015d4be73ae62f053458d21e63fe5ac74d54df4d227e236af008f18e51ae.jpg)

### Tables

![26fb8cb27c2e8d8fad618c3119f3a72dc8ae5ad8e82f8a7e6962ae560d25a0e6.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/26fb8cb27c2e8d8fad618c3119f3a72dc8ae5ad8e82f8a7e6962ae560d25a0e6.jpg)

![43eeb28bae72033d746f7c84edb94822c25829726a9f2cd577951882718b219f.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/43eeb28bae72033d746f7c84edb94822c25829726a9f2cd577951882718b219f.jpg)

![4add4e04e9d6b0f40e2cb1037159c5fe5db837f9c5e1b7150d3329a3b5413c03.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/4add4e04e9d6b0f40e2cb1037159c5fe5db837f9c5e1b7150d3329a3b5413c03.jpg)

![551fb13348ff483508f651ae6e2f6a33a12e5b46439582ee5c268daa7eaa6cd9.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/551fb13348ff483508f651ae6e2f6a33a12e5b46439582ee5c268daa7eaa6cd9.jpg)

![59f76117c47da186153a2f3baf2b3130e813aec8bde83add83e1bc4ac0eb8a6d.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/59f76117c47da186153a2f3baf2b3130e813aec8bde83add83e1bc4ac0eb8a6d.jpg)

![62b1aa8d1e94ce6047c9f143aee8d15d9efe8158da3a93fc82de212561d5b670.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/62b1aa8d1e94ce6047c9f143aee8d15d9efe8158da3a93fc82de212561d5b670.jpg)

![673877572d79dc333dc24991adc7a9987ecb72b364bb5ce64b3d618057f6a270.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/673877572d79dc333dc24991adc7a9987ecb72b364bb5ce64b3d618057f6a270.jpg)

![6a28825526da5c1817acdd8f1316645e67158c3199ea235425132da17284ef01.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/6a28825526da5c1817acdd8f1316645e67158c3199ea235425132da17284ef01.jpg)

![8c1b0a795ee2f921ed4cd7a4edd66690191c198152583e165e6c13d71c838fb9.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/8c1b0a795ee2f921ed4cd7a4edd66690191c198152583e165e6c13d71c838fb9.jpg)

![8cd46434ddf1e0b90a749132a7fcb49a7fb5c232e9926d2fab6e4188f7004a6c.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/8cd46434ddf1e0b90a749132a7fcb49a7fb5c232e9926d2fab6e4188f7004a6c.jpg)

![9165948d09bc87916e324542519990f0ad349c26c4c351eb66bff85504e0e0ba.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/9165948d09bc87916e324542519990f0ad349c26c4c351eb66bff85504e0e0ba.jpg)

![b3a681af3c2836fdc1c4a9b29282221c8ffb02b0acea9506a102fe8a9fd1b0d2.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/b3a681af3c2836fdc1c4a9b29282221c8ffb02b0acea9506a102fe8a9fd1b0d2.jpg)

![d7ea06feb3aaa7ca8a459ef4b7fbba4f0fd8899246d557d1174f725363baca17.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/d7ea06feb3aaa7ca8a459ef4b7fbba4f0fd8899246d557d1174f725363baca17.jpg)

![e5bcba58a3af1a69e8442df16ac95c0ccfa0267ba2eae8717253a716ec60904a.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/e5bcba58a3af1a69e8442df16ac95c0ccfa0267ba2eae8717253a716ec60904a.jpg)

![efd994abb2847ae6ab64d19d9fd6075a437f10a778621e0bf15cf0b85a7d9f6b.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/efd994abb2847ae6ab64d19d9fd6075a437f10a778621e0bf15cf0b85a7d9f6b.jpg)

![f7cbc2d2357724a51f117286313c879d7be950e8542788ae9d79df345ef089aa.jpg](../icml_results/3610_GUDA_%20Counterfactual%20Group-wise%20Training%20Data%20Attribution%20for%20Diffusion%20Models%20via%20Unlearning/tables/f7cbc2d2357724a51f117286313c879d7be950e8542788ae9d79df345ef089aa.jpg)

<a id="paper-3571"></a>

## Discovering Scaling Exponents with Physics-Informed Müntz-Szász Networks


### Images

![12bc1d8173f9c53d92a2c61ddae98432338e0b757cf896668f0ae6acffaed869.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/12bc1d8173f9c53d92a2c61ddae98432338e0b757cf896668f0ae6acffaed869.jpg)

![235ad9451d5c3a971a71c0b0cc7a68aa4c6100e0e78a07ce2550ab1fb44f108e.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/235ad9451d5c3a971a71c0b0cc7a68aa4c6100e0e78a07ce2550ab1fb44f108e.jpg)

![3a8b1390849e5d616e0506b925e8e0b72514ec3b09e430961dde8ce9e1eb4430.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/3a8b1390849e5d616e0506b925e8e0b72514ec3b09e430961dde8ce9e1eb4430.jpg)

![4bc1357962248df429143da477bbf720878dfcdbfd08fe240383ee64285e7341.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/4bc1357962248df429143da477bbf720878dfcdbfd08fe240383ee64285e7341.jpg)

![51f4eb5688ab870a527c8a87896b09f53e947e6c50a858a85a072e0f28e0fba5.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/51f4eb5688ab870a527c8a87896b09f53e947e6c50a858a85a072e0f28e0fba5.jpg)

![5b804e74c80a97fb96fc88ad3b8aa7e46d2cb0df00b4f180d9f9795d2b891197.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/images/5b804e74c80a97fb96fc88ad3b8aa7e46d2cb0df00b4f180d9f9795d2b891197.jpg)

### Tables

![26171526d0945613ae57941b9bb7c5176b5f1d2b7d3127903ed53438f813ff5f.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/26171526d0945613ae57941b9bb7c5176b5f1d2b7d3127903ed53438f813ff5f.jpg)

![26725b3fc639c0ec524063dae834713057427294899e51d9154833b5a351b3eb.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/26725b3fc639c0ec524063dae834713057427294899e51d9154833b5a351b3eb.jpg)

![364a8b641b197fb153447ad8e90a698462c573273e055b4d79043b24d1ff27a9.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/364a8b641b197fb153447ad8e90a698462c573273e055b4d79043b24d1ff27a9.jpg)

![3992a1bc79c1c8dde2426d63be907263d46f4efdbed05d8cf753a766617dcabf.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/3992a1bc79c1c8dde2426d63be907263d46f4efdbed05d8cf753a766617dcabf.jpg)

![42d2f1d20dd3fba48ad54d12d6a63b39c75acc1043eeb6504bedfb17bf6e9e83.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/42d2f1d20dd3fba48ad54d12d6a63b39c75acc1043eeb6504bedfb17bf6e9e83.jpg)

![4cdc807d4d8c0abac407af3a268baa091a709d4667a09307907393dc13dba79c.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/4cdc807d4d8c0abac407af3a268baa091a709d4667a09307907393dc13dba79c.jpg)

![51f1009c1c75c464b4c11fc38b0616dacd7bfe3c288137950bddf6d79b3331d5.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/51f1009c1c75c464b4c11fc38b0616dacd7bfe3c288137950bddf6d79b3331d5.jpg)

![5779c722430437ba7b109692e40f1bffa0dbce91c9d8e462539ae972252ab443.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/5779c722430437ba7b109692e40f1bffa0dbce91c9d8e462539ae972252ab443.jpg)

![58a89f2ad26dab2efc3506adb077461efbfb2ba813cda3ecb5fd5b2dcc483f07.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/58a89f2ad26dab2efc3506adb077461efbfb2ba813cda3ecb5fd5b2dcc483f07.jpg)

![5c32a5ec2f8316997c1904528dd8fbafe0a2479161b36f22a966470fe8b6a10a.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/5c32a5ec2f8316997c1904528dd8fbafe0a2479161b36f22a966470fe8b6a10a.jpg)

![620894c17788d7cb599d0e25187657a2212fb6fe2079a61756ee9b4554f32797.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/620894c17788d7cb599d0e25187657a2212fb6fe2079a61756ee9b4554f32797.jpg)

![6d1cf35a5e1dd5d2087a1744fe33f1857953e8f08a9c1120f51fcf5ebcf4e74c.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/6d1cf35a5e1dd5d2087a1744fe33f1857953e8f08a9c1120f51fcf5ebcf4e74c.jpg)

![97d58c9a2ff7a0d68de2048a5c80806691c983bee3e15d07c7b2fa88126e3850.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/97d58c9a2ff7a0d68de2048a5c80806691c983bee3e15d07c7b2fa88126e3850.jpg)

![98b6ff3461191ecc0e054acbac7a0ecf3096961dded65da1cc3d9a93a7c5eaf1.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/98b6ff3461191ecc0e054acbac7a0ecf3096961dded65da1cc3d9a93a7c5eaf1.jpg)

![b6e029469c2e372dada04504669f6fb1a3449c6461283287df7545e6ec9b4a48.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/b6e029469c2e372dada04504669f6fb1a3449c6461283287df7545e6ec9b4a48.jpg)

![cca7b6c4d0e872ace5ea7b4335a9ebd752150a53b8c5e82b91b43779dacad5ed.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/cca7b6c4d0e872ace5ea7b4335a9ebd752150a53b8c5e82b91b43779dacad5ed.jpg)

![d67ec241fddb914b7dbf55c0b8937ca7dfc6ad36f6f9cbec2fae1de02f272b62.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/d67ec241fddb914b7dbf55c0b8937ca7dfc6ad36f6f9cbec2fae1de02f272b62.jpg)

![f0e2eafa4de0924dd085cde10c09ad735c7abe9fe5cf12e740fc793a6ca65367.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/f0e2eafa4de0924dd085cde10c09ad735c7abe9fe5cf12e740fc793a6ca65367.jpg)

![f9731dbf2298d3d4a51b3fe3d67a57b919d9ec07c1b729beea5d70dac8dd7480.jpg](../icml_results/3611_Discovering%20Scaling%20Exponents%20with%20Physics-Informed%20M%C3%BCntz-Sz%C3%A1sz%20Networks/tables/f9731dbf2298d3d4a51b3fe3d67a57b919d9ec07c1b729beea5d70dac8dd7480.jpg)

<a id="paper-3572"></a>

## Interleaved Selective State Space Models for Efficient WiFi-Based 3D Multi-Person Pose Estimation


### Images

![0bcb4fa6be12da3097007430dd1265b7352800e26663002b9c7fe682a9c669a6.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/0bcb4fa6be12da3097007430dd1265b7352800e26663002b9c7fe682a9c669a6.jpg)

![0c14a629f1bcc7f30ff8c59f8e3e68795d0da183d6d7dedd5449781c5b783f6b.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/0c14a629f1bcc7f30ff8c59f8e3e68795d0da183d6d7dedd5449781c5b783f6b.jpg)

![1f9f046c5f6e5b8c4e79147b228279763c8025de320a2693bcf960212d3e529d.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/1f9f046c5f6e5b8c4e79147b228279763c8025de320a2693bcf960212d3e529d.jpg)

![386e48d94de7b19318b056f667f003a981d58b050b39fe1cfe58bbabee6637cf.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/386e48d94de7b19318b056f667f003a981d58b050b39fe1cfe58bbabee6637cf.jpg)

![3fd41232c8468895456d942e00003e35435d3b90c82287b934679680f9d5bf16.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/3fd41232c8468895456d942e00003e35435d3b90c82287b934679680f9d5bf16.jpg)

![4a9430465bea0f614b9b9f92f8ee3741bfad18be15bdf45852f4d4c81ca0166b.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/4a9430465bea0f614b9b9f92f8ee3741bfad18be15bdf45852f4d4c81ca0166b.jpg)

![514f9f8ac18db162c70b886d3710a4b8262d6764b56e5b6fe21fa40444bd6cd0.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/514f9f8ac18db162c70b886d3710a4b8262d6764b56e5b6fe21fa40444bd6cd0.jpg)

![5f87e05e311f2d6b046b65acbb6c431cc6c177f9b1d00c2c0f8ce60b8d6ed0e6.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/5f87e05e311f2d6b046b65acbb6c431cc6c177f9b1d00c2c0f8ce60b8d6ed0e6.jpg)

![781fa604bd22b23d7da78f7cadbc351ddd16b4c80a5143bff7e457fcc4b20adb.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/images/781fa604bd22b23d7da78f7cadbc351ddd16b4c80a5143bff7e457fcc4b20adb.jpg)

### Tables

![1bfe9d41016cdc788c72209cc20ed2715f81410e24f433da5db3ea811cc693a3.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/1bfe9d41016cdc788c72209cc20ed2715f81410e24f433da5db3ea811cc693a3.jpg)

![2b778f3ced786dc53bc9867303e06af61cc2c212fa3556968292fb2a543c4cbe.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/2b778f3ced786dc53bc9867303e06af61cc2c212fa3556968292fb2a543c4cbe.jpg)

![4e29db4c8025f1ba0e308d7943f119b951b990f4ecd85cd6847128904ace69c2.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/4e29db4c8025f1ba0e308d7943f119b951b990f4ecd85cd6847128904ace69c2.jpg)

![539514c57a65d697076c62965b509c26fe8059deae12b900eaf4cadb61aaac02.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/539514c57a65d697076c62965b509c26fe8059deae12b900eaf4cadb61aaac02.jpg)

![6fe6c91b87f16a6025d70094efa0c2c5d9aa2fe7af1f620e2fa3c14d00bab986.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/6fe6c91b87f16a6025d70094efa0c2c5d9aa2fe7af1f620e2fa3c14d00bab986.jpg)

![943fa17f819d9760f1505d4e834b1c4c676b1bc03fcc1280b6c343369ccc2f3e.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/943fa17f819d9760f1505d4e834b1c4c676b1bc03fcc1280b6c343369ccc2f3e.jpg)

![9bdff96558e17f7c9c930ce8251b9bc4ca9a2c36b7324ce56d5e9fb44510f38e.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/9bdff96558e17f7c9c930ce8251b9bc4ca9a2c36b7324ce56d5e9fb44510f38e.jpg)

![e93b353b74e67fb97e0d09d9b467c4aa2dc5e857991a3022d36a5dace49c187a.jpg](../icml_results/3612_Interleaved%20Selective%20State%20Space%20Models%20for%20Efficient%20WiFi-Based%203D%20Multi-Person%20Pose%20Estimation/tables/e93b353b74e67fb97e0d09d9b467c4aa2dc5e857991a3022d36a5dace49c187a.jpg)

<a id="paper-3573"></a>

## ITSPACE: Monotone Gaussian Optimal Transport Updates


### Images

![210eb1e247c224fb149e0c8b278e474dc64a5d9e48074d5c0233d60945068be0.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/images/210eb1e247c224fb149e0c8b278e474dc64a5d9e48074d5c0233d60945068be0.jpg)

![8585dd7d39fd9336347b49796d0413804bbaae98c52973f4d938bd9aad8896ef.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/images/8585dd7d39fd9336347b49796d0413804bbaae98c52973f4d938bd9aad8896ef.jpg)

![cced8a598b80c29c89e25b8bff3578dfea6be247d97a9528659b83921949a033.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/images/cced8a598b80c29c89e25b8bff3578dfea6be247d97a9528659b83921949a033.jpg)

### Tables

![2da504120186d64db903d904c6398df5b0bc745121f5f86597e8d14e1f643d24.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/2da504120186d64db903d904c6398df5b0bc745121f5f86597e8d14e1f643d24.jpg)

![3c9935d5e3cadcc352b85383c7558e7af6279bd613f8cc574687ab711935d253.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/3c9935d5e3cadcc352b85383c7558e7af6279bd613f8cc574687ab711935d253.jpg)

![617f6cec6ae983b7d9ed1f027f24e2f6745ae75c39069a99dd6a934a34a69e69.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/617f6cec6ae983b7d9ed1f027f24e2f6745ae75c39069a99dd6a934a34a69e69.jpg)

![6a092059305f1a672e121d172007e16c7b8486afe96a7ce2576304e21dabfeac.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/6a092059305f1a672e121d172007e16c7b8486afe96a7ce2576304e21dabfeac.jpg)

![710dae946a5c48bc3cca9092f9ef6daa8a8fd683e2550b3ca74cdb876ee68fd0.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/710dae946a5c48bc3cca9092f9ef6daa8a8fd683e2550b3ca74cdb876ee68fd0.jpg)

![78f8210b40d72d254bb175492f1f81ff519a79da4281044612f6413d441c9fd8.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/78f8210b40d72d254bb175492f1f81ff519a79da4281044612f6413d441c9fd8.jpg)

![aafbb439eec68050d8d4cf4a1b6ced4c94ce00f03de7248e330a7ad53ce37a84.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/aafbb439eec68050d8d4cf4a1b6ced4c94ce00f03de7248e330a7ad53ce37a84.jpg)

![e9d169a73494c10240dc3655d1ac0714c13d1e13d1fc5831b0c7781c981edd69.jpg](../icml_results/3613_ITSPACE_%20Monotone%20Gaussian%20Optimal%20Transport%20Updates/tables/e9d169a73494c10240dc3655d1ac0714c13d1e13d1fc5831b0c7781c981edd69.jpg)

<a id="paper-3574"></a>

## Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models


### Images

![239c6f048524cea2da138c14378a9f5ddd246c54e14d74cdb1999793ea702103.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/239c6f048524cea2da138c14378a9f5ddd246c54e14d74cdb1999793ea702103.jpg)

![71073e405279e7c1ce4c1b2d5e23569d1671701351fc88d4b22ae1bc41ff2241.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/71073e405279e7c1ce4c1b2d5e23569d1671701351fc88d4b22ae1bc41ff2241.jpg)

![a5877c93fccdc42ad03eb05c78b17fd2883c9126cf9e428db040763222c9c5fa.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/a5877c93fccdc42ad03eb05c78b17fd2883c9126cf9e428db040763222c9c5fa.jpg)

![a8f25f10874d7d68367d3b8fa6f60cf5348134cf0de002996993be61a9f4f7f8.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/a8f25f10874d7d68367d3b8fa6f60cf5348134cf0de002996993be61a9f4f7f8.jpg)

![ac1bd2c7ca37a701451d5b8a7da4bf2cd5bae68f04a1b957ad8bfdacbc2e0591.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/ac1bd2c7ca37a701451d5b8a7da4bf2cd5bae68f04a1b957ad8bfdacbc2e0591.jpg)

![c46ada5c0890bea1c42c2069ff68998823f8d6ac70de3b0fb0439455428d47a4.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/c46ada5c0890bea1c42c2069ff68998823f8d6ac70de3b0fb0439455428d47a4.jpg)

![c622105a2fb38a483f344414702137186a83abafc10c7a04c22db9f5230e86fd.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/c622105a2fb38a483f344414702137186a83abafc10c7a04c22db9f5230e86fd.jpg)

![cc6d3246ec2d145d814dbcecdfd17eb91e81cb337104e8997cca77de1fa47f15.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/images/cc6d3246ec2d145d814dbcecdfd17eb91e81cb337104e8997cca77de1fa47f15.jpg)

### Tables

![4c958d71603e5168b09a8e9e5d047cc19b2cc0c4adbcc57091a60b640ee2d424.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/4c958d71603e5168b09a8e9e5d047cc19b2cc0c4adbcc57091a60b640ee2d424.jpg)

![52167d1cb45cd4607f011d0f58e81b561c27b275d9f7634e9acae62eada0b3c6.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/52167d1cb45cd4607f011d0f58e81b561c27b275d9f7634e9acae62eada0b3c6.jpg)

![ca0093dea3a2395d1be83d5314235ba3fa61a3cb1973d4a84fcb3ed3e553c91c.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/ca0093dea3a2395d1be83d5314235ba3fa61a3cb1973d4a84fcb3ed3e553c91c.jpg)

![cc02f56cd41a6265a898e23ab6b2b3a840820daea54d3b91efbb4e50b8c4670c.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/cc02f56cd41a6265a898e23ab6b2b3a840820daea54d3b91efbb4e50b8c4670c.jpg)

![d410095c35a20a7fa5fac48c65136dcf0479dc2a87fe5ed54a1e0628c46e7917.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/d410095c35a20a7fa5fac48c65136dcf0479dc2a87fe5ed54a1e0628c46e7917.jpg)

![d8096ec2a1f6b84bc594ec99230295ff03b920ae828c5c7b77305f3ac52ed658.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/d8096ec2a1f6b84bc594ec99230295ff03b920ae828c5c7b77305f3ac52ed658.jpg)

![db48198467bcf0c9796a3fdaff60c39e0194f036da7287a7e61eed8d9e8f79d5.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/db48198467bcf0c9796a3fdaff60c39e0194f036da7287a7e61eed8d9e8f79d5.jpg)

![e3a8658382c4879326853c96c236658f1bc2db90d4d1fd561ba491eb7b697b5a.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/e3a8658382c4879326853c96c236658f1bc2db90d4d1fd561ba491eb7b697b5a.jpg)

![eb91fd1b38642c25493c6e9e02ba4b4009ff7e229c2efad4b095ce2f7256c9b4.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/eb91fd1b38642c25493c6e9e02ba4b4009ff7e229c2efad4b095ce2f7256c9b4.jpg)

![ff96858f01236505683a8cf63175b6ce43c9c04fca77939be3c155c6a4a95035.jpg](../icml_results/3614_Time-PEFT_%20Temporal%20and%20Multichannel%20Complexity-Based%20Fine-Tuning%20for%20Time-Series%20Foundation%20Models/tables/ff96858f01236505683a8cf63175b6ce43c9c04fca77939be3c155c6a4a95035.jpg)

<a id="paper-3575"></a>

## Hierarchical Causal Abduction: A Foundation Framework for Explainable Model Predictive Control


### Images

![5953bcab0aaabe79eb0544cbbd5ec5d98ccc3f4424c990f36e625e5e25df9122.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/images/5953bcab0aaabe79eb0544cbbd5ec5d98ccc3f4424c990f36e625e5e25df9122.jpg)

![c52b269233944d4b984ae15f39b709437a08885874937368c353bb07f7270685.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/images/c52b269233944d4b984ae15f39b709437a08885874937368c353bb07f7270685.jpg)

### Tables

![0187677daa7e83410f115a4d649813091a2fb0c2dcc072dac534f9ec1277259c.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/0187677daa7e83410f115a4d649813091a2fb0c2dcc072dac534f9ec1277259c.jpg)

![04662062d79e84a9dc3f84fffc57a29895e7f2a1bf9f028a1064a2784f70c40a.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/04662062d79e84a9dc3f84fffc57a29895e7f2a1bf9f028a1064a2784f70c40a.jpg)

![0fd7d1ad8e1af017054cc280f2024734e7328d165a104d6cd1ead9f8bda0059d.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/0fd7d1ad8e1af017054cc280f2024734e7328d165a104d6cd1ead9f8bda0059d.jpg)

![44bb01f753d03de5499bf96c9f148ad82c01fa49601091b2e0ca11f0be24106b.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/44bb01f753d03de5499bf96c9f148ad82c01fa49601091b2e0ca11f0be24106b.jpg)

![7e86a369538083282fae248fcaf6b16d588456a9541314558c5c183a45df6688.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/7e86a369538083282fae248fcaf6b16d588456a9541314558c5c183a45df6688.jpg)

![bcbbaaa38ec762682f4d7434ed02b9160c2060fe90e21c95fbd8209fc17f1c9f.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/bcbbaaa38ec762682f4d7434ed02b9160c2060fe90e21c95fbd8209fc17f1c9f.jpg)

![c0bb82b3e7ca45c1c5a91d89e108732ab2626c56e36ea1f93e5154544a444bf0.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/c0bb82b3e7ca45c1c5a91d89e108732ab2626c56e36ea1f93e5154544a444bf0.jpg)

![cb16a05ba93c9f5bfca2a628a71079c8d57673f7085a1d248bc5104015bc4bf5.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/cb16a05ba93c9f5bfca2a628a71079c8d57673f7085a1d248bc5104015bc4bf5.jpg)

![e2e837a987b597288a5f647d7f9ed31850aad6fa2033cf82c3361126f02334cc.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/e2e837a987b597288a5f647d7f9ed31850aad6fa2033cf82c3361126f02334cc.jpg)

![e5482d712a40d8c0be006b1296f136493c0add30687761359a849f4a7ed8f06d.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/e5482d712a40d8c0be006b1296f136493c0add30687761359a849f4a7ed8f06d.jpg)

![ebec1be4a00d3fee8c2642b438fbd86283243ee224fc8ff0835193d54ab62e63.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/ebec1be4a00d3fee8c2642b438fbd86283243ee224fc8ff0835193d54ab62e63.jpg)

![f9b4ec932b8f8c7dc6fbbd51d89c93dc2c010bd06b66f0c5eb769dfdbec7117d.jpg](../icml_results/3615_Hierarchical%20Causal%20Abduction_%20A%20Foundation%20Framework%20for%20Explainable%20Model%20Predictive%20Control/tables/f9b4ec932b8f8c7dc6fbbd51d89c93dc2c010bd06b66f0c5eb769dfdbec7117d.jpg)

<a id="paper-3576"></a>

## How does Chain of Thought decompose complex tasks?


### Images

![0505f36c442cd5d6a6d59de00536d60347cddaba568c9aee95bec0629716796d.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/0505f36c442cd5d6a6d59de00536d60347cddaba568c9aee95bec0629716796d.jpg)

![0747be29660e57aaf166dbfaf59a77eb7fa36dee197125f8bf87415ad817514c.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/0747be29660e57aaf166dbfaf59a77eb7fa36dee197125f8bf87415ad817514c.jpg)

![386f53dd02a01aea4db8d37a9276facc7b793a0cdb212816090cc2c06fc457bd.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/386f53dd02a01aea4db8d37a9276facc7b793a0cdb212816090cc2c06fc457bd.jpg)

![4c84458a7a1b43ae5b02539691551b3048fa093dec98e0f422318bccded5a460.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/4c84458a7a1b43ae5b02539691551b3048fa093dec98e0f422318bccded5a460.jpg)

![5a5f293f4395cbcf518ba8ea2a46959808535ce0c61582ec50cb607e42400c6d.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/5a5f293f4395cbcf518ba8ea2a46959808535ce0c61582ec50cb607e42400c6d.jpg)

![69fb44ba182ed73a585326831c3dd8f4c435b932ed3a345e9458b5e297cceb6d.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/69fb44ba182ed73a585326831c3dd8f4c435b932ed3a345e9458b5e297cceb6d.jpg)

![898e611afed783398fe82582c2f21b591b4b7d7bd38e1be6e8995153d5163bdf.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/898e611afed783398fe82582c2f21b591b4b7d7bd38e1be6e8995153d5163bdf.jpg)

![9d647c09e007a21153d6525728a152a7fca7740fb35a2b672726aad0f18d47ea.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/9d647c09e007a21153d6525728a152a7fca7740fb35a2b672726aad0f18d47ea.jpg)

![9f1b042025c38b36f148558bd08c6a69cd3916ad5a994b2bafd3d793116ed131.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/9f1b042025c38b36f148558bd08c6a69cd3916ad5a994b2bafd3d793116ed131.jpg)

![bd6175c3373941c4f83012a2adb44721293812d0da738f82382ec805c1190db2.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/bd6175c3373941c4f83012a2adb44721293812d0da738f82382ec805c1190db2.jpg)

![e6d9489a12000aa869ac76d3c89cfb896fbc5dbad7066fe1c5c9ac37c85a4b1c.jpg](../icml_results/3616_How%20does%20Chain%20of%20Thought%20decompose%20complex%20tasks_/images/e6d9489a12000aa869ac76d3c89cfb896fbc5dbad7066fe1c5c9ac37c85a4b1c.jpg)

<a id="paper-3577"></a>

## Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent


### Images

![bb837b3d389868074e20c1f56416b1911265168a74539374cc70ceac3ea5129f.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/images/bb837b3d389868074e20c1f56416b1911265168a74539374cc70ceac3ea5129f.jpg)

![d0be99c74924c9799a5304e84780d771c00f84fc1534ed7dc24a801e425ede2c.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/images/d0be99c74924c9799a5304e84780d771c00f84fc1534ed7dc24a801e425ede2c.jpg)

![d9cabc29579731d9a3b62f3000c36554950bd184a6a5cd8f6d09631c1d7d5ce8.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/images/d9cabc29579731d9a3b62f3000c36554950bd184a6a5cd8f6d09631c1d7d5ce8.jpg)

### Tables

![1898335f2538dee3f22a56a25f20f05b20f1eab4084593389f8157a7129f87e0.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/1898335f2538dee3f22a56a25f20f05b20f1eab4084593389f8157a7129f87e0.jpg)

![2825153bcf1bbe58f3ae4f9b786ef5a121f5ebbd03dedb16e6e5043ae4178fee.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/2825153bcf1bbe58f3ae4f9b786ef5a121f5ebbd03dedb16e6e5043ae4178fee.jpg)

![4861ff5b2649f3cd488024b5a3fc6e936257c8bca082fd90b742c32f69fedc2a.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/4861ff5b2649f3cd488024b5a3fc6e936257c8bca082fd90b742c32f69fedc2a.jpg)

![7759bbdf8018980221102d66bbc3b00b14f26ef2d0fbbf4c93f29d4e8212a0bd.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/7759bbdf8018980221102d66bbc3b00b14f26ef2d0fbbf4c93f29d4e8212a0bd.jpg)

![7d2663bf9f9a94f672bc99df1d88aea56ab828195e94306e6abdc3762a25e138.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/7d2663bf9f9a94f672bc99df1d88aea56ab828195e94306e6abdc3762a25e138.jpg)

![90f81b1638d5fb2ca8ea29937229d4aa9b7fd8f6b2e22b4864056477dd3f3777.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/90f81b1638d5fb2ca8ea29937229d4aa9b7fd8f6b2e22b4864056477dd3f3777.jpg)

![96c61ae947021427321d539b4360a57a011348e2768796dcd07ca28c274a65d7.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/96c61ae947021427321d539b4360a57a011348e2768796dcd07ca28c274a65d7.jpg)

![b1188edb8cd3eb8b05fd04447fac541b77ed827a4e0c4251593f659bb04d1dd8.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/b1188edb8cd3eb8b05fd04447fac541b77ed827a4e0c4251593f659bb04d1dd8.jpg)

![babe96a35c658686b778642044ff251c6689d072c39568346c9ac989e6b08c06.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/babe96a35c658686b778642044ff251c6689d072c39568346c9ac989e6b08c06.jpg)

![d4eb589ccbb18686cb20a50d0142e8a8b44587f52101532554892be199cd91f1.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/d4eb589ccbb18686cb20a50d0142e8a8b44587f52101532554892be199cd91f1.jpg)

![e30a776a4bd02145981bea90e88980a0be14eb9dc4faff73c914a12f9223522a.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/e30a776a4bd02145981bea90e88980a0be14eb9dc4faff73c914a12f9223522a.jpg)

![f6858880a15564170f95355d41dd71d95dfdee282753ba55a2697a22598a5e41.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/f6858880a15564170f95355d41dd71d95dfdee282753ba55a2697a22598a5e41.jpg)

![fc4e70b3e9ece001afe629db26029ecb81f732797d5fedcd9cafa0433dbd5f16.jpg](../icml_results/3617_Adaptive%20Batch%20Sizes%20Using%20Non-Euclidean%20Gradient%20Noise%20Scales%20for%20Stochastic%20Sign%20and%20Spectral%20Desc/tables/fc4e70b3e9ece001afe629db26029ecb81f732797d5fedcd9cafa0433dbd5f16.jpg)

<a id="paper-3578"></a>

## TRIM: Token-wise Attention-Derived Saliency for Data-Efficient Instruction Tuning


### Images

![1dcf7446031db7e7b3162f10331e7a9321cdeec6c414b1ee61c212b5a1dbdd10.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/1dcf7446031db7e7b3162f10331e7a9321cdeec6c414b1ee61c212b5a1dbdd10.jpg)

![29496acf4b48569e8118d32d401975ae399f434ff027eceb979d4d48350b02a5.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/29496acf4b48569e8118d32d401975ae399f434ff027eceb979d4d48350b02a5.jpg)

![3ffb9a793b76e72c803000e249fb019ea241f1167213b492a27dc128f7e7a2c3.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/3ffb9a793b76e72c803000e249fb019ea241f1167213b492a27dc128f7e7a2c3.jpg)

![8dbb27fe7158aac096b33076200badbc67e3e8dafbcf19d63b4e433bc1724d4d.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/8dbb27fe7158aac096b33076200badbc67e3e8dafbcf19d63b4e433bc1724d4d.jpg)

![99d9278466060f50e8a6ddc3cf1cb445ed94072de84ef0daac5a693b72da63b5.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/99d9278466060f50e8a6ddc3cf1cb445ed94072de84ef0daac5a693b72da63b5.jpg)

![edce8404b091d06e61848dece74d5dd067be38686633d2ba97c93f0e63bc694e.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/images/edce8404b091d06e61848dece74d5dd067be38686633d2ba97c93f0e63bc694e.jpg)

### Tables

![10c8f929f8c12332a79d6a34682a4ea8ec05ee43f69637263554f641c572aef9.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/10c8f929f8c12332a79d6a34682a4ea8ec05ee43f69637263554f641c572aef9.jpg)

![1187bdc13a8849db9e9f6037304de3f266e3b2115393f7099030e29b46f35c1d.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/1187bdc13a8849db9e9f6037304de3f266e3b2115393f7099030e29b46f35c1d.jpg)

![1830300fcb8cb6dfa1c69d21c1a6660f9e81b5896d2d4028612885fc99ce8a20.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/1830300fcb8cb6dfa1c69d21c1a6660f9e81b5896d2d4028612885fc99ce8a20.jpg)

![30c4bcd2aeec95881fb7054bafd514712123b0c329c65d47e599439e8d71f0fd.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/30c4bcd2aeec95881fb7054bafd514712123b0c329c65d47e599439e8d71f0fd.jpg)

![319b39eb27b0dc48d968b8c4279be2044489c7ad8174ea0ebea300280e5dc301.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/319b39eb27b0dc48d968b8c4279be2044489c7ad8174ea0ebea300280e5dc301.jpg)

![36ecb1aa1a52a8f4ce8d20d79a5ed1990128e36185434fc0920d58a56e92cbfa.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/36ecb1aa1a52a8f4ce8d20d79a5ed1990128e36185434fc0920d58a56e92cbfa.jpg)

![456d93ebfa4df7c15756eb63ac9a41bc6705a91fb0e52f8af09ac191888c29ca.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/456d93ebfa4df7c15756eb63ac9a41bc6705a91fb0e52f8af09ac191888c29ca.jpg)

![4626fc8149b5fe6d44214dc223580e6488c0e84b5a7d0c4004bf56dee95e2f92.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/4626fc8149b5fe6d44214dc223580e6488c0e84b5a7d0c4004bf56dee95e2f92.jpg)

![4c50d7977efce12acbf9e5aea479df09e08167c737a7f2f0346aab72cfc0abdf.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/4c50d7977efce12acbf9e5aea479df09e08167c737a7f2f0346aab72cfc0abdf.jpg)

![59f94ec001a7996b95331e6bf07b1d3feba7ea21fdad1cd26e82757e991b436b.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/59f94ec001a7996b95331e6bf07b1d3feba7ea21fdad1cd26e82757e991b436b.jpg)

![5af9e90e6abc3e25e06df25444a0a89ea2eca10434127f73bbf77fe0d2e25d33.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/5af9e90e6abc3e25e06df25444a0a89ea2eca10434127f73bbf77fe0d2e25d33.jpg)

![780edec8c0cf73aef7d11a1984307b1f3eabb9b56caf8778c4a7904ee51227c2.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/780edec8c0cf73aef7d11a1984307b1f3eabb9b56caf8778c4a7904ee51227c2.jpg)

![7e77ef2a5916cde176bd8ba95a4be684f6a2de5ea7d7b4d39c7451b9bfec44b9.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/7e77ef2a5916cde176bd8ba95a4be684f6a2de5ea7d7b4d39c7451b9bfec44b9.jpg)

![8e5860f68c2e67901e68b6bea40045b13c3a6384cb9ad0d3cbdacae429abedce.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/8e5860f68c2e67901e68b6bea40045b13c3a6384cb9ad0d3cbdacae429abedce.jpg)

![9b3ac494519d3c04083e0eecb2ef410266214a97ffb8c7df81bdfe16c5234b7f.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/9b3ac494519d3c04083e0eecb2ef410266214a97ffb8c7df81bdfe16c5234b7f.jpg)

![b75384fd1353e11089f369912b4f3a432ffa6c40e55ee562f0741755d45507c8.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/b75384fd1353e11089f369912b4f3a432ffa6c40e55ee562f0741755d45507c8.jpg)

![bae22258076e305689d5630275ed81135f2303655519376db01eb83907507aad.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/bae22258076e305689d5630275ed81135f2303655519376db01eb83907507aad.jpg)

![bdf463da4f01686f0b39090267c2959346f7ee19bc5b51b9c92b02db05d6236f.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/bdf463da4f01686f0b39090267c2959346f7ee19bc5b51b9c92b02db05d6236f.jpg)

![bfb86cc3e10f6f87cc562e9a8e9a69b3432d9385f62bf32af0b7c1c18be57587.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/bfb86cc3e10f6f87cc562e9a8e9a69b3432d9385f62bf32af0b7c1c18be57587.jpg)

![c9d67831da41a8506e499e1675d4ea408d63aa16173c13d5986df81f18dce539.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/c9d67831da41a8506e499e1675d4ea408d63aa16173c13d5986df81f18dce539.jpg)

![ce397d41ee550b9c22bd8f30422ef50a1077fb3bcc2fab6ae94983867347afea.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/ce397d41ee550b9c22bd8f30422ef50a1077fb3bcc2fab6ae94983867347afea.jpg)

![d8c1762c3ccf199e4a53f921f839878bcfb197dd78c410df71078277bfceb183.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/d8c1762c3ccf199e4a53f921f839878bcfb197dd78c410df71078277bfceb183.jpg)

![da8c3418ebbe27e1888960b4725cc8f5ed04ea0f58bc58d5a06ede00e0049e49.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/da8c3418ebbe27e1888960b4725cc8f5ed04ea0f58bc58d5a06ede00e0049e49.jpg)

![e5282a07b374fc7bcdee742be7f234e69484875aaf6f6f0379667afaea5940b5.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/e5282a07b374fc7bcdee742be7f234e69484875aaf6f6f0379667afaea5940b5.jpg)

![e574b5a3f593da6d6e4a0bd32305230fb5863f0a2ba896ad0ebd666fb4bb4af8.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/e574b5a3f593da6d6e4a0bd32305230fb5863f0a2ba896ad0ebd666fb4bb4af8.jpg)

![f60f8102c47fd37eb71685c7766bac38e9a0eb04e683049e4b9f3e4cd8a509e8.jpg](../icml_results/3618_TRIM_%20Token-wise%20Attention-Derived%20Saliency%20for%20Data-Efficient%20Instruction%20Tuning/tables/f60f8102c47fd37eb71685c7766bac38e9a0eb04e683049e4b9f3e4cd8a509e8.jpg)

<a id="paper-3579"></a>

## Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories


### Images

![07abdb616dd9366acc41d2d827be5e781d899b960526d073e048a8ab2266ee14.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/07abdb616dd9366acc41d2d827be5e781d899b960526d073e048a8ab2266ee14.jpg)

![2c67a9f82e8021014ec746ba1cf3aaf651d43b9834fe77f91c7a85ccbd1ba973.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/2c67a9f82e8021014ec746ba1cf3aaf651d43b9834fe77f91c7a85ccbd1ba973.jpg)

![3babfee16f3df0ae8e884985fb58b8b9e2edff24f6f54e9d7f79245f984d5cf6.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/3babfee16f3df0ae8e884985fb58b8b9e2edff24f6f54e9d7f79245f984d5cf6.jpg)

![49d8d3884560e3e29d77782155625ed30aca44db1040f8501781d7a088458304.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/49d8d3884560e3e29d77782155625ed30aca44db1040f8501781d7a088458304.jpg)

![5a7e00c830d5f8b9ff643f58c97ced409140ce0381898c2edb1681fc31fd2866.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/5a7e00c830d5f8b9ff643f58c97ced409140ce0381898c2edb1681fc31fd2866.jpg)

![5d5bf63a1ef2f4a77b86cd92e99f451e47c54cd65deb73876fde702aae3e846c.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/5d5bf63a1ef2f4a77b86cd92e99f451e47c54cd65deb73876fde702aae3e846c.jpg)

![6af3bbf79c0d12b036a2b40e752043174af909874c5485eab4f198dd2639b60a.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/6af3bbf79c0d12b036a2b40e752043174af909874c5485eab4f198dd2639b60a.jpg)

![6b48f1a13f1ed05ac37d3ce96f12fe1dc530d188c84505ef07a6b59dfcd8118a.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/6b48f1a13f1ed05ac37d3ce96f12fe1dc530d188c84505ef07a6b59dfcd8118a.jpg)

![75a2faa86665942dea9b17f82cf1b5a44255ba47d98a22c7161f65fb1c94ca75.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/75a2faa86665942dea9b17f82cf1b5a44255ba47d98a22c7161f65fb1c94ca75.jpg)

![88068a4b763d028c74fa30c5dd85272db53ee78bf30e7bff349fdc2b91162107.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/88068a4b763d028c74fa30c5dd85272db53ee78bf30e7bff349fdc2b91162107.jpg)

![9f23b4e4bb94aae5030b3dbfa0c53ead428244383d6c5181db242f1bbd4ebda9.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/9f23b4e4bb94aae5030b3dbfa0c53ead428244383d6c5181db242f1bbd4ebda9.jpg)

![a384e4e8edd73edb992558a126d4227ab6824b180064c2fdcff9c2386a46458c.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/a384e4e8edd73edb992558a126d4227ab6824b180064c2fdcff9c2386a46458c.jpg)

![a5914c554ec5e219edd7c73dffdefb284a17d1b562b81a0e8b15e180f7e30a13.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/a5914c554ec5e219edd7c73dffdefb284a17d1b562b81a0e8b15e180f7e30a13.jpg)

![c3aa52d5602648573048ea12ce3e371771a2bc7f7cb22504b9bc22a9f608dc15.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/c3aa52d5602648573048ea12ce3e371771a2bc7f7cb22504b9bc22a9f608dc15.jpg)

![c6ded4a13fc6c5776ce35c38b9770a68d070ea9af09f8ae31809f03c1150b912.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/c6ded4a13fc6c5776ce35c38b9770a68d070ea9af09f8ae31809f03c1150b912.jpg)

![cd40db87323e77bc46606574a86307bbe755e9568e5be694e2be6dc7b56b203f.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/cd40db87323e77bc46606574a86307bbe755e9568e5be694e2be6dc7b56b203f.jpg)

![d7e60b209021491bdd87520fdc64c7a866cddfacd16532a30dd2afc150c42d63.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/images/d7e60b209021491bdd87520fdc64c7a866cddfacd16532a30dd2afc150c42d63.jpg)

### Tables

![079e86a488d3ea8421c9aa41984be60498a60a51079306228bbdfc98a0d09297.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/079e86a488d3ea8421c9aa41984be60498a60a51079306228bbdfc98a0d09297.jpg)

![27d19b7a91ce76de6f5a55e160dcd7d28f7cc6d18d52251371c0c34a9b1cbb2d.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/27d19b7a91ce76de6f5a55e160dcd7d28f7cc6d18d52251371c0c34a9b1cbb2d.jpg)

![28201f9a6af843288448ab6a6c877e568563e783715569b601564e27ff80a541.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/28201f9a6af843288448ab6a6c877e568563e783715569b601564e27ff80a541.jpg)

![2c446b707ffea7dd22130fa524b1bc0194b2c0da4c4b37b3f968235c242cc1ac.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/2c446b707ffea7dd22130fa524b1bc0194b2c0da4c4b37b3f968235c242cc1ac.jpg)

![5cbf54d4cd6f0b8eadf789b7411a1fe27ef81c8198a00ec028d6eb4fb389c4a6.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/5cbf54d4cd6f0b8eadf789b7411a1fe27ef81c8198a00ec028d6eb4fb389c4a6.jpg)

![6971982f1b50657e8bf53b13e3e42a71294d9c8bea5fafc5e653ec400478eb21.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/6971982f1b50657e8bf53b13e3e42a71294d9c8bea5fafc5e653ec400478eb21.jpg)

![6d5ac56b2df6dd60facb7fd30de2d45b65f4e8e1982efba186edf36417dce99f.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/6d5ac56b2df6dd60facb7fd30de2d45b65f4e8e1982efba186edf36417dce99f.jpg)

![87c6798e692d046093130867e70fc7d31fd978709a23cd49547ab5771e8fb71b.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/87c6798e692d046093130867e70fc7d31fd978709a23cd49547ab5771e8fb71b.jpg)

![a5e890f0c2587daa63a770d25360907843de55fe130ac69a9f73c2c0910eb9f0.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/a5e890f0c2587daa63a770d25360907843de55fe130ac69a9f73c2c0910eb9f0.jpg)

![ab7ee87013baf99a00ae9436e0d0f8c3809a6c8ee6028b588f468ccba581cd4d.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/ab7ee87013baf99a00ae9436e0d0f8c3809a6c8ee6028b588f468ccba581cd4d.jpg)

![be2f25a32f768e7fb3769acc08dc0891e05d4d012e92ce5d16706d23bf9a851a.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/be2f25a32f768e7fb3769acc08dc0891e05d4d012e92ce5d16706d23bf9a851a.jpg)

![bf127fde06a2f351a903c2c1ec434e5a65c9dd656e0c56f3e254ce941dbd96ca.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/bf127fde06a2f351a903c2c1ec434e5a65c9dd656e0c56f3e254ce941dbd96ca.jpg)

![cfc8dee5d694fd071d6fb11d74e066085c55a9da18eac793a388c98d0411e419.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/cfc8dee5d694fd071d6fb11d74e066085c55a9da18eac793a388c98d0411e419.jpg)

![fa95a8dd5e631a96795556d07438cea6e313e0a4f2e42a0a97e75df026a1ae73.jpg](../icml_results/3619_Data%20Selection%20for%20Fine-tuning%20Vision%20Language%20Models%20via%20Cross%20Modal%20Alignment%20Trajectories/tables/fa95a8dd5e631a96795556d07438cea6e313e0a4f2e42a0a97e75df026a1ae73.jpg)

<a id="paper-3580"></a>

## Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models


### Images

![0b30e7e7a9b9e4cb7a1d6d07db6907f8bb6e46f11a9464f9bb722caa0d50686b.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/0b30e7e7a9b9e4cb7a1d6d07db6907f8bb6e46f11a9464f9bb722caa0d50686b.jpg)

![1d15d36d57a71217700e960bb5c1900cb5409a0cc470c75c2bef2a976fc75212.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/1d15d36d57a71217700e960bb5c1900cb5409a0cc470c75c2bef2a976fc75212.jpg)

![36e5107885654376ed8a776c18350f4b02d354990796b0ac76b254f4b20d4a9a.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/36e5107885654376ed8a776c18350f4b02d354990796b0ac76b254f4b20d4a9a.jpg)

![5824d8ec02b680b0aa70d8b8b9f6b484debe57322f6acfa21e8abd61e682002f.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/5824d8ec02b680b0aa70d8b8b9f6b484debe57322f6acfa21e8abd61e682002f.jpg)

![7b107d026d4807a0dc0464a3de0c37ee6269ee7db9491fbd7d1653af4bcae72c.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/7b107d026d4807a0dc0464a3de0c37ee6269ee7db9491fbd7d1653af4bcae72c.jpg)

![a8c028e64686d0827e2597cd0648fbbcbd9e8e2ca6827852ced1113063192a83.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/a8c028e64686d0827e2597cd0648fbbcbd9e8e2ca6827852ced1113063192a83.jpg)

![affc787731697e7ecf0588bca9a7322e472db0138b3247f44ce73d4bc0c12f42.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/affc787731697e7ecf0588bca9a7322e472db0138b3247f44ce73d4bc0c12f42.jpg)

![b46e7006e7eb06d775d206433d004e422ad7080164b081c80b997ea85decd5e7.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/b46e7006e7eb06d775d206433d004e422ad7080164b081c80b997ea85decd5e7.jpg)

![bf5907d19384b6233eeadb596195a267774f9b161c58f3fa0ca045c1f51aa4c2.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/bf5907d19384b6233eeadb596195a267774f9b161c58f3fa0ca045c1f51aa4c2.jpg)

![c29463528f3f86f2726e31c1d595703378ed295702cbd14a59ee5f39a31ad970.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/c29463528f3f86f2726e31c1d595703378ed295702cbd14a59ee5f39a31ad970.jpg)

![ccb0372fc0ac926cb3003d024dab008dfaec824ef84a242a283f773e0ada792e.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/ccb0372fc0ac926cb3003d024dab008dfaec824ef84a242a283f773e0ada792e.jpg)

![d8a97f43005b0303d316c8283dd0d40546a794a40f96107703cee0a1a8762f34.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/d8a97f43005b0303d316c8283dd0d40546a794a40f96107703cee0a1a8762f34.jpg)

![e884c4313e6f88e1c3d2b952cb0b5727ea5cede8ef184f7718cab5077eaee87a.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/e884c4313e6f88e1c3d2b952cb0b5727ea5cede8ef184f7718cab5077eaee87a.jpg)

![f3557f1f5b0e8d9fa6e8c86bf1ca15c2b71ae02c41fe68323d51d8cf3f312f10.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/images/f3557f1f5b0e8d9fa6e8c86bf1ca15c2b71ae02c41fe68323d51d8cf3f312f10.jpg)

### Tables

![03c356b491c639e7571a7c54cc41ae6dd451958600403de554ca70b38b1386fe.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/03c356b491c639e7571a7c54cc41ae6dd451958600403de554ca70b38b1386fe.jpg)

![2fa0f02c8396f74ced95565e7b6b98768a6795352097c27dff092aee72929a14.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/2fa0f02c8396f74ced95565e7b6b98768a6795352097c27dff092aee72929a14.jpg)

![3cfe278f56fc9b55d454c8e94fa077e80bf0e927c05882d99f0aee3b882cfaab.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/3cfe278f56fc9b55d454c8e94fa077e80bf0e927c05882d99f0aee3b882cfaab.jpg)

![6645071563729e54c5aad82ba90e6227d944f6ccb03c68cc0504e46540f968a4.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/6645071563729e54c5aad82ba90e6227d944f6ccb03c68cc0504e46540f968a4.jpg)

![709d6d95758b18d4e9f7cfc1f0117def34e4e72393c37ef74687010337ab3cc7.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/709d6d95758b18d4e9f7cfc1f0117def34e4e72393c37ef74687010337ab3cc7.jpg)

![7bdc249fe43cfff15ff0d7a4bd464ad045ec903e1b9c63b5e55338eb60b3b5d4.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/7bdc249fe43cfff15ff0d7a4bd464ad045ec903e1b9c63b5e55338eb60b3b5d4.jpg)

![8862cfd9db50b82b328478a30ac5a526f736fca657a5f90c1450ea90ad0e269b.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/8862cfd9db50b82b328478a30ac5a526f736fca657a5f90c1450ea90ad0e269b.jpg)

![8e84c9e13790a1f613580f0ff9b4d07e51bc2e5be545d630a21e6b4ea9098674.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/8e84c9e13790a1f613580f0ff9b4d07e51bc2e5be545d630a21e6b4ea9098674.jpg)

![a8a3b29b39c6455570d7aef64a2d33bc9fe861080003f14651cf7e70da7af21e.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/a8a3b29b39c6455570d7aef64a2d33bc9fe861080003f14651cf7e70da7af21e.jpg)

![b2b3e7d83a2306713c249a4c1e331a87fe1312ece78929893c517bd8554ff5d1.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/b2b3e7d83a2306713c249a4c1e331a87fe1312ece78929893c517bd8554ff5d1.jpg)

![efdedddf52a565321c3e8345e4261540b06a3e7f642a3657fedc1d6ec620f61d.jpg](../icml_results/3620_Less%20Diverse%2C%20Less%20Safe_%20The%20Indirect%20But%20Pervasive%20Risk%20of%20Test-Time%20Scaling%20in%20Large%20Language%20Mode/tables/efdedddf52a565321c3e8345e4261540b06a3e7f642a3657fedc1d6ec620f61d.jpg)

<a id="paper-3581"></a>

## Interpreting Genomic Language Models using Sparse Autoencoders


### Images

![02fba3aec064de7ffe2388c1de23194e3e9f4594d8b9b8ac250144b16d1de065.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/02fba3aec064de7ffe2388c1de23194e3e9f4594d8b9b8ac250144b16d1de065.jpg)

![14f4bb09ea1b3bd3c9f5b21ed2b8b171f156c60abfebb0a17e1cd888ab113274.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/14f4bb09ea1b3bd3c9f5b21ed2b8b171f156c60abfebb0a17e1cd888ab113274.jpg)

![22ec1b3a084299c282c18ea67e3da04a23434c24e15fd8c3253326127b2fffa2.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/22ec1b3a084299c282c18ea67e3da04a23434c24e15fd8c3253326127b2fffa2.jpg)

![26ef164211390089a8f6cf3eb79759b8e8585497d17c7986287b6f83081e676f.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/26ef164211390089a8f6cf3eb79759b8e8585497d17c7986287b6f83081e676f.jpg)

![6f8813cbbddd2a4d302daae793518c29220df8c791605c9ab7564d53e4009dc9.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/6f8813cbbddd2a4d302daae793518c29220df8c791605c9ab7564d53e4009dc9.jpg)

![7b043081a57872b810a94ab9d4b526a991d084c9ab124513f2c76353dece64f7.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/7b043081a57872b810a94ab9d4b526a991d084c9ab124513f2c76353dece64f7.jpg)

![929cb0f5db98cbfb2ea18eb92fc479595a01f49c1294600353c0b27f0e8fc457.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/929cb0f5db98cbfb2ea18eb92fc479595a01f49c1294600353c0b27f0e8fc457.jpg)

![9ca7c08cf78275594bb65a77d5de0c4d0c44a6d5f3a2f2fe0f70c154eadd6d52.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/9ca7c08cf78275594bb65a77d5de0c4d0c44a6d5f3a2f2fe0f70c154eadd6d52.jpg)

![a840ac8d2778dff2862d73a35df4176f6e9f9ee3561c3b4ed3b821dd856a9dbe.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/a840ac8d2778dff2862d73a35df4176f6e9f9ee3561c3b4ed3b821dd856a9dbe.jpg)

![b01449c4a3eb103851d53e5ba8b47440871778d26eae1eae9655718afaa3025b.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/b01449c4a3eb103851d53e5ba8b47440871778d26eae1eae9655718afaa3025b.jpg)

![b0c42c5eacdbabff78c160fe333f44bf0633cd83321c6913a4ab5a8f7bcf98c1.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/b0c42c5eacdbabff78c160fe333f44bf0633cd83321c6913a4ab5a8f7bcf98c1.jpg)

![b354deb162927e6e955f40b95e9a24b9cd01824b241776bb3d8429990a702426.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/b354deb162927e6e955f40b95e9a24b9cd01824b241776bb3d8429990a702426.jpg)

![ce5ce1267b377468f635fbd6aadcdb2403b7e4374aa86f9d0973935670b26ecd.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/ce5ce1267b377468f635fbd6aadcdb2403b7e4374aa86f9d0973935670b26ecd.jpg)

![e7681fad405ffa499c59d8b00213461547afeb927cd1a30ce241abbd6eb637d3.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/e7681fad405ffa499c59d8b00213461547afeb927cd1a30ce241abbd6eb637d3.jpg)

![e9b05c232f989be53d6ef38b59677d095087374e21a940e64b5e9e39d8e10f4f.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/e9b05c232f989be53d6ef38b59677d095087374e21a940e64b5e9e39d8e10f4f.jpg)

![f65a0ccf222b78c4e84e7ba13a51266908068a90f581b96524a3c26bfc4ac877.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/images/f65a0ccf222b78c4e84e7ba13a51266908068a90f581b96524a3c26bfc4ac877.jpg)

### Tables

![09e0f6c2245f045c20e59fad3251010b56ae04bfb0db84223ece8cf94e0326d4.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/09e0f6c2245f045c20e59fad3251010b56ae04bfb0db84223ece8cf94e0326d4.jpg)

![1eccf4ce95e4e5ecf8baeb80e0fc7bbac37bda2599ab888118c858babdbe747e.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/1eccf4ce95e4e5ecf8baeb80e0fc7bbac37bda2599ab888118c858babdbe747e.jpg)

![2276c62974cba0dae1d1471387efbccbb027743d2eced1ec97c884a242d85fa1.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/2276c62974cba0dae1d1471387efbccbb027743d2eced1ec97c884a242d85fa1.jpg)

![23a0a592605b465c441f56f34099bdbe52571c0c50d7a06834e0d1ce3e58da62.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/23a0a592605b465c441f56f34099bdbe52571c0c50d7a06834e0d1ce3e58da62.jpg)

![51c332b03ef5cdfab51f4bd9be606ba8adbe2f1b8116184a78250adbdd3d3b26.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/51c332b03ef5cdfab51f4bd9be606ba8adbe2f1b8116184a78250adbdd3d3b26.jpg)

![77dc2e19f17bb0a61cda222b50d1fe8cab717ccf82d6114ccbff24d51e9947b1.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/77dc2e19f17bb0a61cda222b50d1fe8cab717ccf82d6114ccbff24d51e9947b1.jpg)

![a26d1313dc7a9ab5a24c28d476cb14769350c468418985a956f47a849d71c9e3.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/a26d1313dc7a9ab5a24c28d476cb14769350c468418985a956f47a849d71c9e3.jpg)

![b4402d397f0249c378834224aee23e5e4b3d373fdea3ba1bdab11b451bc24c7d.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/b4402d397f0249c378834224aee23e5e4b3d373fdea3ba1bdab11b451bc24c7d.jpg)

![db1506877685d1f8498ef93bc1a01efc1d7a3b7ac4ac6594a7069c2e1d4292b3.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/db1506877685d1f8498ef93bc1a01efc1d7a3b7ac4ac6594a7069c2e1d4292b3.jpg)

![dc1bdfc6aea488fc4a47e20fb4e8556b67d58dfd19b88146a4f5790cccf2bef1.jpg](../icml_results/3621_Interpreting%20Genomic%20Language%20Models%20using%20Sparse%20Autoencoders/tables/dc1bdfc6aea488fc4a47e20fb4e8556b67d58dfd19b88146a4f5790cccf2bef1.jpg)

<a id="paper-3582"></a>

## Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining


### Images

![097111f670c764ecd7c2a09421d55ef5cbbd2e3c9f3758d1f108f9dbdbdc8496.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/097111f670c764ecd7c2a09421d55ef5cbbd2e3c9f3758d1f108f9dbdbdc8496.jpg)

![0c653662a9b268e5415e80c0c94b4dd1b0fb2d80f0bdc99c8557ad80b6f64096.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/0c653662a9b268e5415e80c0c94b4dd1b0fb2d80f0bdc99c8557ad80b6f64096.jpg)

![1ce8429ca9949e90756ba01d5ba079a5450a3ee831cca66dd2f0f3a49d0c3e82.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/1ce8429ca9949e90756ba01d5ba079a5450a3ee831cca66dd2f0f3a49d0c3e82.jpg)

![2c9e2df73308a27581219672da4f3e6c4e0b73357ad75c9c152b9a7f837ed77c.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/2c9e2df73308a27581219672da4f3e6c4e0b73357ad75c9c152b9a7f837ed77c.jpg)

![2d1209f53be1d43240df4ce603e33821d632f5fe3b16a172922b313510c597c5.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/2d1209f53be1d43240df4ce603e33821d632f5fe3b16a172922b313510c597c5.jpg)

![3418874e493236abf91dbd90d4cd2e35ab7da5f45b1e4091df92dcfddc9f0366.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/3418874e493236abf91dbd90d4cd2e35ab7da5f45b1e4091df92dcfddc9f0366.jpg)

![47e92b4e70db8c761ebac37886712097229cd06a4d3cf636ec0f7c161a308a79.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/47e92b4e70db8c761ebac37886712097229cd06a4d3cf636ec0f7c161a308a79.jpg)

![4a0af4e235b4697ff8548141c4f4e3c62bcbbb0635e0ec0c2fc21dfa52e83f1a.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/4a0af4e235b4697ff8548141c4f4e3c62bcbbb0635e0ec0c2fc21dfa52e83f1a.jpg)

![54de876101eb42a8078590a7a66b4c7c21e9723b53dc47dea20763c43445bebf.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/54de876101eb42a8078590a7a66b4c7c21e9723b53dc47dea20763c43445bebf.jpg)

![5a0b1cea1cf12dda6ac7a51e51067734cd845a38f1a9dad62f1f72eeedb60abe.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/5a0b1cea1cf12dda6ac7a51e51067734cd845a38f1a9dad62f1f72eeedb60abe.jpg)

![5a87a6ae2a549898f1ecb8ce3518b66865afdb2dfa37ca6336e1f3d9e9042076.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/5a87a6ae2a549898f1ecb8ce3518b66865afdb2dfa37ca6336e1f3d9e9042076.jpg)

![715c3373dd364bb16811af1feca9a2aaa818e9a441130abfdacda26e2e46c5c3.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/715c3373dd364bb16811af1feca9a2aaa818e9a441130abfdacda26e2e46c5c3.jpg)

![7a02a2081c1588d4396c669b1216d5dccf00104ec118e6fe9e9ca7d527c83292.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/7a02a2081c1588d4396c669b1216d5dccf00104ec118e6fe9e9ca7d527c83292.jpg)

![86dc25139b727168faf6ccd645cd51d463869e6e1d2de489f9fe87538f848bd7.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/86dc25139b727168faf6ccd645cd51d463869e6e1d2de489f9fe87538f848bd7.jpg)

![94b6d5c50ea47e0f8e9aa908ebcf00443bea245ec5433a642ac8f6a4aa275a2e.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/94b6d5c50ea47e0f8e9aa908ebcf00443bea245ec5433a642ac8f6a4aa275a2e.jpg)

![a1297c68776af1998a29c8ea650be9c729ecc17bc365c8125745c864b3439e7d.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/a1297c68776af1998a29c8ea650be9c729ecc17bc365c8125745c864b3439e7d.jpg)

![b18adfcc8e03ba94b3726fb7609a20de2533b418b08a774500889dd80ac08386.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/b18adfcc8e03ba94b3726fb7609a20de2533b418b08a774500889dd80ac08386.jpg)

![e098b0b1c722a9925965be272a5a0affcfbb73b3fe41bb38b53758dfdefa9d29.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/e098b0b1c722a9925965be272a5a0affcfbb73b3fe41bb38b53758dfdefa9d29.jpg)

![e168e73f495704c42f8218cfcfd7cced2b3650d952c89efae625be6c37091e0b.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/e168e73f495704c42f8218cfcfd7cced2b3650d952c89efae625be6c37091e0b.jpg)

![e2c5900de2a7db1d605e605d23a32774ea8505aed9dafa9ce03fdaa8cb8c32e9.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/e2c5900de2a7db1d605e605d23a32774ea8505aed9dafa9ce03fdaa8cb8c32e9.jpg)

![e3d84fe84493d30a1c4a906ad7fc3b065969004e8e99a18c841c783846c93ec2.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/e3d84fe84493d30a1c4a906ad7fc3b065969004e8e99a18c841c783846c93ec2.jpg)

![fdb4e65056463a27a8f6badf223352fdae1fa06d177881d173cca588b3e9435a.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/images/fdb4e65056463a27a8f6badf223352fdae1fa06d177881d173cca588b3e9435a.jpg)

### Tables

![8fa04d52d8f42318ef85cc58f6d96e5530b0d98f7064579d9df8b3514402f0a1.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/tables/8fa04d52d8f42318ef85cc58f6d96e5530b0d98f7064579d9df8b3514402f0a1.jpg)

![d152ebcf7a6e5df26a168de38851da7b6d7cab0aa0f69688aa7a0307049eebb6.jpg](../icml_results/3622_Removing%20Noise%2C%20not%20Finding%20Gold_%20Quality%20Filtering%20for%20Large-Scale%20Pretraining/tables/d152ebcf7a6e5df26a168de38851da7b6d7cab0aa0f69688aa7a0307049eebb6.jpg)

<a id="paper-3583"></a>

## Aitchison Embeddings for Learning Compositional Graph Representations


### Images

![0c9323eb177c1dbcd1e336a6c5b7b08e4d1b4c2435af0b64aae569e59b2146bf.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/images/0c9323eb177c1dbcd1e336a6c5b7b08e4d1b4c2435af0b64aae569e59b2146bf.jpg)

![0f3947efa8aefb90190e6ddeeb480db247112ad6516723875dcb85d816112661.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/images/0f3947efa8aefb90190e6ddeeb480db247112ad6516723875dcb85d816112661.jpg)

![19b0ff62b3645db0872eb19b31abbeb255fe1c81d7bdbf5216d6f861f5d0dbc4.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/images/19b0ff62b3645db0872eb19b31abbeb255fe1c81d7bdbf5216d6f861f5d0dbc4.jpg)

![329bbd17095691ebf5a343d82ae9cd2de2d85ce5b77a4a029bba1b1744e0bec8.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/images/329bbd17095691ebf5a343d82ae9cd2de2d85ce5b77a4a029bba1b1744e0bec8.jpg)

![ad782a398e14736aa8038b327debcbd6136c24bd86bd10d25bae09b801bbea9c.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/images/ad782a398e14736aa8038b327debcbd6136c24bd86bd10d25bae09b801bbea9c.jpg)

### Tables

![2de19d710f30e4b60e7a3b389e1940b6bb01f0abfe7a436fc27c8aa95f634704.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/2de19d710f30e4b60e7a3b389e1940b6bb01f0abfe7a436fc27c8aa95f634704.jpg)

![380a4f10fb990f5e24b4f6a79549764ab75ead8d6306ba529524ed9dc182e9f8.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/380a4f10fb990f5e24b4f6a79549764ab75ead8d6306ba529524ed9dc182e9f8.jpg)

![4e125b71e85e80a33b3be94b208e98f35c99ad6432b9e4d8562152648da79cb3.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/4e125b71e85e80a33b3be94b208e98f35c99ad6432b9e4d8562152648da79cb3.jpg)

![67dd4276f42e91b4d7a0de122ec84585de717c8b4d78ef33cf5c2beb44286e79.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/67dd4276f42e91b4d7a0de122ec84585de717c8b4d78ef33cf5c2beb44286e79.jpg)

![81a0f0a6aaa54aef5a6c6e333d66ca86f086224164b428c637bc6fa348610217.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/81a0f0a6aaa54aef5a6c6e333d66ca86f086224164b428c637bc6fa348610217.jpg)

![ec70dfa3b3d3fe1d1db0b32fbdf47df9b0e8fdfd78a4e48ff918f1c9a8a1c67e.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/ec70dfa3b3d3fe1d1db0b32fbdf47df9b0e8fdfd78a4e48ff918f1c9a8a1c67e.jpg)

![eec2dc212f07e5ab08467318e9def7703ea059e4145e00de17c9c4c75dd81390.jpg](../icml_results/3623_Aitchison%20Embeddings%20for%20Learning%20Compositional%20Graph%20Representations/tables/eec2dc212f07e5ab08467318e9def7703ea059e4145e00de17c9c4c75dd81390.jpg)

<a id="paper-3584"></a>

## FOAM: Frequency and Operator-Error Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo


### Images

![059253d6f0fa3a7ebc3ac1497947fa144729f74a65cd545bff0cf2d535f5274c.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/059253d6f0fa3a7ebc3ac1497947fa144729f74a65cd545bff0cf2d535f5274c.jpg)

![0e1f396c30bb2b56dea0b0ee85d3376d170e0a92d204d763a67d839af24ef380.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/0e1f396c30bb2b56dea0b0ee85d3376d170e0a92d204d763a67d839af24ef380.jpg)

![2a1f8c7bd62375c0cf025997c6e7dbea135dd175f3f06701cac8a8f57d181e82.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/2a1f8c7bd62375c0cf025997c6e7dbea135dd175f3f06701cac8a8f57d181e82.jpg)

![2c39b02b73dbd32243f5a734a68b042015a17f7da22737dd42ecaffde6d31ff4.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/2c39b02b73dbd32243f5a734a68b042015a17f7da22737dd42ecaffde6d31ff4.jpg)

![46f2d6ff13c3e9e1ae603f4aaeb29c873a201bba2f94be073cb7dc3fd76a1687.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/46f2d6ff13c3e9e1ae603f4aaeb29c873a201bba2f94be073cb7dc3fd76a1687.jpg)

![5b59a94910047e3bea8215f7a0921438b68566176a5a4ce7141f1d71395347fb.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/5b59a94910047e3bea8215f7a0921438b68566176a5a4ce7141f1d71395347fb.jpg)

![7b6dc53a71908ecedde78f823159e0aa73d80839dab12c26c28990a71b32aedd.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/7b6dc53a71908ecedde78f823159e0aa73d80839dab12c26c28990a71b32aedd.jpg)

![a36812ca59576b609e166a973c57798143861cebd6fd0aa1b1f1cd6200f52d48.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/a36812ca59576b609e166a973c57798143861cebd6fd0aa1b1f1cd6200f52d48.jpg)

![d34a3d0b0026eee78f8e0fa27368abe02706122e7b2c1117ef659588d5bf82ae.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/d34a3d0b0026eee78f8e0fa27368abe02706122e7b2c1117ef659588d5bf82ae.jpg)

![f3960e7791923b9bc50fbc1a43d64507b8e2f7a593bdc6f5002d0576e21e7c00.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/images/f3960e7791923b9bc50fbc1a43d64507b8e2f7a593bdc6f5002d0576e21e7c00.jpg)

### Tables

![0324123958079abcd11f355aec4dff72b69d5cebe1ce443fdd4df990b506db74.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/0324123958079abcd11f355aec4dff72b69d5cebe1ce443fdd4df990b506db74.jpg)

![046c64bb1cdc8a7d149e59c3b18a6042c3863441e14317ee6b7c895d515f44ce.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/046c64bb1cdc8a7d149e59c3b18a6042c3863441e14317ee6b7c895d515f44ce.jpg)

![62f3cf1a3ef32c485c209051fd818e04847c91e29e5bee560fa9508bd8a443c6.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/62f3cf1a3ef32c485c209051fd818e04847c91e29e5bee560fa9508bd8a443c6.jpg)

![7938ddc351225e0113fd8a1e666722bdb7ff4958978267eb2a51639cae22000d.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/7938ddc351225e0113fd8a1e666722bdb7ff4958978267eb2a51639cae22000d.jpg)

![7a5735c289f900cb43632b3f7294da2f94c064a80af197b64e91ce24e286bd65.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/7a5735c289f900cb43632b3f7294da2f94c064a80af197b64e91ce24e286bd65.jpg)

![8ad71d417671dc295821662726158ded4aadb523be92de9ca3014663e4c196bc.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/8ad71d417671dc295821662726158ded4aadb523be92de9ca3014663e4c196bc.jpg)

![a101ed8ac02a241f9af7e51cb6f3d32ceba4fcea5d298559eb2cfcf65e3e0669.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/a101ed8ac02a241f9af7e51cb6f3d32ceba4fcea5d298559eb2cfcf65e3e0669.jpg)

![b192566e797adae7dc0dcd2b237885bd4b151bb2911d7ff216ac5bef0c573725.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/b192566e797adae7dc0dcd2b237885bd4b151bb2911d7ff216ac5bef0c573725.jpg)

![b3bfd3406a31f42efbda344b16c97ad6f4fe604233cfd8c01b939d4184db3828.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/b3bfd3406a31f42efbda344b16c97ad6f4fe604233cfd8c01b939d4184db3828.jpg)

![c6f787362b0c8c6222018c1690ddc0a9a527010626e46d879fa2acc62ee9f872.jpg](../icml_results/3624_FOAM_%20Frequency%20and%20Operator-Error%20Based%20Adaptive%20Damping%20Method%20for%20Reducing%20Staleness-Oriented%20Err/tables/c6f787362b0c8c6222018c1690ddc0a9a527010626e46d879fa2acc62ee9f872.jpg)

<a id="paper-3585"></a>

## Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data


### Images

![122b04e7d1a10649408d8d09b9dd78a9a5b2783295e83325152ce38d7a2c5808.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/images/122b04e7d1a10649408d8d09b9dd78a9a5b2783295e83325152ce38d7a2c5808.jpg)

![29fab3a6379c04839905035d998f5ba9b5cc82d009e4587d0ed7862d6a4fd87f.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/images/29fab3a6379c04839905035d998f5ba9b5cc82d009e4587d0ed7862d6a4fd87f.jpg)

![553bd48cfc4abd1183b32e4c3ba7664f2a78c32d8514ee0c16dc28a9135bee7b.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/images/553bd48cfc4abd1183b32e4c3ba7664f2a78c32d8514ee0c16dc28a9135bee7b.jpg)

![6ae417bc8449f92ed3e8b510efd7550c04029df8ceaaba9bd8e1aa697022c040.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/images/6ae417bc8449f92ed3e8b510efd7550c04029df8ceaaba9bd8e1aa697022c040.jpg)

![d24fed7f8dbd658e3a29a74c5030f1e55b95cf8f6d56f8fa981d00ea7ce24afa.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/images/d24fed7f8dbd658e3a29a74c5030f1e55b95cf8f6d56f8fa981d00ea7ce24afa.jpg)

### Tables

![06c3eac4d63055be48ed7a77e220b6f33ceb1bc4b7cb16433fbdad275b396720.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/06c3eac4d63055be48ed7a77e220b6f33ceb1bc4b7cb16433fbdad275b396720.jpg)

![2e285acb6d605b8531361ad5c4f7ed283443eedfea265300d6a308b7843e8c8e.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/2e285acb6d605b8531361ad5c4f7ed283443eedfea265300d6a308b7843e8c8e.jpg)

![80f656ceeb6fe08701bf1f6d92af89646aa23985f474da907cf5e6710c306b71.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/80f656ceeb6fe08701bf1f6d92af89646aa23985f474da907cf5e6710c306b71.jpg)

![bc2a8fc7e30f6ce5881c5a9dd1924b0b236bb2771340805a2a0268a640af0308.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/bc2a8fc7e30f6ce5881c5a9dd1924b0b236bb2771340805a2a0268a640af0308.jpg)

![c666a60c692ecb3ed98bd702ea53ad2ef59b1f2874955338459d034793c9b8b6.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/c666a60c692ecb3ed98bd702ea53ad2ef59b1f2874955338459d034793c9b8b6.jpg)

![c6aa7294a704c53d46622ea22ba2a19e0a7da99e60a2ed784f3d579571ccaa4b.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/c6aa7294a704c53d46622ea22ba2a19e0a7da99e60a2ed784f3d579571ccaa4b.jpg)

![fc5416500e02f710204725979aca414bc46703c138f6058e1dfaa58ad21cf169.jpg](../icml_results/3625_Maximizing%20Mutual%20Information%20Between%20Prompt%20and%20Response%20Improves%20LLM%20Performance%20With%20No%20Additiona/tables/fc5416500e02f710204725979aca414bc46703c138f6058e1dfaa58ad21cf169.jpg)

<a id="paper-3586"></a>

## Causal-JEPA: Learning World Models through Object-Level Latent Masking


### Images

![0cbd752068ea0fb36f1689bcebbb091287d442ae93ac398f0c40ba5dc32d47be.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/0cbd752068ea0fb36f1689bcebbb091287d442ae93ac398f0c40ba5dc32d47be.jpg)

![2a40a145ba9243d8ab907d1515d4c8c67a280ffc69f9072bdf0647c4f9046ed7.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/2a40a145ba9243d8ab907d1515d4c8c67a280ffc69f9072bdf0647c4f9046ed7.jpg)

![4475870e3318bf383b889dd172fe8e72b4386ff1566f4543325cd00ba0b3cd5b.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/4475870e3318bf383b889dd172fe8e72b4386ff1566f4543325cd00ba0b3cd5b.jpg)

![5e85ea8ec20086d4e2044520fa020736b9fb908f5aa0f0e44529af422cd17ef1.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/5e85ea8ec20086d4e2044520fa020736b9fb908f5aa0f0e44529af422cd17ef1.jpg)

![b98aadf46d657a4dbb58404bc256c26bc15ada5c68ef9da2c34be8621c97d0cc.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/b98aadf46d657a4dbb58404bc256c26bc15ada5c68ef9da2c34be8621c97d0cc.jpg)

![be493e8aa366e480dde90b59f604467a44a156541766d08f7ef2d64ec6a8584a.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/be493e8aa366e480dde90b59f604467a44a156541766d08f7ef2d64ec6a8584a.jpg)

![c85b12b952ee55e7de9447dea42732b0320314730e71ff50a1475e67f902aa8d.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/images/c85b12b952ee55e7de9447dea42732b0320314730e71ff50a1475e67f902aa8d.jpg)

### Tables

![27fa89d37b690a683b8a3a6e667d54b847e5b2012520b26424fb2f1155c6fdd8.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/27fa89d37b690a683b8a3a6e667d54b847e5b2012520b26424fb2f1155c6fdd8.jpg)

![36644e684d901387b503af034f0f045bca6175a790412bd6be570564551014f5.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/36644e684d901387b503af034f0f045bca6175a790412bd6be570564551014f5.jpg)

![694e3cba7d19a2332fa23a06687ea960b264fe2800f39053717621c4f7b16ccd.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/694e3cba7d19a2332fa23a06687ea960b264fe2800f39053717621c4f7b16ccd.jpg)

![6b30d048b2575ea316584cdb4bc14d6e4e5e9cca86163a82a44741a6a08c1aed.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/6b30d048b2575ea316584cdb4bc14d6e4e5e9cca86163a82a44741a6a08c1aed.jpg)

![7533d31b68895340ec6797d004b8f57c0560e684ebe52ad957a27a474385675f.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/7533d31b68895340ec6797d004b8f57c0560e684ebe52ad957a27a474385675f.jpg)

![80265474e74258f077345f0b04b53ff1c71033074435ba7ef65912b23c3d9048.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/80265474e74258f077345f0b04b53ff1c71033074435ba7ef65912b23c3d9048.jpg)

![b2a148cb044518062d9f3368e9f62eebbd0a3577ffaf97c57c52027ea3eb9b33.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/b2a148cb044518062d9f3368e9f62eebbd0a3577ffaf97c57c52027ea3eb9b33.jpg)

![e0596f36c4c9b93daff0a0f84eebc5ee9c626007e2e7cb0cba3150c2f40c9d1f.jpg](../icml_results/3626_Causal-JEPA_%20Learning%20World%20Models%20through%20Object-Level%20Latent%20Masking/tables/e0596f36c4c9b93daff0a0f84eebc5ee9c626007e2e7cb0cba3150c2f40c9d1f.jpg)

<a id="paper-3587"></a>

## The Invisible Lottery: How Subtle Cues Steer Algorithm Choice in LLM Code Generation


### Images

![05ed41b1ecd830a6d8ba0e34d9e0d80b3125215c9fc08772ef67f5001fd8e9e8.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/05ed41b1ecd830a6d8ba0e34d9e0d80b3125215c9fc08772ef67f5001fd8e9e8.jpg)

![2b81b0e4bf464198d58ca7b92ea16c44b1f15a342620f82ab62bc4dbe4c09e3c.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/2b81b0e4bf464198d58ca7b92ea16c44b1f15a342620f82ab62bc4dbe4c09e3c.jpg)

![327adcc703aff45e21a0fcf035e4c4bba19fbe8d10e4cc0f84976abee23eeae4.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/327adcc703aff45e21a0fcf035e4c4bba19fbe8d10e4cc0f84976abee23eeae4.jpg)

![84c4d98f1bafae4c9fe1d428c2addf7ec20c7ca09106e34a44d40947b26664eb.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/84c4d98f1bafae4c9fe1d428c2addf7ec20c7ca09106e34a44d40947b26664eb.jpg)

![936b5a573eaaeaffce6095ff5d2cd283666595b8e6f5cb586a6c94b8c929b6a4.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/936b5a573eaaeaffce6095ff5d2cd283666595b8e6f5cb586a6c94b8c929b6a4.jpg)

![eba99e7a50a31ec61ccaebf6af8f1739a7f637b4afcbbfd524ff55b0331f18a6.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/eba99e7a50a31ec61ccaebf6af8f1739a7f637b4afcbbfd524ff55b0331f18a6.jpg)

![ff73b85500507953870657d49444898c9918647ab52082529949b6aa02387618.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/images/ff73b85500507953870657d49444898c9918647ab52082529949b6aa02387618.jpg)

### Tables

![06c60a1ebbd94f34965cc83cd3de424aab1442e1621f235716a6c51a53698880.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/06c60a1ebbd94f34965cc83cd3de424aab1442e1621f235716a6c51a53698880.jpg)

![1fc496008f0f1f8e8a1351479814649f0f28a413dada4c950f3c7748f2a7bf23.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/1fc496008f0f1f8e8a1351479814649f0f28a413dada4c950f3c7748f2a7bf23.jpg)

![291fdf1b7aac6f8eabb84733e4f86cee8910c4d37705a553d598e3f786590bf5.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/291fdf1b7aac6f8eabb84733e4f86cee8910c4d37705a553d598e3f786590bf5.jpg)

![294046bfc328186291153fa2e6962cf5e0019ffc7757ac5526b69dfa94855c30.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/294046bfc328186291153fa2e6962cf5e0019ffc7757ac5526b69dfa94855c30.jpg)

![3f422a4d65bb03b9220255df7b08dc297b0486fa1e867df694fc8bccbd30ee11.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/3f422a4d65bb03b9220255df7b08dc297b0486fa1e867df694fc8bccbd30ee11.jpg)

![4587afe50b69c24ec223439210987b889e59861854abcb3aebeb78128d447547.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/4587afe50b69c24ec223439210987b889e59861854abcb3aebeb78128d447547.jpg)

![4cc078a8970c2008d9f3195a4479696ecd3c53821d15754e5722db8f09b79e93.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/4cc078a8970c2008d9f3195a4479696ecd3c53821d15754e5722db8f09b79e93.jpg)

![51e9ca09da02edce5ad50b86caacb3d0686ac0146b24cc880942f7a5248aa135.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/51e9ca09da02edce5ad50b86caacb3d0686ac0146b24cc880942f7a5248aa135.jpg)

![586dda363fdb5fa216e4730bca6dbaabc29882b928ee7f5d4f0440e967cdc2ef.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/586dda363fdb5fa216e4730bca6dbaabc29882b928ee7f5d4f0440e967cdc2ef.jpg)

![5c6e7bda072b68f5a6877a1148c2fc7b08614da8b735d717f8a98a43b7142c99.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/5c6e7bda072b68f5a6877a1148c2fc7b08614da8b735d717f8a98a43b7142c99.jpg)

![a1d370d83d869efc909e644eacd2cb508c10f0a653f9d2a6cf2c809e1a17f5fb.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/a1d370d83d869efc909e644eacd2cb508c10f0a653f9d2a6cf2c809e1a17f5fb.jpg)

![a70d571fbde4ace224253e56eb102319a2139d246c933e355ce2123f5d3b8786.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/a70d571fbde4ace224253e56eb102319a2139d246c933e355ce2123f5d3b8786.jpg)

![d092b20bc0d42dae61161e6ffa6d1fc5275589e28d41e154f836b6ae8ad923b3.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/d092b20bc0d42dae61161e6ffa6d1fc5275589e28d41e154f836b6ae8ad923b3.jpg)

![da05f35666ac37cbba9ffe6c357be7cf19bbc917937862e5270e5ebee5873a1b.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/da05f35666ac37cbba9ffe6c357be7cf19bbc917937862e5270e5ebee5873a1b.jpg)

![eacc6871bb4115005428cbd25aa3642504c1185bb196d90929c77f8561a97894.jpg](../icml_results/3627_The%20Invisible%20Lottery_%20How%20Subtle%20Cues%20Steer%20Algorithm%20Choice%20in%20LLM%20Code%20Generation/tables/eacc6871bb4115005428cbd25aa3642504c1185bb196d90929c77f8561a97894.jpg)

<a id="paper-3588"></a>

## Telescope: Improving Zero Shot Detection of LLM Generated Content By Measuring Token Repetition Probability


### Images

![12ed2bacfcb1a4d1860341d5f8cb6d902cae37bfa482346d4b7ec766b790bd7d.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/12ed2bacfcb1a4d1860341d5f8cb6d902cae37bfa482346d4b7ec766b790bd7d.jpg)

![1a5d1bebeb473f836924089faff8d52f06d69bfc99a0493d57fbd079972d94aa.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/1a5d1bebeb473f836924089faff8d52f06d69bfc99a0493d57fbd079972d94aa.jpg)

![1b467482043795f8a0f7ff04fbb2d755dc33742e53510b826fba102f326378b0.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/1b467482043795f8a0f7ff04fbb2d755dc33742e53510b826fba102f326378b0.jpg)

![2d6ca2f9b26a168f25cb0764dbe2eeada05f10ad7ee6298be720d5ce2cdb2cc6.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/2d6ca2f9b26a168f25cb0764dbe2eeada05f10ad7ee6298be720d5ce2cdb2cc6.jpg)

![3570b5f9cde55450b281b5f67b90ca619643e13c8c8f3fc9a3f7306135b5dedd.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/3570b5f9cde55450b281b5f67b90ca619643e13c8c8f3fc9a3f7306135b5dedd.jpg)

![433112f06ce6c34cfef208f40a4bdec08908f220da06de0ee50e9a9d095c79f8.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/433112f06ce6c34cfef208f40a4bdec08908f220da06de0ee50e9a9d095c79f8.jpg)

![46f007547b794a56dc47ffc3d88deeb934bfd8cf9f45387aa9ac534bafd4c5d1.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/46f007547b794a56dc47ffc3d88deeb934bfd8cf9f45387aa9ac534bafd4c5d1.jpg)

![5861acfa9e2310996e62f6059b11203b570099c3dbb3fe7244bbb0daa723a833.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/5861acfa9e2310996e62f6059b11203b570099c3dbb3fe7244bbb0daa723a833.jpg)

![59bb73093934c3915ffc7440918fdf34d2942609ca19b35add25af2463dcda91.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/59bb73093934c3915ffc7440918fdf34d2942609ca19b35add25af2463dcda91.jpg)

![5cb2502b20665812fd4a1dfa64b5e70e63419a99ec0e4f0d105055c2dedfad40.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/5cb2502b20665812fd4a1dfa64b5e70e63419a99ec0e4f0d105055c2dedfad40.jpg)

![65a33649cfbdcba07a03b180eb8bde1441e5ebd1ca3b37b191eb58f0f4a2bc5c.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/65a33649cfbdcba07a03b180eb8bde1441e5ebd1ca3b37b191eb58f0f4a2bc5c.jpg)

![7fb30017e55d634a66697fa2de6a17051fcca52169c82d0127a0e0cdde1c7b60.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/7fb30017e55d634a66697fa2de6a17051fcca52169c82d0127a0e0cdde1c7b60.jpg)

![85128af44c9d7a5b2646577448d3a1dc07ee69467974f0005bcdbb9c749c75b9.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/85128af44c9d7a5b2646577448d3a1dc07ee69467974f0005bcdbb9c749c75b9.jpg)

![990954ff52220ee7628fa020d643229f28af6400df3013220d2ffce237fb1e11.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/990954ff52220ee7628fa020d643229f28af6400df3013220d2ffce237fb1e11.jpg)

![aa6662c00e551a61948124364327c7fa0e190f9eeb484f86ea45e46f39701ae0.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/aa6662c00e551a61948124364327c7fa0e190f9eeb484f86ea45e46f39701ae0.jpg)

![b456cb3a012624960b602c28a4ad892003a565cdf356f67de1c3eb9f7cf82047.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/b456cb3a012624960b602c28a4ad892003a565cdf356f67de1c3eb9f7cf82047.jpg)

![be9bca4f0a47e08c092c47bf87a3db677edd42c722bc16478750de18ce7f2a25.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/be9bca4f0a47e08c092c47bf87a3db677edd42c722bc16478750de18ce7f2a25.jpg)

![c30b1578744bde93e326a3fc19d77322a7c804d0781900f06d7fbe2321975ffe.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/c30b1578744bde93e326a3fc19d77322a7c804d0781900f06d7fbe2321975ffe.jpg)

![c7aff385da4266f555a759e3d22ae6dca141286beabe85e6cb33150ee059090b.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/c7aff385da4266f555a759e3d22ae6dca141286beabe85e6cb33150ee059090b.jpg)

![e14903158d92926c3a5e5dbcc9824cefcce24ef111487b8e2c2480bd09031ee3.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/e14903158d92926c3a5e5dbcc9824cefcce24ef111487b8e2c2480bd09031ee3.jpg)

![efd3f3ebcf303f34cc6868f3238af44e698e5debdfcd48fd0aeb0e62151df040.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/efd3f3ebcf303f34cc6868f3238af44e698e5debdfcd48fd0aeb0e62151df040.jpg)

![f6d15b9b12caba94276207091534a4d4ad239e3980de77071afc6906afa13b9b.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/f6d15b9b12caba94276207091534a4d4ad239e3980de77071afc6906afa13b9b.jpg)

![f9e5c60d385005dcb7231e94cc96db4bf1006b78c44eef53caa755203b94623b.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/images/f9e5c60d385005dcb7231e94cc96db4bf1006b78c44eef53caa755203b94623b.jpg)

### Tables

![0073f7a0b029035055c6721f3ccf4f3dd96eea200a6520ebec1a382bf6f5c0f3.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/0073f7a0b029035055c6721f3ccf4f3dd96eea200a6520ebec1a382bf6f5c0f3.jpg)

![0d0d213b889e79fb7120f3f82537af72855a9799f9830a8f2e19e5abcca7ff66.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/0d0d213b889e79fb7120f3f82537af72855a9799f9830a8f2e19e5abcca7ff66.jpg)

![11c8ee86563e1fe176a156f19f4acb47c54dd280b874b0f1b42cf03a747e268c.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/11c8ee86563e1fe176a156f19f4acb47c54dd280b874b0f1b42cf03a747e268c.jpg)

![15fa34209c320042465e27c697742663fcaf53458c9209532a0d9254229fdfa3.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/15fa34209c320042465e27c697742663fcaf53458c9209532a0d9254229fdfa3.jpg)

![1cc4c157ee4034999077fd42ebb83bbd1051fac6a5fc33b625dd088cf15ef483.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/1cc4c157ee4034999077fd42ebb83bbd1051fac6a5fc33b625dd088cf15ef483.jpg)

![24eeba926d4f1a7ae194585d514995d7df392bfac4b81d65fcd442c151417e64.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/24eeba926d4f1a7ae194585d514995d7df392bfac4b81d65fcd442c151417e64.jpg)

![2e47a026e88452530becc3fcbe71bf6d4f6fdd5ee486a6dba0518a46bc685242.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/2e47a026e88452530becc3fcbe71bf6d4f6fdd5ee486a6dba0518a46bc685242.jpg)

![40a4a24ae5aefd013e6494a4b07ef6f24647b59bd658c663a2dc67e23a8e5221.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/40a4a24ae5aefd013e6494a4b07ef6f24647b59bd658c663a2dc67e23a8e5221.jpg)

![4212fc06e4296b6cd65f22a74068029b96380dcc0520a578a36477291d568333.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/4212fc06e4296b6cd65f22a74068029b96380dcc0520a578a36477291d568333.jpg)

![5141e3c9e9459d2432192023a733eb4fd655f16125f35832f0e797e5b3a8334e.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/5141e3c9e9459d2432192023a733eb4fd655f16125f35832f0e797e5b3a8334e.jpg)

![5416978d14ac54ed12f2dce52f7b23f4d76355e4d21b15376dafce5a5bb7c6cc.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/5416978d14ac54ed12f2dce52f7b23f4d76355e4d21b15376dafce5a5bb7c6cc.jpg)

![62ddf6526bb3343a2135f4f5e2ace1b33db48f15e47cc88cde9ab0b3964aac4f.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/62ddf6526bb3343a2135f4f5e2ace1b33db48f15e47cc88cde9ab0b3964aac4f.jpg)

![653239f5d83fe7a214f9f80bf01c1f0f1f0b158695f050f66be79d12a0c83c61.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/653239f5d83fe7a214f9f80bf01c1f0f1f0b158695f050f66be79d12a0c83c61.jpg)

![68cdfe9811309a89e9f1bd77586b8d92fad4578190ced054c6a9d1d4a579596f.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/68cdfe9811309a89e9f1bd77586b8d92fad4578190ced054c6a9d1d4a579596f.jpg)

![69ced538fd314703ec26256b9c4eee4c6f60736a1cfa655114b6fbcc1cfc548e.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/69ced538fd314703ec26256b9c4eee4c6f60736a1cfa655114b6fbcc1cfc548e.jpg)

![6aab70fcabd56ca8ff440ae3d40ca93dccbf5586ebf20fe0e0a6b9e4135f1fad.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/6aab70fcabd56ca8ff440ae3d40ca93dccbf5586ebf20fe0e0a6b9e4135f1fad.jpg)

![6d6e06cb35f4d87095ce469088d099bdeb9e4f04fbdfe712f428de00acb0189e.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/6d6e06cb35f4d87095ce469088d099bdeb9e4f04fbdfe712f428de00acb0189e.jpg)

![7e2bf945f83234579f5afa1e2cc31d16c576e3d3e2789982723b1e549e497fe6.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/7e2bf945f83234579f5afa1e2cc31d16c576e3d3e2789982723b1e549e497fe6.jpg)

![867ed06ae07314c8fcf2aa7eadfaf82ea6ed80f0121a0c1e577a1fd7fad852ab.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/867ed06ae07314c8fcf2aa7eadfaf82ea6ed80f0121a0c1e577a1fd7fad852ab.jpg)

![87168fe5525bb263fbf13a17a8a789633ea03eacaafc12336f9930807dd68856.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/87168fe5525bb263fbf13a17a8a789633ea03eacaafc12336f9930807dd68856.jpg)

![8ce71de08426618d8f86e441ae9b8df0a0d6be31903b9d47769ab35d0db717e0.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/8ce71de08426618d8f86e441ae9b8df0a0d6be31903b9d47769ab35d0db717e0.jpg)

![987f08e0df2bb286199942e75ef6e6e7d04062bf7886e08bd30a8a05cb0fbd88.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/987f08e0df2bb286199942e75ef6e6e7d04062bf7886e08bd30a8a05cb0fbd88.jpg)

![a1ced814b17af7fc80beb2e4a707c8f860e6932fd036e8cfec1d46e4e44c76db.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/a1ced814b17af7fc80beb2e4a707c8f860e6932fd036e8cfec1d46e4e44c76db.jpg)

![a2bedf14fb6037dbe56d40409aa37998fc5bb3a18dd2d03485c696e4840329b3.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/a2bedf14fb6037dbe56d40409aa37998fc5bb3a18dd2d03485c696e4840329b3.jpg)

![ad6f5910ad0f5ebf05ae0db2ec72bfd48f60c708339c260f84de79b0349bebec.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/ad6f5910ad0f5ebf05ae0db2ec72bfd48f60c708339c260f84de79b0349bebec.jpg)

![b428a879347b503a17695c6e47d9095f6a20a764ea1943c3ebe6796b8e3f6019.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/b428a879347b503a17695c6e47d9095f6a20a764ea1943c3ebe6796b8e3f6019.jpg)

![b9064bb5ea3e40505b93a64091e9d4906a606f0c5a3474cd16317489664a0ef2.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/b9064bb5ea3e40505b93a64091e9d4906a606f0c5a3474cd16317489664a0ef2.jpg)

![c64a75aaac76954cf342bce9d2d0e932f3ee41a173852648a25c4992eae74524.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/c64a75aaac76954cf342bce9d2d0e932f3ee41a173852648a25c4992eae74524.jpg)

![d0c66c621da4bc38fe3d5549ef66a5ca9f4383ec7095dd410be42a8c164c4437.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/d0c66c621da4bc38fe3d5549ef66a5ca9f4383ec7095dd410be42a8c164c4437.jpg)

![d26d75aa26d2fc3d48831836fdf890dac0dfbd6c6f281443b11776239373c8e3.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/d26d75aa26d2fc3d48831836fdf890dac0dfbd6c6f281443b11776239373c8e3.jpg)

![d2a30668b58659b610c2b4eb8fb2b98e274603f5c158c166cd6475106fe7210d.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/d2a30668b58659b610c2b4eb8fb2b98e274603f5c158c166cd6475106fe7210d.jpg)

![d6dc950dc4cc47164b0e321dd67271806def3d8cc3fd02fa8a50131098a07d77.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/d6dc950dc4cc47164b0e321dd67271806def3d8cc3fd02fa8a50131098a07d77.jpg)

![dff1a4a8560675b39ac4284baf21ffea908b429ef77bf851a4fecc4d1c64011a.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/dff1a4a8560675b39ac4284baf21ffea908b429ef77bf851a4fecc4d1c64011a.jpg)

![f645264215d3f164ff0ee13fd7ba5b079edb4f9fd9c36ffe05805c9715c3cb8e.jpg](../icml_results/3628_Telescope_%20Improving%20Zero%20Shot%20Detection%20of%20LLM%20Generated%20Content%20By%20Measuring%20Token%20Repetition%20Prob/tables/f645264215d3f164ff0ee13fd7ba5b079edb4f9fd9c36ffe05805c9715c3cb8e.jpg)

<a id="paper-3589"></a>

## One Probe Won’t Catch Them All: Towards Targeted Deception Detection


### Images

![4645e32c0711174f64442a59f2f2751b4ded86a7f949118a7862bf9e4bc00589.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/4645e32c0711174f64442a59f2f2751b4ded86a7f949118a7862bf9e4bc00589.jpg)

![484f9d59af71994946a1bb3d2d522a70e3583cfc35e9d7cdd3acba72bec9a220.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/484f9d59af71994946a1bb3d2d522a70e3583cfc35e9d7cdd3acba72bec9a220.jpg)

![5e1d571ed13f947a3c5616aa5ef8090685de5f75ffc1957dde72a7d4b605c49d.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/5e1d571ed13f947a3c5616aa5ef8090685de5f75ffc1957dde72a7d4b605c49d.jpg)

![7f43997ec17f536d2b596b6706843bbb5131ef53f2028f020789fe7c8020de4e.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/7f43997ec17f536d2b596b6706843bbb5131ef53f2028f020789fe7c8020de4e.jpg)

![9146d2f375b09650762be4818476c889303945e85d287f99dcba3022d5deb710.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/9146d2f375b09650762be4818476c889303945e85d287f99dcba3022d5deb710.jpg)

![aad4eb5b5141a2905b7684c7ca05d57ab85ce7b7ac90f016b375fa5abbfe75ed.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/aad4eb5b5141a2905b7684c7ca05d57ab85ce7b7ac90f016b375fa5abbfe75ed.jpg)

![b09a7b20405f1876d8170a2d0e11ea35ec3f951ef0e9f2579d398e48dfda0434.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/b09a7b20405f1876d8170a2d0e11ea35ec3f951ef0e9f2579d398e48dfda0434.jpg)

![b9d07345c90440ebbd0d60c4cdd51d1b3072c6ed1d41e60bd03aac862b7b9e98.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/b9d07345c90440ebbd0d60c4cdd51d1b3072c6ed1d41e60bd03aac862b7b9e98.jpg)

![c0c22d5e9565f905eac6e24f952149ee6a04cd82cbab56096e3cc401819d4bdd.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/images/c0c22d5e9565f905eac6e24f952149ee6a04cd82cbab56096e3cc401819d4bdd.jpg)

### Tables

![40b58492ef2d8c0dadcb017e4fe60f918ee1408810dd909a3bad36af435caacf.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/tables/40b58492ef2d8c0dadcb017e4fe60f918ee1408810dd909a3bad36af435caacf.jpg)

![5590d583cdb6425edc89f3e7bb625a6b77930608ec204c09fffe4fea62155aa1.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/tables/5590d583cdb6425edc89f3e7bb625a6b77930608ec204c09fffe4fea62155aa1.jpg)

![8e25975838c8d60468fd55d27dac13e651c7de90871c2eda356bcab919f6c09c.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/tables/8e25975838c8d60468fd55d27dac13e651c7de90871c2eda356bcab919f6c09c.jpg)

![9e5d2f1c0667fa6a8a790e9513d4f764b710f3e661b906acec2d681dbd0c0308.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/tables/9e5d2f1c0667fa6a8a790e9513d4f764b710f3e661b906acec2d681dbd0c0308.jpg)

![ec01ce7ce7f6ddbd343d43153509ddcbb1b034ccdf780df4ace52a98c8b6fdf7.jpg](../icml_results/3629_One%20Probe%20Won%E2%80%99t%20Catch%20Them%20All_%20Towards%20Targeted%20Deception%20Detection/tables/ec01ce7ce7f6ddbd343d43153509ddcbb1b034ccdf780df4ace52a98c8b6fdf7.jpg)

<a id="paper-3590"></a>

## Learning Dynamic Stability Landscapes in Synchronization Networks


### Images

![03bde06d4b389d98af947cfcad1295f14cd8878b4ef60c3a7460c54d1db9113f.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/03bde06d4b389d98af947cfcad1295f14cd8878b4ef60c3a7460c54d1db9113f.jpg)

![05a415f9f87bb2a5e2c68dbc800e15ca510e0cbc1e2c3fbbb4967967adf584ad.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/05a415f9f87bb2a5e2c68dbc800e15ca510e0cbc1e2c3fbbb4967967adf584ad.jpg)

![12b07b606d6f943aac1beb1f1fcd1464078e9850210b61bc385e6cc196c127be.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/12b07b606d6f943aac1beb1f1fcd1464078e9850210b61bc385e6cc196c127be.jpg)

![28c3de73df6988db42ca1bc207c4a848e0e2f0d4212d3c64fb6c189134cbee6a.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/28c3de73df6988db42ca1bc207c4a848e0e2f0d4212d3c64fb6c189134cbee6a.jpg)

![297d443299a9629c62e26353e079ec2b43c573ac37d897ceccb23c7d60a98f24.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/297d443299a9629c62e26353e079ec2b43c573ac37d897ceccb23c7d60a98f24.jpg)

![415fa4bc9e699ad8e9dc50ae5391865bd84ad424a49e54c1fc4a42880ac54af1.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/415fa4bc9e699ad8e9dc50ae5391865bd84ad424a49e54c1fc4a42880ac54af1.jpg)

![4f933a3a2583498ec3d7f4e2b140a925d748c9e8c47912ea67a6f877913209d4.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/4f933a3a2583498ec3d7f4e2b140a925d748c9e8c47912ea67a6f877913209d4.jpg)

![54bbdf09cf8f225a4e2be2f6087c69b2081e21640efa08c957297b2160069f04.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/54bbdf09cf8f225a4e2be2f6087c69b2081e21640efa08c957297b2160069f04.jpg)

![8cd6ec388af94833bc0cb934d24ed2564db2bc51b1e645a5292d618c8e070ead.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/8cd6ec388af94833bc0cb934d24ed2564db2bc51b1e645a5292d618c8e070ead.jpg)

![a04cc4974fc032f0d3b345138b4503ca214aea336f974fac9445c8651191f7b0.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/a04cc4974fc032f0d3b345138b4503ca214aea336f974fac9445c8651191f7b0.jpg)

![a156fb53ddaf4e552fc46f3526c7108c0737f9119e9cd664d8ae009743d6101e.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/a156fb53ddaf4e552fc46f3526c7108c0737f9119e9cd664d8ae009743d6101e.jpg)

![ae394ea75032cd93ffdd336853fccbc997617d46d5dad2cdc331d5ac780f206a.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/images/ae394ea75032cd93ffdd336853fccbc997617d46d5dad2cdc331d5ac780f206a.jpg)

### Tables

![0274ac13c02ee993cec033d33cde5cd97dd457118a99b474057cb9dca6eef9ae.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/0274ac13c02ee993cec033d33cde5cd97dd457118a99b474057cb9dca6eef9ae.jpg)

![0745d8e3ab791edf2fce0eded9a3aa24c29c0636ea3d19498048cded7ef2def2.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/0745d8e3ab791edf2fce0eded9a3aa24c29c0636ea3d19498048cded7ef2def2.jpg)

![25391dd962a654565c375f4c2cd765f5ff0d9ea64cf88655c033787c9a782af2.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/25391dd962a654565c375f4c2cd765f5ff0d9ea64cf88655c033787c9a782af2.jpg)

![34c0ee6b382f667210a5092c1a228e9d86c541582bca835d6752c022338e5107.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/34c0ee6b382f667210a5092c1a228e9d86c541582bca835d6752c022338e5107.jpg)

![366aa5f99620808490850e8410578940064b8b4346b1a3d4ea17e09721d79f9a.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/366aa5f99620808490850e8410578940064b8b4346b1a3d4ea17e09721d79f9a.jpg)

![57ef279fc223416367785a514bfe51cc2572853fcc3e013fdc8a5cec39c4de09.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/57ef279fc223416367785a514bfe51cc2572853fcc3e013fdc8a5cec39c4de09.jpg)

![71a6d480baa2cad3e5fc783792c9bd4e2abbfe360aa82d0cfc04393d3f5d3766.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/71a6d480baa2cad3e5fc783792c9bd4e2abbfe360aa82d0cfc04393d3f5d3766.jpg)

![8991c4394f0ccfac271f15be8584d9219839bf66dd6513abba6133143108acdf.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/8991c4394f0ccfac271f15be8584d9219839bf66dd6513abba6133143108acdf.jpg)

![8a680f6ff33631bdee0e7b7be1592557c2401400097a07b8f5542391d7cee04b.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/8a680f6ff33631bdee0e7b7be1592557c2401400097a07b8f5542391d7cee04b.jpg)

![8fd6f4f31f270dca80b61211c15cf0ddd01279d0c926012f6bc731398b4703de.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/8fd6f4f31f270dca80b61211c15cf0ddd01279d0c926012f6bc731398b4703de.jpg)

![96a63f9aba8d5bcb5c616d6ea2e0061d74e8239032fc44f98dbb4a3b8cf0c322.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/96a63f9aba8d5bcb5c616d6ea2e0061d74e8239032fc44f98dbb4a3b8cf0c322.jpg)

![99c3bc2bd3acd244614a31345ba437c3685c737e4daed2024fdeb5aa899bd6b7.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/99c3bc2bd3acd244614a31345ba437c3685c737e4daed2024fdeb5aa899bd6b7.jpg)

![9a2a9e27b3d25693925fa09b735b960aac307a4d2c2437319cb28da117e2eae7.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/9a2a9e27b3d25693925fa09b735b960aac307a4d2c2437319cb28da117e2eae7.jpg)

![9f98148008ad322d6c7d9c08f2ba3d77a0d5c026ae0582f3c184eb1ceb8e44db.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/9f98148008ad322d6c7d9c08f2ba3d77a0d5c026ae0582f3c184eb1ceb8e44db.jpg)

![b598495adc33252ebcdbf318aefada30a02622a1b5834e5bcb223a3d6e89e0a4.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/b598495adc33252ebcdbf318aefada30a02622a1b5834e5bcb223a3d6e89e0a4.jpg)

![c01ad47203809bc2f465cd270384bed9f46ce32d469b20d01f6a8f6bf0b97f0d.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/c01ad47203809bc2f465cd270384bed9f46ce32d469b20d01f6a8f6bf0b97f0d.jpg)

![e3d4003c0759819548bdfed7a61e7bee39afd93ec7b7b9e1c73037fd43c03e6c.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/e3d4003c0759819548bdfed7a61e7bee39afd93ec7b7b9e1c73037fd43c03e6c.jpg)

![e5a971217c4fe137670b5357aa327b81ddd91645f93e22888ba083c7c0b3f8bc.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/e5a971217c4fe137670b5357aa327b81ddd91645f93e22888ba083c7c0b3f8bc.jpg)

![ee52e5346ae2f48d065b2b0be3696e3173afb9e335d99549b56a820e3af3fbcf.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/ee52e5346ae2f48d065b2b0be3696e3173afb9e335d99549b56a820e3af3fbcf.jpg)

![fb9193838fd8c6945d4f5fc724c8fe7daf46f58e4d12e2ce07bdb7dccd39f3d0.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/fb9193838fd8c6945d4f5fc724c8fe7daf46f58e4d12e2ce07bdb7dccd39f3d0.jpg)

![fd9e901e31b5fac5f6f39421d4f42d6d8d8f01683d469ee6f9e80b6e81486eb5.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/fd9e901e31b5fac5f6f39421d4f42d6d8d8f01683d469ee6f9e80b6e81486eb5.jpg)

![fe0817c1bf32654e952497d052be679c9b42f1cd8b8326d549ea19c90b1a43f7.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/fe0817c1bf32654e952497d052be679c9b42f1cd8b8326d549ea19c90b1a43f7.jpg)

![ff1bbc309cb3e73200d70a12b42a580bc3adc469f7fc434fc2fb67c495b574d9.jpg](../icml_results/3630_Learning%20Dynamic%20Stability%20Landscapes%20in%20Synchronization%20Networks/tables/ff1bbc309cb3e73200d70a12b42a580bc3adc469f7fc434fc2fb67c495b574d9.jpg)

<a id="paper-3591"></a>

## Adaptive Node Feature Selection for Graph Neural Networks


### Images

![06ddfe8c3551448b5d435ffe9b7780464428b72a9fbbcf466929f397fabea5b3.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/06ddfe8c3551448b5d435ffe9b7780464428b72a9fbbcf466929f397fabea5b3.jpg)

![5ff0ade9621f6eea860422442004a77f3b6263105634edf87409b43fd3f58988.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/5ff0ade9621f6eea860422442004a77f3b6263105634edf87409b43fd3f58988.jpg)

![83c4f07361a2ade2135c9d7631898f008056163b03deac2c9acf09c8a0dc23f0.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/83c4f07361a2ade2135c9d7631898f008056163b03deac2c9acf09c8a0dc23f0.jpg)

![8553762c1c6c769f47cdb9760af959e5a4b674792feed17f2f36f78451e9fc4f.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/8553762c1c6c769f47cdb9760af959e5a4b674792feed17f2f36f78451e9fc4f.jpg)

![8a354b1e2d306562e82cd19d9fbe9e14c9569e519c3e42a4d82312718eafe1e9.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/8a354b1e2d306562e82cd19d9fbe9e14c9569e519c3e42a4d82312718eafe1e9.jpg)

![8b0be13c275d168adc410a8cea3644113e89a5d6c03a83db86690f7ea50e417a.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/8b0be13c275d168adc410a8cea3644113e89a5d6c03a83db86690f7ea50e417a.jpg)

![97c23309544802715acfc6e6eedcea49d076bca62afecb75b93da99344321b08.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/97c23309544802715acfc6e6eedcea49d076bca62afecb75b93da99344321b08.jpg)

![b310cc03a7056bc34b6a45ca05c50f641bc35367d69121c7f8f08eb78781c738.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/b310cc03a7056bc34b6a45ca05c50f641bc35367d69121c7f8f08eb78781c738.jpg)

![c7d41aec47f1b05f4f779648cbedcc7acd263fa75e7722a39ece8fd3612dd82d.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/c7d41aec47f1b05f4f779648cbedcc7acd263fa75e7722a39ece8fd3612dd82d.jpg)

![d3b9e30c5cf05a1c9b9bd967e2b28d4245efa814b137c873d108d3ee382d2a2c.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/d3b9e30c5cf05a1c9b9bd967e2b28d4245efa814b137c873d108d3ee382d2a2c.jpg)

![da029c92c3c962e63da44ec1803bf81b9af1a9bd97a6a65b2cd70a27cb6f5064.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/da029c92c3c962e63da44ec1803bf81b9af1a9bd97a6a65b2cd70a27cb6f5064.jpg)

![e9a4f599ffde8e913ada1118a8530658826da56021bf3abdd0aa08bdee5c93d9.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/e9a4f599ffde8e913ada1118a8530658826da56021bf3abdd0aa08bdee5c93d9.jpg)

![f2b5b2706823fb87e19404a8697dc8b30d2aefb72b00fb35653687a9440da02e.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/images/f2b5b2706823fb87e19404a8697dc8b30d2aefb72b00fb35653687a9440da02e.jpg)

### Tables

![0f65321c3b1763851b390494ad0481d0f89863e21abd8c6f65488f61c97a7b9d.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/tables/0f65321c3b1763851b390494ad0481d0f89863e21abd8c6f65488f61c97a7b9d.jpg)

![64bebe85e2c40c93c0a7db9a22638ee7347504eff0b0b41f8aa7c70dd9bc0de7.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/tables/64bebe85e2c40c93c0a7db9a22638ee7347504eff0b0b41f8aa7c70dd9bc0de7.jpg)

![8fdfd23cadea885834b8e9fb3ab11d1a3c0c5f6ae9df06292fba6722c98b850a.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/tables/8fdfd23cadea885834b8e9fb3ab11d1a3c0c5f6ae9df06292fba6722c98b850a.jpg)

![b278008ce9c6758b4f79bdbb47d6019309e120501d9e60000c45685395fda185.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/tables/b278008ce9c6758b4f79bdbb47d6019309e120501d9e60000c45685395fda185.jpg)

![cf917110502f75077a282d2f64636dee6f3d927a9cf4b2b21db3b4b6525b252a.jpg](../icml_results/3631_Adaptive%20Node%20Feature%20Selection%20for%20Graph%20Neural%20Networks/tables/cf917110502f75077a282d2f64636dee6f3d927a9cf4b2b21db3b4b6525b252a.jpg)

<a id="paper-3592"></a>

## A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn’t)


### Images

![073125cf9e72e994ab4ed0807657832e678f9996f9b4186188b777a7feb4daa6.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/073125cf9e72e994ab4ed0807657832e678f9996f9b4186188b777a7feb4daa6.jpg)

![17a1d15946461b7d1b7367b534278a17c8a3cc11020daaa429856aca5df673be.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/17a1d15946461b7d1b7367b534278a17c8a3cc11020daaa429856aca5df673be.jpg)

![2441cbdd2ea4c2dd58b5bdf22d2985553406693bb9f54b11832341ff707b184b.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/2441cbdd2ea4c2dd58b5bdf22d2985553406693bb9f54b11832341ff707b184b.jpg)

![2905867b6c255440d2f429e546c082dc9460a3710a5c5fd78a3437d7d8beef23.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/2905867b6c255440d2f429e546c082dc9460a3710a5c5fd78a3437d7d8beef23.jpg)

![2fa44b96f49452a65e82dcc6a26a31066e65f775b8533997a745ca824e7ea4b7.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/2fa44b96f49452a65e82dcc6a26a31066e65f775b8533997a745ca824e7ea4b7.jpg)

![30f6b739dcf7676c2c226b9c2e0aec23e11a9fd025d5ee9ee052a76245a72540.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/30f6b739dcf7676c2c226b9c2e0aec23e11a9fd025d5ee9ee052a76245a72540.jpg)

![36398d775e0ec9abc2a5102a6667c015662765ff78b4ae7c5a7f0ce6480684ed.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/36398d775e0ec9abc2a5102a6667c015662765ff78b4ae7c5a7f0ce6480684ed.jpg)

![44e3bfe9d0e380d159e69a86ca1113fc8ecb1a28bbbe81d26d88f43a12c3c1a4.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/44e3bfe9d0e380d159e69a86ca1113fc8ecb1a28bbbe81d26d88f43a12c3c1a4.jpg)

![483bbce7bf05324222a31fb3e2516a0c9d81919a02f7b75829821a949b64824c.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/483bbce7bf05324222a31fb3e2516a0c9d81919a02f7b75829821a949b64824c.jpg)

![50114865cadd38c08c893975d74ba4faf29ecb4df2c2a0db6364501a312bb96c.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/50114865cadd38c08c893975d74ba4faf29ecb4df2c2a0db6364501a312bb96c.jpg)

![50e01c3ebe3cd5718c9b6c1e3f3e1e6ec0b690c401bb5866496e82a972c4db45.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/50e01c3ebe3cd5718c9b6c1e3f3e1e6ec0b690c401bb5866496e82a972c4db45.jpg)

![67b63418af315d36b07282ce25d00e00adec25e3c55eba1de940025e1ecb1f7c.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/67b63418af315d36b07282ce25d00e00adec25e3c55eba1de940025e1ecb1f7c.jpg)

![7186a6d7349e53f15620ccb21f0a19f683ef5e3f171d85b918c1e5383a5a4c60.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/7186a6d7349e53f15620ccb21f0a19f683ef5e3f171d85b918c1e5383a5a4c60.jpg)

![754e466e956032bb3d592fd3fcd3f16ad50393d8c08bd16f5b983a82cf8a9ede.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/754e466e956032bb3d592fd3fcd3f16ad50393d8c08bd16f5b983a82cf8a9ede.jpg)

![7da1381cc90fc816ebe40dc96b7e9fde366f1c6d7d3dd66aaea85aa620ff29b9.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/7da1381cc90fc816ebe40dc96b7e9fde366f1c6d7d3dd66aaea85aa620ff29b9.jpg)

![a37039612ec342a653dc9b1c1f98e10b75086e9b798be573a2d24462e18fc4c1.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/a37039612ec342a653dc9b1c1f98e10b75086e9b798be573a2d24462e18fc4c1.jpg)

![a42d69d2ccce274fb0ffdecbf696cd9a103beb6e7e6c29935474014e028f036d.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/a42d69d2ccce274fb0ffdecbf696cd9a103beb6e7e6c29935474014e028f036d.jpg)

![b40395f5c71eb392ab6e3f91e7950c2c35ae432e35b666dfdb30fc95022b6b76.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/b40395f5c71eb392ab6e3f91e7950c2c35ae432e35b666dfdb30fc95022b6b76.jpg)

![b8530d3cb13a02e69cf97ee652a96dc30c531dd78453b484e95d38b85d99d8c6.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/b8530d3cb13a02e69cf97ee652a96dc30c531dd78453b484e95d38b85d99d8c6.jpg)

![bf4725a94a7112797cc6015821c60c5caa2fe789cc0db4c8a910eca742ab1c22.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/bf4725a94a7112797cc6015821c60c5caa2fe789cc0db4c8a910eca742ab1c22.jpg)

![cdefa5840a84b163e82bc1e6b52818be401fe5b5444d36ac83ad4c6e1b9b8efe.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/cdefa5840a84b163e82bc1e6b52818be401fe5b5444d36ac83ad4c6e1b9b8efe.jpg)

![ce99f2aaf336e135f97a259edecc0d2e4b7c1130c3c52ee6e067bd00e3a8cf90.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/ce99f2aaf336e135f97a259edecc0d2e4b7c1130c3c52ee6e067bd00e3a8cf90.jpg)

![d652f129a2debf50fdcf9b203997f7e96e809ab09c941607002d516971580efe.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/d652f129a2debf50fdcf9b203997f7e96e809ab09c941607002d516971580efe.jpg)

![e326d12f75b16bcafe7af4a10fafe0d0efdc6fb3837ec5ccea5c28cb3f19e0d2.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/images/e326d12f75b16bcafe7af4a10fafe0d0efdc6fb3837ec5ccea5c28cb3f19e0d2.jpg)

### Tables

![2e363912ea37b89b567d9aa4c57ef80b15ce5f53ec57727edc72cbf8f45c014d.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/tables/2e363912ea37b89b567d9aa4c57ef80b15ce5f53ec57727edc72cbf8f45c014d.jpg)

![3a4821b9a099e53b44565027797909fe16d4358a72b19fc3f3cfb655adcc083e.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/tables/3a4821b9a099e53b44565027797909fe16d4358a72b19fc3f3cfb655adcc083e.jpg)

![75d87b2df342bd2b09c1b1aea07ac1d1e086138c255acfb8e08300e164b4898f.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/tables/75d87b2df342bd2b09c1b1aea07ac1d1e086138c255acfb8e08300e164b4898f.jpg)

![9657fb6da9da9f504d9fe340df2b28c6e528fc1fd2edaa0c75254c9bce08f3b5.jpg](../icml_results/3632_A%20Critical%20Look%20at%20Targeted%20Instruction%20Selection_%20Disentangling%20What%20Matters%20%28and%20What%20Doesn%E2%80%99t%29/tables/9657fb6da9da9f504d9fe340df2b28c6e528fc1fd2edaa0c75254c9bce08f3b5.jpg)

<a id="paper-3593"></a>

## Achieving Logarithmic Regret in KL-Regularized Zero-Sum Markov Games


### Images

![4b9b0ea45e35a69e02e64a34491096b38d3f2e8eb0427d8693fde2d7464791da.jpg](../icml_results/3633_Achieving%20Logarithmic%20Regret%20in%20KL-Regularized%20Zero-Sum%20Markov%20Games/images/4b9b0ea45e35a69e02e64a34491096b38d3f2e8eb0427d8693fde2d7464791da.jpg)

### Tables

![de34f48fae47f294679a9d6349766712c5e8c6c225b3faa4c104b2eae43db4d6.jpg](../icml_results/3633_Achieving%20Logarithmic%20Regret%20in%20KL-Regularized%20Zero-Sum%20Markov%20Games/tables/de34f48fae47f294679a9d6349766712c5e8c6c225b3faa4c104b2eae43db4d6.jpg)

<a id="paper-3594"></a>

## CADFit: Precise Mesh-to-CAD Program Generation with Hybrid Optimization


### Images

![04cc75524f2d72594aabb720e354da7b5e45126d04bf953d16c46085c11f6c86.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/04cc75524f2d72594aabb720e354da7b5e45126d04bf953d16c46085c11f6c86.jpg)

![0fe2d6060828d53ff21c9985da067d00b8a7d1e3fcdcdaaead66410b731eb07e.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/0fe2d6060828d53ff21c9985da067d00b8a7d1e3fcdcdaaead66410b731eb07e.jpg)

![4076499d1eb24022447c3bdaee57196e2cdd929fce0e0ec59fcfa3b1f32f4c83.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/4076499d1eb24022447c3bdaee57196e2cdd929fce0e0ec59fcfa3b1f32f4c83.jpg)

![6088b85dce0c21b004ec4515e6f39d39a94060c42fcedb08c790de3081b819f8.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/6088b85dce0c21b004ec4515e6f39d39a94060c42fcedb08c790de3081b819f8.jpg)

![775194fa58ab1d1cc9ebafdde580b9356317a5f89610768fa834546bcbfa8c97.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/775194fa58ab1d1cc9ebafdde580b9356317a5f89610768fa834546bcbfa8c97.jpg)

![8571be6bd55f03138925d8fb4a310d94681624128b4d1668f7aff7f35c978fbf.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/8571be6bd55f03138925d8fb4a310d94681624128b4d1668f7aff7f35c978fbf.jpg)

![99ee4179af7caf91c7d10ce11ca18b593ca785439957d5a445f134555ee95290.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/99ee4179af7caf91c7d10ce11ca18b593ca785439957d5a445f134555ee95290.jpg)

![a093dd2887e403c300846fc7e8a1599b17b0ee899e78135d61f4dccac6d23f18.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/a093dd2887e403c300846fc7e8a1599b17b0ee899e78135d61f4dccac6d23f18.jpg)

![ae993b5269b7248ada484d88e6bf1c5217182405ce08a2efb6e16e7f45098629.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/ae993b5269b7248ada484d88e6bf1c5217182405ce08a2efb6e16e7f45098629.jpg)

![cc41ab696bf78fd7441a7491cece51e5b1935b61d75468e1324eaddf0f43c922.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/images/cc41ab696bf78fd7441a7491cece51e5b1935b61d75468e1324eaddf0f43c922.jpg)

### Tables

![074ed3a2a045f0b861ed2b35f85c6db160ee16e6695eee7b63354d780ff3b852.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/074ed3a2a045f0b861ed2b35f85c6db160ee16e6695eee7b63354d780ff3b852.jpg)

![204cdfe10d2f52f58729c9110c7fb0238ecc454bd6d72bb55a4cb2c97d8a131f.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/204cdfe10d2f52f58729c9110c7fb0238ecc454bd6d72bb55a4cb2c97d8a131f.jpg)

![54994fea7efcf7866eb66fad07d9b7ab60a14096ca0b2a9a4968a2b5fea67acd.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/54994fea7efcf7866eb66fad07d9b7ab60a14096ca0b2a9a4968a2b5fea67acd.jpg)

![671d1fb4d171f4d77f0559b107b637f6cca45b812ff6eb423110070939d486d4.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/671d1fb4d171f4d77f0559b107b637f6cca45b812ff6eb423110070939d486d4.jpg)

![726bb439b8d9b3749a4aca48b1ccb717433b6b5cfef6e20d706c8a686ddb0941.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/726bb439b8d9b3749a4aca48b1ccb717433b6b5cfef6e20d706c8a686ddb0941.jpg)

![7ee7c74747eba6880c545175ee6b4e17a37d465c8d4c9076af9772266583816a.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/7ee7c74747eba6880c545175ee6b4e17a37d465c8d4c9076af9772266583816a.jpg)

![94f549858215cb2f17d1a8dc141881c2f9da467af311bc4b6e114077f0120eed.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/94f549858215cb2f17d1a8dc141881c2f9da467af311bc4b6e114077f0120eed.jpg)

![a4a6c0d6c60e43566edcbe97584c434817a8f92d3fc9c1bd51413909a1d526e3.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/a4a6c0d6c60e43566edcbe97584c434817a8f92d3fc9c1bd51413909a1d526e3.jpg)

![b7022f646e7af441e54e2ae228affcbebda84c40e8b3810d711babe36633f563.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/b7022f646e7af441e54e2ae228affcbebda84c40e8b3810d711babe36633f563.jpg)

![c13abbe555546ffad6a38808f7c929f6dbc6d0aec0004d876e284ccd7b9cb69b.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/c13abbe555546ffad6a38808f7c929f6dbc6d0aec0004d876e284ccd7b9cb69b.jpg)

![c4b164eb33ab8ee89130bef15500bbca4daf674f978d611b12eae977fb4763ef.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/c4b164eb33ab8ee89130bef15500bbca4daf674f978d611b12eae977fb4763ef.jpg)

![d2571a1a223866c6190c68844c4bf5703e8778fe7b995f8f3de8ded3cec8014f.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/d2571a1a223866c6190c68844c4bf5703e8778fe7b995f8f3de8ded3cec8014f.jpg)

![d6e03229f6715ede24d58015670ce73a3e8aa1c28e5778a670d5a07b6c35394e.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/d6e03229f6715ede24d58015670ce73a3e8aa1c28e5778a670d5a07b6c35394e.jpg)

![e3b5afcf4be7928c0c7ce3ce4e3bf2fb6e5d10ec5fc8a0034770bdf5717bd11a.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/e3b5afcf4be7928c0c7ce3ce4e3bf2fb6e5d10ec5fc8a0034770bdf5717bd11a.jpg)

![ee046889d6b0ad7240b44771c5f5be5bdb4bcf6e2e1096caf1821f0e09b3902e.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/ee046889d6b0ad7240b44771c5f5be5bdb4bcf6e2e1096caf1821f0e09b3902e.jpg)

![ef5095e546a94163d16d0e2e6a716e2d1a03b5d2c6125b17d75dc2ba8e21a14f.jpg](../icml_results/3634_CADFit_%20Precise%20Mesh-to-CAD%20Program%20Generation%20with%20Hybrid%20Optimization/tables/ef5095e546a94163d16d0e2e6a716e2d1a03b5d2c6125b17d75dc2ba8e21a14f.jpg)

<a id="paper-3595"></a>

## LAMP: Data-Efficient Linear Affine Weight-Space Models for Parameter-Controlled 3D Shape Generation and Extrapolation


### Images

![03da17b4cb6da67b03d7649fb497f32f637a72c55c0f0f0319a79614200e79d6.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/03da17b4cb6da67b03d7649fb497f32f637a72c55c0f0f0319a79614200e79d6.jpg)

![085ffe6a93242ac762b1fe3680b5ca0fa585373e4cdcc1e573b0c7787f6fa591.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/085ffe6a93242ac762b1fe3680b5ca0fa585373e4cdcc1e573b0c7787f6fa591.jpg)

![0cdc0d81f699c716f25d41a2010bb490ae284f6ed1517c98421c3ce3e5b0a217.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/0cdc0d81f699c716f25d41a2010bb490ae284f6ed1517c98421c3ce3e5b0a217.jpg)

![0e648a34ac66863043aef4c740cc8fa0449fe851ed324ce3adf1b301e4bff968.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/0e648a34ac66863043aef4c740cc8fa0449fe851ed324ce3adf1b301e4bff968.jpg)

![126f445ef58a26a0783f3f3a2702da3279753e5cde6a5f72d2766991940d6759.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/126f445ef58a26a0783f3f3a2702da3279753e5cde6a5f72d2766991940d6759.jpg)

![26d77aa6f410220ed4d71ab57d9c41fe28a81261fb04ec1a7e43ee56a577f6e3.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/26d77aa6f410220ed4d71ab57d9c41fe28a81261fb04ec1a7e43ee56a577f6e3.jpg)

![2891412bd188fadb3da439f9fe30225549c257d67483ecea1ae7c2731c17f8e0.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/2891412bd188fadb3da439f9fe30225549c257d67483ecea1ae7c2731c17f8e0.jpg)

![29b1d3aac8d22d39754112a1950f11a20b1590c7b6eed8a684eac3f0cc91b2fa.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/29b1d3aac8d22d39754112a1950f11a20b1590c7b6eed8a684eac3f0cc91b2fa.jpg)

![393ca9c460deedd45b58725998710dbb89375ed2d027fbf80cbd945f36e02f73.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/393ca9c460deedd45b58725998710dbb89375ed2d027fbf80cbd945f36e02f73.jpg)

![42c73610621eb02ab659dfac18500468093363c5b80e93b33c51a44442485b99.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/42c73610621eb02ab659dfac18500468093363c5b80e93b33c51a44442485b99.jpg)

![42e24415a5e691330d0d0cef1580358abfa26a820cdb618f90e7b292ec95948e.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/42e24415a5e691330d0d0cef1580358abfa26a820cdb618f90e7b292ec95948e.jpg)

![568063528a8bb3a24cd974d33f33f5f813eba3209feb5c8bb635ae7cacef508d.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/568063528a8bb3a24cd974d33f33f5f813eba3209feb5c8bb635ae7cacef508d.jpg)

![705bd52733b1a4df41cd2ed663ac0cc82720d8be9f73c28cb9dca37ad6b53b2a.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/705bd52733b1a4df41cd2ed663ac0cc82720d8be9f73c28cb9dca37ad6b53b2a.jpg)

![768c4eaeda58531ee2bf4184d258ff0ec35895272f91dbcfdb946d47671aff27.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/768c4eaeda58531ee2bf4184d258ff0ec35895272f91dbcfdb946d47671aff27.jpg)

![7cabc0dd32dd40176f41abdf42a185ab8d291966b5b38c530c3d9097c95aa0aa.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/7cabc0dd32dd40176f41abdf42a185ab8d291966b5b38c530c3d9097c95aa0aa.jpg)

![862bb3b73959531c373ef3d2c70f6d03d6317a71dd91030d871f308f8552f392.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/862bb3b73959531c373ef3d2c70f6d03d6317a71dd91030d871f308f8552f392.jpg)

![87b882f494f4cd523c0218350a48bf0b81346d8e1a699ee7e09b614819a26cd1.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/87b882f494f4cd523c0218350a48bf0b81346d8e1a699ee7e09b614819a26cd1.jpg)

![9f810d00c51cbb27c5c483b0100ecfc5893279d2f48919eb783bd26c365f4cb9.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/9f810d00c51cbb27c5c483b0100ecfc5893279d2f48919eb783bd26c365f4cb9.jpg)

![a265e8bdf937fe93219e9b5b2d95edf08ab23933e7b0111fa94e2a13408284d4.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/a265e8bdf937fe93219e9b5b2d95edf08ab23933e7b0111fa94e2a13408284d4.jpg)

![b4e72d7e1520c5e2b7dcefe23c5271918086c8dde49ad785c31da937f8f6f3db.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/b4e72d7e1520c5e2b7dcefe23c5271918086c8dde49ad785c31da937f8f6f3db.jpg)

![b560b2b7071bc0efa518026eb4fb434ead8f4362a73a7d81376ea43dbe50e204.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/b560b2b7071bc0efa518026eb4fb434ead8f4362a73a7d81376ea43dbe50e204.jpg)

![e0e9334f0d093e818f6d1b2e902db1db7629271917e301ac749183b639b57978.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/e0e9334f0d093e818f6d1b2e902db1db7629271917e301ac749183b639b57978.jpg)

![f182799051743ca2a18a240daee4901e937a3a9d588037ab7691838e02f34325.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/f182799051743ca2a18a240daee4901e937a3a9d588037ab7691838e02f34325.jpg)

![f423d31a0edd1a326506b524a11080079416cc01a662035cd341f83a2de2e46d.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/f423d31a0edd1a326506b524a11080079416cc01a662035cd341f83a2de2e46d.jpg)

![f69c7aab5a37c5e89add839829e1ab87331e3bff90383338c82d10bbfe0728d6.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/f69c7aab5a37c5e89add839829e1ab87331e3bff90383338c82d10bbfe0728d6.jpg)

![f7422375319656c7046bff9cdc8d9021a85ef05c06edf1de4da53da73a18030d.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/f7422375319656c7046bff9cdc8d9021a85ef05c06edf1de4da53da73a18030d.jpg)

![fa7c1bcf9c8fdae0ce963f0ffee70a7f1b899feb2f089aa4452ce81d2a76190e.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/images/fa7c1bcf9c8fdae0ce963f0ffee70a7f1b899feb2f089aa4452ce81d2a76190e.jpg)

### Tables

![093131d084fc2c6d567a6c3296ad1889dfc3c36df0aa5c6522d7704ee479d3e4.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/093131d084fc2c6d567a6c3296ad1889dfc3c36df0aa5c6522d7704ee479d3e4.jpg)

![243588faf6551923be7f7e497d2fb0d628ff3f4834d3b30ab051cde9af0593c0.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/243588faf6551923be7f7e497d2fb0d628ff3f4834d3b30ab051cde9af0593c0.jpg)

![31b578eb27474a8c35113033fda9e41e51101d9616d9e7ce6e9d34d2e25f04d3.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/31b578eb27474a8c35113033fda9e41e51101d9616d9e7ce6e9d34d2e25f04d3.jpg)

![4625b99555db109bc1e67d23903c641baf36e30bcd705b760f1372dc3fe73309.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/4625b99555db109bc1e67d23903c641baf36e30bcd705b760f1372dc3fe73309.jpg)

![5143940da1d1f500493dbb61550cddb1c3d3574514f2a58df0a47a008ff6531b.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/5143940da1d1f500493dbb61550cddb1c3d3574514f2a58df0a47a008ff6531b.jpg)

![62304f93be9791ecb1e9fc93929e7472b9ecf83784a387fb28e2176c436e00ba.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/62304f93be9791ecb1e9fc93929e7472b9ecf83784a387fb28e2176c436e00ba.jpg)

![7a0e570125dbbf73f21d7af60d7bbdd378e81875176b65aa49ad26e3d5252f08.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/7a0e570125dbbf73f21d7af60d7bbdd378e81875176b65aa49ad26e3d5252f08.jpg)

![7fc1fe1ffdc7bfc503222f5345277027836ee1f00fdaa0c94066f38c536dcd0d.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/7fc1fe1ffdc7bfc503222f5345277027836ee1f00fdaa0c94066f38c536dcd0d.jpg)

![c009b2a5ba69349a41f58c5cb3232d775878e0d022c642378b6b7af8b1d695b4.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/c009b2a5ba69349a41f58c5cb3232d775878e0d022c642378b6b7af8b1d695b4.jpg)

![cf28a26fb17d530284cc30b14bd5230b1ca62ac975ebc738425f5c1312b48de3.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/cf28a26fb17d530284cc30b14bd5230b1ca62ac975ebc738425f5c1312b48de3.jpg)

![d63dbe4517a560700a49d0f560ab1fdda1d8bed49ac4610283c0f72416a947ab.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/d63dbe4517a560700a49d0f560ab1fdda1d8bed49ac4610283c0f72416a947ab.jpg)

![d7a79a5b782c17b4b85dec1340a1fd2ed5122bc2c9019870fa6f8a8dfb9875cc.jpg](../icml_results/3635_LAMP_%20Data-Efficient%20Linear%20Affine%20Weight-Space%20Models%20for%20Parameter-Controlled%203D%20Shape%20Generation%20/tables/d7a79a5b782c17b4b85dec1340a1fd2ed5122bc2c9019870fa6f8a8dfb9875cc.jpg)

<a id="paper-3596"></a>

## Universal Representation of Generalized Convex Functions and their Gradients


### Images

![3aed813af2b60e4d72c8667ea6cd355a71da80e9d6cf1b087efab6bd6d02343c.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/images/3aed813af2b60e4d72c8667ea6cd355a71da80e9d6cf1b087efab6bd6d02343c.jpg)

![8a32a5d4255db1bdb61789ddef34fc60bf93de5810955a527e6cacf654b1da1c.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/images/8a32a5d4255db1bdb61789ddef34fc60bf93de5810955a527e6cacf654b1da1c.jpg)

![994485238ee10778e2128f40fb2ab585ddc4eed79df1124673dc589a4626b2be.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/images/994485238ee10778e2128f40fb2ab585ddc4eed79df1124673dc589a4626b2be.jpg)

![b2e7da829610a4a292d534c4399cff6bbf035ea9a8b969b016d6596823b12e07.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/images/b2e7da829610a4a292d534c4399cff6bbf035ea9a8b969b016d6596823b12e07.jpg)

### Tables

![356824935283f3612c08cc44cb0eca50ac50d01c2be5da35f8b17b243ca338c7.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/tables/356824935283f3612c08cc44cb0eca50ac50d01c2be5da35f8b17b243ca338c7.jpg)

![85bda908f5d873dea507e7880e1a61c1b3eaab420b6d352f778f932fe392b574.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/tables/85bda908f5d873dea507e7880e1a61c1b3eaab420b6d352f778f932fe392b574.jpg)

![db32671844ca86f103ce9535f0667a48c133ee0294595b8d8617f35e02e797b8.jpg](../icml_results/3636_Universal%20Representation%20of%20Generalized%20Convex%20Functions%20and%20their%20Gradients/tables/db32671844ca86f103ce9535f0667a48c133ee0294595b8d8617f35e02e797b8.jpg)

<a id="paper-3597"></a>

## Toward Identifiable Sparse Autoencoders

### Images

![a9146363e63a10ede62359d28c3a6e90294b351d08a95972dcb77ccfbc219536.jpg](../icml_results/3637_Toward%20Identifiable%20Sparse%20Autoencoders/images/a9146363e63a10ede62359d28c3a6e90294b351d08a95972dcb77ccfbc219536.jpg)

### Tables

![2d7545a98c61fbbb1a5fee1ddf651cefb1d506326a4e4aba8ede9ddc384bd112.jpg](../icml_results/3637_Toward%20Identifiable%20Sparse%20Autoencoders/tables/2d7545a98c61fbbb1a5fee1ddf651cefb1d506326a4e4aba8ede9ddc384bd112.jpg)

![c83456691b862b097f7730bc66297996c8843c908deb91d89f5e40d76ce2a05c.jpg](../icml_results/3637_Toward%20Identifiable%20Sparse%20Autoencoders/tables/c83456691b862b097f7730bc66297996c8843c908deb91d89f5e40d76ce2a05c.jpg)

![e9dc09000dd6fe8a2e3cc21644d211a3ce62ab2bee3b96265f2428d24e38fe70.jpg](../icml_results/3637_Toward%20Identifiable%20Sparse%20Autoencoders/tables/e9dc09000dd6fe8a2e3cc21644d211a3ce62ab2bee3b96265f2428d24e38fe70.jpg)
