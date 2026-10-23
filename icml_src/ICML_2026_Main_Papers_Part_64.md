# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 64 of 197

## 目录 (Table of Contents)

1. [On the Emergence of Implicit Curriculum in RLVR Learning Dynamics](#paper-2080)
2. [A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention](#paper-2081)
3. [AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in Unified Multimodal Models via Decompositional Verifiable Reward](#paper-2082)
4. [Does Your Reasoning Model Implicitly Know When to Stop Thinking?](#paper-2083)
5. [Real-Time Aligned Reward Model beyond Semantics](#paper-2084)
6. [Delegation and Verification under AI](#paper-2085)
7. [How to Fine-Tune a Reasoning Model? A Teacher–Student Cooperation Framework to Synthesize Student-Consistent SFT Data](#paper-2086)
8. [FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models](#paper-2087)
9. [Exploring Motif-based Heterogeneous Graph Learning for ReDoS Detection](#paper-2088)
10. [LIVE: Long-horizon Interactive Video World Modeling](#paper-2089)
11. [FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](#paper-2090)
12. [Cross-Embodiment Robot Foundation World Models with Latent Actions](#paper-2091)
13. [ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation](#paper-2092)
14. [Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models](#paper-2093)
15. [MAS-Architect: Declarative Multi-Agent System Design via Separation of Concerns](#paper-2094)
16. [Optimal Transport–Guided Stochastic Control for Graph Combinatorial Optimization](#paper-2095)
17. [Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs](#paper-2096)
18. [AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture -of-Transformers for End-to-End Autonomous Driving](#paper-2097)
19. [Prototype-Based Test-Time Adaptation of Vision-Language Models](#paper-2098)
20. [TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments](#paper-2099)
21. [Continuous Variable Hamiltonian Learning at Heisenberg Limit via Displacement-Random Unitary Transformation](#paper-2100)
22. [How Far Can LLM Agents Reason with Tables? Benchmarking Multi-Turn Agentic Table Question Answering in the Wild](#paper-2101)
23. [Deterministic Differentiable Structured Pruning for Large Language Models](#paper-2102)
24. [RLCracker: Evaluating the Worst-Case Vulnerability of LLM Watermarks with Adaptive RL Attacks](#paper-2103)
25. [LVCG: Learning ECG Representations in the Latent Vectorcardiogram Space](#paper-2104)
26. [GeoFlow: Geo-Aware Modeling of Inter-Area Relationships in Origin-Destination Flow Prediction and Generation](#paper-2105)
27. [IACW: Intent-Aware Controllable Watermarking for Scalable Authorial Intent Attribution](#paper-2106)
28. [ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation](#paper-2107)
29. [TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches](#paper-2108)
30. [HONet: Data-Efficient Learning for Exact Cover Tasks via Hypergraph Optimization](#paper-2109)
31. [Reliability-Aware LLM Alignment from Inconsistent Human Feedback](#paper-2110)
32. [Foundation Inference Models for Ordinary Differential Equations](#paper-2111)
33. [Reinforcement Learning via Self-Distillation](#paper-2112)

---


<a id="paper-2080"></a>

## On the Emergence of Implicit Curriculum in RLVR Learning Dynamics

### Images

![19aaebc8d46581d54131d432581eb5a7b7def19bca8a2f8cc206d868c6932370.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/19aaebc8d46581d54131d432581eb5a7b7def19bca8a2f8cc206d868c6932370.jpg)

![21ff0973351b84a482d1cce86d25280dd9534ce9d7e7742bc273fdcc1b41a28c.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/21ff0973351b84a482d1cce86d25280dd9534ce9d7e7742bc273fdcc1b41a28c.jpg)

![30cbdff50b76a9f0e5ba6c3246b117a7fbefbbcfa0a4c01d77239bbc48ce9bd1.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/30cbdff50b76a9f0e5ba6c3246b117a7fbefbbcfa0a4c01d77239bbc48ce9bd1.jpg)

![63b6545f41475f0b6314b2d910d687a748de638454e38d75ab1e464d11bdb7d1.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/63b6545f41475f0b6314b2d910d687a748de638454e38d75ab1e464d11bdb7d1.jpg)

![6617e6bc18feb9fee608f4d74a4ae4162a32ba735f8347c3ca777ff6b45db39e.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/6617e6bc18feb9fee608f4d74a4ae4162a32ba735f8347c3ca777ff6b45db39e.jpg)

![6d2300d1a47605118855c791d029d93c43281a3cbb340024b2f81de2cbd2ffc0.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/6d2300d1a47605118855c791d029d93c43281a3cbb340024b2f81de2cbd2ffc0.jpg)

![84d0295e66d9ebf8d9cff12ede72403958fbfe9d4863fcba3b11e6e54fe7b905.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/84d0295e66d9ebf8d9cff12ede72403958fbfe9d4863fcba3b11e6e54fe7b905.jpg)

![88a0ea78f35f6150d5e9e0bbfad402737aaa4312716d3a57eb85341425dc9cb9.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/88a0ea78f35f6150d5e9e0bbfad402737aaa4312716d3a57eb85341425dc9cb9.jpg)

![b4261aa5dfb7d9357add01adb04b7b668c397a804df50a50956db2239cb53da1.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/b4261aa5dfb7d9357add01adb04b7b668c397a804df50a50956db2239cb53da1.jpg)

![be4b4a0887a579c9793b061ebfe02f9bd713b33cd555c3fced7f62e6ca264763.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/be4b4a0887a579c9793b061ebfe02f9bd713b33cd555c3fced7f62e6ca264763.jpg)

![bf183fb6f3359b4ad6603c2adf771f0806d0e7331ffd43d29796b0cd150fdfc8.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/bf183fb6f3359b4ad6603c2adf771f0806d0e7331ffd43d29796b0cd150fdfc8.jpg)

![dc818c8e6e8c95d8fcb0440bb8494126507874e552db38bae2a3906eff682852.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/dc818c8e6e8c95d8fcb0440bb8494126507874e552db38bae2a3906eff682852.jpg)

![df9336a284eebb9e39bed78acffdb20ab5afc2dbbac45340fcde344b87b5c17d.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/images/df9336a284eebb9e39bed78acffdb20ab5afc2dbbac45340fcde344b87b5c17d.jpg)

### Tables

![0d390d7e709306e69328ac1d57b6e12bf27987186c45c11d56002125233bfa02.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/0d390d7e709306e69328ac1d57b6e12bf27987186c45c11d56002125233bfa02.jpg)

![2276d909357d1692627514e9a94d05d21fef7372ca2b28491c81bff80a585703.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/2276d909357d1692627514e9a94d05d21fef7372ca2b28491c81bff80a585703.jpg)

![254d34c2d279d42711285e29eb95da9ebf57fb7b2f223519e6ff4248bddc3550.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/254d34c2d279d42711285e29eb95da9ebf57fb7b2f223519e6ff4248bddc3550.jpg)

![495177f9f011a21cc64642975a9a71d4fbdbe328fc716c15ebd9362ab4187bd1.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/495177f9f011a21cc64642975a9a71d4fbdbe328fc716c15ebd9362ab4187bd1.jpg)

![497a9fe8b4a7660f23d92891ed3a6d066afe2ffb7337775f5e41b8c201f97cf3.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/497a9fe8b4a7660f23d92891ed3a6d066afe2ffb7337775f5e41b8c201f97cf3.jpg)

![4afeae091f73f23e0cd2b1824d38c9cae158cb6c7d653543e84b67d9dc531935.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/4afeae091f73f23e0cd2b1824d38c9cae158cb6c7d653543e84b67d9dc531935.jpg)

![52bea82d20591dad1c8e3911089fa1debe90e612f39259c694ee63b64582e9dd.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/52bea82d20591dad1c8e3911089fa1debe90e612f39259c694ee63b64582e9dd.jpg)

![612291174493e1d9907f08616ffa610efb91feef6a38443ebacc2fbd842088f3.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/612291174493e1d9907f08616ffa610efb91feef6a38443ebacc2fbd842088f3.jpg)

![6ba63f63b2df745ba1b40fd3b0d27c9c8a26f45f70051681aa5abbcc881ab89a.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/6ba63f63b2df745ba1b40fd3b0d27c9c8a26f45f70051681aa5abbcc881ab89a.jpg)

![90b726090994e181ff09845e6953777c1a3e87598ecc751cc13b3a2a88ffbe9f.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/90b726090994e181ff09845e6953777c1a3e87598ecc751cc13b3a2a88ffbe9f.jpg)

![a5f05752021b4e936032dc78570ac61c196802684a6bd0fdf70b381fa8d69a79.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/a5f05752021b4e936032dc78570ac61c196802684a6bd0fdf70b381fa8d69a79.jpg)

![ce5ff3ec84cbe21b7357d254276c24b9b38891a3b88e420d22dfce28988865ca.jpg](../icml_results/2107_Restoring%20Initial%20Noise%20Sensitivity%20in%20Text-to-Image%20Distillation%20via%20Geometric%20Alignment/tables/ce5ff3ec84cbe21b7357d254276c24b9b38891a3b88e420d22dfce28988865ca.jpg)

## On the Emergence of Implicit Curriculum in RLVR Learning Dynamics


### Images

![7c3e7940bfa2236791314adf47e6dadbe4f16dd8eae3dd6f1e71a2cde86f8e02.jpg](../icml_results/2108_On%20the%20Emergence%20of%20Implicit%20Curriculum%20in%20RLVR%20Learning%20Dynamics/images/7c3e7940bfa2236791314adf47e6dadbe4f16dd8eae3dd6f1e71a2cde86f8e02.jpg)

![bad2aa549c340875d7e8fc3a6983091d6ff109ebfb1d73b2d356e266dcf3c2b8.jpg](../icml_results/2108_On%20the%20Emergence%20of%20Implicit%20Curriculum%20in%20RLVR%20Learning%20Dynamics/images/bad2aa549c340875d7e8fc3a6983091d6ff109ebfb1d73b2d356e266dcf3c2b8.jpg)

<a id="paper-2081"></a>

## A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention


### Images

![166dab1baf353fe30169a0944fd6b783743e7ad8371b979f29d750602efbecfa.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/images/166dab1baf353fe30169a0944fd6b783743e7ad8371b979f29d750602efbecfa.jpg)

![787a79b147ea37a6b1f655aae073e0a4bd907fc56fb2cd0ac96a9e0f686fd1a6.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/images/787a79b147ea37a6b1f655aae073e0a4bd907fc56fb2cd0ac96a9e0f686fd1a6.jpg)

![86d70efa636877832241d85e5c31059f10c0b73978965c80a01465c128940038.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/images/86d70efa636877832241d85e5c31059f10c0b73978965c80a01465c128940038.jpg)

![a321b11a257a6fc9d7440a4ae0e91d95d0d65db581cd4777d0dc1717e43ce404.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/images/a321b11a257a6fc9d7440a4ae0e91d95d0d65db581cd4777d0dc1717e43ce404.jpg)

![eedf3f274f1e6d0d7d48f69a432976769e9054af5b5d2de8d22d47119e69f14a.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/images/eedf3f274f1e6d0d7d48f69a432976769e9054af5b5d2de8d22d47119e69f14a.jpg)

### Tables

![0057c5871f634338d5fc883db84d421e8e182725b5b3244bee3fa525c849b902.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/0057c5871f634338d5fc883db84d421e8e182725b5b3244bee3fa525c849b902.jpg)

![130387d0781cb9d458e279ac7222dc2cb097ed51426c4cf497dfb147b559fb1a.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/130387d0781cb9d458e279ac7222dc2cb097ed51426c4cf497dfb147b559fb1a.jpg)

![1757dee25ad63ab57c1ebbef1c6c7f023b4df648b07c1d3373cf1718cdc7b6e8.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/1757dee25ad63ab57c1ebbef1c6c7f023b4df648b07c1d3373cf1718cdc7b6e8.jpg)

![31e5b4601cac71ae177586f86c11dc25c147b762388b6c65b2605cab245df9bb.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/31e5b4601cac71ae177586f86c11dc25c147b762388b6c65b2605cab245df9bb.jpg)

![392fc03fb6b454d278bde0472dff7178b950a2914ac679074811bcaebb076bf9.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/392fc03fb6b454d278bde0472dff7178b950a2914ac679074811bcaebb076bf9.jpg)

![3b29426fab14f5eb097c75c8645e710fc593decb1124ac94ba702eaed84f6d14.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/3b29426fab14f5eb097c75c8645e710fc593decb1124ac94ba702eaed84f6d14.jpg)

![6e9644fa3c2dcfe5fce2123c618e5597038df2edff2ab81e1ae72914d470c4da.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/6e9644fa3c2dcfe5fce2123c618e5597038df2edff2ab81e1ae72914d470c4da.jpg)

![74ed5a06c798f44cfd8e9dea050742307af68ab7a51e986f633228cc340363fb.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/74ed5a06c798f44cfd8e9dea050742307af68ab7a51e986f633228cc340363fb.jpg)

![76781819414b183fbd065420a5bb15d9d5b31b074d7a4940af473416a7cb0dce.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/76781819414b183fbd065420a5bb15d9d5b31b074d7a4940af473416a7cb0dce.jpg)

![8f82e8d5a4df457f7f130ec9cde3c03322761b32cdc414e4aa53cf482b658d47.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/8f82e8d5a4df457f7f130ec9cde3c03322761b32cdc414e4aa53cf482b658d47.jpg)

![9408149e7458e956dbdb296ffcbf3d3cdd121c4b956cfe706aa6d4db1a7404d0.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/9408149e7458e956dbdb296ffcbf3d3cdd121c4b956cfe706aa6d4db1a7404d0.jpg)

![a559705a8ca05c9b5e3f8ac40c69f436841acb49df57e36b0efee7b1845b7541.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/a559705a8ca05c9b5e3f8ac40c69f436841acb49df57e36b0efee7b1845b7541.jpg)

![b023a83d8ed780630b6448426536e07346c9c1ce7e982e4a737401f821e8dacd.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/b023a83d8ed780630b6448426536e07346c9c1ce7e982e4a737401f821e8dacd.jpg)

![c061e9f9b35f7e3dae9a67b3b31ca075c8563ea6c9111e90dd550a52107204f1.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/c061e9f9b35f7e3dae9a67b3b31ca075c8563ea6c9111e90dd550a52107204f1.jpg)

![c4ddbfa3d413576f843fbeeab68a2d72cdd66f2b2741d3d5f64119f49a168493.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/c4ddbfa3d413576f843fbeeab68a2d72cdd66f2b2741d3d5f64119f49a168493.jpg)

![cf581ea6de1a0ee89be185ada6b89590d5536646f5c083ccdd500ba33b42dda6.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/cf581ea6de1a0ee89be185ada6b89590d5536646f5c083ccdd500ba33b42dda6.jpg)

![d26182da5c0f9e0c01f168d44b2bb0f5a1576ff4cfc0bc852e0dab6d99ef38b3.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/d26182da5c0f9e0c01f168d44b2bb0f5a1576ff4cfc0bc852e0dab6d99ef38b3.jpg)

![d8ffa8ba5b11b75ef10a9387ffff0fbbe9614b6073eb85c8840d716d03320301.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/d8ffa8ba5b11b75ef10a9387ffff0fbbe9614b6073eb85c8840d716d03320301.jpg)

![dbeaf54c541e00be41fd70398d5b63f8e0f0899fbc15d070226f0ef45c7fd459.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/dbeaf54c541e00be41fd70398d5b63f8e0f0899fbc15d070226f0ef45c7fd459.jpg)

![de2380a1c9aa6110b65c3e6156dc74f2386db2d1f8742caff5bcf634de30f695.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/de2380a1c9aa6110b65c3e6156dc74f2386db2d1f8742caff5bcf634de30f695.jpg)

![fb285cc629e23bb9cd4bb498546b16f5a7f2eb7014cf452d260b715affb9e1c5.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/fb285cc629e23bb9cd4bb498546b16f5a7f2eb7014cf452d260b715affb9e1c5.jpg)

![fc1c90278d8bcccc9ee871eb6bf30be98603f249e9584e5a64bdfc611fbaca1e.jpg](../icml_results/2109_A%20General%20Neural%20Backbone%20for%20Mixed-Integer%20Linear%20Optimization%20via%20Dual%20Attention/tables/fc1c90278d8bcccc9ee871eb6bf30be98603f249e9584e5a64bdfc611fbaca1e.jpg)

<a id="paper-2082"></a>

## AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in Unified Multimodal Models via Decompositional Verifiable Reward


### Images

![2082591a7630938b183c0277fb6250e75b7bfc9c6c5edb76eed491445641e7fa.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/2082591a7630938b183c0277fb6250e75b7bfc9c6c5edb76eed491445641e7fa.jpg)

![60dfd4148f359484377e0591d48f0d9544013a01c5e5e4cd5f2eb80cec1997ca.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/60dfd4148f359484377e0591d48f0d9544013a01c5e5e4cd5f2eb80cec1997ca.jpg)

![6a07b48522beec10a9e82dd9936ca3568d2599b89326cecfec395d7dbe1c7dbb.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/6a07b48522beec10a9e82dd9936ca3568d2599b89326cecfec395d7dbe1c7dbb.jpg)

![6a74208d3fbf1bf993fc20e57ba7ad1cb2d264d1e5d82c01368c99537c651ee7.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/6a74208d3fbf1bf993fc20e57ba7ad1cb2d264d1e5d82c01368c99537c651ee7.jpg)

![85c96fb8262d528864735e53034598b2fdb73ebbb1614ad4d5e2c8152d83af45.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/85c96fb8262d528864735e53034598b2fdb73ebbb1614ad4d5e2c8152d83af45.jpg)

![c60418e6536378a8fa86724a76d289717b659c3c2a8089658a93b8fe68c43b14.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/c60418e6536378a8fa86724a76d289717b659c3c2a8089658a93b8fe68c43b14.jpg)

![d6ba4b170a9a0e57fa10cd86defc7b066917c9901342aa9039a748901830f82c.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/d6ba4b170a9a0e57fa10cd86defc7b066917c9901342aa9039a748901830f82c.jpg)

![e3ab4b5d16e6cd9197aa0843f12aa71a858b864592073a0cb1a27a4c532dea70.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/e3ab4b5d16e6cd9197aa0843f12aa71a858b864592073a0cb1a27a4c532dea70.jpg)

![f81a14032d65ad9b67c0d4fc3a2843309e44ded9c5b9b74018f87c0cba9c2e83.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/images/f81a14032d65ad9b67c0d4fc3a2843309e44ded9c5b9b74018f87c0cba9c2e83.jpg)

### Tables

![02c6c9983c504c4fab5b32eea86d53f4f2ba643e4a87f8657037a2bdfae39f00.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/02c6c9983c504c4fab5b32eea86d53f4f2ba643e4a87f8657037a2bdfae39f00.jpg)

![03be42d59fa01bbbb5406c2b5552ce4889e287e6222251b32a7a14663418b368.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/03be42d59fa01bbbb5406c2b5552ce4889e287e6222251b32a7a14663418b368.jpg)

![04957069f792569810b6e440a163f1c2d715a264c2b6799801a5de49facd7b49.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/04957069f792569810b6e440a163f1c2d715a264c2b6799801a5de49facd7b49.jpg)

![077250565cdcb0f1b6c2367b86f49d53b7687ed53dc0e46e869e2f30f58fe36d.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/077250565cdcb0f1b6c2367b86f49d53b7687ed53dc0e46e869e2f30f58fe36d.jpg)

![164d473e47dd3a1134e210d31ed009fa6c1f3380bcc45694df59b7ee127ee19e.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/164d473e47dd3a1134e210d31ed009fa6c1f3380bcc45694df59b7ee127ee19e.jpg)

![1ed933adf8c7645e0bf28fd5e38e9274a14d971f8014f7892e74b3425f88c3df.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/1ed933adf8c7645e0bf28fd5e38e9274a14d971f8014f7892e74b3425f88c3df.jpg)

![237cb0badcbfda39cfb72815325475c9233c1fe570c5ea6ab8b3e8b68e623bc5.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/237cb0badcbfda39cfb72815325475c9233c1fe570c5ea6ab8b3e8b68e623bc5.jpg)

![254e6c2ad0796f7f2059a8f025c668d6684e933d4fb010e56cbb41997707960d.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/254e6c2ad0796f7f2059a8f025c668d6684e933d4fb010e56cbb41997707960d.jpg)

![26ac099e2f1d166540c896accdb3ca6bcdeac49a484da9d93b9bda9a7376f44c.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/26ac099e2f1d166540c896accdb3ca6bcdeac49a484da9d93b9bda9a7376f44c.jpg)

![26d17f64c001c904be3492a607f217bf69eec89fc4ef13753c0c60a3c872a8e8.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/26d17f64c001c904be3492a607f217bf69eec89fc4ef13753c0c60a3c872a8e8.jpg)

![34120a2a41633793720de18df1cb203ebbd6109bebb3589d83c869fbc832955a.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/34120a2a41633793720de18df1cb203ebbd6109bebb3589d83c869fbc832955a.jpg)

![361cd467131bbfc7148382ba016ff55609b57dad9e2bf9944ec18c7bd6b4fd3d.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/361cd467131bbfc7148382ba016ff55609b57dad9e2bf9944ec18c7bd6b4fd3d.jpg)

![4c47bdcf155c8feff44c9d09ccca90d37904807b44ac75eeacdefe18cbc0e44f.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/4c47bdcf155c8feff44c9d09ccca90d37904807b44ac75eeacdefe18cbc0e44f.jpg)

![4dcc6a2f9be26e69a662866b41c1ce482a3301884efb4ea08df786213b396ef0.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/4dcc6a2f9be26e69a662866b41c1ce482a3301884efb4ea08df786213b396ef0.jpg)

![7385193359f114574fd3e70fdc97960e2f93f6ef38098ba8914aa0329c083663.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/7385193359f114574fd3e70fdc97960e2f93f6ef38098ba8914aa0329c083663.jpg)

![74b143d57d2d47dd56042453f10d0e5e8b9de2ecabeb4db7b2485d3844254072.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/74b143d57d2d47dd56042453f10d0e5e8b9de2ecabeb4db7b2485d3844254072.jpg)

![8d63f96fef46b01bab6ba5249a21da09cfbbb9da691c9099a43d01325109a0c1.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/8d63f96fef46b01bab6ba5249a21da09cfbbb9da691c9099a43d01325109a0c1.jpg)

![9d45a1e692ffb9d767b0637f001780f910d3e44a9d282a3735368367312a7157.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/9d45a1e692ffb9d767b0637f001780f910d3e44a9d282a3735368367312a7157.jpg)

![9e57718078cd604ebbed58103e551638867cba2131899d0c5c56f281fd5722e7.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/9e57718078cd604ebbed58103e551638867cba2131899d0c5c56f281fd5722e7.jpg)

![9e98306b3e7cb8aff578c2ef6d39cedeb64017465542414059b3818f9008fae0.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/9e98306b3e7cb8aff578c2ef6d39cedeb64017465542414059b3818f9008fae0.jpg)

![a59b690d764feae3cacf32ca5af34cca45df41ca4fd019b36a2602d5de84116f.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/a59b690d764feae3cacf32ca5af34cca45df41ca4fd019b36a2602d5de84116f.jpg)

![a74c1ecb85e9eb358e2d670ffeb249174952a5662daff47bb1298acb608b90c6.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/a74c1ecb85e9eb358e2d670ffeb249174952a5662daff47bb1298acb608b90c6.jpg)

![b6701a00c8e95be04f1bc5e3e1f66ca56ea1ff2afbd7c1bef8761168a6c95ff3.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/b6701a00c8e95be04f1bc5e3e1f66ca56ea1ff2afbd7c1bef8761168a6c95ff3.jpg)

![bb36f7712fb3d4b1cec7a6b530d35fe017e3d3a9b4b1dcf4eb1b7f4d9965ac89.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/bb36f7712fb3d4b1cec7a6b530d35fe017e3d3a9b4b1dcf4eb1b7f4d9965ac89.jpg)

![df61de71c0ed749a8057d8ccd58bd965b3c6b56f0837b3989c0fcc8976af0a93.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/df61de71c0ed749a8057d8ccd58bd965b3c6b56f0837b3989c0fcc8976af0a93.jpg)

![fd37ed555e7dc6a7f98d72f13eae3cbfda5c65b7652de27c1640f4ac349b349d.jpg](../icml_results/2110_AlphaGRPO_%20Unlocking%20Self-Reflective%20Multimodal%20Generation%20in%20Unified%20Multimodal%20Models%20via%20Decompos/tables/fd37ed555e7dc6a7f98d72f13eae3cbfda5c65b7652de27c1640f4ac349b349d.jpg)

<a id="paper-2083"></a>

## Does Your Reasoning Model Implicitly Know When to Stop Thinking?


### Images

![24075e4afe118c5e4a97fd47fea719c2b13ed05c9bf9056669daaacfcdf24e95.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/24075e4afe118c5e4a97fd47fea719c2b13ed05c9bf9056669daaacfcdf24e95.jpg)

![2badaf1eda6cecf4c0afcfd302eb96525093d0c80483f5dd0aff26c9af1f0c6d.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/2badaf1eda6cecf4c0afcfd302eb96525093d0c80483f5dd0aff26c9af1f0c6d.jpg)

![323ae974386a9175ec547dfcb641322b23aee91021f2ce5579e0841bb6132b57.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/323ae974386a9175ec547dfcb641322b23aee91021f2ce5579e0841bb6132b57.jpg)

![3413108a90198f9d65b9aed8c5f97810a8f2feb6a6cbc79c6872eb2d6aabd744.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/3413108a90198f9d65b9aed8c5f97810a8f2feb6a6cbc79c6872eb2d6aabd744.jpg)

![401d91398f421821dbc5cd8efb972f9e3425e3e3a89591abc784e82c0490aa2c.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/401d91398f421821dbc5cd8efb972f9e3425e3e3a89591abc784e82c0490aa2c.jpg)

![49cbc985fcb8a5134313588c8b76635ad9afb2cffa92243b88a33bd1ad34ace5.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/49cbc985fcb8a5134313588c8b76635ad9afb2cffa92243b88a33bd1ad34ace5.jpg)

![548569ce27df94b0937344c1118a12991404460586e8d3c729f9754c102abfc1.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/548569ce27df94b0937344c1118a12991404460586e8d3c729f9754c102abfc1.jpg)

![6c0437b99488138e653fa5ffae113af6ac339aa410d5829d0f76daf1d1fc025b.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/6c0437b99488138e653fa5ffae113af6ac339aa410d5829d0f76daf1d1fc025b.jpg)

![7bfe3d7af0921d9902e726169dac6e318c26a6b5c3759a054916b9ba48b5eb83.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/7bfe3d7af0921d9902e726169dac6e318c26a6b5c3759a054916b9ba48b5eb83.jpg)

![7d3818b848346cac0b976c7db13a45076ea57a71b828faadee9a7839a0591f99.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/7d3818b848346cac0b976c7db13a45076ea57a71b828faadee9a7839a0591f99.jpg)

![854790bb0a099fddfb544805e103ff499d6525a4284ecf760bb85486f44f28f1.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/854790bb0a099fddfb544805e103ff499d6525a4284ecf760bb85486f44f28f1.jpg)

![9fc6f085ffda8fcfe0fdfe0506f65e1935a5499cf0e95551aa6858c7b43f3934.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/9fc6f085ffda8fcfe0fdfe0506f65e1935a5499cf0e95551aa6858c7b43f3934.jpg)

![abbcc830f3d880f69c00edffa2831a1fc5ee66e8acd37a0221061311871b5503.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/abbcc830f3d880f69c00edffa2831a1fc5ee66e8acd37a0221061311871b5503.jpg)

![b014eae89285d0ac0520ecb8eede9d96bca1ecbc28e3e1d4c661822987ab15a3.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/b014eae89285d0ac0520ecb8eede9d96bca1ecbc28e3e1d4c661822987ab15a3.jpg)

![ccb9e273e4d20da3eaadebb6fb7d52b6ff27fab53ff63187e5acbda9e1b76275.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/ccb9e273e4d20da3eaadebb6fb7d52b6ff27fab53ff63187e5acbda9e1b76275.jpg)

![cef24244d0427a7db5ce562564611b48d93da888694f234a17d79b814bd724d2.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/cef24244d0427a7db5ce562564611b48d93da888694f234a17d79b814bd724d2.jpg)

![d0f3e09d3d8be44ed3116bd101a7b891f8cc8cd965079cb4e5b88d269f7e6986.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/d0f3e09d3d8be44ed3116bd101a7b891f8cc8cd965079cb4e5b88d269f7e6986.jpg)

![d11c4b874a0f4c7156d591f69de82af7cb1e871c76883b0e65cf964e5361e922.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/d11c4b874a0f4c7156d591f69de82af7cb1e871c76883b0e65cf964e5361e922.jpg)

![d700c869dfad7b1b6b39849a0506628ccc4166dc95e163b34711685f37346633.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/d700c869dfad7b1b6b39849a0506628ccc4166dc95e163b34711685f37346633.jpg)

![daa4f94ef80dbd9ca6697630b4b37aef795f84b81180985b050e82419db38a1c.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/daa4f94ef80dbd9ca6697630b4b37aef795f84b81180985b050e82419db38a1c.jpg)

![e09fe9812186b2b9b66521ac75e8f49bc5b2c70a20741552c3bef00df1836c83.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/e09fe9812186b2b9b66521ac75e8f49bc5b2c70a20741552c3bef00df1836c83.jpg)

![e0d242fbc1de8c01cdeb48624b80f495af909d5451e40441097f47aa0eb431a4.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/e0d242fbc1de8c01cdeb48624b80f495af909d5451e40441097f47aa0eb431a4.jpg)

![f6d004f6fb6d22668f0f2f191d863e7ca5d5930b4d503e58cf1936bf898fb953.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/images/f6d004f6fb6d22668f0f2f191d863e7ca5d5930b4d503e58cf1936bf898fb953.jpg)

### Tables

![3310b35ccb07f7dcbeacea8f2313c7aa37da45446de2c860e2183dd23010b7f9.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/3310b35ccb07f7dcbeacea8f2313c7aa37da45446de2c860e2183dd23010b7f9.jpg)

![5b05e14df303102b7840cab531c34c37ec0caaa0aae26c0afa328fcb8b5aab16.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/5b05e14df303102b7840cab531c34c37ec0caaa0aae26c0afa328fcb8b5aab16.jpg)

![6f01d4fdab49d130a07265900806a69a7bf6e74d6cf6b0dbc171d86b93a886a8.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/6f01d4fdab49d130a07265900806a69a7bf6e74d6cf6b0dbc171d86b93a886a8.jpg)

![7a143217582cf5f135a779873fb97b6ef4c90b67d4ff1168805ca8a9692abd53.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/7a143217582cf5f135a779873fb97b6ef4c90b67d4ff1168805ca8a9692abd53.jpg)

![c11f8b4b937e5957dbfcb6b8e929206be77212ab752ba0f09750e81164c9be32.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/c11f8b4b937e5957dbfcb6b8e929206be77212ab752ba0f09750e81164c9be32.jpg)

![dd9801213de0ab5efa07a022814c0b2a359a1653453c794b0bb7cfb648519c1f.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/dd9801213de0ab5efa07a022814c0b2a359a1653453c794b0bb7cfb648519c1f.jpg)

![e6b185f4ee1c71c2622eb27eedb8560f3fce295e8f844223b8bdefba596fc004.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/e6b185f4ee1c71c2622eb27eedb8560f3fce295e8f844223b8bdefba596fc004.jpg)

![f12a3cc519020bff9dde5e06bad1adf6ebf8dd6e2013f9b9f5fb2b30653bff20.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/f12a3cc519020bff9dde5e06bad1adf6ebf8dd6e2013f9b9f5fb2b30653bff20.jpg)

![f272699eba52f1412e56b138f81d0c2e2eb8896a76d7f2fb5fb9cc24a0d2281d.jpg](../icml_results/2111_Does%20Your%20Reasoning%20Model%20Implicitly%20Know%20When%20to%20Stop%20Thinking_/tables/f272699eba52f1412e56b138f81d0c2e2eb8896a76d7f2fb5fb9cc24a0d2281d.jpg)

<a id="paper-2084"></a>

## Real-Time Aligned Reward Model beyond Semantics


### Images

![24684c3b08487ba94a92f351c1cb98094ccdcd238baab1014186123ccd2c0bc2.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/24684c3b08487ba94a92f351c1cb98094ccdcd238baab1014186123ccd2c0bc2.jpg)

![42e76449a90fc2f743d78fdc267dfa1a68645221ebd75d3855e13887c719d4e1.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/42e76449a90fc2f743d78fdc267dfa1a68645221ebd75d3855e13887c719d4e1.jpg)

![662b59d4ae083966964711f78241437b2009e13cd15f900361c9b4c8c2417b16.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/662b59d4ae083966964711f78241437b2009e13cd15f900361c9b4c8c2417b16.jpg)

![69cb700dba220bf70bbf4ceb7f527804505e104ef9e3284d6a142c3a392c0148.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/69cb700dba220bf70bbf4ceb7f527804505e104ef9e3284d6a142c3a392c0148.jpg)

![ab9a7fb6cda3cea0035f8ed6d61dde60b63a0a5659475f037ff83978c7c4b9c7.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/ab9a7fb6cda3cea0035f8ed6d61dde60b63a0a5659475f037ff83978c7c4b9c7.jpg)

![ac4c703c6a012785e0e323649e9610e921faab9fd6aa488a1434feab051e2118.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/ac4c703c6a012785e0e323649e9610e921faab9fd6aa488a1434feab051e2118.jpg)

![b5a0498d3e5846e8537649f03326709b18e51c4f25ccd83a57df85f5a6ed8831.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/b5a0498d3e5846e8537649f03326709b18e51c4f25ccd83a57df85f5a6ed8831.jpg)

![da9c570a93b47679b4d3dc968cab3c0d75631b69532e49becc2a7595a60695a6.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/da9c570a93b47679b4d3dc968cab3c0d75631b69532e49becc2a7595a60695a6.jpg)

![f6a7ed7762f1c5d2fadbc0a26b9fe73380e44642046312253f64322e70f9eb75.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/images/f6a7ed7762f1c5d2fadbc0a26b9fe73380e44642046312253f64322e70f9eb75.jpg)

### Tables

![140eb22c86178fdd5cf07f2101ee10cb184304fab9903f7c08c2820dd20127de.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/tables/140eb22c86178fdd5cf07f2101ee10cb184304fab9903f7c08c2820dd20127de.jpg)

![48a65170bffc171442c502e963e5c6f449da6f30b4e87a9f11b5d3341de79245.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/tables/48a65170bffc171442c502e963e5c6f449da6f30b4e87a9f11b5d3341de79245.jpg)

![96982d45e441c5831a8d03fb06efd4df79a0509ab51b71559021452344995ed2.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/tables/96982d45e441c5831a8d03fb06efd4df79a0509ab51b71559021452344995ed2.jpg)

![ae968b16aca5619750851bb777aabc7092e5533e9d52457bcac80eb5254a5aae.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/tables/ae968b16aca5619750851bb777aabc7092e5533e9d52457bcac80eb5254a5aae.jpg)

![ca48dd141ee05ab051df77d8c7059c9b8807d1c8cd62e062ceb9b75392d6c83d.jpg](../icml_results/2112_Real-Time%20Aligned%20Reward%20Model%20beyond%20Semantics/tables/ca48dd141ee05ab051df77d8c7059c9b8807d1c8cd62e062ceb9b75392d6c83d.jpg)

<a id="paper-2085"></a>

## Delegation and Verification under AI


### Images

![0b7a1af5b5f1d17f296608a19a4642502b54ba13994d9101025da80987d877e5.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/0b7a1af5b5f1d17f296608a19a4642502b54ba13994d9101025da80987d877e5.jpg)

![39c83295a14af014fd022f726ec2d67439002fee9d22f0cdf878800e36ee3f89.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/39c83295a14af014fd022f726ec2d67439002fee9d22f0cdf878800e36ee3f89.jpg)

![3fe51def31efb4ba5ed36f057beb1cd42c272875c5b94c718bb0cbf6f2c4171d.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/3fe51def31efb4ba5ed36f057beb1cd42c272875c5b94c718bb0cbf6f2c4171d.jpg)

![516307eb84ed81f41d70fb9736e057e1dee4593591ec1636ccaba617ca3791b9.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/516307eb84ed81f41d70fb9736e057e1dee4593591ec1636ccaba617ca3791b9.jpg)

![524b56f110838989e92693952aa663e9e39d0de5e4b7d37dc2a47448e8af4fcf.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/524b56f110838989e92693952aa663e9e39d0de5e4b7d37dc2a47448e8af4fcf.jpg)

![5d2192a0c064380961c6bd45de0994c23e1c9577f04086dddc1c59fadf340c04.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/5d2192a0c064380961c6bd45de0994c23e1c9577f04086dddc1c59fadf340c04.jpg)

![8f8b4f5a176ae10c57f14f792effadd876de94998962578897e58abf087a8add.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/8f8b4f5a176ae10c57f14f792effadd876de94998962578897e58abf087a8add.jpg)

![984dae549e8dd71939a2bc214c3d12e8b87654e59f9c86db124040ea74b80588.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/984dae549e8dd71939a2bc214c3d12e8b87654e59f9c86db124040ea74b80588.jpg)

![c7d7dffeb4bf2e4a5b530d9e5f1daaebc71d0c4b0803be62edc428cea42ccf6f.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/c7d7dffeb4bf2e4a5b530d9e5f1daaebc71d0c4b0803be62edc428cea42ccf6f.jpg)

![ca44f1a4dc1bd12b33ff4d1d6b50f2c7a1c888a90e57b3c69496d55425072402.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/ca44f1a4dc1bd12b33ff4d1d6b50f2c7a1c888a90e57b3c69496d55425072402.jpg)

![ce078871d0cd7b796ccc0fcc692dd3d5a10ef7fe1db8dc3a1d4c186e2076322f.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/ce078871d0cd7b796ccc0fcc692dd3d5a10ef7fe1db8dc3a1d4c186e2076322f.jpg)

![de566e4b572dea2e9900181c0dc0d56a6b44ad7dc21b79ffe2e9ac4794821636.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/de566e4b572dea2e9900181c0dc0d56a6b44ad7dc21b79ffe2e9ac4794821636.jpg)

![e2f0b8e5eac6403a5e7a44bb2892fc62ce733c69967a4e0964c3468d036ab7e0.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/e2f0b8e5eac6403a5e7a44bb2892fc62ce733c69967a4e0964c3468d036ab7e0.jpg)

![fb9d8fda1fa0b59580a959a22ca59f2fd524d059a77a87ee3c120a905ef89fef.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/images/fb9d8fda1fa0b59580a959a22ca59f2fd524d059a77a87ee3c120a905ef89fef.jpg)

### Tables

![2e899e975136e3c8a8316cf0f211c199ce9663d9082e1a80b1747c597acbad2c.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/tables/2e899e975136e3c8a8316cf0f211c199ce9663d9082e1a80b1747c597acbad2c.jpg)

![7723957dbc6da1dc80e2fb3d0fbc07a39d8a48b7f4e0c924149f81316c471223.jpg](../icml_results/2113_Delegation%20and%20Verification%20under%20AI/tables/7723957dbc6da1dc80e2fb3d0fbc07a39d8a48b7f4e0c924149f81316c471223.jpg)

<a id="paper-2086"></a>

## How to Fine-Tune a Reasoning Model? A Teacher–Student Cooperation Framework to Synthesize Student-Consistent SFT Data


### Images

![004f0ced52470872ea7743fde8050e34cb2cf3a9fe954b14363662fc36a9e9bf.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/004f0ced52470872ea7743fde8050e34cb2cf3a9fe954b14363662fc36a9e9bf.jpg)

![0c5c6c22564825cd643c925b22fd57894ac7a4cfd77328d5f7273ca6981de075.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/0c5c6c22564825cd643c925b22fd57894ac7a4cfd77328d5f7273ca6981de075.jpg)

![16a444296854a9a294326d78ac508fe8ad7154b10615fe05d876fececd507d5b.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/16a444296854a9a294326d78ac508fe8ad7154b10615fe05d876fececd507d5b.jpg)

![1cb55f745fdc2c6dba322a6f8ba917ca4d3519118e1ce40b4f5c5d497767f2b1.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/1cb55f745fdc2c6dba322a6f8ba917ca4d3519118e1ce40b4f5c5d497767f2b1.jpg)

![1f94c45642152ed36e1b747aa880c582c320c7a09b0b9db8dfb16fccc46a0292.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/1f94c45642152ed36e1b747aa880c582c320c7a09b0b9db8dfb16fccc46a0292.jpg)

![5876f503e0188a47b9c57674d798fd4edcbbf0e8833cf9c5c78bf0bd23165784.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/5876f503e0188a47b9c57674d798fd4edcbbf0e8833cf9c5c78bf0bd23165784.jpg)

![6d09dd43739d02bf049d0e7a55f7dd5f48e9e73b01df7b692a785deb19876b41.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/6d09dd43739d02bf049d0e7a55f7dd5f48e9e73b01df7b692a785deb19876b41.jpg)

![6e18e6fccdb24fe4c9ee2a5dcaae9531c831d5316202e68dd900de1cc3e64dc9.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/6e18e6fccdb24fe4c9ee2a5dcaae9531c831d5316202e68dd900de1cc3e64dc9.jpg)

![88281be13e5461583a31c3f58b6b758a4f3a4a786e957207a2b3f5a04cb1dd9c.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/88281be13e5461583a31c3f58b6b758a4f3a4a786e957207a2b3f5a04cb1dd9c.jpg)

![88f9ff76ce92e261ac3da303d8eae28904b4ec934a35ddcea86d19d9ae182e41.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/88f9ff76ce92e261ac3da303d8eae28904b4ec934a35ddcea86d19d9ae182e41.jpg)

![8a93235c6ae339aa6d7cbd4b82b149ebe65dea7eccfa2c4cefb139f7b50cfef1.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/8a93235c6ae339aa6d7cbd4b82b149ebe65dea7eccfa2c4cefb139f7b50cfef1.jpg)

![8b840a461dd373ddae32d21b86566150649c78971d855641c483e41b1adc7a5f.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/8b840a461dd373ddae32d21b86566150649c78971d855641c483e41b1adc7a5f.jpg)

![8bc8ce46ad07a1cb8926c324e236260f16436725e855682036aa49c7c65ff2aa.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/8bc8ce46ad07a1cb8926c324e236260f16436725e855682036aa49c7c65ff2aa.jpg)

![8fc709eb57ad3b9424e7656e1d2e60224b3c60f9d713d37cc16f7586f10ab9ae.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/8fc709eb57ad3b9424e7656e1d2e60224b3c60f9d713d37cc16f7586f10ab9ae.jpg)

![910ab737c2dfc958da08391f7c4fc347ae4fafa7b80cd32385e975654eba5149.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/910ab737c2dfc958da08391f7c4fc347ae4fafa7b80cd32385e975654eba5149.jpg)

![94faf8fd93e8a990d4fd442805f74a4b39cec9341c04e2675751a417665dd47e.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/94faf8fd93e8a990d4fd442805f74a4b39cec9341c04e2675751a417665dd47e.jpg)

![975880d864db8c554e8fcbef8c787ca9121b809a973236bbfda1ff5ef8f0c2ef.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/975880d864db8c554e8fcbef8c787ca9121b809a973236bbfda1ff5ef8f0c2ef.jpg)

![a899b1dd46736c2964a88d3e4c34aea0f4e76404d1506f9647486ca620a7715e.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/a899b1dd46736c2964a88d3e4c34aea0f4e76404d1506f9647486ca620a7715e.jpg)

![b4bbe4af4cb71fa7f7b7d8c4e51f8c30367260d3f40356327aa7941d213e16c3.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/b4bbe4af4cb71fa7f7b7d8c4e51f8c30367260d3f40356327aa7941d213e16c3.jpg)

![ed38065b0432607839af7cdf81b634842ea1f0bcd5406369205144ad422703b7.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/ed38065b0432607839af7cdf81b634842ea1f0bcd5406369205144ad422703b7.jpg)

![f66804a99982750d97ab3bd24251111b0ea5bfa27ae89855a00da45945e1d873.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/images/f66804a99982750d97ab3bd24251111b0ea5bfa27ae89855a00da45945e1d873.jpg)

### Tables

![031acb9e9f02f921f67b8d9b7dfdb94f595b827be6eeb5f649aa3e3b7bfcd8d4.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/tables/031acb9e9f02f921f67b8d9b7dfdb94f595b827be6eeb5f649aa3e3b7bfcd8d4.jpg)

![8eba63c304bc97ebd904d03ff67a555bd79ba7b1ebca5c9a46b6a9bd2dda5f45.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/tables/8eba63c304bc97ebd904d03ff67a555bd79ba7b1ebca5c9a46b6a9bd2dda5f45.jpg)

![ec74b4d322ad67e326b41278056088309a77db3deed09e92f8f19c4b5255280e.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/tables/ec74b4d322ad67e326b41278056088309a77db3deed09e92f8f19c4b5255280e.jpg)

![ed326fda600232dfc2e82118a38e7a8ec2f2650694d14b3dbc2aee4b96e375f1.jpg](../icml_results/2114_How%20to%20Fine-Tune%20a%20Reasoning%20Model_%20A%20Teacher%E2%80%93Student%20Cooperation%20Framework%20to%20Synthesize%20Student-Co/tables/ed326fda600232dfc2e82118a38e7a8ec2f2650694d14b3dbc2aee4b96e375f1.jpg)

<a id="paper-2087"></a>

## FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models


### Images

![20926a7f763bcf59bc25b9a632ff18fd27389a2b6a5239f65c3d337f99cf6695.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/20926a7f763bcf59bc25b9a632ff18fd27389a2b6a5239f65c3d337f99cf6695.jpg)

![29a77125656a00c28d376526551c410ab320b651d2f151e881c39fb805f2a1a2.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/29a77125656a00c28d376526551c410ab320b651d2f151e881c39fb805f2a1a2.jpg)

![4cf0c8e4640e85c8a7fc7c356a41bf6c8da12934ab6380419f7df2ac0688ed92.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/4cf0c8e4640e85c8a7fc7c356a41bf6c8da12934ab6380419f7df2ac0688ed92.jpg)

![7b3c54a52069892ecec3013a73fc355f6e30fefd79aef7cd43b87bd300b324a4.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/7b3c54a52069892ecec3013a73fc355f6e30fefd79aef7cd43b87bd300b324a4.jpg)

![8f384b7a1e8b2efe108c3de209f3d6bae4241f200bbd8e98dffb73bda1498cb3.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/8f384b7a1e8b2efe108c3de209f3d6bae4241f200bbd8e98dffb73bda1498cb3.jpg)

![abfaa380baf14dd1e977a94b9457fd33a09d8ef7ac06836bdbe4bdf22dd49616.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/abfaa380baf14dd1e977a94b9457fd33a09d8ef7ac06836bdbe4bdf22dd49616.jpg)

![bffec1847e049cbbb1eba17831254a71c277184e886dde0baa5b6e1a54ed0a00.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/bffec1847e049cbbb1eba17831254a71c277184e886dde0baa5b6e1a54ed0a00.jpg)

![fc2412e16153b03dd37b9f4a7a626ca9c4fa25299180777070d1a271958b4cfe.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/images/fc2412e16153b03dd37b9f4a7a626ca9c4fa25299180777070d1a271958b4cfe.jpg)

### Tables

![12f74c7d6ef2006f88a61dfcdec78d98fe80e610c5d2f554ff0f703d242b863a.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/12f74c7d6ef2006f88a61dfcdec78d98fe80e610c5d2f554ff0f703d242b863a.jpg)

![1e118a21fb19879eaff5648bf2afea93936096ab610084494eb28adbcf1ed6c9.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/1e118a21fb19879eaff5648bf2afea93936096ab610084494eb28adbcf1ed6c9.jpg)

![320cdc3cca93023524f918d68330701419a38b4702a5d153dec3b09456bebfde.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/320cdc3cca93023524f918d68330701419a38b4702a5d153dec3b09456bebfde.jpg)

![6b0011b7c347acad1cd70f0d04a01460de4567b4ba87849d52473ebd7eeeedf6.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/6b0011b7c347acad1cd70f0d04a01460de4567b4ba87849d52473ebd7eeeedf6.jpg)

![8fad82c2583e09c0aabe26280eeb50e9addcc364ee12535610e73d7ae93091cf.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/8fad82c2583e09c0aabe26280eeb50e9addcc364ee12535610e73d7ae93091cf.jpg)

![93beec2d7b3f7af32e3b60bb27521d3d735ee2b0389a3e2c9ccf375af642292c.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/93beec2d7b3f7af32e3b60bb27521d3d735ee2b0389a3e2c9ccf375af642292c.jpg)

![9db74b9cdba8b22f631f502c8242e8f721b913e674d53371b0a47a9d68d1a312.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/9db74b9cdba8b22f631f502c8242e8f721b913e674d53371b0a47a9d68d1a312.jpg)

![d64467530125cc7665bda5bec7af67dfeb233c7b423dcb2efeb74c0937b332eb.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/d64467530125cc7665bda5bec7af67dfeb233c7b423dcb2efeb74c0937b332eb.jpg)

![dd37a3bb53d657fe938bad4758c60aa961135fc8ac6e3c1a891e5176a8dc5bdd.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/dd37a3bb53d657fe938bad4758c60aa961135fc8ac6e3c1a891e5176a8dc5bdd.jpg)

![e60aa743bce2ecb9446c4ea1a807a7d6591aa16f7d957af8cadac21678e84702.jpg](../icml_results/2115_FAIR-Calib_%20Frontier-Aware%20Instability-Reweighted%20Calibration%20for%20Post-Training%20Quantization%20of%20Diff/tables/e60aa743bce2ecb9446c4ea1a807a7d6591aa16f7d957af8cadac21678e84702.jpg)

<a id="paper-2088"></a>

## Exploring Motif-based Heterogeneous Graph Learning for ReDoS Detection


### Images

![110361882a8ec519ba310286908bc8d2fd1fbb0aca5556cf94a336d54a065c4f.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/110361882a8ec519ba310286908bc8d2fd1fbb0aca5556cf94a336d54a065c4f.jpg)

![2824576b4e642a25baf6ad3ac12840fc261b5771834b06a7fde15e5d370a2d55.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/2824576b4e642a25baf6ad3ac12840fc261b5771834b06a7fde15e5d370a2d55.jpg)

![36f3722a4d6688ce5834552eaedb9b0b27f27fe6b17190434578219189c63e01.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/36f3722a4d6688ce5834552eaedb9b0b27f27fe6b17190434578219189c63e01.jpg)

![65812da12cf54cecbf1bfcb1ba3d76c62bda93b1f8c02759cfe1fc4ed7afcbed.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/65812da12cf54cecbf1bfcb1ba3d76c62bda93b1f8c02759cfe1fc4ed7afcbed.jpg)

![817a5ef9c5cc125f21073eda203039df47b4a54a339292089fb220a3c0917ee6.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/817a5ef9c5cc125f21073eda203039df47b4a54a339292089fb220a3c0917ee6.jpg)

![8de5d68e73e9bf184d00ea3098dee54b5fec774588ea19159f53352c982f077d.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/8de5d68e73e9bf184d00ea3098dee54b5fec774588ea19159f53352c982f077d.jpg)

![ea4388ced8ff40dbe4566746aeb597b8450c2e1b7ec4bba083dd86908d241d9f.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/ea4388ced8ff40dbe4566746aeb597b8450c2e1b7ec4bba083dd86908d241d9f.jpg)

![f9177f06b832812918de188d7b5ab7148b1ff7132221505fff4b40637fcc340a.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/f9177f06b832812918de188d7b5ab7148b1ff7132221505fff4b40637fcc340a.jpg)

![fd014c21b67bb1fa4d7cd48b0093fc399557ba8b22aa792df30bedc7ab7be5f6.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/images/fd014c21b67bb1fa4d7cd48b0093fc399557ba8b22aa792df30bedc7ab7be5f6.jpg)

### Tables

![2a35d7e315aabfb70fdb71231909dc6321cbb9c253be222762267fa13e7ddb48.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/2a35d7e315aabfb70fdb71231909dc6321cbb9c253be222762267fa13e7ddb48.jpg)

![52700942bd03aa6de5904b7c847e881948f0114cecfa57cfab02d74c814ab2ff.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/52700942bd03aa6de5904b7c847e881948f0114cecfa57cfab02d74c814ab2ff.jpg)

![71918b4b83f3d593c116d8606a9f6f83b40b5cd32f139e4dfd0b24aa33284bd3.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/71918b4b83f3d593c116d8606a9f6f83b40b5cd32f139e4dfd0b24aa33284bd3.jpg)

![82767fb52536ef2988114823751f44b76172ebf8cd4e8b0dd45d6ad854480d9c.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/82767fb52536ef2988114823751f44b76172ebf8cd4e8b0dd45d6ad854480d9c.jpg)

![8ebfa3f7b0c085f5f128a11264bf0ba9a6422af20b525d1e06a945abffeb4c2b.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/8ebfa3f7b0c085f5f128a11264bf0ba9a6422af20b525d1e06a945abffeb4c2b.jpg)

![d24d57e7c6007b4c46bd41b12081321eed48c30e834b4cd70fd5473924fc8f9f.jpg](../icml_results/2116_Exploring%20Motif-based%20Heterogeneous%20Graph%20Learning%20for%20ReDoS%20Detection/tables/d24d57e7c6007b4c46bd41b12081321eed48c30e834b4cd70fd5473924fc8f9f.jpg)

<a id="paper-2089"></a>

## LIVE: Long-horizon Interactive Video World Modeling


### Images

![1c2cb256876d3ded21866c04819b9791f8c654d2b707209059ea072a99acf9ab.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/1c2cb256876d3ded21866c04819b9791f8c654d2b707209059ea072a99acf9ab.jpg)

![1d5c980288180bcc13b3225dfe0d5bb5aaf83538e8c7abf677293a0f8697c5c1.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/1d5c980288180bcc13b3225dfe0d5bb5aaf83538e8c7abf677293a0f8697c5c1.jpg)

![290738b6dc1eb5759ef6794521e4a083ff3820bffa8c83053421c51f5656f64f.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/290738b6dc1eb5759ef6794521e4a083ff3820bffa8c83053421c51f5656f64f.jpg)

![2d7f49eafa6b1847b1d14877047514237bd8ac37df7463408951682c0a507aac.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/2d7f49eafa6b1847b1d14877047514237bd8ac37df7463408951682c0a507aac.jpg)

![2d8d8b19d55d6b495db76b53d65f3dcb86bd4632fab6bd85c2d579030bfcfd95.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/2d8d8b19d55d6b495db76b53d65f3dcb86bd4632fab6bd85c2d579030bfcfd95.jpg)

![7bb341582fafa2164680913d99400f287ca38f450539fdbbe2dd3cacc18950af.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/7bb341582fafa2164680913d99400f287ca38f450539fdbbe2dd3cacc18950af.jpg)

![83cddad5dd8ab6f40f31e24d309a69f22927a4397b3aeb09ad42875392974180.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/83cddad5dd8ab6f40f31e24d309a69f22927a4397b3aeb09ad42875392974180.jpg)

![8744fcfb5697ed65dd997da5afb2835460f9f66a1ba475c5bdec461c6af0c1f8.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/8744fcfb5697ed65dd997da5afb2835460f9f66a1ba475c5bdec461c6af0c1f8.jpg)

![b0488fa0b6a1cec5100f1cced97638b7f3685adb69bcaef7e94c66f7de1f0657.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/b0488fa0b6a1cec5100f1cced97638b7f3685adb69bcaef7e94c66f7de1f0657.jpg)

![b063c473d2676553a448ba0ac9230bca9b426115eebaf1066ad54e433b51623d.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/b063c473d2676553a448ba0ac9230bca9b426115eebaf1066ad54e433b51623d.jpg)

![d1448443972a6f049bce62e9b796242ef8be28e05aae7385f9fb90ceb4aa0992.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/d1448443972a6f049bce62e9b796242ef8be28e05aae7385f9fb90ceb4aa0992.jpg)

![d5d7529bdd5c6d2f7f85da3eccc41b6ba80d75e3fef3f73644c73959de30ccc7.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/d5d7529bdd5c6d2f7f85da3eccc41b6ba80d75e3fef3f73644c73959de30ccc7.jpg)

![dbc5a379264ea49d85f2d3bf168456e8aa7c350d2fc0f3f70b75db81c0a3b2e8.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/dbc5a379264ea49d85f2d3bf168456e8aa7c350d2fc0f3f70b75db81c0a3b2e8.jpg)

![dd220657b83a673f1d2b4b610676770c0143861745dc41f718adbdef89e24202.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/dd220657b83a673f1d2b4b610676770c0143861745dc41f718adbdef89e24202.jpg)

![ea3dd34e1679d6934deba9bf18617a1334ab4ef44936ab3a19684e537277b0d7.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/ea3dd34e1679d6934deba9bf18617a1334ab4ef44936ab3a19684e537277b0d7.jpg)

![eaf822814090c4c52fd3be1c229cd546eac5a0ba3ee3c282607818c225d2fd79.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/eaf822814090c4c52fd3be1c229cd546eac5a0ba3ee3c282607818c225d2fd79.jpg)

![f465152ccc10fdb71e0d005c1f2d5dab7be23ddd738154aa8fa520c523e86f22.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/f465152ccc10fdb71e0d005c1f2d5dab7be23ddd738154aa8fa520c523e86f22.jpg)

![fe1472163ac4dcd133dcaa77721af24ddf5347e0bb57389cb981e06b6a58816e.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/images/fe1472163ac4dcd133dcaa77721af24ddf5347e0bb57389cb981e06b6a58816e.jpg)

### Tables

![036d561852d56cedfda0b5041a1fb8d0d0884ab3b545c198cb15e33720a107ef.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/tables/036d561852d56cedfda0b5041a1fb8d0d0884ab3b545c198cb15e33720a107ef.jpg)

![91540467887bf9bb8721d682586f64074ffcc726ebaf98f61305eedbc6ef5a35.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/tables/91540467887bf9bb8721d682586f64074ffcc726ebaf98f61305eedbc6ef5a35.jpg)

![d543523c6d10ff9794db6698bd403c6ef138d67341ec1b1dd473440d76a6952b.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/tables/d543523c6d10ff9794db6698bd403c6ef138d67341ec1b1dd473440d76a6952b.jpg)

![ee3a2019097ad39e98d20f7853608db631eb4937fcd8148b89a765c43128fe4f.jpg](../icml_results/2117_LIVE_%20Long-horizon%20Interactive%20Video%20World%20Modeling/tables/ee3a2019097ad39e98d20f7853608db631eb4937fcd8148b89a765c43128fe4f.jpg)

<a id="paper-2090"></a>

## FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models


### Images

![0816a1b719690aa624175a756bf7e949ff44b85ab247ff940765629f8186d908.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/0816a1b719690aa624175a756bf7e949ff44b85ab247ff940765629f8186d908.jpg)

![109a572089536af09cc759f05ce51a92ba78833c4cef35d6974a29e987450e0a.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/109a572089536af09cc759f05ce51a92ba78833c4cef35d6974a29e987450e0a.jpg)

![12822d053f7f5eba20cd482e6650543f7507c16c908423dbd495469081bf6432.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/12822d053f7f5eba20cd482e6650543f7507c16c908423dbd495469081bf6432.jpg)

![313d575c161425a73990f5fed0da7b549f807d1baee11441aba16db228d262c5.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/313d575c161425a73990f5fed0da7b549f807d1baee11441aba16db228d262c5.jpg)

![3d7ccd40d2e4a903c88fdc0f4353231d76c61a94ccb8c0281bf40befeb0c1f0e.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/3d7ccd40d2e4a903c88fdc0f4353231d76c61a94ccb8c0281bf40befeb0c1f0e.jpg)

![4ccf8ea611a45a09780f470dd7fba5cf426cacef4132913fdaa50b5da562ed12.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/4ccf8ea611a45a09780f470dd7fba5cf426cacef4132913fdaa50b5da562ed12.jpg)

![5410658c925657ccdcca3cdd21ecb6ec0db6bd2991809b507b60f73457f69488.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/5410658c925657ccdcca3cdd21ecb6ec0db6bd2991809b507b60f73457f69488.jpg)

![7a5919f7ad8efe59010a9300ffd2bcf8e9e26cbabb193a9c2d7b457812c0d1c2.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/7a5919f7ad8efe59010a9300ffd2bcf8e9e26cbabb193a9c2d7b457812c0d1c2.jpg)

![b634bd474c98cc22f7b9845708d47d0a504df883a13b224f5d41898b3545bce2.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/b634bd474c98cc22f7b9845708d47d0a504df883a13b224f5d41898b3545bce2.jpg)

![c1822d384aa25959012499e4a4a804a02bd2553c8fe547847242ea201813184b.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/c1822d384aa25959012499e4a4a804a02bd2553c8fe547847242ea201813184b.jpg)

![f206dab8e9fa4395d7d85a3561a39b9d3001ae4e636bda6a3be50c0cfbdba156.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/f206dab8e9fa4395d7d85a3561a39b9d3001ae4e636bda6a3be50c0cfbdba156.jpg)

![f38bb73dccd83aaf03ac9fbd348e82371f05560a5f2a9e5172fd1f484055c89b.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/f38bb73dccd83aaf03ac9fbd348e82371f05560a5f2a9e5172fd1f484055c89b.jpg)

![f9248992e01f608ccff910670faeb6c7726cb9bd37f0cc5284e72b732bad7cca.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/f9248992e01f608ccff910670faeb6c7726cb9bd37f0cc5284e72b732bad7cca.jpg)

![fa809cf2f275ab51d121f18492a89785207f52c48b8eb2941de8986c8ee01420.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/images/fa809cf2f275ab51d121f18492a89785207f52c48b8eb2941de8986c8ee01420.jpg)

### Tables

![204b3733f75f45a16b64a2a17070c07816bdf2cd6885bf7c660e1eb5f23fd88e.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/204b3733f75f45a16b64a2a17070c07816bdf2cd6885bf7c660e1eb5f23fd88e.jpg)

![2062c63fbb2f364af2b483679323f666a9e38ada72766a1ee005acd0b1504123.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/2062c63fbb2f364af2b483679323f666a9e38ada72766a1ee005acd0b1504123.jpg)

![2eb9815caf7a96057defc360b0c236ab5a3a365d387a4bdf6e4ea47e251a439c.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/2eb9815caf7a96057defc360b0c236ab5a3a365d387a4bdf6e4ea47e251a439c.jpg)

![369737dcf74e29ca3998fe258b7f410984137720488c2f17d07dff9f99ef90c9.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/369737dcf74e29ca3998fe258b7f410984137720488c2f17d07dff9f99ef90c9.jpg)

![375711a9f7f666f6c1c4b70ea292ead93d23ed3d7178f85cfc9301a19184b5cd.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/375711a9f7f666f6c1c4b70ea292ead93d23ed3d7178f85cfc9301a19184b5cd.jpg)

![7855daee75d54e4de874b0d7d52847727dca3578614a4b4019e364b89282abde.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/7855daee75d54e4de874b0d7d52847727dca3578614a4b4019e364b89282abde.jpg)

![a03e825aa264472621ad3d3a4136c118de47b746c7601dfbf45d19887287762a.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/a03e825aa264472621ad3d3a4136c118de47b746c7601dfbf45d19887287762a.jpg)

![db83d44e49d750e069a542108c00193f992fa2d95bf9a6d5a8c3c8a1cbc86214.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/db83d44e49d750e069a542108c00193f992fa2d95bf9a6d5a8c3c8a1cbc86214.jpg)

![dc0ddf524535175324bd9408bf0de7da6068110a9d54fb831ffc77560d768bf0.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/dc0ddf524535175324bd9408bf0de7da6068110a9d54fb831ffc77560d768bf0.jpg)

![e9d869a9fb94e53235d955b8fd4e297ba50e0cdf1d6f88d51b523c2a2afc4b49.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/e9d869a9fb94e53235d955b8fd4e297ba50e0cdf1d6f88d51b523c2a2afc4b49.jpg)

![f8d1532485f02232c1defec859deed02437644157cd52fe04a8a2fb094d6d08e.jpg](../icml_results/2118_FRISM_%20Fine-Grained%20Reasoning%20Injection%20via%20Subspace-Level%20Model%20Merging%20for%20Vision%E2%80%93Language%20Models/tables/f8d1532485f02232c1defec859deed02437644157cd52fe04a8a2fb094d6d08e.jpg)

<a id="paper-2091"></a>

## Cross-Embodiment Robot Foundation World Models with Latent Actions


### Images

![0b2985f967822a2a5e05089d37fb165edc3e3c08dc7840cf69ce92e63f40a593.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/0b2985f967822a2a5e05089d37fb165edc3e3c08dc7840cf69ce92e63f40a593.jpg)

![169403f60dd05cf32ee82d6bdf78aea55b8450d111445f3fcdb62c22042867a6.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/169403f60dd05cf32ee82d6bdf78aea55b8450d111445f3fcdb62c22042867a6.jpg)

![231025b6b3555b880137266d3434ae47453c033392116d65b878794651df75d5.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/231025b6b3555b880137266d3434ae47453c033392116d65b878794651df75d5.jpg)

![30ed891f9e2c2959e63c73ad1fbc3c6901b74c669501d2925f352423ed72f3ee.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/30ed891f9e2c2959e63c73ad1fbc3c6901b74c669501d2925f352423ed72f3ee.jpg)

![62f7507344ee9d8deae7ee0d11b857648ca79077b5b588453c3730661b0e6ee6.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/62f7507344ee9d8deae7ee0d11b857648ca79077b5b588453c3730661b0e6ee6.jpg)

![91b503f198410fa05a8e5b28ced83c9f053a85fbc85bc947c035a5bec6b5ffaa.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/91b503f198410fa05a8e5b28ced83c9f053a85fbc85bc947c035a5bec6b5ffaa.jpg)

![af09637761225242bd60f19cd0c9af7fadd72a2f30c0c0c55843a67e7d578b17.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/af09637761225242bd60f19cd0c9af7fadd72a2f30c0c0c55843a67e7d578b17.jpg)

![d19b6b34674efa8fccf2145a82f733925842576d8a838e49f1d9f5562538d235.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/d19b6b34674efa8fccf2145a82f733925842576d8a838e49f1d9f5562538d235.jpg)

![f1b5e80be72f277148bf6ddab1501ab24e491fa045ad55e03ed4ac2a830f78bc.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/images/f1b5e80be72f277148bf6ddab1501ab24e491fa045ad55e03ed4ac2a830f78bc.jpg)

### Tables

![0e3ca70d3d79769d56cee9d89d1275f9abfc6ada59e246641831ce4e3e6dfab2.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/0e3ca70d3d79769d56cee9d89d1275f9abfc6ada59e246641831ce4e3e6dfab2.jpg)

![37ea7a722c377eb90fdfcc3c4ec951fbcc29d17a4833501f2ea6eca74ea6a6ce.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/37ea7a722c377eb90fdfcc3c4ec951fbcc29d17a4833501f2ea6eca74ea6a6ce.jpg)

![4eafc9180ce64808fc9fcd91fe538fd823d486b8a096acfd7b68540b2cf2ff71.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/4eafc9180ce64808fc9fcd91fe538fd823d486b8a096acfd7b68540b2cf2ff71.jpg)

![67226404496c70a0840c45da32159badb7ebd6e464532af34ab34eab5599db41.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/67226404496c70a0840c45da32159badb7ebd6e464532af34ab34eab5599db41.jpg)

![cbd78cc9a0c6eb9d57bf19288626e6011318f3b94474de778f9cf1cff19983db.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/cbd78cc9a0c6eb9d57bf19288626e6011318f3b94474de778f9cf1cff19983db.jpg)

![dbbec04eefc035d1a202c968ebfabd91474cc780b396d90833031ee2b91b0598.jpg](../icml_results/2119_Cross-Embodiment%20Robot%20Foundation%20World%20Models%20with%20Latent%20Actions/tables/dbbec04eefc035d1a202c968ebfabd91474cc780b396d90833031ee2b91b0598.jpg)

<a id="paper-2092"></a>

## ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation


### Images

![012cdbff611bd1aa968f7babb6c897551e634450c607f63934f1e5dea875f249.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/012cdbff611bd1aa968f7babb6c897551e634450c607f63934f1e5dea875f249.jpg)

![42dc2353dd1252e7ca798eac646f5b5f146d9fcac0b2bb28c8f3fbeead185c3c.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/42dc2353dd1252e7ca798eac646f5b5f146d9fcac0b2bb28c8f3fbeead185c3c.jpg)

![69db492bdce66cd4f9bdc467dfca53302af1548f0f92ce26c8016369ce84af16.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/69db492bdce66cd4f9bdc467dfca53302af1548f0f92ce26c8016369ce84af16.jpg)

![73dc4d676209c96194d946d7a318c8f7d9f8b2246b58565d0689e3e59958021f.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/73dc4d676209c96194d946d7a318c8f7d9f8b2246b58565d0689e3e59958021f.jpg)

![7ee87c64ddfa5d4106944ad34f85376e5da105ed9ccbf767962162d114261dc3.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/7ee87c64ddfa5d4106944ad34f85376e5da105ed9ccbf767962162d114261dc3.jpg)

![9c2e26ccfce3669f0a026dd5b0695b5a5ad88c9d9bdcce69ba303e2d514b853d.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/9c2e26ccfce3669f0a026dd5b0695b5a5ad88c9d9bdcce69ba303e2d514b853d.jpg)

![adfc52dee2769e9d8fc9339bc0a8a1b6ff1bc8dcde827b357394ca6e6f295eb5.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/adfc52dee2769e9d8fc9339bc0a8a1b6ff1bc8dcde827b357394ca6e6f295eb5.jpg)

![bbc268b5d265b0e7a1b8854ff734ec9396c1c43c77e2b69dad8a95135432fb9c.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/bbc268b5d265b0e7a1b8854ff734ec9396c1c43c77e2b69dad8a95135432fb9c.jpg)

![de2077ed6154df71ad85efde029bc5ade2e52b7843698003bd341975a62813b7.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/de2077ed6154df71ad85efde029bc5ade2e52b7843698003bd341975a62813b7.jpg)

![e54eae5e9a5aad3cf56cbddc2fdae0cb98623f5df9e0a9b479a35699fd79d910.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/e54eae5e9a5aad3cf56cbddc2fdae0cb98623f5df9e0a9b479a35699fd79d910.jpg)

![ec4ddac68002442066d66ea9ff2aeead92f8534d985addf6884c33476ddf129b.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/ec4ddac68002442066d66ea9ff2aeead92f8534d985addf6884c33476ddf129b.jpg)

![f27cba34ba302626ed7ec62570f0f8ca6e32ce17dedcd26b16d86aebb1cb07b9.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/images/f27cba34ba302626ed7ec62570f0f8ca6e32ce17dedcd26b16d86aebb1cb07b9.jpg)

### Tables

![007faaba5ab26e91f6644d4c075efecae5d9f1c45421c26067a5ca1ccea02856.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/007faaba5ab26e91f6644d4c075efecae5d9f1c45421c26067a5ca1ccea02856.jpg)

![0c81ae3b9caa070339f464f7cd1d704f2450bfbf601b472843ff5a478136fc14.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/0c81ae3b9caa070339f464f7cd1d704f2450bfbf601b472843ff5a478136fc14.jpg)

![1e4b844bd3a20ea705b573ce4377f83e0c3e6046f6a1ed616735e524e3c2021a.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/1e4b844bd3a20ea705b573ce4377f83e0c3e6046f6a1ed616735e524e3c2021a.jpg)

![2eabe22d8d46c55b57b88f6d0900d0ab15fc332706cc2755bc9702d72d695130.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/2eabe22d8d46c55b57b88f6d0900d0ab15fc332706cc2755bc9702d72d695130.jpg)

![2fe7277f58af660e02652e24a65367a08f72042251127d7d989b84a09ef137da.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/2fe7277f58af660e02652e24a65367a08f72042251127d7d989b84a09ef137da.jpg)

![2ff17d15896ad1e694113ffb4326c4a99193a668228411c632e691275374f608.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/2ff17d15896ad1e694113ffb4326c4a99193a668228411c632e691275374f608.jpg)

![3009cb0debcca5cb6eccc84581e7c2b23995c6c0641906d048a071b7ba27c081.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/3009cb0debcca5cb6eccc84581e7c2b23995c6c0641906d048a071b7ba27c081.jpg)

![36a427fb828c182bc30f97dfc86275b9d03d65d17dee1727f70995f63c84f3b3.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/36a427fb828c182bc30f97dfc86275b9d03d65d17dee1727f70995f63c84f3b3.jpg)

![39d0f546aa270bbc20c6228465f5e6cba49e2e607e8b0aef14f9d1fe6226e705.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/39d0f546aa270bbc20c6228465f5e6cba49e2e607e8b0aef14f9d1fe6226e705.jpg)

![3d1519a96a520ce02afccf0227425d6c29ce0e91777bb02d3376b89ed848181c.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/3d1519a96a520ce02afccf0227425d6c29ce0e91777bb02d3376b89ed848181c.jpg)

![41000a495fb05fd609ec407de884e4c7956c0cb7d7ef26421f11371a5b93e8c7.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/41000a495fb05fd609ec407de884e4c7956c0cb7d7ef26421f11371a5b93e8c7.jpg)

![42ee1244755403e60c778b098fbe1190e380250ac880280666d0978608426f6b.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/42ee1244755403e60c778b098fbe1190e380250ac880280666d0978608426f6b.jpg)

![4a614f2b9d272a09b35c9d0086cc75b4673cfa574ea9cb2bb6edbd6a5e8594a4.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/4a614f2b9d272a09b35c9d0086cc75b4673cfa574ea9cb2bb6edbd6a5e8594a4.jpg)

![5848a1bb885fa86ab8f9ddb4dd28e2439a848af6f4d473651f024907faefbc94.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/5848a1bb885fa86ab8f9ddb4dd28e2439a848af6f4d473651f024907faefbc94.jpg)

![59a630c5fb1fbe5097e981c49538786d821ca6962e0483c8cb6eb61958df5381.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/59a630c5fb1fbe5097e981c49538786d821ca6962e0483c8cb6eb61958df5381.jpg)

![5aa0ef57b916fc1a9db5e2065135a3ee8a5084db748bbf9b9c68d1f258f8cc80.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/5aa0ef57b916fc1a9db5e2065135a3ee8a5084db748bbf9b9c68d1f258f8cc80.jpg)

![5c2b36640a6b8054cb158588d5f748f49e216f8cb9f18148735c724e8ea15fb2.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/5c2b36640a6b8054cb158588d5f748f49e216f8cb9f18148735c724e8ea15fb2.jpg)

![62224cd346041628533a908ecde7608d3fb2eb659f803590f4551092ab7f36d5.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/62224cd346041628533a908ecde7608d3fb2eb659f803590f4551092ab7f36d5.jpg)

![68b824d14d358623d9bc57c71ff3f6c3e020392c1eae6b9fedf98bea480cd5a2.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/68b824d14d358623d9bc57c71ff3f6c3e020392c1eae6b9fedf98bea480cd5a2.jpg)

![70961fa3d84119ca7631aeac639c3c53b24c98a9b86f9c7eb2fe69875255e1a9.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/70961fa3d84119ca7631aeac639c3c53b24c98a9b86f9c7eb2fe69875255e1a9.jpg)

![7c2ddc937aca6248471ed01d9c3a8a310b267692ec61526a581e3f605367ba7a.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/7c2ddc937aca6248471ed01d9c3a8a310b267692ec61526a581e3f605367ba7a.jpg)

![a269c6cd7231ccce4eb54ef7dbdae79fbbc6cde541aaa3656d1da7d03d9901c3.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/a269c6cd7231ccce4eb54ef7dbdae79fbbc6cde541aaa3656d1da7d03d9901c3.jpg)

![a68b33a8d119fcfdb815f1a198846d5cb04595bed0bf68ed6e578b273ed1386f.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/a68b33a8d119fcfdb815f1a198846d5cb04595bed0bf68ed6e578b273ed1386f.jpg)

![ada8f0616611ef90a0962f59aa3bbff3eb5e7b0a7d063250295318861adf11b1.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/ada8f0616611ef90a0962f59aa3bbff3eb5e7b0a7d063250295318861adf11b1.jpg)

![c222ee22b575194bd33eafcfa88d036bb32fc10a12b8a0a45d8aadd6dbf90f06.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/c222ee22b575194bd33eafcfa88d036bb32fc10a12b8a0a45d8aadd6dbf90f06.jpg)

![cbea2cb2549fb02981b42b548c46a8979519fcc90a1b12b48e283ae91f139f6c.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/cbea2cb2549fb02981b42b548c46a8979519fcc90a1b12b48e283ae91f139f6c.jpg)

![d2428f226d0191182077bd6130ccd08df8528eec56ad516fc62223c101a2f464.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/d2428f226d0191182077bd6130ccd08df8528eec56ad516fc62223c101a2f464.jpg)

![e5d4aa65e7f4eeb4b208460a2f043e0ed6ee06fd7209cac80415edb49fcb5813.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/e5d4aa65e7f4eeb4b208460a2f043e0ed6ee06fd7209cac80415edb49fcb5813.jpg)

![f29131e404ffb4214dc538a6b33328dc2c797e264166a8b3e542f6979abcea70.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/f29131e404ffb4214dc538a6b33328dc2c797e264166a8b3e542f6979abcea70.jpg)

![fe3ffed87320661c91eb90c665a948c8a6741d0c1e8d6fdf7cebea373eeb7016.jpg](../icml_results/2120_ProEval_%20Proactive%20Failure%20Discovery%20and%20Efficient%20Performance%20Estimation%20for%20Generative%20AI%20Evaluati/tables/fe3ffed87320661c91eb90c665a948c8a6741d0c1e8d6fdf7cebea373eeb7016.jpg)

<a id="paper-2093"></a>

## Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models


### Images

![09e8bb1851cba7e303b8f895b87a76377b06eb083702e2026ff8966c08bf7c70.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/images/09e8bb1851cba7e303b8f895b87a76377b06eb083702e2026ff8966c08bf7c70.jpg)

![7dae2849dcdde4972d410e71e621f578845639caeef4e5f2e6bb55af482d85f1.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/images/7dae2849dcdde4972d410e71e621f578845639caeef4e5f2e6bb55af482d85f1.jpg)

![c390e7f4748567129474eedd6cfa2c60fa0036d452e4ee3bc3256996aa0815a3.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/images/c390e7f4748567129474eedd6cfa2c60fa0036d452e4ee3bc3256996aa0815a3.jpg)

![dc9c09d67dada350fc332d38e16952577115c998bf7ce972399b2f552f04c7d8.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/images/dc9c09d67dada350fc332d38e16952577115c998bf7ce972399b2f552f04c7d8.jpg)

### Tables

![03a067f965a7490b171ceb4198e33a6432ac466baf268d1818f5e38c205c5ee4.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/03a067f965a7490b171ceb4198e33a6432ac466baf268d1818f5e38c205c5ee4.jpg)

![4f3f386fa72cfd3edb34a19e30d03d89fd4ec3c4ba0917ea200ec8ed40f9bca5.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/4f3f386fa72cfd3edb34a19e30d03d89fd4ec3c4ba0917ea200ec8ed40f9bca5.jpg)

![54f02018f7ba0beb0bb2d7b93114093346923d521d725cea34b9b8fa155b8e2a.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/54f02018f7ba0beb0bb2d7b93114093346923d521d725cea34b9b8fa155b8e2a.jpg)

![6b6bfc3e4430be3766c4301a6e087c9953384e33bfe300c161209b3fb9b56dcd.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/6b6bfc3e4430be3766c4301a6e087c9953384e33bfe300c161209b3fb9b56dcd.jpg)

![8bd05c476e59e3550c84febf3dcbaffda2918309fe8cb265c93ed91710743933.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/8bd05c476e59e3550c84febf3dcbaffda2918309fe8cb265c93ed91710743933.jpg)

![9c6182349a819c122443794215a023d5a8756517287d8706da494de983030e85.jpg](../icml_results/2121_Vision-DeepResearch_%20Incentivizing%20DeepResearch%20Capability%20in%20Multimodal%20Large%20Language%20Models/tables/9c6182349a819c122443794215a023d5a8756517287d8706da494de983030e85.jpg)

<a id="paper-2094"></a>

## MAS-Architect: Declarative Multi-Agent System Design via Separation of Concerns


### Images

![0a2fde77a3fa03e4af8d20346d75d720dfd35450b3e2bf03cb9aa31d51dda138.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/0a2fde77a3fa03e4af8d20346d75d720dfd35450b3e2bf03cb9aa31d51dda138.jpg)

![16639479ebd2f397882e851ffa929b6e864c480ec53629de5cc1c4fc8eb69462.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/16639479ebd2f397882e851ffa929b6e864c480ec53629de5cc1c4fc8eb69462.jpg)

![16ae7045059dea7547315dffbcd8352253dc1edf6166e62b7acc25787bb3a77d.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/16ae7045059dea7547315dffbcd8352253dc1edf6166e62b7acc25787bb3a77d.jpg)

![2715c27a536834b4f9d8a7486a10ddd7caee66fd98ebac9e39b3336b5cf5cdbe.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/2715c27a536834b4f9d8a7486a10ddd7caee66fd98ebac9e39b3336b5cf5cdbe.jpg)

![2c02ff45a9c241ad1648066d1d4fc7ee26874cff1b0ae4dc26825de2242c07ba.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/2c02ff45a9c241ad1648066d1d4fc7ee26874cff1b0ae4dc26825de2242c07ba.jpg)

![3dc20e2d54e3f1ffa46620d0f839702f2f0887b9087c131f92438c2869f5705f.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/3dc20e2d54e3f1ffa46620d0f839702f2f0887b9087c131f92438c2869f5705f.jpg)

![4669b70f56830fac7c7a7108d6bb92d8db15e486beaf63d1fbba0953b26c9005.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/4669b70f56830fac7c7a7108d6bb92d8db15e486beaf63d1fbba0953b26c9005.jpg)

![478e5134cba11e7cd6b830508a953af7babe2e59ea93b9464242113716bdcaec.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/478e5134cba11e7cd6b830508a953af7babe2e59ea93b9464242113716bdcaec.jpg)

![52dcc1ead45d56263d48493c3adf75339285795c0f778ffb8b54f7d2c3f17d8e.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/52dcc1ead45d56263d48493c3adf75339285795c0f778ffb8b54f7d2c3f17d8e.jpg)

![5e3d85932912f446a86a2a3dad4c1755de7c1c08c01d2dec2dc8ad2bb0bb4292.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/5e3d85932912f446a86a2a3dad4c1755de7c1c08c01d2dec2dc8ad2bb0bb4292.jpg)

![6117fee2737026e2f84584aa27c87521ccdedb966e8c3dc219d6de9e4ac9e57f.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/6117fee2737026e2f84584aa27c87521ccdedb966e8c3dc219d6de9e4ac9e57f.jpg)

![6dc53cd942bc0bb243da75043fe3949683577746096c63d8ed0a5e5871454e15.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/6dc53cd942bc0bb243da75043fe3949683577746096c63d8ed0a5e5871454e15.jpg)

![8a627b468c722dde47874ef646f07bc17a1e9db9c55e3423ff5f26e698d8558f.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/8a627b468c722dde47874ef646f07bc17a1e9db9c55e3423ff5f26e698d8558f.jpg)

![962ab19f2c56e32437b27e6f2716d048f4d8f02ff84981ffb25b75464f003d1d.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/962ab19f2c56e32437b27e6f2716d048f4d8f02ff84981ffb25b75464f003d1d.jpg)

![96cd5bf83e30d34d675fc0cf3d8a00c58876d3c5d5e00d16d0b2a112e24d9545.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/96cd5bf83e30d34d675fc0cf3d8a00c58876d3c5d5e00d16d0b2a112e24d9545.jpg)

![a30c9238d00b32806fcba82c04e68c46bc423c3ec6710b929b971d0900cf5ac3.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/a30c9238d00b32806fcba82c04e68c46bc423c3ec6710b929b971d0900cf5ac3.jpg)

![be41deb04150f611390cb49663e2768e5742a6c205b038eb36e7599b93022902.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/be41deb04150f611390cb49663e2768e5742a6c205b038eb36e7599b93022902.jpg)

![d5b1400f1ff607a431fdc4fe372db5a1c65488d9264af247991110523aa47d11.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/d5b1400f1ff607a431fdc4fe372db5a1c65488d9264af247991110523aa47d11.jpg)

![f7224c79fe15be7c619ed9af73639788d54be9295e900506c3f6c7717111fd66.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/images/f7224c79fe15be7c619ed9af73639788d54be9295e900506c3f6c7717111fd66.jpg)

### Tables

![5c37967c0bbcad7b017757cc42d937ea2a758acfc4d80336f3a7ca3003c2a883.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/5c37967c0bbcad7b017757cc42d937ea2a758acfc4d80336f3a7ca3003c2a883.jpg)

![78d1b73aa0108f7e27cebbbfb73ff89267ca5f78984199f76f8e32f319ff2dd0.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/78d1b73aa0108f7e27cebbbfb73ff89267ca5f78984199f76f8e32f319ff2dd0.jpg)

![9fc62fefc74ceb83c8ce3681c3b25880743ef5649ecbdb27fcb80bf401894517.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/9fc62fefc74ceb83c8ce3681c3b25880743ef5649ecbdb27fcb80bf401894517.jpg)

![a311560ab947ba1b0807d0a09202a64192017cb24be5fab8b6463bef14d07c66.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/a311560ab947ba1b0807d0a09202a64192017cb24be5fab8b6463bef14d07c66.jpg)

![adbbafe13e2d7c3cdb1f299ec81f423f56e79fb60c108cd831ed087fc77fd7c3.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/adbbafe13e2d7c3cdb1f299ec81f423f56e79fb60c108cd831ed087fc77fd7c3.jpg)

![ff4ff8923726aab6c2390b026518d313341a3cb2deae7523876bb80c4b80340f.jpg](../icml_results/2122_MAS-Architect_%20Declarative%20Multi-Agent%20System%20Design%20via%20Separation%20of%20Concerns/tables/ff4ff8923726aab6c2390b026518d313341a3cb2deae7523876bb80c4b80340f.jpg)

<a id="paper-2095"></a>

## Optimal Transport–Guided Stochastic Control for Graph Combinatorial Optimization


### Images

![a9583eb949be48077792be30489d2b808e42f2da06c81564592fb2221a5e50be.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/images/a9583eb949be48077792be30489d2b808e42f2da06c81564592fb2221a5e50be.jpg)

![eb7e6d6d2231283e70262c9b6586f3bb09ab19e27b6e5b657879470ea9d3f411.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/images/eb7e6d6d2231283e70262c9b6586f3bb09ab19e27b6e5b657879470ea9d3f411.jpg)

![f94f4e496a4457d55fe2c9f7b821ddbcfa5d943e8b8f9f1e0f854df296f59425.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/images/f94f4e496a4457d55fe2c9f7b821ddbcfa5d943e8b8f9f1e0f854df296f59425.jpg)

### Tables

![0a1934edd4ba582ea8ef03e5670a412c5c0d0bf212e801b1831e15c247143c11.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/tables/0a1934edd4ba582ea8ef03e5670a412c5c0d0bf212e801b1831e15c247143c11.jpg)

![0d43e1246dd18bcdfa1b04c36a0b9a38973f16d9ec772d85778fe2c0c4ed63ad.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/tables/0d43e1246dd18bcdfa1b04c36a0b9a38973f16d9ec772d85778fe2c0c4ed63ad.jpg)

![88dcb98f4b20537c1d896765a83de2d5a68b983cf0cbcc6d8feded11bad5ec2b.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/tables/88dcb98f4b20537c1d896765a83de2d5a68b983cf0cbcc6d8feded11bad5ec2b.jpg)

![b9b621f57ef99f7699f73ddf4e7f1586340c020ddb68324fc851e52ea856a894.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/tables/b9b621f57ef99f7699f73ddf4e7f1586340c020ddb68324fc851e52ea856a894.jpg)

![df01a1ab49a180ca3008d901b96f7e26a4ba198e162742d742924ad3869ff35c.jpg](../icml_results/2123_Optimal%20Transport%E2%80%93Guided%20Stochastic%20Control%20for%20Graph%20Combinatorial%20Optimization/tables/df01a1ab49a180ca3008d901b96f7e26a4ba198e162742d742924ad3869ff35c.jpg)

<a id="paper-2096"></a>

## Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs


### Images

![1c4de76f52648b558e04a972113122a5d709d9825b6f6bef07ebff96f371604b.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/1c4de76f52648b558e04a972113122a5d709d9825b6f6bef07ebff96f371604b.jpg)

![29a52ae75bdd825601d6f4013cf4221f30826a3b02a2126c77dbe9306dffa240.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/29a52ae75bdd825601d6f4013cf4221f30826a3b02a2126c77dbe9306dffa240.jpg)

![405ce40a885dbe42057101bdb15835edfca04f0e7ca7809784e63205ff6f015d.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/405ce40a885dbe42057101bdb15835edfca04f0e7ca7809784e63205ff6f015d.jpg)

![54ae5634ba1cf8318c8a4af50b3220820258c740ffcd4b89c53ad1299124584c.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/54ae5634ba1cf8318c8a4af50b3220820258c740ffcd4b89c53ad1299124584c.jpg)

![5fa2fb1c877ffa796c3263bb816a8dffb4995c847f5a3e80d727f6861d1d5114.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/5fa2fb1c877ffa796c3263bb816a8dffb4995c847f5a3e80d727f6861d1d5114.jpg)

![84191825f6c849268bf66e5054a5220ea7b5ef21fe4e387d42a788570bd59c37.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/84191825f6c849268bf66e5054a5220ea7b5ef21fe4e387d42a788570bd59c37.jpg)

![8df5a777a9af48b51e9a1b41312b31346cb47c45d6899d77d18f84de8895945b.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/8df5a777a9af48b51e9a1b41312b31346cb47c45d6899d77d18f84de8895945b.jpg)

![9adde42646ec4d76b074ea5312d1a4655b1e1d709b8d8828db9fb523fac0a2d5.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/9adde42646ec4d76b074ea5312d1a4655b1e1d709b8d8828db9fb523fac0a2d5.jpg)

![af55a78ca9ed89e50b4b600ffd08538fafd8742dc13f00dc9aa3f8863c1739a8.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/af55a78ca9ed89e50b4b600ffd08538fafd8742dc13f00dc9aa3f8863c1739a8.jpg)

![b807b4073887670fffe966d9922693582bb6511f7a14f3bdc9bb51858494d610.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/b807b4073887670fffe966d9922693582bb6511f7a14f3bdc9bb51858494d610.jpg)

![c5abcaa53c945ccca4f40d0b048ea8d250f64c53f516cf455787b83f2f9ab24e.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/c5abcaa53c945ccca4f40d0b048ea8d250f64c53f516cf455787b83f2f9ab24e.jpg)

![c90b9e73ccb0e2accb8dae1a19fb14d8855ec9d7bd4915fcaf0b3693a501af23.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/c90b9e73ccb0e2accb8dae1a19fb14d8855ec9d7bd4915fcaf0b3693a501af23.jpg)

![d20b77979d982e4d7fdb2078557a69a3ed55f8f7a1ee67ab7af9bacd6b98205e.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/d20b77979d982e4d7fdb2078557a69a3ed55f8f7a1ee67ab7af9bacd6b98205e.jpg)

![d96bfe202fd0a771346094f1278bbc2c7bea7b4a563bc14b72bfd567e64aa4fe.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/d96bfe202fd0a771346094f1278bbc2c7bea7b4a563bc14b72bfd567e64aa4fe.jpg)

![f7c5b8233ea05159d808044f452f2f7d71440ecf386acbb4e533e5b9fc4e4781.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/f7c5b8233ea05159d808044f452f2f7d71440ecf386acbb4e533e5b9fc4e4781.jpg)

![fab634b9ca257528a1810a83249596ac71300ad644289752a7ccc442072e0fe0.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/images/fab634b9ca257528a1810a83249596ac71300ad644289752a7ccc442072e0fe0.jpg)

### Tables

![006be051705d10edad0cfe0709535223af28d2abaaaee8c784b350933fce0e93.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/tables/006be051705d10edad0cfe0709535223af28d2abaaaee8c784b350933fce0e93.jpg)

![19a4f438e527c25e62e992e3479eaa8b3a3d65f2872f8c2f9026fa163e903a5b.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/tables/19a4f438e527c25e62e992e3479eaa8b3a3d65f2872f8c2f9026fa163e903a5b.jpg)

![1cc4351bfef9998402835bec85f6f1e06e9c0a76ad57c31d57a35564ba3e19cc.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/tables/1cc4351bfef9998402835bec85f6f1e06e9c0a76ad57c31d57a35564ba3e19cc.jpg)

![88d5b953285b13f05042a2d5296f949fc7888ce7387dc58ad901ce6e347dd78c.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/tables/88d5b953285b13f05042a2d5296f949fc7888ce7387dc58ad901ce6e347dd78c.jpg)

![b4cea9546613f0ccbefe45b836c371d161ae618e72887963c1bcb3e164eb8500.jpg](../icml_results/2124_Stable%20Asynchrony_%20Variance-Controlled%20Off-Policy%20RL%20for%20LLMs/tables/b4cea9546613f0ccbefe45b836c371d161ae618e72887963c1bcb3e164eb8500.jpg)

<a id="paper-2097"></a>

## AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture -of-Transformers for End-to-End Autonomous Driving


### Images

![59cffec1bc96e669aff746fd2f0a149ed56efcb29dce0fb1c2745c2112640a09.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/images/59cffec1bc96e669aff746fd2f0a149ed56efcb29dce0fb1c2745c2112640a09.jpg)

![5be3172b68935614a786e8460fe234518f6e8491deef461674d385c15f81252f.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/images/5be3172b68935614a786e8460fe234518f6e8491deef461674d385c15f81252f.jpg)

![5be9eedaed14d8bd2665455d07720884f2ff5ee62258ac053f24226cb52422fa.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/images/5be9eedaed14d8bd2665455d07720884f2ff5ee62258ac053f24226cb52422fa.jpg)

![a6c1ba6ae4fb78a2ade765a27d7cdd356e343d3f4ea019d4dbd9b5768f631cea.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/images/a6c1ba6ae4fb78a2ade765a27d7cdd356e343d3f4ea019d4dbd9b5768f631cea.jpg)

![c0910d4daa26cd28669264defe94f74a1e54159cd45c117f7bdd9bc958afb5c9.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/images/c0910d4daa26cd28669264defe94f74a1e54159cd45c117f7bdd9bc958afb5c9.jpg)

### Tables

![0124450927bc51d375b6b836632247bd0f9ed1324db0b6db8e22a349657851e0.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/0124450927bc51d375b6b836632247bd0f9ed1324db0b6db8e22a349657851e0.jpg)

![0134ec19c5dca6e39edc10236cc74009446d211717c6c71c77721d21b6ba9d1b.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/0134ec19c5dca6e39edc10236cc74009446d211717c6c71c77721d21b6ba9d1b.jpg)

![09428d268ee329fd89df6a7a3dcacbd9c9d1aad007194eaf4b16f561efa76676.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/09428d268ee329fd89df6a7a3dcacbd9c9d1aad007194eaf4b16f561efa76676.jpg)

![171add707be9ec6c845c778f669509ad53b017b9818de00c3e4d2f64926ba786.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/171add707be9ec6c845c778f669509ad53b017b9818de00c3e4d2f64926ba786.jpg)

![a032636c6c5825f9f93ad6c0771146ca5488eac9b5887fe9ce08c0f22ed548ef.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/a032636c6c5825f9f93ad6c0771146ca5488eac9b5887fe9ce08c0f22ed548ef.jpg)

![aaded39905eb758eb5af2fc70ee8cfc9df94caa3fa90462f67f72b9b044d229c.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/aaded39905eb758eb5af2fc70ee8cfc9df94caa3fa90462f67f72b9b044d229c.jpg)

![aecec9ced17ac668b1a900beac4c4954b38b3925d22ed30521e11582a189cd76.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/aecec9ced17ac668b1a900beac4c4954b38b3925d22ed30521e11582a189cd76.jpg)

![d9efac1191b34c7c5691ed0586bdaa12a994bea05dc543f26e5a8e71bfe13951.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/d9efac1191b34c7c5691ed0586bdaa12a994bea05dc543f26e5a8e71bfe13951.jpg)

![dde34c4523715aa955a3ede2f8c5e6afa3a4a5580643c9c5101aefd3fcbe5080.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/dde34c4523715aa955a3ede2f8c5e6afa3a4a5580643c9c5101aefd3fcbe5080.jpg)

![e0307da24fdcfda81591656f2239eaeb6fa0b79a73e5e83b8e53aae49688bee9.jpg](../icml_results/2125_AutoMoT_%20A%20Unified%20Vision-Language-Action%20Model%20with%20Asynchronous%20Mixture%20-of-Transformers%20for%20End-t/tables/e0307da24fdcfda81591656f2239eaeb6fa0b79a73e5e83b8e53aae49688bee9.jpg)

<a id="paper-2098"></a>

## Prototype-Based Test-Time Adaptation of Vision-Language Models


### Images

![13fe783cc0cda18bfe1ec79ead6082ed24c26680f7c4e19921fdd154349c52cc.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/images/13fe783cc0cda18bfe1ec79ead6082ed24c26680f7c4e19921fdd154349c52cc.jpg)

![3ec3fe1a85c7ff7d9213c3f145dadc1305df3bd4d9bcb2dd3ae7873770a9de1a.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/images/3ec3fe1a85c7ff7d9213c3f145dadc1305df3bd4d9bcb2dd3ae7873770a9de1a.jpg)

![5e6827cd4dc16621968d15bb3f450af47f3a4088dd4e262c808f379f93fbc745.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/images/5e6827cd4dc16621968d15bb3f450af47f3a4088dd4e262c808f379f93fbc745.jpg)

![656be9a099f1103631a4e93821491853ad9c8eb4a4c54ff13ba738e09d1ad232.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/images/656be9a099f1103631a4e93821491853ad9c8eb4a4c54ff13ba738e09d1ad232.jpg)

![867e14a60280ee229b8c92dcd623fa1d060739b5e0c0f3ca69c125d5a8953adf.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/images/867e14a60280ee229b8c92dcd623fa1d060739b5e0c0f3ca69c125d5a8953adf.jpg)

### Tables

![0d99ebff25538a2ab75919f1b3af6970c43d6a3772586f370dd17a01209fb7cd.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/0d99ebff25538a2ab75919f1b3af6970c43d6a3772586f370dd17a01209fb7cd.jpg)

![11cfa6d033fc23ba5165b4fbe944bebcf11361609a5f668c51eeb42f35f19aad.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/11cfa6d033fc23ba5165b4fbe944bebcf11361609a5f668c51eeb42f35f19aad.jpg)

![1f19fb869523766e596f4950367c41ec7394af3dc6ad2681e44d52024f391fcc.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/1f19fb869523766e596f4950367c41ec7394af3dc6ad2681e44d52024f391fcc.jpg)

![2d134fb1d78c3721008e6545b14bbd2c16cb0fdec668af78417a28a51999b814.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/2d134fb1d78c3721008e6545b14bbd2c16cb0fdec668af78417a28a51999b814.jpg)

![5811bee5cfac07a4fbcb0211819fe3f09ce17b9188ecb33664e1d88f450135fc.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/5811bee5cfac07a4fbcb0211819fe3f09ce17b9188ecb33664e1d88f450135fc.jpg)

![8ae70880e193dca140205e85670e618baacea57f2ea7f71fa4a8270b929f6943.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/8ae70880e193dca140205e85670e618baacea57f2ea7f71fa4a8270b929f6943.jpg)

![9d43c78fc5cba9e79ab34078294d96a6e375170a08c4bdea10e5f7e318752e71.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/9d43c78fc5cba9e79ab34078294d96a6e375170a08c4bdea10e5f7e318752e71.jpg)

![b7492dc5d6a5bccd4af22ef116447ee636a3ee9b42c46736f6b3e1ab945df6c2.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/b7492dc5d6a5bccd4af22ef116447ee636a3ee9b42c46736f6b3e1ab945df6c2.jpg)

![bb7417de439d76d10bda0b28509f417a3b718330cd23390ff18b112edd4cce02.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/bb7417de439d76d10bda0b28509f417a3b718330cd23390ff18b112edd4cce02.jpg)

![dc86983cc69ac6404efd3391687794f8cf4d9abc188b632afbb194a6e68ad4e7.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/dc86983cc69ac6404efd3391687794f8cf4d9abc188b632afbb194a6e68ad4e7.jpg)

![dd0904453523442be2832888cdfb44a14cc62f492dbce2463a00d19823403e4d.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/dd0904453523442be2832888cdfb44a14cc62f492dbce2463a00d19823403e4d.jpg)

![deed8cd9c0fb0452c50068897bffd4826670b4b7cc757b02da62e2f7f3714234.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/deed8cd9c0fb0452c50068897bffd4826670b4b7cc757b02da62e2f7f3714234.jpg)

![df80fdaae121a3058a7bdbdf18f715c8056bc8722005ef8854620f63d73f449c.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/df80fdaae121a3058a7bdbdf18f715c8056bc8722005ef8854620f63d73f449c.jpg)

![dfb4ce07d32e92dac0a0920085abb0856144f1c99e0c42626466dda4c9abae9a.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/dfb4ce07d32e92dac0a0920085abb0856144f1c99e0c42626466dda4c9abae9a.jpg)

![e83b42333605a55e7093c257a67d0882218bc72f228e036dd15dd10507a768a8.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/e83b42333605a55e7093c257a67d0882218bc72f228e036dd15dd10507a768a8.jpg)

![e98285a06ca5e7c029a0f1eda86af68084e6b9f432a3e0fe3651d24bcf7011df.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/e98285a06ca5e7c029a0f1eda86af68084e6b9f432a3e0fe3651d24bcf7011df.jpg)

![eebe2dcb919052bd90ef063398e76fc572522d9ae6f939130794d42d0c8f031d.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/eebe2dcb919052bd90ef063398e76fc572522d9ae6f939130794d42d0c8f031d.jpg)

![f5b1c7ec3098a07340305cc0815e55b609b01136acc562355a6d243e6b676d51.jpg](../icml_results/2126_Prototype-Based%20Test-Time%20Adaptation%20of%20Vision-Language%20Models/tables/f5b1c7ec3098a07340305cc0815e55b609b01136acc562355a6d243e6b676d51.jpg)

<a id="paper-2099"></a>

## TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments


### Images

![0d6b162f2eb2ccdc72fb2597be536c89566524ae91a829e86a45d98f45a25ca8.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/0d6b162f2eb2ccdc72fb2597be536c89566524ae91a829e86a45d98f45a25ca8.jpg)

![137f3e9d955abbcb73b7866426bf723a2f85af1bed5f1735b88f64d0cee1f1f8.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/137f3e9d955abbcb73b7866426bf723a2f85af1bed5f1735b88f64d0cee1f1f8.jpg)

![407262a8085ce7025daa9253b03872586d74cf4f12300580261c15368ae36d84.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/407262a8085ce7025daa9253b03872586d74cf4f12300580261c15368ae36d84.jpg)

![4f4d2986cc83402ba349607fd2a2762e0281088d22be410c25280026c1edee0d.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/4f4d2986cc83402ba349607fd2a2762e0281088d22be410c25280026c1edee0d.jpg)

![54c689b15022440ff982aa945e58c7c870c636799ee4a85fff55694577a9acfc.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/54c689b15022440ff982aa945e58c7c870c636799ee4a85fff55694577a9acfc.jpg)

![791837cd3ab7dcad9e2db361ad5e176517cfc4a50384fdbb74489a9a1c9b46f8.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/791837cd3ab7dcad9e2db361ad5e176517cfc4a50384fdbb74489a9a1c9b46f8.jpg)

![9b6226be69ba91cf6c63bf8a5cc47a474c0e53931dc5cadc69f01bcc1126ea44.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/9b6226be69ba91cf6c63bf8a5cc47a474c0e53931dc5cadc69f01bcc1126ea44.jpg)

![a1d4d27122db85487b31c72dbc0c15566e990e8416bd6975da618b224f22d50b.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/a1d4d27122db85487b31c72dbc0c15566e990e8416bd6975da618b224f22d50b.jpg)

![a6b57fe0a33c6b1292c7fc2247282d4409839965c599c3c8dbbd4ae4a9041a0d.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/a6b57fe0a33c6b1292c7fc2247282d4409839965c599c3c8dbbd4ae4a9041a0d.jpg)

![d27a42d3fc45a21216b3606046cd87be4037f4bc1359b4ad17dfe79c3c3c5412.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/d27a42d3fc45a21216b3606046cd87be4037f4bc1359b4ad17dfe79c3c3c5412.jpg)

![f2c8cb5124dd382cb743f554d5669677bfcfc807899a404bf21c548b8bc55e27.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/f2c8cb5124dd382cb743f554d5669677bfcfc807899a404bf21c548b8bc55e27.jpg)

![fa98ed41aac286410614825905845c04b83f57d039bb3580f6967db4dae87b23.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/images/fa98ed41aac286410614825905845c04b83f57d039bb3580f6967db4dae87b23.jpg)

### Tables

![08ac7dddacee0b0b77b2c6c7b58179867bfcad5e621c834b1130633b164d715a.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/08ac7dddacee0b0b77b2c6c7b58179867bfcad5e621c834b1130633b164d715a.jpg)

![298b32a92c0d703521575264b1d7ac61f854c3544311d7627711055604361ab8.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/298b32a92c0d703521575264b1d7ac61f854c3544311d7627711055604361ab8.jpg)

![2e4eee4e0a206b9aeabdeeb70bddb7b75a82f1fb85a7b94aa6679dae18584e32.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/2e4eee4e0a206b9aeabdeeb70bddb7b75a82f1fb85a7b94aa6679dae18584e32.jpg)

![357fc80b4f786b676077a314cc8a3aa47851bbdb6444d1d03d67e1f3dd2b671b.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/357fc80b4f786b676077a314cc8a3aa47851bbdb6444d1d03d67e1f3dd2b671b.jpg)

![3d094a10aa4bd38f7bbacebdabad39149dd0ea2b11fdfa3ac580aea77742a399.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/3d094a10aa4bd38f7bbacebdabad39149dd0ea2b11fdfa3ac580aea77742a399.jpg)

![407481ec950b03dc7db5336943dc915eeb365a35a2739f18664364c0d8f59ec9.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/407481ec950b03dc7db5336943dc915eeb365a35a2739f18664364c0d8f59ec9.jpg)

![7005e52bfa55b5e255255f32eca3a6b0e761fe6005a231ad6885c0bb5152bf74.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/7005e52bfa55b5e255255f32eca3a6b0e761fe6005a231ad6885c0bb5152bf74.jpg)

![8c8a7f62d6a890ca1b343ea4bb9b70d776fd81638e7681b36679a0ff68800c25.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/8c8a7f62d6a890ca1b343ea4bb9b70d776fd81638e7681b36679a0ff68800c25.jpg)

![968fa87551d6a63242b09b90e57a9c884f389c6f82355122f11642e59daba3a7.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/968fa87551d6a63242b09b90e57a9c884f389c6f82355122f11642e59daba3a7.jpg)

![9c4028a40d126ac7f3242e54637f4dea18169e6ca5a8c9d32e80ce847e81a976.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/9c4028a40d126ac7f3242e54637f4dea18169e6ca5a8c9d32e80ce847e81a976.jpg)

![a921e92e6400be31383c5889103af0142ba48f47f6231eaaddf2646ea6f7dc5b.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/a921e92e6400be31383c5889103af0142ba48f47f6231eaaddf2646ea6f7dc5b.jpg)

![bfe41c1a2a81140446206f29888a7b09a6e29f4ef7e98c42efb11b4214b54bcd.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/bfe41c1a2a81140446206f29888a7b09a6e29f4ef7e98c42efb11b4214b54bcd.jpg)

![d39d0e8ff4b0d545fe2c1de148b41cd53c3c54aa3c0ccbdf7f0083e60c31ed0a.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/d39d0e8ff4b0d545fe2c1de148b41cd53c3c54aa3c0ccbdf7f0083e60c31ed0a.jpg)

![dfbfbabc6e902b1def0cdd0169b1831068a0a8448ba2e67556f159ef7fad9f20.jpg](../icml_results/2127_TIC-VLA_%20A%20Think-in-Control%20Vision-Language-Action%20Model%20for%20Robot%20Navigation%20in%20Dynamic%20Environment/tables/dfbfbabc6e902b1def0cdd0169b1831068a0a8448ba2e67556f159ef7fad9f20.jpg)

<a id="paper-2100"></a>

## Continuous Variable Hamiltonian Learning at Heisenberg Limit via Displacement-Random Unitary Transformation


### Images

![6695cb9d2e7be2f72057439c947c0abc6181bc31505016bd9366e9f3e5fd6b75.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/images/6695cb9d2e7be2f72057439c947c0abc6181bc31505016bd9366e9f3e5fd6b75.jpg)

![67befa2176713c11154eee035fe23175f8b3ac707659f68b46e46c00101a4d86.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/images/67befa2176713c11154eee035fe23175f8b3ac707659f68b46e46c00101a4d86.jpg)

![78dc291956aa14877d5f54603ba3b7f2fd2b768b995993a1ef58a5544ca369b1.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/images/78dc291956aa14877d5f54603ba3b7f2fd2b768b995993a1ef58a5544ca369b1.jpg)

![d286ba3a1a6520cbcc317bc99b5c457836f3ff754c0d2d076577a39d0e7c282c.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/images/d286ba3a1a6520cbcc317bc99b5c457836f3ff754c0d2d076577a39d0e7c282c.jpg)

![dee03449acfa040e5c2e258275054f8d6c1d06d63334374aa8f4be0eec1818e2.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/images/dee03449acfa040e5c2e258275054f8d6c1d06d63334374aa8f4be0eec1818e2.jpg)

### Tables

![22d14a21f2c49b23916bc68138e12467c286ccfe1db29aa00b402d403b7bbac9.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/tables/22d14a21f2c49b23916bc68138e12467c286ccfe1db29aa00b402d403b7bbac9.jpg)

![b607d642780d6ca28f83e28576e49ab19ad4159dab37c1b5cdc7c81230f2fa43.jpg](../icml_results/2128_Continuous%20Variable%20Hamiltonian%20Learning%20at%20Heisenberg%20Limit%20via%20Displacement-Random%20Unitary%20Transfo/tables/b607d642780d6ca28f83e28576e49ab19ad4159dab37c1b5cdc7c81230f2fa43.jpg)

<a id="paper-2101"></a>

## How Far Can LLM Agents Reason with Tables? Benchmarking Multi-Turn Agentic Table Question Answering in the Wild


### Images

![050b8fc4c40e28c9c0bf70dee47de00ed3fe8b855c97468a8c00e54918e3883d.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/050b8fc4c40e28c9c0bf70dee47de00ed3fe8b855c97468a8c00e54918e3883d.jpg)

![19c7d92372abbb83a7decd8a7f626868f66d55c0a057fd55f1c4561b1086c7f8.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/19c7d92372abbb83a7decd8a7f626868f66d55c0a057fd55f1c4561b1086c7f8.jpg)

![5cde56c22bc45881e5d9b55ef17a0dcea78b3e5fa604520d51ae4d619f1a0541.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/5cde56c22bc45881e5d9b55ef17a0dcea78b3e5fa604520d51ae4d619f1a0541.jpg)

![626ce634f5dcb73a9f3fe942cc162205ec5b567f723f8b0900eda01b8b21fa34.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/626ce634f5dcb73a9f3fe942cc162205ec5b567f723f8b0900eda01b8b21fa34.jpg)

![72f23277611dc4a15deb002e10d09de430860e4ea40d57346c74fa1708cbeed2.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/72f23277611dc4a15deb002e10d09de430860e4ea40d57346c74fa1708cbeed2.jpg)

![803c47315288358e32da45495206c4d36e6b221838867bdebc7e6e5091d00283.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/803c47315288358e32da45495206c4d36e6b221838867bdebc7e6e5091d00283.jpg)

![945d257d237a0597cd26246732c2d32657b8c91e285de8fb1fab0a4e552c11f2.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/945d257d237a0597cd26246732c2d32657b8c91e285de8fb1fab0a4e552c11f2.jpg)

![94e0dd932ec1c95e41964134a863fde5085626e23363be92fded4dc4c0ff9725.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/94e0dd932ec1c95e41964134a863fde5085626e23363be92fded4dc4c0ff9725.jpg)

![b8fa5955a76928070a102fe3907d6324011a2e3a75e408f0718722341eb579e3.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/images/b8fa5955a76928070a102fe3907d6324011a2e3a75e408f0718722341eb579e3.jpg)

### Tables

![52cc7756df919a1f4da84d30a4d11805d2c99a093739c0bda363c616b0920602.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/52cc7756df919a1f4da84d30a4d11805d2c99a093739c0bda363c616b0920602.jpg)

![605bb7b2fde364651d0c666e605cfc2d3bb9e5d93cb384707f4dd580b2fdfe2c.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/605bb7b2fde364651d0c666e605cfc2d3bb9e5d93cb384707f4dd580b2fdfe2c.jpg)

![631107b075f7f5c00fc69d96023b675663944f5befee091cbb52b61140a45d12.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/631107b075f7f5c00fc69d96023b675663944f5befee091cbb52b61140a45d12.jpg)

![7c0ed2b0a9393070e4516ad0b71f28684c396aa6cb68b888332e6947fdb10a3e.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/7c0ed2b0a9393070e4516ad0b71f28684c396aa6cb68b888332e6947fdb10a3e.jpg)

![95c2b153d90139f83d5e39845c98494a06c8a9c39643ab2d812a285138ca0b88.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/95c2b153d90139f83d5e39845c98494a06c8a9c39643ab2d812a285138ca0b88.jpg)

![992b9dc5c2b5ef425be003078fb24a0f246523e2085a5cb75e0a5ed514d79226.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/992b9dc5c2b5ef425be003078fb24a0f246523e2085a5cb75e0a5ed514d79226.jpg)

![d9c1850e70929d29232a68d18bca88645736568d57df00db79044e67bf482041.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/d9c1850e70929d29232a68d18bca88645736568d57df00db79044e67bf482041.jpg)

![eb1e4a996d2b8663346e6ab4a2742eac7f18f4cd574bfc5d1a09ebcaa7c33a19.jpg](../icml_results/2129_How%20Far%20Can%20LLM%20Agents%20Reason%20with%20Tables_%20Benchmarking%20Multi-Turn%20Agentic%20Table%20Question%20Answering%20/tables/eb1e4a996d2b8663346e6ab4a2742eac7f18f4cd574bfc5d1a09ebcaa7c33a19.jpg)

<a id="paper-2102"></a>

## Deterministic Differentiable Structured Pruning for Large Language Models


### Images

![13aea9a0534a5edec0e79bc4a16b8d9685ad053116f00e8d7e3fbe91b38473a1.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/13aea9a0534a5edec0e79bc4a16b8d9685ad053116f00e8d7e3fbe91b38473a1.jpg)

![43c1e88e63153c0863c9a910680b9ed836dcea30d9b080c27b574cd22ab04710.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/43c1e88e63153c0863c9a910680b9ed836dcea30d9b080c27b574cd22ab04710.jpg)

![526a6401d231e11111b407ed32088863c6bc74342a6158451f4b11bfabf00d69.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/526a6401d231e11111b407ed32088863c6bc74342a6158451f4b11bfabf00d69.jpg)

![74263588b00c82648927f97a37494c659f15e76952b2d32163a7e7bc29e8ada6.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/74263588b00c82648927f97a37494c659f15e76952b2d32163a7e7bc29e8ada6.jpg)

![8103e11e959018988725a6cb4b65c2a54c01598e4fb9b534e7c05c76f22bcec3.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/8103e11e959018988725a6cb4b65c2a54c01598e4fb9b534e7c05c76f22bcec3.jpg)

![8ddfcdc86605205187b1cb6e216f89c452638fba9e8f81aa814ae60582e6aad4.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/8ddfcdc86605205187b1cb6e216f89c452638fba9e8f81aa814ae60582e6aad4.jpg)

![a964932939bf16a10ed1472e93c3428a679409f0f6c05fb8128eedb7d1098bc0.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/a964932939bf16a10ed1472e93c3428a679409f0f6c05fb8128eedb7d1098bc0.jpg)

![b22dc3a1ccfc2937577dcdbf4540201319b991f1c91e77393f3a582dc68f2a5a.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/images/b22dc3a1ccfc2937577dcdbf4540201319b991f1c91e77393f3a582dc68f2a5a.jpg)

### Tables

![17e56407cab051d215196eb9fb0f10dee7da776ad589ccbf2cb4ce9fd7acdffd.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/17e56407cab051d215196eb9fb0f10dee7da776ad589ccbf2cb4ce9fd7acdffd.jpg)

![250cfba4c893e1e7fc9e7f783ee8fe3aebc717148c9bdc60775c2431b7d4e17f.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/250cfba4c893e1e7fc9e7f783ee8fe3aebc717148c9bdc60775c2431b7d4e17f.jpg)

![3b9ddfd649422d619971ecdb51fb5636d9987ff731e308e3d0d4e9e250be8a65.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/3b9ddfd649422d619971ecdb51fb5636d9987ff731e308e3d0d4e9e250be8a65.jpg)

![3c01d00ecd5151da967bb25b191c165cec832adc6c105f436687bd57bcd8b63e.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/3c01d00ecd5151da967bb25b191c165cec832adc6c105f436687bd57bcd8b63e.jpg)

![5a618165002c38436629d8965c1cb24edeebf4fc475a8b10b94258abbeb9cf95.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/5a618165002c38436629d8965c1cb24edeebf4fc475a8b10b94258abbeb9cf95.jpg)

![5f124a3a2a5e2006ca8b41a932841c12d3d4b3fe25e2387ec4a0d866e8c9f91a.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/5f124a3a2a5e2006ca8b41a932841c12d3d4b3fe25e2387ec4a0d866e8c9f91a.jpg)

![68417ddf5ba16a370024b4c50840bc4505182b00fd97530c1cb7fbdedcb39523.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/68417ddf5ba16a370024b4c50840bc4505182b00fd97530c1cb7fbdedcb39523.jpg)

![728f1b991fdc9b9b700dd5cb8b859426c937c08e96b00b41e149bd33b53854d8.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/728f1b991fdc9b9b700dd5cb8b859426c937c08e96b00b41e149bd33b53854d8.jpg)

![90f1c81148c1bc64b5f28462a4d0b3200a8578fce20ae46f7054a383b0e6bbe8.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/90f1c81148c1bc64b5f28462a4d0b3200a8578fce20ae46f7054a383b0e6bbe8.jpg)

![cb3bff067bc5c522e75c420963e5f334a46b9f71ca58e75f32af53ca5da995da.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/cb3bff067bc5c522e75c420963e5f334a46b9f71ca58e75f32af53ca5da995da.jpg)

![d852a540af822ecf2c3e470fe1e780ae6b7a0ca7e566f5cf928c66bb1ece447a.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/d852a540af822ecf2c3e470fe1e780ae6b7a0ca7e566f5cf928c66bb1ece447a.jpg)

![dd57e9a2beacd5ede0e9593123d7076847de3d0c94fd13bb55dd95ab724037f6.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/dd57e9a2beacd5ede0e9593123d7076847de3d0c94fd13bb55dd95ab724037f6.jpg)

![ee7867685d3259e9db6d74d0160e97e4fea864061463e5d073e4550ad0943145.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/ee7867685d3259e9db6d74d0160e97e4fea864061463e5d073e4550ad0943145.jpg)

![fcd04cf53d7b9ff3cc0db90b0b3783c85ec8a473bc4a251239d53373b81e26fb.jpg](../icml_results/2130_Deterministic%20Differentiable%20Structured%20Pruning%20for%20Large%20Language%20Models/tables/fcd04cf53d7b9ff3cc0db90b0b3783c85ec8a473bc4a251239d53373b81e26fb.jpg)

<a id="paper-2103"></a>

## RLCracker: Evaluating the Worst-Case Vulnerability of LLM Watermarks with Adaptive RL Attacks


### Images

![0d8ef000c41231b7ba7d14ecd74d209dcc0eb6f9230959c027b382da89a9c94f.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/0d8ef000c41231b7ba7d14ecd74d209dcc0eb6f9230959c027b382da89a9c94f.jpg)

![3dd911503fdfb9515e1e3ca481b2808bc5fd7271f70ce97fa7d2d04c75d62e54.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/3dd911503fdfb9515e1e3ca481b2808bc5fd7271f70ce97fa7d2d04c75d62e54.jpg)

![4633a35f42966b3b53ee7f4b84b7086e39663912b674cbd4704e41fdb5d22659.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/4633a35f42966b3b53ee7f4b84b7086e39663912b674cbd4704e41fdb5d22659.jpg)

![518b5af5272ba4b2ebd4c024856f826bf38b853b349c5bd1340bb4cf983f3530.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/518b5af5272ba4b2ebd4c024856f826bf38b853b349c5bd1340bb4cf983f3530.jpg)

![5495df55c4141191a598631074885cad3f28f30d4ed4dd013e65c3faf9c89459.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/5495df55c4141191a598631074885cad3f28f30d4ed4dd013e65c3faf9c89459.jpg)

![6d3e22f1b1bc470b29a0ccb37048c5f4ca3947c1244b5f5eabc84e0163b44c59.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/6d3e22f1b1bc470b29a0ccb37048c5f4ca3947c1244b5f5eabc84e0163b44c59.jpg)

![71d440713d1f2efe4d239e4b2d4381835c7c4ef8f6342156b3d99c12aa904ada.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/71d440713d1f2efe4d239e4b2d4381835c7c4ef8f6342156b3d99c12aa904ada.jpg)

![a0491097100d6f590538ee815187df56eeebe7e3d37678d02998acb82e0071fd.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/a0491097100d6f590538ee815187df56eeebe7e3d37678d02998acb82e0071fd.jpg)

![ca58cb7bf97d469d428f6b27652f3af8c44471839b51ebe0f37f04a617c20191.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/ca58cb7bf97d469d428f6b27652f3af8c44471839b51ebe0f37f04a617c20191.jpg)

![ed9d640efab21f0231e8aff92918f1eced137a4c18f7b47010be4b62cd406cdf.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/images/ed9d640efab21f0231e8aff92918f1eced137a4c18f7b47010be4b62cd406cdf.jpg)

### Tables

![141385d2fad8efbef00494d7297622a5c6a47bc487e1cfe62474f8c22a367629.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/141385d2fad8efbef00494d7297622a5c6a47bc487e1cfe62474f8c22a367629.jpg)

![1613e934e40ccfc2441fafa8a3ed5501f929e9439932a256a147bfb795c10454.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/1613e934e40ccfc2441fafa8a3ed5501f929e9439932a256a147bfb795c10454.jpg)

![197cf23e56612c7ccf9aba5a80a5816a036f10bd71bccbd18fe140d3fe23fcc1.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/197cf23e56612c7ccf9aba5a80a5816a036f10bd71bccbd18fe140d3fe23fcc1.jpg)

![1e93f0a7932bf8e205d4d81686894cd73ad0ab1a3e814f79799f454e4576d676.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/1e93f0a7932bf8e205d4d81686894cd73ad0ab1a3e814f79799f454e4576d676.jpg)

![21977ea9f7ad81d2defbd6675bab639735badce89301dad62020597d7e6611b2.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/21977ea9f7ad81d2defbd6675bab639735badce89301dad62020597d7e6611b2.jpg)

![2c67ff0787136cec801d070108036e06c498a170ee8fd442aa77b70f50fef1b3.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/2c67ff0787136cec801d070108036e06c498a170ee8fd442aa77b70f50fef1b3.jpg)

![302cb320bcc3ab5974851d6f62f02457da9dd382570d8070389c69428a7ac082.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/302cb320bcc3ab5974851d6f62f02457da9dd382570d8070389c69428a7ac082.jpg)

![3507925d9e4bcf64acf39e20b8eaec455a64d63e462fb57dc99571be45adf205.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/3507925d9e4bcf64acf39e20b8eaec455a64d63e462fb57dc99571be45adf205.jpg)

![5bd55c11265204cb9f73ea07960562433884d5da12edc360a38880a936767c7d.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/5bd55c11265204cb9f73ea07960562433884d5da12edc360a38880a936767c7d.jpg)

![787b1a65ad5730dc02ccb56abc0441d6fcdec370c9df13d597cb91a67c66097b.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/787b1a65ad5730dc02ccb56abc0441d6fcdec370c9df13d597cb91a67c66097b.jpg)

![7e29707d5c79e102800bc5302b712a6313a15e50e99284f15caffe20dbf38d17.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/7e29707d5c79e102800bc5302b712a6313a15e50e99284f15caffe20dbf38d17.jpg)

![7fb30e0ca80eabca03681f4c131137f602524ff788f870bb8fb3610cde485f43.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/7fb30e0ca80eabca03681f4c131137f602524ff788f870bb8fb3610cde485f43.jpg)

![8ebeda57f4dd65a1bd93bf7f105919f5c778aecbde01557757f1c81e139b5aa7.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/8ebeda57f4dd65a1bd93bf7f105919f5c778aecbde01557757f1c81e139b5aa7.jpg)

![8eed8c839b6b96f28b9683a86922d60f68dedebc23497ddfc5e524554b8ccf39.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/8eed8c839b6b96f28b9683a86922d60f68dedebc23497ddfc5e524554b8ccf39.jpg)

![917c5ef506b5abd0c6d414070b6369338bcdb9b7f1997ba789f058de7c8ae4a7.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/917c5ef506b5abd0c6d414070b6369338bcdb9b7f1997ba789f058de7c8ae4a7.jpg)

![9cb10c4952f3ab235db42b332d7633800d7d21541a60120b847e8af32376ec4c.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/9cb10c4952f3ab235db42b332d7633800d7d21541a60120b847e8af32376ec4c.jpg)

![a4dcc274b4c21901d0736f5cef7073e04cc48a567d67a7c1faf26f40bc86a736.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/a4dcc274b4c21901d0736f5cef7073e04cc48a567d67a7c1faf26f40bc86a736.jpg)

![b42dc38765ca61f94bfd996631343f2d6e1001471c714ed413c76e709249bf16.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/b42dc38765ca61f94bfd996631343f2d6e1001471c714ed413c76e709249bf16.jpg)

![b6b20b9159fd0df3fa7ba8cf70bdc61260328bd198ffb3f9708142d9e82a2bb0.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/b6b20b9159fd0df3fa7ba8cf70bdc61260328bd198ffb3f9708142d9e82a2bb0.jpg)

![c01c788f97314c44e45c6029b881ad5a11d6976c30eb1bd1efe286c24738409d.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/c01c788f97314c44e45c6029b881ad5a11d6976c30eb1bd1efe286c24738409d.jpg)

![ccc7a60feb9c3fb873563192c485bd248b3c375625fff591b92201bdfdc44704.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/ccc7a60feb9c3fb873563192c485bd248b3c375625fff591b92201bdfdc44704.jpg)

![cf5a40cda66f7acb7f18861de0c11790b18d4547c485f9023095fa1e668e966c.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/cf5a40cda66f7acb7f18861de0c11790b18d4547c485f9023095fa1e668e966c.jpg)

![cfebacac56ec85d0b85d1654d9022c2491d44d58ff277aca583bab1d38837ed7.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/cfebacac56ec85d0b85d1654d9022c2491d44d58ff277aca583bab1d38837ed7.jpg)

![d3b8f9adddb43b9f2f4167304362dd6b347ddcbf56da6f5e7556d2da861c251e.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/d3b8f9adddb43b9f2f4167304362dd6b347ddcbf56da6f5e7556d2da861c251e.jpg)

![d5ee4190d820ae7fa493062042727436be0b133b91837c513fd98c6d196ac860.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/d5ee4190d820ae7fa493062042727436be0b133b91837c513fd98c6d196ac860.jpg)

![dd3b1a3e219fcce57c16db950bec667b9a6b0df5f682c0d7efc1d7c1029e9aad.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/dd3b1a3e219fcce57c16db950bec667b9a6b0df5f682c0d7efc1d7c1029e9aad.jpg)

![de5dbc4e89592c9e92c409d218261016065653a6f19d7f0b14485ab68648fc33.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/de5dbc4e89592c9e92c409d218261016065653a6f19d7f0b14485ab68648fc33.jpg)

![df009286e50b67efcaae04bd10c8385d6ab0780c3b7f3578a35cb2d17cccd8fe.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/df009286e50b67efcaae04bd10c8385d6ab0780c3b7f3578a35cb2d17cccd8fe.jpg)

![e115e7d677653863edd3b9c92156d306252257c6a7cd46b127eb8a3492ff100e.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/e115e7d677653863edd3b9c92156d306252257c6a7cd46b127eb8a3492ff100e.jpg)

![e13214ddf38d2230ab6ef11153022c14cc3571c4aec6c05e2d22759839719742.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/e13214ddf38d2230ab6ef11153022c14cc3571c4aec6c05e2d22759839719742.jpg)

![f33520d7d1baa3d7acecffb665214240c149c211c3aac404ad2fee3fb3b82aa7.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/f33520d7d1baa3d7acecffb665214240c149c211c3aac404ad2fee3fb3b82aa7.jpg)

![fbb11f7202b26d4827bbf080edb1b516d697351b9d0767347f21fb4130314677.jpg](../icml_results/2131_RLCracker_%20Evaluating%20the%20Worst-Case%20Vulnerability%20of%20LLM%20Watermarks%20with%20Adaptive%20RL%20Attacks/tables/fbb11f7202b26d4827bbf080edb1b516d697351b9d0767347f21fb4130314677.jpg)

<a id="paper-2104"></a>

## LVCG: Learning ECG Representations in the Latent Vectorcardiogram Space


### Images

![01398f4e1e565d6bdcfd2dbec05e30ad731f24f1541c11fac588d77131a8b1f5.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/01398f4e1e565d6bdcfd2dbec05e30ad731f24f1541c11fac588d77131a8b1f5.jpg)

![0beb69eabdf273e190d3ebf9489283cfc358697c446e1f71d6ecd99a21dbc575.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/0beb69eabdf273e190d3ebf9489283cfc358697c446e1f71d6ecd99a21dbc575.jpg)

![1b766308ae627210ddc9e01664370f761a2e53b5dcea11b29cda92291a6591a0.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/1b766308ae627210ddc9e01664370f761a2e53b5dcea11b29cda92291a6591a0.jpg)

![6bcd3b4a2cf9cbdbc527b8d3d9d758fd336e97453f23f0cde2afee700d144d1d.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/6bcd3b4a2cf9cbdbc527b8d3d9d758fd336e97453f23f0cde2afee700d144d1d.jpg)

![b23f294f3d7fd97c596493452d6f463c89792d9c523d9c9500add8d7de24a3c6.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/b23f294f3d7fd97c596493452d6f463c89792d9c523d9c9500add8d7de24a3c6.jpg)

![be3a954869becc75201fb47ba4504390db4e678abbb210d0ea76dc44875ec553.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/be3a954869becc75201fb47ba4504390db4e678abbb210d0ea76dc44875ec553.jpg)

![ea6c7ff509587a7809c68101edc1d5f6b92e6b9400ddcc70d5820e0f52ff3487.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/ea6c7ff509587a7809c68101edc1d5f6b92e6b9400ddcc70d5820e0f52ff3487.jpg)

![f447de407ee1b6c5476517d52557079a2a639b56d74f0a7209970c11eb230518.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/images/f447de407ee1b6c5476517d52557079a2a639b56d74f0a7209970c11eb230518.jpg)

### Tables

![0df6330dd0baf0586ef3fed3bf4d69a6226481b5080dea94560ce24a59ae9d97.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/0df6330dd0baf0586ef3fed3bf4d69a6226481b5080dea94560ce24a59ae9d97.jpg)

![3a4c2bf3c181f4d732210e936a0255b88699a7115c8a7d037dce809f150bcf7e.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/3a4c2bf3c181f4d732210e936a0255b88699a7115c8a7d037dce809f150bcf7e.jpg)

![3fc06c5a43a9a059bd5bb1f78987216648080313bcfe35be16513dd226531a65.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/3fc06c5a43a9a059bd5bb1f78987216648080313bcfe35be16513dd226531a65.jpg)

![431c6c26810ae633b4a9a0f4831b780e1c98ff8bc2f8a6594da180d8734bbefc.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/431c6c26810ae633b4a9a0f4831b780e1c98ff8bc2f8a6594da180d8734bbefc.jpg)

![4521cb828f4b4fcdbc04666e32625154ad753de6aaca479e2796f871b3dd0df5.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/4521cb828f4b4fcdbc04666e32625154ad753de6aaca479e2796f871b3dd0df5.jpg)

![5701a8412a0de00c56c4e59f3a4c518c7e79ddf2372330253a40da750c1363f8.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/5701a8412a0de00c56c4e59f3a4c518c7e79ddf2372330253a40da750c1363f8.jpg)

![801ce43082b7c5dfd2be11fb34026a9f47092f8614429028058564ab49f046fa.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/801ce43082b7c5dfd2be11fb34026a9f47092f8614429028058564ab49f046fa.jpg)

![8e9dc889124a2c46adbc7c85a8f580344428dbaee981cd496cb81530c9d4e8c7.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/8e9dc889124a2c46adbc7c85a8f580344428dbaee981cd496cb81530c9d4e8c7.jpg)

![9f71df64f7023d01a30b93747ac6cc1f6b4fe2dbafe8deacd2b8a4c444bf641b.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/9f71df64f7023d01a30b93747ac6cc1f6b4fe2dbafe8deacd2b8a4c444bf641b.jpg)

![b1977492b120ca1c638ad7afdd8ee268bed2d149513f28d9fdb139fffde80f01.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/b1977492b120ca1c638ad7afdd8ee268bed2d149513f28d9fdb139fffde80f01.jpg)

![b2626b2ede82c5167ebda94515d3fced3926069fe171ebc09876e7f2e9fc0d6e.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/b2626b2ede82c5167ebda94515d3fced3926069fe171ebc09876e7f2e9fc0d6e.jpg)

![e34c43fac523035cbc43c94fe1001c75238335e54f1beb46090fe5bd115d784e.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/e34c43fac523035cbc43c94fe1001c75238335e54f1beb46090fe5bd115d784e.jpg)

![e3858d005f012bd348238de296099288cedc59954b9a05ef128f3f26a582c093.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/e3858d005f012bd348238de296099288cedc59954b9a05ef128f3f26a582c093.jpg)

![e8805c9c5939c3e6f632083a3d276353fa4c4df1b46bf8419e05055acacdb18d.jpg](../icml_results/2132_LVCG_%20Learning%20ECG%20Representations%20in%20the%20Latent%20Vectorcardiogram%20Space/tables/e8805c9c5939c3e6f632083a3d276353fa4c4df1b46bf8419e05055acacdb18d.jpg)

<a id="paper-2105"></a>

## GeoFlow: Geo-Aware Modeling of Inter-Area Relationships in Origin-Destination Flow Prediction and Generation


### Images

![247406fc33df05f2c2539cbde6008e059d0894a6cacae9c9a18bf5412ca3f3b9.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/images/247406fc33df05f2c2539cbde6008e059d0894a6cacae9c9a18bf5412ca3f3b9.jpg)

![3d2d8ce7caa5675c558df9a9bf98f6b618ec10090d00a2bd0eb2bc0b566e6926.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/images/3d2d8ce7caa5675c558df9a9bf98f6b618ec10090d00a2bd0eb2bc0b566e6926.jpg)

![919677fbae77b8d7dfaa23be0eec429c3f8d948c567fcf24d4fff745b7ce88e3.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/images/919677fbae77b8d7dfaa23be0eec429c3f8d948c567fcf24d4fff745b7ce88e3.jpg)

![9bf4b30f4b27f7883fec26624dd51a9c44c35dbd7bbf7c0c66fb453940537656.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/images/9bf4b30f4b27f7883fec26624dd51a9c44c35dbd7bbf7c0c66fb453940537656.jpg)

![e63739ba971ee430597996117aa708b8deb0e70cc982e6875e6e125a64302286.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/images/e63739ba971ee430597996117aa708b8deb0e70cc982e6875e6e125a64302286.jpg)

### Tables

![0d534c57894e84070e08e035afa0269246de1c981b59adc42af7902b3ea797f4.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/0d534c57894e84070e08e035afa0269246de1c981b59adc42af7902b3ea797f4.jpg)

![381ef6722aef835e44986297568e083815242a6e6fc3cd6185a3ed24cb0fc0d9.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/381ef6722aef835e44986297568e083815242a6e6fc3cd6185a3ed24cb0fc0d9.jpg)

![3a33a36958bceec02f07791d197064df0e4bc035e2b56adb48eb589ab9ad7df9.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/3a33a36958bceec02f07791d197064df0e4bc035e2b56adb48eb589ab9ad7df9.jpg)

![4147661de410c8877660e9432b2992684aded9448e1d70902d0c8a8831613169.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/4147661de410c8877660e9432b2992684aded9448e1d70902d0c8a8831613169.jpg)

![760fd847e7baf6c8fc236ee1990002379cd16f21aebfe55738dd3e16be7bb384.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/760fd847e7baf6c8fc236ee1990002379cd16f21aebfe55738dd3e16be7bb384.jpg)

![80eb16c4c184ae964988a1159a7371d9bbf588fe761c0f2e20c71881adce1b88.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/80eb16c4c184ae964988a1159a7371d9bbf588fe761c0f2e20c71881adce1b88.jpg)

![91fba1bf5bdee0f6938bd31f72d3759b22e8d489d995ab79919350f38d3d5bdc.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/91fba1bf5bdee0f6938bd31f72d3759b22e8d489d995ab79919350f38d3d5bdc.jpg)

![ac1637afc1e5b3df8bc89fb237b5146c8f2b6e949ff6d93d4a14b5e08490f0ad.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/ac1637afc1e5b3df8bc89fb237b5146c8f2b6e949ff6d93d4a14b5e08490f0ad.jpg)

![b272470022aa076ba1b21bbdc90865f8c565070a9ce721904b8976e64bc9fcf3.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/b272470022aa076ba1b21bbdc90865f8c565070a9ce721904b8976e64bc9fcf3.jpg)

![c13293570027e18b94eb2e9ea3d6be1141d71cfbadc9707db8612d1ab3cfe1a8.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/c13293570027e18b94eb2e9ea3d6be1141d71cfbadc9707db8612d1ab3cfe1a8.jpg)

![c54472442368d616af55df443034d452e5fb6de388e5108a7caffb4a1fc5a6a4.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/c54472442368d616af55df443034d452e5fb6de388e5108a7caffb4a1fc5a6a4.jpg)

![d3cff873f4bce168fd2ad02ec80aa03cb9428beb68f63cc34c5e7398812ec372.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/d3cff873f4bce168fd2ad02ec80aa03cb9428beb68f63cc34c5e7398812ec372.jpg)

![d50a608a1d7a481858e28e73a39b223de95e5ead7cd67689fbc4ec86750e5ad8.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/d50a608a1d7a481858e28e73a39b223de95e5ead7cd67689fbc4ec86750e5ad8.jpg)

![e5f7d291a7b75f10033b5dae9833f455fcaa2e3750aa1dadb28613cdf6e5b53e.jpg](../icml_results/2133_GeoFlow_%20Geo-Aware%20Modeling%20of%20Inter-Area%20Relationships%20in%20Origin-Destination%20Flow%20Prediction%20and%20Ge/tables/e5f7d291a7b75f10033b5dae9833f455fcaa2e3750aa1dadb28613cdf6e5b53e.jpg)

<a id="paper-2106"></a>

## IACW: Intent-Aware Controllable Watermarking for Scalable Authorial Intent Attribution


### Images

![24f3e912de136ff87fa5b360c51da95abcc33b7de7606c77396bea3fbfc50c90.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/images/24f3e912de136ff87fa5b360c51da95abcc33b7de7606c77396bea3fbfc50c90.jpg)

![8457743299e7e324137be1a3e2c87362ab29f09689c86f18ef9f5a904acee048.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/images/8457743299e7e324137be1a3e2c87362ab29f09689c86f18ef9f5a904acee048.jpg)

![9a674c857d84283dd8681fcc41e992098222dfc2d3dd2b90b47f49a846d83c1c.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/images/9a674c857d84283dd8681fcc41e992098222dfc2d3dd2b90b47f49a846d83c1c.jpg)

### Tables

![08f5b8ebf02f5b2b61954a897090564878f3ab944e9672959f1beb86796a891b.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/08f5b8ebf02f5b2b61954a897090564878f3ab944e9672959f1beb86796a891b.jpg)

![26d11c5232a80d45dcaced79776919aa02ecf24fe20419cd35573ae70a1ae8d8.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/26d11c5232a80d45dcaced79776919aa02ecf24fe20419cd35573ae70a1ae8d8.jpg)

![2942c1de9e02c4a0fc7b04dca5da448cef510e31ca6a37cd664b40a21733df4b.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/2942c1de9e02c4a0fc7b04dca5da448cef510e31ca6a37cd664b40a21733df4b.jpg)

![520839201cc163b3a37c2e0dd3608459ef5fc5944b2a69e3aec0418b0248fbdb.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/520839201cc163b3a37c2e0dd3608459ef5fc5944b2a69e3aec0418b0248fbdb.jpg)

![5c6e7a5c5f605abe6a20db467b5b066380a4411c14310a4f8cfd8c90f6597fde.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/5c6e7a5c5f605abe6a20db467b5b066380a4411c14310a4f8cfd8c90f6597fde.jpg)

![a7e35dd4672212a1bcdec731eafc5bf5004a4b9d4e44c449074eb23c3da25d49.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/a7e35dd4672212a1bcdec731eafc5bf5004a4b9d4e44c449074eb23c3da25d49.jpg)

![aac875baedb50d77a90213c3c62d81962b201a66b1229c42b06432c922a2e9d6.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/aac875baedb50d77a90213c3c62d81962b201a66b1229c42b06432c922a2e9d6.jpg)

![d46b95bf37bef065a83c2c2dbad571dd1eb429fa37227ed46b163702b2042640.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/d46b95bf37bef065a83c2c2dbad571dd1eb429fa37227ed46b163702b2042640.jpg)

![d7e7aba024086fb571f400094196d40ea72304f384eac28241cdbf9c464e9a51.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/d7e7aba024086fb571f400094196d40ea72304f384eac28241cdbf9c464e9a51.jpg)

![dfeb3fb8487f99c344c83af6954372b80f73d0380e64c8e9596c032cf1d02d4e.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/dfeb3fb8487f99c344c83af6954372b80f73d0380e64c8e9596c032cf1d02d4e.jpg)

![e647008d59fcfa49f41994737698146064c26ddd9ad6cb91da75e30be4805c29.jpg](../icml_results/2134_IACW_%20Intent-Aware%20Controllable%20Watermarking%20for%20Scalable%20Authorial%20Intent%20Attribution/tables/e647008d59fcfa49f41994737698146064c26ddd9ad6cb91da75e30be4805c29.jpg)

<a id="paper-2107"></a>

## ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation


### Images

![131e8dfa52ba9be1081f0f6693f3204df99ab0d33493e74026321387b23ea9a7.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/131e8dfa52ba9be1081f0f6693f3204df99ab0d33493e74026321387b23ea9a7.jpg)

![33acf1b6f29341ace7ec1119b25f7289456f39bcaa589b97c938e77be4835aae.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/33acf1b6f29341ace7ec1119b25f7289456f39bcaa589b97c938e77be4835aae.jpg)

![432c40a696c5f7226d4352df377847c4a07758e652731455c3118cc431d171cf.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/432c40a696c5f7226d4352df377847c4a07758e652731455c3118cc431d171cf.jpg)

![4e9ccc018626eb797c0fd5be4cda6c8964935c1ecdcdeacbfcfc59775e60ff4f.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/4e9ccc018626eb797c0fd5be4cda6c8964935c1ecdcdeacbfcfc59775e60ff4f.jpg)

![4f2d415f94929665160d0a32c0a728ddeea9d04e30b334d99ed6e2ff8f3978f7.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/4f2d415f94929665160d0a32c0a728ddeea9d04e30b334d99ed6e2ff8f3978f7.jpg)

![6453ad2db1f4fcc9b1cdb27e77b06a59bdbbecec6b2312d0ba11d9afbfeea983.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/6453ad2db1f4fcc9b1cdb27e77b06a59bdbbecec6b2312d0ba11d9afbfeea983.jpg)

![691235461489e5e3ec6dd49bb7e319b651598cf12319d692c7062e2dd1193b1c.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/691235461489e5e3ec6dd49bb7e319b651598cf12319d692c7062e2dd1193b1c.jpg)

![98c50265fc6ecae1f663e13824c542a520c3c0e0503c9a3d7356f1535994def2.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/98c50265fc6ecae1f663e13824c542a520c3c0e0503c9a3d7356f1535994def2.jpg)

![c8ddadf495e37e7eeb714adb2942bdc8dc6bb2f55c443b8be51a99b8157f1692.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/c8ddadf495e37e7eeb714adb2942bdc8dc6bb2f55c443b8be51a99b8157f1692.jpg)

![c917ac7b46e8ac9f27f67bbd50af5ada57aed33d1b11c7f1b05fe3974371afe4.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/c917ac7b46e8ac9f27f67bbd50af5ada57aed33d1b11c7f1b05fe3974371afe4.jpg)

![e270bbaac4df9062896650c221888abd408151dc860701d3aa162fd6188cfeaa.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/images/e270bbaac4df9062896650c221888abd408151dc860701d3aa162fd6188cfeaa.jpg)

### Tables

![2e5691dc53a2ddca8371f1864bb33f420407511c0de442336713e050bb83e081.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/2e5691dc53a2ddca8371f1864bb33f420407511c0de442336713e050bb83e081.jpg)

![4ea381baf4c3480b63ee9ff70dcd4e7d368be9c999bb366c8396d58662dd63a0.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/4ea381baf4c3480b63ee9ff70dcd4e7d368be9c999bb366c8396d58662dd63a0.jpg)

![72e9650b3c7b03eb4d303cac066f3bde7a94671e0970e69b436227013fa93183.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/72e9650b3c7b03eb4d303cac066f3bde7a94671e0970e69b436227013fa93183.jpg)

![ae1d6de587c4bbe6752000fe76d35e42744dc752101c9bda5b7b72131c75dbd6.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/ae1d6de587c4bbe6752000fe76d35e42744dc752101c9bda5b7b72131c75dbd6.jpg)

![d9a07a1522dfc31bb084d9e2f8f745d3a36f61a42d43b2900e80ce3a37996446.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/d9a07a1522dfc31bb084d9e2f8f745d3a36f61a42d43b2900e80ce3a37996446.jpg)

![e68f5f1dbb7ee96c7b457f4048c296335f4bde4a2ff2e2345bbdf4922f8cf649.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/e68f5f1dbb7ee96c7b457f4048c296335f4bde4a2ff2e2345bbdf4922f8cf649.jpg)

![edcdd4f65467e39a342a3059f712351e8a2f6c70c211e22afa2fbdd7d6205fe0.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/edcdd4f65467e39a342a3059f712351e8a2f6c70c211e22afa2fbdd7d6205fe0.jpg)

![f014a6483bd6fe1aee72a892ee823b9cdacd8c7f41e88e40e5130901dded7c32.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/f014a6483bd6fe1aee72a892ee823b9cdacd8c7f41e88e40e5130901dded7c32.jpg)

![f3f14927cbef7e9796186f329a52ea75822e4b3d397430db6a415df520790115.jpg](../icml_results/2135_ConceptMoE_%20Adaptive%20Token-to-Concept%20Compression%20for%20Implicit%20Compute%20Allocation/tables/f3f14927cbef7e9796186f329a52ea75822e4b3d397430db6a415df520790115.jpg)

<a id="paper-2108"></a>

## TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches


### Images

![1ec16111e9eff2c67624d92436c07c2549c7cf7859bd7ae63ab660e3ec81ad53.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/1ec16111e9eff2c67624d92436c07c2549c7cf7859bd7ae63ab660e3ec81ad53.jpg)

![26148ad6e02533f560fdead3c6ca73fb90a0f2b61c0c2b3fc4a3a90d3397d34f.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/26148ad6e02533f560fdead3c6ca73fb90a0f2b61c0c2b3fc4a3a90d3397d34f.jpg)

![4194607770316f1aefeb80b5673d5f891fd0e271897083bcceaa8c92e385265f.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/4194607770316f1aefeb80b5673d5f891fd0e271897083bcceaa8c92e385265f.jpg)

![5b83b4ef98dc6e8de4f84c367968b9f41bc16c04f5a67db8ba656acc81c33867.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/5b83b4ef98dc6e8de4f84c367968b9f41bc16c04f5a67db8ba656acc81c33867.jpg)

![6641020c86662270441ca74d9dbcbaa0730db213e40144ef9f358b1e20a688d9.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/6641020c86662270441ca74d9dbcbaa0730db213e40144ef9f358b1e20a688d9.jpg)

![6d38098d1b1eb20758fc4e4fa9d0e6235f9dc8ac6abbda200e58e734ba12b21b.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/6d38098d1b1eb20758fc4e4fa9d0e6235f9dc8ac6abbda200e58e734ba12b21b.jpg)

![7026002698eee75a076d4082592f7ec93c84be9ed712fcca6a14bc856e45900c.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/7026002698eee75a076d4082592f7ec93c84be9ed712fcca6a14bc856e45900c.jpg)

![8642323c9500099ef0dc8b07a86dd6e68b84012fd48a8071111e78660d4933fa.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/8642323c9500099ef0dc8b07a86dd6e68b84012fd48a8071111e78660d4933fa.jpg)

![87888ce3a3728c58cd1e3c3b67e1e9b254ad17e023d5014a1007df0cace9edea.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/87888ce3a3728c58cd1e3c3b67e1e9b254ad17e023d5014a1007df0cace9edea.jpg)

![88fd80685b1af58ea994da813b891657191aa02d4b225e60e79249b3b8f8c559.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/88fd80685b1af58ea994da813b891657191aa02d4b225e60e79249b3b8f8c559.jpg)

![8edf9db72f0f5aec0b4d7db68a335c405a714b734331f3c98486b0be48608707.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/8edf9db72f0f5aec0b4d7db68a335c405a714b734331f3c98486b0be48608707.jpg)

![a443d562d9c76f1639764bb69a3df93ef096d1e1b3dd344b18f6e05fda0fa84f.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/a443d562d9c76f1639764bb69a3df93ef096d1e1b3dd344b18f6e05fda0fa84f.jpg)

![a7a850526c8ba3e1a0aad736ffc85ed4de5420ad43b8aaeb723e40eced709d9c.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/a7a850526c8ba3e1a0aad736ffc85ed4de5420ad43b8aaeb723e40eced709d9c.jpg)

![ad701e3b756e3ad9fb984d9a1a4c9aaaa0a6c23dec7efc9e4cdf5d969b0dc8f1.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/ad701e3b756e3ad9fb984d9a1a4c9aaaa0a6c23dec7efc9e4cdf5d969b0dc8f1.jpg)

![b05f55e06411a214ce8fe27af1eb8e1a49f7b78838a1b90c553d6c4fa2574cc5.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/b05f55e06411a214ce8fe27af1eb8e1a49f7b78838a1b90c553d6c4fa2574cc5.jpg)

![c020baaee784ec7604751bf98f0ac5001a7da6103d0ba4842949cb13d049f3de.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/c020baaee784ec7604751bf98f0ac5001a7da6103d0ba4842949cb13d049f3de.jpg)

![c037cd97bc826659dcebbcf85dca0d474c7a7c4b9a2afb56d4d3a73e2fe762a9.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/c037cd97bc826659dcebbcf85dca0d474c7a7c4b9a2afb56d4d3a73e2fe762a9.jpg)

![c3ec6dd3d8b428ded3f15e216c792ad3f9e34206084aff393ac2c9a0a61838d0.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/c3ec6dd3d8b428ded3f15e216c792ad3f9e34206084aff393ac2c9a0a61838d0.jpg)

![cc01f999d9bce36965dfa86eff7a6ebde6188a390c2197aae8ed3c8f631fe35b.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/images/cc01f999d9bce36965dfa86eff7a6ebde6188a390c2197aae8ed3c8f631fe35b.jpg)

### Tables

![0adbd6a7ee19b35852c82407602380aa47c46df909daedef48bddb3d9fe2e834.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/0adbd6a7ee19b35852c82407602380aa47c46df909daedef48bddb3d9fe2e834.jpg)

![1e953494c81978fecca03d23c314ef1784164d27f3a9495b169ef4608e2423c7.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/1e953494c81978fecca03d23c314ef1784164d27f3a9495b169ef4608e2423c7.jpg)

![24db056e0edba692d31f9720577a34af84de6ca445543b32f0dd7007202b0e73.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/24db056e0edba692d31f9720577a34af84de6ca445543b32f0dd7007202b0e73.jpg)

![634ecb56f8641439baeb254f690153e66f9ca8bfa26c155d8985fba0db3279e6.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/634ecb56f8641439baeb254f690153e66f9ca8bfa26c155d8985fba0db3279e6.jpg)

![738f5e0be413390397d9c18e7918f7d55f519be3dddd606dc717a5a0921964f5.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/738f5e0be413390397d9c18e7918f7d55f519be3dddd606dc717a5a0921964f5.jpg)

![76c4d72c9a6d434675b465fd594b2333d1925ffc625338629e658c68a5fe8db8.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/76c4d72c9a6d434675b465fd594b2333d1925ffc625338629e658c68a5fe8db8.jpg)

![8219159b8b8fca117363516eac7349775c766a5feff8efb6fe74f3ba7fa34f3a.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/8219159b8b8fca117363516eac7349775c766a5feff8efb6fe74f3ba7fa34f3a.jpg)

![8e5843629810b59a2527733b27f48aa19439bd29c33fffc6f1bef3908c4c9f6b.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/8e5843629810b59a2527733b27f48aa19439bd29c33fffc6f1bef3908c4c9f6b.jpg)

![a1b15d1527713bbf09473488ca673425e1b39a9cf652342baa2a23d56b1a0375.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/a1b15d1527713bbf09473488ca673425e1b39a9cf652342baa2a23d56b1a0375.jpg)

![c6d28781b774c504b485c0f2d0fb1ccb6867c0fb0acc42e09905bc99ccd8aae3.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/c6d28781b774c504b485c0f2d0fb1ccb6867c0fb0acc42e09905bc99ccd8aae3.jpg)

![c8239164300be531900e56c36b236baeb4ba935a02729c20fa28cdca1153c2d9.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/c8239164300be531900e56c36b236baeb4ba935a02729c20fa28cdca1153c2d9.jpg)

![d5d8dcccb675efb2ac3ed03b90b0184a057e456a707f778d5eae4ff1403d7dc7.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/d5d8dcccb675efb2ac3ed03b90b0184a057e456a707f778d5eae4ff1403d7dc7.jpg)

![e24761c152efafd8ab01f67d9d3951c0c5073ea3243d2abb07af70c2fda8823e.jpg](../icml_results/2136_TRAP_%20Hijacking%20VLA%20CoT-Reasoning%20via%20Adversarial%20Patches/tables/e24761c152efafd8ab01f67d9d3951c0c5073ea3243d2abb07af70c2fda8823e.jpg)

<a id="paper-2109"></a>

## HONet: Data-Efficient Learning for Exact Cover Tasks via Hypergraph Optimization


### Images

![21abf3310981c30934159facb4129c8e713ad473143f9f92d8558e3f112ef6e8.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/images/21abf3310981c30934159facb4129c8e713ad473143f9f92d8558e3f112ef6e8.jpg)

![4ef93e4ae47d1dfa21578a2b5f1b94e4680b0ce2b810b70e15dfa22586292d4c.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/images/4ef93e4ae47d1dfa21578a2b5f1b94e4680b0ce2b810b70e15dfa22586292d4c.jpg)

![986e60896d0c25a2b751fc1712720f09a794ba357656963b74f2fcb317f68e04.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/images/986e60896d0c25a2b751fc1712720f09a794ba357656963b74f2fcb317f68e04.jpg)

![dff32ae851b567bca3cdc38e60eeb92f85b92eb44064cad7d76cbe7aec5e7500.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/images/dff32ae851b567bca3cdc38e60eeb92f85b92eb44064cad7d76cbe7aec5e7500.jpg)

### Tables

![1384e7e8c55f80139950c275e976e1e0d29a624a9e0f389165dee443d71f4573.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/1384e7e8c55f80139950c275e976e1e0d29a624a9e0f389165dee443d71f4573.jpg)

![2736668a57a704d5285e099b07b1f89bb0e3367dd7378df757a7611a73088a87.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/2736668a57a704d5285e099b07b1f89bb0e3367dd7378df757a7611a73088a87.jpg)

![3b4eb96a2b471f1966af4366b437059cc49e0dc9dbcdcc6e1b960655ba1fca14.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/3b4eb96a2b471f1966af4366b437059cc49e0dc9dbcdcc6e1b960655ba1fca14.jpg)

![57725bef3af64055c8e4a2012ee0332a82dccb79c5793b6d324219ed1cfa26d4.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/57725bef3af64055c8e4a2012ee0332a82dccb79c5793b6d324219ed1cfa26d4.jpg)

![668d5de168f9cf1fa07549677f881cca552d759d574ff17a3e18c9bae4c95e79.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/668d5de168f9cf1fa07549677f881cca552d759d574ff17a3e18c9bae4c95e79.jpg)

![7611fc911ce9a6dc2699485d1d4d00624c9eafd7041048baf35f2c5d825c65f1.jpg](../icml_results/2137_HONet_%20Data-Efficient%20Learning%20for%20Exact%20Cover%20Tasks%20via%20Hypergraph%20Optimization/tables/7611fc911ce9a6dc2699485d1d4d00624c9eafd7041048baf35f2c5d825c65f1.jpg)

<a id="paper-2110"></a>

## Reliability-Aware LLM Alignment from Inconsistent Human Feedback


### Images

![13959ebb6ee4b80adb3063c143f25545b0f0cb69dbd66da16548fbd4b9c8cd23.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/13959ebb6ee4b80adb3063c143f25545b0f0cb69dbd66da16548fbd4b9c8cd23.jpg)

![2d05ed12f21f0022de8eb804c35a759814b51ebb65d9204080817ee9d3c782cc.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/2d05ed12f21f0022de8eb804c35a759814b51ebb65d9204080817ee9d3c782cc.jpg)

![2d23de11f39c293a6d28ec91d09c046324c468b9ddb7ffc29623f2f24ba2481c.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/2d23de11f39c293a6d28ec91d09c046324c468b9ddb7ffc29623f2f24ba2481c.jpg)

![cce93a07b38ddcf7b79d05f6c23d1741ddb955144c8bef0dca99ce4de2fbd55f.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/cce93a07b38ddcf7b79d05f6c23d1741ddb955144c8bef0dca99ce4de2fbd55f.jpg)

![de01f90212ed7e09dfc705dda88d724080410eaa85e869b6aa65544cab84a61b.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/de01f90212ed7e09dfc705dda88d724080410eaa85e869b6aa65544cab84a61b.jpg)

![e382105fcc55e8cd402c6a20c8104d4cb7d6eb71d2d2650e4f8e704d4278dd04.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/e382105fcc55e8cd402c6a20c8104d4cb7d6eb71d2d2650e4f8e704d4278dd04.jpg)

![f44cd0f3b85d932e2db4028ccfd5983ca79172134e41c1ea703bede43a5de848.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/images/f44cd0f3b85d932e2db4028ccfd5983ca79172134e41c1ea703bede43a5de848.jpg)

### Tables

![05cd64e393fb061f1017898ee7e1667f91ec3696b68c7e2b66d6b046d2cd3b7c.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/05cd64e393fb061f1017898ee7e1667f91ec3696b68c7e2b66d6b046d2cd3b7c.jpg)

![3426bf33d7739c4d3c10363e882578f3a9e301d4dc0638213c4015f10dc9441d.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/3426bf33d7739c4d3c10363e882578f3a9e301d4dc0638213c4015f10dc9441d.jpg)

![49b9b9a5c174072257221ab6165e5caa18e16f82ac667eb3f6c30098a35f5919.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/49b9b9a5c174072257221ab6165e5caa18e16f82ac667eb3f6c30098a35f5919.jpg)

![4a6d5506f9226f319e5d210ed7fae8b270da213c57d01c1453e81b9760c0eb13.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/4a6d5506f9226f319e5d210ed7fae8b270da213c57d01c1453e81b9760c0eb13.jpg)

![514923e9bbcd516323a9437db724b7d363c54c8914a7e2af99ea7d572e7ec912.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/514923e9bbcd516323a9437db724b7d363c54c8914a7e2af99ea7d572e7ec912.jpg)

![595f565a96ba421a098b609df1d9c4e1bb393351bdfa33fe659ad91cd50c7764.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/595f565a96ba421a098b609df1d9c4e1bb393351bdfa33fe659ad91cd50c7764.jpg)

![6573ad2844fbe4e5e9910667d82e8da63b28e486c538bdcbdf5507d254240eeb.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/6573ad2844fbe4e5e9910667d82e8da63b28e486c538bdcbdf5507d254240eeb.jpg)

![6a38644bde161569c6f8f65a2cc7b73875113c49f2872c25e072186805a6cbb4.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/6a38644bde161569c6f8f65a2cc7b73875113c49f2872c25e072186805a6cbb4.jpg)

![83666300e5dd3a6d266a3d181a17efcca58901b37b900fca207fd541e04176c2.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/83666300e5dd3a6d266a3d181a17efcca58901b37b900fca207fd541e04176c2.jpg)

![8dd68330f58bb6e1c1dc3ca2e634c8bf19af1649582e9b2cbac96b6fb0c19d2d.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/8dd68330f58bb6e1c1dc3ca2e634c8bf19af1649582e9b2cbac96b6fb0c19d2d.jpg)

![aaf5339923ca112774f371d32964cae02aaa366e67913a44ecbc808d548d6e05.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/aaf5339923ca112774f371d32964cae02aaa366e67913a44ecbc808d548d6e05.jpg)

![b0ccaa63297edfabc6b63ecd8737ce753308c057b065cecb6930b6082c7b9c0a.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/b0ccaa63297edfabc6b63ecd8737ce753308c057b065cecb6930b6082c7b9c0a.jpg)

![c0446c1b8a9b0942615a202cadd055939ab12d49baa37b0453e996a80a0d70fd.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/c0446c1b8a9b0942615a202cadd055939ab12d49baa37b0453e996a80a0d70fd.jpg)

![f3ba968153c3654809ea2ccce816a82fff4fb7c1e255f73aff839946adb54c45.jpg](../icml_results/2138_Reliability-Aware%20LLM%20Alignment%20from%20Inconsistent%20Human%20Feedback/tables/f3ba968153c3654809ea2ccce816a82fff4fb7c1e255f73aff839946adb54c45.jpg)

<a id="paper-2111"></a>

## Foundation Inference Models for Ordinary Differential Equations


### Images

![0c6245ae494c6f81f81323ce7351ba1b976e54616c1991af81de4ce4478bc4f1.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/0c6245ae494c6f81f81323ce7351ba1b976e54616c1991af81de4ce4478bc4f1.jpg)

![11974c591c83f996f4a8e29f947ddecd7fae79a1cc4d9ace64d7c97cbab7dde1.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/11974c591c83f996f4a8e29f947ddecd7fae79a1cc4d9ace64d7c97cbab7dde1.jpg)

![122219db188283588c7acea878e7652d027ca9cbc6b4c44cf8cb5e3f35c815f1.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/122219db188283588c7acea878e7652d027ca9cbc6b4c44cf8cb5e3f35c815f1.jpg)

![32a75e23a22ca63a16aba1f116e20e79a602891b60592826f1ae9fd90c115cc1.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/32a75e23a22ca63a16aba1f116e20e79a602891b60592826f1ae9fd90c115cc1.jpg)

![34c97db65f133a04656487289f609ba63d5b476d85e5aed5538845c26bea7fdd.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/34c97db65f133a04656487289f609ba63d5b476d85e5aed5538845c26bea7fdd.jpg)

![3976b6d9dbaa51e5a0b20be1849457113aa3b9c20457883cff09252e7d53bb49.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/3976b6d9dbaa51e5a0b20be1849457113aa3b9c20457883cff09252e7d53bb49.jpg)

![58801f8d61e24bcf2e31390af743569efffd7f7b18519805d70a8e524dc01bbf.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/58801f8d61e24bcf2e31390af743569efffd7f7b18519805d70a8e524dc01bbf.jpg)

![6bdb16fadb2ef7c315edce19169ed76da06543b43f1714dc02b55494eb66b578.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/6bdb16fadb2ef7c315edce19169ed76da06543b43f1714dc02b55494eb66b578.jpg)

![6da50b64949cd9fe39b3e9b2cf9324baf7928869fd7990d8c76154f16e362b25.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/6da50b64949cd9fe39b3e9b2cf9324baf7928869fd7990d8c76154f16e362b25.jpg)

![7a6906c2150f9b6f68837645baf35016bf4ca7ee79563ab8f8f8ca2f159addf5.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/7a6906c2150f9b6f68837645baf35016bf4ca7ee79563ab8f8f8ca2f159addf5.jpg)

![8240c6300066280d17536936b6e265defb4a264c3344f4ea0613c002811cb3b4.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/8240c6300066280d17536936b6e265defb4a264c3344f4ea0613c002811cb3b4.jpg)

![90bb63bead513518982531acce72744e034f4b88c926581be5546bd8cb68474b.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/90bb63bead513518982531acce72744e034f4b88c926581be5546bd8cb68474b.jpg)

![9216f7479e51d8c95df50ada257c15d239d5251a54603f1de4f0162051acf333.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/9216f7479e51d8c95df50ada257c15d239d5251a54603f1de4f0162051acf333.jpg)

![975056486641e9649a8f21a00c49d93e9058146758d4cc9c29a8f1fe490b63ad.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/975056486641e9649a8f21a00c49d93e9058146758d4cc9c29a8f1fe490b63ad.jpg)

![a5b52baebb6c71064daeff16e546e98a6fac8f29b0e4a0b5529a54e390666e5f.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/a5b52baebb6c71064daeff16e546e98a6fac8f29b0e4a0b5529a54e390666e5f.jpg)

![b2711bebf47428cb6c73d23a282204b812ff5754c4a6db99dd07644faf3ce7a3.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/b2711bebf47428cb6c73d23a282204b812ff5754c4a6db99dd07644faf3ce7a3.jpg)

![b63e61c8c8f3fc0956ae8d1303ec2ff4670b4e0a1fc391734001aed8926d2532.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/b63e61c8c8f3fc0956ae8d1303ec2ff4670b4e0a1fc391734001aed8926d2532.jpg)

![d1648db3e8e39278c93ef00e5f558ea282dea36dcba91e65580f4c337c8c62fb.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/d1648db3e8e39278c93ef00e5f558ea282dea36dcba91e65580f4c337c8c62fb.jpg)

![dd4a141df1afca66ba65395af9f29189aec1b794f681dd0a31d7c8e7e9c27f13.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/dd4a141df1afca66ba65395af9f29189aec1b794f681dd0a31d7c8e7e9c27f13.jpg)

![ed87a46ef49d071972b1d2263f6954adde5bc57968d81a0394423bece9859dd4.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/ed87a46ef49d071972b1d2263f6954adde5bc57968d81a0394423bece9859dd4.jpg)

![f7134108cb383f89ad9f43c3579a2d331202e0b3262378f7c2278114b071d535.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/f7134108cb383f89ad9f43c3579a2d331202e0b3262378f7c2278114b071d535.jpg)

![ff4d9c9978c0b4b362c12ff613d9d2c6328c749a0af989c4f4705eae48b1fafb.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/images/ff4d9c9978c0b4b362c12ff613d9d2c6328c749a0af989c4f4705eae48b1fafb.jpg)

### Tables

![02c26f888f94e2efa4f0277e69a1adbff30038a991bb80e1f685c6f027c0686c.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/02c26f888f94e2efa4f0277e69a1adbff30038a991bb80e1f685c6f027c0686c.jpg)

![0baeea430f8b59575517c1aceecb0268bac5e3ac611387ff9497a09282c76685.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/0baeea430f8b59575517c1aceecb0268bac5e3ac611387ff9497a09282c76685.jpg)

![2a0e43da6b6ebc2565e644af5739806097450298899dd9dbf4576e80a9d6d817.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/2a0e43da6b6ebc2565e644af5739806097450298899dd9dbf4576e80a9d6d817.jpg)

![34f45424fef2ebbf3464231b53b84f7169a63574d8c1fc2db27e3a17a64ac954.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/34f45424fef2ebbf3464231b53b84f7169a63574d8c1fc2db27e3a17a64ac954.jpg)

![66c0d1ac9d360096ff80924025e10eb11aa5fb7e1c322bdc3f4eb936df37d7b0.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/66c0d1ac9d360096ff80924025e10eb11aa5fb7e1c322bdc3f4eb936df37d7b0.jpg)

![80f4e06cb83e267a5ce3acfea413943cec1080d262986bec4ad817c302066787.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/80f4e06cb83e267a5ce3acfea413943cec1080d262986bec4ad817c302066787.jpg)

![8b67b04201fb58f7621ee43f24d9832cc328e6b298c69bd42ec9031358a04c29.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/8b67b04201fb58f7621ee43f24d9832cc328e6b298c69bd42ec9031358a04c29.jpg)

![8ed475305ead5efc2302685f494c22bc18b35cce5c3897fb0e600eaf8caddf55.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/8ed475305ead5efc2302685f494c22bc18b35cce5c3897fb0e600eaf8caddf55.jpg)

![b33074244ed203a84e7c86d7d515c3e2cc4c4b422263b3124089fab4d200546f.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/b33074244ed203a84e7c86d7d515c3e2cc4c4b422263b3124089fab4d200546f.jpg)

![c665ca7cfcae51820ff94f1ae78280b0a0a381dda78d59c0dea40fe9c421cf8c.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/c665ca7cfcae51820ff94f1ae78280b0a0a381dda78d59c0dea40fe9c421cf8c.jpg)

![d8db18fc41865026832c31cd287d2f8524b017ef9b638523b5db4c51e48d339d.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/d8db18fc41865026832c31cd287d2f8524b017ef9b638523b5db4c51e48d339d.jpg)

![df725293682309d7251975ec0ae80a3e668161f482045333e4ebf5b8f4f16cc5.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/df725293682309d7251975ec0ae80a3e668161f482045333e4ebf5b8f4f16cc5.jpg)

![e43847f7f9a5e9d999aeb19eb613f8b58dcbe4bccecb7d8d2c31f2780078d298.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/e43847f7f9a5e9d999aeb19eb613f8b58dcbe4bccecb7d8d2c31f2780078d298.jpg)

![e9662a5909bfc735ae62ae10f67e01d4ba1d94874c681d3e9426664b7de111c1.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/e9662a5909bfc735ae62ae10f67e01d4ba1d94874c681d3e9426664b7de111c1.jpg)

![fb04d85f130756b5abc4e54b67e64e7a47acdbc6cc8e6539261ff06b065cde0e.jpg](../icml_results/2139_Foundation%20Inference%20Models%20for%20Ordinary%20Differential%20Equations/tables/fb04d85f130756b5abc4e54b67e64e7a47acdbc6cc8e6539261ff06b065cde0e.jpg)

<a id="paper-2112"></a>

## Reinforcement Learning via Self-Distillation

### Images

![029e4515a1b936f9603c4427e82bc703ee459b77e8234cdbad8baa455df5087c.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/029e4515a1b936f9603c4427e82bc703ee459b77e8234cdbad8baa455df5087c.jpg)

![2357245ffcc2b4bdeba1db6f394f40516aa89bf5e8f24b65937645fb4b4230a1.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/2357245ffcc2b4bdeba1db6f394f40516aa89bf5e8f24b65937645fb4b4230a1.jpg)

![25d200df2287c26f2c02f994b5412adb45e0808d098d6ea1f8c6dbce34aaa868.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/25d200df2287c26f2c02f994b5412adb45e0808d098d6ea1f8c6dbce34aaa868.jpg)

![3c40c35483b595c181d352f0f156dc9922419da5ee422126f5e3c97442b93fdb.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/3c40c35483b595c181d352f0f156dc9922419da5ee422126f5e3c97442b93fdb.jpg)

![47cf00690d9cd5f0302fd73e1110bb06f20cd977b7dec384172dd8ddb59aa721.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/47cf00690d9cd5f0302fd73e1110bb06f20cd977b7dec384172dd8ddb59aa721.jpg)

![49b4f593e2fc4299cfb05a001df52d3d802057caf269a830d96f6d5bc6011638.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/49b4f593e2fc4299cfb05a001df52d3d802057caf269a830d96f6d5bc6011638.jpg)

![5255aabf45e36c1aed1409c0578af33c778a4d03c015168b6a67374c284dc0af.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/5255aabf45e36c1aed1409c0578af33c778a4d03c015168b6a67374c284dc0af.jpg)

![528dc86acf93df1232db1d6f83d0663b6bb6441a734661c43ced1d76ce1cee24.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/528dc86acf93df1232db1d6f83d0663b6bb6441a734661c43ced1d76ce1cee24.jpg)

![6d2f97f8ba4fbafbd86fa86d19835586757ce0d73d5a4b77f0205188cf612ce7.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/6d2f97f8ba4fbafbd86fa86d19835586757ce0d73d5a4b77f0205188cf612ce7.jpg)

![767f970c72b59fa3d5d7a35c43c72a9a5bcfe2a9c7988e2e869717b5fc49b4e7.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/767f970c72b59fa3d5d7a35c43c72a9a5bcfe2a9c7988e2e869717b5fc49b4e7.jpg)

![866687c8a9ce3f3be9d48427ff33b8f34c2315d5009c6af2c9c323ccd416d1f3.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/866687c8a9ce3f3be9d48427ff33b8f34c2315d5009c6af2c9c323ccd416d1f3.jpg)

![8d696f45fe6a4c9eb0a3b3d8ef9e79e7413e876ffa55437cd1ab0e8d4f9ca024.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/8d696f45fe6a4c9eb0a3b3d8ef9e79e7413e876ffa55437cd1ab0e8d4f9ca024.jpg)

![90f6bc937251588a93253e79f673d3b7c70820558d6fc5fbdb5576d8542ec21b.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/90f6bc937251588a93253e79f673d3b7c70820558d6fc5fbdb5576d8542ec21b.jpg)

![9445111779010de02f0f8b2270cd363614d559abba547ade1228e5070a0aff6d.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/9445111779010de02f0f8b2270cd363614d559abba547ade1228e5070a0aff6d.jpg)

![9be4fb5ba45378d415c19390073e436eaad1b34d47bbb5cb91f536a7f1c6621e.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/9be4fb5ba45378d415c19390073e436eaad1b34d47bbb5cb91f536a7f1c6621e.jpg)

![bb7d0f2074ee6494b8e2d989c1be6d3d05e15fef08a543c865049f976abbe503.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/bb7d0f2074ee6494b8e2d989c1be6d3d05e15fef08a543c865049f976abbe503.jpg)

![dacda730c85d65f5e50d804868a529b93664d61de75d3950cb4d1c2057712b40.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/dacda730c85d65f5e50d804868a529b93664d61de75d3950cb4d1c2057712b40.jpg)

![e13529d0322715f235309334c7867bce8e8565fdb94592239dd84c9f98469f63.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/e13529d0322715f235309334c7867bce8e8565fdb94592239dd84c9f98469f63.jpg)

![e61bb209d7bd29421e12029ad4ab36bf531f5b8ef6762d0f355736bc69c5d142.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/e61bb209d7bd29421e12029ad4ab36bf531f5b8ef6762d0f355736bc69c5d142.jpg)

![f4d178b65b47d17ad6af3bb88e009af01482727f7ff59182da323115a53953c7.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/images/f4d178b65b47d17ad6af3bb88e009af01482727f7ff59182da323115a53953c7.jpg)

### Tables

![10a6cf9ee7f245cfec7969683acc5ebc24a153e67cb56ce1b6557f5d7efee21b.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/10a6cf9ee7f245cfec7969683acc5ebc24a153e67cb56ce1b6557f5d7efee21b.jpg)

![1c110c427258e1d117ef6b41b1a22827ff8c4fdd79eb19b34651f80d2788450a.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/1c110c427258e1d117ef6b41b1a22827ff8c4fdd79eb19b34651f80d2788450a.jpg)

![1fd4ea496d24878fb9a98c2107ef3e423e45db4bb4df87ed4b81c95540a26457.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/1fd4ea496d24878fb9a98c2107ef3e423e45db4bb4df87ed4b81c95540a26457.jpg)

![24168c8de32827a526e8ae39833bc154f44e9bcce486c255930c859961366617.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/24168c8de32827a526e8ae39833bc154f44e9bcce486c255930c859961366617.jpg)

![410c3495d38c7bd0a605f8e35313a433487a7f5ff9ad033967d451963393fefb.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/410c3495d38c7bd0a605f8e35313a433487a7f5ff9ad033967d451963393fefb.jpg)

![7a35ed26e0b1fea1ba3a5b4361a9fa0e7a78bded1746dea7846c504cd9fbd9ca.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/7a35ed26e0b1fea1ba3a5b4361a9fa0e7a78bded1746dea7846c504cd9fbd9ca.jpg)

![7fba128c9f3a4d59d87ac509adf27dc846e7adfa0517529bb48eb16d23137ed5.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/7fba128c9f3a4d59d87ac509adf27dc846e7adfa0517529bb48eb16d23137ed5.jpg)

![94ad850d4eb35c10abfcf2797d712a18619054639665ce26c27baf1ad20814fb.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/94ad850d4eb35c10abfcf2797d712a18619054639665ce26c27baf1ad20814fb.jpg)

![c2b0c3238f5dba234fcb4c093aef4c8379d96596704f5da45154628f4696860a.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/c2b0c3238f5dba234fcb4c093aef4c8379d96596704f5da45154628f4696860a.jpg)

![d2676fbf01ba09e073b2de17afea3705f903d1162526e80d20f3ec3711036deb.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/d2676fbf01ba09e073b2de17afea3705f903d1162526e80d20f3ec3711036deb.jpg)

![d882be046efb7f455a8827e67dd719de06d1a0c667fe9b041319a7f4015569c1.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/d882be046efb7f455a8827e67dd719de06d1a0c667fe9b041319a7f4015569c1.jpg)

![efae0da729525a385643f7727ee2ac389ef7ed41f161002f9a99ba550453bc63.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/efae0da729525a385643f7727ee2ac389ef7ed41f161002f9a99ba550453bc63.jpg)

![f2cfc1b24d521721516e219c3b1deb9be7e5e98c2066a5f80f583706427022f3.jpg](../icml_results/2140_Reinforcement%20Learning%20via%20Self-Distillation/tables/f2cfc1b24d521721516e219c3b1deb9be7e5e98c2066a5f80f583706427022f3.jpg)
