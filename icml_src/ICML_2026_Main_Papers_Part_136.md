# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 136 of 197

## 目录 (Table of Contents)

1. [GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy](#paper-4456)
2. [Semi-Supervised Gaze Estimation via Disentangled Subspace Contrastive Learning](#paper-4457)
3. [Breaking the Self-Confirming Loop: Diagnosing and Mitigating Systemic Reward Bias in Self-Rewarding RL](#paper-4458)
4. [Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases](#paper-4459)
5. [See, Act, Adapt: Active Perception for Unsupervised Cross-Domain Visual Adaptation via Personalized VLM-Guided Agent](#paper-4460)
6. [Federated Graph Learning via Structure-Aware Fusion Using a Kalman Framework with Learnable Dynamics](#paper-4461)
7. [CRPO: Character-centric Group Relative Policy Optimization for Role-aware Reasoning in Role-playing Agents](#paper-4462)
8. [Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?](#paper-4463)
9. [RAST-MoE-RL: A Regime-Aware Spatio-Temporal MoE Framework for Deep Reinforcement Learning in Ride-Hailing](#paper-4464)
10. [Escaping the Likelihood Trap: Geometric Diversity Optimization for Long-Form Image Captioning](#paper-4465)
11. [Identifying and Mitigating Errors in Gradient Aggregation of Distributed Data Parallel Training](#paper-4466)
12. [Towards Efficient Large Language Reasoning Models via Extreme-Ratio Chain-of-Thought Compression](#paper-4467)
13. [Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction](#paper-4468)
14. [PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization](#paper-4469)
15. [Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale](#paper-4470)
16. [Reconstruction Outcomes Look Similar but Processes Differ: Improving Context Consistency and Coverage in Graph Masked Auto-Encoder](#paper-4471)
17. [Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](#paper-4472)
18. [ReLAM: Learning Anticipation Model for Rewarding Visual Robotic Manipulation](#paper-4473)
19. [Glimpse: Geometry Learning of Multi-scale Structural Priors for 3D Pose Estimation](#paper-4474)
20. [When Generalized Zero-Shot Learning Meets PU Learning: A Plug-and-Play Framework for Seen-Class Bias Mitigation](#paper-4475)
21. [Federated Data and Feature Selection by Generalized CUR Decomposition](#paper-4476)
22. [The Extra Tokens Matter: Disentangled Representation Learning with Vision Transformers](#paper-4477)
23. [Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers](#paper-4478)
24. [Artemis: Structured Visual Reasoning for Perception Policy Learning](#paper-4479)
25. [Rational Neural Networks have Expressivity Advantages](#paper-4480)
26. [MineDraft: A Framework for Batch Parallel Speculative Decoding](#paper-4481)
27. [Learning to Rank by Directly Optimizing Full-Order Probabilities](#paper-4482)
28. [SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection](#paper-4483)
29. [Thinking in Latent Space: Progressive Multimodal Simplification for Visual Reasoning](#paper-4484)
30. [CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling](#paper-4485)
31. [RSPO: Regularized Self-Play Alignment of Large Language Models](#paper-4486)
32. [Euclean: Automated Geometry Problem Formalization with Unified Verification in Lean](#paper-4487)
33. [Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models](#paper-4488)

---


<a id="paper-4456"></a>

## GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy

### Images

![031fecb6975af763cdbb1c83a9a000c3031f723fdd1bc39d20f1656d1cde5eae.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/031fecb6975af763cdbb1c83a9a000c3031f723fdd1bc39d20f1656d1cde5eae.jpg)

![3ada460b8d8226857cc92181ca57fa6667ae1d943b0b4a14ebd4b7cbc8f73c0c.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/3ada460b8d8226857cc92181ca57fa6667ae1d943b0b4a14ebd4b7cbc8f73c0c.jpg)

![6834703a58cfdca7de64daf09c2070c8b8694a71e921e470d6643eb43ad38853.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/6834703a58cfdca7de64daf09c2070c8b8694a71e921e470d6643eb43ad38853.jpg)

![70ac431e68e3c65f49747ffa537316a606b3b3fd80b2860986bcf8e84f57f66e.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/70ac431e68e3c65f49747ffa537316a606b3b3fd80b2860986bcf8e84f57f66e.jpg)

![98c99f4f4667b7840e97ccf477e1cd9d218f1cc0731e0421b0c70e1985f49045.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/98c99f4f4667b7840e97ccf477e1cd9d218f1cc0731e0421b0c70e1985f49045.jpg)

![b76ecca1e8070d05630f1367b80a4a3d5657394038149c8519701097b132aae2.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/b76ecca1e8070d05630f1367b80a4a3d5657394038149c8519701097b132aae2.jpg)

![b823cc0cfb4f99bb7214804d7570c5ef445662f9f995affd7b72d94f824faabb.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/b823cc0cfb4f99bb7214804d7570c5ef445662f9f995affd7b72d94f824faabb.jpg)

![c1206f17e04327daf0dcee54287406ce90e31a2569dec465807086c4a2689d81.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/c1206f17e04327daf0dcee54287406ce90e31a2569dec465807086c4a2689d81.jpg)

![d706bcacec6cb69029c62b45c449f22f2795de63eb0ccb76b5487f0848b21fdc.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/d706bcacec6cb69029c62b45c449f22f2795de63eb0ccb76b5487f0848b21fdc.jpg)

![da60f170b4a216f6defb08cd74a072a95d1dd97f941b32c74444a53e1a8fdf83.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/da60f170b4a216f6defb08cd74a072a95d1dd97f941b32c74444a53e1a8fdf83.jpg)

![ff596cec3c820a4d73437ff8fd7bf3ea509f177f0c4784d2e1f9308e15aaf573.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/images/ff596cec3c820a4d73437ff8fd7bf3ea509f177f0c4784d2e1f9308e15aaf573.jpg)

### Tables

![0064646731b00954c54533b25ba95402abdaa989663f73b6e53964d669a29c5c.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/0064646731b00954c54533b25ba95402abdaa989663f73b6e53964d669a29c5c.jpg)

![1b0252ffee8441af0cb96003ff1ff9e9bd03b22906761b6275585016bf413624.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/1b0252ffee8441af0cb96003ff1ff9e9bd03b22906761b6275585016bf413624.jpg)

![21d07b314c4bb7af3a36117e7a9bcd89b74d6b23e405f326a65a0baf4f2bc5d7.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/21d07b314c4bb7af3a36117e7a9bcd89b74d6b23e405f326a65a0baf4f2bc5d7.jpg)

![7a6cfb7fc4796b9efb534432c6bc07225a882c63cc78ef6803bf4b20a6b1ee12.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/7a6cfb7fc4796b9efb534432c6bc07225a882c63cc78ef6803bf4b20a6b1ee12.jpg)

![7bdfb2d78804a9b261187b9b5640dcb4b2c7bd877e0e56216b6b9d52e0812190.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/7bdfb2d78804a9b261187b9b5640dcb4b2c7bd877e0e56216b6b9d52e0812190.jpg)

![93a122ca9f47e18aaf164486ccce4b0822d2331dd84b08c0bf4ae1c2a08dc822.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/93a122ca9f47e18aaf164486ccce4b0822d2331dd84b08c0bf4ae1c2a08dc822.jpg)

![a2ee54fc60eebdb27eb70cd98a1cc30388df8ae19242b90e21429609389b40aa.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/a2ee54fc60eebdb27eb70cd98a1cc30388df8ae19242b90e21429609389b40aa.jpg)

![c64f5023107a2237afc08a63651d8a1c78cc0c6b40144df7a8437beb57264aab.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/c64f5023107a2237afc08a63651d8a1c78cc0c6b40144df7a8437beb57264aab.jpg)

![cb5b199d146e4cf9af32fdd98fbdf548017cc3b23a406319519fd276ac0a7bcf.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/cb5b199d146e4cf9af32fdd98fbdf548017cc3b23a406319519fd276ac0a7bcf.jpg)

![ea503fb3ec2a52984305c80487e9263b99ec5c2099b13aa02a26bb048fc19755.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/ea503fb3ec2a52984305c80487e9263b99ec5c2099b13aa02a26bb048fc19755.jpg)

![fd382cf8bcbe367bac080f32945ba59049e039846543573b8ddb191a51430ca2.jpg](../icml_results/4502_Restoring%20Exploration%20after%20Post-Training_%20Latent%20Exploration%20Decoding%20for%20Large%20Reasoning%20Models/tables/fd382cf8bcbe367bac080f32945ba59049e039846543573b8ddb191a51430ca2.jpg)

## GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy


### Images

![0e5a92766536a5ceaea707e272e4a2c69f0d24ae0dff15823bcfd3190e0adee5.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/0e5a92766536a5ceaea707e272e4a2c69f0d24ae0dff15823bcfd3190e0adee5.jpg)

![10709d0a363caf05e82c779ed876ca18254eccd96d396ddc89584c73af60f555.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/10709d0a363caf05e82c779ed876ca18254eccd96d396ddc89584c73af60f555.jpg)

![2f9c4bda5ba2b42066d5395ea24421542270be2a4afdbadff2b99021db443050.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/2f9c4bda5ba2b42066d5395ea24421542270be2a4afdbadff2b99021db443050.jpg)

![824a495c1e54da6a8fcdc586692274819b373ab83387ea46647ef825696de8d8.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/824a495c1e54da6a8fcdc586692274819b373ab83387ea46647ef825696de8d8.jpg)

![862390b8dd96db26b49e3d0ca2e8af2e64f418b7f4cb21aa5968b817ce25cafb.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/862390b8dd96db26b49e3d0ca2e8af2e64f418b7f4cb21aa5968b817ce25cafb.jpg)

![9a1724c412d923b195babe470b6601da692e85c87de1051991453c99ec6e45d5.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/9a1724c412d923b195babe470b6601da692e85c87de1051991453c99ec6e45d5.jpg)

![a1a031c4a9b913ca9d2a12e32eb379db8862658f561125174c6b05c203a094aa.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/a1a031c4a9b913ca9d2a12e32eb379db8862658f561125174c6b05c203a094aa.jpg)

![aba2c206f003204eb4a8693d499c4eaa5042b3d7d2a7cfbeab2c5ea578771a22.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/aba2c206f003204eb4a8693d499c4eaa5042b3d7d2a7cfbeab2c5ea578771a22.jpg)

![bb0269d2a2192aed6e8bf0e29c5b3ea8d90fdbce01dcf04a5166cb9921397d48.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/bb0269d2a2192aed6e8bf0e29c5b3ea8d90fdbce01dcf04a5166cb9921397d48.jpg)

![bf1a7f79bc21a3ca2d255a4ae9e2e5d56a7d82164b2d8a4170d66eb6479f45f1.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/bf1a7f79bc21a3ca2d255a4ae9e2e5d56a7d82164b2d8a4170d66eb6479f45f1.jpg)

![c9fb1e2ee027c08cea6d3a6baa784c37acdfcb71ef1a6d7b46d933557a78503a.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/c9fb1e2ee027c08cea6d3a6baa784c37acdfcb71ef1a6d7b46d933557a78503a.jpg)

![d209530232c8938e0a2dfbb12f9264d73169c70c2cf5c8c735c5474302d53bbd.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/d209530232c8938e0a2dfbb12f9264d73169c70c2cf5c8c735c5474302d53bbd.jpg)

![fe7cd33569a88b5143645b411f7b00e5f2f262fe77488a674ab7bcb85937b76f.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/images/fe7cd33569a88b5143645b411f7b00e5f2f262fe77488a674ab7bcb85937b76f.jpg)

### Tables

![5d7ba42a230003d4639ae5ed0ecfbfadf25d3a12d0ecae3de8269b1ea1d0dd8a.jpg](../icml_results/4503_GTPO%20and%20GRPO-S_%20Token%20and%20Sequence-Level%20Reward%20Shaping%20with%20Policy%20Entropy/tables/5d7ba42a230003d4639ae5ed0ecfbfadf25d3a12d0ecae3de8269b1ea1d0dd8a.jpg)

<a id="paper-4457"></a>

## Semi-Supervised Gaze Estimation via Disentangled Subspace Contrastive Learning


### Images

![1bae2c0c3c8904822f71ee0dfb40b8d702ddd557a09dfe4379782e5f3ef4d91e.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/1bae2c0c3c8904822f71ee0dfb40b8d702ddd557a09dfe4379782e5f3ef4d91e.jpg)

![44fb55ea665cee0694720a747347e7b2353e3d6d26e119167b84a2996b165a75.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/44fb55ea665cee0694720a747347e7b2353e3d6d26e119167b84a2996b165a75.jpg)

![4bd77261fdf936e315159998869f382c0e09f10bb3e74b82b08768408b668aef.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/4bd77261fdf936e315159998869f382c0e09f10bb3e74b82b08768408b668aef.jpg)

![7758b2561c399b77273e7250fb088dc4767c40c0d5a0524a16ea64b3a3d66fe2.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/7758b2561c399b77273e7250fb088dc4767c40c0d5a0524a16ea64b3a3d66fe2.jpg)

![999e6c4612a7105f0bf4a42bdf6b106b0762aa6b32fea0a3aea4ef253ab4e3c2.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/999e6c4612a7105f0bf4a42bdf6b106b0762aa6b32fea0a3aea4ef253ab4e3c2.jpg)

![c4a32b717e69878cee8038af31fe4a6a839a2dee2b5156fc357b1657736d6712.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/c4a32b717e69878cee8038af31fe4a6a839a2dee2b5156fc357b1657736d6712.jpg)

![d21c15d40f67dd6348d2878753a3c0e55fe99bea4a5d0773b13d4705be9c2070.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/d21c15d40f67dd6348d2878753a3c0e55fe99bea4a5d0773b13d4705be9c2070.jpg)

![d384e5aa38733a205ffbb9dfa9d3b4785153884c204730eb17c05a8c65b40a9c.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/d384e5aa38733a205ffbb9dfa9d3b4785153884c204730eb17c05a8c65b40a9c.jpg)

![e28342fabc16cff3f41cf864d09ee81eff380e265712a9967bf84ba09de6740a.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/e28342fabc16cff3f41cf864d09ee81eff380e265712a9967bf84ba09de6740a.jpg)

![ef1b91946948eeb421b2dfc55524ad4b72d67058ea9a5550d0e34cbec15a7cf6.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/images/ef1b91946948eeb421b2dfc55524ad4b72d67058ea9a5550d0e34cbec15a7cf6.jpg)

### Tables

![0902fbf460f424e64c8273aa9e342967a7c230821a1d16b2af2791468b78d83f.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/0902fbf460f424e64c8273aa9e342967a7c230821a1d16b2af2791468b78d83f.jpg)

![3124f381f9737bdee0f6bc819471f2da2a6c9cd8ff688a425a6c4f0993b812a6.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/3124f381f9737bdee0f6bc819471f2da2a6c9cd8ff688a425a6c4f0993b812a6.jpg)

![352fac5a85184c25dcc4d19adbcb763673fa03a7fe5e1ab774a9fc7f5dd3f242.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/352fac5a85184c25dcc4d19adbcb763673fa03a7fe5e1ab774a9fc7f5dd3f242.jpg)

![4214e30675ef53f3cfc222103d4b3294088cba234f644eab98fa30e23965065e.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/4214e30675ef53f3cfc222103d4b3294088cba234f644eab98fa30e23965065e.jpg)

![608f9cc546a608771fcc9d3b6215c9d94dca198dbe37169fa6e3388e2c222d83.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/608f9cc546a608771fcc9d3b6215c9d94dca198dbe37169fa6e3388e2c222d83.jpg)

![7a5ba5aa7c99f4267b1e4630e0ce61e97b060672956b71e3ccd68fe097d5b5ba.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/7a5ba5aa7c99f4267b1e4630e0ce61e97b060672956b71e3ccd68fe097d5b5ba.jpg)

![846f0339cee758a3b244225c50df608f84898635166fc1d8e589820ba3d23fef.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/846f0339cee758a3b244225c50df608f84898635166fc1d8e589820ba3d23fef.jpg)

![87e45316821a459c00526e9afab9295bbd86b439ecc9f70077f7ce04bc8a7230.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/87e45316821a459c00526e9afab9295bbd86b439ecc9f70077f7ce04bc8a7230.jpg)

![f0d2df7252ab25d1d10f8010436fcda6eef345ce49527d0978eb8e9c18e3e7e5.jpg](../icml_results/4504_Semi-Supervised%20Gaze%20Estimation%20via%20Disentangled%20Subspace%20Contrastive%20Learning/tables/f0d2df7252ab25d1d10f8010436fcda6eef345ce49527d0978eb8e9c18e3e7e5.jpg)

<a id="paper-4458"></a>

## Breaking the Self-Confirming Loop: Diagnosing and Mitigating Systemic Reward Bias in Self-Rewarding RL


### Images

![120b78e160b85bb8e92df50e22a0563ae922e6e032a4d266b96a9bcc5e6aa787.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/120b78e160b85bb8e92df50e22a0563ae922e6e032a4d266b96a9bcc5e6aa787.jpg)

![17884dd50a5a231ef74e2d0f14e8f55aa7be92eb62a5605fa86f243567500df1.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/17884dd50a5a231ef74e2d0f14e8f55aa7be92eb62a5605fa86f243567500df1.jpg)

![4047d398f9381a33ad78913d13a97dbb5aba2cbb3304f28ba016de8a53433ed3.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/4047d398f9381a33ad78913d13a97dbb5aba2cbb3304f28ba016de8a53433ed3.jpg)

![7006158ce3e9664202e8822d3a497eb97d1f8363d242e102c0a7d1b39ac55639.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/7006158ce3e9664202e8822d3a497eb97d1f8363d242e102c0a7d1b39ac55639.jpg)

![96f036e556066ceea0d34a420b680300e1c9d71dd9f5cab97b35e880ee405723.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/96f036e556066ceea0d34a420b680300e1c9d71dd9f5cab97b35e880ee405723.jpg)

![a88d8495ef298e6fc14d49837051a11cdd69065338589ef92d1d18d7beecf3b7.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/a88d8495ef298e6fc14d49837051a11cdd69065338589ef92d1d18d7beecf3b7.jpg)

![ba0f5385653d87c45dedfd4f05ab7ededed000a911bf7aafa2295844f5b766c8.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/ba0f5385653d87c45dedfd4f05ab7ededed000a911bf7aafa2295844f5b766c8.jpg)

![d731688cad1068d9d0072db6e0dce99b8164a2f591bb4beb5c754f71b3fad3be.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/d731688cad1068d9d0072db6e0dce99b8164a2f591bb4beb5c754f71b3fad3be.jpg)

![db5fb91cb0c2fc997423b4e44a11f15bec7d086e8f026bc5123ff3993c0c222a.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/images/db5fb91cb0c2fc997423b4e44a11f15bec7d086e8f026bc5123ff3993c0c222a.jpg)

### Tables

![24c56b0316ec528ce6bc1f1ea28cdb4a543904d85b1e1834221772b5870ccdc6.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/24c56b0316ec528ce6bc1f1ea28cdb4a543904d85b1e1834221772b5870ccdc6.jpg)

![31f938c27947f3d5864d4245e2c2fb586ab1369530018aa08384e7e76e5c78f3.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/31f938c27947f3d5864d4245e2c2fb586ab1369530018aa08384e7e76e5c78f3.jpg)

![379ef707e4176aee07f215990449e9bb80d72fb464aa6a6ca2ef87ccb919626b.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/379ef707e4176aee07f215990449e9bb80d72fb464aa6a6ca2ef87ccb919626b.jpg)

![554476645926084dbcebda4832fb63e18811198832adb65ccd9d085c24a24d80.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/554476645926084dbcebda4832fb63e18811198832adb65ccd9d085c24a24d80.jpg)

![6d04d9c8ec4f904036b57ee0360c21eab52ed07092f52b805d6a2b2998591f30.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/6d04d9c8ec4f904036b57ee0360c21eab52ed07092f52b805d6a2b2998591f30.jpg)

![89d11dbff9be8f194657bfb42d7048d41a19b4a28b282b9d20e124aa39107feb.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/89d11dbff9be8f194657bfb42d7048d41a19b4a28b282b9d20e124aa39107feb.jpg)

![92c75ce4bdf81cc79d6e3c84d864fd80d54a3d741263a268bbe42952ab57c400.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/92c75ce4bdf81cc79d6e3c84d864fd80d54a3d741263a268bbe42952ab57c400.jpg)

![aa3e24599aa13d29b028d7b8266a89acc346c7d850aa0d925f9359a534db8757.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/aa3e24599aa13d29b028d7b8266a89acc346c7d850aa0d925f9359a534db8757.jpg)

![b1aefda1345513207b05a4eeec65916bd2011aa847d4c47be33c677093d39028.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/b1aefda1345513207b05a4eeec65916bd2011aa847d4c47be33c677093d39028.jpg)

![de4f2222838517cf0f8d8574e40a148b4d40493875e688e3a5acf5d45c418b50.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/de4f2222838517cf0f8d8574e40a148b4d40493875e688e3a5acf5d45c418b50.jpg)

![e9ed449c6ca4c745b83844ee9be67bbb48c0b5cd5507f13a18cd5ad48e5855b5.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/e9ed449c6ca4c745b83844ee9be67bbb48c0b5cd5507f13a18cd5ad48e5855b5.jpg)

![f09bfc92392c8c2a8c0baf04a2901bf87059551a32e9fbd6974644e70c1db68a.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/f09bfc92392c8c2a8c0baf04a2901bf87059551a32e9fbd6974644e70c1db68a.jpg)

![fbaa899e8d41290c8d6d24f92be9c1961a6b826123121e78fb44d78df61afc41.jpg](../icml_results/4505_Breaking%20the%20Self-Confirming%20Loop_%20Diagnosing%20and%20Mitigating%20Systemic%20Reward%20Bias%20in%20Self-Rewarding%20/tables/fbaa899e8d41290c8d6d24f92be9c1961a6b826123121e78fb44d78df61afc41.jpg)

<a id="paper-4459"></a>

## Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases


### Images

![36dde2db50bc872e333a13e3281b8d9e286116bdeae6e321629c28fd9bb8bee7.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/36dde2db50bc872e333a13e3281b8d9e286116bdeae6e321629c28fd9bb8bee7.jpg)

![513fcf6ccddc11c0b092e3b12d029e0d785cfd4ed9bff8d23fd016a4d956a9b3.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/513fcf6ccddc11c0b092e3b12d029e0d785cfd4ed9bff8d23fd016a4d956a9b3.jpg)

![5ffcc478ce4fbc858a6e2948d4db64083fb7f84bf01800408be9691fdec1fda8.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/5ffcc478ce4fbc858a6e2948d4db64083fb7f84bf01800408be9691fdec1fda8.jpg)

![a0aaa93666629f8fff29827fbc616cd064ef6369c270bc1b1015019eba5cad29.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/a0aaa93666629f8fff29827fbc616cd064ef6369c270bc1b1015019eba5cad29.jpg)

![dbc9560b4087e1edfd52582fe9e7b3af847eb5b891516fec80dec9df4d3f6da4.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/dbc9560b4087e1edfd52582fe9e7b3af847eb5b891516fec80dec9df4d3f6da4.jpg)

![f22d905926a1fb8c99821e1528348d06802c754c5837e2839866dda6ae405f7d.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/f22d905926a1fb8c99821e1528348d06802c754c5837e2839866dda6ae405f7d.jpg)

![f3ba9b8675c0572900d8412afb3d50111168adc552a3c0f30e26ea3a09005a31.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/images/f3ba9b8675c0572900d8412afb3d50111168adc552a3c0f30e26ea3a09005a31.jpg)

### Tables

![076d28e1ce9f7ee70e20490c1a11333239110939a2272c91322b53fb5e00263e.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/076d28e1ce9f7ee70e20490c1a11333239110939a2272c91322b53fb5e00263e.jpg)

![2da0b452c23e9b488ae24c969a1d9cad6ef3a3bbd9187e416a1ef9a1d735ed7c.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/2da0b452c23e9b488ae24c969a1d9cad6ef3a3bbd9187e416a1ef9a1d735ed7c.jpg)

![4811e97a91e014bb875a7f52fb7ee0673455cf94448441440d8b238facd71568.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/4811e97a91e014bb875a7f52fb7ee0673455cf94448441440d8b238facd71568.jpg)

![674c7858875774d78b27e77d93462f2dbfc6d0d3806a493420408b189a9b2265.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/674c7858875774d78b27e77d93462f2dbfc6d0d3806a493420408b189a9b2265.jpg)

![80c673a813e55bda5de4e3e5f3e8b07f545a704c290558a91da95744b6e75ace.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/80c673a813e55bda5de4e3e5f3e8b07f545a704c290558a91da95744b6e75ace.jpg)

![8e0bfdfe13069fc7f0f58784855f64efc0c6d0ad8af60e7acb442e7fdfc7cb42.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/8e0bfdfe13069fc7f0f58784855f64efc0c6d0ad8af60e7acb442e7fdfc7cb42.jpg)

![90315e9b45d809ecc18d0712af08d842513e274758b4a2fa31ca937073352140.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/90315e9b45d809ecc18d0712af08d842513e274758b4a2fa31ca937073352140.jpg)

![bdc4a8102a10d92875906c6c2304d518c72333c024e1c65ee9ceeae0e42cbe53.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/bdc4a8102a10d92875906c6c2304d518c72333c024e1c65ee9ceeae0e42cbe53.jpg)

![c5fede61fbcc8cd40cad73bed9968f4914d47e6b4c5a627dd8154e11f5646c20.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/c5fede61fbcc8cd40cad73bed9968f4914d47e6b4c5a627dd8154e11f5646c20.jpg)

![e88f8b7fc720aa191dacd2f890d678c66fea189844cad3e8d86a052e5c077a00.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/e88f8b7fc720aa191dacd2f890d678c66fea189844cad3e8d86a052e5c077a00.jpg)

![eb753bd9eb94b11e1dc882fa4a21f2701dbaf2bc80292a2219d015ca31518964.jpg](../icml_results/4506_Sonar-TS_%20Search-Then-Verify%20Natural%20Language%20Querying%20for%20Time%20Series%20Databases/tables/eb753bd9eb94b11e1dc882fa4a21f2701dbaf2bc80292a2219d015ca31518964.jpg)

<a id="paper-4460"></a>

## See, Act, Adapt: Active Perception for Unsupervised Cross-Domain Visual Adaptation via Personalized VLM-Guided Agent


### Images

![08bccf081129987a52e9ce91491389c3f40d8967063d465848d742e2dd8055da.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/08bccf081129987a52e9ce91491389c3f40d8967063d465848d742e2dd8055da.jpg)

![295aeb7686755f778d902e3d78eacba5517256403e25d190eb979e2c3ade4e5a.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/295aeb7686755f778d902e3d78eacba5517256403e25d190eb979e2c3ade4e5a.jpg)

![325031262d95b1f3b9c5e8fd915ac86b874d4606192350c46caa6b0d13291101.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/325031262d95b1f3b9c5e8fd915ac86b874d4606192350c46caa6b0d13291101.jpg)

![37b32b67dc608600c74cd61f64d488d8f76d930244a75faf7d1dd4f816161dab.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/37b32b67dc608600c74cd61f64d488d8f76d930244a75faf7d1dd4f816161dab.jpg)

![3e5906fb30d3c2f583bd9338431ad979450b297a949315167b897c08e4222fd0.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/3e5906fb30d3c2f583bd9338431ad979450b297a949315167b897c08e4222fd0.jpg)

![51f25f3d9850ffb128454bbe2e83614d2141f6a6d1aae49470b3cb48a3e0bac8.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/51f25f3d9850ffb128454bbe2e83614d2141f6a6d1aae49470b3cb48a3e0bac8.jpg)

![8123b442bc476b929b81554fb24c91dcc656b0e05764557c96009e6006839e06.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/8123b442bc476b929b81554fb24c91dcc656b0e05764557c96009e6006839e06.jpg)

![82a3cf6ab06186236fd2ed926e319468afb2e5aaa3d0ae40eaa97df753f81b22.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/82a3cf6ab06186236fd2ed926e319468afb2e5aaa3d0ae40eaa97df753f81b22.jpg)

![93bafbe1975dcba932a2bf30a493ed064a1e35b58d89312e61877a9d5436201a.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/93bafbe1975dcba932a2bf30a493ed064a1e35b58d89312e61877a9d5436201a.jpg)

![b05ab4451562d4d7fae1ab59a7917a673b1a2a74925bf3bd1cac3ad8d0ef753b.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/b05ab4451562d4d7fae1ab59a7917a673b1a2a74925bf3bd1cac3ad8d0ef753b.jpg)

![e3ea697f2a6f5c85a2ff3b0d13c885d54f9dc22701390025ebe3a764cc5d6a64.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/images/e3ea697f2a6f5c85a2ff3b0d13c885d54f9dc22701390025ebe3a764cc5d6a64.jpg)

### Tables

![2a67adf07c960440fab6fec8d01f8d3bf443f1473136cf0e74008e251a313be6.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/2a67adf07c960440fab6fec8d01f8d3bf443f1473136cf0e74008e251a313be6.jpg)

![30619820e80d9071f5a695d929d8598075c93b52199d52c9a7f85bb09f82e1e3.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/30619820e80d9071f5a695d929d8598075c93b52199d52c9a7f85bb09f82e1e3.jpg)

![481eb5d2ecdea4f29af40a5b7ea1fdecde7c43c844c8701cba800669e48a712a.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/481eb5d2ecdea4f29af40a5b7ea1fdecde7c43c844c8701cba800669e48a712a.jpg)

![4b9659367ff0b535dadbd87fe0c29b557665787fb010a684dedb5d5396209c4b.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/4b9659367ff0b535dadbd87fe0c29b557665787fb010a684dedb5d5396209c4b.jpg)

![502e9e1fd7d53a7e223fde8c2bddbfe88de937c74585521fc762dc01f65bf0ae.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/502e9e1fd7d53a7e223fde8c2bddbfe88de937c74585521fc762dc01f65bf0ae.jpg)

![57e50860f18d8600dc11d3a4ee48e6a30f950e445dcbf94b957758d710edc91d.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/57e50860f18d8600dc11d3a4ee48e6a30f950e445dcbf94b957758d710edc91d.jpg)

![7a98507ec5dc591d751887ba1151f2abed88b22983d7eb657f4c0de3906c2f85.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/7a98507ec5dc591d751887ba1151f2abed88b22983d7eb657f4c0de3906c2f85.jpg)

![7f3a99e05cf734ed52eb2e08e35a85c0fb9c456b7f9b3033d991eaeaec9f8b21.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/7f3a99e05cf734ed52eb2e08e35a85c0fb9c456b7f9b3033d991eaeaec9f8b21.jpg)

![8437f99f8f9898169567dc35ba9c875dc27141832a9c7193ee43188a860c794c.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/8437f99f8f9898169567dc35ba9c875dc27141832a9c7193ee43188a860c794c.jpg)

![b16e00b7c0424a4c9dd88ef1c096a71f4b78f13d59b7d777f3bb6ac43ef746a0.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/b16e00b7c0424a4c9dd88ef1c096a71f4b78f13d59b7d777f3bb6ac43ef746a0.jpg)

![b54f8866aa5cef98bc8d209d4af6fa5d15887f3cb3553d3e2cc013b7e50eb9e2.jpg](../icml_results/4507_See%2C%20Act%2C%20Adapt_%20Active%20Perception%20for%20Unsupervised%20Cross-Domain%20Visual%20Adaptation%20via%20Personalized%20/tables/b54f8866aa5cef98bc8d209d4af6fa5d15887f3cb3553d3e2cc013b7e50eb9e2.jpg)

<a id="paper-4461"></a>

## Federated Graph Learning via Structure-Aware Fusion Using a Kalman Framework with Learnable Dynamics


### Images

![2691e3fd4c7776bd481ff786d6cff0c1f4c8a2446cf2b7ad36e5b7a26073fea0.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/2691e3fd4c7776bd481ff786d6cff0c1f4c8a2446cf2b7ad36e5b7a26073fea0.jpg)

![2d562afe0a51e1e17e1cbddc76ee9208c55979596f2a13070d1bfe78d5509127.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/2d562afe0a51e1e17e1cbddc76ee9208c55979596f2a13070d1bfe78d5509127.jpg)

![2fef8f45b7fe18a10fcda666ae121fa192a15110f12c3196eef7095324a6ad7c.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/2fef8f45b7fe18a10fcda666ae121fa192a15110f12c3196eef7095324a6ad7c.jpg)

![3ce4fe578bc0cd90649e06431ff72aa2cf64de501f3ef81f80ef9f8ad0bccbaa.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/3ce4fe578bc0cd90649e06431ff72aa2cf64de501f3ef81f80ef9f8ad0bccbaa.jpg)

![61fbd8fb4de9c9a2dda5aefa5913648416b01c388642a4fcc9158c166ead52f0.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/61fbd8fb4de9c9a2dda5aefa5913648416b01c388642a4fcc9158c166ead52f0.jpg)

![65fb9c664d99576e0a17915b342be8046b841bc7ba10ddf0da65aa75cd8abae8.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/65fb9c664d99576e0a17915b342be8046b841bc7ba10ddf0da65aa75cd8abae8.jpg)

![6774092cf10f94c4f0a9ff4b748b68b7026598ab810dbb4b3341c97d6692b0da.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/6774092cf10f94c4f0a9ff4b748b68b7026598ab810dbb4b3341c97d6692b0da.jpg)

![913662f92108b7eeed76b594190beed115029c2a8313a40993c769445fafe677.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/913662f92108b7eeed76b594190beed115029c2a8313a40993c769445fafe677.jpg)

![95b52c73cb247911aa8bd18c77655e749e10ba2973bc792e230b8296ac1c1b3c.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/95b52c73cb247911aa8bd18c77655e749e10ba2973bc792e230b8296ac1c1b3c.jpg)

![fe84706df375acdfa9bb158d2e2616455b5bad3e70f4cf9b4b43e4bbe9a26a34.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/images/fe84706df375acdfa9bb158d2e2616455b5bad3e70f4cf9b4b43e4bbe9a26a34.jpg)

### Tables

![124d35d0b090b2cd614237d66e139fe18340a21e7e9c6b75a4bd8d1a657109e5.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/124d35d0b090b2cd614237d66e139fe18340a21e7e9c6b75a4bd8d1a657109e5.jpg)

![136d1180e984ac9c548a1868017fe04ae5b521ba94ade4affcc3fd42ffbcbd57.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/136d1180e984ac9c548a1868017fe04ae5b521ba94ade4affcc3fd42ffbcbd57.jpg)

![140d0193f4317f6dac9c500e90a49198c221558894bd5c52e14f2333943091d5.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/140d0193f4317f6dac9c500e90a49198c221558894bd5c52e14f2333943091d5.jpg)

![3d5a9042b67995cab54584fcf16d457bf7300205c7007eece83f00b5245022b0.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/3d5a9042b67995cab54584fcf16d457bf7300205c7007eece83f00b5245022b0.jpg)

![3fbc2333b2807538a3df97d52718a4031eda8d2ac36536c3637429d9b5745a01.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/3fbc2333b2807538a3df97d52718a4031eda8d2ac36536c3637429d9b5745a01.jpg)

![40ad3ad569835563e99f50c7efa6b0d72a5f755057fbe78f70f9f79f864271ac.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/40ad3ad569835563e99f50c7efa6b0d72a5f755057fbe78f70f9f79f864271ac.jpg)

![4416d64f9e56409a6bd164b2676f4aabe20ce5b34bc9b77bc3b45a7356505dd5.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/4416d64f9e56409a6bd164b2676f4aabe20ce5b34bc9b77bc3b45a7356505dd5.jpg)

![48a01950acd29ecb70526080168789493e48ca54d063ea315ffd768abe5c7db4.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/48a01950acd29ecb70526080168789493e48ca54d063ea315ffd768abe5c7db4.jpg)

![4c242a80e1ea2e5f9a47c9796ab0e8e5770403779abe4f543dc13f5f16c1f256.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/4c242a80e1ea2e5f9a47c9796ab0e8e5770403779abe4f543dc13f5f16c1f256.jpg)

![6b5473bd9bd9c8283ffc8a4be21becc3bc1512af764732f27584fa852453221a.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/6b5473bd9bd9c8283ffc8a4be21becc3bc1512af764732f27584fa852453221a.jpg)

![6bcb20b8622178c687a120cea090db0aa5014bac2308221aba975643a73fceb3.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/6bcb20b8622178c687a120cea090db0aa5014bac2308221aba975643a73fceb3.jpg)

![87251933acdb2f4ec64f1f9f1662df1ea88eac4667197a427a5a0c8586ba0764.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/87251933acdb2f4ec64f1f9f1662df1ea88eac4667197a427a5a0c8586ba0764.jpg)

![8b5f0ed16b47b3408d287d19cf4c9a3b6cbbcc68bd6c1f1fc1eee454ba88f013.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/8b5f0ed16b47b3408d287d19cf4c9a3b6cbbcc68bd6c1f1fc1eee454ba88f013.jpg)

![9bd6960baf86f5f54292a70a25c5c1965bc996a6af4d00eda28ed71cd240c733.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/9bd6960baf86f5f54292a70a25c5c1965bc996a6af4d00eda28ed71cd240c733.jpg)

![d3a7282aa9685d2cdb69b8afbe347760f5299a1652b85986340ad78405e9d7d4.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/d3a7282aa9685d2cdb69b8afbe347760f5299a1652b85986340ad78405e9d7d4.jpg)

![d6bf3035a3acf8aaaaf5a8539f9ac6ed1ce06f88832d24344e6ab90dc6595717.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/d6bf3035a3acf8aaaaf5a8539f9ac6ed1ce06f88832d24344e6ab90dc6595717.jpg)

![f000132b38065faeef8fa1fd519616c96406fa3ff907afb69687b1a5858aa126.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/f000132b38065faeef8fa1fd519616c96406fa3ff907afb69687b1a5858aa126.jpg)

![fbed15346b27fbcc125f3507d4681a8c92f318b9c9d84463993385086d9ae0d9.jpg](../icml_results/4508_Federated%20Graph%20Learning%20via%20Structure-Aware%20Fusion%20Using%20a%20Kalman%20Framework%20with%20Learnable%20Dynamics/tables/fbed15346b27fbcc125f3507d4681a8c92f318b9c9d84463993385086d9ae0d9.jpg)

<a id="paper-4462"></a>

## CRPO: Character-centric Group Relative Policy Optimization for Role-aware Reasoning in Role-playing Agents


### Images

![0dd6442ab3907e18cb1a09e4a0b1453dc5397f2a2b9d2bad08a671a21ddbc741.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/0dd6442ab3907e18cb1a09e4a0b1453dc5397f2a2b9d2bad08a671a21ddbc741.jpg)

![21d70b050ca768e8397df3c8d2ff290511b1e16d5229b0eec084b3190e2fe863.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/21d70b050ca768e8397df3c8d2ff290511b1e16d5229b0eec084b3190e2fe863.jpg)

![4fb85fb44d45d5b7be86f7b9691b622efe0de44302f42773e288e8f3e871be15.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/4fb85fb44d45d5b7be86f7b9691b622efe0de44302f42773e288e8f3e871be15.jpg)

![6697992a6badd13d1eb6fcea1367b5da2102a36c732493bbc7180275b52fdfc5.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/6697992a6badd13d1eb6fcea1367b5da2102a36c732493bbc7180275b52fdfc5.jpg)

![69081134043d20074bd3526d55ad13e45a4a0e7980c3c5879d13fb88653dd913.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/69081134043d20074bd3526d55ad13e45a4a0e7980c3c5879d13fb88653dd913.jpg)

![9bff4519ea6b784f7d5a3633a2d4ced1e44e3359bc1cb2d6258b07f05edd4be1.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/9bff4519ea6b784f7d5a3633a2d4ced1e44e3359bc1cb2d6258b07f05edd4be1.jpg)

![bf79839d385f9c40f37df806374c22fd0b0e972cac1672aaa16eb2649d7066d2.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/bf79839d385f9c40f37df806374c22fd0b0e972cac1672aaa16eb2649d7066d2.jpg)

![dfe71f2e3efae56ab3b0686678c7dfb79ad5077a3a8d131758f5715a42803d98.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/images/dfe71f2e3efae56ab3b0686678c7dfb79ad5077a3a8d131758f5715a42803d98.jpg)

### Tables

![07e3b170d002f9416c4d12e4aae91650bada7afe36f943660f8ee8508bc8f5f1.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/07e3b170d002f9416c4d12e4aae91650bada7afe36f943660f8ee8508bc8f5f1.jpg)

![1095f0b32f6bc794021c6cd7de76fc38fe5a28ad6289ac103307f4bd3f40219c.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/1095f0b32f6bc794021c6cd7de76fc38fe5a28ad6289ac103307f4bd3f40219c.jpg)

![180b295bc9df1f412c437eedbfb9465f4ad0d12c5f03d213a3f21dd258d48455.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/180b295bc9df1f412c437eedbfb9465f4ad0d12c5f03d213a3f21dd258d48455.jpg)

![32693652579213572f66659e3f968d6781af22415b7ff1d6552d5a3d2e53e5ee.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/32693652579213572f66659e3f968d6781af22415b7ff1d6552d5a3d2e53e5ee.jpg)

![3d91b5ed393db00d35bc101ff2062d327b5d0746b97a3c4ff8292562519e5389.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/3d91b5ed393db00d35bc101ff2062d327b5d0746b97a3c4ff8292562519e5389.jpg)

![3ec989c4e528be3546452dddf3f183ba14533ecabd26a8d4b30f89595fba6ae8.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/3ec989c4e528be3546452dddf3f183ba14533ecabd26a8d4b30f89595fba6ae8.jpg)

![6c12fc6ce03d367ac689aa08496aedb62fc38737a1f956af637994e31b0729c7.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/6c12fc6ce03d367ac689aa08496aedb62fc38737a1f956af637994e31b0729c7.jpg)

![7224b8e9e8ccb43a8e6841587a16fc16b4e18a73a09cdda9af2153d05d6383f2.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/7224b8e9e8ccb43a8e6841587a16fc16b4e18a73a09cdda9af2153d05d6383f2.jpg)

![7a03b1cf04070c504df4554811a02b6f9f0e9b8b9deef0290007355ac65ff741.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/7a03b1cf04070c504df4554811a02b6f9f0e9b8b9deef0290007355ac65ff741.jpg)

![7a66407662a39cb22523c9aef3f5d82171fa4a7e95fd09dcb723cd5d29c08130.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/7a66407662a39cb22523c9aef3f5d82171fa4a7e95fd09dcb723cd5d29c08130.jpg)

![7d8079adfa0e4817e6be54d8231d174a1a0406ef0d24171543739c6cafde0b0d.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/7d8079adfa0e4817e6be54d8231d174a1a0406ef0d24171543739c6cafde0b0d.jpg)

![87c02e81ad70b09713bd9c99abd60e007b3401046b1a0c984bb43692ee4442e2.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/87c02e81ad70b09713bd9c99abd60e007b3401046b1a0c984bb43692ee4442e2.jpg)

![f48d1ca76b6ebee7bf8367248416a4e7a80024b7dc74b4b45a08dfb157b071a5.jpg](../icml_results/4509_CRPO_%20Character-centric%20Group%20Relative%20Policy%20Optimization%20for%20Role-aware%20Reasoning%20in%20Role-playing%20/tables/f48d1ca76b6ebee7bf8367248416a4e7a80024b7dc74b4b45a08dfb157b071a5.jpg)

<a id="paper-4463"></a>

## Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?


### Images

![08e27e4b4ef4c8d1cc2aaacc44f05680a6470830a04a49795da13cd3891192a6.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/08e27e4b4ef4c8d1cc2aaacc44f05680a6470830a04a49795da13cd3891192a6.jpg)

![26d0a2d44d3ad6568accb555b9e409cefea875bd0105c220e742467e205f2a48.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/26d0a2d44d3ad6568accb555b9e409cefea875bd0105c220e742467e205f2a48.jpg)

![32e377c90716e5ee97f7e9844d068cb768c19e72af7b36971757438890bc3da5.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/32e377c90716e5ee97f7e9844d068cb768c19e72af7b36971757438890bc3da5.jpg)

![3e86865a13bc70c78f2a84e6b3bf9277f4d7a4b652eb45d8c07abfe38fffe0fe.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/3e86865a13bc70c78f2a84e6b3bf9277f4d7a4b652eb45d8c07abfe38fffe0fe.jpg)

![40f0d7c3d87a1386617fd933f3e19e17ec8be31a99ed63f41719c9b84d882ec8.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/40f0d7c3d87a1386617fd933f3e19e17ec8be31a99ed63f41719c9b84d882ec8.jpg)

![5e93d19c047cf32a72df485b23755696913d5686ea45c5938716316588b3d247.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/5e93d19c047cf32a72df485b23755696913d5686ea45c5938716316588b3d247.jpg)

![637ade10224c4fad92328c737f9a71f5a1ee0f2c8d3bc9db81f7394f8e089655.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/637ade10224c4fad92328c737f9a71f5a1ee0f2c8d3bc9db81f7394f8e089655.jpg)

![834dddf7ae397bc2f0b967d63789364fa1d1ed22a8fb84e39428d666b5acd3b8.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/834dddf7ae397bc2f0b967d63789364fa1d1ed22a8fb84e39428d666b5acd3b8.jpg)

![85366fcf83cd11625d94e9a755a92a0205648f88cca8c39e50e2e70229222b9a.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/85366fcf83cd11625d94e9a755a92a0205648f88cca8c39e50e2e70229222b9a.jpg)

![8fca037a819066da7c295744a673324ccca3f71429c196d0a4624858690470ae.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/8fca037a819066da7c295744a673324ccca3f71429c196d0a4624858690470ae.jpg)

![93913156ab3be2ab022dbe0be3c307bb56d5dac04855d4da4d3ec1448f4eebbc.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/93913156ab3be2ab022dbe0be3c307bb56d5dac04855d4da4d3ec1448f4eebbc.jpg)

![95ceaae2fb760846483bde8a1cf5057d52176088a18de25895b461495f9200bc.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/95ceaae2fb760846483bde8a1cf5057d52176088a18de25895b461495f9200bc.jpg)

![aab473d3c9366554cdc0e6bb75302f082c9ff71240d48f54fdbc15078c6a78fe.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/aab473d3c9366554cdc0e6bb75302f082c9ff71240d48f54fdbc15078c6a78fe.jpg)

![ae38abdc72a4d80b4731848e7dfb05d4aa40cfe744dfb87a6341928f6320dc6d.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/ae38abdc72a4d80b4731848e7dfb05d4aa40cfe744dfb87a6341928f6320dc6d.jpg)

![b47eb873f9b7e3a24ec8aea6f6bc20a02966472344ef8e6704f4e6d933944674.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/b47eb873f9b7e3a24ec8aea6f6bc20a02966472344ef8e6704f4e6d933944674.jpg)

![b702a55f1dbbd32d37e2ad97ac10c50d73c4e3b3c8f744e7eab22a1614309177.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/b702a55f1dbbd32d37e2ad97ac10c50d73c4e3b3c8f744e7eab22a1614309177.jpg)

![c5018270f47ac38d4be1a258a565a344a22deb9943401b90f8a8e040590a0383.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/c5018270f47ac38d4be1a258a565a344a22deb9943401b90f8a8e040590a0383.jpg)

![db5ed0fd00e4811797f158272addb0cc0c74281e92adc93134f152b0dc527598.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/db5ed0fd00e4811797f158272addb0cc0c74281e92adc93134f152b0dc527598.jpg)

![e70d13d7fe9bd7ea66ef07df80ff2f72a39e2149866ee23037df968c06394ad2.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/images/e70d13d7fe9bd7ea66ef07df80ff2f72a39e2149866ee23037df968c06394ad2.jpg)

### Tables

![05caf03b5d1ecfe3f08cafaea1ddf3487fcd273e98a89f087799671968f8fd26.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/05caf03b5d1ecfe3f08cafaea1ddf3487fcd273e98a89f087799671968f8fd26.jpg)

![0a6be20d9b2f6385ae75d6814533e544ae08626a924a074a9a8bda56ee95bc46.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/0a6be20d9b2f6385ae75d6814533e544ae08626a924a074a9a8bda56ee95bc46.jpg)

![0d68ed3785abb15d517df1de5e5b4523036d0126b7454ab3512c5c4001e7bf55.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/0d68ed3785abb15d517df1de5e5b4523036d0126b7454ab3512c5c4001e7bf55.jpg)

![130c5f2eaab02c1bbe577e8188a1dd0975e562c6b6be53c410cf505cab97f441.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/130c5f2eaab02c1bbe577e8188a1dd0975e562c6b6be53c410cf505cab97f441.jpg)

![147c42513f4a587b1c6c21db904ade266b01be14e0c431ab07ab75219fa1fa7c.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/147c42513f4a587b1c6c21db904ade266b01be14e0c431ab07ab75219fa1fa7c.jpg)

![2eef79e68149f014ad4661aec9eaa69e05cc36c4b2f31cfbc2911acada9a9d9d.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/2eef79e68149f014ad4661aec9eaa69e05cc36c4b2f31cfbc2911acada9a9d9d.jpg)

![3c088b0707d5785176e438b72420b8dea8bc3b8f4d1c7de78993b07a9c90c629.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/3c088b0707d5785176e438b72420b8dea8bc3b8f4d1c7de78993b07a9c90c629.jpg)

![45d6f1d71c8be95e66ce8a6144bd997886b9b21b7eecc0ac0e877903690c067d.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/45d6f1d71c8be95e66ce8a6144bd997886b9b21b7eecc0ac0e877903690c067d.jpg)

![545a3d94bb1007a264a329872036bb0a620f72b302bf229032722d4770b49ae9.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/545a3d94bb1007a264a329872036bb0a620f72b302bf229032722d4770b49ae9.jpg)

![55cba6969c64e46d1d13a509d05c9a52c5c01d00cb2c699ce740791d853b90f1.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/55cba6969c64e46d1d13a509d05c9a52c5c01d00cb2c699ce740791d853b90f1.jpg)

![5a092e9ccbe5aa298441421b4a73433dd07fcdca97c3718ac08919726511de30.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/5a092e9ccbe5aa298441421b4a73433dd07fcdca97c3718ac08919726511de30.jpg)

![6196a01d2250deed09931ca054b12e0eab8275c7ba5386f38d1c387069af7cbe.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/6196a01d2250deed09931ca054b12e0eab8275c7ba5386f38d1c387069af7cbe.jpg)

![6cbfb21bff2d96142359855da782511809d91fc14a7c40203c63a59f19589c38.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/6cbfb21bff2d96142359855da782511809d91fc14a7c40203c63a59f19589c38.jpg)

![b55fb10ff19951d56a9f510f434090edac5f0f8ffc38a369c38647fd2ffb2cb3.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/b55fb10ff19951d56a9f510f434090edac5f0f8ffc38a369c38647fd2ffb2cb3.jpg)

![bfd3e67db05954c291aae11198af7f84b698e8268d4498f00a1ee2d1ce742635.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/bfd3e67db05954c291aae11198af7f84b698e8268d4498f00a1ee2d1ce742635.jpg)

![c15a34a5f4e40d2f2a205268c44d8c8066f7df1cc43a6b54767143ac721e7269.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/c15a34a5f4e40d2f2a205268c44d8c8066f7df1cc43a6b54767143ac721e7269.jpg)

![ed0fae46539c5264475b01a15045358ccde12bc81c51c909508f3db92b56170e.jpg](../icml_results/4510_Robust-U1_%20Can%20MLLMs%20Self-Recover%20Corrupted%20Visual%20Content%20for%20Robust%20Understanding_/tables/ed0fae46539c5264475b01a15045358ccde12bc81c51c909508f3db92b56170e.jpg)

<a id="paper-4464"></a>

## RAST-MoE-RL: A Regime-Aware Spatio-Temporal MoE Framework for Deep Reinforcement Learning in Ride-Hailing


### Images

![2440af8b60be9800717438fffe3b34608d62f5c3d88901381f5aa359e51c138a.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/2440af8b60be9800717438fffe3b34608d62f5c3d88901381f5aa359e51c138a.jpg)

![4cb0019c23714ad022567fd383a6ae1fe4427507ca21a0fb51e914659e1cd842.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/4cb0019c23714ad022567fd383a6ae1fe4427507ca21a0fb51e914659e1cd842.jpg)

![740aca9c0fb37030ae143a46f318dfcff77ab683f260e9f9c7507bd4f55d106f.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/740aca9c0fb37030ae143a46f318dfcff77ab683f260e9f9c7507bd4f55d106f.jpg)

![93cfa108e53eb48f538d44145f18e2742284e5d3314e08f724b591521cb893ff.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/93cfa108e53eb48f538d44145f18e2742284e5d3314e08f724b591521cb893ff.jpg)

![9ed95b7067f07b473d71a7882f916f6bbd99949535e270c6ebcd46614c7f3fe7.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/9ed95b7067f07b473d71a7882f916f6bbd99949535e270c6ebcd46614c7f3fe7.jpg)

![bf753048edf57aa4fa9d933e880c18df09d17ee174846f2fcd571469f18fea30.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/bf753048edf57aa4fa9d933e880c18df09d17ee174846f2fcd571469f18fea30.jpg)

![c49710c5c9a44cdc393297ddb5f665d60bc4241beac690fd2a4937aa71c3bee9.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/images/c49710c5c9a44cdc393297ddb5f665d60bc4241beac690fd2a4937aa71c3bee9.jpg)

### Tables

![15ca5668f67aacc3d115a7a20e61b8a97b8c9d34f878c4c30174984078850cd7.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/15ca5668f67aacc3d115a7a20e61b8a97b8c9d34f878c4c30174984078850cd7.jpg)

![282361062091fdf4005895b214f39afcedd9c6fdfa72d0bc43438426933bd223.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/282361062091fdf4005895b214f39afcedd9c6fdfa72d0bc43438426933bd223.jpg)

![33f66ef152de9fe3a224ff3985abe12c8285b3f716d186d2fb8c89c29c2b0795.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/33f66ef152de9fe3a224ff3985abe12c8285b3f716d186d2fb8c89c29c2b0795.jpg)

![6d9fe8436352af18e0fb70cb141f668c25eda99aa9cd38f3ddee85258ee62fb5.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/6d9fe8436352af18e0fb70cb141f668c25eda99aa9cd38f3ddee85258ee62fb5.jpg)

![86ef81c1453b9f1b598c798f4eafdecaf9f431dcc4858f9cd46c70d663c68986.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/86ef81c1453b9f1b598c798f4eafdecaf9f431dcc4858f9cd46c70d663c68986.jpg)

![98821e430ad5217023d2f220f0839cd575813df1b8aca7bc6dabf4c886056c7f.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/98821e430ad5217023d2f220f0839cd575813df1b8aca7bc6dabf4c886056c7f.jpg)

![c0329e7c54dfb8124762f32564c94a87ffc24a0a0cd16d437a8ca6f04645da9f.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/c0329e7c54dfb8124762f32564c94a87ffc24a0a0cd16d437a8ca6f04645da9f.jpg)

![e348186159a5600b91654bf92832fb4994ade7b85ba01cb66bf0e49ac9b6cc7d.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/e348186159a5600b91654bf92832fb4994ade7b85ba01cb66bf0e49ac9b6cc7d.jpg)

![e54d4bd40c86e2f2df2e2f8fb208fd479c8b6d768700e49587c9d746ced6fee4.jpg](../icml_results/4511_RAST-MoE-RL_%20A%20Regime-Aware%20Spatio-Temporal%20MoE%20Framework%20for%20Deep%20Reinforcement%20Learning%20in%20Ride-Ha/tables/e54d4bd40c86e2f2df2e2f8fb208fd479c8b6d768700e49587c9d746ced6fee4.jpg)

<a id="paper-4465"></a>

## Escaping the Likelihood Trap: Geometric Diversity Optimization for Long-Form Image Captioning


### Images

![3d05efb444b13fca6ecdd3ad441e715538e38821e25f8e895994658e3cf2ab26.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/images/3d05efb444b13fca6ecdd3ad441e715538e38821e25f8e895994658e3cf2ab26.jpg)

![8e8e4648065e4b0aee14bd615b044ff79976018dd5692c4394c777fe87d7f9eb.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/images/8e8e4648065e4b0aee14bd615b044ff79976018dd5692c4394c777fe87d7f9eb.jpg)

![9b0ed69231ef601e4e68907703fbdb1f649088a6649e97805f817061aaa45903.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/images/9b0ed69231ef601e4e68907703fbdb1f649088a6649e97805f817061aaa45903.jpg)

![ab656b699822141791675d2daa76ff729700af5a248b49b06d85375fc74672c6.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/images/ab656b699822141791675d2daa76ff729700af5a248b49b06d85375fc74672c6.jpg)

![ca0ae051781f4ddfab483072fb7a29e170261829b92c54ef1d8cfb3e0f706591.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/images/ca0ae051781f4ddfab483072fb7a29e170261829b92c54ef1d8cfb3e0f706591.jpg)

### Tables

![05888dd5429719f8bdb3de6eb0c742c2438aac95813884c58f5d7fb25f227095.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/05888dd5429719f8bdb3de6eb0c742c2438aac95813884c58f5d7fb25f227095.jpg)

![63e1099b5ee1355528eeb5af4c8f1f04910996da159439b88ca8f44242c3621f.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/63e1099b5ee1355528eeb5af4c8f1f04910996da159439b88ca8f44242c3621f.jpg)

![9413bc37a6d04b7c88b7f6f7ebc613f7a3f18ae6bcc1d329414ab7fa004eba9d.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/9413bc37a6d04b7c88b7f6f7ebc613f7a3f18ae6bcc1d329414ab7fa004eba9d.jpg)

![965e38a754b9c52d439976222a667916db47f14fb0145cf9b77730a28bb1cac5.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/965e38a754b9c52d439976222a667916db47f14fb0145cf9b77730a28bb1cac5.jpg)

![b0c13bc3a3ba1f61cc1106d41e45001a969c7b67465b9c173688ea7c60bc3b75.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/b0c13bc3a3ba1f61cc1106d41e45001a969c7b67465b9c173688ea7c60bc3b75.jpg)

![c55587372be5edf92d7aa6910058a085b1dbecf905086ce03ebab0f35b6b5d6d.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/c55587372be5edf92d7aa6910058a085b1dbecf905086ce03ebab0f35b6b5d6d.jpg)

![c7f911c57956625eb0ae9b4316719268883788a01e911984710d52b582da9431.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/c7f911c57956625eb0ae9b4316719268883788a01e911984710d52b582da9431.jpg)

![d21c150a4e84131a699193279112f6d6bbdb04507a7402a32a18c804c9344bc7.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/d21c150a4e84131a699193279112f6d6bbdb04507a7402a32a18c804c9344bc7.jpg)

![d3422c1a820d4c348601df3cd1ec5305ff2a7fcf339dacbaa91489f3f7c64889.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/d3422c1a820d4c348601df3cd1ec5305ff2a7fcf339dacbaa91489f3f7c64889.jpg)

![e3e26167a46ef4add9fccd5520ec224c4e27fcc5d54af8b7407edb2e3cc6ec40.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/e3e26167a46ef4add9fccd5520ec224c4e27fcc5d54af8b7407edb2e3cc6ec40.jpg)

![f0b44c7bc47b66af7c0c44219d83466b0eeeecb821cd20dfa27adb95484ef76e.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/f0b44c7bc47b66af7c0c44219d83466b0eeeecb821cd20dfa27adb95484ef76e.jpg)

![f7b97f0ae7edff5b08be4232e5bef312a7d2a72565b5f004f131a97bff956a78.jpg](../icml_results/4512_Escaping%20the%20Likelihood%20Trap_%20Geometric%20Diversity%20Optimization%20for%20Long-Form%20Image%20Captioning/tables/f7b97f0ae7edff5b08be4232e5bef312a7d2a72565b5f004f131a97bff956a78.jpg)

<a id="paper-4466"></a>

## Identifying and Mitigating Errors in Gradient Aggregation of Distributed Data Parallel Training


### Images

![13fc0cd41d1b05f8364780b28d7a81065b94409eaec2411924041ea28c3f94ee.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/13fc0cd41d1b05f8364780b28d7a81065b94409eaec2411924041ea28c3f94ee.jpg)

![440b4adb4cae56f5bb84ea84d523aa9a94faa1388d4e7102e0585f5284b29769.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/440b4adb4cae56f5bb84ea84d523aa9a94faa1388d4e7102e0585f5284b29769.jpg)

![4ce51749573c75f318f624734a4d10a25f2e44d99a34b42d26edf365c4a64cf7.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/4ce51749573c75f318f624734a4d10a25f2e44d99a34b42d26edf365c4a64cf7.jpg)

![534a498ad1d0ed01e7e27563cce20e33cfcddc56d42ac720668a9b61f7b960fb.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/534a498ad1d0ed01e7e27563cce20e33cfcddc56d42ac720668a9b61f7b960fb.jpg)

![579f3b4e53a759874345e53a2aebbe165e6a69ab0c97067b7e761e0011bacef5.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/579f3b4e53a759874345e53a2aebbe165e6a69ab0c97067b7e761e0011bacef5.jpg)

![5abb25e26fcf353abcfc9a72521c0f2294d1a74c5d3ba959721e5a055aaebc52.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/5abb25e26fcf353abcfc9a72521c0f2294d1a74c5d3ba959721e5a055aaebc52.jpg)

![6e055fb8d2349f4bd56df02c96bfb306308c5fce909ce7bc444523efb4779351.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/6e055fb8d2349f4bd56df02c96bfb306308c5fce909ce7bc444523efb4779351.jpg)

![79b44ec795fcebb65dfc5351269fc726d5d4a0cd2994113f043773f81b109713.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/79b44ec795fcebb65dfc5351269fc726d5d4a0cd2994113f043773f81b109713.jpg)

![865824a40b472cb2956fc4cf7a485563889e7222a4c630baf9cc4a3ce0ec37a3.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/865824a40b472cb2956fc4cf7a485563889e7222a4c630baf9cc4a3ce0ec37a3.jpg)

![988d956810b788493b1b2551ebe186462405d54faddc9fd5d89dd8017a8960ca.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/988d956810b788493b1b2551ebe186462405d54faddc9fd5d89dd8017a8960ca.jpg)

![98ff8d586decda92e75b6a1c6f909f0d9430b3b4f676cefeeb71470400f7ff98.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/98ff8d586decda92e75b6a1c6f909f0d9430b3b4f676cefeeb71470400f7ff98.jpg)

![a08f07430b61790acac01784c66a1f5efd78f486cab6ffd0df364a30d15b0edb.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/a08f07430b61790acac01784c66a1f5efd78f486cab6ffd0df364a30d15b0edb.jpg)

![b2bd5447f66745b9277108c4d00ecea4ea8444653d741cce3b15f280e39247c5.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/b2bd5447f66745b9277108c4d00ecea4ea8444653d741cce3b15f280e39247c5.jpg)

![c6eda480e073eac27024960e47f2767983824c8c19fb0a56716be8b9c47f46ae.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/c6eda480e073eac27024960e47f2767983824c8c19fb0a56716be8b9c47f46ae.jpg)

![cd58dd6451febde15c592b5ff440fffac383746107973e8d7b741eadbdf29844.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/cd58dd6451febde15c592b5ff440fffac383746107973e8d7b741eadbdf29844.jpg)

![cf1302156559b951cfc5ed2a3e1b6a138adf7fadeb5ca4bf4b964a70a4a50c1a.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/cf1302156559b951cfc5ed2a3e1b6a138adf7fadeb5ca4bf4b964a70a4a50c1a.jpg)

![d961a94406befd22df7706c301d60a046eb9a8973ba80b460697ee23c7d99a30.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/d961a94406befd22df7706c301d60a046eb9a8973ba80b460697ee23c7d99a30.jpg)

![df1a809dc2aa8e69bcb0c5b7c0ffd51ccb4f3825a2f4aa124bc0ad16195a4c06.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/df1a809dc2aa8e69bcb0c5b7c0ffd51ccb4f3825a2f4aa124bc0ad16195a4c06.jpg)

![dfc5683d01c5c02df75688e82e477fba54e94e22804b10147a5b52c77a9cfd29.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/images/dfc5683d01c5c02df75688e82e477fba54e94e22804b10147a5b52c77a9cfd29.jpg)

### Tables

![37ac9da8ecc728af9d7f19f489d353dfe5f3be514efe36d77a4a7b0f7edad235.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/tables/37ac9da8ecc728af9d7f19f489d353dfe5f3be514efe36d77a4a7b0f7edad235.jpg)

![6b021c70b9d5612a88c31e74d19f3ff28f469fa21f1514f64f0979e00ca03f36.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/tables/6b021c70b9d5612a88c31e74d19f3ff28f469fa21f1514f64f0979e00ca03f36.jpg)

![8e35c64a5b088ca87a7082c2264ce156b4ccc0f3e2e96f28d0c32109726dc6dc.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/tables/8e35c64a5b088ca87a7082c2264ce156b4ccc0f3e2e96f28d0c32109726dc6dc.jpg)

![b3a7702ce4543566850ab191948da69aaee3e63d140f3861879412d74eec2b05.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/tables/b3a7702ce4543566850ab191948da69aaee3e63d140f3861879412d74eec2b05.jpg)

![c7cc2a7c12056ee70f859fa447b403fe6b7689081915fc6338b932a55e28aa79.jpg](../icml_results/4513_Identifying%20and%20Mitigating%20Errors%20in%20Gradient%20Aggregation%20of%20Distributed%20Data%20Parallel%20Training/tables/c7cc2a7c12056ee70f859fa447b403fe6b7689081915fc6338b932a55e28aa79.jpg)

<a id="paper-4467"></a>

## Towards Efficient Large Language Reasoning Models via Extreme-Ratio Chain-of-Thought Compression


### Images

![176c38218ecafd1647da71383f091d500f1e4a9b62bc5d57515af47143fdb838.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/176c38218ecafd1647da71383f091d500f1e4a9b62bc5d57515af47143fdb838.jpg)

![252e05125ca2aea5e06485803063bbd58957988a048dae977f1fb210807ef203.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/252e05125ca2aea5e06485803063bbd58957988a048dae977f1fb210807ef203.jpg)

![280b0dfdd006da443bbadf3fe72bd7f719f7b784bd2af8db6d324cee952dce55.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/280b0dfdd006da443bbadf3fe72bd7f719f7b784bd2af8db6d324cee952dce55.jpg)

![6505bd03e924c3acf03bd29b2c94b275f23b875ffde35ac776023ef9a849b2dc.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/6505bd03e924c3acf03bd29b2c94b275f23b875ffde35ac776023ef9a849b2dc.jpg)

![9bd729458036c95456063828acc21e1cf29482944db66cc566037cb3911d4d48.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/9bd729458036c95456063828acc21e1cf29482944db66cc566037cb3911d4d48.jpg)

![fb5a04443ff3fa1c3aee7a560979273065f572d22ca036cdece90e329995b686.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/images/fb5a04443ff3fa1c3aee7a560979273065f572d22ca036cdece90e329995b686.jpg)

### Tables

![02ce623c2236d06639a09411326249a89a11dd5675dd3a0b0c72415aa7f5dcab.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/02ce623c2236d06639a09411326249a89a11dd5675dd3a0b0c72415aa7f5dcab.jpg)

![051348453e2de130cfaf33e2d79d76c646838c36276820135338dd674c05c283.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/051348453e2de130cfaf33e2d79d76c646838c36276820135338dd674c05c283.jpg)

![168f7c9e0ad67b2844b8861e67bd450265a934ee077537b989c51a39ffdf983e.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/168f7c9e0ad67b2844b8861e67bd450265a934ee077537b989c51a39ffdf983e.jpg)

![621d09113d60a56a782d15548dd430a572067d70591b9da6f28f293e285a8087.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/621d09113d60a56a782d15548dd430a572067d70591b9da6f28f293e285a8087.jpg)

![7a592e164039b3ca73294e3f47b1dcad65d2dc271c34f6ec480c82f3d9df35c4.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/7a592e164039b3ca73294e3f47b1dcad65d2dc271c34f6ec480c82f3d9df35c4.jpg)

![8e4f10b0de9fd191759f39c53f0857311d678bc6e2cbb94bc633b1279451289f.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/8e4f10b0de9fd191759f39c53f0857311d678bc6e2cbb94bc633b1279451289f.jpg)

![b1f0bbb8ada2bbd7c0a0ef248f2d3057e121dffba6d2176b8e2d1115f034ec86.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/b1f0bbb8ada2bbd7c0a0ef248f2d3057e121dffba6d2176b8e2d1115f034ec86.jpg)

![cb7bb1cb44e2b4b1c234361912806cce76aea7176c326bc0cc261eb3aac60b49.jpg](../icml_results/4514_Towards%20Efficient%20Large%20Language%20Reasoning%20Models%20via%20Extreme-Ratio%20Chain-of-Thought%20Compression/tables/cb7bb1cb44e2b4b1c234361912806cce76aea7176c326bc0cc261eb3aac60b49.jpg)

<a id="paper-4468"></a>

## Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction


### Images

![1eca8700d107a0ef214f28bf335a3b64d9c72850ee0e314f60c36ae2cd2f5f8f.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/1eca8700d107a0ef214f28bf335a3b64d9c72850ee0e314f60c36ae2cd2f5f8f.jpg)

![369c5e9fc7704683d8dbca0d97e1e2875db096b21e4b13e134bbef4f472e87b5.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/369c5e9fc7704683d8dbca0d97e1e2875db096b21e4b13e134bbef4f472e87b5.jpg)

![404d725229f39f6bb384cd8f257fabca37cacbd33e80dd0f8651ef0a9e8395be.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/404d725229f39f6bb384cd8f257fabca37cacbd33e80dd0f8651ef0a9e8395be.jpg)

![60148845a628c0e6db1568fb83a35bac91a23e75a5c61dbcb0e1ba2c0bfcb92e.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/60148845a628c0e6db1568fb83a35bac91a23e75a5c61dbcb0e1ba2c0bfcb92e.jpg)

![7e17011d3ab173f847827be5e363c60ceac300b3095b7f86529fc5d6711e07f0.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/7e17011d3ab173f847827be5e363c60ceac300b3095b7f86529fc5d6711e07f0.jpg)

![835ef59c4c60ac4348a4f66ba430207a6e75046421ec58bc1b44145e2092c76a.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/835ef59c4c60ac4348a4f66ba430207a6e75046421ec58bc1b44145e2092c76a.jpg)

![89d737fa3eb363070464120fc3c4b6c310f0f778bb9ecc06a5e196c0b8610dc5.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/89d737fa3eb363070464120fc3c4b6c310f0f778bb9ecc06a5e196c0b8610dc5.jpg)

![8ad4cdfba58b58815c9b52a4b534809036a46d05dd9a2c3307e042cb12c59134.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/8ad4cdfba58b58815c9b52a4b534809036a46d05dd9a2c3307e042cb12c59134.jpg)

![8b4add2f41702822c91b99042dffbaf29b6ff63077996da678c2396717d7b20d.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/images/8b4add2f41702822c91b99042dffbaf29b6ff63077996da678c2396717d7b20d.jpg)

### Tables

![0257e5a9a16ad48c46eb16be8ab0a603f522352c5ca428c9d56754213b400c04.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/0257e5a9a16ad48c46eb16be8ab0a603f522352c5ca428c9d56754213b400c04.jpg)

![0e4792d66f434dd22be5b7169f2ad8995e693196a0a8b61bd30758015331ba29.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/0e4792d66f434dd22be5b7169f2ad8995e693196a0a8b61bd30758015331ba29.jpg)

![0f8fc58db0b98d77a77382eb2af474931f8bd85e156c0eb9029efc22b9b2c1e3.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/0f8fc58db0b98d77a77382eb2af474931f8bd85e156c0eb9029efc22b9b2c1e3.jpg)

![17e92ab4d1a7cac75113ea116f270a9a2482b409cdce135b511f4dcd5155df28.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/17e92ab4d1a7cac75113ea116f270a9a2482b409cdce135b511f4dcd5155df28.jpg)

![2fd3e8ae2a1c2b1b20b1ea9833e17efb416bcf49bb688695dc4f28314a47f85b.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/2fd3e8ae2a1c2b1b20b1ea9833e17efb416bcf49bb688695dc4f28314a47f85b.jpg)

![621333563f9e49db9e94ad432d5ed10acfcaebc446a4cad432fd59dc0980fb55.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/621333563f9e49db9e94ad432d5ed10acfcaebc446a4cad432fd59dc0980fb55.jpg)

![74a9b37c77442bb749987a1f1fec9613f23a211af0e44e5a75957ea8990cb1d2.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/74a9b37c77442bb749987a1f1fec9613f23a211af0e44e5a75957ea8990cb1d2.jpg)

![7a7b3872dbc1ceea37c3170293312f32ec3bbab7844e74e4133468db7bdea1e5.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/7a7b3872dbc1ceea37c3170293312f32ec3bbab7844e74e4133468db7bdea1e5.jpg)

![820b06df97dccb71e9cae662aa2feff5f04fd8ca6ae7cb78ab8a72c804780c75.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/820b06df97dccb71e9cae662aa2feff5f04fd8ca6ae7cb78ab8a72c804780c75.jpg)

![a317e1647805aef384c31a4dce6fb192f7274111c06bb9d5a3463a77cc063502.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/a317e1647805aef384c31a4dce6fb192f7274111c06bb9d5a3463a77cc063502.jpg)

![a700a8d82e375567e4965b0130e5fbd357b19bb2b9ffa6e2d45f47691ff6eb88.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/a700a8d82e375567e4965b0130e5fbd357b19bb2b9ffa6e2d45f47691ff6eb88.jpg)

![aed0d11fc163a67c39495a3ae00e7c175e9d29c59a1235567980e0e685f4d751.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/aed0d11fc163a67c39495a3ae00e7c175e9d29c59a1235567980e0e685f4d751.jpg)

![b67e586658281c14f4589ea06a9d07faf83a47b13923d7fe87da708159a07183.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/b67e586658281c14f4589ea06a9d07faf83a47b13923d7fe87da708159a07183.jpg)

![dbc4f3e6d53671df8ab04ada271b08a4ba16fe134fc7fef8694a4b28b75dadb1.jpg](../icml_results/4515_Predicting%20Future%20Utility_%20Global%20Combinatorial%20Optimization%20for%20Task-Agnostic%20KV%20Cache%20Eviction/tables/dbc4f3e6d53671df8ab04ada271b08a4ba16fe134fc7fef8694a4b28b75dadb1.jpg)

<a id="paper-4469"></a>

## PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization


### Images

![05fd4533b5ad4f0d9d050aa253bbdaddd482fa5da2c3c306bfd58f7acad45f82.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/05fd4533b5ad4f0d9d050aa253bbdaddd482fa5da2c3c306bfd58f7acad45f82.jpg)

![0b41475995462eea06d5f7a747d8a1969e625607198f5de81757872a83a72372.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/0b41475995462eea06d5f7a747d8a1969e625607198f5de81757872a83a72372.jpg)

![0bf02ce8943cce90bc43b493e749595b6dbf2669b3e56e4c16593ee54bca4499.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/0bf02ce8943cce90bc43b493e749595b6dbf2669b3e56e4c16593ee54bca4499.jpg)

![154428e8299e12aa95115407b33dd28bcd0b394bc7ba4971409af64a7ff66abd.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/154428e8299e12aa95115407b33dd28bcd0b394bc7ba4971409af64a7ff66abd.jpg)

![1c3cbde7fbd55af203e7f70b3c78cf7e0d1c4d28b853de3f62e09196b980a337.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/1c3cbde7fbd55af203e7f70b3c78cf7e0d1c4d28b853de3f62e09196b980a337.jpg)

![6d3d36b238179aa48a73763bee00d04f6bfca3b83dab816f4f0c6c4621630ccb.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/6d3d36b238179aa48a73763bee00d04f6bfca3b83dab816f4f0c6c4621630ccb.jpg)

![79d9f611af70f5fd4b41129bd9fbdd04c6c59424fb6e1cff02313f958db465bb.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/79d9f611af70f5fd4b41129bd9fbdd04c6c59424fb6e1cff02313f958db465bb.jpg)

![7ccb6676a3624dede47db17338d95fef02a76782255393692c10c998c3c6e844.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/7ccb6676a3624dede47db17338d95fef02a76782255393692c10c998c3c6e844.jpg)

![a45a2d0a301b3e1802b72a1bd66771d7dd694f7c510d482574fc4d4ccdfe978d.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/a45a2d0a301b3e1802b72a1bd66771d7dd694f7c510d482574fc4d4ccdfe978d.jpg)

![b5f3c83488efd14314cfb0470c9e156d3af78b475fc901b5ec5bebb811d5ff52.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/b5f3c83488efd14314cfb0470c9e156d3af78b475fc901b5ec5bebb811d5ff52.jpg)

![c8619e47ca5af40d7f757c3792e47497e0a269276c9a095470bd98fd37d8e34c.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/c8619e47ca5af40d7f757c3792e47497e0a269276c9a095470bd98fd37d8e34c.jpg)

![e084f3e40cc1176e6db3a829460e539b3d2d89e70f55cbcdfb258eae4dfcdc79.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/e084f3e40cc1176e6db3a829460e539b3d2d89e70f55cbcdfb258eae4dfcdc79.jpg)

![e5c3a03539dcb5a971cd8249d0b00b22b09b8384972b73e78332d85ec7dab59d.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/e5c3a03539dcb5a971cd8249d0b00b22b09b8384972b73e78332d85ec7dab59d.jpg)

![e7e5ebbdc4ba07ffd6d723eed9b8c76c059762af60b542132bd5cbd3feec86e9.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/e7e5ebbdc4ba07ffd6d723eed9b8c76c059762af60b542132bd5cbd3feec86e9.jpg)

![f3f45e3d0e5043d77d229ff53bee5a683b9f0d3520e91094d4112a919f4e5e0c.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/f3f45e3d0e5043d77d229ff53bee5a683b9f0d3520e91094d4112a919f4e5e0c.jpg)

![f6dccda11ebfae58340b7d21824c3b62104c4e955463d8371c6027837122af78.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/f6dccda11ebfae58340b7d21824c3b62104c4e955463d8371c6027837122af78.jpg)

![f85ace332862947660e311e8dcf2eb0721dd443e617ad4c5aa03c14be407b200.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/f85ace332862947660e311e8dcf2eb0721dd443e617ad4c5aa03c14be407b200.jpg)

![fddf6156c0a5914a4776ac9a9f80e9cc46289a3e516de80323bfe6b3d715509b.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/images/fddf6156c0a5914a4776ac9a9f80e9cc46289a3e516de80323bfe6b3d715509b.jpg)

### Tables

![0c3ef0d23cb7205c8197d3e407ae7dc8d5898f664de18b59ec939cf86d809d0a.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/0c3ef0d23cb7205c8197d3e407ae7dc8d5898f664de18b59ec939cf86d809d0a.jpg)

![1fc0110bfd6ec6ddca778511b80a09040014eaf88fbdff88719f9993599c6567.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/1fc0110bfd6ec6ddca778511b80a09040014eaf88fbdff88719f9993599c6567.jpg)

![4814812f6df130e8b75d99298914b24b0e41bbfb974f2b42d711118e1c27ef97.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/4814812f6df130e8b75d99298914b24b0e41bbfb974f2b42d711118e1c27ef97.jpg)

![55ca873110f081496773c9cfbd25b74406d963ce1f9f440e9e1c49ec6bb6a6d7.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/55ca873110f081496773c9cfbd25b74406d963ce1f9f440e9e1c49ec6bb6a6d7.jpg)

![6a7d3d289240b9d9bffb8c4642a862f27e79201762698fd8a0eded9f46da0cef.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/6a7d3d289240b9d9bffb8c4642a862f27e79201762698fd8a0eded9f46da0cef.jpg)

![9321bb1f47346e1ac0de5bccb0ee63877cf22f32d2262bb9df1543be2d846ece.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/9321bb1f47346e1ac0de5bccb0ee63877cf22f32d2262bb9df1543be2d846ece.jpg)

![9a1c744c311c407ead8978da7a0cd2144a25b3f1a69cf17530f60f85b895c9be.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/9a1c744c311c407ead8978da7a0cd2144a25b3f1a69cf17530f60f85b895c9be.jpg)

![c0a5756ed9e4f4cad65ce0ded728dfd754e413bee3d28013ba9cea6e323057e4.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/c0a5756ed9e4f4cad65ce0ded728dfd754e413bee3d28013ba9cea6e323057e4.jpg)

![cbc6054f1eac49b926042605b1b5aec5df55d9c3aa8233177c7ffe55259cf503.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/cbc6054f1eac49b926042605b1b5aec5df55d9c3aa8233177c7ffe55259cf503.jpg)

![d1d4721e2a35e8ecc0ae6c22336a3d42da1ad7af7de6e324f1fa616509d0f1d8.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/d1d4721e2a35e8ecc0ae6c22336a3d42da1ad7af7de6e324f1fa616509d0f1d8.jpg)

![d4fe507285381848d8dc3728577316b8de31e8ff49b42d79e3cf646a93d84101.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/d4fe507285381848d8dc3728577316b8de31e8ff49b42d79e3cf646a93d84101.jpg)

![f26ee5063b4feb61149c32744cbf7194f58ac2fdb5311633a92e96421c11ace8.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/f26ee5063b4feb61149c32744cbf7194f58ac2fdb5311633a92e96421c11ace8.jpg)

![fa0a5a481bac83442b07e709475d35f97caae4ffd6c7f29be982d168495faf6d.jpg](../icml_results/4516_PRISM_%20Synergizing%20Vision%20Foundation%20Models%20via%20Self-organized%20Expert%20Specialization/tables/fa0a5a481bac83442b07e709475d35f97caae4ffd6c7f29be982d168495faf6d.jpg)

<a id="paper-4470"></a>

## Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale


### Images

![322c44f6365c66320e54a1a3b5a52c74df2d67ee088126524234215a7d958282.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/322c44f6365c66320e54a1a3b5a52c74df2d67ee088126524234215a7d958282.jpg)

![51d07d78a090c264c5df33b6a18ecd6f68a6aa57788eb0ed6776698ee6892b66.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/51d07d78a090c264c5df33b6a18ecd6f68a6aa57788eb0ed6776698ee6892b66.jpg)

![85b0ce0f21f106695e984bb97df53ef146f4f9c906c6456ad7abe76cb4b03122.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/85b0ce0f21f106695e984bb97df53ef146f4f9c906c6456ad7abe76cb4b03122.jpg)

![9297287513c45f5639aab3c430fd968b9a91c1bbe59db925c11523d9d54e532b.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/9297287513c45f5639aab3c430fd968b9a91c1bbe59db925c11523d9d54e532b.jpg)

![bdad9c51a6561b814d8fa854f2a25ca09ae5e56e42d22b7aa92b9531b2281e29.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/bdad9c51a6561b814d8fa854f2a25ca09ae5e56e42d22b7aa92b9531b2281e29.jpg)

![c264c3b02ab1f1bc6015f3971f5f4f20e85f3612b540ef7a8361c3e1ac1783d8.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/c264c3b02ab1f1bc6015f3971f5f4f20e85f3612b540ef7a8361c3e1ac1783d8.jpg)

![efbb9762a4e619925677c0093da4bbc28721d8ea8729b562cd4d5ecf9c2f0a4c.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/images/efbb9762a4e619925677c0093da4bbc28721d8ea8729b562cd4d5ecf9c2f0a4c.jpg)

### Tables

![3912f82a37a75fb743e3c1d6880f8e4118e6ca00078e511e0962d1375e919ab8.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/3912f82a37a75fb743e3c1d6880f8e4118e6ca00078e511e0962d1375e919ab8.jpg)

![745cff347843c122cf439c402cc5daa9772f7454f4addfe90d62c96027e75819.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/745cff347843c122cf439c402cc5daa9772f7454f4addfe90d62c96027e75819.jpg)

![973a03e96e730099c9c126c523faa0d183e519ee57e402366e162d2618f03387.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/973a03e96e730099c9c126c523faa0d183e519ee57e402366e162d2618f03387.jpg)

![afe2bb607ef0b0827abdb629367c2eed281b9dc07f01291af839f8b7131e5352.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/afe2bb607ef0b0827abdb629367c2eed281b9dc07f01291af839f8b7131e5352.jpg)

![bfe962442ee746663a751927896dfb4e4943709c14b6098a6f4210e8db31967a.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/bfe962442ee746663a751927896dfb4e4943709c14b6098a6f4210e8db31967a.jpg)

![d9c7722861b96c49a3f8bf9db056c562ac1dc596cd5be82454672971bd708d02.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/d9c7722861b96c49a3f8bf9db056c562ac1dc596cd5be82454672971bd708d02.jpg)

![f04703fb07d78e8ac597b87abb8e310f4c51e5916a0bbd40fed3bbadae5ec9cd.jpg](../icml_results/4517_Beyond%20Gemini-3-Pro_%20Revisiting%20LLM%20Routing%20and%20Aggregation%20at%20Scale/tables/f04703fb07d78e8ac597b87abb8e310f4c51e5916a0bbd40fed3bbadae5ec9cd.jpg)

<a id="paper-4471"></a>

## Reconstruction Outcomes Look Similar but Processes Differ: Improving Context Consistency and Coverage in Graph Masked Auto-Encoder


### Images

![186e05fceaa79c3617b8919ecb6711897181f9151434426ccc390a959a986515.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/186e05fceaa79c3617b8919ecb6711897181f9151434426ccc390a959a986515.jpg)

![22cc695101513ac7f358f9c73027f9b56c11850db8f4f99755b4a7316fbce82c.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/22cc695101513ac7f358f9c73027f9b56c11850db8f4f99755b4a7316fbce82c.jpg)

![290082cd3f6d8908ca9ef42af4b21f9c91359e0e629e021735b65bba08fa65ec.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/290082cd3f6d8908ca9ef42af4b21f9c91359e0e629e021735b65bba08fa65ec.jpg)

![429731735552d2309b035fddcc8d78294e777030ff7dd06112937036e0455f0e.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/429731735552d2309b035fddcc8d78294e777030ff7dd06112937036e0455f0e.jpg)

![c6787897fb1c5542137a16a2548b94d5b0adfc454cc002aedfb13a76a3f1b114.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/c6787897fb1c5542137a16a2548b94d5b0adfc454cc002aedfb13a76a3f1b114.jpg)

![d77cf82ac3ea827a16454cc28f1d6ec968a8ba5de9e135deda1218f2b5aa9875.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/images/d77cf82ac3ea827a16454cc28f1d6ec968a8ba5de9e135deda1218f2b5aa9875.jpg)

### Tables

![099abf5815dcc70367c08b02aca1144e4773b20f4106c13bac6b862db87e3e49.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/099abf5815dcc70367c08b02aca1144e4773b20f4106c13bac6b862db87e3e49.jpg)

![26f545134d55896030450fd86c39b4b981d5ee0c02effa04d5434ff0b88e473f.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/26f545134d55896030450fd86c39b4b981d5ee0c02effa04d5434ff0b88e473f.jpg)

![27b466517cae82ed21483ad6b5b0b4e00a0fb23da75b90ca0ae7aad7ede06e0d.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/27b466517cae82ed21483ad6b5b0b4e00a0fb23da75b90ca0ae7aad7ede06e0d.jpg)

![447a17ca445f0c55a10aceb69fa2ded3818b64f946b6a62e8b6157f0a16e650e.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/447a17ca445f0c55a10aceb69fa2ded3818b64f946b6a62e8b6157f0a16e650e.jpg)

![4ac736ba561d1d7de94a22450219a93da7ae2e58609777cb34c05fdc02f778a0.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/4ac736ba561d1d7de94a22450219a93da7ae2e58609777cb34c05fdc02f778a0.jpg)

![54d85c57ce20968bb18aba76a5f05c6836007d3f2196d18de5ab991b7109ad0f.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/54d85c57ce20968bb18aba76a5f05c6836007d3f2196d18de5ab991b7109ad0f.jpg)

![60a715f614484966591b9ad04c3cbf01140d9a223dd657c48b20d294ba681e0c.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/60a715f614484966591b9ad04c3cbf01140d9a223dd657c48b20d294ba681e0c.jpg)

![813821670377e7882f7daeb948673e51fee1ccb35b2e1f8894b47971280f08f1.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/813821670377e7882f7daeb948673e51fee1ccb35b2e1f8894b47971280f08f1.jpg)

![89224a4e042f7d69e34eccb7936da57845abe3c83449788ac812af1046452ec7.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/89224a4e042f7d69e34eccb7936da57845abe3c83449788ac812af1046452ec7.jpg)

![9523df4bba4d0f304951d2896d422341d4337f37379d864911cc2c5cb5f3d320.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/9523df4bba4d0f304951d2896d422341d4337f37379d864911cc2c5cb5f3d320.jpg)

![9b85a42047451dd38b0103ec11e8cf15e8c7e0b144fc4a5b138b475faec822dc.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/9b85a42047451dd38b0103ec11e8cf15e8c7e0b144fc4a5b138b475faec822dc.jpg)

![ae991f7c36a131df207605d4c204030567c55b4534a51802be67aaaa418a7a53.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/ae991f7c36a131df207605d4c204030567c55b4534a51802be67aaaa418a7a53.jpg)

![b0be08a34fe26360d1d548fb671c5a612c7134f4e5a5630a38f529b5fce98fbf.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/b0be08a34fe26360d1d548fb671c5a612c7134f4e5a5630a38f529b5fce98fbf.jpg)

![c6fa0f3d52c0a34ae3d1a42dc7873e3605713c470881bfc7fe57fabd78d4fcd7.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/c6fa0f3d52c0a34ae3d1a42dc7873e3605713c470881bfc7fe57fabd78d4fcd7.jpg)

![d7ebba1b782bd975d3d6d975673ccfea9f74ab983a7da96fb7064e02f15486f3.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/d7ebba1b782bd975d3d6d975673ccfea9f74ab983a7da96fb7064e02f15486f3.jpg)

![ebf4edf1016a791bfabfe9c1c42f0eef7ec25e15e7d676d5f79eeaecec446625.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/ebf4edf1016a791bfabfe9c1c42f0eef7ec25e15e7d676d5f79eeaecec446625.jpg)

![f6c848948f667b274055e2eaaed21cf83cb79f050e8814d86aaa46ac1176e7ca.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/f6c848948f667b274055e2eaaed21cf83cb79f050e8814d86aaa46ac1176e7ca.jpg)

![fcb91e0a5bb9fe913b2c5e6b935f9f1a74dbde68566683f9c85182c13a075753.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/fcb91e0a5bb9fe913b2c5e6b935f9f1a74dbde68566683f9c85182c13a075753.jpg)

![fcd1ec15ea258b4de63ac1ed8465a3d52ef449d7f4f1df1dd63879373a2d31fb.jpg](../icml_results/4518_Reconstruction%20Outcomes%20Look%20Similar%20but%20Processes%20Differ_%20Improving%20Context%20Consistency%20and%20Coverag/tables/fcd1ec15ea258b4de63ac1ed8465a3d52ef449d7f4f1df1dd63879373a2d31fb.jpg)

<a id="paper-4472"></a>

## Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering


### Images

![1780dccbee0ad3f07ddb267c4b4a0af1da35c59f36f5d264f58035398bfc7ffc.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/1780dccbee0ad3f07ddb267c4b4a0af1da35c59f36f5d264f58035398bfc7ffc.jpg)

![500c458ac35ca51b2cd536cfae4fda1d772cdbc9ebdd919e706b9e722f54ccf9.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/500c458ac35ca51b2cd536cfae4fda1d772cdbc9ebdd919e706b9e722f54ccf9.jpg)

![6dcb41bf64ac9ad81f61b70d0f31a20d96b3aaaee8ec2457c3c624727f758428.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/6dcb41bf64ac9ad81f61b70d0f31a20d96b3aaaee8ec2457c3c624727f758428.jpg)

![9474ce867dc40fb6b3ada5a8bc1517c6e368c51d64661f420501aaa69226dc76.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/9474ce867dc40fb6b3ada5a8bc1517c6e368c51d64661f420501aaa69226dc76.jpg)

![959214e7eecd6bb847b561675133c96ed973b99979fe5a4e6d6e3b263dd38b06.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/959214e7eecd6bb847b561675133c96ed973b99979fe5a4e6d6e3b263dd38b06.jpg)

![e5eec848db9852a1e8f3416b9b9705e5b358d5c9f2b82d689d09f904fdffa14f.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/e5eec848db9852a1e8f3416b9b9705e5b358d5c9f2b82d689d09f904fdffa14f.jpg)

![fd3b1f397b0bf65517b0b653b3e5d3f2ba13acbee94b607dbf5e7bd39f0a093d.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/images/fd3b1f397b0bf65517b0b653b3e5d3f2ba13acbee94b607dbf5e7bd39f0a093d.jpg)

### Tables

![165c86233054b5b7ad67f2856f4f0644c68dfba15f09e7b58e5cb7c6d3801fbf.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/165c86233054b5b7ad67f2856f4f0644c68dfba15f09e7b58e5cb7c6d3801fbf.jpg)

![426d5c5eaa24fef04f8e0d38ec77b7971884faf9df68d9ab38eb0776878260f3.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/426d5c5eaa24fef04f8e0d38ec77b7971884faf9df68d9ab38eb0776878260f3.jpg)

![62297b98ee017003e6f72c14e686826460007f6beaefa9cac4ed72498dcc483c.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/62297b98ee017003e6f72c14e686826460007f6beaefa9cac4ed72498dcc483c.jpg)

![7086d92719c8fbbba0d5384dbf39a6c88f4c5f652f82e4aa854bb8038103b179.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/7086d92719c8fbbba0d5384dbf39a6c88f4c5f652f82e4aa854bb8038103b179.jpg)

![739b4a94f6d1720fdd6514d9761964abb3c027c39c8323c2d14a24e6b76cf8e1.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/739b4a94f6d1720fdd6514d9761964abb3c027c39c8323c2d14a24e6b76cf8e1.jpg)

![9fc0bbeaeec97a7865819512f97222990de42dc3a5e25bfcfc8e695659d4069f.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/9fc0bbeaeec97a7865819512f97222990de42dc3a5e25bfcfc8e695659d4069f.jpg)

![ac58050c39fd02d0f4eadf6bce1f4d83374ba2c239ed058ccba33b94bad840c0.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/ac58050c39fd02d0f4eadf6bce1f4d83374ba2c239ed058ccba33b94bad840c0.jpg)

![b45cec27f302c1042b28df6674811386b15a797dbea422cea566087a560a3117.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/b45cec27f302c1042b28df6674811386b15a797dbea422cea566087a560a3117.jpg)

![f01ed379fbb5ec5892118209c5f9f82524c4d12688b3cd2a7613fd1a74aa3333.jpg](../icml_results/4519_Mitigating%20Error%20Accumulation%20in%20Continuous%20Navigation%20via%20Memory-Augmented%20Kalman%20Filtering/tables/f01ed379fbb5ec5892118209c5f9f82524c4d12688b3cd2a7613fd1a74aa3333.jpg)

<a id="paper-4473"></a>

## ReLAM: Learning Anticipation Model for Rewarding Visual Robotic Manipulation


### Images

![097007a48e9909368f01fda59fa85c406ebcd23dd745f5f211592869491de12b.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/097007a48e9909368f01fda59fa85c406ebcd23dd745f5f211592869491de12b.jpg)

![14df0546a3a275e0dd74473a40bd5c10a8024d34c0fd1fa601268053c5bc8f89.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/14df0546a3a275e0dd74473a40bd5c10a8024d34c0fd1fa601268053c5bc8f89.jpg)

![30a1058672a37b7bfa072e4a4c8b98994bdb470fe31dd303035305d1646369b2.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/30a1058672a37b7bfa072e4a4c8b98994bdb470fe31dd303035305d1646369b2.jpg)

![368b8f7e5937d881d640ab233ccec52186c5884d545ff1d5fc70902519c44dde.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/368b8f7e5937d881d640ab233ccec52186c5884d545ff1d5fc70902519c44dde.jpg)

![65887975c204cc4bff5baedf3013797b658d477009646ac7e234cba6fcb91551.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/65887975c204cc4bff5baedf3013797b658d477009646ac7e234cba6fcb91551.jpg)

![6a2f2767c57638ecb2abbfcc529dbd1537451c40239fafdde628d5c58ead4303.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/6a2f2767c57638ecb2abbfcc529dbd1537451c40239fafdde628d5c58ead4303.jpg)

![74dcafbfab693e45ff5e206dc4c569f40743198ab22420f09951ff626f32ec3e.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/74dcafbfab693e45ff5e206dc4c569f40743198ab22420f09951ff626f32ec3e.jpg)

![8bd8f4fdf8817f1a730bb58a3dfa516437b6e546470260079cc5e7524fddb247.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/8bd8f4fdf8817f1a730bb58a3dfa516437b6e546470260079cc5e7524fddb247.jpg)

![a137f4b0a50052cac71d494971f5bc772b3bc98bc94cd17f0128ea083031b808.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/a137f4b0a50052cac71d494971f5bc772b3bc98bc94cd17f0128ea083031b808.jpg)

![af3d46eb5f9a10b89fc3119cb873bc83a477462c6160dc50f8a70022ec80cdb6.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/af3d46eb5f9a10b89fc3119cb873bc83a477462c6160dc50f8a70022ec80cdb6.jpg)

![c704d500f6a1b20ef633f98cb21ac5feba5aeadd4a78c6c0a65c6fea39c0d8ba.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/c704d500f6a1b20ef633f98cb21ac5feba5aeadd4a78c6c0a65c6fea39c0d8ba.jpg)

![f60f736a0907895cd7acd85796e6f39a36c4205720519fdaa2361bad5b114b30.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/f60f736a0907895cd7acd85796e6f39a36c4205720519fdaa2361bad5b114b30.jpg)

![fcbbb716a9d869dbe0fc3bdc43b0bda1a7c747be3b8f4b62a94dc1111e2cd017.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/images/fcbbb716a9d869dbe0fc3bdc43b0bda1a7c747be3b8f4b62a94dc1111e2cd017.jpg)

### Tables

![1bd32dc178b1af63836538bd07409c827d3881801f811720a2866e492fe6cc9a.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/1bd32dc178b1af63836538bd07409c827d3881801f811720a2866e492fe6cc9a.jpg)

![2091b7630b776864c2dde5d8ec5d3f587254213b2027716ea538be1a6ccc28d3.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/2091b7630b776864c2dde5d8ec5d3f587254213b2027716ea538be1a6ccc28d3.jpg)

![6ae94e25f524867787e39d2e6db1241d2b630574d53b55c5f2a19a1e4e622217.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/6ae94e25f524867787e39d2e6db1241d2b630574d53b55c5f2a19a1e4e622217.jpg)

![bd59690ca629d9ee06a1b29f6b26470d434f57832c994a8a91373560b6404d5f.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/bd59690ca629d9ee06a1b29f6b26470d434f57832c994a8a91373560b6404d5f.jpg)

![c044c570f5e0f28c071f30b5e1a008515623137967972185d8762c692b253794.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/c044c570f5e0f28c071f30b5e1a008515623137967972185d8762c692b253794.jpg)

![c5214ce1f5fdff06baf8377ced04657701bb36acfb38a16d4f6a774cdfc46104.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/c5214ce1f5fdff06baf8377ced04657701bb36acfb38a16d4f6a774cdfc46104.jpg)

![c9ba2ec3385d01044f7e7ab016e378304c7721a278ddbed78b44461edd066caf.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/c9ba2ec3385d01044f7e7ab016e378304c7721a278ddbed78b44461edd066caf.jpg)

![f20b144e0ebb442692c7c2b2c90afa4c84dc6924193985e4eec268a4838e161f.jpg](../icml_results/4520_ReLAM_%20Learning%20Anticipation%20Model%20for%20Rewarding%20Visual%20Robotic%20Manipulation/tables/f20b144e0ebb442692c7c2b2c90afa4c84dc6924193985e4eec268a4838e161f.jpg)

<a id="paper-4474"></a>

## Glimpse: Geometry Learning of Multi-scale Structural Priors for 3D Pose Estimation


### Images

![030f24a730c6b31559def788669a059bb70c0d25034d2317b8c80ccc753b563f.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/030f24a730c6b31559def788669a059bb70c0d25034d2317b8c80ccc753b563f.jpg)

![0e5dafe0ae09328b745ff4904105ca85adf6a44d63e73e9bf2b21879170f27bf.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/0e5dafe0ae09328b745ff4904105ca85adf6a44d63e73e9bf2b21879170f27bf.jpg)

![25bdceeb3ffb495c8e9d6930105c52e7e95c8174c738f79f9793407d2ccf13d0.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/25bdceeb3ffb495c8e9d6930105c52e7e95c8174c738f79f9793407d2ccf13d0.jpg)

![7f51db289440575afc54034a06bd3669e30ed5637cf90f1375c384e0f7bf187f.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/7f51db289440575afc54034a06bd3669e30ed5637cf90f1375c384e0f7bf187f.jpg)

![aca9ff40bbad9aa1f716bfebbac77b6d229a481d2e45e56b7396808190c10847.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/aca9ff40bbad9aa1f716bfebbac77b6d229a481d2e45e56b7396808190c10847.jpg)

![d18bf6cf54ad04d4636eb34c2b66810a7207d48d0988fac6cfd601f55e10d60c.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/d18bf6cf54ad04d4636eb34c2b66810a7207d48d0988fac6cfd601f55e10d60c.jpg)

![dc2ea86879bbaf15cafabf88e66cb796fd6c971363cc2d42375958b5c1447bc5.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/dc2ea86879bbaf15cafabf88e66cb796fd6c971363cc2d42375958b5c1447bc5.jpg)

![dce92a67b48d5915e17e804e795abc7088182b2f559f357549b8d09519d18b6f.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/dce92a67b48d5915e17e804e795abc7088182b2f559f357549b8d09519d18b6f.jpg)

![e651dbb9b6b274f68fcc1562afd1ea613c57c253332f9c4aad68dbab42d954d2.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/e651dbb9b6b274f68fcc1562afd1ea613c57c253332f9c4aad68dbab42d954d2.jpg)

![e842febc63d50e40aefece9571d9743f3627d5a59d57412f7f26708be7838825.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/e842febc63d50e40aefece9571d9743f3627d5a59d57412f7f26708be7838825.jpg)

![e9b79ee14699794b2411723dcfde64758d3d2372dde99831f929b74aacaf391b.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/e9b79ee14699794b2411723dcfde64758d3d2372dde99831f929b74aacaf391b.jpg)

![fee8d985bd41f6fa5c9479f95a1afe4eac7b5793982338b59e94aa93c5945530.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/images/fee8d985bd41f6fa5c9479f95a1afe4eac7b5793982338b59e94aa93c5945530.jpg)

### Tables

![8cad63495ef0cd5d008193de69eb6890f5652bb5eed72a11be849dcc22201cb6.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/tables/8cad63495ef0cd5d008193de69eb6890f5652bb5eed72a11be849dcc22201cb6.jpg)

![aec8413a283296ffb2cec61596d2ff96ad2505f63696cfcd2311ab243ab7410e.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/tables/aec8413a283296ffb2cec61596d2ff96ad2505f63696cfcd2311ab243ab7410e.jpg)

![af605614723c0edc70913d85d8e4aa6d4b74ff189e7872faad48660315dbbd1e.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/tables/af605614723c0edc70913d85d8e4aa6d4b74ff189e7872faad48660315dbbd1e.jpg)

![d2e824296b580b41b948b96bb3cad8563aca8da58a0bddfceb5b1ecd7b91f72a.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/tables/d2e824296b580b41b948b96bb3cad8563aca8da58a0bddfceb5b1ecd7b91f72a.jpg)

![ed46f7b3d55a988f0633c162a2a9b8716a18bb0c040123b08e785335fa4bac67.jpg](../icml_results/4521_Glimpse_%20Geometry%20Learning%20of%20Multi-scale%20Structural%20Priors%20for%203D%20Pose%20Estimation/tables/ed46f7b3d55a988f0633c162a2a9b8716a18bb0c040123b08e785335fa4bac67.jpg)

<a id="paper-4475"></a>

## When Generalized Zero-Shot Learning Meets PU Learning: A Plug-and-Play Framework for Seen-Class Bias Mitigation


### Images

![0657798001dcb287ae62bebe0f19bd7540ce63e68652902186e95fad24da344e.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/0657798001dcb287ae62bebe0f19bd7540ce63e68652902186e95fad24da344e.jpg)

![137c6aa8739c3af9c3bdf2d20cd3d363ab331ec1f82584d92a59fdc8f2c0f976.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/137c6aa8739c3af9c3bdf2d20cd3d363ab331ec1f82584d92a59fdc8f2c0f976.jpg)

![18588f5a5da7059cc2dde38c518bfb5722956c902e2d59623ed97e4fae7520f3.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/18588f5a5da7059cc2dde38c518bfb5722956c902e2d59623ed97e4fae7520f3.jpg)

![19dcce49117f73de1a2ab296b30dc9c28d7a82ef600cc1ab483421ee4253dc88.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/19dcce49117f73de1a2ab296b30dc9c28d7a82ef600cc1ab483421ee4253dc88.jpg)

![24d5bfc753dd44efc622ef18b6d5f6fa36905cddb19562ef92b9e798806b7212.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/24d5bfc753dd44efc622ef18b6d5f6fa36905cddb19562ef92b9e798806b7212.jpg)

![5aa6c4d7f1cb6a0cc449c4e9b1d93506ec0e1bde75c88502978c432dba18c0f5.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/5aa6c4d7f1cb6a0cc449c4e9b1d93506ec0e1bde75c88502978c432dba18c0f5.jpg)

![72afed6f38d1052b870f701ccf577093d387207744a1d41357707f3f70f9ee7c.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/72afed6f38d1052b870f701ccf577093d387207744a1d41357707f3f70f9ee7c.jpg)

![ff928724e8b33c1741d07fe58c8177c3f8e7c9e8e8d165c7e580c44b8266254b.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/images/ff928724e8b33c1741d07fe58c8177c3f8e7c9e8e8d165c7e580c44b8266254b.jpg)

### Tables

![1cd84ca1e76fe3e1c075d6fe7d2ebb16b80236e8091b3dc3341da313f4f4e0cc.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/1cd84ca1e76fe3e1c075d6fe7d2ebb16b80236e8091b3dc3341da313f4f4e0cc.jpg)

![293260d2c3265942bc12e9b61061cd0bef147a3e067b03c9b80c2f979a1573f7.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/293260d2c3265942bc12e9b61061cd0bef147a3e067b03c9b80c2f979a1573f7.jpg)

![70f5347a3ed8fe5551211470b787e0b68690ccd3d7c2b9a40b895ac037765946.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/70f5347a3ed8fe5551211470b787e0b68690ccd3d7c2b9a40b895ac037765946.jpg)

![8b72c72ffa42c1d0d20c344d08086e7c7d0e5be86b3de6564c036f2f53d0ae4d.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/8b72c72ffa42c1d0d20c344d08086e7c7d0e5be86b3de6564c036f2f53d0ae4d.jpg)

![99094bd411a31ce4570e2b925b5b9b988e885b05c9fc8e41ea0efd2e4bca7be1.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/99094bd411a31ce4570e2b925b5b9b988e885b05c9fc8e41ea0efd2e4bca7be1.jpg)

![c8971da720c12217d9fa719b44d0ddd9d385b17ab7fece5fe16219823f2318cd.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/c8971da720c12217d9fa719b44d0ddd9d385b17ab7fece5fe16219823f2318cd.jpg)

![e45338282ff89b3b3be18a34dddbf608390fcd64ea98f6a833b1272a6e341739.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/e45338282ff89b3b3be18a34dddbf608390fcd64ea98f6a833b1272a6e341739.jpg)

![f5f0580c7dcb908a6c013b8e130efa9c8da58d02f687b6f72e0b25f060110512.jpg](../icml_results/4522_When%20Generalized%20Zero-Shot%20Learning%20Meets%20PU%20Learning_%20A%20Plug-and-Play%20Framework%20for%20Seen-Class%20Bias/tables/f5f0580c7dcb908a6c013b8e130efa9c8da58d02f687b6f72e0b25f060110512.jpg)

<a id="paper-4476"></a>

## Federated Data and Feature Selection by Generalized CUR Decomposition


### Images

![35cebc114deac4757e6d2d98e3692b1abe9423d92d0ac43d31abfc30e965ccc6.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/images/35cebc114deac4757e6d2d98e3692b1abe9423d92d0ac43d31abfc30e965ccc6.jpg)

![6a6e3c25ebf41e1274f81f37673bce0014ca8a57c8caf211d7cefdc1cd43ed6f.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/images/6a6e3c25ebf41e1274f81f37673bce0014ca8a57c8caf211d7cefdc1cd43ed6f.jpg)

### Tables

![3fb00aba60a6cfa604356be72ccbe5e1cf9012f0242603675223650fa191cd40.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/3fb00aba60a6cfa604356be72ccbe5e1cf9012f0242603675223650fa191cd40.jpg)

![4606ee29726d49ba2797ae55c3e646d044b3274fd8a19b14971eae91517461cf.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/4606ee29726d49ba2797ae55c3e646d044b3274fd8a19b14971eae91517461cf.jpg)

![4df40d37712452a174d4f9dab64789bbbb54a166a3ef4ea33fe61ecbad5dd1c9.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/4df40d37712452a174d4f9dab64789bbbb54a166a3ef4ea33fe61ecbad5dd1c9.jpg)

![6c23486ded2dd66e67d90f1bf54e95d2c7d8eb0233f6c9d3de3dfb63ab9d7932.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/6c23486ded2dd66e67d90f1bf54e95d2c7d8eb0233f6c9d3de3dfb63ab9d7932.jpg)

![bc45e7989824d44a993d9f64ffd2bac7153f2f381a5523c9a9fa2d3752864569.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/bc45e7989824d44a993d9f64ffd2bac7153f2f381a5523c9a9fa2d3752864569.jpg)

![bfb619cf4b320254a60192253a0658b31122a36e09b225d53369b1fe3bc91932.jpg](../icml_results/4523_Federated%20Data%20and%20Feature%20Selection%20by%20Generalized%20CUR%20Decomposition/tables/bfb619cf4b320254a60192253a0658b31122a36e09b225d53369b1fe3bc91932.jpg)

<a id="paper-4477"></a>

## The Extra Tokens Matter: Disentangled Representation Learning with Vision Transformers


### Images

![4869cb5f17bf8e431c446dfe0da3c611aebbf47dee90aeed03ec4de3af0870c8.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/4869cb5f17bf8e431c446dfe0da3c611aebbf47dee90aeed03ec4de3af0870c8.jpg)

![53decd3af12b7a808fa751b0037666e9cfd1964ddee90a797a2a15f6a8b08ece.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/53decd3af12b7a808fa751b0037666e9cfd1964ddee90a797a2a15f6a8b08ece.jpg)

![60aab5278fa00759902d5124b26023a945cca7d367f857afbb8ad3015e2b4c51.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/60aab5278fa00759902d5124b26023a945cca7d367f857afbb8ad3015e2b4c51.jpg)

![887268e164a5c7192d86d77879547a284e0c04afcef7b4b246dbf57b6e032805.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/887268e164a5c7192d86d77879547a284e0c04afcef7b4b246dbf57b6e032805.jpg)

![ac9006fa452bd4147e1db778a3417d3a28776e89e1686b316fcd52d70d5ed21c.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/ac9006fa452bd4147e1db778a3417d3a28776e89e1686b316fcd52d70d5ed21c.jpg)

![ad14c717ea77d1927f383c6a4e5f95d041667bfe7da3d1b7837798c0efe59a5b.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/ad14c717ea77d1927f383c6a4e5f95d041667bfe7da3d1b7837798c0efe59a5b.jpg)

![c88aa6c9700e3d036acf4fef9c3dafea2bee7eaf9dff34cb160e481e26843ca6.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/c88aa6c9700e3d036acf4fef9c3dafea2bee7eaf9dff34cb160e481e26843ca6.jpg)

![da4f8c22ac3b6735a842242aff0d5ac4718142bb7690fc087a5624bc6fc0975c.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/da4f8c22ac3b6735a842242aff0d5ac4718142bb7690fc087a5624bc6fc0975c.jpg)

![fae418223d27bfb24cbf9feeb381eede005a8558da272ed15c9a2b7a283a79ef.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/images/fae418223d27bfb24cbf9feeb381eede005a8558da272ed15c9a2b7a283a79ef.jpg)

### Tables

![1c5753f5ece50603e0fd9fe9333119d88fbcc340dd268b766be61e9f70f9fc87.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/1c5753f5ece50603e0fd9fe9333119d88fbcc340dd268b766be61e9f70f9fc87.jpg)

![1e25f1dbaabf62b3f7218b13de293c545c565a6ffc01bc46a6a1e941a077bd92.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/1e25f1dbaabf62b3f7218b13de293c545c565a6ffc01bc46a6a1e941a077bd92.jpg)

![20c9312dfe1a3a5af4a7c085328515663678181d57ffe07894cee37a594303a8.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/20c9312dfe1a3a5af4a7c085328515663678181d57ffe07894cee37a594303a8.jpg)

![21c25c066455f54565766ace40af1bc6cadfa4ec7523dc4ef17185f3294e3757.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/21c25c066455f54565766ace40af1bc6cadfa4ec7523dc4ef17185f3294e3757.jpg)

![233f6896bcefd492643e025b15f50a8db4f8dd1569a49131088fcd65f418e61d.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/233f6896bcefd492643e025b15f50a8db4f8dd1569a49131088fcd65f418e61d.jpg)

![2ad9639f014397744535d88d42f008e5d63ffc1b9638392515a1aee58be8ad71.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/2ad9639f014397744535d88d42f008e5d63ffc1b9638392515a1aee58be8ad71.jpg)

![33c5a92b8d3dc43c3ef65e09516db71b26542019c2a85e2622f59b680538aef9.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/33c5a92b8d3dc43c3ef65e09516db71b26542019c2a85e2622f59b680538aef9.jpg)

![4f6fc45301cbe96ba95f978f0c42abc02c8a37516b0b018c53603d61b22c60a7.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/4f6fc45301cbe96ba95f978f0c42abc02c8a37516b0b018c53603d61b22c60a7.jpg)

![66f54feac9b92776356180b3d576d20491f15b23cb3aa92f3b44a051049eb162.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/66f54feac9b92776356180b3d576d20491f15b23cb3aa92f3b44a051049eb162.jpg)

![67093f65270eb0dc7ca0f1877039fa944ce15955b783d8517cca121e2beba17b.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/67093f65270eb0dc7ca0f1877039fa944ce15955b783d8517cca121e2beba17b.jpg)

![6a4501b5ccd28a58554a007f3b0c3cbcd919921bd9e20762b4b08fb121bb928d.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/6a4501b5ccd28a58554a007f3b0c3cbcd919921bd9e20762b4b08fb121bb928d.jpg)

![6d88b08c3e61f6d475e735777029fb890c32d39950fd13b6f7a871e07857aecb.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/6d88b08c3e61f6d475e735777029fb890c32d39950fd13b6f7a871e07857aecb.jpg)

![71f2b28b9e7e7c99d28e9f352b8402ba06a4939c2e06b7a3257a1922b452c0d5.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/71f2b28b9e7e7c99d28e9f352b8402ba06a4939c2e06b7a3257a1922b452c0d5.jpg)

![7573f29129088c0d2c75a82eeb3dcf3dadc7f39ccedc8a677721fbb2f7739292.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/7573f29129088c0d2c75a82eeb3dcf3dadc7f39ccedc8a677721fbb2f7739292.jpg)

![8539fae6a47348c99edf78ed5fffe3748519bd1bf50f37843396600ddd7187f5.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/8539fae6a47348c99edf78ed5fffe3748519bd1bf50f37843396600ddd7187f5.jpg)

![8c99aa5dc9f7c82acdf1fb24132c485bcbb9f57c4b7f3f459670f0e00de9e6a8.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/8c99aa5dc9f7c82acdf1fb24132c485bcbb9f57c4b7f3f459670f0e00de9e6a8.jpg)

![8db3f49ce630a12967395b10a50a91e7b05aecc34d7d1ab25c65caf6d95c58ce.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/8db3f49ce630a12967395b10a50a91e7b05aecc34d7d1ab25c65caf6d95c58ce.jpg)

![9ac221043dc7dbe98f7c071b70912d943c4569eb1ff8abfa432b7af25f1a1c6b.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/9ac221043dc7dbe98f7c071b70912d943c4569eb1ff8abfa432b7af25f1a1c6b.jpg)

![afb357a845703f90e28e226ebfcbc752c27c4de0ce6f3ccb02b79ae52e30d22c.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/afb357a845703f90e28e226ebfcbc752c27c4de0ce6f3ccb02b79ae52e30d22c.jpg)

![b59f80938c32c39b5cc5378e0fbc495525f21e3eb644f576744c090a8558e267.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/b59f80938c32c39b5cc5378e0fbc495525f21e3eb644f576744c090a8558e267.jpg)

![da31d9228e1826c4f47697a4909216edd0c336024a677b0971e0b998425f2e40.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/da31d9228e1826c4f47697a4909216edd0c336024a677b0971e0b998425f2e40.jpg)

![e0df4bdb1b4927f2c540e556859ecaf333e0403ac4a4e3a86f3df4eee1780d7d.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/e0df4bdb1b4927f2c540e556859ecaf333e0403ac4a4e3a86f3df4eee1780d7d.jpg)

![ea64be69ab0fd6356cee9bd3b1a31b9e37320917cfc96b039000f9fe35a83214.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/ea64be69ab0fd6356cee9bd3b1a31b9e37320917cfc96b039000f9fe35a83214.jpg)

![f593b29384007910a904a26a47a6a2a562974b98b116366ca3d0eb27f19e22e4.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/f593b29384007910a904a26a47a6a2a562974b98b116366ca3d0eb27f19e22e4.jpg)

![f72bfcca2ae4e439c9398339161447d35ea0bcb2dde8e7b845a4d755f9e42116.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/f72bfcca2ae4e439c9398339161447d35ea0bcb2dde8e7b845a4d755f9e42116.jpg)

![f7fa9dabf55ee3bb7e5def08584994933eb87ba8ff7ed70bc7f1204c33ed8d44.jpg](../icml_results/4524_The%20Extra%20Tokens%20Matter_%20Disentangled%20Representation%20Learning%20with%20Vision%20Transformers/tables/f7fa9dabf55ee3bb7e5def08584994933eb87ba8ff7ed70bc7f1204c33ed8d44.jpg)

<a id="paper-4478"></a>

## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers


### Images

![0168fcf6d73e25069367c594ad31d3f6ee36f4f8aecdb975b7299a8b185007a9.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/0168fcf6d73e25069367c594ad31d3f6ee36f4f8aecdb975b7299a8b185007a9.jpg)

![0f7195ed1af84c5b75fdad81d86ec23e51575a3b42593ac4e0448ecf2a04162a.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/0f7195ed1af84c5b75fdad81d86ec23e51575a3b42593ac4e0448ecf2a04162a.jpg)

![135d232e083336422034ade4b4c0351148d0baffbc27f340ad01208ac4800ca6.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/135d232e083336422034ade4b4c0351148d0baffbc27f340ad01208ac4800ca6.jpg)

![1b99a8bfcbc437399af91022d6cc39c2e277fadbebc7804c465b4d4663bc89fc.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/1b99a8bfcbc437399af91022d6cc39c2e277fadbebc7804c465b4d4663bc89fc.jpg)

![4c57bfb1c335041b00e6b62057d8c913666d5cba5748ca2cd46af0ab825d0ab5.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/4c57bfb1c335041b00e6b62057d8c913666d5cba5748ca2cd46af0ab825d0ab5.jpg)

![5732e46e051567337f2dde32c5ead260871862251e070e4c1d4d0511d11eb257.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/5732e46e051567337f2dde32c5ead260871862251e070e4c1d4d0511d11eb257.jpg)

![595b8ed3954de49c51d6d8006400547c2042b07b30dee6a2ca7b15bc99362175.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/595b8ed3954de49c51d6d8006400547c2042b07b30dee6a2ca7b15bc99362175.jpg)

![685b1ea294521ee066039d6c2c08128cdb52ddcfb5728b76a515846e9f152011.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/685b1ea294521ee066039d6c2c08128cdb52ddcfb5728b76a515846e9f152011.jpg)

![91b674a5ae4ca5f8fce7880fe8e131d56415d3aedd4e9845ed8ecd9b7a699dbd.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/91b674a5ae4ca5f8fce7880fe8e131d56415d3aedd4e9845ed8ecd9b7a699dbd.jpg)

![af151283bce5472f10fa94ae9b66ea1a7d672748def619a31749897d01254938.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/af151283bce5472f10fa94ae9b66ea1a7d672748def619a31749897d01254938.jpg)

![b01234ae41da4f3f28ca0aa52ba1ac9969002e1675e157a51d17b7c297f69060.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/b01234ae41da4f3f28ca0aa52ba1ac9969002e1675e157a51d17b7c297f69060.jpg)

![c0e0c5e39174a87c86f728c42ca5fbc6a8946186bcf570a5636ed270feebce88.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/c0e0c5e39174a87c86f728c42ca5fbc6a8946186bcf570a5636ed270feebce88.jpg)

![d955c230a4be340ad04c0088c7d48abaa6c805a0897458198f1c9f2a1409259f.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/d955c230a4be340ad04c0088c7d48abaa6c805a0897458198f1c9f2a1409259f.jpg)

![ee05c5d090ef7ebce222dcdea35b737e64561c1c5b9decc77a1cf1abffa111ab.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/ee05c5d090ef7ebce222dcdea35b737e64561c1c5b9decc77a1cf1abffa111ab.jpg)

![fd7851ed178aa648a86a62463a48e0f3eaf4aa07d131ee6b10e8c86ee8dc0514.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/images/fd7851ed178aa648a86a62463a48e0f3eaf4aa07d131ee6b10e8c86ee8dc0514.jpg)

### Tables

![17500b5210add724b6a727abdbb1183238e1ffe5f594fd087f20820545a2be54.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/17500b5210add724b6a727abdbb1183238e1ffe5f594fd087f20820545a2be54.jpg)

![1d3ba733c65f081b19160ef5bcfd208a617a800113f3fce28e4dffd699b58f59.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/1d3ba733c65f081b19160ef5bcfd208a617a800113f3fce28e4dffd699b58f59.jpg)

![20775f3d70b79bd9413444a7db5c4a2b0566d8e427d39a2a56b23fa1831d43b1.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/20775f3d70b79bd9413444a7db5c4a2b0566d8e427d39a2a56b23fa1831d43b1.jpg)

![33faeeea370fdd1d7f83f11e488b40cbfbabd33ff767a2d6ae579f7f43cb546f.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/33faeeea370fdd1d7f83f11e488b40cbfbabd33ff767a2d6ae579f7f43cb546f.jpg)

![36261d0a4c02e389dbe1538f6778fc0d464a96d9e87165d90783c93c5b8216dc.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/36261d0a4c02e389dbe1538f6778fc0d464a96d9e87165d90783c93c5b8216dc.jpg)

![36f9f3339345e4e8ed240d7e806088cff0ad641b2102bd20774ec876ea67ea44.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/36f9f3339345e4e8ed240d7e806088cff0ad641b2102bd20774ec876ea67ea44.jpg)

![37429b1173f1e2f06bc962c1ea09b9bc519fb740faccca47bf6f00dbea1979e8.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/37429b1173f1e2f06bc962c1ea09b9bc519fb740faccca47bf6f00dbea1979e8.jpg)

![4d064a57c516118e3ffac9f111cf9020f48ceca5ebd34ad85e8a2543ef2d5342.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/4d064a57c516118e3ffac9f111cf9020f48ceca5ebd34ad85e8a2543ef2d5342.jpg)

![52631e9fbfd2e3e68387b0c7fc582aad875898a90077233a9c0899d70f2fd3ed.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/52631e9fbfd2e3e68387b0c7fc582aad875898a90077233a9c0899d70f2fd3ed.jpg)

![5aa1331bdabef06cb88d8a098af1f6d88b05898110b6c79a97344a64529d1fab.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/5aa1331bdabef06cb88d8a098af1f6d88b05898110b6c79a97344a64529d1fab.jpg)

![6e391cd21aec3275c73b3e020cdadf6fae495aec8e6f86535cc878fc1c1ccba9.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/6e391cd21aec3275c73b3e020cdadf6fae495aec8e6f86535cc878fc1c1ccba9.jpg)

![903225443ee437b511ce84577a57601e78159128b8c4fe4724606de5e1b2e0ad.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/903225443ee437b511ce84577a57601e78159128b8c4fe4724606de5e1b2e0ad.jpg)

![a5cfbc752e65c8c0998ef0144614d443de03220ab762ffcf4d08b2ff5959484e.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/a5cfbc752e65c8c0998ef0144614d443de03220ab762ffcf4d08b2ff5959484e.jpg)

![d4d5880d3aa905ae80a3b5c494589475c503d02129756919b5604926d806824c.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/d4d5880d3aa905ae80a3b5c494589475c503d02129756919b5604926d806824c.jpg)

![e6ed644299a0a139dc68231eb0e6aa1ba7f93ad0a4aceef4c574a6367bb2ccdf.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/e6ed644299a0a139dc68231eb0e6aa1ba7f93ad0a4aceef4c574a6367bb2ccdf.jpg)

![f2a7a524e632c6a5cf79479bada78f8a0ce3c5fdd82a6626bb676e2c01280e14.jpg](../icml_results/4525_Elastic%20Attention_%20Test-time%20Adaptive%20Sparsity%20Ratios%20for%20Efficient%20Transformers/tables/f2a7a524e632c6a5cf79479bada78f8a0ce3c5fdd82a6626bb676e2c01280e14.jpg)

<a id="paper-4479"></a>

## Artemis: Structured Visual Reasoning for Perception Policy Learning


### Images

![07026afe043d79d9f82397f1a455809d240e1ecc987e2ebd704cbdffef90a30a.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/07026afe043d79d9f82397f1a455809d240e1ecc987e2ebd704cbdffef90a30a.jpg)

![1386d62e11877b7edde38b45a3895e38cf81c176ab926eab63e0c2bb6fbe37f8.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/1386d62e11877b7edde38b45a3895e38cf81c176ab926eab63e0c2bb6fbe37f8.jpg)

![1a83aeaa444e005f7119c87f822985fe80e814d171e6603aa72fe138c68dd54d.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/1a83aeaa444e005f7119c87f822985fe80e814d171e6603aa72fe138c68dd54d.jpg)

![1f547742fe90433a7f79f393bbf9bbcd8e97186ead6b0baa652fabf9c91dbfc4.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/1f547742fe90433a7f79f393bbf9bbcd8e97186ead6b0baa652fabf9c91dbfc4.jpg)

![4b62ab348811fd577f963652a0ba71e8417b22b5c3f27742b269ed9b6b3b7fe8.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/4b62ab348811fd577f963652a0ba71e8417b22b5c3f27742b269ed9b6b3b7fe8.jpg)

![561e2dee9e747a3925fa5c53e95f5081a16cac096c88a8060aafb9ac4e9eeae4.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/561e2dee9e747a3925fa5c53e95f5081a16cac096c88a8060aafb9ac4e9eeae4.jpg)

![5e28f8a53ee67eba1b921867059ee32ea5fad6c738112e9c561d99176124553c.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/5e28f8a53ee67eba1b921867059ee32ea5fad6c738112e9c561d99176124553c.jpg)

![6856187c24e6675e27af7cc8974d1bfab6233ae58b9945bcab6e8a8b54ab9aed.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/6856187c24e6675e27af7cc8974d1bfab6233ae58b9945bcab6e8a8b54ab9aed.jpg)

![706990462dae5cd09b2e6f018ad782360911d98c70908ffcc8850ee429fdecc2.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/706990462dae5cd09b2e6f018ad782360911d98c70908ffcc8850ee429fdecc2.jpg)

![7a5d42a77ed147f8f1016172f8f6ff3eb5a27a9d8e871dde7352848f3fd3fbdd.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/7a5d42a77ed147f8f1016172f8f6ff3eb5a27a9d8e871dde7352848f3fd3fbdd.jpg)

![824861d8fc023f198131aa8e595214cc235ead22fd5e18d7787d32245e1f30a3.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/824861d8fc023f198131aa8e595214cc235ead22fd5e18d7787d32245e1f30a3.jpg)

![8bd3efc57510a938fcfc646640f7a5fea9112584ad2b5f3804b9afa5aac3f06d.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/8bd3efc57510a938fcfc646640f7a5fea9112584ad2b5f3804b9afa5aac3f06d.jpg)

![90c45c5aaa1953e20e7cd4c20e68c1625453909cb26ed5708b61277676dc817a.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/90c45c5aaa1953e20e7cd4c20e68c1625453909cb26ed5708b61277676dc817a.jpg)

![ad1e74fc35f1bba75b50445eafdc3738dea17065018b3c8551feb2ba6f781188.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/ad1e74fc35f1bba75b50445eafdc3738dea17065018b3c8551feb2ba6f781188.jpg)

![ae54a30a18dd9f9cbc38fcda9c64eeff7eb12fce5679f9e9f12cf718f0871d02.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/ae54a30a18dd9f9cbc38fcda9c64eeff7eb12fce5679f9e9f12cf718f0871d02.jpg)

![bcab14f9181240e2c7c4c8340befab117d28dc569e72a70078ba9995348f81f2.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/bcab14f9181240e2c7c4c8340befab117d28dc569e72a70078ba9995348f81f2.jpg)

![bea550d52b3ca8a6402e853e2db180343a5a5399d57215161e8df79eb4fe11cd.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/bea550d52b3ca8a6402e853e2db180343a5a5399d57215161e8df79eb4fe11cd.jpg)

![bfba4590a03143c6fc492084f81a703b8caefe748a2607a331c0a3ce8a7d55bc.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/bfba4590a03143c6fc492084f81a703b8caefe748a2607a331c0a3ce8a7d55bc.jpg)

![c78adde0e67a80d2a775e0db167d11a8f3a411c26f3ecf5f7c05ecbac4cd4384.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/c78adde0e67a80d2a775e0db167d11a8f3a411c26f3ecf5f7c05ecbac4cd4384.jpg)

![d50c2eb6ecb46cb1c8b8b0829c4b483b734d20989caf2fd9b579fd0e1db9164e.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/d50c2eb6ecb46cb1c8b8b0829c4b483b734d20989caf2fd9b579fd0e1db9164e.jpg)

![d92319195a2f2b212f917e73e1181d1cc906654a1f21145ca24012fea5674030.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/d92319195a2f2b212f917e73e1181d1cc906654a1f21145ca24012fea5674030.jpg)

![e06c0ef9cdb4758bf015a70bfd08dedccae9fb12406a25fc029a5bbd45e4c7c6.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/e06c0ef9cdb4758bf015a70bfd08dedccae9fb12406a25fc029a5bbd45e4c7c6.jpg)

![e7bd302cf6388034560f17169cfeb7646f5d0eb498cc6377366b2beba820bcf1.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/e7bd302cf6388034560f17169cfeb7646f5d0eb498cc6377366b2beba820bcf1.jpg)

![f203482eb90b86f51b746b4a9e00494e3279ff22d49720b51e2cb5c9c3384cb0.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/f203482eb90b86f51b746b4a9e00494e3279ff22d49720b51e2cb5c9c3384cb0.jpg)

![f51ffe47ff49835c6f92c6fdee2552e68aaa3066e053027b85f1213c76b870df.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/images/f51ffe47ff49835c6f92c6fdee2552e68aaa3066e053027b85f1213c76b870df.jpg)

### Tables

![15a02422ff8cfd25d68e46f922624c3abb2d516b01835ec99884b104ed98def9.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/15a02422ff8cfd25d68e46f922624c3abb2d516b01835ec99884b104ed98def9.jpg)

![1a980f66880d5500779b7986aab2835fe6d7108b8db0c532276c4d269381e1a1.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/1a980f66880d5500779b7986aab2835fe6d7108b8db0c532276c4d269381e1a1.jpg)

![46ac51e5a498a03141d79069b5b4b9cc7a5b886c8c3a78fd62dc396dd583cf30.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/46ac51e5a498a03141d79069b5b4b9cc7a5b886c8c3a78fd62dc396dd583cf30.jpg)

![47b24cf9b980eb6e28dc1c4f60d9ed58cad6b17c00198b9a0083533f579f768b.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/47b24cf9b980eb6e28dc1c4f60d9ed58cad6b17c00198b9a0083533f579f768b.jpg)

![6f30883cc64977bdd2bc5976edd49d49e86787f5c2c28d6a3a79e16266cd595e.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/6f30883cc64977bdd2bc5976edd49d49e86787f5c2c28d6a3a79e16266cd595e.jpg)

![9370caf652ee4502bd3193ea34b5ab9f6d2b6df9c597bd22ca0ed70fd7e157f0.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/9370caf652ee4502bd3193ea34b5ab9f6d2b6df9c597bd22ca0ed70fd7e157f0.jpg)

![a42ea24259c40d1d95c5d41ba711eef68c54eb9b786fb7e71ef8e543403a1d85.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/a42ea24259c40d1d95c5d41ba711eef68c54eb9b786fb7e71ef8e543403a1d85.jpg)

![bcd3b68d007f82eacd7e0214525317e03035846d049b9371ac3722a45ce06d35.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/bcd3b68d007f82eacd7e0214525317e03035846d049b9371ac3722a45ce06d35.jpg)

![dd48a8c9da1809b6071554e4c796bc2e7c62fbdce03e8d359c801fd2c1898a5d.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/dd48a8c9da1809b6071554e4c796bc2e7c62fbdce03e8d359c801fd2c1898a5d.jpg)

![e977807e7fc6d7af42438d345bb71f30664fba79cbb4da349d979c3e50beab2d.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/e977807e7fc6d7af42438d345bb71f30664fba79cbb4da349d979c3e50beab2d.jpg)

![fbf2e879e7a261cad01a8e7a801d477302bd788b1bcf681fc998359610fcc4ac.jpg](../icml_results/4526_Artemis_%20Structured%20Visual%20Reasoning%20for%20Perception%20Policy%20Learning/tables/fbf2e879e7a261cad01a8e7a801d477302bd788b1bcf681fc998359610fcc4ac.jpg)

<a id="paper-4480"></a>

## Rational Neural Networks have Expressivity Advantages


### Images

![121356804867cb24c8b736a53e8ca32f2658e5e0cd767728f9d07e8697c62fd8.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/121356804867cb24c8b736a53e8ca32f2658e5e0cd767728f9d07e8697c62fd8.jpg)

![1f1803e247aae5b9656ea4205005459f59f5344b62d04323254ffffcec54a557.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/1f1803e247aae5b9656ea4205005459f59f5344b62d04323254ffffcec54a557.jpg)

![24e4051f7181241d7ceb2eae6b5614caf8bbfcf67bf1ecd4e7b886e0f1b8ee0f.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/24e4051f7181241d7ceb2eae6b5614caf8bbfcf67bf1ecd4e7b886e0f1b8ee0f.jpg)

![469f2365bfdb29280149d3a5a9ba8c3f5fcc80392257e82d467b0f2d2a0bc740.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/469f2365bfdb29280149d3a5a9ba8c3f5fcc80392257e82d467b0f2d2a0bc740.jpg)

![4f793a10528cf6a569162271052b2e4ee7edab6a83cd9dc2d0babb8b7a69faff.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/4f793a10528cf6a569162271052b2e4ee7edab6a83cd9dc2d0babb8b7a69faff.jpg)

![668ba8766b10ba94c7a4098a2783fc6fbc9e6f31dcddf3632858399a87d3e4b7.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/668ba8766b10ba94c7a4098a2783fc6fbc9e6f31dcddf3632858399a87d3e4b7.jpg)

![a898547a5522ff5d0e96cd21ff8e3956bbd3c056c8488275a3a7d1687de3fbe4.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/a898547a5522ff5d0e96cd21ff8e3956bbd3c056c8488275a3a7d1687de3fbe4.jpg)

![b1cd424e194c00173235ebe50598d66018bcf13bc0a888c539327735842cc600.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/b1cd424e194c00173235ebe50598d66018bcf13bc0a888c539327735842cc600.jpg)

![c6ae7589bf07a06369ecf90457e1ed4fcd453314a7974a817b92830f27f4bcb4.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/c6ae7589bf07a06369ecf90457e1ed4fcd453314a7974a817b92830f27f4bcb4.jpg)

![ca831b89e938a83f2ea85230721f733beb528a7e28c9e328f65ba52681117ad8.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/ca831b89e938a83f2ea85230721f733beb528a7e28c9e328f65ba52681117ad8.jpg)

![cdbf14848bc85fff5fbabb92b1ffbb318954a68dfcd6c84dc0811fc53930bfef.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/images/cdbf14848bc85fff5fbabb92b1ffbb318954a68dfcd6c84dc0811fc53930bfef.jpg)

### Tables

![268993dae9d01d9924cf99b0268835dabfc5b9c13dde23830d1d7a9bc3832809.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/268993dae9d01d9924cf99b0268835dabfc5b9c13dde23830d1d7a9bc3832809.jpg)

![28f0f205d5f17b6a2c2aa82e691f2c17d3d70131e6a06e7434f43fb9458a9f0e.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/28f0f205d5f17b6a2c2aa82e691f2c17d3d70131e6a06e7434f43fb9458a9f0e.jpg)

![3dd14fb18a336079d3539afdcfc873277bcd6e4eefb3a7af0e904b8e63a2e5a0.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/3dd14fb18a336079d3539afdcfc873277bcd6e4eefb3a7af0e904b8e63a2e5a0.jpg)

![462c0bc1e9cefddeec8911db75bbc14db1ccb9f00ec5e9b2afd89cce485e965e.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/462c0bc1e9cefddeec8911db75bbc14db1ccb9f00ec5e9b2afd89cce485e965e.jpg)

![5aeefb091646bf3079269f6a5de0209a8b818db2daee9460f1eda2b57e0132f3.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/5aeefb091646bf3079269f6a5de0209a8b818db2daee9460f1eda2b57e0132f3.jpg)

![75d15020158ffcf5d6260fc65499e62c552490b994b69b277e1f5ce021918e39.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/75d15020158ffcf5d6260fc65499e62c552490b994b69b277e1f5ce021918e39.jpg)

![9a80d1edbdcb3ef9b9a2450078a93e518176e2005da5cab25a1b233242c1000f.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/9a80d1edbdcb3ef9b9a2450078a93e518176e2005da5cab25a1b233242c1000f.jpg)

![9d4883b70c171a34cefda877ff150a8465013e6f95e0a166674c75df4c88b37c.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/9d4883b70c171a34cefda877ff150a8465013e6f95e0a166674c75df4c88b37c.jpg)

![bd9f339b816aa0ab7e637fa2f27bcac7cfb4a40c445b290617c5dc964b5ba067.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/bd9f339b816aa0ab7e637fa2f27bcac7cfb4a40c445b290617c5dc964b5ba067.jpg)

![d6b0a3457e096950e7b1825541753c76765c9720511a259f420c0bc694c42f81.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/d6b0a3457e096950e7b1825541753c76765c9720511a259f420c0bc694c42f81.jpg)

![f7aabd3d8e4e9ae6e10af5e2090608311dedeadc802104f82458bc03d4f29db1.jpg](../icml_results/4527_Rational%20Neural%20Networks%20have%20Expressivity%20Advantages/tables/f7aabd3d8e4e9ae6e10af5e2090608311dedeadc802104f82458bc03d4f29db1.jpg)

<a id="paper-4481"></a>

## MineDraft: A Framework for Batch Parallel Speculative Decoding


### Images

![11077805d87f9cbbe76c79f0b6bc7413b866c81055b16d99b9315d88b4b42037.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/11077805d87f9cbbe76c79f0b6bc7413b866c81055b16d99b9315d88b4b42037.jpg)

![15109f393e1629c87414e9eaa832ac6ab32bccfc89f7febb47bce5c84c0a5147.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/15109f393e1629c87414e9eaa832ac6ab32bccfc89f7febb47bce5c84c0a5147.jpg)

![2f7f39bc9211a7a1a48eb03f6d759ed8dbc883ccefa8b1cd73219965e6ffa7f2.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/2f7f39bc9211a7a1a48eb03f6d759ed8dbc883ccefa8b1cd73219965e6ffa7f2.jpg)

![353d39fa1edf83e17016bf709e17c1246691f02c08e64a1727f48fcb752e8183.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/353d39fa1edf83e17016bf709e17c1246691f02c08e64a1727f48fcb752e8183.jpg)

![35dd529b29de4ebc450ba3690831bb3339615b6a30c1f6b980353b5efd16718c.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/35dd529b29de4ebc450ba3690831bb3339615b6a30c1f6b980353b5efd16718c.jpg)

![3678832cd1876e6c6d6869a26f352520fa6999569f08375d11638b6a614d4707.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/3678832cd1876e6c6d6869a26f352520fa6999569f08375d11638b6a614d4707.jpg)

![3fd9ddfde1691aa7b138c6a3cd8cdfe2676e7ff227c013de8abab6670039060d.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/3fd9ddfde1691aa7b138c6a3cd8cdfe2676e7ff227c013de8abab6670039060d.jpg)

![480e25de234ceeedb6418b3a6f2d443b3891f77eb01aba1a6baa47b3f31139da.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/480e25de234ceeedb6418b3a6f2d443b3891f77eb01aba1a6baa47b3f31139da.jpg)

![6cdeeea0c4849fb41ba1c38294b08237c7404271435902e4581640ba6415d458.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/6cdeeea0c4849fb41ba1c38294b08237c7404271435902e4581640ba6415d458.jpg)

![6d0d0186a84b160212ef7614a6ca786f12ebc7179276e8312d48bdeceb4fc37e.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/6d0d0186a84b160212ef7614a6ca786f12ebc7179276e8312d48bdeceb4fc37e.jpg)

![729e01091aba16a9ab33f9290d79d6557c3ba16597de10a0b28115e7b5223d99.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/729e01091aba16a9ab33f9290d79d6557c3ba16597de10a0b28115e7b5223d99.jpg)

![8f9769f9a83e73adf730cd6d9e2897f7237623d40fe60976bcd3987ec1e1981e.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/8f9769f9a83e73adf730cd6d9e2897f7237623d40fe60976bcd3987ec1e1981e.jpg)

![9122f0ab4f2c902a3f7df10f9a49d9cdf88e10ce1368e19419cf33bf4cf87f4d.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/9122f0ab4f2c902a3f7df10f9a49d9cdf88e10ce1368e19419cf33bf4cf87f4d.jpg)

![9a174c2d329188d402797fd17e93de112e0b7ef24bb48ad8116a3dd2eb88ed2c.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/9a174c2d329188d402797fd17e93de112e0b7ef24bb48ad8116a3dd2eb88ed2c.jpg)

![9b0a7459308a69bad98662761a790c428845a808371fb42124913ac221e06404.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/9b0a7459308a69bad98662761a790c428845a808371fb42124913ac221e06404.jpg)

![9c141b9d4b5733351d5c6be6317c8a50823b5ddd7b58cde96a4a3993045367bb.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/9c141b9d4b5733351d5c6be6317c8a50823b5ddd7b58cde96a4a3993045367bb.jpg)

![9ebed1a25b6752e580c71c6b70558cac3a4499e250a6722e4a75951f2f1f417f.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/9ebed1a25b6752e580c71c6b70558cac3a4499e250a6722e4a75951f2f1f417f.jpg)

![a40b1453126b0c3acf5aec124d0a06fa75d0864c790dd4b303843d017c051d43.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/a40b1453126b0c3acf5aec124d0a06fa75d0864c790dd4b303843d017c051d43.jpg)

![af781aaecee2b08b6e3d653f4b891874035dbe4dd9796f8e1e485630b6bf6027.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/af781aaecee2b08b6e3d653f4b891874035dbe4dd9796f8e1e485630b6bf6027.jpg)

![bbc7e3765466972a416d70464a66b124c5534b0fc5137af3099f2307573a276b.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/bbc7e3765466972a416d70464a66b124c5534b0fc5137af3099f2307573a276b.jpg)

![c446b0e4061f24122ddae8a03801dac66cc6bbcce3c51ede748d742d3614bc8f.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/c446b0e4061f24122ddae8a03801dac66cc6bbcce3c51ede748d742d3614bc8f.jpg)

![c6c620f23e3e85679b872dcb004637cd7e0b0aca059f0e80e9f319392d95d0ae.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/c6c620f23e3e85679b872dcb004637cd7e0b0aca059f0e80e9f319392d95d0ae.jpg)

![dc23dcb59dabff8888cee02de350fe73f2408f76269794aaf9ad0e63934c5c98.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/dc23dcb59dabff8888cee02de350fe73f2408f76269794aaf9ad0e63934c5c98.jpg)

![e7e11ea7d23a020dd31a75833dfab847603f3819ebf023af042981b76b22914c.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/e7e11ea7d23a020dd31a75833dfab847603f3819ebf023af042981b76b22914c.jpg)

![e933f95004bab0cfee400682f0ad33d0b79808e7ef23ef918db40170c7660155.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/e933f95004bab0cfee400682f0ad33d0b79808e7ef23ef918db40170c7660155.jpg)

![f035a02344d2de17845b870a1c50eff3de08c3fadae4d2ff30e17054684876e0.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/f035a02344d2de17845b870a1c50eff3de08c3fadae4d2ff30e17054684876e0.jpg)

![f80092dde0dcf31b4560ef0ab36aa3e6800899d45819f147df0218817e4f74c3.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/f80092dde0dcf31b4560ef0ab36aa3e6800899d45819f147df0218817e4f74c3.jpg)

![fecf1563f8f5fd82a7c03f4cdd7abae54aa3b1c79995fa1c8731f6041f6399de.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/fecf1563f8f5fd82a7c03f4cdd7abae54aa3b1c79995fa1c8731f6041f6399de.jpg)

![ff8044e63f2f8e9399c3b8ce6ed1e1b608f660bb06415cd4d295cfcf007f203b.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/images/ff8044e63f2f8e9399c3b8ce6ed1e1b608f660bb06415cd4d295cfcf007f203b.jpg)

### Tables

![091b4c1b397e4158a5dd84c83e274f259ff463d1c9bafbe3fc7d688cc59900e9.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/tables/091b4c1b397e4158a5dd84c83e274f259ff463d1c9bafbe3fc7d688cc59900e9.jpg)

![73723c52df7502860daef57a2294e8ee3f9c9233be167a8bfe84069b251de8ca.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/tables/73723c52df7502860daef57a2294e8ee3f9c9233be167a8bfe84069b251de8ca.jpg)

![84559376a6ff5c2639dd30df4706c83216d2be7866977a9730b5bce298876ca1.jpg](../icml_results/4528_MineDraft_%20A%20Framework%20for%20Batch%20Parallel%20Speculative%20Decoding/tables/84559376a6ff5c2639dd30df4706c83216d2be7866977a9730b5bce298876ca1.jpg)

<a id="paper-4482"></a>

## Learning to Rank by Directly Optimizing Full-Order Probabilities


### Images

![38c8c1cd5fda43a27d01d58068888520438cc7687555f03719205d5bc5c68ebf.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/images/38c8c1cd5fda43a27d01d58068888520438cc7687555f03719205d5bc5c68ebf.jpg)

![d3d2e521e3ba3da8de045da66fbf8d5d4be5fd0864d5c04c3bb16eee7aa9953e.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/images/d3d2e521e3ba3da8de045da66fbf8d5d4be5fd0864d5c04c3bb16eee7aa9953e.jpg)

![f65990ddb0eacfcfa73c151ccc88077584f56827576c422c240d1a6d2e88fb64.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/images/f65990ddb0eacfcfa73c151ccc88077584f56827576c422c240d1a6d2e88fb64.jpg)

### Tables

![1c8cb89a50a047e93b608fd84f0478c232127414eb3a7a63f3d999730f10c229.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/1c8cb89a50a047e93b608fd84f0478c232127414eb3a7a63f3d999730f10c229.jpg)

![580d4a69cfe8aabda0a3aec801231ed3124d7fe4ab8f3809757eb741c32e9739.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/580d4a69cfe8aabda0a3aec801231ed3124d7fe4ab8f3809757eb741c32e9739.jpg)

![6fce122ef864c5c3da2f505a897113cc3f9776009910f844230fabfcb0c55587.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/6fce122ef864c5c3da2f505a897113cc3f9776009910f844230fabfcb0c55587.jpg)

![832105bd6d285ce73b5f17c35a923b818035b34a68656d17c945d39089fd2e52.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/832105bd6d285ce73b5f17c35a923b818035b34a68656d17c945d39089fd2e52.jpg)

![88ff20a7b5233060348d41559ee50ba0171248af1074d95d6a388fceb5872a97.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/88ff20a7b5233060348d41559ee50ba0171248af1074d95d6a388fceb5872a97.jpg)

![9a6a5bdb6f6f52fc278eff2880a22f48d3070c8e7d5414a385ceb88afdfd86ca.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/9a6a5bdb6f6f52fc278eff2880a22f48d3070c8e7d5414a385ceb88afdfd86ca.jpg)

![9e329ae51fd3015aa05a966e5b3e5c3a374c0c75728a011bf0116b797510e04a.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/9e329ae51fd3015aa05a966e5b3e5c3a374c0c75728a011bf0116b797510e04a.jpg)

![a7d113a25ca865999a2253809128c419414ee62180b0cf29959bd8f01307a353.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/a7d113a25ca865999a2253809128c419414ee62180b0cf29959bd8f01307a353.jpg)

![b0c40803050955294e5595aac9d5d6883db16c4905127420335081a97bc9423c.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/b0c40803050955294e5595aac9d5d6883db16c4905127420335081a97bc9423c.jpg)

![bc0a34662d5b45eede19a719ceffa9973afcc82566fd73fbfcb58d1584c2f9d7.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/bc0a34662d5b45eede19a719ceffa9973afcc82566fd73fbfcb58d1584c2f9d7.jpg)

![c6abcc4da27aca92d26b1539b3a23330f0d3b351745109ece777a2c9fc9b07c1.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/c6abcc4da27aca92d26b1539b3a23330f0d3b351745109ece777a2c9fc9b07c1.jpg)

![d15c5083d42d16fb191c8b1ead76ce4819af3f584b83e477d566ccde08ee8d49.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/d15c5083d42d16fb191c8b1ead76ce4819af3f584b83e477d566ccde08ee8d49.jpg)

![df82d23b6f1ed7a77d6c37b6228e9a166a17fb4b80a9cd96c48b70562af3244c.jpg](../icml_results/4529_Learning%20to%20Rank%20by%20Directly%20Optimizing%20Full-Order%20Probabilities/tables/df82d23b6f1ed7a77d6c37b6228e9a166a17fb4b80a9cd96c48b70562af3244c.jpg)

<a id="paper-4483"></a>

## SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection


### Images

![0bb3e2f7888e5c047e0f2ea8097acb3399be923e2236973d27c4883cebc9d55e.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/images/0bb3e2f7888e5c047e0f2ea8097acb3399be923e2236973d27c4883cebc9d55e.jpg)

![4919c47fbc95de776c47e716dce00d1458c3a0be14931e099e2cd738bac22310.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/images/4919c47fbc95de776c47e716dce00d1458c3a0be14931e099e2cd738bac22310.jpg)

![65de0d7949e5784c9f0a4d478684a87f82cff11b28420538f0330b198d7337bc.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/images/65de0d7949e5784c9f0a4d478684a87f82cff11b28420538f0330b198d7337bc.jpg)

![afa781daeba66826cdd0a64bacb0806a1724e4675783a61204e46ce432f64724.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/images/afa781daeba66826cdd0a64bacb0806a1724e4675783a61204e46ce432f64724.jpg)

### Tables

![12b7c2b8d7b9ef20669a616f3ae7c522ce7f0d3660e636b799c3fcec1c846315.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/12b7c2b8d7b9ef20669a616f3ae7c522ce7f0d3660e636b799c3fcec1c846315.jpg)

![171635735c224f2917b8c96084c1f0e7d57efddd7dd76a884191f8288ee10e49.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/171635735c224f2917b8c96084c1f0e7d57efddd7dd76a884191f8288ee10e49.jpg)

![3414dd306fdd8db3fe0abe3b2acd88972e41e331bbe23475121a67f023646b4e.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/3414dd306fdd8db3fe0abe3b2acd88972e41e331bbe23475121a67f023646b4e.jpg)

![36c94e4f9fc13a2bccc28c44bcd5c0e70740a36972cecb89985b6a3c7bd6673b.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/36c94e4f9fc13a2bccc28c44bcd5c0e70740a36972cecb89985b6a3c7bd6673b.jpg)

![5c79c0053256007329e31c15e6bab8cb59c1009ca6d3e1cab530286af36d7a1b.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/5c79c0053256007329e31c15e6bab8cb59c1009ca6d3e1cab530286af36d7a1b.jpg)

![83dc31e0c8ee6b4321364ed4e5e9165720924960e2d8ec183ac604b5e38ad77b.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/83dc31e0c8ee6b4321364ed4e5e9165720924960e2d8ec183ac604b5e38ad77b.jpg)

![8aa1a25b6d1135b564bfaa286059b2092e8885045c2b3522db89fc2a5f502738.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/8aa1a25b6d1135b564bfaa286059b2092e8885045c2b3522db89fc2a5f502738.jpg)

![b81fb98763cce797fc05946a00245442b095ddc87ae3f5b74835b99170be8897.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/b81fb98763cce797fc05946a00245442b095ddc87ae3f5b74835b99170be8897.jpg)

![c8065789e7093674d0e0811b20bdff62ddac3c4dd5e8c01f8fe0e7473c28428c.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/c8065789e7093674d0e0811b20bdff62ddac3c4dd5e8c01f8fe0e7473c28428c.jpg)

![f66b1f3ab05d05d3e2ae75cd898a8ac74a39fd9e4d91347da0e2dd896e230156.jpg](../icml_results/4530_SPA_%20A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Knowledge%20Injection/tables/f66b1f3ab05d05d3e2ae75cd898a8ac74a39fd9e4d91347da0e2dd896e230156.jpg)

<a id="paper-4484"></a>

## Thinking in Latent Space: Progressive Multimodal Simplification for Visual Reasoning


### Images

![0aee60a538fe5f0db6353e2096b9891ecc72c8d3a07ef8800a264d796ea41985.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/images/0aee60a538fe5f0db6353e2096b9891ecc72c8d3a07ef8800a264d796ea41985.jpg)

![3463503ae407e75a929d17da221d24f7e3257558e400ab33645ce8af7e689ed0.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/images/3463503ae407e75a929d17da221d24f7e3257558e400ab33645ce8af7e689ed0.jpg)

![5e711761ef7e43cda0b45b99f51991ae19e0a1d7fe938fe6d721d3c602822798.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/images/5e711761ef7e43cda0b45b99f51991ae19e0a1d7fe938fe6d721d3c602822798.jpg)

![daeefe3e0ebf5c5ad2be2c4839a23c316da02dfc5e7d0963b28f373ab1118b2b.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/images/daeefe3e0ebf5c5ad2be2c4839a23c316da02dfc5e7d0963b28f373ab1118b2b.jpg)

![f57dc4688b62bd7263e4bb617dc81aaca9d572d33c38b9a1ddbf5b3004910cd6.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/images/f57dc4688b62bd7263e4bb617dc81aaca9d572d33c38b9a1ddbf5b3004910cd6.jpg)

### Tables

![10cdfb980cb2d931c2169770200ae386967be87b03a523c6d0052d3a60d088a5.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/tables/10cdfb980cb2d931c2169770200ae386967be87b03a523c6d0052d3a60d088a5.jpg)

![59a1861fd44e1739ff57061a51c10382f3098f705c887fa12d0c953cda894680.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/tables/59a1861fd44e1739ff57061a51c10382f3098f705c887fa12d0c953cda894680.jpg)

![9efd0146395ed8405512b01e20ee2a5d1f69906593104c4a6a1d023bf8d53b9a.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/tables/9efd0146395ed8405512b01e20ee2a5d1f69906593104c4a6a1d023bf8d53b9a.jpg)

![ff656ce0b6dc70de89afc69c5ebfd971aa603f8484c6c512aa9eae39d46992f1.jpg](../icml_results/4531_Thinking%20in%20Latent%20Space_%20Progressive%20Multimodal%20Simplification%20for%20Visual%20Reasoning/tables/ff656ce0b6dc70de89afc69c5ebfd971aa603f8484c6c512aa9eae39d46992f1.jpg)

<a id="paper-4485"></a>

## CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling


### Images

![080327475f1580a0a28725f3cb3a6dcc38d4f3537eacf8896b5d2fc88b6a42b2.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/080327475f1580a0a28725f3cb3a6dcc38d4f3537eacf8896b5d2fc88b6a42b2.jpg)

![0d897302eaaa82d1d028ee9ef18d34d6cd835e167f4213b1284cfd6980f17991.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/0d897302eaaa82d1d028ee9ef18d34d6cd835e167f4213b1284cfd6980f17991.jpg)

![1b1b0e66844ab2941f80656e737dd588224d8e3cc4842d31675d0344152d1f41.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/1b1b0e66844ab2941f80656e737dd588224d8e3cc4842d31675d0344152d1f41.jpg)

![39cda45ac2a5cda848ec9669486414e5e0cc04cdd01afe9309f464e1c20c1ae4.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/39cda45ac2a5cda848ec9669486414e5e0cc04cdd01afe9309f464e1c20c1ae4.jpg)

![3fa5fb4e0bc88ec0ba91212d5f427b5a0e0e9bc8795d28932eb2c16ccc662222.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/3fa5fb4e0bc88ec0ba91212d5f427b5a0e0e9bc8795d28932eb2c16ccc662222.jpg)

![4226da0d1fb5e7a905211e161bfd880ca1b7fa016fe9b6e6a9ed56e9b0ecbae5.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/4226da0d1fb5e7a905211e161bfd880ca1b7fa016fe9b6e6a9ed56e9b0ecbae5.jpg)

![6bde3984f0e0689ca0a2565c9678e41f7c43a41f226b9ded80e6e1fd9652d6e4.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/6bde3984f0e0689ca0a2565c9678e41f7c43a41f226b9ded80e6e1fd9652d6e4.jpg)

![761e31ea49d7db0315693ba997e02a0aac571fb1784bbed3f7f16141388fec82.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/761e31ea49d7db0315693ba997e02a0aac571fb1784bbed3f7f16141388fec82.jpg)

![7c4e76af6fc5eb67fbb46fdc5a52181cbd0e00801e210bb4abda7928a0d10c9b.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/7c4e76af6fc5eb67fbb46fdc5a52181cbd0e00801e210bb4abda7928a0d10c9b.jpg)

![7f3688182ba77e8e2d7b487e69ce807bb3cb361bf8b606c21dd95074bd6112ba.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/7f3688182ba77e8e2d7b487e69ce807bb3cb361bf8b606c21dd95074bd6112ba.jpg)

![ab6ecfcb8cdab2263de2a5025afd2221c5ef0995b6b1612680a6ddefda2d2c04.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/ab6ecfcb8cdab2263de2a5025afd2221c5ef0995b6b1612680a6ddefda2d2c04.jpg)

![be22577af93afc88865bb8be74fc643999c88764de2e99c3598f71b9048d5cbc.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/images/be22577af93afc88865bb8be74fc643999c88764de2e99c3598f71b9048d5cbc.jpg)

### Tables

![09e763be9c43a95c9f323a5d4ae21647776b182ecc7269619595a9366d6e3ca5.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/09e763be9c43a95c9f323a5d4ae21647776b182ecc7269619595a9366d6e3ca5.jpg)

![172166010459ccba8fe72ce72270210d55789f301a3eb225379ed9172ac64ad6.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/172166010459ccba8fe72ce72270210d55789f301a3eb225379ed9172ac64ad6.jpg)

![4dcbaac833a9d5b3aeabeeb6410656e6df86eba60bc875415496637e0b05123f.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/4dcbaac833a9d5b3aeabeeb6410656e6df86eba60bc875415496637e0b05123f.jpg)

![52257a6eda04e8701229457606bc06a4c7423ebafe7df68be82a3656bf89a5a2.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/52257a6eda04e8701229457606bc06a4c7423ebafe7df68be82a3656bf89a5a2.jpg)

![6308d4573910f2e32445210a9d69d190645095624ba365e0984a63c11377cae2.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/6308d4573910f2e32445210a9d69d190645095624ba365e0984a63c11377cae2.jpg)

![7e5cf521ad2e98afa03b75232338879931a6c97553ddd1780e7a46f42085ae8f.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/7e5cf521ad2e98afa03b75232338879931a6c97553ddd1780e7a46f42085ae8f.jpg)

![9fdaeaffa58dba733f5c1b09222c33f5dbdada147ffafaa94e0ddf780164bcc1.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/9fdaeaffa58dba733f5c1b09222c33f5dbdada147ffafaa94e0ddf780164bcc1.jpg)

![afce1915a031a5cb6c7318e02e6bddded95eb6b40bf2e5eff048b4f29e71ba2e.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/afce1915a031a5cb6c7318e02e6bddded95eb6b40bf2e5eff048b4f29e71ba2e.jpg)

![b9a3652f5aef30518c55b6317b78c03e08416e905ddf8d23b8b603e08cc77241.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/b9a3652f5aef30518c55b6317b78c03e08416e905ddf8d23b8b603e08cc77241.jpg)

![be43bd7ac15ad7fbdfd559a5adedff71945a17943938960c4e5a5fd9a664e513.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/be43bd7ac15ad7fbdfd559a5adedff71945a17943938960c4e5a5fd9a664e513.jpg)

![c160c00c603040b2151ab691258ca389af8a06732a1aedbd3a0b218e61b9053f.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/c160c00c603040b2151ab691258ca389af8a06732a1aedbd3a0b218e61b9053f.jpg)

![dcd30eff25f3be60e0e03519ddfd003b5166ae34bc64cbbec599a19cc80911d8.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/dcd30eff25f3be60e0e03519ddfd003b5166ae34bc64cbbec599a19cc80911d8.jpg)

![e9d495e84d7a8d995617f428a790967e938daf16d6cfc014a47c8a2e54cc3eb9.jpg](../icml_results/4532_CALM%20Before%20the%20STORM_%20Unlocking%20Native%20Reasoning%20for%20Optimization%20Modeling/tables/e9d495e84d7a8d995617f428a790967e938daf16d6cfc014a47c8a2e54cc3eb9.jpg)

<a id="paper-4486"></a>

## RSPO: Regularized Self-Play Alignment of Large Language Models


### Images

![10553e1b10e84b224759d6148ca80f3f19b03abe1aeee83ec9a2f33bbd90f246.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/10553e1b10e84b224759d6148ca80f3f19b03abe1aeee83ec9a2f33bbd90f246.jpg)

![499e6c4ef5254d205309d2bf722583855c673b3600aaa71eabb2fa00c87db666.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/499e6c4ef5254d205309d2bf722583855c673b3600aaa71eabb2fa00c87db666.jpg)

![8a51abf5824619c776816c986e8cfa7de7d3f1180ecd79a9eb80cac96c0db745.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/8a51abf5824619c776816c986e8cfa7de7d3f1180ecd79a9eb80cac96c0db745.jpg)

![a2a1d24b75f8aa983ea4fc2d5c0982c5f0a01266071b03f042b5c5d2e9234690.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/a2a1d24b75f8aa983ea4fc2d5c0982c5f0a01266071b03f042b5c5d2e9234690.jpg)

![a4e8f6e124b2dc7817ec38b8ef58de92e43c59d5989bcb55563bc8081e01bba0.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/a4e8f6e124b2dc7817ec38b8ef58de92e43c59d5989bcb55563bc8081e01bba0.jpg)

![cb81aed7b935b878876eda8a2c81a4ccea6505d19a1375f20b62dd1dfe8874c2.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/cb81aed7b935b878876eda8a2c81a4ccea6505d19a1375f20b62dd1dfe8874c2.jpg)

![e48c1e48132d0286fd9cd586d204549d73a2a6e1539bf07b910bd7b15b1fb894.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/e48c1e48132d0286fd9cd586d204549d73a2a6e1539bf07b910bd7b15b1fb894.jpg)

![faecc6f3ae91ce5fb69c6808fc4c4f6731d3a627b89b12cad188ad739a5e8d5e.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/images/faecc6f3ae91ce5fb69c6808fc4c4f6731d3a627b89b12cad188ad739a5e8d5e.jpg)

### Tables

![05097a303b1f4cd254ebb35e75586d379a2d814a71f0bfc02dfd1b62be3bebed.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/05097a303b1f4cd254ebb35e75586d379a2d814a71f0bfc02dfd1b62be3bebed.jpg)

![1141410fe8138002ba7e145fd95fd29deed78208cf13dbc73f091721237ba7fb.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/1141410fe8138002ba7e145fd95fd29deed78208cf13dbc73f091721237ba7fb.jpg)

![127287f91ed28b953db268cfc2648c1a74134ae4fde31a76f80f6dd278e3b0d8.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/127287f91ed28b953db268cfc2648c1a74134ae4fde31a76f80f6dd278e3b0d8.jpg)

![186996411dd84c3c463c0bc8fa6de88b61d1f7013cd7418d971a54ebc45a7130.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/186996411dd84c3c463c0bc8fa6de88b61d1f7013cd7418d971a54ebc45a7130.jpg)

![300dcf872422c3447e77c04ce60d32ac33b2ec940848deac35d8486ee6966fe8.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/300dcf872422c3447e77c04ce60d32ac33b2ec940848deac35d8486ee6966fe8.jpg)

![31606bc99be688c7b6dc7edc9d66ef54b6e89e382e4fc66fd0763faa42508b6a.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/31606bc99be688c7b6dc7edc9d66ef54b6e89e382e4fc66fd0763faa42508b6a.jpg)

![3be8140f575030562ee1e90801dcecbf2a4c8ae832b7ccf117086978046ff5ea.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/3be8140f575030562ee1e90801dcecbf2a4c8ae832b7ccf117086978046ff5ea.jpg)

![43e2fa4c2f236a70dccef194ed996a4c4ea21f1ae2c593cde81e5c7616bf7b76.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/43e2fa4c2f236a70dccef194ed996a4c4ea21f1ae2c593cde81e5c7616bf7b76.jpg)

![4e09f20dacdac825638ea04f83c5f5f77caef67fb4ee7ed3b94db869f73a2d6f.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/4e09f20dacdac825638ea04f83c5f5f77caef67fb4ee7ed3b94db869f73a2d6f.jpg)

![7a3346d37b48c005b9d3ee7f036b0f16f2b22c6933c11738cc8f2d7ec5a226c0.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/7a3346d37b48c005b9d3ee7f036b0f16f2b22c6933c11738cc8f2d7ec5a226c0.jpg)

![bc8a2f35b000dd1af598e38a632cc92868b359ce6b504e4c28f376e15fdc5a7d.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/bc8a2f35b000dd1af598e38a632cc92868b359ce6b504e4c28f376e15fdc5a7d.jpg)

![c2406bc3664792e6a235ec80e8d172a721b909858a854aea3d1687850b596112.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/c2406bc3664792e6a235ec80e8d172a721b909858a854aea3d1687850b596112.jpg)

![ca115cdae5aae45ab547d4fc4c98285ae1cd4b192e9d8e5d087550e9a755f25f.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/ca115cdae5aae45ab547d4fc4c98285ae1cd4b192e9d8e5d087550e9a755f25f.jpg)

![fc6bc4019aa90e267ad56b5b0ce64023b7307f1b497b918d8953c014e991757d.jpg](../icml_results/4533_RSPO_%20Regularized%20Self-Play%20Alignment%20of%20Large%20Language%20Models/tables/fc6bc4019aa90e267ad56b5b0ce64023b7307f1b497b918d8953c014e991757d.jpg)

<a id="paper-4487"></a>

## Euclean: Automated Geometry Problem Formalization with Unified Verification in Lean


### Images

![cdc80a9881e2c6f54c406dba5084f9f411b134cf46b165a9c36fac31781699a6.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/images/cdc80a9881e2c6f54c406dba5084f9f411b134cf46b165a9c36fac31781699a6.jpg)

![f18caf0544ffe7c6b038d8c30f1c2db31824efa91815d664a7d4a47f31357930.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/images/f18caf0544ffe7c6b038d8c30f1c2db31824efa91815d664a7d4a47f31357930.jpg)

### Tables

![1a060a6272d9bc258a9d1a76d51971e8890ba092191e8f3fc7ce10d6617539fe.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/1a060a6272d9bc258a9d1a76d51971e8890ba092191e8f3fc7ce10d6617539fe.jpg)

![2d02d36509cf36a79b4f182027c4c7e21169a314732351271762af169f875eba.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/2d02d36509cf36a79b4f182027c4c7e21169a314732351271762af169f875eba.jpg)

![56f9fe316ea3e290d3f891c532b682e4a1c0b5ed70bbb34a69b6f14c9d48bf15.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/56f9fe316ea3e290d3f891c532b682e4a1c0b5ed70bbb34a69b6f14c9d48bf15.jpg)

![6a54e00176760ba9b46971bdd2e32ba1f0b138a5da2f5e0839e991b0dce79b1a.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/6a54e00176760ba9b46971bdd2e32ba1f0b138a5da2f5e0839e991b0dce79b1a.jpg)

![7e8f101181e5110928fef98812c42a85eb748e27f88a2f821e0fba3ff57ee736.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/7e8f101181e5110928fef98812c42a85eb748e27f88a2f821e0fba3ff57ee736.jpg)

![8624e1b30aec69fa936b4415663d426672b9cd17b918499d704003b27f20f8dd.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/8624e1b30aec69fa936b4415663d426672b9cd17b918499d704003b27f20f8dd.jpg)

![9642f439c111db928dfee8797a509d2d9ee1b46d7ba24ce7e85c518c6ab15adb.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/9642f439c111db928dfee8797a509d2d9ee1b46d7ba24ce7e85c518c6ab15adb.jpg)

![a0e9c807008478508736acedefa176a7fcd892767906ed96960ad7ce7f38faa4.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/a0e9c807008478508736acedefa176a7fcd892767906ed96960ad7ce7f38faa4.jpg)

![c69fd5c462a771eb1e5c4f0dfe0a747e6f5af0f2edb73691642cddfeac6e6f6f.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/c69fd5c462a771eb1e5c4f0dfe0a747e6f5af0f2edb73691642cddfeac6e6f6f.jpg)

![d0598ec1b4ca48c309970a1ca883b8ac197051fb36c238a6a9aeff27f72b5191.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/d0598ec1b4ca48c309970a1ca883b8ac197051fb36c238a6a9aeff27f72b5191.jpg)

![db03ffdad5dd8fe1ad0ee61f1d31fbecc65b70d169004a45fd8ef01d41b0af24.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/db03ffdad5dd8fe1ad0ee61f1d31fbecc65b70d169004a45fd8ef01d41b0af24.jpg)

![eb3a3c5ad7ba8afbe82b0d87546bcb6eb77f4c471b8f85b2b4eccf0bbaca6027.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/eb3a3c5ad7ba8afbe82b0d87546bcb6eb77f4c471b8f85b2b4eccf0bbaca6027.jpg)

![fcef1ddb3081a1d8cb44d26e38d7cfcc5bb162896561682bf3c71332b59ed830.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/fcef1ddb3081a1d8cb44d26e38d7cfcc5bb162896561682bf3c71332b59ed830.jpg)

![feaf24ea6cbf48b335ac328d10f4b79f809b6ad9037aef39ba99cdf409db4bba.jpg](../icml_results/4534_Euclean_%20Automated%20Geometry%20Problem%20Formalization%20with%20Unified%20Verification%20in%20Lean/tables/feaf24ea6cbf48b335ac328d10f4b79f809b6ad9037aef39ba99cdf409db4bba.jpg)

<a id="paper-4488"></a>

## Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models

### Images

![03cb404315a8f310f437f56e168478d6d7e2fda0957bc738ccc83381d29775d2.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/03cb404315a8f310f437f56e168478d6d7e2fda0957bc738ccc83381d29775d2.jpg)

![069e2171cb5ac186962a7c7fe3a36b8e39d34d3815c6cc2164a3ed4751cd970a.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/069e2171cb5ac186962a7c7fe3a36b8e39d34d3815c6cc2164a3ed4751cd970a.jpg)

![0c3d015490ba4a84e5b4540849c30169e283f7f9fb5aeb7959e3bbea20d52ef9.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/0c3d015490ba4a84e5b4540849c30169e283f7f9fb5aeb7959e3bbea20d52ef9.jpg)

![0e1544d24e89349b4ee180716688b4787c5c3d3fde509a5f42c0ce76f59fbbab.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/0e1544d24e89349b4ee180716688b4787c5c3d3fde509a5f42c0ce76f59fbbab.jpg)

![42d96342984c97520bf722142228f107fc433bf7926a57d796e6ad58337cbb02.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/42d96342984c97520bf722142228f107fc433bf7926a57d796e6ad58337cbb02.jpg)

![4dc61137563483f74fdbcac27dbbb79d0ff6dfc7fa5130c6d02d4e890a7b888f.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/4dc61137563483f74fdbcac27dbbb79d0ff6dfc7fa5130c6d02d4e890a7b888f.jpg)

![7cd0ecbd0c6330e4140f65771d68910997bc6972fcd3630bb91fb0a19a4a8a29.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/7cd0ecbd0c6330e4140f65771d68910997bc6972fcd3630bb91fb0a19a4a8a29.jpg)

![c17e9a8415de080b14ca49952bc0e9c4905d2eb123e7c148c333a949c0af2543.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/c17e9a8415de080b14ca49952bc0e9c4905d2eb123e7c148c333a949c0af2543.jpg)

![ccb649fdf83a25e5e64022f7578667ecaae116c6cb14e698666a444fb297d80c.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/ccb649fdf83a25e5e64022f7578667ecaae116c6cb14e698666a444fb297d80c.jpg)

![fd887371ac40a09cfe42d8bfb5fc9da6741414c765713b3a1f9205f1689715cd.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/images/fd887371ac40a09cfe42d8bfb5fc9da6741414c765713b3a1f9205f1689715cd.jpg)

### Tables

![8d11a913912b2dc4689aa7751d3138f1ed6a552268e91bebff26d9f9126f8c75.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/tables/8d11a913912b2dc4689aa7751d3138f1ed6a552268e91bebff26d9f9126f8c75.jpg)

![987cc73e1a4b9bf294204da2bce24521b3659262288333bb637105bca000b520.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/tables/987cc73e1a4b9bf294204da2bce24521b3659262288333bb637105bca000b520.jpg)

![a4192d32b8fa25431fae3354e636dd3e20a88e658afd23ba62102a86a94651ef.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/tables/a4192d32b8fa25431fae3354e636dd3e20a88e658afd23ba62102a86a94651ef.jpg)

![ad677dc9043bd050c3b06f20158b8238f1d10afd2217cd94917854b01e6c75a9.jpg](../icml_results/4535_Is%20Your%20Diffusion%20Sampler%20Actually%20Correct_%20A%20Sampler-Centric%20Evaluation%20of%20Discrete%20Diffusion%20Langu/tables/ad677dc9043bd050c3b06f20158b8238f1d10afd2217cd94917854b01e6c75a9.jpg)
