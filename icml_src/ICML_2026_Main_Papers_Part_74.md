# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 74 of 197

## 目录 (Table of Contents)

1. [CatFlow: Co-generation of Slab-Adsorbate Systems via Flow Matching](#paper-2410)
2. [Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR](#paper-2411)
3. [Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](#paper-2412)
4. [Fine-Tuning Masked Diffusion for Provable Self-Correction](#paper-2413)
5. [Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors](#paper-2414)
6. [Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression](#paper-2415)
7. [BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference](#paper-2416)
8. [EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments](#paper-2417)
9. [From Drift to Coherence: Stabilizing Beliefs in LLMs](#paper-2418)
10. [Frictional Q-Learning](#paper-2419)
11. [Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway](#paper-2420)
12. [Breaking the Reference Bottleneck via Learning to Rewrite Conversational Queries without Gold Reference Passages](#paper-2421)
13. [ST-Veto: Spatio-Temporal Token Veto for Diffusion MLLMs via Taylor Prediction and Visual Grounding](#paper-2422)
14. [WAVE: Window-Aware Vocabulary-Efficient Early-Exit for Training-Free LLM Acceleration](#paper-2423)
15. [Machine Learning Hamiltonians are Accurate Energy-Force Predictors](#paper-2424)
16. [Contrastive Representation Regularization for Vision-Language-Action Models](#paper-2425)
17. [Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](#paper-2426)
18. [Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](#paper-2427)
19. [Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality](#paper-2428)
20. [Adversarial Robustness of Implicit Neural Representation-Based Classifiers](#paper-2429)
21. [BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback](#paper-2430)
22. [Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor](#paper-2431)
23. [DiscoverLLM: From Executing Intents to Discovering Them](#paper-2432)
24. [Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](#paper-2433)
25. [MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks](#paper-2434)
26. [NNiT: Width-Agnostic Neural Network Generation with Structurally Aligned Weight Spaces](#paper-2435)
27. [Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch](#paper-2436)
28. [Investigating Component Contributions in Multi-Agent ML Systems](#paper-2437)
29. [Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness, and Safety](#paper-2438)
30. [TokenDrop: Token-Level Importance-Aware Backward Propagation Skipping for Efficient LLM Fine-Tuning](#paper-2439)
31. [Identifying Connectivity Distributions from Neural Dynamics Using Flows](#paper-2440)
32. [Stability Analysis of Sharpness-Aware Minimization](#paper-2441)
33. [GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache](#paper-2442)

---


<a id="paper-2410"></a>

## CatFlow: Co-generation of Slab-Adsorbate Systems via Flow Matching

### Images

![0d68e6c42db12b1bda658fde418886152af9ea0867729111e1d29100e962487b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/0d68e6c42db12b1bda658fde418886152af9ea0867729111e1d29100e962487b.jpg)

![3e9bdc5ecd241ff573e24f37a63d31fb1c1776b15cb338cee3c9e265b5ff23db.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/3e9bdc5ecd241ff573e24f37a63d31fb1c1776b15cb338cee3c9e265b5ff23db.jpg)

![4b954cd98b6b2d9638f9806e7aa50aec70f40459dc76ad18c202e51e6206eb4a.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/4b954cd98b6b2d9638f9806e7aa50aec70f40459dc76ad18c202e51e6206eb4a.jpg)

![606c5a67ca5b1cb19808c71e06d0f201de66ace594624a411875d1b8fbb6d364.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/606c5a67ca5b1cb19808c71e06d0f201de66ace594624a411875d1b8fbb6d364.jpg)

![8d183b6bdd4392ab3133ae858ee6b5a613a3df9004fbdf11abfbb38ffdaa5e93.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/8d183b6bdd4392ab3133ae858ee6b5a613a3df9004fbdf11abfbb38ffdaa5e93.jpg)

![a66048060adbb6d5b00bd77178567471f26947588cfbe3d662db2a4f261014f8.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/a66048060adbb6d5b00bd77178567471f26947588cfbe3d662db2a4f261014f8.jpg)

![c482a3db423d2cd547b14320f21a97e15227a8a26da7836d79bee45ad4ce5888.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/c482a3db423d2cd547b14320f21a97e15227a8a26da7836d79bee45ad4ce5888.jpg)

![c86ef8519b65e5c8f01631ed0d60b72c55049353aadc2c4704d3c9707bf48212.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/c86ef8519b65e5c8f01631ed0d60b72c55049353aadc2c4704d3c9707bf48212.jpg)

![dd24993b03862234af04c46ad8c4dc10b8369af2afb14863868f30b68d4fdab9.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/dd24993b03862234af04c46ad8c4dc10b8369af2afb14863868f30b68d4fdab9.jpg)

![de2345c7f2ecda8a35247a3a3ffe9cb5fb996057dfa718fa52422281e5a164e7.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/de2345c7f2ecda8a35247a3a3ffe9cb5fb996057dfa718fa52422281e5a164e7.jpg)

![f88fff8b9e0ed2013ca79b629bd45db8634f4d6c8a5e57682c2af0a8087f730d.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/f88fff8b9e0ed2013ca79b629bd45db8634f4d6c8a5e57682c2af0a8087f730d.jpg)

### Tables

![23302c630704da62d3072c667f08c6b9a522f7afdc9cbc0610fa5c627e562d6c.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/23302c630704da62d3072c667f08c6b9a522f7afdc9cbc0610fa5c627e562d6c.jpg)

![2c02a461e9ad4f839c750fe28d8c7cb60c846437cd4ef88b898c73f26795898b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/2c02a461e9ad4f839c750fe28d8c7cb60c846437cd4ef88b898c73f26795898b.jpg)

![2f5c563d1a325fd24af04e1ccd8678e29bc7cffe1c2710358c65f1fe87419870.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/2f5c563d1a325fd24af04e1ccd8678e29bc7cffe1c2710358c65f1fe87419870.jpg)

![308e4eb74fd168b1565d98f3328a2c6236785fe3d91e023b90e6d7c180a8deb0.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/308e4eb74fd168b1565d98f3328a2c6236785fe3d91e023b90e6d7c180a8deb0.jpg)

![436c3f80a9c794f21fdf862008b4571a1872e17e965b8233d43a0c31a599a542.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/436c3f80a9c794f21fdf862008b4571a1872e17e965b8233d43a0c31a599a542.jpg)

![530e6567d02ea872dce18537d34d70c523e7148da96fb6962274195d9229fb68.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/530e6567d02ea872dce18537d34d70c523e7148da96fb6962274195d9229fb68.jpg)

![55b4423b7d42d9c4d4f57794e4a8e4db803483ba6f6d31ac5a7cb237f0bcf4c8.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/55b4423b7d42d9c4d4f57794e4a8e4db803483ba6f6d31ac5a7cb237f0bcf4c8.jpg)

![5c460a090a90083fdfbc3f5751ba95fd9aca9cf1a14e670b00ce0e658f7acb91.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/5c460a090a90083fdfbc3f5751ba95fd9aca9cf1a14e670b00ce0e658f7acb91.jpg)

![5eb0f1948c445abed6ec083366f8f3bc3d8fb387400445a2906edaedae5b1ae4.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/5eb0f1948c445abed6ec083366f8f3bc3d8fb387400445a2906edaedae5b1ae4.jpg)

![63d989a51d66fcd6534e07c110f6fcec3ea783665af65fdf1c6ff1fac91699ba.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/63d989a51d66fcd6534e07c110f6fcec3ea783665af65fdf1c6ff1fac91699ba.jpg)

![727e8a98e33b28603f560058b8d8a575d8647e52f2223963e52b5305c630f4ee.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/727e8a98e33b28603f560058b8d8a575d8647e52f2223963e52b5305c630f4ee.jpg)

![789fd92690effc93af291d0d08697c60e805b9b3c05d7051bb32248ed75a67ce.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/789fd92690effc93af291d0d08697c60e805b9b3c05d7051bb32248ed75a67ce.jpg)

![8717170930f337075e8ce023a926776ac8a1ce91cc19c2c36ccd981ed41270cb.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8717170930f337075e8ce023a926776ac8a1ce91cc19c2c36ccd981ed41270cb.jpg)

![8a815f2365080bb81e43267b8a01f6d517bf2e7a196dec32428674a8d839f3c4.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8a815f2365080bb81e43267b8a01f6d517bf2e7a196dec32428674a8d839f3c4.jpg)

![8b9216d4fc9f748888c6c1f7746451c20777c9f31105031d403ad1d99b3e7f76.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8b9216d4fc9f748888c6c1f7746451c20777c9f31105031d403ad1d99b3e7f76.jpg)

![9344a5629519c26449340378dfb1ae785f9a104aa316d32daea502d4be7202cc.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/9344a5629519c26449340378dfb1ae785f9a104aa316d32daea502d4be7202cc.jpg)

![98821807693e516fb200743cbfb94fb8794455de4aa6823cb1471c19ded69526.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/98821807693e516fb200743cbfb94fb8794455de4aa6823cb1471c19ded69526.jpg)

![a0d106f1fed2a77d79ee37148d2c44e713c3d33ac3936b3bbd478a3ac0d5b528.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/a0d106f1fed2a77d79ee37148d2c44e713c3d33ac3936b3bbd478a3ac0d5b528.jpg)

![aaf94a8b1242fc8c1e157e7ddc853020ddf212b25faca208a1569c1c424db914.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/aaf94a8b1242fc8c1e157e7ddc853020ddf212b25faca208a1569c1c424db914.jpg)

![c6a1a8552495007ba51e0402070b3c237b5ed7a7033317412c9b647e1886d12b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/c6a1a8552495007ba51e0402070b3c237b5ed7a7033317412c9b647e1886d12b.jpg)

![c8186a01fb8e1735174b00f08477fe2cfdaef7996ada4c5e89eaccd7bb36239e.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/c8186a01fb8e1735174b00f08477fe2cfdaef7996ada4c5e89eaccd7bb36239e.jpg)

![e27d501f9cfe08e5039cf2213a31e26cec9fd42faec2e5338c39d2617957500f.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/e27d501f9cfe08e5039cf2213a31e26cec9fd42faec2e5338c39d2617957500f.jpg)

![e30449da837b2506b69515a35a53283b8cb0a2c5443b0288b385ed5f33a86238.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/e30449da837b2506b69515a35a53283b8cb0a2c5443b0288b385ed5f33a86238.jpg)

![eae153fd796ef163ad5e7c5d2f5f56d965c09984006d9924e1ab324a02c51b75.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/eae153fd796ef163ad5e7c5d2f5f56d965c09984006d9924e1ab324a02c51b75.jpg)

![ece70be24eb9d16dac694f776aa4f768ef6e43936843b09fd1e965b65641fc5e.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/ece70be24eb9d16dac694f776aa4f768ef6e43936843b09fd1e965b65641fc5e.jpg)

![f368c65ad8a8324eec20d954fbf1a8ea09a51f287f6dd60455c01d561a04b73a.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/f368c65ad8a8324eec20d954fbf1a8ea09a51f287f6dd60455c01d561a04b73a.jpg)

## CatFlow: Co-generation of Slab-Adsorbate Systems via Flow Matching


### Images

![0d706f97bb15328dce29ff210bd2982f968e8290b7df590160f8035fe8f4a838.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/0d706f97bb15328dce29ff210bd2982f968e8290b7df590160f8035fe8f4a838.jpg)

![168ed932d7ed3890b5ec7af3bd5a948e44e248fad6e86295d2028f8ab82e3306.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/168ed932d7ed3890b5ec7af3bd5a948e44e248fad6e86295d2028f8ab82e3306.jpg)

![2bfe3af4d3465efa08177bb3350dadb859eccecd7e4d9df1cca537c7151cf7c9.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/2bfe3af4d3465efa08177bb3350dadb859eccecd7e4d9df1cca537c7151cf7c9.jpg)

![2d81b6ce030562af924a57347d8c29c41f488ac0f70dcba3fa3f2ff10e359b28.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/2d81b6ce030562af924a57347d8c29c41f488ac0f70dcba3fa3f2ff10e359b28.jpg)

![31b2ef07515ab5e412dbad3708886c4d739de5e66749c47f6711e5b8ea0916de.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/31b2ef07515ab5e412dbad3708886c4d739de5e66749c47f6711e5b8ea0916de.jpg)

![5e22af142dd939035fe3c71844599940baa9d98267ef5f6b74e40a3456c32145.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/5e22af142dd939035fe3c71844599940baa9d98267ef5f6b74e40a3456c32145.jpg)

![a7d88f79228dcab4fc21cd2517008be106f1043fb5eabe8c8cc45419273d4dd8.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/a7d88f79228dcab4fc21cd2517008be106f1043fb5eabe8c8cc45419273d4dd8.jpg)

![cf1c3604ec8924ff6833a09d62d6ef296b6a681b0bdb4595ae52919bf07879a2.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/images/cf1c3604ec8924ff6833a09d62d6ef296b6a681b0bdb4595ae52919bf07879a2.jpg)

### Tables

![068934c0ebe0d775ce6e5c11fdc010bbb5b18697cd9833e1238c413c5660852e.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/068934c0ebe0d775ce6e5c11fdc010bbb5b18697cd9833e1238c413c5660852e.jpg)

![2d8e07aca2f814856f796dfc2b5fc5ac935f54d5f1c88572e76f0a5b9b332360.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/2d8e07aca2f814856f796dfc2b5fc5ac935f54d5f1c88572e76f0a5b9b332360.jpg)

![2f45eaa147075d274fa99663a0ce5a819126441fd030a21fd429662f36bfd67d.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/2f45eaa147075d274fa99663a0ce5a819126441fd030a21fd429662f36bfd67d.jpg)

![330c244b48721c8c2b333bda293d4e2ae76507ddce80bf83f42232b72fff6447.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/330c244b48721c8c2b333bda293d4e2ae76507ddce80bf83f42232b72fff6447.jpg)

![4bceb711bc3f16c71c720c88ed9f7ca33a6e158f5e374ab09610d4a96cae292f.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/4bceb711bc3f16c71c720c88ed9f7ca33a6e158f5e374ab09610d4a96cae292f.jpg)

![5e083f817cd5f747ab4df2bebbc78e2f84d98e5effaf5e32a415e45553d5ce69.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/5e083f817cd5f747ab4df2bebbc78e2f84d98e5effaf5e32a415e45553d5ce69.jpg)

![7799efa239180f1d1d196be2eed2b11602aa7224fba96dccd6dfc4367fb910f3.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/7799efa239180f1d1d196be2eed2b11602aa7224fba96dccd6dfc4367fb910f3.jpg)

![96d02aeed68ef06568a912db4c5b760de6ab792d93758cb0115e11e70766c31f.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/96d02aeed68ef06568a912db4c5b760de6ab792d93758cb0115e11e70766c31f.jpg)

![b8a15082e317241fbac1699d108aec6fbe277bda3618c2e45e2beaabaa79e374.jpg](../icml_results/2440_CatFlow_%20Co-generation%20of%20Slab-Adsorbate%20Systems%20via%20Flow%20Matching/tables/b8a15082e317241fbac1699d108aec6fbe277bda3618c2e45e2beaabaa79e374.jpg)

<a id="paper-2411"></a>

## Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR


### Images

![0167e536ec63fb62036012fbcfdcb18c22d32121e4a6cd3073f4757710517d85.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/0167e536ec63fb62036012fbcfdcb18c22d32121e4a6cd3073f4757710517d85.jpg)

![24f78e5308b51a3a6b57251a1561ac6aef8d4057b0a4c487001be87e326e2b9f.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/24f78e5308b51a3a6b57251a1561ac6aef8d4057b0a4c487001be87e326e2b9f.jpg)

![2c0e4c109acfdb754ec1fe80796460f5f8a05e3cc6821164f3b03396cc77b34b.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/2c0e4c109acfdb754ec1fe80796460f5f8a05e3cc6821164f3b03396cc77b34b.jpg)

![32a6b918bd4558296eeef2f28af06b97c766d4d1bb9fdd9201d2808374f3649a.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/32a6b918bd4558296eeef2f28af06b97c766d4d1bb9fdd9201d2808374f3649a.jpg)

![45c0675e1971acd0576bc4e7df1c2ebe9d392ab869eeb47d55cc62cae27dadfa.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/45c0675e1971acd0576bc4e7df1c2ebe9d392ab869eeb47d55cc62cae27dadfa.jpg)

![586e048562c9b9ce526db5b057a93d6ab59394082babe5053efb0e75e36e2aed.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/586e048562c9b9ce526db5b057a93d6ab59394082babe5053efb0e75e36e2aed.jpg)

![aaf5e2c38b23ff8fce388cdc245c206cebcc82156f8b0efd0fa62bf10e473db1.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/aaf5e2c38b23ff8fce388cdc245c206cebcc82156f8b0efd0fa62bf10e473db1.jpg)

![c220fe2e07641e530f470cd6d3cbb31d1ae9df0807493498f52690156394b1aa.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/c220fe2e07641e530f470cd6d3cbb31d1ae9df0807493498f52690156394b1aa.jpg)

![d9a7160a9b0cfbef119b79c106cb01d1d3c6211c948c6cff0f6da6ef6b83df10.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/images/d9a7160a9b0cfbef119b79c106cb01d1d3c6211c948c6cff0f6da6ef6b83df10.jpg)

### Tables

![01ce56b919313980eaa3df13f0249f2ebcf9084d162100902e6cd708fdeb5207.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/01ce56b919313980eaa3df13f0249f2ebcf9084d162100902e6cd708fdeb5207.jpg)

![0a135a2a55508bca6ffdc9de9eb72723713689d4faee22d98ee8d4fa29b19ebe.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/0a135a2a55508bca6ffdc9de9eb72723713689d4faee22d98ee8d4fa29b19ebe.jpg)

![10e60535f6092ab326e9f09a2480b9267134a83880a9e5b908a1145d0c421ac3.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/10e60535f6092ab326e9f09a2480b9267134a83880a9e5b908a1145d0c421ac3.jpg)

![7e8d9eeae76e44c67568a194ce99b82785be878b3e4be760bc849f7eac170727.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/7e8d9eeae76e44c67568a194ce99b82785be878b3e4be760bc849f7eac170727.jpg)

![9a860f9df8355ca6bc89d81e69d02667bbc1c5880d08abb0e02c8a4ed9ef834e.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/9a860f9df8355ca6bc89d81e69d02667bbc1c5880d08abb0e02c8a4ed9ef834e.jpg)

![9c602566c44f722ac04c3f6d99f09b7899bc0d4d0723d2f5e9c2e86cea983123.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/9c602566c44f722ac04c3f6d99f09b7899bc0d4d0723d2f5e9c2e86cea983123.jpg)

![abcd9fa147c6967b46d8ef4bb754eca69a391b32463f426967be60328f9cf099.jpg](../icml_results/2441_Beyond%20Normalization_%20Rethinking%20the%20Partition%20Function%20as%20a%20Difficulty%20Scheduler%20for%20RLVR/tables/abcd9fa147c6967b46d8ef4bb754eca69a391b32463f426967be60328f9cf099.jpg)

<a id="paper-2412"></a>

## Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data


### Images

![1c37257a619ec1c1456adad74bdd358386c56857c6d3430c09a57b94540fa03e.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/1c37257a619ec1c1456adad74bdd358386c56857c6d3430c09a57b94540fa03e.jpg)

![226f8bd3ca34e4206200dae280a4e0c415df5db7ea6e0b098be2630b304fdbf5.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/226f8bd3ca34e4206200dae280a4e0c415df5db7ea6e0b098be2630b304fdbf5.jpg)

![3c9756bcac4dc8ff665481f68af2829837a8763791a5a764ca7d870d6c5786a5.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/3c9756bcac4dc8ff665481f68af2829837a8763791a5a764ca7d870d6c5786a5.jpg)

![46c6264645c1bd1e54d27f0d3c1e720d655900a64abb7e680fc1480592f30822.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/46c6264645c1bd1e54d27f0d3c1e720d655900a64abb7e680fc1480592f30822.jpg)

![474da9a96d131b06afb1b9c5fab1f263ac73f46a3eb9564b979b9cea8e0be8eb.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/474da9a96d131b06afb1b9c5fab1f263ac73f46a3eb9564b979b9cea8e0be8eb.jpg)

![4b99d5d3328bf2fd6bb442a81ec6225bc03042a86f60e4a75481bc204ea71c85.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/4b99d5d3328bf2fd6bb442a81ec6225bc03042a86f60e4a75481bc204ea71c85.jpg)

![57b17d048f4b1ba42fab77305688977addbe8deecafdd356b581c8b706afd17d.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/57b17d048f4b1ba42fab77305688977addbe8deecafdd356b581c8b706afd17d.jpg)

![60f38be8014612e43c4e5861d56c4ed00fe2f70b8abd390d973d9c9fe99d5aab.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/60f38be8014612e43c4e5861d56c4ed00fe2f70b8abd390d973d9c9fe99d5aab.jpg)

![66adc67e532c497bdf7f832cb52fde0794fd8f8c7a036ebe796f5ffa3e5d7ce9.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/66adc67e532c497bdf7f832cb52fde0794fd8f8c7a036ebe796f5ffa3e5d7ce9.jpg)

![7c9d53d55535cfd6d914e1213907193421ecfb6592d3ee69291ff0fb6e3213e4.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/7c9d53d55535cfd6d914e1213907193421ecfb6592d3ee69291ff0fb6e3213e4.jpg)

![bf0da74f581970d8e12b62a843babb08e2af3499e94cf17efde7336701e991cf.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/bf0da74f581970d8e12b62a843babb08e2af3499e94cf17efde7336701e991cf.jpg)

![df457ed45e6df18cd02de6bd420466ceba06f19a4d9f3c0b907c1aed6b647be0.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/df457ed45e6df18cd02de6bd420466ceba06f19a4d9f3c0b907c1aed6b647be0.jpg)

![eb60008e08b16a206f1d2153b7b223acfdcb9f36ecb18f7968729f3e23afc7e2.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/eb60008e08b16a206f1d2153b7b223acfdcb9f36ecb18f7968729f3e23afc7e2.jpg)

![f10c29cff4e8db4a07db1e08a1ee5c1c356b9b423ad1bca79b5009674e3c682f.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/f10c29cff4e8db4a07db1e08a1ee5c1c356b9b423ad1bca79b5009674e3c682f.jpg)

![f884a67ad93772a85a67c935e3f38b0bee22ad8099b5d5aecdde1d8249c51bd5.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/f884a67ad93772a85a67c935e3f38b0bee22ad8099b5d5aecdde1d8249c51bd5.jpg)

![fd795c2f52ef8666910d8bb5953a454db6c4d8bf6e37fee149b8459df78d13b7.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/images/fd795c2f52ef8666910d8bb5953a454db6c4d8bf6e37fee149b8459df78d13b7.jpg)

### Tables

![13e517b80b03188623929d10daaaaca5915e8170e5597370ea7b818312d348d6.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/13e517b80b03188623929d10daaaaca5915e8170e5597370ea7b818312d348d6.jpg)

![1af224b7513c1ccf51940ae3add601ad7bbe0e26e15777314de3c586304b8798.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/1af224b7513c1ccf51940ae3add601ad7bbe0e26e15777314de3c586304b8798.jpg)

![2f4b8e4d7c0acefdeea3d95b9169279df6b4cd27d83aeff0e26461321278ae03.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/2f4b8e4d7c0acefdeea3d95b9169279df6b4cd27d83aeff0e26461321278ae03.jpg)

![32ce24e6d8941b236c5c01e429b85dce5bf00ff8d3fc37c2bdd6408a5f970cca.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/32ce24e6d8941b236c5c01e429b85dce5bf00ff8d3fc37c2bdd6408a5f970cca.jpg)

![666248f4f4ccca05d6a36a57a4ad510fc3f519e369f7f0d5d23582f2e7ce39c8.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/666248f4f4ccca05d6a36a57a4ad510fc3f519e369f7f0d5d23582f2e7ce39c8.jpg)

![8c91be2a598ac686dfa7d454c5a5e57271cc9ddcd707ec8f07147c30eb0a871b.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/8c91be2a598ac686dfa7d454c5a5e57271cc9ddcd707ec8f07147c30eb0a871b.jpg)

![918935cd8893fe7dcce2c83bac863160527d1734ae3aadd136f0e9958c7f2ae5.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/918935cd8893fe7dcce2c83bac863160527d1734ae3aadd136f0e9958c7f2ae5.jpg)

![af701ac3fd4b7daa6f76d1887a09ea51e93296b66f4890f01e950bc14e5dd268.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/af701ac3fd4b7daa6f76d1887a09ea51e93296b66f4890f01e950bc14e5dd268.jpg)

![c40b6ea34c86c362130e7c1283831ec318d5b46397181cc9395deae1900387a5.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/c40b6ea34c86c362130e7c1283831ec318d5b46397181cc9395deae1900387a5.jpg)

![d0c2b0dd8742cdd0256b6ce6f2f8c34ed2d22ea8e0b46ad7d5b7c3b69f521b7d.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/d0c2b0dd8742cdd0256b6ce6f2f8c34ed2d22ea8e0b46ad7d5b7c3b69f521b7d.jpg)

![e79d148799b88af6669511c42e004de038447b5e98eba5aaf939dd1bfcbffd76.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/e79d148799b88af6669511c42e004de038447b5e98eba5aaf939dd1bfcbffd76.jpg)

![ec16d9a378f22023e41157513a08ad721655d6d2a3deaa6d05f5473e2da6fe5b.jpg](../icml_results/2442_Inconsistency-Aware%20Minimization_%20Improving%20Generalization%20with%20Unlabeled%20Data/tables/ec16d9a378f22023e41157513a08ad721655d6d2a3deaa6d05f5473e2da6fe5b.jpg)

<a id="paper-2413"></a>

## Fine-Tuning Masked Diffusion for Provable Self-Correction


### Images

![0b1ddf2aa86cb2319d3f2319eb082e34a14a6495cd2d9999d84e50d9e40ad1dd.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/0b1ddf2aa86cb2319d3f2319eb082e34a14a6495cd2d9999d84e50d9e40ad1dd.jpg)

![0ffbb02983fc889c763ec27050f32bb783f816e29980bed211174b474e635ff7.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/0ffbb02983fc889c763ec27050f32bb783f816e29980bed211174b474e635ff7.jpg)

![2220c2acd3b3790bbff212f72160adf9088ed111e2394eec08388d8e0334a924.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/2220c2acd3b3790bbff212f72160adf9088ed111e2394eec08388d8e0334a924.jpg)

![29eed58220a3dcca44e3a69006cbd48f80d0704e704a90977543f6c370a5189d.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/29eed58220a3dcca44e3a69006cbd48f80d0704e704a90977543f6c370a5189d.jpg)

![7425f9a431ea981b2a109d30da7eb5afbb58cf2c0e7a752caeb23933a622700d.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/7425f9a431ea981b2a109d30da7eb5afbb58cf2c0e7a752caeb23933a622700d.jpg)

![807c5d3630b816e426ee101b8def8d2b625f5fcff483da2c1e197d3d320410f2.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/807c5d3630b816e426ee101b8def8d2b625f5fcff483da2c1e197d3d320410f2.jpg)

![9205e705897c4ab3dc8f489921da1cfff01820216dd0e6ead9748b3ab0ceccb9.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/9205e705897c4ab3dc8f489921da1cfff01820216dd0e6ead9748b3ab0ceccb9.jpg)

![bcb5968b0828b09a84553db229c52552d1de439786321fbc4ee8d5dcb82b3873.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/images/bcb5968b0828b09a84553db229c52552d1de439786321fbc4ee8d5dcb82b3873.jpg)

### Tables

![00f10116ac08d94d38e090b803c6e202bef97a83c4aa287cd3ef5194b75533f8.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/tables/00f10116ac08d94d38e090b803c6e202bef97a83c4aa287cd3ef5194b75533f8.jpg)

![0c4eb29552beaa8f423075dd6c70091e3ed67d560da9c699b6dc969159e81bc9.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/tables/0c4eb29552beaa8f423075dd6c70091e3ed67d560da9c699b6dc969159e81bc9.jpg)

![2c5ee9fcac5b6dfb09b05fdd0952bf1ec0211c2bdf115e36f891a780918a9999.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/tables/2c5ee9fcac5b6dfb09b05fdd0952bf1ec0211c2bdf115e36f891a780918a9999.jpg)

![b6001ef899a54044707c586650b1f58790635547e003e0ce4cc161b6d22dc729.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/tables/b6001ef899a54044707c586650b1f58790635547e003e0ce4cc161b6d22dc729.jpg)

![f7016f561100181c722ba4f8ce2bb0a536f382e1218ed516a640ab25accd5b5a.jpg](../icml_results/2443_Fine-Tuning%20Masked%20Diffusion%20for%20Provable%20Self-Correction/tables/f7016f561100181c722ba4f8ce2bb0a536f382e1218ed516a640ab25accd5b5a.jpg)

<a id="paper-2414"></a>

## Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors


### Images

![4800f36bd23cbd458c277170456d175c2319919334373d20aac4a065ba09613a.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/4800f36bd23cbd458c277170456d175c2319919334373d20aac4a065ba09613a.jpg)

![4fec7f54149ffd42c8126e5f31f03aee161d8b3d37cf8d650c07c249b794182a.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/4fec7f54149ffd42c8126e5f31f03aee161d8b3d37cf8d650c07c249b794182a.jpg)

![59c56a3672b27987b8e50df0a2cb3f89d377195c0f955b9777de0994477c74c7.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/59c56a3672b27987b8e50df0a2cb3f89d377195c0f955b9777de0994477c74c7.jpg)

![997d671a3255f1358b6f9dd3887d5549d26618febfc52922903f35502f7145cf.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/997d671a3255f1358b6f9dd3887d5549d26618febfc52922903f35502f7145cf.jpg)

![afedd1b9653249c71cfb050ac0a3ce61adb19e7ef242777df3cfcd7dd27059c7.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/afedd1b9653249c71cfb050ac0a3ce61adb19e7ef242777df3cfcd7dd27059c7.jpg)

![b8cf61a2d5fa6bdc86585c4984645cee6facbf39a144f7707b1a6f1e5e2510e2.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/b8cf61a2d5fa6bdc86585c4984645cee6facbf39a144f7707b1a6f1e5e2510e2.jpg)

![c0d296337802ba8655f7049abb902bb2ed1517fe25b302d0a975dd6c0b193ef4.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/c0d296337802ba8655f7049abb902bb2ed1517fe25b302d0a975dd6c0b193ef4.jpg)

![c19e6a19f15623614e8b062dbdf9a4a59f1d3130c39f772de75ce530c4b5073b.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/c19e6a19f15623614e8b062dbdf9a4a59f1d3130c39f772de75ce530c4b5073b.jpg)

![e6332bbf0d256ca18510cb13939ef8daad82369c2866291c5f08c4c79da2e837.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/e6332bbf0d256ca18510cb13939ef8daad82369c2866291c5f08c4c79da2e837.jpg)

![eed3da8596d3a1f293b195b7c52f701201f4acd2c4821eabae9ed70554f085ab.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/eed3da8596d3a1f293b195b7c52f701201f4acd2c4821eabae9ed70554f085ab.jpg)

![fa829112dbc785c5bdd7f80133f2302d70db1e6a3c408e836588d92c402bb23f.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/images/fa829112dbc785c5bdd7f80133f2302d70db1e6a3c408e836588d92c402bb23f.jpg)

### Tables

![0223e56fb0adfef3973f875f3a18f868238114678c42bc386527489b129299d7.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/0223e56fb0adfef3973f875f3a18f868238114678c42bc386527489b129299d7.jpg)

![0f5d5894385ee59e1f947c85832616fe37e2a9e977e4e70cc7faa4d6083d986e.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/0f5d5894385ee59e1f947c85832616fe37e2a9e977e4e70cc7faa4d6083d986e.jpg)

![124314120ceea9ed1ec7f9c06d2567ee8e5521da60b1030eb3564ca917b975e3.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/124314120ceea9ed1ec7f9c06d2567ee8e5521da60b1030eb3564ca917b975e3.jpg)

![5ba70c0f1dcf2787f65dfc81082f10b9bf7cfcdb8c5863083a57cc880fc825bb.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/5ba70c0f1dcf2787f65dfc81082f10b9bf7cfcdb8c5863083a57cc880fc825bb.jpg)

![ac7387626c458b0868bec91bf40bda75de49add52c8c389b65f6d83a14d07598.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/ac7387626c458b0868bec91bf40bda75de49add52c8c389b65f6d83a14d07598.jpg)

![c352983389316dc2dccb8cc964fee2b43f70ce36fb9d86b63a90933f5de84e3d.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/c352983389316dc2dccb8cc964fee2b43f70ce36fb9d86b63a90933f5de84e3d.jpg)

![ddd7e5a383a1a533c29860ba312172a1e770059327ff9384584ff4c5eac1bab3.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/ddd7e5a383a1a533c29860ba312172a1e770059327ff9384584ff4c5eac1bab3.jpg)

![e8ff09f3ce447cb31613dbb74fa6cc6184b9f957286f0e8d401874adb3c07c3b.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/e8ff09f3ce447cb31613dbb74fa6cc6184b9f957286f0e8d401874adb3c07c3b.jpg)

![ef434e2c176255c331163ff1e417096f261bd0a7a0927b27b21239bdd12910b4.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/ef434e2c176255c331163ff1e417096f261bd0a7a0927b27b21239bdd12910b4.jpg)

![f91435b58805c01d064b80c5357f8908094e206272f5034aa40b6b488447ac4f.jpg](../icml_results/2444_Synthesizable%20Molecular%20Generation%20via%20Soft-constrained%20GFlowNets%20with%20Rich%20Chemical%20Priors/tables/f91435b58805c01d064b80c5357f8908094e206272f5034aa40b6b488447ac4f.jpg)

<a id="paper-2415"></a>

## Semi-Supervised Learning with Noisy Proxy Covariates: Generalization Bounds and Distribution Regression


### Images

![a1a736999a2ef4cd64a145d58ad9f9421536686273450c98a686244070329e9c.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/images/a1a736999a2ef4cd64a145d58ad9f9421536686273450c98a686244070329e9c.jpg)

### Tables

![4371c62946d9432935c306b685e5661042ddfab91969084b4ef7f3f95149c061.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/tables/4371c62946d9432935c306b685e5661042ddfab91969084b4ef7f3f95149c061.jpg)

![6546e3eaf453bdd8c378f7c931bce14625ea153ae49cdb91366071e07dde8aad.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/tables/6546e3eaf453bdd8c378f7c931bce14625ea153ae49cdb91366071e07dde8aad.jpg)

![6c2258a375d685a11b394d7b3048dfd13a20e7a6b756efb6abc4666909328f6d.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/tables/6c2258a375d685a11b394d7b3048dfd13a20e7a6b756efb6abc4666909328f6d.jpg)

![895b589772985de7c1e1d5b278d9ac5ef454a59b0562c6bc4eff18ac67c86426.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/tables/895b589772985de7c1e1d5b278d9ac5ef454a59b0562c6bc4eff18ac67c86426.jpg)

![d71520ecd9ace8eebc3da8841cfadbb07249a35549ed200b528fe1519d593dba.jpg](../icml_results/2445_Semi-Supervised%20Learning%20with%20Noisy%20Proxy%20Covariates_%20Generalization%20Bounds%20and%20Distribution%20Regress/tables/d71520ecd9ace8eebc3da8841cfadbb07249a35549ed200b528fe1519d593dba.jpg)

<a id="paper-2416"></a>

## BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference


### Images

![0cf724f91f452e2a914150dcc9ba531430dc9a90c726519a79f91000fa0803c8.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/0cf724f91f452e2a914150dcc9ba531430dc9a90c726519a79f91000fa0803c8.jpg)

![2603c51db35b1c46594b17faa6e16ab732024f54f12b5e4f657c6baad0d28292.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/2603c51db35b1c46594b17faa6e16ab732024f54f12b5e4f657c6baad0d28292.jpg)

![4aa385e232ab0e123ee1c121de6c15cc7d9b82377fd88e7e99497154fd141692.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/4aa385e232ab0e123ee1c121de6c15cc7d9b82377fd88e7e99497154fd141692.jpg)

![54a688ec6e43e33da3ee10eaf441478afecf942ceaa68a318500b9f3791c514a.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/54a688ec6e43e33da3ee10eaf441478afecf942ceaa68a318500b9f3791c514a.jpg)

![60b5ef4e135fd4d154879fcfe88dfd639f24b1705258e0911b08584e04dcd8fa.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/60b5ef4e135fd4d154879fcfe88dfd639f24b1705258e0911b08584e04dcd8fa.jpg)

![6b2c15d71e01704f214b7382a494efb523624e36d72bec1937224faa5d1fed4e.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/6b2c15d71e01704f214b7382a494efb523624e36d72bec1937224faa5d1fed4e.jpg)

![a5ab40eb16d8c7c525feaf3a70542ef9e024b698e9d365453ba41b59ed19b9fd.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/a5ab40eb16d8c7c525feaf3a70542ef9e024b698e9d365453ba41b59ed19b9fd.jpg)

![b993aa3e7107969f57f5df0b4af10877e4137d667ebbcdfcaa2dfbd45da82bfc.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/b993aa3e7107969f57f5df0b4af10877e4137d667ebbcdfcaa2dfbd45da82bfc.jpg)

![d5fa665cfcb53aad99a93bc6fe7381f2922f0d898a842a80751a268e86c385a7.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/images/d5fa665cfcb53aad99a93bc6fe7381f2922f0d898a842a80751a268e86c385a7.jpg)

### Tables

![2281dcf2a4580c8f4b85dfcd7e58ac1768dd377623001368fae5727fc0ab9f7e.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/2281dcf2a4580c8f4b85dfcd7e58ac1768dd377623001368fae5727fc0ab9f7e.jpg)

![2b604f85e9d444200b020c7747550efd680f4729d0e201e26e89e4201b7bedc1.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/2b604f85e9d444200b020c7747550efd680f4729d0e201e26e89e4201b7bedc1.jpg)

![5af656d3e1c76b84a20383155472cfeb9a75d07766d2667d527d5915dcdfc30d.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/5af656d3e1c76b84a20383155472cfeb9a75d07766d2667d527d5915dcdfc30d.jpg)

![a87310cce2a8b8191fcd015d3a302008d09f015bcfc4dbd826104114b29d5b3d.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/a87310cce2a8b8191fcd015d3a302008d09f015bcfc4dbd826104114b29d5b3d.jpg)

![ade4acd4d69b833bc8c05a750025e9b28ab2e7c9f3bc0cb1e2a6470271e6ccba.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/ade4acd4d69b833bc8c05a750025e9b28ab2e7c9f3bc0cb1e2a6470271e6ccba.jpg)

![e3703d7052de0b41be4a690e0c4845e153ff3aa96e295cdb39c7789f8a410c1d.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/e3703d7052de0b41be4a690e0c4845e153ff3aa96e295cdb39c7789f8a410c1d.jpg)

![fb7987a51b101f845c6ddb7829b71365f322eb68d52e7ee5e0a9abbd39f4a678.jpg](../icml_results/2446_BeaconKV_%20Key-Value%20Cache%20Compression%20Guided%20by%20Beacon%20Queries%20for%20Efficient%20Large%20Reasoning%20Model%20I/tables/fb7987a51b101f845c6ddb7829b71365f322eb68d52e7ee5e0a9abbd39f4a678.jpg)

<a id="paper-2417"></a>

## EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments


### Images

![064df8f80e0389a215c2980bb175cbd92eb38352dd1053004bdf627dc9bed033.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/064df8f80e0389a215c2980bb175cbd92eb38352dd1053004bdf627dc9bed033.jpg)

![316c5d67693626a82345f063a4a801f3542663e3c70f7065cea32c98ade3690d.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/316c5d67693626a82345f063a4a801f3542663e3c70f7065cea32c98ade3690d.jpg)

![4fe9eb687847336196bf68ed1e53270e1c24c94f4858b6e3be24a01c5ad7e278.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/4fe9eb687847336196bf68ed1e53270e1c24c94f4858b6e3be24a01c5ad7e278.jpg)

![59b989221cd1a9fe163c3168ef9bfb8cbb36be88f2b59642e0f6c8ddd712ffcd.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/59b989221cd1a9fe163c3168ef9bfb8cbb36be88f2b59642e0f6c8ddd712ffcd.jpg)

![70cfadc7332b520380546a0728cbe1a3a8f333df7b38df01ceade1a8d531c945.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/70cfadc7332b520380546a0728cbe1a3a8f333df7b38df01ceade1a8d531c945.jpg)

![802f0e0589962ec1c0c80fae508fd1a39eeeec2650297e16f0aaa2f21c877fe2.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/802f0e0589962ec1c0c80fae508fd1a39eeeec2650297e16f0aaa2f21c877fe2.jpg)

![bd87c83fd2202108d05a7d4e1c4c4eaa49414855dbd855bfcf9767ff5256e8a9.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/bd87c83fd2202108d05a7d4e1c4c4eaa49414855dbd855bfcf9767ff5256e8a9.jpg)

![bf044572e6739b003a9475d1418ebfa071d79f0de0b977b53864a79dc91baf66.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/bf044572e6739b003a9475d1418ebfa071d79f0de0b977b53864a79dc91baf66.jpg)

![c4cd9e7f94225320fde5a7ddc61566c83498c29b572e13a8d4a85207beb7f16c.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/c4cd9e7f94225320fde5a7ddc61566c83498c29b572e13a8d4a85207beb7f16c.jpg)

![d007979f43ed208a6f618ede50e6cfe91430b54e6e33c430ce3e5442f29a3e88.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/d007979f43ed208a6f618ede50e6cfe91430b54e6e33c430ce3e5442f29a3e88.jpg)

![d712aeb7b5f129c7f9a14073191da1e06855ed51b6fa56c08cf7687f08d83aa4.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/d712aeb7b5f129c7f9a14073191da1e06855ed51b6fa56c08cf7687f08d83aa4.jpg)

![db9d108e7185f1b35d577b34b4e9c9765b5b2d4dfbe04f39197d59f333d78fc2.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/db9d108e7185f1b35d577b34b4e9c9765b5b2d4dfbe04f39197d59f333d78fc2.jpg)

![e7c53b70c38bd06ca2f7b41964834feb4b1d745ee4b782203f5ec25959070c9a.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/images/e7c53b70c38bd06ca2f7b41964834feb4b1d745ee4b782203f5ec25959070c9a.jpg)

### Tables

![1441d9bc5fde54c2c491469e1e3d5d56ecd1aa686427fa95a753fdf8ff556c88.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/1441d9bc5fde54c2c491469e1e3d5d56ecd1aa686427fa95a753fdf8ff556c88.jpg)

![14f02466865821fa0899fc1dcda60d9c556cda7b276dd3f52c06f487157bf9d2.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/14f02466865821fa0899fc1dcda60d9c556cda7b276dd3f52c06f487157bf9d2.jpg)

![49d6c77c703404087a3ce0568c0a7a068a8f188021746483c66c525be132b637.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/49d6c77c703404087a3ce0568c0a7a068a8f188021746483c66c525be132b637.jpg)

![8862d19a6f7c18f21813fc5fb838d9d3daafba06afad788798eea9944ff7fdf7.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/8862d19a6f7c18f21813fc5fb838d9d3daafba06afad788798eea9944ff7fdf7.jpg)

![934115c8366b9e7e1d18a2fa6e36d2c0baa04a74fc43292c821b3a8fc4039426.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/934115c8366b9e7e1d18a2fa6e36d2c0baa04a74fc43292c821b3a8fc4039426.jpg)

![95f2091ccbab583f085b4717051058aea2cd2f2bc0c1d4db0b9c83740020d6cb.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/95f2091ccbab583f085b4717051058aea2cd2f2bc0c1d4db0b9c83740020d6cb.jpg)

![b80250942528a2ca55a6b1c9c746679e16120e8282dc42e64722c97f1f96c607.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/b80250942528a2ca55a6b1c9c746679e16120e8282dc42e64722c97f1f96c607.jpg)

![dcd9da2b569cbdfc37e5561ec9a93bea5714b983f30e59d4149e44be05c3b266.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/dcd9da2b569cbdfc37e5561ec9a93bea5714b983f30e59d4149e44be05c3b266.jpg)

![ebc0d86ac30b8fd916b628cdd73e6ff653a5702a42dd0f000776064309f29767.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/ebc0d86ac30b8fd916b628cdd73e6ff653a5702a42dd0f000776064309f29767.jpg)

![fc37453072066efbba823f6a429a9fad8233af2e90a23d20979d4ec0173d7f86.jpg](../icml_results/2447_EpiCache_%20Episodic%20KV%20Cache%20Management%20for%20Long-Term%20Conversation%20on%20Resource-Constrained%20Environmen/tables/fc37453072066efbba823f6a429a9fad8233af2e90a23d20979d4ec0173d7f86.jpg)

<a id="paper-2418"></a>

## From Drift to Coherence: Stabilizing Beliefs in LLMs


### Images

![0b5ed11c7fe5bc60b638abfc5c5f59d81014e58c65015ab6236f3cd60b603af3.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/0b5ed11c7fe5bc60b638abfc5c5f59d81014e58c65015ab6236f3cd60b603af3.jpg)

![4714cec142207c8a36b497a876ca6750256494569bd7d54719f8aa2963d1d693.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/4714cec142207c8a36b497a876ca6750256494569bd7d54719f8aa2963d1d693.jpg)

![5de91b6c4705804e3e0c9d1377e65ff12778ca50cd458c42e46d43088d65f98e.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/5de91b6c4705804e3e0c9d1377e65ff12778ca50cd458c42e46d43088d65f98e.jpg)

![6c8eb0e56efecfe7409cecebc294eaa64a6f3a0cc9a2d20719a78653cb2cb20a.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/6c8eb0e56efecfe7409cecebc294eaa64a6f3a0cc9a2d20719a78653cb2cb20a.jpg)

![7051cabb95c249f29ef8f40874e57b9332e09ba7446985b2daf73aa93594e7b6.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/7051cabb95c249f29ef8f40874e57b9332e09ba7446985b2daf73aa93594e7b6.jpg)

![95ee15454ade1fc74131992a8e9b37b9c1bdab00ddfe4326e827f3d59cc6664c.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/95ee15454ade1fc74131992a8e9b37b9c1bdab00ddfe4326e827f3d59cc6664c.jpg)

![cd82818676cd28b717feb1058e8d28ab19db13e9631a437068a0bfa4e5fda233.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/images/cd82818676cd28b717feb1058e8d28ab19db13e9631a437068a0bfa4e5fda233.jpg)

### Tables

![411a1eb6d561886e7538fe1346cf7e0dddcc9aeaf9603d28c0a60d6d5b151576.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/411a1eb6d561886e7538fe1346cf7e0dddcc9aeaf9603d28c0a60d6d5b151576.jpg)

![7b399caf092c96c98af9296541b6e9ea73ab9f3c9357a825a2bc61e61f9e0ed9.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/7b399caf092c96c98af9296541b6e9ea73ab9f3c9357a825a2bc61e61f9e0ed9.jpg)

![a4639e7f17ab2c93957d9cccaf3f9e16eac9dcc36f94934fa229891aa97d3d76.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/a4639e7f17ab2c93957d9cccaf3f9e16eac9dcc36f94934fa229891aa97d3d76.jpg)

![b5925c61838f7eb5fafdd693aea1f1162aab19ac02a4ceec57e2263d4f0746d4.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/b5925c61838f7eb5fafdd693aea1f1162aab19ac02a4ceec57e2263d4f0746d4.jpg)

![c83aa3a441ad3ce0d3117f0b03ad0c53a009455574e6246c48df13d1ff3539f9.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/c83aa3a441ad3ce0d3117f0b03ad0c53a009455574e6246c48df13d1ff3539f9.jpg)

![d076563e9911d15ed612a7baf0b5e9b999be4fe5635a7bd2a126f33e6baafded.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/d076563e9911d15ed612a7baf0b5e9b999be4fe5635a7bd2a126f33e6baafded.jpg)

![ecb700224ad50b0533e933e4d603e118a26269132022faddae6de33c975edbd7.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/ecb700224ad50b0533e933e4d603e118a26269132022faddae6de33c975edbd7.jpg)

![f40f42baa5936ac5dbe3b12d19c94ca66046ea14527a066c90a1583f1b4d88cf.jpg](../icml_results/2448_From%20Drift%20to%20Coherence_%20Stabilizing%20Beliefs%20in%20LLMs/tables/f40f42baa5936ac5dbe3b12d19c94ca66046ea14527a066c90a1583f1b4d88cf.jpg)

<a id="paper-2419"></a>

## Frictional Q-Learning


### Images

![1ec1c35a41f467b6da008c05079e5a3e4aecd1d635d44b742a04d52c0e73cba8.jpg](../icml_results/2449_Frictional%20Q-Learning/images/1ec1c35a41f467b6da008c05079e5a3e4aecd1d635d44b742a04d52c0e73cba8.jpg)

![42b455c06bd5b34546d9cef396726eebfaf1d7a767e3271fff6e04a9e2658d30.jpg](../icml_results/2449_Frictional%20Q-Learning/images/42b455c06bd5b34546d9cef396726eebfaf1d7a767e3271fff6e04a9e2658d30.jpg)

![4e15f349b2fa56590f1441c536fccbbf0dc77e50c10aedf9fc2dcb3acc908a11.jpg](../icml_results/2449_Frictional%20Q-Learning/images/4e15f349b2fa56590f1441c536fccbbf0dc77e50c10aedf9fc2dcb3acc908a11.jpg)

![ae3d0f001b695f38cc6e8bbd7e79b65a404c11ec6220302739dc2399417bc907.jpg](../icml_results/2449_Frictional%20Q-Learning/images/ae3d0f001b695f38cc6e8bbd7e79b65a404c11ec6220302739dc2399417bc907.jpg)

![ae7d54efb73eff4ceba27224fe39da2e68d57d9754a6e4e972bba56e360b8597.jpg](../icml_results/2449_Frictional%20Q-Learning/images/ae7d54efb73eff4ceba27224fe39da2e68d57d9754a6e4e972bba56e360b8597.jpg)

### Tables

![05893a88384c01ebd532f98272abf56d59c068bc3944f20480f652aea4819449.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/05893a88384c01ebd532f98272abf56d59c068bc3944f20480f652aea4819449.jpg)

![3472aad117a53f9366d1aa288990fd432206290f818c55b77a1eb98c026d9dc1.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/3472aad117a53f9366d1aa288990fd432206290f818c55b77a1eb98c026d9dc1.jpg)

![4c67d36ba655a81b0e65edbf9b15369fefbf8ba3775347d039b09128e8850cd3.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/4c67d36ba655a81b0e65edbf9b15369fefbf8ba3775347d039b09128e8850cd3.jpg)

![7ab685a0d712f3a203f0b45590f80073124d663c2c817f8b03df932f53e6a4d7.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/7ab685a0d712f3a203f0b45590f80073124d663c2c817f8b03df932f53e6a4d7.jpg)

![94d726b792cddd85da5f64ab70847522b1873e34a6f0f36ef7b2a6dcb68668d2.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/94d726b792cddd85da5f64ab70847522b1873e34a6f0f36ef7b2a6dcb68668d2.jpg)

![9dacd7e560f20efc0c93663a3bda2691be2e5af60fccd6968811536f528e17e4.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/9dacd7e560f20efc0c93663a3bda2691be2e5af60fccd6968811536f528e17e4.jpg)

![b385ea264826ee4b371d583bec522c6765ca099f92f02f7efc42d4d2a90e813c.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/b385ea264826ee4b371d583bec522c6765ca099f92f02f7efc42d4d2a90e813c.jpg)

![e6eb426a0d7a0bb52e49e9ffca530c9708c3684a66f5de9e4f16512fef3fd9ab.jpg](../icml_results/2449_Frictional%20Q-Learning/tables/e6eb426a0d7a0bb52e49e9ffca530c9708c3684a66f5de9e4f16512fef3fd9ab.jpg)

<a id="paper-2420"></a>

## Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway


### Images

![4fbee2cbd7ebc0a1c73c98615a9eabc55c6516b91266554cd926a4f59b59aad4.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/4fbee2cbd7ebc0a1c73c98615a9eabc55c6516b91266554cd926a4f59b59aad4.jpg)

![68484b7f0272ece92f391a8a337c2f032d494b375e27713a63df362e0473b302.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/68484b7f0272ece92f391a8a337c2f032d494b375e27713a63df362e0473b302.jpg)

![759fa8610f29752f0bba324c3a587463d4cc4750570c9a9ade5fa3690d1d2a9e.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/759fa8610f29752f0bba324c3a587463d4cc4750570c9a9ade5fa3690d1d2a9e.jpg)

![869e0f0f9dca4ba87bd3aed19b8c97a8c04b2b8695d47e20d22b8300f05d9e46.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/869e0f0f9dca4ba87bd3aed19b8c97a8c04b2b8695d47e20d22b8300f05d9e46.jpg)

![c34bf131f5dbb288bf2d163a5aff24488dd925e64d4fd2366cfebc0acc59b1d7.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/c34bf131f5dbb288bf2d163a5aff24488dd925e64d4fd2366cfebc0acc59b1d7.jpg)

![c4cda55900a7c50fef3d278a70dd749bdb744bb2a639a3b17374488ceebe422a.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/c4cda55900a7c50fef3d278a70dd749bdb744bb2a639a3b17374488ceebe422a.jpg)

![c6a8b5d3e61ff15b71974ef9e38863ffa8ca874e853436435a2c4780134ca437.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/c6a8b5d3e61ff15b71974ef9e38863ffa8ca874e853436435a2c4780134ca437.jpg)

![cf0ee8f321b8b6cef68142ec39b13857acacef38eadf912a25244d1796f8f1e0.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/cf0ee8f321b8b6cef68142ec39b13857acacef38eadf912a25244d1796f8f1e0.jpg)

![dc3dc4108d191ce8439364ff236d4a51e345ad0e4e41dd0ac3bf62046d3852ed.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/dc3dc4108d191ce8439364ff236d4a51e345ad0e4e41dd0ac3bf62046d3852ed.jpg)

![e63ae1921cc1c98059ef1e330b24b04f52edea96a9c1cd4e2ba919e6167a0a12.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/images/e63ae1921cc1c98059ef1e330b24b04f52edea96a9c1cd4e2ba919e6167a0a12.jpg)

### Tables

![2840bc13749636c2af7e8d2d394d9ad101edab039ea03c3c862aa4ebc26e8f67.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/tables/2840bc13749636c2af7e8d2d394d9ad101edab039ea03c3c862aa4ebc26e8f67.jpg)

![f27478c7f035c063322f12935efd51773b2ff9eb9746e41d5b2ef9a970cd66ad.jpg](../icml_results/2450_Gradient%20Descent%20with%20Large%20Step%20Size%20Restores%20Symmetry%20in%20Deep%20Linear%20Networks%20with%20Multi-Pathway/tables/f27478c7f035c063322f12935efd51773b2ff9eb9746e41d5b2ef9a970cd66ad.jpg)

<a id="paper-2421"></a>

## Breaking the Reference Bottleneck via Learning to Rewrite Conversational Queries without Gold Reference Passages


### Images

![2f22d9f73cee28f94ca0f955bb6e529dd94bb0c7b9c94084c17c9953a782f0b0.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/2f22d9f73cee28f94ca0f955bb6e529dd94bb0c7b9c94084c17c9953a782f0b0.jpg)

![34d69bc689cec2daa9e45a9f5f09da1bd849dc8817b216423f68633dbddbf5c5.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/34d69bc689cec2daa9e45a9f5f09da1bd849dc8817b216423f68633dbddbf5c5.jpg)

![415c8f9dae9598f39c87e5c9fcdb4a5728e2b675effdb7baf2c22130165de972.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/415c8f9dae9598f39c87e5c9fcdb4a5728e2b675effdb7baf2c22130165de972.jpg)

![4bbff9c72da320744b6039564539f87943599199ecc3bb541d52e2f1746d431f.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/4bbff9c72da320744b6039564539f87943599199ecc3bb541d52e2f1746d431f.jpg)

![61ea73ea4f0363acaf189d7e337cfd8affe0bb508f0ccb77fb258f80dff28314.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/61ea73ea4f0363acaf189d7e337cfd8affe0bb508f0ccb77fb258f80dff28314.jpg)

![8847ee14f6d116c503d31465b72ea3396ceb0f52f74ab584c655360da629550a.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/8847ee14f6d116c503d31465b72ea3396ceb0f52f74ab584c655360da629550a.jpg)

![9be11c93c1a552737f99171471f319048942fa272a4544b44e028fe7ae8e9d53.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/images/9be11c93c1a552737f99171471f319048942fa272a4544b44e028fe7ae8e9d53.jpg)

### Tables

![05fdf9e8fda82227ccd634bb651e75280bbcbea0c793300bdfdb0320bcabebcf.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/05fdf9e8fda82227ccd634bb651e75280bbcbea0c793300bdfdb0320bcabebcf.jpg)

![12813056756cb7681d9cad0be5f5923c49f09d81b2c90ae0790e4723493976db.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/12813056756cb7681d9cad0be5f5923c49f09d81b2c90ae0790e4723493976db.jpg)

![1bfe6262cce2e46fef8cbf1a9151bf082261d2dd613e5f4dff25ddefa3b70dff.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/1bfe6262cce2e46fef8cbf1a9151bf082261d2dd613e5f4dff25ddefa3b70dff.jpg)

![1d89e17b72db6becff8a5638986de76e4f18ea25a85c81b019b83e273feaed3e.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/1d89e17b72db6becff8a5638986de76e4f18ea25a85c81b019b83e273feaed3e.jpg)

![2bd07cd5d9bed5c16aa272af4ca8cbd18fdeedb06ae173cb497eb149891675ca.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/2bd07cd5d9bed5c16aa272af4ca8cbd18fdeedb06ae173cb497eb149891675ca.jpg)

![2c99d2371202f715584872802b5bf3a0cddc80dab597b7c8b64fe2583b8446ea.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/2c99d2371202f715584872802b5bf3a0cddc80dab597b7c8b64fe2583b8446ea.jpg)

![445e6d0de59d8f84e22092ab59af279c79d42741aa4b03dcefa11e9392e7084e.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/445e6d0de59d8f84e22092ab59af279c79d42741aa4b03dcefa11e9392e7084e.jpg)

![4c84b81b887d84f8136e23660c216abac41be46613a60b3abb95ce49351f6ccc.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/4c84b81b887d84f8136e23660c216abac41be46613a60b3abb95ce49351f6ccc.jpg)

![65ef38f39a0fc09c5a9edcbf1fa68e84fb73d734faf614d0839824dd9afba062.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/65ef38f39a0fc09c5a9edcbf1fa68e84fb73d734faf614d0839824dd9afba062.jpg)

![6a08ca63137887bd21b275162e198cb0a742bb59246fbea3d0fc8d97e8470ece.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/6a08ca63137887bd21b275162e198cb0a742bb59246fbea3d0fc8d97e8470ece.jpg)

![729da955ce82a8c2bcc8c6edbefa7251ba1b9c37ab7ca308bbd4c5bb17c74f79.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/729da955ce82a8c2bcc8c6edbefa7251ba1b9c37ab7ca308bbd4c5bb17c74f79.jpg)

![787ef91028029f3d4451ee4d1802824b5d3f6e2d0e72df6cb47166570d179e43.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/787ef91028029f3d4451ee4d1802824b5d3f6e2d0e72df6cb47166570d179e43.jpg)

![7901063264aa690ce1c04968a68725e2a889795d655539c3259a9fc48d791214.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/7901063264aa690ce1c04968a68725e2a889795d655539c3259a9fc48d791214.jpg)

![7d1f9c0c58d1bbf86675311ccdffd1b4c43e31265466041d85d4df45fce23064.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/7d1f9c0c58d1bbf86675311ccdffd1b4c43e31265466041d85d4df45fce23064.jpg)

![8d860ebb9fbb7c0643a9bf715aeafb93e7ae735b83c005c3a82266a574e412e8.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/8d860ebb9fbb7c0643a9bf715aeafb93e7ae735b83c005c3a82266a574e412e8.jpg)

![92b2e063b6f50e5e821fabfa01532d2e868ceed6276896ebf5aea0f16c0d35af.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/92b2e063b6f50e5e821fabfa01532d2e868ceed6276896ebf5aea0f16c0d35af.jpg)

![9b2dd64fd69b396a77b198305516fccf03f4aa571cecd4dae3e0536cf1444bee.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/9b2dd64fd69b396a77b198305516fccf03f4aa571cecd4dae3e0536cf1444bee.jpg)

![9ecd95afd70e51b6c2d5bb84b21e9921c4d372e8bae1464f446a1fec0a818fca.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/9ecd95afd70e51b6c2d5bb84b21e9921c4d372e8bae1464f446a1fec0a818fca.jpg)

![a1f8ce3a8e308a988014bd7ea248eeb25f8d90f7256011f3f081963f47fc4b70.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/a1f8ce3a8e308a988014bd7ea248eeb25f8d90f7256011f3f081963f47fc4b70.jpg)

![abdb76f334e61ea28a4842beed4c17b4462b78fa6541325a1ce1b68c023bb827.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/abdb76f334e61ea28a4842beed4c17b4462b78fa6541325a1ce1b68c023bb827.jpg)

![bf067c049ed2a6ef23ac70d6e956eb306b3a59197911b5f961abfa459b9f1b78.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/bf067c049ed2a6ef23ac70d6e956eb306b3a59197911b5f961abfa459b9f1b78.jpg)

![c90a319491e9e0bb5d0c9e21d9643c3db17914f0a1d2a3b00b313f88759ac84f.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/c90a319491e9e0bb5d0c9e21d9643c3db17914f0a1d2a3b00b313f88759ac84f.jpg)

![e60d3b1aee96ce1693a5e1a18033d6d0bf009f648c521663e0364ba537098912.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/e60d3b1aee96ce1693a5e1a18033d6d0bf009f648c521663e0364ba537098912.jpg)

![ec95b8d02243d625178e738ac3b859be8933d4fc6a497233113322cd8e2c041c.jpg](../icml_results/2451_Breaking%20the%20Reference%20Bottleneck%20via%20Learning%20to%20Rewrite%20Conversational%20Queries%20without%20Gold%20Refere/tables/ec95b8d02243d625178e738ac3b859be8933d4fc6a497233113322cd8e2c041c.jpg)

<a id="paper-2422"></a>

## ST-Veto: Spatio-Temporal Token Veto for Diffusion MLLMs via Taylor Prediction and Visual Grounding


### Images

![25236113d40623f7ed5c34d68689321ee6f454ed6958d71b94bddca8a982d50a.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/images/25236113d40623f7ed5c34d68689321ee6f454ed6958d71b94bddca8a982d50a.jpg)

![2b58ac4ddfa75c3165ddc357716d04d8e1c595d34f6ec7bfbf3d13befd60fb60.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/images/2b58ac4ddfa75c3165ddc357716d04d8e1c595d34f6ec7bfbf3d13befd60fb60.jpg)

![4c2278a400faa2516701113d2f9ae62b303418397c35be202feaba15ee6a046f.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/images/4c2278a400faa2516701113d2f9ae62b303418397c35be202feaba15ee6a046f.jpg)

### Tables

![0623562f6731a8acaaa3be8324e9a08224c4081f38483ffd2e805cff252a99d8.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/0623562f6731a8acaaa3be8324e9a08224c4081f38483ffd2e805cff252a99d8.jpg)

![313da9bb85442318793f99809363297a417f11619d3513680d54b32b74d3d4e8.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/313da9bb85442318793f99809363297a417f11619d3513680d54b32b74d3d4e8.jpg)

![3b7256787d8ddadb411e66a0f2212880a5e60dbac54e5054523746056bc00f71.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/3b7256787d8ddadb411e66a0f2212880a5e60dbac54e5054523746056bc00f71.jpg)

![41dda634fa0caefd9a1405e8cca85a39d2b113a312b18214881dabf72944fe12.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/41dda634fa0caefd9a1405e8cca85a39d2b113a312b18214881dabf72944fe12.jpg)

![4b701bc4d3c4f5d001a5395a1ef72dfbeb00836625f42bd439e84772f8d88c54.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/4b701bc4d3c4f5d001a5395a1ef72dfbeb00836625f42bd439e84772f8d88c54.jpg)

![4e28a0865db046a0a8a0b7598af7e7af70f804b5eb4d388cbce8c59a4229843e.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/4e28a0865db046a0a8a0b7598af7e7af70f804b5eb4d388cbce8c59a4229843e.jpg)

![52921027d2781f320adb7dcc2e717df7e84f13da5e2e2f8bbe084b24aea16a67.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/52921027d2781f320adb7dcc2e717df7e84f13da5e2e2f8bbe084b24aea16a67.jpg)

![6e4902145079206df1db9b16f998c59e02e76be3b09d7b63470045c1943b9c7d.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/6e4902145079206df1db9b16f998c59e02e76be3b09d7b63470045c1943b9c7d.jpg)

![7a3f1a2e0731e41efe7c39c7d9676a0581346c4c190885a59e1236427066063d.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/7a3f1a2e0731e41efe7c39c7d9676a0581346c4c190885a59e1236427066063d.jpg)

![7c02ceb7aba8ef71c48b330f18099901677c0882cb20f0f30b8db236db3ef494.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/7c02ceb7aba8ef71c48b330f18099901677c0882cb20f0f30b8db236db3ef494.jpg)

![a2af8949789330e292c9fed1c0aa83c07076ad8a5a707f34117df1c03eece3ef.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/a2af8949789330e292c9fed1c0aa83c07076ad8a5a707f34117df1c03eece3ef.jpg)

![d8875d9224ada7d89f15d0f171b4d702c0c9c7b7980886295cedad7dcd6cd855.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/d8875d9224ada7d89f15d0f171b4d702c0c9c7b7980886295cedad7dcd6cd855.jpg)

![ebd0dc8cec52f50d8f45ec12c48621488bb59a2e4e17c7c52b22ddcabf775c19.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/ebd0dc8cec52f50d8f45ec12c48621488bb59a2e4e17c7c52b22ddcabf775c19.jpg)

![f4ed40f60337e4075dc257a1f0ac325025ba48d704c524a75711ac7b97378f8d.jpg](../icml_results/2452_ST-Veto_%20Spatio-Temporal%20Token%20Veto%20for%20Diffusion%20MLLMs%20via%20Taylor%20Prediction%20and%20Visual%20Grounding/tables/f4ed40f60337e4075dc257a1f0ac325025ba48d704c524a75711ac7b97378f8d.jpg)

<a id="paper-2423"></a>

## WAVE: Window-Aware Vocabulary-Efficient Early-Exit for Training-Free LLM Acceleration


### Images

![1ec78cb486a8a3d36ea52509297fafd2aa31e5c20021fada67937b364982cd1e.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/1ec78cb486a8a3d36ea52509297fafd2aa31e5c20021fada67937b364982cd1e.jpg)

![5ef1bf4fad8ab0393a9a7e9279710a56cb1012a07250899f070e7c48a6397ef7.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/5ef1bf4fad8ab0393a9a7e9279710a56cb1012a07250899f070e7c48a6397ef7.jpg)

![5f742f50ce4504fde5f45394e244a1ca9304b5452e65a972f9f5dcae62bdd534.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/5f742f50ce4504fde5f45394e244a1ca9304b5452e65a972f9f5dcae62bdd534.jpg)

![6719f44eb1b896dcd4c435dd301b34b04e72dc9a6a3548966b6e6eed9087d485.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/6719f44eb1b896dcd4c435dd301b34b04e72dc9a6a3548966b6e6eed9087d485.jpg)

![71640059694afeef7a84c9989bc3ac84b1e332d9c2cfe727a6402efe4ef095b4.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/71640059694afeef7a84c9989bc3ac84b1e332d9c2cfe727a6402efe4ef095b4.jpg)

![84a3fec0cd88e4d5d9daf104368a4a7da957aa7cba053801a488bac840b8ad1a.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/84a3fec0cd88e4d5d9daf104368a4a7da957aa7cba053801a488bac840b8ad1a.jpg)

![991c8533d2cd97c17d3815158dbd91785036e322c4abdcb38ed65c5f9940d0e4.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/991c8533d2cd97c17d3815158dbd91785036e322c4abdcb38ed65c5f9940d0e4.jpg)

![c962ea55a31bbddb67845deb5853c0d3dba54d16d74601bc72ab1bbf0e902e43.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/c962ea55a31bbddb67845deb5853c0d3dba54d16d74601bc72ab1bbf0e902e43.jpg)

![e3624ce10538b1f90f26f40c7d00c12e637d5ab3fab889483d51501de2181fca.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/images/e3624ce10538b1f90f26f40c7d00c12e637d5ab3fab889483d51501de2181fca.jpg)

### Tables

![85f981998c536d1dbff35200a18d292e43518da6371795a0f721ac0700fd3362.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/85f981998c536d1dbff35200a18d292e43518da6371795a0f721ac0700fd3362.jpg)

![c50ab95c291087d77b5de73c281e8d826d83e3b248ef89a80232e087025df34e.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/c50ab95c291087d77b5de73c281e8d826d83e3b248ef89a80232e087025df34e.jpg)

![c7ea6b5588f04a1788c2719929e44035ea3f3927368759aad7e7173bd4ede49b.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/c7ea6b5588f04a1788c2719929e44035ea3f3927368759aad7e7173bd4ede49b.jpg)

![c8005da2dac6f317091321d146106356277bd93c0e542ccd5f237486bbb525f9.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/c8005da2dac6f317091321d146106356277bd93c0e542ccd5f237486bbb525f9.jpg)

![d1c94bad3e7eade0173ca0d27722567063c9b39aa01b4a8c7e91a87899ff081e.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/d1c94bad3e7eade0173ca0d27722567063c9b39aa01b4a8c7e91a87899ff081e.jpg)

![d3dcf2bb18074ffa07dd1c0c11629281dd3f93b09e0e37fb4c4d1f41cfec2b83.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/d3dcf2bb18074ffa07dd1c0c11629281dd3f93b09e0e37fb4c4d1f41cfec2b83.jpg)

![d515db3078a1a1e08392f028cec4adc07ccd8670c2ba03971ef82ad2b5ebcf7e.jpg](../icml_results/2453_WAVE_%20Window-Aware%20Vocabulary-Efficient%20Early-Exit%20for%20Training-Free%20LLM%20Acceleration/tables/d515db3078a1a1e08392f028cec4adc07ccd8670c2ba03971ef82ad2b5ebcf7e.jpg)

<a id="paper-2424"></a>

## Machine Learning Hamiltonians are Accurate Energy-Force Predictors


### Images

![0da72b43c4ca8be4444ee6bd5b5ff72962a258c10d85891321da5c7986520710.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/0da72b43c4ca8be4444ee6bd5b5ff72962a258c10d85891321da5c7986520710.jpg)

![0db044b5eb7a720c0141a7be2db7b7c47daa3a9c2e66ea593e7c0d2401bb4f3b.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/0db044b5eb7a720c0141a7be2db7b7c47daa3a9c2e66ea593e7c0d2401bb4f3b.jpg)

![24ad325335d93a66e01575d5a3833a99f65cb35173218c5c9b6525eb2d00685c.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/24ad325335d93a66e01575d5a3833a99f65cb35173218c5c9b6525eb2d00685c.jpg)

![5be3594d228daf4090250fa8641bf0788de90c1fcd14b2029f9f67bf84640a29.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/5be3594d228daf4090250fa8641bf0788de90c1fcd14b2029f9f67bf84640a29.jpg)

![73fac546b7f43da42ae2cbbdbe609d2cf342dfa6965d8fefcc191536b0500e9b.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/73fac546b7f43da42ae2cbbdbe609d2cf342dfa6965d8fefcc191536b0500e9b.jpg)

![78e8a35ef625653db244e02b35b4b74bba5a256ad81ab9f930d30e0e0723d0a9.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/78e8a35ef625653db244e02b35b4b74bba5a256ad81ab9f930d30e0e0723d0a9.jpg)

![854235f4fa69492b7faa628e2fa4530eb57cf05ed210d5771ffa490bb532b2b1.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/854235f4fa69492b7faa628e2fa4530eb57cf05ed210d5771ffa490bb532b2b1.jpg)

![eda3d34f9b3a87130924e28805c7460254256dca3446c3b5ccbb14e81bc7213f.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/images/eda3d34f9b3a87130924e28805c7460254256dca3446c3b5ccbb14e81bc7213f.jpg)

### Tables

![02b1a59f68a32c9949b24ffde1f544e84165113891c86e10210e14a931106fdf.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/02b1a59f68a32c9949b24ffde1f544e84165113891c86e10210e14a931106fdf.jpg)

![0a4132f91ce9653f76ab7336fe730d837f1de3e14ee61ab54d34e61c23dcaf58.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/0a4132f91ce9653f76ab7336fe730d837f1de3e14ee61ab54d34e61c23dcaf58.jpg)

![2dee5b095119167476d9e5851c6b2db059356debd34bfea807b3410a168a9fe7.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/2dee5b095119167476d9e5851c6b2db059356debd34bfea807b3410a168a9fe7.jpg)

![5027a0d663db061b667f936d9f109b5f48e38f34247f35fbc5d017a4170d870a.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/5027a0d663db061b667f936d9f109b5f48e38f34247f35fbc5d017a4170d870a.jpg)

![565d55cea64c5c8c79708b78f86c0c19aec64be0045e9a613b882ec1079099c8.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/565d55cea64c5c8c79708b78f86c0c19aec64be0045e9a613b882ec1079099c8.jpg)

![7031060dbba135a94ee90f01172e15de387ddc40861bd57f5e07d3af850cf067.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/7031060dbba135a94ee90f01172e15de387ddc40861bd57f5e07d3af850cf067.jpg)

![7e4244dea50d7cf5b05a01713b3313a1f6d9113711784c168a001fdea858af29.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/7e4244dea50d7cf5b05a01713b3313a1f6d9113711784c168a001fdea858af29.jpg)

![7ec16e694ed79518c13ee1190e6de3cdafb90e7384f0347e545c073af17443ae.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/7ec16e694ed79518c13ee1190e6de3cdafb90e7384f0347e545c073af17443ae.jpg)

![81a39796d3f1686d992dff014cff9a65d12776065fc388382f8939ac7c872c6a.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/81a39796d3f1686d992dff014cff9a65d12776065fc388382f8939ac7c872c6a.jpg)

![87773df8a52b75c1109887fdf79775ca84bcce26d53a49d7a698b55f53e83384.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/87773df8a52b75c1109887fdf79775ca84bcce26d53a49d7a698b55f53e83384.jpg)

![891618fa0c42aa7f170189836ed50079343b2850276dfe0eeff39a4920c665b9.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/891618fa0c42aa7f170189836ed50079343b2850276dfe0eeff39a4920c665b9.jpg)

![897327c7688fbccd0c2912695258708a5d313230feb7cf01df87e9a651c2acf7.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/897327c7688fbccd0c2912695258708a5d313230feb7cf01df87e9a651c2acf7.jpg)

![93394686ea30595e896241586709a45e296feee9be8fd5a1fcd7557060364ab7.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/93394686ea30595e896241586709a45e296feee9be8fd5a1fcd7557060364ab7.jpg)

![935ad72d120bdd5b8f3c6554903f5739c9f0bb639b7c4e8f498e44d57097a7e1.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/935ad72d120bdd5b8f3c6554903f5739c9f0bb639b7c4e8f498e44d57097a7e1.jpg)

![a1ac8fbf4fe62667f30acc6136432fa5c983080020db4dc9ec0211764414524c.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/a1ac8fbf4fe62667f30acc6136432fa5c983080020db4dc9ec0211764414524c.jpg)

![a2c9280661260fce111a564032fb13a8f2b96eb38392427cfe76258a31460603.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/a2c9280661260fce111a564032fb13a8f2b96eb38392427cfe76258a31460603.jpg)

![b03045e3057094604d4f470d0c55bbaf981feeca305addd4dd06ec84fdf44b27.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/b03045e3057094604d4f470d0c55bbaf981feeca305addd4dd06ec84fdf44b27.jpg)

![b1831894d13b545b821bffaf56246db6e51f7b1335f2410e9f09615a5c12cb55.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/b1831894d13b545b821bffaf56246db6e51f7b1335f2410e9f09615a5c12cb55.jpg)

![b568faa94c8dc8e6919e62aba5bca41055a00501f01da177ab5f4419e39900d0.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/b568faa94c8dc8e6919e62aba5bca41055a00501f01da177ab5f4419e39900d0.jpg)

![bc8efafba0da09f6ae2dbffaf25c044bd3654a9eac984d169d7dbdc27a1c69f4.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/bc8efafba0da09f6ae2dbffaf25c044bd3654a9eac984d169d7dbdc27a1c69f4.jpg)

![bf732a53b35eaef9e5b373e6b059661c2fb18281b030ffb21d3a2f2c51efe878.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/bf732a53b35eaef9e5b373e6b059661c2fb18281b030ffb21d3a2f2c51efe878.jpg)

![bf8fbb29d1d28362fa15557315ba04b416d1bf11422cfa468d022f8c091f5e9b.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/bf8fbb29d1d28362fa15557315ba04b416d1bf11422cfa468d022f8c091f5e9b.jpg)

![c08c62114c0971581cf766d9ea3e302d175b7cbb049d0bd24b57cc556813b922.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/c08c62114c0971581cf766d9ea3e302d175b7cbb049d0bd24b57cc556813b922.jpg)

![cbb218b0dee187c324d7d0cd15feb36f29df9232962b9beb2fe14a12aab6f551.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/cbb218b0dee187c324d7d0cd15feb36f29df9232962b9beb2fe14a12aab6f551.jpg)

![cbe3246eb7cb4693829a32a37a8419e41cff970f6afdf0bcfbbfd1c534afc3c9.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/cbe3246eb7cb4693829a32a37a8419e41cff970f6afdf0bcfbbfd1c534afc3c9.jpg)

![e606397f781c854afcfed71d6d39fc6efad8697f985aba492ab128e1b4ceebc2.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/e606397f781c854afcfed71d6d39fc6efad8697f985aba492ab128e1b4ceebc2.jpg)

![e88b0c94de750271a6117702ad6be1c390c2804788d01498f653c4a1315ff7b1.jpg](../icml_results/2454_Machine%20Learning%20Hamiltonians%20are%20Accurate%20Energy-Force%20Predictors/tables/e88b0c94de750271a6117702ad6be1c390c2804788d01498f653c4a1315ff7b1.jpg)

<a id="paper-2425"></a>

## Contrastive Representation Regularization for Vision-Language-Action Models


### Images

![111d5687d1a437ce3c98183b4734ba125f06f95c77ab7d564090adc17dea8c8b.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/111d5687d1a437ce3c98183b4734ba125f06f95c77ab7d564090adc17dea8c8b.jpg)

![1ec4e4e91c541c7b6182d3a647614794775c19cb348c84ac6987061fda605f01.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/1ec4e4e91c541c7b6182d3a647614794775c19cb348c84ac6987061fda605f01.jpg)

![2abb523a2113c50cf76c8fa76757fcfe05b8a42f1d7a619c68e726ed47e5a58b.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/2abb523a2113c50cf76c8fa76757fcfe05b8a42f1d7a619c68e726ed47e5a58b.jpg)

![32f3ae518c0a67d9e6e755e8356dd36b852b9a215f185a1f574f1b8fda84a371.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/32f3ae518c0a67d9e6e755e8356dd36b852b9a215f185a1f574f1b8fda84a371.jpg)

![3b0b2f85a3b0f1c81c614ab537366b1c05347bb6b77015fac149c8eb9cd6192d.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/3b0b2f85a3b0f1c81c614ab537366b1c05347bb6b77015fac149c8eb9cd6192d.jpg)

![5df2151c0154bd0248ecb80193e51f52d571547333cb7c8e7ae247128a64f67a.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/5df2151c0154bd0248ecb80193e51f52d571547333cb7c8e7ae247128a64f67a.jpg)

![650786ababf1bb265ba7d099dd394ca2209580d0e5cdfb943fc926b16b34e0aa.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/650786ababf1bb265ba7d099dd394ca2209580d0e5cdfb943fc926b16b34e0aa.jpg)

![667ba25f60ef17fe2c2cad28ed9e21f98132fbd8087f44a32e1bdc766f3c8034.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/667ba25f60ef17fe2c2cad28ed9e21f98132fbd8087f44a32e1bdc766f3c8034.jpg)

![6d9f0596854da1412c6b5bbc183a187485e4c1601ed08e1148527a45b46c1679.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/6d9f0596854da1412c6b5bbc183a187485e4c1601ed08e1148527a45b46c1679.jpg)

![78e3e6c4c60ac76686e0834bc5fa78f7b92ba2eb44cfb8d59791df94b25e7a05.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/78e3e6c4c60ac76686e0834bc5fa78f7b92ba2eb44cfb8d59791df94b25e7a05.jpg)

![79d34f873701d383a63fec46777e7ab2c8c56bd3247819f27c35e3cdc8760d59.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/79d34f873701d383a63fec46777e7ab2c8c56bd3247819f27c35e3cdc8760d59.jpg)

![828447d12e97ea3352f51e4be54955a55be55e4aa0958da805d2339d0619d40f.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/828447d12e97ea3352f51e4be54955a55be55e4aa0958da805d2339d0619d40f.jpg)

![92ab5f500d7bdf7d776654af3a0a72bc8a9810ba33b43539bf0033da673670ff.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/92ab5f500d7bdf7d776654af3a0a72bc8a9810ba33b43539bf0033da673670ff.jpg)

![965ad762f80c5834b23d30507296b06181293b91efb421528ff9e351ef41d2a3.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/965ad762f80c5834b23d30507296b06181293b91efb421528ff9e351ef41d2a3.jpg)

![a1f308302541786e8db400f74d4d9c3be0cdea22ba856664a7690fa86baa7975.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/a1f308302541786e8db400f74d4d9c3be0cdea22ba856664a7690fa86baa7975.jpg)

![b30e458969211d29244ddf7eebf4b7f9e30753208f88629710ed8b328248efb3.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/b30e458969211d29244ddf7eebf4b7f9e30753208f88629710ed8b328248efb3.jpg)

![c6813c11313425ef0c8c8ac8b45d230559715f7b1ba9419300786b5111a330ce.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/c6813c11313425ef0c8c8ac8b45d230559715f7b1ba9419300786b5111a330ce.jpg)

![d32dd621496d1dd6ecfb82c817441aa724366af344d7795651f598f54c0e660b.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/d32dd621496d1dd6ecfb82c817441aa724366af344d7795651f598f54c0e660b.jpg)

![da6de747c6e094db6ed553b1a7ded34d762af7bcef6170da42ac2df7fbf015d0.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/da6de747c6e094db6ed553b1a7ded34d762af7bcef6170da42ac2df7fbf015d0.jpg)

![e70cab2ac095fe85804482c565d23a65ccdaff96529dfada715e92afa6e51184.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/e70cab2ac095fe85804482c565d23a65ccdaff96529dfada715e92afa6e51184.jpg)

![e8982e525e74dda98e6d141d658055b00525abf7bb48d272061ea4b450950d91.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/e8982e525e74dda98e6d141d658055b00525abf7bb48d272061ea4b450950d91.jpg)

![f14ad569f7eda50581ae64507b60e9239968cb2bcfc83e48be99fec47855788c.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/f14ad569f7eda50581ae64507b60e9239968cb2bcfc83e48be99fec47855788c.jpg)

![fed44cdab8c95fc1e076c97fa3bb7d75ac7d3732ca9da5201ac97171007a5dbe.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/images/fed44cdab8c95fc1e076c97fa3bb7d75ac7d3732ca9da5201ac97171007a5dbe.jpg)

### Tables

![0805758a3174d80a2af50bcee4630f71223b414fe0dc7a9060e4d3261102b4e4.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/0805758a3174d80a2af50bcee4630f71223b414fe0dc7a9060e4d3261102b4e4.jpg)

![117e331a339355c240885257804c3be80a07b5db275c7f1ea971d1e50e76a96a.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/117e331a339355c240885257804c3be80a07b5db275c7f1ea971d1e50e76a96a.jpg)

![1eeda9ab2e863fcf3772f97f600b62eac6657cfeb29228a3d2a7e0f8a701174c.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/1eeda9ab2e863fcf3772f97f600b62eac6657cfeb29228a3d2a7e0f8a701174c.jpg)

![1f155ed935844fe061b54aa38f66e4816be3d819013065038ac87a976a053c5f.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/1f155ed935844fe061b54aa38f66e4816be3d819013065038ac87a976a053c5f.jpg)

![4bd84c7bb0c2d80a3472ce8e2d7b4c2eaa8730c4b200ddc7a5dd08116acd8a31.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/4bd84c7bb0c2d80a3472ce8e2d7b4c2eaa8730c4b200ddc7a5dd08116acd8a31.jpg)

![4f4fd023167e6d3bfffe1d6bf000394572f1fccd8b93f5091c9c2b6adb2f9a92.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/4f4fd023167e6d3bfffe1d6bf000394572f1fccd8b93f5091c9c2b6adb2f9a92.jpg)

![710cb30171fa52c208bc5c81d190dcb9b34458cc66d154ff1e15ad067254c15b.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/710cb30171fa52c208bc5c81d190dcb9b34458cc66d154ff1e15ad067254c15b.jpg)

![766d1d3e88662a5a83acf4383d938747fc390f42561d5fa4718e3617342efda8.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/766d1d3e88662a5a83acf4383d938747fc390f42561d5fa4718e3617342efda8.jpg)

![82f38c69800eb0429f8bf4ebdd23397bf567773331bb086a0f0260f428b28975.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/82f38c69800eb0429f8bf4ebdd23397bf567773331bb086a0f0260f428b28975.jpg)

![9439c4e263c9a760b7261d5ad832dcaa0ba8f36f15abf9b1192f7114309efbfa.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/9439c4e263c9a760b7261d5ad832dcaa0ba8f36f15abf9b1192f7114309efbfa.jpg)

![94d53fae91356a2f8efe802c6598212243be20b9a2eba59b692dbe201501c5f3.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/94d53fae91356a2f8efe802c6598212243be20b9a2eba59b692dbe201501c5f3.jpg)

![96e9414d553ed06b1e02b02f08c67bcf9e0df4ac493a9bc6bb0dfbe2e7ba02c0.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/96e9414d553ed06b1e02b02f08c67bcf9e0df4ac493a9bc6bb0dfbe2e7ba02c0.jpg)

![a6c04adfce9636f2b4058bd0ed24909ca363fc2ae11f6d8bae93bc1ef304a28f.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/a6c04adfce9636f2b4058bd0ed24909ca363fc2ae11f6d8bae93bc1ef304a28f.jpg)

![c9d5e12a246ff29db732faa6b232227e1520402a8e2de368fa3bc6e5973b1c5d.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/c9d5e12a246ff29db732faa6b232227e1520402a8e2de368fa3bc6e5973b1c5d.jpg)

![ed3c891a66a08873d3fcd73999349de9a2e96977d4fdfd96b12a08f0387f69ea.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/ed3c891a66a08873d3fcd73999349de9a2e96977d4fdfd96b12a08f0387f69ea.jpg)

![f0c75865068b8c91155daf9f5a80a73bd7967294f588add78c2d04bd43907c54.jpg](../icml_results/2455_Contrastive%20Representation%20Regularization%20for%20Vision-Language-Action%20Models/tables/f0c75865068b8c91155daf9f5a80a73bd7967294f588add78c2d04bd43907c54.jpg)

<a id="paper-2426"></a>

## Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History


### Images

![042ca635fc8ae9165dd1dabfbabb41dc619e7270b4a0d148b8049292bd38a182.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/images/042ca635fc8ae9165dd1dabfbabb41dc619e7270b4a0d148b8049292bd38a182.jpg)

![34fd927c65732c8ed51dd0f263376e2974235c789d3ce158b0e56a8473e7ec8b.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/images/34fd927c65732c8ed51dd0f263376e2974235c789d3ce158b0e56a8473e7ec8b.jpg)

![699c303359faf453fc3d43bd088e0fc2e513b3a56d33cc00926736a40feb33af.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/images/699c303359faf453fc3d43bd088e0fc2e513b3a56d33cc00926736a40feb33af.jpg)

![6ef7a1f333fb8e7205aa3d81d31bff4307dbb7d79042f98718ba45a801b9d4c2.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/images/6ef7a1f333fb8e7205aa3d81d31bff4307dbb7d79042f98718ba45a801b9d4c2.jpg)

![fcca1c00727585711a0b8f9182b0eb8336a62614fa3eb1446d921681977e9709.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/images/fcca1c00727585711a0b8f9182b0eb8336a62614fa3eb1446d921681977e9709.jpg)

### Tables

![11fbc4b10736148ed08df193959cff2580762603b32c87b9008f5c2605af3e7d.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/11fbc4b10736148ed08df193959cff2580762603b32c87b9008f5c2605af3e7d.jpg)

![3ed9ae20776b0877a7dab9ca81d4879b5210aac4eb293faca98565067d84e8b6.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/3ed9ae20776b0877a7dab9ca81d4879b5210aac4eb293faca98565067d84e8b6.jpg)

![45e37485cffd4427a66425f621123b7ebc718fbfd10351419dd4a95bf938cf6a.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/45e37485cffd4427a66425f621123b7ebc718fbfd10351419dd4a95bf938cf6a.jpg)

![53173cd8470eb47c9715ff8cabe9dae061cf656b29473385fa9348d8c65824b1.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/53173cd8470eb47c9715ff8cabe9dae061cf656b29473385fa9348d8c65824b1.jpg)

![56b918ecdf25de4f05276e210cfdb77bbc1927aaea8f04fcf470fd7141d9900a.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/56b918ecdf25de4f05276e210cfdb77bbc1927aaea8f04fcf470fd7141d9900a.jpg)

![6fe4a6049fa46e6ec7e5ea9b84e33fa70efb2fa2ac1ee8dc5913b38d113bf8dc.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/6fe4a6049fa46e6ec7e5ea9b84e33fa70efb2fa2ac1ee8dc5913b38d113bf8dc.jpg)

![730141d1c096599232cd803279c9cae51ec60cb4e4de57f24dc19d15ac96cd77.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/730141d1c096599232cd803279c9cae51ec60cb4e4de57f24dc19d15ac96cd77.jpg)

![7fb0257c1721cd9a1c0e135d874827aff01de3507d918b68a0511860665e6fd3.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/7fb0257c1721cd9a1c0e135d874827aff01de3507d918b68a0511860665e6fd3.jpg)

![953b2d262fbe929dc2682b8230671b8e8258050db5f06d6298a8e4c5db937af9.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/953b2d262fbe929dc2682b8230671b8e8258050db5f06d6298a8e4c5db937af9.jpg)

![97f5be663a42227946a1b4d9b2b7616ca66ec0b581277192074f5527d680743c.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/97f5be663a42227946a1b4d9b2b7616ca66ec0b581277192074f5527d680743c.jpg)

![b0314aed465c8e7450026ea147221f1a8c99890acf5d7dca45a0cc6cd7ebe0ed.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/b0314aed465c8e7450026ea147221f1a8c99890acf5d7dca45a0cc6cd7ebe0ed.jpg)

![c4da8b23b65778be942fd6c59c275cb181cb560027054384eae275331d3f28ce.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/c4da8b23b65778be942fd6c59c275cb181cb560027054384eae275331d3f28ce.jpg)

![de3fff4827a4f8d12c36ab4da60f981f1dccf06ebb0b58520a0bf9f3848a1b0e.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/de3fff4827a4f8d12c36ab4da60f981f1dccf06ebb0b58520a0bf9f3848a1b0e.jpg)

![e33f0675ec45db911df255323b7a85fb5515246fd8a58123f69e58034db44c9c.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/e33f0675ec45db911df255323b7a85fb5515246fd8a58123f69e58034db44c9c.jpg)

![e4e5b234b57ff335b38d4154885674a315f7694ee020f93ffd1651ea15a58bd9.jpg](../icml_results/2456_Persona2Web_%20Benchmarking%20Personalized%20Web%20Agents%20for%20Contextual%20Reasoning%20with%20User%20History/tables/e4e5b234b57ff335b38d4154885674a315f7694ee020f93ffd1651ea15a58bd9.jpg)

<a id="paper-2427"></a>

## Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences


### Images

![044b4d0ff05e90106e495e664ed9fd9785d28058b0000783fe5d3229c83ba015.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/044b4d0ff05e90106e495e664ed9fd9785d28058b0000783fe5d3229c83ba015.jpg)

![1894ca5b918f60fd187832a613cdf3f08d4ab559be4b8fa8e5b1a9950f94ded9.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/1894ca5b918f60fd187832a613cdf3f08d4ab559be4b8fa8e5b1a9950f94ded9.jpg)

![28d64f900f8057ad5f3c0b15b4c3e887df07b8f40a48115079cd21240a4129a9.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/28d64f900f8057ad5f3c0b15b4c3e887df07b8f40a48115079cd21240a4129a9.jpg)

![3977ef99c2aac028b7dda925255b77f437cbe207ca3d5c85e5f36efc27d6b753.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/3977ef99c2aac028b7dda925255b77f437cbe207ca3d5c85e5f36efc27d6b753.jpg)

![4959b259f51aef0acc979e5af5ba779cc3e0119d846899f784ab5debc12f9593.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/4959b259f51aef0acc979e5af5ba779cc3e0119d846899f784ab5debc12f9593.jpg)

![62cb18e1bac9f444cd406e1e5b9c22d97dcb8f8556a99fddb6a197469a556128.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/62cb18e1bac9f444cd406e1e5b9c22d97dcb8f8556a99fddb6a197469a556128.jpg)

![80cb9a8fe86dce4cd62749d4c1877f54682e0d71d9ddbe384698837b05641c39.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/images/80cb9a8fe86dce4cd62749d4c1877f54682e0d71d9ddbe384698837b05641c39.jpg)

### Tables

![2b3f550642fffb22d91ddcb82f41de6b03ae05a51cb9641f221945d7f995e691.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/2b3f550642fffb22d91ddcb82f41de6b03ae05a51cb9641f221945d7f995e691.jpg)

![38b7580da6f4a21d76e2a52c9058a224a64949f35dbc089f665f14c937542f41.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/38b7580da6f4a21d76e2a52c9058a224a64949f35dbc089f665f14c937542f41.jpg)

![445aaeedb98f472e8397a39cc073b6a44875ffa5127914780f5064cbc57484be.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/445aaeedb98f472e8397a39cc073b6a44875ffa5127914780f5064cbc57484be.jpg)

![50bc3b822dabf32030f8bfdf0846cb806577e7245cbfef78372dcd2a5701f59f.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/50bc3b822dabf32030f8bfdf0846cb806577e7245cbfef78372dcd2a5701f59f.jpg)

![6a9b28b7f5d2493cfafaedcfc2b6792ea74909cdf16652bf62d089fc046c4dc9.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/6a9b28b7f5d2493cfafaedcfc2b6792ea74909cdf16652bf62d089fc046c4dc9.jpg)

![7f770634da24ee81e9710abba0d2ac6ba09ba7f6a0eaabea2b14fdf78e5cfeff.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/7f770634da24ee81e9710abba0d2ac6ba09ba7f6a0eaabea2b14fdf78e5cfeff.jpg)

![86916406414f4f08185b1962d2b3d1e3b240352ced7b4982fb504b4ea99fc473.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/86916406414f4f08185b1962d2b3d1e3b240352ced7b4982fb504b4ea99fc473.jpg)

![d4ef33300982c8913fe914b3a2cd8e8c46f227c174eb8cf369b85de38a061319.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/d4ef33300982c8913fe914b3a2cd8e8c46f227c174eb8cf369b85de38a061319.jpg)

![e36423c06c408344139a2ec94becb578abea773450e2cf227365c4185c511359.jpg](../icml_results/2457_Localizing%20Memorized%20Regions%20in%20Diffusion%20Models%20via%20Coordinate-Wise%20Curvature%20Differences/tables/e36423c06c408344139a2ec94becb578abea773450e2cf227365c4185c511359.jpg)

<a id="paper-2428"></a>

## Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality


### Images

![30ae11e05c057fdef70763bd3a43133edb06594173b905dec4cbd0bd6bdbc841.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/30ae11e05c057fdef70763bd3a43133edb06594173b905dec4cbd0bd6bdbc841.jpg)

![604eb8092e7debf52a18e5860a47433fa42d9aa6017579d8d46de13b0c15a6d2.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/604eb8092e7debf52a18e5860a47433fa42d9aa6017579d8d46de13b0c15a6d2.jpg)

![79cd559b03a53fbb01d0c5a3b4e5f3927edf78552d7b475604e0a7f0e7596b8a.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/79cd559b03a53fbb01d0c5a3b4e5f3927edf78552d7b475604e0a7f0e7596b8a.jpg)

![bfeac6242d0d9a0396bcc5eac75d760fe6669952a098b8675814681a86b8799a.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/bfeac6242d0d9a0396bcc5eac75d760fe6669952a098b8675814681a86b8799a.jpg)

![d827fa24bbce6833f73400d1682f96f2e50982543fa99bd1b5003e8b9dffdc20.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/d827fa24bbce6833f73400d1682f96f2e50982543fa99bd1b5003e8b9dffdc20.jpg)

![dd476bdce37751f2e8db906573f984b2faa5be0b5ad736c40156bdef5966af7a.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/dd476bdce37751f2e8db906573f984b2faa5be0b5ad736c40156bdef5966af7a.jpg)

![ddea1ed8f6363597ed8db3aa6a23b69de81c201fcdd71ee69ae9491e597977b7.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/ddea1ed8f6363597ed8db3aa6a23b69de81c201fcdd71ee69ae9491e597977b7.jpg)

![efc1eb16a87a36d33c76815731e228d9dd097f439933d0afe311566105f218a8.jpg](../icml_results/2458_Follow-the-Perturbed-Leader%20for%20Decoupled%20Bandits_%20Best-of-Both-Worlds%20and%20Practicality/images/efc1eb16a87a36d33c76815731e228d9dd097f439933d0afe311566105f218a8.jpg)

<a id="paper-2429"></a>

## Adversarial Robustness of Implicit Neural Representation-Based Classifiers


### Images

![a7db75961305464c09276823cd88da80e37fa5193bfc3c74e89eb06908633a34.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/images/a7db75961305464c09276823cd88da80e37fa5193bfc3c74e89eb06908633a34.jpg)

![bdfde5f40db8c517c37aadfd992f748cee9d7b3a9d8f2584ab14a39f56279f2b.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/images/bdfde5f40db8c517c37aadfd992f748cee9d7b3a9d8f2584ab14a39f56279f2b.jpg)

![c274f68e1bc342115cac433548f39843d4255ff282609bdb494033515350c83e.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/images/c274f68e1bc342115cac433548f39843d4255ff282609bdb494033515350c83e.jpg)

![ceea41c5845c4fb62913020d11c7b696347a172404d38662d0709cd048409143.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/images/ceea41c5845c4fb62913020d11c7b696347a172404d38662d0709cd048409143.jpg)

### Tables

![164944782c935d8ef01a0d02bacf212c606dfb9edf6b961c79108786d8e07fb0.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/164944782c935d8ef01a0d02bacf212c606dfb9edf6b961c79108786d8e07fb0.jpg)

![171ad6009563a1c3329d9a80fac6f5a352034ca1749107a4f16211267c93ec4e.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/171ad6009563a1c3329d9a80fac6f5a352034ca1749107a4f16211267c93ec4e.jpg)

![1af901643eb40f50323fb8032d55d1e86e968e75853b1bf49c29092d1481c762.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/1af901643eb40f50323fb8032d55d1e86e968e75853b1bf49c29092d1481c762.jpg)

![2d7c66205cdb1f19fbbbf50a61d05216a6da9769a08cff38a81c93fd1d823237.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/2d7c66205cdb1f19fbbbf50a61d05216a6da9769a08cff38a81c93fd1d823237.jpg)

![2eb240dc1dc9b833349ee8e2f2269e648a20f045c62d7e63bfdb39718f1f65a3.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/2eb240dc1dc9b833349ee8e2f2269e648a20f045c62d7e63bfdb39718f1f65a3.jpg)

![39997111721cffe88f6a974acad7953fd9212d71ce6d8f68181ee498419cc502.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/39997111721cffe88f6a974acad7953fd9212d71ce6d8f68181ee498419cc502.jpg)

![44b4fd9ce0ce337bdb6a85bcb3a4ac84b049f18da7f6f3845ebe0e68dc9f3133.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/44b4fd9ce0ce337bdb6a85bcb3a4ac84b049f18da7f6f3845ebe0e68dc9f3133.jpg)

![552c20b4cdd42708e3b3c0ef8c939bca2f0025158b32c1b921c5c158deb80b27.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/552c20b4cdd42708e3b3c0ef8c939bca2f0025158b32c1b921c5c158deb80b27.jpg)

![56f35a1e85ae5c77ef6c9a94b64a0e73f7ac56cf094d5aa588eeee36e7729628.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/56f35a1e85ae5c77ef6c9a94b64a0e73f7ac56cf094d5aa588eeee36e7729628.jpg)

![5f550918d39e99113fd52bec1d884f4afba013995e0135d3f39c7684b4fbccab.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/5f550918d39e99113fd52bec1d884f4afba013995e0135d3f39c7684b4fbccab.jpg)

![69f02ea0543947c3adafdc30969ad3cced7f412595f315358c98b95d3c1a1ef4.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/69f02ea0543947c3adafdc30969ad3cced7f412595f315358c98b95d3c1a1ef4.jpg)

![6a387392e717be4a9a2219f3fd5680b1c51f0158a02b52a18482c9b49674b215.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/6a387392e717be4a9a2219f3fd5680b1c51f0158a02b52a18482c9b49674b215.jpg)

![6c6a5bf9872c22898a5f18d0c94094b78ef760bde8013f52223d75de3655c25f.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/6c6a5bf9872c22898a5f18d0c94094b78ef760bde8013f52223d75de3655c25f.jpg)

![6cc67fb415dd040e634f92e949fe866d7fe7b594653ca9f359bec96970cf026c.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/6cc67fb415dd040e634f92e949fe866d7fe7b594653ca9f359bec96970cf026c.jpg)

![6ce6cfcb5903e7fcfe08499faf0c4b23714da432275c823717b7243670f61786.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/6ce6cfcb5903e7fcfe08499faf0c4b23714da432275c823717b7243670f61786.jpg)

![793cb3b08027b0a9fd91c29ed4668957081cdef5a5db5d406ba92cf46e92c4eb.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/793cb3b08027b0a9fd91c29ed4668957081cdef5a5db5d406ba92cf46e92c4eb.jpg)

![8516bfb665f95e842c7aa70c909c51cb07d5128c7a5772791ae63413f9ef59fa.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/8516bfb665f95e842c7aa70c909c51cb07d5128c7a5772791ae63413f9ef59fa.jpg)

![94eeb1d1dc54152759fd4376e21bfe9ea1cb2fc62f91844ba7d52f33fa891f25.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/94eeb1d1dc54152759fd4376e21bfe9ea1cb2fc62f91844ba7d52f33fa891f25.jpg)

![a425eead655980cedb99e845aa1b81690c14c0af4824c24177e5584f2c688c09.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/a425eead655980cedb99e845aa1b81690c14c0af4824c24177e5584f2c688c09.jpg)

![b9188f83c82a2e6c76da36d47f86e67a366ca80870db44c93fab8104ed5c235a.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/b9188f83c82a2e6c76da36d47f86e67a366ca80870db44c93fab8104ed5c235a.jpg)

![ba025c4d28eab4a8393accf8fafe14904ec8f10cfdd8e5fac148b8799a90f51b.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/ba025c4d28eab4a8393accf8fafe14904ec8f10cfdd8e5fac148b8799a90f51b.jpg)

![c28dbebd119d1196023929d0fa22a169b5030a08044c4c7b9461a4a76f6c3e6d.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/c28dbebd119d1196023929d0fa22a169b5030a08044c4c7b9461a4a76f6c3e6d.jpg)

![c8208838a9827fc7035193d281408d877a1fe029be4484592d6a11392a6e5b3d.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/c8208838a9827fc7035193d281408d877a1fe029be4484592d6a11392a6e5b3d.jpg)

![cf8b51318d97970eabd79e014fd37910430bfe801ea9b853c493fcb095338811.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/cf8b51318d97970eabd79e014fd37910430bfe801ea9b853c493fcb095338811.jpg)

![d4e3b584c4fe2fda46a5d08918dd12a7d2387f8e48da5c78cbc4570bbc5f18ca.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/d4e3b584c4fe2fda46a5d08918dd12a7d2387f8e48da5c78cbc4570bbc5f18ca.jpg)

![d66b1a560548776a80da1e85fb1f76781819cf914b451a7e906d2eaccb710f43.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/d66b1a560548776a80da1e85fb1f76781819cf914b451a7e906d2eaccb710f43.jpg)

![dca3a42f39644a0a53c7cb37d8d69fffc8223f5e5b49aa541d78b0502f62b20f.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/dca3a42f39644a0a53c7cb37d8d69fffc8223f5e5b49aa541d78b0502f62b20f.jpg)

![dcabfc91d12c6ea7e2fc947c99fcbeadd0564efebf7e203732c0857845327ecd.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/dcabfc91d12c6ea7e2fc947c99fcbeadd0564efebf7e203732c0857845327ecd.jpg)

![e0c8b450fdee1643194d26828faa435d962e9b59859107bf748a14f68a412fc2.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/e0c8b450fdee1643194d26828faa435d962e9b59859107bf748a14f68a412fc2.jpg)

![e6c8bc44e02e6582f9027bbfaf67a37d96a63948e1e867d758ae97fe7d19e16f.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/e6c8bc44e02e6582f9027bbfaf67a37d96a63948e1e867d758ae97fe7d19e16f.jpg)

![ea46c8ceed57acd89cb81faffec5c749f6a9f848df0d6e7fbd5efed9e1f765f5.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/ea46c8ceed57acd89cb81faffec5c749f6a9f848df0d6e7fbd5efed9e1f765f5.jpg)

![ef48d73bed7409c8c9b4992e54bfddd8525c99d782a178a13320e33c4d9e448a.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/ef48d73bed7409c8c9b4992e54bfddd8525c99d782a178a13320e33c4d9e448a.jpg)

![f004819d885e2271c386e158abb5b4cdcbe1b43267078f7ad030d662e25cc5d8.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/f004819d885e2271c386e158abb5b4cdcbe1b43267078f7ad030d662e25cc5d8.jpg)

![f6f75b5691e137d647a13920e8fd0996a13f959b932f179f8199af84e927d4b9.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/f6f75b5691e137d647a13920e8fd0996a13f959b932f179f8199af84e927d4b9.jpg)

![f7f2a241992b4eff7dfe95b0d9f723c239e6d7764a56f280f9c061bea9820263.jpg](../icml_results/2459_Adversarial%20Robustness%20of%20Implicit%20Neural%20Representation-Based%20Classifiers/tables/f7f2a241992b4eff7dfe95b0d9f723c239e6d7764a56f280f9c061bea9820263.jpg)

<a id="paper-2430"></a>

## BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback


### Images

![83b3b23ec916a0d72a3f0d925efbf1bdf3f42b623ed7402dabfe663f2e45cbb2.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/images/83b3b23ec916a0d72a3f0d925efbf1bdf3f42b623ed7402dabfe663f2e45cbb2.jpg)

![a01fe39482eb6478ef2ccc7d88be18c743f67ebfceb8530808c1be591fa98f42.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/images/a01fe39482eb6478ef2ccc7d88be18c743f67ebfceb8530808c1be591fa98f42.jpg)

![c91314fdbb7a8c116ebadc2e7298d341d1fc37e577e021ec105f2a2d9f7db74f.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/images/c91314fdbb7a8c116ebadc2e7298d341d1fc37e577e021ec105f2a2d9f7db74f.jpg)

### Tables

![040881029bd1de6e118dec30e43372f5ceb93b5e1d8081ddd7f6600e8313187d.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/040881029bd1de6e118dec30e43372f5ceb93b5e1d8081ddd7f6600e8313187d.jpg)

![075d2196a10b51b5ca49a9a6b6608e7fb62bee23902e10c345598d7d94300d3b.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/075d2196a10b51b5ca49a9a6b6608e7fb62bee23902e10c345598d7d94300d3b.jpg)

![09f7dec52e65986fb0f288d5b844278888626bfd2dda1c2516434d20ebf1ff9a.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/09f7dec52e65986fb0f288d5b844278888626bfd2dda1c2516434d20ebf1ff9a.jpg)

![103099a81008b113d4b3a9dae176a62fa67ee1db8fa0f1f876e1178d2f15ebea.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/103099a81008b113d4b3a9dae176a62fa67ee1db8fa0f1f876e1178d2f15ebea.jpg)

![1d5dce1c4f87b78fab0fceec434cd9fa40d1ebeb96856229cff31b39136ee552.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/1d5dce1c4f87b78fab0fceec434cd9fa40d1ebeb96856229cff31b39136ee552.jpg)

![399a83dfe4e00fcef2121dc7c3ed26a7aa07429f4c08159bb3beae2d197510e1.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/399a83dfe4e00fcef2121dc7c3ed26a7aa07429f4c08159bb3beae2d197510e1.jpg)

![3e6959d11d40d50e510b97d9b076e35b1de43f872581752c92860724bde098dd.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/3e6959d11d40d50e510b97d9b076e35b1de43f872581752c92860724bde098dd.jpg)

![441a2419e01252707c21a05fded61d8a896a873bc495470aca05662f1ae46d00.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/441a2419e01252707c21a05fded61d8a896a873bc495470aca05662f1ae46d00.jpg)

![4da6274b04d82bf39673ce029bba5eafdb8a5da8b227a494a4b9addb89b7bd58.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/4da6274b04d82bf39673ce029bba5eafdb8a5da8b227a494a4b9addb89b7bd58.jpg)

![4db6e734a6b0562821f61f64d873f0aac24b39826695a79a942358865670f059.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/4db6e734a6b0562821f61f64d873f0aac24b39826695a79a942358865670f059.jpg)

![6efa29451de191772a6390d7ba1c39a729d86cd9c83a3f3f18b216b64b2daa46.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/6efa29451de191772a6390d7ba1c39a729d86cd9c83a3f3f18b216b64b2daa46.jpg)

![88249d2cafd2f2470f1dada461ea50930578d1ee963d86b47395d56b8c5dbdd4.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/88249d2cafd2f2470f1dada461ea50930578d1ee963d86b47395d56b8c5dbdd4.jpg)

![8acd55b1809bcf5ee94792f4830d95c0159295f66897ade7587422f30f0ed163.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/8acd55b1809bcf5ee94792f4830d95c0159295f66897ade7587422f30f0ed163.jpg)

![9bd1e8e44fb8aa79e53284dcb32cf1101f636da94dfebd29d72d66b12b2b6689.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/9bd1e8e44fb8aa79e53284dcb32cf1101f636da94dfebd29d72d66b12b2b6689.jpg)

![c1835a8ddffdc5deef924d64d8728f00c16027c15c22bf99d97d175f57e69fa7.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/c1835a8ddffdc5deef924d64d8728f00c16027c15c22bf99d97d175f57e69fa7.jpg)

![c7335c22f18e3370f1ffe438998a2664576a8cd04558c3e8cff4927c7b7cc574.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/c7335c22f18e3370f1ffe438998a2664576a8cd04558c3e8cff4927c7b7cc574.jpg)

![cfd44699ff8cd7956fb244a6053355d734b79c8c088148546827e787d2aa1482.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/cfd44699ff8cd7956fb244a6053355d734b79c8c088148546827e787d2aa1482.jpg)

![e14274f4b7e3834fca56700c3faf8cdbb61c9883f02a0e49635e63d1a031541a.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/e14274f4b7e3834fca56700c3faf8cdbb61c9883f02a0e49635e63d1a031541a.jpg)

![ee92b9fca7fbab33bfaf72df33f9be0cb2c6f8b0e6657991abe5715be06f2125.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/ee92b9fca7fbab33bfaf72df33f9be0cb2c6f8b0e6657991abe5715be06f2125.jpg)

![fda3069a451265c8a6494de4e23e6ecc3f43e652c18e2f8f268380f1f0b5e722.jpg](../icml_results/2460_BESPOKE_%20Benchmark%20for%20Search-Augmented%20Large%20Language%20Model%20Personalization%20via%20Diagnostic%20Feedback/tables/fda3069a451265c8a6494de4e23e6ecc3f43e652c18e2f8f268380f1f0b5e722.jpg)

<a id="paper-2431"></a>

## Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor


### Images

![502c5b074401ae84eb7c7d9e984d0cedd472f8edb2d8ee7fbaca114fa524fc94.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/images/502c5b074401ae84eb7c7d9e984d0cedd472f8edb2d8ee7fbaca114fa524fc94.jpg)

![654dbe97f559da4122a32d28e75c8cf15b79b75fe22bc4c1cc7f868b45d16708.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/images/654dbe97f559da4122a32d28e75c8cf15b79b75fe22bc4c1cc7f868b45d16708.jpg)

![c7f2e29fd249e41a05a76439d7e992ded5dc867d3770f98fccae1430445ddc4a.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/images/c7f2e29fd249e41a05a76439d7e992ded5dc867d3770f98fccae1430445ddc4a.jpg)

![db6f011a4a2d58bd5936d2b8144377fe60a5badf9a9b2f2283d41d704a9f6edd.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/images/db6f011a4a2d58bd5936d2b8144377fe60a5badf9a9b2f2283d41d704a9f6edd.jpg)

### Tables

![062cd33b291c15706eb2f0de277cd6810c48409c95a2ea04472b1f872fd4938d.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/062cd33b291c15706eb2f0de277cd6810c48409c95a2ea04472b1f872fd4938d.jpg)

![21deead6acb72f58c344e270c256eabcbb733b75b45d0dbae83901cedcf4b2bf.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/21deead6acb72f58c344e270c256eabcbb733b75b45d0dbae83901cedcf4b2bf.jpg)

![2d2e335d40ab3b26f18bdde702aba5b4d96a4bbfc80206ce3d951099ef0751bc.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/2d2e335d40ab3b26f18bdde702aba5b4d96a4bbfc80206ce3d951099ef0751bc.jpg)

![32237c5a17bcf1eb5632cbe6c63ffe0bb7fb1143db5a86d51a5eff24dd0f4df1.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/32237c5a17bcf1eb5632cbe6c63ffe0bb7fb1143db5a86d51a5eff24dd0f4df1.jpg)

![47c0ee01b0e6bacad1c0278ae733ed66bdeedad682353cd65f0e60a3fc0006e3.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/47c0ee01b0e6bacad1c0278ae733ed66bdeedad682353cd65f0e60a3fc0006e3.jpg)

![5cce084722242fffd2d4137be8cf5a75f3e60cd978e7217bb17adf57d7195ca6.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/5cce084722242fffd2d4137be8cf5a75f3e60cd978e7217bb17adf57d7195ca6.jpg)

![5fd2962032d04e966b180c45f528958425a0729998c846b52b4fd2c290774b35.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/5fd2962032d04e966b180c45f528958425a0729998c846b52b4fd2c290774b35.jpg)

![69bcda087f6a63574e130f8b6da2f28d96c0421205e3486b4db9b416b96952ca.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/69bcda087f6a63574e130f8b6da2f28d96c0421205e3486b4db9b416b96952ca.jpg)

![9f9b4f4a926633615e0102c4e73fc23c4ddca631b9b67329de7b58695310486a.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/9f9b4f4a926633615e0102c4e73fc23c4ddca631b9b67329de7b58695310486a.jpg)

![a08c39c50fce84ec4c9da37bb49cb90d584517718119b223d5b1bb1771e82d78.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/a08c39c50fce84ec4c9da37bb49cb90d584517718119b223d5b1bb1771e82d78.jpg)

![aa671ddc605f51b73dbceb21287ee623b4e4a5215d00945e051746742e3b62d8.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/aa671ddc605f51b73dbceb21287ee623b4e4a5215d00945e051746742e3b62d8.jpg)

![bcac27cd74681bec5508cfaca56aed1cf6c2f1df8896cb5079e5679ea0477dfe.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/bcac27cd74681bec5508cfaca56aed1cf6c2f1df8896cb5079e5679ea0477dfe.jpg)

![ca965d211757e4b80ae7d6ce1f8c3d368fe825ad1c5cedaee3146b1fef87568b.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/ca965d211757e4b80ae7d6ce1f8c3d368fe825ad1c5cedaee3146b1fef87568b.jpg)

![e02dc56c9d179dc8475d35802a5fd225ecbe754d2227467791d3e29dfb535601.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/e02dc56c9d179dc8475d35802a5fd225ecbe754d2227467791d3e29dfb535601.jpg)

![eeb7ad2f802fba71f3a3b3462f8f76a9836cdab8399f99ea618d2fbf73b6bff1.jpg](../icml_results/2461_Teaching%20Molecular%20Dynamics%20to%20a%20Non-Autoregressive%20Ionic%20Transport%20Predictor/tables/eeb7ad2f802fba71f3a3b3462f8f76a9836cdab8399f99ea618d2fbf73b6bff1.jpg)

<a id="paper-2432"></a>

## DiscoverLLM: From Executing Intents to Discovering Them


### Images

![1bbd3a4fc6668235178b1b37951b85c0366ba8aa0fb3202b431f52e82e601199.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/1bbd3a4fc6668235178b1b37951b85c0366ba8aa0fb3202b431f52e82e601199.jpg)

![1c259608a2f373032d955a42c081b76c1c95e8e35e6794a554f344eb1a23c5db.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/1c259608a2f373032d955a42c081b76c1c95e8e35e6794a554f344eb1a23c5db.jpg)

![26237f9feda4041d8bc8e02ec76de7ef419fb698a1c3548cfa6b141303c2afe9.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/26237f9feda4041d8bc8e02ec76de7ef419fb698a1c3548cfa6b141303c2afe9.jpg)

![28c4091ec8bc480ea15a85b0763b7b250269182a63bd4014e1b88dfda381b0bf.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/28c4091ec8bc480ea15a85b0763b7b250269182a63bd4014e1b88dfda381b0bf.jpg)

![3eebfc53ef87fec43cbf2556f40df5a02567757f724b367b96113f5d88f55792.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/3eebfc53ef87fec43cbf2556f40df5a02567757f724b367b96113f5d88f55792.jpg)

![bf282a2a904e047394b01432e047318c1ad763b954cb6c627a4bd3bdec888b97.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/bf282a2a904e047394b01432e047318c1ad763b954cb6c627a4bd3bdec888b97.jpg)

![eae74d75e8fdc279b911019839ee54997d42c6b8727af574a1615881bc62c652.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/eae74d75e8fdc279b911019839ee54997d42c6b8727af574a1615881bc62c652.jpg)

![ec6e72958c33297ee34e37c3fad242f923b253c2167adf1b7ee02b448be49c65.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/images/ec6e72958c33297ee34e37c3fad242f923b253c2167adf1b7ee02b448be49c65.jpg)

### Tables

![175a45ddb7de0ac9ec67e9c764f9f725ba56bee924073f4c8222234783471b20.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/175a45ddb7de0ac9ec67e9c764f9f725ba56bee924073f4c8222234783471b20.jpg)

![3b6992b85885252054f7ae6c320031dcdb4c7e42805b4d4d6e0e2587b3b01a45.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/3b6992b85885252054f7ae6c320031dcdb4c7e42805b4d4d6e0e2587b3b01a45.jpg)

![3f10ae39334ddf0d0820e0914cb936154b4e296d81dce9a19abe359f491fc0a7.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/3f10ae39334ddf0d0820e0914cb936154b4e296d81dce9a19abe359f491fc0a7.jpg)

![5ae60058a229fb3e518e66045261088f6d5178686b5c75846014c0e234cc2015.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/5ae60058a229fb3e518e66045261088f6d5178686b5c75846014c0e234cc2015.jpg)

![7022f26880c0e08342735ec7c51f4ee7a371ec2be887ce058a73bef8083fe051.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/7022f26880c0e08342735ec7c51f4ee7a371ec2be887ce058a73bef8083fe051.jpg)

![93e72c1abe298a2526c3edc09e6bb44a263437f2eec629bbc8d8c25c41b249eb.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/93e72c1abe298a2526c3edc09e6bb44a263437f2eec629bbc8d8c25c41b249eb.jpg)

![a2294f75add20ce1ae5ab8a0e28bbf5ecf23d7cbe352fc0233100cf74dd2e3f2.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/a2294f75add20ce1ae5ab8a0e28bbf5ecf23d7cbe352fc0233100cf74dd2e3f2.jpg)

![a727be4065b54bfffe1ce92f6331f3d8c7e8abf2d09a76071de8e34462db7c99.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/a727be4065b54bfffe1ce92f6331f3d8c7e8abf2d09a76071de8e34462db7c99.jpg)

![ac5546accb34ea9a3c60abf018a1471bc98ea278eef36610a63ab10840542612.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/ac5546accb34ea9a3c60abf018a1471bc98ea278eef36610a63ab10840542612.jpg)

![ac61b3af4c7573658af70525266e05d5fc5edb4e5757b1ce134206166ff7565e.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/ac61b3af4c7573658af70525266e05d5fc5edb4e5757b1ce134206166ff7565e.jpg)

![b75af63c261b56698813dcd1e28b039ec0b5703db74f33e49ff8670456e572d6.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/b75af63c261b56698813dcd1e28b039ec0b5703db74f33e49ff8670456e572d6.jpg)

![c8bd2d039bef3b8b00895dd34234f2dd12874592c07c793b4935952a7df4c272.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/c8bd2d039bef3b8b00895dd34234f2dd12874592c07c793b4935952a7df4c272.jpg)

![e3b5cf853615854d9382f45cd1916b841ad8353c26f4652aa93176091db59e38.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/e3b5cf853615854d9382f45cd1916b841ad8353c26f4652aa93176091db59e38.jpg)

![ec66636bb4cb44f9be627eb140c76cc4be3de26f9cfde24df692844d6e14818c.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/ec66636bb4cb44f9be627eb140c76cc4be3de26f9cfde24df692844d6e14818c.jpg)

![ec71115ff59865aa72b1364033bef72b12a705259fc0af3f1848e71a90f6450d.jpg](../icml_results/2462_DiscoverLLM_%20From%20Executing%20Intents%20to%20Discovering%20Them/tables/ec71115ff59865aa72b1364033bef72b12a705259fc0af3f1848e71a90f6450d.jpg)

<a id="paper-2433"></a>

## Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution


### Images

![00d88aa3e7232250a5c30534ffbfec67ba23699f2dd862c6c514c1ab93203f77.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/00d88aa3e7232250a5c30534ffbfec67ba23699f2dd862c6c514c1ab93203f77.jpg)

![157881f0c38e9bf739bc830f45f4eee6cb77e6eeeadc6f28fbc6ee0440056f5a.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/157881f0c38e9bf739bc830f45f4eee6cb77e6eeeadc6f28fbc6ee0440056f5a.jpg)

![47ff556d47abeabfea7c8c76d48ad8e70824a6906fb0cc05841fdc5034a7fe93.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/47ff556d47abeabfea7c8c76d48ad8e70824a6906fb0cc05841fdc5034a7fe93.jpg)

![4b5b2a0758f3a1f4be37284a0ee25af3a214a7046a396646e160867cc2fbb9bc.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/4b5b2a0758f3a1f4be37284a0ee25af3a214a7046a396646e160867cc2fbb9bc.jpg)

![5bc4fe4cdbbb02a54251e1a2f63305f5a743b4985e8e8613cababe2d2fb363e5.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/5bc4fe4cdbbb02a54251e1a2f63305f5a743b4985e8e8613cababe2d2fb363e5.jpg)

![71b8e921778cf3a8ed0bf19297d018cc91947dc176e2a68d6e4c71d83299da4a.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/71b8e921778cf3a8ed0bf19297d018cc91947dc176e2a68d6e4c71d83299da4a.jpg)

![84c6deb4c61c0a4fac8c2169865db0b8a844b2cafa7fe6f70c968ac8d901b2b5.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/84c6deb4c61c0a4fac8c2169865db0b8a844b2cafa7fe6f70c968ac8d901b2b5.jpg)

![9a0f872c276979a91f67c2c38b48d0f3a43774ff2683430bccaab78f9f7d49b1.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/9a0f872c276979a91f67c2c38b48d0f3a43774ff2683430bccaab78f9f7d49b1.jpg)

![9e69d7c53e6623566d44f6070ba3017b857835244d3641cca8f949fd42c8ab77.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/9e69d7c53e6623566d44f6070ba3017b857835244d3641cca8f949fd42c8ab77.jpg)

![c986c990d111c5eebbea6a22e368ad0460e244efa560b98dda68c81cf5391b80.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/c986c990d111c5eebbea6a22e368ad0460e244efa560b98dda68c81cf5391b80.jpg)

![d106d5e7edb4b309d2e8db4890c0488b9a422ad79b4818ec2301e4a037a7dd00.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/d106d5e7edb4b309d2e8db4890c0488b9a422ad79b4818ec2301e4a037a7dd00.jpg)

![d5ceb584a8bd5726cf500415d2ec50e963922030fd5145c3d55653be503127ad.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/d5ceb584a8bd5726cf500415d2ec50e963922030fd5145c3d55653be503127ad.jpg)

![ec710a66c44928c64aa99567cd13478a0e7dbe60493ed7ac8fb85e02ae8da322.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/images/ec710a66c44928c64aa99567cd13478a0e7dbe60493ed7ac8fb85e02ae8da322.jpg)

### Tables

![0f0752fc318d5ba8355df4794f2af633a8c52061ba1d43bda5fda5ab42f26ac2.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/0f0752fc318d5ba8355df4794f2af633a8c52061ba1d43bda5fda5ab42f26ac2.jpg)

![212a3380cdd2084ba3afb53e48a754983231679a8ddd4b9f11ffe0da649e29b5.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/212a3380cdd2084ba3afb53e48a754983231679a8ddd4b9f11ffe0da649e29b5.jpg)

![2c97368fa29c02da35735a47e90b7eb50ea1b22f0ec5767d71d6b8f07039f61e.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/2c97368fa29c02da35735a47e90b7eb50ea1b22f0ec5767d71d6b8f07039f61e.jpg)

![2e0c0e7f92025cc6ef03213d97640aeffa6f234d96d8a217ffcc71d63c495a9f.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/2e0c0e7f92025cc6ef03213d97640aeffa6f234d96d8a217ffcc71d63c495a9f.jpg)

![6d4207389160fb2359d053687c7f95949012d9d556ac7dfdb1223fd757b098f2.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/6d4207389160fb2359d053687c7f95949012d9d556ac7dfdb1223fd757b098f2.jpg)

![82c5bef4678907135388674064ee6242f89c77bb87ab4b3fbc16058991777121.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/82c5bef4678907135388674064ee6242f89c77bb87ab4b3fbc16058991777121.jpg)

![8534355f5850cf7a086aeec71b4954e35fd2d37ebb2c791b850c8e44ef5b2899.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/8534355f5850cf7a086aeec71b4954e35fd2d37ebb2c791b850c8e44ef5b2899.jpg)

![a20d71de54e8979bb424bcb7793e770332323a5cea69575e778fe94b72ed746b.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/a20d71de54e8979bb424bcb7793e770332323a5cea69575e778fe94b72ed746b.jpg)

![d54fb9cfbc51c917b76afa1124c392f717a76718ac0befdf9f48d4f0962355c5.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/d54fb9cfbc51c917b76afa1124c392f717a76718ac0befdf9f48d4f0962355c5.jpg)

![d708c679f3825a47ef012d33b2d8ea932a6bb71038bf940a24785305460afecf.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/d708c679f3825a47ef012d33b2d8ea932a6bb71038bf940a24785305460afecf.jpg)

![d8f3d1e1395c10a160321c2904690a8b8c49cae0e36d099f8cbf2529d1b6f6e1.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/d8f3d1e1395c10a160321c2904690a8b8c49cae0e36d099f8cbf2529d1b6f6e1.jpg)

![f61bb4ebba04c3cfe8f13296f328ffbe9d8838da9e426337159fc09c755a8e1b.jpg](../icml_results/2463_Manifold-Aligned%20Guided%20Integrated%20Gradients%20for%20Reliable%20Feature%20Attribution/tables/f61bb4ebba04c3cfe8f13296f328ffbe9d8838da9e426337159fc09c755a8e1b.jpg)

<a id="paper-2434"></a>

## MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks


### Images

![2987a1e4ee996bf3477f7aa246c77edb6abd3f6ea5157e62775f6f0c6c68820d.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/images/2987a1e4ee996bf3477f7aa246c77edb6abd3f6ea5157e62775f6f0c6c68820d.jpg)

### Tables

![00cbf456c3c34663e692f6fed263557a62cb22d4066c9e15d6b9a61b58896044.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/00cbf456c3c34663e692f6fed263557a62cb22d4066c9e15d6b9a61b58896044.jpg)

![1e7f67e148a49ac5b482a833fce8cd0834d1e19a840890bcb07cd7b84dc6d207.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/1e7f67e148a49ac5b482a833fce8cd0834d1e19a840890bcb07cd7b84dc6d207.jpg)

![1f88bcb57db3e5656648ad22261aa045ae42559d0d911eba6e459ddb775be9fb.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/1f88bcb57db3e5656648ad22261aa045ae42559d0d911eba6e459ddb775be9fb.jpg)

![1fc3aece281e7687edec5a815959053e236e99bfc4c0199416f60d80557bb47b.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/1fc3aece281e7687edec5a815959053e236e99bfc4c0199416f60d80557bb47b.jpg)

![3ea8b69f734c9a727d7ee4b6de0daf8910d5b0ee500e2899d704c8566d06c55a.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/3ea8b69f734c9a727d7ee4b6de0daf8910d5b0ee500e2899d704c8566d06c55a.jpg)

![50bd8db15e6ce943dd4d49d0d68b5301737aeea8e296c805ffcd0738f2238c32.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/50bd8db15e6ce943dd4d49d0d68b5301737aeea8e296c805ffcd0738f2238c32.jpg)

![526a4dca6fe8b56d195359bbea296a90cc4635d3cf140a3f79958e44fe2039c2.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/526a4dca6fe8b56d195359bbea296a90cc4635d3cf140a3f79958e44fe2039c2.jpg)

![7b605956164110056c4a3e90da5e8ed57a3934c3a19c51fb92e1dfa520f40ea6.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/7b605956164110056c4a3e90da5e8ed57a3934c3a19c51fb92e1dfa520f40ea6.jpg)

![949883ee5e855c3c2c061c2417304cb1b0b220e9212d549bda81f1bf608535d7.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/949883ee5e855c3c2c061c2417304cb1b0b220e9212d549bda81f1bf608535d7.jpg)

![bf7b1a8effee8aa664a226f922f8335426d1cde90eb0caf35456cf0faf1e9150.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/bf7b1a8effee8aa664a226f922f8335426d1cde90eb0caf35456cf0faf1e9150.jpg)

![e0b8848c4010c964a2284c184076060a7a13d3f4fcc633dfd9bc3542df69f644.jpg](../icml_results/2464_MoCo-EA_%20Exploiting%20Adversarial%20Mode%20Connectivity%20for%20Efficient%20Evolutionary%20Attacks/tables/e0b8848c4010c964a2284c184076060a7a13d3f4fcc633dfd9bc3542df69f644.jpg)

<a id="paper-2435"></a>

## NNiT: Width-Agnostic Neural Network Generation with Structurally Aligned Weight Spaces


### Images

![0c8b1d0d086cc24bb26ddfc892b802874e1362a3c5304114e5e431b1dd21e6aa.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/0c8b1d0d086cc24bb26ddfc892b802874e1362a3c5304114e5e431b1dd21e6aa.jpg)

![2290f43f88b6a18ddd9fe1ed3ce69110dd3a837122382069f350f3cc77d014ae.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/2290f43f88b6a18ddd9fe1ed3ce69110dd3a837122382069f350f3cc77d014ae.jpg)

![2ad7529b612e5a81f3e360cd07826bf7ae26aa8109d457a824cc7fb741daec8b.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/2ad7529b612e5a81f3e360cd07826bf7ae26aa8109d457a824cc7fb741daec8b.jpg)

![783909fa6a173dea3e460c02390c1a03355506663334e75399d33e2d7279a3d1.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/783909fa6a173dea3e460c02390c1a03355506663334e75399d33e2d7279a3d1.jpg)

![85dd537476b58b9c0ccf2a08fd79aabb3d42b26aa1edfe6ed441d90475d2dc3f.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/85dd537476b58b9c0ccf2a08fd79aabb3d42b26aa1edfe6ed441d90475d2dc3f.jpg)

![b873372e92b5afb67cf328b52d17790513129a121a8c51ee409da42faab336ad.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/b873372e92b5afb67cf328b52d17790513129a121a8c51ee409da42faab336ad.jpg)

![ca3e609b6263325a7e53e9920aafc29d0c5532350f467bdb8b8b2bf5c7f94d78.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/ca3e609b6263325a7e53e9920aafc29d0c5532350f467bdb8b8b2bf5c7f94d78.jpg)

![d69689fb9cd6bed522ea1b395b79277648d45a6b4a2f72b86bcf1df0391107dc.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/d69689fb9cd6bed522ea1b395b79277648d45a6b4a2f72b86bcf1df0391107dc.jpg)

![e90d0bf8dea2cbefd304fdda05768a56e64acbbeef236b4b65e28f649e492d69.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/images/e90d0bf8dea2cbefd304fdda05768a56e64acbbeef236b4b65e28f649e492d69.jpg)

### Tables

![109cbb690a0a4da7f4b29727172c35b68a8ea1ef20e11e5a25b7a0b6a38cf45a.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/109cbb690a0a4da7f4b29727172c35b68a8ea1ef20e11e5a25b7a0b6a38cf45a.jpg)

![3e12071345fd2b0f139ab8b15c2bbc3bf042defa54ef48fc83a641fa6435f720.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/3e12071345fd2b0f139ab8b15c2bbc3bf042defa54ef48fc83a641fa6435f720.jpg)

![46b47f21dd3cbac4ff7fb7a38169fd201eb8e63ae7f3e72a7d37be447fd80723.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/46b47f21dd3cbac4ff7fb7a38169fd201eb8e63ae7f3e72a7d37be447fd80723.jpg)

![4fa207f7ce5398142d9b10487e0fb34a122c120af6bd3b26bf98308150fb8cfe.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/4fa207f7ce5398142d9b10487e0fb34a122c120af6bd3b26bf98308150fb8cfe.jpg)

![7a03f137178d874a39a8116c6f6f5cd02a0c1fa3ab41bbc9df080092c64caec2.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/7a03f137178d874a39a8116c6f6f5cd02a0c1fa3ab41bbc9df080092c64caec2.jpg)

![c0a2440ba59521b6facffd7e4ef340a2e5f56e5364ac30eca1946b3057c35cbe.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/c0a2440ba59521b6facffd7e4ef340a2e5f56e5364ac30eca1946b3057c35cbe.jpg)

![c57479f9250fe146536e03cbad6f0c72472490d6f5b7f3c14db73f4c041bb577.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/c57479f9250fe146536e03cbad6f0c72472490d6f5b7f3c14db73f4c041bb577.jpg)

![cd165a0f94f197bf20fc14eeb4ef900cd798f0704b6995a1a44553e801976e79.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/cd165a0f94f197bf20fc14eeb4ef900cd798f0704b6995a1a44553e801976e79.jpg)

![f52f4395b1e7369e1b68c93b8b709e8e5d1d1ac64e4cb10394c09c36c3665211.jpg](../icml_results/2465_NNiT_%20Width-Agnostic%20Neural%20Network%20Generation%20with%20Structurally%20Aligned%20Weight%20Spaces/tables/f52f4395b1e7369e1b68c93b8b709e8e5d1d1ac64e4cb10394c09c36c3665211.jpg)

<a id="paper-2436"></a>

## Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch


### Images

![6cb5c210dc554ba85909dd15d552a1e527b536e766f65e304800ddd2dab833db.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/images/6cb5c210dc554ba85909dd15d552a1e527b536e766f65e304800ddd2dab833db.jpg)

![79fe681a4c822af2404f8f17d9e66baca59a1052d6de312c3b3dc1c90f866cf3.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/images/79fe681a4c822af2404f8f17d9e66baca59a1052d6de312c3b3dc1c90f866cf3.jpg)

![87636c7cb0212d2039e774e3a44b3614d72b5337bfa2b25e3722bc38c3314e49.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/images/87636c7cb0212d2039e774e3a44b3614d72b5337bfa2b25e3722bc38c3314e49.jpg)

![b6884c577388b80b30826ab6d41e228e45fc94f456cfaf11614383b9eb8e0633.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/images/b6884c577388b80b30826ab6d41e228e45fc94f456cfaf11614383b9eb8e0633.jpg)

![e99eca5263208c2b84f5daff32704030f5f46bf32588e788f082fe645d5d25d5.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/images/e99eca5263208c2b84f5daff32704030f5f46bf32588e788f082fe645d5d25d5.jpg)

### Tables

![062112e17916abe184636dc7403a3bff9eed381b931c0c4d22b6ff914343b6ee.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/062112e17916abe184636dc7403a3bff9eed381b931c0c4d22b6ff914343b6ee.jpg)

![286a5c7d46830f9529eb6a57dad9ddb96f83094d28847db3b2efc50dc3703e28.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/286a5c7d46830f9529eb6a57dad9ddb96f83094d28847db3b2efc50dc3703e28.jpg)

![31d2fb114554dc2da4206c86a8521a2aea3e89cc07f24cae970bc5aef09aa984.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/31d2fb114554dc2da4206c86a8521a2aea3e89cc07f24cae970bc5aef09aa984.jpg)

![3293bc307445e39f23165b441b42ad34c56bc159fb85ef9069a1df629c8f0cdd.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/3293bc307445e39f23165b441b42ad34c56bc159fb85ef9069a1df629c8f0cdd.jpg)

![49720e2528f8eb313ef78c3b3433c67369936c489f174c70c1be6459ab0dab79.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/49720e2528f8eb313ef78c3b3433c67369936c489f174c70c1be6459ab0dab79.jpg)

![548075a5b57dc7e61c9dfde791b9c98b8d763622214765ddab2391044c1d9b7e.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/548075a5b57dc7e61c9dfde791b9c98b8d763622214765ddab2391044c1d9b7e.jpg)

![748cea84bdba3930a861e36af61d6f683d738f10e9b8aba70a53f488a146965e.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/748cea84bdba3930a861e36af61d6f683d738f10e9b8aba70a53f488a146965e.jpg)

![77fa83afd62da941ea472839f0501e7ba8b0e9b81552e0e2cd84b13aa7f81ef9.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/77fa83afd62da941ea472839f0501e7ba8b0e9b81552e0e2cd84b13aa7f81ef9.jpg)

![7d8312f70dd03d8168fe3508abddbb26b6cd70823465126ad64aa113da2459f7.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/7d8312f70dd03d8168fe3508abddbb26b6cd70823465126ad64aa113da2459f7.jpg)

![7db490819b6c35cf88157f3775008621851433723314132cb2ea70979324a6d7.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/7db490819b6c35cf88157f3775008621851433723314132cb2ea70979324a6d7.jpg)

![9cec1593263629c657ba4f9986c6e380a459a1a6aa255ce3cc24282c155e9dc2.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/9cec1593263629c657ba4f9986c6e380a459a1a6aa255ce3cc24282c155e9dc2.jpg)

![a2468b6e7a05e111d30fd92c06895925425ece9918b968d6ae863c5e72ea0585.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/a2468b6e7a05e111d30fd92c06895925425ece9918b968d6ae863c5e72ea0585.jpg)

![a63cf69c691e2475dc655edbd095d51a981cda5753a14f9e5609dbbc42b2fa99.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/a63cf69c691e2475dc655edbd095d51a981cda5753a14f9e5609dbbc42b2fa99.jpg)

![c7221f3c1b7efb16acd7bba5f316faf65cf5b1d819597951def965b3ec0c4263.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/c7221f3c1b7efb16acd7bba5f316faf65cf5b1d819597951def965b3ec0c4263.jpg)

![e6501ea7f7397539cac6f6d340cccf459aae17df0cad2f156734fed3e9d8c349.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/e6501ea7f7397539cac6f6d340cccf459aae17df0cad2f156734fed3e9d8c349.jpg)

![e6f6b4d9aa635f1b33d9e173741b08a94c5474c1b5f12f3011bec4114f7d06c4.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/e6f6b4d9aa635f1b33d9e173741b08a94c5474c1b5f12f3011bec4114f7d06c4.jpg)

![e6f993ea6279856bd09f2d39eca0533637635cd81db584a4b41725074b7352f3.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/e6f993ea6279856bd09f2d39eca0533637635cd81db584a4b41725074b7352f3.jpg)

![e80d175d164984c0d696e3126348f035669df58f285b7d963a131e8d5412504f.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/e80d175d164984c0d696e3126348f035669df58f285b7d963a131e8d5412504f.jpg)

![f0780679c0a3eb66d9edd23305e35def42fbadf83a5f29c0cca4d6706c83b0f3.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/f0780679c0a3eb66d9edd23305e35def42fbadf83a5f29c0cca4d6706c83b0f3.jpg)

![f94810c50fb7ccd779cd62729a7f824550d70b7861b8e8e17aaec0c67de3481f.jpg](../icml_results/2466_Privasis_%20Synthesizing%20the%20Largest%20_Public_%20Private%20Dataset%20from%20Scratch/tables/f94810c50fb7ccd779cd62729a7f824550d70b7861b8e8e17aaec0c67de3481f.jpg)

<a id="paper-2437"></a>

## Investigating Component Contributions in Multi-Agent ML Systems


### Images

![65052c7e52c43bb7542d83d6c62dc389b57db05599a474c0e53619e8ea652252.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/images/65052c7e52c43bb7542d83d6c62dc389b57db05599a474c0e53619e8ea652252.jpg)

![c1cbaf4ed2e96f049c65cf7b43990efbae51f1ac0a744cc90b270e29a34f9761.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/images/c1cbaf4ed2e96f049c65cf7b43990efbae51f1ac0a744cc90b270e29a34f9761.jpg)

### Tables

![07c21b30b5b6f71bceba20e0b9a229d7a4a3d4b08e3ab552f74b0e90e3b48328.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/07c21b30b5b6f71bceba20e0b9a229d7a4a3d4b08e3ab552f74b0e90e3b48328.jpg)

![0f1e03d3f88fa0770161f112a549ee000427779f86f1414dc0c6efd6825058ec.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/0f1e03d3f88fa0770161f112a549ee000427779f86f1414dc0c6efd6825058ec.jpg)

![0fe53314e5e9b7f7f34309445c1517deb4af0d78092d9a497818f7521806a5fb.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/0fe53314e5e9b7f7f34309445c1517deb4af0d78092d9a497818f7521806a5fb.jpg)

![114ba8ec338da62413aa5c8fd41d03c67ea8bccf56bd633f8aa32a0816e3c7d0.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/114ba8ec338da62413aa5c8fd41d03c67ea8bccf56bd633f8aa32a0816e3c7d0.jpg)

![156561b74638d13125e5872f0f8625de207ed17961768fd3b173a6349838231a.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/156561b74638d13125e5872f0f8625de207ed17961768fd3b173a6349838231a.jpg)

![1d1f99189d191478a47c729a48d36fa8849ab27b7e9525397354e1a674d8b7ee.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/1d1f99189d191478a47c729a48d36fa8849ab27b7e9525397354e1a674d8b7ee.jpg)

![1e023ea7637a5bcdf386d0a8dd52458707793c35e66c80f12a00319e167c4588.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/1e023ea7637a5bcdf386d0a8dd52458707793c35e66c80f12a00319e167c4588.jpg)

![1e050401942f7fe593ff9eee68eb73e8d66324d7451f8c5205b7291c9c02907e.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/1e050401942f7fe593ff9eee68eb73e8d66324d7451f8c5205b7291c9c02907e.jpg)

![25d7b2064e965bce84dc96bc06a06e9aa4c5bf6261e9a6a134d5ec76796d34bb.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/25d7b2064e965bce84dc96bc06a06e9aa4c5bf6261e9a6a134d5ec76796d34bb.jpg)

![2734b104e06c746ef07497af5f168a04675b2892f8ab13b0ce068ca9520d3730.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/2734b104e06c746ef07497af5f168a04675b2892f8ab13b0ce068ca9520d3730.jpg)

![30aaa09d3e177ddb96d43d28cc612e3f5d2a7811efdaaf48ee03b3a8b8123a01.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/30aaa09d3e177ddb96d43d28cc612e3f5d2a7811efdaaf48ee03b3a8b8123a01.jpg)

![3fd128776a80e74f08db53e2982dd81a2a3d811662d2fa1255460c783f7e6050.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/3fd128776a80e74f08db53e2982dd81a2a3d811662d2fa1255460c783f7e6050.jpg)

![43f343d2262515847922b872ea8f9ce5650f1989a02bd687c5d9b6971288e9e8.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/43f343d2262515847922b872ea8f9ce5650f1989a02bd687c5d9b6971288e9e8.jpg)

![52199f227d70478394366df3d46fda3133da747e433be6bfeed8da0093c38dd4.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/52199f227d70478394366df3d46fda3133da747e433be6bfeed8da0093c38dd4.jpg)

![596effc2d07177c427d76fd8f9d6ea6de13d6ea89b015cd8e2f2a40b9f512229.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/596effc2d07177c427d76fd8f9d6ea6de13d6ea89b015cd8e2f2a40b9f512229.jpg)

![666cf0aa21099b56dd66e036f32c16235089911f555f46d01e83a1519943ee22.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/666cf0aa21099b56dd66e036f32c16235089911f555f46d01e83a1519943ee22.jpg)

![6a2e1ab075c2399907635217ec6bd01c4fe85ab67b2d96eb6bc661f062b6ca5a.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/6a2e1ab075c2399907635217ec6bd01c4fe85ab67b2d96eb6bc661f062b6ca5a.jpg)

![7b7bddd1ae0b0c5df44e630fd5e3462c4d54108764170492bd8d706414822edf.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/7b7bddd1ae0b0c5df44e630fd5e3462c4d54108764170492bd8d706414822edf.jpg)

![7bc4c5dec00710e33dd321384a22f06f8d86a18f29519bdfe4057616d89a4ae5.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/7bc4c5dec00710e33dd321384a22f06f8d86a18f29519bdfe4057616d89a4ae5.jpg)

![8874f6b4af8493e2e58910950864b07b290e19a4d07ebda364aefa212a0699d4.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/8874f6b4af8493e2e58910950864b07b290e19a4d07ebda364aefa212a0699d4.jpg)

![98a2ce944acf0cd215f901ce4ede7eeb5cb8b3a0297cdc58887353c0c253301b.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/98a2ce944acf0cd215f901ce4ede7eeb5cb8b3a0297cdc58887353c0c253301b.jpg)

![9c8e49e4eafcef5368122d61c93b35bd24685b0c42aa069cfc7560a572076f5d.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/9c8e49e4eafcef5368122d61c93b35bd24685b0c42aa069cfc7560a572076f5d.jpg)

![a3eeba05babc286295ea7a978b349d0d3a5085dee0f153b2748cf6c933b1a428.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/a3eeba05babc286295ea7a978b349d0d3a5085dee0f153b2748cf6c933b1a428.jpg)

![a98068446c7fb672782d174e05a580c3c9070038009d44e3b10beed6ebf342e6.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/a98068446c7fb672782d174e05a580c3c9070038009d44e3b10beed6ebf342e6.jpg)

![b8fc66ad27a9b890d0a5a45d69a9fc3e57c9068c988ff02f3c27df6ab4a920e1.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/b8fc66ad27a9b890d0a5a45d69a9fc3e57c9068c988ff02f3c27df6ab4a920e1.jpg)

![ba5c8d48a35f95107f8d4acba6f4adc895cc7e87738ff2faa7cca1a655816218.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/ba5c8d48a35f95107f8d4acba6f4adc895cc7e87738ff2faa7cca1a655816218.jpg)

![c46d18c4d537c6b7fe011cd58a16464bd3b2afbbc446b3d79174ff149cf903cc.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/c46d18c4d537c6b7fe011cd58a16464bd3b2afbbc446b3d79174ff149cf903cc.jpg)

![c666dc934c80f3332fd2a3536217a49b96d38e9caf7a9431e5ed732515922198.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/c666dc934c80f3332fd2a3536217a49b96d38e9caf7a9431e5ed732515922198.jpg)

![d39b7a4c65bb3bd4d08c4fc7c13afc38746b1da68145a69eb62fca4767661c9a.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/d39b7a4c65bb3bd4d08c4fc7c13afc38746b1da68145a69eb62fca4767661c9a.jpg)

![d85e13c38a4829206df07d47fb1568e712a9fa882dee2a39d0e8421e980eae85.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/d85e13c38a4829206df07d47fb1568e712a9fa882dee2a39d0e8421e980eae85.jpg)

![dcfa170b3faccb32984fc5257328af8606c18defd10091c444612dd6699873a0.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/dcfa170b3faccb32984fc5257328af8606c18defd10091c444612dd6699873a0.jpg)

![ef6662bb5d75752cc7021b9670cfe26c963e075364fc63d42ce7f8c969749eae.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/ef6662bb5d75752cc7021b9670cfe26c963e075364fc63d42ce7f8c969749eae.jpg)

![fec9eebb7d1822d2d6a04242c407400bdab46fb6c76bd29be447468fc270da8f.jpg](../icml_results/2467_Investigating%20Component%20Contributions%20in%20Multi-Agent%20ML%20Systems/tables/fec9eebb7d1822d2d6a04242c407400bdab46fb6c76bd29be447468fc270da8f.jpg)

<a id="paper-2438"></a>

## Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness, and Safety


### Images

![32c2e69ae2e33f58c62469e47cd34dfe001724f7c4b88be38870792d2b84e9e7.jpg](../icml_results/2468_Multi-task%20Linear%20Regression%20without%20Eigenvalue%20Lower%20Bounds_%20Adaptivity%2C%20Robustness%2C%20and%20Safety/images/32c2e69ae2e33f58c62469e47cd34dfe001724f7c4b88be38870792d2b84e9e7.jpg)

![360b1f334ccdd3d28ceebcc2480c41750a501b854973255c0598a08a621f05ac.jpg](../icml_results/2468_Multi-task%20Linear%20Regression%20without%20Eigenvalue%20Lower%20Bounds_%20Adaptivity%2C%20Robustness%2C%20and%20Safety/images/360b1f334ccdd3d28ceebcc2480c41750a501b854973255c0598a08a621f05ac.jpg)

![7e0e189950e0af6eacc0166fb54d15bb01ced2da901fdf75233594a1ed9b2276.jpg](../icml_results/2468_Multi-task%20Linear%20Regression%20without%20Eigenvalue%20Lower%20Bounds_%20Adaptivity%2C%20Robustness%2C%20and%20Safety/images/7e0e189950e0af6eacc0166fb54d15bb01ced2da901fdf75233594a1ed9b2276.jpg)

![d00ea501b6364f016c62c1861a1d323c98f748a3066902db4a93c8b5f4b4dd5f.jpg](../icml_results/2468_Multi-task%20Linear%20Regression%20without%20Eigenvalue%20Lower%20Bounds_%20Adaptivity%2C%20Robustness%2C%20and%20Safety/images/d00ea501b6364f016c62c1861a1d323c98f748a3066902db4a93c8b5f4b4dd5f.jpg)

### Tables

![e0e5469d555ee7c296cbe1fa97557a186577664104eac18dd31cd84403046f84.jpg](../icml_results/2468_Multi-task%20Linear%20Regression%20without%20Eigenvalue%20Lower%20Bounds_%20Adaptivity%2C%20Robustness%2C%20and%20Safety/tables/e0e5469d555ee7c296cbe1fa97557a186577664104eac18dd31cd84403046f84.jpg)

<a id="paper-2439"></a>

## TokenDrop: Token-Level Importance-Aware Backward Propagation Skipping for Efficient LLM Fine-Tuning


### Images

![0627d951cd976a8a5c215930efd256be305a452aaf1144a909e18931268cb9b4.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/0627d951cd976a8a5c215930efd256be305a452aaf1144a909e18931268cb9b4.jpg)

![12244acae758f089de56fca413e4c557e3846723c0e64af645bb08092d89d16f.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/12244acae758f089de56fca413e4c557e3846723c0e64af645bb08092d89d16f.jpg)

![1c0d84a6deca07a76cd5c4503f1292c7ff953fcd9b2d8a194a320c1bdb7fa2dd.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/1c0d84a6deca07a76cd5c4503f1292c7ff953fcd9b2d8a194a320c1bdb7fa2dd.jpg)

![1eb9a712c24abecf5f0b4fa61822fea5c610a5604ac8a9770e11fa4d1de49a5d.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/1eb9a712c24abecf5f0b4fa61822fea5c610a5604ac8a9770e11fa4d1de49a5d.jpg)

![2b5ac484a94e50eeacce92c557bfe6e0a4e4ec2fb60093227482126fdf5a74d7.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/2b5ac484a94e50eeacce92c557bfe6e0a4e4ec2fb60093227482126fdf5a74d7.jpg)

![3583558e2cf684404814b2a509794f1bcf4043853d3fead655b2ed8357ce708e.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/3583558e2cf684404814b2a509794f1bcf4043853d3fead655b2ed8357ce708e.jpg)

![84b749df3e44a8a98745915ca9b6b60665ad0218b3b304572979d295f5b52696.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/84b749df3e44a8a98745915ca9b6b60665ad0218b3b304572979d295f5b52696.jpg)

![e28fd6869f80aca6fb69545253b3250bbd6979504a50b14582549f7eaface66e.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/images/e28fd6869f80aca6fb69545253b3250bbd6979504a50b14582549f7eaface66e.jpg)

### Tables

![026cb96309f2882b6ef3f92232b8cd0892b96002e3a77d80efa9203619b6e73f.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/026cb96309f2882b6ef3f92232b8cd0892b96002e3a77d80efa9203619b6e73f.jpg)

![20c3cd379939298d1ecbbbf6f46efeb0ca0ef4965c77c68972daa4ce33e17b18.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/20c3cd379939298d1ecbbbf6f46efeb0ca0ef4965c77c68972daa4ce33e17b18.jpg)

![27a607b8cac528e7402246b190ca89d58da30cc81f048cc4f67534ecdd92ef36.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/27a607b8cac528e7402246b190ca89d58da30cc81f048cc4f67534ecdd92ef36.jpg)

![3d75bc9e506419503efeb8304ba1e2ef336e5b56d94602a733005aca03af0372.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/3d75bc9e506419503efeb8304ba1e2ef336e5b56d94602a733005aca03af0372.jpg)

![47d92048089e488999304a300fde56a5e6515910414b8a2e26ca3a7c2ee34d80.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/47d92048089e488999304a300fde56a5e6515910414b8a2e26ca3a7c2ee34d80.jpg)

![487194a742ed526bbf437182c635087d9b1be075cb2d56f139caf4bfa0b3477a.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/487194a742ed526bbf437182c635087d9b1be075cb2d56f139caf4bfa0b3477a.jpg)

![586c46f13c6ed7d6e5b3ff63a6aedc271ba5b0af746a4b96f94602ff62494759.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/586c46f13c6ed7d6e5b3ff63a6aedc271ba5b0af746a4b96f94602ff62494759.jpg)

![8706950d70f8e5c294da8e76fac985f4b0be75b82c26375889327e650fa3da6d.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/8706950d70f8e5c294da8e76fac985f4b0be75b82c26375889327e650fa3da6d.jpg)

![92808339a389f8eaf756cdf81992515bfeefe580b5bcb3949ed4fdba8a29f7d8.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/92808339a389f8eaf756cdf81992515bfeefe580b5bcb3949ed4fdba8a29f7d8.jpg)

![98d2644eb40bb8c28aa5711ad2066f82397470cfb8239dcd417c68b3de4d8625.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/98d2644eb40bb8c28aa5711ad2066f82397470cfb8239dcd417c68b3de4d8625.jpg)

![c481ab732024df5baf001e6cddb30f5a74ec16d5a3643f72a297b52b65b3f76b.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/c481ab732024df5baf001e6cddb30f5a74ec16d5a3643f72a297b52b65b3f76b.jpg)

![d0d6b26419ba0fbc2b952161c1e329d0e83d8b6300859e6758b4cba1c6724ac5.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/d0d6b26419ba0fbc2b952161c1e329d0e83d8b6300859e6758b4cba1c6724ac5.jpg)

![d9396df34e10a6024147808a91601c6fc29447cac894204e5ea7ac614b33b23f.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/d9396df34e10a6024147808a91601c6fc29447cac894204e5ea7ac614b33b23f.jpg)

![f982dc17b5eae67efc6bcb114872d13bf14dbb323ac4617da3aa72b56f53521c.jpg](../icml_results/2469_TokenDrop_%20Token-Level%20Importance-Aware%20Backward%20Propagation%20Skipping%20for%20Efficient%20LLM%20Fine-Tuning/tables/f982dc17b5eae67efc6bcb114872d13bf14dbb323ac4617da3aa72b56f53521c.jpg)

<a id="paper-2440"></a>

## Identifying Connectivity Distributions from Neural Dynamics Using Flows


### Images

![0c656b6213f00affb3dfdb10f574fc37f180c9966a6219e7e61edb97092fe648.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/0c656b6213f00affb3dfdb10f574fc37f180c9966a6219e7e61edb97092fe648.jpg)

![285030aaff6d4c8a7433bfaf8a0e18524035d4e1487d2c4ce8782d8d6e5ab592.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/285030aaff6d4c8a7433bfaf8a0e18524035d4e1487d2c4ce8782d8d6e5ab592.jpg)

![4b345a383134f0932fc9a2c0353b8c4c2b32aabf14f061bd79869dd1dc63a7b9.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/4b345a383134f0932fc9a2c0353b8c4c2b32aabf14f061bd79869dd1dc63a7b9.jpg)

![5ccaf26f50632c17c7b7a70d573b316cd2fefc64bdb8f02331bf3a62166e2a06.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/5ccaf26f50632c17c7b7a70d573b316cd2fefc64bdb8f02331bf3a62166e2a06.jpg)

![6443552fca4d4f604ba73d582f6121af9e45223113fbdd45b46e1540e00c7855.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/6443552fca4d4f604ba73d582f6121af9e45223113fbdd45b46e1540e00c7855.jpg)

![690119fd6c38c80fd2547c2bf73b97ef989825577037346cf629e9c9634ae43d.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/690119fd6c38c80fd2547c2bf73b97ef989825577037346cf629e9c9634ae43d.jpg)

![6c87ebd4c6a001b6d1e558b195a7a18e19cddf58e3d444083ca1027c90970e4d.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/6c87ebd4c6a001b6d1e558b195a7a18e19cddf58e3d444083ca1027c90970e4d.jpg)

![6e334fb5de9b21a205b154dd35ec640ca9230d4b271318cce64eed7e1b8949e2.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/6e334fb5de9b21a205b154dd35ec640ca9230d4b271318cce64eed7e1b8949e2.jpg)

![7312c9c5b16a6c0c31c3afd69b1989010c0f61ba2902a586e73171dbe87151ca.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/7312c9c5b16a6c0c31c3afd69b1989010c0f61ba2902a586e73171dbe87151ca.jpg)

![97a1fe1c27794e3b546dab9d8c75a64ff3ae524108e9332685c8592bcac35aea.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/97a1fe1c27794e3b546dab9d8c75a64ff3ae524108e9332685c8592bcac35aea.jpg)

![abc108bf662214d479a78c08c36f15257fc8cd68d31cbf63b4bde93fc5b1651e.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/abc108bf662214d479a78c08c36f15257fc8cd68d31cbf63b4bde93fc5b1651e.jpg)

![b6a0d224a35adcba11a339295fc27ac929b516513fc1c41e7f4fd9c2a5676953.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/b6a0d224a35adcba11a339295fc27ac929b516513fc1c41e7f4fd9c2a5676953.jpg)

![bcdeb01396745e53f5340f703e4e334ed7b7c8fd4653a13284fa545c5b65c2d1.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/bcdeb01396745e53f5340f703e4e334ed7b7c8fd4653a13284fa545c5b65c2d1.jpg)

![c8a19929ef29f5f6eb8dfc68ec34796167f6b70e996119b9c5d6f45038ee4b92.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/c8a19929ef29f5f6eb8dfc68ec34796167f6b70e996119b9c5d6f45038ee4b92.jpg)

![e1a5182a6fde12fea943ba1a32bec8e66c233444621e9a7357eb0de22e500464.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/e1a5182a6fde12fea943ba1a32bec8e66c233444621e9a7357eb0de22e500464.jpg)

![f77c75ff9e85220fd7269551188a3cb23cb0d226361be886acb14a541e652793.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/f77c75ff9e85220fd7269551188a3cb23cb0d226361be886acb14a541e652793.jpg)

![ff6fec3b237193fb1cb2f68885e0bb921f1e5b0960a9daf64c0acf80fc2ad7ac.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/images/ff6fec3b237193fb1cb2f68885e0bb921f1e5b0960a9daf64c0acf80fc2ad7ac.jpg)

### Tables

![6ad662b1b83280c1c2915b3b2ef623bd4918b3ac981afe215900201ac5287e42.jpg](../icml_results/2470_Identifying%20Connectivity%20Distributions%20from%20Neural%20Dynamics%20Using%20Flows/tables/6ad662b1b83280c1c2915b3b2ef623bd4918b3ac981afe215900201ac5287e42.jpg)

<a id="paper-2441"></a>

## Stability Analysis of Sharpness-Aware Minimization


### Images

![0a9128138b853056cdcdcd5ac0e4ead464decb6d418a9212a376899768351dd9.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/0a9128138b853056cdcdcd5ac0e4ead464decb6d418a9212a376899768351dd9.jpg)

![0d54778c7e8c47ff3809a45bef3b9a519cd58124307bd891fbb1333a505ea758.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/0d54778c7e8c47ff3809a45bef3b9a519cd58124307bd891fbb1333a505ea758.jpg)

![341b858a04d754ccf222c8bd8798130d1770a86f965e9700a5a5b868b8020e6a.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/341b858a04d754ccf222c8bd8798130d1770a86f965e9700a5a5b868b8020e6a.jpg)

![34c64c4bbf49d76065f0d71681ecf5bf3275961c80c4fd4973d80c71bc362d85.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/34c64c4bbf49d76065f0d71681ecf5bf3275961c80c4fd4973d80c71bc362d85.jpg)

![54850c62cf34988bfbcff14402ef9b5e9750dffa8a7f8a9aa821082542c809fb.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/54850c62cf34988bfbcff14402ef9b5e9750dffa8a7f8a9aa821082542c809fb.jpg)

![7acdbb1a81ccb3ed9c7c314e3f9ae71e92494c3eeec789887ad4810d756ebd52.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/7acdbb1a81ccb3ed9c7c314e3f9ae71e92494c3eeec789887ad4810d756ebd52.jpg)

![8e37b18fc15f9277f93622a7649b4bd800171675c0c73a0cac09d54b8d38e460.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/8e37b18fc15f9277f93622a7649b4bd800171675c0c73a0cac09d54b8d38e460.jpg)

![d7d56b7406498e9b50d02e5c717c38c9e07bc3aa3b1bd55b9025923c6969ea55.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/d7d56b7406498e9b50d02e5c717c38c9e07bc3aa3b1bd55b9025923c6969ea55.jpg)

![fc5ba43f434e49d9493ab0c79f80d92f3863733d30f64fb98d7c00aba3ef9d45.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/images/fc5ba43f434e49d9493ab0c79f80d92f3863733d30f64fb98d7c00aba3ef9d45.jpg)

### Tables

![29bbbf1fed2e02d11f7ed4e11c58194dcb8888a58e4567c745738a8fb1da52a9.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/tables/29bbbf1fed2e02d11f7ed4e11c58194dcb8888a58e4567c745738a8fb1da52a9.jpg)

![3d2da39b906c289dc078b1e54583c7efafa6b1cc8c7a02a8f8266a095df29183.jpg](../icml_results/2471_Stability%20Analysis%20of%20Sharpness-Aware%20Minimization/tables/3d2da39b906c289dc078b1e54583c7efafa6b1cc8c7a02a8f8266a095df29183.jpg)

<a id="paper-2442"></a>

## GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache

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
