# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 42 of 197

## 目录 (Table of Contents)

1. [SafeSearch: Automated Red-Teaming of LLM-Based Search Agents](#paper-1354)
2. [TQL: Scaling Q-Functions with Transformers by Preventing Attention Collapse](#paper-1355)
3. [Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](#paper-1356)
4. [How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs](#paper-1357)
5. [HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference](#paper-1358)
6. [ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution](#paper-1359)
7. [PolarDepth: Monocular Transparent Object Depth from Polar-Physics Priors](#paper-1360)
8. [Romberg-Extrapolated Zeroth-Order Gradient Estimator: Higher-Order Bias Reduction with Preserved Leading Directional Variance](#paper-1361)
9. [TurboGS: Accelerating 3D Gaussian Splatting via Error-Guided Sparse Pixel Sampling and Optimization](#paper-1362)
10. [RePack then Refine: Efficient Diffusion Transformers with Vision Foundation Models](#paper-1363)
11. [Mean Flow Policy Optimization](#paper-1364)
12. [Algorithmic Recourse of In-Context Learning for Tabular Data](#paper-1365)
13. [Evaluating LLMs When They Do Not Know the Answer: Statistical Evaluation of Mathematical Reasoning via Comparative Signals](#paper-1366)
14. [Bits That Count: Quantifying and Predicting Capabilities of Language Models](#paper-1367)
15. [Q-Flow: Stable and Expressive Reinforcement Learning with Flow-based Policy](#paper-1368)
16. [Why ReLU? A Bit-Model Dichotomy for Deep Network Training](#paper-1369)
17. [Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction](#paper-1370)
18. [DNA: Uncovering Universal Latent Forgery Knowledge](#paper-1371)
19. [Plan Then Action: High-Level Planning Guidance Reinforcement Learning for LLM Reasoning](#paper-1372)
20. [Conformal Calibration Transfer](#paper-1373)
21. [Fast kernel methods: Sobolev, physics-informed, and additive models](#paper-1374)
22. [Watermarking Graph Neural Networks via Explanations for Ownership Protection](#paper-1375)
23. [Online Compatible Reward Identification from Preference Feedback](#paper-1376)
24. [Chiral Symmetry Breaking in Transformers: A Group-Equivariant Framework for Addressing the Reversal Curse via Adjoint Manifold Mappings](#paper-1377)
25. [Stabilizing PPO via Latent-Space Regularization and KDE-Driven Exploration](#paper-1378)
26. [AesFormer: Transform Everyday Photos into Beautiful Memories](#paper-1379)
27. [FedEMoE: Improving Personalization on Heterogeneous Federated Learning via Elastic Mixture of Experts Architecture](#paper-1380)
28. [Which Heads Matter for Reasoning? RL-Guided KV Cache Compression](#paper-1381)
29. [BioFormer: Rethinking Cross-Subject Generalization via Spectral Structural Alignment in Biomedical Time-Series](#paper-1382)
30. [Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction](#paper-1383)
31. [MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics](#paper-1384)
32. [Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](#paper-1385)
33. [Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](#paper-1386)

---


<a id="paper-1354"></a>

## SafeSearch: Automated Red-Teaming of LLM-Based Search Agents

### Images

![10e1eab35adc7e1e759ef0810f9d578b6cbe6948958c77950fea26bda5750810.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/10e1eab35adc7e1e759ef0810f9d578b6cbe6948958c77950fea26bda5750810.jpg)

![1837248c5848414c17f6909dd1a53628a2695183c666f0c95baf0ae6310daed7.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/1837248c5848414c17f6909dd1a53628a2695183c666f0c95baf0ae6310daed7.jpg)

![1ae724cc2be4aeea9f18ec8d57072f76b6b2f30ba3e7cc3786866f42f5a910e7.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/1ae724cc2be4aeea9f18ec8d57072f76b6b2f30ba3e7cc3786866f42f5a910e7.jpg)

![1cd10bceb6355f0a4a5f546a2a3649ae1e1bf78270cf4e59fd8448ac4543fab5.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/1cd10bceb6355f0a4a5f546a2a3649ae1e1bf78270cf4e59fd8448ac4543fab5.jpg)

![28a01a0e2a5373d940a56b8f7a88e3939bfda776be82b589e3315e1a8c69c1ac.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/28a01a0e2a5373d940a56b8f7a88e3939bfda776be82b589e3315e1a8c69c1ac.jpg)

![3aa297b4b6534ed5902870bc9b9e85e9130537743ecc2b3f60d0c903efced9f7.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/3aa297b4b6534ed5902870bc9b9e85e9130537743ecc2b3f60d0c903efced9f7.jpg)

![4f4b86f1e35fc6cbe035edaca7f63dae85438ab103165328db671f8520eabaf0.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/4f4b86f1e35fc6cbe035edaca7f63dae85438ab103165328db671f8520eabaf0.jpg)

![51b31ef18b751d0b95b7e8cc7d156c7ab498cb81ca77a260f668f1a4a7526d35.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/51b31ef18b751d0b95b7e8cc7d156c7ab498cb81ca77a260f668f1a4a7526d35.jpg)

![5dfd71ea77d467dc716a348c1cd5721af06d9907b30a88e2182525cb43f27c9e.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/5dfd71ea77d467dc716a348c1cd5721af06d9907b30a88e2182525cb43f27c9e.jpg)

![5fc9ca5ad56227c36e46ed241fb0626a454835b80aedd027c2c494fb25fa3ff4.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/5fc9ca5ad56227c36e46ed241fb0626a454835b80aedd027c2c494fb25fa3ff4.jpg)

![6c642fc6258a77b7f66543362926f01653d245c09203430e285ddc34e1734d44.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/6c642fc6258a77b7f66543362926f01653d245c09203430e285ddc34e1734d44.jpg)

![787e591966c4ff684f4729438cdf50918f817ed806e301e2adc9e8280e88893f.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/787e591966c4ff684f4729438cdf50918f817ed806e301e2adc9e8280e88893f.jpg)

![78ab5ff8f6552c6ab45674db2e83bde7dba7a58839b6fe559b816f95b6fd2ba9.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/78ab5ff8f6552c6ab45674db2e83bde7dba7a58839b6fe559b816f95b6fd2ba9.jpg)

![834e5f7126362e13a22ada2949debae13f0c0f47a2f0d69a6ec9596787f1489d.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/834e5f7126362e13a22ada2949debae13f0c0f47a2f0d69a6ec9596787f1489d.jpg)

![96dcf0fb509dc1988f962cf5831a680fefb0ca0bfceeb97a8c5c4eb32f2a733f.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/96dcf0fb509dc1988f962cf5831a680fefb0ca0bfceeb97a8c5c4eb32f2a733f.jpg)

![988eb518fb680e87c90116832faa05d2c415f6281095a00feed4bb743164e32d.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/988eb518fb680e87c90116832faa05d2c415f6281095a00feed4bb743164e32d.jpg)

![a2fd9bf62d449c5e828d8eb72ae4389fb08390bc9fe4556621d9b77c7c63b72c.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/a2fd9bf62d449c5e828d8eb72ae4389fb08390bc9fe4556621d9b77c7c63b72c.jpg)

![c1798698c0bca920d2914cb688dfd920a71fd6bcc8fbd0a6e665dfa0ba55b81d.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/images/c1798698c0bca920d2914cb688dfd920a71fd6bcc8fbd0a6e665dfa0ba55b81d.jpg)

### Tables

![8f8bf2572f5afbac298d7131122f6ff93eea5b78462cd29660793b1ea2612b41.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/tables/8f8bf2572f5afbac298d7131122f6ff93eea5b78462cd29660793b1ea2612b41.jpg)

![a03d8d5961b7f438ca9cde693c6ef3f0a886a5ada7f9174da0ef4535c33696e1.jpg](../icml_results/1373_Probing%20RLVR%20Training%20Instability%20through%20the%20Lens%20of%20Objective-Level%20Hacking/tables/a03d8d5961b7f438ca9cde693c6ef3f0a886a5ada7f9174da0ef4535c33696e1.jpg)

## SafeSearch: Automated Red-Teaming of LLM-Based Search Agents


### Images

![0b1a5e9eb1ebc360e343d625deb8347414f89aba017ce8b900865cbdc1f09fb0.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/0b1a5e9eb1ebc360e343d625deb8347414f89aba017ce8b900865cbdc1f09fb0.jpg)

![41437d2fe18f441a660edec6d13ac88b9d867c157660093ff8f6385424d63326.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/41437d2fe18f441a660edec6d13ac88b9d867c157660093ff8f6385424d63326.jpg)

![455d7b8bd9103afe1da8126736eed66aa6c2854b974a465939890808f715a2ba.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/455d7b8bd9103afe1da8126736eed66aa6c2854b974a465939890808f715a2ba.jpg)

![4c4fc946baf2457cec5b0c81161262b8f58a674728961a94cf538232c69b55b5.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/4c4fc946baf2457cec5b0c81161262b8f58a674728961a94cf538232c69b55b5.jpg)

![64fb4ff79ac96cc573b9bcf510d8b3f131f325f95898c4f6c4cd6093d9d46f6f.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/64fb4ff79ac96cc573b9bcf510d8b3f131f325f95898c4f6c4cd6093d9d46f6f.jpg)

![6a166b1c2c11e6e296442779a82e33c362d1b3160a149454fd2f5a64f8ddeda6.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/6a166b1c2c11e6e296442779a82e33c362d1b3160a149454fd2f5a64f8ddeda6.jpg)

![78a2c5530f0c2707999731602252cf4968f9dab887c01be27932d50c4ed76911.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/78a2c5530f0c2707999731602252cf4968f9dab887c01be27932d50c4ed76911.jpg)

![92f675a2a5300ef753bd298846df64d5fecb31cc8c0d9c345e0dcf37fc59acd9.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/92f675a2a5300ef753bd298846df64d5fecb31cc8c0d9c345e0dcf37fc59acd9.jpg)

![a3f4d42340b61ff4b70ae2290a6a75707a2327a0918260822314c904eef4db1b.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/a3f4d42340b61ff4b70ae2290a6a75707a2327a0918260822314c904eef4db1b.jpg)

![ab732cc65a1a1a1e1ba04d80bb4cd33fb51f5725c3078c92a8690a013c535739.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/ab732cc65a1a1a1e1ba04d80bb4cd33fb51f5725c3078c92a8690a013c535739.jpg)

![af74c31dcf02fa1e0b9e0c9c04d082b59c43631bb463c3ae96f496f513e069fd.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/af74c31dcf02fa1e0b9e0c9c04d082b59c43631bb463c3ae96f496f513e069fd.jpg)

![c5942e2408e8f8aa8a8a84638b88c44f63d6b08a56d813be83f8392d9ba4c0ae.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/c5942e2408e8f8aa8a8a84638b88c44f63d6b08a56d813be83f8392d9ba4c0ae.jpg)

![c7c5ee2008f6018584e7045be2b7d5cae90cc00efc98f538dca26e1905fb2784.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/c7c5ee2008f6018584e7045be2b7d5cae90cc00efc98f538dca26e1905fb2784.jpg)

![d3df33436c75bab09fad7c2f351017f04f73a50251a6a1a31bba9695bcb4561a.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/d3df33436c75bab09fad7c2f351017f04f73a50251a6a1a31bba9695bcb4561a.jpg)

![f7b5d52b6e1f7c174f8e4620bd96ff1461700683ff3434e5a92863e3ebdc9b5d.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/images/f7b5d52b6e1f7c174f8e4620bd96ff1461700683ff3434e5a92863e3ebdc9b5d.jpg)

### Tables

![056281b34076ce9ff6c8837886aca6ca8048e8b2a3ea21753216b8c5b7a7a187.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/056281b34076ce9ff6c8837886aca6ca8048e8b2a3ea21753216b8c5b7a7a187.jpg)

![066f8fd69c3c79f709b06dfed1dc5b04e3d8e30fa4cbef3ef25474340df5dbb8.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/066f8fd69c3c79f709b06dfed1dc5b04e3d8e30fa4cbef3ef25474340df5dbb8.jpg)

![19d04e20e04365d84402c359e195a698254db904411893fedce5ac6a02ac1ba0.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/19d04e20e04365d84402c359e195a698254db904411893fedce5ac6a02ac1ba0.jpg)

![390008ae262fc07978f46b1c6a69032398d8e8ae6170c388c3e378f85eb1b3cc.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/390008ae262fc07978f46b1c6a69032398d8e8ae6170c388c3e378f85eb1b3cc.jpg)

![4ec4b9b720318f671b92f9aadb08c71a7e649ba0db93d5901ae6a14958c32665.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/4ec4b9b720318f671b92f9aadb08c71a7e649ba0db93d5901ae6a14958c32665.jpg)

![4f2070d8c49b2a6bee5ca4c4da100a7cec47f5cdd4bcda1a960f5aaa37e52177.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/4f2070d8c49b2a6bee5ca4c4da100a7cec47f5cdd4bcda1a960f5aaa37e52177.jpg)

![592a37f987e31b5282ebe226165096070028efa58cad6edd74c4fb6a7752e1ac.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/592a37f987e31b5282ebe226165096070028efa58cad6edd74c4fb6a7752e1ac.jpg)

![5ca00922489baa51973a80b16b3770fd919f29377f941e6b28b7c39ffc652ac2.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/5ca00922489baa51973a80b16b3770fd919f29377f941e6b28b7c39ffc652ac2.jpg)

![92803f559807fa87e97b17e238b63df9f1b101b22a29582456e21db30a1bd4bc.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/92803f559807fa87e97b17e238b63df9f1b101b22a29582456e21db30a1bd4bc.jpg)

![a2b915085086710125cd46233f506e0ead2824d023aaa11c1ac927766d5b2a90.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/a2b915085086710125cd46233f506e0ead2824d023aaa11c1ac927766d5b2a90.jpg)

![b5b73270dee85b3937a3bea9537785b1207afd1283acbd0324a12d06b99672a8.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/b5b73270dee85b3937a3bea9537785b1207afd1283acbd0324a12d06b99672a8.jpg)

![bbc7f8f1c1863de00ba0b39f91513e06c4f4558dd89bc5053e68a5c2512e0e6f.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/bbc7f8f1c1863de00ba0b39f91513e06c4f4558dd89bc5053e68a5c2512e0e6f.jpg)

![dc139ebbef1d8784c7f9107cd5287dee0b257c787c4ab413e7ce8dc083a3483c.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/dc139ebbef1d8784c7f9107cd5287dee0b257c787c4ab413e7ce8dc083a3483c.jpg)

![e8dea407f329c0fba4306d67a59a1f86b802213c1bd264df371479bbc5f8ca05.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/e8dea407f329c0fba4306d67a59a1f86b802213c1bd264df371479bbc5f8ca05.jpg)

![e9321155d849a7948c1377c1004dde17693c6df2a291269a16b66094ef5fbf09.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/e9321155d849a7948c1377c1004dde17693c6df2a291269a16b66094ef5fbf09.jpg)

![f69605e75be9f0bafb67d216bc658931b0669e7dcfd5aa8f59367fec0dfeae92.jpg](../icml_results/1374_SafeSearch_%20Automated%20Red-Teaming%20of%20LLM-Based%20Search%20Agents/tables/f69605e75be9f0bafb67d216bc658931b0669e7dcfd5aa8f59367fec0dfeae92.jpg)

<a id="paper-1355"></a>

## TQL: Scaling Q-Functions with Transformers by Preventing Attention Collapse


### Images

![274b6b42cea3ce92ece371697c39d219d300d531f2cfdfecc6b8d39a72c4d080.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/274b6b42cea3ce92ece371697c39d219d300d531f2cfdfecc6b8d39a72c4d080.jpg)

![2774386df2230f870687006ad64a354e927ce34e4b24a16f91bdad5f40f11bfe.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/2774386df2230f870687006ad64a354e927ce34e4b24a16f91bdad5f40f11bfe.jpg)

![40d785a95617241b91e6503a86b823f7e8c8eb98fbc5f692ead1b36dc81c5225.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/40d785a95617241b91e6503a86b823f7e8c8eb98fbc5f692ead1b36dc81c5225.jpg)

![458587870f750811d44ff974fd9a202d238f94769880fd6a17b1882dd380c70b.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/458587870f750811d44ff974fd9a202d238f94769880fd6a17b1882dd380c70b.jpg)

![6dc15e06e9fd73c9de1bccbe57cde39750637c19762c24bd3078c78ab2bc076c.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/6dc15e06e9fd73c9de1bccbe57cde39750637c19762c24bd3078c78ab2bc076c.jpg)

![8c92643bb9fa1a6080a27bc49134e9ef767fb78c69494efa71b1c83385a54cf6.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/8c92643bb9fa1a6080a27bc49134e9ef767fb78c69494efa71b1c83385a54cf6.jpg)

![985e70c495e0012049d69f9e27b95f00fc759b0583c10c2ec21d35db1a781f3e.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/985e70c495e0012049d69f9e27b95f00fc759b0583c10c2ec21d35db1a781f3e.jpg)

![99166e1d414245f87616309602cd03aa5ddd91f91b358db2f1abec6a80cf3ef7.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/99166e1d414245f87616309602cd03aa5ddd91f91b358db2f1abec6a80cf3ef7.jpg)

![af94d40478969b6c10ef4cadaad56ed0f716935c5d784b37a3bfeaf32b6fec17.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/af94d40478969b6c10ef4cadaad56ed0f716935c5d784b37a3bfeaf32b6fec17.jpg)

![c260053a2da45bb0474ecf1064589dbcfb69261bdb73dd9b298c063a164b90b0.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/c260053a2da45bb0474ecf1064589dbcfb69261bdb73dd9b298c063a164b90b0.jpg)

![e44a435659cd12ab167d3d7e3b4b4baa4e3dd2501a8ad7974744f5111aa8af8d.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/e44a435659cd12ab167d3d7e3b4b4baa4e3dd2501a8ad7974744f5111aa8af8d.jpg)

![e5f28c3c0e3202d71d2c36debbb51c03d6d2f68a5b6f49723bbb8c9430d09a20.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/e5f28c3c0e3202d71d2c36debbb51c03d6d2f68a5b6f49723bbb8c9430d09a20.jpg)

![f6d7e064e19eec40a7f537c9e2376e44ce786c65d7875aa526bc3ecf075dfbfd.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/f6d7e064e19eec40a7f537c9e2376e44ce786c65d7875aa526bc3ecf075dfbfd.jpg)

![fafe4b9ce5200563417cb75a5b5a7e0a6379ac1f30520adf5b3936bd8914ff4d.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/images/fafe4b9ce5200563417cb75a5b5a7e0a6379ac1f30520adf5b3936bd8914ff4d.jpg)

### Tables

![600bfe7601950d6d0521b9c80b25f351e23312805b1fe03a8d58ebb3bf75b8a2.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/tables/600bfe7601950d6d0521b9c80b25f351e23312805b1fe03a8d58ebb3bf75b8a2.jpg)

![91bcaf10dc9da67e928d1b9821ab5d7f147e58ab18cad5a0b8c36215a3e2ac44.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/tables/91bcaf10dc9da67e928d1b9821ab5d7f147e58ab18cad5a0b8c36215a3e2ac44.jpg)

![9d6aade2a9dae76a30c3417642f5452c5bbe387fa3c6799bc31a8e1e44b1d3e0.jpg](../icml_results/1375_TQL_%20Scaling%20Q-Functions%20with%20Transformers%20by%20Preventing%20Attention%20Collapse/tables/9d6aade2a9dae76a30c3417642f5452c5bbe387fa3c6799bc31a8e1e44b1d3e0.jpg)

<a id="paper-1356"></a>

## Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift


### Images

![090d4086380704c856e9aed34f7cffa71af9cf2971e02a746c99f9a59c63c6ec.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/images/090d4086380704c856e9aed34f7cffa71af9cf2971e02a746c99f9a59c63c6ec.jpg)

![e5c447b37cba0fc1520a6d26e560cd7e687ea3b45ce2234ee349c14be408c7c0.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/images/e5c447b37cba0fc1520a6d26e560cd7e687ea3b45ce2234ee349c14be408c7c0.jpg)

![ed5f1037b3acff9a7300de495d057439da5ecd7f524ae3a98b6f0d23def3774d.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/images/ed5f1037b3acff9a7300de495d057439da5ecd7f524ae3a98b6f0d23def3774d.jpg)

### Tables

![359aaf91db333b89a6f48d58ce48de6df6c8eaf5d629774d32a3a67f8f227360.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/359aaf91db333b89a6f48d58ce48de6df6c8eaf5d629774d32a3a67f8f227360.jpg)

![4a8e8c12c51b50bf485c4e7ee94577c5499f459e818afa6bb610502d8820702a.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/4a8e8c12c51b50bf485c4e7ee94577c5499f459e818afa6bb610502d8820702a.jpg)

![4b30c04b7facead4c735bb05c70122295668077384ae07751c697dcd0ac2c263.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/4b30c04b7facead4c735bb05c70122295668077384ae07751c697dcd0ac2c263.jpg)

![54de7bf02abad8a6f34bcf7b06eb380a9324e0cbe3d8ecf26e28aeefc4f62b67.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/54de7bf02abad8a6f34bcf7b06eb380a9324e0cbe3d8ecf26e28aeefc4f62b67.jpg)

![5f126a2e090f7d3df56c5559f2e058f9471cdcf7f6e900b0cc60134458d0b949.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/5f126a2e090f7d3df56c5559f2e058f9471cdcf7f6e900b0cc60134458d0b949.jpg)

![5fa04caeebe5ad12c29c5d43425167c797a96d8c89ac38a98b04e50bf8fb26f2.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/5fa04caeebe5ad12c29c5d43425167c797a96d8c89ac38a98b04e50bf8fb26f2.jpg)

![928d6d20e6f02d6fb8daff4a708ba76eca31b8941e46d6bf1f9f10cf99ab9953.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/928d6d20e6f02d6fb8daff4a708ba76eca31b8941e46d6bf1f9f10cf99ab9953.jpg)

![f6f042b095541e69c127c8912107d99114473236244c6af54d473267b6a99afa.jpg](../icml_results/1376_Expectation%20Consistency%20Loss_%20Rethink%20Confidence%20Calibration%20under%20Covariate%20Shift/tables/f6f042b095541e69c127c8912107d99114473236244c6af54d473267b6a99afa.jpg)

<a id="paper-1357"></a>

## How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs


### Images

![2776165e7fe97ee20dda581dd8fd7210d038140a93e8ff4a21a4e3ba28f1e908.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/2776165e7fe97ee20dda581dd8fd7210d038140a93e8ff4a21a4e3ba28f1e908.jpg)

![3530366b1ec1ebe1affcabc1b9826cbe6306cf88838ea72baae51666e150a5da.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/3530366b1ec1ebe1affcabc1b9826cbe6306cf88838ea72baae51666e150a5da.jpg)

![72fc54c71a4983ba56d0e282e147ef97f72648ab0c85b8a08827814e03669633.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/72fc54c71a4983ba56d0e282e147ef97f72648ab0c85b8a08827814e03669633.jpg)

![805d2d1bdee69d487f760ef2ef7c528a016d4552977eded9501237e9df8df635.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/805d2d1bdee69d487f760ef2ef7c528a016d4552977eded9501237e9df8df635.jpg)

![a2a281361dd8529ce33590e1ce7572f54ea4e89b7ea1f45ed4a1807b64cc49e0.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/a2a281361dd8529ce33590e1ce7572f54ea4e89b7ea1f45ed4a1807b64cc49e0.jpg)

![a7f5918b36cab25e23f650764af17fe77bd840ce7e7b565c2426d5ce3a2f409d.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/a7f5918b36cab25e23f650764af17fe77bd840ce7e7b565c2426d5ce3a2f409d.jpg)

![e6adb0f296ee8020e72863c17be52a564431c274e0e1f333dca225d86f78726d.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/images/e6adb0f296ee8020e72863c17be52a564431c274e0e1f333dca225d86f78726d.jpg)

### Tables

![0903af99ac284b51539cd343f97ecd07f0516cc7d897c40cd5143331730fb3f6.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/0903af99ac284b51539cd343f97ecd07f0516cc7d897c40cd5143331730fb3f6.jpg)

![57601d6b15fe1bd95b078d129b651558dd4a96f8291348e063e755ec70fb44a0.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/57601d6b15fe1bd95b078d129b651558dd4a96f8291348e063e755ec70fb44a0.jpg)

![7dfc9c0308cf5a5375cbd836e44bc8110cb3a7deba36c42d61e0aeffc22f637b.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/7dfc9c0308cf5a5375cbd836e44bc8110cb3a7deba36c42d61e0aeffc22f637b.jpg)

![86e9003be8e42bcb546d6046e53e6c641f2de099f40ddef9ae1614902485aaa1.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/86e9003be8e42bcb546d6046e53e6c641f2de099f40ddef9ae1614902485aaa1.jpg)

![a225ce1f1e475ea6c88f2b9d0346a12ffe9edb4509c8b58d872229cfa14121c0.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/a225ce1f1e475ea6c88f2b9d0346a12ffe9edb4509c8b58d872229cfa14121c0.jpg)

![b7a9ad1106707eaebbeff9c3abf7b6182641037d7973d86b355b2c7fc5be5a88.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/b7a9ad1106707eaebbeff9c3abf7b6182641037d7973d86b355b2c7fc5be5a88.jpg)

![d238a52e60c8ca63510b85a71a0d337a7eecf868f769969e3f6f3520f7dfd4c1.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/d238a52e60c8ca63510b85a71a0d337a7eecf868f769969e3f6f3520f7dfd4c1.jpg)

![d99336ab62a87649fbba850363d3081ce1b95110f941900a8ad59c5db0986634.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/d99336ab62a87649fbba850363d3081ce1b95110f941900a8ad59c5db0986634.jpg)

![e21902125c2a81b7180f5297a23b93ed07dc592b09ab17b47625111f7298d5dc.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/e21902125c2a81b7180f5297a23b93ed07dc592b09ab17b47625111f7298d5dc.jpg)

![e60689a4f15d40d4a43437618f76d6a2beef3e8b3936812805efb02be120cea6.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/e60689a4f15d40d4a43437618f76d6a2beef3e8b3936812805efb02be120cea6.jpg)

![efd76bb0684886382602f20362215e2b4c15327428a42a6c562aa65a262a6525.jpg](../icml_results/1377_How%20Does%20Reasoning%20Flow_%20Tracing%20Attention-Induced%20Information%20Flow%20for%20Targeted%20RL%20in%20LLMs/tables/efd76bb0684886382602f20362215e2b4c15327428a42a6c562aa65a262a6525.jpg)

<a id="paper-1358"></a>

## HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference


### Images

![19cf6b49817c8cace3fd973735c16075d0881ff9c349463da4365c853eaeef2d.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/images/19cf6b49817c8cace3fd973735c16075d0881ff9c349463da4365c853eaeef2d.jpg)

![7e477409ff68f2f49516d40e89c914e54e1678abe07ba9906b3465adb6818cb9.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/images/7e477409ff68f2f49516d40e89c914e54e1678abe07ba9906b3465adb6818cb9.jpg)

![91b7a3fd90365552ec92ecac46e201b95343a1ea62a147b2d4ae894d64bcb78f.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/images/91b7a3fd90365552ec92ecac46e201b95343a1ea62a147b2d4ae894d64bcb78f.jpg)

![a60cc19441c6fd0aab10d55e075691435bbc94f9238616c06b3fe6e5023514db.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/images/a60cc19441c6fd0aab10d55e075691435bbc94f9238616c06b3fe6e5023514db.jpg)

![b7c98f90174d61a69aa41329b0be44ba0fe0536aa1a9c12b5dc952fe4b671edd.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/images/b7c98f90174d61a69aa41329b0be44ba0fe0536aa1a9c12b5dc952fe4b671edd.jpg)

### Tables

![1907f8698cf514567ec03c035cb8c6182f50869e35ae2ce6c608024d98065a01.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/1907f8698cf514567ec03c035cb8c6182f50869e35ae2ce6c608024d98065a01.jpg)

![3371eb5f2c1b5d5c40b7f3d2224a119efe461e3773e97baf3a54a56e92c26528.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/3371eb5f2c1b5d5c40b7f3d2224a119efe461e3773e97baf3a54a56e92c26528.jpg)

![3ecbc25985d8d29d99bbb22ef4b6e5080628557121a76a67c5466d92cea5dc0f.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/3ecbc25985d8d29d99bbb22ef4b6e5080628557121a76a67c5466d92cea5dc0f.jpg)

![95f91252c28f81d85d2cd241fb6b1821b1db58e2fec9b82f77feecbd623f9713.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/95f91252c28f81d85d2cd241fb6b1821b1db58e2fec9b82f77feecbd623f9713.jpg)

![99b2e67758914e0295fc977dc659d4917690bea3151adab6f5688cfdea1b60dc.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/99b2e67758914e0295fc977dc659d4917690bea3151adab6f5688cfdea1b60dc.jpg)

![b71c380b71d8d91e9ff332c0a74856fa30b27d5720b10c6c836f862b6222e342.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/b71c380b71d8d91e9ff332c0a74856fa30b27d5720b10c6c836f862b6222e342.jpg)

![c79de3b4a2b476f698efe3525108722ef0bff865c2a2158a179f53f99ea98501.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/c79de3b4a2b476f698efe3525108722ef0bff865c2a2158a179f53f99ea98501.jpg)

![e0b43fb214a2c6c56cd816ee891e0828b1c5738459b3c1aeb4272a6a39144840.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/e0b43fb214a2c6c56cd816ee891e0828b1c5738459b3c1aeb4272a6a39144840.jpg)

![f8b50ba7a424a55384c63a2151dcdad55f0ef65f2378b52bda0d98074c735481.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/f8b50ba7a424a55384c63a2151dcdad55f0ef65f2378b52bda0d98074c735481.jpg)

![fb45efa9d8a0b557a77edb0e1c1ec0ac7545f530372513db4597ffcd06c4d012.jpg](../icml_results/1378_HybridFlow_%20Resource-Adaptive%20Subtask%20Routing%20for%20Efficient%20Edge-Cloud%20LLM%20Inference/tables/fb45efa9d8a0b557a77edb0e1c1ec0ac7545f530372513db4597ffcd06c4d012.jpg)

<a id="paper-1359"></a>

## ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution


### Images

![20b70f2fba7fb6fe92d04d8ba2dc9a3475f0b2042b637b6f70edd4060f5dc52c.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/images/20b70f2fba7fb6fe92d04d8ba2dc9a3475f0b2042b637b6f70edd4060f5dc52c.jpg)

![7a623f6224c9419055a4070751b0ec3c33338a0774dd4c51b1f1792fed6fdbb0.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/images/7a623f6224c9419055a4070751b0ec3c33338a0774dd4c51b1f1792fed6fdbb0.jpg)

![948b39c8bb926eb08765bfa019863980684460b6f29fad21fbdab8482b26193f.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/images/948b39c8bb926eb08765bfa019863980684460b6f29fad21fbdab8482b26193f.jpg)

![d5af8969a928fcd3bdc3123523f3db531fd9409141b0e2c83603ba30949f2a38.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/images/d5af8969a928fcd3bdc3123523f3db531fd9409141b0e2c83603ba30949f2a38.jpg)

### Tables

![0ab8b05f8f0ec4ed7c228f8c5e59a2fead1fd9e0692633591a4b193c7bf90253.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/0ab8b05f8f0ec4ed7c228f8c5e59a2fead1fd9e0692633591a4b193c7bf90253.jpg)

![0b8144130ccb8c78536535bea7314bae357a183a85436f3bbf23bdbf40f740fe.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/0b8144130ccb8c78536535bea7314bae357a183a85436f3bbf23bdbf40f740fe.jpg)

![194654179b23d2a2178867c614ba64ee75e6fcdfa16dc6d2384665bf2461e1ee.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/194654179b23d2a2178867c614ba64ee75e6fcdfa16dc6d2384665bf2461e1ee.jpg)

![194a125c9cc8da6adf07710ccca7144e112fd9382c0972c7ba4fda32530c466e.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/194a125c9cc8da6adf07710ccca7144e112fd9382c0972c7ba4fda32530c466e.jpg)

![1a2d1e245db37fbb7c794a934e177fff784558dc9bf9f51b8fd37aa2a31804d3.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/1a2d1e245db37fbb7c794a934e177fff784558dc9bf9f51b8fd37aa2a31804d3.jpg)

![1d8c40d3375651ee4b006176348488cb83d0bc5a0fe0788fae35cd8b37653fbe.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/1d8c40d3375651ee4b006176348488cb83d0bc5a0fe0788fae35cd8b37653fbe.jpg)

![226d7d665d7ee514ddcc0ceac0c7e42c5ab565d861f84928009acea02e44e084.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/226d7d665d7ee514ddcc0ceac0c7e42c5ab565d861f84928009acea02e44e084.jpg)

![41684d2410003eb0cf75864cc7ed53ea34ed1cee4ded05f87ba91f28cf515cf2.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/41684d2410003eb0cf75864cc7ed53ea34ed1cee4ded05f87ba91f28cf515cf2.jpg)

![548f2feaddcabf43d201a02128eeb10959bc5a3b91e6366fbf8dc1cfb858e4ac.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/548f2feaddcabf43d201a02128eeb10959bc5a3b91e6366fbf8dc1cfb858e4ac.jpg)

![99c91e6d0c6c81204c387babe8b9e56218fea825c68862cc16c5428d57e1af45.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/99c91e6d0c6c81204c387babe8b9e56218fea825c68862cc16c5428d57e1af45.jpg)

![9a55213f182b6339acdc8d984d0cecb9d5bd119eddf3b8bfb7288ff3ce22113b.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/9a55213f182b6339acdc8d984d0cecb9d5bd119eddf3b8bfb7288ff3ce22113b.jpg)

![9ac31dacede8e0e38ae960bf6fe72a17d7876e31d1790a14b3c980ec324bfbab.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/9ac31dacede8e0e38ae960bf6fe72a17d7876e31d1790a14b3c980ec324bfbab.jpg)

![c6e8b6f348ff27699bceba73a6839bd8880ffc729e307e63b34a888b3743bc22.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/c6e8b6f348ff27699bceba73a6839bd8880ffc729e307e63b34a888b3743bc22.jpg)

![cc19157f7dbf57e4b27a5c81dfacb7716052a52d7772a53de39d2db1436135d1.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/cc19157f7dbf57e4b27a5c81dfacb7716052a52d7772a53de39d2db1436135d1.jpg)

![d0cb53e5bb6d7ff0d13ffa2ac101bcd9fa8dae58f07386b52a16a5c08b23e5af.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/d0cb53e5bb6d7ff0d13ffa2ac101bcd9fa8dae58f07386b52a16a5c08b23e5af.jpg)

![e863f633878a7c5ec03197128aa6eb88d772655f1d0d68ccca75f7a40b00fdf3.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/e863f633878a7c5ec03197128aa6eb88d772655f1d0d68ccca75f7a40b00fdf3.jpg)

![eb11970561807b28cf1769adb94516fc1e368d337f93222adf89f8d2de2b8ef9.jpg](../icml_results/1379_ForesightKV_%20Optimizing%20KV%20Cache%20Eviction%20for%20Reasoning%20Models%20by%20Learning%20Long-Term%20Contribution/tables/eb11970561807b28cf1769adb94516fc1e368d337f93222adf89f8d2de2b8ef9.jpg)

<a id="paper-1360"></a>

## PolarDepth: Monocular Transparent Object Depth from Polar-Physics Priors


### Images

![1e8bdfde573d057260b12582affafd2aecf526c4ef69834616a06bb5d134df5a.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/1e8bdfde573d057260b12582affafd2aecf526c4ef69834616a06bb5d134df5a.jpg)

![3bddd43cacf550c116a75c6f692dc1d9d752905a57d4bec3df38151affc3d512.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/3bddd43cacf550c116a75c6f692dc1d9d752905a57d4bec3df38151affc3d512.jpg)

![3d133d2354c00c2f28d10ddd31d334d495059dd4282abb4c4a832bc56126f819.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/3d133d2354c00c2f28d10ddd31d334d495059dd4282abb4c4a832bc56126f819.jpg)

![40665404792dc8e8f4e2ba31b5dd51942f69f8a33bfe888d8eb780d707e941a9.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/40665404792dc8e8f4e2ba31b5dd51942f69f8a33bfe888d8eb780d707e941a9.jpg)

![7898cd9448190adc22a71213d6ecba5ce9dd3312face124f9a126fa0dbb38adc.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/7898cd9448190adc22a71213d6ecba5ce9dd3312face124f9a126fa0dbb38adc.jpg)

![7dbdf0677707c5a47bbc01c0d68d12781f45c8d2b3ccea056771306d1cd03254.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/7dbdf0677707c5a47bbc01c0d68d12781f45c8d2b3ccea056771306d1cd03254.jpg)

![7e2ac11e2c2611c4edcc527367a0dfb642884b02d3235ea767485a5d2b48847d.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/7e2ac11e2c2611c4edcc527367a0dfb642884b02d3235ea767485a5d2b48847d.jpg)

![a148abadbb89c1fa829c4f2d2201da77bd329a645d7dd22d072f3bd363138eb4.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/a148abadbb89c1fa829c4f2d2201da77bd329a645d7dd22d072f3bd363138eb4.jpg)

![bf3ced1cc61c32bfdc315355cca4752f9ca5ee05e34643612b82b03ea15f7e4d.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/bf3ced1cc61c32bfdc315355cca4752f9ca5ee05e34643612b82b03ea15f7e4d.jpg)

![e8c8e3a8eb8849e2aa2ccabb238108767b00b90996074824988e80549291a692.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/images/e8c8e3a8eb8849e2aa2ccabb238108767b00b90996074824988e80549291a692.jpg)

### Tables

![05d24d39dfe88eebc8bfc01f91b89538912c6f16ed9742b5e91ab2b4e13cec32.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/05d24d39dfe88eebc8bfc01f91b89538912c6f16ed9742b5e91ab2b4e13cec32.jpg)

![2d2da3d269a6c27a6836fddaa3e2ee54ad0146a3c7c7af1e2f998a48115d593b.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/2d2da3d269a6c27a6836fddaa3e2ee54ad0146a3c7c7af1e2f998a48115d593b.jpg)

![2e709b739bb283c4d8a2074f94f42609edb2bf80ca6d826a2d4a75a8773ff340.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/2e709b739bb283c4d8a2074f94f42609edb2bf80ca6d826a2d4a75a8773ff340.jpg)

![975aac0566c212def1d0429909b605f04b8f7cedec9dda1219a632282240bf7d.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/975aac0566c212def1d0429909b605f04b8f7cedec9dda1219a632282240bf7d.jpg)

![a8be8678d852e2bc3a2477cd10a56ce817692155e31aa8e9521a979c9e8e02c0.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/a8be8678d852e2bc3a2477cd10a56ce817692155e31aa8e9521a979c9e8e02c0.jpg)

![dde006081c66acea784879918a8740bb7196f3d0cc0a8c426628f499756c01d0.jpg](../icml_results/1380_PolarDepth_%20Monocular%20Transparent%20Object%20Depth%20from%20Polar-Physics%20Priors/tables/dde006081c66acea784879918a8740bb7196f3d0cc0a8c426628f499756c01d0.jpg)

<a id="paper-1361"></a>

## Romberg-Extrapolated Zeroth-Order Gradient Estimator: Higher-Order Bias Reduction with Preserved Leading Directional Variance


### Images

![0415d046ac4b9c7bee3da871e5ac75d2acf9292c76b0238686cba9fac50e5c0f.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/0415d046ac4b9c7bee3da871e5ac75d2acf9292c76b0238686cba9fac50e5c0f.jpg)

![50a5b4e70cb200f3cd18c23ad504e4e5ee01c8dc8d4806384317c55fee61cc05.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/50a5b4e70cb200f3cd18c23ad504e4e5ee01c8dc8d4806384317c55fee61cc05.jpg)

![5be709e315a7ddcea46b2bbe32991f1a1369bb3b05cdb38049b92d56fe206b39.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/5be709e315a7ddcea46b2bbe32991f1a1369bb3b05cdb38049b92d56fe206b39.jpg)

![61cd5f0efb5b9480a9c3884b657f47f9bb74b079f6a0592293c822ca50bd241b.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/61cd5f0efb5b9480a9c3884b657f47f9bb74b079f6a0592293c822ca50bd241b.jpg)

![6b20f22ee62bc8bc75f5eb0088e98c52d3518bc79b11aea47b4f242943fbe59f.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/6b20f22ee62bc8bc75f5eb0088e98c52d3518bc79b11aea47b4f242943fbe59f.jpg)

![7b013262687a5e5c926baaeddfa70db2bf64227d98a6f156f712fcf9148b837a.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/7b013262687a5e5c926baaeddfa70db2bf64227d98a6f156f712fcf9148b837a.jpg)

![823562aac8a4d36a022c270cd295bccfa6ced12e827b9651579964263594bd6e.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/823562aac8a4d36a022c270cd295bccfa6ced12e827b9651579964263594bd6e.jpg)

![94992e8cd382abb7118e311df123c19e8686ba0052c8b9f1ef434305b71b3dc1.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/94992e8cd382abb7118e311df123c19e8686ba0052c8b9f1ef434305b71b3dc1.jpg)

![a53c8dfbbe8431a0623b8a954b8e6a5307249d4ee166263b01aa5fc34079045b.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/a53c8dfbbe8431a0623b8a954b8e6a5307249d4ee166263b01aa5fc34079045b.jpg)

![c66d0c5d55e9115e9987f390668199e751bcb46c4fe24b2fcbba58259f90edba.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/images/c66d0c5d55e9115e9987f390668199e751bcb46c4fe24b2fcbba58259f90edba.jpg)

### Tables

![1869b36a2aa02f0ee608fa9e7cf999ff99595f8043681b66b1cfe1748b255d73.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/1869b36a2aa02f0ee608fa9e7cf999ff99595f8043681b66b1cfe1748b255d73.jpg)

![50d670a8dcf9edf5821e62f82dd3bf2c4b0664deaca9fb772aededa0b8dc59f2.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/50d670a8dcf9edf5821e62f82dd3bf2c4b0664deaca9fb772aededa0b8dc59f2.jpg)

![742a8964f6fa47c0d3d015a2e383f9fbb8b947c5102f44985dd42e964b0ab0a3.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/742a8964f6fa47c0d3d015a2e383f9fbb8b947c5102f44985dd42e964b0ab0a3.jpg)

![cfc5faf8b83343f560cbc2f6cecd39efc94d6f729bc456354c97890575db28b4.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/cfc5faf8b83343f560cbc2f6cecd39efc94d6f729bc456354c97890575db28b4.jpg)

![df880d41e0a006328aa84e01ba16df21f9c1bc06dd42532dfe8bdc1c869eaf04.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/df880d41e0a006328aa84e01ba16df21f9c1bc06dd42532dfe8bdc1c869eaf04.jpg)

![fab6e1d5a862b2402a0294749348d5bf7b542bba0ba09091a16312ba938ce27f.jpg](../icml_results/1381_Romberg-Extrapolated%20Zeroth-Order%20Gradient%20Estimator_%20Higher-Order%20Bias%20Reduction%20with%20Preserved%20Lea/tables/fab6e1d5a862b2402a0294749348d5bf7b542bba0ba09091a16312ba938ce27f.jpg)

<a id="paper-1362"></a>

## TurboGS: Accelerating 3D Gaussian Splatting via Error-Guided Sparse Pixel Sampling and Optimization


### Images

![0e62e0e8c93a5a8a5c66e3f685dd490e28eeda4d559974c19a5dd407a905659b.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/0e62e0e8c93a5a8a5c66e3f685dd490e28eeda4d559974c19a5dd407a905659b.jpg)

![19cb6d33f391612ea26cff1b2a0c2a8703c0e9d70fd3c7693bb9b4953d36886b.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/19cb6d33f391612ea26cff1b2a0c2a8703c0e9d70fd3c7693bb9b4953d36886b.jpg)

![40969d7c4688cc2316248da9514d8a08b5c612fc2126370271ff2a186d6962b8.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/40969d7c4688cc2316248da9514d8a08b5c612fc2126370271ff2a186d6962b8.jpg)

![4f00429199445cccde110bbf7c47169936716202ae579e3718d6446c7937a9f8.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/4f00429199445cccde110bbf7c47169936716202ae579e3718d6446c7937a9f8.jpg)

![690609c6febcc94f809cecd440e0ffa0bae6e3ae9cd355a48c17caa8a48d287a.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/690609c6febcc94f809cecd440e0ffa0bae6e3ae9cd355a48c17caa8a48d287a.jpg)

![7d711590d7bde8b356733106353ee49bdf33c598dd1bf2452c3eafbf7cfbdb3b.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/7d711590d7bde8b356733106353ee49bdf33c598dd1bf2452c3eafbf7cfbdb3b.jpg)

![88a9d1dbc65a827270a41fe8197a9f8ea652563bb5b23c900ef584811333e349.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/88a9d1dbc65a827270a41fe8197a9f8ea652563bb5b23c900ef584811333e349.jpg)

![923dca1fcfcdf8beb4d092240d4ae02493ecdd4494674fbfb57d05bcbd0cce91.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/923dca1fcfcdf8beb4d092240d4ae02493ecdd4494674fbfb57d05bcbd0cce91.jpg)

![9f5c15c9e4de292d675d6ee7af532cfc1ed2edc0acfd2f94d200690710ae0ae3.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/9f5c15c9e4de292d675d6ee7af532cfc1ed2edc0acfd2f94d200690710ae0ae3.jpg)

![b20a4ecd93aaf4d3efcca01d7f797069b46ebfe877604fb361d659605d7e7f91.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/b20a4ecd93aaf4d3efcca01d7f797069b46ebfe877604fb361d659605d7e7f91.jpg)

![c6b4b153e5bf6e3b003b06d289bfa198001ff649f155d0c3e48caa7583d53f2d.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/c6b4b153e5bf6e3b003b06d289bfa198001ff649f155d0c3e48caa7583d53f2d.jpg)

![cd8a7460fed4a02b4c2b5edef5bce2c2d67ff9acbef8e3728a06d43186601b6f.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/cd8a7460fed4a02b4c2b5edef5bce2c2d67ff9acbef8e3728a06d43186601b6f.jpg)

![cfb9adfd9446cdd5a53bf23592ea87dc14d04119426d73307d621e9b6128eff8.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/cfb9adfd9446cdd5a53bf23592ea87dc14d04119426d73307d621e9b6128eff8.jpg)

![e771c56155b5755e0f76ec20c7bd4b1ad66b2974060b89e88188c63ee62cd784.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/e771c56155b5755e0f76ec20c7bd4b1ad66b2974060b89e88188c63ee62cd784.jpg)

![eac3289346f4ea5e15b282daddd2fccc9fc46c5ac974768dea0df6daeca29990.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/eac3289346f4ea5e15b282daddd2fccc9fc46c5ac974768dea0df6daeca29990.jpg)

![f894078f0b05b1f884e309c71bd01995960800a7f166d44fdad8fc8e003d823f.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/images/f894078f0b05b1f884e309c71bd01995960800a7f166d44fdad8fc8e003d823f.jpg)

### Tables

![17b476ac40b6fa1e06372c98323b37c6d3a847250fb07f0b63ed23b2e0d9a028.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/17b476ac40b6fa1e06372c98323b37c6d3a847250fb07f0b63ed23b2e0d9a028.jpg)

![2254ebab727ad3f7c54508217089831aa9d457005ad6659244de68c9160449ee.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/2254ebab727ad3f7c54508217089831aa9d457005ad6659244de68c9160449ee.jpg)

![297a26c0fab9d91ba5baad78d221d902c09b50c612db9d2aff29d9592ffa0528.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/297a26c0fab9d91ba5baad78d221d902c09b50c612db9d2aff29d9592ffa0528.jpg)

![4862df84e1332c2007b5b26d40416d16ab4d73c012686386cbff33c2cb958fa0.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/4862df84e1332c2007b5b26d40416d16ab4d73c012686386cbff33c2cb958fa0.jpg)

![6b90cf39310446bb87dd37b9094c719cc437d23057f20800d62c67e04d86398e.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/6b90cf39310446bb87dd37b9094c719cc437d23057f20800d62c67e04d86398e.jpg)

![8611c37483039c2558ac419923190516ca304c07e5e3c0d000d0d8c0294dad25.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/8611c37483039c2558ac419923190516ca304c07e5e3c0d000d0d8c0294dad25.jpg)

![8d5768741ca05d14cf03ac1238daad58d365a40b18deee80dac69065c03dce75.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/8d5768741ca05d14cf03ac1238daad58d365a40b18deee80dac69065c03dce75.jpg)

![8e9221307fbeefe57910e8ea6038c45b4b46e07af879e159d141bb899c76bf9e.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/8e9221307fbeefe57910e8ea6038c45b4b46e07af879e159d141bb899c76bf9e.jpg)

![a2c835f24be5fd19e5598ed412d40a57fa9bfa000ce7cd2a2baa708586215ef5.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/a2c835f24be5fd19e5598ed412d40a57fa9bfa000ce7cd2a2baa708586215ef5.jpg)

![a8de577caa50b000f3104b9248c513ad0de18214807a9c2b0afe64797d9c3b38.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/a8de577caa50b000f3104b9248c513ad0de18214807a9c2b0afe64797d9c3b38.jpg)

![b6715febc093318da8bbbef3c3796b5c86295d08d67fd89b2a5bc2438c335b41.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/b6715febc093318da8bbbef3c3796b5c86295d08d67fd89b2a5bc2438c335b41.jpg)

![ba9a1e3e2147a5b8ef6859ad5d5b19eaec71ae17085bc847c375aa91ed337e31.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/ba9a1e3e2147a5b8ef6859ad5d5b19eaec71ae17085bc847c375aa91ed337e31.jpg)

![c92696a9b8c94a74c3a4f751f11160d9e494d69520ab661ab7846f19270a2892.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/c92696a9b8c94a74c3a4f751f11160d9e494d69520ab661ab7846f19270a2892.jpg)

![fe5cc140fb0da3800a6cd2857637657203489ca92ea8abe97d160f46010f5f58.jpg](../icml_results/1382_TurboGS_%20Accelerating%203D%20Gaussian%20Splatting%20via%20Error-Guided%20Sparse%20Pixel%20Sampling%20and%20Optimization/tables/fe5cc140fb0da3800a6cd2857637657203489ca92ea8abe97d160f46010f5f58.jpg)

<a id="paper-1363"></a>

## RePack then Refine: Efficient Diffusion Transformers with Vision Foundation Models


### Images

![0dd0e4c638cc60d128bc44bf2ba58297fbce39cd7a9aa18f84aa84b99a776bac.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/0dd0e4c638cc60d128bc44bf2ba58297fbce39cd7a9aa18f84aa84b99a776bac.jpg)

![0e838f37e23134da98f78662433e0e92e9418e36f3f3838587e4a1f798e1a0b1.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/0e838f37e23134da98f78662433e0e92e9418e36f3f3838587e4a1f798e1a0b1.jpg)

![304e6230622af8ce25e27fa12103c1a3dbdc01cfeb23c01f750f8469d19d1bed.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/304e6230622af8ce25e27fa12103c1a3dbdc01cfeb23c01f750f8469d19d1bed.jpg)

![58c44fc3cee7a4c4dd93f568182bb135a99a37ea95e10316b380da0a2591b212.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/58c44fc3cee7a4c4dd93f568182bb135a99a37ea95e10316b380da0a2591b212.jpg)

![5fb5b7b0d90abe418756dae7bc41a86c174efcbb664c27a4581c2c7e36257704.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/5fb5b7b0d90abe418756dae7bc41a86c174efcbb664c27a4581c2c7e36257704.jpg)

![6a4f33ed6b4a1b4bc4f7b306106e5c887847211daea0a2aeb0e6d873d78292fc.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/6a4f33ed6b4a1b4bc4f7b306106e5c887847211daea0a2aeb0e6d873d78292fc.jpg)

![a62b7a20b545471eb2168625f56514281f02cd31780f2317be02d66ecc1e9289.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/a62b7a20b545471eb2168625f56514281f02cd31780f2317be02d66ecc1e9289.jpg)

![b23aaf47fd66f5a731586cea1e871a6548c594f02a46d0df9bfc23d4d8b2ce06.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/b23aaf47fd66f5a731586cea1e871a6548c594f02a46d0df9bfc23d4d8b2ce06.jpg)

![b51277e99ea9b77e7b174d12a79c26040082e1022a723e44d1539859bd0edbcb.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/b51277e99ea9b77e7b174d12a79c26040082e1022a723e44d1539859bd0edbcb.jpg)

![e3a7b81835183a6e809c327bdbd6352c81a94529f65416a11f3d006550502e80.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/images/e3a7b81835183a6e809c327bdbd6352c81a94529f65416a11f3d006550502e80.jpg)

### Tables

![22a0b3c338ba02167afc89f3a429d9eefb92043a1bc5f3eabbebadacf516dd0d.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/22a0b3c338ba02167afc89f3a429d9eefb92043a1bc5f3eabbebadacf516dd0d.jpg)

![260ceff350cea1e0a59e91d54f1b4782e6c4f0b0b543e4d6c362a9e48f1b5bb9.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/260ceff350cea1e0a59e91d54f1b4782e6c4f0b0b543e4d6c362a9e48f1b5bb9.jpg)

![29846e40232988e4a7cb4064af6c38a378e76f93a508ef044f8120c775365fa6.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/29846e40232988e4a7cb4064af6c38a378e76f93a508ef044f8120c775365fa6.jpg)

![2d2d0a8537da3f476098f7600d706f4c349d563ceca8afde9f71364c6f5f6774.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/2d2d0a8537da3f476098f7600d706f4c349d563ceca8afde9f71364c6f5f6774.jpg)

![5560a9bc6d1d440296394b2a7ac99d2388d2e8bf144b7797280b57c3b3c053c5.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/5560a9bc6d1d440296394b2a7ac99d2388d2e8bf144b7797280b57c3b3c053c5.jpg)

![84cf6cdbb762e4c37b56e56c409ca22cb7745c4cf86fa9a9b8f96c78981118a5.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/84cf6cdbb762e4c37b56e56c409ca22cb7745c4cf86fa9a9b8f96c78981118a5.jpg)

![8c9facbdee242a53866b1d47a40acede78baf266c3250e3ac0c0d8aa6519d714.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/8c9facbdee242a53866b1d47a40acede78baf266c3250e3ac0c0d8aa6519d714.jpg)

![9697cc519463d1b8d84c6a2d9b92f6663b5d324b514ea8a341b86a838cb8ab54.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/9697cc519463d1b8d84c6a2d9b92f6663b5d324b514ea8a341b86a838cb8ab54.jpg)

![a31a89427dfe70fdfa2704b322b841a60f58db0326915b1db4ef0b5b2345b59f.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/a31a89427dfe70fdfa2704b322b841a60f58db0326915b1db4ef0b5b2345b59f.jpg)

![a5657bc381a4eace8af6de037bf9d5b989b1a7951d5f88342138aa6d8c6754f5.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/a5657bc381a4eace8af6de037bf9d5b989b1a7951d5f88342138aa6d8c6754f5.jpg)

![b9cd42341ff842f6134917184b4e305251b6bfd450cef170c4d8ad07c780f66a.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/b9cd42341ff842f6134917184b4e305251b6bfd450cef170c4d8ad07c780f66a.jpg)

![b9cd75496a0fd5aecbaaa63b17d48a434627ae1d3dd45f39c15ea4fc730c59b4.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/b9cd75496a0fd5aecbaaa63b17d48a434627ae1d3dd45f39c15ea4fc730c59b4.jpg)

![cb46fe07d38b9e62fa0dee11fd2f4591dcd72911aa046144fe8d91c733a75737.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/cb46fe07d38b9e62fa0dee11fd2f4591dcd72911aa046144fe8d91c733a75737.jpg)

![d4b86d95ed35f615ebda3db985474550144f79ded660e4d8d4da23db35f3aa88.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/d4b86d95ed35f615ebda3db985474550144f79ded660e4d8d4da23db35f3aa88.jpg)

![e0d8cb695be780e671028c1608cef06a326b785ef66afb2c4e321f068b293354.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/e0d8cb695be780e671028c1608cef06a326b785ef66afb2c4e321f068b293354.jpg)

![f2e7e50d055638ff9ca7319cef4a1718053eee369cfcb22205244118374de4dd.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/f2e7e50d055638ff9ca7319cef4a1718053eee369cfcb22205244118374de4dd.jpg)

![f5feb1de7f2d65460daee750fb446f3492ae93d357b96dbde7f6b4cc27d7f0b3.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/f5feb1de7f2d65460daee750fb446f3492ae93d357b96dbde7f6b4cc27d7f0b3.jpg)

![fbe761e6a3e40ed815c90213025ad2e2adbae7c9fd9389001b9cba24d7bb9cf2.jpg](../icml_results/1383_RePack%20then%20Refine_%20Efficient%20Diffusion%20Transformers%20with%20Vision%20Foundation%20Models/tables/fbe761e6a3e40ed815c90213025ad2e2adbae7c9fd9389001b9cba24d7bb9cf2.jpg)

<a id="paper-1364"></a>

## Mean Flow Policy Optimization


### Images

![10b5b2f98ec1231ef2c3668ead57d8a635b47fb3750b20a8591e85f79486ea7c.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/10b5b2f98ec1231ef2c3668ead57d8a635b47fb3750b20a8591e85f79486ea7c.jpg)

![1a6d24e4ec425008736b9b76be311797897e334c4d5928fdd0bb84a175300d5d.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/1a6d24e4ec425008736b9b76be311797897e334c4d5928fdd0bb84a175300d5d.jpg)

![3d68a35c7f61574e87df9c3ab8d32a6da2ad1a401cdc8767c637aae06c41d5fb.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/3d68a35c7f61574e87df9c3ab8d32a6da2ad1a401cdc8767c637aae06c41d5fb.jpg)

![43e55cbe87925f0334ea4d37ce6b65a0751793401a21a43354861204b1368d3d.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/43e55cbe87925f0334ea4d37ce6b65a0751793401a21a43354861204b1368d3d.jpg)

![467593292fff6c1f74f199041acf33fbe2f93f71768ee3b0a581f4dbed25a466.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/467593292fff6c1f74f199041acf33fbe2f93f71768ee3b0a581f4dbed25a466.jpg)

![55df2c3bb0bbb29478d5fa1dc6e481b4464309b109b4fb2ce010ef0d6f4dccf5.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/55df2c3bb0bbb29478d5fa1dc6e481b4464309b109b4fb2ce010ef0d6f4dccf5.jpg)

![73c32c3e8686768d9d7ae4e082de3b4275280618f32c2eea9534c8803d83a07a.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/73c32c3e8686768d9d7ae4e082de3b4275280618f32c2eea9534c8803d83a07a.jpg)

![8bc061d8f58bb292ed80653d39b9ffe943731c84931323d383e2b271bd59459e.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/8bc061d8f58bb292ed80653d39b9ffe943731c84931323d383e2b271bd59459e.jpg)

![a56701a442b406f8bb5a4a08c9a5bc6b691a277ad4734e47b42bcc8ec6664d25.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/a56701a442b406f8bb5a4a08c9a5bc6b691a277ad4734e47b42bcc8ec6664d25.jpg)

![b50a2d33c71f5c2a9b659c33064b62cf911f4f42239316c0151ee71bec81a824.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/b50a2d33c71f5c2a9b659c33064b62cf911f4f42239316c0151ee71bec81a824.jpg)

![c6a17b1b95d348c3ff9272fe591099225caf14014a2077b27e95cc6b1aa0a9f8.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/c6a17b1b95d348c3ff9272fe591099225caf14014a2077b27e95cc6b1aa0a9f8.jpg)

![ef69511b492cbfd78d432f14420ed8679cb084f29580ad5e8f2f03f3925d444d.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/images/ef69511b492cbfd78d432f14420ed8679cb084f29580ad5e8f2f03f3925d444d.jpg)

### Tables

![30b3ebc5536bdaa566b7df0c9079bae96f5531405b4d9f909b89e3bb1e00389d.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/tables/30b3ebc5536bdaa566b7df0c9079bae96f5531405b4d9f909b89e3bb1e00389d.jpg)

![a00618f118b528e68c94f598408e9f4400da2f85e84f54c2ed08142cf2221fd3.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/tables/a00618f118b528e68c94f598408e9f4400da2f85e84f54c2ed08142cf2221fd3.jpg)

![b20a65765cd4c0596b464d1a409f6e0f62701ea2ecf7c7844d77fb1fd2803b50.jpg](../icml_results/1384_Mean%20Flow%20Policy%20Optimization/tables/b20a65765cd4c0596b464d1a409f6e0f62701ea2ecf7c7844d77fb1fd2803b50.jpg)

<a id="paper-1365"></a>

## Algorithmic Recourse of In-Context Learning for Tabular Data


### Images

![3bf5fc1cea4a7d7b7cff2340007bf1e3d370c3d42b9509f0e1e438cf41e2593c.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/3bf5fc1cea4a7d7b7cff2340007bf1e3d370c3d42b9509f0e1e438cf41e2593c.jpg)

![3c908c9f63785846cdfe9975abafb6fa4b64cd145bb642698d3b76296a3fcd91.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/3c908c9f63785846cdfe9975abafb6fa4b64cd145bb642698d3b76296a3fcd91.jpg)

![6b55db6729d60ff9a443dc4269c445ea0a96cf11ec738e88533e74dcffaac8a6.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/6b55db6729d60ff9a443dc4269c445ea0a96cf11ec738e88533e74dcffaac8a6.jpg)

![9c3afde84dfdd77adae83e43dc1b0a23c041bec98d9ad7595b910f8a2128e87f.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/9c3afde84dfdd77adae83e43dc1b0a23c041bec98d9ad7595b910f8a2128e87f.jpg)

![c43e66c206ad19ddd723a9000c28e6f94a27cdd58b737a8e05dd7bf8354548dd.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/c43e66c206ad19ddd723a9000c28e6f94a27cdd58b737a8e05dd7bf8354548dd.jpg)

![cc0d7deae06284cfd4c10b16d906ac9cfa33479c5c0488946da097005a7f277b.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/cc0d7deae06284cfd4c10b16d906ac9cfa33479c5c0488946da097005a7f277b.jpg)

![f965dfb5009638fed665d39b6a881f3a038029fb02ea012aa5dcd2fe8e0466cb.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/f965dfb5009638fed665d39b6a881f3a038029fb02ea012aa5dcd2fe8e0466cb.jpg)

![fc3499e54185834f39aa976a3dbceca76e5c726ba98dfcc91107f5f6d4ba0f7b.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/images/fc3499e54185834f39aa976a3dbceca76e5c726ba98dfcc91107f5f6d4ba0f7b.jpg)

### Tables

![05cebf50224f7fde9c5724a85d9547fee19b87774b732de712c19e1cc6177c60.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/05cebf50224f7fde9c5724a85d9547fee19b87774b732de712c19e1cc6177c60.jpg)

![066e28876750aba1a3f4b5c6bd9cc9014897b78ab89ce68bba82b21d3f4040cb.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/066e28876750aba1a3f4b5c6bd9cc9014897b78ab89ce68bba82b21d3f4040cb.jpg)

![2dc64c4c182bd57898e152a90ed3336008d3e3d67862d87a9f552273e9a47fdc.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/2dc64c4c182bd57898e152a90ed3336008d3e3d67862d87a9f552273e9a47fdc.jpg)

![4e7e0d043e8cb5ebc40cfa1ae11fdb62a8e5927fdcfbf2471ee41f1655cfd5cb.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/4e7e0d043e8cb5ebc40cfa1ae11fdb62a8e5927fdcfbf2471ee41f1655cfd5cb.jpg)

![a26396c83950939231b351f91f9b34226db0e0429741fc91479789c87b824036.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/a26396c83950939231b351f91f9b34226db0e0429741fc91479789c87b824036.jpg)

![b41dc1368d3c033785fde5e0bd23a17372d3f58f4c263e3b863a3458db66c8c3.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/b41dc1368d3c033785fde5e0bd23a17372d3f58f4c263e3b863a3458db66c8c3.jpg)

![f49a7d9074dbcf187f34ff0155014f5beaf708a7c9606bb2e62f68af16608fec.jpg](../icml_results/1385_Algorithmic%20Recourse%20of%20In-Context%20Learning%20for%20Tabular%20Data/tables/f49a7d9074dbcf187f34ff0155014f5beaf708a7c9606bb2e62f68af16608fec.jpg)

<a id="paper-1366"></a>

## Evaluating LLMs When They Do Not Know the Answer: Statistical Evaluation of Mathematical Reasoning via Comparative Signals


### Images

![12d3a2f20118f1313c240bc39ba3fb283b1b38af3d3dcf7e5392a95f979f6593.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/images/12d3a2f20118f1313c240bc39ba3fb283b1b38af3d3dcf7e5392a95f979f6593.jpg)

![265aea8a221d6a97c2c198ea4d0ab1f5fc380e881a8575793353a6342ac8b00f.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/images/265aea8a221d6a97c2c198ea4d0ab1f5fc380e881a8575793353a6342ac8b00f.jpg)

![8022279aa123e561f7c3f326c201c385ef9bc08e13f2fa0a5065fb75cd94fd02.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/images/8022279aa123e561f7c3f326c201c385ef9bc08e13f2fa0a5065fb75cd94fd02.jpg)

![bf179a22d7c971b053b43ee9a868f672cf98a694bef587aed3f0b4ce9a8e4094.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/images/bf179a22d7c971b053b43ee9a868f672cf98a694bef587aed3f0b4ce9a8e4094.jpg)

![d9437b6b19b206c8e90428515c897ebac7c644950e681c0d08bfe3e56815d833.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/images/d9437b6b19b206c8e90428515c897ebac7c644950e681c0d08bfe3e56815d833.jpg)

### Tables

![34105be828c8b2b8ac4684564b479521c093f0d0a80761a8520b174291d66cf8.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/34105be828c8b2b8ac4684564b479521c093f0d0a80761a8520b174291d66cf8.jpg)

![5958f5314ebb402dc9f2472c038a8f8c1ac4a85b0731eee42c3b42380d9f3dc2.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/5958f5314ebb402dc9f2472c038a8f8c1ac4a85b0731eee42c3b42380d9f3dc2.jpg)

![89bfd2ebff4ff4a43eb5b6ccf73119fd991098d17466ff87c9e5ca0984fb2b1b.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/89bfd2ebff4ff4a43eb5b6ccf73119fd991098d17466ff87c9e5ca0984fb2b1b.jpg)

![8c53246e0c96c69eaf45aea2e4f86d397e005ffb0c6c68230ce4b5502ef23b4a.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/8c53246e0c96c69eaf45aea2e4f86d397e005ffb0c6c68230ce4b5502ef23b4a.jpg)

![abcba8d54d397ee352d32da5a533aab4183560f1fbeeb37465e9b0997c9ea4bd.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/abcba8d54d397ee352d32da5a533aab4183560f1fbeeb37465e9b0997c9ea4bd.jpg)

![c2f65e5000ee9195dc8b7154e8db75ff54809c916c08b0741adf38bb2679fb4f.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/c2f65e5000ee9195dc8b7154e8db75ff54809c916c08b0741adf38bb2679fb4f.jpg)

![cb9f4313872b250768775c0135da8160d05038eecbb709e84982a98574f2648a.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/cb9f4313872b250768775c0135da8160d05038eecbb709e84982a98574f2648a.jpg)

![d22667ed31b24000543182ce914169c2d3ea486c6156e185f91aa54b61430fad.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/d22667ed31b24000543182ce914169c2d3ea486c6156e185f91aa54b61430fad.jpg)

![d8bc094241acdda97ed6c6dc4b693547833eb37866a208f593627e4570d20370.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/d8bc094241acdda97ed6c6dc4b693547833eb37866a208f593627e4570d20370.jpg)

![f119dcb51253d4c520b3f69817a1abe76424ae740c9452764da39643ca4d56ad.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/f119dcb51253d4c520b3f69817a1abe76424ae740c9452764da39643ca4d56ad.jpg)

![f9ab43df3137610819ffa6cd6f447b3c01ee5ad01e9cae81df9471320de23ee6.jpg](../icml_results/1386_Evaluating%20LLMs%20When%20They%20Do%20Not%20Know%20the%20Answer_%20Statistical%20Evaluation%20of%20Mathematical%20Reasoning%20v/tables/f9ab43df3137610819ffa6cd6f447b3c01ee5ad01e9cae81df9471320de23ee6.jpg)

<a id="paper-1367"></a>

## Bits That Count: Quantifying and Predicting Capabilities of Language Models


### Images

![0c83e70e4200be2d4303bc4d2df3bec4d015518837c29cf05d39ddc7666d7907.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/0c83e70e4200be2d4303bc4d2df3bec4d015518837c29cf05d39ddc7666d7907.jpg)

![1d8dcf28c8fda218888bb8da3e94f0b2103a97c19f0d27f2e68bfb8b798d9521.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/1d8dcf28c8fda218888bb8da3e94f0b2103a97c19f0d27f2e68bfb8b798d9521.jpg)

![355025156ba42cf6f8d26371d58b0d0e577e2f534c1897e6961d1410bcb4e2a8.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/355025156ba42cf6f8d26371d58b0d0e577e2f534c1897e6961d1410bcb4e2a8.jpg)

![39865ca382a77d1027361b3348c5ca97e08f9ae3c8a2486a4ddf4f1162937d0f.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/39865ca382a77d1027361b3348c5ca97e08f9ae3c8a2486a4ddf4f1162937d0f.jpg)

![39b0a2e7ebfcf56bf09ee24b5cba23d6c18470ffcdebec8f1473ebf08186ed1b.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/39b0a2e7ebfcf56bf09ee24b5cba23d6c18470ffcdebec8f1473ebf08186ed1b.jpg)

![3d88bc9e768d6635d4c29adb0743c983adfbd8ced4c59999df50913e54169c59.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/3d88bc9e768d6635d4c29adb0743c983adfbd8ced4c59999df50913e54169c59.jpg)

![8d32e523baacd106b58e9b150f200a2f56d3c06fa9bf72a9a461bbbddcd04ee7.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/8d32e523baacd106b58e9b150f200a2f56d3c06fa9bf72a9a461bbbddcd04ee7.jpg)

![a79eabee0304f132218817b38daf3bab04d7521d48287e77ea72da4a7b7924e6.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/a79eabee0304f132218817b38daf3bab04d7521d48287e77ea72da4a7b7924e6.jpg)

![c122387f8da04357677a94c5cebdd3f4564d225cea86ae4566293a88f73aac13.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/c122387f8da04357677a94c5cebdd3f4564d225cea86ae4566293a88f73aac13.jpg)

![f1277f062fde24f028946d814d1dfb4f012bbd432e3caea1820eb07d8cf05cd5.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/images/f1277f062fde24f028946d814d1dfb4f012bbd432e3caea1820eb07d8cf05cd5.jpg)

### Tables

![0af3d1d23401506ae5a3827a408a915e84d1c7bef690eb41eaa88b58f9ecb2da.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/0af3d1d23401506ae5a3827a408a915e84d1c7bef690eb41eaa88b58f9ecb2da.jpg)

![137f29630fee36ed664dcef834b7f4b7984b048215b381828bb08f8625b93dec.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/137f29630fee36ed664dcef834b7f4b7984b048215b381828bb08f8625b93dec.jpg)

![2706e30c755d673ad1d14484a1e0f25682d4d0dc76bd4c9fbdaa1733af116003.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/2706e30c755d673ad1d14484a1e0f25682d4d0dc76bd4c9fbdaa1733af116003.jpg)

![2971e7f43b11b498516397f48e124fc0d17e92104efb5465c99a13f977b08828.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/2971e7f43b11b498516397f48e124fc0d17e92104efb5465c99a13f977b08828.jpg)

![2d3b9f87784c87702eda0dccbc1c20a181c461ea81156aa30a5d762ea52af8a0.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/2d3b9f87784c87702eda0dccbc1c20a181c461ea81156aa30a5d762ea52af8a0.jpg)

![757f76f0617f4d9ebf51eab609aaa60909054b66d45bca1f68d47d8dd45b362e.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/757f76f0617f4d9ebf51eab609aaa60909054b66d45bca1f68d47d8dd45b362e.jpg)

![863429d11462f9d497d42461062e3b92bd63a58f2bec785e6b26f33d4514c9db.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/863429d11462f9d497d42461062e3b92bd63a58f2bec785e6b26f33d4514c9db.jpg)

![9f88acd4a2ab3dc394f76ce440c5fa0c4a477b6db9161945f2fdf6997128e7d5.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/9f88acd4a2ab3dc394f76ce440c5fa0c4a477b6db9161945f2fdf6997128e7d5.jpg)

![c5a48834f402d4afe74281bae5e3554cf087bc0321781725c6e5dfe2465d33d2.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/c5a48834f402d4afe74281bae5e3554cf087bc0321781725c6e5dfe2465d33d2.jpg)

![d839a634a2c0f1f5f9f6c73f6040385515827a2c2e21f5acdef223a2093c2511.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/d839a634a2c0f1f5f9f6c73f6040385515827a2c2e21f5acdef223a2093c2511.jpg)

![dcfda0fd16ad399db8664c866e6f4ba1e2cdff138f5d74df2b8df81eb179ba28.jpg](../icml_results/1387_Bits%20That%20Count_%20Quantifying%20and%20Predicting%20Capabilities%20of%20Language%20Models/tables/dcfda0fd16ad399db8664c866e6f4ba1e2cdff138f5d74df2b8df81eb179ba28.jpg)

<a id="paper-1368"></a>

## Q-Flow: Stable and Expressive Reinforcement Learning with Flow-based Policy


### Images

![0605ca4144ed772fe4667490a25bf7e862f9871c3c18131b05be42f93f1b8468.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/0605ca4144ed772fe4667490a25bf7e862f9871c3c18131b05be42f93f1b8468.jpg)

![07b671eda4a9d597a0edeb68e24393200b4aa96579ff1ab26a48eb44f27b6e01.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/07b671eda4a9d597a0edeb68e24393200b4aa96579ff1ab26a48eb44f27b6e01.jpg)

![29dfd308f1da02b06970d49b962cdca6815692cf7289609812ca359b732ecc09.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/29dfd308f1da02b06970d49b962cdca6815692cf7289609812ca359b732ecc09.jpg)

![44c441e5de8063e0559bbf2d5ae31b4aad167b1fc3cb88886c5a10b3a5c87f83.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/44c441e5de8063e0559bbf2d5ae31b4aad167b1fc3cb88886c5a10b3a5c87f83.jpg)

![44ea4ac7eca05684d1738c1f9fbec28f71530074280d5c08c3c54c1f084c382c.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/44ea4ac7eca05684d1738c1f9fbec28f71530074280d5c08c3c54c1f084c382c.jpg)

![4eff9c11893b93e7426c4b5ec1da361354b3cd85d67d761331c3a0b5c317f9de.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/4eff9c11893b93e7426c4b5ec1da361354b3cd85d67d761331c3a0b5c317f9de.jpg)

![4f68dc7d5aeff0df355dd42fd6a18e9408f33bea48f20e0b7130d9cffe29b61a.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/4f68dc7d5aeff0df355dd42fd6a18e9408f33bea48f20e0b7130d9cffe29b61a.jpg)

![64da00ffd32957656d9658aa139827ed07b26c94a38138d6b0d17c09c1de2576.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/64da00ffd32957656d9658aa139827ed07b26c94a38138d6b0d17c09c1de2576.jpg)

![684faeb3330919f89bfc2d986b8bddabdfe329f8e7cc4a0a85027fc1d0ebe2a4.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/684faeb3330919f89bfc2d986b8bddabdfe329f8e7cc4a0a85027fc1d0ebe2a4.jpg)

![75468d98477252ac5e9abdd92a30c15c32c5030ac14db3133d65e0a235a6e9f7.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/75468d98477252ac5e9abdd92a30c15c32c5030ac14db3133d65e0a235a6e9f7.jpg)

![87d87daa028d54de04f543cb08ab3134f0f040c63bc4e04f379ae034444cd6ff.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/87d87daa028d54de04f543cb08ab3134f0f040c63bc4e04f379ae034444cd6ff.jpg)

![89a3340a9094baad12191d6d9810fed63f52a68bf79ccf1fc82b219a71a50f57.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/89a3340a9094baad12191d6d9810fed63f52a68bf79ccf1fc82b219a71a50f57.jpg)

![a15d2401f3d9261f9919ee7e0d476f7bd26890e6b2036b1afcd7d5e3e26a96d5.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/a15d2401f3d9261f9919ee7e0d476f7bd26890e6b2036b1afcd7d5e3e26a96d5.jpg)

![adf39f6678c4f8be4ab17e5b9b1fa3ed6a9202c526f2a8c809fd609eee9bae1c.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/adf39f6678c4f8be4ab17e5b9b1fa3ed6a9202c526f2a8c809fd609eee9bae1c.jpg)

![b02f2a4e8e97df4228b878cf1b5166ea280baaf722eae5efeac5287580513e08.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/b02f2a4e8e97df4228b878cf1b5166ea280baaf722eae5efeac5287580513e08.jpg)

![b14084677f56d210155d88afad4e4e57c6f2fe6840a0bac318133e4dc7108ec7.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/b14084677f56d210155d88afad4e4e57c6f2fe6840a0bac318133e4dc7108ec7.jpg)

![cc20b436cddba7fafe0c4bfdbc54882e9c70a417c3288cace0cbc70cfca30e70.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/cc20b436cddba7fafe0c4bfdbc54882e9c70a417c3288cace0cbc70cfca30e70.jpg)

![d85eca9a8dbbb24fdfcb7e95507fc9a67068d16e366f699a938cba4d58752dc4.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/images/d85eca9a8dbbb24fdfcb7e95507fc9a67068d16e366f699a938cba4d58752dc4.jpg)

### Tables

![091aacd97500ef74bf29ad16e1a493cb2b627cfa3ea215bd98a867e65e07501b.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/091aacd97500ef74bf29ad16e1a493cb2b627cfa3ea215bd98a867e65e07501b.jpg)

![1062aa835fa2cdc87d66948c9841bb2a8404785267e15a46b10a61309fe6d588.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/1062aa835fa2cdc87d66948c9841bb2a8404785267e15a46b10a61309fe6d588.jpg)

![278c481b134a6e87e95a92efcf71f47714dd9a7813836c4a23989378b3cebefd.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/278c481b134a6e87e95a92efcf71f47714dd9a7813836c4a23989378b3cebefd.jpg)

![285eac7c336204993332647fc270bc0d4cba4b6970a3de1b311b1f12b5c11c3c.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/285eac7c336204993332647fc270bc0d4cba4b6970a3de1b311b1f12b5c11c3c.jpg)

![331a42eff8774b0c610ef4fec9742604da0ece163494fbfb5ab3bacf26dc3d93.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/331a42eff8774b0c610ef4fec9742604da0ece163494fbfb5ab3bacf26dc3d93.jpg)

![49fd25456e1e6c8b699182d90df2d81f141f6cacfb4723bc2e758ad80be9d681.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/49fd25456e1e6c8b699182d90df2d81f141f6cacfb4723bc2e758ad80be9d681.jpg)

![53485c31b61009a83a7cbae0201cb09482d93125577c0bcae0411c182af5d6c0.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/53485c31b61009a83a7cbae0201cb09482d93125577c0bcae0411c182af5d6c0.jpg)

![72bb0fa1c867e131cc1ed2f8c311abc5fd94f49e109a47d36b70570b6d07ae9e.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/72bb0fa1c867e131cc1ed2f8c311abc5fd94f49e109a47d36b70570b6d07ae9e.jpg)

![8fd5e3b854bf1f855291e677b2bd4b7c834d18dc9b73b76592df7d5720f34e88.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/8fd5e3b854bf1f855291e677b2bd4b7c834d18dc9b73b76592df7d5720f34e88.jpg)

![9755e359199c19e783aa583e595635b40871ba1e0821139580446f51e1856e56.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/9755e359199c19e783aa583e595635b40871ba1e0821139580446f51e1856e56.jpg)

![a54f9a9dc32275683cef3ea2abb2a0f27e521e32f984169fef63e0705edb748b.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/a54f9a9dc32275683cef3ea2abb2a0f27e521e32f984169fef63e0705edb748b.jpg)

![bd0e7c34707670eff194bea25d34c38b499df6dac4e65a59706b1c65711c00fc.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/bd0e7c34707670eff194bea25d34c38b499df6dac4e65a59706b1c65711c00fc.jpg)

![d12ddb9cf2f9e2060fd9ede973c0a1db4c03aeb35f9a9ff619b11fb7efa27614.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/d12ddb9cf2f9e2060fd9ede973c0a1db4c03aeb35f9a9ff619b11fb7efa27614.jpg)

![df6d2d962882a41cce764b611e0573a730611363a171dd00468112a8c7377d1a.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/df6d2d962882a41cce764b611e0573a730611363a171dd00468112a8c7377d1a.jpg)

![f07588b8add17b28847de03f20443f741bd463cff7d7aac708c9cea16522129d.jpg](../icml_results/1388_Q-Flow_%20Stable%20and%20Expressive%20Reinforcement%20Learning%20with%20Flow-based%20Policy/tables/f07588b8add17b28847de03f20443f741bd463cff7d7aac708c9cea16522129d.jpg)

<a id="paper-1369"></a>

## Why ReLU? A Bit-Model Dichotomy for Deep Network Training


### Images

![79475c59fdccff20663557282d0c3b8146c39e7c64a349a5dfa2987249311013.jpg](../icml_results/1389_Why%20ReLU_%20A%20Bit-Model%20Dichotomy%20for%20Deep%20Network%20Training/images/79475c59fdccff20663557282d0c3b8146c39e7c64a349a5dfa2987249311013.jpg)

![d46f7d9265b609c94159f70673c61ab0c86cf4395d5bd535123751707543f888.jpg](../icml_results/1389_Why%20ReLU_%20A%20Bit-Model%20Dichotomy%20for%20Deep%20Network%20Training/images/d46f7d9265b609c94159f70673c61ab0c86cf4395d5bd535123751707543f888.jpg)

![e032a760b919ecb508f0251b566c407ded48ab7d858e491c0dde52e2fa367ccc.jpg](../icml_results/1389_Why%20ReLU_%20A%20Bit-Model%20Dichotomy%20for%20Deep%20Network%20Training/images/e032a760b919ecb508f0251b566c407ded48ab7d858e491c0dde52e2fa367ccc.jpg)

### Tables

![e110dff73ee99c52026b38af9c8bbf6b636f6908c9c887c904707ad0fb95ed61.jpg](../icml_results/1389_Why%20ReLU_%20A%20Bit-Model%20Dichotomy%20for%20Deep%20Network%20Training/tables/e110dff73ee99c52026b38af9c8bbf6b636f6908c9c887c904707ad0fb95ed61.jpg)

<a id="paper-1370"></a>

## Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction


### Images

![309e398e482872174068027e12fa62f5385b08afe548054f07931a0a26e35fd7.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/309e398e482872174068027e12fa62f5385b08afe548054f07931a0a26e35fd7.jpg)

![582e03c0fd03c4ba0f2fbf67162e6fa161c8c473a8da3e4a2ec500b5c412a9d2.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/582e03c0fd03c4ba0f2fbf67162e6fa161c8c473a8da3e4a2ec500b5c412a9d2.jpg)

![61be2a047459b5174472473d969278bb9571aae4629ac47f5ac35ce7f4dc2169.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/61be2a047459b5174472473d969278bb9571aae4629ac47f5ac35ce7f4dc2169.jpg)

![97f831c9c173b1715c2eed407ec0b546599532212516a1ebfeaef1349b25ac5f.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/97f831c9c173b1715c2eed407ec0b546599532212516a1ebfeaef1349b25ac5f.jpg)

![ab1419ce311985502a391f15018b05498b46d6c7d1ca24bc11a0a307c7d5dd6a.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/ab1419ce311985502a391f15018b05498b46d6c7d1ca24bc11a0a307c7d5dd6a.jpg)

![b3e5f0246a3ce6367b2d3f0464f88013760827fc9e868b310bddb6538f7a3ba6.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/b3e5f0246a3ce6367b2d3f0464f88013760827fc9e868b310bddb6538f7a3ba6.jpg)

![c7fd9f20dc477a2f96584d01274afb1f71e488fbac9fc8ea386fb5ffc2d485e1.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/c7fd9f20dc477a2f96584d01274afb1f71e488fbac9fc8ea386fb5ffc2d485e1.jpg)

![c84ac8c5713ab1f310f6b26b1d94d58f6bf9dcfeb64da5054c63caa1a75ee073.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/c84ac8c5713ab1f310f6b26b1d94d58f6bf9dcfeb64da5054c63caa1a75ee073.jpg)

![e1e3c2ac8422cd3b4d0669b9127d4991f67217200fcb52cd6550975df1ea9fa4.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/e1e3c2ac8422cd3b4d0669b9127d4991f67217200fcb52cd6550975df1ea9fa4.jpg)

![fa394c0cb9a22e2067bdea3b94da38bf1056ecac1c456b72f3116ab8a62dc13b.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/images/fa394c0cb9a22e2067bdea3b94da38bf1056ecac1c456b72f3116ab8a62dc13b.jpg)

### Tables

![1dd3af22f8c711ef97e5e3cf079a30180b7f64612d0702e5dcd5576a061dc2bc.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/1dd3af22f8c711ef97e5e3cf079a30180b7f64612d0702e5dcd5576a061dc2bc.jpg)

![2080fb43753a462e2739a39b033b9ba312e810a8cf9d245f72bf96be804e68b9.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/2080fb43753a462e2739a39b033b9ba312e810a8cf9d245f72bf96be804e68b9.jpg)

![25fdc64684930305e346a1acc0ac61554a85b271c31180514bc7ef9e865662c5.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/25fdc64684930305e346a1acc0ac61554a85b271c31180514bc7ef9e865662c5.jpg)

![49d57b13cb607f74554a05eaa73dcc39a45bdf8c002a7616fea8a1e972e1b5a8.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/49d57b13cb607f74554a05eaa73dcc39a45bdf8c002a7616fea8a1e972e1b5a8.jpg)

![54b30b49280ffb31f983cd039ee1cc016f437e3eb4665c229599d855dcf77542.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/54b30b49280ffb31f983cd039ee1cc016f437e3eb4665c229599d855dcf77542.jpg)

![6147294288218ea41044d0105ee26e1db0b55a430bb4ce15ec55c9529b981657.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/6147294288218ea41044d0105ee26e1db0b55a430bb4ce15ec55c9529b981657.jpg)

![9016e62fbd57f13f9c840c86b70dc6439b3014188cdb7c5663ab4d2607947a2e.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/9016e62fbd57f13f9c840c86b70dc6439b3014188cdb7c5663ab4d2607947a2e.jpg)

![a04d9ef1141136fdb4a405e9678f4c6d0fa04d57014518348b18fd2b388b3f4f.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/a04d9ef1141136fdb4a405e9678f4c6d0fa04d57014518348b18fd2b388b3f4f.jpg)

![a5b72c1f38ad6e267bfda93d1bd3c0001e5b49e762e42f2aec6e0ebb45d3058e.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/a5b72c1f38ad6e267bfda93d1bd3c0001e5b49e762e42f2aec6e0ebb45d3058e.jpg)

![a5cef9ccbb5f2cc34090b419b478c59c90fb696cd40606c13af5b81f974762e7.jpg](../icml_results/1390_Plug-and-Play%20Guidance%20for%20Discrete%20Diffusion%20Models%20via%20Gradient-Informed%20Logit%20Correction/tables/a5cef9ccbb5f2cc34090b419b478c59c90fb696cd40606c13af5b81f974762e7.jpg)

<a id="paper-1371"></a>

## DNA: Uncovering Universal Latent Forgery Knowledge


### Images

![0b9db4d2f98de6adba88d27d5768970cd3f488a0122e32de2f691c8ec8c543f7.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/0b9db4d2f98de6adba88d27d5768970cd3f488a0122e32de2f691c8ec8c543f7.jpg)

![19954f94504cd56b42f3fbae51099c0f330d2a7218f2b7ddbb06ebd2acd1a533.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/19954f94504cd56b42f3fbae51099c0f330d2a7218f2b7ddbb06ebd2acd1a533.jpg)

![1a376c5f23f0bf8e6894ea538077b2bba1785a64c22bc3784a461738e897ba20.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/1a376c5f23f0bf8e6894ea538077b2bba1785a64c22bc3784a461738e897ba20.jpg)

![1ba650a6b006fe674e7fb5b2b18bfa9c129ae99aba97558a0e0061882bc214cd.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/1ba650a6b006fe674e7fb5b2b18bfa9c129ae99aba97558a0e0061882bc214cd.jpg)

![205fda152694857aa17aebe50d7f86e5bef76f7065f8dfd6f64f46bb3700e3ca.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/205fda152694857aa17aebe50d7f86e5bef76f7065f8dfd6f64f46bb3700e3ca.jpg)

![3387877c965609aefe54d9232e1370cb5dac2d89e4f52f0f94a92f48f36c682f.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/3387877c965609aefe54d9232e1370cb5dac2d89e4f52f0f94a92f48f36c682f.jpg)

![5bd87ab74a48ca26a126c4c72acb548827b43151b77277eb532a89d3859612b3.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/5bd87ab74a48ca26a126c4c72acb548827b43151b77277eb532a89d3859612b3.jpg)

![70d8f25d8b5d9576d1fdc4b152d2c17a75ae9ad593609ff0f2418de73b30876f.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/70d8f25d8b5d9576d1fdc4b152d2c17a75ae9ad593609ff0f2418de73b30876f.jpg)

![a24732e494a8fc793f6198621d3133cf2d6662eb31fca81565c787320bd24566.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/a24732e494a8fc793f6198621d3133cf2d6662eb31fca81565c787320bd24566.jpg)

![bb8fccb87909dc71387f99f15acc9c0c8d6d0909f7c090748f2a0c42d740e00b.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/bb8fccb87909dc71387f99f15acc9c0c8d6d0909f7c090748f2a0c42d740e00b.jpg)

![c003dbc1413f1e501d3734f1be4f8ec782e528565ad8bfa56ce5627bf8c2bdc2.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/c003dbc1413f1e501d3734f1be4f8ec782e528565ad8bfa56ce5627bf8c2bdc2.jpg)

![c92a5e7f66cb7104ecb600589cf2194cf046822ec22959d9e23f25410485b1df.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/c92a5e7f66cb7104ecb600589cf2194cf046822ec22959d9e23f25410485b1df.jpg)

![cdbf6f8e83be215880ad4dcb5d769b33cc85cff1a3826d4292929036217c35b8.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/cdbf6f8e83be215880ad4dcb5d769b33cc85cff1a3826d4292929036217c35b8.jpg)

![d0b82cd1415619f598237b221a8da735ea0b0ebc4382c47b930eb96a28a4c895.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/d0b82cd1415619f598237b221a8da735ea0b0ebc4382c47b930eb96a28a4c895.jpg)

![d116d0067efd50b3216f1654a8dd10c10e08813837729c9e8904906dfafaa980.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/d116d0067efd50b3216f1654a8dd10c10e08813837729c9e8904906dfafaa980.jpg)

![d76c682a5e05e37d12b6293411dea69304aa329f85c2405153ff108c0de790a8.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/images/d76c682a5e05e37d12b6293411dea69304aa329f85c2405153ff108c0de790a8.jpg)

### Tables

![3da5bfd2e5b4e6c6390f6496028e123a4d62e294bd320cf2a145a55f66e048a5.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/tables/3da5bfd2e5b4e6c6390f6496028e123a4d62e294bd320cf2a145a55f66e048a5.jpg)

![6b6052617925c5e6f0dd1bd1633107bca01d9dbba9ae490753b0371808f769e9.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/tables/6b6052617925c5e6f0dd1bd1633107bca01d9dbba9ae490753b0371808f769e9.jpg)

![6c327a40188cc0c0425d84eb01e119ab13b91d502e44f34b418b13915e25d74f.jpg](../icml_results/1391_DNA_%20Uncovering%20Universal%20Latent%20Forgery%20Knowledge/tables/6c327a40188cc0c0425d84eb01e119ab13b91d502e44f34b418b13915e25d74f.jpg)

<a id="paper-1372"></a>

## Plan Then Action: High-Level Planning Guidance Reinforcement Learning for LLM Reasoning


### Images

![28da1b45d36885e8f5083b4022d0c80a3b15b4ce39aefbc2d951386062ca1013.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/28da1b45d36885e8f5083b4022d0c80a3b15b4ce39aefbc2d951386062ca1013.jpg)

![6448bd577d6a9bf7530db0e2917a47a207acb16b2f9c9e41232e0441b733152e.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/6448bd577d6a9bf7530db0e2917a47a207acb16b2f9c9e41232e0441b733152e.jpg)

![6f97629b93c5dff0f2c1e81f51bac6ee898c43c18bfdbf9e67d8a6fbfe9d031d.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/6f97629b93c5dff0f2c1e81f51bac6ee898c43c18bfdbf9e67d8a6fbfe9d031d.jpg)

![8fef6464c79aa4e9e04b0b7a46dc3f11d7dc54d91ab9ddf83f299d747825a697.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/8fef6464c79aa4e9e04b0b7a46dc3f11d7dc54d91ab9ddf83f299d747825a697.jpg)

![cf36ba7b25f020a1aa0ad8bafce377e49736b75f2d332da727f2ef9fff8ce166.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/cf36ba7b25f020a1aa0ad8bafce377e49736b75f2d332da727f2ef9fff8ce166.jpg)

![f2f3acfa2c0afa6323dfbbdfb512203bcc71b051e06803904efb38e06546076e.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/f2f3acfa2c0afa6323dfbbdfb512203bcc71b051e06803904efb38e06546076e.jpg)

![faefb0f08ed11e3f5aa57d6783a2b860fa02e81db8a4d1d50c4bd054cb1ceca8.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/faefb0f08ed11e3f5aa57d6783a2b860fa02e81db8a4d1d50c4bd054cb1ceca8.jpg)

### Tables

![06b0870d832fa2a4ed1812f9cc5739826b2964d7e834a9124a2afb8a9052c811.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/06b0870d832fa2a4ed1812f9cc5739826b2964d7e834a9124a2afb8a9052c811.jpg)

![5f14574037077770f85542507b6fb1d78b8c1c16e0ac02b0d7a94bf1aff91c92.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/5f14574037077770f85542507b6fb1d78b8c1c16e0ac02b0d7a94bf1aff91c92.jpg)

![65df8f66d006e938c861f5e7ff337ef3b4b08258ba75e5faa3ef27dfb12ad531.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/65df8f66d006e938c861f5e7ff337ef3b4b08258ba75e5faa3ef27dfb12ad531.jpg)

![adc17a9cc204ac2a86caa7d875169bb50cbe0cebf33187566ed95b8675ba47be.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/adc17a9cc204ac2a86caa7d875169bb50cbe0cebf33187566ed95b8675ba47be.jpg)

![b1cbe52103f6cbc6e9ba2544770dffd312820d8a5cf4eddb73e17c00866a859e.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/b1cbe52103f6cbc6e9ba2544770dffd312820d8a5cf4eddb73e17c00866a859e.jpg)

![c6326b24504f7682a2e3db8da8273616425bf4117c7831862d0833e18c9524f9.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/c6326b24504f7682a2e3db8da8273616425bf4117c7831862d0833e18c9524f9.jpg)

![c90404ec251a083a2df763988aed84bae48ce23fdfb96d457a5cb33602bda82a.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/c90404ec251a083a2df763988aed84bae48ce23fdfb96d457a5cb33602bda82a.jpg)

![cdf13efea95a7daed0d3f5b4902721e3528c251fa53adf12c595c04750e475a2.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/cdf13efea95a7daed0d3f5b4902721e3528c251fa53adf12c595c04750e475a2.jpg)

![dfd4511da1e409cc461a9561ea8a9949f4ed9ab5f9ad631645198d08a86ae476.jpg](../icml_results/1392_Plan%20Then%20Action_%20High-Level%20Planning%20Guidance%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/dfd4511da1e409cc461a9561ea8a9949f4ed9ab5f9ad631645198d08a86ae476.jpg)

<a id="paper-1373"></a>

## Conformal Calibration Transfer


### Images

![0eb4cee465c49eb97ddd487fe14f8dc142c10b4b19e7bd815e5b5f574ae75f14.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/images/0eb4cee465c49eb97ddd487fe14f8dc142c10b4b19e7bd815e5b5f574ae75f14.jpg)

![c2eb61b79afcf98cd52f09be47b0b89bcebcd79f8d6f4f25390930b5242df543.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/images/c2eb61b79afcf98cd52f09be47b0b89bcebcd79f8d6f4f25390930b5242df543.jpg)

![dc4a3f7f5ad66f485b2e297deeb5e7ce276255137da7c1887f4b47899c947127.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/images/dc4a3f7f5ad66f485b2e297deeb5e7ce276255137da7c1887f4b47899c947127.jpg)

![f234fa726e74399f8efa9bdf33a6ba1a32687de464e0d3b8a7c118be9e546ea2.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/images/f234fa726e74399f8efa9bdf33a6ba1a32687de464e0d3b8a7c118be9e546ea2.jpg)

### Tables

![0cfbb16411ebc2c65ed9593e7231df840f28d04bb02a82a41104077875daa6b3.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/0cfbb16411ebc2c65ed9593e7231df840f28d04bb02a82a41104077875daa6b3.jpg)

![15f978f76bc58a88facd4b05d9767997511b011cf29128988ba02cd57e790d7e.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/15f978f76bc58a88facd4b05d9767997511b011cf29128988ba02cd57e790d7e.jpg)

![2368a4666e3f4cdd6b6242f48d9b166399d7b16115958440b3c11376308c7c7a.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/2368a4666e3f4cdd6b6242f48d9b166399d7b16115958440b3c11376308c7c7a.jpg)

![27a785143c0e81881b27ef454c1c24e44b7449cd06e8e51bd8d752cf896e94e9.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/27a785143c0e81881b27ef454c1c24e44b7449cd06e8e51bd8d752cf896e94e9.jpg)

![29979f97f0927b0f822bdb691f097428762c3317203d2fa7df173dcc64f452d7.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/29979f97f0927b0f822bdb691f097428762c3317203d2fa7df173dcc64f452d7.jpg)

![4ac2053928e806dff5e45c3489fdf4ee8bf6e3b1ba17ce9f2546130073624ef4.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/4ac2053928e806dff5e45c3489fdf4ee8bf6e3b1ba17ce9f2546130073624ef4.jpg)

![548ba0c5f18733a20dbee2f44d9aad03e24c7530b1799029a56830e2ce130e69.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/548ba0c5f18733a20dbee2f44d9aad03e24c7530b1799029a56830e2ce130e69.jpg)

![5a37915d5c679fe3f75a84e772e1cf9b7cea633c0c1e6f94ea7a8b73e5da3899.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/5a37915d5c679fe3f75a84e772e1cf9b7cea633c0c1e6f94ea7a8b73e5da3899.jpg)

![96556d6aa946d24343a8a3a6b199e57ede368c2dacc914680974960b64bcb6be.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/96556d6aa946d24343a8a3a6b199e57ede368c2dacc914680974960b64bcb6be.jpg)

![9a3c7887901c682e580a71ac390a7c7c41d3d5dcc65b1af20348cd70b01db195.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/9a3c7887901c682e580a71ac390a7c7c41d3d5dcc65b1af20348cd70b01db195.jpg)

![9e5845ff4982b4518e3ad8a3249f4c5fad2f38555eb48448242bdd7a34a0a65c.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/9e5845ff4982b4518e3ad8a3249f4c5fad2f38555eb48448242bdd7a34a0a65c.jpg)

![bde8acd83722f21171143edd3757414d749b0bdd829d6d2cefff61934832cd6a.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/bde8acd83722f21171143edd3757414d749b0bdd829d6d2cefff61934832cd6a.jpg)

![cb782f54f752b66296483ed976eba94c98dca0304b8ab8c11af9bd399516c772.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/cb782f54f752b66296483ed976eba94c98dca0304b8ab8c11af9bd399516c772.jpg)

![dc8d34dec5f31889802f215cd67d1b6361666cd60aa8cf0e313a2aebffb89089.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/dc8d34dec5f31889802f215cd67d1b6361666cd60aa8cf0e313a2aebffb89089.jpg)

![dfed9758d86569a2a1f2fe7b331299ef3f461586d01f1dcce344707cd02bab56.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/dfed9758d86569a2a1f2fe7b331299ef3f461586d01f1dcce344707cd02bab56.jpg)

![ef71b3fb66433d49aada92c7ef862915d449f72451904481fc5fee2b10a4184c.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/ef71b3fb66433d49aada92c7ef862915d449f72451904481fc5fee2b10a4184c.jpg)

![f3c17a2e7330077c9cfb5f4644779738c7a9a510de602fe30dd16697c78b59a3.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/f3c17a2e7330077c9cfb5f4644779738c7a9a510de602fe30dd16697c78b59a3.jpg)

![f8a1972fc23765d48e8123d7618294e114407504cd1c2d2affb298fbe50aa48f.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/f8a1972fc23765d48e8123d7618294e114407504cd1c2d2affb298fbe50aa48f.jpg)

![fe3c22c6c63b27785611dfbe51d8f9bf69772d8f81e6fa4e26702a97d56cfe45.jpg](../icml_results/1393_Conformal%20Calibration%20Transfer/tables/fe3c22c6c63b27785611dfbe51d8f9bf69772d8f81e6fa4e26702a97d56cfe45.jpg)

<a id="paper-1374"></a>

## Fast kernel methods: Sobolev, physics-informed, and additive models


### Images

![0bd71da02cd4ad0145b40c31e5430b21a5b2550525bf7dc7e64be3e8d1b7f83d.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/0bd71da02cd4ad0145b40c31e5430b21a5b2550525bf7dc7e64be3e8d1b7f83d.jpg)

![2ac0402aec287602e6529d756e3c42945d589ea6911c6cc4cbabcb41d592ec4e.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/2ac0402aec287602e6529d756e3c42945d589ea6911c6cc4cbabcb41d592ec4e.jpg)

![61f22dfa6020bb70fa204a7237d81d926dfbed64b6601d2cc804e42b66189f09.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/61f22dfa6020bb70fa204a7237d81d926dfbed64b6601d2cc804e42b66189f09.jpg)

![92e6c7f43dffd736106cc0591487950d47854c4a59d2a8835286489b7fc94e87.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/92e6c7f43dffd736106cc0591487950d47854c4a59d2a8835286489b7fc94e87.jpg)

![ce2a75e067327d0b808d925e44baee6fa608a9dc6f813069794aacf330ef5fd8.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/ce2a75e067327d0b808d925e44baee6fa608a9dc6f813069794aacf330ef5fd8.jpg)

![e28fc61419d4f058a543a3495672270c7f50ffc17159d5cf51ce18c7dbfb9205.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/e28fc61419d4f058a543a3495672270c7f50ffc17159d5cf51ce18c7dbfb9205.jpg)

![e8aed4bdf0f822c001d7fc0b034d89d98987ca1752868c94efa32c2ff04d9d59.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/images/e8aed4bdf0f822c001d7fc0b034d89d98987ca1752868c94efa32c2ff04d9d59.jpg)

### Tables

![f23731e095a5f2bc24441827075228c967b3c3a065b6b500c6a422351a11453a.jpg](../icml_results/1394_Fast%20kernel%20methods_%20Sobolev%2C%20physics-informed%2C%20and%20additive%20models/tables/f23731e095a5f2bc24441827075228c967b3c3a065b6b500c6a422351a11453a.jpg)

<a id="paper-1375"></a>

## Watermarking Graph Neural Networks via Explanations for Ownership Protection


### Images

![238d6a70865dfb0f40eb67a9a0f989ac328f351af024529f693286535450e305.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/238d6a70865dfb0f40eb67a9a0f989ac328f351af024529f693286535450e305.jpg)

![50dc246db6d833ba3cc7ca241f80ad923045531ec96a611e11b300a4ffcb28ed.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/50dc246db6d833ba3cc7ca241f80ad923045531ec96a611e11b300a4ffcb28ed.jpg)

![51246fb54b45d416caaafe47b0c0d9ba14042fe44a480a9292439418e2ef2536.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/51246fb54b45d416caaafe47b0c0d9ba14042fe44a480a9292439418e2ef2536.jpg)

![b32aeaa93fa58c2e1593f149450f1ca61270210cb8b0451e8c934d59041bb8ff.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/b32aeaa93fa58c2e1593f149450f1ca61270210cb8b0451e8c934d59041bb8ff.jpg)

![b973edb49b23bf1262ca7c004378cb633650937fcf3e9e4fe03485e91a23956c.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/b973edb49b23bf1262ca7c004378cb633650937fcf3e9e4fe03485e91a23956c.jpg)

![be6ff258a3822c8835a375f813411587606732db98f206a61c16d89350b96770.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/be6ff258a3822c8835a375f813411587606732db98f206a61c16d89350b96770.jpg)

![c179ad8128bf572d2978dafdcc4377daf6b15d7ca01320c27d799d1a50aff685.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/c179ad8128bf572d2978dafdcc4377daf6b15d7ca01320c27d799d1a50aff685.jpg)

![c636f6dfcd845e7c9479e6ab3092b22de22a94de56b9aee7bf3f850b600d6d1b.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/c636f6dfcd845e7c9479e6ab3092b22de22a94de56b9aee7bf3f850b600d6d1b.jpg)

![cd31658cb1f07a42d6c95c40f2428a908189fd50340d9b8323920a16282344f6.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/cd31658cb1f07a42d6c95c40f2428a908189fd50340d9b8323920a16282344f6.jpg)

![da29a011940979e2d82257dd7e99c8a8dc40a7fc7bcffbc5dfcf21d7c501b5be.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/da29a011940979e2d82257dd7e99c8a8dc40a7fc7bcffbc5dfcf21d7c501b5be.jpg)

![f2f552ba5aa0c8ac67e290452567a37ea24e0b0c4b689527013da2e014aa5273.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/images/f2f552ba5aa0c8ac67e290452567a37ea24e0b0c4b689527013da2e014aa5273.jpg)

### Tables

![0c6c979f287f967b1c0b3c264552e3022c4dde61a6ca31d66db84422907dba69.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/0c6c979f287f967b1c0b3c264552e3022c4dde61a6ca31d66db84422907dba69.jpg)

![0f170328e269c93bed0243feea24ee3c70e11e6c2e192e2de7650bc52260a95b.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/0f170328e269c93bed0243feea24ee3c70e11e6c2e192e2de7650bc52260a95b.jpg)

![2398c43886f623a6992867b165c7d531f31d488f34c1a4177cd17bf321caea10.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/2398c43886f623a6992867b165c7d531f31d488f34c1a4177cd17bf321caea10.jpg)

![332ae650fe66a5caebdc116465250c15603a8a8c12e49773db3355118e15b693.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/332ae650fe66a5caebdc116465250c15603a8a8c12e49773db3355118e15b693.jpg)

![5c41edde1d0ac2ba60da7e71d44948f1a383c16deeba44ef3a358226ccde1989.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/5c41edde1d0ac2ba60da7e71d44948f1a383c16deeba44ef3a358226ccde1989.jpg)

![74597c39ea0501035a61644ad9d5f4150cd39bd9f637a8b7b142de958ef715f2.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/74597c39ea0501035a61644ad9d5f4150cd39bd9f637a8b7b142de958ef715f2.jpg)

![763699d5d8388360e05c1a82667a7a83e7037f46bcb98ceae3a42f9bd00bbe65.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/763699d5d8388360e05c1a82667a7a83e7037f46bcb98ceae3a42f9bd00bbe65.jpg)

![9620b1370a6d0c104e50118c6f8c2724aa9e93ddb2918ee7ce4350ced40bfacf.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/9620b1370a6d0c104e50118c6f8c2724aa9e93ddb2918ee7ce4350ced40bfacf.jpg)

![a08a220edea2e1ae8b9fd3f158cffceedff9c477b508623f1bdeea9b1c56bbf3.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/a08a220edea2e1ae8b9fd3f158cffceedff9c477b508623f1bdeea9b1c56bbf3.jpg)

![a5d08627e45dafcc830345ca2a736232adda43874ad3fe9f99860b369a0aeab2.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/a5d08627e45dafcc830345ca2a736232adda43874ad3fe9f99860b369a0aeab2.jpg)

![a5e1981f786b0750a24a2d67b809078d59698fb743ce13ec9c6e5474c072510c.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/a5e1981f786b0750a24a2d67b809078d59698fb743ce13ec9c6e5474c072510c.jpg)

![b5ebbad181bb528cccd739aa8fe366f54672409a3c028efeddd46d0e1e16e1a8.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/b5ebbad181bb528cccd739aa8fe366f54672409a3c028efeddd46d0e1e16e1a8.jpg)

![ccd0e935856a0b81d3572321696c4fb55aaf4fdfedc55b70d51b94c71760b12e.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/ccd0e935856a0b81d3572321696c4fb55aaf4fdfedc55b70d51b94c71760b12e.jpg)

![e5d28fb40e667bfdd040c36261604f13375a746403284dd65e64580da67983ee.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/e5d28fb40e667bfdd040c36261604f13375a746403284dd65e64580da67983ee.jpg)

![f05820f2b9f09065dc34688829e9b86febebd946ff1e55a2c9f397f110a46e4d.jpg](../icml_results/1395_Watermarking%20Graph%20Neural%20Networks%20via%20Explanations%20for%20Ownership%20Protection/tables/f05820f2b9f09065dc34688829e9b86febebd946ff1e55a2c9f397f110a46e4d.jpg)

<a id="paper-1376"></a>

## Online Compatible Reward Identification from Preference Feedback


### Tables

![0b735e546caedd3bc8bf38f33ec419fb5cfa38b4e250b95bd3aca2b22329e8df.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/0b735e546caedd3bc8bf38f33ec419fb5cfa38b4e250b95bd3aca2b22329e8df.jpg)

![1fa4ee85443caccea537a232f6e4b1265f59a077d07a76642bdfdf8564636323.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/1fa4ee85443caccea537a232f6e4b1265f59a077d07a76642bdfdf8564636323.jpg)

![5b2f06fe968af33daa27c571b0f3081b6b89a673b8c708d6a6c81fb538357f3a.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/5b2f06fe968af33daa27c571b0f3081b6b89a673b8c708d6a6c81fb538357f3a.jpg)

![74edd7ce2c4800c26496fcece12443d9f3d1dd976302ab711e07130fb02e25e2.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/74edd7ce2c4800c26496fcece12443d9f3d1dd976302ab711e07130fb02e25e2.jpg)

![78f482dc7e75f2041c3a65bd3961975e0ec4ecbeb2cf09db27a41bed1add4e42.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/78f482dc7e75f2041c3a65bd3961975e0ec4ecbeb2cf09db27a41bed1add4e42.jpg)

![8259261fa1e4586a1890f39150005399c16d8a05d6ba6adf5e7aef46634f778a.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/8259261fa1e4586a1890f39150005399c16d8a05d6ba6adf5e7aef46634f778a.jpg)

![9794728fba8a809a29baf284ccfd2fb44b944f7cbd3247d2f3591fd58811167c.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/9794728fba8a809a29baf284ccfd2fb44b944f7cbd3247d2f3591fd58811167c.jpg)

![9b7fcd1af53f8e26b7705b85941c78200ad41c5b1cc0d3a0874d66bc48080899.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/9b7fcd1af53f8e26b7705b85941c78200ad41c5b1cc0d3a0874d66bc48080899.jpg)

![a1b25e6f50c486cfb1aa750435d4d7242aeb1e5e33ac6d902a05fd724b7b97fb.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/a1b25e6f50c486cfb1aa750435d4d7242aeb1e5e33ac6d902a05fd724b7b97fb.jpg)

![ae1d65b197621190616afef77e9c8196d0d36d582f623e5186d53fe3773d7069.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/ae1d65b197621190616afef77e9c8196d0d36d582f623e5186d53fe3773d7069.jpg)

![bdda9fea6e71df1eebf90acadc0a52626325596bbe1c0c6be39cf90e3473c03d.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/bdda9fea6e71df1eebf90acadc0a52626325596bbe1c0c6be39cf90e3473c03d.jpg)

![cf997a703254f6bce1ac91eae287147ccd05f7f6554a70091fb7184a78ea276d.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/cf997a703254f6bce1ac91eae287147ccd05f7f6554a70091fb7184a78ea276d.jpg)

![eaea177daad5caddcf067710e84b8452fb01857439ee5958fad21797babd3b0d.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/eaea177daad5caddcf067710e84b8452fb01857439ee5958fad21797babd3b0d.jpg)

![f2793278ab049724ee54fd4ceb054983a71f9d165b352168e3c8fc8151f236b6.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/f2793278ab049724ee54fd4ceb054983a71f9d165b352168e3c8fc8151f236b6.jpg)

![fe8299a6279fac3e128f1b3a132d592b706287be76520df8df5bfa40f8c2e492.jpg](../icml_results/1396_Online%20Compatible%20Reward%20Identification%20from%20Preference%20Feedback/tables/fe8299a6279fac3e128f1b3a132d592b706287be76520df8df5bfa40f8c2e492.jpg)

<a id="paper-1377"></a>

## Chiral Symmetry Breaking in Transformers: A Group-Equivariant Framework for Addressing the Reversal Curse via Adjoint Manifold Mappings


### Images

![00999c3024f0988332a4cd8eb73d8bd7aadcfd7dca38956e5e17f5bc78de74b3.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/00999c3024f0988332a4cd8eb73d8bd7aadcfd7dca38956e5e17f5bc78de74b3.jpg)

![30cc8db7f43592489d1ae6d847b8a5108f79446f5b823f0e31387891e9690c4e.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/30cc8db7f43592489d1ae6d847b8a5108f79446f5b823f0e31387891e9690c4e.jpg)

![5993dd01870c27646081df28668b074bb5241528c8945fc52aaf7d02a4180ba9.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/5993dd01870c27646081df28668b074bb5241528c8945fc52aaf7d02a4180ba9.jpg)

![5d4cdf8ecc947d972b1aa3b70c141c0e9f7acf6e049819ff098968b9a4b98bc7.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/5d4cdf8ecc947d972b1aa3b70c141c0e9f7acf6e049819ff098968b9a4b98bc7.jpg)

![7a391c3b9d667cb909ccab0d39ffd2f046c7abf5aac60b3407144f754309e364.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/7a391c3b9d667cb909ccab0d39ffd2f046c7abf5aac60b3407144f754309e364.jpg)

![7c8207731701e12148bd11a89217bd8bb7e56642445df16942dd74e6229c126c.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/7c8207731701e12148bd11a89217bd8bb7e56642445df16942dd74e6229c126c.jpg)

![86ff39a4361f17bf35860871536593f286a4bdaab2456c44f073d86d77497103.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/86ff39a4361f17bf35860871536593f286a4bdaab2456c44f073d86d77497103.jpg)

![876e5b299f8b7318be14ea6cde9308173e3cb8b0d2700aa9f01a7d169e3cd8c3.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/876e5b299f8b7318be14ea6cde9308173e3cb8b0d2700aa9f01a7d169e3cd8c3.jpg)

![a6c805d544aa01927f4fdeaf269ae565973e1743b57a9f66227f657faea69f4e.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/a6c805d544aa01927f4fdeaf269ae565973e1743b57a9f66227f657faea69f4e.jpg)

![c11cf44ee80abb31a7ab2f87baa713f5eb658fe91596d7309b2aa26ac64b49a5.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/c11cf44ee80abb31a7ab2f87baa713f5eb658fe91596d7309b2aa26ac64b49a5.jpg)

![c9589bfca40149d0134802551d9f4816a0fdf8625719f4fe9e735b8397985537.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/images/c9589bfca40149d0134802551d9f4816a0fdf8625719f4fe9e735b8397985537.jpg)

### Tables

![01a39315d2d1faddf2ba8d9223f671faa3689a89c55bb1d110064fbadb36382d.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/01a39315d2d1faddf2ba8d9223f671faa3689a89c55bb1d110064fbadb36382d.jpg)

![3baab035a2d0ea16dc2572e22707c4e26d708b9853e294519e5667b4d637c941.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/3baab035a2d0ea16dc2572e22707c4e26d708b9853e294519e5667b4d637c941.jpg)

![6fc74b2154c8db7878443cfeaceea7409ed8edd4bb780963e8f18949e770c205.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/6fc74b2154c8db7878443cfeaceea7409ed8edd4bb780963e8f18949e770c205.jpg)

![8465915232971ef95a57c294b796aca481e08734270eecf55a1a29aa1a82d73d.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/8465915232971ef95a57c294b796aca481e08734270eecf55a1a29aa1a82d73d.jpg)

![b53afd6511082059ed5603bee94e0a5e8db0dc0bed11da120027ee7b0f0ede33.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/b53afd6511082059ed5603bee94e0a5e8db0dc0bed11da120027ee7b0f0ede33.jpg)

![d47eab6dd67eb916041d5b32ed0ef165ed1439a8f5fdbb9f019adcf85df2f8c5.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/d47eab6dd67eb916041d5b32ed0ef165ed1439a8f5fdbb9f019adcf85df2f8c5.jpg)

![ec8fe91252746e56364f948007e2f79be41dbf216ddd246a47ff231615c968c6.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/ec8fe91252746e56364f948007e2f79be41dbf216ddd246a47ff231615c968c6.jpg)

![f9d31d23dd06aefa3d605c1491a5baa7771ceb11aafe602ffa79bc9754807a20.jpg](../icml_results/1397_Chiral%20Symmetry%20Breaking%20in%20Transformers_%20A%20Group-Equivariant%20Framework%20for%20Addressing%20the%20Reversal%20/tables/f9d31d23dd06aefa3d605c1491a5baa7771ceb11aafe602ffa79bc9754807a20.jpg)

<a id="paper-1378"></a>

## Stabilizing PPO via Latent-Space Regularization and KDE-Driven Exploration


### Images

![1020376ae39f028e72ec14663fcf0d40f518a7e39c547940282fc8f51ac6631d.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/1020376ae39f028e72ec14663fcf0d40f518a7e39c547940282fc8f51ac6631d.jpg)

![1a48e5ba860ec01d012e5b642bf375e908b26170eeb14cbd3d0e5a93c49519e3.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/1a48e5ba860ec01d012e5b642bf375e908b26170eeb14cbd3d0e5a93c49519e3.jpg)

![1a7c2373dd068c740bcdcca29a0e1e919b28ac543b6c9347556b3c9af8668c72.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/1a7c2373dd068c740bcdcca29a0e1e919b28ac543b6c9347556b3c9af8668c72.jpg)

![1b005459d49e9c4eb26283c20eafac7bb042875de88a371bf1aac1358dd90fa6.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/1b005459d49e9c4eb26283c20eafac7bb042875de88a371bf1aac1358dd90fa6.jpg)

![3f4b9b8ecfcdda6b9e037057423f89cf36cfcc3c153fb17ece156b2b86347b04.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/3f4b9b8ecfcdda6b9e037057423f89cf36cfcc3c153fb17ece156b2b86347b04.jpg)

![53ddd1dd340cc05625dde4805825f1c4b6aacf0d4d9e747a105ec91bae561da2.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/53ddd1dd340cc05625dde4805825f1c4b6aacf0d4d9e747a105ec91bae561da2.jpg)

![7cc0fd968df3fd309d7c50db8eb4b2f1079cb2d7f43c107e232ab26d2c895b3b.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/7cc0fd968df3fd309d7c50db8eb4b2f1079cb2d7f43c107e232ab26d2c895b3b.jpg)

![9605acbc78e620e3884882daaf70fd3bf40b3f64d665e6ab823fb58453aea4f6.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/9605acbc78e620e3884882daaf70fd3bf40b3f64d665e6ab823fb58453aea4f6.jpg)

![98f39ea1bbbe78e8002946ba9724e27a7420086beae8c10ad57652d88201dbc1.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/98f39ea1bbbe78e8002946ba9724e27a7420086beae8c10ad57652d88201dbc1.jpg)

![a42e57511c9bb0f2a938d7772e5783328f06be1a68da7060ed6805fcbfb5b577.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/a42e57511c9bb0f2a938d7772e5783328f06be1a68da7060ed6805fcbfb5b577.jpg)

![b50e48fe7e70e3c066fbbacebc3dde912519cdf8a5aae2920e9b4687934fb44d.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/b50e48fe7e70e3c066fbbacebc3dde912519cdf8a5aae2920e9b4687934fb44d.jpg)

![c6785ca8f0d4b4851f5cfbb58e373e92b7ec0d19aa8f77d214344f1adffe90c0.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/c6785ca8f0d4b4851f5cfbb58e373e92b7ec0d19aa8f77d214344f1adffe90c0.jpg)

![d854b656fc0b846ad05d5e5cbd8993ac9615606506c7506f245da42f34632cca.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/d854b656fc0b846ad05d5e5cbd8993ac9615606506c7506f245da42f34632cca.jpg)

![dcb398fcd5b3c555755415c437e2524c37ad99aee9e062ea8e3ad7ae52f7abb3.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/dcb398fcd5b3c555755415c437e2524c37ad99aee9e062ea8e3ad7ae52f7abb3.jpg)

![f3bc95e9c76aa1f2d5a3b611b67d90a6c4cbc08b19b12440916a54bad00d9e6f.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/f3bc95e9c76aa1f2d5a3b611b67d90a6c4cbc08b19b12440916a54bad00d9e6f.jpg)

![f9a40e5d647ae831cbe7c8f43ca077e0ccd1f6b60146d859655033cb30679325.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/images/f9a40e5d647ae831cbe7c8f43ca077e0ccd1f6b60146d859655033cb30679325.jpg)

### Tables

![02a7e81881a65265c16c291fa9df40c1bbf002af66aa562a26d47226316c7174.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/02a7e81881a65265c16c291fa9df40c1bbf002af66aa562a26d47226316c7174.jpg)

![0cd495a95221403b268d5371347d59d0a112178f35b7830f0478a2f23700bcde.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/0cd495a95221403b268d5371347d59d0a112178f35b7830f0478a2f23700bcde.jpg)

![152a0310f8c2df7e7b8e8db200a2c34ac8daba434b5f72d5dc07c64d68631261.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/152a0310f8c2df7e7b8e8db200a2c34ac8daba434b5f72d5dc07c64d68631261.jpg)

![372792247cd3587f3dfea2dfadd68408ef009efdf14c48a33de3580e6bf4e88f.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/372792247cd3587f3dfea2dfadd68408ef009efdf14c48a33de3580e6bf4e88f.jpg)

![4552b4197bb9223722eab26afaa0245fc51af340d8250010538bae5f7e120feb.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/4552b4197bb9223722eab26afaa0245fc51af340d8250010538bae5f7e120feb.jpg)

![4be70531445a14476a1dcf3482057dd4e9a2a48fb5ce7eabd058139ecf2f1d04.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/4be70531445a14476a1dcf3482057dd4e9a2a48fb5ce7eabd058139ecf2f1d04.jpg)

![77950e3b348141d0bc3acab42701bb9082b6079c0beedc067ced66eeb2cfa54a.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/77950e3b348141d0bc3acab42701bb9082b6079c0beedc067ced66eeb2cfa54a.jpg)

![7de5dccf26271a4e243f3ef18e8799c646de5a54743264c02c371a0094af57d6.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/7de5dccf26271a4e243f3ef18e8799c646de5a54743264c02c371a0094af57d6.jpg)

![8ce660803cdd739c3ca7adda2a65f119c4a0f5c8536d28a0e26b1ebad5afbaf2.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/8ce660803cdd739c3ca7adda2a65f119c4a0f5c8536d28a0e26b1ebad5afbaf2.jpg)

![8fc394ff0913b614622f321fbb441051ed5f9292ea80075646302743c252c513.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/8fc394ff0913b614622f321fbb441051ed5f9292ea80075646302743c252c513.jpg)

![956820d414a087d671c941cbbae4ff1aa138f0cda2b92b7e3cc448de2858629f.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/956820d414a087d671c941cbbae4ff1aa138f0cda2b92b7e3cc448de2858629f.jpg)

![b7c6a98ea41d64cb9915c43a956dc787281a5760d27dbdfdc5a94070462f030b.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/b7c6a98ea41d64cb9915c43a956dc787281a5760d27dbdfdc5a94070462f030b.jpg)

![b7eab10683239796d435ca3904c19c89c478e83464b9334aee908d2c2ff8a47d.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/b7eab10683239796d435ca3904c19c89c478e83464b9334aee908d2c2ff8a47d.jpg)

![dbc815ca52b1fa5e4aad8d9a4bd73b4d9d2e1bba35421ecf58eb07b97030929a.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/dbc815ca52b1fa5e4aad8d9a4bd73b4d9d2e1bba35421ecf58eb07b97030929a.jpg)

![e123c2eb7b1f61480231bfdcb96d4be940f0384ee9bb8e27ad4ea054b7c1b2ea.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/e123c2eb7b1f61480231bfdcb96d4be940f0384ee9bb8e27ad4ea054b7c1b2ea.jpg)

![e590fdbc2ee40d05efc6ef42d2434fe0dfa1464ef2dac09c29f4bb8a4edda9dc.jpg](../icml_results/1398_Stabilizing%20PPO%20via%20Latent-Space%20Regularization%20and%20KDE-Driven%20Exploration/tables/e590fdbc2ee40d05efc6ef42d2434fe0dfa1464ef2dac09c29f4bb8a4edda9dc.jpg)

<a id="paper-1379"></a>

## AesFormer: Transform Everyday Photos into Beautiful Memories


### Images

![26ab4898112271e9ca65ef8b013856cfc0927c70fa61335196eac3c7643d248c.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/images/26ab4898112271e9ca65ef8b013856cfc0927c70fa61335196eac3c7643d248c.jpg)

![2e1d8fa017dfd7a37a24021d03f54a0b1a42fc79d050678f875833b118203883.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/images/2e1d8fa017dfd7a37a24021d03f54a0b1a42fc79d050678f875833b118203883.jpg)

![339d4886373d6a4ab3af201c126206d30e538ca318ea13ee8f8271cc0e3a5481.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/images/339d4886373d6a4ab3af201c126206d30e538ca318ea13ee8f8271cc0e3a5481.jpg)

![9fe847131c5eed24a7506332986cf04819fb6ca4f52ebdb48ebb7e45fa7e84f3.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/images/9fe847131c5eed24a7506332986cf04819fb6ca4f52ebdb48ebb7e45fa7e84f3.jpg)

### Tables

![0ab18a6e574b4741c4ad1fcd80b421e17e590bb62cd6f49a48738eaa125b23ac.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/tables/0ab18a6e574b4741c4ad1fcd80b421e17e590bb62cd6f49a48738eaa125b23ac.jpg)

![4a1c80db31e96b22b2f73b0b02a044ab050b6b1d74b685a15e217535c07bd166.jpg](../icml_results/1399_AesFormer_%20Transform%20Everyday%20Photos%20into%20Beautiful%20Memories/tables/4a1c80db31e96b22b2f73b0b02a044ab050b6b1d74b685a15e217535c07bd166.jpg)

<a id="paper-1380"></a>

## FedEMoE: Improving Personalization on Heterogeneous Federated Learning via Elastic Mixture of Experts Architecture


### Images

![0635b659589d5d9a08801dd1f48b7a01bfb9cf5a9dc6326c819e1befc04d932b.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/0635b659589d5d9a08801dd1f48b7a01bfb9cf5a9dc6326c819e1befc04d932b.jpg)

![a53d604386fc4cace02686ec6f2bee1704ccae1825c9612ccc49a8d602a066c6.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/a53d604386fc4cace02686ec6f2bee1704ccae1825c9612ccc49a8d602a066c6.jpg)

![ce5cd90e7863ad663f263aaac216ea2b80eda1528dd5d22af07ef4b12c174c84.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/ce5cd90e7863ad663f263aaac216ea2b80eda1528dd5d22af07ef4b12c174c84.jpg)

![d97786a32d4866f41db03a5447fe26b4f26d8be632aad9844d8ff8367287a683.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/d97786a32d4866f41db03a5447fe26b4f26d8be632aad9844d8ff8367287a683.jpg)

![ded9f36a9c2c4cb0e6c267c00bc92358f37fb5bd5e6a132a9f047ee2fbf793fd.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/ded9f36a9c2c4cb0e6c267c00bc92358f37fb5bd5e6a132a9f047ee2fbf793fd.jpg)

![eac6af93f8594578bbffacd2f758fb3a8dc1f2a31a5c88a5ddd18b4e9d1ae62c.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/eac6af93f8594578bbffacd2f758fb3a8dc1f2a31a5c88a5ddd18b4e9d1ae62c.jpg)

![fa7f6ff6596a87c50ad5f90777d09171c4d3f27f4d8b214a968e7f1e1a7080b7.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/images/fa7f6ff6596a87c50ad5f90777d09171c4d3f27f4d8b214a968e7f1e1a7080b7.jpg)

### Tables

![0426ca1fcdb11a436fed0134fc5d87fe36d369eae68200e56e0012de3c421470.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/0426ca1fcdb11a436fed0134fc5d87fe36d369eae68200e56e0012de3c421470.jpg)

![0c7f26016c68be856349fcc0f9892d0a963e4450c9adfe24a3b3664a3b86c1ae.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/0c7f26016c68be856349fcc0f9892d0a963e4450c9adfe24a3b3664a3b86c1ae.jpg)

![1a9489d60f60f95f6878b937e1a28975f2b2344a21df49e4523e66cf51cba75a.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/1a9489d60f60f95f6878b937e1a28975f2b2344a21df49e4523e66cf51cba75a.jpg)

![1e0e7d488cf81bd90b2f931b132c0cc76b1f7f478e68cefffa0a375d179de514.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/1e0e7d488cf81bd90b2f931b132c0cc76b1f7f478e68cefffa0a375d179de514.jpg)

![1ff2e50027fb97718e314db151ef4eb8704e6578a1d35a9261536905b22c7d56.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/1ff2e50027fb97718e314db151ef4eb8704e6578a1d35a9261536905b22c7d56.jpg)

![544c19085134d8628fe49fc4a84bf6393b2c057c6c70e90912a8c33de89613b7.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/544c19085134d8628fe49fc4a84bf6393b2c057c6c70e90912a8c33de89613b7.jpg)

![6445c87f99762baa73dd7d5bf80aee756fb9467d12a93c6369586987f221cfd3.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/6445c87f99762baa73dd7d5bf80aee756fb9467d12a93c6369586987f221cfd3.jpg)

![6a218bb801d8ab9083b9308462bca4fc28bbd893604d3c317d75fc2c08f3bc15.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/6a218bb801d8ab9083b9308462bca4fc28bbd893604d3c317d75fc2c08f3bc15.jpg)

![706db2f8b05cb7984e7136e677c1154744227034d70fe73e282f25845e845626.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/706db2f8b05cb7984e7136e677c1154744227034d70fe73e282f25845e845626.jpg)

![78dcc53be7869dae87bcdc16f98cb30d04e415295f590dfeb31eeb1ab2108496.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/78dcc53be7869dae87bcdc16f98cb30d04e415295f590dfeb31eeb1ab2108496.jpg)

![7d33a780bbd7da33c2596e88bf2492f05db13461835c8ebcf435d830cefba394.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/7d33a780bbd7da33c2596e88bf2492f05db13461835c8ebcf435d830cefba394.jpg)

![8fe0619207d19be940ca4c5fe1c17a51598392c570b90af5723535b11a210d19.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/8fe0619207d19be940ca4c5fe1c17a51598392c570b90af5723535b11a210d19.jpg)

![981c91dd6cb2d17f7c507903c8bf69637d52d66ebe80898260924760242f3483.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/981c91dd6cb2d17f7c507903c8bf69637d52d66ebe80898260924760242f3483.jpg)

![99799fc270935bba75a7eb99723dcd993a0f808d9ab28044e9c40b4c30c56a99.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/99799fc270935bba75a7eb99723dcd993a0f808d9ab28044e9c40b4c30c56a99.jpg)

![a99e2de792a9380217c02c7d43a9255454a2714e7a1b8a7c307a9abf93dfc643.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/a99e2de792a9380217c02c7d43a9255454a2714e7a1b8a7c307a9abf93dfc643.jpg)

![c38017ed381ff6a2b2ac342bd8fba19485979880e1396fb8be6d157a5f2f93bd.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/c38017ed381ff6a2b2ac342bd8fba19485979880e1396fb8be6d157a5f2f93bd.jpg)

![c76ca785e33720ff0bd9cf90ab480e99ecc18e205b205c8ba07a67a761b2e4af.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/c76ca785e33720ff0bd9cf90ab480e99ecc18e205b205c8ba07a67a761b2e4af.jpg)

![cbc6dd4b7b3ccb14dd24afe463387dc265aca92765d4c92bc0caf68aa73299d2.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/cbc6dd4b7b3ccb14dd24afe463387dc265aca92765d4c92bc0caf68aa73299d2.jpg)

![d4489ccfb6c6b39c1372a8ae6ff96c231033148df0bce63f5a118eab65c0a4f9.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/d4489ccfb6c6b39c1372a8ae6ff96c231033148df0bce63f5a118eab65c0a4f9.jpg)

![e2747a6761cdcc85d8b97037479e7a7861bfa8a687f314c88b44aa1231034aca.jpg](../icml_results/1400_FedEMoE_%20Improving%20Personalization%20on%20Heterogeneous%20Federated%20Learning%20via%20Elastic%20Mixture%20of%20Expert/tables/e2747a6761cdcc85d8b97037479e7a7861bfa8a687f314c88b44aa1231034aca.jpg)

<a id="paper-1381"></a>

## Which Heads Matter for Reasoning? RL-Guided KV Cache Compression


### Images

![41091af311b0410238933de0acbd57d2e4a80e4b26d70c85375c73334cf2a843.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/41091af311b0410238933de0acbd57d2e4a80e4b26d70c85375c73334cf2a843.jpg)

![51f6ea4cc12c4159a0a37f886f23adb6d4bc552bc6c5a9de56087616d1df6b8f.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/51f6ea4cc12c4159a0a37f886f23adb6d4bc552bc6c5a9de56087616d1df6b8f.jpg)

![52201e069866711c387cd16f0419bfb465bac96296b2e8d5979637caae063f85.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/52201e069866711c387cd16f0419bfb465bac96296b2e8d5979637caae063f85.jpg)

![7d689aafe4ca4e0815f91ebb6038171147b5682a58d50c4c7c4e716097c20d01.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/7d689aafe4ca4e0815f91ebb6038171147b5682a58d50c4c7c4e716097c20d01.jpg)

![867f2994fb902addabee94ad38142bddd350d0813f642ecfff6398e60b8511de.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/867f2994fb902addabee94ad38142bddd350d0813f642ecfff6398e60b8511de.jpg)

![8c582613e797bfd83b5172e25651046b25af981715d9596ccbc59f58e0d14f14.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/8c582613e797bfd83b5172e25651046b25af981715d9596ccbc59f58e0d14f14.jpg)

![8e35b30598d3e022c7ede41b86c916c26c4b69eb1114ac0aef62de6bd65883e3.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/8e35b30598d3e022c7ede41b86c916c26c4b69eb1114ac0aef62de6bd65883e3.jpg)

![9446203be2e5f45b6ad484d96a47a3442a5e523929386738a7693d41c81b0431.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/9446203be2e5f45b6ad484d96a47a3442a5e523929386738a7693d41c81b0431.jpg)

![9b48afae6d08b0f077960270b449423cf27c9c85facd7ec3a18aa9dfe714f6b0.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/9b48afae6d08b0f077960270b449423cf27c9c85facd7ec3a18aa9dfe714f6b0.jpg)

![bc63a07f91921e177b9e077c9dc587e20922989dcc4924316186a1ccf042c0b1.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/bc63a07f91921e177b9e077c9dc587e20922989dcc4924316186a1ccf042c0b1.jpg)

![c0b223bc7dae584cc35664f79bde0229a2ca7658abae9ef1b0953e392812e470.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/c0b223bc7dae584cc35664f79bde0229a2ca7658abae9ef1b0953e392812e470.jpg)

![d9a60c620b8c5de31099ba17569931852ae30d311a3b12bcfc09fd1c04a2c6c6.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/d9a60c620b8c5de31099ba17569931852ae30d311a3b12bcfc09fd1c04a2c6c6.jpg)

![e0c18e827682f93013ac7f7e929dfca4730f265719866d7f398289aaed38e1d8.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/e0c18e827682f93013ac7f7e929dfca4730f265719866d7f398289aaed38e1d8.jpg)

![e2f5c5891a2d28a0b8ce4a87737210dac8152991fa0dfb80b11a29eb61c96534.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/images/e2f5c5891a2d28a0b8ce4a87737210dac8152991fa0dfb80b11a29eb61c96534.jpg)

### Tables

![2a04310216384d200dbbdaa841a009d63a5d69da77fa7752fdfbde46fa663ad8.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/2a04310216384d200dbbdaa841a009d63a5d69da77fa7752fdfbde46fa663ad8.jpg)

![383d74e8f6b729b6f50382504cd37fc3ecb81c1fd0dd4b04aa88791a6726edbd.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/383d74e8f6b729b6f50382504cd37fc3ecb81c1fd0dd4b04aa88791a6726edbd.jpg)

![6bfa8adaec62fd1ca46c9a15062b89930a001b4c1ab8d3be40ca7ee312924602.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/6bfa8adaec62fd1ca46c9a15062b89930a001b4c1ab8d3be40ca7ee312924602.jpg)

![6ca13e159c71417d4ae64cb4549c557c2b274c970f2953193296a8c40812121f.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/6ca13e159c71417d4ae64cb4549c557c2b274c970f2953193296a8c40812121f.jpg)

![859cb7b033838363910b1ed79fe5da9b3e79d76fcc22c2ab82b4653c731f8ee3.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/859cb7b033838363910b1ed79fe5da9b3e79d76fcc22c2ab82b4653c731f8ee3.jpg)

![99b58d5365b250ddd5d5820f6c22073385398d7b8bca6ae17a61bf48cfbe68d6.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/99b58d5365b250ddd5d5820f6c22073385398d7b8bca6ae17a61bf48cfbe68d6.jpg)

![a2d83f694ac5e527644082c0d5989a6e8afdf6478332c278729069a7cdf433f9.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/a2d83f694ac5e527644082c0d5989a6e8afdf6478332c278729069a7cdf433f9.jpg)

![c39c5d9f5371a97048fd1b8f25e3bd6468feeb230c91c2539f4853c249aa3e2c.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/c39c5d9f5371a97048fd1b8f25e3bd6468feeb230c91c2539f4853c249aa3e2c.jpg)

![d4567900b02a9f029bced41d9b5cd89d1e40a0a62d88d1963a2308e8d6810c1a.jpg](../icml_results/1401_Which%20Heads%20Matter%20for%20Reasoning_%20RL-Guided%20KV%20Cache%20Compression/tables/d4567900b02a9f029bced41d9b5cd89d1e40a0a62d88d1963a2308e8d6810c1a.jpg)

<a id="paper-1382"></a>

## BioFormer: Rethinking Cross-Subject Generalization via Spectral Structural Alignment in Biomedical Time-Series


### Images

![1afa16b566179850c4efa8334d98c8b1cb6b37cb8de0e02004ebb41695e457fa.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/1afa16b566179850c4efa8334d98c8b1cb6b37cb8de0e02004ebb41695e457fa.jpg)

![1b57b37e78180678e8cae1e174418dc0c800c5272f0132e8377a04a202e85224.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/1b57b37e78180678e8cae1e174418dc0c800c5272f0132e8377a04a202e85224.jpg)

![3fe8a5d167ad841531da108bfe4dd4385aca6ac43d35869270dbd4cd130945fb.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/3fe8a5d167ad841531da108bfe4dd4385aca6ac43d35869270dbd4cd130945fb.jpg)

![401212cd9028dba3f19c5b965eea0c6f340a6e2d5400c13502e509be9b539d54.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/401212cd9028dba3f19c5b965eea0c6f340a6e2d5400c13502e509be9b539d54.jpg)

![5110d7280c3fb310393f7d609a39086484223aa9fd0b2389747a6dca8cff4d71.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/5110d7280c3fb310393f7d609a39086484223aa9fd0b2389747a6dca8cff4d71.jpg)

![5793a1b7dc7124e9b82d176a9b526f5e399f39311a65404893b2a658273c83b2.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/5793a1b7dc7124e9b82d176a9b526f5e399f39311a65404893b2a658273c83b2.jpg)

![794795ac1e292310e37bc3f1be4174ee6ff79015e4fa1f28fb307f86bb291284.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/794795ac1e292310e37bc3f1be4174ee6ff79015e4fa1f28fb307f86bb291284.jpg)

![8d3ca29f8c9328004742f3dff9ac41199c56e2b6a0db1cc48aa84216561fcc84.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/8d3ca29f8c9328004742f3dff9ac41199c56e2b6a0db1cc48aa84216561fcc84.jpg)

![914a423ee74143a180518595aae3d3a1bafd0e4f5e924f8b7ab4fbf2ad74cf85.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/914a423ee74143a180518595aae3d3a1bafd0e4f5e924f8b7ab4fbf2ad74cf85.jpg)

![b49f124bd24d91c45df50ba6f8512d7a07befd854da5286ab3a65ac00e2fe7ca.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/b49f124bd24d91c45df50ba6f8512d7a07befd854da5286ab3a65ac00e2fe7ca.jpg)

![ba932fd585e4dedc9286c745ac3e673744ef221e4038cfa49fffdab7125fdd70.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/ba932fd585e4dedc9286c745ac3e673744ef221e4038cfa49fffdab7125fdd70.jpg)

![c24570a8a05de715373c3b0b1c7497355590868943f75a91ce01e1500de6789a.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/c24570a8a05de715373c3b0b1c7497355590868943f75a91ce01e1500de6789a.jpg)

![fa909295bba1a9f68113d4d493b1a3d58eef59bc59ce8e82629ad19a8292ef30.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/images/fa909295bba1a9f68113d4d493b1a3d58eef59bc59ce8e82629ad19a8292ef30.jpg)

### Tables

![11cbd6b1be249599417691ba1b3f7cf9268fc5755923721c9368f87382edb6fd.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/11cbd6b1be249599417691ba1b3f7cf9268fc5755923721c9368f87382edb6fd.jpg)

![2a9500a810b1adf9fdef9759c15606e779f8080aaa974799bd9503a2c28a5d8d.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/2a9500a810b1adf9fdef9759c15606e779f8080aaa974799bd9503a2c28a5d8d.jpg)

![44071388543ee7e4ae0b017292bd04949caeca2318bd796f64ba67c0c3fae603.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/44071388543ee7e4ae0b017292bd04949caeca2318bd796f64ba67c0c3fae603.jpg)

![518b434a67cf97963e4f3d7bbc7ca3944e8c8c5ac15fc5d76c5709f1a7b6c6a1.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/518b434a67cf97963e4f3d7bbc7ca3944e8c8c5ac15fc5d76c5709f1a7b6c6a1.jpg)

![6d15c3512409d07fd17e273474853c5e84faec65ca77246c4cd3e7ac638f5bcc.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/6d15c3512409d07fd17e273474853c5e84faec65ca77246c4cd3e7ac638f5bcc.jpg)

![6f9e94bc6b1a42809a5847e3e17bc2584c8c5068c2e4daa6e2ef98cafa82e01e.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/6f9e94bc6b1a42809a5847e3e17bc2584c8c5068c2e4daa6e2ef98cafa82e01e.jpg)

![7fcd1bae844763c8f8f339787bfa1bafd1782973c2c649c778117c95020d1b10.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/7fcd1bae844763c8f8f339787bfa1bafd1782973c2c649c778117c95020d1b10.jpg)

![b1f84bd62a0f021d806e123ef44bf2052351ebc06bc1de1ad192342972f791f5.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/b1f84bd62a0f021d806e123ef44bf2052351ebc06bc1de1ad192342972f791f5.jpg)

![c218db14f502c0ca81ab8d2417cb4c75386562c74c2e57c71ba85f452af21f94.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/c218db14f502c0ca81ab8d2417cb4c75386562c74c2e57c71ba85f452af21f94.jpg)

![c5ab61a1dc48b51f4bb88be522cd954b6cd6cc143bf33a8b696b4e5a8c65a4e8.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/c5ab61a1dc48b51f4bb88be522cd954b6cd6cc143bf33a8b696b4e5a8c65a4e8.jpg)

![c647feec95ec574d5221c28a518c0e3fa8dbd9544c75980baabe67346b04ed76.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/c647feec95ec574d5221c28a518c0e3fa8dbd9544c75980baabe67346b04ed76.jpg)

![d4f313d28201b71ca56952fcfe4d254895cf64520428f0fd59881ec2a07e7aa9.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/d4f313d28201b71ca56952fcfe4d254895cf64520428f0fd59881ec2a07e7aa9.jpg)

![d53bdb2478b8d581a41a9d56b61fdcad16980b4e1261d69e7647dda709b3117f.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/d53bdb2478b8d581a41a9d56b61fdcad16980b4e1261d69e7647dda709b3117f.jpg)

![f66797ad59c0ebfe9cfe8155e4e16b8276b482d15b0678f0f42e3e53546a06aa.jpg](../icml_results/1402_BioFormer_%20Rethinking%20Cross-Subject%20Generalization%20via%20Spectral%20Structural%20Alignment%20in%20Biomedical%20T/tables/f66797ad59c0ebfe9cfe8155e4e16b8276b482d15b0678f0f42e3e53546a06aa.jpg)

<a id="paper-1383"></a>

## Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction


### Images

![121cc32771cc92a118f04793eda394317681a6c7d5d7e0d2e43ccc2bd8ec62be.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/121cc32771cc92a118f04793eda394317681a6c7d5d7e0d2e43ccc2bd8ec62be.jpg)

![26e75bd070c9546063a2311b57cae5fe4bb4373d97c8b7d47c7688d45868f94d.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/26e75bd070c9546063a2311b57cae5fe4bb4373d97c8b7d47c7688d45868f94d.jpg)

![83b6922a69f79ce17ed1dd4440a50297c795793f7d5e02ed00e3415846c20fcd.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/83b6922a69f79ce17ed1dd4440a50297c795793f7d5e02ed00e3415846c20fcd.jpg)

![8628c2f54d78457f3212a3e33be1b73228e71cffc625b354b8caae8fb58e5829.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/8628c2f54d78457f3212a3e33be1b73228e71cffc625b354b8caae8fb58e5829.jpg)

![8e96a5a3ea22106549acb2b24f7bf17bb6cd9a60b55c3cf659fc1dd124f9badc.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/8e96a5a3ea22106549acb2b24f7bf17bb6cd9a60b55c3cf659fc1dd124f9badc.jpg)

![a2e9f5ecb83d3b15946e412952ff9f6599e98c793aacc0f32c55cd7e22942114.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/a2e9f5ecb83d3b15946e412952ff9f6599e98c793aacc0f32c55cd7e22942114.jpg)

![a9605853d39807310065e917764d9e3724e2a31f9e10d90f3580c6812a4bd825.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/a9605853d39807310065e917764d9e3724e2a31f9e10d90f3580c6812a4bd825.jpg)

![bbf31290967c6cfb7455b34979495fb668562b7baba41834e6a200b69e340f68.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/bbf31290967c6cfb7455b34979495fb668562b7baba41834e6a200b69e340f68.jpg)

![d32f6443f68366de49830a2b2d5c9c2ca9304131c7544045ff1125e9790c9b7c.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/d32f6443f68366de49830a2b2d5c9c2ca9304131c7544045ff1125e9790c9b7c.jpg)

![e1d621e6747a2c2859eef64cef886c68021122fa620c088b5a40f2279829e9de.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/e1d621e6747a2c2859eef64cef886c68021122fa620c088b5a40f2279829e9de.jpg)

![e3a28524fb84b820aa155a969e9df96ccd07d8f9a6227848c5d4e7a3bc19738b.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/e3a28524fb84b820aa155a969e9df96ccd07d8f9a6227848c5d4e7a3bc19738b.jpg)

![eb47dc001538ebe36cfe62f46f9c3a3034023c86cb545c05fbbdc57829004ff3.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/eb47dc001538ebe36cfe62f46f9c3a3034023c86cb545c05fbbdc57829004ff3.jpg)

![f8e5fb6d7b79c315ee918c35ad55aa57c76d2064d913f1f13124c148e061d381.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/f8e5fb6d7b79c315ee918c35ad55aa57c76d2064d913f1f13124c148e061d381.jpg)

![fde857c078a40b5525cfda57320b2c6e227a6c632c89e721e09ae3b0dd461dc4.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/images/fde857c078a40b5525cfda57320b2c6e227a6c632c89e721e09ae3b0dd461dc4.jpg)

### Tables

![13c05c7102247d1969344053d94af6ee4f1a69f343f9efe607d364f46f060244.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/13c05c7102247d1969344053d94af6ee4f1a69f343f9efe607d364f46f060244.jpg)

![31c16fb265a5c07a60f548d615d10b3bd75a228e8e1d17ae62f2fd941d0c7c0a.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/31c16fb265a5c07a60f548d615d10b3bd75a228e8e1d17ae62f2fd941d0c7c0a.jpg)

![3695e8d22f46631339b836da39544f3a93e6b322ce79c88d9aca183668f416ff.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/3695e8d22f46631339b836da39544f3a93e6b322ce79c88d9aca183668f416ff.jpg)

![3aec37718dc6fcc83a8cfa401aa66417781e537c8d73d65d859df72f5440a792.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/3aec37718dc6fcc83a8cfa401aa66417781e537c8d73d65d859df72f5440a792.jpg)

![7a469cfd9809e2cd88a27235df497ec611976f0418e2946bf20efcda326bf970.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/7a469cfd9809e2cd88a27235df497ec611976f0418e2946bf20efcda326bf970.jpg)

![8ba2cb52486b8a62438875ac9bf606de3295383b5357beb1d3afa0450ddc732f.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/8ba2cb52486b8a62438875ac9bf606de3295383b5357beb1d3afa0450ddc732f.jpg)

![a8b8430832acd16dddcd4a9a2e62ca2287425de0220dfe64f76979c359f4fa32.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/a8b8430832acd16dddcd4a9a2e62ca2287425de0220dfe64f76979c359f4fa32.jpg)

![ac933840751983d77bfe8ce97251180b9f5ecaf516b2ccffb94c30eddf1c5542.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/ac933840751983d77bfe8ce97251180b9f5ecaf516b2ccffb94c30eddf1c5542.jpg)

![b37331c752d444bcf4997c16efeebd3589ade0d2b3b2f3475de20ff36f6da5f9.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/b37331c752d444bcf4997c16efeebd3589ade0d2b3b2f3475de20ff36f6da5f9.jpg)

![c39dd300ca4b05513a39973c55e388f358ab2bb70664b5aaba76a630d165732c.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/c39dd300ca4b05513a39973c55e388f358ab2bb70664b5aaba76a630d165732c.jpg)

![c7dec0c89ac09d7d7f79ee7adc4ea40fdfba0fcedef1a949e30b83531c5c057e.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/c7dec0c89ac09d7d7f79ee7adc4ea40fdfba0fcedef1a949e30b83531c5c057e.jpg)

![ded5eaedfc62f3fcb6a6cf681f8c8ef0abdf3fc697417224c508c75b2dfb1bb8.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/ded5eaedfc62f3fcb6a6cf681f8c8ef0abdf3fc697417224c508c75b2dfb1bb8.jpg)

![e8b3568310484d364955201a8d445acaa06d0f8fde50ddd37972e974e7fbc987.jpg](../icml_results/1403_Can%20We%20Build%20a%20Monolithic%20Model%20for%20Fake%20Image%20Detection_%20SICA_%20Semantic-Induced%20Constrained%20Adaptat/tables/e8b3568310484d364955201a8d445acaa06d0f8fde50ddd37972e974e7fbc987.jpg)

<a id="paper-1384"></a>

## MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics


### Images

![012fc2a0d6e3a8ba0c263a1e845b18dd149128e5ac2d51a18132759384ef6dec.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/012fc2a0d6e3a8ba0c263a1e845b18dd149128e5ac2d51a18132759384ef6dec.jpg)

![014d240dc36e9d7e24195c671419a2963bd3912252fbd73e49a632ae657ae3c2.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/014d240dc36e9d7e24195c671419a2963bd3912252fbd73e49a632ae657ae3c2.jpg)

![186b8aea209e6950d10f2934df11ea24d25ddda9567a2dc0abd9034c913c4a32.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/186b8aea209e6950d10f2934df11ea24d25ddda9567a2dc0abd9034c913c4a32.jpg)

![1be4d171768e3cacc3fd7f2ee02da70181827a43fac04bb757646c3a08e571fc.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/1be4d171768e3cacc3fd7f2ee02da70181827a43fac04bb757646c3a08e571fc.jpg)

![37b7f8e590ef16c37fe82fa8c36330c42a50c112fe59b4793a721ee0f3764a49.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/37b7f8e590ef16c37fe82fa8c36330c42a50c112fe59b4793a721ee0f3764a49.jpg)

![678d906cfd793f247f62930920bda120cdbd2126db0f14563b9c28f6091e8452.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/678d906cfd793f247f62930920bda120cdbd2126db0f14563b9c28f6091e8452.jpg)

![8c5f6ae0690316d9ac08621a1b1418b312eed27095f9b50e89ecda917388dc45.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/8c5f6ae0690316d9ac08621a1b1418b312eed27095f9b50e89ecda917388dc45.jpg)

![92ab243c8f051eef005a6c982783a499e081586cc362616f5d41e4d319145240.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/92ab243c8f051eef005a6c982783a499e081586cc362616f5d41e4d319145240.jpg)

![996ace951c7b0eca2d419585aa93b4e5e62db4579d6bc937eeee059680ab498f.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/996ace951c7b0eca2d419585aa93b4e5e62db4579d6bc937eeee059680ab498f.jpg)

![9a1d9c644f6914fd842fa123b690ed2f3e0bb1a09dc625639c2f313f22fdcc79.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/9a1d9c644f6914fd842fa123b690ed2f3e0bb1a09dc625639c2f313f22fdcc79.jpg)

![ab1d0a471543b30282fafd07d3debf99f5dd3f618a82423a3f246d0c92a46609.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/ab1d0a471543b30282fafd07d3debf99f5dd3f618a82423a3f246d0c92a46609.jpg)

![b0c3c4d735d89f90bd127f33855fd705f7d7f96410e9f6e3b83514e2d1efdc63.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/b0c3c4d735d89f90bd127f33855fd705f7d7f96410e9f6e3b83514e2d1efdc63.jpg)

![b2ba9beaf7285d62adb4485ad637010fdfe2c39390e40f0ab817fb014839ea59.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/b2ba9beaf7285d62adb4485ad637010fdfe2c39390e40f0ab817fb014839ea59.jpg)

![b9c36be42f5ea3cb1a075a79116d2c44cd1b03b3634649dce5699334dfb61b0e.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/b9c36be42f5ea3cb1a075a79116d2c44cd1b03b3634649dce5699334dfb61b0e.jpg)

![ba4ee39b4891ce969f6c94884d92f6cafebc4a10252b201164c59ccba888197b.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/ba4ee39b4891ce969f6c94884d92f6cafebc4a10252b201164c59ccba888197b.jpg)

![c3da971de0907e57faa5708a22b18477ef3626c24b7949612677293e934f4ee5.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/c3da971de0907e57faa5708a22b18477ef3626c24b7949612677293e934f4ee5.jpg)

![c6eeafd4cb2b03245bce1b39b9f5503000b721e770a4eff5fa54ef753896ee91.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/c6eeafd4cb2b03245bce1b39b9f5503000b721e770a4eff5fa54ef753896ee91.jpg)

![d10a0a497421a7972297916a8d11dffa015ce370b750446167c42e1bb6d92da2.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/d10a0a497421a7972297916a8d11dffa015ce370b750446167c42e1bb6d92da2.jpg)

![dd04999c4ac09b4e0f71c4aafe561d3f2389c844231257afcfd8f770ce729e12.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/dd04999c4ac09b4e0f71c4aafe561d3f2389c844231257afcfd8f770ce729e12.jpg)

![e5981ae9ce49f28fc317626762b62d0602b0959505b21ce6779548c33c255d24.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/e5981ae9ce49f28fc317626762b62d0602b0959505b21ce6779548c33c255d24.jpg)

![ea0d0f8ff9fbdbfa2454d0cf65591471a9876e9d8b54ca88bda707d658659e6a.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/ea0d0f8ff9fbdbfa2454d0cf65591471a9876e9d8b54ca88bda707d658659e6a.jpg)

![ebfaa4d6a633f83f1ea09a6eb4d0286c1470c5b8cfc691ff380c75657c1821a2.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/ebfaa4d6a633f83f1ea09a6eb4d0286c1470c5b8cfc691ff380c75657c1821a2.jpg)

![f94003d5dd4bfb728f0a247780605a62d592482e5d8cf65d11fa7b2d59b4b520.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/images/f94003d5dd4bfb728f0a247780605a62d592482e5d8cf65d11fa7b2d59b4b520.jpg)

### Tables

![132248acc943dd49cdbdbcb5002c5297862a22f07b64c36c11f78ce718baac29.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/132248acc943dd49cdbdbcb5002c5297862a22f07b64c36c11f78ce718baac29.jpg)

![25071ffa0ac2155707333c7b8ae0d2dd71ca590709390415509ec5451eb5a868.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/25071ffa0ac2155707333c7b8ae0d2dd71ca590709390415509ec5451eb5a868.jpg)

![36c232b06549d5765e4e1a929ccfd67528d752326b860996c928c4cfd4405afa.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/36c232b06549d5765e4e1a929ccfd67528d752326b860996c928c4cfd4405afa.jpg)

![42daa21087d9129b47044a20985eaba1ed7ad4776af57f176945728be135baef.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/42daa21087d9129b47044a20985eaba1ed7ad4776af57f176945728be135baef.jpg)

![57fd22cd789a5151d698636bbbe12e16d19fa39a9288d377e409214d5a11878e.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/57fd22cd789a5151d698636bbbe12e16d19fa39a9288d377e409214d5a11878e.jpg)

![72f4bba6e88721d65409c049a6bc4347a02aa8e8f494e234ae7e2d391720e729.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/72f4bba6e88721d65409c049a6bc4347a02aa8e8f494e234ae7e2d391720e729.jpg)

![7819ab9a1b407c7aa342670a084dfd90c68e66b3c2d43201e6eb6dd5410e3835.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/7819ab9a1b407c7aa342670a084dfd90c68e66b3c2d43201e6eb6dd5410e3835.jpg)

![790f28d124bfcba7df88568c9307f8be39e1c630c101eb999ed8145df68f626f.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/790f28d124bfcba7df88568c9307f8be39e1c630c101eb999ed8145df68f626f.jpg)

![8a4957b0a41d18e8a27c6b02fec92df7ae85e7a179a25b22d1fe0449333d0a05.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/8a4957b0a41d18e8a27c6b02fec92df7ae85e7a179a25b22d1fe0449333d0a05.jpg)

![9f2fd55eb2694be39eb84ae23b7ea9c64edea4d8c9e72072439727473923e60c.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/9f2fd55eb2694be39eb84ae23b7ea9c64edea4d8c9e72072439727473923e60c.jpg)

![af417893078a09c62d2ee120f3ebc20726ce73bd28cdb7d5b96fc0c67748f6fa.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/af417893078a09c62d2ee120f3ebc20726ce73bd28cdb7d5b96fc0c67748f6fa.jpg)

![f3206f18ec3dea839b0e8b7933780b2a1e8d2cc53c487e5ec8f5f3ad64e338ec.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/f3206f18ec3dea839b0e8b7933780b2a1e8d2cc53c487e5ec8f5f3ad64e338ec.jpg)

![f93d34f68ec8a3f590f35008b58e6d888b7d51fbd43cf00ab444296d8a18f57e.jpg](../icml_results/1404_MetaDNS_%20Enhancing%20Exploration%20in%20Discrete%20Neural%20Samplers%20via%20Well-Tempered%20Metadynamics/tables/f93d34f68ec8a3f590f35008b58e6d888b7d51fbd43cf00ab444296d8a18f57e.jpg)

<a id="paper-1385"></a>

## Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech


### Images

![00a9c44d7347468ef8046b18dd0b1f71e8fe2674c419d1fd3f86386f11c8dd01.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/00a9c44d7347468ef8046b18dd0b1f71e8fe2674c419d1fd3f86386f11c8dd01.jpg)

![2cce7d8e1f07cbe5ff9316692a4bcc8ea7d621ebf5624766112d74787581147f.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/2cce7d8e1f07cbe5ff9316692a4bcc8ea7d621ebf5624766112d74787581147f.jpg)

![49c47d82153ec163aeb4a193ea02d2e619b2b066d4fdcba5b4ed67f79008fcfc.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/49c47d82153ec163aeb4a193ea02d2e619b2b066d4fdcba5b4ed67f79008fcfc.jpg)

![5fc7c989978d52656d1059af51b9812a1441534df66abe9f0e391d33b2b62be8.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/5fc7c989978d52656d1059af51b9812a1441534df66abe9f0e391d33b2b62be8.jpg)

![7400ecb9bf85bff500b1c258b79507c2aa965758060c03b286280c4f53870790.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/7400ecb9bf85bff500b1c258b79507c2aa965758060c03b286280c4f53870790.jpg)

![79bc72be654b1ad63028a6a6882861065e5e338b1b6758f4da174c512272f72f.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/79bc72be654b1ad63028a6a6882861065e5e338b1b6758f4da174c512272f72f.jpg)

![7a2f2042c21a37b22489aeb60c1527f70d5b3026e31663590dedfa25879a6bc5.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/7a2f2042c21a37b22489aeb60c1527f70d5b3026e31663590dedfa25879a6bc5.jpg)

![7bd875a794fe19f7304074c5c72fa340cdcddef75d0cc10ad432289f3037785e.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/7bd875a794fe19f7304074c5c72fa340cdcddef75d0cc10ad432289f3037785e.jpg)

![9f9d9ff18236113fb802353496c50c574168db7202ab9a679881d97c793fa627.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/9f9d9ff18236113fb802353496c50c574168db7202ab9a679881d97c793fa627.jpg)

![ace84785318e2d8c7b85f259ae32ace8d104969156918d13eb7fb2301803c9b8.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/ace84785318e2d8c7b85f259ae32ace8d104969156918d13eb7fb2301803c9b8.jpg)

![b18dae91be4158ef95e77c952364fa1046a0d7ce8f8d4338a9af71b9a5cc80a9.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/b18dae91be4158ef95e77c952364fa1046a0d7ce8f8d4338a9af71b9a5cc80a9.jpg)

![cb8d39331ddee65e810aeb63db21e3b560cb54f8355bb35cdcca3e77a0996917.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/cb8d39331ddee65e810aeb63db21e3b560cb54f8355bb35cdcca3e77a0996917.jpg)

![cc3e3e3c63313f6d8e6e463635825262c01125e18b98bcb78a9bb9b966c8e1a2.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/images/cc3e3e3c63313f6d8e6e463635825262c01125e18b98bcb78a9bb9b966c8e1a2.jpg)

### Tables

![33f6d2956f84b23ff3f1555250944f7883c781314e76f5d51b9831b1e9203df6.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/tables/33f6d2956f84b23ff3f1555250944f7883c781314e76f5d51b9831b1e9203df6.jpg)

![531fdcc92c926de894f546e4740c2c997e0c3339445973f5b9092c96a76272d1.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/tables/531fdcc92c926de894f546e4740c2c997e0c3339445973f5b9092c96a76272d1.jpg)

![a2a86a2a3a43910b3cc7dc8d37ac3139ddb91e462e2bfa073463b75622aa5fc1.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/tables/a2a86a2a3a43910b3cc7dc8d37ac3139ddb91e462e2bfa073463b75622aa5fc1.jpg)

![b0204218686aadb5fee07d61bc9c9a551c5b5580484788903b4bc7cfe32c0709.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/tables/b0204218686aadb5fee07d61bc9c9a551c5b5580484788903b4bc7cfe32c0709.jpg)

![dc527c6c16150e67fe082a57b36d2150a825e41cba3298e9e854500ac85630b1.jpg](../icml_results/1405_Sparse%20Autoencoders%20for%20Interpretable%20Emotion%20Control%20in%20Text-to-Speech/tables/dc527c6c16150e67fe082a57b36d2150a825e41cba3298e9e854500ac85630b1.jpg)

<a id="paper-1386"></a>

## Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression

### Images

![1e4839b4adcc4df446cc7f0517bbb3833001c39b6fe1d550898fd21101ec672c.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/1e4839b4adcc4df446cc7f0517bbb3833001c39b6fe1d550898fd21101ec672c.jpg)

![2798c0fd04191f17d327ad6163bfb52f84dddb1775b587d2638f095575c3c334.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/2798c0fd04191f17d327ad6163bfb52f84dddb1775b587d2638f095575c3c334.jpg)

![2c1a4370cc8d4761da6ba5f62f0709af0d381d40753c7801621092d4a55e3d4a.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/2c1a4370cc8d4761da6ba5f62f0709af0d381d40753c7801621092d4a55e3d4a.jpg)

![3307da0822d3ac9bb9e59098f5011e891aa4948ac07901b92eaf0ea489391aca.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/3307da0822d3ac9bb9e59098f5011e891aa4948ac07901b92eaf0ea489391aca.jpg)

![3496e698452a39057d32b6c1f6880c11feee33bcced2ac43b19fdefbbf66cfd7.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/3496e698452a39057d32b6c1f6880c11feee33bcced2ac43b19fdefbbf66cfd7.jpg)

![5f576e80b4775cac90618fb8f2af85001d03b15fd7c114a5fac8d85191961fb3.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/5f576e80b4775cac90618fb8f2af85001d03b15fd7c114a5fac8d85191961fb3.jpg)

![a3a2ad8ba6f5f519688bc8aa76f5263837aed7b63863e1367856f5afabae767b.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/a3a2ad8ba6f5f519688bc8aa76f5263837aed7b63863e1367856f5afabae767b.jpg)

![b43cc81a12471c3face32634307f54f4c0a2c27e5afed1aed2c94e6cc7a5251d.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/b43cc81a12471c3face32634307f54f4c0a2c27e5afed1aed2c94e6cc7a5251d.jpg)

![c01e7be138ce4a738c318414e332000859b231cb503969ab95050123a39316aa.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/c01e7be138ce4a738c318414e332000859b231cb503969ab95050123a39316aa.jpg)

![dbe04f2ae46e04e60cf7168c9da4a630d50cd597478982fa521c1c3a8b4d4cb0.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/dbe04f2ae46e04e60cf7168c9da4a630d50cd597478982fa521c1c3a8b4d4cb0.jpg)

![e300b0f6c0c1812c354856d77d09409c50028f42fac2fd487a08f8dd4837282f.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/e300b0f6c0c1812c354856d77d09409c50028f42fac2fd487a08f8dd4837282f.jpg)

![f3e2fe39a168e38bcc631d4bd1a40602f3396d98081f53a1b865fcf4b6ec7b7f.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/f3e2fe39a168e38bcc631d4bd1a40602f3396d98081f53a1b865fcf4b6ec7b7f.jpg)

![f9b9f9595d025eb9d9775097db70cc689630912c6113fcca2d338c77d9659ec3.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/images/f9b9f9595d025eb9d9775097db70cc689630912c6113fcca2d338c77d9659ec3.jpg)

### Tables

![141328c5ddae70ae536f58f9b06b9b1c04ae6818b880b592fc17fab1b4032651.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/141328c5ddae70ae536f58f9b06b9b1c04ae6818b880b592fc17fab1b4032651.jpg)

![179cb4d9c73e12116a3f6c624c242332a09895df3413bb0650f407eac373473f.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/179cb4d9c73e12116a3f6c624c242332a09895df3413bb0650f407eac373473f.jpg)

![38def90cc9effb4fa7798a1e877dc3db116b3fa869d2505e58a53d08ab83a964.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/38def90cc9effb4fa7798a1e877dc3db116b3fa869d2505e58a53d08ab83a964.jpg)

![3ab197e50d85a8b0b1e299e0a70d7cf72268813620b72b32ddd5ff9e23b57eca.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/3ab197e50d85a8b0b1e299e0a70d7cf72268813620b72b32ddd5ff9e23b57eca.jpg)

![4b530246404565ae577e2a7da271fb198803d4befb92c6a46028cd2b75e4382e.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/4b530246404565ae577e2a7da271fb198803d4befb92c6a46028cd2b75e4382e.jpg)

![7c01aaa2dcaa10d903fee725d1dcdcbf1c0e6528ecae011f642b21e0eca2d567.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/7c01aaa2dcaa10d903fee725d1dcdcbf1c0e6528ecae011f642b21e0eca2d567.jpg)

![820e8a53cad6bfff5fc1ea59f67ad1b03280e05d75286f1ba9fbd187852c0c56.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/820e8a53cad6bfff5fc1ea59f67ad1b03280e05d75286f1ba9fbd187852c0c56.jpg)

![a5edd1ee52ef4d85032686c86f773c8cd2f9cc2dd1737a6e0ca417d8cc97a3d2.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/a5edd1ee52ef4d85032686c86f773c8cd2f9cc2dd1737a6e0ca417d8cc97a3d2.jpg)

![d2910069924c9c69a162366b5b78f86d6b7c1d11fdb2462c6e41c07b0877256a.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/d2910069924c9c69a162366b5b78f86d6b7c1d11fdb2462c6e41c07b0877256a.jpg)

![d710d143ca054e773aaf3ede0d8ee8f0ff8b03bb3f4391145e16b98982aa2351.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/d710d143ca054e773aaf3ede0d8ee8f0ff8b03bb3f4391145e16b98982aa2351.jpg)

![d7a77dc4cf725fa058bf64c3f8989cd81b129a7f6bc119b5eb9d10cfb57d5d8d.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/d7a77dc4cf725fa058bf64c3f8989cd81b129a7f6bc119b5eb9d10cfb57d5d8d.jpg)

![e224eac201b5e81d802622c412e389363bb951b1c1bc702269475f3f5408cd12.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/e224eac201b5e81d802622c412e389363bb951b1c1bc702269475f3f5408cd12.jpg)

![e5ea8fb0c3263ca0f613a9e16aaf247a6824502303b4b39321945ef2f9d51aea.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/e5ea8fb0c3263ca0f613a9e16aaf247a6824502303b4b39321945ef2f9d51aea.jpg)

![fab67487413a0e407bbc26d2dc4bc44e42f622badf578bd3782b49f6e2a8f23f.jpg](../icml_results/1406_Injecting%20Distributional%20Awareness%20into%20MLLMs%20via%20Reinforcement%20Learning%20for%20Deep%20Imbalanced%20Regress/tables/fab67487413a0e407bbc26d2dc4bc44e42f622badf578bd3782b49f6e2a8f23f.jpg)
