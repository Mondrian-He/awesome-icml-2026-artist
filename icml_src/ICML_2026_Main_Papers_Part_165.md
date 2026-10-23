# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 165 of 197

## 目录 (Table of Contents)

1. [Concept-Guided Tokenization: Closing the Gap Between Reconstruction and Generation](#paper-5413)
2. [Beyond Description: Federated Adaptation via Semantic-Visual Prototype Alignment](#paper-5414)
3. [Second-Order Bilevel Optimization with Accelerated Convergence Rates](#paper-5415)
4. [From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning](#paper-5416)
5. [Step-Level Sparse Autoencoder for Reasoning Process Interpretation](#paper-5417)
6. [Counterfactual Occlusion-Aware Learning via Visibility Intervention for LiDAR Anomaly Detection](#paper-5418)
7. [One-Way Policy Optimization for Self-Evolving LLMs](#paper-5419)
8. [Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals](#paper-5420)
9. [Towards Whole-corpus Reconstruction of Heterogeneous RAG Knowledge Bases](#paper-5421)
10. [Batched Contextual Reinforcement](#paper-5422)
11. [Improving the Sensitivity of Backdoor Detectors via Class Subspace Orthogonalization](#paper-5423)
12. [Semantic Robustness Certification for Vision-Language Models](#paper-5424)
13. [ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution](#paper-5425)
14. [One-Shot Weighted Ensemble Estimation for Federated Quantile Regression: Optimal Statistical Guarantees under Heterogeneous Structured Data](#paper-5426)
15. [Toward Safe Quantization-Aware Fine-tuning: Understanding and Mitigating Safety Alignment Degradation](#paper-5427)
16. [Wasserstein Geometry-Aware Adaptive Control via Meta-Learning](#paper-5428)
17. [Forward-Chaining Temporal Point Process](#paper-5429)
18. [HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression](#paper-5430)
19. [Diffusion Controller: Framework, Algorithms and Parameterization](#paper-5431)
20. [Branching Diffusion for Point Processes in Time and Space](#paper-5432)
21. [Orthogonal Model Merging](#paper-5433)
22. [AgentTailor: A Semantic-Aware LLM-Based Multi-Agent System with Actor-Critic Structure](#paper-5434)
23. [Recursive Models for Long-Horizon Reasoning](#paper-5435)
24. [Data Agent: Learning to Select Data via End-to-End Dynamic Optimization](#paper-5436)
25. [DecFus: Decentralized Layer-wise Fusion with Dynamic Exploration and Exploitation](#paper-5437)
26. [WMVLM: Evaluating Diffusion Model Image Watermarking via Vision-Language Models](#paper-5438)
27. [ExVerus: Verus Proof Repair via Counterexample Reasoning](#paper-5439)
28. [AutoMat: Physics-Guided Agentic Reasoning for Solving Ill-Posed Inverse Microscopy Problems](#paper-5440)
29. [Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective](#paper-5441)
30. [Revisiting Robustness for LLM Safety Alignment via Selective Geometry Control](#paper-5442)
31. [Improving Sampling for Masked Diffusion Models via Information Gain](#paper-5443)
32. [Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design](#paper-5444)
33. [Accelerating Langevin Monte Carlo via Efficient Stochastic Runge-Kutta Methods beyond Log-Concavity](#paper-5445)

---


<a id="paper-5413"></a>

## Concept-Guided Tokenization: Closing the Gap Between Reconstruction and Generation

### Images

![02613e242fa7fdf5283d19f93b9d6dfc991de7873de1b520e7a0fb1cb226e4a6.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/02613e242fa7fdf5283d19f93b9d6dfc991de7873de1b520e7a0fb1cb226e4a6.jpg)

![0a310f61ad639a9769849c0388a443b1330730a5f4b9e621f8b35be8787ae4a0.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/0a310f61ad639a9769849c0388a443b1330730a5f4b9e621f8b35be8787ae4a0.jpg)

![27a62d55b4d17e8ee59fd5c39856d4cac3bfbb422ee5e43b42b142152ace7608.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/27a62d55b4d17e8ee59fd5c39856d4cac3bfbb422ee5e43b42b142152ace7608.jpg)

![2ad1c474d38b41e171af2c311a502871280c63986c64efbd08802f1e6384cb3f.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/2ad1c474d38b41e171af2c311a502871280c63986c64efbd08802f1e6384cb3f.jpg)

![2f686c2fcb7cbc74275402ed3bf64f7bf498441090f3466354c605ae52614dbc.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/2f686c2fcb7cbc74275402ed3bf64f7bf498441090f3466354c605ae52614dbc.jpg)

![313eab0ee1c8ce7993420b039728145b0dcc3bd68aadd5d2ed436c76b393211d.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/313eab0ee1c8ce7993420b039728145b0dcc3bd68aadd5d2ed436c76b393211d.jpg)

![39d3fca866b0c9425537963fb3d9a16ecb506261f90384520d41497d1967ff41.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/39d3fca866b0c9425537963fb3d9a16ecb506261f90384520d41497d1967ff41.jpg)

![3ad1c1c8b22c2235f1440f246a0c6c0e874d618fed376ab489d94066870a1718.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/3ad1c1c8b22c2235f1440f246a0c6c0e874d618fed376ab489d94066870a1718.jpg)

![3ed0502fd1f28b05f8b669b6444e13375638d0705e28836ad8c8f72165b897db.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/3ed0502fd1f28b05f8b669b6444e13375638d0705e28836ad8c8f72165b897db.jpg)

![3f002991d1b7315e9f83c863fd029ceaf92577edd156f52e08a42ef5bb7da6ba.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/3f002991d1b7315e9f83c863fd029ceaf92577edd156f52e08a42ef5bb7da6ba.jpg)

![408860b6801b99fffcd637b52d0a95ad37ac573972573ac2c7e981b2d089c2d1.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/408860b6801b99fffcd637b52d0a95ad37ac573972573ac2c7e981b2d089c2d1.jpg)

![4bc3c01d5e5fc284ca99cf4554399b2112942f8002f48d8ad5ec6347fdf174b8.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/4bc3c01d5e5fc284ca99cf4554399b2112942f8002f48d8ad5ec6347fdf174b8.jpg)

![4d90739709d36f049c67c9ba097599f51e65115a77bed540b829af94e6da4133.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/4d90739709d36f049c67c9ba097599f51e65115a77bed540b829af94e6da4133.jpg)

![550d887f7dccfd0991b155c16180a50e93b24bb3ed85a2086695ef7b99332dba.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/550d887f7dccfd0991b155c16180a50e93b24bb3ed85a2086695ef7b99332dba.jpg)

![56b4e3eab268fa765bc60578ae62a8b7a47c2b004444d14c6e669362a46ffb7e.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/56b4e3eab268fa765bc60578ae62a8b7a47c2b004444d14c6e669362a46ffb7e.jpg)

![5eb76c3dc82b26fbda5d59b1c1946610b7554d47ceb5f5f99ac44acdf7c52ed0.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/5eb76c3dc82b26fbda5d59b1c1946610b7554d47ceb5f5f99ac44acdf7c52ed0.jpg)

![67ae817c208fc6a762fd400b83ec6d645b61939dad8a400d1cf2d4a3de761b50.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/67ae817c208fc6a762fd400b83ec6d645b61939dad8a400d1cf2d4a3de761b50.jpg)

![6b6fd6b48048afb9b54fc6ccfdf4f74c3dd5092301f86dd868dfbd12b966db0a.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/6b6fd6b48048afb9b54fc6ccfdf4f74c3dd5092301f86dd868dfbd12b966db0a.jpg)

![6e622c742ac26bf36a429439d00476c2a427dc2908e5f34e6b43f7c2d4d869e1.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/6e622c742ac26bf36a429439d00476c2a427dc2908e5f34e6b43f7c2d4d869e1.jpg)

![81ce37d4e3497f49a20dc7c40ecf82489834691fa7908adfcd347fd341d13a5a.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/81ce37d4e3497f49a20dc7c40ecf82489834691fa7908adfcd347fd341d13a5a.jpg)

![87b2d1ddeb13ffd46d3bd977b352373eec47358de9f41333af259d0123529b67.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/87b2d1ddeb13ffd46d3bd977b352373eec47358de9f41333af259d0123529b67.jpg)

![96e0469523f653017b592b2d4f6efe0cf51db075de7a57d02ee35bd6d1846f81.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/96e0469523f653017b592b2d4f6efe0cf51db075de7a57d02ee35bd6d1846f81.jpg)

![af9c04d087cbca4422a10dc28f1e3ff500b14cd70726b73a99bda634e5920651.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/af9c04d087cbca4422a10dc28f1e3ff500b14cd70726b73a99bda634e5920651.jpg)

![b0ef36ebca3b4a5c4def5aa4bda0b0757716d038493ecd60ccf414f51afc4cf9.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/b0ef36ebca3b4a5c4def5aa4bda0b0757716d038493ecd60ccf414f51afc4cf9.jpg)

![bcf467ac65fb43185ae558b1da2b3c65ff688fe21662cbb1ae3ebbc98a310cae.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/bcf467ac65fb43185ae558b1da2b3c65ff688fe21662cbb1ae3ebbc98a310cae.jpg)

![cb8fd024e73b649246539216b39ae4554b89fe70dcad6e7a91bf127692aa51f4.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/cb8fd024e73b649246539216b39ae4554b89fe70dcad6e7a91bf127692aa51f4.jpg)

![cf4894f86f1681388bdfc2b6985cf372f946d3bef24c898c88952d5a595d4a2d.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/cf4894f86f1681388bdfc2b6985cf372f946d3bef24c898c88952d5a595d4a2d.jpg)

![cfd086670ffd2fe8d9d4dde111ff93e02ce2097f84436c63026c0d6ccc32d8b0.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/cfd086670ffd2fe8d9d4dde111ff93e02ce2097f84436c63026c0d6ccc32d8b0.jpg)

![d26c3fa6ecd0f2103abe7077294e8d76c892ab74ea34389c8a3e39863e489b8f.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/d26c3fa6ecd0f2103abe7077294e8d76c892ab74ea34389c8a3e39863e489b8f.jpg)

![e5884af87d7ee7c99b63220f179b556174c928d3ec8c0ead0260e9a789046c2a.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/e5884af87d7ee7c99b63220f179b556174c928d3ec8c0ead0260e9a789046c2a.jpg)

![e59dd4f0d256d77f6da0afbea491d0b09afece57814832fd95e5f4ac365986c9.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/e59dd4f0d256d77f6da0afbea491d0b09afece57814832fd95e5f4ac365986c9.jpg)

![ec6326b98114badb2fcefadbbe713a125afb90ee40108484d62f747810d82fd2.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/ec6326b98114badb2fcefadbbe713a125afb90ee40108484d62f747810d82fd2.jpg)

![ecf8a4321f2b0863218a86109f97dc5f8d77e85943fd6abc9fc75f61deb4c413.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/ecf8a4321f2b0863218a86109f97dc5f8d77e85943fd6abc9fc75f61deb4c413.jpg)

![f33aa30aa7271a06ebb43d3d40bac89ddc71452d78d5cac7d0b72b14712a45f2.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/f33aa30aa7271a06ebb43d3d40bac89ddc71452d78d5cac7d0b72b14712a45f2.jpg)

![f80cc3725d7602001e061484161d78880fc6b8eacef066fc70247b8d19c29e71.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/f80cc3725d7602001e061484161d78880fc6b8eacef066fc70247b8d19c29e71.jpg)

![f9c7f8b84aea217f586d8bbcc285361318510f8053e0961ca1f523d146562551.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/images/f9c7f8b84aea217f586d8bbcc285361318510f8053e0961ca1f523d146562551.jpg)

### Tables

![076c74435fbfbcdf6a7fcb6188c18f2dcb33dad2bd57c059e79a082d6bfbefce.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/076c74435fbfbcdf6a7fcb6188c18f2dcb33dad2bd57c059e79a082d6bfbefce.jpg)

![232b646e564707b1d6d2739bb961e825632a56a8f54e209ae4d742844f1263c9.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/232b646e564707b1d6d2739bb961e825632a56a8f54e209ae4d742844f1263c9.jpg)

![25ebd54a1c3029ec04ecfdcfc40032a102547d3039f03c50f855ed1a43871cbe.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/25ebd54a1c3029ec04ecfdcfc40032a102547d3039f03c50f855ed1a43871cbe.jpg)

![2f86a1e7da4b81808d4f67067f44af7a75e40ab6de2ba4bcd8d64adf3f84e92e.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/2f86a1e7da4b81808d4f67067f44af7a75e40ab6de2ba4bcd8d64adf3f84e92e.jpg)

![4e2860e867dcec9e5dfbd11f2b1bd8952ef3d5451f443d826530ee028f5e82ad.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/4e2860e867dcec9e5dfbd11f2b1bd8952ef3d5451f443d826530ee028f5e82ad.jpg)

![68fb34144510d15da04a008aaa15fe98b9fc85ed9248ee080f579b64547589e5.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/68fb34144510d15da04a008aaa15fe98b9fc85ed9248ee080f579b64547589e5.jpg)

![74d256e7cd8709a4ae32f5c3f9c72fb7a3ebd8b54bd03bfdcfb8f0f0fe75d3bd.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/74d256e7cd8709a4ae32f5c3f9c72fb7a3ebd8b54bd03bfdcfb8f0f0fe75d3bd.jpg)

![94d1d84d96f6857bba175fedd7cd952ca019ee22834a285ca4eec8895566da15.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/94d1d84d96f6857bba175fedd7cd952ca019ee22834a285ca4eec8895566da15.jpg)

![aee6ca4ea59a416f74f2ebe349d39461eeb4888915f220d3011dacf8d40923ef.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/aee6ca4ea59a416f74f2ebe349d39461eeb4888915f220d3011dacf8d40923ef.jpg)

![cce426e2e2c71d59a849e80e2e01d96598328832853509fbeae34fa537de74da.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/cce426e2e2c71d59a849e80e2e01d96598328832853509fbeae34fa537de74da.jpg)

![e13ee42fb34d02a1055fc1462518aa154a24dc0acb330a2ebdfafff2c5292a9e.jpg](../icml_results/5460_Thinking%20in%20Structures_%20Evaluating%20Spatial%20Intelligence%20in%20Constraint-Governed%20Spaces/tables/e13ee42fb34d02a1055fc1462518aa154a24dc0acb330a2ebdfafff2c5292a9e.jpg)

## Concept-Guided Tokenization: Closing the Gap Between Reconstruction and Generation


### Images

![0bb441ac5099c65c824b14ff3b0266173d65a8fa869f760abea2598d31472a97.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/0bb441ac5099c65c824b14ff3b0266173d65a8fa869f760abea2598d31472a97.jpg)

![3f43181823341a7962977d0755cee0e584c694490bd2692e600e8c8388000751.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/3f43181823341a7962977d0755cee0e584c694490bd2692e600e8c8388000751.jpg)

![53d19f0426dbc5004f99fff2a0cb78d561fce9be176273722028c67ccdd64410.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/53d19f0426dbc5004f99fff2a0cb78d561fce9be176273722028c67ccdd64410.jpg)

![83e08e93a2cb2216a8f40277ab36ee48019775e4d8633940455adedfbe889efc.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/83e08e93a2cb2216a8f40277ab36ee48019775e4d8633940455adedfbe889efc.jpg)

![850cf1e1de9f9ebb6bf2196ed30613bc670eb8cfacb8403b6f79476d8d81bdb3.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/850cf1e1de9f9ebb6bf2196ed30613bc670eb8cfacb8403b6f79476d8d81bdb3.jpg)

![899a8abe6443ebc1760e35830853f843a3a76a45e3c13ed8fb9f686c2ee2263a.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/899a8abe6443ebc1760e35830853f843a3a76a45e3c13ed8fb9f686c2ee2263a.jpg)

![8fde113e83a4d3a88c7caac6c38af154f84ca011493b10c0a1c4f727bb40c5ca.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/8fde113e83a4d3a88c7caac6c38af154f84ca011493b10c0a1c4f727bb40c5ca.jpg)

![9ba7fa4cdecbfcfdaf04a2066d146d863a47e4bbd9003b00da650fb82638bd4d.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/9ba7fa4cdecbfcfdaf04a2066d146d863a47e4bbd9003b00da650fb82638bd4d.jpg)

![bd673a9cae2f55b224a74702b163ca39d832c0114eac95004f85d7a9ce58ac63.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/bd673a9cae2f55b224a74702b163ca39d832c0114eac95004f85d7a9ce58ac63.jpg)

![f23f543aaf75f521ac81bedb1c8e96aa8edcc090c16e8bf496f9b46b3706a0ad.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/f23f543aaf75f521ac81bedb1c8e96aa8edcc090c16e8bf496f9b46b3706a0ad.jpg)

![f921268b9b8ad2289a5cec65bae399587aa9e08741e11730d19e301d64a1a132.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/images/f921268b9b8ad2289a5cec65bae399587aa9e08741e11730d19e301d64a1a132.jpg)

### Tables

![1f32dd83c3ad795ab7237bf35416876a255860ceaa094eb34c9b8779ea4ee801.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/1f32dd83c3ad795ab7237bf35416876a255860ceaa094eb34c9b8779ea4ee801.jpg)

![24c0c9e9467ab97da532eb07756c8bded580139a23296f94de323318567ef38a.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/24c0c9e9467ab97da532eb07756c8bded580139a23296f94de323318567ef38a.jpg)

![3cca882aab9af9cbc3e6a82997ab9792b84d9818dfdae0c8649622b91f646460.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/3cca882aab9af9cbc3e6a82997ab9792b84d9818dfdae0c8649622b91f646460.jpg)

![714189e6988b0a8f300ce34dc47d37ee07bbac168e46a08f60382a869231d027.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/714189e6988b0a8f300ce34dc47d37ee07bbac168e46a08f60382a869231d027.jpg)

![828a0c6bfb7efce8cfa875db8d1c7183d4418c65621d569c74389c898daecfb5.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/828a0c6bfb7efce8cfa875db8d1c7183d4418c65621d569c74389c898daecfb5.jpg)

![8eefca22a3d8dbf4bd127ce0c88823632797ceab216648ecab6a9296ad7f6405.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/8eefca22a3d8dbf4bd127ce0c88823632797ceab216648ecab6a9296ad7f6405.jpg)

![9430d8627fdd1c6efb99ca544f9eeb73bf336791226b285b28fe6a49f136eed2.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/9430d8627fdd1c6efb99ca544f9eeb73bf336791226b285b28fe6a49f136eed2.jpg)

![a9a9be098f6f544c40a5d6bd45fe36ebc8fe7615387336b8738f706d6c121f1b.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/a9a9be098f6f544c40a5d6bd45fe36ebc8fe7615387336b8738f706d6c121f1b.jpg)

![bf9f2e74bfc40daf0b4e99052d672dd07748eab874324127ec87bfce4d7f7d2c.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/bf9f2e74bfc40daf0b4e99052d672dd07748eab874324127ec87bfce4d7f7d2c.jpg)

![c4e64dd4c5300f64fc6de174223e675e09f4eebf7543befe20b919e268f2a607.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/c4e64dd4c5300f64fc6de174223e675e09f4eebf7543befe20b919e268f2a607.jpg)

![d253ae2093e32d741ea3f497b4f0672efbc3b5077ed749c592804c382f2c971b.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/d253ae2093e32d741ea3f497b4f0672efbc3b5077ed749c592804c382f2c971b.jpg)

![ed74438311727a7a4d51ccb5a4ea8d0b51b30a9b49ea60b9d37c16153b20f527.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/ed74438311727a7a4d51ccb5a4ea8d0b51b30a9b49ea60b9d37c16153b20f527.jpg)

![f951b4f8654660adeaab2c9f8da836ce75a0fb31921d5c858b1db82c44feb206.jpg](../icml_results/5461_Concept-Guided%20Tokenization_%20Closing%20the%20Gap%20Between%20Reconstruction%20and%20Generation/tables/f951b4f8654660adeaab2c9f8da836ce75a0fb31921d5c858b1db82c44feb206.jpg)

<a id="paper-5414"></a>

## Beyond Description: Federated Adaptation via Semantic-Visual Prototype Alignment


### Images

![13b5777e73979c9ed43e652f5d6d133c1345a1a9e661fe525167a56731f924c6.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/images/13b5777e73979c9ed43e652f5d6d133c1345a1a9e661fe525167a56731f924c6.jpg)

![53478e8134ca29329fa0a1db1b0caf66149dfc12c7d4499eca2d7b51ba1934c6.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/images/53478e8134ca29329fa0a1db1b0caf66149dfc12c7d4499eca2d7b51ba1934c6.jpg)

![6119af1e2602907a618f2b25a2d23fe65fad43eb24bba86b96e6600cae8967d6.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/images/6119af1e2602907a618f2b25a2d23fe65fad43eb24bba86b96e6600cae8967d6.jpg)

![7741c3327178652be824bddaa588a4d33bc000f01dac47195d96566c1c5d2d0e.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/images/7741c3327178652be824bddaa588a4d33bc000f01dac47195d96566c1c5d2d0e.jpg)

![d7a154ce214c3da35c054b9d13036c7b053fe9129de857c669562c0ab4e6d1bc.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/images/d7a154ce214c3da35c054b9d13036c7b053fe9129de857c669562c0ab4e6d1bc.jpg)

### Tables

![06be9c005db40a43f2ec6a69490c62bbd7d750d5996b183a8c633d70ba0355f6.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/06be9c005db40a43f2ec6a69490c62bbd7d750d5996b183a8c633d70ba0355f6.jpg)

![1fd46dc3d99e9325d00091637acadef8e5fa761dcb4da193e9dba1553960db18.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/1fd46dc3d99e9325d00091637acadef8e5fa761dcb4da193e9dba1553960db18.jpg)

![2be5b49a86d503569d6da63d7999346bc37d88ca66b80789663ca99842272ca4.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/2be5b49a86d503569d6da63d7999346bc37d88ca66b80789663ca99842272ca4.jpg)

![2c46df8baa585fd6bb12d1da695372c368192dd71256a3732023fe96d275862a.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/2c46df8baa585fd6bb12d1da695372c368192dd71256a3732023fe96d275862a.jpg)

![51c85501d6d72c3522f62d63c81cc1cc73cd91ffe089c1e9e86be15803cbb436.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/51c85501d6d72c3522f62d63c81cc1cc73cd91ffe089c1e9e86be15803cbb436.jpg)

![51dcbd33ab34d7a78278a2604bd9bf6a3f46da613067a2c6dcbb180b672e6a1e.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/51dcbd33ab34d7a78278a2604bd9bf6a3f46da613067a2c6dcbb180b672e6a1e.jpg)

![5d8d94d8c62777c737d48df5ed19c0507878dc759f88bfce807174a29c505287.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/5d8d94d8c62777c737d48df5ed19c0507878dc759f88bfce807174a29c505287.jpg)

![98dfe51eb811f4bc8c64db9e7ea22b98f8a8cefa52560140ede67a46b08d381d.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/98dfe51eb811f4bc8c64db9e7ea22b98f8a8cefa52560140ede67a46b08d381d.jpg)

![9f7a225cf61c7ce3de62e70f5ceeebf41258dd72dc1def86772180b8ee31cf4f.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/9f7a225cf61c7ce3de62e70f5ceeebf41258dd72dc1def86772180b8ee31cf4f.jpg)

![a73c2d787bd5ea7cb473d13d75828f8c5b98537ab850b32bd8fb4b6a3d0b69a9.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/a73c2d787bd5ea7cb473d13d75828f8c5b98537ab850b32bd8fb4b6a3d0b69a9.jpg)

![bc5b7d20d91c6265afb0787a047053c91e5517913c8866fc25b9deef5582324a.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/bc5b7d20d91c6265afb0787a047053c91e5517913c8866fc25b9deef5582324a.jpg)

![f4cefa1adc33342caa2c14bb7b922a65c3faae0bd4d502a696c696929597db6d.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/f4cefa1adc33342caa2c14bb7b922a65c3faae0bd4d502a696c696929597db6d.jpg)

![fee50a071dbc57f9a16ebcd5b7b2c9db36c2d764b0b60c61d6cefa516c8f7d98.jpg](../icml_results/5462_Beyond%20Description_%20Federated%20Adaptation%20via%20Semantic-Visual%20Prototype%20Alignment/tables/fee50a071dbc57f9a16ebcd5b7b2c9db36c2d764b0b60c61d6cefa516c8f7d98.jpg)

<a id="paper-5415"></a>

## Second-Order Bilevel Optimization with Accelerated Convergence Rates


### Images

![35a567ec29096425875a864568e27366b6aef035a4f31c0d94c68529f9850eaf.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/35a567ec29096425875a864568e27366b6aef035a4f31c0d94c68529f9850eaf.jpg)

![3793243fc4573adc321c919588756cec809e5f20ebdf7a7372d844630c0646a8.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/3793243fc4573adc321c919588756cec809e5f20ebdf7a7372d844630c0646a8.jpg)

![3a5b6312ac9043c73a9ffdc908733a22813db100c2352e4c45cc01564b5d08c0.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/3a5b6312ac9043c73a9ffdc908733a22813db100c2352e4c45cc01564b5d08c0.jpg)

![898346bdada1f2c520b73f6c6847130544cad8a9532ca7ba0427cd08996ee832.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/898346bdada1f2c520b73f6c6847130544cad8a9532ca7ba0427cd08996ee832.jpg)

![b2d7c26e93a6106b6db49b61f193782b14423fbcbef728cd776b34de7ca02746.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/b2d7c26e93a6106b6db49b61f193782b14423fbcbef728cd776b34de7ca02746.jpg)

![cab12e0336af247f02e1aa482347177d92ceb53eca78f3d560e49c4be9c399b1.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/cab12e0336af247f02e1aa482347177d92ceb53eca78f3d560e49c4be9c399b1.jpg)

![ce6a181c505874f5769ff97fbe5180c7aed0c30b9fb6d0a390e558f09ad349b7.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/ce6a181c505874f5769ff97fbe5180c7aed0c30b9fb6d0a390e558f09ad349b7.jpg)

![f5f28b647bc2720914acd7d205979d3e9e1180f385e7fa2057deaa9d1105cda3.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/images/f5f28b647bc2720914acd7d205979d3e9e1180f385e7fa2057deaa9d1105cda3.jpg)

### Tables

![369bc80d8d8b6e71f187992bab3ba5499f72540193fb138fe5f2370f2abf6bf9.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/tables/369bc80d8d8b6e71f187992bab3ba5499f72540193fb138fe5f2370f2abf6bf9.jpg)

![ae096eeda102a18c3400e636044f9548a47c3d1a6f66034d33ef2fc3ef52e5a6.jpg](../icml_results/5463_Second-Order%20Bilevel%20Optimization%20with%20Accelerated%20Convergence%20Rates/tables/ae096eeda102a18c3400e636044f9548a47c3d1a6f66034d33ef2fc3ef52e5a6.jpg)

<a id="paper-5416"></a>

## From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning


### Images

![0e55cc97ed1bff7f829d9cccca502abfde3b033a76651f0d9bd9420cf1e7838c.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/0e55cc97ed1bff7f829d9cccca502abfde3b033a76651f0d9bd9420cf1e7838c.jpg)

![119a71c3ea1cc9af59e9f526b7bbf3840a076693e7ccf56b5c8bc73eec3265a6.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/119a71c3ea1cc9af59e9f526b7bbf3840a076693e7ccf56b5c8bc73eec3265a6.jpg)

![19c081a38bb478385aaa51bc5aa8708723f1d18f215b915efb9342f6e187b59a.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/19c081a38bb478385aaa51bc5aa8708723f1d18f215b915efb9342f6e187b59a.jpg)

![1f81fb8de65c0428eacaf7dacc24a4a3f7eec16289627e771accad9f7188e591.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/1f81fb8de65c0428eacaf7dacc24a4a3f7eec16289627e771accad9f7188e591.jpg)

![24a80b2fb264a5b68fd3b82ef811aeb3bbb308d90c816e07582a85befee8cdfc.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/24a80b2fb264a5b68fd3b82ef811aeb3bbb308d90c816e07582a85befee8cdfc.jpg)

![26d011fb294e0e68b0e13a6053b609d3b2bd58a1c4d079a657fdf7c4d2eb237b.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/26d011fb294e0e68b0e13a6053b609d3b2bd58a1c4d079a657fdf7c4d2eb237b.jpg)

![2c5d413f583fc7afab9e476e220b421f3d5ba8bd34866a9183f7941bcaedc877.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/2c5d413f583fc7afab9e476e220b421f3d5ba8bd34866a9183f7941bcaedc877.jpg)

![4f64c90fca4598d47f0c1ac91acd2797441532a05599f525b1ab5d8d89bfd210.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/4f64c90fca4598d47f0c1ac91acd2797441532a05599f525b1ab5d8d89bfd210.jpg)

![5d6fcb3b62b8ab33478025791120d074cb81171bb071162b30c5ca98be576a6e.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/5d6fcb3b62b8ab33478025791120d074cb81171bb071162b30c5ca98be576a6e.jpg)

![6e0f171496d447c7f379e254d3ccdef6bc783def794fa628052824dba51e6132.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/6e0f171496d447c7f379e254d3ccdef6bc783def794fa628052824dba51e6132.jpg)

![7b88f7f18105147318b2731a65025db449dad786ee17330b1991e0a6cdc3552f.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/7b88f7f18105147318b2731a65025db449dad786ee17330b1991e0a6cdc3552f.jpg)

![8fb89b69033f47b0294943b38d3c51cc617583da2ff9ca9ca8cf66838851e115.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/8fb89b69033f47b0294943b38d3c51cc617583da2ff9ca9ca8cf66838851e115.jpg)

![a5f4eb63c29980ae0d345aa8a1919fcb6387a3722afc6a62be2751a9eb8941cf.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/a5f4eb63c29980ae0d345aa8a1919fcb6387a3722afc6a62be2751a9eb8941cf.jpg)

![a7f169d8608c3e6996b30b570ae46e376bb1231500be92f35c0ae8ca5c47bf9b.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/a7f169d8608c3e6996b30b570ae46e376bb1231500be92f35c0ae8ca5c47bf9b.jpg)

![a8c813fae37d82b1c859397861938f43051d58783f0d47200b22cf0c730d2765.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/a8c813fae37d82b1c859397861938f43051d58783f0d47200b22cf0c730d2765.jpg)

![aa520d62e24d0960818db2f6781246aab0441669c75496622317beb87589c43c.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/aa520d62e24d0960818db2f6781246aab0441669c75496622317beb87589c43c.jpg)

![bffc05039e131424d6593505b8b773390a5918d6dc95f673eb4ae9071752a968.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/bffc05039e131424d6593505b8b773390a5918d6dc95f673eb4ae9071752a968.jpg)

![cfdacfb7a4229ff97029dd4238e095807b5348cd8febdab85a5ca824d22e19ab.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/images/cfdacfb7a4229ff97029dd4238e095807b5348cd8febdab85a5ca824d22e19ab.jpg)

### Tables

![06f7783d3c4906a6f2dbfd212574d12eb2f8106c4d6e2fe307984a224b3a5078.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/tables/06f7783d3c4906a6f2dbfd212574d12eb2f8106c4d6e2fe307984a224b3a5078.jpg)

![b026709d794d999fb62c99b73cf08fdc8f07b1f2b953ac8e36ce34c20b0e556a.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/tables/b026709d794d999fb62c99b73cf08fdc8f07b1f2b953ac8e36ce34c20b0e556a.jpg)

![fba9664f61de33cfdeb7d038a411b8a41f8cfb90ee9341dc8229c9118f841159.jpg](../icml_results/5464_From%20Perception%20to%20Planning_%20Evolving%20Ego-Centric%20Task-Oriented%20Spatiotemporal%20Reasoning%20via%20Curricu/tables/fba9664f61de33cfdeb7d038a411b8a41f8cfb90ee9341dc8229c9118f841159.jpg)

<a id="paper-5417"></a>

## Step-Level Sparse Autoencoder for Reasoning Process Interpretation


### Images

![28b5673d5007cd3754a04c85d379f8e67e686ae98944c4f0e2823559426bcdcd.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/images/28b5673d5007cd3754a04c85d379f8e67e686ae98944c4f0e2823559426bcdcd.jpg)

![4ec24f01141e52bc7e0bb0ec1ed81967fe8e45f34437798656ecf7562b18200c.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/images/4ec24f01141e52bc7e0bb0ec1ed81967fe8e45f34437798656ecf7562b18200c.jpg)

![986fd233e7648f9d8f7a18250e4aa68544374705c26eaff6c0e4a20d1009c0bf.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/images/986fd233e7648f9d8f7a18250e4aa68544374705c26eaff6c0e4a20d1009c0bf.jpg)

![a8a2c97382e382709d382f2cbffd6ef5849e731a74c9bd1a9af97d1927714b4a.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/images/a8a2c97382e382709d382f2cbffd6ef5849e731a74c9bd1a9af97d1927714b4a.jpg)

### Tables

![07582724b3c7d9a651281ca0add7a2f35e251ba6e63d93e7deb7965a54b78740.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/07582724b3c7d9a651281ca0add7a2f35e251ba6e63d93e7deb7965a54b78740.jpg)

![1c2568fb11694a427347cbba6014761630e950c64c7beb6e3a4cd3f4fe9c5b5a.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/1c2568fb11694a427347cbba6014761630e950c64c7beb6e3a4cd3f4fe9c5b5a.jpg)

![84a2e93ccc9b1d548a215907ff246d61c7e320dc48e5061f3ddb60fa04600cdb.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/84a2e93ccc9b1d548a215907ff246d61c7e320dc48e5061f3ddb60fa04600cdb.jpg)

![c4e4bf7fb869447aa868fe37ca5dc9f74c1d43ee75194d3574500fe1a8a2fefd.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/c4e4bf7fb869447aa868fe37ca5dc9f74c1d43ee75194d3574500fe1a8a2fefd.jpg)

![d0da51e0823e802c2b97948d52bae1fecf856f5c6f4eed57b2a8e53bd14d5940.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/d0da51e0823e802c2b97948d52bae1fecf856f5c6f4eed57b2a8e53bd14d5940.jpg)

![f82e296438f86ca5992cfaf0f0165055d5903a71a6fbee2026af723bf87aca80.jpg](../icml_results/5465_Step-Level%20Sparse%20Autoencoder%20for%20Reasoning%20Process%20Interpretation/tables/f82e296438f86ca5992cfaf0f0165055d5903a71a6fbee2026af723bf87aca80.jpg)

<a id="paper-5418"></a>

## Counterfactual Occlusion-Aware Learning via Visibility Intervention for LiDAR Anomaly Detection


### Images

![0a3d9c4cfc6de4b347708177e7e7dfff504e361e76c79e304a010c051acc9741.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/0a3d9c4cfc6de4b347708177e7e7dfff504e361e76c79e304a010c051acc9741.jpg)

![19bcdf3e290af94d26a1b938cc3e178f3defed4a742fa345f09f9395cd3aa0e8.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/19bcdf3e290af94d26a1b938cc3e178f3defed4a742fa345f09f9395cd3aa0e8.jpg)

![2d8c566cd86294693a6038800f82cc0d4969334bd9058fab94ec737ea8fe36d1.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/2d8c566cd86294693a6038800f82cc0d4969334bd9058fab94ec737ea8fe36d1.jpg)

![5940a899a1e18d0d8e10d74df478ca012f9ecf7db7681861fbbd37216dd35585.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/5940a899a1e18d0d8e10d74df478ca012f9ecf7db7681861fbbd37216dd35585.jpg)

![63e835197f0b8d1e850aa0caf8d42193003fdeb5cf9f9ad53060b0d3c4e0224c.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/63e835197f0b8d1e850aa0caf8d42193003fdeb5cf9f9ad53060b0d3c4e0224c.jpg)

![8540f025d499fdd699ec7c962dbc69f9e84767de2e3519349bacd74b50861cff.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/8540f025d499fdd699ec7c962dbc69f9e84767de2e3519349bacd74b50861cff.jpg)

![b4c49314c4d175e3d53747263026e444656eda036b9b08c5fa7b5be9730969bb.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/b4c49314c4d175e3d53747263026e444656eda036b9b08c5fa7b5be9730969bb.jpg)

![bb057e6a01f36e4d2c73763000898204323af2f6bcf8035792578ab65ab9bd31.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/bb057e6a01f36e4d2c73763000898204323af2f6bcf8035792578ab65ab9bd31.jpg)

![f76fd026cd4b43d9b5b97463387b555463827f3c2efbbae4bd56fa9969e858df.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/images/f76fd026cd4b43d9b5b97463387b555463827f3c2efbbae4bd56fa9969e858df.jpg)

### Tables

![00b949816545d667d84ce5c842ce105bb9f666f1b43328eb6bdafd87d3bb8eac.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/00b949816545d667d84ce5c842ce105bb9f666f1b43328eb6bdafd87d3bb8eac.jpg)

![0f1d580a98731f461836668af8028e0f63b9fc791367d7e000334d7867b01356.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/0f1d580a98731f461836668af8028e0f63b9fc791367d7e000334d7867b01356.jpg)

![3704f3a4abe84a29a27ba90bde84193a6bd36c8e331991f9b470acaf01fb19ea.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/3704f3a4abe84a29a27ba90bde84193a6bd36c8e331991f9b470acaf01fb19ea.jpg)

![3fbf1ae36397f4da1c01951ba393c68796ffb58b973a1795fd48688f8fb3c659.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/3fbf1ae36397f4da1c01951ba393c68796ffb58b973a1795fd48688f8fb3c659.jpg)

![5b55428bf95ea840e804a09ede66e654c3a111b6de1d56be577585489fc35e23.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/5b55428bf95ea840e804a09ede66e654c3a111b6de1d56be577585489fc35e23.jpg)

![8a9ea99fa4ca10ffc3a5a338811897b0c8efa04614a3df879600e2274ec4c134.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/8a9ea99fa4ca10ffc3a5a338811897b0c8efa04614a3df879600e2274ec4c134.jpg)

![9fe28b392886dc7d30af973a8fb9e12bc6512b4f288e6389b9ccdf6c83991ce7.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/9fe28b392886dc7d30af973a8fb9e12bc6512b4f288e6389b9ccdf6c83991ce7.jpg)

![a0e1877debdd159e13503a913dabc535b9d9c0aaca0de0c4ea934bf36dca00e7.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/a0e1877debdd159e13503a913dabc535b9d9c0aaca0de0c4ea934bf36dca00e7.jpg)

![b37590180ed51e5ca2a8b86cd1fc9c59e34d6d07e28b1430f32f50379c69843a.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/b37590180ed51e5ca2a8b86cd1fc9c59e34d6d07e28b1430f32f50379c69843a.jpg)

![b71b4f1d80ee76d38e40735da0953bccf6ed7f0864248edffa85c27dc4d7168a.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/b71b4f1d80ee76d38e40735da0953bccf6ed7f0864248edffa85c27dc4d7168a.jpg)

![c653310ba88632e11a9e8b41a9ab3b65be9333b9ce322459251f768b463b55f8.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/c653310ba88632e11a9e8b41a9ab3b65be9333b9ce322459251f768b463b55f8.jpg)

![e4020f6d59a50766d650868d48d5bb424d9b62fdc016c345a55c34bb8f2898ca.jpg](../icml_results/5466_Counterfactual%20Occlusion-Aware%20Learning%20via%20Visibility%20Intervention%20for%20LiDAR%20Anomaly%20Detection/tables/e4020f6d59a50766d650868d48d5bb424d9b62fdc016c345a55c34bb8f2898ca.jpg)

<a id="paper-5419"></a>

## One-Way Policy Optimization for Self-Evolving LLMs


### Images

![17dab0c84beae6e9a2944aafe742a414253c427cdac3d3bd6b8feeea353eca4b.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/17dab0c84beae6e9a2944aafe742a414253c427cdac3d3bd6b8feeea353eca4b.jpg)

![1a495fbb20968bb3222e5d97616b1cb8bc4c927181bbc0f323e1e03ffa8d3c58.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/1a495fbb20968bb3222e5d97616b1cb8bc4c927181bbc0f323e1e03ffa8d3c58.jpg)

![77afdd40b5114ce474a5563ec15a4ea55bddcef03283ac729d8afc62dc3b22c8.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/77afdd40b5114ce474a5563ec15a4ea55bddcef03283ac729d8afc62dc3b22c8.jpg)

![7c5e8ffea36d9921bbf72f42ff7a0248d11a5e65d9a965cfa7dbf97ba1ce83a4.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/7c5e8ffea36d9921bbf72f42ff7a0248d11a5e65d9a965cfa7dbf97ba1ce83a4.jpg)

![b8a20efce43299bcb239c8ac25199bf2035ae54965c9b282b77dfe8d6595e578.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/b8a20efce43299bcb239c8ac25199bf2035ae54965c9b282b77dfe8d6595e578.jpg)

![d3b64501952273b288d3624549a172db96f76f8c905042f07019494d36ec95a7.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/images/d3b64501952273b288d3624549a172db96f76f8c905042f07019494d36ec95a7.jpg)

### Tables

![092b16ace3c7cc247bab15f6db0ee7aa00611158e175c84abd3d0bc74e39580c.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/092b16ace3c7cc247bab15f6db0ee7aa00611158e175c84abd3d0bc74e39580c.jpg)

![48820f438f6b5648ef8ade7eaf01b9b204c197af8a4905f301248effff196055.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/48820f438f6b5648ef8ade7eaf01b9b204c197af8a4905f301248effff196055.jpg)

![85c378d4b77a219a540002648956bfc83def3dfd6ebdfa8da01e92c8c9179da9.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/85c378d4b77a219a540002648956bfc83def3dfd6ebdfa8da01e92c8c9179da9.jpg)

![c56e8cd9d9c283c8d44e5f8a1a5254037df73a8932cc44f85ad4911be68a1517.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/c56e8cd9d9c283c8d44e5f8a1a5254037df73a8932cc44f85ad4911be68a1517.jpg)

![d6a0bffffb7d84a7877cc9c6e1aec86dac74b6ed0c372945382e6fa4c11a0362.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/d6a0bffffb7d84a7877cc9c6e1aec86dac74b6ed0c372945382e6fa4c11a0362.jpg)

![ff279eb502d974a9b4b8e05849efd3708a4cabd361238e5a74fef4429a374589.jpg](../icml_results/5467_One-Way%20Policy%20Optimization%20for%20Self-Evolving%20LLMs/tables/ff279eb502d974a9b4b8e05849efd3708a4cabd361238e5a74fef4429a374589.jpg)

<a id="paper-5420"></a>

## Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals


### Images

![25a705ea8f1e86f9294278cc350d5090b62971d6ab7e6bbc6f4d6a942b6dffd3.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/25a705ea8f1e86f9294278cc350d5090b62971d6ab7e6bbc6f4d6a942b6dffd3.jpg)

![44d320dfc30f0d93d5d6b8611abb2f28f5cac27076a44aa32873801277ae9ea4.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/44d320dfc30f0d93d5d6b8611abb2f28f5cac27076a44aa32873801277ae9ea4.jpg)

![5346e186393431cef698711133cba2e60b4709da05a5f1970553d798cfb4d9d5.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/5346e186393431cef698711133cba2e60b4709da05a5f1970553d798cfb4d9d5.jpg)

![5d35e1eae482f8ec7156ce07a5249431904825396895743a354ced8f115a7b49.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/5d35e1eae482f8ec7156ce07a5249431904825396895743a354ced8f115a7b49.jpg)

![9447b863fdc7e0141c2782213b0afac8c8a0294f2a8335d44620428ab3a4818a.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/9447b863fdc7e0141c2782213b0afac8c8a0294f2a8335d44620428ab3a4818a.jpg)

![cda31e1b3db1403c1befc167727b404a6777e6571cc4a9484db7ceeb7e5c1a02.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/cda31e1b3db1403c1befc167727b404a6777e6571cc4a9484db7ceeb7e5c1a02.jpg)

![d02bf88e151fcb469efb085997eb639168c087b075fbe5f5a7c328f9b848fd91.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/d02bf88e151fcb469efb085997eb639168c087b075fbe5f5a7c328f9b848fd91.jpg)

![d3ae6bed6714167128e15177cc82c6ccdb715fd02ce5b11bd62ca1ebcdc84dd6.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/d3ae6bed6714167128e15177cc82c6ccdb715fd02ce5b11bd62ca1ebcdc84dd6.jpg)

![e44f1624cef280a4abff3604395ddd10c49b1c02ad7ce5a2a7aaad08c15ed80f.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/e44f1624cef280a4abff3604395ddd10c49b1c02ad7ce5a2a7aaad08c15ed80f.jpg)

![ee020fc13664728a1e2967e8e350c0c9c7da0289b16d356004680093dc5302ba.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/ee020fc13664728a1e2967e8e350c0c9c7da0289b16d356004680093dc5302ba.jpg)

![faef96c243a1764c4961336ac4f9a0637c8780896fdfb25ca18270d547126146.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/images/faef96c243a1764c4961336ac4f9a0637c8780896fdfb25ca18270d547126146.jpg)

### Tables

![4721d90843876650612e30123ca1f4223399af86eac3859fb390260479de0458.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/tables/4721d90843876650612e30123ca1f4223399af86eac3859fb390260479de0458.jpg)

![dca27a2f4e12e904d2829cef437f6d6e853227b241d08ef73b4eb6bd36ffb42d.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/tables/dca27a2f4e12e904d2829cef437f6d6e853227b241d08ef73b4eb6bd36ffb42d.jpg)

![e3b5a72d5f490d9abf2027ae8d980a5a44cf3be1f67bd77a7c55361a373c4daf.jpg](../icml_results/5468_Clipping%20Bottleneck_%20Stabilizing%20RLVR%20via%20Stochastic%20Recovery%20of%20Near-Boundary%20Signals/tables/e3b5a72d5f490d9abf2027ae8d980a5a44cf3be1f67bd77a7c55361a373c4daf.jpg)

<a id="paper-5421"></a>

## Towards Whole-corpus Reconstruction of Heterogeneous RAG Knowledge Bases


### Images

![0a7f73056550b84f2a084a19865505b2ad7f4d750ee11dcd86b8cc2f72743f88.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/0a7f73056550b84f2a084a19865505b2ad7f4d750ee11dcd86b8cc2f72743f88.jpg)

![0b2d626259938e035321741286bcd1b1d5315f7f6aca43bb62c601ef5785ab0c.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/0b2d626259938e035321741286bcd1b1d5315f7f6aca43bb62c601ef5785ab0c.jpg)

![1a94232385bb4875484206cd458c2e3bbdb4ccc5ae2ba674b022ffd65d90c0c8.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/1a94232385bb4875484206cd458c2e3bbdb4ccc5ae2ba674b022ffd65d90c0c8.jpg)

![2223941d68a292c1e52f2ae6bc8ecfb42a19c6b095aa0b48ae16ae5d4067eb34.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/2223941d68a292c1e52f2ae6bc8ecfb42a19c6b095aa0b48ae16ae5d4067eb34.jpg)

![263f0bfe38272216e6fc066124fc286dea18b59dedb5b8730598a42275e4d5da.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/263f0bfe38272216e6fc066124fc286dea18b59dedb5b8730598a42275e4d5da.jpg)

![456b5494af50ff38401ce6e6acda2caab12afafa2343b1b65fd77ab30484c848.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/456b5494af50ff38401ce6e6acda2caab12afafa2343b1b65fd77ab30484c848.jpg)

![57fcf479b234dc92c6721773aefbd36a1edea4b9e239affddf612defd5591e98.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/57fcf479b234dc92c6721773aefbd36a1edea4b9e239affddf612defd5591e98.jpg)

![84d7d6ea37c5511bf2a83f2ac2eb24e059cda7fc67d50f18a4eadda23eabce58.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/84d7d6ea37c5511bf2a83f2ac2eb24e059cda7fc67d50f18a4eadda23eabce58.jpg)

![84dbbe25e4f34f0dd8c93b1a0ab6ad29e4c5fdadd6cce2b889609aae7b191622.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/84dbbe25e4f34f0dd8c93b1a0ab6ad29e4c5fdadd6cce2b889609aae7b191622.jpg)

![a2da539a2ae9fc63467c8c7c07018f726815f96e87ec19ac216a85d44a776275.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/a2da539a2ae9fc63467c8c7c07018f726815f96e87ec19ac216a85d44a776275.jpg)

![b81b34c82472708c0a0d31c1f9b54c1e52a176395bd0ed7b32f97a72f948cdb0.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/b81b34c82472708c0a0d31c1f9b54c1e52a176395bd0ed7b32f97a72f948cdb0.jpg)

![c5416155a3a696318aa72991c3a48fa85d6fc9eb98e1fe8bc17315bda6cd3791.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/c5416155a3a696318aa72991c3a48fa85d6fc9eb98e1fe8bc17315bda6cd3791.jpg)

![c8b285ce5bcfc4f06afbadceb3ad1746d2dbaf9b12db6fc33063e86a21edc40e.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/c8b285ce5bcfc4f06afbadceb3ad1746d2dbaf9b12db6fc33063e86a21edc40e.jpg)

![d1fccf3e01064093d25f807aa80f94dc44363d335a37ccd8473bfd9ed25c587b.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/d1fccf3e01064093d25f807aa80f94dc44363d335a37ccd8473bfd9ed25c587b.jpg)

![f9ee0bd29e4df02f085310f6bce737fb310a12a986c966352a67d82068b6f669.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/images/f9ee0bd29e4df02f085310f6bce737fb310a12a986c966352a67d82068b6f669.jpg)

### Tables

![ee02e5b608853ac9417308f01fda3d9689f663aa517329f28bea454518c0965e.jpg](../icml_results/5469_Towards%20Whole-corpus%20Reconstruction%20of%20Heterogeneous%20RAG%20Knowledge%20Bases/tables/ee02e5b608853ac9417308f01fda3d9689f663aa517329f28bea454518c0965e.jpg)

<a id="paper-5422"></a>

## Batched Contextual Reinforcement


### Images

![0ba36c52a488004b3e936e7559d92c2e0dc2a6c05c37d7a9519a48897e511a91.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/images/0ba36c52a488004b3e936e7559d92c2e0dc2a6c05c37d7a9519a48897e511a91.jpg)

![0fb325f7f5bd23f6df6fd95a906ff78b3a98a0dca010ca9467385363ec76e795.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/images/0fb325f7f5bd23f6df6fd95a906ff78b3a98a0dca010ca9467385363ec76e795.jpg)

![51c957699a1057fcbfc1c7884c811de1a889c9f9d3f038a8301665794302d367.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/images/51c957699a1057fcbfc1c7884c811de1a889c9f9d3f038a8301665794302d367.jpg)

![6959289155ac007c3854f520e590287018ffc4e65a7f0ac99f2b8d19991b57c3.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/images/6959289155ac007c3854f520e590287018ffc4e65a7f0ac99f2b8d19991b57c3.jpg)

![78efe847a272b4e17b67228ac227fb217f36238da04f5a4614bf7bc64fc33ac9.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/images/78efe847a272b4e17b67228ac227fb217f36238da04f5a4614bf7bc64fc33ac9.jpg)

### Tables

![00c3169ca6e61acd7fe6a2ca0cb6a67011e0ef244e6b62f0c6d4c4ad760de989.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/00c3169ca6e61acd7fe6a2ca0cb6a67011e0ef244e6b62f0c6d4c4ad760de989.jpg)

![0310ab4e90d3225d043b920a43c49dc802880c2acf174a3350991c1467d5fa52.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/0310ab4e90d3225d043b920a43c49dc802880c2acf174a3350991c1467d5fa52.jpg)

![0a53401ee76f76aa57fd6f614abf99577b79b00ee5d687d332169960aea7ee41.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/0a53401ee76f76aa57fd6f614abf99577b79b00ee5d687d332169960aea7ee41.jpg)

![0be7d862d7fa3046a68313c2a3e496df830cbeae0276f81bf8a7c8275ed356b5.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/0be7d862d7fa3046a68313c2a3e496df830cbeae0276f81bf8a7c8275ed356b5.jpg)

![14d8586c25c6f416539e2ae3103b5ab632c8d716f13ae234d98523597cb925fa.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/14d8586c25c6f416539e2ae3103b5ab632c8d716f13ae234d98523597cb925fa.jpg)

![1bf811338a38da8b1e2aab7ab21be2030ffd1cf79db7f733bf5ff522d651be27.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/1bf811338a38da8b1e2aab7ab21be2030ffd1cf79db7f733bf5ff522d651be27.jpg)

![55a49a85e2bb8cd17ab1fd1440924c6fee0dcec6f8729dfcd28c600214992b6d.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/55a49a85e2bb8cd17ab1fd1440924c6fee0dcec6f8729dfcd28c600214992b6d.jpg)

![587534d79ffb7f826fcc6ef9232c23a7ec433452fb7eb35ca2d44bf80fa460d7.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/587534d79ffb7f826fcc6ef9232c23a7ec433452fb7eb35ca2d44bf80fa460d7.jpg)

![60bc5965725df6afbc0cc0b45d5970d20f8643905a7a5205f84f437d36f3765f.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/60bc5965725df6afbc0cc0b45d5970d20f8643905a7a5205f84f437d36f3765f.jpg)

![6849ec26fa736de5740522ed49f327ad28f5cb181bec54cdc79049b3322ad82a.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/6849ec26fa736de5740522ed49f327ad28f5cb181bec54cdc79049b3322ad82a.jpg)

![6d0e9bf0f9c5d387cb6e72ab3ef2475cb06600c8714d053113f71cd34867b7c1.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/6d0e9bf0f9c5d387cb6e72ab3ef2475cb06600c8714d053113f71cd34867b7c1.jpg)

![71850e7846474baa21ea69dac4b84e97295255308e96e6de0806f5797f9050a2.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/71850e7846474baa21ea69dac4b84e97295255308e96e6de0806f5797f9050a2.jpg)

![9092d3603f0f5032ba0dfe3a69e07b6b4f449ca4ac55753aa18c05ad7ea57c0c.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/9092d3603f0f5032ba0dfe3a69e07b6b4f449ca4ac55753aa18c05ad7ea57c0c.jpg)

![9710a171f9259b57a7b0181b289a9034d5cf116342cedc4a72b4b85b260ee8c2.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/9710a171f9259b57a7b0181b289a9034d5cf116342cedc4a72b4b85b260ee8c2.jpg)

![9e2f8a56efb88142986d6cea3761970a80f718ed6f3c20504a59c175bdf0fe83.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/9e2f8a56efb88142986d6cea3761970a80f718ed6f3c20504a59c175bdf0fe83.jpg)

![a05915ab070b05b80e304f14378acf36689558d2e847f433f552204f90c61192.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/a05915ab070b05b80e304f14378acf36689558d2e847f433f552204f90c61192.jpg)

![a2b23f45f60ac09325b571eb46b9b757db2d319c428f41e29d3068ade02dbad0.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/a2b23f45f60ac09325b571eb46b9b757db2d319c428f41e29d3068ade02dbad0.jpg)

![a69917219c919c6f93a5f765d4946038f45e9ddca35dd6d9bea88d4b009b530f.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/a69917219c919c6f93a5f765d4946038f45e9ddca35dd6d9bea88d4b009b530f.jpg)

![a7a031d66a2f402380713d23418af15f12a6cf3c244bc81f5d6edc7aa56cb479.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/a7a031d66a2f402380713d23418af15f12a6cf3c244bc81f5d6edc7aa56cb479.jpg)

![a9631247114d659f45a56521c3b84cff1a4be5b664958b319afb2239f0b6d80f.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/a9631247114d659f45a56521c3b84cff1a4be5b664958b319afb2239f0b6d80f.jpg)

![b3c7ba115bd381f3e726efcff06b928f8abbf48d7e69616e002961cbf23c46b7.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/b3c7ba115bd381f3e726efcff06b928f8abbf48d7e69616e002961cbf23c46b7.jpg)

![b699c6576f2c06d5a4849099446b61b9d743e7aaaf5862d4e2125ff040acd973.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/b699c6576f2c06d5a4849099446b61b9d743e7aaaf5862d4e2125ff040acd973.jpg)

![cde275edbcac24747319356c83c950b7c69d48739df45238ad7ee3cdfc112dc2.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/cde275edbcac24747319356c83c950b7c69d48739df45238ad7ee3cdfc112dc2.jpg)

![cfb92f232d0ccef4f2ddac563d46955300c0ddbbff94d91671c70229a090ddf6.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/cfb92f232d0ccef4f2ddac563d46955300c0ddbbff94d91671c70229a090ddf6.jpg)

![d15e4e981956eeba91ca0fef9a765d2b159437f87bce41df5e805a3f4427a8af.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/d15e4e981956eeba91ca0fef9a765d2b159437f87bce41df5e805a3f4427a8af.jpg)

![d991c450addeac5b970a9eb5e19feaa85ffd8772907f236669099ba0ac899e34.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/d991c450addeac5b970a9eb5e19feaa85ffd8772907f236669099ba0ac899e34.jpg)

![dc009c043a40ef608a512aa883d4f1801eebf86909d15f31b1d76fadcc1a88a5.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/dc009c043a40ef608a512aa883d4f1801eebf86909d15f31b1d76fadcc1a88a5.jpg)

![ddc2e73f4c58df443a5e6baec57661ef3be9ef67d663c12f0de8a5116ef0ca88.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/ddc2e73f4c58df443a5e6baec57661ef3be9ef67d663c12f0de8a5116ef0ca88.jpg)

![ed533299df1fad008aa2f770ab72c4ea5be97362e3c72cbf8ed2a04ca2027874.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/ed533299df1fad008aa2f770ab72c4ea5be97362e3c72cbf8ed2a04ca2027874.jpg)

![f661617ff3d73edd002d5d8996b2efc63b1125dede8de813c4032d213b1c4f51.jpg](../icml_results/5470_Batched%20Contextual%20Reinforcement/tables/f661617ff3d73edd002d5d8996b2efc63b1125dede8de813c4032d213b1c4f51.jpg)

<a id="paper-5423"></a>

## Improving the Sensitivity of Backdoor Detectors via Class Subspace Orthogonalization


### Images

![2e5221903610bd11d7a75ba624c5217cdd883eee1339f7087d2f5fd66091f5ca.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/images/2e5221903610bd11d7a75ba624c5217cdd883eee1339f7087d2f5fd66091f5ca.jpg)

![33613945813f9a75ed1258ef33ffbd1b3f93ed7e2e01eb2c71266f51fe2c1081.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/images/33613945813f9a75ed1258ef33ffbd1b3f93ed7e2e01eb2c71266f51fe2c1081.jpg)

![396aa350d7ef73159ce75d01d9779f0ce05d5ccf423c8d0879ee9835d40ba088.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/images/396aa350d7ef73159ce75d01d9779f0ce05d5ccf423c8d0879ee9835d40ba088.jpg)

![4fecec7e562c62023d3f14781991ffe345d8a6a7b8c058e929405c3f70118056.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/images/4fecec7e562c62023d3f14781991ffe345d8a6a7b8c058e929405c3f70118056.jpg)

### Tables

![06f06335d986bab9b25748bf09d66dfdb612d8bfaea89e97a7d65cbc541ae6f9.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/06f06335d986bab9b25748bf09d66dfdb612d8bfaea89e97a7d65cbc541ae6f9.jpg)

![0b2283358f91e3ce2cce43523267bd963002724405c28579c6a6d5f4cd352b06.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/0b2283358f91e3ce2cce43523267bd963002724405c28579c6a6d5f4cd352b06.jpg)

![23fd164fbb9b503afab7d82a50cbbd99e02813a2851fd06190699faf64f04fe8.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/23fd164fbb9b503afab7d82a50cbbd99e02813a2851fd06190699faf64f04fe8.jpg)

![27683513c95a4e7056f089f236f211a334e9fa6e8f85bb77d927cb2db201d650.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/27683513c95a4e7056f089f236f211a334e9fa6e8f85bb77d927cb2db201d650.jpg)

![2ff884556136164e461c7e30026f82c3a2125a9b34849e9bc159be20b0b81a40.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/2ff884556136164e461c7e30026f82c3a2125a9b34849e9bc159be20b0b81a40.jpg)

![358831c8c4f69e9998ca4af2a11dcbbfbb6de14877243c5a6f824966d7d4ffeb.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/358831c8c4f69e9998ca4af2a11dcbbfbb6de14877243c5a6f824966d7d4ffeb.jpg)

![3b5571356aa086c5e0978536b893ae8d28eaef6702776ccb66038592f1798e5c.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/3b5571356aa086c5e0978536b893ae8d28eaef6702776ccb66038592f1798e5c.jpg)

![3bc7b22999fbcee49cc0b62a269bb6b97d0b8936f457d0df3f6f53ca0a00be44.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/3bc7b22999fbcee49cc0b62a269bb6b97d0b8936f457d0df3f6f53ca0a00be44.jpg)

![4c552a87a44a1aa1958bbf631c90693cd454d06aa14131c84fd3366f80f89920.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/4c552a87a44a1aa1958bbf631c90693cd454d06aa14131c84fd3366f80f89920.jpg)

![4fe1598c64b148f2a14ada9ea8f9710e80176943cf0ae86bf02a4fe20c627c1c.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/4fe1598c64b148f2a14ada9ea8f9710e80176943cf0ae86bf02a4fe20c627c1c.jpg)

![59d2bce282de280fe2c27b0cbfee020b8f74005f34abeba19481497134209a7f.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/59d2bce282de280fe2c27b0cbfee020b8f74005f34abeba19481497134209a7f.jpg)

![5bdb3e0aaed2a8a1d6c31ed09ec4ad7789c7ed815e8fbb02afcc5a12bf9a1fb5.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/5bdb3e0aaed2a8a1d6c31ed09ec4ad7789c7ed815e8fbb02afcc5a12bf9a1fb5.jpg)

![5f9c4e9534da8e182eb52cb0f755b6fcccf866418e86db52aa38e58352a6a51e.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/5f9c4e9534da8e182eb52cb0f755b6fcccf866418e86db52aa38e58352a6a51e.jpg)

![754611c171d03826e78390489ec862e315a130596320b00a3d68836cf1294e50.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/754611c171d03826e78390489ec862e315a130596320b00a3d68836cf1294e50.jpg)

![75bcd975e17d99981df4db6900f17e94c7aa4b136ff18a408732b225ce4fe00b.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/75bcd975e17d99981df4db6900f17e94c7aa4b136ff18a408732b225ce4fe00b.jpg)

![7694229dbab91c54fe2d9562327a8e7df997f49ea440f02aef8140ba4508373b.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/7694229dbab91c54fe2d9562327a8e7df997f49ea440f02aef8140ba4508373b.jpg)

![83a91da4700e3b04f631b8e300ba2a1b50c1bc6618209c322a1cb82c50413f37.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/83a91da4700e3b04f631b8e300ba2a1b50c1bc6618209c322a1cb82c50413f37.jpg)

![9db4d3e30d2da59c0591fc7977878df26c26c544506179635cdc2d9285ee2f8e.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/9db4d3e30d2da59c0591fc7977878df26c26c544506179635cdc2d9285ee2f8e.jpg)

![a492d82d214d7259368a8a8ab15ace4c1c7e3a8a4085ada6cffc9e57257e392a.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/a492d82d214d7259368a8a8ab15ace4c1c7e3a8a4085ada6cffc9e57257e392a.jpg)

![a9c858cd75327feb22716ba86b63d51bd6eceb380b1864d12f3bc0b6ba0114cd.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/a9c858cd75327feb22716ba86b63d51bd6eceb380b1864d12f3bc0b6ba0114cd.jpg)

![abb7ddfd021fdf935716d9e8c41889974dcd1be1c53a6196b3046f83b747c6b9.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/abb7ddfd021fdf935716d9e8c41889974dcd1be1c53a6196b3046f83b747c6b9.jpg)

![ad1ed455877839a69f5dc534fc6530388aa62da483648b22e8a3a2d0204aa02b.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/ad1ed455877839a69f5dc534fc6530388aa62da483648b22e8a3a2d0204aa02b.jpg)

![bc60a896982c33582cc685ae92e97e962c82859e5196904645031161de9bd86d.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/bc60a896982c33582cc685ae92e97e962c82859e5196904645031161de9bd86d.jpg)

![bf1e21a783d96d8852be43fb0b269784ce5574826abf54f27e0c05f4681fa448.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/bf1e21a783d96d8852be43fb0b269784ce5574826abf54f27e0c05f4681fa448.jpg)

![c2e52394a956e070509c613796ed8ef65f67a45fad6fcfe67b7da872f00e6c36.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/c2e52394a956e070509c613796ed8ef65f67a45fad6fcfe67b7da872f00e6c36.jpg)

![e38187078ab963fffe0d49692194735dd643a109e55aebd0030b7801e6f74c02.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/e38187078ab963fffe0d49692194735dd643a109e55aebd0030b7801e6f74c02.jpg)

![e3e58bc49505957ef2fda4cfae146109c2e62766e33f23ef1919305e588bd015.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/e3e58bc49505957ef2fda4cfae146109c2e62766e33f23ef1919305e588bd015.jpg)

![e5849ed49853830e6bfc53c8c65bf5a74782656815fea31f369ae39752221fd5.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/e5849ed49853830e6bfc53c8c65bf5a74782656815fea31f369ae39752221fd5.jpg)

![f59c0fc4276bf04e62ea9d51b75cf373cb62f16a80b26189689cf5cb3d04c41d.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/f59c0fc4276bf04e62ea9d51b75cf373cb62f16a80b26189689cf5cb3d04c41d.jpg)

![f60875d88e0e03a9611e254b764bbc0666cb149dceb2de4e55eace9904c3ad0e.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/f60875d88e0e03a9611e254b764bbc0666cb149dceb2de4e55eace9904c3ad0e.jpg)

![f6391402a70ee291af76eb106efd07138386d1e3b7ad50ae457f84113ba173e5.jpg](../icml_results/5471_Improving%20the%20Sensitivity%20of%20Backdoor%20Detectors%20via%20Class%20Subspace%20Orthogonalization/tables/f6391402a70ee291af76eb106efd07138386d1e3b7ad50ae457f84113ba173e5.jpg)

<a id="paper-5424"></a>

## Semantic Robustness Certification for Vision-Language Models


### Images

![3b6750f171577761a9db77523459742b3f19540da78d48377da85af3f1435b8d.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/3b6750f171577761a9db77523459742b3f19540da78d48377da85af3f1435b8d.jpg)

![55a65e0db1894262b1fcbef4483e715d2eb50b6e61db320e1b50e950ac488766.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/55a65e0db1894262b1fcbef4483e715d2eb50b6e61db320e1b50e950ac488766.jpg)

![64ab62a258d6ad182dc4077873b7c35f1ecd6faf25903cf60320076b15e71612.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/64ab62a258d6ad182dc4077873b7c35f1ecd6faf25903cf60320076b15e71612.jpg)

![88b3e6ceb24cc67a3d7900855536c3f6685a83c082faf5e0e19b28d7ba3dc42c.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/88b3e6ceb24cc67a3d7900855536c3f6685a83c082faf5e0e19b28d7ba3dc42c.jpg)

![ad2987cd0d1f8e8af728b3845b195c2a8db1ff4834a95e0b8080e9a2e2eae881.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/ad2987cd0d1f8e8af728b3845b195c2a8db1ff4834a95e0b8080e9a2e2eae881.jpg)

![b87108ee7cebb8f1a982a64c159bec85604f790a3bc3a278ad07131d1b433b9c.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/b87108ee7cebb8f1a982a64c159bec85604f790a3bc3a278ad07131d1b433b9c.jpg)

![bd20b1ab45d272f2beabda55464a62d060eaa5896fd92ffdc8410e5e890041d0.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/bd20b1ab45d272f2beabda55464a62d060eaa5896fd92ffdc8410e5e890041d0.jpg)

![df3cb975672d201b90df6b3ec58b74ee11d8fd71b3f3ce3baafd04e1430fdc05.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/df3cb975672d201b90df6b3ec58b74ee11d8fd71b3f3ce3baafd04e1430fdc05.jpg)

![ec8a9576560f197e8190955f12f9e681e5f364896e06673ca5b65870d168f197.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/ec8a9576560f197e8190955f12f9e681e5f364896e06673ca5b65870d168f197.jpg)

![f5776152792b83a1b078afe068b90259b191220f3fbc0db3942dc247527e0525.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/f5776152792b83a1b078afe068b90259b191220f3fbc0db3942dc247527e0525.jpg)

![f7a99df46e8584a817c25ff6a6350bc49c8e1c6b97862980e6a8a600040d782d.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/images/f7a99df46e8584a817c25ff6a6350bc49c8e1c6b97862980e6a8a600040d782d.jpg)

### Tables

![1a139a21ebc15994130986cd72c69a96d306688907538585e0e6a23eb55ae590.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/tables/1a139a21ebc15994130986cd72c69a96d306688907538585e0e6a23eb55ae590.jpg)

![2b06a65f94e30a944e412b71bc610d1b7f7fa8ba9be553a6eedd231efe391988.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/tables/2b06a65f94e30a944e412b71bc610d1b7f7fa8ba9be553a6eedd231efe391988.jpg)

![5d8f0cc33d0f9bbd985df584c59d471e10151d55dbfe2b51c2edbcac7edde81c.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/tables/5d8f0cc33d0f9bbd985df584c59d471e10151d55dbfe2b51c2edbcac7edde81c.jpg)

![7e9bc69cb95ced181e1021acdeba99b34565efff50918a244b189d46f4eb02d5.jpg](../icml_results/5472_Semantic%20Robustness%20Certification%20for%20Vision-Language%20Models/tables/7e9bc69cb95ced181e1021acdeba99b34565efff50918a244b189d46f4eb02d5.jpg)

<a id="paper-5425"></a>

## ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution


### Images

![0e8679e74a9884a175d3e4621948840b5989ebab1e179005aaf0615efa139fcc.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/0e8679e74a9884a175d3e4621948840b5989ebab1e179005aaf0615efa139fcc.jpg)

![0feed738a31530439709163fce7356f66cc364ca780fb77eb9ebc75663b34ff6.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/0feed738a31530439709163fce7356f66cc364ca780fb77eb9ebc75663b34ff6.jpg)

![27b2e1891395fcb6bb8f047d104ffc57cac864ed02da180caeb133775d593a59.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/27b2e1891395fcb6bb8f047d104ffc57cac864ed02da180caeb133775d593a59.jpg)

![2ea460bcbf30283cb1b94b48fe3b1f8eb9dfab1e9b04680492058aae40a1bb16.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/2ea460bcbf30283cb1b94b48fe3b1f8eb9dfab1e9b04680492058aae40a1bb16.jpg)

![3d082e11ef2f68373c202f802f102278a0438bd2d6b3700b2c7ca7902a612127.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/3d082e11ef2f68373c202f802f102278a0438bd2d6b3700b2c7ca7902a612127.jpg)

![49d4dc6ea2b0dc99289b5f1fb7ee353aa0807b180f025eeecc2949735ad30c7b.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/49d4dc6ea2b0dc99289b5f1fb7ee353aa0807b180f025eeecc2949735ad30c7b.jpg)

![51dbb1282522253bf15253ed6be2f6ffb8c42851ae3a0e9a4d91e08130c8c0c1.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/51dbb1282522253bf15253ed6be2f6ffb8c42851ae3a0e9a4d91e08130c8c0c1.jpg)

![661791e025b0d8fa0e96d19cc6ba07aee6d62b195380d51864da01f4e3baefb1.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/661791e025b0d8fa0e96d19cc6ba07aee6d62b195380d51864da01f4e3baefb1.jpg)

![6a5916b77255c2ef0b81f2e135c13c5e4727c90bbcdee425a42d276ec3d4ad87.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/6a5916b77255c2ef0b81f2e135c13c5e4727c90bbcdee425a42d276ec3d4ad87.jpg)

![6b1bd91059c0c6f8202812078f92bf977ef9622b1e8529125d6e4b751443d461.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/6b1bd91059c0c6f8202812078f92bf977ef9622b1e8529125d6e4b751443d461.jpg)

![722601d4f4c748091db6e09f26788c792d347d5e32b2b768647e8f16a9310c9b.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/722601d4f4c748091db6e09f26788c792d347d5e32b2b768647e8f16a9310c9b.jpg)

![7e251d4523383a5830aa6af297f890baba6f36da7c9763a135edc14357880b95.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/7e251d4523383a5830aa6af297f890baba6f36da7c9763a135edc14357880b95.jpg)

![840e5ebeda481b10dced5a710d33b8f4904877fb540b14d191910d9a282219aa.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/840e5ebeda481b10dced5a710d33b8f4904877fb540b14d191910d9a282219aa.jpg)

![84848e9c5b92b79289c184f0d42aee9c34e0428bef120bfcd022540fb6fc98b5.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/84848e9c5b92b79289c184f0d42aee9c34e0428bef120bfcd022540fb6fc98b5.jpg)

![8faf440ba318f8d4446f8398d9b06b5eaaad51e1be0531683e634e029a74c12f.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/8faf440ba318f8d4446f8398d9b06b5eaaad51e1be0531683e634e029a74c12f.jpg)

![91dc829c4da7bde911fd93ba52b1c81da6b523ad426bc83a068cf5d60a692630.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/91dc829c4da7bde911fd93ba52b1c81da6b523ad426bc83a068cf5d60a692630.jpg)

![98937b14bffef57af400ad197bbe0d4e9431faad544c04dc76fb601f10eb4ebd.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/98937b14bffef57af400ad197bbe0d4e9431faad544c04dc76fb601f10eb4ebd.jpg)

![99f44771abe76c326b862583a1f71221f18bf1b966b8f54aa667c52b01001ce8.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/99f44771abe76c326b862583a1f71221f18bf1b966b8f54aa667c52b01001ce8.jpg)

![a0e01341b2b6fe1607b3adfacab83a2806368e265fab48be8764208466567884.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/a0e01341b2b6fe1607b3adfacab83a2806368e265fab48be8764208466567884.jpg)

![b39ce4964fa2af67d4b37b0a8d47f6a8583e693af43670d31857562903275746.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/b39ce4964fa2af67d4b37b0a8d47f6a8583e693af43670d31857562903275746.jpg)

![c2447a5a25a4e9da5c8cc664e848ffc60f4e908c79ec95860617f1e636d2ea16.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/c2447a5a25a4e9da5c8cc664e848ffc60f4e908c79ec95860617f1e636d2ea16.jpg)

![df7308d98a8d6ea64f8697e79f13ae0a101e8db0bf6e7be015a1853f04e926e2.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/df7308d98a8d6ea64f8697e79f13ae0a101e8db0bf6e7be015a1853f04e926e2.jpg)

![e522d4efc13c3f8a93fefc6c2f61aa0b3c77281c87e9355e1fb4912d606d49b6.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/e522d4efc13c3f8a93fefc6c2f61aa0b3c77281c87e9355e1fb4912d606d49b6.jpg)

![f0ec370d900c9cddfb974073e18de08fec87fdd776826b287e3c9f99148979f4.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/f0ec370d900c9cddfb974073e18de08fec87fdd776826b287e3c9f99148979f4.jpg)

![f65dc04be0fabb35693f0a95018107038edfee9330fe4442c581678c3f50fb1a.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/images/f65dc04be0fabb35693f0a95018107038edfee9330fe4442c581678c3f50fb1a.jpg)

### Tables

![0c60d920b35589d1de5d30975998c831dcfaa92a47fe0141b220345baf8c682b.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/0c60d920b35589d1de5d30975998c831dcfaa92a47fe0141b220345baf8c682b.jpg)

![222e6881b87e7c71c2ba67d6a0f38b13e66a34599c4d494b2f81d108d7d46e1c.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/222e6881b87e7c71c2ba67d6a0f38b13e66a34599c4d494b2f81d108d7d46e1c.jpg)

![52f6fcec1497cfbd9f8e64c2cef42bc34a1f5beec34a6ffa624fd5042c34896d.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/52f6fcec1497cfbd9f8e64c2cef42bc34a1f5beec34a6ffa624fd5042c34896d.jpg)

![809bdc65489b3551cf0a4206adedfd9bf8e201f90f5ebf7b9e1fa31b04afcdad.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/809bdc65489b3551cf0a4206adedfd9bf8e201f90f5ebf7b9e1fa31b04afcdad.jpg)

![93baea9c159bec3af17023c711ff37641591564e15b461c719fe0c09538237ba.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/93baea9c159bec3af17023c711ff37641591564e15b461c719fe0c09538237ba.jpg)

![a6e973c01f5ab6eccdb3c4659f2bbff6f5d928d2e88131a89a47946cfc5f28ff.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/a6e973c01f5ab6eccdb3c4659f2bbff6f5d928d2e88131a89a47946cfc5f28ff.jpg)

![abd87f77295772c42cdcac6f15274051e7f41425d239681c8e936859d71c698d.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/abd87f77295772c42cdcac6f15274051e7f41425d239681c8e936859d71c698d.jpg)

![d4b1dbc2bf2ca8e2a970bf66f91d9512860d09faff9b1b40404cf1f385a46246.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/d4b1dbc2bf2ca8e2a970bf66f91d9512860d09faff9b1b40404cf1f385a46246.jpg)

![f8a5986e794508a0577a1d1e61fed9d3ae97d863722774ce1649e427b63e3030.jpg](../icml_results/5473_ParEVO_%20Synthesizing%20Code%20for%20Irregular%20Data_%20High-Performance%20Parallelism%20through%20Agentic%20Evolution/tables/f8a5986e794508a0577a1d1e61fed9d3ae97d863722774ce1649e427b63e3030.jpg)

<a id="paper-5426"></a>

## One-Shot Weighted Ensemble Estimation for Federated Quantile Regression: Optimal Statistical Guarantees under Heterogeneous Structured Data


### Images

![033e8326d33572f8f70c598171c428e8959fc54bb61c69a412f37f52675aab2e.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/033e8326d33572f8f70c598171c428e8959fc54bb61c69a412f37f52675aab2e.jpg)

![10af4f615073dfd235d8777cf388363061963b323ae391338678f01449f61a8b.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/10af4f615073dfd235d8777cf388363061963b323ae391338678f01449f61a8b.jpg)

![2a15ad87ceeb45c5383aad6835fa51d8ec52db3e299c9248034b42f9a31ec866.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/2a15ad87ceeb45c5383aad6835fa51d8ec52db3e299c9248034b42f9a31ec866.jpg)

![4960e4ff210618874d72b3cfacb76307adb7beb5a54fb7203124c53d233f7e61.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/4960e4ff210618874d72b3cfacb76307adb7beb5a54fb7203124c53d233f7e61.jpg)

![54c967bea2ba855f332d0252cbad31e6e9115fc7a4a65bdd15af0c2ac4f89cc5.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/54c967bea2ba855f332d0252cbad31e6e9115fc7a4a65bdd15af0c2ac4f89cc5.jpg)

![8b2a709b5e9d4ec4cb6259bbd40972378c5b9c98cc3f69e44cc463ea57e32e73.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/8b2a709b5e9d4ec4cb6259bbd40972378c5b9c98cc3f69e44cc463ea57e32e73.jpg)

![9027d8788da2625f82d35e5bdd88e7af2cb1c5f25cd7222cfe7f882655d5c9a0.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/9027d8788da2625f82d35e5bdd88e7af2cb1c5f25cd7222cfe7f882655d5c9a0.jpg)

![975c2f288b0496bafae9fbc69e2936dc306b0eb52d2a0a6284b23932f12184d7.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/975c2f288b0496bafae9fbc69e2936dc306b0eb52d2a0a6284b23932f12184d7.jpg)

![df8bd8fc7bc3995f9e1aaa56e8717e956a3cafae85502a848dc0c0da9c8fbb1d.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/df8bd8fc7bc3995f9e1aaa56e8717e956a3cafae85502a848dc0c0da9c8fbb1d.jpg)

![eb1f354403caccba9680a5cb4eaad16f5829e331b43c1d23e7299d659570c567.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/eb1f354403caccba9680a5cb4eaad16f5829e331b43c1d23e7299d659570c567.jpg)

![f5049f61e985f1202424133509c378029a7b1b0278c1a968a33adcf8f71c7492.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/images/f5049f61e985f1202424133509c378029a7b1b0278c1a968a33adcf8f71c7492.jpg)

### Tables

![0d60be1e19359b5db8503062629e45f036467b047d8543c034e99ca1ff1600d7.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/0d60be1e19359b5db8503062629e45f036467b047d8543c034e99ca1ff1600d7.jpg)

![60dfe59f87743837df862a9dbc194c4f1e72c7a694151e526cfb5e2d5f683e24.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/60dfe59f87743837df862a9dbc194c4f1e72c7a694151e526cfb5e2d5f683e24.jpg)

![689bc63081835f1f124f563d1fb23cc9fd116ef3cccb6f36fc0a20812741f1ee.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/689bc63081835f1f124f563d1fb23cc9fd116ef3cccb6f36fc0a20812741f1ee.jpg)

![69ef2e9be78061dd8f22f203831188bea0330e231f866a8b585638a318aad837.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/69ef2e9be78061dd8f22f203831188bea0330e231f866a8b585638a318aad837.jpg)

![8052f4aacaa515fcf9d4f2f8a7bbd4f69bb73c6d1d5812657319512be0a5de95.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/8052f4aacaa515fcf9d4f2f8a7bbd4f69bb73c6d1d5812657319512be0a5de95.jpg)

![8e0e5a11e72cd1b17b370fed9048aacff89a7410f363623e6b939ddc62ebccc7.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/8e0e5a11e72cd1b17b370fed9048aacff89a7410f363623e6b939ddc62ebccc7.jpg)

![9680ee94560a23301324723d156f58c286c664cd175abd6266f2657cc4d5c245.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/9680ee94560a23301324723d156f58c286c664cd175abd6266f2657cc4d5c245.jpg)

![ad34f871676944d1838c83e8f53a9f29a5500f5a7423fc07ab2dae5a62bc868c.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/ad34f871676944d1838c83e8f53a9f29a5500f5a7423fc07ab2dae5a62bc868c.jpg)

![f580f4a79f5ac96e6c385d55a185130be6518414c1a9e5fe466ee7700a30dd14.jpg](../icml_results/5474_One-Shot%20Weighted%20Ensemble%20Estimation%20for%20Federated%20Quantile%20Regression_%20Optimal%20Statistical%20Guarant/tables/f580f4a79f5ac96e6c385d55a185130be6518414c1a9e5fe466ee7700a30dd14.jpg)

<a id="paper-5427"></a>

## Toward Safe Quantization-Aware Fine-tuning: Understanding and Mitigating Safety Alignment Degradation


### Images

![0e28756b4a8c91accc4903c83b3d580c6c71e82d765ca9c2aa027be90deb49e2.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/0e28756b4a8c91accc4903c83b3d580c6c71e82d765ca9c2aa027be90deb49e2.jpg)

![2743f74b0cb09559935aa73a09cade5d911b2ff333b399e0c1e83fd182a4e85c.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/2743f74b0cb09559935aa73a09cade5d911b2ff333b399e0c1e83fd182a4e85c.jpg)

![2aa10a8270975f4002350c550f8293ca55fa73aad2f8d299f195fc8749291329.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/2aa10a8270975f4002350c550f8293ca55fa73aad2f8d299f195fc8749291329.jpg)

![68ef5cd84da10166ddbbb5cd83e320dfd295f391b945a7f93eaf7caaf6dca2e0.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/68ef5cd84da10166ddbbb5cd83e320dfd295f391b945a7f93eaf7caaf6dca2e0.jpg)

![70c12dd555f93f51b3beb8d09f3c585aaf2d7bfd0aa477df207863ed95ea252c.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/70c12dd555f93f51b3beb8d09f3c585aaf2d7bfd0aa477df207863ed95ea252c.jpg)

![8cca11cdffddb39775fd8b3755350a0049c18c2f4a4a48f1b50a5d812436d6bd.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/8cca11cdffddb39775fd8b3755350a0049c18c2f4a4a48f1b50a5d812436d6bd.jpg)

![beda43d7f98a786b8e1e12ad6e0db107137499460bbe49344b1b01e201973dac.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/beda43d7f98a786b8e1e12ad6e0db107137499460bbe49344b1b01e201973dac.jpg)

![f16dd757e4da8dd3f7ac6b0367d01ba0a49e77b8554c7ad83d69163a3493fb3d.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/images/f16dd757e4da8dd3f7ac6b0367d01ba0a49e77b8554c7ad83d69163a3493fb3d.jpg)

### Tables

![04fba77273eaf79f886b41ced83a98bd88567959ee6739fa8a8a9c469c9d1c44.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/04fba77273eaf79f886b41ced83a98bd88567959ee6739fa8a8a9c469c9d1c44.jpg)

![0e46a132e22f93f4c37fd55447f59fcbab9116fbdc9d20a361fc23a7e28fe09c.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/0e46a132e22f93f4c37fd55447f59fcbab9116fbdc9d20a361fc23a7e28fe09c.jpg)

![1da1acd85d8dd0aa82c76a68a0e470c71510b3e76ea609b3a251374d96fb1c16.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/1da1acd85d8dd0aa82c76a68a0e470c71510b3e76ea609b3a251374d96fb1c16.jpg)

![3f49537006ab2914951d77872ccca232e8c5f7afee65d064c1ab09f466e6549d.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/3f49537006ab2914951d77872ccca232e8c5f7afee65d064c1ab09f466e6549d.jpg)

![77698fdeac7cfe4f6665c06eb7a87446853127a446e4793c57014d936e066a7a.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/77698fdeac7cfe4f6665c06eb7a87446853127a446e4793c57014d936e066a7a.jpg)

![b68b0f9f067a415a1a85c75c29455e00ceb7a71aee63a62e0d43eb0ee8074701.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/b68b0f9f067a415a1a85c75c29455e00ceb7a71aee63a62e0d43eb0ee8074701.jpg)

![d9ed1dc5e39d7615d363bf55097bc0ab0ce86d569d6e8256a0ccec932225d18c.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/d9ed1dc5e39d7615d363bf55097bc0ab0ce86d569d6e8256a0ccec932225d18c.jpg)

![db583df47db618369f7fb419979ad9eafdc0a6394ee9911d654d37c49784be2f.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/db583df47db618369f7fb419979ad9eafdc0a6394ee9911d654d37c49784be2f.jpg)

![f710ae3f95302b11c428daa6c9419bf5b273201936791527490b742a3e5996de.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/f710ae3f95302b11c428daa6c9419bf5b273201936791527490b742a3e5996de.jpg)

![f7ae8caf9407fa04b71293bd12db3d9f3a5edd31358501a12e467c5ce7f23aa9.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/f7ae8caf9407fa04b71293bd12db3d9f3a5edd31358501a12e467c5ce7f23aa9.jpg)

![f87042521660ac047f792655ef87bd50f0a66c5d4cfcd2024e2baec5cad3f9a5.jpg](../icml_results/5475_Toward%20Safe%20Quantization-Aware%20Fine-tuning_%20Understanding%20and%20Mitigating%20Safety%20Alignment%20Degradatio/tables/f87042521660ac047f792655ef87bd50f0a66c5d4cfcd2024e2baec5cad3f9a5.jpg)

<a id="paper-5428"></a>

## Wasserstein Geometry-Aware Adaptive Control via Meta-Learning


### Images

![05ce13c283800a3e9a4528ddda23e79f33cb08c99953de32c89d7e9b4ff74e47.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/05ce13c283800a3e9a4528ddda23e79f33cb08c99953de32c89d7e9b4ff74e47.jpg)

![085b88e884cf32d6f05bec26fc85f160f274065a2e0fee2bf24c28167a6a0474.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/085b88e884cf32d6f05bec26fc85f160f274065a2e0fee2bf24c28167a6a0474.jpg)

![1a56aab736516abebf9e11cb5b619d9537ce6a3db5343d293411d404680cf2bb.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/1a56aab736516abebf9e11cb5b619d9537ce6a3db5343d293411d404680cf2bb.jpg)

![246ae66add9cdd4fcf6d2e18b01842e86f22d11c0d1ddc1794ae1aafb6d6ed3a.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/246ae66add9cdd4fcf6d2e18b01842e86f22d11c0d1ddc1794ae1aafb6d6ed3a.jpg)

![2c9531482a9e1ed94948e9055f82f00b01ca05b81e74f5e0c95797541d9642d9.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/2c9531482a9e1ed94948e9055f82f00b01ca05b81e74f5e0c95797541d9642d9.jpg)

![45e925bb58458aac598b58f5520e4f7d0a75d1667a40153ee8acbc5d617ccb53.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/45e925bb58458aac598b58f5520e4f7d0a75d1667a40153ee8acbc5d617ccb53.jpg)

![5e0c4d927764eea1d6734fce64a42faca3b0216ef0b617f5a922bc3d6d462c4b.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/5e0c4d927764eea1d6734fce64a42faca3b0216ef0b617f5a922bc3d6d462c4b.jpg)

![73b1adcc88ea6ed417d51328bf64f2e6bc278aaa3b093fd67fe9fb2374687399.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/73b1adcc88ea6ed417d51328bf64f2e6bc278aaa3b093fd67fe9fb2374687399.jpg)

![7a817bb480c7e2a81b9809ddc4662dbcb91cc6a145b4e1a26ed17f9dffb8326e.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/7a817bb480c7e2a81b9809ddc4662dbcb91cc6a145b4e1a26ed17f9dffb8326e.jpg)

![81cac5dd67940dc430edc95b13418aebf0bfe5994a15ad3b648295747a6e0b76.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/81cac5dd67940dc430edc95b13418aebf0bfe5994a15ad3b648295747a6e0b76.jpg)

![966948906bbffd8ea1cdd80a61aaa3d418236784e8cbb7209888a82b58374d49.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/966948906bbffd8ea1cdd80a61aaa3d418236784e8cbb7209888a82b58374d49.jpg)

![97264ec05ceba4bfbfc604e438feb4f7179aac60cf07a8f3730d356d4d1c305c.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/97264ec05ceba4bfbfc604e438feb4f7179aac60cf07a8f3730d356d4d1c305c.jpg)

![bc043946aeedf14446b79b286b3fb5ab1d17a38834f6df71e96fa212ef30faa2.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/images/bc043946aeedf14446b79b286b3fb5ab1d17a38834f6df71e96fa212ef30faa2.jpg)

### Tables

![f3902e9b332d3c842798c61ae917a353c75a23840e2b8f69794ee3302cbe2f73.jpg](../icml_results/5476_Wasserstein%20Geometry-Aware%20Adaptive%20Control%20via%20Meta-Learning/tables/f3902e9b332d3c842798c61ae917a353c75a23840e2b8f69794ee3302cbe2f73.jpg)

<a id="paper-5429"></a>

## Forward-Chaining Temporal Point Process


### Images

![0c8a00b4adfbbabc38dc6a3531ff5f002a645bdff4234cb2f1edf1edc77799f5.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/0c8a00b4adfbbabc38dc6a3531ff5f002a645bdff4234cb2f1edf1edc77799f5.jpg)

![0eb8e3339ce2a1cc414eaa2f8cea3a87336e0529514ddbf6799799a5b2e39643.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/0eb8e3339ce2a1cc414eaa2f8cea3a87336e0529514ddbf6799799a5b2e39643.jpg)

![1b37f7e617be3c4c2b30d7e17fea5d5c3847aa24bf7814209863c2e6364d9b59.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/1b37f7e617be3c4c2b30d7e17fea5d5c3847aa24bf7814209863c2e6364d9b59.jpg)

![3aa6cbab9cead9873afe1d5f69b6b18f459185b51bc54c639de2609d4ff96ea7.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/3aa6cbab9cead9873afe1d5f69b6b18f459185b51bc54c639de2609d4ff96ea7.jpg)

![49c2666cd2a9c40144a2277120a6016c8d80b6b26606a83085422d3019cdb3ff.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/49c2666cd2a9c40144a2277120a6016c8d80b6b26606a83085422d3019cdb3ff.jpg)

![5ae81aefb2993bbb24b6a5dc7274709e619713dbf5da7197dc72bc8582b4c304.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/5ae81aefb2993bbb24b6a5dc7274709e619713dbf5da7197dc72bc8582b4c304.jpg)

![7286874f1ddedcaccf9009f9ea826f9d832bde89524ba7c7ab09f41b68cc20a9.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/7286874f1ddedcaccf9009f9ea826f9d832bde89524ba7c7ab09f41b68cc20a9.jpg)

![7a0e32523cb05fcd304171a8cb640dc62e9c42f44b33387bc8c04f80eb362dbc.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/7a0e32523cb05fcd304171a8cb640dc62e9c42f44b33387bc8c04f80eb362dbc.jpg)

![e395036cf538838707039ce18852fd0fa61da57a77c35247e9fcf03e45162f7e.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/images/e395036cf538838707039ce18852fd0fa61da57a77c35247e9fcf03e45162f7e.jpg)

### Tables

![0059626211665e56630780ed653f3269dee3601efdfde7cc4e5d00cc4e93918d.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/0059626211665e56630780ed653f3269dee3601efdfde7cc4e5d00cc4e93918d.jpg)

![1232defcbfee1c856d05ee45db645da75168aa96f33bd98267c399804f65156b.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/1232defcbfee1c856d05ee45db645da75168aa96f33bd98267c399804f65156b.jpg)

![1527e3c0265d1c3f7c331d0d7efe60b7648d657f98ef637b072d284b6100f514.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/1527e3c0265d1c3f7c331d0d7efe60b7648d657f98ef637b072d284b6100f514.jpg)

![29f82ab669565562e6f048e28c4ca355e62de59987426e9a378c9157b269021c.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/29f82ab669565562e6f048e28c4ca355e62de59987426e9a378c9157b269021c.jpg)

![377cc620216a05bd128a1c133a394d79c3f6bf7261d47f0c7230ea205895fd74.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/377cc620216a05bd128a1c133a394d79c3f6bf7261d47f0c7230ea205895fd74.jpg)

![42f5ebdd130a020f4557319556707e71ef9bf72552371c1d95a22d2de3b2f41a.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/42f5ebdd130a020f4557319556707e71ef9bf72552371c1d95a22d2de3b2f41a.jpg)

![453675074cd1e55807d9909549e814c817ffd99f7eab3041abc0cd0fd0abceea.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/453675074cd1e55807d9909549e814c817ffd99f7eab3041abc0cd0fd0abceea.jpg)

![510e49f4a6894f451ab952c1f8b1f24572cc509bbf800ebf11ef3008722fe579.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/510e49f4a6894f451ab952c1f8b1f24572cc509bbf800ebf11ef3008722fe579.jpg)

![7d6519ed649a548005889e6460b9606900fdf9c8d81a9a1709814dd3b5b0a668.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/7d6519ed649a548005889e6460b9606900fdf9c8d81a9a1709814dd3b5b0a668.jpg)

![9038de00a5a7ac86b334378efe9cdb2d343db8b60797e6fde8a181984aaf6b5b.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/9038de00a5a7ac86b334378efe9cdb2d343db8b60797e6fde8a181984aaf6b5b.jpg)

![98702805701ce66b84ba90d979cc2830d930d3e2273d870c236a975b89c95824.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/98702805701ce66b84ba90d979cc2830d930d3e2273d870c236a975b89c95824.jpg)

![a1d30a9fd98f610f3ad6ba54fd338b641153bc180736539ff1b2de4cdae9c374.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/a1d30a9fd98f610f3ad6ba54fd338b641153bc180736539ff1b2de4cdae9c374.jpg)

![bc34a53374fe5db379d61eaf4225311d7f1ea76e3cd7edfa13d5d22f2acf814f.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/bc34a53374fe5db379d61eaf4225311d7f1ea76e3cd7edfa13d5d22f2acf814f.jpg)

![d8ee4ba2a00ed76d01d6aa001344a6ef0c838bf16125fc9839218bc44e373003.jpg](../icml_results/5477_Forward-Chaining%20Temporal%20Point%20Process/tables/d8ee4ba2a00ed76d01d6aa001344a6ef0c838bf16125fc9839218bc44e373003.jpg)

<a id="paper-5430"></a>

## HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression


### Images

![0de03ffac0b335a3acfffb6dbbc5b143b07a8f41570b3c828c6ae7cc9269beb7.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/0de03ffac0b335a3acfffb6dbbc5b143b07a8f41570b3c828c6ae7cc9269beb7.jpg)

![110caddb2b6960c4d06344d4cb4e94d33f70724369ebc6e858891ab79fdfa238.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/110caddb2b6960c4d06344d4cb4e94d33f70724369ebc6e858891ab79fdfa238.jpg)

![159ed81389c16734f74615692a4416c53c84f0472e6b0956903081995c58b87e.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/159ed81389c16734f74615692a4416c53c84f0472e6b0956903081995c58b87e.jpg)

![1d178c490456ffa886c54833d7309ce7720da8296ccceb74e8f4236d8bd8773b.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/1d178c490456ffa886c54833d7309ce7720da8296ccceb74e8f4236d8bd8773b.jpg)

![236276f3e8d89f0544df7e31981542a1b0b927d6b7e0a90b2e9bbedda3ae12b3.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/236276f3e8d89f0544df7e31981542a1b0b927d6b7e0a90b2e9bbedda3ae12b3.jpg)

![4e76944c8667ddfc9684b79e4e1531f1c194927dfa93dbee8fd12c3b57a7d794.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/4e76944c8667ddfc9684b79e4e1531f1c194927dfa93dbee8fd12c3b57a7d794.jpg)

![4ecb10e0799df97f7e9b7b63cc1f9d6b788eb890bab35634b91684e842a28c8e.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/4ecb10e0799df97f7e9b7b63cc1f9d6b788eb890bab35634b91684e842a28c8e.jpg)

![5adfbe13237d9d1b575d22ab55ca8b9b7f3682eb3ccae9960e019f9b04be2dd2.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/5adfbe13237d9d1b575d22ab55ca8b9b7f3682eb3ccae9960e019f9b04be2dd2.jpg)

![6a0029ca005cd32328de2593c9378ddc0d5e3c9cdbc74a3c663841211bf8c690.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/6a0029ca005cd32328de2593c9378ddc0d5e3c9cdbc74a3c663841211bf8c690.jpg)

![70c051774ab832304fd8efb5148bcef0f7a8a6f469005acbb1b1de41ebec2fac.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/70c051774ab832304fd8efb5148bcef0f7a8a6f469005acbb1b1de41ebec2fac.jpg)

![7a4c80a6c622f794f45f9f444df9dd5469f8fde0a364c71667cf2cd1381c984b.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/7a4c80a6c622f794f45f9f444df9dd5469f8fde0a364c71667cf2cd1381c984b.jpg)

![7da86ae8506df413df2e736abc072c054114da8bfa736a459c358e796eee67be.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/7da86ae8506df413df2e736abc072c054114da8bfa736a459c358e796eee67be.jpg)

![7dafec92e4c6cd1ee4eea6264d6c83c2502949ac472b49fe96ffd131613b6ee8.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/7dafec92e4c6cd1ee4eea6264d6c83c2502949ac472b49fe96ffd131613b6ee8.jpg)

![7fb990b86c63c8bd8dba532538198df42a89cb6aa1fc29089ac95b676f411511.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/7fb990b86c63c8bd8dba532538198df42a89cb6aa1fc29089ac95b676f411511.jpg)

![89d276565bfdba4e9c57bc7ebc5e41a3f53acabf700bb963ed4633569c9b116e.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/89d276565bfdba4e9c57bc7ebc5e41a3f53acabf700bb963ed4633569c9b116e.jpg)

![8c2a5a4b8c7d8de34d19c5c3a5518ab3580dc15b0e7a7aa23a7c3f2db302cb78.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/8c2a5a4b8c7d8de34d19c5c3a5518ab3580dc15b0e7a7aa23a7c3f2db302cb78.jpg)

![8d9ba9824d50aa3f7ae9356a8fd18fd7ff6b337c92e95200bb965ec4a0fcbb32.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/8d9ba9824d50aa3f7ae9356a8fd18fd7ff6b337c92e95200bb965ec4a0fcbb32.jpg)

![9e7c08d1af58353931c910f3d9b56f364cd54b87e8a3233444c85d5738a9edac.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/9e7c08d1af58353931c910f3d9b56f364cd54b87e8a3233444c85d5738a9edac.jpg)

![9f826483261ec81a0beff1d5d1859041b123e91bcb1e515d2b0b9bf5324b56b0.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/9f826483261ec81a0beff1d5d1859041b123e91bcb1e515d2b0b9bf5324b56b0.jpg)

![aaa21f00eb11dde677cef043062622ccc642710887d0f5577b56963bdd6be313.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/aaa21f00eb11dde677cef043062622ccc642710887d0f5577b56963bdd6be313.jpg)

![b15f379a8842c71ce9d0e4e4b3b95a524b8c412d1b5732526e8321dc4fdf68ab.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/b15f379a8842c71ce9d0e4e4b3b95a524b8c412d1b5732526e8321dc4fdf68ab.jpg)

![b75eee7d18d452c93f14f43828e53cbe337f00aea9f86160539ef0a1a174187f.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/b75eee7d18d452c93f14f43828e53cbe337f00aea9f86160539ef0a1a174187f.jpg)

![bda2ac2396d15544fff7672524230e1bdb04f4468b05266f6994c72e3824194a.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/bda2ac2396d15544fff7672524230e1bdb04f4468b05266f6994c72e3824194a.jpg)

![c323e8426cdf83e3ba4ca0157dbd673120b4fa527729a875a6c85318d24c5764.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/c323e8426cdf83e3ba4ca0157dbd673120b4fa527729a875a6c85318d24c5764.jpg)

![dcd2daaaa47f043170fd3dec9cc045c93091634b309a6c0a6d256b9a8ccdfd0d.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/dcd2daaaa47f043170fd3dec9cc045c93091634b309a6c0a6d256b9a8ccdfd0d.jpg)

![ddd658ffb384792ed03c22d3f5e108e385ca5946216d2ef7e1820494a68d9df9.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/ddd658ffb384792ed03c22d3f5e108e385ca5946216d2ef7e1820494a68d9df9.jpg)

![e872687ca6e0f528ae1613416e003f58a97ab6c5ae152945f87aa32c4dad10a6.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/images/e872687ca6e0f528ae1613416e003f58a97ab6c5ae152945f87aa32c4dad10a6.jpg)

### Tables

![18deb4c4cada6b96f74c72e2164765007426f7a208308e00228fb4226eeb7bce.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/18deb4c4cada6b96f74c72e2164765007426f7a208308e00228fb4226eeb7bce.jpg)

![247abe9fc26318ebdd851dff8dac4efb62292918f6b0752a69d79da0783b4d9f.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/247abe9fc26318ebdd851dff8dac4efb62292918f6b0752a69d79da0783b4d9f.jpg)

![30f050599976b8457ab777eb2dd2817bd1c59b9fc52f790eb489e432439d5b85.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/30f050599976b8457ab777eb2dd2817bd1c59b9fc52f790eb489e432439d5b85.jpg)

![586c685900b0c0af7cb9a551166bec89337c4b5b642fd59611fbc86b2adf6bc7.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/586c685900b0c0af7cb9a551166bec89337c4b5b642fd59611fbc86b2adf6bc7.jpg)

![810d143f4dfac69a8df38ba0406da86cc017a0da3212cef6ffd51d1ec01961c7.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/810d143f4dfac69a8df38ba0406da86cc017a0da3212cef6ffd51d1ec01961c7.jpg)

![c21733feff9c019041e391a5a106613e916df0617cffeda6314f198a47e4d09b.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/c21733feff9c019041e391a5a106613e916df0617cffeda6314f198a47e4d09b.jpg)

![e53d2a227f57e4c966818adca950104bea8bd4e91a62a10cf1006b4c1aeb19e9.jpg](../icml_results/5478_HARD-KV_%20Head-Adaptive%20Regularization%20for%20Decoding-time%20KV%20Compression/tables/e53d2a227f57e4c966818adca950104bea8bd4e91a62a10cf1006b4c1aeb19e9.jpg)

<a id="paper-5431"></a>

## Diffusion Controller: Framework, Algorithms and Parameterization


### Images

![011370e23262e7ef1c596d7061dc29e9a7ff629769565ddeee11162cad504990.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/011370e23262e7ef1c596d7061dc29e9a7ff629769565ddeee11162cad504990.jpg)

![146218e33a27ea9ee94c44eb323bcd86a6e14fc6e16475e077a7cf79d08a8186.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/146218e33a27ea9ee94c44eb323bcd86a6e14fc6e16475e077a7cf79d08a8186.jpg)

![21a9ef5637aadd97d9931c6bb882834f47413931c3bfb6885cc861e237349269.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/21a9ef5637aadd97d9931c6bb882834f47413931c3bfb6885cc861e237349269.jpg)

![35119060c1def754af0d2abc757d3ba8a4dab0606961030b926b044465cf9f3e.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/35119060c1def754af0d2abc757d3ba8a4dab0606961030b926b044465cf9f3e.jpg)

![6bc7aff3c0bb04ab782d24e24dfa7e0da6a435896ecaafc1cedb633f61cd339a.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/6bc7aff3c0bb04ab782d24e24dfa7e0da6a435896ecaafc1cedb633f61cd339a.jpg)

![7447fd09ff0b487aed6a03ad82e204381b5e6134662a09b3de34ab49e0fe09d8.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/7447fd09ff0b487aed6a03ad82e204381b5e6134662a09b3de34ab49e0fe09d8.jpg)

![7ccf70ba88de0175e24efe34cc05be06afd5bf155112cf19cf3f11f8249400c8.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/7ccf70ba88de0175e24efe34cc05be06afd5bf155112cf19cf3f11f8249400c8.jpg)

![809849f6f88c2c6848e08b491322b08803b7b7113ca4c5466be8cd8ef8931935.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/809849f6f88c2c6848e08b491322b08803b7b7113ca4c5466be8cd8ef8931935.jpg)

![8a3ba80a098e054a02e64aeed47f24e2b03d1a17270e4c0e127889cba42b9ad1.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/8a3ba80a098e054a02e64aeed47f24e2b03d1a17270e4c0e127889cba42b9ad1.jpg)

![a9135fec87b80e2b0d2da38e60337d1a909b7c963a6b60e505d37ec5047e448a.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/a9135fec87b80e2b0d2da38e60337d1a909b7c963a6b60e505d37ec5047e448a.jpg)

![a9e8ad6e1f4013e2bc6ae38dcc55a76793427725bbbacbf16982c9f6b4e95b7e.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/a9e8ad6e1f4013e2bc6ae38dcc55a76793427725bbbacbf16982c9f6b4e95b7e.jpg)

![af4af8cc964765875b85fb19ee09cd659ed03b9c9505b579884bc18e101e8c41.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/af4af8cc964765875b85fb19ee09cd659ed03b9c9505b579884bc18e101e8c41.jpg)

![c711ebd599477d99665e73a11f5f055b7c7db6eef5ad44be404dfd36366d0a7a.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/c711ebd599477d99665e73a11f5f055b7c7db6eef5ad44be404dfd36366d0a7a.jpg)

![d53cbeceb62e12dfac793c76b127a3b4c5bdaa1d1e91eb10f7b3c74785e154e9.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/d53cbeceb62e12dfac793c76b127a3b4c5bdaa1d1e91eb10f7b3c74785e154e9.jpg)

![e50736afdfdbb7749d5ee656ccf6c4c1e13ca98770395657d8fbb623058bd6a0.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/e50736afdfdbb7749d5ee656ccf6c4c1e13ca98770395657d8fbb623058bd6a0.jpg)

![f3b867798643b41044a60ca23ab06761b475202f4fea965b1549f58c81c89bdf.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/images/f3b867798643b41044a60ca23ab06761b475202f4fea965b1549f58c81c89bdf.jpg)

### Tables

![00abb172ee6b0b97f620478697130e113d4a15888970b98e49e99432c4902447.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/00abb172ee6b0b97f620478697130e113d4a15888970b98e49e99432c4902447.jpg)

![276667b96f13cbbb92e45fe5f0bb6d9b0c73453670f17e3a41f70aa2fec04a55.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/276667b96f13cbbb92e45fe5f0bb6d9b0c73453670f17e3a41f70aa2fec04a55.jpg)

![3a2a1aa9a253300593d94f2f859bbc1adb0af3209aaeb6b15aec1018a5f84150.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/3a2a1aa9a253300593d94f2f859bbc1adb0af3209aaeb6b15aec1018a5f84150.jpg)

![3ad4e54f8627f4add2d8bc90ba4948acd1fbd84efe70fc7b07abe806ccb12840.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/3ad4e54f8627f4add2d8bc90ba4948acd1fbd84efe70fc7b07abe806ccb12840.jpg)

![a07140b0926e34c6f9ec60d1ab5b4a1c15afae6d0380a6d37abeae16230c93a4.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/a07140b0926e34c6f9ec60d1ab5b4a1c15afae6d0380a6d37abeae16230c93a4.jpg)

![a966340024cdf8ff6babefc040ce6d5701b2d4f601e7358aa84308e6ae9e7999.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/a966340024cdf8ff6babefc040ce6d5701b2d4f601e7358aa84308e6ae9e7999.jpg)

![b7299a86b88e12552b4ac9bacb945f352d35c41fef9d718cbe329f5367e55597.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/b7299a86b88e12552b4ac9bacb945f352d35c41fef9d718cbe329f5367e55597.jpg)

![f1b1fbbcf5afa0416fa25c8b814b25a465f512920e99c7fa7195bfd6ac4f9e94.jpg](../icml_results/5479_Diffusion%20Controller_%20Framework%2C%20Algorithms%20and%20Parameterization/tables/f1b1fbbcf5afa0416fa25c8b814b25a465f512920e99c7fa7195bfd6ac4f9e94.jpg)

<a id="paper-5432"></a>

## Branching Diffusion for Point Processes in Time and Space


### Images

![24568f3ee3f2351a824e549cb4671aa43cf461e1dd1fe6a6729d3c46e02e26a0.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/24568f3ee3f2351a824e549cb4671aa43cf461e1dd1fe6a6729d3c46e02e26a0.jpg)

![45a6f96080cd93bc74d62e9ce8315e6c542e7410f5189262a13ce87c3b8b0dff.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/45a6f96080cd93bc74d62e9ce8315e6c542e7410f5189262a13ce87c3b8b0dff.jpg)

![6393d18527dc6d1f1a03c692783935f392476c114c016d11df26f262f31e48e9.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/6393d18527dc6d1f1a03c692783935f392476c114c016d11df26f262f31e48e9.jpg)

![6f192080da3d8f7e82f34acad0ff77525a4f4e95bf704df7b981e34c41a1ba1a.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/6f192080da3d8f7e82f34acad0ff77525a4f4e95bf704df7b981e34c41a1ba1a.jpg)

![8995171d6036ba73df70228038bfbb6c68d50e2d2307a29f2de523916c2ac3bd.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/8995171d6036ba73df70228038bfbb6c68d50e2d2307a29f2de523916c2ac3bd.jpg)

![aa54ead47bd3c7c05a1e9c0620c916da9dacfc52bbfcd61087a767aff10e9a85.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/aa54ead47bd3c7c05a1e9c0620c916da9dacfc52bbfcd61087a767aff10e9a85.jpg)

![b895f0eaed02dbb43bc809abac43bc32bb3a437f04fa4b59922f391fad6c67a9.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/b895f0eaed02dbb43bc809abac43bc32bb3a437f04fa4b59922f391fad6c67a9.jpg)

![bf2fad1a8d97c0a2e592e89c9a123357eab93408a51c5a8a681163f7dcddafd9.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/bf2fad1a8d97c0a2e592e89c9a123357eab93408a51c5a8a681163f7dcddafd9.jpg)

![c7df7f30ea283ac36678bd2fa2ef9d79cad0d800919011dfc86223ee0fae961f.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/c7df7f30ea283ac36678bd2fa2ef9d79cad0d800919011dfc86223ee0fae961f.jpg)

![d900c6a7de11e6c218c93f5b14db41ba1e479092dcb87981dec16abdc3f293ab.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/d900c6a7de11e6c218c93f5b14db41ba1e479092dcb87981dec16abdc3f293ab.jpg)

![e36a0d10962fe4c5b6c0246a876582d6554784b513298e783c90192769bbef0f.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/images/e36a0d10962fe4c5b6c0246a876582d6554784b513298e783c90192769bbef0f.jpg)

### Tables

![28f6f25aa1425b7ecd542eeb4537e2bdfae18ac0e1b29e1d50e868b34566494f.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/tables/28f6f25aa1425b7ecd542eeb4537e2bdfae18ac0e1b29e1d50e868b34566494f.jpg)

![6f013bb10fa22bf290aea96df6b06377f60142203b429c76b2434f4b6dd3aff2.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/tables/6f013bb10fa22bf290aea96df6b06377f60142203b429c76b2434f4b6dd3aff2.jpg)

![7d80c40815b2a7bf773ebb9fe47617a65f4932dffb8e2c7482552a14a4bca04f.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/tables/7d80c40815b2a7bf773ebb9fe47617a65f4932dffb8e2c7482552a14a4bca04f.jpg)

![9aa068874dee6c4aa7e898c8be0b4bc5297ee5882adfc5aab2c1b5a93e2634e7.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/tables/9aa068874dee6c4aa7e898c8be0b4bc5297ee5882adfc5aab2c1b5a93e2634e7.jpg)

![e4a47495d55c556f8b56f7770a5a0b48329d7e70389416e03febbc07fb975229.jpg](../icml_results/5480_Branching%20Diffusion%20for%20Point%20Processes%20in%20Time%20and%20Space/tables/e4a47495d55c556f8b56f7770a5a0b48329d7e70389416e03febbc07fb975229.jpg)

<a id="paper-5433"></a>

## Orthogonal Model Merging


### Images

![153ea42dc42715fce2a7751bdd970ff0aaba7ffa6ca9807146d6e6d11f051a1a.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/153ea42dc42715fce2a7751bdd970ff0aaba7ffa6ca9807146d6e6d11f051a1a.jpg)

![3327e2689a4ab73859bed17e6019d2540498b87c9d4bc91a22ecbeb0d286f301.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/3327e2689a4ab73859bed17e6019d2540498b87c9d4bc91a22ecbeb0d286f301.jpg)

![577fc186a5c49ee103bee47d30f883e16e88d5f431c99972ff3c2e15ec9dcabc.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/577fc186a5c49ee103bee47d30f883e16e88d5f431c99972ff3c2e15ec9dcabc.jpg)

![7a11f26573d2bba4088a4aebb5320e66cc290793e6c6f1a2bb1a7b3547088686.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/7a11f26573d2bba4088a4aebb5320e66cc290793e6c6f1a2bb1a7b3547088686.jpg)

![8a9859d99ccb428eeb082ad126a9334b1d7d9d67d1adaaf41749f47b14abf416.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/8a9859d99ccb428eeb082ad126a9334b1d7d9d67d1adaaf41749f47b14abf416.jpg)

![a24e1c3e1c3e120f1845def8ebd3f88601852921dfa3dd5c0ce2b2b577a2c7c9.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/a24e1c3e1c3e120f1845def8ebd3f88601852921dfa3dd5c0ce2b2b577a2c7c9.jpg)

![b553a2cf976f5989ab4b6d7c3a3f9a342a1d1e916c1ced9586d0e70a48ea5562.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/b553a2cf976f5989ab4b6d7c3a3f9a342a1d1e916c1ced9586d0e70a48ea5562.jpg)

![bc5a8fbc7f310a2f0bfbf2a9778ac575bff7465442d44031c433fde3d86c715e.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/bc5a8fbc7f310a2f0bfbf2a9778ac575bff7465442d44031c433fde3d86c715e.jpg)

![cf5e1895a93a182a60d3a600219b3715ed00239ca0945ab08f8199dfed285726.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/images/cf5e1895a93a182a60d3a600219b3715ed00239ca0945ab08f8199dfed285726.jpg)

### Tables

![00503c3c3577734e7fbf5ee49d285fa218bfb0c9b41de2841e689d23e05f96d4.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/00503c3c3577734e7fbf5ee49d285fa218bfb0c9b41de2841e689d23e05f96d4.jpg)

![07d3eefc70031554020fcce15fdc538f35a684b1c26492ed607fba676cba4fbc.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/07d3eefc70031554020fcce15fdc538f35a684b1c26492ed607fba676cba4fbc.jpg)

![0e7852131c65cb4f94f94c5742069e2e20a6baa4dca43f72e8e285747a19dfea.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/0e7852131c65cb4f94f94c5742069e2e20a6baa4dca43f72e8e285747a19dfea.jpg)

![171ea1dc51450c86b793c07cc587b9bfda4a949a2ff32e4028752747498b1d58.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/171ea1dc51450c86b793c07cc587b9bfda4a949a2ff32e4028752747498b1d58.jpg)

![3a90aceb6c9ecdd7d9cfef56b65a7209e035baf2bf7bff0a6b23dc75cc70b2b4.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/3a90aceb6c9ecdd7d9cfef56b65a7209e035baf2bf7bff0a6b23dc75cc70b2b4.jpg)

![55b314be5925a9cedc1a3410e61bd3ea4d0676272bcd2ba9725d48cac899d9d9.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/55b314be5925a9cedc1a3410e61bd3ea4d0676272bcd2ba9725d48cac899d9d9.jpg)

![58d627f096885397d6a9520487deec6b51e4663333ee3b33ecc47a564763aa31.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/58d627f096885397d6a9520487deec6b51e4663333ee3b33ecc47a564763aa31.jpg)

![5c12267990915c3d35ee6155d8fac4a178c8a849c5833f5c3dc7a3263c92d85e.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/5c12267990915c3d35ee6155d8fac4a178c8a849c5833f5c3dc7a3263c92d85e.jpg)

![5d33020f819ff21f06e3e5c0c1487fa8a09df2b40f469edc14910057b2b361ce.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/5d33020f819ff21f06e3e5c0c1487fa8a09df2b40f469edc14910057b2b361ce.jpg)

![762e66d99a5c27442abe3f75883981ead5508a6a8cb2f38b1dcbd08feb866416.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/762e66d99a5c27442abe3f75883981ead5508a6a8cb2f38b1dcbd08feb866416.jpg)

![a14b7790eb758b49e212b9408d831ae915db08c43a65afa2acc440be279601b1.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/a14b7790eb758b49e212b9408d831ae915db08c43a65afa2acc440be279601b1.jpg)

![d4f7a70b2b6823df4ee184c79479fbf1e1f799c67b12d1c52ea0d05b6aa1365e.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/d4f7a70b2b6823df4ee184c79479fbf1e1f799c67b12d1c52ea0d05b6aa1365e.jpg)

![fcbaecbe96be847eded0a2a562d6d95d90d38b7f5c544d1bbf5fcd33e684af53.jpg](../icml_results/5481_Orthogonal%20Model%20Merging/tables/fcbaecbe96be847eded0a2a562d6d95d90d38b7f5c544d1bbf5fcd33e684af53.jpg)

<a id="paper-5434"></a>

## AgentTailor: A Semantic-Aware LLM-Based Multi-Agent System with Actor-Critic Structure


### Images

![0b90b5b37d3fdccbc3523d404c42bd22d286ff879edcc4e7b16113cca42469f2.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/images/0b90b5b37d3fdccbc3523d404c42bd22d286ff879edcc4e7b16113cca42469f2.jpg)

![1a398c7eb178ede1e740f11c2da863b228a4df4f4477420dac3fd919ae044f76.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/images/1a398c7eb178ede1e740f11c2da863b228a4df4f4477420dac3fd919ae044f76.jpg)

![334c7e4d7f1c8eefff5532e1ef96333045f15e059b820db182afd8014b8650ae.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/images/334c7e4d7f1c8eefff5532e1ef96333045f15e059b820db182afd8014b8650ae.jpg)

![5316d8e77476382b537ddc2ab098d322aa6fb5ad4101b3ffb0839e573112754a.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/images/5316d8e77476382b537ddc2ab098d322aa6fb5ad4101b3ffb0839e573112754a.jpg)

![dbb8256f5d8ad5ba6fb1b98d0f564865da2636344b8c93d24870cc403570e8b8.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/images/dbb8256f5d8ad5ba6fb1b98d0f564865da2636344b8c93d24870cc403570e8b8.jpg)

### Tables

![030849a27ffb80bbe4012a0e6b5345db87508b00aa4e3e12062f97080024bc0d.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/030849a27ffb80bbe4012a0e6b5345db87508b00aa4e3e12062f97080024bc0d.jpg)

![07618df376497d77db8d950ad7c29f4d46b5e457e481910bc66e2ab564b35f33.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/07618df376497d77db8d950ad7c29f4d46b5e457e481910bc66e2ab564b35f33.jpg)

![31174196b401c6e0fa0d4aafad9f367a3bdd2ca0b4e709bf1688391d4df726de.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/31174196b401c6e0fa0d4aafad9f367a3bdd2ca0b4e709bf1688391d4df726de.jpg)

![4058ca8f43754b47b7673d70029e36737df2d4d46be7410f1a4e53a98f07284d.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/4058ca8f43754b47b7673d70029e36737df2d4d46be7410f1a4e53a98f07284d.jpg)

![4e5de76869e0dba3a80c84f211d5e365edcf360dc48f9565ea60be4e1172b373.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/4e5de76869e0dba3a80c84f211d5e365edcf360dc48f9565ea60be4e1172b373.jpg)

![5a70cc99c0ecfab92272f09d7ad1ca90eb7e71b756c05f0321c4cf8001c13089.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/5a70cc99c0ecfab92272f09d7ad1ca90eb7e71b756c05f0321c4cf8001c13089.jpg)

![6def9377c268f6d6c8036981590a7177b50da4c9f863258cc6bac4030a39ee11.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/6def9377c268f6d6c8036981590a7177b50da4c9f863258cc6bac4030a39ee11.jpg)

![7005a35bffaccef5942e5b0b3e6e20e5e0f0f4644fbf6be20d7a98704ca08a20.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/7005a35bffaccef5942e5b0b3e6e20e5e0f0f4644fbf6be20d7a98704ca08a20.jpg)

![775c96af7baf5fc34806de807ba208a750a9826897e95a435658aff86f34da9f.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/775c96af7baf5fc34806de807ba208a750a9826897e95a435658aff86f34da9f.jpg)

![7d53620fb72aef2a3b51acf748823be9ce1b451ab8eb14293a9d4c6e08a9a705.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/7d53620fb72aef2a3b51acf748823be9ce1b451ab8eb14293a9d4c6e08a9a705.jpg)

![a6f36b99cc2944ac3a16a964f7975295666dbd6ab501e1410b7ec237f9ae0a91.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/a6f36b99cc2944ac3a16a964f7975295666dbd6ab501e1410b7ec237f9ae0a91.jpg)

![ea8fb1ec11b65a8c6315e862ce003e538ab91457c4667a554c168f02057faa76.jpg](../icml_results/5482_AgentTailor_%20A%20Semantic-Aware%20LLM-Based%20Multi-Agent%20System%20with%20Actor-Critic%20Structure/tables/ea8fb1ec11b65a8c6315e862ce003e538ab91457c4667a554c168f02057faa76.jpg)

<a id="paper-5435"></a>

## Recursive Models for Long-Horizon Reasoning


### Images

![0d56bebf2fb78f3729309c5dd5d62a44bf39815e74f9b993e24a2f4c78e41bf1.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/images/0d56bebf2fb78f3729309c5dd5d62a44bf39815e74f9b993e24a2f4c78e41bf1.jpg)

![4f56fe7291b87909affa27c6b7a429bfc8850cfba03ab4f90ee284bce896f26c.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/images/4f56fe7291b87909affa27c6b7a429bfc8850cfba03ab4f90ee284bce896f26c.jpg)

![61507aa2ef8beb7b2bb23cf3b671f24a30f405ddfc277445267102466fb78d4a.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/images/61507aa2ef8beb7b2bb23cf3b671f24a30f405ddfc277445267102466fb78d4a.jpg)

![bdc9855b16b6a96f09123ad96e7f177ea8e378cc7ec71ed13d5052c8cb7ebf47.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/images/bdc9855b16b6a96f09123ad96e7f177ea8e378cc7ec71ed13d5052c8cb7ebf47.jpg)

### Tables

![0844d61239bd151c0a0e7de739419d962963cc780b3e369a4366746e99613a21.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/tables/0844d61239bd151c0a0e7de739419d962963cc780b3e369a4366746e99613a21.jpg)

![8e1fc03c1590db9e71cf8bbb6daeb4376090534e97b4c772c9c7f9e600d346cf.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/tables/8e1fc03c1590db9e71cf8bbb6daeb4376090534e97b4c772c9c7f9e600d346cf.jpg)

![8f04e743028bdac80f07cc1a892eb8ad0ebe7465ab7880d83cc9d1f5b879f52a.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/tables/8f04e743028bdac80f07cc1a892eb8ad0ebe7465ab7880d83cc9d1f5b879f52a.jpg)

![9e976dd90eafd37ac236c08a5afd0218408d57696066049a9a896f8a74cf2ac7.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/tables/9e976dd90eafd37ac236c08a5afd0218408d57696066049a9a896f8a74cf2ac7.jpg)

![c79277ade8acc010d3164d5bd70f6a316149126a76a60aa5d23c496d299471e8.jpg](../icml_results/5483_Recursive%20Models%20for%20Long-Horizon%20Reasoning/tables/c79277ade8acc010d3164d5bd70f6a316149126a76a60aa5d23c496d299471e8.jpg)

<a id="paper-5436"></a>

## Data Agent: Learning to Select Data via End-to-End Dynamic Optimization


### Images

![0e09953f689339f7e08fef2c0540dfe5a141f61ff3329ae4d74c1180a86f3c8a.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/0e09953f689339f7e08fef2c0540dfe5a141f61ff3329ae4d74c1180a86f3c8a.jpg)

![2e61ef1fe15310352252396085503a3d59a5877b79a5d9b2e51e74efb1ef920b.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/2e61ef1fe15310352252396085503a3d59a5877b79a5d9b2e51e74efb1ef920b.jpg)

![4e3e41a3ed9ee9bb1b122e21da50fb762ab1582a28f7d9e7fde6eaf66504e5bd.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/4e3e41a3ed9ee9bb1b122e21da50fb762ab1582a28f7d9e7fde6eaf66504e5bd.jpg)

![52331a9e8a2d935344ca27b66030ab01cf4fe3ffb2e0aae2ad240f3599ebf173.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/52331a9e8a2d935344ca27b66030ab01cf4fe3ffb2e0aae2ad240f3599ebf173.jpg)

![55e13edeb332b606c2fcb73c64fa42ecfab7264858ea9a5c3a78841d6fbf6122.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/55e13edeb332b606c2fcb73c64fa42ecfab7264858ea9a5c3a78841d6fbf6122.jpg)

![679e42fa56f22028f25a467f9a387409242985529708759cc1b592272e40dcfd.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/images/679e42fa56f22028f25a467f9a387409242985529708759cc1b592272e40dcfd.jpg)

### Tables

![12f0030c089a0adff91aae7a4c54f852c160ad82e2a6cfd492d1d54cde028195.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/12f0030c089a0adff91aae7a4c54f852c160ad82e2a6cfd492d1d54cde028195.jpg)

![45edc9c7d0a52f8c5862ca3488be257b14b600c9657eab98296e234a1c78e0f9.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/45edc9c7d0a52f8c5862ca3488be257b14b600c9657eab98296e234a1c78e0f9.jpg)

![6a6818fc077a361d5987a231085e754902598f6f282b0a2a52007430c2e582b4.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/6a6818fc077a361d5987a231085e754902598f6f282b0a2a52007430c2e582b4.jpg)

![6c48ca97db399618b71ebdd26e77590162e3aae1e5032cef21364886b72f4e12.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/6c48ca97db399618b71ebdd26e77590162e3aae1e5032cef21364886b72f4e12.jpg)

![6c6f58566164c987d5db8a2111f7411c44782d928f9e5c36175dc5e3950e21d6.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/6c6f58566164c987d5db8a2111f7411c44782d928f9e5c36175dc5e3950e21d6.jpg)

![b95d79c0e6a29e0bfc298cd612b4fb7b701bad4cd1d5eada6ab5a912516d174e.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/b95d79c0e6a29e0bfc298cd612b4fb7b701bad4cd1d5eada6ab5a912516d174e.jpg)

![c52a31719508d892c219e79c7de6a0044e9ecc96f8a9c038c005479dd94ee810.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/c52a31719508d892c219e79c7de6a0044e9ecc96f8a9c038c005479dd94ee810.jpg)

![e68dbfd6b55864aa45671d6662ee94817213bdcbf424e80f44919f71a7da889c.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/e68dbfd6b55864aa45671d6662ee94817213bdcbf424e80f44919f71a7da889c.jpg)

![e9c6e41df5df20d3db9b86d1f307f81284be8388f9bf59c4f094c2267c87235f.jpg](../icml_results/5484_Data%20Agent_%20Learning%20to%20Select%20Data%20via%20End-to-End%20Dynamic%20Optimization/tables/e9c6e41df5df20d3db9b86d1f307f81284be8388f9bf59c4f094c2267c87235f.jpg)

<a id="paper-5437"></a>

## DecFus: Decentralized Layer-wise Fusion with Dynamic Exploration and Exploitation


### Images

![18f8ff5ad0805b05c02e266fd35d6bc9aeaf816305a708f52db619771662738e.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/18f8ff5ad0805b05c02e266fd35d6bc9aeaf816305a708f52db619771662738e.jpg)

![1d719157567c521cbd7e8dd9414304fc7742c404987833754d2d09dbde8784f0.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/1d719157567c521cbd7e8dd9414304fc7742c404987833754d2d09dbde8784f0.jpg)

![2bc85a30564c3e5e568044e331d6a64ce53cf2396cb82df7cc385d8651403024.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/2bc85a30564c3e5e568044e331d6a64ce53cf2396cb82df7cc385d8651403024.jpg)

![79174016cc64bf2e9a443d13dee40b90387ba10a3b2f82183a48b65b2fd7f300.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/79174016cc64bf2e9a443d13dee40b90387ba10a3b2f82183a48b65b2fd7f300.jpg)

![8e3230219b1f22699692886442820d77664f77711a1d867152f21177e6bff5b2.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/8e3230219b1f22699692886442820d77664f77711a1d867152f21177e6bff5b2.jpg)

![9e949be0eea2a6c219d82e836f43688362e2218c3619032068aacefaf4c722c1.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/9e949be0eea2a6c219d82e836f43688362e2218c3619032068aacefaf4c722c1.jpg)

![a6b312a9ca8b03df12d425e68b626277da98365c8f21f2c7b4ff2134fbfd8294.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/a6b312a9ca8b03df12d425e68b626277da98365c8f21f2c7b4ff2134fbfd8294.jpg)

![d6985b83e87e6138fc290b93a7a8a80f0b57a840b7ff95f72de4a1d2a0b7cc62.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/d6985b83e87e6138fc290b93a7a8a80f0b57a840b7ff95f72de4a1d2a0b7cc62.jpg)

![e6c92cc59404c09197074930418a925eb72f5ffc28b4f48f25e8c106bfaae676.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/images/e6c92cc59404c09197074930418a925eb72f5ffc28b4f48f25e8c106bfaae676.jpg)

### Tables

![191bb0a4ff5855c44551b90e0696c71f96a31214b7bdf72a791ca5fe5c6c9da5.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/191bb0a4ff5855c44551b90e0696c71f96a31214b7bdf72a791ca5fe5c6c9da5.jpg)

![1c01930a83dfb7c2269570dc3dc5bb22a6a44f98ebb93a771493e3d1b9d80fe4.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/1c01930a83dfb7c2269570dc3dc5bb22a6a44f98ebb93a771493e3d1b9d80fe4.jpg)

![520ad37da7389cdaed90d6797ead3d1a2cd3249cece07aa9fe7f0bcae6418a6b.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/520ad37da7389cdaed90d6797ead3d1a2cd3249cece07aa9fe7f0bcae6418a6b.jpg)

![777c6b8b7848ee261e9a996ed1fe1ac7222dccfcce84aa14e022597cc99e2b5a.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/777c6b8b7848ee261e9a996ed1fe1ac7222dccfcce84aa14e022597cc99e2b5a.jpg)

![7a2a6559f9fca747868055f504a1a386bd3ab1bfdf7715e9676294c236c48306.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/7a2a6559f9fca747868055f504a1a386bd3ab1bfdf7715e9676294c236c48306.jpg)

![86977518d90996f25134c106781fb763c8dcdd83d508c7a589685ac626d65311.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/86977518d90996f25134c106781fb763c8dcdd83d508c7a589685ac626d65311.jpg)

![87119d6492d666e705da11804ffce192d3c1cc9bce57bd2ff785d5371d8fb701.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/87119d6492d666e705da11804ffce192d3c1cc9bce57bd2ff785d5371d8fb701.jpg)

![b2c5c80aabfde6e4d73bd644544405b7185683cc473a791bafaddd7e0acd7270.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/b2c5c80aabfde6e4d73bd644544405b7185683cc473a791bafaddd7e0acd7270.jpg)

![f7ea0172676ed6e9f02d1eb5b0e174e60b1383707b205cd081d140e58db05440.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/f7ea0172676ed6e9f02d1eb5b0e174e60b1383707b205cd081d140e58db05440.jpg)

![fe998ed89ea5c2ee93a2016704e2b0cbc4d2d7da8eb295feb78c715a986fe788.jpg](../icml_results/5485_DecFus_%20Decentralized%20Layer-wise%20Fusion%20with%20Dynamic%20Exploration%20and%20Exploitation/tables/fe998ed89ea5c2ee93a2016704e2b0cbc4d2d7da8eb295feb78c715a986fe788.jpg)

<a id="paper-5438"></a>

## WMVLM: Evaluating Diffusion Model Image Watermarking via Vision-Language Models


### Images

![017d7a72c3fb70595b42f326e00badb2c9cf569f6dca94133555192ac4b4bb47.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/017d7a72c3fb70595b42f326e00badb2c9cf569f6dca94133555192ac4b4bb47.jpg)

![09b48e73a83fbba3d5d949d5db7997f17f90739d5e9556520ce49b41f39c79c4.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/09b48e73a83fbba3d5d949d5db7997f17f90739d5e9556520ce49b41f39c79c4.jpg)

![16bbca87bca39a18a8355892f334b2c3b91bd3739632101552ecc0e5fedbbc20.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/16bbca87bca39a18a8355892f334b2c3b91bd3739632101552ecc0e5fedbbc20.jpg)

![1aadc4246d649045104cc748a0a4362aa2a7667264403a30600947c29f03bbe7.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/1aadc4246d649045104cc748a0a4362aa2a7667264403a30600947c29f03bbe7.jpg)

![7fb2a4e0c1e645da7c4fad8195ffc4dbb397a1db0c51e9b691fb690960922976.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/7fb2a4e0c1e645da7c4fad8195ffc4dbb397a1db0c51e9b691fb690960922976.jpg)

![8f317e93538c80c0d98f6520f3f40aec11e84c45bdf99993dee36707dd5a93f4.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/8f317e93538c80c0d98f6520f3f40aec11e84c45bdf99993dee36707dd5a93f4.jpg)

![d4c3ee781182e479aa09e125084db530c913fe79291e2b839c7151825289f0c3.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/images/d4c3ee781182e479aa09e125084db530c913fe79291e2b839c7151825289f0c3.jpg)

### Tables

![08655e7ff5546f747118a64b15b9c41362f9b339441a1fe2e53bfb421b0f7824.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/08655e7ff5546f747118a64b15b9c41362f9b339441a1fe2e53bfb421b0f7824.jpg)

![1182b74286c15bead2956a7b269f55c2e7f13cad8837d7dbb099a0f3f8d992cc.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/1182b74286c15bead2956a7b269f55c2e7f13cad8837d7dbb099a0f3f8d992cc.jpg)

![1cbefb00b4a960d938f95dc6118883eaf68e68e0c6807966b4e88b0bd24977b5.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/1cbefb00b4a960d938f95dc6118883eaf68e68e0c6807966b4e88b0bd24977b5.jpg)

![2aef23956e0c5002c5aec28a59925a9ab02f1f322c4edb522515ea3c22cb5ff7.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/2aef23956e0c5002c5aec28a59925a9ab02f1f322c4edb522515ea3c22cb5ff7.jpg)

![3b273ff204972d22f59d459d614010f03f13d9383571c6a8ca89b3d3c015046c.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/3b273ff204972d22f59d459d614010f03f13d9383571c6a8ca89b3d3c015046c.jpg)

![3c17109dfa7d189797ff7f5e9f98c97c438d95b1bae5021b188f8961daac5c2d.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/3c17109dfa7d189797ff7f5e9f98c97c438d95b1bae5021b188f8961daac5c2d.jpg)

![4a1ee62f284979b39ea7387b52d2d3cc23476d5350382e4915a883f0a5a59fd5.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/4a1ee62f284979b39ea7387b52d2d3cc23476d5350382e4915a883f0a5a59fd5.jpg)

![b35f30f11622fffa22c251981500ec5bcb82ccbf0596f31bd58dd6e105b2d758.jpg](../icml_results/5486_WMVLM_%20Evaluating%20Diffusion%20Model%20Image%20Watermarking%20via%20Vision-Language%20Models/tables/b35f30f11622fffa22c251981500ec5bcb82ccbf0596f31bd58dd6e105b2d758.jpg)

<a id="paper-5439"></a>

## ExVerus: Verus Proof Repair via Counterexample Reasoning


### Images

![0c6d8d9a35995dd3bef83c651171345c8b4e535a81f3d588cc9ac692046c46c6.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/0c6d8d9a35995dd3bef83c651171345c8b4e535a81f3d588cc9ac692046c46c6.jpg)

![14ed9c1c952a471bc6c6c26f083aaaf44a0e4a2b6d0b8f681d962c8578cdb7a4.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/14ed9c1c952a471bc6c6c26f083aaaf44a0e4a2b6d0b8f681d962c8578cdb7a4.jpg)

![14fcd86a5653244419370ad551d5cf49f15a9d15f6b7b7069b4a18c20702316b.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/14fcd86a5653244419370ad551d5cf49f15a9d15f6b7b7069b4a18c20702316b.jpg)

![463a66c65845eea02af5a390e093351b1f45f4a55ce95399c75f7b97566a731e.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/463a66c65845eea02af5a390e093351b1f45f4a55ce95399c75f7b97566a731e.jpg)

![6be1512d032e0cf764a901904565da59df9b52331a97516cff482b341bf8b042.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/6be1512d032e0cf764a901904565da59df9b52331a97516cff482b341bf8b042.jpg)

![722ed8bedf23d2f11d1e9182b1045ebb8f8c0ffcdf2de64d570735d4ecafd89b.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/722ed8bedf23d2f11d1e9182b1045ebb8f8c0ffcdf2de64d570735d4ecafd89b.jpg)

![8a988bdb505ff33d5a55545a9f18b5059b3e1e098b7a7bf249ce3319048313a9.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/8a988bdb505ff33d5a55545a9f18b5059b3e1e098b7a7bf249ce3319048313a9.jpg)

![90c5118bdcd645a80a0aff652c3a69053c939511fc265466b62f4fb5a03ccff8.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/90c5118bdcd645a80a0aff652c3a69053c939511fc265466b62f4fb5a03ccff8.jpg)

![9c0cf703eeb102ee05f22a11aadc8cff308650c32a5805d796c9a75512e14236.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/images/9c0cf703eeb102ee05f22a11aadc8cff308650c32a5805d796c9a75512e14236.jpg)

### Tables

![3d51fccd037d1ee0ad88b7c03ab6b0a76470cb093d57a503b2dc0e3ab924519c.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/tables/3d51fccd037d1ee0ad88b7c03ab6b0a76470cb093d57a503b2dc0e3ab924519c.jpg)

![58bbfac12af3f97d44535b71f093ef68849d26170661c88b00d3f06a74d1f252.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/tables/58bbfac12af3f97d44535b71f093ef68849d26170661c88b00d3f06a74d1f252.jpg)

![687e5125b7809c0482ae755ace969a7b95dcf917ce7a63a2b8785e8bdc9fbfbf.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/tables/687e5125b7809c0482ae755ace969a7b95dcf917ce7a63a2b8785e8bdc9fbfbf.jpg)

![bea4760db9ba6c0880c442585aa20fd25990fe3f67ebe7a09cad2651bf53cb7a.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/tables/bea4760db9ba6c0880c442585aa20fd25990fe3f67ebe7a09cad2651bf53cb7a.jpg)

![ed6f2d3ad46443f7d56e2fef352c3c902cd483c124eaa10d382191a8818d16b8.jpg](../icml_results/5487_ExVerus_%20Verus%20Proof%20Repair%20via%20Counterexample%20Reasoning/tables/ed6f2d3ad46443f7d56e2fef352c3c902cd483c124eaa10d382191a8818d16b8.jpg)

<a id="paper-5440"></a>

## AutoMat: Physics-Guided Agentic Reasoning for Solving Ill-Posed Inverse Microscopy Problems


### Images

![1bf9835bfbc8b45328e3cdc01e97d6fc0f7c97d2cb260c3176a2e390ac719b1f.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/1bf9835bfbc8b45328e3cdc01e97d6fc0f7c97d2cb260c3176a2e390ac719b1f.jpg)

![36ac3d2a31280c2aace283e23e2409b87f076aeab0d058a4a81657473355a40a.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/36ac3d2a31280c2aace283e23e2409b87f076aeab0d058a4a81657473355a40a.jpg)

![60fbea3a41100417e9bef65e2b3725ef3a56d1558f4daa3ef5b1bee5287b26d6.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/60fbea3a41100417e9bef65e2b3725ef3a56d1558f4daa3ef5b1bee5287b26d6.jpg)

![9528767eeefc105353f7fa417662c13d0c8da63507c055e109d5ef062931689e.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/9528767eeefc105353f7fa417662c13d0c8da63507c055e109d5ef062931689e.jpg)

![e22a1e5f696da4d9320a8c9b04e2274474d494c1ca8004af4066cb588a4b73a6.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/e22a1e5f696da4d9320a8c9b04e2274474d494c1ca8004af4066cb588a4b73a6.jpg)

![e6aaae118dc9e2924a4d028a7d17fc96ce06f47a53dc140377875c84c30d934a.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/e6aaae118dc9e2924a4d028a7d17fc96ce06f47a53dc140377875c84c30d934a.jpg)

![fe471ab5ab4ece88442551cb53765c1cd17bea6d5728892d088361aa3ba406f5.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/images/fe471ab5ab4ece88442551cb53765c1cd17bea6d5728892d088361aa3ba406f5.jpg)

### Tables

![00ada0288a717515f4c216bfc753d631a1c526059488032d88368a2babb59259.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/00ada0288a717515f4c216bfc753d631a1c526059488032d88368a2babb59259.jpg)

![28da3b7bda7d25ef3efda96ae8719304a350d69930767421ab9f7fb7355e1150.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/28da3b7bda7d25ef3efda96ae8719304a350d69930767421ab9f7fb7355e1150.jpg)

![52da3dabc6a9b2cd932b78966fe66431c1debb79ae837ebf9a0b4099b4e20837.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/52da3dabc6a9b2cd932b78966fe66431c1debb79ae837ebf9a0b4099b4e20837.jpg)

![81aec9e69223ea258484c3ba6924524c41e6eeb38c070193f34de1af99c834ff.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/81aec9e69223ea258484c3ba6924524c41e6eeb38c070193f34de1af99c834ff.jpg)

![9268e2098d1153c60e6e20e94c0d0e81c683de62335363d7080f43207a465cb7.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/9268e2098d1153c60e6e20e94c0d0e81c683de62335363d7080f43207a465cb7.jpg)

![ab3170ed7b521af2a8b7da25697a0b8d97afad98224803d0c4ebc57884f9b717.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/ab3170ed7b521af2a8b7da25697a0b8d97afad98224803d0c4ebc57884f9b717.jpg)

![b6da9375d99d8a77243918f493a629e5f4eca833825b4c02e82365ef129ae73e.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/b6da9375d99d8a77243918f493a629e5f4eca833825b4c02e82365ef129ae73e.jpg)

![d49add24ae21c2f10c3e51aa193b6a7fbf54bde3f4b4601ecfe39f253fe7abe3.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/d49add24ae21c2f10c3e51aa193b6a7fbf54bde3f4b4601ecfe39f253fe7abe3.jpg)

![da6ebe6897f1b800dde6b7e7f7a105eacea591da2ea3f7f7178f8da6e06dfc7b.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/da6ebe6897f1b800dde6b7e7f7a105eacea591da2ea3f7f7178f8da6e06dfc7b.jpg)

![ea8945f8c49b02bfe3b93b72cd6ca8c9990757dac859f6fbed265549caa084b1.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/ea8945f8c49b02bfe3b93b72cd6ca8c9990757dac859f6fbed265549caa084b1.jpg)

![f1a20f392bb7a4b550c90f255f54611129a2afa2979207b9cb2cdb766e8354bb.jpg](../icml_results/5488_AutoMat_%20Physics-Guided%20Agentic%20Reasoning%20for%20Solving%20Ill-Posed%20Inverse%20Microscopy%20Problems/tables/f1a20f392bb7a4b550c90f255f54611129a2afa2979207b9cb2cdb766e8354bb.jpg)

<a id="paper-5441"></a>

## Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective


### Images

![09905db3189b6ba3b759bde2e9639872b4d84cc72f3c115c6fa2a93d4b6c9d58.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/09905db3189b6ba3b759bde2e9639872b4d84cc72f3c115c6fa2a93d4b6c9d58.jpg)

![0ddbd4115dcf9591a43e9034c974b5c651ae733f9d6fee58f904cbafeaea9625.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/0ddbd4115dcf9591a43e9034c974b5c651ae733f9d6fee58f904cbafeaea9625.jpg)

![1408e0ff6f88c3f518f11fa92c9265ed930a8e4ac7e6e830c984895daeecd30d.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/1408e0ff6f88c3f518f11fa92c9265ed930a8e4ac7e6e830c984895daeecd30d.jpg)

![4307ffb9a384f633e85e28abe9abd33d121638b776cf833e57128a6c7e6b2a1f.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/4307ffb9a384f633e85e28abe9abd33d121638b776cf833e57128a6c7e6b2a1f.jpg)

![9ab3b2f5d7c972e04a4b4090610e5a6701682c8880752a8698b5e2b4de6aded5.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/9ab3b2f5d7c972e04a4b4090610e5a6701682c8880752a8698b5e2b4de6aded5.jpg)

![e387f0c7c180f7790d403dca197ceee4179c0b00ae9ce25e4752df61a34250ea.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/images/e387f0c7c180f7790d403dca197ceee4179c0b00ae9ce25e4752df61a34250ea.jpg)

### Tables

![039751066ce6775a3b62fc43a9ec32fe280b1b37cf1e58bfb09352a1e1c584a4.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/039751066ce6775a3b62fc43a9ec32fe280b1b37cf1e58bfb09352a1e1c584a4.jpg)

![0adcece8a991bccd08ef41a22dfba25ec526653d400a6dc0c8c2c587e731ddea.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/0adcece8a991bccd08ef41a22dfba25ec526653d400a6dc0c8c2c587e731ddea.jpg)

![20a33f989ff8312f6d497b6cf074a648ecc2786975185929cf6ca69534d0cd78.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/20a33f989ff8312f6d497b6cf074a648ecc2786975185929cf6ca69534d0cd78.jpg)

![4e34c09dc5433ae6ffc5e7abcff48129b152d78c0b25cda819f2a5c7fbd8d78e.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/4e34c09dc5433ae6ffc5e7abcff48129b152d78c0b25cda819f2a5c7fbd8d78e.jpg)

![546e94b400c58c2e82535ed0ab41f6dcaa510f7af17f9095dcab7ff94170d991.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/546e94b400c58c2e82535ed0ab41f6dcaa510f7af17f9095dcab7ff94170d991.jpg)

![571747c34d6507294c3e3b6d93783341469f9c769e5d205182d5c47d063c169f.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/571747c34d6507294c3e3b6d93783341469f9c769e5d205182d5c47d063c169f.jpg)

![b97b6c84046cd813952d3da733b42e1246ac271ec43b3cb0ead5590c8948dcb5.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/b97b6c84046cd813952d3da733b42e1246ac271ec43b3cb0ead5590c8948dcb5.jpg)

![e3a90bd3cca42d01c151eceec009f7f1b38a627f8f2966453f08c3f0ff397703.jpg](../icml_results/5489_Rethinking%20KV%20Cache%20Eviction%20via%20a%20Unified%20Information-Theoretic%20Objective/tables/e3a90bd3cca42d01c151eceec009f7f1b38a627f8f2966453f08c3f0ff397703.jpg)

<a id="paper-5442"></a>

## Revisiting Robustness for LLM Safety Alignment via Selective Geometry Control


### Images

![0a9925ca91111cc74cd5fae88ed6e51e4dd84a888d2b752ca9caf54a84d009dd.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/0a9925ca91111cc74cd5fae88ed6e51e4dd84a888d2b752ca9caf54a84d009dd.jpg)

![10e7bc9d9d42b4e356246b10797bdfc756f88f5c7647015aa65ac16e90dced72.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/10e7bc9d9d42b4e356246b10797bdfc756f88f5c7647015aa65ac16e90dced72.jpg)

![32f3f165dbccde348f14b7e8c48181f8087d5a39b621b167a172e1aee2c2a1a1.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/32f3f165dbccde348f14b7e8c48181f8087d5a39b621b167a172e1aee2c2a1a1.jpg)

![3790a5ce826d594cbdfa92debaf77b6550759016d1406f2a33a429ab9224ab1c.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/3790a5ce826d594cbdfa92debaf77b6550759016d1406f2a33a429ab9224ab1c.jpg)

![5c0d2c0d5b5c83463514ebd98e53f8008f46c7e7840ac22acbffd5a86201f430.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/5c0d2c0d5b5c83463514ebd98e53f8008f46c7e7840ac22acbffd5a86201f430.jpg)

![68819505b2f80315782ec99018a24f55877a181b0e354e3b1d1f869aec64f999.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/68819505b2f80315782ec99018a24f55877a181b0e354e3b1d1f869aec64f999.jpg)

![9b0e9a7cb189fcfd8e91eec3097feca6fa7600a52a5de42e1f9141a4092ee2b4.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/9b0e9a7cb189fcfd8e91eec3097feca6fa7600a52a5de42e1f9141a4092ee2b4.jpg)

![e35204d3989d54384e118400283729fc04ae8dee9836d69aaf6d8e8f40117763.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/images/e35204d3989d54384e118400283729fc04ae8dee9836d69aaf6d8e8f40117763.jpg)

### Tables

![010a02c40a447a5301597f48fb89be8286150d60b0d1a19950073fcf5a2fcd08.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/010a02c40a447a5301597f48fb89be8286150d60b0d1a19950073fcf5a2fcd08.jpg)

![2dee688f0014efd7fd9763c47ffea3960d48dd379ebd87f44374c1995d4554b0.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/2dee688f0014efd7fd9763c47ffea3960d48dd379ebd87f44374c1995d4554b0.jpg)

![344491bd0452eb776befc2334d78f5ba608213e5d728692380bb7259135ac5ae.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/344491bd0452eb776befc2334d78f5ba608213e5d728692380bb7259135ac5ae.jpg)

![67a83b9484b9d4da848615a313404e80ed29ce95b4b06257a3862e727782695f.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/67a83b9484b9d4da848615a313404e80ed29ce95b4b06257a3862e727782695f.jpg)

![88d78d792fe1f78ce341445921b0db76771ca8fdc313997bb1ffb08f1f9191ec.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/88d78d792fe1f78ce341445921b0db76771ca8fdc313997bb1ffb08f1f9191ec.jpg)

![8b8ff4b673d46800d89dc8f8e99034ae6c19c8a9b8196d913b0917c089b960ba.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/8b8ff4b673d46800d89dc8f8e99034ae6c19c8a9b8196d913b0917c089b960ba.jpg)

![d74ebcdab50e2910a15cccd7544953f1669021314a0404ee6a446087520b9916.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/d74ebcdab50e2910a15cccd7544953f1669021314a0404ee6a446087520b9916.jpg)

![f894c36710fb42099d4e3602aed92ee931bd1f0922dee35264f113199845f944.jpg](../icml_results/5490_Revisiting%20Robustness%20for%20LLM%20Safety%20Alignment%20via%20Selective%20Geometry%20Control/tables/f894c36710fb42099d4e3602aed92ee931bd1f0922dee35264f113199845f944.jpg)

<a id="paper-5443"></a>

## Improving Sampling for Masked Diffusion Models via Information Gain


### Images

![03cac044b61e5a9e8369ef258c5099fc07bd617bdab6387ec98f057b6ff357be.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/03cac044b61e5a9e8369ef258c5099fc07bd617bdab6387ec98f057b6ff357be.jpg)

![0a100624e6bf2b7b52cb38fe4d35555d785d1d23f0aca9ab20214feac3a926f6.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/0a100624e6bf2b7b52cb38fe4d35555d785d1d23f0aca9ab20214feac3a926f6.jpg)

![223c1cb49b135b35b51d16dcf258f12a48e24357b31e1e6ed734dce1b6389e4e.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/223c1cb49b135b35b51d16dcf258f12a48e24357b31e1e6ed734dce1b6389e4e.jpg)

![2c269589617141c467ce9bf7590080c2dd55b69597088fdd5920da70b170412e.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/2c269589617141c467ce9bf7590080c2dd55b69597088fdd5920da70b170412e.jpg)

![655b75c99ad7f6b3f96f2140b9c55edef698f79886a5bdb107d23c469de058e7.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/655b75c99ad7f6b3f96f2140b9c55edef698f79886a5bdb107d23c469de058e7.jpg)

![7cb958682675d59747e5d2fa4b2b0a08ad97bc716f350f2ff9da0d80f8fde7b5.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/7cb958682675d59747e5d2fa4b2b0a08ad97bc716f350f2ff9da0d80f8fde7b5.jpg)

![963a5575e33259a61762901985af24b18825d95bb1583bfcd6d59a64656ac592.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/963a5575e33259a61762901985af24b18825d95bb1583bfcd6d59a64656ac592.jpg)

![a11a06bf1ec8c8a61b748ec13df7fe7e793f865a42924ba42d662d90dc1a4949.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/a11a06bf1ec8c8a61b748ec13df7fe7e793f865a42924ba42d662d90dc1a4949.jpg)

![ce41ae3e93aa90721b4c203d674e071932a6a8c1e28c54a872510de1f318b2ff.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/ce41ae3e93aa90721b4c203d674e071932a6a8c1e28c54a872510de1f318b2ff.jpg)

![ece83e2f07caf4045f1b70950da4992003032702e3710ef7369154e1e35e94f8.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/images/ece83e2f07caf4045f1b70950da4992003032702e3710ef7369154e1e35e94f8.jpg)

### Tables

![21e8ba073fc3859fcab406e3e17ea13967e88892ee76960d2488cbe0832bb12c.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/21e8ba073fc3859fcab406e3e17ea13967e88892ee76960d2488cbe0832bb12c.jpg)

![32b411ccf9b0d56de8146dc68e5b8c4ac4db8218581e743ecfa12ec2f57bdff7.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/32b411ccf9b0d56de8146dc68e5b8c4ac4db8218581e743ecfa12ec2f57bdff7.jpg)

![374afb68585f0397b66784d93d9dcf404c61301ec4a43fbfc953cba2728c29e7.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/374afb68585f0397b66784d93d9dcf404c61301ec4a43fbfc953cba2728c29e7.jpg)

![42dc0c31d49e16e2a71fb0e86e4d7666b9151a8ea04625cf50a04a78ce4d8d62.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/42dc0c31d49e16e2a71fb0e86e4d7666b9151a8ea04625cf50a04a78ce4d8d62.jpg)

![9625e5263e32ebb20cf2b55a297d0235010f49b73eff96fe206ec75fc3cc22e4.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/9625e5263e32ebb20cf2b55a297d0235010f49b73eff96fe206ec75fc3cc22e4.jpg)

![c27640f25801a41e76e632c8e154ef2eec73a31c7da6385a8db9929b49e31735.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/c27640f25801a41e76e632c8e154ef2eec73a31c7da6385a8db9929b49e31735.jpg)

![ce55c38c177c2ddfe6ac579fe63846545c19e14c45a43f3b4978229057a6f324.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/ce55c38c177c2ddfe6ac579fe63846545c19e14c45a43f3b4978229057a6f324.jpg)

![d52aa77015207c2a5e787548007a743c92dbaabe4154fee17f65249cb6edbd84.jpg](../icml_results/5491_Improving%20Sampling%20for%20Masked%20Diffusion%20Models%20via%20Information%20Gain/tables/d52aa77015207c2a5e787548007a743c92dbaabe4154fee17f65249cb6edbd84.jpg)

<a id="paper-5444"></a>

## Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design


### Images

![45102aedd8a1a95496e0404c09dc8b1adfb4d649fe85e76c29404e38e816066e.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/45102aedd8a1a95496e0404c09dc8b1adfb4d649fe85e76c29404e38e816066e.jpg)

![56628d24708926b7fee47cbfd0a5a84b1cd7653fe0b39f8eb5f47e562dac721b.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/56628d24708926b7fee47cbfd0a5a84b1cd7653fe0b39f8eb5f47e562dac721b.jpg)

![5732f6a7f22442d4d72f4a45c5e430df4d2757d7654fcb26ea1e8fbf7d340697.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/5732f6a7f22442d4d72f4a45c5e430df4d2757d7654fcb26ea1e8fbf7d340697.jpg)

![5c8e5da96c46e5c2539239f2ecc6201cf0ea4e8c88c1b5ad9c6c5d25aa3529f0.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/5c8e5da96c46e5c2539239f2ecc6201cf0ea4e8c88c1b5ad9c6c5d25aa3529f0.jpg)

![6738aa7e9866b494ecd812c8f539d45667ece4cac410c7a5decaf9a5281ecdd7.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/6738aa7e9866b494ecd812c8f539d45667ece4cac410c7a5decaf9a5281ecdd7.jpg)

![6e3941e78067c4650d094484ffe0bfdecc64740287bab908b0e2f3f4c64aab8f.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/6e3941e78067c4650d094484ffe0bfdecc64740287bab908b0e2f3f4c64aab8f.jpg)

![9941ea9212fed2761e050779e5abbfb82503186c877c58e9896a8b09863d20f0.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/9941ea9212fed2761e050779e5abbfb82503186c877c58e9896a8b09863d20f0.jpg)

![ac3528687613cf4aec34073c8cc32e1fd8eee88e3c7ca930002c5f7452b69bde.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/ac3528687613cf4aec34073c8cc32e1fd8eee88e3c7ca930002c5f7452b69bde.jpg)

![b6fc0bd82db6d38a70e9810b879ec6b8dd7259492e763f111c5626a352c8af0e.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/b6fc0bd82db6d38a70e9810b879ec6b8dd7259492e763f111c5626a352c8af0e.jpg)

![c245af44daf11f52d650a0d2280330f71912dc6ffab0279369e3166fa3bcd58d.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/c245af44daf11f52d650a0d2280330f71912dc6ffab0279369e3166fa3bcd58d.jpg)

![d5607431a3ad237c62be5fcef6126b9d8f64e7a1e6640c573e0fc601803914d2.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/d5607431a3ad237c62be5fcef6126b9d8f64e7a1e6640c573e0fc601803914d2.jpg)

![e2a19545b7640854730ff45f4c3007bc271dcd1bae40494a975b139ad2f70713.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/e2a19545b7640854730ff45f4c3007bc271dcd1bae40494a975b139ad2f70713.jpg)

![fd6bc86588e3caa54b96dbbc7ae1676b826cdfa29c9211309c21bdd838e0fdab.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/images/fd6bc86588e3caa54b96dbbc7ae1676b826cdfa29c9211309c21bdd838e0fdab.jpg)

### Tables

![06d6872b6ba6093400a937766212c3e75eec9824cb3f3322f5a67839e6df6e5b.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/06d6872b6ba6093400a937766212c3e75eec9824cb3f3322f5a67839e6df6e5b.jpg)

![0ae3649e88393d37da853cebff122478a98abd673bd7911ccc6d5d80c697c9df.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/0ae3649e88393d37da853cebff122478a98abd673bd7911ccc6d5d80c697c9df.jpg)

![1a4573994d179e986d562d0aa49d060fd64a671a30ca9931d0556414d1f32a96.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/1a4573994d179e986d562d0aa49d060fd64a671a30ca9931d0556414d1f32a96.jpg)

![2e48fac32910a2f1700c2eaf8f996663d0a8c50b8e96a1f16a830e95f190ad55.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/2e48fac32910a2f1700c2eaf8f996663d0a8c50b8e96a1f16a830e95f190ad55.jpg)

![30376fc9dccccfe0037e6be8c04c3cef3a9097fe24ae39244a5580ce64898249.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/30376fc9dccccfe0037e6be8c04c3cef3a9097fe24ae39244a5580ce64898249.jpg)

![30e9d9f075484a95a74d8c3384b99f73ffcc55c0ab1ddd9b6afcebd5104ccd8b.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/30e9d9f075484a95a74d8c3384b99f73ffcc55c0ab1ddd9b6afcebd5104ccd8b.jpg)

![5ad818677f506f5993a3e9210f79290a8c1a23085b2106c24c4daddbb25dd57c.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/5ad818677f506f5993a3e9210f79290a8c1a23085b2106c24c4daddbb25dd57c.jpg)

![7fa1c77ea6e903c8fcca39847b214e63b9af1c2f8e04822020e7c492cd50988d.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/7fa1c77ea6e903c8fcca39847b214e63b9af1c2f8e04822020e7c492cd50988d.jpg)

![d2ed6bb4b5a279f89b6d434beb5f8cd25f238f4756a8b8f7ece9037d0f7b652e.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/d2ed6bb4b5a279f89b6d434beb5f8cd25f238f4756a8b8f7ece9037d0f7b652e.jpg)

![d510623eb90f3c331ef1312fd29164f03d3d22195c35ebaaffef6bca53118222.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/d510623eb90f3c331ef1312fd29164f03d3d22195c35ebaaffef6bca53118222.jpg)

![de6999cd44e0062054c0b8d29e1681bdae3a05ebd15511e7760ef6ac75b4ba4e.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/de6999cd44e0062054c0b8d29e1681bdae3a05ebd15511e7760ef6ac75b4ba4e.jpg)

![defffdf8c976dd2eedd6a5bcade34006421a61e07cd5fcf0c736ec4e2dbf1d41.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/defffdf8c976dd2eedd6a5bcade34006421a61e07cd5fcf0c736ec4e2dbf1d41.jpg)

![e35a207372b54725a1bdc988905bdb95cf8bcefe00b6b48e40fc3f812416a35b.jpg](../icml_results/5492_Cross-Chirality%20Generalization%20by%20Axial%20Vectors%20for%20Hetero-Chiral%20Protein-Peptide%20Interaction%20Design/tables/e35a207372b54725a1bdc988905bdb95cf8bcefe00b6b48e40fc3f812416a35b.jpg)

<a id="paper-5445"></a>

## Accelerating Langevin Monte Carlo via Efficient Stochastic Runge-Kutta Methods beyond Log-Concavity

### Images

![1c7fc7016ebe2abda7b328f92bfdb073af1ce4c0e3fe6da2edf9b3c603cfe932.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/1c7fc7016ebe2abda7b328f92bfdb073af1ce4c0e3fe6da2edf9b3c603cfe932.jpg)

![ae63d09afff8d70f54c585b92a577cf78942c7d5e06a9f0cd74932e27f04a7a0.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/ae63d09afff8d70f54c585b92a577cf78942c7d5e06a9f0cd74932e27f04a7a0.jpg)

![cd60fb6fe08b36925206668744bbab7a976a14c4776457e1bede47b5b8d6bcf8.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/cd60fb6fe08b36925206668744bbab7a976a14c4776457e1bede47b5b8d6bcf8.jpg)

![dd508242f4e1504ddf18e8d46e3dabc925da3a2cfe1bfe5bfbd717864c25ff34.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/dd508242f4e1504ddf18e8d46e3dabc925da3a2cfe1bfe5bfbd717864c25ff34.jpg)

![ed37c32254e2ae5700f881a0c6324214857d1c4cc0c008c2bf146ef4c93c691d.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/ed37c32254e2ae5700f881a0c6324214857d1c4cc0c008c2bf146ef4c93c691d.jpg)

![f687c5e7ad584cb2c9d30d93edb71b8b9c8f04f683fe078e3ea242e4a69292b6.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/f687c5e7ad584cb2c9d30d93edb71b8b9c8f04f683fe078e3ea242e4a69292b6.jpg)

### Tables

![78f561734c82fcc19f2bfde218c8cb33db19a53bdaf417bbc59369725e90407a.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/tables/78f561734c82fcc19f2bfde218c8cb33db19a53bdaf417bbc59369725e90407a.jpg)
