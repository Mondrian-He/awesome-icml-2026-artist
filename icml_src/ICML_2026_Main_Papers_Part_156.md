# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 156 of 197

## 目录 (Table of Contents)

1. [The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](#paper-5116)
2. [Beyond Literal Translation: Evaluating Cultural Effectiveness in Social Media UGC](#paper-5117)
3. [A Game-Theoretic Analysis of Attacks on Large Language Models via Compositional Skills](#paper-5118)
4. [CHESS: Chebyshev Spectral Synthesis for Trajectory Condensation](#paper-5119)
5. [Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language](#paper-5120)
6. [Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation](#paper-5121)
7. [BrainJanus: A Unified Model for Understanding and Generation across Brain, Vision, and Language](#paper-5122)
8. [What Makes a Strong Model? A Unified Spectral Analysis of Knowledge Transfer over High-dimensional Linear Regression](#paper-5123)
9. [HieraScaffold: Learning Compact Hierarchical Representations for Scalable 4D LiDAR Generation](#paper-5124)
10. [VividCam: Learning Unconventional Camera Motions from Virtual Synthetic Videos](#paper-5125)
11. [LagLLM: LLM-empowered lead–lag dependency learning for spatial-temporal time series forecasting](#paper-5126)
12. [Rank-guided Diffusion for Noise Few-Shot Learning](#paper-5127)
13. [rePIRL: Learn PRM with Inverse RL for LLM Reasoning](#paper-5128)
14. [CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](#paper-5129)
15. [Hierarchical Goal Abstractions via Learned Subset Relations](#paper-5130)
16. [Q-Tab: Quantized Tabular Data Generator](#paper-5131)
17. [Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting](#paper-5132)
18. [Video-BCI: Bayesian Cognitive Integration of Self-Prior Hypotheses for Video Understanding](#paper-5133)
19. [Does Reinforcement Fine-Tuning Improve Generalization of LLM Agents? An Empirical Study](#paper-5134)
20. [Awakening Visual Reasoning: Mitigating Post-Training Failure in Vision-Text Compression](#paper-5135)
21. [LoSA: Locality Aware Sparse Attention in Diffusion Language Models](#paper-5136)
22. [Large Vision-Language Models Get Lost in Attention](#paper-5137)
23. [Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](#paper-5138)
24. [Spectral Bridge Variational Inference: Dynamic LoRA via Bures-Wasserstein Gradient Flows](#paper-5139)
25. [Do Neural Operators Forget Geometry? The Forgetting Hypothesis in Deep Operator Learning](#paper-5140)
26. [Factored Classifier-Free Guidance](#paper-5141)
27. [EchoAttention: Exploiting Token-Pair Redundancy and Frame-Block Similarity for Efficient Video Generation](#paper-5142)
28. [BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction](#paper-5143)
29. [MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts](#paper-5144)
30. [LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](#paper-5145)
31. [DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery](#paper-5146)
32. [Towards Scalable and Consistent 3D Editing](#paper-5147)
33. [Boost the Identity-Preserving Embedding for Consistent Visual Generation](#paper-5148)

---


<a id="paper-5116"></a>

## The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning

### Images

![13ba2949439438d76e22733748a22c3f3b4f698da517f729e2cf086979cecd76.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/13ba2949439438d76e22733748a22c3f3b4f698da517f729e2cf086979cecd76.jpg)

![19db39eb5a6984901a020105caf99019b90d43d3aca5dd1640b7cda380b46671.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/19db39eb5a6984901a020105caf99019b90d43d3aca5dd1640b7cda380b46671.jpg)

![37e991c2a78ff077ef94ea01cb7dc988a87177dd31870a6258a3a8362b957c1d.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/37e991c2a78ff077ef94ea01cb7dc988a87177dd31870a6258a3a8362b957c1d.jpg)

![590bf80b8adcf37ac8af29a0ea02131f20f560ac899b4a60a1d9f21a518d42fc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/590bf80b8adcf37ac8af29a0ea02131f20f560ac899b4a60a1d9f21a518d42fc.jpg)

![811798c2fe390dbbd8271e27d7bf49444ab8a59fbfc26563be1460932fa49e9d.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/811798c2fe390dbbd8271e27d7bf49444ab8a59fbfc26563be1460932fa49e9d.jpg)

![a01222e84970b3749c6f54f1a659e83b7d24a5d4438b79825eafd7b307edebfd.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/a01222e84970b3749c6f54f1a659e83b7d24a5d4438b79825eafd7b307edebfd.jpg)

![aa2813e33fc8b4a842e3e6eda568664f9072a479e5a42802424c3da7781fb49a.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/aa2813e33fc8b4a842e3e6eda568664f9072a479e5a42802424c3da7781fb49a.jpg)

![e6afeae01250477053b083b11bffc2665f0b9148bc9a7cf0e1ceb8bde31481fd.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/e6afeae01250477053b083b11bffc2665f0b9148bc9a7cf0e1ceb8bde31481fd.jpg)

### Tables

![04e20483395c0382f59a660d0b5234b81f3f0482012b24cbd052d6338355f8df.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/04e20483395c0382f59a660d0b5234b81f3f0482012b24cbd052d6338355f8df.jpg)

![079db83692a70432d784cd74a5533d9893fcfc618c76299452f6c82c0751afdc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/079db83692a70432d784cd74a5533d9893fcfc618c76299452f6c82c0751afdc.jpg)

![0ab89662e1812f020f915551abf838653bd564d43e034a94ace292a3b65d44b6.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/0ab89662e1812f020f915551abf838653bd564d43e034a94ace292a3b65d44b6.jpg)

![5890cf8dd0d47fb0cdb5acfaf6ca3301cfec55512d0678e6103f43acee11922b.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/5890cf8dd0d47fb0cdb5acfaf6ca3301cfec55512d0678e6103f43acee11922b.jpg)

![a39b90a9a9cd81faac07cf3693b3ce143988b3aec82a038b3c04386820e65765.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/a39b90a9a9cd81faac07cf3693b3ce143988b3aec82a038b3c04386820e65765.jpg)

![b02793e86bd3da326a6b8b2d21cc0df4cac88785df483776344e1f5ef89b84c2.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/b02793e86bd3da326a6b8b2d21cc0df4cac88785df483776344e1f5ef89b84c2.jpg)

![bde83d168891eef800fecf6f3ac05f0b4f9ede1ad7f87fa76aa261be1d9b5fd1.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/bde83d168891eef800fecf6f3ac05f0b4f9ede1ad7f87fa76aa261be1d9b5fd1.jpg)

![c3e0f29b0ced5771a6828b353786e761ebb08efd003ecd05e3f1c2cbbeebbcfc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/c3e0f29b0ced5771a6828b353786e761ebb08efd003ecd05e3f1c2cbbeebbcfc.jpg)

![f2e7b70efa1f70ab9875556683bc9cb174c988032d902b03cc3cae9c344d9133.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/f2e7b70efa1f70ab9875556683bc9cb174c988032d902b03cc3cae9c344d9133.jpg)

![f5e92f590223fe7360135f65e12328148598f953393da1b8c123ddfe540682cf.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/f5e92f590223fe7360135f65e12328148598f953393da1b8c123ddfe540682cf.jpg)

![feff04e64ff625ed10012e4a4b6d4fd8a1eb7b97e48a7682b3008fc054179336.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/feff04e64ff625ed10012e4a4b6d4fd8a1eb7b97e48a7682b3008fc054179336.jpg)

## The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning


### Images

![08f817cde3a1c842b01a12e7f734140c6153c756b7e1e016cd0eaf089286f170.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/images/08f817cde3a1c842b01a12e7f734140c6153c756b7e1e016cd0eaf089286f170.jpg)

![4241244d32ce1ee851a047f85e36117298c66cb4a5f04606d930a1e17dbd58e7.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/images/4241244d32ce1ee851a047f85e36117298c66cb4a5f04606d930a1e17dbd58e7.jpg)

![a565468c0bb1b20494ebcee422f86126e8c63abd0a698089f855b95d60306c2e.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/images/a565468c0bb1b20494ebcee422f86126e8c63abd0a698089f855b95d60306c2e.jpg)

### Tables

![3088b3828665790cdf2f94e46918e20da4a9a3f773fc3ba47cc0af01c81219d0.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/tables/3088b3828665790cdf2f94e46918e20da4a9a3f773fc3ba47cc0af01c81219d0.jpg)

![3c99d378126a28cc6536a5302d5f5abfa7248722b8c9b94cb97437531d663257.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/tables/3c99d378126a28cc6536a5302d5f5abfa7248722b8c9b94cb97437531d663257.jpg)

![4457298acb5aa085925b8ebe1b48fe3ce49a89bfc45d3e7fe0d66acaeec7309a.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/tables/4457298acb5aa085925b8ebe1b48fe3ce49a89bfc45d3e7fe0d66acaeec7309a.jpg)

![766efb7828f04123c1042034981ee60c8b64aee90a8bd26a113bddb0d7aa0c4b.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/tables/766efb7828f04123c1042034981ee60c8b64aee90a8bd26a113bddb0d7aa0c4b.jpg)

![7b8e8c088f2931c103cd622f2d1bfea23fc68a7b3eda1d8177e9112846b50d40.jpg](../icml_results/5164_The%20Silent%20Thought_%20Modeling%20Internal%20Cognition%20in%20Full-Duplex%20Spoken%20Dialogue%20Models%20via%20Latent%20Rea/tables/7b8e8c088f2931c103cd622f2d1bfea23fc68a7b3eda1d8177e9112846b50d40.jpg)

<a id="paper-5117"></a>

## Beyond Literal Translation: Evaluating Cultural Effectiveness in Social Media UGC


### Images

![04f89c4d7342bcf6d4b7a5a4ac19015f08208e08c349b2405ff1f3ed68561025.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/04f89c4d7342bcf6d4b7a5a4ac19015f08208e08c349b2405ff1f3ed68561025.jpg)

![3db7e7808c39e87050b508f2e8281208d00d8ba476a91d190b0a6723ccd1e885.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/3db7e7808c39e87050b508f2e8281208d00d8ba476a91d190b0a6723ccd1e885.jpg)

![7ce8e6a41500e68ffb9419601905a2463046d535b5f1c275d2483c1d7c7bc4a8.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/7ce8e6a41500e68ffb9419601905a2463046d535b5f1c275d2483c1d7c7bc4a8.jpg)

![7db8fa93e3b05e96debffa4d1259a4b7505556257d5cf0b660d39ec92a5613b2.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/7db8fa93e3b05e96debffa4d1259a4b7505556257d5cf0b660d39ec92a5613b2.jpg)

![9af6718051d3958e765d5a33dea30ad500535c90374a28252ee6f3981cc3a015.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/9af6718051d3958e765d5a33dea30ad500535c90374a28252ee6f3981cc3a015.jpg)

![aad0e67ca2e86dae37f51671082338e944a11b3cc2d5944f9837a72ac4a6893e.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/aad0e67ca2e86dae37f51671082338e944a11b3cc2d5944f9837a72ac4a6893e.jpg)

![b4b5ca6fdb140590fc8e53d7df8029944289d11e061faa2184e06a2c735735e6.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/b4b5ca6fdb140590fc8e53d7df8029944289d11e061faa2184e06a2c735735e6.jpg)

![c145fa798e7e3c4cf9d8b699fd81dc9b1c564f7b6e04b909976b40ba63f6c5bd.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/c145fa798e7e3c4cf9d8b699fd81dc9b1c564f7b6e04b909976b40ba63f6c5bd.jpg)

![c2d2c5a45b71aa884920a756a599bbb553e0a68b16e043b75fed55d8b950908c.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/c2d2c5a45b71aa884920a756a599bbb553e0a68b16e043b75fed55d8b950908c.jpg)

![eac6ebd3529a5b8a13018b3a41be3940482de54b9205ea572e28dfac5ccca120.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/images/eac6ebd3529a5b8a13018b3a41be3940482de54b9205ea572e28dfac5ccca120.jpg)

### Tables

![07dd646ea8f1b1cdd4dd54e2d8174bc164c67121149d07f322b2565001e26dc0.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/07dd646ea8f1b1cdd4dd54e2d8174bc164c67121149d07f322b2565001e26dc0.jpg)

![0b77e2c31713a1f5eb05bf61c00271135d8e708c15f3f8bcdbba92bc860bd8b2.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/0b77e2c31713a1f5eb05bf61c00271135d8e708c15f3f8bcdbba92bc860bd8b2.jpg)

![3a317e3cd82ae8b0421154f1403f887d38566a4b463fca2c8ae3c8a9edf6c64f.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/3a317e3cd82ae8b0421154f1403f887d38566a4b463fca2c8ae3c8a9edf6c64f.jpg)

![498395c178acf0883ee463d13a40c1b0f08d075ffbdee36492d3f2a6bd697111.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/498395c178acf0883ee463d13a40c1b0f08d075ffbdee36492d3f2a6bd697111.jpg)

![5c5df766068858679622450668a0ae08e16c6ce4cef5ec12ede0e8ce72128e08.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/5c5df766068858679622450668a0ae08e16c6ce4cef5ec12ede0e8ce72128e08.jpg)

![62668f0c3c851375ca17932f40521d82130b717c86c37b2c418d72da8558639c.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/62668f0c3c851375ca17932f40521d82130b717c86c37b2c418d72da8558639c.jpg)

![6840a5b0a146f8784c25f8e67b5086a7719b124df775271fe7e5d38c2069265d.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/6840a5b0a146f8784c25f8e67b5086a7719b124df775271fe7e5d38c2069265d.jpg)

![d87d4946d79f0776096b1e4d9125f31e886c241164a585429b5df3675629ec02.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/d87d4946d79f0776096b1e4d9125f31e886c241164a585429b5df3675629ec02.jpg)

![e594371236b2c6db7d0d2583d06b6e5de4e2ee7f2681870e59b30b36cb861d1b.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/e594371236b2c6db7d0d2583d06b6e5de4e2ee7f2681870e59b30b36cb861d1b.jpg)

![f6ba66111d78a188d49437439314a6b0095282c3e75fa03b23eb9181b9afcec4.jpg](../icml_results/5165_Beyond%20Literal%20Translation_%20Evaluating%20Cultural%20Effectiveness%20in%20Social%20Media%20UGC/tables/f6ba66111d78a188d49437439314a6b0095282c3e75fa03b23eb9181b9afcec4.jpg)

<a id="paper-5118"></a>

## A Game-Theoretic Analysis of Attacks on Large Language Models via Compositional Skills


### Images

![01aa14bc6388f27813657d27f720a915b67dcdc6fa96493620eedd90ef110849.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/01aa14bc6388f27813657d27f720a915b67dcdc6fa96493620eedd90ef110849.jpg)

![49603cb13b8dc8068e1676929a79fcc4d0e078c16875e1cb4d8792020a840aa3.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/49603cb13b8dc8068e1676929a79fcc4d0e078c16875e1cb4d8792020a840aa3.jpg)

![4cbd07cccc6b342486fc4ce8ad24e79d3ede04e6e01dcd8f035266c3b6f50796.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/4cbd07cccc6b342486fc4ce8ad24e79d3ede04e6e01dcd8f035266c3b6f50796.jpg)

![5ff547c8d0980412a8f90901941b718baf862ded5c380f2937afb1e0651cfa1d.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/5ff547c8d0980412a8f90901941b718baf862ded5c380f2937afb1e0651cfa1d.jpg)

![76173b02da4111852db849cfe7a334c4667e9425cad864321cd11c832c5ec0bc.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/76173b02da4111852db849cfe7a334c4667e9425cad864321cd11c832c5ec0bc.jpg)

![bb994685cfd211eece24cc9844985355e9d038b4e7e5bbd73dad2ea0aa108ede.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/bb994685cfd211eece24cc9844985355e9d038b4e7e5bbd73dad2ea0aa108ede.jpg)

![bd5f5d0ad295ded64ede5886c165af9bce4522f47ba7666c358f511d9b3be175.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/bd5f5d0ad295ded64ede5886c165af9bce4522f47ba7666c358f511d9b3be175.jpg)

![c5c34538f21ac4ba7e2119f4398f8c98ee2703dec79f3e7ad7ebc7c1527bb575.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/images/c5c34538f21ac4ba7e2119f4398f8c98ee2703dec79f3e7ad7ebc7c1527bb575.jpg)

### Tables

![29152d2901d8c51b2967e764dbbe1276dfdbce957f8c5ce7775c75403c3b4b1e.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/29152d2901d8c51b2967e764dbbe1276dfdbce957f8c5ce7775c75403c3b4b1e.jpg)

![2d818eebf51f5e0ceb722b078de5c20432404dc8a0c97a6415587aa1e4eea7fd.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/2d818eebf51f5e0ceb722b078de5c20432404dc8a0c97a6415587aa1e4eea7fd.jpg)

![343afd7bad5eb2657d7dab978ee269d1ffa987f0ac10f394a09b20a7e87aded7.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/343afd7bad5eb2657d7dab978ee269d1ffa987f0ac10f394a09b20a7e87aded7.jpg)

![370bdd52e63152606c798c38de3c607272eb3e404fb0760a85adad1318c87bf7.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/370bdd52e63152606c798c38de3c607272eb3e404fb0760a85adad1318c87bf7.jpg)

![4118b500369d7b8b24ddee68d95a246f298193b53b1e0a98af1bbdd243d65856.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/4118b500369d7b8b24ddee68d95a246f298193b53b1e0a98af1bbdd243d65856.jpg)

![572909e4d08555a4a79f5b74113e6a013c1fa33a755d1c17fdabf19d2ebab396.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/572909e4d08555a4a79f5b74113e6a013c1fa33a755d1c17fdabf19d2ebab396.jpg)

![962541d20d93e0cb7c1d88e83f5cb8ed6c408ef5e681f0e7887658d331d59385.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/962541d20d93e0cb7c1d88e83f5cb8ed6c408ef5e681f0e7887658d331d59385.jpg)

![bbda85b0eaa1835570267cd46be6044a24da86dae3b7ebb48e7d9d2cb0e7efeb.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/bbda85b0eaa1835570267cd46be6044a24da86dae3b7ebb48e7d9d2cb0e7efeb.jpg)

![c6a2395b1c82fe35d92fbff4ea3a0a5e8cddf36d44aba181472db937953f52fa.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/c6a2395b1c82fe35d92fbff4ea3a0a5e8cddf36d44aba181472db937953f52fa.jpg)

![d7cfab046e482667f099d83241a18dd54cdc11d334350b187cf3efba952755eb.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/d7cfab046e482667f099d83241a18dd54cdc11d334350b187cf3efba952755eb.jpg)

![df0c3f445b7964caffedb7708d24a40ff3104e37002707d5bc85eb7d30004a2c.jpg](../icml_results/5166_A%20Game-Theoretic%20Analysis%20of%20Attacks%20on%20Large%20Language%20Models%20via%20Compositional%20Skills/tables/df0c3f445b7964caffedb7708d24a40ff3104e37002707d5bc85eb7d30004a2c.jpg)

<a id="paper-5119"></a>

## CHESS: Chebyshev Spectral Synthesis for Trajectory Condensation


### Images

![139bf79ec0d16deaa5f463e9e0f8842fed7c93bf78c9d5f3d03080a18f9492b6.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/139bf79ec0d16deaa5f463e9e0f8842fed7c93bf78c9d5f3d03080a18f9492b6.jpg)

![1c8785f882e057f16cc94fd9976c5feece32d1fa4050a55924c3b9a31e7367ca.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/1c8785f882e057f16cc94fd9976c5feece32d1fa4050a55924c3b9a31e7367ca.jpg)

![42075c88109d42adb0b79008dadc61a3da57fd9f2575525700f3ca832622fec2.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/42075c88109d42adb0b79008dadc61a3da57fd9f2575525700f3ca832622fec2.jpg)

![46a81e8ec3304316a3d1d616c1263e9a852215fc43bffd38b8e0c22145624b01.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/46a81e8ec3304316a3d1d616c1263e9a852215fc43bffd38b8e0c22145624b01.jpg)

![49738ef352586754c1735f0a596bab4357f7fe9e504f9c5e7209786e4e4b36fb.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/49738ef352586754c1735f0a596bab4357f7fe9e504f9c5e7209786e4e4b36fb.jpg)

![4fb6329bc1542a7b8ad577e5315a3057f82c038fdfb67ff4e32306359304710b.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/4fb6329bc1542a7b8ad577e5315a3057f82c038fdfb67ff4e32306359304710b.jpg)

![55e799cd8348d67a23c74cb11a96f24551f7a63dbe79385fd66b681c5819c9ed.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/55e799cd8348d67a23c74cb11a96f24551f7a63dbe79385fd66b681c5819c9ed.jpg)

![5e46a2dc2e95afe8baf84aabc4e2f48bb027157da3d461e7d6a501dc74b29aba.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/5e46a2dc2e95afe8baf84aabc4e2f48bb027157da3d461e7d6a501dc74b29aba.jpg)

![6fe83b01e366986868863d731a38c80ed42e26f7643b9dec3b36fc6cbed0c726.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/6fe83b01e366986868863d731a38c80ed42e26f7643b9dec3b36fc6cbed0c726.jpg)

![74920bd23e95068a2051525b9218c81068eda15d36585d713fd7766b2685438f.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/74920bd23e95068a2051525b9218c81068eda15d36585d713fd7766b2685438f.jpg)

![7b051697b36b5747b66fab9f57d3cca449bf6574c0c7d3ae20e43f97473a64ac.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/7b051697b36b5747b66fab9f57d3cca449bf6574c0c7d3ae20e43f97473a64ac.jpg)

![8b0c35e6bc080765abd803eb434176e8edf45790b7cb42f44be09eaac429fb0a.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/8b0c35e6bc080765abd803eb434176e8edf45790b7cb42f44be09eaac429fb0a.jpg)

![8f0e8b7d6695de6230e477e64d704d0694120780b664dbc75ad66e1381c7ae2b.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/8f0e8b7d6695de6230e477e64d704d0694120780b664dbc75ad66e1381c7ae2b.jpg)

![918b7cc33c489aa4185036bb4c185e2a449a2d3f363b58c0f8aa276b34102285.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/918b7cc33c489aa4185036bb4c185e2a449a2d3f363b58c0f8aa276b34102285.jpg)

![9b3e5512c8353f4169ac805ec96afdf618112025c0ea4411d8b5eb29f339f42d.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/9b3e5512c8353f4169ac805ec96afdf618112025c0ea4411d8b5eb29f339f42d.jpg)

![ad6d975270e2f7fcf3004ca0fcece8d639b8306cbef7570fd033f776805f9131.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/ad6d975270e2f7fcf3004ca0fcece8d639b8306cbef7570fd033f776805f9131.jpg)

![b4e1f9b64b3d48a8e8bec86dd08135c25cfee86f2102c7d61066004a578fdda9.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/b4e1f9b64b3d48a8e8bec86dd08135c25cfee86f2102c7d61066004a578fdda9.jpg)

![b7f906bf2381a7a0a598fc011351e973a37cdef947025d88af953351ba46dddb.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/b7f906bf2381a7a0a598fc011351e973a37cdef947025d88af953351ba46dddb.jpg)

![c436d6a7587f55e200bdedc4f788a93365a5cd7815351a1a72b957b430bdc9ed.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/c436d6a7587f55e200bdedc4f788a93365a5cd7815351a1a72b957b430bdc9ed.jpg)

![c498e875d71220aa42b63659e13719d6e98edb1d2e05fe26b8bc0b4ead08a171.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/c498e875d71220aa42b63659e13719d6e98edb1d2e05fe26b8bc0b4ead08a171.jpg)

![dd70dba4a358ac4efcf4efd96cdc8c7dce274ac8381a8a336d36d2c646fb88df.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/dd70dba4a358ac4efcf4efd96cdc8c7dce274ac8381a8a336d36d2c646fb88df.jpg)

![df2d7b8c12302e85ceec2d6209f6a16c37a8c4f94f7d9de2aee79355a2f4a0ba.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/df2d7b8c12302e85ceec2d6209f6a16c37a8c4f94f7d9de2aee79355a2f4a0ba.jpg)

![e24ca6363b755daab4d8f0356db478d78e9b35deb5cfb1445f1b98e2d2f50167.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/e24ca6363b755daab4d8f0356db478d78e9b35deb5cfb1445f1b98e2d2f50167.jpg)

![e3dfd1353f0a4ec5793b240e9c050f8ea75b296a0326a7bc9dc6a8d5df2fe214.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/e3dfd1353f0a4ec5793b240e9c050f8ea75b296a0326a7bc9dc6a8d5df2fe214.jpg)

![e4900bc10ede6454f88438dfb6fc719295abfe8d10d927f298e3f3630162a1b3.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/e4900bc10ede6454f88438dfb6fc719295abfe8d10d927f298e3f3630162a1b3.jpg)

![ecb2924d2711711b9f0d484ab9e71ccac09fdc2c0c1d4b384c6291ba90068a9f.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/ecb2924d2711711b9f0d484ab9e71ccac09fdc2c0c1d4b384c6291ba90068a9f.jpg)

![f3eb3bdeea5b0ac9b2d7b414053c4defaab9d324a435143eaaf1674fdf387ff3.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/f3eb3bdeea5b0ac9b2d7b414053c4defaab9d324a435143eaaf1674fdf387ff3.jpg)

![ff5b5d291d06002776a68f739481eea108a3b26786acd6582506c64b0d7b7125.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/images/ff5b5d291d06002776a68f739481eea108a3b26786acd6582506c64b0d7b7125.jpg)

### Tables

![029a4affc62f9b2e79bd1047af5e56fc3ca29b31a4df308ffd18dfe9759bb761.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/029a4affc62f9b2e79bd1047af5e56fc3ca29b31a4df308ffd18dfe9759bb761.jpg)

![2b0e6317b00ba1400455e27d52af71101dceb7f08e6b4ef0db1c08096b0f95f4.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/2b0e6317b00ba1400455e27d52af71101dceb7f08e6b4ef0db1c08096b0f95f4.jpg)

![33a33c07d6762364182f2c45757b609512816b12fe86fee930b006e3f20e379e.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/33a33c07d6762364182f2c45757b609512816b12fe86fee930b006e3f20e379e.jpg)

![3cd0f8740439a188302999a91751aea5c572589ea7127923cd4b89460bf6fb24.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/3cd0f8740439a188302999a91751aea5c572589ea7127923cd4b89460bf6fb24.jpg)

![8e6a161c273010796d5e8bae5a080a4e62bc784b65b56af3e8c4c8177ea7c425.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/8e6a161c273010796d5e8bae5a080a4e62bc784b65b56af3e8c4c8177ea7c425.jpg)

![91a9e06c8f647594c902dab1903bb240d8c3b7a818765735e5ee3d1c0eea5376.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/91a9e06c8f647594c902dab1903bb240d8c3b7a818765735e5ee3d1c0eea5376.jpg)

![9594abd786a0fbe25ecf5e7b4b7a9bf68ba7c8abb1ee51074febbaa2d86853a0.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/9594abd786a0fbe25ecf5e7b4b7a9bf68ba7c8abb1ee51074febbaa2d86853a0.jpg)

![a07eb9c963b7ab8dbeb305abcee0d8d6ea33d438b2b75ecd60e04623bb6957a5.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/a07eb9c963b7ab8dbeb305abcee0d8d6ea33d438b2b75ecd60e04623bb6957a5.jpg)

![aba000de2d2cacab549639b4fee7ce302615f4d186c133f8cdc52582bedef518.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/aba000de2d2cacab549639b4fee7ce302615f4d186c133f8cdc52582bedef518.jpg)

![c17725e415303d72811bd4f46e51470c0b57067f8dcd8ad2d7d2288d5cf1a7e4.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/c17725e415303d72811bd4f46e51470c0b57067f8dcd8ad2d7d2288d5cf1a7e4.jpg)

![d5012cabe2e818e7f2348bd5c686fcb19d25c665e0ffae6a897851b0a82353ec.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/d5012cabe2e818e7f2348bd5c686fcb19d25c665e0ffae6a897851b0a82353ec.jpg)

![e021af7a26ee3390f296b1c5adb74e1cd62f7cd46c48a5d94986eb015116bedf.jpg](../icml_results/5167_CHESS_%20Chebyshev%20Spectral%20Synthesis%20for%20Trajectory%20Condensation/tables/e021af7a26ee3390f296b1c5adb74e1cd62f7cd46c48a5d94986eb015116bedf.jpg)

<a id="paper-5120"></a>

## Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language


### Images

![524d7b2563a4a818e7838c4b7737ebcc3930de6185966621167b45a895ba6630.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/524d7b2563a4a818e7838c4b7737ebcc3930de6185966621167b45a895ba6630.jpg)

![5e57332c3b99baaa22dc9e0a3c11f4baf34ad436a6b1b014d12d4f171e785858.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/5e57332c3b99baaa22dc9e0a3c11f4baf34ad436a6b1b014d12d4f171e785858.jpg)

![6f8c3b86915573c98284d0ae2e4be3ecf131296df8e93ad9a303f9075e4b55ea.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/6f8c3b86915573c98284d0ae2e4be3ecf131296df8e93ad9a303f9075e4b55ea.jpg)

![bc47703740ead8a89ae984acb514bb2fdbc39f46014085a047d56a9db855972c.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/bc47703740ead8a89ae984acb514bb2fdbc39f46014085a047d56a9db855972c.jpg)

![cc186f828210acf261b50f4637a3d1bd68b81cb1714dcce0ef92aafa16d0427f.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/cc186f828210acf261b50f4637a3d1bd68b81cb1714dcce0ef92aafa16d0427f.jpg)

![d490fe90ce067ee5d97b107d682d8f9cde3299024ad21ac4202fe2b44a9b7bb1.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/d490fe90ce067ee5d97b107d682d8f9cde3299024ad21ac4202fe2b44a9b7bb1.jpg)

![eb4cebbcad8c9ec34114b653f0d535ca6ef2a802ea9a6c7f8d701f6f18e8a708.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/images/eb4cebbcad8c9ec34114b653f0d535ca6ef2a802ea9a6c7f8d701f6f18e8a708.jpg)

### Tables

![22a30c500d728934891b77d3465a584939302a03d58c595060822065b82e91ac.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/22a30c500d728934891b77d3465a584939302a03d58c595060822065b82e91ac.jpg)

![4e8be385e5076585e9b92ac558d562b86660b9b00e28ea486ec56ab3a92001b4.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/4e8be385e5076585e9b92ac558d562b86660b9b00e28ea486ec56ab3a92001b4.jpg)

![5aa6d4aaf901e3735a401b981b55c658a37f6e8cd1695b7ce0c76ce0c8e318f6.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/5aa6d4aaf901e3735a401b981b55c658a37f6e8cd1695b7ce0c76ce0c8e318f6.jpg)

![7bd27131db2449f4e476640a1d70d19f549bf28d0d5d2d88451f4a5314bee7af.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/7bd27131db2449f4e476640a1d70d19f549bf28d0d5d2d88451f4a5314bee7af.jpg)

![967e427f1bfc46f1e1dc866686aba6874c9d0b04ec1a8394f9455443be655ce6.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/967e427f1bfc46f1e1dc866686aba6874c9d0b04ec1a8394f9455443be655ce6.jpg)

![983b9d08685f44764467bc3e3279ccc19055e50ba3789e244a0e8dce88044fe8.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/983b9d08685f44764467bc3e3279ccc19055e50ba3789e244a0e8dce88044fe8.jpg)

![a584e497f0e1ea87fcbf2c93fecf963204ecd3288e2d8c58d899d0eddaaf27de.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/a584e497f0e1ea87fcbf2c93fecf963204ecd3288e2d8c58d899d0eddaaf27de.jpg)

![aba373e5e41923847c39d475bcfc2e95670be2e6cf498cfaf636aa32be03b6ee.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/aba373e5e41923847c39d475bcfc2e95670be2e6cf498cfaf636aa32be03b6ee.jpg)

![ad1aa32bfbf03af4866095b62ea568503a51b65ad3108ead9ec22aa6d3390b50.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/ad1aa32bfbf03af4866095b62ea568503a51b65ad3108ead9ec22aa6d3390b50.jpg)

![b99b766f56ecd5e04268d89dd8b6b8bc906f1dbff1f13d13c64ab1568fc06b03.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/b99b766f56ecd5e04268d89dd8b6b8bc906f1dbff1f13d13c64ab1568fc06b03.jpg)

![b9b085d044ff846b741b36f4baa47e571c1142bfca599064cbb72a04a1e98634.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/b9b085d044ff846b741b36f4baa47e571c1142bfca599064cbb72a04a1e98634.jpg)

![d99a969fe545ca391a390171451deb9a0771705c8f1c7e3b634dfefc4888a4e6.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/d99a969fe545ca391a390171451deb9a0771705c8f1c7e3b634dfefc4888a4e6.jpg)

![da9933890081c2238573cb0eb8b7443b5b7c99def26f52107fe7bf4bbd88e033.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/da9933890081c2238573cb0eb8b7443b5b7c99def26f52107fe7bf4bbd88e033.jpg)

![edf40bedfa98177543cc778206688dbfedacc0ae9563f6c954944eb279ed86d7.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/edf40bedfa98177543cc778206688dbfedacc0ae9563f6c954944eb279ed86d7.jpg)

![f59d568d06356e69321cbb92771119128ddc1c3f6556ac2c9d3fe80b71318845.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/f59d568d06356e69321cbb92771119128ddc1c3f6556ac2c9d3fe80b71318845.jpg)

![fb0f97c9cb278ef68d2aea7301738491ae12fd75953d844d58c9f60dc71e6d5c.jpg](../icml_results/5168_Tabero_%20Learning%20Gentle%20Manipulation%20with%20Closed-Loop%20Force%20Feedback%20from%20Vision%2C%20Touch%2C%20and%20Languag/tables/fb0f97c9cb278ef68d2aea7301738491ae12fd75953d844d58c9f60dc71e6d5c.jpg)

<a id="paper-5121"></a>

## Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation


### Images

![29d5839779e0929a4c5d2c80b177a660f3bfe0391403168d259d6a2078697553.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/images/29d5839779e0929a4c5d2c80b177a660f3bfe0391403168d259d6a2078697553.jpg)

![56860d2690d328a166f98275e90970a45bce61f319de2b43b68c7a16f65fd82b.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/images/56860d2690d328a166f98275e90970a45bce61f319de2b43b68c7a16f65fd82b.jpg)

### Tables

![09185a0c04d148eded29a929d943aa5f90240df0604d6d90cf56789415a687ac.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/09185a0c04d148eded29a929d943aa5f90240df0604d6d90cf56789415a687ac.jpg)

![1112c13222eb03a3c6f410c7cc4f293efeb839a4e6d4b40213c7e07915cde9bb.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/1112c13222eb03a3c6f410c7cc4f293efeb839a4e6d4b40213c7e07915cde9bb.jpg)

![1416fbc72d9ca7716d90da75944151bfdc4a73b81fce19b87c5485b96b680e98.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/1416fbc72d9ca7716d90da75944151bfdc4a73b81fce19b87c5485b96b680e98.jpg)

![1d512f9a7da8cc4d41db6c58a34d389174b4e6c5f30f1c3a2c3f70d4c6441076.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/1d512f9a7da8cc4d41db6c58a34d389174b4e6c5f30f1c3a2c3f70d4c6441076.jpg)

![7790ea5b4cba802e08c7d7c316c2302ce3c792bfc4a69d4bf222acb599b0e4d1.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/7790ea5b4cba802e08c7d7c316c2302ce3c792bfc4a69d4bf222acb599b0e4d1.jpg)

![7aeaa06a22ada2a06cc4d653e46eae0b7f6e6869ce72c905638ea6e042ee4aac.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/7aeaa06a22ada2a06cc4d653e46eae0b7f6e6869ce72c905638ea6e042ee4aac.jpg)

![7ebf46154a12d6eddec7ca257bd9bfdee118e65170739388535fb1b7e738f9ac.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/7ebf46154a12d6eddec7ca257bd9bfdee118e65170739388535fb1b7e738f9ac.jpg)

![87d8dff9d63fb2c778e616230bae6be76698d2ad416a27bd7173506f4c691d32.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/87d8dff9d63fb2c778e616230bae6be76698d2ad416a27bd7173506f4c691d32.jpg)

![87dc578a78531fee3b18470c3009a78798452137fc53c21ff49af9705af5c5ba.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/87dc578a78531fee3b18470c3009a78798452137fc53c21ff49af9705af5c5ba.jpg)

![8cab8894b72ebe4626c205b6ca48f9857cba5fc353c6c07d56a14998411a8fbc.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/8cab8894b72ebe4626c205b6ca48f9857cba5fc353c6c07d56a14998411a8fbc.jpg)

![b2790238e9366e2f593688596a1853f4ce7ee098e7dbea60762cfa7525bfb166.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/b2790238e9366e2f593688596a1853f4ce7ee098e7dbea60762cfa7525bfb166.jpg)

![baec030e38fb4103dcbd0f862c1f05be07d3dc640381773bfac3824388fed619.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/baec030e38fb4103dcbd0f862c1f05be07d3dc640381773bfac3824388fed619.jpg)

![bd3176e08b631989332585ec4f1effd8d4ec1e2388fddf6719c2080d1a93f127.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/bd3176e08b631989332585ec4f1effd8d4ec1e2388fddf6719c2080d1a93f127.jpg)

![c05a182c384154d4eb27e8239031b8a6ba50553ab1848fcc3fef4b80fef29a3c.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/c05a182c384154d4eb27e8239031b8a6ba50553ab1848fcc3fef4b80fef29a3c.jpg)

![c0b49cd67637723ec5d0ba6e8cf499669f1d97f528e71caf67b7a0cc4f24f2e0.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/c0b49cd67637723ec5d0ba6e8cf499669f1d97f528e71caf67b7a0cc4f24f2e0.jpg)

![c4efd17727ba653a29a8304018d062b7f6cb48f6e5a091442c1e5d7936cdd871.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/c4efd17727ba653a29a8304018d062b7f6cb48f6e5a091442c1e5d7936cdd871.jpg)

![ce333342967510e32066b1b2c7c39516946a0b2e256a56d151fe0f484b38d1d1.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/ce333342967510e32066b1b2c7c39516946a0b2e256a56d151fe0f484b38d1d1.jpg)

![dc82f125010535e0bca4451dfb33da77661f46a2acc6265caad69f1987bf3cf8.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/dc82f125010535e0bca4451dfb33da77661f46a2acc6265caad69f1987bf3cf8.jpg)

![e2eae0de901063390d6573a48774cd7b30c41b103669d3798edc7435305b25f7.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/e2eae0de901063390d6573a48774cd7b30c41b103669d3798edc7435305b25f7.jpg)

![e3d9e99cb833a1d0207ae08bc108ef66f9a74780830843b6269226342db7c67f.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/e3d9e99cb833a1d0207ae08bc108ef66f9a74780830843b6269226342db7c67f.jpg)

![ff3690736fa262516578245b253c7a166d15f79ec6b284cca61638ec4691caf0.jpg](../icml_results/5169_Securing%20Multi-Agent%20Systems%20Against%20Corruptions%20via%20Node%20Contribution%20Backpropagation/tables/ff3690736fa262516578245b253c7a166d15f79ec6b284cca61638ec4691caf0.jpg)

<a id="paper-5122"></a>

## BrainJanus: A Unified Model for Understanding and Generation across Brain, Vision, and Language


### Images

![061adda711cb26748272915202f603960b6cd78ab0d5891985de3509493af9fa.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/061adda711cb26748272915202f603960b6cd78ab0d5891985de3509493af9fa.jpg)

![3531936deb8708f68b91ead2310caf5dc25149412a9da0d15d11231006157e70.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/3531936deb8708f68b91ead2310caf5dc25149412a9da0d15d11231006157e70.jpg)

![3554d90cf914e48ace38d3b0b40884d8d1bb6c2e221ae1a05328b99dc80f648f.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/3554d90cf914e48ace38d3b0b40884d8d1bb6c2e221ae1a05328b99dc80f648f.jpg)

![380f03cf4f9e125f7553758a4dad637d1e9936b05307768ffe2d20bb1ff7aca8.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/380f03cf4f9e125f7553758a4dad637d1e9936b05307768ffe2d20bb1ff7aca8.jpg)

![55f56f18a7938440d97b78b1886cf7f56bbee18097747eb10fbb803bbe404686.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/55f56f18a7938440d97b78b1886cf7f56bbee18097747eb10fbb803bbe404686.jpg)

![62c826ebce90b1ccd9f5cf07d4dad75fd7afe37bcda32c670348e2ab7ef46244.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/62c826ebce90b1ccd9f5cf07d4dad75fd7afe37bcda32c670348e2ab7ef46244.jpg)

![7b6c356d69166aeac0b23c060da274d1ef600a144eb1e6ce49dde90d447a54be.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/7b6c356d69166aeac0b23c060da274d1ef600a144eb1e6ce49dde90d447a54be.jpg)

![8504449f08de138e8ccb58c812f1216b9caf5a28b569d87efce69f35a17b4d14.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/8504449f08de138e8ccb58c812f1216b9caf5a28b569d87efce69f35a17b4d14.jpg)

![9a457da96789e5b343bb4c5aaeb05cae779a7ddcc22add0132ad9bb788d24218.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/9a457da96789e5b343bb4c5aaeb05cae779a7ddcc22add0132ad9bb788d24218.jpg)

![9fa1af521439397bde5738830d158df321f8fedf25c01c06b862cd0e3c2a273f.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/9fa1af521439397bde5738830d158df321f8fedf25c01c06b862cd0e3c2a273f.jpg)

![a7e09db72b3b81534fede7e742cc1d55dcac7989b5dfbf712e7ab23c4d7bb825.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/a7e09db72b3b81534fede7e742cc1d55dcac7989b5dfbf712e7ab23c4d7bb825.jpg)

![c881f938b85a13647cd4dd3774e69d1c527fc6fd368705a962907d4fad7b0273.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/c881f938b85a13647cd4dd3774e69d1c527fc6fd368705a962907d4fad7b0273.jpg)

![eb5c23f75fd05d651b853e8a7610adcb43a419d0d81dbe057f3abd2e07066eac.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/eb5c23f75fd05d651b853e8a7610adcb43a419d0d81dbe057f3abd2e07066eac.jpg)

![f0df0e7aac09e0286a0f290951b4b8e124e4b64d9802483523170f8edc2174aa.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/f0df0e7aac09e0286a0f290951b4b8e124e4b64d9802483523170f8edc2174aa.jpg)

![f6db85f72221e23ee5f1e83a7733627d031cb76cce9332b0c000ae45e46b703d.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/f6db85f72221e23ee5f1e83a7733627d031cb76cce9332b0c000ae45e46b703d.jpg)

![fadd84a5a3d7dc4c984844b8544c73548b5516e2647f5963144ce10d529372d0.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/images/fadd84a5a3d7dc4c984844b8544c73548b5516e2647f5963144ce10d529372d0.jpg)

### Tables

![1e35e8cbc6dd2a0a8d7cf7e6fa5ec882512f7ad9814317121d815fd7a2dcd4a2.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/1e35e8cbc6dd2a0a8d7cf7e6fa5ec882512f7ad9814317121d815fd7a2dcd4a2.jpg)

![26318a8b18fc8ef5ea96f56b6de27e6b15aeba9f794d54e64de5347df84cda15.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/26318a8b18fc8ef5ea96f56b6de27e6b15aeba9f794d54e64de5347df84cda15.jpg)

![5aab788ae2feaa9d0f105f26600284cbe3866c03cccb9590f4584cbde04b01a4.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/5aab788ae2feaa9d0f105f26600284cbe3866c03cccb9590f4584cbde04b01a4.jpg)

![739ec6e228ffec1c027b8533c3244dad4abccaa760e763f08fa26ce4e2e63b29.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/739ec6e228ffec1c027b8533c3244dad4abccaa760e763f08fa26ce4e2e63b29.jpg)

![7b17d80fca6fbc5f55ad96372144f029282c8e7e28a14c01f88cc2b701a6bc67.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/7b17d80fca6fbc5f55ad96372144f029282c8e7e28a14c01f88cc2b701a6bc67.jpg)

![a44c8237aee1361b4bd07986faf973c5e6cb700c6318b3023f7f8e0d6f7a0a5c.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/a44c8237aee1361b4bd07986faf973c5e6cb700c6318b3023f7f8e0d6f7a0a5c.jpg)

![bbd3ea6f1261624f7d6bf243519cbd247fae7198950e35d1b54eae81a816aeec.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/bbd3ea6f1261624f7d6bf243519cbd247fae7198950e35d1b54eae81a816aeec.jpg)

![f0abdbf914b4e3cb4afb8268a04058e8b66fc4ee389e6dae012a1bd12b934af7.jpg](../icml_results/5170_BrainJanus_%20A%20Unified%20Model%20for%20Understanding%20and%20Generation%20across%20Brain%2C%20Vision%2C%20and%20Language/tables/f0abdbf914b4e3cb4afb8268a04058e8b66fc4ee389e6dae012a1bd12b934af7.jpg)

<a id="paper-5123"></a>

## What Makes a Strong Model? A Unified Spectral Analysis of Knowledge Transfer over High-dimensional Linear Regression


### Images

![3a5031c02f79c6c4e9b6c7071c40baf63da6b5480b58dc87fa700ce1e00df5bc.jpg](../icml_results/5171_What%20Makes%20a%20Strong%20Model_%20A%20Unified%20Spectral%20Analysis%20of%20Knowledge%20Transfer%20over%20High-dimensional%20L/images/3a5031c02f79c6c4e9b6c7071c40baf63da6b5480b58dc87fa700ce1e00df5bc.jpg)

![496c1765f435509de21c7860400e93aec9eb0e6d71cbcaeba312c83951c04f53.jpg](../icml_results/5171_What%20Makes%20a%20Strong%20Model_%20A%20Unified%20Spectral%20Analysis%20of%20Knowledge%20Transfer%20over%20High-dimensional%20L/images/496c1765f435509de21c7860400e93aec9eb0e6d71cbcaeba312c83951c04f53.jpg)

![da2da722efafffbf921b0e4dad90fa4a25d463aa967da1c5f8a14916739ac1c6.jpg](../icml_results/5171_What%20Makes%20a%20Strong%20Model_%20A%20Unified%20Spectral%20Analysis%20of%20Knowledge%20Transfer%20over%20High-dimensional%20L/images/da2da722efafffbf921b0e4dad90fa4a25d463aa967da1c5f8a14916739ac1c6.jpg)

### Tables

![585275e6b974cc41f6ef1ab4f5938576eab1294104bc3fb4244167b5dc1c48da.jpg](../icml_results/5171_What%20Makes%20a%20Strong%20Model_%20A%20Unified%20Spectral%20Analysis%20of%20Knowledge%20Transfer%20over%20High-dimensional%20L/tables/585275e6b974cc41f6ef1ab4f5938576eab1294104bc3fb4244167b5dc1c48da.jpg)

<a id="paper-5124"></a>

## HieraScaffold: Learning Compact Hierarchical Representations for Scalable 4D LiDAR Generation


### Images

![1173a5249104b5f3db4d6f1288df9ee842427b07152ee0171b5f0f0f0691b7ad.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/1173a5249104b5f3db4d6f1288df9ee842427b07152ee0171b5f0f0f0691b7ad.jpg)

![181bf844bd4b53c519670e507747f58cde5bda7a785940049f5144e9c5e7e876.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/181bf844bd4b53c519670e507747f58cde5bda7a785940049f5144e9c5e7e876.jpg)

![23d1441fa8846b737b6628ad8f20d3cc95805c986e35955ea74defc076d41f3f.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/23d1441fa8846b737b6628ad8f20d3cc95805c986e35955ea74defc076d41f3f.jpg)

![61a0492bbf2b3b2c5b2f9163bb88bac22219d60fd222fbe64abb32f67f53df8d.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/61a0492bbf2b3b2c5b2f9163bb88bac22219d60fd222fbe64abb32f67f53df8d.jpg)

![635987f2e336b388933db014e7bede57158ef8f47c657793a0e2987e38c79352.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/635987f2e336b388933db014e7bede57158ef8f47c657793a0e2987e38c79352.jpg)

![7bcfe7105efb69d562255fb8bff8c840c993db1baa09d369048f99bc2a73d80f.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/7bcfe7105efb69d562255fb8bff8c840c993db1baa09d369048f99bc2a73d80f.jpg)

![7f250a8cc65d826197e786f8ceb880398c89b20942fd25a8d4feeb345e6b51a8.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/7f250a8cc65d826197e786f8ceb880398c89b20942fd25a8d4feeb345e6b51a8.jpg)

![837a0ceba8a8e9cb17df79843f3c7e0be0e6e88f98a871c199f5bd429b786998.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/837a0ceba8a8e9cb17df79843f3c7e0be0e6e88f98a871c199f5bd429b786998.jpg)

![84a848771dace2ded50f52fbc546d36e803b43033306ba016d3694409eec4ef9.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/84a848771dace2ded50f52fbc546d36e803b43033306ba016d3694409eec4ef9.jpg)

![9a281c73375cd97c21cb8b8831979d5e63d0471767e3036620ba2805152f8d1f.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/9a281c73375cd97c21cb8b8831979d5e63d0471767e3036620ba2805152f8d1f.jpg)

![c8c4643d89ffb79de47411a7fe8e6bd39b8f03c8b8a7b09c73a808c944947aa0.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/c8c4643d89ffb79de47411a7fe8e6bd39b8f03c8b8a7b09c73a808c944947aa0.jpg)

![e2072db39b67de7ac594dfb923f3d8af6f5a27148588f7cea5e777092b2875f7.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/e2072db39b67de7ac594dfb923f3d8af6f5a27148588f7cea5e777092b2875f7.jpg)

![f6d59b340da38d599eacd3b882b9e3e4f24144e562ec5513030041578947516c.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/images/f6d59b340da38d599eacd3b882b9e3e4f24144e562ec5513030041578947516c.jpg)

### Tables

![1cefe02ead9615126aeb06782cbf664f5a1e41b7102e046603541013d12a7013.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/1cefe02ead9615126aeb06782cbf664f5a1e41b7102e046603541013d12a7013.jpg)

![414838d3ad0391280e2d52ace31ae32b4a279c13a793b040a3c85353ee1761c8.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/414838d3ad0391280e2d52ace31ae32b4a279c13a793b040a3c85353ee1761c8.jpg)

![45bb78cee15931e436bf26e2f910793bca049735a1f5a8c4d62db191f62f6d61.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/45bb78cee15931e436bf26e2f910793bca049735a1f5a8c4d62db191f62f6d61.jpg)

![5b3195ef7497f86ac2c99914848bb01d5d478e45bba3512e5230b64c34ce641c.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/5b3195ef7497f86ac2c99914848bb01d5d478e45bba3512e5230b64c34ce641c.jpg)

![5bff4f827e3dd4b9257757aac41a18f469a6b1080e0220981ff79ee8632d4f06.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/5bff4f827e3dd4b9257757aac41a18f469a6b1080e0220981ff79ee8632d4f06.jpg)

![642d6be45e05e777dec8dca26426746a4c5970a3185f57465c06cd66069b9b52.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/642d6be45e05e777dec8dca26426746a4c5970a3185f57465c06cd66069b9b52.jpg)

![85d4610a8de00bc65e76282185d7ee4143b30c32e8df1bc4573b275d38371bcf.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/85d4610a8de00bc65e76282185d7ee4143b30c32e8df1bc4573b275d38371bcf.jpg)

![ac7ef833c70c0ad0cdbc61d2a8ea3337dc37ddb2dd0d808cc5d0effcd4a944db.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/ac7ef833c70c0ad0cdbc61d2a8ea3337dc37ddb2dd0d808cc5d0effcd4a944db.jpg)

![c1772bb568c010e9200e781b8d1cf924bb1117dde359a2eb23a3e20deb5bc510.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/c1772bb568c010e9200e781b8d1cf924bb1117dde359a2eb23a3e20deb5bc510.jpg)

![dea77bc4778413e386163c230dcccf7054d148e2835e4559f22777f00fc14310.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/dea77bc4778413e386163c230dcccf7054d148e2835e4559f22777f00fc14310.jpg)

![f455dc4f46551986be4a326c54ab3c028c3f2db58c95b32431ec0b918e65231b.jpg](../icml_results/5172_HieraScaffold_%20Learning%20Compact%20Hierarchical%20Representations%20for%20Scalable%204D%20LiDAR%20Generation/tables/f455dc4f46551986be4a326c54ab3c028c3f2db58c95b32431ec0b918e65231b.jpg)

<a id="paper-5125"></a>

## VividCam: Learning Unconventional Camera Motions from Virtual Synthetic Videos


### Images

![00854156bc2618b2cfa601ede2ba8c284f29ecde364971afd33a34f5448ce725.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/00854156bc2618b2cfa601ede2ba8c284f29ecde364971afd33a34f5448ce725.jpg)

![02a06e6f5bfca079d5ff3da93abae432046d3cd7f245ed181d2db229861d0cd0.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/02a06e6f5bfca079d5ff3da93abae432046d3cd7f245ed181d2db229861d0cd0.jpg)

![0736a49469e26cfc37af82f1a85f19d763ce49d1ee12c3e36f41cf87730ad9c7.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/0736a49469e26cfc37af82f1a85f19d763ce49d1ee12c3e36f41cf87730ad9c7.jpg)

![407efce78c9fbc415aa8157ab6fa63b25c48a31ba387e18b438fb2eccb8fc5aa.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/407efce78c9fbc415aa8157ab6fa63b25c48a31ba387e18b438fb2eccb8fc5aa.jpg)

![470f6dbf5e3bfd5e5ce49c2dde5ff155a58fa7d2aa4b179101fc16a3d5f316e9.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/470f6dbf5e3bfd5e5ce49c2dde5ff155a58fa7d2aa4b179101fc16a3d5f316e9.jpg)

![4b63b859212da10e188f12595b13ea033d53a96aa3378b0fab5382f43e642b2c.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/4b63b859212da10e188f12595b13ea033d53a96aa3378b0fab5382f43e642b2c.jpg)

![4d4e008b6d27b63c186429ff294b28c753d94ebb916ca97c8efc4711a920f6f3.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/4d4e008b6d27b63c186429ff294b28c753d94ebb916ca97c8efc4711a920f6f3.jpg)

![8548fe9933c593d56f7da9c7b6c9bf434d1dac68ebb6c653ff2a5a08625478f3.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/8548fe9933c593d56f7da9c7b6c9bf434d1dac68ebb6c653ff2a5a08625478f3.jpg)

![8550f012f4ddff19bb04c2901060ed43cfd52b724c246c9e736bd40a6a4d6d55.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/8550f012f4ddff19bb04c2901060ed43cfd52b724c246c9e736bd40a6a4d6d55.jpg)

![869d3e0666264db1da8ea12a4f2fbb6df946de794b2e5903a2dc191675dcdab1.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/869d3e0666264db1da8ea12a4f2fbb6df946de794b2e5903a2dc191675dcdab1.jpg)

![8e4927dcd97e9518c2a86272ef35243a072e2ec868bc855a8904bbf1f6178993.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/8e4927dcd97e9518c2a86272ef35243a072e2ec868bc855a8904bbf1f6178993.jpg)

![94f8ff6b2e9397578745683a756cc4aef1a6f8926fdd64e7385da32f6a6991db.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/94f8ff6b2e9397578745683a756cc4aef1a6f8926fdd64e7385da32f6a6991db.jpg)

![9bcf85a5e552a55c671787eadb520496d4811f42b4bed4879189076e9a1f2c9a.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/9bcf85a5e552a55c671787eadb520496d4811f42b4bed4879189076e9a1f2c9a.jpg)

![9d0d52055056712f372fefe3dbe47d87bcc91c133cb646d0dbc7b396d2ae1363.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/9d0d52055056712f372fefe3dbe47d87bcc91c133cb646d0dbc7b396d2ae1363.jpg)

![a0f22692a18405e9cb72a627ddffb6dd825c69dce632dac96424cbf32c17646a.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/a0f22692a18405e9cb72a627ddffb6dd825c69dce632dac96424cbf32c17646a.jpg)

![a701bcbefa0cfab1961c7863c792ca8cf8d2d5c2cf8352fc3930757b3172c538.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/a701bcbefa0cfab1961c7863c792ca8cf8d2d5c2cf8352fc3930757b3172c538.jpg)

![d744d511771b7331a179542dfe9b2ee37c31382513ac7bb6b309830f65210241.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/d744d511771b7331a179542dfe9b2ee37c31382513ac7bb6b309830f65210241.jpg)

![fdb7de4c7296084499b5312cefdae0a9121de374cbbd00a450c5fe3c078490c2.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/images/fdb7de4c7296084499b5312cefdae0a9121de374cbbd00a450c5fe3c078490c2.jpg)

### Tables

![0634b80b6d21194a0f7d303ec8c02c486fbe7401bd0c93b1359e46961c6f9c7d.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/0634b80b6d21194a0f7d303ec8c02c486fbe7401bd0c93b1359e46961c6f9c7d.jpg)

![0a165e516f7e812d8499a4c565080c43e1700a379a150340ad58f1fb3be6b44a.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/0a165e516f7e812d8499a4c565080c43e1700a379a150340ad58f1fb3be6b44a.jpg)

![23093f1534a22c5298aa11cefc4f66609c7f077e43e388a736198a753b826e47.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/23093f1534a22c5298aa11cefc4f66609c7f077e43e388a736198a753b826e47.jpg)

![2760ea8e176ba224f733e22043312b82588ddcde88aac9713b876837ab70e284.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/2760ea8e176ba224f733e22043312b82588ddcde88aac9713b876837ab70e284.jpg)

![2e6da3c8717bc385c34bee5574c259d703e4a5bfbf8e2caed942a344664600a8.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/2e6da3c8717bc385c34bee5574c259d703e4a5bfbf8e2caed942a344664600a8.jpg)

![3359ce53b378a90abb11abd4ff48671d57ac5d0529e3e2c69a73594533a22aab.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/3359ce53b378a90abb11abd4ff48671d57ac5d0529e3e2c69a73594533a22aab.jpg)

![5446156073ef17b336dfa7ee8d29bc9e0b662840707abc3abfaf66257e3d5ace.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/5446156073ef17b336dfa7ee8d29bc9e0b662840707abc3abfaf66257e3d5ace.jpg)

![5b24167b27c4e644b6765d97a0a6bdc6af454f827aa294a8d9fcd728d130472a.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/5b24167b27c4e644b6765d97a0a6bdc6af454f827aa294a8d9fcd728d130472a.jpg)

![61b5429b527cff5d3b9a8882d9a1cf419388ca6492dc778027ff5e9e6a8ac7c7.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/61b5429b527cff5d3b9a8882d9a1cf419388ca6492dc778027ff5e9e6a8ac7c7.jpg)

![7480472ae25598fe3d1efaa9bb67b0f1063a1f796e398d61bf6ea5d785ab1c44.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/7480472ae25598fe3d1efaa9bb67b0f1063a1f796e398d61bf6ea5d785ab1c44.jpg)

![9464c171183e1f8ace7143c237b9e51f708b27ce685c11a2f1a2126e329c3263.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/9464c171183e1f8ace7143c237b9e51f708b27ce685c11a2f1a2126e329c3263.jpg)

![ad1d1d7f31170a2839c094a4fd926bf8d5a26f97cf4098c7fedea8e46017048e.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/ad1d1d7f31170a2839c094a4fd926bf8d5a26f97cf4098c7fedea8e46017048e.jpg)

![c43cd477e08ec5ef7234114f89d6f2829b45ad0f565623ebb013ff5c2b8c1578.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/c43cd477e08ec5ef7234114f89d6f2829b45ad0f565623ebb013ff5c2b8c1578.jpg)

![f291824d02bee9268898c75388d03b8307bdf5935ad3ffec364e2a0255a8000e.jpg](../icml_results/5173_VividCam_%20Learning%20Unconventional%20Camera%20Motions%20from%20Virtual%20Synthetic%20Videos/tables/f291824d02bee9268898c75388d03b8307bdf5935ad3ffec364e2a0255a8000e.jpg)

<a id="paper-5126"></a>

## LagLLM: LLM-empowered lead–lag dependency learning for spatial-temporal time series forecasting


### Images

![2676e7bffe9bc1e348db0a9d1d9a41fab91264ca443b95c8eb6167dd57b58c8c.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/images/2676e7bffe9bc1e348db0a9d1d9a41fab91264ca443b95c8eb6167dd57b58c8c.jpg)

![31a4492d0c5a4d622531bb8406cd1a6158db17992bf2fd077de7fe394ea0c5f4.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/images/31a4492d0c5a4d622531bb8406cd1a6158db17992bf2fd077de7fe394ea0c5f4.jpg)

![3de22110c099789fef20294d0a1e633fa5412da6c8a0c8b002b33f9572bc340e.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/images/3de22110c099789fef20294d0a1e633fa5412da6c8a0c8b002b33f9572bc340e.jpg)

![50578bceda8f025f7a9b2d6eb8c52110a80520a99052059ce1cd947679387387.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/images/50578bceda8f025f7a9b2d6eb8c52110a80520a99052059ce1cd947679387387.jpg)

![b92f62a9d6c77093c05a404bbc6b3a53e6e26440439d053ca9fb4383783f38c9.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/images/b92f62a9d6c77093c05a404bbc6b3a53e6e26440439d053ca9fb4383783f38c9.jpg)

### Tables

![12db3198b5a1428f863a436e6aee384b90261c2b9c45b02b8a408e4e6fcf6fbe.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/12db3198b5a1428f863a436e6aee384b90261c2b9c45b02b8a408e4e6fcf6fbe.jpg)

![1b03f68f83b1539a6c7b874b35a9949127e1bfdc7c9e648b94402da8827f4037.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/1b03f68f83b1539a6c7b874b35a9949127e1bfdc7c9e648b94402da8827f4037.jpg)

![4a302c7343b57fa8bcc1dd2f7cc129f2f2f77c123bdc658fb44f78b38c850da8.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/4a302c7343b57fa8bcc1dd2f7cc129f2f2f77c123bdc658fb44f78b38c850da8.jpg)

![6983cd8f6e7cc64857542d17052afdf8623c9507320f17bba9eab0e3d2a41d09.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/6983cd8f6e7cc64857542d17052afdf8623c9507320f17bba9eab0e3d2a41d09.jpg)

![7d68f0e6074cf13aef0aa3b2d1d3e564c54d5c9adb876afa9c4147e25cdbf66a.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/7d68f0e6074cf13aef0aa3b2d1d3e564c54d5c9adb876afa9c4147e25cdbf66a.jpg)

![99aafcd2b49588f631de6fced5f98dcfa9ddde974e941e1eb33606b93281b4bd.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/99aafcd2b49588f631de6fced5f98dcfa9ddde974e941e1eb33606b93281b4bd.jpg)

![9f2259a56ff9b8b103dc873cf68c8d94643aec1800765a3d007df084098560d4.jpg](../icml_results/5174_LagLLM_%20LLM-empowered%20lead%E2%80%93lag%20dependency%20learning%20for%20spatial-temporal%20time%20series%20forecasting/tables/9f2259a56ff9b8b103dc873cf68c8d94643aec1800765a3d007df084098560d4.jpg)

<a id="paper-5127"></a>

## Rank-guided Diffusion for Noise Few-Shot Learning


### Images

![076dd8a6ba3d4b89f94ff984393537c42fa58cde0a10b18c74996f8c3b2415dc.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/076dd8a6ba3d4b89f94ff984393537c42fa58cde0a10b18c74996f8c3b2415dc.jpg)

![090e56f07678ea127d8a2c478babaebc4ed8abcba2078bdcd191f7481307a3d9.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/090e56f07678ea127d8a2c478babaebc4ed8abcba2078bdcd191f7481307a3d9.jpg)

![11c2acb9173a54536282aaea915ed38e78ebfebfc17b24df84e8c3367c375c30.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/11c2acb9173a54536282aaea915ed38e78ebfebfc17b24df84e8c3367c375c30.jpg)

![61c15f49c57546abbca1043cd24e78076e3d89369a5df068537c8d2463994c16.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/61c15f49c57546abbca1043cd24e78076e3d89369a5df068537c8d2463994c16.jpg)

![8af5690699b43c2b445414a167da2ce5f56450bc41e4be0717ad3db1e3eb4555.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/8af5690699b43c2b445414a167da2ce5f56450bc41e4be0717ad3db1e3eb4555.jpg)

![b7a71ff46cd805785d097703bc8924aafb17ba9e4a8f7950b1a7f68e1c6ef001.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/b7a71ff46cd805785d097703bc8924aafb17ba9e4a8f7950b1a7f68e1c6ef001.jpg)

![f1d57390031c2d10313187fa27ad8546d5f2c990923ba0626d3dc91b4967cb8a.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/images/f1d57390031c2d10313187fa27ad8546d5f2c990923ba0626d3dc91b4967cb8a.jpg)

### Tables

![2cd082334ac1dd336843a2bb7ce8d0bb08d48ce329938262bc54bf7d2975a5da.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/2cd082334ac1dd336843a2bb7ce8d0bb08d48ce329938262bc54bf7d2975a5da.jpg)

![49b124bf4e747a2ee776d631e88a0f153b540b638ad512a9e110d23cab34bda1.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/49b124bf4e747a2ee776d631e88a0f153b540b638ad512a9e110d23cab34bda1.jpg)

![69642431686873341418ef27e25e101a56d815e26ee1303776f457596da74f25.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/69642431686873341418ef27e25e101a56d815e26ee1303776f457596da74f25.jpg)

![92a591af9241de1f21a59a1c6a7ba89a78b4cd21f711a33edcc2a34080881d53.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/92a591af9241de1f21a59a1c6a7ba89a78b4cd21f711a33edcc2a34080881d53.jpg)

![b79205b742768ca78c160f9380f9e89d09e0667b8b99044af243e26ca6ae7e3b.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/b79205b742768ca78c160f9380f9e89d09e0667b8b99044af243e26ca6ae7e3b.jpg)

![bc90c6b511ceaf72a01d6657a00592a1211a1a4a8c729f7307956efa439148bb.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/bc90c6b511ceaf72a01d6657a00592a1211a1a4a8c729f7307956efa439148bb.jpg)

![e66d03eb34eeefd8a01f70c87e2c44ea3ca7d2f944eff8a9107762d974d8d560.jpg](../icml_results/5175_Rank-guided%20Diffusion%20for%20Noise%20Few-Shot%20Learning/tables/e66d03eb34eeefd8a01f70c87e2c44ea3ca7d2f944eff8a9107762d974d8d560.jpg)

<a id="paper-5128"></a>

## rePIRL: Learn PRM with Inverse RL for LLM Reasoning


### Images

![2c552281090c82eba697c6d5287048b963caf7bb5ac0ada26d23a63252c9280e.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/images/2c552281090c82eba697c6d5287048b963caf7bb5ac0ada26d23a63252c9280e.jpg)

![7d9b4702f0ed13ed1799386713a51be909b205dca861708918fe89cccc11c094.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/images/7d9b4702f0ed13ed1799386713a51be909b205dca861708918fe89cccc11c094.jpg)

![c18e47d206483ea31a957fe378a85aab6cd67e830e690f0b92cafdb9d281e635.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/images/c18e47d206483ea31a957fe378a85aab6cd67e830e690f0b92cafdb9d281e635.jpg)

![e8bde4b7131287941787dc06cfe4089dc4338796000176e30072381db000ca14.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/images/e8bde4b7131287941787dc06cfe4089dc4338796000176e30072381db000ca14.jpg)

### Tables

![0ce0c9f30dd54b68fbc7819a5a034e72efc2b9f9b8689cba03c9d86dd21c7fee.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/0ce0c9f30dd54b68fbc7819a5a034e72efc2b9f9b8689cba03c9d86dd21c7fee.jpg)

![1ab44c4f4b3110a10aca8e68df7b366aa8b845a09a38840558bf4f2890da5b61.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/1ab44c4f4b3110a10aca8e68df7b366aa8b845a09a38840558bf4f2890da5b61.jpg)

![773d591e82d22026ce330a8c0fa07055503e0de130c0a5ac0285457eb63f38d6.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/773d591e82d22026ce330a8c0fa07055503e0de130c0a5ac0285457eb63f38d6.jpg)

![9d65ee2d3988d3dadf875992202ff1f62318d7ca8070f002cd1c79e5a56ebc61.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/9d65ee2d3988d3dadf875992202ff1f62318d7ca8070f002cd1c79e5a56ebc61.jpg)

![b52f77ae29b45f87be965590ace8a9020d7303544522563d71c56a66813a6f4c.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/b52f77ae29b45f87be965590ace8a9020d7303544522563d71c56a66813a6f4c.jpg)

![bf0760d08c0a0de7050274f28f6b9b0e6b993be1e6e85bec7237f07f91ae281f.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/bf0760d08c0a0de7050274f28f6b9b0e6b993be1e6e85bec7237f07f91ae281f.jpg)

![d0b2a586d845732b0e7e2b67d159b776b5d3144959b290c8603b3d18619e7754.jpg](../icml_results/5176_rePIRL_%20Learn%20PRM%20with%20Inverse%20RL%20for%20LLM%20Reasoning/tables/d0b2a586d845732b0e7e2b67d159b776b5d3144959b290c8603b3d18619e7754.jpg)

<a id="paper-5129"></a>

## CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations


### Images

![092e5f6db4b92a640a081fdcde177cdd9e781eeb4f064a1ad5f1f9e87f792e96.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/092e5f6db4b92a640a081fdcde177cdd9e781eeb4f064a1ad5f1f9e87f792e96.jpg)

![192e817c1da83c8d625a9e3558c472b82d8eb05586af19869cb1dbca9b0b2bc5.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/192e817c1da83c8d625a9e3558c472b82d8eb05586af19869cb1dbca9b0b2bc5.jpg)

![1f9c8dd65b38eaab80894a70720e49da1dc6d1a3434e818a4f0e647db1a0bb20.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/1f9c8dd65b38eaab80894a70720e49da1dc6d1a3434e818a4f0e647db1a0bb20.jpg)

![24b6217442f04c76983ce1c1f58d103ed66d19821ca595b9bc183c5197fd3da2.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/24b6217442f04c76983ce1c1f58d103ed66d19821ca595b9bc183c5197fd3da2.jpg)

![a69d3f32323a2267c12a9ca982f1a833498b68c8c34e846d6748e95ee00aff83.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/a69d3f32323a2267c12a9ca982f1a833498b68c8c34e846d6748e95ee00aff83.jpg)

![ced91eb015ef2ff6e6b4ddc2a27b01c1ffd3eb57332afdf86565e7e152b2eeb5.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/ced91eb015ef2ff6e6b4ddc2a27b01c1ffd3eb57332afdf86565e7e152b2eeb5.jpg)

![e3fab1559d234d0a116614dae7b1c62e0cd53c478ddb9d214c7fa99ad6c87079.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/images/e3fab1559d234d0a116614dae7b1c62e0cd53c478ddb9d214c7fa99ad6c87079.jpg)

### Tables

![0862496110a7c4e8643a4c76f141eab00a5264bbc62a3f4242df0a1d01fb4e3b.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/0862496110a7c4e8643a4c76f141eab00a5264bbc62a3f4242df0a1d01fb4e3b.jpg)

![09216c1d7cfa314dabd17e0b1b1ee7040f655a6fc8c2d52c4c5f0b7d0ad0df92.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/09216c1d7cfa314dabd17e0b1b1ee7040f655a6fc8c2d52c4c5f0b7d0ad0df92.jpg)

![0dd0344870c2723c4eaa475ceea2932f003c29ba4fcdf0bc40ec1ca38577d72e.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/0dd0344870c2723c4eaa475ceea2932f003c29ba4fcdf0bc40ec1ca38577d72e.jpg)

![2cb970341235a8610a374aab09f59bb34194b0e1a730703ace4a5d52cf4126c5.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/2cb970341235a8610a374aab09f59bb34194b0e1a730703ace4a5d52cf4126c5.jpg)

![42b695f98dea6e3dc6cf12f14e512b5c48a00f4bc258fe4beed96a154f8fbd08.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/42b695f98dea6e3dc6cf12f14e512b5c48a00f4bc258fe4beed96a154f8fbd08.jpg)

![6a47ca8cac06cf678c51df39a84eaae77bd5a0efe17a20f471c1ba9eb5168b4c.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/6a47ca8cac06cf678c51df39a84eaae77bd5a0efe17a20f471c1ba9eb5168b4c.jpg)

![6b29f1fcb5c8bdcead92c64a0215b2c3e3f1ab37de4069e692079a0511b19342.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/6b29f1fcb5c8bdcead92c64a0215b2c3e3f1ab37de4069e692079a0511b19342.jpg)

![6e7dc879ccce451bb964d9ef863838f2d7376fb767c21c8e98d7b0877908d0a0.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/6e7dc879ccce451bb964d9ef863838f2d7376fb767c21c8e98d7b0877908d0a0.jpg)

![81f87e6d4c19b3c93670da3a2b1647942e24b5cc3ca557b75295654c028bb6d1.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/81f87e6d4c19b3c93670da3a2b1647942e24b5cc3ca557b75295654c028bb6d1.jpg)

![a69e8d47f14d6c0f066807243c7f7fb305ded0f10a8a846e375049b8ca077883.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/a69e8d47f14d6c0f066807243c7f7fb305ded0f10a8a846e375049b8ca077883.jpg)

![bf96e4d6c65708296f7d70444d3434b7d3c36e33a5bbb667e53359226ac00317.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/bf96e4d6c65708296f7d70444d3434b7d3c36e33a5bbb667e53359226ac00317.jpg)

![c340fc0bcd67af4ae88f61270fb8e9759d7030db6d70ab6201583d74e19e6aee.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/c340fc0bcd67af4ae88f61270fb8e9759d7030db6d70ab6201583d74e19e6aee.jpg)

![c979b957f409956eedf2ab8cca9f1ecc565f76e1ecebe8adfb6d757285c96ca9.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/c979b957f409956eedf2ab8cca9f1ecc565f76e1ecebe8adfb6d757285c96ca9.jpg)

![cde223a4a72a5655438aab71d05d2b561652f578d8cc54125158d6ae319da0f4.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/cde223a4a72a5655438aab71d05d2b561652f578d8cc54125158d6ae319da0f4.jpg)

![d4b6ea503f6f25c895e08d074c698a403af9e1f306a67de2f488f9c8bdbbc831.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/d4b6ea503f6f25c895e08d074c698a403af9e1f306a67de2f488f9c8bdbbc831.jpg)

![ed751286c60c6efe382d75d5ba3d3b9ef9b21c2005942098edda9c058f107cfc.jpg](../icml_results/5177_CORE-MTL_%20Rethinking%20Gradient%20Balancing%20via%20Causal%20Orthogonal%20Representations/tables/ed751286c60c6efe382d75d5ba3d3b9ef9b21c2005942098edda9c058f107cfc.jpg)

<a id="paper-5130"></a>

## Hierarchical Goal Abstractions via Learned Subset Relations


### Images

![27d09ebd38ac0f0e2ef9bb189ec2f58355dfd6e4fa798d595a8c4471f6c0afc2.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/27d09ebd38ac0f0e2ef9bb189ec2f58355dfd6e4fa798d595a8c4471f6c0afc2.jpg)

![2d10bf6e71609131c43c0daf5996afba558530db7e3e634eb745eb88a40ef020.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/2d10bf6e71609131c43c0daf5996afba558530db7e3e634eb745eb88a40ef020.jpg)

![42a62490d6f3348aeafe34e6cbe4d5b9b34925e0553fd89d416da54b3887fd53.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/42a62490d6f3348aeafe34e6cbe4d5b9b34925e0553fd89d416da54b3887fd53.jpg)

![48d78e16d348d74aeafa7034f03d7904749807771c4d37dd87c6b1e5774a201f.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/48d78e16d348d74aeafa7034f03d7904749807771c4d37dd87c6b1e5774a201f.jpg)

![4e813ba37d7b7f27ff0a12d55295219f56c5715bc7811854eb9e5901d95becbd.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/4e813ba37d7b7f27ff0a12d55295219f56c5715bc7811854eb9e5901d95becbd.jpg)

![59f7e716c99f9346ff7ed144a2db413f484efaf23b05fc7bb4752f5193f7d7ca.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/59f7e716c99f9346ff7ed144a2db413f484efaf23b05fc7bb4752f5193f7d7ca.jpg)

![8a93d0adf0df15f2752bfd0f84da93579f8503a32335ec25faa515076a947be9.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/8a93d0adf0df15f2752bfd0f84da93579f8503a32335ec25faa515076a947be9.jpg)

![93c41760d5df6ce3f57d5036e2a4a096742a7b5605f4019099da958a70fd86fa.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/93c41760d5df6ce3f57d5036e2a4a096742a7b5605f4019099da958a70fd86fa.jpg)

![94449614dd8f576246eb87b882a82ea16dec9ec258839a6ec4a4c72e628e0946.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/94449614dd8f576246eb87b882a82ea16dec9ec258839a6ec4a4c72e628e0946.jpg)

![992093dafaf3a8209b9325e35dc2fe4bb2ce7fb2963c5e5140e1e91054ff4fbc.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/992093dafaf3a8209b9325e35dc2fe4bb2ce7fb2963c5e5140e1e91054ff4fbc.jpg)

![ba6803e7861ff12304f2ba86f81652d6d9acb94b2c05dbe30017f645a4fbd2c9.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/ba6803e7861ff12304f2ba86f81652d6d9acb94b2c05dbe30017f645a4fbd2c9.jpg)

![bd42bbb00b4257ca7eddf6fa231dfdd2cc88bfaa2afc83a6cd699885c6416ca4.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/bd42bbb00b4257ca7eddf6fa231dfdd2cc88bfaa2afc83a6cd699885c6416ca4.jpg)

![c08d3454f4a46c8849cd0161caced8ffff9a423e6cf50a2622b01f7a476ed288.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/c08d3454f4a46c8849cd0161caced8ffff9a423e6cf50a2622b01f7a476ed288.jpg)

![d6b996e432992597e25cf9cbc7677dd23c4aca959a501e8bc30c77df94afd1a8.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/d6b996e432992597e25cf9cbc7677dd23c4aca959a501e8bc30c77df94afd1a8.jpg)

![df5c916fcc0b2fdc50a2b32f3313c87167deaa1b31dd62f7b135c3a7915715ca.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/df5c916fcc0b2fdc50a2b32f3313c87167deaa1b31dd62f7b135c3a7915715ca.jpg)

![fb3f49f75e0458b4b40c5a38b0aeb8204270b83fbc5f32d06b3c142247aedf26.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/images/fb3f49f75e0458b4b40c5a38b0aeb8204270b83fbc5f32d06b3c142247aedf26.jpg)

### Tables

![061fd034d7a878fdacd8d736ffab8039a8f873acf8f183239f8ca3f613a6f055.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/061fd034d7a878fdacd8d736ffab8039a8f873acf8f183239f8ca3f613a6f055.jpg)

![609ab9f8b7c564c21a99deecce313742706d3ea27f0d44d48137ec0877ec703b.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/609ab9f8b7c564c21a99deecce313742706d3ea27f0d44d48137ec0877ec703b.jpg)

![6279b51bd5b80e1a5dd432578e3f022ca3bbbe73711287065049677c17097215.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/6279b51bd5b80e1a5dd432578e3f022ca3bbbe73711287065049677c17097215.jpg)

![86a11d1eef2f75039c2a8da9225603ca365c07e98ea937fcc362009e5c4c4d8b.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/86a11d1eef2f75039c2a8da9225603ca365c07e98ea937fcc362009e5c4c4d8b.jpg)

![a4bd052afd3a01a66cffa4c33750e78153bc6fbbae590c18dc571715a60a8a03.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/a4bd052afd3a01a66cffa4c33750e78153bc6fbbae590c18dc571715a60a8a03.jpg)

![cbfe8ae8c8999755a7c4802adefa9a225faec91fc2317377b0b5add5413c2f29.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/cbfe8ae8c8999755a7c4802adefa9a225faec91fc2317377b0b5add5413c2f29.jpg)

![d83ce5c5c082dbe92f684a0a17beb84cd3433c5cfa5eb3acca13f11dd1fb2882.jpg](../icml_results/5178_Hierarchical%20Goal%20Abstractions%20via%20Learned%20Subset%20Relations/tables/d83ce5c5c082dbe92f684a0a17beb84cd3433c5cfa5eb3acca13f11dd1fb2882.jpg)

<a id="paper-5131"></a>

## Q-Tab: Quantized Tabular Data Generator


### Images

![959595cc7a856f488d3e4402a8b952da1256a542a23173f4470bd4c0ea8e9b9b.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/images/959595cc7a856f488d3e4402a8b952da1256a542a23173f4470bd4c0ea8e9b9b.jpg)

![ada536b0a7c75b93c3578ad752b15fdd4afdeadf24e69fb3a92fd4d7f2ac610b.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/images/ada536b0a7c75b93c3578ad752b15fdd4afdeadf24e69fb3a92fd4d7f2ac610b.jpg)

### Tables

![04cd8f1dc66f2449dec27c6ae8b3f19bfaee019a8c4d3f30403e45401fd97b98.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/04cd8f1dc66f2449dec27c6ae8b3f19bfaee019a8c4d3f30403e45401fd97b98.jpg)

![099649dd52040484d7eee4d5b878970537d40fa408686931e6f6b3529ca936fa.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/099649dd52040484d7eee4d5b878970537d40fa408686931e6f6b3529ca936fa.jpg)

![13c629d4c25bc373d5fbe63b3e4370f491dcf98546b34a9edd9efd16f40a262f.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/13c629d4c25bc373d5fbe63b3e4370f491dcf98546b34a9edd9efd16f40a262f.jpg)

![3ea622641d372baa67eadc2ec226b0c197fb340c0436857a613014d23c060baf.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/3ea622641d372baa67eadc2ec226b0c197fb340c0436857a613014d23c060baf.jpg)

![4a3e149feccf58f927b07aee5181b6eff69aa209e59ce3790789319d6b34c229.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/4a3e149feccf58f927b07aee5181b6eff69aa209e59ce3790789319d6b34c229.jpg)

![4d2a2d9bbe53525afc695126d80a3700a1fb4f4350410781215c481680e48608.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/4d2a2d9bbe53525afc695126d80a3700a1fb4f4350410781215c481680e48608.jpg)

![5d508472f3f28a603e09895729e424aa8e60fb333039e07f11370ae9705d4d13.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/5d508472f3f28a603e09895729e424aa8e60fb333039e07f11370ae9705d4d13.jpg)

![6abf3ca8f5dd057798acb9cee5bdb8f95d1bd43c0052c347eac1499fdd48db00.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/6abf3ca8f5dd057798acb9cee5bdb8f95d1bd43c0052c347eac1499fdd48db00.jpg)

![783ddaa2fbe1a79ff964530c07e23b8471771f5b42662d9e8341fd748ca41513.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/783ddaa2fbe1a79ff964530c07e23b8471771f5b42662d9e8341fd748ca41513.jpg)

![862407e94f0fc9e9075ee04a6d31bfe7179b1e3fdc3696e03a40aae55ca6965c.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/862407e94f0fc9e9075ee04a6d31bfe7179b1e3fdc3696e03a40aae55ca6965c.jpg)

![89552165a300664485969d10faebf3f6951c4ae8619daea1d931b09dc68eab84.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/89552165a300664485969d10faebf3f6951c4ae8619daea1d931b09dc68eab84.jpg)

![9a86f492b489b48ca2b6e706d111705b2a47a1916db5c0d6938f03f3f7188057.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/9a86f492b489b48ca2b6e706d111705b2a47a1916db5c0d6938f03f3f7188057.jpg)

![a01a63cc9ee35a6213b68bbd78f5fee8e445a9aacd3104bb30f1d3b038bae99f.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/a01a63cc9ee35a6213b68bbd78f5fee8e445a9aacd3104bb30f1d3b038bae99f.jpg)

![a6e9229d6288574d2ab64d6b87f904578e49b7a8ad0c750ef02a74a957ab85af.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/a6e9229d6288574d2ab64d6b87f904578e49b7a8ad0c750ef02a74a957ab85af.jpg)

![beabd749245133e1436615412af8de48f27aa8cb85887590d81688a12d8dc8ef.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/beabd749245133e1436615412af8de48f27aa8cb85887590d81688a12d8dc8ef.jpg)

![bf49120dbe888ce69f96bf2d6fc60382fc28b6a6828cadecdce6533ed414698a.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/bf49120dbe888ce69f96bf2d6fc60382fc28b6a6828cadecdce6533ed414698a.jpg)

![c79b2e7fa2b5ede007e1c57ce5515ab6fc2f220970a43a2c8126e668c95d3ccf.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/c79b2e7fa2b5ede007e1c57ce5515ab6fc2f220970a43a2c8126e668c95d3ccf.jpg)

![d31eb5db00da6b96ca1e877b3503f10d7435ffd872edd84007d322de90a16959.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/d31eb5db00da6b96ca1e877b3503f10d7435ffd872edd84007d322de90a16959.jpg)

![dc8238f0b77353874fe9d74b671023787b40f6daeb0fae036d49cb5354b0a81e.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/dc8238f0b77353874fe9d74b671023787b40f6daeb0fae036d49cb5354b0a81e.jpg)

![ecd70441695255aba39eee470ae10b2989be5dc79ad2602fb7eafe4346fdceb1.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/ecd70441695255aba39eee470ae10b2989be5dc79ad2602fb7eafe4346fdceb1.jpg)

![ee8d22e297793ad0166cd557eb2affd2ffd2611f7789d049342f3e7d8cb3039b.jpg](../icml_results/5179_Q-Tab_%20Quantized%20Tabular%20Data%20Generator/tables/ee8d22e297793ad0166cd557eb2affd2ffd2611f7789d049342f3e7d8cb3039b.jpg)

<a id="paper-5132"></a>

## Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting


### Images

![0a0018429538e47fe33ee470cc1847f7ddf67e62eab93399f6eb393be54e488d.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/0a0018429538e47fe33ee470cc1847f7ddf67e62eab93399f6eb393be54e488d.jpg)

![15d24673f86a3537a8db08d8215fb2d87cb28dea93115a47504481e8b9b8ab4a.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/15d24673f86a3537a8db08d8215fb2d87cb28dea93115a47504481e8b9b8ab4a.jpg)

![20199b8da0b5e6ede8ffd1207b52c8515f0cc1bbaa4023e2552520242da3b0bf.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/20199b8da0b5e6ede8ffd1207b52c8515f0cc1bbaa4023e2552520242da3b0bf.jpg)

![20650bb790a6ef61998b0cf674c670ef7d2e3f4761c8aa32cfc9901311bd5adf.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/20650bb790a6ef61998b0cf674c670ef7d2e3f4761c8aa32cfc9901311bd5adf.jpg)

![225b0cca0a6e8071fb3ac860e2b9afb7717db2f49000ca0136879e502a26cf94.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/225b0cca0a6e8071fb3ac860e2b9afb7717db2f49000ca0136879e502a26cf94.jpg)

![2278e0b78c616b9997e64e060215580e6ae9c371cce78f32f555ba5c9bea4671.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/2278e0b78c616b9997e64e060215580e6ae9c371cce78f32f555ba5c9bea4671.jpg)

![2a0a3d31aff4259e6ef404f9fc6ba18f433ef76f3bcada64426d910bdebe8276.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/2a0a3d31aff4259e6ef404f9fc6ba18f433ef76f3bcada64426d910bdebe8276.jpg)

![31f4d8f120b2a57907b99d5bf2a177a478803f41b724d3277014bf8e4739250d.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/31f4d8f120b2a57907b99d5bf2a177a478803f41b724d3277014bf8e4739250d.jpg)

![3b5c921efcb8361c22eeaa1d9de0c01bd1c093783fd74c75017aee1908237020.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/3b5c921efcb8361c22eeaa1d9de0c01bd1c093783fd74c75017aee1908237020.jpg)

![6628f30438b510e634803092e8232a3c7f0ccbfac5b1da26ec768c62a6c1fb55.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/6628f30438b510e634803092e8232a3c7f0ccbfac5b1da26ec768c62a6c1fb55.jpg)

![6b036a0101ead770660edba325a6ce63c47265369cf72228f6844c28de70a2e1.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/6b036a0101ead770660edba325a6ce63c47265369cf72228f6844c28de70a2e1.jpg)

![6c4799d754115d43a2ba6fafb8f167f86dd0653a9ed2ed01bed25eb745826a63.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/6c4799d754115d43a2ba6fafb8f167f86dd0653a9ed2ed01bed25eb745826a63.jpg)

![7129e88895c6fae6e6576c588c7f6b6eb816f21f074867fe5c24b24e73f7fe98.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/7129e88895c6fae6e6576c588c7f6b6eb816f21f074867fe5c24b24e73f7fe98.jpg)

![7f40e05e30e1a3a66ba1bc75416c2e9aac2602a38c5b80036a4b2ff1f90b7408.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/7f40e05e30e1a3a66ba1bc75416c2e9aac2602a38c5b80036a4b2ff1f90b7408.jpg)

![8a77346520fad3c88c977040a3fa50ad0d3f0421545021334ef15f92682b682e.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/8a77346520fad3c88c977040a3fa50ad0d3f0421545021334ef15f92682b682e.jpg)

![923d1191a742624f3258de6cd29019eead5582a7cf8b5524953d06c5391e2b66.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/923d1191a742624f3258de6cd29019eead5582a7cf8b5524953d06c5391e2b66.jpg)

![ab11741da109afae59a372a8aab1d692855e6a8c473e8ab66c2ff5d196cc3841.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/ab11741da109afae59a372a8aab1d692855e6a8c473e8ab66c2ff5d196cc3841.jpg)

![bdbf598b24957519bb322217f08f18f265998d994bf8fbca6648f94183fb35a9.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/bdbf598b24957519bb322217f08f18f265998d994bf8fbca6648f94183fb35a9.jpg)

![ca3d57a5818722bef75ac8bac603e4680a792f2b844505ab79e1d4bc937280b8.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/ca3d57a5818722bef75ac8bac603e4680a792f2b844505ab79e1d4bc937280b8.jpg)

![d5129fbf21a9d5c19e9c0b31b96244e392d8ab2aa10b4d357c3511ce0d7b624b.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/d5129fbf21a9d5c19e9c0b31b96244e392d8ab2aa10b4d357c3511ce0d7b624b.jpg)

![eb9abd5a4be3ecff217ee009bf3980b81c7942399e04fe33fedd895c3d462fc2.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/eb9abd5a4be3ecff217ee009bf3980b81c7942399e04fe33fedd895c3d462fc2.jpg)

![f6c3fcdadd2b250cd58a916a454061d0b5d8ddb49a2831eab423bc6a167a70df.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/f6c3fcdadd2b250cd58a916a454061d0b5d8ddb49a2831eab423bc6a167a70df.jpg)

![fbbc9979eddd3ce7776efee65ec8088ec9a70e2adcc297c3459c30e40b582185.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/fbbc9979eddd3ce7776efee65ec8088ec9a70e2adcc297c3459c30e40b582185.jpg)

![fbd7e2ee414abb1c7e7527f5aaa8f46d4947825dc3e2cfbdafd2e3fbcaa94c8a.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/fbd7e2ee414abb1c7e7527f5aaa8f46d4947825dc3e2cfbdafd2e3fbcaa94c8a.jpg)

![fdd415422416e0daba13e821f730acfe24b110bf8b6df88d95b5001a696f513e.jpg](../icml_results/5180_Controlling%20the%20Risk%20of%20Corrupted%20Contexts%20for%20Language%20Models%20via%20Early-Exiting/images/fdd415422416e0daba13e821f730acfe24b110bf8b6df88d95b5001a696f513e.jpg)

<a id="paper-5133"></a>

## Video-BCI: Bayesian Cognitive Integration of Self-Prior Hypotheses for Video Understanding


### Images

![190be08947a6f6891656789b8dc98cc6f76ea5fc55594f274bd8e0109dd60c45.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/190be08947a6f6891656789b8dc98cc6f76ea5fc55594f274bd8e0109dd60c45.jpg)

![1eba20daf91ce7dc61179803783b37eeebcc480060e1b0d0dc6e76979430d567.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/1eba20daf91ce7dc61179803783b37eeebcc480060e1b0d0dc6e76979430d567.jpg)

![2ba5dc841432ec7029f8dd81483d48ff05f014e73fc3692d4f10e84ba8a53334.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/2ba5dc841432ec7029f8dd81483d48ff05f014e73fc3692d4f10e84ba8a53334.jpg)

![332f65a65302320ed4e701e9271583909b29a8ce08006122a68f9da17ad66111.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/332f65a65302320ed4e701e9271583909b29a8ce08006122a68f9da17ad66111.jpg)

![427a06f685a4244fc6f250d1fb14318e3b13c387333a82e9a0921b7acde7cd4f.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/427a06f685a4244fc6f250d1fb14318e3b13c387333a82e9a0921b7acde7cd4f.jpg)

![42efa9e0e14755b32e842cbd5554923331308cdae9f79ccbd83468801a462ace.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/42efa9e0e14755b32e842cbd5554923331308cdae9f79ccbd83468801a462ace.jpg)

![5a2d9582ded2a6cfc073ae9c364ff046bc16cc970601b5e82a740ef09c5fa42a.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/5a2d9582ded2a6cfc073ae9c364ff046bc16cc970601b5e82a740ef09c5fa42a.jpg)

![5ae32f04aaa4f0a428b133251ca81caba1af9b7c328b8a5a60676154d8a6fc2b.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/5ae32f04aaa4f0a428b133251ca81caba1af9b7c328b8a5a60676154d8a6fc2b.jpg)

![5d7171b48c9754fbd6589e8ee56022bdf367f55924f6f99899d15d463d71bf9a.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/5d7171b48c9754fbd6589e8ee56022bdf367f55924f6f99899d15d463d71bf9a.jpg)

![67edd683eedc874206f3865f2251f942ce0e6b4c60ec1207155f2d78ed6fe6ff.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/67edd683eedc874206f3865f2251f942ce0e6b4c60ec1207155f2d78ed6fe6ff.jpg)

![ae2d4ed9312fe861e3e8640dac69168820675f7a4c26f4a9e80a8fd7c9bc9de2.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/ae2d4ed9312fe861e3e8640dac69168820675f7a4c26f4a9e80a8fd7c9bc9de2.jpg)

![b91273beb5ca8fcba3f07d5b5509497004e40ccc0e3f3630b9a6d379bc673345.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/b91273beb5ca8fcba3f07d5b5509497004e40ccc0e3f3630b9a6d379bc673345.jpg)

![e096a2a37d00a5d56af73942b5a1b9cee59131ae4f8e18d2188aa59718c2e2dc.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/e096a2a37d00a5d56af73942b5a1b9cee59131ae4f8e18d2188aa59718c2e2dc.jpg)

![e39e2148a8d990a40e5f428059b82a7ab9112c6f5b1e30e0880baaf1a38a02d7.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/images/e39e2148a8d990a40e5f428059b82a7ab9112c6f5b1e30e0880baaf1a38a02d7.jpg)

### Tables

![1ce896e7c0e25df7f2514032a4d181a7545c4a5991f4c8627006447e48cc3f04.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/tables/1ce896e7c0e25df7f2514032a4d181a7545c4a5991f4c8627006447e48cc3f04.jpg)

![4e826833c1f0bdcb759d815cbba8b77d3d80b02fe34ea7758883fce6ddbc9e9f.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/tables/4e826833c1f0bdcb759d815cbba8b77d3d80b02fe34ea7758883fce6ddbc9e9f.jpg)

![de853ea4154dfbcccd4480bf57a4e7b79647b7674c747292918cb8d68c4a1487.jpg](../icml_results/5181_Video-BCI_%20Bayesian%20Cognitive%20Integration%20of%20Self-Prior%20Hypotheses%20for%20Video%20Understanding/tables/de853ea4154dfbcccd4480bf57a4e7b79647b7674c747292918cb8d68c4a1487.jpg)

<a id="paper-5134"></a>

## Does Reinforcement Fine-Tuning Improve Generalization of LLM Agents? An Empirical Study


### Images

![37946577fad6d3529c0c154e969eda34b984da254b8ad67aeeb334e99f855ee7.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/37946577fad6d3529c0c154e969eda34b984da254b8ad67aeeb334e99f855ee7.jpg)

![4def8ba1ed3ac01c3797efb929d34355f4d8dff9822ddcbc10f7423b563248e9.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/4def8ba1ed3ac01c3797efb929d34355f4d8dff9822ddcbc10f7423b563248e9.jpg)

![590bb77c702d6b16de984e0206dde06d7face891daf291ee78e74e5595c51f23.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/590bb77c702d6b16de984e0206dde06d7face891daf291ee78e74e5595c51f23.jpg)

![86dbb0a39459889669859cd8167e3241b8a765396460cb4b2c9fe317d76cf9e5.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/86dbb0a39459889669859cd8167e3241b8a765396460cb4b2c9fe317d76cf9e5.jpg)

![8dccd254051981578d7a3835e9c398e356e9b9e91ae5fd5c7ce1dbc0b6c73ce7.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/8dccd254051981578d7a3835e9c398e356e9b9e91ae5fd5c7ce1dbc0b6c73ce7.jpg)

![945fa07e9f8273689a8fa97abce1f640614357d5eab32b8388196ea30358e91a.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/945fa07e9f8273689a8fa97abce1f640614357d5eab32b8388196ea30358e91a.jpg)

![bada08d5cca54a2e3bec4e81428e8f6016b133e5b69876ebfdc42b4bb987668c.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/bada08d5cca54a2e3bec4e81428e8f6016b133e5b69876ebfdc42b4bb987668c.jpg)

![c0704a70bc30a4bbe842de98e88e01a27899b254a3447e8f1b5b68a6fccde64d.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/c0704a70bc30a4bbe842de98e88e01a27899b254a3447e8f1b5b68a6fccde64d.jpg)

![c92f732e2b89a2d1f64e07b473b00affdb979c9fc8ce4677578be71d631145d6.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/c92f732e2b89a2d1f64e07b473b00affdb979c9fc8ce4677578be71d631145d6.jpg)

![edbf4ee97dccd29f7bbb81ea5bfc0c88273898f63926d1ed7d677aa63ea20d99.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/edbf4ee97dccd29f7bbb81ea5bfc0c88273898f63926d1ed7d677aa63ea20d99.jpg)

![f2689a97e5347120242be24c240062ad45a87b2b427e8a0f48cb4a374a3fe0b6.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/f2689a97e5347120242be24c240062ad45a87b2b427e8a0f48cb4a374a3fe0b6.jpg)

![fd734aa12da8cf314926e91dd49ed0e824de43a3973df81b67c2008edcc02e47.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/images/fd734aa12da8cf314926e91dd49ed0e824de43a3973df81b67c2008edcc02e47.jpg)

### Tables

![0dc620e642786603d46d8c2ed9fd976f0347ac773467432f359ffe94fb02f2c3.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/0dc620e642786603d46d8c2ed9fd976f0347ac773467432f359ffe94fb02f2c3.jpg)

![224b7d210ccf5fbb9ab688ad9959b6239ebb5f43e7570a296a477421cf847e8d.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/224b7d210ccf5fbb9ab688ad9959b6239ebb5f43e7570a296a477421cf847e8d.jpg)

![229e1588a47f386b4d67fdbe651e6efd30b8114bde3cd249b1dc7b3c0c6c3290.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/229e1588a47f386b4d67fdbe651e6efd30b8114bde3cd249b1dc7b3c0c6c3290.jpg)

![2dfa632f6a33aa646eccdc7c8a32ab62bd662a814371018e4357068d0889f5c6.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/2dfa632f6a33aa646eccdc7c8a32ab62bd662a814371018e4357068d0889f5c6.jpg)

![686d87d8b4d2e9be760a233732012346e08f36710bf32f70c82b13a941bb65ab.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/686d87d8b4d2e9be760a233732012346e08f36710bf32f70c82b13a941bb65ab.jpg)

![6ace5dad62e21a2c5917b13a1130ac60e389243018353fd980e218891ef0b7c1.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/6ace5dad62e21a2c5917b13a1130ac60e389243018353fd980e218891ef0b7c1.jpg)

![6b4a98cdd8279205dd7b5b7b5bc9cf36ce0ca6481eea8572eb8a9d41716b9311.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/6b4a98cdd8279205dd7b5b7b5bc9cf36ce0ca6481eea8572eb8a9d41716b9311.jpg)

![82ec76f7961d4451219c010c6b9add5417285555daf116e61f05f9f2c294290f.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/82ec76f7961d4451219c010c6b9add5417285555daf116e61f05f9f2c294290f.jpg)

![9f7c44e7cc6693ade2a36c2ddfa8aaad38576203b86415fa4a716d3c3ff0cdc3.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/9f7c44e7cc6693ade2a36c2ddfa8aaad38576203b86415fa4a716d3c3ff0cdc3.jpg)

![e179568ce20087898ec83885430ca68d82d214f37c3d1721bc457d58f46f5948.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/e179568ce20087898ec83885430ca68d82d214f37c3d1721bc457d58f46f5948.jpg)

![f6db94c06e9183ebc0abd50f1404d0693a8e66f74eb173f9f45fd80a25ffe295.jpg](../icml_results/5182_Does%20Reinforcement%20Fine-Tuning%20Improve%20Generalization%20of%20LLM%20Agents_%20An%20Empirical%20Study/tables/f6db94c06e9183ebc0abd50f1404d0693a8e66f74eb173f9f45fd80a25ffe295.jpg)

<a id="paper-5135"></a>

## Awakening Visual Reasoning: Mitigating Post-Training Failure in Vision-Text Compression


### Images

![5fb4c2bd2c53c06b4bccbe8c27565fe3e4a20ee31654c2ca27e3e8709e5325ed.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/images/5fb4c2bd2c53c06b4bccbe8c27565fe3e4a20ee31654c2ca27e3e8709e5325ed.jpg)

![c38250470fa54e31bc1125d013524abee23cd009e7f6eb25578a4925061084de.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/images/c38250470fa54e31bc1125d013524abee23cd009e7f6eb25578a4925061084de.jpg)

### Tables

![08611d67bcaa20986a197642311ff973bf1bba2ac8c5290510bee6318a58fb01.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/08611d67bcaa20986a197642311ff973bf1bba2ac8c5290510bee6318a58fb01.jpg)

![7000b01f0342bc5bb3c22bb9914dd3fb7b15f8bbfaf3b52e5e74a12d95520aa9.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/7000b01f0342bc5bb3c22bb9914dd3fb7b15f8bbfaf3b52e5e74a12d95520aa9.jpg)

![71772ec30d1d4b5fae8e605bb7af48e8886c24ddf530b12d5c0f1b81efc6c766.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/71772ec30d1d4b5fae8e605bb7af48e8886c24ddf530b12d5c0f1b81efc6c766.jpg)

![74990236390bee1da426af5d941a57c725db1a076c8e7d664793a65d78d1d831.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/74990236390bee1da426af5d941a57c725db1a076c8e7d664793a65d78d1d831.jpg)

![9052c1e1a5bbf07e2fa01aa19587c72d468abc3a19454bbfd678120689959f8a.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/9052c1e1a5bbf07e2fa01aa19587c72d468abc3a19454bbfd678120689959f8a.jpg)

![9f3dad1a73ad470974410db4f892db4522ee2331bc2f35ba08fd314f02467351.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/9f3dad1a73ad470974410db4f892db4522ee2331bc2f35ba08fd314f02467351.jpg)

![e057f24ffcbdcf9bde64f12109a27a277fd12f8c51be6c43c5a1bb254d16eaec.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/e057f24ffcbdcf9bde64f12109a27a277fd12f8c51be6c43c5a1bb254d16eaec.jpg)

![ed22526bea71fb13e2ee55a10be88c1d09647ef452813a8efff8903be0149314.jpg](../icml_results/5183_Awakening%20Visual%20Reasoning_%20Mitigating%20Post-Training%20Failure%20in%20Vision-Text%20Compression/tables/ed22526bea71fb13e2ee55a10be88c1d09647ef452813a8efff8903be0149314.jpg)

<a id="paper-5136"></a>

## LoSA: Locality Aware Sparse Attention in Diffusion Language Models


### Images

![0a06717783925eb3550ff557009aadb55179ad82e9d45a49c717a610047e6714.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/0a06717783925eb3550ff557009aadb55179ad82e9d45a49c717a610047e6714.jpg)

![0cad8211c27a8565389ea4515eb865679126c91083a5fbbd9bd7f409f5bc605a.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/0cad8211c27a8565389ea4515eb865679126c91083a5fbbd9bd7f409f5bc605a.jpg)

![191f640a1828ca6f237725180d2668e061c3c057c89d4f3a66e7b4a891d3dfea.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/191f640a1828ca6f237725180d2668e061c3c057c89d4f3a66e7b4a891d3dfea.jpg)

![25fa2cdf3ba84473bbe2193ac3ada33a72edb44d08faef03cb7f0569a422ef1d.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/25fa2cdf3ba84473bbe2193ac3ada33a72edb44d08faef03cb7f0569a422ef1d.jpg)

![40e27f5f0d0d7f6f95e94b30362dbd91f1fdfa18452a0d2af520182f1c942b81.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/40e27f5f0d0d7f6f95e94b30362dbd91f1fdfa18452a0d2af520182f1c942b81.jpg)

![56877652c0663651badf76e67706850ca391cbe98c380ca3cb87ddc53ce53daa.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/56877652c0663651badf76e67706850ca391cbe98c380ca3cb87ddc53ce53daa.jpg)

![59c105c247a374e34916e7bfbfa58783f60ee769396a41d130ac02c23386d62e.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/59c105c247a374e34916e7bfbfa58783f60ee769396a41d130ac02c23386d62e.jpg)

![b70fd1a5269bda370d1d15e5ff1b1439e42ee3d2ae59d272dd44277daead58d2.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/b70fd1a5269bda370d1d15e5ff1b1439e42ee3d2ae59d272dd44277daead58d2.jpg)

![d6e523434766bec396793ca6a3794e0bf5c8a7bb1e0e751931cef9b6f8427efc.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/d6e523434766bec396793ca6a3794e0bf5c8a7bb1e0e751931cef9b6f8427efc.jpg)

![e21a9e1f4951b57800d5e0fffcc555f943b090dcafd7d3fe1f1d90ce9c4e44e5.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/e21a9e1f4951b57800d5e0fffcc555f943b090dcafd7d3fe1f1d90ce9c4e44e5.jpg)

![f7eebeb3f88a207e3cefa2687c8265bf4b344bf14e8783f630c21b55a67a18aa.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/images/f7eebeb3f88a207e3cefa2687c8265bf4b344bf14e8783f630c21b55a67a18aa.jpg)

### Tables

![1f0ee58b48e8d8ce14c9e2ef15d1246850f01a658ec05d3992adf980fea5fcd6.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/1f0ee58b48e8d8ce14c9e2ef15d1246850f01a658ec05d3992adf980fea5fcd6.jpg)

![69d474dacd6731caa98caf4493b757d5f1b278539ddbb9ef60df964997231b5b.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/69d474dacd6731caa98caf4493b757d5f1b278539ddbb9ef60df964997231b5b.jpg)

![74187f73375ecd1bf2d09958c9063b73f50f1d72412f47c24ed847da1b461ff9.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/74187f73375ecd1bf2d09958c9063b73f50f1d72412f47c24ed847da1b461ff9.jpg)

![80cdb97258edfd40ac27c1f09bec242ccc142e0433f51d2e7f7b61d638b3f09d.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/80cdb97258edfd40ac27c1f09bec242ccc142e0433f51d2e7f7b61d638b3f09d.jpg)

![c92934c13d4a778d79dc6b2b1d8894003452b440c606a2e2b617dcc85be71418.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/c92934c13d4a778d79dc6b2b1d8894003452b440c606a2e2b617dcc85be71418.jpg)

![d53866816c16b69cff9f994ad65d20241f5cc21c61c39da3e651b1826d3628ae.jpg](../icml_results/5184_LoSA_%20Locality%20Aware%20Sparse%20Attention%20in%20Diffusion%20Language%20Models/tables/d53866816c16b69cff9f994ad65d20241f5cc21c61c39da3e651b1826d3628ae.jpg)

<a id="paper-5137"></a>

## Large Vision-Language Models Get Lost in Attention


### Images

![083e7e1cfcfa12a611edb1cc201a69f751504b6c132eae376a09f977587cfaa3.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/083e7e1cfcfa12a611edb1cc201a69f751504b6c132eae376a09f977587cfaa3.jpg)

![3c1923174d4a7a1231692155b3070d24e621562ddc9828270d0e2dccecd54025.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/3c1923174d4a7a1231692155b3070d24e621562ddc9828270d0e2dccecd54025.jpg)

![3e7a6026a724b0e052760a3aa75b6ce90810cfbdc65d3a1c3303ba3f80079654.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/3e7a6026a724b0e052760a3aa75b6ce90810cfbdc65d3a1c3303ba3f80079654.jpg)

![445df0af677eb686056fa69eb87e9b9a978b715cf35691b1e0ab6b7f485c02fc.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/445df0af677eb686056fa69eb87e9b9a978b715cf35691b1e0ab6b7f485c02fc.jpg)

![585153e3a6d854724778d9b84746e6f54ba9cf54a390812e30525d65799a5311.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/585153e3a6d854724778d9b84746e6f54ba9cf54a390812e30525d65799a5311.jpg)

![6eca9a4a4079db0ebdb2e2021f432daa5cd0bb4ad6faada2241ffce135233766.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/6eca9a4a4079db0ebdb2e2021f432daa5cd0bb4ad6faada2241ffce135233766.jpg)

![7e24c96ebe45ef75d48060b4d61f6217c39f4e029996812bc6a8848d1c6ae481.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/7e24c96ebe45ef75d48060b4d61f6217c39f4e029996812bc6a8848d1c6ae481.jpg)

![9679a5324dd3c492b2b70a99f1546e821beda227b1ffe1f2be81cc503cf84ad7.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/9679a5324dd3c492b2b70a99f1546e821beda227b1ffe1f2be81cc503cf84ad7.jpg)

![99310331320b318e476b94b94d45d2284716eedb0149f9d686b4674ea8961228.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/99310331320b318e476b94b94d45d2284716eedb0149f9d686b4674ea8961228.jpg)

![9c812381843f879c034aeae600696c1a0ce8d18d8b3063db0d9551f831304a84.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/images/9c812381843f879c034aeae600696c1a0ce8d18d8b3063db0d9551f831304a84.jpg)

### Tables

![40301d23c644486d81ded8cc38da452ac468f8e04fe56978f48ec836cf2441fb.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/40301d23c644486d81ded8cc38da452ac468f8e04fe56978f48ec836cf2441fb.jpg)

![5837b2ce49257ebb4439174e699f94103e8201022891b63e56da2b02ae04bd36.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/5837b2ce49257ebb4439174e699f94103e8201022891b63e56da2b02ae04bd36.jpg)

![5abc91880fe0fe151ed47d6482f9f2956d22f69c987420cdb030ec43530820e5.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/5abc91880fe0fe151ed47d6482f9f2956d22f69c987420cdb030ec43530820e5.jpg)

![6dd6c3fc3f086b7d30064c8ef3fe25d4b375f6888133b3edc358e170cb6a2502.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/6dd6c3fc3f086b7d30064c8ef3fe25d4b375f6888133b3edc358e170cb6a2502.jpg)

![8d23363278a7bb77da2b6f7e209d13f7d3e875b234b7a84bab84e5656ac3f253.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/8d23363278a7bb77da2b6f7e209d13f7d3e875b234b7a84bab84e5656ac3f253.jpg)

![8e543540ceb97ffce66a909b0c57bdd777fb82a9d363469cfc81db08f2548b8d.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/8e543540ceb97ffce66a909b0c57bdd777fb82a9d363469cfc81db08f2548b8d.jpg)

![ae3744c9752838b96e93b9f01a3ac633446ebbef1b93b01ff125fa9255f0c7dc.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/ae3744c9752838b96e93b9f01a3ac633446ebbef1b93b01ff125fa9255f0c7dc.jpg)

![c0d28c83c4bf6cc3d2cdd8aa3aa269618b7a0b53d8a8a84844fc67d7547c8b7d.jpg](../icml_results/5185_Large%20Vision-Language%20Models%20Get%20Lost%20in%20Attention/tables/c0d28c83c4bf6cc3d2cdd8aa3aa269618b7a0b53d8a8a84844fc67d7547c8b7d.jpg)

<a id="paper-5138"></a>

## Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization


### Images

![0aa46af16acd3db4b10bac5c74005c28feef6d89040acca4bb1193c266eef85f.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/0aa46af16acd3db4b10bac5c74005c28feef6d89040acca4bb1193c266eef85f.jpg)

![1ce6aaa2a12cbd84fcd991028a322e1bc98f5dc003a9bbb58d5904b41f79f44b.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/1ce6aaa2a12cbd84fcd991028a322e1bc98f5dc003a9bbb58d5904b41f79f44b.jpg)

![1e823df7f10d910c64cc39e40734e7beaefaa190f44b7287d35ebfe7a6617025.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/1e823df7f10d910c64cc39e40734e7beaefaa190f44b7287d35ebfe7a6617025.jpg)

![440995ca1732a58a1286372448fdd96a35812e665f7fb8a87732db222bc5a823.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/440995ca1732a58a1286372448fdd96a35812e665f7fb8a87732db222bc5a823.jpg)

![6f509609d66980ff4c21e90b0c7cd879113e326d5b9271365dadaac25d29eecb.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/6f509609d66980ff4c21e90b0c7cd879113e326d5b9271365dadaac25d29eecb.jpg)

![c1d5d72e03d7f49736b28de7ad44919d78d5f3a539deb5f0dec64aebae169918.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/c1d5d72e03d7f49736b28de7ad44919d78d5f3a539deb5f0dec64aebae169918.jpg)

![dbf3dc236e31a8d6df052d2c5cae67ba99a7d44b112f02351c95ce289bcc20e6.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/images/dbf3dc236e31a8d6df052d2c5cae67ba99a7d44b112f02351c95ce289bcc20e6.jpg)

### Tables

![1bb980658409a8eeceba743ebca3ba280af0011b7bf1f9625961192be2d99a87.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/1bb980658409a8eeceba743ebca3ba280af0011b7bf1f9625961192be2d99a87.jpg)

![1e80a6300ee16795ee3b5d1a21360d000dc70eba87ba0d2b4483c31f43c92925.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/1e80a6300ee16795ee3b5d1a21360d000dc70eba87ba0d2b4483c31f43c92925.jpg)

![3f8971da4cf122d1448ba9171c91a295159dd20f8e7759b0d64c4323de89c7be.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/3f8971da4cf122d1448ba9171c91a295159dd20f8e7759b0d64c4323de89c7be.jpg)

![a85b9ef733fa9302a9b2f38cb3e501baf2d4eb6812bee442d147141058258d99.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/a85b9ef733fa9302a9b2f38cb3e501baf2d4eb6812bee442d147141058258d99.jpg)

![bea6aa8de3feab1d31f72df0c55d423ff011624c632f705ba9cd67063d271d7d.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/bea6aa8de3feab1d31f72df0c55d423ff011624c632f705ba9cd67063d271d7d.jpg)

![c2e095598520d08a47a11860e8fe626fef9be7dc832dbd611fd1c6a1c4e1551c.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/c2e095598520d08a47a11860e8fe626fef9be7dc832dbd611fd1c6a1c4e1551c.jpg)

![d0028b6502e42d93ca083e518cb8cacf1765732899435f8d37327e16e1dd5e75.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/d0028b6502e42d93ca083e518cb8cacf1765732899435f8d37327e16e1dd5e75.jpg)

![ee7b462cdd8fd9927b8cffd9771620c2189d72849f48d9743072ae22fd1a9556.jpg](../icml_results/5186_Quant%20VideoGen_%20Auto-Regressive%20Long%20Video%20Generation%20via%202-Bit%20KV-Cache%20Quantization/tables/ee7b462cdd8fd9927b8cffd9771620c2189d72849f48d9743072ae22fd1a9556.jpg)

<a id="paper-5139"></a>

## Spectral Bridge Variational Inference: Dynamic LoRA via Bures-Wasserstein Gradient Flows


### Images

![1c8adf3b959d535c03162fc2f7a20239d66d22905cbcc85e051dcfe9947f4123.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/images/1c8adf3b959d535c03162fc2f7a20239d66d22905cbcc85e051dcfe9947f4123.jpg)

![6407985c58eeef2ade4e762e71c153243adcad19f8f04e8deea21c9e08c25a2c.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/images/6407985c58eeef2ade4e762e71c153243adcad19f8f04e8deea21c9e08c25a2c.jpg)

![c21037fa8d29d9bfd18d7656a6a9a4efc4d5f836270faf5c72f614e3477b7f85.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/images/c21037fa8d29d9bfd18d7656a6a9a4efc4d5f836270faf5c72f614e3477b7f85.jpg)

### Tables

![1b0aca64370fee8ab828ed0f44b489b409731633beaf12570b2e2c7e0bfad6ab.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/1b0aca64370fee8ab828ed0f44b489b409731633beaf12570b2e2c7e0bfad6ab.jpg)

![32cd9fc5e496cd593adc110432c3015e5af19cf387cdc19255994d07125d8ecf.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/32cd9fc5e496cd593adc110432c3015e5af19cf387cdc19255994d07125d8ecf.jpg)

![3ca1e75887d68a2391743d79b477abd71c62a771069ba7b62237f7721deb7558.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/3ca1e75887d68a2391743d79b477abd71c62a771069ba7b62237f7721deb7558.jpg)

![430568223a459a571996930e23fdf4c0315803d0274fd73a919c392d6b07605f.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/430568223a459a571996930e23fdf4c0315803d0274fd73a919c392d6b07605f.jpg)

![558b556c3ebf7a6547f9b76c870f23bd497f157dfd08d4652bae0b18da0b7dff.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/558b556c3ebf7a6547f9b76c870f23bd497f157dfd08d4652bae0b18da0b7dff.jpg)

![562869e576d68c3bc0dff6df75e317577ee12a37d3293f74769a12f6cb89adcc.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/562869e576d68c3bc0dff6df75e317577ee12a37d3293f74769a12f6cb89adcc.jpg)

![5b0eb1f51e9e7a189af32ff2cc88f793cc8d6dfd928049a80c643b51a9af3f9c.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/5b0eb1f51e9e7a189af32ff2cc88f793cc8d6dfd928049a80c643b51a9af3f9c.jpg)

![741cfe09e71d760625eab5ffb8893dc0c8db4ca58d00c8c63e42ca611788e37c.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/741cfe09e71d760625eab5ffb8893dc0c8db4ca58d00c8c63e42ca611788e37c.jpg)

![8efb8fd4ff25f054e57aa0f1bc63a420318647bf1066f552b7e33e29a276e171.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/8efb8fd4ff25f054e57aa0f1bc63a420318647bf1066f552b7e33e29a276e171.jpg)

![99db858a5e7ed664c56b50daeacbd8dbd06ccff341127f0c62535fafea5cd9b1.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/99db858a5e7ed664c56b50daeacbd8dbd06ccff341127f0c62535fafea5cd9b1.jpg)

![9c88218eb33bf1dd2e598596f87ff2a74b9fa89b9b0fdfa3f84f2ccefc76ed46.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/9c88218eb33bf1dd2e598596f87ff2a74b9fa89b9b0fdfa3f84f2ccefc76ed46.jpg)

![db186f93ccd88a331d63fd1f43833b612009c4c136402cc46dcc2a7654fa4783.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/db186f93ccd88a331d63fd1f43833b612009c4c136402cc46dcc2a7654fa4783.jpg)

![db829ce6e5fc104fd8b9d64a5d897cca9415d54b2417803ce907ca241e153caf.jpg](../icml_results/5187_Spectral%20Bridge%20Variational%20Inference_%20Dynamic%20LoRA%20via%20Bures-Wasserstein%20Gradient%20Flows/tables/db829ce6e5fc104fd8b9d64a5d897cca9415d54b2417803ce907ca241e153caf.jpg)

<a id="paper-5140"></a>

## Do Neural Operators Forget Geometry? The Forgetting Hypothesis in Deep Operator Learning


### Images

![068d7d45e216a9166f3f0675fda6d5b55a7655de2296fb92a6cc25e38727109e.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/068d7d45e216a9166f3f0675fda6d5b55a7655de2296fb92a6cc25e38727109e.jpg)

![0a9261dd06788a8800ed47e2f77dc86426f88f3985d8dadc9704d1767dc3260a.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/0a9261dd06788a8800ed47e2f77dc86426f88f3985d8dadc9704d1767dc3260a.jpg)

![2ba45f640d846dbddf3af557bd812eab0c311aaf91e37b31a2ea9ec59bdb3ace.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/2ba45f640d846dbddf3af557bd812eab0c311aaf91e37b31a2ea9ec59bdb3ace.jpg)

![2d5546583ea9540817473c564e7d7500a8108a168bc81e99d21c15e396d7e97b.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/2d5546583ea9540817473c564e7d7500a8108a168bc81e99d21c15e396d7e97b.jpg)

![54fa1097ee4222927510c6108fcb565954ee2615fd0f61fb27b55737dae9c297.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/54fa1097ee4222927510c6108fcb565954ee2615fd0f61fb27b55737dae9c297.jpg)

![6753e61c3c146c29f55c528e7e4527f33b73b8bbecd66e50dfbe3d5dcfa268cd.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/6753e61c3c146c29f55c528e7e4527f33b73b8bbecd66e50dfbe3d5dcfa268cd.jpg)

![786f8d7c65311023f81b82016b37c4402efc6a4328bfa81fe1aa24034a1fb405.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/786f8d7c65311023f81b82016b37c4402efc6a4328bfa81fe1aa24034a1fb405.jpg)

![8f1bace81eec7cff964053276a589ffd91941863c557c9fe5340b31bcb184cc6.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/8f1bace81eec7cff964053276a589ffd91941863c557c9fe5340b31bcb184cc6.jpg)

![92939b4432372da72f36d9890c68dbb0946705b84bf82c65dcb2eca82b19eae2.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/92939b4432372da72f36d9890c68dbb0946705b84bf82c65dcb2eca82b19eae2.jpg)

![ac3e9237b1cba7caad1945d33fefaa00f087d80628d3b9b7b16128d2bfba1c36.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/ac3e9237b1cba7caad1945d33fefaa00f087d80628d3b9b7b16128d2bfba1c36.jpg)

![bb0850e3b0975177c6c520a6a3c5356aff8ac6ca5ae0353f0b4ad03248134cfe.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/bb0850e3b0975177c6c520a6a3c5356aff8ac6ca5ae0353f0b4ad03248134cfe.jpg)

![c8686fa55c27789be511ec7949a53978bf4b20f633137715e3dacbb6955f5c2d.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/c8686fa55c27789be511ec7949a53978bf4b20f633137715e3dacbb6955f5c2d.jpg)

![d172fb26efe57985178cc20ac9ed3f07427f6ea17edb87ef5708ef25fc3304f8.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/d172fb26efe57985178cc20ac9ed3f07427f6ea17edb87ef5708ef25fc3304f8.jpg)

![db5b7f6c75288501e2c1cb146764d48f9b4a707de7b8717d38020f8b46ce6ea0.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/db5b7f6c75288501e2c1cb146764d48f9b4a707de7b8717d38020f8b46ce6ea0.jpg)

![df90d1edf2c8e35dc6aac4191ea8c8f4f6a67798e490d93101169d67d5671b72.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/df90d1edf2c8e35dc6aac4191ea8c8f4f6a67798e490d93101169d67d5671b72.jpg)

![e6140805c1e3b8f9ae1192c7e4f10fed2c1317169ed5875f7c680aad9a0d98d9.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/e6140805c1e3b8f9ae1192c7e4f10fed2c1317169ed5875f7c680aad9a0d98d9.jpg)

![faa5ab1034817502c2d74ac51158c31901686b9cf292a85a430dc620441fc569.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/images/faa5ab1034817502c2d74ac51158c31901686b9cf292a85a430dc620441fc569.jpg)

### Tables

![165f43be06eb86f4378d3cd33a85ea2f3f5795788a0e07d4d51aa743fd7fdab8.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/165f43be06eb86f4378d3cd33a85ea2f3f5795788a0e07d4d51aa743fd7fdab8.jpg)

![3f79ef32b172d15a595f8a1c4f6202dfe6103b79f6820ce7afba2b604a7c01d1.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/3f79ef32b172d15a595f8a1c4f6202dfe6103b79f6820ce7afba2b604a7c01d1.jpg)

![4c8f826374da86f59c9b14e9cc44008c5213111f4ea78ba1433b76100eb46b2f.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/4c8f826374da86f59c9b14e9cc44008c5213111f4ea78ba1433b76100eb46b2f.jpg)

![51f6a06a41f2d5d2e94a69b6fef25c52cb2c7438d967700506f32bb58ba7175b.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/51f6a06a41f2d5d2e94a69b6fef25c52cb2c7438d967700506f32bb58ba7175b.jpg)

![725b92e75092f20d25943787466f86e7c0e28b2796e0576f8861f1a594cabf5f.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/725b92e75092f20d25943787466f86e7c0e28b2796e0576f8861f1a594cabf5f.jpg)

![b3109ce7570ddeac85de1a18aa498050efaecd2269f93ec33669caac21f52cc1.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/b3109ce7570ddeac85de1a18aa498050efaecd2269f93ec33669caac21f52cc1.jpg)

![d799702b25d220a5459ea507d8054995c42c51deb6b473770224274b7f5cf123.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/d799702b25d220a5459ea507d8054995c42c51deb6b473770224274b7f5cf123.jpg)

![e066dc3e6dff5650cfd062fb6c7675ae5071ec09555318377d9e2d41b0e80d18.jpg](../icml_results/5188_Do%20Neural%20Operators%20Forget%20Geometry_%20The%20Forgetting%20Hypothesis%20in%20Deep%20Operator%20Learning/tables/e066dc3e6dff5650cfd062fb6c7675ae5071ec09555318377d9e2d41b0e80d18.jpg)

<a id="paper-5141"></a>

## Factored Classifier-Free Guidance


### Images

![015748bef9b2337153a375eb8b94fc0dc47e215416cc571602572802d4a1e6fd.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/015748bef9b2337153a375eb8b94fc0dc47e215416cc571602572802d4a1e6fd.jpg)

![04b30a70c560329ed5793c829013481ed2dd8c54b9b7a6eddeae72fa2415b85b.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/04b30a70c560329ed5793c829013481ed2dd8c54b9b7a6eddeae72fa2415b85b.jpg)

![09729f64445810b18fcf090b290507a76fea01a7d38117d2ea2542291345956e.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/09729f64445810b18fcf090b290507a76fea01a7d38117d2ea2542291345956e.jpg)

![1223740521a75897d9c628b5b3d74815a0edc076219b8aca2a5ae23927f8cf7f.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/1223740521a75897d9c628b5b3d74815a0edc076219b8aca2a5ae23927f8cf7f.jpg)

![2129f7ff071d22f19362f4edbaccb167a6643adbb9ef38606a00878c6e886e88.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/2129f7ff071d22f19362f4edbaccb167a6643adbb9ef38606a00878c6e886e88.jpg)

![23bf6ce2cb6bc41129acf662d19921e16d7de9b209071a5cb83f647640efb9f5.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/23bf6ce2cb6bc41129acf662d19921e16d7de9b209071a5cb83f647640efb9f5.jpg)

![2659e2f0d2bc703d59aafe5437fd63b31a3de2f466c8a3f59100be0594bb3751.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/2659e2f0d2bc703d59aafe5437fd63b31a3de2f466c8a3f59100be0594bb3751.jpg)

![28c227224adec9f9fac1fdbbdebd0ad69cde860b81e98ee3492920ceafe19437.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/28c227224adec9f9fac1fdbbdebd0ad69cde860b81e98ee3492920ceafe19437.jpg)

![2e6c4cd9ed3aa257b35fa63050f5320ef61b11eadf06070c744ca486b7183776.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/2e6c4cd9ed3aa257b35fa63050f5320ef61b11eadf06070c744ca486b7183776.jpg)

![3153b3d82cfc821402da66bc4e848905f14b87197563ad93844ac38756f4bf22.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/3153b3d82cfc821402da66bc4e848905f14b87197563ad93844ac38756f4bf22.jpg)

![329e0e41f6dbde6c94f91c3e67c0f233f600327f9a2227dde9bf209a0fab9a81.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/329e0e41f6dbde6c94f91c3e67c0f233f600327f9a2227dde9bf209a0fab9a81.jpg)

![449f5bad687e3e10a6159952dfcfd8bd6ea9a86611829ecb95a29ce2e4cbba87.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/449f5bad687e3e10a6159952dfcfd8bd6ea9a86611829ecb95a29ce2e4cbba87.jpg)

![4d6709608fcef1ab172e0a249bbac104c9e9132ad475ba1f5126e98b63c25f0d.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/4d6709608fcef1ab172e0a249bbac104c9e9132ad475ba1f5126e98b63c25f0d.jpg)

![4faa81e7d07904c41512b0514152bd295fc5873ac800361459af082e1c1e2fe7.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/4faa81e7d07904c41512b0514152bd295fc5873ac800361459af082e1c1e2fe7.jpg)

![557fb93e08409afe098dcc2698e14c4dd72d93a040a19ad0e7c2a312175f486f.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/557fb93e08409afe098dcc2698e14c4dd72d93a040a19ad0e7c2a312175f486f.jpg)

![61cade504cc0f76d5597b19572f800df9149f9a4846987f0b4ff812e4fcb8a69.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/61cade504cc0f76d5597b19572f800df9149f9a4846987f0b4ff812e4fcb8a69.jpg)

![65ea1e9063f7ce6ae2cc7ba5f475a0c44f975ee74198a8ce2ab09e0ef9ef7043.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/65ea1e9063f7ce6ae2cc7ba5f475a0c44f975ee74198a8ce2ab09e0ef9ef7043.jpg)

![6c52c5de2146bccdd47a6406e5d7e090bd6649f6e82e1333e5114935ab67d50f.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/6c52c5de2146bccdd47a6406e5d7e090bd6649f6e82e1333e5114935ab67d50f.jpg)

![78f301b75b6555698f96e37e7cb3b8738f7af368f9cfe37fbd8abbac08175777.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/78f301b75b6555698f96e37e7cb3b8738f7af368f9cfe37fbd8abbac08175777.jpg)

![7e9292832378f9e2ca0717034d050a8ed4e79763d376a26a4c3a30dbcae15096.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/7e9292832378f9e2ca0717034d050a8ed4e79763d376a26a4c3a30dbcae15096.jpg)

![81ff2c896907ffed369a692057394ab2f20bff4b7e3325a46687bc6a8296e718.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/81ff2c896907ffed369a692057394ab2f20bff4b7e3325a46687bc6a8296e718.jpg)

![82b113b158094c45219843fef98b9e771c2f8924f542d40707e736565d0f2724.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/82b113b158094c45219843fef98b9e771c2f8924f542d40707e736565d0f2724.jpg)

![87abd9005e46e14e53bf2f5dce9d5c4254a04e90e705ccc8d0e7de9cfd338f4f.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/87abd9005e46e14e53bf2f5dce9d5c4254a04e90e705ccc8d0e7de9cfd338f4f.jpg)

![8b68845d670ad1c69aa2c4168509e9480ac5ae8342a6cb8348e310b9443be155.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/8b68845d670ad1c69aa2c4168509e9480ac5ae8342a6cb8348e310b9443be155.jpg)

![8ebc188f5f26951a631c6d3a3f10df0aea6b1c51903381ca7ddafc82f7bc0262.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/8ebc188f5f26951a631c6d3a3f10df0aea6b1c51903381ca7ddafc82f7bc0262.jpg)

![91c872afba6e7b4a5ccc4f2c489b3d79c82ff54a5e8abffe3fc5bd58df34e965.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/91c872afba6e7b4a5ccc4f2c489b3d79c82ff54a5e8abffe3fc5bd58df34e965.jpg)

![932143987e5d39a88cccd498e606cb96ca71df4e7232fb6b297c78e1c3014817.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/932143987e5d39a88cccd498e606cb96ca71df4e7232fb6b297c78e1c3014817.jpg)

![9444de84d7bd2f25bd9613555fa25872ee45e2b5de1170743c7cb702f554653e.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/9444de84d7bd2f25bd9613555fa25872ee45e2b5de1170743c7cb702f554653e.jpg)

![9ac5c2be82bf7d391b02b0d653f68ce2933277b954ec1b8b3dd85978296ab85b.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/9ac5c2be82bf7d391b02b0d653f68ce2933277b954ec1b8b3dd85978296ab85b.jpg)

![a116cf068f06acd24b5dc602a31529a4f7f85c1000dc20594ac474101fd930ae.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/a116cf068f06acd24b5dc602a31529a4f7f85c1000dc20594ac474101fd930ae.jpg)

![a866a526a73e985066cb2319e8e3d5dc6dd802c1fd2e90c21294188a38585039.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/a866a526a73e985066cb2319e8e3d5dc6dd802c1fd2e90c21294188a38585039.jpg)

![b32dda0db45ac61b962ea988dd56fe61cbaededa0886770ac33b6f03e9eb390d.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/b32dda0db45ac61b962ea988dd56fe61cbaededa0886770ac33b6f03e9eb390d.jpg)

![b505b76f0a19550bec6eddbb9038bdea01b7b6028555a26615223acf8cc89b49.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/b505b76f0a19550bec6eddbb9038bdea01b7b6028555a26615223acf8cc89b49.jpg)

![b688a1b8c46fa5d9cfcf7f2080bb18fe96b265e4cec5ac984da3cc048fc2d47b.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/b688a1b8c46fa5d9cfcf7f2080bb18fe96b265e4cec5ac984da3cc048fc2d47b.jpg)

![b875a3d701bc5dce1f287a12663a91b3a17eea6153ce4380cbfa58b2701483ca.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/b875a3d701bc5dce1f287a12663a91b3a17eea6153ce4380cbfa58b2701483ca.jpg)

![ba98a94cb45deac6a11a81e9d40bfa699f64f8acab5824e904315fd7d7e59765.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/ba98a94cb45deac6a11a81e9d40bfa699f64f8acab5824e904315fd7d7e59765.jpg)

![c149db27655befb106deff06d11b5c63349d5f59cd426352bfafe18198e0df60.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/c149db27655befb106deff06d11b5c63349d5f59cd426352bfafe18198e0df60.jpg)

![c44e9c8e115e3644c4f2b58f2eb1c145dcb4eda357927ec465400fdd87f388a4.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/c44e9c8e115e3644c4f2b58f2eb1c145dcb4eda357927ec465400fdd87f388a4.jpg)

![c801a968b6ab02aa9d821a4dc9e13117c47321f67d099e9ff6dd2e47029452e3.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/c801a968b6ab02aa9d821a4dc9e13117c47321f67d099e9ff6dd2e47029452e3.jpg)

![cab2a692995f0ded7e6578c8541270ff8b2c574be66765c14e05f0415aaf628f.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/cab2a692995f0ded7e6578c8541270ff8b2c574be66765c14e05f0415aaf628f.jpg)

![d3b23dfda06c89bbf9f40c7875b1317b2d8b2cdc8fd50d3a5558bb2ec72effdb.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/d3b23dfda06c89bbf9f40c7875b1317b2d8b2cdc8fd50d3a5558bb2ec72effdb.jpg)

![dd56f511a7d94107c6f25bd2010e3baf6dbbcb6b6d4f8d6c24be5e7928bb4580.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/dd56f511a7d94107c6f25bd2010e3baf6dbbcb6b6d4f8d6c24be5e7928bb4580.jpg)

![e2717abb10b7c1a04a78e96536bb91a549f8138a040baadc7a29e54d41f39292.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/e2717abb10b7c1a04a78e96536bb91a549f8138a040baadc7a29e54d41f39292.jpg)

![e2838cd9e2b37a724cc3af942f635f2b51161eb65e65b313152d2316bded49c5.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/e2838cd9e2b37a724cc3af942f635f2b51161eb65e65b313152d2316bded49c5.jpg)

![eb40e48446cd609cdb9aba7911557d6d875f2c28403963fc4d999cc9628fdeb3.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/eb40e48446cd609cdb9aba7911557d6d875f2c28403963fc4d999cc9628fdeb3.jpg)

![ef2a297e09455897414120440210be30c9780d16d8b5e8459a458b2afac70633.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/ef2a297e09455897414120440210be30c9780d16d8b5e8459a458b2afac70633.jpg)

![f300471f1d7f35046a745af66e64ba216000dd7095cd07028be80fac02d797cd.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/f300471f1d7f35046a745af66e64ba216000dd7095cd07028be80fac02d797cd.jpg)

![f455b1292506f32777b1d633d35dfde57e57c6237f723682862eaaf9b04b4a89.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/f455b1292506f32777b1d633d35dfde57e57c6237f723682862eaaf9b04b4a89.jpg)

![f7326508b8f83688091f5b185d11f2786166fd994c65e349405760811d051acb.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/f7326508b8f83688091f5b185d11f2786166fd994c65e349405760811d051acb.jpg)

![f96374a365f653038d50707f2be206d5b346d68d429574b03c856e9e0c26b748.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/f96374a365f653038d50707f2be206d5b346d68d429574b03c856e9e0c26b748.jpg)

![ffebf92e864bb48cc2caf528f5b90367fde5b6c88f53f3373c2391656fad7191.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/images/ffebf92e864bb48cc2caf528f5b90367fde5b6c88f53f3373c2391656fad7191.jpg)

### Tables

![0c5af370590706924ca0e93a820a0fcc629c32c51b59b48d4ce3517e3180ffc9.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/0c5af370590706924ca0e93a820a0fcc629c32c51b59b48d4ce3517e3180ffc9.jpg)

![1cb9698976736ae5ac6f9441a5015c1646147495ef261997f259a7d9a157f847.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/1cb9698976736ae5ac6f9441a5015c1646147495ef261997f259a7d9a157f847.jpg)

![368c415aeb892baada07a611fd979c03fa8605a60860a9a789625430433a1442.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/368c415aeb892baada07a611fd979c03fa8605a60860a9a789625430433a1442.jpg)

![3c5f6cca0e8da92b85ed59e1a0da6087b3d7773437544dd802f2ba610ee5a205.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/3c5f6cca0e8da92b85ed59e1a0da6087b3d7773437544dd802f2ba610ee5a205.jpg)

![501aaae3b93252c3d6e154694d6f5029d764d34a1a4e4683d02e3f1398938a88.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/501aaae3b93252c3d6e154694d6f5029d764d34a1a4e4683d02e3f1398938a88.jpg)

![583d3e35dc82d517ac548fa37500c07d53b13b170ded3d23eabe8ada0febb49e.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/583d3e35dc82d517ac548fa37500c07d53b13b170ded3d23eabe8ada0febb49e.jpg)

![5e3b4c322eff04cb52a366bae3c762a3ee26abbacdc16b2b864228c7b8adc9c6.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/5e3b4c322eff04cb52a366bae3c762a3ee26abbacdc16b2b864228c7b8adc9c6.jpg)

![6eb4db65fffb6e58692993d8c09b0260fccd5e5c932c88b2b1af1ba783ec74c6.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/6eb4db65fffb6e58692993d8c09b0260fccd5e5c932c88b2b1af1ba783ec74c6.jpg)

![71ed4ad0bf69b259bc697fc3f9d53b0bcb04e5c104a0b9a5248d964febb76501.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/71ed4ad0bf69b259bc697fc3f9d53b0bcb04e5c104a0b9a5248d964febb76501.jpg)

![888181d9cdc6617dcea67bfe8a0008e4bece6eb569a92b4a6d886bb4868fc374.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/888181d9cdc6617dcea67bfe8a0008e4bece6eb569a92b4a6d886bb4868fc374.jpg)

![8b6ac5f91aad5ab908f8f0d67e5c9cbb5f4777f29ff4b44b6644547b3d27cfff.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/8b6ac5f91aad5ab908f8f0d67e5c9cbb5f4777f29ff4b44b6644547b3d27cfff.jpg)

![bf6fc43e37b56078f41422b357b73c7c08534d204044f3fcce5b2813fb30a2f7.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/bf6fc43e37b56078f41422b357b73c7c08534d204044f3fcce5b2813fb30a2f7.jpg)

![e059783807af09c330d6708501a7497bfcb68a296d09f4ad445270c11e5fed7e.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/e059783807af09c330d6708501a7497bfcb68a296d09f4ad445270c11e5fed7e.jpg)

![e09407275aca69c343a94ce2897681f7ed8676c8664f7f7ffbc5da0d51e27cfa.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/e09407275aca69c343a94ce2897681f7ed8676c8664f7f7ffbc5da0d51e27cfa.jpg)

![e15a958b86d4ce9683a85d857b39e2fd23c6f0543b81f3c1e91b4fa86e7f64c3.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/e15a958b86d4ce9683a85d857b39e2fd23c6f0543b81f3c1e91b4fa86e7f64c3.jpg)

![e71803fcb7df1154562c6f989cc1c084bfa475c8d4bccf1f579f596cdeb481a0.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/e71803fcb7df1154562c6f989cc1c084bfa475c8d4bccf1f579f596cdeb481a0.jpg)

![e912f55e4d4f0ab13dde482559a551cfb6b139c0ceb44aee286fe3d5a4b05932.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/e912f55e4d4f0ab13dde482559a551cfb6b139c0ceb44aee286fe3d5a4b05932.jpg)

![fcec2f43fdb197ba44650a90ad87fddd45bba3eb344419f3f938a3a4abde8540.jpg](../icml_results/5189_Factored%20Classifier-Free%20Guidance/tables/fcec2f43fdb197ba44650a90ad87fddd45bba3eb344419f3f938a3a4abde8540.jpg)

<a id="paper-5142"></a>

## EchoAttention: Exploiting Token-Pair Redundancy and Frame-Block Similarity for Efficient Video Generation


### Images

![096bb3e11748d4f488295ab2237165c30ca7bc40269b7cc7452fc090309d94bf.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/096bb3e11748d4f488295ab2237165c30ca7bc40269b7cc7452fc090309d94bf.jpg)

![154fa4c29041d86aaa1f63f41e5c4df62c487205770be02142daa3f2bb51e5fb.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/154fa4c29041d86aaa1f63f41e5c4df62c487205770be02142daa3f2bb51e5fb.jpg)

![1be641dd980cd024d009d662cf619dbb37b30b9bc09904a62872a3f3edc79e30.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/1be641dd980cd024d009d662cf619dbb37b30b9bc09904a62872a3f3edc79e30.jpg)

![33a13814b9b88cf9d79811a55df90fbe29efdbbf81f7e800882e9e8c0cd49c86.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/33a13814b9b88cf9d79811a55df90fbe29efdbbf81f7e800882e9e8c0cd49c86.jpg)

![34b6e4e33ed4eb4b9c25dce2ba1d404115acb912cdf6752bbbfb6e53d322a622.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/34b6e4e33ed4eb4b9c25dce2ba1d404115acb912cdf6752bbbfb6e53d322a622.jpg)

![3eabe5e13d90c337d9acb05cd7c0d478860c85c2fe2697f145a787e18e5b8096.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/3eabe5e13d90c337d9acb05cd7c0d478860c85c2fe2697f145a787e18e5b8096.jpg)

![422f0390e501c7bb0e10c97313bd981027b39535fd412698e9ef7049570978de.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/422f0390e501c7bb0e10c97313bd981027b39535fd412698e9ef7049570978de.jpg)

![83850dbbaf2c5d03b7cbdeda147c275c5b3dfdf6201222e890015347ed9918a5.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/83850dbbaf2c5d03b7cbdeda147c275c5b3dfdf6201222e890015347ed9918a5.jpg)

![927e2d685ada2d5848470e889a4d6e4208e6b44b9040c64966d1d82306543ddf.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/927e2d685ada2d5848470e889a4d6e4208e6b44b9040c64966d1d82306543ddf.jpg)

![ae9d085e88a787427dbb43c9956354653a875bc481db2f2e4cf3ca773462f966.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/ae9d085e88a787427dbb43c9956354653a875bc481db2f2e4cf3ca773462f966.jpg)

![cc6ea71c6f7d53e859e96302e5ff46f04d3bb634390d0ae781cdf391b2ffe3e9.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/cc6ea71c6f7d53e859e96302e5ff46f04d3bb634390d0ae781cdf391b2ffe3e9.jpg)

![cd4072434fe3b6d96724e0a23d383c76310ed4718856fa773df843fde074c435.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/cd4072434fe3b6d96724e0a23d383c76310ed4718856fa773df843fde074c435.jpg)

![d6e02e822be246ef532d7a92588b4f15245c46e14313a7251495dec151ee5025.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/d6e02e822be246ef532d7a92588b4f15245c46e14313a7251495dec151ee5025.jpg)

![e36f8ab58e30318cc607c5664b1f8c20e98a84bbbe9ec94711407144eb33f72a.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/e36f8ab58e30318cc607c5664b1f8c20e98a84bbbe9ec94711407144eb33f72a.jpg)

![ec2c9d02dbcc029fb295bb1bf57d4d8ffbd4209e413412e4f4549a16d58127cd.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/ec2c9d02dbcc029fb295bb1bf57d4d8ffbd4209e413412e4f4549a16d58127cd.jpg)

![f1a3670226f11a7107d4dacf9b54a9f358fd7ff70cfb24cddc3261b9bce48d0e.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/f1a3670226f11a7107d4dacf9b54a9f358fd7ff70cfb24cddc3261b9bce48d0e.jpg)

![f30147fabb443fcebb37f1c14d136eafbcf9723e874096b3375cfb89af361c4b.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/f30147fabb443fcebb37f1c14d136eafbcf9723e874096b3375cfb89af361c4b.jpg)

![f68e1056f051b834bbb8866860f8f4a9db3e8bd4c70ffb1159fe84c4b029277d.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/f68e1056f051b834bbb8866860f8f4a9db3e8bd4c70ffb1159fe84c4b029277d.jpg)

![fb56af3e8e5947d5960006a624a68883dccb5965eb8b5411ec146b374bd941aa.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/images/fb56af3e8e5947d5960006a624a68883dccb5965eb8b5411ec146b374bd941aa.jpg)

### Tables

![0b2f249dca41bac1eed4d7b11a0adcf729e7c5b291d3bdc473b4aa44dd7ca38e.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/0b2f249dca41bac1eed4d7b11a0adcf729e7c5b291d3bdc473b4aa44dd7ca38e.jpg)

![283e3df619f94964fed564009e6edaead9f08908c9298553e53837581f212663.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/283e3df619f94964fed564009e6edaead9f08908c9298553e53837581f212663.jpg)

![3a38c911c029047f0d4f3056bea0efe8013db8c80c69f76ccceaf548bebdbec9.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/3a38c911c029047f0d4f3056bea0efe8013db8c80c69f76ccceaf548bebdbec9.jpg)

![68ebdfae6b793239de58b1c798d8cb1cbc844af6b1cb23bc004658c7a5a75e76.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/68ebdfae6b793239de58b1c798d8cb1cbc844af6b1cb23bc004658c7a5a75e76.jpg)

![79e0a1ef6069c3e55a7e21d6b9cab655fff38a4e73c310d83a6bb316fc90a7d9.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/79e0a1ef6069c3e55a7e21d6b9cab655fff38a4e73c310d83a6bb316fc90a7d9.jpg)

![7fc80cd8f7010a4a24cd1913d07ce106f73114c8fbeabcc1b05f285a0727d5e1.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/7fc80cd8f7010a4a24cd1913d07ce106f73114c8fbeabcc1b05f285a0727d5e1.jpg)

![8d42e6684c3d35c88b12333987bd8c91880f4bc8fbe26c2e1eaa73b7299e86de.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/8d42e6684c3d35c88b12333987bd8c91880f4bc8fbe26c2e1eaa73b7299e86de.jpg)

![8e9e4edcc9413d44d58b3e136c5064ff29493727977da2c4eb2cd53828b39a76.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/8e9e4edcc9413d44d58b3e136c5064ff29493727977da2c4eb2cd53828b39a76.jpg)

![927eace2ce99507fd735c3c72b030766f77b483364f55bf6bd5f5658d22680d6.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/927eace2ce99507fd735c3c72b030766f77b483364f55bf6bd5f5658d22680d6.jpg)

![cda8f108646043bbadb601f6f24b01a634029ecba88631530bd29953bc256b37.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/cda8f108646043bbadb601f6f24b01a634029ecba88631530bd29953bc256b37.jpg)

![cdb641a15bc07104bfe9dad73728319c2704b1877e8962ef4a72a73fb71ec5aa.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/cdb641a15bc07104bfe9dad73728319c2704b1877e8962ef4a72a73fb71ec5aa.jpg)

![d6034aba39091502a75871e74ba411d17730ed1691939296af870d304f713b11.jpg](../icml_results/5190_EchoAttention_%20Exploiting%20Token-Pair%20Redundancy%20and%20Frame-Block%20Similarity%20for%20Efficient%20Video%20Gener/tables/d6034aba39091502a75871e74ba411d17730ed1691939296af870d304f713b11.jpg)

<a id="paper-5143"></a>

## BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction


### Images

![108602235116b9e0c52423e2e7f011fe64df6fecc44a56519f213ed203761725.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/108602235116b9e0c52423e2e7f011fe64df6fecc44a56519f213ed203761725.jpg)

![16055f93d13dd04ff191c74fa42ef28150734ca2b8462e8186eb365e21682031.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/16055f93d13dd04ff191c74fa42ef28150734ca2b8462e8186eb365e21682031.jpg)

![25f764d4e1fe36a954bce746e64d686c227bb29f625a362fdaa56cb10ac104d7.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/25f764d4e1fe36a954bce746e64d686c227bb29f625a362fdaa56cb10ac104d7.jpg)

![29a1b8e7e807b2a9e09b056fac8c3fb4b73fa5e530e533aa11adaed9299a1c6f.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/29a1b8e7e807b2a9e09b056fac8c3fb4b73fa5e530e533aa11adaed9299a1c6f.jpg)

![3b1d3832330297eb5c141f6cc5acfe1b7e5b86ea7eb7d4d5c57ccb39145397f3.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/3b1d3832330297eb5c141f6cc5acfe1b7e5b86ea7eb7d4d5c57ccb39145397f3.jpg)

![7eebd9059bbbd96469021aef000c42e7ea97910816451bad1e2e811c8ed69875.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/7eebd9059bbbd96469021aef000c42e7ea97910816451bad1e2e811c8ed69875.jpg)

![817f898e2f02f0415712dfb113c8c9ff6fce8d7fac9603236779de442eb75d8b.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/817f898e2f02f0415712dfb113c8c9ff6fce8d7fac9603236779de442eb75d8b.jpg)

![9840235bbfc9c99d5613ad58c5096f29ecf1a27666e8107fa75f1e86f254c1b5.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/9840235bbfc9c99d5613ad58c5096f29ecf1a27666e8107fa75f1e86f254c1b5.jpg)

![9f7da1bd970e66fd7f894588c20c4c818c1608b52a94a3f285e3eea583d647f2.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/9f7da1bd970e66fd7f894588c20c4c818c1608b52a94a3f285e3eea583d647f2.jpg)

![a4ab11c815f407e44ed4a1b03bb1fba3dd69cc4ebeb1d8f58934be8e9c7fd9cd.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/a4ab11c815f407e44ed4a1b03bb1fba3dd69cc4ebeb1d8f58934be8e9c7fd9cd.jpg)

![b6e843e4541d5e7ee1a04cd4224d24d30c809b71fa857afb175f96c116f294a5.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/b6e843e4541d5e7ee1a04cd4224d24d30c809b71fa857afb175f96c116f294a5.jpg)

![fcd74a64b8e406000273cf6a0b9786f78a700aa981cdbcae23153b06c0de781f.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/images/fcd74a64b8e406000273cf6a0b9786f78a700aa981cdbcae23153b06c0de781f.jpg)

### Tables

![1ccde721cbe7ff499fbc4797efe23e3ac5dae85994d7eec230f5665bb4b9d540.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/1ccde721cbe7ff499fbc4797efe23e3ac5dae85994d7eec230f5665bb4b9d540.jpg)

![3c5f3688d85d2568b5682d4293e048029e9842117cebd0a9d323a49eea75c674.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/3c5f3688d85d2568b5682d4293e048029e9842117cebd0a9d323a49eea75c674.jpg)

![4035bc5a7ee723066db8db32eb736a488bc5f70435345d9b5803a0a4f165913e.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/4035bc5a7ee723066db8db32eb736a488bc5f70435345d9b5803a0a4f165913e.jpg)

![700c1fb4105f32c650630d0c17f9a4f1993b57edf13a7e37f4e7a709ae8c034c.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/700c1fb4105f32c650630d0c17f9a4f1993b57edf13a7e37f4e7a709ae8c034c.jpg)

![9cd23302b41b8aa8a8667167ff264fdda13668c9f46f8e12ce145b4ff0d32e3d.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/9cd23302b41b8aa8a8667167ff264fdda13668c9f46f8e12ce145b4ff0d32e3d.jpg)

![c12c0397cc9225310af90ab25a20a78b25a35e331ef72010c046cdbaf139aee0.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/c12c0397cc9225310af90ab25a20a78b25a35e331ef72010c046cdbaf139aee0.jpg)

![d060fcf89afdd9d5f719e4c734ba4d5dfaab37ea879627df1980e7d22737b1dc.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/d060fcf89afdd9d5f719e4c734ba4d5dfaab37ea879627df1980e7d22737b1dc.jpg)

![e085c2200415da151e92a1a419c310de828e38c2ca25a153405e9652cbf03bb2.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/e085c2200415da151e92a1a419c310de828e38c2ca25a153405e9652cbf03bb2.jpg)

![e9ced78173ef8ab16e089cae333e32724fd58dbbfb6c5201871f2502cd49220c.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/e9ced78173ef8ab16e089cae333e32724fd58dbbfb6c5201871f2502cd49220c.jpg)

![eadcf0528e8ba5956e27554603b42f29b3aefe3f5e9987bc8cfe059df592ce4c.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/eadcf0528e8ba5956e27554603b42f29b3aefe3f5e9987bc8cfe059df592ce4c.jpg)

![edb66c179e30f3edfac996d1c0473555f8111fc71ff3204b72d7ae6acc205687.jpg](../icml_results/5191_BuildArena_%20A%20Physics-Aligned%20Interactive%20Benchmark%20of%20LLMs%20for%20Engineering%20Construction/tables/edb66c179e30f3edfac996d1c0473555f8111fc71ff3204b72d7ae6acc205687.jpg)

<a id="paper-5144"></a>

## MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts


### Images

![09d92b64ab159f18653752368e291e5f34f57e5360105a4ebdf723c936597378.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/09d92b64ab159f18653752368e291e5f34f57e5360105a4ebdf723c936597378.jpg)

![5224f97757ef96496556cbf9550670d220d68b9d8a4c9f847fa2511612220341.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/5224f97757ef96496556cbf9550670d220d68b9d8a4c9f847fa2511612220341.jpg)

![5f3440b6b0696f26bd57d1743dc00ad4cd17600fd37f3940d3b298f271c5951e.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/5f3440b6b0696f26bd57d1743dc00ad4cd17600fd37f3940d3b298f271c5951e.jpg)

![a826931abf5cb81f0a3b01e7638ff54f214fb9f8dda4a58d7ddf69978c971d72.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/a826931abf5cb81f0a3b01e7638ff54f214fb9f8dda4a58d7ddf69978c971d72.jpg)

![bd943620216232c49ce639762219bdab1fa5b52088c30569c35174f6261cc501.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/bd943620216232c49ce639762219bdab1fa5b52088c30569c35174f6261cc501.jpg)

![c3674e9477d8e5c221b35c3ba427438030737373bce319d9b9a711f69dce8b5a.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/images/c3674e9477d8e5c221b35c3ba427438030737373bce319d9b9a711f69dce8b5a.jpg)

### Tables

![3237824f74ed8e572730a509eca5a5407f39b34fda9f2b1ef6efcc32bb5380d5.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/3237824f74ed8e572730a509eca5a5407f39b34fda9f2b1ef6efcc32bb5380d5.jpg)

![4ca09ab83d7123fa7b5abd8d7c30ed7c3e027b9bece83490be43563fc1a35fc9.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/4ca09ab83d7123fa7b5abd8d7c30ed7c3e027b9bece83490be43563fc1a35fc9.jpg)

![5366873f70f9f2a8878e03698a26d42f3561abc8eb3c8d75a037f21f726afac4.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/5366873f70f9f2a8878e03698a26d42f3561abc8eb3c8d75a037f21f726afac4.jpg)

![58b4c576b8b8456263e003aaad54df2efc1b6e8bc8cffebe3e19ba517f64779c.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/58b4c576b8b8456263e003aaad54df2efc1b6e8bc8cffebe3e19ba517f64779c.jpg)

![884c621aad1616ab2baa650b4f86726bc6fa80eefb2e68ca63cf060682ffb77d.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/884c621aad1616ab2baa650b4f86726bc6fa80eefb2e68ca63cf060682ffb77d.jpg)

![8852193f3ab281542af2d61e8e1da98601f16efec2824407580a4fb0c0864395.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/8852193f3ab281542af2d61e8e1da98601f16efec2824407580a4fb0c0864395.jpg)

![9f7d08c5513830eb9e76fda2a53d3f04735394a854166dd82e931e7af3cfa210.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/9f7d08c5513830eb9e76fda2a53d3f04735394a854166dd82e931e7af3cfa210.jpg)

![a595d67df51ba691341a050593c9e0029040743f2d307a16ca30f6fbe044db0a.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/a595d67df51ba691341a050593c9e0029040743f2d307a16ca30f6fbe044db0a.jpg)

![b27896a7f0cc702c265c2cd8943e11441fede7061a6bac1d2406dd92864bb24b.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/b27896a7f0cc702c265c2cd8943e11441fede7061a6bac1d2406dd92864bb24b.jpg)

![b5512499a7bc0a28c3a43f991f14a146d34103e3f2027615102e562a1c9c3402.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/b5512499a7bc0a28c3a43f991f14a146d34103e3f2027615102e562a1c9c3402.jpg)

![df9f49d00d6dbb4b327ec10d6023d08ff25cd0823a7afe3ff72fe255c973f5f6.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/df9f49d00d6dbb4b327ec10d6023d08ff25cd0823a7afe3ff72fe255c973f5f6.jpg)

![e870e0c6b4b33852fe99c9e8c81f9913bb3c2e678e18b942b9259c535524e4a9.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/e870e0c6b4b33852fe99c9e8c81f9913bb3c2e678e18b942b9259c535524e4a9.jpg)

![eb3cb1186197817c757e90d4a4ae6993174c2361fd60ad72d2876ee2619552dd.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/eb3cb1186197817c757e90d4a4ae6993174c2361fd60ad72d2876ee2619552dd.jpg)

![f18f59834cec15a4155b6f078b4f08babed3222413bdcee5404290ae654acd68.jpg](../icml_results/5192_MedREK_%20Retrieval-Based%20Editing%20for%20Medical%20LLMs%20with%20Key-Aware%20Prompts/tables/f18f59834cec15a4155b6f078b4f08babed3222413bdcee5404290ae654acd68.jpg)

<a id="paper-5145"></a>

## LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding


### Images

![195e9ab70e0a4c6fab4debb7bdbe24f9bfc28603b51339774472cc6cc6f2fac8.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/195e9ab70e0a4c6fab4debb7bdbe24f9bfc28603b51339774472cc6cc6f2fac8.jpg)

![25b0047bd7fb2cbda6179de738037e9ae941e1f35375da8552f7f01cdfbccc5e.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/25b0047bd7fb2cbda6179de738037e9ae941e1f35375da8552f7f01cdfbccc5e.jpg)

![27a2366f8b24b793ea330dc01d7a45fb6a58cf70ccdbc1278e909f129aada116.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/27a2366f8b24b793ea330dc01d7a45fb6a58cf70ccdbc1278e909f129aada116.jpg)

![4b943b5f4eebfd8b6818d0d3174e171582229702f63e89fe7031fbc378866dfb.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/4b943b5f4eebfd8b6818d0d3174e171582229702f63e89fe7031fbc378866dfb.jpg)

![bb0b72a6e203362346978349e62813ad137da62bf39f2447c2bf151cb74ad7ad.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/bb0b72a6e203362346978349e62813ad137da62bf39f2447c2bf151cb74ad7ad.jpg)

![c60a3260995b12b722cb1c3815357dd2172b86136387135726d6d6e4becba46b.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/c60a3260995b12b722cb1c3815357dd2172b86136387135726d6d6e4becba46b.jpg)

![caa1c6eba250a22af4fc08a25f8a66d278677fdada21c21636a45c240b5e9437.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/caa1c6eba250a22af4fc08a25f8a66d278677fdada21c21636a45c240b5e9437.jpg)

![d7b63cb73ba1c3752d39f8f85f11b352f09b25930ed32ff1d2d888c75a604120.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/d7b63cb73ba1c3752d39f8f85f11b352f09b25930ed32ff1d2d888c75a604120.jpg)

![eb523c1a6cac026a6a2befe0703b8a3debe5db7d47b0c5fe80c2961e66306cfd.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/images/eb523c1a6cac026a6a2befe0703b8a3debe5db7d47b0c5fe80c2961e66306cfd.jpg)

### Tables

![21b1746571f064a289524f4eb1cda16a90489d0e1f519ae65bc3d957ba5d3873.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/21b1746571f064a289524f4eb1cda16a90489d0e1f519ae65bc3d957ba5d3873.jpg)

![328581d5eeef0479eec7c63250f0470adda0ab5c5885c40e2990434bc3bbcf0e.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/328581d5eeef0479eec7c63250f0470adda0ab5c5885c40e2990434bc3bbcf0e.jpg)

![34c04d2da9c1cfdd30380ad8959640da60353eaf9b104f1bdfe14ecfe171c6b9.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/34c04d2da9c1cfdd30380ad8959640da60353eaf9b104f1bdfe14ecfe171c6b9.jpg)

![3da57d88b64cd5c0223b47d0d1b84b18bda557c60f22ce5ce02c5348d6a3e99a.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/3da57d88b64cd5c0223b47d0d1b84b18bda557c60f22ce5ce02c5348d6a3e99a.jpg)

![4a47135102b5d74e1714ae6424ba386d478ce43abac3e71102e22c23ad6855d8.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/4a47135102b5d74e1714ae6424ba386d478ce43abac3e71102e22c23ad6855d8.jpg)

![57c2fe8ac1cd850eafb64d9e428b6afe8a7d475d041a7775e20b7a6b9b62a2e9.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/57c2fe8ac1cd850eafb64d9e428b6afe8a7d475d041a7775e20b7a6b9b62a2e9.jpg)

![5812993042b961c548a91fb884662c8748045bf634b61835d46e7dde542973f9.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/5812993042b961c548a91fb884662c8748045bf634b61835d46e7dde542973f9.jpg)

![8f714306c80e708cf6cb3e4c83bd6585d1b5acede77663c552f2ed39794fd39a.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/8f714306c80e708cf6cb3e4c83bd6585d1b5acede77663c552f2ed39794fd39a.jpg)

![a0ef6f2ffa4179cc0e8ad401e05ae16e02d5b9ab0041115051ceac5fce305769.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/a0ef6f2ffa4179cc0e8ad401e05ae16e02d5b9ab0041115051ceac5fce305769.jpg)

![eda9f6b2036e39df3f8d7b64b449cfef67f5b16cd92cfa10b071ac012475bf65.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/eda9f6b2036e39df3f8d7b64b449cfef67f5b16cd92cfa10b071ac012475bf65.jpg)

![fbbdc5bbf2fd1d46baec649cb9fe7ed8ef91f697c90cc89462f8202c619f2639.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/fbbdc5bbf2fd1d46baec649cb9fe7ed8ef91f697c90cc89462f8202c619f2639.jpg)

![fc732c0e48806120b787ebf05507d902fdc8678a40955d198df9eae72e570545.jpg](../icml_results/5193_LazyAttention_%20Efficient%20Retrieval-Augmented%20Generation%20with%20Deferred%20Positional%20Encoding/tables/fc732c0e48806120b787ebf05507d902fdc8678a40955d198df9eae72e570545.jpg)

<a id="paper-5146"></a>

## DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery


### Images

![0726122d3699dabd03bfe635ddd1bd0ba67be9e87c6eb98188bd5f056b45311e.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/0726122d3699dabd03bfe635ddd1bd0ba67be9e87c6eb98188bd5f056b45311e.jpg)

![0b3b496c9f623eba65687695a1a40ae940bf99af1dad61898b9ea5140503cc91.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/0b3b496c9f623eba65687695a1a40ae940bf99af1dad61898b9ea5140503cc91.jpg)

![0be35b2db885b66b4dfc6fd6e401d736476dfe26e06db358fe7f94c2be80dcb8.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/0be35b2db885b66b4dfc6fd6e401d736476dfe26e06db358fe7f94c2be80dcb8.jpg)

![48904c0bc422006e8402f25b420a29981ff577d8483f7595d5d0df6a13754035.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/48904c0bc422006e8402f25b420a29981ff577d8483f7595d5d0df6a13754035.jpg)

![5ec56e5f72ddcf991ed4f532694c71c881e306d129069698893cf4249d78ade3.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/5ec56e5f72ddcf991ed4f532694c71c881e306d129069698893cf4249d78ade3.jpg)

![6d1cb83ecc6d60f2ff15f3422dca58f38b0afcbc3a1a40836dfc34897e47cb1a.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/6d1cb83ecc6d60f2ff15f3422dca58f38b0afcbc3a1a40836dfc34897e47cb1a.jpg)

![773654c15b690bcfbf790287ec86e448cda021839f6da74c8853a7d826b0e820.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/773654c15b690bcfbf790287ec86e448cda021839f6da74c8853a7d826b0e820.jpg)

![a17535d7f0bc2e92f763aedbd354f9403a7afcc8eae8c308018236ffd4817534.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/a17535d7f0bc2e92f763aedbd354f9403a7afcc8eae8c308018236ffd4817534.jpg)

![af776ca3cee48aaf8f65690d49fd31faeaf09cb09abf467e006613f14c6ff9ef.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/af776ca3cee48aaf8f65690d49fd31faeaf09cb09abf467e006613f14c6ff9ef.jpg)

![bf81f080e231eeadbc69190d27ec7e5d92d934d69f61eb2ab69067bcc1d0b4ea.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/bf81f080e231eeadbc69190d27ec7e5d92d934d69f61eb2ab69067bcc1d0b4ea.jpg)

![c2183f8512cedbc7cf3af9fd91c93fccb88f29dfece60a9366e110a841e79f83.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/c2183f8512cedbc7cf3af9fd91c93fccb88f29dfece60a9366e110a841e79f83.jpg)

![c746ee758f9ae7dc466896ac1791095f7993bc8c3388d5356db643e9ed9c4be9.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/c746ee758f9ae7dc466896ac1791095f7993bc8c3388d5356db643e9ed9c4be9.jpg)

![c9ebc4ad8cf614e3124ff03d75babef06244f409748b9051456a46e61178f4da.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/c9ebc4ad8cf614e3124ff03d75babef06244f409748b9051456a46e61178f4da.jpg)

![fe5837715a7683d171150d6d73889b3e7f4cb7cb227b6e9be309593fe0e4a281.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/images/fe5837715a7683d171150d6d73889b3e7f4cb7cb227b6e9be309593fe0e4a281.jpg)

### Tables

![2342b43b7e3cbc0d0e21b9bdd135b1597fa5ab5ca8257403c42dc9bf210963af.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/2342b43b7e3cbc0d0e21b9bdd135b1597fa5ab5ca8257403c42dc9bf210963af.jpg)

![2ac0a632a7913d96aa6067921767b4a1044c6fa77a0482dce55308431cf43b6e.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/2ac0a632a7913d96aa6067921767b4a1044c6fa77a0482dce55308431cf43b6e.jpg)

![3df21463a7816df0dd907460aa2be8e50e98808b93c7549bf51b985202ec659a.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/3df21463a7816df0dd907460aa2be8e50e98808b93c7549bf51b985202ec659a.jpg)

![402921129a3aa37850615083f5ad6776b6629d69d6bca56b88bfb0d8dab18598.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/402921129a3aa37850615083f5ad6776b6629d69d6bca56b88bfb0d8dab18598.jpg)

![435a6e7633485625804c72ca02c4658ee70099e23955b6ed9e22443190f227a0.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/435a6e7633485625804c72ca02c4658ee70099e23955b6ed9e22443190f227a0.jpg)

![682500212e20317e21d8e1c4267a279a8780484740e4a3ef915db480fa36ae68.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/682500212e20317e21d8e1c4267a279a8780484740e4a3ef915db480fa36ae68.jpg)

![9790d1aa61b7bffe526d8e6fe3645f8ebef048fc478bb9fa559ddcd510b84862.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/9790d1aa61b7bffe526d8e6fe3645f8ebef048fc478bb9fa559ddcd510b84862.jpg)

![9803db085fc6d2c6b24c8b44cebaab3ae9b28cfae18c242ae894b002fd54d6dd.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/9803db085fc6d2c6b24c8b44cebaab3ae9b28cfae18c242ae894b002fd54d6dd.jpg)

![9ff8d1aaf5e69f20c2583e1964844e64bc02fa8bd5c13b5762c8d09ede20233c.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/9ff8d1aaf5e69f20c2583e1964844e64bc02fa8bd5c13b5762c8d09ede20233c.jpg)

![b0f11a97d6e955053bc9d96fcac8c4c2b216432530090db7db7be0c19bac7078.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/b0f11a97d6e955053bc9d96fcac8c4c2b216432530090db7db7be0c19bac7078.jpg)

![b365b0be2ff471393686742efa419dd7f0b104a8e92b70e4c04832da3a46b2e6.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/b365b0be2ff471393686742efa419dd7f0b104a8e92b70e4c04832da3a46b2e6.jpg)

![d8873953a434d1892252e6068513243359a6b12dd087f4862d3e65fd7fecbae3.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/d8873953a434d1892252e6068513243359a6b12dd087f4862d3e65fd7fecbae3.jpg)

![f47b125e29e9f174bd66a3a23fc1778722782e7e10ef340afb11a0e25afcd776.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/f47b125e29e9f174bd66a3a23fc1778722782e7e10ef340afb11a0e25afcd776.jpg)

![f61bd5dcf54508a4a31688ff4e1633d8cd98150fe335da9f0b102c6d90c6ad96.jpg](../icml_results/5194_DIVER_%20Diving%20Deeper%20into%20Distilled%20Data%20via%20Expressive%20Semantic%20Recovery/tables/f61bd5dcf54508a4a31688ff4e1633d8cd98150fe335da9f0b102c6d90c6ad96.jpg)

<a id="paper-5147"></a>

## Towards Scalable and Consistent 3D Editing


### Images

![03afdd64d1951ad48ef1a67edcc3ac5f499b2eb6089a1c609e2a9c24128e7aad.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/03afdd64d1951ad48ef1a67edcc3ac5f499b2eb6089a1c609e2a9c24128e7aad.jpg)

![1bc7ffa87a251bc0b0c57636ea9c0d16b831609eaf2669124b1f9b19972f133d.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/1bc7ffa87a251bc0b0c57636ea9c0d16b831609eaf2669124b1f9b19972f133d.jpg)

![2a44d697ec5cd4e7dc6023a8df7db5abba42f79b280625524498ad1ede39afb5.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/2a44d697ec5cd4e7dc6023a8df7db5abba42f79b280625524498ad1ede39afb5.jpg)

![64d518f369735bdb72ec66b245288c0764a16ed51a0dc59a9ec46a311cbfc863.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/64d518f369735bdb72ec66b245288c0764a16ed51a0dc59a9ec46a311cbfc863.jpg)

![6e4870034726b1b811a8ad78179b0c3689817dfb31d3932f362edc2f699fcaf9.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/6e4870034726b1b811a8ad78179b0c3689817dfb31d3932f362edc2f699fcaf9.jpg)

![727a5cc2899f7c2fa8542e7455300a87c71e3ff8239648ec2a2a0a43a62ca084.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/727a5cc2899f7c2fa8542e7455300a87c71e3ff8239648ec2a2a0a43a62ca084.jpg)

![86914bea5119a03b0c87573890668d7e4c7fad6fa4b851b45bb1474f9e2364b3.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/86914bea5119a03b0c87573890668d7e4c7fad6fa4b851b45bb1474f9e2364b3.jpg)

![870cd6f9b494f72362171f3853d16ab4819eb2d6559df5c974e6e7af8b1d64e2.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/870cd6f9b494f72362171f3853d16ab4819eb2d6559df5c974e6e7af8b1d64e2.jpg)

![8ed04b31de52dbb242122a535e20c9107562ad91a063b5700fdad074701657c5.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/8ed04b31de52dbb242122a535e20c9107562ad91a063b5700fdad074701657c5.jpg)

![918abd9746af4902350f2f7095eda8feb349006b249ada5f58bfb970128b429a.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/918abd9746af4902350f2f7095eda8feb349006b249ada5f58bfb970128b429a.jpg)

![b40173bc1d40d39044c8fcc27fcf4cb0aed23ba7e3d01555ccb61af7336467d3.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/b40173bc1d40d39044c8fcc27fcf4cb0aed23ba7e3d01555ccb61af7336467d3.jpg)

![eadda85ea8c11260039a072ae4a059420d5e1949ee0e5a003a91009ce9d55649.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/eadda85ea8c11260039a072ae4a059420d5e1949ee0e5a003a91009ce9d55649.jpg)

![eb28af7332b8a13da2eff2bab0bd7951e6de1a4675a80bdc9f059c255b4ff33d.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/images/eb28af7332b8a13da2eff2bab0bd7951e6de1a4675a80bdc9f059c255b4ff33d.jpg)

### Tables

![1dd1c8f6d3388f90750fbe15db35266a9574a022ad6d2545a16527d1fb7a6a95.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/tables/1dd1c8f6d3388f90750fbe15db35266a9574a022ad6d2545a16527d1fb7a6a95.jpg)

![43a87d3efe240c11fdd325c7ee23be62e86bd1948c66b1dc3729d272d9107268.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/tables/43a87d3efe240c11fdd325c7ee23be62e86bd1948c66b1dc3729d272d9107268.jpg)

![62af291577ca0088f62342cfcb7314affb1aa10ad7d5c0c13b12333643975b96.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/tables/62af291577ca0088f62342cfcb7314affb1aa10ad7d5c0c13b12333643975b96.jpg)

![7b2bee41bc37a4f29da257c82d830996af4e63105d2269a9f2b2e41985877fc4.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/tables/7b2bee41bc37a4f29da257c82d830996af4e63105d2269a9f2b2e41985877fc4.jpg)

![7c41c024f325c009fd360d25a3e97f0c11aa7f7e3fc933ee6f6e73ac9e77af2c.jpg](../icml_results/5195_Towards%20Scalable%20and%20Consistent%203D%20Editing/tables/7c41c024f325c009fd360d25a3e97f0c11aa7f7e3fc933ee6f6e73ac9e77af2c.jpg)

<a id="paper-5148"></a>

## Boost the Identity-Preserving Embedding for Consistent Visual Generation

### Images

![243d6d45d3c108aee8e00394341ec12b6841211706dd15229ef2f4b38d8766ed.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/243d6d45d3c108aee8e00394341ec12b6841211706dd15229ef2f4b38d8766ed.jpg)

![2c06691430daf669e53940f9a7a4ed6bd598b2826711160657bccdce04e9c34b.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/2c06691430daf669e53940f9a7a4ed6bd598b2826711160657bccdce04e9c34b.jpg)

![2e62629c96b655ac1b7eb6e4af3c918264b5e8c69ce0d2f22cbdf63da9e511fb.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/2e62629c96b655ac1b7eb6e4af3c918264b5e8c69ce0d2f22cbdf63da9e511fb.jpg)

![32234d1bed8ffb6b00e1a7a5ae360b654d00488f578fae710a526c2158ba6e23.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/32234d1bed8ffb6b00e1a7a5ae360b654d00488f578fae710a526c2158ba6e23.jpg)

![40f8d52d9fd321283ca8ff7f10546a8e7a4d86d19869d0dff59eb5f3121e6f56.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/40f8d52d9fd321283ca8ff7f10546a8e7a4d86d19869d0dff59eb5f3121e6f56.jpg)

![5312df83daf9854f2d0d3335e70e7f168e66199bdd952ceac9290fb5cd5c1cd8.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/5312df83daf9854f2d0d3335e70e7f168e66199bdd952ceac9290fb5cd5c1cd8.jpg)

![573c49196dee7e4aa524a2695549dde63ea4b559c406a64d24d009939455b0ac.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/573c49196dee7e4aa524a2695549dde63ea4b559c406a64d24d009939455b0ac.jpg)

![5a7650970287dcf22441afdab451eb89422bf531221834209b3dd0ce436aeb0a.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/5a7650970287dcf22441afdab451eb89422bf531221834209b3dd0ce436aeb0a.jpg)

![699c27705821f8082bf99a190c8b5cf3527992f57ceca9218202a8cca90ec570.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/699c27705821f8082bf99a190c8b5cf3527992f57ceca9218202a8cca90ec570.jpg)

![6d6efefd99818e0ecbd16603498fafb917cce3631839cd90f8f82deb36897f62.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/6d6efefd99818e0ecbd16603498fafb917cce3631839cd90f8f82deb36897f62.jpg)

![79daf903415a32b2c6aea4e3a662c364e98d074fb51f33786f033881be78c51c.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/79daf903415a32b2c6aea4e3a662c364e98d074fb51f33786f033881be78c51c.jpg)

![8d1814d4ba203673a054443a55175a5897d5916bc34bc3f1dcf185ab62f0094c.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/8d1814d4ba203673a054443a55175a5897d5916bc34bc3f1dcf185ab62f0094c.jpg)

![a148c87c932071eda5861a5fd0bca297eea7d6421efe778eec9c8e1706a75f3f.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a148c87c932071eda5861a5fd0bca297eea7d6421efe778eec9c8e1706a75f3f.jpg)

![a37f9d354d4038c106b99fbe8af59a334b3282cb6681ad9ca20a18d763a533bf.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a37f9d354d4038c106b99fbe8af59a334b3282cb6681ad9ca20a18d763a533bf.jpg)

![a5dff9e81921f5de4c535a314b4c7689ab67736e6042fba5c7a40f8ed58608c7.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a5dff9e81921f5de4c535a314b4c7689ab67736e6042fba5c7a40f8ed58608c7.jpg)

![a877ad4c69a1059581a30b50fe454e5a071018267493e3a1201d011363bd01a1.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a877ad4c69a1059581a30b50fe454e5a071018267493e3a1201d011363bd01a1.jpg)

![e981b953afe7ed27dd3cc5aa3825324a37b4154337c4bda0131a5c3be6e70173.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/e981b953afe7ed27dd3cc5aa3825324a37b4154337c4bda0131a5c3be6e70173.jpg)

![eed30645f75afb516a390d78ac90b4d70fac184744576a33f81a2aabd67c8491.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/eed30645f75afb516a390d78ac90b4d70fac184744576a33f81a2aabd67c8491.jpg)

![f9ce066979f0efcf862f3f65a48c4c482e3e56ebb0619f8c46250d07a82fe394.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/f9ce066979f0efcf862f3f65a48c4c482e3e56ebb0619f8c46250d07a82fe394.jpg)

### Tables

![1c181fa300dbc0c58437ce3a1a2aefb07deab1b0beee3a658c43ea818fb75927.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/1c181fa300dbc0c58437ce3a1a2aefb07deab1b0beee3a658c43ea818fb75927.jpg)

![25117ed6050a7a2fd439df110c2eef7a985fb9131dda7cc8a06fe59afbef1674.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/25117ed6050a7a2fd439df110c2eef7a985fb9131dda7cc8a06fe59afbef1674.jpg)

![28e141bb53e63285b283533e1f10f010e3a53006750126caff71fac8f53cb9d6.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/28e141bb53e63285b283533e1f10f010e3a53006750126caff71fac8f53cb9d6.jpg)

![2aed140ce7ebd2ee0b879350b0ff8ca8269b6be4bae05d27c7daace36ff937c7.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/2aed140ce7ebd2ee0b879350b0ff8ca8269b6be4bae05d27c7daace36ff937c7.jpg)

![3cf7d9df052e957ac09c124906fa2edc310d37ea296f55fc64990a272bac1b00.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/3cf7d9df052e957ac09c124906fa2edc310d37ea296f55fc64990a272bac1b00.jpg)

![5672012a8319f1cd5293bf1e21c4daf5ec7629564d5f22c467e53ac04d339b17.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/5672012a8319f1cd5293bf1e21c4daf5ec7629564d5f22c467e53ac04d339b17.jpg)

![60a21388704589f7e5ac90ad4c49aeab424fc6ad290cc94819eaa24aa936eaf5.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/60a21388704589f7e5ac90ad4c49aeab424fc6ad290cc94819eaa24aa936eaf5.jpg)

![749c084fa8be72a69175a64ae50e8c66e0e3a1e35d55686426c3ce4d73a59eb5.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/749c084fa8be72a69175a64ae50e8c66e0e3a1e35d55686426c3ce4d73a59eb5.jpg)

![bc0e4aca066c664c92e3b78632842c195841e419a3500c6e1c2a8d3326fa44e0.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/bc0e4aca066c664c92e3b78632842c195841e419a3500c6e1c2a8d3326fa44e0.jpg)
