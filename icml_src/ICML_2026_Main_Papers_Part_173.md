# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 173 of 197

## 目录 (Table of Contents)

1. [The Safety-Aware Denoiser for Text Diffusion Models](#paper-5677)
2. [Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](#paper-5678)
3. [Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](#paper-5679)
4. [Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](#paper-5680)
5. [Weasel: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection](#paper-5681)
6. [Differentially Private Submodular Maximization with a Knapsack Constraint](#paper-5682)
7. [Geometry-Aware Tabular Diffusion](#paper-5683)
8. [When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs](#paper-5684)
9. [REViT: Roto-reflection Equivariant Convolutional Vision Transformer](#paper-5685)
10. [Self-Supervised Learning as Discrete Communication](#paper-5686)
11. [Federated Learning with Unlabeled Clients: Personalization Can Happen in Low Dimensions](#paper-5687)
12. [The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning](#paper-5688)
13. [HDTree: Generative Modeling of Cellular Hierarchies for Robust Lineage Inference](#paper-5689)
14. [Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios](#paper-5690)
15. [HIVE-3D: Hierarchical Voxel Enhancement for High-Quality 3D Scene Generation](#paper-5691)
16. [CINOC: Cardinality-Invariant Neural Operator Policies for Scalable PDE Control](#paper-5692)
17. [Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](#paper-5693)
18. [SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](#paper-5694)
19. [MentisOculi: Revealing the Limits of Reasoning with Mental Imagery](#paper-5695)
20. [Rethinking Neural Network Learning Rates: A Stackelberg Perspective](#paper-5696)
21. [POLIA: Policy Optimization with Visual-Object-Level Intrinsic Advantage for Multimodal Reasoning](#paper-5697)
22. [Symbiosis-Inspired Knowledge Distillation for Incremental Object Detection](#paper-5698)
23. [Bootstrapped Exploration with Causal Reasoning: A Training Paradigm for Adaptive Forecasting Agent](#paper-5699)
24. [Beyond Euclidean Summaries: Online Change Point Detection for Distribution-Valued Data](#paper-5700)
25. [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](#paper-5701)
26. [Attention with Routed-Memory for Learnable Sparse Control](#paper-5702)
27. [UCPO: Uncertainty-Aware Policy Optimization](#paper-5703)
28. [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](#paper-5704)
29. [Convex Dataset Valuation for Post-Training](#paper-5705)
30. [Large Language Models Explore by Latent Distilling](#paper-5706)
31. [TaskLoom: Weaving Knowledge Across Tasks in World Models](#paper-5707)
32. [Normality Calibration in Semi-supervised Graph Anomaly Detection](#paper-5708)
33. [MTNL: A Unified Modeling Perspective for Enhancing Tensor Network Learning](#paper-5709)

---


<a id="paper-5677"></a>

## The Safety-Aware Denoiser for Text Diffusion Models

### Images

![11dd2a99edf098c9ba5d7d801919991920af99855e65ff7141531a94cfaf3fdb.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/11dd2a99edf098c9ba5d7d801919991920af99855e65ff7141531a94cfaf3fdb.jpg)

![26ba965e946a991103184057d45a04c9b6d41a37c0f68f9fde63ad7dbdb29974.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/26ba965e946a991103184057d45a04c9b6d41a37c0f68f9fde63ad7dbdb29974.jpg)

![2f7b289a6a69418a9ba052f5f924b95dae12477971a7db5aaa7476b3c10d3f8f.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/2f7b289a6a69418a9ba052f5f924b95dae12477971a7db5aaa7476b3c10d3f8f.jpg)

![3561d2a8a7f1ff9c3bd5e35574cea5628e2a11a47f5bd8d6b775add01a6b18dd.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/3561d2a8a7f1ff9c3bd5e35574cea5628e2a11a47f5bd8d6b775add01a6b18dd.jpg)

![363dab3b997b14a4ac9c17353f327cc454014fcf813fe7d79b7cacc15842ebd2.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/363dab3b997b14a4ac9c17353f327cc454014fcf813fe7d79b7cacc15842ebd2.jpg)

![387cdd9b65337059a763941749c35df347ef19273696d90ed318fed4d5dced6c.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/387cdd9b65337059a763941749c35df347ef19273696d90ed318fed4d5dced6c.jpg)

![3c6d2ee6d529c9b35e6763764793e748bea80880839451c21160a31aa484baf1.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/3c6d2ee6d529c9b35e6763764793e748bea80880839451c21160a31aa484baf1.jpg)

![43ecd2edc0851abaa7b4caf9da897438793c3f3970d648494aa0a76cef4b689d.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/43ecd2edc0851abaa7b4caf9da897438793c3f3970d648494aa0a76cef4b689d.jpg)

![4c75a7c782e0f6abf4655f0b82533d8fe79c53aabb9cd7118c5f0d0a2f8ccb7d.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/4c75a7c782e0f6abf4655f0b82533d8fe79c53aabb9cd7118c5f0d0a2f8ccb7d.jpg)

![534fe68739aa621b091e23cac09e28d0481e9b304c42f9ae7577f897691d8526.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/534fe68739aa621b091e23cac09e28d0481e9b304c42f9ae7577f897691d8526.jpg)

![5c6e03c7a507da1a67c0b088ffc443b6a2d0e2942f1541de65b6abbe73c0b584.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/5c6e03c7a507da1a67c0b088ffc443b6a2d0e2942f1541de65b6abbe73c0b584.jpg)

![76823bed6be450d7bac2ff71a3d46cf6f408755bb7669af735abef1a07c1364b.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/76823bed6be450d7bac2ff71a3d46cf6f408755bb7669af735abef1a07c1364b.jpg)

![80fd98b8264f5a27e07e07c213c319a305b2ae2174059adebad5de3e39bd312e.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/80fd98b8264f5a27e07e07c213c319a305b2ae2174059adebad5de3e39bd312e.jpg)

![868e3a7723485f3546c5cef171d801cb3aeaac268e74e04951d0e5f2ce1a5236.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/868e3a7723485f3546c5cef171d801cb3aeaac268e74e04951d0e5f2ce1a5236.jpg)

![8d3ab7f5af7914e52ea43842cba20f5fed19acb5b258272738b91db3459d6896.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/8d3ab7f5af7914e52ea43842cba20f5fed19acb5b258272738b91db3459d6896.jpg)

![9bb020e4cdb6d348133b0d437d83ead4c72112b4336b26a1dd95b3c93ff36dd6.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/9bb020e4cdb6d348133b0d437d83ead4c72112b4336b26a1dd95b3c93ff36dd6.jpg)

![a787bf82ca6acb0096b698212165f929f98a322d0f9f2f4e66cb025afeb4df42.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/a787bf82ca6acb0096b698212165f929f98a322d0f9f2f4e66cb025afeb4df42.jpg)

![aad347351368ca6a91dad2174e102694c99865d7afb488f6403931856666508a.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/aad347351368ca6a91dad2174e102694c99865d7afb488f6403931856666508a.jpg)

![b52ce4c108c9813deb92e1fa0b6659af8728f739308d2e50dd50f9ad63d5a03a.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/b52ce4c108c9813deb92e1fa0b6659af8728f739308d2e50dd50f9ad63d5a03a.jpg)

![c143220765940e9243cb0e318029519167e6070ec501847148fa1be8ef42cfad.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/c143220765940e9243cb0e318029519167e6070ec501847148fa1be8ef42cfad.jpg)

![cba8653c5a366813933b5a8f9545f2cf84577a60feb6046aa4df292c852c1180.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/cba8653c5a366813933b5a8f9545f2cf84577a60feb6046aa4df292c852c1180.jpg)

![ce183de70bd680b5712b77254410be835e8fcb49f044b4053ffa09777a6a4d6e.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/ce183de70bd680b5712b77254410be835e8fcb49f044b4053ffa09777a6a4d6e.jpg)

![cee6e467bb4d057a86a13be0852d6c33d49f4f1822e2db5758a3ef072df96674.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/cee6e467bb4d057a86a13be0852d6c33d49f4f1822e2db5758a3ef072df96674.jpg)

![cf10fa486ec37ef8a96162ed936c5c9495745b393eda7487dbdc8dda9ccf51ab.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/cf10fa486ec37ef8a96162ed936c5c9495745b393eda7487dbdc8dda9ccf51ab.jpg)

![d50e1cd9613fd8f020f9943b9cce3728a64f8db8323a9f3de74f37f2158cc57d.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/d50e1cd9613fd8f020f9943b9cce3728a64f8db8323a9f3de74f37f2158cc57d.jpg)

![edc2690009634742ac2b3c511c9647186314d3aaeea9f5e067fe4a4c65060509.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/images/edc2690009634742ac2b3c511c9647186314d3aaeea9f5e067fe4a4c65060509.jpg)

### Tables

![097bae49de8f370dd64bbd3f39e5051f6d555c8eedf38076cc0cfc3a3d613874.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/097bae49de8f370dd64bbd3f39e5051f6d555c8eedf38076cc0cfc3a3d613874.jpg)

![14ea802f96145a184d77d7db967f3f708689c272d2c369673bb0f58d6841a00b.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/14ea802f96145a184d77d7db967f3f708689c272d2c369673bb0f58d6841a00b.jpg)

![164dc4f3ecafed9b4c4f02b7d36e79d1783e5711e4680df47df7146327cccf3a.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/164dc4f3ecafed9b4c4f02b7d36e79d1783e5711e4680df47df7146327cccf3a.jpg)

![21568543d0b576e8ad8527d694e79dcd821d7b6ef581036f4481eaa36edc0f00.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/21568543d0b576e8ad8527d694e79dcd821d7b6ef581036f4481eaa36edc0f00.jpg)

![3df97b85a6d76e6f16c504b6d34c283a14433fb888b9542e539fc63fbb71e7d4.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/3df97b85a6d76e6f16c504b6d34c283a14433fb888b9542e539fc63fbb71e7d4.jpg)

![59f2ccb2051028c38ea80551e25c7cf1910904be9a84e93168f1d3b69ff28d7d.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/59f2ccb2051028c38ea80551e25c7cf1910904be9a84e93168f1d3b69ff28d7d.jpg)

![6a80f965c9242e4a8031f1f49c618d9d0339d09ce7384d2a93fd720bba73a966.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/6a80f965c9242e4a8031f1f49c618d9d0339d09ce7384d2a93fd720bba73a966.jpg)

![6c93cdb5bda613c408fb87326f238831caae6bd9f8323dac63d56954b481fa93.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/6c93cdb5bda613c408fb87326f238831caae6bd9f8323dac63d56954b481fa93.jpg)

![73513c1eb2e62794f569c12b0a1b71cf1cbbaa453b5f1c23f1286da41c539137.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/73513c1eb2e62794f569c12b0a1b71cf1cbbaa453b5f1c23f1286da41c539137.jpg)

![76ae1998db3583d65e5c9ea49ee70b56320fbcfd336986b0b2156ac9bea5a62d.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/76ae1998db3583d65e5c9ea49ee70b56320fbcfd336986b0b2156ac9bea5a62d.jpg)

![82b4039a9507792788523e5d3c3d8ccd1899dfd99aba5265292fa1cf17908710.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/82b4039a9507792788523e5d3c3d8ccd1899dfd99aba5265292fa1cf17908710.jpg)

![9783e6c2c66329cf4edc64f1f4a2f0225284e2eea44649470bc340086567f8ac.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/9783e6c2c66329cf4edc64f1f4a2f0225284e2eea44649470bc340086567f8ac.jpg)

![c7d38baaa4723f5beb87f8c0f3b4b4c409b67d5c7e2976a3e08b6924b37637b0.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/c7d38baaa4723f5beb87f8c0f3b4b4c409b67d5c7e2976a3e08b6924b37637b0.jpg)

![c8e0b803a54abf0b82b0da91a266b0bc12f1f55eeb596a2139302af08f1d69ed.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/c8e0b803a54abf0b82b0da91a266b0bc12f1f55eeb596a2139302af08f1d69ed.jpg)

![ceb2b26e3187118d2afeb6ec7b6f7055f951b67278e504d040928986909ba816.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/ceb2b26e3187118d2afeb6ec7b6f7055f951b67278e504d040928986909ba816.jpg)

![d24d2a649e9e9558950bb8779fc5c677ba0599d2c752b5c237daa040df4319ef.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/d24d2a649e9e9558950bb8779fc5c677ba0599d2c752b5c237daa040df4319ef.jpg)

![d4d6a800e1f22cf9c75227c39412989b5b43364e1262a5aa4da92253572def39.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/d4d6a800e1f22cf9c75227c39412989b5b43364e1262a5aa4da92253572def39.jpg)

![ddac809d064c5ba9c01916f9a809b7cbe4d3d0bf388d8ba7f29d6ab92009fa76.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/ddac809d064c5ba9c01916f9a809b7cbe4d3d0bf388d8ba7f29d6ab92009fa76.jpg)

![e0ae70f2622b5e17be7fd6e986aea6f1b57902b818b2ec70f8ac9a236328baa0.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/e0ae70f2622b5e17be7fd6e986aea6f1b57902b818b2ec70f8ac9a236328baa0.jpg)

![e23d3923ab8750592ff504a5797ab19407341a66dcabc4d1b0187eb896640a17.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/e23d3923ab8750592ff504a5797ab19407341a66dcabc4d1b0187eb896640a17.jpg)

![e72c3de7e35b6ee550a8c9ec652514fc644f4c51d1e78992cf6046b455df7167.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/e72c3de7e35b6ee550a8c9ec652514fc644f4c51d1e78992cf6046b455df7167.jpg)

![f2ff2c7f630bd9405c8c8a12bda544c536af89e60fd358cf50eab54d7b3dc61c.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/f2ff2c7f630bd9405c8c8a12bda544c536af89e60fd358cf50eab54d7b3dc61c.jpg)

![f35cf83b6c75c0386fbea9fcbecc92c811b1fcc95bf20961a6b58b443b5ccf7c.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/f35cf83b6c75c0386fbea9fcbecc92c811b1fcc95bf20961a6b58b443b5ccf7c.jpg)

![fab1e4c88fd99f51dd929a2dd0be47baf37a34a3cf3e22d4276ed299e349c6d9.jpg](../icml_results/5725_Active%20Attacks_%20Red-teaming%20LLMs%20via%20Adaptive%20Environments/tables/fab1e4c88fd99f51dd929a2dd0be47baf37a34a3cf3e22d4276ed299e349c6d9.jpg)

## The Safety-Aware Denoiser for Text Diffusion Models


### Images

![0546ff948afed8fdb97acaf98ea8e72e111b6e98eecbc86fef0aacc3ba62ffe9.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/images/0546ff948afed8fdb97acaf98ea8e72e111b6e98eecbc86fef0aacc3ba62ffe9.jpg)

![4e6ad1dfa82bf1e43554043ec7526b53cc0d783f7ce14135b8ac716fbddc29f4.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/images/4e6ad1dfa82bf1e43554043ec7526b53cc0d783f7ce14135b8ac716fbddc29f4.jpg)

![651a6d0c64cdecc978b9e392062349885fedf809749ce391726fe17e7a1fa7ac.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/images/651a6d0c64cdecc978b9e392062349885fedf809749ce391726fe17e7a1fa7ac.jpg)

![b2b677ee971bfdc401caa21d80e929d40fe136e5718f03e5fb022d77a9586109.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/images/b2b677ee971bfdc401caa21d80e929d40fe136e5718f03e5fb022d77a9586109.jpg)

![f0fb1467e57242299957419df29abd701aae7b323c140e3beaade25c2d1a3dae.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/images/f0fb1467e57242299957419df29abd701aae7b323c140e3beaade25c2d1a3dae.jpg)

### Tables

![003e9fb6bfddf08030486f6ff4fcb36784a44dbd821f9628b0da68543f879545.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/003e9fb6bfddf08030486f6ff4fcb36784a44dbd821f9628b0da68543f879545.jpg)

![17f4d332acf75abecf41ba606af81e0a91a1e3d66c49aa3fb8e367236fafa1bc.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/17f4d332acf75abecf41ba606af81e0a91a1e3d66c49aa3fb8e367236fafa1bc.jpg)

![1a714f59d15fae8ac961bd38737fe38c06db089e2dd37c1af94be6b696a7d501.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/1a714f59d15fae8ac961bd38737fe38c06db089e2dd37c1af94be6b696a7d501.jpg)

![231dd61c6737109c6b2b0f2bfac214deebd4e60f046edbf0012c2efd831b418e.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/231dd61c6737109c6b2b0f2bfac214deebd4e60f046edbf0012c2efd831b418e.jpg)

![3935b031b229a7b1d36eb5e270112fa2c1361adc4f94d5f268f824fa1cd8a6fd.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/3935b031b229a7b1d36eb5e270112fa2c1361adc4f94d5f268f824fa1cd8a6fd.jpg)

![4954b7429269e184d4dbd6c9127671b76ce2f4a881835a3f5a7a7c16f4043f29.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/4954b7429269e184d4dbd6c9127671b76ce2f4a881835a3f5a7a7c16f4043f29.jpg)

![4c39fbd09c2cfb0639fec53044a64e3dfe8c925facd21085c3ac009cad85616f.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/4c39fbd09c2cfb0639fec53044a64e3dfe8c925facd21085c3ac009cad85616f.jpg)

![4c74f78aaa2f50880f3561cd1475fe35a18b3132b1efd80f43bfc448e428217f.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/4c74f78aaa2f50880f3561cd1475fe35a18b3132b1efd80f43bfc448e428217f.jpg)

![5631e32a87b5056682b41c900540f68f8b915195aa36728d653c9fb3013520b2.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/5631e32a87b5056682b41c900540f68f8b915195aa36728d653c9fb3013520b2.jpg)

![572881a0ff0dcba0d7e91cb59897c121e78a165197e62062c085a830c9f7a8fc.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/572881a0ff0dcba0d7e91cb59897c121e78a165197e62062c085a830c9f7a8fc.jpg)

![6a4df34a815573d50c808e7d89f15384bc8494900f42f63690bbb06b1531b02e.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/6a4df34a815573d50c808e7d89f15384bc8494900f42f63690bbb06b1531b02e.jpg)

![7cd58ef4f2e8ba9e42246e482f0d36a17f1bf09ca3fe119194a1539335177098.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/7cd58ef4f2e8ba9e42246e482f0d36a17f1bf09ca3fe119194a1539335177098.jpg)

![823ebba3c6a46d5c7a8f471814b4f48e2c84c5cfe5e3a82534f39fe8fb601bf3.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/823ebba3c6a46d5c7a8f471814b4f48e2c84c5cfe5e3a82534f39fe8fb601bf3.jpg)

![97ddd6c4715d857ae7fb6cb8b1dbcc9dcb8f785d62ac28bd6c66cc5130667a78.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/97ddd6c4715d857ae7fb6cb8b1dbcc9dcb8f785d62ac28bd6c66cc5130667a78.jpg)

![9cc2b7f70da825043e30b9b70b06c1fae63822de0f8cda8204182ff5e21968b4.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/9cc2b7f70da825043e30b9b70b06c1fae63822de0f8cda8204182ff5e21968b4.jpg)

![a49910f617df13c6fcfc1553548a278fac0095c9cbd169c6798682c735991c26.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/a49910f617df13c6fcfc1553548a278fac0095c9cbd169c6798682c735991c26.jpg)

![a6552a94c58feead31c56c20e3fbfc82b63f6a87026b735dbe508c1903019916.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/a6552a94c58feead31c56c20e3fbfc82b63f6a87026b735dbe508c1903019916.jpg)

![b3c1706dd55f99d61635435d3d93a93afc285ee5129e7ff353163002ac4a9ab6.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/b3c1706dd55f99d61635435d3d93a93afc285ee5129e7ff353163002ac4a9ab6.jpg)

![c8ddf4436dac3c01e1ed60fc478c9c50f8f9b9466382ad897de4b184b2fee8cd.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/c8ddf4436dac3c01e1ed60fc478c9c50f8f9b9466382ad897de4b184b2fee8cd.jpg)

![dc1677a4b5a13ea7f7ea27e51ba192ef28eacb44e57f8cfd3f8667fa7d86252b.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/dc1677a4b5a13ea7f7ea27e51ba192ef28eacb44e57f8cfd3f8667fa7d86252b.jpg)

![e1c7cc91ca550c7eeacb639c405499629e68270be0e325dced49cbe8fb058b28.jpg](../icml_results/5726_The%20Safety-Aware%20Denoiser%20for%20Text%20Diffusion%20Models/tables/e1c7cc91ca550c7eeacb639c405499629e68270be0e325dced49cbe8fb058b28.jpg)

<a id="paper-5678"></a>

## Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs


### Images

![038f08b1a28f30070447f9694c3e668ee95203c76c68c525a019db935d8f763d.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/038f08b1a28f30070447f9694c3e668ee95203c76c68c525a019db935d8f763d.jpg)

![07d154ba43e5fb55a80439cc1ceec13bf0cc91614ca646e14f0571ec9c0f5baf.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/07d154ba43e5fb55a80439cc1ceec13bf0cc91614ca646e14f0571ec9c0f5baf.jpg)

![1cc4af18992634eee3c6f3a284825e79340dcbec83261821b73cb527f21c49b4.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/1cc4af18992634eee3c6f3a284825e79340dcbec83261821b73cb527f21c49b4.jpg)

![249016419ceb07d304b0da0380c113643b67cd4672cc51dda36e22c576dce90f.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/249016419ceb07d304b0da0380c113643b67cd4672cc51dda36e22c576dce90f.jpg)

![318e960c66a3ea39352dba5a8b75ba64cf1a1bd00b8d194d61e72acfb9d1f909.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/318e960c66a3ea39352dba5a8b75ba64cf1a1bd00b8d194d61e72acfb9d1f909.jpg)

![3fa4cf58179c77f4324f0fe011ffe5b082e7739c7aaf300155ac8be818b0f6f0.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/3fa4cf58179c77f4324f0fe011ffe5b082e7739c7aaf300155ac8be818b0f6f0.jpg)

![4dc44d9b9e89146d7787d6a537b48e9986a5425bfef0d1f6c205caf806169525.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/4dc44d9b9e89146d7787d6a537b48e9986a5425bfef0d1f6c205caf806169525.jpg)

![4dfab465be3e2cd5c9c2966b17f8c65c23b9087a64aaccc14b84647f4ff72781.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/4dfab465be3e2cd5c9c2966b17f8c65c23b9087a64aaccc14b84647f4ff72781.jpg)

![6d4e2aed133961bf62ad3019ec975f6521be279cdd75a7d0b93a8fd6d406d769.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/6d4e2aed133961bf62ad3019ec975f6521be279cdd75a7d0b93a8fd6d406d769.jpg)

![83a58bb0e4c8b0fba9f23244cb5decd04041c13e8a6b3c9555449ee0b041fa18.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/83a58bb0e4c8b0fba9f23244cb5decd04041c13e8a6b3c9555449ee0b041fa18.jpg)

![a562342e01ccf3184ccdcaad948bdc60ffcd3fd2d37ba34721fdcf4c807c43c2.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/a562342e01ccf3184ccdcaad948bdc60ffcd3fd2d37ba34721fdcf4c807c43c2.jpg)

![cb36e5964a83874b6300d61a195cba0bccc6063b5746e7b82d29eee25912d25e.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/cb36e5964a83874b6300d61a195cba0bccc6063b5746e7b82d29eee25912d25e.jpg)

![cca14a2d787ecd657026c36d3a43aaec850ff628cacc1716a1cc10c0b8f46a65.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/cca14a2d787ecd657026c36d3a43aaec850ff628cacc1716a1cc10c0b8f46a65.jpg)

![cf26af120bb51e55e182e1813918fc5ccf0ab189cf8f6ca6eb26aa80e3a95546.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/cf26af120bb51e55e182e1813918fc5ccf0ab189cf8f6ca6eb26aa80e3a95546.jpg)

![ef26da0c598a5e8b1cc6cbb37407b2c164b6016507bfccc7e7b3517d1d901f20.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/ef26da0c598a5e8b1cc6cbb37407b2c164b6016507bfccc7e7b3517d1d901f20.jpg)

![f2654d49c3b67e064981744ef9131c413895f0ea123cf6a27b4e13dac2856a37.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/f2654d49c3b67e064981744ef9131c413895f0ea123cf6a27b4e13dac2856a37.jpg)

![fede512d6b70a63d2655f1e46ca4b7d0377fbe66932cee714ed829a14a4e1200.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/images/fede512d6b70a63d2655f1e46ca4b7d0377fbe66932cee714ed829a14a4e1200.jpg)

### Tables

![64495343610d4b5aeb66f5367f78a3225c9649f08ca5fdddda2b8638ef1fde0f.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/tables/64495343610d4b5aeb66f5367f78a3225c9649f08ca5fdddda2b8638ef1fde0f.jpg)

![6e8f8634000a9eee77c29cf02c27b94ca203d4704a9042933175abd5bb6e497a.jpg](../icml_results/5727_Unveiling%20the%20Structure%20of%20Do-Calculus%20Reasoning%20via%20Derivation%20Graphs/tables/6e8f8634000a9eee77c29cf02c27b94ca203d4704a9042933175abd5bb6e497a.jpg)

<a id="paper-5679"></a>

## Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning


### Images

![133c3cc4c04c455a1107d61126f0c3469fabdd06b29de6e4e4c84be60534aaae.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/133c3cc4c04c455a1107d61126f0c3469fabdd06b29de6e4e4c84be60534aaae.jpg)

![21d3dbc3107a0ff8b3b3980926d36d211aeadd9e1e2802e754139184c960ecfa.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/21d3dbc3107a0ff8b3b3980926d36d211aeadd9e1e2802e754139184c960ecfa.jpg)

![2c6b4f9a0620a11579acf1ea4017137fed4bead0ac9e6010df265bb6f99ad5f5.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/2c6b4f9a0620a11579acf1ea4017137fed4bead0ac9e6010df265bb6f99ad5f5.jpg)

![3c475db89b1a1b3379b4e46efbd6180ecda066d1907cc4bc0531a6bfef5bb230.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/3c475db89b1a1b3379b4e46efbd6180ecda066d1907cc4bc0531a6bfef5bb230.jpg)

![3eba70671a1ff2798c404812acd6a0c0d19a40deecea21c25e5aec25bf9607db.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/3eba70671a1ff2798c404812acd6a0c0d19a40deecea21c25e5aec25bf9607db.jpg)

![439d7fe1fc44af843427fa34639fa6c99da6b11fcb6ddca19244c32cc3440561.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/439d7fe1fc44af843427fa34639fa6c99da6b11fcb6ddca19244c32cc3440561.jpg)

![4a2c48493eb246ba0960958bd5ca09822f1f761d2d1f2fbcb32c839d4da24e3a.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/4a2c48493eb246ba0960958bd5ca09822f1f761d2d1f2fbcb32c839d4da24e3a.jpg)

![4bdcd95910515180eb942f91e15323c0ed3cff827dee277503488a27f7f86dc8.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/4bdcd95910515180eb942f91e15323c0ed3cff827dee277503488a27f7f86dc8.jpg)

![59194eebae730c097ee73b33124d11bb1f323989e0123cdce4da738ef8af493e.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/59194eebae730c097ee73b33124d11bb1f323989e0123cdce4da738ef8af493e.jpg)

![710fa028d991c98bcdade736e17357f45cdf72c448e3d0af70c5afbe1b2b8230.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/710fa028d991c98bcdade736e17357f45cdf72c448e3d0af70c5afbe1b2b8230.jpg)

![8fa25de9cf5fee24e4c6712a9ff2bdd71e6c57c41d24acba6f68f8bdb10540f2.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/8fa25de9cf5fee24e4c6712a9ff2bdd71e6c57c41d24acba6f68f8bdb10540f2.jpg)

![a5dbd0fe6efcc6a556b963aa293a43eec48e35641b67ebc507cf7c79b5cdc747.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/a5dbd0fe6efcc6a556b963aa293a43eec48e35641b67ebc507cf7c79b5cdc747.jpg)

![a70a885818ab4ba2f1db098954c62ca94b220babe55bbb6f9d29be26a1b1c2e5.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/a70a885818ab4ba2f1db098954c62ca94b220babe55bbb6f9d29be26a1b1c2e5.jpg)

![abd50ae91bfd71ffeec939ba73e53a6d4f81a9599ab2578b2f774d9b5551b59d.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/abd50ae91bfd71ffeec939ba73e53a6d4f81a9599ab2578b2f774d9b5551b59d.jpg)

![bd02343f7dc26c4c5e91d30cfaf873f61eb3ca560d5020f37a0dbe770651b5eb.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/bd02343f7dc26c4c5e91d30cfaf873f61eb3ca560d5020f37a0dbe770651b5eb.jpg)

![bd5a58aad858e17c818d92a3d54bbc77944112762fe7f251b6855abf67b32bde.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/bd5a58aad858e17c818d92a3d54bbc77944112762fe7f251b6855abf67b32bde.jpg)

![cd7c71f26de871c4f72b87653168795efb9c9e65fa5e4aec81e8d7d79d90799e.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/cd7c71f26de871c4f72b87653168795efb9c9e65fa5e4aec81e8d7d79d90799e.jpg)

![e23d1164ffdd9b475bca788b4784c2155091de9d6fe8e69f02538ca00355b00c.jpg](../icml_results/5728_Re-FORC_%20Adaptive%20Reward%20Prediction%20for%20Efficient%20Chain-of-Thought%20Reasoning/images/e23d1164ffdd9b475bca788b4784c2155091de9d6fe8e69f02538ca00355b00c.jpg)

<a id="paper-5680"></a>

## Shifting the Breaking Point of Flow Matching for Multi-Instance Editing


### Images

![09f7bb74d5e616dcc7809dae1671cc617a7f64cdacb46decc834d7d4f2ea01e1.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/09f7bb74d5e616dcc7809dae1671cc617a7f64cdacb46decc834d7d4f2ea01e1.jpg)

![0bb6bd0396ebcd034e34df2f848598c5da3f23a8ece0bb4010e4e296b3b078de.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/0bb6bd0396ebcd034e34df2f848598c5da3f23a8ece0bb4010e4e296b3b078de.jpg)

![179a1323e16c357566d4c30934093fa46947a6c9cc50d09f8fa28828fe65f906.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/179a1323e16c357566d4c30934093fa46947a6c9cc50d09f8fa28828fe65f906.jpg)

![48518fd63e139b8d26c5626bea2939f9e22bdbd52ed61c9cf277c056a291e1fe.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/48518fd63e139b8d26c5626bea2939f9e22bdbd52ed61c9cf277c056a291e1fe.jpg)

![6b901d05b36cef7c80e58ae20a9a46d26b58c2a1fb134971b7498b7dcd1f5fd2.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/6b901d05b36cef7c80e58ae20a9a46d26b58c2a1fb134971b7498b7dcd1f5fd2.jpg)

![93caae5035ff8bcc2fb5f10cd03a9d4208f42b44b0de0b329df8cc1795b8322a.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/93caae5035ff8bcc2fb5f10cd03a9d4208f42b44b0de0b329df8cc1795b8322a.jpg)

![9a2a277d8768639d9199c9c6d6e510a8870ac38ce93e9e0f7cc6b324e7c485b5.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/9a2a277d8768639d9199c9c6d6e510a8870ac38ce93e9e0f7cc6b324e7c485b5.jpg)

![a949cfe4aa89c054f8c752384441602e57d5bcae3f33e8a6f6f9e3cafb84a156.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/a949cfe4aa89c054f8c752384441602e57d5bcae3f33e8a6f6f9e3cafb84a156.jpg)

![b872cabb811e9a5b4971740360fc97e3921ef2edef4f26eed303e67e6bddcc47.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/b872cabb811e9a5b4971740360fc97e3921ef2edef4f26eed303e67e6bddcc47.jpg)

![c1fcf1a70a3dffafa7ba643c18f578e010adff4c09c89440f26a52b7776429e0.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/c1fcf1a70a3dffafa7ba643c18f578e010adff4c09c89440f26a52b7776429e0.jpg)

![d0a3edf6bf17c9c65f58039b76fe053585dab52f8dc38d89f9b789f60a67d735.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/d0a3edf6bf17c9c65f58039b76fe053585dab52f8dc38d89f9b789f60a67d735.jpg)

![dc164d9b975cfea9cc2f58cd0deb1416dafb9154af6dfa70654566b3bdbcfd25.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/dc164d9b975cfea9cc2f58cd0deb1416dafb9154af6dfa70654566b3bdbcfd25.jpg)

![deceabaea205c487646c732f28f5b9ecc4c8429965807254075c497bbe0dea0d.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/deceabaea205c487646c732f28f5b9ecc4c8429965807254075c497bbe0dea0d.jpg)

![e886c5595c3df1aae64c0fd077e2d2eb7cf7d1810e1169ea99a2774e3c713d80.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/e886c5595c3df1aae64c0fd077e2d2eb7cf7d1810e1169ea99a2774e3c713d80.jpg)

![edcc8a451a03837e2583992987fdbd7f1ec3e8bb4abac1ae275d6b33d0c238d7.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/images/edcc8a451a03837e2583992987fdbd7f1ec3e8bb4abac1ae275d6b33d0c238d7.jpg)

### Tables

![01907530205cf2196e9fa6fc75f4b3598a394a5895783982e5eb141142f61cc1.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/01907530205cf2196e9fa6fc75f4b3598a394a5895783982e5eb141142f61cc1.jpg)

![074035dbf5a1e94605256a29edd319082738a77e0b0d0d7a10c4786187ce8ec3.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/074035dbf5a1e94605256a29edd319082738a77e0b0d0d7a10c4786187ce8ec3.jpg)

![693a700a590ac59053918e13cffdd2818ff7b3bd4b13e4db7ef6280a53dcca50.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/693a700a590ac59053918e13cffdd2818ff7b3bd4b13e4db7ef6280a53dcca50.jpg)

![9b48bd3b9954a01fb09c4054e19e2ca9fe6b0b4c60615a3125b9b29a94d96ffe.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/9b48bd3b9954a01fb09c4054e19e2ca9fe6b0b4c60615a3125b9b29a94d96ffe.jpg)

![b8e4b3153d0c7dc570f7be53dbc877409bee6ca6107c0fc9dfd4444f0c96398a.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/b8e4b3153d0c7dc570f7be53dbc877409bee6ca6107c0fc9dfd4444f0c96398a.jpg)

![d54382a1a810907eaea2bb49e60360f7552415cd37103081613242bb768c4e79.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/d54382a1a810907eaea2bb49e60360f7552415cd37103081613242bb768c4e79.jpg)

![d65c261d459d7270815e576a7048e19d88f0951c432024b2c1f0a0f226e3e951.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/d65c261d459d7270815e576a7048e19d88f0951c432024b2c1f0a0f226e3e951.jpg)

![e63d4d80acfd60e1a769b67d08c549a3547170a713d76132f36bfaa8d99bd789.jpg](../icml_results/5729_Shifting%20the%20Breaking%20Point%20of%20Flow%20Matching%20for%20Multi-Instance%20Editing/tables/e63d4d80acfd60e1a769b67d08c549a3547170a713d76132f36bfaa8d99bd789.jpg)

<a id="paper-5681"></a>

## Weasel: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection


### Images

![6b439222980eb71e637f89ce4cb3dbd9e1a844a690da0b13863cb31a0d216264.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/6b439222980eb71e637f89ce4cb3dbd9e1a844a690da0b13863cb31a0d216264.jpg)

![833d1db60482b6367d943b0a5502efb15eb84ce10d2de79c648ad2a3592c454c.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/833d1db60482b6367d943b0a5502efb15eb84ce10d2de79c648ad2a3592c454c.jpg)

![83cfb1e0138d626a1e8f0b997e8b75f722c66c4a3d7a92f5ca9bebf467cc7c50.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/83cfb1e0138d626a1e8f0b997e8b75f722c66c4a3d7a92f5ca9bebf467cc7c50.jpg)

![a316456f325d955d4b5f13d1e9ddc346f21cf34ff186019e2e2667bc0499900a.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/a316456f325d955d4b5f13d1e9ddc346f21cf34ff186019e2e2667bc0499900a.jpg)

![bb2a2c1b6ed15d474c839be273980c5e5c94434a3ed4fc8a984367c2c0b31039.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/bb2a2c1b6ed15d474c839be273980c5e5c94434a3ed4fc8a984367c2c0b31039.jpg)

![cd0eea87c159e6bea26fbdd430f8ba4a78acb0104565def52ff8b90e586f57c6.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/cd0eea87c159e6bea26fbdd430f8ba4a78acb0104565def52ff8b90e586f57c6.jpg)

![d9f51a2a4915d7b872232fd1884edaf1fa98dafbae324febc7310362d62217ab.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/d9f51a2a4915d7b872232fd1884edaf1fa98dafbae324febc7310362d62217ab.jpg)

![f69d7510f46d63509490f8dad3d1fb51acc86b3c4e1253224f2dcc7f09f1c378.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/images/f69d7510f46d63509490f8dad3d1fb51acc86b3c4e1253224f2dcc7f09f1c378.jpg)

### Tables

![02e4417393f503a5106bf8aa09f61f150d8576b6fe697ad9eda379f9e4393253.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/02e4417393f503a5106bf8aa09f61f150d8576b6fe697ad9eda379f9e4393253.jpg)

![10feae1eaf1391f8b8ff37ab359c8f800f29759ab070d811ca68a18e461e0709.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/10feae1eaf1391f8b8ff37ab359c8f800f29759ab070d811ca68a18e461e0709.jpg)

![125b314e1308e930390352e7bfc7f5311185a5df11b3b547854e5d8a70a756b7.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/125b314e1308e930390352e7bfc7f5311185a5df11b3b547854e5d8a70a756b7.jpg)

![1dd740ac0075d47067ca4b09e76bc9c74d439958d030ccd80d4bc674e5394e40.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/1dd740ac0075d47067ca4b09e76bc9c74d439958d030ccd80d4bc674e5394e40.jpg)

![3be14faae9e686b45837a9ad5d664969048404e69b9c5d1cfc73092fc42dd0aa.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/3be14faae9e686b45837a9ad5d664969048404e69b9c5d1cfc73092fc42dd0aa.jpg)

![94589eb9e01864c51f36bbbc16ceffdac23937fdedf88f15bdcf06f1b2ec0885.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/94589eb9e01864c51f36bbbc16ceffdac23937fdedf88f15bdcf06f1b2ec0885.jpg)

![a39a6d6a69adc856b9d8065935153ab61f8b6294600b56c881283951be9d6bb3.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/a39a6d6a69adc856b9d8065935153ab61f8b6294600b56c881283951be9d6bb3.jpg)

![a7bebf38fda73c23aa83e215a9ff55ac2ef67abc56dfd0bc3eec2df834a3ad63.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/a7bebf38fda73c23aa83e215a9ff55ac2ef67abc56dfd0bc3eec2df834a3ad63.jpg)

![b335e497fdbfaf20631b34eaa9d6c5f33eda39aae54f73dbd1435be52344a7bc.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/b335e497fdbfaf20631b34eaa9d6c5f33eda39aae54f73dbd1435be52344a7bc.jpg)

![b3ef9ee9fe3e9e68d2d9b3e09b4ca2b2715471ab30f6c485e99295c360024e0c.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/b3ef9ee9fe3e9e68d2d9b3e09b4ca2b2715471ab30f6c485e99295c360024e0c.jpg)

![c492173cd4471e857a92b8aeb2f69b2da41ab73fd8e242e7d20d96637d45aea1.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/c492173cd4471e857a92b8aeb2f69b2da41ab73fd8e242e7d20d96637d45aea1.jpg)

![d5cd8ad4d0b70a8884f1dbc57690a9cf5fb0143a24cf23985e9eb9fab24be5b1.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/d5cd8ad4d0b70a8884f1dbc57690a9cf5fb0143a24cf23985e9eb9fab24be5b1.jpg)

![f7c0364a5b952e98440067113497d57c8a181584f1821a801f681f671f502f6c.jpg](../icml_results/5730_Weasel_%20Out-of-Domain%20Generalization%20for%20Web%20Agents%20via%20Importance-Diversity%20Data%20Selection/tables/f7c0364a5b952e98440067113497d57c8a181584f1821a801f681f671f502f6c.jpg)

<a id="paper-5682"></a>

## Differentially Private Submodular Maximization with a Knapsack Constraint


### Images

![44a803a26ca9b36856959b67bf797b09b835e54a6a9b247a077fdb82b0a00e9b.jpg](../icml_results/5731_Differentially%20Private%20Submodular%20Maximization%20with%20a%20Knapsack%20Constraint/images/44a803a26ca9b36856959b67bf797b09b835e54a6a9b247a077fdb82b0a00e9b.jpg)

![6dc2e0e6790fee4dacd646ec4a83956b947245d58b2ae71681dd18662cf8f04e.jpg](../icml_results/5731_Differentially%20Private%20Submodular%20Maximization%20with%20a%20Knapsack%20Constraint/images/6dc2e0e6790fee4dacd646ec4a83956b947245d58b2ae71681dd18662cf8f04e.jpg)

![a08458473eb125c3e2947e311dc8ad9682a6aedc6d7afc78f1d22f46726f4a09.jpg](../icml_results/5731_Differentially%20Private%20Submodular%20Maximization%20with%20a%20Knapsack%20Constraint/images/a08458473eb125c3e2947e311dc8ad9682a6aedc6d7afc78f1d22f46726f4a09.jpg)

![b736a0ddd363aef878b85107821a2700c808418028a6b9cf3ffef8eafebaf80c.jpg](../icml_results/5731_Differentially%20Private%20Submodular%20Maximization%20with%20a%20Knapsack%20Constraint/images/b736a0ddd363aef878b85107821a2700c808418028a6b9cf3ffef8eafebaf80c.jpg)

### Tables

![ba88aea8f7c7e800fb88ed491af85c363219cc3050851f937c0956912d4a1772.jpg](../icml_results/5731_Differentially%20Private%20Submodular%20Maximization%20with%20a%20Knapsack%20Constraint/tables/ba88aea8f7c7e800fb88ed491af85c363219cc3050851f937c0956912d4a1772.jpg)

<a id="paper-5683"></a>

## Geometry-Aware Tabular Diffusion


### Images

![1b17978fd1f433d5443f5afe56730201728321dde58a52b72b0c727c3d958c14.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/1b17978fd1f433d5443f5afe56730201728321dde58a52b72b0c727c3d958c14.jpg)

![1c1f17f11c448b27e6af49b5dcc4a1e58206c94fc3c27ae09bcb0b919e0f8f5f.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/1c1f17f11c448b27e6af49b5dcc4a1e58206c94fc3c27ae09bcb0b919e0f8f5f.jpg)

![1d3a08ac9428a6e14359df3e2c534aeab302d2c08974ec856250a06d9290a5de.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/1d3a08ac9428a6e14359df3e2c534aeab302d2c08974ec856250a06d9290a5de.jpg)

![343e616aabad55d6b33510589c14c9d13e910b39ded7a38dc71676afee53c3ed.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/343e616aabad55d6b33510589c14c9d13e910b39ded7a38dc71676afee53c3ed.jpg)

![4b8a84c784ef04113444c7757a5c9c90d759ae793f11331b7f17cae2863d9ce1.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/4b8a84c784ef04113444c7757a5c9c90d759ae793f11331b7f17cae2863d9ce1.jpg)

![5f7111f22ff700e8c75034c81de9c1eb910156e83d88d91c8a2f217d40fbc8de.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/5f7111f22ff700e8c75034c81de9c1eb910156e83d88d91c8a2f217d40fbc8de.jpg)

![618d60500084740749fa2826ce7c786c695f7c52784ed7bff7f4fbda55efd481.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/618d60500084740749fa2826ce7c786c695f7c52784ed7bff7f4fbda55efd481.jpg)

![656b733061cfa6925533fdbac93c16d6ddeb77076e8bca4f3fb80aa09dc553dc.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/656b733061cfa6925533fdbac93c16d6ddeb77076e8bca4f3fb80aa09dc553dc.jpg)

![bd3b330cd3a6145ed4bc1728acae027afc5f9b8d1c90f2389bbd73075783adb8.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/bd3b330cd3a6145ed4bc1728acae027afc5f9b8d1c90f2389bbd73075783adb8.jpg)

![dbe6e4d18f0e6ca49962cd1fb1149717b75e770975869a6dcf644be788c9ab23.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/images/dbe6e4d18f0e6ca49962cd1fb1149717b75e770975869a6dcf644be788c9ab23.jpg)

### Tables

![0c9093c2d0a7d388b373f186632a2848495259ac008ca623ba2cc49c1b12e855.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/0c9093c2d0a7d388b373f186632a2848495259ac008ca623ba2cc49c1b12e855.jpg)

![1379fdc7e71e2745d8e60a170ffff88ad7932a0c1af8c898eccf96ab1acad226.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/1379fdc7e71e2745d8e60a170ffff88ad7932a0c1af8c898eccf96ab1acad226.jpg)

![145ca02f427b43c9e2aad517eaabfeaeb1e3fa095b1de1205b81775c2dc5f491.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/145ca02f427b43c9e2aad517eaabfeaeb1e3fa095b1de1205b81775c2dc5f491.jpg)

![2118ca373e18df05a7d9d1068a476a5b7867e9c0dd057cf55d0d2834a44b50c3.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/2118ca373e18df05a7d9d1068a476a5b7867e9c0dd057cf55d0d2834a44b50c3.jpg)

![267cb51c16d9e85feb206b2277c659a07eca09f7dbc699af2988d50f21d0b0ef.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/267cb51c16d9e85feb206b2277c659a07eca09f7dbc699af2988d50f21d0b0ef.jpg)

![631a5a0cb207df0235af5cebb8acb22cf29700bb078e73554b01541b7854cec4.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/631a5a0cb207df0235af5cebb8acb22cf29700bb078e73554b01541b7854cec4.jpg)

![6ab8b6debac907665ae5ae27015422e792a99420f5e1bbf0d30d6e3175b044de.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/6ab8b6debac907665ae5ae27015422e792a99420f5e1bbf0d30d6e3175b044de.jpg)

![741afb63941eae651feb934504f6104ae3b0adba9f1cdef59a2a52173bbd6f44.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/741afb63941eae651feb934504f6104ae3b0adba9f1cdef59a2a52173bbd6f44.jpg)

![86198bc2c7bad73b0e300aacb2dba4f62aeb379b8a469e93c8ee3e66c58260dd.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/86198bc2c7bad73b0e300aacb2dba4f62aeb379b8a469e93c8ee3e66c58260dd.jpg)

![89d1302969818d45e6e411c25289f848ec90faf5abdceb3b17785854152ef364.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/89d1302969818d45e6e411c25289f848ec90faf5abdceb3b17785854152ef364.jpg)

![92c76d01701b6835bfeb0c9c737c7db6cc83974d4a99ff29395bd4d801eae89d.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/92c76d01701b6835bfeb0c9c737c7db6cc83974d4a99ff29395bd4d801eae89d.jpg)

![9b91ec29ed80562358f679bf805affb2061b91b989f7610209c717ed163bf6fc.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/9b91ec29ed80562358f679bf805affb2061b91b989f7610209c717ed163bf6fc.jpg)

![a7c63ebadc66082476e82ed8ce3e6e81a11c30580a2e01041ff20b9d99647e3f.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/a7c63ebadc66082476e82ed8ce3e6e81a11c30580a2e01041ff20b9d99647e3f.jpg)

![a7d30b1c657c7ea1059edab4c765c3c207cc7873c4e327a63fa33adc7bacbbc0.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/a7d30b1c657c7ea1059edab4c765c3c207cc7873c4e327a63fa33adc7bacbbc0.jpg)

![b1b2e87558ba332d01e9f072dfd5b615041656a4ef2d08ba5b44e06551293354.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/b1b2e87558ba332d01e9f072dfd5b615041656a4ef2d08ba5b44e06551293354.jpg)

![b4bd0db1a74744e5acdfcc917a143bfc0e06d4629f27728ebd38aa96a2bd65b8.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/b4bd0db1a74744e5acdfcc917a143bfc0e06d4629f27728ebd38aa96a2bd65b8.jpg)

![ca692c3a5e99d9689ad9e3eab61d9fa5584bcddb00ecbe25c297f904ba0f52f4.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/ca692c3a5e99d9689ad9e3eab61d9fa5584bcddb00ecbe25c297f904ba0f52f4.jpg)

![ce7848c331c4943e8b6dd33c8064508d21d7e6444b9e476263b1257b22e134db.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/ce7848c331c4943e8b6dd33c8064508d21d7e6444b9e476263b1257b22e134db.jpg)

![d5cf04bfe0edee174235b6f56ffe21c6534ec123049a9e1ef5ba3b247c28a101.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/d5cf04bfe0edee174235b6f56ffe21c6534ec123049a9e1ef5ba3b247c28a101.jpg)

![dd1ced31d8c88f862c91e579ff8bc8c715d4ca746a28fb17ee13864fc5868b0d.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/dd1ced31d8c88f862c91e579ff8bc8c715d4ca746a28fb17ee13864fc5868b0d.jpg)

![eb65d81b7dba7df4cd6eb8f7b96f57ba0cc5b19597a8f4bf4ebde564ce2dc7fa.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/eb65d81b7dba7df4cd6eb8f7b96f57ba0cc5b19597a8f4bf4ebde564ce2dc7fa.jpg)

![ef0a13768ce530db2e4748a3a034c8ff557b307ad2f6f28bb0b668ffa9038f05.jpg](../icml_results/5732_Geometry-Aware%20Tabular%20Diffusion/tables/ef0a13768ce530db2e4748a3a034c8ff557b307ad2f6f28bb0b668ffa9038f05.jpg)

<a id="paper-5684"></a>

## When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs


### Images

![0b48b77062e07aeb7d93cf097f47f7ad74c3f33949dba0d79c56ccc07b562a8e.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/0b48b77062e07aeb7d93cf097f47f7ad74c3f33949dba0d79c56ccc07b562a8e.jpg)

![16973752fa7e4990273a3f344f28fd121c9576845eaf0ea5ed17edd263464b3c.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/16973752fa7e4990273a3f344f28fd121c9576845eaf0ea5ed17edd263464b3c.jpg)

![1ae12c73e8a1bdc33c6c431ee65f8aec32fb24204e57f403cf164924cd9cba07.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/1ae12c73e8a1bdc33c6c431ee65f8aec32fb24204e57f403cf164924cd9cba07.jpg)

![201cb574b2e1c3b4cda95883295e594868539b3543cba5fc226c6d3c4a3468ec.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/201cb574b2e1c3b4cda95883295e594868539b3543cba5fc226c6d3c4a3468ec.jpg)

![241067d7950aff5da7ad62d85b77804176aca2feb3ebc54ffa84befad67ac61e.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/241067d7950aff5da7ad62d85b77804176aca2feb3ebc54ffa84befad67ac61e.jpg)

![275483246bb859119893ef4ee8828edf4178e30d97099648531d69f1f2fba204.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/275483246bb859119893ef4ee8828edf4178e30d97099648531d69f1f2fba204.jpg)

![2b609ed3cade832404fa29eb0298c2a92526767721acd875de2bb4c75c99fca7.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/2b609ed3cade832404fa29eb0298c2a92526767721acd875de2bb4c75c99fca7.jpg)

![4f2c8dd1069929073b2a89ca87796fa859668eb3b0a96523f3dad121d723da5c.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/4f2c8dd1069929073b2a89ca87796fa859668eb3b0a96523f3dad121d723da5c.jpg)

![515685c16655dd85bf965f9977a42177187183f5a961156ee8488f8a085cf8aa.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/515685c16655dd85bf965f9977a42177187183f5a961156ee8488f8a085cf8aa.jpg)

![59ebb2be03cb720a40d422791121bfefcd88e2a625944a560a9366a7ea869eb2.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/59ebb2be03cb720a40d422791121bfefcd88e2a625944a560a9366a7ea869eb2.jpg)

![5dc3300e97502ea95f41f7dc27cd80cf73a0eb66f21bd38324bd7f2ddce618d9.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/5dc3300e97502ea95f41f7dc27cd80cf73a0eb66f21bd38324bd7f2ddce618d9.jpg)

![60fb365a0a2dc8d9ea5d2eb88b71482593af617523795bb608c66e29c1f65019.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/60fb365a0a2dc8d9ea5d2eb88b71482593af617523795bb608c66e29c1f65019.jpg)

![6624bcb315a3587ecbddba17c2f2909382bfcdf28c49fd3a0a07f3048a778959.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/6624bcb315a3587ecbddba17c2f2909382bfcdf28c49fd3a0a07f3048a778959.jpg)

![71e33456aa8bcd0a290339e26bcb7ef212acb1a5523f66ac52fe97e39b59dc8d.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/71e33456aa8bcd0a290339e26bcb7ef212acb1a5523f66ac52fe97e39b59dc8d.jpg)

![72522d4744112f62fb57f2705368c50dbcd3863f99acad61323a55193dbcf610.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/72522d4744112f62fb57f2705368c50dbcd3863f99acad61323a55193dbcf610.jpg)

![79ed9b39a5c26fe74d9796e7c8baea58cb551d926a814f395b41897e6efea722.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/79ed9b39a5c26fe74d9796e7c8baea58cb551d926a814f395b41897e6efea722.jpg)

![7e467b14ca238340814c96d81657da2fb24017918ff5af31d5ee66f2f1bb1399.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/7e467b14ca238340814c96d81657da2fb24017918ff5af31d5ee66f2f1bb1399.jpg)

![90bbab0668550cbb3dc7328f5313cae2e473d20906c310f69d071f8a4503bd66.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/90bbab0668550cbb3dc7328f5313cae2e473d20906c310f69d071f8a4503bd66.jpg)

![9305ef5cf3b4174890f9061c3fb7ca65a4b7347f17d5bf340d5cded734b4e918.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/9305ef5cf3b4174890f9061c3fb7ca65a4b7347f17d5bf340d5cded734b4e918.jpg)

![9948f59c98c3cf935b02023485272748d51175a14dacc4f24417a4afb14c1768.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/9948f59c98c3cf935b02023485272748d51175a14dacc4f24417a4afb14c1768.jpg)

![9dcfd8b2eccbb1e2aeb2cc8af5b8e77e4830ff4ed9e53fe8d7896a3b2a0873f1.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/9dcfd8b2eccbb1e2aeb2cc8af5b8e77e4830ff4ed9e53fe8d7896a3b2a0873f1.jpg)

![a27b57419f483f5450923c8f991df36dc925f9334f55e3284dc8668494a11bb8.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/a27b57419f483f5450923c8f991df36dc925f9334f55e3284dc8668494a11bb8.jpg)

![b37a0b3b8a24fe559e4c77dd81bd81cecf0afb1e6a9557bade1e4c9ee4e80afa.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/b37a0b3b8a24fe559e4c77dd81bd81cecf0afb1e6a9557bade1e4c9ee4e80afa.jpg)

![b8e71f2ad8dd48441b9599fa035343ccc34522fdec13f07dd7806d62cf83544c.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/b8e71f2ad8dd48441b9599fa035343ccc34522fdec13f07dd7806d62cf83544c.jpg)

![bc001e932105af5a6096c9b479f48546b985221824b2512f96d7f2f6485610aa.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/bc001e932105af5a6096c9b479f48546b985221824b2512f96d7f2f6485610aa.jpg)

![c421ab4467e87e515533e5ee30ba7aa0ef1d1d55f017e13a53bbb9c164c848d8.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/c421ab4467e87e515533e5ee30ba7aa0ef1d1d55f017e13a53bbb9c164c848d8.jpg)

![c599e0d40348e041607c3c3af861ec7c6030491ed9726ce0d7d7391001bd9c08.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/c599e0d40348e041607c3c3af861ec7c6030491ed9726ce0d7d7391001bd9c08.jpg)

![c7c72d5676eb5f199036c3bfd95955b29bfb030be581a03709706bc9d755d5d1.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/c7c72d5676eb5f199036c3bfd95955b29bfb030be581a03709706bc9d755d5d1.jpg)

![ce3867771243f1ce4251d4e94299d8b6568d9e4e2e677858f220ee59b9d7ccc4.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/ce3867771243f1ce4251d4e94299d8b6568d9e4e2e677858f220ee59b9d7ccc4.jpg)

![d2802072a2ef05dfef9980a78c9cbf73f53dd4fa8abe50ebe7856e5622379dc4.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/d2802072a2ef05dfef9980a78c9cbf73f53dd4fa8abe50ebe7856e5622379dc4.jpg)

![d32f7dec91fe79faa1d01fcc9ad69914aa441336cdac197e4799418b49dab3d6.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/d32f7dec91fe79faa1d01fcc9ad69914aa441336cdac197e4799418b49dab3d6.jpg)

![d4e22a88531fcfdd7ed90bd0faa3cb782fef92589af4ba51c697eec18aef4621.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/d4e22a88531fcfdd7ed90bd0faa3cb782fef92589af4ba51c697eec18aef4621.jpg)

![da4aec918749a69ac89c3edd951e8c5204bfbea438f799f0d0b325fa5a38668f.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/da4aec918749a69ac89c3edd951e8c5204bfbea438f799f0d0b325fa5a38668f.jpg)

![dccbcdb76a9d7d76ad4f6c25181ed07ae0555037226b1d1a26c0f593c78305c0.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/dccbcdb76a9d7d76ad4f6c25181ed07ae0555037226b1d1a26c0f593c78305c0.jpg)

![def73a0bb8e72cda490e6c7cd52b59671bc6af0eff1dfe27e36a8ce96132441a.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/def73a0bb8e72cda490e6c7cd52b59671bc6af0eff1dfe27e36a8ce96132441a.jpg)

![ebb4642bb939af2a8b006cb36a4cca34f8c9bbe18a860f35d3d9dab850e51110.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/ebb4642bb939af2a8b006cb36a4cca34f8c9bbe18a860f35d3d9dab850e51110.jpg)

![ebf639a317b37327e86c9a9d725831a5689ef2234a34babe560550fb13ea46d9.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/ebf639a317b37327e86c9a9d725831a5689ef2234a34babe560550fb13ea46d9.jpg)

![eda4596ed24ef94d6ce3158851dd4a1e9199022c471370e156c14a91b6320053.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/eda4596ed24ef94d6ce3158851dd4a1e9199022c471370e156c14a91b6320053.jpg)

![edec81a23f733173afc7625cd0d2124224f0ab9ebd148d899513bcd9db40e883.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/edec81a23f733173afc7625cd0d2124224f0ab9ebd148d899513bcd9db40e883.jpg)

![f216ae60c655d3f1bc03bca329dbd022db5825c0ff218725aca0d66b28a20753.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/f216ae60c655d3f1bc03bca329dbd022db5825c0ff218725aca0d66b28a20753.jpg)

![fa4c6cb39b95720408352e1c1b7776fe7d87d11d4307955992971db18592f0a1.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/fa4c6cb39b95720408352e1c1b7776fe7d87d11d4307955992971db18592f0a1.jpg)

![fdf4eb7b78ef4c7989b4b20e86fc173c5cc648e71ea11f965e0599e5b33d3d96.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/fdf4eb7b78ef4c7989b4b20e86fc173c5cc648e71ea11f965e0599e5b33d3d96.jpg)

![ff186b5115392fe75385783df2693058e2c984b3a280e4610349f423e50adf88.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/images/ff186b5115392fe75385783df2693058e2c984b3a280e4610349f423e50adf88.jpg)

### Tables

![06b8958a810c593878af3a886a00f225cc81624352a4a02584cda97e69c085ab.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/06b8958a810c593878af3a886a00f225cc81624352a4a02584cda97e69c085ab.jpg)

![072f695100d43100815a35859a26c7baa2761368891dbf51acf08459590c848d.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/072f695100d43100815a35859a26c7baa2761368891dbf51acf08459590c848d.jpg)

![10b9303f9c670a9002a7ec09562ce1e3477513062066609341463dae3aa19b53.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/10b9303f9c670a9002a7ec09562ce1e3477513062066609341463dae3aa19b53.jpg)

![21ec2ebb359c279b849f915cbfbb291229b707be944efb3f51ed9a6711c6fc40.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/21ec2ebb359c279b849f915cbfbb291229b707be944efb3f51ed9a6711c6fc40.jpg)

![24add5a0d625d55febf43be2eda190e2d7f36df0ca1348276e339667417590c8.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/24add5a0d625d55febf43be2eda190e2d7f36df0ca1348276e339667417590c8.jpg)

![31b256c3f6660ebc236e9e42c04dfe5ae9dd786e1e1404af68392dbc696cc550.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/31b256c3f6660ebc236e9e42c04dfe5ae9dd786e1e1404af68392dbc696cc550.jpg)

![32a86a961cd08774c90fab4cf4dc0f9a4c736cd06089bb03b181d49f1d8c987c.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/32a86a961cd08774c90fab4cf4dc0f9a4c736cd06089bb03b181d49f1d8c987c.jpg)

![3c861ee3ae9484b11b2408f498ca8921457406d27facd8968abb315bfdf5b884.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/3c861ee3ae9484b11b2408f498ca8921457406d27facd8968abb315bfdf5b884.jpg)

![4aa5a2d06174932247749e89d386d9d49e5eabf919a9176064a52bb0d2767d73.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/4aa5a2d06174932247749e89d386d9d49e5eabf919a9176064a52bb0d2767d73.jpg)

![55f6a7daed536a50c911114df1114b609812be67ae7269dd43e8861ba0b74327.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/55f6a7daed536a50c911114df1114b609812be67ae7269dd43e8861ba0b74327.jpg)

![595db28df9c7602cff26b91d23770c6bbfe73f179e90a7e063385c393693bb65.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/595db28df9c7602cff26b91d23770c6bbfe73f179e90a7e063385c393693bb65.jpg)

![6a1a8e7f576b6fecd9e535cdb33d934a33252b0c9ce5ebcc7e72c398e3ead98c.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/6a1a8e7f576b6fecd9e535cdb33d934a33252b0c9ce5ebcc7e72c398e3ead98c.jpg)

![750ed903b7ec4454ca5e427957e1078ce99c9e896d106ad97d459b6167017a95.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/750ed903b7ec4454ca5e427957e1078ce99c9e896d106ad97d459b6167017a95.jpg)

![7f4083096daea46c84d4cb2abfdc34c8d7c986a23e95db456f72f8bdb5251407.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/7f4083096daea46c84d4cb2abfdc34c8d7c986a23e95db456f72f8bdb5251407.jpg)

![81aefa1609b076e5fd0c2175d15a214ac7aa3b615d72cf0096675c7313f01ff7.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/81aefa1609b076e5fd0c2175d15a214ac7aa3b615d72cf0096675c7313f01ff7.jpg)

![9c2c1a95fa75b873964a17675284bd2b110e0f9c41ef267ea3ec3359507e0534.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/9c2c1a95fa75b873964a17675284bd2b110e0f9c41ef267ea3ec3359507e0534.jpg)

![9efabcd7bf138b6355ba5e7b10b18150e6ba3e11fdc4051cd5752720ad171712.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/9efabcd7bf138b6355ba5e7b10b18150e6ba3e11fdc4051cd5752720ad171712.jpg)

![a554b67fff0ae4758b8557f23d605617338d6b99f84b54c14924f972a39e601f.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/a554b67fff0ae4758b8557f23d605617338d6b99f84b54c14924f972a39e601f.jpg)

![b3e47e2a800d2521c2f470321545e63b4fbdc17e6001a029fefe842cc1260712.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/b3e47e2a800d2521c2f470321545e63b4fbdc17e6001a029fefe842cc1260712.jpg)

![b515c3f0e1e782f185f1ccf0116c8cd5a497f329dc59a281edf113306ee95201.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/b515c3f0e1e782f185f1ccf0116c8cd5a497f329dc59a281edf113306ee95201.jpg)

![b8d3c1c82cea0cf1a7887e004a313802d492f42cc89e3c9b1d062be3e8789133.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/b8d3c1c82cea0cf1a7887e004a313802d492f42cc89e3c9b1d062be3e8789133.jpg)

![be7c72d144c69e35a9d2a3e7e5e1920d55d30265a397a43b8dd7dbfc11484e6b.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/be7c72d144c69e35a9d2a3e7e5e1920d55d30265a397a43b8dd7dbfc11484e6b.jpg)

![c8e1121e598a4a15b115d667a928b86ff16117688d5850b011e56a08fc3c6f48.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/c8e1121e598a4a15b115d667a928b86ff16117688d5850b011e56a08fc3c6f48.jpg)

![c9102bba3821252d52fba9264dd620888fb6f9d43da733ee1daf25ea29b63430.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/c9102bba3821252d52fba9264dd620888fb6f9d43da733ee1daf25ea29b63430.jpg)

![d7f5bb2370b2763e1388a104ba6e5778e9646ef220b837996a01f9f39e5346dd.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/d7f5bb2370b2763e1388a104ba6e5778e9646ef220b837996a01f9f39e5346dd.jpg)

![dd2d2ade44d5da95ea6db7ce95d6ac90c2ed4dfc2cd8b460e229c274fb6e9a76.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/dd2d2ade44d5da95ea6db7ce95d6ac90c2ed4dfc2cd8b460e229c274fb6e9a76.jpg)

![e20b108aa3fec85f5a4032ea8f7bf0ec8c354a1ddb7b85a91852a5333a4a9c7f.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/e20b108aa3fec85f5a4032ea8f7bf0ec8c354a1ddb7b85a91852a5333a4a9c7f.jpg)

![f077671eafaf1b8f4d71b5c4f8bd18532378d6f7fa0b97439dd085ab58574749.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/f077671eafaf1b8f4d71b5c4f8bd18532378d6f7fa0b97439dd085ab58574749.jpg)

![f4dd70d17bd55f85069be059c07d099bbf12ecb0f3fb5e487d7b369c44046ab3.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/f4dd70d17bd55f85069be059c07d099bbf12ecb0f3fb5e487d7b369c44046ab3.jpg)

![f6db3e913cf0a065f6c26f711c44cf0dc3462b6b36ed27bf01e4d2ed9cdd1b3e.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/f6db3e913cf0a065f6c26f711c44cf0dc3462b6b36ed27bf01e4d2ed9cdd1b3e.jpg)

![ff6c50ec6d2abe87be6d2302334080da9cae4a7d7728f430ab187aa0f5bccf36.jpg](../icml_results/5733_When%20Single%20Answer%20Is%20Not%20Enough_%20Rethinking%20Single-Step%20Retrosynthesis%20Benchmarks%20for%20LLMs/tables/ff6c50ec6d2abe87be6d2302334080da9cae4a7d7728f430ab187aa0f5bccf36.jpg)

<a id="paper-5685"></a>

## REViT: Roto-reflection Equivariant Convolutional Vision Transformer


### Images

![3bddc76f3d2d32d22745151fa16bde47b748d96c238517d6c1ffab5b2b46ce2d.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/images/3bddc76f3d2d32d22745151fa16bde47b748d96c238517d6c1ffab5b2b46ce2d.jpg)

![7cd9655af81ed2c880a1f63570a7f4aee2be1e6408de2ba4ef3da372308cad04.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/images/7cd9655af81ed2c880a1f63570a7f4aee2be1e6408de2ba4ef3da372308cad04.jpg)

![93f197e8ecc09500f3016838bf8a4cdab3783162e2c30732289f3c5c8258bf9a.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/images/93f197e8ecc09500f3016838bf8a4cdab3783162e2c30732289f3c5c8258bf9a.jpg)

![e38cb853066b730905145174446386de11de68febb0b475ff604e3e7bb5490a8.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/images/e38cb853066b730905145174446386de11de68febb0b475ff604e3e7bb5490a8.jpg)

### Tables

![013a1d3f508b0da646687e8e9419936e123625aba40af7c3e10ea727fc77057b.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/013a1d3f508b0da646687e8e9419936e123625aba40af7c3e10ea727fc77057b.jpg)

![21e8bf731650db13283991f17c8f602ceef538b1241b1df4e9d753c836ae5ca9.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/21e8bf731650db13283991f17c8f602ceef538b1241b1df4e9d753c836ae5ca9.jpg)

![367cc8e7d319c44c373a9491ecd0c2af2852483305ac9991596861399a0166cf.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/367cc8e7d319c44c373a9491ecd0c2af2852483305ac9991596861399a0166cf.jpg)

![5247d45206508c13c9e7ee1da71636763b3c8a6165db9aa4c4389e1ad1e408b4.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/5247d45206508c13c9e7ee1da71636763b3c8a6165db9aa4c4389e1ad1e408b4.jpg)

![5d0f488766a06a165638117a14b91096b081a822349189bde15ce5644e55d828.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/5d0f488766a06a165638117a14b91096b081a822349189bde15ce5644e55d828.jpg)

![6037104e91d43aba5a43957f9e2a0ebef29152c1db32b5315422fac02ca85cf7.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/6037104e91d43aba5a43957f9e2a0ebef29152c1db32b5315422fac02ca85cf7.jpg)

![764622043a45d58cfdc2d6c7486388ebac0c731d860f22cc348a05c4222466a0.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/764622043a45d58cfdc2d6c7486388ebac0c731d860f22cc348a05c4222466a0.jpg)

![887359c160b34311eff3f104fe5b2563c9e414fd8d0199b8c1725bd081b495f0.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/887359c160b34311eff3f104fe5b2563c9e414fd8d0199b8c1725bd081b495f0.jpg)

![89519068d6b84eced928f1f5f84549cdf290697e26d6fca8e49a2d38eae1d2b9.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/89519068d6b84eced928f1f5f84549cdf290697e26d6fca8e49a2d38eae1d2b9.jpg)

![8d109e32a8833c31a1ab1410724f198cd91c0e7142142e80470d409e48275ca7.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/8d109e32a8833c31a1ab1410724f198cd91c0e7142142e80470d409e48275ca7.jpg)

![a56f63f687bc9eaa5ca743b92d3505442b0520bd2a00ab97a7b30edc540236da.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/a56f63f687bc9eaa5ca743b92d3505442b0520bd2a00ab97a7b30edc540236da.jpg)

![b3b325b340cc266f138bf1ce57ab7856b8df79928abc91da3fc51140ebd63adf.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/b3b325b340cc266f138bf1ce57ab7856b8df79928abc91da3fc51140ebd63adf.jpg)

![d92d2f56e395f987afbb3813a1f00f9d19cf74af27cdad62f150c9dcb6f4db8b.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/d92d2f56e395f987afbb3813a1f00f9d19cf74af27cdad62f150c9dcb6f4db8b.jpg)

![f353329416c625c02d8646b36a3807c3143e24782bb2540755c1aeba1c9dce88.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/f353329416c625c02d8646b36a3807c3143e24782bb2540755c1aeba1c9dce88.jpg)

![f73f7dcf3b76a337e075cdff9184a9dcdb02ae6852ca966c11869dcc29664629.jpg](../icml_results/5734_REViT_%20Roto-reflection%20Equivariant%20Convolutional%20Vision%20Transformer/tables/f73f7dcf3b76a337e075cdff9184a9dcdb02ae6852ca966c11869dcc29664629.jpg)

<a id="paper-5686"></a>

## Self-Supervised Learning as Discrete Communication


### Images

![04cb66abaac55f31285e50dcb03fde069976aae6b4776d942146031610ad7541.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/04cb66abaac55f31285e50dcb03fde069976aae6b4776d942146031610ad7541.jpg)

![0a08f33884ea8e2064a2f3489c0482c200adde07ca92547bcce67f6af3abc724.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/0a08f33884ea8e2064a2f3489c0482c200adde07ca92547bcce67f6af3abc724.jpg)

![1be699f2cc4163f1259ffa23693a2150c34fe9c51cc6db1898d36094d2938c06.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/1be699f2cc4163f1259ffa23693a2150c34fe9c51cc6db1898d36094d2938c06.jpg)

![5b03338daa0960b5526df1b4b8661c78b368097083b826bb0c202a7e3333a000.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/5b03338daa0960b5526df1b4b8661c78b368097083b826bb0c202a7e3333a000.jpg)

![623956a5d05dc7aaaad2a6ce333e0677cc6196373a6b2a63839adeaa532c3555.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/623956a5d05dc7aaaad2a6ce333e0677cc6196373a6b2a63839adeaa532c3555.jpg)

![6e56fe832272019d55d725ff5af61e8b9224eaef3f2a8288a2ccbe15d3deb590.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/6e56fe832272019d55d725ff5af61e8b9224eaef3f2a8288a2ccbe15d3deb590.jpg)

![7da68660edbabe97fb23377a698650b2735e0b7570865014e15577892c0d42b4.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/7da68660edbabe97fb23377a698650b2735e0b7570865014e15577892c0d42b4.jpg)

![8f4648385ddb4041148d14fdaee567f9020164b47aa738b55c27ef270db7c27a.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/8f4648385ddb4041148d14fdaee567f9020164b47aa738b55c27ef270db7c27a.jpg)

![a3aca1963521b857ce32eab3711c802f67138ff75908e45fb3b4923395afff4c.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/a3aca1963521b857ce32eab3711c802f67138ff75908e45fb3b4923395afff4c.jpg)

![afe3003d63f7d94cbb8f16ccee64f8ec2d168ff6e946d4f410608baa4a6c6531.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/afe3003d63f7d94cbb8f16ccee64f8ec2d168ff6e946d4f410608baa4a6c6531.jpg)

![d672394ce3798b590211f4941c53b024abc42a36d1cde81cbb75e93e2e6de27b.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/images/d672394ce3798b590211f4941c53b024abc42a36d1cde81cbb75e93e2e6de27b.jpg)

### Tables

![1bc2b89cf8cd47cdba0d75ee0d94a86e944affac3ad840a453b273ff9dc036de.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/1bc2b89cf8cd47cdba0d75ee0d94a86e944affac3ad840a453b273ff9dc036de.jpg)

![3e910c3a5bee7b3edbd7d10239bfe2ef194bb8a729e0843868394b854ce2edaa.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/3e910c3a5bee7b3edbd7d10239bfe2ef194bb8a729e0843868394b854ce2edaa.jpg)

![603b9d486561e1f900bf6af7547bd34bcdaac0b9ec565c6aa1f486b45a15e6da.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/603b9d486561e1f900bf6af7547bd34bcdaac0b9ec565c6aa1f486b45a15e6da.jpg)

![656d096e71c44594f7cc6236bc56a7a56670932029e72c33a7cf9ab19da5813e.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/656d096e71c44594f7cc6236bc56a7a56670932029e72c33a7cf9ab19da5813e.jpg)

![66aea26d370d97498c88e2409b33ab58aad79fa0964d476293bddab099f540be.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/66aea26d370d97498c88e2409b33ab58aad79fa0964d476293bddab099f540be.jpg)

![6fbed224d0da4278a4ca0b82de2e32210411766cabeb0188556348e40a930088.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/6fbed224d0da4278a4ca0b82de2e32210411766cabeb0188556348e40a930088.jpg)

![844645680f62ba2829d675cfff3a30393ec59d37119d2f0ecd69178bb7105e95.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/844645680f62ba2829d675cfff3a30393ec59d37119d2f0ecd69178bb7105e95.jpg)

![92707b739c055dbb6b0a3c646a37c3db914b8095cd35f862de2a1364a4f3aacc.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/92707b739c055dbb6b0a3c646a37c3db914b8095cd35f862de2a1364a4f3aacc.jpg)

![9e573a535e2672dcbf75becc248194abd8edd9c9a6b25f34fed13783249b0a73.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/9e573a535e2672dcbf75becc248194abd8edd9c9a6b25f34fed13783249b0a73.jpg)

![b1f72e966a69fd4980123bcad1d5201cda513ae49bdcfaa413e4e713ed0ba33e.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/b1f72e966a69fd4980123bcad1d5201cda513ae49bdcfaa413e4e713ed0ba33e.jpg)

![b3311b6760afad4f3349143aed1b7016ec26feccde3e00021689c10605892add.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/b3311b6760afad4f3349143aed1b7016ec26feccde3e00021689c10605892add.jpg)

![da95e8937d78624d68ee9edd358736d35f0df6da09631d6f19002081da01d0ba.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/da95e8937d78624d68ee9edd358736d35f0df6da09631d6f19002081da01d0ba.jpg)

![e2ee6b4da1e7fc87ae86d124313d20ec46f7e088cefa3971dba63ee99fa6cef1.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/e2ee6b4da1e7fc87ae86d124313d20ec46f7e088cefa3971dba63ee99fa6cef1.jpg)

![ece218c2974446419870016c0cc138f92f8f76ad9b928dab502c88904ae6a813.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/ece218c2974446419870016c0cc138f92f8f76ad9b928dab502c88904ae6a813.jpg)

![f3468370ced83b529164a5a9b793d54ef42dfc37d9951ead900346f15401af7a.jpg](../icml_results/5735_Self-Supervised%20Learning%20as%20Discrete%20Communication/tables/f3468370ced83b529164a5a9b793d54ef42dfc37d9951ead900346f15401af7a.jpg)

<a id="paper-5687"></a>

## Federated Learning with Unlabeled Clients: Personalization Can Happen in Low Dimensions


### Images

![93f4d6f691f98c6754c1e67ec49abf0213e1d015942832acb44fc7075a9377c0.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/images/93f4d6f691f98c6754c1e67ec49abf0213e1d015942832acb44fc7075a9377c0.jpg)

### Tables

![060d2f3309efbd0a6c9fb0d2dcbbfc67932a96fb172a8d5239ee237be1b53001.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/060d2f3309efbd0a6c9fb0d2dcbbfc67932a96fb172a8d5239ee237be1b53001.jpg)

![0dae55daa699f8e55b7c23e8e5b5cbbd8f372efcdcf25666fde43ed9cc18abee.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/0dae55daa699f8e55b7c23e8e5b5cbbd8f372efcdcf25666fde43ed9cc18abee.jpg)

![1aa3dc577e66f4a6d99c6254df4a5d8869b93a440c56c4ea62f352588df256bb.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/1aa3dc577e66f4a6d99c6254df4a5d8869b93a440c56c4ea62f352588df256bb.jpg)

![1c52857f550450d6cb1007943d5b1d182cbb3a30c29f0a06d8bd400cb48ddd14.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/1c52857f550450d6cb1007943d5b1d182cbb3a30c29f0a06d8bd400cb48ddd14.jpg)

![4ecda63feae54d3dbeb8820b3105b3d39e8c93a57abda4e463909b91360d5bc7.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/4ecda63feae54d3dbeb8820b3105b3d39e8c93a57abda4e463909b91360d5bc7.jpg)

![5862cea88b2a35e1d7384b93b621462d73589830e2b1a604d1790e0663954687.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/5862cea88b2a35e1d7384b93b621462d73589830e2b1a604d1790e0663954687.jpg)

![5cd54b5b0e40e762dc142b91b3ea2d75366b43ab0f0d3e274aec7b1595cdf068.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/5cd54b5b0e40e762dc142b91b3ea2d75366b43ab0f0d3e274aec7b1595cdf068.jpg)

![7a32e32546f5daf34150cb2e9c29cd3493e9ee29c312d1667b38cf0de3856e06.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/7a32e32546f5daf34150cb2e9c29cd3493e9ee29c312d1667b38cf0de3856e06.jpg)

![c26e5c812b2c965d660c6d0ee1b1b31f96376efd2c72fd01b09b86a0e70805d0.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/c26e5c812b2c965d660c6d0ee1b1b31f96376efd2c72fd01b09b86a0e70805d0.jpg)

![e96c57232b689269f4117084e32082ba65253393227f8f2e8787a95955add8c9.jpg](../icml_results/5736_Federated%20Learning%20with%20Unlabeled%20Clients_%20Personalization%20Can%20Happen%20in%20Low%20Dimensions/tables/e96c57232b689269f4117084e32082ba65253393227f8f2e8787a95955add8c9.jpg)

<a id="paper-5688"></a>

## The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning


### Images

![1b15e0e875fd5255e3a84c42d74e727ddfd2153c496c62fa8d6b058e2af0952e.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/images/1b15e0e875fd5255e3a84c42d74e727ddfd2153c496c62fa8d6b058e2af0952e.jpg)

![32dbf5496b37438398f659bb992cfd634b67e748479bbd38502d053e40c0f4cd.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/images/32dbf5496b37438398f659bb992cfd634b67e748479bbd38502d053e40c0f4cd.jpg)

![398b3538a10f0793a3970c263874c91b6eaa760b892fef2777c05323579ffd57.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/images/398b3538a10f0793a3970c263874c91b6eaa760b892fef2777c05323579ffd57.jpg)

![81070347762c627e9e8d31d482cf66ddeeefc87211ffbe54ad14cbb254332e37.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/images/81070347762c627e9e8d31d482cf66ddeeefc87211ffbe54ad14cbb254332e37.jpg)

![95bc81c823f6c1386999917c021ed9d5b811fa03c46d087d94642a4ab2eb3cd1.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/images/95bc81c823f6c1386999917c021ed9d5b811fa03c46d087d94642a4ab2eb3cd1.jpg)

### Tables

![0cc62b56bdfc9542c2caa6c1abf7e76cd96991829c8ccf873066b0582d8a0f16.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/0cc62b56bdfc9542c2caa6c1abf7e76cd96991829c8ccf873066b0582d8a0f16.jpg)

![2a98fd3e730a43e3e8583a340e4b5f7f06178fd5ed59b2c36a634fc3e4363483.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/2a98fd3e730a43e3e8583a340e4b5f7f06178fd5ed59b2c36a634fc3e4363483.jpg)

![6f90aae07b8304da82b6cabb62496c7591801881d07908aa8aeddc7764abe6aa.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/6f90aae07b8304da82b6cabb62496c7591801881d07908aa8aeddc7764abe6aa.jpg)

![752cc0977abe40d6ba23da4d946bb8e99996c7bfd681a3cdbbb95942a28d682e.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/752cc0977abe40d6ba23da4d946bb8e99996c7bfd681a3cdbbb95942a28d682e.jpg)

![7b89a7036a05a1688ae5f7f2ef9e4c3a76c5913a1370d5cb9a00cc13b3be1780.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/7b89a7036a05a1688ae5f7f2ef9e4c3a76c5913a1370d5cb9a00cc13b3be1780.jpg)

![8c0a0e58a52e0febc0a3b60d36faa3d4562d49f9cca463ca5c55d8eae110d75b.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/8c0a0e58a52e0febc0a3b60d36faa3d4562d49f9cca463ca5c55d8eae110d75b.jpg)

![abe31d7cbd5251cac7c3d335e2a63249f546c1abf37b7df049dbacaedc0a134a.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/abe31d7cbd5251cac7c3d335e2a63249f546c1abf37b7df049dbacaedc0a134a.jpg)

![af98b14b20483ece44a76e88d4f98b28530cf3c493f44eae1e4329f3c0503fbe.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/af98b14b20483ece44a76e88d4f98b28530cf3c493f44eae1e4329f3c0503fbe.jpg)

![c6980249a9cc187464f2d668c44a554d0fbef264da0ecd6168bfe739df844abe.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/c6980249a9cc187464f2d668c44a554d0fbef264da0ecd6168bfe739df844abe.jpg)

![cca74c0b3b09ab88956627a7783dd3b6ca712c820f4c55ddda473fb98fc3454b.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/cca74c0b3b09ab88956627a7783dd3b6ca712c820f4c55ddda473fb98fc3454b.jpg)

![dc251a2c82dae339928234912a572ed1cbfeefd635564b3fa51c93a4b56572e8.jpg](../icml_results/5737_The%20Loss%20Is%20Not%20Enough_%20Sampling%20Conditions%20and%20Inductive%20Bias%20in%20Contrastive%20Representation%20Learnin/tables/dc251a2c82dae339928234912a572ed1cbfeefd635564b3fa51c93a4b56572e8.jpg)

<a id="paper-5689"></a>

## HDTree: Generative Modeling of Cellular Hierarchies for Robust Lineage Inference


### Images

![5c49c3903b85af51a1f463ce61fed233666f73701e2b3ecc180814cf428851c7.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/5c49c3903b85af51a1f463ce61fed233666f73701e2b3ecc180814cf428851c7.jpg)

![72551bf82fa484e0b5d9e6fd8a698d521c3e19596544b940f1e9cfa07914e2f0.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/72551bf82fa484e0b5d9e6fd8a698d521c3e19596544b940f1e9cfa07914e2f0.jpg)

![730567a0adf9f491e0442be37e941de91989ffd38449639921b509c2770fa052.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/730567a0adf9f491e0442be37e941de91989ffd38449639921b509c2770fa052.jpg)

![8265e35d27e524772904c3b44c63f384d2fccedef24c9c4646a3175d38bfba9f.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/8265e35d27e524772904c3b44c63f384d2fccedef24c9c4646a3175d38bfba9f.jpg)

![ca5b7486ec7aafaf58daab1b6571c1c17ab0047e20cd8a8a10a9f294b5d5fbf6.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/ca5b7486ec7aafaf58daab1b6571c1c17ab0047e20cd8a8a10a9f294b5d5fbf6.jpg)

![cc0fc045a07d2666cc540091560ef911afdcf291d7d76da32e22690e14596ed1.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/cc0fc045a07d2666cc540091560ef911afdcf291d7d76da32e22690e14596ed1.jpg)

![d3250a5c4652b9b8214a4bd047772017844eca142ed50f758c3c07b3ee33687f.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/images/d3250a5c4652b9b8214a4bd047772017844eca142ed50f758c3c07b3ee33687f.jpg)

### Tables

![037677fa0fa7d1e262049885f25e7b3738977e9637ff17efca9b8dfcb2246bd9.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/037677fa0fa7d1e262049885f25e7b3738977e9637ff17efca9b8dfcb2246bd9.jpg)

![288814082b819018258fe6caadd362f6ae147e91b0094a43da3b52fd0a8a785a.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/288814082b819018258fe6caadd362f6ae147e91b0094a43da3b52fd0a8a785a.jpg)

![911fe0dee499fb3d4530a8cc75d7388f943e9da1f1907f890ea5f05b65c7a15b.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/911fe0dee499fb3d4530a8cc75d7388f943e9da1f1907f890ea5f05b65c7a15b.jpg)

![942f9635e08d46ef381decb9b243d59b8b2e7f1e35eb828f98e4b460615b7922.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/942f9635e08d46ef381decb9b243d59b8b2e7f1e35eb828f98e4b460615b7922.jpg)

![9bda83bcc5b4fa53495378ca6a965fcc477c6e40f658a1014d94750ef5ece60c.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/9bda83bcc5b4fa53495378ca6a965fcc477c6e40f658a1014d94750ef5ece60c.jpg)

![e381826ad917f9649975f5839fb217a582e1db52f2f13bca69a2c785d720bf38.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/e381826ad917f9649975f5839fb217a582e1db52f2f13bca69a2c785d720bf38.jpg)

![e7ebbb508719658f764cce9e8ccb357a23d80093cd55dbb82df2692bfc0da0e3.jpg](../icml_results/5738_HDTree_%20Generative%20Modeling%20of%20Cellular%20Hierarchies%20for%20Robust%20Lineage%20Inference/tables/e7ebbb508719658f764cce9e8ccb357a23d80093cd55dbb82df2692bfc0da0e3.jpg)

<a id="paper-5690"></a>

## Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios


### Images

![2db873920d7b7b84e4975c1401ec756429ed28bf63145c19d30bd6fc2f5053fc.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/2db873920d7b7b84e4975c1401ec756429ed28bf63145c19d30bd6fc2f5053fc.jpg)

![2e90f5719fb4e064074d04b12d50ad408b8e0589026fe6eaeaef239f8091fd54.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/2e90f5719fb4e064074d04b12d50ad408b8e0589026fe6eaeaef239f8091fd54.jpg)

![3ff99a027885b7c68cba5b067b803a91b6c3b5788d1f03d39f106cc39ab52b05.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/3ff99a027885b7c68cba5b067b803a91b6c3b5788d1f03d39f106cc39ab52b05.jpg)

![52d7cd950822592c7380b58d3b1256ad41bcebd981ea65e65363163728ffa226.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/52d7cd950822592c7380b58d3b1256ad41bcebd981ea65e65363163728ffa226.jpg)

![7c5175cfd9773826a26c5675a8ec88356117a93e35cb7fec97332399ce0decbd.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/7c5175cfd9773826a26c5675a8ec88356117a93e35cb7fec97332399ce0decbd.jpg)

![9ed2332a5570b60adb45493691f2957e5140765124c26d0485fd93dff2efc31b.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/9ed2332a5570b60adb45493691f2957e5140765124c26d0485fd93dff2efc31b.jpg)

![a30dc92d1157a5455e57d852fa77abab7ab3a2135d1022f63377c8d199fdf94c.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/a30dc92d1157a5455e57d852fa77abab7ab3a2135d1022f63377c8d199fdf94c.jpg)

![a39d6cdaf494f7776ca0e77b90484bd2a0c2320614906e2077104365754becb6.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/a39d6cdaf494f7776ca0e77b90484bd2a0c2320614906e2077104365754becb6.jpg)

![bb4e03639e2a506521d94d8d34a8d55a70a030a92f7167ecd8614da9a19b90a3.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/bb4e03639e2a506521d94d8d34a8d55a70a030a92f7167ecd8614da9a19b90a3.jpg)

![e061dda136cdaf2653f01cce297ab9bd29a864e52081fb1f56347000e05ae89c.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/e061dda136cdaf2653f01cce297ab9bd29a864e52081fb1f56347000e05ae89c.jpg)

![e8e106a263943c79906617f26a2623a9e53de807dc95be1171bc997db8f2f943.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/e8e106a263943c79906617f26a2623a9e53de807dc95be1171bc997db8f2f943.jpg)

![f2c681c2b449923069d0bb137fe02739dcf701110db711c80ac01a10bf5a295b.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/images/f2c681c2b449923069d0bb137fe02739dcf701110db711c80ac01a10bf5a295b.jpg)

### Tables

![10bf11220d4e7f80b5bdbaf74d9494b4104d5aef50ddddb8b9cffe5eaabc6860.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/10bf11220d4e7f80b5bdbaf74d9494b4104d5aef50ddddb8b9cffe5eaabc6860.jpg)

![2fafcc882cf445a8c98d4440e316dfb5aba0f37bc0e7b4daf52d23c9e3abc645.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/2fafcc882cf445a8c98d4440e316dfb5aba0f37bc0e7b4daf52d23c9e3abc645.jpg)

![40ef5154760fb1463d92c9414739db3830925a0d5577386f6c54d5ed38a2a708.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/40ef5154760fb1463d92c9414739db3830925a0d5577386f6c54d5ed38a2a708.jpg)

![427937bccbe05e557daacfbcff7490ad4f110c92699c3515e3e44c3fb6712c8a.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/427937bccbe05e557daacfbcff7490ad4f110c92699c3515e3e44c3fb6712c8a.jpg)

![445874791c11fe943b4653fe1a69802c3c7827bfba759491779331e82c89aab3.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/445874791c11fe943b4653fe1a69802c3c7827bfba759491779331e82c89aab3.jpg)

![5eb511867e249aee74dce0a9d910e940cda59fe63c858d8b4ee38fb222cca8e8.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/5eb511867e249aee74dce0a9d910e940cda59fe63c858d8b4ee38fb222cca8e8.jpg)

![732968dc02c583d6a620d0d04cfd94a0e09e7d614ac889bc0b4a076287741b25.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/732968dc02c583d6a620d0d04cfd94a0e09e7d614ac889bc0b4a076287741b25.jpg)

![79e9b13ab604ad051f1e3dfb7ee210ba8c24291a9d7e93d9f90fd4898df79369.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/79e9b13ab604ad051f1e3dfb7ee210ba8c24291a9d7e93d9f90fd4898df79369.jpg)

![7eb63589896bedf8e0ae15d8820c10a52c010003c7d4f884dc050fba29015137.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/7eb63589896bedf8e0ae15d8820c10a52c010003c7d4f884dc050fba29015137.jpg)

![81791bb470da90c4adf738dc3f61dc625e0bcb32a970ac73d08c6f8807495158.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/81791bb470da90c4adf738dc3f61dc625e0bcb32a970ac73d08c6f8807495158.jpg)

![98708d0e60637c5d1fdf21ae5d8dcc1f78e50a5b275a7d419a6d93f8cea1f0f7.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/98708d0e60637c5d1fdf21ae5d8dcc1f78e50a5b275a7d419a6d93f8cea1f0f7.jpg)

![ab8e0484ede6163652497ae8d312a01e1511bb6477f265d0a7ed402a1df7beaf.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/ab8e0484ede6163652497ae8d312a01e1511bb6477f265d0a7ed402a1df7beaf.jpg)

![b860df9909477bad5c38641814bbe6c92d454c0829c83fd0dda837e7db7dcc60.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/b860df9909477bad5c38641814bbe6c92d454c0829c83fd0dda837e7db7dcc60.jpg)

![e79ce4672988b66add3eeccc4ba2bbb490f5a9c70cfe113f47cb19bbab82b3a4.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/e79ce4672988b66add3eeccc4ba2bbb490f5a9c70cfe113f47cb19bbab82b3a4.jpg)

![eb57475239d852c861c7063844f217729af61b3de471817836cb435d0bf70a20.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/eb57475239d852c861c7063844f217729af61b3de471817836cb435d0bf70a20.jpg)

![f9181d34d097e195fab90dabab821323f0184d1a04eae55cdf921a53febfe89e.jpg](../icml_results/5739_Reward%20Auditor_%20Inference%20on%20Reward%20Modeling%20Suitability%20in%20Real-World%20Perturbed%20Scenarios/tables/f9181d34d097e195fab90dabab821323f0184d1a04eae55cdf921a53febfe89e.jpg)

<a id="paper-5691"></a>

## HIVE-3D: Hierarchical Voxel Enhancement for High-Quality 3D Scene Generation


### Images

![12a61d471621eab0d49546763d7aaa832be6abfecea9e1a0987aeef1b92aab61.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/12a61d471621eab0d49546763d7aaa832be6abfecea9e1a0987aeef1b92aab61.jpg)

![27dca8ba0a34e079fff5e2ea096f910024e8e6c9cabe1636a2fa758d532f772e.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/27dca8ba0a34e079fff5e2ea096f910024e8e6c9cabe1636a2fa758d532f772e.jpg)

![3a77fed0e3eeecabd0d30d1b6bada4babf2c14f6348009b070cdd9124528aed6.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/3a77fed0e3eeecabd0d30d1b6bada4babf2c14f6348009b070cdd9124528aed6.jpg)

![4d9ece688ac03c669171b20030bc049ba63b7e055b0507f22202819d89eb303d.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/4d9ece688ac03c669171b20030bc049ba63b7e055b0507f22202819d89eb303d.jpg)

![6624c571c5f2e900570e716ab9b854ee92d9cc46e34a91d39281483a00ea9f35.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/6624c571c5f2e900570e716ab9b854ee92d9cc46e34a91d39281483a00ea9f35.jpg)

![864ffe4dc709325a86f29d6b03c88b5427459db694dd1fff54851027c41138e9.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/864ffe4dc709325a86f29d6b03c88b5427459db694dd1fff54851027c41138e9.jpg)

![88e6610760d1a1ab4eb9063899670465a7699fa33d5528c1bf1c455aa771a464.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/88e6610760d1a1ab4eb9063899670465a7699fa33d5528c1bf1c455aa771a464.jpg)

![9cf3943c7f7dffd1163b708758edfaed33cb147d6bfd0b0150edfa727533990d.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/9cf3943c7f7dffd1163b708758edfaed33cb147d6bfd0b0150edfa727533990d.jpg)

![9f3bdf7a680ac145bf35a7ebfba872697e329d308da9dd627e79c39ea7a27823.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/9f3bdf7a680ac145bf35a7ebfba872697e329d308da9dd627e79c39ea7a27823.jpg)

![c63abe434d3ede3035283c3f2ac3c6f04a24a488df96a221d2c5852287dd4085.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/c63abe434d3ede3035283c3f2ac3c6f04a24a488df96a221d2c5852287dd4085.jpg)

![e153555fe29bb010b17a952998bab1e4cdec8cfe187165af3e74d8ff572a2173.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/e153555fe29bb010b17a952998bab1e4cdec8cfe187165af3e74d8ff572a2173.jpg)

![f6eaa06e5ab3256c86df33111ebf66c89b286c4dbd6cbcde786d927b01cc21e4.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/images/f6eaa06e5ab3256c86df33111ebf66c89b286c4dbd6cbcde786d927b01cc21e4.jpg)

### Tables

![15eb90483a141510cb68cc6e8c7e293366356c947061229f421d484f1e29aafe.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/15eb90483a141510cb68cc6e8c7e293366356c947061229f421d484f1e29aafe.jpg)

![38d66d53c148e233df6a9154ce67dff30c07b9c999d397e6502642eafe83b12f.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/38d66d53c148e233df6a9154ce67dff30c07b9c999d397e6502642eafe83b12f.jpg)

![4741b463ac69313416931e4e000361ac6a3d817354e1b929d54298539762f78f.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/4741b463ac69313416931e4e000361ac6a3d817354e1b929d54298539762f78f.jpg)

![9357f6f9d37b47d768dc8d1fe704424e1bd25d2b2c9f00a03685527a0fc33cd4.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/9357f6f9d37b47d768dc8d1fe704424e1bd25d2b2c9f00a03685527a0fc33cd4.jpg)

![9f67879be4750271c41763c14b7d42cb764554a4a985722b6c7ce3f1da84550d.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/9f67879be4750271c41763c14b7d42cb764554a4a985722b6c7ce3f1da84550d.jpg)

![a3d93ddf6fadf25970b5e9ba732e6db24e51a678056d8e058eb9d61de1744d5f.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/a3d93ddf6fadf25970b5e9ba732e6db24e51a678056d8e058eb9d61de1744d5f.jpg)

![d7b84bfc546f457b8244954f55e806d605071d7a94753ac7d72d0a1c98c9ed03.jpg](../icml_results/5740_HIVE-3D_%20Hierarchical%20Voxel%20Enhancement%20for%20High-Quality%203D%20Scene%20Generation/tables/d7b84bfc546f457b8244954f55e806d605071d7a94753ac7d72d0a1c98c9ed03.jpg)

<a id="paper-5692"></a>

## CINOC: Cardinality-Invariant Neural Operator Policies for Scalable PDE Control


### Images

![16e8146208514bdcf3d1b7656138169cc990156026e18b50de1f0db54db98a33.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/16e8146208514bdcf3d1b7656138169cc990156026e18b50de1f0db54db98a33.jpg)

![1f8dd5bfecfb321f6edd9697a6a7b8cb357a35dfcb3cb87b7be6fe9ab24fcb00.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/1f8dd5bfecfb321f6edd9697a6a7b8cb357a35dfcb3cb87b7be6fe9ab24fcb00.jpg)

![2a4e0f807a2d72f568fb0a0614ee53b0bcd50a7e46fad87bd9d5e2ec7e28977f.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/2a4e0f807a2d72f568fb0a0614ee53b0bcd50a7e46fad87bd9d5e2ec7e28977f.jpg)

![33b69eab2291b15b8edae4217d6e53bd9e63babef90f972333a4988eca79ab1a.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/33b69eab2291b15b8edae4217d6e53bd9e63babef90f972333a4988eca79ab1a.jpg)

![40bc416d4609b63ab411e3a11537326dcd9e8578d388b565279d75fb942a6ea7.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/40bc416d4609b63ab411e3a11537326dcd9e8578d388b565279d75fb942a6ea7.jpg)

![4354843cdb6518d75a863537c1ddb55aa005b9cdcd61a241d151bd6a7a58785b.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/4354843cdb6518d75a863537c1ddb55aa005b9cdcd61a241d151bd6a7a58785b.jpg)

![4b42d0e370a5a9309553fd2bc76b0dca0c5b52ec2e8f2d94d37f4f5dcfdc4be8.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/4b42d0e370a5a9309553fd2bc76b0dca0c5b52ec2e8f2d94d37f4f5dcfdc4be8.jpg)

![531d31a3fdfd8d86cadb0d06066a4f957201b7446589c12be4777d1ed67b5314.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/531d31a3fdfd8d86cadb0d06066a4f957201b7446589c12be4777d1ed67b5314.jpg)

![53af14e66946d556826d30863e08c4efe2496b13418d42a9e5fc8b49dd75a849.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/53af14e66946d556826d30863e08c4efe2496b13418d42a9e5fc8b49dd75a849.jpg)

![59f26ac7cebb6bdeeb25107adba4805445ca2e436d76f4b66af0648bbd239b09.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/59f26ac7cebb6bdeeb25107adba4805445ca2e436d76f4b66af0648bbd239b09.jpg)

![807b6608ce5b214aff3c832a536dd67e56258f099332c23ef596997b3ce08010.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/807b6608ce5b214aff3c832a536dd67e56258f099332c23ef596997b3ce08010.jpg)

![8aa8bd316b73e08ea2f6ae15194eec6d92b3a94d120663791bd16b65fd9f444c.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/8aa8bd316b73e08ea2f6ae15194eec6d92b3a94d120663791bd16b65fd9f444c.jpg)

![9ec61521c10abcd73385f423c0b44aaa591ff4becd8c3f39a5b169e8520df778.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/9ec61521c10abcd73385f423c0b44aaa591ff4becd8c3f39a5b169e8520df778.jpg)

![a0ed7b35411063954a6f0f940189a0e096881d0c51b2538239b83d6f5b3bf020.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/a0ed7b35411063954a6f0f940189a0e096881d0c51b2538239b83d6f5b3bf020.jpg)

![b2ae2e521837ef749a5e9ae1d67585a52d2b9fc2eeb195a887cdd81987cea624.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/b2ae2e521837ef749a5e9ae1d67585a52d2b9fc2eeb195a887cdd81987cea624.jpg)

![c33035c83d5367751744202a73272d6d5067435914dd5752bb5f647f7e47d17f.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/c33035c83d5367751744202a73272d6d5067435914dd5752bb5f647f7e47d17f.jpg)

![d1a681ec66bd52618a3116955423fa6db29ce1fd640859c104a5d6b45c323864.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/d1a681ec66bd52618a3116955423fa6db29ce1fd640859c104a5d6b45c323864.jpg)

![e08b9ff80b83f6eb0c0ad09b374a9dc6988dd8fac22ef1a488ad601f51b4f219.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/e08b9ff80b83f6eb0c0ad09b374a9dc6988dd8fac22ef1a488ad601f51b4f219.jpg)

![ef4c8ef25679455acf3a1c46296924e61bd9da2c0182eb59038bcf3aa58eb3de.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/ef4c8ef25679455acf3a1c46296924e61bd9da2c0182eb59038bcf3aa58eb3de.jpg)

![f052e53a4e679e668619a1878b199da577341cf18b436c028f8d7ac91c338ec7.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/images/f052e53a4e679e668619a1878b199da577341cf18b436c028f8d7ac91c338ec7.jpg)

### Tables

![0cbeccab3fb6cfbade42cf9a6f5968d142829942f8941ade54950a00a1109999.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/0cbeccab3fb6cfbade42cf9a6f5968d142829942f8941ade54950a00a1109999.jpg)

![12e5667c9198cb0984f2b8dd29a8a2053239303df174368cf0ce4ac985448dfe.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/12e5667c9198cb0984f2b8dd29a8a2053239303df174368cf0ce4ac985448dfe.jpg)

![1a0ed7a796dd815c2898b1c9292d61788cde45d94b60a83da37f51b3f23ce936.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/1a0ed7a796dd815c2898b1c9292d61788cde45d94b60a83da37f51b3f23ce936.jpg)

![26c0a228bd17bd25a8de54121df3c4c4b0249936b9c1145cfc0c806a6944e42f.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/26c0a228bd17bd25a8de54121df3c4c4b0249936b9c1145cfc0c806a6944e42f.jpg)

![35e3323c178298898282e1a431336bbf245c0fb1f0e98a635621bb08aa2022b8.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/35e3323c178298898282e1a431336bbf245c0fb1f0e98a635621bb08aa2022b8.jpg)

![35eee1703f90cda1a8e21793ff9afae18a20d742d1cde427fa861b2b15962c32.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/35eee1703f90cda1a8e21793ff9afae18a20d742d1cde427fa861b2b15962c32.jpg)

![39d820ac3a87639ab6b0b292c353e068e35c87d2332d5f194ead494032d620fa.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/39d820ac3a87639ab6b0b292c353e068e35c87d2332d5f194ead494032d620fa.jpg)

![3d56ea02d39708fc4f9dcb86583b233873386a90936741758af446b64c9e5112.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/3d56ea02d39708fc4f9dcb86583b233873386a90936741758af446b64c9e5112.jpg)

![43102db726267e306271a16bdda2adb1e36402118156c54d610fbc8be1a8272a.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/43102db726267e306271a16bdda2adb1e36402118156c54d610fbc8be1a8272a.jpg)

![45c7471b644134be8169f4aa60414c1e4914a08f2bf3bbe350e728dc4963b7b4.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/45c7471b644134be8169f4aa60414c1e4914a08f2bf3bbe350e728dc4963b7b4.jpg)

![5e988df50cb0fd17661fdae9a992498d05b731a8dfe84011b7dfbf0c25c3b487.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/5e988df50cb0fd17661fdae9a992498d05b731a8dfe84011b7dfbf0c25c3b487.jpg)

![648949daeed37adc093dcbfbfad586e065f1919059a702069abdcada3b558430.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/648949daeed37adc093dcbfbfad586e065f1919059a702069abdcada3b558430.jpg)

![6d070a9156565c961a4ce30ff5f5b8244008d0d52c7ebf6ae15ab9f89f88f0c4.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/6d070a9156565c961a4ce30ff5f5b8244008d0d52c7ebf6ae15ab9f89f88f0c4.jpg)

![73d51d0f5a77f5ca94fe456982a0803129b168fa9594418c62723b9575e100e1.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/73d51d0f5a77f5ca94fe456982a0803129b168fa9594418c62723b9575e100e1.jpg)

![795f47c72d0f1f0ad378a0b859b225c43f0451c35d3f3f3c66faac73e916c852.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/795f47c72d0f1f0ad378a0b859b225c43f0451c35d3f3f3c66faac73e916c852.jpg)

![7a463d9ce366b55e6907d25207b2eb252d225b20726c8fbe958e2683e6182da1.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/7a463d9ce366b55e6907d25207b2eb252d225b20726c8fbe958e2683e6182da1.jpg)

![87be26b6211ea8c5cb42b8a8aefa9cbc9fc9d99be54bc00d191d917fc04bee73.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/87be26b6211ea8c5cb42b8a8aefa9cbc9fc9d99be54bc00d191d917fc04bee73.jpg)

![8953ae4d7cc999b3df8cd89adc09144ea6d6303dc0d272d12b38f116d7e3eeb6.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/8953ae4d7cc999b3df8cd89adc09144ea6d6303dc0d272d12b38f116d7e3eeb6.jpg)

![b586328ba4e2f8a49d716bf255db0dca6cede689fef44018b2f2e4829eee8b22.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/b586328ba4e2f8a49d716bf255db0dca6cede689fef44018b2f2e4829eee8b22.jpg)

![c86f32902f01aaecc9b78c095ce551700d8e600394e72a5e1001d503a546f063.jpg](../icml_results/5741_CINOC_%20Cardinality-Invariant%20Neural%20Operator%20Policies%20for%20Scalable%20PDE%20Control/tables/c86f32902f01aaecc9b78c095ce551700d8e600394e72a5e1001d503a546f063.jpg)

<a id="paper-5693"></a>

## Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation


### Images

![2b872ad7cf1179c7f0977a4fe5715f2150242712700991f29a1a5e239ff553ef.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/2b872ad7cf1179c7f0977a4fe5715f2150242712700991f29a1a5e239ff553ef.jpg)

![52b73a5a416ddc26ec19e6fb0608684434902457cfe21c1dadbf6c1878d45d75.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/52b73a5a416ddc26ec19e6fb0608684434902457cfe21c1dadbf6c1878d45d75.jpg)

![6372d35ba04658a27771b53e2f15878b6cd219857b23fa593ee682cfd883eb33.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/6372d35ba04658a27771b53e2f15878b6cd219857b23fa593ee682cfd883eb33.jpg)

![74f0092b7a7844fdb2dcafcf184283ceb932d39528348c28722bcb62ed1ef102.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/74f0092b7a7844fdb2dcafcf184283ceb932d39528348c28722bcb62ed1ef102.jpg)

![9b417fe5f7c76572447ab48c1c3f3364f6ed66090b3dc1c6cfa8b6559daf0f24.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/9b417fe5f7c76572447ab48c1c3f3364f6ed66090b3dc1c6cfa8b6559daf0f24.jpg)

![b5a3d34158ff9c14bc4c7b353187185b65e3b45487c702d89d298d20d8b9dd13.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/b5a3d34158ff9c14bc4c7b353187185b65e3b45487c702d89d298d20d8b9dd13.jpg)

![ea6ea0804c8bec16c8387c6d495e33c245c371fe994c88b2ae1915d263fb23d6.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/images/ea6ea0804c8bec16c8387c6d495e33c245c371fe994c88b2ae1915d263fb23d6.jpg)

### Tables

![203ff55260b80b4520146d3b9256b5dbad0037d2cdc43728cf5460ce0d2c196b.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/tables/203ff55260b80b4520146d3b9256b5dbad0037d2cdc43728cf5460ce0d2c196b.jpg)

![36ac17b4318228ad77dfed21a6078c48cfa9c8a9389d27dca36f1270390178f3.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/tables/36ac17b4318228ad77dfed21a6078c48cfa9c8a9389d27dca36f1270390178f3.jpg)

![44f567204b1078650551c55938ebd847c49e889165313a62347312e3bed193c8.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/tables/44f567204b1078650551c55938ebd847c49e889165313a62347312e3bed193c8.jpg)

![e18e8d5affb7f1ed5a8c71a3cfe9139655892fcea35b273a51af5bf9b41319c4.jpg](../icml_results/5742_Interpretable%20Self-Supervised%20Learning%20via%20Representer%20Landmarks%20and%20Nystr%C3%B6m%20Approximation/tables/e18e8d5affb7f1ed5a8c71a3cfe9139655892fcea35b273a51af5bf9b41319c4.jpg)

<a id="paper-5694"></a>

## SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs


### Images

![17b4851f2c20ec8bc9fbb4da691b35502a42170b2031ede7e56a5e20d080d9b0.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/17b4851f2c20ec8bc9fbb4da691b35502a42170b2031ede7e56a5e20d080d9b0.jpg)

![1c7f7a626300e5406f1d6d14b1976b4298ffb8587d170d0a10601faec098c3f2.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/1c7f7a626300e5406f1d6d14b1976b4298ffb8587d170d0a10601faec098c3f2.jpg)

![23969f6a29403ed935253361918e52e523c3d3ce2913a825d7e7a9424c81923b.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/23969f6a29403ed935253361918e52e523c3d3ce2913a825d7e7a9424c81923b.jpg)

![4f9527bfc45491923ba8afd8ee19a0845c8a5b428c3d6596b1e1b86012fa735f.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/4f9527bfc45491923ba8afd8ee19a0845c8a5b428c3d6596b1e1b86012fa735f.jpg)

![aa9e4a092816a9851a513c1269ee365c2578b4b2a13015e50285e0b1dcbabb7b.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/aa9e4a092816a9851a513c1269ee365c2578b4b2a13015e50285e0b1dcbabb7b.jpg)

![c09e7bd39c3b3a76d6c639c5d477772085c1a00027aaacf116b09c7c69733863.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/c09e7bd39c3b3a76d6c639c5d477772085c1a00027aaacf116b09c7c69733863.jpg)

![c2aaa70dc0eb01f29685499b134395a1a1ef4a91c7f4ac1fce07f6019c354dc1.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/c2aaa70dc0eb01f29685499b134395a1a1ef4a91c7f4ac1fce07f6019c354dc1.jpg)

![e77443c4568a1793f1ac2aee1b2e141a5d567dfdd305950a713d61f615b0f7d2.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/images/e77443c4568a1793f1ac2aee1b2e141a5d567dfdd305950a713d61f615b0f7d2.jpg)

### Tables

![0e53a7f40baaf81936d6aa1aa043db2efaa586177f3c982e3eee1d7239412bca.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/0e53a7f40baaf81936d6aa1aa043db2efaa586177f3c982e3eee1d7239412bca.jpg)

![1b9b451f94ed5c81140f3b8b533fe965eea7ef444d320a23d1aae8eb545c6a61.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/1b9b451f94ed5c81140f3b8b533fe965eea7ef444d320a23d1aae8eb545c6a61.jpg)

![209d34c96cc3f5445ee772e6dcacd038d4d0e7b0eef905f3838336312b8750b7.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/209d34c96cc3f5445ee772e6dcacd038d4d0e7b0eef905f3838336312b8750b7.jpg)

![92843ceccf720b5b0207cc094c5131b748b607fa99f2a3b111b38ad5f19265b7.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/92843ceccf720b5b0207cc094c5131b748b607fa99f2a3b111b38ad5f19265b7.jpg)

![a1a8eb429d37b9ffd6929be7392d9b7d2de0a15b8109da8c0b43f87126d9b1bf.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/a1a8eb429d37b9ffd6929be7392d9b7d2de0a15b8109da8c0b43f87126d9b1bf.jpg)

![b65fbd403d13791e26cb10f7be7371fd8b2258d663b18cb27228473396e27bd9.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/b65fbd403d13791e26cb10f7be7371fd8b2258d663b18cb27228473396e27bd9.jpg)

![cfc9b2db9e89663a14d1494577f90490add057ba6af82a41596568a305d0a35d.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/cfc9b2db9e89663a14d1494577f90490add057ba6af82a41596568a305d0a35d.jpg)

![e7ae037f595a8b82ec46e88aca3ea429d626bce2292ace2f4c104593a0221032.jpg](../icml_results/5743_SoftMoE_%20Soft%20Differentiable%20Routing%20for%20Mixture-of-Experts%20in%20LLMs/tables/e7ae037f595a8b82ec46e88aca3ea429d626bce2292ace2f4c104593a0221032.jpg)

<a id="paper-5695"></a>

## MentisOculi: Revealing the Limits of Reasoning with Mental Imagery


### Images

![081014a012dec2b4f1c387256bb084806216924cfe9aee304e7ca4466fc5e501.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/081014a012dec2b4f1c387256bb084806216924cfe9aee304e7ca4466fc5e501.jpg)

![0f756e7fb2e423426e057c006cd1b9249fa91bb95933b8582eddaee5aa616f91.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/0f756e7fb2e423426e057c006cd1b9249fa91bb95933b8582eddaee5aa616f91.jpg)

![1e96fe69b804ba4596929fdf290df22a94f62a21bc98cc15937c6c9ecb2591ea.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/1e96fe69b804ba4596929fdf290df22a94f62a21bc98cc15937c6c9ecb2591ea.jpg)

![1eac811d01c2d81bd57118c4d08cca02bd62c9d591ffede6bc4f2e3d62449e5f.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/1eac811d01c2d81bd57118c4d08cca02bd62c9d591ffede6bc4f2e3d62449e5f.jpg)

![1feb3c174ba9dcfa1ae14aa1ac51ddaff23979633861a898cdf8fc2b2460bbce.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/1feb3c174ba9dcfa1ae14aa1ac51ddaff23979633861a898cdf8fc2b2460bbce.jpg)

![20f295576bc6b20b31fade9f1d234cfb9a181ec896e7c915840facc8a0e59eb5.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/20f295576bc6b20b31fade9f1d234cfb9a181ec896e7c915840facc8a0e59eb5.jpg)

![2650186e37d430daa1320520284e91d1c245256c5a9fd40f33e3eee20892a559.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/2650186e37d430daa1320520284e91d1c245256c5a9fd40f33e3eee20892a559.jpg)

![30989167e5da4a4c4672b90e1ef5bb525090c9087fb10563ad9ce4d46b2679f9.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/30989167e5da4a4c4672b90e1ef5bb525090c9087fb10563ad9ce4d46b2679f9.jpg)

![3bd0ea83cb7b88df49a40e7268146c57f34f5f941290f061a4c9d485faf9d3b9.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/3bd0ea83cb7b88df49a40e7268146c57f34f5f941290f061a4c9d485faf9d3b9.jpg)

![47522342051607524336be17b521c21164eb52bb84842f4dda2611311c6c6b93.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/47522342051607524336be17b521c21164eb52bb84842f4dda2611311c6c6b93.jpg)

![542a795cb47f5e16689a95218a6ca41b0a7205aedec1e8c22d1216d6de51fbb3.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/542a795cb47f5e16689a95218a6ca41b0a7205aedec1e8c22d1216d6de51fbb3.jpg)

![5ace955c6db61e0b2d759217e45d76ec4ce765f15d1004db9df2719dfababf63.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/5ace955c6db61e0b2d759217e45d76ec4ce765f15d1004db9df2719dfababf63.jpg)

![5b565b7e7b841939d38922a542c70db844410e64b7020f4fa413f4fea5e568c7.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/5b565b7e7b841939d38922a542c70db844410e64b7020f4fa413f4fea5e568c7.jpg)

![61699537c2a735c6ef2c41535d53e5dff617b09e089a055597f09b956ea27f1b.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/61699537c2a735c6ef2c41535d53e5dff617b09e089a055597f09b956ea27f1b.jpg)

![6429d1cb7aabaf4eb68201ffdbbc9c7be7e449403d1b4e52b1fd089dad6a6c7e.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/6429d1cb7aabaf4eb68201ffdbbc9c7be7e449403d1b4e52b1fd089dad6a6c7e.jpg)

![779773be9bf1ead3e3190594c2fe16253802caa92e213d1199961ff62ab6050b.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/779773be9bf1ead3e3190594c2fe16253802caa92e213d1199961ff62ab6050b.jpg)

![865ff5636294b9edaf7a5dc80c27a2df8beed4a1c03e763dba1d5efd39070433.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/865ff5636294b9edaf7a5dc80c27a2df8beed4a1c03e763dba1d5efd39070433.jpg)

![a2948b8914fd11c9131d3168c445139dd17dae381564dd7617029c657f1049c1.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/a2948b8914fd11c9131d3168c445139dd17dae381564dd7617029c657f1049c1.jpg)

![aa55935396a9b077651a3d0045e0b5f593ea3e0957e5edad7a8efe519b00337e.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/aa55935396a9b077651a3d0045e0b5f593ea3e0957e5edad7a8efe519b00337e.jpg)

![b2bb43a3498b8b14e0f947da1dfc78fe224ed604eea44b81fd855342299849eb.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/b2bb43a3498b8b14e0f947da1dfc78fe224ed604eea44b81fd855342299849eb.jpg)

![b4a3a0744083ec9f96ce85c4cd5b431910177aac9bc2226fa975b60840fce666.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/b4a3a0744083ec9f96ce85c4cd5b431910177aac9bc2226fa975b60840fce666.jpg)

![bdd63cdf15f7c9de23df34b471f3b792ad7f1bb1f0f5b466e4571a3852984e7f.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/bdd63cdf15f7c9de23df34b471f3b792ad7f1bb1f0f5b466e4571a3852984e7f.jpg)

![c06582c2741564e0043130eaabfca16d53ce070982d10514d445dd2c5c2d1847.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/c06582c2741564e0043130eaabfca16d53ce070982d10514d445dd2c5c2d1847.jpg)

![c44d90437e1206267a0186922a4e53b5aa49ae4058e65c78c2538539e7520a0d.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/c44d90437e1206267a0186922a4e53b5aa49ae4058e65c78c2538539e7520a0d.jpg)

![c561100d14ff5f94203d3d87f9f3b496498d6493a70a0c1aaf6ca28358d9bc93.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/c561100d14ff5f94203d3d87f9f3b496498d6493a70a0c1aaf6ca28358d9bc93.jpg)

![cfcf2d248690f18a2ac481298e10a9396e87774e7cc0189e5ee496cb2cabe88e.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/cfcf2d248690f18a2ac481298e10a9396e87774e7cc0189e5ee496cb2cabe88e.jpg)

![e9ab237decb352159c42c8b27d2c4a17276c10a1e1428a30849d064da843c503.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/e9ab237decb352159c42c8b27d2c4a17276c10a1e1428a30849d064da843c503.jpg)

![eb9881497f8a686359d5c4849ebce7fd0488a4d1b650d93cdd815a77639a6cb1.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/eb9881497f8a686359d5c4849ebce7fd0488a4d1b650d93cdd815a77639a6cb1.jpg)

![efbdf761e548a22feae4b90cecb1ba55925dcd40d64c54d44d9ef673a09905d2.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/efbdf761e548a22feae4b90cecb1ba55925dcd40d64c54d44d9ef673a09905d2.jpg)

![f1ce12fd4347dc1fafb0a21a911c0697f14a45bb9e8fd661c1c95f676af0e968.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/f1ce12fd4347dc1fafb0a21a911c0697f14a45bb9e8fd661c1c95f676af0e968.jpg)

![f34c1962b66dc314f13c4ad564b6c2b4a9a119868dbcf5da01b435b89fe5e83f.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/f34c1962b66dc314f13c4ad564b6c2b4a9a119868dbcf5da01b435b89fe5e83f.jpg)

![fb7cd5003552a34dc8a334e8e142c9723e00b2d7970dc893073419f635788af6.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/fb7cd5003552a34dc8a334e8e142c9723e00b2d7970dc893073419f635788af6.jpg)

![fd87ab0e0cccb481d28b8309ee5f3ad3cab13b305a87590b17f6f1c4eba9d2af.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/images/fd87ab0e0cccb481d28b8309ee5f3ad3cab13b305a87590b17f6f1c4eba9d2af.jpg)

### Tables

![65efaab679a752252ee86a3f7c7ff1dd1eb0c40b32640c2e0064e0c743ff95ca.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/tables/65efaab679a752252ee86a3f7c7ff1dd1eb0c40b32640c2e0064e0c743ff95ca.jpg)

![fdaf50c2186bfc733f40850eded23b73416a449edeae7e0a2896ebeb96b4b5cd.jpg](../icml_results/5744_MentisOculi_%20Revealing%20the%20Limits%20of%20Reasoning%20with%20Mental%20Imagery/tables/fdaf50c2186bfc733f40850eded23b73416a449edeae7e0a2896ebeb96b4b5cd.jpg)

<a id="paper-5696"></a>

## Rethinking Neural Network Learning Rates: A Stackelberg Perspective


### Images

![0f57938ba8a43d3945ef9603f685b827883e377b2d83ff9d1c65f528d6ae955f.jpg](../icml_results/5745_Rethinking%20Neural%20Network%20Learning%20Rates_%20A%20Stackelberg%20Perspective/images/0f57938ba8a43d3945ef9603f685b827883e377b2d83ff9d1c65f528d6ae955f.jpg)

![585d3e2271ab6c956c962de51e3d07e2841d4b69eed0ede5ec7752a3d2893489.jpg](../icml_results/5745_Rethinking%20Neural%20Network%20Learning%20Rates_%20A%20Stackelberg%20Perspective/images/585d3e2271ab6c956c962de51e3d07e2841d4b69eed0ede5ec7752a3d2893489.jpg)

![5bbf2522bee04571744402df1fde60489b273da614f66904f18620ea3320e01b.jpg](../icml_results/5745_Rethinking%20Neural%20Network%20Learning%20Rates_%20A%20Stackelberg%20Perspective/images/5bbf2522bee04571744402df1fde60489b273da614f66904f18620ea3320e01b.jpg)

![de2c2c764f1f9ea84bbdc5f221e6a8495c6ebeb0bc41c60ce198b547bab88abe.jpg](../icml_results/5745_Rethinking%20Neural%20Network%20Learning%20Rates_%20A%20Stackelberg%20Perspective/images/de2c2c764f1f9ea84bbdc5f221e6a8495c6ebeb0bc41c60ce198b547bab88abe.jpg)

![e945cd1a65304dc83e7dd943e95fe2068d3cd9b5c760fb7e73a7c5dc66c0815f.jpg](../icml_results/5745_Rethinking%20Neural%20Network%20Learning%20Rates_%20A%20Stackelberg%20Perspective/images/e945cd1a65304dc83e7dd943e95fe2068d3cd9b5c760fb7e73a7c5dc66c0815f.jpg)

<a id="paper-5697"></a>

## POLIA: Policy Optimization with Visual-Object-Level Intrinsic Advantage for Multimodal Reasoning


### Images

![018fed82d88a4176a3e3c89660406ac78aa56800e51d8fd33b17f7ad3b73249f.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/018fed82d88a4176a3e3c89660406ac78aa56800e51d8fd33b17f7ad3b73249f.jpg)

![199b04f6847ba0fb4eded504842598200dad64e0123ab9867716b62bf88e1c1f.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/199b04f6847ba0fb4eded504842598200dad64e0123ab9867716b62bf88e1c1f.jpg)

![244dc9bfae84f3bfdcf6b2bbf8aedb58b880d3ce8a5ba59436bbb716482d14af.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/244dc9bfae84f3bfdcf6b2bbf8aedb58b880d3ce8a5ba59436bbb716482d14af.jpg)

![299a5a1e4dcf7a6151f45b599bf0f2bb15231ddd4ae3422d920e8875568ec8c6.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/299a5a1e4dcf7a6151f45b599bf0f2bb15231ddd4ae3422d920e8875568ec8c6.jpg)

![37cfa987d62d2b3a9904fd260fd519dafa281a91631d1210fc9dfbd8d5c18033.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/37cfa987d62d2b3a9904fd260fd519dafa281a91631d1210fc9dfbd8d5c18033.jpg)

![3c39be8ada399a002e5a5ce49c5646f0a20121eb72d5b1bc24bdf8a9387c1388.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/3c39be8ada399a002e5a5ce49c5646f0a20121eb72d5b1bc24bdf8a9387c1388.jpg)

![4e09235738892fb4bf47ed2a787f87ac32d441803c6a8d3c66a0cd3b0d707c2a.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/4e09235738892fb4bf47ed2a787f87ac32d441803c6a8d3c66a0cd3b0d707c2a.jpg)

![51c8d6545ffb63140c2bd2e1bd50447fb24e18a07d5798eb09cbdf97589e5fe2.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/51c8d6545ffb63140c2bd2e1bd50447fb24e18a07d5798eb09cbdf97589e5fe2.jpg)

![62f77e26f5c52d03f26b988354d273172a80a1da13296a02189fd0aa7bc4d143.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/62f77e26f5c52d03f26b988354d273172a80a1da13296a02189fd0aa7bc4d143.jpg)

![6bda1afaf5636dbd592bc01b01439709834c81d09ef5f6e22c96fad7e11784f6.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/6bda1afaf5636dbd592bc01b01439709834c81d09ef5f6e22c96fad7e11784f6.jpg)

![8c7d3fc34c0035f54d3e3ec09c8ca6b3375890f5bde4b2d678c4bfe7d9521719.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/8c7d3fc34c0035f54d3e3ec09c8ca6b3375890f5bde4b2d678c4bfe7d9521719.jpg)

![a32fed57ab76da0fcd46578c2e5869773ea9d9327faf67a4f3333b832d7e2b6c.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/a32fed57ab76da0fcd46578c2e5869773ea9d9327faf67a4f3333b832d7e2b6c.jpg)

![b11a9698dd7ea558977464c907887d88708db8a598ae0cde770e73e2ae8c0063.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/b11a9698dd7ea558977464c907887d88708db8a598ae0cde770e73e2ae8c0063.jpg)

![b80b6fbad7b26125ceb8f8a32ba95105451a432b24f3a6cde6f63528f6b744f6.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/b80b6fbad7b26125ceb8f8a32ba95105451a432b24f3a6cde6f63528f6b744f6.jpg)

![ceb8a5b50b321c1d627c810164759de05ed13bda2f506d583d438899672bda93.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/ceb8a5b50b321c1d627c810164759de05ed13bda2f506d583d438899672bda93.jpg)

![e3c1325694f4b73901f3fb56ff9481bc6874628914d611d114ee6fb932f1dc66.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/e3c1325694f4b73901f3fb56ff9481bc6874628914d611d114ee6fb932f1dc66.jpg)

![f821155b2f130114f5e5816c730a47bf8741ea021c0600886621b857e94d0b99.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/f821155b2f130114f5e5816c730a47bf8741ea021c0600886621b857e94d0b99.jpg)

![fd8404d83012f4030a29171f678489aa41e76be20dece351cdc3fc8205c608f0.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/images/fd8404d83012f4030a29171f678489aa41e76be20dece351cdc3fc8205c608f0.jpg)

### Tables

![563dc617f1caa3c65a887244959235b9d68aba4b95ce41eece4135e3f20099ae.jpg](../icml_results/5746_POLIA_%20Policy%20Optimization%20with%20Visual-Object-Level%20Intrinsic%20Advantage%20for%20Multimodal%20Reasoning/tables/563dc617f1caa3c65a887244959235b9d68aba4b95ce41eece4135e3f20099ae.jpg)

<a id="paper-5698"></a>

## Symbiosis-Inspired Knowledge Distillation for Incremental Object Detection


### Images

![051a19c7e1149a397da676b6f678c27de6a4cd5a86060e2695beb77384ce2639.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/051a19c7e1149a397da676b6f678c27de6a4cd5a86060e2695beb77384ce2639.jpg)

![556792f178c1277a552eaab31e6a4ab3546326b017cbc87b0b8526d57d684a90.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/556792f178c1277a552eaab31e6a4ab3546326b017cbc87b0b8526d57d684a90.jpg)

![5607d141d2ed02cb7b180392d0d229961649e2697144a68fec1e0baa736648fd.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/5607d141d2ed02cb7b180392d0d229961649e2697144a68fec1e0baa736648fd.jpg)

![5d5e04fd762a65761e38bf8f4d20b0de7b6da23980d3a2013e21c2c97272b7fc.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/5d5e04fd762a65761e38bf8f4d20b0de7b6da23980d3a2013e21c2c97272b7fc.jpg)

![77a6e106f2df1d7ecf8067cfc6f4b0cff6fb685d476efd62062a43b69a5ca467.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/77a6e106f2df1d7ecf8067cfc6f4b0cff6fb685d476efd62062a43b69a5ca467.jpg)

![95975be1a760ca58b76ea2744d846685ba28f66e3aef20e9946cbe7ad5abf8d2.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/95975be1a760ca58b76ea2744d846685ba28f66e3aef20e9946cbe7ad5abf8d2.jpg)

![d1a5419e7a44c741e914d5565ed8b7d3f1180ae6121b16d8ca954c0264731492.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/d1a5419e7a44c741e914d5565ed8b7d3f1180ae6121b16d8ca954c0264731492.jpg)

![e9d37468f50d43f695c5af94fa6d0236445ea8235f33cc905abbf42579a85448.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/images/e9d37468f50d43f695c5af94fa6d0236445ea8235f33cc905abbf42579a85448.jpg)

### Tables

![175b2ed33a73c2ff996ea775bba6d114dec9f32ebe2782532a873c5cf189d1f0.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/175b2ed33a73c2ff996ea775bba6d114dec9f32ebe2782532a873c5cf189d1f0.jpg)

![1cf71c1dafb48525d322569d8943ac2d33b27a07a2267bc5f35863ba30da9256.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/1cf71c1dafb48525d322569d8943ac2d33b27a07a2267bc5f35863ba30da9256.jpg)

![2655e68c4db04239fcbd5ace351e4515144bdcafa100c0596f299961861fc863.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/2655e68c4db04239fcbd5ace351e4515144bdcafa100c0596f299961861fc863.jpg)

![3d15f1e66428f6a7960933be5ff2b5dadd39d17b4aafcdbce6c01cab80e35cc6.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/3d15f1e66428f6a7960933be5ff2b5dadd39d17b4aafcdbce6c01cab80e35cc6.jpg)

![42a08144a477fe428d936082f2e5e33ecd5d6475b308759f344ee0c1335376ee.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/42a08144a477fe428d936082f2e5e33ecd5d6475b308759f344ee0c1335376ee.jpg)

![70c181bdffbb3c38ed70c8ea62ca6af6392fc7911a64444a1e79dddc1b44107e.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/70c181bdffbb3c38ed70c8ea62ca6af6392fc7911a64444a1e79dddc1b44107e.jpg)

![e753898f6d55653ec973decc1731d7f85ba3f9f5ff359ee86f751f1037b547ba.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/e753898f6d55653ec973decc1731d7f85ba3f9f5ff359ee86f751f1037b547ba.jpg)

![ee866d9115d07ea72649f40f280a254f4e028976e8dc0f87d303e225168bda0c.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/ee866d9115d07ea72649f40f280a254f4e028976e8dc0f87d303e225168bda0c.jpg)

![f7640f9201f3f6e76ede4b10cf4ae1cd4ff63aadef4c14b7ebaedd7e17d8566d.jpg](../icml_results/5747_Symbiosis-Inspired%20Knowledge%20Distillation%20for%20Incremental%20Object%20Detection/tables/f7640f9201f3f6e76ede4b10cf4ae1cd4ff63aadef4c14b7ebaedd7e17d8566d.jpg)

<a id="paper-5699"></a>

## Bootstrapped Exploration with Causal Reasoning: A Training Paradigm for Adaptive Forecasting Agent


### Images

![342d4ec0581a51a25d01ad214fa2bdd82ba501be48420513457d37451e1c3c73.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/images/342d4ec0581a51a25d01ad214fa2bdd82ba501be48420513457d37451e1c3c73.jpg)

![47be60dc015e1d8a530fe4b7d8ff7981fdd644492c8f940319fdd617cb4b7f85.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/images/47be60dc015e1d8a530fe4b7d8ff7981fdd644492c8f940319fdd617cb4b7f85.jpg)

![9139f3550c9dfea1aba25d6a0e46acf542e747d9c583ca69f192d444fb4bd1a4.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/images/9139f3550c9dfea1aba25d6a0e46acf542e747d9c583ca69f192d444fb4bd1a4.jpg)

![a48e686c4c0a1959a380c80bb01c12a8576346614580073bc7328b317fd7cf43.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/images/a48e686c4c0a1959a380c80bb01c12a8576346614580073bc7328b317fd7cf43.jpg)

### Tables

![423d4cabee239cf48959eabd5a03514acf4eece02d4e9507f172aa3b1d8a6832.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/423d4cabee239cf48959eabd5a03514acf4eece02d4e9507f172aa3b1d8a6832.jpg)

![814718549c7b38bb1c47765ea743a74b3f710b59709feb0e0e36a141c735e937.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/814718549c7b38bb1c47765ea743a74b3f710b59709feb0e0e36a141c735e937.jpg)

![ca7b83d891898c8c424e15647322ac9f5bc3f44a0251e9419a9ee01e629257bf.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/ca7b83d891898c8c424e15647322ac9f5bc3f44a0251e9419a9ee01e629257bf.jpg)

![d19ceb8c4159dfb693fc9293e849d7a10743da4135212e94d1fa60aad20ff8a9.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/d19ceb8c4159dfb693fc9293e849d7a10743da4135212e94d1fa60aad20ff8a9.jpg)

![d2ff68919eb509e8269b6bc98ca1bfbacce18e0a158f7c70a5d361f807f5e4ad.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/d2ff68919eb509e8269b6bc98ca1bfbacce18e0a158f7c70a5d361f807f5e4ad.jpg)

![e64ba344c2f8dc5fc1f51ab5e0bb8dd51120acd39c9551cd69134a1aa36836e6.jpg](../icml_results/5748_Bootstrapped%20Exploration%20with%20Causal%20Reasoning_%20A%20Training%20Paradigm%20for%20Adaptive%20Forecasting%20Agent/tables/e64ba344c2f8dc5fc1f51ab5e0bb8dd51120acd39c9551cd69134a1aa36836e6.jpg)

<a id="paper-5700"></a>

## Beyond Euclidean Summaries: Online Change Point Detection for Distribution-Valued Data


### Images

![08aeb526cb16e8d24141a87890724eac6a1b5e5bff290a5dedb1bade87c3666f.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/08aeb526cb16e8d24141a87890724eac6a1b5e5bff290a5dedb1bade87c3666f.jpg)

![24c40ea9fdcff6babd9b626b15d2f439acc9739d5562c8ac6c16ab189f111b99.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/24c40ea9fdcff6babd9b626b15d2f439acc9739d5562c8ac6c16ab189f111b99.jpg)

![2587378586b3288f137c815f3c73afa0ad94a4a3c28600d9f972062271124fae.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/2587378586b3288f137c815f3c73afa0ad94a4a3c28600d9f972062271124fae.jpg)

![2a79eed000880f9186c0694347460cb2e20a0480c4f33d107ae8323662f01e2e.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/2a79eed000880f9186c0694347460cb2e20a0480c4f33d107ae8323662f01e2e.jpg)

![4760dabbb30df60bc549e5a3a0b82a38c635b358ec99fa1109d3677f7a03604b.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/4760dabbb30df60bc549e5a3a0b82a38c635b358ec99fa1109d3677f7a03604b.jpg)

![6445244a637a23431e35373d99445b67a47d294b780935d595220e76ef8bc523.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/6445244a637a23431e35373d99445b67a47d294b780935d595220e76ef8bc523.jpg)

![89f123821dceadb72ca45c4db9dcac4517f447ed8d3ca54ddb08a3cdec5bb492.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/89f123821dceadb72ca45c4db9dcac4517f447ed8d3ca54ddb08a3cdec5bb492.jpg)

![b40e54444a0e4174cadb7fb6a28e635473530295316475e01c5b2c55ed605ab0.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/b40e54444a0e4174cadb7fb6a28e635473530295316475e01c5b2c55ed605ab0.jpg)

![d056d973c87116e7595f47e1ee45bf26a3bf2c9000f54475c57c4ecb20d26d71.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/d056d973c87116e7595f47e1ee45bf26a3bf2c9000f54475c57c4ecb20d26d71.jpg)

![e33f7fc24b7e4fdb4bbbde5e7eb49fc41c2a701e41b961048d50df154c1f96cb.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/e33f7fc24b7e4fdb4bbbde5e7eb49fc41c2a701e41b961048d50df154c1f96cb.jpg)

![eebe78d708673d1211c0fc030f51cdffcc283913b01372b39894e31bec7b3fac.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/images/eebe78d708673d1211c0fc030f51cdffcc283913b01372b39894e31bec7b3fac.jpg)

### Tables

![17d525c8e933b267fc5033d3b7d2a37ba8290b5ba88f5127a67f705ab6b0b72f.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/17d525c8e933b267fc5033d3b7d2a37ba8290b5ba88f5127a67f705ab6b0b72f.jpg)

![187c310e4f61457e80ef7b67a83c3ba7372ed6799b0d9df185c9eb9c497993b5.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/187c310e4f61457e80ef7b67a83c3ba7372ed6799b0d9df185c9eb9c497993b5.jpg)

![1fa7b1ef6fe5f10881bf4570d8dc07b368584bd1558bbe92a874a00dce18065b.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/1fa7b1ef6fe5f10881bf4570d8dc07b368584bd1558bbe92a874a00dce18065b.jpg)

![34f1a898b508b5a97adb039e4753fcaeaecbaafe72dc88aed55edd2009768ea1.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/34f1a898b508b5a97adb039e4753fcaeaecbaafe72dc88aed55edd2009768ea1.jpg)

![35091675a3bfb2859d767808bb82ac9005387421578d59a8436c207962ac5258.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/35091675a3bfb2859d767808bb82ac9005387421578d59a8436c207962ac5258.jpg)

![47cda067d47b59c130b60231818ba76caa53618dc71b13c1a8f11b5f7cd0e7e3.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/47cda067d47b59c130b60231818ba76caa53618dc71b13c1a8f11b5f7cd0e7e3.jpg)

![5863dd03dc7a8510705dfdd9c2415aaab170505a23aed27632dc55aca1e42128.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/5863dd03dc7a8510705dfdd9c2415aaab170505a23aed27632dc55aca1e42128.jpg)

![5f84d35bc73fab241ec2a1377b2619b4b17f20813ec47fb69d21b2b0ee2736a3.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/5f84d35bc73fab241ec2a1377b2619b4b17f20813ec47fb69d21b2b0ee2736a3.jpg)

![66b70771cf6f5813437447948fbfd3a0006c4a388ba79a2cd367a4d62b813d0b.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/66b70771cf6f5813437447948fbfd3a0006c4a388ba79a2cd367a4d62b813d0b.jpg)

![6b37ad587cf3f445f26434f882d7de0bf15e646cf7ebf71a3e00be00c655714e.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/6b37ad587cf3f445f26434f882d7de0bf15e646cf7ebf71a3e00be00c655714e.jpg)

![6f4fbdded6f643388a06d8ace19156fbc8457eb9cd0560324bc0419d7a01be33.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/6f4fbdded6f643388a06d8ace19156fbc8457eb9cd0560324bc0419d7a01be33.jpg)

![79754e8b9e00714d55f0fb42fb17332dc9485a05f49e5fb94b90b47ea5eba4f6.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/79754e8b9e00714d55f0fb42fb17332dc9485a05f49e5fb94b90b47ea5eba4f6.jpg)

![7f5c0ced09d5337a275acac099d3f0f7e67a5b2c168127a9475b6a336b00c78a.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/7f5c0ced09d5337a275acac099d3f0f7e67a5b2c168127a9475b6a336b00c78a.jpg)

![86d65aae2368d3bf5fe842337153f3912d12273b79fa0b534a3e7111b7594a3e.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/86d65aae2368d3bf5fe842337153f3912d12273b79fa0b534a3e7111b7594a3e.jpg)

![8efbe53f49684516270a428b842d0430621dc53a89c7b1549d21916328a5fdf0.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/8efbe53f49684516270a428b842d0430621dc53a89c7b1549d21916328a5fdf0.jpg)

![9a74d4867d8ab726cddf0583dbb031412b770a4c54e9fef288fd579e902fdaf0.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/9a74d4867d8ab726cddf0583dbb031412b770a4c54e9fef288fd579e902fdaf0.jpg)

![9c4d9f83f633ac9bc9e95a5499f7d0a2a0196af69a916e669c795746209064b2.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/9c4d9f83f633ac9bc9e95a5499f7d0a2a0196af69a916e669c795746209064b2.jpg)

![aaac0e75baf90824802e761c1d28b0a5dca991a54941986c910843fbab56498c.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/aaac0e75baf90824802e761c1d28b0a5dca991a54941986c910843fbab56498c.jpg)

![de9a72681670d386d7b69b7df8eebcb12bf56197c206020894b1dd14a8fe3e90.jpg](../icml_results/5749_Beyond%20Euclidean%20Summaries_%20Online%20Change%20Point%20Detection%20for%20Distribution-Valued%20Data/tables/de9a72681670d386d7b69b7df8eebcb12bf56197c206020894b1dd14a8fe3e90.jpg)

<a id="paper-5701"></a>

## LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth


### Images

![05d5c43f733396a3a71190bc48bd8fe8f3375d3ea32d152eb62169aa1839c1d9.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/images/05d5c43f733396a3a71190bc48bd8fe8f3375d3ea32d152eb62169aa1839c1d9.jpg)

![dbd42518e23019ef3be9027247aebbf80dc3f177e7e906850e4940f76b08757f.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/images/dbd42518e23019ef3be9027247aebbf80dc3f177e7e906850e4940f76b08757f.jpg)

![e0dc6333d0944d9b296bd119d1a784af987e48cc9920518d3738af572e8328dc.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/images/e0dc6333d0944d9b296bd119d1a784af987e48cc9920518d3738af572e8328dc.jpg)

![eafaab7788bbb2d01577afbd89753b34e35931860d208f407f66d08bc2db9543.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/images/eafaab7788bbb2d01577afbd89753b34e35931860d208f407f66d08bc2db9543.jpg)

### Tables

![096e84f50cc0efa9153d2733d2ebd57f816d8d721742364a54473f7a52bd9d28.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/096e84f50cc0efa9153d2733d2ebd57f816d8d721742364a54473f7a52bd9d28.jpg)

![0ab5ceb42f93486744db579ebc063308ba2dc68fad9104ea5e1f9c87477dc82d.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/0ab5ceb42f93486744db579ebc063308ba2dc68fad9104ea5e1f9c87477dc82d.jpg)

![2b2ffe73ce00563b18cd4ba6a739eeeeb66fab6d287dbb9100a2f6c77c6ec386.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/2b2ffe73ce00563b18cd4ba6a739eeeeb66fab6d287dbb9100a2f6c77c6ec386.jpg)

![2ee9f8bf753a0b65ac6f68e3763f4eaa9822300f02e167c0ee5131e81f50e1e0.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/2ee9f8bf753a0b65ac6f68e3763f4eaa9822300f02e167c0ee5131e81f50e1e0.jpg)

![4abcd96822ddc1aaa7b0fd5834d150e1c3aad7923caf6581491aaf5bf090732e.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/4abcd96822ddc1aaa7b0fd5834d150e1c3aad7923caf6581491aaf5bf090732e.jpg)

![4e81cc4b5ea15cdba8cf4b9776252bdc09f6a0176514224091058e1082ffc6c8.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/4e81cc4b5ea15cdba8cf4b9776252bdc09f6a0176514224091058e1082ffc6c8.jpg)

![54c1dd6053d9f2f62bf7085022f2aff5893bf3537a4340cd40273671b6b86b2d.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/54c1dd6053d9f2f62bf7085022f2aff5893bf3537a4340cd40273671b6b86b2d.jpg)

![87fb8ff9973482b628f939fd3d25a1a0daf1ba83fc1041d06ff034e6560b2480.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/87fb8ff9973482b628f939fd3d25a1a0daf1ba83fc1041d06ff034e6560b2480.jpg)

![8e47eb45e2658777fb01c60ee6877ee2a9e01a403e373ecaa79724efa8bf0adb.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/8e47eb45e2658777fb01c60ee6877ee2a9e01a403e373ecaa79724efa8bf0adb.jpg)

![990091b245652c9ffdda18d32f6a4e5d1245f66fa0eb2aa54e50940d85ad4676.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/990091b245652c9ffdda18d32f6a4e5d1245f66fa0eb2aa54e50940d85ad4676.jpg)

![9af0cb8da3bf57234cf6d7b21c9940a984542a83ccacaf797ab1cfabec3c076b.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/9af0cb8da3bf57234cf6d7b21c9940a984542a83ccacaf797ab1cfabec3c076b.jpg)

![9c68aa5a59ceaa1c083749f8908be988a801cf445dd373737087a6eb8c0fd54e.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/9c68aa5a59ceaa1c083749f8908be988a801cf445dd373737087a6eb8c0fd54e.jpg)

![c871748a636e9cc63d5c90614aefad26640b5874970b8ff7cc974c4ed4d3d116.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/c871748a636e9cc63d5c90614aefad26640b5874970b8ff7cc974c4ed4d3d116.jpg)

![e0ee1c7ca04058814c7383eb640bb5c88e3bff60e93dceb35d81b2afedea549b.jpg](../icml_results/5750_LOCA-bench_%20Benchmarking%20Language%20Agents%20Under%20Controllable%20and%20Extreme%20Context%20Growth/tables/e0ee1c7ca04058814c7383eb640bb5c88e3bff60e93dceb35d81b2afedea549b.jpg)

<a id="paper-5702"></a>

## Attention with Routed-Memory for Learnable Sparse Control


### Images

![36a52f882424deb11126c660ea9e6e53ef841b215e59aa413a15c181252de3d5.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/images/36a52f882424deb11126c660ea9e6e53ef841b215e59aa413a15c181252de3d5.jpg)

![c26e8bd3f848057f923412f9856278104b2059bc693b184b763c8701fa7fedd7.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/images/c26e8bd3f848057f923412f9856278104b2059bc693b184b763c8701fa7fedd7.jpg)

### Tables

![02a350a1834a18790a87d91e8fb9265d800b04df11d6f8f39f7ac792c1b41d63.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/02a350a1834a18790a87d91e8fb9265d800b04df11d6f8f39f7ac792c1b41d63.jpg)

![1a85c6628bd7e1a72152b67405142b7cd521a09c7cb78fb320a70a21e9c8e3d7.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/1a85c6628bd7e1a72152b67405142b7cd521a09c7cb78fb320a70a21e9c8e3d7.jpg)

![1e3cd73bf2e15474b715e755bc9c16f25f6255214d824626620b344a054219bb.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/1e3cd73bf2e15474b715e755bc9c16f25f6255214d824626620b344a054219bb.jpg)

![31d6b066dd4e1981a57a18e2f519b69e41ac47efae16859bf6e6f2b4b46f1dba.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/31d6b066dd4e1981a57a18e2f519b69e41ac47efae16859bf6e6f2b4b46f1dba.jpg)

![5da54af3f0c109f992f1d9aa3b5cbc7d690151d064432e7196a4b2ca92bac650.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/5da54af3f0c109f992f1d9aa3b5cbc7d690151d064432e7196a4b2ca92bac650.jpg)

![8a3eca7620e80ea6715f320166734a83a36307ea06163161477cb961a8071e2d.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/8a3eca7620e80ea6715f320166734a83a36307ea06163161477cb961a8071e2d.jpg)

![8ef32158cebb203296e8b406bee8cc8eef1a0a058430063ccb47d6ba06a931c9.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/8ef32158cebb203296e8b406bee8cc8eef1a0a058430063ccb47d6ba06a931c9.jpg)

![e8ed7911791f9f16ae6027ac6caff437839ef212d5b397f2d01582585dd6e301.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/e8ed7911791f9f16ae6027ac6caff437839ef212d5b397f2d01582585dd6e301.jpg)

![f85f887aebbf9124ece47ec87b4a5ccaac8f97c58e0342a2f7849bd1e0ee881b.jpg](../icml_results/5751_Attention%20with%20Routed-Memory%20for%20Learnable%20Sparse%20Control/tables/f85f887aebbf9124ece47ec87b4a5ccaac8f97c58e0342a2f7849bd1e0ee881b.jpg)

<a id="paper-5703"></a>

## UCPO: Uncertainty-Aware Policy Optimization


### Images

![055af34de0df8933c2ed1a9b1f52704459aef0349fdd46c1ee5080efb6311086.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/055af34de0df8933c2ed1a9b1f52704459aef0349fdd46c1ee5080efb6311086.jpg)

![09aa4a6172d6fd273eb0d8ce1691eaabe0ec7901935dcfa5d8927b9d70b3959e.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/09aa4a6172d6fd273eb0d8ce1691eaabe0ec7901935dcfa5d8927b9d70b3959e.jpg)

![0c4a4d6a45ef44b5adb83c9db677a79749d2d26d82f2f0ba312a0affe179200a.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/0c4a4d6a45ef44b5adb83c9db677a79749d2d26d82f2f0ba312a0affe179200a.jpg)

![4b1b6b5bae173813ed16dfd0a79f7198ba5669de072bc3a9f63b27e0f8a9a716.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/4b1b6b5bae173813ed16dfd0a79f7198ba5669de072bc3a9f63b27e0f8a9a716.jpg)

![52d0a8356bd6ac155e62f8ab769e40846a95ca90491bbc85fdb8031f083e7687.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/52d0a8356bd6ac155e62f8ab769e40846a95ca90491bbc85fdb8031f083e7687.jpg)

![609dcac82ec973e2ec3fa287d9503581cbe774adac462ae8cf0e99dc272695dd.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/609dcac82ec973e2ec3fa287d9503581cbe774adac462ae8cf0e99dc272695dd.jpg)

![6d0b8917b4675188adbea0882663488c56d066bf4749143e0bce90cdd5a9fb69.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/6d0b8917b4675188adbea0882663488c56d066bf4749143e0bce90cdd5a9fb69.jpg)

![77d35f7de98c74ac58d819e6cf9f4a65ce6d29205cba79dac9b9ccff4d36496c.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/77d35f7de98c74ac58d819e6cf9f4a65ce6d29205cba79dac9b9ccff4d36496c.jpg)

![795e62c9aec99176f5576d40e886a2dc0a02d871c3257525de2417397a4ffb8f.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/795e62c9aec99176f5576d40e886a2dc0a02d871c3257525de2417397a4ffb8f.jpg)

![ab78ed63cf55d9849cb0fb2a11b4e3911cc2f9b30a499b1d71d5ba3785325a53.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/ab78ed63cf55d9849cb0fb2a11b4e3911cc2f9b30a499b1d71d5ba3785325a53.jpg)

![f3e6fd8e2de80a9366912517c358bf984c84497720e6a1e79e9c833626cd6345.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/f3e6fd8e2de80a9366912517c358bf984c84497720e6a1e79e9c833626cd6345.jpg)

![f48f92f8124daa374598fa0a1c93cd6997e3fa2cb7ac40f1459a4cfb19408dad.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/images/f48f92f8124daa374598fa0a1c93cd6997e3fa2cb7ac40f1459a4cfb19408dad.jpg)

### Tables

![61c2ae9a165e82da6149e25484f227f5760105fdbaed233d0bb8f909731e783f.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/61c2ae9a165e82da6149e25484f227f5760105fdbaed233d0bb8f909731e783f.jpg)

![6dcb1e61b041823633096033370e81f832bbbff90a001f00a9780ed3818a65e4.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/6dcb1e61b041823633096033370e81f832bbbff90a001f00a9780ed3818a65e4.jpg)

![76b10e121841ee6bb7fbf3b1153050b4bb0febcfda91dfa11ba1404d18b133db.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/76b10e121841ee6bb7fbf3b1153050b4bb0febcfda91dfa11ba1404d18b133db.jpg)

![80b36fd5fb78784bb88d5e39076ac27f851cd986c8c1ca818efee5304eb1add4.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/80b36fd5fb78784bb88d5e39076ac27f851cd986c8c1ca818efee5304eb1add4.jpg)

![85937b8ca47e36d5e528340d41c593bd892d1004ebd5be1777c162e8cf1b97d8.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/85937b8ca47e36d5e528340d41c593bd892d1004ebd5be1777c162e8cf1b97d8.jpg)

![927c1872371514e9dcd5734601be34d89e030bd8bf6f65fdacbe8e2aa32fc461.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/927c1872371514e9dcd5734601be34d89e030bd8bf6f65fdacbe8e2aa32fc461.jpg)

![9bdec74e06aceea734fe4474ae69e2b1d957cc270d874184f401a8cde2b400a4.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/9bdec74e06aceea734fe4474ae69e2b1d957cc270d874184f401a8cde2b400a4.jpg)

![c934f766cc0b64076ea51af242afd1e6c931922a48158e6cb9a603ef7b885376.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/c934f766cc0b64076ea51af242afd1e6c931922a48158e6cb9a603ef7b885376.jpg)

![ed413d158f8e04167ae7a05dbf32c759ac55655547954cbb490ff66d1056c8b8.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/ed413d158f8e04167ae7a05dbf32c759ac55655547954cbb490ff66d1056c8b8.jpg)

![ef053a12258de0806c1e1536c949296ee9597e0c80b2dd686742099e60c98f4c.jpg](../icml_results/5752_UCPO_%20Uncertainty-Aware%20Policy%20Optimization/tables/ef053a12258de0806c1e1536c949296ee9597e0c80b2dd686742099e60c98f4c.jpg)

<a id="paper-5704"></a>

## RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments


### Images

![0a426cb92056e2d98c3a0f077d333cc561e40d9a47d03e82e648192ff657342e.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/0a426cb92056e2d98c3a0f077d333cc561e40d9a47d03e82e648192ff657342e.jpg)

![10124416c71306683d9912650970a6a533e00f5b04e642a4523c1196657ee486.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/10124416c71306683d9912650970a6a533e00f5b04e642a4523c1196657ee486.jpg)

![461b03d8691552ec4ef7ae38af152196db2f63485946e01eec85fa5dab701242.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/461b03d8691552ec4ef7ae38af152196db2f63485946e01eec85fa5dab701242.jpg)

![6446e0f052bff34f543897bddd0469165821328a65ce7d6ba91de54be0e993b6.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/6446e0f052bff34f543897bddd0469165821328a65ce7d6ba91de54be0e993b6.jpg)

![ae628351bbd91e84492a05f5cc5c74e61b4522c72ae514094c741bdb663b7acd.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/ae628351bbd91e84492a05f5cc5c74e61b4522c72ae514094c741bdb663b7acd.jpg)

![b585bd870925f02fd881bcdb806d944f785f966c7b6d75926d481224e3ccc0a0.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/b585bd870925f02fd881bcdb806d944f785f966c7b6d75926d481224e3ccc0a0.jpg)

![b97636d396eedc83dffff331f9f3a99b4705fccb91403614a8af5db3c10aea31.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/b97636d396eedc83dffff331f9f3a99b4705fccb91403614a8af5db3c10aea31.jpg)

![c220f6598a02a89320ad73ba863a8d565b91d267281764af4c7fdd47fc80176d.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/c220f6598a02a89320ad73ba863a8d565b91d267281764af4c7fdd47fc80176d.jpg)

![e6f37a57211f7c6c8f465545e0cb3b455003574982e8d11dad11fc6fdf997591.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/images/e6f37a57211f7c6c8f465545e0cb3b455003574982e8d11dad11fc6fdf997591.jpg)

### Tables

![981a67e8092fc32af34e31a456be916e8ca8acba7c5afb90d14d407cd82176e1.jpg](../icml_results/5753_RLVE_%20Scaling%20Up%20Reinforcement%20Learning%20for%20Language%20Models%20with%20Adaptive%20Verifiable%20Environments/tables/981a67e8092fc32af34e31a456be916e8ca8acba7c5afb90d14d407cd82176e1.jpg)

<a id="paper-5705"></a>

## Convex Dataset Valuation for Post-Training


### Images

![170967b8137371d75228acee97fd02f4171abf7256430fa980ef3c75ddf51a30.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/170967b8137371d75228acee97fd02f4171abf7256430fa980ef3c75ddf51a30.jpg)

![6aabd8c746d9f8423de64a8a078810fb8572b172d4b00afc0b44273875838a75.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/6aabd8c746d9f8423de64a8a078810fb8572b172d4b00afc0b44273875838a75.jpg)

![6c8998ddd85b52861a764a829839577153e7d362e42024556a9dfc8aa0430fc4.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/6c8998ddd85b52861a764a829839577153e7d362e42024556a9dfc8aa0430fc4.jpg)

![a6cb3b96bce6d9edf4778b0c04d3227fc563da8d88501991b28cc7dadfb0fd66.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/a6cb3b96bce6d9edf4778b0c04d3227fc563da8d88501991b28cc7dadfb0fd66.jpg)

![aa86bd6e75e4a3e843d3c42d6ae8a74fea23bb5194e552e62cb4012dbdcf0cca.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/aa86bd6e75e4a3e843d3c42d6ae8a74fea23bb5194e552e62cb4012dbdcf0cca.jpg)

![e5297fea1803886c737ea891be9890da45ff3b53d876a5f94cc938cca56ce402.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/e5297fea1803886c737ea891be9890da45ff3b53d876a5f94cc938cca56ce402.jpg)

![fcd681f3b2bc6f22cfc2116007397f6d73f571d4f6efd48e9aba4a9348872840.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/fcd681f3b2bc6f22cfc2116007397f6d73f571d4f6efd48e9aba4a9348872840.jpg)

![ffc0e17ac51e12d7cf9522fb1f794977d8bd9b8b818bb6f1a87eed33665a26c0.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/images/ffc0e17ac51e12d7cf9522fb1f794977d8bd9b8b818bb6f1a87eed33665a26c0.jpg)

### Tables

![033b1eb4471d4ff2ad1b3777414e7e88538cc960fdb83c21ec358fa92b8b270d.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/033b1eb4471d4ff2ad1b3777414e7e88538cc960fdb83c21ec358fa92b8b270d.jpg)

![08a80410938107b98904642b6983d16434dbb024bb62d9db43d5c653d0c7d692.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/08a80410938107b98904642b6983d16434dbb024bb62d9db43d5c653d0c7d692.jpg)

![24f4224e4fc1d56b6637c63b40916db81dfbf2b3221295a65fca060ecdca62b8.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/24f4224e4fc1d56b6637c63b40916db81dfbf2b3221295a65fca060ecdca62b8.jpg)

![46a73868083a1d8f05fced8f496798858437c430e5f066bccd4973e05a71817a.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/46a73868083a1d8f05fced8f496798858437c430e5f066bccd4973e05a71817a.jpg)

![46aa5a8369085bb8051dc41d1e13538bd3f8170edd9b72f1fa922be1f9548942.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/46aa5a8369085bb8051dc41d1e13538bd3f8170edd9b72f1fa922be1f9548942.jpg)

![5981b327591a4d701e78dbc80909958a5f9ce687d86c3d9f21497d1e463901cc.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/5981b327591a4d701e78dbc80909958a5f9ce687d86c3d9f21497d1e463901cc.jpg)

![83862dca959b11450cdd143bab6ecbeceb344fca55a469485a1ba7014babe882.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/83862dca959b11450cdd143bab6ecbeceb344fca55a469485a1ba7014babe882.jpg)

![878115f2cd03cac1d2f5228371f67c6047a0a8cdf580614732ab0f417d6cf561.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/878115f2cd03cac1d2f5228371f67c6047a0a8cdf580614732ab0f417d6cf561.jpg)

![8a6abd41782b0ab55013a8a428e10ca85ef5c2831be71d4862aea9e9881d8e03.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/8a6abd41782b0ab55013a8a428e10ca85ef5c2831be71d4862aea9e9881d8e03.jpg)

![ab4b17664c49b7aa5f2b0b01050803dd72dd55015f64b554971dc3d5cdac45ac.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/ab4b17664c49b7aa5f2b0b01050803dd72dd55015f64b554971dc3d5cdac45ac.jpg)

![acf7c6f6b1824f0f15892107449f2b48f8eec05cd404e2f50cb61466f420aa5d.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/acf7c6f6b1824f0f15892107449f2b48f8eec05cd404e2f50cb61466f420aa5d.jpg)

![b7c4fcd918b83b6e4f36f163dd495acbd13d02f983d10caaf2ff3a251d74ad12.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/b7c4fcd918b83b6e4f36f163dd495acbd13d02f983d10caaf2ff3a251d74ad12.jpg)

![bb305698dbea2651e61a7ec5e93dddeea03afb440e93f24e84ed19b9d82c35b2.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/bb305698dbea2651e61a7ec5e93dddeea03afb440e93f24e84ed19b9d82c35b2.jpg)

![c058a4cd5101112b2034a849f8691071aed8ffacb4898d0bdbcf5a9ef848f790.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/c058a4cd5101112b2034a849f8691071aed8ffacb4898d0bdbcf5a9ef848f790.jpg)

![c0eb783861c12bb1afc108b2a73c2dd782c84f5d602d3660aec660f7d73c2f19.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/c0eb783861c12bb1afc108b2a73c2dd782c84f5d602d3660aec660f7d73c2f19.jpg)

![ca9576675fc4a1b3b58737db2812df7cf7b0b8e596f7928425ccf10497854b6b.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/ca9576675fc4a1b3b58737db2812df7cf7b0b8e596f7928425ccf10497854b6b.jpg)

![d1eb18665936a472fc543faf66eb014533af5b2f3fddd7ab9e747b63c65de594.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/d1eb18665936a472fc543faf66eb014533af5b2f3fddd7ab9e747b63c65de594.jpg)

![e7e02c0a1e835a0e597eb0a17bfa3c7f427a13ace1a0dcc041d0727827f15a54.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/e7e02c0a1e835a0e597eb0a17bfa3c7f427a13ace1a0dcc041d0727827f15a54.jpg)

![fc23acd589c00d746485f495f70f45e62c1d1ed045c2f8d64671ca1fc83ce4e5.jpg](../icml_results/5754_Convex%20Dataset%20Valuation%20for%20Post-Training/tables/fc23acd589c00d746485f495f70f45e62c1d1ed045c2f8d64671ca1fc83ce4e5.jpg)

<a id="paper-5706"></a>

## Large Language Models Explore by Latent Distilling


### Images

![4a96970947e3a4b76d16acc261c4ca978ee3087674436e7c740f18226050cdd0.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/images/4a96970947e3a4b76d16acc261c4ca978ee3087674436e7c740f18226050cdd0.jpg)

![8a6b189833c6f8fa8b8e4f011d7cdeb68a82af7dd84ed982fd645baa01fe5040.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/images/8a6b189833c6f8fa8b8e4f011d7cdeb68a82af7dd84ed982fd645baa01fe5040.jpg)

![8d69082f47dc9add046f649004ec5be31283caa9680ea9fd0eed093c7af61781.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/images/8d69082f47dc9add046f649004ec5be31283caa9680ea9fd0eed093c7af61781.jpg)

![9a226b29f57c5217c25a2898546aee88c78f3a1542e2b165ea8dbc76735796a7.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/images/9a226b29f57c5217c25a2898546aee88c78f3a1542e2b165ea8dbc76735796a7.jpg)

![cc59b8391f2d562f73a78a7662e0c6cc364e8e522b89eae2b53d1a64044ea922.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/images/cc59b8391f2d562f73a78a7662e0c6cc364e8e522b89eae2b53d1a64044ea922.jpg)

### Tables

![00eff1c4608531172350a0e19122459cb94b450cb16806c9307b7ddbbeb51ba3.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/00eff1c4608531172350a0e19122459cb94b450cb16806c9307b7ddbbeb51ba3.jpg)

![12184c49f7656b967d0641b49e8a3d05e4231fa675a74d9e2b85b94aecd9b816.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/12184c49f7656b967d0641b49e8a3d05e4231fa675a74d9e2b85b94aecd9b816.jpg)

![176685e2e4bd4008af45f4d3ef30ed9c3a58f9c47b98371340fbb094fe8ddbf4.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/176685e2e4bd4008af45f4d3ef30ed9c3a58f9c47b98371340fbb094fe8ddbf4.jpg)

![460e7677f1b64ff2f8754574cf602086ec55760bd06d414c6ff36656f64727ea.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/460e7677f1b64ff2f8754574cf602086ec55760bd06d414c6ff36656f64727ea.jpg)

![5d3bc31f2e13a69cde79d2833e8243436850ed947c230f0a139ab3181d204ec9.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/5d3bc31f2e13a69cde79d2833e8243436850ed947c230f0a139ab3181d204ec9.jpg)

![62f4fec94a0ad5bd1b03417493e540377df681ca490b3356751d2582c54ae24c.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/62f4fec94a0ad5bd1b03417493e540377df681ca490b3356751d2582c54ae24c.jpg)

![699fb436538ed4c495842adfe162f8e6c81e7ed8e582230d809b326d3c2dab43.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/699fb436538ed4c495842adfe162f8e6c81e7ed8e582230d809b326d3c2dab43.jpg)

![72064ca006dc5008e89c77befd8b8c62a7675eeb9187fe03cea26b4457902090.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/72064ca006dc5008e89c77befd8b8c62a7675eeb9187fe03cea26b4457902090.jpg)

![7ccf0b14ec1339edd06e384fd926bbf2da598150a9ef4adfae86c939af02c257.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/7ccf0b14ec1339edd06e384fd926bbf2da598150a9ef4adfae86c939af02c257.jpg)

![9039eec8baba22b92b6e11a0feb3e6902a9eb635cb0bf447a86b78f9195ed0bd.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/9039eec8baba22b92b6e11a0feb3e6902a9eb635cb0bf447a86b78f9195ed0bd.jpg)

![9c23c6fdf96b1751459e575313d32ee2a74ec488ccf473d1be7956d1f968cdc3.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/9c23c6fdf96b1751459e575313d32ee2a74ec488ccf473d1be7956d1f968cdc3.jpg)

![b13d0838d4126717cff4b5e39a9ef20447336def4f8dd793d6421e685c3b7554.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/b13d0838d4126717cff4b5e39a9ef20447336def4f8dd793d6421e685c3b7554.jpg)

![b5153692ca33d7ef04b82590c06775e9afd67387a03cad29e5dbb3ae9f718ede.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/b5153692ca33d7ef04b82590c06775e9afd67387a03cad29e5dbb3ae9f718ede.jpg)

![b51888164f4d5b6fb36ec187ee9378a7c29b34fffb264b62982c4ced39d9b9c4.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/b51888164f4d5b6fb36ec187ee9378a7c29b34fffb264b62982c4ced39d9b9c4.jpg)

![bbf1233b6b2836d78a468d8257a4f5613b2ef19999d55d887ab4c34c2ee88123.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/bbf1233b6b2836d78a468d8257a4f5613b2ef19999d55d887ab4c34c2ee88123.jpg)

![bf9222fe70979dcb2c2a28db5fc3ef51ff1e57a60c315dc4a95aebcc6a8a86f0.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/bf9222fe70979dcb2c2a28db5fc3ef51ff1e57a60c315dc4a95aebcc6a8a86f0.jpg)

![c8caad504bab6c83e7d2da1bb7a717a1cbca1500bfb88e0da2f902335221262e.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/c8caad504bab6c83e7d2da1bb7a717a1cbca1500bfb88e0da2f902335221262e.jpg)

![d728b4f2be832aaa12ca7bd7af247d0415a6963245ce369d874f4714579df836.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/d728b4f2be832aaa12ca7bd7af247d0415a6963245ce369d874f4714579df836.jpg)

![df5b9cc98f739065b9192d2112c6537a0a2fffb620015756c8c6d7fb617db620.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/df5b9cc98f739065b9192d2112c6537a0a2fffb620015756c8c6d7fb617db620.jpg)

![e4fb09591ca55d753edc49c6f8395055cd38d5f4a91b7734a4c50954574696e6.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/e4fb09591ca55d753edc49c6f8395055cd38d5f4a91b7734a4c50954574696e6.jpg)

![e96ee7150edc178fa16a3c7c7dcc6c9b94758d87af81f80a783084caef985de5.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/e96ee7150edc178fa16a3c7c7dcc6c9b94758d87af81f80a783084caef985de5.jpg)

![fa6618fcce151e3ece4904c6ea1abf7164ad675b15b39486e60463bd78e3875e.jpg](../icml_results/5755_Large%20Language%20Models%20Explore%20by%20Latent%20Distilling/tables/fa6618fcce151e3ece4904c6ea1abf7164ad675b15b39486e60463bd78e3875e.jpg)

<a id="paper-5707"></a>

## TaskLoom: Weaving Knowledge Across Tasks in World Models


### Images

![02e63df2b126c9d04e4ffcf0974a0f6dec0b9426072917cd50e99998009aa0f4.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/02e63df2b126c9d04e4ffcf0974a0f6dec0b9426072917cd50e99998009aa0f4.jpg)

![1001c61c9c51165c85f4803c56ccfae024cbc027149da616122b0726ec3d1a74.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/1001c61c9c51165c85f4803c56ccfae024cbc027149da616122b0726ec3d1a74.jpg)

![68e7d9177b93e6484644c7440a13bd6e7e02d1cc87104697704d9cbc4d3cbc22.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/68e7d9177b93e6484644c7440a13bd6e7e02d1cc87104697704d9cbc4d3cbc22.jpg)

![97286e3b51c0fde47cf1b21d05521f486aaebd4867ffe9c296730c39b4c26602.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/97286e3b51c0fde47cf1b21d05521f486aaebd4867ffe9c296730c39b4c26602.jpg)

![d528920a5ffa4607b2c58058d76f156c49f27df1c84b198eba1d7e321ed17330.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/d528920a5ffa4607b2c58058d76f156c49f27df1c84b198eba1d7e321ed17330.jpg)

![d9884c1b7d27ecdfd9d48b4f9218634f3310f3b007c02433c7f938e4da010efe.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/d9884c1b7d27ecdfd9d48b4f9218634f3310f3b007c02433c7f938e4da010efe.jpg)

![efff6e1f5d4633b31c075219e7da6faca2aa0d845ffda52cb759937c22ec6fb9.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/efff6e1f5d4633b31c075219e7da6faca2aa0d845ffda52cb759937c22ec6fb9.jpg)

![f5b5d77a141af8b5b0ad54257af900d75b1369ce7d50f521d0ddebb61b92e3aa.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/images/f5b5d77a141af8b5b0ad54257af900d75b1369ce7d50f521d0ddebb61b92e3aa.jpg)

### Tables

![0b465de0dda45f521b6e3c32eeede09caac89e3086e6349a11647463648280c4.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/0b465de0dda45f521b6e3c32eeede09caac89e3086e6349a11647463648280c4.jpg)

![0d8cbdd22deecbd9d3c452e1c18b751353f18d02f5a6dad5a705d333ff3d5eb8.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/0d8cbdd22deecbd9d3c452e1c18b751353f18d02f5a6dad5a705d333ff3d5eb8.jpg)

![0f5633a1a4cb5ed1c8b9690afc14ca7dc05d9190acf91ef82703f0832fff3bc9.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/0f5633a1a4cb5ed1c8b9690afc14ca7dc05d9190acf91ef82703f0832fff3bc9.jpg)

![1791b61333e06e2b5356571a3796bcfdabeac024cc242adc062d7282a7792413.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/1791b61333e06e2b5356571a3796bcfdabeac024cc242adc062d7282a7792413.jpg)

![1cd9130a82b5669d6b96ca09e3efc87c09852add276fdddf65cfce544ba42eac.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/1cd9130a82b5669d6b96ca09e3efc87c09852add276fdddf65cfce544ba42eac.jpg)

![25f1c1d20d2a9db409aef23922b48bda90a0628e842c9c48b66bb5a8291a4ea6.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/25f1c1d20d2a9db409aef23922b48bda90a0628e842c9c48b66bb5a8291a4ea6.jpg)

![3197c140dc2df83c28554a4f57b68d5029d7bb78961c1bb9dbc702eb9cbd2159.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/3197c140dc2df83c28554a4f57b68d5029d7bb78961c1bb9dbc702eb9cbd2159.jpg)

![5e6502c019bc65a3be101dff46ae4f228dfda6de11982f9f10f32bcf10fbb058.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/5e6502c019bc65a3be101dff46ae4f228dfda6de11982f9f10f32bcf10fbb058.jpg)

![74c042690bc9c6f35f094678899c296e7206e403f4619fc8e1181676975f4c9f.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/74c042690bc9c6f35f094678899c296e7206e403f4619fc8e1181676975f4c9f.jpg)

![77791155867edbadb1682e67532980e63f3b352a47841e6589fd1c2767c2cfce.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/77791155867edbadb1682e67532980e63f3b352a47841e6589fd1c2767c2cfce.jpg)

![80f40464ba05caab6b70b744259ab6bbcbe0dcd0fd56f51df5fde5b46d096454.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/80f40464ba05caab6b70b744259ab6bbcbe0dcd0fd56f51df5fde5b46d096454.jpg)

![88656c879eb23ed72cc1b4903545072e878ae8c350723a5938e8777f9bfad3a2.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/88656c879eb23ed72cc1b4903545072e878ae8c350723a5938e8777f9bfad3a2.jpg)

![8a9e1c6c387a1a71d2a6e3a2218c9462999589094ec6269b047cf8869a2fec20.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/8a9e1c6c387a1a71d2a6e3a2218c9462999589094ec6269b047cf8869a2fec20.jpg)

![a37228b52aa9a2c52245e423d443753b6ad4017b55e72ac564d73d39e3a796c7.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/a37228b52aa9a2c52245e423d443753b6ad4017b55e72ac564d73d39e3a796c7.jpg)

![b9a4a415869d2ce75956b1b72d8dd02186bb755d7514e8bd4a4d9d221985d21d.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/b9a4a415869d2ce75956b1b72d8dd02186bb755d7514e8bd4a4d9d221985d21d.jpg)

![c854bd81355146ff2dbccb5c1f98a791e085be937d7c81b40ce1469691ac0cb2.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/c854bd81355146ff2dbccb5c1f98a791e085be937d7c81b40ce1469691ac0cb2.jpg)

![dc0d5348aa1f67c8a4770b060a26a6cce586620083cfbf2252cd71f2fd900382.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/dc0d5348aa1f67c8a4770b060a26a6cce586620083cfbf2252cd71f2fd900382.jpg)

![fdb259d09550fb77069033d90539e386a2cfdbba8ffe37e9df1b493fad21b675.jpg](../icml_results/5756_TaskLoom_%20Weaving%20Knowledge%20Across%20Tasks%20in%20World%20Models/tables/fdb259d09550fb77069033d90539e386a2cfdbba8ffe37e9df1b493fad21b675.jpg)

<a id="paper-5708"></a>

## Normality Calibration in Semi-supervised Graph Anomaly Detection


### Images

![0c1651213f134ce9d66bf482d3726d488b1d7218710d4657d01204234fe3a21d.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/0c1651213f134ce9d66bf482d3726d488b1d7218710d4657d01204234fe3a21d.jpg)

![23d76a9f8eca678405df0cda5fbe9c03e17f5d0f971d469a1babbe13321f2001.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/23d76a9f8eca678405df0cda5fbe9c03e17f5d0f971d469a1babbe13321f2001.jpg)

![4ab70468d34f1e170ac9e1b5730c0c9c51215cc32275983fa7968825c10d193c.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/4ab70468d34f1e170ac9e1b5730c0c9c51215cc32275983fa7968825c10d193c.jpg)

![60e4c2cfe9bb1e864574d97122e0bb5669db67ad605e86e340cfea567abd1143.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/60e4c2cfe9bb1e864574d97122e0bb5669db67ad605e86e340cfea567abd1143.jpg)

![62127dd10f8d30f9f96688966c84a0dde984e4b55a78770dde7e10bc4af7da52.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/62127dd10f8d30f9f96688966c84a0dde984e4b55a78770dde7e10bc4af7da52.jpg)

![c724c795db33a1edd8fc0b335fd8927f3f0b1bc592e257b7f9b06aedcaae643b.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/c724c795db33a1edd8fc0b335fd8927f3f0b1bc592e257b7f9b06aedcaae643b.jpg)

![cdb90d1619621984288234b2f391726f8b308988dcba94ef2b1f141db1708194.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/cdb90d1619621984288234b2f391726f8b308988dcba94ef2b1f141db1708194.jpg)

![d6bd107a62b86ea7ea0d7b7d09b89da4703dc30a19b1e4d4e622ab1f72b34f10.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/d6bd107a62b86ea7ea0d7b7d09b89da4703dc30a19b1e4d4e622ab1f72b34f10.jpg)

![dca397c4d2a2ac3d918933f2f463acc6f037ab045cd18d57c1a31163ce9614ca.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/dca397c4d2a2ac3d918933f2f463acc6f037ab045cd18d57c1a31163ce9614ca.jpg)

![dfe981bee4bf16a8b5e2e8df6e186e200351f31a35332ab5c6e533f94157f839.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/dfe981bee4bf16a8b5e2e8df6e186e200351f31a35332ab5c6e533f94157f839.jpg)

![f5609b54988e23d5c87e64258260dcefaeb0c015488a603a890787ce48dcd955.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/images/f5609b54988e23d5c87e64258260dcefaeb0c015488a603a890787ce48dcd955.jpg)

### Tables

![09f33ded3a569b737cce2821bd1936a203dd8f86ee52437f5c223228a1f1867a.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/09f33ded3a569b737cce2821bd1936a203dd8f86ee52437f5c223228a1f1867a.jpg)

![1aad813913e1fb1014f86d8324b2b121f20a5c460fb9500d153c8ab3a5b29dd4.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/1aad813913e1fb1014f86d8324b2b121f20a5c460fb9500d153c8ab3a5b29dd4.jpg)

![2cbdab0a5f936db8817f78e7ecbfca736e673752fa4d86e8ba9b978469c5d65e.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/2cbdab0a5f936db8817f78e7ecbfca736e673752fa4d86e8ba9b978469c5d65e.jpg)

![7db3a5b1e96d905032f0829de080d123d31adc40ca4e91fe9a1bdbb21ab5a898.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/7db3a5b1e96d905032f0829de080d123d31adc40ca4e91fe9a1bdbb21ab5a898.jpg)

![8354604b5bbd29b9d6d9365eebdc438585aff2bb3b8a1f392a2554910ad18d5c.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/8354604b5bbd29b9d6d9365eebdc438585aff2bb3b8a1f392a2554910ad18d5c.jpg)

![8ebece282af19dca75601eecfbc2a5d3bd4efa3c470a4322e9441324f1427e59.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/8ebece282af19dca75601eecfbc2a5d3bd4efa3c470a4322e9441324f1427e59.jpg)

![af3dbd2a4f0873c66b0c94915dded797ac951760039809f9efe0b4c37643a927.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/af3dbd2a4f0873c66b0c94915dded797ac951760039809f9efe0b4c37643a927.jpg)

![ca5197284432a7bd61ff3cdf734205f3c0595a8c81cb22d899d5b5dbca2198b2.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/ca5197284432a7bd61ff3cdf734205f3c0595a8c81cb22d899d5b5dbca2198b2.jpg)

![d7b06b2b6cfd5e9b360861e3fa37ccf8b3ad261660f6426a1235cc6eda7ea05b.jpg](../icml_results/5757_Normality%20Calibration%20in%20Semi-supervised%20Graph%20Anomaly%20Detection/tables/d7b06b2b6cfd5e9b360861e3fa37ccf8b3ad261660f6426a1235cc6eda7ea05b.jpg)

<a id="paper-5709"></a>

## MTNL: A Unified Modeling Perspective for Enhancing Tensor Network Learning

### Images

![073830bd6636fa74de2307a5dd84cc4936a15e218af009517c023371b336aecf.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/073830bd6636fa74de2307a5dd84cc4936a15e218af009517c023371b336aecf.jpg)

![0ed275c2782d436f72b3ad799c00a48f8724c2b563d469be0e57784fc28ebee6.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/0ed275c2782d436f72b3ad799c00a48f8724c2b563d469be0e57784fc28ebee6.jpg)

![207199890ae8b75748e3df0b9b0f2d6e06fcf2148b850ba7aae7064e734f354c.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/207199890ae8b75748e3df0b9b0f2d6e06fcf2148b850ba7aae7064e734f354c.jpg)

![338c7b59546c37528f04a1e22c24ef6026340afbda7b57c2dabd95102da842a0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/338c7b59546c37528f04a1e22c24ef6026340afbda7b57c2dabd95102da842a0.jpg)

![4d390c9cbe303d304436c94cdadb9172af87cee7e3730c7c80c35ab3d72bdd39.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/4d390c9cbe303d304436c94cdadb9172af87cee7e3730c7c80c35ab3d72bdd39.jpg)

![567b836c723512d53197982a9296d375ab59f25b02c06738a5e9dea52472fb3c.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/567b836c723512d53197982a9296d375ab59f25b02c06738a5e9dea52472fb3c.jpg)

![75c68ed947d825f00ccec6ad2bce419a61e1ceb9658eba071b2d5ba45c8ae8d2.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/75c68ed947d825f00ccec6ad2bce419a61e1ceb9658eba071b2d5ba45c8ae8d2.jpg)

![7879b48ab52dec32238e7743beb7c32306a7b5a22024c92fe1b0457b92aa0f1e.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/7879b48ab52dec32238e7743beb7c32306a7b5a22024c92fe1b0457b92aa0f1e.jpg)

![8ba4b0f74892bbd37fde5409b17bade2b73ae71fc9f600d1aeb93d87d6bafb66.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8ba4b0f74892bbd37fde5409b17bade2b73ae71fc9f600d1aeb93d87d6bafb66.jpg)

![8c93bb7bd636384e69de5603a4d68bc0a218abbb337fb3bd0e4312db9c62c751.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8c93bb7bd636384e69de5603a4d68bc0a218abbb337fb3bd0e4312db9c62c751.jpg)

![8f6cbfb81659ccbe1474fa04c7ab735481c37a2aa34801a0f57140f75794e0d0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8f6cbfb81659ccbe1474fa04c7ab735481c37a2aa34801a0f57140f75794e0d0.jpg)

![bbec98d5889f250066b792e4d043def982fab0496ff01b15f8221e5cecebc0ff.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/bbec98d5889f250066b792e4d043def982fab0496ff01b15f8221e5cecebc0ff.jpg)

![f2d08441ec64541391405bd385b9e347ac99d7098775d665e84175c1874dd3e4.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/f2d08441ec64541391405bd385b9e347ac99d7098775d665e84175c1874dd3e4.jpg)

![fc68999691ffe18f9366eac47b55f2897300858b114c1f80cd32ad93048d1aa3.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/fc68999691ffe18f9366eac47b55f2897300858b114c1f80cd32ad93048d1aa3.jpg)

### Tables

![13902792d787e3bf74f3ca06cf735b105a1444f6ac2f38d8e33858b1a551f002.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/13902792d787e3bf74f3ca06cf735b105a1444f6ac2f38d8e33858b1a551f002.jpg)

![1b604ffa9f84c76f7e3dddd96c4ab90d53ddcdf44c72df4a401d75dcb94251c7.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/1b604ffa9f84c76f7e3dddd96c4ab90d53ddcdf44c72df4a401d75dcb94251c7.jpg)

![1e81e24d52fbda8ba2dac488937fb21fdab53925164051d43ae9aaebfaedcc56.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/1e81e24d52fbda8ba2dac488937fb21fdab53925164051d43ae9aaebfaedcc56.jpg)

![234d4bfdbe5d4ac4f8c3f499eb69c70de73baf60101482bcc49ffb961aac93d4.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/234d4bfdbe5d4ac4f8c3f499eb69c70de73baf60101482bcc49ffb961aac93d4.jpg)

![30dc4cd67aaefe0dd649f07ddc2e6074720ef8951a60d5b8773c5a481a3aad97.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/30dc4cd67aaefe0dd649f07ddc2e6074720ef8951a60d5b8773c5a481a3aad97.jpg)

![46c5b3347d84d6c5382b81e82206f93402e037f3d9f5030e3bcea483f32307bc.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/46c5b3347d84d6c5382b81e82206f93402e037f3d9f5030e3bcea483f32307bc.jpg)

![5eeb089b9db585c1f35bb9cdf59b7bd53a039b17d8f5cf75d4a8fb2bca1ac8dd.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/5eeb089b9db585c1f35bb9cdf59b7bd53a039b17d8f5cf75d4a8fb2bca1ac8dd.jpg)

![81426a7b3e766082c81421716d3b83dc280b87972b29b134261dacc0031adc79.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/81426a7b3e766082c81421716d3b83dc280b87972b29b134261dacc0031adc79.jpg)

![89899bb885df214ec1c66e796b7365d856363417e58eb2b8791dea43967705f5.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/89899bb885df214ec1c66e796b7365d856363417e58eb2b8791dea43967705f5.jpg)

![a98710a731a520cf3580213f8d6d8323f01df3dbc4735a7ec1177f497b5e167d.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/a98710a731a520cf3580213f8d6d8323f01df3dbc4735a7ec1177f497b5e167d.jpg)

![b723436b59de61998ebd3fca6ac2b92a934fe0f0058aada0a158be2bade97a97.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/b723436b59de61998ebd3fca6ac2b92a934fe0f0058aada0a158be2bade97a97.jpg)

![c55d10b9b8b30dd35cf1fbc705ff21944faf247f6482371a067a9880b26ba3f0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/c55d10b9b8b30dd35cf1fbc705ff21944faf247f6482371a067a9880b26ba3f0.jpg)

![d299bc48ecd816ff82ee495f88cfe6dff6193d35ad4eaa469026deba3254ab14.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/d299bc48ecd816ff82ee495f88cfe6dff6193d35ad4eaa469026deba3254ab14.jpg)
