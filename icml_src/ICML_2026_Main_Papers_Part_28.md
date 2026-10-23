# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 28 of 197

## 目录 (Table of Contents)

1. [Conditional Diffusion Sampling](#paper-0892)
2. [Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](#paper-0893)
3. [Geometric Convergence of Gauss–Newton for Neural Networks: Riemannian Geometry and Adaptive Damping](#paper-0894)
4. [PartCo: Part-Level Correspondence Priors Enhance Category Discovery](#paper-0895)
5. [Sparser, Faster, Lighter Transformer Language Models](#paper-0896)
6. [KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](#paper-0897)
7. [Safe and Scalable Web Agent Learning via Recreated Websites](#paper-0898)
8. [Efficient Online Variational Estimation via Monte Carlo Sampling](#paper-0899)
9. [N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout](#paper-0900)
10. [From Diagrams to Code: Multilingual Programming with Visual Design](#paper-0901)
11. [GXPO: Group Cross-Lingual Relative Policy Optimization for Code Generation](#paper-0902)
12. [Probabilistically-routed Bayesian Additive Spanning Trees for Learning on Constrained Domains](#paper-0903)
13. [Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](#paper-0904)
14. [Randomized Feasibility Methods for Constrained Optimization with Adaptive Step Sizes](#paper-0905)
15. [DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning](#paper-0906)
16. [Continual Learning With Participation Privacy: An Auditable Buffering-Aggregation Recipe](#paper-0907)
17. [Curating the Future: A Scalable Recipe for Training Open-Ended Forecasters](#paper-0908)
18. [The Surprising Difficulty of Search in Model-Based Reinforcement Learning](#paper-0909)
19. [MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models](#paper-0910)
20. [ContrastiveCFG: Guiding Diffusion Sampling by Contrasting Positive and Negative Concepts](#paper-0911)
21. [xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](#paper-0912)
22. [How2Everything: Mining the Web for How-to Procedures to Evaluate and Improve LLMs](#paper-0913)
23. [RetrOrchestrator: A Multi-Step Retrosynthesis Agent Dynamically Orchestrating Single-Step Transition Models](#paper-0914)
24. [Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders](#paper-0915)
25. [Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problem](#paper-0916)
26. [SpanNorm: Reconciling Training Stability and Performance in Deep Transformers](#paper-0917)
27. [Doc-to-LoRA: Learning to Instantly Internalize Contexts](#paper-0918)
28. [Tokenised Flow Matching for Hierarchical Simulation Based Inference](#paper-0919)
29. [Faithful Relational Reasoning with Region-based Embeddings: Expressivity of Convex Coordinate-wise Models](#paper-0920)
30. [MetaPerch: Learning from metadata for bioacoustics foundation models](#paper-0921)
31. [Bioacoustic Geolocation: Species Sounds as Geographic Signals](#paper-0922)
32. [One-shot Conditional Sampling: MMD meets Nearest Neighbors](#paper-0923)
33. [Blending Neural Control Density Functions for Stabilization and Safety](#paper-0924)

---


<a id="paper-0892"></a>

## Conditional Diffusion Sampling

### Images

![01f0266d7a93cea4476fad3f6435e48cb30f856911e7151208d913690574a4d5.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/01f0266d7a93cea4476fad3f6435e48cb30f856911e7151208d913690574a4d5.jpg)

![022b3706931af949035feaac83ad3908aac374bed7c45af009b3e9b6e5f153a8.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/022b3706931af949035feaac83ad3908aac374bed7c45af009b3e9b6e5f153a8.jpg)

![785838cba864a3d38054992449131a0a0e66fb22db33dcb0b520782210c4a1df.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/785838cba864a3d38054992449131a0a0e66fb22db33dcb0b520782210c4a1df.jpg)

![8c87cf269b034cb1dd4200f2c737b57ffdde152aec81121767ede800c0aaa195.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/8c87cf269b034cb1dd4200f2c737b57ffdde152aec81121767ede800c0aaa195.jpg)

![a3b365f86c94a4a80917e0c7618499f176e067e927a1b6540f74874f86861609.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/a3b365f86c94a4a80917e0c7618499f176e067e927a1b6540f74874f86861609.jpg)

![b173735f5f58059e825302cd7b364013855460eac27afc0dc06490343aa20944.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/b173735f5f58059e825302cd7b364013855460eac27afc0dc06490343aa20944.jpg)

![ba164b09a162207ec94d44f41735e1b01a41dc14d551290623bffc2fdb1c5e50.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/ba164b09a162207ec94d44f41735e1b01a41dc14d551290623bffc2fdb1c5e50.jpg)

![c9e8ea25cd6d9d0fc5fc758457aa23bff5a2aa9191017f1d7d5cf9a227d2fac9.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/c9e8ea25cd6d9d0fc5fc758457aa23bff5a2aa9191017f1d7d5cf9a227d2fac9.jpg)

![d07115c2dc45915c2bb4dd9414aac771a0cb311d1ca6e6b8d5976b00abf904f5.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/d07115c2dc45915c2bb4dd9414aac771a0cb311d1ca6e6b8d5976b00abf904f5.jpg)

![e51127897387a4729339f7d19300de6405ff32bed4c77be6d12515057651bb22.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/e51127897387a4729339f7d19300de6405ff32bed4c77be6d12515057651bb22.jpg)

![ee5a28f3c56823f24509499b0e7363c0a98ef6e05f05561af4f7dbfe0df02e98.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/images/ee5a28f3c56823f24509499b0e7363c0a98ef6e05f05561af4f7dbfe0df02e98.jpg)

### Tables

![2686c348e8d410bbf1b8bd5f34af02feb64a5a19a54605dbae89000e80e1c26c.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/2686c348e8d410bbf1b8bd5f34af02feb64a5a19a54605dbae89000e80e1c26c.jpg)

![2bd0f83f43c57886925850c43de072fae1b70633eced176f0e11b55e29eb8fd0.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/2bd0f83f43c57886925850c43de072fae1b70633eced176f0e11b55e29eb8fd0.jpg)

![4f1972356661c1a4d96686428763d16a2c00610fca04f5e7a821ae650d23dff5.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/4f1972356661c1a4d96686428763d16a2c00610fca04f5e7a821ae650d23dff5.jpg)

![773b71ad6a06b2855a94c20df2541eda63c252dc47396a56744f396bc7c0740b.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/773b71ad6a06b2855a94c20df2541eda63c252dc47396a56744f396bc7c0740b.jpg)

![942d731bdce36c01fa4bfbfeca2363a7497d114901d7ddfdb95c926725539ae8.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/942d731bdce36c01fa4bfbfeca2363a7497d114901d7ddfdb95c926725539ae8.jpg)

![b6862c1f0aa42841c1afd8bc0570f4ece10164755f7716c2512f9394401087ad.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/b6862c1f0aa42841c1afd8bc0570f4ece10164755f7716c2512f9394401087ad.jpg)

![bf689eb6c5181b405633347fbec36187274b6881f30c86eed935a649c93a3f53.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/bf689eb6c5181b405633347fbec36187274b6881f30c86eed935a649c93a3f53.jpg)

![d9246e6f3f7e508008326aeb1368c0b5800fc583715b0ed611516e2d024bd404.jpg](../icml_results/908_Balanced%20LoRA_%20Removing%20Parameter%20Invariance%20to%20Accelerate%20Convergence/tables/d9246e6f3f7e508008326aeb1368c0b5800fc583715b0ed611516e2d024bd404.jpg)

## Conditional Diffusion Sampling


### Images

![0bead14ce05536877575d9ee61912c700e0c1b86143411ec960bca230e2c4386.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/0bead14ce05536877575d9ee61912c700e0c1b86143411ec960bca230e2c4386.jpg)

![37f472c49e1e3a86a134129e170b77db6e6c31cedbb6614d37d310ab17fb0dc4.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/37f472c49e1e3a86a134129e170b77db6e6c31cedbb6614d37d310ab17fb0dc4.jpg)

![3dd0c17d3122a3d568db4510dbd1763e1fe2afed506aad9363a5404228fd80a4.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/3dd0c17d3122a3d568db4510dbd1763e1fe2afed506aad9363a5404228fd80a4.jpg)

![439e18b81db7fb56706d529ab041f2665d69a30ce674449a1268751ff5b73eff.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/439e18b81db7fb56706d529ab041f2665d69a30ce674449a1268751ff5b73eff.jpg)

![536cc4ffa01b2517128f57dd2016aa3259174d933ca97424b14eb2fe6d9e81e0.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/536cc4ffa01b2517128f57dd2016aa3259174d933ca97424b14eb2fe6d9e81e0.jpg)

![56f700f9fcb20a2b1d650a1567f89fd904d93562e2ab0724c923ea0529092beb.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/56f700f9fcb20a2b1d650a1567f89fd904d93562e2ab0724c923ea0529092beb.jpg)

![6345adac53f8e6456d5c0ba48765d9eb599cb87bbd64f35e098c49dff155ea42.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/6345adac53f8e6456d5c0ba48765d9eb599cb87bbd64f35e098c49dff155ea42.jpg)

![8133e4901b093ccbca3e9ee01614f2069600894c92d286cf3c790236a32ffab0.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/8133e4901b093ccbca3e9ee01614f2069600894c92d286cf3c790236a32ffab0.jpg)

![8d736a87860b63d0484d1fc0c4301474b7b9cf25e8661692ae9bf35b3af26407.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/8d736a87860b63d0484d1fc0c4301474b7b9cf25e8661692ae9bf35b3af26407.jpg)

![8f0fc457e513f204cad8af15ef6139f1c27dde3d73c411057c0818462542a587.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/8f0fc457e513f204cad8af15ef6139f1c27dde3d73c411057c0818462542a587.jpg)

![902de0db68d53d4903fdc1057043680d9bd62b38f9b0218beb97df68b062b14a.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/902de0db68d53d4903fdc1057043680d9bd62b38f9b0218beb97df68b062b14a.jpg)

![ae82c8f24e7eaab71e3c8288efb9425917f9c83efe2fb3272e71205b54ff023e.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/ae82c8f24e7eaab71e3c8288efb9425917f9c83efe2fb3272e71205b54ff023e.jpg)

![b9af48f268fa25fce61f81131ab4e3cd130ca0a303688f295c7684d511e758a3.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/b9af48f268fa25fce61f81131ab4e3cd130ca0a303688f295c7684d511e758a3.jpg)

![c205f3469424d6754a8f83da6419dc37f1990e3ab38f8c136ad94dadc75a5145.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/c205f3469424d6754a8f83da6419dc37f1990e3ab38f8c136ad94dadc75a5145.jpg)

![c22ea5a2a690bab19134369ddf77b2c07ecebf3746295786624704764220aa63.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/c22ea5a2a690bab19134369ddf77b2c07ecebf3746295786624704764220aa63.jpg)

![ca2de1410b98de80cfa08bf5ae9512079ef270596376ff80d2aaef081a86d297.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/ca2de1410b98de80cfa08bf5ae9512079ef270596376ff80d2aaef081a86d297.jpg)

![f0f1bae497f524c4a5f44ee1867f9c5d1c379d5ac24adac2f302d4adf79ec393.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/f0f1bae497f524c4a5f44ee1867f9c5d1c379d5ac24adac2f302d4adf79ec393.jpg)

![f5bdc43126abec44d84dac903b938b8b322d3de3fce782c6dfec2d2450cd27fb.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/f5bdc43126abec44d84dac903b938b8b322d3de3fce782c6dfec2d2450cd27fb.jpg)

![fe6e92fdb72ee148a5766dce392e17710e5e34e4fe2741c025b9dd751f0240a2.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/images/fe6e92fdb72ee148a5766dce392e17710e5e34e4fe2741c025b9dd751f0240a2.jpg)

### Tables

![57f45fa69bec3b30fb6218654b8d4032262f953f49eb60ffc516541bc88bb76f.jpg](../icml_results/909_Conditional%20Diffusion%20Sampling/tables/57f45fa69bec3b30fb6218654b8d4032262f953f49eb60ffc516541bc88bb76f.jpg)

<a id="paper-0893"></a>

## Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time


### Images

![55b4f20b764beffedd4f396c737b8ccda3e3f520271d810c845e1230469b1d33.jpg](../icml_results/910_Bregman%20meets%20L%C3%A9vy_%20Stochastic%20Mirror%20Descent%20with%20Heavy-Tailed%20Noise%20in%20Continuous%20and%20Discrete%20Tim/images/55b4f20b764beffedd4f396c737b8ccda3e3f520271d810c845e1230469b1d33.jpg)

<a id="paper-0894"></a>

## Geometric Convergence of Gauss–Newton for Neural Networks: Riemannian Geometry and Adaptive Damping


### Images

![2e38e0cc10e24b7fff4e658e4c5355941ae8d7ae88876550e1c2f4c4878c1169.jpg](../icml_results/911_Geometric%20Convergence%20of%20Gauss%E2%80%93Newton%20for%20Neural%20Networks_%20Riemannian%20Geometry%20and%20Adaptive%20Damping/images/2e38e0cc10e24b7fff4e658e4c5355941ae8d7ae88876550e1c2f4c4878c1169.jpg)

![5b2d6f37a6a91c7332fcfc38d4b1c31701d26fe58fba2b3c55f6a36ac0f7c280.jpg](../icml_results/911_Geometric%20Convergence%20of%20Gauss%E2%80%93Newton%20for%20Neural%20Networks_%20Riemannian%20Geometry%20and%20Adaptive%20Damping/images/5b2d6f37a6a91c7332fcfc38d4b1c31701d26fe58fba2b3c55f6a36ac0f7c280.jpg)

![aa67c8b2bf1e82486009d73aad01ceb4f076e8f572983cb6f94979fe1017fef5.jpg](../icml_results/911_Geometric%20Convergence%20of%20Gauss%E2%80%93Newton%20for%20Neural%20Networks_%20Riemannian%20Geometry%20and%20Adaptive%20Damping/images/aa67c8b2bf1e82486009d73aad01ceb4f076e8f572983cb6f94979fe1017fef5.jpg)

<a id="paper-0895"></a>

## PartCo: Part-Level Correspondence Priors Enhance Category Discovery


### Images

![02902a020455bda332f3e5738fe797c9114155d9167e716ad689fa0ca1efbbc7.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/02902a020455bda332f3e5738fe797c9114155d9167e716ad689fa0ca1efbbc7.jpg)

![1a84d1ecc43351ec7eb1e9eb14bd0eb32376ed22a248bf73ca652077bcfd25dd.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/1a84d1ecc43351ec7eb1e9eb14bd0eb32376ed22a248bf73ca652077bcfd25dd.jpg)

![2606f581d8b420c40187a5dc46a83c1cb0a0c452187abd3f585fb204cde5bb73.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/2606f581d8b420c40187a5dc46a83c1cb0a0c452187abd3f585fb204cde5bb73.jpg)

![36b00f1e070daa84bfe79e817be320a761291d21a75ecc7c1f83962b65ceb9d5.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/36b00f1e070daa84bfe79e817be320a761291d21a75ecc7c1f83962b65ceb9d5.jpg)

![456b5027786b301df9fa026b2466f758396161b140984282992b13d4f0facd16.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/456b5027786b301df9fa026b2466f758396161b140984282992b13d4f0facd16.jpg)

![64355feea7a0a21e3c0174a12e56bb7fe61eb191cdd0638c358a4eda7371e79b.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/64355feea7a0a21e3c0174a12e56bb7fe61eb191cdd0638c358a4eda7371e79b.jpg)

![82705909c4ed7191de9a003e36552ee488a0044ccf9582bc9c9da835242669dd.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/82705909c4ed7191de9a003e36552ee488a0044ccf9582bc9c9da835242669dd.jpg)

![96c56ddab3195267624d5c5c77b67fae5ca0a83d2943a2abcb88dbe322f8a07c.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/96c56ddab3195267624d5c5c77b67fae5ca0a83d2943a2abcb88dbe322f8a07c.jpg)

![a43c73b2ded71f3d19122054db275b2772b9c4518060f495243633db6f539943.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/a43c73b2ded71f3d19122054db275b2772b9c4518060f495243633db6f539943.jpg)

![a78830d756a5d8e6ae0e911294af316ab219e74a73774fc294e997bd6bce44d6.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/a78830d756a5d8e6ae0e911294af316ab219e74a73774fc294e997bd6bce44d6.jpg)

![b99d22fd025ba1960e47f7101994ad28fbe9f72e6b5fb8d98ad42b354b0252bf.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/b99d22fd025ba1960e47f7101994ad28fbe9f72e6b5fb8d98ad42b354b0252bf.jpg)

![dcaa2bc1cac3f49a16e21d08960ce443768ff6b27e0cc66e99693ee34c198ae3.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/dcaa2bc1cac3f49a16e21d08960ce443768ff6b27e0cc66e99693ee34c198ae3.jpg)

![e665c9806f60d9ede2b5a751d5e6ee5b3b798fe2451cb1822364493a82af6485.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/e665c9806f60d9ede2b5a751d5e6ee5b3b798fe2451cb1822364493a82af6485.jpg)

![f19d3e4b1269aed0812faaa9acdbe9ba942c099dec0804644442c078a55959f8.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/f19d3e4b1269aed0812faaa9acdbe9ba942c099dec0804644442c078a55959f8.jpg)

![f669dcbd4335f697195ae25c31ec6ba1a8f013b7f55e8df0f07699ad78a0b6d8.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/images/f669dcbd4335f697195ae25c31ec6ba1a8f013b7f55e8df0f07699ad78a0b6d8.jpg)

### Tables

![0a3569a3505866ef82063b7803f6c6e99802b5182f30dcbf4f82f8f22a6896eb.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/0a3569a3505866ef82063b7803f6c6e99802b5182f30dcbf4f82f8f22a6896eb.jpg)

![28a706c4d8e5bed2864326f888aaba6708f32cac186ed19291f0a2455645aea6.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/28a706c4d8e5bed2864326f888aaba6708f32cac186ed19291f0a2455645aea6.jpg)

![38efeec8e13b8a2e7f2c182d92a8115e98202f6845aaf1d0c62dac1d0e912867.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/38efeec8e13b8a2e7f2c182d92a8115e98202f6845aaf1d0c62dac1d0e912867.jpg)

![6de457d9a0a90829bc6ccb8b143106ec5d76e1f25d0f93f9543faec93aec960f.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/6de457d9a0a90829bc6ccb8b143106ec5d76e1f25d0f93f9543faec93aec960f.jpg)

![75328769b3c9f17f3d4b0d34f426f8eb491aba28c018ddc24bc74a44dd1b34fb.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/75328769b3c9f17f3d4b0d34f426f8eb491aba28c018ddc24bc74a44dd1b34fb.jpg)

![8613903efe18d9cddab647e28cc167a4780e91c77de2a72124774663d6c71bcf.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/8613903efe18d9cddab647e28cc167a4780e91c77de2a72124774663d6c71bcf.jpg)

![8d4e3098dbdee6229813b83024878cea90518eb25c0475686a192102f1e6e8fe.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/8d4e3098dbdee6229813b83024878cea90518eb25c0475686a192102f1e6e8fe.jpg)

![8f5e772cbabd3a5712382006ce93ae7dd7cfa1cdf2d91a9e70a972f156779c9e.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/8f5e772cbabd3a5712382006ce93ae7dd7cfa1cdf2d91a9e70a972f156779c9e.jpg)

![abb9fbd94b6615d5b1362ba408a7db5ebc3866a3c88377bb413f27ed0fbbd72b.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/abb9fbd94b6615d5b1362ba408a7db5ebc3866a3c88377bb413f27ed0fbbd72b.jpg)

![adacbdf2406d14b7a095f2dbf7c29314b737709a5c2d1fc018a24bfabacbc38b.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/adacbdf2406d14b7a095f2dbf7c29314b737709a5c2d1fc018a24bfabacbc38b.jpg)

![ae76c20f4037ef5bc040c1467d7099223f935be5764c350b508bae18f6af7b41.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/ae76c20f4037ef5bc040c1467d7099223f935be5764c350b508bae18f6af7b41.jpg)

![b1b00b22da07405312f052b1a2cc2475bd7563404fc4ae501e4718788bc18cbd.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/b1b00b22da07405312f052b1a2cc2475bd7563404fc4ae501e4718788bc18cbd.jpg)

![b975112c21cd27c387936913375731bca5a0161825c8a0e04b5c488c49c5de09.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/b975112c21cd27c387936913375731bca5a0161825c8a0e04b5c488c49c5de09.jpg)

![bf6f59d2ebeedb46725876ca71590d97d98d608d3ad21f7bbb65669f317990f0.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/bf6f59d2ebeedb46725876ca71590d97d98d608d3ad21f7bbb65669f317990f0.jpg)

![e6004512a55e9b7f8af2601e32163e327b15fee5d8032f9e398d84b32a454522.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/e6004512a55e9b7f8af2601e32163e327b15fee5d8032f9e398d84b32a454522.jpg)

![f714c78a266daab0c9999ea87c976575e4a5c0fd3789d28d9f783771fca09571.jpg](../icml_results/912_PartCo_%20Part-Level%20Correspondence%20Priors%20Enhance%20Category%20Discovery/tables/f714c78a266daab0c9999ea87c976575e4a5c0fd3789d28d9f783771fca09571.jpg)

<a id="paper-0896"></a>

## Sparser, Faster, Lighter Transformer Language Models


### Images

![03b131775d90bfd05cdb2736ac97cb93595241dc87d8c19a88e543cbc492cddc.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/03b131775d90bfd05cdb2736ac97cb93595241dc87d8c19a88e543cbc492cddc.jpg)

![083b8d5a1ffee7ae8f47a6f8587cb56800bfdaab79fcbaebcbb451513dc80d2e.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/083b8d5a1ffee7ae8f47a6f8587cb56800bfdaab79fcbaebcbb451513dc80d2e.jpg)

![11fa8c233cc18147a356e54567c9ffedc50e643d490c3448c39713d66e12f0d5.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/11fa8c233cc18147a356e54567c9ffedc50e643d490c3448c39713d66e12f0d5.jpg)

![3e843a6a7d005e65f6d9b78ae89710b7f0fabbf9ff7c2749c40d22ab467fcfa3.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/3e843a6a7d005e65f6d9b78ae89710b7f0fabbf9ff7c2749c40d22ab467fcfa3.jpg)

![42bea9474290b8fdfd05202395e1a6a75d11fc250e8f0a4fcd8e21bbfc2d8cd2.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/42bea9474290b8fdfd05202395e1a6a75d11fc250e8f0a4fcd8e21bbfc2d8cd2.jpg)

![64020d86d0927ba431ad22443aa0a69dcb3cf3eb6e806eefbc39a2186ba0ac81.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/64020d86d0927ba431ad22443aa0a69dcb3cf3eb6e806eefbc39a2186ba0ac81.jpg)

![6ab9b91770ed0a11b0d1d6e7e6b613c364566e42ba38f1246745410ac68d8ba0.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/6ab9b91770ed0a11b0d1d6e7e6b613c364566e42ba38f1246745410ac68d8ba0.jpg)

![6bf57674393856bc8fb640e82d1c0b124b51cb9b6f44cfffa6cff056ddccb7cb.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/6bf57674393856bc8fb640e82d1c0b124b51cb9b6f44cfffa6cff056ddccb7cb.jpg)

![791532ec9025f07cde557667ef52950d94a1f2a2c97f167dc2043722e696f172.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/791532ec9025f07cde557667ef52950d94a1f2a2c97f167dc2043722e696f172.jpg)

![795095100189a824e49d2347544866be6772eef152768c72d8b1fef1871649dd.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/795095100189a824e49d2347544866be6772eef152768c72d8b1fef1871649dd.jpg)

![7ea73eaf597643ca73e1ac29244ed328c4c6b77bc8e7a36ca10ec3f5b67286b1.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/7ea73eaf597643ca73e1ac29244ed328c4c6b77bc8e7a36ca10ec3f5b67286b1.jpg)

![82c1865d3c14b1ccebab71942c2c849128c7cb1aff195b83c9f8875feb222862.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/82c1865d3c14b1ccebab71942c2c849128c7cb1aff195b83c9f8875feb222862.jpg)

![886c1ae01e3b3a2fd967530a4d8430b991b09396472c6c4ed73fc8e816542841.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/886c1ae01e3b3a2fd967530a4d8430b991b09396472c6c4ed73fc8e816542841.jpg)

![b3320cedc8fe221108aa4b305bdae6dc43ffcc4949fa14eed460e6db9175d066.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/b3320cedc8fe221108aa4b305bdae6dc43ffcc4949fa14eed460e6db9175d066.jpg)

![b943dd0b1858c61c2745f93087a48642aaaca85ec6e28e0d7f18fa7991d74aff.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/b943dd0b1858c61c2745f93087a48642aaaca85ec6e28e0d7f18fa7991d74aff.jpg)

![c06ab95fe1cac54680f86b8448b5b236ecdb57281f6327592bedca3b981f1692.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/c06ab95fe1cac54680f86b8448b5b236ecdb57281f6327592bedca3b981f1692.jpg)

![dd273b81c8d7b1ade7e35aeac1f97c65efe90975f260205e438dd59a9e6424da.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/dd273b81c8d7b1ade7e35aeac1f97c65efe90975f260205e438dd59a9e6424da.jpg)

![e0b859e3aa648d911422489233f1a3c08f0b814539cb343118c5f1cd7dcedce7.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/e0b859e3aa648d911422489233f1a3c08f0b814539cb343118c5f1cd7dcedce7.jpg)

![e1907fa656cb99c8b82c4d91a6d73a902ac7aa69c7d06169a9e1216094a20dba.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/images/e1907fa656cb99c8b82c4d91a6d73a902ac7aa69c7d06169a9e1216094a20dba.jpg)

### Tables

![0a878e89e28b91f149e85224784d682b083f51df398960a0a90bcc377e3b07a1.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/0a878e89e28b91f149e85224784d682b083f51df398960a0a90bcc377e3b07a1.jpg)

![144d6a7435a32ebff093b1a23aa923f818e9a9a34af33d9b1b88fd5fff189930.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/144d6a7435a32ebff093b1a23aa923f818e9a9a34af33d9b1b88fd5fff189930.jpg)

![15f12a50ae18c2571237c53843313df7926ea3ddc4d379ce1e03bae3dab60e90.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/15f12a50ae18c2571237c53843313df7926ea3ddc4d379ce1e03bae3dab60e90.jpg)

![1b05b910bd48babdfb51389e0aadc3ae8ac77ebdd2d8d7d3583ced7f86e10990.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/1b05b910bd48babdfb51389e0aadc3ae8ac77ebdd2d8d7d3583ced7f86e10990.jpg)

![302ffaa912f871fa19d6097428fdf51abd00b9018be9655f4335e542c3eab37f.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/302ffaa912f871fa19d6097428fdf51abd00b9018be9655f4335e542c3eab37f.jpg)

![34d0065d4c60d4ce02b86fb46fb59b98186b6dcc5267233b0a35ac0199ba6c14.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/34d0065d4c60d4ce02b86fb46fb59b98186b6dcc5267233b0a35ac0199ba6c14.jpg)

![385541e81e1720900014c71e9043fd1c9a33a572a36c955e788bbcc94ad9b254.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/385541e81e1720900014c71e9043fd1c9a33a572a36c955e788bbcc94ad9b254.jpg)

![559ebae6c71e0cb21a4678837344612671e5e22a9af40c349ff0d994476d7b64.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/559ebae6c71e0cb21a4678837344612671e5e22a9af40c349ff0d994476d7b64.jpg)

![9cca3cae0fc33823d1d83cc4ff991c816e88926312b2201d06ab06aa5662c3a5.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/9cca3cae0fc33823d1d83cc4ff991c816e88926312b2201d06ab06aa5662c3a5.jpg)

![a7800c6e166dbdac2dd0f97cf8d79f908b0371a8fe9aaef334e0b3a80e8ff875.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/a7800c6e166dbdac2dd0f97cf8d79f908b0371a8fe9aaef334e0b3a80e8ff875.jpg)

![f9589afde247b519ecc34a43d814af70b2e0288e880475854719691f43e37cdb.jpg](../icml_results/913_Sparser%2C%20Faster%2C%20Lighter%20Transformer%20Language%20Models/tables/f9589afde247b519ecc34a43d814af70b2e0288e880475854719691f43e37cdb.jpg)

<a id="paper-0897"></a>

## KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem


### Images

![00c14542a98c1d9dee7d9aef98fe3d79f6250cadda343145ca975f570432e935.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/00c14542a98c1d9dee7d9aef98fe3d79f6250cadda343145ca975f570432e935.jpg)

![8d5585fbcb014bdb946a612bda40c6d50b201f0faa4c1bae077cd4d77f286006.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/8d5585fbcb014bdb946a612bda40c6d50b201f0faa4c1bae077cd4d77f286006.jpg)

![9b56fc8fd50708e8fafd866564f495f7c53d6feee071a0f157c1cbfb86b99dbf.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/9b56fc8fd50708e8fafd866564f495f7c53d6feee071a0f157c1cbfb86b99dbf.jpg)

![ecf741a25fca95a2ac36738c8e1676e7a94ccda217c3748f1db7bc1e039b1662.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/ecf741a25fca95a2ac36738c8e1676e7a94ccda217c3748f1db7bc1e039b1662.jpg)

![ee9278d7dde831e6f57b192b231b551cd3d5b22cccbfeceded3010411acc8687.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/ee9278d7dde831e6f57b192b231b551cd3d5b22cccbfeceded3010411acc8687.jpg)

![efae83bce5b661da4f5c2d736692e32f275e27552131928b9b9e341f7281cc43.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/efae83bce5b661da4f5c2d736692e32f275e27552131928b9b9e341f7281cc43.jpg)

![f153680a0dffe1baa68ebfe4ad6beed52126432c5502efc32af100081dcfc688.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/images/f153680a0dffe1baa68ebfe4ad6beed52126432c5502efc32af100081dcfc688.jpg)

### Tables

![0eccc83ac35b1d19ab11bd7f0b72c9cc862855bebb2c8e08fa28c1fa08d7342b.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/tables/0eccc83ac35b1d19ab11bd7f0b72c9cc862855bebb2c8e08fa28c1fa08d7342b.jpg)

![23af8dd19a3833692f70231f131ccfa2bd1fa3fa3a95856810da554cd440b848.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/tables/23af8dd19a3833692f70231f131ccfa2bd1fa3fa3a95856810da554cd440b848.jpg)

![c559321fbc7f6e6fe297632c2bb41d110291c3e9b3a0bf5324c5a3044bcb9e7e.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/tables/c559321fbc7f6e6fe297632c2bb41d110291c3e9b3a0bf5324c5a3044bcb9e7e.jpg)

![e8d948968e7e268e927a04711f99ab3f7c1ade55d466ef4967815bc97f0ad1af.jpg](../icml_results/914_KnapSpec_%20Self-Speculative%20Decoding%20via%20Adaptive%20Layer%20Selection%20as%20a%20Knapsack%20Problem/tables/e8d948968e7e268e927a04711f99ab3f7c1ade55d466ef4967815bc97f0ad1af.jpg)

<a id="paper-0898"></a>

## Safe and Scalable Web Agent Learning via Recreated Websites


### Images

![00690a0994b92ac64c5ed1dc5671910a6243d96aff38d0ff4c6d8dd4f762e457.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/00690a0994b92ac64c5ed1dc5671910a6243d96aff38d0ff4c6d8dd4f762e457.jpg)

![02aa0b46124014c01fcbc56e0b287b1674d40d9371b6da5060bae69399adb023.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/02aa0b46124014c01fcbc56e0b287b1674d40d9371b6da5060bae69399adb023.jpg)

![062a2264530d419800f2ec25195d7291fd58e8a08a316b298ff28b3def2f86c9.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/062a2264530d419800f2ec25195d7291fd58e8a08a316b298ff28b3def2f86c9.jpg)

![09b0a083e63b3b14022cd221b0e215e4cef9d72c1d0a2f952b1ec719a6aa6ccc.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/09b0a083e63b3b14022cd221b0e215e4cef9d72c1d0a2f952b1ec719a6aa6ccc.jpg)

![0f62871eff77a0498cf2aebd50830764f8a2b3f6343736182cde1eefe18b2ff0.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/0f62871eff77a0498cf2aebd50830764f8a2b3f6343736182cde1eefe18b2ff0.jpg)

![106170c93fe858b11b214d7df17026e5529a3e216f6b14e76133002301bb3d23.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/106170c93fe858b11b214d7df17026e5529a3e216f6b14e76133002301bb3d23.jpg)

![1b096bb9b67447da401d7c7c696ca76f901f9631eadf18d3215dd892adf3a8b0.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/1b096bb9b67447da401d7c7c696ca76f901f9631eadf18d3215dd892adf3a8b0.jpg)

![1e6742f098e61173290c17bece1835bc2c6d967a7bd4a8794823c3607c2315e9.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/1e6742f098e61173290c17bece1835bc2c6d967a7bd4a8794823c3607c2315e9.jpg)

![1ec2b9ca8aafe05e8847c8de4605aab30dcff5284ebe245b47482576b95760dd.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/1ec2b9ca8aafe05e8847c8de4605aab30dcff5284ebe245b47482576b95760dd.jpg)

![289e7c616758baa8f58e45f615790383e553a128d0141e05a54a0c130b450190.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/289e7c616758baa8f58e45f615790383e553a128d0141e05a54a0c130b450190.jpg)

![293d9d6f7d0fb5ed303010734439d116b9bcb3dd0d001c162ea727cf5dea8806.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/293d9d6f7d0fb5ed303010734439d116b9bcb3dd0d001c162ea727cf5dea8806.jpg)

![2ef93c04c9f4c82a371f42d2c58e3be3602e2e735c8f82d3018ddcb6ba11115f.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/2ef93c04c9f4c82a371f42d2c58e3be3602e2e735c8f82d3018ddcb6ba11115f.jpg)

![324b14cae380874040160924853be3860e910059caf6320746b7ae947ba6f274.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/324b14cae380874040160924853be3860e910059caf6320746b7ae947ba6f274.jpg)

![348b6f580dd015e6ac67ceefbb8703a4f58650f1ce8d190f2f7c45bccffd50be.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/348b6f580dd015e6ac67ceefbb8703a4f58650f1ce8d190f2f7c45bccffd50be.jpg)

![4018a2397050c40be7103dfbba7d1946643b7b3094e86b1276ed6eca1227d6ee.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/4018a2397050c40be7103dfbba7d1946643b7b3094e86b1276ed6eca1227d6ee.jpg)

![4a6eda82dd21cc2b59da4eae3f6d49a36cda2f6eb5fad95339364a07f506b705.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/4a6eda82dd21cc2b59da4eae3f6d49a36cda2f6eb5fad95339364a07f506b705.jpg)

![4b83961e9bb97893fe16ca9adc4e99ed74f946d04210a6a7412d177bb1ace489.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/4b83961e9bb97893fe16ca9adc4e99ed74f946d04210a6a7412d177bb1ace489.jpg)

![4c736494df60e266448e9bd35cbc0b272a075d75919e2a5ba581f996b17604c9.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/4c736494df60e266448e9bd35cbc0b272a075d75919e2a5ba581f996b17604c9.jpg)

![66c386bf382ce71ad40c9b545c3d50ea9cfc1225382d7ccf900f2c44359c8388.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/66c386bf382ce71ad40c9b545c3d50ea9cfc1225382d7ccf900f2c44359c8388.jpg)

![813b147852d5ddeda349cc889e906b2f04e23b1c770886ebd87d3d1ea5ffd1e5.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/813b147852d5ddeda349cc889e906b2f04e23b1c770886ebd87d3d1ea5ffd1e5.jpg)

![aa9d9674ef9b799eec4abd90ddfed0c7d4dbf42049eb554daa23875abe245ad6.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/aa9d9674ef9b799eec4abd90ddfed0c7d4dbf42049eb554daa23875abe245ad6.jpg)

![c030f762d943e8ced1fb064e88338fa47303064a9a2e8148f26d9b90013c3cb2.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/c030f762d943e8ced1fb064e88338fa47303064a9a2e8148f26d9b90013c3cb2.jpg)

![d4d9ab0bd652816a788c1764db607bcc4abab788797e0bf61298b3e8bf0965a6.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/d4d9ab0bd652816a788c1764db607bcc4abab788797e0bf61298b3e8bf0965a6.jpg)

![da246e17033797cdb4abe643850d1e75ced2b6fa52a879674f26bab91a6c3c8f.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/da246e17033797cdb4abe643850d1e75ced2b6fa52a879674f26bab91a6c3c8f.jpg)

![ec002bdbc390d7531c73be1b509458f2aab63f5045e9c5b3731a3693c5d6f0c8.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/ec002bdbc390d7531c73be1b509458f2aab63f5045e9c5b3731a3693c5d6f0c8.jpg)

![efec332f4f02b441e47eae4860c29fd8c02718014f85429e15d5bfd815deb9cd.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/images/efec332f4f02b441e47eae4860c29fd8c02718014f85429e15d5bfd815deb9cd.jpg)

### Tables

![17536007d3222e4bd6afec900849dfe6d7631a5863579199a7980c9742ca3c7e.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/17536007d3222e4bd6afec900849dfe6d7631a5863579199a7980c9742ca3c7e.jpg)

![220d9bf663a36de241ffe41301cdbeb36bd87c8de99fb86fbde7418d16615272.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/220d9bf663a36de241ffe41301cdbeb36bd87c8de99fb86fbde7418d16615272.jpg)

![335003a70406a60127a04ffe06a56cf77d34a4c85428f394dda122d85a6bfd35.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/335003a70406a60127a04ffe06a56cf77d34a4c85428f394dda122d85a6bfd35.jpg)

![36f09722010b41492b01cf3f006c2b51297b9a655bf59d7b22d4864416b8a47f.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/36f09722010b41492b01cf3f006c2b51297b9a655bf59d7b22d4864416b8a47f.jpg)

![3ac3e82d6b1b2f2b1a0b75eb72f1a7e5a082b03449d154cc263a7cba103706b0.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/3ac3e82d6b1b2f2b1a0b75eb72f1a7e5a082b03449d154cc263a7cba103706b0.jpg)

![3e49bf7cec7bf3bdea82df7c59922830dafb3e1411401df340460087c36e8e42.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/3e49bf7cec7bf3bdea82df7c59922830dafb3e1411401df340460087c36e8e42.jpg)

![49d022ac2fc80c14cf6568a3713a3630e0bf7826c37379aa7a603202436eaef7.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/49d022ac2fc80c14cf6568a3713a3630e0bf7826c37379aa7a603202436eaef7.jpg)

![747724b2bef281d16393f828638340558eb9a7aaa79d86ad3e12ad33a3e4bdf9.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/747724b2bef281d16393f828638340558eb9a7aaa79d86ad3e12ad33a3e4bdf9.jpg)

![7acf2bb7bec42799ba79d0dce059bdfe696dd6054080bdb635194ead895e1fe7.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/7acf2bb7bec42799ba79d0dce059bdfe696dd6054080bdb635194ead895e1fe7.jpg)

![c7d264e3f255a8a3f2c2b1f703ea1629b43d7363adf80632b8bf51e60bf51bd0.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/c7d264e3f255a8a3f2c2b1f703ea1629b43d7363adf80632b8bf51e60bf51bd0.jpg)

![ce68d28667f21216a32e8d09c4c309fe7d357453a0df4933de3d3680ebd16c74.jpg](../icml_results/915_Safe%20and%20Scalable%20Web%20Agent%20Learning%20via%20Recreated%20Websites/tables/ce68d28667f21216a32e8d09c4c309fe7d357453a0df4933de3d3680ebd16c74.jpg)

<a id="paper-0899"></a>

## Efficient Online Variational Estimation via Monte Carlo Sampling


### Images

![03b02fac0d33670d314454030b05bce438295afefbeb5fac8dd181f2b5d8a26f.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/03b02fac0d33670d314454030b05bce438295afefbeb5fac8dd181f2b5d8a26f.jpg)

![1524c955bc982d8b38ace6e8e7610a57cd7ff0d89d9b413922e9a274a3baf045.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/1524c955bc982d8b38ace6e8e7610a57cd7ff0d89d9b413922e9a274a3baf045.jpg)

![454205c61e788a0ed088ec875e4cfe79b2ae5babc32e3cc4f25f6a323ed36b5c.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/454205c61e788a0ed088ec875e4cfe79b2ae5babc32e3cc4f25f6a323ed36b5c.jpg)

![97c2d05754fbf0d83148ce2fec14c4a1ecda76692c58557a0f26c963c4486164.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/97c2d05754fbf0d83148ce2fec14c4a1ecda76692c58557a0f26c963c4486164.jpg)

![9cee93066ec858854dbf652af50e4bee66de1322df175cc0658df8928e08eb5e.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/9cee93066ec858854dbf652af50e4bee66de1322df175cc0658df8928e08eb5e.jpg)

![ccc41b4d3df077d9c96a2b4b45ffc06ca6713fcf87bbd9e58d2aad1619ace701.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/ccc41b4d3df077d9c96a2b4b45ffc06ca6713fcf87bbd9e58d2aad1619ace701.jpg)

![d3a5e29f3242c070006a4443403608ce0e7cd900c4ba75262ad9a6b663580da4.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/d3a5e29f3242c070006a4443403608ce0e7cd900c4ba75262ad9a6b663580da4.jpg)

![dcfb95207bd424d38f196f9bc6db2ab27f689b2d184456774306bb0eddb6497f.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/dcfb95207bd424d38f196f9bc6db2ab27f689b2d184456774306bb0eddb6497f.jpg)

![e56e09feffb619e7fe2efb30bffed3018470c73b70dedcada1d654b80ff0693b.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/e56e09feffb619e7fe2efb30bffed3018470c73b70dedcada1d654b80ff0693b.jpg)

![ef8c5e8ef67d4471c59702f9b5def1ad9d0a5c146bf3f6105912f8719eabb8f9.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/images/ef8c5e8ef67d4471c59702f9b5def1ad9d0a5c146bf3f6105912f8719eabb8f9.jpg)

### Tables

![3c994e70281e36656a8814e15c386a1f943b392592731b55057ee5a5fbdf6a98.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/tables/3c994e70281e36656a8814e15c386a1f943b392592731b55057ee5a5fbdf6a98.jpg)

![4ff6cac3a03de2614b954cba95aa06e226875ec045027bf031513740e5bc7655.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/tables/4ff6cac3a03de2614b954cba95aa06e226875ec045027bf031513740e5bc7655.jpg)

![8e49aff489641c2d86b2aa7d30f33a12f1750515733328b91ef28e2fb29ada47.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/tables/8e49aff489641c2d86b2aa7d30f33a12f1750515733328b91ef28e2fb29ada47.jpg)

![b242d3c87e50b95222e05171b707d64684a8c837c0137d07c04755699ef885ec.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/tables/b242d3c87e50b95222e05171b707d64684a8c837c0137d07c04755699ef885ec.jpg)

![e3c88bbe679b4b8d3a93fb0c95264814c8ffa319abeb090a83faf8aaa7d39046.jpg](../icml_results/916_Efficient%20Online%20Variational%20Estimation%20via%20Monte%20Carlo%20Sampling/tables/e3c88bbe679b4b8d3a93fb0c95264814c8ffa319abeb090a83faf8aaa7d39046.jpg)

<a id="paper-0900"></a>

## N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout


### Images

![02853ba68b175ea17b1b02b8058d8d4ec6fd428a8fd762b87d7f65aebc8334ae.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/02853ba68b175ea17b1b02b8058d8d4ec6fd428a8fd762b87d7f65aebc8334ae.jpg)

![1e8e0688bc146d252911c82b8a7cabd9f618e092f32333a4e16b60ab833cf127.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/1e8e0688bc146d252911c82b8a7cabd9f618e092f32333a4e16b60ab833cf127.jpg)

![238cfe56b1406c934dc435836b4e7c6ad2df9eb9880084320ac6d2d674d09281.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/238cfe56b1406c934dc435836b4e7c6ad2df9eb9880084320ac6d2d674d09281.jpg)

![2b5664d987be741f31ce06c05211e01df4006d84995edc986dddb1ca4a8ef315.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/2b5664d987be741f31ce06c05211e01df4006d84995edc986dddb1ca4a8ef315.jpg)

![372be20ddeca2b268bf7dca984faaed8282d07061ccef994bb91699fbfbd6b56.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/372be20ddeca2b268bf7dca984faaed8282d07061ccef994bb91699fbfbd6b56.jpg)

![397922eb2e2e3567d37babc84589520aee6577442699a3f7c794be0a8f416c5d.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/397922eb2e2e3567d37babc84589520aee6577442699a3f7c794be0a8f416c5d.jpg)

![7435ed898354e9bc09feddf54d988a1ca03fdc764d39b41a474ef88281402179.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/7435ed898354e9bc09feddf54d988a1ca03fdc764d39b41a474ef88281402179.jpg)

![75bfd7e090c20881d7519252d91735ce42a96198634c04dbc206c2204987cdcf.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/75bfd7e090c20881d7519252d91735ce42a96198634c04dbc206c2204987cdcf.jpg)

![803a9382cdbbf8129f541767cbce5be9002566a22e693449843c7bf2f6f16e51.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/803a9382cdbbf8129f541767cbce5be9002566a22e693449843c7bf2f6f16e51.jpg)

![804b59577da554fa2cb8676ba7ce1304ab9b9876fb7fa3d02498848db5251028.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/804b59577da554fa2cb8676ba7ce1304ab9b9876fb7fa3d02498848db5251028.jpg)

![9747888eb2fc9c75bcbd02ad7554104074aa7294fae37fcec35fac853bb33fbd.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/9747888eb2fc9c75bcbd02ad7554104074aa7294fae37fcec35fac853bb33fbd.jpg)

![b8fbdd6671dcf76bf139d896270cbfd41e92e7e4021e328df7a3252f0b7878b8.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/b8fbdd6671dcf76bf139d896270cbfd41e92e7e4021e328df7a3252f0b7878b8.jpg)

![ba08ac13608cb410f5ba97867c8afb99c74d00dd63350a9678e40d91cda94fe6.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/ba08ac13608cb410f5ba97867c8afb99c74d00dd63350a9678e40d91cda94fe6.jpg)

![d4f03d1891bbe0292e69fe8dafc3ce1287c314942dece27a8bbe89e31fd5b4ff.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/d4f03d1891bbe0292e69fe8dafc3ce1287c314942dece27a8bbe89e31fd5b4ff.jpg)

![e2ebfe39e5d02c09cb5496976df3f0a163f374fc917b69b2036935a1f5fb669a.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/e2ebfe39e5d02c09cb5496976df3f0a163f374fc917b69b2036935a1f5fb669a.jpg)

![e55dfdb43a8ac0a2fe7110de0d3c869c80c3b5b46dfee860522ddd22e8937e69.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/e55dfdb43a8ac0a2fe7110de0d3c869c80c3b5b46dfee860522ddd22e8937e69.jpg)

![ea375c7e89cde1809ae33d21955b06a7c633b88d00ad2773ec9a83041ea3ffee.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/images/ea375c7e89cde1809ae33d21955b06a7c633b88d00ad2773ec9a83041ea3ffee.jpg)

### Tables

![b3f312f60accfc35538cfda6b21ae586546b4880c06524db6d04bd82f399b88d.jpg](../icml_results/917_N2M_%20Bridging%20Navigation%20and%20Manipulation%20by%20Learning%20Pose%20Preference%20from%20Rollout/tables/b3f312f60accfc35538cfda6b21ae586546b4880c06524db6d04bd82f399b88d.jpg)

<a id="paper-0901"></a>

## From Diagrams to Code: Multilingual Programming with Visual Design


### Images

![0284ffe0933bb0c9180d01cce4302b8006a6d55336bc17f52e760252cc2781e6.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/0284ffe0933bb0c9180d01cce4302b8006a6d55336bc17f52e760252cc2781e6.jpg)

![0a91c1139b70525bb2054861f1f0a50287d330041ee3f24f2be8c05bd42d06c6.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/0a91c1139b70525bb2054861f1f0a50287d330041ee3f24f2be8c05bd42d06c6.jpg)

![12196b56a83c443f3fb7f5e6668b8b4e89e24f2a41a88fb46470810ae20852cb.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/12196b56a83c443f3fb7f5e6668b8b4e89e24f2a41a88fb46470810ae20852cb.jpg)

![1843b57fc0f60f3113ae58049e9d6e43bc45516618d25980e86a3de86da11cb7.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/1843b57fc0f60f3113ae58049e9d6e43bc45516618d25980e86a3de86da11cb7.jpg)

![211dee20b49c962fcda5bb7c0383b02c3b8c95a7dee9066db9297baf437d967c.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/211dee20b49c962fcda5bb7c0383b02c3b8c95a7dee9066db9297baf437d967c.jpg)

![21c6abd4900ebebac553a3b60f353f98dc3ae068c94f038413de7aeed05fbf06.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/21c6abd4900ebebac553a3b60f353f98dc3ae068c94f038413de7aeed05fbf06.jpg)

![3d5121535c79b6b4dfb80b643703c7fb2e0da5c10f0358444e89b210cf13fa07.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/3d5121535c79b6b4dfb80b643703c7fb2e0da5c10f0358444e89b210cf13fa07.jpg)

![46dce786a5ac9a38f57cdee5c8ce6b1e2291e65271ad18757b0f9fc70e51f33c.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/46dce786a5ac9a38f57cdee5c8ce6b1e2291e65271ad18757b0f9fc70e51f33c.jpg)

![4d02e86e96825e2ffae36b74c593f5d76fcf4dc4e412b58691383d37e9957f6e.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/4d02e86e96825e2ffae36b74c593f5d76fcf4dc4e412b58691383d37e9957f6e.jpg)

![5192c55bc94ab1d978ca7677bcc7b68702b07b11c4242bb9068c6e71127378e9.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/5192c55bc94ab1d978ca7677bcc7b68702b07b11c4242bb9068c6e71127378e9.jpg)

![5b6a81997cbaff426a1047a23dcd34f1a5c268184b42f949ef8222ac3c49c5e9.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/5b6a81997cbaff426a1047a23dcd34f1a5c268184b42f949ef8222ac3c49c5e9.jpg)

![5ec7415329a8b77f8a71ce563759a186c00b85f8391b2a7787f1d1774e4420a6.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/5ec7415329a8b77f8a71ce563759a186c00b85f8391b2a7787f1d1774e4420a6.jpg)

![61c80b684c0ab0d58281df932e293784e5fb37e0a21c9e9f2fdccd91ef7d282d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/61c80b684c0ab0d58281df932e293784e5fb37e0a21c9e9f2fdccd91ef7d282d.jpg)

![68ce7250cc66aeb6cd977f41d12cb759d04d8396a9bf933f885892b7303da403.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/68ce7250cc66aeb6cd977f41d12cb759d04d8396a9bf933f885892b7303da403.jpg)

![698de9be1486130e1e65fed22fa618f2f15a219d43b95d82e89bf22b95a62945.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/698de9be1486130e1e65fed22fa618f2f15a219d43b95d82e89bf22b95a62945.jpg)

![69cd5f765ba39ec615a1818ab69c2ce5fb3273bee32104ffe9c5807cd9c902fe.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/69cd5f765ba39ec615a1818ab69c2ce5fb3273bee32104ffe9c5807cd9c902fe.jpg)

![70bdb3538f0e652a4d6a04414e7e40cf382e1e6f4bc9c85a1f45a830a81b5f28.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/70bdb3538f0e652a4d6a04414e7e40cf382e1e6f4bc9c85a1f45a830a81b5f28.jpg)

![72a50440a1743e7a0e9067bad5e9c27b5f5d06a5b75085426fe2c48f8cc289c0.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/72a50440a1743e7a0e9067bad5e9c27b5f5d06a5b75085426fe2c48f8cc289c0.jpg)

![7a9e235693ec6f879a81bac78be02513dbf000c512d6038d629191b09edecc1c.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/7a9e235693ec6f879a81bac78be02513dbf000c512d6038d629191b09edecc1c.jpg)

![7fe3dc3c0803031d49efcf2aaffffe3b82dc999592104fd54fb609bd0539e0ff.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/7fe3dc3c0803031d49efcf2aaffffe3b82dc999592104fd54fb609bd0539e0ff.jpg)

![89e3eb1acb03ef8fcb5432b57ac95369dfe99d3272d33e0f92b3ff9ead555534.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/89e3eb1acb03ef8fcb5432b57ac95369dfe99d3272d33e0f92b3ff9ead555534.jpg)

![920f508aef0282b646dc24c220bbbf76652ead9138c2c5c8e234caf70d3689aa.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/920f508aef0282b646dc24c220bbbf76652ead9138c2c5c8e234caf70d3689aa.jpg)

![94ae1f51d09e4d2375602e36e2d5b58d1bd94f0024e838ca1d6959d2d07d5f1b.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/94ae1f51d09e4d2375602e36e2d5b58d1bd94f0024e838ca1d6959d2d07d5f1b.jpg)

![9ae664787dd0f461116324092c58447d1af3538a9ef8136ba0d0eaad1f4aea4c.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/9ae664787dd0f461116324092c58447d1af3538a9ef8136ba0d0eaad1f4aea4c.jpg)

![9c005e77a544bf0310a562d454b64ff585f2a4ffd285decb4e049ffa101d716d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/9c005e77a544bf0310a562d454b64ff585f2a4ffd285decb4e049ffa101d716d.jpg)

![b3d0d5dc381d42ece14fe95d4c0d6400a5d7f132ba4376159637feea0aa732d5.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/b3d0d5dc381d42ece14fe95d4c0d6400a5d7f132ba4376159637feea0aa732d5.jpg)

![b4d41c507ccbfbb5482aeddd8f3fa77f044495e8744bc1ca1ea80ad759583e6d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/b4d41c507ccbfbb5482aeddd8f3fa77f044495e8744bc1ca1ea80ad759583e6d.jpg)

![b567667417a1deec8b51dbe0f468916fe98de75f6c9d621d544189eac9e256cc.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/b567667417a1deec8b51dbe0f468916fe98de75f6c9d621d544189eac9e256cc.jpg)

![c3165236117b58f040e56a3606d5c523b05daef1d994831c616b4691db05c854.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/c3165236117b58f040e56a3606d5c523b05daef1d994831c616b4691db05c854.jpg)

![c63be39dfec812f00a83aae1139ed3f840031ca5540a2d3adf019d23357c2a4e.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/c63be39dfec812f00a83aae1139ed3f840031ca5540a2d3adf019d23357c2a4e.jpg)

![ca086507281c5fadc0dd5459eb0ee226ca43a90ff0966c08d4089cd6de997386.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/ca086507281c5fadc0dd5459eb0ee226ca43a90ff0966c08d4089cd6de997386.jpg)

![cd7b6d6a92d15dcc88ec1bdbed1bdbd0d24b9bba806192600a072e42ae9441c8.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/cd7b6d6a92d15dcc88ec1bdbed1bdbd0d24b9bba806192600a072e42ae9441c8.jpg)

![d6c65cce5c221127ca18feadcbe956301572cbb4bf696099f1f9451d39a9e150.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/d6c65cce5c221127ca18feadcbe956301572cbb4bf696099f1f9451d39a9e150.jpg)

![d8be7bfd1aec61be77d691d0785ae38bac7ba8f5c31ce4b4f14bdb1f18fa9888.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/d8be7bfd1aec61be77d691d0785ae38bac7ba8f5c31ce4b4f14bdb1f18fa9888.jpg)

![e00025ad81aea0ba416144153e49e97d68cdc612d52492b1b74db78efc404d9a.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/e00025ad81aea0ba416144153e49e97d68cdc612d52492b1b74db78efc404d9a.jpg)

![e94dd3375aa62f756bca94624113422f922c4c3b9f73d8ba86208bf222ecb32d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/e94dd3375aa62f756bca94624113422f922c4c3b9f73d8ba86208bf222ecb32d.jpg)

![ea6c92c8a99b82ee2fe285151db478b266fd24c1ce0bfaf45f38051e9c074571.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/ea6c92c8a99b82ee2fe285151db478b266fd24c1ce0bfaf45f38051e9c074571.jpg)

![eb813fcaf762477b4a086c594d283a70823dbfd2b26bde9b746794907b5193b7.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/eb813fcaf762477b4a086c594d283a70823dbfd2b26bde9b746794907b5193b7.jpg)

![ef85d20c4bda6603b00be33a2dbea82f90ad772c61e86dd8c0baf4460a7acc17.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/ef85d20c4bda6603b00be33a2dbea82f90ad772c61e86dd8c0baf4460a7acc17.jpg)

![efa8487f8e5b5a3418a6030739ea944a00c5b9aab5eafc87e2359a2557befd39.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/efa8487f8e5b5a3418a6030739ea944a00c5b9aab5eafc87e2359a2557befd39.jpg)

![f6bd4b51ef81aa1ef140b120b30534dc1006e0436da5319dd7997ed45d99069a.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/images/f6bd4b51ef81aa1ef140b120b30534dc1006e0436da5319dd7997ed45d99069a.jpg)

### Tables

![0704e5f7e985db911003e295a47ec94f3f18d313d5edb2d228bc3f5c22913524.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/0704e5f7e985db911003e295a47ec94f3f18d313d5edb2d228bc3f5c22913524.jpg)

![2099ded2a6d52a1cfbcb3e57148c070354d9423d6902a1a936500cee4cb9278d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/2099ded2a6d52a1cfbcb3e57148c070354d9423d6902a1a936500cee4cb9278d.jpg)

![32896c51ab5ada4a411e469085693fba557ffd5f73c924f6a5a66e45af36f1b8.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/32896c51ab5ada4a411e469085693fba557ffd5f73c924f6a5a66e45af36f1b8.jpg)

![558bdb870098839daf7ee548212de9b3a8e2dd3a1fd54d3314de3338ec76c41b.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/558bdb870098839daf7ee548212de9b3a8e2dd3a1fd54d3314de3338ec76c41b.jpg)

![5bc348e7bdb7a023db1e5a3946e36b1f46c53ef7021848539c48a929caca0109.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/5bc348e7bdb7a023db1e5a3946e36b1f46c53ef7021848539c48a929caca0109.jpg)

![634f9d76b21a91e20561f1cba6ef9aea9e5ae5872bc113f3715c69aa38073203.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/634f9d76b21a91e20561f1cba6ef9aea9e5ae5872bc113f3715c69aa38073203.jpg)

![983b056195a13c84f0467e44c8276ed047ae83ead6a9ec61c1fd60ea91324085.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/983b056195a13c84f0467e44c8276ed047ae83ead6a9ec61c1fd60ea91324085.jpg)

![a6b712104c75809c15ecce6a1fdf1684659b920b3d08d681a15e14be73004a4d.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/a6b712104c75809c15ecce6a1fdf1684659b920b3d08d681a15e14be73004a4d.jpg)

![a995b04ff588f4c160655559e5d269346e632513908b90839ab15d82745d9488.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/a995b04ff588f4c160655559e5d269346e632513908b90839ab15d82745d9488.jpg)

![c7979eb5bc358896104203a8baa299de03ae73b76dcb844b6269da5512fe8a8b.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/c7979eb5bc358896104203a8baa299de03ae73b76dcb844b6269da5512fe8a8b.jpg)

![ca4e7a7bed8557544282f6d14027f4ab12faf550395a245cc788dc29bc2199f1.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/ca4e7a7bed8557544282f6d14027f4ab12faf550395a245cc788dc29bc2199f1.jpg)

![d4fff7d11bda0d74a84ce4c0f65f9555f01d892c014aea3708468c4e77b2166c.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/d4fff7d11bda0d74a84ce4c0f65f9555f01d892c014aea3708468c4e77b2166c.jpg)

![d6fd257f04963b32010d8cebcce4261017d64d6960780c1c86b868b6eab81834.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/d6fd257f04963b32010d8cebcce4261017d64d6960780c1c86b868b6eab81834.jpg)

![dcb61cb1669b06e028d4d72c8ca980df0fda212dc19c498ef082abe07d076344.jpg](../icml_results/918_From%20Diagrams%20to%20Code_%20Multilingual%20Programming%20with%20Visual%20Design/tables/dcb61cb1669b06e028d4d72c8ca980df0fda212dc19c498ef082abe07d076344.jpg)

<a id="paper-0902"></a>

## GXPO: Group Cross-Lingual Relative Policy Optimization for Code Generation


### Images

![2432cd15f39863f60d76b26ea94d47a8f0ef6c51a6b3aa5a055bbcafe2d81a41.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/2432cd15f39863f60d76b26ea94d47a8f0ef6c51a6b3aa5a055bbcafe2d81a41.jpg)

![a7ea4254700204afa3001668012de1371289e1eb087dd5e1623060621c973279.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/a7ea4254700204afa3001668012de1371289e1eb087dd5e1623060621c973279.jpg)

![d89d3a3d0f2c4c7dd4f1f6e7a82ff1c1c29a3cc8010c7631dc0707b1577335de.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/d89d3a3d0f2c4c7dd4f1f6e7a82ff1c1c29a3cc8010c7631dc0707b1577335de.jpg)

![df57b25796094ab1ba64a2fc10fe077f1d1df2bec9610c5b688beb4927e2247b.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/df57b25796094ab1ba64a2fc10fe077f1d1df2bec9610c5b688beb4927e2247b.jpg)

![e140d3c34aed2cdb03b7b3c82bef9c59257604ca818b845c660b72d5f935a7a6.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/e140d3c34aed2cdb03b7b3c82bef9c59257604ca818b845c660b72d5f935a7a6.jpg)

![fca01c289ff4a94bbf95652bb87d6d2d60ba79707aba1454da1797c95f7d97be.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/images/fca01c289ff4a94bbf95652bb87d6d2d60ba79707aba1454da1797c95f7d97be.jpg)

### Tables

![22cb5e02bd24e9309a66c4d0b2045e18d5b0540e292975c33d5267bad6dae9aa.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/tables/22cb5e02bd24e9309a66c4d0b2045e18d5b0540e292975c33d5267bad6dae9aa.jpg)

![8f16972f27ab3645086c7f5bf7b56c08d83eaa6340ac72bd879a4e7c9a0d52f6.jpg](../icml_results/919_GXPO_%20Group%20Cross-Lingual%20Relative%20Policy%20Optimization%20for%20Code%20Generation/tables/8f16972f27ab3645086c7f5bf7b56c08d83eaa6340ac72bd879a4e7c9a0d52f6.jpg)

<a id="paper-0903"></a>

## Probabilistically-routed Bayesian Additive Spanning Trees for Learning on Constrained Domains


### Images

![1ddc62b2656a3c6c18f697e4ea42ad4be32d276f7e2d59b92c4c4e9bbde3894c.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/images/1ddc62b2656a3c6c18f697e4ea42ad4be32d276f7e2d59b92c4c4e9bbde3894c.jpg)

![8392d1837204bd8f3f90a12771dda61bdab1e26c98f57e7540b192eb7f7cecad.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/images/8392d1837204bd8f3f90a12771dda61bdab1e26c98f57e7540b192eb7f7cecad.jpg)

![a7f17524f76c83c0e2b6239a9e7671f7d41a24dc82f1fb0c5664e42fae9dc460.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/images/a7f17524f76c83c0e2b6239a9e7671f7d41a24dc82f1fb0c5664e42fae9dc460.jpg)

![df0cf591c487b801c0fadc24245813e45596261c92069e3c015cc2530a2f8766.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/images/df0cf591c487b801c0fadc24245813e45596261c92069e3c015cc2530a2f8766.jpg)

### Tables

![052b7f629ce8d0f6d805c82502cb3a9d3eda9b4b25ed188ebec89a48907ef26f.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/052b7f629ce8d0f6d805c82502cb3a9d3eda9b4b25ed188ebec89a48907ef26f.jpg)

![39a5ac9bfe91797ecaa2343ffd308dfbd61a73d1ebe2b0dfeef04fee464b89ef.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/39a5ac9bfe91797ecaa2343ffd308dfbd61a73d1ebe2b0dfeef04fee464b89ef.jpg)

![62ccddae6f84bb6a44b3dcc896b4559f8db8e79848fc7302366e108f04128d78.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/62ccddae6f84bb6a44b3dcc896b4559f8db8e79848fc7302366e108f04128d78.jpg)

![7604baa2f649319be8c9b0b5d89662f28aeccf07a654a8f37109093e9667613c.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/7604baa2f649319be8c9b0b5d89662f28aeccf07a654a8f37109093e9667613c.jpg)

![92a02f473d0a4f743435abcb0a94430c6161ce2a3fc3857e45820cd94343286b.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/92a02f473d0a4f743435abcb0a94430c6161ce2a3fc3857e45820cd94343286b.jpg)

![9794ef87a79dab44f892c74d8b0f2393652a4068937baa6a385d92a7e0093114.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/9794ef87a79dab44f892c74d8b0f2393652a4068937baa6a385d92a7e0093114.jpg)

![e5d1a2f3a60adbcccaeb5569d96f5540f58dbc74e2c82bf1cebdee4dabf625b8.jpg](../icml_results/920_Probabilistically-routed%20Bayesian%20Additive%20Spanning%20Trees%20for%20Learning%20on%20Constrained%20Domains/tables/e5d1a2f3a60adbcccaeb5569d96f5540f58dbc74e2c82bf1cebdee4dabf625b8.jpg)

<a id="paper-0904"></a>

## Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking


### Images

![167a8a17de8f5adf3fe5ca05799bb180c26dd16e3e43eb3e68690b2115a9680d.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/167a8a17de8f5adf3fe5ca05799bb180c26dd16e3e43eb3e68690b2115a9680d.jpg)

![655a8d5f55d446ebffb8cc8c92b1355b4b137202922c80df31b9ed5a6266d9b7.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/655a8d5f55d446ebffb8cc8c92b1355b4b137202922c80df31b9ed5a6266d9b7.jpg)

![7676c9bf18ceac67c12095be1c3db51f72b381e6fba7eca04839c6995bb74227.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/7676c9bf18ceac67c12095be1c3db51f72b381e6fba7eca04839c6995bb74227.jpg)

![a9110991c4085f8343bcccc4e18528d7b876674f9ad53fa7ba23a7805266961e.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/a9110991c4085f8343bcccc4e18528d7b876674f9ad53fa7ba23a7805266961e.jpg)

![d171ca9363808edc3864b08259495692752d64fcbadcaef539beb85492f7292b.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/d171ca9363808edc3864b08259495692752d64fcbadcaef539beb85492f7292b.jpg)

![e094961101aca836b7efa48b46a5617dd1cce5943eb55746c018a01e2b96b160.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/e094961101aca836b7efa48b46a5617dd1cce5943eb55746c018a01e2b96b160.jpg)

![f49ceb41c1635245a6b9739209b5b3371e5df5608c42a848543518588bd31a14.jpg](../icml_results/921_Fairness%20in%20Aggregation_%20Optimal%20Top-%24k%24%20and%20Improved%20Full%20Ranking/images/f49ceb41c1635245a6b9739209b5b3371e5df5608c42a848543518588bd31a14.jpg)

<a id="paper-0905"></a>

## Randomized Feasibility Methods for Constrained Optimization with Adaptive Step Sizes


### Images

![200874118c5f0c02e9e593f715922a80b11990eca2b4b7f152e2b662ae674626.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/200874118c5f0c02e9e593f715922a80b11990eca2b4b7f152e2b662ae674626.jpg)

![27a35e3c1fb823f0d24ba2bfa14e66709e7eb080e752c6d71342a86b06374589.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/27a35e3c1fb823f0d24ba2bfa14e66709e7eb080e752c6d71342a86b06374589.jpg)

![5a5d3daf1aa962a3c47667483efd9fda0b829f58a9716a326efae5c7aa2ffc28.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/5a5d3daf1aa962a3c47667483efd9fda0b829f58a9716a326efae5c7aa2ffc28.jpg)

![897265b7093103b1a36acc44b29769c9e8f0ee9a08b4c470b0361f9904b7ed1e.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/897265b7093103b1a36acc44b29769c9e8f0ee9a08b4c470b0361f9904b7ed1e.jpg)

![8ea5f65b52438e44168c3a66041a6df1e7c9188eb91ee5eb706cae33a4252f6e.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/8ea5f65b52438e44168c3a66041a6df1e7c9188eb91ee5eb706cae33a4252f6e.jpg)

![a4dc1734945c1ffb6037f7e367b379019e79499ebb3da4aebab5c3af660d2230.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/images/a4dc1734945c1ffb6037f7e367b379019e79499ebb3da4aebab5c3af660d2230.jpg)

### Tables

![4df0490892bf8f0ada06667567f636266d8efd198a5b244f3fe04e164f5ae472.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/tables/4df0490892bf8f0ada06667567f636266d8efd198a5b244f3fe04e164f5ae472.jpg)

![76823daca00a7c7f2f54010e40a4ed65e7f1775bb83b96ff9ec8d81218c8b7b5.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/tables/76823daca00a7c7f2f54010e40a4ed65e7f1775bb83b96ff9ec8d81218c8b7b5.jpg)

![a16d07ba018c3592950ec29071a27743510df1edc37a97ce71fecd7cdadac30b.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/tables/a16d07ba018c3592950ec29071a27743510df1edc37a97ce71fecd7cdadac30b.jpg)

![b895abe19e08f47ef10b32bee01610545a66816424a2582d64c45b6aab8aab7c.jpg](../icml_results/922_Randomized%20Feasibility%20Methods%20for%20Constrained%20Optimization%20with%20Adaptive%20Step%20Sizes/tables/b895abe19e08f47ef10b32bee01610545a66816424a2582d64c45b6aab8aab7c.jpg)

<a id="paper-0906"></a>

## DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning


### Images

![0b13f8f556024ca7cee9e4fbde346b353fe250b730c437341986a7936af6cc4a.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/0b13f8f556024ca7cee9e4fbde346b353fe250b730c437341986a7936af6cc4a.jpg)

![100f22e2ff61098d736e3eb336a6b6059dc3850d82af91aef1f3908b2acf8262.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/100f22e2ff61098d736e3eb336a6b6059dc3850d82af91aef1f3908b2acf8262.jpg)

![17cf5802603cd9439910046a29f5dd68164d4032b74a556e9312a9768315b62b.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/17cf5802603cd9439910046a29f5dd68164d4032b74a556e9312a9768315b62b.jpg)

![1f6fa3543c3ad7c73c50e774f617384b308f86cb47fffd2424779955ebd9931c.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/1f6fa3543c3ad7c73c50e774f617384b308f86cb47fffd2424779955ebd9931c.jpg)

![2050c1a8bdc5ec782d923193b9a7fc709fa6545e2f73ccb60d5e39367a035a1e.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/2050c1a8bdc5ec782d923193b9a7fc709fa6545e2f73ccb60d5e39367a035a1e.jpg)

![28fe2794d7e6aef1bfd5bd9283166d97c96ab96202ac34f386b88e7b22f1b3af.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/28fe2794d7e6aef1bfd5bd9283166d97c96ab96202ac34f386b88e7b22f1b3af.jpg)

![70b9d9cfb5489524b600340ac8fcecafdeb06e96d90be6808e4c72fc33dc44f4.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/70b9d9cfb5489524b600340ac8fcecafdeb06e96d90be6808e4c72fc33dc44f4.jpg)

![72eb49c1ffdc8678e2c8d79cb60d1152624e98a394f7c26772d664f9c1b7eb3c.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/72eb49c1ffdc8678e2c8d79cb60d1152624e98a394f7c26772d664f9c1b7eb3c.jpg)

![784fd3b38b6e7298d01842a5c853637b860b89176afb4d496bf128f821675c17.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/784fd3b38b6e7298d01842a5c853637b860b89176afb4d496bf128f821675c17.jpg)

![919b37c8ddee5cd28ec77ed73447bd03f41260cfb88d9403aff76e51be1f95d0.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/919b37c8ddee5cd28ec77ed73447bd03f41260cfb88d9403aff76e51be1f95d0.jpg)

![97869985530ebcdeed400dd6201b3c102dcb665f633039acb25a5727fd341da0.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/97869985530ebcdeed400dd6201b3c102dcb665f633039acb25a5727fd341da0.jpg)

![aba831c9a2eee3cd6cb082b431eee0f7aae0febe6a46ff288c81a803c89e8bce.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/aba831c9a2eee3cd6cb082b431eee0f7aae0febe6a46ff288c81a803c89e8bce.jpg)

![b16a0d41e5b80efc305c03b84d4886e9c4ce4d9e9dd07d090ba9b5e19f45735b.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/b16a0d41e5b80efc305c03b84d4886e9c4ce4d9e9dd07d090ba9b5e19f45735b.jpg)

![b43f64ea7fd970631e6ef590c3cd8d777d6c5057869b52a42d5a9f9b219f2fa0.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/b43f64ea7fd970631e6ef590c3cd8d777d6c5057869b52a42d5a9f9b219f2fa0.jpg)

![b671c9ac74f8eab574453b995a919cd716ae7b5c4fe42b438dbf6e585af07007.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/b671c9ac74f8eab574453b995a919cd716ae7b5c4fe42b438dbf6e585af07007.jpg)

![dc4b871f52b2088f2fe3c540697120959df366a5322025a31341d28b8c2d51bb.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/dc4b871f52b2088f2fe3c540697120959df366a5322025a31341d28b8c2d51bb.jpg)

![e3e17f1c20021d96be2f22d0faa178c6cd3a44feafd92621e44ba1379eac1179.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/e3e17f1c20021d96be2f22d0faa178c6cd3a44feafd92621e44ba1379eac1179.jpg)

![f5465cddbf9f2745c13ac9c2fc630bae54a2b4d6d9770cb73ce412d32bad9b44.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/images/f5465cddbf9f2745c13ac9c2fc630bae54a2b4d6d9770cb73ce412d32bad9b44.jpg)

### Tables

![1a4fc2495f82d5a69706871ef3c42759bc9c23fb69a7054696e8969ed4463eb1.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/1a4fc2495f82d5a69706871ef3c42759bc9c23fb69a7054696e8969ed4463eb1.jpg)

![452ecd7b1f02c298df59b1a586875d22d74dcb1445d6b9a0472c277fb1d3b890.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/452ecd7b1f02c298df59b1a586875d22d74dcb1445d6b9a0472c277fb1d3b890.jpg)

![6f976eb637f3ae760c5fe7b6753dd5e9bc1391253cd273eaeb544a6bfd55d84d.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/6f976eb637f3ae760c5fe7b6753dd5e9bc1391253cd273eaeb544a6bfd55d84d.jpg)

![83b4b6a80d204081ca5a014f68a4e274718ffc4471e6809264a8b0fc9fe50c72.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/83b4b6a80d204081ca5a014f68a4e274718ffc4471e6809264a8b0fc9fe50c72.jpg)

![a2f7a19e109b3100c704f92ec2271c6e90b06a545dc055316d3d593330894fec.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/a2f7a19e109b3100c704f92ec2271c6e90b06a545dc055316d3d593330894fec.jpg)

![d01764403f8fb23363e4a39c418519e7582de622a5ab9fe090883606c780459d.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/d01764403f8fb23363e4a39c418519e7582de622a5ab9fe090883606c780459d.jpg)

![e10d2fba30495f730c779d941a2e892f5497f1b49a1a4f773469b09a2f7a19d6.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/e10d2fba30495f730c779d941a2e892f5497f1b49a1a4f773469b09a2f7a19d6.jpg)

![ec3e5c2459f011320a10de5def03004117d9a32bed69c83d0f72ad9572c48a31.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/ec3e5c2459f011320a10de5def03004117d9a32bed69c83d0f72ad9572c48a31.jpg)

![f9760a41e7660a3b642f3d450b993a3fc7abaa510147bcae3341a30919dc128b.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/f9760a41e7660a3b642f3d450b993a3fc7abaa510147bcae3341a30919dc128b.jpg)

![fe3d70352aaec950504ca57efb033df0d9ef6b9f6c8c186260a017f04ccdfcf0.jpg](../icml_results/923_DC-W2S_%20Dual-Consensus%20Weak-to-Strong%20Training%20for%20Reliable%20Process%20Reward%20Modeling%20in%20Biological%20Re/tables/fe3d70352aaec950504ca57efb033df0d9ef6b9f6c8c186260a017f04ccdfcf0.jpg)

<a id="paper-0907"></a>

## Continual Learning With Participation Privacy: An Auditable Buffering-Aggregation Recipe


### Images

![0bd43fef8a08f42b7de0774628268fb1a6cf7cbe662a3e495269f5debff9a1fe.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/0bd43fef8a08f42b7de0774628268fb1a6cf7cbe662a3e495269f5debff9a1fe.jpg)

![0f0a8142bdd3038c905e4fd8d2aa9f42f7aa0bc60eb3cbf9ab076865694efdab.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/0f0a8142bdd3038c905e4fd8d2aa9f42f7aa0bc60eb3cbf9ab076865694efdab.jpg)

![1124d612fbfa120c127dde29583ab4c04274e305dac6f5d02b0f4f0ef8a17614.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/1124d612fbfa120c127dde29583ab4c04274e305dac6f5d02b0f4f0ef8a17614.jpg)

![673885a53dc1a4af082fd0107c564d2302cdd069a80bab7884f60870908eb2e1.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/673885a53dc1a4af082fd0107c564d2302cdd069a80bab7884f60870908eb2e1.jpg)

![7189a4083b1c1c39729688fe7fb68e8806764afd2a990d2798ef7e7be9900bd5.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/7189a4083b1c1c39729688fe7fb68e8806764afd2a990d2798ef7e7be9900bd5.jpg)

![79b347049d97f4a09c8e46e3115fffe4363d7effdc84efd582f35ce13f667b28.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/79b347049d97f4a09c8e46e3115fffe4363d7effdc84efd582f35ce13f667b28.jpg)

![816cc176f914d1b2e60797135fd75079c7153cd5a4c1da869d9c7607952f54e7.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/816cc176f914d1b2e60797135fd75079c7153cd5a4c1da869d9c7607952f54e7.jpg)

![88c5a893477e3a5c70116e3b4195720ea6c3bd16fca3eb0481f8c63d2777a75c.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/88c5a893477e3a5c70116e3b4195720ea6c3bd16fca3eb0481f8c63d2777a75c.jpg)

![ab68193e665bfb5f12050d6eeff3f7d2c2d8c443ec2de1e588282d832aa0dd98.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/ab68193e665bfb5f12050d6eeff3f7d2c2d8c443ec2de1e588282d832aa0dd98.jpg)

![baef2bd13d60e5489b85d15fdd961b263f711069aac8746a94fa52859575d6f4.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/baef2bd13d60e5489b85d15fdd961b263f711069aac8746a94fa52859575d6f4.jpg)

![d6b74562b30e8825aa9b432a679917ed2f8618d02034c8729935e433a7ae5643.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/images/d6b74562b30e8825aa9b432a679917ed2f8618d02034c8729935e433a7ae5643.jpg)

### Tables

![2151890beb5e2dcba9e4ae5e7b1ac2228b22402ae46fed0a3a35d6a52a9a98dd.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/tables/2151890beb5e2dcba9e4ae5e7b1ac2228b22402ae46fed0a3a35d6a52a9a98dd.jpg)

![6c6ca6f676e2e6bdc64b5ffcca9a602be42b3c4be02f4884a08565994397c093.jpg](../icml_results/924_Continual%20Learning%20With%20Participation%20Privacy_%20An%20Auditable%20Buffering-Aggregation%20Recipe/tables/6c6ca6f676e2e6bdc64b5ffcca9a602be42b3c4be02f4884a08565994397c093.jpg)

<a id="paper-0908"></a>

## Curating the Future: A Scalable Recipe for Training Open-Ended Forecasters


### Images

![089bccd96f1f838b34be42da8d6de557799accc556f14ff41a30a91a5fbd861b.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/089bccd96f1f838b34be42da8d6de557799accc556f14ff41a30a91a5fbd861b.jpg)

![1596a11c48c80fa406e13501c14f995a318cb0a5560a2ba968e7b0777014a158.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/1596a11c48c80fa406e13501c14f995a318cb0a5560a2ba968e7b0777014a158.jpg)

![15cb74ad35de7bf80354650dd70ab4fd6e692cd1741be7df3101e902f2b38522.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/15cb74ad35de7bf80354650dd70ab4fd6e692cd1741be7df3101e902f2b38522.jpg)

![1db4790d5f2a5c9d71d2b876e478cb3f5df668217684370cdea7a73f86278161.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/1db4790d5f2a5c9d71d2b876e478cb3f5df668217684370cdea7a73f86278161.jpg)

![1e942c49c5faa05b6331e46e3960e4b22dd4ee59f74164ef072667edfddcf5cb.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/1e942c49c5faa05b6331e46e3960e4b22dd4ee59f74164ef072667edfddcf5cb.jpg)

![22c830910f577703945b665edf4d966fad6e779fa161245f7b2bd2791c4e7e70.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/22c830910f577703945b665edf4d966fad6e779fa161245f7b2bd2791c4e7e70.jpg)

![2341198cfc1c053e4adda8d7460667539fcb7f92953bc911238b70faae671843.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/2341198cfc1c053e4adda8d7460667539fcb7f92953bc911238b70faae671843.jpg)

![2bceff995fff44f41ccea87b554e9fbcb4aa5009aa603913cdc36acd9839cc96.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/2bceff995fff44f41ccea87b554e9fbcb4aa5009aa603913cdc36acd9839cc96.jpg)

![380e6915b5553902c13f63e9db899a3d943393f6e02b839d443b14aa937cf442.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/380e6915b5553902c13f63e9db899a3d943393f6e02b839d443b14aa937cf442.jpg)

![55915691e9fa09bad6487357e85943bf5c727b24a0b99c4092785c4d6e99347d.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/55915691e9fa09bad6487357e85943bf5c727b24a0b99c4092785c4d6e99347d.jpg)

![5e66ff6c1b44a351ebfb93509c55970c627efb275b31d107b966b16e68c4259c.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/5e66ff6c1b44a351ebfb93509c55970c627efb275b31d107b966b16e68c4259c.jpg)

![88969a9d517ba3331636c2f6163bff76f485d72767cda43a07107129b208aa0d.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/88969a9d517ba3331636c2f6163bff76f485d72767cda43a07107129b208aa0d.jpg)

![9d32ada4f0dca462498eb4fa5ff6270735289c08cd629161bfc7b3170df55239.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/9d32ada4f0dca462498eb4fa5ff6270735289c08cd629161bfc7b3170df55239.jpg)

![a2bb90c194dc4eeec69de6f784f5c239059bfe2405587f54722a69ef1ef21026.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/a2bb90c194dc4eeec69de6f784f5c239059bfe2405587f54722a69ef1ef21026.jpg)

![aba388ceea5c80f1a5f26c68e68a7d3897b371e4da73e28b9c9675825e8b311b.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/aba388ceea5c80f1a5f26c68e68a7d3897b371e4da73e28b9c9675825e8b311b.jpg)

![cdcc8a6327e0d74ef2efee985902452a993b2a6768a463af9f60aed66ebbfd47.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/cdcc8a6327e0d74ef2efee985902452a993b2a6768a463af9f60aed66ebbfd47.jpg)

![ded1f222984ce8afcd5b2b9de7d00db351ded9dc5e1de8f31c41e3f5c0fe5240.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/ded1f222984ce8afcd5b2b9de7d00db351ded9dc5e1de8f31c41e3f5c0fe5240.jpg)

![e15fd007f7eacea108af7229e34a9d490a78ce52df66d8d9b4c0cfaafc83a28c.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/e15fd007f7eacea108af7229e34a9d490a78ce52df66d8d9b4c0cfaafc83a28c.jpg)

![e42c6c8040348333d3207c832c30e45e9a6ac9918685ecc8e27691e47a13490e.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/e42c6c8040348333d3207c832c30e45e9a6ac9918685ecc8e27691e47a13490e.jpg)

![ed90ddb8468172538e827655d927a7c2776b4597528cef413a781f419b1b0a1c.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/images/ed90ddb8468172538e827655d927a7c2776b4597528cef413a781f419b1b0a1c.jpg)

### Tables

![0b7194d75888d7dde0a5716b2cad89b061b4036b34c8f4749dc89f862ddab899.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/0b7194d75888d7dde0a5716b2cad89b061b4036b34c8f4749dc89f862ddab899.jpg)

![2be5413ea55e074a9dc2050eee9ba613f10d0342790c4d9d0b730ed113d7fc1b.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/2be5413ea55e074a9dc2050eee9ba613f10d0342790c4d9d0b730ed113d7fc1b.jpg)

![5b9cf223eceb5f7f005dfc403e982f6d8463d386b5093bae40306aa630d83dcf.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/5b9cf223eceb5f7f005dfc403e982f6d8463d386b5093bae40306aa630d83dcf.jpg)

![625c2cfc0202f61b3656209934e698f5f524be42f73b5e34132276f0b5a2d0b2.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/625c2cfc0202f61b3656209934e698f5f524be42f73b5e34132276f0b5a2d0b2.jpg)

![8cf96d16e89d8c6380fa497851fb51275d0dd13ff9ebf6a1a0204dd0284376fc.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/8cf96d16e89d8c6380fa497851fb51275d0dd13ff9ebf6a1a0204dd0284376fc.jpg)

![909c9ee538212f4440ae7a423b54fa425701530fb9a50457fa236272f5c52047.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/909c9ee538212f4440ae7a423b54fa425701530fb9a50457fa236272f5c52047.jpg)

![ce9eb84203eefbdd86592015acf2cf5213b94e514358cf3df3f09e3b0a198e6e.jpg](../icml_results/925_Curating%20the%20Future_%20A%20Scalable%20Recipe%20for%20Training%20Open-Ended%20Forecasters/tables/ce9eb84203eefbdd86592015acf2cf5213b94e514358cf3df3f09e3b0a198e6e.jpg)

<a id="paper-0909"></a>

## The Surprising Difficulty of Search in Model-Based Reinforcement Learning


### Images

![72fd3ca5ac6eea35ba589a014a4f6b0b78bb710dc85420d30ececf32dce7e407.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/72fd3ca5ac6eea35ba589a014a4f6b0b78bb710dc85420d30ececf32dce7e407.jpg)

![8c0b1cd766d4d44ebf9cf14fc65dfaf2d996a9e98b469bad69b3bb415df31654.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/8c0b1cd766d4d44ebf9cf14fc65dfaf2d996a9e98b469bad69b3bb415df31654.jpg)

![93750dab5695a656cdd5c12ecc916a7a318b0eec33cd2eed1d7aee18fa2cb40b.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/93750dab5695a656cdd5c12ecc916a7a318b0eec33cd2eed1d7aee18fa2cb40b.jpg)

![9bde191d288e598cc014feb4626581d0ded7caf272558443335e048d2bcbdc1e.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/9bde191d288e598cc014feb4626581d0ded7caf272558443335e048d2bcbdc1e.jpg)

![aa069fe96a1ba5a4f44b6d84d78727816ca81bf46f5ffa5da9b6b9276154571c.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/aa069fe96a1ba5a4f44b6d84d78727816ca81bf46f5ffa5da9b6b9276154571c.jpg)

![b25b90855fcb39114a27bfc3858ab063f29fb97fdc68dfbf1da06c98fd4f9cc8.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/b25b90855fcb39114a27bfc3858ab063f29fb97fdc68dfbf1da06c98fd4f9cc8.jpg)

![bae3f461aa68104c11452348a5eb2c6e30607d9356c1303b7cc329f1ae01d65f.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/bae3f461aa68104c11452348a5eb2c6e30607d9356c1303b7cc329f1ae01d65f.jpg)

![dfa6074d8fa6b8a15d3f9db2982d5bd23e462b90f1ac335c499989d1af2ed102.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/dfa6074d8fa6b8a15d3f9db2982d5bd23e462b90f1ac335c499989d1af2ed102.jpg)

![f871fa28a72318055ca413e3348a1fd8beb1ae5bd121e4614849466b206ffc13.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/images/f871fa28a72318055ca413e3348a1fd8beb1ae5bd121e4614849466b206ffc13.jpg)

### Tables

![02575c922eaebf306e609c4d2c0ebf0feefb55230b18dda75ab4dd1391a4ce7b.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/02575c922eaebf306e609c4d2c0ebf0feefb55230b18dda75ab4dd1391a4ce7b.jpg)

![05858f1ba576fda634934d07edebf4ffa63bbc63f9dd8564b2782a201521f33f.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/05858f1ba576fda634934d07edebf4ffa63bbc63f9dd8564b2782a201521f33f.jpg)

![1d1749b2bdf2ddc0157629b3957451f1ba5a892d008a78e803c7ab6198c94256.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/1d1749b2bdf2ddc0157629b3957451f1ba5a892d008a78e803c7ab6198c94256.jpg)

![4065fa6693fa7613cb9504401759c5b4bb8e840f2037d8802fa04c299ac20e62.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/4065fa6693fa7613cb9504401759c5b4bb8e840f2037d8802fa04c299ac20e62.jpg)

![445b41f40e64b086b25945ad9570d7792010bb7a362181a00e24ad3ef11eb84c.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/445b41f40e64b086b25945ad9570d7792010bb7a362181a00e24ad3ef11eb84c.jpg)

![4c0e0b59d54b2ca3bf2e8c70b911297c5816c84b9df477b89dfff150ef644f98.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/4c0e0b59d54b2ca3bf2e8c70b911297c5816c84b9df477b89dfff150ef644f98.jpg)

![614adda32e617059517683f1fee01c30d16b513f8c9e80d0a2c1176f6760aa97.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/614adda32e617059517683f1fee01c30d16b513f8c9e80d0a2c1176f6760aa97.jpg)

![7ff47d41b23e09d5a912300b676daa093908d5ff51f0946d47c95fb479666d28.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/7ff47d41b23e09d5a912300b676daa093908d5ff51f0946d47c95fb479666d28.jpg)

![8a9426ded1973dc348d7b7c42bcee8581f58efaaf59312d55992843e03464c88.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/8a9426ded1973dc348d7b7c42bcee8581f58efaaf59312d55992843e03464c88.jpg)

![8f8c1c8a3d3c511ba4457c063e1a0e7d728a3e4b0196d226ec7fe8b01cd9da6d.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/8f8c1c8a3d3c511ba4457c063e1a0e7d728a3e4b0196d226ec7fe8b01cd9da6d.jpg)

![93d99d09f700bd9ccd93ae2f3a83edc90c1575ecc952f94e329df777da8e6486.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/93d99d09f700bd9ccd93ae2f3a83edc90c1575ecc952f94e329df777da8e6486.jpg)

![a4f3261011588b1099ce34cb4b210f981c1fa5f3d4602b178a14552b89f7990a.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/a4f3261011588b1099ce34cb4b210f981c1fa5f3d4602b178a14552b89f7990a.jpg)

![b367e7318ac97d289173082c7e720984b5b5ff81440323572fd910678681e683.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/b367e7318ac97d289173082c7e720984b5b5ff81440323572fd910678681e683.jpg)

![b3b7e954efb4c2251f2cca5b5c9320e850556220ddd59b46f1650560f5fe17e2.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/b3b7e954efb4c2251f2cca5b5c9320e850556220ddd59b46f1650560f5fe17e2.jpg)

![b815530d6be893ff2ef5f5c73b7e64ffea07ffb01a3aa3452f482ea7ae5ebc83.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/b815530d6be893ff2ef5f5c73b7e64ffea07ffb01a3aa3452f482ea7ae5ebc83.jpg)

![c8557398cf4867c27013f4c3099e775c5b1c75b56b476fb66ff5a84656805f5c.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/c8557398cf4867c27013f4c3099e775c5b1c75b56b476fb66ff5a84656805f5c.jpg)

![ca5a8d0bb48221858eb4b93ee63585720ec6cad9a5d49aea3ed21a0cf55b0fd5.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/ca5a8d0bb48221858eb4b93ee63585720ec6cad9a5d49aea3ed21a0cf55b0fd5.jpg)

![d0beef7cddbc383ff3f391a37b425a11bf3dfa11cdfbbd16cf2ef1575d640afa.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/d0beef7cddbc383ff3f391a37b425a11bf3dfa11cdfbbd16cf2ef1575d640afa.jpg)

![e24ef464fb89b9a9e3d7da87e5c22536faca4065031a66d56c10e0765b68d16e.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/e24ef464fb89b9a9e3d7da87e5c22536faca4065031a66d56c10e0765b68d16e.jpg)

![f25eb68962cd18397f17bcee24297abd14dae25b80a18ef675fd26cb7a1e105a.jpg](../icml_results/926_The%20Surprising%20Difficulty%20of%20Search%20in%20Model-Based%20Reinforcement%20Learning/tables/f25eb68962cd18397f17bcee24297abd14dae25b80a18ef675fd26cb7a1e105a.jpg)

<a id="paper-0910"></a>

## MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models


### Images

![2bd1f2caaa960399de5494a83eb07ba1efcbbc1cabc92f82f95a8097b0b58796.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/images/2bd1f2caaa960399de5494a83eb07ba1efcbbc1cabc92f82f95a8097b0b58796.jpg)

![a3907f026c7bd76f6e4d528d456a3e181b87fcc63a079d16254797acbe822d2d.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/images/a3907f026c7bd76f6e4d528d456a3e181b87fcc63a079d16254797acbe822d2d.jpg)

![f3826683691d2ec3dc27252b45884764fd676f96be83fdd170f9c9477577c7f5.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/images/f3826683691d2ec3dc27252b45884764fd676f96be83fdd170f9c9477577c7f5.jpg)

![f7ad3272f3c13b1d0dc8631140689af0f9f821bf447d3be545fa9c1e0509a260.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/images/f7ad3272f3c13b1d0dc8631140689af0f9f821bf447d3be545fa9c1e0509a260.jpg)

### Tables

![0bad7ea4a4ae5e3feb54428938bebb28ba6ac0432eb425b03cf2f6cf702a661c.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/0bad7ea4a4ae5e3feb54428938bebb28ba6ac0432eb425b03cf2f6cf702a661c.jpg)

![121722bd740e498220eecc183539712f059ba4173ed4f00ec638607a2b1c1a31.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/121722bd740e498220eecc183539712f059ba4173ed4f00ec638607a2b1c1a31.jpg)

![23d50fc7bca09a884703a661c50934afd3451ec4181ced600c628812519cc937.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/23d50fc7bca09a884703a661c50934afd3451ec4181ced600c628812519cc937.jpg)

![32cb0c3c3e7f70b5feb730fcb3a545b5062834f146f26daa0a3171ab2b146f6c.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/32cb0c3c3e7f70b5feb730fcb3a545b5062834f146f26daa0a3171ab2b146f6c.jpg)

![6a696b732be2ef77cc8eea861a69442b3bca415573aa09a27ac64855779defc3.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/6a696b732be2ef77cc8eea861a69442b3bca415573aa09a27ac64855779defc3.jpg)

![72f4132baad27e06012709f7c6152e7e0b86c9fe35a086ccfdbf1fb005f1bf9d.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/72f4132baad27e06012709f7c6152e7e0b86c9fe35a086ccfdbf1fb005f1bf9d.jpg)

![964b2a106131c0d4cbe692178ab570e660952d7768ba013d0f1844805180b774.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/964b2a106131c0d4cbe692178ab570e660952d7768ba013d0f1844805180b774.jpg)

![acdc463d6601b7afdd30264d96261f4438217f565c1713e2c42d07a4e86fdffe.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/acdc463d6601b7afdd30264d96261f4438217f565c1713e2c42d07a4e86fdffe.jpg)

![bc558624a45871442312e5694f070eb678b2e7230e8716bae93f220042ca8dbf.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/bc558624a45871442312e5694f070eb678b2e7230e8716bae93f220042ca8dbf.jpg)

![bf8d5d885fffd6fe2ac23a6c6ed0793291ef41c2e4c57f42810060410a6d5ac2.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/bf8d5d885fffd6fe2ac23a6c6ed0793291ef41c2e4c57f42810060410a6d5ac2.jpg)

![df0ea21c893b1a72b051157ebd04bdef76dbd18a72cd4e14f3614d0e570ad4b8.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/df0ea21c893b1a72b051157ebd04bdef76dbd18a72cd4e14f3614d0e570ad4b8.jpg)

![eeb91ead0a971aad71de53e799fe1b47ca6716ff8effa444ca1fc5cece3eb903.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/eeb91ead0a971aad71de53e799fe1b47ca6716ff8effa444ca1fc5cece3eb903.jpg)

![f632f7f288e1a725219b7545bb3c73a45f6830bf75d024af7651e12f04e9edff.jpg](../icml_results/927_MedSIGHT_%20Towards%20Grounded%20Visual%20Comprehension%20in%20Medical%20Large%20Vision-Language%20Models/tables/f632f7f288e1a725219b7545bb3c73a45f6830bf75d024af7651e12f04e9edff.jpg)

<a id="paper-0911"></a>

## ContrastiveCFG: Guiding Diffusion Sampling by Contrasting Positive and Negative Concepts


### Images

![0fda4e08e712eb2c1355a8763ce8656379e9eef5f9714dca908fbcc76a1d0dd6.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/0fda4e08e712eb2c1355a8763ce8656379e9eef5f9714dca908fbcc76a1d0dd6.jpg)

![31c06961d880ca35a62549903015d089606fd658cf4a707495aa62a5e0def61d.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/31c06961d880ca35a62549903015d089606fd658cf4a707495aa62a5e0def61d.jpg)

![3c8217ba5da19b13a947641104654a8f74fd967c270ef2bccb28e39a0c8dcb21.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/3c8217ba5da19b13a947641104654a8f74fd967c270ef2bccb28e39a0c8dcb21.jpg)

![3fc59412e1482dc053b98e5f92f2f24872c15558da2f93d78386d5d1f0c750f4.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/3fc59412e1482dc053b98e5f92f2f24872c15558da2f93d78386d5d1f0c750f4.jpg)

![66c4db2c6d014bb9055729675225a5db9ed368ba4bb248cb99212e6f986c4676.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/66c4db2c6d014bb9055729675225a5db9ed368ba4bb248cb99212e6f986c4676.jpg)

![811a2d92f17b88b335fed8168487e036f9328e30016bbc4b9555854e2531b36e.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/811a2d92f17b88b335fed8168487e036f9328e30016bbc4b9555854e2531b36e.jpg)

![9d27fa28485ba15e691f85e1028daafa56700eb15845e270578ed41ecb6a2da0.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/9d27fa28485ba15e691f85e1028daafa56700eb15845e270578ed41ecb6a2da0.jpg)

![d3c2a925c01f93a60afe692f0714b736e2c2d83afec493f48f161f6a4557b879.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/d3c2a925c01f93a60afe692f0714b736e2c2d83afec493f48f161f6a4557b879.jpg)

![ec22a2c3f4b9a6eb3d1fda89e659c27d25e324eecc1f6f56b3af55912c33270d.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/ec22a2c3f4b9a6eb3d1fda89e659c27d25e324eecc1f6f56b3af55912c33270d.jpg)

![ee83f676fd5ca3d06441be8da302b13e1d46b8f6da191b37464ee972a2ae51e6.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/ee83f676fd5ca3d06441be8da302b13e1d46b8f6da191b37464ee972a2ae51e6.jpg)

![f2931e783bbedb5b1b09147189993e44f2c3e35f4bda86d8b0cb0bf022a6388a.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/f2931e783bbedb5b1b09147189993e44f2c3e35f4bda86d8b0cb0bf022a6388a.jpg)

![f93329266348f810764369f1627f135f5cafce66ebdd83aaabd95374c2a147e5.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/images/f93329266348f810764369f1627f135f5cafce66ebdd83aaabd95374c2a147e5.jpg)

### Tables

![172810e93079bbaeda8ca679fa45fe0acbfc1ccc40234715fb82e66c16416312.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/172810e93079bbaeda8ca679fa45fe0acbfc1ccc40234715fb82e66c16416312.jpg)

![525087fddf905128574049a22abac46e0d17ef72fa7dc403b5a897bc38c53779.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/525087fddf905128574049a22abac46e0d17ef72fa7dc403b5a897bc38c53779.jpg)

![5d707f64d3a4551c557d1104b17a09da06a9221287035f991eabd08a00712751.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/5d707f64d3a4551c557d1104b17a09da06a9221287035f991eabd08a00712751.jpg)

![c44e41c109f0e9f7e0a9cab433d4360f034ca64938c1ce1b03e4608c91373007.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/c44e41c109f0e9f7e0a9cab433d4360f034ca64938c1ce1b03e4608c91373007.jpg)

![c9d8cbffa289a2a5eb50044c1076f0d421f63425a970d03f798b0e79399848a1.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/c9d8cbffa289a2a5eb50044c1076f0d421f63425a970d03f798b0e79399848a1.jpg)

![d7102b265a05d9b859b4c9ea00f5c2a271403d29961e714dc83fb0556a15cf51.jpg](../icml_results/928_ContrastiveCFG_%20Guiding%20Diffusion%20Sampling%20by%20Contrasting%20Positive%20and%20Negative%20Concepts/tables/d7102b265a05d9b859b4c9ea00f5c2a271403d29961e714dc83fb0556a15cf51.jpg)

<a id="paper-0912"></a>

## xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction


### Images

![15ac5bec97a1c97bc0b3a062714bf161b265db70ee820067c8961fa545b8432b.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/15ac5bec97a1c97bc0b3a062714bf161b265db70ee820067c8961fa545b8432b.jpg)

![3896baa8ad8588b79b8524ea616ff9226824107b6ec6e0b02232fc2a213cb16e.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/3896baa8ad8588b79b8524ea616ff9226824107b6ec6e0b02232fc2a213cb16e.jpg)

![3f0bf5c59040d260ec30fa1ffe8145bc9154346114d81a079aa226959ed91028.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/3f0bf5c59040d260ec30fa1ffe8145bc9154346114d81a079aa226959ed91028.jpg)

![659724b007ce113d7b7331c840e35f80a8e83a31ce50f19333b04bc5590e5209.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/659724b007ce113d7b7331c840e35f80a8e83a31ce50f19333b04bc5590e5209.jpg)

![6f3966ae5c8e4179046aea0c415f71ae0a4c89d1dfc8087665a95bbe2f247e96.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/6f3966ae5c8e4179046aea0c415f71ae0a4c89d1dfc8087665a95bbe2f247e96.jpg)

![a257ebda4be2796d80e7f360bfe1f4e5c2cba0a521747dae5377faa0fe133eca.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/a257ebda4be2796d80e7f360bfe1f4e5c2cba0a521747dae5377faa0fe133eca.jpg)

![ac540b4e9ead625c5cf4683109a138f42a66c36d4eeefc1a2f153aceb9109d1c.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/ac540b4e9ead625c5cf4683109a138f42a66c36d4eeefc1a2f153aceb9109d1c.jpg)

![c94d555c17e5f76762dcbad7eae4aa20441a2739cc2a5b4e9354500911fdb583.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/images/c94d555c17e5f76762dcbad7eae4aa20441a2739cc2a5b4e9354500911fdb583.jpg)

### Tables

![02141af97f30cb1338ad5843b041d7b88e4ebbfbd154e2fda8aeb3de89591022.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/02141af97f30cb1338ad5843b041d7b88e4ebbfbd154e2fda8aeb3de89591022.jpg)

![4ea98fb4608eab90eb04af0ff67e5e9a067d38e7c2cc309aa0ca7a0512a0bc62.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/4ea98fb4608eab90eb04af0ff67e5e9a067d38e7c2cc309aa0ca7a0512a0bc62.jpg)

![517f377a78bb44ba559a74cf2caa652ea7630c2cd3cd756c59fc8d2eb9ea89fd.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/517f377a78bb44ba559a74cf2caa652ea7630c2cd3cd756c59fc8d2eb9ea89fd.jpg)

![8cf38845e0eea3cb324bfdc9ee96af6cc9bc195f26f944571953a956934b8bb5.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/8cf38845e0eea3cb324bfdc9ee96af6cc9bc195f26f944571953a956934b8bb5.jpg)

![8d22c57d7d23a6c0e2a5f078c80c7c4f29c73c0f68412816e0f732488c7a5f8b.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/8d22c57d7d23a6c0e2a5f078c80c7c4f29c73c0f68412816e0f732488c7a5f8b.jpg)

![8f5b4952571c83fc0233f45ecbe311cb40dbae28281c91f58c3d111ce22e7a06.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/8f5b4952571c83fc0233f45ecbe311cb40dbae28281c91f58c3d111ce22e7a06.jpg)

![a2cf4235a29dea5ee24f785c9c44402987f6f56005f5245fee42191dd83aa760.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/a2cf4235a29dea5ee24f785c9c44402987f6f56005f5245fee42191dd83aa760.jpg)

![c8a241a70eda12197369a4de269eb35531e5c42d68281894ed96e7ffd1c4b0ba.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/c8a241a70eda12197369a4de269eb35531e5c42d68281894ed96e7ffd1c4b0ba.jpg)

![d559252a9083c8102e05514095c5341712ccb08c51505644fbb70bd4d904ea75.jpg](../icml_results/929_xKV_%20Cross-Layer%20KV-Cache%20Compression%20via%20Aligned%20Singular%20Vector%20Extraction/tables/d559252a9083c8102e05514095c5341712ccb08c51505644fbb70bd4d904ea75.jpg)

<a id="paper-0913"></a>

## How2Everything: Mining the Web for How-to Procedures to Evaluate and Improve LLMs


### Images

![28f47f59f8b3c7e702b1131d25e443adca5c4cc7988e126fd04686398fce2bf1.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/28f47f59f8b3c7e702b1131d25e443adca5c4cc7988e126fd04686398fce2bf1.jpg)

![380025cd5098b5984775903edfb51ddbb3098e6c86ebc2a314e88af9017f9327.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/380025cd5098b5984775903edfb51ddbb3098e6c86ebc2a314e88af9017f9327.jpg)

![3e04b1a079aab4857f4d4b01bdb7cb5037d976764e3d7e54d4823b0d25c20e69.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/3e04b1a079aab4857f4d4b01bdb7cb5037d976764e3d7e54d4823b0d25c20e69.jpg)

![40ecc32915cbb3011f2e7970ce9ebd0ae648634b2dc32f70501b5ba66d7a4773.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/40ecc32915cbb3011f2e7970ce9ebd0ae648634b2dc32f70501b5ba66d7a4773.jpg)

![4ddffba04aadeaf25b1990b69d1705b1a283b12facbafc0dc82971d9f8362623.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/4ddffba04aadeaf25b1990b69d1705b1a283b12facbafc0dc82971d9f8362623.jpg)

![5826baed8f5da490cb1945e8a247c19b41bd78bc20e938653198e726855a8c4b.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/5826baed8f5da490cb1945e8a247c19b41bd78bc20e938653198e726855a8c4b.jpg)

![5dd46ae5bd0df5a40249842733c36fdc16d98b26d2aa5f440c88cc10099a1ed0.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/5dd46ae5bd0df5a40249842733c36fdc16d98b26d2aa5f440c88cc10099a1ed0.jpg)

![705b36531e98396ed3edf59cef68a10a7e6e10ca9e48a282e6072bc059344c41.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/705b36531e98396ed3edf59cef68a10a7e6e10ca9e48a282e6072bc059344c41.jpg)

![71658dd7be43be4a9b8ce9cc75712d8a3220ef4d4f1db8f1b26d2f4dd4d16e76.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/71658dd7be43be4a9b8ce9cc75712d8a3220ef4d4f1db8f1b26d2f4dd4d16e76.jpg)

![7a1d986f93cf73be17c3b19972fa9348facbb7ae6c65412eb8685e959fef3a9a.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/7a1d986f93cf73be17c3b19972fa9348facbb7ae6c65412eb8685e959fef3a9a.jpg)

![91579872ba5227c63f56c3773f7c85508622d20cecfcde9be23fa07cc1bf388e.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/91579872ba5227c63f56c3773f7c85508622d20cecfcde9be23fa07cc1bf388e.jpg)

![9cff85fa55756f88aef6acf39b90092c04bc5317890426072ef7ea2de9afff73.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/9cff85fa55756f88aef6acf39b90092c04bc5317890426072ef7ea2de9afff73.jpg)

![aa241a70fa8144eb00fbe9bd3d80f7aeb12ec5f72c1f15abf1d9b4981e39902b.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/aa241a70fa8144eb00fbe9bd3d80f7aeb12ec5f72c1f15abf1d9b4981e39902b.jpg)

![af3993aae1173dcad59b226480ff44e8c6a0e7a0cf43b2a3ca1f7509b7c8f04a.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/af3993aae1173dcad59b226480ff44e8c6a0e7a0cf43b2a3ca1f7509b7c8f04a.jpg)

![b336718743011853f9e57f2fd2ee927763982f38cd16981a174be7328fde69e6.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/b336718743011853f9e57f2fd2ee927763982f38cd16981a174be7328fde69e6.jpg)

![d65f431eae439839872ecaa4fbea2e7b8fa101dda42452b5bd731e8ba375d18f.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/d65f431eae439839872ecaa4fbea2e7b8fa101dda42452b5bd731e8ba375d18f.jpg)

![fcb159dcf2673acc33419f7feddd2788f7b0717d02ad7b3f9e55ecd15cb32e8b.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/images/fcb159dcf2673acc33419f7feddd2788f7b0717d02ad7b3f9e55ecd15cb32e8b.jpg)

### Tables

![0d01da4f9f2728838cd5e59b086cb833c509906cae48355ca111e9d6c15a2f8e.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/0d01da4f9f2728838cd5e59b086cb833c509906cae48355ca111e9d6c15a2f8e.jpg)

![0d5c2c26712a555907efc4165a8d8080370ad2c0539edfccb3ddaa377118a248.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/0d5c2c26712a555907efc4165a8d8080370ad2c0539edfccb3ddaa377118a248.jpg)

![15e49b70e00b06c07ec11afab039c1619fabc08f85dfc023b34a7334191bb6ca.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/15e49b70e00b06c07ec11afab039c1619fabc08f85dfc023b34a7334191bb6ca.jpg)

![16e62917c70882e3d6859705a59594b203c7c37ca5bc6c785aa4f5f3938880cd.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/16e62917c70882e3d6859705a59594b203c7c37ca5bc6c785aa4f5f3938880cd.jpg)

![294a5a4835ddd22b157c5606143875369411e87883a533dfb5791654b18dd391.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/294a5a4835ddd22b157c5606143875369411e87883a533dfb5791654b18dd391.jpg)

![297085a1a043c9737b5788fca3b540cfb6409c6a3dc4caabba0ccdfb23e5d780.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/297085a1a043c9737b5788fca3b540cfb6409c6a3dc4caabba0ccdfb23e5d780.jpg)

![36aba61d3952a457546bf727dd8c1019dce98e34e39db39457da64e0ffb18b9c.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/36aba61d3952a457546bf727dd8c1019dce98e34e39db39457da64e0ffb18b9c.jpg)

![65709fcd1cc97e8362ef185fc10796322b4b17c2e4dea43d98ff158628259e8b.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/65709fcd1cc97e8362ef185fc10796322b4b17c2e4dea43d98ff158628259e8b.jpg)

![6b85b4a5863a3930592da1acbd6ea3463eaa4cedb14eb26016709ac19db301df.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/6b85b4a5863a3930592da1acbd6ea3463eaa4cedb14eb26016709ac19db301df.jpg)

![82071752e2de4114d843b24823ed4e4eb6d4b649bcae5c9c8bc4783e7f780bb9.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/82071752e2de4114d843b24823ed4e4eb6d4b649bcae5c9c8bc4783e7f780bb9.jpg)

![82755c5d3ea6b7d404d5286971b23a732eeecbaee9f88c47c9a403381a73c17d.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/82755c5d3ea6b7d404d5286971b23a732eeecbaee9f88c47c9a403381a73c17d.jpg)

![91b1c7d25f33ebe497a453540007f7816bb15e7290b3e818042d55bf290cbe02.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/91b1c7d25f33ebe497a453540007f7816bb15e7290b3e818042d55bf290cbe02.jpg)

![a1018570dc2a04966e3e25d0776e05e2a15366105a119f45cb17a19978b62904.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/a1018570dc2a04966e3e25d0776e05e2a15366105a119f45cb17a19978b62904.jpg)

![a2c3ce24bac0ad7a75b5692ea39ec00d8d2ef192dc77670171d5f52d8d25baf6.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/a2c3ce24bac0ad7a75b5692ea39ec00d8d2ef192dc77670171d5f52d8d25baf6.jpg)

![ae8142180929abc886531b90cf0fff6167acf80653ba0075374289de2e91484c.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/ae8142180929abc886531b90cf0fff6167acf80653ba0075374289de2e91484c.jpg)

![b9e9a5163362551ea4c0a60320949be7d871a0907d25231002827768de6eadeb.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/b9e9a5163362551ea4c0a60320949be7d871a0907d25231002827768de6eadeb.jpg)

![be4285189e40d780521f31fd8491c87ad44523343a8e786bd1970ebe4f6b6c62.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/be4285189e40d780521f31fd8491c87ad44523343a8e786bd1970ebe4f6b6c62.jpg)

![c8685a298f90ef718c90de81df7ef83a18a3000aff15072c07195690473ef74e.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/c8685a298f90ef718c90de81df7ef83a18a3000aff15072c07195690473ef74e.jpg)

![db3f0414b21bf346dbd6f449d19d2525409f0adfa5bffe4443cc47467097a0ca.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/db3f0414b21bf346dbd6f449d19d2525409f0adfa5bffe4443cc47467097a0ca.jpg)

![ec915e5fa298441266af46b6d4d116e935ba6b37a90c5273c1bd19db2b3e0ca0.jpg](../icml_results/930_How2Everything_%20Mining%20the%20Web%20for%20How-to%20Procedures%20to%20Evaluate%20and%20Improve%20LLMs/tables/ec915e5fa298441266af46b6d4d116e935ba6b37a90c5273c1bd19db2b3e0ca0.jpg)

<a id="paper-0914"></a>

## RetrOrchestrator: A Multi-Step Retrosynthesis Agent Dynamically Orchestrating Single-Step Transition Models


### Images

![0cbd329a0a1050c73f61c8d990739552fda4629216b20db159c31e5574c05a91.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/0cbd329a0a1050c73f61c8d990739552fda4629216b20db159c31e5574c05a91.jpg)

![18ea5879b429623a4cc2584fd3a467828e776d08cd016540120e871a215b11d3.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/18ea5879b429623a4cc2584fd3a467828e776d08cd016540120e871a215b11d3.jpg)

![3bb574526eb22670f7a41efaea2b30c6eaf2d87f709b6f37ce9402d754138a1f.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/3bb574526eb22670f7a41efaea2b30c6eaf2d87f709b6f37ce9402d754138a1f.jpg)

![3efeb187cdd0c4265c96cd2fb8e3538d9fcd6083f7899fa095fab1c033cdfbc0.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/3efeb187cdd0c4265c96cd2fb8e3538d9fcd6083f7899fa095fab1c033cdfbc0.jpg)

![6a3544f8803c431a0f2e2479455fe1c75f51f2aea770618b85de3dc752ebb37a.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/6a3544f8803c431a0f2e2479455fe1c75f51f2aea770618b85de3dc752ebb37a.jpg)

![6f36d8f1bb50104c93e17f94f88387bab4207820907e06f9059593a84d11dd55.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/6f36d8f1bb50104c93e17f94f88387bab4207820907e06f9059593a84d11dd55.jpg)

![71eaa795fe83cdefccc4621a1da7f081be47660cd37bd4f50823bc8f925db8f1.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/71eaa795fe83cdefccc4621a1da7f081be47660cd37bd4f50823bc8f925db8f1.jpg)

![77c2d9343aad455eb133b5515c94b8cb095746a101dcfb1298b06a0c0a918221.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/77c2d9343aad455eb133b5515c94b8cb095746a101dcfb1298b06a0c0a918221.jpg)

![794f3db95aba427e299367c1e0ab2e5094c0bdae8687fe2e610a4b674206fe6e.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/794f3db95aba427e299367c1e0ab2e5094c0bdae8687fe2e610a4b674206fe6e.jpg)

![7e76f9e45118eff86d0f8baac32794a9b06ece0b6da11bc615bfaf81a4209a74.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/7e76f9e45118eff86d0f8baac32794a9b06ece0b6da11bc615bfaf81a4209a74.jpg)

![8a8388990aba0ee5a3d58103a9f6c48d4bc389ffe86fa34b92602fc32f90da77.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/8a8388990aba0ee5a3d58103a9f6c48d4bc389ffe86fa34b92602fc32f90da77.jpg)

![8bd6795f9767acf4f8fe5985777565b8cb0bb2ae020e51fcc45dd2c0b29bd550.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/8bd6795f9767acf4f8fe5985777565b8cb0bb2ae020e51fcc45dd2c0b29bd550.jpg)

![8cadbe8eead1aa7bd916d97a2d414562257af8b9fe3b8d50a3fd08b083d9cc3b.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/8cadbe8eead1aa7bd916d97a2d414562257af8b9fe3b8d50a3fd08b083d9cc3b.jpg)

![931671e4d613ec9237a844e7a0b818567601ae62ead9e8aeb44d30e721b4e42b.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/931671e4d613ec9237a844e7a0b818567601ae62ead9e8aeb44d30e721b4e42b.jpg)

![978738b3bb03dad384bba0f4e7e972b45f56c315008364e2801df0ef23b598db.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/978738b3bb03dad384bba0f4e7e972b45f56c315008364e2801df0ef23b598db.jpg)

![9fe3f7803499dc5678f92418d5530b472f9be7cd42810aebb689c641c8e1668c.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/9fe3f7803499dc5678f92418d5530b472f9be7cd42810aebb689c641c8e1668c.jpg)

![ad2b897b076d2c5f3b8ea01c7305a65908bd29c47f0fcc3477d60dcfbde98955.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/ad2b897b076d2c5f3b8ea01c7305a65908bd29c47f0fcc3477d60dcfbde98955.jpg)

![c7c99b40c305544d325c78ddacdbd4c7a49ba241e6aad9df13b7fc263214e7e6.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/c7c99b40c305544d325c78ddacdbd4c7a49ba241e6aad9df13b7fc263214e7e6.jpg)

![caf537838572c9a98ee55faff85a48f679abb460b92d38084ce2bb7d6de93c7b.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/caf537838572c9a98ee55faff85a48f679abb460b92d38084ce2bb7d6de93c7b.jpg)

![cb7c6c7e9a34461e6336f74055443e9a7c3b51fe450e2cee1a1287aa69c33f73.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/cb7c6c7e9a34461e6336f74055443e9a7c3b51fe450e2cee1a1287aa69c33f73.jpg)

![cc9ddc9df4bd92e49279cd269b1bf0574edd4ae1d09bcd7ac88d9f625b865d44.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/cc9ddc9df4bd92e49279cd269b1bf0574edd4ae1d09bcd7ac88d9f625b865d44.jpg)

![cdb4f751c9b03e35eb2e6c752d424a7535f37a5fb2a8582c90a76b7c4faa66ef.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/cdb4f751c9b03e35eb2e6c752d424a7535f37a5fb2a8582c90a76b7c4faa66ef.jpg)

![ce4463d95ba207a41f79be27934b0796bc6ae0a362b4a1fd0af750f9cfe3d23a.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/ce4463d95ba207a41f79be27934b0796bc6ae0a362b4a1fd0af750f9cfe3d23a.jpg)

![db0362e28c4cfe157f40de843575d79e026dfee35d7932df6cf627bd45cbfd06.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/db0362e28c4cfe157f40de843575d79e026dfee35d7932df6cf627bd45cbfd06.jpg)

![ea5c5be6b5347ad49d08b7fc9b691fb662d9c68be428ca5940a921e6f939654a.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/images/ea5c5be6b5347ad49d08b7fc9b691fb662d9c68be428ca5940a921e6f939654a.jpg)

### Tables

![16cdf8f177d7bf67ce69cb654f67199ac70f94c545906e7ac59ad6f31c751342.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/16cdf8f177d7bf67ce69cb654f67199ac70f94c545906e7ac59ad6f31c751342.jpg)

![2eebb27eeead56f5b06debe71841cc68bbb234bc96d35bc63a0a2e6741965e07.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/2eebb27eeead56f5b06debe71841cc68bbb234bc96d35bc63a0a2e6741965e07.jpg)

![3724cb26e9b55cbac40d22ac143fae8207cc687643192a69b00e146a4bd35439.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/3724cb26e9b55cbac40d22ac143fae8207cc687643192a69b00e146a4bd35439.jpg)

![4ae3fedd015ff71c9acd5ec847f98b7def018bbbd238763badcce05b16441e96.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/4ae3fedd015ff71c9acd5ec847f98b7def018bbbd238763badcce05b16441e96.jpg)

![5be545309859fbc22a2970185f5d9dd21599e68cf60d05e42e5f8ee111769634.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/5be545309859fbc22a2970185f5d9dd21599e68cf60d05e42e5f8ee111769634.jpg)

![6129da002a111b403efec368230d0865894226b40dd345834988b3154cd22667.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/6129da002a111b403efec368230d0865894226b40dd345834988b3154cd22667.jpg)

![6bf0de21cb732dff9bf5b733165f1f8994a8beeab7087fa803c815b5cb0b7c99.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/6bf0de21cb732dff9bf5b733165f1f8994a8beeab7087fa803c815b5cb0b7c99.jpg)

![92e5ea74d7fca039c889078cdbcc2f784bcc991d63205b5decadc224684f19d0.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/92e5ea74d7fca039c889078cdbcc2f784bcc991d63205b5decadc224684f19d0.jpg)

![b786a6e398ad95544d3f1463d4b580583d44d0cffebb71dc3c635261a8710cf9.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/b786a6e398ad95544d3f1463d4b580583d44d0cffebb71dc3c635261a8710cf9.jpg)

![c8697a3fc3c5451da11fa31234669765a39a2bd52159a80c907eda7103f8ee0e.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/c8697a3fc3c5451da11fa31234669765a39a2bd52159a80c907eda7103f8ee0e.jpg)

![d884d402404ed9500c39333b74abc9145935dfd920ce9b70be28683461762249.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/d884d402404ed9500c39333b74abc9145935dfd920ce9b70be28683461762249.jpg)

![e3aefc196888fbcb0a486cbf5f9a6bce69af8ab56184c9681b27c04795a7c3f9.jpg](../icml_results/931_RetrOrchestrator_%20A%20Multi-Step%20Retrosynthesis%20Agent%20Dynamically%20Orchestrating%20Single-Step%20Transition/tables/e3aefc196888fbcb0a486cbf5f9a6bce69af8ab56184c9681b27c04795a7c3f9.jpg)

<a id="paper-0915"></a>

## Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders


### Images

![0b0a00e9ef3aaf8b271790e28bdb6137d15da40f6c3456e5208a9c706f1a75a6.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/0b0a00e9ef3aaf8b271790e28bdb6137d15da40f6c3456e5208a9c706f1a75a6.jpg)

![0ea6f92aa971fe10962fd3d4589562755f3e5c54b2dcaa302551acee2f9144ff.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/0ea6f92aa971fe10962fd3d4589562755f3e5c54b2dcaa302551acee2f9144ff.jpg)

![1195d9290394921ff34d8628365d775997e99d59d3a64ab6491d0c9bb004d4c7.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/1195d9290394921ff34d8628365d775997e99d59d3a64ab6491d0c9bb004d4c7.jpg)

![3878cc84db92ec380cf1959c0f8d7fe074ddd5bc704698d6761f9147f8997a68.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/3878cc84db92ec380cf1959c0f8d7fe074ddd5bc704698d6761f9147f8997a68.jpg)

![437029fe18977c15ef0602c36dfa9baa05c8736f38730a39c10b5e4f56e7a9e5.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/437029fe18977c15ef0602c36dfa9baa05c8736f38730a39c10b5e4f56e7a9e5.jpg)

![43cb6f3b1e8edb6203a76530d9c5aea31318fe2959faf4df85c3a4110e7cf00f.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/43cb6f3b1e8edb6203a76530d9c5aea31318fe2959faf4df85c3a4110e7cf00f.jpg)

![4de0be9ae65f903abea670c7edfd73212389623906df3569fa5b388428a5f480.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/4de0be9ae65f903abea670c7edfd73212389623906df3569fa5b388428a5f480.jpg)

![4eeeb54063df90b71973f97da038c70ffcb76829adfb0b4867d4a793284d0650.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/4eeeb54063df90b71973f97da038c70ffcb76829adfb0b4867d4a793284d0650.jpg)

![5b9e6061d669735ba72dc282862e7b3cfdce19877c557b1701d609f832627a6e.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/5b9e6061d669735ba72dc282862e7b3cfdce19877c557b1701d609f832627a6e.jpg)

![5f928bda2edd35eb9f14a9d26e1d7f93b9fe238b19ede3f938c4929b3e087dce.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/5f928bda2edd35eb9f14a9d26e1d7f93b9fe238b19ede3f938c4929b3e087dce.jpg)

![63d6b04074bd76803c8098317ede440430819115e95c988030d9044efbd5ca4e.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/63d6b04074bd76803c8098317ede440430819115e95c988030d9044efbd5ca4e.jpg)

![6e411914e48defb6f8fbc342ee0ea090acabd9cf5b601c0452ac526bf956c45d.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/6e411914e48defb6f8fbc342ee0ea090acabd9cf5b601c0452ac526bf956c45d.jpg)

![8d93518fb5bfca4d8c2696b6c6f4f9a3b3b0a9d5a0f926fd378928d09f6f6990.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/8d93518fb5bfca4d8c2696b6c6f4f9a3b3b0a9d5a0f926fd378928d09f6f6990.jpg)

![8edcee3c03e06a476cdfedfa2daafbdb3b97b149293a3d63031ac6941288a65f.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/8edcee3c03e06a476cdfedfa2daafbdb3b97b149293a3d63031ac6941288a65f.jpg)

![a918459ec3baa9176b93c3068f9fa1825602e0354f77d5fca2c0bd0934278ca7.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/a918459ec3baa9176b93c3068f9fa1825602e0354f77d5fca2c0bd0934278ca7.jpg)

![abb2187d08df23bf280d7aad870465705d1e825a6cf2cfd8e201810ed6835543.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/abb2187d08df23bf280d7aad870465705d1e825a6cf2cfd8e201810ed6835543.jpg)

![b0872e811e9f2ba6724fbd2b6c93a2c32125c0d01e1b55cb5dda226e9e366eb4.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/b0872e811e9f2ba6724fbd2b6c93a2c32125c0d01e1b55cb5dda226e9e366eb4.jpg)

![bb2f8f70c52c2d7baff5088d4f6cd54e2a071f32bdb6e272d084d347f6004ac4.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/bb2f8f70c52c2d7baff5088d4f6cd54e2a071f32bdb6e272d084d347f6004ac4.jpg)

![e39ecb58ce6d1242e257d343412e111012fab6dd966ceb1bd189f50a2d66855a.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/e39ecb58ce6d1242e257d343412e111012fab6dd966ceb1bd189f50a2d66855a.jpg)

![e4fe57b133caa71d4750c614152cf81ff67aecd16ab72011814a7c29ff8494ba.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/e4fe57b133caa71d4750c614152cf81ff67aecd16ab72011814a7c29ff8494ba.jpg)

![e572ebf0ef7b41e7b5289d6d67add9357cad8f7555543ef168ac4fc3dc733982.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/e572ebf0ef7b41e7b5289d6d67add9357cad8f7555543ef168ac4fc3dc733982.jpg)

![e63336541a7867c016f94d544e9b1f106aab688f92b147c50d9290fd80bf0b91.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/e63336541a7867c016f94d544e9b1f106aab688f92b147c50d9290fd80bf0b91.jpg)

![ea12c3bff174b136c933dc05ad20076e4b4bc3585160e00a56b1f217264bdef6.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/ea12c3bff174b136c933dc05ad20076e4b4bc3585160e00a56b1f217264bdef6.jpg)

![eb9591e720aa5468e461164833b6d11c8929526701b3dd8557dcca1908eb50ae.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/eb9591e720aa5468e461164833b6d11c8929526701b3dd8557dcca1908eb50ae.jpg)

![ebe12b3cb1c09003cef235d976dd7369441d5e68bdf0982ace780eb90dcfb5f8.jpg](../icml_results/932_Sparse%20but%20Wrong_%20Incorrect%20L0%20Leads%20to%20Incorrect%20Features%20in%20Sparse%20Autoencoders/images/ebe12b3cb1c09003cef235d976dd7369441d5e68bdf0982ace780eb90dcfb5f8.jpg)

<a id="paper-0916"></a>

## Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problem


### Images

![25f1f2a825c974d25f5ea2ad913ffecb2f98280958be69bc5e546307f93069ec.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/25f1f2a825c974d25f5ea2ad913ffecb2f98280958be69bc5e546307f93069ec.jpg)

![3e92df52e55bfa558e10ebfefb2eff129ba43866745e77bafbdeabdfbad74f27.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/3e92df52e55bfa558e10ebfefb2eff129ba43866745e77bafbdeabdfbad74f27.jpg)

![4ab90f368ae0519b47bf27ac768cfe3dd9f35807acb447e430004fe18df60323.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/4ab90f368ae0519b47bf27ac768cfe3dd9f35807acb447e430004fe18df60323.jpg)

![78b7c9f38f34614f986df3239a1e97e9caf5cfaae5d732fdac8fd88ba7981b0b.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/78b7c9f38f34614f986df3239a1e97e9caf5cfaae5d732fdac8fd88ba7981b0b.jpg)

![a7a34454fb1cbff3b0d075728b443e23092225d982fc85b1ace2cd10943013ea.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/a7a34454fb1cbff3b0d075728b443e23092225d982fc85b1ace2cd10943013ea.jpg)

![bf51fefba045bf0e98e8db590158c346a67563515094509e410cb6ad769cb92f.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/bf51fefba045bf0e98e8db590158c346a67563515094509e410cb6ad769cb92f.jpg)

![e38a0ca1f0396c3445e4009174f4e58c3c972e890a02910401bbed9661e6a7b8.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/images/e38a0ca1f0396c3445e4009174f4e58c3c972e890a02910401bbed9661e6a7b8.jpg)

### Tables

![243614bb8631ae5c0a5b104ec89c001ada68238a6d9b59c6900255a40c6b76d0.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/tables/243614bb8631ae5c0a5b104ec89c001ada68238a6d9b59c6900255a40c6b76d0.jpg)

![a6384687e2dad7167c1f76ce7799207d5e5622a572ca13ffc64cebbafd41dec8.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/tables/a6384687e2dad7167c1f76ce7799207d5e5622a572ca13ffc64cebbafd41dec8.jpg)

![a75f38c8d89afdcdd8cccc15c6d71a064453aeb72d05d2c0271fd9de4d213c36.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/tables/a75f38c8d89afdcdd8cccc15c6d71a064453aeb72d05d2c0271fd9de4d213c36.jpg)

![f608638b845a4d11cce23e9ab52e867dffca9d94042a29d8d224a428a676fd79.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/tables/f608638b845a4d11cce23e9ab52e867dffca9d94042a29d8d224a428a676fd79.jpg)

![fb371829e265062c50a3539357f30ab2ed7cb5b751ec58e4102f0d4a321067f8.jpg](../icml_results/933_Last-iterate%20Convergence%20of%20ADMM%20on%20Multi-affine%20Quadratic%20Equality%20Constrained%20Problem/tables/fb371829e265062c50a3539357f30ab2ed7cb5b751ec58e4102f0d4a321067f8.jpg)

<a id="paper-0917"></a>

## SpanNorm: Reconciling Training Stability and Performance in Deep Transformers


### Images

![0d8d852e0e37da2ed7c902f63fdee416acb486ceff8a9b403a868515f3c214ce.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/0d8d852e0e37da2ed7c902f63fdee416acb486ceff8a9b403a868515f3c214ce.jpg)

![1c529730059f00133facb126b53c7002d872d9d2d56e7e45e823edc98f5fb204.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/1c529730059f00133facb126b53c7002d872d9d2d56e7e45e823edc98f5fb204.jpg)

![203813b7105579f3bc8819177e165f8648dbaed51a0e98e963474ccfc5cc01db.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/203813b7105579f3bc8819177e165f8648dbaed51a0e98e963474ccfc5cc01db.jpg)

![25f796236873b88ad1c6ec472a85d71d6a95dbae3752e70534aa8b8e792fbfb2.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/25f796236873b88ad1c6ec472a85d71d6a95dbae3752e70534aa8b8e792fbfb2.jpg)

![2992a7f61d62e66b50bac3fe76d2a3e0432cb9171f27ffbfb8fe488e137750c0.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/2992a7f61d62e66b50bac3fe76d2a3e0432cb9171f27ffbfb8fe488e137750c0.jpg)

![3f0073521eea1999663aec3b61eaa63c96bd140a85729a25ce0a33e7fdb27970.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/3f0073521eea1999663aec3b61eaa63c96bd140a85729a25ce0a33e7fdb27970.jpg)

![5935eb01e68c043e830cad797c68ea8bc48127ece3e03c8652b9d8f22d92668a.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/5935eb01e68c043e830cad797c68ea8bc48127ece3e03c8652b9d8f22d92668a.jpg)

![608987eee0a25b2867673f2718fc8692a4966f6afb0d0b02cd59589d0637430d.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/608987eee0a25b2867673f2718fc8692a4966f6afb0d0b02cd59589d0637430d.jpg)

![70adb97507ec0a4384f1f74ceca554fff15933dbfe0f642c5f1096c28c51f0ae.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/70adb97507ec0a4384f1f74ceca554fff15933dbfe0f642c5f1096c28c51f0ae.jpg)

![76d7e61da292734ef8bc2891a66384e9a31baf67dd7e74c6cc4a78cea352a933.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/76d7e61da292734ef8bc2891a66384e9a31baf67dd7e74c6cc4a78cea352a933.jpg)

![99c97eee5af955e90a037ca9ddf52ad3c6dff5aeea880fca1508ddcffa31b8db.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/99c97eee5af955e90a037ca9ddf52ad3c6dff5aeea880fca1508ddcffa31b8db.jpg)

![9c7e4b167735740a309434dab9504594a93af364b467fcdc970815efa75ebcfb.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/9c7e4b167735740a309434dab9504594a93af364b467fcdc970815efa75ebcfb.jpg)

![ca517043291472c75067566ad694b39fe4d69cd69739d6a8fe87a5c31380f597.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/ca517043291472c75067566ad694b39fe4d69cd69739d6a8fe87a5c31380f597.jpg)

![fbd3737ccc815bea4383db86ab970182efd8ed9ac6e996ed060e38a28f263ce4.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/images/fbd3737ccc815bea4383db86ab970182efd8ed9ac6e996ed060e38a28f263ce4.jpg)

### Tables

![4b4aa2f1a01818d4d33bd91ea9ef8e97759775de0b57e6769e321b489b3ab5db.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/4b4aa2f1a01818d4d33bd91ea9ef8e97759775de0b57e6769e321b489b3ab5db.jpg)

![84cb2df8c5881a7583816105e56ccbea86fc5f0793ef90ab1de933f61d7837f4.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/84cb2df8c5881a7583816105e56ccbea86fc5f0793ef90ab1de933f61d7837f4.jpg)

![a141039443a4e3a3914d5f1e4d7e4941c9d196d5b2dea2e4e8b94516f54e1935.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/a141039443a4e3a3914d5f1e4d7e4941c9d196d5b2dea2e4e8b94516f54e1935.jpg)

![bfe5641ffff27d2518114259c1a27015d23ae8565ca0e89a4a62af736e6978e8.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/bfe5641ffff27d2518114259c1a27015d23ae8565ca0e89a4a62af736e6978e8.jpg)

![c79bcdab3491d4a245b12a39908b388ebf3d7bc2517faec3f75418690a716514.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/c79bcdab3491d4a245b12a39908b388ebf3d7bc2517faec3f75418690a716514.jpg)

![c857b8bb6764076f31bdccf5bd128e1ee23f5f47bc7fb1ea9f58178852649d75.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/c857b8bb6764076f31bdccf5bd128e1ee23f5f47bc7fb1ea9f58178852649d75.jpg)

![d96aada8d352effb9f557747041fada1827c52d2367e0cf6031ab1ab56ec52b3.jpg](../icml_results/934_SpanNorm_%20Reconciling%20Training%20Stability%20and%20Performance%20in%20Deep%20Transformers/tables/d96aada8d352effb9f557747041fada1827c52d2367e0cf6031ab1ab56ec52b3.jpg)

<a id="paper-0918"></a>

## Doc-to-LoRA: Learning to Instantly Internalize Contexts


### Images

![01219e5650153a5730aa98afd365d62d8cf2489707fa43fbf88d544e2c6d1b1b.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/01219e5650153a5730aa98afd365d62d8cf2489707fa43fbf88d544e2c6d1b1b.jpg)

![1015dbfc83da3c5086ec143fd6dc568e4af37111f0b5cf31f392f3477c371010.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/1015dbfc83da3c5086ec143fd6dc568e4af37111f0b5cf31f392f3477c371010.jpg)

![1c0e8686bc0b2eba89e6b5404bc18862d3deb1fc5c15d0b6b81dcac7dff366ba.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/1c0e8686bc0b2eba89e6b5404bc18862d3deb1fc5c15d0b6b81dcac7dff366ba.jpg)

![1d8c6bee96de83530295abb28e03d65e51fcec77b394d7e9d296bf94d979e9dc.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/1d8c6bee96de83530295abb28e03d65e51fcec77b394d7e9d296bf94d979e9dc.jpg)

![247ea8b3bfb08155341c5bdd2c2b701b55fa485faea4f51697e0d54af0eab3f1.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/247ea8b3bfb08155341c5bdd2c2b701b55fa485faea4f51697e0d54af0eab3f1.jpg)

![3695161d82388e742c09d93daae854add3d1827e47575ca2eadb9e56741ea5bf.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/3695161d82388e742c09d93daae854add3d1827e47575ca2eadb9e56741ea5bf.jpg)

![4043c46a9ff24e4b80036263b5d713a1c0c4f5edda7ad018f15989d575c5e8df.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/4043c46a9ff24e4b80036263b5d713a1c0c4f5edda7ad018f15989d575c5e8df.jpg)

![43746b1961dc3f4d1d5b8093c7f4dd454aabdf1826bfd21c0b5030b4bdf71636.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/43746b1961dc3f4d1d5b8093c7f4dd454aabdf1826bfd21c0b5030b4bdf71636.jpg)

![46676edb0679da7626a86a26adec198d5853c533cd9fb5409e3adabf32ea41cb.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/46676edb0679da7626a86a26adec198d5853c533cd9fb5409e3adabf32ea41cb.jpg)

![7a77a4924f601f7b3ac924e67c2e7d191a8229603c4f5ffef62962006239c132.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/7a77a4924f601f7b3ac924e67c2e7d191a8229603c4f5ffef62962006239c132.jpg)

![9ac03c8895b9b20f2c6eef0be11e34012171dfc3c4fd5f86c7ec8ad3b6fda176.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/9ac03c8895b9b20f2c6eef0be11e34012171dfc3c4fd5f86c7ec8ad3b6fda176.jpg)

![a75865d14352b756f68f2d4591cf92c08b9a0d3a434e0f62e3a7948aedada0c8.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/a75865d14352b756f68f2d4591cf92c08b9a0d3a434e0f62e3a7948aedada0c8.jpg)

![afb68d5ee8346a055262e48fa1f600af3bc70eaf5f2dd1792279326a886706db.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/afb68d5ee8346a055262e48fa1f600af3bc70eaf5f2dd1792279326a886706db.jpg)

![c2c59b4c35cd55bcb70648bdcdd39e5b02a42a67a8b6ab1787e1f76f2ece46a0.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/c2c59b4c35cd55bcb70648bdcdd39e5b02a42a67a8b6ab1787e1f76f2ece46a0.jpg)

![c9bb8f2e4c2ede9bfe5600dcdf9b2646a32a026da8499f35b789a92152459210.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/c9bb8f2e4c2ede9bfe5600dcdf9b2646a32a026da8499f35b789a92152459210.jpg)

![dfadb9234d113e75cf26ddfbc6ad1e4f3c326348292e3f97d190b67110bcaa99.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/dfadb9234d113e75cf26ddfbc6ad1e4f3c326348292e3f97d190b67110bcaa99.jpg)

![e8cb4104cc28386f37bdf6ff97782681e554cdf4ba398778575ddb3e9a46e683.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/e8cb4104cc28386f37bdf6ff97782681e554cdf4ba398778575ddb3e9a46e683.jpg)

![ff90df618bd2630076fab2ba90adadf9ce1c870a050784945c0200af643ec336.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/images/ff90df618bd2630076fab2ba90adadf9ce1c870a050784945c0200af643ec336.jpg)

### Tables

![1efe48db49a58de92e220dc477a78d6086afd683f324ab68976c554203b13f20.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/1efe48db49a58de92e220dc477a78d6086afd683f324ab68976c554203b13f20.jpg)

![536bcd37d71b7c94a110286a3c5ba2266bd363879242878d660dd0a4b19000e9.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/536bcd37d71b7c94a110286a3c5ba2266bd363879242878d660dd0a4b19000e9.jpg)

![7a5102de2468b9947f210c5ea159eb757d18ced79e7a61cc9f4c79f95f0d5a88.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/7a5102de2468b9947f210c5ea159eb757d18ced79e7a61cc9f4c79f95f0d5a88.jpg)

![81f6856c784265f0635a3e3f0af34e8cce14db403b0c5f3a44e2e14f58fe0853.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/81f6856c784265f0635a3e3f0af34e8cce14db403b0c5f3a44e2e14f58fe0853.jpg)

![8b83f9b96aa2aeecd39659796ca329a4c8a9ed7167372533cc8adcef473aaf7f.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/8b83f9b96aa2aeecd39659796ca329a4c8a9ed7167372533cc8adcef473aaf7f.jpg)

![99dc92ae7a35219bf6c58a7ece96e61adc77ac6c19fd71631c21144bd60b9e95.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/99dc92ae7a35219bf6c58a7ece96e61adc77ac6c19fd71631c21144bd60b9e95.jpg)

![ac542142de3213b428b526c9e1d34c48bdf91daacd0d4e1b5336587522196d48.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/ac542142de3213b428b526c9e1d34c48bdf91daacd0d4e1b5336587522196d48.jpg)

![b237f4e8ca7bf5e7a6f04b2afe12fd03dd4ba0775f7f461eda0b6e297fbc15e4.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/b237f4e8ca7bf5e7a6f04b2afe12fd03dd4ba0775f7f461eda0b6e297fbc15e4.jpg)

![c5fd8dd36ef4c687f3a8f24c6dfde8d19d11e66de096ff12fda6d9274b9dcf92.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/c5fd8dd36ef4c687f3a8f24c6dfde8d19d11e66de096ff12fda6d9274b9dcf92.jpg)

![e2061c09dd2d3b8432d51990fe030c418a9f4d5b89612a69e97fc31c5284279f.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/e2061c09dd2d3b8432d51990fe030c418a9f4d5b89612a69e97fc31c5284279f.jpg)

![f0d845af0028545046e4c5f2e8486e3511d89d93805bde7dd0d7fc984c015df4.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/f0d845af0028545046e4c5f2e8486e3511d89d93805bde7dd0d7fc984c015df4.jpg)

![f2511f03abdb13a0f530a88f243c232da675367e14b852a613127139f53f10da.jpg](../icml_results/935_Doc-to-LoRA_%20Learning%20to%20Instantly%20Internalize%20Contexts/tables/f2511f03abdb13a0f530a88f243c232da675367e14b852a613127139f53f10da.jpg)

<a id="paper-0919"></a>

## Tokenised Flow Matching for Hierarchical Simulation Based Inference


### Images

![1720c9c67fc9394206a356572a265347c44815fd6cc3d968b2412b302d05e714.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/1720c9c67fc9394206a356572a265347c44815fd6cc3d968b2412b302d05e714.jpg)

![2d6e688fbed7a9c0c0283573c83729d9e44503e15e21af9284576528eb39305a.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/2d6e688fbed7a9c0c0283573c83729d9e44503e15e21af9284576528eb39305a.jpg)

![4a48b254fd09a5e8f83bd674c8558fbd298868f6587af1d11022999313fc593f.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/4a48b254fd09a5e8f83bd674c8558fbd298868f6587af1d11022999313fc593f.jpg)

![54f58091c48c226637f39b3456be56f5f92dfd39828ec5f49db4ea2f38056e9d.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/54f58091c48c226637f39b3456be56f5f92dfd39828ec5f49db4ea2f38056e9d.jpg)

![61f77644ed507cb17c26f4055ef5160868401a8186b69aa1c2181fa29ec29448.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/61f77644ed507cb17c26f4055ef5160868401a8186b69aa1c2181fa29ec29448.jpg)

![73edb8a880a5548763dff0929d9aec7336dbc319fe5d152fb0df60531c49d7a1.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/73edb8a880a5548763dff0929d9aec7336dbc319fe5d152fb0df60531c49d7a1.jpg)

![777e6a3314a986ccc5e51252d19763814a16250bc89eaf562e6bb05cf69d5b2d.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/777e6a3314a986ccc5e51252d19763814a16250bc89eaf562e6bb05cf69d5b2d.jpg)

![7e5c95ef486f2fce168e871c46fd2c038a54e6483dba993105ad27048f62816b.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/7e5c95ef486f2fce168e871c46fd2c038a54e6483dba993105ad27048f62816b.jpg)

![7ffd89d4a95722ad5bc19e63b61dcf4e710c0ed7380b6f0db2386d0177fe6f34.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/7ffd89d4a95722ad5bc19e63b61dcf4e710c0ed7380b6f0db2386d0177fe6f34.jpg)

![a136255e06808863ade0d379a15eb8f233c0015408e8bfc13e401f42fabdab67.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/a136255e06808863ade0d379a15eb8f233c0015408e8bfc13e401f42fabdab67.jpg)

![e932d736991a36ed30973af73e7633cacee1a290c12f38a73a0b2596e7e5f3fc.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/images/e932d736991a36ed30973af73e7633cacee1a290c12f38a73a0b2596e7e5f3fc.jpg)

### Tables

![1bb943f36cad656e7e7d0b002b2e96c41fd7c457eafcccc766dcfacf919c5c04.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/1bb943f36cad656e7e7d0b002b2e96c41fd7c457eafcccc766dcfacf919c5c04.jpg)

![2574b1ae4ea08a8fbf5e4f84d3823c985ae7c298efa774e37684799be31f7371.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/2574b1ae4ea08a8fbf5e4f84d3823c985ae7c298efa774e37684799be31f7371.jpg)

![3d9021ec78b52d693faa860cc739c6548ea67685d99a37dbaa384a77f54b4c96.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/3d9021ec78b52d693faa860cc739c6548ea67685d99a37dbaa384a77f54b4c96.jpg)

![5f328236d24f7112ad3fb9509c4a2ad8181fb2b27d443b53441c9c576fb7ddca.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/5f328236d24f7112ad3fb9509c4a2ad8181fb2b27d443b53441c9c576fb7ddca.jpg)

![7b83b1986ba002bcd6adbaf9faa2781c3f09ddac9e8533a2176ec209ee2c6d22.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/7b83b1986ba002bcd6adbaf9faa2781c3f09ddac9e8533a2176ec209ee2c6d22.jpg)

![bf078665d2de092bd652acf6aeff69e21f76b70a5d0c71929217011438acb3cd.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/bf078665d2de092bd652acf6aeff69e21f76b70a5d0c71929217011438acb3cd.jpg)

![d4b17875f1acd3d1d27188f6282697e9ed6be5d11d14e12c67260d6c6a21eea7.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/d4b17875f1acd3d1d27188f6282697e9ed6be5d11d14e12c67260d6c6a21eea7.jpg)

![ef17353f3bd9f5ba4d74dac689aa52a4a621923e80df459e162c7db713d5c687.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/ef17353f3bd9f5ba4d74dac689aa52a4a621923e80df459e162c7db713d5c687.jpg)

![f3edb28648f6eb544624d9f013b9b304f928b291f0c9c30dc7ade3f82ae4fb10.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/f3edb28648f6eb544624d9f013b9b304f928b291f0c9c30dc7ade3f82ae4fb10.jpg)

![fac5ae3355c939357a70aed8e5c7370990b66ed8fb2a74be787706dc52b5a0c4.jpg](../icml_results/936_Tokenised%20Flow%20Matching%20for%20Hierarchical%20Simulation%20Based%20Inference/tables/fac5ae3355c939357a70aed8e5c7370990b66ed8fb2a74be787706dc52b5a0c4.jpg)

<a id="paper-0920"></a>

## Faithful Relational Reasoning with Region-based Embeddings: Expressivity of Convex Coordinate-wise Models


### Images

![30248047cc7209c8307ed5413219c774e4bd05a6f6ee8d0500403635f4572474.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/images/30248047cc7209c8307ed5413219c774e4bd05a6f6ee8d0500403635f4572474.jpg)

![32479e1a310b9daf43358bcf2c8727c464eb891b0f9c3416df13579b1e0ac953.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/images/32479e1a310b9daf43358bcf2c8727c464eb891b0f9c3416df13579b1e0ac953.jpg)

![875e6d2feee92ea36061dd51dacc1f586ba912c9ce479fd1c8a249d6d972448e.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/images/875e6d2feee92ea36061dd51dacc1f586ba912c9ce479fd1c8a249d6d972448e.jpg)

![c5a73117bcbc72d85e747842fa679b27a83b1c8a8c6c7f99e380a15a1504e426.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/images/c5a73117bcbc72d85e747842fa679b27a83b1c8a8c6c7f99e380a15a1504e426.jpg)

![d2115853eab8a8f947853b8c0a3eaf77061c131867384d6bc694f0302acade4c.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/images/d2115853eab8a8f947853b8c0a3eaf77061c131867384d6bc694f0302acade4c.jpg)

### Tables

![0702709a75670cadeb57b31b07e731956cfdbabc1f515e146269f3b54c3544b7.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/tables/0702709a75670cadeb57b31b07e731956cfdbabc1f515e146269f3b54c3544b7.jpg)

![7971b0bcd4c19ec3b798fae525d4a758978ce46982c54a56d13017b4752b9b56.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/tables/7971b0bcd4c19ec3b798fae525d4a758978ce46982c54a56d13017b4752b9b56.jpg)

![be9cb7c8842714f4e332dff819e47cd0adf9f34f8c8f03abed1441e82bcb4146.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/tables/be9cb7c8842714f4e332dff819e47cd0adf9f34f8c8f03abed1441e82bcb4146.jpg)

![e3a328e2ac4b34a39fe3901091f7139eee49329af902f50d89ef60a50d0458db.jpg](../icml_results/937_Faithful%20Relational%20Reasoning%20with%20Region-based%20Embeddings_%20Expressivity%20of%20Convex%20Coordinate-wise%20M/tables/e3a328e2ac4b34a39fe3901091f7139eee49329af902f50d89ef60a50d0458db.jpg)

<a id="paper-0921"></a>

## MetaPerch: Learning from metadata for bioacoustics foundation models


### Images

![00bae25d01ccc97f9f507b44c042be8ae7094f76fc29ac82c2ceadfb81735fd8.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/00bae25d01ccc97f9f507b44c042be8ae7094f76fc29ac82c2ceadfb81735fd8.jpg)

![064acf855b8dda8c911dafaf9f1753fd8cb3b935b0ef0ecc1a49f535c27ffd8e.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/064acf855b8dda8c911dafaf9f1753fd8cb3b935b0ef0ecc1a49f535c27ffd8e.jpg)

![14e1a1ec32c9a5f291ca2bf565c908ac5dfd471f551c3c1d087a0b048eb436b2.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/14e1a1ec32c9a5f291ca2bf565c908ac5dfd471f551c3c1d087a0b048eb436b2.jpg)

![4b5dc4480fadcf65c81674466459c71d4b4e39940db7b61baaa4455dc128c804.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/4b5dc4480fadcf65c81674466459c71d4b4e39940db7b61baaa4455dc128c804.jpg)

![6ec18d18adefc47d13a7dde838174903127a607d1419d5e1e9debee09f38ef31.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/6ec18d18adefc47d13a7dde838174903127a607d1419d5e1e9debee09f38ef31.jpg)

![72b5b2184e0d0f5922525ee87d0431a6fd22389aba157813062d8fe1a81874fe.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/72b5b2184e0d0f5922525ee87d0431a6fd22389aba157813062d8fe1a81874fe.jpg)

![91e2d8aadab2dfa55a1518b317cdfeb6601d82993bb89ffa82835a274b7f44d3.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/91e2d8aadab2dfa55a1518b317cdfeb6601d82993bb89ffa82835a274b7f44d3.jpg)

![9a02107e8000d6c47737e06ae9f6a853dbf43ef60389224323cb0d2ad37f7faf.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/9a02107e8000d6c47737e06ae9f6a853dbf43ef60389224323cb0d2ad37f7faf.jpg)

![a9dd71eae00a4fd869f18d1a7255bfcfdcbfebab86d2a865bc99d305311be20f.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/a9dd71eae00a4fd869f18d1a7255bfcfdcbfebab86d2a865bc99d305311be20f.jpg)

![b3203458bafe2971e78b5d8794157dbdf98801b6f95bd64638c2c2c4cc41e712.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/b3203458bafe2971e78b5d8794157dbdf98801b6f95bd64638c2c2c4cc41e712.jpg)

![c1b0b7d0c652480b05372fe33facc995962ef09c13f207b24cec491cc9114b28.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/c1b0b7d0c652480b05372fe33facc995962ef09c13f207b24cec491cc9114b28.jpg)

![c352a0f877a593294ff04bf4f36b33d93028a1867e743947cf0fc6ee2da86c2f.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/c352a0f877a593294ff04bf4f36b33d93028a1867e743947cf0fc6ee2da86c2f.jpg)

![cfc4a5133461afb1242423560fe29fc1261f4d1eb542d65a1b41d747d700ab3d.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/cfc4a5133461afb1242423560fe29fc1261f4d1eb542d65a1b41d747d700ab3d.jpg)

![ddd35146b415cb2c925634169f3aaa708211c2a05dbd9dc254b3c5486511b4b5.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/ddd35146b415cb2c925634169f3aaa708211c2a05dbd9dc254b3c5486511b4b5.jpg)

![e105f101e34bdbab00fa0bae9a587df8e24985377f8aee144f41ef9ebb7bfc67.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/e105f101e34bdbab00fa0bae9a587df8e24985377f8aee144f41ef9ebb7bfc67.jpg)

![e8d71492d95076dd62c4e2fc664151dd15e6d41c9ede501d94fb82e436fd46f5.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/e8d71492d95076dd62c4e2fc664151dd15e6d41c9ede501d94fb82e436fd46f5.jpg)

![f0296d8be4f2ce42f32d6dc0deaeda86891d89ad37707b892872199f6afd11d0.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/images/f0296d8be4f2ce42f32d6dc0deaeda86891d89ad37707b892872199f6afd11d0.jpg)

### Tables

![05da1cc622b44fc8cdd42d5c9ae1ea1a1506abed77ee365a71ac7eed5aa9d49d.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/05da1cc622b44fc8cdd42d5c9ae1ea1a1506abed77ee365a71ac7eed5aa9d49d.jpg)

![14529961880f3021c9b890b136fc1ab7b3fe131464c4ed9763b5bf0880d8ed55.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/14529961880f3021c9b890b136fc1ab7b3fe131464c4ed9763b5bf0880d8ed55.jpg)

![2e5a78a808ed28a878c6be1329444d77101b3fa7ec1dd8be0f76f21ac5455f68.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/2e5a78a808ed28a878c6be1329444d77101b3fa7ec1dd8be0f76f21ac5455f68.jpg)

![37e320bac4915f6af4f73b14863fcde9b528982baecc2cb42571d1516daf8f6d.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/37e320bac4915f6af4f73b14863fcde9b528982baecc2cb42571d1516daf8f6d.jpg)

![5332681d90658b21ed4549edfbea7906ffc318e91cede1e613a19d9934be98cb.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/5332681d90658b21ed4549edfbea7906ffc318e91cede1e613a19d9934be98cb.jpg)

![60520ed548475d9e7ec694af0d5ebac261b4dc4d22c6ba004ae3c5da1481b500.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/60520ed548475d9e7ec694af0d5ebac261b4dc4d22c6ba004ae3c5da1481b500.jpg)

![60a77afbefcf4093458f696a99202613d0db1b7d7a48eeb12debdb98d6ab4d46.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/60a77afbefcf4093458f696a99202613d0db1b7d7a48eeb12debdb98d6ab4d46.jpg)

![62eeff6c63ad03bcd48302c8399cd911acdfa76a422844561b09dea204067901.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/62eeff6c63ad03bcd48302c8399cd911acdfa76a422844561b09dea204067901.jpg)

![6a995b336f1a3082a7f78b787eb99c0859b22007ebf96117f7d7efacce99613c.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/6a995b336f1a3082a7f78b787eb99c0859b22007ebf96117f7d7efacce99613c.jpg)

![9f4e51cd3b4626bd4a2417622ce9cbf9acc8343279e21751e7d2f798f5807d70.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/9f4e51cd3b4626bd4a2417622ce9cbf9acc8343279e21751e7d2f798f5807d70.jpg)

![e11f09f3bb7140813e40cd1afdea8e2c9e47bef6179ae22cbffdc04d704e5fdf.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/e11f09f3bb7140813e40cd1afdea8e2c9e47bef6179ae22cbffdc04d704e5fdf.jpg)

![e3dbb70bf74f0d545de30acca59690255083b3dd6f990dde9b1c5932d3ff4db2.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/e3dbb70bf74f0d545de30acca59690255083b3dd6f990dde9b1c5932d3ff4db2.jpg)

![fb93c01a22e3678d05707fc126e918226bc9dce7abd5c2b493b4e58f7836d269.jpg](../icml_results/938_MetaPerch_%20Learning%20from%20metadata%20for%20bioacoustics%20foundation%20models/tables/fb93c01a22e3678d05707fc126e918226bc9dce7abd5c2b493b4e58f7836d269.jpg)

<a id="paper-0922"></a>

## Bioacoustic Geolocation: Species Sounds as Geographic Signals


### Images

![05b0a36257109d0ce09f750020242e418856e0489c46a0dcbb515a82ecf9f14c.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/05b0a36257109d0ce09f750020242e418856e0489c46a0dcbb515a82ecf9f14c.jpg)

![1d5df6bd5e19919faae7b6ea7b27f32501cbb84a0d523391d3524731e9a33fde.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/1d5df6bd5e19919faae7b6ea7b27f32501cbb84a0d523391d3524731e9a33fde.jpg)

![249cfa7b81c6d632f19af30657243504f5a4b09873f1f82f2b524214be903693.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/249cfa7b81c6d632f19af30657243504f5a4b09873f1f82f2b524214be903693.jpg)

![3a3fcc52944400acf1fb7f2c6691062763c15f631eaa8ef1033df83df200ed40.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/3a3fcc52944400acf1fb7f2c6691062763c15f631eaa8ef1033df83df200ed40.jpg)

![3ada97076d39603a8dbee8ed25e7614e9b5637f0efe74ea5c5f1c17d91346ba1.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/3ada97076d39603a8dbee8ed25e7614e9b5637f0efe74ea5c5f1c17d91346ba1.jpg)

![3e00080844fd7912f1365ba19326a22b8c59235613c3fb95c72ce3e2e453c9b9.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/3e00080844fd7912f1365ba19326a22b8c59235613c3fb95c72ce3e2e453c9b9.jpg)

![42c9b4820c2661de4863fe67944da4ba1c655693d3fe44fa4af218b14b16f293.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/42c9b4820c2661de4863fe67944da4ba1c655693d3fe44fa4af218b14b16f293.jpg)

![53725c8048fd02988c7d161bc474c0041dd2a67f4c2ef8c71401f762bd47b7f6.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/53725c8048fd02988c7d161bc474c0041dd2a67f4c2ef8c71401f762bd47b7f6.jpg)

![57c5d32a0e4615dade6144ad9cc3f31b09fa8cc4f4ab4bbce00e5cff3ade22da.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/57c5d32a0e4615dade6144ad9cc3f31b09fa8cc4f4ab4bbce00e5cff3ade22da.jpg)

![63eafe2a0213d8a332a5b392c99111ee5b306df04ad6322e4c6c6ca92bffaec5.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/63eafe2a0213d8a332a5b392c99111ee5b306df04ad6322e4c6c6ca92bffaec5.jpg)

![66ea1c4e3857622c76c982a847ed2b06d32ef318405b25bd8ad499d6682b4043.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/66ea1c4e3857622c76c982a847ed2b06d32ef318405b25bd8ad499d6682b4043.jpg)

![82f2328c6a6b7a35c530bf2c7db67102226b592db8c22d53029a607cd78fc362.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/82f2328c6a6b7a35c530bf2c7db67102226b592db8c22d53029a607cd78fc362.jpg)

![8c03ffbe87501b248c600034ebb983892f5d661beb02a97cf2a0de7c41ceb40e.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/8c03ffbe87501b248c600034ebb983892f5d661beb02a97cf2a0de7c41ceb40e.jpg)

![9ef6c8f9ed519f7aadccb13fbbe3de157aaed63b76e7977fc5cb377280f76ad0.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/9ef6c8f9ed519f7aadccb13fbbe3de157aaed63b76e7977fc5cb377280f76ad0.jpg)

![ae0c833e0f5508744bf8a1e1bb124d824335c6da19af8b7803f389847337a0e0.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/ae0c833e0f5508744bf8a1e1bb124d824335c6da19af8b7803f389847337a0e0.jpg)

![c43e805fe3b56934b6b04352a97af207282c4aaf408d2f0bd3d089f341494cf8.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/c43e805fe3b56934b6b04352a97af207282c4aaf408d2f0bd3d089f341494cf8.jpg)

![cfd9b75b2cd714b8786718270644e4cd5326bc798fdaebf613e7fc1bf373436a.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/cfd9b75b2cd714b8786718270644e4cd5326bc798fdaebf613e7fc1bf373436a.jpg)

![fea87af590c02222db20f980513cd898c8f181e7d90eb406f2496f3a3cff46e7.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/images/fea87af590c02222db20f980513cd898c8f181e7d90eb406f2496f3a3cff46e7.jpg)

### Tables

![02a93ac5567b8c24fd930e8af352fbf0cd0be6d0031eb18e3f0dd473f07e47bf.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/02a93ac5567b8c24fd930e8af352fbf0cd0be6d0031eb18e3f0dd473f07e47bf.jpg)

![12443bee2c3011b1b159505dfaad89fd2bf1865da619282346ef7230a978504f.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/12443bee2c3011b1b159505dfaad89fd2bf1865da619282346ef7230a978504f.jpg)

![239aec6fe4f84e4d066d0be4c6ce7aeda4e5409aa8fe7da97181cc75e347e5cf.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/239aec6fe4f84e4d066d0be4c6ce7aeda4e5409aa8fe7da97181cc75e347e5cf.jpg)

![23bbd890748fc1eb55b4de5297840e82b04e1b930a47eab9683dcc82c44d30d2.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/23bbd890748fc1eb55b4de5297840e82b04e1b930a47eab9683dcc82c44d30d2.jpg)

![38f97ba3aee2181ecfa9af6177869535ce5db69397f832f3e27f3f69ac925bcf.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/38f97ba3aee2181ecfa9af6177869535ce5db69397f832f3e27f3f69ac925bcf.jpg)

![3a01f888ff4c2478e826083e1834d98d88864e5bb793236f852d55548a76b421.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/3a01f888ff4c2478e826083e1834d98d88864e5bb793236f852d55548a76b421.jpg)

![6acb120353ddb235b039ef879adb5aa192f6cc4b43e07ef8b51d801db76eae7a.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/6acb120353ddb235b039ef879adb5aa192f6cc4b43e07ef8b51d801db76eae7a.jpg)

![8538e80c4b23b1643dc0c965aaaae6fee36dfcd192da13ced4bf3be27ecce2aa.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/8538e80c4b23b1643dc0c965aaaae6fee36dfcd192da13ced4bf3be27ecce2aa.jpg)

![9fb3b99aa6f217208143c758ecca2b967b33a74280c50c91636b4e402ef26d06.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/9fb3b99aa6f217208143c758ecca2b967b33a74280c50c91636b4e402ef26d06.jpg)

![a96b1b0ef9face5fb1ce21282b22fd5b01cb9060efe32ae33b435f9e596d0770.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/a96b1b0ef9face5fb1ce21282b22fd5b01cb9060efe32ae33b435f9e596d0770.jpg)

![b8083177441f1588ae3da02cdf645e117eead66f54b8029a97ef11d8ba1b03bb.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/b8083177441f1588ae3da02cdf645e117eead66f54b8029a97ef11d8ba1b03bb.jpg)

![e225514bc7431aa90481f65845d5c0a74cf9dddf90c03c8aac243edfb5429033.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/e225514bc7431aa90481f65845d5c0a74cf9dddf90c03c8aac243edfb5429033.jpg)

![e36eff6f8b73643626b10b5096f3dd57d3bedc22d8439b2dd43e7af5214074c1.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/e36eff6f8b73643626b10b5096f3dd57d3bedc22d8439b2dd43e7af5214074c1.jpg)

![e6121372329722e4902ed293ab7170d35348dd10b7a1a9b8627a16d131ea1319.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/e6121372329722e4902ed293ab7170d35348dd10b7a1a9b8627a16d131ea1319.jpg)

![f2ad3dce420f16c8ef05d8fbfc49bd6c487923893568dad98044ccfcc8e741fa.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/f2ad3dce420f16c8ef05d8fbfc49bd6c487923893568dad98044ccfcc8e741fa.jpg)

![f7bc768aac705ad8b448c03bf248c79dfd0890e7da71114aa09cb73818aa8b68.jpg](../icml_results/939_Bioacoustic%20Geolocation_%20Species%20Sounds%20as%20Geographic%20Signals/tables/f7bc768aac705ad8b448c03bf248c79dfd0890e7da71114aa09cb73818aa8b68.jpg)

<a id="paper-0923"></a>

## One-shot Conditional Sampling: MMD meets Nearest Neighbors


### Images

![045e3ecc0e2d3f8dfb872c61f4b50ee39b908ef593b02a57c2ac7565972273df.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/045e3ecc0e2d3f8dfb872c61f4b50ee39b908ef593b02a57c2ac7565972273df.jpg)

![0a56936afcc01d1eb09a33fab3a7999848c1f2c512c9652befa05024ef8a70f7.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/0a56936afcc01d1eb09a33fab3a7999848c1f2c512c9652befa05024ef8a70f7.jpg)

![15d17fac1ae93a7e560ea3451a6d1da69d75d7491c00be1baa4f5557b7e5b3ce.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/15d17fac1ae93a7e560ea3451a6d1da69d75d7491c00be1baa4f5557b7e5b3ce.jpg)

![2b9db23af9cd8b8bcb31afc72c849eecc9d29a1665e55d1edf5b7a8318a2f27d.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/2b9db23af9cd8b8bcb31afc72c849eecc9d29a1665e55d1edf5b7a8318a2f27d.jpg)

![2ca8461da249f20f21fc5a09a779d295021b72f852a366873a3511930dc7b29c.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/2ca8461da249f20f21fc5a09a779d295021b72f852a366873a3511930dc7b29c.jpg)

![3d888f507cec1e21a92b398d5a50c9e8d2746c08f3765e827690923fb04d253f.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/3d888f507cec1e21a92b398d5a50c9e8d2746c08f3765e827690923fb04d253f.jpg)

![47595208a0ec3348b927a22c815382a23835be829af9300a79c29f3b99bb4aca.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/47595208a0ec3348b927a22c815382a23835be829af9300a79c29f3b99bb4aca.jpg)

![57c8c40c42b44a2a7eee4fb6a19e42ac7a0c7a83494d154da7c0d9d2f1591bb0.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/57c8c40c42b44a2a7eee4fb6a19e42ac7a0c7a83494d154da7c0d9d2f1591bb0.jpg)

![5cd91da94add08fc5747ba233a3cc99446205d649515b309506aa758338bd23b.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/5cd91da94add08fc5747ba233a3cc99446205d649515b309506aa758338bd23b.jpg)

![9c4cf293fc6bab172457f67c0d32b22b0b4f34414e9deceb859713f89c368ae2.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/9c4cf293fc6bab172457f67c0d32b22b0b4f34414e9deceb859713f89c368ae2.jpg)

![9f137218a1138602919821b45db88ea00e8667bf0cbbdfc67bc4c52e3c4fd9cd.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/9f137218a1138602919821b45db88ea00e8667bf0cbbdfc67bc4c52e3c4fd9cd.jpg)

![b025897f3d605aa639f1894c0c72d8e637b746efc683020c032f1ee9b1e4ca67.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/b025897f3d605aa639f1894c0c72d8e637b746efc683020c032f1ee9b1e4ca67.jpg)

![bd450e278a9ef08390dfa1e17651bc0cb2e38790aa311ddc1dd7449687b7eae3.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/bd450e278a9ef08390dfa1e17651bc0cb2e38790aa311ddc1dd7449687b7eae3.jpg)

![d955e6cfe226a9dd3cf268b6b4b4a853398b3d2bcc411773d8d82fa225564c3d.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/d955e6cfe226a9dd3cf268b6b4b4a853398b3d2bcc411773d8d82fa225564c3d.jpg)

![dd41b4ba213f48cf00c765ce78758d83fdc50cc435ad51d40cbb395964794e6a.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/dd41b4ba213f48cf00c765ce78758d83fdc50cc435ad51d40cbb395964794e6a.jpg)

![e0e99513dec2cf46550c0b528d426a46532715b7a2f92bc258506928110cad6f.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/e0e99513dec2cf46550c0b528d426a46532715b7a2f92bc258506928110cad6f.jpg)

![e2f36e3ba91d9959a144d48a3f2f0b2e5e6b7cbd5468e7d2f80cd0302adfc3fd.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/e2f36e3ba91d9959a144d48a3f2f0b2e5e6b7cbd5468e7d2f80cd0302adfc3fd.jpg)

![e34a3fcff83bb7e3bded1a0db014446b737c4e82679a0e20088e0e3ccb9c6878.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/e34a3fcff83bb7e3bded1a0db014446b737c4e82679a0e20088e0e3ccb9c6878.jpg)

![eab26f24d111d8b0f04a95e8cdf6032451ffec303dba28750db47c6d2f4be406.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/images/eab26f24d111d8b0f04a95e8cdf6032451ffec303dba28750db47c6d2f4be406.jpg)

### Tables

![366332164ec7eacb569955973d637ee9016ea1caad4f100ccd17e9ce4ce00750.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/tables/366332164ec7eacb569955973d637ee9016ea1caad4f100ccd17e9ce4ce00750.jpg)

![366735c2b51e2f4ad006fd748848e9e95843b1bb9ec4669a2ab0ab8e3e9adbad.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/tables/366735c2b51e2f4ad006fd748848e9e95843b1bb9ec4669a2ab0ab8e3e9adbad.jpg)

![61c9c51ed68d8405ceb672768a6ba0593e06eaf9a3243986a2c43aca4d3171cb.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/tables/61c9c51ed68d8405ceb672768a6ba0593e06eaf9a3243986a2c43aca4d3171cb.jpg)

![c9ce8072caecc2e3dd867f21e54cb8eb8d404cb5353df5b43e5905745e1d3ec9.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/tables/c9ce8072caecc2e3dd867f21e54cb8eb8d404cb5353df5b43e5905745e1d3ec9.jpg)

![e938a30db935524009e59681f91e5200b6285ab1b1b87abf6a3bc49f695fa2d2.jpg](../icml_results/940_One-shot%20Conditional%20Sampling_%20MMD%20meets%20Nearest%20Neighbors/tables/e938a30db935524009e59681f91e5200b6285ab1b1b87abf6a3bc49f695fa2d2.jpg)

<a id="paper-0924"></a>

## Blending Neural Control Density Functions for Stabilization and Safety

### Images

![0180ce0b4c0c8b0906983a8d640491d3ae6d343eb68b8a03cb809766257b03d0.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/0180ce0b4c0c8b0906983a8d640491d3ae6d343eb68b8a03cb809766257b03d0.jpg)

![199c9a84d3f854f0875af3ff81bb8e23eb77496a12f6f83f7bdc084d8c0b5485.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/199c9a84d3f854f0875af3ff81bb8e23eb77496a12f6f83f7bdc084d8c0b5485.jpg)

![2aed4260ebdb0690e9985e42b2d0392c4e8c962303c844fdd052405429ec65da.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/2aed4260ebdb0690e9985e42b2d0392c4e8c962303c844fdd052405429ec65da.jpg)

![3a8f1968639772e3162c260b7bb514f9007be0a3d0a652e079548478a0e5c433.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/3a8f1968639772e3162c260b7bb514f9007be0a3d0a652e079548478a0e5c433.jpg)

![4fdf527141ba7dd1057844e5541e958fbb8ac9575662098bbc9463e0b8f3cac3.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/4fdf527141ba7dd1057844e5541e958fbb8ac9575662098bbc9463e0b8f3cac3.jpg)

![50a81b8a6c4b71fa7c3feb40a4946af994015afb8a71717ce01e92caf493eaa2.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/50a81b8a6c4b71fa7c3feb40a4946af994015afb8a71717ce01e92caf493eaa2.jpg)

![518c03f13a89432fe4df624e13b12ebf2eacc225b43f36f0f28ec0d43636cb09.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/518c03f13a89432fe4df624e13b12ebf2eacc225b43f36f0f28ec0d43636cb09.jpg)

![65edef592cc14e73af398fecc22b5645018100dd3aac9c3b599acb24378316dc.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/65edef592cc14e73af398fecc22b5645018100dd3aac9c3b599acb24378316dc.jpg)

![66237e86c329da5ac1f536bc76008c6412b914cf6c932c9917d53ae97812a591.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/66237e86c329da5ac1f536bc76008c6412b914cf6c932c9917d53ae97812a591.jpg)

![a87341bc0c5be727faec43ccf74b43b1c03aa8794093f1280f28f4ef3928b1a1.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/a87341bc0c5be727faec43ccf74b43b1c03aa8794093f1280f28f4ef3928b1a1.jpg)

![bfa3c6f43bc343eece93aca937c55302ede101004e095381d4edd75722267ef6.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/bfa3c6f43bc343eece93aca937c55302ede101004e095381d4edd75722267ef6.jpg)

![d409aa96327b35012a8860d8e3d3610ad27b84bad8e36e14117ea0bf8c45fd4c.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/d409aa96327b35012a8860d8e3d3610ad27b84bad8e36e14117ea0bf8c45fd4c.jpg)

![ef56a1c3c17b0b3aae2c9b388a209be5feffdff9575f04ab19be2e07c5e3d6f3.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/ef56a1c3c17b0b3aae2c9b388a209be5feffdff9575f04ab19be2e07c5e3d6f3.jpg)

### Tables

![1df1ecc8f36ec6ac2f6636aa941b804b8cfa7b217a982b3e460a1eb0d4c77933.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/tables/1df1ecc8f36ec6ac2f6636aa941b804b8cfa7b217a982b3e460a1eb0d4c77933.jpg)

![ef82468835d5d72009fa07c79cc342bca2257e4f586e1c61d024d02993602680.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/tables/ef82468835d5d72009fa07c79cc342bca2257e4f586e1c61d024d02993602680.jpg)
