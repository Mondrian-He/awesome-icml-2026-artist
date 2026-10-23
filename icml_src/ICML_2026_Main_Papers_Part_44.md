# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 44 of 197

## 目录 (Table of Contents)

1. [Learning Permutation from Structure Without Supervision](#paper-1420)
2. [Relational Structural Causal Models](#paper-1421)
3. [Learning Treatment Allocations with Risk Control Under Partial Identifiability](#paper-1422)
4. [Revisiting Efficiency–Accuracy Scaling in Mixture-of-Experts Architectures](#paper-1423)
5. [MAD: Manifold Attracted Diffusion](#paper-1424)
6. [Don’t Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference](#paper-1425)
7. [TSP with Predictions: Heatmap to Tour with Provable Guarantees](#paper-1426)
8. [Graph Neural Dynamics via Learned Energy and Tangential Flows](#paper-1427)
9. [ConEx: Human-Interpretable Saliency Maps via Concept-Aware Attribution](#paper-1428)
10. [Hide&Seek: Learning to Explain in an End-to-End Differentiable Network](#paper-1429)
11. [Training LLM Agents to Empower Humans](#paper-1430)
12. [Global Plane Waves from Local Gaussians: Periodic Charge Densities in a Blink](#paper-1431)
13. [Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training](#paper-1432)
14. [GraphPFN: A Prior-Data Fitted Graph Foundation Model](#paper-1433)
15. [Clipping Makes Distributed and Federated Asynchronous SGD Robust to Stragglers](#paper-1434)
16. [When Softmax Fails at the Top: Extreme-Value Corrections for InfoNCE](#paper-1435)
17. [When Can You Poison Rewards? A Tight Characterization of Reward Poisoning in Linear MDPs](#paper-1436)
18. [Spectrally-Guided Diffusion Noise Schedules](#paper-1437)
19. [Automatic Construction of Clinical Scoring Systems with LLM Agents](#paper-1438)
20. [CellBRIDGE: Learning Cellular Trajectories via Interaction-Aware Alignment](#paper-1439)
21. [Width Independent Bounds for the Local Lipschitz Constant of Deep Neural Networks at Random Initialization and after Lazy Training](#paper-1440)
22. [Entropy-informed Decoding: Adaptive Information-Driven Branching](#paper-1441)
23. [Theory of Minimal Weight Perturbations in Deep Networks and its Applications for Low-Rank Activated Backdoor Attacks](#paper-1442)
24. [Stochastic Linear Bandits with Parameter Noise](#paper-1443)
25. [Unfolded Laplacian Spectral Embedding: A Theoretically Grounded Approach to Dynamic Network Representation](#paper-1444)
26. [Robust Bayesian Optimisation with Unbounded Corruptions](#paper-1445)
27. [BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](#paper-1446)
28. [Understanding Data Temporality Impact on Large Language Models Pre-training](#paper-1447)
29. [The Two-Hump Problem: Bridging the Difficulty Gap in Mathematical Reinforcement Learning](#paper-1448)
30. [Trading Complexity for Expressivity Through Structured Generalized Linear Token Mixing](#paper-1449)
31. [Local Policies for Graph-Structured Markov Decision Processes](#paper-1450)
32. [Curated Synthetic Data Doesn’t Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences](#paper-1451)
33. [NAVIGATE: Evaluating Visual-Guided Search Decision-Making on the Open Web](#paper-1452)

---


<a id="paper-1420"></a>

## Learning Permutation from Structure Without Supervision

### Images

![0a3ede988428dc643309e66af2a26fbccdba0b1fb0e593c564312961fd4a334f.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/0a3ede988428dc643309e66af2a26fbccdba0b1fb0e593c564312961fd4a334f.jpg)

![159edbc84b06b79e8797509030a3133f40ae8978e62bd1dd9ae3d705dae01f01.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/159edbc84b06b79e8797509030a3133f40ae8978e62bd1dd9ae3d705dae01f01.jpg)

![1e84164e04809cc92a55fb838cc375a6669315722923146034819180a7334e81.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/1e84164e04809cc92a55fb838cc375a6669315722923146034819180a7334e81.jpg)

![2ebebb5526c0d98b93d62680da048c22cd754f69d4f4c025b7db2fe9517620dd.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/2ebebb5526c0d98b93d62680da048c22cd754f69d4f4c025b7db2fe9517620dd.jpg)

![3e724984334824cf438028b9da7040638cfc693b9696273e04c3fee302e14f93.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/3e724984334824cf438028b9da7040638cfc693b9696273e04c3fee302e14f93.jpg)

![48d6242b52121f0f6e38eca898878d0689028aba4ee0e7b1a6eb69af6b5d39e2.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/48d6242b52121f0f6e38eca898878d0689028aba4ee0e7b1a6eb69af6b5d39e2.jpg)

![580de71cc54a86c3c88d0dad0259a9b141bb2f4460c0f43e7a5b9d10aa8c06cb.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/580de71cc54a86c3c88d0dad0259a9b141bb2f4460c0f43e7a5b9d10aa8c06cb.jpg)

![79440181adb62386fa5a233c4f612c5141487ad20be507be10481d80ec10d8b2.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/79440181adb62386fa5a233c4f612c5141487ad20be507be10481d80ec10d8b2.jpg)

![7b8c992453d88da67c3212ac1be5e2f1edc5da18be623f9863a55796e1cd1e75.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/7b8c992453d88da67c3212ac1be5e2f1edc5da18be623f9863a55796e1cd1e75.jpg)

![8124bad5439fce98d923d9c665cd586991146be0adc3f3b5e8895023096da917.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/8124bad5439fce98d923d9c665cd586991146be0adc3f3b5e8895023096da917.jpg)

![8492e2b981e91d8acdfda2052365c36738454aa97007997ff919b3b6665c07d5.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/8492e2b981e91d8acdfda2052365c36738454aa97007997ff919b3b6665c07d5.jpg)

![8ec2c27324f1c6b67af9053713f881a128ba827deb520d94363c7eeda35a2082.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/8ec2c27324f1c6b67af9053713f881a128ba827deb520d94363c7eeda35a2082.jpg)

![98427009fca9a5aa0e2dd75e3ff4c96c619a0dbd4e9b8153101f9e6c22bb9f36.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/98427009fca9a5aa0e2dd75e3ff4c96c619a0dbd4e9b8153101f9e6c22bb9f36.jpg)

![a8762e19fba67f4d9fb59302b4d8d5c5e10835dc8e91ec7fd670c366f1863c0d.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/a8762e19fba67f4d9fb59302b4d8d5c5e10835dc8e91ec7fd670c366f1863c0d.jpg)

![fc5caf9edc7b28f398cf14c7944e9f539ad5f030e18fe5f7cb47dd21da8c0184.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/fc5caf9edc7b28f398cf14c7944e9f539ad5f030e18fe5f7cb47dd21da8c0184.jpg)

![fcba2d25cab86673c984e095c7d2a3e28ce7207cda737dc28ad2961c7dff6380.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/images/fcba2d25cab86673c984e095c7d2a3e28ce7207cda737dc28ad2961c7dff6380.jpg)

### Tables

![cb88e7b30492b73c3f3c55e34afa03b13ce51c9247342bf44a76bcc6308f9c32.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/tables/cb88e7b30492b73c3f3c55e34afa03b13ce51c9247342bf44a76bcc6308f9c32.jpg)

![fd5adfd7070a65d7881b9359ecf122df22d88a9133480ea5cf425b8a8101c5f7.jpg](../icml_results/1439_ExPLAIND_%20Unifying%20Model%2C%20Data%2C%20and%20Training%20Attribution%20to%20Study%20Model%20Behavior/tables/fd5adfd7070a65d7881b9359ecf122df22d88a9133480ea5cf425b8a8101c5f7.jpg)

## Learning Permutation from Structure Without Supervision


### Images

![11a29520bf710eaa9d713e68e3d83ff1b26ccf74b9cdbca06e8f159a4974c611.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/images/11a29520bf710eaa9d713e68e3d83ff1b26ccf74b9cdbca06e8f159a4974c611.jpg)

![29d4b76ae27df6f4c93039d1e19c079a8187107444ff4755ef93412eb2464733.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/images/29d4b76ae27df6f4c93039d1e19c079a8187107444ff4755ef93412eb2464733.jpg)

![aa95adee78237c7dfef7e2dd1e27a93e0017b7fb378f23e2fa2606fd1a1ca9ed.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/images/aa95adee78237c7dfef7e2dd1e27a93e0017b7fb378f23e2fa2606fd1a1ca9ed.jpg)

![d35b484b261ad65a7ad9e15151093a073ea4a249f69d15180ca5e4f6c3fb1fbc.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/images/d35b484b261ad65a7ad9e15151093a073ea4a249f69d15180ca5e4f6c3fb1fbc.jpg)

### Tables

![0b1672e095bc23d5f1521bf689b6a823ccc23bd00975c1755b31a1463ee375c8.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/0b1672e095bc23d5f1521bf689b6a823ccc23bd00975c1755b31a1463ee375c8.jpg)

![7a2370a669856d2b3c6e7aa4b0ecf8228f964ed35e01b871dbb17d91ac82bfec.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/7a2370a669856d2b3c6e7aa4b0ecf8228f964ed35e01b871dbb17d91ac82bfec.jpg)

![8be57f7c432e88ef5cb702dfc949b51252e520997feb45cc56a241999a41a228.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/8be57f7c432e88ef5cb702dfc949b51252e520997feb45cc56a241999a41a228.jpg)

![a0784506b4d5f4be29546cbcef49a6250f78b6b5b0ced21c606012157c7adbf7.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/a0784506b4d5f4be29546cbcef49a6250f78b6b5b0ced21c606012157c7adbf7.jpg)

![ba84213de352b0e565005ba922a77e271a471fc45a57bb25e97caa0587944705.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/ba84213de352b0e565005ba922a77e271a471fc45a57bb25e97caa0587944705.jpg)

![c34b1c91abcf9800d8aa0d8747eb6240cf86a1b298b70646b8c8087ee2415a96.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/c34b1c91abcf9800d8aa0d8747eb6240cf86a1b298b70646b8c8087ee2415a96.jpg)

![ce7391dd406972bc0350652b00282444c42fba6523e83a9dcba9ac8915d8bc59.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/ce7391dd406972bc0350652b00282444c42fba6523e83a9dcba9ac8915d8bc59.jpg)

![d668137e0f2fc987cc85cddab732cd258db01f19275eb3f53a37dfac0e18ef68.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/d668137e0f2fc987cc85cddab732cd258db01f19275eb3f53a37dfac0e18ef68.jpg)

![ece362fe20695d03fd954132809b834d91a628d1b104908a1db763909626d3ce.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/ece362fe20695d03fd954132809b834d91a628d1b104908a1db763909626d3ce.jpg)

![f0d57942700d57dceb2bb42d6ce60b5be4c316a32bf9963c8dfd1439404e08ef.jpg](../icml_results/1440_Learning%20Permutation%20from%20Structure%20Without%20Supervision/tables/f0d57942700d57dceb2bb42d6ce60b5be4c316a32bf9963c8dfd1439404e08ef.jpg)

<a id="paper-1421"></a>

## Relational Structural Causal Models


### Images

![10187516766b673d463222d840bca1bc48db6d4b087bf20493913d0b33da6d11.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/10187516766b673d463222d840bca1bc48db6d4b087bf20493913d0b33da6d11.jpg)

![130f1b39db363f72737eb37e904e4acb14dc7c56658ded8a20c146eb9eedff5b.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/130f1b39db363f72737eb37e904e4acb14dc7c56658ded8a20c146eb9eedff5b.jpg)

![21718c8d118e7991ed7c82e29970298980e8ce4eb39fb8de1d5e5679b40381ed.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/21718c8d118e7991ed7c82e29970298980e8ce4eb39fb8de1d5e5679b40381ed.jpg)

![2e3c5ce8c2a64cd4ff948b8c8177e88059c58fcdc4c1353ca53196025d3e12a6.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/2e3c5ce8c2a64cd4ff948b8c8177e88059c58fcdc4c1353ca53196025d3e12a6.jpg)

![43201142a5e65e4f2b6e2ba94056601666e5801a7616f80bae5fcba90e055539.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/43201142a5e65e4f2b6e2ba94056601666e5801a7616f80bae5fcba90e055539.jpg)

![4e8953523e9330975c1968d7a34ad69664b38da0ba8b63d82a173c80112cbd3c.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/4e8953523e9330975c1968d7a34ad69664b38da0ba8b63d82a173c80112cbd3c.jpg)

![67b693849dbfcc0f704a356272e9b80a1af1d79854523e7b06ae5834efbc0498.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/67b693849dbfcc0f704a356272e9b80a1af1d79854523e7b06ae5834efbc0498.jpg)

![7093be8bfdd82a14bdf2c9938420f1f5ff448283603a72188c3f93929b442b15.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/7093be8bfdd82a14bdf2c9938420f1f5ff448283603a72188c3f93929b442b15.jpg)

![7a5b5c75c871ca0a0257f9273ca372a146d84a3ab1d930cd5931abaaf7694295.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/7a5b5c75c871ca0a0257f9273ca372a146d84a3ab1d930cd5931abaaf7694295.jpg)

![87110eeedf08b729052c69e87015ab54a3b64d13c31d9fff8f2dd05beccaa504.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/87110eeedf08b729052c69e87015ab54a3b64d13c31d9fff8f2dd05beccaa504.jpg)

![964017dbfa2f2eb0fb89de7fc449ad2a2f743ca1f4753fa38b376a14f2dc8fa5.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/964017dbfa2f2eb0fb89de7fc449ad2a2f743ca1f4753fa38b376a14f2dc8fa5.jpg)

![a660a9fddd081edbe06757ff4679f5337d55022c87525c1fa326c78f3fcb0362.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/a660a9fddd081edbe06757ff4679f5337d55022c87525c1fa326c78f3fcb0362.jpg)

![a81d63cb6caf2e6518170d803bc6d86d026e1dfb11de3d995ab697c12da9a79b.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/a81d63cb6caf2e6518170d803bc6d86d026e1dfb11de3d995ab697c12da9a79b.jpg)

![bb1516cb2bc3bbd91607d7db8bdd342416c23615c54021fd68e9ef851148467c.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/bb1516cb2bc3bbd91607d7db8bdd342416c23615c54021fd68e9ef851148467c.jpg)

![c89b80424aa26239e9b42aff7cb6621079f8959deb9c280e5848c6053e054856.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/images/c89b80424aa26239e9b42aff7cb6621079f8959deb9c280e5848c6053e054856.jpg)

### Tables

![7d9c85523f5cd4112dc392e6990acb5518073e6b4050d5b61d5f872b76662d57.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/tables/7d9c85523f5cd4112dc392e6990acb5518073e6b4050d5b61d5f872b76662d57.jpg)

![ae5fa5044b1dfd39789c0842ce0768e864693a986dc81c569375635efadd2b4f.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/tables/ae5fa5044b1dfd39789c0842ce0768e864693a986dc81c569375635efadd2b4f.jpg)

![f4d6352c2d81bd21834c2711eba5177b13337128713c38591db321e05d0ac3f7.jpg](../icml_results/1441_Relational%20Structural%20Causal%20Models/tables/f4d6352c2d81bd21834c2711eba5177b13337128713c38591db321e05d0ac3f7.jpg)

<a id="paper-1422"></a>

## Learning Treatment Allocations with Risk Control Under Partial Identifiability


### Images

![0539ea3abfdcda3fb4a9c2d1de6cc3bc3234fdf10a22050080527a25a0a3a671.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/0539ea3abfdcda3fb4a9c2d1de6cc3bc3234fdf10a22050080527a25a0a3a671.jpg)

![1fcc987a69a14247d5f0b5653f1063cb18453c115dc9b58c78d821f5603c9686.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/1fcc987a69a14247d5f0b5653f1063cb18453c115dc9b58c78d821f5603c9686.jpg)

![343f953d52a915cb953dc2d979bdcf0b04d92a594694b0b9061fd60906c8b28f.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/343f953d52a915cb953dc2d979bdcf0b04d92a594694b0b9061fd60906c8b28f.jpg)

![4ca7429df56bbae9d7f784c8d84a9e6a2f103ff6a0eeda2826ea672f89f68239.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/4ca7429df56bbae9d7f784c8d84a9e6a2f103ff6a0eeda2826ea672f89f68239.jpg)

![72ea790fdac15567fc54470e94f15eb780a103e19a2325e83da112a6d3d50273.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/72ea790fdac15567fc54470e94f15eb780a103e19a2325e83da112a6d3d50273.jpg)

![a3d036aa8e404749308c8ab7915603aaf5d7c5af7313c04d650352442f885461.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/a3d036aa8e404749308c8ab7915603aaf5d7c5af7313c04d650352442f885461.jpg)

![b5f1ce24c9d0a839f895282ecec1e55a429d7cd04c58c16d269210bf7234a93d.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/b5f1ce24c9d0a839f895282ecec1e55a429d7cd04c58c16d269210bf7234a93d.jpg)

![cbe076e0e8d5c0adcf4fa6bfc6873de7b395b23f9a89dba7ba44c835ebe4e856.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/cbe076e0e8d5c0adcf4fa6bfc6873de7b395b23f9a89dba7ba44c835ebe4e856.jpg)

![d72399974d1bd2969a421b9a1a7f0da2d118f4dc5077bfc56eeefdb5a68fc81b.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/d72399974d1bd2969a421b9a1a7f0da2d118f4dc5077bfc56eeefdb5a68fc81b.jpg)

![e07b295d7a2739fa9f8f4f614b5a8b64538846428ce23306d419e69e6f0929d8.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/e07b295d7a2739fa9f8f4f614b5a8b64538846428ce23306d419e69e6f0929d8.jpg)

![f027124089d4dd8d588c1721fa17d55763089bbc03163c1ddf8648a1b68e0d5e.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/images/f027124089d4dd8d588c1721fa17d55763089bbc03163c1ddf8648a1b68e0d5e.jpg)

### Tables

![f24d672277c9f7b5d6b3b68c0bb541a7c5af3f82af90c85c99cac08d78926300.jpg](../icml_results/1442_Learning%20Treatment%20Allocations%20with%20Risk%20Control%20Under%20Partial%20Identifiability/tables/f24d672277c9f7b5d6b3b68c0bb541a7c5af3f82af90c85c99cac08d78926300.jpg)

<a id="paper-1423"></a>

## Revisiting Efficiency–Accuracy Scaling in Mixture-of-Experts Architectures


### Images

![18a29c841970c37487790da1884993b2a3178615d2f725dde21397ffebf5127a.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/images/18a29c841970c37487790da1884993b2a3178615d2f725dde21397ffebf5127a.jpg)

![2123e543eb7a646956d1a7b638e7701661f44b45d659a9f7f6316078de119d6e.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/images/2123e543eb7a646956d1a7b638e7701661f44b45d659a9f7f6316078de119d6e.jpg)

![4db9384809d679d57b3cde4d666e7b11f9ed40a7eef74e27fc7ba2cedefdbd54.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/images/4db9384809d679d57b3cde4d666e7b11f9ed40a7eef74e27fc7ba2cedefdbd54.jpg)

![8ef85a9d15a1b831d6fd98cde249506bcf24671c636b49146237a4ed80c9ee50.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/images/8ef85a9d15a1b831d6fd98cde249506bcf24671c636b49146237a4ed80c9ee50.jpg)

![ce0e5efe2db6e6cc8211bdcce76886648c8ff8e98394d6ea4d1352abecaa6603.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/images/ce0e5efe2db6e6cc8211bdcce76886648c8ff8e98394d6ea4d1352abecaa6603.jpg)

### Tables

![14a00710b4979d59e648f8ab68bfc81cd6a88125626fa50dee0c181cc286cc64.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/14a00710b4979d59e648f8ab68bfc81cd6a88125626fa50dee0c181cc286cc64.jpg)

![36b35d9669abad05ce98cb129f388aeca143599cd698659876de52c14a65d642.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/36b35d9669abad05ce98cb129f388aeca143599cd698659876de52c14a65d642.jpg)

![8877880ba2029237eb296b6da8cdf8d3bf08bdb8328db0643bf76d0c329029ff.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/8877880ba2029237eb296b6da8cdf8d3bf08bdb8328db0643bf76d0c329029ff.jpg)

![a677fd08ca434cfd11f5f61aa4faa7155553f8cc577adcec767875e30fb49b2b.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/a677fd08ca434cfd11f5f61aa4faa7155553f8cc577adcec767875e30fb49b2b.jpg)

![adbe4141e73e1f91c12758b52f35f4158d6988a46c1bfd95e234c0308d5d3b6e.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/adbe4141e73e1f91c12758b52f35f4158d6988a46c1bfd95e234c0308d5d3b6e.jpg)

![f57cb1ffacc764da25f7bcc75bbe6a884b1245997607c22871669f265cbe59b4.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/f57cb1ffacc764da25f7bcc75bbe6a884b1245997607c22871669f265cbe59b4.jpg)

![fba0aa2aa9302ee1f9c8a0cdf9f19783822ee1eb63cd80bc89370b3f8e54992d.jpg](../icml_results/1443_Revisiting%20Efficiency%E2%80%93Accuracy%20Scaling%20in%20Mixture-of-Experts%20Architectures/tables/fba0aa2aa9302ee1f9c8a0cdf9f19783822ee1eb63cd80bc89370b3f8e54992d.jpg)

<a id="paper-1424"></a>

## MAD: Manifold Attracted Diffusion


### Images

![0b3b576ffa00e15723fc38cb52ccd058fa89ea8b00f43912df67dd495ef24413.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/0b3b576ffa00e15723fc38cb52ccd058fa89ea8b00f43912df67dd495ef24413.jpg)

![17df85012a8bc0e63524a9edffce76cb07b5e7e813a53c7b5c23bd8651eddb98.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/17df85012a8bc0e63524a9edffce76cb07b5e7e813a53c7b5c23bd8651eddb98.jpg)

![1c45d81371a5e0b2f7d3bd07228b462d2053236d92b0c94bf4206cd9b12b530b.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/1c45d81371a5e0b2f7d3bd07228b462d2053236d92b0c94bf4206cd9b12b530b.jpg)

![2153db2371f39f62ffbc53271fb1980a7c4b8b4aa827ad9e895fe0705986faec.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/2153db2371f39f62ffbc53271fb1980a7c4b8b4aa827ad9e895fe0705986faec.jpg)

![3b5afe31b51978d97fb6e17297c4d1e684aa3b96af861ab05539ed7621e47fa1.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/3b5afe31b51978d97fb6e17297c4d1e684aa3b96af861ab05539ed7621e47fa1.jpg)

![48e444534c90b6aa2592f87dc2babc0251bcf22fb3940c4b8ad3f16fd2945f78.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/48e444534c90b6aa2592f87dc2babc0251bcf22fb3940c4b8ad3f16fd2945f78.jpg)

![4c7f2d5d4d694380ed0f84d0b896d94a3bdb330a0715bf6a93cb6ee135f5b69d.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/4c7f2d5d4d694380ed0f84d0b896d94a3bdb330a0715bf6a93cb6ee135f5b69d.jpg)

![622e854a72f90f1e7fee00bfa4a7232575f2f5b9da02889255515dd3c2119e5e.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/622e854a72f90f1e7fee00bfa4a7232575f2f5b9da02889255515dd3c2119e5e.jpg)

![63c342524d0ec01d1f49b66cde0b43e5ee1f290a716f2aa046b6efd2449b1c94.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/63c342524d0ec01d1f49b66cde0b43e5ee1f290a716f2aa046b6efd2449b1c94.jpg)

![8092c8d2aca693cecf0032452a7156b01dcde30bce70cf4e7d5c5fc9fa92a612.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/8092c8d2aca693cecf0032452a7156b01dcde30bce70cf4e7d5c5fc9fa92a612.jpg)

![882f2e9968a1daadf6abd7ea239773fca19c21ae0cc9c27902eebee01b9df06e.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/882f2e9968a1daadf6abd7ea239773fca19c21ae0cc9c27902eebee01b9df06e.jpg)

![99644e434458f0c1206312c1b17a24a2358a84eb69c6fd4567cfde0633be9783.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/99644e434458f0c1206312c1b17a24a2358a84eb69c6fd4567cfde0633be9783.jpg)

![a987a18ab1384863925940d2e58fc99f04784c0c84c9b1c5a7f9b105b7f6f19c.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/a987a18ab1384863925940d2e58fc99f04784c0c84c9b1c5a7f9b105b7f6f19c.jpg)

![a99764e9bbb07c79e137444f528af536590658f9ff632f170cdf9d4b17b96706.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/a99764e9bbb07c79e137444f528af536590658f9ff632f170cdf9d4b17b96706.jpg)

![ab49008c4c54b89d156ce22a403c9b1f1bfb70b083c861d49702a500ff085505.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/ab49008c4c54b89d156ce22a403c9b1f1bfb70b083c861d49702a500ff085505.jpg)

![b1760cba9ab3e6ac441c2a9d77465bc754c9e75a0efa7218c15ce7f18a91eace.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/b1760cba9ab3e6ac441c2a9d77465bc754c9e75a0efa7218c15ce7f18a91eace.jpg)

![d5275dbf7538bd411a8e91351045ed3d768f18998067002fc04b28f317764bbd.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/d5275dbf7538bd411a8e91351045ed3d768f18998067002fc04b28f317764bbd.jpg)

![e5206debb51c714e136ba4c4074867d523839d00eacd1b7e828cd053f4812caa.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/e5206debb51c714e136ba4c4074867d523839d00eacd1b7e828cd053f4812caa.jpg)

![f38dc9806a31595e447e725b4373ca9fca0f75dae5cc3819a1a4f3e06dde6f73.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/images/f38dc9806a31595e447e725b4373ca9fca0f75dae5cc3819a1a4f3e06dde6f73.jpg)

### Tables

![3aa1997613c3d570f349018af478ba1efb2c9395a43df2e83ef63d2d1c79a91d.jpg](../icml_results/1444_MAD_%20Manifold%20Attracted%20Diffusion/tables/3aa1997613c3d570f349018af478ba1efb2c9395a43df2e83ef63d2d1c79a91d.jpg)

<a id="paper-1425"></a>

## Don’t Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference


### Images

![0d47e88bb598115ef1ef44cb443cbc263dfe50b981689090d38c395f2cba75ed.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/0d47e88bb598115ef1ef44cb443cbc263dfe50b981689090d38c395f2cba75ed.jpg)

![310f73b489c912bb4d273409be031d050459e79833272a997bebb0574f9ef393.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/310f73b489c912bb4d273409be031d050459e79833272a997bebb0574f9ef393.jpg)

![490c6375a85d1909e5b0205cb938b84fd72a5a33e2110c412a9f07c11f9edc8e.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/490c6375a85d1909e5b0205cb938b84fd72a5a33e2110c412a9f07c11f9edc8e.jpg)

![4c36d2565455a63a1f1c71e79b4856b5133e5f0e9c377266f9516af2936f6ce7.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/4c36d2565455a63a1f1c71e79b4856b5133e5f0e9c377266f9516af2936f6ce7.jpg)

![75dead4db57e3a2bec16b94c5bd4b9b1db7a48e4efb200166fe28a3544866c17.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/75dead4db57e3a2bec16b94c5bd4b9b1db7a48e4efb200166fe28a3544866c17.jpg)

![94e78f266f3cd38a129a8ac27aadc20d897f4e4ff9153ae2b38a6554d665becf.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/94e78f266f3cd38a129a8ac27aadc20d897f4e4ff9153ae2b38a6554d665becf.jpg)

![993154a31c14367c898352a6405975b617447a277ce7fa7aaf709aa55d6723de.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/993154a31c14367c898352a6405975b617447a277ce7fa7aaf709aa55d6723de.jpg)

![bcc9a46879bb87a8766d0ee3a08bbed34ebdb4d16c1bc2a995177b20873d3e64.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/bcc9a46879bb87a8766d0ee3a08bbed34ebdb4d16c1bc2a995177b20873d3e64.jpg)

![c790382a34cc2a3916a0e2d4554e8446f537e7f4a451b64e63df5a8187fe7c06.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/c790382a34cc2a3916a0e2d4554e8446f537e7f4a451b64e63df5a8187fe7c06.jpg)

![d653226d2ccf56f78f547c66fbe191865284b9c5f2f72079188a050e0aa2dd6c.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/d653226d2ccf56f78f547c66fbe191865284b9c5f2f72079188a050e0aa2dd6c.jpg)

![fb65cc95305fe6c0d20ce4ddb2fc278796ea98151bde49fd9cebafec6d5be995.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/fb65cc95305fe6c0d20ce4ddb2fc278796ea98151bde49fd9cebafec6d5be995.jpg)

![fb67aa79210205e4fdb49d2a32fd07327bd7c0d4dcf756fc89a5efefb4296b92.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/images/fb67aa79210205e4fdb49d2a32fd07327bd7c0d4dcf756fc89a5efefb4296b92.jpg)

### Tables

![0db20e52ba3a705d189ea29ff2be444b07a1795e9d366e2ccf8a1bc09be11ea7.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/0db20e52ba3a705d189ea29ff2be444b07a1795e9d366e2ccf8a1bc09be11ea7.jpg)

![17a559ea7f5adaa3f40f74d3b8f1ee01ec3413163af7e1f428680daa87056e47.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/17a559ea7f5adaa3f40f74d3b8f1ee01ec3413163af7e1f428680daa87056e47.jpg)

![2dbbf6798dfae310877e0dbe9e732f6792f6ba27a9eaa3ced0c6151a0786e634.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/2dbbf6798dfae310877e0dbe9e732f6792f6ba27a9eaa3ced0c6151a0786e634.jpg)

![44f17963c94059058ecbca8caf986b6d96ad30435fd78cb0177173e7ad362909.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/44f17963c94059058ecbca8caf986b6d96ad30435fd78cb0177173e7ad362909.jpg)

![7d97b475f94ebea35bde48178add175aca8dc643328c0fbd4aeff963f86dcd01.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/7d97b475f94ebea35bde48178add175aca8dc643328c0fbd4aeff963f86dcd01.jpg)

![9add83e0c951976858494ff9f96e2237cf2e7ad19ed3a4ed6531129eb0c38156.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/9add83e0c951976858494ff9f96e2237cf2e7ad19ed3a4ed6531129eb0c38156.jpg)

![be7e9beb9f4954f91af4ef401612b3c65463b1738bb0bcd333a62fb51f285a60.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/be7e9beb9f4954f91af4ef401612b3c65463b1738bb0bcd333a62fb51f285a60.jpg)

![fb08b8520331262de7345dcc9841697079effbf8392fe4142a17ef67f5500e9a.jpg](../icml_results/1445_Don%E2%80%99t%20Drop%20Dropout_%20Optimizing%20Layer%20Sparsity%20for%20Efficient%20LLM%20Training%20and%20Inference/tables/fb08b8520331262de7345dcc9841697079effbf8392fe4142a17ef67f5500e9a.jpg)

<a id="paper-1426"></a>

## TSP with Predictions: Heatmap to Tour with Provable Guarantees


### Images

![0271b9a9836acb1ef3081c987cb4c9ab47ad80a2be24d63a505a77515257513a.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/0271b9a9836acb1ef3081c987cb4c9ab47ad80a2be24d63a505a77515257513a.jpg)

![0a2de1640039061efce3adcbd67080f1f698aecee6d8b41b9858e95cbfab43d5.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/0a2de1640039061efce3adcbd67080f1f698aecee6d8b41b9858e95cbfab43d5.jpg)

![44d34cce6c9ad1a573485c39856c6ece6b0939d1f544e945dd3ba4ec2b1c8484.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/44d34cce6c9ad1a573485c39856c6ece6b0939d1f544e945dd3ba4ec2b1c8484.jpg)

![645c9dc8407231ba6cc50ec878b73df8583dbc43293672599ede555f8169d60e.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/645c9dc8407231ba6cc50ec878b73df8583dbc43293672599ede555f8169d60e.jpg)

![68e7b7489149dba14f865211854cf0c0455e0ac9c50604a0ac9c3d988bbcc201.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/68e7b7489149dba14f865211854cf0c0455e0ac9c50604a0ac9c3d988bbcc201.jpg)

![72f1a2351b6fa7425beea0f27d43c8da80982d3ca5a683f0da4fbfabf10635cc.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/72f1a2351b6fa7425beea0f27d43c8da80982d3ca5a683f0da4fbfabf10635cc.jpg)

![751cb11a4bf4352728892a9dfd6fd4acda49a7898ce9ac4f5e8c17b1c6b5e73e.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/751cb11a4bf4352728892a9dfd6fd4acda49a7898ce9ac4f5e8c17b1c6b5e73e.jpg)

![7b45d9d7fff8a103c15b664df658e982261fa552978e696a9c2204d1dabd38e0.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/7b45d9d7fff8a103c15b664df658e982261fa552978e696a9c2204d1dabd38e0.jpg)

![826b6590bbdc2f97cab9b29b58a24ce1ed0dc2c76f8e6d5694bd57d34be5eeaf.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/826b6590bbdc2f97cab9b29b58a24ce1ed0dc2c76f8e6d5694bd57d34be5eeaf.jpg)

![b16d2b10d674345a64ee93bf843a38e895afe1f1153abe79bc16770923d341d0.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/b16d2b10d674345a64ee93bf843a38e895afe1f1153abe79bc16770923d341d0.jpg)

![c64120a4390d734db077d0e96d36a73a44cc97c7d0c015c8a7c918f3a7184502.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/c64120a4390d734db077d0e96d36a73a44cc97c7d0c015c8a7c918f3a7184502.jpg)

![c721abd146e94a6afe46bc5423e1f7cb6aa6cbf8537b6bea21b70ca413237d6e.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/c721abd146e94a6afe46bc5423e1f7cb6aa6cbf8537b6bea21b70ca413237d6e.jpg)

![de8ef67be5f3ba397c9810e2884039cde27f28f344de159369660a1f3930151b.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/de8ef67be5f3ba397c9810e2884039cde27f28f344de159369660a1f3930151b.jpg)

![e2a61cd9967e07d2381e89a2716a65f220ce36daae851044b9601e47ae52dc93.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/e2a61cd9967e07d2381e89a2716a65f220ce36daae851044b9601e47ae52dc93.jpg)

![f2a2195229d27040fc27b18a255687103f984d914d0d57f72e6dd771272896c3.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/f2a2195229d27040fc27b18a255687103f984d914d0d57f72e6dd771272896c3.jpg)

![f47fe2d0b483ce28633ea66826beb1a101da6141f148de59c2164b6642939540.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/images/f47fe2d0b483ce28633ea66826beb1a101da6141f148de59c2164b6642939540.jpg)

### Tables

![5061d8c0f6de89fc3ffaf99df69883252b85ef972eb8711120841ff0c8b32d51.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/tables/5061d8c0f6de89fc3ffaf99df69883252b85ef972eb8711120841ff0c8b32d51.jpg)

![6c1a2fae375edc919a41696b12e3ffa11cccad1add6ad24b237858ca29fa7a17.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/tables/6c1a2fae375edc919a41696b12e3ffa11cccad1add6ad24b237858ca29fa7a17.jpg)

![ba5b9225206a612d5e06eb12a701783e1447feaf448767c13d749f68258de6a9.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/tables/ba5b9225206a612d5e06eb12a701783e1447feaf448767c13d749f68258de6a9.jpg)

![f50ca9b015c3f0f3f9cfacf5afdf87291d3e8ea3e43079c1ac743fd817126128.jpg](../icml_results/1446_TSP%20with%20Predictions_%20Heatmap%20to%20Tour%20with%20Provable%20Guarantees/tables/f50ca9b015c3f0f3f9cfacf5afdf87291d3e8ea3e43079c1ac743fd817126128.jpg)

<a id="paper-1427"></a>

## Graph Neural Dynamics via Learned Energy and Tangential Flows


### Images

![7750b85c52e5c73097564d20698b455394f3de1e5108a92019c93c6a8a625460.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/images/7750b85c52e5c73097564d20698b455394f3de1e5108a92019c93c6a8a625460.jpg)

![8fc2fb945b86a0be030431be3d5b8fe8d2e516c82bf0da86a0b3cd1e6547ca8d.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/images/8fc2fb945b86a0be030431be3d5b8fe8d2e516c82bf0da86a0b3cd1e6547ca8d.jpg)

### Tables

![19477e8e6480f0fd8fa67dcbce845c4b228e5395d35218cb7ecc95b0c5a99eeb.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/19477e8e6480f0fd8fa67dcbce845c4b228e5395d35218cb7ecc95b0c5a99eeb.jpg)

![1e58aeb9881017dddc80e86d0a0ef49995f36df26211172884c6f6d8c633935c.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/1e58aeb9881017dddc80e86d0a0ef49995f36df26211172884c6f6d8c633935c.jpg)

![2f8329dc12b50bc57733cdb4befc431b39432c81c2b2cddb703271c0be1c70d7.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/2f8329dc12b50bc57733cdb4befc431b39432c81c2b2cddb703271c0be1c70d7.jpg)

![45bdc21d26e80b1fcd9e90a34e1da3e4f49d2abab6aff287efd4d603c952f4e6.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/45bdc21d26e80b1fcd9e90a34e1da3e4f49d2abab6aff287efd4d603c952f4e6.jpg)

![4ae19ede9e53b7fe21a8f30afc2ed5369eba6f9b4f917d62b49c5e0896efe361.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/4ae19ede9e53b7fe21a8f30afc2ed5369eba6f9b4f917d62b49c5e0896efe361.jpg)

![505986d74931398e9bea96c1dd36bfdfcec94fdc5e4f6b638828ad4813b10592.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/505986d74931398e9bea96c1dd36bfdfcec94fdc5e4f6b638828ad4813b10592.jpg)

![8392dbdc328fdb12c1eb8230ff66777474e19044c6d531d9a8279d5f75360e32.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/8392dbdc328fdb12c1eb8230ff66777474e19044c6d531d9a8279d5f75360e32.jpg)

![a4ebf92ac0a1752fe505522c93239dac95b04d7d146fe26aa575da9c4fdbac72.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/a4ebf92ac0a1752fe505522c93239dac95b04d7d146fe26aa575da9c4fdbac72.jpg)

![b256351c62797cafd101b901e7e8d4d677428e2e696e5f9cc4208e04c07fb40e.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/b256351c62797cafd101b901e7e8d4d677428e2e696e5f9cc4208e04c07fb40e.jpg)

![b57cd5eb65ecbc01fb571ba4df4341e5fc00deb8acf6038e35b8de420c67e036.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/b57cd5eb65ecbc01fb571ba4df4341e5fc00deb8acf6038e35b8de420c67e036.jpg)

![b947b3f35abb56d8219031ec3825f9c5f07e6a2148c95a12a5792dc787f5c519.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/b947b3f35abb56d8219031ec3825f9c5f07e6a2148c95a12a5792dc787f5c519.jpg)

![bdc74789b36b5a23025683caeffc81bb15e0f31200c2ee9d1d46ba22ef7cad6d.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/bdc74789b36b5a23025683caeffc81bb15e0f31200c2ee9d1d46ba22ef7cad6d.jpg)

![c1b0fbc93fac099d5ffe96fb964a080b3827cc45089469f6ddf2f3734c5716bc.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/c1b0fbc93fac099d5ffe96fb964a080b3827cc45089469f6ddf2f3734c5716bc.jpg)

![cbcd003f4d335a8cd9053ab6142a52ba80c93ff5b47a59c9fef70dccf06db8e6.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/cbcd003f4d335a8cd9053ab6142a52ba80c93ff5b47a59c9fef70dccf06db8e6.jpg)

![d83d121d58a384672de6d5faaced0edc9a184449a202d3db026849c06e8ea13d.jpg](../icml_results/1447_Graph%20Neural%20Dynamics%20via%20Learned%20Energy%20and%20Tangential%20Flows/tables/d83d121d58a384672de6d5faaced0edc9a184449a202d3db026849c06e8ea13d.jpg)

<a id="paper-1428"></a>

## ConEx: Human-Interpretable Saliency Maps via Concept-Aware Attribution


### Images

![8934ec7b930880c63f3abe82cc16c6dce39ca4ab27db3442221a4d59f64fbd68.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/images/8934ec7b930880c63f3abe82cc16c6dce39ca4ab27db3442221a4d59f64fbd68.jpg)

![d93b2f1516c848dae6b4ea2065df14459cd07fee127dc88d3776adf5a3d68580.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/images/d93b2f1516c848dae6b4ea2065df14459cd07fee127dc88d3776adf5a3d68580.jpg)

![fc5ea7f68a166e18ae77773564bbf60bada1e5aaa1ee3bd42b1912b41354868b.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/images/fc5ea7f68a166e18ae77773564bbf60bada1e5aaa1ee3bd42b1912b41354868b.jpg)

### Tables

![102146add97f9d786e8c9e65b2895d5ecec2837b844b576d7bb69a98e73f3a26.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/102146add97f9d786e8c9e65b2895d5ecec2837b844b576d7bb69a98e73f3a26.jpg)

![80bfb9a98fb3ef19caec33b0976e08e8ae5df2e32b776744d6d07d7177a17d55.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/80bfb9a98fb3ef19caec33b0976e08e8ae5df2e32b776744d6d07d7177a17d55.jpg)

![93d339bdf12d10731e59774be6b8701c77458e40a9c05259ff63ebc414992fa2.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/93d339bdf12d10731e59774be6b8701c77458e40a9c05259ff63ebc414992fa2.jpg)

![b081082a5d4e093fcc0a3c04ca3f5524f23fcbc45a6110da0f66b9b53988adef.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/b081082a5d4e093fcc0a3c04ca3f5524f23fcbc45a6110da0f66b9b53988adef.jpg)

![ba5ca21727943385036c65621efd862a74553b3e255aa733e9cfe00e817f8e79.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/ba5ca21727943385036c65621efd862a74553b3e255aa733e9cfe00e817f8e79.jpg)

![bd35b8c2f02af88bad6ba7dfe837f132d03da2f2a46c43b2cf240d3c5c8563ac.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/bd35b8c2f02af88bad6ba7dfe837f132d03da2f2a46c43b2cf240d3c5c8563ac.jpg)

![c47658624876ac657a18747929e1e616472171d3eb3804a85e7dbd0daacc3c35.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/c47658624876ac657a18747929e1e616472171d3eb3804a85e7dbd0daacc3c35.jpg)

![d1271f454c02e395ae940c27c1d3d626b0622f3f1494c20910095b497c9a4be3.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/d1271f454c02e395ae940c27c1d3d626b0622f3f1494c20910095b497c9a4be3.jpg)

![d4f495eea9caf423b389343c7f39fb0d5710e2619c033fcc7fc183a3fb161c7c.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/d4f495eea9caf423b389343c7f39fb0d5710e2619c033fcc7fc183a3fb161c7c.jpg)

![fa7e03720eedcb9eb2694ea1f9d57998fd3207ef12cea9905785e4cad156808c.jpg](../icml_results/1448_ConEx_%20Human-Interpretable%20Saliency%20Maps%20via%20Concept-Aware%20Attribution/tables/fa7e03720eedcb9eb2694ea1f9d57998fd3207ef12cea9905785e4cad156808c.jpg)

<a id="paper-1429"></a>

## Hide&Seek: Learning to Explain in an End-to-End Differentiable Network


### Images

![225a566bf7811f73826092c6e11145e92b6e2add21def5137a189ed3f7f6a045.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/225a566bf7811f73826092c6e11145e92b6e2add21def5137a189ed3f7f6a045.jpg)

![3967e4ec1497aed04039a41e783c22cc4228f499a44b50e892007ce59a2bfc34.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/3967e4ec1497aed04039a41e783c22cc4228f499a44b50e892007ce59a2bfc34.jpg)

![438d106170d5778c54432c89bb4ac4474fed0fc78df1a914addd883ff9ab06b9.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/438d106170d5778c54432c89bb4ac4474fed0fc78df1a914addd883ff9ab06b9.jpg)

![52f8573731b605751ae1bde61e5c71cbd2dc8c3802d2435ea771faa0529947ce.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/52f8573731b605751ae1bde61e5c71cbd2dc8c3802d2435ea771faa0529947ce.jpg)

![6ab3f71247e3e3d7df83dabb1e429f1fa76b7f63b4b2ccec0ffb738173d600d1.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/6ab3f71247e3e3d7df83dabb1e429f1fa76b7f63b4b2ccec0ffb738173d600d1.jpg)

![755a132e0beabad89ba46589bf6875a467a494bfda1b5ff541ee7fd4f2ee10c3.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/755a132e0beabad89ba46589bf6875a467a494bfda1b5ff541ee7fd4f2ee10c3.jpg)

![9af92b7c0c2eb30567f7ef87f52406f370c7e690d8084520cb42c03e483fbbc4.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/9af92b7c0c2eb30567f7ef87f52406f370c7e690d8084520cb42c03e483fbbc4.jpg)

![bcdf6c505172522a1677ff3397b7e3a4d3674e53453fde97c9334d71c2316986.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/bcdf6c505172522a1677ff3397b7e3a4d3674e53453fde97c9334d71c2316986.jpg)

![cd8ea48227f9ef50e56572a53d3267d70ea717fb2f6ab7aba19dc6e99f3a771f.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/cd8ea48227f9ef50e56572a53d3267d70ea717fb2f6ab7aba19dc6e99f3a771f.jpg)

![d2c84eaf67238219b8f82ae6b024e8b44025b62d8f299079c4dc9cca939d8c8d.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/d2c84eaf67238219b8f82ae6b024e8b44025b62d8f299079c4dc9cca939d8c8d.jpg)

![dfab7ef4dd19cd7d1ff87107f2a84f2dc10ab724b28610a3884537f50d61efca.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/dfab7ef4dd19cd7d1ff87107f2a84f2dc10ab724b28610a3884537f50d61efca.jpg)

![e83904960d437ffaf28916d6fc86a3546afc139f75b58245f95dedb5b9e9ece4.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/e83904960d437ffaf28916d6fc86a3546afc139f75b58245f95dedb5b9e9ece4.jpg)

![ea654338d5d9071e5571d0d2e685bbaa235839bd1b33842237dc3f5e8fddbf9f.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/images/ea654338d5d9071e5571d0d2e685bbaa235839bd1b33842237dc3f5e8fddbf9f.jpg)

### Tables

![09da2ac9af72e9c28e326f27fdf207c9cd095a037e834a852ed17bee88afa39f.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/09da2ac9af72e9c28e326f27fdf207c9cd095a037e834a852ed17bee88afa39f.jpg)

![1cac7e8ab2e9f00ba8b758949bdb7db56d1766fa6606a39bc82833fb8642ac36.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/1cac7e8ab2e9f00ba8b758949bdb7db56d1766fa6606a39bc82833fb8642ac36.jpg)

![52d8e1d67698dddc698fc8fa4864faaf46f71de58400af47b6239351435becf9.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/52d8e1d67698dddc698fc8fa4864faaf46f71de58400af47b6239351435becf9.jpg)

![5e71ced826a1e35cbf2aeaa104ef61396f1c807d7f1b3f00421634425644affa.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/5e71ced826a1e35cbf2aeaa104ef61396f1c807d7f1b3f00421634425644affa.jpg)

![76d211aadadb271df5ca3936174a4528ad8a4377fb271839d2815af0694837f4.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/76d211aadadb271df5ca3936174a4528ad8a4377fb271839d2815af0694837f4.jpg)

![78cab0c53e7af3733d4d396d997de673ab952c7addcd5db3b9bab22006517db7.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/78cab0c53e7af3733d4d396d997de673ab952c7addcd5db3b9bab22006517db7.jpg)

![8ed16fdc3b5a39e61db8c4ce20a1708e46d0510692c18c78d4c3b0416e2f72eb.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/8ed16fdc3b5a39e61db8c4ce20a1708e46d0510692c18c78d4c3b0416e2f72eb.jpg)

![9567ff8835516ecf58572a103eab493fca3b2ffe7cb1dc7011c3f543a7ea5a7f.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/9567ff8835516ecf58572a103eab493fca3b2ffe7cb1dc7011c3f543a7ea5a7f.jpg)

![a70b2cda2762261b4daacdb3163fda7f68b27a41636eb3c7df6357a83b0437fa.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/a70b2cda2762261b4daacdb3163fda7f68b27a41636eb3c7df6357a83b0437fa.jpg)

![bacd5da8e292e3a83162936cfa14c56605f1071b02ca5b8e4d83c0edb6064eeb.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/bacd5da8e292e3a83162936cfa14c56605f1071b02ca5b8e4d83c0edb6064eeb.jpg)

![d64c565010ffeb5b7c531f993f241b27f4e50d963d991c99baa6c2daf45789b9.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/d64c565010ffeb5b7c531f993f241b27f4e50d963d991c99baa6c2daf45789b9.jpg)

![dee0ba89d8462b7c0d4250968fe960ce422c0c364e2b83899a4e7f5435495f06.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/dee0ba89d8462b7c0d4250968fe960ce422c0c364e2b83899a4e7f5435495f06.jpg)

![ea12254143f59066d8aed359e0298310f87437ac1f5c4bc03f36b9d974edcb82.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/ea12254143f59066d8aed359e0298310f87437ac1f5c4bc03f36b9d974edcb82.jpg)

![ebf447f09507bdd1381ee21047b2bbbafff76d41422283883f5d6bc0b9477dbd.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/ebf447f09507bdd1381ee21047b2bbbafff76d41422283883f5d6bc0b9477dbd.jpg)

![edab24da0c73ca87cf5e9d639a1e9057ff04ea7abd3dab3120d4e52e207a3ab7.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/edab24da0c73ca87cf5e9d639a1e9057ff04ea7abd3dab3120d4e52e207a3ab7.jpg)

![fedd49256a61ccdea7f16fcfb4c8e5700cecee1c24c823a09b6ead894e89d34e.jpg](../icml_results/1449_Hide%26Seek_%20Learning%20to%20Explain%20in%20an%20End-to-End%20Differentiable%20Network/tables/fedd49256a61ccdea7f16fcfb4c8e5700cecee1c24c823a09b6ead894e89d34e.jpg)

<a id="paper-1430"></a>

## Training LLM Agents to Empower Humans


### Images

![1d2eecd77515a1b5a5a4d732304cc62771ba9b97d2eaac48b052238c2fdeddcd.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/1d2eecd77515a1b5a5a4d732304cc62771ba9b97d2eaac48b052238c2fdeddcd.jpg)

![32c49bea2fe8654e29d3b9aadb03f7f1140244ffe3b3c0af02075f3a7f842606.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/32c49bea2fe8654e29d3b9aadb03f7f1140244ffe3b3c0af02075f3a7f842606.jpg)

![35f57a3b975a6e2b95473f411a2ce772e97a10742ba462904b4d69fd8e9e2cd8.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/35f57a3b975a6e2b95473f411a2ce772e97a10742ba462904b4d69fd8e9e2cd8.jpg)

![424ddd121a0914c1516b5ce00efc738555418b83de3de4aa52fa41bcc4a61b3b.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/424ddd121a0914c1516b5ce00efc738555418b83de3de4aa52fa41bcc4a61b3b.jpg)

![4500ccecacd3ce9b62dcf05f3c5a6b542f575dd0757a59a5713d687ab98b7aae.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/4500ccecacd3ce9b62dcf05f3c5a6b542f575dd0757a59a5713d687ab98b7aae.jpg)

![75b43bc9f6a2fc869e9241943d6862659e71a0255730276316b9ec5d35f8aa96.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/75b43bc9f6a2fc869e9241943d6862659e71a0255730276316b9ec5d35f8aa96.jpg)

![b30199261461aef31d3f658e9310e61924c625f9ae3815a95cf9021bbe1c093a.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/b30199261461aef31d3f658e9310e61924c625f9ae3815a95cf9021bbe1c093a.jpg)

![cc10a8f812c434dd3cb816a0e13fb3715df034a8941006fa7fe26ea0d7a48052.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/cc10a8f812c434dd3cb816a0e13fb3715df034a8941006fa7fe26ea0d7a48052.jpg)

![d327f129683a918b344f5032d8580717794fc16491970c68a1795875179a88ff.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/d327f129683a918b344f5032d8580717794fc16491970c68a1795875179a88ff.jpg)

![d664cbea0f0bbb0c19f05598c1d564a827b59ef28efa808991ac567d7f895336.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/d664cbea0f0bbb0c19f05598c1d564a827b59ef28efa808991ac567d7f895336.jpg)

![dd0898c7136a45b53e147610668e4d6a3db2fb1d72a92787bd0078c427edbb1b.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/images/dd0898c7136a45b53e147610668e4d6a3db2fb1d72a92787bd0078c427edbb1b.jpg)

### Tables

![0e446b3025ff8dbc00f001ef65d2ef52ed42cb9cb42aeabc8bceb02efe6aee3f.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/tables/0e446b3025ff8dbc00f001ef65d2ef52ed42cb9cb42aeabc8bceb02efe6aee3f.jpg)

![3052a802c62f6a734a56ff65da91f25d088e7ad03e16374dcc68a32314826493.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/tables/3052a802c62f6a734a56ff65da91f25d088e7ad03e16374dcc68a32314826493.jpg)

![dc67d172c942b80b5b5a3a44e28a895b28bce6ff356da395b50c03198b704021.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/tables/dc67d172c942b80b5b5a3a44e28a895b28bce6ff356da395b50c03198b704021.jpg)

![f71432c6389363e41a8c078b1613b75dc4bb1d4106df010318ae5cc7d0434a9a.jpg](../icml_results/1450_Training%20LLM%20Agents%20to%20Empower%20Humans/tables/f71432c6389363e41a8c078b1613b75dc4bb1d4106df010318ae5cc7d0434a9a.jpg)

<a id="paper-1431"></a>

## Global Plane Waves from Local Gaussians: Periodic Charge Densities in a Blink


### Images

![080015d4f578a0dad674474fd884bb1413e6f294be2a882e095d27c4d47b4aeb.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/080015d4f578a0dad674474fd884bb1413e6f294be2a882e095d27c4d47b4aeb.jpg)

![08d6ae81602c66c0c79f10f17cbdd08ae8997c4fb3d10a0ae1d7255cf44ec4b6.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/08d6ae81602c66c0c79f10f17cbdd08ae8997c4fb3d10a0ae1d7255cf44ec4b6.jpg)

![092a2a01e669187fa938a88f7abc3609c0524f0d0ceb5ae6daa2c7cbf2db05bf.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/092a2a01e669187fa938a88f7abc3609c0524f0d0ceb5ae6daa2c7cbf2db05bf.jpg)

![13d63407c0ae2b28d08205e83b894dfd7afc0112bec5f8fb75eb9d9877754809.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/13d63407c0ae2b28d08205e83b894dfd7afc0112bec5f8fb75eb9d9877754809.jpg)

![1a41f34db801bfd5723c3915fc5538e4df690620b2307c3a5b20b6ea6e5fb543.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/1a41f34db801bfd5723c3915fc5538e4df690620b2307c3a5b20b6ea6e5fb543.jpg)

![1cdaa0ca9ac9ed41da8f78fed367350e9fdd1af3362a6f1d38c5d1e348c16d5e.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/1cdaa0ca9ac9ed41da8f78fed367350e9fdd1af3362a6f1d38c5d1e348c16d5e.jpg)

![299774ce37b7b121afd486610f4aeb87621a900d44b0b7d8cfe064de3b970b43.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/299774ce37b7b121afd486610f4aeb87621a900d44b0b7d8cfe064de3b970b43.jpg)

![4f2fcec856d34ad4256a8b513cc6cfe3cbd8275e307c129695322e7f527f51cc.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/4f2fcec856d34ad4256a8b513cc6cfe3cbd8275e307c129695322e7f527f51cc.jpg)

![7d103c4c0e77bd8f969b7705be99f23a4b195bac8b0bf8e82663979f96373b42.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/7d103c4c0e77bd8f969b7705be99f23a4b195bac8b0bf8e82663979f96373b42.jpg)

![7fee6651b71309e46c7b0c5345fc394cf436e25c97f062383383ec6d693a6be5.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/7fee6651b71309e46c7b0c5345fc394cf436e25c97f062383383ec6d693a6be5.jpg)

![90c2de44c3709aaee3f227b6a375bc9696509eea8f0f60ddb81e4bc28b15468f.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/90c2de44c3709aaee3f227b6a375bc9696509eea8f0f60ddb81e4bc28b15468f.jpg)

![c8313d69955c9b35e28b6c00d54f8c8ece0f3bef9727aa6dc33d27bd66a4121d.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/c8313d69955c9b35e28b6c00d54f8c8ece0f3bef9727aa6dc33d27bd66a4121d.jpg)

![db116111d1fb5548bc2c776b830265cebb1d69381a32353138e0591309c1c488.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/db116111d1fb5548bc2c776b830265cebb1d69381a32353138e0591309c1c488.jpg)

![ea7a7bbfc714a51af4b41e0ab15dfcbb84b718b2e401d31d83349b5c5f6605f4.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/ea7a7bbfc714a51af4b41e0ab15dfcbb84b718b2e401d31d83349b5c5f6605f4.jpg)

![f4be97f43a9556baef58dcf0655ada127315e4c43ee87cd455f546e81adb0d90.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/f4be97f43a9556baef58dcf0655ada127315e4c43ee87cd455f546e81adb0d90.jpg)

![fe3c262bd5f36fcc15b70738eecf33969c3230444f750f02dd495f5dd5df95de.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/images/fe3c262bd5f36fcc15b70738eecf33969c3230444f750f02dd495f5dd5df95de.jpg)

### Tables

![1fbb6d17ca430e0176d2bc64db4767d2aa46103e7da52fa86169409206c7e2bd.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/1fbb6d17ca430e0176d2bc64db4767d2aa46103e7da52fa86169409206c7e2bd.jpg)

![3b9feb689bf33501ad4f85b2ddb5b4b551301bb27ba533c80f3af0dc9edd37ce.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/3b9feb689bf33501ad4f85b2ddb5b4b551301bb27ba533c80f3af0dc9edd37ce.jpg)

![5143217a9b5223abdf62b9ab1f1f2bb4ca11201586874231f1610a6b2d4087f2.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/5143217a9b5223abdf62b9ab1f1f2bb4ca11201586874231f1610a6b2d4087f2.jpg)

![68df1076d8bf302ee61e57ae90b0e071ba340c48bada30dbb61f577336e4201a.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/68df1076d8bf302ee61e57ae90b0e071ba340c48bada30dbb61f577336e4201a.jpg)

![b43ba1d70c4ff1c17211c4ddd70210fa14bc125bcb363fe53c26bcf9e685649a.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/b43ba1d70c4ff1c17211c4ddd70210fa14bc125bcb363fe53c26bcf9e685649a.jpg)

![c0f7509ae1cddbb2a3598732ccc09e286cb556fdd7e0d0e96d8a66e1b3c68958.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/c0f7509ae1cddbb2a3598732ccc09e286cb556fdd7e0d0e96d8a66e1b3c68958.jpg)

![c895e86a3b37586398e857d6f99e104f59013fc9079bac297d59a1b31e938b58.jpg](../icml_results/1451_Global%20Plane%20Waves%20from%20Local%20Gaussians_%20Periodic%20Charge%20Densities%20in%20a%20Blink/tables/c895e86a3b37586398e857d6f99e104f59013fc9079bac297d59a1b31e938b58.jpg)

<a id="paper-1432"></a>

## Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training


### Images

![10ee88376815cf546eeb07d848a2a255422b09189b3f551ef93c082324033363.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/images/10ee88376815cf546eeb07d848a2a255422b09189b3f551ef93c082324033363.jpg)

![511c4ae80858184351c40a026fc5a0e1e1e320f5d9b48a15a7e5e1d7c182309b.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/images/511c4ae80858184351c40a026fc5a0e1e1e320f5d9b48a15a7e5e1d7c182309b.jpg)

![68b069ed4ebeeb527077f5e38da039f835995fc51bd702f0bfaaf0943019e6eb.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/images/68b069ed4ebeeb527077f5e38da039f835995fc51bd702f0bfaaf0943019e6eb.jpg)

### Tables

![04848fffa2f356c8c31fda5275c724effbc414ae92291ed51d952337ff7f06b9.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/04848fffa2f356c8c31fda5275c724effbc414ae92291ed51d952337ff7f06b9.jpg)

![2f3c66491bf4bcee29676221df045086ee881cf05600477d119a39351a9b1d98.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/2f3c66491bf4bcee29676221df045086ee881cf05600477d119a39351a9b1d98.jpg)

![3757901231905ad4039e5935c57aa9ba72feb33c7d2f278983cd26e82bcd6d40.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/3757901231905ad4039e5935c57aa9ba72feb33c7d2f278983cd26e82bcd6d40.jpg)

![41a32f8003ad94421641709c5b3a2ea2a689991ad5fc9faf7610ff94e39d9d9a.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/41a32f8003ad94421641709c5b3a2ea2a689991ad5fc9faf7610ff94e39d9d9a.jpg)

![4e204ddb82e6c11567a8ad69464bdabd64705477668a0dd047ef7f699838ac7b.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/4e204ddb82e6c11567a8ad69464bdabd64705477668a0dd047ef7f699838ac7b.jpg)

![57451ae9d069784efea96035e29704dad7d0fff54e911c594da4535c218bf126.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/57451ae9d069784efea96035e29704dad7d0fff54e911c594da4535c218bf126.jpg)

![68be8b445db0584bde52468ee0a7531368645faae31ab7cf2243570bee42e915.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/68be8b445db0584bde52468ee0a7531368645faae31ab7cf2243570bee42e915.jpg)

![799c7ad5f5fc19045110c8660b598cc9c5dc3f8a199f7faa4be94b10ba7f6768.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/799c7ad5f5fc19045110c8660b598cc9c5dc3f8a199f7faa4be94b10ba7f6768.jpg)

![7fac55a4fcc7f2646a1cd92e412f744317de5fc5258bc23ce96e436991efb6c1.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/7fac55a4fcc7f2646a1cd92e412f744317de5fc5258bc23ce96e436991efb6c1.jpg)

![a143b86da8295bf365bd09d5f6aeb32c64325284a3c43edd7cc6aca044b60017.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/a143b86da8295bf365bd09d5f6aeb32c64325284a3c43edd7cc6aca044b60017.jpg)

![bfb5b2cd3a70d972262708b1807b0df711af937cd2230b79bf34cee71b6af056.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/bfb5b2cd3a70d972262708b1807b0df711af937cd2230b79bf34cee71b6af056.jpg)

![bfcd7fe3c270b8da866ccd71a7cf997b8a7f95663605018929137e7c7df2947a.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/bfcd7fe3c270b8da866ccd71a7cf997b8a7f95663605018929137e7c7df2947a.jpg)

![e1c5712ecd0bbffe5932e50cce981769a57300a95706eef821f02e20e8707286.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/e1c5712ecd0bbffe5932e50cce981769a57300a95706eef821f02e20e8707286.jpg)

![e872349d0bbc5c23f97d0c8b2ce548a8de6a6193fce61e31e498af795182e62e.jpg](../icml_results/1452_Rank-Aware%20Spectral%20Bounds%20on%20Attention%20Logits%20for%20Stable%20Low-Precision%20Training/tables/e872349d0bbc5c23f97d0c8b2ce548a8de6a6193fce61e31e498af795182e62e.jpg)

<a id="paper-1433"></a>

## GraphPFN: A Prior-Data Fitted Graph Foundation Model


### Images

![19dfeb713224dc5a02cb6b7f6a3e24a62c881af63975dfb9c3c62d6225cfcf55.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/images/19dfeb713224dc5a02cb6b7f6a3e24a62c881af63975dfb9c3c62d6225cfcf55.jpg)

![8055fb101cb1efe99e292cfe77029538e9d2373d551fbf63e3123aeeca2971a6.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/images/8055fb101cb1efe99e292cfe77029538e9d2373d551fbf63e3123aeeca2971a6.jpg)

![8f4ba8d73a71680c8eab49201266c5dcac82bbe63751f6056c01e5b88240c02d.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/images/8f4ba8d73a71680c8eab49201266c5dcac82bbe63751f6056c01e5b88240c02d.jpg)

### Tables

![0b367f01c7adbc748aa59b9ab202ffd3efd59ae0a2b29a9e454a5ff8a3ab5e3f.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/0b367f01c7adbc748aa59b9ab202ffd3efd59ae0a2b29a9e454a5ff8a3ab5e3f.jpg)

![4cc84f08de865a2f3f2db10410f54723e3b089b8af467554eabf2d2e7ab1cc21.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/4cc84f08de865a2f3f2db10410f54723e3b089b8af467554eabf2d2e7ab1cc21.jpg)

![56f20874d163cf6d033b029ac396504466857429f166ea77ea9c2f83e32e23ce.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/56f20874d163cf6d033b029ac396504466857429f166ea77ea9c2f83e32e23ce.jpg)

![6eda81d2178b5f20a874f57e4589464073f3189c6d3adfc6471bc7eb40639484.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/6eda81d2178b5f20a874f57e4589464073f3189c6d3adfc6471bc7eb40639484.jpg)

![782e7410e7b0596bd4c182604c4ce021083b5512b3bb4087c1fdc0cc41b418fa.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/782e7410e7b0596bd4c182604c4ce021083b5512b3bb4087c1fdc0cc41b418fa.jpg)

![80e5184ba4062343da5113aead9e7bfb6e5082c651e0d67435d00bb61fb27b11.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/80e5184ba4062343da5113aead9e7bfb6e5082c651e0d67435d00bb61fb27b11.jpg)

![82335e36410e2cd7023688e4921cb675d03e004bd3547b8fc1b18a9a1fbec918.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/82335e36410e2cd7023688e4921cb675d03e004bd3547b8fc1b18a9a1fbec918.jpg)

![844d323df20cd663d930420ef9875c56685a465b165e330c7d1f4a7d4e018112.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/844d323df20cd663d930420ef9875c56685a465b165e330c7d1f4a7d4e018112.jpg)

![8cd757e0d7949397839df066d5f52fccba5e6feb1f9568553b79faeadbd739ea.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/8cd757e0d7949397839df066d5f52fccba5e6feb1f9568553b79faeadbd739ea.jpg)

![a60a5ae036795d2e2baf5d7dd9af45b5303eba49d8498b76cf5b058a8a4c888a.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/a60a5ae036795d2e2baf5d7dd9af45b5303eba49d8498b76cf5b058a8a4c888a.jpg)

![de392f2bc6a75d89fca4bd0d6b054a0fbcd885b683ca4b3f14dde4d2bfdb5c15.jpg](../icml_results/1454_GraphPFN_%20A%20Prior-Data%20Fitted%20Graph%20Foundation%20Model/tables/de392f2bc6a75d89fca4bd0d6b054a0fbcd885b683ca4b3f14dde4d2bfdb5c15.jpg)

<a id="paper-1434"></a>

## Clipping Makes Distributed and Federated Asynchronous SGD Robust to Stragglers


### Images

![1b5e621813a5290251a31fbacab7c9bda63e5d6c916bd6d6f0681220ccc342eb.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/images/1b5e621813a5290251a31fbacab7c9bda63e5d6c916bd6d6f0681220ccc342eb.jpg)

![22fcb7057f99b76779f39968f28ffb0d3f1328d9f776504ede08a10396a9e85c.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/images/22fcb7057f99b76779f39968f28ffb0d3f1328d9f776504ede08a10396a9e85c.jpg)

![e7a19fe37c96283338af22073f84882a02687a0a44a264aa5dc881ac11adfcf1.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/images/e7a19fe37c96283338af22073f84882a02687a0a44a264aa5dc881ac11adfcf1.jpg)

![efd7d00338cc8e6f0c83dc62e9ea9d8372cefb77e7fd101b48dafc74d25f52b1.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/images/efd7d00338cc8e6f0c83dc62e9ea9d8372cefb77e7fd101b48dafc74d25f52b1.jpg)

### Tables

![cb236038f8d4d406fdf50bb500e5ff688b6e5d64629f2473b4f4e6a6a78aeced.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/tables/cb236038f8d4d406fdf50bb500e5ff688b6e5d64629f2473b4f4e6a6a78aeced.jpg)

![f0bbba91bd610d3576558c9fd32885f4541d4694c9d7b0cf4b2c817517a7e572.jpg](../icml_results/1455_Clipping%20Makes%20Distributed%20and%20Federated%20Asynchronous%20SGD%20Robust%20to%20Stragglers/tables/f0bbba91bd610d3576558c9fd32885f4541d4694c9d7b0cf4b2c817517a7e572.jpg)

<a id="paper-1435"></a>

## When Softmax Fails at the Top: Extreme-Value Corrections for InfoNCE


### Images

![0a67aecae5ad5c1c6bd7a93834ce7a14a1be1c2113efe8d4ee25fdcc9f83dc08.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/0a67aecae5ad5c1c6bd7a93834ce7a14a1be1c2113efe8d4ee25fdcc9f83dc08.jpg)

![392251018a426afa67b897914ad22a27b486f901a01983a6b2ed2a8f1545113c.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/392251018a426afa67b897914ad22a27b486f901a01983a6b2ed2a8f1545113c.jpg)

![715efbf5169620dd356008330972e2f1ff8607ee4aa457c2daf1e1a85f5185ec.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/715efbf5169620dd356008330972e2f1ff8607ee4aa457c2daf1e1a85f5185ec.jpg)

![71ba1cba6b5c4c8baa6e66e5de29439728a9711b3c9c3c5aa06a50b2b80a9ed8.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/71ba1cba6b5c4c8baa6e66e5de29439728a9711b3c9c3c5aa06a50b2b80a9ed8.jpg)

![866b3b6fa73935293f1a5953e2f0e0c947ce9cdbaee63e574a8873a7cf7d7f73.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/866b3b6fa73935293f1a5953e2f0e0c947ce9cdbaee63e574a8873a7cf7d7f73.jpg)

![936f633a7d1bda06548fc660ee7903e0b20b8f1f1b3e425e5a5be3eaeefaf989.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/936f633a7d1bda06548fc660ee7903e0b20b8f1f1b3e425e5a5be3eaeefaf989.jpg)

![988883062d375b1b53badb6c7a3a61c7cc27d21033c6e548a8c0b33b11bb5115.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/988883062d375b1b53badb6c7a3a61c7cc27d21033c6e548a8c0b33b11bb5115.jpg)

![9d03df2c3fc1271f7d3b62231a6dfc80ffde887ef26aa9b10eb2d10ede5a425a.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/9d03df2c3fc1271f7d3b62231a6dfc80ffde887ef26aa9b10eb2d10ede5a425a.jpg)

![b8633d80fef7ce2f11b5b0fa0343fa0646ac9a886564a0c3873ceb25e289321d.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/b8633d80fef7ce2f11b5b0fa0343fa0646ac9a886564a0c3873ceb25e289321d.jpg)

![c45c6c2c33d91612a3977db8b5bf79f823067dc2fc095ed67964fc3c2c7178e6.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/c45c6c2c33d91612a3977db8b5bf79f823067dc2fc095ed67964fc3c2c7178e6.jpg)

![f3dbe14e492164e1f2ef90e3191ab3502ce1d354842e050386f9e8b588369d67.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/images/f3dbe14e492164e1f2ef90e3191ab3502ce1d354842e050386f9e8b588369d67.jpg)

### Tables

![356fe400ae7b1974f9693e8407b56eedfe2be2c22683e9450dcdeb08397e6040.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/356fe400ae7b1974f9693e8407b56eedfe2be2c22683e9450dcdeb08397e6040.jpg)

![789d304466e02939a5451aecae4965e1a3fd0d233a1f515851902ffb389d286e.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/789d304466e02939a5451aecae4965e1a3fd0d233a1f515851902ffb389d286e.jpg)

![8f90fe2f7a2121b6c4fe07811d36c293c5467d5a2a76285ea7cd8a59af75610b.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/8f90fe2f7a2121b6c4fe07811d36c293c5467d5a2a76285ea7cd8a59af75610b.jpg)

![9bf6f73af8bbda028bf981f1abdc9596fe2cca3dc433239e9e41317552c9d41f.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/9bf6f73af8bbda028bf981f1abdc9596fe2cca3dc433239e9e41317552c9d41f.jpg)

![a335b20f4237c0a257f990dd4c953f1b72f5e485d3cda29f9b05c94b28068f59.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/a335b20f4237c0a257f990dd4c953f1b72f5e485d3cda29f9b05c94b28068f59.jpg)

![b12eb925b3b3d9147c726c2f101ca6e7aa8c7bf9798fe5429a9edab69963fb68.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/b12eb925b3b3d9147c726c2f101ca6e7aa8c7bf9798fe5429a9edab69963fb68.jpg)

![c3f7f04597275587250efaa29f37c268fa36923d1963069d28efdb77925521e9.jpg](../icml_results/1456_When%20Softmax%20Fails%20at%20the%20Top_%20Extreme-Value%20Corrections%20for%20InfoNCE/tables/c3f7f04597275587250efaa29f37c268fa36923d1963069d28efdb77925521e9.jpg)

<a id="paper-1436"></a>

## When Can You Poison Rewards? A Tight Characterization of Reward Poisoning in Linear MDPs


### Images

![01b0a1f0ce84c9b567fd917c28dcd7ac1ee89d43ce7af36fbcfd4c836b1e5b9f.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/images/01b0a1f0ce84c9b567fd917c28dcd7ac1ee89d43ce7af36fbcfd4c836b1e5b9f.jpg)

![3e60975a2c2c07551a785a0c183be6a0df428033c36d0160ddc38ec20fba710c.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/images/3e60975a2c2c07551a785a0c183be6a0df428033c36d0160ddc38ec20fba710c.jpg)

![866f3a7080ec7e267895438cc781ca1d21fdd9c1fe431f99ed95cc80bb62dd90.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/images/866f3a7080ec7e267895438cc781ca1d21fdd9c1fe431f99ed95cc80bb62dd90.jpg)

![cf7d151fc227be1977ba1e9d355faf5c67ba5efe7b7569691415e83a204fe4d4.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/images/cf7d151fc227be1977ba1e9d355faf5c67ba5efe7b7569691415e83a204fe4d4.jpg)

![ed4fcb94043c4fcd3be8ba6aec2ada2b35d1dbd5b559a703921ed2aec74435bd.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/images/ed4fcb94043c4fcd3be8ba6aec2ada2b35d1dbd5b559a703921ed2aec74435bd.jpg)

### Tables

![017cbc0db8b08020c780161e809545cf372e3196ee0baa344eac2c1507d41415.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/017cbc0db8b08020c780161e809545cf372e3196ee0baa344eac2c1507d41415.jpg)

![0c8ddf5b20b722247000d96ec0f51da2a26f45fa00211c8b4510b399f86fa5e4.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/0c8ddf5b20b722247000d96ec0f51da2a26f45fa00211c8b4510b399f86fa5e4.jpg)

![9c8164ad01a1a353f195d93d7acd923dee1c19f0f9a9165ffb37d03cbd1a0d78.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/9c8164ad01a1a353f195d93d7acd923dee1c19f0f9a9165ffb37d03cbd1a0d78.jpg)

![a19e609128bbed78f197d49beb5b1d78888c3abae6f057778ee12be499ab202e.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/a19e609128bbed78f197d49beb5b1d78888c3abae6f057778ee12be499ab202e.jpg)

![bbb47c2af9942a26701e61229ac5c09a651b7364b8bccb90d12d83d9157eca3a.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/bbb47c2af9942a26701e61229ac5c09a651b7364b8bccb90d12d83d9157eca3a.jpg)

![cdd7880a8f3aace2290302fe780450777ab5953e0046c5f7c904b84c531c8698.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/cdd7880a8f3aace2290302fe780450777ab5953e0046c5f7c904b84c531c8698.jpg)

![d09378fc1d85ab8e2fcdebb6a9e84601c478c5097c5d34a1904319e73fd02dea.jpg](../icml_results/1457_When%20Can%20You%20Poison%20Rewards_%20A%20Tight%20Characterization%20of%20Reward%20Poisoning%20in%20Linear%20MDPs/tables/d09378fc1d85ab8e2fcdebb6a9e84601c478c5097c5d34a1904319e73fd02dea.jpg)

<a id="paper-1437"></a>

## Spectrally-Guided Diffusion Noise Schedules


### Images

![296ea4655f52df35586a95f9b10d47b3a43ec921ae73461cd456ac2ebb02f52b.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/296ea4655f52df35586a95f9b10d47b3a43ec921ae73461cd456ac2ebb02f52b.jpg)

![53edfd2c5d2f48d2237a5cd704d451b838425d544a1cb339a2bd3299736bba2d.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/53edfd2c5d2f48d2237a5cd704d451b838425d544a1cb339a2bd3299736bba2d.jpg)

![79a9b8b910437182612f3a0cdcc5b9e99a61668a4bac86bcc1571be346109f0f.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/79a9b8b910437182612f3a0cdcc5b9e99a61668a4bac86bcc1571be346109f0f.jpg)

![8fc5aaad8b2de833c238c80f4447f5802b4070649d86da504bee06e2a48dcae7.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/8fc5aaad8b2de833c238c80f4447f5802b4070649d86da504bee06e2a48dcae7.jpg)

![96eaaa5991d55dfb25564ca59a6ae97606f6465f33c3d85cc71e7098a966c13a.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/96eaaa5991d55dfb25564ca59a6ae97606f6465f33c3d85cc71e7098a966c13a.jpg)

![afac270f217ed6588221f0a15d37b7335f407bf9c02c5af128dac56e712357d5.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/afac270f217ed6588221f0a15d37b7335f407bf9c02c5af128dac56e712357d5.jpg)

![c1563782a15905a738a101125bf91b5e377024f86ae1ec7727a7543904f82541.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/c1563782a15905a738a101125bf91b5e377024f86ae1ec7727a7543904f82541.jpg)

![d0f6d95b8f2d5080c8de3cee4e7dfce1e562eaa3b66ddbcb3bea8e4660a79a8e.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/images/d0f6d95b8f2d5080c8de3cee4e7dfce1e562eaa3b66ddbcb3bea8e4660a79a8e.jpg)

### Tables

![0bdf8dc48b3326b2c0fc56d246ed8421d4e7c8d67a904e8281a0b40383bfafb7.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/0bdf8dc48b3326b2c0fc56d246ed8421d4e7c8d67a904e8281a0b40383bfafb7.jpg)

![1f371b7735bff8c6ca6efb112d6ebb8684b9f8f8a0568527b057507d1e227a53.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/1f371b7735bff8c6ca6efb112d6ebb8684b9f8f8a0568527b057507d1e227a53.jpg)

![1f38890b6e655d3e9411c7e205da7853c23d5bfd36795e921992cf0cb5f11a29.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/1f38890b6e655d3e9411c7e205da7853c23d5bfd36795e921992cf0cb5f11a29.jpg)

![20c3e31e9ddbd611a272e58c39e60e5322c856c77b1c9b9407ff488ee624a09d.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/20c3e31e9ddbd611a272e58c39e60e5322c856c77b1c9b9407ff488ee624a09d.jpg)

![400feae5a0296ff267e755ff0d3e43ab9c7d760b51875f89a6b3d28d8232b10c.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/400feae5a0296ff267e755ff0d3e43ab9c7d760b51875f89a6b3d28d8232b10c.jpg)

![5849b5fdba4a909bb00d7c73bf3df9bed27605f40106b68abfe2490195ece576.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/5849b5fdba4a909bb00d7c73bf3df9bed27605f40106b68abfe2490195ece576.jpg)

![836aa914ce78daf49aaec2f9ea46703b929f44852cde2390691f465d3f5b7099.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/836aa914ce78daf49aaec2f9ea46703b929f44852cde2390691f465d3f5b7099.jpg)

![87bf7a8bf449d2b0f7b32b856bf1a4e03e775c4da3e8a2ad6770ab72d509498d.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/87bf7a8bf449d2b0f7b32b856bf1a4e03e775c4da3e8a2ad6770ab72d509498d.jpg)

![978910bb05b6a2e39a43d62cd8e55fa677c5a7f94ca46076c0787df7fea2120c.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/978910bb05b6a2e39a43d62cd8e55fa677c5a7f94ca46076c0787df7fea2120c.jpg)

![c1d812b74eaedfb57666bbf0c870f9a21aeaeed3cda3978e9eabcf1fd135ef3c.jpg](../icml_results/1458_Spectrally-Guided%20Diffusion%20Noise%20Schedules/tables/c1d812b74eaedfb57666bbf0c870f9a21aeaeed3cda3978e9eabcf1fd135ef3c.jpg)

<a id="paper-1438"></a>

## Automatic Construction of Clinical Scoring Systems with LLM Agents


### Images

![17ddbcbf0b5160c2579406891c8d627f54ab114be67648d892c4aeacbe201c54.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/17ddbcbf0b5160c2579406891c8d627f54ab114be67648d892c4aeacbe201c54.jpg)

![590a37cc104d15d4d25fa7dc7acb038ddd138ecf167fa72576992e3dc08dd8f3.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/590a37cc104d15d4d25fa7dc7acb038ddd138ecf167fa72576992e3dc08dd8f3.jpg)

![63658d1b4a9521bd4aa15d67f5ce721d2c1244b6a790fd8d1b4e1c2a2ade773e.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/63658d1b4a9521bd4aa15d67f5ce721d2c1244b6a790fd8d1b4e1c2a2ade773e.jpg)

![6a5f4a49810475627e0b2453f3734a680a3b93cc1598c2345169f4c1ca9887cc.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/6a5f4a49810475627e0b2453f3734a680a3b93cc1598c2345169f4c1ca9887cc.jpg)

![7e50e8fa9ef5bcfe5b07ea582b1b7093265124375259e80607371e6114fcd82b.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/7e50e8fa9ef5bcfe5b07ea582b1b7093265124375259e80607371e6114fcd82b.jpg)

![94ad75c6adefc182c649c83136b328a74521c8a47bb2823dfca876ad03c88a48.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/94ad75c6adefc182c649c83136b328a74521c8a47bb2823dfca876ad03c88a48.jpg)

![9c68cf0e8df26b5b64b3c73245d5d2a5140583b10f312550f7e24279d52be69d.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/9c68cf0e8df26b5b64b3c73245d5d2a5140583b10f312550f7e24279d52be69d.jpg)

![b466dd6f174402a69f6ab59079208a186c743eb131c65efa91574f475e6eb34e.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/images/b466dd6f174402a69f6ab59079208a186c743eb131c65efa91574f475e6eb34e.jpg)

### Tables

![104d1f163b0712e0ff1c86c36d5d2adac0b4dcbbbc5407de2c927ab3f6a53885.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/104d1f163b0712e0ff1c86c36d5d2adac0b4dcbbbc5407de2c927ab3f6a53885.jpg)

![1808b23f874040ee393e9cf58d0ebb5aa62b7e3793a6dbad9249e44f72f76cd8.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/1808b23f874040ee393e9cf58d0ebb5aa62b7e3793a6dbad9249e44f72f76cd8.jpg)

![26eafb67fa8040db7f09436f3907ab240c27325c3d93fce24d156348d8d9b9a6.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/26eafb67fa8040db7f09436f3907ab240c27325c3d93fce24d156348d8d9b9a6.jpg)

![32835d9f8729036b94d40d24c74cd8af1ae5d84502bfd1d02494d096816de935.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/32835d9f8729036b94d40d24c74cd8af1ae5d84502bfd1d02494d096816de935.jpg)

![33388be60a79399aa94b7552570c02689e65068acbb5a6d75ba2ce1bb1a94c19.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/33388be60a79399aa94b7552570c02689e65068acbb5a6d75ba2ce1bb1a94c19.jpg)

![3484747425420ae5265d53913c648be3761b4786d9d77675c44877e33fd3ef5c.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/3484747425420ae5265d53913c648be3761b4786d9d77675c44877e33fd3ef5c.jpg)

![57838b66a8c0eeb0ab0e4151e72d98751cd4335d635901650d349919a110827c.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/57838b66a8c0eeb0ab0e4151e72d98751cd4335d635901650d349919a110827c.jpg)

![5c8e1a3eb89523e5684011379e647637c09a7624a76be4b5110bb99d399856f7.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/5c8e1a3eb89523e5684011379e647637c09a7624a76be4b5110bb99d399856f7.jpg)

![600276b4e77c307978e3e65a6b403c5e1bc7c32db71fa27fb463d817f54db147.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/600276b4e77c307978e3e65a6b403c5e1bc7c32db71fa27fb463d817f54db147.jpg)

![682261e7aae40305900fe0c5230a998bcb1b91c21e5bbaac6936dfc89cfb0a2a.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/682261e7aae40305900fe0c5230a998bcb1b91c21e5bbaac6936dfc89cfb0a2a.jpg)

![735f2dc43f1f5d66da89f6123833f3ad478e90f775b30529302a694149add79e.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/735f2dc43f1f5d66da89f6123833f3ad478e90f775b30529302a694149add79e.jpg)

![79b23920c4292b95338b33bb38c75669a8b9fbae1649446e04018e69f2cac732.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/79b23920c4292b95338b33bb38c75669a8b9fbae1649446e04018e69f2cac732.jpg)

![7f9bc3ac32feb8d4a002d877f3e47d1b2e4ae356c7b16a9301c828af5bf537b1.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/7f9bc3ac32feb8d4a002d877f3e47d1b2e4ae356c7b16a9301c828af5bf537b1.jpg)

![8a7c73ec7b66f35d682e1282edb741ef0bf74d0abfd5ab1796c1efa26ffe520f.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/8a7c73ec7b66f35d682e1282edb741ef0bf74d0abfd5ab1796c1efa26ffe520f.jpg)

![a95edf5eed6fb709a1e10c54d225276ec8497cc3ef449f06400bd0d593253277.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/a95edf5eed6fb709a1e10c54d225276ec8497cc3ef449f06400bd0d593253277.jpg)

![b5d019febcc3752ba102ab32b9c38e33d37a73d35057aabfe1773487b89efdff.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/b5d019febcc3752ba102ab32b9c38e33d37a73d35057aabfe1773487b89efdff.jpg)

![b8944f7181e1cd198b96389506c2dd1faed61d1bd98f236a376f50a1a449625b.jpg](../icml_results/1459_Automatic%20Construction%20of%20Clinical%20Scoring%20Systems%20with%20LLM%20Agents/tables/b8944f7181e1cd198b96389506c2dd1faed61d1bd98f236a376f50a1a449625b.jpg)

<a id="paper-1439"></a>

## CellBRIDGE: Learning Cellular Trajectories via Interaction-Aware Alignment


### Images

![12cbf649d6172fb35f3c9663a44fe091dc039d07e44ab37fc7c37bce04231b33.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/12cbf649d6172fb35f3c9663a44fe091dc039d07e44ab37fc7c37bce04231b33.jpg)

![18bd30d5b82f8f7d09f2efe14b28198de7123553514a0aefa63713992e4e7aed.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/18bd30d5b82f8f7d09f2efe14b28198de7123553514a0aefa63713992e4e7aed.jpg)

![197640e8d391b425dabc4b4d65c679bf20adde4b0f7294ba2793997069442da2.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/197640e8d391b425dabc4b4d65c679bf20adde4b0f7294ba2793997069442da2.jpg)

![25806eab75c8a764dc4352cbe38557e73d27a93724255748bd9e1c0711e39b5e.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/25806eab75c8a764dc4352cbe38557e73d27a93724255748bd9e1c0711e39b5e.jpg)

![45f2e4528db1e3c109abee13675fbbbd8f27a086721e0a64cec6eaea76ae9513.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/45f2e4528db1e3c109abee13675fbbbd8f27a086721e0a64cec6eaea76ae9513.jpg)

![526545ed31051b070da84f76f0570f94973888e0ecd5676f70fba46edd4f2734.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/526545ed31051b070da84f76f0570f94973888e0ecd5676f70fba46edd4f2734.jpg)

![661b0bfcd9fedb5329cb9ccdf4e76036a7569c6cdd49c0d6f71626d7cb001f98.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/661b0bfcd9fedb5329cb9ccdf4e76036a7569c6cdd49c0d6f71626d7cb001f98.jpg)

![69e1ab694dabed9d3df83f1a160b9cdc7bb6d83e1cf645756dd98a547c1dc854.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/69e1ab694dabed9d3df83f1a160b9cdc7bb6d83e1cf645756dd98a547c1dc854.jpg)

![74df9e89f3492582b9254fa5e8da286904bb7d8c2dabdc0ac7ec3095858502e8.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/74df9e89f3492582b9254fa5e8da286904bb7d8c2dabdc0ac7ec3095858502e8.jpg)

![9f84cd0e7c47b6ab62a7983c013a6dc06519266883b5ced2f0c39746968c1113.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/9f84cd0e7c47b6ab62a7983c013a6dc06519266883b5ced2f0c39746968c1113.jpg)

![c89ccdf4c6b6a1fd492676c9f3219c5289b3271ed21cbb9a3f08a533f9eb2a59.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/c89ccdf4c6b6a1fd492676c9f3219c5289b3271ed21cbb9a3f08a533f9eb2a59.jpg)

![c9abe258bc3419d7739222cf6118acd5e8bbaa0f9eff5ff10528ca3f21463df2.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/c9abe258bc3419d7739222cf6118acd5e8bbaa0f9eff5ff10528ca3f21463df2.jpg)

![f844950c22a285cc9713ad27d4a6f503e4d92bea3fb3ed9e9332813754b3f71d.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/f844950c22a285cc9713ad27d4a6f503e4d92bea3fb3ed9e9332813754b3f71d.jpg)

![f9f78f5bf5c49d3250c39e81453188303e9f08fa39c53e359c418f5877fd0efa.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/images/f9f78f5bf5c49d3250c39e81453188303e9f08fa39c53e359c418f5877fd0efa.jpg)

### Tables

![0cee3c9d62a8b6fd8a482eafaf1501f9e4a064f54a46d060f5e276e15ee15dcd.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/0cee3c9d62a8b6fd8a482eafaf1501f9e4a064f54a46d060f5e276e15ee15dcd.jpg)

![13ffd20b98706b41e74693e9e07c83bc876b3f5f59e807f752385b444dea2aa4.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/13ffd20b98706b41e74693e9e07c83bc876b3f5f59e807f752385b444dea2aa4.jpg)

![241a2a181b1d2ef74403561764b67d47ca9dcdd1efb3b1c004bf20c6b4a0b837.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/241a2a181b1d2ef74403561764b67d47ca9dcdd1efb3b1c004bf20c6b4a0b837.jpg)

![422487dd4fdb196128ab0eee1f9505c025d39b98b60271f6c0e32e2303c78f6b.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/422487dd4fdb196128ab0eee1f9505c025d39b98b60271f6c0e32e2303c78f6b.jpg)

![6cf011675ae26f36629353d8684a13f6e0d0512e1396de884a44e71ca3c41fd7.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/6cf011675ae26f36629353d8684a13f6e0d0512e1396de884a44e71ca3c41fd7.jpg)

![7867af8344b29b916caad9ebcfc241ab292204e80c12036580ef9fbb8e717dba.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/7867af8344b29b916caad9ebcfc241ab292204e80c12036580ef9fbb8e717dba.jpg)

![7bb861718e1adf614bbab57526c1e86df8e4c9b63422f42c58133587023c55b3.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/7bb861718e1adf614bbab57526c1e86df8e4c9b63422f42c58133587023c55b3.jpg)

![8a1e0995ab38f5f0ffc19b9ffead888bf658efda1ed27466d91c997d5312585b.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/8a1e0995ab38f5f0ffc19b9ffead888bf658efda1ed27466d91c997d5312585b.jpg)

![8ecac3590c98a75dbe238f2133ced95ad957c3852d6fe63369c82a73a4f1db50.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/8ecac3590c98a75dbe238f2133ced95ad957c3852d6fe63369c82a73a4f1db50.jpg)

![a237d4c046b4750594c93161839e82baab3cd63e45d014939f67566b724825fb.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/a237d4c046b4750594c93161839e82baab3cd63e45d014939f67566b724825fb.jpg)

![a92d1334be241ba66276b41d31858b4b928389257e44e34f640c79a678315e32.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/a92d1334be241ba66276b41d31858b4b928389257e44e34f640c79a678315e32.jpg)

![ba2140ebd28447d1d53784daa076e22e4aa1f54c929dfd53ff0d439e23fcf485.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/ba2140ebd28447d1d53784daa076e22e4aa1f54c929dfd53ff0d439e23fcf485.jpg)

![c86c528e5c7df94b9d9accf2fb950945b92881f3e8e7505fcc415996ff3ed40b.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/c86c528e5c7df94b9d9accf2fb950945b92881f3e8e7505fcc415996ff3ed40b.jpg)

![dab1f69464c822d5df5adaabfda7f8f42705b3f3ce652b16744d5fadba8ac7d5.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/dab1f69464c822d5df5adaabfda7f8f42705b3f3ce652b16744d5fadba8ac7d5.jpg)

![dbaa9f1f8916434e6e925b9fbca334998fcf46105e536d4b5d4cc58bf4f66171.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/dbaa9f1f8916434e6e925b9fbca334998fcf46105e536d4b5d4cc58bf4f66171.jpg)

![f50050c5cc550a64018e15c28ef20f96e69ba2fe643d9ec554180a99403950b4.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/f50050c5cc550a64018e15c28ef20f96e69ba2fe643d9ec554180a99403950b4.jpg)

![f52cb29d994d3ba83070ed6673e37a028b38da68ed7de811be03269adb4e6b40.jpg](../icml_results/1460_CellBRIDGE_%20Learning%20Cellular%20Trajectories%20via%20Interaction-Aware%20Alignment/tables/f52cb29d994d3ba83070ed6673e37a028b38da68ed7de811be03269adb4e6b40.jpg)

<a id="paper-1440"></a>

## Width Independent Bounds for the Local Lipschitz Constant of Deep Neural Networks at Random Initialization and after Lazy Training


### Images

![55fae36e3a057c2901202a7d27c6d2684f717af8809ace54a86230151aa81e01.jpg](../icml_results/1461_Width%20Independent%20Bounds%20for%20the%20Local%20Lipschitz%20Constant%20of%20Deep%20Neural%20Networks%20at%20Random%20Initiali/images/55fae36e3a057c2901202a7d27c6d2684f717af8809ace54a86230151aa81e01.jpg)

![ee64490a92fc54b0399ac1d42aac4233a3eb8a65fdb7dde0e7735483d9d9916f.jpg](../icml_results/1461_Width%20Independent%20Bounds%20for%20the%20Local%20Lipschitz%20Constant%20of%20Deep%20Neural%20Networks%20at%20Random%20Initiali/images/ee64490a92fc54b0399ac1d42aac4233a3eb8a65fdb7dde0e7735483d9d9916f.jpg)

### Tables

![73bc684f6a6d50afe32cb1293d94fe664db3bab58d903a8c7e9f91d4b62f6b65.jpg](../icml_results/1461_Width%20Independent%20Bounds%20for%20the%20Local%20Lipschitz%20Constant%20of%20Deep%20Neural%20Networks%20at%20Random%20Initiali/tables/73bc684f6a6d50afe32cb1293d94fe664db3bab58d903a8c7e9f91d4b62f6b65.jpg)

![864a9a806ec615e0d5233c2c18e16d1345094c4469d2425b3429146238eb3f00.jpg](../icml_results/1461_Width%20Independent%20Bounds%20for%20the%20Local%20Lipschitz%20Constant%20of%20Deep%20Neural%20Networks%20at%20Random%20Initiali/tables/864a9a806ec615e0d5233c2c18e16d1345094c4469d2425b3429146238eb3f00.jpg)

![cd0f3c73c0ee4dd79eeb96567f044cd1b85c656ec910430f441bda2ab43acb16.jpg](../icml_results/1461_Width%20Independent%20Bounds%20for%20the%20Local%20Lipschitz%20Constant%20of%20Deep%20Neural%20Networks%20at%20Random%20Initiali/tables/cd0f3c73c0ee4dd79eeb96567f044cd1b85c656ec910430f441bda2ab43acb16.jpg)

<a id="paper-1441"></a>

## Entropy-informed Decoding: Adaptive Information-Driven Branching


### Images

![11ed369359d9c209569a1cda1fb5df31a4a7e3eb34c13df3e858deed978b86cb.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/11ed369359d9c209569a1cda1fb5df31a4a7e3eb34c13df3e858deed978b86cb.jpg)

![13c7fc662aca3cce06ed31ec1287a3ba53ef2f6ef91ae9f0c4de4e0a00aa6aa9.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/13c7fc662aca3cce06ed31ec1287a3ba53ef2f6ef91ae9f0c4de4e0a00aa6aa9.jpg)

![27380e4fff06e332b96f7c2705fff524f489f08b8f07a660f6b140892811d9a6.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/27380e4fff06e332b96f7c2705fff524f489f08b8f07a660f6b140892811d9a6.jpg)

![3b669bb27669bb5f6c2c6891f759f8386cd92b004d94b7a6ea3d6cd5105bb7f3.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/3b669bb27669bb5f6c2c6891f759f8386cd92b004d94b7a6ea3d6cd5105bb7f3.jpg)

![73a30cfca546fd75874c0b3f6f50f3173c94626a53eb75035c875a0d32a1aa75.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/73a30cfca546fd75874c0b3f6f50f3173c94626a53eb75035c875a0d32a1aa75.jpg)

![74911f386b9930ac01f8d3540c433a0c44c6ada2eb8642674ca7b502e37ca709.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/74911f386b9930ac01f8d3540c433a0c44c6ada2eb8642674ca7b502e37ca709.jpg)

![b8bd7a13f3b3c4d074d207b08305fca79f2d24edc8cb452a744bacd422b3d801.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/b8bd7a13f3b3c4d074d207b08305fca79f2d24edc8cb452a744bacd422b3d801.jpg)

![d4f3c6e43cd8472be5c29e5a1dc0ad66d9427b444ac03e0ba6f506e0733cb732.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/d4f3c6e43cd8472be5c29e5a1dc0ad66d9427b444ac03e0ba6f506e0733cb732.jpg)

![e3c855b3391a783493d714b0cc797a72bdbb7e47364b2b348e70e953f4c18998.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/e3c855b3391a783493d714b0cc797a72bdbb7e47364b2b348e70e953f4c18998.jpg)

![f85085e04b0381503009fd1e81e35852c0908f89fcb7ea1771202ddd0805441b.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/images/f85085e04b0381503009fd1e81e35852c0908f89fcb7ea1771202ddd0805441b.jpg)

### Tables

![2045899684e9dda7c9b1f27722cf8ab0d7a0088ce1b2946cac9c1c67ee892fd9.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/2045899684e9dda7c9b1f27722cf8ab0d7a0088ce1b2946cac9c1c67ee892fd9.jpg)

![46f69b8889974ff819320f24787f37a4449b9f8899d0f2a6ffade58e9fb13d9f.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/46f69b8889974ff819320f24787f37a4449b9f8899d0f2a6ffade58e9fb13d9f.jpg)

![95b4558874953b4b6ee4ebc5aa06a571ffaf4a0d761434ea9c48edc0b73b9229.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/95b4558874953b4b6ee4ebc5aa06a571ffaf4a0d761434ea9c48edc0b73b9229.jpg)

![a787fc6181daea62207f12755d8a21cc1ecf44a45dfbafa32af170e1fb9d19dd.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/a787fc6181daea62207f12755d8a21cc1ecf44a45dfbafa32af170e1fb9d19dd.jpg)

![c2d88df77c5fee5434a0a00007a950554d3a38256f58f98074bade4e95aa3040.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/c2d88df77c5fee5434a0a00007a950554d3a38256f58f98074bade4e95aa3040.jpg)

![fd499df257f48d59bf554d3bb3567021227c537abcbd0072124df6bffa9fb0cf.jpg](../icml_results/1462_Entropy-informed%20Decoding_%20Adaptive%20Information-Driven%20Branching/tables/fd499df257f48d59bf554d3bb3567021227c537abcbd0072124df6bffa9fb0cf.jpg)

<a id="paper-1442"></a>

## Theory of Minimal Weight Perturbations in Deep Networks and its Applications for Low-Rank Activated Backdoor Attacks


### Images

![197d5a1d6aef51c30a5ec38299b9373b12cad233009ef7b2b3c3511c75e82441.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/197d5a1d6aef51c30a5ec38299b9373b12cad233009ef7b2b3c3511c75e82441.jpg)

![4e4dec2be5a7b39db90d8446cec556da5990e3538de84d306f0d582932854ef0.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/4e4dec2be5a7b39db90d8446cec556da5990e3538de84d306f0d582932854ef0.jpg)

![a71c88773301853e48f210a15461f0565e3f67e957d73d5a9ac2f69d28cad635.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/a71c88773301853e48f210a15461f0565e3f67e957d73d5a9ac2f69d28cad635.jpg)

![bc86593d84083a9982c94162fc0fab7df9db16bf269777295c4b2fcacda10edb.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/bc86593d84083a9982c94162fc0fab7df9db16bf269777295c4b2fcacda10edb.jpg)

![cfa351916c36e4e9dd98c9ce0c6ba791c30606d1744573a52cb1371404e3d953.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/cfa351916c36e4e9dd98c9ce0c6ba791c30606d1744573a52cb1371404e3d953.jpg)

![d6d77a3ccfdee336f4e7fa74d7558a6271dbc1e8166151a5dffaac9364d78d9c.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/images/d6d77a3ccfdee336f4e7fa74d7558a6271dbc1e8166151a5dffaac9364d78d9c.jpg)

### Tables

![08187e625c05f649f3df7df1bfe11d9db48f2dbbf8d12929b420f3c7bf5b908a.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/08187e625c05f649f3df7df1bfe11d9db48f2dbbf8d12929b420f3c7bf5b908a.jpg)

![2932c62944c70e99aae3eb252b13ad5bb94cb6cc91edfb7df9da8019c9a2b851.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/2932c62944c70e99aae3eb252b13ad5bb94cb6cc91edfb7df9da8019c9a2b851.jpg)

![2d33eb9b5becc5a55548c4306c076f4e835781f555abd9e787566103f9c882f8.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/2d33eb9b5becc5a55548c4306c076f4e835781f555abd9e787566103f9c882f8.jpg)

![581ae3b4f27d5e41849805b519391838e42afb00c56fb42f42d1ef6dfb93033e.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/581ae3b4f27d5e41849805b519391838e42afb00c56fb42f42d1ef6dfb93033e.jpg)

![5c714cdec5d8a217a4ffdca56a98677f0bbd4ec3b5735a9db23aad12cac97f10.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/5c714cdec5d8a217a4ffdca56a98677f0bbd4ec3b5735a9db23aad12cac97f10.jpg)

![5e7f2144cefce1f9782d36b0f7873d07a4f4f0204997c0636a9ed4fa583f4318.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/5e7f2144cefce1f9782d36b0f7873d07a4f4f0204997c0636a9ed4fa583f4318.jpg)

![62cc98fca79ae58d215ec026aaf480c270b6958725d49c864802ac613b7b954d.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/62cc98fca79ae58d215ec026aaf480c270b6958725d49c864802ac613b7b954d.jpg)

![7f6e9199eca5a1f7f3a00ad66766c7ba101a0b407bed8011083832faffe14bd4.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/7f6e9199eca5a1f7f3a00ad66766c7ba101a0b407bed8011083832faffe14bd4.jpg)

![845a54b23ad88d5e04c4d641985f287411050441eba3c588d5945f30a0ac879d.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/845a54b23ad88d5e04c4d641985f287411050441eba3c588d5945f30a0ac879d.jpg)

![89c1120d0d97fa76761cc69f9fbfd80543a69102c2d2bf03f89f3628bfd8464c.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/89c1120d0d97fa76761cc69f9fbfd80543a69102c2d2bf03f89f3628bfd8464c.jpg)

![a32357d9d0eade53888e8e3bc225c53bac0a485e26afbe68f3601f09d97f5358.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/a32357d9d0eade53888e8e3bc225c53bac0a485e26afbe68f3601f09d97f5358.jpg)

![aa46b93611cc7550a7825279cfa260f079054d394f4cfbe50e862ec8bbe6529e.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/aa46b93611cc7550a7825279cfa260f079054d394f4cfbe50e862ec8bbe6529e.jpg)

![af60c92aaf0aa558b5a398a143587c22c4c8ca034bd62ba3bdc7d56e96fc4f29.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/af60c92aaf0aa558b5a398a143587c22c4c8ca034bd62ba3bdc7d56e96fc4f29.jpg)

![b72846d7d3b36b6c900b14a72e3b8f9d5955153d6a7586fdafd4a8b9e7b546e6.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/b72846d7d3b36b6c900b14a72e3b8f9d5955153d6a7586fdafd4a8b9e7b546e6.jpg)

![c7219ffe081649e17b39ff98fce6ea8425dd3c55b2196373276fe86daefefa36.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/c7219ffe081649e17b39ff98fce6ea8425dd3c55b2196373276fe86daefefa36.jpg)

![edce391bfc28c981a187da97fb8c49bd4e5c0edd80ade5bd9780fc1153e255d0.jpg](../icml_results/1463_Theory%20of%20Minimal%20Weight%20Perturbations%20in%20Deep%20Networks%20and%20its%20Applications%20for%20Low-Rank%20Activated%20/tables/edce391bfc28c981a187da97fb8c49bd4e5c0edd80ade5bd9780fc1153e255d0.jpg)

<a id="paper-1443"></a>

## Stochastic Linear Bandits with Parameter Noise


### Tables

![1bbd9b82e9f28562bae8c0e1383c069629eafeaa664a9acaf493d69832690f44.jpg](../icml_results/1464_Stochastic%20Linear%20Bandits%20with%20Parameter%20Noise/tables/1bbd9b82e9f28562bae8c0e1383c069629eafeaa664a9acaf493d69832690f44.jpg)

<a id="paper-1444"></a>

## Unfolded Laplacian Spectral Embedding: A Theoretically Grounded Approach to Dynamic Network Representation


### Images

![01df1306a1fcd2079e5ce53741a6db227cfa6c385e6327f730d7f323b50b739f.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/01df1306a1fcd2079e5ce53741a6db227cfa6c385e6327f730d7f323b50b739f.jpg)

![25faefbb87926827d9b2af64d05ff56968ce5ef1856ead1998fa476eefc4dd26.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/25faefbb87926827d9b2af64d05ff56968ce5ef1856ead1998fa476eefc4dd26.jpg)

![7f80b4f309fa741e4e50380ed883655ce183e99d32d2f752be48b229bbff8ad5.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/7f80b4f309fa741e4e50380ed883655ce183e99d32d2f752be48b229bbff8ad5.jpg)

![9da49e4f8b376fe17ece17f6e2f48fa54c6eeb9cb8225b404c6f7fb255112e65.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/9da49e4f8b376fe17ece17f6e2f48fa54c6eeb9cb8225b404c6f7fb255112e65.jpg)

![a515f92966af289e703e541b891df572e6aeb1f24901e7300e7cb50ffed60335.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/a515f92966af289e703e541b891df572e6aeb1f24901e7300e7cb50ffed60335.jpg)

![bc699a20bc4bdd1e48c01f3e49e20382914929e089bc3a52f5831254589c4ba2.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/bc699a20bc4bdd1e48c01f3e49e20382914929e089bc3a52f5831254589c4ba2.jpg)

![d3dbd8579334e162a013319a4fee2734be4dc85b67378d2eda5bee954d1dfd51.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/d3dbd8579334e162a013319a4fee2734be4dc85b67378d2eda5bee954d1dfd51.jpg)

![eba41e8eca17aad4b980f767968d1bd439330e66666168b0251d911cd63ce414.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/eba41e8eca17aad4b980f767968d1bd439330e66666168b0251d911cd63ce414.jpg)

![f2bb319e3b2747f030df118f3c69327cca19f95bc11b0cb316fc573d54e8dab1.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/images/f2bb319e3b2747f030df118f3c69327cca19f95bc11b0cb316fc573d54e8dab1.jpg)

### Tables

![4553785d0fe6b415bc898c2c3a89693be3b32ca6d345622ea7cbe918eafeae84.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/tables/4553785d0fe6b415bc898c2c3a89693be3b32ca6d345622ea7cbe918eafeae84.jpg)

![ec14e4c90256564f567af16e43ede03c3de24f39a501bde491f504679aa6c146.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/tables/ec14e4c90256564f567af16e43ede03c3de24f39a501bde491f504679aa6c146.jpg)

![fd0bba767d5cb3466d7b1525c8126d13f232b6acf35548b449f6712e222acf0c.jpg](../icml_results/1465_Unfolded%20Laplacian%20Spectral%20Embedding_%20A%20Theoretically%20Grounded%20Approach%20to%20Dynamic%20Network%20Represen/tables/fd0bba767d5cb3466d7b1525c8126d13f232b6acf35548b449f6712e222acf0c.jpg)

<a id="paper-1445"></a>

## Robust Bayesian Optimisation with Unbounded Corruptions


### Images

![78228bee013e97b13511706e14dcc9dce753964c2ac9b5672948971f1b4fcbaf.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/images/78228bee013e97b13511706e14dcc9dce753964c2ac9b5672948971f1b4fcbaf.jpg)

![920c20cc1d6892b44bc85c9cf6b576697a0a18303d6ae0662d5d7e8683699f56.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/images/920c20cc1d6892b44bc85c9cf6b576697a0a18303d6ae0662d5d7e8683699f56.jpg)

![a7f585b10a5d69cf98e4570c4349a7de4686dce37b50457e5cb55ab5b3beafeb.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/images/a7f585b10a5d69cf98e4570c4349a7de4686dce37b50457e5cb55ab5b3beafeb.jpg)

![c0882c8ef7a3169d3f67d32c8e1f3bc537cb91838c3008c7b778e41716effa4e.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/images/c0882c8ef7a3169d3f67d32c8e1f3bc537cb91838c3008c7b778e41716effa4e.jpg)

![d746e2205355d7c56ef9eada64ca3d8c2f461c6dfd88d4e712cb1e87f2cd9b82.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/images/d746e2205355d7c56ef9eada64ca3d8c2f461c6dfd88d4e712cb1e87f2cd9b82.jpg)

### Tables

![34f6549c1f433d9d944fea30468256ab8d0b134b198182ced5e25cbedf932a5c.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/34f6549c1f433d9d944fea30468256ab8d0b134b198182ced5e25cbedf932a5c.jpg)

![36b52fc48c29f1e3f2d98fa9bfd9737d9d21448d1e02afb2e32a6457347cf75a.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/36b52fc48c29f1e3f2d98fa9bfd9737d9d21448d1e02afb2e32a6457347cf75a.jpg)

![73d1d58fe779f1d81b88c8658c54fce48a0c67206c4f6edc3c536f620ca1afdc.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/73d1d58fe779f1d81b88c8658c54fce48a0c67206c4f6edc3c536f620ca1afdc.jpg)

![99381256ff8d72151a2dd26fd3c123ec1fc009e4673761117004f6a1fcce8f28.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/99381256ff8d72151a2dd26fd3c123ec1fc009e4673761117004f6a1fcce8f28.jpg)

![9c580d0ede7d380c19de6f24a0c5c04de1b89d9fe9e360b488beb11d8293f6f4.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/9c580d0ede7d380c19de6f24a0c5c04de1b89d9fe9e360b488beb11d8293f6f4.jpg)

![9dab9a7beecf79aea431cd09a71b31e7e0fecfecb5b1aede84321fe00c9d624a.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/9dab9a7beecf79aea431cd09a71b31e7e0fecfecb5b1aede84321fe00c9d624a.jpg)

![e21b474078d5d47d83d6c050c376c621c6f7682e8def2e3c49f486d03217f4cd.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/e21b474078d5d47d83d6c050c376c621c6f7682e8def2e3c49f486d03217f4cd.jpg)

![fe2c291d74a697e548cdd3def46a79225cc7845e5ade5ee8ffd18544232f3dcf.jpg](../icml_results/1466_Robust%20Bayesian%20Optimisation%20with%20Unbounded%20Corruptions/tables/fe2c291d74a697e548cdd3def46a79225cc7845e5ade5ee8ffd18544232f3dcf.jpg)

<a id="paper-1446"></a>

## BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics


### Images

![0b4cfb4f0d853e3ca67f5b3e732df6d38b613a690793026b071e24a8910153d3.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/images/0b4cfb4f0d853e3ca67f5b3e732df6d38b613a690793026b071e24a8910153d3.jpg)

![832a6774e4e18f56de31aa19290a02846c02f28246d487d0b9e2fe879480a626.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/images/832a6774e4e18f56de31aa19290a02846c02f28246d487d0b9e2fe879480a626.jpg)

![ba86a314152c1f819b582b34b4d3fcdc73db3daf37abb08d4029799b0621f77b.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/images/ba86a314152c1f819b582b34b4d3fcdc73db3daf37abb08d4029799b0621f77b.jpg)

### Tables

![1af7bcaaba8b5a851bc1fdf2d94c1140c7ffb795e4d9320dfef85e3ebeca621c.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/tables/1af7bcaaba8b5a851bc1fdf2d94c1140c7ffb795e4d9320dfef85e3ebeca621c.jpg)

![41cbf4cf1305552dc44c4584e968b5fb90fc63a555658345e9d6d8561f3bdb70.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/tables/41cbf4cf1305552dc44c4584e968b5fb90fc63a555658345e9d6d8561f3bdb70.jpg)

![41de953046059b519006f7918c5d9c58af718fdb1442b30c70e7e5c7db6547bc.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/tables/41de953046059b519006f7918c5d9c58af718fdb1442b30c70e7e5c7db6547bc.jpg)

![7a97e6ea6cc67ce8f2b86d2104666921d722763f21b57ca16c2d08cb414936b3.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/tables/7a97e6ea6cc67ce8f2b86d2104666921d722763f21b57ca16c2d08cb414936b3.jpg)

![c685ff6e54507f8310ebdf42f7224e7f419bf201b80e707a3fcbbe8473bade25.jpg](../icml_results/1467_BioAgent%20Bench_%20An%20AI%20Agent%20Evaluation%20Suite%20for%20Bioinformatics/tables/c685ff6e54507f8310ebdf42f7224e7f419bf201b80e707a3fcbbe8473bade25.jpg)

<a id="paper-1447"></a>

## Understanding Data Temporality Impact on Large Language Models Pre-training


### Images

![0ae0cab0fe0e30c8952b9ecb750652dea64afad438503d956d755529ca44d017.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/0ae0cab0fe0e30c8952b9ecb750652dea64afad438503d956d755529ca44d017.jpg)

![1f59e132f3082b1c8a15bbd5dc619ce94e8b6a4cde448e8d044944be08a207cd.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/1f59e132f3082b1c8a15bbd5dc619ce94e8b6a4cde448e8d044944be08a207cd.jpg)

![231e3281faac3d5cae4b63e3c8d344751ac5533cc55ecb377427d7da145040d6.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/231e3281faac3d5cae4b63e3c8d344751ac5533cc55ecb377427d7da145040d6.jpg)

![5a62c1d210144340123a1bf54ede945f0498f46e40262992e1a0de78eb715d0e.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/5a62c1d210144340123a1bf54ede945f0498f46e40262992e1a0de78eb715d0e.jpg)

![78602e37f8d2eed29c07b608e9de8671e39bbc48ba082bb5e14036c4fac2ce33.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/78602e37f8d2eed29c07b608e9de8671e39bbc48ba082bb5e14036c4fac2ce33.jpg)

![7b61ddf4a320d6573043ad3dc1b7b3e2df7333dc81463b3dc94d6930972eef4a.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/7b61ddf4a320d6573043ad3dc1b7b3e2df7333dc81463b3dc94d6930972eef4a.jpg)

![8f8f63f514481addfe7f63521bf8f3522bfb7a29530c8e19204e0708b0b25741.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/8f8f63f514481addfe7f63521bf8f3522bfb7a29530c8e19204e0708b0b25741.jpg)

![963ffde2a0f5dbc5df7baff47551324ac59c079c9179031b7abda9c0c99a42ca.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/963ffde2a0f5dbc5df7baff47551324ac59c079c9179031b7abda9c0c99a42ca.jpg)

![967b11e03b9540fd2729177abca8dce2ec65f3c6fe697e4d5f5053a1ede08b43.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/967b11e03b9540fd2729177abca8dce2ec65f3c6fe697e4d5f5053a1ede08b43.jpg)

![9fe481089624ce8ad8a8eac15e63ae27a5856aa2cd6c99048e6cfa70bb75365e.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/9fe481089624ce8ad8a8eac15e63ae27a5856aa2cd6c99048e6cfa70bb75365e.jpg)

![a14e12934be8651ddc5a28e7bb4a9979f3691e8ca1ae01c545734c7b5dd070c9.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/a14e12934be8651ddc5a28e7bb4a9979f3691e8ca1ae01c545734c7b5dd070c9.jpg)

![c3cee819d57c7c59dfdd5a4569ba5f5163ce737286472fa5452a6116053d6888.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/c3cee819d57c7c59dfdd5a4569ba5f5163ce737286472fa5452a6116053d6888.jpg)

![ca01ef93de6f199d7f026d236a48dd50f8a30d7a21d68813a996b4fb296d8ba7.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/ca01ef93de6f199d7f026d236a48dd50f8a30d7a21d68813a996b4fb296d8ba7.jpg)

![ed10968b79d4a1303d345a4ed2703a25057f83e8f23dbbd3afdc20e5d50b098f.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/images/ed10968b79d4a1303d345a4ed2703a25057f83e8f23dbbd3afdc20e5d50b098f.jpg)

### Tables

![434f8018fbe3f91a894a22356a6b0165d85398d82e7475a5d541df59c47a774d.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/434f8018fbe3f91a894a22356a6b0165d85398d82e7475a5d541df59c47a774d.jpg)

![46de8a4ee6e831717dd76f10f414342cf020879d1c6595b2bc381b8acd5cda1c.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/46de8a4ee6e831717dd76f10f414342cf020879d1c6595b2bc381b8acd5cda1c.jpg)

![55a727b8b58b4500eb2487f48da85a34cadc509cfc0c40f4da3e68185ea0ee5d.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/55a727b8b58b4500eb2487f48da85a34cadc509cfc0c40f4da3e68185ea0ee5d.jpg)

![5affb1f679a774497d3053f8e1a09a2e679ce8e047b7cc1dbd987f07025a8542.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/5affb1f679a774497d3053f8e1a09a2e679ce8e047b7cc1dbd987f07025a8542.jpg)

![73c76d027320be24fb01a673cfa1a14bbeb0a2a4fb7a26aabfcd902040306081.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/73c76d027320be24fb01a673cfa1a14bbeb0a2a4fb7a26aabfcd902040306081.jpg)

![da151eb45de3b49866933272d5a47d0259469bea31241f2143fdfbf15c33fc1f.jpg](../icml_results/1468_Understanding%20Data%20Temporality%20Impact%20on%20Large%20Language%20Models%20Pre-training/tables/da151eb45de3b49866933272d5a47d0259469bea31241f2143fdfbf15c33fc1f.jpg)

<a id="paper-1448"></a>

## The Two-Hump Problem: Bridging the Difficulty Gap in Mathematical Reinforcement Learning


### Images

![1c0f40856b7de0ecf1286c40cc48400a40776b725464e8bd44fa68929649eeb8.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/1c0f40856b7de0ecf1286c40cc48400a40776b725464e8bd44fa68929649eeb8.jpg)

![243b86eac67da08ad205d7749f75921e15d3ab8c4cfdab912a95a861da58aa19.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/243b86eac67da08ad205d7749f75921e15d3ab8c4cfdab912a95a861da58aa19.jpg)

![43ec308f53135705e21d9f20d3f838ba119c02dfcd9d25a639c7a29b8539c10f.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/43ec308f53135705e21d9f20d3f838ba119c02dfcd9d25a639c7a29b8539c10f.jpg)

![588975e6b3bff2a383fdd74d36dc144e42dc6d0bf49622d88529b0e8ebb96396.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/588975e6b3bff2a383fdd74d36dc144e42dc6d0bf49622d88529b0e8ebb96396.jpg)

![6279dd55427a5c3800d294bce4bdeb10a2f1500c4b8997db22d53990fd917cc3.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/6279dd55427a5c3800d294bce4bdeb10a2f1500c4b8997db22d53990fd917cc3.jpg)

![7eb70f2a7bf41e7326d77bce58aa14c53d260e4e8fea50639f7f6c31d384e025.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/7eb70f2a7bf41e7326d77bce58aa14c53d260e4e8fea50639f7f6c31d384e025.jpg)

![84dd20e2184298f2794f53dc5019fd4b32a5cb70c727d2234d71d3582160abdb.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/84dd20e2184298f2794f53dc5019fd4b32a5cb70c727d2234d71d3582160abdb.jpg)

![c3962a550ac03c85fc634dc43fac54def22746cec97ca33ca1fe7f594c6c6613.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/c3962a550ac03c85fc634dc43fac54def22746cec97ca33ca1fe7f594c6c6613.jpg)

![f4b2681fc12c28a62ed83e4a4961111f1e54010f9c4982352c1ba24e0fb6b744.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/images/f4b2681fc12c28a62ed83e4a4961111f1e54010f9c4982352c1ba24e0fb6b744.jpg)

### Tables

![0768bce38be82b20c070a22b50fd17afed7716dea55ee65cc63bb97f7d7af1e5.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/0768bce38be82b20c070a22b50fd17afed7716dea55ee65cc63bb97f7d7af1e5.jpg)

![0e414c20f40db9421e9c0d54f16a8ba2dc30813fea839955e0c0df17464cb70b.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/0e414c20f40db9421e9c0d54f16a8ba2dc30813fea839955e0c0df17464cb70b.jpg)

![1afe232eb08c0e89509d2c1aff0f084beb97ed28e70437198ab91fd85f146d6b.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/1afe232eb08c0e89509d2c1aff0f084beb97ed28e70437198ab91fd85f146d6b.jpg)

![28ff77403635d30308b8a95a5b962d17868c4e479cf4230fadad27626be51695.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/28ff77403635d30308b8a95a5b962d17868c4e479cf4230fadad27626be51695.jpg)

![2e2445a022eb0757c0036b80081dd3ddbf58942f36398675a64d58af75adb8f2.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/2e2445a022eb0757c0036b80081dd3ddbf58942f36398675a64d58af75adb8f2.jpg)

![32578df1404003f50fd3d25b2e3f26db98bfed87ee95738130d02189b0281771.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/32578df1404003f50fd3d25b2e3f26db98bfed87ee95738130d02189b0281771.jpg)

![37c2c858e14efb16b350705cc9fba588c2a53d6bc50b1dddd3bc8f652a807a8c.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/37c2c858e14efb16b350705cc9fba588c2a53d6bc50b1dddd3bc8f652a807a8c.jpg)

![3c6770e1b7689a513498dabc9788cfbebb0eb0e555d60580fb1cc619d0bed675.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/3c6770e1b7689a513498dabc9788cfbebb0eb0e555d60580fb1cc619d0bed675.jpg)

![43539a7ad99f75ef1a71ed034cd586bdbc282a7516cbd4fa1f02c45d35067b09.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/43539a7ad99f75ef1a71ed034cd586bdbc282a7516cbd4fa1f02c45d35067b09.jpg)

![54aabf761897a5d7ba9ac08963e72f2213593cb15f2ccee05a6bec87de4d1bac.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/54aabf761897a5d7ba9ac08963e72f2213593cb15f2ccee05a6bec87de4d1bac.jpg)

![5c912737f6c33dc6dde9a2f57879f2abe7ed33f20b0cb9f21dfecfa7d38279a0.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/5c912737f6c33dc6dde9a2f57879f2abe7ed33f20b0cb9f21dfecfa7d38279a0.jpg)

![712f9374450d399f3585e842caf8fa3070e74a97c893ee289300f19458d3df08.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/712f9374450d399f3585e842caf8fa3070e74a97c893ee289300f19458d3df08.jpg)

![762c707ad6b176bb75dd612032b82fe7d38c50ea120f6a083d117c3592b8283c.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/762c707ad6b176bb75dd612032b82fe7d38c50ea120f6a083d117c3592b8283c.jpg)

![799389c536001afdd93e80193a6876e6be8cd3b02b3c52639827fb171bd72353.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/799389c536001afdd93e80193a6876e6be8cd3b02b3c52639827fb171bd72353.jpg)

![8a02ab3a85c4741bbd2eaebbd101522a6c8b58c60165f72cba3e53eb52cfedaf.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/8a02ab3a85c4741bbd2eaebbd101522a6c8b58c60165f72cba3e53eb52cfedaf.jpg)

![921482943ad1aaa4a1e7971c99911c3cb7314cd15b37694c1d15a0cbf2fce423.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/921482943ad1aaa4a1e7971c99911c3cb7314cd15b37694c1d15a0cbf2fce423.jpg)

![92911b331cbb6a41b9465348c71fa56854ccee5fb31f8a9915042a15545824ad.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/92911b331cbb6a41b9465348c71fa56854ccee5fb31f8a9915042a15545824ad.jpg)

![a4b3554d7050fe5d210071d55d63fef72422a415d4cc4f9b5c4851f4e8356c45.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/a4b3554d7050fe5d210071d55d63fef72422a415d4cc4f9b5c4851f4e8356c45.jpg)

![a9216dfb170c168930ade8839e474c5df617cda7057abd376e56e5250d52007f.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/a9216dfb170c168930ade8839e474c5df617cda7057abd376e56e5250d52007f.jpg)

![b8b8dc3ed155cf18e9d48b399d5f53618a3f53b666aa7a153329d4865167be89.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/b8b8dc3ed155cf18e9d48b399d5f53618a3f53b666aa7a153329d4865167be89.jpg)

![d96429bdec3790d492dc0b72da320837db0db9389aca5a05ca9f1953b68abb23.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/d96429bdec3790d492dc0b72da320837db0db9389aca5a05ca9f1953b68abb23.jpg)

![d9eeaa854c58464126f496a96258128280ca26ed8e899b877d70a89be2a80be9.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/d9eeaa854c58464126f496a96258128280ca26ed8e899b877d70a89be2a80be9.jpg)

![e1d1972ca9f94dccdfa01796d34267658262c34f42b347b5ccbf2679091eeb69.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/e1d1972ca9f94dccdfa01796d34267658262c34f42b347b5ccbf2679091eeb69.jpg)

![ec65f2d890dd4d269bddbd2cd7e1f4bd936ec56814a51715915827bac8e86af0.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/ec65f2d890dd4d269bddbd2cd7e1f4bd936ec56814a51715915827bac8e86af0.jpg)

![eea0d47d20400a8dad166326c62efacb532d9ed6c9c7cb7b8e193b1f12b9cc57.jpg](../icml_results/1469_The%20Two-Hump%20Problem_%20Bridging%20the%20Difficulty%20Gap%20in%20Mathematical%20Reinforcement%20Learning/tables/eea0d47d20400a8dad166326c62efacb532d9ed6c9c7cb7b8e193b1f12b9cc57.jpg)

<a id="paper-1449"></a>

## Trading Complexity for Expressivity Through Structured Generalized Linear Token Mixing


### Images

![624c5ad18e349b9b17dfa4a5c40149e92919b74e85b3d96bcd9417e36f8effe1.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/images/624c5ad18e349b9b17dfa4a5c40149e92919b74e85b3d96bcd9417e36f8effe1.jpg)

![a840436278d4064be3cf5dfb68b32dec459f4e6bc860d054a324ea86a4486126.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/images/a840436278d4064be3cf5dfb68b32dec459f4e6bc860d054a324ea86a4486126.jpg)

![e628084f76c2c9fd37d751b398306de3ebd20015dc882960eb9de0807c1fbae2.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/images/e628084f76c2c9fd37d751b398306de3ebd20015dc882960eb9de0807c1fbae2.jpg)

### Tables

![474b0c3e7ac1511169b2f8af07313da6a84193573026b83bc6d26032c3b2d277.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/tables/474b0c3e7ac1511169b2f8af07313da6a84193573026b83bc6d26032c3b2d277.jpg)

![6b91eb0a196237ab8a44ad96a4d448bbc3877f3df9c912c9dabd0661d7273d1d.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/tables/6b91eb0a196237ab8a44ad96a4d448bbc3877f3df9c912c9dabd0661d7273d1d.jpg)

![aa765985f48425eb943e7526e7f88142be8aa08ac3d8a56f95c9b17bf4f6dbef.jpg](../icml_results/1470_Trading%20Complexity%20for%20Expressivity%20Through%20Structured%20Generalized%20Linear%20Token%20Mixing/tables/aa765985f48425eb943e7526e7f88142be8aa08ac3d8a56f95c9b17bf4f6dbef.jpg)

<a id="paper-1450"></a>

## Local Policies for Graph-Structured Markov Decision Processes


### Images

![1cb24021b2bd74b2a4cc2d49b9a6fd4a4ea5ec11262821cd82a0626f9d1449d1.jpg](../icml_results/1471_Local%20Policies%20for%20Graph-Structured%20Markov%20Decision%20Processes/images/1cb24021b2bd74b2a4cc2d49b9a6fd4a4ea5ec11262821cd82a0626f9d1449d1.jpg)

<a id="paper-1451"></a>

## Curated Synthetic Data Doesn’t Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences


### Images

![0e329bae01f152dbf39cb429c5754fe5db11c050306d80efd178dcd962a13d4b.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/0e329bae01f152dbf39cb429c5754fe5db11c050306d80efd178dcd962a13d4b.jpg)

![0f96890b545a3f9d3978bd44e25777f48874a6c24618134024566ce4c84eea61.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/0f96890b545a3f9d3978bd44e25777f48874a6c24618134024566ce4c84eea61.jpg)

![18722da29f8f9afcec8b99d43e78174060b25fcc83bd8be0f3e896d1dc52fc3b.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/18722da29f8f9afcec8b99d43e78174060b25fcc83bd8be0f3e896d1dc52fc3b.jpg)

![2c53fad244bfa40caaaa3ef00614d0249c65d44318ad8d6ab125eb30b382d401.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/2c53fad244bfa40caaaa3ef00614d0249c65d44318ad8d6ab125eb30b382d401.jpg)

![2d5e2057ff7cf50736a8fb90b572381baa1b5eb4b99e8d0cbbda57c73a2b5e40.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/2d5e2057ff7cf50736a8fb90b572381baa1b5eb4b99e8d0cbbda57c73a2b5e40.jpg)

![43b99b73ba843a30812befd8b7daeed494723dbf907f38bef227da85ccf59e65.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/43b99b73ba843a30812befd8b7daeed494723dbf907f38bef227da85ccf59e65.jpg)

![47b8154e2eb1ec1d36268bbf83de06c46e9efd8cf7ca753639f78067cbbca246.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/47b8154e2eb1ec1d36268bbf83de06c46e9efd8cf7ca753639f78067cbbca246.jpg)

![4e25d2ce82bd18993097415845bc011e52160b195397cf77b9c852dd153158b7.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/4e25d2ce82bd18993097415845bc011e52160b195397cf77b9c852dd153158b7.jpg)

![537523bf7d45480e703f78bf8346ba12f01a6b4089dc4c59c7f8dee8a72afc3a.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/537523bf7d45480e703f78bf8346ba12f01a6b4089dc4c59c7f8dee8a72afc3a.jpg)

![72b86365fcd84a637f774ef9e48ec66e736591a5b71e146ef47a6bf84a879cb9.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/72b86365fcd84a637f774ef9e48ec66e736591a5b71e146ef47a6bf84a879cb9.jpg)

![804560c4520877397a70565a4f2b8fc668149d6c4dfec10034eb5af529ca5a60.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/804560c4520877397a70565a4f2b8fc668149d6c4dfec10034eb5af529ca5a60.jpg)

![912f51d88473db5bae87e5c01be92a08b561fc88b20561f96c532057d0349f0d.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/912f51d88473db5bae87e5c01be92a08b561fc88b20561f96c532057d0349f0d.jpg)

![ea01aa649a76fe7edc186eacbd3c5a72a807a0dbf4bfdb6c05dc1cc7dd3d49af.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/ea01aa649a76fe7edc186eacbd3c5a72a807a0dbf4bfdb6c05dc1cc7dd3d49af.jpg)

![f229a676929752ef3415a1e67b82c71425b560c8a50ac9ba1000ddc585bd6fe2.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/f229a676929752ef3415a1e67b82c71425b560c8a50ac9ba1000ddc585bd6fe2.jpg)

![fd6b7f2d44d39310e3643ebc35dd10a06a444d6191cd135d7829466092fea15f.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/images/fd6b7f2d44d39310e3643ebc35dd10a06a444d6191cd135d7829466092fea15f.jpg)

### Tables

![6c4a38722dc09c9692fab3d05971a90a6a15fd6c1c6a54c92094abf1fb5af9d2.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/tables/6c4a38722dc09c9692fab3d05971a90a6a15fd6c1c6a54c92094abf1fb5af9d2.jpg)

![79f8ad3cef1c3132aeca1c74f41edb1578eb670c9cdfe92ec2642fbb4f08e3fd.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/tables/79f8ad3cef1c3132aeca1c74f41edb1578eb670c9cdfe92ec2642fbb4f08e3fd.jpg)

![afcb75e3277adf921c81fa842cbf659fd8ba71791020afbc2d130298cf2205d6.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/tables/afcb75e3277adf921c81fa842cbf659fd8ba71791020afbc2d130298cf2205d6.jpg)

![eeffbf457833240d1c1af675198ac95a23ccb8d6c445816e9c89f1b9a0b4590c.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/tables/eeffbf457833240d1c1af675198ac95a23ccb8d6c445816e9c89f1b9a0b4590c.jpg)

![fec3e6374531428472bd620d8708c3c6da741b8df3cf50e471651a92648cc119.jpg](../icml_results/1472_Curated%20Synthetic%20Data%20Doesn%E2%80%99t%20Have%20to%20Collapse_%20A%20Theoretical%20Study%20of%20Generative%20Retraining%20with%20P/tables/fec3e6374531428472bd620d8708c3c6da741b8df3cf50e471651a92648cc119.jpg)

<a id="paper-1452"></a>

## NAVIGATE: Evaluating Visual-Guided Search Decision-Making on the Open Web

### Images

![05fb4ff9005f06b46e06bdcf88960722e301719ee1cd92b96f31048441fd445a.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/05fb4ff9005f06b46e06bdcf88960722e301719ee1cd92b96f31048441fd445a.jpg)

![09664c1401ef942af56d93063221de330fa2059687eea684f050516027296b26.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/09664c1401ef942af56d93063221de330fa2059687eea684f050516027296b26.jpg)

![0ce47809a9ee474c3225aaf15fac30fd4b95750cfa9ac618d575d5343f61fc11.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/0ce47809a9ee474c3225aaf15fac30fd4b95750cfa9ac618d575d5343f61fc11.jpg)

![1d9401bcb7e1c48b4322c8385501ff3d560ed9c1560143c385b07ecda4af93c9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/1d9401bcb7e1c48b4322c8385501ff3d560ed9c1560143c385b07ecda4af93c9.jpg)

![1de0a1d63151d49c6fabe1ac5efb71ad36f79e87b5b00a3d241620194a89ce9c.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/1de0a1d63151d49c6fabe1ac5efb71ad36f79e87b5b00a3d241620194a89ce9c.jpg)

![211744b4fb56b0cfada0779199cdaf360636d62b1cf374c119c2d9cd5c813d85.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/211744b4fb56b0cfada0779199cdaf360636d62b1cf374c119c2d9cd5c813d85.jpg)

![240687eb77f85400ef78e818fbad6f32b8e24138617c59aad3b7ed94e97dd66b.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/240687eb77f85400ef78e818fbad6f32b8e24138617c59aad3b7ed94e97dd66b.jpg)

![27165a0b271403ea23802d66ed51e1eaf49e7cb4fc52410104017b412bd928b0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/27165a0b271403ea23802d66ed51e1eaf49e7cb4fc52410104017b412bd928b0.jpg)

![2b478e33fc3152e68f1748891b4663697dd1acb75a129585157009cb6a072e26.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/2b478e33fc3152e68f1748891b4663697dd1acb75a129585157009cb6a072e26.jpg)

![3a710d444071bb21f28be681ffcd9d78d85b09a35c792182b473b6b9b0c8ae61.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/3a710d444071bb21f28be681ffcd9d78d85b09a35c792182b473b6b9b0c8ae61.jpg)

![45e8daae3517522848394e108b78f5e27a234d75fa0a208978b4906e97cfc8e0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/45e8daae3517522848394e108b78f5e27a234d75fa0a208978b4906e97cfc8e0.jpg)

![478774a8ad32c4e14768040dd4523e6a6ab1d484f97d3e509f53f93c1b4405d2.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/478774a8ad32c4e14768040dd4523e6a6ab1d484f97d3e509f53f93c1b4405d2.jpg)

![551a3c8cb5e60a994549aced2129b05971c07eed7d52bbbb8f90ca32962fa77e.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/551a3c8cb5e60a994549aced2129b05971c07eed7d52bbbb8f90ca32962fa77e.jpg)

![5c6f7fd8270a0b3836d8b21b4dd817904c9e926b8d1b8d674e8c4c4618c44d57.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5c6f7fd8270a0b3836d8b21b4dd817904c9e926b8d1b8d674e8c4c4618c44d57.jpg)

![5cde67a52649b3c27377a4697ea54b7bd5dc9f8ee00a6b3d2ceae20eca5bcbd7.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5cde67a52649b3c27377a4697ea54b7bd5dc9f8ee00a6b3d2ceae20eca5bcbd7.jpg)

![5d6b551eac2b0553ddfe10854b8fb59513c6b64e081f01295cdb93c53f75dde0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5d6b551eac2b0553ddfe10854b8fb59513c6b64e081f01295cdb93c53f75dde0.jpg)

![5e65f54114a43365ef27e979e02252af0b327746a233ddd1caf7533db736a9d3.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5e65f54114a43365ef27e979e02252af0b327746a233ddd1caf7533db736a9d3.jpg)

![5f445b58b36192474e0f593ee766ac3d9faa386ed162dcc3518a6bb21ac528f9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5f445b58b36192474e0f593ee766ac3d9faa386ed162dcc3518a6bb21ac528f9.jpg)

![7316ae64e2805b229ee4f4e34a84887753a70ff2e3c5d2cf7820e63ca5ddf9f6.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/7316ae64e2805b229ee4f4e34a84887753a70ff2e3c5d2cf7820e63ca5ddf9f6.jpg)

![7c6a4824b9fa4b625a5a88275d3adff1d18a497fdc623ff19ce79bf933478b79.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/7c6a4824b9fa4b625a5a88275d3adff1d18a497fdc623ff19ce79bf933478b79.jpg)

![a206107934e09e42777ca22b6fd5cb7541d1f22b8dcf111be55301ff2d6549dd.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a206107934e09e42777ca22b6fd5cb7541d1f22b8dcf111be55301ff2d6549dd.jpg)

![a3748a6a889ceb4040e974fcca706865a06de47781cc45394060f5f533f10e4d.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a3748a6a889ceb4040e974fcca706865a06de47781cc45394060f5f533f10e4d.jpg)

![a744ff389ad855fbecc9dd939b2177a71ddd0e487d50007da7d8fb2e3e84a0b9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a744ff389ad855fbecc9dd939b2177a71ddd0e487d50007da7d8fb2e3e84a0b9.jpg)

![c0d857b162dacdc20f4bc0c9d978a0034f628a1a7532730fe9029b35b2bdd5a0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c0d857b162dacdc20f4bc0c9d978a0034f628a1a7532730fe9029b35b2bdd5a0.jpg)

![c28146104a452ad4ac7ba9ed76cecafc613fe4a53310951ce72f6d78198c43ed.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c28146104a452ad4ac7ba9ed76cecafc613fe4a53310951ce72f6d78198c43ed.jpg)

![c7b7306bd04bdd79a497a391b2ef6291977bb86d7c5668b8be59c3804bf31749.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c7b7306bd04bdd79a497a391b2ef6291977bb86d7c5668b8be59c3804bf31749.jpg)

![d41f39854c1279b700d8010bf9e39c720b6654dc0cbd92eb8adcaec3eff5d909.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/d41f39854c1279b700d8010bf9e39c720b6654dc0cbd92eb8adcaec3eff5d909.jpg)

![e41909636de20c0932ffe773f4d42e5054de159b838431aaa73f27e15c3ab178.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/e41909636de20c0932ffe773f4d42e5054de159b838431aaa73f27e15c3ab178.jpg)

![ea7ef55d82302a521912240366e351d3bb55951d0d185998f6de2c9dde9865b4.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/ea7ef55d82302a521912240366e351d3bb55951d0d185998f6de2c9dde9865b4.jpg)

![ef8062053eb17c5baf0b8a8b7e936780af1a52d0303697db01e95c960bffdb2b.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/ef8062053eb17c5baf0b8a8b7e936780af1a52d0303697db01e95c960bffdb2b.jpg)

![efe71c6ce8af2d20cac32786d9d7f44b366c878d262c12af27a3545b1b10669a.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/efe71c6ce8af2d20cac32786d9d7f44b366c878d262c12af27a3545b1b10669a.jpg)

![f922a11762efdd25a66adc0a79450c1a5dfba7b0d90753c2f08bba4b0ac0c625.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/f922a11762efdd25a66adc0a79450c1a5dfba7b0d90753c2f08bba4b0ac0c625.jpg)

### Tables

![0a8e9e20b17bbe6a14777dd7e0b1a5df41448d4dac529052b705e133adc56a4f.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/0a8e9e20b17bbe6a14777dd7e0b1a5df41448d4dac529052b705e133adc56a4f.jpg)

![84c629591a833029561fa0104e470182f8d261fd8b4eac027cf31980546ddbca.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/84c629591a833029561fa0104e470182f8d261fd8b4eac027cf31980546ddbca.jpg)

![a4aab0c2be438c817614063ffefd86403c36ca86fed598668c29bd4b81a7da5c.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/a4aab0c2be438c817614063ffefd86403c36ca86fed598668c29bd4b81a7da5c.jpg)
