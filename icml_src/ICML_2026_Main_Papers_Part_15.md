# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 15 of 197

## 目录 (Table of Contents)

1. [Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment](#paper-0463)
2. [PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models](#paper-0464)
3. [Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions](#paper-0465)
4. [FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU](#paper-0466)
5. [Securing Multimodal AI through Internal Information Decomposition](#paper-0467)
6. [PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion](#paper-0468)
7. [Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models](#paper-0469)
8. [Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](#paper-0470)
9. [Latent Laplace Diffusion for Irregular Multivariate Time Series](#paper-0471)
10. [Hierarchical Successor Representation for Robust Transfer](#paper-0472)
11. [Geometric Flow Grounding: A Unified Manifold Decoupling Framework for Dynamics Discovery and Verification](#paper-0473)
12. [Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](#paper-0474)
13. [Spectral-Informed Neural Networks Outperform Spectral methods in High-dimensional PDEs](#paper-0475)
14. [FIRE: Multi-Fidelity Regression with Distribution-Conditioned In-Context Learning Using Tabular Foundation Models](#paper-0476)
15. [Improved Dimension Dependence for Bandit Convex Optimization with Gradient Variation](#paper-0477)
16. [Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference](#paper-0478)
17. [Markov Chain Monte Carlo without Evaluating the Target: an Auxiliary Variable Approach](#paper-0479)
18. [Learning to Discover at Test Time](#paper-0480)
19. [End-to-End Compression for Tabular Foundation Models](#paper-0481)
20. [FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](#paper-0482)
21. [daVinci-Dev: Agent-native Mid-training for Software Engineering](#paper-0483)
22. [PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space](#paper-0484)
23. [EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video](#paper-0485)
24. [CoEvol-NO: State and Coordinate Co-Evolution with an Error-Driven Predictor-Corrector Paradigm for Neural Operator Transformer](#paper-0486)
25. [3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](#paper-0487)
26. [Failure-Driven Workflow Refinement](#paper-0488)
27. [DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making](#paper-0489)
28. [The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models](#paper-0490)
29. [UniMapping: Unified SLAM Framework for Map-Centric Embodied Perception](#paper-0491)
30. [HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](#paper-0492)
31. [How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning](#paper-0493)
32. [Prescriptive Scaling Reveals the Evolution of Language Model Capabilities](#paper-0494)
33. [Characterizing, Evaluating, and Optimizing Complex Reasoning](#paper-0495)

---


<a id="paper-0463"></a>

## Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment

### Images

![09d2d996cb5c63f6d71fa1be20cf5895767817f6d713dcf870d3bf10a666047e.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/09d2d996cb5c63f6d71fa1be20cf5895767817f6d713dcf870d3bf10a666047e.jpg)

![4bf54fd020e4140af2674c4e61ef0eea84c2e6c1308fbd8cf738edaffd00025b.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/4bf54fd020e4140af2674c4e61ef0eea84c2e6c1308fbd8cf738edaffd00025b.jpg)

![6352a6fd2600a7e2a7f4023e65da0e24e8daec9cd4d5b95887c40e231a7e4855.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/6352a6fd2600a7e2a7f4023e65da0e24e8daec9cd4d5b95887c40e231a7e4855.jpg)

![6d7452b7400fc99efca98674c8167bf514806068e89153461e7097e1473e994f.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/6d7452b7400fc99efca98674c8167bf514806068e89153461e7097e1473e994f.jpg)

![7fcf7869b781722d3a829568cfd935b85a6404cbaf82f9509a106c7fcc7f546c.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/7fcf7869b781722d3a829568cfd935b85a6404cbaf82f9509a106c7fcc7f546c.jpg)

![94c680dbcb3ccb89fa716bd35685defc084b76b6c07fb0a51320cbb5ed42b624.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/94c680dbcb3ccb89fa716bd35685defc084b76b6c07fb0a51320cbb5ed42b624.jpg)

![9ce95869568ad818949607af47f4d365d9514634ec811bdb0b492fc12db0b586.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/9ce95869568ad818949607af47f4d365d9514634ec811bdb0b492fc12db0b586.jpg)

![b3dde4de515aa9de4cd688bbb48188a70be3f15c13ba8d936a4879ac747ed022.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/b3dde4de515aa9de4cd688bbb48188a70be3f15c13ba8d936a4879ac747ed022.jpg)

![bb3b305ec6207e87e191a8907130084f26ea6fef274eda73f4bcf3535e6b2ef4.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/bb3b305ec6207e87e191a8907130084f26ea6fef274eda73f4bcf3535e6b2ef4.jpg)

![cbf63088918cb4b38e86dc88768291d37ca09cd1b3fb812cfdeb8c684f22e00a.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/cbf63088918cb4b38e86dc88768291d37ca09cd1b3fb812cfdeb8c684f22e00a.jpg)

![d68630c3ef4c35d917e701b050702a622711f5cd1466aaf663fa4ebef58324ee.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/images/d68630c3ef4c35d917e701b050702a622711f5cd1466aaf663fa4ebef58324ee.jpg)

### Tables

![04ef695377f28bb145640fbf8c790475f7d5708a3ef7d74ee14174374607376b.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/04ef695377f28bb145640fbf8c790475f7d5708a3ef7d74ee14174374607376b.jpg)

![0df5036df2de4c723b056565ded57535c82f11f1ae5aaca6fd739aab31cc41b2.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/0df5036df2de4c723b056565ded57535c82f11f1ae5aaca6fd739aab31cc41b2.jpg)

![1cfd792075f5ccf82c4073bcde1feb26c6b3dbaa098453b9f3b93a95d6574781.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/1cfd792075f5ccf82c4073bcde1feb26c6b3dbaa098453b9f3b93a95d6574781.jpg)

![21270ac6badffde4080bb7a3ad6164f56e4825c519ed2ce9934b1b968f63a76a.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/21270ac6badffde4080bb7a3ad6164f56e4825c519ed2ce9934b1b968f63a76a.jpg)

![31e3fe54d2a60b9c28d4402d326f9fc863da53d58399f842e34ecbf1469be838.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/31e3fe54d2a60b9c28d4402d326f9fc863da53d58399f842e34ecbf1469be838.jpg)

![46366431bc2c2960ded15f9aac14608816a46a960244b8a7f50fb8bed534f5fe.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/46366431bc2c2960ded15f9aac14608816a46a960244b8a7f50fb8bed534f5fe.jpg)

![5003c0fff160e8003e8154f23d5c527f1928f392aa35bb8ee3743ebf0372667f.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/5003c0fff160e8003e8154f23d5c527f1928f392aa35bb8ee3743ebf0372667f.jpg)

![55765d3c83a3a81d625878dbedd2eb23afb460ecbc15828eada419774faebe33.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/55765d3c83a3a81d625878dbedd2eb23afb460ecbc15828eada419774faebe33.jpg)

![5dfb766c0c8fc49727860d52c0354b9ce5b4e66e5a7c7e7bd77bf50379a6777b.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/5dfb766c0c8fc49727860d52c0354b9ce5b4e66e5a7c7e7bd77bf50379a6777b.jpg)

![6d54848e2ad6b9544d7b14e413ee95dd28c59d285c5ca7fb8249b7b9dee63a4f.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/6d54848e2ad6b9544d7b14e413ee95dd28c59d285c5ca7fb8249b7b9dee63a4f.jpg)

![8bb7e555dad15af14441a7de65ee9efc185c4e50ed5d204b863d244372679bee.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/8bb7e555dad15af14441a7de65ee9efc185c4e50ed5d204b863d244372679bee.jpg)

![9aa23fac5b240c564adb7da537e7354b604833e4715033154eee97ebe8d4bce9.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/9aa23fac5b240c564adb7da537e7354b604833e4715033154eee97ebe8d4bce9.jpg)

![a027db6301ad0e422f6fe8d694e52a4544608434dd016dcef9af92e08debe733.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/a027db6301ad0e422f6fe8d694e52a4544608434dd016dcef9af92e08debe733.jpg)

![ada656d3ef76ba8e30ef103b559833ad62a796bb04537aef12c954757be24dad.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/ada656d3ef76ba8e30ef103b559833ad62a796bb04537aef12c954757be24dad.jpg)

![b067ec62c0c6577a6453eef389d898854fef349612b372d851d633b6f61d3083.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/b067ec62c0c6577a6453eef389d898854fef349612b372d851d633b6f61d3083.jpg)

![bcc7957ca52bbf8da56ea6825ae2820b36474cf70aeff8118af403ea945f4bf2.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/bcc7957ca52bbf8da56ea6825ae2820b36474cf70aeff8118af403ea945f4bf2.jpg)

![d4af366cbdab3de348ce762cda7dd2638cf12c73db8c9981b1a7f0bf08868eeb.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/d4af366cbdab3de348ce762cda7dd2638cf12c73db8c9981b1a7f0bf08868eeb.jpg)

![df8b67a21fa203c5ea2e5dbee602012e3d2c2f5572b1fed16c3700d3baf6c48c.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/df8b67a21fa203c5ea2e5dbee602012e3d2c2f5572b1fed16c3700d3baf6c48c.jpg)

![e1216ea430f1a93b4181672d7a7994153a703a6820c315af6fff079bc2d8730e.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/e1216ea430f1a93b4181672d7a7994153a703a6820c315af6fff079bc2d8730e.jpg)

![e835f37193e2dfc5a9115416284b0e45c79a4a8d28e53c2ab7533a38c4ffc1a3.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/e835f37193e2dfc5a9115416284b0e45c79a4a8d28e53c2ab7533a38c4ffc1a3.jpg)

![ef5747e327f65dee264bcf11b6657ebfa6725e00e59952388b3d5410a9a8200d.jpg](../icml_results/471_Suppress%20and%20Diversify_%20Refining%20Robust%20Pathways%20for%20Corruption%20Robustness/tables/ef5747e327f65dee264bcf11b6657ebfa6725e00e59952388b3d5410a9a8200d.jpg)

## Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment


### Images

![56d4b558028ec9c0a58db3f57bf21c2c5b80dfd1e344d796872631a12f5400f5.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/images/56d4b558028ec9c0a58db3f57bf21c2c5b80dfd1e344d796872631a12f5400f5.jpg)

![e99a56106a260073e0add2779fb5a3b6edc9acf9aa6aee83d9db0150f02d7a40.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/images/e99a56106a260073e0add2779fb5a3b6edc9acf9aa6aee83d9db0150f02d7a40.jpg)

### Tables

![163066f8fedd6436ee6093c921acf25a81c692f832d89308b6fc6c780dbefea7.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/163066f8fedd6436ee6093c921acf25a81c692f832d89308b6fc6c780dbefea7.jpg)

![412fdce911ad112dc25f1a68190f1fc7df110627a940fc43d1b064b4f0bec4f3.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/412fdce911ad112dc25f1a68190f1fc7df110627a940fc43d1b064b4f0bec4f3.jpg)

![469f9e289a64d8defdc71363dfe16398f80d7bdb9db2c14e48e8beac583d1eea.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/469f9e289a64d8defdc71363dfe16398f80d7bdb9db2c14e48e8beac583d1eea.jpg)

![7c8207a2c1631474541161c054459bc60d902fd7d7dd5cf0dcf96b7e14dd13b9.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/7c8207a2c1631474541161c054459bc60d902fd7d7dd5cf0dcf96b7e14dd13b9.jpg)

![838e44d2d437aa55672e799249c75cfb82a9b323a61462b32070ad6ce32c1f72.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/838e44d2d437aa55672e799249c75cfb82a9b323a61462b32070ad6ce32c1f72.jpg)

![b7e806e000ecd5f7f25265288da2cec1c2c1492ba48442c57f6d3b2a165c9a73.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/b7e806e000ecd5f7f25265288da2cec1c2c1492ba48442c57f6d3b2a165c9a73.jpg)

![bdc11bb74d95098426652b77c9856b421304a4fe1bfd7104c2a7dfad4d0a8b94.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/bdc11bb74d95098426652b77c9856b421304a4fe1bfd7104c2a7dfad4d0a8b94.jpg)

![f3d2b2120ebada634ed6e65c991cd487b23e6131360076fbe115903a7e4d1af4.jpg](../icml_results/472_Conditional%20Equivalence%20of%20DPO%20and%20RLHF_%20Implicit%20Assumption%2C%20Failure%20Modes%2C%20and%20Provable%20Alignment/tables/f3d2b2120ebada634ed6e65c991cd487b23e6131360076fbe115903a7e4d1af4.jpg)

<a id="paper-0464"></a>

## PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models


### Images

![003b59a829cafdaaf7cbf6bc628c6d7761a280fc049c49e902b1e79a719aafd6.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/003b59a829cafdaaf7cbf6bc628c6d7761a280fc049c49e902b1e79a719aafd6.jpg)

![026fc39aa8021c359516617a51529cd085dad0118fd1631aa06ff6d83d4a95e7.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/026fc39aa8021c359516617a51529cd085dad0118fd1631aa06ff6d83d4a95e7.jpg)

![0399ad0c896333672f1343f88933ff4e999c5028944ebc55202e76a2ed2312e8.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/0399ad0c896333672f1343f88933ff4e999c5028944ebc55202e76a2ed2312e8.jpg)

![0e2a5b0ab61ec983f55ab5db1aab59a5a21dea506b7e23aff0d3c737b91baa40.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/0e2a5b0ab61ec983f55ab5db1aab59a5a21dea506b7e23aff0d3c737b91baa40.jpg)

![179507bb72fd64b46261015cf2635c4115003b24652c3270d99da18af579e1f4.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/179507bb72fd64b46261015cf2635c4115003b24652c3270d99da18af579e1f4.jpg)

![2ef090eec13caf71b7e4c3bcb4d73692f4c6a378b1bc7cc7803eb175828b5681.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/2ef090eec13caf71b7e4c3bcb4d73692f4c6a378b1bc7cc7803eb175828b5681.jpg)

![88d4b306453b2fbc26e9bc9dd7cb2dcdad7ea656ef9655cf8cc09ad0b4bd47d8.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/88d4b306453b2fbc26e9bc9dd7cb2dcdad7ea656ef9655cf8cc09ad0b4bd47d8.jpg)

![b0b3ab66d35dacf6ace764fda38225974879db9a0d227b2f3f7f67b9d923ee4d.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/b0b3ab66d35dacf6ace764fda38225974879db9a0d227b2f3f7f67b9d923ee4d.jpg)

![b644c0f20b1c21630f2f46fefbc0bd5aa0b4a1bb39045876ceb2a7ea7d1f9fc9.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/b644c0f20b1c21630f2f46fefbc0bd5aa0b4a1bb39045876ceb2a7ea7d1f9fc9.jpg)

![b68e83f2b6fb1e374d0a54c98c2fc0364bdfbb9ac45fcb3cad4b43a89e6f6f35.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/b68e83f2b6fb1e374d0a54c98c2fc0364bdfbb9ac45fcb3cad4b43a89e6f6f35.jpg)

![beca082226226c120f9408a46e52dbcb2bd46a2a16c9cd6be5e95686551e7cfd.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/beca082226226c120f9408a46e52dbcb2bd46a2a16c9cd6be5e95686551e7cfd.jpg)

![f19269dab7e0710d649845b35cfe8a19b3720ed783d77c361c918aad2e21c9e4.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/f19269dab7e0710d649845b35cfe8a19b3720ed783d77c361c918aad2e21c9e4.jpg)

![fd81f60fc54c835d0cde57841a6922aa51f516f8b8dd9d54ad134dab4f95728f.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/images/fd81f60fc54c835d0cde57841a6922aa51f516f8b8dd9d54ad134dab4f95728f.jpg)

### Tables

![01b1031eac16895291be6af835b9d041d1b39be1292393de7414506262ef0db4.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/01b1031eac16895291be6af835b9d041d1b39be1292393de7414506262ef0db4.jpg)

![0db6c840a2e8ac149daf9f1451d4bff54138b94db26dc85ed75f4db9281f30b2.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/0db6c840a2e8ac149daf9f1451d4bff54138b94db26dc85ed75f4db9281f30b2.jpg)

![4b81e13964e0ed43cfc0777647c7d179e1d11a43b8a5f9aa1ecd45b076e34eaa.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/4b81e13964e0ed43cfc0777647c7d179e1d11a43b8a5f9aa1ecd45b076e34eaa.jpg)

![959cf84c90e7108911057ad55867e4987478c0b21f61f00e1e777d01be47c429.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/959cf84c90e7108911057ad55867e4987478c0b21f61f00e1e777d01be47c429.jpg)

![bf54a154d6cbfebe40c40f0d65987021d00d0daf749706d6ad234c105465c11c.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/bf54a154d6cbfebe40c40f0d65987021d00d0daf749706d6ad234c105465c11c.jpg)

![d436b9b59c048e7c8058c3927427b94c56b61eafe2344ac6d8fb2e2846ea52be.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/d436b9b59c048e7c8058c3927427b94c56b61eafe2344ac6d8fb2e2846ea52be.jpg)

![db89c8dbb775ce5accda36f8ade48592d1ba7a538b6ce414da4bc3135b56f91c.jpg](../icml_results/473_PhotoAgent_%20Exploratory%20Visual%20Aesthetic%20Planning%20with%20Large%20Vision%20Models/tables/db89c8dbb775ce5accda36f8ade48592d1ba7a538b6ce414da4bc3135b56f91c.jpg)

<a id="paper-0465"></a>

## Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions


### Images

![02d78135268cec9736143825fe605809ea830cf79ba5f80f5c48998a88d60416.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/02d78135268cec9736143825fe605809ea830cf79ba5f80f5c48998a88d60416.jpg)

![0db460a75204f3ad812332a884d9c95c981cfb83e37bada0373a67e6af2b6932.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/0db460a75204f3ad812332a884d9c95c981cfb83e37bada0373a67e6af2b6932.jpg)

![0e0a96a95b19381ab4c83d4791de8a2fb90701d34df042c4300317090fe55af4.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/0e0a96a95b19381ab4c83d4791de8a2fb90701d34df042c4300317090fe55af4.jpg)

![0fbae5bf67fdb53a41da5b7709151210b7a1dd0dc5c5547edab7be2e90eefaf4.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/0fbae5bf67fdb53a41da5b7709151210b7a1dd0dc5c5547edab7be2e90eefaf4.jpg)

![10fc9bcce0887b8f85364288de98d010c3255d5c64e47ac4ed19966ba66d3182.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/10fc9bcce0887b8f85364288de98d010c3255d5c64e47ac4ed19966ba66d3182.jpg)

![1b515ca7998e60a9e89a5535de8211ad14872245a32536f3203e1d8e67283ad0.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/1b515ca7998e60a9e89a5535de8211ad14872245a32536f3203e1d8e67283ad0.jpg)

![1d1b71a51cce40f7591693d095c28425a5de1a0544eab7accee09f24c3a9c727.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/1d1b71a51cce40f7591693d095c28425a5de1a0544eab7accee09f24c3a9c727.jpg)

![27a9c367c477c1aafbd189fd712836a73836a7e566c7050deeb8fb2c496dcc7e.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/27a9c367c477c1aafbd189fd712836a73836a7e566c7050deeb8fb2c496dcc7e.jpg)

![3fb0f99d8ed811546eca7450f5245fba32046942bcb9e331e6aba6c4bdbd9c0f.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/3fb0f99d8ed811546eca7450f5245fba32046942bcb9e331e6aba6c4bdbd9c0f.jpg)

![4f1c58c0bea688cd26ffd76c92e6168ef2a2cb0d6b7f7675213f239d268e8c1b.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/4f1c58c0bea688cd26ffd76c92e6168ef2a2cb0d6b7f7675213f239d268e8c1b.jpg)

![5460d37e2a507addd9efd0bd86ca5a21fc08cf1ca76509bc5093af44d12fbca4.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/5460d37e2a507addd9efd0bd86ca5a21fc08cf1ca76509bc5093af44d12fbca4.jpg)

![5b37b9268e02c6a9e2e72665473bf0e05fbc903a0b745f0b56fa83d8075ba001.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/5b37b9268e02c6a9e2e72665473bf0e05fbc903a0b745f0b56fa83d8075ba001.jpg)

![6335d630d39a94d7031b8f8b4dfbebb91d3466fb8b22bf595d46899077e9dc98.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/6335d630d39a94d7031b8f8b4dfbebb91d3466fb8b22bf595d46899077e9dc98.jpg)

![6add3e0222e6069c6f1589f28d71500cfaa4940050e46095114dc000213504a6.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/6add3e0222e6069c6f1589f28d71500cfaa4940050e46095114dc000213504a6.jpg)

![757ea7958aca7a6b2a41051453067155b75986f164a2636d2ad9749f8d728e67.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/757ea7958aca7a6b2a41051453067155b75986f164a2636d2ad9749f8d728e67.jpg)

![81a6111331d681c2dd0554ce6393f9e4558c711abe448c37d90d15fa6a52a136.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/81a6111331d681c2dd0554ce6393f9e4558c711abe448c37d90d15fa6a52a136.jpg)

![877fa4bf93120ee917b0ac1d9a5774f685399013029d7e84ebab4713e6a8a9f2.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/877fa4bf93120ee917b0ac1d9a5774f685399013029d7e84ebab4713e6a8a9f2.jpg)

![961a4322b85ad91c1cb461a993553f08c384376dd60ee256fe148975461200e3.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/961a4322b85ad91c1cb461a993553f08c384376dd60ee256fe148975461200e3.jpg)

![bb3765e3846e698961e6992bcedc65d66dfcb06f4f6ba4f9fbbc9052be4bc284.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/bb3765e3846e698961e6992bcedc65d66dfcb06f4f6ba4f9fbbc9052be4bc284.jpg)

![bfaeb0ba5934ef5fc646d5f3d835140c127ccb7dd7c6d0dcd56a7bb0b1a47a90.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/bfaeb0ba5934ef5fc646d5f3d835140c127ccb7dd7c6d0dcd56a7bb0b1a47a90.jpg)

![c438ea73c457dd2144cd28579c1f3a6706e6d78e2d4706b123dff19889da8798.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/c438ea73c457dd2144cd28579c1f3a6706e6d78e2d4706b123dff19889da8798.jpg)

![de5e7cd194ba5f69d4f885752c622fcf94475da5e3f464b8425569db409fd65c.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/de5e7cd194ba5f69d4f885752c622fcf94475da5e3f464b8425569db409fd65c.jpg)

![e938b7b1a211e3d1d3bf84c60aaa546856f0655586803e0f502c5848ad5ecfb3.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/e938b7b1a211e3d1d3bf84c60aaa546856f0655586803e0f502c5848ad5ecfb3.jpg)

![ecb1bf2ff2daf95fdcc9f6591a5d227862a77880c09aa9fb3a21e6b305a6af5b.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/ecb1bf2ff2daf95fdcc9f6591a5d227862a77880c09aa9fb3a21e6b305a6af5b.jpg)

![ed072cd281547bbec10ed489bc8f094f366caecc4efb7862af4047549bd68415.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/images/ed072cd281547bbec10ed489bc8f094f366caecc4efb7862af4047549bd68415.jpg)

### Tables

![361ef2509cf9d9e7b3c12024a24815bdf5470f772d15e0d4732a4987ee7a75bf.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/tables/361ef2509cf9d9e7b3c12024a24815bdf5470f772d15e0d4732a4987ee7a75bf.jpg)

![55236a22096142cca280a80f92d31e4fe5d26b433ab67a2524a5f6caf1c6cbee.jpg](../icml_results/474_Gradient%20Flow%20Through%20Diagram%20Expansions_%20Learning%20Regimes%20and%20Explicit%20Solutions/tables/55236a22096142cca280a80f92d31e4fe5d26b433ab67a2524a5f6caf1c6cbee.jpg)

<a id="paper-0466"></a>

## FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU


### Images

![0f9a171645d443b71f8162cb43a42168fd750347d1fe21eedfc0d8c5f6231add.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/0f9a171645d443b71f8162cb43a42168fd750347d1fe21eedfc0d8c5f6231add.jpg)

![4a2969d5a544619e48a9852e5bf393fb47cbbbbd11a3f4f6e62044cc471c1f97.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/4a2969d5a544619e48a9852e5bf393fb47cbbbbd11a3f4f6e62044cc471c1f97.jpg)

![5889b4e28d28ce3157b69bd22c34b262b02abf0b4d1f2dfb2ba22298a51dcc23.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/5889b4e28d28ce3157b69bd22c34b262b02abf0b4d1f2dfb2ba22298a51dcc23.jpg)

![65291bcd9454092d8928a499fca16d44c8f8c94e3823a20016019118a1d85f64.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/65291bcd9454092d8928a499fca16d44c8f8c94e3823a20016019118a1d85f64.jpg)

![7b60bed77bc85d8e4b0aff51b3ce52a9293e1c5fc38c39a6c355c349baf3dd69.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/7b60bed77bc85d8e4b0aff51b3ce52a9293e1c5fc38c39a6c355c349baf3dd69.jpg)

![87417dc221fa443d5ff3c70874da355ce068eb39636984297f2d40f56bf93600.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/87417dc221fa443d5ff3c70874da355ce068eb39636984297f2d40f56bf93600.jpg)

![d3cea28e2c2e22d6369fd2a757c002d6f81e75c594f90caddfa2312770c7f30a.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/d3cea28e2c2e22d6369fd2a757c002d6f81e75c594f90caddfa2312770c7f30a.jpg)

![f5c33ad554cf4859763542cf087c34e546440742e43108eb52f5354585ae8742.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/images/f5c33ad554cf4859763542cf087c34e546440742e43108eb52f5354585ae8742.jpg)

### Tables

![05b1f8d868c8e7cdb8cafc9e241d5fb77ffad4d54308cdd8549fe7109526b15c.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/05b1f8d868c8e7cdb8cafc9e241d5fb77ffad4d54308cdd8549fe7109526b15c.jpg)

![0f344b62f927a4095a7cf31e0abec8e884fc75351faaa09b86e1e841a453b9c8.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/0f344b62f927a4095a7cf31e0abec8e884fc75351faaa09b86e1e841a453b9c8.jpg)

![10a478394e215a12e8af707b4d8b1335ca248723775389d35cebc42fc992548d.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/10a478394e215a12e8af707b4d8b1335ca248723775389d35cebc42fc992548d.jpg)

![14f0c9c399a8dccd8686e4714fa8de50e950892b337cc86fb6891ec7772058a0.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/14f0c9c399a8dccd8686e4714fa8de50e950892b337cc86fb6891ec7772058a0.jpg)

![1fe842a4a416d1e9598929632b67f9586b0d3f1a406ee2cba44a50424aff7925.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/1fe842a4a416d1e9598929632b67f9586b0d3f1a406ee2cba44a50424aff7925.jpg)

![276655a8acb3e69d25b7d44a5689e66d3bd0d7cbbc5225e9de4d77bd1ac5152a.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/276655a8acb3e69d25b7d44a5689e66d3bd0d7cbbc5225e9de4d77bd1ac5152a.jpg)

![28e83075e4d66f90e3e16d7c981b7bebee9dff23df2b3ce7b75521204a0c64f7.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/28e83075e4d66f90e3e16d7c981b7bebee9dff23df2b3ce7b75521204a0c64f7.jpg)

![298b1b7496edcd9a1a3aed323f395556ab61fbb6f80ec00e3cc27553efe5a901.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/298b1b7496edcd9a1a3aed323f395556ab61fbb6f80ec00e3cc27553efe5a901.jpg)

![3666ed22a13653681f2f437d9623028f1aea5aef8ec647f3e5f46983cc964e61.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/3666ed22a13653681f2f437d9623028f1aea5aef8ec647f3e5f46983cc964e61.jpg)

![3a1f9375a431658a3a96218fe7a4b385d12dd94b696eade9da18436b6973820d.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/3a1f9375a431658a3a96218fe7a4b385d12dd94b696eade9da18436b6973820d.jpg)

![83121d3e30f498d46673b1a9a1aafe33689a200a50cbcbda893d2aa7d39890ee.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/83121d3e30f498d46673b1a9a1aafe33689a200a50cbcbda893d2aa7d39890ee.jpg)

![942510d8df36f2f5184b7edd5994b2d5ffe211ab75578cff12e81466d77be9c3.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/942510d8df36f2f5184b7edd5994b2d5ffe211ab75578cff12e81466d77be9c3.jpg)

![98475a1fa17568c927a6de3c2637d20428130f1c30ff534bea7e74e2b209c081.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/98475a1fa17568c927a6de3c2637d20428130f1c30ff534bea7e74e2b209c081.jpg)

![a42ceaa5041f86119df2b375adcfdaebfd564626dce0c16c726514d3fec7f62e.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/a42ceaa5041f86119df2b375adcfdaebfd564626dce0c16c726514d3fec7f62e.jpg)

![c528462f7f4835ae0e213b9b2bbe2ad5689845c19467622526e48377b20900ce.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/c528462f7f4835ae0e213b9b2bbe2ad5689845c19467622526e48377b20900ce.jpg)

![ca3c38c29c8f6921f7c755a4dd76d49b7e456dab73d3397c5fee00421b093a4e.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/ca3c38c29c8f6921f7c755a4dd76d49b7e456dab73d3397c5fee00421b093a4e.jpg)

![cd6a476987ef747c337cf3df1fc0dcfe3c819b0c573508dbc937cdf5a7c2198a.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/cd6a476987ef747c337cf3df1fc0dcfe3c819b0c573508dbc937cdf5a7c2198a.jpg)

![cdf25907676c5b9087765f64f1811202fea5f259bd4878c56df6a88a45fb4191.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/cdf25907676c5b9087765f64f1811202fea5f259bd4878c56df6a88a45fb4191.jpg)

![d16736adf5c0219e29e8435e42f7e70663b8d238f250b28a9b0c4354df7098f6.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/d16736adf5c0219e29e8435e42f7e70663b8d238f250b28a9b0c4354df7098f6.jpg)

![dc0238b4623403be58d28ff5c6ac6ba4dbc5ebdc51cb9f00bd78bc51d598646f.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/dc0238b4623403be58d28ff5c6ac6ba4dbc5ebdc51cb9f00bd78bc51d598646f.jpg)

![e37a96a21a8b51a026cda05e3508d7f0edaa358d05493610a7496dd8a56ec4ea.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/e37a96a21a8b51a026cda05e3508d7f0edaa358d05493610a7496dd8a56ec4ea.jpg)

![e92055e5624062efe21ec0decf823316c3e3b7e141a01b93c89d2f431b6fb005.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/e92055e5624062efe21ec0decf823316c3e3b7e141a01b93c89d2f431b6fb005.jpg)

![ecf586af6b213168ae670af226a42fa4342ffbeb8053b3959d8beef9b96d4ed8.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/ecf586af6b213168ae670af226a42fa4342ffbeb8053b3959d8beef9b96d4ed8.jpg)

![efbdd0decde61f3363735ae1f71511e69629bcb4d35b3ee63a111c238f1ce719.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/efbdd0decde61f3363735ae1f71511e69629bcb4d35b3ee63a111c238f1ce719.jpg)

![fcd836a8a89c7280949050c5327b80a82f1b198e23772786e5fcc0f6ec2b8760.jpg](../icml_results/475_FlashSinkhorn_%20IO-Aware%20Entropic%20Optimal%20Transport%20on%20GPU/tables/fcd836a8a89c7280949050c5327b80a82f1b198e23772786e5fcc0f6ec2b8760.jpg)

<a id="paper-0467"></a>

## Securing Multimodal AI through Internal Information Decomposition


### Images

![1bc67befb882b0db12795b675754ddcdb3766aefe71b4aa3f960607105c7d822.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/images/1bc67befb882b0db12795b675754ddcdb3766aefe71b4aa3f960607105c7d822.jpg)

![37cd7eb70c90256d1f534ac318072ddeb892663e40c0354517e75628210cb037.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/images/37cd7eb70c90256d1f534ac318072ddeb892663e40c0354517e75628210cb037.jpg)

![4c224f4a838cc289964161271704667e2adf60621dc3c3b521309d1f61e3c9f8.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/images/4c224f4a838cc289964161271704667e2adf60621dc3c3b521309d1f61e3c9f8.jpg)

![66216e698c98f4fb07a446327ef64bd9d7f1e900986da975dc860e968baa2bf6.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/images/66216e698c98f4fb07a446327ef64bd9d7f1e900986da975dc860e968baa2bf6.jpg)

![b079989af9234d74d92d02ee9028997f09e9fd3c40c37b8ba1ddece71ecb1873.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/images/b079989af9234d74d92d02ee9028997f09e9fd3c40c37b8ba1ddece71ecb1873.jpg)

### Tables

![1a3be2ea4f6ede8bbdfe1a0b7f235ec01f2b8ca365a559efa6a28fc00bba23f9.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/1a3be2ea4f6ede8bbdfe1a0b7f235ec01f2b8ca365a559efa6a28fc00bba23f9.jpg)

![28d3deed7d936abe0c84dcd90131bc8a9f72e2cc6e1d66d3a36d08ad599425cf.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/28d3deed7d936abe0c84dcd90131bc8a9f72e2cc6e1d66d3a36d08ad599425cf.jpg)

![454853818d88b7af2e047851b859e9f0e8729a23266116804fcb649696be0a64.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/454853818d88b7af2e047851b859e9f0e8729a23266116804fcb649696be0a64.jpg)

![57c96498d0fb896f4764c5f6a163af6492d59c96d1f73fe679b40c80c940c34c.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/57c96498d0fb896f4764c5f6a163af6492d59c96d1f73fe679b40c80c940c34c.jpg)

![6217a8ce7dd189be843c216ad759c3ccf2dd9380b1903e7a8114be0d972d9a56.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/6217a8ce7dd189be843c216ad759c3ccf2dd9380b1903e7a8114be0d972d9a56.jpg)

![8c57bbe9ce8b33627498d56c3b4a429dd4e2a1a9a0f43ee3712bced78a04b301.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/8c57bbe9ce8b33627498d56c3b4a429dd4e2a1a9a0f43ee3712bced78a04b301.jpg)

![9ae1fb7658f167faa66b035fc0fe193d184793da4d22521bd547575a6f9549d4.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/9ae1fb7658f167faa66b035fc0fe193d184793da4d22521bd547575a6f9549d4.jpg)

![aa4c4ef487331852b47de890e0832cb7326612966b5e191c64217d518a9b2754.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/aa4c4ef487331852b47de890e0832cb7326612966b5e191c64217d518a9b2754.jpg)

![aaa4effbfb1b457f6e43b3ed3e3c39aa9d38703d621b4aaad3a403f0d2aa6877.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/aaa4effbfb1b457f6e43b3ed3e3c39aa9d38703d621b4aaad3a403f0d2aa6877.jpg)

![ada8aec8ebe4c6e0a41cdf5bd15493c257d82133cf4194f343aa857bbc299a85.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/ada8aec8ebe4c6e0a41cdf5bd15493c257d82133cf4194f343aa857bbc299a85.jpg)

![b0cb736ba1964739aacc65292243d3772c349150ba68d86ead920a88355fcb9d.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/b0cb736ba1964739aacc65292243d3772c349150ba68d86ead920a88355fcb9d.jpg)

![bbddf1641751b1334a1c52957b3f62f0d71e2446fa813b89e875259fff85dfc9.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/bbddf1641751b1334a1c52957b3f62f0d71e2446fa813b89e875259fff85dfc9.jpg)

![c5f1472a7bb9f7c29b00008579441f507fcb256947139e6ecd8865727e495a40.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/c5f1472a7bb9f7c29b00008579441f507fcb256947139e6ecd8865727e495a40.jpg)

![d0ca7077eb61a69cbcf99f62dc575928ffd5b884a03d970b06b40f4aa4e0570b.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/d0ca7077eb61a69cbcf99f62dc575928ffd5b884a03d970b06b40f4aa4e0570b.jpg)

![e520efd986fc03c323a40a52d4460991f95ccae555459ccb982482338e31ec95.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/e520efd986fc03c323a40a52d4460991f95ccae555459ccb982482338e31ec95.jpg)

![ea67c3e4f7eb68e3e2de375d6cdd8218b889bb0048254eea357a7b086f27810b.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/ea67c3e4f7eb68e3e2de375d6cdd8218b889bb0048254eea357a7b086f27810b.jpg)

![fdf72bfb2cafc29e1699fdaf63734175c55ca0799dcc8d0e07c08a016031d2e8.jpg](../icml_results/476_Securing%20Multimodal%20AI%20through%20Internal%20Information%20Decomposition/tables/fdf72bfb2cafc29e1699fdaf63734175c55ca0799dcc8d0e07c08a016031d2e8.jpg)

<a id="paper-0468"></a>

## PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion


### Images

![0897ce9e2e27309daae1545a9ae42f6b3231d0ba79ccb3f4a999f46785f2d9e9.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/0897ce9e2e27309daae1545a9ae42f6b3231d0ba79ccb3f4a999f46785f2d9e9.jpg)

![09440af83278b35130054b11850e4f18f05717d951a8e394cc471a1baff36a54.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/09440af83278b35130054b11850e4f18f05717d951a8e394cc471a1baff36a54.jpg)

![0cbc65561d151b8f4cb5d08a6fe8976bab9cc39caf49ca7ad11cfcef926b8e92.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/0cbc65561d151b8f4cb5d08a6fe8976bab9cc39caf49ca7ad11cfcef926b8e92.jpg)

![0dc0b62b5cc17fbba79f01cb279a6edd797d8db25d9f98adc6e26e3387647191.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/0dc0b62b5cc17fbba79f01cb279a6edd797d8db25d9f98adc6e26e3387647191.jpg)

![3fa7934601eeba9e19cc0c2ecc554d191c0a7d7001066cb518f5efb0be0c01c4.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/3fa7934601eeba9e19cc0c2ecc554d191c0a7d7001066cb518f5efb0be0c01c4.jpg)

![47459c0b016d036646f5bd5184f1c1395468f5d67f5fce78f439105fff59bca0.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/47459c0b016d036646f5bd5184f1c1395468f5d67f5fce78f439105fff59bca0.jpg)

![6c7846ce82cf51183f9fc6eaacf747383e2f55cb4a2d89ca19bf5c7b54e544d6.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/6c7846ce82cf51183f9fc6eaacf747383e2f55cb4a2d89ca19bf5c7b54e544d6.jpg)

![79f8047ad8274ae524dc495e0d7afefbf6817d28d3616b3e47d77c2bfc4b3f7c.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/79f8047ad8274ae524dc495e0d7afefbf6817d28d3616b3e47d77c2bfc4b3f7c.jpg)

![9f25ce488c6d774b70001896a3a51a9731a4742455eec3a6cea7fd88ccad8e08.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/9f25ce488c6d774b70001896a3a51a9731a4742455eec3a6cea7fd88ccad8e08.jpg)

![cf47d0dc2849684205bf6b468e9e4c26b197e08941034cf4325e7965e12ea2aa.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/images/cf47d0dc2849684205bf6b468e9e4c26b197e08941034cf4325e7965e12ea2aa.jpg)

### Tables

![006fe887038596b8ed9db3e41b65149260dfa311ba013062914687f347ce03fd.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/tables/006fe887038596b8ed9db3e41b65149260dfa311ba013062914687f347ce03fd.jpg)

![41b0bb59e4825a5acf042771038a5fd4381e5b1c952f2899a562991fc88d2c81.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/tables/41b0bb59e4825a5acf042771038a5fd4381e5b1c952f2899a562991fc88d2c81.jpg)

![4efa07637c93f52519571ed51d65d8b9d9d5e7f87b77ae14f19e3511e8b16a85.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/tables/4efa07637c93f52519571ed51d65d8b9d9d5e7f87b77ae14f19e3511e8b16a85.jpg)

![7f346f84a777c3d49c521c184bce0c8d7422fa4ffb32c27308ef57566cc5d7db.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/tables/7f346f84a777c3d49c521c184bce0c8d7422fa4ffb32c27308ef57566cc5d7db.jpg)

![8ebe4562b120238d30e0ba7ec351133818b963340e4decfa7b02f4ef3029c19b.jpg](../icml_results/477_PanoWorld-X_%20Generating%20Explorable%20Panoramic%20Worlds%20via%20Sphere-Aware%20Video%20Diffusion/tables/8ebe4562b120238d30e0ba7ec351133818b963340e4decfa7b02f4ef3029c19b.jpg)

<a id="paper-0469"></a>

## Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models


### Images

![04b71c764b43f8705f50777ba1249a4ae782d2a8f674092a88301ff053e8ccba.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/04b71c764b43f8705f50777ba1249a4ae782d2a8f674092a88301ff053e8ccba.jpg)

![0532d000dc9f374c7cc37d35b5f2135195fa70383f68e6f2fb9f7447451f6795.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/0532d000dc9f374c7cc37d35b5f2135195fa70383f68e6f2fb9f7447451f6795.jpg)

![2348beba7cb5ae0f2711303063b877c3b3fa354dc54c97abc9f0fee92efcb7b3.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/2348beba7cb5ae0f2711303063b877c3b3fa354dc54c97abc9f0fee92efcb7b3.jpg)

![262fe6ed79f144e77db91f474b4854c8bd9c66d1400c25149a00ccc16c7fd9f3.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/262fe6ed79f144e77db91f474b4854c8bd9c66d1400c25149a00ccc16c7fd9f3.jpg)

![26f3f767ffbf7ffc3198185cc75c10925dc7c59a3b2e0c64283265a0b7361977.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/26f3f767ffbf7ffc3198185cc75c10925dc7c59a3b2e0c64283265a0b7361977.jpg)

![2e687e2ff5b9ce8f02186be107da6b71eb426d750cea548ea0eff94535461689.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/2e687e2ff5b9ce8f02186be107da6b71eb426d750cea548ea0eff94535461689.jpg)

![32cb86a0a7fea1de88c5b862494667f7099bf233025e137a5daa21777ac09673.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/32cb86a0a7fea1de88c5b862494667f7099bf233025e137a5daa21777ac09673.jpg)

![34f4ac55c8ac457ee40584f4f36c36d6c58ed1d0dccbf18538683433de9a71c1.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/34f4ac55c8ac457ee40584f4f36c36d6c58ed1d0dccbf18538683433de9a71c1.jpg)

![3caaa9643e6728d85e2e80c9ba67e5ee4f604f76b250e028f54a95c6999fc7a5.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/3caaa9643e6728d85e2e80c9ba67e5ee4f604f76b250e028f54a95c6999fc7a5.jpg)

![496bc3a44d734b30ba5a43e156a9f52ce3b2cb270ba69621cac5119d1c9b208c.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/496bc3a44d734b30ba5a43e156a9f52ce3b2cb270ba69621cac5119d1c9b208c.jpg)

![7979324687264ed383cf71033622d0982b0f9e53f4746b4b920a9a4a9e7a24cf.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/7979324687264ed383cf71033622d0982b0f9e53f4746b4b920a9a4a9e7a24cf.jpg)

![ab4502c6d152dadb836f68444947b6a33f70dac743b44ce1d2092670f3ac329b.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/ab4502c6d152dadb836f68444947b6a33f70dac743b44ce1d2092670f3ac329b.jpg)

![afc2a326fdaf8ce1947d40650a2560e2970a80fad481c7729cea056ed4d24531.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/afc2a326fdaf8ce1947d40650a2560e2970a80fad481c7729cea056ed4d24531.jpg)

![c050031ebe2360b6676c13b9e1b1fab902656aa46f11d025c83ce5c0607e6a62.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/c050031ebe2360b6676c13b9e1b1fab902656aa46f11d025c83ce5c0607e6a62.jpg)

![d9c366bbf61edcac1b0d5c4eda66225ba2ae86b825216c0d4247cf759deabffb.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/d9c366bbf61edcac1b0d5c4eda66225ba2ae86b825216c0d4247cf759deabffb.jpg)

![e69441f021c277cb103e234bc1c7e429b0a2bee9d5aa271944db22a6d5d0879f.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/e69441f021c277cb103e234bc1c7e429b0a2bee9d5aa271944db22a6d5d0879f.jpg)

![eaad5b9458f38239d178868ae996b4ba67c9938407476bd70a15a6208a1003e1.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/eaad5b9458f38239d178868ae996b4ba67c9938407476bd70a15a6208a1003e1.jpg)

![f21a6d8f979a6081d94f3b6f8c79b03fb882538ddfa15bb38be8f345e0e2cee5.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/images/f21a6d8f979a6081d94f3b6f8c79b03fb882538ddfa15bb38be8f345e0e2cee5.jpg)

### Tables

![0b3d8d6c55c06acc1971ddde9a5322e4423f7c2b45e411e1032b3d8bcf360ba9.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/0b3d8d6c55c06acc1971ddde9a5322e4423f7c2b45e411e1032b3d8bcf360ba9.jpg)

![40b8ffe5b896aaa2b8ab50b6ee254a0198615dfe2493edb2b171f8215160c630.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/40b8ffe5b896aaa2b8ab50b6ee254a0198615dfe2493edb2b171f8215160c630.jpg)

![50aa106f46c6d1af512af8197ab3934ae76281ee8dfe28652245ac3dfb48ab65.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/50aa106f46c6d1af512af8197ab3934ae76281ee8dfe28652245ac3dfb48ab65.jpg)

![9a398cfa6bede2c662638a783304e2cffeb14ba62b9fd2fe5c6e6db9ea3a9e01.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/9a398cfa6bede2c662638a783304e2cffeb14ba62b9fd2fe5c6e6db9ea3a9e01.jpg)

![9f14b73c017601a859cad7e3b7e46cd2296e50e6078ea7918664678617bb9a56.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/9f14b73c017601a859cad7e3b7e46cd2296e50e6078ea7918664678617bb9a56.jpg)

![a08de95bf4b6603210731b99b6d07e793620befd5ef08261dd155163bc608c40.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/a08de95bf4b6603210731b99b6d07e793620befd5ef08261dd155163bc608c40.jpg)

![bd230b3cd0b06132c99f7136cfe7a374b20e3feae06e5e9ede754944ccda480b.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/bd230b3cd0b06132c99f7136cfe7a374b20e3feae06e5e9ede754944ccda480b.jpg)

![c77712693c0071ec89520f667be94f0aa41ffde4424ae9505224165fa4a3f760.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/c77712693c0071ec89520f667be94f0aa41ffde4424ae9505224165fa4a3f760.jpg)

![ca02a8f33f61351cf44d29a6087abd01802ea797e446663fb908ffb5374029b5.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/ca02a8f33f61351cf44d29a6087abd01802ea797e446663fb908ffb5374029b5.jpg)

![e4bdc3535d45a0b121bf610c5115df95e083d5cf93926e12d15adea19abb16c8.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/e4bdc3535d45a0b121bf610c5115df95e083d5cf93926e12d15adea19abb16c8.jpg)

![e5afcfa9a7ea3a85b411d41d781c4fd8ae5864e5a95edd1895ff0169d4994a00.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/e5afcfa9a7ea3a85b411d41d781c4fd8ae5864e5a95edd1895ff0169d4994a00.jpg)

![ec1904d7a247c7ec25a067385968f122b8a8dac8f2bfdb9528a6229167784ac6.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/ec1904d7a247c7ec25a067385968f122b8a8dac8f2bfdb9528a6229167784ac6.jpg)

![f5e3d70030681e5a394ea9f999ed1a822268f0c96edaf050cb4dca7ffc109312.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/f5e3d70030681e5a394ea9f999ed1a822268f0c96edaf050cb4dca7ffc109312.jpg)

![ff604f850ae371425a07e0755a172b8cb618fb3bbf4aa6897a7ee1f4be0f99bd.jpg](../icml_results/478_Robust%20Harmful%20Features%20Under%20Jailbreak%20Attacks_%20Mechanistic%20Evidence%20from%20Attention%20Head%20Specializa/tables/ff604f850ae371425a07e0755a172b8cb618fb3bbf4aa6897a7ee1f4be0f99bd.jpg)

<a id="paper-0470"></a>

## Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding


### Images

![08e74a0932251bb12f98c076019fc03a108c13abfb5e564427301ebc2ac2d720.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/08e74a0932251bb12f98c076019fc03a108c13abfb5e564427301ebc2ac2d720.jpg)

![0bbe1f9827be00a1d64eecdae456356e0d9c5f93d8824ff71d638816c6bd6bfd.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/0bbe1f9827be00a1d64eecdae456356e0d9c5f93d8824ff71d638816c6bd6bfd.jpg)

![0fee3d91ab764b8ed68674c65a22d60ba0c1f9e0827f2236003870cb6a7e5f69.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/0fee3d91ab764b8ed68674c65a22d60ba0c1f9e0827f2236003870cb6a7e5f69.jpg)

![160a7f7d4d3d724f3222ffa9c8a5bafb8d256a015ce676acdbbb7eb8b978d2fa.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/160a7f7d4d3d724f3222ffa9c8a5bafb8d256a015ce676acdbbb7eb8b978d2fa.jpg)

![1ac17a1c5922d29c080b17a984d7ad7aa65f04edeca0119324eecaf99dec5053.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/1ac17a1c5922d29c080b17a984d7ad7aa65f04edeca0119324eecaf99dec5053.jpg)

![1ac265960b86f13f54e7c46e948441a7bf7bed38d8fad421bd9e44907ad213e6.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/1ac265960b86f13f54e7c46e948441a7bf7bed38d8fad421bd9e44907ad213e6.jpg)

![24703f72b83d0c02433c7c478e789c1cb3df7473932827fc1760b467d1dec3e7.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/24703f72b83d0c02433c7c478e789c1cb3df7473932827fc1760b467d1dec3e7.jpg)

![360c6b9438b2736d39fddf4cced1d5d1c21c344f26498bd48da7223b2721402f.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/360c6b9438b2736d39fddf4cced1d5d1c21c344f26498bd48da7223b2721402f.jpg)

![4f7d6d637d616e34722101933f0271b7c72c5bc769c8e3970dfdd808164e6192.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/4f7d6d637d616e34722101933f0271b7c72c5bc769c8e3970dfdd808164e6192.jpg)

![616ee2a2ee61cef311ee465666cad681128f06c64bbecd52c1fe337c1d932d52.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/616ee2a2ee61cef311ee465666cad681128f06c64bbecd52c1fe337c1d932d52.jpg)

![6b71f742249f56f9493bab6075d423183148c680b55d1e854f1a1df7a6caac8b.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/6b71f742249f56f9493bab6075d423183148c680b55d1e854f1a1df7a6caac8b.jpg)

![94854f1740c1e986c5539144546f5abd194051d82147d11dd5cf638e282f1834.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/94854f1740c1e986c5539144546f5abd194051d82147d11dd5cf638e282f1834.jpg)

![a688a117d3aac55d81ce3130a0e4b72587acd0813ed500270f878aa14e7eda2a.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/a688a117d3aac55d81ce3130a0e4b72587acd0813ed500270f878aa14e7eda2a.jpg)

![b8ae8e3ab73faaff6e4625167ae23818bf4344c7b1466883e896222935cf7aa4.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/b8ae8e3ab73faaff6e4625167ae23818bf4344c7b1466883e896222935cf7aa4.jpg)

![bf32068cb1dd242cf28d58c0bb8a194f9309a4abe2284f03c6c3b1fa91cfbd23.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/bf32068cb1dd242cf28d58c0bb8a194f9309a4abe2284f03c6c3b1fa91cfbd23.jpg)

![cbdfae398f2213255fd8023986d76edfb617c9b402f6e696d507927f2bd2b3c1.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/cbdfae398f2213255fd8023986d76edfb617c9b402f6e696d507927f2bd2b3c1.jpg)

![d74d87c71ecf001ae61815f92d6d6843c661f391d9c3b218ff13e7c0636bc9a3.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/d74d87c71ecf001ae61815f92d6d6843c661f391d9c3b218ff13e7c0636bc9a3.jpg)

![e3492d1a470bb46ab7f58c1f9d22df980ef895a39bd80212b355cdfd4cb57eba.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/e3492d1a470bb46ab7f58c1f9d22df980ef895a39bd80212b355cdfd4cb57eba.jpg)

![ff24367a52114c6e8e95662ca740263cce1f29df1a6b8de40f8625090ef13143.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/images/ff24367a52114c6e8e95662ca740263cce1f29df1a6b8de40f8625090ef13143.jpg)

### Tables

![0c5bdcb7512b09b0b5d291d47a1784f4aebc90bdad2413a2a4bb3e0da1c619d0.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/tables/0c5bdcb7512b09b0b5d291d47a1784f4aebc90bdad2413a2a4bb3e0da1c619d0.jpg)

![54ef776e2c553fcacbb121340720e4524f31c45f8dff67b0356135b534ce5f06.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/tables/54ef776e2c553fcacbb121340720e4524f31c45f8dff67b0356135b534ce5f06.jpg)

![5e65ffeed427d74b49fbac268b166f6e17073243bd3efadac1c49fba8c071a8b.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/tables/5e65ffeed427d74b49fbac268b166f6e17073243bd3efadac1c49fba8c071a8b.jpg)

![c8ca63a7503e68290989e9788b4fa738fca62112186517f73ef8b34265467740.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/tables/c8ca63a7503e68290989e9788b4fa738fca62112186517f73ef8b34265467740.jpg)

![f01ab808626d6b86921e0d932f1c97f55daa67715ed9e1fa19747741496876fc.jpg](../icml_results/479_Decomposed%20On-Policy%20Distillation%20for%20Vision-Language%20Reasoning_%20Steering%20Gradients%20for%20Visual%20Groun/tables/f01ab808626d6b86921e0d932f1c97f55daa67715ed9e1fa19747741496876fc.jpg)

<a id="paper-0471"></a>

## Latent Laplace Diffusion for Irregular Multivariate Time Series


### Images

![077150f8387769cd4ba67eb48ef129f836a8034ea9b479bbe9cf7095cbde00b6.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/077150f8387769cd4ba67eb48ef129f836a8034ea9b479bbe9cf7095cbde00b6.jpg)

![16746c67ff741e6399d1a42e36a5900d8c365365bbb6eebfda352a5e4ad8e405.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/16746c67ff741e6399d1a42e36a5900d8c365365bbb6eebfda352a5e4ad8e405.jpg)

![a325ad31c8d41e46dd4b10927da92a950d03e3f36fb88bd79e952e1daac2dd5a.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/a325ad31c8d41e46dd4b10927da92a950d03e3f36fb88bd79e952e1daac2dd5a.jpg)

![a879d6af65e935400f9fa900ef3f7c64ab82d6947e2d3b55dea6a6c04ab24062.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/a879d6af65e935400f9fa900ef3f7c64ab82d6947e2d3b55dea6a6c04ab24062.jpg)

![cf0003872aa00b310abf67dfbeb5fc781f7faa2eafa7e5ba0f827bd4b780c049.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/cf0003872aa00b310abf67dfbeb5fc781f7faa2eafa7e5ba0f827bd4b780c049.jpg)

![fd71e312530abbf4cc6cbc692bbacea5389453803191eaf2b6f161f177c974e3.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/images/fd71e312530abbf4cc6cbc692bbacea5389453803191eaf2b6f161f177c974e3.jpg)

### Tables

![0de15adddaa17316fde003cbf25f52efc21da742a0809e53668c719aff7c612a.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/0de15adddaa17316fde003cbf25f52efc21da742a0809e53668c719aff7c612a.jpg)

![15f95376ee4cd02b49154bd793ed94020875fed955300c7eb2b9d7004721818a.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/15f95376ee4cd02b49154bd793ed94020875fed955300c7eb2b9d7004721818a.jpg)

![4e88aa89c6e6ef653028b2b7224af1f0c2285006fa1a8c653afb4b61870ea216.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/4e88aa89c6e6ef653028b2b7224af1f0c2285006fa1a8c653afb4b61870ea216.jpg)

![5e6866df263abcd3c57cd25dbbe0931a2616d8c5d883baa043e9d7a32e6f11aa.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/5e6866df263abcd3c57cd25dbbe0931a2616d8c5d883baa043e9d7a32e6f11aa.jpg)

![703974415fb181d0ae58adc5a4cf512dd3ba174c050508708d626398966ff981.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/703974415fb181d0ae58adc5a4cf512dd3ba174c050508708d626398966ff981.jpg)

![8477e137f6d410650d4e3503a2251117bf6180ca7266919120845405335a834e.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/8477e137f6d410650d4e3503a2251117bf6180ca7266919120845405335a834e.jpg)

![af712b81a2d6deb725b515f6e82be52708b875758c6b0a23f031a3c4d834f3da.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/af712b81a2d6deb725b515f6e82be52708b875758c6b0a23f031a3c4d834f3da.jpg)

![af8a5e0601c5415f1675ce050ce1c33a988a0bc61f5d0fe73dce489898eac1cf.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/af8a5e0601c5415f1675ce050ce1c33a988a0bc61f5d0fe73dce489898eac1cf.jpg)

![c1cc91c9f5911e7b74d13a98f09128700d4355e97284bfbeb1255a0afe42602a.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/c1cc91c9f5911e7b74d13a98f09128700d4355e97284bfbeb1255a0afe42602a.jpg)

![d49b81f76971a72810e2a5e4ae6458a8ec5f5743b11d5c408537651b6fa3c47e.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/d49b81f76971a72810e2a5e4ae6458a8ec5f5743b11d5c408537651b6fa3c47e.jpg)

![f865c6b3252109bbc5b56c016042fad2b94fde48071b6a1aedf7890ebf3b8348.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/f865c6b3252109bbc5b56c016042fad2b94fde48071b6a1aedf7890ebf3b8348.jpg)

![f9cb93f3224085174ce359f01b3018e8c1103a573bb825b54ab1c0925bb8c8ec.jpg](../icml_results/480_Latent%20Laplace%20Diffusion%20for%20Irregular%20Multivariate%20Time%20Series/tables/f9cb93f3224085174ce359f01b3018e8c1103a573bb825b54ab1c0925bb8c8ec.jpg)

<a id="paper-0472"></a>

## Hierarchical Successor Representation for Robust Transfer


### Images

![0fdc5a9bfecfa868b6d822c1af116ef0b1360f229d2976da8bf511e7a97583fd.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/0fdc5a9bfecfa868b6d822c1af116ef0b1360f229d2976da8bf511e7a97583fd.jpg)

![27d3232e05b7b92a0ba25b1b77c84a8a456891609152881cd36937b83cc3b97a.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/27d3232e05b7b92a0ba25b1b77c84a8a456891609152881cd36937b83cc3b97a.jpg)

![2f74557eb5f4676c9a87ed9160258bf6b5b0b54ea41dc36a6157557859d352da.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/2f74557eb5f4676c9a87ed9160258bf6b5b0b54ea41dc36a6157557859d352da.jpg)

![3d77c1c48f25fcf4dce5af2def0fd80ef9d5faec3aeee61df87982ba7d3ba71d.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/3d77c1c48f25fcf4dce5af2def0fd80ef9d5faec3aeee61df87982ba7d3ba71d.jpg)

![440ac709ac96448316dcb871f477c7f22dfb0357d4fcd4cc8f79351401520215.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/440ac709ac96448316dcb871f477c7f22dfb0357d4fcd4cc8f79351401520215.jpg)

![5a23f2590076617082b9cf358f5beff17da32538f17115dae7bcc1e071f337e4.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/5a23f2590076617082b9cf358f5beff17da32538f17115dae7bcc1e071f337e4.jpg)

![816da0e76e76da1b281bf376855485de3a9fe5337b3dd9ae7fa74cea7cb71f81.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/816da0e76e76da1b281bf376855485de3a9fe5337b3dd9ae7fa74cea7cb71f81.jpg)

![8ec8de95725da5b1c70f9ccfaad6112e9237c888ef137bea7ded57f884150d49.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/8ec8de95725da5b1c70f9ccfaad6112e9237c888ef137bea7ded57f884150d49.jpg)

![a84e73ea1d4215780a7f846c7d443291c1dcf98515c18daf1217409492f0b815.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/a84e73ea1d4215780a7f846c7d443291c1dcf98515c18daf1217409492f0b815.jpg)

![d83a2b43c5b0439e96f0c626a394e2e9a997b7744b666d6c47cb145a0a9bcc98.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/d83a2b43c5b0439e96f0c626a394e2e9a997b7744b666d6c47cb145a0a9bcc98.jpg)

![f9f9769e238a972d91a443ec314768bbcd7a38ce13e11fec46281bebb0cb4c0a.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/f9f9769e238a972d91a443ec314768bbcd7a38ce13e11fec46281bebb0cb4c0a.jpg)

![fe4fe67b34fb73ea50dd3a5e1586a55cc6b5dae11a00fcec6ef11b4fb5a33340.jpg](../icml_results/481_Hierarchical%20Successor%20Representation%20for%20Robust%20Transfer/images/fe4fe67b34fb73ea50dd3a5e1586a55cc6b5dae11a00fcec6ef11b4fb5a33340.jpg)

<a id="paper-0473"></a>

## Geometric Flow Grounding: A Unified Manifold Decoupling Framework for Dynamics Discovery and Verification


### Images

![0fecdddcfed7e7870221d85b260a9377dcaa8a38de60c45beeb7fffa897d9c0a.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/0fecdddcfed7e7870221d85b260a9377dcaa8a38de60c45beeb7fffa897d9c0a.jpg)

![1105cda16781aa384a7e85403848c89deb8d1c875bc816928b203571df8843cd.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/1105cda16781aa384a7e85403848c89deb8d1c875bc816928b203571df8843cd.jpg)

![17681d1424d85d92fa7e764df0bf5d78eba93e0366f6680c8d5c29fac9e838a2.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/17681d1424d85d92fa7e764df0bf5d78eba93e0366f6680c8d5c29fac9e838a2.jpg)

![2c503ebd7155263743bceda5b2ae58616f14567311fe64a745376d9dd3178389.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/2c503ebd7155263743bceda5b2ae58616f14567311fe64a745376d9dd3178389.jpg)

![542e930eb1b04bfe450e6151d8aea4832e7fe178d372f0b49747a6bd5678e844.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/542e930eb1b04bfe450e6151d8aea4832e7fe178d372f0b49747a6bd5678e844.jpg)

![706d1411531e399eee83fd7f113327b105aeb168d3b7ac247af05318405fe0d8.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/706d1411531e399eee83fd7f113327b105aeb168d3b7ac247af05318405fe0d8.jpg)

![afcc4e949516622a5d9317321a8e4b01a2b7c1c3872ed51dc25b0a3ea5e222d0.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/afcc4e949516622a5d9317321a8e4b01a2b7c1c3872ed51dc25b0a3ea5e222d0.jpg)

![d83683708cc062d614d9d6e8ef1b4476ae66026ed122ed90e4a7e65a3f96ab6b.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/images/d83683708cc062d614d9d6e8ef1b4476ae66026ed122ed90e4a7e65a3f96ab6b.jpg)

### Tables

![0f4a3925e78f8ad0f38d15d1674985d0d9178f8faa0efc122eb0a4ba9568b515.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/0f4a3925e78f8ad0f38d15d1674985d0d9178f8faa0efc122eb0a4ba9568b515.jpg)

![186cd8d106d1fc3e8e5580a10e2cab840f8f81be651c018cc5a56a5a68e88919.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/186cd8d106d1fc3e8e5580a10e2cab840f8f81be651c018cc5a56a5a68e88919.jpg)

![21a1e7f2a6c23e1e80463cfd6d47c15843a8bedcce36cb558cb0f1627d97bbeb.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/21a1e7f2a6c23e1e80463cfd6d47c15843a8bedcce36cb558cb0f1627d97bbeb.jpg)

![292d6d35d23eb3c4ba909d8f1399bba85b4d5211b37cff9c1c836589a1294ec3.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/292d6d35d23eb3c4ba909d8f1399bba85b4d5211b37cff9c1c836589a1294ec3.jpg)

![2fe8a6aade366ab12cc434bae0bbdc88a4b212558deb7f7648cdd1ab05f951b3.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/2fe8a6aade366ab12cc434bae0bbdc88a4b212558deb7f7648cdd1ab05f951b3.jpg)

![44ecc3e396056cb91c44c80afa86d304d230525d41891fbb46f55726a0c20339.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/44ecc3e396056cb91c44c80afa86d304d230525d41891fbb46f55726a0c20339.jpg)

![4afec99af4b36578831935eb05d510eb8ab90c7b8a01477958dcf4c0626ff51e.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/4afec99af4b36578831935eb05d510eb8ab90c7b8a01477958dcf4c0626ff51e.jpg)

![5647b7fbc5884575bd65420d233fb75d8ad64caf435cdcf1a7e476cbf35c8013.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/5647b7fbc5884575bd65420d233fb75d8ad64caf435cdcf1a7e476cbf35c8013.jpg)

![927dd1dcbac865fe8fb68d1ee82ba4f80071074d068fc3411f9a229c39abe625.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/927dd1dcbac865fe8fb68d1ee82ba4f80071074d068fc3411f9a229c39abe625.jpg)

![c22b4f6baf862281cca166ac3fbfd049ff20fa36d9d3385233c790e446d42c23.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/c22b4f6baf862281cca166ac3fbfd049ff20fa36d9d3385233c790e446d42c23.jpg)

![c6b8f1a6d3c054933082fe72658a084319968ef37ca272241e19e49c5f5c770e.jpg](../icml_results/482_Geometric%20Flow%20Grounding_%20A%20Unified%20Manifold%20Decoupling%20Framework%20for%20Dynamics%20Discovery%20and%20Verific/tables/c6b8f1a6d3c054933082fe72658a084319968ef37ca272241e19e49c5f5c770e.jpg)

<a id="paper-0474"></a>

## Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates


### Images

![5a86ef19ef5f59380ebefb40ddb2b2cb994495c57861f7e2385e1a88f871813b.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/5a86ef19ef5f59380ebefb40ddb2b2cb994495c57861f7e2385e1a88f871813b.jpg)

![8de1e10ffdafdbc083f96aebe933d724dcea8e1c29eb7027bf9de4878c1fd884.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/8de1e10ffdafdbc083f96aebe933d724dcea8e1c29eb7027bf9de4878c1fd884.jpg)

![9616236341b707540dc7409f41308cc3ff343f4426d9016b202f931fde1e7d39.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/9616236341b707540dc7409f41308cc3ff343f4426d9016b202f931fde1e7d39.jpg)

![bfb7493754b3317ba86bd6abc0119e90f4587dead0445e97dccc7053aba3fe0f.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/bfb7493754b3317ba86bd6abc0119e90f4587dead0445e97dccc7053aba3fe0f.jpg)

![f6168010414d410746f82f2f5c44a1d16865b372a7322568eebca9fca6f9c91a.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/f6168010414d410746f82f2f5c44a1d16865b372a7322568eebca9fca6f9c91a.jpg)

![fecec0f646c1cfaabc2fc2b4cd94d4ae3daa45878ad3d19d309fb76a5c92eac8.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/images/fecec0f646c1cfaabc2fc2b4cd94d4ae3daa45878ad3d19d309fb76a5c92eac8.jpg)

### Tables

![03fe616d69601f33daa73ceb72e039188cecd19a84261f73f42a2875246ab079.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/03fe616d69601f33daa73ceb72e039188cecd19a84261f73f42a2875246ab079.jpg)

![7b2e6ddd00caed5ecdef2c6bff0df78a74a3d9fddcee9702e2bb0fa5d15ae4d0.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/7b2e6ddd00caed5ecdef2c6bff0df78a74a3d9fddcee9702e2bb0fa5d15ae4d0.jpg)

![7bad04d5ebfb8a48e4b3c8073bb7bf1d278b2edcc5613daced6383f4de8421a0.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/7bad04d5ebfb8a48e4b3c8073bb7bf1d278b2edcc5613daced6383f4de8421a0.jpg)

![7bc84e1870e069f30d41809dd703812e5ae44993be481808ca99a19f47f099c8.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/7bc84e1870e069f30d41809dd703812e5ae44993be481808ca99a19f47f099c8.jpg)

![ca71f1e06edf61bb5f504441ee9e34133c36898bc5a1541622c3992a9703b76b.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/ca71f1e06edf61bb5f504441ee9e34133c36898bc5a1541622c3992a9703b76b.jpg)

![fca9bfdf63ffdef2966568abb4465a4282c156254dd7b34918d326959cc9b504.jpg](../icml_results/483_Practical%20and%20Optimal%20Algorithm%20for%20Linear%20Contextual%20Bandits%20with%20Rare%20Parameter%20Updates/tables/fca9bfdf63ffdef2966568abb4465a4282c156254dd7b34918d326959cc9b504.jpg)

<a id="paper-0475"></a>

## Spectral-Informed Neural Networks Outperform Spectral methods in High-dimensional PDEs


### Images

![0612b2bac1fa65d2074beb332aee7c186f106b4151982a625cf5ef6ed4deb257.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/0612b2bac1fa65d2074beb332aee7c186f106b4151982a625cf5ef6ed4deb257.jpg)

![273fa2819841e0dfbc28fccdbae98b290f93648ac8fc048bc1b9e0d8794c125e.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/273fa2819841e0dfbc28fccdbae98b290f93648ac8fc048bc1b9e0d8794c125e.jpg)

![3d2cefa03827f17d8787f5c443625edc718f1b7e9033a080277b73fe8be0722e.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/3d2cefa03827f17d8787f5c443625edc718f1b7e9033a080277b73fe8be0722e.jpg)

![3e6ede717a61d18099e7d3c1c394711435bae24887a4060e32d93f39fd6c03a0.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/3e6ede717a61d18099e7d3c1c394711435bae24887a4060e32d93f39fd6c03a0.jpg)

![6d0d32cb5118e40de4d23785a4d801dec9357d3ae8a077d09a71cfe2962c0d2c.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/6d0d32cb5118e40de4d23785a4d801dec9357d3ae8a077d09a71cfe2962c0d2c.jpg)

![7793428417a95fcabd0d91fd9cf78cb570a212e48a0d7de795cb39cf78b514f0.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/7793428417a95fcabd0d91fd9cf78cb570a212e48a0d7de795cb39cf78b514f0.jpg)

![8879fd55495a41d9a5b50674de329f72c01f23a4a1dba1cd4a8feec77f8045bd.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/8879fd55495a41d9a5b50674de329f72c01f23a4a1dba1cd4a8feec77f8045bd.jpg)

![a82f876a3f4ba65041fe3a832a017493774d18ab94440009e11e14ca82d514e7.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/a82f876a3f4ba65041fe3a832a017493774d18ab94440009e11e14ca82d514e7.jpg)

![ac2f7507ebe5597fbf4c72bd266ad2855026a272eebb72871456c58886ae491b.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/ac2f7507ebe5597fbf4c72bd266ad2855026a272eebb72871456c58886ae491b.jpg)

![b4d1ee8dad0abd16e7d34ce1463148de4d4d31c5650b4b43cd56f1844231b2a0.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/b4d1ee8dad0abd16e7d34ce1463148de4d4d31c5650b4b43cd56f1844231b2a0.jpg)

![c88f7e22544be33ff7098a639af61d68e293b67ffebeb985a7f51385894a8c72.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/c88f7e22544be33ff7098a639af61d68e293b67ffebeb985a7f51385894a8c72.jpg)

![e71ae66c0918bffb33558bbb679f54e6b2fd651607baec66810461a984980f4c.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/e71ae66c0918bffb33558bbb679f54e6b2fd651607baec66810461a984980f4c.jpg)

![fae7c9c3c6d1f941fb33213901a5e76a42c965ad64c4cdbf8cfa0a5b4112348c.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/images/fae7c9c3c6d1f941fb33213901a5e76a42c965ad64c4cdbf8cfa0a5b4112348c.jpg)

### Tables

![1a6a86a974323939627270e7e43eb62513acdacd24199a2ba1af4090bc41db7f.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/1a6a86a974323939627270e7e43eb62513acdacd24199a2ba1af4090bc41db7f.jpg)

![1ab571a9a3ac87355ed621f10d4d411f903eaf6d2ef6ef3cab73faa2d5a90a7c.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/1ab571a9a3ac87355ed621f10d4d411f903eaf6d2ef6ef3cab73faa2d5a90a7c.jpg)

![26ba7f44e97ff02900da1a6a6223b96a0747e3a78e31895e8a5889a05621a4d4.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/26ba7f44e97ff02900da1a6a6223b96a0747e3a78e31895e8a5889a05621a4d4.jpg)

![57e7ddb47dc626b60b2a31ab837261a81ae7ea845aea359f22a6c8740625045b.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/57e7ddb47dc626b60b2a31ab837261a81ae7ea845aea359f22a6c8740625045b.jpg)

![5b70c3c5e57898c31bdd3d381e0191bcac6fee0ebaeae7753e1eb32297f4b2ba.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/5b70c3c5e57898c31bdd3d381e0191bcac6fee0ebaeae7753e1eb32297f4b2ba.jpg)

![6029b92cc06848dcd32b53700ff358c29e5f7a3add409d84875efe56e5aea1fa.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/6029b92cc06848dcd32b53700ff358c29e5f7a3add409d84875efe56e5aea1fa.jpg)

![6acdecdabc68888bc22e924f17b6b33b13844ed348acf8931f3c43fbbf8eabcd.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/6acdecdabc68888bc22e924f17b6b33b13844ed348acf8931f3c43fbbf8eabcd.jpg)

![6ee882c89a4664e261a09aec5192eb62beed70ddbe267853159ece3ffff59062.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/6ee882c89a4664e261a09aec5192eb62beed70ddbe267853159ece3ffff59062.jpg)

![8dabd39ad8ce54b0b14cf99a8ae6cacf81417f0af17c3ef48aec945e589765e9.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/8dabd39ad8ce54b0b14cf99a8ae6cacf81417f0af17c3ef48aec945e589765e9.jpg)

![a22d69db5a1aa5dafc72f04481c9499d0c035120abde1d340c5cdd2076689f2d.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/a22d69db5a1aa5dafc72f04481c9499d0c035120abde1d340c5cdd2076689f2d.jpg)

![eb35078cfd4bd94d5a6fe61ee199e35ad6aec3c2d0f0da0982f2fa5dc556c04e.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/eb35078cfd4bd94d5a6fe61ee199e35ad6aec3c2d0f0da0982f2fa5dc556c04e.jpg)

![f467694e866be4e0d2f441bfcea23a0ad355e4f994497ecc1c3a74629ebc8062.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/f467694e866be4e0d2f441bfcea23a0ad355e4f994497ecc1c3a74629ebc8062.jpg)

![f5017ccdf65f3344c413b128901d3b0aeab46e1eab6276104434a684c8b403c7.jpg](../icml_results/484_Spectral-Informed%20Neural%20Networks%20Outperform%20Spectral%20methods%20in%20High-dimensional%20PDEs/tables/f5017ccdf65f3344c413b128901d3b0aeab46e1eab6276104434a684c8b403c7.jpg)

<a id="paper-0476"></a>

## FIRE: Multi-Fidelity Regression with Distribution-Conditioned In-Context Learning Using Tabular Foundation Models


### Images

![07515c3121eb26fdb8d4d0f85b8a9aeb6d80d732d72a1aff915281bf1b0f49df.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/07515c3121eb26fdb8d4d0f85b8a9aeb6d80d732d72a1aff915281bf1b0f49df.jpg)

![104b06f7bcc26331454db17f9c47ab08a429167389ebf61bfb6b30be800c5df6.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/104b06f7bcc26331454db17f9c47ab08a429167389ebf61bfb6b30be800c5df6.jpg)

![164c4c726b9c4bd8cfbe88319f45d663558c3dbf47cfc0f38e699cadc79fc32d.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/164c4c726b9c4bd8cfbe88319f45d663558c3dbf47cfc0f38e699cadc79fc32d.jpg)

![16957816bed7591158d08a0bcbec24345fbfd61c1b515d984b950889cb4931e2.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/16957816bed7591158d08a0bcbec24345fbfd61c1b515d984b950889cb4931e2.jpg)

![35b2cf54b56b915bab1fc0f9ce921f97b1649e3efa9a6a28a642e841df858ce1.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/35b2cf54b56b915bab1fc0f9ce921f97b1649e3efa9a6a28a642e841df858ce1.jpg)

![3a6978abb2b15366e3f84ca726d9ab522e6be3cd8f4dd46ff4fd62eb22c806ac.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/3a6978abb2b15366e3f84ca726d9ab522e6be3cd8f4dd46ff4fd62eb22c806ac.jpg)

![531a519a25835d3e1f148c47cc5ddb4bf4e049ff5c585d2bcd43f5d2807f4c70.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/531a519a25835d3e1f148c47cc5ddb4bf4e049ff5c585d2bcd43f5d2807f4c70.jpg)

![549fc34bec75196bdcc7f85cd0fb3cf205c8c4e5bf4c0f4b726927a157cc1128.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/549fc34bec75196bdcc7f85cd0fb3cf205c8c4e5bf4c0f4b726927a157cc1128.jpg)

![56439fca626c552c992c648b8d548730edaed895d50d60c0ec2cdaadc7625dee.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/56439fca626c552c992c648b8d548730edaed895d50d60c0ec2cdaadc7625dee.jpg)

![5b7128c02cca9e0f7e4343b0ca486acc8c407d3b19ea7e9d25185253001e6d01.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/5b7128c02cca9e0f7e4343b0ca486acc8c407d3b19ea7e9d25185253001e6d01.jpg)

![67168e024f0e3216e9b43787b45bb6523e73d9137725f8654ad90a54105cbf4a.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/67168e024f0e3216e9b43787b45bb6523e73d9137725f8654ad90a54105cbf4a.jpg)

![7129d19206bd8ca497d52c7f1dd920018e487df1655c37b22bc93b86d3116f39.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/7129d19206bd8ca497d52c7f1dd920018e487df1655c37b22bc93b86d3116f39.jpg)

![73e4b1898dd287d8f91eac567b16e5f8f087df07f1c8969c2cb9ecbbeb7aa454.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/73e4b1898dd287d8f91eac567b16e5f8f087df07f1c8969c2cb9ecbbeb7aa454.jpg)

![78407bc9fb6e1f11eaf07aa2efff9a4912f5e7ba8c72d12710a27aa4f9f86a41.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/78407bc9fb6e1f11eaf07aa2efff9a4912f5e7ba8c72d12710a27aa4f9f86a41.jpg)

![7e5235cd1165201b228a9dd92fb2970ab3c4b13c83c4418a05183bdbb1fce576.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/7e5235cd1165201b228a9dd92fb2970ab3c4b13c83c4418a05183bdbb1fce576.jpg)

![8a395c8059bf93a6cf65763a780e3cd58b715b93cd1057a945df227f18a02805.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/8a395c8059bf93a6cf65763a780e3cd58b715b93cd1057a945df227f18a02805.jpg)

![9b53cc4a04ca04d969f1feead4f2f13072df6ce78f7ad6cb7113c241e8e7a1f2.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/9b53cc4a04ca04d969f1feead4f2f13072df6ce78f7ad6cb7113c241e8e7a1f2.jpg)

![9cffde511d212928e6270a40486bf580e58e7f7101d976d405218675df1e5cbc.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/9cffde511d212928e6270a40486bf580e58e7f7101d976d405218675df1e5cbc.jpg)

![aec871e1dccc16f5b5d970e3b4c21e05ca66152c859a2ac5a7b8d5d88f87b478.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/aec871e1dccc16f5b5d970e3b4c21e05ca66152c859a2ac5a7b8d5d88f87b478.jpg)

![c083e25936ff4678d8973d3f375089c2db5d2efcb6510ff7fb4085d5794baf6d.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/c083e25936ff4678d8973d3f375089c2db5d2efcb6510ff7fb4085d5794baf6d.jpg)

![c6575f4ff995c7ec104c19f19b0f1119440a113b0153bc1d9f3c926a153da79c.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/c6575f4ff995c7ec104c19f19b0f1119440a113b0153bc1d9f3c926a153da79c.jpg)

![c7860371229ef0e404ed35ffc92e634e2f013c0980044ead60ef73913f750a9b.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/c7860371229ef0e404ed35ffc92e634e2f013c0980044ead60ef73913f750a9b.jpg)

![d3e425e583d09a0991a13631a36d6510ca346accaf52f824b0cacfda563f7426.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/d3e425e583d09a0991a13631a36d6510ca346accaf52f824b0cacfda563f7426.jpg)

![d5c9cd229a5eedc6e0ce85a16d17ad8a0305e5e3269b2d293acf708473094b84.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/d5c9cd229a5eedc6e0ce85a16d17ad8a0305e5e3269b2d293acf708473094b84.jpg)

![d63bfa1e9d2ca944d59153df359a108f9b5eb7b1e229b3b1ac81d220e42ef038.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/d63bfa1e9d2ca944d59153df359a108f9b5eb7b1e229b3b1ac81d220e42ef038.jpg)

![da3105ca547076c52da72d19ef141d43605fc0215be3ba26ac222f4b56208efd.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/da3105ca547076c52da72d19ef141d43605fc0215be3ba26ac222f4b56208efd.jpg)

![e72af4cdbfbd596cd308687363ec5dbec280be2a33048d526d3026d5a481f871.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/e72af4cdbfbd596cd308687363ec5dbec280be2a33048d526d3026d5a481f871.jpg)

![ec16ab134c9e33cd9ee03b533d73108eca6e3a57fd46df0720af849c185b5f21.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/ec16ab134c9e33cd9ee03b533d73108eca6e3a57fd46df0720af849c185b5f21.jpg)

![f899d9c3566431d99de73082d8b410411b98ef8293391fee9d4fa1b0274b2bac.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/f899d9c3566431d99de73082d8b410411b98ef8293391fee9d4fa1b0274b2bac.jpg)

![f8bc391d72aa27b8c0b42879f9d1e8d2c4f48d81785a0f9f74e3a8c4a4fae10e.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/f8bc391d72aa27b8c0b42879f9d1e8d2c4f48d81785a0f9f74e3a8c4a4fae10e.jpg)

![fc094f12368429f7a959b0d4ea137379976a2883f10208dd9317a07268e6c3b9.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/fc094f12368429f7a959b0d4ea137379976a2883f10208dd9317a07268e6c3b9.jpg)

![fcb9381299683515b6c207495df28f738f9f5c14fb4f9c89a7efba645d5ede2a.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/images/fcb9381299683515b6c207495df28f738f9f5c14fb4f9c89a7efba645d5ede2a.jpg)

### Tables

![30efe3af95ddc96f735ee226ad97f44611e9896f525db0455d96eac79ff71ee1.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/tables/30efe3af95ddc96f735ee226ad97f44611e9896f525db0455d96eac79ff71ee1.jpg)

![3edd3e7e7a233b98c8baa522aef1e58072d203ff41e5c70b674a28cdc0ce825b.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/tables/3edd3e7e7a233b98c8baa522aef1e58072d203ff41e5c70b674a28cdc0ce825b.jpg)

![831a53a2fb560cd031c7479931ec01c45cd6da2e4d3a65b1dfaa1dea182721d8.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/tables/831a53a2fb560cd031c7479931ec01c45cd6da2e4d3a65b1dfaa1dea182721d8.jpg)

![f85f1a0ccb0e978534597c1e93ef31ba1e632550b32e95348d683f1dd14ff951.jpg](../icml_results/485_FIRE_%20Multi-Fidelity%20Regression%20with%20Distribution-Conditioned%20In-Context%20Learning%20Using%20Tabular%20Foun/tables/f85f1a0ccb0e978534597c1e93ef31ba1e632550b32e95348d683f1dd14ff951.jpg)

<a id="paper-0477"></a>

## Improved Dimension Dependence for Bandit Convex Optimization with Gradient Variation


### Tables

![0b6d7e140944ab9d87f9f0b386cb5ae9455974629435ca309225c8e725454867.jpg](../icml_results/486_Improved%20Dimension%20Dependence%20for%20Bandit%20Convex%20Optimization%20with%20Gradient%20Variation/tables/0b6d7e140944ab9d87f9f0b386cb5ae9455974629435ca309225c8e725454867.jpg)

![4ccfb24a8e83725758007e7cb41c92ff93dd88bdc3dfb0f947888e3985de2d73.jpg](../icml_results/486_Improved%20Dimension%20Dependence%20for%20Bandit%20Convex%20Optimization%20with%20Gradient%20Variation/tables/4ccfb24a8e83725758007e7cb41c92ff93dd88bdc3dfb0f947888e3985de2d73.jpg)

<a id="paper-0478"></a>

## Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference


### Tables

![00ab03da9266c828660b3ae9dad42a3e848b8889978c26cf15c7476486717320.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/00ab03da9266c828660b3ae9dad42a3e848b8889978c26cf15c7476486717320.jpg)

![1ef1e1631c72b7aab8de1070271d1988870010a6e0e6c08e17c106b93b34a4c0.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/1ef1e1631c72b7aab8de1070271d1988870010a6e0e6c08e17c106b93b34a4c0.jpg)

![1f2e86ac7ad0ccf4de43f48176e0ddc28fda740f722054b619a6afc56399ec52.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/1f2e86ac7ad0ccf4de43f48176e0ddc28fda740f722054b619a6afc56399ec52.jpg)

![602c1efde8854d05c086b91c46ad4a71281a265533a8212e742e089f7b816239.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/602c1efde8854d05c086b91c46ad4a71281a265533a8212e742e089f7b816239.jpg)

![67a9015a04d7ec79e20971414b092fef3befbe06fce2e87ef65ecf6cdcd52727.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/67a9015a04d7ec79e20971414b092fef3befbe06fce2e87ef65ecf6cdcd52727.jpg)

![772cb834b1db5574239d0802223c72df0a06a89c4175f44f8cd883902bed9e47.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/772cb834b1db5574239d0802223c72df0a06a89c4175f44f8cd883902bed9e47.jpg)

![d72c28d8dcabb53015ac2356f22f2a159fd351b6c3c2c8d91a8c65b7f3170b4e.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/d72c28d8dcabb53015ac2356f22f2a159fd351b6c3c2c8d91a8c65b7f3170b4e.jpg)

![dcd4619e70cb5414272c9db3931384fa4f0062cbd0d5e015de08f376b0032386.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/dcd4619e70cb5414272c9db3931384fa4f0062cbd0d5e015de08f376b0032386.jpg)

![e8e9f89b3da1ef43e1e432b35ba7f9fbfb9ed42fff25027ebe822d7a9fca64a8.jpg](../icml_results/487_Generative%20Modeling%20of%20Irregular%20Time%20Series%20via%20SDE-Induced%20Continuous-Discrete%20Variational%20Inferen/tables/e8e9f89b3da1ef43e1e432b35ba7f9fbfb9ed42fff25027ebe822d7a9fca64a8.jpg)

<a id="paper-0479"></a>

## Markov Chain Monte Carlo without Evaluating the Target: an Auxiliary Variable Approach


### Images

![013b31169e69af79c4a89bf7f18286dcf9aa59849340d95e9102381fb0925ee4.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/013b31169e69af79c4a89bf7f18286dcf9aa59849340d95e9102381fb0925ee4.jpg)

![04084ae86b7ed8e57dba75b4ab05973a7d9d6fc8eb7644f8dc30b1be30f9c29c.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/04084ae86b7ed8e57dba75b4ab05973a7d9d6fc8eb7644f8dc30b1be30f9c29c.jpg)

![20a5209208ef185abb670103d07c3fb176492a53056e5c9bc62a1ab2825b1e85.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/20a5209208ef185abb670103d07c3fb176492a53056e5c9bc62a1ab2825b1e85.jpg)

![2676718d48179f3734a4ce00a0188fd59999346f07741dc4001538e10bb30cc0.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/2676718d48179f3734a4ce00a0188fd59999346f07741dc4001538e10bb30cc0.jpg)

![34a4484811a9caa8a669878e5a73dd056cb5d46e6412653f88ab50272c152136.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/34a4484811a9caa8a669878e5a73dd056cb5d46e6412653f88ab50272c152136.jpg)

![427669195dd23c2b8f0bd2774b9fe602da43c6804a4c1c6178c06abe8e96010c.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/427669195dd23c2b8f0bd2774b9fe602da43c6804a4c1c6178c06abe8e96010c.jpg)

![4809dbb0685706f3644f4ebedaf8ce748ba3836cfe0e23745f428c0a0d898feb.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/4809dbb0685706f3644f4ebedaf8ce748ba3836cfe0e23745f428c0a0d898feb.jpg)

![5520f8847ba5d61cbd36c4772a0b03d50ad58334497973067d1f3dda90f12537.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/5520f8847ba5d61cbd36c4772a0b03d50ad58334497973067d1f3dda90f12537.jpg)

![886d67bd50b70af2cc984503a7748426555614695e2448e2335d73b3c1255e12.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/886d67bd50b70af2cc984503a7748426555614695e2448e2335d73b3c1255e12.jpg)

![b22b29b58fed6e5c7341617eb9d7a1c4f3574798ae47590b4488068ebd65c7fb.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/b22b29b58fed6e5c7341617eb9d7a1c4f3574798ae47590b4488068ebd65c7fb.jpg)

![b33c162e54c131111ba923dda3b3b0693efe1ee6bf811edc2e464823c450c7d5.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/b33c162e54c131111ba923dda3b3b0693efe1ee6bf811edc2e464823c450c7d5.jpg)

![b787d7e2b267d2b3832167e329a80af4ed6fb84df0b84dc041dcb4279d11d372.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/b787d7e2b267d2b3832167e329a80af4ed6fb84df0b84dc041dcb4279d11d372.jpg)

![b88a98c552e186556a5abace1e643882edfd77719e7bb09918f24e8eb0d9f6bd.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/b88a98c552e186556a5abace1e643882edfd77719e7bb09918f24e8eb0d9f6bd.jpg)

![be3dea6eb844dc5a49ad8983bcd88e300df04dae611eeb21b377cc6fb7805d69.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/be3dea6eb844dc5a49ad8983bcd88e300df04dae611eeb21b377cc6fb7805d69.jpg)

![c17bc1709f8b7847f076b8e7a16029f40c468325d9ff5599cebf8110dc84d2e4.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/c17bc1709f8b7847f076b8e7a16029f40c468325d9ff5599cebf8110dc84d2e4.jpg)

![d93fbe24091047a90b453660af7136fe7110e46539a15e06ada14303a1bea7f1.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/d93fbe24091047a90b453660af7136fe7110e46539a15e06ada14303a1bea7f1.jpg)

![dd64cc24f4bec31fac5a15a65ab00d69007fcae8f9e9e4ac1cb1a5c70a439bed.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/dd64cc24f4bec31fac5a15a65ab00d69007fcae8f9e9e4ac1cb1a5c70a439bed.jpg)

![f2ec3e585d5ceae42a793a68955aebd200d5863e360954426043269831203ecc.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/f2ec3e585d5ceae42a793a68955aebd200d5863e360954426043269831203ecc.jpg)

![f8e25ca6f38936766cdb01a96a235af49f99e2a6330d227ba583274742d668e5.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/f8e25ca6f38936766cdb01a96a235af49f99e2a6330d227ba583274742d668e5.jpg)

![ffd72beb1b5e2da97feffea6b06e14f82b482dcf5b101d53a7e25e3aabe1abfb.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/images/ffd72beb1b5e2da97feffea6b06e14f82b482dcf5b101d53a7e25e3aabe1abfb.jpg)

### Tables

![3395b1006b9f2f0036b38e31f8a2bbba0dba4453cbff6d835d8710815005dee7.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/3395b1006b9f2f0036b38e31f8a2bbba0dba4453cbff6d835d8710815005dee7.jpg)

![429d4ba54705c5b2c2d08805a73a1d1041efb2391c826b49c28c261883fd6c86.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/429d4ba54705c5b2c2d08805a73a1d1041efb2391c826b49c28c261883fd6c86.jpg)

![6baa52ff02d9c9fa0cbb3f5078be02676af6e43f74b5cf40944556b37bacc6a7.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/6baa52ff02d9c9fa0cbb3f5078be02676af6e43f74b5cf40944556b37bacc6a7.jpg)

![74ebb444881e9fefffe5eee5b154a928b9eae1f310b2a6925d3b004cd5b5491e.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/74ebb444881e9fefffe5eee5b154a928b9eae1f310b2a6925d3b004cd5b5491e.jpg)

![8789c8551cd16e8bdc0b3e2b8dd28b79e397a8b0c12400de1ac414b878fd97ed.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/8789c8551cd16e8bdc0b3e2b8dd28b79e397a8b0c12400de1ac414b878fd97ed.jpg)

![987ab3e693d5ed28fafdbb3f77c1d25d61bfe70b0e344d7d4e3fe68ec2bab87e.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/987ab3e693d5ed28fafdbb3f77c1d25d61bfe70b0e344d7d4e3fe68ec2bab87e.jpg)

![98f4908c2b2817f7600192e8bde4ca0de92014d814ba36a3a80fa569dee0d5ea.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/98f4908c2b2817f7600192e8bde4ca0de92014d814ba36a3a80fa569dee0d5ea.jpg)

![eb4e9aa0853214d0cf42121e4a37a38fa809bd791220354c6fdc2297e08182e6.jpg](../icml_results/488_Markov%20Chain%20Monte%20Carlo%20without%20Evaluating%20the%20Target_%20an%20Auxiliary%20Variable%20Approach/tables/eb4e9aa0853214d0cf42121e4a37a38fa809bd791220354c6fdc2297e08182e6.jpg)

<a id="paper-0480"></a>

## Learning to Discover at Test Time


### Images

![06c97d1a481b64d0b5618b99944f51f75f455e4eef562020734389c8d33cc50a.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/06c97d1a481b64d0b5618b99944f51f75f455e4eef562020734389c8d33cc50a.jpg)

![20e1d1ac7c9ea60d72ce4bdae1bc061cb519b050eda6db9af41f39052e7b6bc6.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/20e1d1ac7c9ea60d72ce4bdae1bc061cb519b050eda6db9af41f39052e7b6bc6.jpg)

![230fb38e0c59f6cae4aaad05dce1b32b91b97df70a11ef822999cf7d05ae2231.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/230fb38e0c59f6cae4aaad05dce1b32b91b97df70a11ef822999cf7d05ae2231.jpg)

![347b04c911ac2824c1f507d9d086ddfa16aab432f60c0ccecc4984ee99ee1bcd.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/347b04c911ac2824c1f507d9d086ddfa16aab432f60c0ccecc4984ee99ee1bcd.jpg)

![3f36bb72073d7b104e9c4fcfa2dda833c6b95375baea37485304b9d16e68f047.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/3f36bb72073d7b104e9c4fcfa2dda833c6b95375baea37485304b9d16e68f047.jpg)

![441ef37f732b2be79c59f015444906561363b1a6ed91dbf600ef11803838e7f3.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/441ef37f732b2be79c59f015444906561363b1a6ed91dbf600ef11803838e7f3.jpg)

![4cd5f1ef6f057bb0d77e05fd1994a14b4dce360806d7d884eaebe7ea3c9511fd.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/4cd5f1ef6f057bb0d77e05fd1994a14b4dce360806d7d884eaebe7ea3c9511fd.jpg)

![5c21939ddcd191fbf51b89a6ab3b4122f964424204cf97bfd0d2cd48e0f7eefe.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/5c21939ddcd191fbf51b89a6ab3b4122f964424204cf97bfd0d2cd48e0f7eefe.jpg)

![79d6949b5ee2e853a4c04170103df8304caa5669ecb266c704432c74e0d46f1c.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/79d6949b5ee2e853a4c04170103df8304caa5669ecb266c704432c74e0d46f1c.jpg)

![947eac24f794ae833b204269f88c2c112dc8b7bcdae2d8523b0e49b16f15f2d2.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/947eac24f794ae833b204269f88c2c112dc8b7bcdae2d8523b0e49b16f15f2d2.jpg)

![b57eeae5f46a08089a1a0624a9fd73a899cf04f1965ebe2e3cc96812781c476e.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/b57eeae5f46a08089a1a0624a9fd73a899cf04f1965ebe2e3cc96812781c476e.jpg)

![c351cc4669b61813c4f9dc6bae3d81029646fa0681802b8aa3c54915a91c54dd.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/c351cc4669b61813c4f9dc6bae3d81029646fa0681802b8aa3c54915a91c54dd.jpg)

![c53d68de031dc43de220fa42d226693e6bcab1c08c49bb5b355bd91ebfb23d0a.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/c53d68de031dc43de220fa42d226693e6bcab1c08c49bb5b355bd91ebfb23d0a.jpg)

![e480cb3aa83a56a13f12d3d31828d31a2b6a8f9560ed210fb9a4ec1d4ef68f52.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/e480cb3aa83a56a13f12d3d31828d31a2b6a8f9560ed210fb9a4ec1d4ef68f52.jpg)

![e5999215ffe071c7de721ec5859b1f0fd7c8cb4f95143337e37d8af6f7e71f78.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/images/e5999215ffe071c7de721ec5859b1f0fd7c8cb4f95143337e37d8af6f7e71f78.jpg)

### Tables

![2bbbfbf8f5c5fe03d3c9da6290fadc848838faebb465cc1f02aef5ac075aea03.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/2bbbfbf8f5c5fe03d3c9da6290fadc848838faebb465cc1f02aef5ac075aea03.jpg)

![3548b53d640a7827e476880a94376bf3e65326dc7138d2aff5b06299ad5995a1.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/3548b53d640a7827e476880a94376bf3e65326dc7138d2aff5b06299ad5995a1.jpg)

![49310bc18d0b8f576a8f559b9b18a6dab76623d41395dbec5def49eb071c885e.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/49310bc18d0b8f576a8f559b9b18a6dab76623d41395dbec5def49eb071c885e.jpg)

![5f2713fde120bccae0b6246f317eca51865f61980c6d9cee53aea2490af152f4.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/5f2713fde120bccae0b6246f317eca51865f61980c6d9cee53aea2490af152f4.jpg)

![65d75b1f89e03549ebf8051a246341a17ce309d716416ba667f219bf0a7b65b4.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/65d75b1f89e03549ebf8051a246341a17ce309d716416ba667f219bf0a7b65b4.jpg)

![6cc232678c7d5aef654c93fdb2770ed29c6e18cfcb2ebe1b3a3a20f013f41470.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/6cc232678c7d5aef654c93fdb2770ed29c6e18cfcb2ebe1b3a3a20f013f41470.jpg)

![8056232559e19a1e1c879ca5ea676bcbaba14001060b453326e40c1bce1aada1.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/8056232559e19a1e1c879ca5ea676bcbaba14001060b453326e40c1bce1aada1.jpg)

![8507c3684eae7ae3e6aff05d7786694d160ac83d730aceb27ff53efa39b3d484.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/8507c3684eae7ae3e6aff05d7786694d160ac83d730aceb27ff53efa39b3d484.jpg)

![99ca2d7df2eb373e3fdddcad92d05cc4ea6aff54418b6093d5e2680f3d49bc08.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/99ca2d7df2eb373e3fdddcad92d05cc4ea6aff54418b6093d5e2680f3d49bc08.jpg)

![a168593402fbc57add790c60c0aabe9cc7945c1734e5546cbd8dac8c865ab5f7.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/a168593402fbc57add790c60c0aabe9cc7945c1734e5546cbd8dac8c865ab5f7.jpg)

![c124894f77b478a5b75596a0ceb02f721f51e787f18240cf7db625657f062ebb.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/c124894f77b478a5b75596a0ceb02f721f51e787f18240cf7db625657f062ebb.jpg)

![de55d912153e2aa167d3f43894662bcd34d25f1c133c4ade4ffa68b1cb287a87.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/de55d912153e2aa167d3f43894662bcd34d25f1c133c4ade4ffa68b1cb287a87.jpg)

![e1376afc12eb7ca787d15b6b85cca7f31a9dc0297a608deb12c02afcd8a73346.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/e1376afc12eb7ca787d15b6b85cca7f31a9dc0297a608deb12c02afcd8a73346.jpg)

![f121ca3b41e05b0e1de8d8843d560470db0e46e402d56c17d3a685bd4506c6cf.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/f121ca3b41e05b0e1de8d8843d560470db0e46e402d56c17d3a685bd4506c6cf.jpg)

![f63e99441c5d787e32ec3b1820edca2735a01eacc5349192f2295ca968700b79.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/f63e99441c5d787e32ec3b1820edca2735a01eacc5349192f2295ca968700b79.jpg)

![f77330df14ffbe29376d2109afafa09ef9346ffc25456004ca0af491d7688d6b.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/f77330df14ffbe29376d2109afafa09ef9346ffc25456004ca0af491d7688d6b.jpg)

![f778019431ce1a3e8fbb2ac5fd1fc8d73a952385b97bef83a7db61754ffb9864.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/f778019431ce1a3e8fbb2ac5fd1fc8d73a952385b97bef83a7db61754ffb9864.jpg)

![f932402620da74b5e37950866831352d7fbf8216799887a68a1aa2c4a30a75d4.jpg](../icml_results/489_Learning%20to%20Discover%20at%20Test%20Time/tables/f932402620da74b5e37950866831352d7fbf8216799887a68a1aa2c4a30a75d4.jpg)

<a id="paper-0481"></a>

## End-to-End Compression for Tabular Foundation Models


### Images

![0b918521204e9a20457d715260353126095551ed55f432ec42e8274880b50d51.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/0b918521204e9a20457d715260353126095551ed55f432ec42e8274880b50d51.jpg)

![1832dcdd8046a60c20f769245f6e0190cbf67d93be255e6c4a84ac327d4079b0.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/1832dcdd8046a60c20f769245f6e0190cbf67d93be255e6c4a84ac327d4079b0.jpg)

![19f1d04dc17acef841de4c3e164a6bd0455bc4ecbb630c802fd48c713e2f3036.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/19f1d04dc17acef841de4c3e164a6bd0455bc4ecbb630c802fd48c713e2f3036.jpg)

![2506d49916f5aa43730cf61d6abebbeb12d461d78edcc174bb7541cee3f75090.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/2506d49916f5aa43730cf61d6abebbeb12d461d78edcc174bb7541cee3f75090.jpg)

![2bd34e84975f0168fd76cbf7374b0a33e9e62a91d0615fb87a92dc00b8c4746c.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/2bd34e84975f0168fd76cbf7374b0a33e9e62a91d0615fb87a92dc00b8c4746c.jpg)

![2ed9c11b395fa67943d891d29d3e4f913712b01ea3aa11a4a5ab931c000c1906.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/2ed9c11b395fa67943d891d29d3e4f913712b01ea3aa11a4a5ab931c000c1906.jpg)

![6d5867f5dd5401e85280fe730d7b24438353723763c90306d3278f4f77a7abee.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/6d5867f5dd5401e85280fe730d7b24438353723763c90306d3278f4f77a7abee.jpg)

![7ddba83216fd88ca1a17a87e4a3e41f3776b20f56621d74f32b2c200ba3980db.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/7ddba83216fd88ca1a17a87e4a3e41f3776b20f56621d74f32b2c200ba3980db.jpg)

![8cefe8f7fb6fad20aa4e5ca7c6b8cb97871430a7f37556feeb329c7a3e8ecac9.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/8cefe8f7fb6fad20aa4e5ca7c6b8cb97871430a7f37556feeb329c7a3e8ecac9.jpg)

![94bbc843038829d9deafc0b6eb358ac443f9b708bd64115f340c54e9784ce329.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/94bbc843038829d9deafc0b6eb358ac443f9b708bd64115f340c54e9784ce329.jpg)

![9abc8867a7b95bd3d3a684cf7a15a0af48a1d29ad42734dfa1e19c5e4c668501.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/9abc8867a7b95bd3d3a684cf7a15a0af48a1d29ad42734dfa1e19c5e4c668501.jpg)

![a2d78e5a7770df3a9d34b56313f47eb4f6a6f92a9a920e1f4b3c0356f59973b3.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/a2d78e5a7770df3a9d34b56313f47eb4f6a6f92a9a920e1f4b3c0356f59973b3.jpg)

![a551036df88948dcd411b676d8ee420cb3d994338c0713ddccaf42faade81586.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/a551036df88948dcd411b676d8ee420cb3d994338c0713ddccaf42faade81586.jpg)

![b06f0a6eb0bb4c904fc810325c019d2960fcb276dd5615bbda39692d7134f427.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/b06f0a6eb0bb4c904fc810325c019d2960fcb276dd5615bbda39692d7134f427.jpg)

![c31c6750edad9bd319029fe28dedc4f0d2f7c5ddd7b7620800569215d307f04e.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/c31c6750edad9bd319029fe28dedc4f0d2f7c5ddd7b7620800569215d307f04e.jpg)

![dc45ce040ddf6c7193f72f317181bd53ed578e0b205a924c33d8b0179f53e1bb.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/dc45ce040ddf6c7193f72f317181bd53ed578e0b205a924c33d8b0179f53e1bb.jpg)

![ea021f680bb75cb6902fea3c070640479f255d65c154b5701b0ba800bc989426.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/ea021f680bb75cb6902fea3c070640479f255d65c154b5701b0ba800bc989426.jpg)

![f7486f1ca5ffc3e97ad1453fe1fd5b95d08c735c226d3d818e2c534b298a5085.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/images/f7486f1ca5ffc3e97ad1453fe1fd5b95d08c735c226d3d818e2c534b298a5085.jpg)

### Tables

![08d3402cb62d7a55ec2d6499bbc2ea864c57a126bc57bbaffc65f460007d96f9.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/08d3402cb62d7a55ec2d6499bbc2ea864c57a126bc57bbaffc65f460007d96f9.jpg)

![27b795a6ef34db8c368ec658b1111f262fed43c3570e44f9ec23e5ea107584ca.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/27b795a6ef34db8c368ec658b1111f262fed43c3570e44f9ec23e5ea107584ca.jpg)

![28061e313c28c87d83a946cdc5fee83b8316b30f7a324817b947691ee3fff776.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/28061e313c28c87d83a946cdc5fee83b8316b30f7a324817b947691ee3fff776.jpg)

![44000d2f2d30c60a9ba5676ab550a27511cf9d4617200ed99676cb40dcc03bbb.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/44000d2f2d30c60a9ba5676ab550a27511cf9d4617200ed99676cb40dcc03bbb.jpg)

![503fa413b4758ed04830f4b6a126cf97bfd5e754f3971019dc01bc3da9614b66.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/503fa413b4758ed04830f4b6a126cf97bfd5e754f3971019dc01bc3da9614b66.jpg)

![55a09452cc9a11d3585a28ca751c906c82f91c18b9c56b7da7682f3b1273f72e.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/55a09452cc9a11d3585a28ca751c906c82f91c18b9c56b7da7682f3b1273f72e.jpg)

![c7666d7b3369549fe0cb79aaf7737ce24c9c96b6dcb11b6bff5755665627a53f.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/c7666d7b3369549fe0cb79aaf7737ce24c9c96b6dcb11b6bff5755665627a53f.jpg)

![e4f53d77b8234b280fd9f2fabf05c12a5467ba561085ceae436cca2f596522ad.jpg](../icml_results/490_End-to-End%20Compression%20for%20Tabular%20Foundation%20Models/tables/e4f53d77b8234b280fd9f2fabf05c12a5467ba561085ceae436cca2f596522ad.jpg)

<a id="paper-0482"></a>

## FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment


### Images

![1f03c8e4699927a560e34e37ad2adab9d309009e4ff38607ab96dd9a2a3eb97e.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/1f03c8e4699927a560e34e37ad2adab9d309009e4ff38607ab96dd9a2a3eb97e.jpg)

![308c3907749d99a646c83e37c82ffaf0dacd83cbaac788f8d24ff10653b8ff48.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/308c3907749d99a646c83e37c82ffaf0dacd83cbaac788f8d24ff10653b8ff48.jpg)

![5d2e4316ee45967f4b11a21907f01d67426d7bab9fe9b17fa1d7075eac5dc226.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/5d2e4316ee45967f4b11a21907f01d67426d7bab9fe9b17fa1d7075eac5dc226.jpg)

![6291034a8214651768a9cdb06801918bfaa53b8ea043d5f073a5b1b0963c2779.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/6291034a8214651768a9cdb06801918bfaa53b8ea043d5f073a5b1b0963c2779.jpg)

![739fd800d04cc6b4064b84cb779dd9fcf1ef734bd6deca4f8a8aa2d6cc342535.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/739fd800d04cc6b4064b84cb779dd9fcf1ef734bd6deca4f8a8aa2d6cc342535.jpg)

![8e6d2a032baf01a04568318b7c158103c0e6562ee43ce94fd3658d6b2521d3b5.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/8e6d2a032baf01a04568318b7c158103c0e6562ee43ce94fd3658d6b2521d3b5.jpg)

![a32a40a247dfb5fe073d9ebf960dc7a4ac1145181b71ed151b9652e7eebe8ae8.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/a32a40a247dfb5fe073d9ebf960dc7a4ac1145181b71ed151b9652e7eebe8ae8.jpg)

![cae6393fb2a56e6fdff47d787315a5e6267349e6a24aadce0338e6ff1a83af50.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/cae6393fb2a56e6fdff47d787315a5e6267349e6a24aadce0338e6ff1a83af50.jpg)

![d208a511c40bf7e5250c8db96b8c3c7797f4297c07d858fa8816cba7e0fc325f.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/d208a511c40bf7e5250c8db96b8c3c7797f4297c07d858fa8816cba7e0fc325f.jpg)

![fd4aa52860c53f4e73235307f9ecb8a7c824032b152a04a2a0607f7b0f903974.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/images/fd4aa52860c53f4e73235307f9ecb8a7c824032b152a04a2a0607f7b0f903974.jpg)

### Tables

![2be8f6f116d10ad0b18baf0da7b430e75c24768b0ae29879b4351f5b954459db.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/tables/2be8f6f116d10ad0b18baf0da7b430e75c24768b0ae29879b4351f5b954459db.jpg)

![8c794a4e2f237bd01ff752ebc32a58dfa5cc55b5d50765265946749200b800e5.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/tables/8c794a4e2f237bd01ff752ebc32a58dfa5cc55b5d50765265946749200b800e5.jpg)

![fcb378ef6d12d13dc2390c7634e684d3d6c273fee466f54c0b09e991ecdb33d6.jpg](../icml_results/491_FlexRank_%20Nested%20Low-Rank%20Knowledge%20Decomposition%20for%20Adaptive%20Model%20Deployment/tables/fcb378ef6d12d13dc2390c7634e684d3d6c273fee466f54c0b09e991ecdb33d6.jpg)

<a id="paper-0483"></a>

## daVinci-Dev: Agent-native Mid-training for Software Engineering


### Images

![214a98fd9e165be5984b93d89fc5a81687adfc55a63051e7b18127fd54c64ba5.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/images/214a98fd9e165be5984b93d89fc5a81687adfc55a63051e7b18127fd54c64ba5.jpg)

![d8d85ba26c0f687cf5be9fbe0647dc6738958463b4dc11105fefd6defb12ba07.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/images/d8d85ba26c0f687cf5be9fbe0647dc6738958463b4dc11105fefd6defb12ba07.jpg)

![f85506ad25843f38416e4f20fee0753aad7811b221899d2accd1cccad56f5682.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/images/f85506ad25843f38416e4f20fee0753aad7811b221899d2accd1cccad56f5682.jpg)

![fdbe56b3d4211cab3009f54c5afb75544d6e5b80e003f412cd1ee21737e39e6d.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/images/fdbe56b3d4211cab3009f54c5afb75544d6e5b80e003f412cd1ee21737e39e6d.jpg)

### Tables

![2daac50730cda865eadfeab139d20f8f8ae27852cab2444222b76900243be55d.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/2daac50730cda865eadfeab139d20f8f8ae27852cab2444222b76900243be55d.jpg)

![5327c74ba5c20ab8ffd13ed6f1535c80db437e5ddcf123ec578cc1389d3ea358.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/5327c74ba5c20ab8ffd13ed6f1535c80db437e5ddcf123ec578cc1389d3ea358.jpg)

![b422207017d1cb16274c5919085eb71b88d5d5487cb64125324603ebd0143cec.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/b422207017d1cb16274c5919085eb71b88d5d5487cb64125324603ebd0143cec.jpg)

![c40cb18164318270f1d5c2d348f8207b7183ffc1f3d54501476dd326528cdf6e.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/c40cb18164318270f1d5c2d348f8207b7183ffc1f3d54501476dd326528cdf6e.jpg)

![e8c6153f8f67cbd0db42d9f082262ea546ea9c6b03518de56e9dbb4de5468a0e.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/e8c6153f8f67cbd0db42d9f082262ea546ea9c6b03518de56e9dbb4de5468a0e.jpg)

![ec3604d81680bdc5e41476f7fec6fd2160e5b8f35d86d326289e8319cf6c315f.jpg](../icml_results/492_daVinci-Dev_%20Agent-native%20Mid-training%20for%20Software%20Engineering/tables/ec3604d81680bdc5e41476f7fec6fd2160e5b8f35d86d326289e8319cf6c315f.jpg)

<a id="paper-0484"></a>

## PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space


### Images

![2b904b7c31a8481461493b63c3615e3462e95d5853ace15138903bb99e506a47.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/2b904b7c31a8481461493b63c3615e3462e95d5853ace15138903bb99e506a47.jpg)

![2fd1603c59d5840e915bafb8130c928f0a8e63a304c81b07f088ff741438387b.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/2fd1603c59d5840e915bafb8130c928f0a8e63a304c81b07f088ff741438387b.jpg)

![3d162ca5b806f88128716d490967b2cda5006c210f80ffe242823833774fbb8b.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/3d162ca5b806f88128716d490967b2cda5006c210f80ffe242823833774fbb8b.jpg)

![4fa604fdbda564d6768f330a5664f6d0034ff48bbf193a224010c55d92c42ae3.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/4fa604fdbda564d6768f330a5664f6d0034ff48bbf193a224010c55d92c42ae3.jpg)

![5a166727f8c6298f28b8f55e860e742f19b847d7173e81ecc099c28c93e0067d.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/5a166727f8c6298f28b8f55e860e742f19b847d7173e81ecc099c28c93e0067d.jpg)

![5da418a1ed4748b0074a295c9f86ae247a9110dd19384a7534b7cdd7a272a08f.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/5da418a1ed4748b0074a295c9f86ae247a9110dd19384a7534b7cdd7a272a08f.jpg)

![6ef14d932fc7c6f6435f01507d00ffbb9a2d11cd7525b778ec226ac4ae8a8d38.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/6ef14d932fc7c6f6435f01507d00ffbb9a2d11cd7525b778ec226ac4ae8a8d38.jpg)

![7840b2ca1e150486a60dc591850ec6d027e1094197cc04580dedb8f243ca7c54.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/7840b2ca1e150486a60dc591850ec6d027e1094197cc04580dedb8f243ca7c54.jpg)

![7c578e2a8bc164a8cdbc0475a59fc429e06cc4ae825b7d040e1b65a9a65e0762.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/7c578e2a8bc164a8cdbc0475a59fc429e06cc4ae825b7d040e1b65a9a65e0762.jpg)

![96243803051a0ebc82640488c0d45c458fb9f1b513051513a0c6240c347a669b.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/96243803051a0ebc82640488c0d45c458fb9f1b513051513a0c6240c347a669b.jpg)

![a111d2abb07a7af28865c550eb49bf2ca5c6ce90a83611312a67c9d0067847c2.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/a111d2abb07a7af28865c550eb49bf2ca5c6ce90a83611312a67c9d0067847c2.jpg)

![a41bec41a4426096c05d57777984931d662a369bd338540ed505730c71ed2a91.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/a41bec41a4426096c05d57777984931d662a369bd338540ed505730c71ed2a91.jpg)

![be3f8a13dd52f6a14f69aa36b48635b88c33f31ecb017a37589bdf28495ff8a4.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/be3f8a13dd52f6a14f69aa36b48635b88c33f31ecb017a37589bdf28495ff8a4.jpg)

![be8f8508ab932cd982d0b829539e55ee862438820b1c2e54333615057d70855c.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/be8f8508ab932cd982d0b829539e55ee862438820b1c2e54333615057d70855c.jpg)

![c6075537209c2d04c2c757ce026f5847e36a46f5239a5540d6a955a2f6226fab.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/c6075537209c2d04c2c757ce026f5847e36a46f5239a5540d6a955a2f6226fab.jpg)

![de7350b9bbc3f7165d0c84e80cb4b08217072fada4674ec9c24bdca8e3709f8e.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/de7350b9bbc3f7165d0c84e80cb4b08217072fada4674ec9c24bdca8e3709f8e.jpg)

![e420af94b534c2b1846a3779647404851af09a6df67e6ff0bf07fb173ba8ca1c.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/e420af94b534c2b1846a3779647404851af09a6df67e6ff0bf07fb173ba8ca1c.jpg)

![eb7d4554db701cc4b3f8b17b489b15c6deeb0b9b55ba1c7464ce797adff651c9.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/eb7d4554db701cc4b3f8b17b489b15c6deeb0b9b55ba1c7464ce797adff651c9.jpg)

![ec3a08fb2734f31bb3a2cfe9e2d438dd35c90ab3e59a00127d55260be304ea26.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/ec3a08fb2734f31bb3a2cfe9e2d438dd35c90ab3e59a00127d55260be304ea26.jpg)

![f83a43ac5a45acd9fafa79914c23cdfa5cb62154e1cb173beb44e713c6b4dae5.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/images/f83a43ac5a45acd9fafa79914c23cdfa5cb62154e1cb173beb44e713c6b4dae5.jpg)

### Tables

![28d85116b8160b8d0c5a5dbf257db6dfcd5ab83713004c597a534ca13fee4426.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/28d85116b8160b8d0c5a5dbf257db6dfcd5ab83713004c597a534ca13fee4426.jpg)

![3f0e0e4be92bae288879434bdbd0bc9d906241c91049c7298514a76933a0d7fd.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/3f0e0e4be92bae288879434bdbd0bc9d906241c91049c7298514a76933a0d7fd.jpg)

![41eedab823a87988f2603f28049a9b5d2191d18113baf998d1b94345d9d7960f.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/41eedab823a87988f2603f28049a9b5d2191d18113baf998d1b94345d9d7960f.jpg)

![49a2feca7de8891206513325103238e60f6cd18d91925055042b492601a48e5b.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/49a2feca7de8891206513325103238e60f6cd18d91925055042b492601a48e5b.jpg)

![5c76eb5755863faf7ab7b9d1ea13711af8e4da67c44702f32cc9c961d08cf8c9.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/5c76eb5755863faf7ab7b9d1ea13711af8e4da67c44702f32cc9c961d08cf8c9.jpg)

![6356c7d99b4d6db7fb3cf64be2856b2d260c42e144846992eeb43ce11465764a.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/6356c7d99b4d6db7fb3cf64be2856b2d260c42e144846992eeb43ce11465764a.jpg)

![6940c278f5f2225d66bb3cdc57fdec26733dca683ed858fa97ef44c945c82375.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/6940c278f5f2225d66bb3cdc57fdec26733dca683ed858fa97ef44c945c82375.jpg)

![6ba4f549869b4622cdcf6635716e49f2eb353b8d7e014d4e707a5febce5867d0.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/6ba4f549869b4622cdcf6635716e49f2eb353b8d7e014d4e707a5febce5867d0.jpg)

![6fc70b54603f7d155285272cb34011f155af8919af0848f325dd162a9b2f6262.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/6fc70b54603f7d155285272cb34011f155af8919af0848f325dd162a9b2f6262.jpg)

![72ef1314353cdbc7808d9f2d21055fa30c6822a3993d4addc594b6afeb4dfc33.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/72ef1314353cdbc7808d9f2d21055fa30c6822a3993d4addc594b6afeb4dfc33.jpg)

![7659ca33672d522f86569d16270ea95708a8f092864e9893c4c7aa9c5acf12a6.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/7659ca33672d522f86569d16270ea95708a8f092864e9893c4c7aa9c5acf12a6.jpg)

![7967c4721a674e78647cd93f331f40b0c201cfe71a11a54f336a88f914e49ac1.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/7967c4721a674e78647cd93f331f40b0c201cfe71a11a54f336a88f914e49ac1.jpg)

![9e17e976fcc5d432eae1bd36c432567f9476c383a01eca1d0221686444f2c74e.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/9e17e976fcc5d432eae1bd36c432567f9476c383a01eca1d0221686444f2c74e.jpg)

![bba14b0fa9dc0c0762384b3c0c6bea660d8d55df92d3414c356e97220ccdf22d.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/bba14b0fa9dc0c0762384b3c0c6bea660d8d55df92d3414c356e97220ccdf22d.jpg)

![e6209db0e00effd6ba6e3a4de572415ed78bd65bda127474bfa4af94ceefe362.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/e6209db0e00effd6ba6e3a4de572415ed78bd65bda127474bfa4af94ceefe362.jpg)

![f8ae6b658674029c4f8b8db47ab475db8cefc957b055e047e66ef18e0a60df04.jpg](../icml_results/493_PonderLM-2_%20Pretraining%20LLM%20with%20Latent%20Thoughts%20in%20Continuous%20Space/tables/f8ae6b658674029c4f8b8db47ab475db8cefc957b055e047e66ef18e0a60df04.jpg)

<a id="paper-0485"></a>

## EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video


### Images

![05bdb4831ef5fd880a82de8844c007a3cbc324bcbaf3db90caf6910b7e9b638f.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/05bdb4831ef5fd880a82de8844c007a3cbc324bcbaf3db90caf6910b7e9b638f.jpg)

![080793eaa90c9548b431bc939c9b98f0d8edfa49057625bf90d64563062964fc.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/080793eaa90c9548b431bc939c9b98f0d8edfa49057625bf90d64563062964fc.jpg)

![130a43bda3574209856ea8e81305743d90e2d578ccf391152c2407d5867db252.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/130a43bda3574209856ea8e81305743d90e2d578ccf391152c2407d5867db252.jpg)

![24ccc5536fc70f4942d961c559c6e815038969dc30a9536cd13910345a2389e3.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/24ccc5536fc70f4942d961c559c6e815038969dc30a9536cd13910345a2389e3.jpg)

![2d9dcc61f82bc0e0b6b55c8963495218b710f5e8c6578749e379fcb3366ec419.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/2d9dcc61f82bc0e0b6b55c8963495218b710f5e8c6578749e379fcb3366ec419.jpg)

![356bd293ae4a9133bb06d208b7f4e6442f33d12940de5ea06d23723866759d76.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/356bd293ae4a9133bb06d208b7f4e6442f33d12940de5ea06d23723866759d76.jpg)

![48cf4b572872ed741811465da98c943c96b32fe5f40f18cc5702873b9d0ec8f6.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/48cf4b572872ed741811465da98c943c96b32fe5f40f18cc5702873b9d0ec8f6.jpg)

![59288864c60062530776e556ca4769eaee5b9ed344c8dab82bd2e82543cb713d.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/59288864c60062530776e556ca4769eaee5b9ed344c8dab82bd2e82543cb713d.jpg)

![5c684a1d8c23897fc5ce78fca190723a0449b22c4a06edab24b2ed524a55c1ab.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/5c684a1d8c23897fc5ce78fca190723a0449b22c4a06edab24b2ed524a55c1ab.jpg)

![5dd7d79696d999eb692dab233afe084ce2d086d18fa84b02ecbfff45abd8a6a4.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/5dd7d79696d999eb692dab233afe084ce2d086d18fa84b02ecbfff45abd8a6a4.jpg)

![68ecd8d42e5afdf33c3ae2c3bef35999e1b81ce990074433cf3a97bf19c21e8e.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/68ecd8d42e5afdf33c3ae2c3bef35999e1b81ce990074433cf3a97bf19c21e8e.jpg)

![89a5140e4b8a9f297331c6b2191a54d6c1a696dd47fd86d864e1c641347fc39a.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/89a5140e4b8a9f297331c6b2191a54d6c1a696dd47fd86d864e1c641347fc39a.jpg)

![a933cb0a1ea8a6e785b6ad28a1545d8cba489f82a203e71f73887c58617532f0.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/a933cb0a1ea8a6e785b6ad28a1545d8cba489f82a203e71f73887c58617532f0.jpg)

![b6e6d30cf4dcdb531072c8cbb3abc0d1a4d8d51ee600aeba1c1051d754fdda58.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/b6e6d30cf4dcdb531072c8cbb3abc0d1a4d8d51ee600aeba1c1051d754fdda58.jpg)

![e35f84aefdc4ea982587b6fa51432a98ab854c4f77cb444fc228c3f0fde5c77a.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/e35f84aefdc4ea982587b6fa51432a98ab854c4f77cb444fc228c3f0fde5c77a.jpg)

![e6e27430a12f10629e74b9e16a4ed68c986ad51ef261a78c34295e32e7eb5d8d.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/e6e27430a12f10629e74b9e16a4ed68c986ad51ef261a78c34295e32e7eb5d8d.jpg)

![f940797196e6001dfa5acaa57c660e028f8c849380506bd36bf9e9adb375348b.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/images/f940797196e6001dfa5acaa57c660e028f8c849380506bd36bf9e9adb375348b.jpg)

### Tables

![28778a42964834325804c0165cc8dfa47b24b4d6607c6f6469a1898d85c5a939.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/28778a42964834325804c0165cc8dfa47b24b4d6607c6f6469a1898d85c5a939.jpg)

![36db9f49826bf8c03ca2437b10e90d29f8e1abc52d2ad3a70e00d3b24d6e8f05.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/36db9f49826bf8c03ca2437b10e90d29f8e1abc52d2ad3a70e00d3b24d6e8f05.jpg)

![463c14bb6b855d29bcd5903585e049bb21ca4b0ded20b63a72842fd45b06c819.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/463c14bb6b855d29bcd5903585e049bb21ca4b0ded20b63a72842fd45b06c819.jpg)

![50a1553f1cf398f3311b405888d3a843ef661458fdb30be94674bcc560931d6e.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/50a1553f1cf398f3311b405888d3a843ef661458fdb30be94674bcc560931d6e.jpg)

![85ef413392225b5880109b82ed90ae4b9690fe07bb06fd265562d7a44a0c7ba9.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/85ef413392225b5880109b82ed90ae4b9690fe07bb06fd265562d7a44a0c7ba9.jpg)

![9f550b713eb3ee08b44b5d0a41b50fcc40d904629b6b78fb36915343eb43b735.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/9f550b713eb3ee08b44b5d0a41b50fcc40d904629b6b78fb36915343eb43b735.jpg)

![a35d280dde7fc476575290772a6df640b681d04334ce20fad933cce7fa69339e.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/a35d280dde7fc476575290772a6df640b681d04334ce20fad933cce7fa69339e.jpg)

![b2b79c8293301a8579e8ddcfa324538021587e3f1d17a80efcd8f74923af2ef8.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/b2b79c8293301a8579e8ddcfa324538021587e3f1d17a80efcd8f74923af2ef8.jpg)

![ce27ea0f3494ccb38b6107fa1c2436302b8488b7b35ae7fef0593ba2e0451f50.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/ce27ea0f3494ccb38b6107fa1c2436302b8488b7b35ae7fef0593ba2e0451f50.jpg)

![ddc5f2bd27282e2a62b1743bd78b65b737ca5883a8d6ae0a5ce184c5b05d46a2.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/ddc5f2bd27282e2a62b1743bd78b65b737ca5883a8d6ae0a5ce184c5b05d46a2.jpg)

![e061ba6d9ecbb8a67e7d093d51bb0c6832b752c3a2f608d58490b89cf8ea7783.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/e061ba6d9ecbb8a67e7d093d51bb0c6832b752c3a2f608d58490b89cf8ea7783.jpg)

![f423fb4a70e3b9ee2b28aaa4e198f3b21e6cfca41e7b6e3c726da9bb0f776e5c.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/f423fb4a70e3b9ee2b28aaa4e198f3b21e6cfca41e7b6e3c726da9bb0f776e5c.jpg)

![f6fa6490ecd0a5886f5aca8ac975d7975c6663c61b5a98298cfa97acb0a5ef01.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/f6fa6490ecd0a5886f5aca8ac975d7975c6663c61b5a98298cfa97acb0a5ef01.jpg)

![f7366a4a835c810bd29785a21d3fd03d032ea00d47709ecc74ff8ee74eb07ad3.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/f7366a4a835c810bd29785a21d3fd03d032ea00d47709ecc74ff8ee74eb07ad3.jpg)

![ff04a1aab39690b7884c9f52dff240c5cf296b99874e5dac130723d545a75fc5.jpg](../icml_results/494_EgoTactile_%20Learning%20Grasp%20Pressure%20for%20Everyday%20Objects%20from%20Egocentric%20Video/tables/ff04a1aab39690b7884c9f52dff240c5cf296b99874e5dac130723d545a75fc5.jpg)

<a id="paper-0486"></a>

## CoEvol-NO: State and Coordinate Co-Evolution with an Error-Driven Predictor-Corrector Paradigm for Neural Operator Transformer


### Images

![17bf5bb371284067d3acbe1617bc5e298ba14b695a7451d1abbaa5789ab2e411.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/17bf5bb371284067d3acbe1617bc5e298ba14b695a7451d1abbaa5789ab2e411.jpg)

![1ae7c9d5ff1ebb2d6c417dc2035cfbd2c1fd589843db835197906a1eb6ccd950.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/1ae7c9d5ff1ebb2d6c417dc2035cfbd2c1fd589843db835197906a1eb6ccd950.jpg)

![4b79cef6dd4727ff80d60122f1fb08afd5f480732fcfae663c655b6cbfba034c.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/4b79cef6dd4727ff80d60122f1fb08afd5f480732fcfae663c655b6cbfba034c.jpg)

![88a8565cc874ab127866229391fc61615e0a30710c45fccc81d65a7de83084d0.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/88a8565cc874ab127866229391fc61615e0a30710c45fccc81d65a7de83084d0.jpg)

![be36cbc69db205a7a239fe3ffecb52c04f85483adc08424b3a3e141c1dd74fa5.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/be36cbc69db205a7a239fe3ffecb52c04f85483adc08424b3a3e141c1dd74fa5.jpg)

![d3706f4d4fcbdc0ecf70248ee24dd798ae03b0b42024a9916693a9b8ecb8e053.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/d3706f4d4fcbdc0ecf70248ee24dd798ae03b0b42024a9916693a9b8ecb8e053.jpg)

![db678cdaef3696501dabc8680a21abb2cac51773fc2e08f78d0638f5fdb282f2.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/db678cdaef3696501dabc8680a21abb2cac51773fc2e08f78d0638f5fdb282f2.jpg)

![e9c3453c1e93da3ed37d90dfe99ebe2183738f3378489e08d5b1286d7fbd44ad.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/images/e9c3453c1e93da3ed37d90dfe99ebe2183738f3378489e08d5b1286d7fbd44ad.jpg)

### Tables

![04a94ffedeb8dd145e584bd087f42631f11854f1700ed9874f820d5308c236aa.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/04a94ffedeb8dd145e584bd087f42631f11854f1700ed9874f820d5308c236aa.jpg)

![09737d56309f7dfee8cd2720cab3c5ba6c4f87c918b5738e6ee6edcc95ec3cfe.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/09737d56309f7dfee8cd2720cab3c5ba6c4f87c918b5738e6ee6edcc95ec3cfe.jpg)

![1a5e74d4a8285971e1ecc2a71d73904da17c6c383ae56080810ccd405bf9d333.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/1a5e74d4a8285971e1ecc2a71d73904da17c6c383ae56080810ccd405bf9d333.jpg)

![1dcedc30eb81331fc73c47e860d1fbe5d2fbfe8a3fd7be6b527cef25d87f151e.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/1dcedc30eb81331fc73c47e860d1fbe5d2fbfe8a3fd7be6b527cef25d87f151e.jpg)

![3ad7ee5d0d5abba186e9b5002236917655610fdaec2bc0ae9b15acbdb0eefdaa.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/3ad7ee5d0d5abba186e9b5002236917655610fdaec2bc0ae9b15acbdb0eefdaa.jpg)

![4dd50933e9d7ad28adef333f6397885b0b5f6f72fe9dac41c717ea5ea16de312.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/4dd50933e9d7ad28adef333f6397885b0b5f6f72fe9dac41c717ea5ea16de312.jpg)

![5114867e6c3c2cdc57f76bf2f0a8c6825887fb372c2d108f0631ffe53b8c2ae9.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/5114867e6c3c2cdc57f76bf2f0a8c6825887fb372c2d108f0631ffe53b8c2ae9.jpg)

![5ca969c9104e2f5dcb7c3b679d5fc4a743972539915384ba79ab06a0c410040d.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/5ca969c9104e2f5dcb7c3b679d5fc4a743972539915384ba79ab06a0c410040d.jpg)

![6773779846f977ffe6d23e05202d9cd9586f2063d66584532d29920618a0f8c8.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/6773779846f977ffe6d23e05202d9cd9586f2063d66584532d29920618a0f8c8.jpg)

![9d6889d921c8aae14251e86627f0b7b7bdb8086cfa457d8b1d6cc1d4f50ae0da.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/9d6889d921c8aae14251e86627f0b7b7bdb8086cfa457d8b1d6cc1d4f50ae0da.jpg)

![ae7ae8a523bcf9c748116351c9038f6cfbc49f2ba5eaeccbf45d475e26552ddf.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/ae7ae8a523bcf9c748116351c9038f6cfbc49f2ba5eaeccbf45d475e26552ddf.jpg)

![b12af81ca5894b8c5809ee570470a49e594e09b4b4a0ca5a07a050cc93b532db.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/b12af81ca5894b8c5809ee570470a49e594e09b4b4a0ca5a07a050cc93b532db.jpg)

![bf1c9b8065ee77b12bc4dc78e2f1ff2d3a57e4df06adde34314a7b56f483124d.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/bf1c9b8065ee77b12bc4dc78e2f1ff2d3a57e4df06adde34314a7b56f483124d.jpg)

![c30415517bf5d6838ac9a2711fabecd486a6f3042a3a774afdecdeb69b7db3f4.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/c30415517bf5d6838ac9a2711fabecd486a6f3042a3a774afdecdeb69b7db3f4.jpg)

![d30bbc2a6967aaaf31ec75e46236570d4d6afe086668df199f8ccee18ad9cee1.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/d30bbc2a6967aaaf31ec75e46236570d4d6afe086668df199f8ccee18ad9cee1.jpg)

![d3ecfa6a6bffae287462081e5719fcbaaef695cdbe9c589faa00d6449348610e.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/d3ecfa6a6bffae287462081e5719fcbaaef695cdbe9c589faa00d6449348610e.jpg)

![f764a742afa208b0c1f6be2953903de704398ea7433dde1c978505da27b622ce.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/f764a742afa208b0c1f6be2953903de704398ea7433dde1c978505da27b622ce.jpg)

![fb1208b4e193fb31f29a6463ec31cfeeee00ad88a5804b5308d47c72c4c34bee.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/fb1208b4e193fb31f29a6463ec31cfeeee00ad88a5804b5308d47c72c4c34bee.jpg)

![fe9e8e4edb426221ec17613ec49bed9247f09aedebaffd8e4a55a84b0395dfe1.jpg](../icml_results/495_CoEvol-NO_%20State%20and%20Coordinate%20Co-Evolution%20with%20an%20Error-Driven%20Predictor-Corrector%20Paradigm%20for%20N/tables/fe9e8e4edb426221ec17613ec49bed9247f09aedebaffd8e4a55a84b0395dfe1.jpg)

<a id="paper-0487"></a>

## 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models


### Images

![12fcc54ae75391be262195c1be66ae052e7cdf044625aaef402d16a60516e29d.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/12fcc54ae75391be262195c1be66ae052e7cdf044625aaef402d16a60516e29d.jpg)

![1927278581bc48b24b9a3ba162bb7ea381f326190378b8ba43a681afe8e18911.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/1927278581bc48b24b9a3ba162bb7ea381f326190378b8ba43a681afe8e18911.jpg)

![44085652b76e280df48f20847da69a1ffb582bb0601c3e6a408db56461b493ef.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/44085652b76e280df48f20847da69a1ffb582bb0601c3e6a408db56461b493ef.jpg)

![47b5c96483aee89b6ae65882217787895536c1aebd28afdc76b1d0fa2a756b55.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/47b5c96483aee89b6ae65882217787895536c1aebd28afdc76b1d0fa2a756b55.jpg)

![7d2882c10f883cddce1f510d386acb7fb01c4369a763c2f3e4c66f9757b9cb8e.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/7d2882c10f883cddce1f510d386acb7fb01c4369a763c2f3e4c66f9757b9cb8e.jpg)

![a24da5298b998f255dfd8074680ed3bba7c79ac546e809f28126edd852f27ee7.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/a24da5298b998f255dfd8074680ed3bba7c79ac546e809f28126edd852f27ee7.jpg)

![a9722045a7e7b936ce5c297cdb9a1db21cd2bcb4bb04bad0d4616410b302d884.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/a9722045a7e7b936ce5c297cdb9a1db21cd2bcb4bb04bad0d4616410b302d884.jpg)

![bc128057d727b52e9407c6e62b30a5eafc02bd7f905f5711d1fbf67cb35d8990.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/bc128057d727b52e9407c6e62b30a5eafc02bd7f905f5711d1fbf67cb35d8990.jpg)

![c3b5ab70aa8388cc006260cecd3579c7ae8e01e8b42cca242bd1a71ffa812f46.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/c3b5ab70aa8388cc006260cecd3579c7ae8e01e8b42cca242bd1a71ffa812f46.jpg)

![d153775bf6a70a0550607416c6bbed62ef83e4aecc414c7f32f928137b4ebbe0.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/d153775bf6a70a0550607416c6bbed62ef83e4aecc414c7f32f928137b4ebbe0.jpg)

![dd9282072db4775e5638f38f381c38294d7a15fa5d046453a2cd065a8e6d4cd4.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/images/dd9282072db4775e5638f38f381c38294d7a15fa5d046453a2cd065a8e6d4cd4.jpg)

### Tables

![082708544175efa09f0eb0c70f221b8f08a23c08159f3fbe9fed95a129ee9d69.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/082708544175efa09f0eb0c70f221b8f08a23c08159f3fbe9fed95a129ee9d69.jpg)

![2ac2d6da703ebbf502c8898788398f2a3670cdfef72e121c93d470a88e613a4a.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/2ac2d6da703ebbf502c8898788398f2a3670cdfef72e121c93d470a88e613a4a.jpg)

![4bb4f6f91fbd3feb40f2be01aa6e3540792a0845adb3320c5646b3d79e89526c.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/4bb4f6f91fbd3feb40f2be01aa6e3540792a0845adb3320c5646b3d79e89526c.jpg)

![51d1fb48b5aed548335a0d7921b8474dd68599489bae5df73015aa3cb12ca7fe.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/51d1fb48b5aed548335a0d7921b8474dd68599489bae5df73015aa3cb12ca7fe.jpg)

![5c5e6cf7a69744d75237248d1ca19f3d7ec1682778cb563addf5d6ab1a7c5669.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/5c5e6cf7a69744d75237248d1ca19f3d7ec1682778cb563addf5d6ab1a7c5669.jpg)

![74a3c1edecf4e6c47b888558d625f2f862f42ab5e89aba6da82e6a3506cea3ad.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/74a3c1edecf4e6c47b888558d625f2f862f42ab5e89aba6da82e6a3506cea3ad.jpg)

![a0002dd7978972cfd3d38ebb1b9b772a9c6c1080e8052732c5aa0d5d010fbe6f.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/a0002dd7978972cfd3d38ebb1b9b772a9c6c1080e8052732c5aa0d5d010fbe6f.jpg)

![badf4e404de1566b7dab34467192556f21e152f10f865eb7bcba4e150bb5da17.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/badf4e404de1566b7dab34467192556f21e152f10f865eb7bcba4e150bb5da17.jpg)

![ce9d16a283098418bc3712d1f5ffb21deec107e1a22600a4223d2e099de32487.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/ce9d16a283098418bc3712d1f5ffb21deec107e1a22600a4223d2e099de32487.jpg)

![d499b750fbca54c42199a36f45008803d321c8f8416037071d3e4513294e9391.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/d499b750fbca54c42199a36f45008803d321c8f8416037071d3e4513294e9391.jpg)

![f3da83113d72636a09833663d1892851d1855742631b2f9c6e3c9eaa45e58c69.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/f3da83113d72636a09833663d1892851d1855742631b2f9c6e3c9eaa45e58c69.jpg)

![fb4b7dfbb4bdbc8bf905cdf88740286b059bbc632284ced48f4a644a1c82f8d5.jpg](../icml_results/496_3ViewSense_%20Spatial%20and%20Mental%20Perspective%20Reasoning%20from%20Orthographic%20Views%20in%20Vision-Language%20Mode/tables/fb4b7dfbb4bdbc8bf905cdf88740286b059bbc632284ced48f4a644a1c82f8d5.jpg)

<a id="paper-0488"></a>

## Failure-Driven Workflow Refinement


### Images

![0f8e5eabd9d08321c36244789a57fa9ff8f13d2c8acee208e38f65e1e4447d79.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/0f8e5eabd9d08321c36244789a57fa9ff8f13d2c8acee208e38f65e1e4447d79.jpg)

![171834ec07cfcd7ba1672e36e301fa475254bd5a5e8ecc823beeb2540caa04ad.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/171834ec07cfcd7ba1672e36e301fa475254bd5a5e8ecc823beeb2540caa04ad.jpg)

![6105f3b20b17fb730b46ccc3798e60bb08c8822c3d3af8d784daf04daf7efcd3.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/6105f3b20b17fb730b46ccc3798e60bb08c8822c3d3af8d784daf04daf7efcd3.jpg)

![82d50f36bb3232a9b02bf1db4d6f44cd31026d56ea7e4cf55e601837b38612da.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/82d50f36bb3232a9b02bf1db4d6f44cd31026d56ea7e4cf55e601837b38612da.jpg)

![93a73957f745196dc15fa894d5c1a4731f197d240690ceb42c9cce822d8d3711.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/93a73957f745196dc15fa894d5c1a4731f197d240690ceb42c9cce822d8d3711.jpg)

![a8e7222c5a24277e4ad3ff6ae7fad9e31bea22ea1f49b1c3b2c53864db6b19e9.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/a8e7222c5a24277e4ad3ff6ae7fad9e31bea22ea1f49b1c3b2c53864db6b19e9.jpg)

![ab9482c407f96436185af6f56b467e01160fecaac51d346b539e9cf891d1b5ff.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/ab9482c407f96436185af6f56b467e01160fecaac51d346b539e9cf891d1b5ff.jpg)

![b3bd44474515eacbea3741c1078f086bddf6d426ff3d5571421d6dcaceee5435.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/images/b3bd44474515eacbea3741c1078f086bddf6d426ff3d5571421d6dcaceee5435.jpg)

### Tables

![0d8cce3dc9b4a058ca13944c709fec6cd60c69f1981e5dbf30dc0ef27a279419.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/0d8cce3dc9b4a058ca13944c709fec6cd60c69f1981e5dbf30dc0ef27a279419.jpg)

![2549e472fa8c1023d5700139c6724400f077411fe3c3b15afa5de67e67d7ae79.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/2549e472fa8c1023d5700139c6724400f077411fe3c3b15afa5de67e67d7ae79.jpg)

![34f36f713e00adf84764d64a6292cdcb77d561d6d33814dfc5575761d1e2afd2.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/34f36f713e00adf84764d64a6292cdcb77d561d6d33814dfc5575761d1e2afd2.jpg)

![3a09ed4e8c0f92ba38238755795910e9704252df4d7269c8a1c597b626e4626b.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/3a09ed4e8c0f92ba38238755795910e9704252df4d7269c8a1c597b626e4626b.jpg)

![3b9fcc973c95d4c87d69c5995105517b8956370f0fd31044bd00b7a65652182e.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/3b9fcc973c95d4c87d69c5995105517b8956370f0fd31044bd00b7a65652182e.jpg)

![51a3c19a83b18f015f693dfe2e027b6f1c170e0149792f9c2dd5008ca6bf4ec1.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/51a3c19a83b18f015f693dfe2e027b6f1c170e0149792f9c2dd5008ca6bf4ec1.jpg)

![b89009df5b7cd942ab91508fcbf6317b0c92dc2e4c0a059e0049dba79bea4e10.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/b89009df5b7cd942ab91508fcbf6317b0c92dc2e4c0a059e0049dba79bea4e10.jpg)

![c0f01c81d298e6ad10dcdf48b6fa65083e34bbbf4fdd8f34edc6061d0b649e3e.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/c0f01c81d298e6ad10dcdf48b6fa65083e34bbbf4fdd8f34edc6061d0b649e3e.jpg)

![df9d8814fa39f2255ac38c75e5f662b42035c2debb3c35e6a6d8d24d6d9f2a81.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/df9d8814fa39f2255ac38c75e5f662b42035c2debb3c35e6a6d8d24d6d9f2a81.jpg)

![f60a9c23aca918f8eb5d5b181a900831e11f94472e46a99cba27935e1416ccdd.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/f60a9c23aca918f8eb5d5b181a900831e11f94472e46a99cba27935e1416ccdd.jpg)

![fb24e42bafe1f07693e07f735ad0121c033ceb2c58487e0865bff37baa1e6a1a.jpg](../icml_results/497_Failure-Driven%20Workflow%20Refinement/tables/fb24e42bafe1f07693e07f735ad0121c033ceb2c58487e0865bff37baa1e6a1a.jpg)

<a id="paper-0489"></a>

## DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making


### Images

![3d490ef946f748ea7049086c018a20f2e5e0f52566aaa2a50c4a299d9a107ec9.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/3d490ef946f748ea7049086c018a20f2e5e0f52566aaa2a50c4a299d9a107ec9.jpg)

![515b927a9b63d18df0a7caaa453ef5d7b3b264463869a7b3139942b861389be5.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/515b927a9b63d18df0a7caaa453ef5d7b3b264463869a7b3139942b861389be5.jpg)

![5b36b6f785e36470b048f65ee71d879dfa6f0e2de225af525090e3856028cee1.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/5b36b6f785e36470b048f65ee71d879dfa6f0e2de225af525090e3856028cee1.jpg)

![6dd8c72905132d2c279811bb1a758481486c3513c52c5ffe7b16786775962f4d.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/6dd8c72905132d2c279811bb1a758481486c3513c52c5ffe7b16786775962f4d.jpg)

![6efd77bf6c435e89ed48a3f9a98156cbcf8808bf5ed1d64f26855cd3ba9412e8.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/6efd77bf6c435e89ed48a3f9a98156cbcf8808bf5ed1d64f26855cd3ba9412e8.jpg)

![88ee29ee80b5173fd087b9d1aa3ae351d9cc0fb551813c479457f463cd49a55a.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/88ee29ee80b5173fd087b9d1aa3ae351d9cc0fb551813c479457f463cd49a55a.jpg)

![8a1dbaeda35dec52ff9d5aa483210c0df0e35c6f52e4d0464ab3f94682c08caa.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/8a1dbaeda35dec52ff9d5aa483210c0df0e35c6f52e4d0464ab3f94682c08caa.jpg)

![d2d40ac65c062b2cbd719e992ee4f4ae9670e904cae0176b9124745a55e5de99.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/d2d40ac65c062b2cbd719e992ee4f4ae9670e904cae0176b9124745a55e5de99.jpg)

![dc5c9389f6abae7db58cf76a64f9b0cf53626e6ef027aa8d144906b04cca5cc1.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/dc5c9389f6abae7db58cf76a64f9b0cf53626e6ef027aa8d144906b04cca5cc1.jpg)

![e55a96757a823952551c656e841b00a103407b0e8d30d82f4def1da5ad86974b.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/e55a96757a823952551c656e841b00a103407b0e8d30d82f4def1da5ad86974b.jpg)

![f34cbb1461055e8ecd281acc78d8a152c9b8098d80d582a55dba09ee5e9e8edc.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/images/f34cbb1461055e8ecd281acc78d8a152c9b8098d80d582a55dba09ee5e9e8edc.jpg)

### Tables

![20a4bb0a5f931466eeacd2eaf47af2d5e927547938f328d24491c97b5928df37.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/20a4bb0a5f931466eeacd2eaf47af2d5e927547938f328d24491c97b5928df37.jpg)

![2b0fa9919e2fd5154d0a9fcd1d4d23cc8106d7b0f325174a4cb6dfd05de8ce53.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/2b0fa9919e2fd5154d0a9fcd1d4d23cc8106d7b0f325174a4cb6dfd05de8ce53.jpg)

![3def5c11ef74cfae5622aa041e986e6dfcd9f98710a94758c2046f5f53316a84.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/3def5c11ef74cfae5622aa041e986e6dfcd9f98710a94758c2046f5f53316a84.jpg)

![8945a8115995b9326736019ce9dfcf02bad703a18ca44c6bc12f6d2a6a7f2a0b.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/8945a8115995b9326736019ce9dfcf02bad703a18ca44c6bc12f6d2a6a7f2a0b.jpg)

![9d42af17f82abe43349b2655985a81f8f813f68d1d4068e548d11108679fa323.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/9d42af17f82abe43349b2655985a81f8f813f68d1d4068e548d11108679fa323.jpg)

![b65f1d94e9b6a6b494417af2cddea29a3b57d65a99e72658ed6d5b4dd44027e6.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/b65f1d94e9b6a6b494417af2cddea29a3b57d65a99e72658ed6d5b4dd44027e6.jpg)

![ef3af6c7642fcc65fc4776d36030f7dc68b94b1f7cdd6dbd762f66aef6979431.jpg](../icml_results/498_DLM_%20Unified%20Decision%20Language%20Model%20for%20Offline%20Multi-Agent%20Sequential%20Decision%20Making/tables/ef3af6c7642fcc65fc4776d36030f7dc68b94b1f7cdd6dbd762f66aef6979431.jpg)

<a id="paper-0490"></a>

## The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models


### Images

![04b1906d138baff214dde5c8d7c9de57e88520bcd69a3eea0edf822e94aca475.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/04b1906d138baff214dde5c8d7c9de57e88520bcd69a3eea0edf822e94aca475.jpg)

![099f1fe7f91cae66993561c19b1b354fd9507d377c0ab1fc78413f2cd31c0173.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/099f1fe7f91cae66993561c19b1b354fd9507d377c0ab1fc78413f2cd31c0173.jpg)

![0c3ba7895c01a9e4592431b8baa6b13f3f0846691c9568d6eaad7cab8c38d935.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/0c3ba7895c01a9e4592431b8baa6b13f3f0846691c9568d6eaad7cab8c38d935.jpg)

![0d4814d575ac1764b4eaff3d04cd88c1d7ff2bb40fc24eed29e0b9c91194770a.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/0d4814d575ac1764b4eaff3d04cd88c1d7ff2bb40fc24eed29e0b9c91194770a.jpg)

![150ae74a28bbaff532c298c5ed527d5122fd3f30cd069d7a26e5afcc7180f141.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/150ae74a28bbaff532c298c5ed527d5122fd3f30cd069d7a26e5afcc7180f141.jpg)

![1c1e99928ab2d2ca8a2a6295710bd091fb53bcdeb92910806c1c78dc3ad847bd.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/1c1e99928ab2d2ca8a2a6295710bd091fb53bcdeb92910806c1c78dc3ad847bd.jpg)

![1f7f3448592aedf7b3c29a399346bd65f58c8a8a01f2c823e016e09f457a25f6.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/1f7f3448592aedf7b3c29a399346bd65f58c8a8a01f2c823e016e09f457a25f6.jpg)

![1fa4c7d307a50c2a7b65680afd9d3430e0e41d08365ddfb34e5760564477ab99.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/1fa4c7d307a50c2a7b65680afd9d3430e0e41d08365ddfb34e5760564477ab99.jpg)

![238f96fcce2ba1e4421317313124a19344617f712b48002cb371f5b27e76c1d4.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/238f96fcce2ba1e4421317313124a19344617f712b48002cb371f5b27e76c1d4.jpg)

![2742df1640f234f125f4fc31bd0929e82e4a1ca3971ced12c95f88cb77b5b07a.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/2742df1640f234f125f4fc31bd0929e82e4a1ca3971ced12c95f88cb77b5b07a.jpg)

![28f609aa36edecb0d8419bf8c9c433d0a176770352dea51de9ac21b934db094a.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/28f609aa36edecb0d8419bf8c9c433d0a176770352dea51de9ac21b934db094a.jpg)

![294b05c53e58739998dea93f2f7db23d3eddc5c1aeced1a122b5e66e27672cf9.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/294b05c53e58739998dea93f2f7db23d3eddc5c1aeced1a122b5e66e27672cf9.jpg)

![2c595c793aa1d09be427e44efcd82282d12378c1c81cb2a35225cc0fb082a27c.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/2c595c793aa1d09be427e44efcd82282d12378c1c81cb2a35225cc0fb082a27c.jpg)

![2d02cf85356da82d9b252c6ba31943f954771a0df578d66beaebfe09db5ec4cb.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/2d02cf85356da82d9b252c6ba31943f954771a0df578d66beaebfe09db5ec4cb.jpg)

![36131b66b6db57824098049a9d24d1d0fb15726d0f4576eb7bc6627e181ab5c5.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/36131b66b6db57824098049a9d24d1d0fb15726d0f4576eb7bc6627e181ab5c5.jpg)

![39e420499d28cc8dfdb1ead2d74d4dc611bcd3f61f775da7342605b8a6d2d5c5.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/39e420499d28cc8dfdb1ead2d74d4dc611bcd3f61f775da7342605b8a6d2d5c5.jpg)

![48f7e6dd3e47ff74e96a1d651c8126d0c37ce0f588d4543d1b9c0381165fc33f.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/48f7e6dd3e47ff74e96a1d651c8126d0c37ce0f588d4543d1b9c0381165fc33f.jpg)

![490f0ea7d8ea9e067ca0259e59f34bdd77d53969febc6b4e58525177fd537c90.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/490f0ea7d8ea9e067ca0259e59f34bdd77d53969febc6b4e58525177fd537c90.jpg)

![4971ef47196d08073a26a3b8125537f328c9c0d81d6aced6059d88887876e479.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/4971ef47196d08073a26a3b8125537f328c9c0d81d6aced6059d88887876e479.jpg)

![4be817090a23d3f52ef71f24029e24d869e8cedce4761a9aa070f60a4f4e8301.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/4be817090a23d3f52ef71f24029e24d869e8cedce4761a9aa070f60a4f4e8301.jpg)

![4c16912b2a4fb84af30b0fbd2525136ccf3e408d1a416eea0ba0ef40dda6067a.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/4c16912b2a4fb84af30b0fbd2525136ccf3e408d1a416eea0ba0ef40dda6067a.jpg)

![600c58b7bac16ba7895a56be77f6963ff19888c62961335c8f3f272624c60c73.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/600c58b7bac16ba7895a56be77f6963ff19888c62961335c8f3f272624c60c73.jpg)

![6613fd320bc6a7dd6b8f9d5d63edf8195c47734d4cc09298e804078236bfc8c6.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/6613fd320bc6a7dd6b8f9d5d63edf8195c47734d4cc09298e804078236bfc8c6.jpg)

![6ed1b0adb0aa857f282e0c9bc6525d7d65e7cacd1f36051a5a960e6360c8ff2c.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/6ed1b0adb0aa857f282e0c9bc6525d7d65e7cacd1f36051a5a960e6360c8ff2c.jpg)

![6fbd4366b1b9239bbfcaa021387d643803ef8ac137e4b07cb26e389d14893130.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/6fbd4366b1b9239bbfcaa021387d643803ef8ac137e4b07cb26e389d14893130.jpg)

![7e0b6afc782d78fe238da00355d48de7ce3b9a1eb413c565ca6940b5c43aec5f.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/7e0b6afc782d78fe238da00355d48de7ce3b9a1eb413c565ca6940b5c43aec5f.jpg)

![83d69be7cf5ad9a38f7314b99771ba254be73a0fe8d5e18676db4e2ca719d9dc.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/83d69be7cf5ad9a38f7314b99771ba254be73a0fe8d5e18676db4e2ca719d9dc.jpg)

![95245439d0926bbcb2f1043126bd78762790679dcd733d0683a8d6459d2001b3.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/95245439d0926bbcb2f1043126bd78762790679dcd733d0683a8d6459d2001b3.jpg)

![95d1246b1832eedf19549e3035762e5bd30ed0743d1b8f46668c08b1bdd69306.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/95d1246b1832eedf19549e3035762e5bd30ed0743d1b8f46668c08b1bdd69306.jpg)

![9639486b19e9b54768ab4b98ac7496a534a596ced24b7e819c668ce86e020e0f.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/9639486b19e9b54768ab4b98ac7496a534a596ced24b7e819c668ce86e020e0f.jpg)

![a058d1c7d463673615145a71958cc1c986e7af1967da80a8368ebcaf5af83b8c.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/a058d1c7d463673615145a71958cc1c986e7af1967da80a8368ebcaf5af83b8c.jpg)

![a3ae378e8be16f7d63312f517f340fb93e59bf22dec33efcf057ce98f235cbc0.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/a3ae378e8be16f7d63312f517f340fb93e59bf22dec33efcf057ce98f235cbc0.jpg)

![a5552dc2ea5556fc1997b317ef10409fbdd919b1222a771719bc1c609d193185.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/a5552dc2ea5556fc1997b317ef10409fbdd919b1222a771719bc1c609d193185.jpg)

![a8e71dd750ebd9a67688b39f508b7d388d1bc8e127a6c4db03d9536112b6ade1.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/a8e71dd750ebd9a67688b39f508b7d388d1bc8e127a6c4db03d9536112b6ade1.jpg)

![aa246a97bd86950d263e7fe0fbc6e773e73bf3e492959ad486d19b7e2479a9c0.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/aa246a97bd86950d263e7fe0fbc6e773e73bf3e492959ad486d19b7e2479a9c0.jpg)

![acdc7dacf4a14eece3b27438ffedf64dd774099e2fa12d5ebe0d219c47ed73fe.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/acdc7dacf4a14eece3b27438ffedf64dd774099e2fa12d5ebe0d219c47ed73fe.jpg)

![b468ab06a23e07e85c3fefda6de318e61acc635af42c2e1a8121082f2c3c3529.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/b468ab06a23e07e85c3fefda6de318e61acc635af42c2e1a8121082f2c3c3529.jpg)

![bbc06350b3c4b7889e0318351e21bf491f25847c073ce5e1523810dca349b2b1.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/bbc06350b3c4b7889e0318351e21bf491f25847c073ce5e1523810dca349b2b1.jpg)

![ca24a8839f2a40cdbb858dece9009cc4a97e5058a6799c1f027cdc5d3a730645.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/ca24a8839f2a40cdbb858dece9009cc4a97e5058a6799c1f027cdc5d3a730645.jpg)

![cb44db1b6e486e336bc9ebf3e061b27c146e4abfaa961534a519f0ba5edf8746.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/cb44db1b6e486e336bc9ebf3e061b27c146e4abfaa961534a519f0ba5edf8746.jpg)

![d740d265568a7abf742daf5383570a2d8e3c45e7bbd3092bd3510ebac74ecc07.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/d740d265568a7abf742daf5383570a2d8e3c45e7bbd3092bd3510ebac74ecc07.jpg)

![d76b27101183cfed28c1cba882197f321f17235f0446ccd6d4cf749b622ef5f7.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/d76b27101183cfed28c1cba882197f321f17235f0446ccd6d4cf749b622ef5f7.jpg)

![d9359e2425dae2ecd3f7e051a699d86050375e97862ad1745520031f399d8336.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/d9359e2425dae2ecd3f7e051a699d86050375e97862ad1745520031f399d8336.jpg)

![da665b08e10146b56b3c1c19091624d634133ffe82b7db34f61af039b0a35052.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/da665b08e10146b56b3c1c19091624d634133ffe82b7db34f61af039b0a35052.jpg)

![e3b6bc1f1f779ce3f32bc8f07db755ffedc7bac51551fa7a50889e05a389fefb.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/e3b6bc1f1f779ce3f32bc8f07db755ffedc7bac51551fa7a50889e05a389fefb.jpg)

![f0d73aec1c1aa9c0144dae3f1f7f92646850a2339d71e87c8bf578acdc18ab8a.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/f0d73aec1c1aa9c0144dae3f1f7f92646850a2339d71e87c8bf578acdc18ab8a.jpg)

![f2ee8781bc94c4063e0d48bf91f9934978a29a133e4fde8ce85a43467f5e000c.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/f2ee8781bc94c4063e0d48bf91f9934978a29a133e4fde8ce85a43467f5e000c.jpg)

![f303b8e66896674cc259ae6f5061f0eb0120e1c936a3f89a86c04e9ac28ed6b5.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/f303b8e66896674cc259ae6f5061f0eb0120e1c936a3f89a86c04e9ac28ed6b5.jpg)

![f5a6f79b63a5f4044384f6ad4c7a175481492d7b30d70043fdc315d173b41323.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/f5a6f79b63a5f4044384f6ad4c7a175481492d7b30d70043fdc315d173b41323.jpg)

![fc7928d5645cf280ec4d953951dde0e2fc1b043bf255e7d3f51c010f46a6fc90.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/fc7928d5645cf280ec4d953951dde0e2fc1b043bf255e7d3f51c010f46a6fc90.jpg)

![fd7ee13280e3946a6af43462fe2f66d8587dd98f8b61ce46e6e2f725249ed1b3.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/images/fd7ee13280e3946a6af43462fe2f66d8587dd98f8b61ce46e6e2f725249ed1b3.jpg)

### Tables

![0b2fc568441ca82e7fc41f4d428b63776f5850140c9b061c17ae99372182c8ac.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/0b2fc568441ca82e7fc41f4d428b63776f5850140c9b061c17ae99372182c8ac.jpg)

![10108474544ef8bfe745287af19a1e709ff8eb214564c5ce8ab106a8c1a60653.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/10108474544ef8bfe745287af19a1e709ff8eb214564c5ce8ab106a8c1a60653.jpg)

![1ec2860d920a089241b37c591c8f88507ebec11cc0952897787932f8a269143e.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/1ec2860d920a089241b37c591c8f88507ebec11cc0952897787932f8a269143e.jpg)

![426c8638a49703358fee3689f8195aee7e0124f9e47b32842dfcc29fcab5a7bf.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/426c8638a49703358fee3689f8195aee7e0124f9e47b32842dfcc29fcab5a7bf.jpg)

![46380a5d8e2ccd4c7f00c1ff3501a108040a15799dbb001b610e0d904949d249.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/46380a5d8e2ccd4c7f00c1ff3501a108040a15799dbb001b610e0d904949d249.jpg)

![59ccf834f3d24784531c2593c9ab7d712d09f0c6c4a59d1dabe8881f420fe518.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/59ccf834f3d24784531c2593c9ab7d712d09f0c6c4a59d1dabe8881f420fe518.jpg)

![6e188cfef770538a05870935bfe999c417f75ba6a4deab52a611a3198b65c8e9.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/6e188cfef770538a05870935bfe999c417f75ba6a4deab52a611a3198b65c8e9.jpg)

![9601d38a1af33c65137670cf9f53e2b8b7ad27b71430d9c2159f63206d7d606e.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/9601d38a1af33c65137670cf9f53e2b8b7ad27b71430d9c2159f63206d7d606e.jpg)

![a2b9b55ee6c1f512ffee9c3a8e75eff003f5eb9a2d2ba4f66f4b963f8e983af6.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/a2b9b55ee6c1f512ffee9c3a8e75eff003f5eb9a2d2ba4f66f4b963f8e983af6.jpg)

![c471d9057666c67dbef43d2e49e9d64f9348b711cc5a11910cd94205759362f1.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/c471d9057666c67dbef43d2e49e9d64f9348b711cc5a11910cd94205759362f1.jpg)

![d0278bd7c4ddf60b628125a399a4e5d5b98afaa14bedd8e2c9833626b56d694f.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/d0278bd7c4ddf60b628125a399a4e5d5b98afaa14bedd8e2c9833626b56d694f.jpg)

![e1fa9210dbbb0e5ee66ea61ab0fcf3925f9a83ecac42cadba2d28149cb0e7083.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/e1fa9210dbbb0e5ee66ea61ab0fcf3925f9a83ecac42cadba2d28149cb0e7083.jpg)

![e400ca494cfb3fbf317769b94207d397efa727db7a267f897540a9dce742e820.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/e400ca494cfb3fbf317769b94207d397efa727db7a267f897540a9dce742e820.jpg)

![e7e0c4eaad398a3260e996b07fb3185cf728dbdf5de0bc087f2ecef4a399da17.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/e7e0c4eaad398a3260e996b07fb3185cf728dbdf5de0bc087f2ecef4a399da17.jpg)

![fc8600f659ca9a218d37fff5f59486fa7b394f4c5c1e58363ec41621cb63a407.jpg](../icml_results/499_The%20Tell-Tale%20Norm_%20%24_ell_2%24%20Magnitude%20as%20a%20Signal%20for%20Reasoning%20Dynamics%20in%20Large%20Language%20Models/tables/fc8600f659ca9a218d37fff5f59486fa7b394f4c5c1e58363ec41621cb63a407.jpg)

<a id="paper-0491"></a>

## UniMapping: Unified SLAM Framework for Map-Centric Embodied Perception


### Images

![1718be8ac2b8dbd40bd3a6d1b8f1b400099d93033a428048ae71bc16154c05ff.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/1718be8ac2b8dbd40bd3a6d1b8f1b400099d93033a428048ae71bc16154c05ff.jpg)

![2c849da4a7157357da84114d882a7ea0ca4a777d05b185c0e06868e885ce4308.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/2c849da4a7157357da84114d882a7ea0ca4a777d05b185c0e06868e885ce4308.jpg)

![5eb02804031d63aca0d6aaa46a88326c46a84e8336e16738a85801b0ce0a9748.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/5eb02804031d63aca0d6aaa46a88326c46a84e8336e16738a85801b0ce0a9748.jpg)

![6c44faa764a909fa15835dba5b09f2b104083557eb0b4ee6c2a3b389013202d7.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/6c44faa764a909fa15835dba5b09f2b104083557eb0b4ee6c2a3b389013202d7.jpg)

![9a81d2ed01839f477962bb04530f11f5e2d3295e978dac5034e972186c36b82f.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/9a81d2ed01839f477962bb04530f11f5e2d3295e978dac5034e972186c36b82f.jpg)

![b89ea5d3413b7d56301e1cc92d89e697d019d0323e6be270b5b13b7e73348c4d.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/b89ea5d3413b7d56301e1cc92d89e697d019d0323e6be270b5b13b7e73348c4d.jpg)

![cb9ac44bc8b1812acfb04125171950b7914cf643687aece3a46559b79f12ed96.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/cb9ac44bc8b1812acfb04125171950b7914cf643687aece3a46559b79f12ed96.jpg)

![d7694393419a0eaf19ac37e42e1e23c726e0e9e8beb208d0b670f267e01637fd.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/images/d7694393419a0eaf19ac37e42e1e23c726e0e9e8beb208d0b670f267e01637fd.jpg)

### Tables

![17791af788f7cb43cf402ad29b2acfded23204674fe18bf65dc2c2a1937d0c4b.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/17791af788f7cb43cf402ad29b2acfded23204674fe18bf65dc2c2a1937d0c4b.jpg)

![226f5fc58eed59344531d490dad2d535173eadf0b31c30f74c9303bb854e96de.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/226f5fc58eed59344531d490dad2d535173eadf0b31c30f74c9303bb854e96de.jpg)

![4838cfc42d8bfd103c258555d013663e9d57b47af586af65b1d0e5d0e8740dd6.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/4838cfc42d8bfd103c258555d013663e9d57b47af586af65b1d0e5d0e8740dd6.jpg)

![76c5251273762371aefe1bd7d78277722e8a0b95611028cbb84c5676b842d739.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/76c5251273762371aefe1bd7d78277722e8a0b95611028cbb84c5676b842d739.jpg)

![7e86a3a54dc4eaf58bf4672540ef11ed2908eb517a74dc961c907b4123810f01.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/7e86a3a54dc4eaf58bf4672540ef11ed2908eb517a74dc961c907b4123810f01.jpg)

![8cf0efc40c80ccc7c76c8eea322ef8f5ed98275896b98d13340498212e5affb4.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/8cf0efc40c80ccc7c76c8eea322ef8f5ed98275896b98d13340498212e5affb4.jpg)

![9c602c6d379995cf8ada2adb55b5858dfd6a6282dcba618e749c9976dd693ebd.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/9c602c6d379995cf8ada2adb55b5858dfd6a6282dcba618e749c9976dd693ebd.jpg)

![a4dcce31697a82c62223843a5d3c35dce52b8e87a877a8bd047af54a0a548233.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/a4dcce31697a82c62223843a5d3c35dce52b8e87a877a8bd047af54a0a548233.jpg)

![a55c9f5dcf964a5ded0056c3d807ba8accd68fb71c7915cfc82607e4435c9566.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/a55c9f5dcf964a5ded0056c3d807ba8accd68fb71c7915cfc82607e4435c9566.jpg)

![db0c3ac283050035f9faa5a465b8d8591f230f60d7feddb0e05c38a898323789.jpg](../icml_results/500_UniMapping_%20Unified%20SLAM%20Framework%20for%20Map-Centric%20Embodied%20Perception/tables/db0c3ac283050035f9faa5a465b8d8591f230f60d7feddb0e05c38a898323789.jpg)

<a id="paper-0492"></a>

## HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation


### Images

![3aba56e5449bf336f1b004661724c52bd9aa4ef237922c2db7533779d4b486ac.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/3aba56e5449bf336f1b004661724c52bd9aa4ef237922c2db7533779d4b486ac.jpg)

![5b1390dcfa8a7a92401c409bf91c5632024d0d169267080828bdd7e570f90a80.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/5b1390dcfa8a7a92401c409bf91c5632024d0d169267080828bdd7e570f90a80.jpg)

![6c06c9f6aab2ec54107747dc701fb4d3e0eee10e8d62d5740ecc44bade53334d.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/6c06c9f6aab2ec54107747dc701fb4d3e0eee10e8d62d5740ecc44bade53334d.jpg)

![b362ea4ce769d09d3a2c1e7b7fc553cd251690b756b48b28e6c20577c2815936.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/b362ea4ce769d09d3a2c1e7b7fc553cd251690b756b48b28e6c20577c2815936.jpg)

![c2786e7f4397f8884a792b69ec30985ea4542f7460ffd5b70e0dd4ccf638101c.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/c2786e7f4397f8884a792b69ec30985ea4542f7460ffd5b70e0dd4ccf638101c.jpg)

![d1d6e08a5bc0e6f34b5fce650e0246dcdf0c517ebaca6f960d32eb0db679d71d.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/d1d6e08a5bc0e6f34b5fce650e0246dcdf0c517ebaca6f960d32eb0db679d71d.jpg)

![e5df9ec211fd9fc064fa08b68cbd915fea72f39195ffe1f1e67d3439ed0900b9.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/images/e5df9ec211fd9fc064fa08b68cbd915fea72f39195ffe1f1e67d3439ed0900b9.jpg)

### Tables

![0794b44a43989ab3c82212d813a0eec08370cf0f9d4c136c4bdb5cef33ddf032.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/0794b44a43989ab3c82212d813a0eec08370cf0f9d4c136c4bdb5cef33ddf032.jpg)

![0aa3643a111284830e315d76944de5008c8be9498296cb10e2615f2281f82018.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/0aa3643a111284830e315d76944de5008c8be9498296cb10e2615f2281f82018.jpg)

![0ddffd25d169c982cc5e53fc1e47f0425788f0c1ab7161f7a74956a28a3d6b16.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/0ddffd25d169c982cc5e53fc1e47f0425788f0c1ab7161f7a74956a28a3d6b16.jpg)

![19edbaccc1274d2133a98ab112d0f1c94609b3431adadbbd0ed812a863f496ea.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/19edbaccc1274d2133a98ab112d0f1c94609b3431adadbbd0ed812a863f496ea.jpg)

![1d6557616c25f74443eceeb380c338540103f0748fb18e2dee0bc9f0ebdfdcc3.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/1d6557616c25f74443eceeb380c338540103f0748fb18e2dee0bc9f0ebdfdcc3.jpg)

![202528686a7e08f436b5453d9924f9f7e64423c76a9382b942d5b43d92897a7b.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/202528686a7e08f436b5453d9924f9f7e64423c76a9382b942d5b43d92897a7b.jpg)

![37f8ae0a88105fc885032e047e165c3cba8b0398bd66beab6119455ad0c7948b.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/37f8ae0a88105fc885032e047e165c3cba8b0398bd66beab6119455ad0c7948b.jpg)

![4c4ca2175b6027119c27eb5d2e9c284943a465f2d28bdd14b799aeb26b80dcd8.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/4c4ca2175b6027119c27eb5d2e9c284943a465f2d28bdd14b799aeb26b80dcd8.jpg)

![5217d6bf829b8721d0be09a256ca7f0480cd04f1101e59f39d36282f7fe08598.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/5217d6bf829b8721d0be09a256ca7f0480cd04f1101e59f39d36282f7fe08598.jpg)

![60135e63abdab410b3f7309a27e6e77ff55736b7d4abf93af11e66918076045b.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/60135e63abdab410b3f7309a27e6e77ff55736b7d4abf93af11e66918076045b.jpg)

![63167cff5c8bcbe12f56ab1b5355810b9af8a70942bdafbfd06c4f60b7582091.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/63167cff5c8bcbe12f56ab1b5355810b9af8a70942bdafbfd06c4f60b7582091.jpg)

![64578833a4582a0056ea96702593c6f416a30365f0b983739ec53a5ffe57ef41.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/64578833a4582a0056ea96702593c6f416a30365f0b983739ec53a5ffe57ef41.jpg)

![6a629fb7942e7c5e5fd148fd7aea0f0460da559122b65e23aad12c27e809eccb.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/6a629fb7942e7c5e5fd148fd7aea0f0460da559122b65e23aad12c27e809eccb.jpg)

![6caa84420a290790df9e257f5f9e3ab1d1916329d8c5c03d42ca8b298a3df211.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/6caa84420a290790df9e257f5f9e3ab1d1916329d8c5c03d42ca8b298a3df211.jpg)

![6dc926416dde7a3b4dc856b22c89b68d31f7d024a23c8f15e41eebc27e345163.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/6dc926416dde7a3b4dc856b22c89b68d31f7d024a23c8f15e41eebc27e345163.jpg)

![7f5815009dd9a5ec39c55e55277b45440fb8e71d91480deae00a4d95d5f7adbf.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/7f5815009dd9a5ec39c55e55277b45440fb8e71d91480deae00a4d95d5f7adbf.jpg)

![8262ed95946e72a4b1cd4178da5b72319a6768c95fe4f72d75b474f2aa9b4ecb.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/8262ed95946e72a4b1cd4178da5b72319a6768c95fe4f72d75b474f2aa9b4ecb.jpg)

![846bc9066f9af2c8dff63baf7caf4f07a891f46b6f56a7469561701657a3bc72.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/846bc9066f9af2c8dff63baf7caf4f07a891f46b6f56a7469561701657a3bc72.jpg)

![87774f3b446b354fc54cfe2c407027394bd8e5729e865d3de5936a092e3b7993.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/87774f3b446b354fc54cfe2c407027394bd8e5729e865d3de5936a092e3b7993.jpg)

![8d3f8f2f8cba036d810f89dce606b8aabbc9678c569229424c28cfcf9bf783c6.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/8d3f8f2f8cba036d810f89dce606b8aabbc9678c569229424c28cfcf9bf783c6.jpg)

![9a847a00b7d2fdb67d241145d759fcc0538c3fd68359b1f9c6899774043a698f.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/9a847a00b7d2fdb67d241145d759fcc0538c3fd68359b1f9c6899774043a698f.jpg)

![9eb3fba540e057beb7a80b33a20e42453f660790ba9e3a1ebf9b74d61f7b8abe.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/9eb3fba540e057beb7a80b33a20e42453f660790ba9e3a1ebf9b74d61f7b8abe.jpg)

![a389f86b452c1aafbb071202431487fc0e62f7c4a3ce552910887a60286c8275.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/a389f86b452c1aafbb071202431487fc0e62f7c4a3ce552910887a60286c8275.jpg)

![a99a642dd6ac5ef2f77c66ebc1afc327359ceac8326c7d88258ca1c80986e064.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/a99a642dd6ac5ef2f77c66ebc1afc327359ceac8326c7d88258ca1c80986e064.jpg)

![aadca95cba4db85cab98b4ab1a56851cfb2a0c70624ee74ef52a183e60a09592.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/aadca95cba4db85cab98b4ab1a56851cfb2a0c70624ee74ef52a183e60a09592.jpg)

![abb410006e7cec8b10082630004a85c2c980ee7c02a473eef23a7deb5986e2e8.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/abb410006e7cec8b10082630004a85c2c980ee7c02a473eef23a7deb5986e2e8.jpg)

![af057a43723b0115423d152941009c142bfa3d8ec9b13b3aa3c8798024d1ab91.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/af057a43723b0115423d152941009c142bfa3d8ec9b13b3aa3c8798024d1ab91.jpg)

![bae73a58b1b47a5e11a4ffd87e00b1bb30fc8cdbb320f643967308cfcbf7272c.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/bae73a58b1b47a5e11a4ffd87e00b1bb30fc8cdbb320f643967308cfcbf7272c.jpg)

![bb2c87b0ff8d4a17cd521cc0584b09b8a1977fbdb74b43f251e180f143cf30d4.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/bb2c87b0ff8d4a17cd521cc0584b09b8a1977fbdb74b43f251e180f143cf30d4.jpg)

![c0dd2ecebf181cd96828d143ba1dae9ebafe01ae51639e2900e701f75ecc7f97.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/c0dd2ecebf181cd96828d143ba1dae9ebafe01ae51639e2900e701f75ecc7f97.jpg)

![c746054287208b7d911b751dfdd2808da973b49cb68d30369260784618f1508c.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/c746054287208b7d911b751dfdd2808da973b49cb68d30369260784618f1508c.jpg)

![db367a0c5dcd89d4def2becc6c3f9c1222b355568f57e39ea40c6fc2da5adf94.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/db367a0c5dcd89d4def2becc6c3f9c1222b355568f57e39ea40c6fc2da5adf94.jpg)

![f2d08c4534c903aef4c63c52ec49bd7cba6ab866dda73bd4a227cdf019e43550.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/f2d08c4534c903aef4c63c52ec49bd7cba6ab866dda73bd4a227cdf019e43550.jpg)

![f696cf558211dbadef16dba3f99a9306960f4024c891e5c44f32490a2cf50ce3.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/f696cf558211dbadef16dba3f99a9306960f4024c891e5c44f32490a2cf50ce3.jpg)

![f98337a85309a4c4de9603f90d7c590ac2b451899a5328473da215558f7ddb8a.jpg](../icml_results/501_HELIX_%20Hybrid%20Encoding%20with%20Learnable%20Identity%20and%20Cross-dimensional%20Synthesis%20for%20Time%20Series%20Imput/tables/f98337a85309a4c4de9603f90d7c590ac2b451899a5328473da215558f7ddb8a.jpg)

<a id="paper-0493"></a>

## How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning


### Images

![150ea543e406866e94c86d8de560c29b3656160ffda1a16db90b375b743d4505.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/150ea543e406866e94c86d8de560c29b3656160ffda1a16db90b375b743d4505.jpg)

![16c3c1d696ce400b3748a3896264e2817a1d275c2b964968938527aa9cbdf2dd.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/16c3c1d696ce400b3748a3896264e2817a1d275c2b964968938527aa9cbdf2dd.jpg)

![1a7d0de022e5d487d4e138331bb12235cad8cce883f563a6ebfeda2591edd76f.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/1a7d0de022e5d487d4e138331bb12235cad8cce883f563a6ebfeda2591edd76f.jpg)

![1ed072f66517e945d47ebcf3a700fef2054c7be81b45f600d4b5964b2c98cb97.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/1ed072f66517e945d47ebcf3a700fef2054c7be81b45f600d4b5964b2c98cb97.jpg)

![300706b84b45c3085811903bbf24383302e7131392e0f292ef83aa5c4d9a0a87.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/300706b84b45c3085811903bbf24383302e7131392e0f292ef83aa5c4d9a0a87.jpg)

![45d2d350eaa6e2cca44d88b13162562889a7c7d3e35f997de8c258ecde44d83d.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/45d2d350eaa6e2cca44d88b13162562889a7c7d3e35f997de8c258ecde44d83d.jpg)

![5129392f7dff47f3b798f673df3564a40040fe52a11df3d2bcedad7a9dff477b.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/5129392f7dff47f3b798f673df3564a40040fe52a11df3d2bcedad7a9dff477b.jpg)

![53c70c816a23a7bf2fb1205204674e4990627989460017de7eaeabcce6729053.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/53c70c816a23a7bf2fb1205204674e4990627989460017de7eaeabcce6729053.jpg)

![614d07540ae9dfb4de0076c2b60306662b67cd707190842d835776fc3919761d.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/614d07540ae9dfb4de0076c2b60306662b67cd707190842d835776fc3919761d.jpg)

![651df6a75d5075c954f0893e8fcba4f22e003cd10fe717443fed70fdb6c06772.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/651df6a75d5075c954f0893e8fcba4f22e003cd10fe717443fed70fdb6c06772.jpg)

![8bddf7f3b0cff4180e7972c5945938c4feb70887a3c731e36556ccc81134e912.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/8bddf7f3b0cff4180e7972c5945938c4feb70887a3c731e36556ccc81134e912.jpg)

![90bcec3567313c9f02053dc546d6fd00a0f2787acd78e1ab9b904f52c8f73ac3.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/90bcec3567313c9f02053dc546d6fd00a0f2787acd78e1ab9b904f52c8f73ac3.jpg)

![d0eeeda1cc1a6406a002687a33a335061baa153ed1220675a92853db1a2065eb.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/d0eeeda1cc1a6406a002687a33a335061baa153ed1220675a92853db1a2065eb.jpg)

![d6f0b6f272df3c2ed8f43d446f9966a64cd6019adcf31404f3189f746589729f.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/d6f0b6f272df3c2ed8f43d446f9966a64cd6019adcf31404f3189f746589729f.jpg)

![e1842528b58aa4719539147a5857b93e88fc57da0117e9c014118d904402fb3b.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/e1842528b58aa4719539147a5857b93e88fc57da0117e9c014118d904402fb3b.jpg)

![f8c6f2bad09948708e1148566ea47b0b4e968f12293bad4ffd41400e474ac47c.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/f8c6f2bad09948708e1148566ea47b0b4e968f12293bad4ffd41400e474ac47c.jpg)

![ff8fd003b589dbacc750e27e5ed9be11f1d95273cdc1945249da6f88796e1e48.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/images/ff8fd003b589dbacc750e27e5ed9be11f1d95273cdc1945249da6f88796e1e48.jpg)

### Tables

![10812630c60590ab48a68d12056193fc8fddc07f0f97a7c9386bbda38f1266ac.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/10812630c60590ab48a68d12056193fc8fddc07f0f97a7c9386bbda38f1266ac.jpg)

![256fdf8fd0a5d619461e497f1e821f0eef46d2cff764204606c9a9c943f87def.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/256fdf8fd0a5d619461e497f1e821f0eef46d2cff764204606c9a9c943f87def.jpg)

![316147113b7c65408b7e4493d40a40501ae0484a786fc10972e534388e4a4a29.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/316147113b7c65408b7e4493d40a40501ae0484a786fc10972e534388e4a4a29.jpg)

![426d21fa08612d4bcfbff50d413f969ad65248f2c6cfbce46821fa236bd535c2.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/426d21fa08612d4bcfbff50d413f969ad65248f2c6cfbce46821fa236bd535c2.jpg)

![6115a761847f3c4d8475d65ec626aee984cad8a1e2ab845fc817d0ea0483c9f0.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/6115a761847f3c4d8475d65ec626aee984cad8a1e2ab845fc817d0ea0483c9f0.jpg)

![7c6bb7a93c3f542f2f0c38d4411da192e05fe2cc1ec569181cdc6a1825fe47af.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/7c6bb7a93c3f542f2f0c38d4411da192e05fe2cc1ec569181cdc6a1825fe47af.jpg)

![85027925a557e468d051795bf9e7fb15f5b68779d2d67f0141685e17ebe7e9e1.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/85027925a557e468d051795bf9e7fb15f5b68779d2d67f0141685e17ebe7e9e1.jpg)

![859d5548e4486712b460ec4d706eece7b4b0d073ff20d1346110f950e6c2273a.jpg](../icml_results/502_How%20RL%20Unlocks%20the%20Aha%20Moment%20in%20Geometric%20Interleaved%20Reasoning/tables/859d5548e4486712b460ec4d706eece7b4b0d073ff20d1346110f950e6c2273a.jpg)

<a id="paper-0494"></a>

## Prescriptive Scaling Reveals the Evolution of Language Model Capabilities


### Images

![0563e8505464876ac289bee489d02c34422d3998bf7544f708e66cc57ef44c63.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/0563e8505464876ac289bee489d02c34422d3998bf7544f708e66cc57ef44c63.jpg)

![18c4c77de3e244f589aa42fe99f6701750ed8892d2340b96f32712b10ef812ca.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/18c4c77de3e244f589aa42fe99f6701750ed8892d2340b96f32712b10ef812ca.jpg)

![30b324cedfafa9e86303e6f70b64afdb83c476772ab16cada6f9aabdb68a0233.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/30b324cedfafa9e86303e6f70b64afdb83c476772ab16cada6f9aabdb68a0233.jpg)

![3261c1f37e148bd42972675fb53ebb84fbafe5159a0e51aefd7b29a6a877fe7c.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/3261c1f37e148bd42972675fb53ebb84fbafe5159a0e51aefd7b29a6a877fe7c.jpg)

![357250947b4009cab534fbd10c033128aaa93afc70431a8b6e6ee2f9152a7a3c.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/357250947b4009cab534fbd10c033128aaa93afc70431a8b6e6ee2f9152a7a3c.jpg)

![44b9f6b94c15e61626d4c0f054fe64dae84104c924e0abf469240e45b7db7c37.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/44b9f6b94c15e61626d4c0f054fe64dae84104c924e0abf469240e45b7db7c37.jpg)

![58e8f8cd5dabd2220840994e1a6e57d3d2f72afadc28aeabcd246f62cb5f81ca.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/58e8f8cd5dabd2220840994e1a6e57d3d2f72afadc28aeabcd246f62cb5f81ca.jpg)

![662960deb58acabbe023011b9395bf730bac99a5b5542986ac94b833db08ec84.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/662960deb58acabbe023011b9395bf730bac99a5b5542986ac94b833db08ec84.jpg)

![6864f6e000a94d73cd49561abebaaf933f3d955b0402f102c76151b9d5e5779e.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/6864f6e000a94d73cd49561abebaaf933f3d955b0402f102c76151b9d5e5779e.jpg)

![6ffeedeae0548b8c5ad9858fa3f016db0fa3504644c6215eda9d9bbb740cacb4.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/6ffeedeae0548b8c5ad9858fa3f016db0fa3504644c6215eda9d9bbb740cacb4.jpg)

![76d40daa606f4a76b55045d326b73a1db0becc33cdc4c05939ce31d0c5771541.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/76d40daa606f4a76b55045d326b73a1db0becc33cdc4c05939ce31d0c5771541.jpg)

![84ffb369b8bcd4327c00073fa3517efc7a62850c7d3b06627bf0361094309f8c.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/84ffb369b8bcd4327c00073fa3517efc7a62850c7d3b06627bf0361094309f8c.jpg)

![88d0b76930487a3bfa964cb7c5718f63ea705c0e0274eac7c19de75e615ed02b.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/88d0b76930487a3bfa964cb7c5718f63ea705c0e0274eac7c19de75e615ed02b.jpg)

![8a4502640ad798260a0d67f27e5f7c5a1f0e60713013d1f30de15c3b8c6a7999.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/8a4502640ad798260a0d67f27e5f7c5a1f0e60713013d1f30de15c3b8c6a7999.jpg)

![8ef6aeec67212de3eddd3559456b82d1a8d82ce254a3076a911f34a2d1817f55.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/8ef6aeec67212de3eddd3559456b82d1a8d82ce254a3076a911f34a2d1817f55.jpg)

![9aba61fc4c938caf892513ef43d2467bc0ecb010ae546ba203adda034f447aac.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/9aba61fc4c938caf892513ef43d2467bc0ecb010ae546ba203adda034f447aac.jpg)

![a639ee5c4819db24d6eeaefa682a4ed733ed48eb591688eb2d459f4e07392ee2.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/a639ee5c4819db24d6eeaefa682a4ed733ed48eb591688eb2d459f4e07392ee2.jpg)

![bb1644dad21ea369ef3eb8c44fb09f39c3579dd9397967d20df6191e09e671b0.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/bb1644dad21ea369ef3eb8c44fb09f39c3579dd9397967d20df6191e09e671b0.jpg)

![bb75cbf7a488efc51e94739e5d7a75a767008df14f14116a533b7b8ac5e6b126.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/bb75cbf7a488efc51e94739e5d7a75a767008df14f14116a533b7b8ac5e6b126.jpg)

![bccd52f36568e966731cd5c0dd3066b4c61b4a2eaa03916fd84851a435392002.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/bccd52f36568e966731cd5c0dd3066b4c61b4a2eaa03916fd84851a435392002.jpg)

![bf33d438f09cc661ac08ad4b6f464e0f510dc908ae6801a0a8d7778d7d7f8481.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/bf33d438f09cc661ac08ad4b6f464e0f510dc908ae6801a0a8d7778d7d7f8481.jpg)

![c247315081e311e9a8245a7c87b060515ebbd0610a0d955f520281bfd014ee76.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/c247315081e311e9a8245a7c87b060515ebbd0610a0d955f520281bfd014ee76.jpg)

![cd77dd6be9389c491e39af6939293af8665e475ec9fb8fbc67a735599f2f6af5.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/cd77dd6be9389c491e39af6939293af8665e475ec9fb8fbc67a735599f2f6af5.jpg)

![d926180a4e6e6f6eb792eac658a2690eb79d16bf1f4a85a0578e91b8127f1f6f.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/d926180a4e6e6f6eb792eac658a2690eb79d16bf1f4a85a0578e91b8127f1f6f.jpg)

![db7265537fc6bff668703f421adb9bbd49110b408c4806d81b773a569be9f53a.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/db7265537fc6bff668703f421adb9bbd49110b408c4806d81b773a569be9f53a.jpg)

![e0269a3fe4cbb01ff0df15e403d501949c0cfd93fdf11a93d2e2b17e9851b577.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/e0269a3fe4cbb01ff0df15e403d501949c0cfd93fdf11a93d2e2b17e9851b577.jpg)

![e25c334c5cafcd17d5732f108fce3f5402960521d41966864451030d6bef849c.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/e25c334c5cafcd17d5732f108fce3f5402960521d41966864451030d6bef849c.jpg)

![e3562b6877551fe725386c55b23e7086706c32cf8a45c00fdefa548e0f657345.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/e3562b6877551fe725386c55b23e7086706c32cf8a45c00fdefa548e0f657345.jpg)

![e405e37f619d2700b78ea111cc8704d845797b5d1b4c8a83a8c12e94411e3774.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/e405e37f619d2700b78ea111cc8704d845797b5d1b4c8a83a8c12e94411e3774.jpg)

![e9e56c4b5c00517f1a27b607f8a6293d7df0a34f339d28487085829d2e8a0b84.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/e9e56c4b5c00517f1a27b607f8a6293d7df0a34f339d28487085829d2e8a0b84.jpg)

![fe414d7409fc2b311c32eaaf2873e5be146327eee53ebf9630bbf0ce810b8743.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/images/fe414d7409fc2b311c32eaaf2873e5be146327eee53ebf9630bbf0ce810b8743.jpg)

### Tables

![081e8d7ec0bd733564c4e8e53da99f93d323280964981965ee7617a1a217818d.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/tables/081e8d7ec0bd733564c4e8e53da99f93d323280964981965ee7617a1a217818d.jpg)

![13eba5e6024461cfc4bb815f6266a5d47dfe495f784500a849401cd54ab2f373.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/tables/13eba5e6024461cfc4bb815f6266a5d47dfe495f784500a849401cd54ab2f373.jpg)

![727c7a174fa8a7e4596c2a8cee5185e720e4602d3642e6c0046fc19bf986f040.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/tables/727c7a174fa8a7e4596c2a8cee5185e720e4602d3642e6c0046fc19bf986f040.jpg)

![adfeb0699619a3002e12af0658240e6c99c54bc5a8ba8ab81738ac740f3bb907.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/tables/adfeb0699619a3002e12af0658240e6c99c54bc5a8ba8ab81738ac740f3bb907.jpg)

![d12e68beda837312ec477bbbbab06ffee0cee1090b8686eb05c5f5a7152fe857.jpg](../icml_results/503_Prescriptive%20Scaling%20Reveals%20the%20Evolution%20of%20Language%20Model%20Capabilities/tables/d12e68beda837312ec477bbbbab06ffee0cee1090b8686eb05c5f5a7152fe857.jpg)

<a id="paper-0495"></a>

## Characterizing, Evaluating, and Optimizing Complex Reasoning

### Images

![18c29e0c5e99a01d576370149a07f3e6c6e606aff2e79a5cfc9ee047da6f261e.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/18c29e0c5e99a01d576370149a07f3e6c6e606aff2e79a5cfc9ee047da6f261e.jpg)

![3041c9fe988c3a7c5fe8368a4a6014842114a671debc6b4a05c997cf0704c2d9.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/3041c9fe988c3a7c5fe8368a4a6014842114a671debc6b4a05c997cf0704c2d9.jpg)

![35ff2627e59aa469f68adc90ce3d454c3fe76061f8e7696006be04f930100679.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/35ff2627e59aa469f68adc90ce3d454c3fe76061f8e7696006be04f930100679.jpg)

![48eb4e1e68eb525ff2d5e4c47692f502fd5d14090b0c1053c8fbc6503d1c35b3.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/48eb4e1e68eb525ff2d5e4c47692f502fd5d14090b0c1053c8fbc6503d1c35b3.jpg)

![66bef6b908e32f950d1f2a9657832e5446f2331e6ec53ef5df0785d3a43ad2cd.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/66bef6b908e32f950d1f2a9657832e5446f2331e6ec53ef5df0785d3a43ad2cd.jpg)

![69b27278fc661526029a00b45840acbf270c9a1239a49710f070061b9fce5c8c.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/69b27278fc661526029a00b45840acbf270c9a1239a49710f070061b9fce5c8c.jpg)

![7e6ff45b48a9b44bd3e1471a68750849a07b475e159be9cd0be8e74c2172ec62.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/7e6ff45b48a9b44bd3e1471a68750849a07b475e159be9cd0be8e74c2172ec62.jpg)

![8dfe04012beb10eeaa545d818eb683f48b5e39a91f2bfb1e80357286879197ec.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/8dfe04012beb10eeaa545d818eb683f48b5e39a91f2bfb1e80357286879197ec.jpg)

![ab90164db646bbe37a35241ec1248583c5f15dbc9aee21f8f10897003580ad99.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/ab90164db646bbe37a35241ec1248583c5f15dbc9aee21f8f10897003580ad99.jpg)

![ae91a5071290d5425fc246622aed2e5a2aaaa4a12be4e92288c540deada9c60a.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/images/ae91a5071290d5425fc246622aed2e5a2aaaa4a12be4e92288c540deada9c60a.jpg)

### Tables

![0adb8b09aee9f20eb86b733d89c97b689292c3fb1f6598e2866dbb6069efdff0.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/0adb8b09aee9f20eb86b733d89c97b689292c3fb1f6598e2866dbb6069efdff0.jpg)

![0f46fabdaff99516038d1ee41500b2b01a0b4e0e89c1d82b0b16324798e9cda9.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/0f46fabdaff99516038d1ee41500b2b01a0b4e0e89c1d82b0b16324798e9cda9.jpg)

![1bdfbc5823fa6b3291fefd4aa3a2dd541e0e498a6321636d01ebe417492f09b0.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/1bdfbc5823fa6b3291fefd4aa3a2dd541e0e498a6321636d01ebe417492f09b0.jpg)

![1eabeb19ae5eaf41c1d2bb4a3bf5a53c004c78bcd0c407bac0d87aac90adb76d.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/1eabeb19ae5eaf41c1d2bb4a3bf5a53c004c78bcd0c407bac0d87aac90adb76d.jpg)

![6070278194744baa0fc5bbbaa4f3f909128e92f1f1447d855d9539b111a4eaef.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/6070278194744baa0fc5bbbaa4f3f909128e92f1f1447d855d9539b111a4eaef.jpg)

![9e74fcab698c34a8efb36a0715a6eccf7715b21bc9323d0bacd4dc74560f5bf2.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/9e74fcab698c34a8efb36a0715a6eccf7715b21bc9323d0bacd4dc74560f5bf2.jpg)

![c03b48c874a5b98fbca0f15c09e40231c1a789201b49cc4cc446f7ecc6afbbcf.jpg](../icml_results/504_Characterizing%2C%20Evaluating%2C%20and%20Optimizing%20Complex%20Reasoning/tables/c03b48c874a5b98fbca0f15c09e40231c1a789201b49cc4cc446f7ecc6afbbcf.jpg)
