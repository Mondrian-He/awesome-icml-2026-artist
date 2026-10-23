# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 31 of 197

## 目录 (Table of Contents)

1. [Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation](#paper-0991)
2. [Understanding SAM through Minimax Perspective](#paper-0992)
3. [Bi-Anchor Interpolation Solver for Accelerating Generative Modeling](#paper-0993)
4. [Diversity-aware Weight Perturbation Promotes Robust Adaptation](#paper-0994)
5. [$φ$-Balancing for Mixture-of-Experts Training](#paper-0995)
6. [Selective Disclosure Watermarking for Large Language Models](#paper-0996)
7. [BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models](#paper-0997)
8. [Towards Docking-oriented De Novo Ligand Design via Gradient Inversion](#paper-0998)
9. [DFlash: Block Diffusion for Flash Speculative Decoding](#paper-0999)
10. [TSMGen: Target-Specific Molecule Generation via Higher-Order Structural Dependencies and Context-Aware Bidirectional Fusion](#paper-1000)
11. [Code2Video: A Code-centric Paradigm for Educational Video Creation](#paper-1001)
12. [Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling](#paper-1002)
13. [Threat2Traffic: Multi-Agent Environment Synthesis for Malware Traffic Generation from Threat Intelligence](#paper-1003)
14. [ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning](#paper-1004)
15. [WEVSR: Video Diffusion Generators for Real-World Video Super-Resolution with Wavelet-Enhanced VAE Encoder](#paper-1005)
16. [InteractScience: Programmatic and Visually-Grounded Evaluation of Interactive Scientific Demonstration Code Generation](#paper-1006)
17. [LoKiFormer: Locality-aware Attention with Decoupled Knowledge Memory for Efficient Large Language Model Pretraining](#paper-1007)
18. [Hamiltonian Asymmetric Fusion: One-Way Safe Directed Refinement under Modality Imbalance](#paper-1008)
19. [Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models](#paper-1009)
20. [Learning-Guided Integration Contours Construction for Fast Large-Scale Generalized Eigensolvers](#paper-1010)
21. [MALICE: Memory-aware Loop Invariants Generation on Symbolic Execution Traces](#paper-1011)
22. [Focusing: View-Consistent Sparse Voxels for Efficient 3D VAE Training](#paper-1012)
23. [Rethinking Instruction Drift as a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](#paper-1013)
24. [Personalized Additive Modeling for Multi-level Federated Learning](#paper-1014)
25. [Efficient Inference for Noisy LLM-as-a-Judge Evaluation](#paper-1015)
26. [DANCE: Dynamic, Available, Neighbor-gated Condensation for Federated Text-Attributed Graphs](#paper-1016)
27. [Towards Theoretical Understanding of Transformer Test-Time Computing: Investigation on In-Context Linear Regression](#paper-1017)
28. [SLAE: Strictly Local All-atom Environment for Protein Representation](#paper-1018)
29. [Co-Generative De Novo Functional Protein Design](#paper-1019)
30. [RADAR: Defending RAG Dynamically against Retrieval Corruption](#paper-1020)
31. [Escaping the Diversity Trap in Robotic Manipulation via Anchor-Centric Adaptation](#paper-1021)
32. [Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](#paper-1022)
33. [How can we assess human-agent interactions? Case studies in software agent design](#paper-1023)

---


<a id="paper-0991"></a>

## Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation

### Images

![f17943f01defb50d26081aa67769d86a835272cb8b4570250eedd534f87707a8.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/images/f17943f01defb50d26081aa67769d86a835272cb8b4570250eedd534f87707a8.jpg)

### Tables

![0a2ff7a35bdb2cbc0c977bf573689728a1a70f10d0d49fd016d56f3228f543f1.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/0a2ff7a35bdb2cbc0c977bf573689728a1a70f10d0d49fd016d56f3228f543f1.jpg)

![11ed87c7c521c9495b2fdcfda3229c571291cbc8b5ae61bba99dd9af257978dd.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/11ed87c7c521c9495b2fdcfda3229c571291cbc8b5ae61bba99dd9af257978dd.jpg)

![35db4cab6777602a5f8987a04eeb0e3c4864efc76aa20e6dabf7e9fd3dcd0c0a.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/35db4cab6777602a5f8987a04eeb0e3c4864efc76aa20e6dabf7e9fd3dcd0c0a.jpg)

![416a0d3fd068f97db5e21b41205d5d2a3524ef5b734f1d8dc6c20c46c9de8e39.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/416a0d3fd068f97db5e21b41205d5d2a3524ef5b734f1d8dc6c20c46c9de8e39.jpg)

![564279f9a05aa6680c0f8603507ccbaef654a530cc1e25120e9bdef2982797f6.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/564279f9a05aa6680c0f8603507ccbaef654a530cc1e25120e9bdef2982797f6.jpg)

![706b83cbd3d57cca6c233080e2085c9222dc8c9d9901f348c78eb52c91e61460.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/706b83cbd3d57cca6c233080e2085c9222dc8c9d9901f348c78eb52c91e61460.jpg)

![ac63ad66273c44dec3c86f32f49d48f275bd18c0190023a161a988c971cfc4df.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/ac63ad66273c44dec3c86f32f49d48f275bd18c0190023a161a988c971cfc4df.jpg)

![b5cfe196e88b0f6a56cd26af4cbc0f8b02cf7fd2b8195d68f2c5e4fbe58a894c.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/b5cfe196e88b0f6a56cd26af4cbc0f8b02cf7fd2b8195d68f2c5e4fbe58a894c.jpg)

![bd9a00904fdab6632ff999afa0f9d54b480b6183890da2808231c0388311d53f.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/bd9a00904fdab6632ff999afa0f9d54b480b6183890da2808231c0388311d53f.jpg)

![f26c0ae1bcc68727e09f13bc0761251696b50746ef62d3484d11a55376c47f22.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/f26c0ae1bcc68727e09f13bc0761251696b50746ef62d3484d11a55376c47f22.jpg)

![f74f591b10249ef58eb015b58d9ae651f1c3c4dff0503035ecb2b66dc3431a13.jpg](../icml_results/1007_Colorful%20Pinball_%20Density-Weighted%20Quantile%20Regression%20for%20Conditional%20Guarantee%20of%20Conformal%20Predic/tables/f74f591b10249ef58eb015b58d9ae651f1c3c4dff0503035ecb2b66dc3431a13.jpg)

## Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation


### Images

![087fa4a7a37869987b15dc5aaa0334c367270960d2220ce6276fe13056fa8def.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/087fa4a7a37869987b15dc5aaa0334c367270960d2220ce6276fe13056fa8def.jpg)

![1a9d38b0178d62400206e1fbcff9f98af88d6960db679b77447533bb70e849c1.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/1a9d38b0178d62400206e1fbcff9f98af88d6960db679b77447533bb70e849c1.jpg)

![1d662f4ee8dbf05a8fff46d19b650eea8f897f30b1f763bc47176a31eda36835.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/1d662f4ee8dbf05a8fff46d19b650eea8f897f30b1f763bc47176a31eda36835.jpg)

![28429ebad821837e682e249758c7b52c70f394ec19ff6bdd45933a72f8bd4f26.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/28429ebad821837e682e249758c7b52c70f394ec19ff6bdd45933a72f8bd4f26.jpg)

![29fea1a04ab8dace8b6b49dd294e2d3be5ba17b7055e236dfd4580c6598c177c.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/29fea1a04ab8dace8b6b49dd294e2d3be5ba17b7055e236dfd4580c6598c177c.jpg)

![332811dd4bcf0fa434840a7889265241a9c3eb13500d40aeb00b288880ea0f57.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/332811dd4bcf0fa434840a7889265241a9c3eb13500d40aeb00b288880ea0f57.jpg)

![3e9bcea624698293e57f13d7af17035a22e592c0a60b56037d49c152e2a7d678.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/3e9bcea624698293e57f13d7af17035a22e592c0a60b56037d49c152e2a7d678.jpg)

![50d622d5e88b0a938d1f110a3d76ae95063d0707e7fbaaeab80d09f41da8b373.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/50d622d5e88b0a938d1f110a3d76ae95063d0707e7fbaaeab80d09f41da8b373.jpg)

![65367653f69595fca68acb05023794196e3c6b7491433381a0fd047b0cd1414f.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/65367653f69595fca68acb05023794196e3c6b7491433381a0fd047b0cd1414f.jpg)

![81aed47e7b8195b2b79a44208bea0761817fd5fecb3d97f889c8609b8826f33f.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/81aed47e7b8195b2b79a44208bea0761817fd5fecb3d97f889c8609b8826f33f.jpg)

![854760e595e7a37e026b9e05c91567d663104e587fd4198d5e1c3182207c2c3e.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/854760e595e7a37e026b9e05c91567d663104e587fd4198d5e1c3182207c2c3e.jpg)

![86d29aae70a78672709ac8fa37dfcde0530025957202bc562b8cc63a576bb1ef.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/86d29aae70a78672709ac8fa37dfcde0530025957202bc562b8cc63a576bb1ef.jpg)

![8c9f9a2677a4c431c5b629e1976c4443ddf9398403e1737d465e87a4cfc3df2b.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/8c9f9a2677a4c431c5b629e1976c4443ddf9398403e1737d465e87a4cfc3df2b.jpg)

![924685563c37a8f4be8c853c129370c99cf89b9b9ccd3ea682f758e563f9f4b7.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/924685563c37a8f4be8c853c129370c99cf89b9b9ccd3ea682f758e563f9f4b7.jpg)

![a058d4ce11de8b89264167c1d80cd7a30ce19e74e0a67fa104a59cf262a4d57a.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/a058d4ce11de8b89264167c1d80cd7a30ce19e74e0a67fa104a59cf262a4d57a.jpg)

![a6012eb080898d10b7e9159258aa49dcfa390e585177bf020cb337477ad7fab6.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/a6012eb080898d10b7e9159258aa49dcfa390e585177bf020cb337477ad7fab6.jpg)

![aba8cccf207a6df6d5d1ca2d2ab4579412ed9edf17139c60659b1875fd92e6d3.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/aba8cccf207a6df6d5d1ca2d2ab4579412ed9edf17139c60659b1875fd92e6d3.jpg)

![ad988454e15f1873013c4f4aaec4b9a6969ebf92d59ce900b632ff2a7c75b20e.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/ad988454e15f1873013c4f4aaec4b9a6969ebf92d59ce900b632ff2a7c75b20e.jpg)

![b7184f6ec2e3e103eda305caf73a0f4dc678ea706680d50b52ec5761cf4658b9.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/b7184f6ec2e3e103eda305caf73a0f4dc678ea706680d50b52ec5761cf4658b9.jpg)

![b749417983b07f55edd519754d54f467a0c4568c02c1dc929e87ad8a66235ba1.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/images/b749417983b07f55edd519754d54f467a0c4568c02c1dc929e87ad8a66235ba1.jpg)

### Tables

![0a7b76e8297be90af88f459184944b714c971a22847d0f5a1e8087914dda5a8e.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/0a7b76e8297be90af88f459184944b714c971a22847d0f5a1e8087914dda5a8e.jpg)

![234ee69954b2394d9fd6f30a8688e57cfdee356c42c4eb810a3cba7451a03025.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/234ee69954b2394d9fd6f30a8688e57cfdee356c42c4eb810a3cba7451a03025.jpg)

![3302f6e8800db1eddf6e09559b40606d51fa7f91da1c584685b983bfd08c84e6.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/3302f6e8800db1eddf6e09559b40606d51fa7f91da1c584685b983bfd08c84e6.jpg)

![3ae971f67d732edb5c183f17449a3d95bea9a29e8f587c7b0c2381b988deaf3a.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/3ae971f67d732edb5c183f17449a3d95bea9a29e8f587c7b0c2381b988deaf3a.jpg)

![45bf970e287b1704ecde00e3a3ca93593efb0baff4043ffdbe78af4b6047d4a0.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/45bf970e287b1704ecde00e3a3ca93593efb0baff4043ffdbe78af4b6047d4a0.jpg)

![56f08e792ac88b4e4bfbbaa8dc7e8e0836467a43619e83022215fb3dfb7a934e.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/56f08e792ac88b4e4bfbbaa8dc7e8e0836467a43619e83022215fb3dfb7a934e.jpg)

![67e2f4a48f74ab5b52704cf65d3e85f409f6411ca5bd40f6cac09967c8b2f719.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/67e2f4a48f74ab5b52704cf65d3e85f409f6411ca5bd40f6cac09967c8b2f719.jpg)

![6d7193ca21d1086164ba0b071853005b40a44aade874a05231a99244328be23a.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/6d7193ca21d1086164ba0b071853005b40a44aade874a05231a99244328be23a.jpg)

![80f7a1625f97ed317f08d1b108945aabbadb2a4ac1249ed58732b636ccc3489c.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/80f7a1625f97ed317f08d1b108945aabbadb2a4ac1249ed58732b636ccc3489c.jpg)

![855f3d19d3673040802d710abdcc79db675912262b5d92f12930c75fa4d37005.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/855f3d19d3673040802d710abdcc79db675912262b5d92f12930c75fa4d37005.jpg)

![87b97782c36231be890b2b44bf09b69f916e9c04f2aa6d7a62adcc387d0a08f2.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/87b97782c36231be890b2b44bf09b69f916e9c04f2aa6d7a62adcc387d0a08f2.jpg)

![96b25f8b28582f9be54e3ee7a7d5e5f6f9b1f88bf3903fa93723cbf154c80c52.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/96b25f8b28582f9be54e3ee7a7d5e5f6f9b1f88bf3903fa93723cbf154c80c52.jpg)

![9a80b28897ba35bbed7fd8ddd60e724a11f7afffc65ffdf262e84b8e4a572e13.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/9a80b28897ba35bbed7fd8ddd60e724a11f7afffc65ffdf262e84b8e4a572e13.jpg)

![a2a1bcdcf4041af42e3f204da59bbe222fce899a446f9dfa287ae1f542604bc8.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/a2a1bcdcf4041af42e3f204da59bbe222fce899a446f9dfa287ae1f542604bc8.jpg)

![ae3812d786d7eb8acf2f91a4aea8472d4662839ca3cf7b6575da07b6ca640694.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/ae3812d786d7eb8acf2f91a4aea8472d4662839ca3cf7b6575da07b6ca640694.jpg)

![ae8c0025a18255b7a96437eb63e373b651576b5425c3778c21d2471bee47d8dc.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/ae8c0025a18255b7a96437eb63e373b651576b5425c3778c21d2471bee47d8dc.jpg)

![aec27de96a866103cd47316ebadfffead7e602f08c638b0a472942e9683327ec.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/aec27de96a866103cd47316ebadfffead7e602f08c638b0a472942e9683327ec.jpg)

![b53d6eaf63b67446e8225b9206420f371ca5869e1f7a8ca32b520399fe144133.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/b53d6eaf63b67446e8225b9206420f371ca5869e1f7a8ca32b520399fe144133.jpg)

![ed37277280362acc47806009b19386a349ac2ed036ed53fc72135cd8b3f05b88.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/ed37277280362acc47806009b19386a349ac2ed036ed53fc72135cd8b3f05b88.jpg)

![ed62f1b18f9c77494258bbdfec7567b60a35bee355cb590779671f0887602545.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/ed62f1b18f9c77494258bbdfec7567b60a35bee355cb590779671f0887602545.jpg)

![fe494a2a57113904acc2278a4b6f04092ac12750940e6268e171609a78f83267.jpg](../icml_results/1008_Language%20Bias%20in%20LVLMs_%20From%20In-Depth%20Analysis%20to%20Simple%20and%20Effective%20Mitigation/tables/fe494a2a57113904acc2278a4b6f04092ac12750940e6268e171609a78f83267.jpg)

<a id="paper-0992"></a>

## Understanding SAM through Minimax Perspective


### Images

![2f8fc87938bac696a21e28e9fe4ce71d1d3d60812e4f2b5f805cb042827427a9.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/images/2f8fc87938bac696a21e28e9fe4ce71d1d3d60812e4f2b5f805cb042827427a9.jpg)

![2faa1ef0a737cc2eba60a2cbb28bcf29bf8496228e5e353e61fb7de8a5398f7a.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/images/2faa1ef0a737cc2eba60a2cbb28bcf29bf8496228e5e353e61fb7de8a5398f7a.jpg)

### Tables

![03e48af5acbc4638569138f5cbce06e84839807d64c3c523c124cabbb063d453.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/03e48af5acbc4638569138f5cbce06e84839807d64c3c523c124cabbb063d453.jpg)

![04195dd05cdfe9b1d817badb99c037f7b6e8cd0eb057025fc9335376b7a6a359.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/04195dd05cdfe9b1d817badb99c037f7b6e8cd0eb057025fc9335376b7a6a359.jpg)

![06328a87e23609badfca2e34b274a171646e857edf0eb0b873883838137fde83.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/06328a87e23609badfca2e34b274a171646e857edf0eb0b873883838137fde83.jpg)

![1410f2e82823cdab181b11dee34c4510d9ad155515e6d2d0c0350e5aaf1f3175.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/1410f2e82823cdab181b11dee34c4510d9ad155515e6d2d0c0350e5aaf1f3175.jpg)

![149d72f81fffdbc671e8d8f3b978948a15f747a7464537b40f32d144329ba761.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/149d72f81fffdbc671e8d8f3b978948a15f747a7464537b40f32d144329ba761.jpg)

![333c75524ce5014afad41fa312baae1c82f3a681989ff7f99cf6c78fac801017.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/333c75524ce5014afad41fa312baae1c82f3a681989ff7f99cf6c78fac801017.jpg)

![60c27c7a605f445a63e57be34517ca2a75115b5f63bab85e9445435543a1c79a.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/60c27c7a605f445a63e57be34517ca2a75115b5f63bab85e9445435543a1c79a.jpg)

![70c0be4d5d84fe318ee74dd89c12f5504a84229e1c7ab7d54fb4242af5915cdf.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/70c0be4d5d84fe318ee74dd89c12f5504a84229e1c7ab7d54fb4242af5915cdf.jpg)

![8bac4879c75cb0924a9b98906e2c79f07f7037780c8c21bead175b9838aa0d67.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/8bac4879c75cb0924a9b98906e2c79f07f7037780c8c21bead175b9838aa0d67.jpg)

![92cf62f32419b9f7de051cef9ef91b1f7037bb3ef276660415213df4753f0540.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/92cf62f32419b9f7de051cef9ef91b1f7037bb3ef276660415213df4753f0540.jpg)

![a7c7818b8654f9ac3c282fabf48b8bde9ae0c9fc74a8980bcaca89cdb983c4e2.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/a7c7818b8654f9ac3c282fabf48b8bde9ae0c9fc74a8980bcaca89cdb983c4e2.jpg)

![ad696cf777a3138492fc3ace0093f845b5b1150280213f07b37deeb7846d082b.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/ad696cf777a3138492fc3ace0093f845b5b1150280213f07b37deeb7846d082b.jpg)

![bc45e1f21118b02e1daa3ed8e8c20deed8f669da3ce7588cb5770446162621f1.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/bc45e1f21118b02e1daa3ed8e8c20deed8f669da3ce7588cb5770446162621f1.jpg)

![c043fdcbc4894669f72fd772896a8ae63da4c725cdcb5e81b67e2a6c8cea6c9d.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/c043fdcbc4894669f72fd772896a8ae63da4c725cdcb5e81b67e2a6c8cea6c9d.jpg)

![c78feb348c6d94965fc96c908eca9a14a25f517c423e3bb56f128e6b9159c5eb.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/c78feb348c6d94965fc96c908eca9a14a25f517c423e3bb56f128e6b9159c5eb.jpg)

![d0b00785abc16fefc9ccd925ee9ecaee5f1ad1e3786ebee25c5676a1c1d98e33.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/d0b00785abc16fefc9ccd925ee9ecaee5f1ad1e3786ebee25c5676a1c1d98e33.jpg)

![d755d9a9a3ed358599d6c252e1684f1c289fc9bb8e5b021460be103eb8d4caf0.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/d755d9a9a3ed358599d6c252e1684f1c289fc9bb8e5b021460be103eb8d4caf0.jpg)

![f0f614a2b213026f05219fb27601e9290c33193b65deebdf1510972c349fca97.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/f0f614a2b213026f05219fb27601e9290c33193b65deebdf1510972c349fca97.jpg)

![fdfd6d2a39e9cb7ae4e65e5d366d0e52cb17da5417ba371f551d9974d5779b98.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/fdfd6d2a39e9cb7ae4e65e5d366d0e52cb17da5417ba371f551d9974d5779b98.jpg)

![fe4ab38b0dfed2a6bde740c870bc581b1e094f3e3b1549d817971ca31ab44071.jpg](../icml_results/1009_Understanding%20SAM%20through%20Minimax%20Perspective/tables/fe4ab38b0dfed2a6bde740c870bc581b1e094f3e3b1549d817971ca31ab44071.jpg)

<a id="paper-0993"></a>

## Bi-Anchor Interpolation Solver for Accelerating Generative Modeling


### Images

![0344c2c95354a1e069f6ec66ab9d29e9ccd10535a46409bc2e6fb89d6d4fdef7.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/0344c2c95354a1e069f6ec66ab9d29e9ccd10535a46409bc2e6fb89d6d4fdef7.jpg)

![22ab103d655be3fcf9c5190e8b1d2219e6c069f4c780b0cd472b4ab1644cbd11.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/22ab103d655be3fcf9c5190e8b1d2219e6c069f4c780b0cd472b4ab1644cbd11.jpg)

![4099f3e373758eac25af8c5c3332ee7196695b71756da2771b595f2ea5a60e50.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/4099f3e373758eac25af8c5c3332ee7196695b71756da2771b595f2ea5a60e50.jpg)

![455390463b20d9772ae64e9a1c516d5ee8c54705d381c521655ce7119fd18e16.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/455390463b20d9772ae64e9a1c516d5ee8c54705d381c521655ce7119fd18e16.jpg)

![76b168590db123e93a88270e88094869ab1bfd05ab11cd96fd019d2d7a9110b0.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/76b168590db123e93a88270e88094869ab1bfd05ab11cd96fd019d2d7a9110b0.jpg)

![c698e51ce9a3372d0e23e991fd7c17a4f6f8d85013f40a228827eb279d065c8e.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/c698e51ce9a3372d0e23e991fd7c17a4f6f8d85013f40a228827eb279d065c8e.jpg)

![c85e98c8638718d37ca564b71e7baa9ebf2130a5d0c478b5026418385beedf50.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/images/c85e98c8638718d37ca564b71e7baa9ebf2130a5d0c478b5026418385beedf50.jpg)

### Tables

![0a4676454d3800054e4f30bd31acbfb4ffeb05556243dc89918e4201713a858e.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/0a4676454d3800054e4f30bd31acbfb4ffeb05556243dc89918e4201713a858e.jpg)

![12381b08a9fc0900e8bc10f758b2dc24189ce4715c050d139417b3de29ccf86b.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/12381b08a9fc0900e8bc10f758b2dc24189ce4715c050d139417b3de29ccf86b.jpg)

![15f846873b0ec6f20c78beb374f0a12392f739980767e9fac88c572760df3eeb.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/15f846873b0ec6f20c78beb374f0a12392f739980767e9fac88c572760df3eeb.jpg)

![3ea94baaf751e88b49595b26e627824213de2eaa420c0cf085440eb886fab28f.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/3ea94baaf751e88b49595b26e627824213de2eaa420c0cf085440eb886fab28f.jpg)

![8d4453b1fc364a4c445bfaddf9808ab975e67e36a51b59c3954276610f99c778.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/8d4453b1fc364a4c445bfaddf9808ab975e67e36a51b59c3954276610f99c778.jpg)

![b48cb8357a955cb5d385f99e924e8d1d2b7dbea69251d3c7f65e332f4c50f366.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/b48cb8357a955cb5d385f99e924e8d1d2b7dbea69251d3c7f65e332f4c50f366.jpg)

![c6d1e3e02672aa125ba20f5ae5db24e944b0d705bbd5efd4cc0c23810c69a694.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/c6d1e3e02672aa125ba20f5ae5db24e944b0d705bbd5efd4cc0c23810c69a694.jpg)

![dec9a442b3af2151da07a46062e411fb0cf74a49055980effdf0c315e7cda8d7.jpg](../icml_results/1010_Bi-Anchor%20Interpolation%20Solver%20for%20Accelerating%20Generative%20Modeling/tables/dec9a442b3af2151da07a46062e411fb0cf74a49055980effdf0c315e7cda8d7.jpg)

<a id="paper-0994"></a>

## Diversity-aware Weight Perturbation Promotes Robust Adaptation


### Images

![13f56e02f313ae060dc3e3ded020c3967631652c9c6e0b218d5e40eeed1d3cde.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/13f56e02f313ae060dc3e3ded020c3967631652c9c6e0b218d5e40eeed1d3cde.jpg)

![b068d4f5e4d601a10f592b54420912eef9353791cbdeca881034f92e2afa366c.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/b068d4f5e4d601a10f592b54420912eef9353791cbdeca881034f92e2afa366c.jpg)

![be3db73b042d8e9812c452ec1e6f2d414ac4c69e4d08b5455862c46d71b7147d.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/be3db73b042d8e9812c452ec1e6f2d414ac4c69e4d08b5455862c46d71b7147d.jpg)

![c68997b74a99e5a6dcaa11cac60522f33fc0bfea18a24de92ed04d786116ebac.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/c68997b74a99e5a6dcaa11cac60522f33fc0bfea18a24de92ed04d786116ebac.jpg)

![daf6db1bb9d9ae679771ee9e3f518721cd7e7381e4666f956e005e6c9919b6fe.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/daf6db1bb9d9ae679771ee9e3f518721cd7e7381e4666f956e005e6c9919b6fe.jpg)

![e08c383ed92c2f2bd8d2389b87bd0aa0d667849e9211b56caa552d13d8de9ab3.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/e08c383ed92c2f2bd8d2389b87bd0aa0d667849e9211b56caa552d13d8de9ab3.jpg)

![e8c2ecc6169b08a9f428a7d38f35af62ff535d842592bac6c424706ab4487c19.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/images/e8c2ecc6169b08a9f428a7d38f35af62ff535d842592bac6c424706ab4487c19.jpg)

### Tables

![1ca1be757025f55ddd41e5a3423937ea8b558b932bf213147a01f58bc42b5b2c.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/1ca1be757025f55ddd41e5a3423937ea8b558b932bf213147a01f58bc42b5b2c.jpg)

![208919226d269e444f7afaf23ff40abe3eb1fa6d8dbc67a4c8b58b4bb0395457.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/208919226d269e444f7afaf23ff40abe3eb1fa6d8dbc67a4c8b58b4bb0395457.jpg)

![6f3d7f876b704ac688205096373358828e827043dfb791dfb0703c17dd15f818.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/6f3d7f876b704ac688205096373358828e827043dfb791dfb0703c17dd15f818.jpg)

![95b322e4edc0bd1a3c341bbb826ce3a5d69da1cfc5d506d23ef025df5516c752.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/95b322e4edc0bd1a3c341bbb826ce3a5d69da1cfc5d506d23ef025df5516c752.jpg)

![b350a190ee8da754682369e9e1b44929e9f4ee77c9f0c902ceee2a7eb068b79c.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/b350a190ee8da754682369e9e1b44929e9f4ee77c9f0c902ceee2a7eb068b79c.jpg)

![d3eb6e614b56780d28b135b8d9e7302c574ce99389fcd80058d673cd1223774d.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/d3eb6e614b56780d28b135b8d9e7302c574ce99389fcd80058d673cd1223774d.jpg)

![da11a6fa4dc49d68737c7a62b31d846a8b2535cab56a07a5625e50d6d93e95bd.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/da11a6fa4dc49d68737c7a62b31d846a8b2535cab56a07a5625e50d6d93e95bd.jpg)

![eda75cabbe2edf9ca373b2ce5ecd7aef2cc776943ee6d2a8303b3f053bce630f.jpg](../icml_results/1011_Diversity-aware%20Weight%20Perturbation%20Promotes%20Robust%20Adaptation/tables/eda75cabbe2edf9ca373b2ce5ecd7aef2cc776943ee6d2a8303b3f053bce630f.jpg)

<a id="paper-0995"></a>

## $φ$-Balancing for Mixture-of-Experts Training


### Images

![640c1da364eaa46e5a9ba6d4478e3cbd2c1d7ffa52786dda5468773cfddf9516.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/640c1da364eaa46e5a9ba6d4478e3cbd2c1d7ffa52786dda5468773cfddf9516.jpg)

![6f5fe20202b12c601002eb33d4989baf7425c1faeca2dce675a54f724654b289.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/6f5fe20202b12c601002eb33d4989baf7425c1faeca2dce675a54f724654b289.jpg)

![87971a17d85210a5087b56d77c4f4c9ac26b112e1015e9d27c6a3cc30c90ada4.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/87971a17d85210a5087b56d77c4f4c9ac26b112e1015e9d27c6a3cc30c90ada4.jpg)

![9725a0dc59c934c1346b18fe11c46ca376a01831f390050289eb341732ac4c04.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/9725a0dc59c934c1346b18fe11c46ca376a01831f390050289eb341732ac4c04.jpg)

![b0fe600fd52ed3eab160bae74c9846ceb32c3defcdf16b2daa06712ddd91ced6.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/b0fe600fd52ed3eab160bae74c9846ceb32c3defcdf16b2daa06712ddd91ced6.jpg)

![d49914e185146f64d00337132fa59eb769319aefc3a71fc0c8879ca06667630b.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/d49914e185146f64d00337132fa59eb769319aefc3a71fc0c8879ca06667630b.jpg)

![f9e0cee159706567a76ced6b3f3b227e153230ce1ef33339fd0cb6f316a84c57.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/f9e0cee159706567a76ced6b3f3b227e153230ce1ef33339fd0cb6f316a84c57.jpg)

![fd905a53e38c369c5e5eb6e3a7e9474321b5e844105415cfc0de5598d7adda7b.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/images/fd905a53e38c369c5e5eb6e3a7e9474321b5e844105415cfc0de5598d7adda7b.jpg)

### Tables

![0148ea7274abfc2b7ad46a835b12a8f57d72d9f470db616535f558680f3d027f.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/0148ea7274abfc2b7ad46a835b12a8f57d72d9f470db616535f558680f3d027f.jpg)

![20d26425d5e12950b1ed383a73f12b48128dab0ebdafad1a0e15de10fe122b18.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/20d26425d5e12950b1ed383a73f12b48128dab0ebdafad1a0e15de10fe122b18.jpg)

![2df9587fda6ebc41f409eeff352ee3a5bd195f959dc71c24cd538f208ca4056c.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/2df9587fda6ebc41f409eeff352ee3a5bd195f959dc71c24cd538f208ca4056c.jpg)

![5585cb7359a302bdfc3ed9dcd5160ca6253d171b3fa0e746889b99635d77ae21.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/5585cb7359a302bdfc3ed9dcd5160ca6253d171b3fa0e746889b99635d77ae21.jpg)

![985136fc72088d92bc2b392f14961e2f3491d8559b0f093a74387db4db5baa16.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/985136fc72088d92bc2b392f14961e2f3491d8559b0f093a74387db4db5baa16.jpg)

![a45f08e0e9f79389f9b8f98afa57c2bf010968bc69f1566eefc0bd82b3177836.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/a45f08e0e9f79389f9b8f98afa57c2bf010968bc69f1566eefc0bd82b3177836.jpg)

![ac4ce21dc00363e125990225bc44d846b8f6c3249d084db324fad8e72eb1d5ed.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/ac4ce21dc00363e125990225bc44d846b8f6c3249d084db324fad8e72eb1d5ed.jpg)

![f98fe576b43585f019d4e53bd3e99bd2c249b3690b2c3d866c2bef8f9ac99b33.jpg](../icml_results/1012_%24%CF%86%24-Balancing%20for%20Mixture-of-Experts%20Training/tables/f98fe576b43585f019d4e53bd3e99bd2c249b3690b2c3d866c2bef8f9ac99b33.jpg)

<a id="paper-0996"></a>

## Selective Disclosure Watermarking for Large Language Models


### Images

![00e845b15146a03aef6b22a90ca3c9ebf735c992c9d1946e8b17ba5889d6c8cd.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/00e845b15146a03aef6b22a90ca3c9ebf735c992c9d1946e8b17ba5889d6c8cd.jpg)

![0359d2c21564a7583d2877bfc03406f035605823c0af0b1aace101a03cbdc2ad.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/0359d2c21564a7583d2877bfc03406f035605823c0af0b1aace101a03cbdc2ad.jpg)

![04e926c1220118c4ddf8c22906093c286cd2708bb388f277d315fde8f6fa4def.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/04e926c1220118c4ddf8c22906093c286cd2708bb388f277d315fde8f6fa4def.jpg)

![07dad4452b5296bb600114249a5f079155d8c6b045443f3b1d4cb824197bbaa1.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/07dad4452b5296bb600114249a5f079155d8c6b045443f3b1d4cb824197bbaa1.jpg)

![35eeda6b8cb86a65a871aa4b2fd9e53d4be445553ede3032d03537b35bc79eb6.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/35eeda6b8cb86a65a871aa4b2fd9e53d4be445553ede3032d03537b35bc79eb6.jpg)

![3d4c38dfbef86f574efdc25c14c37edee4f290d8d41c6a556f7370b9b460d8cf.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/3d4c38dfbef86f574efdc25c14c37edee4f290d8d41c6a556f7370b9b460d8cf.jpg)

![3f21a89f092ef575017ec276e1ffb36ecd59f5f73f5c30625de5f6fb09ee5599.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/3f21a89f092ef575017ec276e1ffb36ecd59f5f73f5c30625de5f6fb09ee5599.jpg)

![516d5c76c8fdca6e45efdfddca85a8356689b1f01fac6810138193061b414e40.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/516d5c76c8fdca6e45efdfddca85a8356689b1f01fac6810138193061b414e40.jpg)

![5b92f0df0f0451d02af40189ab5e2a30f60fab47bc7295f985cf48e1128c44a7.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/5b92f0df0f0451d02af40189ab5e2a30f60fab47bc7295f985cf48e1128c44a7.jpg)

![64b11a4c6d1ae904396610bfbb26937d5dacbc934473c5cf2354c002a2ef02cd.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/64b11a4c6d1ae904396610bfbb26937d5dacbc934473c5cf2354c002a2ef02cd.jpg)

![890a9e65b9a4ccde392af2da10bc8710c736ed781e36855f51b788a9f53bb553.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/890a9e65b9a4ccde392af2da10bc8710c736ed781e36855f51b788a9f53bb553.jpg)

![8f0cae1979d662cb923dd6e59deaf75481f82f6463bf93e443b7718c278dd412.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/8f0cae1979d662cb923dd6e59deaf75481f82f6463bf93e443b7718c278dd412.jpg)

![9d1dc6eca3330e44535b9ad53020970a47422df98fc4cfd33abbcd5f47982e18.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/9d1dc6eca3330e44535b9ad53020970a47422df98fc4cfd33abbcd5f47982e18.jpg)

![a9497ec7692a81c83bcbb2fb015fb17d7b81d97e0bdfba04347a696fe59d3bc6.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/a9497ec7692a81c83bcbb2fb015fb17d7b81d97e0bdfba04347a696fe59d3bc6.jpg)

![ae9b48743f31a323333c34755233fd2c92c7d32dc532c98c54b0d339500cb4d5.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/ae9b48743f31a323333c34755233fd2c92c7d32dc532c98c54b0d339500cb4d5.jpg)

![bee78d4006362ab53f992f1fc1f13f85843d45465a6cdfd8974d1d2e047ade21.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/bee78d4006362ab53f992f1fc1f13f85843d45465a6cdfd8974d1d2e047ade21.jpg)

![c31135d3919011d1addfef1ec30bdb0f09a7cb3fac1843484d10c7fa802d1644.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/c31135d3919011d1addfef1ec30bdb0f09a7cb3fac1843484d10c7fa802d1644.jpg)

![c4f624a1b5316e56eba34180d753b1ca7f83dcd63e9d7609ddde14c16b486250.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/c4f624a1b5316e56eba34180d753b1ca7f83dcd63e9d7609ddde14c16b486250.jpg)

![c64499d2928dc7fd931aae10874618c1c7524affa66bb210ff88c08fb4805b82.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/c64499d2928dc7fd931aae10874618c1c7524affa66bb210ff88c08fb4805b82.jpg)

![e86f59f07bd60f5ac7ff847dfec327860d8fc15b45b2410433d62cb16e95a089.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/e86f59f07bd60f5ac7ff847dfec327860d8fc15b45b2410433d62cb16e95a089.jpg)

![f3038d3dd3022e4e56d7070d2ed2989dc40fb72848a752f4f0f9c32bc5ee5e30.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/f3038d3dd3022e4e56d7070d2ed2989dc40fb72848a752f4f0f9c32bc5ee5e30.jpg)

![fc1423a456037bf06b50dda972ecbda90d3c513aa41b4728185f1f21530a5ac3.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/images/fc1423a456037bf06b50dda972ecbda90d3c513aa41b4728185f1f21530a5ac3.jpg)

### Tables

![1bf65885e80af97a2f67559e8b4dc830b2a160615b422c65c6d17f8b539ed0cf.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/1bf65885e80af97a2f67559e8b4dc830b2a160615b422c65c6d17f8b539ed0cf.jpg)

![1c7fb3659b9290dc08033e278666f100fc8e0cf9e1868e217f7cdf64ac437db4.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/1c7fb3659b9290dc08033e278666f100fc8e0cf9e1868e217f7cdf64ac437db4.jpg)

![31e7f337d27f09e693d22cbcbffe96a69ffbb59a253e51115d533ecc4b6504d4.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/31e7f337d27f09e693d22cbcbffe96a69ffbb59a253e51115d533ecc4b6504d4.jpg)

![4f696b937e2187c1dd25b65415a165339c94ec66558e2da6e428e56c065d63b9.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/4f696b937e2187c1dd25b65415a165339c94ec66558e2da6e428e56c065d63b9.jpg)

![50aaabe71f6c385f79854c104b132920aa056e686eb80ea9e2c4579d06176cc9.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/50aaabe71f6c385f79854c104b132920aa056e686eb80ea9e2c4579d06176cc9.jpg)

![626810e939305469980146d34173fdcbd71a4e21365778a42ff739243cd843e0.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/626810e939305469980146d34173fdcbd71a4e21365778a42ff739243cd843e0.jpg)

![94d613689d20cc22320e90daeddf24861fb381c40e6cd597089ca77eade2d60a.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/94d613689d20cc22320e90daeddf24861fb381c40e6cd597089ca77eade2d60a.jpg)

![960b505cdfef2a446f59c1aab0c4bfc3ef83cd395cd9711bdd795c4bf042159c.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/960b505cdfef2a446f59c1aab0c4bfc3ef83cd395cd9711bdd795c4bf042159c.jpg)

![a32fc40daa7d987d958b102aa809060fcd09923c6b1dc507b33c42396c4bdb6f.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/a32fc40daa7d987d958b102aa809060fcd09923c6b1dc507b33c42396c4bdb6f.jpg)

![d162b5f25b9e50f79f00f67042ac8e86eb4c6c48241967f93287fb6e6bee15b8.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/d162b5f25b9e50f79f00f67042ac8e86eb4c6c48241967f93287fb6e6bee15b8.jpg)

![f367ab6330f9a68016636aafab838891adb83e0ea82eff06749070c3e9bbf1b5.jpg](../icml_results/1013_Selective%20Disclosure%20Watermarking%20for%20Large%20Language%20Models/tables/f367ab6330f9a68016636aafab838891adb83e0ea82eff06749070c3e9bbf1b5.jpg)

<a id="paper-0997"></a>

## BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models


### Images

![7cb440d520cd98d82737d06bfafd0cf5d198fa1a7b180dcdafa5d826defbc5fd.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/images/7cb440d520cd98d82737d06bfafd0cf5d198fa1a7b180dcdafa5d826defbc5fd.jpg)

![e7c7edf6c29649b3c8a5cdf8f115ada5f4ffbb19510f5a4f47a649915fe82d2a.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/images/e7c7edf6c29649b3c8a5cdf8f115ada5f4ffbb19510f5a4f47a649915fe82d2a.jpg)

### Tables

![063ce2d2d61d64a728b2a3986bdbec514087dee8423a559de07a7b342d96964d.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/063ce2d2d61d64a728b2a3986bdbec514087dee8423a559de07a7b342d96964d.jpg)

![158e9a952c1a061509acb93ee5b7d75994cb9655abdb63844a710670e75ead3c.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/158e9a952c1a061509acb93ee5b7d75994cb9655abdb63844a710670e75ead3c.jpg)

![454d95c46bd39116e7b8b144be743dbafdf2503908d1ee6f6eb71820d001a223.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/454d95c46bd39116e7b8b144be743dbafdf2503908d1ee6f6eb71820d001a223.jpg)

![588584e8148a7affbecd90a4c3a958e2d0fd84cada213b84d523147e12a4aa32.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/588584e8148a7affbecd90a4c3a958e2d0fd84cada213b84d523147e12a4aa32.jpg)

![7f5f9400e8f99b58a32449c13e233204fd8b18fcaea5802ff8c13aed5d13af16.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/7f5f9400e8f99b58a32449c13e233204fd8b18fcaea5802ff8c13aed5d13af16.jpg)

![8446a73591d0074eaf87d02b03de674c3239089bcb80d6b40a0cd4db94ebaae1.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/8446a73591d0074eaf87d02b03de674c3239089bcb80d6b40a0cd4db94ebaae1.jpg)

![be8631608e57994bdefdec00c6d6fcf7914237108aed79ce6461371a21c10e75.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/be8631608e57994bdefdec00c6d6fcf7914237108aed79ce6461371a21c10e75.jpg)

![e20b309cb1a3ede5f6ae6a7a27b6b2b547c5a39e3cc1cc2376c774009bd221d9.jpg](../icml_results/1014_BPDQ_%20Bit-Plane%20Decomposition%20Quantization%20on%20a%20Variable%20Grid%20for%20Large%20Language%20Models/tables/e20b309cb1a3ede5f6ae6a7a27b6b2b547c5a39e3cc1cc2376c774009bd221d9.jpg)

<a id="paper-0998"></a>

## Towards Docking-oriented De Novo Ligand Design via Gradient Inversion


### Images

![07abf90ca775c3fa225f1b585e10458d31fbc7b0b878fbeaf4276d4e755f28f6.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/07abf90ca775c3fa225f1b585e10458d31fbc7b0b878fbeaf4276d4e755f28f6.jpg)

![1181200b2ab631a31ed1ebff9bfbf70d496671a44e0b77407f4628e27b060db4.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/1181200b2ab631a31ed1ebff9bfbf70d496671a44e0b77407f4628e27b060db4.jpg)

![3cdfe402271bfd85a63b17f554f5f374146de8afac63a9046b6ed4e6da3b6dab.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/3cdfe402271bfd85a63b17f554f5f374146de8afac63a9046b6ed4e6da3b6dab.jpg)

![4af6be89e00da9fdf17c93ae1b58384737ab279b09279b4aac19752494e79f24.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/4af6be89e00da9fdf17c93ae1b58384737ab279b09279b4aac19752494e79f24.jpg)

![4ba265b49b6e3b6ec445b4d15ab79f3d8ae0ca9389ee033e7f23059d09ee8d4b.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/4ba265b49b6e3b6ec445b4d15ab79f3d8ae0ca9389ee033e7f23059d09ee8d4b.jpg)

![4c3f069998cd4f644cbddbbc6d031e4dff980b25cf94bffd60a4012452808eb7.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/4c3f069998cd4f644cbddbbc6d031e4dff980b25cf94bffd60a4012452808eb7.jpg)

![5447b9d29b9b3e662b5be4e6cf6ce1b76d9da3198c1373433d17619839e88cff.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/5447b9d29b9b3e662b5be4e6cf6ce1b76d9da3198c1373433d17619839e88cff.jpg)

![6b5212d102d9e7fda564da7501f6eb8beb138d07ba5ec8a61d715777467d6d65.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/6b5212d102d9e7fda564da7501f6eb8beb138d07ba5ec8a61d715777467d6d65.jpg)

![70442f224ddb16884f17eb063a4c6024855dedc51bbdafc57e8dc4c75873bf85.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/70442f224ddb16884f17eb063a4c6024855dedc51bbdafc57e8dc4c75873bf85.jpg)

![70adf607659887d65170cd4016ea6c23fb1d41fb5c4eb09e6dd23d6453fe1da4.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/70adf607659887d65170cd4016ea6c23fb1d41fb5c4eb09e6dd23d6453fe1da4.jpg)

![738c99957246cfa4b74e63f2564c952fd5d22887f7290078855446a8e713cfcf.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/738c99957246cfa4b74e63f2564c952fd5d22887f7290078855446a8e713cfcf.jpg)

![74859ada0d8a5a122357b8841cd533baa30c266bbdfc8fea28048cb80eb0f6f2.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/74859ada0d8a5a122357b8841cd533baa30c266bbdfc8fea28048cb80eb0f6f2.jpg)

![9d574b02338464327bbb0a91f4666d6f4d812a76f9af06b1af13c8973bb924be.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/9d574b02338464327bbb0a91f4666d6f4d812a76f9af06b1af13c8973bb924be.jpg)

![a3f786b0cdb41790a9459b76bf1647f5bb3ef9910f6b5c1d1f208b2ba9c43fd0.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/a3f786b0cdb41790a9459b76bf1647f5bb3ef9910f6b5c1d1f208b2ba9c43fd0.jpg)

![a9f39c24e147575771d3f466104e7aa4c0485d446956fede357406586cf65413.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/a9f39c24e147575771d3f466104e7aa4c0485d446956fede357406586cf65413.jpg)

![bf76103ef9cb0d272dd4b509fd8722b0ef2c052ae56106f9c3a220b1b780025d.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/bf76103ef9cb0d272dd4b509fd8722b0ef2c052ae56106f9c3a220b1b780025d.jpg)

![d254d629843a0246bb82a02c8abdd980fd530fee0da864e4c48954bada579d1c.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/d254d629843a0246bb82a02c8abdd980fd530fee0da864e4c48954bada579d1c.jpg)

![ea192ebb24477b5a2e8c436c70229bcc146f16fd5ea623b43fd1ee9149e52c3a.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/ea192ebb24477b5a2e8c436c70229bcc146f16fd5ea623b43fd1ee9149e52c3a.jpg)

![f3fd7b77c54a22eca7ba7bbda41d208f3a7752c0d6dcc95e616b88d1450d84ff.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/images/f3fd7b77c54a22eca7ba7bbda41d208f3a7752c0d6dcc95e616b88d1450d84ff.jpg)

### Tables

![053de8243a48b74b58c0a15ea3eb0688434246a80d2a47c7885fb1477715a083.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/053de8243a48b74b58c0a15ea3eb0688434246a80d2a47c7885fb1477715a083.jpg)

![24c5ba9d4f584f2c413dfc1ba399ff5426ea8275518d12d7af10268eb91e29dd.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/24c5ba9d4f584f2c413dfc1ba399ff5426ea8275518d12d7af10268eb91e29dd.jpg)

![29844901533acd45d80e707c283df614b7ee10c7946063272d01855c9b52721f.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/29844901533acd45d80e707c283df614b7ee10c7946063272d01855c9b52721f.jpg)

![2e3def8abec871fbfa222a9692e9b91fc090f0b4b197a62ed0204a072abb4d3c.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/2e3def8abec871fbfa222a9692e9b91fc090f0b4b197a62ed0204a072abb4d3c.jpg)

![36beda8715b1441ff34b0dd72c191af087569f0ec3154b896f23257dc82182db.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/36beda8715b1441ff34b0dd72c191af087569f0ec3154b896f23257dc82182db.jpg)

![58975f665976819b45b00167acc4a00d5f24536d83f76f33760a2ed2fff3299e.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/58975f665976819b45b00167acc4a00d5f24536d83f76f33760a2ed2fff3299e.jpg)

![849b6bf448848bae45b6b2261f0b1946fa794c0cb0b1ce9457eef2822e8193dd.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/849b6bf448848bae45b6b2261f0b1946fa794c0cb0b1ce9457eef2822e8193dd.jpg)

![87a578e84e3e13150d3b00caf9a3309dd904c1032cb00b13dc77e5f1038e6be1.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/87a578e84e3e13150d3b00caf9a3309dd904c1032cb00b13dc77e5f1038e6be1.jpg)

![a57f47e7a6782c1a0c2f363ab3d31fc570c490a768c39c052b25bc80f6664d7c.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/a57f47e7a6782c1a0c2f363ab3d31fc570c490a768c39c052b25bc80f6664d7c.jpg)

![a75ef87e12a7960c5e14ad90862c690d6dbe3bc70a510f6b6f580c49775ab567.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/a75ef87e12a7960c5e14ad90862c690d6dbe3bc70a510f6b6f580c49775ab567.jpg)

![d71fc1cbab475867fc6cc00cf5b4bbdc7d2397fd63b0e6c6902ce46f5953365a.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/d71fc1cbab475867fc6cc00cf5b4bbdc7d2397fd63b0e6c6902ce46f5953365a.jpg)

![d979e1eaf77ce988679f763e39444761f619fa2ab59ca8c47a2f55862525513f.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/d979e1eaf77ce988679f763e39444761f619fa2ab59ca8c47a2f55862525513f.jpg)

![defb0940420771cefab6a7f5f0365ced423f77b33ee466a2f4f99e3216ad755e.jpg](../icml_results/1015_Towards%20Docking-oriented%20De%20Novo%20Ligand%20Design%20via%20Gradient%20Inversion/tables/defb0940420771cefab6a7f5f0365ced423f77b33ee466a2f4f99e3216ad755e.jpg)

<a id="paper-0999"></a>

## DFlash: Block Diffusion for Flash Speculative Decoding


### Images

![24ae560cd50ef0db09e0d218e8fb3596c382ec03c9b1c9d4373d0548d9ffdcd9.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/images/24ae560cd50ef0db09e0d218e8fb3596c382ec03c9b1c9d4373d0548d9ffdcd9.jpg)

![4f19788c811f9dadeae3117d17848b72574e9003f8c49e901ec4a5e93a5daef0.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/images/4f19788c811f9dadeae3117d17848b72574e9003f8c49e901ec4a5e93a5daef0.jpg)

![6686ad3cb70d673511f9e6e1a8cd88bfe611a17280e50597add3beff4b897774.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/images/6686ad3cb70d673511f9e6e1a8cd88bfe611a17280e50597add3beff4b897774.jpg)

![b820c4ed0816040eda5da8848cb3d961761f0c1a9cf4206ef008946913550a91.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/images/b820c4ed0816040eda5da8848cb3d961761f0c1a9cf4206ef008946913550a91.jpg)

![eac36fb18687e380e59058d17a36b76a9f5d6a3fb96ad3575ae251ae1d348569.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/images/eac36fb18687e380e59058d17a36b76a9f5d6a3fb96ad3575ae251ae1d348569.jpg)

### Tables

![1567fbaeee42dc417ee34f2daf2ddd6ace9880586afae51a3663c5f3500fbc28.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/1567fbaeee42dc417ee34f2daf2ddd6ace9880586afae51a3663c5f3500fbc28.jpg)

![1a5e155cb200aa3680305376e5df1eff5a558c363921c95fee9acf4c2cf0d32a.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/1a5e155cb200aa3680305376e5df1eff5a558c363921c95fee9acf4c2cf0d32a.jpg)

![1c816e277b6fa524fca3d0786e9b79ba5bd35236a5f3933de97feab4c668333c.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/1c816e277b6fa524fca3d0786e9b79ba5bd35236a5f3933de97feab4c668333c.jpg)

![52ab72cd0b7601f7680d4be32cc6c625a007cddb68431827d9b68605e0d479d1.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/52ab72cd0b7601f7680d4be32cc6c625a007cddb68431827d9b68605e0d479d1.jpg)

![54705431293286b2c185853b1c7d3457b2d0f8b44c8b96aacdc5a660ba27d052.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/54705431293286b2c185853b1c7d3457b2d0f8b44c8b96aacdc5a660ba27d052.jpg)

![59384822890ee9817894a57e1b11bf61b33b9347151a9c739a009b990e2f08d3.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/59384822890ee9817894a57e1b11bf61b33b9347151a9c739a009b990e2f08d3.jpg)

![5ead1a1a735b6b7cc2d1d66ec8c87f27282795481d31e781998acc294dd2ccc9.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/5ead1a1a735b6b7cc2d1d66ec8c87f27282795481d31e781998acc294dd2ccc9.jpg)

![b7ecdde187404b4443fdcc43242ef9bfcf4de55deff00644a00ed7dfccd444b5.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/b7ecdde187404b4443fdcc43242ef9bfcf4de55deff00644a00ed7dfccd444b5.jpg)

![c00595f596c99263dacbd99a66df9ae6ac06ba338c8d03a2d98c5dce0440ef0a.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/c00595f596c99263dacbd99a66df9ae6ac06ba338c8d03a2d98c5dce0440ef0a.jpg)

![c149736f1e22a18adb8f8e84b96c1c11694a9a881e5f39f6fb44a251890ce5c8.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/c149736f1e22a18adb8f8e84b96c1c11694a9a881e5f39f6fb44a251890ce5c8.jpg)

![c58086a20d2d72fd6d9fec234e245000259a37b6790966866b038baf53e0657d.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/c58086a20d2d72fd6d9fec234e245000259a37b6790966866b038baf53e0657d.jpg)

![e0d228c502c23db05d987d896c77a328118cab453c8368ebd34c84e8531c33bb.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/e0d228c502c23db05d987d896c77a328118cab453c8368ebd34c84e8531c33bb.jpg)

![f4ad749af53ce7508f497af8a37911581d2d75621618bb70e564a7b0f414143a.jpg](../icml_results/1016_DFlash_%20Block%20Diffusion%20for%20Flash%20Speculative%20Decoding/tables/f4ad749af53ce7508f497af8a37911581d2d75621618bb70e564a7b0f414143a.jpg)

<a id="paper-1000"></a>

## TSMGen: Target-Specific Molecule Generation via Higher-Order Structural Dependencies and Context-Aware Bidirectional Fusion


### Images

![24732e738560b486a987154f06f33eb1ad2e97b8626543ff15c45c70b71bade7.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/24732e738560b486a987154f06f33eb1ad2e97b8626543ff15c45c70b71bade7.jpg)

![56344acc306067efe14d82819d1f9681136876fb16a93641addf84a064d749ec.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/56344acc306067efe14d82819d1f9681136876fb16a93641addf84a064d749ec.jpg)

![5cc910352535c9454fcc34117992fafb1ab740f27c909dfa8e10a63fc71d911f.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/5cc910352535c9454fcc34117992fafb1ab740f27c909dfa8e10a63fc71d911f.jpg)

![835a5d84f28951b1418473703ceda5598e86a8a58286c6241f8a036624767e42.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/835a5d84f28951b1418473703ceda5598e86a8a58286c6241f8a036624767e42.jpg)

![bb59545f99ee0cc6312e5dd6fe8fff2b50977edfeeaaf75e125265e668488fd0.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/bb59545f99ee0cc6312e5dd6fe8fff2b50977edfeeaaf75e125265e668488fd0.jpg)

![cb7109db797fd0b0c93f990d78e5fa541bc1bb2eaa50344a0c41ac277b726d7b.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/cb7109db797fd0b0c93f990d78e5fa541bc1bb2eaa50344a0c41ac277b726d7b.jpg)

![d5f162fd410fd4be30cf72af5c020bc2e87bd1611912c4aa257cf0296409d15e.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/d5f162fd410fd4be30cf72af5c020bc2e87bd1611912c4aa257cf0296409d15e.jpg)

![fff1322c8abd9d8264dd4edb7a3a29f7cc7a9104e4140bc7f6aa66ecd3e9fe92.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/images/fff1322c8abd9d8264dd4edb7a3a29f7cc7a9104e4140bc7f6aa66ecd3e9fe92.jpg)

### Tables

![2e1068ad9e849307c3b2e336718c594d291b0d7293be3f0d82c0505c42a7352f.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/tables/2e1068ad9e849307c3b2e336718c594d291b0d7293be3f0d82c0505c42a7352f.jpg)

![acb0a6beb7160316ad833ad28735a9b0aed7511bdcd6c21fad684672d9c02c91.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/tables/acb0a6beb7160316ad833ad28735a9b0aed7511bdcd6c21fad684672d9c02c91.jpg)

![ccf6883bec600ec5f0a677387401a6a65a129e7c8e27de8857add3ed7d5b25e8.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/tables/ccf6883bec600ec5f0a677387401a6a65a129e7c8e27de8857add3ed7d5b25e8.jpg)

![f659fb2241810319446dd92931ae60cf07f2830c1d4d23d67980f3f2fa634f35.jpg](../icml_results/1017_TSMGen_%20Target-Specific%20Molecule%20Generation%20via%20Higher-Order%20Structural%20Dependencies%20and%20Context-Awa/tables/f659fb2241810319446dd92931ae60cf07f2830c1d4d23d67980f3f2fa634f35.jpg)

<a id="paper-1001"></a>

## Code2Video: A Code-centric Paradigm for Educational Video Creation


### Images

![1d3b7c6b6f45a93f412f44814299043f2dbb1c82c842db1af2c86f59971c5fce.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/1d3b7c6b6f45a93f412f44814299043f2dbb1c82c842db1af2c86f59971c5fce.jpg)

![528a768bada0e92cdbcedb7a429211c724885643999b7e6963a0e1fee3787395.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/528a768bada0e92cdbcedb7a429211c724885643999b7e6963a0e1fee3787395.jpg)

![5b806414d6bcb43e07dddb5cab3357193a261ce65574a50d555a80de7de3d363.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/5b806414d6bcb43e07dddb5cab3357193a261ce65574a50d555a80de7de3d363.jpg)

![9bc3e1fae37656bd44f55b6de7cf3d19ce7552f3e1b37e9c6e5cde685f65d1c0.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/9bc3e1fae37656bd44f55b6de7cf3d19ce7552f3e1b37e9c6e5cde685f65d1c0.jpg)

![b9bde3380dea44d1c64b713b21b9bbb36eeb690a6a6e5ca2adb1e12080f9a363.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/b9bde3380dea44d1c64b713b21b9bbb36eeb690a6a6e5ca2adb1e12080f9a363.jpg)

![bc459bbe4022e592f10e38e57c89f88e776f120e12cf8765665b41479c2b79cd.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/bc459bbe4022e592f10e38e57c89f88e776f120e12cf8765665b41479c2b79cd.jpg)

![c07e94f03d2dff7c23817d0992ec3d9c0500dc1e6a0f669436492faae36a671d.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/c07e94f03d2dff7c23817d0992ec3d9c0500dc1e6a0f669436492faae36a671d.jpg)

![c8e96bafa02b47c7e74e624cd9c1c17bc8c5c4d9679e981c528cfab300a65711.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/c8e96bafa02b47c7e74e624cd9c1c17bc8c5c4d9679e981c528cfab300a65711.jpg)

![c9f5cac1d61f1060d918c7297bacb88a58d2372ca21d59a8932a22bc94eb97bf.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/c9f5cac1d61f1060d918c7297bacb88a58d2372ca21d59a8932a22bc94eb97bf.jpg)

![d4f3a84fe23bc866336d05dbb303be44913873f92fa1c4dde66753614650da40.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/d4f3a84fe23bc866336d05dbb303be44913873f92fa1c4dde66753614650da40.jpg)

![fb885e778c02e47faab2fd7b275a2194ea08072cd7448c3253f09fd1c80f90bb.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/images/fb885e778c02e47faab2fd7b275a2194ea08072cd7448c3253f09fd1c80f90bb.jpg)

### Tables

![277dc04cb679e79c5579586e5902aa021ec7eec00188c8cc6be7b3f64af11d04.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/277dc04cb679e79c5579586e5902aa021ec7eec00188c8cc6be7b3f64af11d04.jpg)

![3444d1e306c2b108d58a1b97e0635bb27bb843ae85d488c117d290258218184a.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/3444d1e306c2b108d58a1b97e0635bb27bb843ae85d488c117d290258218184a.jpg)

![534040e95c1260d3660d4e6956b992ef602c535d486ec8bca8978145af12556f.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/534040e95c1260d3660d4e6956b992ef602c535d486ec8bca8978145af12556f.jpg)

![6c090880b35d8c63e4c41dcd858fb9671d68778b10ae40795d6ff1dcd9a27317.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/6c090880b35d8c63e4c41dcd858fb9671d68778b10ae40795d6ff1dcd9a27317.jpg)

![a186f92861fe3ef484b093c9b20a60c30ad0f8effe3d6cb24899f25da319fbf7.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/a186f92861fe3ef484b093c9b20a60c30ad0f8effe3d6cb24899f25da319fbf7.jpg)

![ae0c8b73034fc3f019342bf35ae9ffdfa7eeae06656122794f50de2003b0b282.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/ae0c8b73034fc3f019342bf35ae9ffdfa7eeae06656122794f50de2003b0b282.jpg)

![b3e1d70fca8f32135548ded56adec46c6d524fd0465086e2120322f9f48d97ff.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/b3e1d70fca8f32135548ded56adec46c6d524fd0465086e2120322f9f48d97ff.jpg)

![d042c7dfd1d84ab9c89446ba0aa265fbb03b1cc1cbb15732b50f1c74e65ddc73.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/d042c7dfd1d84ab9c89446ba0aa265fbb03b1cc1cbb15732b50f1c74e65ddc73.jpg)

![e829e9e725abe64660f4f410c2e0ee70dd25acd5b5a8325c7bf626271c944ca8.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/e829e9e725abe64660f4f410c2e0ee70dd25acd5b5a8325c7bf626271c944ca8.jpg)

![f00dd7e82a8a9cbd2a128e3a4020706e750f78074596c4550cad6e57ae74a6f4.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/f00dd7e82a8a9cbd2a128e3a4020706e750f78074596c4550cad6e57ae74a6f4.jpg)

![f6cca291ee1d4ddd855229080f2669d1f2917797709d1eb8912106200eb972fb.jpg](../icml_results/1018_Code2Video_%20A%20Code-centric%20Paradigm%20for%20Educational%20Video%20Creation/tables/f6cca291ee1d4ddd855229080f2669d1f2917797709d1eb8912106200eb972fb.jpg)

<a id="paper-1002"></a>

## Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling


### Images

![262d31c1e48a3e7b834a9e2906d394850ce689f47e68561c2d48709c92b959d1.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/262d31c1e48a3e7b834a9e2906d394850ce689f47e68561c2d48709c92b959d1.jpg)

![5b29b23bf69bb4b43df78b775d2da05b97de69241c18d96452083d48c9b5e543.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/5b29b23bf69bb4b43df78b775d2da05b97de69241c18d96452083d48c9b5e543.jpg)

![5fba3f4afd354cb8346ab6ac71c3acbd3f81a8b705cca4faae8624070bb4fcd9.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/5fba3f4afd354cb8346ab6ac71c3acbd3f81a8b705cca4faae8624070bb4fcd9.jpg)

![6a4fe8866cbaa2443534d02fffaa9862d9af90b7894e48c8685e5970e620b470.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/6a4fe8866cbaa2443534d02fffaa9862d9af90b7894e48c8685e5970e620b470.jpg)

![6f447d9f3f06704fddc6fdba45ad36f9cb8f0943e46b9997186841760564fcb3.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/6f447d9f3f06704fddc6fdba45ad36f9cb8f0943e46b9997186841760564fcb3.jpg)

![87912fafb3237af78155b8ca9d7f548f8654d5661226e4100773d83288fe434c.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/87912fafb3237af78155b8ca9d7f548f8654d5661226e4100773d83288fe434c.jpg)

![a55770add7c210f89bd4c6a9e72c7173306d93512a4631a6975487fe70b5a8c0.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/a55770add7c210f89bd4c6a9e72c7173306d93512a4631a6975487fe70b5a8c0.jpg)

![bed721ed6f927a6a7c0a1aa9abc37d5c57ce6bb805c66884cc1ddc71db9a5ee5.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/bed721ed6f927a6a7c0a1aa9abc37d5c57ce6bb805c66884cc1ddc71db9a5ee5.jpg)

![e9d2974862a439c8a914c54a00f70925c343c74e20ef2412d9a4d97c4c4c2eed.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/images/e9d2974862a439c8a914c54a00f70925c343c74e20ef2412d9a4d97c4c4c2eed.jpg)

### Tables

![5911651d47e0497f375d4532a60695c0a1aff53aa5f92782970ba82fe3b6b4be.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/5911651d47e0497f375d4532a60695c0a1aff53aa5f92782970ba82fe3b6b4be.jpg)

![745d5d9266c27ab9e71e911d0bbbe916433af77bd3a19983ea4ad2d9084cc02a.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/745d5d9266c27ab9e71e911d0bbbe916433af77bd3a19983ea4ad2d9084cc02a.jpg)

![88c22923308a5c6aecc8dd91548180175e1e05ef3b48133254dfe843fd39a820.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/88c22923308a5c6aecc8dd91548180175e1e05ef3b48133254dfe843fd39a820.jpg)

![91a87396ec8e0c76df7e0d1316a3041f46d79ab9ac82ebf2c21828c14bc4b549.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/91a87396ec8e0c76df7e0d1316a3041f46d79ab9ac82ebf2c21828c14bc4b549.jpg)

![abbd92c1aaf0b7be1f270cca29623782d4286121e252fb56ff721ac32646de3c.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/abbd92c1aaf0b7be1f270cca29623782d4286121e252fb56ff721ac32646de3c.jpg)

![dec9546bee55842c3c185d6b1b1c004fe931e9a129f7bc42fd450b726d06c060.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/dec9546bee55842c3c185d6b1b1c004fe931e9a129f7bc42fd450b726d06c060.jpg)

![fdf713a795ec1b0b375ec286076b2d6235ff8a79128b3b1a626baf5e50831f40.jpg](../icml_results/1019_Rotation-Invariant%20Spherical%20Watermarking%20via%20Third-Order%20SO%283%29%20Representation%20Coupling/tables/fdf713a795ec1b0b375ec286076b2d6235ff8a79128b3b1a626baf5e50831f40.jpg)

<a id="paper-1003"></a>

## Threat2Traffic: Multi-Agent Environment Synthesis for Malware Traffic Generation from Threat Intelligence


### Images

![1629a21ef93e00cb63e2664ced2ac6c4e296de2912a3fd51c8b6916be767fb1b.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/1629a21ef93e00cb63e2664ced2ac6c4e296de2912a3fd51c8b6916be767fb1b.jpg)

![217fe0ac515c2026a5ba4100426333c720584e8481969dc2a56b7d4a9342f884.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/217fe0ac515c2026a5ba4100426333c720584e8481969dc2a56b7d4a9342f884.jpg)

![2fcda889066226f9c17861417e2a65ccada7def72433e79a316db50bf4d1f849.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/2fcda889066226f9c17861417e2a65ccada7def72433e79a316db50bf4d1f849.jpg)

![32726c00c9effa704cf0d25e7cabe22c10f73ea04a05ae0fb756393ab2cd3ec5.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/32726c00c9effa704cf0d25e7cabe22c10f73ea04a05ae0fb756393ab2cd3ec5.jpg)

![7578a642985d3aaaa2ef2000e07d671bc348195c9aba9d4b8ec9e978af6a2830.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/7578a642985d3aaaa2ef2000e07d671bc348195c9aba9d4b8ec9e978af6a2830.jpg)

![9379fd81bddcd5173e6f445e1dee7a75abd0a8bd2685d9e141417598fd0ee575.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/9379fd81bddcd5173e6f445e1dee7a75abd0a8bd2685d9e141417598fd0ee575.jpg)

![b7c7ba6d1299ccc34a828379264bce4f1861b93647d2794a408b708bd84cd030.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/b7c7ba6d1299ccc34a828379264bce4f1861b93647d2794a408b708bd84cd030.jpg)

![d3b8ee43691135fc2bdc7d6b9972943b6bb39d4cb6f52f863048a050db861fb0.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/d3b8ee43691135fc2bdc7d6b9972943b6bb39d4cb6f52f863048a050db861fb0.jpg)

![f9003579201f56228ef50ed0bda0b78e0bc964af34f3a3fec618c306aaa4f6f0.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/images/f9003579201f56228ef50ed0bda0b78e0bc964af34f3a3fec618c306aaa4f6f0.jpg)

### Tables

![182b371dc1f5e89362177dbafb1071df8827c2724b057b67417e922d12c1847e.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/182b371dc1f5e89362177dbafb1071df8827c2724b057b67417e922d12c1847e.jpg)

![1c477a270f70f0c15be92513f2471793497a953e8141d35f97c87c2ab4acbd99.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/1c477a270f70f0c15be92513f2471793497a953e8141d35f97c87c2ab4acbd99.jpg)

![2769b6f5025ef4e2fb042f0ce43be0ef339afc003048daf83b4bbe0b17700667.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/2769b6f5025ef4e2fb042f0ce43be0ef339afc003048daf83b4bbe0b17700667.jpg)

![442233d3f913167dbc55c8b898bd337c9ecdebe6cdb1bd88ae25e9abc5cb1241.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/442233d3f913167dbc55c8b898bd337c9ecdebe6cdb1bd88ae25e9abc5cb1241.jpg)

![4cf31d06a6f5f31e77b538baec4f8f53dae25dd69dfdf00ee9bd28117f814dad.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/4cf31d06a6f5f31e77b538baec4f8f53dae25dd69dfdf00ee9bd28117f814dad.jpg)

![51376793369b0cecfdf84c9cbc73ccff2772591e4d02df45b1db9b8410a53fc5.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/51376793369b0cecfdf84c9cbc73ccff2772591e4d02df45b1db9b8410a53fc5.jpg)

![9aac5dbe8be418e579ba5e69a2b85dbd8690d09821702d49460bb7785d9439e7.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/9aac5dbe8be418e579ba5e69a2b85dbd8690d09821702d49460bb7785d9439e7.jpg)

![b59d0f0ac5e5db9f0a05d007f835ef2c427b7647cce590883e3eeb35f189a06b.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/b59d0f0ac5e5db9f0a05d007f835ef2c427b7647cce590883e3eeb35f189a06b.jpg)

![bea629f73f9e8976c616fd69920a40a5b9f94f06679280497b038033e947668c.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/bea629f73f9e8976c616fd69920a40a5b9f94f06679280497b038033e947668c.jpg)

![c692cce85e7df0afd06d9f54b5542c0f8b625d63366907cf07e0689c91bef38f.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/c692cce85e7df0afd06d9f54b5542c0f8b625d63366907cf07e0689c91bef38f.jpg)

![edba198c12a58a867191d8b3c45924283a5881e87b9f2cdbd3d7a5914d2a4fe8.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/edba198c12a58a867191d8b3c45924283a5881e87b9f2cdbd3d7a5914d2a4fe8.jpg)

![fb8993b0a695ff27b85de6456c3a8e1f044d8edc6ff7a1fc0f0883a5f73445cb.jpg](../icml_results/1020_Threat2Traffic_%20Multi-Agent%20Environment%20Synthesis%20for%20Malware%20Traffic%20Generation%20from%20Threat%20Intelli/tables/fb8993b0a695ff27b85de6456c3a8e1f044d8edc6ff7a1fc0f0883a5f73445cb.jpg)

<a id="paper-1004"></a>

## ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning


### Images

![495fdd7a1cc2d8a01d9ebb1b480a5a637af815c39684a7379959b898012e2e40.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/images/495fdd7a1cc2d8a01d9ebb1b480a5a637af815c39684a7379959b898012e2e40.jpg)

![b358a19151ce1539d51da3fe7b32aa7f7d81084c4793e6a25ed9bffe1b0d7a20.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/images/b358a19151ce1539d51da3fe7b32aa7f7d81084c4793e6a25ed9bffe1b0d7a20.jpg)

![e3d90d6364621725f11748ff4d4e2477abfb4e86334f1c4efaab9f7ef2e980a1.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/images/e3d90d6364621725f11748ff4d4e2477abfb4e86334f1c4efaab9f7ef2e980a1.jpg)

### Tables

![1388c7062d0c11985d8f403093ca2d000ed7e991448484ba9633fa983b1e4eba.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/1388c7062d0c11985d8f403093ca2d000ed7e991448484ba9633fa983b1e4eba.jpg)

![399312c129af71d1075b959a3565c8565e3c7f38637c878cddfadacb9dcd057d.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/399312c129af71d1075b959a3565c8565e3c7f38637c878cddfadacb9dcd057d.jpg)

![492074bb366b2245901f4ac5b56f9cff91539ac726458ddd3d622efc44fbc677.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/492074bb366b2245901f4ac5b56f9cff91539ac726458ddd3d622efc44fbc677.jpg)

![6ea4e6be534dcb5297cbb97df91d0dd1e45601cbf83e8c54945c05cb60a95afe.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/6ea4e6be534dcb5297cbb97df91d0dd1e45601cbf83e8c54945c05cb60a95afe.jpg)

![7ce8e65871bc7690e7225181d8078378d4c69b2c3101278dc40d50a46589f2b0.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/7ce8e65871bc7690e7225181d8078378d4c69b2c3101278dc40d50a46589f2b0.jpg)

![7fa869fd16cf8209f533a774db851a3c44e9ee8b651c480180f081834da4c25a.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/7fa869fd16cf8209f533a774db851a3c44e9ee8b651c480180f081834da4c25a.jpg)

![c36fec848da797a04f6b7b5be7f5dac3972d3ef525e4c2cc2196eadc1eda0f05.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/c36fec848da797a04f6b7b5be7f5dac3972d3ef525e4c2cc2196eadc1eda0f05.jpg)

![cd450542d28687a3b4b43d7adbaa6ef47cc598e8b8fc2613816ce85c127b2af4.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/cd450542d28687a3b4b43d7adbaa6ef47cc598e8b8fc2613816ce85c127b2af4.jpg)

![da3c3d5d839616e0b8fd2d4d5f27bb5f664cb170e57338717f34317de8bb314a.jpg](../icml_results/1022_ArborKV_%20Structure-Aware%20KV%20Cache%20Management%20for%20Scaling%20Tree-based%20LLM%20Reasoning/tables/da3c3d5d839616e0b8fd2d4d5f27bb5f664cb170e57338717f34317de8bb314a.jpg)

<a id="paper-1005"></a>

## WEVSR: Video Diffusion Generators for Real-World Video Super-Resolution with Wavelet-Enhanced VAE Encoder


### Images

![0b4683e3448934aaa135b48ed2c87ea7f67a053e3770591273a97c4bc85ee998.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/0b4683e3448934aaa135b48ed2c87ea7f67a053e3770591273a97c4bc85ee998.jpg)

![1823b6ed16ce43065920460d16b1b1536e27c9fe885d34f5540fbb233f8b6138.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/1823b6ed16ce43065920460d16b1b1536e27c9fe885d34f5540fbb233f8b6138.jpg)

![1b3f10e7771d32083967f5dc9ea0bbaf6b1037ca66732cf658779b6064bfc06a.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/1b3f10e7771d32083967f5dc9ea0bbaf6b1037ca66732cf658779b6064bfc06a.jpg)

![240e667892ca76094e172eec879b60392949a39e88dfc2ba26d033b8f2bef883.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/240e667892ca76094e172eec879b60392949a39e88dfc2ba26d033b8f2bef883.jpg)

![c528c624ee74bd1fc4ea7151bd2a34f03be309710b3075b85acb21abda3bce19.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/c528c624ee74bd1fc4ea7151bd2a34f03be309710b3075b85acb21abda3bce19.jpg)

![ce3b926e161287c0bb2e4edadf3eb207e7b9374fefd61bbfa2b28a1e555f6775.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/ce3b926e161287c0bb2e4edadf3eb207e7b9374fefd61bbfa2b28a1e555f6775.jpg)

![f61002b4d869246dc5f46c3c866f3b1d523100d6019010c199bb04aa9b88e9f7.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/images/f61002b4d869246dc5f46c3c866f3b1d523100d6019010c199bb04aa9b88e9f7.jpg)

### Tables

![16fe9bbde66eb343f4e8a3cc1906e23bcc2230c1ee4bc627a13d0c2536c4c395.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/tables/16fe9bbde66eb343f4e8a3cc1906e23bcc2230c1ee4bc627a13d0c2536c4c395.jpg)

![3a7b32393121be1a12d1ecc13b8abf8ee8ca50f689468b707cb405974da7b212.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/tables/3a7b32393121be1a12d1ecc13b8abf8ee8ca50f689468b707cb405974da7b212.jpg)

![fc456b031f38313ab5418d44dcf2ef3725484f11ce0e4e9f9a85533e30476f08.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/tables/fc456b031f38313ab5418d44dcf2ef3725484f11ce0e4e9f9a85533e30476f08.jpg)

![feb2d8d6b891fb92238355a191f6368f05d41525d49b53a0f2f6ce7d50d62bac.jpg](../icml_results/1023_WEVSR_%20Video%20Diffusion%20Generators%20for%20Real-World%20Video%20Super-Resolution%20with%20Wavelet-Enhanced%20VAE%20En/tables/feb2d8d6b891fb92238355a191f6368f05d41525d49b53a0f2f6ce7d50d62bac.jpg)

<a id="paper-1006"></a>

## InteractScience: Programmatic and Visually-Grounded Evaluation of Interactive Scientific Demonstration Code Generation


### Images

![1587eb2ea707090c996288262324ced4b9e303b5a1871526f1349e3fa6c30484.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/1587eb2ea707090c996288262324ced4b9e303b5a1871526f1349e3fa6c30484.jpg)

![158aa9cc05e13a495c07f8bcc539c03cb41e1f6261cd42572024b37a45bc97de.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/158aa9cc05e13a495c07f8bcc539c03cb41e1f6261cd42572024b37a45bc97de.jpg)

![28a1553352795d2b1429f851312324560333e3d6a5c939de37686e4810b66772.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/28a1553352795d2b1429f851312324560333e3d6a5c939de37686e4810b66772.jpg)

![4cdd9c6825ab0a842b050bf0ba7f2225f170f1bcb437d594be71fe45d1d361a8.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/4cdd9c6825ab0a842b050bf0ba7f2225f170f1bcb437d594be71fe45d1d361a8.jpg)

![6af3272eaa4cb728ecd3ca2a486c3995479680fb5014e7598964095bd4fab738.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/6af3272eaa4cb728ecd3ca2a486c3995479680fb5014e7598964095bd4fab738.jpg)

![6bbc0e3b29946990eee50f5d18e922d567fc475becbe2431b951f94bf6bc9651.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/6bbc0e3b29946990eee50f5d18e922d567fc475becbe2431b951f94bf6bc9651.jpg)

![972651ed2788d7069d67a1c11215c45e4a915e8027136ac10772b6a03c79739e.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/972651ed2788d7069d67a1c11215c45e4a915e8027136ac10772b6a03c79739e.jpg)

![a536a674dfd6de1afd86225cbf950a6e7a9d7dd1ffc12485fde924d0c7c2c0d0.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/a536a674dfd6de1afd86225cbf950a6e7a9d7dd1ffc12485fde924d0c7c2c0d0.jpg)

![e727afe65adc0c9fb9e47d486b2497d5a576827a9ea8eaa71563c3d58ac4dbb3.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/images/e727afe65adc0c9fb9e47d486b2497d5a576827a9ea8eaa71563c3d58ac4dbb3.jpg)

### Tables

![011f1b4dc1cd49c7e320eaec6f0065d4c0cad808bbe4f89aa5d8d4915e894e31.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/011f1b4dc1cd49c7e320eaec6f0065d4c0cad808bbe4f89aa5d8d4915e894e31.jpg)

![1b1f4dfbe40cc2ae8b833585747da05ec0e0010f757dbf325341765445f82d3c.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/1b1f4dfbe40cc2ae8b833585747da05ec0e0010f757dbf325341765445f82d3c.jpg)

![4dd74c25ae672aa21fc7107103d46b355b5dbd94d2ddc226ce07f515f172e719.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/4dd74c25ae672aa21fc7107103d46b355b5dbd94d2ddc226ce07f515f172e719.jpg)

![50a7cd951c3247c650a4be75b2a0dbfa67435de878d129f0c981b93deaa17bcd.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/50a7cd951c3247c650a4be75b2a0dbfa67435de878d129f0c981b93deaa17bcd.jpg)

![68215baef39a013600aebb066f5292d65b85b34ae921d829c344ff8eca37fe8b.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/68215baef39a013600aebb066f5292d65b85b34ae921d829c344ff8eca37fe8b.jpg)

![9c4a2110a64ceaf59b1a2f2f99691fc1fd8ae1c6b2070fe9fb312da029720332.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/9c4a2110a64ceaf59b1a2f2f99691fc1fd8ae1c6b2070fe9fb312da029720332.jpg)

![9e9e87437f507abf2e7be57aa921e8dc1e7862d72dfd494146cb8b2a274f0492.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/9e9e87437f507abf2e7be57aa921e8dc1e7862d72dfd494146cb8b2a274f0492.jpg)

![9f9f23e34946d7cc6c70d69a395ffddfeb1a6c5870daa54871651de5b7f074bd.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/9f9f23e34946d7cc6c70d69a395ffddfeb1a6c5870daa54871651de5b7f074bd.jpg)

![d54f0636deacbcccffe576f78931a90e9cc9402057750fb6b0c0bf02a924e4f5.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/d54f0636deacbcccffe576f78931a90e9cc9402057750fb6b0c0bf02a924e4f5.jpg)

![f61af46a15e01cb6e1a04a611dab0a5e5dd55b9a10e9a618293d0779ca049a69.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/f61af46a15e01cb6e1a04a611dab0a5e5dd55b9a10e9a618293d0779ca049a69.jpg)

![f6344363ae601d0dc7c990b2d28d98cfe1e63aa12a0ded8b256fa05937945154.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/f6344363ae601d0dc7c990b2d28d98cfe1e63aa12a0ded8b256fa05937945154.jpg)

![f7c988df406ca8b07f386d845fcd6374185f6f340c6257326f26b6dc509c2607.jpg](../icml_results/1024_InteractScience_%20Programmatic%20and%20Visually-Grounded%20Evaluation%20of%20Interactive%20Scientific%20Demonstrati/tables/f7c988df406ca8b07f386d845fcd6374185f6f340c6257326f26b6dc509c2607.jpg)

<a id="paper-1007"></a>

## LoKiFormer: Locality-aware Attention with Decoupled Knowledge Memory for Efficient Large Language Model Pretraining


### Images

![5c702663f6a31b906dbe71815f9d95bb8daa3063457ea8b2b9929384f1ba1c65.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/5c702663f6a31b906dbe71815f9d95bb8daa3063457ea8b2b9929384f1ba1c65.jpg)

![8560a04df8786cb28d98c23bafdf5620a14099024236bcaa427540fcf0909c47.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/8560a04df8786cb28d98c23bafdf5620a14099024236bcaa427540fcf0909c47.jpg)

![865b18284014ef5bb04b56b68da3fce57887cb7eb1587b0bd748a3f56d47eed5.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/865b18284014ef5bb04b56b68da3fce57887cb7eb1587b0bd748a3f56d47eed5.jpg)

![868feae08ec53879bbd3b250c5bce775c7c3e6a6edb3d76913a441b30a1f94e0.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/868feae08ec53879bbd3b250c5bce775c7c3e6a6edb3d76913a441b30a1f94e0.jpg)

![87d508baa7f07f87ffb070343c0740ceb1ff1f422594e3b7cf8ef125c9b054e1.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/87d508baa7f07f87ffb070343c0740ceb1ff1f422594e3b7cf8ef125c9b054e1.jpg)

![9e3acfd5237313e2966526567b7975eef16da552f9e6d0e3a665840ac7f0fd91.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/9e3acfd5237313e2966526567b7975eef16da552f9e6d0e3a665840ac7f0fd91.jpg)

![a3be15c8d0aac6327561db5e988ed672b0300b4fae23118de2e4585c27e7977b.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/a3be15c8d0aac6327561db5e988ed672b0300b4fae23118de2e4585c27e7977b.jpg)

![b050df09391eab0c44e261678233b68e10c1b1e19f901600d2d028d21f6e889d.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/b050df09391eab0c44e261678233b68e10c1b1e19f901600d2d028d21f6e889d.jpg)

![bca6b466247acdf1ad1fcac09e08f4ce9890f20d21d830b11db1218d0d94c83e.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/bca6b466247acdf1ad1fcac09e08f4ce9890f20d21d830b11db1218d0d94c83e.jpg)

![c4d89bae309cdf35cec5105a9c619b562382f01fecf177b003a20d0fb31cd0b0.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/c4d89bae309cdf35cec5105a9c619b562382f01fecf177b003a20d0fb31cd0b0.jpg)

![f4249714969ca3093bf7f4481e3a157fc98fcdd49fe14fcf32fad57ebd4b8876.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/images/f4249714969ca3093bf7f4481e3a157fc98fcdd49fe14fcf32fad57ebd4b8876.jpg)

### Tables

![1593f4140b3f2b890e6331ba01773ab7518857df8605cebb071d780c6433bf04.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/1593f4140b3f2b890e6331ba01773ab7518857df8605cebb071d780c6433bf04.jpg)

![1e628d2ae56f3fd3f0685a437ceab516c962f0a6a67612edb4b07a3b13a46680.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/1e628d2ae56f3fd3f0685a437ceab516c962f0a6a67612edb4b07a3b13a46680.jpg)

![49302a91aabf3a4d21416f8f6cc4266a24f950c7c189655ec8afefa0e4b7017e.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/49302a91aabf3a4d21416f8f6cc4266a24f950c7c189655ec8afefa0e4b7017e.jpg)

![4a50e3a7582dd2292fc5af5c712f64be3c64098cb2ee89cea77b58811fe5837f.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/4a50e3a7582dd2292fc5af5c712f64be3c64098cb2ee89cea77b58811fe5837f.jpg)

![944538cfbaadb18a49192c2ad2cbb04503160b1706af82b04048da6c9be173f6.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/944538cfbaadb18a49192c2ad2cbb04503160b1706af82b04048da6c9be173f6.jpg)

![a8c37a83702797d49248d85dcfdc1e985f6f51db41be9b750ef61a7e6d373bb6.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/a8c37a83702797d49248d85dcfdc1e985f6f51db41be9b750ef61a7e6d373bb6.jpg)

![bf524061f2a67f4c9d11330c81ca9a603ebae4265e6e4875a99294b4bf407f01.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/bf524061f2a67f4c9d11330c81ca9a603ebae4265e6e4875a99294b4bf407f01.jpg)

![f7d3f6b8750c70423aeb18942726a6dec6ccfec4b35e25834a79887120b61e72.jpg](../icml_results/1025_LoKiFormer_%20Locality-aware%20Attention%20with%20Decoupled%20Knowledge%20Memory%20for%20Efficient%20Large%20Language%20Mo/tables/f7d3f6b8750c70423aeb18942726a6dec6ccfec4b35e25834a79887120b61e72.jpg)

<a id="paper-1008"></a>

## Hamiltonian Asymmetric Fusion: One-Way Safe Directed Refinement under Modality Imbalance


### Images

![04f41c0a6461d21e48c48f9ba9b83c451131aa4b0b82cf0b73878a1aedd0f311.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/04f41c0a6461d21e48c48f9ba9b83c451131aa4b0b82cf0b73878a1aedd0f311.jpg)

![069d90bc3890518bfaa8db39ce6bc79d9c5578134cc2ef7fbc9a29e768360329.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/069d90bc3890518bfaa8db39ce6bc79d9c5578134cc2ef7fbc9a29e768360329.jpg)

![0719ea84b67370d4f9938dfcc3ecde3799605e7f19be0e145fef0d0abab6b511.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/0719ea84b67370d4f9938dfcc3ecde3799605e7f19be0e145fef0d0abab6b511.jpg)

![4186b99ec2eb4b16160a2b31231a03a8f9687d79da5a504ff8e54a5f89921f82.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/4186b99ec2eb4b16160a2b31231a03a8f9687d79da5a504ff8e54a5f89921f82.jpg)

![6bd0861f5b199881521ac2ff100ce66cd5d20ac4886ac620c83d6f1ed2fd88f4.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/6bd0861f5b199881521ac2ff100ce66cd5d20ac4886ac620c83d6f1ed2fd88f4.jpg)

![9a54dd1cd578c29487ae17be86c9b6c48bf7dc271beb6c4f2d14613cd98e0024.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/9a54dd1cd578c29487ae17be86c9b6c48bf7dc271beb6c4f2d14613cd98e0024.jpg)

![9a58800b609da7b27c05cc3a50bb258c1840875fa3c2653ae16f76eef195385d.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/9a58800b609da7b27c05cc3a50bb258c1840875fa3c2653ae16f76eef195385d.jpg)

![d8a3a33bf88d63ba008de9e66aed7e02109052b69a21d38fecd26c97c1274b43.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/d8a3a33bf88d63ba008de9e66aed7e02109052b69a21d38fecd26c97c1274b43.jpg)

![df0ea6ffaa7cd28a5b34123d08d50683fdfb106af2cdfadb13464bd8891918bb.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/images/df0ea6ffaa7cd28a5b34123d08d50683fdfb106af2cdfadb13464bd8891918bb.jpg)

### Tables

![0836e683eeef678c940ce21fb26b7491f09f1889321832010bd16dd231604025.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/0836e683eeef678c940ce21fb26b7491f09f1889321832010bd16dd231604025.jpg)

![1215296564509b597da66b506b2f7a7326051e41dacb610da20ecfb9a258adf5.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/1215296564509b597da66b506b2f7a7326051e41dacb610da20ecfb9a258adf5.jpg)

![19c8bff44e444e9070150b0264a75548066bbd5d4221b46c5711fb303abbb25a.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/19c8bff44e444e9070150b0264a75548066bbd5d4221b46c5711fb303abbb25a.jpg)

![232c14fc226e72b491608a121f39c3f47c7e6d93118bab5e72f6fbc1d67032da.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/232c14fc226e72b491608a121f39c3f47c7e6d93118bab5e72f6fbc1d67032da.jpg)

![381c9f238ae3fac7d6f43318d390bc7c51d7d5b22242bdf77f8197ef2c2f47af.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/381c9f238ae3fac7d6f43318d390bc7c51d7d5b22242bdf77f8197ef2c2f47af.jpg)

![420c71132117c74c28e20dc18817d7737608fdc94af977106bcc09aa9ac290c5.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/420c71132117c74c28e20dc18817d7737608fdc94af977106bcc09aa9ac290c5.jpg)

![4795561083cf3e8a1ec32ba85ac2491c7ae7362110786cda4a78e5306101b69e.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/4795561083cf3e8a1ec32ba85ac2491c7ae7362110786cda4a78e5306101b69e.jpg)

![7d6e8251912cf84fda7352bf6f74eff0e4d043b29d3f83dbe9609d1c9bf36e1d.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/7d6e8251912cf84fda7352bf6f74eff0e4d043b29d3f83dbe9609d1c9bf36e1d.jpg)

![c1dd078445143d4127f1e58b1b44c97944dffe2553176a47ea7f6ccb53ad8460.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/c1dd078445143d4127f1e58b1b44c97944dffe2553176a47ea7f6ccb53ad8460.jpg)

![ec1a7aa687d8a3e40d2748e122790a2ccfeed8165f75d44048465b3c474063b5.jpg](../icml_results/1026_Hamiltonian%20Asymmetric%20Fusion_%20One-Way%20Safe%20Directed%20Refinement%20under%20Modality%20Imbalance/tables/ec1a7aa687d8a3e40d2748e122790a2ccfeed8165f75d44048465b3c474063b5.jpg)

<a id="paper-1009"></a>

## Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models


### Images

![0d9e96bde1374159f5a9fef89ccbeb46f3f319d082bc00f2be1ea3929d84c5a2.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/0d9e96bde1374159f5a9fef89ccbeb46f3f319d082bc00f2be1ea3929d84c5a2.jpg)

![25bf768a26db7095dda99a4604fce71faf935c60675aa91b38f596fd73c315c3.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/25bf768a26db7095dda99a4604fce71faf935c60675aa91b38f596fd73c315c3.jpg)

![400094e2312b4b87bd919888de11a4c69f996e95fb044f8e767ce9b1ad16de0a.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/400094e2312b4b87bd919888de11a4c69f996e95fb044f8e767ce9b1ad16de0a.jpg)

![5dbde261220c6bf11d405fcfc64be033c861f83d53cc3d597262321bb77ca1f6.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/5dbde261220c6bf11d405fcfc64be033c861f83d53cc3d597262321bb77ca1f6.jpg)

![746d7cd3227b4f7176e31ffbd719396fc8a6a529bb6e9eb8bf5bace369c06e5b.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/746d7cd3227b4f7176e31ffbd719396fc8a6a529bb6e9eb8bf5bace369c06e5b.jpg)

![89d5bd9c8f922bebf9abb8c0d7b52a497a682288213b4e22086c72dfedfc5ae5.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/89d5bd9c8f922bebf9abb8c0d7b52a497a682288213b4e22086c72dfedfc5ae5.jpg)

![8f54866f7884bebdb5aee0757f044707a8f51309c5e9f65d9ee4a2efdd417cc6.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/8f54866f7884bebdb5aee0757f044707a8f51309c5e9f65d9ee4a2efdd417cc6.jpg)

![9415eb59c1eed933d8c6739a53078dba87cb2947f5e9a3a3ace7891dfd2c86a9.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/9415eb59c1eed933d8c6739a53078dba87cb2947f5e9a3a3ace7891dfd2c86a9.jpg)

![b6ea9630fd37254e261f8a5446c5e917bf24798cb9b1ff465939c01f920e861d.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/b6ea9630fd37254e261f8a5446c5e917bf24798cb9b1ff465939c01f920e861d.jpg)

![d5976c21734ac280c3a352db3a0c7b3a6ee0563d130045c147001cf4bd96607d.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/d5976c21734ac280c3a352db3a0c7b3a6ee0563d130045c147001cf4bd96607d.jpg)

![d65bffaa947359fb4aa13ab74a1f8a54fdfec163f145e6df48229600a3f28b2a.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/d65bffaa947359fb4aa13ab74a1f8a54fdfec163f145e6df48229600a3f28b2a.jpg)

![f4e20100bb3385d7dee6595dd956821cf5666074c8377f7c8306aca2ed0e6711.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/f4e20100bb3385d7dee6595dd956821cf5666074c8377f7c8306aca2ed0e6711.jpg)

![f96f39b99558e7752c700e85a916de304ca414405dfd14ee596644a6bbe71fc5.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/images/f96f39b99558e7752c700e85a916de304ca414405dfd14ee596644a6bbe71fc5.jpg)

### Tables

![197d9295f3c80ccdce4b9cb38fb1be33929a8d2cb683edd572af9137f8828348.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/197d9295f3c80ccdce4b9cb38fb1be33929a8d2cb683edd572af9137f8828348.jpg)

![19c3e2c77b812075c59f0b3a2183806f74a11c6dafbedd9952b589b430292f20.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/19c3e2c77b812075c59f0b3a2183806f74a11c6dafbedd9952b589b430292f20.jpg)

![4cba3c9d60cf339178b8597cc4ac01a773349744e4333fc12a478742e7ef890d.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/4cba3c9d60cf339178b8597cc4ac01a773349744e4333fc12a478742e7ef890d.jpg)

![4d72e863be086fe2bcb120c8f3fe6acae7154a5c9f2b1db96b815da1685d37a7.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/4d72e863be086fe2bcb120c8f3fe6acae7154a5c9f2b1db96b815da1685d37a7.jpg)

![580f98dd07f59c685ef6441a8f27f6b2e64d73b1087cd22e9d8741021a4bf9dc.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/580f98dd07f59c685ef6441a8f27f6b2e64d73b1087cd22e9d8741021a4bf9dc.jpg)

![831cd15a506268aee40c7217d3c7add54bae377b33110701639d35dfcb127ee7.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/831cd15a506268aee40c7217d3c7add54bae377b33110701639d35dfcb127ee7.jpg)

![d9502520bee5b51b77d111bda234a7c57ac7d670f402885ada050bedeed230ed.jpg](../icml_results/1027_Beyond%20Confidence_%20Adaptive%20and%20Coherent%20Decoding%20for%20Diffusion%20Language%20Models/tables/d9502520bee5b51b77d111bda234a7c57ac7d670f402885ada050bedeed230ed.jpg)

<a id="paper-1010"></a>

## Learning-Guided Integration Contours Construction for Fast Large-Scale Generalized Eigensolvers


### Images

![0b064cfaac1e5fb9df88754c116d6ab78e28e0a9033548dee6b6ebedf77ad16f.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/0b064cfaac1e5fb9df88754c116d6ab78e28e0a9033548dee6b6ebedf77ad16f.jpg)

![3cdec678906d70edb6bab844747d61a16f5c4a6536354fabcf4e1d3df1f15975.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/3cdec678906d70edb6bab844747d61a16f5c4a6536354fabcf4e1d3df1f15975.jpg)

![67ddba688be50f8ee198c8f4f2c038e4081d75f8737aa35b1200d8d4a49931fa.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/67ddba688be50f8ee198c8f4f2c038e4081d75f8737aa35b1200d8d4a49931fa.jpg)

![83d91a35a2aabe0843019a8104b542d8cf9e68520978bb4a3050eb4d3d9c6dd2.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/83d91a35a2aabe0843019a8104b542d8cf9e68520978bb4a3050eb4d3d9c6dd2.jpg)

![8563bcf6a35a2c3daecee0f8ef42614fb92b9e83cd6df192401233571b69cd2e.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/8563bcf6a35a2c3daecee0f8ef42614fb92b9e83cd6df192401233571b69cd2e.jpg)

![869260d0018b4c56f9dd3baa17f9fc2872c488f8846695bfe214329251f3bcf5.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/869260d0018b4c56f9dd3baa17f9fc2872c488f8846695bfe214329251f3bcf5.jpg)

![aecaf039d2f90e1ddffe04dc570ae9dbc734b1298ae421a37f0c56820acf4426.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/aecaf039d2f90e1ddffe04dc570ae9dbc734b1298ae421a37f0c56820acf4426.jpg)

![f412132979209cd7da48fa1784e4e3ce5c8c1704a298e3ac89b7901a7e947210.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/images/f412132979209cd7da48fa1784e4e3ce5c8c1704a298e3ac89b7901a7e947210.jpg)

### Tables

![16089078a8612e4362c87fbfcc78c0711a5c16de5087b004edc4a76206215d9d.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/16089078a8612e4362c87fbfcc78c0711a5c16de5087b004edc4a76206215d9d.jpg)

![16a03f2de1b4fb83e1b6ccdb3bd88bfb0e3f5d88d9ddd0a39721e35a4e8e3f35.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/16a03f2de1b4fb83e1b6ccdb3bd88bfb0e3f5d88d9ddd0a39721e35a4e8e3f35.jpg)

![22d3009ae8eadab6cf9cadd3eb943893b29353d595243cf352dfe7a0433c76d2.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/22d3009ae8eadab6cf9cadd3eb943893b29353d595243cf352dfe7a0433c76d2.jpg)

![4c28fb2a4d04ba70f4471c6dbfd57fc4524c93cc4f06853f2fc5759e9a25d1b6.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/4c28fb2a4d04ba70f4471c6dbfd57fc4524c93cc4f06853f2fc5759e9a25d1b6.jpg)

![4f508a6ed2192cf5ed56579980415a1c41bb4117e1e2d24c01e14127aef44366.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/4f508a6ed2192cf5ed56579980415a1c41bb4117e1e2d24c01e14127aef44366.jpg)

![59ae45b55d2654f65602771b98d7d5c0016803d78ecd0120f045c63ee6b3301c.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/59ae45b55d2654f65602771b98d7d5c0016803d78ecd0120f045c63ee6b3301c.jpg)

![6adefc7d8e685159666828d906d9be722e4308f24d3d216697d439428e09b0e5.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/6adefc7d8e685159666828d906d9be722e4308f24d3d216697d439428e09b0e5.jpg)

![6c40efc86f63a75ee5892c2e9b8de8dafbc57b1306146877ee34130249907e0a.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/6c40efc86f63a75ee5892c2e9b8de8dafbc57b1306146877ee34130249907e0a.jpg)

![6c74ae2dfb51104d2f449f7728db519dd7c7e5ceb00af40ebe6fc4c5b7cd711a.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/6c74ae2dfb51104d2f449f7728db519dd7c7e5ceb00af40ebe6fc4c5b7cd711a.jpg)

![79c41426a6877fcfb48bfa92a1c1e359e1c82097bb168ff7bf59ba0db4b3cb0b.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/79c41426a6877fcfb48bfa92a1c1e359e1c82097bb168ff7bf59ba0db4b3cb0b.jpg)

![972e80aa83c018991d87f2839aea5b9458673c08527c1d24dc8a420c8886a055.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/972e80aa83c018991d87f2839aea5b9458673c08527c1d24dc8a420c8886a055.jpg)

![9f603da1099d8e1d88d0036eb8796653991aee89931f177ff93301e2779b74bd.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/9f603da1099d8e1d88d0036eb8796653991aee89931f177ff93301e2779b74bd.jpg)

![a9d3699e8358b2d5ea937fe27fcb880decd3ccee9484936dcc0b04a8cc2f6643.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/a9d3699e8358b2d5ea937fe27fcb880decd3ccee9484936dcc0b04a8cc2f6643.jpg)

![b9d1ee8997b6e933b8cf2e36ff1f81b4fdc9e7939e361b4ca5dbe9a62390753e.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/b9d1ee8997b6e933b8cf2e36ff1f81b4fdc9e7939e361b4ca5dbe9a62390753e.jpg)

![c4154e85db646316157876ed8523232cfbbd68c7f2604e144c022d8788c864be.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/c4154e85db646316157876ed8523232cfbbd68c7f2604e144c022d8788c864be.jpg)

![c7882bdc0e56019f81b6120a913a8f85c69407ba6a37667f2708e7bcf0d0a584.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/c7882bdc0e56019f81b6120a913a8f85c69407ba6a37667f2708e7bcf0d0a584.jpg)

![dcb3b92fa54fe31b61628e9de500d2e8565e3ff4e8e41f3a7381592ab35c5908.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/dcb3b92fa54fe31b61628e9de500d2e8565e3ff4e8e41f3a7381592ab35c5908.jpg)

![ea1d3761ba27052342c87b7af74e6c9b09ca8632388cac808f8578816bb94f9f.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/ea1d3761ba27052342c87b7af74e6c9b09ca8632388cac808f8578816bb94f9f.jpg)

![f50ebaa731acfb5742380fcb24848973dc69f3135cb9ff8ea0c58a488d987a20.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/f50ebaa731acfb5742380fcb24848973dc69f3135cb9ff8ea0c58a488d987a20.jpg)

![fd8432791e791b007272e4ccbc65cc5fcaeaa95fc307b1972892c9d40026c3fe.jpg](../icml_results/1028_Learning-Guided%20Integration%20Contours%20Construction%20for%20Fast%20Large-Scale%20Generalized%20Eigensolvers/tables/fd8432791e791b007272e4ccbc65cc5fcaeaa95fc307b1972892c9d40026c3fe.jpg)

<a id="paper-1011"></a>

## MALICE: Memory-aware Loop Invariants Generation on Symbolic Execution Traces


### Images

![0b8cc4e94ffc789649a13181a30a5b50729c2e93e80b4894368253c26f95bb38.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/images/0b8cc4e94ffc789649a13181a30a5b50729c2e93e80b4894368253c26f95bb38.jpg)

![11e65d76075afbfff0b285922b620c11e46ebb46f087976b9b97bba97fbe4185.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/images/11e65d76075afbfff0b285922b620c11e46ebb46f087976b9b97bba97fbe4185.jpg)

![470b604695ca309d3b863a077cf60406062618b99b73fa9214359a0c85982894.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/images/470b604695ca309d3b863a077cf60406062618b99b73fa9214359a0c85982894.jpg)

### Tables

![29f1db583abe8b134daae9c9789629b7a9de9e16cc03ec5de83ae413efa02d28.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/29f1db583abe8b134daae9c9789629b7a9de9e16cc03ec5de83ae413efa02d28.jpg)

![3712d94b1fb9f510cb3586c31823a4f2a22c991bed3aa0192dc1d2f545d2503c.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/3712d94b1fb9f510cb3586c31823a4f2a22c991bed3aa0192dc1d2f545d2503c.jpg)

![4c3d75e2443ead4ddd56cdf99dd33ea11f9293fc0fed13d646100b2d1df687c0.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/4c3d75e2443ead4ddd56cdf99dd33ea11f9293fc0fed13d646100b2d1df687c0.jpg)

![50e6ddd9e2040f18367db6940c94f5f7fbfc188e18a320eb8b0586d587c9b62f.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/50e6ddd9e2040f18367db6940c94f5f7fbfc188e18a320eb8b0586d587c9b62f.jpg)

![5272cb93723f6b89f0877b12b9d65bb76706a13a3fb4f81f5b6dd4dc88be8727.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/5272cb93723f6b89f0877b12b9d65bb76706a13a3fb4f81f5b6dd4dc88be8727.jpg)

![99d4ce5d23174509e3aa4249be234b6253f12f5f6c98056e7fcd3db5f3ac2afc.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/99d4ce5d23174509e3aa4249be234b6253f12f5f6c98056e7fcd3db5f3ac2afc.jpg)

![db7b6e33c299108053638cad12e0d36caabb0d38fc4c270294acd25687ee9b51.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/db7b6e33c299108053638cad12e0d36caabb0d38fc4c270294acd25687ee9b51.jpg)

![fb4bf2b6681f16741ef1dbdb0aa7db3bb65c5ecadbc3c6d876b10b93ab42210e.jpg](../icml_results/1029_MALICE_%20Memory-aware%20Loop%20Invariants%20Generation%20on%20Symbolic%20Execution%20Traces/tables/fb4bf2b6681f16741ef1dbdb0aa7db3bb65c5ecadbc3c6d876b10b93ab42210e.jpg)

<a id="paper-1012"></a>

## Focusing: View-Consistent Sparse Voxels for Efficient 3D VAE Training


### Images

![10cafb78fdda7fa9da034451e8f0caadfc48a32dd3b0ac9478b88372e76e337a.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/10cafb78fdda7fa9da034451e8f0caadfc48a32dd3b0ac9478b88372e76e337a.jpg)

![11abb5c380e389fe8ed211d125cbac93f53294880165a20b3f2e918f99fdc986.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/11abb5c380e389fe8ed211d125cbac93f53294880165a20b3f2e918f99fdc986.jpg)

![2bb6037cff2d6d188210ab2d5f2d7d9cbd42aae9ca400910e185bdc3f995ddbe.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/2bb6037cff2d6d188210ab2d5f2d7d9cbd42aae9ca400910e185bdc3f995ddbe.jpg)

![44b9d4d59af30661bc4b5c8aae834ae7ba7c0c662d96d5ef2ba4ab19f3051537.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/44b9d4d59af30661bc4b5c8aae834ae7ba7c0c662d96d5ef2ba4ab19f3051537.jpg)

![5cf1611b6238b50b45ad87bf1e31ed9d9b1bf090b053ab5cc6aa3645235fce4f.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/5cf1611b6238b50b45ad87bf1e31ed9d9b1bf090b053ab5cc6aa3645235fce4f.jpg)

![6866354434c4cbaecd556621f8f484da73ea5a110fc1025695d2e2171f9380cf.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/6866354434c4cbaecd556621f8f484da73ea5a110fc1025695d2e2171f9380cf.jpg)

![69a94b07b263e807f7175393ab22e4c761ace287dc23165b2e422e9756624ee8.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/69a94b07b263e807f7175393ab22e4c761ace287dc23165b2e422e9756624ee8.jpg)

![8ca040096f609dbf10ba79479e5991c109fd1fb7f8b29c9c48b1f8ddd4e1e668.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/8ca040096f609dbf10ba79479e5991c109fd1fb7f8b29c9c48b1f8ddd4e1e668.jpg)

![8d2ed6b611eb76984fa96989b7444e0beebcdd514cc76affb80904fd49a514fb.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/8d2ed6b611eb76984fa96989b7444e0beebcdd514cc76affb80904fd49a514fb.jpg)

![df810a966367cdb999a5d53fbb44d4a8e8d0e911509ce1ed8955492e504553a5.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/df810a966367cdb999a5d53fbb44d4a8e8d0e911509ce1ed8955492e504553a5.jpg)

![ec2b80f585d9a8939677ba18d143c7614a4dc2dadd8c4619b9230579dcc75ea7.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/images/ec2b80f585d9a8939677ba18d143c7614a4dc2dadd8c4619b9230579dcc75ea7.jpg)

### Tables

![09c0afd8740bdcde0a3baf7be73dac89f0c614a03859db89e81fe5198f91f522.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/tables/09c0afd8740bdcde0a3baf7be73dac89f0c614a03859db89e81fe5198f91f522.jpg)

![19be2e6df2d5e5095f10b5b6b7b3e2dec3c671ea25f753cf3e2d9190ab27c6f0.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/tables/19be2e6df2d5e5095f10b5b6b7b3e2dec3c671ea25f753cf3e2d9190ab27c6f0.jpg)

![27a348c1c6c7ec98d16952222579d8c909a7480f7b520e408366d31f6da7c88a.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/tables/27a348c1c6c7ec98d16952222579d8c909a7480f7b520e408366d31f6da7c88a.jpg)

![2941a7799ff9bc9a86a2835169f473798ecddb6ecceb79f1e54bfc4ede069868.jpg](../icml_results/1030_Focusing_%20View-Consistent%20Sparse%20Voxels%20for%20Efficient%203D%20VAE%20Training/tables/2941a7799ff9bc9a86a2835169f473798ecddb6ecceb79f1e54bfc4ede069868.jpg)

<a id="paper-1013"></a>

## Rethinking Instruction Drift as a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning


### Images

![1ce28cb30bb059498b7dacaeee8966500f95662c01636d027cef619fc2b45379.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/1ce28cb30bb059498b7dacaeee8966500f95662c01636d027cef619fc2b45379.jpg)

![72350d4da8afc9f0ec7fd60796bcf01315074452edb928d9042083408ae09f9b.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/72350d4da8afc9f0ec7fd60796bcf01315074452edb928d9042083408ae09f9b.jpg)

![7982b9d25f973546fbf31548abdae9aec86516ac0d0fff5aea00e34e00e0a73c.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/7982b9d25f973546fbf31548abdae9aec86516ac0d0fff5aea00e34e00e0a73c.jpg)

![caf13746ba73b2fbbab5a035452178345e7c598e9bfd0839dc6e47ba8b867be7.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/caf13746ba73b2fbbab5a035452178345e7c598e9bfd0839dc6e47ba8b867be7.jpg)

![dca42104b593a1074ad35d8e4cb69410a7daac3b2d9cad91e8309e5f76180b72.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/dca42104b593a1074ad35d8e4cb69410a7daac3b2d9cad91e8309e5f76180b72.jpg)

![dd9f886d90eb362e40428e7fa72b427eb16d275e475759f531c44cf2ebfa228c.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/dd9f886d90eb362e40428e7fa72b427eb16d275e475759f531c44cf2ebfa228c.jpg)

![e8c3c1cb512341bfcab227b560d8743aad24fd2e9e74abd208f5cf7e73907188.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/images/e8c3c1cb512341bfcab227b560d8743aad24fd2e9e74abd208f5cf7e73907188.jpg)

### Tables

![231d1e0d2f001045b87921bc5e57092c4c3502208c67c406ac2dc90c0e551c63.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/231d1e0d2f001045b87921bc5e57092c4c3502208c67c406ac2dc90c0e551c63.jpg)

![65d30fee666cfa895ffe3af923aed1c70fd79003590c78fec8b0ee566b460874.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/65d30fee666cfa895ffe3af923aed1c70fd79003590c78fec8b0ee566b460874.jpg)

![65d90f6349598a70b310022befd2bd7171403d76e102e7065a906c64931c3fe2.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/65d90f6349598a70b310022befd2bd7171403d76e102e7065a906c64931c3fe2.jpg)

![6a3d6c40bb72d0a3beadcfd98cd15616f67f5033bff3209a6d66bc05d44b3ec6.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/6a3d6c40bb72d0a3beadcfd98cd15616f67f5033bff3209a6d66bc05d44b3ec6.jpg)

![8e370c574b03c08a8a574fd5dcc9aa240c8637ac7070085fc0eabe9e368a47eb.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/8e370c574b03c08a8a574fd5dcc9aa240c8637ac7070085fc0eabe9e368a47eb.jpg)

![ac3df89d520fa960f3e7ac7d33373684a8e12308b21015e544e463a08c42fbfb.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/ac3df89d520fa960f3e7ac7d33373684a8e12308b21015e544e463a08c42fbfb.jpg)

![ac71b409b7017560e2284e98028bcaf3b1f2ede4e8d47d6639b107006f77e44c.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/ac71b409b7017560e2284e98028bcaf3b1f2ede4e8d47d6639b107006f77e44c.jpg)

![bf6023ed6ce4a3b6aad1a23f260313089cf94e9959e2a7472b9dcc2784611bf0.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/bf6023ed6ce4a3b6aad1a23f260313089cf94e9959e2a7472b9dcc2784611bf0.jpg)

![d5f55cc33f575620f29f17ddb443c4ddd33993537583971a73733cfa130c0f6d.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/d5f55cc33f575620f29f17ddb443c4ddd33993537583971a73733cfa130c0f6d.jpg)

![dbcf7cbc3827a844cb826cc8494a5892c05e3e3852179379a9637e5f6dfb8185.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/dbcf7cbc3827a844cb826cc8494a5892c05e3e3852179379a9637e5f6dfb8185.jpg)

![ff49343f6c0640d1307fc0e9b7f037dc3093a7d4ab46d5ef8cc4bada33bd7ed7.jpg](../icml_results/1031_Rethinking%20Instruction%20Drift%20as%20a%20Sampling%20Error_%20SNR-Aware%20Power%20Distributions%20for%20Long-Horizon%20Rob/tables/ff49343f6c0640d1307fc0e9b7f037dc3093a7d4ab46d5ef8cc4bada33bd7ed7.jpg)

<a id="paper-1014"></a>

## Personalized Additive Modeling for Multi-level Federated Learning


### Images

![34e2fe099bd86fef295e78176eadcb49a3adb325426aba94c184197455614cb4.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/34e2fe099bd86fef295e78176eadcb49a3adb325426aba94c184197455614cb4.jpg)

![3613a62c9b8b589f575757e201d8c1c04200077a6539c377c2f96a981c0b2915.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/3613a62c9b8b589f575757e201d8c1c04200077a6539c377c2f96a981c0b2915.jpg)

![3ff16714244adbb8b2851b530c1ce4125a584a2880109f88bb77526979474c4d.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/3ff16714244adbb8b2851b530c1ce4125a584a2880109f88bb77526979474c4d.jpg)

![4ac59b16af7f165dfea794d15e13287c90e004b44e64d40c0b47a340e23c9cec.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/4ac59b16af7f165dfea794d15e13287c90e004b44e64d40c0b47a340e23c9cec.jpg)

![4ca5b22b7521f387f25164f63757f746e31a0714c2c9a686f19c79d0994a6e11.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/4ca5b22b7521f387f25164f63757f746e31a0714c2c9a686f19c79d0994a6e11.jpg)

![4ee50f20d430a29ae5441ef9c9d18094d1eca8f7ca6a71bf1f6294b6361fba7c.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/4ee50f20d430a29ae5441ef9c9d18094d1eca8f7ca6a71bf1f6294b6361fba7c.jpg)

![5e7cd8c6ff0e2cab3b268edb68e5f13935461d948b7887f0ca4da59521dbca07.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/5e7cd8c6ff0e2cab3b268edb68e5f13935461d948b7887f0ca4da59521dbca07.jpg)

![6696a38eadf0269926992803da22bf9d402ddc1117e266afb278989491e91539.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/6696a38eadf0269926992803da22bf9d402ddc1117e266afb278989491e91539.jpg)

![7cfd7caece46202b79da2cb6afa3c723acf8b71458b66411ee8d271994ca7820.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/7cfd7caece46202b79da2cb6afa3c723acf8b71458b66411ee8d271994ca7820.jpg)

![b1073317d3c0132c34e6d919a33528769a60b118464e306e9bf8acb8d480183e.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/b1073317d3c0132c34e6d919a33528769a60b118464e306e9bf8acb8d480183e.jpg)

![d8d3b32c22c85c81c1faaff191a0b0f25d17daf07b33441ab796b961df244469.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/d8d3b32c22c85c81c1faaff191a0b0f25d17daf07b33441ab796b961df244469.jpg)

![fb7450e23a2d6866400d3eaab7ee8be89412826b7fc23aa25733fadc6830eb58.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/images/fb7450e23a2d6866400d3eaab7ee8be89412826b7fc23aa25733fadc6830eb58.jpg)

### Tables

![114b2db08f4963e0ccbfd00d33a304f8351dd0d3176421c2b831394ae80fc011.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/114b2db08f4963e0ccbfd00d33a304f8351dd0d3176421c2b831394ae80fc011.jpg)

![24f04f2343dffd3bf65bd38b22554f4fd934cc7310102ea06b295fe9ae6bffad.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/24f04f2343dffd3bf65bd38b22554f4fd934cc7310102ea06b295fe9ae6bffad.jpg)

![6b037ed0823788806fa1bbda91a5bef83a69d6dcecc7929a01657fba6802405c.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/6b037ed0823788806fa1bbda91a5bef83a69d6dcecc7929a01657fba6802405c.jpg)

![6b1ddbc79154c0acd7f21612fca31cd8bb6c588a6fd8f1f9a01ba482f6e4f97a.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/6b1ddbc79154c0acd7f21612fca31cd8bb6c588a6fd8f1f9a01ba482f6e4f97a.jpg)

![9fcf1d3a7cb4612dfee46f295b4e71cc4cdb967c3adec56494c14ce8c14ee749.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/9fcf1d3a7cb4612dfee46f295b4e71cc4cdb967c3adec56494c14ce8c14ee749.jpg)

![ca019eaa8ac93eb2dc1c606284a4aae122334d42ca869aab7e5e74390407884d.jpg](../icml_results/1032_Personalized%20Additive%20Modeling%20for%20Multi-level%20Federated%20Learning/tables/ca019eaa8ac93eb2dc1c606284a4aae122334d42ca869aab7e5e74390407884d.jpg)

<a id="paper-1015"></a>

## Efficient Inference for Noisy LLM-as-a-Judge Evaluation


### Images

![220c76b50cc8e1f5ab911f93ac46ca58caad77556fd90a9937482abe1f2f90f4.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/220c76b50cc8e1f5ab911f93ac46ca58caad77556fd90a9937482abe1f2f90f4.jpg)

![50a1e92c313fa22abf3a2c4c16721e36770576c04f94979dd6e170bf2715b99c.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/50a1e92c313fa22abf3a2c4c16721e36770576c04f94979dd6e170bf2715b99c.jpg)

![76688df691d2b73ba811f29872bc31be5e19e13e63c988c079c3490b032686bf.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/76688df691d2b73ba811f29872bc31be5e19e13e63c988c079c3490b032686bf.jpg)

![81a70a32a3d33a48f53739f61bb320c5e4e67b6a046bfc156ae3af754e22bb46.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/81a70a32a3d33a48f53739f61bb320c5e4e67b6a046bfc156ae3af754e22bb46.jpg)

![a867dbefc561c0909b95c7e11791d587355068925fff4964a8f9784c1b84412e.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/a867dbefc561c0909b95c7e11791d587355068925fff4964a8f9784c1b84412e.jpg)

![c91d684c00fdd98710672acc25ce400dd5ba40dbe3fce22e7630f694d0efe567.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/c91d684c00fdd98710672acc25ce400dd5ba40dbe3fce22e7630f694d0efe567.jpg)

![cc47144d2c2c5282d722c205d28205824f51c105210828450ebee2c7be86dc78.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/cc47144d2c2c5282d722c205d28205824f51c105210828450ebee2c7be86dc78.jpg)

![e243a225b8ed74c64acbf27c1c03e2ec76d870f4ecc11f1316afebed02bb34a2.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/images/e243a225b8ed74c64acbf27c1c03e2ec76d870f4ecc11f1316afebed02bb34a2.jpg)

### Tables

![1abd0814be6eb64b2fbffc513e137f011f597d3580752ed11d7a37df156e571a.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/1abd0814be6eb64b2fbffc513e137f011f597d3580752ed11d7a37df156e571a.jpg)

![1c288c0078b78ea326112757240228d59219fff71a612b5b615e05254d1abfa5.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/1c288c0078b78ea326112757240228d59219fff71a612b5b615e05254d1abfa5.jpg)

![27c55d9ce4e331520d432b40ee7a541c4b5ffdf7e9d28dcb8d14ce684065daad.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/27c55d9ce4e331520d432b40ee7a541c4b5ffdf7e9d28dcb8d14ce684065daad.jpg)

![71abd799d112b97d5856f5d177157fb26654a838273941caab83064ba809d89a.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/71abd799d112b97d5856f5d177157fb26654a838273941caab83064ba809d89a.jpg)

![835a06c8f09cd23c0b287231a9e20930824b19754ba82979a4c8185e09766eb2.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/835a06c8f09cd23c0b287231a9e20930824b19754ba82979a4c8185e09766eb2.jpg)

![d9bc688130deb4b97546fabba7748cceee00124b35c081773c79016a386c476c.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/d9bc688130deb4b97546fabba7748cceee00124b35c081773c79016a386c476c.jpg)

![f518e164cbb4d71a4c58f397ff0fd683634472a285a28ed5859adb858a9d04db.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/f518e164cbb4d71a4c58f397ff0fd683634472a285a28ed5859adb858a9d04db.jpg)

![fbc642487828ccb2c93f299fa190c70d423a6686dbf9660a39890d0d9526adfd.jpg](../icml_results/1033_Efficient%20Inference%20for%20Noisy%20LLM-as-a-Judge%20Evaluation/tables/fbc642487828ccb2c93f299fa190c70d423a6686dbf9660a39890d0d9526adfd.jpg)

<a id="paper-1016"></a>

## DANCE: Dynamic, Available, Neighbor-gated Condensation for Federated Text-Attributed Graphs


### Images

![0c61e021ffad1624f97de678d8dfb65f74813702966f934261ec32787fe50e10.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/0c61e021ffad1624f97de678d8dfb65f74813702966f934261ec32787fe50e10.jpg)

![0d509a6b7bae61412a836609bf36a4da0f9c3e63274b69bbcad56079ce79018e.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/0d509a6b7bae61412a836609bf36a4da0f9c3e63274b69bbcad56079ce79018e.jpg)

![11fd92a8506eb3daff52e04e952293fad7478419d29384eeddadc17d1e2d7b85.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/11fd92a8506eb3daff52e04e952293fad7478419d29384eeddadc17d1e2d7b85.jpg)

![43dfe84f4dcbfa402eeaa356329ef43bc8f011c1b951530812a5895820a671d6.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/43dfe84f4dcbfa402eeaa356329ef43bc8f011c1b951530812a5895820a671d6.jpg)

![75babee7f43487fc6d1822d88ae1859ba3419d0d085421af0543c5cb5fd799b9.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/75babee7f43487fc6d1822d88ae1859ba3419d0d085421af0543c5cb5fd799b9.jpg)

![c5475eb0fe9794cf5613e0097620a794264d6d756800192be5f8063c367ecf9a.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/c5475eb0fe9794cf5613e0097620a794264d6d756800192be5f8063c367ecf9a.jpg)

![d0af4fc454119ebacbdde7c8f96fd26f5fb667d2c936c71d41786daafa701b04.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/d0af4fc454119ebacbdde7c8f96fd26f5fb667d2c936c71d41786daafa701b04.jpg)

![edcbab3b19ccc809a819ec2b6e2d8f3c2ba3cff0de7ac7076227f99f3f79dfd5.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/edcbab3b19ccc809a819ec2b6e2d8f3c2ba3cff0de7ac7076227f99f3f79dfd5.jpg)

![f739de1a369cde0b165f54a8c8c79e8758775034dbd5e027890c61a8e2694ca4.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/images/f739de1a369cde0b165f54a8c8c79e8758775034dbd5e027890c61a8e2694ca4.jpg)

### Tables

![19b0028c0368a3ca359eed976ee83533d7fa4da4c778a983a2f22b9c0ea66251.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/19b0028c0368a3ca359eed976ee83533d7fa4da4c778a983a2f22b9c0ea66251.jpg)

![226f568a25f7a689bac45e5e66e6da0ffe8dd59856d52f30bc632de33b95327f.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/226f568a25f7a689bac45e5e66e6da0ffe8dd59856d52f30bc632de33b95327f.jpg)

![5c2622061bc4f18f2357ee7776bb508fda61089e9d5de1884a72526d11ade398.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/5c2622061bc4f18f2357ee7776bb508fda61089e9d5de1884a72526d11ade398.jpg)

![6a1abfbdd9b885cd02a58112e7e3b44e88a024c0703d5a49dbb5040239d6485a.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/6a1abfbdd9b885cd02a58112e7e3b44e88a024c0703d5a49dbb5040239d6485a.jpg)

![86a119c021a0466270063a767fe8c17450e38c38d0f67d4250033ce8b7d0e8f9.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/86a119c021a0466270063a767fe8c17450e38c38d0f67d4250033ce8b7d0e8f9.jpg)

![92324d0117dd50815bbd32ab2eb2ce08b8b8203471de63e8ce7bc59a101029c6.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/92324d0117dd50815bbd32ab2eb2ce08b8b8203471de63e8ce7bc59a101029c6.jpg)

![a53c1d2bd296fd48db116c5de5552de9718803abe440e4bd9b2bc7a6df923ba1.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/a53c1d2bd296fd48db116c5de5552de9718803abe440e4bd9b2bc7a6df923ba1.jpg)

![c062023feabf211bf1c03fb063bf9d9b38c429845140c142d75e7631bb86ea69.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/c062023feabf211bf1c03fb063bf9d9b38c429845140c142d75e7631bb86ea69.jpg)

![c844072bfe014237b44bef9a44659b513d8671d44355b71708c9297e00e663a0.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/c844072bfe014237b44bef9a44659b513d8671d44355b71708c9297e00e663a0.jpg)

![cb28fffbcfea7a1ebde5576e0870ba02e2000086a371b85142fc0393eedaac9e.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/cb28fffbcfea7a1ebde5576e0870ba02e2000086a371b85142fc0393eedaac9e.jpg)

![e2e5f9ce501697f3268bd50ce902730556a75f4c462b7f8a0f33772f1b846cca.jpg](../icml_results/1034_DANCE_%20Dynamic%2C%20Available%2C%20Neighbor-gated%20Condensation%20for%20Federated%20Text-Attributed%20Graphs/tables/e2e5f9ce501697f3268bd50ce902730556a75f4c462b7f8a0f33772f1b846cca.jpg)

<a id="paper-1017"></a>

## Towards Theoretical Understanding of Transformer Test-Time Computing: Investigation on In-Context Linear Regression


### Images

![0cff9025cb27a0d717027a01e846240cecbe073c9309d67c7bbc01066702b4d7.jpg](../icml_results/1035_Towards%20Theoretical%20Understanding%20of%20Transformer%20Test-Time%20Computing_%20Investigation%20on%20In-Context%20Li/images/0cff9025cb27a0d717027a01e846240cecbe073c9309d67c7bbc01066702b4d7.jpg)

![2ef74f50a23cc74bee9fa25aa4fee3c2707eaedd48e9c2f97074988fa2a80b27.jpg](../icml_results/1035_Towards%20Theoretical%20Understanding%20of%20Transformer%20Test-Time%20Computing_%20Investigation%20on%20In-Context%20Li/images/2ef74f50a23cc74bee9fa25aa4fee3c2707eaedd48e9c2f97074988fa2a80b27.jpg)

![8e696ccdbe32b0646e2158f9ccf9708e9976c88f2a31d3222359fbb6a24b50f1.jpg](../icml_results/1035_Towards%20Theoretical%20Understanding%20of%20Transformer%20Test-Time%20Computing_%20Investigation%20on%20In-Context%20Li/images/8e696ccdbe32b0646e2158f9ccf9708e9976c88f2a31d3222359fbb6a24b50f1.jpg)

### Tables

![a78bbc95938fd383d7f748a48f5739189bdf57f86dd1cf6ca71ac9a3149a0ac1.jpg](../icml_results/1035_Towards%20Theoretical%20Understanding%20of%20Transformer%20Test-Time%20Computing_%20Investigation%20on%20In-Context%20Li/tables/a78bbc95938fd383d7f748a48f5739189bdf57f86dd1cf6ca71ac9a3149a0ac1.jpg)

<a id="paper-1018"></a>

## SLAE: Strictly Local All-atom Environment for Protein Representation


### Images

![0328c9dd4517e0cf0bc2a8e4b79d25a7c2c078b63a51158302a7ba78d9e0b6b1.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/0328c9dd4517e0cf0bc2a8e4b79d25a7c2c078b63a51158302a7ba78d9e0b6b1.jpg)

![2637a53b5bed59152b3c368eea711ea8561b04aba1ec971ecedcfc0a2fbd8d0d.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/2637a53b5bed59152b3c368eea711ea8561b04aba1ec971ecedcfc0a2fbd8d0d.jpg)

![26a23e23b15c74d2c1dcb2c5e410038031531da60a11dcce679b158b4db339ef.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/26a23e23b15c74d2c1dcb2c5e410038031531da60a11dcce679b158b4db339ef.jpg)

![2da29a9245495cab43adef59ed15390b6d5d3840d8e3857c98f8dfd8178bf392.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/2da29a9245495cab43adef59ed15390b6d5d3840d8e3857c98f8dfd8178bf392.jpg)

![4c2cb733ae29d92f40974dbe8b2f518a5e9d2334d26ef994e30d6894e4e69bdd.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/4c2cb733ae29d92f40974dbe8b2f518a5e9d2334d26ef994e30d6894e4e69bdd.jpg)

![572a3336c2a9d66fb2f543d9bd693830ac801555ecab50cb416c971b090e3f51.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/572a3336c2a9d66fb2f543d9bd693830ac801555ecab50cb416c971b090e3f51.jpg)

![8238c8dfe12157464ece5ac7098d17a5572310cba38b8a0d86d9d590df2b9512.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/8238c8dfe12157464ece5ac7098d17a5572310cba38b8a0d86d9d590df2b9512.jpg)

![999d2e8fc642bbbd261d5d78f5366ff841fb7f9eab42f7ae8f55c9be76a41376.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/999d2e8fc642bbbd261d5d78f5366ff841fb7f9eab42f7ae8f55c9be76a41376.jpg)

![a7e38b5dcf2490a96cf6b3f5f2fec95b67791345c59b642f5cea8279ca78030a.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/a7e38b5dcf2490a96cf6b3f5f2fec95b67791345c59b642f5cea8279ca78030a.jpg)

![b80c891d1d4a34165cbbf6dc204fcfc0c6b960db18023d9da1483dc13aaf1d58.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/b80c891d1d4a34165cbbf6dc204fcfc0c6b960db18023d9da1483dc13aaf1d58.jpg)

![c0fc3a8aec0679857147887b844c78daf925e69019f1ba1c43cb388d99f53a9b.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/c0fc3a8aec0679857147887b844c78daf925e69019f1ba1c43cb388d99f53a9b.jpg)

![e710e6f08775b0beef106bda5488ba1979340d208664f3b9890f1a05cb6e3e2f.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/e710e6f08775b0beef106bda5488ba1979340d208664f3b9890f1a05cb6e3e2f.jpg)

![e8f5eaf29f55e371bc239ea09334c28a243d712c0076a76c9aaffd4e8d018b8e.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/images/e8f5eaf29f55e371bc239ea09334c28a243d712c0076a76c9aaffd4e8d018b8e.jpg)

### Tables

![02a4b2f514da71681ee2f687173422e200bde3fff0e8bf830991d40a6a8ea942.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/02a4b2f514da71681ee2f687173422e200bde3fff0e8bf830991d40a6a8ea942.jpg)

![176e2fc23983eb1117148f657d4f9251ac4c2a43c991904348f0aff8ec230b82.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/176e2fc23983eb1117148f657d4f9251ac4c2a43c991904348f0aff8ec230b82.jpg)

![19d362edc88025614571281d2420a692fd101a228b86ec85df07187446967b6b.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/19d362edc88025614571281d2420a692fd101a228b86ec85df07187446967b6b.jpg)

![3f2c0f43361258bef13a834bc8cfb5f9f7478e4ef4e9a566b26d3eca5b39a989.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/3f2c0f43361258bef13a834bc8cfb5f9f7478e4ef4e9a566b26d3eca5b39a989.jpg)

![52a637b8fe17e4ba1e267fffa60a3c67b66b4f70cdf3220a14f684e6c8257839.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/52a637b8fe17e4ba1e267fffa60a3c67b66b4f70cdf3220a14f684e6c8257839.jpg)

![54a79c267ff45d4a1b08ee0291ad40c633c9399c1d62860e1d6926e3f486d3cb.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/54a79c267ff45d4a1b08ee0291ad40c633c9399c1d62860e1d6926e3f486d3cb.jpg)

![5b63d9f42f59358ae7d11fd9d2d31ba56a3b0a215e646442c295432b595ad7aa.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/5b63d9f42f59358ae7d11fd9d2d31ba56a3b0a215e646442c295432b595ad7aa.jpg)

![9d6918e4e35a9a041426a6a978d0ffb0bc22a0d3accf421fb053a939810c2ad4.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/9d6918e4e35a9a041426a6a978d0ffb0bc22a0d3accf421fb053a939810c2ad4.jpg)

![a83f18f5fdf9152567bf2b811a85943a1d5f151b8c91b4962990b8880cf2d310.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/a83f18f5fdf9152567bf2b811a85943a1d5f151b8c91b4962990b8880cf2d310.jpg)

![d6814185c0a272f5edc54804a957097d26c8ba25052e43bdd55bd722e82ae388.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/d6814185c0a272f5edc54804a957097d26c8ba25052e43bdd55bd722e82ae388.jpg)

![db5ec2561bc70386981b912b3f5671a40ee8f3dfc18ae275bc96a6c162edf4eb.jpg](../icml_results/1036_SLAE_%20Strictly%20Local%20All-atom%20Environment%20for%20Protein%20Representation/tables/db5ec2561bc70386981b912b3f5671a40ee8f3dfc18ae275bc96a6c162edf4eb.jpg)

<a id="paper-1019"></a>

## Co-Generative De Novo Functional Protein Design


### Images

![0bac67118d6cff916a94cffe1845d27017830088dff29816e8d2d78d4909c28b.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/0bac67118d6cff916a94cffe1845d27017830088dff29816e8d2d78d4909c28b.jpg)

![1d4abf93cd8474b9e5a7cede26c3d06c4c18e0a89eaea8741872350869aa2b75.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/1d4abf93cd8474b9e5a7cede26c3d06c4c18e0a89eaea8741872350869aa2b75.jpg)

![3f44301b007660831db2f4e7f647224fa83ebcf15ecead9f95916ae158e70fca.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/3f44301b007660831db2f4e7f647224fa83ebcf15ecead9f95916ae158e70fca.jpg)

![5b90b0741d80f29c92276b5980a8eb7676f53adf884179212090755c844a14e3.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/5b90b0741d80f29c92276b5980a8eb7676f53adf884179212090755c844a14e3.jpg)

![7d753f11c807800a88d60c97384f87f6fec7e22e4976d72107f0ab20218ad197.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/7d753f11c807800a88d60c97384f87f6fec7e22e4976d72107f0ab20218ad197.jpg)

![9d8740d973eb788b68d123cdfd4950d2030c62b7b345e0078610adfd86f7995f.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/9d8740d973eb788b68d123cdfd4950d2030c62b7b345e0078610adfd86f7995f.jpg)

![bbd005dbb6bbb1f3bddab57d3fef57cc193faeadbc2c880b5e8b5cd920aa04fa.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/bbd005dbb6bbb1f3bddab57d3fef57cc193faeadbc2c880b5e8b5cd920aa04fa.jpg)

![c78213d6e70de46165c9bafcf9abb4bfd7bf0c8c114753959cf33df5ca09f39f.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/c78213d6e70de46165c9bafcf9abb4bfd7bf0c8c114753959cf33df5ca09f39f.jpg)

![ddb0d2852ad91db79286cf446c295b4c3282f36480df21a38b29f661b358e86d.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/ddb0d2852ad91db79286cf446c295b4c3282f36480df21a38b29f661b358e86d.jpg)

![e5ce525ecb410e872830b50bc81519801e2a6c530a0b2b8b85ae3619d7639dba.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/e5ce525ecb410e872830b50bc81519801e2a6c530a0b2b8b85ae3619d7639dba.jpg)

![f089412382588aeda0980a7f4f49a4b4d1c0a55bc5aef323a571ba2fd6cb98a6.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/images/f089412382588aeda0980a7f4f49a4b4d1c0a55bc5aef323a571ba2fd6cb98a6.jpg)

### Tables

![03c423d2821e1bab9b4502919477e5ef82abcbde2fb19cff4d3a26f425b9b0ec.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/03c423d2821e1bab9b4502919477e5ef82abcbde2fb19cff4d3a26f425b9b0ec.jpg)

![18770aff95432da9ccf47b144fa07444de1bd263e5acf0920ebbfb82be704821.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/18770aff95432da9ccf47b144fa07444de1bd263e5acf0920ebbfb82be704821.jpg)

![1dd1232faf48dc6921db95dcfcc1de0d7d19aad8fd4f221dd0979dc25240c638.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/1dd1232faf48dc6921db95dcfcc1de0d7d19aad8fd4f221dd0979dc25240c638.jpg)

![32c40bbab3f4c3017895362b2d772004c6040da52cfb3e43ec55de2017781d78.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/32c40bbab3f4c3017895362b2d772004c6040da52cfb3e43ec55de2017781d78.jpg)

![37803d7aac926da05c024b073ce577483135b287fae4afe18e2735e3811aae2d.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/37803d7aac926da05c024b073ce577483135b287fae4afe18e2735e3811aae2d.jpg)

![38532e0fa94ddb9b919d905ffd55d16517bfb4da3a5935f4873bae2ab61d0a49.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/38532e0fa94ddb9b919d905ffd55d16517bfb4da3a5935f4873bae2ab61d0a49.jpg)

![695be1e3db7de04f28314e8baa0e055e723448fc9d2aeb896c87a5969b3ae8ec.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/695be1e3db7de04f28314e8baa0e055e723448fc9d2aeb896c87a5969b3ae8ec.jpg)

![7ad18aab137ab31c9ebfbc68ecd764a9f1ef43565489eef8c9cccf0f74aef9f7.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/7ad18aab137ab31c9ebfbc68ecd764a9f1ef43565489eef8c9cccf0f74aef9f7.jpg)

![bc4e108c1588f9cd4b67d20c448308e7852d22f783fd846e89aea3a377e47ae3.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/bc4e108c1588f9cd4b67d20c448308e7852d22f783fd846e89aea3a377e47ae3.jpg)

![c09f2f7a5714cfb8572e5fe95e7dfea20fe2aa241556f8ba29d31f21006f75c3.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/c09f2f7a5714cfb8572e5fe95e7dfea20fe2aa241556f8ba29d31f21006f75c3.jpg)

![d2be4b957b5edcd121b2c833f58d32743f38dd81a5be11ba462887f40491f18e.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/d2be4b957b5edcd121b2c833f58d32743f38dd81a5be11ba462887f40491f18e.jpg)

![db98b0ab1530a7c5a9dea043dba7aebf979282875b3fcb92df7bb1c45398f64e.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/db98b0ab1530a7c5a9dea043dba7aebf979282875b3fcb92df7bb1c45398f64e.jpg)

![fe3b265badbdd3136431747f7ebe8db6e820c84c6068447d512d4377ae2231d6.jpg](../icml_results/1037_Co-Generative%20De%20Novo%20Functional%20Protein%20Design/tables/fe3b265badbdd3136431747f7ebe8db6e820c84c6068447d512d4377ae2231d6.jpg)

<a id="paper-1020"></a>

## RADAR: Defending RAG Dynamically against Retrieval Corruption


### Images

![240f7de1429b5d52a78aeef76c6f6a58f35a1e6f23db61ef830a57bc8cb9c39f.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/images/240f7de1429b5d52a78aeef76c6f6a58f35a1e6f23db61ef830a57bc8cb9c39f.jpg)

![27a3548cd32d2051cdd43e57792e95345e29e688181461281832744e8d76be46.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/images/27a3548cd32d2051cdd43e57792e95345e29e688181461281832744e8d76be46.jpg)

![d1b00e7a92a6d2cd9ed7590d57f57dd7d9cfabb218d952bfd6c9ef50443a0f1a.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/images/d1b00e7a92a6d2cd9ed7590d57f57dd7d9cfabb218d952bfd6c9ef50443a0f1a.jpg)

### Tables

![1c3a1de7ca1b4c4e4a45a4b03cabc82c61f1b803d312a90b2c4cd05bde9932f5.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/1c3a1de7ca1b4c4e4a45a4b03cabc82c61f1b803d312a90b2c4cd05bde9932f5.jpg)

![258d74352b9275cc51eb7ad6bff4636ee05b2ce917d8b54f05238eef9cc86d64.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/258d74352b9275cc51eb7ad6bff4636ee05b2ce917d8b54f05238eef9cc86d64.jpg)

![300aa28a09c220ec9443198c0ca8424379695f369a741e6a0d011293d9a8ec1c.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/300aa28a09c220ec9443198c0ca8424379695f369a741e6a0d011293d9a8ec1c.jpg)

![30ec5a99f9de563b0be8a92f63c8e1b8fb5fe6d8f2ff672869e956f86d47a429.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/30ec5a99f9de563b0be8a92f63c8e1b8fb5fe6d8f2ff672869e956f86d47a429.jpg)

![4050affb17bc8009de258eeaab500861d93f4e179de5bba5ab72ded1d1dfbcb7.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/4050affb17bc8009de258eeaab500861d93f4e179de5bba5ab72ded1d1dfbcb7.jpg)

![41e287f81e01d6187fdc2358524ac5ca433b98e4e989d1774859644bc5e325bc.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/41e287f81e01d6187fdc2358524ac5ca433b98e4e989d1774859644bc5e325bc.jpg)

![4e597f43a7c299a5fa4dff3076db2a850ab2665401f7e5e16b2fc7e13bf52a92.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/4e597f43a7c299a5fa4dff3076db2a850ab2665401f7e5e16b2fc7e13bf52a92.jpg)

![68ae513a7b3c8b980f2168092fe110506dc43aa7e10aaeec912f221ef998c37a.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/68ae513a7b3c8b980f2168092fe110506dc43aa7e10aaeec912f221ef998c37a.jpg)

![6e84cbf10364a2d528352dfb5c45bfdd8aa8dfbfb4762c7f63760e5219802f34.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/6e84cbf10364a2d528352dfb5c45bfdd8aa8dfbfb4762c7f63760e5219802f34.jpg)

![6eadd0aa77aada64abd93ab39e7284331b2ee72a5467a144daaa0fbbc34f8cfa.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/6eadd0aa77aada64abd93ab39e7284331b2ee72a5467a144daaa0fbbc34f8cfa.jpg)

![95df31d66687cea6d05db177140a3728710f151120bebe8debdf19cb68f4c042.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/95df31d66687cea6d05db177140a3728710f151120bebe8debdf19cb68f4c042.jpg)

![b1950c33d8c76e2b3118e504bb5acdd95c3b97e992998911a200b77bbb33be47.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/b1950c33d8c76e2b3118e504bb5acdd95c3b97e992998911a200b77bbb33be47.jpg)

![bf9036d1190397d72a88f1a756d30f1e49e695ba83896a631d24b4add13b30c8.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/bf9036d1190397d72a88f1a756d30f1e49e695ba83896a631d24b4add13b30c8.jpg)

![c1470dabaf1de0138d2db15ce67bc0e38027b60a48cabdc7228d38b71286e220.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/c1470dabaf1de0138d2db15ce67bc0e38027b60a48cabdc7228d38b71286e220.jpg)

![cd3b52b2aa801408f1944e70b0ec4743aacf132591f7539b91be822f6de42559.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/cd3b52b2aa801408f1944e70b0ec4743aacf132591f7539b91be822f6de42559.jpg)

![d4cf84856aa79d386ee5e07400b311610167b201a035b0a35abdcb6c39af1c4f.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/d4cf84856aa79d386ee5e07400b311610167b201a035b0a35abdcb6c39af1c4f.jpg)

![ee6c79515231bddba2c158eb935d7f267993d10abd0233a5d27ffe6464debe85.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/ee6c79515231bddba2c158eb935d7f267993d10abd0233a5d27ffe6464debe85.jpg)

![fe3dabcaaf3c72231ae3221538a4454ab9528494b124636ef5514e292022e291.jpg](../icml_results/1038_RADAR_%20Defending%20RAG%20Dynamically%20against%20Retrieval%20Corruption/tables/fe3dabcaaf3c72231ae3221538a4454ab9528494b124636ef5514e292022e291.jpg)

<a id="paper-1021"></a>

## Escaping the Diversity Trap in Robotic Manipulation via Anchor-Centric Adaptation


### Images

![195b46ae56ce3c7259e8b3cb21f43783ae7f2c2853c3507b4b0a65f6527fa4bc.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/195b46ae56ce3c7259e8b3cb21f43783ae7f2c2853c3507b4b0a65f6527fa4bc.jpg)

![290ab476fcc0d54fd6a3ecbf4e9743652de3bb0c203b675f82724431e1650439.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/290ab476fcc0d54fd6a3ecbf4e9743652de3bb0c203b675f82724431e1650439.jpg)

![33f8cce7b7f39880f2abb434d310e727482b9762a2bdf06f28864b717446e186.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/33f8cce7b7f39880f2abb434d310e727482b9762a2bdf06f28864b717446e186.jpg)

![52dff114d5189dc0a87a86d081ae055e2dd6cd2c15c7a1d3ba2c7f116c4d7420.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/52dff114d5189dc0a87a86d081ae055e2dd6cd2c15c7a1d3ba2c7f116c4d7420.jpg)

![76eb70ebd8aa5504101ad84525ef420302ba5dbd33f0ccb3ba8b1a93c004e4ed.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/76eb70ebd8aa5504101ad84525ef420302ba5dbd33f0ccb3ba8b1a93c004e4ed.jpg)

![96719fa1f8dafe21dc043bbe050be0dac59a3339f6d2ca8187f855d44af75032.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/96719fa1f8dafe21dc043bbe050be0dac59a3339f6d2ca8187f855d44af75032.jpg)

![9f30c856e1b1d7c7f832673ebeb09c26e0c48f5900a7c3a17c5ffef656e729ac.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/9f30c856e1b1d7c7f832673ebeb09c26e0c48f5900a7c3a17c5ffef656e729ac.jpg)

![ce5d36892532669501fd17c6c245676b8e70344b457c18c75ad5c725574d1ecd.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/images/ce5d36892532669501fd17c6c245676b8e70344b457c18c75ad5c725574d1ecd.jpg)

### Tables

![1fc9b476faa9b0ccb28e017f537d8ce8c9d2d68b4bd35f6b925b415f91221230.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/tables/1fc9b476faa9b0ccb28e017f537d8ce8c9d2d68b4bd35f6b925b415f91221230.jpg)

![61de83b9cf8731a5e0bb14fcc3fdf6fc40f094bc98f582357714395fc7c905b7.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/tables/61de83b9cf8731a5e0bb14fcc3fdf6fc40f094bc98f582357714395fc7c905b7.jpg)

![960db3a548a9dc26daf916dca9b2679fdf03f6440777f568308db38f931c2151.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/tables/960db3a548a9dc26daf916dca9b2679fdf03f6440777f568308db38f931c2151.jpg)

![cdb8e6251938b874026aa85979804eb8577847fa5de7914a8dbc90c6877ef8fc.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/tables/cdb8e6251938b874026aa85979804eb8577847fa5de7914a8dbc90c6877ef8fc.jpg)

![fb8515ffa9df994d0be048cd884fdc5bbb9db064401eff64a61ae9d69444723e.jpg](../icml_results/1039_Escaping%20the%20Diversity%20Trap%20in%20Robotic%20Manipulation%20via%20Anchor-Centric%20Adaptation/tables/fb8515ffa9df994d0be048cd884fdc5bbb9db064401eff64a61ae9d69444723e.jpg)

<a id="paper-1022"></a>

## Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective


### Images

![11ddd87e72ee561349fe90065f4fee5f659f3873a5c4f43516e77f606ed8b5e7.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/images/11ddd87e72ee561349fe90065f4fee5f659f3873a5c4f43516e77f606ed8b5e7.jpg)

![4f633da08827936330fc241ee40813ad14b343320da1beffe31313ebb2ea12ea.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/images/4f633da08827936330fc241ee40813ad14b343320da1beffe31313ebb2ea12ea.jpg)

![6c524c1a011cc5e200f4b81b7baf4ca0b56f036333ed70cb2a6b4ca079c9eecd.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/images/6c524c1a011cc5e200f4b81b7baf4ca0b56f036333ed70cb2a6b4ca079c9eecd.jpg)

![a41daceeae6c60ed50a3f8afa760e79cf853dc4ea30c184f3523aa9b0adc2586.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/images/a41daceeae6c60ed50a3f8afa760e79cf853dc4ea30c184f3523aa9b0adc2586.jpg)

### Tables

![07851b1112bb40aa160354a26d5e39038a7f0a3eb04ac71f70527917a0f5031a.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/07851b1112bb40aa160354a26d5e39038a7f0a3eb04ac71f70527917a0f5031a.jpg)

![16eb5f776b560c2253d1132432233e81d0e8c841b7ad928d0b07047c0a9fc86b.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/16eb5f776b560c2253d1132432233e81d0e8c841b7ad928d0b07047c0a9fc86b.jpg)

![614b1f7536156347e7c7ed2a38bbe56be3afbc3878debe858d62f0cb3332d39b.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/614b1f7536156347e7c7ed2a38bbe56be3afbc3878debe858d62f0cb3332d39b.jpg)

![774ea238df26cfe434604a84b63a798147a08d9487c653358ef8fa6c2e916d65.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/774ea238df26cfe434604a84b63a798147a08d9487c653358ef8fa6c2e916d65.jpg)

![9443ab2cb77d01154f45373d8ff03e4caa12b9453a4170bf68a56268645b0ae7.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/9443ab2cb77d01154f45373d8ff03e4caa12b9453a4170bf68a56268645b0ae7.jpg)

![999fc508746782b1ba60455976708e8d3fbfa4b42f9ba76fa3b5dd8969145e19.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/999fc508746782b1ba60455976708e8d3fbfa4b42f9ba76fa3b5dd8969145e19.jpg)

![99cf2de63ac303a782ac6c39b399a53bcef33b8ad0def7ccf35b6e0210d02a95.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/99cf2de63ac303a782ac6c39b399a53bcef33b8ad0def7ccf35b6e0210d02a95.jpg)

![b08eed9a075d8edc6a405953736cf66e4d7fe0fa8e5c63f340df24f2653e170f.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/b08eed9a075d8edc6a405953736cf66e4d7fe0fa8e5c63f340df24f2653e170f.jpg)

![b3e13917af7bdbfb091fe3b8437db3573c78f7c35792536681b3db2b119836f3.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/b3e13917af7bdbfb091fe3b8437db3573c78f7c35792536681b3db2b119836f3.jpg)

![cfc541cc977e803a45ebe5c391b78a4cb116f4739136515ecf938df761b127ba.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/cfc541cc977e803a45ebe5c391b78a4cb116f4739136515ecf938df761b127ba.jpg)

![d8c8d41afe13e9d0b737a3709950e3fd63e4c843664e8ea938aba4fed72579fd.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/d8c8d41afe13e9d0b737a3709950e3fd63e4c843664e8ea938aba4fed72579fd.jpg)

![e03003e746b49d63e3dc911442ec4ef838f8f414f9b947b26f7f73eda09ef427.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/e03003e746b49d63e3dc911442ec4ef838f8f414f9b947b26f7f73eda09ef427.jpg)

![f08e5b8bd5cb0ef53773cdbade2a1ad90872a980579583dab2ebd87710539659.jpg](../icml_results/1040_Message%20Tuning%20Outshines%20Graph%20Prompt%20Tuning_%20A%20Prismatic%20Space%20Perspective/tables/f08e5b8bd5cb0ef53773cdbade2a1ad90872a980579583dab2ebd87710539659.jpg)

<a id="paper-1023"></a>

## How can we assess human-agent interactions? Case studies in software agent design

### Images

![03594c0fa775d8d83a64205f6bd1f7c2eac65fd8c56093bb779d1551cf7b12b0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/03594c0fa775d8d83a64205f6bd1f7c2eac65fd8c56093bb779d1551cf7b12b0.jpg)

![4bf3515e85a60c6e207d55de53a85871df6ec55752edcb3960da22f0d8138700.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/4bf3515e85a60c6e207d55de53a85871df6ec55752edcb3960da22f0d8138700.jpg)

![54682ca1e4a141b90282fa91c510b5ab05eabab9daa653a96a0df57b4edf044f.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/54682ca1e4a141b90282fa91c510b5ab05eabab9daa653a96a0df57b4edf044f.jpg)

![7101e96261d7c2676ac56adc1ddaa143e49e9eed22c1cb0c2965dbebb11465c7.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/7101e96261d7c2676ac56adc1ddaa143e49e9eed22c1cb0c2965dbebb11465c7.jpg)

![948d7bee64dc69764d8902d4a0b97d551054c9243e892bcbed69a1d985f6b180.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/948d7bee64dc69764d8902d4a0b97d551054c9243e892bcbed69a1d985f6b180.jpg)

![cf1e165cbf01c0e9b817309536a8d76a896f1f62d5bbfafdcf9473f612205833.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/cf1e165cbf01c0e9b817309536a8d76a896f1f62d5bbfafdcf9473f612205833.jpg)

![d02abb096e77018a9b8d8d737ea6ea7b55a61d52cbb5aa2b2c88863c450ead8d.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/d02abb096e77018a9b8d8d737ea6ea7b55a61d52cbb5aa2b2c88863c450ead8d.jpg)

![d5b6d8196d798520b24ab191ed55ea3f3416631643d7e3dd4ed6f8adf7e6bb74.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/d5b6d8196d798520b24ab191ed55ea3f3416631643d7e3dd4ed6f8adf7e6bb74.jpg)

![e60c77b9713f747d396c6cbdfcd3a22646afc8f81b8aff7a3cd4cea028bf822f.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/e60c77b9713f747d396c6cbdfcd3a22646afc8f81b8aff7a3cd4cea028bf822f.jpg)

### Tables

![1d34850b2fefe03534b65ab589ca9186b252c82ab4e2a2ab588dea3b3c511c24.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/1d34850b2fefe03534b65ab589ca9186b252c82ab4e2a2ab588dea3b3c511c24.jpg)

![26b68780df479dc2ff86185b18275ff969c28b84f5da10daceef64b4cc6f66a0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/26b68780df479dc2ff86185b18275ff969c28b84f5da10daceef64b4cc6f66a0.jpg)

![2f1f0421f998522cda46ea6aefeee233e679742ad3be4b134a07917a18a798e6.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/2f1f0421f998522cda46ea6aefeee233e679742ad3be4b134a07917a18a798e6.jpg)

![452c6c9c6a7242e60798c42365ab29fc0c95fb418798616b8495f8b50fd2e0f1.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/452c6c9c6a7242e60798c42365ab29fc0c95fb418798616b8495f8b50fd2e0f1.jpg)

![47e1e1f91452d7666620a388159649338d397e90a575f7fe2f4b07ce9e9ccdb4.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/47e1e1f91452d7666620a388159649338d397e90a575f7fe2f4b07ce9e9ccdb4.jpg)

![7b20143ae0ea82828869aa914767d44ee68464a1fbaebbb695ee70df22cf0960.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/7b20143ae0ea82828869aa914767d44ee68464a1fbaebbb695ee70df22cf0960.jpg)

![802f97807470ecf7458f53b9d6f4bfa4b6233accaf68cb27fc99c36fbe68d72a.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/802f97807470ecf7458f53b9d6f4bfa4b6233accaf68cb27fc99c36fbe68d72a.jpg)

![84c93f103ac2c81ce7cf362e7d8c20f4d3515bce372c319d07c0ee7288e125c0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/84c93f103ac2c81ce7cf362e7d8c20f4d3515bce372c319d07c0ee7288e125c0.jpg)

![8bbb5fce0cd1e6a4d6d3098dd4dcffea4f7c54d559f8c3d0fb46593c7edd965d.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/8bbb5fce0cd1e6a4d6d3098dd4dcffea4f7c54d559f8c3d0fb46593c7edd965d.jpg)

![9d2ad05fa9ae844a82fea2d31a201204862b6d2fa8d615bd8ed4a75fbf89c9e9.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/9d2ad05fa9ae844a82fea2d31a201204862b6d2fa8d615bd8ed4a75fbf89c9e9.jpg)

![aeff04730a76e85ed975bedd1bf55f5157b51374fc7c2dff7c9068446b245daf.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/aeff04730a76e85ed975bedd1bf55f5157b51374fc7c2dff7c9068446b245daf.jpg)

![eb22e2153c625a159e4628987ce8da85fa19e8918316a5c938b88437ed665301.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/eb22e2153c625a159e4628987ce8da85fa19e8918316a5c938b88437ed665301.jpg)
