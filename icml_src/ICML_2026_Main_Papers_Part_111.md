# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 111 of 197

## 目录 (Table of Contents)

1. [ECO: Quantized Training without Full-Precision Master Weights](#paper-3631)
2. [Spectral Reach: Understanding Neural Scaling as Progress into the Spectral Tail](#paper-3632)
3. [InvGNN: Learning Invertible Node Representations on Graphs](#paper-3633)
4. [Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning](#paper-3634)
5. [Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](#paper-3635)
6. [V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning](#paper-3636)
7. [When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents](#paper-3637)
8. [UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis](#paper-3638)
9. [Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting](#paper-3639)
10. [Log-Normal Multiplicative Dynamics for Stable Low-Precision Deep Learning](#paper-3640)
11. [Inference-time Alignment with Rewards in Besov Spaces: Provable Advantages of Feature Learning and Multi-Step Policy Updates](#paper-3641)
12. [Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying](#paper-3642)
13. [From Absolute to Relative: Rethinking Reward Shaping in Group-Based Reinforcement Learning](#paper-3643)
14. [Stem: Rethinking Causal Information Flow in Sparse Attention](#paper-3644)
15. [WISE: World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](#paper-3645)
16. [MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](#paper-3646)
17. [Flexibility-Aware Geometric Latent Diffusion for Full-Atom Peptide Design](#paper-3647)
18. [gp2Scale: A Class of Compactly Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points](#paper-3648)
19. [Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning](#paper-3649)
20. [Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics](#paper-3650)
21. [QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs](#paper-3651)
22. [Multi-Round Human–AI Collaboration with User-Specified Requirements](#paper-3652)
23. [Human-AI Collaborative Uncertainty Quantification](#paper-3653)
24. [Deep sequence models tend to memorize geometrically; it is unclear why](#paper-3654)
25. [Resilient Coresets and Clustering](#paper-3655)
26. [MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation](#paper-3656)
27. [MaMa: A Game-Theoretic Approach for Designing Safe Agentic Systems](#paper-3657)
28. [Grokking Finite-Dimensional Algebra](#paper-3658)
29. [A Unified Framework for Diffusion Model Unlearning with f-Divergence](#paper-3659)
30. [ABCD: All Biases Come Disguised](#paper-3660)
31. [An Algebraic View of the Expressivity of Recurrent Language Models](#paper-3661)
32. [GAUSS: Graph-Assisted Uncertainty Quantification using Structure and Semantics for Long-Form Generation in LLMs](#paper-3662)
33. [Joint Enhancement and Classification using Coupled Diffusion Models of Signals and Logits](#paper-3663)

---


<a id="paper-3631"></a>

## ECO: Quantized Training without Full-Precision Master Weights

### Images

![095693d169f12d675c517b01d8294cac69a24aabfab87bcbd4e5d99ee07699e4.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/095693d169f12d675c517b01d8294cac69a24aabfab87bcbd4e5d99ee07699e4.jpg)

![190b6295d3bae6964ae2bd39a1d92d0f10c1cb3a7781e6971efaadd5200a08fb.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/190b6295d3bae6964ae2bd39a1d92d0f10c1cb3a7781e6971efaadd5200a08fb.jpg)

![27c877ac2689864c214cbd652d0dd59734d57772536ec5f979ca0d1576c67e81.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/27c877ac2689864c214cbd652d0dd59734d57772536ec5f979ca0d1576c67e81.jpg)

![3e2c9f5d0a51d9f6ae8305368fa8d1e14c217d2589012bbbe26343a61063b096.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/3e2c9f5d0a51d9f6ae8305368fa8d1e14c217d2589012bbbe26343a61063b096.jpg)

![3f1d10d91cf875a13206124532350a0885f0db6a858ebb8d9f846b265e6c93fc.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/3f1d10d91cf875a13206124532350a0885f0db6a858ebb8d9f846b265e6c93fc.jpg)

![426121a0597e31f424015fd2be135142fc90032100bab322226df4932ab97ac8.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/426121a0597e31f424015fd2be135142fc90032100bab322226df4932ab97ac8.jpg)

![4d403947f3cfe75a39cf827e4419cf3d5103b8139cec65da3912587dd3703b9c.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/4d403947f3cfe75a39cf827e4419cf3d5103b8139cec65da3912587dd3703b9c.jpg)

![71a619b8ef71059a38d76bad45e696d4c4e33f1a21eb5a90068894aedcb85c3b.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/71a619b8ef71059a38d76bad45e696d4c4e33f1a21eb5a90068894aedcb85c3b.jpg)

![a5132758e567097fa49c922fae46468f6091bc8dc3aaec5cd39346aa48708b54.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/a5132758e567097fa49c922fae46468f6091bc8dc3aaec5cd39346aa48708b54.jpg)

![c3556b047a30299dc1780d4fcff81427ce2eb8bfc25305633cdd21bd6ac0d9f4.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/c3556b047a30299dc1780d4fcff81427ce2eb8bfc25305633cdd21bd6ac0d9f4.jpg)

![c4f6efe6865e7edef5918fd08b4cd623927410229f890db39fbcea40063a5dc2.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/c4f6efe6865e7edef5918fd08b4cd623927410229f890db39fbcea40063a5dc2.jpg)

![cd50fcc1b3942f15503fca3bdd9f76d40666245190509a7e3f0fe69cd6ad0a66.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/cd50fcc1b3942f15503fca3bdd9f76d40666245190509a7e3f0fe69cd6ad0a66.jpg)

![d05006493e81e9a6ff7d8034d27ae07a1418759543580c10409e80f75991572f.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/d05006493e81e9a6ff7d8034d27ae07a1418759543580c10409e80f75991572f.jpg)

![dd860c18f41bcd74af83a673095dbf24152200f3b91ac0ab6106be81a21d6bec.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/images/dd860c18f41bcd74af83a673095dbf24152200f3b91ac0ab6106be81a21d6bec.jpg)

### Tables

![03ac3949777e8afd9e50861305fdb72f6e405cf459955da49f55ed17d81bb2d4.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/03ac3949777e8afd9e50861305fdb72f6e405cf459955da49f55ed17d81bb2d4.jpg)

![0e526d03beaab482877aed06b08a9a8954519153c2e619bfc827d556a54958cd.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/0e526d03beaab482877aed06b08a9a8954519153c2e619bfc827d556a54958cd.jpg)

![1acffb57269ca4c163f189bda860c272a622acc7c75d89b26c5d7b4502f33c14.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/1acffb57269ca4c163f189bda860c272a622acc7c75d89b26c5d7b4502f33c14.jpg)

![43c3791aac820aab8437bc5400293b6ec6237ca7b03a0ecd490ff57c3509f1ea.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/43c3791aac820aab8437bc5400293b6ec6237ca7b03a0ecd490ff57c3509f1ea.jpg)

![5019be8e28e19944436dc3bf6902c281c9019b61079461b96f3205a33a8ba309.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/5019be8e28e19944436dc3bf6902c281c9019b61079461b96f3205a33a8ba309.jpg)

![58956ce8446320eeaf010950c8867b5df9e27154ff18d80c2602ba6687d27dbf.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/58956ce8446320eeaf010950c8867b5df9e27154ff18d80c2602ba6687d27dbf.jpg)

![9ee7e9aab15a8a412d2f4d8dcab2ef5b1d03db70bfa7012369b152ef3bbc8f2e.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/9ee7e9aab15a8a412d2f4d8dcab2ef5b1d03db70bfa7012369b152ef3bbc8f2e.jpg)

![e5cf0689fc0d9f42d89d121c5a59a12e2c589631c6a1b3e676d0bfb92f7dfa09.jpg](../icml_results/3670_Asymmetric%20Contrastive%20Objectives%20for%20Efficient%20Phenotypic%20Screening/tables/e5cf0689fc0d9f42d89d121c5a59a12e2c589631c6a1b3e676d0bfb92f7dfa09.jpg)

## ECO: Quantized Training without Full-Precision Master Weights


### Images

![40873516c26118f00d3d0a4e30c86f9f662565c144d98a7572ebf41ab75135f1.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/images/40873516c26118f00d3d0a4e30c86f9f662565c144d98a7572ebf41ab75135f1.jpg)

![6a6de8a42db6c21220c7c6fcd09a4f59786c557f6582db000547c6bc1624b810.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/images/6a6de8a42db6c21220c7c6fcd09a4f59786c557f6582db000547c6bc1624b810.jpg)

![af6102f47cb498835adde34d0bb172d822eabc55e7d078bcddebe4c7b7f0966e.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/images/af6102f47cb498835adde34d0bb172d822eabc55e7d078bcddebe4c7b7f0966e.jpg)

![b2b7884de5dfb0185833b143f17c2d8a6188adbc4c4fb85456fcd7f7eb0923c6.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/images/b2b7884de5dfb0185833b143f17c2d8a6188adbc4c4fb85456fcd7f7eb0923c6.jpg)

### Tables

![4694433ca1ff0944314c92ee23f5dae2616660fe776f7c24521c960519bcc046.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/4694433ca1ff0944314c92ee23f5dae2616660fe776f7c24521c960519bcc046.jpg)

![536e7912f787074ab5e6edfda399343b4523c904daf1f24a5dd11f3334eb2589.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/536e7912f787074ab5e6edfda399343b4523c904daf1f24a5dd11f3334eb2589.jpg)

![63c6d7f8751c48595b02d59e30c6bd6e531f8b960c8812871e111ccad514bb2c.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/63c6d7f8751c48595b02d59e30c6bd6e531f8b960c8812871e111ccad514bb2c.jpg)

![d1c38bda95ae20fa6ffa0f44d6523af989478be91a5446144059acae12b6bca2.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/d1c38bda95ae20fa6ffa0f44d6523af989478be91a5446144059acae12b6bca2.jpg)

![e69b2cecf762847a7c49ea1f759745d5e3a8aed35dbc6e8be75aac0505be23ee.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/e69b2cecf762847a7c49ea1f759745d5e3a8aed35dbc6e8be75aac0505be23ee.jpg)

![e866b7f6537a10d60b279f0ac6da2f6f9d8ff3d24cd599a2677bae17ca6beaff.jpg](../icml_results/3671_ECO_%20Quantized%20Training%20without%20Full-Precision%20Master%20Weights/tables/e866b7f6537a10d60b279f0ac6da2f6f9d8ff3d24cd599a2677bae17ca6beaff.jpg)

<a id="paper-3632"></a>

## Spectral Reach: Understanding Neural Scaling as Progress into the Spectral Tail


### Images

![0634ae72e3a07ca938daf955461376b60c576d6d79e908e241683e52e0478a72.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/0634ae72e3a07ca938daf955461376b60c576d6d79e908e241683e52e0478a72.jpg)

![09a11eddd8ef44c3a4c4bbf86b57958e50475d2dab01422c779ee9325d7d5770.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/09a11eddd8ef44c3a4c4bbf86b57958e50475d2dab01422c779ee9325d7d5770.jpg)

![1835ad55f823b9a58ea6684fc27731bcf6cf088f1f1b45b7c0f04fa2928ff399.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/1835ad55f823b9a58ea6684fc27731bcf6cf088f1f1b45b7c0f04fa2928ff399.jpg)

![240acff55ace82b0fae9b8c949a21e24f2b2fb4f969b6212bdde802758197eb4.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/240acff55ace82b0fae9b8c949a21e24f2b2fb4f969b6212bdde802758197eb4.jpg)

![33f52c74fff73da1563b0b6a2498f0dddb2b98c474c9cc816847d0a468832935.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/33f52c74fff73da1563b0b6a2498f0dddb2b98c474c9cc816847d0a468832935.jpg)

![48207441915db5f2dbdc4d9725704427762b2ffff60e2ae73c96f1f0ca6f5d2b.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/48207441915db5f2dbdc4d9725704427762b2ffff60e2ae73c96f1f0ca6f5d2b.jpg)

![48c0ae0da2399b85e1b8fa7cca4c4f2c31e56ae1b633445b4a262e6783bfc560.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/48c0ae0da2399b85e1b8fa7cca4c4f2c31e56ae1b633445b4a262e6783bfc560.jpg)

![8339cdff00c4bca45ebef87eedfc7360bd4293c1d6a81828d6ae7523e4b21db1.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/8339cdff00c4bca45ebef87eedfc7360bd4293c1d6a81828d6ae7523e4b21db1.jpg)

![8e8966cb4413b20cfb894e30b5168f0007fcd4ee299e7e8aa11f438e10a4e4cd.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/8e8966cb4413b20cfb894e30b5168f0007fcd4ee299e7e8aa11f438e10a4e4cd.jpg)

![d0a1afc745df1009c165f2b121a0d424fc96c5ea55fd2cca98dab954b714efe8.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/d0a1afc745df1009c165f2b121a0d424fc96c5ea55fd2cca98dab954b714efe8.jpg)

![d4d31cb7a8a7c91a9a84c27799d72cdd4d125ea0907646e8c20c3832f5d1a9e5.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/d4d31cb7a8a7c91a9a84c27799d72cdd4d125ea0907646e8c20c3832f5d1a9e5.jpg)

![e0d13e0f8725bed660cb07b4f358c1dbf2da1838dbfbf1657da7d2eae07ae161.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/images/e0d13e0f8725bed660cb07b4f358c1dbf2da1838dbfbf1657da7d2eae07ae161.jpg)

### Tables

![7ec8d3dc071eab38ba9bba6d0211fedbfc0ef2c3614b84480c95bde9a3f53155.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/tables/7ec8d3dc071eab38ba9bba6d0211fedbfc0ef2c3614b84480c95bde9a3f53155.jpg)

![898d74e365f7c69aafec932c4e53175bd48226dc9251082d1ef45944bbf625b1.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/tables/898d74e365f7c69aafec932c4e53175bd48226dc9251082d1ef45944bbf625b1.jpg)

![9376f2ef45e975599d4a6c1694f10491908f2ab7b904347973f53d16828d96c8.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/tables/9376f2ef45e975599d4a6c1694f10491908f2ab7b904347973f53d16828d96c8.jpg)

![bbd283a2bf522c492d531e68cd08bb24aab07647b959822201b93c650f04a8fc.jpg](../icml_results/3672_Spectral%20Reach_%20Understanding%20Neural%20Scaling%20as%20Progress%20into%20the%20Spectral%20Tail/tables/bbd283a2bf522c492d531e68cd08bb24aab07647b959822201b93c650f04a8fc.jpg)

<a id="paper-3633"></a>

## InvGNN: Learning Invertible Node Representations on Graphs


### Images

![0ed902e72937098e918c0da632f6967c56b845e1a14501db5e2b259a51fbffa0.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/0ed902e72937098e918c0da632f6967c56b845e1a14501db5e2b259a51fbffa0.jpg)

![224f1c8205b83ac7a35d34c46baf5372b31237770d1dd0394f148d88cb721d3b.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/224f1c8205b83ac7a35d34c46baf5372b31237770d1dd0394f148d88cb721d3b.jpg)

![484e36e070fc6cc2d0763c74c860afa8e3ab794b1186c6b7b9d65cffc5cecb90.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/484e36e070fc6cc2d0763c74c860afa8e3ab794b1186c6b7b9d65cffc5cecb90.jpg)

![86d839578ccdc74ce98c270109c4cf3873229adc6ef9af71ecf0d9e0b2edb7e5.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/86d839578ccdc74ce98c270109c4cf3873229adc6ef9af71ecf0d9e0b2edb7e5.jpg)

![d387b9d4d5730e0aaca93d4de1a686bd6af26c1acded86634072e06cc73c475f.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/d387b9d4d5730e0aaca93d4de1a686bd6af26c1acded86634072e06cc73c475f.jpg)

![e253be60d97dfe8e21af54374ee64a8b3c0ecb1b12c67d007057832c1bc160b4.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/images/e253be60d97dfe8e21af54374ee64a8b3c0ecb1b12c67d007057832c1bc160b4.jpg)

### Tables

![08ea9ddd95480d666e73d2154bd6ba1d7bb4c47407cc39b2d6488cf5a4e61187.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/08ea9ddd95480d666e73d2154bd6ba1d7bb4c47407cc39b2d6488cf5a4e61187.jpg)

![1a23c6d354a9c8ae72f9afca5757ebf254f9a37013fca0f9a014e7b4c92b76be.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/1a23c6d354a9c8ae72f9afca5757ebf254f9a37013fca0f9a014e7b4c92b76be.jpg)

![2b6b6059037d09817a79796cd8866ab0e78c13fce0481d363f0e8594126d1d6c.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/2b6b6059037d09817a79796cd8866ab0e78c13fce0481d363f0e8594126d1d6c.jpg)

![69679f5a8ce68046b3ccedc9e8679bc8e480889d460a05de8cd7e0ca59e183de.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/69679f5a8ce68046b3ccedc9e8679bc8e480889d460a05de8cd7e0ca59e183de.jpg)

![8e375055310f23341705c737cf446df7ec964c832043c05abfad2a41b93678ff.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/8e375055310f23341705c737cf446df7ec964c832043c05abfad2a41b93678ff.jpg)

![c29c349f622bb4fca9b64d7dc4f40dc0f3c2bd7c2dbfae47607ceeb0b270109e.jpg](../icml_results/3673_InvGNN_%20Learning%20Invertible%20Node%20Representations%20on%20Graphs/tables/c29c349f622bb4fca9b64d7dc4f40dc0f3c2bd7c2dbfae47607ceeb0b270109e.jpg)

<a id="paper-3634"></a>

## Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning


### Images

![68eb7c83ad14d3a53538d2e8e31c767f6141e8f20e16c8830f7e69c60bb7c1fe.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/68eb7c83ad14d3a53538d2e8e31c767f6141e8f20e16c8830f7e69c60bb7c1fe.jpg)

![694517114a5f655b937896ad8611bc5faae498673cc8aa73e01ce0c0a353fd19.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/694517114a5f655b937896ad8611bc5faae498673cc8aa73e01ce0c0a353fd19.jpg)

![698260718947302520b4b19dd2bfdd29b6c95a12faf71e7110b73de56287de18.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/698260718947302520b4b19dd2bfdd29b6c95a12faf71e7110b73de56287de18.jpg)

![93ffe2bdc0375592f9f173e5ce7f0dd64023aab699ca0c857f71cd8a44fd4460.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/93ffe2bdc0375592f9f173e5ce7f0dd64023aab699ca0c857f71cd8a44fd4460.jpg)

![b1a64b9c8cd7775bdbde35a41a39f252360c4eebd5c3c0d1043e75f7b285fe0c.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/b1a64b9c8cd7775bdbde35a41a39f252360c4eebd5c3c0d1043e75f7b285fe0c.jpg)

![cae37076a598e51cb96ddcf084f30ad789558c55dd06e8bf5a13cb3e47bb7bfb.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/cae37076a598e51cb96ddcf084f30ad789558c55dd06e8bf5a13cb3e47bb7bfb.jpg)

![e395eae1f75b4faa3ca11abf7b6bae99764703d24af6e43c07b810cda5e6f36a.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/e395eae1f75b4faa3ca11abf7b6bae99764703d24af6e43c07b810cda5e6f36a.jpg)

![f2a69f8214cd19cf8e4d15cc174d99a65a74db1f91e749a8f87624e79762ddba.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/f2a69f8214cd19cf8e4d15cc174d99a65a74db1f91e749a8f87624e79762ddba.jpg)

![fbe8d26461ebc02a41f1b211f1cfb8bf5a22c932b599928439ad8273240e1c00.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/images/fbe8d26461ebc02a41f1b211f1cfb8bf5a22c932b599928439ad8273240e1c00.jpg)

### Tables

![0860ecce1f3bb333cd5cb075c54ed666b394bbd2e06530cd811a87158d74af81.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/0860ecce1f3bb333cd5cb075c54ed666b394bbd2e06530cd811a87158d74af81.jpg)

![107f6a195a396aa3e0fee6f2ec7b4055601f4aa68cc11725f86ed95f7f761176.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/107f6a195a396aa3e0fee6f2ec7b4055601f4aa68cc11725f86ed95f7f761176.jpg)

![141fee57370712790bbf7ce91b700b9fc0d3f00973ad881ea1588a0f715afb9b.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/141fee57370712790bbf7ce91b700b9fc0d3f00973ad881ea1588a0f715afb9b.jpg)

![22ddb19ee3c00d897b8def5f95dd7a2ac373a2e44c34f97ca8531cfa8cc6f36a.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/22ddb19ee3c00d897b8def5f95dd7a2ac373a2e44c34f97ca8531cfa8cc6f36a.jpg)

![38a1de5250b9a7dc43c419dc277fae7d12d5bd08e32f2b35f2984c7f7a4c32f2.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/38a1de5250b9a7dc43c419dc277fae7d12d5bd08e32f2b35f2984c7f7a4c32f2.jpg)

![66f3b0096620000cca9bb7a9bab8499f869e00513fa578909db681208f3e96d8.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/66f3b0096620000cca9bb7a9bab8499f869e00513fa578909db681208f3e96d8.jpg)

![95b8dc9ca71bde04cc9fa172365194cd27709198ba440f6ac8dfb35b687412f4.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/95b8dc9ca71bde04cc9fa172365194cd27709198ba440f6ac8dfb35b687412f4.jpg)

![9bbe816f69d6afa9b8ff7363b74fce64b787618f07ee7d6d4a18d2ea7d82b720.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/9bbe816f69d6afa9b8ff7363b74fce64b787618f07ee7d6d4a18d2ea7d82b720.jpg)

![b34fd104ca8dd4eb1993b0cb7d6cc55cbadf4dcacdadb1aa689240f4c2190e1b.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/b34fd104ca8dd4eb1993b0cb7d6cc55cbadf4dcacdadb1aa689240f4c2190e1b.jpg)

![c1b09c254e0ac7995f61552718cdcaa8a0c28082d52b57c3096496f94cfccbb7.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/c1b09c254e0ac7995f61552718cdcaa8a0c28082d52b57c3096496f94cfccbb7.jpg)

![f4622c5f12ff5a2d566f79e540066792726f28e2505be87cd30964264270f0b2.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/f4622c5f12ff5a2d566f79e540066792726f28e2505be87cd30964264270f0b2.jpg)

![f79f5eb7d590402e0f57399bcdb4acbe427797295599a9cacfbae962af522a22.jpg](../icml_results/3674_Squeezing%20More%20from%20the%20Stream%20_%20Learning%20Representation%20Online%20for%20Streaming%20Reinforcement%20Learning/tables/f79f5eb7d590402e0f57399bcdb4acbe427797295599a9cacfbae962af522a22.jpg)

<a id="paper-3635"></a>

## Agent-Omit: Adaptive Context Omission for Efficient LLM Agents


### Images

![0d71411f6e2851d8a882ae8331ec0511f9b659b61590bc3bd87f26be96385686.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/0d71411f6e2851d8a882ae8331ec0511f9b659b61590bc3bd87f26be96385686.jpg)

![21e1cac61521fb2dbaa43cf2ae7174fbf185a3eba2cf0604892acbbdea440362.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/21e1cac61521fb2dbaa43cf2ae7174fbf185a3eba2cf0604892acbbdea440362.jpg)

![59b7186db98f1db777a360eddf34705e66a0f0ca3d4b12db2c8a1d164129be4d.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/59b7186db98f1db777a360eddf34705e66a0f0ca3d4b12db2c8a1d164129be4d.jpg)

![72eb4f6b1921d075e09e7e8011aa01925efb7be1f8148139b7650649389b5632.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/72eb4f6b1921d075e09e7e8011aa01925efb7be1f8148139b7650649389b5632.jpg)

![779769facb44186369c052514e3f4c5e9834fdeca6d2856364a35ebd94c9bf88.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/779769facb44186369c052514e3f4c5e9834fdeca6d2856364a35ebd94c9bf88.jpg)

![999cd41211bc209550b514c01fae5d2682363e61797dce732de8b65aca015cf8.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/999cd41211bc209550b514c01fae5d2682363e61797dce732de8b65aca015cf8.jpg)

![9f243ac0514b63a2252bdf89426e8b4dea55a72cadd5db6a24740044ac8b4a54.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/9f243ac0514b63a2252bdf89426e8b4dea55a72cadd5db6a24740044ac8b4a54.jpg)

![f01dfea43d3aacdb8c973c61be1504ac8afc9a238449538b223278d1a474ed32.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/images/f01dfea43d3aacdb8c973c61be1504ac8afc9a238449538b223278d1a474ed32.jpg)

### Tables

![2a81e1420092b82c0ccb474f529b9e524124440f3280f1fccdb96fee56857f00.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/tables/2a81e1420092b82c0ccb474f529b9e524124440f3280f1fccdb96fee56857f00.jpg)

![3c07bf552d35289449767c44c3cde2de3efe2bb009ea21da3e9fc78a263ffb24.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/tables/3c07bf552d35289449767c44c3cde2de3efe2bb009ea21da3e9fc78a263ffb24.jpg)

![721dc2d52f1c6aab3454822aeffd1dca59c47302890cc98e06aa34d5bfe012b7.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/tables/721dc2d52f1c6aab3454822aeffd1dca59c47302890cc98e06aa34d5bfe012b7.jpg)

![f498cbc4cead069f7293c587b1562f4dd67f89ae749fd2819d8508775374614f.jpg](../icml_results/3675_Agent-Omit_%20Adaptive%20Context%20Omission%20for%20Efficient%20LLM%20Agents/tables/f498cbc4cead069f7293c587b1562f4dd67f89ae749fd2819d8508775374614f.jpg)

<a id="paper-3636"></a>

## V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning


### Images

![0d77479a29ac0fd4a08963662be930ac2a34d042b067ac9662b2dd7b1287b67f.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/images/0d77479a29ac0fd4a08963662be930ac2a34d042b067ac9662b2dd7b1287b67f.jpg)

![27be6eee6e23ec5c1b8bdb9b8eeb99a3e8d76da18dfa08a9c16dc077ea533b10.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/images/27be6eee6e23ec5c1b8bdb9b8eeb99a3e8d76da18dfa08a9c16dc077ea533b10.jpg)

![3750ea62260b2b474dc304a2f4e8c6051c7c6cda938c2ee7ddea9e97d0530dd4.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/images/3750ea62260b2b474dc304a2f4e8c6051c7c6cda938c2ee7ddea9e97d0530dd4.jpg)

![c8da9354db7ad29584c929d40aa7b137a4f802104b7bd7a5b964a6bb6dc428f0.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/images/c8da9354db7ad29584c929d40aa7b137a4f802104b7bd7a5b964a6bb6dc428f0.jpg)

![f5be94d2225ce4f7ed9d01c156f0d631289a7566f5ea3e5c02062b7adfd6ca0c.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/images/f5be94d2225ce4f7ed9d01c156f0d631289a7566f5ea3e5c02062b7adfd6ca0c.jpg)

### Tables

![252372f8935f8186bf71b0be7534435169adfabb3df0dbc66f2195bc7b23d9c1.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/252372f8935f8186bf71b0be7534435169adfabb3df0dbc66f2195bc7b23d9c1.jpg)

![444e31b90b7aebb0f18ed1217541221fe1db1890cf8e2a9ae0bcf9df9f99ca4f.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/444e31b90b7aebb0f18ed1217541221fe1db1890cf8e2a9ae0bcf9df9f99ca4f.jpg)

![4f7fad757722ec5d972d29a3f3da032257539ab979eb005035fe22eb9589f9f4.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/4f7fad757722ec5d972d29a3f3da032257539ab979eb005035fe22eb9589f9f4.jpg)

![50524c6063f3fb9c2c254c2a9d3c7286acade20a5f6e4b4cc62a0fce09e5f408.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/50524c6063f3fb9c2c254c2a9d3c7286acade20a5f6e4b4cc62a0fce09e5f408.jpg)

![82f562a61da2368557f16aeadb6322110f9bb1fc74a32221e215653a686a23d9.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/82f562a61da2368557f16aeadb6322110f9bb1fc74a32221e215653a686a23d9.jpg)

![90a37491c019c591a2087c6c086ab91f0264575511c97e0d0e0a645b18e06f78.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/90a37491c019c591a2087c6c086ab91f0264575511c97e0d0e0a645b18e06f78.jpg)

![949af2b028a8fabc62ab994fd22ce9490d1f0025b777b6e2978655e78e12811c.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/949af2b028a8fabc62ab994fd22ce9490d1f0025b777b6e2978655e78e12811c.jpg)

![b7986c5abab06c8a93fcb9ec9fd6070a711f7fc8990ff98a2fda2e9205bd6ae0.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/b7986c5abab06c8a93fcb9ec9fd6070a711f7fc8990ff98a2fda2e9205bd6ae0.jpg)

![d2ffa72e360a41e652a7fb9433ee929c8a4e40d75cf1d799ff87808ffccfcea4.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/d2ffa72e360a41e652a7fb9433ee929c8a4e40d75cf1d799ff87808ffccfcea4.jpg)

![dd32e8d8bb134f58547e74c7b3a82949dc3ea55061a3091e83cb244668a49f28.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/dd32e8d8bb134f58547e74c7b3a82949dc3ea55061a3091e83cb244668a49f28.jpg)

![ebfe30123ff29190b8ea7efb685c667319eb4bbc5d5206b9455b7b4491308510.jpg](../icml_results/3676_V-ABS_%20Action-Observer%20Driven%20Beam%20Search%20for%20Dynamic%20Visual%20Reasoning/tables/ebfe30123ff29190b8ea7efb685c667319eb4bbc5d5206b9455b7b4491308510.jpg)

<a id="paper-3637"></a>

## When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents


### Images

![3bde7b218097aaad3856991e5ac49f30d9119f90a1549fd4d87d94a2bcf26729.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/3bde7b218097aaad3856991e5ac49f30d9119f90a1549fd4d87d94a2bcf26729.jpg)

![4776219fd771523dc5a7aa5b1edc3230a45143ce7a52e6a15924b232c2aac5dc.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/4776219fd771523dc5a7aa5b1edc3230a45143ce7a52e6a15924b232c2aac5dc.jpg)

![7513a8c6dcbb19bc4c3d50eed7671eb33a8ff78bd548b0b9e6ff3a38e945241b.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/7513a8c6dcbb19bc4c3d50eed7671eb33a8ff78bd548b0b9e6ff3a38e945241b.jpg)

![76e5f712b72b122f7fc5e58102d3c5fbe2f36c460192ff0961cd4dd89bf9d251.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/76e5f712b72b122f7fc5e58102d3c5fbe2f36c460192ff0961cd4dd89bf9d251.jpg)

![7f1c01216819c1fb83183b860504a2064cfed7b0772642ced2110f94755a4630.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/7f1c01216819c1fb83183b860504a2064cfed7b0772642ced2110f94755a4630.jpg)

![9e6de354ed66f82f904e7844238fe7e7c80270882992ce100b38c0d696b547ef.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/9e6de354ed66f82f904e7844238fe7e7c80270882992ce100b38c0d696b547ef.jpg)

![a2fdaddf092e3463a2358a3d3bebf3a58e7ad63ea62fe4d4af43ae17ca449f16.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/a2fdaddf092e3463a2358a3d3bebf3a58e7ad63ea62fe4d4af43ae17ca449f16.jpg)

![a498868154fbec4175aa58f36f336a25e97d8f2c45a9785a94086592122b1a84.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/a498868154fbec4175aa58f36f336a25e97d8f2c45a9785a94086592122b1a84.jpg)

![c1a3fb0ca5f8a9e4e9a8b6f16f48b1e709e16fcbadcb3798b92e90342d627ed0.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/c1a3fb0ca5f8a9e4e9a8b6f16f48b1e709e16fcbadcb3798b92e90342d627ed0.jpg)

![d601e18e9b9a7ca475f201dd2bc54323b08685a1cf0efe237543c8bba1a6b159.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/d601e18e9b9a7ca475f201dd2bc54323b08685a1cf0efe237543c8bba1a6b159.jpg)

![dd21593a746a841889e448afa84ac9c74f6899d071c4b31c627afc291aba8d07.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/dd21593a746a841889e448afa84ac9c74f6899d071c4b31c627afc291aba8d07.jpg)

![dd4e239eb2e72fc8c091a36f739fabb3d4bc7a968f93fd0bc1281057eec528d4.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/dd4e239eb2e72fc8c091a36f739fabb3d4bc7a968f93fd0bc1281057eec528d4.jpg)

![f60791c777741ad2b7a068a0442e854b1e0099c165331a508ba571c4d7ad73f7.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/images/f60791c777741ad2b7a068a0442e854b1e0099c165331a508ba571c4d7ad73f7.jpg)

### Tables

![05dd2aa76b58c0509fcc68ef20181ce65e14052e759fd00fe1b1927ad1632300.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/05dd2aa76b58c0509fcc68ef20181ce65e14052e759fd00fe1b1927ad1632300.jpg)

![128e23e4e87c25e69361c5e7c25aafe18e2c26996f3cbdedb56af5d53d17b5b5.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/128e23e4e87c25e69361c5e7c25aafe18e2c26996f3cbdedb56af5d53d17b5b5.jpg)

![1584fd427208bb01620c0938a2c321099fc0384ff5fcec10ce4768269066686e.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/1584fd427208bb01620c0938a2c321099fc0384ff5fcec10ce4768269066686e.jpg)

![1c06abec3208f7a1915b8b3d795aa95c5b7b1b6a775e5372615b9da480e0d4dc.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/1c06abec3208f7a1915b8b3d795aa95c5b7b1b6a775e5372615b9da480e0d4dc.jpg)

![2c42e57c492280bca4768c3e96303668f51fb012821657e30598bf35c2f10a28.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/2c42e57c492280bca4768c3e96303668f51fb012821657e30598bf35c2f10a28.jpg)

![301cd0876834e6ce406a22ef904143ba073ca9663f6ab9e8e6255b20fb8878a1.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/301cd0876834e6ce406a22ef904143ba073ca9663f6ab9e8e6255b20fb8878a1.jpg)

![5012d6a092c437b2aededc39ac41a23292b031da34fe7fe0ad7068d7a3429a64.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/5012d6a092c437b2aededc39ac41a23292b031da34fe7fe0ad7068d7a3429a64.jpg)

![6fc3dfba932b1dc1eac1b3815b6414523e8a971cc8f3415b145e86199e7c5c50.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/6fc3dfba932b1dc1eac1b3815b6414523e8a971cc8f3415b145e86199e7c5c50.jpg)

![6fcfe7c9cc05f52e3961bd2d7e9a713e3e69eeadac34cd253bba25b39d926b2d.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/6fcfe7c9cc05f52e3961bd2d7e9a713e3e69eeadac34cd253bba25b39d926b2d.jpg)

![95ec09d424cb66c95c521815431dab445b122f07da47f5e002cf2ccea17f9797.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/95ec09d424cb66c95c521815431dab445b122f07da47f5e002cf2ccea17f9797.jpg)

![a0b74f583f52b71e49d0eb2c830956656e93be1530c83c403d28f5ab740f4181.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/a0b74f583f52b71e49d0eb2c830956656e93be1530c83c403d28f5ab740f4181.jpg)

![abc2d77a5e101f3580ac74995f94023f153625aba4804760ff754319dddca3d7.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/abc2d77a5e101f3580ac74995f94023f153625aba4804760ff754319dddca3d7.jpg)

![b8c057fea4780bbd9acdac9985e572241d35e6e49d82c0e346f464ef0d37651c.jpg](../icml_results/3677_When%20Actions%20Go%20Off-Task_%20Detecting%20and%20Correcting%20Misaligned%20Actions%20in%20Computer-Use%20Agents/tables/b8c057fea4780bbd9acdac9985e572241d35e6e49d82c0e346f464ef0d37651c.jpg)

<a id="paper-3638"></a>

## UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis


### Images

![0d5dce1dc15613a78ef34ffac062b8ccc9bcf46974a40a79979f1e50dc97ff93.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/0d5dce1dc15613a78ef34ffac062b8ccc9bcf46974a40a79979f1e50dc97ff93.jpg)

![10b03b1561afe47e158e05b162e262b647fa418fbdc6d88e6e38d334c7410eb8.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/10b03b1561afe47e158e05b162e262b647fa418fbdc6d88e6e38d334c7410eb8.jpg)

![120d13a95d3b20b9d5ec2aadad29d5ee020872c1e021dbb3e560ca151cf15ff0.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/120d13a95d3b20b9d5ec2aadad29d5ee020872c1e021dbb3e560ca151cf15ff0.jpg)

![22421fe0e3e7698467f5700b62b278d3344e87c2ecf308ea0f6a2d3b942ae9b3.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/22421fe0e3e7698467f5700b62b278d3344e87c2ecf308ea0f6a2d3b942ae9b3.jpg)

![24eba6a7e897898f20099a28c8f29b817edddf627e5c73ddc5a6b94c64c93068.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/24eba6a7e897898f20099a28c8f29b817edddf627e5c73ddc5a6b94c64c93068.jpg)

![28c43c24ccec3de6c9082c4c7b6c72602e9869b221919b2ad1bd1eb8307d5e2b.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/28c43c24ccec3de6c9082c4c7b6c72602e9869b221919b2ad1bd1eb8307d5e2b.jpg)

![2a02c4931d4dc811df24a5d6aae81c4e993b041e903f7bbf6b07990f6a6aaf4f.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/2a02c4931d4dc811df24a5d6aae81c4e993b041e903f7bbf6b07990f6a6aaf4f.jpg)

![35456772bacd87e7f60b8b7b1d7b24a2aa1f8761b4abc2d40fc7376260ad5edd.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/35456772bacd87e7f60b8b7b1d7b24a2aa1f8761b4abc2d40fc7376260ad5edd.jpg)

![35c000d683a835934e6809d5e5fa921e2507d5f0bcac99518fff9b058ce56b2d.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/35c000d683a835934e6809d5e5fa921e2507d5f0bcac99518fff9b058ce56b2d.jpg)

![3da4bd9c2151ac87c9dfbc17b125d48f0f5e1dae9f5708c3ea67650841500226.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/3da4bd9c2151ac87c9dfbc17b125d48f0f5e1dae9f5708c3ea67650841500226.jpg)

![495c8c7409e7d6d9b8f942df091ed3b4c1581a73a9d0b098bbae15ee505088d1.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/495c8c7409e7d6d9b8f942df091ed3b4c1581a73a9d0b098bbae15ee505088d1.jpg)

![49700ebe9adb1501d85f30d4c2fe69a0d4f1d5df203c125a9e4ef553fb07c2fd.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/49700ebe9adb1501d85f30d4c2fe69a0d4f1d5df203c125a9e4ef553fb07c2fd.jpg)

![5983f6a86e684c0aa30471f9a7e7409d4a111ea0b97ca89dfe931d1b9a58c722.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/5983f6a86e684c0aa30471f9a7e7409d4a111ea0b97ca89dfe931d1b9a58c722.jpg)

![5a3f85445c83e15792443bfb60ae899b7dc63fc68aa2b0ae9d223feab5e5cc1d.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/5a3f85445c83e15792443bfb60ae899b7dc63fc68aa2b0ae9d223feab5e5cc1d.jpg)

![5b5a3a964eda42d6b826bd80ab80dd6dfd374066759a1c9ee5e13882b44cb312.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/5b5a3a964eda42d6b826bd80ab80dd6dfd374066759a1c9ee5e13882b44cb312.jpg)

![5c21364387a3d80eab7fb41937e2dcdd971cb3a0a32ea910fabce8197baf5c8a.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/5c21364387a3d80eab7fb41937e2dcdd971cb3a0a32ea910fabce8197baf5c8a.jpg)

![5e4605abb5098832c256fa9ab27811856658ab5572a057b5856807a4a1a2fce4.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/5e4605abb5098832c256fa9ab27811856658ab5572a057b5856807a4a1a2fce4.jpg)

![719d70a683fb825439353e9d6dec956ebd80625b09ef83fb85d26c2e6ba17453.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/719d70a683fb825439353e9d6dec956ebd80625b09ef83fb85d26c2e6ba17453.jpg)

![7368b526f0552292220937367a92ff8dd79834ef32192dac5f2e8f12137951a1.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/7368b526f0552292220937367a92ff8dd79834ef32192dac5f2e8f12137951a1.jpg)

![82fae6881f3e29747c675442388c65e6c0ec7dd9b3fce47bdc5af7df8871a265.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/82fae6881f3e29747c675442388c65e6c0ec7dd9b3fce47bdc5af7df8871a265.jpg)

![86e93abd6c93328d7300de5001a753425c5bf8de2deec4816782174802010fef.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/86e93abd6c93328d7300de5001a753425c5bf8de2deec4816782174802010fef.jpg)

![8a5ce956c9174656cc3746f0bfbb0249dd2aa1a8c5464370b28d67be2d6a4496.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/8a5ce956c9174656cc3746f0bfbb0249dd2aa1a8c5464370b28d67be2d6a4496.jpg)

![8a61b6edcd5d764b07f194c8a070c12a0efec05b26d3d7b14e24e889490a7ce3.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/8a61b6edcd5d764b07f194c8a070c12a0efec05b26d3d7b14e24e889490a7ce3.jpg)

![8dc9b0c859898e3ed010f5bb9b63920dbed4402f5578dbf61e856c9c05d5203f.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/8dc9b0c859898e3ed010f5bb9b63920dbed4402f5578dbf61e856c9c05d5203f.jpg)

![90883dd96d7abf26351dbf236ae763104b700f73152f1dbd07656ba453153e1b.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/90883dd96d7abf26351dbf236ae763104b700f73152f1dbd07656ba453153e1b.jpg)

![94209c3b001d86fa18269db6d57945d7ae2b783f66b62ca464af095392ad5fe5.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/94209c3b001d86fa18269db6d57945d7ae2b783f66b62ca464af095392ad5fe5.jpg)

![99df0aa667685549920855b4e20e1963440ed385daf447834eaf71a86193d8e8.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/99df0aa667685549920855b4e20e1963440ed385daf447834eaf71a86193d8e8.jpg)

![a937d6699408a8c37d9ea651749434cd62837396d0e94b93a9258c5c07ec894d.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/a937d6699408a8c37d9ea651749434cd62837396d0e94b93a9258c5c07ec894d.jpg)

![ab3cfb35a96bf6475c98e90b9f1a6def63604500fe5f7181b44e205ab8d5631b.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/ab3cfb35a96bf6475c98e90b9f1a6def63604500fe5f7181b44e205ab8d5631b.jpg)

![b1280373fab907912cf2e588340fceb6151344c82377c0658d15aa3f0149fbb7.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/b1280373fab907912cf2e588340fceb6151344c82377c0658d15aa3f0149fbb7.jpg)

![b95ab1298ef4e3ae3b961cbb5704f97f0e0bf7c5f04731c3a77454fcc25cecd6.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/b95ab1298ef4e3ae3b961cbb5704f97f0e0bf7c5f04731c3a77454fcc25cecd6.jpg)

![be0c80420387ca7aff29653f812c4eaf3625ba6070da446a06d195506781b9a6.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/be0c80420387ca7aff29653f812c4eaf3625ba6070da446a06d195506781b9a6.jpg)

![c106752862146361d9fb138b76bad22d9f3332c52766619a034cb1c2df2b87e5.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/c106752862146361d9fb138b76bad22d9f3332c52766619a034cb1c2df2b87e5.jpg)

![c3432c3fe79c5f86cd0e37c20de5566d13b35ae7760534591904ec9d7fa1f0db.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/c3432c3fe79c5f86cd0e37c20de5566d13b35ae7760534591904ec9d7fa1f0db.jpg)

![c7d81afe3dbfe3fa392f73b78289ae03ae12f6e5bff988238bb7f67c4cf6e282.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/c7d81afe3dbfe3fa392f73b78289ae03ae12f6e5bff988238bb7f67c4cf6e282.jpg)

![cb780f71b9fdbbed26d22520ef7df7651c330dabb9c9e157712bcc27dbcb441e.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/cb780f71b9fdbbed26d22520ef7df7651c330dabb9c9e157712bcc27dbcb441e.jpg)

![daca91ccea6031fcd8befe71e627ae582212c8ff159edf7961c52544ec4a4351.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/daca91ccea6031fcd8befe71e627ae582212c8ff159edf7961c52544ec4a4351.jpg)

![e0394d76204321ffe0a720d0663a43e0a2c398763b878c15aa499a8b8b18f772.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/e0394d76204321ffe0a720d0663a43e0a2c398763b878c15aa499a8b8b18f772.jpg)

![e46af288a47c7520b793d1fb8fb9d0d9595e52e009bf3506e6fc1bb32ffdea35.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/e46af288a47c7520b793d1fb8fb9d0d9595e52e009bf3506e6fc1bb32ffdea35.jpg)

![e597a6723025ba49049e97500ba63e8d20bb7a60d8412242ad44380f09035bd0.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/e597a6723025ba49049e97500ba63e8d20bb7a60d8412242ad44380f09035bd0.jpg)

![ea9b9ad7d985485f1e3fb0bc671129205b7b85b9b4ce8e7a2537159f0e3e4879.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/ea9b9ad7d985485f1e3fb0bc671129205b7b85b9b4ce8e7a2537159f0e3e4879.jpg)

![edbe3a764c185b047350b9defac68b4bc530ade0a327fbb97203a5d4bee0fbc6.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/edbe3a764c185b047350b9defac68b4bc530ade0a327fbb97203a5d4bee0fbc6.jpg)

![f038f6b1629a14270214208fc4238a2d26fd873cf2eec00062064239654578bf.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/f038f6b1629a14270214208fc4238a2d26fd873cf2eec00062064239654578bf.jpg)

![f1a87fd33d66a55f47a3e1822a6119e2330f2248dc1577c52b0c09760e6e0e37.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/images/f1a87fd33d66a55f47a3e1822a6119e2330f2248dc1577c52b0c09760e6e0e37.jpg)

### Tables

![08babab4089373cd25e4bcce21460fc765e290ef3b5d9068bdd62087a93685b9.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/08babab4089373cd25e4bcce21460fc765e290ef3b5d9068bdd62087a93685b9.jpg)

![1a2e8b42855fb18bd4895ce415d7525edeae9926aeb4bf1cdc01014d2e84caa1.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/1a2e8b42855fb18bd4895ce415d7525edeae9926aeb4bf1cdc01014d2e84caa1.jpg)

![23ea32e07d83a052714d5ebba71c13de585dcf9bfaf752792778c5480151a5e0.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/23ea32e07d83a052714d5ebba71c13de585dcf9bfaf752792778c5480151a5e0.jpg)

![25de0397e5c02cde19fba2faf67e2744a06ced6145a4d8d798efa141aba1c019.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/25de0397e5c02cde19fba2faf67e2744a06ced6145a4d8d798efa141aba1c019.jpg)

![32f87ee2fa1bbccc909268ac294963a17d1259c9d1532038f3c078572da08980.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/32f87ee2fa1bbccc909268ac294963a17d1259c9d1532038f3c078572da08980.jpg)

![3ee423175375be00481a38f910f5f207a1d2c541abfc9e406f7b033b942bd612.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/3ee423175375be00481a38f910f5f207a1d2c541abfc9e406f7b033b942bd612.jpg)

![3faad9fe5a116a545ce682376bf85e29c3a86fe7c4dec064b54c43d3ebcf67ad.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/3faad9fe5a116a545ce682376bf85e29c3a86fe7c4dec064b54c43d3ebcf67ad.jpg)

![53aa91f7b118cfd983a661c074037bf5a304e75ad16cc1db8c9ee319b9a55b5a.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/53aa91f7b118cfd983a661c074037bf5a304e75ad16cc1db8c9ee319b9a55b5a.jpg)

![5c03f1e257702d1f36f492d73655a0f6f7eb83d862b09bdc8fc3b12dd28c377c.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/5c03f1e257702d1f36f492d73655a0f6f7eb83d862b09bdc8fc3b12dd28c377c.jpg)

![5c28766412ea72b8378eba4487bfbd0035f8815063cee399e1015115444cc71e.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/5c28766412ea72b8378eba4487bfbd0035f8815063cee399e1015115444cc71e.jpg)

![6ca950e7781e55d7400351300ca95a56791c24d4b248227e261f5f62ddfb56bd.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/6ca950e7781e55d7400351300ca95a56791c24d4b248227e261f5f62ddfb56bd.jpg)

![8a2ce3277fd58b90485129e494abd226871852e9d2facafdf8c243e357e7eb22.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/8a2ce3277fd58b90485129e494abd226871852e9d2facafdf8c243e357e7eb22.jpg)

![980837bac94ba272a7f84df23fabaa3ce31d3dca5681453ed5c252a71b028ece.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/980837bac94ba272a7f84df23fabaa3ce31d3dca5681453ed5c252a71b028ece.jpg)

![9cea209cbbecac3b7a4dd944ebc2c23fde830ea9d39a1b271285341fbfe96d1d.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/9cea209cbbecac3b7a4dd944ebc2c23fde830ea9d39a1b271285341fbfe96d1d.jpg)

![a2e10e17548d78b421b99336b3fbf3b0d224cf3f6533592e344f2a08e034a1ff.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/a2e10e17548d78b421b99336b3fbf3b0d224cf3f6533592e344f2a08e034a1ff.jpg)

![ae10e4b73de3faa2be8f394fff76481029546e020fe225642f3b30f2ea7fc42a.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/ae10e4b73de3faa2be8f394fff76481029546e020fe225642f3b30f2ea7fc42a.jpg)

![b16d053e5d0f308f49321c596243c408fb7ea166ab66a0dd594ee8934e379ff4.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/b16d053e5d0f308f49321c596243c408fb7ea166ab66a0dd594ee8934e379ff4.jpg)

![bf99ee2bff8ddffce7c75934518951262e1d8a13cf27c587503f402c51705002.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/bf99ee2bff8ddffce7c75934518951262e1d8a13cf27c587503f402c51705002.jpg)

![cd4be94b60dfb1d8cefb8e6db6611a9b74d7094f4ea8f5d6659bcd6bdf323d0f.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/cd4be94b60dfb1d8cefb8e6db6611a9b74d7094f4ea8f5d6659bcd6bdf323d0f.jpg)

![cd86cf1bf2d8bf3747e6e4d3210864eb4c48d3fc34daf8236818579c433e59f4.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/cd86cf1bf2d8bf3747e6e4d3210864eb4c48d3fc34daf8236818579c433e59f4.jpg)

![d9d7fc1c18eff822578e6feb35d8f9d10055fe75317f4e4c7e04378d94815004.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/d9d7fc1c18eff822578e6feb35d8f9d10055fe75317f4e4c7e04378d94815004.jpg)

![db78abcc8daf73df9552d2a396bd2b3f87f361aef6433007c0ead2546f791b49.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/db78abcc8daf73df9552d2a396bd2b3f87f361aef6433007c0ead2546f791b49.jpg)

![e84101e0d01fd9d5ecb97900b1dc42335965334c0b142d9ce0e55c24340fa878.jpg](../icml_results/3678_UniMedVL_%20Unifying%20Medical%20Multimodal%20Understanding%20and%20Generation%20through%20Observation-Knowledge-Ana/tables/e84101e0d01fd9d5ecb97900b1dc42335965334c0b142d9ce0e55c24340fa878.jpg)

<a id="paper-3639"></a>

## Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting


### Images

![059afb8a8b83fb557af5c32bbf2ad570ae029ea96de99500ecae2a0ed06494a1.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/059afb8a8b83fb557af5c32bbf2ad570ae029ea96de99500ecae2a0ed06494a1.jpg)

![3c359700df03b1e460a5eff13af63ef45f96149a9e0ecfc26ffdb12dddd09516.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/3c359700df03b1e460a5eff13af63ef45f96149a9e0ecfc26ffdb12dddd09516.jpg)

![75fb48e16f0035d8e9d4bdb4ed2208fcf46089af81de0b605c11918b406b96fb.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/75fb48e16f0035d8e9d4bdb4ed2208fcf46089af81de0b605c11918b406b96fb.jpg)

![ab35fbcd37cdd32ccd6f46a05f7571896dfc5f64db46e4176057c1091b679359.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/ab35fbcd37cdd32ccd6f46a05f7571896dfc5f64db46e4176057c1091b679359.jpg)

![ab56ad8b588a139c8f9728ff53bf9d339d09c510eb4785148b46a3eaf07be65c.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/ab56ad8b588a139c8f9728ff53bf9d339d09c510eb4785148b46a3eaf07be65c.jpg)

![ea96aa4d2c044ed78823606e6974ce60261604ab90ec0ed61dd31cae65e8764a.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/images/ea96aa4d2c044ed78823606e6974ce60261604ab90ec0ed61dd31cae65e8764a.jpg)

### Tables

![33b3edbbfcb3f52c27bb7479226c85772bb0d7b9428fdad4998d436e421482e0.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/33b3edbbfcb3f52c27bb7479226c85772bb0d7b9428fdad4998d436e421482e0.jpg)

![3453c155f0b3964fdaf8a69e7ecc7d426e04191d1eb65e34d498d5c67cb730a4.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/3453c155f0b3964fdaf8a69e7ecc7d426e04191d1eb65e34d498d5c67cb730a4.jpg)

![707eb2c50ce8e979340cf68ab8c0c65a23e1ca8c75072532f909700636c031f3.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/707eb2c50ce8e979340cf68ab8c0c65a23e1ca8c75072532f909700636c031f3.jpg)

![8e83517fe7550c43cec95f376441c50185eda983a23783741732986b425c0e06.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/8e83517fe7550c43cec95f376441c50185eda983a23783741732986b425c0e06.jpg)

![915dfc3b95f659a6d2e2bf2dbc4323405a9dff2d795ed15846b37796fe9f7454.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/915dfc3b95f659a6d2e2bf2dbc4323405a9dff2d795ed15846b37796fe9f7454.jpg)

![93d001e35b64ac2430524f872df6bfb866a201d4427a9bd7e2fc195b78ba58d7.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/93d001e35b64ac2430524f872df6bfb866a201d4427a9bd7e2fc195b78ba58d7.jpg)

![a24b675091b7b6603bb8b99be34e967b7044562f099df961741367faa623131d.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/a24b675091b7b6603bb8b99be34e967b7044562f099df961741367faa623131d.jpg)

![c0df4d637385bd834654703a7ca4c9e014fb3afe0d747d6c93b2642fd04c516a.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/c0df4d637385bd834654703a7ca4c9e014fb3afe0d747d6c93b2642fd04c516a.jpg)

![fc0517e029ecf2635c3dbc0e2303e34718346dc9d4b41ffe97c5bbc4c6493d0e.jpg](../icml_results/3679_Beyond%20Heuristics_%20Learnable%20Density%20Control%20for%203D%20Gaussian%20Splatting/tables/fc0517e029ecf2635c3dbc0e2303e34718346dc9d4b41ffe97c5bbc4c6493d0e.jpg)

<a id="paper-3640"></a>

## Log-Normal Multiplicative Dynamics for Stable Low-Precision Deep Learning


### Images

![28355667ff89e4b06b7c98661299e42172c41044ad5d7764d58e733434b6ef1f.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/28355667ff89e4b06b7c98661299e42172c41044ad5d7764d58e733434b6ef1f.jpg)

![4b6af0d588e0ed8a70a9f247cf93113e4d5f6579409620c7d10ed8cd39024119.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/4b6af0d588e0ed8a70a9f247cf93113e4d5f6579409620c7d10ed8cd39024119.jpg)

![7980f2cb1bdc17de1bc3168f88d2ad3692bcd59a7c86bbd1d48f97bcfb4fa8f3.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/7980f2cb1bdc17de1bc3168f88d2ad3692bcd59a7c86bbd1d48f97bcfb4fa8f3.jpg)

![9833de34ce8de7e946724b4e31ba31d64f1b80b0b33da02067845ae15b63dde1.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/9833de34ce8de7e946724b4e31ba31d64f1b80b0b33da02067845ae15b63dde1.jpg)

![bc8b2de60cdfa3315e5b01a9a039cfc5798c6122e7111d7763ebe113a527f7e2.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/bc8b2de60cdfa3315e5b01a9a039cfc5798c6122e7111d7763ebe113a527f7e2.jpg)

![bc9b629e6a7285e28a204cd236e6de81da7311a40f957ceaa67b82d36669bd8c.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/bc9b629e6a7285e28a204cd236e6de81da7311a40f957ceaa67b82d36669bd8c.jpg)

![db9d7a0502ce5d09ecbe8e1bf82544239d794bab78b2e12d1dd4b78493e08265.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/db9d7a0502ce5d09ecbe8e1bf82544239d794bab78b2e12d1dd4b78493e08265.jpg)

![f460f3175b57758ea1efb6f49f3de6957d56a0244c8e1ba8975c904e48d767f7.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/images/f460f3175b57758ea1efb6f49f3de6957d56a0244c8e1ba8975c904e48d767f7.jpg)

### Tables

![32bbd3959510af264b8fa63aed9e04ae4295e5aa8baa072f68ce5e84b1f5aa51.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/32bbd3959510af264b8fa63aed9e04ae4295e5aa8baa072f68ce5e84b1f5aa51.jpg)

![70cf013a8deb430712370dcc935d06c53b15834e20a6877cbab03c46c7b07fc2.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/70cf013a8deb430712370dcc935d06c53b15834e20a6877cbab03c46c7b07fc2.jpg)

![8b83271f33ecc159769108ed5ee06a4586a6db824ed9707c5c3ad56052941ad0.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/8b83271f33ecc159769108ed5ee06a4586a6db824ed9707c5c3ad56052941ad0.jpg)

![a213ef53403484780388232a0e93aa4fcb9e61559a7ddeed1ec37e92003aa7eb.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/a213ef53403484780388232a0e93aa4fcb9e61559a7ddeed1ec37e92003aa7eb.jpg)

![a3c1f4481a31deff264532f654a202119a74ae0d6407406d5d3d8731ce228121.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/a3c1f4481a31deff264532f654a202119a74ae0d6407406d5d3d8731ce228121.jpg)

![cb826a73fc9065b296997cdfe168d4732a5e5548e93db764bf814bddecf829ea.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/cb826a73fc9065b296997cdfe168d4732a5e5548e93db764bf814bddecf829ea.jpg)

![db856ca9c42ee379bac98d40450e425360c257ccc0fddb0cf33d0f016125419f.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/db856ca9c42ee379bac98d40450e425360c257ccc0fddb0cf33d0f016125419f.jpg)

![e49e06f5c2bb5900251ff3927a9c10fc89cd874b2605a7eda426c079712f7b8e.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/e49e06f5c2bb5900251ff3927a9c10fc89cd874b2605a7eda426c079712f7b8e.jpg)

![f64ac2b3b2d5031569a67e05a8710ddb9758c0a3591be7d23c7dcb12e3411e0f.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/f64ac2b3b2d5031569a67e05a8710ddb9758c0a3591be7d23c7dcb12e3411e0f.jpg)

![ff38e20e0de4c8dc219ad6bbcea26ff35bf68ad0637ab270477e387911af69b6.jpg](../icml_results/3680_Log-Normal%20Multiplicative%20Dynamics%20for%20Stable%20Low-Precision%20Deep%20Learning/tables/ff38e20e0de4c8dc219ad6bbcea26ff35bf68ad0637ab270477e387911af69b6.jpg)

<a id="paper-3641"></a>

## Inference-time Alignment with Rewards in Besov Spaces: Provable Advantages of Feature Learning and Multi-Step Policy Updates


### Images

![a2fe78cd0f9ec3ca877bfdbe6424551a21e7184efa5f2077704c14a59cb3997f.jpg](../icml_results/3681_Inference-time%20Alignment%20with%20Rewards%20in%20Besov%20Spaces_%20Provable%20Advantages%20of%20Feature%20Learning%20and%20M/images/a2fe78cd0f9ec3ca877bfdbe6424551a21e7184efa5f2077704c14a59cb3997f.jpg)

![a9b4a3ca653b5af0ecb7def6480572cd7e46800e3d1f81ea50b3f98588f08959.jpg](../icml_results/3681_Inference-time%20Alignment%20with%20Rewards%20in%20Besov%20Spaces_%20Provable%20Advantages%20of%20Feature%20Learning%20and%20M/images/a9b4a3ca653b5af0ecb7def6480572cd7e46800e3d1f81ea50b3f98588f08959.jpg)

<a id="paper-3642"></a>

## Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying


### Images

![0b8f1b4f40488ff6e8134c5b02661be8eb01ae58bfce81039caca554b9352f55.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/0b8f1b4f40488ff6e8134c5b02661be8eb01ae58bfce81039caca554b9352f55.jpg)

![14157125d2e0dfcd59ab8e6b6ee0d089d2bc5d7e6fd20b6538af30fc6b4c98e8.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/14157125d2e0dfcd59ab8e6b6ee0d089d2bc5d7e6fd20b6538af30fc6b4c98e8.jpg)

![2c8a3c60fb5525d83530154ed0eb000fc0102acfdbf75d381fdc57189cb106e0.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/2c8a3c60fb5525d83530154ed0eb000fc0102acfdbf75d381fdc57189cb106e0.jpg)

![2d5727b0e4429a8af3f8e8e6a35071693877483e20c1a83e822f56f0adf8723b.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/2d5727b0e4429a8af3f8e8e6a35071693877483e20c1a83e822f56f0adf8723b.jpg)

![2ed7ca187728d8927d8694dca321e5db2ec2f8ab4c1b1cb719c8096be3f3a573.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/2ed7ca187728d8927d8694dca321e5db2ec2f8ab4c1b1cb719c8096be3f3a573.jpg)

![3b84d301b1505c6756fd5fbee6f3bb15309904753b5a75b949950201f05c5b05.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/3b84d301b1505c6756fd5fbee6f3bb15309904753b5a75b949950201f05c5b05.jpg)

![4e1cecf082cad543f8f3d2cf1909dbd858ab74915e6a9dac041dbd6805afbc08.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/4e1cecf082cad543f8f3d2cf1909dbd858ab74915e6a9dac041dbd6805afbc08.jpg)

![54da7a53ca009023239fc83a898a668377dfe172a618c79558aca2871bcfe77e.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/54da7a53ca009023239fc83a898a668377dfe172a618c79558aca2871bcfe77e.jpg)

![788a9db36cdad500810fb00d67bde1e4cc6c0fc264ae58c6f08d6ee25650f83d.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/788a9db36cdad500810fb00d67bde1e4cc6c0fc264ae58c6f08d6ee25650f83d.jpg)

![792f267e0e42647cb276e4d8cc3e0fc30832bdf651a9e1ce3b370c331421b347.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/792f267e0e42647cb276e4d8cc3e0fc30832bdf651a9e1ce3b370c331421b347.jpg)

![b26f6d8ac692df9bc117d4da63c7d77cf52aa9b74f90bd6e361918b4e84c56ed.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/b26f6d8ac692df9bc117d4da63c7d77cf52aa9b74f90bd6e361918b4e84c56ed.jpg)

![bb7b71c396f2718946580045dcf2632f066b8c8d64cd6ca5d99e31bdbbf35a0e.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/bb7b71c396f2718946580045dcf2632f066b8c8d64cd6ca5d99e31bdbbf35a0e.jpg)

![cda53ba3557229bfd453e1fcd9e3ce7f44b6d3a5b7f7200f257f5314debb522e.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/cda53ba3557229bfd453e1fcd9e3ce7f44b6d3a5b7f7200f257f5314debb522e.jpg)

![d39f25c15270ed4f3c17f41ee748f2941db8959628d5428948daf92ff03b6a0d.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/d39f25c15270ed4f3c17f41ee748f2941db8959628d5428948daf92ff03b6a0d.jpg)

![f766f45a8bd6ed1d40cc583b055e155e9b815d0837450396efcab2e785054405.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/f766f45a8bd6ed1d40cc583b055e155e9b815d0837450396efcab2e785054405.jpg)

![f956a739ddf537d36e579af1ca5505f41b9e52666494e2f0153cdb73c70cb3f5.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/images/f956a739ddf537d36e579af1ca5505f41b9e52666494e2f0153cdb73c70cb3f5.jpg)

### Tables

![0beee3ec8ccf2af6341950ded24c7410e81ad4e850c2158471a64d708c29459b.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/0beee3ec8ccf2af6341950ded24c7410e81ad4e850c2158471a64d708c29459b.jpg)

![21a1ae504cefec8fe5280fba54902fe607786a455c29b47d3aba59f40ed175e6.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/21a1ae504cefec8fe5280fba54902fe607786a455c29b47d3aba59f40ed175e6.jpg)

![40cd04e878b5c505e3cc3b5b76b53dbfa0399d0908f5752fa03d0fdc3ccc3ee7.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/40cd04e878b5c505e3cc3b5b76b53dbfa0399d0908f5752fa03d0fdc3ccc3ee7.jpg)

![cc8562e57235bc6f871c5fe2b5911db9594fe4f52fdad1db5eedc2b4d6a60911.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/cc8562e57235bc6f871c5fe2b5911db9594fe4f52fdad1db5eedc2b4d6a60911.jpg)

![cd487139797cc75d82f525faa38550e34ac494121b58e9d070e05ffe7ecf843a.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/cd487139797cc75d82f525faa38550e34ac494121b58e9d070e05ffe7ecf843a.jpg)

![d3446598c5996fcb05c73cdaacd9bb39ef42cb5ef955ca72bd538dc48a58432a.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/d3446598c5996fcb05c73cdaacd9bb39ef42cb5ef955ca72bd538dc48a58432a.jpg)

![d97c7a19b4f8cd98e98b687912cc409d5fecf09fe10a5260e4d3864e6d1a53dd.jpg](../icml_results/3682_Emergence%20of%20Exploration%20in%20Policy%20Gradient%20Reinforcement%20Learning%20via%20Retrying/tables/d97c7a19b4f8cd98e98b687912cc409d5fecf09fe10a5260e4d3864e6d1a53dd.jpg)

<a id="paper-3643"></a>

## From Absolute to Relative: Rethinking Reward Shaping in Group-Based Reinforcement Learning


### Images

![18996d98722c42be5b73a8cd254e83c4987c989150f0c2d035e1f95c609c7acf.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/18996d98722c42be5b73a8cd254e83c4987c989150f0c2d035e1f95c609c7acf.jpg)

![6737894d8102b61194c4010bd2bf04991007c7610ca6232b01534e612089b3a8.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/6737894d8102b61194c4010bd2bf04991007c7610ca6232b01534e612089b3a8.jpg)

![68dff91cf00f25cefbbc13002408e3033716707ebcc2aa2f2a40efa0a52af40b.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/68dff91cf00f25cefbbc13002408e3033716707ebcc2aa2f2a40efa0a52af40b.jpg)

![781fd93c0772d567ee1efb7b9bb50cec76e60c8cf5f70d3fc2481ed52215cec8.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/781fd93c0772d567ee1efb7b9bb50cec76e60c8cf5f70d3fc2481ed52215cec8.jpg)

![7da60c1b6f6f6acd620c1787cf55680de8bb2c5719a7778d6c596d7817f0c41e.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/7da60c1b6f6f6acd620c1787cf55680de8bb2c5719a7778d6c596d7817f0c41e.jpg)

![9e8e212043c152bdc6ce4fbeb689e323c98c9641501e42bafe54e9009e772e0e.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/9e8e212043c152bdc6ce4fbeb689e323c98c9641501e42bafe54e9009e772e0e.jpg)

![cdb913bb840bf321436d4f472c2cbbe1cc0e044f5d370e3db221d88d271a9225.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/cdb913bb840bf321436d4f472c2cbbe1cc0e044f5d370e3db221d88d271a9225.jpg)

![ef09b9385d56d86e7824c72b8e02dbf195340d187645adee97045f886d35152a.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/ef09b9385d56d86e7824c72b8e02dbf195340d187645adee97045f886d35152a.jpg)

![f7175b22628cddb56b8621f732c0f3ccb966b12c3a70c795a5ef14da3be8e4ec.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/images/f7175b22628cddb56b8621f732c0f3ccb966b12c3a70c795a5ef14da3be8e4ec.jpg)

### Tables

![215694f1355a8d1ca7e5d8e4255a33f50cbb8ba12590efcc7e6eba303d65b21a.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/215694f1355a8d1ca7e5d8e4255a33f50cbb8ba12590efcc7e6eba303d65b21a.jpg)

![43614dc0c93087dedc8b815f0e398fcb5d5b048ef51cc6c4ac2f7da1df826a22.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/43614dc0c93087dedc8b815f0e398fcb5d5b048ef51cc6c4ac2f7da1df826a22.jpg)

![732af7a477acb40ae61f70a14ad9b292eaf4a004edd73eb072415c85d82c38ce.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/732af7a477acb40ae61f70a14ad9b292eaf4a004edd73eb072415c85d82c38ce.jpg)

![7d7ed992316fad28330fbe59a69f31a7e69758d7d3a8db1d646ac97f65354d5d.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/7d7ed992316fad28330fbe59a69f31a7e69758d7d3a8db1d646ac97f65354d5d.jpg)

![7ea8cf7d6aa0a4ef26654d0187c1b2f38b07af0cb22254f31fc5029b470b04c7.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/7ea8cf7d6aa0a4ef26654d0187c1b2f38b07af0cb22254f31fc5029b470b04c7.jpg)

![9b7a1f0ea9174d2e44939845644189731442c15f517f8e995fe006e2648c1294.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/9b7a1f0ea9174d2e44939845644189731442c15f517f8e995fe006e2648c1294.jpg)

![aa1359f8f09de4576a17c2804829ee8a1a58412ef8cfda5480b22a7802a6f581.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/aa1359f8f09de4576a17c2804829ee8a1a58412ef8cfda5480b22a7802a6f581.jpg)

![c3edc4657f5f8db779f6efab47cf3d40f91405994c807ac1d9010515e2231d91.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/c3edc4657f5f8db779f6efab47cf3d40f91405994c807ac1d9010515e2231d91.jpg)

![e21a071ed83b5e2c794066919cd066536762de71ba9a54b12d18ad66f1887a44.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/e21a071ed83b5e2c794066919cd066536762de71ba9a54b12d18ad66f1887a44.jpg)

![edc63be08f906525c8dcc094cf41fc76387a962d3cea97fac0bc035bae45240e.jpg](../icml_results/3683_From%20Absolute%20to%20Relative_%20Rethinking%20Reward%20Shaping%20in%20Group-Based%20Reinforcement%20Learning/tables/edc63be08f906525c8dcc094cf41fc76387a962d3cea97fac0bc035bae45240e.jpg)

<a id="paper-3644"></a>

## Stem: Rethinking Causal Information Flow in Sparse Attention


### Images

![34e4b2612ec8f11fa2c9b2d240a5815cadffb1dc6f11d920630cb9f7373d394c.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/images/34e4b2612ec8f11fa2c9b2d240a5815cadffb1dc6f11d920630cb9f7373d394c.jpg)

![641d6854ad0c9ad173e2c3a93054268b9bcf79c38649aca03523612cf5e68014.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/images/641d6854ad0c9ad173e2c3a93054268b9bcf79c38649aca03523612cf5e68014.jpg)

![75b1df387ae4952c5da614a2d291e1159f58420329fada2a3d4b61d0d1a0d995.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/images/75b1df387ae4952c5da614a2d291e1159f58420329fada2a3d4b61d0d1a0d995.jpg)

![926896d550760dd1e5bc266f79b1cb11b81601c9689dcc18c9b6ec10e066405f.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/images/926896d550760dd1e5bc266f79b1cb11b81601c9689dcc18c9b6ec10e066405f.jpg)

![b7d9e08442dcaa79499187080324715803480e9b4d1cc4d551eeb747bffd67f2.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/images/b7d9e08442dcaa79499187080324715803480e9b4d1cc4d551eeb747bffd67f2.jpg)

### Tables

![2b3fe74cc31c3fdb02ed48b735e18c10b2429f74c294196d51f01d95a3dd85b7.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/tables/2b3fe74cc31c3fdb02ed48b735e18c10b2429f74c294196d51f01d95a3dd85b7.jpg)

![3aef27a13f687b4eb4f0eaba87ff0e98290164d64a6e8e259476d4d9c327ad1f.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/tables/3aef27a13f687b4eb4f0eaba87ff0e98290164d64a6e8e259476d4d9c327ad1f.jpg)

![bfb6a692944edd999eefc31a86b0910703db48a6893aa3fb68e488422081db21.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/tables/bfb6a692944edd999eefc31a86b0910703db48a6893aa3fb68e488422081db21.jpg)

![eee9df1a89875813ec74dac6c562c9fc6b6f76878de87a961418e6a6450521ac.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/tables/eee9df1a89875813ec74dac6c562c9fc6b6f76878de87a961418e6a6450521ac.jpg)

![fa16fc89911c276d3ef8cf1e8a88d05b7580ec95411a8838015e32fda29b7ced.jpg](../icml_results/3684_Stem_%20Rethinking%20Causal%20Information%20Flow%20in%20Sparse%20Attention/tables/fa16fc89911c276d3ef8cf1e8a88d05b7580ec95411a8838015e32fda29b7ced.jpg)

<a id="paper-3645"></a>

## WISE: World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation


### Images

![2b3d27f90ae48ace5b2628c1c0554966b72c85562e97f951865f45d88e1205cb.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/images/2b3d27f90ae48ace5b2628c1c0554966b72c85562e97f951865f45d88e1205cb.jpg)

![a7918f0b3ad2b4a5419e657c092f816e5476023f83aacd17592af83b3cdc1311.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/images/a7918f0b3ad2b4a5419e657c092f816e5476023f83aacd17592af83b3cdc1311.jpg)

![d6f02478f92885986cc287088f7fd9211bb9ffc6e19afb0e57fc268970917930.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/images/d6f02478f92885986cc287088f7fd9211bb9ffc6e19afb0e57fc268970917930.jpg)

![f148b5e01f5fe86c6170ef65189295fcaf20de4869322014e92b69f56194e8a9.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/images/f148b5e01f5fe86c6170ef65189295fcaf20de4869322014e92b69f56194e8a9.jpg)

### Tables

![3416345be89c949bff6e759a52beea7f659ec1664d4c269d5e2415b2cdc0e51e.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/3416345be89c949bff6e759a52beea7f659ec1664d4c269d5e2415b2cdc0e51e.jpg)

![3d07be1a7db68ce10f319f998132cb8034bdf75695087e23e4f8d794e6e35974.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/3d07be1a7db68ce10f319f998132cb8034bdf75695087e23e4f8d794e6e35974.jpg)

![5ccf306dcf76c3a27123a35559a2cdc5c703daf6f87385c94383c02b6dd22f1a.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/5ccf306dcf76c3a27123a35559a2cdc5c703daf6f87385c94383c02b6dd22f1a.jpg)

![7876386a092c0a2c88a90201e190d75006da2be42af2be5001cf4fa278c78e4f.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/7876386a092c0a2c88a90201e190d75006da2be42af2be5001cf4fa278c78e4f.jpg)

![8a92d4c41286e199ffd98a358ee92efd305670a93ff7004139a168484a4e00e6.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/8a92d4c41286e199ffd98a358ee92efd305670a93ff7004139a168484a4e00e6.jpg)

![9dfb7bdacf0e3d8468a354af1c571c08edc7c50b66e3da698edab1c0b8885a82.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/9dfb7bdacf0e3d8468a354af1c571c08edc7c50b66e3da698edab1c0b8885a82.jpg)

![a637d0e71f471a6eba0862ca80729759034ff6c864d0b794f8d41fa2d421984e.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/a637d0e71f471a6eba0862ca80729759034ff6c864d0b794f8d41fa2d421984e.jpg)

![b10da7496c379e640bd9ef58711e60c42f5282b783e5787b537972f279d2e23e.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/b10da7496c379e640bd9ef58711e60c42f5282b783e5787b537972f279d2e23e.jpg)

![b36d2fd8b5ef12072e19330ce4c55ccecadd1f6b9ce273bd9a4f49038c50e72e.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/b36d2fd8b5ef12072e19330ce4c55ccecadd1f6b9ce273bd9a4f49038c50e72e.jpg)

![e2d9666929c27ea7edc4b94e5d7c25eadddcb84b14f401a5b8da5257ecd881e8.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/e2d9666929c27ea7edc4b94e5d7c25eadddcb84b14f401a5b8da5257ecd881e8.jpg)

![ea7a94ca278566bf2039ed8303a142a34228204d588e73ad5d37be8682f78f60.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/ea7a94ca278566bf2039ed8303a142a34228204d588e73ad5d37be8682f78f60.jpg)

![f84061b57d59e31e1159779ed40ae480e85640dd69dd8cfea994f72d1aa1d406.jpg](../icml_results/3685_WISE_%20World%20Knowledge-Informed%20Semantic%20Evaluation%20for%20Text-to-Image%20Generation/tables/f84061b57d59e31e1159779ed40ae480e85640dd69dd8cfea994f72d1aa1d406.jpg)

<a id="paper-3646"></a>

## MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks


### Images

![329e6a2c5426645ee03d988f678a58059ae1f59b235003827e41da370601d328.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/images/329e6a2c5426645ee03d988f678a58059ae1f59b235003827e41da370601d328.jpg)

![4d06e72f9d1bb5591d9a7f79486d9368c35e85c8e8155b8af6454782a69310ed.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/images/4d06e72f9d1bb5591d9a7f79486d9368c35e85c8e8155b8af6454782a69310ed.jpg)

![99c45d88e625e5912c133901d08b0638986cb95d864d2e8152fdf64760f342c8.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/images/99c45d88e625e5912c133901d08b0638986cb95d864d2e8152fdf64760f342c8.jpg)

![d16abe954b898a47feb301f9042675f6f2ba27820da81cd62f215c06fa8d8357.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/images/d16abe954b898a47feb301f9042675f6f2ba27820da81cd62f215c06fa8d8357.jpg)

![e0dfe7e99e20fa167b7bc41e4c8298efd8056cecaee0a710096e1562aa9b6d5e.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/images/e0dfe7e99e20fa167b7bc41e4c8298efd8056cecaee0a710096e1562aa9b6d5e.jpg)

### Tables

![154610f382b908c6a1eb12f53af4cffc1537ab4ab57fefec113d7d0b71ee3bd2.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/154610f382b908c6a1eb12f53af4cffc1537ab4ab57fefec113d7d0b71ee3bd2.jpg)

![155361d8cc56a38a8620334a53352ab92cc44aa03524512c341efebacff49dc8.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/155361d8cc56a38a8620334a53352ab92cc44aa03524512c341efebacff49dc8.jpg)

![1900d3e724aa8251b0384948609dcbc5a854cf1ecd8b22a7da0eeafe15940e1e.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/1900d3e724aa8251b0384948609dcbc5a854cf1ecd8b22a7da0eeafe15940e1e.jpg)

![2193b4b7824a8edd31a2d09ef8651eef62d8e5dabbe39bfab199905a5a669c5d.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/2193b4b7824a8edd31a2d09ef8651eef62d8e5dabbe39bfab199905a5a669c5d.jpg)

![2233f45c674278bf331945f4423c916c88913369dd6ab7351d42d757dc0443db.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/2233f45c674278bf331945f4423c916c88913369dd6ab7351d42d757dc0443db.jpg)

![286a5359da276e6db45314ceffd6faa36fac1300178a53d425f1dced88a2d238.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/286a5359da276e6db45314ceffd6faa36fac1300178a53d425f1dced88a2d238.jpg)

![674bc89837d665575e714f70e9786bba8e208d94c8a6ec8f16ff3cd7141c4712.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/674bc89837d665575e714f70e9786bba8e208d94c8a6ec8f16ff3cd7141c4712.jpg)

![69080a10de72c8165f221bee8da949669ce07f82a8253e07e516941ba07ebea1.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/69080a10de72c8165f221bee8da949669ce07f82a8253e07e516941ba07ebea1.jpg)

![6e9461f104576f69aa15d9d2eea72bcc2655bd56b99314b3285e6ff03c2d7950.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/6e9461f104576f69aa15d9d2eea72bcc2655bd56b99314b3285e6ff03c2d7950.jpg)

![78203fa782bd576d08851d6c204f1e6cb0797b306f057d02d380dd9e7e2ca2fd.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/78203fa782bd576d08851d6c204f1e6cb0797b306f057d02d380dd9e7e2ca2fd.jpg)

![7f110a8650e4bcd6b64f9a0cf098db2dd4f7e12e251a05131789281c24eeee06.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/7f110a8650e4bcd6b64f9a0cf098db2dd4f7e12e251a05131789281c24eeee06.jpg)

![96f72a3bc1e631fd5319afb35a1ee02e0e66341d6441b210b5a1a0a692de8bb5.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/96f72a3bc1e631fd5319afb35a1ee02e0e66341d6441b210b5a1a0a692de8bb5.jpg)

![9fb07d848dc8184845b6c8070f0884e5a2a97210ad627e39e477148198197802.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/9fb07d848dc8184845b6c8070f0884e5a2a97210ad627e39e477148198197802.jpg)

![ae553ace71e9559430f4b4b2634b7828bc0fa05df19dfe5482e9dff33bef6242.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/ae553ace71e9559430f4b4b2634b7828bc0fa05df19dfe5482e9dff33bef6242.jpg)

![b6c19a12671de4a799f4cf0ace6779685f06bcf72969a9c317e7879d7a4d6ae3.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/b6c19a12671de4a799f4cf0ace6779685f06bcf72969a9c317e7879d7a4d6ae3.jpg)

![d24a8eade106b678057aeab031ec74bad9c9fa48c9fba0b6f89248ef97a095b7.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/d24a8eade106b678057aeab031ec74bad9c9fa48c9fba0b6f89248ef97a095b7.jpg)

![d3f9c7083a30f9885b5a87754d61156fc146908ce405c51fbf5c6ce48aeca453.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/d3f9c7083a30f9885b5a87754d61156fc146908ce405c51fbf5c6ce48aeca453.jpg)

![d429e58fb5b8795917707ecd7e5f4a8f36d80d09863f18c5813446964eb7b764.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/d429e58fb5b8795917707ecd7e5f4a8f36d80d09863f18c5813446964eb7b764.jpg)

![d78b301da102f47a9e7bc0568806fa9545a18af18a037cd9beaa5c8d5452608b.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/d78b301da102f47a9e7bc0568806fa9545a18af18a037cd9beaa5c8d5452608b.jpg)

![e3af5349d0cb73b8c91822e36066e874456e13e82689a67fc406ec65d0c787ab.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/e3af5349d0cb73b8c91822e36066e874456e13e82689a67fc406ec65d0c787ab.jpg)

![e4ee23bbf0d15cc11e6ce76b445678f97646f6888e0632c5332573834ada2bb6.jpg](../icml_results/3686_MECAT_%20A%20Multi-Experts%20Constructed%20Benchmark%20for%20Fine-Grained%20Audio%20Understanding%20Tasks/tables/e4ee23bbf0d15cc11e6ce76b445678f97646f6888e0632c5332573834ada2bb6.jpg)

<a id="paper-3647"></a>

## Flexibility-Aware Geometric Latent Diffusion for Full-Atom Peptide Design


### Images

![0731f722ac979ab77de0930eb17602a5cf38bddef2be157e5c4f12fccee96d80.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/0731f722ac979ab77de0930eb17602a5cf38bddef2be157e5c4f12fccee96d80.jpg)

![24bd4daf18228b060f94208826057fbafbf4fa8ef905460279c340f6570e421a.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/24bd4daf18228b060f94208826057fbafbf4fa8ef905460279c340f6570e421a.jpg)

![42883dc05e8e465c505ec37fb9eb272490d370050d19ccababf55c6c54fdf78b.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/42883dc05e8e465c505ec37fb9eb272490d370050d19ccababf55c6c54fdf78b.jpg)

![471d40945f24ee76e2c2afcdc3b47ced172de89f1d43cd6427d6fc27938cafc9.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/471d40945f24ee76e2c2afcdc3b47ced172de89f1d43cd6427d6fc27938cafc9.jpg)

![4f4930dedc85e07505033403fe57a37592d3d48fcc26d1bae421007d374c7153.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/4f4930dedc85e07505033403fe57a37592d3d48fcc26d1bae421007d374c7153.jpg)

![84849fe46ebc6a8eccd6197b9c356432e6800e55d1fd63df56d351e8a6f08926.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/84849fe46ebc6a8eccd6197b9c356432e6800e55d1fd63df56d351e8a6f08926.jpg)

![9075978cc55f1b0b0f24c77a7e657d3dc0866ceb2acd202e420c1004c15058c1.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/9075978cc55f1b0b0f24c77a7e657d3dc0866ceb2acd202e420c1004c15058c1.jpg)

![a14cc3fa5aa2eccf9bbae7f333b27860cafbcba09c9b55d75e05e7cf9baff441.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/a14cc3fa5aa2eccf9bbae7f333b27860cafbcba09c9b55d75e05e7cf9baff441.jpg)

![a1bf2d6f0ae8a6720cd82785aa60ee50dd84941e4766c57c409bcbdc28cb114a.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/a1bf2d6f0ae8a6720cd82785aa60ee50dd84941e4766c57c409bcbdc28cb114a.jpg)

![dec0ec7f98bcd56149ef99548b318df2cba73c381bf08e1151f5e0d841209a5b.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/images/dec0ec7f98bcd56149ef99548b318df2cba73c381bf08e1151f5e0d841209a5b.jpg)

### Tables

![0b59a4af927755fafa367d458c9b7fe2cd4f997cb5ab06232e51769c11dbb725.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/0b59a4af927755fafa367d458c9b7fe2cd4f997cb5ab06232e51769c11dbb725.jpg)

![0f98270d07e2804d9212fe0ed27f534bff0e784ef632bbdc3d3bc5fd9a7bb78d.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/0f98270d07e2804d9212fe0ed27f534bff0e784ef632bbdc3d3bc5fd9a7bb78d.jpg)

![876e9101bcd3849c449752f709ced72c7cbcfd538324b0bad8cd400dfcc46d05.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/876e9101bcd3849c449752f709ced72c7cbcfd538324b0bad8cd400dfcc46d05.jpg)

![89a634c46120faffd62c60596f62465922d63e250ae7407991f607ad8b56cc07.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/89a634c46120faffd62c60596f62465922d63e250ae7407991f607ad8b56cc07.jpg)

![a14c1d1ad30c2f9c105e6347c7a50b9542fe51f93098741a4a9dfc2cfd28beb0.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/a14c1d1ad30c2f9c105e6347c7a50b9542fe51f93098741a4a9dfc2cfd28beb0.jpg)

![b89d3a181b94658ded66819909f36a584c9981367c7870216c38572f5beba3fd.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/b89d3a181b94658ded66819909f36a584c9981367c7870216c38572f5beba3fd.jpg)

![e517bfaa2ae8893ca1123107d5d176be935c992451d6c1dbd6e95b0eb1b43ed0.jpg](../icml_results/3687_Flexibility-Aware%20Geometric%20Latent%20Diffusion%20for%20Full-Atom%20Peptide%20Design/tables/e517bfaa2ae8893ca1123107d5d176be935c992451d6c1dbd6e95b0eb1b43ed0.jpg)

<a id="paper-3648"></a>

## gp2Scale: A Class of Compactly Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points


### Images

![1321034c76e8f240ce921ed6b79aef25c462692e0f1957858603716fd422b2c1.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/images/1321034c76e8f240ce921ed6b79aef25c462692e0f1957858603716fd422b2c1.jpg)

![610c07c8cff9dcc28a62d7d535a03e3b3743815bcef174269687824e998307fc.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/images/610c07c8cff9dcc28a62d7d535a03e3b3743815bcef174269687824e998307fc.jpg)

![64f26096efcfc0a25a41936c3a78be026b44df276a6707fa6269bf4fe597d6ab.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/images/64f26096efcfc0a25a41936c3a78be026b44df276a6707fa6269bf4fe597d6ab.jpg)

![abbc702ed379ea83772150a71edd7f43fa9897413f7186def792d280440f3339.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/images/abbc702ed379ea83772150a71edd7f43fa9897413f7186def792d280440f3339.jpg)

### Tables

![45795bf77e2b0301d3f6a2cdf8e8afcbde2afa5428f388d9c152dbdfea4fa997.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/45795bf77e2b0301d3f6a2cdf8e8afcbde2afa5428f388d9c152dbdfea4fa997.jpg)

![67935e46baa7dc474888beb0fd84e490626f9a2d8b562aa7268bf6c8a22c19af.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/67935e46baa7dc474888beb0fd84e490626f9a2d8b562aa7268bf6c8a22c19af.jpg)

![6f7adb9aec553858eb7e92549c466e4b91dcb53383bad4383bed7256ef121cfc.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/6f7adb9aec553858eb7e92549c466e4b91dcb53383bad4383bed7256ef121cfc.jpg)

![81fa9291b3f61d7a17dd628f01bde80ddd5171bb9cfcba7a08925cf38a65ea6e.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/81fa9291b3f61d7a17dd628f01bde80ddd5171bb9cfcba7a08925cf38a65ea6e.jpg)

![8e970d0e65abea43aef37720e8e5e6190d6d7e7ce1eabd33278021273f4acf5d.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/8e970d0e65abea43aef37720e8e5e6190d6d7e7ce1eabd33278021273f4acf5d.jpg)

![e0a17c70b3c2602b6a15ee05b8fa6befe6e09c5f494b97849ae2799ae0e9fb47.jpg](../icml_results/3688_gp2Scale_%20A%20Class%20of%20Compactly%20Supported%20Non-Stationary%20Kernels%20and%20Distributed%20Computing%20for%20Exact%20/tables/e0a17c70b3c2602b6a15ee05b8fa6befe6e09c5f494b97849ae2799ae0e9fb47.jpg)

<a id="paper-3649"></a>

## Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning


### Images

![00b8c5c38a9668f5e7a380d8786cfeab7ab3296a4b1779e13d9286ab4d5c0b22.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/00b8c5c38a9668f5e7a380d8786cfeab7ab3296a4b1779e13d9286ab4d5c0b22.jpg)

![2e539833251ce8d949c9b99540aafb842c60baf74d32347024a76ff5cf9506fa.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/2e539833251ce8d949c9b99540aafb842c60baf74d32347024a76ff5cf9506fa.jpg)

![4187b878be64063be1e7d3bad9d470cf02381947ea8c0efd20f06109a1c0ada6.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/4187b878be64063be1e7d3bad9d470cf02381947ea8c0efd20f06109a1c0ada6.jpg)

![4822497cd16e3dfd3237691941fde24042e5064cba7e2ff345b589b832afc536.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/4822497cd16e3dfd3237691941fde24042e5064cba7e2ff345b589b832afc536.jpg)

![54e8b520bb7b79538e2e6fab5436c8f690d735fd57c60eadb264a64ff2fc73bd.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/54e8b520bb7b79538e2e6fab5436c8f690d735fd57c60eadb264a64ff2fc73bd.jpg)

![8feabf2bdf0dd7fdbceba13fc1e0e45bd123473322267fb16eecadc057f72bd9.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/8feabf2bdf0dd7fdbceba13fc1e0e45bd123473322267fb16eecadc057f72bd9.jpg)

![95a714ba36af442dd7372cdabfb9e56bab24952ce12999c1628180ac33a032c0.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/95a714ba36af442dd7372cdabfb9e56bab24952ce12999c1628180ac33a032c0.jpg)

![ac026b13b956e31031d438202e6b19d616eea810f8cada5e0a3802647e2089b9.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/ac026b13b956e31031d438202e6b19d616eea810f8cada5e0a3802647e2089b9.jpg)

![d2e982a5ec964679833b38f906df72d94ef19f4e45ced24e4e6a74f05ac1f03b.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/d2e982a5ec964679833b38f906df72d94ef19f4e45ced24e4e6a74f05ac1f03b.jpg)

![d315d18bf137dc296e7a4fd642d3b30ede11bce1830c661c4080b7d4987a44af.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/d315d18bf137dc296e7a4fd642d3b30ede11bce1830c661c4080b7d4987a44af.jpg)

![d8b62e81d1df81286ccbbb7dab795183c25014549811dcd4446a4c9bbdc29846.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/d8b62e81d1df81286ccbbb7dab795183c25014549811dcd4446a4c9bbdc29846.jpg)

![e44988399675e501d7a30b92b22aa886ffdddd769a0c3edb980fe38c45548f89.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/e44988399675e501d7a30b92b22aa886ffdddd769a0c3edb980fe38c45548f89.jpg)

![fcaadb2daa6efdf6ebc5088e6bcf55b0897bf6e449e67f6a11d385def99e065a.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/images/fcaadb2daa6efdf6ebc5088e6bcf55b0897bf6e449e67f6a11d385def99e065a.jpg)

### Tables

![004523c3abc23ad646be9b82ec86a4915e9844a8572e0773ed43421bbcc234dd.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/004523c3abc23ad646be9b82ec86a4915e9844a8572e0773ed43421bbcc234dd.jpg)

![1bc572e3903dc602627094d430c86b2c45dd20546fbb18a0afc7d30ee537fbcb.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/1bc572e3903dc602627094d430c86b2c45dd20546fbb18a0afc7d30ee537fbcb.jpg)

![1ff3bf95595491b0e30d20ca56a18746ad0872438d98e019f6ebe76e18d41ab8.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/1ff3bf95595491b0e30d20ca56a18746ad0872438d98e019f6ebe76e18d41ab8.jpg)

![30c5c397d0c781dc497689a0c986caf3aaa418a17fa1d57a44d344181e4ce272.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/30c5c397d0c781dc497689a0c986caf3aaa418a17fa1d57a44d344181e4ce272.jpg)

![48854083019d0b53bf9bf16b904fd2a2bed0adc056468ac60db0c13f8aaa4a90.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/48854083019d0b53bf9bf16b904fd2a2bed0adc056468ac60db0c13f8aaa4a90.jpg)

![4ab787d851c670ff56c43c5ed0b5251c2748665567e526c75c41fea5cd3a9117.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/4ab787d851c670ff56c43c5ed0b5251c2748665567e526c75c41fea5cd3a9117.jpg)

![4fee15e95c14121587a3d596290e296c72a2fc595900507c263df13a67c32b75.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/4fee15e95c14121587a3d596290e296c72a2fc595900507c263df13a67c32b75.jpg)

![6318081395760244b39abade5d96790b24fa3970b2869e2382c10735bf40f40c.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/6318081395760244b39abade5d96790b24fa3970b2869e2382c10735bf40f40c.jpg)

![6e0cb51722020c8bddce056e187f24f03509e8b9c7d1e53be8881f212a469a41.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/6e0cb51722020c8bddce056e187f24f03509e8b9c7d1e53be8881f212a469a41.jpg)

![7355ffd4bc265fa8def3e2412f00cb4007ce639e206c78717fa0d9b7433e1340.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/7355ffd4bc265fa8def3e2412f00cb4007ce639e206c78717fa0d9b7433e1340.jpg)

![7a9581f2b3189f70d93f4b4bb1f433d2a5fe7f985ddc6549451fd370bd326003.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/7a9581f2b3189f70d93f4b4bb1f433d2a5fe7f985ddc6549451fd370bd326003.jpg)

![7c7755dc1c9d790ae3a6fa98780bb05e9fa62d88e49c1816a05afc89ce382159.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/7c7755dc1c9d790ae3a6fa98780bb05e9fa62d88e49c1816a05afc89ce382159.jpg)

![86f27e6e2849716c6e607aa1da467abfed0cdb3a9b5f022e473e5d618f67f78e.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/86f27e6e2849716c6e607aa1da467abfed0cdb3a9b5f022e473e5d618f67f78e.jpg)

![99e246917d244707e9d310d6d91a923d799a4a9b556cbfe23c6403e9a5b9dcc2.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/99e246917d244707e9d310d6d91a923d799a4a9b556cbfe23c6403e9a5b9dcc2.jpg)

![9b2513fa2b5df498a490d7c81c290300682e954c2efdf85916c4f33b48c3d408.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/9b2513fa2b5df498a490d7c81c290300682e954c2efdf85916c4f33b48c3d408.jpg)

![a10e41387bff704eda1a3139a9319c5d234150c626efe6f064f46def0377efcd.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/a10e41387bff704eda1a3139a9319c5d234150c626efe6f064f46def0377efcd.jpg)

![bc0fa96118119cb5e1e1d95b7ef179ffb221765f371d5392627b92cb932a81b3.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/bc0fa96118119cb5e1e1d95b7ef179ffb221765f371d5392627b92cb932a81b3.jpg)

![c1e5bb5375e512025ba1b4b2f499347c8d973bdecb72defff9c5b0961a88007d.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/c1e5bb5375e512025ba1b4b2f499347c8d973bdecb72defff9c5b0961a88007d.jpg)

![c4af25bce93fcb204c7b44e58875b610d3b90f32055f0c8df3351326e327491c.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/c4af25bce93fcb204c7b44e58875b610d3b90f32055f0c8df3351326e327491c.jpg)

![c555f04ce9fc6a301099913284058deeb4cf0b54785cbe128aef4b1ca5d39ef3.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/c555f04ce9fc6a301099913284058deeb4cf0b54785cbe128aef4b1ca5d39ef3.jpg)

![c557cd469eee4776fd9ea99f8340577e42b4ded48909975b604857ae2b49cd4b.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/c557cd469eee4776fd9ea99f8340577e42b4ded48909975b604857ae2b49cd4b.jpg)

![cc95a38ff7e70c9df4a7e515890337cf2b2e81f38cd854b94f78e80aad4ffb04.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/cc95a38ff7e70c9df4a7e515890337cf2b2e81f38cd854b94f78e80aad4ffb04.jpg)

![cd8f4122a3ce64b6197e65dc68612c4d9ac119698eba4da86fc7707dc5f5221e.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/cd8f4122a3ce64b6197e65dc68612c4d9ac119698eba4da86fc7707dc5f5221e.jpg)

![d7fb6dd3de9eb663ef6ed0818f64029abac4c13cf3b5b4dea559727d9d06db2e.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/d7fb6dd3de9eb663ef6ed0818f64029abac4c13cf3b5b4dea559727d9d06db2e.jpg)

![dc7c371d222573b2cc6573ef06d9def64ec2caccbd1a59b5e7d364045e15b851.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/dc7c371d222573b2cc6573ef06d9def64ec2caccbd1a59b5e7d364045e15b851.jpg)

![dc933731b67b4822dc50dd13a7962fba1c2de7127bb5e4fe104b287a9cff825e.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/dc933731b67b4822dc50dd13a7962fba1c2de7127bb5e4fe104b287a9cff825e.jpg)

![e0bb9bdab4cd031b6953e033b7aa8a5123ca6f5e741f34a003e03030e17a4aed.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/e0bb9bdab4cd031b6953e033b7aa8a5123ca6f5e741f34a003e03030e17a4aed.jpg)

![e3368a2cf82acd19e8178e9b38df4e016016b61b76c8f5ce4a29a3b2c1e454ed.jpg](../icml_results/3689_Geometry%20of%20Reason_%20Spectral%20Signatures%20of%20Valid%20Mathematical%20Reasoning/tables/e3368a2cf82acd19e8178e9b38df4e016016b61b76c8f5ce4a29a3b2c1e454ed.jpg)

<a id="paper-3650"></a>

## Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics


### Images

![12085a9b4dce6725f4d1f616b82c6d2caf319b73169770b00806b30f72dc02d6.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/12085a9b4dce6725f4d1f616b82c6d2caf319b73169770b00806b30f72dc02d6.jpg)

![4ea41f0759c6d3954cddf7b22b67873535bbfd2cea426614c209d4ab5162d01e.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/4ea41f0759c6d3954cddf7b22b67873535bbfd2cea426614c209d4ab5162d01e.jpg)

![6dda77e3b13b5b31247d59ae93edbacf77e63d84d60414be3c979e57f162a30c.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/6dda77e3b13b5b31247d59ae93edbacf77e63d84d60414be3c979e57f162a30c.jpg)

![7cb42294d610b6042a73b6649a08baaa5447e4d42ba52ee7dfcaf7faa3c84b12.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/7cb42294d610b6042a73b6649a08baaa5447e4d42ba52ee7dfcaf7faa3c84b12.jpg)

![c25300d87ef450d5a0406a5bacd5438a3c0322cc77487d44d162b4bfa62c9976.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/c25300d87ef450d5a0406a5bacd5438a3c0322cc77487d44d162b4bfa62c9976.jpg)

![da21e946fc6efe7ec4c8fb53d88d03eb305198d78532ac210d91dffddd852095.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/da21e946fc6efe7ec4c8fb53d88d03eb305198d78532ac210d91dffddd852095.jpg)

![dc0d45379310e669814b56f6eb622274f129773b6eb6d93e5fe0b27db5b56606.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/images/dc0d45379310e669814b56f6eb622274f129773b6eb6d93e5fe0b27db5b56606.jpg)

### Tables

![246ed7cb26673f5b99bdae0a14355d374b15f7fb6b95901b63a6be4e0bb9e0ee.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/246ed7cb26673f5b99bdae0a14355d374b15f7fb6b95901b63a6be4e0bb9e0ee.jpg)

![2496bde4d6bcb6be01fc1517f5f77b172160c97dec363a8e81e75f120551795d.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/2496bde4d6bcb6be01fc1517f5f77b172160c97dec363a8e81e75f120551795d.jpg)

![35a958673f33b0e75aa18398f41ec387adfccbc330d4f17f2cb5430f8a11ba37.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/35a958673f33b0e75aa18398f41ec387adfccbc330d4f17f2cb5430f8a11ba37.jpg)

![35ca828116627199d351431d34d59204ce4c2f598600ab584b2371c7fb38faea.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/35ca828116627199d351431d34d59204ce4c2f598600ab584b2371c7fb38faea.jpg)

![3a753b1ba0c8306561efacc6f70e28a3d6d9b6b637d83f0b54f5ac38db5bd8ca.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/3a753b1ba0c8306561efacc6f70e28a3d6d9b6b637d83f0b54f5ac38db5bd8ca.jpg)

![523cf820f683b2451cc1be7e470cb2b40b4a25f3e195be455b2327171bba0c3a.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/523cf820f683b2451cc1be7e470cb2b40b4a25f3e195be455b2327171bba0c3a.jpg)

![56edaa670b34f6b48b92addb16a4169103ae6939d5b75f763471050aec06540a.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/56edaa670b34f6b48b92addb16a4169103ae6939d5b75f763471050aec06540a.jpg)

![573db5462fe782ad0d0bc9ed3808ec11de55ed83e57751d7bbdeb1460b592b6b.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/573db5462fe782ad0d0bc9ed3808ec11de55ed83e57751d7bbdeb1460b592b6b.jpg)

![72eec4630e67efd2dcec1587891563ed9b59498620357aeddbfe30b1ad3c6af0.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/72eec4630e67efd2dcec1587891563ed9b59498620357aeddbfe30b1ad3c6af0.jpg)

![7c9b28d9519197e29bc264e7bb2a95ca4761101a471428ee5baf90863dc853f1.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/7c9b28d9519197e29bc264e7bb2a95ca4761101a471428ee5baf90863dc853f1.jpg)

![884b7c21bcec25f3d030c9a71cdc71eb5d6e509038f05a002ec1c1e9fa2af724.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/884b7c21bcec25f3d030c9a71cdc71eb5d6e509038f05a002ec1c1e9fa2af724.jpg)

![8d275662e2a1dfd99940e7a9a471e618950565b6799dc5073201b25531f5c29a.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/8d275662e2a1dfd99940e7a9a471e618950565b6799dc5073201b25531f5c29a.jpg)

![9f9618be7518dde1444ff8e7c22b3e6ff562a734553f384e2362824dc88f567e.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/9f9618be7518dde1444ff8e7c22b3e6ff562a734553f384e2362824dc88f567e.jpg)

![c1a95aa332db58417c28fa66acbe34f94586fc63e7b866f07bd855bd03b7bf4f.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/c1a95aa332db58417c28fa66acbe34f94586fc63e7b866f07bd855bd03b7bf4f.jpg)

![c7934f41a94dc8b1c8618fe4f148900a5e46e783b03ca99331ad14a0f7d31087.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/c7934f41a94dc8b1c8618fe4f148900a5e46e783b03ca99331ad14a0f7d31087.jpg)

![e850925ed8e218e1cc9f09f61d8586a3b0a430f1fb0ba2d9d749c13c5e31160d.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/e850925ed8e218e1cc9f09f61d8586a3b0a430f1fb0ba2d9d749c13c5e31160d.jpg)

![fa545dae69896c9bc9bf0d0dcec0bb4c3e9e263c5f7c5d3b46e7090e2e74ed75.jpg](../icml_results/3690_Mesh%20Field%20Theory_%20Port%E2%80%93Hamiltonian%20Formulation%20of%20Mesh-Based%20Physics/tables/fa545dae69896c9bc9bf0d0dcec0bb4c3e9e263c5f7c5d3b46e7090e2e74ed75.jpg)

<a id="paper-3651"></a>

## QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs


### Images

![08bbc0ae9de1a3079e210b2895ef6115692d4751fc99f10417f1399e588167aa.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/08bbc0ae9de1a3079e210b2895ef6115692d4751fc99f10417f1399e588167aa.jpg)

![2f6472e6623ff0ae7bc49da52dc5ba78d3cc2ac5879a198da434dc7496c7c92c.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/2f6472e6623ff0ae7bc49da52dc5ba78d3cc2ac5879a198da434dc7496c7c92c.jpg)

![559679a341acd60d22ce9e54f64f2a4ec03e2fe29e593e446d29924a243e0271.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/559679a341acd60d22ce9e54f64f2a4ec03e2fe29e593e446d29924a243e0271.jpg)

![7f4fc61466be0fe40c8eb5141f820373fe19d809ba8d0cd79571ecd82710df90.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/7f4fc61466be0fe40c8eb5141f820373fe19d809ba8d0cd79571ecd82710df90.jpg)

![92a5fef9e5f7c1438f853044532f3a293d543725ed145f67ace3585fcb3b57ab.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/92a5fef9e5f7c1438f853044532f3a293d543725ed145f67ace3585fcb3b57ab.jpg)

![b9c3d7dc7f81e2c25d41b4bc6687f7da99af7664758f9e3bf310f13f54b5f5fb.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/b9c3d7dc7f81e2c25d41b4bc6687f7da99af7664758f9e3bf310f13f54b5f5fb.jpg)

![d1305923514c802639df25647678e69419347daaaabed7ae974dd8eda88d7840.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/d1305923514c802639df25647678e69419347daaaabed7ae974dd8eda88d7840.jpg)

![d1865f7f6fff9ce14b35a0dc113b9e630a4c7ef9bfd8b22064df3948783ff770.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/d1865f7f6fff9ce14b35a0dc113b9e630a4c7ef9bfd8b22064df3948783ff770.jpg)

![e9e67c4dfcbb2072b98577f8de939fc78c0ee01f34c8acd5bc17bc3c1a9922f3.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/e9e67c4dfcbb2072b98577f8de939fc78c0ee01f34c8acd5bc17bc3c1a9922f3.jpg)

![ec6c04f8908c6c9003380f0fae82203860e8ce167914304e7f002a9904d45306.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/images/ec6c04f8908c6c9003380f0fae82203860e8ce167914304e7f002a9904d45306.jpg)

### Tables

![0bde1dadac6691f664088a8e43ed335a91e11b519f59b505031e0a9a69a96ee2.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/0bde1dadac6691f664088a8e43ed335a91e11b519f59b505031e0a9a69a96ee2.jpg)

![1768ecd89f971b3fb5dfaa99a8efa3498a37391628c807f183e340f6f5c62781.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/1768ecd89f971b3fb5dfaa99a8efa3498a37391628c807f183e340f6f5c62781.jpg)

![20b12bad158d36444cb3712eedde7c816ef8bc36cbd48a20efabf837dd1f9251.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/20b12bad158d36444cb3712eedde7c816ef8bc36cbd48a20efabf837dd1f9251.jpg)

![3c13cff5735ea5c1150f7d67eb05627f07d786d79fa78e692540e827869b02d1.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/3c13cff5735ea5c1150f7d67eb05627f07d786d79fa78e692540e827869b02d1.jpg)

![3db18b2758ab8bb0052901cabf8b45d40ed7140c082acaede951846b988f8803.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/3db18b2758ab8bb0052901cabf8b45d40ed7140c082acaede951846b988f8803.jpg)

![49f63bf3df3d7e83423583aee6228ef4d81288a21cb0c5b584847ce709e2b9f6.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/49f63bf3df3d7e83423583aee6228ef4d81288a21cb0c5b584847ce709e2b9f6.jpg)

![5500836bbb9c26a7de01c3275f89a86355d9a8a32812cd7e52629c5a313e463b.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/5500836bbb9c26a7de01c3275f89a86355d9a8a32812cd7e52629c5a313e463b.jpg)

![5b405df9aab8798072e7a503ee291f85027588e04b6ef854fd9c83b7b6ff0c50.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/5b405df9aab8798072e7a503ee291f85027588e04b6ef854fd9c83b7b6ff0c50.jpg)

![645daa46d0917b62962706c853ad8c95ffb720330f2a9542a87fd70fac0a62a0.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/645daa46d0917b62962706c853ad8c95ffb720330f2a9542a87fd70fac0a62a0.jpg)

![71873e7b51bd482bb976d84d8c0aedcebe690e8f2e370d22adea6bab37eebebc.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/71873e7b51bd482bb976d84d8c0aedcebe690e8f2e370d22adea6bab37eebebc.jpg)

![9623fed7bfc9f435d81fb7f548394bea523a1a85d2143020b2988c067df0024a.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/9623fed7bfc9f435d81fb7f548394bea523a1a85d2143020b2988c067df0024a.jpg)

![96d23cf4fe4ebe755b6da49cf750bead88133c30a3a7a86ebb9ef12d9a054bfa.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/96d23cf4fe4ebe755b6da49cf750bead88133c30a3a7a86ebb9ef12d9a054bfa.jpg)

![9afe7071a2bcbdb4b14ecab72b39a5ec31b8c9293dc7e43768651e881f136a3d.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/9afe7071a2bcbdb4b14ecab72b39a5ec31b8c9293dc7e43768651e881f136a3d.jpg)

![9d5174bc04a06690a220b826492a2cc6e58f2794bcd5530c834a3afed3fece4a.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/9d5174bc04a06690a220b826492a2cc6e58f2794bcd5530c834a3afed3fece4a.jpg)

![b1d1f64f6120cf5052e82c1a8a450184ec4e35c8e2e5e1cef4e1de1666dc5427.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/b1d1f64f6120cf5052e82c1a8a450184ec4e35c8e2e5e1cef4e1de1666dc5427.jpg)

![c7fb87a5fa37961dc22f6b647f55aad7f56d47ba10b20abc7214739af55892a2.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/c7fb87a5fa37961dc22f6b647f55aad7f56d47ba10b20abc7214739af55892a2.jpg)

![ca26f9641e0dd00c441b0da4c346df74eb9757ec1063dd4b4047da13b73aa9c6.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/ca26f9641e0dd00c441b0da4c346df74eb9757ec1063dd4b4047da13b73aa9c6.jpg)

![dabca0f778e4e417da84ebd3cc9a70f089daf1812763e348221cad10906f8a19.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/dabca0f778e4e417da84ebd3cc9a70f089daf1812763e348221cad10906f8a19.jpg)

![dca858adf2150f8762da6eff551fe41e5f7b1f186207910ad87e49897a1f0435.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/dca858adf2150f8762da6eff551fe41e5f7b1f186207910ad87e49897a1f0435.jpg)

![ec8d56685b6c33b64af8e7af12857974935d35f731acd04f621149991f963661.jpg](../icml_results/3691_QTALE_%20Quantization-Robust%20Token-Adaptive%20Layer%20Execution%20for%20LLMs/tables/ec8d56685b6c33b64af8e7af12857974935d35f731acd04f621149991f963661.jpg)

<a id="paper-3652"></a>

## Multi-Round Human–AI Collaboration with User-Specified Requirements


### Images

![1043a9bf43af94c64c3c59dcda42c903537b440bb22cba233342217e0a3705b4.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/1043a9bf43af94c64c3c59dcda42c903537b440bb22cba233342217e0a3705b4.jpg)

![24cc4815437e30c800f44a28d9fec79f46c12460fa3bd8bc4bf65591ad35881f.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/24cc4815437e30c800f44a28d9fec79f46c12460fa3bd8bc4bf65591ad35881f.jpg)

![390d6cb8a82e5e4873dc2fdfd40994b75588085283d75d757c88fe64ae41e36f.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/390d6cb8a82e5e4873dc2fdfd40994b75588085283d75d757c88fe64ae41e36f.jpg)

![4caaafc64b126d91b76333555461e97981e5e59867340a339f28865cbb8ee78f.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/4caaafc64b126d91b76333555461e97981e5e59867340a339f28865cbb8ee78f.jpg)

![5d62bb40672389aed3506ee24231b24b152924c721c484761f749a85f598bbe0.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/5d62bb40672389aed3506ee24231b24b152924c721c484761f749a85f598bbe0.jpg)

![946166cf67f888964de444e6bb7f9fcc7b6335c16cd07daa3deab6f9bfb1b4b4.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/946166cf67f888964de444e6bb7f9fcc7b6335c16cd07daa3deab6f9bfb1b4b4.jpg)

![94d1ee3f68fec3bb6d3f4c663246c8b8fab4f182fd43c5de78e402872216aa00.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/94d1ee3f68fec3bb6d3f4c663246c8b8fab4f182fd43c5de78e402872216aa00.jpg)

![bde901eb8f4377af9e30106fbb4a56dcbeb1fa63d70d6292243428b085f706f0.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/bde901eb8f4377af9e30106fbb4a56dcbeb1fa63d70d6292243428b085f706f0.jpg)

![c1667a94586b2d2fc1aaadf3cf78c5ef34280b9fbcb265d93763f9b86475714d.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/c1667a94586b2d2fc1aaadf3cf78c5ef34280b9fbcb265d93763f9b86475714d.jpg)

![ce707746d24dceeb3e622b0ce523ce508742bffa673fd7aa555f3f4f2297f6fe.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/ce707746d24dceeb3e622b0ce523ce508742bffa673fd7aa555f3f4f2297f6fe.jpg)

![d1377f6393c223144679aec5a1b206c4d2647bba9c02fd0f33bed01fed56bc99.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/d1377f6393c223144679aec5a1b206c4d2647bba9c02fd0f33bed01fed56bc99.jpg)

![e6f654abacf81acd6cb0803845cc9e4fbecfb6b2f7df9823fe539d2c90f867d3.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/e6f654abacf81acd6cb0803845cc9e4fbecfb6b2f7df9823fe539d2c90f867d3.jpg)

![eb4fb56b027713ae1bd98ec3da56d76756276b3b6443a3dfcea7fe9d9cde2b2a.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/eb4fb56b027713ae1bd98ec3da56d76756276b3b6443a3dfcea7fe9d9cde2b2a.jpg)

![ffb714ab89987b4d0c295f40b1fc5d869e8eb626e599418b4f240f37f939e9a1.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/images/ffb714ab89987b4d0c295f40b1fc5d869e8eb626e599418b4f240f37f939e9a1.jpg)

### Tables

![5d10bae77d7d089cda78f522582798d30f624281b1eb54d4a9dd48412554151b.jpg](../icml_results/3692_Multi-Round%20Human%E2%80%93AI%20Collaboration%20with%20User-Specified%20Requirements/tables/5d10bae77d7d089cda78f522582798d30f624281b1eb54d4a9dd48412554151b.jpg)

<a id="paper-3653"></a>

## Human-AI Collaborative Uncertainty Quantification


### Images

![159571fa8324614d6c85ce3f559ed0459573f13762f28ab8496eb2a5e1931901.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/159571fa8324614d6c85ce3f559ed0459573f13762f28ab8496eb2a5e1931901.jpg)

![172af7b29da149c21483791742c49d10fd7b7a1c02ace0a349033ad80c1d6dd4.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/172af7b29da149c21483791742c49d10fd7b7a1c02ace0a349033ad80c1d6dd4.jpg)

![695d487bee17aacc4130bd2d0b62989d41b86d85758cc669b28a15b9f5e86b0e.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/695d487bee17aacc4130bd2d0b62989d41b86d85758cc669b28a15b9f5e86b0e.jpg)

![8dd947ca719ba393751f0901bb15b07c119acc9e5d0512889e4ff480cc4d4d0b.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/8dd947ca719ba393751f0901bb15b07c119acc9e5d0512889e4ff480cc4d4d0b.jpg)

![9696a12d1bcd5d80a564fc26750c7cf87c3c4dd565ca329f5699bf7183e6da0c.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/9696a12d1bcd5d80a564fc26750c7cf87c3c4dd565ca329f5699bf7183e6da0c.jpg)

![a83e7051c44eff7c18bd094d66e47d32b0e8f806d2029cb63ca0e39c1b968c04.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/a83e7051c44eff7c18bd094d66e47d32b0e8f806d2029cb63ca0e39c1b968c04.jpg)

![b4091b2f402ac54dec0ea370bb4ab5d5c7e8cecf52cfebe728541e2ef5dd5c12.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/b4091b2f402ac54dec0ea370bb4ab5d5c7e8cecf52cfebe728541e2ef5dd5c12.jpg)

![f79d565235339e84b9e71fb57eb7b0d7d7db436d6e53d2a6009790abdbd96fa4.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/f79d565235339e84b9e71fb57eb7b0d7d7db436d6e53d2a6009790abdbd96fa4.jpg)

![fb9645ffaaac8b1d331c3d7a842fe5f58b3ccb948e69443ddc956c259d4ea5fd.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/images/fb9645ffaaac8b1d331c3d7a842fe5f58b3ccb948e69443ddc956c259d4ea5fd.jpg)

### Tables

![1ad9c7cd693ac626a891d4277b4f09f73d45a829c86a7d2b5999de0782e7f5cb.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/tables/1ad9c7cd693ac626a891d4277b4f09f73d45a829c86a7d2b5999de0782e7f5cb.jpg)

![3340a5635299aeff7903e0a5aecc4566abe9ec1da5a3e4b53d17be4cf1bccee7.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/tables/3340a5635299aeff7903e0a5aecc4566abe9ec1da5a3e4b53d17be4cf1bccee7.jpg)

![39df3c9db848bef45dc013d14661d5bbf711f9310f728d4be79bd905d0784dd4.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/tables/39df3c9db848bef45dc013d14661d5bbf711f9310f728d4be79bd905d0784dd4.jpg)

![87a8cd6393dd88ecdb8661a9518c153e6f2652bbed85b4f05053607577a63777.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/tables/87a8cd6393dd88ecdb8661a9518c153e6f2652bbed85b4f05053607577a63777.jpg)

![99f7ded1a34f24657f39adff13f38b9e0f31aeca0c0928a0d0ec84c50b45fa5d.jpg](../icml_results/3693_Human-AI%20Collaborative%20Uncertainty%20Quantification/tables/99f7ded1a34f24657f39adff13f38b9e0f31aeca0c0928a0d0ec84c50b45fa5d.jpg)

<a id="paper-3654"></a>

## Deep sequence models tend to memorize geometrically; it is unclear why


### Images

![03cc1087cf6c6c338f9eca6cdf759f12a28fa76049300540ded177fb5c0ad453.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/03cc1087cf6c6c338f9eca6cdf759f12a28fa76049300540ded177fb5c0ad453.jpg)

![12b6fd4031de8cb233d79fae771d515b321a00fd1888a2a52efb5efa7d812b24.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/12b6fd4031de8cb233d79fae771d515b321a00fd1888a2a52efb5efa7d812b24.jpg)

![15d778e0ffd98b53f759522bc6e13e0fd6ecd747fe5a5f0f3c8b2fb87e0c173b.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/15d778e0ffd98b53f759522bc6e13e0fd6ecd747fe5a5f0f3c8b2fb87e0c173b.jpg)

![20f66e5776235813136da667ef3171afe723c2d1298b3eb927ab09a78f219768.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/20f66e5776235813136da667ef3171afe723c2d1298b3eb927ab09a78f219768.jpg)

![32ea8d6583a8cdff4fdc063a0cbee81d47dbceba15a86ee7421b7f4783cb1823.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/32ea8d6583a8cdff4fdc063a0cbee81d47dbceba15a86ee7421b7f4783cb1823.jpg)

![34cc0439d1a5f012ebe13f651e0f0ce1ded994c85d51ac96e7c082c97b974e13.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/34cc0439d1a5f012ebe13f651e0f0ce1ded994c85d51ac96e7c082c97b974e13.jpg)

![3535765472ffec52cced301857f14d9eb9771015cbcb1e1907f83d253989649d.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/3535765472ffec52cced301857f14d9eb9771015cbcb1e1907f83d253989649d.jpg)

![3a20d63c8177cafb5c5dd785f14568244177abe05202ac5dd22e29cce96e9f13.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/3a20d63c8177cafb5c5dd785f14568244177abe05202ac5dd22e29cce96e9f13.jpg)

![3fb5eaf3c83f745ff32aa9c78aee5d3a005852ee1faa97b71ca844c6d8908c99.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/3fb5eaf3c83f745ff32aa9c78aee5d3a005852ee1faa97b71ca844c6d8908c99.jpg)

![404b8b7817572aa96e31844e0f93190963ed9293ff095f06ecb59c4776134514.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/404b8b7817572aa96e31844e0f93190963ed9293ff095f06ecb59c4776134514.jpg)

![43d9ea3fa4056523033602f78feff96ff7e48fd5ebada7777fa4cb6d8def00ce.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/43d9ea3fa4056523033602f78feff96ff7e48fd5ebada7777fa4cb6d8def00ce.jpg)

![43e4cc9680da7d8934694cea8925bf3bffe28bb2bdbb79c817cf4f7f72e7d7bb.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/43e4cc9680da7d8934694cea8925bf3bffe28bb2bdbb79c817cf4f7f72e7d7bb.jpg)

![4d1caa4d6d7f5b3b78e80844801fb967ce830d79243a802aee9a221103ec8ae6.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/4d1caa4d6d7f5b3b78e80844801fb967ce830d79243a802aee9a221103ec8ae6.jpg)

![4ecb41ee88767abbe1fd4670243bb9c39c4eff79015cb3c0eb7c29d6db7f7dbb.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/4ecb41ee88767abbe1fd4670243bb9c39c4eff79015cb3c0eb7c29d6db7f7dbb.jpg)

![586976e79b2ded0daa43acf99e0c2b68a6887afb5b90f58f0ed9910ede7d499b.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/586976e79b2ded0daa43acf99e0c2b68a6887afb5b90f58f0ed9910ede7d499b.jpg)

![5f6408e3ea3d9395a9eaf384ca908ff8f9ef8babb461e670ae449280c7dc1499.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/5f6408e3ea3d9395a9eaf384ca908ff8f9ef8babb461e670ae449280c7dc1499.jpg)

![61fd25c9a3c855b5ef1052182acef4b1415131185d1910e7946c964b3e6cbe8f.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/61fd25c9a3c855b5ef1052182acef4b1415131185d1910e7946c964b3e6cbe8f.jpg)

![72d0e07234e85192d1d9a57ef792f567f1ff200d69ec675ffdd89575d20ec624.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/72d0e07234e85192d1d9a57ef792f567f1ff200d69ec675ffdd89575d20ec624.jpg)

![77ce02e04a1bd91ab5926ccff5718ee45325506312d8262dbece12ad413580d0.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/77ce02e04a1bd91ab5926ccff5718ee45325506312d8262dbece12ad413580d0.jpg)

![7b8636d9977e1d5791b3c39a9c548106bc66bc15f9cd3df45946e09692331cf0.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/7b8636d9977e1d5791b3c39a9c548106bc66bc15f9cd3df45946e09692331cf0.jpg)

![8391675e9a9c80650f5d88cea1741122a94f51db8a229d7848e6352bd99fcd5b.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/8391675e9a9c80650f5d88cea1741122a94f51db8a229d7848e6352bd99fcd5b.jpg)

![87afe398ad0b297bf2a02fdebd78d543cfe77772568e30619aa15a7cca7ccc66.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/87afe398ad0b297bf2a02fdebd78d543cfe77772568e30619aa15a7cca7ccc66.jpg)

![89a1bd16eaaf6621182b695eccab99c2e400f676f8cbe890b7b525d5a733a8d1.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/89a1bd16eaaf6621182b695eccab99c2e400f676f8cbe890b7b525d5a733a8d1.jpg)

![8c58d6feaef970131627b74ebafb6a460f24ccf359a1931ac7b3b71f7b888681.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/8c58d6feaef970131627b74ebafb6a460f24ccf359a1931ac7b3b71f7b888681.jpg)

![941ed6ca8537337cb5a37f5f7685aaaadd104ea469a2c25660f954e14121b2c2.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/941ed6ca8537337cb5a37f5f7685aaaadd104ea469a2c25660f954e14121b2c2.jpg)

![94df7d2d05d7b44fe59740455b0791d3bcc2639dca796c3fbd79081d425a5362.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/94df7d2d05d7b44fe59740455b0791d3bcc2639dca796c3fbd79081d425a5362.jpg)

![96fed056f6267be2034dbcf413323eedfbf9cf2c5b3f3eaea29781b4eb90df37.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/96fed056f6267be2034dbcf413323eedfbf9cf2c5b3f3eaea29781b4eb90df37.jpg)

![9f633effbb6b38a72be2f8dfe9ecc020307599c027b2d0e10a79206b01faf657.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/9f633effbb6b38a72be2f8dfe9ecc020307599c027b2d0e10a79206b01faf657.jpg)

![a668395d063234debf13bb93598793992736078e2bdb4f473c6573e45958777a.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/a668395d063234debf13bb93598793992736078e2bdb4f473c6573e45958777a.jpg)

![b03260a8236db0859e488bec189264697688d04fc4631ccccebecaa747b91bee.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/b03260a8236db0859e488bec189264697688d04fc4631ccccebecaa747b91bee.jpg)

![b21d0651c27cccd04d11787698b80c20c76a8abd3920aa954c0cfc688a2be899.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/b21d0651c27cccd04d11787698b80c20c76a8abd3920aa954c0cfc688a2be899.jpg)

![bc7c408e5234acb9be2bf77230b6bb3d84130f424d6e1de96b91a48bb3d1b925.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/bc7c408e5234acb9be2bf77230b6bb3d84130f424d6e1de96b91a48bb3d1b925.jpg)

![c157a9a143eea264af5c93238043281a59b574e5df59d083b164e87480dd2072.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/c157a9a143eea264af5c93238043281a59b574e5df59d083b164e87480dd2072.jpg)

![c2f48b5c3c9aeec69d60f302017d4223e20cf221da63c768d7f8975522344a36.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/c2f48b5c3c9aeec69d60f302017d4223e20cf221da63c768d7f8975522344a36.jpg)

![d219492ad70414b137307752d936ad06c355a5e0d4e2c11a2104f980bf998ffd.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d219492ad70414b137307752d936ad06c355a5e0d4e2c11a2104f980bf998ffd.jpg)

![d2251acf8038b4e9787a2623e9ea29c54b0caed2aac99d9c4f30920b7a9ea911.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d2251acf8038b4e9787a2623e9ea29c54b0caed2aac99d9c4f30920b7a9ea911.jpg)

![d37adcd2aa4d9d84f1f6a7a489a0de051ce42e0ca12cae9b5e4f0522c6f26a31.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d37adcd2aa4d9d84f1f6a7a489a0de051ce42e0ca12cae9b5e4f0522c6f26a31.jpg)

![d40235eb2f414f631380facab72775eb3faaaac4f7cbe0eaf3ffe5e5bf082c8a.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d40235eb2f414f631380facab72775eb3faaaac4f7cbe0eaf3ffe5e5bf082c8a.jpg)

![d436ca5ee59dec72355bac87b684b285d942575556cab3f41a66a709a0ba7955.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d436ca5ee59dec72355bac87b684b285d942575556cab3f41a66a709a0ba7955.jpg)

![d4508495f80b1751499cb2eafc2b6672446595fef267a9afd1ecc95928cfffc6.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/d4508495f80b1751499cb2eafc2b6672446595fef267a9afd1ecc95928cfffc6.jpg)

![dfa8ecef301093510b9bb7539180d17fe383e8d1ed0cc3a5edd62ea0c0d1b202.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/dfa8ecef301093510b9bb7539180d17fe383e8d1ed0cc3a5edd62ea0c0d1b202.jpg)

![ecc6765953fe1e1333636dd022b816e49e5c097b525e4008d2edcbadad8414dd.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/images/ecc6765953fe1e1333636dd022b816e49e5c097b525e4008d2edcbadad8414dd.jpg)

### Tables

![da6e176f20b677af23536bd94d5eadbf23505c5f90fcebde413ab4217cfb5bf7.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/tables/da6e176f20b677af23536bd94d5eadbf23505c5f90fcebde413ab4217cfb5bf7.jpg)

![e52d679f4422c38fdc039d8989284e5575c28b438dde8906fb3255bd215169a1.jpg](../icml_results/3694_Deep%20sequence%20models%20tend%20to%20memorize%20geometrically%3B%20it%20is%20unclear%20why/tables/e52d679f4422c38fdc039d8989284e5575c28b438dde8906fb3255bd215169a1.jpg)

<a id="paper-3655"></a>

## Resilient Coresets and Clustering


### Images

![1de38531705c89a5eec49f312124ddf8618ed1cad3aa33c0ddd5937e502c9878.jpg](../icml_results/3695_Resilient%20Coresets%20and%20Clustering/images/1de38531705c89a5eec49f312124ddf8618ed1cad3aa33c0ddd5937e502c9878.jpg)

![22adb2e1ea2fca292433e8adabeb5897012d2f7c412ff3c287e84959eb14da0f.jpg](../icml_results/3695_Resilient%20Coresets%20and%20Clustering/images/22adb2e1ea2fca292433e8adabeb5897012d2f7c412ff3c287e84959eb14da0f.jpg)

<a id="paper-3656"></a>

## MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation


### Images

![006d54a6ce840f2ef33f05a362830a5b0536d1a7bf88e132731cc90ac366b22a.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/006d54a6ce840f2ef33f05a362830a5b0536d1a7bf88e132731cc90ac366b22a.jpg)

![0846616b90a126260beaa8ce3b3c3d95927cf99a6a00691a0056ade47e213b45.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/0846616b90a126260beaa8ce3b3c3d95927cf99a6a00691a0056ade47e213b45.jpg)

![0a0e53ee86f32e9e54f7c5b4ce7aa1828bda1531bb4a6cdabd5239f075231ddb.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/0a0e53ee86f32e9e54f7c5b4ce7aa1828bda1531bb4a6cdabd5239f075231ddb.jpg)

![0e9004eda4048321efd8b58ed745adc94b46ea0447fc111ce02aaff077225cd9.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/0e9004eda4048321efd8b58ed745adc94b46ea0447fc111ce02aaff077225cd9.jpg)

![129da10a639d74ad4a122501b6c1e3eab25e957f765666cad01777f20a5e5d19.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/129da10a639d74ad4a122501b6c1e3eab25e957f765666cad01777f20a5e5d19.jpg)

![2b459dad93c09781381f4b9e70da1d5d970a2d193b6e0da5014ef47a40b138e5.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/2b459dad93c09781381f4b9e70da1d5d970a2d193b6e0da5014ef47a40b138e5.jpg)

![32f36b529215d3f0f159fd18ed3cd095719be7fbbda9d64d065e06b5e9946656.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/32f36b529215d3f0f159fd18ed3cd095719be7fbbda9d64d065e06b5e9946656.jpg)

![34877547c6724913779396021dc6cefd65a46367c33e995296b36d2501b977b6.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/34877547c6724913779396021dc6cefd65a46367c33e995296b36d2501b977b6.jpg)

![37abf6cbd2060b3d8569ce9d2e2e7da55dbdba0a5d8da0c4fa3016770541395f.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/37abf6cbd2060b3d8569ce9d2e2e7da55dbdba0a5d8da0c4fa3016770541395f.jpg)

![3ab021dcb07bc9482cef21b601311d316d0767c844c216e4a547dc71ec10bee1.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/3ab021dcb07bc9482cef21b601311d316d0767c844c216e4a547dc71ec10bee1.jpg)

![3c8f3f056d4104232dbf896f010f43bcdc19e9e838ab9fd4d5cc11ea9909cbae.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/3c8f3f056d4104232dbf896f010f43bcdc19e9e838ab9fd4d5cc11ea9909cbae.jpg)

![4957fecf8205a7d9a39d5539e4aba0f81fd9af338cb4fb056195adf44039542e.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/4957fecf8205a7d9a39d5539e4aba0f81fd9af338cb4fb056195adf44039542e.jpg)

![53d47c8e643379c25b0b7bd919e74e14fd7d83a6d398d3a6407fc2351ef5a0f4.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/53d47c8e643379c25b0b7bd919e74e14fd7d83a6d398d3a6407fc2351ef5a0f4.jpg)

![6d21b342e7ad82ae818e4994585c931010296195b4175a037695008cef939c06.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/6d21b342e7ad82ae818e4994585c931010296195b4175a037695008cef939c06.jpg)

![72f1b64525786f1725e13eae925b0d9dcd39f8466601fb7fbbbacf0ae7efca2a.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/72f1b64525786f1725e13eae925b0d9dcd39f8466601fb7fbbbacf0ae7efca2a.jpg)

![8623b319f348dc81773d59fa4934bab72758c320f6c61d254aaac4373de46453.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/8623b319f348dc81773d59fa4934bab72758c320f6c61d254aaac4373de46453.jpg)

![8a15b66d39702b2692d0a139993fabd50f3e671b42f52e3b2b380227aa931f99.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/8a15b66d39702b2692d0a139993fabd50f3e671b42f52e3b2b380227aa931f99.jpg)

![93271771831b2c9267434b3a039fd03694e352cd47b204b668dc33aa35712044.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/93271771831b2c9267434b3a039fd03694e352cd47b204b668dc33aa35712044.jpg)

![a15033417cd57d5dd493cb9a10c49fd69d521dadef193427869eb13eb0289840.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/a15033417cd57d5dd493cb9a10c49fd69d521dadef193427869eb13eb0289840.jpg)

![a2e58cc7229d0ec311933e6bc74446d6d31680837129fd45ac7d697309df03f9.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/a2e58cc7229d0ec311933e6bc74446d6d31680837129fd45ac7d697309df03f9.jpg)

![ab93ca044cee3d08cdc762f1860d37da8b9dd06836b9a255651b4131cee0fd74.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/ab93ca044cee3d08cdc762f1860d37da8b9dd06836b9a255651b4131cee0fd74.jpg)

![c758ddb223820539f39693091669d6f370186bac5e4601d7374ca3339e7613f8.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/c758ddb223820539f39693091669d6f370186bac5e4601d7374ca3339e7613f8.jpg)

![ce9dd6cb78187898ef566e3415db70bacecf99040a2eedea3bfb43d1f6fa568e.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/ce9dd6cb78187898ef566e3415db70bacecf99040a2eedea3bfb43d1f6fa568e.jpg)

![d4c27301826e7b7c2eecf3f6df98dbef0a43b1fe3bdb91558dcb390a506ac8ce.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/d4c27301826e7b7c2eecf3f6df98dbef0a43b1fe3bdb91558dcb390a506ac8ce.jpg)

![d50798f14bd2c2f01111d7d537859637d58115b3f05ec2fae7f554f5c29512b7.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/d50798f14bd2c2f01111d7d537859637d58115b3f05ec2fae7f554f5c29512b7.jpg)

![e1567ed416cc124fa5abd7a5f6f8cd96bd5e88fe674a4b70ff661a2ce1a0ee49.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/e1567ed416cc124fa5abd7a5f6f8cd96bd5e88fe674a4b70ff661a2ce1a0ee49.jpg)

![f42de9b9fd214e0cd538893b7b17cdbccff119a82411cfe2c35f8b4aa73846d0.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/f42de9b9fd214e0cd538893b7b17cdbccff119a82411cfe2c35f8b4aa73846d0.jpg)

![fc4a6d0533c98f3c19d60b540cd4bc48b56630a69e486d0356458acf8ad029dd.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/images/fc4a6d0533c98f3c19d60b540cd4bc48b56630a69e486d0356458acf8ad029dd.jpg)

### Tables

![1361ef28e3ee3e316abe7d9e2dc4931343d6a2fb0f849f44d27c34951302d9db.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/tables/1361ef28e3ee3e316abe7d9e2dc4931343d6a2fb0f849f44d27c34951302d9db.jpg)

![36b3bf0fb8d8ad83fb82d99fa1221bbc23769f058a3bf564f2841b238a402608.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/tables/36b3bf0fb8d8ad83fb82d99fa1221bbc23769f058a3bf564f2841b238a402608.jpg)

![5eebb67e78b7fd2d6d6db18ffacd0b193671a68d4a087911c55c6c6e5a70767b.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/tables/5eebb67e78b7fd2d6d6db18ffacd0b193671a68d4a087911c55c6c6e5a70767b.jpg)

![8f370c1efae5f78851fd05177eebffc594d2f0dba15704b860f63e65a3e7f45f.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/tables/8f370c1efae5f78851fd05177eebffc594d2f0dba15704b860f63e65a3e7f45f.jpg)

![b77c4258cc74b3c9979b0535360f6d226c1f662603090283c02180da7d2e53ca.jpg](../icml_results/3696_MVR-cache_%20Optimizing%20Semantic%20Caching%20via%20Multi-Vector%20Retrieval%20and%20Learned%20Prompt%20Segmentation/tables/b77c4258cc74b3c9979b0535360f6d226c1f662603090283c02180da7d2e53ca.jpg)

<a id="paper-3657"></a>

## MaMa: A Game-Theoretic Approach for Designing Safe Agentic Systems


### Images

![0f44583f1a0dd462a570c26550d6b957e77fcbece1ddeedf371df90fdd3f3b0f.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/0f44583f1a0dd462a570c26550d6b957e77fcbece1ddeedf371df90fdd3f3b0f.jpg)

![271345d0d37b340daf5877e363162c492d68aa74353ec5bd3558f626e2801c16.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/271345d0d37b340daf5877e363162c492d68aa74353ec5bd3558f626e2801c16.jpg)

![ab23446ce949906d3b2cb18b454dc26671cea432f12441d51bf39acddbf072a9.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/ab23446ce949906d3b2cb18b454dc26671cea432f12441d51bf39acddbf072a9.jpg)

![cc92da66fac55aa0afbfa102a99f2959ab61c9a463fa1c5855cbdd82cc105019.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/cc92da66fac55aa0afbfa102a99f2959ab61c9a463fa1c5855cbdd82cc105019.jpg)

![dc9d41400786034abfb729041c41b479923281cb5a5364a3ba0ef4664562e41c.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/dc9d41400786034abfb729041c41b479923281cb5a5364a3ba0ef4664562e41c.jpg)

![fa666da64de91786832d87606f1fe67902b825bea463b16bf937c3835c7eb5e7.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/fa666da64de91786832d87606f1fe67902b825bea463b16bf937c3835c7eb5e7.jpg)

![fb326b8dec56d4b65b96c0996ea4e418bfa8bf10755be9a1cf5309ab69c02bb1.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/images/fb326b8dec56d4b65b96c0996ea4e418bfa8bf10755be9a1cf5309ab69c02bb1.jpg)

### Tables

![04da63d2e500932d39366a9ab74e68d14be4f2cedcfdba622376a2c2aefb13e1.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/04da63d2e500932d39366a9ab74e68d14be4f2cedcfdba622376a2c2aefb13e1.jpg)

![0899f8ab719a5fde93d6058b90de3718a1068bfcea68625060da47dbf36f2dfc.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/0899f8ab719a5fde93d6058b90de3718a1068bfcea68625060da47dbf36f2dfc.jpg)

![195474c42a83d30a8815a16b455a6f220b933e5a9765b7df4d371740adb7c1f7.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/195474c42a83d30a8815a16b455a6f220b933e5a9765b7df4d371740adb7c1f7.jpg)

![6c79a1769536e78808d2074ef076da880a3a02e357c02de033aa213c90bf2e4d.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/6c79a1769536e78808d2074ef076da880a3a02e357c02de033aa213c90bf2e4d.jpg)

![6f269861f5f7f6ddedb30fdb096c71bb61d67077d044d3dc41999fa7dafcc7a7.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/6f269861f5f7f6ddedb30fdb096c71bb61d67077d044d3dc41999fa7dafcc7a7.jpg)

![968e259db57fd12d3fc30ed31538bc7c6319001219e50b0e6c4c36237d5c981c.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/968e259db57fd12d3fc30ed31538bc7c6319001219e50b0e6c4c36237d5c981c.jpg)

![9fe50ccdaadf323fc02dfb46b0bdff574508a217046eb22326e4235accff53e2.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/9fe50ccdaadf323fc02dfb46b0bdff574508a217046eb22326e4235accff53e2.jpg)

![a7bad6c1d676ea658944c1cf77689ab0c87023b3e0b48c20054a9df07e783043.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/a7bad6c1d676ea658944c1cf77689ab0c87023b3e0b48c20054a9df07e783043.jpg)

![d8e5fe1f6aec29273111f5b19b8ee74b0c1892abefcb7dcb08311425563f8c31.jpg](../icml_results/3697_MaMa_%20A%20Game-Theoretic%20Approach%20for%20Designing%20Safe%20Agentic%20Systems/tables/d8e5fe1f6aec29273111f5b19b8ee74b0c1892abefcb7dcb08311425563f8c31.jpg)

<a id="paper-3658"></a>

## Grokking Finite-Dimensional Algebra


### Images

![1d3f47223829a242563298f3e3a7d6cdd6db9e01a2c37762595fc63f0d7e7a15.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/1d3f47223829a242563298f3e3a7d6cdd6db9e01a2c37762595fc63f0d7e7a15.jpg)

![1e6639971f5036fd8bb79060bc42f0482cef86a1ccd730eeb0a3ac6e8fda06c4.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/1e6639971f5036fd8bb79060bc42f0482cef86a1ccd730eeb0a3ac6e8fda06c4.jpg)

![2296aa7711b0ad08fa0d263824af1e4a8c835a67306cfb3a126cb11d17b0ba49.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/2296aa7711b0ad08fa0d263824af1e4a8c835a67306cfb3a126cb11d17b0ba49.jpg)

![248144696d7148625b4fe0f9e46a53939c21e9e91007adeb33ac299e7b272871.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/248144696d7148625b4fe0f9e46a53939c21e9e91007adeb33ac299e7b272871.jpg)

![33c3f688758427b869aea5b90e4cca5896f15081a25569d3dd4e0d9a448e102e.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/33c3f688758427b869aea5b90e4cca5896f15081a25569d3dd4e0d9a448e102e.jpg)

![3fd4f62706445ce3f153752a331b0dd5d0c5b44c87c27db795c772c6893e600b.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/3fd4f62706445ce3f153752a331b0dd5d0c5b44c87c27db795c772c6893e600b.jpg)

![4627ed75e6047b0eee22ab26bde20ceb5e5baf52a10c5b635114d89e40d4aeb1.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/4627ed75e6047b0eee22ab26bde20ceb5e5baf52a10c5b635114d89e40d4aeb1.jpg)

![55a00a4dc10ed85ef2ca4fcd81c490794f2743afe83d17ac40e4d54a5bb025a9.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/55a00a4dc10ed85ef2ca4fcd81c490794f2743afe83d17ac40e4d54a5bb025a9.jpg)

![a4af9796c5b05ac91b59a9eda69c2c8e4cb84971f9fe7b1d9f0a906c24adb9df.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/a4af9796c5b05ac91b59a9eda69c2c8e4cb84971f9fe7b1d9f0a906c24adb9df.jpg)

![a6d10fdedb985b3a4d01c4b719d8d560a30e24d20d828a146873a1314188a57b.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/a6d10fdedb985b3a4d01c4b719d8d560a30e24d20d828a146873a1314188a57b.jpg)

![c87e694d0331a15361521ed5bb795b8df90d5cfb9f1174b22e166a8702d2e51a.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/c87e694d0331a15361521ed5bb795b8df90d5cfb9f1174b22e166a8702d2e51a.jpg)

![d44c2bbe24118bbce73ded33481b16e7dd62af7d738e2d8a4cdf8ff5729812c3.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/d44c2bbe24118bbce73ded33481b16e7dd62af7d738e2d8a4cdf8ff5729812c3.jpg)

![eedd9284843db21b25cbc3a675ed8af403ad4a7af34a2e27d8e14e25bb6fb038.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/eedd9284843db21b25cbc3a675ed8af403ad4a7af34a2e27d8e14e25bb6fb038.jpg)

![f5fc0d3e6c149e42235695123f0897a8ceffda6e5e5954589804475483b67dba.jpg](../icml_results/3698_Grokking%20Finite-Dimensional%20Algebra/images/f5fc0d3e6c149e42235695123f0897a8ceffda6e5e5954589804475483b67dba.jpg)

<a id="paper-3659"></a>

## A Unified Framework for Diffusion Model Unlearning with f-Divergence


### Images

![1a4b4c86a3ab80e2f88711e0fbc4103eaa3cf0c7c21df1fbda1352df75511cf3.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/1a4b4c86a3ab80e2f88711e0fbc4103eaa3cf0c7c21df1fbda1352df75511cf3.jpg)

![20548f5092743e5d1581848419e5c09df37a6b1db0b2a3008b19462914e35514.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/20548f5092743e5d1581848419e5c09df37a6b1db0b2a3008b19462914e35514.jpg)

![26c8c20145117d751506e3b35ad3367938c86421775a86a5d42991255e88a48e.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/26c8c20145117d751506e3b35ad3367938c86421775a86a5d42991255e88a48e.jpg)

![36ff8611b3362bd46c70973bf039174d352dada4690619a6304226f81f5a1c2e.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/36ff8611b3362bd46c70973bf039174d352dada4690619a6304226f81f5a1c2e.jpg)

![43c95bb13380dc29fd7ec09c102ae472980ccd01ea321eaa0c0af9710ec5f7b6.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/43c95bb13380dc29fd7ec09c102ae472980ccd01ea321eaa0c0af9710ec5f7b6.jpg)

![4d8e3b337fc3d4de491c5eced1b69a06ea78438eadbb34ea3f0b509138baf171.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/4d8e3b337fc3d4de491c5eced1b69a06ea78438eadbb34ea3f0b509138baf171.jpg)

![4e3b281aced0dd79aa1d076d3eaee379144276ba96b878cb4804b6a63722eac8.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/4e3b281aced0dd79aa1d076d3eaee379144276ba96b878cb4804b6a63722eac8.jpg)

![6ebcae91208cf4298f9e9bc86252a13508b2e2cc8c81d5ff1a8e906551fde494.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/6ebcae91208cf4298f9e9bc86252a13508b2e2cc8c81d5ff1a8e906551fde494.jpg)

![7335bf9253f7f913a55322a6f983c6b30f90dc62d1ccccb005e45379a4e872a9.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/7335bf9253f7f913a55322a6f983c6b30f90dc62d1ccccb005e45379a4e872a9.jpg)

![86429ffa778baab9a59cf3b624e8f150ef714d759eb98598ea278bf7c0a8a9a6.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/86429ffa778baab9a59cf3b624e8f150ef714d759eb98598ea278bf7c0a8a9a6.jpg)

![9953822762f5e39700d0ae53b571757ff16788154f526719b1e8a81103479b06.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/9953822762f5e39700d0ae53b571757ff16788154f526719b1e8a81103479b06.jpg)

![9adac69b9f24ea6235ec24feca95c546880653ed30eb925eff115e9743ef7ace.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/9adac69b9f24ea6235ec24feca95c546880653ed30eb925eff115e9743ef7ace.jpg)

![de1eca88007c6c01dd0866441f57f9a3cdd5b26179cfde52ef5d514c0bd20d20.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/de1eca88007c6c01dd0866441f57f9a3cdd5b26179cfde52ef5d514c0bd20d20.jpg)

![e2eb644294989169c3f9be05094549f5243bcc59ce60c94e51ac667c3c5566a4.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/e2eb644294989169c3f9be05094549f5243bcc59ce60c94e51ac667c3c5566a4.jpg)

![f0898adab74823f343c8327d2df9627be83eaa43b044780f53c711455e0aa6be.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/f0898adab74823f343c8327d2df9627be83eaa43b044780f53c711455e0aa6be.jpg)

![f853f3f6a0dba9705db9c3334c5f0634074a3d5a273033bc5a6c219e3ea67e11.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/images/f853f3f6a0dba9705db9c3334c5f0634074a3d5a273033bc5a6c219e3ea67e11.jpg)

### Tables

![0f3745c71e9455341d3ed5f1cb1ac3e830f2b31d1a0b1abef713ced35dddf735.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/0f3745c71e9455341d3ed5f1cb1ac3e830f2b31d1a0b1abef713ced35dddf735.jpg)

![16485ed4d90bf7a9dde890c7dc4f51f92f988f1ffe42cf0425516967e4ded2f4.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/16485ed4d90bf7a9dde890c7dc4f51f92f988f1ffe42cf0425516967e4ded2f4.jpg)

![17c223c1562f14b0bf9eff0fcdc5050e658e2b0390cec193aa1cd4c1bb12cbcf.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/17c223c1562f14b0bf9eff0fcdc5050e658e2b0390cec193aa1cd4c1bb12cbcf.jpg)

![3476d192e38f2b7feb6b31e19cbc9a8233dc50fa7e76c719c38881c511f52061.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/3476d192e38f2b7feb6b31e19cbc9a8233dc50fa7e76c719c38881c511f52061.jpg)

![4445e71fa73f4a68fe84dbd994a4e26609554b30644f520e7d6f922c5295f735.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/4445e71fa73f4a68fe84dbd994a4e26609554b30644f520e7d6f922c5295f735.jpg)

![4a8cc5377caa56ac72248f2b31200277e29c7b07708a67d3d01e8240de266236.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/4a8cc5377caa56ac72248f2b31200277e29c7b07708a67d3d01e8240de266236.jpg)

![4acf558f6747e910eb9636952da5f17b37bcb0d98608c6b3ca770676b9399ef0.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/4acf558f6747e910eb9636952da5f17b37bcb0d98608c6b3ca770676b9399ef0.jpg)

![52996d96e1500c290bbae5a5092c7c6c5c0a54be7a9c5a8d7181dcc0968da6e7.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/52996d96e1500c290bbae5a5092c7c6c5c0a54be7a9c5a8d7181dcc0968da6e7.jpg)

![99057c8117e398ea6b4bc1ebd76e07181a39598e2a87c2be4e4068d702c70357.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/99057c8117e398ea6b4bc1ebd76e07181a39598e2a87c2be4e4068d702c70357.jpg)

![acf995c78fd9605dd043703d627866f90f963b9f0464a6fdae7c9a74c03137be.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/acf995c78fd9605dd043703d627866f90f963b9f0464a6fdae7c9a74c03137be.jpg)

![afb0438fe0377b5747b56911df9f917fdfee0e95e41715fb37d74b71be6efb83.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/afb0438fe0377b5747b56911df9f917fdfee0e95e41715fb37d74b71be6efb83.jpg)

![b0d001ed982c7eb873ffbc267dfd1512cfad96ba231740a10cccd104286cfc1f.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/b0d001ed982c7eb873ffbc267dfd1512cfad96ba231740a10cccd104286cfc1f.jpg)

![b11fafce0947f8272d08c20776f40544a508520ecd55c0eb9f30726b78ea5086.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/b11fafce0947f8272d08c20776f40544a508520ecd55c0eb9f30726b78ea5086.jpg)

![df993e1e11ffc3bf2f49d3cab875097ed4991412ccd7b6cd2de1c5a1cdaf7aa0.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/df993e1e11ffc3bf2f49d3cab875097ed4991412ccd7b6cd2de1c5a1cdaf7aa0.jpg)

![e6381ea86f308d683e0097bdfd176ba0e0bb6cdd2b191b840c4567f947e6078f.jpg](../icml_results/3699_A%20Unified%20Framework%20for%20Diffusion%20Model%20Unlearning%20with%20f-Divergence/tables/e6381ea86f308d683e0097bdfd176ba0e0bb6cdd2b191b840c4567f947e6078f.jpg)

<a id="paper-3660"></a>

## ABCD: All Biases Come Disguised


### Images

![2358c6ecea0697ddd5ddeaedd2d254d4d85e49408aef9db507325665c3ac5858.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/2358c6ecea0697ddd5ddeaedd2d254d4d85e49408aef9db507325665c3ac5858.jpg)

![2b3f599baad8f3339519ffa142843417ea4b06d8e79e18d458eca0ca3756a01a.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/2b3f599baad8f3339519ffa142843417ea4b06d8e79e18d458eca0ca3756a01a.jpg)

![2b81dbd01a4b946e560904facfccd5b9af2095a5555acbe213f5ec3724353588.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/2b81dbd01a4b946e560904facfccd5b9af2095a5555acbe213f5ec3724353588.jpg)

![36fa21579ab311abbb70c2ff5e1d6ff4001a1f54c6ce7805322159c22db3e634.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/36fa21579ab311abbb70c2ff5e1d6ff4001a1f54c6ce7805322159c22db3e634.jpg)

![4d7a5c2b74154cf499867df79e395c4acf96c12e0d16e9671c759453fa30a32c.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/4d7a5c2b74154cf499867df79e395c4acf96c12e0d16e9671c759453fa30a32c.jpg)

![4f2957af100223d3b02460a97fa096fce336d8841a5cf126179b80b91e985392.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/4f2957af100223d3b02460a97fa096fce336d8841a5cf126179b80b91e985392.jpg)

![5a7609d3db848ff418dba6464b6036eb3ac11cc2669b740402d634ea729eb2dd.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/5a7609d3db848ff418dba6464b6036eb3ac11cc2669b740402d634ea729eb2dd.jpg)

![71e1f36d5577a2564b4ec30f9b65a7a68d0d47d5f5818bde281022b4021528f7.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/71e1f36d5577a2564b4ec30f9b65a7a68d0d47d5f5818bde281022b4021528f7.jpg)

![73064f29a5e05d563d7cae7ced4a588ca32d2d0ee77c10f3f5b2ae76705debcd.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/73064f29a5e05d563d7cae7ced4a588ca32d2d0ee77c10f3f5b2ae76705debcd.jpg)

![741a9ee17d217cb2c3458601001f40afc51ae659674d9b60e339fee7316804b7.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/741a9ee17d217cb2c3458601001f40afc51ae659674d9b60e339fee7316804b7.jpg)

![7f0e546feeafe743ec445b24b558211b92d1c64eded84cd4b63e8110fa64d572.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/7f0e546feeafe743ec445b24b558211b92d1c64eded84cd4b63e8110fa64d572.jpg)

![8d2a133980329a6d4e7fb9940ed85e3c344ca7ee9ff5d72502d53547bdb935b8.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/8d2a133980329a6d4e7fb9940ed85e3c344ca7ee9ff5d72502d53547bdb935b8.jpg)

![8f1e8412a085fa3c0f56358d54d8901ddf146b894325a6f88a0241f0855456e4.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/8f1e8412a085fa3c0f56358d54d8901ddf146b894325a6f88a0241f0855456e4.jpg)

![987dcab8c48725d0c251a6f6545ccf850e2a7afcd9090a80041ab8a1aa499c3a.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/987dcab8c48725d0c251a6f6545ccf850e2a7afcd9090a80041ab8a1aa499c3a.jpg)

![9f76757acc60093a411f24f92d3ebcb46d76be1b708aed8912c45721b7c28a49.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/9f76757acc60093a411f24f92d3ebcb46d76be1b708aed8912c45721b7c28a49.jpg)

![b2ad084392621498264e42a27ba761fea12ebda941fd85e437c9a2295642c5f1.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/b2ad084392621498264e42a27ba761fea12ebda941fd85e437c9a2295642c5f1.jpg)

![ba1958d375057bd8c3023bca23eb5095be4283c05cd5d4f6e6eabbb4c7732340.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/ba1958d375057bd8c3023bca23eb5095be4283c05cd5d4f6e6eabbb4c7732340.jpg)

![bf17727ad2f9c6a528b89a0bc106000f840cdf7f22871e31f863b99e05344d8d.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/bf17727ad2f9c6a528b89a0bc106000f840cdf7f22871e31f863b99e05344d8d.jpg)

![c458a4969e60218caa2a0df59283493a860251b30cddabbdb82909f6f85428ee.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/c458a4969e60218caa2a0df59283493a860251b30cddabbdb82909f6f85428ee.jpg)

![c656f15f8d0e085a1ea857ec4fb55ead0ac46a15148f724d60c5b9820547470f.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/c656f15f8d0e085a1ea857ec4fb55ead0ac46a15148f724d60c5b9820547470f.jpg)

![d6065ebd18a5875cae7299e99ef32e39173d1a71092b8f89d4baccfebd7616ed.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/images/d6065ebd18a5875cae7299e99ef32e39173d1a71092b8f89d4baccfebd7616ed.jpg)

### Tables

![0504d47b37ad724330f76c9683835db6598adb1d5c1cb26abba01aa38ff96795.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/0504d47b37ad724330f76c9683835db6598adb1d5c1cb26abba01aa38ff96795.jpg)

![057a7cf42d0c295901a5398610e1c973eddbf0ba50a650e1feafc6015c190402.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/057a7cf42d0c295901a5398610e1c973eddbf0ba50a650e1feafc6015c190402.jpg)

![08b63957c5e9bb8d7f41b7746b671f2e3b93aa9df098f53fd7df54e6f3b59bfd.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/08b63957c5e9bb8d7f41b7746b671f2e3b93aa9df098f53fd7df54e6f3b59bfd.jpg)

![1b99753287a6f6ca05ace2469222abccda1eefca8eae61d35f9ad8fc16015ce1.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/1b99753287a6f6ca05ace2469222abccda1eefca8eae61d35f9ad8fc16015ce1.jpg)

![22f76f9b73a311f6bf15945e803a0432f8f32267ecbd3b3a848913fc39f9be8d.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/22f76f9b73a311f6bf15945e803a0432f8f32267ecbd3b3a848913fc39f9be8d.jpg)

![57c66b8827ca9d817d0d7d406893ee2a19f52a92f1aba4e9c9cd88ee280d1c6b.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/57c66b8827ca9d817d0d7d406893ee2a19f52a92f1aba4e9c9cd88ee280d1c6b.jpg)

![685fef98d803972f4ee4498319e191b586b9f44860884e52f808002af48be1bc.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/685fef98d803972f4ee4498319e191b586b9f44860884e52f808002af48be1bc.jpg)

![780b90cfb60862653a35178ed63c9fddcae8761876069406d62a923b32a05751.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/780b90cfb60862653a35178ed63c9fddcae8761876069406d62a923b32a05751.jpg)

![7ba9c68e372e737d10d0c8d2bb8e63cbe6aba4373a71ff1d6ad2d0075ab199c3.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/7ba9c68e372e737d10d0c8d2bb8e63cbe6aba4373a71ff1d6ad2d0075ab199c3.jpg)

![85f80d9920078cc48f1d5a5add164aabf3067c6f0ffe7542ac562cad24cfac78.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/85f80d9920078cc48f1d5a5add164aabf3067c6f0ffe7542ac562cad24cfac78.jpg)

![86cbc7a4f7010b555aaa45955e3c624de425cbc7d2506790d54d50561b5f2292.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/86cbc7a4f7010b555aaa45955e3c624de425cbc7d2506790d54d50561b5f2292.jpg)

![b5b2ed39c8010b7deba8d07d173570eb8e64d9df21d86d08464d6ae8316f4a98.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/b5b2ed39c8010b7deba8d07d173570eb8e64d9df21d86d08464d6ae8316f4a98.jpg)

![bdaf8c3de58d19458cee07192f5cb1b1ebf7f4cb1a8853ebea3a43018615ae59.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/bdaf8c3de58d19458cee07192f5cb1b1ebf7f4cb1a8853ebea3a43018615ae59.jpg)

![c6cd038e00b35da8b97d4b15bc2224f479ef9d8565461a72dec8799873b1b530.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/c6cd038e00b35da8b97d4b15bc2224f479ef9d8565461a72dec8799873b1b530.jpg)

![e4929547f8fb93d8a40067b0beca6665fd3a10a7e93ad6c65cdcec175fa21382.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/e4929547f8fb93d8a40067b0beca6665fd3a10a7e93ad6c65cdcec175fa21382.jpg)

![f9dd02c67ad6dd3d0b1020ee25b6881fc880a15ad9cb8ef79b4ad2708b37dcd8.jpg](../icml_results/3700_ABCD_%20All%20Biases%20Come%20Disguised/tables/f9dd02c67ad6dd3d0b1020ee25b6881fc880a15ad9cb8ef79b4ad2708b37dcd8.jpg)

<a id="paper-3661"></a>

## An Algebraic View of the Expressivity of Recurrent Language Models


### Images

![04f34f2df30263084463297161f1384f061d1635b5b8d1396395d71e5c6309cf.jpg](../icml_results/3701_An%20Algebraic%20View%20of%20the%20Expressivity%20of%20Recurrent%20Language%20Models/images/04f34f2df30263084463297161f1384f061d1635b5b8d1396395d71e5c6309cf.jpg)

![5a1c2baba85b4b9d629f73892f904a69d007a55e5a3e2d9f3711f51ceda074f6.jpg](../icml_results/3701_An%20Algebraic%20View%20of%20the%20Expressivity%20of%20Recurrent%20Language%20Models/images/5a1c2baba85b4b9d629f73892f904a69d007a55e5a3e2d9f3711f51ceda074f6.jpg)

### Tables

![3d57aa42f0d5d804eb0b56ef45349ddfd1d3c10f0c28efcb075162daa9067507.jpg](../icml_results/3701_An%20Algebraic%20View%20of%20the%20Expressivity%20of%20Recurrent%20Language%20Models/tables/3d57aa42f0d5d804eb0b56ef45349ddfd1d3c10f0c28efcb075162daa9067507.jpg)

![dfd9c2c011cbb4c5b7afdc50c67c1831d0747525ccf14718a473ace1415e1cef.jpg](../icml_results/3701_An%20Algebraic%20View%20of%20the%20Expressivity%20of%20Recurrent%20Language%20Models/tables/dfd9c2c011cbb4c5b7afdc50c67c1831d0747525ccf14718a473ace1415e1cef.jpg)

![f97ae330a2167d814912c01d215d0430a6136f08afd30bd9606d24fb8df20426.jpg](../icml_results/3701_An%20Algebraic%20View%20of%20the%20Expressivity%20of%20Recurrent%20Language%20Models/tables/f97ae330a2167d814912c01d215d0430a6136f08afd30bd9606d24fb8df20426.jpg)

<a id="paper-3662"></a>

## GAUSS: Graph-Assisted Uncertainty Quantification using Structure and Semantics for Long-Form Generation in LLMs


### Images

![0e2b34aaf6894910911387d75aa9d707aae71e8d8982a879a32815f01ecaa903.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/0e2b34aaf6894910911387d75aa9d707aae71e8d8982a879a32815f01ecaa903.jpg)

![30cb99a579d770b9b02c49661d57ac2ee4a2a9ec16df7e090dc10a155eba43f3.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/30cb99a579d770b9b02c49661d57ac2ee4a2a9ec16df7e090dc10a155eba43f3.jpg)

![49923a7e3a2b01ff2140d13d6f6b79d00976992021c84e85adc4f4e253a8c281.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/49923a7e3a2b01ff2140d13d6f6b79d00976992021c84e85adc4f4e253a8c281.jpg)

![6f205304f3bb29353be566342d62af4f99e5547f3fd49812ff2692e4304b567a.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/6f205304f3bb29353be566342d62af4f99e5547f3fd49812ff2692e4304b567a.jpg)

![90fdbeb592c38a0125db009a26c650ec11eec675def73cc82addaff018dd6642.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/90fdbeb592c38a0125db009a26c650ec11eec675def73cc82addaff018dd6642.jpg)

![9e8893b06826faec5e5f755dc1880f18165eae36204a183849ad5473575d5b8e.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/images/9e8893b06826faec5e5f755dc1880f18165eae36204a183849ad5473575d5b8e.jpg)

### Tables

![1a44c7a39ddc5ef31ff29a4154606001aed894a9e440542154488fd23d73e8f1.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/1a44c7a39ddc5ef31ff29a4154606001aed894a9e440542154488fd23d73e8f1.jpg)

![1e2f127fc8280818f71babefe92905d43e0f5a72b013b57ed2e604572e279858.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/1e2f127fc8280818f71babefe92905d43e0f5a72b013b57ed2e604572e279858.jpg)

![20e7ddf1c86be7fb18bb22368a765158386f139c5b9e5491ba6dcb93d314173b.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/20e7ddf1c86be7fb18bb22368a765158386f139c5b9e5491ba6dcb93d314173b.jpg)

![57d27dd7bd13bae369c7618766d7e7376d6a500de53ed83301b6c47b40a6a15d.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/57d27dd7bd13bae369c7618766d7e7376d6a500de53ed83301b6c47b40a6a15d.jpg)

![6ff7b14d1022f4e7d341b036f30aa4ff25b77f634eba74011af8554474f2ea02.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/6ff7b14d1022f4e7d341b036f30aa4ff25b77f634eba74011af8554474f2ea02.jpg)

![77ecfcb23d293919f4378866ad49e6751625942ee08c1ab1a6f50846349cceef.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/77ecfcb23d293919f4378866ad49e6751625942ee08c1ab1a6f50846349cceef.jpg)

![976ef195370b1d513df99d6169cb74e493f0f4c3a1f4f72d6da4947591604e9f.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/976ef195370b1d513df99d6169cb74e493f0f4c3a1f4f72d6da4947591604e9f.jpg)

![a12f147de4a7e5d4d92735f59355285b595666ee2e668869a2caa3eb227bc7da.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/a12f147de4a7e5d4d92735f59355285b595666ee2e668869a2caa3eb227bc7da.jpg)

![ac0578fbab5c8a1a33b84cdcc575c84165eb33efd5a2b94ad7e8c42fe1fb3b41.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/ac0578fbab5c8a1a33b84cdcc575c84165eb33efd5a2b94ad7e8c42fe1fb3b41.jpg)

![aeffab6c18b58c1758912565f81c83fe52ecb5346b08092526824c27bd970939.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/aeffab6c18b58c1758912565f81c83fe52ecb5346b08092526824c27bd970939.jpg)

![b440d2da2aa386488d0149c44fe590799263c7b4489956e1dea5fca406b0f1b3.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/b440d2da2aa386488d0149c44fe590799263c7b4489956e1dea5fca406b0f1b3.jpg)

![f09efa48992b4057f641ce46af6b485fc49146a62875a4f6fde9f12ee36680ac.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/f09efa48992b4057f641ce46af6b485fc49146a62875a4f6fde9f12ee36680ac.jpg)

![f1136662488b0fda6f6086164bffcabb0453e16960f50bca45ad56b7c588c5cf.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/f1136662488b0fda6f6086164bffcabb0453e16960f50bca45ad56b7c588c5cf.jpg)

![fec4cf66956713ca530dba9dc7b5a877dee0013d67ad3b9de0bc4057d614c334.jpg](../icml_results/3702_GAUSS_%20Graph-Assisted%20Uncertainty%20Quantification%20using%20Structure%20and%20Semantics%20for%20Long-Form%20Generat/tables/fec4cf66956713ca530dba9dc7b5a877dee0013d67ad3b9de0bc4057d614c334.jpg)

<a id="paper-3663"></a>

## Joint Enhancement and Classification using Coupled Diffusion Models of Signals and Logits

### Images

![1026205496a3483644a98f80913afc8d2751399e50fe17511f2ee3a5367b8b44.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/images/1026205496a3483644a98f80913afc8d2751399e50fe17511f2ee3a5367b8b44.jpg)

![1ed13954aecedad436c74c43856b4b275bea082557bb571b50828d6af165d362.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/images/1ed13954aecedad436c74c43856b4b275bea082557bb571b50828d6af165d362.jpg)

![65b61125834f891b99f15f63ee23bc81bf0f731f17771fd3b010adf74157fa77.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/images/65b61125834f891b99f15f63ee23bc81bf0f731f17771fd3b010adf74157fa77.jpg)

![74acfab1e7bf329629bc8a85ac832b793d9977d61a030623020aeb580803de40.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/images/74acfab1e7bf329629bc8a85ac832b793d9977d61a030623020aeb580803de40.jpg)

![9c14e2fadbf504692531c727d6008d12c6028310d077ae5ffdc34194623330c2.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/images/9c14e2fadbf504692531c727d6008d12c6028310d077ae5ffdc34194623330c2.jpg)

### Tables

![569c347dbcd393471c5c176f016f6b4f5747f40bf7663c3a498b84b2e821c38b.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/tables/569c347dbcd393471c5c176f016f6b4f5747f40bf7663c3a498b84b2e821c38b.jpg)

![6e3f778737ea3ad3dff4c53648a89bcd0b2ef672dce7933d7a8e4f6e1675b095.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/tables/6e3f778737ea3ad3dff4c53648a89bcd0b2ef672dce7933d7a8e4f6e1675b095.jpg)

![7d9ee9d8f2df016f3d2105fe2431e15560ad3d4959f9ef9f6b28c18fff9f1f5e.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/tables/7d9ee9d8f2df016f3d2105fe2431e15560ad3d4959f9ef9f6b28c18fff9f1f5e.jpg)

![ace532641ee303b4e0928f674e9c0bb094ca2a23abf64d7f7a192a5a92279907.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/tables/ace532641ee303b4e0928f674e9c0bb094ca2a23abf64d7f7a192a5a92279907.jpg)

![cdc2db1cadcf249ee24c9109172a56826439a8e90d0217f9389105d6c30fd38f.jpg](../icml_results/3703_Joint%20Enhancement%20and%20Classification%20using%20Coupled%20Diffusion%20Models%20of%20Signals%20and%20Logits/tables/cdc2db1cadcf249ee24c9109172a56826439a8e90d0217f9389105d6c30fd38f.jpg)
