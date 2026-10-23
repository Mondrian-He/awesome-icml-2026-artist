# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 188 of 197

## 目录 (Table of Contents)

1. [Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](#paper-6172)
2. [On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](#paper-6173)
3. [Disentangling Consensus and Value-Specific Representations for Controllable Pluralistic Value Alignment in LLMs](#paper-6174)
4. [URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization](#paper-6175)
5. [Bayes-inspired Integration of Pretrained Priors and Few-Shot Evidence for Few-Shot Classification](#paper-6176)
6. [Statistically Calibrated Scaling for Token Merging in Transformers](#paper-6177)
7. [HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](#paper-6178)
8. [BAS: Bridging Adam and SignSGD for Memory-Efficient LLM Training](#paper-6179)
9. [AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning](#paper-6180)
10. [Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting](#paper-6181)
11. [Demystifying Mergeability: Interpretable Properties to Predict Model Merging Success](#paper-6182)
12. [Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](#paper-6183)
13. [How Language Models Process Negation](#paper-6184)
14. [Mitigating the Modality Gap in Vision–Language Models with Fractal Spectral Geometry](#paper-6185)
15. [Tailoring the Training: Difficulty-Aware Learning Strategy Allocation for Large Language Models](#paper-6186)
16. [ForensicConcept: Transferable Forensic Concepts for AIGI Detection](#paper-6187)
17. [Structured Multi-step Jailbreaking under a Hamiltonian Generative Formulation](#paper-6188)
18. [Calibrating Decision Robustness via Inverse Conformal Risk Control](#paper-6189)
19. [Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing](#paper-6190)
20. [EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](#paper-6191)
21. [Graph-GRPO: Training Graph Flow Models with Reinforcement Learning](#paper-6192)
22. [G$^2$TAM: Geometry Grounded Track Anything Model](#paper-6193)
23. [Pull Requests as a Training Signal for Repo-Level Code Editing](#paper-6194)
24. [When Do Graph Foundation Models Transfer? A Data-Centric Theory](#paper-6195)
25. [Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents](#paper-6196)
26. [Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems](#paper-6197)
27. [From Guessing to Placeholding: A Cost-Theoretic Framework for Uncertainty-Aware Code Completion](#paper-6198)
28. [Simple Denoising Diffusion Language Models](#paper-6199)
29. [Learning Transferable Interaction Primitives from Game Videos for Humanoid Locomotion](#paper-6200)
30. [Alethia: a Foundational Encoder for Voice Deepfakes](#paper-6201)
31. [Learning-augmented Rent-or-Buy with a Sample](#paper-6202)
32. [EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](#paper-6203)
33. [FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects](#paper-6204)

---


<a id="paper-6172"></a>

## Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

### Images

![27d7ae20e18a549d38aa475678a731dce7ed10d8da71cb40c83c4a10684bde11.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/27d7ae20e18a549d38aa475678a731dce7ed10d8da71cb40c83c4a10684bde11.jpg)

![5bf89a3a3b9f724c8cd1d5404421452f125a622ed8dd696e473f36ba92b573c5.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/5bf89a3a3b9f724c8cd1d5404421452f125a622ed8dd696e473f36ba92b573c5.jpg)

![83aad9cd536f26c32e2fefb61b52565d45466315dae9fa954c2b2a26e8a98d7d.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/83aad9cd536f26c32e2fefb61b52565d45466315dae9fa954c2b2a26e8a98d7d.jpg)

![8a7ac348b7e529ae662500285b297668a00e645e11b080009ecf5ef049b0577f.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/8a7ac348b7e529ae662500285b297668a00e645e11b080009ecf5ef049b0577f.jpg)

![b96e27815d54b54f71149a81ee667761f5edbeab60464da1eb40ba30b16e48f9.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/b96e27815d54b54f71149a81ee667761f5edbeab60464da1eb40ba30b16e48f9.jpg)

![f5fe0862e94a3fdf31f3dcdae72ef0514bfdd497bcfa8b847b12313327dc0eef.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/f5fe0862e94a3fdf31f3dcdae72ef0514bfdd497bcfa8b847b12313327dc0eef.jpg)

### Tables

![2bb39541f8bc00a61b9b7b99cb4bd4d2981be3b0e47944b348666d105d7c398a.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/2bb39541f8bc00a61b9b7b99cb4bd4d2981be3b0e47944b348666d105d7c398a.jpg)

![68b00f06189d0fc201df98b356b11f37208ffb5fefabfa9b93a7880ce26510b3.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/68b00f06189d0fc201df98b356b11f37208ffb5fefabfa9b93a7880ce26510b3.jpg)

![75e8564389d75aac5198fdea4eb17ab7d460f33a2abaea047daa07fd20e99b86.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/75e8564389d75aac5198fdea4eb17ab7d460f33a2abaea047daa07fd20e99b86.jpg)

![80e1e62022e1dc5a3c88568d3542913afd6a02244fcc11745e179cdc06748f38.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/80e1e62022e1dc5a3c88568d3542913afd6a02244fcc11745e179cdc06748f38.jpg)

![9cf1ff36a523149b26a41a633c76d296d6a30f4cf86c15fc813866de7205051a.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9cf1ff36a523149b26a41a633c76d296d6a30f4cf86c15fc813866de7205051a.jpg)

![9d03468c515e65bb2d10d7f90219f775766e3c7a597a11fb08a62d24a4257688.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9d03468c515e65bb2d10d7f90219f775766e3c7a597a11fb08a62d24a4257688.jpg)

![9d527c78dcfa0e7d2ed5dc56cad0d5f06030c0dc128ec0ff1aaa50f2ec4f71f2.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9d527c78dcfa0e7d2ed5dc56cad0d5f06030c0dc128ec0ff1aaa50f2ec4f71f2.jpg)

![b0f1ff95c5eea0b207685f3be306fd48d20f8c450c55eaed39d3e9f8b2010f58.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/b0f1ff95c5eea0b207685f3be306fd48d20f8c450c55eaed39d3e9f8b2010f58.jpg)

![bb8db74b32be8ec3e2f1005fd4f2d06406ab078af6a7a01f64d08d80bbe626ce.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/bb8db74b32be8ec3e2f1005fd4f2d06406ab078af6a7a01f64d08d80bbe626ce.jpg)

![ca085a5a7a2b75310504696eee88c092229a9830b0fe022ce808eeb0cdc71161.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/ca085a5a7a2b75310504696eee88c092229a9830b0fe022ce808eeb0cdc71161.jpg)

## Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner


### Images

![0a9d37bd84561527c5e10e43cec98d4d9052afa87cb93acc61b5a6b610e93bd5.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/images/0a9d37bd84561527c5e10e43cec98d4d9052afa87cb93acc61b5a6b610e93bd5.jpg)

![26dd183ee3e33c085721f8ec953e8f6da25a54a0f11f3521560ae4431d49bd4a.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/images/26dd183ee3e33c085721f8ec953e8f6da25a54a0f11f3521560ae4431d49bd4a.jpg)

![a4f9d5f24876a942adacff812115be4671b30a32143d5cf2ba01d69568e25a02.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/images/a4f9d5f24876a942adacff812115be4671b30a32143d5cf2ba01d69568e25a02.jpg)

![e5bd5dc382b6f7e45c04e8bd1dc84f5e5fe0014345966416cccbe3cf7a83f8c3.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/images/e5bd5dc382b6f7e45c04e8bd1dc84f5e5fe0014345966416cccbe3cf7a83f8c3.jpg)

![efe4af3b918d36922602505a5b245315d3f54191007abb6e6d1e0b185687108d.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/images/efe4af3b918d36922602505a5b245315d3f54191007abb6e6d1e0b185687108d.jpg)

### Tables

![01d1650e4d86a7f7470c44b75fc425e64f4915d27a9e59df20e64b4d19219ed8.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/01d1650e4d86a7f7470c44b75fc425e64f4915d27a9e59df20e64b4d19219ed8.jpg)

![173446906e7796dba66d7e18c6f833783690194932fc1a084fb0281eb472f3b3.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/173446906e7796dba66d7e18c6f833783690194932fc1a084fb0281eb472f3b3.jpg)

![227aae436e7f15fded62ff9354f0cafdab66f097a454dc2eecd563ecea3ef194.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/227aae436e7f15fded62ff9354f0cafdab66f097a454dc2eecd563ecea3ef194.jpg)

![2c46d4daeeeb69ca879f61a38ecba29c439f4a47e8ef4e14ae4136aca15388c4.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/2c46d4daeeeb69ca879f61a38ecba29c439f4a47e8ef4e14ae4136aca15388c4.jpg)

![2f4a2a31db4e625aaa0a09e11faa08908803d5132f2960b4016691a97139baff.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/2f4a2a31db4e625aaa0a09e11faa08908803d5132f2960b4016691a97139baff.jpg)

![47065f0987d250ea17a6551171fd2c95dbc16a4984b5e6979116e633f1f4a012.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/47065f0987d250ea17a6551171fd2c95dbc16a4984b5e6979116e633f1f4a012.jpg)

![9d8e03118e0171ab905542fa78b58010df65b6ccaf0bc404397b273fdf07d854.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/9d8e03118e0171ab905542fa78b58010df65b6ccaf0bc404397b273fdf07d854.jpg)

![b7504cb8dd662e3ee330b317e73d0dca1a144c7f4ed54cc6321f3179aeb77833.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/b7504cb8dd662e3ee330b317e73d0dca1a144c7f4ed54cc6321f3179aeb77833.jpg)

![eee69afe07950c00d4f298547ba417006b117fa19bfc4a75eec9a3b1a4a70b0d.jpg](../icml_results/6222_Coevolutionary%20Continuous%20Discrete%20Diffusion_%20Make%20Your%20Diffusion%20Language%20Model%20a%20Latent%20Reasoner/tables/eee69afe07950c00d4f298547ba417006b117fa19bfc4a75eec9a3b1a4a70b0d.jpg)

<a id="paper-6173"></a>

## On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective


### Images

![37afcaeb62e48bc5224b91294197afd65639b3c0c2b340d432c40bdc3562b873.jpg](../icml_results/6223_On%20the%20Learnability%20of%20Test-Time%20Adaptation_%20A%20Recovery%20Complexity%20Perspective/images/37afcaeb62e48bc5224b91294197afd65639b3c0c2b340d432c40bdc3562b873.jpg)

### Tables

![30525ee84ed55960d0de2db9125e8af2ad62f07f466eb73223ecfde2436c98da.jpg](../icml_results/6223_On%20the%20Learnability%20of%20Test-Time%20Adaptation_%20A%20Recovery%20Complexity%20Perspective/tables/30525ee84ed55960d0de2db9125e8af2ad62f07f466eb73223ecfde2436c98da.jpg)

![6b60ddf057b4a7178408f5ff0feb87844f83b757b53e857104434e7ffb94b37e.jpg](../icml_results/6223_On%20the%20Learnability%20of%20Test-Time%20Adaptation_%20A%20Recovery%20Complexity%20Perspective/tables/6b60ddf057b4a7178408f5ff0feb87844f83b757b53e857104434e7ffb94b37e.jpg)

![c5d84a5817ee1c76bfaa921e80f189fbd15720ea4ef86dec8d07f70abf1697e3.jpg](../icml_results/6223_On%20the%20Learnability%20of%20Test-Time%20Adaptation_%20A%20Recovery%20Complexity%20Perspective/tables/c5d84a5817ee1c76bfaa921e80f189fbd15720ea4ef86dec8d07f70abf1697e3.jpg)

![eb0effd27f2e38528557480e2881e9030ac1ed698796f7c61001c7605cfb205b.jpg](../icml_results/6223_On%20the%20Learnability%20of%20Test-Time%20Adaptation_%20A%20Recovery%20Complexity%20Perspective/tables/eb0effd27f2e38528557480e2881e9030ac1ed698796f7c61001c7605cfb205b.jpg)

<a id="paper-6174"></a>

## Disentangling Consensus and Value-Specific Representations for Controllable Pluralistic Value Alignment in LLMs


### Images

![1c87ec5ea7175e94b9071f86979f563d4bd74cfbf0317b0e9d6c954a75680e1f.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/1c87ec5ea7175e94b9071f86979f563d4bd74cfbf0317b0e9d6c954a75680e1f.jpg)

![3eb6df4dce3cc3a38295f3b51db181a3ed78c582d51bee52356b1f20bb5976e0.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/3eb6df4dce3cc3a38295f3b51db181a3ed78c582d51bee52356b1f20bb5976e0.jpg)

![485e06070eae741310c699fb2a9d52d1983f319f7ce63770ea371f88a22a883e.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/485e06070eae741310c699fb2a9d52d1983f319f7ce63770ea371f88a22a883e.jpg)

![49ba8274fad955cba692bbf6fb34678eb065b4af694799c52c4cebfaad1923c5.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/49ba8274fad955cba692bbf6fb34678eb065b4af694799c52c4cebfaad1923c5.jpg)

![76114e7636a469c1b01f811b2962d0a7df3a89e8da4946c769de0e3281e1f513.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/76114e7636a469c1b01f811b2962d0a7df3a89e8da4946c769de0e3281e1f513.jpg)

![87863e1105dbae37c54b1a282d180a59f7632255b494a70689b99e9a5d2d6969.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/87863e1105dbae37c54b1a282d180a59f7632255b494a70689b99e9a5d2d6969.jpg)

![b7781321f33eba16636a6e541c2203912a5c4b4cef129b24dd82d034c662e3f0.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/b7781321f33eba16636a6e541c2203912a5c4b4cef129b24dd82d034c662e3f0.jpg)

![bcfce55b853633e8563defaaa1caec52c50dc82d0d8ac5055ead1cf646a3dc9b.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/bcfce55b853633e8563defaaa1caec52c50dc82d0d8ac5055ead1cf646a3dc9b.jpg)

![da9c112bc84e84164e627c27221ec796468fab2a789643681e1e12609e406966.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/da9c112bc84e84164e627c27221ec796468fab2a789643681e1e12609e406966.jpg)

![f275306f67e751e7ab789ec9e65d241c0fd20bdde2ee5fed37b281ea498a78d3.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/images/f275306f67e751e7ab789ec9e65d241c0fd20bdde2ee5fed37b281ea498a78d3.jpg)

### Tables

![03c5488d31e6f10bfdb75d20c5029616d83fbe7ffca77ab6f2930796a43a2f62.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/03c5488d31e6f10bfdb75d20c5029616d83fbe7ffca77ab6f2930796a43a2f62.jpg)

![210b5a0271c060112d90e6c34ac86659c443ffe6f454cc6099ae6eaafd17b631.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/210b5a0271c060112d90e6c34ac86659c443ffe6f454cc6099ae6eaafd17b631.jpg)

![2dd093784ec66aaa1f4940111b66009477f96f52568dcb56c086813b616be90b.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/2dd093784ec66aaa1f4940111b66009477f96f52568dcb56c086813b616be90b.jpg)

![4432e5812f01469dc33bdbdf19048a1bf87b10499dd25707baa69127688769fd.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/4432e5812f01469dc33bdbdf19048a1bf87b10499dd25707baa69127688769fd.jpg)

![5cd547bd7bfe0e5f54169bab9a7749f33e46f6736d947dda4efd319ee389ab24.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/5cd547bd7bfe0e5f54169bab9a7749f33e46f6736d947dda4efd319ee389ab24.jpg)

![7a9e821b726d19007198cc2aa9687443913827ebc7b3c584b269406e0429958e.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/7a9e821b726d19007198cc2aa9687443913827ebc7b3c584b269406e0429958e.jpg)

![94774344694310ae09f523da5f8e1b2b7256256de371cfd8ec301945986b95c1.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/94774344694310ae09f523da5f8e1b2b7256256de371cfd8ec301945986b95c1.jpg)

![94837b34e546628111c07141f1a71961e3c945e34507643eb813f4ba4a41a270.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/94837b34e546628111c07141f1a71961e3c945e34507643eb813f4ba4a41a270.jpg)

![da5735ad3c38cd649955c44fcafc89efc1ea03a2a15d02969e97ddb1639ea74a.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/da5735ad3c38cd649955c44fcafc89efc1ea03a2a15d02969e97ddb1639ea74a.jpg)

![e2e1e843e3dda4f7e42f151b7b0c080871c1913b3caa7f1e481d8ff1c7c8c7ec.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/e2e1e843e3dda4f7e42f151b7b0c080871c1913b3caa7f1e481d8ff1c7c8c7ec.jpg)

![f081189b321dff89efbce41e5589bbfebf961ca3c4bf85017ea25d4d68603b88.jpg](../icml_results/6224_Disentangling%20Consensus%20and%20Value-Specific%20Representations%20for%20Controllable%20Pluralistic%20Value%20Alignm/tables/f081189b321dff89efbce41e5589bbfebf961ca3c4bf85017ea25d4d68603b88.jpg)

<a id="paper-6175"></a>

## URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization


### Images

![11fac48c190e2ebb7ef0ed1667ea84e918e25a1b9ab5363227dfeeb4d9f0f62c.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/images/11fac48c190e2ebb7ef0ed1667ea84e918e25a1b9ab5363227dfeeb4d9f0f62c.jpg)

![1d4cb0f65fb2ca5bbe29e193c0ca2fb2cea9c5c63deae7362db4bf44b7909ae5.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/images/1d4cb0f65fb2ca5bbe29e193c0ca2fb2cea9c5c63deae7362db4bf44b7909ae5.jpg)

![374e6b6b7b3df418c58fd0f17c3030a9391f296c71bc8ed3ecda1eb8e09a411a.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/images/374e6b6b7b3df418c58fd0f17c3030a9391f296c71bc8ed3ecda1eb8e09a411a.jpg)

![e4793d7c05bab25b2a406bfeb87622505dde7a7f627e11af82bbab309f5ec267.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/images/e4793d7c05bab25b2a406bfeb87622505dde7a7f627e11af82bbab309f5ec267.jpg)

### Tables

![04b31bc4cdbbb1917d6ce5e212b5f019e7d3cd6aac4d4d1298cc9e98005f2d1c.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/04b31bc4cdbbb1917d6ce5e212b5f019e7d3cd6aac4d4d1298cc9e98005f2d1c.jpg)

![060a756df2a1ce9ec464712189934f8e0ae0ca90be01282c495423c4bbea7ad8.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/060a756df2a1ce9ec464712189934f8e0ae0ca90be01282c495423c4bbea7ad8.jpg)

![08a93d1d0a6768d490806554307c2c851ea50b8cca9eaa5e824b510e3a808c81.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/08a93d1d0a6768d490806554307c2c851ea50b8cca9eaa5e824b510e3a808c81.jpg)

![0b591386301d82ad591e101df91b3458dde21f75e244f1c4e6abe47eed493b52.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/0b591386301d82ad591e101df91b3458dde21f75e244f1c4e6abe47eed493b52.jpg)

![0eacbfe352faf847b821edd36a15be3d316ab563487c2126c1f81f5c5ab7c6df.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/0eacbfe352faf847b821edd36a15be3d316ab563487c2126c1f81f5c5ab7c6df.jpg)

![1523bfa71d26290587cea23db21330a13b4cfb06c328f203889b3058943532d2.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/1523bfa71d26290587cea23db21330a13b4cfb06c328f203889b3058943532d2.jpg)

![16a26b58bac8cde0971dfbdc97cb032338eb514c59bae4207950069ab85e242c.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/16a26b58bac8cde0971dfbdc97cb032338eb514c59bae4207950069ab85e242c.jpg)

![1ebc517e2d277fb47dfa64d64c6a897fe4bb0f510888852905643efb9f8e8b1a.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/1ebc517e2d277fb47dfa64d64c6a897fe4bb0f510888852905643efb9f8e8b1a.jpg)

![2295280f655ffd58b41e9d6a2bc809944acba567a493df8a8a46185487499fcf.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/2295280f655ffd58b41e9d6a2bc809944acba567a493df8a8a46185487499fcf.jpg)

![2732de47d26afdde2c6af8e26bf2976e359739716b465fde10bbecba18e52d51.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/2732de47d26afdde2c6af8e26bf2976e359739716b465fde10bbecba18e52d51.jpg)

![38a4dca99bba2bc5a128ace996ec5034fd5d1c0f2ceff555ecc8bc30d25ae3a7.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/38a4dca99bba2bc5a128ace996ec5034fd5d1c0f2ceff555ecc8bc30d25ae3a7.jpg)

![3db1b7525ee64e11d56a522705b32169e53eb7e6ba06b6876393c00ee581ca57.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/3db1b7525ee64e11d56a522705b32169e53eb7e6ba06b6876393c00ee581ca57.jpg)

![4e4cb034919f7a4435985d867850804a423fc7ec909a3c8467c0851260bf495c.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/4e4cb034919f7a4435985d867850804a423fc7ec909a3c8467c0851260bf495c.jpg)

![6790ca1a6132893cb2ae8f8decd2ea4b4d02b3a320c319c42e48eeb14215e6cc.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/6790ca1a6132893cb2ae8f8decd2ea4b4d02b3a320c319c42e48eeb14215e6cc.jpg)

![7461561145f6e31f4824565278d22a0f047005499a4e337912cc03fb0d08ca11.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/7461561145f6e31f4824565278d22a0f047005499a4e337912cc03fb0d08ca11.jpg)

![80bd6c6fd8f76536ecddb0975a7093a4ecf94f5d60fb9ce785fc32cfa02dadab.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/80bd6c6fd8f76536ecddb0975a7093a4ecf94f5d60fb9ce785fc32cfa02dadab.jpg)

![9a496cd45cc6bb0477a968b677b449d5a3f25da196bfd1b788e3c64923249c64.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/9a496cd45cc6bb0477a968b677b449d5a3f25da196bfd1b788e3c64923249c64.jpg)

![a4fff55188da6b3285464ea611edd6d5e1d06e73e08aaf570ceaf58d8cd855c8.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/a4fff55188da6b3285464ea611edd6d5e1d06e73e08aaf570ceaf58d8cd855c8.jpg)

![a71e0477e30c5680535069eca82594617779b30de9284188b570f0046ff54bdc.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/a71e0477e30c5680535069eca82594617779b30de9284188b570f0046ff54bdc.jpg)

![a9a92106c779c0ef3f5f8c6e233ae19551b6a551ec8f3dab17f046a327c94d94.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/a9a92106c779c0ef3f5f8c6e233ae19551b6a551ec8f3dab17f046a327c94d94.jpg)

![adab3abe15af90889a0f13926b372d8f0b751fb31e15171bab2ce1a9596049f9.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/adab3abe15af90889a0f13926b372d8f0b751fb31e15171bab2ce1a9596049f9.jpg)

![c22eec485fd3db6b093949d32f3854bae848111352c333e2abf01c8b6fdae070.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/c22eec485fd3db6b093949d32f3854bae848111352c333e2abf01c8b6fdae070.jpg)

![c23f56fb837c7998077727123d73abdaca4d9e98d814f2e2f3a973c215fa3b6e.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/c23f56fb837c7998077727123d73abdaca4d9e98d814f2e2f3a973c215fa3b6e.jpg)

![c62b35bb70b5ee33c86cd4375ae8b987884009ffeee0f5953fdfcb5e773487c8.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/c62b35bb70b5ee33c86cd4375ae8b987884009ffeee0f5953fdfcb5e773487c8.jpg)

![c8e74dbda05109294837571abfee22f2f5fe448bc3d3e1950a68b6f42b9303c0.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/c8e74dbda05109294837571abfee22f2f5fe448bc3d3e1950a68b6f42b9303c0.jpg)

![d6be7778713d951060ace1b9250b521af04d5a29c39d8ad4518e60e05089c428.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/d6be7778713d951060ace1b9250b521af04d5a29c39d8ad4518e60e05089c428.jpg)

![e02ff72f911fe9f0db9775c383a092414f6aa7f619c74689342336453647bb9e.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/e02ff72f911fe9f0db9775c383a092414f6aa7f619c74689342336453647bb9e.jpg)

![ef31d08fb18ce56c6d57c5df4d2f2b02db0f3ac4a448371e7726152f68aac3af.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/ef31d08fb18ce56c6d57c5df4d2f2b02db0f3ac4a448371e7726152f68aac3af.jpg)

![febe35c627cf8268a7336e212f1a6d6392cb1fbb0bbb23c2a94556e4dd3453ec.jpg](../icml_results/6225_URS_%20A%20Unified%20Neural%20Routing%20Solver%20for%20Cross-Problem%20Zero-Shot%20Generalization/tables/febe35c627cf8268a7336e212f1a6d6392cb1fbb0bbb23c2a94556e4dd3453ec.jpg)

<a id="paper-6176"></a>

## Bayes-inspired Integration of Pretrained Priors and Few-Shot Evidence for Few-Shot Classification


### Images

![5a30eb0ea22af3243e67c2997effa5a6d92bbc0bc1503bff632234d7137409e8.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/images/5a30eb0ea22af3243e67c2997effa5a6d92bbc0bc1503bff632234d7137409e8.jpg)

![a4c1d7ca4e4e3438f89bb716d43b7a7a2be64d1c9b6116dbe6b5b23244f56706.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/images/a4c1d7ca4e4e3438f89bb716d43b7a7a2be64d1c9b6116dbe6b5b23244f56706.jpg)

![bff3c0ca21131191a189d04e80fae3d953fbc3f450c323df97f78fe96f0d6418.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/images/bff3c0ca21131191a189d04e80fae3d953fbc3f450c323df97f78fe96f0d6418.jpg)

![c77d2b3576282c30344f6c302f2b10b3e099d5d058033ff165777e1be5400c60.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/images/c77d2b3576282c30344f6c302f2b10b3e099d5d058033ff165777e1be5400c60.jpg)

### Tables

![28958fbb7bb7ccbfde50fd60f3041c59cfb9988c12b8da07449e144a5b2007e5.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/28958fbb7bb7ccbfde50fd60f3041c59cfb9988c12b8da07449e144a5b2007e5.jpg)

![3a4b3e9263fa0792bcbf4d911e7d43468e2cf210aa3e986ab861ecc4b6d0f1ed.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/3a4b3e9263fa0792bcbf4d911e7d43468e2cf210aa3e986ab861ecc4b6d0f1ed.jpg)

![5852aca12a3c3f76d02f6b171d1a022bc6329b2a4a337d5566a8f7fe6f493bdf.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/5852aca12a3c3f76d02f6b171d1a022bc6329b2a4a337d5566a8f7fe6f493bdf.jpg)

![5bfcb532323f3220401da90b29d3790bbe29f9305bfa8fe1dbffdb0a0446ca81.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/5bfcb532323f3220401da90b29d3790bbe29f9305bfa8fe1dbffdb0a0446ca81.jpg)

![a82ca14a5933f7f0c6d1e109cae712d51a9425f556197df3f067b8c08beba9fa.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/a82ca14a5933f7f0c6d1e109cae712d51a9425f556197df3f067b8c08beba9fa.jpg)

![c43374b9b86f0c79d1b2a09cd7f4640f16fc9e04b1585dcb2b3fea2b7884ae3d.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/c43374b9b86f0c79d1b2a09cd7f4640f16fc9e04b1585dcb2b3fea2b7884ae3d.jpg)

![e362bae905cb8f7856f77eed4124976d1e5d113c96fb5d75723226fbbe2f1e2c.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/e362bae905cb8f7856f77eed4124976d1e5d113c96fb5d75723226fbbe2f1e2c.jpg)

![e721ad2d1a8de173abb9dd243cf19cf9f5511e894f1f8666698acc203f55f003.jpg](../icml_results/6226_Bayes-inspired%20Integration%20of%20Pretrained%20Priors%20and%20Few-Shot%20Evidence%20for%20Few-Shot%20Classification/tables/e721ad2d1a8de173abb9dd243cf19cf9f5511e894f1f8666698acc203f55f003.jpg)

<a id="paper-6177"></a>

## Statistically Calibrated Scaling for Token Merging in Transformers


### Images

![0aa2ce80e576848c50735b0f42eacd554ef42fb69627268df20e36f99f1cb2f9.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/0aa2ce80e576848c50735b0f42eacd554ef42fb69627268df20e36f99f1cb2f9.jpg)

![0fbe121cacc4369dbcaa6cc25c2602b289833c3195466c05757ebc02dd7f1bbe.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/0fbe121cacc4369dbcaa6cc25c2602b289833c3195466c05757ebc02dd7f1bbe.jpg)

![6d0e83d604028b1106fd8b9a091bd1a9f19545e4f874d9d0e1fa4b0577a1cc27.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/6d0e83d604028b1106fd8b9a091bd1a9f19545e4f874d9d0e1fa4b0577a1cc27.jpg)

![7982dfb98f7b734d70860f7de08995dec588ed0844070bf2c4b85911d3d0d87c.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/7982dfb98f7b734d70860f7de08995dec588ed0844070bf2c4b85911d3d0d87c.jpg)

![84cdea3eb10ab319b26d4d9cf777e11a5c4fec67e3079c13da2ed7f924695770.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/84cdea3eb10ab319b26d4d9cf777e11a5c4fec67e3079c13da2ed7f924695770.jpg)

![cb38ee33f0803bc4c08ae4c0dc77b0be8670a6bab10dad3fef168a4b4d489bf5.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/cb38ee33f0803bc4c08ae4c0dc77b0be8670a6bab10dad3fef168a4b4d489bf5.jpg)

![ce3cbd37925cf3c241779a8984ad6b6ff4c0868d2391cd617fa7998a4064c0eb.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/ce3cbd37925cf3c241779a8984ad6b6ff4c0868d2391cd617fa7998a4064c0eb.jpg)

![e67566c6914bb46f71e35326b4f512d33141e583712a1151e832be6dc1fafcf4.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/images/e67566c6914bb46f71e35326b4f512d33141e583712a1151e832be6dc1fafcf4.jpg)

### Tables

![43286db616bc4f7aa7b7d33421cdf527f1b19455667084681266bf188933d271.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/tables/43286db616bc4f7aa7b7d33421cdf527f1b19455667084681266bf188933d271.jpg)

![4932cd8d10fc6af7738cae92b51c00c824fc05a7ae6306e0adbf2a73526f136b.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/tables/4932cd8d10fc6af7738cae92b51c00c824fc05a7ae6306e0adbf2a73526f136b.jpg)

![b39f49242f8e8a99fd905942b53dffd6072effdfe2df66f8bbe277585e393848.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/tables/b39f49242f8e8a99fd905942b53dffd6072effdfe2df66f8bbe277585e393848.jpg)

![c3023a1946d17314e6527025f76c41f65c06e20f216a4be05bd17357e2fa2817.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/tables/c3023a1946d17314e6527025f76c41f65c06e20f216a4be05bd17357e2fa2817.jpg)

![fa815ea68d4fafa089f35d831d5ba4499178577495705141eed16575eb7426bb.jpg](../icml_results/6227_Statistically%20Calibrated%20Scaling%20for%20Token%20Merging%20in%20Transformers/tables/fa815ea68d4fafa089f35d831d5ba4499178577495705141eed16575eb7426bb.jpg)

<a id="paper-6178"></a>

## HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling


### Images

![41d820525ae97e0684841420708e7eca7237255f29fbc50ff2635c0d80097fad.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/images/41d820525ae97e0684841420708e7eca7237255f29fbc50ff2635c0d80097fad.jpg)

![7eeadc55da07221262ed7016bde83c402b0d281b3665c644f809022f699cae26.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/images/7eeadc55da07221262ed7016bde83c402b0d281b3665c644f809022f699cae26.jpg)

![ea3cb458824307e97d41c272b1419bb3961d0d4f338d2c28781a94df8e823cce.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/images/ea3cb458824307e97d41c272b1419bb3961d0d4f338d2c28781a94df8e823cce.jpg)

![fc8b26b016cf015d3b76ec7364904f69725e16567e93f8c57dc307ea157a37bf.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/images/fc8b26b016cf015d3b76ec7364904f69725e16567e93f8c57dc307ea157a37bf.jpg)

### Tables

![283d8c61d423e81ee5a05c50006f1d0092972b0ba1f4525749307a0817864ca9.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/283d8c61d423e81ee5a05c50006f1d0092972b0ba1f4525749307a0817864ca9.jpg)

![613ab1ec21491e3fb3c307397762c18fe76780dfa6870f4ddd0249eb3a7a3c63.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/613ab1ec21491e3fb3c307397762c18fe76780dfa6870f4ddd0249eb3a7a3c63.jpg)

![7cf40ca83b7289bc2a2d440bd32cb5529ea5ce5c8883b4bb4f7ca4a7929d6cab.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/7cf40ca83b7289bc2a2d440bd32cb5529ea5ce5c8883b4bb4f7ca4a7929d6cab.jpg)

![8ebcce281db6461ef89217f4244be1a3accee1213b83f7ff40eb8a09efbc688f.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/8ebcce281db6461ef89217f4244be1a3accee1213b83f7ff40eb8a09efbc688f.jpg)

![a92f4dfa85ececfbdc09fdbbd65892764e5eee067330e288a4c514d559123377.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/a92f4dfa85ececfbdc09fdbbd65892764e5eee067330e288a4c514d559123377.jpg)

![dbfc231e6d51b7aca8c020bb7b492ed21baf961ac191c4f377ee8a4dd23febd4.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/dbfc231e6d51b7aca8c020bb7b492ed21baf961ac191c4f377ee8a4dd23febd4.jpg)

![e5e59f05b21b83401eab3e6c7b256beca4036ec14ad2e768f383e68eb0fb10e2.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/e5e59f05b21b83401eab3e6c7b256beca4036ec14ad2e768f383e68eb0fb10e2.jpg)

![f8bda1fd3deeb282baf11505271788a238b43b5012f2abec1247f672dbcd848e.jpg](../icml_results/6228_HGMem_%20Hypergraph-based%20Working%20Memory%20to%20Improve%20Multi-step%20RAG%20for%20Long-Context%20Complex%20Relational/tables/f8bda1fd3deeb282baf11505271788a238b43b5012f2abec1247f672dbcd848e.jpg)

<a id="paper-6179"></a>

## BAS: Bridging Adam and SignSGD for Memory-Efficient LLM Training


### Images

![01b8a7014292b99133545cc41f129d2874f048b80ced653d5b12bcee02f95f7a.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/01b8a7014292b99133545cc41f129d2874f048b80ced653d5b12bcee02f95f7a.jpg)

![0826a887c7484866c6906fc11f0e7644c786c73ac4d74dc87887756a29428c0d.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/0826a887c7484866c6906fc11f0e7644c786c73ac4d74dc87887756a29428c0d.jpg)

![1be6d063dbbcec850e7976c8f68b9da251e8609d31c305f803a934e82e1d5473.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/1be6d063dbbcec850e7976c8f68b9da251e8609d31c305f803a934e82e1d5473.jpg)

![23dffb1690968fcb530ca6b08448ba13896609be10ce22b494beb54c81d3a37a.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/23dffb1690968fcb530ca6b08448ba13896609be10ce22b494beb54c81d3a37a.jpg)

![89fbc5929651e0114738920f2fea3dc676d2e2706f1284555171ba6d8d883310.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/89fbc5929651e0114738920f2fea3dc676d2e2706f1284555171ba6d8d883310.jpg)

![8a99b77ff0920ed10cd12df9b26cce36b3f4ae8e40cb3da5d461883e9554c909.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/8a99b77ff0920ed10cd12df9b26cce36b3f4ae8e40cb3da5d461883e9554c909.jpg)

![c4b2c55c8320dffbbd800c76c82e08694eed7df496da47e281344c7c1f1b1424.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/images/c4b2c55c8320dffbbd800c76c82e08694eed7df496da47e281344c7c1f1b1424.jpg)

### Tables

![04e0620824f13f090e2ef3df08f44dc697e9e4806486c55561ba488e44a14e31.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/04e0620824f13f090e2ef3df08f44dc697e9e4806486c55561ba488e44a14e31.jpg)

![0fc6ed1c9955ca44a781c178676684a670820c95f19ed244c8828bfcb7ee3a44.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/0fc6ed1c9955ca44a781c178676684a670820c95f19ed244c8828bfcb7ee3a44.jpg)

![4ba53a7ed4a80ae259c0a197b719081abdf4f8a4ec1b2ca3e7f266b0f3620d63.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/4ba53a7ed4a80ae259c0a197b719081abdf4f8a4ec1b2ca3e7f266b0f3620d63.jpg)

![53c69558834a90686157d6782f3cc23016c111ce4ded1d901832b54a6c69db55.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/53c69558834a90686157d6782f3cc23016c111ce4ded1d901832b54a6c69db55.jpg)

![80da06622d8c2e9b16371dd00869cf5637b7e903d245539a9f9fcdc709307277.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/80da06622d8c2e9b16371dd00869cf5637b7e903d245539a9f9fcdc709307277.jpg)

![842b23df9cf06d09e59f28e97fbf1eb54a8dc803dceb7991a4ffc71d663c025f.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/842b23df9cf06d09e59f28e97fbf1eb54a8dc803dceb7991a4ffc71d663c025f.jpg)

![ab54560afaef872343adc989aec0cb7c4b8071ec41ff5f5db1ebc6d01f9233de.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/ab54560afaef872343adc989aec0cb7c4b8071ec41ff5f5db1ebc6d01f9233de.jpg)

![abc00e4c8fb52966a67906043ec78ee287c8190f87d7dbbdf52f75ccf4d39c28.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/abc00e4c8fb52966a67906043ec78ee287c8190f87d7dbbdf52f75ccf4d39c28.jpg)

![d685d36ee5fff5e92986ad697168429593074c21c69cfe832bfa3210544b1a8d.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/d685d36ee5fff5e92986ad697168429593074c21c69cfe832bfa3210544b1a8d.jpg)

![fa047029839155c302c9552defc038bb03e71b919f230a256f8112ee46295a89.jpg](../icml_results/6229_BAS_%20Bridging%20Adam%20and%20SignSGD%20for%20Memory-Efficient%20LLM%20Training/tables/fa047029839155c302c9552defc038bb03e71b919f230a256f8112ee46295a89.jpg)

<a id="paper-6180"></a>

## AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning


### Images

![29110291d301ac46195924dc6859db333fd3f2ab72c88d0ef66c41ead48d4593.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/29110291d301ac46195924dc6859db333fd3f2ab72c88d0ef66c41ead48d4593.jpg)

![3a05c29a1800dadf4a9e3f48d565ee8bd0290973946fb07fedb51e93ff219f11.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/3a05c29a1800dadf4a9e3f48d565ee8bd0290973946fb07fedb51e93ff219f11.jpg)

![47204c8c67af501e5ec3171145355a3ad7c4a1a99c0ec0178bfdb5dcf2f08370.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/47204c8c67af501e5ec3171145355a3ad7c4a1a99c0ec0178bfdb5dcf2f08370.jpg)

![7cfe2fb2a0b381929a3ec3d765c3e6e5fa4c7377829dc2fec69d33ac30b7f2bc.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/7cfe2fb2a0b381929a3ec3d765c3e6e5fa4c7377829dc2fec69d33ac30b7f2bc.jpg)

![837b93ae95ee604c02029c4b95d3019a14402e5463ee0edf34aa48bb85f2950f.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/837b93ae95ee604c02029c4b95d3019a14402e5463ee0edf34aa48bb85f2950f.jpg)

![9875b9dce9b302d7688d2ecd8d2844ba0266d27e5891aa600e90a62df585de94.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/9875b9dce9b302d7688d2ecd8d2844ba0266d27e5891aa600e90a62df585de94.jpg)

![b7e0d4a086fca323f5b10e668cf64e7761cf9f6a3c7f8473f6e1c25f3deb8ca5.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/b7e0d4a086fca323f5b10e668cf64e7761cf9f6a3c7f8473f6e1c25f3deb8ca5.jpg)

![c53a8ad6d0a533a62df4f16cb979073b17440d34b897784fddad52fa3f64faa7.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/c53a8ad6d0a533a62df4f16cb979073b17440d34b897784fddad52fa3f64faa7.jpg)

![cce729ae615aa72237e8a171bf81368942bd2298fb7f734f25cd2d709cdf9ba0.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/cce729ae615aa72237e8a171bf81368942bd2298fb7f734f25cd2d709cdf9ba0.jpg)

![fba0edd07288f339f88eee8ee15cd3b9034e63f16c630f8ef598ae65c82e0aa3.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/images/fba0edd07288f339f88eee8ee15cd3b9034e63f16c630f8ef598ae65c82e0aa3.jpg)

### Tables

![0b87fcf366b015166e83336f5dcd9f70373ff6f21ab2a9d3dc28f719de48c2e2.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/0b87fcf366b015166e83336f5dcd9f70373ff6f21ab2a9d3dc28f719de48c2e2.jpg)

![0def40d9588076744585a9420fd21017506026545f7068f2b20a3a68a89eab14.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/0def40d9588076744585a9420fd21017506026545f7068f2b20a3a68a89eab14.jpg)

![0e831239c91149bad6e25cb2e7d44261fd766ab795ac7c7470b8ddeaf070dd9d.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/0e831239c91149bad6e25cb2e7d44261fd766ab795ac7c7470b8ddeaf070dd9d.jpg)

![14dd5eec346c143b59b815c1df4b6fb6e98108c28e3bde47306ad44988b615a5.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/14dd5eec346c143b59b815c1df4b6fb6e98108c28e3bde47306ad44988b615a5.jpg)

![2bf42a47d1f185e7865b378a63af209b8aae33f887ab3dbd3b38e622dfcb9c85.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/2bf42a47d1f185e7865b378a63af209b8aae33f887ab3dbd3b38e622dfcb9c85.jpg)

![52c2c0159bc5d40cc0cf725d7d2679ac833262ee287eb1d2c5d4d6ea9f72ee9c.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/52c2c0159bc5d40cc0cf725d7d2679ac833262ee287eb1d2c5d4d6ea9f72ee9c.jpg)

![7f8460f15edb1afb618782db1083984780da96407685fd4c3274fc2ec06c79eb.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/7f8460f15edb1afb618782db1083984780da96407685fd4c3274fc2ec06c79eb.jpg)

![cedadc279ffacdc5d971e41c78d86ab4376af5c9c7ccac600bdc86f4694615ba.jpg](../icml_results/6230_AutoQRA_%20Joint%20Optimization%20of%20Mixed-Precision%20Quantization%20and%20Low-rank%20Adapters%20for%20Efficient%20LLM%20/tables/cedadc279ffacdc5d971e41c78d86ab4376af5c9c7ccac600bdc86f4694615ba.jpg)

<a id="paper-6181"></a>

## Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting


### Images

![0865cf32f05bd19492656edc234ca165bd220d4dce4c6de46e564381d623447a.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/0865cf32f05bd19492656edc234ca165bd220d4dce4c6de46e564381d623447a.jpg)

![12dcb784ac5f1b7959cf151435e806b35f0761be406c1c90520bcb1ac59f4562.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/12dcb784ac5f1b7959cf151435e806b35f0761be406c1c90520bcb1ac59f4562.jpg)

![1facb3bb713b5947ae9531db06cb179694c814d8c25686a8f5d93ddc8394728a.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/1facb3bb713b5947ae9531db06cb179694c814d8c25686a8f5d93ddc8394728a.jpg)

![27d862a6e3f6f30c14343cd1018fd6eb9e5572dfacd4d35a1ee720599f779116.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/27d862a6e3f6f30c14343cd1018fd6eb9e5572dfacd4d35a1ee720599f779116.jpg)

![7706855e98bf2be928e56639254491406e01222f763d07bf64edafba53f20978.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/7706855e98bf2be928e56639254491406e01222f763d07bf64edafba53f20978.jpg)

![7ed904878ef416d1407279104c68018da43f4d35f16c9263fcb15570de2489ed.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/7ed904878ef416d1407279104c68018da43f4d35f16c9263fcb15570de2489ed.jpg)

![9ed03edd6478dcf9721b0ae0388aa14d9c202cbe1311282f735f6321582e8e86.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/9ed03edd6478dcf9721b0ae0388aa14d9c202cbe1311282f735f6321582e8e86.jpg)

![a6e6ec5613e3324d063d11c4816c311c2467d6fb3ca13b4e51b367924a4a2db2.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/a6e6ec5613e3324d063d11c4816c311c2467d6fb3ca13b4e51b367924a4a2db2.jpg)

![b3f7a030794b3723ea3587e3a0ce08b936f8e68994a39d38d5214dec847b34a3.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/b3f7a030794b3723ea3587e3a0ce08b936f8e68994a39d38d5214dec847b34a3.jpg)

![d9ed1b98c3b8a88a0e745342da38b95d0e0b57826134059e43e0fc7d8b92b39a.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/images/d9ed1b98c3b8a88a0e745342da38b95d0e0b57826134059e43e0fc7d8b92b39a.jpg)

### Tables

![11fca080089ec8a968f35c2f175944e853b3c3a01a8c39946795b97cdcf40c30.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/11fca080089ec8a968f35c2f175944e853b3c3a01a8c39946795b97cdcf40c30.jpg)

![1746039241a66d8cce41890edb83dfcfd1f28c5c1790929a4258dbe873e8309e.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/1746039241a66d8cce41890edb83dfcfd1f28c5c1790929a4258dbe873e8309e.jpg)

![23085eccd5c80ee763476254105719aa47e00a986ffdbcc1ba34fa0018c638c0.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/23085eccd5c80ee763476254105719aa47e00a986ffdbcc1ba34fa0018c638c0.jpg)

![2485651686306e1b1ea679971b28f1b5c0ab62be6044cde27613c70cc2519363.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/2485651686306e1b1ea679971b28f1b5c0ab62be6044cde27613c70cc2519363.jpg)

![2891b7e0371c2a48d36a389a39bd6b891f74d03968bd0d9c5e0ee994c5c2ebd9.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/2891b7e0371c2a48d36a389a39bd6b891f74d03968bd0d9c5e0ee994c5c2ebd9.jpg)

![53d75fbe8b4f20a86455107a64aab59877c3bceea631174a4623688871f7015d.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/53d75fbe8b4f20a86455107a64aab59877c3bceea631174a4623688871f7015d.jpg)

![5b54cea5a4d58525a604279d154f7ced504e4c6493943c60434572f231a7ec60.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/5b54cea5a4d58525a604279d154f7ced504e4c6493943c60434572f231a7ec60.jpg)

![65fabf7c4d80f26cac7067c6067bac4a693b64fe34f69b569f51330a2456b3f7.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/65fabf7c4d80f26cac7067c6067bac4a693b64fe34f69b569f51330a2456b3f7.jpg)

![81b2ba94f72638281509bc96a0bd2f60480b6df025b608cdfe2396aff37e6613.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/81b2ba94f72638281509bc96a0bd2f60480b6df025b608cdfe2396aff37e6613.jpg)

![be44d072835cfd20cbcedbc325d9e0b59680a48356728652593ee704a9943d68.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/be44d072835cfd20cbcedbc325d9e0b59680a48356728652593ee704a9943d68.jpg)

![bfa78b5b03a64ea7982cb16e85b87e6ac3c73b70413e827ff43c342561969a74.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/bfa78b5b03a64ea7982cb16e85b87e6ac3c73b70413e827ff43c342561969a74.jpg)

![d069387bd800dfbcf999817fa114caa8039c8f9e1a54f94d469bf044ead85802.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/d069387bd800dfbcf999817fa114caa8039c8f9e1a54f94d469bf044ead85802.jpg)

![d24cfc7c760215335315ce56737c54610a59cce569be7adfc5afc9d9d985ad2c.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/d24cfc7c760215335315ce56737c54610a59cce569be7adfc5afc9d9d985ad2c.jpg)

![e60ee209d5c995366b033c55abb9833e3cfa658d59b40a995ac4b0730c931c0c.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/e60ee209d5c995366b033c55abb9833e3cfa658d59b40a995ac4b0730c931c0c.jpg)

![f49633b190c150aff8889b495c4a6543813705da137d116853115dc5316ce283.jpg](../icml_results/6231_Learning%20to%20Refine_%20Spectral-Decoupled%20Iterative%20Refinement%20Framework%20for%20Precipitation%20Nowcasting/tables/f49633b190c150aff8889b495c4a6543813705da137d116853115dc5316ce283.jpg)

<a id="paper-6182"></a>

## Demystifying Mergeability: Interpretable Properties to Predict Model Merging Success


### Images

![3d8b8efeffdaaa0c0fe640a6b608365e481385d76c65cff908d1fa7a8e576dec.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/3d8b8efeffdaaa0c0fe640a6b608365e481385d76c65cff908d1fa7a8e576dec.jpg)

![4eaa3dcd67901f505537b4225614c1c08d23f31170bce123a585db03803d7fea.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/4eaa3dcd67901f505537b4225614c1c08d23f31170bce123a585db03803d7fea.jpg)

![75455c0bedb7cd03c8429081c6863cc6e9a277ef804175ffff96ef2480e71758.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/75455c0bedb7cd03c8429081c6863cc6e9a277ef804175ffff96ef2480e71758.jpg)

![81bfab2be0bd0872295f7cdb900221667c187b03b9a77961cc4419f8fc7352c9.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/81bfab2be0bd0872295f7cdb900221667c187b03b9a77961cc4419f8fc7352c9.jpg)

![c8688b0a2bf43bcb1f00a52b4bcd2c4444c9edcb564a508aba6223a4bd2a26c4.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/c8688b0a2bf43bcb1f00a52b4bcd2c4444c9edcb564a508aba6223a4bd2a26c4.jpg)

![e379160d00bbc0ed9fedcb9fc8e4890da5af8eb49bb61b08d5d1f11fab8af95d.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/images/e379160d00bbc0ed9fedcb9fc8e4890da5af8eb49bb61b08d5d1f11fab8af95d.jpg)

### Tables

![14e6f1f72420ca39d2361baf25c7a60554ca7119811be6025d9c34073662ed50.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/14e6f1f72420ca39d2361baf25c7a60554ca7119811be6025d9c34073662ed50.jpg)

![1e5de0b3cfef9bb479e0d8777a20df17e9095618db36301d197971b7ed440b3e.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/1e5de0b3cfef9bb479e0d8777a20df17e9095618db36301d197971b7ed440b3e.jpg)

![31ee48d7151852f70a60a3d507598d14a0e8aa2daf1532a67d529579e3fcd7a1.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/31ee48d7151852f70a60a3d507598d14a0e8aa2daf1532a67d529579e3fcd7a1.jpg)

![361cad59b3cc17c364dfbd48bb0f2ada78239a173966d40cc02e4659e2fc9a63.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/361cad59b3cc17c364dfbd48bb0f2ada78239a173966d40cc02e4659e2fc9a63.jpg)

![3d06b65c5e5ae0a29cbe1188143b31dee0e8355deb9778a5dc399d12b3e876b1.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/3d06b65c5e5ae0a29cbe1188143b31dee0e8355deb9778a5dc399d12b3e876b1.jpg)

![3d3d8471ef2431551ecdb8bbb0a4bba67807636448f3769d2d901265b3ce5d09.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/3d3d8471ef2431551ecdb8bbb0a4bba67807636448f3769d2d901265b3ce5d09.jpg)

![41e1e4583eea17b72455858bf4def827ba260b85b94be1de8a141e34e4d3233a.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/41e1e4583eea17b72455858bf4def827ba260b85b94be1de8a141e34e4d3233a.jpg)

![482e2d5574557334cbb8c41cd3db0525f18a8939a41863a7604fcccf8b50dc03.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/482e2d5574557334cbb8c41cd3db0525f18a8939a41863a7604fcccf8b50dc03.jpg)

![4d8fcd2ce5564f1c9da1ae0b67aa88311e0b0b534f65b6c1e196d00b80a2437e.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/4d8fcd2ce5564f1c9da1ae0b67aa88311e0b0b534f65b6c1e196d00b80a2437e.jpg)

![50c8bdad86afea9557d224e96de4220b252113fc69126e268105b9ba2bb18b90.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/50c8bdad86afea9557d224e96de4220b252113fc69126e268105b9ba2bb18b90.jpg)

![57415f0c4ff360a730eb75cf1986753f155822ee52376d34e5cdce127dfcfb94.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/57415f0c4ff360a730eb75cf1986753f155822ee52376d34e5cdce127dfcfb94.jpg)

![81b7bfba0659468706c94ef0e0fce307b81ce00f7c1b6cd543590252a112be85.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/81b7bfba0659468706c94ef0e0fce307b81ce00f7c1b6cd543590252a112be85.jpg)

![84401a61716a9174aaea229a20d233528c3eb68ed61f9aa5580eaa9c7f87e072.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/84401a61716a9174aaea229a20d233528c3eb68ed61f9aa5580eaa9c7f87e072.jpg)

![85e1c2d4cb87640b6a625015f03953c387f8aef3519feb1714e7380efbda09b0.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/85e1c2d4cb87640b6a625015f03953c387f8aef3519feb1714e7380efbda09b0.jpg)

![8b73109044ae95636f783ab12b252534836d526c2c7e70b384be9f95d39c3c52.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/8b73109044ae95636f783ab12b252534836d526c2c7e70b384be9f95d39c3c52.jpg)

![912aa05129955ca56aee12ed939633377b52dfd7bc92bca95fed06331e76cca9.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/912aa05129955ca56aee12ed939633377b52dfd7bc92bca95fed06331e76cca9.jpg)

![959e61560374a8ed300685408a8da0192765930f2f74ddf06aed1ad21e5dedb5.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/959e61560374a8ed300685408a8da0192765930f2f74ddf06aed1ad21e5dedb5.jpg)

![981b07b7c11b2f2b9ea56b1059d73a5cf1a7427761498373898b2f42240ac68a.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/981b07b7c11b2f2b9ea56b1059d73a5cf1a7427761498373898b2f42240ac68a.jpg)

![98fc2182f220c0a454705f022a9f80bdaebb301409bb7653860558d8329f9a04.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/98fc2182f220c0a454705f022a9f80bdaebb301409bb7653860558d8329f9a04.jpg)

![9b0988fd23e2eaa77ef50d893141e61caed5ad02d33261f980ea59e1d6667591.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/9b0988fd23e2eaa77ef50d893141e61caed5ad02d33261f980ea59e1d6667591.jpg)

![a319ff7be042b0f25f98318d0ec5b359177665632af018e2fc6b91cdb1769ec2.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/a319ff7be042b0f25f98318d0ec5b359177665632af018e2fc6b91cdb1769ec2.jpg)

![c0b5f86de2c2e43edf8565baeb6539f1946eed2ba02635effcda47d2087527a9.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/c0b5f86de2c2e43edf8565baeb6539f1946eed2ba02635effcda47d2087527a9.jpg)

![c9fd75eeb9023a0382b2cc23d7e6198aea922a070e27dba023d530d215dc8aeb.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/c9fd75eeb9023a0382b2cc23d7e6198aea922a070e27dba023d530d215dc8aeb.jpg)

![ce83aedc28569bfe5d56337e749202f219c44eba2b5cdc0de4bbeb53cbc4d982.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/ce83aedc28569bfe5d56337e749202f219c44eba2b5cdc0de4bbeb53cbc4d982.jpg)

![e01e74eda6707862feff97a674b2c510561e18c619e7ca1b4db456b6b03ed9a6.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/e01e74eda6707862feff97a674b2c510561e18c619e7ca1b4db456b6b03ed9a6.jpg)

![fe9e4735b3bb4ced937bcf7bcb8512f19dcca3cf43483585520f5115a422a390.jpg](../icml_results/6232_Demystifying%20Mergeability_%20Interpretable%20Properties%20to%20Predict%20Model%20Merging%20Success/tables/fe9e4735b3bb4ced937bcf7bcb8512f19dcca3cf43483585520f5115a422a390.jpg)

<a id="paper-6183"></a>

## Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization


### Images

![2ad23d1da1260a48dc731f8a8242fa62d034701307aca0caa86539bf17d18169.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/images/2ad23d1da1260a48dc731f8a8242fa62d034701307aca0caa86539bf17d18169.jpg)

![480ce745adc3c306773c1a24ee65b2a5009c3fffb0a617663e3fd9b12cffe97c.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/images/480ce745adc3c306773c1a24ee65b2a5009c3fffb0a617663e3fd9b12cffe97c.jpg)

![571810d9cd9f5349744c2856fe5b5a96aaa04430540982e40d2f92b8b3077a0e.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/images/571810d9cd9f5349744c2856fe5b5a96aaa04430540982e40d2f92b8b3077a0e.jpg)

![5982cebebd0d91772c16b52675b8f7d244f3aae63df0435b65a212552342dcf6.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/images/5982cebebd0d91772c16b52675b8f7d244f3aae63df0435b65a212552342dcf6.jpg)

![b813eac18c0067180c8c2b380e8393f165f09d360b5f4dfa973702d9d1f1e8d7.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/images/b813eac18c0067180c8c2b380e8393f165f09d360b5f4dfa973702d9d1f1e8d7.jpg)

### Tables

![006ec89be55d87c296cba871c65d4cbdc6cf544f27e2cf631a26d79d688195d7.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/006ec89be55d87c296cba871c65d4cbdc6cf544f27e2cf631a26d79d688195d7.jpg)

![152d526b1f5876cbf88df616b5d98f8042e3f84def1c852eea2cc8e5b54737c4.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/152d526b1f5876cbf88df616b5d98f8042e3f84def1c852eea2cc8e5b54737c4.jpg)

![24e16a3f7809c70e8bc31165a93b580e52cb6ddbdb2e3663242058d960160e2d.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/24e16a3f7809c70e8bc31165a93b580e52cb6ddbdb2e3663242058d960160e2d.jpg)

![25458dbd60a17bfb092deebfec74e1a2433ad4113b50c5ee187af07bb74eaa74.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/25458dbd60a17bfb092deebfec74e1a2433ad4113b50c5ee187af07bb74eaa74.jpg)

![4c152be979e6cf9fdeeb443fcc23c69af195f25d5647f22dd2d7000b103c90ac.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/4c152be979e6cf9fdeeb443fcc23c69af195f25d5647f22dd2d7000b103c90ac.jpg)

![6843ba11fe11518e59ec4e2a3d2d39be92e4e48fd2637a9f12f3532e90c54222.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/6843ba11fe11518e59ec4e2a3d2d39be92e4e48fd2637a9f12f3532e90c54222.jpg)

![6c15560b68e533907869ce36242aad0fad81325661f6894e474a4c72ce00ec6c.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/6c15560b68e533907869ce36242aad0fad81325661f6894e474a4c72ce00ec6c.jpg)

![745c18f749754c1d1249148f78c1c7f8fe67b162e57b27b17cbf1b53720ca267.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/745c18f749754c1d1249148f78c1c7f8fe67b162e57b27b17cbf1b53720ca267.jpg)

![840f3f037168f9b5b8695103f80f089a056c55faeba79670380abd09e0e82115.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/840f3f037168f9b5b8695103f80f089a056c55faeba79670380abd09e0e82115.jpg)

![8ee3a84e8845fc25721588662e00efd2f39b6a32006ec13203abb251e340e2a3.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/8ee3a84e8845fc25721588662e00efd2f39b6a32006ec13203abb251e340e2a3.jpg)

![a310dc1ea85097b28538110bf9556bf947919542bf52741fb14569b0f4653a15.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/a310dc1ea85097b28538110bf9556bf947919542bf52741fb14569b0f4653a15.jpg)

![a406de33d9aeb07c05f53332dcc010d23193bdc0fb8603a701e3679432a7be85.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/a406de33d9aeb07c05f53332dcc010d23193bdc0fb8603a701e3679432a7be85.jpg)

![d46c48493d7c49822e6a194fbbfc723ece80855e80655f7684be172593bb0d38.jpg](../icml_results/6233_Metis_%20Learning%20to%20Jailbreak%20LLMs%20via%20Self-Evolving%20Metacognitive%20Policy%20Optimization/tables/d46c48493d7c49822e6a194fbbfc723ece80855e80655f7684be172593bb0d38.jpg)

<a id="paper-6184"></a>

## How Language Models Process Negation


### Images

![44d33d20f1a36f886f91c54e4b6941fcf90fc663f61139274a0554a9bfbd01a1.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/44d33d20f1a36f886f91c54e4b6941fcf90fc663f61139274a0554a9bfbd01a1.jpg)

![4d7619547a17362cecaf6c78ff747a8036827e8b0b1ec159af3c1acec034aca0.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/4d7619547a17362cecaf6c78ff747a8036827e8b0b1ec159af3c1acec034aca0.jpg)

![655f270fc3a0b933e1a2d3535bb2dd6465e6f98b7a6747383651df7b42cd1097.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/655f270fc3a0b933e1a2d3535bb2dd6465e6f98b7a6747383651df7b42cd1097.jpg)

![7bb0d1d0a7ec2a6c1505bc3393581dbc95482156dfde5475e52e88686526da44.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/7bb0d1d0a7ec2a6c1505bc3393581dbc95482156dfde5475e52e88686526da44.jpg)

![7ff19666bee6749345f11999ca526500440adfb4a606c360820c7ba5a353f63c.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/7ff19666bee6749345f11999ca526500440adfb4a606c360820c7ba5a353f63c.jpg)

![8f244cc398ab028d0e47d4f52a87fadc30308f9599b7d02265187e52a2150f95.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/8f244cc398ab028d0e47d4f52a87fadc30308f9599b7d02265187e52a2150f95.jpg)

![98635f6cf98be4e1ac2b28036ccb9e6e6f4dd3d313ee744aae22426d4e5a90e4.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/98635f6cf98be4e1ac2b28036ccb9e6e6f4dd3d313ee744aae22426d4e5a90e4.jpg)

![a604eb22eba6a4aecbed0f5eb3f80db9dcfbcee847e62e715dbd295dfbef65a9.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/a604eb22eba6a4aecbed0f5eb3f80db9dcfbcee847e62e715dbd295dfbef65a9.jpg)

![bb590499c772b0ad0c09a008fb27f275391e8a38ed09c1b897072ebd32aeca8b.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/bb590499c772b0ad0c09a008fb27f275391e8a38ed09c1b897072ebd32aeca8b.jpg)

![c2e19735ee31bb65429363b162d81ee817c59dfc9f0d2517efc078214d3467f2.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/c2e19735ee31bb65429363b162d81ee817c59dfc9f0d2517efc078214d3467f2.jpg)

![c953a608e9e9680b424bd938444daa7b8b6d9c8946a836c48c8194f1006a5f74.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/c953a608e9e9680b424bd938444daa7b8b6d9c8946a836c48c8194f1006a5f74.jpg)

![daa887db99a41ef6febef4c475299f7368104b4982c811a2e58d4adf4cd660df.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/daa887db99a41ef6febef4c475299f7368104b4982c811a2e58d4adf4cd660df.jpg)

![dd514e62cba829f1d05862a0434df8b85594f4a25fe9e1bcf9ba3811200931c1.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/dd514e62cba829f1d05862a0434df8b85594f4a25fe9e1bcf9ba3811200931c1.jpg)

![fea74d6cac77244dd2e04bf72e9138185cfb4b0b1f1b3f294edde9eed972b363.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/images/fea74d6cac77244dd2e04bf72e9138185cfb4b0b1f1b3f294edde9eed972b363.jpg)

### Tables

![03c63122da00976e46cb06df7e866a1586ab7f905ed75c4c433ae927102ad53a.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/03c63122da00976e46cb06df7e866a1586ab7f905ed75c4c433ae927102ad53a.jpg)

![1c19bb4dbc58030bd74ae19bfa28f73272f8c352b7907c71a67d76eb64361013.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/1c19bb4dbc58030bd74ae19bfa28f73272f8c352b7907c71a67d76eb64361013.jpg)

![4cbaaeb9b6d01c29dae30f8bbf87cde9b8b9c27e181c61aaa7a4828c406b6f83.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/4cbaaeb9b6d01c29dae30f8bbf87cde9b8b9c27e181c61aaa7a4828c406b6f83.jpg)

![4f2583c4df99286ab08c5ef32e05c7d775d486634bac06532507566d22abb0b6.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/4f2583c4df99286ab08c5ef32e05c7d775d486634bac06532507566d22abb0b6.jpg)

![5dc4af04ac01158f560fbcff7375b6ad5f02d8fffd1794370ac65a5344453de4.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/5dc4af04ac01158f560fbcff7375b6ad5f02d8fffd1794370ac65a5344453de4.jpg)

![64d3e938114297fb7fd4d8e3e3d0e8d3ee75d56e422cfe4bbcad75ada9697ef0.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/64d3e938114297fb7fd4d8e3e3d0e8d3ee75d56e422cfe4bbcad75ada9697ef0.jpg)

![6b2cc6d090103ba6211f2c3f3af7fe62af23984081e9a546ad45db58d66a6a80.jpg](../icml_results/6234_How%20Language%20Models%20Process%20Negation/tables/6b2cc6d090103ba6211f2c3f3af7fe62af23984081e9a546ad45db58d66a6a80.jpg)

<a id="paper-6185"></a>

## Mitigating the Modality Gap in Vision–Language Models with Fractal Spectral Geometry


### Images

![1e840a15664b301d7b70cc3ed273eced707f4593f3c92a55a4fa17284e69dfeb.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/images/1e840a15664b301d7b70cc3ed273eced707f4593f3c92a55a4fa17284e69dfeb.jpg)

![6bce6ec220ed82202311cb2fd29a7d7c42d099681afccf08025789cc17806239.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/images/6bce6ec220ed82202311cb2fd29a7d7c42d099681afccf08025789cc17806239.jpg)

![cdd55975a9203e96381ee55d56828cf39f30875dd03d2bdcd7667d0e5da38ca6.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/images/cdd55975a9203e96381ee55d56828cf39f30875dd03d2bdcd7667d0e5da38ca6.jpg)

![e047f089a8e79c479f312f04d91fe654379f5a95e2b26cb1f2d8c89331aaa64a.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/images/e047f089a8e79c479f312f04d91fe654379f5a95e2b26cb1f2d8c89331aaa64a.jpg)

![ecd4ba864e574730cbcc0043eb0f5fde85721b40d80db9bee4c5a72dd44797c7.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/images/ecd4ba864e574730cbcc0043eb0f5fde85721b40d80db9bee4c5a72dd44797c7.jpg)

### Tables

![11e685d22f6a258df79b3bdb1ff4c91ccc9f9fcc0f3a8ec5e54618aa4139c072.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/11e685d22f6a258df79b3bdb1ff4c91ccc9f9fcc0f3a8ec5e54618aa4139c072.jpg)

![2c2234ae73b96b0b2c4feac737ef82925bac578569c7f6bfd6f1973b6bcb195f.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/2c2234ae73b96b0b2c4feac737ef82925bac578569c7f6bfd6f1973b6bcb195f.jpg)

![30b5650694717204fef3ab78f2dbe97017697c298161328dc455e934c12c5110.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/30b5650694717204fef3ab78f2dbe97017697c298161328dc455e934c12c5110.jpg)

![34297c78e7fec730f3b6079ba301bf5bad65e2d40920f8359c98a0ef967de963.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/34297c78e7fec730f3b6079ba301bf5bad65e2d40920f8359c98a0ef967de963.jpg)

![39d464809b21d552cfcc6e9305e4a41f7649e1f6ee3e9e711ee86c333c9d2e76.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/39d464809b21d552cfcc6e9305e4a41f7649e1f6ee3e9e711ee86c333c9d2e76.jpg)

![46c1c0abc929d2f969b9327fe84342a1d1a3a0cdc8d029a0b5f741426e621cb1.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/46c1c0abc929d2f969b9327fe84342a1d1a3a0cdc8d029a0b5f741426e621cb1.jpg)

![4f7ce32cf921fe2d37d1360385e51c496a2b3a840231039bd03639bc02344bd6.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/4f7ce32cf921fe2d37d1360385e51c496a2b3a840231039bd03639bc02344bd6.jpg)

![7152b9aa5940b3fa58d7a54b565bca38a6501688ad0f18972de1bf1d42f5e314.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/7152b9aa5940b3fa58d7a54b565bca38a6501688ad0f18972de1bf1d42f5e314.jpg)

![7799b79069b8dcbc8aa6d0b8682ab19ac6ea824c60f26fb1805281c14d3441f3.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/7799b79069b8dcbc8aa6d0b8682ab19ac6ea824c60f26fb1805281c14d3441f3.jpg)

![7c99e07e7090ac2895ecfd3eea4ffbdaf4ba0267375425fff76a0e57b7a43183.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/7c99e07e7090ac2895ecfd3eea4ffbdaf4ba0267375425fff76a0e57b7a43183.jpg)

![80294beb64377d2da9d6a9b2aecbffa37702204a6eb597e2d86fc035eeeee737.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/80294beb64377d2da9d6a9b2aecbffa37702204a6eb597e2d86fc035eeeee737.jpg)

![81f59b89c6b8068237429126b4cc73366717a235d64fc1d34351b12ed2384c1b.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/81f59b89c6b8068237429126b4cc73366717a235d64fc1d34351b12ed2384c1b.jpg)

![92739dfcf0efa11272a3984f60d9c379ba60e75aecd75ef6fda854cc83592b5f.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/92739dfcf0efa11272a3984f60d9c379ba60e75aecd75ef6fda854cc83592b5f.jpg)

![97a071e422ea7f0939211b1f810c40d06f863d20fce769d45da181fa75cfa3ed.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/97a071e422ea7f0939211b1f810c40d06f863d20fce769d45da181fa75cfa3ed.jpg)

![9ff81e31464cb33f9f0382b35d91890562940a3c52d232d33966fc17b310c70a.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/9ff81e31464cb33f9f0382b35d91890562940a3c52d232d33966fc17b310c70a.jpg)

![a1ed581b8f67f19227598428a929aa4f75fa1e3d5b180ff590c35952e7979283.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/a1ed581b8f67f19227598428a929aa4f75fa1e3d5b180ff590c35952e7979283.jpg)

![a9bdf9484b48d231681fadc416dca2ce0d906fd61fc929f6ae555d5639c6f1e9.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/a9bdf9484b48d231681fadc416dca2ce0d906fd61fc929f6ae555d5639c6f1e9.jpg)

![ae794b2ed18872588aed0e6f36ceb1af0b9b7c122f1a29bf071ac2d2480ee06b.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/ae794b2ed18872588aed0e6f36ceb1af0b9b7c122f1a29bf071ac2d2480ee06b.jpg)

![bd6c004a452a11538ca4e47b67a8dfcbf2c3899423a903760e689ea1d81790e0.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/bd6c004a452a11538ca4e47b67a8dfcbf2c3899423a903760e689ea1d81790e0.jpg)

![bd82ab7b220dfbed0747aa2cfba505002753e66e0f30613f6aac78f9449c5866.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/bd82ab7b220dfbed0747aa2cfba505002753e66e0f30613f6aac78f9449c5866.jpg)

![c29a649d38c45255ef4006eddacab2e6dec868cc7f352c6ab2a3a168f19d5f39.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/c29a649d38c45255ef4006eddacab2e6dec868cc7f352c6ab2a3a168f19d5f39.jpg)

![e40707a5ba72d2904d5a91d602379a3f58fb878cecd4c4f04f4a2cd5e515ee7c.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/e40707a5ba72d2904d5a91d602379a3f58fb878cecd4c4f04f4a2cd5e515ee7c.jpg)

![f4b2416de8e0011f721ec98f778019d8435cb91f39d3a6943b539b602ba827d1.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/f4b2416de8e0011f721ec98f778019d8435cb91f39d3a6943b539b602ba827d1.jpg)

![f6ae29bd1f1511b323e47e6152db7b5ca7844723aba7240048c0d4e669ee2994.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/f6ae29bd1f1511b323e47e6152db7b5ca7844723aba7240048c0d4e669ee2994.jpg)

![faeabfe77a9b1b5ddefbfdd1ffa5c35b16d0f29db1d8ccd1c0fc4d2c25ecbdbf.jpg](../icml_results/6235_Mitigating%20the%20Modality%20Gap%20in%20Vision%E2%80%93Language%20Models%20with%20Fractal%20Spectral%20Geometry/tables/faeabfe77a9b1b5ddefbfdd1ffa5c35b16d0f29db1d8ccd1c0fc4d2c25ecbdbf.jpg)

<a id="paper-6186"></a>

## Tailoring the Training: Difficulty-Aware Learning Strategy Allocation for Large Language Models


### Images

![0dc200ab75fed2e05421c7ba8594883c7330135c7991ef5af809bdf7ee8927b9.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/0dc200ab75fed2e05421c7ba8594883c7330135c7991ef5af809bdf7ee8927b9.jpg)

![37a857d4c165cec7976941f206979082a81ad97849baa9325417b43d71a1e41e.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/37a857d4c165cec7976941f206979082a81ad97849baa9325417b43d71a1e41e.jpg)

![3fecbfa0db5c4d447a30ad8956c665300a9175ef9df9c7e2067e45e38e4bc01e.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/3fecbfa0db5c4d447a30ad8956c665300a9175ef9df9c7e2067e45e38e4bc01e.jpg)

![50e0c4c3225a13907528c0761e3605bb14045e68a6a70f54754475c61e828812.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/50e0c4c3225a13907528c0761e3605bb14045e68a6a70f54754475c61e828812.jpg)

![6df638e20b360fb94f298588e1795d35403fe11fc87eebdd1f68aa2390a7333a.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/6df638e20b360fb94f298588e1795d35403fe11fc87eebdd1f68aa2390a7333a.jpg)

![e6fe8db8a61f3caac04f2bcf080be30434e1bd13422d95ae2f6d3756ba08884b.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/images/e6fe8db8a61f3caac04f2bcf080be30434e1bd13422d95ae2f6d3756ba08884b.jpg)

### Tables

![2c68aa5a9fa2aa55056bccfc6b21327d75434b3b768afe05b1aec23b6a75e459.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/2c68aa5a9fa2aa55056bccfc6b21327d75434b3b768afe05b1aec23b6a75e459.jpg)

![4831e69c3eefe3cdbc8660b5142fc4ca2daeb0236d92da5f1580175db7306a35.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/4831e69c3eefe3cdbc8660b5142fc4ca2daeb0236d92da5f1580175db7306a35.jpg)

![60422797a2c77805d810bbded160d356584c2d816ded87b9d64291306cd38275.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/60422797a2c77805d810bbded160d356584c2d816ded87b9d64291306cd38275.jpg)

![c9fb25a2bbd7a3865ff31180e1f54eea0b57ee3d4ede9f5f952bd033ece506ad.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/c9fb25a2bbd7a3865ff31180e1f54eea0b57ee3d4ede9f5f952bd033ece506ad.jpg)

![d5341cd5c4a3121e42eacf1db9355d66f4757087ce2f6438c0f063a42e780e37.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/d5341cd5c4a3121e42eacf1db9355d66f4757087ce2f6438c0f063a42e780e37.jpg)

![db8d91441e3b0a5ac938a79eb8e8529f528ed7aa0edc04120a9e2539a9275b74.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/db8d91441e3b0a5ac938a79eb8e8529f528ed7aa0edc04120a9e2539a9275b74.jpg)

![e789aac2f19637564e5b995077f7b0fe197740e868c3e3437f7555e170041bbe.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/e789aac2f19637564e5b995077f7b0fe197740e868c3e3437f7555e170041bbe.jpg)

![ee30ee7f3890994dcea440657a57cb65af1ceffb2b4e8dc9aadfc09552a541b9.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/ee30ee7f3890994dcea440657a57cb65af1ceffb2b4e8dc9aadfc09552a541b9.jpg)

![f3a24ae9952e5072c54f8420021161a688341c9d16034927dda27047ecac0671.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/f3a24ae9952e5072c54f8420021161a688341c9d16034927dda27047ecac0671.jpg)

![f66ccf5f6ec8b0537810c26e33fba4251dbdef2749c6754307737b97dfc5af75.jpg](../icml_results/6236_Tailoring%20the%20Training_%20Difficulty-Aware%20Learning%20Strategy%20Allocation%20for%20Large%20Language%20Models/tables/f66ccf5f6ec8b0537810c26e33fba4251dbdef2749c6754307737b97dfc5af75.jpg)

<a id="paper-6187"></a>

## ForensicConcept: Transferable Forensic Concepts for AIGI Detection


### Images

![08d7d297213d5b9de75ad3e2ec99d35bee3653c917d517a6ebd1fddfab3afc15.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/08d7d297213d5b9de75ad3e2ec99d35bee3653c917d517a6ebd1fddfab3afc15.jpg)

![0e86bab898dceee4de6266a2dad37c41e43625353944acc696fdf870cb6f0e89.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/0e86bab898dceee4de6266a2dad37c41e43625353944acc696fdf870cb6f0e89.jpg)

![11319092cc69bf4cb1dfc7652f8469cf515e105b3acb89b45274e373c362f666.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/11319092cc69bf4cb1dfc7652f8469cf515e105b3acb89b45274e373c362f666.jpg)

![123f4ab9100a6dec5d78767a9e2d089dec371511db5aa6fb9466173d27ccba83.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/123f4ab9100a6dec5d78767a9e2d089dec371511db5aa6fb9466173d27ccba83.jpg)

![162918cfd9e4f83dbfcab734e42ef2e0c2027c9bf774ea26b9fbfef4920b9db3.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/162918cfd9e4f83dbfcab734e42ef2e0c2027c9bf774ea26b9fbfef4920b9db3.jpg)

![17730a6445e867d629847e83b13d5c0ef989c0302cf40e4f80847c52cb9bdbe7.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/17730a6445e867d629847e83b13d5c0ef989c0302cf40e4f80847c52cb9bdbe7.jpg)

![237b81f946dc1f5ee15e0d7470c8482a434988d1890d18fcfa475cb0871e6b2b.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/237b81f946dc1f5ee15e0d7470c8482a434988d1890d18fcfa475cb0871e6b2b.jpg)

![2d298098b31b2647bba49a77891a2f535f270ea534269ccbbf651c9b0894e32e.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/2d298098b31b2647bba49a77891a2f535f270ea534269ccbbf651c9b0894e32e.jpg)

![39d20dfd07bca2c55c0753d9353f033cfe2cb52f68e06f51271fc0283c4dfab4.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/39d20dfd07bca2c55c0753d9353f033cfe2cb52f68e06f51271fc0283c4dfab4.jpg)

![3afb219199cfb72a2e7a52bdf0de6a51c70753efd8b05a9f407a71e4a9cbe2e7.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3afb219199cfb72a2e7a52bdf0de6a51c70753efd8b05a9f407a71e4a9cbe2e7.jpg)

![3ddb56eecd02dea049f2da84ff794510da7d7a0c9c58c48c6dc4fba9b57b7604.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3ddb56eecd02dea049f2da84ff794510da7d7a0c9c58c48c6dc4fba9b57b7604.jpg)

![3e22142872a1c0e8179fb072c0fa8180c0e24646ee68946fae2dd908d12ade3d.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3e22142872a1c0e8179fb072c0fa8180c0e24646ee68946fae2dd908d12ade3d.jpg)

![3ebc35a69029bf5dbeaa129e99e127f520d7cf3c33dfdd26997e7215acd35a03.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3ebc35a69029bf5dbeaa129e99e127f520d7cf3c33dfdd26997e7215acd35a03.jpg)

![3ed42dc4005721421ba1496696ba5c2cf60df831fd2982da212b8580e50ac97e.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3ed42dc4005721421ba1496696ba5c2cf60df831fd2982da212b8580e50ac97e.jpg)

![3fb37f6944271414cfcdc65f8c2cc28eeb0d6b5742b05f35b7f8726e6d54bfdb.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/3fb37f6944271414cfcdc65f8c2cc28eeb0d6b5742b05f35b7f8726e6d54bfdb.jpg)

![45e6478d8722a9bfd90e629077be78f0f78b309746fb7828391a42b08e020f2a.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/45e6478d8722a9bfd90e629077be78f0f78b309746fb7828391a42b08e020f2a.jpg)

![4b77ae4bfff8b203b73229776a4be1bd1a25af673b2465fe51abbafb4c59a278.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/4b77ae4bfff8b203b73229776a4be1bd1a25af673b2465fe51abbafb4c59a278.jpg)

![4bb363cbc4e9b810c0b101790ee4b07f9ed212a72860c18c20fc43f38b0ce826.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/4bb363cbc4e9b810c0b101790ee4b07f9ed212a72860c18c20fc43f38b0ce826.jpg)

![51edf4a7c3138899f2aa72f7f17037d42948cbe30933e04e3a58e4fcec469263.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/51edf4a7c3138899f2aa72f7f17037d42948cbe30933e04e3a58e4fcec469263.jpg)

![52b81f0dc46f671bedfd7f17f4e6799b68594b881d6e0e986c5dc9310d7e56ea.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/52b81f0dc46f671bedfd7f17f4e6799b68594b881d6e0e986c5dc9310d7e56ea.jpg)

![5e1f46d106ea9f175afd62197999332f55688e4e5b69dadc66be1a67e9d07f03.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/5e1f46d106ea9f175afd62197999332f55688e4e5b69dadc66be1a67e9d07f03.jpg)

![5e45b8ddf91d9a5c3c80cc909f26f551af5d5f087ff7d5084c3d25a1d954ffda.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/5e45b8ddf91d9a5c3c80cc909f26f551af5d5f087ff7d5084c3d25a1d954ffda.jpg)

![6302ea7d1605214ceb9f552bb40f0c0b53479fcbd82299f53e1b7956d3cfff32.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/6302ea7d1605214ceb9f552bb40f0c0b53479fcbd82299f53e1b7956d3cfff32.jpg)

![7496d7f9057c02981bc7c4dcd521e2edca0cf155b44a30740f89125bafb63831.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/7496d7f9057c02981bc7c4dcd521e2edca0cf155b44a30740f89125bafb63831.jpg)

![8389b682ac46447a4897f080df3af4be4290b2c16deb1260926d7f06bdc65519.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/8389b682ac46447a4897f080df3af4be4290b2c16deb1260926d7f06bdc65519.jpg)

![8472165bb2a6b12411bee06e8d5c92aec4a0dc85e8e1e76ba3c2cb44c37c9018.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/8472165bb2a6b12411bee06e8d5c92aec4a0dc85e8e1e76ba3c2cb44c37c9018.jpg)

![8a36f23fc2686319282fe66f5804c501383af6f05dc5853f83def6758cb3319c.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/8a36f23fc2686319282fe66f5804c501383af6f05dc5853f83def6758cb3319c.jpg)

![8d94a8f9a2941f63a2609b7069b761684f887a55cd9e7c6e2bcbe6e0777b5ca4.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/8d94a8f9a2941f63a2609b7069b761684f887a55cd9e7c6e2bcbe6e0777b5ca4.jpg)

![91176b197ea5afccd5c0ec0d57ecfc0d8bdca1dd8b09b6b55d373b295b21fdd1.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/91176b197ea5afccd5c0ec0d57ecfc0d8bdca1dd8b09b6b55d373b295b21fdd1.jpg)

![9f514d43a5f66ac97651a354be9edd1e2f3446ab8fc8e2b861f189ebe1fcad27.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/9f514d43a5f66ac97651a354be9edd1e2f3446ab8fc8e2b861f189ebe1fcad27.jpg)

![a04dee8534e14d33dec6a85bfd0b0a98cee8e09c5dd31890c63ff0d9f5b22ff7.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/a04dee8534e14d33dec6a85bfd0b0a98cee8e09c5dd31890c63ff0d9f5b22ff7.jpg)

![a25359443584f2a40a120058c804507d702ada46a41db607418baa538498426b.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/a25359443584f2a40a120058c804507d702ada46a41db607418baa538498426b.jpg)

![a278146085bea1e45e9aa4aaf5d918d04b13735aaa1200eeee7980f44a56f64a.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/a278146085bea1e45e9aa4aaf5d918d04b13735aaa1200eeee7980f44a56f64a.jpg)

![a54e54970854c841e5036a03c93cca589923c9226678087766748f1d02550888.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/a54e54970854c841e5036a03c93cca589923c9226678087766748f1d02550888.jpg)

![a83e718b97a3bf4a9515c3d624bd9bccd27c24d9cc4b89840b7b90f258c71af4.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/a83e718b97a3bf4a9515c3d624bd9bccd27c24d9cc4b89840b7b90f258c71af4.jpg)

![aa464ed22d6a2d2333a18dc96082fd83b39f7d9f2b0d6e90ac66b0165e6a163e.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/aa464ed22d6a2d2333a18dc96082fd83b39f7d9f2b0d6e90ac66b0165e6a163e.jpg)

![ab27a5e69fb3de7c9eedd172c69c8f955fe7c34b4e8c11f386c52fd104a9b45b.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/ab27a5e69fb3de7c9eedd172c69c8f955fe7c34b4e8c11f386c52fd104a9b45b.jpg)

![b2d1460ef1d56f7a656ad6905f6021d4432279a5c214e90b0fd465f70d5f922c.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/b2d1460ef1d56f7a656ad6905f6021d4432279a5c214e90b0fd465f70d5f922c.jpg)

![b5c1023a9de961c2ad17b377c3177d538e62a02c83ae2a0f9513079845de6b19.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/b5c1023a9de961c2ad17b377c3177d538e62a02c83ae2a0f9513079845de6b19.jpg)

![b5fcfd5da5b2f4fbf8033380ce65a5ff5492dd5b6822c68b7284c2a6047c6550.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/b5fcfd5da5b2f4fbf8033380ce65a5ff5492dd5b6822c68b7284c2a6047c6550.jpg)

![b637bcd073e20b788982998939384e7e49a62853916476820f612cf72adc3516.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/b637bcd073e20b788982998939384e7e49a62853916476820f612cf72adc3516.jpg)

![b68b8155bdea21a8f0bca289f7cd75c5f819df42e054097cf33825c591527850.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/b68b8155bdea21a8f0bca289f7cd75c5f819df42e054097cf33825c591527850.jpg)

![bede83a930e7639c82c4c77167b604d741526b60392b8cc70544e065940f3c0c.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/bede83a930e7639c82c4c77167b604d741526b60392b8cc70544e065940f3c0c.jpg)

![c5d1cee7477119482594cc973fedfbfe218d5d34ecd598fd539ba1b5971ac403.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/c5d1cee7477119482594cc973fedfbfe218d5d34ecd598fd539ba1b5971ac403.jpg)

![c7ca4b45d58d096db8ce859c881c73245b2c3a36be37a8f91d3f4bd97b11edcf.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/c7ca4b45d58d096db8ce859c881c73245b2c3a36be37a8f91d3f4bd97b11edcf.jpg)

![cd95fc7e455a8f750fd089594cdb6d214a28bcfbef5aaed8c6a7b91826f344e1.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/cd95fc7e455a8f750fd089594cdb6d214a28bcfbef5aaed8c6a7b91826f344e1.jpg)

![cdb09fab6b6948775df6b83b24579b9fd3f537d2d342bac3884c72daf945ec29.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/cdb09fab6b6948775df6b83b24579b9fd3f537d2d342bac3884c72daf945ec29.jpg)

![d13523cfe8b2d13b881ec28805ae546ff35efd70200b99edb6aafdbe393e3e8e.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/d13523cfe8b2d13b881ec28805ae546ff35efd70200b99edb6aafdbe393e3e8e.jpg)

![d14433daba0e050c6e91e211b140a4ad123cbf739d83a6d234515dd66e5baa69.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/d14433daba0e050c6e91e211b140a4ad123cbf739d83a6d234515dd66e5baa69.jpg)

![df8e476870f9091fd43ae655cdb5909d05053ecba261631f8ee7588c92ffbb45.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/df8e476870f9091fd43ae655cdb5909d05053ecba261631f8ee7588c92ffbb45.jpg)

![e2073fd1c983585bdb74e0285a0868cbb1a78aa2d5453a145644b7c7a450db00.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/e2073fd1c983585bdb74e0285a0868cbb1a78aa2d5453a145644b7c7a450db00.jpg)

![e47d2684f2deb7f55820071b852e3ebe91835f94860aa496720c0aee24c6285c.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/e47d2684f2deb7f55820071b852e3ebe91835f94860aa496720c0aee24c6285c.jpg)

![eaf7e7d9bde0ded75af8c04b6a4710fac4038929d372542aa48e675e0c57c051.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/eaf7e7d9bde0ded75af8c04b6a4710fac4038929d372542aa48e675e0c57c051.jpg)

![ec7a93ab36edbecd33ddb5be5e5d6dadd7367840761c41c7aede2c03a0a5ec3e.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/ec7a93ab36edbecd33ddb5be5e5d6dadd7367840761c41c7aede2c03a0a5ec3e.jpg)

![f4e3b181e228b7888ddf364e6ddb71b58ddea4c64011a7ff6fbfc53268ba7690.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/f4e3b181e228b7888ddf364e6ddb71b58ddea4c64011a7ff6fbfc53268ba7690.jpg)

![fa3f97db193d1b1b270e27f73d69782988e238cb23268a60ae36fe8911d13273.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/fa3f97db193d1b1b270e27f73d69782988e238cb23268a60ae36fe8911d13273.jpg)

![fc76299c7ac8388780c23fa88384ec58abc36f49291bb1b864a83e9c09130971.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/fc76299c7ac8388780c23fa88384ec58abc36f49291bb1b864a83e9c09130971.jpg)

![fcdbaaa5db592bebe3ba99e9fd0629b0c535a1bd1eabafd565ea04cb8753196b.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/images/fcdbaaa5db592bebe3ba99e9fd0629b0c535a1bd1eabafd565ea04cb8753196b.jpg)

### Tables

![0168837f1fabf687fbc1e83d0931660eff7c2bd3d0c1862e58bf9ff167282cc8.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/0168837f1fabf687fbc1e83d0931660eff7c2bd3d0c1862e58bf9ff167282cc8.jpg)

![20ed2aea80c1617cbc9d9339803a15c5543679c944e255c9bfa47fe2d050d029.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/20ed2aea80c1617cbc9d9339803a15c5543679c944e255c9bfa47fe2d050d029.jpg)

![30ed5b0dbd7c72dfc32ee1d6b83ba60b9b8f4d403609df901e678127282509e8.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/30ed5b0dbd7c72dfc32ee1d6b83ba60b9b8f4d403609df901e678127282509e8.jpg)

![34670664324a57e9dcedeeadf74f2cdf32293dfbf626e01df0f0781375131c56.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/34670664324a57e9dcedeeadf74f2cdf32293dfbf626e01df0f0781375131c56.jpg)

![391dd53a86147793c4bcb21e4e51b73d6760ad2a31daa8f516462921b01e9e08.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/391dd53a86147793c4bcb21e4e51b73d6760ad2a31daa8f516462921b01e9e08.jpg)

![5c0242a7907fbadf5538c4376cbeea04f7b0e00f57043264d71f03bbccd1a1cc.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/5c0242a7907fbadf5538c4376cbeea04f7b0e00f57043264d71f03bbccd1a1cc.jpg)

![7d20712d75603b035a7567c3e0bdae610f950e9b6527d6874dbd84d594f23d58.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/7d20712d75603b035a7567c3e0bdae610f950e9b6527d6874dbd84d594f23d58.jpg)

![a2f1fdd183aa84c6a9613a9d573166492d9c8466bd8ea661076c61b077b920e3.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/a2f1fdd183aa84c6a9613a9d573166492d9c8466bd8ea661076c61b077b920e3.jpg)

![b1b5c42a038872ce7ff44cd9033006149a3dde053136c32b8df9dd1346c66fb7.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/b1b5c42a038872ce7ff44cd9033006149a3dde053136c32b8df9dd1346c66fb7.jpg)

![d639d2f6c820ca42a14d455cd4308bda3ef44efa82f7950f816f578733abe696.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/d639d2f6c820ca42a14d455cd4308bda3ef44efa82f7950f816f578733abe696.jpg)

![e7fece9ae7493fb8e19bf2bf37e37efd86055f05328278085ff89d1d82cc03f5.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/e7fece9ae7493fb8e19bf2bf37e37efd86055f05328278085ff89d1d82cc03f5.jpg)

![eb94edce87e72d4d4d074a54da4e0abdcfa1c1b94391297b07c5546e10a529bb.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/eb94edce87e72d4d4d074a54da4e0abdcfa1c1b94391297b07c5546e10a529bb.jpg)

![fa0dd4a763dd2acaa762bc640fa601f8837e84fd4e065de29f6b6e98c726f42a.jpg](../icml_results/6237_ForensicConcept_%20Transferable%20Forensic%20Concepts%20for%20AIGI%20Detection/tables/fa0dd4a763dd2acaa762bc640fa601f8837e84fd4e065de29f6b6e98c726f42a.jpg)

<a id="paper-6188"></a>

## Structured Multi-step Jailbreaking under a Hamiltonian Generative Formulation


### Images

![0b39134f08dc1006effc0dd4c5c32cbddd81a8a604225bbf497cd20928b26070.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/images/0b39134f08dc1006effc0dd4c5c32cbddd81a8a604225bbf497cd20928b26070.jpg)

![17e9603aedc5c0fe8859e89c12fd823cbfa3784633c4da6bbb5494e1ab2e3195.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/images/17e9603aedc5c0fe8859e89c12fd823cbfa3784633c4da6bbb5494e1ab2e3195.jpg)

![6184977d89483ac436fbed040fca3bec9a6bc88c245e24360b525ca756831041.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/images/6184977d89483ac436fbed040fca3bec9a6bc88c245e24360b525ca756831041.jpg)

![7142b99505fd7f09e3f275df533054be1fcb3cd9c36d44075782861fa6f3676f.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/images/7142b99505fd7f09e3f275df533054be1fcb3cd9c36d44075782861fa6f3676f.jpg)

![f258e25664c6463665fa735e25ed22344e345fd48a71706aad895ab163740c8b.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/images/f258e25664c6463665fa735e25ed22344e345fd48a71706aad895ab163740c8b.jpg)

### Tables

![0bc28d8d94a6c2081b9c33fc3b1407a5e9e7bca77cb966550d76d77e7ca9503e.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/0bc28d8d94a6c2081b9c33fc3b1407a5e9e7bca77cb966550d76d77e7ca9503e.jpg)

![0d9357e305aae74096d9707b5c004518cf5f67aa4d82f98539a76a06f25309a4.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/0d9357e305aae74096d9707b5c004518cf5f67aa4d82f98539a76a06f25309a4.jpg)

![11dc7197994c8d35acc2aeb13553e334e7f84794a9254e7baa640ddb2f425d42.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/11dc7197994c8d35acc2aeb13553e334e7f84794a9254e7baa640ddb2f425d42.jpg)

![202485dc4a63124dba0b39f6ca39e6922f809eb2e9101bda8af7ad8272060069.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/202485dc4a63124dba0b39f6ca39e6922f809eb2e9101bda8af7ad8272060069.jpg)

![2b0e34e84883c7c055ed41581a3cd1bae8bf2e7bdc894d235bc8920dd276fecc.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/2b0e34e84883c7c055ed41581a3cd1bae8bf2e7bdc894d235bc8920dd276fecc.jpg)

![2ce82fdce457df2f595172328e85dd61b3643f13e00dbb676a11bbfd36edaac6.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/2ce82fdce457df2f595172328e85dd61b3643f13e00dbb676a11bbfd36edaac6.jpg)

![34446c6edf0e23e96a097601b59f3b67ef53bd11f2648129607fab6e9c966092.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/34446c6edf0e23e96a097601b59f3b67ef53bd11f2648129607fab6e9c966092.jpg)

![35b2427e92d4a34f2044780c4212061568e6d37189a39accdbf9035d41f1a9e8.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/35b2427e92d4a34f2044780c4212061568e6d37189a39accdbf9035d41f1a9e8.jpg)

![390e2886420b394f8b5aba09cc5e550d3abac8894640af400d12991655353ddc.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/390e2886420b394f8b5aba09cc5e550d3abac8894640af400d12991655353ddc.jpg)

![3a6f96916a62d89b9854dd7cb4ab5566e5f6629d46270380d8189dc1fb568e59.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/3a6f96916a62d89b9854dd7cb4ab5566e5f6629d46270380d8189dc1fb568e59.jpg)

![4d2e1aa807f215d3e2437ff5036aaf574afd29f75bffbe92be24ca5048e3ca09.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/4d2e1aa807f215d3e2437ff5036aaf574afd29f75bffbe92be24ca5048e3ca09.jpg)

![4d4658ae67ec7b5d0b5b7c93c239663c40204e040ca21166e8d5ffc9ec33e44d.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/4d4658ae67ec7b5d0b5b7c93c239663c40204e040ca21166e8d5ffc9ec33e44d.jpg)

![5567421b64e097d6bb08323dd745c4a0da7b51ee5503b6be7a68d872598a4fde.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/5567421b64e097d6bb08323dd745c4a0da7b51ee5503b6be7a68d872598a4fde.jpg)

![59eb3e248ccd2660ec3ff93314ab9264096cd27ebf1b4db10d1777dd129a2c95.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/59eb3e248ccd2660ec3ff93314ab9264096cd27ebf1b4db10d1777dd129a2c95.jpg)

![6dbf702e1b8fb248ac8d50f885ecfaedd57576c65640067d4173f21e5ddaadaf.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/6dbf702e1b8fb248ac8d50f885ecfaedd57576c65640067d4173f21e5ddaadaf.jpg)

![6fa7c8f4940109f53c8238d562f1c7fa71f259f59b67cabb21f022a5ddb3d427.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/6fa7c8f4940109f53c8238d562f1c7fa71f259f59b67cabb21f022a5ddb3d427.jpg)

![733efe7436a0a083209c25c62f90c79a2a585f9793e0a36896d59a896e1659b2.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/733efe7436a0a083209c25c62f90c79a2a585f9793e0a36896d59a896e1659b2.jpg)

![83a80bffcee8c74d395fff2291c2533cf460ed4fc0cf260590396326844b4a74.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/83a80bffcee8c74d395fff2291c2533cf460ed4fc0cf260590396326844b4a74.jpg)

![85179f5e76dd902a73ce428798642cc55343dcac086fa5b8832ed0570a6c5fb8.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/85179f5e76dd902a73ce428798642cc55343dcac086fa5b8832ed0570a6c5fb8.jpg)

![9483c406fcbefb6e3379e3e80b9b4c6ff73afd560d240eee59781b1eaaf31ddc.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/9483c406fcbefb6e3379e3e80b9b4c6ff73afd560d240eee59781b1eaaf31ddc.jpg)

![9a1e6b01b6c0195fdf40d926b3688377baa3c869bd7d89f6cc707a7268791659.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/9a1e6b01b6c0195fdf40d926b3688377baa3c869bd7d89f6cc707a7268791659.jpg)

![9acd94853f823acf9eb03c1d49e5232204d300e2aac75f7d7d9fcfc105fb2d0f.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/9acd94853f823acf9eb03c1d49e5232204d300e2aac75f7d7d9fcfc105fb2d0f.jpg)

![9d99d5b3127207de676f9199284daa917448de20882a430dc81e5cc848bf58d7.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/9d99d5b3127207de676f9199284daa917448de20882a430dc81e5cc848bf58d7.jpg)

![a3cfc8878c42b696f26fb51caf08d3b183fd38c7fe84c712678c088ffc9a0363.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/a3cfc8878c42b696f26fb51caf08d3b183fd38c7fe84c712678c088ffc9a0363.jpg)

![a4489e221d771e7317f8a610834d344d4e627769d03d7163134e9e7b9cd833d5.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/a4489e221d771e7317f8a610834d344d4e627769d03d7163134e9e7b9cd833d5.jpg)

![ab7fbd6030c2e909e2a58a312d0b892d31a6d9f1c9514917f01dd97abd5f1c1e.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/ab7fbd6030c2e909e2a58a312d0b892d31a6d9f1c9514917f01dd97abd5f1c1e.jpg)

![b07fb3dcb72877972ab0b1c3385f94a51bb5ac0628354cf9f0e819ff913a423b.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/b07fb3dcb72877972ab0b1c3385f94a51bb5ac0628354cf9f0e819ff913a423b.jpg)

![bd926942362601eaf3afb8fc2a2b651965d12eb8526c06302e5782aca231b0d9.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/bd926942362601eaf3afb8fc2a2b651965d12eb8526c06302e5782aca231b0d9.jpg)

![bf5ebb5405c95bda1e7a2261d96128c6be5c3eadfb6e784eae38ee0cd481d55c.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/bf5ebb5405c95bda1e7a2261d96128c6be5c3eadfb6e784eae38ee0cd481d55c.jpg)

![c770b42b16ec7db634801c33b789c446b01a91ac90b3b4e77eae1718fb8cf39a.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/c770b42b16ec7db634801c33b789c446b01a91ac90b3b4e77eae1718fb8cf39a.jpg)

![cbb333816cc9c4b22d5558be34c60be397df75449042f3bbc520c780b289efcd.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/cbb333816cc9c4b22d5558be34c60be397df75449042f3bbc520c780b289efcd.jpg)

![dcde11b1d663d1249121ddc9f4813af202a841d02d2101ece4832a94cbb23f82.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/dcde11b1d663d1249121ddc9f4813af202a841d02d2101ece4832a94cbb23f82.jpg)

![f4455a42ae08e40655484518c986492e539fcca307b42442abeffe724f6fbedb.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/f4455a42ae08e40655484518c986492e539fcca307b42442abeffe724f6fbedb.jpg)

![f44f99508ee81fd92bd39072a621c3194394c970e3be2d1dafd201c7ef49aa73.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/f44f99508ee81fd92bd39072a621c3194394c970e3be2d1dafd201c7ef49aa73.jpg)

![f71bd0198b8713e19147beef711bc306475a90802cb07a9fb9dd226f0d904787.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/f71bd0198b8713e19147beef711bc306475a90802cb07a9fb9dd226f0d904787.jpg)

![fad5f7b44b0d9b3a1bc5201184f6f736354c764b97f12cb3dedf25934c2e8ae5.jpg](../icml_results/6238_Structured%20Multi-step%20Jailbreaking%20under%20a%20Hamiltonian%20Generative%20Formulation/tables/fad5f7b44b0d9b3a1bc5201184f6f736354c764b97f12cb3dedf25934c2e8ae5.jpg)

<a id="paper-6189"></a>

## Calibrating Decision Robustness via Inverse Conformal Risk Control


### Images

![06821cfd40751665819799b85d44cef963df65fae56de784965dc7b18528c7a0.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/06821cfd40751665819799b85d44cef963df65fae56de784965dc7b18528c7a0.jpg)

![07e24c2ceba3cefe6ffd74bcbcca5597cabecc8f9f06b4a0ff7ada6a8d96cdbe.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/07e24c2ceba3cefe6ffd74bcbcca5597cabecc8f9f06b4a0ff7ada6a8d96cdbe.jpg)

![14d0784f0680b7eb60024060c6317d0922fcbcd5aa06c51fc06f3669987d572a.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/14d0784f0680b7eb60024060c6317d0922fcbcd5aa06c51fc06f3669987d572a.jpg)

![194d28920bbbe4456e09415bde9c9d4adb9f4da76796880d7d816c952126944c.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/194d28920bbbe4456e09415bde9c9d4adb9f4da76796880d7d816c952126944c.jpg)

![7a87f283ba7be0bf4c17feb260a8c1732ed25637500ce51f32b7469bec9e2e58.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/7a87f283ba7be0bf4c17feb260a8c1732ed25637500ce51f32b7469bec9e2e58.jpg)

![b059f590ea1a2698465fc9defa86f699066747fb27dad71b871b3182e9182c14.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/b059f590ea1a2698465fc9defa86f699066747fb27dad71b871b3182e9182c14.jpg)

![bdd6af28cef791e9ecb1ec72ac0188c7bf6929cbff9e4704cabb73c0c8d6f419.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/bdd6af28cef791e9ecb1ec72ac0188c7bf6929cbff9e4704cabb73c0c8d6f419.jpg)

![c2219de68cf4ad457a89f92d69e7dd744b79a904e18ebb7539a2d75c79043c10.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/c2219de68cf4ad457a89f92d69e7dd744b79a904e18ebb7539a2d75c79043c10.jpg)

![d39c432549cb79969f5beb6798686451d8b2df0ff19781392bad3e699843f701.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/d39c432549cb79969f5beb6798686451d8b2df0ff19781392bad3e699843f701.jpg)

![fad846827f29275c8632b6771adf17ab725b2c8b4a6a68545dc7f430323ae517.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/images/fad846827f29275c8632b6771adf17ab725b2c8b4a6a68545dc7f430323ae517.jpg)

### Tables

![92f3ebdf4d2bf3fbd396ae28379b73a28e578a0b33e1fe47682b453ab8b1f7a8.jpg](../icml_results/6239_Calibrating%20Decision%20Robustness%20via%20Inverse%20Conformal%20Risk%20Control/tables/92f3ebdf4d2bf3fbd396ae28379b73a28e578a0b33e1fe47682b453ab8b1f7a8.jpg)

<a id="paper-6190"></a>

## Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing


### Images

![14aa88bc22bbd439f2566cac8411619fd7014b49828c8aca1f446278c1de1b38.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/14aa88bc22bbd439f2566cac8411619fd7014b49828c8aca1f446278c1de1b38.jpg)

![15025a98f16c0c29c7fa5848cb04caa40326206f843faf45ca0ddc069a2a992d.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/15025a98f16c0c29c7fa5848cb04caa40326206f843faf45ca0ddc069a2a992d.jpg)

![5b564414f160f96298bb58ffa656aa807f641aea0ed29a273c4dd4fa200bff88.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/5b564414f160f96298bb58ffa656aa807f641aea0ed29a273c4dd4fa200bff88.jpg)

![7b0476f304f47d4ba12814187ee74e8d47257a1fbea7225e5a60b86048a08604.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/7b0476f304f47d4ba12814187ee74e8d47257a1fbea7225e5a60b86048a08604.jpg)

![833ccba25576ac626502d781daec640e0583a4bec004941d57ea331a7f80de9a.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/833ccba25576ac626502d781daec640e0583a4bec004941d57ea331a7f80de9a.jpg)

![b4632937c0cd777eaab95ecd1f9affbb0377c7d84c616f8cec8c9233818a93b9.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/b4632937c0cd777eaab95ecd1f9affbb0377c7d84c616f8cec8c9233818a93b9.jpg)

![c4af0f483aa9e747443d3bb0445716f771c8c667fb3ef0ad4626f131361dafd5.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/c4af0f483aa9e747443d3bb0445716f771c8c667fb3ef0ad4626f131361dafd5.jpg)

![dcf1080f827b83bd812077ed981a5ab43e09ad131bf3d1849bb9fbd355afc7a9.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/dcf1080f827b83bd812077ed981a5ab43e09ad131bf3d1849bb9fbd355afc7a9.jpg)

![f7533c994e753f4cd485d5161029881c4f5ecf009f6ebbab36f42b784d733339.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/images/f7533c994e753f4cd485d5161029881c4f5ecf009f6ebbab36f42b784d733339.jpg)

### Tables

![00524c748cebd2ab4caf9239b818cc3441d5c07738cdb20ae47b3eb3d0a56a6b.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/00524c748cebd2ab4caf9239b818cc3441d5c07738cdb20ae47b3eb3d0a56a6b.jpg)

![05d0ce8cd91fad9608822420f411f856787bb7fa7359aeec052166fe40d8f05d.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/05d0ce8cd91fad9608822420f411f856787bb7fa7359aeec052166fe40d8f05d.jpg)

![155c1b998907ada9ac6cc6ae3c10ef646a298b00749efd701ce2f5147f55f0bc.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/155c1b998907ada9ac6cc6ae3c10ef646a298b00749efd701ce2f5147f55f0bc.jpg)

![305f893c6bd10cd6e98874f80937d8e3a5f27956f6885bbb3092215c67b6587e.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/305f893c6bd10cd6e98874f80937d8e3a5f27956f6885bbb3092215c67b6587e.jpg)

![3b3915f39bdbb161b4e9654941320e68197e29cd4b1a9fb2442a311b64a5bcdc.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/3b3915f39bdbb161b4e9654941320e68197e29cd4b1a9fb2442a311b64a5bcdc.jpg)

![6b816f442580aad6ebe9dc158292a087db271f7e4f6d052c508e0f49c9bc01b0.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/6b816f442580aad6ebe9dc158292a087db271f7e4f6d052c508e0f49c9bc01b0.jpg)

![8fb64cabcd241de3a35436a7718fdea7f2514ae0588e5e78048adead47c623c1.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/8fb64cabcd241de3a35436a7718fdea7f2514ae0588e5e78048adead47c623c1.jpg)

![b30f4ea1451ef3f03066b44ee3534b5a6527d406069c84b24d85a907b7cda581.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/b30f4ea1451ef3f03066b44ee3534b5a6527d406069c84b24d85a907b7cda581.jpg)

![b725f44ee7bb53dd31acfea9c125daa94abadc3d0ea6e180eed21e0c69f339ba.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/b725f44ee7bb53dd31acfea9c125daa94abadc3d0ea6e180eed21e0c69f339ba.jpg)

![fa6637692e6a46d090cc6248be4ed7e19a1e6ccc785b5083153837d0ca118ff8.jpg](../icml_results/6240_Modeling%20Attributional%20Style%20at%20Scale_%20A%20Dataset%20and%20Analysis%20for%20Psychological%20Attribution%20Assessme/tables/fa6637692e6a46d090cc6248be4ed7e19a1e6ccc785b5083153837d0ca118ff8.jpg)

<a id="paper-6191"></a>

## EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions


### Images

![22deee47299e6863721f99742037477216c482196932fb1003cdc44223b55641.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/22deee47299e6863721f99742037477216c482196932fb1003cdc44223b55641.jpg)

![461da564d9ff1fa06bbb38ddcd1811ebfc1134d8438901424bfabfaf3931f605.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/461da564d9ff1fa06bbb38ddcd1811ebfc1134d8438901424bfabfaf3931f605.jpg)

![87edc3817c3b155b3087dc508151eae9b82d9827ebf1d2b2218faebb18791441.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/87edc3817c3b155b3087dc508151eae9b82d9827ebf1d2b2218faebb18791441.jpg)

![a87fec2e758ea546a90d948af5f630dae1fc8083da467f2ca40eefd15a881fd2.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/a87fec2e758ea546a90d948af5f630dae1fc8083da467f2ca40eefd15a881fd2.jpg)

![c43a55129ff1df34137bbe69ee0e2f82b6028fdccaa1171615311cd454e46d94.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/c43a55129ff1df34137bbe69ee0e2f82b6028fdccaa1171615311cd454e46d94.jpg)

![d36b144a58e7253f49ea3ca4c5b3d3d407060b57ef9be6f57a0281a98589cec4.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/images/d36b144a58e7253f49ea3ca4c5b3d3d407060b57ef9be6f57a0281a98589cec4.jpg)

### Tables

![3c5f67836577bf2a4271cd74f562ba647868c381da82ea2fc8c5b78d74770951.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/3c5f67836577bf2a4271cd74f562ba647868c381da82ea2fc8c5b78d74770951.jpg)

![42e4df2333486854a733621fc20f421ad52962756fe191afd6bac706275fd7b1.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/42e4df2333486854a733621fc20f421ad52962756fe191afd6bac706275fd7b1.jpg)

![5b953db1e26c31741f03497a917586f9c4973064b6fb56b7cc57560bf0bd8f35.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/5b953db1e26c31741f03497a917586f9c4973064b6fb56b7cc57560bf0bd8f35.jpg)

![5cba1354cbf9162d606813a1c5b007a6ec81aefcb53eb31d6823bec9735d6d07.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/5cba1354cbf9162d606813a1c5b007a6ec81aefcb53eb31d6823bec9735d6d07.jpg)

![adcffb6b7219018f8fdb0ec62355c42ccf361f3678b401571f8024e201c6d6ad.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/adcffb6b7219018f8fdb0ec62355c42ccf361f3678b401571f8024e201c6d6ad.jpg)

![ed4f49698dd4bd84deb12c6fe947566756283ada9476a32d96dc83e79cbeb199.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/ed4f49698dd4bd84deb12c6fe947566756283ada9476a32d96dc83e79cbeb199.jpg)

![fffea3b7b7c05f107cb8e2eafa8023ececcb6cd2ab422ceac498d2dc44c9379c.jpg](../icml_results/6241_EngiAgent_%20Fully%20Connected%20Coordination%20of%20LLM%20Agents%20for%20Solving%20Open-ended%20Engineering%20Problems%20wi/tables/fffea3b7b7c05f107cb8e2eafa8023ececcb6cd2ab422ceac498d2dc44c9379c.jpg)

<a id="paper-6192"></a>

## Graph-GRPO: Training Graph Flow Models with Reinforcement Learning


### Images

![0a129fc3863b18f945295800f9cb88423f6df4986191f853ffec92e4fa449c86.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/0a129fc3863b18f945295800f9cb88423f6df4986191f853ffec92e4fa449c86.jpg)

![2c214cee2067a44362af8b5aa56de78333e4dcbcbe2f70886282b67cc7c0388f.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/2c214cee2067a44362af8b5aa56de78333e4dcbcbe2f70886282b67cc7c0388f.jpg)

![2cf4f1e633975beab17b740b77c447c216bf9c494a34fd5fdf528926ca6ccf2f.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/2cf4f1e633975beab17b740b77c447c216bf9c494a34fd5fdf528926ca6ccf2f.jpg)

![4fb53e49a6a47636f38337c68e7ba7db37578fdfad81d762d67007967ab9ce0f.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/4fb53e49a6a47636f38337c68e7ba7db37578fdfad81d762d67007967ab9ce0f.jpg)

![b94a5d5a11b43fa0287e48b2e1f28718f673e2a6e0c6fd629aa96107bdfcd0d9.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/b94a5d5a11b43fa0287e48b2e1f28718f673e2a6e0c6fd629aa96107bdfcd0d9.jpg)

![e38824948a9708994feb93995e561482b516be7b91882e0e4af08702d3bda33b.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/e38824948a9708994feb93995e561482b516be7b91882e0e4af08702d3bda33b.jpg)

![f03b0de1d021ddddf963fed8a247b1a8c66d06586b975d62e59a37ef8a685449.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/f03b0de1d021ddddf963fed8a247b1a8c66d06586b975d62e59a37ef8a685449.jpg)

![f33d88ab1f0edee99d36a198efbfbbe8568736e8f7e14c09b989bc637e4d72e8.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/f33d88ab1f0edee99d36a198efbfbbe8568736e8f7e14c09b989bc637e4d72e8.jpg)

![fe44a5507f2f7db0a06354089ccc182054dc60fc0bed8559ff3d8bd08751743e.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/images/fe44a5507f2f7db0a06354089ccc182054dc60fc0bed8559ff3d8bd08751743e.jpg)

### Tables

![048807fa47dddf82986ea702a97a97e33f8939d4e957817cb1335f61c8ca5e58.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/048807fa47dddf82986ea702a97a97e33f8939d4e957817cb1335f61c8ca5e58.jpg)

![0a45719e1e27044d08baea07aa9a1bea751885d1d7debf4e6d19f3622aba3bdb.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/0a45719e1e27044d08baea07aa9a1bea751885d1d7debf4e6d19f3622aba3bdb.jpg)

![0de0eac1ba7913dbae4e5d09e0a02c6043f53d21e6ee3b02963d2cb0fe5a4a13.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/0de0eac1ba7913dbae4e5d09e0a02c6043f53d21e6ee3b02963d2cb0fe5a4a13.jpg)

![3d2604e1d01f63ca72673249b1b1bc9579cba68b4afa0e5b80d69412e18d25b8.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/3d2604e1d01f63ca72673249b1b1bc9579cba68b4afa0e5b80d69412e18d25b8.jpg)

![6bf44087402515000c28661ae7aa3f0a7bea43fd6f9d574b2e5fa51cdd974d16.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/6bf44087402515000c28661ae7aa3f0a7bea43fd6f9d574b2e5fa51cdd974d16.jpg)

![6ce9bb3c1217366c2b40da6f12ec494e9bcbd7a82c8e14e768a9bc021cf05425.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/6ce9bb3c1217366c2b40da6f12ec494e9bcbd7a82c8e14e768a9bc021cf05425.jpg)

![7a0ba52c2655d0d15cd94307cf3e35e0211be40fab17f6063d292606beaf5483.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/7a0ba52c2655d0d15cd94307cf3e35e0211be40fab17f6063d292606beaf5483.jpg)

![8a106b43ad1a3d6129f00142b0fd0fb90cb0b087e06ceacb84275f1190bae12a.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/8a106b43ad1a3d6129f00142b0fd0fb90cb0b087e06ceacb84275f1190bae12a.jpg)

![8ccad0e2285ab2b5fa45c4ea081fedc3c35f7dd6b6ac6dd6304a11066f90e25b.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/8ccad0e2285ab2b5fa45c4ea081fedc3c35f7dd6b6ac6dd6304a11066f90e25b.jpg)

![9ee3c05324d5cb50d1e32b675b122ea6aff990d327229ad4f8e1e2afb7c15b01.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/9ee3c05324d5cb50d1e32b675b122ea6aff990d327229ad4f8e1e2afb7c15b01.jpg)

![da83dcabf79d805967e015f9b0907a601846ad346315fc1769c37e7c53b17db2.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/da83dcabf79d805967e015f9b0907a601846ad346315fc1769c37e7c53b17db2.jpg)

![df7a8e0f6d85205dd68d59ea37d68ea1ccd42f225a8fa94264568ef8ae599c85.jpg](../icml_results/6242_Graph-GRPO_%20Training%20Graph%20Flow%20Models%20with%20Reinforcement%20Learning/tables/df7a8e0f6d85205dd68d59ea37d68ea1ccd42f225a8fa94264568ef8ae599c85.jpg)

<a id="paper-6193"></a>

## G$^2$TAM: Geometry Grounded Track Anything Model


### Images

![133fdeda75da36bb5a18e7dc4f7f88a75178a53d9412dd9b1969e4f58158c30d.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/133fdeda75da36bb5a18e7dc4f7f88a75178a53d9412dd9b1969e4f58158c30d.jpg)

![23738a27eb69c03ed59e752aca5c96ca6ef9d2804a46b2d3bcc03d896762c88b.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/23738a27eb69c03ed59e752aca5c96ca6ef9d2804a46b2d3bcc03d896762c88b.jpg)

![36d35595514470e97e51dbbc9b34a1c21c1116b26276e617bcd23f795b16f8b1.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/36d35595514470e97e51dbbc9b34a1c21c1116b26276e617bcd23f795b16f8b1.jpg)

![4b0dd0a601cb0e61584f2187c2c7ad1c85012194ff1b610487ce7f1eca93d277.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/4b0dd0a601cb0e61584f2187c2c7ad1c85012194ff1b610487ce7f1eca93d277.jpg)

![4d5a238429b7ab479d7f0efb858426ead5b38ad6ad585233bdee5ebd85db6f24.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/4d5a238429b7ab479d7f0efb858426ead5b38ad6ad585233bdee5ebd85db6f24.jpg)

![509ad33397d3088f01de67321e63942e0610896b9e75acbc5146d93a7a52a40a.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/509ad33397d3088f01de67321e63942e0610896b9e75acbc5146d93a7a52a40a.jpg)

![53201d7786f83fbcde774e88216c9ebb59f7b255572896efd9f00998306521d0.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/53201d7786f83fbcde774e88216c9ebb59f7b255572896efd9f00998306521d0.jpg)

![884d558f592f65b5525cff8826efcf578a4e5d9e847458bb7d3966a7cd1c6ec9.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/884d558f592f65b5525cff8826efcf578a4e5d9e847458bb7d3966a7cd1c6ec9.jpg)

![bd890156e7392bab530716a7a65290de762ec434fcaf3fd876c20da5f4cd3bb5.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/bd890156e7392bab530716a7a65290de762ec434fcaf3fd876c20da5f4cd3bb5.jpg)

![d0cfdfb7957f1a3436bae040e81b71a5a139d4f732b5c6aad8aeabd43d1703d5.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/d0cfdfb7957f1a3436bae040e81b71a5a139d4f732b5c6aad8aeabd43d1703d5.jpg)

![e65442995fdd95d0caf0e44f8e1cbd0bcdcf3ba7ca8302c7f0e8682fd93a961b.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/images/e65442995fdd95d0caf0e44f8e1cbd0bcdcf3ba7ca8302c7f0e8682fd93a961b.jpg)

### Tables

![10bb42ca9282e1f957ed2831fc15e780898ca7e3123d76b6d90dca2e16691e60.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/10bb42ca9282e1f957ed2831fc15e780898ca7e3123d76b6d90dca2e16691e60.jpg)

![171e4cd662e495ecd2f236c5f46d851e70dbf4a5065f1fa5a494194d9005362f.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/171e4cd662e495ecd2f236c5f46d851e70dbf4a5065f1fa5a494194d9005362f.jpg)

![19640a1d703c56e32e1f29e7a498e4e49cec157d78a02402bfb5176673bdc121.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/19640a1d703c56e32e1f29e7a498e4e49cec157d78a02402bfb5176673bdc121.jpg)

![1bcf69000e5a875cda027395e5a79cbf17b1615c110a8d016ab93777e521a8ca.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/1bcf69000e5a875cda027395e5a79cbf17b1615c110a8d016ab93777e521a8ca.jpg)

![374fb5bbead88e638580af91b4e4798b0308aec43de87666aa02f9da003d73e0.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/374fb5bbead88e638580af91b4e4798b0308aec43de87666aa02f9da003d73e0.jpg)

![4194a16b38ecc78005d24861cefdfda8fb6e557ac4aeaa0fe15cbf93fa6cdf73.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/4194a16b38ecc78005d24861cefdfda8fb6e557ac4aeaa0fe15cbf93fa6cdf73.jpg)

![48fe82c1fb5b9a80b74500a3d40aa24c89bc211e63ab233e00a2d931f09a6616.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/48fe82c1fb5b9a80b74500a3d40aa24c89bc211e63ab233e00a2d931f09a6616.jpg)

![700b5901a0010624554dc989b0508b9373272705117ed20125b5a74b57ad7c13.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/700b5901a0010624554dc989b0508b9373272705117ed20125b5a74b57ad7c13.jpg)

![8033b5523ecbf443788c2cb6510beb8ed77176eecfe13e8789f3f17550d38f07.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/8033b5523ecbf443788c2cb6510beb8ed77176eecfe13e8789f3f17550d38f07.jpg)

![ba049ca89704e34b24ca0e78b89d5b6b5b2f9085d1c9b5d81fefa933b9b8ce88.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/ba049ca89704e34b24ca0e78b89d5b6b5b2f9085d1c9b5d81fefa933b9b8ce88.jpg)

![bd5d28127100a3f97a1154aecd6c7206eb8a39c27a6a83b82d9245e810c5d3a6.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/bd5d28127100a3f97a1154aecd6c7206eb8a39c27a6a83b82d9245e810c5d3a6.jpg)

![cd598821080d427b4ea0f338ddec78a90b7d2f9dd11ee8002f311a6c37ff8bcf.jpg](../icml_results/6243_G%24%5E2%24TAM_%20Geometry%20Grounded%20Track%20Anything%20Model/tables/cd598821080d427b4ea0f338ddec78a90b7d2f9dd11ee8002f311a6c37ff8bcf.jpg)

<a id="paper-6194"></a>

## Pull Requests as a Training Signal for Repo-Level Code Editing


### Images

![44f37464c5f03905e70a2987f8745aaf8fbf458c2d5357e0a9eb51ab007cbf3b.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/images/44f37464c5f03905e70a2987f8745aaf8fbf458c2d5357e0a9eb51ab007cbf3b.jpg)

![ac5f1f6a1594fc7595222837d7c817466ceb92c945b81393112db7ab2d683077.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/images/ac5f1f6a1594fc7595222837d7c817466ceb92c945b81393112db7ab2d683077.jpg)

![afa6d27ef3b518eb1b7c1b380a01c3ac8842e631cdbca9c229a6e3d8e4bbab98.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/images/afa6d27ef3b518eb1b7c1b380a01c3ac8842e631cdbca9c229a6e3d8e4bbab98.jpg)

![b2dac616c50bf2ec68045fda6f9f74c623070c67db508367b64755854f27b906.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/images/b2dac616c50bf2ec68045fda6f9f74c623070c67db508367b64755854f27b906.jpg)

![c5a70763c482c6a29f83e9a911b35c79f4d0bdf997fa74789742c0204cd2f8da.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/images/c5a70763c482c6a29f83e9a911b35c79f4d0bdf997fa74789742c0204cd2f8da.jpg)

### Tables

![1386fbaf6043b3aee1d38b3e5eb2ee47052e4aad79c1af6449553bf5666ab645.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/1386fbaf6043b3aee1d38b3e5eb2ee47052e4aad79c1af6449553bf5666ab645.jpg)

![248f08d619d070e9d8400ead8bdead267bdcbbf544cc1f2d430a24ca7c1d25b7.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/248f08d619d070e9d8400ead8bdead267bdcbbf544cc1f2d430a24ca7c1d25b7.jpg)

![2eff9ef15d8c657fb520ae4b6f95b78c64c33cb21e49f0d0fcbed761cf5479f6.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/2eff9ef15d8c657fb520ae4b6f95b78c64c33cb21e49f0d0fcbed761cf5479f6.jpg)

![4e8655006ce5068b886a33b39e17afe60fa3e279706b1ac388ba183596793fd3.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/4e8655006ce5068b886a33b39e17afe60fa3e279706b1ac388ba183596793fd3.jpg)

![4f4a3c29e7fdb1495d1a3494d243cbe6bd553a136e6681fa701d950e081d3245.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/4f4a3c29e7fdb1495d1a3494d243cbe6bd553a136e6681fa701d950e081d3245.jpg)

![52d221bd4385fd9873050d95941faae80f2d9e9d85d8411c651b80e039916547.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/52d221bd4385fd9873050d95941faae80f2d9e9d85d8411c651b80e039916547.jpg)

![5520d76d7ec3bc4ac5f4a137838445c8b3ebd1f20a5f30e0401268591e1ac5a2.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/5520d76d7ec3bc4ac5f4a137838445c8b3ebd1f20a5f30e0401268591e1ac5a2.jpg)

![621f6441db233ef8c393ac8bebb014239663bc2105e473ee965904b03cfc4cef.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/621f6441db233ef8c393ac8bebb014239663bc2105e473ee965904b03cfc4cef.jpg)

![6f86eb1f8e1f3cfcc023bb5f796058fd99cd615f490b68b31e8d789040ae3cba.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/6f86eb1f8e1f3cfcc023bb5f796058fd99cd615f490b68b31e8d789040ae3cba.jpg)

![76c12cb98f2de1cb40fc2813a5ed16cfc2d0ff3e5e113a578aae5c1956d238ed.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/76c12cb98f2de1cb40fc2813a5ed16cfc2d0ff3e5e113a578aae5c1956d238ed.jpg)

![7e612aba34b910a79fe603886a4056987f36ec750a527cda7dbb8ce37af125d3.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/7e612aba34b910a79fe603886a4056987f36ec750a527cda7dbb8ce37af125d3.jpg)

![844da165564a9cf976a26ce9126997c7c55dba9e1e472f7755229657e8c456a5.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/844da165564a9cf976a26ce9126997c7c55dba9e1e472f7755229657e8c456a5.jpg)

![ab4cd7982475a65fb7625b8b3bba4df479ad08e55ab380cef755487debc1dc90.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/ab4cd7982475a65fb7625b8b3bba4df479ad08e55ab380cef755487debc1dc90.jpg)

![bad78662cb87daa086fb277d08e79159d75813acb1e7261cf9988312ed2a8200.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/bad78662cb87daa086fb277d08e79159d75813acb1e7261cf9988312ed2a8200.jpg)

![dfa06929abe062e2fbaf33760978bda0934875f68f74bed34f6ab2cce0256a38.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/dfa06929abe062e2fbaf33760978bda0934875f68f74bed34f6ab2cce0256a38.jpg)

![ecc28fc615454e06dd412466a9a4d778027553d6c2887aa964f3aadf66939f7f.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/ecc28fc615454e06dd412466a9a4d778027553d6c2887aa964f3aadf66939f7f.jpg)

![fc8e4a907c089b8e20318be5a75fc31ed39ed6c6162cd1d47302ab29700fdebd.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/fc8e4a907c089b8e20318be5a75fc31ed39ed6c6162cd1d47302ab29700fdebd.jpg)

![fc9b62e0d371cfa2a954dc74842b6af4ee048a2cc9c36c89cb71a345c0503872.jpg](../icml_results/6244_Pull%20Requests%20as%20a%20Training%20Signal%20for%20Repo-Level%20Code%20Editing/tables/fc9b62e0d371cfa2a954dc74842b6af4ee048a2cc9c36c89cb71a345c0503872.jpg)

<a id="paper-6195"></a>

## When Do Graph Foundation Models Transfer? A Data-Centric Theory


### Images

![54ea7dc02f0290cb1daccb2db8b5153f327248f626869758c1dea4e359b895e5.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/54ea7dc02f0290cb1daccb2db8b5153f327248f626869758c1dea4e359b895e5.jpg)

![61c411853f527e9b490d7444d8fccbbca5e735ffd9b0d24ac200b899add740af.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/61c411853f527e9b490d7444d8fccbbca5e735ffd9b0d24ac200b899add740af.jpg)

![8215652ec5d8afaacbefa9a4954abe3234e8754e2642014cdd5d047f5a1cf03a.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/8215652ec5d8afaacbefa9a4954abe3234e8754e2642014cdd5d047f5a1cf03a.jpg)

![86ea7fe3b4905b9c2b426b51e0888671318a30fce3985f0c481a859ffc930694.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/86ea7fe3b4905b9c2b426b51e0888671318a30fce3985f0c481a859ffc930694.jpg)

![d386c333fe07f3cd7eda9b71bbe1255d91c87f592c46c91c74f5b3088d7ffbde.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/d386c333fe07f3cd7eda9b71bbe1255d91c87f592c46c91c74f5b3088d7ffbde.jpg)

![e48dcd69e17c3ba3000e770d5dcad50f42b2ff364d5160f5dc67c22c2f9ba1f7.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/e48dcd69e17c3ba3000e770d5dcad50f42b2ff364d5160f5dc67c22c2f9ba1f7.jpg)

![e80f16c47761a18b7eddda538eef72638de65244ec837be66927da40ca03f2ab.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/e80f16c47761a18b7eddda538eef72638de65244ec837be66927da40ca03f2ab.jpg)

![f5e4b8b7c97572d52c73fcd3f2aa312987ac4716b7ef8c48fe2937d706b6594b.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/f5e4b8b7c97572d52c73fcd3f2aa312987ac4716b7ef8c48fe2937d706b6594b.jpg)

![f9ed586888515baf820eced7f673914acdd88d956554ac8e4fab282a7d8e66f5.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/images/f9ed586888515baf820eced7f673914acdd88d956554ac8e4fab282a7d8e66f5.jpg)

### Tables

![c110c813fa20dc9507216e67fa147b59008a1eb11e2f40e73ae2db5b0c9243e5.jpg](../icml_results/6245_When%20Do%20Graph%20Foundation%20Models%20Transfer_%20A%20Data-Centric%20Theory/tables/c110c813fa20dc9507216e67fa147b59008a1eb11e2f40e73ae2db5b0c9243e5.jpg)

<a id="paper-6196"></a>

## Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents


### Images

![636a9678345aa6f2110480a4be3be0d4e9264fe9eff45690471384a83447d579.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/images/636a9678345aa6f2110480a4be3be0d4e9264fe9eff45690471384a83447d579.jpg)

![75a90ee7ae0d851c2b78f0a71680bc4e0f0e1e4d267504afebada47b873915ce.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/images/75a90ee7ae0d851c2b78f0a71680bc4e0f0e1e4d267504afebada47b873915ce.jpg)

### Tables

![027c2d2a8e4c314541482953eab09f6ca9e681a53ab720091977eb64cbf61bcb.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/027c2d2a8e4c314541482953eab09f6ca9e681a53ab720091977eb64cbf61bcb.jpg)

![22ba7503082f7838963de0d4b96e41c503154df00a7f04da41bf8523b0e36efd.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/22ba7503082f7838963de0d4b96e41c503154df00a7f04da41bf8523b0e36efd.jpg)

![29dd787e43a0019f2b46379081ad7c9d0cb63ac5289541a4f65a377d54c8f3d0.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/29dd787e43a0019f2b46379081ad7c9d0cb63ac5289541a4f65a377d54c8f3d0.jpg)

![5620392da0ca0ab34362dd06ae365e97d62767e0dd61371f129d4c31c24c0b2d.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/5620392da0ca0ab34362dd06ae365e97d62767e0dd61371f129d4c31c24c0b2d.jpg)

![5c8e69f7138f9176184b1adc0b8f315f549f69030e99dd09e8d27b470e3fedcc.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/5c8e69f7138f9176184b1adc0b8f315f549f69030e99dd09e8d27b470e3fedcc.jpg)

![afd63f784a28240c0800f93c9af0d45a66e001bb9f6d7342843a58a261f980c7.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/afd63f784a28240c0800f93c9af0d45a66e001bb9f6d7342843a58a261f980c7.jpg)

![d0a28f660113c15c1b9775f99345a48a2356840cec7a2b4c9fdc21514447e602.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/d0a28f660113c15c1b9775f99345a48a2356840cec7a2b4c9fdc21514447e602.jpg)

![d5f0352557e870d35f44a295c160301208a2b15f521002cecdadbe8e24ca8a7f.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/d5f0352557e870d35f44a295c160301208a2b15f521002cecdadbe8e24ca8a7f.jpg)

![dab0cfc1221f61b3cfbc615b1dfeac735256250f8700f6c16fa6a61cd6810a01.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/dab0cfc1221f61b3cfbc615b1dfeac735256250f8700f6c16fa6a61cd6810a01.jpg)

![de47c079354d6313f326f6547b3d452ae66445ac1f4be71375fdf6ebc7808795.jpg](../icml_results/6246_Stratified%20GRPO_%20Handling%20Structural%20Heterogeneity%20in%20Reinforcement%20Learning%20of%20LLM%20Search%20Agents/tables/de47c079354d6313f326f6547b3d452ae66445ac1f4be71375fdf6ebc7808795.jpg)

<a id="paper-6197"></a>

## Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems


### Images

![1ef14c9209d8798cbd550e65efd39aa7380df51aa0e3f2fea21ab21a25e6025e.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/1ef14c9209d8798cbd550e65efd39aa7380df51aa0e3f2fea21ab21a25e6025e.jpg)

![2c0e5f0c72ee4cde5da079be70497a5d9a4c18b84fbf03698931face1adc08ea.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/2c0e5f0c72ee4cde5da079be70497a5d9a4c18b84fbf03698931face1adc08ea.jpg)

![39d016a22c63d4211b8dd3213649346c731f57a1d11b5a0ef119133fd4239159.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/39d016a22c63d4211b8dd3213649346c731f57a1d11b5a0ef119133fd4239159.jpg)

![466cffd302387be46287e95cd9824b55a5a08c4ee4eaef0ddad37b8593dbbd69.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/466cffd302387be46287e95cd9824b55a5a08c4ee4eaef0ddad37b8593dbbd69.jpg)

![570430cd693b0034d1d691c6db1acb5a1c1de35f20d1ca710a44fc06443197ee.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/570430cd693b0034d1d691c6db1acb5a1c1de35f20d1ca710a44fc06443197ee.jpg)

![6b2d7aeddef3f087a7a79135a25e9a5debfdb155a07bda2859139eda75501cfd.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/6b2d7aeddef3f087a7a79135a25e9a5debfdb155a07bda2859139eda75501cfd.jpg)

![a362458d56111feb1a7af43be5b8b7efc0c9b5c156e1d6238fafad6f19c6f13e.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/a362458d56111feb1a7af43be5b8b7efc0c9b5c156e1d6238fafad6f19c6f13e.jpg)

![bcd83481bddcd59d5809545742572046e78b823a30ccf74ec24dd4818de4ac05.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/bcd83481bddcd59d5809545742572046e78b823a30ccf74ec24dd4818de4ac05.jpg)

![c0487c1f1ac731fa29dc7d1c4e7fac044b5d7df93cdd99ac1fda34e507a3fe41.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/c0487c1f1ac731fa29dc7d1c4e7fac044b5d7df93cdd99ac1fda34e507a3fe41.jpg)

![f8649c3697f412ffa9e1790a62e671ae3d0bc6713b699098dc27bd83dac6bfea.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/images/f8649c3697f412ffa9e1790a62e671ae3d0bc6713b699098dc27bd83dac6bfea.jpg)

### Tables

![5904c3f5129eb98f85b7ce15322c630b966f9602a14224fc36180c1530adf4a7.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/5904c3f5129eb98f85b7ce15322c630b966f9602a14224fc36180c1530adf4a7.jpg)

![74fdd7d79db1b1718d9c95badbff55b00512092faa28dffe0301b179c74a67eb.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/74fdd7d79db1b1718d9c95badbff55b00512092faa28dffe0301b179c74a67eb.jpg)

![7f492bc62c33b442be5900d7ce6c7b36173c50ab31df9df378b8e06cce97afeb.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/7f492bc62c33b442be5900d7ce6c7b36173c50ab31df9df378b8e06cce97afeb.jpg)

![8e1fb8bc50981c4c4f3df0210d08f562b36b89c64bb79089f875d2d8f209a335.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/8e1fb8bc50981c4c4f3df0210d08f562b36b89c64bb79089f875d2d8f209a335.jpg)

![e0343e0c31e0d842b01ba175a43151d14582e8d96998232adfee198ba3e257a5.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/e0343e0c31e0d842b01ba175a43151d14582e8d96998232adfee198ba3e257a5.jpg)

![f7a68ed45aacc3df4f9c9318d57e14e2a21d01025f1fe0ed53732a9ced61bf4b.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/f7a68ed45aacc3df4f9c9318d57e14e2a21d01025f1fe0ed53732a9ced61bf4b.jpg)

![fd7c5817f319b3ae6f9d3457f77b13f7f8e44c97718b1f13f477a0a543119d3d.jpg](../icml_results/6247_Recognize%20Your%20Orchestrator_%20An%20Entropy%20Dynamics%20Perspective%20for%20LLM%20Multi-Agent%20Systems/tables/fd7c5817f319b3ae6f9d3457f77b13f7f8e44c97718b1f13f477a0a543119d3d.jpg)

<a id="paper-6198"></a>

## From Guessing to Placeholding: A Cost-Theoretic Framework for Uncertainty-Aware Code Completion


### Images

![36cb7fe11cf4a0883995232b578dc1316427e5c8585f0ac5d1dae9325475ed74.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/images/36cb7fe11cf4a0883995232b578dc1316427e5c8585f0ac5d1dae9325475ed74.jpg)

![47636a47a94a0cf903b6fa82de345d25eaf61f7c7bdde23cea9172b02e434415.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/images/47636a47a94a0cf903b6fa82de345d25eaf61f7c7bdde23cea9172b02e434415.jpg)

![61daebad5b44b1e36d83fc832c278502c68476ca9e13b128805c8958fd6ef820.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/images/61daebad5b44b1e36d83fc832c278502c68476ca9e13b128805c8958fd6ef820.jpg)

### Tables

![0948bdc9616ab37d61ea1a9f0df1ed9509a7b16a22d391352ef6a516942c90f7.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/0948bdc9616ab37d61ea1a9f0df1ed9509a7b16a22d391352ef6a516942c90f7.jpg)

![2b4177e9d29e55684148440c0ff6483a202905565090d3d0656dce300d514e1f.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/2b4177e9d29e55684148440c0ff6483a202905565090d3d0656dce300d514e1f.jpg)

![4c9573c7418ae518d844fcd895259e5e3c7cfea7d974ca5f8ee3578764a69a59.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/4c9573c7418ae518d844fcd895259e5e3c7cfea7d974ca5f8ee3578764a69a59.jpg)

![52806a89a38c773fafb289b9d12247e517c5bb203908697de80751a44cccd513.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/52806a89a38c773fafb289b9d12247e517c5bb203908697de80751a44cccd513.jpg)

![6600d318c529e205a59de2c8fc8848dfd4d20946ff094890ce497e9e06109d1c.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/6600d318c529e205a59de2c8fc8848dfd4d20946ff094890ce497e9e06109d1c.jpg)

![679945731b1812f70ab5e1fbaa8267bdc8036cfbf1f451915cfc29c8c9417301.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/679945731b1812f70ab5e1fbaa8267bdc8036cfbf1f451915cfc29c8c9417301.jpg)

![a4af92ad43847cf1f2265e64c584ef7550e00b832687ded0464366467f5f837e.jpg](../icml_results/6248_From%20Guessing%20to%20Placeholding_%20A%20Cost-Theoretic%20Framework%20for%20Uncertainty-Aware%20Code%20Completion/tables/a4af92ad43847cf1f2265e64c584ef7550e00b832687ded0464366467f5f837e.jpg)

<a id="paper-6199"></a>

## Simple Denoising Diffusion Language Models


### Images

![326ea678ed87ba0300ec8117bd5e1605af802f2d66bbf8084c6df23fa7e02f0d.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/images/326ea678ed87ba0300ec8117bd5e1605af802f2d66bbf8084c6df23fa7e02f0d.jpg)

![351a5d434d79b5916992e7123626364707f1412e64452a9a027a3925220dc4c9.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/images/351a5d434d79b5916992e7123626364707f1412e64452a9a027a3925220dc4c9.jpg)

![6cf7bd664a40938fc880485052988e6bfb73cfc5c11f95f1c3040724611ae20c.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/images/6cf7bd664a40938fc880485052988e6bfb73cfc5c11f95f1c3040724611ae20c.jpg)

![81fb97603118ba9be1d7e24894e5a3263baf4250313c3525c4544681640a0b2b.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/images/81fb97603118ba9be1d7e24894e5a3263baf4250313c3525c4544681640a0b2b.jpg)

![8f44abf75281514c9aa8dc25bc2ab6437dbd6ed2c03f4cb5f0030723028a0f12.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/images/8f44abf75281514c9aa8dc25bc2ab6437dbd6ed2c03f4cb5f0030723028a0f12.jpg)

### Tables

![4858f4042cf9e653c698fb4783500ff042fc990bf547bd12a08a58191f27bbad.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/4858f4042cf9e653c698fb4783500ff042fc990bf547bd12a08a58191f27bbad.jpg)

![4a0727ef8c60f5818659346a593b8a85b3e840e762a99b8d9e7453b232bdcc21.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/4a0727ef8c60f5818659346a593b8a85b3e840e762a99b8d9e7453b232bdcc21.jpg)

![6aae2c966568a615270f581957479e5689c27c3ebb07567da89f275355aa0a43.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/6aae2c966568a615270f581957479e5689c27c3ebb07567da89f275355aa0a43.jpg)

![b4cd456d9207b9cb91cc0f755dce475a96f91c902c7a47738073e7548938760e.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/b4cd456d9207b9cb91cc0f755dce475a96f91c902c7a47738073e7548938760e.jpg)

![bf03cfc9635dee5c3cd2e249d1ea74fe48c877530be0590686a6240327ff2294.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/bf03cfc9635dee5c3cd2e249d1ea74fe48c877530be0590686a6240327ff2294.jpg)

![cc3f1ece86bbfaa4d13f80e400c98ffc2af4a3def3d56c440b11a431e7037438.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/cc3f1ece86bbfaa4d13f80e400c98ffc2af4a3def3d56c440b11a431e7037438.jpg)

![d8a723a734629c7382cecc80468b97a58ae7bd811e00e6b5d7bb5c164c2d6489.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/d8a723a734629c7382cecc80468b97a58ae7bd811e00e6b5d7bb5c164c2d6489.jpg)

![e8b79c0dee7a11b52a4ec42540f743cf65d9af7514409647d76619836f9fd60c.jpg](../icml_results/6249_Simple%20Denoising%20Diffusion%20Language%20Models/tables/e8b79c0dee7a11b52a4ec42540f743cf65d9af7514409647d76619836f9fd60c.jpg)

<a id="paper-6200"></a>

## Learning Transferable Interaction Primitives from Game Videos for Humanoid Locomotion


### Images

![02136b9790dc024c0d5d738e600f4253283f1c809ea9fc9b33b474f80868766d.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/02136b9790dc024c0d5d738e600f4253283f1c809ea9fc9b33b474f80868766d.jpg)

![1c3d4a0c767c2003dbf8f9017aa5e528b88cd2386017748713e63cc54e1674d2.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/1c3d4a0c767c2003dbf8f9017aa5e528b88cd2386017748713e63cc54e1674d2.jpg)

![332e10a21d95bf0f257a8d652d8e7557746eae367f886adaa9a3545a8ce90139.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/332e10a21d95bf0f257a8d652d8e7557746eae367f886adaa9a3545a8ce90139.jpg)

![37c0ecb8ad0265bab848e4be690edc1ff680ce49022a656dcdceecfea59e50b2.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/37c0ecb8ad0265bab848e4be690edc1ff680ce49022a656dcdceecfea59e50b2.jpg)

![4102ba724cab50966e1f369102ab2f4f9baf2f263c76df84a80bbf6a48eda97c.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/4102ba724cab50966e1f369102ab2f4f9baf2f263c76df84a80bbf6a48eda97c.jpg)

![5941265369859b6b71bf41a98de25701d021c68156d1a4218e722319598bfdce.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/5941265369859b6b71bf41a98de25701d021c68156d1a4218e722319598bfdce.jpg)

![97362595a9a5afaecf4e584fec626eec6d839132d1ad80f01cd3d61d3d64402f.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/97362595a9a5afaecf4e584fec626eec6d839132d1ad80f01cd3d61d3d64402f.jpg)

![98efe2240f261f2db56c6b4da7f59a11926001bf574053f399008983c538ad58.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/98efe2240f261f2db56c6b4da7f59a11926001bf574053f399008983c538ad58.jpg)

![ad09e83b755b68cd7f104641d9651bf9077bb614951424b6255ccc221827747c.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/ad09e83b755b68cd7f104641d9651bf9077bb614951424b6255ccc221827747c.jpg)

![f8dfbc0e7ea84c45759eda463c17f42a154727e7bac6930338375eca508adfa0.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/images/f8dfbc0e7ea84c45759eda463c17f42a154727e7bac6930338375eca508adfa0.jpg)

### Tables

![3462222df63c566ca9b0e786bb61678678a84c43cf78163f3d6e1637612cbcc8.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/3462222df63c566ca9b0e786bb61678678a84c43cf78163f3d6e1637612cbcc8.jpg)

![4bef532587c0f1125ec6047e62ca269ccdbecc33290dd3a06838f20709ed1d11.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/4bef532587c0f1125ec6047e62ca269ccdbecc33290dd3a06838f20709ed1d11.jpg)

![5f4ddb558a9355055863f07b4b7f95df0dcd3e37f0071bba69dc82ee188704c3.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/5f4ddb558a9355055863f07b4b7f95df0dcd3e37f0071bba69dc82ee188704c3.jpg)

![6d16346488172008fb8fb7d90121e34e01fa9399cbc851912fa888cec87d8f7f.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/6d16346488172008fb8fb7d90121e34e01fa9399cbc851912fa888cec87d8f7f.jpg)

![881b39fb86aaf5c3299c57d8f848f828b958de85bb9b0e40e298e7ef693a2fa3.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/881b39fb86aaf5c3299c57d8f848f828b958de85bb9b0e40e298e7ef693a2fa3.jpg)

![cc884eeac89d7a2cde1e53bae69c2873343814d9b7c45f634e7e24e833f47b8e.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/cc884eeac89d7a2cde1e53bae69c2873343814d9b7c45f634e7e24e833f47b8e.jpg)

![d0f4e52eaf9eb8cfeebe223669b2a66511920f83e9168166bebe566bf36f5aca.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/d0f4e52eaf9eb8cfeebe223669b2a66511920f83e9168166bebe566bf36f5aca.jpg)

![d8af751b1b2a6f913f54beec04cdf4b7d679daf863277c421ff65e0c107053bb.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/d8af751b1b2a6f913f54beec04cdf4b7d679daf863277c421ff65e0c107053bb.jpg)

![fca30e6e88afd3754ab629bd53bc6620e4d037d8cffde5f29b8f7b8fb418fcda.jpg](../icml_results/6250_Learning%20Transferable%20Interaction%20Primitives%20from%20Game%20Videos%20for%20Humanoid%20Locomotion/tables/fca30e6e88afd3754ab629bd53bc6620e4d037d8cffde5f29b8f7b8fb418fcda.jpg)

<a id="paper-6201"></a>

## Alethia: a Foundational Encoder for Voice Deepfakes


### Images

![2a1fb35ae97a5d6699b1c38454cbaeb0ccce3562ebee8f489301961359e957da.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/2a1fb35ae97a5d6699b1c38454cbaeb0ccce3562ebee8f489301961359e957da.jpg)

![2af6fe36c8e395258714130a6dfd5372e5b0b41714733f62e4f10646525bd4b5.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/2af6fe36c8e395258714130a6dfd5372e5b0b41714733f62e4f10646525bd4b5.jpg)

![3e98b0155c61c1a9d73dae50750e4502729b752bc0a5badfe2c36537b63ffcc3.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/3e98b0155c61c1a9d73dae50750e4502729b752bc0a5badfe2c36537b63ffcc3.jpg)

![6d15c57e6f78a98ded45f307698fe252f5753d69b0ee3955498da715e0f3de4f.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/6d15c57e6f78a98ded45f307698fe252f5753d69b0ee3955498da715e0f3de4f.jpg)

![756dabcaea54e46ab2b043ceff472a8f416c5b2701c901f43d9e30b96b6f80b3.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/756dabcaea54e46ab2b043ceff472a8f416c5b2701c901f43d9e30b96b6f80b3.jpg)

![7c20ac1fb042a4b397b54ec655bbc7681fe173bbc9faa922d3cd056e887142fc.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/7c20ac1fb042a4b397b54ec655bbc7681fe173bbc9faa922d3cd056e887142fc.jpg)

![b4987027185671caba79baeced44d73b128daf5a0dd0b65de0ac63a4cfc681cc.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/images/b4987027185671caba79baeced44d73b128daf5a0dd0b65de0ac63a4cfc681cc.jpg)

### Tables

![09497cf76a601988ddd4cc096b83de753655d84ce3977ae6eca353f6c22b96e5.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/09497cf76a601988ddd4cc096b83de753655d84ce3977ae6eca353f6c22b96e5.jpg)

![17c58e6f7d3011eeefda2abac154e1c4cb578c0e60bb917eba5256ea6399ae6a.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/17c58e6f7d3011eeefda2abac154e1c4cb578c0e60bb917eba5256ea6399ae6a.jpg)

![298dc211a440f8d01459cfcb9deffd74bfe5e5f628472b133508b69512bb26b4.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/298dc211a440f8d01459cfcb9deffd74bfe5e5f628472b133508b69512bb26b4.jpg)

![3463e4e0ae1b35051a7fbc05b9933cdaca0d3ce52f99a713f6d608eef56eb6fb.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/3463e4e0ae1b35051a7fbc05b9933cdaca0d3ce52f99a713f6d608eef56eb6fb.jpg)

![351022c689314961f3376dd42576d749f1eb58e72ba730828ca78cf5f93744e7.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/351022c689314961f3376dd42576d749f1eb58e72ba730828ca78cf5f93744e7.jpg)

![4461138f4fa168aa5bdd593697df87884c79a7b28e589f758fec4b07b937725b.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/4461138f4fa168aa5bdd593697df87884c79a7b28e589f758fec4b07b937725b.jpg)

![4ce5ff8127e240cc7232b0442a0b02073aba3571470b5ecd59bfeead53c5b233.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/4ce5ff8127e240cc7232b0442a0b02073aba3571470b5ecd59bfeead53c5b233.jpg)

![6ab3ca29702c0ed649c78210e5944f16fab5432d44a4e8a90ec7143a81e78714.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/6ab3ca29702c0ed649c78210e5944f16fab5432d44a4e8a90ec7143a81e78714.jpg)

![7541619e012e55d74e5aac206f704e16f7e9232e3028a657eca00e2b16ebb92b.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/7541619e012e55d74e5aac206f704e16f7e9232e3028a657eca00e2b16ebb92b.jpg)

![8f3f14c720763f33f1f7f5be3ab502c1fa5131b72969c6c286c709a1bbc6aa67.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/8f3f14c720763f33f1f7f5be3ab502c1fa5131b72969c6c286c709a1bbc6aa67.jpg)

![94e77fcf2e2512b128c3c373c63b332b5947b6c3dda5e7c5183bd7757bf4b14b.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/94e77fcf2e2512b128c3c373c63b332b5947b6c3dda5e7c5183bd7757bf4b14b.jpg)

![a564c0f5cb8848bfa3eec3c7c1bad72d263111f1c5ac80c6b7990e971818b081.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/a564c0f5cb8848bfa3eec3c7c1bad72d263111f1c5ac80c6b7990e971818b081.jpg)

![b002337b4837a7d9dcad7cc0dee7e49dca3dbebbd305475dbfb3cc5934d7581e.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/b002337b4837a7d9dcad7cc0dee7e49dca3dbebbd305475dbfb3cc5934d7581e.jpg)

![c8fefe7e589757d02a090347600af765f72d3d8df973e25fa782da02ac3f80c5.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/c8fefe7e589757d02a090347600af765f72d3d8df973e25fa782da02ac3f80c5.jpg)

![ce0f066625bb3c013ba27c26d2be4d6339eb1c75cfda18f6e638a9043286a353.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/ce0f066625bb3c013ba27c26d2be4d6339eb1c75cfda18f6e638a9043286a353.jpg)

![d3566b9e58afff93c9fd360dafff9db1fae1d661548198b5b381c111182a454d.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/d3566b9e58afff93c9fd360dafff9db1fae1d661548198b5b381c111182a454d.jpg)

![ea512a979d183442ccfa663b27ebef3ae264a191fc1cd77bf2c7529e86bf8bc6.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/ea512a979d183442ccfa663b27ebef3ae264a191fc1cd77bf2c7529e86bf8bc6.jpg)

![f166f995c42482648eef8f2997866769a34a942cba21a36819c2a139f2155b6a.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/f166f995c42482648eef8f2997866769a34a942cba21a36819c2a139f2155b6a.jpg)

![f371e6a3eebebbf0067a7788bff3b6211c370b23a50dd69aeb1024a5aa35e598.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/f371e6a3eebebbf0067a7788bff3b6211c370b23a50dd69aeb1024a5aa35e598.jpg)

![f502c8f0b61512ce5209d1d5821324674dff2f73cc4745b24fc8349d81b02dfb.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/f502c8f0b61512ce5209d1d5821324674dff2f73cc4745b24fc8349d81b02dfb.jpg)

![fd1e3530ee27535d36865513c18495bb0de5bde5e6b022ffc40b26d0e80ba193.jpg](../icml_results/6252_Alethia_%20a%20Foundational%20Encoder%20for%20Voice%20Deepfakes/tables/fd1e3530ee27535d36865513c18495bb0de5bde5e6b022ffc40b26d0e80ba193.jpg)

<a id="paper-6202"></a>

## Learning-augmented Rent-or-Buy with a Sample


### Images

![16afc64ec163af0590e1ad792d0a0e4fc36c8bff506253d3ab6cd7a9c46ebfa5.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/16afc64ec163af0590e1ad792d0a0e4fc36c8bff506253d3ab6cd7a9c46ebfa5.jpg)

![248792c13f10d0f3f8405ddf993975e30f0a0bc65b5eea7136d52c2fead6bc1e.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/248792c13f10d0f3f8405ddf993975e30f0a0bc65b5eea7136d52c2fead6bc1e.jpg)

![35baa293e7602be777d1a4f474c0b39ab4ec020b40e91d9bff8045cec4e5be34.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/35baa293e7602be777d1a4f474c0b39ab4ec020b40e91d9bff8045cec4e5be34.jpg)

![514cd8d2ae1ca53889ec4d7130869db9694c6bfb70caf85a5959b8be73aceb9e.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/514cd8d2ae1ca53889ec4d7130869db9694c6bfb70caf85a5959b8be73aceb9e.jpg)

![61b2558c890b0e656ee974aeabbc2a3950b36adf5f39e834a014769985e9015e.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/61b2558c890b0e656ee974aeabbc2a3950b36adf5f39e834a014769985e9015e.jpg)

![6d47e1bb289bcbd9b5a0934e583d78c23d77dfc96ff75a0039100bc047b44d97.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/6d47e1bb289bcbd9b5a0934e583d78c23d77dfc96ff75a0039100bc047b44d97.jpg)

![8b40730f2b61ca6e6903a0c14584a0ac2d826eee43effa43e1927b54dcacb3a2.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/8b40730f2b61ca6e6903a0c14584a0ac2d826eee43effa43e1927b54dcacb3a2.jpg)

![9cd7d6c964e31f55914f563db9586c50fa77d34a2482c530570f5cf835a29b2c.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/9cd7d6c964e31f55914f563db9586c50fa77d34a2482c530570f5cf835a29b2c.jpg)

![ad5f8d68a918179d6b646f45e4952ce53e5d5694b8bdedd90c6555f55cc37b1c.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/ad5f8d68a918179d6b646f45e4952ce53e5d5694b8bdedd90c6555f55cc37b1c.jpg)

![d08cb3f04de6525edafbe8643172d781d0b8a5595a80f08c8d7dd3fbb0e8c0aa.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/d08cb3f04de6525edafbe8643172d781d0b8a5595a80f08c8d7dd3fbb0e8c0aa.jpg)

![d24f622c197ea3967dcf9b18b946797652706c960aeb04adb9eca0a1661721fc.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/d24f622c197ea3967dcf9b18b946797652706c960aeb04adb9eca0a1661721fc.jpg)

![d3573188438f40948c4278ff0399e2b0f3e661c2d21b3d9a895cf0787d5ce51f.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/d3573188438f40948c4278ff0399e2b0f3e661c2d21b3d9a895cf0787d5ce51f.jpg)

![ed13d38428bc28cc740d37db271ad7eb14c2da839e31966735f39bd7004b7f46.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/ed13d38428bc28cc740d37db271ad7eb14c2da839e31966735f39bd7004b7f46.jpg)

![fd8d1aeb60665f795fcb2ce04607e263d90d0f00ab5e3106637374785372403f.jpg](../icml_results/6253_Learning-augmented%20Rent-or-Buy%20with%20a%20Sample/images/fd8d1aeb60665f795fcb2ce04607e263d90d0f00ab5e3106637374785372403f.jpg)

<a id="paper-6203"></a>

## EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation


### Images

![16e39c1e49f166bf4877bd87c70c0a801911341605e449e91e0abb21f1343e00.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/16e39c1e49f166bf4877bd87c70c0a801911341605e449e91e0abb21f1343e00.jpg)

![2f8a6ecb6e17ac5af217bb78a76bb2d0cdb82a6c4455a130280ed38846bd9f4e.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/2f8a6ecb6e17ac5af217bb78a76bb2d0cdb82a6c4455a130280ed38846bd9f4e.jpg)

![63405f7075a1e1f89ecf2481337bfce9301d8346fef36715037181a1104a1c28.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/63405f7075a1e1f89ecf2481337bfce9301d8346fef36715037181a1104a1c28.jpg)

![63bb1fb6ec7a11bfcaa52bc65ec027150ed3d88438e35be87eaf0eeea2d2655b.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/63bb1fb6ec7a11bfcaa52bc65ec027150ed3d88438e35be87eaf0eeea2d2655b.jpg)

![65e53b2a66ec24aadf39976783a6daf12876feef74fe3b21d6b6a858f0caaea2.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/65e53b2a66ec24aadf39976783a6daf12876feef74fe3b21d6b6a858f0caaea2.jpg)

![7da148be46fb4f643a463a632ab1b5f96fc108451fbe956d8bfe28018a0a01ba.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/7da148be46fb4f643a463a632ab1b5f96fc108451fbe956d8bfe28018a0a01ba.jpg)

![d7fed5b3ff659e7cef438856d7db4f455e8b6f100882b7be2d0040540974f03c.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/images/d7fed5b3ff659e7cef438856d7db4f455e8b6f100882b7be2d0040540974f03c.jpg)

### Tables

![38fe0c87bffdc238167c8df8984a51fcc61f5c40a85bdad42c6fa9aa4ede1c31.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/38fe0c87bffdc238167c8df8984a51fcc61f5c40a85bdad42c6fa9aa4ede1c31.jpg)

![565ca612762abf7151e82893a3a508337d803cb5b7fee91fa344d251a26b134e.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/565ca612762abf7151e82893a3a508337d803cb5b7fee91fa344d251a26b134e.jpg)

![97345a075f65ff434bb591f28b8477905c6fe2bc0178d85cd782e27340a56390.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/97345a075f65ff434bb591f28b8477905c6fe2bc0178d85cd782e27340a56390.jpg)

![c725823d3650edb7eda4241febe6e5773660615aa18cf69ae4209ad4d194ce21.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/c725823d3650edb7eda4241febe6e5773660615aa18cf69ae4209ad4d194ce21.jpg)

![f54333976a080c74755fb77739b16b41b2845957cb07e32ce826907b91a3f2d6.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/f54333976a080c74755fb77739b16b41b2845957cb07e32ce826907b91a3f2d6.jpg)

![fdbb25311eb24058f330d388896f6269c329cbb8bd7887ef5b3379f93eca1365.jpg](../icml_results/6254_EPS3D_%20End-to-End%20Feed-Forward%203D%20Panoptic%20Segmentation/tables/fdbb25311eb24058f330d388896f6269c329cbb8bd7887ef5b3379f93eca1365.jpg)

<a id="paper-6204"></a>

## FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects

### Images

![02d96adb1457933260a80b675258997ceea78591260c8aed2ce1a2bd496bba29.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/02d96adb1457933260a80b675258997ceea78591260c8aed2ce1a2bd496bba29.jpg)

![038de44d51cbaf903310e07f3080ff446147532ff8a761148948f30c6af0505b.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/038de44d51cbaf903310e07f3080ff446147532ff8a761148948f30c6af0505b.jpg)

![07e153044403d7542540a49add57bcd1ea03a8024c061c2ca2bd093d7b3b4c79.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/07e153044403d7542540a49add57bcd1ea03a8024c061c2ca2bd093d7b3b4c79.jpg)

![64ea07969449d8ac308d0311455c531be9016e0b6f26ece880d444000af25cc7.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/64ea07969449d8ac308d0311455c531be9016e0b6f26ece880d444000af25cc7.jpg)

![6cb41cbd5136ffedb834f682dab90a9a986289620ed8b7249e92488951a71440.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/6cb41cbd5136ffedb834f682dab90a9a986289620ed8b7249e92488951a71440.jpg)

![72595a0090461f22c874187942e384410d0c7b62439464afc13187a6a049fdd6.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/72595a0090461f22c874187942e384410d0c7b62439464afc13187a6a049fdd6.jpg)

![a231e6da45dbe14d1b4793cf5ca8f1e8954b7432f7692b1ea8ad8eeb72ff14fd.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/a231e6da45dbe14d1b4793cf5ca8f1e8954b7432f7692b1ea8ad8eeb72ff14fd.jpg)

![b185f260b5b8e6c93ac31b571c9fe74a5ae50ba3426dbfeaf8424045e0999b47.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/b185f260b5b8e6c93ac31b571c9fe74a5ae50ba3426dbfeaf8424045e0999b47.jpg)

![bb438ea421869d0afad99270caa130b4e7715d52ca2ec730bea234d1959f4465.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/bb438ea421869d0afad99270caa130b4e7715d52ca2ec730bea234d1959f4465.jpg)

![c20f16d385f0f0c43d14500cb01b678ea8d47b10a11cf06596dfd692ed14a887.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/c20f16d385f0f0c43d14500cb01b678ea8d47b10a11cf06596dfd692ed14a887.jpg)

![c490c52242fcb3fec98c1f90f3547bfbf57bdcd8d9af5cc44bf742490a09986c.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/c490c52242fcb3fec98c1f90f3547bfbf57bdcd8d9af5cc44bf742490a09986c.jpg)

![cabaa840e56fc3c1021489b663609b84dad4638213a8ddc2230f858f60cb7904.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/cabaa840e56fc3c1021489b663609b84dad4638213a8ddc2230f858f60cb7904.jpg)

![d8c8ea164a541a3929149ef5b41f9815b6a5efe32ef4d274ab80ec7b8b546b2c.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/d8c8ea164a541a3929149ef5b41f9815b6a5efe32ef4d274ab80ec7b8b546b2c.jpg)

![d9fe5d419105ec308e785e8a6ac18842898509bf866456c8993e346910da523c.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/d9fe5d419105ec308e785e8a6ac18842898509bf866456c8993e346910da523c.jpg)

![f781de21a832771e3fce1a5cce434cf241c322d2d2db5477bfc4cc9d3a304812.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/images/f781de21a832771e3fce1a5cce434cf241c322d2d2db5477bfc4cc9d3a304812.jpg)

### Tables

![1cd07ee0bb7953c7447755a7316b6a2fc71fbe3c5558f1f481986661b3d0eb5d.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/1cd07ee0bb7953c7447755a7316b6a2fc71fbe3c5558f1f481986661b3d0eb5d.jpg)

![50e71589e809c3ca5f1f07d3f76841c461a25dc2511c214207251eff97f084cc.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/50e71589e809c3ca5f1f07d3f76841c461a25dc2511c214207251eff97f084cc.jpg)

![64aeda5d44d5b78d8642876994d3b821d9778f7d0267a04e0a6594fffd39a2fe.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/64aeda5d44d5b78d8642876994d3b821d9778f7d0267a04e0a6594fffd39a2fe.jpg)

![aac9e211791e60668e600131a427259a98b063eb55108f30ea37cada4512520c.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/aac9e211791e60668e600131a427259a98b063eb55108f30ea37cada4512520c.jpg)

![b5add1830a3b5a7d0bf81c9cde2d136ae748b707be4f2341ec45eecaa88552db.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/b5add1830a3b5a7d0bf81c9cde2d136ae748b707be4f2341ec45eecaa88552db.jpg)

![e621d2504a39b581ed70a7f7b93523b9e2b4a9b2a1900db0d16406c2f5fd770c.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/e621d2504a39b581ed70a7f7b93523b9e2b4a9b2a1900db0d16406c2f5fd770c.jpg)

![e67cf5b0b82cc2f2f8995d48950d06a8e1393abc36aac7bee26931af56b0c804.jpg](../icml_results/6255_FlatLab_%20A%20Unified%20Methodology%20Framework%20and%20Simulation-Based%20Benchmark%20for%20Robotic%20Manipulation%20of%20/tables/e67cf5b0b82cc2f2f8995d48950d06a8e1393abc36aac7bee26931af56b0c804.jpg)
