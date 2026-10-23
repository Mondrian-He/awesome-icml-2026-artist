# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 167 of 197

## 目录 (Table of Contents)

1. [MARS-SQL: A Multi-Agent Reinforcement Learning Framework For Text-To-SQL](#paper-5479)
2. [SSDCN: Spatial-Spectral Dual-Clustering-based Network for Hyperspectral Image Super-resolution](#paper-5480)
3. [Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention](#paper-5481)
4. [Zero-source LLM Hallucination Detection with Human-like Criteria Probing](#paper-5482)
5. [TMD-Bench: A Multi-Level Evaluation Paradigm for Music–Dance Co-Generation](#paper-5483)
6. [RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization](#paper-5484)
7. [Return-to-Go Is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](#paper-5485)
8. [NBCG: Nash-Bargained Causal Game for Long-Tailed Multi-Label NLP](#paper-5486)
9. [Stable Velocity: A Variance Perspective on Flow Matching](#paper-5487)
10. [EpiTwin: Spatiotemporal Graph Transformers for Epileptic sEEG Signal Reconstruction](#paper-5488)
11. [ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling](#paper-5489)
12. [Adaptive Multiscale Binary Expansion Tests for Independence](#paper-5490)
13. [Multi-Objective Protein Design via Memory-Aware Test-Time Scaling in Diffusion Models](#paper-5491)
14. [Profiling the Irrational Agent: Cognitive Modeling of LLM Behaviors in Sequential Jailbreaks](#paper-5492)
15. [Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction](#paper-5493)
16. [Shapley Regularized Neural Granger Causality](#paper-5494)
17. [Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization](#paper-5495)
18. [Prompt Reinjection: Alleviating Prompt Forgetting in Multimodal Diffusion Transformers](#paper-5496)
19. [ParamMem: Augmenting Language Agents with Parametric Reflective Memory](#paper-5497)
20. [Beyond Looking Up, Try Looking Around: Harmonizing Global Structure and Local Consistency in Optimal Transport for Short Text Clustering](#paper-5498)
21. [H$^2$CL: Heterogeneity-Aware Hypergraph Contrastive Learning for Robust Representation Learning](#paper-5499)
22. [Integrated Episodic and Semantic Memory via Modulating Transformer FeedForward Layers](#paper-5500)
23. [HieraMAS: Optimizing Intra-Node LLM Mixtures and Inter-Node Topology for Multi-Agent Systems](#paper-5501)
24. [Role-Level Inductive Bias for Cross-Task Generalization in Multi-Agent Reinforcement Learning](#paper-5502)
25. [SLAT: Segment-Level Adaptive Trimming for Efficient CoT Reasoning](#paper-5503)
26. [The Perception–Physics Paradox: Probing Scientific Alignment with TC-Bench](#paper-5504)
27. [MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery](#paper-5505)
28. [Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](#paper-5506)
29. [Identifiable Nonlinear Differentiable Causal Discovery via Independence and Adaptive Group Sparsity](#paper-5507)
30. [TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions](#paper-5508)
31. [CAReDiO: Enhancing Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization](#paper-5509)
32. [MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline](#paper-5510)
33. [A Linearly Convergent Proximal Subgradient Algorithm for Sparse Portfolio Optimization with Transaction Cost](#paper-5511)

---


<a id="paper-5479"></a>

## MARS-SQL: A Multi-Agent Reinforcement Learning Framework For Text-To-SQL

### Images

![038ddb4927de64fd68fe78896359c02a9cb273722e3bfff4b37690f479ff84c3.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/038ddb4927de64fd68fe78896359c02a9cb273722e3bfff4b37690f479ff84c3.jpg)

![31b0ba5d4150cee5a781a068aadcfa335b812aa70a345e5edacdf681238060f0.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/31b0ba5d4150cee5a781a068aadcfa335b812aa70a345e5edacdf681238060f0.jpg)

![c8899ffdb7b308712b85d578998180268ef8823d36febacae70e84f2d6ef446f.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/c8899ffdb7b308712b85d578998180268ef8823d36febacae70e84f2d6ef446f.jpg)

![cb17792a7f2982d4c2983050b7e15a23968c6509b4d9fa8a3cff4b1578804cde.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/cb17792a7f2982d4c2983050b7e15a23968c6509b4d9fa8a3cff4b1578804cde.jpg)

![e4954973ab3e1934bb641a4525392356bdb6638fc2bb06cd6f143c6b7f8105c6.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/e4954973ab3e1934bb641a4525392356bdb6638fc2bb06cd6f143c6b7f8105c6.jpg)

![f408ec9a58e2c92649439ee1463f8f7fac1f91c38aaee74c5149fb801436c797.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/f408ec9a58e2c92649439ee1463f8f7fac1f91c38aaee74c5149fb801436c797.jpg)

![fc2c4f2760b62628a24c605189f0ff980aa7cbad970af3cdaf50525d00414073.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/fc2c4f2760b62628a24c605189f0ff980aa7cbad970af3cdaf50525d00414073.jpg)

### Tables

![0a301f01c69fd6fc06b29fac448d1a037bab0fcf9da3be3d68db9ffa08d1f5f9.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/0a301f01c69fd6fc06b29fac448d1a037bab0fcf9da3be3d68db9ffa08d1f5f9.jpg)

![189c98cfca5e6fb86657e340348f2fd5da03b8eee524e99bae7e46adb86677a5.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/189c98cfca5e6fb86657e340348f2fd5da03b8eee524e99bae7e46adb86677a5.jpg)

![2187082c4bcb6434034f34c02a0e515b1b214639e0bc7ef7eedac73ea6408824.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/2187082c4bcb6434034f34c02a0e515b1b214639e0bc7ef7eedac73ea6408824.jpg)

![266da9c013dddfb5761ab93ef7cefe86a7e9bfa60566a8c58bfa66ce948e8142.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/266da9c013dddfb5761ab93ef7cefe86a7e9bfa60566a8c58bfa66ce948e8142.jpg)

![3cdec41037ed47932331619f185d2360e4f666fc657c4e59b5a13553144ab1f3.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/3cdec41037ed47932331619f185d2360e4f666fc657c4e59b5a13553144ab1f3.jpg)

![631b13c02cd867c5a6255d2a5c0feafc22b22461487c3bde1ff5ba6de7b9750b.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/631b13c02cd867c5a6255d2a5c0feafc22b22461487c3bde1ff5ba6de7b9750b.jpg)

![bf8f2c0bceff9b84ba8da135121cf42ad865d6ee01eb5ad1c52eb4135e4e9fd1.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/bf8f2c0bceff9b84ba8da135121cf42ad865d6ee01eb5ad1c52eb4135e4e9fd1.jpg)

![e033f295f4a2f88a20a09aeac7ee55be5af25138277476dcfa99ed9c1d382dd8.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/e033f295f4a2f88a20a09aeac7ee55be5af25138277476dcfa99ed9c1d382dd8.jpg)

![f9014777d2b861b7b42b0f5a0560335410027ff0f3c1379a95ee859634592fe2.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/f9014777d2b861b7b42b0f5a0560335410027ff0f3c1379a95ee859634592fe2.jpg)

![fda2179aabce8ccface65b23a3fb91f853e305b152bf339ccd7960a713305f2e.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/fda2179aabce8ccface65b23a3fb91f853e305b152bf339ccd7960a713305f2e.jpg)

## MARS-SQL: A Multi-Agent Reinforcement Learning Framework For Text-To-SQL


### Images

![148858c5f2655b69bb49d8095a8e1aeeec8665ea8f19edc725a63a174abbc4f0.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/148858c5f2655b69bb49d8095a8e1aeeec8665ea8f19edc725a63a174abbc4f0.jpg)

![198ba12c78c6c69240b622cbbf5d12ed1dbc6e610c5609cbe3f1ff98ef21192f.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/198ba12c78c6c69240b622cbbf5d12ed1dbc6e610c5609cbe3f1ff98ef21192f.jpg)

![5a38d347b746e26ccc2b214a2392dd6e4b46f645b07b7a5b67d3cdb2b92beba9.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/5a38d347b746e26ccc2b214a2392dd6e4b46f645b07b7a5b67d3cdb2b92beba9.jpg)

![72c5cc4a4749f1386892d42ab6edeb64eff21679957e39d1d10889cafe44c2ee.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/72c5cc4a4749f1386892d42ab6edeb64eff21679957e39d1d10889cafe44c2ee.jpg)

![7e8f9afa0cda3cf9ec617164139fc255118e480ad0f742bceade232b709a6bfe.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/7e8f9afa0cda3cf9ec617164139fc255118e480ad0f742bceade232b709a6bfe.jpg)

![ae52544a31d9f67078a5e2acd22e4da0d3393007e419fc37f4f2ee52c2d0a974.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/ae52544a31d9f67078a5e2acd22e4da0d3393007e419fc37f4f2ee52c2d0a974.jpg)

![e32d1e77b69357adec91e49595d303cb238efa0664431947b357a2ceda20899c.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/images/e32d1e77b69357adec91e49595d303cb238efa0664431947b357a2ceda20899c.jpg)

### Tables

![065fa30a2cb14cf6268c3c7a8ffc4c5d7f95c8f7bdaec9ffcbbbb607880a0749.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/065fa30a2cb14cf6268c3c7a8ffc4c5d7f95c8f7bdaec9ffcbbbb607880a0749.jpg)

![105b65ad4718c1bce7484dbe844a92f02a5855c1a836928d7b881c11a0e5eed1.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/105b65ad4718c1bce7484dbe844a92f02a5855c1a836928d7b881c11a0e5eed1.jpg)

![1074f8c964d5d1cc823ed19b4c3425d9c51e8bc9339d0ae366db9810afcf5634.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/1074f8c964d5d1cc823ed19b4c3425d9c51e8bc9339d0ae366db9810afcf5634.jpg)

![16d6f5c9025b324c1b0207e74f8e546fda3fd83ef5b0b44cf865ca254b0de213.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/16d6f5c9025b324c1b0207e74f8e546fda3fd83ef5b0b44cf865ca254b0de213.jpg)

![23517b874bcd7b66064ff5785bea0da0121f8aa6ef1ee0b96b1ad047d69d0871.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/23517b874bcd7b66064ff5785bea0da0121f8aa6ef1ee0b96b1ad047d69d0871.jpg)

![238a095fa64c07adce2aa83669e81854e69ea76b810fcccaf8cac04f8d7d471b.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/238a095fa64c07adce2aa83669e81854e69ea76b810fcccaf8cac04f8d7d471b.jpg)

![287a333397ba24b7c3bdd44026f699d384f2db0e6999b284f5332d86795ed43e.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/287a333397ba24b7c3bdd44026f699d384f2db0e6999b284f5332d86795ed43e.jpg)

![2bee760dd1c682cdef89d72428a8dac7a47a1670d4b06b1d6e7726fcfaafed5e.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/2bee760dd1c682cdef89d72428a8dac7a47a1670d4b06b1d6e7726fcfaafed5e.jpg)

![34cdeb9990c10af5774ea9d98130cace617a8d4f873e91bd4573125d0a10b66c.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/34cdeb9990c10af5774ea9d98130cace617a8d4f873e91bd4573125d0a10b66c.jpg)

![370a4b410bd3610d83644770046fa412baa63161b3113129c5e69548fb28f87c.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/370a4b410bd3610d83644770046fa412baa63161b3113129c5e69548fb28f87c.jpg)

![3c49fc7c64c528cd0930eab05072c4bfe98400b711e1051116133d72be918eb9.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/3c49fc7c64c528cd0930eab05072c4bfe98400b711e1051116133d72be918eb9.jpg)

![3f6b33d4b98f4bd91a3b1b2a36d89acd75f684ad44212a6d997849e7e3e80eec.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/3f6b33d4b98f4bd91a3b1b2a36d89acd75f684ad44212a6d997849e7e3e80eec.jpg)

![4128b9e8e7e1cf585114b7239fe996560a352c8bdc91a8b72644c60cf2bf1f6a.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/4128b9e8e7e1cf585114b7239fe996560a352c8bdc91a8b72644c60cf2bf1f6a.jpg)

![59766bbda2b9cbeb0ea1ffe56bcc7fcf2243bf6578a2c86edefbc6a634a4ac66.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/59766bbda2b9cbeb0ea1ffe56bcc7fcf2243bf6578a2c86edefbc6a634a4ac66.jpg)

![5d29dda612b9435da5337d5ba0c639f2bcfb62a52cd1e350eadca7b7d594faa6.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/5d29dda612b9435da5337d5ba0c639f2bcfb62a52cd1e350eadca7b7d594faa6.jpg)

![6422aa92046fcade110c6c9c2d1597aaeeb90fc1e94384847922dc889a7860be.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/6422aa92046fcade110c6c9c2d1597aaeeb90fc1e94384847922dc889a7860be.jpg)

![65ed37cf5d7676fcabc132b3cd7891c28641a7ff953a5b30db7e37e7c81656a8.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/65ed37cf5d7676fcabc132b3cd7891c28641a7ff953a5b30db7e37e7c81656a8.jpg)

![795e196e29d05e0f72175bf3941bbcd5251e9a89b1465a5d146c1442631468ef.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/795e196e29d05e0f72175bf3941bbcd5251e9a89b1465a5d146c1442631468ef.jpg)

![7f96fd261c1a86f45be9bb83d2aee5879dc401ec5aac45218da567b7888fbc12.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/7f96fd261c1a86f45be9bb83d2aee5879dc401ec5aac45218da567b7888fbc12.jpg)

![8f626238408ac04fb00ed11494eb1cfc6b01dfb95ad826224a085fafaf2b6688.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/8f626238408ac04fb00ed11494eb1cfc6b01dfb95ad826224a085fafaf2b6688.jpg)

![904e090108d2ac8e3f3400c7db4cac7cd3130d8e9ba3775944543c0392a9a5dc.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/904e090108d2ac8e3f3400c7db4cac7cd3130d8e9ba3775944543c0392a9a5dc.jpg)

![95610a9aa3772987bc6d66c0b6ee0c1a2a85f752af53663a33e558725209fc7a.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/95610a9aa3772987bc6d66c0b6ee0c1a2a85f752af53663a33e558725209fc7a.jpg)

![a00b73c2f508af60dad574c6f754e7f593ca57c4430b80dd73756a5effe07ac8.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/a00b73c2f508af60dad574c6f754e7f593ca57c4430b80dd73756a5effe07ac8.jpg)

![a22976faff0893d1e2ac7842aa90d88ae981a23bb0c678009425cf62fa6f4498.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/a22976faff0893d1e2ac7842aa90d88ae981a23bb0c678009425cf62fa6f4498.jpg)

![aacec86176ee5e2010a9f2740a9b4dd15827db75b8f50d107f0f392c15030e48.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/aacec86176ee5e2010a9f2740a9b4dd15827db75b8f50d107f0f392c15030e48.jpg)

![adbad3d833814a3efbe5bab3658d1930e3b9cde307c3b4294e5a0ea0749d6341.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/adbad3d833814a3efbe5bab3658d1930e3b9cde307c3b4294e5a0ea0749d6341.jpg)

![beaef39618c6342db37cc82a524a44d29f333fd9f358294f2328f85836b4f097.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/beaef39618c6342db37cc82a524a44d29f333fd9f358294f2328f85836b4f097.jpg)

![bff4c81aea3ab0be62e5b9e4f614e706fd814085c37ab41d4eac93dbf9a787b4.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/bff4c81aea3ab0be62e5b9e4f614e706fd814085c37ab41d4eac93dbf9a787b4.jpg)

![c9d7b15fa3d640020fdf536eedddf6786a617ef881ff8acf35ad7b16546f036f.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/c9d7b15fa3d640020fdf536eedddf6786a617ef881ff8acf35ad7b16546f036f.jpg)

![d1707c8c8900f157155d397145472bfd9b59fadeda0315c81190105b22443894.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/d1707c8c8900f157155d397145472bfd9b59fadeda0315c81190105b22443894.jpg)

![d2a1cf16136a6a6b0b3211e1c6daafc2f22f471ca3fed4eb81c9b8e386337f82.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/d2a1cf16136a6a6b0b3211e1c6daafc2f22f471ca3fed4eb81c9b8e386337f82.jpg)

![eb63adc2bdb43e1189cd6acbde888d1f61dc6dee3aced248cb5e50d4a3cdd853.jpg](../icml_results/5527_MARS-SQL_%20A%20Multi-Agent%20Reinforcement%20Learning%20Framework%20For%20Text-To-SQL/tables/eb63adc2bdb43e1189cd6acbde888d1f61dc6dee3aced248cb5e50d4a3cdd853.jpg)

<a id="paper-5480"></a>

## SSDCN: Spatial-Spectral Dual-Clustering-based Network for Hyperspectral Image Super-resolution


### Images

![0e5231bc70ca674bd1ec5835f91f85164c1059505e58987c927b8d6476aad988.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/0e5231bc70ca674bd1ec5835f91f85164c1059505e58987c927b8d6476aad988.jpg)

![1b08159aee4e24adc56088f9b04e67a03f6f898b5bc28c064b05bbdbbab75f85.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/1b08159aee4e24adc56088f9b04e67a03f6f898b5bc28c064b05bbdbbab75f85.jpg)

![1bf597429455cde0529b499e90229247c7e207ab4ae14b4b1e75247d45828094.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/1bf597429455cde0529b499e90229247c7e207ab4ae14b4b1e75247d45828094.jpg)

![20493756cfce73f0b79f16966f43e413c0e6c7272cda3994a9411263994f221f.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/20493756cfce73f0b79f16966f43e413c0e6c7272cda3994a9411263994f221f.jpg)

![2880c5ce5bb690c8e9db225869c0af5008d43d8fa56d12c72d2fcacf64bb9cd6.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/2880c5ce5bb690c8e9db225869c0af5008d43d8fa56d12c72d2fcacf64bb9cd6.jpg)

![3807aca3b6762ba0638b4009e0d4758075b8f8813d29e13481e2a7992ce8553b.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/3807aca3b6762ba0638b4009e0d4758075b8f8813d29e13481e2a7992ce8553b.jpg)

![3864d5b45b3905890e81ea59ee1d79898ec0c26f6ea5a528c2f9c18d6fb55f06.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/3864d5b45b3905890e81ea59ee1d79898ec0c26f6ea5a528c2f9c18d6fb55f06.jpg)

![65c0c813f113cda697065f5dd9d3fa714bfe3642ee9f477d58666dca6f039ab0.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/65c0c813f113cda697065f5dd9d3fa714bfe3642ee9f477d58666dca6f039ab0.jpg)

![804dc8f4712132394ee696c9d6b2eb408603c1bf494d8bf180165a11c9307012.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/804dc8f4712132394ee696c9d6b2eb408603c1bf494d8bf180165a11c9307012.jpg)

![a2b89ac907c10a1337ce3ce05b18370f2a9fb778a8134e158b9f9a2a57587057.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/a2b89ac907c10a1337ce3ce05b18370f2a9fb778a8134e158b9f9a2a57587057.jpg)

![b27e4bb52ad324ad873b7dda795744cbe5cbce59739f98592a12f34d8d58b2ac.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/b27e4bb52ad324ad873b7dda795744cbe5cbce59739f98592a12f34d8d58b2ac.jpg)

![e7cba9934277853e6bd0fa2dfea4515fa40e9a08917f91039f20e4e1156c3a02.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/e7cba9934277853e6bd0fa2dfea4515fa40e9a08917f91039f20e4e1156c3a02.jpg)

![f7c103ab2d8e78918f1cec8e8a13d66d9fc81c11c3b3a33adce588fcd8210b62.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/f7c103ab2d8e78918f1cec8e8a13d66d9fc81c11c3b3a33adce588fcd8210b62.jpg)

![f92f79e39e1c0376a5f56c9109a9932643b54d98c9109f73b00590a7a2449fe9.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/images/f92f79e39e1c0376a5f56c9109a9932643b54d98c9109f73b00590a7a2449fe9.jpg)

### Tables

![096c3566a164ac73052fcfccd8867d553b7a5d7ceffb138f1c97d33c5898261d.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/096c3566a164ac73052fcfccd8867d553b7a5d7ceffb138f1c97d33c5898261d.jpg)

![0bc413ee102a461b9a0d86d89508179c8fa19c1125c6238fdb426a75a0f698b2.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/0bc413ee102a461b9a0d86d89508179c8fa19c1125c6238fdb426a75a0f698b2.jpg)

![1fb9f4c6c13297a6383e66b3978ab26f04ae184f892a1f924807136f045cd500.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/1fb9f4c6c13297a6383e66b3978ab26f04ae184f892a1f924807136f045cd500.jpg)

![3cc4268fb476c9b6c564a6a1cbc5dce8d4f31edfc1c3f9c5c3fe8ee8dea0d779.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/3cc4268fb476c9b6c564a6a1cbc5dce8d4f31edfc1c3f9c5c3fe8ee8dea0d779.jpg)

![3f5301c82aeef4cac5c7509d006f7e58482d0c2a1ed3a0a0bdbcd2fb3c7505ea.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/3f5301c82aeef4cac5c7509d006f7e58482d0c2a1ed3a0a0bdbcd2fb3c7505ea.jpg)

![4f7f06acd1a5600263ceb71d5879dda67824a8b6f708de1de39b4e90f8177945.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/4f7f06acd1a5600263ceb71d5879dda67824a8b6f708de1de39b4e90f8177945.jpg)

![590bf6286528b8f9bf3470a64b150d8ea8d609f7de0ba1b35fe17820310cb21f.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/590bf6286528b8f9bf3470a64b150d8ea8d609f7de0ba1b35fe17820310cb21f.jpg)

![961e10f8f55d27704068be2c41cb8c8283c2b703470076df385437e390baee56.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/961e10f8f55d27704068be2c41cb8c8283c2b703470076df385437e390baee56.jpg)

![c277dc40c0cbdf9dc3ee09987e1e64e3bdaa3d50bd0198fdf830838466da9d18.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/c277dc40c0cbdf9dc3ee09987e1e64e3bdaa3d50bd0198fdf830838466da9d18.jpg)

![c838942a54fb39dd4cb058ba7f1935e57cb9f14a3d10d5b5a297c687d29eba71.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/c838942a54fb39dd4cb058ba7f1935e57cb9f14a3d10d5b5a297c687d29eba71.jpg)

![c8442f457e0f854fb09ce4edfc7cf18ce7b05f750af3aa4dea3bb4e139574f65.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/c8442f457e0f854fb09ce4edfc7cf18ce7b05f750af3aa4dea3bb4e139574f65.jpg)

![dbecdc71a689a3f0c75147d185f81c9fc527169c929a6b9cca5ac9a4c7d4f324.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/dbecdc71a689a3f0c75147d185f81c9fc527169c929a6b9cca5ac9a4c7d4f324.jpg)

![e071c6440421782ef09e49e9de521b6e2df2fa0cc3a23e9716f33b44a9f1fc61.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/e071c6440421782ef09e49e9de521b6e2df2fa0cc3a23e9716f33b44a9f1fc61.jpg)

![e2c09d1fca9cdd1184d45b34f95d923a4be8314c21bd0c24e102b6302253585d.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/e2c09d1fca9cdd1184d45b34f95d923a4be8314c21bd0c24e102b6302253585d.jpg)

![ee4a7c8823df1ada2f0beb97da5f65aae937b0a0caecce2a07f61dc6dbe6f3e8.jpg](../icml_results/5528_SSDCN_%20Spatial-Spectral%20Dual-Clustering-based%20Network%20for%20Hyperspectral%20Image%20Super-resolution/tables/ee4a7c8823df1ada2f0beb97da5f65aae937b0a0caecce2a07f61dc6dbe6f3e8.jpg)

<a id="paper-5481"></a>

## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention


### Images

![13c4a46d3f81e3c1dbf212e258ce0a6d2c4eadb4ddedff641d2f7c58c29e43fb.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/13c4a46d3f81e3c1dbf212e258ce0a6d2c4eadb4ddedff641d2f7c58c29e43fb.jpg)

![3967afba13212cba91e39f6aac3feb7821adeb4575df25df8fec982a260fac11.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/3967afba13212cba91e39f6aac3feb7821adeb4575df25df8fec982a260fac11.jpg)

![48fbe2969b5a56a71a1e15d7646f42f2f5f319cac82ebf0e43be8ea2562d3e81.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/48fbe2969b5a56a71a1e15d7646f42f2f5f319cac82ebf0e43be8ea2562d3e81.jpg)

![72b9405d3594603d39d6168391ea521365eece1be269ca96fb7b1530c0470f21.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/72b9405d3594603d39d6168391ea521365eece1be269ca96fb7b1530c0470f21.jpg)

![85399ab2bd196b906b42d21f782ea4eeb63ad0a9b87358bc60bc2017d765c400.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/85399ab2bd196b906b42d21f782ea4eeb63ad0a9b87358bc60bc2017d765c400.jpg)

![8777a8e8a3b365f5ddfedab407cb2b6eebf328a8dd5846559e3e4575395e2405.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/8777a8e8a3b365f5ddfedab407cb2b6eebf328a8dd5846559e3e4575395e2405.jpg)

![93caedd6efcdbf1b971cfeb0498cb1d858bec34a5c00b760ee73fc1c05455faf.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/93caedd6efcdbf1b971cfeb0498cb1d858bec34a5c00b760ee73fc1c05455faf.jpg)

![a27a42c8e2fbe6a7d6fdb279e0875e43be90a7137c9b08a52f4d8524004b3334.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/a27a42c8e2fbe6a7d6fdb279e0875e43be90a7137c9b08a52f4d8524004b3334.jpg)

![c377603b918db94d78ab5246cdd69b8fc16a85bcd923be92148d222d984e5ded.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/c377603b918db94d78ab5246cdd69b8fc16a85bcd923be92148d222d984e5ded.jpg)

![d17f2e5ad5fb783ca783dfcfdbc99fc576dca0c0d1f46f7e23d76e50c1e97d24.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/d17f2e5ad5fb783ca783dfcfdbc99fc576dca0c0d1f46f7e23d76e50c1e97d24.jpg)

![debc29355bbce4ec4df0cb231ea222227b0f756155a875a35645a853a800737c.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/debc29355bbce4ec4df0cb231ea222227b0f756155a875a35645a853a800737c.jpg)

![e73a88cfb6c24eeeaff7c7bfdce4c4047d2b8d70576322e7fb2f2c30b2da1f35.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/images/e73a88cfb6c24eeeaff7c7bfdce4c4047d2b8d70576322e7fb2f2c30b2da1f35.jpg)

### Tables

![2beaad9c478c0dac566def0faee2e5c6e29a438f5812b0fbf36c09d1fe1da49e.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/2beaad9c478c0dac566def0faee2e5c6e29a438f5812b0fbf36c09d1fe1da49e.jpg)

![341ee0ad770107b6ce6cedbcf4a486796ab0241230531d7e02906be37dd964ce.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/341ee0ad770107b6ce6cedbcf4a486796ab0241230531d7e02906be37dd964ce.jpg)

![5c79cf7908a391395b4e314a12a7d8080eb3ed72ca52646ee0731ffd9c4d6c26.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/5c79cf7908a391395b4e314a12a7d8080eb3ed72ca52646ee0731ffd9c4d6c26.jpg)

![6cdeaac6cb70b6077f0649c3738556a6c159d405fd17186b4d6efd05d081843c.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/6cdeaac6cb70b6077f0649c3738556a6c159d405fd17186b4d6efd05d081843c.jpg)

![773b93e576d796e98fad965ac85a56e5bbf2835ea2dd0f1c38f1e2ce0a5f3807.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/773b93e576d796e98fad965ac85a56e5bbf2835ea2dd0f1c38f1e2ce0a5f3807.jpg)

![86bdf70b1a77cc5f9c9c4aaf433fe2323a3d4775a6e60810d6b77ebfb20b94af.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/86bdf70b1a77cc5f9c9c4aaf433fe2323a3d4775a6e60810d6b77ebfb20b94af.jpg)

![89bfc03f1fb656fbbebaf8bd8442e0f163f89831ca594043b6a112ac9c714867.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/89bfc03f1fb656fbbebaf8bd8442e0f163f89831ca594043b6a112ac9c714867.jpg)

![8d51550f1f29c729a8d27fc10c50f0831abc047f4069ecaff7172e92c5445986.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/8d51550f1f29c729a8d27fc10c50f0831abc047f4069ecaff7172e92c5445986.jpg)

![8e15b6af2559eb334dff6cce953c80c3d90dd7a55c541e0d25b566a6100e7bca.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/8e15b6af2559eb334dff6cce953c80c3d90dd7a55c541e0d25b566a6100e7bca.jpg)

![a17649db2ef5441ee1d8fcd4f87233621c391c330ec2523f047e91391a350a80.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/a17649db2ef5441ee1d8fcd4f87233621c391c330ec2523f047e91391a350a80.jpg)

![c6383325874947753ba9a0c4c2b2aed4fd5dc958374e14a9b647416a935b3f4a.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/c6383325874947753ba9a0c4c2b2aed4fd5dc958374e14a9b647416a935b3f4a.jpg)

![ee74042f88600089506460850d91ebbe79bc615a54a49a41808520817aea9a9a.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/ee74042f88600089506460850d91ebbe79bc615a54a49a41808520817aea9a9a.jpg)

![ef5e453e3f68c057670364b5c548d200ae834b0fc2bcac2f4cfb19eb2c2b7c45.jpg](../icml_results/5529_Less%20Is%20More_%20Fast%20and%20Accurate%20Reasoning%20with%20Cross-Head%20Unified%20Sparse%20Attention/tables/ef5e453e3f68c057670364b5c548d200ae834b0fc2bcac2f4cfb19eb2c2b7c45.jpg)

<a id="paper-5482"></a>

## Zero-source LLM Hallucination Detection with Human-like Criteria Probing


### Images

![08b45dfef09414420eaa4578dd89300a1eb0c2b620ce8eb64b85346616bee419.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/images/08b45dfef09414420eaa4578dd89300a1eb0c2b620ce8eb64b85346616bee419.jpg)

![1be3a9c88a980cb4779429acfc00aec4c1ea3c520b710f0a563c1b71cadad65b.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/images/1be3a9c88a980cb4779429acfc00aec4c1ea3c520b710f0a563c1b71cadad65b.jpg)

![2cba21d21fb85139deb80bc085063e33085687d311a2dfc8b1e9f09f0971a110.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/images/2cba21d21fb85139deb80bc085063e33085687d311a2dfc8b1e9f09f0971a110.jpg)

![37b6da2abf706df90f0bccb1357325e956a5f105ba968f6880a3087153527ac2.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/images/37b6da2abf706df90f0bccb1357325e956a5f105ba968f6880a3087153527ac2.jpg)

![949abc2bfaca862f3a04397e8806d4cd362abc7b0a0a5c306f67ce5f998333d6.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/images/949abc2bfaca862f3a04397e8806d4cd362abc7b0a0a5c306f67ce5f998333d6.jpg)

### Tables

![0b08f3a6515f5111517c46ec7a7f4f7cab1f51bcd70d7aa6b85b67ee76efebe8.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/0b08f3a6515f5111517c46ec7a7f4f7cab1f51bcd70d7aa6b85b67ee76efebe8.jpg)

![0bc8c1669a567c5316ca1b70096be69a781d964bc4b0db221f5fdf3b3b959d04.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/0bc8c1669a567c5316ca1b70096be69a781d964bc4b0db221f5fdf3b3b959d04.jpg)

![10d31b9a5b435d9d5cff718e5af9364db37037c7ecad0fd3daa2b037ae2b0576.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/10d31b9a5b435d9d5cff718e5af9364db37037c7ecad0fd3daa2b037ae2b0576.jpg)

![13e54a3bafe434077060ef4c821ed482150ece296c2c74b301fda8a6782839ce.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/13e54a3bafe434077060ef4c821ed482150ece296c2c74b301fda8a6782839ce.jpg)

![18163f5e89d9eaef0ec9ab2f278620fb280860280ad2ef7078d4a384525ac8db.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/18163f5e89d9eaef0ec9ab2f278620fb280860280ad2ef7078d4a384525ac8db.jpg)

![1a76bb284c0351591960c231c8f808fcdbddf5e7e38edb2879a5e78a1c469f39.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/1a76bb284c0351591960c231c8f808fcdbddf5e7e38edb2879a5e78a1c469f39.jpg)

![2151b97c4af4951c0321a8878599a26ccaf6a23d3dacadbe1ab08ccc4016a9f7.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/2151b97c4af4951c0321a8878599a26ccaf6a23d3dacadbe1ab08ccc4016a9f7.jpg)

![2ddf9f5be0785c272b878e7713dd11df23da3a2c702be0fb7352e3bd73f42dd6.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/2ddf9f5be0785c272b878e7713dd11df23da3a2c702be0fb7352e3bd73f42dd6.jpg)

![335ac1bbc65c898ad39141c3aee845f4a1781af171389d548328f14d36d857f6.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/335ac1bbc65c898ad39141c3aee845f4a1781af171389d548328f14d36d857f6.jpg)

![37c5082d66e37afcd4796b87b9c1d55464b5ef294fc0c14d9e333a62eef188d8.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/37c5082d66e37afcd4796b87b9c1d55464b5ef294fc0c14d9e333a62eef188d8.jpg)

![3d984fdc464aaf31d988eaac3e6847b11ecba1522d6eb6c2264b68fa075fab48.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/3d984fdc464aaf31d988eaac3e6847b11ecba1522d6eb6c2264b68fa075fab48.jpg)

![44c1eb1d841a039c54aeb393a1541f7d922cc9263a91c4f82c449f99ce4cc179.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/44c1eb1d841a039c54aeb393a1541f7d922cc9263a91c4f82c449f99ce4cc179.jpg)

![5448489cebc620fa16b3d821f598a2df2853bf63c0524f9ed37cda1e93bcc4dd.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/5448489cebc620fa16b3d821f598a2df2853bf63c0524f9ed37cda1e93bcc4dd.jpg)

![63f13f1fe234c331b36fbd3fb45d81fd5475a7c026c7baabf15be5329aa89114.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/63f13f1fe234c331b36fbd3fb45d81fd5475a7c026c7baabf15be5329aa89114.jpg)

![66487ef3beef01a864ee9993a03fe07ebfe53b447912fa058baf6691c0014b1a.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/66487ef3beef01a864ee9993a03fe07ebfe53b447912fa058baf6691c0014b1a.jpg)

![67dc8ed783b4bd5e84d311a70c776745243b184d6ce269ee23efeb151deb2f72.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/67dc8ed783b4bd5e84d311a70c776745243b184d6ce269ee23efeb151deb2f72.jpg)

![6e192292edf1bd43521563ea1f4cc7f2cbbe7ddf0d5a8ab94d7c9d0c4ce8956e.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/6e192292edf1bd43521563ea1f4cc7f2cbbe7ddf0d5a8ab94d7c9d0c4ce8956e.jpg)

![6f928c77cb1dd3d8b693b65e986ea4f43dcf148417c2bae916adbb40e073a495.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/6f928c77cb1dd3d8b693b65e986ea4f43dcf148417c2bae916adbb40e073a495.jpg)

![744bf8f235461d93be174303c0181222fe3e8d83541ec8c486b51281d8ec4621.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/744bf8f235461d93be174303c0181222fe3e8d83541ec8c486b51281d8ec4621.jpg)

![76aa51cfd8b874506fd17cfecdb91da7a1169682bf74a42b8cb9f4c170b67a12.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/76aa51cfd8b874506fd17cfecdb91da7a1169682bf74a42b8cb9f4c170b67a12.jpg)

![7fe6fb65eef58742d3e537ede21db331cb79a09757e9ea2ecfd983feca5c029c.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/7fe6fb65eef58742d3e537ede21db331cb79a09757e9ea2ecfd983feca5c029c.jpg)

![8a22359a9cc371a3452a49825e374245b110c3d00155da03e2da9c8abf6643ba.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/8a22359a9cc371a3452a49825e374245b110c3d00155da03e2da9c8abf6643ba.jpg)

![8e39339a2ae581441b81a99269967905cc3c5079f10002fe63efd543a43bfbe8.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/8e39339a2ae581441b81a99269967905cc3c5079f10002fe63efd543a43bfbe8.jpg)

![9720616fba4428302fb190520fa1bb438edb8b4a792eff7caa9c34e141b08de2.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/9720616fba4428302fb190520fa1bb438edb8b4a792eff7caa9c34e141b08de2.jpg)

![a4625d19ed79a71451f4af7c1862f2e3c0841a58c3ba8d4e3f91343e8bc4c3f1.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/a4625d19ed79a71451f4af7c1862f2e3c0841a58c3ba8d4e3f91343e8bc4c3f1.jpg)

![ce4ffd9e0fd886342c910f60b814e26c429e72d788f648a03e09b845100ac25b.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/ce4ffd9e0fd886342c910f60b814e26c429e72d788f648a03e09b845100ac25b.jpg)

![d437105bd376641401023629dfbf900cb202995927d7c3c3b645732a0befdd58.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/d437105bd376641401023629dfbf900cb202995927d7c3c3b645732a0befdd58.jpg)

![d6f02c4791f0e2f4762a4b215c054bca438365e7c1ce3dfd8649ba762b1805e4.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/d6f02c4791f0e2f4762a4b215c054bca438365e7c1ce3dfd8649ba762b1805e4.jpg)

![da003693aa7af70ecce118ed15a7a29e1ab7e2b22f631f2f8fe5accce47dab81.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/da003693aa7af70ecce118ed15a7a29e1ab7e2b22f631f2f8fe5accce47dab81.jpg)

![e45cfa7377ab3389bf2112598cc489d657598efef7589600d8b8e91e065185e6.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/e45cfa7377ab3389bf2112598cc489d657598efef7589600d8b8e91e065185e6.jpg)

![e54592cf26f900745c2ffb0f2fd451fa7a1532439f79bdca953de817f0bfe051.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/e54592cf26f900745c2ffb0f2fd451fa7a1532439f79bdca953de817f0bfe051.jpg)

![f6068ab14bab05259c8a8d77bdac108edb69e213f6d33a9a9cbc699114cf469a.jpg](../icml_results/5530_Zero-source%20LLM%20Hallucination%20Detection%20with%20Human-like%20Criteria%20Probing/tables/f6068ab14bab05259c8a8d77bdac108edb69e213f6d33a9a9cbc699114cf469a.jpg)

<a id="paper-5483"></a>

## TMD-Bench: A Multi-Level Evaluation Paradigm for Music–Dance Co-Generation


### Images

![2544850706ea9a05b944116382e26b03c73b1e1d37f7375f6a6633c296c24ffe.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/2544850706ea9a05b944116382e26b03c73b1e1d37f7375f6a6633c296c24ffe.jpg)

![290ee73659d9e80407a11f29c50d0cb9490bcf77949dd072c471f3c639ae642d.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/290ee73659d9e80407a11f29c50d0cb9490bcf77949dd072c471f3c639ae642d.jpg)

![2a425c324de18c839d8bfb1fb3295ab41771b91d51b41ced517caf50d9a9070d.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/2a425c324de18c839d8bfb1fb3295ab41771b91d51b41ced517caf50d9a9070d.jpg)

![3fb6cbca267b9ede44e49c74ee76a43f28d047ec99955a947706e953067f22e7.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/3fb6cbca267b9ede44e49c74ee76a43f28d047ec99955a947706e953067f22e7.jpg)

![77996dd6c27bdc639f5c456cf422ec7a755bd9c4cd463a6b24cb541c6233e0f8.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/77996dd6c27bdc639f5c456cf422ec7a755bd9c4cd463a6b24cb541c6233e0f8.jpg)

![8cd4d3da26321e4c9f3bc03febd208506bce1bf096677d7d262e0864d763f3c2.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/8cd4d3da26321e4c9f3bc03febd208506bce1bf096677d7d262e0864d763f3c2.jpg)

![a1c77155ed8dfebf4b0e9177f1fdd4595b0af354eea9cfa7fc10647c534cca16.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/a1c77155ed8dfebf4b0e9177f1fdd4595b0af354eea9cfa7fc10647c534cca16.jpg)

![cfb11d7009f8de49f17ae15803661bedcf3bc4cab711278ea71015cceca6d82f.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/cfb11d7009f8de49f17ae15803661bedcf3bc4cab711278ea71015cceca6d82f.jpg)

![d9c3dc0b57f54a697773d0097eb89f877b3882feab55d88a18ea539a58c421d5.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/images/d9c3dc0b57f54a697773d0097eb89f877b3882feab55d88a18ea539a58c421d5.jpg)

### Tables

![1c3204a32f0c0f89fac68bd071f5e120225a76a1c8eddd37105329be25788308.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/1c3204a32f0c0f89fac68bd071f5e120225a76a1c8eddd37105329be25788308.jpg)

![3a82f8d2672a167faf6739a450300c592b4027e7472aa5bb28766650815a562c.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/3a82f8d2672a167faf6739a450300c592b4027e7472aa5bb28766650815a562c.jpg)

![3a8fb2b8fca465cd13bceb6c97e1e92f61adf427ad87fd4d00e8b04d20a91b98.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/3a8fb2b8fca465cd13bceb6c97e1e92f61adf427ad87fd4d00e8b04d20a91b98.jpg)

![578615018ae025d30947df1aa2854a26fbab1073c8bd03d035dfa2e2c74fe0e6.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/578615018ae025d30947df1aa2854a26fbab1073c8bd03d035dfa2e2c74fe0e6.jpg)

![66b5c84c199e93112363916c2942ce29e4b81ac66bf36b6e3e836d356d688526.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/66b5c84c199e93112363916c2942ce29e4b81ac66bf36b6e3e836d356d688526.jpg)

![a928d5b14468390316af36f4b35816e3d96912e6dbd854206138503f29768ed6.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/a928d5b14468390316af36f4b35816e3d96912e6dbd854206138503f29768ed6.jpg)

![cbf43dd15927421d519721b1ff916b1136f113b45ad50e3f5f20c8c2130b9a82.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/cbf43dd15927421d519721b1ff916b1136f113b45ad50e3f5f20c8c2130b9a82.jpg)

![f99bffa160210f50db2cc01799e20b7af2bb5125b8e17cd457bf085298484979.jpg](../icml_results/5531_TMD-Bench_%20A%20Multi-Level%20Evaluation%20Paradigm%20for%20Music%E2%80%93Dance%20Co-Generation/tables/f99bffa160210f50db2cc01799e20b7af2bb5125b8e17cd457bf085298484979.jpg)

<a id="paper-5484"></a>

## RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization


### Images

![21283054d55a47e83ac9845b298d98385a5a3585d1749d5583eaf9f857b27b52.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/21283054d55a47e83ac9845b298d98385a5a3585d1749d5583eaf9f857b27b52.jpg)

![40ecd8c7d5aa4f92bc46f10ae079a908eaaaa8776aaf4e36968fbd0d7eb52464.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/40ecd8c7d5aa4f92bc46f10ae079a908eaaaa8776aaf4e36968fbd0d7eb52464.jpg)

![4cdbf006f2ba57f6c42a1c0287638725ffb0db8d988f245da2653839e03b95cf.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/4cdbf006f2ba57f6c42a1c0287638725ffb0db8d988f245da2653839e03b95cf.jpg)

![5af17accedcb3735f895b973155a68901ab6cf6e5e46df02f71a1a2f00e8e1ee.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/5af17accedcb3735f895b973155a68901ab6cf6e5e46df02f71a1a2f00e8e1ee.jpg)

![5db8694c82c419347252d04c9fd55ecaa51a9df5b9651cd17c8529d7ae71fe61.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/5db8694c82c419347252d04c9fd55ecaa51a9df5b9651cd17c8529d7ae71fe61.jpg)

![744b8cb0f76734036e70dfe5f2950261acfafeea95b859c2c684592527c26468.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/744b8cb0f76734036e70dfe5f2950261acfafeea95b859c2c684592527c26468.jpg)

![7f627a4d13ee28bfb7977ee35661decbecdebf77fd38039b18067eb9bce5ec8e.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/7f627a4d13ee28bfb7977ee35661decbecdebf77fd38039b18067eb9bce5ec8e.jpg)

![940101bd0014633e254a8d80d5821f5d4ed993b4234d4e61bf8a6fa836423a56.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/940101bd0014633e254a8d80d5821f5d4ed993b4234d4e61bf8a6fa836423a56.jpg)

![9a297449d506eb2fbf84080c079db6763dd3d9b9d2eb2c0f97278596aa908c72.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/9a297449d506eb2fbf84080c079db6763dd3d9b9d2eb2c0f97278596aa908c72.jpg)

![deecf91171f32e554e1a62e1cdb6b4b7355cd3abd45d9e95e67d80d0cb2de64c.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/deecf91171f32e554e1a62e1cdb6b4b7355cd3abd45d9e95e67d80d0cb2de64c.jpg)

![e5706914c7d8d0e007bc2bfd1d9352631dd9696939373e0eccd20a01f204ac60.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/e5706914c7d8d0e007bc2bfd1d9352631dd9696939373e0eccd20a01f204ac60.jpg)

![ef63f5a4fe5d87a66221e4a6230112b1188f792a776811e6d5b2f43de9b051f6.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/images/ef63f5a4fe5d87a66221e4a6230112b1188f792a776811e6d5b2f43de9b051f6.jpg)

### Tables

![18bce3d136fe5201b0b9b885f732a2dba55f651b26161c1a26f17b9ee91bca49.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/18bce3d136fe5201b0b9b885f732a2dba55f651b26161c1a26f17b9ee91bca49.jpg)

![2415659bb9029ebcd0961ee4f680bbc2b706699a9a89b9ca5b9e145f136415e0.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/2415659bb9029ebcd0961ee4f680bbc2b706699a9a89b9ca5b9e145f136415e0.jpg)

![4144d663a6081cce2784ad5c7fa79a6cfb8b3fcadbff1a5ffe739d28fc2cbdfb.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/4144d663a6081cce2784ad5c7fa79a6cfb8b3fcadbff1a5ffe739d28fc2cbdfb.jpg)

![834f88b8e1b3682d4e254b0bba184124d28f18460bb1440e9c1e1f715d9ee4ca.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/834f88b8e1b3682d4e254b0bba184124d28f18460bb1440e9c1e1f715d9ee4ca.jpg)

![9c3f1c66cdf303eb92d2859adcc28b349d4e770fbd8177ebf749cf123dbd1a4a.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/9c3f1c66cdf303eb92d2859adcc28b349d4e770fbd8177ebf749cf123dbd1a4a.jpg)

![9d9a01618b785ec6be9f39bc82dc6de1712fa136e748b5e5efa6eb117daa30f6.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/9d9a01618b785ec6be9f39bc82dc6de1712fa136e748b5e5efa6eb117daa30f6.jpg)

![b6746e876f7c4fc3a42e3baf0577ba161d99bf5eaff8a273a591bfebfb52c558.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/b6746e876f7c4fc3a42e3baf0577ba161d99bf5eaff8a273a591bfebfb52c558.jpg)

![bddd42bca89aec6c0655c83838ef194ff30d5b346536bd913a4c76f83c7cdd08.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/bddd42bca89aec6c0655c83838ef194ff30d5b346536bd913a4c76f83c7cdd08.jpg)

![d58cfbf2b66155bb8ec4c69ae4a5e002f4be522c204d0b79d9cd1c03786af89f.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/d58cfbf2b66155bb8ec4c69ae4a5e002f4be522c204d0b79d9cd1c03786af89f.jpg)

![e02e05b31dfcf043304e6460ab61b6ad739328e1a264ae5a2886efb10c70b8e2.jpg](../icml_results/5532_RobuQ_%20Pushing%20DiTs%20to%20W1.58A2%20via%20Robust%20Activation%20Quantization/tables/e02e05b31dfcf043304e6460ab61b6ad739328e1a264ae5a2886efb10c70b8e2.jpg)

<a id="paper-5485"></a>

## Return-to-Go Is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning


### Images

![10e49ed20cffed6dbdeb06a9705d11e642996cc72abd24faefecadf4785a7224.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/10e49ed20cffed6dbdeb06a9705d11e642996cc72abd24faefecadf4785a7224.jpg)

![65156d87c47850eaaac6d8ddd79e3c2cf5fa363613ce6211ab822e8252fa4eaa.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/65156d87c47850eaaac6d8ddd79e3c2cf5fa363613ce6211ab822e8252fa4eaa.jpg)

![6c665bedcdc9fd108b9deaf25b7325a77f91600e7e2276ff59b2366450fdff3b.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/6c665bedcdc9fd108b9deaf25b7325a77f91600e7e2276ff59b2366450fdff3b.jpg)

![708fd71f96bfb96f05606db097c07ce41a93e29a6e592b7b2810fff7a5a8013a.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/708fd71f96bfb96f05606db097c07ce41a93e29a6e592b7b2810fff7a5a8013a.jpg)

![7a5dee277f51c3f46cce6266650cedd57c9465a6ebe110741a28fa8f03ac9061.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/7a5dee277f51c3f46cce6266650cedd57c9465a6ebe110741a28fa8f03ac9061.jpg)

![a44eae90b952eac781a06c7a2dd31a0df682efb0a2f97cc709ad2698f3a7819a.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/a44eae90b952eac781a06c7a2dd31a0df682efb0a2f97cc709ad2698f3a7819a.jpg)

![db424195a09b4fb10b162404374d10ebbf7f3463fadc466ef3ef03d3e2f0fdbb.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/db424195a09b4fb10b162404374d10ebbf7f3463fadc466ef3ef03d3e2f0fdbb.jpg)

![eba234bd60f50125464ffe3033c9fc364547ec3c4131afc2a98fcce53a9d7ac4.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/eba234bd60f50125464ffe3033c9fc364547ec3c4131afc2a98fcce53a9d7ac4.jpg)

![f57b6cbe7be369087957ad18197329a079863ef049f179587b8b64139ad56b1e.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/images/f57b6cbe7be369087957ad18197329a079863ef049f179587b8b64139ad56b1e.jpg)

### Tables

![028217eca81ddbb6608a4ef4adbc6103e10a9bfaf13891b2e8c487b0b79a4a14.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/028217eca81ddbb6608a4ef4adbc6103e10a9bfaf13891b2e8c487b0b79a4a14.jpg)

![38f6b5e1c13ccc9449c145c2327f8351460cbdc0fd0f621703573625a082590b.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/38f6b5e1c13ccc9449c145c2327f8351460cbdc0fd0f621703573625a082590b.jpg)

![3963066d1193f599e0ba4bb18c74d392622dd063d26b5f2c7482b6dbff3baad8.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/3963066d1193f599e0ba4bb18c74d392622dd063d26b5f2c7482b6dbff3baad8.jpg)

![3d2cdb349d8f13913351f8a4886b86b2b1fe130361def9fc0201fcbb36d0d7d7.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/3d2cdb349d8f13913351f8a4886b86b2b1fe130361def9fc0201fcbb36d0d7d7.jpg)

![3f3060fe7087f5de05c808431527c4d30f31befa21693a0c68cb2619bed464b2.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/3f3060fe7087f5de05c808431527c4d30f31befa21693a0c68cb2619bed464b2.jpg)

![55ea8b588cf477002b7c727cb8a5678b838851238e23176c7a1a3da8bd1b7d8d.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/55ea8b588cf477002b7c727cb8a5678b838851238e23176c7a1a3da8bd1b7d8d.jpg)

![68b1515d257655827b6b0cd77ea23ac5632e58511f2cc293340ad5e7af0f1ec4.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/68b1515d257655827b6b0cd77ea23ac5632e58511f2cc293340ad5e7af0f1ec4.jpg)

![73ae0fee52e6fd1470b7d6debc843f6d7b0a612840f6649d16247bc34916f9c9.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/73ae0fee52e6fd1470b7d6debc843f6d7b0a612840f6649d16247bc34916f9c9.jpg)

![76243b2c1e85b09e2758879d4c981ca39476bbf2ea9786092ff4b2c9e9060848.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/76243b2c1e85b09e2758879d4c981ca39476bbf2ea9786092ff4b2c9e9060848.jpg)

![7ac92c8afae9b3264102a7a9a0df8b0e1c83f8e2230a790de596b7da2d63c47c.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/7ac92c8afae9b3264102a7a9a0df8b0e1c83f8e2230a790de596b7da2d63c47c.jpg)

![8925750277680603452fd3492769e38f807c287552dc999ee93397c867be3f46.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/8925750277680603452fd3492769e38f807c287552dc999ee93397c867be3f46.jpg)

![9d9715c8fe10140327565ea4ef2456fcc8f7f853b8ce294a8272ff414d56e3c7.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/9d9715c8fe10140327565ea4ef2456fcc8f7f853b8ce294a8272ff414d56e3c7.jpg)

![a97e40f292406ca34af910e11434691500a0f1e5f43531627e29a637c0edf57f.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/a97e40f292406ca34af910e11434691500a0f1e5f43531627e29a637c0edf57f.jpg)

![ae32f6033c55554fb41ea1c0367964d31a6e79c397dc54d143d701ed286b281a.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/ae32f6033c55554fb41ea1c0367964d31a6e79c397dc54d143d701ed286b281a.jpg)

![b3a2ae5c8bfdb0a283f7fdb36430dd20344b17d0db92cae6bc9caa8fed6c0ac0.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/b3a2ae5c8bfdb0a283f7fdb36430dd20344b17d0db92cae6bc9caa8fed6c0ac0.jpg)

![ba41a73bbff0cde7f02157ced5887664b714deb02a57172236354865f07283e2.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/ba41a73bbff0cde7f02157ced5887664b714deb02a57172236354865f07283e2.jpg)

![f82b3265df22bea01e8af64baeaa0393a8619ac71ee4777106c7e89883c5ead6.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/f82b3265df22bea01e8af64baeaa0393a8619ac71ee4777106c7e89883c5ead6.jpg)

![f8916b6798bd4884bf21f72fcc55c09e4ea2003033ebcd229818ddc41b31890e.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/f8916b6798bd4884bf21f72fcc55c09e4ea2003033ebcd229818ddc41b31890e.jpg)

![fca002cf2cd57ebd3894632be7ce2177bb44ec9fb397d237d9cb0efb5c1616c3.jpg](../icml_results/5533_Return-to-Go%20Is%20More%20Than%20a%20Number_%20Q-Guided%20Alignment%20for%20Return-Conditioned%20Supervised%20Learning/tables/fca002cf2cd57ebd3894632be7ce2177bb44ec9fb397d237d9cb0efb5c1616c3.jpg)

<a id="paper-5486"></a>

## NBCG: Nash-Bargained Causal Game for Long-Tailed Multi-Label NLP


### Images

![32ce4f534df611ffa4445ba6d05362c22cc9cdb879a4aa348ee93aa5649ac7c6.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/32ce4f534df611ffa4445ba6d05362c22cc9cdb879a4aa348ee93aa5649ac7c6.jpg)

![439ebeb9f78dd3dfe17b3bd5daf0d2ecbd7c884649d22ac3139e1ca1de72212c.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/439ebeb9f78dd3dfe17b3bd5daf0d2ecbd7c884649d22ac3139e1ca1de72212c.jpg)

![4ff698a3f4b126f40b4279796b70d8580cdf56c995516be30b8e6999902e2bd5.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/4ff698a3f4b126f40b4279796b70d8580cdf56c995516be30b8e6999902e2bd5.jpg)

![6a526eee25cc8b17d6310bfbf3e9414df9c22aceb0c1546b3d3b46c680053743.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/6a526eee25cc8b17d6310bfbf3e9414df9c22aceb0c1546b3d3b46c680053743.jpg)

![af45bd7099cec6c268cd72bd277d404c2554dd86a16809035834035eb185dd16.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/af45bd7099cec6c268cd72bd277d404c2554dd86a16809035834035eb185dd16.jpg)

![d0f33db2ff2c67bffe59cd3b78d2d9e06d6ace0bbe40f56055971cf3466f6923.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/d0f33db2ff2c67bffe59cd3b78d2d9e06d6ace0bbe40f56055971cf3466f6923.jpg)

![d4105c127bd55b7e74b2634b6ef13728597cccea59520c6e5a2b5dd5200b4276.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/images/d4105c127bd55b7e74b2634b6ef13728597cccea59520c6e5a2b5dd5200b4276.jpg)

### Tables

![1ef8151286353463df5c32dff6f724f6bceb1f087ba4de67d743f28bc2a8c0f3.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/1ef8151286353463df5c32dff6f724f6bceb1f087ba4de67d743f28bc2a8c0f3.jpg)

![2257f81dd6bab75e8ee75c164eff496cdee5bb337de88495231b08de107afab1.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/2257f81dd6bab75e8ee75c164eff496cdee5bb337de88495231b08de107afab1.jpg)

![33f72158f010530a817b971ad8710ee32070ff74daf557a11579905720f0d0ff.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/33f72158f010530a817b971ad8710ee32070ff74daf557a11579905720f0d0ff.jpg)

![4de8488f9c5aa40247eda78bdde2b2b38d7937dc6dc61510ebdfc92b2ecdf6be.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/4de8488f9c5aa40247eda78bdde2b2b38d7937dc6dc61510ebdfc92b2ecdf6be.jpg)

![58a80451b5bed23cc294bedff3a1e139ddb0ec350515d052ac7212ab809b6cc1.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/58a80451b5bed23cc294bedff3a1e139ddb0ec350515d052ac7212ab809b6cc1.jpg)

![a1d816f5fca6217772e9364a687b04ec8b5b5677b3ca38156deb403ee6ac7232.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/a1d816f5fca6217772e9364a687b04ec8b5b5677b3ca38156deb403ee6ac7232.jpg)

![a25082017b1fad193355bd46881bde108a876a3fe646a509fcef85bd85812b1a.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/a25082017b1fad193355bd46881bde108a876a3fe646a509fcef85bd85812b1a.jpg)

![b905d240fc01ce1c5e26f6efedb1384dd4d7ae65cda29048dbbd42528064262f.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/b905d240fc01ce1c5e26f6efedb1384dd4d7ae65cda29048dbbd42528064262f.jpg)

![c980bce36a1a74c207d0fbe6fe608ccf84e44cce0686f0f687a644e6a4bf4b5c.jpg](../icml_results/5534_NBCG_%20Nash-Bargained%20Causal%20Game%20for%20Long-Tailed%20Multi-Label%20NLP/tables/c980bce36a1a74c207d0fbe6fe608ccf84e44cce0686f0f687a644e6a4bf4b5c.jpg)

<a id="paper-5487"></a>

## Stable Velocity: A Variance Perspective on Flow Matching


### Images

![0020b9dcd518b6f1e861d8e670aa01d682768520fd215e4a6f29fc5f6f7c05aa.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/0020b9dcd518b6f1e861d8e670aa01d682768520fd215e4a6f29fc5f6f7c05aa.jpg)

![0067a61ddc71c08a3e49ea9b9b25106af5b56823c41305be51801645cbae00ae.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/0067a61ddc71c08a3e49ea9b9b25106af5b56823c41305be51801645cbae00ae.jpg)

![0b57cc8a8005a291407b49169c501b84bd8a6741121d3295ba8f1099563a4c4d.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/0b57cc8a8005a291407b49169c501b84bd8a6741121d3295ba8f1099563a4c4d.jpg)

![1fd54abe7adf6950c3180e70a7f9ee2bad9c603bf1b044543b157833dc60b993.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/1fd54abe7adf6950c3180e70a7f9ee2bad9c603bf1b044543b157833dc60b993.jpg)

![27be275a0d0afec031a063bb3ce03cb590c3c9ed1a2a927fd97847ea02db2be6.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/27be275a0d0afec031a063bb3ce03cb590c3c9ed1a2a927fd97847ea02db2be6.jpg)

![2c96d1bdae467e809659bf6b0f3f811dd1d3d779c579c96fc54a50b9992258b9.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/2c96d1bdae467e809659bf6b0f3f811dd1d3d779c579c96fc54a50b9992258b9.jpg)

![4b86a953946ce84dee5f23a47c60228f6c3b822e2005fe0192856a0887a3036d.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/4b86a953946ce84dee5f23a47c60228f6c3b822e2005fe0192856a0887a3036d.jpg)

![6a4f52c41f3af34170aba5aea78a7501a052fd874a2fa6ccf07239054c97a5a3.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/6a4f52c41f3af34170aba5aea78a7501a052fd874a2fa6ccf07239054c97a5a3.jpg)

![8499e031c0fa469882dedf4c2d724c056d766e60ad1568fcbe22a4acaf163461.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/8499e031c0fa469882dedf4c2d724c056d766e60ad1568fcbe22a4acaf163461.jpg)

![97badb40ba56fb70cbda719bb46f193fe68d0665aa826ef44b66e44061f56254.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/97badb40ba56fb70cbda719bb46f193fe68d0665aa826ef44b66e44061f56254.jpg)

![c18e6fb0b55833d5d5948da48e06b57c8ba5b9df9ad8e6026053e14966231999.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/c18e6fb0b55833d5d5948da48e06b57c8ba5b9df9ad8e6026053e14966231999.jpg)

![d23d77a236db21e30601124ff050b0c0aa148c5e99beb75548ebb939b00e8991.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/d23d77a236db21e30601124ff050b0c0aa148c5e99beb75548ebb939b00e8991.jpg)

![dc5e38c7ad54a0c0a2e137779ef0df1bba6dd1566980154c4dcbb593c8daee74.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/dc5e38c7ad54a0c0a2e137779ef0df1bba6dd1566980154c4dcbb593c8daee74.jpg)

![f121c38f4084c5132897b591c1e0fb8094bf3e204190b003b6f538ba5a199adb.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/f121c38f4084c5132897b591c1e0fb8094bf3e204190b003b6f538ba5a199adb.jpg)

![f6fe72a509ca203520008bf5f815fc920840dd948d04646ba5eb3dec8748a172.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/images/f6fe72a509ca203520008bf5f815fc920840dd948d04646ba5eb3dec8748a172.jpg)

### Tables

![0a693ae741033f3b8280d0d5158d3f36aa512546ed99261566567309cde23f99.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/0a693ae741033f3b8280d0d5158d3f36aa512546ed99261566567309cde23f99.jpg)

![505f6b19e434b1ffd5cf03836d308f3a2677bad742de2b3e05b46f357419a4f7.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/505f6b19e434b1ffd5cf03836d308f3a2677bad742de2b3e05b46f357419a4f7.jpg)

![577da0172874dbc3152f2d34774790384d2dbfe1e3f436290b872ba681899bc1.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/577da0172874dbc3152f2d34774790384d2dbfe1e3f436290b872ba681899bc1.jpg)

![a079e9b70a56175dc9855aff08bbb71f9581aacac384f7bfe3e9453f5dfff87f.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/a079e9b70a56175dc9855aff08bbb71f9581aacac384f7bfe3e9453f5dfff87f.jpg)

![cc646c2d0d6e1e0db95fb61137e99e3d5ac2190f63a36d9f3d14cbffff705db7.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/cc646c2d0d6e1e0db95fb61137e99e3d5ac2190f63a36d9f3d14cbffff705db7.jpg)

![d505f42b3609bfbf7f805914977547089d671ffa339a6fd26658424d6ead1594.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/d505f42b3609bfbf7f805914977547089d671ffa339a6fd26658424d6ead1594.jpg)

![db14de66ce525e4f44dcb113596be72a53d19c74ada3ffa696bbeb5d4fc6b5ad.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/db14de66ce525e4f44dcb113596be72a53d19c74ada3ffa696bbeb5d4fc6b5ad.jpg)

![f398e1cfc5334470f7847540f87e18c452b44587a9b82981aa09247ed7930480.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/f398e1cfc5334470f7847540f87e18c452b44587a9b82981aa09247ed7930480.jpg)

![fbe37133264e603cb72e834189932c78767091c31560327f4682efcd2f2b7604.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/fbe37133264e603cb72e834189932c78767091c31560327f4682efcd2f2b7604.jpg)

![fdfb5fa51051986e63b883e943047ceaeb383dc360584f341984a78ebdfc04b0.jpg](../icml_results/5535_Stable%20Velocity_%20A%20Variance%20Perspective%20on%20Flow%20Matching/tables/fdfb5fa51051986e63b883e943047ceaeb383dc360584f341984a78ebdfc04b0.jpg)

<a id="paper-5488"></a>

## EpiTwin: Spatiotemporal Graph Transformers for Epileptic sEEG Signal Reconstruction


### Images

![2ead72b9e5c11108b011f89dfbfeda0ae5d022d7ed088f18664fdde90bf34dd0.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/2ead72b9e5c11108b011f89dfbfeda0ae5d022d7ed088f18664fdde90bf34dd0.jpg)

![38528ea04d4a1ee6484fd9ab4f2b2cd2804ae2a83d5257ef6d553a2d50b78f57.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/38528ea04d4a1ee6484fd9ab4f2b2cd2804ae2a83d5257ef6d553a2d50b78f57.jpg)

![5a6d45bce157d11096a7e0b78e68e9943d81dd7a864d0f05f02b49ea48bdb1bf.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/5a6d45bce157d11096a7e0b78e68e9943d81dd7a864d0f05f02b49ea48bdb1bf.jpg)

![a11222f09d460d39c9145becb74f8bf05e5f4fdd6627ddecb0bc09f9a2c06ee4.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/a11222f09d460d39c9145becb74f8bf05e5f4fdd6627ddecb0bc09f9a2c06ee4.jpg)

![b3d0feee03019830399425be2494ed63e3a34a9047e6f5c0349b67155d1837a7.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/b3d0feee03019830399425be2494ed63e3a34a9047e6f5c0349b67155d1837a7.jpg)

![c9626a0de28353cf02ff885c5e8af31cf525973980385cba6f11c73306e0f4b7.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/c9626a0de28353cf02ff885c5e8af31cf525973980385cba6f11c73306e0f4b7.jpg)

![e4ab36b9c1d929b52c68daf3863f0e8e753d568c316a8fdd71b151863003edc5.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/e4ab36b9c1d929b52c68daf3863f0e8e753d568c316a8fdd71b151863003edc5.jpg)

![ecaf94c750a5a49d0b65fc35179a994f71b8abbfefeb5cc209bbe4e9e799deb8.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/images/ecaf94c750a5a49d0b65fc35179a994f71b8abbfefeb5cc209bbe4e9e799deb8.jpg)

### Tables

![5da51470a8b9efb06fcbf7fcaf53daab4a387e4a85d4996a2653d58a4fd1c3ac.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/5da51470a8b9efb06fcbf7fcaf53daab4a387e4a85d4996a2653d58a4fd1c3ac.jpg)

![67d50541e4a73ae63baf1f0c0519ab6dc500a523ec157073b9d0a8f455ec69d6.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/67d50541e4a73ae63baf1f0c0519ab6dc500a523ec157073b9d0a8f455ec69d6.jpg)

![8275115a84da52d4e6f5f0b20c74036572aee87a634dc872a388d35408bb6ae0.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/8275115a84da52d4e6f5f0b20c74036572aee87a634dc872a388d35408bb6ae0.jpg)

![8cfec6cf4029d5918e95ba52589ef85e5512aeeee4ff75eb206968e43d5916ce.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/8cfec6cf4029d5918e95ba52589ef85e5512aeeee4ff75eb206968e43d5916ce.jpg)

![8d9ba5d51deca0aedaf1e2c8d8be6f546e5885dbb37340d4b114e0a17c9ab497.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/8d9ba5d51deca0aedaf1e2c8d8be6f546e5885dbb37340d4b114e0a17c9ab497.jpg)

![8daba53ab89f149c343e5e39bd9688c6758718da6b91aba4c0ed4a9052926ade.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/8daba53ab89f149c343e5e39bd9688c6758718da6b91aba4c0ed4a9052926ade.jpg)

![956aed239a34ec01812d2ce277cc032e4a7b221a9b746e414bfd2d386f57e913.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/956aed239a34ec01812d2ce277cc032e4a7b221a9b746e414bfd2d386f57e913.jpg)

![958e84417fbd221290901aaa7540f86f25980baf4369f9d543b703fa3c85b494.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/958e84417fbd221290901aaa7540f86f25980baf4369f9d543b703fa3c85b494.jpg)

![c3518713ddda2ad4ee4458c31908360308cbbbdd510e6e9a681f4e9cee943ec8.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/c3518713ddda2ad4ee4458c31908360308cbbbdd510e6e9a681f4e9cee943ec8.jpg)

![c792179d87900e344d8e135d12624f01a805db957c7b127f24663b89173e1d5d.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/c792179d87900e344d8e135d12624f01a805db957c7b127f24663b89173e1d5d.jpg)

![c9c544cdb96d19b979aaef85dbe874c482c7818d7eba6f60aa73dfd9eee72507.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/c9c544cdb96d19b979aaef85dbe874c482c7818d7eba6f60aa73dfd9eee72507.jpg)

![cd5d636ec40ddc1223d7d29ad5e8b4464c5960d377eac5866935e98c6ec31e93.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/cd5d636ec40ddc1223d7d29ad5e8b4464c5960d377eac5866935e98c6ec31e93.jpg)

![cecbe68318c25700e62184dd572db2c67a9356392b4d31fa2ab7ab9f0301f550.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/cecbe68318c25700e62184dd572db2c67a9356392b4d31fa2ab7ab9f0301f550.jpg)

![dff63aa0304c901bfdd240e60432a06a0f21efded399ba5bfe486c48f7cb4f6c.jpg](../icml_results/5536_EpiTwin_%20Spatiotemporal%20Graph%20Transformers%20for%20Epileptic%20sEEG%20Signal%20Reconstruction/tables/dff63aa0304c901bfdd240e60432a06a0f21efded399ba5bfe486c48f7cb4f6c.jpg)

<a id="paper-5489"></a>

## ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling


### Images

![151c30cb13ddb071816cc6397f5a8c9f419cfe81f51750db889ca0ce9911617a.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/151c30cb13ddb071816cc6397f5a8c9f419cfe81f51750db889ca0ce9911617a.jpg)

![26ec7a280d4e03230ded3d891e69960bab7da4a4a5c269a36248c50af6791317.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/26ec7a280d4e03230ded3d891e69960bab7da4a4a5c269a36248c50af6791317.jpg)

![305e0504882ef053fb0ddc17ab48c0e51585be2654f68add6ca57a403d737ad7.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/305e0504882ef053fb0ddc17ab48c0e51585be2654f68add6ca57a403d737ad7.jpg)

![33db82f958eea0e555c3e8a147055d7012360d7fede28762a6a9812df2b5d717.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/33db82f958eea0e555c3e8a147055d7012360d7fede28762a6a9812df2b5d717.jpg)

![6536abecc89ea71f3bf1ce4a9299fa9a4810c5e6c25b8513085c9f67c03eacfe.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/6536abecc89ea71f3bf1ce4a9299fa9a4810c5e6c25b8513085c9f67c03eacfe.jpg)

![699b4215e06d4cf4d94efb6e0d54fe295c332e0763a966410b3c3098da9c68ce.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/699b4215e06d4cf4d94efb6e0d54fe295c332e0763a966410b3c3098da9c68ce.jpg)

![6e147b07609be9b95028058f9c9882803e891c8fea3980a2f2f07091b8d1022d.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/6e147b07609be9b95028058f9c9882803e891c8fea3980a2f2f07091b8d1022d.jpg)

![73773d1718504639538a7b87cc8f3347efb5cae6dc2eca0a47f49c8a6e48b4dd.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/73773d1718504639538a7b87cc8f3347efb5cae6dc2eca0a47f49c8a6e48b4dd.jpg)

![9a6d010c9147ac5596ca683040ddb9ff1c297638c7e62f75b3c89e635085ab97.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/9a6d010c9147ac5596ca683040ddb9ff1c297638c7e62f75b3c89e635085ab97.jpg)

![9fecbf107c1047794c10ee399ba594efc9f2ab1439730a067d8cc5d6c1b66c49.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/9fecbf107c1047794c10ee399ba594efc9f2ab1439730a067d8cc5d6c1b66c49.jpg)

![aca2cbeded1e504f03af395f30b629ba2c7d7ca9169dc3b7c1c4f949039987eb.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/aca2cbeded1e504f03af395f30b629ba2c7d7ca9169dc3b7c1c4f949039987eb.jpg)

![ebbe26497465db2d9fb11e1a2042433bfac47cc98125c3e9c8626ec7eac8c9e5.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/images/ebbe26497465db2d9fb11e1a2042433bfac47cc98125c3e9c8626ec7eac8c9e5.jpg)

### Tables

![2af4ab585acb63ece01ccf2e363475af3394bd9c64095bf1d68a831210070b7b.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/2af4ab585acb63ece01ccf2e363475af3394bd9c64095bf1d68a831210070b7b.jpg)

![3a61705bec3df5fddd2b9ed9fc3faac93fd807a4314a0f14c914340a5b5a62f5.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/3a61705bec3df5fddd2b9ed9fc3faac93fd807a4314a0f14c914340a5b5a62f5.jpg)

![4c7a8d70e168f0319e700b05a2972cfddf52aead330df2492b13e2e1e36ad2f1.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/4c7a8d70e168f0319e700b05a2972cfddf52aead330df2492b13e2e1e36ad2f1.jpg)

![8ab1410e776e60c809ad7291b3c47fd7955fa0b6bdc9643cd9d9b0c3c853ec05.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/8ab1410e776e60c809ad7291b3c47fd7955fa0b6bdc9643cd9d9b0c3c853ec05.jpg)

![93d1521ede3b99b705dceb08220c7b168ff163e0e30bd7469098d28a80769ec4.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/93d1521ede3b99b705dceb08220c7b168ff163e0e30bd7469098d28a80769ec4.jpg)

![a7e75d783acb622292733623512557bee6e1fff500479d9522749eae672385bf.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/a7e75d783acb622292733623512557bee6e1fff500479d9522749eae672385bf.jpg)

![d6017094e34cd282f17c910d2b14c12e0818980d38eb64b496d60698eee05965.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/d6017094e34cd282f17c910d2b14c12e0818980d38eb64b496d60698eee05965.jpg)

![ea4d3abd4a7349f113ec32baf647a5007d004b06aa095b7d90e20b4b24a79d2e.jpg](../icml_results/5537_ZipMoE_%20Efficient%20On-Device%20MoE%20Serving%20via%20Lossless%20Compression%20and%20Cache-Affinity%20Scheduling/tables/ea4d3abd4a7349f113ec32baf647a5007d004b06aa095b7d90e20b4b24a79d2e.jpg)

<a id="paper-5490"></a>

## Adaptive Multiscale Binary Expansion Tests for Independence


### Images

![125ab61f7f6f35a05756543848fbf6535e7ff13efd675572d15c672b11a1ef61.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/125ab61f7f6f35a05756543848fbf6535e7ff13efd675572d15c672b11a1ef61.jpg)

![3edc051340cfa7ddabbdeda09a3eb7620a45c350b8eed9d7a86ad9f243ceb2fa.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/3edc051340cfa7ddabbdeda09a3eb7620a45c350b8eed9d7a86ad9f243ceb2fa.jpg)

![485f98b737694c89197668589f92cbd902c32ce779c652fea3a882d141dde6db.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/485f98b737694c89197668589f92cbd902c32ce779c652fea3a882d141dde6db.jpg)

![78831831d6c0cbfeceb7901f67c52014ef53b4da2c6e744b1181e791ad85073d.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/78831831d6c0cbfeceb7901f67c52014ef53b4da2c6e744b1181e791ad85073d.jpg)

![a4402d55fb3d842e9469d6a9669859f3e06c02490975028c1ee3f52f7fbea6b9.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/a4402d55fb3d842e9469d6a9669859f3e06c02490975028c1ee3f52f7fbea6b9.jpg)

![a61008445cb2cca9e815e0e86db391d0a81f8ba08f4f0946302bf9156b417282.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/a61008445cb2cca9e815e0e86db391d0a81f8ba08f4f0946302bf9156b417282.jpg)

![f59821dfaa1990912f8fe9fb0c64711959833622db7ecf3cbde82efb0ae93667.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/images/f59821dfaa1990912f8fe9fb0c64711959833622db7ecf3cbde82efb0ae93667.jpg)

### Tables

![33a249135cf484f3c0b58633d02dd676b77527b76bcfac633df44bfc2b1d7f47.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/tables/33a249135cf484f3c0b58633d02dd676b77527b76bcfac633df44bfc2b1d7f47.jpg)

![44846e73295caa6c571d45e1d1522a31f3b1732164b72b159b9d1c845d6d725c.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/tables/44846e73295caa6c571d45e1d1522a31f3b1732164b72b159b9d1c845d6d725c.jpg)

![6b41b927f52059c7d3fc65c8ea812cdded84b5cbb87a009abc360d825e719b86.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/tables/6b41b927f52059c7d3fc65c8ea812cdded84b5cbb87a009abc360d825e719b86.jpg)

![c90a34255fd71316e5788492a269ed336f6b2bc04230846164c9c29e72a50730.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/tables/c90a34255fd71316e5788492a269ed336f6b2bc04230846164c9c29e72a50730.jpg)

![db7763a1c5ba0f80e6a2e9603663ef32bd787e529f602e3d880ec0b64c5f76d2.jpg](../icml_results/5538_Adaptive%20Multiscale%20Binary%20Expansion%20Tests%20for%20Independence/tables/db7763a1c5ba0f80e6a2e9603663ef32bd787e529f602e3d880ec0b64c5f76d2.jpg)

<a id="paper-5491"></a>

## Multi-Objective Protein Design via Memory-Aware Test-Time Scaling in Diffusion Models


### Images

![1c09370ee77e65088016e067f31f480d4cf1075a83c126fd440dcfa21244ad3e.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/1c09370ee77e65088016e067f31f480d4cf1075a83c126fd440dcfa21244ad3e.jpg)

![29bd9fbf12308c32a2f2298ba42f7dd178471cb89cd5a9b33b7935b2235859fa.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/29bd9fbf12308c32a2f2298ba42f7dd178471cb89cd5a9b33b7935b2235859fa.jpg)

![46ba9370649208a95628fb1f16e3ef575bef1ba30cd0aef31dfde64e0de5144a.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/46ba9370649208a95628fb1f16e3ef575bef1ba30cd0aef31dfde64e0de5144a.jpg)

![7820349097ea9d54bf8fe4e2e1b0be265b6fa704ad5f7a9702aa10d706cde5ef.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/7820349097ea9d54bf8fe4e2e1b0be265b6fa704ad5f7a9702aa10d706cde5ef.jpg)

![aa0c7d18f749749a86bec58f784fc576db9fb982aa2f58c39e764a51d5c0d230.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/aa0c7d18f749749a86bec58f784fc576db9fb982aa2f58c39e764a51d5c0d230.jpg)

![c7faebac08f372bf4fc5e3397fa26dca2ad2f5bbd164300996db4895784c303f.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/images/c7faebac08f372bf4fc5e3397fa26dca2ad2f5bbd164300996db4895784c303f.jpg)

### Tables

![15cf0d020c4b97260075e03cf8e1adf424cb234264d107ab49cac5ca1a0b50d3.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/15cf0d020c4b97260075e03cf8e1adf424cb234264d107ab49cac5ca1a0b50d3.jpg)

![3d47886cdc4a907d8d5f68717763ddf949029f7f51744d41d496588298692ff4.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/3d47886cdc4a907d8d5f68717763ddf949029f7f51744d41d496588298692ff4.jpg)

![56943a6e37f500bde824673d96f664329615bd071d64691db153e14180a6a1ae.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/56943a6e37f500bde824673d96f664329615bd071d64691db153e14180a6a1ae.jpg)

![57e5e50ca0324b04dcb0225bfb2353d2db2b5847177dbffdfbdd178e98b6432a.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/57e5e50ca0324b04dcb0225bfb2353d2db2b5847177dbffdfbdd178e98b6432a.jpg)

![6b7d585646c6f605c947523e9cd2e16c61b86cd4e14ca31c7649d72258421bb8.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/6b7d585646c6f605c947523e9cd2e16c61b86cd4e14ca31c7649d72258421bb8.jpg)

![9d4613ae7c44f015d8601d62d95329734c02ccb17f36d77b99dc63390f70c1e1.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/9d4613ae7c44f015d8601d62d95329734c02ccb17f36d77b99dc63390f70c1e1.jpg)

![a5b46c4771622d13099906e873430dbde1b55ab93528a36e3aa198dfb89f718f.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/a5b46c4771622d13099906e873430dbde1b55ab93528a36e3aa198dfb89f718f.jpg)

![f2f731e4a28d745dfdd5c03249a5e139d16168100286675dbdcbc9d73636b6f4.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/f2f731e4a28d745dfdd5c03249a5e139d16168100286675dbdcbc9d73636b6f4.jpg)

![fc965359e76dcb607dfd563c1c7a9dfed139c75dc0d1140a336da7b872de6990.jpg](../icml_results/5539_Multi-Objective%20Protein%20Design%20via%20Memory-Aware%20Test-Time%20Scaling%20in%20Diffusion%20Models/tables/fc965359e76dcb607dfd563c1c7a9dfed139c75dc0d1140a336da7b872de6990.jpg)

<a id="paper-5492"></a>

## Profiling the Irrational Agent: Cognitive Modeling of LLM Behaviors in Sequential Jailbreaks


### Images

![05b89f076eacde4c06584ce92ce40c1b95e0499ad5f28714384f1ff0f37ecb71.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/05b89f076eacde4c06584ce92ce40c1b95e0499ad5f28714384f1ff0f37ecb71.jpg)

![0647b0f19b054b1a13158dc2c6407fe8d1e3c3028703851dc5802044f950fa62.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/0647b0f19b054b1a13158dc2c6407fe8d1e3c3028703851dc5802044f950fa62.jpg)

![1ee52c5189e23ae9ddd4b790d09cc4ca99ebd3a6967f860431534a5202359f8f.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/1ee52c5189e23ae9ddd4b790d09cc4ca99ebd3a6967f860431534a5202359f8f.jpg)

![5f1632d792d93bc24c7595aa7aff4cfee91dc918247ac82b90a080a371034c29.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/5f1632d792d93bc24c7595aa7aff4cfee91dc918247ac82b90a080a371034c29.jpg)

![76a3ffb82313ad9099983f7d9cca622eb20a87325a6bf0591ee89761154dab8d.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/76a3ffb82313ad9099983f7d9cca622eb20a87325a6bf0591ee89761154dab8d.jpg)

![a6926f3109a3401c227420a069b067faed10440f532d87769ee430375c223acc.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/a6926f3109a3401c227420a069b067faed10440f532d87769ee430375c223acc.jpg)

![cc48f1e21951bb5ab0228c9dd9363d1bf9483327ec50f1b0626843fbdadabf84.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/cc48f1e21951bb5ab0228c9dd9363d1bf9483327ec50f1b0626843fbdadabf84.jpg)

![dca962946eb49f7179d0fcde7d0e6b18d3ffafd5385a82fc6de9f8bd831772d6.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/dca962946eb49f7179d0fcde7d0e6b18d3ffafd5385a82fc6de9f8bd831772d6.jpg)

![f4bcb83a726a7f8caa11851a70888a9172709c2df5878b441e01806e8872d4e2.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/images/f4bcb83a726a7f8caa11851a70888a9172709c2df5878b441e01806e8872d4e2.jpg)

### Tables

![1eac07307c47bb585df5d1ee35737cb35d1ba8bee6702de1be4cd9a88b627ee2.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/1eac07307c47bb585df5d1ee35737cb35d1ba8bee6702de1be4cd9a88b627ee2.jpg)

![28244e8aedcccfa4d60ecb613ad7283429daa8804f1ba16caa6780ad3c717d78.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/28244e8aedcccfa4d60ecb613ad7283429daa8804f1ba16caa6780ad3c717d78.jpg)

![5396468c998859eae03e7956e21df4c20f7a1faeff90db85d8cc6848eb3a379e.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/5396468c998859eae03e7956e21df4c20f7a1faeff90db85d8cc6848eb3a379e.jpg)

![590f5b36acfbb67f4e0baed2fd5248d7bb7800fad504a34237eec4eead63b0eb.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/590f5b36acfbb67f4e0baed2fd5248d7bb7800fad504a34237eec4eead63b0eb.jpg)

![5afa8c1c02fcf2ea903241b93b8a8eb83d7c7abd0592655a95595dba9f6730a4.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/5afa8c1c02fcf2ea903241b93b8a8eb83d7c7abd0592655a95595dba9f6730a4.jpg)

![ab1b01c9f560973a1bf43544fbe4b4ef4e494cfbed33225e6b9d00e9958406ee.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/ab1b01c9f560973a1bf43544fbe4b4ef4e494cfbed33225e6b9d00e9958406ee.jpg)

![bf0cd51af92621bb72f65a2fc798bb043b74043b644e23e513099ba4d8100ecb.jpg](../icml_results/5540_Profiling%20the%20Irrational%20Agent_%20Cognitive%20Modeling%20of%20LLM%20Behaviors%20in%20Sequential%20Jailbreaks/tables/bf0cd51af92621bb72f65a2fc798bb043b74043b644e23e513099ba4d8100ecb.jpg)

<a id="paper-5493"></a>

## Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction


### Images

![9387c265576d29273e0ba44861d9bb23dbce40380dd87c49b909cd33fda2be86.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/images/9387c265576d29273e0ba44861d9bb23dbce40380dd87c49b909cd33fda2be86.jpg)

![94d15459ef2c944e1457094bc8928c89428c0f86073950167267762b6ef8cda3.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/images/94d15459ef2c944e1457094bc8928c89428c0f86073950167267762b6ef8cda3.jpg)

![97d212bd9dcf62d6fda431c6dd27671e6c8f349743be7e111ab94dc1920c45b7.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/images/97d212bd9dcf62d6fda431c6dd27671e6c8f349743be7e111ab94dc1920c45b7.jpg)

![ba300d445bd15d928878bc7553bc6e3df98ab4107dd23cdb2af2a7d8f522947f.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/images/ba300d445bd15d928878bc7553bc6e3df98ab4107dd23cdb2af2a7d8f522947f.jpg)

### Tables

![00459b08f785bef58a681807eb40be650894be0e0b56cb70ab5adf31d0371539.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/00459b08f785bef58a681807eb40be650894be0e0b56cb70ab5adf31d0371539.jpg)

![0ca0d49b479df9fee4f5a88a3a0076650ba2ba30ac458c35d8b1ec1cdadc4cf0.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/0ca0d49b479df9fee4f5a88a3a0076650ba2ba30ac458c35d8b1ec1cdadc4cf0.jpg)

![2d2e6d96631963ef9be8667ec7e4edefa1910d623f24d2c5f9b0b426df6d3ed9.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/2d2e6d96631963ef9be8667ec7e4edefa1910d623f24d2c5f9b0b426df6d3ed9.jpg)

![49647a31a005a3eebedb53f645618dc163e10565d446f4070a2586456830ec1f.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/49647a31a005a3eebedb53f645618dc163e10565d446f4070a2586456830ec1f.jpg)

![530e3db16da3e6179175253846d096e9cc0804a235513e8412207188381f98f9.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/530e3db16da3e6179175253846d096e9cc0804a235513e8412207188381f98f9.jpg)

![a28e48ef2c1a63e9bd605ec004da23ebe144ac70af7e47513010493c619d1be4.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/a28e48ef2c1a63e9bd605ec004da23ebe144ac70af7e47513010493c619d1be4.jpg)

![ae40afaa266c2f175c157b0f431a0718e3f15b5ebb7747397ccd338649497ee8.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/ae40afaa266c2f175c157b0f431a0718e3f15b5ebb7747397ccd338649497ee8.jpg)

![b3544ab463b4a3a543f9ed7f5922672869f89c47159a70c46a734ae7db7b5c2f.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/b3544ab463b4a3a543f9ed7f5922672869f89c47159a70c46a734ae7db7b5c2f.jpg)

![b86d4389260820df3a267a767a6155335544fb6db84cab0fd0ca3eb3127a2c7b.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/b86d4389260820df3a267a767a6155335544fb6db84cab0fd0ca3eb3127a2c7b.jpg)

![d4ee55e9fb97ddf87ef380f904b523ec45cdb0888197c8c0644f87f5e9dadc7d.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/d4ee55e9fb97ddf87ef380f904b523ec45cdb0888197c8c0644f87f5e9dadc7d.jpg)

![e6fb432332965f6b6623f0d0e63e10d0bb77c2f3deecb2f9cd04fc3718c3c9b5.jpg](../icml_results/5541_Probing%20the%20Knowledge%20Boundary_%20An%20Interactive%20Agentic%20Framework%20for%20Deep%20Knowledge%20Extraction/tables/e6fb432332965f6b6623f0d0e63e10d0bb77c2f3deecb2f9cd04fc3718c3c9b5.jpg)

<a id="paper-5494"></a>

## Shapley Regularized Neural Granger Causality


### Images

![462f976ff7959c3c12142dc5aaf81041a55b93ea52a2d00044f65d44fbdcb40b.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/images/462f976ff7959c3c12142dc5aaf81041a55b93ea52a2d00044f65d44fbdcb40b.jpg)

![5674b4055396648e8c6063d5310e8c780700b55b152d8c192b7f1c72bc76b76c.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/images/5674b4055396648e8c6063d5310e8c780700b55b152d8c192b7f1c72bc76b76c.jpg)

![697c42b9cc64e6c129da09f25f1ff8794dd25ff6d8c82d1322cba33cd22a73bf.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/images/697c42b9cc64e6c129da09f25f1ff8794dd25ff6d8c82d1322cba33cd22a73bf.jpg)

![e466ef9f6bdb78ec9d8fd342cacb8929ad6ce4bef6c8dc9d046dcd0948943f9b.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/images/e466ef9f6bdb78ec9d8fd342cacb8929ad6ce4bef6c8dc9d046dcd0948943f9b.jpg)

### Tables

![845cb08e9b2ee2fb1049611c217e9cb2f1ecf73d0d21be516ec9fc49cbe6ddb5.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/845cb08e9b2ee2fb1049611c217e9cb2f1ecf73d0d21be516ec9fc49cbe6ddb5.jpg)

![ba16184e000952bf3d4cedea7e262d2967dbbaf505276f88c019dae445d02d6c.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/ba16184e000952bf3d4cedea7e262d2967dbbaf505276f88c019dae445d02d6c.jpg)

![cdf0c7c2b208fa19b05cb1d5c8a6d79e81e1772155946940cfbc2f42a2c27229.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/cdf0c7c2b208fa19b05cb1d5c8a6d79e81e1772155946940cfbc2f42a2c27229.jpg)

![de5824d26f4ed3298e4ea7135db0560807e6cff3479f6a434da804d89da57838.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/de5824d26f4ed3298e4ea7135db0560807e6cff3479f6a434da804d89da57838.jpg)

![e6d9dba0dab0d879782619feebf9e479b1bad34d38cbed615acad34b37ac4924.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/e6d9dba0dab0d879782619feebf9e479b1bad34d38cbed615acad34b37ac4924.jpg)

![fff8d9a1af0707c85aea1caafb7853c5029a5fcabfb1a9a97ce208a17eb1cb4b.jpg](../icml_results/5542_Shapley%20Regularized%20Neural%20Granger%20Causality/tables/fff8d9a1af0707c85aea1caafb7853c5029a5fcabfb1a9a97ce208a17eb1cb4b.jpg)

<a id="paper-5495"></a>

## Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization


### Images

![3e98af2b19d8e2f4cfce38df84da9f916a6a333d5958c672ac64e2c1ff7e5e77.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/3e98af2b19d8e2f4cfce38df84da9f916a6a333d5958c672ac64e2c1ff7e5e77.jpg)

![6334dd9eb8763def405fad9f20c8a69f34a7ee3c51c1a75ad40c13c824fc880f.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/6334dd9eb8763def405fad9f20c8a69f34a7ee3c51c1a75ad40c13c824fc880f.jpg)

![68daab3db090c68d047da51c96c43036099bed9503d883e9cb5d4325d3acdc7c.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/68daab3db090c68d047da51c96c43036099bed9503d883e9cb5d4325d3acdc7c.jpg)

![71643d7306f718d5fc1395d63078c83d1adf8af9f1117e5d4a65efe791281a24.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/71643d7306f718d5fc1395d63078c83d1adf8af9f1117e5d4a65efe791281a24.jpg)

![7594cd7a84822c364938e491c874008bdf02e73ea8c8c39904db5bc47d043e8d.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/7594cd7a84822c364938e491c874008bdf02e73ea8c8c39904db5bc47d043e8d.jpg)

![7b75ab737940ce5bee9bf35190fa7289f44eda5e0279c4a5a4266967de626a7c.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/7b75ab737940ce5bee9bf35190fa7289f44eda5e0279c4a5a4266967de626a7c.jpg)

![88688534ca864f88b73066b71e6c5c91340bcef77cb00b2c43487311aa9ae209.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/88688534ca864f88b73066b71e6c5c91340bcef77cb00b2c43487311aa9ae209.jpg)

![8b94c8ee2793fad471d94d3f75788cb571d3306b723734003f5bf9152906b5f3.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/8b94c8ee2793fad471d94d3f75788cb571d3306b723734003f5bf9152906b5f3.jpg)

![90d22da4425e3a4885928d44135b44a406bcb8c650d9246318dceecd59e4ebb4.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/90d22da4425e3a4885928d44135b44a406bcb8c650d9246318dceecd59e4ebb4.jpg)

![bebd80279ecdf1b4720f79b5d7209b7103f149c947c701d9ff2ad850bc9b393f.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/bebd80279ecdf1b4720f79b5d7209b7103f149c947c701d9ff2ad850bc9b393f.jpg)

![c9dd47af034fa5f9c377a49efa65f9ba3ce21dcddb891441141cc3f82e35e0db.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/c9dd47af034fa5f9c377a49efa65f9ba3ce21dcddb891441141cc3f82e35e0db.jpg)

![d231be240bc00c438ac8af8154ca57e91e6b6a64b3ff64c7877ceda30e334fa1.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/d231be240bc00c438ac8af8154ca57e91e6b6a64b3ff64c7877ceda30e334fa1.jpg)

![d28084620a8fbe2f61c6bd031199e14087de276146b6091da9abf9091872f5dc.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/d28084620a8fbe2f61c6bd031199e14087de276146b6091da9abf9091872f5dc.jpg)

![f6b7a21e73c42cf2a06faf157458690af5c9c3025896ece16f77e70352a51506.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/images/f6b7a21e73c42cf2a06faf157458690af5c9c3025896ece16f77e70352a51506.jpg)

### Tables

![1c425648376dc8bb73988de13aaa7fe2a641c0b46ed29457e52fbb7a93623eda.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/1c425648376dc8bb73988de13aaa7fe2a641c0b46ed29457e52fbb7a93623eda.jpg)

![48355bd5fb02ed7e1f673d52889a101beaccc00e7f6e1c6a700575340c0afb2a.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/48355bd5fb02ed7e1f673d52889a101beaccc00e7f6e1c6a700575340c0afb2a.jpg)

![b17d101e37001dc5f54728109eef5b6edec593f75613d94b687f25c0e27fb8db.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/b17d101e37001dc5f54728109eef5b6edec593f75613d94b687f25c0e27fb8db.jpg)

![bda935b6595516d9dfe04da8097a231fe5df85137a9564607d124aae88040b89.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/bda935b6595516d9dfe04da8097a231fe5df85137a9564607d124aae88040b89.jpg)

![cd9be8a1179ee6460510775e5eae84b84c9228776050b685d3b44b5ce0470d14.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/cd9be8a1179ee6460510775e5eae84b84c9228776050b685d3b44b5ce0470d14.jpg)

![d89a5bebfc1bfb7421b97683e9900b7e982bb769d562fdae377990922719b032.jpg](../icml_results/5543_Pushing%20Forward%20Pareto%20Frontiers%20of%20Proactive%20Agents%20with%20Behavioral%20Agentic%20Optimization/tables/d89a5bebfc1bfb7421b97683e9900b7e982bb769d562fdae377990922719b032.jpg)

<a id="paper-5496"></a>

## Prompt Reinjection: Alleviating Prompt Forgetting in Multimodal Diffusion Transformers


### Images

![1399e9a35987982afaac9e45046cc590eea581959f48c68835b96ffc97459bb3.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/1399e9a35987982afaac9e45046cc590eea581959f48c68835b96ffc97459bb3.jpg)

![287f33615a4dcfe33c6451c87f76636e6f084eb4603fec3fc100143b4110aee9.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/287f33615a4dcfe33c6451c87f76636e6f084eb4603fec3fc100143b4110aee9.jpg)

![292d57d4cc518565e23c7f34fd8c17e78722b48b7ee53c2d6582ba131ddf73cd.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/292d57d4cc518565e23c7f34fd8c17e78722b48b7ee53c2d6582ba131ddf73cd.jpg)

![350655a5df945cbd873060c5762f1f280a7756730603f679397ba4e71f4bf35e.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/350655a5df945cbd873060c5762f1f280a7756730603f679397ba4e71f4bf35e.jpg)

![3510a08de211b162edd86f77e0358e167dd83f2e0d439663c1c8bf513a68edec.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/3510a08de211b162edd86f77e0358e167dd83f2e0d439663c1c8bf513a68edec.jpg)

![4f0400446c275e1434c48e0f1cc257a38fb4b0fad790dbd2167fc318a5eccdca.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/4f0400446c275e1434c48e0f1cc257a38fb4b0fad790dbd2167fc318a5eccdca.jpg)

![4ffe2baccac95d130d4d68208cb5e602a4bd1500fde47a036f404ddfeef36bab.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/4ffe2baccac95d130d4d68208cb5e602a4bd1500fde47a036f404ddfeef36bab.jpg)

![55a2808653a2ac53c7ef67b32e4ed4f342130e4f576ac81bacd9a425cb9392c4.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/55a2808653a2ac53c7ef67b32e4ed4f342130e4f576ac81bacd9a425cb9392c4.jpg)

![676f0600890e8b013863b3e9a97754a20c852d43f77c62dc55ca66ccbabdb358.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/676f0600890e8b013863b3e9a97754a20c852d43f77c62dc55ca66ccbabdb358.jpg)

![6e8c3cabdced8576c188798b0b37969d98ca8ed414b194129bef38753e20c97e.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/6e8c3cabdced8576c188798b0b37969d98ca8ed414b194129bef38753e20c97e.jpg)

![a634d74b9d55061dc2f1ed148eb8672689ab0b481df3c7b05a67e8f4e2eabe2d.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/a634d74b9d55061dc2f1ed148eb8672689ab0b481df3c7b05a67e8f4e2eabe2d.jpg)

![aaba39bd2d53051a3bfaf50e7dfb3bd9a00d13c94fd34027f55ec53b799fbdc9.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/aaba39bd2d53051a3bfaf50e7dfb3bd9a00d13c94fd34027f55ec53b799fbdc9.jpg)

![aca59db8c3cd6aeb0cc7947cee50ba30d30f7fb41dfcb9a5a94a6ef677cd2d4e.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/aca59db8c3cd6aeb0cc7947cee50ba30d30f7fb41dfcb9a5a94a6ef677cd2d4e.jpg)

![c466d845b7dcb532490e4223dc6d40c5297f695d25e2fd11aba57d50908752f9.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/c466d845b7dcb532490e4223dc6d40c5297f695d25e2fd11aba57d50908752f9.jpg)

![d0f4f1b2cafad8d2ecea9a03c29d4515e18bfe97d1b7b9ac91e2f6f224e49a7a.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/d0f4f1b2cafad8d2ecea9a03c29d4515e18bfe97d1b7b9ac91e2f6f224e49a7a.jpg)

![e156778ba833799c2138b40fc798658cdfb240346fc457b51b1cd5acbf461d04.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/e156778ba833799c2138b40fc798658cdfb240346fc457b51b1cd5acbf461d04.jpg)

![e6f5866912b4fd05e741d8c5b0e0e9c5464af375d1bf238b4bbaa64067f64cbe.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/e6f5866912b4fd05e741d8c5b0e0e9c5464af375d1bf238b4bbaa64067f64cbe.jpg)

![f2f6a8131c5f868b1cdeaa120d0f8ee43794cdda3c2b3357d8b65c577ab7f642.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/images/f2f6a8131c5f868b1cdeaa120d0f8ee43794cdda3c2b3357d8b65c577ab7f642.jpg)

### Tables

![10832fc9576eae4cfac163cbe3e377ea4bd0d7fbe76ad4a9a03f561c7643a928.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/10832fc9576eae4cfac163cbe3e377ea4bd0d7fbe76ad4a9a03f561c7643a928.jpg)

![11c8eb66ffa0c5998663e6a8d51c85863729a69a5a611df77c7162d75f0138cf.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/11c8eb66ffa0c5998663e6a8d51c85863729a69a5a611df77c7162d75f0138cf.jpg)

![1681dec7e8b397432cd73d73c0f9e0ffd04b9a60a77e7b2d6916ade4adb6a371.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/1681dec7e8b397432cd73d73c0f9e0ffd04b9a60a77e7b2d6916ade4adb6a371.jpg)

![22fa49045a6cbd03f25a7c502de5a273300aa8a717e46dd996e475c95f19b167.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/22fa49045a6cbd03f25a7c502de5a273300aa8a717e46dd996e475c95f19b167.jpg)

![232be6dc6013dfa83042d01cf5ce763f920f74fd1c595e0b6d65637d51f3701d.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/232be6dc6013dfa83042d01cf5ce763f920f74fd1c595e0b6d65637d51f3701d.jpg)

![3b241906476d624b17aca6f9e7ac3f045639162b5ba4c1e749115ce696181f4a.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/3b241906476d624b17aca6f9e7ac3f045639162b5ba4c1e749115ce696181f4a.jpg)

![62e7759ccf78d53b021e5ca7551141e04f2687f1e940b5e0e20224d855216764.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/62e7759ccf78d53b021e5ca7551141e04f2687f1e940b5e0e20224d855216764.jpg)

![72f6ad066536e0d6d24cacecde48e00a28409aaff7b55a0014768034ff45bcd4.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/72f6ad066536e0d6d24cacecde48e00a28409aaff7b55a0014768034ff45bcd4.jpg)

![7612fd5288408de70335d71cd488e73a0ac95e87ef56abf72daf4d5723145679.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/7612fd5288408de70335d71cd488e73a0ac95e87ef56abf72daf4d5723145679.jpg)

![802297eaedf98b118ad7a4d2b17e4b36a6ca38763d433386656fe82478635d91.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/802297eaedf98b118ad7a4d2b17e4b36a6ca38763d433386656fe82478635d91.jpg)

![9f5756bf63b3d3f5c8ed7f575c67eae5fdd7f0c58bc16139cf0ec224e7a1a843.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/9f5756bf63b3d3f5c8ed7f575c67eae5fdd7f0c58bc16139cf0ec224e7a1a843.jpg)

![a49c4cb4fb0dca770246edd44df2728e18c8c798976d05420c3bd6cabec003fe.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/a49c4cb4fb0dca770246edd44df2728e18c8c798976d05420c3bd6cabec003fe.jpg)

![b0aba6d5e989e42818554224f6a431344c1f2a5c91b960810d76317a0d7a9bbc.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/b0aba6d5e989e42818554224f6a431344c1f2a5c91b960810d76317a0d7a9bbc.jpg)

![b3c18ae4495ec66178ca40ae55aa46e44261e39e4076240e4ccf90668fa0055c.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/b3c18ae4495ec66178ca40ae55aa46e44261e39e4076240e4ccf90668fa0055c.jpg)

![c4e99c6af826980fec614c400857c35d88a5d28a79c40dbe55bbd291db9069d1.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/c4e99c6af826980fec614c400857c35d88a5d28a79c40dbe55bbd291db9069d1.jpg)

![c90de00426b5b7538558bb22e0fc8d5377d0c0724217ef260921d59d5a000ba1.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/c90de00426b5b7538558bb22e0fc8d5377d0c0724217ef260921d59d5a000ba1.jpg)

![dfdb6ab4913fa7be837a957d7c0073db8ef6865b02446a52459ac70d19534e80.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/dfdb6ab4913fa7be837a957d7c0073db8ef6865b02446a52459ac70d19534e80.jpg)

![e3ed260e02947684bc2a1d5de95f58740af8b21507d2224c74c08f752cf6706e.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/e3ed260e02947684bc2a1d5de95f58740af8b21507d2224c74c08f752cf6706e.jpg)

![f1b781cf26c8a5604b53d7c8a1b296a8a47617589c41ce5b819d505cc063ac79.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/f1b781cf26c8a5604b53d7c8a1b296a8a47617589c41ce5b819d505cc063ac79.jpg)

![f1f65ed9bcaee3c033a6273efb2d749c8829e6b15a220cf87c5ffdbad0123610.jpg](../icml_results/5544_Prompt%20Reinjection_%20Alleviating%20Prompt%20Forgetting%20in%20Multimodal%20Diffusion%20Transformers/tables/f1f65ed9bcaee3c033a6273efb2d749c8829e6b15a220cf87c5ffdbad0123610.jpg)

<a id="paper-5497"></a>

## ParamMem: Augmenting Language Agents with Parametric Reflective Memory


### Images

![0813e378fc0dd2e64bae2dc20f567a53598e48694d7f8c63ef9cf1f32f92c903.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/0813e378fc0dd2e64bae2dc20f567a53598e48694d7f8c63ef9cf1f32f92c903.jpg)

![5c4fe24acd24a299531e13943ad8d14b341a0fd584f9e63d8b809830a0d885de.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/5c4fe24acd24a299531e13943ad8d14b341a0fd584f9e63d8b809830a0d885de.jpg)

![5f38680f644ec32796ec604c63ba00fd963bec9501eb8517eb7244a61cb942c4.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/5f38680f644ec32796ec604c63ba00fd963bec9501eb8517eb7244a61cb942c4.jpg)

![729d589e6b3d4268a9519e4949dbd6a586b69c49fbfd2ab19e2d55dc9f4a879a.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/729d589e6b3d4268a9519e4949dbd6a586b69c49fbfd2ab19e2d55dc9f4a879a.jpg)

![e7955b3d5e0b1584291a37a74c198cc941f816c6d6dfdeb0b4ef2ed51063efbe.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/e7955b3d5e0b1584291a37a74c198cc941f816c6d6dfdeb0b4ef2ed51063efbe.jpg)

![ebc89da7defd4079ecec4ebba04cda59f0837c6fc557716bb2c25694a2599822.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/ebc89da7defd4079ecec4ebba04cda59f0837c6fc557716bb2c25694a2599822.jpg)

![f453e22700ef0ea0ee2a2ded17acae02de6c10e36032fef4f262a5e7e89101be.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/f453e22700ef0ea0ee2a2ded17acae02de6c10e36032fef4f262a5e7e89101be.jpg)

![fe02660718b3ad3bb4ba6b6d421e08403e207277f5874d67e226521a6969b033.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/images/fe02660718b3ad3bb4ba6b6d421e08403e207277f5874d67e226521a6969b033.jpg)

### Tables

![26dcb964c5e01f22827feeb5eaf4ca9003fd23645281c20a0002a41abe297b2d.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/26dcb964c5e01f22827feeb5eaf4ca9003fd23645281c20a0002a41abe297b2d.jpg)

![2a67ff42440f37a7868b7d12a78593f596858e441f6eea1fd2688f0c8b5cd482.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/2a67ff42440f37a7868b7d12a78593f596858e441f6eea1fd2688f0c8b5cd482.jpg)

![366ed69ba7675877504b2934cdcdce6d5631bf03cd71119e3d8cd01d5c4c1da9.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/366ed69ba7675877504b2934cdcdce6d5631bf03cd71119e3d8cd01d5c4c1da9.jpg)

![3fae43951ddf520f0a853262fda8f2d95d9c3e216418651602e909d81a256b12.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/3fae43951ddf520f0a853262fda8f2d95d9c3e216418651602e909d81a256b12.jpg)

![4d292c676fe205f1cf9408f9b3774b76802afbb890bd0889017df5937d2036db.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/4d292c676fe205f1cf9408f9b3774b76802afbb890bd0889017df5937d2036db.jpg)

![62041c792419bf0affae4fc5802d43356bc92b0569b66d5ba6af8582cff1a022.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/62041c792419bf0affae4fc5802d43356bc92b0569b66d5ba6af8582cff1a022.jpg)

![ba96ecd34e3717bf6acf2e260ecd2a4cad096f5017aa87721fc912a85ee7afdc.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/ba96ecd34e3717bf6acf2e260ecd2a4cad096f5017aa87721fc912a85ee7afdc.jpg)

![ecf4ada42bbc33ebeb8c1e021602955e9c5b6881ec4f7617df1cdc3622ff46d4.jpg](../icml_results/5545_ParamMem_%20Augmenting%20Language%20Agents%20with%20Parametric%20Reflective%20Memory/tables/ecf4ada42bbc33ebeb8c1e021602955e9c5b6881ec4f7617df1cdc3622ff46d4.jpg)

<a id="paper-5498"></a>

## Beyond Looking Up, Try Looking Around: Harmonizing Global Structure and Local Consistency in Optimal Transport for Short Text Clustering


### Images

![04fe41ef48e5d2e480407ded75afe00af1cfb12c4f9281088421002909995258.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/04fe41ef48e5d2e480407ded75afe00af1cfb12c4f9281088421002909995258.jpg)

![0ae4a250882b19ec8556328a9e6dfdc26cc1046d2b548d6f57b5826974168894.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/0ae4a250882b19ec8556328a9e6dfdc26cc1046d2b548d6f57b5826974168894.jpg)

![10defa4b14ae880bf81d0d3994d3e31bc8bae278a730c962e7c4263051604b40.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/10defa4b14ae880bf81d0d3994d3e31bc8bae278a730c962e7c4263051604b40.jpg)

![915d5e13c2cfb44895b5d0c1e42b60f9c8026aaa52da349f6eb9066cc4b2c5c5.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/915d5e13c2cfb44895b5d0c1e42b60f9c8026aaa52da349f6eb9066cc4b2c5c5.jpg)

![ab72ff08d2f3ae31316c6391871b00aab7fa20b4fc519adf19ff00d43b9bb142.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/ab72ff08d2f3ae31316c6391871b00aab7fa20b4fc519adf19ff00d43b9bb142.jpg)

![ec7b6f96d47d9c330d613ecb2eddd707ee34bc02b54ad0d6c767321affc2ce29.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/ec7b6f96d47d9c330d613ecb2eddd707ee34bc02b54ad0d6c767321affc2ce29.jpg)

![f602981fc019f4232a82861301484c86fdab94e4aad8391f075eddc8b13112cf.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/images/f602981fc019f4232a82861301484c86fdab94e4aad8391f075eddc8b13112cf.jpg)

### Tables

![03eda7f12cbb945fad1f9a19c1ef11ddcc82624cbaff24f078062bad2110cd86.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/03eda7f12cbb945fad1f9a19c1ef11ddcc82624cbaff24f078062bad2110cd86.jpg)

![07a8fe81f5646ab68318556b058213b2eb0bbfc830cc06a9c65affae471a0af5.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/07a8fe81f5646ab68318556b058213b2eb0bbfc830cc06a9c65affae471a0af5.jpg)

![15eda56073de2bba6c187a7c8fe0ca494439586ed02b92188bb9ee1b4385fa92.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/15eda56073de2bba6c187a7c8fe0ca494439586ed02b92188bb9ee1b4385fa92.jpg)

![1a85bd67b6a2ae10edeabfb13ab3b4ace4695e9b73237558c1b311b75af0eec3.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/1a85bd67b6a2ae10edeabfb13ab3b4ace4695e9b73237558c1b311b75af0eec3.jpg)

![2913d1c6390be67c391d6c08dd897903e172c6395f44f65b87ab9e548609a0fd.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/2913d1c6390be67c391d6c08dd897903e172c6395f44f65b87ab9e548609a0fd.jpg)

![5b035401dd385e04c7f3b1d43d183f72109351ff2f14c73dc73f52cffb4feeac.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/5b035401dd385e04c7f3b1d43d183f72109351ff2f14c73dc73f52cffb4feeac.jpg)

![df82dc49975970d928757c83c708641bb25254e93cddb257098d99cc56b25f5b.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/df82dc49975970d928757c83c708641bb25254e93cddb257098d99cc56b25f5b.jpg)

![e3c23a708d1ea69db2eee015152363ac9d29b9399a87c754c65a060601389b67.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/e3c23a708d1ea69db2eee015152363ac9d29b9399a87c754c65a060601389b67.jpg)

![f2b49334144c00f08d1427c93008f924fefeb53930541a0650d6707e44638e60.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/f2b49334144c00f08d1427c93008f924fefeb53930541a0650d6707e44638e60.jpg)

![fb3cec88f90333eb82888c5f3c7127197bffdf65633922e43562ee2356e48ad2.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/fb3cec88f90333eb82888c5f3c7127197bffdf65633922e43562ee2356e48ad2.jpg)

![fc1b53ef72450192665233f70bb2ee50078405832c214976814b4f335e8fb368.jpg](../icml_results/5546_Beyond%20Looking%20Up%2C%20Try%20Looking%20Around_%20Harmonizing%20Global%20Structure%20and%20Local%20Consistency%20in%20Optimal/tables/fc1b53ef72450192665233f70bb2ee50078405832c214976814b4f335e8fb368.jpg)

<a id="paper-5499"></a>

## H$^2$CL: Heterogeneity-Aware Hypergraph Contrastive Learning for Robust Representation Learning


### Images

![1029c4a7e98ae67322c4d90c3fa9071568036f1d9fcc5a53cd61d913114e7498.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/images/1029c4a7e98ae67322c4d90c3fa9071568036f1d9fcc5a53cd61d913114e7498.jpg)

![8593ded777011267246421f71a1d88ca309d6e3d6e2696d4008e67cb4f28aa0c.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/images/8593ded777011267246421f71a1d88ca309d6e3d6e2696d4008e67cb4f28aa0c.jpg)

![aca0a9ecb7a10a1b3b5f61b75f4333fe0f92f4af9356020e493de1c82eacbb0c.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/images/aca0a9ecb7a10a1b3b5f61b75f4333fe0f92f4af9356020e493de1c82eacbb0c.jpg)

![f307fd8a9c7b4c5d734c9e9fc7bddb4b08a9ff8bc8721e744f54ba7e4c8da82e.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/images/f307fd8a9c7b4c5d734c9e9fc7bddb4b08a9ff8bc8721e744f54ba7e4c8da82e.jpg)

### Tables

![0056773d699593b2e6eeae4d9a0441b37b4e245214e522b90f56314490ecc470.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/0056773d699593b2e6eeae4d9a0441b37b4e245214e522b90f56314490ecc470.jpg)

![0ef6be66dd789cfd71c57ef4bcb9357a271e3c0ac76ed8c344f8ec142ece093a.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/0ef6be66dd789cfd71c57ef4bcb9357a271e3c0ac76ed8c344f8ec142ece093a.jpg)

![3916a7bea894435abad30af1942cfe9dae0b20c08d28e751685287cd5ebb5f61.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/3916a7bea894435abad30af1942cfe9dae0b20c08d28e751685287cd5ebb5f61.jpg)

![4581e0ba0396e4f58cbcb59ed5a7a39ce9ba5a68e2fe1ef482424711c1a02980.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/4581e0ba0396e4f58cbcb59ed5a7a39ce9ba5a68e2fe1ef482424711c1a02980.jpg)

![4dd4da3a9a72804868cbf32f938c435bbd3ea07574fa1916688af9594bdda453.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/4dd4da3a9a72804868cbf32f938c435bbd3ea07574fa1916688af9594bdda453.jpg)

![a275ccdfa41dfaa20fcbd28fbbc472202324c2b3521e84b8e9a592c6e51c0a6d.jpg](../icml_results/5547_H%24%5E2%24CL_%20Heterogeneity-Aware%20Hypergraph%20Contrastive%20Learning%20for%20Robust%20Representation%20Learning/tables/a275ccdfa41dfaa20fcbd28fbbc472202324c2b3521e84b8e9a592c6e51c0a6d.jpg)

<a id="paper-5500"></a>

## Integrated Episodic and Semantic Memory via Modulating Transformer FeedForward Layers


### Images

![4c4c9cb4f3ca93cabb03571a8ab50efa13ab4c1c43d6122d9f5a78f98b186350.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/images/4c4c9cb4f3ca93cabb03571a8ab50efa13ab4c1c43d6122d9f5a78f98b186350.jpg)

![50bb25d2e19af347ebf81b5d8cb06e755d5758792c85ab7b490bb715341411a8.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/images/50bb25d2e19af347ebf81b5d8cb06e755d5758792c85ab7b490bb715341411a8.jpg)

![531c9b1df5a3f59f80dcffc29c37cea31d93353a4fd815f32198e75d0ac84b00.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/images/531c9b1df5a3f59f80dcffc29c37cea31d93353a4fd815f32198e75d0ac84b00.jpg)

![8b505b71363e113ca6545357f33a552dbf7fd2759a8de09c6d9d7c47c8603a47.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/images/8b505b71363e113ca6545357f33a552dbf7fd2759a8de09c6d9d7c47c8603a47.jpg)

### Tables

![10290cf737d8b8b36055415c92f11536c69e68400f2da99778fcd85f3b87b3d6.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/tables/10290cf737d8b8b36055415c92f11536c69e68400f2da99778fcd85f3b87b3d6.jpg)

![242d2422148556e04c1f37a3b5a7d70d2478067c04ec8a70f46643fbc4896d9c.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/tables/242d2422148556e04c1f37a3b5a7d70d2478067c04ec8a70f46643fbc4896d9c.jpg)

![4182a6cf7e2785c5d7e86bca02de6915cd3aa7a0f194c5e502ca1cfdaeecd40a.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/tables/4182a6cf7e2785c5d7e86bca02de6915cd3aa7a0f194c5e502ca1cfdaeecd40a.jpg)

![93e9e917e1dbababb180f6ca7b38fc723684c1d236dc02099bc8844f1497f63a.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/tables/93e9e917e1dbababb180f6ca7b38fc723684c1d236dc02099bc8844f1497f63a.jpg)

![afbb9582f3fa1015e7ae8e340000af4a85c2a85db0f672e8a1b7ff9ae29ff83e.jpg](../icml_results/5549_Integrated%20Episodic%20and%20Semantic%20Memory%20via%20Modulating%20Transformer%20FeedForward%20Layers/tables/afbb9582f3fa1015e7ae8e340000af4a85c2a85db0f672e8a1b7ff9ae29ff83e.jpg)

<a id="paper-5501"></a>

## HieraMAS: Optimizing Intra-Node LLM Mixtures and Inter-Node Topology for Multi-Agent Systems


### Images

![62ef919370a191c595535ab0585c66bc31bbb2aeacb8fd3391c9c76bd13cf154.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/62ef919370a191c595535ab0585c66bc31bbb2aeacb8fd3391c9c76bd13cf154.jpg)

![67df2852c24788764bc38736d124ae42423121a80df931694b0b3c1690179c9f.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/67df2852c24788764bc38736d124ae42423121a80df931694b0b3c1690179c9f.jpg)

![6d2e7259c2fc8656a8c0a6f2d97a33103af3249979976cf2cef5fe55d3b4cfea.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/6d2e7259c2fc8656a8c0a6f2d97a33103af3249979976cf2cef5fe55d3b4cfea.jpg)

![885e6b254679469f14ca9e67484b2cd081b363b9c5c9468e8aa141db0712ac58.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/885e6b254679469f14ca9e67484b2cd081b363b9c5c9468e8aa141db0712ac58.jpg)

![9a6a5c8b22355a8464636ba6c166cb4415db05878f7a993698054eaf4e0eac29.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/9a6a5c8b22355a8464636ba6c166cb4415db05878f7a993698054eaf4e0eac29.jpg)

![b224542a578917b1acde976ed527afd82426cbf3f8539324dfedd9c8601e696b.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/images/b224542a578917b1acde976ed527afd82426cbf3f8539324dfedd9c8601e696b.jpg)

### Tables

![19c9b45bd2c5c05fb0dacfca6cc06d8d6730d7d225ec29cc82c814859c2690d9.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/19c9b45bd2c5c05fb0dacfca6cc06d8d6730d7d225ec29cc82c814859c2690d9.jpg)

![1edf26503e0cfadeaab5f256efe9a2ae2970eca730ab7723f4c49b588ec3dde6.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/1edf26503e0cfadeaab5f256efe9a2ae2970eca730ab7723f4c49b588ec3dde6.jpg)

![2486f90368dc857f14969cb9348e202b7201a9d9605a4eb38aad7f7f5d2bb8f3.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/2486f90368dc857f14969cb9348e202b7201a9d9605a4eb38aad7f7f5d2bb8f3.jpg)

![254f704121f185ba3549494faea07a65cfaf6c95ad57f48adc2b269d7b18ca15.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/254f704121f185ba3549494faea07a65cfaf6c95ad57f48adc2b269d7b18ca15.jpg)

![478e2e3e404c05a41710716160c07525ae2453da24a8bedc2d8b50c165aae3de.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/478e2e3e404c05a41710716160c07525ae2453da24a8bedc2d8b50c165aae3de.jpg)

![48ee014d364b5c0f58e9e48e12e47794e8274acca5d054e2d6552fd51ae43aaf.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/48ee014d364b5c0f58e9e48e12e47794e8274acca5d054e2d6552fd51ae43aaf.jpg)

![695a4c00e42e46ceb3b162b698fa257b79e37168762bef88710624cbaf7e61ed.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/695a4c00e42e46ceb3b162b698fa257b79e37168762bef88710624cbaf7e61ed.jpg)

![70792ad1c186cbeea2be24259b2cd27ab51fb0a262f5fc2c2a6f6444a70a9c7e.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/70792ad1c186cbeea2be24259b2cd27ab51fb0a262f5fc2c2a6f6444a70a9c7e.jpg)

![7ad3676b90ee4faec18816ca8b671bf93aceb7d129b95c493d239f15cd5e6426.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/7ad3676b90ee4faec18816ca8b671bf93aceb7d129b95c493d239f15cd5e6426.jpg)

![8b7a9c748a5b42154e12af5d9ed17c525d7f78f1653789f2847d943f9eebca8f.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/8b7a9c748a5b42154e12af5d9ed17c525d7f78f1653789f2847d943f9eebca8f.jpg)

![c44081e83d02d8915dc16a1ea0bcb5638ec2ebb7ba322648addb969e812ae0d8.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/c44081e83d02d8915dc16a1ea0bcb5638ec2ebb7ba322648addb969e812ae0d8.jpg)

![f93add1a586379d74d4e001d633c108254f0b7d4d36aedbb852d3ddce5fe756f.jpg](../icml_results/5550_HieraMAS_%20Optimizing%20Intra-Node%20LLM%20Mixtures%20and%20Inter-Node%20Topology%20for%20Multi-Agent%20Systems/tables/f93add1a586379d74d4e001d633c108254f0b7d4d36aedbb852d3ddce5fe756f.jpg)

<a id="paper-5502"></a>

## Role-Level Inductive Bias for Cross-Task Generalization in Multi-Agent Reinforcement Learning


### Images

![1340c2d551077cf5b208509fa4ebebf81be40e858391704d3a55eab07fb3252d.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/1340c2d551077cf5b208509fa4ebebf81be40e858391704d3a55eab07fb3252d.jpg)

![28cc8a2c96a4576cf41f8b4ab78641b6a311ca8fb6e43ae6430f4253bb7138f1.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/28cc8a2c96a4576cf41f8b4ab78641b6a311ca8fb6e43ae6430f4253bb7138f1.jpg)

![66b88f289884b44897b6b9725a03bf4e2f4cb72d6724d2e3ba453732c1489e7f.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/66b88f289884b44897b6b9725a03bf4e2f4cb72d6724d2e3ba453732c1489e7f.jpg)

![6960b49c4aff2ea339d3e62987b15abaedd7090ce658d5b090e7cf7ce08ff7e5.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/6960b49c4aff2ea339d3e62987b15abaedd7090ce658d5b090e7cf7ce08ff7e5.jpg)

![7689d117f040640585464c4be896543e097da5511340950e1f21fb8ecd6654f7.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/7689d117f040640585464c4be896543e097da5511340950e1f21fb8ecd6654f7.jpg)

![c0107e503faf03a85dc5de17ec6f99484d52af1ef16ee6b730b19374a3503347.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/c0107e503faf03a85dc5de17ec6f99484d52af1ef16ee6b730b19374a3503347.jpg)

![dc581bd4b5b5b2284ff333c4bd23ce700d6ebdc523c6ffaec1bcef108cc183d9.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/dc581bd4b5b5b2284ff333c4bd23ce700d6ebdc523c6ffaec1bcef108cc183d9.jpg)

![e0a618e0564f2d323e69789d9cdec962cbb09483e016dc552b201559c00f64a9.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/e0a618e0564f2d323e69789d9cdec962cbb09483e016dc552b201559c00f64a9.jpg)

![f3aee374a994638a1bfd55d03d1e8584f280d582332878cbbc63aeda8602bb17.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/images/f3aee374a994638a1bfd55d03d1e8584f280d582332878cbbc63aeda8602bb17.jpg)

### Tables

![1cd3d60f914af103c7c004f431cca9fc10ea1daab286861e1f3fd63794fc01d3.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/tables/1cd3d60f914af103c7c004f431cca9fc10ea1daab286861e1f3fd63794fc01d3.jpg)

![3a8cb4b857020d3b2d902c827bf28f743e4fac59c3063c2cb5b6e148976636eb.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/tables/3a8cb4b857020d3b2d902c827bf28f743e4fac59c3063c2cb5b6e148976636eb.jpg)

![68aeece3de49459829e788be8449c3e59d02a32dc074c3a2268ab0fc22cebe38.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/tables/68aeece3de49459829e788be8449c3e59d02a32dc074c3a2268ab0fc22cebe38.jpg)

![6a5231c702dcf28a991eaefc60f3b2ad7b2e9d090048d6536e1125cbf30c8040.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/tables/6a5231c702dcf28a991eaefc60f3b2ad7b2e9d090048d6536e1125cbf30c8040.jpg)

![71f3efa708b51577331052b6f07b8c47bc6a4a98541a3e4513012d539c5f59f0.jpg](../icml_results/5551_Role-Level%20Inductive%20Bias%20for%20Cross-Task%20Generalization%20in%20Multi-Agent%20Reinforcement%20Learning/tables/71f3efa708b51577331052b6f07b8c47bc6a4a98541a3e4513012d539c5f59f0.jpg)

<a id="paper-5503"></a>

## SLAT: Segment-Level Adaptive Trimming for Efficient CoT Reasoning


### Images

![1df5e8d312625feff1d4dd5213e47bb223b5201e7756da624e8b6f7dd4ee7fe9.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/images/1df5e8d312625feff1d4dd5213e47bb223b5201e7756da624e8b6f7dd4ee7fe9.jpg)

![4203c300fb799cf3ab87c3bf86ed14b86ccee5e20d790ada204bed424bd93443.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/images/4203c300fb799cf3ab87c3bf86ed14b86ccee5e20d790ada204bed424bd93443.jpg)

![9f45d0a11e67c4812a1b31f79fc00905416f24f84c846156e2e5745d66a3d4cd.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/images/9f45d0a11e67c4812a1b31f79fc00905416f24f84c846156e2e5745d66a3d4cd.jpg)

![9f6e1a7f2924e40600ccb1b91b606038d83f107028c18c1a13fc56a3613bf9e4.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/images/9f6e1a7f2924e40600ccb1b91b606038d83f107028c18c1a13fc56a3613bf9e4.jpg)

![e09e8bf55f96abc1412ac9a800e8beb419f5e3baa401145dcea2f997bdd5ca30.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/images/e09e8bf55f96abc1412ac9a800e8beb419f5e3baa401145dcea2f997bdd5ca30.jpg)

### Tables

![0b71de6c1c45a2c0dc6651b5499f6dacbadb8e3d5bccc8b933a31431d087a564.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/0b71de6c1c45a2c0dc6651b5499f6dacbadb8e3d5bccc8b933a31431d087a564.jpg)

![78cbadf6858a5a887e51f4c57fcdab49538c923131f51a6624bb203d0515bfca.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/78cbadf6858a5a887e51f4c57fcdab49538c923131f51a6624bb203d0515bfca.jpg)

![8b816eae5d106903b8f43b718fa4d04b1aba865ef6464a80febfd4fc26dbface.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/8b816eae5d106903b8f43b718fa4d04b1aba865ef6464a80febfd4fc26dbface.jpg)

![aab28718bce41f96967719d21dbc9fda2c767a227d90ef8769a506f081334d62.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/aab28718bce41f96967719d21dbc9fda2c767a227d90ef8769a506f081334d62.jpg)

![da3b60076b2da65adb4bd919d69e5073264504b1fc2664d2cb1088aecf712ab7.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/da3b60076b2da65adb4bd919d69e5073264504b1fc2664d2cb1088aecf712ab7.jpg)

![f18ad91889079e1fe29275c2345b57d7b724c4d68c012f264cea03ae1273a469.jpg](../icml_results/5552_SLAT_%20Segment-Level%20Adaptive%20Trimming%20for%20Efficient%20CoT%20Reasoning/tables/f18ad91889079e1fe29275c2345b57d7b724c4d68c012f264cea03ae1273a469.jpg)

<a id="paper-5504"></a>

## The Perception–Physics Paradox: Probing Scientific Alignment with TC-Bench


### Images

![159440657347b33384a6f92f32e10ed5ed793bf4baa69cc9f9501e924f95cc86.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/159440657347b33384a6f92f32e10ed5ed793bf4baa69cc9f9501e924f95cc86.jpg)

![1617985e7c9510a36acc8c437242fb22da48932e318b166f814ba0285753a623.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/1617985e7c9510a36acc8c437242fb22da48932e318b166f814ba0285753a623.jpg)

![30f55403aaa4cc5ab46448e1a0849e3f25b151a16bb07e6960c92ae8db9398ae.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/30f55403aaa4cc5ab46448e1a0849e3f25b151a16bb07e6960c92ae8db9398ae.jpg)

![33707ecb1ff1dd10b09bebb9c7b4b4a93936099c63c4b28981f7cb2c355486de.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/33707ecb1ff1dd10b09bebb9c7b4b4a93936099c63c4b28981f7cb2c355486de.jpg)

![4c7ce81dc985bb5a7bce94521c0837f5dae3b132edd90dcf64e2d78ca98abd44.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/4c7ce81dc985bb5a7bce94521c0837f5dae3b132edd90dcf64e2d78ca98abd44.jpg)

![6892a6737c998db8079cceb4242213e06eb6251601f7d202dd3e35a92a52c35b.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/6892a6737c998db8079cceb4242213e06eb6251601f7d202dd3e35a92a52c35b.jpg)

![73f5bbcca7caf92ebf03ac08bb107d9cfdeafe1d1fea36cbd7a06469a28eed38.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/73f5bbcca7caf92ebf03ac08bb107d9cfdeafe1d1fea36cbd7a06469a28eed38.jpg)

![7e427f06e844f1b35dc0b7147d25bdb0e2cbd96fa5720bd1f85efaa50df13c23.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/7e427f06e844f1b35dc0b7147d25bdb0e2cbd96fa5720bd1f85efaa50df13c23.jpg)

![948675a1806c78ba27955f781c39a9ac4e067be6b38624f17e0187d0b8e0e3e1.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/948675a1806c78ba27955f781c39a9ac4e067be6b38624f17e0187d0b8e0e3e1.jpg)

![982f32865ee47bde5cf63a9db7aed40b608f5c280ca19f9c4182bac757d808ef.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/982f32865ee47bde5cf63a9db7aed40b608f5c280ca19f9c4182bac757d808ef.jpg)

![a27f03b253e480faaa8d49eb56dff5d7fae73182c9fee3bf0a4aa70997c55fff.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/a27f03b253e480faaa8d49eb56dff5d7fae73182c9fee3bf0a4aa70997c55fff.jpg)

![a741264ca9513d9a36004960e88b5bd9478d098d116b8d1a65a5c63fcde56a41.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/a741264ca9513d9a36004960e88b5bd9478d098d116b8d1a65a5c63fcde56a41.jpg)

![aa60c01f01443f33724717fe1b4ef940532dd09fc7005d08aa07b75620749b59.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/aa60c01f01443f33724717fe1b4ef940532dd09fc7005d08aa07b75620749b59.jpg)

![b24f0836513c1d4f19af24054a5e42380912ff0d3787212f63a21cbd22102bc1.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/b24f0836513c1d4f19af24054a5e42380912ff0d3787212f63a21cbd22102bc1.jpg)

![c25246d0afe4c9c42e0ffb87d461fde2be8cee7f9815245b0776648f64e6936c.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/c25246d0afe4c9c42e0ffb87d461fde2be8cee7f9815245b0776648f64e6936c.jpg)

![c876b4d6c3a1a47d509cb36ff106b1fd160645b2e576db5704ad7b3e1fba5a71.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/c876b4d6c3a1a47d509cb36ff106b1fd160645b2e576db5704ad7b3e1fba5a71.jpg)

![e554ce49369ea0f11646a1ea1c549b15db341dee03155adf9c30ee30d3c4e14a.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/e554ce49369ea0f11646a1ea1c549b15db341dee03155adf9c30ee30d3c4e14a.jpg)

![f36e763691ed6786900e0d77656258f03f9d483deb05d548911797eb39e65855.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/f36e763691ed6786900e0d77656258f03f9d483deb05d548911797eb39e65855.jpg)

![f5a9b4a74a772286c78151faaf9210067535c51a433727cb0a342f85279a54d2.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/f5a9b4a74a772286c78151faaf9210067535c51a433727cb0a342f85279a54d2.jpg)

![fe1df3cb439da7c57b1c7c1882136f4f476ac686b4382df7e40a7afc5019e11a.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/fe1df3cb439da7c57b1c7c1882136f4f476ac686b4382df7e40a7afc5019e11a.jpg)

![ff1c048fd2d4ffe7986a4e21b42731525e74de8f90bd087f9f7b8694db8f7d8e.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/images/ff1c048fd2d4ffe7986a4e21b42731525e74de8f90bd087f9f7b8694db8f7d8e.jpg)

### Tables

![14816cdc6309284e7534e1c5bf69ff92fd397ff6fec5f15ba3d8cb9a1345ab3e.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/tables/14816cdc6309284e7534e1c5bf69ff92fd397ff6fec5f15ba3d8cb9a1345ab3e.jpg)

![831bc0df4e918108b953ab165d0c55de505406c94bf126f5886b91cf2c58810b.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/tables/831bc0df4e918108b953ab165d0c55de505406c94bf126f5886b91cf2c58810b.jpg)

![a8b5d4974b22f5f4e2c3fa25a65cb91c59848af602a6eebb6b3c3a6c082ab5cb.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/tables/a8b5d4974b22f5f4e2c3fa25a65cb91c59848af602a6eebb6b3c3a6c082ab5cb.jpg)

![b626bcc9302453218c8c1a2b2fc5378e89f8772d7d18c0171b77e1004a904948.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/tables/b626bcc9302453218c8c1a2b2fc5378e89f8772d7d18c0171b77e1004a904948.jpg)

![faf3cacc0273f9e1a8a39de466435c0736e200dc22105b054d2809ebfdce1739.jpg](../icml_results/5553_The%20Perception%E2%80%93Physics%20Paradox_%20Probing%20Scientific%20Alignment%20with%20TC-Bench/tables/faf3cacc0273f9e1a8a39de466435c0736e200dc22105b054d2809ebfdce1739.jpg)

<a id="paper-5505"></a>

## MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery


### Images

![0638ea9a1b933cc6acdd6b3e4ab898ae7ba610ff4071006cbee836066e2083f7.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/images/0638ea9a1b933cc6acdd6b3e4ab898ae7ba610ff4071006cbee836066e2083f7.jpg)

![6740b58226507694f559aa9dd2ec67658e3d33937276c8eb7d7f6078cce63ec0.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/images/6740b58226507694f559aa9dd2ec67658e3d33937276c8eb7d7f6078cce63ec0.jpg)

![98f0dcb15745a431e1997b0b9e5611b606c69f0638e96bda6199cc7133207855.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/images/98f0dcb15745a431e1997b0b9e5611b606c69f0638e96bda6199cc7133207855.jpg)

![d93a53f1b46d86d891ff2af602088041eedbecb3acde9d7b9321c8cf82cbae06.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/images/d93a53f1b46d86d891ff2af602088041eedbecb3acde9d7b9321c8cf82cbae06.jpg)

![dd6a1d84608992088ef9178e409baa6f5c18507900394e359bad97d05e307808.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/images/dd6a1d84608992088ef9178e409baa6f5c18507900394e359bad97d05e307808.jpg)

### Tables

![0dda96c3a79c8851c395f7dbf3bd87219241f3deb8225d39e642764bffcc2fc2.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/0dda96c3a79c8851c395f7dbf3bd87219241f3deb8225d39e642764bffcc2fc2.jpg)

![24e3f220fd169a95be29acd5a34705bc651dec534529cd7a4ef33d45c4b90aba.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/24e3f220fd169a95be29acd5a34705bc651dec534529cd7a4ef33d45c4b90aba.jpg)

![833010a7f443d49532237d888bb610488bc1a83d609dc52421f75afe9f4dd29b.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/833010a7f443d49532237d888bb610488bc1a83d609dc52421f75afe9f4dd29b.jpg)

![ca5419b13a071054ef7cd5a891b828389157a4b82d6aa7bb6c2dde303cffb711.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/ca5419b13a071054ef7cd5a891b828389157a4b82d6aa7bb6c2dde303cffb711.jpg)

![edfd264fd5d009baa3e87db2195d66bb00fc5168c0e7ca18d95a71a72a92981f.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/edfd264fd5d009baa3e87db2195d66bb00fc5168c0e7ca18d95a71a72a92981f.jpg)

![fe9aad054ef183dbb2a375899d779a84c0cf368cd4ccd1cd1924e1ae80d5a288.jpg](../icml_results/5554_MotionGRPO_%20Overcoming%20Low%20Intra-Group%20Diversity%20in%20GRPO-Based%20Egocentric%20Motion%20Recovery/tables/fe9aad054ef183dbb2a375899d779a84c0cf368cd4ccd1cd1924e1ae80d5a288.jpg)

<a id="paper-5506"></a>

## Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain


### Images

![07ae9138c3c15a8f6e8e2cf661c376355febfb96cdc20e3a23c45a44fda81420.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/images/07ae9138c3c15a8f6e8e2cf661c376355febfb96cdc20e3a23c45a44fda81420.jpg)

![a6d2b97924b5885dd4547e51971b497d2c52b71655277b220154e5e4de1f9729.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/images/a6d2b97924b5885dd4547e51971b497d2c52b71655277b220154e5e4de1f9729.jpg)

![a85496b63a6f29a1115fa8e4131347582cbfd6d8e31a46a33bba03d610c88ed3.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/images/a85496b63a6f29a1115fa8e4131347582cbfd6d8e31a46a33bba03d610c88ed3.jpg)

![ea613db2f829556edeccbde604854a6a1782cba0e2f5969ad284aa1f05f34880.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/images/ea613db2f829556edeccbde604854a6a1782cba0e2f5969ad284aa1f05f34880.jpg)

![f6930f225916e3634f281ac8bfd8ac6a60b9f08ff14fc91367c8bb62673b4608.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/images/f6930f225916e3634f281ac8bfd8ac6a60b9f08ff14fc91367c8bb62673b4608.jpg)

### Tables

![1bfcf77a48b8eb3b3c1aa3d425b8837e702060fec80e1a0730aac6ebbc44c406.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/1bfcf77a48b8eb3b3c1aa3d425b8837e702060fec80e1a0730aac6ebbc44c406.jpg)

![50405b2e08b80a414d765a8ef675adb7d9b221ea4cf5c9fe7cf71b3493c61293.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/50405b2e08b80a414d765a8ef675adb7d9b221ea4cf5c9fe7cf71b3493c61293.jpg)

![540100b4af12c2732d4b7b18413759923b153d4b7cceb835ed893d385ebe25d9.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/540100b4af12c2732d4b7b18413759923b153d4b7cceb835ed893d385ebe25d9.jpg)

![55bdf54cc9f6fd3088fad3e6ca7d1b4772b8cf350a8fb5a830d10595ca9afaf3.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/55bdf54cc9f6fd3088fad3e6ca7d1b4772b8cf350a8fb5a830d10595ca9afaf3.jpg)

![61efc2a9c03a5f57827b46787ebcd8f9caa9cbd3427f561d69a3e7698c62e09f.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/61efc2a9c03a5f57827b46787ebcd8f9caa9cbd3427f561d69a3e7698c62e09f.jpg)

![701ff9264a9936ac3e73103c157c51d965560dd81870d38c9fb51072c84de418.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/701ff9264a9936ac3e73103c157c51d965560dd81870d38c9fb51072c84de418.jpg)

![7374d78651850bb797ef57c4822dd2a033a6489d44ab67dd50987624d75bed0f.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/7374d78651850bb797ef57c4822dd2a033a6489d44ab67dd50987624d75bed0f.jpg)

![7d93b02137406124aa164218b8d5d988eade624e63c75d915ac7095f11738413.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/7d93b02137406124aa164218b8d5d988eade624e63c75d915ac7095f11738413.jpg)

![8a2c6e98ee37b8d89c92751fd418ebab970869f7a73fca88b2a49e456864d283.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/8a2c6e98ee37b8d89c92751fd418ebab970869f7a73fca88b2a49e456864d283.jpg)

![8a400e842c020b4b8c4f3a9b3b43b1cc71a9c7da2d9f1d34dc400ab6b92080d8.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/8a400e842c020b4b8c4f3a9b3b43b1cc71a9c7da2d9f1d34dc400ab6b92080d8.jpg)

![936f9ef8d7d30ff3dd2e44a110c151f21c3d2c952972fa2a956bae00542935cd.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/936f9ef8d7d30ff3dd2e44a110c151f21c3d2c952972fa2a956bae00542935cd.jpg)

![94b7aab59cc1ebe0743ba5219086a4631b67d85e3adace5982c034e8bb6df4f8.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/94b7aab59cc1ebe0743ba5219086a4631b67d85e3adace5982c034e8bb6df4f8.jpg)

![d0c270ee77391458d2beb753078750d9d8c46f8160197726a57d31f5fc241961.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/d0c270ee77391458d2beb753078750d9d8c46f8160197726a57d31f5fc241961.jpg)

![ee5b334428fcc516e96bed3f68e20ae1b73765e8d7b24f8976637f54d672115b.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/ee5b334428fcc516e96bed3f68e20ae1b73765e8d7b24f8976637f54d672115b.jpg)

![f626358d608a9448087308fae3808f040c1626324eb867ebe90d33fad8603c1f.jpg](../icml_results/5555_Semi-Supervised%20Noise%20Adaptation_%20Transferring%20Knowledge%20from%20Noise%20Domain/tables/f626358d608a9448087308fae3808f040c1626324eb867ebe90d33fad8603c1f.jpg)

<a id="paper-5507"></a>

## Identifiable Nonlinear Differentiable Causal Discovery via Independence and Adaptive Group Sparsity


### Images

![3e9074f125df9ba8bfa4714d1f0f86856b22d6ebf3b07cef6f1d0256cb2e65dc.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/3e9074f125df9ba8bfa4714d1f0f86856b22d6ebf3b07cef6f1d0256cb2e65dc.jpg)

![6a24936e7a7d81868ca1bc4e824a8136281f389f7b3435447e73765edc629773.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/6a24936e7a7d81868ca1bc4e824a8136281f389f7b3435447e73765edc629773.jpg)

![6a7e7064e7ce9af6a31970db5a2c4c194fdd8f99046d7f32248a488718b162bf.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/6a7e7064e7ce9af6a31970db5a2c4c194fdd8f99046d7f32248a488718b162bf.jpg)

![746e8a34e7cedcda0e71bf71b18969908bedcbceead7ff363108b02f0843bc91.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/746e8a34e7cedcda0e71bf71b18969908bedcbceead7ff363108b02f0843bc91.jpg)

![8acb6b6928044fb3a9df707f4339631251f1edadf422c1b827452c8f76f4b219.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/8acb6b6928044fb3a9df707f4339631251f1edadf422c1b827452c8f76f4b219.jpg)

![bb4328c707c203a1395722b8d11fe44d8dbc4cfda1d99898a2d400818bf301d0.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/bb4328c707c203a1395722b8d11fe44d8dbc4cfda1d99898a2d400818bf301d0.jpg)

![caa7cc61477458cb98c0c2b85ef1bc3532a814ac18e4886d74807879d7188378.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/images/caa7cc61477458cb98c0c2b85ef1bc3532a814ac18e4886d74807879d7188378.jpg)

### Tables

![026293950a4271f6ee96c2aaee34dbf90f2a5ff624db8cbeaef20fcc065659aa.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/026293950a4271f6ee96c2aaee34dbf90f2a5ff624db8cbeaef20fcc065659aa.jpg)

![4cf6c488efe2d46129cf24c6907c3e0cbdf5115175815487ff41f0a27199603b.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/4cf6c488efe2d46129cf24c6907c3e0cbdf5115175815487ff41f0a27199603b.jpg)

![7b50fe2bfa715e5a16d2cfef5374ba48ab60b1b373d6a825eebf36287baffa41.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/7b50fe2bfa715e5a16d2cfef5374ba48ab60b1b373d6a825eebf36287baffa41.jpg)

![bbbf137b9f854ee0c273974f21703dc124f9de03d831551a98e7a882df144607.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/bbbf137b9f854ee0c273974f21703dc124f9de03d831551a98e7a882df144607.jpg)

![c3cf6916369701d30f5b6fb3c7e8ad5ed71ba369136bf755ea3c9920ec30880b.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/c3cf6916369701d30f5b6fb3c7e8ad5ed71ba369136bf755ea3c9920ec30880b.jpg)

![dd372f1dbddb86a99ffa13d616b834b15c38b5c2542c760063ad04a51f066a1f.jpg](../icml_results/5556_Identifiable%20Nonlinear%20Differentiable%20Causal%20Discovery%20via%20Independence%20and%20Adaptive%20Group%20Sparsity/tables/dd372f1dbddb86a99ffa13d616b834b15c38b5c2542c760063ad04a51f066a1f.jpg)

<a id="paper-5508"></a>

## TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions


### Images

![07ce5e811c2b6635dbdec0c8b284dec6cbe22e4716006739e63a211485a3db2b.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/07ce5e811c2b6635dbdec0c8b284dec6cbe22e4716006739e63a211485a3db2b.jpg)

![13e6e1ea7d5fe456da75b62cc91b2f7d17cec362f0572f91dc301cc60b678f2a.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/13e6e1ea7d5fe456da75b62cc91b2f7d17cec362f0572f91dc301cc60b678f2a.jpg)

![1b28053ef0166f13a6078739d2902b3722b46b43fc29241577206e2d7d555fbd.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/1b28053ef0166f13a6078739d2902b3722b46b43fc29241577206e2d7d555fbd.jpg)

![263c33b609e9f1a2c689ce6f1314475006b9c00c4fe16ce7f9bb1a2e210fa210.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/263c33b609e9f1a2c689ce6f1314475006b9c00c4fe16ce7f9bb1a2e210fa210.jpg)

![3e31c0308719a0766e9d90f4cccec8f5aedf9f1211b6f4b52fd02807f356005a.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/3e31c0308719a0766e9d90f4cccec8f5aedf9f1211b6f4b52fd02807f356005a.jpg)

![5f2cfde2fc742220df3377769ea33d484e17702417fe4279b47d5ec45d182bdc.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/5f2cfde2fc742220df3377769ea33d484e17702417fe4279b47d5ec45d182bdc.jpg)

![7e15dceef469f96b9c50ba70ae0069ff4753f5e322721f7749ef103cddcfb218.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/7e15dceef469f96b9c50ba70ae0069ff4753f5e322721f7749ef103cddcfb218.jpg)

![82f4b69b4dfe736597a9e4942204801d5972605573ff56055262062c1b2252de.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/82f4b69b4dfe736597a9e4942204801d5972605573ff56055262062c1b2252de.jpg)

![98e5c88b1b146ed53fb082faa1ab7ff76c393556958cb7207520bffca2c95d4b.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/98e5c88b1b146ed53fb082faa1ab7ff76c393556958cb7207520bffca2c95d4b.jpg)

![a9316a05e6cfebeebd60965b25ab9bb317170f326f7e99f39f34ffec36050b6b.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/images/a9316a05e6cfebeebd60965b25ab9bb317170f326f7e99f39f34ffec36050b6b.jpg)

### Tables

![15f697b3813b432e7a4c11f050c20f1eb60be1545db4098ad3eeffdde8f28aa6.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/15f697b3813b432e7a4c11f050c20f1eb60be1545db4098ad3eeffdde8f28aa6.jpg)

![42c55192a016657d4f0e93138bde40f190421de0ae07c6c9d38e3d191a40c0ad.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/42c55192a016657d4f0e93138bde40f190421de0ae07c6c9d38e3d191a40c0ad.jpg)

![52c3576aec73b3d7e762096d6e046e06f676c617f1ff113d026915d286f1b524.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/52c3576aec73b3d7e762096d6e046e06f676c617f1ff113d026915d286f1b524.jpg)

![56720732612ad97e936337c1a70c42d8942455bf83d8b20590bb3d75a4d52fd4.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/56720732612ad97e936337c1a70c42d8942455bf83d8b20590bb3d75a4d52fd4.jpg)

![6daaf3a50ecc7f7e1ceeb0194468ac40142e5541fadea874895be272e6a60d16.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/6daaf3a50ecc7f7e1ceeb0194468ac40142e5541fadea874895be272e6a60d16.jpg)

![ad7b86b142cce56000a12c929d5946674de27002109bdd4015a5faab773ecc40.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/ad7b86b142cce56000a12c929d5946674de27002109bdd4015a5faab773ecc40.jpg)

![ba86b2a773e0b048a1f127f28d048c5d1830605ace6597ff73a4b064887ceacd.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/ba86b2a773e0b048a1f127f28d048c5d1830605ace6597ff73a4b064887ceacd.jpg)

![c74534b5bd07b50d91a0bb4db260c804584b79cd32526bfa4932ed2b685d7f54.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/c74534b5bd07b50d91a0bb4db260c804584b79cd32526bfa4932ed2b685d7f54.jpg)

![d1799e5161963621b3e78e3cdcfad215b61592369a6b5c0809c288ed359b2e36.jpg](../icml_results/5557_TimeChat-Captioner_%20Scripting%20Multi-Scene%20Videos%20with%20Time-Aware%20and%20Structural%20Audio-Visual%20Caption/tables/d1799e5161963621b3e78e3cdcfad215b61592369a6b5c0809c288ed359b2e36.jpg)

<a id="paper-5509"></a>

## CAReDiO: Enhancing Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization


### Images

![05f3263d682fb7e5885c47675d8afa8d130040df3af8e2ee3d399a204c39d1e3.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/05f3263d682fb7e5885c47675d8afa8d130040df3af8e2ee3d399a204c39d1e3.jpg)

![0cbcc8d549d2857ba8d6de15f1a22e98ad1bd5d679b04e619386e481b627fc87.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/0cbcc8d549d2857ba8d6de15f1a22e98ad1bd5d679b04e619386e481b627fc87.jpg)

![22b1eeeae8435b81bb09b86b197c9801bf083910e27f7f201d9ce4a5f89d5778.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/22b1eeeae8435b81bb09b86b197c9801bf083910e27f7f201d9ce4a5f89d5778.jpg)

![2c6986610494978b7c236a849ffb4a6f5349ff690fadf5c3f91645e11f7f774b.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/2c6986610494978b7c236a849ffb4a6f5349ff690fadf5c3f91645e11f7f774b.jpg)

![31ae33c0950a3ca8d0838d6b029d16c32054f121e6759bd61e10af61a99d8f0b.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/31ae33c0950a3ca8d0838d6b029d16c32054f121e6759bd61e10af61a99d8f0b.jpg)

![35b9191bd99fdbe001e310cacb792b522baf3a38309ba3659159b1403977a610.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/35b9191bd99fdbe001e310cacb792b522baf3a38309ba3659159b1403977a610.jpg)

![3b66756959737ab675dcf319b8de9d0a2e2f5b0968fd90b49b1492e74568a90b.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/3b66756959737ab675dcf319b8de9d0a2e2f5b0968fd90b49b1492e74568a90b.jpg)

![4b8d09fddfe4550e66410339ec78c563773b0270939e4f1dca72342e60988beb.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/4b8d09fddfe4550e66410339ec78c563773b0270939e4f1dca72342e60988beb.jpg)

![512c3f37c3fb932cb07176b19032d49ab28e7e191273bceb9d24045433e485a8.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/512c3f37c3fb932cb07176b19032d49ab28e7e191273bceb9d24045433e485a8.jpg)

![6161cd3f1e1da4c14921e488f66b518dfefdfc99229d28174e5599aa10167cb8.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/6161cd3f1e1da4c14921e488f66b518dfefdfc99229d28174e5599aa10167cb8.jpg)

![69d4b6b7f9e7ed3c22a5b583459a499c15d55ded8dcf3b9573b4e5fc783b3c81.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/69d4b6b7f9e7ed3c22a5b583459a499c15d55ded8dcf3b9573b4e5fc783b3c81.jpg)

![7b998fd77cec7d5c1dfb233f59559225086367ac10f9862899aa952ab3cef8b5.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/7b998fd77cec7d5c1dfb233f59559225086367ac10f9862899aa952ab3cef8b5.jpg)

![88419ced4653162dfa2059b752547f9d0fbd84b49aa5a0db4c0b823cf4b9eee4.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/88419ced4653162dfa2059b752547f9d0fbd84b49aa5a0db4c0b823cf4b9eee4.jpg)

![a44775751bbabd0e71f598d607eab88fcd300e1487a5f46b8cba410707aa205e.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/a44775751bbabd0e71f598d607eab88fcd300e1487a5f46b8cba410707aa205e.jpg)

![ac00952ef343bcbb60cbc93272b3a882e967d3ecb3bf2a3dc04d758d40f3f499.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/ac00952ef343bcbb60cbc93272b3a882e967d3ecb3bf2a3dc04d758d40f3f499.jpg)

![b0b280d5d1a240a4bed0909a3719932b116f7d91f7cd6b25d256b0903cd1bc99.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/b0b280d5d1a240a4bed0909a3719932b116f7d91f7cd6b25d256b0903cd1bc99.jpg)

![b1274bbb6bf4fe49bcae4594b58cbc82c61949fbd80358734e769aa915170742.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/b1274bbb6bf4fe49bcae4594b58cbc82c61949fbd80358734e769aa915170742.jpg)

![e56e66cb019bfafd046f4f9691202a112079be3f09b93aeedd3de23ffeca0920.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/e56e66cb019bfafd046f4f9691202a112079be3f09b93aeedd3de23ffeca0920.jpg)

![ed51cb2c9d5a407ea1f0000822910f9e058eeec2ba55cc120dc04c825a6c4a9f.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/images/ed51cb2c9d5a407ea1f0000822910f9e058eeec2ba55cc120dc04c825a6c4a9f.jpg)

### Tables

![01ee352da6c89b9b08380bc292a6fbb4a0cdc4d5dd0d8919a8cebc9afbe5ed0e.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/01ee352da6c89b9b08380bc292a6fbb4a0cdc4d5dd0d8919a8cebc9afbe5ed0e.jpg)

![0fe56682422105f59ee7b67549dc6f0862ce65f6095cf3583e77c9eace151ec7.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/0fe56682422105f59ee7b67549dc6f0862ce65f6095cf3583e77c9eace151ec7.jpg)

![18ce4e24f41192dcbf71bbc01b2c7442be44923aff466d2440478c86606cb400.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/18ce4e24f41192dcbf71bbc01b2c7442be44923aff466d2440478c86606cb400.jpg)

![19a15921aeb9b293aa640bfc22b4280beca82d868eb982800130853faf6af007.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/19a15921aeb9b293aa640bfc22b4280beca82d868eb982800130853faf6af007.jpg)

![1fb472fd0e6bf254b5dfcb9a78610cf33ff1e535822b61501a7b40a1371c402f.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/1fb472fd0e6bf254b5dfcb9a78610cf33ff1e535822b61501a7b40a1371c402f.jpg)

![2198759c432506c9698bdad6f9bc335f18196b600cc6ac980eb33452b6b45313.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/2198759c432506c9698bdad6f9bc335f18196b600cc6ac980eb33452b6b45313.jpg)

![27e7fd8055a93893d62da9be3dab1947f49ce3f8c8dab7a4e82ba2d38320f685.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/27e7fd8055a93893d62da9be3dab1947f49ce3f8c8dab7a4e82ba2d38320f685.jpg)

![29ba142ca2ae15e75777f0a4a8490b3078adc7721606f8d93f1ba6f13d4ab37a.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/29ba142ca2ae15e75777f0a4a8490b3078adc7721606f8d93f1ba6f13d4ab37a.jpg)

![2fab7a04d72216bce6e3ced191a8a35625222e15066e017ad78dd482ba42c503.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/2fab7a04d72216bce6e3ced191a8a35625222e15066e017ad78dd482ba42c503.jpg)

![363199e4fb00f67754f1206e05f9eb3b020f4a47bbde7438daf360a92f764c00.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/363199e4fb00f67754f1206e05f9eb3b020f4a47bbde7438daf360a92f764c00.jpg)

![3c25d166cac80d7aba99aa1da621c0d6c6c7aea322bd465db79651b4dcee5e22.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/3c25d166cac80d7aba99aa1da621c0d6c6c7aea322bd465db79651b4dcee5e22.jpg)

![3d57a21f6367912c847988c3c60b8b46bc888e67de0ce90e598863e7460cd189.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/3d57a21f6367912c847988c3c60b8b46bc888e67de0ce90e598863e7460cd189.jpg)

![4cfe7b127be010da360e5fa89432dada09406088088e51e94d130dc5ac9785fe.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/4cfe7b127be010da360e5fa89432dada09406088088e51e94d130dc5ac9785fe.jpg)

![4f4d629ab0f70c8e365a71d3f3c7c9bfdc59aee4a62fdc50dbf0f001b606d18f.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/4f4d629ab0f70c8e365a71d3f3c7c9bfdc59aee4a62fdc50dbf0f001b606d18f.jpg)

![55da1dae639c1c66d8f9e21967f92f4aced517dc04cf1fccdc995490acedab20.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/55da1dae639c1c66d8f9e21967f92f4aced517dc04cf1fccdc995490acedab20.jpg)

![5d3f5042b2b79154319e6364cd0a052147d5783970d4bff21931cc34ac3f3171.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/5d3f5042b2b79154319e6364cd0a052147d5783970d4bff21931cc34ac3f3171.jpg)

![5d647781040fb39801e89a7e88e670bd0acac917794dbbc026209d3fb7bc3af4.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/5d647781040fb39801e89a7e88e670bd0acac917794dbbc026209d3fb7bc3af4.jpg)

![67da5c6e033983345a54c7b3e5f890e5d006e6ef97ec7ea47fb99638c9554eb4.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/67da5c6e033983345a54c7b3e5f890e5d006e6ef97ec7ea47fb99638c9554eb4.jpg)

![6abb6178e1e48105dba0a96ac469a9c370cb623f2f8e58b6398c48e8e6f8fc84.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/6abb6178e1e48105dba0a96ac469a9c370cb623f2f8e58b6398c48e8e6f8fc84.jpg)

![801d17bc88b583a6bec677440f037d849ace48a5f961c422c48d9192cc97a171.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/801d17bc88b583a6bec677440f037d849ace48a5f961c422c48d9192cc97a171.jpg)

![8bc96a88b2939470e6224f6215cdf96d793ff734f6180ad8de13e0477ee61e54.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/8bc96a88b2939470e6224f6215cdf96d793ff734f6180ad8de13e0477ee61e54.jpg)

![991ed15f06f357e16bd19ee11c6b3ebe6c69b4677d77ba0e3f6dc7c3252649db.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/991ed15f06f357e16bd19ee11c6b3ebe6c69b4677d77ba0e3f6dc7c3252649db.jpg)

![a0498d53820e2b6a2815ca1d2c2285524b557ce699cb9ba03cb4e4ef3e38421c.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/a0498d53820e2b6a2815ca1d2c2285524b557ce699cb9ba03cb4e4ef3e38421c.jpg)

![a33b7ea2392567a51775a4d7d2a9430bb74b967d6f04509fe1caf15e2c8cc76f.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/a33b7ea2392567a51775a4d7d2a9430bb74b967d6f04509fe1caf15e2c8cc76f.jpg)

![b8ed42c0954a5f144a92a91a5da6d06032d5db60920cf3ba4c68c0044fc9eaa3.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/b8ed42c0954a5f144a92a91a5da6d06032d5db60920cf3ba4c68c0044fc9eaa3.jpg)

![bcaf99ebd1e3ddad25424d7be23406861adf5168ee5b39b34777df426656d749.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/bcaf99ebd1e3ddad25424d7be23406861adf5168ee5b39b34777df426656d749.jpg)

![d77e3c255f54bba9f675c6d4564e02957b80ac214ba5da95b329d5ce042c900e.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/d77e3c255f54bba9f675c6d4564e02957b80ac214ba5da95b329d5ce042c900e.jpg)

![decdb1a19663b5cdf011ae804c9e5fba22de98029610e0b49c4785e20ef7edb6.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/decdb1a19663b5cdf011ae804c9e5fba22de98029610e0b49c4785e20ef7edb6.jpg)

![df5f0f0141d758e5a59507d6c85a3c288bedfb928a5e6b3e8c17f57682654caf.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/df5f0f0141d758e5a59507d6c85a3c288bedfb928a5e6b3e8c17f57682654caf.jpg)

![ebb19f524a6b7b4e175d83879e8cd257d633f1f98806fa7951b68a4125a66df1.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/ebb19f524a6b7b4e175d83879e8cd257d633f1f98806fa7951b68a4125a66df1.jpg)

![f666503b0e135490100be5424b8499dd7c5bcf37769c2028c43c979d46c94d0d.jpg](../icml_results/5558_CAReDiO_%20Enhancing%20Cultural%20Alignment%20of%20LLM%20via%20Representativeness%20and%20Distinctiveness%20Guided%20Data%20/tables/f666503b0e135490100be5424b8499dd7c5bcf37769c2028c43c979d46c94d0d.jpg)

<a id="paper-5510"></a>

## MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline


### Images

![041a16f59592dabbd34fcca9d8f921cdf54825582af1094dc87a772e75cbbcc2.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/041a16f59592dabbd34fcca9d8f921cdf54825582af1094dc87a772e75cbbcc2.jpg)

![060bedd9de1477d33b23ecd5df2aebd394dde374d50f52e400dce2a12c57c121.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/060bedd9de1477d33b23ecd5df2aebd394dde374d50f52e400dce2a12c57c121.jpg)

![06e87170dace6edcdfc89e0e94c501ada7afb42d8783765eaca59154ffc268bd.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/06e87170dace6edcdfc89e0e94c501ada7afb42d8783765eaca59154ffc268bd.jpg)

![11525b20531e325a8fad014a9d83417f31f41678f0944eaec42bc4db7cfc18c7.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/11525b20531e325a8fad014a9d83417f31f41678f0944eaec42bc4db7cfc18c7.jpg)

![155389e00ce69374e4fe2ba0d7ec1cde9faaea01c8edccb0dfff6951327d9bb2.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/155389e00ce69374e4fe2ba0d7ec1cde9faaea01c8edccb0dfff6951327d9bb2.jpg)

![1d62c406ddf156d1c515890a4eedfb4d3897094c60c2d5bf219f298ebb865982.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/1d62c406ddf156d1c515890a4eedfb4d3897094c60c2d5bf219f298ebb865982.jpg)

![21a5021a15c54b3afb1229d0c670c4bd078c991c059513be224f2bf81924572e.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/21a5021a15c54b3afb1229d0c670c4bd078c991c059513be224f2bf81924572e.jpg)

![2b7405f13676913a8da42d4358cedd3f6f1ea2d9a12fe09e3d30149416e961ba.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/2b7405f13676913a8da42d4358cedd3f6f1ea2d9a12fe09e3d30149416e961ba.jpg)

![3e42d7876452f0bbea420acc8e31c392890c27e9fe83fb65664d3e6ae4f62eb9.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/3e42d7876452f0bbea420acc8e31c392890c27e9fe83fb65664d3e6ae4f62eb9.jpg)

![43f31bbed09cc97ad4afb86bedd4073953689f40567e30459f06af63e7dd9149.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/43f31bbed09cc97ad4afb86bedd4073953689f40567e30459f06af63e7dd9149.jpg)

![497a4d3de3c7740f733d03725ae3c8ecfb21bb5489e027313341d6aa46339d45.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/497a4d3de3c7740f733d03725ae3c8ecfb21bb5489e027313341d6aa46339d45.jpg)

![4cb4919599f5529d0b06929ce66e72106257a504ccedcca322e455b9265a7c8d.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/4cb4919599f5529d0b06929ce66e72106257a504ccedcca322e455b9265a7c8d.jpg)

![61634644b802a1b27e4c99bbbccbd9bc3c77d6c1c0ac6e63d28535536aa2db07.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/61634644b802a1b27e4c99bbbccbd9bc3c77d6c1c0ac6e63d28535536aa2db07.jpg)

![6ab7e61154a366a8eef8be040c320e702df7edba4c4c2b962bd68737326e51da.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/6ab7e61154a366a8eef8be040c320e702df7edba4c4c2b962bd68737326e51da.jpg)

![7240147a889990c70e9a80e0ad7f17025f6d65a779b7c11434af4b9fff061736.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/7240147a889990c70e9a80e0ad7f17025f6d65a779b7c11434af4b9fff061736.jpg)

![778322e8acdcd8e40c3bcc3485ac8f50e8fcebd6f5bda710df8654bf97caf22b.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/778322e8acdcd8e40c3bcc3485ac8f50e8fcebd6f5bda710df8654bf97caf22b.jpg)

![7db3babdd0db6f1b2b8f1814038a21101c37affdf20c1ebeba14bff45e10b322.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/7db3babdd0db6f1b2b8f1814038a21101c37affdf20c1ebeba14bff45e10b322.jpg)

![848f31e294799db9f31dad7d86b58059c734035c346db34eb76d22d0dcff6165.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/848f31e294799db9f31dad7d86b58059c734035c346db34eb76d22d0dcff6165.jpg)

![84b746327a9091029688a54cc9acbd7c1bf332b96f89428207a14f5d118bf49f.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/84b746327a9091029688a54cc9acbd7c1bf332b96f89428207a14f5d118bf49f.jpg)

![97e4c4837f59ebfe688376cc0823a9da3a988b45db5907c8645326446626ffa8.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/97e4c4837f59ebfe688376cc0823a9da3a988b45db5907c8645326446626ffa8.jpg)

![af98d586cd3a522280af0e82574c814d6962938b3874af9a6270a83541620eab.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/af98d586cd3a522280af0e82574c814d6962938b3874af9a6270a83541620eab.jpg)

![b2e977319226ed3d45ebbad891b3b0e61705688a0a3e9178564d3a06170787c5.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/b2e977319226ed3d45ebbad891b3b0e61705688a0a3e9178564d3a06170787c5.jpg)

![b64c3a7b57033500b2def34d408fc4391680fc8c7067030e678b845b78e22092.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/b64c3a7b57033500b2def34d408fc4391680fc8c7067030e678b845b78e22092.jpg)

![b824fdfc07d2a87dad63e82e8cd4d555b9e1a455ba46d23babb38274c2628975.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/b824fdfc07d2a87dad63e82e8cd4d555b9e1a455ba46d23babb38274c2628975.jpg)

![d1674ff221e5ccc97198e8c3a737d952819cddeb5d8b9a9c890920b4cb7c97dc.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/d1674ff221e5ccc97198e8c3a737d952819cddeb5d8b9a9c890920b4cb7c97dc.jpg)

![d9f16944c6a6deb062cffe4d212cadc175a855613a6522ec16ae945cafebe68f.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/d9f16944c6a6deb062cffe4d212cadc175a855613a6522ec16ae945cafebe68f.jpg)

![ddce31dc0d6528884b47f65719ce5fef5ca20b6e382d7d5afd66d7a667b9029e.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/ddce31dc0d6528884b47f65719ce5fef5ca20b6e382d7d5afd66d7a667b9029e.jpg)

![e1056be91c3f564eeae8e35ac9c439244d8f5d4df11199d9c7e7466b61f299fc.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/e1056be91c3f564eeae8e35ac9c439244d8f5d4df11199d9c7e7466b61f299fc.jpg)

![eac97f30ad5581e3b8d8cf8364ca8e2c6f27c02d7ef77deda91da9a117f94b6e.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/eac97f30ad5581e3b8d8cf8364ca8e2c6f27c02d7ef77deda91da9a117f94b6e.jpg)

![ebeb295a1766fc0dcf5df2eb8625b8de4785d05831e61e48e6e6aec230be3f14.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/ebeb295a1766fc0dcf5df2eb8625b8de4785d05831e61e48e6e6aec230be3f14.jpg)

![f42473fdbe06caa03cf9d80441e4c909686941b6b221d562387ad2350e959901.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/f42473fdbe06caa03cf9d80441e4c909686941b6b221d562387ad2350e959901.jpg)

![f5672202d69af7591cae1d9b2505374135f33363c836c18e7a34fabfb1d09f7a.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/f5672202d69af7591cae1d9b2505374135f33363c836c18e7a34fabfb1d09f7a.jpg)

![f8e395b50366eec3c0b3bb28c64e9036d234960a6d346b13776b837d2dae05fb.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/f8e395b50366eec3c0b3bb28c64e9036d234960a6d346b13776b837d2dae05fb.jpg)

![f9869f96d3f75621735c6fc1b6fb271e1dcd7c1b59ad732015452183f9d707d9.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/images/f9869f96d3f75621735c6fc1b6fb271e1dcd7c1b59ad732015452183f9d707d9.jpg)

### Tables

![20025df330937b5a48f704c0de8896d6e11686a4a905710a5b736ab1533ed673.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/20025df330937b5a48f704c0de8896d6e11686a4a905710a5b736ab1533ed673.jpg)

![209a48f681c930192a37b9947592358df9341cd2853a500b886a93bc198f5347.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/209a48f681c930192a37b9947592358df9341cd2853a500b886a93bc198f5347.jpg)

![23896e87b58f26811da80a7346c152282fdf01c06b444c4124b303b78a311f13.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/23896e87b58f26811da80a7346c152282fdf01c06b444c4124b303b78a311f13.jpg)

![34c7f81dadf572a2072aad960a26bd82dc8edaa28fd6c16cccaae8c6d71402c2.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/34c7f81dadf572a2072aad960a26bd82dc8edaa28fd6c16cccaae8c6d71402c2.jpg)

![366d0cdbc645c7fe64d9afcd31249eae1f432cea3bfeeba9f8b1fef5c4b80535.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/366d0cdbc645c7fe64d9afcd31249eae1f432cea3bfeeba9f8b1fef5c4b80535.jpg)

![51e5f0f2505f65a14be2c5836e0275823de8c2094ec8c7fdf6bbbe6ab8196092.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/51e5f0f2505f65a14be2c5836e0275823de8c2094ec8c7fdf6bbbe6ab8196092.jpg)

![8abbf0e96d535894683faa241f0575680dfe777075a6170717536766482ab6f9.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/8abbf0e96d535894683faa241f0575680dfe777075a6170717536766482ab6f9.jpg)

![a69dce8ddc8f49ab258127473d179f57bbf88cb4f6adab68898543df7afb298f.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/a69dce8ddc8f49ab258127473d179f57bbf88cb4f6adab68898543df7afb298f.jpg)

![b2c7f7d987118f627811ae48d687772786c7c389e0118208fe2e7617bba3922d.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/b2c7f7d987118f627811ae48d687772786c7c389e0118208fe2e7617bba3922d.jpg)

![c54e774dd580b77e3b405bd39b304cf0e60dbaea64380aa1ebc23ce01c8908be.jpg](../icml_results/5559_MM-DeepResearch_%20A%20Simple%20and%20Effective%20Multimodal%20Agentic%20Search%20Baseline/tables/c54e774dd580b77e3b405bd39b304cf0e60dbaea64380aa1ebc23ce01c8908be.jpg)

<a id="paper-5511"></a>

## A Linearly Convergent Proximal Subgradient Algorithm for Sparse Portfolio Optimization with Transaction Cost

### Images

![f9cd7fc45b485d9bf65b9df4853734b4f09a37bbfff481c2b61664fe74b0a569.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/images/f9cd7fc45b485d9bf65b9df4853734b4f09a37bbfff481c2b61664fe74b0a569.jpg)

### Tables

![17553e57bb975b3ac7faea6fc9cea27f62cf51f16a8d57a3fcfb59de759696b1.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/17553e57bb975b3ac7faea6fc9cea27f62cf51f16a8d57a3fcfb59de759696b1.jpg)

![2779fd7c73089ea3c527201c4027e5c8703d149866ae3017afdee0cff2252e3f.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/2779fd7c73089ea3c527201c4027e5c8703d149866ae3017afdee0cff2252e3f.jpg)

![2a4533abf9d4d4e769170f9c1a2582f467311f60b47da14cfa13a18e6b6977cb.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/2a4533abf9d4d4e769170f9c1a2582f467311f60b47da14cfa13a18e6b6977cb.jpg)

![8650fa555ab1a5c2b22765eb11d60417dd6dc4a09dd211745738a1dad3275251.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/8650fa555ab1a5c2b22765eb11d60417dd6dc4a09dd211745738a1dad3275251.jpg)

![b33060404c705af3976bba179b9e2cd3e50ea1ce7b6257c8216c234553466e9e.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/b33060404c705af3976bba179b9e2cd3e50ea1ce7b6257c8216c234553466e9e.jpg)

![b9f2e04d049dcbb1e720c41c07edb3d41a57d4ba83745c5ad284eea771c65996.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/b9f2e04d049dcbb1e720c41c07edb3d41a57d4ba83745c5ad284eea771c65996.jpg)

![f3eec133719a18e5632497ca8e977dae61405416bd72a7fcb7a92a5123223d0e.jpg](../icml_results/5560_A%20Linearly%20Convergent%20Proximal%20Subgradient%20Algorithm%20for%20Sparse%20Portfolio%20Optimization%20with%20Transact/tables/f3eec133719a18e5632497ca8e977dae61405416bd72a7fcb7a92a5123223d0e.jpg)
