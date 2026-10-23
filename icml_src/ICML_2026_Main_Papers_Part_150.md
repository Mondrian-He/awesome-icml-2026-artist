# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 150 of 197

## 目录 (Table of Contents)

1. [Easier to Judge than to Find: Predicting In-Context Learning Success for Demonstration Selection](#paper-4918)
2. [Unbiased Reward Modeling from Implicit Feedback for LLM Alignment](#paper-4919)
3. [WorldTravel: A Realistic Multimodal Travel-Planning Benchmark with Tightly Coupled Constraints](#paper-4920)
4. [KAST-BAR: Knowledge-Anchored Semantically-Dynamic Topology Brain Autoregressive Modeling for Universal Neural Interpretation](#paper-4921)
5. [Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](#paper-4922)
6. [Towards Context-Invariant Safety Alignment for Large Language Models](#paper-4923)
7. [Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model](#paper-4924)
8. [MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation](#paper-4925)
9. [$\mathbbR^2k$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](#paper-4926)
10. [GenExam: A Multidisciplinary Text-to-Image Exam](#paper-4927)
11. [Relational In-Context Learning via Synthetic Pre-training with Structural Prior](#paper-4928)
12. [Deep Scientific Reasoning under Physical Constraints: Structure-Aware Spectrum Prediction](#paper-4929)
13. [Cache Coherent Resampling for Efficient Test Time Scaling in LLM Reasoning via Adaptive Sequential Monte Carlo](#paper-4930)
14. [LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization](#paper-4931)
15. [Spiked-CFR: Causal Representation Learning from LLMs via Wasserstein Projection Pursuit](#paper-4932)
16. [MemPot: Defending Against Memory Extraction Attack with Optimized Honeypots](#paper-4933)
17. [AES: Curing Optimizer Blindness in Long-Tailed Recognition via State-Aware Correction](#paper-4934)
18. [The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](#paper-4935)
19. [Partial Identification under High-Dimensional Potential Outcomes and Confounders via Optimal Transport](#paper-4936)
20. [PromptRL: Prompt Matters in RL for Flow-Based Image Generation](#paper-4937)
21. [Non-Parametric Probabilistic Robustness: A Conservative Risk Estimator under Unknown Perturbation Distributions](#paper-4938)
22. [From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](#paper-4939)
23. [Target-Driven Policy Optimization for Sequential Counterfactual Outcome Control](#paper-4940)
24. [Dual-branch Robust Unlearnable Examples](#paper-4941)
25. [Finding Stationary Points by Comparisons](#paper-4942)
26. [AVI-Bench: Toward Human-like Audio-Visual Intelligence of Omni-MLLMs](#paper-4943)
27. [CLINIC: Towards High-quality Graph Out-Of-Distribution Detection](#paper-4944)
28. [Scalable and Stable Estimation of Amari $α$-Divergence using Random Fourier Features](#paper-4945)
29. [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](#paper-4946)
30. [LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](#paper-4947)
31. [Autobidding Auctions with LLM-Powered Creatives](#paper-4948)
32. [Do LLMs "Feel"? Emotion Circuits Discovery and Control](#paper-4949)
33. [Co-Evolving Latent Action World Models](#paper-4950)

---


<a id="paper-4918"></a>

## Easier to Judge than to Find: Predicting In-Context Learning Success for Demonstration Selection

### Images

![0262c54019961df9788d990e7063321fe7d15401fac699c0d0281ba60e02b88c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/0262c54019961df9788d990e7063321fe7d15401fac699c0d0281ba60e02b88c.jpg)

![1b1b64777afa81476fb89dd1a79b3f56c37da25033e09183baa6f57dd95dea57.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/1b1b64777afa81476fb89dd1a79b3f56c37da25033e09183baa6f57dd95dea57.jpg)

![2c8904a3d815a67644f951c323a3b869e05f0cd6d45a78f18c802c3be1445c77.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/2c8904a3d815a67644f951c323a3b869e05f0cd6d45a78f18c802c3be1445c77.jpg)

![30a83262e2eb19199f6632a6a2c9a7d985ca073abeb23f62a46d9b8fd4e72e2c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/30a83262e2eb19199f6632a6a2c9a7d985ca073abeb23f62a46d9b8fd4e72e2c.jpg)

![3dbdc74cef5fd950bbe1b9b50f226c39b9192532cd9aa086742fa6e42ab2fd99.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/3dbdc74cef5fd950bbe1b9b50f226c39b9192532cd9aa086742fa6e42ab2fd99.jpg)

![574613e5aead075321468bc357c908b3f5c913decc7304592b539a4b8e82a269.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/574613e5aead075321468bc357c908b3f5c913decc7304592b539a4b8e82a269.jpg)

![5dc2d3e145da6ce2ef73a63fc173a936444e4b32fda7eb42839ee8e828bfffc4.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/5dc2d3e145da6ce2ef73a63fc173a936444e4b32fda7eb42839ee8e828bfffc4.jpg)

![61149a11905734d98e4828a0b76aebb9b148c3837dd0c648a1e2affa42ab23cb.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/61149a11905734d98e4828a0b76aebb9b148c3837dd0c648a1e2affa42ab23cb.jpg)

![65c6e6c241f6d48fb96aa74e2078f576b171c13dee74104765cc21e9e541e843.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/65c6e6c241f6d48fb96aa74e2078f576b171c13dee74104765cc21e9e541e843.jpg)

![88bbb85f29750dd85eeb4badcbe1222c3df59d5201703caeace66cb2bce8b6f1.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/88bbb85f29750dd85eeb4badcbe1222c3df59d5201703caeace66cb2bce8b6f1.jpg)

![8c8ee5a5638c0c313b90b5ee51515a76178420050aa12c850aa11edb9b62aa14.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/8c8ee5a5638c0c313b90b5ee51515a76178420050aa12c850aa11edb9b62aa14.jpg)

![b65b41125a703d8de00f263546edaf27838e0d0adc34e23e36b0799f8848d48c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/b65b41125a703d8de00f263546edaf27838e0d0adc34e23e36b0799f8848d48c.jpg)

![bb050ccf67ca144dbc18f21c49e4d8e8ec0de5c4651e1e8e5f316d2f27a1caf2.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/bb050ccf67ca144dbc18f21c49e4d8e8ec0de5c4651e1e8e5f316d2f27a1caf2.jpg)

![c1231407316b80ac363ec93393f24c8fff868b696fc78576c16a9c47f1e9777d.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/c1231407316b80ac363ec93393f24c8fff868b696fc78576c16a9c47f1e9777d.jpg)

![d779af0f4436c0aaba14d5b6e79290f48aa97bb928975bc707a416dd0b5389dd.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/d779af0f4436c0aaba14d5b6e79290f48aa97bb928975bc707a416dd0b5389dd.jpg)

### Tables

![05199d524772c9836d1c7246d1770ce7c51f542f94f61e54db4cd4baabe2bc35.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/05199d524772c9836d1c7246d1770ce7c51f542f94f61e54db4cd4baabe2bc35.jpg)

![0c9defc8713dc2a7b44d4b735445283713ad76a7d3d9658f0c7a41d6a976c9ef.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/0c9defc8713dc2a7b44d4b735445283713ad76a7d3d9658f0c7a41d6a976c9ef.jpg)

![14ece19dff36ea8e53f8b3b65efea52fb442b0ed169bad122ba5e297dc26e445.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/14ece19dff36ea8e53f8b3b65efea52fb442b0ed169bad122ba5e297dc26e445.jpg)

![1b0e30efcf9a89742af6b76c04cb46a8f30a751409aec35e18e140b4fceca33f.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/1b0e30efcf9a89742af6b76c04cb46a8f30a751409aec35e18e140b4fceca33f.jpg)

![23bbcde4c3d2db4af9bc1e3d5a3e5fc56ff971986e668c417cd68605b05194f7.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/23bbcde4c3d2db4af9bc1e3d5a3e5fc56ff971986e668c417cd68605b05194f7.jpg)

![2830db14853b796c53c575377c66d3905b789f40d4c250a044dfa11e35ad6e7c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2830db14853b796c53c575377c66d3905b789f40d4c250a044dfa11e35ad6e7c.jpg)

![2a128e18b17153231cd720085abba87dc08bf79c9862e9d97758a508ff18c441.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2a128e18b17153231cd720085abba87dc08bf79c9862e9d97758a508ff18c441.jpg)

![2faf7845b14b9abcd0210af0d698157e66c3e5ba8fa73dffbdb5819390be8802.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2faf7845b14b9abcd0210af0d698157e66c3e5ba8fa73dffbdb5819390be8802.jpg)

![34bdb897915e2d78703501fb841d952c10609dcf0a9992bfdbcee059968b8d1f.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/34bdb897915e2d78703501fb841d952c10609dcf0a9992bfdbcee059968b8d1f.jpg)

![6420519a172b26fcd93443d0a800fc5e56b9c04a95f6d1ee340d8fc2e9ebbe60.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/6420519a172b26fcd93443d0a800fc5e56b9c04a95f6d1ee340d8fc2e9ebbe60.jpg)

![6d51a197d11a465f1999804c5996acdd1408f9f1116dcdcc1f1926f5e12c7f0a.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/6d51a197d11a465f1999804c5996acdd1408f9f1116dcdcc1f1926f5e12c7f0a.jpg)

![7270f1822eb385b39adfe7a8c1992dbee31fc47938cd673ba917be70ef589ed1.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/7270f1822eb385b39adfe7a8c1992dbee31fc47938cd673ba917be70ef589ed1.jpg)

![7f81978600d001919ed0bb67a95f12f538f1265e2321cf1ff0e098ea9da4c609.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/7f81978600d001919ed0bb67a95f12f538f1265e2321cf1ff0e098ea9da4c609.jpg)

![b2a5edbba8832fdb08ccddc32b1c03077a0e8993cc50247535ed3aa5f205aac0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/b2a5edbba8832fdb08ccddc32b1c03077a0e8993cc50247535ed3aa5f205aac0.jpg)

![bf6b2a9a8dc4c4a71d26a98be6556d18b30d7ba21bc35dfaa40b807a7fabb8e2.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/bf6b2a9a8dc4c4a71d26a98be6556d18b30d7ba21bc35dfaa40b807a7fabb8e2.jpg)

![c237df5c7fdd61f67233cf5ded140f79d93af2d02c19eb61190d8298c8e9e79a.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/c237df5c7fdd61f67233cf5ded140f79d93af2d02c19eb61190d8298c8e9e79a.jpg)

![c4f0a7e240d2a947dfa36d55483070b9628c614b80c49041e8c8cca0e03602d0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/c4f0a7e240d2a947dfa36d55483070b9628c614b80c49041e8c8cca0e03602d0.jpg)

![cd3e613ad19889fdd32c454872a9aba65c16c15d056749c4d0a63b481109026c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/cd3e613ad19889fdd32c454872a9aba65c16c15d056749c4d0a63b481109026c.jpg)

![d16863e8477e27da87879639a7ee68488ed6a5078911a20ee82e4b05b935d6b5.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/d16863e8477e27da87879639a7ee68488ed6a5078911a20ee82e4b05b935d6b5.jpg)

![d7cb340723362cb9087ee6392e5ac6466f5e3e2c99641d33cc83a17904638916.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/d7cb340723362cb9087ee6392e5ac6466f5e3e2c99641d33cc83a17904638916.jpg)

![e255d51d2e8cf326230078a7731df47bd3e9cff07d3530b8f35fc835056c23e5.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/e255d51d2e8cf326230078a7731df47bd3e9cff07d3530b8f35fc835056c23e5.jpg)

![e873269391baac5a80d98ff3c671edf105a1c56d8e68e51c7601b55d9864efbc.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/e873269391baac5a80d98ff3c671edf105a1c56d8e68e51c7601b55d9864efbc.jpg)

![ecc3243945a0fedc288d0350da4076cf30bedd5c2abd9c8bd4d511fa67511d32.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/ecc3243945a0fedc288d0350da4076cf30bedd5c2abd9c8bd4d511fa67511d32.jpg)

![ed596527fa250c89c3b42c9bc58198885c33fc7247838e0e39cdbe78e3562ef0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/ed596527fa250c89c3b42c9bc58198885c33fc7247838e0e39cdbe78e3562ef0.jpg)

## Easier to Judge than to Find: Predicting In-Context Learning Success for Demonstration Selection


### Images

![1111aa5a1e4d4e08ba83a4cd87c4da3036fbf3b22ccd23dc7c63864b7ecb8081.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/1111aa5a1e4d4e08ba83a4cd87c4da3036fbf3b22ccd23dc7c63864b7ecb8081.jpg)

![17b86a183c2e37a02545ea07b9595150f944d2ba2f699ca58df32157d5227eb0.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/17b86a183c2e37a02545ea07b9595150f944d2ba2f699ca58df32157d5227eb0.jpg)

![22fdfd00fe9c4df671ae7390f8160768011b944f864f49dbeb2dc73bb2e9fc78.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/22fdfd00fe9c4df671ae7390f8160768011b944f864f49dbeb2dc73bb2e9fc78.jpg)

![24711e4d1b6ca0be51272362e443212c5ec7037367eed470b86cc90603d0acd8.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/24711e4d1b6ca0be51272362e443212c5ec7037367eed470b86cc90603d0acd8.jpg)

![4022daf319bdf59c675f5dc6388554f7b9e2b4308b4ae03850946550e34daf21.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/4022daf319bdf59c675f5dc6388554f7b9e2b4308b4ae03850946550e34daf21.jpg)

![472a6430c458492fe3b4855f590140fc59b59058aff23e15f4f90815ed8e65a5.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/472a6430c458492fe3b4855f590140fc59b59058aff23e15f4f90815ed8e65a5.jpg)

![764f510a1d549b4b50f615b45d6042222f55422a3fc9e333e337db0cc1cafde4.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/764f510a1d549b4b50f615b45d6042222f55422a3fc9e333e337db0cc1cafde4.jpg)

![a2aa67a21c577fa3a3cfab2c0c2bb3820c7b08c28042c9b14712f267994eaf62.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/a2aa67a21c577fa3a3cfab2c0c2bb3820c7b08c28042c9b14712f267994eaf62.jpg)

![aab453152a1fac779d928af3b1a2801f202f8f7c4c69b36cf6cb6ac7fd776538.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/images/aab453152a1fac779d928af3b1a2801f202f8f7c4c69b36cf6cb6ac7fd776538.jpg)

### Tables

![1d2d8de864a5c7f90a1983cabdbbdda668e1dfaf5e5bfc9fb03aae579ba98e94.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/1d2d8de864a5c7f90a1983cabdbbdda668e1dfaf5e5bfc9fb03aae579ba98e94.jpg)

![2fae394c00814ce153778db4676a9facdbb78271e5b35d8dfe06a92bda370e8a.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/2fae394c00814ce153778db4676a9facdbb78271e5b35d8dfe06a92bda370e8a.jpg)

![5876c0893bffa23a3a9951a38329ed55fc52355a1a880428cdf9658f5395325d.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/5876c0893bffa23a3a9951a38329ed55fc52355a1a880428cdf9658f5395325d.jpg)

![6103b3892a588eeac99686e23652f7fa259b3f4d7cb8d6a4e82f55ee9e4c11c8.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/6103b3892a588eeac99686e23652f7fa259b3f4d7cb8d6a4e82f55ee9e4c11c8.jpg)

![6d5a30f40ce7d6424bcacabc5736ddfc081b7fd01129f4661e89fe83a21fb8f4.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/6d5a30f40ce7d6424bcacabc5736ddfc081b7fd01129f4661e89fe83a21fb8f4.jpg)

![cd4632387f42f6b7dc2c1749f4662f0305d4eae728771cf02a32b2dd4ad89c6e.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/cd4632387f42f6b7dc2c1749f4662f0305d4eae728771cf02a32b2dd4ad89c6e.jpg)

![e2ae4aa399ed65b2cececa1e85c5775e9c17496e6ddc9480dfd7d3feab1ac686.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/e2ae4aa399ed65b2cececa1e85c5775e9c17496e6ddc9480dfd7d3feab1ac686.jpg)

![f8e5fd495810a53fd9b2370288a075c8e8aaec23947b61b42624b26a55336c5b.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/f8e5fd495810a53fd9b2370288a075c8e8aaec23947b61b42624b26a55336c5b.jpg)

![fc3b35bbdfc3a762a5457401c1830c541ae22026d4eede532b71df63bc9f8bb8.jpg](../icml_results/4966_Easier%20to%20Judge%20than%20to%20Find_%20Predicting%20In-Context%20Learning%20Success%20for%20Demonstration%20Selection/tables/fc3b35bbdfc3a762a5457401c1830c541ae22026d4eede532b71df63bc9f8bb8.jpg)

<a id="paper-4919"></a>

## Unbiased Reward Modeling from Implicit Feedback for LLM Alignment


### Images

![01912ca431f9a5a100f3a8d3dd5b5f9bcc7b7b74d9b1d1a491a4f8744c78005a.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/images/01912ca431f9a5a100f3a8d3dd5b5f9bcc7b7b74d9b1d1a491a4f8744c78005a.jpg)

![2bdf47e82ea1a1e94ee013bea7c4b865b1e49962ead5a76b247ed1b4b6014a69.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/images/2bdf47e82ea1a1e94ee013bea7c4b865b1e49962ead5a76b247ed1b4b6014a69.jpg)

![6e1eaeca40f14ecf98aeebd0f2ea7bdd3e438979635f81a4aedf6dac0d1e8704.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/images/6e1eaeca40f14ecf98aeebd0f2ea7bdd3e438979635f81a4aedf6dac0d1e8704.jpg)

![7124c723f24cdf11192bcb123c40c7975fc11bb147eb90641086971697ab0716.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/images/7124c723f24cdf11192bcb123c40c7975fc11bb147eb90641086971697ab0716.jpg)

### Tables

![1b66113d0406c9828ab138beac02912dbf73fd67016a8a45b4d24a65097d5f22.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/1b66113d0406c9828ab138beac02912dbf73fd67016a8a45b4d24a65097d5f22.jpg)

![7dc17e069f41c2db5e5185baefef8f2adde10b41709e0b8f7b358cbc6cb43da6.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/7dc17e069f41c2db5e5185baefef8f2adde10b41709e0b8f7b358cbc6cb43da6.jpg)

![a07ffaa7dd9bcf2870e9b4545c02e4d64ef68e9a2ac70f24971efd7c3ab4a5fe.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/a07ffaa7dd9bcf2870e9b4545c02e4d64ef68e9a2ac70f24971efd7c3ab4a5fe.jpg)

![ae086b46a89320c996355cd5cb845fc1090d4030a36408bbcbfbbaebb8374a11.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/ae086b46a89320c996355cd5cb845fc1090d4030a36408bbcbfbbaebb8374a11.jpg)

![b084f706442586415634017c735644b74f74a8c85a33d6252794c76e30c18970.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/b084f706442586415634017c735644b74f74a8c85a33d6252794c76e30c18970.jpg)

![eb676e0058659267d821860387621c88847fa5196df036164bc33e4c53eb361f.jpg](../icml_results/4967_Unbiased%20Reward%20Modeling%20from%20Implicit%20Feedback%20for%20LLM%20Alignment/tables/eb676e0058659267d821860387621c88847fa5196df036164bc33e4c53eb361f.jpg)

<a id="paper-4920"></a>

## WorldTravel: A Realistic Multimodal Travel-Planning Benchmark with Tightly Coupled Constraints


### Images

![0b45a4f18f3c902a28e6fbb94bf388ab96e0c861a6ca971a6a67677bd58d31b0.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/0b45a4f18f3c902a28e6fbb94bf388ab96e0c861a6ca971a6a67677bd58d31b0.jpg)

![0cea6c9d615379eabb6472626026702fa01b1aa5ea1b53179ca381303f2a5ff5.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/0cea6c9d615379eabb6472626026702fa01b1aa5ea1b53179ca381303f2a5ff5.jpg)

![11a9ffafe493586242a659934d06bf5c4c3ef7af7d9163891c4ee4531c48b75d.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/11a9ffafe493586242a659934d06bf5c4c3ef7af7d9163891c4ee4531c48b75d.jpg)

![13b27314ae7c0b49a0909b15d4ec6832a9eb449bc3ac36f063a07911a66249b0.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/13b27314ae7c0b49a0909b15d4ec6832a9eb449bc3ac36f063a07911a66249b0.jpg)

![1e784ddb9aca2114e50fa3a8f2bf7d37fc30cb7efb1b3de331f421df18df3740.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/1e784ddb9aca2114e50fa3a8f2bf7d37fc30cb7efb1b3de331f421df18df3740.jpg)

![1f36899b72529908a55a5feded409e18c347f8f477dd448cbafd6c0c50532135.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/1f36899b72529908a55a5feded409e18c347f8f477dd448cbafd6c0c50532135.jpg)

![219f4a75aa30a33a30b4b09a5f05892d99098402759aa46e9bcd04bdb5686d41.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/219f4a75aa30a33a30b4b09a5f05892d99098402759aa46e9bcd04bdb5686d41.jpg)

![222b94c6a7312db10610b9d236cd8176d794e48682f86c5f28135c66b63ff13e.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/222b94c6a7312db10610b9d236cd8176d794e48682f86c5f28135c66b63ff13e.jpg)

![5629b8a11e43083aa6ddaafacc05ab30cfa47ec76ef386d6757b001f25317599.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/5629b8a11e43083aa6ddaafacc05ab30cfa47ec76ef386d6757b001f25317599.jpg)

![5a2de0a16744c41066dcd2f6f1aa6abb42a9adb8453bd10f7a3c5d643cdfd58a.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/5a2de0a16744c41066dcd2f6f1aa6abb42a9adb8453bd10f7a3c5d643cdfd58a.jpg)

![624ed3698530ef7f55fb5b60d77c3ae0e291c88e12eae003d45db76032a5a19c.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/624ed3698530ef7f55fb5b60d77c3ae0e291c88e12eae003d45db76032a5a19c.jpg)

![735f36f8ee2b072bc736576c17ce643997d3813ee3eaeea281f430c74b77450c.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/735f36f8ee2b072bc736576c17ce643997d3813ee3eaeea281f430c74b77450c.jpg)

![74b46aa87d5a8edea4e0cd915fdfef21761aba682a249a7024810eebcceb9908.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/74b46aa87d5a8edea4e0cd915fdfef21761aba682a249a7024810eebcceb9908.jpg)

![7f30052095b4897c6bd6cb1d64b79c8b75b10b3c5e0a0ba8f5a52ab3f9bedd89.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/7f30052095b4897c6bd6cb1d64b79c8b75b10b3c5e0a0ba8f5a52ab3f9bedd89.jpg)

![86a737cd3cf3154aab839cba38dd6306856213420cbf77265b2a308ff46c8580.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/86a737cd3cf3154aab839cba38dd6306856213420cbf77265b2a308ff46c8580.jpg)

![881011451655d21b273a9582a152b54ed2c3bd6a9d3784c03e01fd62fd8eae6d.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/881011451655d21b273a9582a152b54ed2c3bd6a9d3784c03e01fd62fd8eae6d.jpg)

![8bd32e22660d1ca88f100e1708e7e677d778125c1c7ef612eeac42e4bfd2c93a.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/8bd32e22660d1ca88f100e1708e7e677d778125c1c7ef612eeac42e4bfd2c93a.jpg)

![9c7bb0aa2d5628389b755bf043ba75e5f78c367a9fc2aeeddb741f01c6155b50.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/9c7bb0aa2d5628389b755bf043ba75e5f78c367a9fc2aeeddb741f01c6155b50.jpg)

![9e7a2b996c8293e2f4fa82579187d9546e7f48733365f2bbb9c33344741d6da3.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/9e7a2b996c8293e2f4fa82579187d9546e7f48733365f2bbb9c33344741d6da3.jpg)

![a239cdca9d7269b8ccb6944cd39b57bfd7de9c1c9ecfac0cea47817e7897abe9.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/a239cdca9d7269b8ccb6944cd39b57bfd7de9c1c9ecfac0cea47817e7897abe9.jpg)

![a4fe120e67225d3e308d94f33dc81597d675da09afe0949fde861634ae1c9c54.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/a4fe120e67225d3e308d94f33dc81597d675da09afe0949fde861634ae1c9c54.jpg)

![b243cf40fb739f06a366d824bb78b088993e301b9a8172bb421f799aa0c9f7a6.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/b243cf40fb739f06a366d824bb78b088993e301b9a8172bb421f799aa0c9f7a6.jpg)

![b5bb1984c4748c487b15847da1ba0a92966b3a9de78ccdf57fc84341619ee88b.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/b5bb1984c4748c487b15847da1ba0a92966b3a9de78ccdf57fc84341619ee88b.jpg)

![cc2aa84b323fa6d9899b4d1defada111f890044f72a3c1829420aa290dd506c4.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/cc2aa84b323fa6d9899b4d1defada111f890044f72a3c1829420aa290dd506c4.jpg)

![cc7ca6993d63f6016fabb218d76424c314a355c1dadcb370017d0afbeed17a7e.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/cc7ca6993d63f6016fabb218d76424c314a355c1dadcb370017d0afbeed17a7e.jpg)

![e48888c08612f8c06ee666b54feacc46dbebcd437f2a24ed936752b4f9359a30.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/e48888c08612f8c06ee666b54feacc46dbebcd437f2a24ed936752b4f9359a30.jpg)

![f9b169969a1358e546be633da86fa9df400f72e756a73bf5178f0642016e32d6.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/images/f9b169969a1358e546be633da86fa9df400f72e756a73bf5178f0642016e32d6.jpg)

### Tables

![0a7d0a51e013fd5770f1422be34aa4c03685b418375ffc3895757ef275e86199.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/0a7d0a51e013fd5770f1422be34aa4c03685b418375ffc3895757ef275e86199.jpg)

![0adeb8143a117fc5558f55a203d8abebf21ed19518769c0c0bc668551c3c5016.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/0adeb8143a117fc5558f55a203d8abebf21ed19518769c0c0bc668551c3c5016.jpg)

![20fef0984aae7e79c84ffe9c98fe040d0ab939f587991546e86b320378f18122.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/20fef0984aae7e79c84ffe9c98fe040d0ab939f587991546e86b320378f18122.jpg)

![37b25010697e30867b1163a62ce204225a889a8ca40152407f709a2d05569970.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/37b25010697e30867b1163a62ce204225a889a8ca40152407f709a2d05569970.jpg)

![415b7641af02a58d95d5f77a242a75c4a50d7e4a29897b8f3ef51dc60ff49489.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/415b7641af02a58d95d5f77a242a75c4a50d7e4a29897b8f3ef51dc60ff49489.jpg)

![463f2e9395ca10e113d1577f22c28b4c3acb004084c78f2f8e88883327bd8832.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/463f2e9395ca10e113d1577f22c28b4c3acb004084c78f2f8e88883327bd8832.jpg)

![4944a4af252c181cb06847d1faad53700cd6e7cebe1b790bcc766f1871868c71.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/4944a4af252c181cb06847d1faad53700cd6e7cebe1b790bcc766f1871868c71.jpg)

![4ce377308b5d77fd0d5efca7df2b3f4750043facbe639f79dc08a6d0b3781c39.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/4ce377308b5d77fd0d5efca7df2b3f4750043facbe639f79dc08a6d0b3781c39.jpg)

![5b2c43d855a2ff3487db3c663f0429db6f1341511da422227655535e1e90da4a.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/5b2c43d855a2ff3487db3c663f0429db6f1341511da422227655535e1e90da4a.jpg)

![74834568a1be3510986488eea517d341cf214a6b4d23e035ee1a9c6226cb2e0e.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/74834568a1be3510986488eea517d341cf214a6b4d23e035ee1a9c6226cb2e0e.jpg)

![88b3586eeef4685b0a295dadf18f46c4885e3dbe5aebd049c8a51ed89eca7932.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/88b3586eeef4685b0a295dadf18f46c4885e3dbe5aebd049c8a51ed89eca7932.jpg)

![9d12f5a1184fef8b043c78e275b17e4dac4a87bb72a40f25e2c11355b4cb1605.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/9d12f5a1184fef8b043c78e275b17e4dac4a87bb72a40f25e2c11355b4cb1605.jpg)

![ba5569c57a85dd933c66043e0331f5a30fb69646a4b5361299006153cf04b9e6.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/ba5569c57a85dd933c66043e0331f5a30fb69646a4b5361299006153cf04b9e6.jpg)

![dc58690da36dd29d6c7ef4002775f6860cbeaa125f66cb42c38ec9260f18adeb.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/dc58690da36dd29d6c7ef4002775f6860cbeaa125f66cb42c38ec9260f18adeb.jpg)

![dd0cc934921e097068eca09e87788d0792829cc8c7db4a966b4194cfcbfcbeb1.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/dd0cc934921e097068eca09e87788d0792829cc8c7db4a966b4194cfcbfcbeb1.jpg)

![ded591c664c4e10160676e1381c32b2508753fd3356fc369096bd2349cb0b0a8.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/ded591c664c4e10160676e1381c32b2508753fd3356fc369096bd2349cb0b0a8.jpg)

![e89625ea15cd94200b639e06cb1f5dcf906e8756e8be550dd8b93f650e228653.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/e89625ea15cd94200b639e06cb1f5dcf906e8756e8be550dd8b93f650e228653.jpg)

![ea84c3aa9ef9479c49c6cc857093747e0156d66e1f369a0ac91bc8fc8c5e3cbd.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/ea84c3aa9ef9479c49c6cc857093747e0156d66e1f369a0ac91bc8fc8c5e3cbd.jpg)

![f3a3c179c4a27e611a934c1003bab93cb1208df456097c96a38a80b19ca1bab3.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/f3a3c179c4a27e611a934c1003bab93cb1208df456097c96a38a80b19ca1bab3.jpg)

![f4079bde452087cd4ed59b39e49ec4835507ec87b8c9bb9111820796c5af3c38.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/f4079bde452087cd4ed59b39e49ec4835507ec87b8c9bb9111820796c5af3c38.jpg)

![fc6ae76f041c0c2be8372837fec6e4f3943be3a5a22c72c848d2b4775080f92f.jpg](../icml_results/4968_WorldTravel_%20A%20Realistic%20Multimodal%20Travel-Planning%20Benchmark%20with%20Tightly%20Coupled%20Constraints/tables/fc6ae76f041c0c2be8372837fec6e4f3943be3a5a22c72c848d2b4775080f92f.jpg)

<a id="paper-4921"></a>

## KAST-BAR: Knowledge-Anchored Semantically-Dynamic Topology Brain Autoregressive Modeling for Universal Neural Interpretation


### Images

![0d73d50f9a48549a6e7e9e5e214185696ccf0ebd6158a17e3d3b79a9fd2bb2e8.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/0d73d50f9a48549a6e7e9e5e214185696ccf0ebd6158a17e3d3b79a9fd2bb2e8.jpg)

![21f5b53059a8eed9a586091d670de24e2dce9b46de1bf71e3007e3c2be072d8e.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/21f5b53059a8eed9a586091d670de24e2dce9b46de1bf71e3007e3c2be072d8e.jpg)

![4d045fbb2f44aa1ef3db35cb2df3394c4bad37baae18b8c3e048e369f1908b37.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/4d045fbb2f44aa1ef3db35cb2df3394c4bad37baae18b8c3e048e369f1908b37.jpg)

![51c79612769d3a171fda030b3ac265ce361fe9034c43e2e2822cb55533866b94.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/51c79612769d3a171fda030b3ac265ce361fe9034c43e2e2822cb55533866b94.jpg)

![68d79db186d5d852f2adaf6473e1142c153db6cb89dcb3f360499f532853926a.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/68d79db186d5d852f2adaf6473e1142c153db6cb89dcb3f360499f532853926a.jpg)

![74996011c333d2bab2ce5ed8fa15d9be6458f9597a6e9c4953803fcad9be2aba.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/74996011c333d2bab2ce5ed8fa15d9be6458f9597a6e9c4953803fcad9be2aba.jpg)

![86a65137e65bb297ac046b4a9a5b93318aabad37f23710a6375027e2b15c7b1a.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/86a65137e65bb297ac046b4a9a5b93318aabad37f23710a6375027e2b15c7b1a.jpg)

![96e09656729678a83148cd31a4e2c18b661f8a6c36d3a0fc6622a07321662660.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/96e09656729678a83148cd31a4e2c18b661f8a6c36d3a0fc6622a07321662660.jpg)

![b25f705e44435602881144f78bcb911ee6adb8facb59b813578d66fa6c84597e.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/b25f705e44435602881144f78bcb911ee6adb8facb59b813578d66fa6c84597e.jpg)

![b65f68888337ee67c467d3cad8de3bbd86c17e13c04eaa27d80f12b56fdeb46c.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/b65f68888337ee67c467d3cad8de3bbd86c17e13c04eaa27d80f12b56fdeb46c.jpg)

![d59c70a8266553cdc8eb824ddf1b766f2aa71049d369d6229a87d76f686fe164.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/d59c70a8266553cdc8eb824ddf1b766f2aa71049d369d6229a87d76f686fe164.jpg)

![e08847e002a4f953e0a325c464e0067533c8b720bfe5f7cd34082e07d8964059.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/e08847e002a4f953e0a325c464e0067533c8b720bfe5f7cd34082e07d8964059.jpg)

![eb81066a01b4212aaca08a58ad24ac2565f47bc5a08ad9aac86469cfef69dceb.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/images/eb81066a01b4212aaca08a58ad24ac2565f47bc5a08ad9aac86469cfef69dceb.jpg)

### Tables

![37ccfcc70a09866c4cfd45449fb1cde155ef2a3ceda4b29b55a4055a1e13ee83.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/37ccfcc70a09866c4cfd45449fb1cde155ef2a3ceda4b29b55a4055a1e13ee83.jpg)

![466a25e1c22802d744dc81c104da27c669397ef0b47076f73640b47a3bd955db.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/466a25e1c22802d744dc81c104da27c669397ef0b47076f73640b47a3bd955db.jpg)

![6028af79dfaca4ca9b196ef6c4203737b51c2e7243028670c6c0856d9b75740c.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/6028af79dfaca4ca9b196ef6c4203737b51c2e7243028670c6c0856d9b75740c.jpg)

![687006f0aead4df8034a0e2230dc566801bd902e0360df994d8104272b17bf65.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/687006f0aead4df8034a0e2230dc566801bd902e0360df994d8104272b17bf65.jpg)

![7e2d2e59033a0f720f51776a4f1ecc0de2eba38311de484654244597c35c48af.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/7e2d2e59033a0f720f51776a4f1ecc0de2eba38311de484654244597c35c48af.jpg)

![9ce2c00be34556a66a4409409644913c3b675f29a5cea84ea7e3c543f1f9c4c9.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/9ce2c00be34556a66a4409409644913c3b675f29a5cea84ea7e3c543f1f9c4c9.jpg)

![a101e1af9dd241c6a701c5e8a1edd41608419c55d530498ba5def921dffd9c65.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/a101e1af9dd241c6a701c5e8a1edd41608419c55d530498ba5def921dffd9c65.jpg)

![b00d28f0c32f4f9eb445775d1e5d9f3b5200d5f335ce81507fd9b24901ba0678.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/b00d28f0c32f4f9eb445775d1e5d9f3b5200d5f335ce81507fd9b24901ba0678.jpg)

![c032febf54471d71c9eee6e53c6c1dae94c19045e9697d66228ad9febc085709.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/c032febf54471d71c9eee6e53c6c1dae94c19045e9697d66228ad9febc085709.jpg)

![e8391b9da1eb1d47c5ebfc918f2b3e0fc02a4b90e6765b5c5d95790d2076ddb1.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/e8391b9da1eb1d47c5ebfc918f2b3e0fc02a4b90e6765b5c5d95790d2076ddb1.jpg)

![eecd6f24cb99127d8941314fd2c1b4a83296477c5962d3f3a97760b7ce3321d5.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/eecd6f24cb99127d8941314fd2c1b4a83296477c5962d3f3a97760b7ce3321d5.jpg)

![f154970c748f73478b86e04ffb534abb3f68b50a92b6fedf2b72ec9f04b06656.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/f154970c748f73478b86e04ffb534abb3f68b50a92b6fedf2b72ec9f04b06656.jpg)

![f2284844e494600a8713fa7f38139a38c04dd26b50690f143df38899a4791b8b.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/f2284844e494600a8713fa7f38139a38c04dd26b50690f143df38899a4791b8b.jpg)

![f7b9a47f4bd1a2497ea5e7eed05ae6c549419fd1b6037357123576d590dadb7d.jpg](../icml_results/4969_KAST-BAR_%20Knowledge-Anchored%20Semantically-Dynamic%20Topology%20Brain%20Autoregressive%20Modeling%20for%20Univers/tables/f7b9a47f4bd1a2497ea5e7eed05ae6c549419fd1b6037357123576d590dadb7d.jpg)

<a id="paper-4922"></a>

## Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning


### Images

![326c53ec4f77f9077a253f387dec1c34b009b0e263a584b4fe1be9a579104cfb.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/images/326c53ec4f77f9077a253f387dec1c34b009b0e263a584b4fe1be9a579104cfb.jpg)

![6f2ca95489ea9875bc3caa4a8e5f51d613e08240b4f52c73dcd8b0d477575cb1.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/images/6f2ca95489ea9875bc3caa4a8e5f51d613e08240b4f52c73dcd8b0d477575cb1.jpg)

![787b2563b9785726d36ef1274c636f14a1e4e924183ec7aa4b023854d0553818.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/images/787b2563b9785726d36ef1274c636f14a1e4e924183ec7aa4b023854d0553818.jpg)

![8ff687a3889de2329114ad3213a843caf7885a7516ffd21888aef14cf9a1385c.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/images/8ff687a3889de2329114ad3213a843caf7885a7516ffd21888aef14cf9a1385c.jpg)

![c518f265c7d6dbebc1dfc509d3ed19c5fae5c129ba1c8538a0fb3126de6a5348.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/images/c518f265c7d6dbebc1dfc509d3ed19c5fae5c129ba1c8538a0fb3126de6a5348.jpg)

### Tables

![0b746a96780e167bd4144623c4a69a6bd7a0e91ed78f8117b55a70e43aa37f80.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/0b746a96780e167bd4144623c4a69a6bd7a0e91ed78f8117b55a70e43aa37f80.jpg)

![1a9ffc3b9b2380c8e1a87eed3357a8317155241489020f899ab4f216b2f4bb66.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/1a9ffc3b9b2380c8e1a87eed3357a8317155241489020f899ab4f216b2f4bb66.jpg)

![2639e8a3c941daee5e1b12e19d3df7d196bba9b09537994557c7318be6aa587f.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/2639e8a3c941daee5e1b12e19d3df7d196bba9b09537994557c7318be6aa587f.jpg)

![4973ced1aa1ff039c2e4a44dd8771527d52cb5cadedb7494e5f5a6a2cbb41cde.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/4973ced1aa1ff039c2e4a44dd8771527d52cb5cadedb7494e5f5a6a2cbb41cde.jpg)

![6561f6b8afd66590b7b78f39caccff79814411dd5f1cf4cb1145d849db7ead92.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/6561f6b8afd66590b7b78f39caccff79814411dd5f1cf4cb1145d849db7ead92.jpg)

![aacbf9c1254511d83db0b78526dacabda2ac202cc5db9d7afb27882b79ce55c5.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/aacbf9c1254511d83db0b78526dacabda2ac202cc5db9d7afb27882b79ce55c5.jpg)

![aea7986cd3c65017f803e83c77de0828b7dc514322e49b6c152f24837a6fb747.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/aea7986cd3c65017f803e83c77de0828b7dc514322e49b6c152f24837a6fb747.jpg)

![d8d7bc9eba6cb3e7b601ba7fe02e1558bab20db52c503eaf0b5e7ac9a170809a.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/d8d7bc9eba6cb3e7b601ba7fe02e1558bab20db52c503eaf0b5e7ac9a170809a.jpg)

![dfb7aa9ce764e6b22591c89b2d6d1fbf9dd8a62663419b96ca64b796e0302456.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/dfb7aa9ce764e6b22591c89b2d6d1fbf9dd8a62663419b96ca64b796e0302456.jpg)

![e47a3b88900875087cbc7baa86cb89f9886fbb0265d3481d1676bb6a66640f3f.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/e47a3b88900875087cbc7baa86cb89f9886fbb0265d3481d1676bb6a66640f3f.jpg)

![ed8c05634ef1d230487c3bf60bf2e985f6a665cd489aca819ed96089a2487c6d.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/ed8c05634ef1d230487c3bf60bf2e985f6a665cd489aca819ed96089a2487c6d.jpg)

![f276c645a8739c9da499c6d5516c4c0e4e3d1047aa5b5f710e1a167316412ba1.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/f276c645a8739c9da499c6d5516c4c0e4e3d1047aa5b5f710e1a167316412ba1.jpg)

![fcf8cc88ecf916897825a44b907db188008d787cd788dbf4397bff002ad7d1e1.jpg](../icml_results/4970_Beyond%20Test-Time%20Memory_%20State-Space%20Optimal%20Control%20for%20LLM%20Reasoning/tables/fcf8cc88ecf916897825a44b907db188008d787cd788dbf4397bff002ad7d1e1.jpg)

<a id="paper-4923"></a>

## Towards Context-Invariant Safety Alignment for Large Language Models


### Images

![1143517f5fc2d1f8637375d41b563caac7e93879c2d7a7a39578bc55ce928387.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/images/1143517f5fc2d1f8637375d41b563caac7e93879c2d7a7a39578bc55ce928387.jpg)

![8c757bf3b93c94408cdff549de2ae749d7d4e20550c8f11c206ea622231a9aa8.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/images/8c757bf3b93c94408cdff549de2ae749d7d4e20550c8f11c206ea622231a9aa8.jpg)

![9f6b20bb1ab584e727cd7d1b0c1bc8132c0b53d1e10790127287836973c20c19.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/images/9f6b20bb1ab584e727cd7d1b0c1bc8132c0b53d1e10790127287836973c20c19.jpg)

![aba38825d12c4298e3449b15ae189a9506ea7fcaad36469fee393fa66861e624.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/images/aba38825d12c4298e3449b15ae189a9506ea7fcaad36469fee393fa66861e624.jpg)

![d4186fb20096da58741093e9c819305fc0c745ea77e98f3c745ad5c878ff90b9.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/images/d4186fb20096da58741093e9c819305fc0c745ea77e98f3c745ad5c878ff90b9.jpg)

### Tables

![c5d8dcfbdb8b9b579a8bd683c523debaf4f5d45bdf94035783e4187e50e2c4b0.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/tables/c5d8dcfbdb8b9b579a8bd683c523debaf4f5d45bdf94035783e4187e50e2c4b0.jpg)

![df26c0bce91dc06f042877f40dd4602a36a0c06f10d10231cd3efccc38005480.jpg](../icml_results/4971_Towards%20Context-Invariant%20Safety%20Alignment%20for%20Large%20Language%20Models/tables/df26c0bce91dc06f042877f40dd4602a36a0c06f10d10231cd3efccc38005480.jpg)

<a id="paper-4924"></a>

## Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model


### Images

![035774cd712707d349f11ae989c717361b86758510d728747e5ddbf8b6efe8dd.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/035774cd712707d349f11ae989c717361b86758510d728747e5ddbf8b6efe8dd.jpg)

![14d6a8e7ea66bb8910616f7b6590b59d5d551c45b1e393f5ccb24f178f4d5548.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/14d6a8e7ea66bb8910616f7b6590b59d5d551c45b1e393f5ccb24f178f4d5548.jpg)

![36b4dbbd021edf98de660e2f911c89109dee5e2f1ed5f78bfa8807b31371adcc.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/36b4dbbd021edf98de660e2f911c89109dee5e2f1ed5f78bfa8807b31371adcc.jpg)

![98a938327fbbb81d94735df5dbec50507c429cffc4b38ae41a334e02dde360a7.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/98a938327fbbb81d94735df5dbec50507c429cffc4b38ae41a334e02dde360a7.jpg)

![a8937f173485cc0774b656784d77bb4194f3a793bdd38771dbba6fa4e070d0b3.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/a8937f173485cc0774b656784d77bb4194f3a793bdd38771dbba6fa4e070d0b3.jpg)

![d103a7f06d083f3c1bce9074d42212b6d540bc8f421420c81d545140e348339d.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/d103a7f06d083f3c1bce9074d42212b6d540bc8f421420c81d545140e348339d.jpg)

![d8db9ef35307ea930ae6b2a999d3883c37fc81739bf0eee2ca779545d2e2d1d9.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/images/d8db9ef35307ea930ae6b2a999d3883c37fc81739bf0eee2ca779545d2e2d1d9.jpg)

### Tables

![1e637ca190aa68907e31839e5ceaaac154f8fafac39cc7b26b51df732e6d8293.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/1e637ca190aa68907e31839e5ceaaac154f8fafac39cc7b26b51df732e6d8293.jpg)

![29fc5adc3155fe69e74272d691fc8319f68f018b93a3cb68f95fa3ef292f8106.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/29fc5adc3155fe69e74272d691fc8319f68f018b93a3cb68f95fa3ef292f8106.jpg)

![36f4240d05c3478a9b30959f433b125e1bef488febf8879cf7847e9fd4690e34.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/36f4240d05c3478a9b30959f433b125e1bef488febf8879cf7847e9fd4690e34.jpg)

![6d337d6a4860acfb79293075c3d9fba77983d6b325326f26c2fe525de99a1f23.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/6d337d6a4860acfb79293075c3d9fba77983d6b325326f26c2fe525de99a1f23.jpg)

![6fa9af91917c88f40ed855e4146a059b411c858db243ff4ee304a9c9aea18cad.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/6fa9af91917c88f40ed855e4146a059b411c858db243ff4ee304a9c9aea18cad.jpg)

![7a925b6e089b93d6e7ab8591d9164e6d99840167502812c82c13478520ac5f1e.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/7a925b6e089b93d6e7ab8591d9164e6d99840167502812c82c13478520ac5f1e.jpg)

![a2f5c9d3b761189479fb25550eac3877eebec5bbfa2dc0f15ee90c337c239006.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/a2f5c9d3b761189479fb25550eac3877eebec5bbfa2dc0f15ee90c337c239006.jpg)

![a67a80aa4f8997ea86610dafc1d48b44965db8c9011010e6baa3f2afcf2bec55.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/a67a80aa4f8997ea86610dafc1d48b44965db8c9011010e6baa3f2afcf2bec55.jpg)

![aba8b64dc33c841eefd97b2dc6f25cf81eea3ba9f6abc9dbf6a0e5843465a5f5.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/aba8b64dc33c841eefd97b2dc6f25cf81eea3ba9f6abc9dbf6a0e5843465a5f5.jpg)

![c3c7b8251a0b58de5683431aa3363ad5648d59a1ded18971bc8adf813574666b.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/c3c7b8251a0b58de5683431aa3363ad5648d59a1ded18971bc8adf813574666b.jpg)

![c6b1b924463b0457832a0fa2f39c9f94cafe6c01a6b6724cbd1b1a9dc10df79e.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/c6b1b924463b0457832a0fa2f39c9f94cafe6c01a6b6724cbd1b1a9dc10df79e.jpg)

![d51751a5f1736e8033a03a653d8ddfad46656df56daaafbd0e3eff3e32f5ebbf.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/d51751a5f1736e8033a03a653d8ddfad46656df56daaafbd0e3eff3e32f5ebbf.jpg)

![e8b46ea795b8e478b0d1cc51480a35675e14636450b0c7842566ab4c6cf1ac02.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/e8b46ea795b8e478b0d1cc51480a35675e14636450b0c7842566ab4c6cf1ac02.jpg)

![f1a65a8b2db3ca5b6d150962b528058d324935577e07860883de1e5f8852ae16.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/f1a65a8b2db3ca5b6d150962b528058d324935577e07860883de1e5f8852ae16.jpg)

![f23a3fb34de5d36f963e67739fd0f86a58afc33db7c7c767ce44f386117bd187.jpg](../icml_results/4972_Omni-fMRI_%20A%20Universal%20Atlas-Free%20fMRI%20Foundation%20Model/tables/f23a3fb34de5d36f963e67739fd0f86a58afc33db7c7c767ce44f386117bd187.jpg)

<a id="paper-4925"></a>

## MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation


### Images

![0bd67bd6828851a7990071da5d5c5cc5843437965ef10a26cc1caa11fec322d4.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/0bd67bd6828851a7990071da5d5c5cc5843437965ef10a26cc1caa11fec322d4.jpg)

![2f861c8b52fa69b06223b38a79c6f54cce5fb6dea348c4d53805fbfcd133083b.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/2f861c8b52fa69b06223b38a79c6f54cce5fb6dea348c4d53805fbfcd133083b.jpg)

![320e56253c8675d3f2012dfe326d566ffecf0ed7655550b37d2031e3a1bd28b5.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/320e56253c8675d3f2012dfe326d566ffecf0ed7655550b37d2031e3a1bd28b5.jpg)

![6a862e5eaa4dad1b1759916d528ee840463918bd5fbcc5dc8863e44b610037f4.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/6a862e5eaa4dad1b1759916d528ee840463918bd5fbcc5dc8863e44b610037f4.jpg)

![765b839f8cde4bda80ece7977ce145b4fc05e25b05a12d3ee221398fbfbb4af0.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/765b839f8cde4bda80ece7977ce145b4fc05e25b05a12d3ee221398fbfbb4af0.jpg)

![933683aab6f510db3722220cc928f9a275ea1845afc30398f8fea431df717ed2.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/933683aab6f510db3722220cc928f9a275ea1845afc30398f8fea431df717ed2.jpg)

![969db0612e0db62012467530b7d3caba04f52d5d3e152fc8218f07b020447904.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/969db0612e0db62012467530b7d3caba04f52d5d3e152fc8218f07b020447904.jpg)

![a2e16ce3caa84e0e4828d6f9df0a7a3ee281dab2e82b04e260156dc1afaaf122.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/a2e16ce3caa84e0e4828d6f9df0a7a3ee281dab2e82b04e260156dc1afaaf122.jpg)

![a78fa4df41f6505f3cff8f34b2ba22b05a267d11b62ee860d0d3f57be753728f.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/a78fa4df41f6505f3cff8f34b2ba22b05a267d11b62ee860d0d3f57be753728f.jpg)

![bc766d5bd256b9c1e90cb8effc2587a2f879016c9b02a6507bf1978c28ace7af.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/bc766d5bd256b9c1e90cb8effc2587a2f879016c9b02a6507bf1978c28ace7af.jpg)

![c1c94966ac14ef691af61290aae756d328191f6f4f40c75eaecce8ece512ea61.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/c1c94966ac14ef691af61290aae756d328191f6f4f40c75eaecce8ece512ea61.jpg)

![c4d705b32cff0d1e9ad24d3796ef45c743a280058e9716f7198a558a534e0f72.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/c4d705b32cff0d1e9ad24d3796ef45c743a280058e9716f7198a558a534e0f72.jpg)

![e266cfbcd2a8401f715012ff2bf74a2e087f0311fa02a8657327d5b136c059df.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/e266cfbcd2a8401f715012ff2bf74a2e087f0311fa02a8657327d5b136c059df.jpg)

![e81d9552aaedf6001bfbdcb62879bfdfb80298a44d27c8f456efb598e93f35e8.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/e81d9552aaedf6001bfbdcb62879bfdfb80298a44d27c8f456efb598e93f35e8.jpg)

![ec75d35cf5d711bfcf86d128578dda27ce20bdafae23d67252990996702522c8.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/ec75d35cf5d711bfcf86d128578dda27ce20bdafae23d67252990996702522c8.jpg)

![f9f3669bd7d752113d9ca4cac12ea47f4d9eee56e0da7d029a9541de6e2c168b.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/images/f9f3669bd7d752113d9ca4cac12ea47f4d9eee56e0da7d029a9541de6e2c168b.jpg)

### Tables

![0abe2c4026893af5ca5599c079e00aa5a2b438d65a89e802f04ad11cd88d023f.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/0abe2c4026893af5ca5599c079e00aa5a2b438d65a89e802f04ad11cd88d023f.jpg)

![29b6fec4342c9191491e91ba578ac38f6bf9dcc9671e86e85ab73d009d6d5fcf.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/29b6fec4342c9191491e91ba578ac38f6bf9dcc9671e86e85ab73d009d6d5fcf.jpg)

![3b6d01d5d6de93c2872f88323df03ff727ceb87bd4a36f2d0c9df7f169cc07d1.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/3b6d01d5d6de93c2872f88323df03ff727ceb87bd4a36f2d0c9df7f169cc07d1.jpg)

![47c6b33bb5e7f385bb133949a98a5d6f16bfd00d419bdd89e44d7ba129f9890c.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/47c6b33bb5e7f385bb133949a98a5d6f16bfd00d419bdd89e44d7ba129f9890c.jpg)

![4bea05c63356a8f329b72141eaa4ce4925bb2866e614119fda162fb8988517c7.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/4bea05c63356a8f329b72141eaa4ce4925bb2866e614119fda162fb8988517c7.jpg)

![579c753394c15c7c87f4c883f6fbdcc99978af991dc3e37e2a46ee8284168707.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/579c753394c15c7c87f4c883f6fbdcc99978af991dc3e37e2a46ee8284168707.jpg)

![6168541ab836ec8307b85dbeaa59ae56b11452cf6f02e856b292dd1bb13da962.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/6168541ab836ec8307b85dbeaa59ae56b11452cf6f02e856b292dd1bb13da962.jpg)

![7e460dcd0cf65aa80766b47de08b2008d298ba2b12738911aa019edf41d0f655.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/7e460dcd0cf65aa80766b47de08b2008d298ba2b12738911aa019edf41d0f655.jpg)

![8b87d6891f1d55fd8344f9fd197df0694763f495276d95b789eb3fa5e209b51b.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/8b87d6891f1d55fd8344f9fd197df0694763f495276d95b789eb3fa5e209b51b.jpg)

![8e063b2bb1590d614afd90af064d1e2197ad703375c88c287ec93bd40c8c3b89.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/8e063b2bb1590d614afd90af064d1e2197ad703375c88c287ec93bd40c8c3b89.jpg)

![a6ad47afc4dafb60dfd38f958deb1f753805d6e76974fc055dff863f0c0ced01.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/a6ad47afc4dafb60dfd38f958deb1f753805d6e76974fc055dff863f0c0ced01.jpg)

![c1cee11ba715e3fcc629c4e5e7f0ea30e2776adca713316f441e037a0b076bd3.jpg](../icml_results/4973_MVISTA-4D_%20View-Consistent%204D%20World%20Model%20with%20Test-Time%20Action%20Inference%20for%20Robotic%20Manipulation/tables/c1cee11ba715e3fcc629c4e5e7f0ea30e2776adca713316f441e037a0b076bd3.jpg)

<a id="paper-4926"></a>

## $\mathbbR^2k$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval


### Images

![02712effd9795e369ce830012e112d59101a497364af1ccf5df6a0956350d0f3.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/images/02712effd9795e369ce830012e112d59101a497364af1ccf5df6a0956350d0f3.jpg)

![28eaf35628b42ab92a02e2ba45ece47afb9dd5301bb833b9e6b5b14e7ddff427.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/images/28eaf35628b42ab92a02e2ba45ece47afb9dd5301bb833b9e6b5b14e7ddff427.jpg)

### Tables

![4716fb29580a305e77a083ff9bcd2622bc6e2349650ca523673001184959e969.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/tables/4716fb29580a305e77a083ff9bcd2622bc6e2349650ca523673001184959e969.jpg)

![528e38f4265d4e0e0ae1207d67e0eb47afe06ad2d283a5f9f12ee9c7cdbe1c2f.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/tables/528e38f4265d4e0e0ae1207d67e0eb47afe06ad2d283a5f9f12ee9c7cdbe1c2f.jpg)

![78183580117ec26c6e3d1fd8ae8b6b935707abccb93ffee95ef6b380d58be500.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/tables/78183580117ec26c6e3d1fd8ae8b6b935707abccb93ffee95ef6b380d58be500.jpg)

![d4acd7d3307af5ced2569ef622b9873d2650caf456110ae88b3c3d2f28cbd3da.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/tables/d4acd7d3307af5ced2569ef622b9873d2650caf456110ae88b3c3d2f28cbd3da.jpg)

![dd50c494d5398c8f1942f3ca4de65caa22ad6ce2f201b3a630aeb97b53747ac7.jpg](../icml_results/4974_%24_mathbbR%5E2k%24%20is%20Theoretically%20Large%20Enough%20for%20Embedding-based%20Top-%24k%24%20Retrieval/tables/dd50c494d5398c8f1942f3ca4de65caa22ad6ce2f201b3a630aeb97b53747ac7.jpg)

<a id="paper-4927"></a>

## GenExam: A Multidisciplinary Text-to-Image Exam


### Images

![01193f6153b28031a1b537b57442a9fe500cc189a74c46dd4844b3758f801fd9.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/01193f6153b28031a1b537b57442a9fe500cc189a74c46dd4844b3758f801fd9.jpg)

![092aabf13a014e9efc1c9ab702feea786bba1f2e5e5039c96eea19a03d137e65.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/092aabf13a014e9efc1c9ab702feea786bba1f2e5e5039c96eea19a03d137e65.jpg)

![73480cbf7a12bdbe76dde50ad65aba0dd62ea4bd5a1d97e5db6f04e5f1713628.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/73480cbf7a12bdbe76dde50ad65aba0dd62ea4bd5a1d97e5db6f04e5f1713628.jpg)

![7388449d049271648c1140fdd2927c5f58d6565f57cf293db2f7e44e7a3448e4.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/7388449d049271648c1140fdd2927c5f58d6565f57cf293db2f7e44e7a3448e4.jpg)

![7619b3539915301b4290e63313bb8cd42a1929d03272b8e435749313901ff98a.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/7619b3539915301b4290e63313bb8cd42a1929d03272b8e435749313901ff98a.jpg)

![7aa085f23335a1029024394128987aa1f99b7c977835b322a1025857daee5cd4.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/7aa085f23335a1029024394128987aa1f99b7c977835b322a1025857daee5cd4.jpg)

![8ac6b63f378f90425b11d83799aeca1d5f9cdba5a4ec6191903dd9f2824ce618.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/8ac6b63f378f90425b11d83799aeca1d5f9cdba5a4ec6191903dd9f2824ce618.jpg)

![8c78f65090d177127173927453287fe6a5c5f2170081e060439495a8221a3ca6.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/8c78f65090d177127173927453287fe6a5c5f2170081e060439495a8221a3ca6.jpg)

![b18f27f005cc8982f1a41120b2a07d70c7447724dfced635c7136a1517b4a349.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/b18f27f005cc8982f1a41120b2a07d70c7447724dfced635c7136a1517b4a349.jpg)

![b50b857dd8b3beb90e913aa43fee66ec9cf101cf93e979fd94f4eb303c2df765.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/b50b857dd8b3beb90e913aa43fee66ec9cf101cf93e979fd94f4eb303c2df765.jpg)

![b5bd76eb2bfa65476a8d155690df14e1d35310fee7d057c24589891562587e33.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/b5bd76eb2bfa65476a8d155690df14e1d35310fee7d057c24589891562587e33.jpg)

![c255d949c8e9a97b2236c538c0868668cf10e41772d09ce36137494ef4904b0e.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/c255d949c8e9a97b2236c538c0868668cf10e41772d09ce36137494ef4904b0e.jpg)

![c4bc6e889dd523bd6f6759d967b61767163a107bc33d22ce4b565a680ddfd823.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/c4bc6e889dd523bd6f6759d967b61767163a107bc33d22ce4b565a680ddfd823.jpg)

![d22657ce8127c43d73a1d6f114335f4e13c19c7473f26b20475b351e1bf2f66a.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/d22657ce8127c43d73a1d6f114335f4e13c19c7473f26b20475b351e1bf2f66a.jpg)

![da3cff2275be3d5f11b78c4de90b0cceef2f52c31005019d6f32b9dab920d209.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/da3cff2275be3d5f11b78c4de90b0cceef2f52c31005019d6f32b9dab920d209.jpg)

![dfbf682c64624dae1dee5779c74f123b351598512d700ca07a1b6819d6ecd9fc.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/dfbf682c64624dae1dee5779c74f123b351598512d700ca07a1b6819d6ecd9fc.jpg)

![eb4e966e60b642902e7b387d93c1694d9644bda324c5708e2f66708cc15748b9.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/eb4e966e60b642902e7b387d93c1694d9644bda324c5708e2f66708cc15748b9.jpg)

![f047c23f62d99149539c4357cc60e9f43d643610e604f56d372c7d6d6ac2018e.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/f047c23f62d99149539c4357cc60e9f43d643610e604f56d372c7d6d6ac2018e.jpg)

![f69cb6f20ac78c7c0b6ec33a04905d63925c34d5fb8182d6a7b732bb45c98af0.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/f69cb6f20ac78c7c0b6ec33a04905d63925c34d5fb8182d6a7b732bb45c98af0.jpg)

![fb2df48ed9e6acbbc022a7cd13f78fbdd40dbde2fec722758c017f9969acd1c3.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/images/fb2df48ed9e6acbbc022a7cd13f78fbdd40dbde2fec722758c017f9969acd1c3.jpg)

### Tables

![1a19ffb22d6859d48213c64b431f8175fdd7c662f218708e197fdf4313a56b93.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/1a19ffb22d6859d48213c64b431f8175fdd7c662f218708e197fdf4313a56b93.jpg)

![337c9dea9783f276a0fc0ccfe94803b7aaef04c95ab21aeabf6882469aedd4a6.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/337c9dea9783f276a0fc0ccfe94803b7aaef04c95ab21aeabf6882469aedd4a6.jpg)

![3ba18c684a46ecf990ddb0a4239022e7c68e03d19a4dd9202a4572bc064e58e0.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/3ba18c684a46ecf990ddb0a4239022e7c68e03d19a4dd9202a4572bc064e58e0.jpg)

![3f456bba226873becf4fbf813eb7ae9b50d2910ee67fb102c969a995e43a5242.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/3f456bba226873becf4fbf813eb7ae9b50d2910ee67fb102c969a995e43a5242.jpg)

![421c4f524ae4b7393e2fcc6d1f25deedc30244c9b09dc654d9f8e7e88d16890a.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/421c4f524ae4b7393e2fcc6d1f25deedc30244c9b09dc654d9f8e7e88d16890a.jpg)

![5f88483fd3a8573b5c8257e24fc4b7165b6ca74b5dda5661a85389469b15b21b.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/5f88483fd3a8573b5c8257e24fc4b7165b6ca74b5dda5661a85389469b15b21b.jpg)

![672c4b63bce3bdd6c1ff9d5fdcd0d0f4ac35cd6bb774c01994098e04dfe57e15.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/672c4b63bce3bdd6c1ff9d5fdcd0d0f4ac35cd6bb774c01994098e04dfe57e15.jpg)

![8aeed090d6b9398d65b4cea80975636e86c5ea04f95046a8be70ee494d0a1809.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/8aeed090d6b9398d65b4cea80975636e86c5ea04f95046a8be70ee494d0a1809.jpg)

![a11ffbaa58edb4812794b3d2d66df15cab31de58aa70e99d75a8fe2ec7ea3ba5.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/a11ffbaa58edb4812794b3d2d66df15cab31de58aa70e99d75a8fe2ec7ea3ba5.jpg)

![a92fa87a3e4a66839019db1767a72d0a65b4b961fcb908b239f7422df2339957.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/a92fa87a3e4a66839019db1767a72d0a65b4b961fcb908b239f7422df2339957.jpg)

![aabffd0ec6d82392ef2ecc00eb3f39d5155f75f12a1a9407efecf6d4ce1ac6b4.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/aabffd0ec6d82392ef2ecc00eb3f39d5155f75f12a1a9407efecf6d4ce1ac6b4.jpg)

![ad17742ae892a7397a4c0ef5670aaa4a92e5a68fb0f1a448766f1f66737cbf59.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/ad17742ae892a7397a4c0ef5670aaa4a92e5a68fb0f1a448766f1f66737cbf59.jpg)

![c532526d26a361de6786feb09ae2084af894e17f804addf842c130960b19e5c8.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/c532526d26a361de6786feb09ae2084af894e17f804addf842c130960b19e5c8.jpg)

![c6b06df0279c7b75a7b5c585b4afd55ff640bb9213c0d57929b279c34614c6d6.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/c6b06df0279c7b75a7b5c585b4afd55ff640bb9213c0d57929b279c34614c6d6.jpg)

![dd647798cba8207d119363fb5ede7ced097820fdce7217f034978e5c8fe12f17.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/dd647798cba8207d119363fb5ede7ced097820fdce7217f034978e5c8fe12f17.jpg)

![eca72aabfffd9e986846d8f42ccdcc4621e824730ca77b45add052702fbd506c.jpg](../icml_results/4975_GenExam_%20A%20Multidisciplinary%20Text-to-Image%20Exam/tables/eca72aabfffd9e986846d8f42ccdcc4621e824730ca77b45add052702fbd506c.jpg)

<a id="paper-4928"></a>

## Relational In-Context Learning via Synthetic Pre-training with Structural Prior


### Images

![01136b06d07fb2b3348f471fdde465af25c17d506a133c8026e1677d29f8e4d0.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/01136b06d07fb2b3348f471fdde465af25c17d506a133c8026e1677d29f8e4d0.jpg)

![0a9a43ca614696d48bee02b2fcede7e74f81f3eb5ae8091997e6c9b81aff6dc2.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/0a9a43ca614696d48bee02b2fcede7e74f81f3eb5ae8091997e6c9b81aff6dc2.jpg)

![1546c2f0fc4f6108cbc0c5f38885ca09d152870c5fb19fad23d542b32a7ca971.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/1546c2f0fc4f6108cbc0c5f38885ca09d152870c5fb19fad23d542b32a7ca971.jpg)

![17a6dcd29b1fc9e23e1ec513359e1ecbb172ba001f06f6130c9b23d20cea81dd.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/17a6dcd29b1fc9e23e1ec513359e1ecbb172ba001f06f6130c9b23d20cea81dd.jpg)

![1a8042bd7b8ff746f040e4c7ce2535917ae4d210222f244e4dbe5545a5f5491c.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/1a8042bd7b8ff746f040e4c7ce2535917ae4d210222f244e4dbe5545a5f5491c.jpg)

![5dbd5ec3d18396c8da22065492b4765afc3e4282d07f2b7c02935a278d8c24e1.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/5dbd5ec3d18396c8da22065492b4765afc3e4282d07f2b7c02935a278d8c24e1.jpg)

![72a981be4a6ea1b7d11b9dad84cb93a681b4895471c94d900540f895a24bb1ad.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/72a981be4a6ea1b7d11b9dad84cb93a681b4895471c94d900540f895a24bb1ad.jpg)

![87b4258fd3ab314cb1e1107b9298388dd92b1ac646086f021f4bcc2a6b585e8f.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/87b4258fd3ab314cb1e1107b9298388dd92b1ac646086f021f4bcc2a6b585e8f.jpg)

![a4bb0ee955fc06ef467c2e1ebea80da6791d027098561c2b6842d571a01ccfdc.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/a4bb0ee955fc06ef467c2e1ebea80da6791d027098561c2b6842d571a01ccfdc.jpg)

![e0bbd4e6a00703d00fe08888058ff247b0916fb8864f527cc9b34f3229b6e806.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/e0bbd4e6a00703d00fe08888058ff247b0916fb8864f527cc9b34f3229b6e806.jpg)

![f65cd1ffb83aeea3a714cc7aba8e328f3b6dfb8a84ee5bc95688d74244350d0d.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/images/f65cd1ffb83aeea3a714cc7aba8e328f3b6dfb8a84ee5bc95688d74244350d0d.jpg)

### Tables

![02a2dc846db385be7be5391fce73d3f052e5a2343dce84708aa854715876fb45.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/02a2dc846db385be7be5391fce73d3f052e5a2343dce84708aa854715876fb45.jpg)

![0757626a6566544af921e7606023475cbff08fc63a92df111be81269e4f24eb8.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/0757626a6566544af921e7606023475cbff08fc63a92df111be81269e4f24eb8.jpg)

![09401d38be54dafb69242dea6f70ce946ae3d7009217c39388baa47eff03003c.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/09401d38be54dafb69242dea6f70ce946ae3d7009217c39388baa47eff03003c.jpg)

![0dac8a7fbe4989d8581d7f0c4efad17ac28b7f13fbfa02477131600974e51c54.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/0dac8a7fbe4989d8581d7f0c4efad17ac28b7f13fbfa02477131600974e51c54.jpg)

![11ac28a87612f1cbaaa93eafaf4b6ec7fb757fc6176defa6e3739f6ce9fe56d6.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/11ac28a87612f1cbaaa93eafaf4b6ec7fb757fc6176defa6e3739f6ce9fe56d6.jpg)

![1466dc032b7a14b8083466c9502c66df5706c2add613233786f1602f88785ef6.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/1466dc032b7a14b8083466c9502c66df5706c2add613233786f1602f88785ef6.jpg)

![1a0b976fe28b9019f8aee8d1fd636d08a368505de918a72bc65f778da0e0327b.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/1a0b976fe28b9019f8aee8d1fd636d08a368505de918a72bc65f778da0e0327b.jpg)

![2638bf43893c2fd9f2c103e9037769c05e3fa2ddc4794f99de466de27d3c4865.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/2638bf43893c2fd9f2c103e9037769c05e3fa2ddc4794f99de466de27d3c4865.jpg)

![2ad0b858a2d507ef216099d23a70fbc208bc373b8e5db5f52f476af00478dbeb.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/2ad0b858a2d507ef216099d23a70fbc208bc373b8e5db5f52f476af00478dbeb.jpg)

![38e9b6cbd589200dd531d2da2c712ef9bf18a4cb2052391f89119df2f35beb89.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/38e9b6cbd589200dd531d2da2c712ef9bf18a4cb2052391f89119df2f35beb89.jpg)

![4134d2211587dc12fea9334a4b7df368a5839520aeb57a5940f79bc2b5e3eb9b.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/4134d2211587dc12fea9334a4b7df368a5839520aeb57a5940f79bc2b5e3eb9b.jpg)

![491c1cb2d2e1c67a1549782c20c2ff0d122937329f6a84cf9382f85fe0b33988.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/491c1cb2d2e1c67a1549782c20c2ff0d122937329f6a84cf9382f85fe0b33988.jpg)

![5171de9ea4435f7a476fb949c3a5a48fa572722fd0d58bab78dcc1b0edfd725e.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/5171de9ea4435f7a476fb949c3a5a48fa572722fd0d58bab78dcc1b0edfd725e.jpg)

![52ab7cd3de2437daed78f7524209a30b656fb93619baf7d6be56d14fb18b92d8.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/52ab7cd3de2437daed78f7524209a30b656fb93619baf7d6be56d14fb18b92d8.jpg)

![572d6c2f0ab9a5188707d5fd6ddaf77ee48cf871be67e224ad6c264e047377a1.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/572d6c2f0ab9a5188707d5fd6ddaf77ee48cf871be67e224ad6c264e047377a1.jpg)

![58f2de8f1ae85288131058f1bea97d3ac650cfecce7aba13d5b80dc01d908bc4.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/58f2de8f1ae85288131058f1bea97d3ac650cfecce7aba13d5b80dc01d908bc4.jpg)

![5e05ce1398ae829771c9cddea7c0158b8cfb860109e3ad90bbb138ad0542f78c.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/5e05ce1398ae829771c9cddea7c0158b8cfb860109e3ad90bbb138ad0542f78c.jpg)

![62f5841d0b90a272ea887e4d7e15fe29b87edf1bf5f4c1cd27653555e1b89b81.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/62f5841d0b90a272ea887e4d7e15fe29b87edf1bf5f4c1cd27653555e1b89b81.jpg)

![65004f7866f7486789ebd2142cb7a9652eedc092533a6db2654495cfe2f6a669.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/65004f7866f7486789ebd2142cb7a9652eedc092533a6db2654495cfe2f6a669.jpg)

![6c49339d679631baa5deea8dbdf7eaa501ac70892a22b267d9fee08351339088.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/6c49339d679631baa5deea8dbdf7eaa501ac70892a22b267d9fee08351339088.jpg)

![6c90a0a5c78cff64ef6f7391ed8272ad283a3c5024841336c1339a7e4bed110a.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/6c90a0a5c78cff64ef6f7391ed8272ad283a3c5024841336c1339a7e4bed110a.jpg)

![79894b459d40cea4d460c6c89114534727c215487a10b03039d52355304fdfb0.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/79894b459d40cea4d460c6c89114534727c215487a10b03039d52355304fdfb0.jpg)

![7e745676b4564d5225890e9ca82d49f2d40ddc1093cba15f88f65b0c53419d8a.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/7e745676b4564d5225890e9ca82d49f2d40ddc1093cba15f88f65b0c53419d8a.jpg)

![8861d89156aa22552db3fd53375e7595a640fcb9a0c852c66a6cf8569d9b3971.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/8861d89156aa22552db3fd53375e7595a640fcb9a0c852c66a6cf8569d9b3971.jpg)

![8dcc90556cc230df0061b5bffff1ce8129d3f45a02c69aafb9e9e517368f8a52.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/8dcc90556cc230df0061b5bffff1ce8129d3f45a02c69aafb9e9e517368f8a52.jpg)

![942fc4930c961905a31b7436af2ad760df2ff209fe3421a321dd5e0f75401543.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/942fc4930c961905a31b7436af2ad760df2ff209fe3421a321dd5e0f75401543.jpg)

![a42eb6695b15f6d54255227af0749d8c0cc9a746411a75539ab2cf26e6ebd8bd.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/a42eb6695b15f6d54255227af0749d8c0cc9a746411a75539ab2cf26e6ebd8bd.jpg)

![ab6cf3afca3baefb2d2c9a54ba6b627536ac31e426d81d687e73693e4a362f83.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/ab6cf3afca3baefb2d2c9a54ba6b627536ac31e426d81d687e73693e4a362f83.jpg)

![ad8b2a0d477be6ec99930d19110d163b177aa8f842135632ebc97b1fb19f3cf4.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/ad8b2a0d477be6ec99930d19110d163b177aa8f842135632ebc97b1fb19f3cf4.jpg)

![b29a1f49fac7cfbb86f06ced7fd92d6daad1f22af31082fa0b4ca9745f501eb5.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/b29a1f49fac7cfbb86f06ced7fd92d6daad1f22af31082fa0b4ca9745f501eb5.jpg)

![b41fb908be6ed9d0b14580fe5624c39ff8bbf661850be04685a3e5658f42f8e7.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/b41fb908be6ed9d0b14580fe5624c39ff8bbf661850be04685a3e5658f42f8e7.jpg)

![b48c286341f773d601f1be23cadece4e5335ccd738ae90cd2ca863ee88083db6.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/b48c286341f773d601f1be23cadece4e5335ccd738ae90cd2ca863ee88083db6.jpg)

![b63d7e123755b8ba14b699bf3146b298ddc6136ad5af531f1c89a396075256ae.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/b63d7e123755b8ba14b699bf3146b298ddc6136ad5af531f1c89a396075256ae.jpg)

![ba1609b9df6088c4a0c412a59ad990db3f016de51713e2a40a887f80f0cc4892.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/ba1609b9df6088c4a0c412a59ad990db3f016de51713e2a40a887f80f0cc4892.jpg)

![c053d23d65b07a1228224516f00ee9f1c5f6ac4ef7f4d966b11174ace12c08f4.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/c053d23d65b07a1228224516f00ee9f1c5f6ac4ef7f4d966b11174ace12c08f4.jpg)

![dc59b9a1f4937a777e805bcb85d3952b215bf6ab54f03f7c21f0a6eee2a7b99a.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/dc59b9a1f4937a777e805bcb85d3952b215bf6ab54f03f7c21f0a6eee2a7b99a.jpg)

![dfea40a6bbe589e9d595ed323fc890a113fd86ad22f34a837420b347820a7353.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/dfea40a6bbe589e9d595ed323fc890a113fd86ad22f34a837420b347820a7353.jpg)

![e41daf52a5359c18eaccdb486f1f0809fdddabd6580f891e11a0110dc0132e88.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/e41daf52a5359c18eaccdb486f1f0809fdddabd6580f891e11a0110dc0132e88.jpg)

![f2cfa12e06b253e25367ba20f7258ff25e216f641cad8e561d1cb3b73a6ce145.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/f2cfa12e06b253e25367ba20f7258ff25e216f641cad8e561d1cb3b73a6ce145.jpg)

![f70b535d6c32a433c9358be5cd88469b3958dedb91fe25485288e9439674c2a1.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/f70b535d6c32a433c9358be5cd88469b3958dedb91fe25485288e9439674c2a1.jpg)

![fbec0eaac2b87d1061b38a3261e5af526e9065104290d14557b5765c48a85b26.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/fbec0eaac2b87d1061b38a3261e5af526e9065104290d14557b5765c48a85b26.jpg)

![fc87d21b593b72602ea65ce5f519e514c1c78c501e75457f7a6ea70871ee76d3.jpg](../icml_results/4976_Relational%20In-Context%20Learning%20via%20Synthetic%20Pre-training%20with%20Structural%20Prior/tables/fc87d21b593b72602ea65ce5f519e514c1c78c501e75457f7a6ea70871ee76d3.jpg)

<a id="paper-4929"></a>

## Deep Scientific Reasoning under Physical Constraints: Structure-Aware Spectrum Prediction


### Images

![099b9f65d516bfc4eb3de326b9d59c8543ed14998629ea27f9431cbea1acc201.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/099b9f65d516bfc4eb3de326b9d59c8543ed14998629ea27f9431cbea1acc201.jpg)

![785120a7cf8aeb76cfe2efe3d42ca13d695877d77a1ef10b0fcd3522b5ae0b1f.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/785120a7cf8aeb76cfe2efe3d42ca13d695877d77a1ef10b0fcd3522b5ae0b1f.jpg)

![9b39546363cf3fc3bac3d52f9c7e00e82db137337fa7e9c06f9b651989e4ad99.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/9b39546363cf3fc3bac3d52f9c7e00e82db137337fa7e9c06f9b651989e4ad99.jpg)

![b74ccb301e3f5cc2bfc93ca136dc015616267cbdc748ab92c4c847f18aad880e.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/b74ccb301e3f5cc2bfc93ca136dc015616267cbdc748ab92c4c847f18aad880e.jpg)

![da74d1430c6e81b02fc78de2847442a4c11d21a8ec8173afef27d459ee69e617.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/da74d1430c6e81b02fc78de2847442a4c11d21a8ec8173afef27d459ee69e617.jpg)

![ec10d3743262e59e0d61fc450de3d16c62d7a91993ba8b3de39f967048459c24.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/images/ec10d3743262e59e0d61fc450de3d16c62d7a91993ba8b3de39f967048459c24.jpg)

### Tables

![0a6acc75e606a8bb869a103c731907d4e4f6f7181733cac2e9238cb68dd84818.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/0a6acc75e606a8bb869a103c731907d4e4f6f7181733cac2e9238cb68dd84818.jpg)

![700ceda8a1504004d585acbb98d5baa0f9916d406a723c9e93c743acc01b44c7.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/700ceda8a1504004d585acbb98d5baa0f9916d406a723c9e93c743acc01b44c7.jpg)

![8d799515bfbf36e5c6b3a8b36d1c08bfd3606e10991ec097a3fd4161e02d302a.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/8d799515bfbf36e5c6b3a8b36d1c08bfd3606e10991ec097a3fd4161e02d302a.jpg)

![b3bb52ac96c0c626c140923a2991c1b01dc061cf868a13486e847780681ed7c0.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/b3bb52ac96c0c626c140923a2991c1b01dc061cf868a13486e847780681ed7c0.jpg)

![b488380d8db0612091d138652cdb93c78eaec8cd7b84748bdc62d3f9734be745.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/b488380d8db0612091d138652cdb93c78eaec8cd7b84748bdc62d3f9734be745.jpg)

![d1ac45081dffbc86dfe5013660731a4968d22490e95d77876d2f84eff1e43861.jpg](../icml_results/4977_Deep%20Scientific%20Reasoning%20under%20Physical%20Constraints_%20Structure-Aware%20Spectrum%20Prediction/tables/d1ac45081dffbc86dfe5013660731a4968d22490e95d77876d2f84eff1e43861.jpg)

<a id="paper-4930"></a>

## Cache Coherent Resampling for Efficient Test Time Scaling in LLM Reasoning via Adaptive Sequential Monte Carlo


### Images

![399aaec4e0e97b4e92ca44efeebb4327bedef32eb24e6737ac2c36bfc760183c.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/399aaec4e0e97b4e92ca44efeebb4327bedef32eb24e6737ac2c36bfc760183c.jpg)

![698d5335332a215d82ff9203b7cf33c952e0512757a02c6972835237cc5ecb20.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/698d5335332a215d82ff9203b7cf33c952e0512757a02c6972835237cc5ecb20.jpg)

![b0251805bf40789f17804797c1b6cffca15f795051e3299b48e0e9cf70810797.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/b0251805bf40789f17804797c1b6cffca15f795051e3299b48e0e9cf70810797.jpg)

![d028149ad817fdb94007a47f76e34a8a9a00aac0279fa034436a2e0267e85f65.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/d028149ad817fdb94007a47f76e34a8a9a00aac0279fa034436a2e0267e85f65.jpg)

![e7a6f12ce1faf034481054fb455532b705dd57e0d5873cde19ae1dcbba0b7124.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/e7a6f12ce1faf034481054fb455532b705dd57e0d5873cde19ae1dcbba0b7124.jpg)

![f1c38464277af5422de44714b17a710dc62136848edf81316df16b9c21754330.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/images/f1c38464277af5422de44714b17a710dc62136848edf81316df16b9c21754330.jpg)

### Tables

![6fb0c517c23ee02dfe7e833ae5a6053cc1928a964411dbbbeee332877d610914.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/6fb0c517c23ee02dfe7e833ae5a6053cc1928a964411dbbbeee332877d610914.jpg)

![7da18a9e4f7aa4ef8c12c0195a943df5a9e53cdf94d0fdb411453880af4aaf1b.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/7da18a9e4f7aa4ef8c12c0195a943df5a9e53cdf94d0fdb411453880af4aaf1b.jpg)

![a21d557c6a49aae5c39a1f1d8432ff1f168f21f0559f24bcee4f4b787a368ce4.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/a21d557c6a49aae5c39a1f1d8432ff1f168f21f0559f24bcee4f4b787a368ce4.jpg)

![a8bb810c376a4d72c4460a1cfddda5e1c88ae5e9c42f926b99d0bf2fb4cff36f.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/a8bb810c376a4d72c4460a1cfddda5e1c88ae5e9c42f926b99d0bf2fb4cff36f.jpg)

![f9d550e9443be4ee0f998af3ae88d255fb18b5c443349994cd4947190badd4e6.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/f9d550e9443be4ee0f998af3ae88d255fb18b5c443349994cd4947190badd4e6.jpg)

![feec9f413286b4c0f69aceebc85befbd0d8b78ba777aa2f762c0610351ebc60a.jpg](../icml_results/4978_Cache%20Coherent%20Resampling%20for%20Efficient%20Test%20Time%20Scaling%20in%20LLM%20Reasoning%20via%20Adaptive%20Sequential%20M/tables/feec9f413286b4c0f69aceebc85befbd0d8b78ba777aa2f762c0610351ebc60a.jpg)

<a id="paper-4931"></a>

## LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization


### Images

![1d2180a8f9f2b225026221d069d8a3f727646e465454ebd13331b233f21e89f0.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/1d2180a8f9f2b225026221d069d8a3f727646e465454ebd13331b233f21e89f0.jpg)

![3fea870f06b206eb9b959a4b435d4fe448d47e006bf808a89bdf8485e5f75da4.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/3fea870f06b206eb9b959a4b435d4fe448d47e006bf808a89bdf8485e5f75da4.jpg)

![8c101976525ce4455b99d460cd6ed7f4fbb8ea8a8e0c41ec06ab697a9d64717d.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/8c101976525ce4455b99d460cd6ed7f4fbb8ea8a8e0c41ec06ab697a9d64717d.jpg)

![96f8c65afdf740218a09e83238c33b73caa4b4a41f08d100a23f11ddcf1f6d2f.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/96f8c65afdf740218a09e83238c33b73caa4b4a41f08d100a23f11ddcf1f6d2f.jpg)

![c19315528887db1ecd89b3395df9ad65d2a569f3855b01de5e8624e3f74ff2e5.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/c19315528887db1ecd89b3395df9ad65d2a569f3855b01de5e8624e3f74ff2e5.jpg)

![d62121ddb49aa02f19a33d886642b93a03b4ee8988ec1cb06f3a554167d40fca.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/d62121ddb49aa02f19a33d886642b93a03b4ee8988ec1cb06f3a554167d40fca.jpg)

![edc776f5545d04c054dfc5afaf0661191b80fc0d2d114d57cef9cfd398e4d574.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/images/edc776f5545d04c054dfc5afaf0661191b80fc0d2d114d57cef9cfd398e4d574.jpg)

### Tables

![0dcdc22b81779ce7412340e202ed325afcd5b0405773b88774e03e18a0c66707.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/0dcdc22b81779ce7412340e202ed325afcd5b0405773b88774e03e18a0c66707.jpg)

![23b5f2fdefa1bff3076c443ddca5663b5a769b3f5acd4294125b182b8f223add.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/23b5f2fdefa1bff3076c443ddca5663b5a769b3f5acd4294125b182b8f223add.jpg)

![2ee12507857730275c7346b565731aa54a95bb7fa94cb0014ed5178bdd0599a0.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/2ee12507857730275c7346b565731aa54a95bb7fa94cb0014ed5178bdd0599a0.jpg)

![41223dd7d27f34d18f74ec3c76160ad220509e1e19822c67c47baa3c6faa3728.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/41223dd7d27f34d18f74ec3c76160ad220509e1e19822c67c47baa3c6faa3728.jpg)

![587a8f5cde0cbb7dc0ca7e809e4ff928d2dca0b3c02902bb8e538dd30af615a1.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/587a8f5cde0cbb7dc0ca7e809e4ff928d2dca0b3c02902bb8e538dd30af615a1.jpg)

![5ae92b0714eabeafc8f8823c6797087fead4fa5ca05972e2403afe4ddb4e62f7.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/5ae92b0714eabeafc8f8823c6797087fead4fa5ca05972e2403afe4ddb4e62f7.jpg)

![79b6bc0148ec120b230280ab9cb76d3017de85a230ed14bf3d98ed7f3ba6ca97.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/79b6bc0148ec120b230280ab9cb76d3017de85a230ed14bf3d98ed7f3ba6ca97.jpg)

![895b13802efc84c882bc51156dab16235e3133f20833938035c3904619843522.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/895b13802efc84c882bc51156dab16235e3133f20833938035c3904619843522.jpg)

![9cca47aa1e1b57216191e5b6ef73878bbc7a02409b0b91e3aaf4fcfe0fdedcf6.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/9cca47aa1e1b57216191e5b6ef73878bbc7a02409b0b91e3aaf4fcfe0fdedcf6.jpg)

![a8bfc0759f2b28f1198f625b75c70f411d64c4b0fedbc518381a8b949b1f1f6a.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/a8bfc0759f2b28f1198f625b75c70f411d64c4b0fedbc518381a8b949b1f1f6a.jpg)

![f72076f7a42936b91b943395e4302c96ae1e9f8e1876d678b8a661478eb0bf86.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/f72076f7a42936b91b943395e4302c96ae1e9f8e1876d678b8a661478eb0bf86.jpg)

![fa8e29ba3bf22f9a7ac5dc2a1e90625b455a60bf1719ec30813f0fa7b27d39b9.jpg](../icml_results/4979_LC-QAT_%20Data-Efficient%202-Bit%20QAT%20for%20LLMs%20via%20Linear-Constrained%20Vector%20Quantization/tables/fa8e29ba3bf22f9a7ac5dc2a1e90625b455a60bf1719ec30813f0fa7b27d39b9.jpg)

<a id="paper-4932"></a>

## Spiked-CFR: Causal Representation Learning from LLMs via Wasserstein Projection Pursuit


### Images

![022104f46487d85db97b1f14581b52a0a9d1572f33a993f03b502097daa3f5c7.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/022104f46487d85db97b1f14581b52a0a9d1572f33a993f03b502097daa3f5c7.jpg)

![872d3246a95305787005cd0772879c72b7f2ee1fd5aef686102c87f7234edd84.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/872d3246a95305787005cd0772879c72b7f2ee1fd5aef686102c87f7234edd84.jpg)

![939a6c41de6bc7881dc3091a3e2e2f68f6a6a2fe72afef49eea17b1a0bab7ece.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/939a6c41de6bc7881dc3091a3e2e2f68f6a6a2fe72afef49eea17b1a0bab7ece.jpg)

![d76c8103acbce93afa81eda5d89b034111235965e7554e512c6ed7ce2f266b6d.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/d76c8103acbce93afa81eda5d89b034111235965e7554e512c6ed7ce2f266b6d.jpg)

![dde9301f34f0dc91acd16f9307876ece81463ed941d8f97f25092c4eacf7f25b.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/dde9301f34f0dc91acd16f9307876ece81463ed941d8f97f25092c4eacf7f25b.jpg)

![ddf52c7b0d003ff233b98b97cc6eb265010c24440b13dbf7136cc3e0e54c339b.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/ddf52c7b0d003ff233b98b97cc6eb265010c24440b13dbf7136cc3e0e54c339b.jpg)

![eb20b874d268f08ea5daa3e7cef82375f2814846aed9a1d2ca01ec60b939eae3.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/images/eb20b874d268f08ea5daa3e7cef82375f2814846aed9a1d2ca01ec60b939eae3.jpg)

### Tables

![156be8a583f9f358c909b618b75ce118ec83ded357cddb2f32a3b7b2f80aa624.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/156be8a583f9f358c909b618b75ce118ec83ded357cddb2f32a3b7b2f80aa624.jpg)

![450a7d0db9cc81a6f5e4f514a76cf0ba78e6cea88a5910c70bdf83c56ff61bd2.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/450a7d0db9cc81a6f5e4f514a76cf0ba78e6cea88a5910c70bdf83c56ff61bd2.jpg)

![4efc845e8570ecb650471951f1a43b4d5acc5307fb897cbc531f54febf40c108.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/4efc845e8570ecb650471951f1a43b4d5acc5307fb897cbc531f54febf40c108.jpg)

![9828a90582605d3f45f0fe1e350e6ff98744bfb746ced01c864930c5d94972de.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/9828a90582605d3f45f0fe1e350e6ff98744bfb746ced01c864930c5d94972de.jpg)

![a65af369fe43ce5512d1e56bcbd80234f5d7ee156e2750bd06d09ea195b40cdf.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/a65af369fe43ce5512d1e56bcbd80234f5d7ee156e2750bd06d09ea195b40cdf.jpg)

![b3e3a9aa3fcbdca15438de48a898af098efbf1e96eeb2afef98ec4dd050c3f93.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/b3e3a9aa3fcbdca15438de48a898af098efbf1e96eeb2afef98ec4dd050c3f93.jpg)

![c6419084d9edf8b552c0ec414cf6fbc2a11ee7e763bc3946312e6723d9371624.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/c6419084d9edf8b552c0ec414cf6fbc2a11ee7e763bc3946312e6723d9371624.jpg)

![ec3d567c89913768dc58574540f7549b7aa0539637023d2301956b3d545281eb.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/ec3d567c89913768dc58574540f7549b7aa0539637023d2301956b3d545281eb.jpg)

![f3cd158c92c75dbb3c6326e2345a08d337decde31b6396d4e3eec28e494cdf3e.jpg](../icml_results/4980_Spiked-CFR_%20Causal%20Representation%20Learning%20from%20LLMs%20via%20Wasserstein%20Projection%20Pursuit/tables/f3cd158c92c75dbb3c6326e2345a08d337decde31b6396d4e3eec28e494cdf3e.jpg)

<a id="paper-4933"></a>

## MemPot: Defending Against Memory Extraction Attack with Optimized Honeypots


### Images

![b40ca9fe22ea54129df5069d3614eb7bf3eb97c702547b3a2acd2556c03f99e0.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/images/b40ca9fe22ea54129df5069d3614eb7bf3eb97c702547b3a2acd2556c03f99e0.jpg)

![ccffd7c15bd5b84abdbd9cf460bba520748283c39ef6b1e9ce648ee1f42ca65f.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/images/ccffd7c15bd5b84abdbd9cf460bba520748283c39ef6b1e9ce648ee1f42ca65f.jpg)

![dea055024be11565dce9b24fe936f14fae18c49273b33f9d76b2da0316fcc3ea.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/images/dea055024be11565dce9b24fe936f14fae18c49273b33f9d76b2da0316fcc3ea.jpg)

### Tables

![0a7b8b8b62541f00d8ada6cfcfdbe62f313333a6e2c5e2efd6ec622e03da082e.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/0a7b8b8b62541f00d8ada6cfcfdbe62f313333a6e2c5e2efd6ec622e03da082e.jpg)

![186bd3869a047048d74e8d2394b3934778d16fdc42fa9145f5d90f5d0496323d.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/186bd3869a047048d74e8d2394b3934778d16fdc42fa9145f5d90f5d0496323d.jpg)

![1f700ae039c30d6be87c5b8a8d714c51d35da6d7342371ba50eed8fce65c5042.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/1f700ae039c30d6be87c5b8a8d714c51d35da6d7342371ba50eed8fce65c5042.jpg)

![2fd6c3bc44b1e267e69deb7cabef743390ca54a6747775112a8ee2dbd431216a.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/2fd6c3bc44b1e267e69deb7cabef743390ca54a6747775112a8ee2dbd431216a.jpg)

![314887e54c9e96adaa2653a5978a14442e0e6b608348888c30bbd57f4f4fe618.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/314887e54c9e96adaa2653a5978a14442e0e6b608348888c30bbd57f4f4fe618.jpg)

![5522e5b0d79413e1af59779d22fdf03090057eed721b6236dd53134eda386de3.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/5522e5b0d79413e1af59779d22fdf03090057eed721b6236dd53134eda386de3.jpg)

![8a93f21a316cadf40627e3d815992459231d22a24ceb273a07ea18005958062b.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/8a93f21a316cadf40627e3d815992459231d22a24ceb273a07ea18005958062b.jpg)

![90ba71c6f96e61abe2c2e0ecdad3a8635725e4a59f3b1c378b9289b228ba0522.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/90ba71c6f96e61abe2c2e0ecdad3a8635725e4a59f3b1c378b9289b228ba0522.jpg)

![a673d4d7922772c78267f48a4a517ddac740bab9af4ce63abb8125a7708bfd07.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/a673d4d7922772c78267f48a4a517ddac740bab9af4ce63abb8125a7708bfd07.jpg)

![e0246f56584a02ef900e8af003228ffaa4a45786a4ffd50cc238040c653bcacc.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/e0246f56584a02ef900e8af003228ffaa4a45786a4ffd50cc238040c653bcacc.jpg)

![e4ac31db069ad0d95b7b17c9ae61c78857d630741d5d04a8c043a690af2e4340.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/e4ac31db069ad0d95b7b17c9ae61c78857d630741d5d04a8c043a690af2e4340.jpg)

![f120840f878247f4a20a237a9248505782a45394b9e8a4483d77b001f7612141.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/f120840f878247f4a20a237a9248505782a45394b9e8a4483d77b001f7612141.jpg)

![f124be2e87c00aa218ddf8d5fb4bd8589eba0c8d27619db25234e49ccd8ccc6e.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/f124be2e87c00aa218ddf8d5fb4bd8589eba0c8d27619db25234e49ccd8ccc6e.jpg)

![f2a138a74b34ed026494c0aff6e605598681835a84084a902b7e0eccd45a947f.jpg](../icml_results/4981_MemPot_%20Defending%20Against%20Memory%20Extraction%20Attack%20with%20Optimized%20Honeypots/tables/f2a138a74b34ed026494c0aff6e605598681835a84084a902b7e0eccd45a947f.jpg)

<a id="paper-4934"></a>

## AES: Curing Optimizer Blindness in Long-Tailed Recognition via State-Aware Correction


### Images

![691d447355a5e07f4dafa076538cdf0b5dccee559a16f8b2e3cb923e0333dd4b.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/691d447355a5e07f4dafa076538cdf0b5dccee559a16f8b2e3cb923e0333dd4b.jpg)

![6c93bce1030333c6e145fcf34e6ba732e64b8f2cc0aaa2fa04d557b141ad865d.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/6c93bce1030333c6e145fcf34e6ba732e64b8f2cc0aaa2fa04d557b141ad865d.jpg)

![7f4c0804bb309e1e1439eabe8efc942c7e6790eca6b8cc975b3c1a78112f7fb7.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/7f4c0804bb309e1e1439eabe8efc942c7e6790eca6b8cc975b3c1a78112f7fb7.jpg)

![8d5c36efe5d0d4eb8236cd73caf37f91436416118824660ba216715124fdf62f.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/8d5c36efe5d0d4eb8236cd73caf37f91436416118824660ba216715124fdf62f.jpg)

![966d067672764d2c6892e8ae9206862886107f38e4c11bed5bdfda7d12c671cf.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/966d067672764d2c6892e8ae9206862886107f38e4c11bed5bdfda7d12c671cf.jpg)

![a424840df18ceb97abbc0d7077de4b1d02500625152207f6fa3581c50abc25e6.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/a424840df18ceb97abbc0d7077de4b1d02500625152207f6fa3581c50abc25e6.jpg)

![ab519e9030f880a19612a289f70fda4179908c08b6be1c780d52c36f524c67f6.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/images/ab519e9030f880a19612a289f70fda4179908c08b6be1c780d52c36f524c67f6.jpg)

### Tables

![079589c250979eac6b7cbaf6e6bfea772983f543d544d9888d78c06be8129194.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/079589c250979eac6b7cbaf6e6bfea772983f543d544d9888d78c06be8129194.jpg)

![12ced5b29d04659ae23db20049aa70770a93f947ca194948ca9f5afcaf6141d0.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/12ced5b29d04659ae23db20049aa70770a93f947ca194948ca9f5afcaf6141d0.jpg)

![15932432910ea428797fd9e9f0ec54f3ec8ee5335f984ff0dea7e29615c6b230.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/15932432910ea428797fd9e9f0ec54f3ec8ee5335f984ff0dea7e29615c6b230.jpg)

![2514c7238d50a2011cfbdd93b19880678d383ad90c5796ea0e814faa7cd6f716.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/2514c7238d50a2011cfbdd93b19880678d383ad90c5796ea0e814faa7cd6f716.jpg)

![296d6acf49b2d6acb1cf48121f1b74cf95b1c056b3860f05bbb7720c84d85f72.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/296d6acf49b2d6acb1cf48121f1b74cf95b1c056b3860f05bbb7720c84d85f72.jpg)

![2a585627eeeb7cacb2aa82db35765d735d2ebb002d804adf2354a95492073f2c.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/2a585627eeeb7cacb2aa82db35765d735d2ebb002d804adf2354a95492073f2c.jpg)

![306522bf48641513011336e756b7817ab18d8a7960855db4d421b21d9cbb9436.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/306522bf48641513011336e756b7817ab18d8a7960855db4d421b21d9cbb9436.jpg)

![47f35284687db94e4fd3a5f59ea20dce573ab9f286fb03e04dd2ee8bd4e7fb02.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/47f35284687db94e4fd3a5f59ea20dce573ab9f286fb03e04dd2ee8bd4e7fb02.jpg)

![6f2e7d5acd6daad14b3b9409c5778c166d22ae6889926265f9092600cbcddab3.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/6f2e7d5acd6daad14b3b9409c5778c166d22ae6889926265f9092600cbcddab3.jpg)

![707da394c852d491bdd9ef5df2e83d3bc50d559f3f40a6a45c625ebc781ddf22.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/707da394c852d491bdd9ef5df2e83d3bc50d559f3f40a6a45c625ebc781ddf22.jpg)

![880cfb93318eab190a98cb31e724ed161c5e29a2fc13fba0726b86adcd896ebd.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/880cfb93318eab190a98cb31e724ed161c5e29a2fc13fba0726b86adcd896ebd.jpg)

![925d1f77834f345fb54e8960654a9635bf7729c26f325f1fdf8d3c27c7b3c284.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/925d1f77834f345fb54e8960654a9635bf7729c26f325f1fdf8d3c27c7b3c284.jpg)

![9ab37ef8a842d0e793403ed1cfe2bc249101116129474ce41c12fd2cc3bbe218.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/9ab37ef8a842d0e793403ed1cfe2bc249101116129474ce41c12fd2cc3bbe218.jpg)

![9c0fb72a6c42393a7fceb57ec231643dd426f6ac02ec544907d8dfe1c538a141.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/9c0fb72a6c42393a7fceb57ec231643dd426f6ac02ec544907d8dfe1c538a141.jpg)

![bc732433910f9f1e0b1f9cafe3f401ca725acc7701134a2d8d14fb0ab48a92df.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/bc732433910f9f1e0b1f9cafe3f401ca725acc7701134a2d8d14fb0ab48a92df.jpg)

![dec129c4ce4991e0fb6f25925c38f2d90e45a80edf25bb36c7a313a788a2300c.jpg](../icml_results/4982_AES_%20Curing%20Optimizer%20Blindness%20in%20Long-Tailed%20Recognition%20via%20State-Aware%20Correction/tables/dec129c4ce4991e0fb6f25925c38f2d90e45a80edf25bb36c7a313a788a2300c.jpg)

<a id="paper-4935"></a>

## The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models


### Images

![07fd191349a347985f061713687ee6f10ccc0b6758f055b403aa9a1c2feac6a5.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/07fd191349a347985f061713687ee6f10ccc0b6758f055b403aa9a1c2feac6a5.jpg)

![110d8ea8b8e04449f08a6171affdd5fe90bc58c44208be4295e5c767af97a301.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/110d8ea8b8e04449f08a6171affdd5fe90bc58c44208be4295e5c767af97a301.jpg)

![1ad39ca1b605e7c2cf43034fdb76b35f24e8b770654062ba6aa7f4823627ff76.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/1ad39ca1b605e7c2cf43034fdb76b35f24e8b770654062ba6aa7f4823627ff76.jpg)

![8718cdd74831114153973c859dd677e72b9e5612b34b2c6a293ccb5543b8c95b.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/8718cdd74831114153973c859dd677e72b9e5612b34b2c6a293ccb5543b8c95b.jpg)

![87876ac686725684cb68aec86d26daa1f6a2a0a4d452ad287b1ecb995342b330.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/87876ac686725684cb68aec86d26daa1f6a2a0a4d452ad287b1ecb995342b330.jpg)

![c808ebf24d2053cc290927d78c6591a8b7c9a55684e2e63e8f2267d76f80ad04.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/c808ebf24d2053cc290927d78c6591a8b7c9a55684e2e63e8f2267d76f80ad04.jpg)

![ceb5e80693ca312a7e683f6cfb209cf25d6bf67623e62f7756d7ee592af6af47.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/images/ceb5e80693ca312a7e683f6cfb209cf25d6bf67623e62f7756d7ee592af6af47.jpg)

### Tables

![14a62b4a4e907c9740ae3495a0d8559b1d2b837ae01c1dd1df7f0a43ac040873.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/14a62b4a4e907c9740ae3495a0d8559b1d2b837ae01c1dd1df7f0a43ac040873.jpg)

![1f4180edbb185cd7e614efae0af1ac224ffc746c87b9b39c1845d30c9d963f23.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/1f4180edbb185cd7e614efae0af1ac224ffc746c87b9b39c1845d30c9d963f23.jpg)

![2f9e3ec8adf18a5c7ab83b42a8b7dc3012dd3706bc8584d51129348c52968284.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/2f9e3ec8adf18a5c7ab83b42a8b7dc3012dd3706bc8584d51129348c52968284.jpg)

![31ba455aedef19fc480d82491314533217e991a612620ea263864612ea4a8497.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/31ba455aedef19fc480d82491314533217e991a612620ea263864612ea4a8497.jpg)

![58b3322953caebccc5d18d3f734f9ac90adcb720b7ef7fc5b811df157c82bee2.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/58b3322953caebccc5d18d3f734f9ac90adcb720b7ef7fc5b811df157c82bee2.jpg)

![701d2fd921fcc1ec7b927a7612553c91f8968d4fa983be47c5a4d74c25df5325.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/701d2fd921fcc1ec7b927a7612553c91f8968d4fa983be47c5a4d74c25df5325.jpg)

![7c03dd0d971f926c46857ac304bf55d7380c1afe15787d17f1353162f84a5eea.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/7c03dd0d971f926c46857ac304bf55d7380c1afe15787d17f1353162f84a5eea.jpg)

![8d31220dc9b47cdc77a5e8e00f37554c799e6be484f56e4c28d2dbeaa3f09b18.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/8d31220dc9b47cdc77a5e8e00f37554c799e6be484f56e4c28d2dbeaa3f09b18.jpg)

![acc80a98102a3ae39b3698cd800d53f49116ab989520576b770f025eb594e7be.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/acc80a98102a3ae39b3698cd800d53f49116ab989520576b770f025eb594e7be.jpg)

![ad132c83e0920c6256485cbf8c50b83df6643a822ea032d0def81acb0c24adf3.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/ad132c83e0920c6256485cbf8c50b83df6643a822ea032d0def81acb0c24adf3.jpg)

![b083dd55f2ffa98b5c0b321119211028997b70ae2102996d25610120a710c7a9.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/b083dd55f2ffa98b5c0b321119211028997b70ae2102996d25610120a710c7a9.jpg)

![bcf1a3fa32fd02df6db45e8f45dbf3ea16678c7d5ed87083c0a5a56e56b73006.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/bcf1a3fa32fd02df6db45e8f45dbf3ea16678c7d5ed87083c0a5a56e56b73006.jpg)

![c640fd15b4585884cb6d07722d817c51ef9e62a49efaeaf689dfe6cfc112387b.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/c640fd15b4585884cb6d07722d817c51ef9e62a49efaeaf689dfe6cfc112387b.jpg)

![d72565bf3d78be4376e11887e57fcf8f15beacf6826e0dc321a2925651c98e79.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/d72565bf3d78be4376e11887e57fcf8f15beacf6826e0dc321a2925651c98e79.jpg)

![ead554d0f15b17a859d355bb1c8dc698dc3475765573dfcbd5c478159c6e1405.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/ead554d0f15b17a859d355bb1c8dc698dc3475765573dfcbd5c478159c6e1405.jpg)

![f14bcc189aaa9fa615b1054197875feabf5e4ac24559e10b715a6fc0dc952a9e.jpg](../icml_results/4983_The%20Labyrinth%20and%20the%20Thread_%20Rethinking%20Regularizations%20in%20Sequential%20Knowledge%20Editing%20for%20Large%20L/tables/f14bcc189aaa9fa615b1054197875feabf5e4ac24559e10b715a6fc0dc952a9e.jpg)

<a id="paper-4936"></a>

## Partial Identification under High-Dimensional Potential Outcomes and Confounders via Optimal Transport


### Images

![552a8b0f15daaeafe4f32fb013569fd10632d7bcceaaec08500adf5d2bb510e7.jpg](../icml_results/4984_Partial%20Identification%20under%20High-Dimensional%20Potential%20Outcomes%20and%20Confounders%20via%20Optimal%20Transpo/images/552a8b0f15daaeafe4f32fb013569fd10632d7bcceaaec08500adf5d2bb510e7.jpg)

![7bf2b704335924f08140a79a4f4bf115d027e8eb9161fadf9eb33c9b42651c84.jpg](../icml_results/4984_Partial%20Identification%20under%20High-Dimensional%20Potential%20Outcomes%20and%20Confounders%20via%20Optimal%20Transpo/images/7bf2b704335924f08140a79a4f4bf115d027e8eb9161fadf9eb33c9b42651c84.jpg)

![a6e5375e5e00db41890a7eda9cdf2ade8b61c5bb419a9efad57832fc140e9a7e.jpg](../icml_results/4984_Partial%20Identification%20under%20High-Dimensional%20Potential%20Outcomes%20and%20Confounders%20via%20Optimal%20Transpo/images/a6e5375e5e00db41890a7eda9cdf2ade8b61c5bb419a9efad57832fc140e9a7e.jpg)

<a id="paper-4937"></a>

## PromptRL: Prompt Matters in RL for Flow-Based Image Generation


### Images

![549845e3e9dc4c258d51ae859ed7753e36beacc9569575b7cfdf38d28c4bd618.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/549845e3e9dc4c258d51ae859ed7753e36beacc9569575b7cfdf38d28c4bd618.jpg)

![5ba893cfabac972b6f98575cdd5c3ad9c8f618a5f602d274bd9e07c14281d592.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/5ba893cfabac972b6f98575cdd5c3ad9c8f618a5f602d274bd9e07c14281d592.jpg)

![8dca9f9bcfd6bec91dd79ba3e92b3ce8386a2157228f4d9c8f6ab0b7943ad43c.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/8dca9f9bcfd6bec91dd79ba3e92b3ce8386a2157228f4d9c8f6ab0b7943ad43c.jpg)

![a02b60004da71b47da3237247bb2412a67fe6fc317815fb46c3559e1af10c220.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/a02b60004da71b47da3237247bb2412a67fe6fc317815fb46c3559e1af10c220.jpg)

![a5ede7dd59a8a93a6594ed1f23fddb047a1a984f620fe11f5c9baf4d3b967726.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/a5ede7dd59a8a93a6594ed1f23fddb047a1a984f620fe11f5c9baf4d3b967726.jpg)

![b59336763fa41178a8734696c2b7b12ae103fe81bbd2e690b27fd06ef95c4c1b.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/b59336763fa41178a8734696c2b7b12ae103fe81bbd2e690b27fd06ef95c4c1b.jpg)

![ef1b82617d617c5ebc31aa73465ef55802a02e31f265d64f904ea57a8fceb040.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/images/ef1b82617d617c5ebc31aa73465ef55802a02e31f265d64f904ea57a8fceb040.jpg)

### Tables

![0096494e735b6886f19cbd10e0cf519b65e6215d1c084f846743b07cd51e87bc.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/0096494e735b6886f19cbd10e0cf519b65e6215d1c084f846743b07cd51e87bc.jpg)

![0627743780552537ff924be052ae580a3f27f9c384cf9d2af0864a8ce7b5ed05.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/0627743780552537ff924be052ae580a3f27f9c384cf9d2af0864a8ce7b5ed05.jpg)

![11076d70683137ab99faee836783db5c514b1ae694f1ffcc0811a2465714a7f4.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/11076d70683137ab99faee836783db5c514b1ae694f1ffcc0811a2465714a7f4.jpg)

![26f2639d1e5dce062adbef8945837395cb78beca358d6a6d4cb137f1571917df.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/26f2639d1e5dce062adbef8945837395cb78beca358d6a6d4cb137f1571917df.jpg)

![2edd8a1e9fb0a4e5bcf40859e816b63cbfceebbeb51163badc80316d989d02aa.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/2edd8a1e9fb0a4e5bcf40859e816b63cbfceebbeb51163badc80316d989d02aa.jpg)

![2fdcb2ed096edb5cd946ba690c9c09d1172b49008ce576051d775b03dea018d8.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/2fdcb2ed096edb5cd946ba690c9c09d1172b49008ce576051d775b03dea018d8.jpg)

![5f501a2c79119fbd7e5311bc5f2287c2596ad74452bbd17208a294cfe17f00e1.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/5f501a2c79119fbd7e5311bc5f2287c2596ad74452bbd17208a294cfe17f00e1.jpg)

![625aeaa5dc9d0f66f47a0c35f6533e64a083bf4eba646dbd9aab75c563aaa60a.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/625aeaa5dc9d0f66f47a0c35f6533e64a083bf4eba646dbd9aab75c563aaa60a.jpg)

![6c5739ab24158610b87b08830fd0369b7af3a735a338892e056c4bb36f276871.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/6c5739ab24158610b87b08830fd0369b7af3a735a338892e056c4bb36f276871.jpg)

![7a4ac3443a6967ef8a72394ddb2f86b8166af681ad7ac348d86f27fdca3ff92b.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/7a4ac3443a6967ef8a72394ddb2f86b8166af681ad7ac348d86f27fdca3ff92b.jpg)

![87c3507fe4d7ec3b6dbb33584f81b28300a52d73ca50ba7a2a8187945bc4c6da.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/87c3507fe4d7ec3b6dbb33584f81b28300a52d73ca50ba7a2a8187945bc4c6da.jpg)

![938943f923d5bd2d2561624bb8ecc480e79376bc8bd650e646ecc62f35ed0e1b.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/938943f923d5bd2d2561624bb8ecc480e79376bc8bd650e646ecc62f35ed0e1b.jpg)

![ada960c23615b036af9a5c40c29f4d08c97314799d34d355e6cc9bbfd2a285c1.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/ada960c23615b036af9a5c40c29f4d08c97314799d34d355e6cc9bbfd2a285c1.jpg)

![df991028617cf27cb8220528064f3b66b6bbcfb6beb3c3b1d6bd621a10d658f8.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/df991028617cf27cb8220528064f3b66b6bbcfb6beb3c3b1d6bd621a10d658f8.jpg)

![f0e6cc8f1bcab886db4471c54a540be6e32ac2bcb2db6b8f7f7290f0b664ff6b.jpg](../icml_results/4985_PromptRL_%20Prompt%20Matters%20in%20RL%20for%20Flow-Based%20Image%20Generation/tables/f0e6cc8f1bcab886db4471c54a540be6e32ac2bcb2db6b8f7f7290f0b664ff6b.jpg)

<a id="paper-4938"></a>

## Non-Parametric Probabilistic Robustness: A Conservative Risk Estimator under Unknown Perturbation Distributions


### Images

![1edb8b7538da8e07e2e61b2ff0211f478948cd26c9e7b96ee0407c100c769942.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/1edb8b7538da8e07e2e61b2ff0211f478948cd26c9e7b96ee0407c100c769942.jpg)

![22a07679e5b617c85665b1a5e14bb1efb26894c5f8a25fed5c4821e7226ab6b7.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/22a07679e5b617c85665b1a5e14bb1efb26894c5f8a25fed5c4821e7226ab6b7.jpg)

![2b4d225da813988e3582c13127dd09b9a28ac2d021b8ab6f00e0372bf66305dc.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/2b4d225da813988e3582c13127dd09b9a28ac2d021b8ab6f00e0372bf66305dc.jpg)

![4c6a3f8e72aa18ff289cf7245295895bf92c40d5bd9cad964b2c137efaa10217.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/4c6a3f8e72aa18ff289cf7245295895bf92c40d5bd9cad964b2c137efaa10217.jpg)

![5d1e3bc66f0ea3d6d8c41b227ce3d6d59151a096d96755f286b0cc38856b03a5.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/5d1e3bc66f0ea3d6d8c41b227ce3d6d59151a096d96755f286b0cc38856b03a5.jpg)

![bdbb0d11f30fddab71c36ddfbb3ace2bfcddc685ff22638350713c89d5a7b7e0.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/bdbb0d11f30fddab71c36ddfbb3ace2bfcddc685ff22638350713c89d5a7b7e0.jpg)

![c3d2aeccd1ef527db9339375d23b2df2528a4d5ec96a04b6d7ce564c1e665d98.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/c3d2aeccd1ef527db9339375d23b2df2528a4d5ec96a04b6d7ce564c1e665d98.jpg)

![e9342d87ef09aadc4e1867c84ac0c7d4145c27d90641cd853cc3e580a2c5f2a2.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/e9342d87ef09aadc4e1867c84ac0c7d4145c27d90641cd853cc3e580a2c5f2a2.jpg)

![fd342ee2fd4d8633e360d14c8abef57e8c11a73d318f2df5c6fd112301f1c42d.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/images/fd342ee2fd4d8633e360d14c8abef57e8c11a73d318f2df5c6fd112301f1c42d.jpg)

### Tables

![1a982171fd6c651b0f3a3fae17cf92f5d50ce9a3253b8038b044b54d9f21e1a0.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/1a982171fd6c651b0f3a3fae17cf92f5d50ce9a3253b8038b044b54d9f21e1a0.jpg)

![1f19b37685373468222ff67f259b6e6823d3ac1229c479fdadaff250097834c1.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/1f19b37685373468222ff67f259b6e6823d3ac1229c479fdadaff250097834c1.jpg)

![3c84482da6d6e005deba7dc97e8bde2b446c876423fd46b40f27de777dde0fb0.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/3c84482da6d6e005deba7dc97e8bde2b446c876423fd46b40f27de777dde0fb0.jpg)

![435dc055dfea5913f836e426a5a093c7702ffdca632ed51b538808fe45533f36.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/435dc055dfea5913f836e426a5a093c7702ffdca632ed51b538808fe45533f36.jpg)

![4649b04cff9e0159222f8db3d0ab317278b84e2db4fefeb0c034365d4c5c18b2.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/4649b04cff9e0159222f8db3d0ab317278b84e2db4fefeb0c034365d4c5c18b2.jpg)

![50bf39114e0116a032c18cd8f066b48677a96d1f7605ea282e44cc3ccfbcbc59.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/50bf39114e0116a032c18cd8f066b48677a96d1f7605ea282e44cc3ccfbcbc59.jpg)

![567086cb7742524017a5dc54b836db0692332ab43fb4694d30627e173f142364.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/567086cb7742524017a5dc54b836db0692332ab43fb4694d30627e173f142364.jpg)

![72ebaab96facd5f04d81fba8b9e9e4197a3747aa79db7c8d703e07a9fafac4a5.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/72ebaab96facd5f04d81fba8b9e9e4197a3747aa79db7c8d703e07a9fafac4a5.jpg)

![94d4d94497f5eda85255c55e43636d4e6379d0fe158e65cbcf7e44bf42193639.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/94d4d94497f5eda85255c55e43636d4e6379d0fe158e65cbcf7e44bf42193639.jpg)

![b76b2e07eb45fcb4ca588e95f3e4ebde1951cbc9d3a99923f1fe3151825b2f6d.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/b76b2e07eb45fcb4ca588e95f3e4ebde1951cbc9d3a99923f1fe3151825b2f6d.jpg)

![bd96f19200d0892d1c4fb7b11541f74ef6fc5517ea17ed845c6a4502722ef851.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/bd96f19200d0892d1c4fb7b11541f74ef6fc5517ea17ed845c6a4502722ef851.jpg)

![e9be4eeedaef052310b14fb8b0c033b37ae23343780461bfd10b0ba3b6344de4.jpg](../icml_results/4986_Non-Parametric%20Probabilistic%20Robustness_%20A%20Conservative%20Risk%20Estimator%20under%20Unknown%20Perturbation%20Di/tables/e9be4eeedaef052310b14fb8b0c033b37ae23343780461bfd10b0ba3b6344de4.jpg)

<a id="paper-4939"></a>

## From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning


### Images

![0f1402f87318ca664f8f3e438262a0c5651dcf12d5202e34811ab8be39eb3f2e.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/0f1402f87318ca664f8f3e438262a0c5651dcf12d5202e34811ab8be39eb3f2e.jpg)

![437e49351e1db056bfcf8fdda1566ff3ff749fabbd6451b7bc81bbba472ebbc7.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/437e49351e1db056bfcf8fdda1566ff3ff749fabbd6451b7bc81bbba472ebbc7.jpg)

![4e814f0baa9af35337c28bee933e4198b3a2d31d297648da54eef2f69ecb06c0.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/4e814f0baa9af35337c28bee933e4198b3a2d31d297648da54eef2f69ecb06c0.jpg)

![739f1a7effd15dca28901895007465caba7bef772d37714710ad2884609ba45c.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/739f1a7effd15dca28901895007465caba7bef772d37714710ad2884609ba45c.jpg)

![794d8389041e2ed04b5eca8555e2126b5d148ffcc7101b95afd1798a757d38ef.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/794d8389041e2ed04b5eca8555e2126b5d148ffcc7101b95afd1798a757d38ef.jpg)

![8b28d2245fd661abf04ad82cbd1d6c939b9754833be3fe86bb6e773d57d97a42.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/8b28d2245fd661abf04ad82cbd1d6c939b9754833be3fe86bb6e773d57d97a42.jpg)

![c83912a2da756be45b134ae5e87535b4e7fe9eaea6ff252bd90a5a41f01bc3fb.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/c83912a2da756be45b134ae5e87535b4e7fe9eaea6ff252bd90a5a41f01bc3fb.jpg)

![decbe00158435979965578851bea9bdebf755c78786d305259991059a6b35c39.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/decbe00158435979965578851bea9bdebf755c78786d305259991059a6b35c39.jpg)

![f808622b6b126a1816f2e4da3981c28ebb9d63aaed493eabc3b0858d15ba2a59.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/f808622b6b126a1816f2e4da3981c28ebb9d63aaed493eabc3b0858d15ba2a59.jpg)

![f9a2b7fd3b03003bdd9e20acab9a6317112f6239c143b3f63aa962be9a70ffe6.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/images/f9a2b7fd3b03003bdd9e20acab9a6317112f6239c143b3f63aa962be9a70ffe6.jpg)

### Tables

![0b8d917f642b7822fc85d89e52d811868492e50f92af1ec2327a18e80e6f6462.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/0b8d917f642b7822fc85d89e52d811868492e50f92af1ec2327a18e80e6f6462.jpg)

![17162ec66f318ab0c7f18f3096636bd3704a578705a5ddb0c5591e9e60488111.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/17162ec66f318ab0c7f18f3096636bd3704a578705a5ddb0c5591e9e60488111.jpg)

![1a9ce55a415ebb821327870b7d7b64e1c7f078da548255280fa42332eafdbbc6.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/1a9ce55a415ebb821327870b7d7b64e1c7f078da548255280fa42332eafdbbc6.jpg)

![32d4652eb7f5c227e81bd58c1f45adc0b6daf0cbb79118562aff3b1efb6f86fd.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/32d4652eb7f5c227e81bd58c1f45adc0b6daf0cbb79118562aff3b1efb6f86fd.jpg)

![369e347b0f52497dc9fab13547401c33c6478b03b52f9de3b198ddd805ec7cc0.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/369e347b0f52497dc9fab13547401c33c6478b03b52f9de3b198ddd805ec7cc0.jpg)

![5cfb2fca73df457225d8a0445f4bc43e1184f90841c6d1576daec5aab635cf45.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/5cfb2fca73df457225d8a0445f4bc43e1184f90841c6d1576daec5aab635cf45.jpg)

![91570a244564954a1b877e1e92ac3bd2a17017894ab4f4b26dca7d2439617768.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/91570a244564954a1b877e1e92ac3bd2a17017894ab4f4b26dca7d2439617768.jpg)

![9a6cd113bedf24bf981d2b2b3f54a78da5a445fbe470cf07ac6d41e45f894fb0.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/9a6cd113bedf24bf981d2b2b3f54a78da5a445fbe470cf07ac6d41e45f894fb0.jpg)

![a0653275420bd2a8143960ec40347c71c71faec8c359acb4b90aab3fc8eaa9ba.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/a0653275420bd2a8143960ec40347c71c71faec8c359acb4b90aab3fc8eaa9ba.jpg)

![b3ff193b5a33ccbd8e96d85753a3b6c3a85024641c23dda12aaaadc771515409.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/b3ff193b5a33ccbd8e96d85753a3b6c3a85024641c23dda12aaaadc771515409.jpg)

![bd80659b4d4371b2c94a250ef394e05623baa72c0713cf5b385b6d9014953566.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/bd80659b4d4371b2c94a250ef394e05623baa72c0713cf5b385b6d9014953566.jpg)

![d3d5d7e75b10a2810b44d81a8f53a61c7c376c99b293a2cc838b81adf443ee64.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/d3d5d7e75b10a2810b44d81a8f53a61c7c376c99b293a2cc838b81adf443ee64.jpg)

![f5fe64a45a562f0ac871614704207c02b54f216ecfff11b0bf391ef1bea06655.jpg](../icml_results/4987_From%20Parameter%20Dynamics%20to%20Risk%20Scoring_%20Quantifying%20Sample-Level%20Safety%20Degradation%20in%20LLM%20Fine-tun/tables/f5fe64a45a562f0ac871614704207c02b54f216ecfff11b0bf391ef1bea06655.jpg)

<a id="paper-4940"></a>

## Target-Driven Policy Optimization for Sequential Counterfactual Outcome Control


### Images

![079c9541889f657563fa99372b0397da78e6f448488c3354f3171ab7597696f6.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/079c9541889f657563fa99372b0397da78e6f448488c3354f3171ab7597696f6.jpg)

![28011fe997ee97b0722d4ce8e1990938994f0ab4ca13b08aa163b7e29b3c2748.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/28011fe997ee97b0722d4ce8e1990938994f0ab4ca13b08aa163b7e29b3c2748.jpg)

![4033073fd78985bc57a9ff23d3ef2af4ec98a9b30cf75a067ae1bcb9fc626343.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/4033073fd78985bc57a9ff23d3ef2af4ec98a9b30cf75a067ae1bcb9fc626343.jpg)

![5eb7624d8ffd7d0250805ea38e32efbdf5da293c51a263e43b01e08e0d223e85.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/5eb7624d8ffd7d0250805ea38e32efbdf5da293c51a263e43b01e08e0d223e85.jpg)

![898fd5f33457c9291177a817cd589ef437864cac06d2d33b51e0931e0ac35b83.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/898fd5f33457c9291177a817cd589ef437864cac06d2d33b51e0931e0ac35b83.jpg)

![baf7b6fcfac339bd9f38c412f284788e414c6ab46bfd1c412e426c7eee58d455.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/baf7b6fcfac339bd9f38c412f284788e414c6ab46bfd1c412e426c7eee58d455.jpg)

![c0facaa8d7632ea56720b5e6f77423499918133556b6610d72e47e9711d111e0.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/c0facaa8d7632ea56720b5e6f77423499918133556b6610d72e47e9711d111e0.jpg)

![db586473320a04b8afa06dcfb66b7472ee1a84518b5f454c63d5c858f692f4f0.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/db586473320a04b8afa06dcfb66b7472ee1a84518b5f454c63d5c858f692f4f0.jpg)

![e7f4eb59a81352aa960e5f9bbd02cd03b20e73d56d3c1e4dee9288332837f678.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/e7f4eb59a81352aa960e5f9bbd02cd03b20e73d56d3c1e4dee9288332837f678.jpg)

![f3f53ac6be794cf4696cf93fa05aeef9dd06123f2c63854ee336707742a79073.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/f3f53ac6be794cf4696cf93fa05aeef9dd06123f2c63854ee336707742a79073.jpg)

![f8223ae5c4a82310c3af19f290fcdf7592d1bd7466471708f764f2d4bc7e8ec4.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/f8223ae5c4a82310c3af19f290fcdf7592d1bd7466471708f764f2d4bc7e8ec4.jpg)

![fca13135cc139775230e85d81b2762f5c5c63aedd6874d96573367ed0603fc0f.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/images/fca13135cc139775230e85d81b2762f5c5c63aedd6874d96573367ed0603fc0f.jpg)

### Tables

![01f33802bb6d3f872fecd455b6c3d56d0fb59f7defb7405184ac9bd0c4e7ea52.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/01f33802bb6d3f872fecd455b6c3d56d0fb59f7defb7405184ac9bd0c4e7ea52.jpg)

![057d5243d10a64981ac7fc98159e024bd7eb8f1d0c8039c93988ece60bc3aa94.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/057d5243d10a64981ac7fc98159e024bd7eb8f1d0c8039c93988ece60bc3aa94.jpg)

![09a4ff6399efb6f689bb9c6afd64be5ee423f751e2ac8ae1a393b664d148f257.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/09a4ff6399efb6f689bb9c6afd64be5ee423f751e2ac8ae1a393b664d148f257.jpg)

![25799d12ced6f43380c82b1f3e228a93de7e53c1907328b00a5757148b068b0d.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/25799d12ced6f43380c82b1f3e228a93de7e53c1907328b00a5757148b068b0d.jpg)

![291093493de4f0ea403108bbba0540aedb7c18faf953c1f56dba48dee6f62144.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/291093493de4f0ea403108bbba0540aedb7c18faf953c1f56dba48dee6f62144.jpg)

![2b3daf405d19e75feadfe95dbc0adde272ba77855b81eb9a3c456a7ecada136f.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/2b3daf405d19e75feadfe95dbc0adde272ba77855b81eb9a3c456a7ecada136f.jpg)

![431b816353b49c4009d5e324b42819cb0fe89ef75d41a1a66b1f99ace4510f5e.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/431b816353b49c4009d5e324b42819cb0fe89ef75d41a1a66b1f99ace4510f5e.jpg)

![5ea11144ad1f763cdf59ec16375eeb994fb05e2508336ec16efb6bb717bbee09.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/5ea11144ad1f763cdf59ec16375eeb994fb05e2508336ec16efb6bb717bbee09.jpg)

![673a82cd53b14cf7dee0f5d9453c77a0705b5bd1faeed331d5721bb37e077e40.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/673a82cd53b14cf7dee0f5d9453c77a0705b5bd1faeed331d5721bb37e077e40.jpg)

![7be26eedc3aaf0a0b85247a6c3a521fb8ec928e832404f9e546e427b83579a90.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/7be26eedc3aaf0a0b85247a6c3a521fb8ec928e832404f9e546e427b83579a90.jpg)

![aa5f22f52b685bb898d977af1b83d6b881e107b61d26a134188e96eaee87c993.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/aa5f22f52b685bb898d977af1b83d6b881e107b61d26a134188e96eaee87c993.jpg)

![ab9f4d2ad92554e0ebf5b643b2b4e4e693bc3e382bc0a1b71c2120e8897efba1.jpg](../icml_results/4988_Target-Driven%20Policy%20Optimization%20for%20Sequential%20Counterfactual%20Outcome%20Control/tables/ab9f4d2ad92554e0ebf5b643b2b4e4e693bc3e382bc0a1b71c2120e8897efba1.jpg)

<a id="paper-4941"></a>

## Dual-branch Robust Unlearnable Examples


### Images

![2dbba16cd60a22f913d9539fe3749906e0ed2c9d0423c213b242737b5b7d2462.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/2dbba16cd60a22f913d9539fe3749906e0ed2c9d0423c213b242737b5b7d2462.jpg)

![6be8b225f4b0f58abab76643778d4cf5615426b4aff35af161d6b0eb490ae506.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/6be8b225f4b0f58abab76643778d4cf5615426b4aff35af161d6b0eb490ae506.jpg)

![70f0e2a28fe5008198ac6202a46c8f5dad1a8f14ef1fa3e26200e334bca8d266.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/70f0e2a28fe5008198ac6202a46c8f5dad1a8f14ef1fa3e26200e334bca8d266.jpg)

![99c106ddd8170349aa3af88ba48c1d51739eb03789997fa5421dcd38795eca3a.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/99c106ddd8170349aa3af88ba48c1d51739eb03789997fa5421dcd38795eca3a.jpg)

![a393de20b0828c497a69c5f4684c911c858783e77a6d3b561ee49008c20b8ea4.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/a393de20b0828c497a69c5f4684c911c858783e77a6d3b561ee49008c20b8ea4.jpg)

![d9164f4f83b947c6ff881f27fb24b66212a9460b4258190e5e42ccba689b23f2.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/d9164f4f83b947c6ff881f27fb24b66212a9460b4258190e5e42ccba689b23f2.jpg)

![f2d94fae829cd27b397e5fad703cd5d392a2ad344cf5c2c6696000ec4dadf655.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/f2d94fae829cd27b397e5fad703cd5d392a2ad344cf5c2c6696000ec4dadf655.jpg)

![f91c8c8b1b4541d56f51affc46194516c97e16894222c9d5f56ed0bd81f452a1.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/images/f91c8c8b1b4541d56f51affc46194516c97e16894222c9d5f56ed0bd81f452a1.jpg)

### Tables

![2236210116ae9246ed85e004bbcfebcb24e792a24511621e9bc72879f23fa127.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/2236210116ae9246ed85e004bbcfebcb24e792a24511621e9bc72879f23fa127.jpg)

![4a4a32d3cfc5955e60d71c9b4c1f3e3328d9f33a5b1daa8050640b04e655fdbb.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/4a4a32d3cfc5955e60d71c9b4c1f3e3328d9f33a5b1daa8050640b04e655fdbb.jpg)

![5bea880cb17f504ebe03d6ec8da94fb07819ceb18a54b7dc4b95bcbc6339612a.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/5bea880cb17f504ebe03d6ec8da94fb07819ceb18a54b7dc4b95bcbc6339612a.jpg)

![83d80582a9da9267f8edf75971223c6b394ecc555d2e1f1f908a4a535519e2bd.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/83d80582a9da9267f8edf75971223c6b394ecc555d2e1f1f908a4a535519e2bd.jpg)

![8a5afda4c35d8e6fe448d86f6fe998bf7536268a054e7664aa7c7a6a32d4248a.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/8a5afda4c35d8e6fe448d86f6fe998bf7536268a054e7664aa7c7a6a32d4248a.jpg)

![8b68f260e1f53c2f51bbf30446a6552a2b401574b529b221b978f3735373727a.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/8b68f260e1f53c2f51bbf30446a6552a2b401574b529b221b978f3735373727a.jpg)

![ce5c4f65d2c531f5baaa93abce406704e127feaab8c22413cefe7719917a370d.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/ce5c4f65d2c531f5baaa93abce406704e127feaab8c22413cefe7719917a370d.jpg)

![d466c13c47a83607c1b2a9c172b30868f45986be66d4f5739ba56005c973a1c5.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/d466c13c47a83607c1b2a9c172b30868f45986be66d4f5739ba56005c973a1c5.jpg)

![eb2f5da4585b395e3f1ad5aeb5c068e9a4adf78cbf28f842851141bec1a72ea8.jpg](../icml_results/4989_Dual-branch%20Robust%20Unlearnable%20Examples/tables/eb2f5da4585b395e3f1ad5aeb5c068e9a4adf78cbf28f842851141bec1a72ea8.jpg)

<a id="paper-4942"></a>

## Finding Stationary Points by Comparisons


### Images

![2956c25ef0466c581acc8bfb43686e13bb9f742d51da59e7fcebb9b7a605f796.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/images/2956c25ef0466c581acc8bfb43686e13bb9f742d51da59e7fcebb9b7a605f796.jpg)

![d479611501c6e3fb98470a567e08c83d90faaf68d416a9ae645519d0fc9baa55.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/images/d479611501c6e3fb98470a567e08c83d90faaf68d416a9ae645519d0fc9baa55.jpg)

![ee27e8d0685835f0480f908e4f2a4be2871a6095939f31ecea33b5c748ce9e20.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/images/ee27e8d0685835f0480f908e4f2a4be2871a6095939f31ecea33b5c748ce9e20.jpg)

![f13d17f5571a311e8616c92d4c01e3f38eaeb1929beaa744c815e591f5fe924e.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/images/f13d17f5571a311e8616c92d4c01e3f38eaeb1929beaa744c815e591f5fe924e.jpg)

![f6ff1dff6cd1773913ec4a02751fa55f938746d374e9fb5906db0e08fe880882.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/images/f6ff1dff6cd1773913ec4a02751fa55f938746d374e9fb5906db0e08fe880882.jpg)

### Tables

![259e4b13cb8bb5b868e7f2b5b9ef786f9686facc748f1f1f6fec9b9564edc941.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/tables/259e4b13cb8bb5b868e7f2b5b9ef786f9686facc748f1f1f6fec9b9564edc941.jpg)

![984e56433a53cd68b5e318f3fc5dee83faaad6277b335b95f6a9c5bc8ad24df9.jpg](../icml_results/4990_Finding%20Stationary%20Points%20by%20Comparisons/tables/984e56433a53cd68b5e318f3fc5dee83faaad6277b335b95f6a9c5bc8ad24df9.jpg)

<a id="paper-4943"></a>

## AVI-Bench: Toward Human-like Audio-Visual Intelligence of Omni-MLLMs


### Images

![0b5afb6f60116770d6dd14fe9fc251c4b243b5fc43a7371ee74392edd7243ede.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/0b5afb6f60116770d6dd14fe9fc251c4b243b5fc43a7371ee74392edd7243ede.jpg)

![3a90d896cb8163a5c37550822019a8bd1b86fb34566cd0da203b454a7e0cd0bd.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/3a90d896cb8163a5c37550822019a8bd1b86fb34566cd0da203b454a7e0cd0bd.jpg)

![53a90093b657b82f88f83821c1891f5c5724a77d1774a1b2459173b5599740fa.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/53a90093b657b82f88f83821c1891f5c5724a77d1774a1b2459173b5599740fa.jpg)

![7b72d38ac4109d74b115a20d8cf0d757e22d1452163d93dd39bfbf26e7f39453.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/7b72d38ac4109d74b115a20d8cf0d757e22d1452163d93dd39bfbf26e7f39453.jpg)

![84bf3ae585985fcd383582f079485faa8c1749f3edba6e5667fe37045c5ffd99.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/84bf3ae585985fcd383582f079485faa8c1749f3edba6e5667fe37045c5ffd99.jpg)

![94ab06f101c5ab427cfffac5ac7430720b418420411fd6b5ab28080155a88148.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/94ab06f101c5ab427cfffac5ac7430720b418420411fd6b5ab28080155a88148.jpg)

![9607556d9e8fd709925e31af761e10ada15959dd586ec8995bb5699803473eaf.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/9607556d9e8fd709925e31af761e10ada15959dd586ec8995bb5699803473eaf.jpg)

![9a2738ca55c7a0115b5563c9e131e7a35d71d66476d2b24999b059afb0e1357c.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/9a2738ca55c7a0115b5563c9e131e7a35d71d66476d2b24999b059afb0e1357c.jpg)

![de52fdd0ec25779e157e3e9ff61e9d59084a0530a8c29fe49d142dd9e7d1687f.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/images/de52fdd0ec25779e157e3e9ff61e9d59084a0530a8c29fe49d142dd9e7d1687f.jpg)

### Tables

![0dd0294197659de5308d3e04ece60c8d0993283684cc4bb83957e501937f1418.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/0dd0294197659de5308d3e04ece60c8d0993283684cc4bb83957e501937f1418.jpg)

![191a1e1d55b3cd1964e86ce4c5110fa792991f5b2798c10e14c50b22ef131def.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/191a1e1d55b3cd1964e86ce4c5110fa792991f5b2798c10e14c50b22ef131def.jpg)

![4a08578e7837b923b424d78341e83892f1a885ce1942b787e041ca444d614089.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/4a08578e7837b923b424d78341e83892f1a885ce1942b787e041ca444d614089.jpg)

![74590e3bec99ccb7cccafb98696ef920b263db5ff284e383bbc27ca5ca544c47.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/74590e3bec99ccb7cccafb98696ef920b263db5ff284e383bbc27ca5ca544c47.jpg)

![773da943aa45561ecf921713c9ceb2bbb883d1ed3304d2b11ff4ccfc387d1458.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/773da943aa45561ecf921713c9ceb2bbb883d1ed3304d2b11ff4ccfc387d1458.jpg)

![95a566d930e46354ff84eb1c7b0928d89e6d56ae2f46951fea6b35b96a07b19b.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/95a566d930e46354ff84eb1c7b0928d89e6d56ae2f46951fea6b35b96a07b19b.jpg)

![ae3711f99e4e0741e9b62a9d25139bb80617734fd9777387d4380ca95646e67d.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/ae3711f99e4e0741e9b62a9d25139bb80617734fd9777387d4380ca95646e67d.jpg)

![b3aa4e198d8861a97aded2b618ad2ba322fd5f8d4a793e4c3d7a195dff04eea0.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/b3aa4e198d8861a97aded2b618ad2ba322fd5f8d4a793e4c3d7a195dff04eea0.jpg)

![b8b2d5d2582e08572aebb75064b51656e7469aa4abda12084d569e2524bb6204.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/b8b2d5d2582e08572aebb75064b51656e7469aa4abda12084d569e2524bb6204.jpg)

![be1e9451ceb8a485f46d18b1273601820d34b101cfcefcd7ceb1317b5c7cf93e.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/be1e9451ceb8a485f46d18b1273601820d34b101cfcefcd7ceb1317b5c7cf93e.jpg)

![e721b28b8e766d9d0ac098cf732bdf237bef06199f297bd41a0ab1a8284b616a.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/e721b28b8e766d9d0ac098cf732bdf237bef06199f297bd41a0ab1a8284b616a.jpg)

![eb96aac849efe9cb3c5bf677e31c325ff3cebc9b1cf2ffab9e6f00c8932c89ab.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/eb96aac849efe9cb3c5bf677e31c325ff3cebc9b1cf2ffab9e6f00c8932c89ab.jpg)

![f0ab947065c1f87d5cb1774f3716dd1ca728a33adc9156742dd5ca5fb1907cfd.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/f0ab947065c1f87d5cb1774f3716dd1ca728a33adc9156742dd5ca5fb1907cfd.jpg)

![f9c63a6f6e3eb836d6e21e875d50968230840996c04cb578d22ea1396d8abde2.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/f9c63a6f6e3eb836d6e21e875d50968230840996c04cb578d22ea1396d8abde2.jpg)

![fd6a281155bb1e96442d9aecbd91c35cd6e2b6927a4898934218f5ca84a614dd.jpg](../icml_results/4991_AVI-Bench_%20Toward%20Human-like%20Audio-Visual%20Intelligence%20of%20Omni-MLLMs/tables/fd6a281155bb1e96442d9aecbd91c35cd6e2b6927a4898934218f5ca84a614dd.jpg)

<a id="paper-4944"></a>

## CLINIC: Towards High-quality Graph Out-Of-Distribution Detection


### Images

![0ec3820465951a994c46e7cfed951d1322f0223c626d9ef7023efe4060beb969.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/0ec3820465951a994c46e7cfed951d1322f0223c626d9ef7023efe4060beb969.jpg)

![0ee99db97f76d2acf55a27d7993e1f3216970aba138936d73469075e3a17dc24.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/0ee99db97f76d2acf55a27d7993e1f3216970aba138936d73469075e3a17dc24.jpg)

![1536d0cf8c88ec9849db8f402865fa1d804bd1eaadac25fa42707d889841b1ae.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/1536d0cf8c88ec9849db8f402865fa1d804bd1eaadac25fa42707d889841b1ae.jpg)

![20de9518fd976d82fb2011c26aec0a0567ffb5c676cd4da875b60e08fb5f39a8.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/20de9518fd976d82fb2011c26aec0a0567ffb5c676cd4da875b60e08fb5f39a8.jpg)

![2778337943dbb7a1c75a910f209f0e2f807e2b14e6fff7cad98fd09002e842a5.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/2778337943dbb7a1c75a910f209f0e2f807e2b14e6fff7cad98fd09002e842a5.jpg)

![8b720709f0fd59ff90fdc538a25b0d0c4de8c464e7d03b301d430d926a610886.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/8b720709f0fd59ff90fdc538a25b0d0c4de8c464e7d03b301d430d926a610886.jpg)

![b7933cc55776ebea3847e2c72d604c851e5c084b2bd46089e49e2e57e681f80d.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/images/b7933cc55776ebea3847e2c72d604c851e5c084b2bd46089e49e2e57e681f80d.jpg)

### Tables

![5758b28264d7f3476acd9d65bd8e05b45cd0b1caf47937b203329453ccd86d00.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/tables/5758b28264d7f3476acd9d65bd8e05b45cd0b1caf47937b203329453ccd86d00.jpg)

![9dd2b9c498b304abbd40f3663eb9f60ee4084748cdae41d2b10c677cce1321cc.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/tables/9dd2b9c498b304abbd40f3663eb9f60ee4084748cdae41d2b10c677cce1321cc.jpg)

![ae10ab6e650a2a713c79e1d8356c3979f7cf731f5e09445b9b4ed416d656b8bc.jpg](../icml_results/4992_CLINIC_%20Towards%20High-quality%20Graph%20Out-Of-Distribution%20Detection/tables/ae10ab6e650a2a713c79e1d8356c3979f7cf731f5e09445b9b4ed416d656b8bc.jpg)

<a id="paper-4945"></a>

## Scalable and Stable Estimation of Amari $α$-Divergence using Random Fourier Features


### Images

![09cf1f7e92477ad54aece2e9fc30a1e19d5e395216a11889ddb7ff5a7d1f3d18.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/images/09cf1f7e92477ad54aece2e9fc30a1e19d5e395216a11889ddb7ff5a7d1f3d18.jpg)

![5372ee080bf8466f568ac7345ae5866975dc97c3514c777fa7162e9113033a69.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/images/5372ee080bf8466f568ac7345ae5866975dc97c3514c777fa7162e9113033a69.jpg)

![63d7ceeec945119531acbf749caccdf907b75338d8d84f410df6d7c31f00750e.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/images/63d7ceeec945119531acbf749caccdf907b75338d8d84f410df6d7c31f00750e.jpg)

![b43405457650bb0d1b91177a2942889f5f0fc037086d0783495ee55f5872f95c.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/images/b43405457650bb0d1b91177a2942889f5f0fc037086d0783495ee55f5872f95c.jpg)

### Tables

![3f3624ad4fbc55672a09ea373e3181dfce94677e8b71224b4664765d2b32fa64.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/tables/3f3624ad4fbc55672a09ea373e3181dfce94677e8b71224b4664765d2b32fa64.jpg)

![f0bc7ad17b50ac5cf810c7c501ef19b5bed0d1e451b90a05c67da8a3c67ee39d.jpg](../icml_results/4993_Scalable%20and%20Stable%20Estimation%20of%20Amari%20%24%CE%B1%24-Divergence%20using%20Random%20Fourier%20Features/tables/f0bc7ad17b50ac5cf810c7c501ef19b5bed0d1e451b90a05c67da8a3c67ee39d.jpg)

<a id="paper-4946"></a>

## ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning


### Images

![0ed2864cd917dc027061ddb67d88f476566df89d87bd4fd3253c8a45b61535a6.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/0ed2864cd917dc027061ddb67d88f476566df89d87bd4fd3253c8a45b61535a6.jpg)

![13e492889076eeeb665978757750774600155790ddb7f0b60da567859d0e3cc0.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/13e492889076eeeb665978757750774600155790ddb7f0b60da567859d0e3cc0.jpg)

![13e6d6c82e6aa2a5d7cc450a63aeffcf2bf7f105b77f168e33d898d1e55a76eb.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/13e6d6c82e6aa2a5d7cc450a63aeffcf2bf7f105b77f168e33d898d1e55a76eb.jpg)

![1e19f9017a8e0f3aca9d1ed6012e3cce7e08291154752da08572f6072043af35.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/1e19f9017a8e0f3aca9d1ed6012e3cce7e08291154752da08572f6072043af35.jpg)

![280ad02a26a25dee5fd4dcef932323ef3d0e8d96cbb1b71be4a45c49a6b92cea.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/280ad02a26a25dee5fd4dcef932323ef3d0e8d96cbb1b71be4a45c49a6b92cea.jpg)

![496036f591f39c62daddbb238a474832a42b936c49ae79bd8cdaceed6c0e98cf.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/496036f591f39c62daddbb238a474832a42b936c49ae79bd8cdaceed6c0e98cf.jpg)

![6980af52d19b481b2892a79cf69a9aa681606621844310136ffbd3c9615e33c5.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/6980af52d19b481b2892a79cf69a9aa681606621844310136ffbd3c9615e33c5.jpg)

![801b4a2a6fd648284f0082fe590c9ac745a674619970348743d0601de4e98b7c.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/801b4a2a6fd648284f0082fe590c9ac745a674619970348743d0601de4e98b7c.jpg)

![93e03d852151ca73b1b63312747035d7515f3cd47ba9878bf450fcc5d24659b8.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/93e03d852151ca73b1b63312747035d7515f3cd47ba9878bf450fcc5d24659b8.jpg)

![a75e489f645ad8ffe4d6756c6e4ef2d2eed334684e019b8439646533fb82447f.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/a75e489f645ad8ffe4d6756c6e4ef2d2eed334684e019b8439646533fb82447f.jpg)

![a9468a4ef9c9b00c6821b0c88959a376711e29684fb551420b09df51de7c415e.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/a9468a4ef9c9b00c6821b0c88959a376711e29684fb551420b09df51de7c415e.jpg)

![b85f9acac289e1be9a95488e3fb9d85a2a60c0805bbbdeb8176782d7ed8e4dc3.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/images/b85f9acac289e1be9a95488e3fb9d85a2a60c0805bbbdeb8176782d7ed8e4dc3.jpg)

### Tables

![16504b07fc8b4cdde321ab394188e59b00a8185216468d66bf76a6655e8bcdef.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/16504b07fc8b4cdde321ab394188e59b00a8185216468d66bf76a6655e8bcdef.jpg)

![4098126295c7aaad95fb6e000e6d6016be4689ce5427231977339d59240862ee.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/4098126295c7aaad95fb6e000e6d6016be4689ce5427231977339d59240862ee.jpg)

![5a426892fc09806bbf0b54ee84582d227b186163ea6d18db0dbd1b7785abc93e.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/5a426892fc09806bbf0b54ee84582d227b186163ea6d18db0dbd1b7785abc93e.jpg)

![5c2e52bc14324ec48aa3d5792bf063492859b45cc78a4328829c9f8bea6b6ef3.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/5c2e52bc14324ec48aa3d5792bf063492859b45cc78a4328829c9f8bea6b6ef3.jpg)

![60fc869f1a012bb387c100ce4804afaf3aad782c1f16a3290ea907747c309288.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/60fc869f1a012bb387c100ce4804afaf3aad782c1f16a3290ea907747c309288.jpg)

![7177a5852ebb5f21748616de804e23c0ca181a018e72c56924ea19f44c4c3412.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/7177a5852ebb5f21748616de804e23c0ca181a018e72c56924ea19f44c4c3412.jpg)

![8a31aeb4bbe697007e6caaaf5b2c166515cdea2a62f9443c2d4b8c66fe8ef996.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/8a31aeb4bbe697007e6caaaf5b2c166515cdea2a62f9443c2d4b8c66fe8ef996.jpg)

![8ca5cadcff03eff93176157339570939389b3b11c48531ab46af348af263b8c0.jpg](../icml_results/4994_ARLArena_%20A%20Unified%20Framework%20for%20Stable%20Agentic%20Reinforcement%20Learning/tables/8ca5cadcff03eff93176157339570939389b3b11c48531ab46af348af263b8c0.jpg)

<a id="paper-4947"></a>

## LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models


### Images

![0b5cae06b35bc4251e247d671249a397d40ba591b6621e122168705ea3e723e1.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/0b5cae06b35bc4251e247d671249a397d40ba591b6621e122168705ea3e723e1.jpg)

![0d5680ac965e6dbe498f8d0b26449a38163565186a7f238a3a1ba981ed1bd271.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/0d5680ac965e6dbe498f8d0b26449a38163565186a7f238a3a1ba981ed1bd271.jpg)

![471db623fcf9ddf5b8074f2baf16c22335616fe1d0fd7b29c1f5180ea38dc9de.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/471db623fcf9ddf5b8074f2baf16c22335616fe1d0fd7b29c1f5180ea38dc9de.jpg)

![548329899bfc82afb6d1ff3611457cfe776006ae4861056f4eaac88ec4d8c8cb.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/548329899bfc82afb6d1ff3611457cfe776006ae4861056f4eaac88ec4d8c8cb.jpg)

![71587c7701ee4c88e9906a1046923a2ea53ecf8492e009aaa04cc5b24e34f5e4.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/71587c7701ee4c88e9906a1046923a2ea53ecf8492e009aaa04cc5b24e34f5e4.jpg)

![89ffe24fc1c90c4e917be0ba377f4a24949a49b4d74ce22b322b8da83aa18424.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/89ffe24fc1c90c4e917be0ba377f4a24949a49b4d74ce22b322b8da83aa18424.jpg)

![c715d89d01c9d8db3af9b2d98f97254b19f2c4498bbc3bbfa0aa6afb23a13f45.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/images/c715d89d01c9d8db3af9b2d98f97254b19f2c4498bbc3bbfa0aa6afb23a13f45.jpg)

### Tables

![014f30e43910ec7b72f986abe4d801fc746402e8b27b3c3ca35feef93b585908.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/014f30e43910ec7b72f986abe4d801fc746402e8b27b3c3ca35feef93b585908.jpg)

![034066fd18c9ad545827d299ff7f64fb0fc93cab3a135481cce49add24ea3a71.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/034066fd18c9ad545827d299ff7f64fb0fc93cab3a135481cce49add24ea3a71.jpg)

![07f7eb4742eb0f9ea60be2d5c0af061d922b38638e38854275c89cf6af913717.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/07f7eb4742eb0f9ea60be2d5c0af061d922b38638e38854275c89cf6af913717.jpg)

![0d21fc84074f568d9a6e396aea48477da9c55426e95225bfe66748240df58f58.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/0d21fc84074f568d9a6e396aea48477da9c55426e95225bfe66748240df58f58.jpg)

![252882419c9a06ef415cd1a9b6cc0d72167542a25f02f4d0621e08e73e55d7ed.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/252882419c9a06ef415cd1a9b6cc0d72167542a25f02f4d0621e08e73e55d7ed.jpg)

![271d8ee1f6bb1523d49988edac5ca1fb028f5542df3ce3b7ed9dd11dbf79f1f4.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/271d8ee1f6bb1523d49988edac5ca1fb028f5542df3ce3b7ed9dd11dbf79f1f4.jpg)

![46de909928d316dbf603e1617f7ad83357ca360b5fd44dfd4f8ff29c2003bf49.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/46de909928d316dbf603e1617f7ad83357ca360b5fd44dfd4f8ff29c2003bf49.jpg)

![4a1f87196ad30d7bb85ce978c5ee69ee3b552d040ed1616b3e4aebfbf60b930a.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/4a1f87196ad30d7bb85ce978c5ee69ee3b552d040ed1616b3e4aebfbf60b930a.jpg)

![4abade19b55366d958fa802d8355732aaa2a4b81cdb17f4938c4ba43b882f036.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/4abade19b55366d958fa802d8355732aaa2a4b81cdb17f4938c4ba43b882f036.jpg)

![5355149762e8df25373c17fc9f11c38d71331eab86d4f0edea6b7b2aaf648119.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/5355149762e8df25373c17fc9f11c38d71331eab86d4f0edea6b7b2aaf648119.jpg)

![62e5c00d92aa3d37e47cb539d8fa47fd03ff0dc4b7eea2f5c7801b8fd335c91b.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/62e5c00d92aa3d37e47cb539d8fa47fd03ff0dc4b7eea2f5c7801b8fd335c91b.jpg)

![690a074925ce0286501d96f2107cb59c8087e1292901864f20e01c3721f714b8.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/690a074925ce0286501d96f2107cb59c8087e1292901864f20e01c3721f714b8.jpg)

![7789b54504ee5712eb42fcbd283d07568de70fd4e02e85254cc568ba503d5d13.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/7789b54504ee5712eb42fcbd283d07568de70fd4e02e85254cc568ba503d5d13.jpg)

![859afa00456538d51a7be6373a2f3fd29fe68e16d842f24a73ebed5f92c1d85a.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/859afa00456538d51a7be6373a2f3fd29fe68e16d842f24a73ebed5f92c1d85a.jpg)

![86eb7031c2a0383fc17a5cd2e95ca8b75683f1c998f6b727e80e747bacad7175.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/86eb7031c2a0383fc17a5cd2e95ca8b75683f1c998f6b727e80e747bacad7175.jpg)

![97cb2dc1cf76dab964d0fa2d2b3392f56374e581195166d8535e58ce92ee958f.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/97cb2dc1cf76dab964d0fa2d2b3392f56374e581195166d8535e58ce92ee958f.jpg)

![acfda18fc5c9b3cb59ed92ebbac03c1b80f6bbe694f455bd3d9ce7431ee3416c.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/acfda18fc5c9b3cb59ed92ebbac03c1b80f6bbe694f455bd3d9ce7431ee3416c.jpg)

![b31a9cdd79efce8f586b190eae6d02c42baa426217be387813329f9f48f19a31.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/b31a9cdd79efce8f586b190eae6d02c42baa426217be387813329f9f48f19a31.jpg)

![b5e92c3f6bc059de5b9c144101d344e438a4711b8482a49405db3b6e65705886.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/b5e92c3f6bc059de5b9c144101d344e438a4711b8482a49405db3b6e65705886.jpg)

![babf931969537ac5d9d3bb64b99cc016ac81335e4c309be92f829ff86a41cd1f.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/babf931969537ac5d9d3bb64b99cc016ac81335e4c309be92f829ff86a41cd1f.jpg)

![c54208974203b6306babc53790b2d91a9ed46f5ca57c435c21d3d4fd12adcc8d.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/c54208974203b6306babc53790b2d91a9ed46f5ca57c435c21d3d4fd12adcc8d.jpg)

![cf6a8aa327577bd2de60fcfc59e92acebba1989bde4d8839f3269f74a32708a2.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/cf6a8aa327577bd2de60fcfc59e92acebba1989bde4d8839f3269f74a32708a2.jpg)

![d1ae801c0d57476012cee44b23991972d3b82e37dbeab80607c75e5cdce82f59.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/d1ae801c0d57476012cee44b23991972d3b82e37dbeab80607c75e5cdce82f59.jpg)

![e7c12386f23b46f88cfa1bef5c95c6eec2c0171164c1f70d92e47602213f4848.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/e7c12386f23b46f88cfa1bef5c95c6eec2c0171164c1f70d92e47602213f4848.jpg)

![f2ca0beb30a5067a72e0519a2a0b6cb9945abd1b396c79bdcb109f072248e9f6.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/f2ca0beb30a5067a72e0519a2a0b6cb9945abd1b396c79bdcb109f072248e9f6.jpg)

![fc46e90aab40bda39a9920b5a58af9abaa912f699794c888a3d8f84ea36e076b.jpg](../icml_results/4995_LimiX-2M_%20Mitigating%20Low-Rank%20Collapse%20and%20Attention%20Bottlenecks%20in%20Tabular%20Foundation%20Models/tables/fc46e90aab40bda39a9920b5a58af9abaa912f699794c888a3d8f84ea36e076b.jpg)

<a id="paper-4948"></a>

## Autobidding Auctions with LLM-Powered Creatives


### Images

![68102d8a84e5059343a7bb3bbfa834cc2b7a830be8c84a61e50852d78bc1b81f.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/images/68102d8a84e5059343a7bb3bbfa834cc2b7a830be8c84a61e50852d78bc1b81f.jpg)

![f91ef629247f63f49a7a3bb24c430620d564e2b3ac9f64d4f1ffd0e7f86ad33d.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/images/f91ef629247f63f49a7a3bb24c430620d564e2b3ac9f64d4f1ffd0e7f86ad33d.jpg)

### Tables

![301fbd1c943197eace9bf388f35ccc567ac1b0b416299bba46dbb593ce394fb7.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/301fbd1c943197eace9bf388f35ccc567ac1b0b416299bba46dbb593ce394fb7.jpg)

![4804a2e5c86b17007088152fbf254661dc23a61acce69825cddf6a3a5002c538.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/4804a2e5c86b17007088152fbf254661dc23a61acce69825cddf6a3a5002c538.jpg)

![afb76fcdac5a286129a1af05a4c6b55e7cb9e9a9df45ad53e91781ad8dbd9411.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/afb76fcdac5a286129a1af05a4c6b55e7cb9e9a9df45ad53e91781ad8dbd9411.jpg)

![b137bd809edeb2a00728a31c92329eeaba15346f881a7697f0d2698da78d6d69.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/b137bd809edeb2a00728a31c92329eeaba15346f881a7697f0d2698da78d6d69.jpg)

![b4abf3e1d83ea0a79842fd4755b6c1fe213577efed821ece8ffb45a61ef38deb.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/b4abf3e1d83ea0a79842fd4755b6c1fe213577efed821ece8ffb45a61ef38deb.jpg)

![ecfb5af33dcaac1b7a75c7450d154925e4ffc70de7a334a2c623a023b1e145ad.jpg](../icml_results/4996_Autobidding%20Auctions%20with%20LLM-Powered%20Creatives/tables/ecfb5af33dcaac1b7a75c7450d154925e4ffc70de7a334a2c623a023b1e145ad.jpg)

<a id="paper-4949"></a>

## Do LLMs "Feel"? Emotion Circuits Discovery and Control


### Images

![0fec660f341e0587ebd813b87c847f79d2f7ca6198feeba01f9ff8fcffe7ffff.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/0fec660f341e0587ebd813b87c847f79d2f7ca6198feeba01f9ff8fcffe7ffff.jpg)

![219916312d476873c92f4cdb2e46c23bfa45341fe1d900d152c4e72e8463c61e.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/219916312d476873c92f4cdb2e46c23bfa45341fe1d900d152c4e72e8463c61e.jpg)

![2b22bba4414fda5e417aae0bf9f70f387a128e5ea4f2ede35a2bbc030472177f.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/2b22bba4414fda5e417aae0bf9f70f387a128e5ea4f2ede35a2bbc030472177f.jpg)

![384488ff3595804919b2a914386d28fcfa8b4c1f76cf6123bede85a4ce4dd887.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/384488ff3595804919b2a914386d28fcfa8b4c1f76cf6123bede85a4ce4dd887.jpg)

![4beb8620296f2010d1e99b1495db6cf2f9e89b1bd0ee8ca2129fa2871a2a1372.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/4beb8620296f2010d1e99b1495db6cf2f9e89b1bd0ee8ca2129fa2871a2a1372.jpg)

![92e1ecf655c22660e1cba641361ec51da6471228ae9c049850e40c5708fb22d5.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/92e1ecf655c22660e1cba641361ec51da6471228ae9c049850e40c5708fb22d5.jpg)

![a85651d0ba2ba8a279c7f3c368a86dbe8334c8559fd152498fda60e50c10a533.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/a85651d0ba2ba8a279c7f3c368a86dbe8334c8559fd152498fda60e50c10a533.jpg)

![d9d5bb267d61edfd5ec97a9967cdfc7d50455e86f5aa395b30ec8af169c54195.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/images/d9d5bb267d61edfd5ec97a9967cdfc7d50455e86f5aa395b30ec8af169c54195.jpg)

### Tables

![04f4b7a0eb0acf69a0dc99ea7a12627ffe32bd451ee55177bb04061ecb18cd14.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/04f4b7a0eb0acf69a0dc99ea7a12627ffe32bd451ee55177bb04061ecb18cd14.jpg)

![070bea42155ab584372f86bb3f1e6dc17efe69661fc5d2cd30822b207c901538.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/070bea42155ab584372f86bb3f1e6dc17efe69661fc5d2cd30822b207c901538.jpg)

![29fe9450af90ad03b09112021a31eddb6a38f9fe69aeb3c89e5f2ded7941537a.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/29fe9450af90ad03b09112021a31eddb6a38f9fe69aeb3c89e5f2ded7941537a.jpg)

![6695e38f79eb767742ae1bb6887d7e66d3089e17f9d525e75e04a3121d3ec15e.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/6695e38f79eb767742ae1bb6887d7e66d3089e17f9d525e75e04a3121d3ec15e.jpg)

![88e117c8f50a58402472e88f01a115412e6e82ccb7d2ec7be7979506b5d4847e.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/88e117c8f50a58402472e88f01a115412e6e82ccb7d2ec7be7979506b5d4847e.jpg)

![a0514d3dd60858c1dbf26c781178928033aae78e7f0cc5ca68ee5a3749d6abf0.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/a0514d3dd60858c1dbf26c781178928033aae78e7f0cc5ca68ee5a3749d6abf0.jpg)

![ab53f14097bc4fe26fd48bb272983f42dd2e8be76f26fd30d875706fab4caf94.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/ab53f14097bc4fe26fd48bb272983f42dd2e8be76f26fd30d875706fab4caf94.jpg)

![c1b9a3fb456a40b656f9f2abc85c481a0219bd242c6ba7d7992a95008e79bcc6.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/c1b9a3fb456a40b656f9f2abc85c481a0219bd242c6ba7d7992a95008e79bcc6.jpg)

![d43b873a1eb95f1f3dd8f8bd1b2f21858896614962c391a1f678f6693d49728f.jpg](../icml_results/4997_Do%20LLMs%20_Feel__%20Emotion%20Circuits%20Discovery%20and%20Control/tables/d43b873a1eb95f1f3dd8f8bd1b2f21858896614962c391a1f678f6693d49728f.jpg)

<a id="paper-4950"></a>

## Co-Evolving Latent Action World Models

### Images

![40aefaf392b4ceb3f8c8677ffce494ba384396aa058b0de3204fdce930e2e7af.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/40aefaf392b4ceb3f8c8677ffce494ba384396aa058b0de3204fdce930e2e7af.jpg)

![585a49742032168314510a480f411c309519e37985c6e14f293a35507dc7ac45.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/585a49742032168314510a480f411c309519e37985c6e14f293a35507dc7ac45.jpg)

![738cbf5d1a64f26ff48b84b600db6a01fce3f7830389d7aeb428b3ee1e7247c0.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/738cbf5d1a64f26ff48b84b600db6a01fce3f7830389d7aeb428b3ee1e7247c0.jpg)

![75db6550cdeb594d6c6de2d9e33419c4cb521978cd7be7bd2841632bac3069dc.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/75db6550cdeb594d6c6de2d9e33419c4cb521978cd7be7bd2841632bac3069dc.jpg)

![9ea7bf17794da23c54dfd145421d5fedc516486f73af88b5f3affd6ad18de054.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/9ea7bf17794da23c54dfd145421d5fedc516486f73af88b5f3affd6ad18de054.jpg)

![ad603919e796da8aff6ec46943b766e98b101ce37d035bd412c9d08065057f4a.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/ad603919e796da8aff6ec46943b766e98b101ce37d035bd412c9d08065057f4a.jpg)

![c0eee348727a3493eda5a98be3e81c6421df9dcded71ff84fe60e5045087571e.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/c0eee348727a3493eda5a98be3e81c6421df9dcded71ff84fe60e5045087571e.jpg)

![ed03abdeb491168ba275541eee99379bb39d6348b7f5215359cd48da7e07ddb3.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/ed03abdeb491168ba275541eee99379bb39d6348b7f5215359cd48da7e07ddb3.jpg)

![efe1f871407177390bbfba87a7041b27ed43cc8ec54bf637d1db77a49dd0a881.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/images/efe1f871407177390bbfba87a7041b27ed43cc8ec54bf637d1db77a49dd0a881.jpg)

### Tables

![0e8f0382f85897a163147aaf402ec328858d7a6a7bd4a6dc284d2224f7c33c96.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/0e8f0382f85897a163147aaf402ec328858d7a6a7bd4a6dc284d2224f7c33c96.jpg)

![17111ed9ff4114f14a458af870bb1899c304398d383bb0c56a14db8632a5aafb.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/17111ed9ff4114f14a458af870bb1899c304398d383bb0c56a14db8632a5aafb.jpg)

![23e8f7be5075d18bed094853fd0a31a8d274f8f43287e87f4421c0aea719c6a2.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/23e8f7be5075d18bed094853fd0a31a8d274f8f43287e87f4421c0aea719c6a2.jpg)

![7122bb1988d7a01ef4585d998ae5e165e1e485b7e20e06ac531afb049abc3f51.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/7122bb1988d7a01ef4585d998ae5e165e1e485b7e20e06ac531afb049abc3f51.jpg)

![80e1fe85f664177ddcda7a260c13a90e8d77506bba882fb6637df0a4c29775c4.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/80e1fe85f664177ddcda7a260c13a90e8d77506bba882fb6637df0a4c29775c4.jpg)

![97f9cf5a0bfc94f2cf41b266aa8fb03fd812ff1c11ff637ba08d3eb4e250c314.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/97f9cf5a0bfc94f2cf41b266aa8fb03fd812ff1c11ff637ba08d3eb4e250c314.jpg)

![a62a6165f29dc3547f7690ffd771465afc1521802f5ed41b1850436e6a665604.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/a62a6165f29dc3547f7690ffd771465afc1521802f5ed41b1850436e6a665604.jpg)

![d36318f4b7acf6fc5ac55270f22da6b7073166709e40318ee1d6866ac6e851e4.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/d36318f4b7acf6fc5ac55270f22da6b7073166709e40318ee1d6866ac6e851e4.jpg)

![e3eda3de917c7bfeb2ebc4e9f04cb7c4c0fabb6d19f0d798442abb1f22096b79.jpg](../icml_results/4998_Co-Evolving%20Latent%20Action%20World%20Models/tables/e3eda3de917c7bfeb2ebc4e9f04cb7c4c0fabb6d19f0d798442abb1f22096b79.jpg)
