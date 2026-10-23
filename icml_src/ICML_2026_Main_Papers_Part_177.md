# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 177 of 197

## 目录 (Table of Contents)

1. [Is Task-Specific Training Necessary for Anomaly Detection?](#paper-5809)
2. [Reasoning over Boundaries: Enhancing Specification Alignment via Test-time Deliberation](#paper-5810)
3. [D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](#paper-5811)
4. [AdaMEM: Test-Time Adaptive Memory for Language Agents](#paper-5812)
5. [DualTimesField: Rethinking Time Series as Continuous-Time Trends and Events](#paper-5813)
6. [VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models](#paper-5814)
7. [You Don’t Protect if You Don’t Expect: Breaking the Key Assumption behind CLIP’s Test-Time Defenses](#paper-5815)
8. [The Latent Guardian: Defending Collaborative Perception via Feature-Level Consistency Verification](#paper-5816)
9. [Quantifying and Optimizing Simplicity via Polynomial Representations](#paper-5817)
10. [In-Context Learning as Rate–Distortion Optimization](#paper-5818)
11. [Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls](#paper-5819)
12. [Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs](#paper-5820)
13. [Trajectory-Stabilized Inference for Diffusion-Based Video Inpainting](#paper-5821)
14. [CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions](#paper-5822)
15. [Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance](#paper-5823)
16. [Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](#paper-5824)
17. [MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification](#paper-5825)
18. [VELR: Efficient Video Reward Feedback via Ensemble Latent Reward Models](#paper-5826)
19. [Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models](#paper-5827)
20. [Rethinking Human Intent-to-CAD: Parametric CAD Model Generation via Cooperative Multi-Task Alignment and Spatial-Aware Reinforcement Learning](#paper-5828)
21. [Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images](#paper-5829)
22. [CLIMB: Taming the LoRA Residency Cliff in Multi-LoRA Serving](#paper-5830)
23. [ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](#paper-5831)
24. [Adversarial Reinforcement Learning for Robust Diffusion Large Language Model Unlearning](#paper-5832)
25. [R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model?](#paper-5833)
26. [UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation](#paper-5834)
27. [On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](#paper-5835)
28. [InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution](#paper-5836)
29. [Regime-Adaptive Bayesian Optimization via Dirichlet Process Mixtures of Gaussian Processes](#paper-5837)
30. [Contrastive Flow Map Matching](#paper-5838)
31. [Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training](#paper-5839)
32. [Adaptive Visual Autoregressive Acceleration via Dual-Linkage Entropy Analysis](#paper-5840)
33. [Multimodal Meta-Verifier with Explicit Structured Recalibration](#paper-5841)

---


<a id="paper-5809"></a>

## Is Task-Specific Training Necessary for Anomaly Detection?

### Images

![5fe8b7dd3b67640c85dd21985283bffcf1d169dbfb22a32f73567e6a63d6e116.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/images/5fe8b7dd3b67640c85dd21985283bffcf1d169dbfb22a32f73567e6a63d6e116.jpg)

![7ba53b3458e218daa5a88f4f58506d6f94a837f63a02ae0f032b4583e4abdb61.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/images/7ba53b3458e218daa5a88f4f58506d6f94a837f63a02ae0f032b4583e4abdb61.jpg)

### Tables

![35dd0d9ef8174049e4cf168d6bf0099609d766062e26f5fc0c0174bc7bc84de1.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/tables/35dd0d9ef8174049e4cf168d6bf0099609d766062e26f5fc0c0174bc7bc84de1.jpg)

![77024e1f3b84818f7826b2c0039bde8cfe86b94feb1a7f2dbbb0c499112227b0.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/tables/77024e1f3b84818f7826b2c0039bde8cfe86b94feb1a7f2dbbb0c499112227b0.jpg)

![b20f6230b26ff034da4cb218d4e14900d2f3af13b10d8609f8b922622fe73fc7.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/tables/b20f6230b26ff034da4cb218d4e14900d2f3af13b10d8609f8b922622fe73fc7.jpg)

![de0fb760b80283fbc8b5707471843d8e8428715474c8de8839942779c457853c.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/tables/de0fb760b80283fbc8b5707471843d8e8428715474c8de8839942779c457853c.jpg)

![e6f0fa5970b62fa85b9071348c93ee9d61eb5c26bff2a7e85ba102ef94bc0aab.jpg](../icml_results/5857_Demystifying%20the%20Optimal%20Fair%20Classifier%20in%20Multi-Class%20Classification/tables/e6f0fa5970b62fa85b9071348c93ee9d61eb5c26bff2a7e85ba102ef94bc0aab.jpg)

## Is Task-Specific Training Necessary for Anomaly Detection?


### Images

![1c51a7fe0d144b408427164d9516c7f5409fd4571de5ce9d962573df0460fd73.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/1c51a7fe0d144b408427164d9516c7f5409fd4571de5ce9d962573df0460fd73.jpg)

![386cc7c52836c60cd5a89a2c1f1a589a8b07a373781d37a348d54016ee5ec716.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/386cc7c52836c60cd5a89a2c1f1a589a8b07a373781d37a348d54016ee5ec716.jpg)

![68aa96e194fc8bc9241fb2574dedeff63b3d647028f392189b1f17d3c3540a84.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/68aa96e194fc8bc9241fb2574dedeff63b3d647028f392189b1f17d3c3540a84.jpg)

![dba38f34a63d177c1de3572cd79ee60758a7136f47a3d8ed6f0223b377a19c6e.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/dba38f34a63d177c1de3572cd79ee60758a7136f47a3d8ed6f0223b377a19c6e.jpg)

![ded90511687dc53b471d782cccebedbb0ccce65b98546851f0b0c13321899355.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/ded90511687dc53b471d782cccebedbb0ccce65b98546851f0b0c13321899355.jpg)

![e9c59b33cdaaaaa9578107e101c6acb1cfa0d7b87aa2e75d2273813f49f14b9a.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/e9c59b33cdaaaaa9578107e101c6acb1cfa0d7b87aa2e75d2273813f49f14b9a.jpg)

![fb845b6a7ef446adb00444449577997693cd3d15c9a58235bad62b1284d0ddf3.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/images/fb845b6a7ef446adb00444449577997693cd3d15c9a58235bad62b1284d0ddf3.jpg)

### Tables

![05dc23340512022bcf2ee6905fd1889dbf2255ae71c56bf79908a7999bda5106.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/05dc23340512022bcf2ee6905fd1889dbf2255ae71c56bf79908a7999bda5106.jpg)

![1abc46d528edb500f81a73015ce9b6e525d5f8f98b109421170b4493a0c34685.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/1abc46d528edb500f81a73015ce9b6e525d5f8f98b109421170b4493a0c34685.jpg)

![22a3f316574aecc362b5d28a4bce864416890a53e47ada9d45f3b71280f3dee3.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/22a3f316574aecc362b5d28a4bce864416890a53e47ada9d45f3b71280f3dee3.jpg)

![28580c20ff4772ae326d395da340bfe14b0f2c6132d2f32519cce1900ef03674.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/28580c20ff4772ae326d395da340bfe14b0f2c6132d2f32519cce1900ef03674.jpg)

![28cc3b321e236f5d478845d951ad999e580041ffcb347612d0eb833fcb7254e8.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/28cc3b321e236f5d478845d951ad999e580041ffcb347612d0eb833fcb7254e8.jpg)

![3ba5fabd741aa7106cad0ea00710e119a3010949c37692358f02bfe32cc76e61.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/3ba5fabd741aa7106cad0ea00710e119a3010949c37692358f02bfe32cc76e61.jpg)

![3d0a18d1b8c50968c2eb78e2f12e2d53b30982d441e434db7bf35c71f12268d2.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/3d0a18d1b8c50968c2eb78e2f12e2d53b30982d441e434db7bf35c71f12268d2.jpg)

![5cd05bd134ea67ca34b8c02d7f00cd0c43e03ffb9177650cd57f941b7e979987.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/5cd05bd134ea67ca34b8c02d7f00cd0c43e03ffb9177650cd57f941b7e979987.jpg)

![8455554cbebf705aa3da5950dbc625eb4be528996c89148165e79b1125859d81.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/8455554cbebf705aa3da5950dbc625eb4be528996c89148165e79b1125859d81.jpg)

![8fecf1d70dd2e65c4dfd75ff11b86c1f1260a648281fbff1c0f04f0c9bc1d217.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/8fecf1d70dd2e65c4dfd75ff11b86c1f1260a648281fbff1c0f04f0c9bc1d217.jpg)

![9c4692a5f5a4a57a34ac43d183f83551d5e6136c4a7a195897adea991dfad761.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/9c4692a5f5a4a57a34ac43d183f83551d5e6136c4a7a195897adea991dfad761.jpg)

![a873625114e4f8957e489ebc545d6388fd3f12b87202a00d738f62467289e3a5.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/a873625114e4f8957e489ebc545d6388fd3f12b87202a00d738f62467289e3a5.jpg)

![a8e0e31ba949edd485ffabbbf536746cd0cbaf93e9f330ac1784c2456fd201f6.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/a8e0e31ba949edd485ffabbbf536746cd0cbaf93e9f330ac1784c2456fd201f6.jpg)

![b04c116aa8a8edf980f4fc4245a0a13f4607872067b9921d1fbf98c83f9dab82.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/b04c116aa8a8edf980f4fc4245a0a13f4607872067b9921d1fbf98c83f9dab82.jpg)

![bc0b09cafc109283ee15d64dfe59b48e081c5db5c6149ae80988468546727316.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/bc0b09cafc109283ee15d64dfe59b48e081c5db5c6149ae80988468546727316.jpg)

![cbc5a5a902324299292b47decf9df1678cddf827e0a963f8a750e076a387401e.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/cbc5a5a902324299292b47decf9df1678cddf827e0a963f8a750e076a387401e.jpg)

![d3106d4fe5c6e4877d8eed08622b445793f873bd58c671018e6cc9cad465ea03.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/d3106d4fe5c6e4877d8eed08622b445793f873bd58c671018e6cc9cad465ea03.jpg)

![fcc6183e4a9a17fc209383d3d5347d78942c2e5cc05b00bd55a0e93a4c143089.jpg](../icml_results/5858_Is%20Task-Specific%20Training%20Necessary%20for%20Anomaly%20Detection_/tables/fcc6183e4a9a17fc209383d3d5347d78942c2e5cc05b00bd55a0e93a4c143089.jpg)

<a id="paper-5810"></a>

## Reasoning over Boundaries: Enhancing Specification Alignment via Test-time Deliberation


### Images

![19ab60ced972de300bb206449223d68f45da137a8248a87a804ea12d50887d42.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/19ab60ced972de300bb206449223d68f45da137a8248a87a804ea12d50887d42.jpg)

![1b6965b118c30ee63156412acd3a080dcf8adc97de4af724128eb88f24403606.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/1b6965b118c30ee63156412acd3a080dcf8adc97de4af724128eb88f24403606.jpg)

![2b0a4c9f2127f5e6071148aff939eccd6940085e5bfb4e3c88e61430895cbac2.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/2b0a4c9f2127f5e6071148aff939eccd6940085e5bfb4e3c88e61430895cbac2.jpg)

![316981961d642102574a85a015e0ea4c5d40e5109afb46bbdb79eba154cb40ae.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/316981961d642102574a85a015e0ea4c5d40e5109afb46bbdb79eba154cb40ae.jpg)

![51685c269b67f877d0581657b330fb5d7b275cbfdc60b6892bc095e4e2e25950.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/51685c269b67f877d0581657b330fb5d7b275cbfdc60b6892bc095e4e2e25950.jpg)

![6b76caae577a9490445db8eba2f2205ae85446306b7f27f50cfa42d3547e3ce1.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/6b76caae577a9490445db8eba2f2205ae85446306b7f27f50cfa42d3547e3ce1.jpg)

![7748f64e994c451620bdac9b5ecd1b5f12067769af69354cd4bb9ca16533c75b.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/7748f64e994c451620bdac9b5ecd1b5f12067769af69354cd4bb9ca16533c75b.jpg)

![89e6f9aa0da86f6ce84a120447761431dddf312a1516c35824d4af7e9c8430e1.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/89e6f9aa0da86f6ce84a120447761431dddf312a1516c35824d4af7e9c8430e1.jpg)

![8c9c83d197321bc6b72e25a05ec25f7a7044e329086c66f62473c21b48b4ffc8.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/8c9c83d197321bc6b72e25a05ec25f7a7044e329086c66f62473c21b48b4ffc8.jpg)

![91cc415e2fadb43f0664360d73b5ba5553dec241ee5a2e8adba26c587dfb2494.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/91cc415e2fadb43f0664360d73b5ba5553dec241ee5a2e8adba26c587dfb2494.jpg)

![a3b489965c5879f2a721ebf4bc84368367a969611352d84c5ab9143478f08eb5.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/a3b489965c5879f2a721ebf4bc84368367a969611352d84c5ab9143478f08eb5.jpg)

![abcc6a5b6c906f196f91e63ddb4c90fd0f385590582ed82c015839f13705e0d5.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/abcc6a5b6c906f196f91e63ddb4c90fd0f385590582ed82c015839f13705e0d5.jpg)

![aeac7fd37bb433cd07f7e4898c426b7ff1fcabcee97e81d825264af88e5589bd.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/aeac7fd37bb433cd07f7e4898c426b7ff1fcabcee97e81d825264af88e5589bd.jpg)

![e932ba6142bfc47921904879b86706ce5bd48a6bb02f8c6d311599b1ff0c9c95.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/images/e932ba6142bfc47921904879b86706ce5bd48a6bb02f8c6d311599b1ff0c9c95.jpg)

### Tables

![032e1fe6223176016303221b726fcfde7ea2ec6ad09b6c4bf8f976169321b32e.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/032e1fe6223176016303221b726fcfde7ea2ec6ad09b6c4bf8f976169321b32e.jpg)

![148ea2b9df9470208b43c01ea740eb804db35a7d241d5780968f3e2efcdf1c4d.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/148ea2b9df9470208b43c01ea740eb804db35a7d241d5780968f3e2efcdf1c4d.jpg)

![2483d792b9cac8954c9c5a5cff2f2c8996a822097c44a494a8ac97307a4d3c26.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/2483d792b9cac8954c9c5a5cff2f2c8996a822097c44a494a8ac97307a4d3c26.jpg)

![2caf3b3e5870cde2536892d4746ea9b8b0f4bc9a2e6bd0f7dffded0f18839d1a.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/2caf3b3e5870cde2536892d4746ea9b8b0f4bc9a2e6bd0f7dffded0f18839d1a.jpg)

![40efe4e69e116ad0db124f88209d53b7fdb76fe5ef431eabea2ad6bdf44b3546.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/40efe4e69e116ad0db124f88209d53b7fdb76fe5ef431eabea2ad6bdf44b3546.jpg)

![7c406764ec6ceb29906d48539c3724bd59e9840d9117fd8b948f97758c1db391.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/7c406764ec6ceb29906d48539c3724bd59e9840d9117fd8b948f97758c1db391.jpg)

![8e8cc4754c8d3429e7cb271181ce54801a1fae9d2340c1e29f0adcb797a4f759.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/8e8cc4754c8d3429e7cb271181ce54801a1fae9d2340c1e29f0adcb797a4f759.jpg)

![8f965ca541cd0cad774e0b44e8b37b26e726622430b70c6ba3723496cfe8e6e8.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/8f965ca541cd0cad774e0b44e8b37b26e726622430b70c6ba3723496cfe8e6e8.jpg)

![94340615583dae11048a436a0186770c6b8fab6a8d4a0f1ea0719a7724764592.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/94340615583dae11048a436a0186770c6b8fab6a8d4a0f1ea0719a7724764592.jpg)

![bb6592d73124546710d4d7aa66e56c90d68b96fe2bf59b4f9ce32de945e5816f.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/bb6592d73124546710d4d7aa66e56c90d68b96fe2bf59b4f9ce32de945e5816f.jpg)

![ca381ff50f11fbfb39d08a4eb97d3078ae39e465a63d029a58d5d95d7d01998a.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/ca381ff50f11fbfb39d08a4eb97d3078ae39e465a63d029a58d5d95d7d01998a.jpg)

![dd92277a78685f29ba639026970e4dea4606a5a85a98bccbe9de7bcb075673b2.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/dd92277a78685f29ba639026970e4dea4606a5a85a98bccbe9de7bcb075673b2.jpg)

![e01324fc11169c37bd148069c222905dac3c35f923decc8704fa2b860b262c49.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/e01324fc11169c37bd148069c222905dac3c35f923decc8704fa2b860b262c49.jpg)

![e473cd7de490e4a1333212e809052b1181e533e2678a822ee0d4312e171d01b3.jpg](../icml_results/5859_Reasoning%20over%20Boundaries_%20Enhancing%20Specification%20Alignment%20via%20Test-time%20Deliberation/tables/e473cd7de490e4a1333212e809052b1181e533e2678a822ee0d4312e171d01b3.jpg)

<a id="paper-5811"></a>

## D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning


### Images

![029f5a287e08dbf8eeed8f779a4e30bbf095985c5eb6b3b67d68c4a08cffc92f.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/images/029f5a287e08dbf8eeed8f779a4e30bbf095985c5eb6b3b67d68c4a08cffc92f.jpg)

![510d86883ba5be7c2555d86c94cf1f0437cb5823dab7a67a53b4b1e3d24c4521.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/images/510d86883ba5be7c2555d86c94cf1f0437cb5823dab7a67a53b4b1e3d24c4521.jpg)

![7b8ca555482f59249f5dfd80fc70d991b45ec6bd0b27d027ee677b6805bf4f89.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/images/7b8ca555482f59249f5dfd80fc70d991b45ec6bd0b27d027ee677b6805bf4f89.jpg)

![f193e8b01804c4f2c9bc03d15d71b524179bf97e8c52928b7c383cc794e96eb0.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/images/f193e8b01804c4f2c9bc03d15d71b524179bf97e8c52928b7c383cc794e96eb0.jpg)

![fdd7517c9f06f845264b121b61d1c7af23fa8a8f1b244bf082b66972d30bc03c.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/images/fdd7517c9f06f845264b121b61d1c7af23fa8a8f1b244bf082b66972d30bc03c.jpg)

### Tables

![07d7cffedcc768f2d478f64aa379139055cf0df9ccce1f6558eb6772d6db504c.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/07d7cffedcc768f2d478f64aa379139055cf0df9ccce1f6558eb6772d6db504c.jpg)

![3f153e9c86ab6b21ba18082da2362efa3f0232c16fc2e8c91a0f7f2e6984e7d8.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/3f153e9c86ab6b21ba18082da2362efa3f0232c16fc2e8c91a0f7f2e6984e7d8.jpg)

![56e0a181b60c32e66608b0f7350a262fb11cf22a2aabe15ac72c0d584c68081f.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/56e0a181b60c32e66608b0f7350a262fb11cf22a2aabe15ac72c0d584c68081f.jpg)

![5b46511c0dfe30b4799334b986b506fe27f820e5ba450a54d0473644dc1e70af.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/5b46511c0dfe30b4799334b986b506fe27f820e5ba450a54d0473644dc1e70af.jpg)

![74cf0c6bb5371a1d7d9ba67eefd26c98592896b85fcec21ef92b05c6459db36f.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/74cf0c6bb5371a1d7d9ba67eefd26c98592896b85fcec21ef92b05c6459db36f.jpg)

![95e283731b6b01549a5cdd441ce629d903773a6b72125bb7ecb4ce3ea1b2418c.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/95e283731b6b01549a5cdd441ce629d903773a6b72125bb7ecb4ce3ea1b2418c.jpg)

![ab883a80f88b8cd8d2ccd899390be40ae496c7e5885d0091bd56df439292d7bd.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/ab883a80f88b8cd8d2ccd899390be40ae496c7e5885d0091bd56df439292d7bd.jpg)

![afdcafb0bfb1885bc8844e3f2620cae22c7ac9999ad1101adab2bcf6a1d02322.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/afdcafb0bfb1885bc8844e3f2620cae22c7ac9999ad1101adab2bcf6a1d02322.jpg)

![c33c793cc0b593777a81abdfaea57b32d2052ecdf96a9e607ffd63265ecaf3ee.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/c33c793cc0b593777a81abdfaea57b32d2052ecdf96a9e607ffd63265ecaf3ee.jpg)

![dab1afd003af95d029ea6ad9c4a86c2aa00c134fa05a07b12498c07cb5dca443.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/dab1afd003af95d029ea6ad9c4a86c2aa00c134fa05a07b12498c07cb5dca443.jpg)

![dd0a86611ed3512b0559c8254193648a02a5c6b4a0a63d3ed04a16523706d284.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/dd0a86611ed3512b0559c8254193648a02a5c6b4a0a63d3ed04a16523706d284.jpg)

![ddd08b1f0b14ff7761dbdd1ce5442918a3def3f189bf69946e7fc26a8b494195.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/ddd08b1f0b14ff7761dbdd1ce5442918a3def3f189bf69946e7fc26a8b494195.jpg)

![e2c2b49908beedc1047ac40691fdc5ed5414b0c1b16406874209f73c45954fef.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/e2c2b49908beedc1047ac40691fdc5ed5414b0c1b16406874209f73c45954fef.jpg)

![ee2dcc56efcc20099d7689417170e7784b4b45364236b2f7ecc78f79899d49f9.jpg](../icml_results/5860_D%24%5E2%24Evo_%20Dual%20Difficulty-Aware%20Self-Evolution%20for%20Data-Efficient%20Reinforcement%20Learning/tables/ee2dcc56efcc20099d7689417170e7784b4b45364236b2f7ecc78f79899d49f9.jpg)

<a id="paper-5812"></a>

## AdaMEM: Test-Time Adaptive Memory for Language Agents


### Images

![08038bd9de0ae48adfa3bb387699db82834661f85abe04bf418ef61f4f01b3d7.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/08038bd9de0ae48adfa3bb387699db82834661f85abe04bf418ef61f4f01b3d7.jpg)

![3b316614e12f1ceade932e0b7c14bf40d4a5538e9458070197e95a280fddcadc.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/3b316614e12f1ceade932e0b7c14bf40d4a5538e9458070197e95a280fddcadc.jpg)

![444f06a9e1937d557ab0aca44130b050f3b3e0e43442b6748b7e99a7220d08a8.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/444f06a9e1937d557ab0aca44130b050f3b3e0e43442b6748b7e99a7220d08a8.jpg)

![667341f45016458e7c7f97c16fda462df17bf5c03355e7e617f801e7ebd41490.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/667341f45016458e7c7f97c16fda462df17bf5c03355e7e617f801e7ebd41490.jpg)

![7b262d21a8604b38d4ef14171a5ad016786f798e4945250825c54ed4fd6904ce.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/7b262d21a8604b38d4ef14171a5ad016786f798e4945250825c54ed4fd6904ce.jpg)

![871c0bb3a982ee910be208be8e34f28d65595711f77840d6992c194bf6feae05.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/871c0bb3a982ee910be208be8e34f28d65595711f77840d6992c194bf6feae05.jpg)

![93139aaf3ac88aed822d8976c5a7288bc59980a357d7fd2f48510c031e8a5be6.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/93139aaf3ac88aed822d8976c5a7288bc59980a357d7fd2f48510c031e8a5be6.jpg)

![a973b8d39207956c059288dd171491185eabfc24736df72a3f33a58086f50dc7.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/images/a973b8d39207956c059288dd171491185eabfc24736df72a3f33a58086f50dc7.jpg)

### Tables

![2b5586d310cc8115dbb073d7f1233d0264a8a4f51daf248a2eee8265da3694f8.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/2b5586d310cc8115dbb073d7f1233d0264a8a4f51daf248a2eee8265da3694f8.jpg)

![34529917f4af8bc258c8d4038785d3718f0dae6693e1ef607864ca1b9323e4b0.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/34529917f4af8bc258c8d4038785d3718f0dae6693e1ef607864ca1b9323e4b0.jpg)

![37b513c488c1753ed13664c14f5cc9179dbdb20bc6c757b1b2cb66545f0919be.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/37b513c488c1753ed13664c14f5cc9179dbdb20bc6c757b1b2cb66545f0919be.jpg)

![46e6a70f0e0d4ee4ce0a4e56d49ca12ad7adc4eccad1d7e1a24c83d7d5dba6fd.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/46e6a70f0e0d4ee4ce0a4e56d49ca12ad7adc4eccad1d7e1a24c83d7d5dba6fd.jpg)

![6610e0a414031ff028f51a59cec68c5786e62d958fa93ceb20da118bb70db0f1.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/6610e0a414031ff028f51a59cec68c5786e62d958fa93ceb20da118bb70db0f1.jpg)

![7da4230a6775bf53046db48548f704cbbcede5508f7225d22fb927724429999e.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/7da4230a6775bf53046db48548f704cbbcede5508f7225d22fb927724429999e.jpg)

![b8fd5e7c9751067108b6429724d3d8306ec88c0d96f522a2d6eae695118cebbc.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/b8fd5e7c9751067108b6429724d3d8306ec88c0d96f522a2d6eae695118cebbc.jpg)

![ea9c8942cf41b65c128562a0dfc154f4febd79d56fcce066c4af97fc9f49cd9b.jpg](../icml_results/5861_AdaMEM_%20Test-Time%20Adaptive%20Memory%20for%20Language%20Agents/tables/ea9c8942cf41b65c128562a0dfc154f4febd79d56fcce066c4af97fc9f49cd9b.jpg)

<a id="paper-5813"></a>

## DualTimesField: Rethinking Time Series as Continuous-Time Trends and Events


### Images

![023489c2a473e987e02777012c5f9b8bb226f190882f98476042a2da50cc41cb.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/023489c2a473e987e02777012c5f9b8bb226f190882f98476042a2da50cc41cb.jpg)

![217964299d0b70bf15c5497debbac7fbe77c53dfd6dbd108efce599be4b5b87c.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/217964299d0b70bf15c5497debbac7fbe77c53dfd6dbd108efce599be4b5b87c.jpg)

![32d74bbf354588e046148c57e55cd9a0e192424031bba8da477bed743757d941.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/32d74bbf354588e046148c57e55cd9a0e192424031bba8da477bed743757d941.jpg)

![4ab671a6e5cc86b51055aecb7b2e9fd29b775ca6f57546cf48dce4bc88bb647a.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/4ab671a6e5cc86b51055aecb7b2e9fd29b775ca6f57546cf48dce4bc88bb647a.jpg)

![4e3e2a4138f6c9f4e6cf8f9cfbaca0387de929463432e621e03b3233d4c5e13b.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/4e3e2a4138f6c9f4e6cf8f9cfbaca0387de929463432e621e03b3233d4c5e13b.jpg)

![62d7e0598e6c9df7d26371ea885a70e63b0673cafcea6b9d7219050f656fe8dd.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/62d7e0598e6c9df7d26371ea885a70e63b0673cafcea6b9d7219050f656fe8dd.jpg)

![69cba863db8587c9a6421fc0e9938202ef2115a27d871772f54b8a548f8a2732.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/69cba863db8587c9a6421fc0e9938202ef2115a27d871772f54b8a548f8a2732.jpg)

![7316991edf11dae87c40d97a85b0a2a735645b3e516aee2a5355f5f732a55406.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/7316991edf11dae87c40d97a85b0a2a735645b3e516aee2a5355f5f732a55406.jpg)

![743556914b30d66a18d2f6487c29f0a925a75d76fa5d21cf730577cb891bb54c.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/743556914b30d66a18d2f6487c29f0a925a75d76fa5d21cf730577cb891bb54c.jpg)

![9bd98b28370e618de35f942867d8b23b84580d0ed31bc3b2c6026066ea8866a2.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/9bd98b28370e618de35f942867d8b23b84580d0ed31bc3b2c6026066ea8866a2.jpg)

![c0cd4ec38478c44e4708b508b09dce9b318e5accc7744799125c05b5e42e767b.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/c0cd4ec38478c44e4708b508b09dce9b318e5accc7744799125c05b5e42e767b.jpg)

![ccc8466c6720a3bc43bb62ba33d98599b7d5413e646115896b05726638d3d514.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/ccc8466c6720a3bc43bb62ba33d98599b7d5413e646115896b05726638d3d514.jpg)

![ffe00953165b441893bd848b759be06d652dbeaa13a46778b5143e122a1a3420.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/images/ffe00953165b441893bd848b759be06d652dbeaa13a46778b5143e122a1a3420.jpg)

### Tables

![7fb95eb76608333cc5df3fb831bb1935da0dadee025181bf8e41a6315e9789fc.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/7fb95eb76608333cc5df3fb831bb1935da0dadee025181bf8e41a6315e9789fc.jpg)

![9c7fe1ab76060c05318743f46e4d4370fc7c7a4ed2257da4d758b1c80b356054.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/9c7fe1ab76060c05318743f46e4d4370fc7c7a4ed2257da4d758b1c80b356054.jpg)

![a05441954bed593e017d80aaf8f750f232fa3172b596b8939bb9651919748961.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/a05441954bed593e017d80aaf8f750f232fa3172b596b8939bb9651919748961.jpg)

![aa1851101b14c805c245861b466643ba9d8de6af3706c5020a247ff729232c09.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/aa1851101b14c805c245861b466643ba9d8de6af3706c5020a247ff729232c09.jpg)

![adaf92380f77545e694ca3bc41a5b23754da0c2f098cddc21d4be368af8992d2.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/adaf92380f77545e694ca3bc41a5b23754da0c2f098cddc21d4be368af8992d2.jpg)

![bbcf1908a77a4bbc9f72eeaecda9fbb44f91723673cc89a4bc04d62c05619b35.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/bbcf1908a77a4bbc9f72eeaecda9fbb44f91723673cc89a4bc04d62c05619b35.jpg)

![e50761a65a0b5d54dc8d74e37450f29b3452804167fc4872937a8c676ffcbc0b.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/e50761a65a0b5d54dc8d74e37450f29b3452804167fc4872937a8c676ffcbc0b.jpg)

![f8e77190a75087a5993aa161a9b0e4b472266b91796d2a7b63294ca2a1e66f84.jpg](../icml_results/5862_DualTimesField_%20Rethinking%20Time%20Series%20as%20Continuous-Time%20Trends%20and%20Events/tables/f8e77190a75087a5993aa161a9b0e4b472266b91796d2a7b63294ca2a1e66f84.jpg)

<a id="paper-5814"></a>

## VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models


### Images

![03469784be04239096f767d7937e88d4a4b003ef989e342e22632f370bee5a99.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/03469784be04239096f767d7937e88d4a4b003ef989e342e22632f370bee5a99.jpg)

![09dcf777e0d8cf4148b9f5a1be165a91c39bdde972c8629446939a4ea99bc376.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/09dcf777e0d8cf4148b9f5a1be165a91c39bdde972c8629446939a4ea99bc376.jpg)

![1132ef6ebfbc80833ccf96ff6e62e6958daf0243d1a7d66d7310e478c1ca2e99.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/1132ef6ebfbc80833ccf96ff6e62e6958daf0243d1a7d66d7310e478c1ca2e99.jpg)

![19a10221479a1945f212e75b8a1dcb96d7534790c77fd3d67653cb4976b8f7c5.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/19a10221479a1945f212e75b8a1dcb96d7534790c77fd3d67653cb4976b8f7c5.jpg)

![235c7c2acefda61cf9ac30f10e39d8847a6537f03b40a41f0ad2cb0ee160f10a.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/235c7c2acefda61cf9ac30f10e39d8847a6537f03b40a41f0ad2cb0ee160f10a.jpg)

![2461658d305336c0377643a61f1ffc85a9a96c18b4a961266c870aa007e29838.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/2461658d305336c0377643a61f1ffc85a9a96c18b4a961266c870aa007e29838.jpg)

![2b962ad6b706911c8cf3c976bbb2905d89f5d9f72f856aeb208bd3113377a5f1.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/2b962ad6b706911c8cf3c976bbb2905d89f5d9f72f856aeb208bd3113377a5f1.jpg)

![2de8ddf0b80fd855d9c9449f7873750a08a4ec208d01f1dc383239e983c3189d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/2de8ddf0b80fd855d9c9449f7873750a08a4ec208d01f1dc383239e983c3189d.jpg)

![397965cac05dd7b029fa31a74bd01353c61c02fa40f2f8e0ae0f7ef882905e1c.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/397965cac05dd7b029fa31a74bd01353c61c02fa40f2f8e0ae0f7ef882905e1c.jpg)

![39f37f9007bec17bbd3d689b12e61d2c1831029291eff00b8b0e12eeb38493a7.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/39f37f9007bec17bbd3d689b12e61d2c1831029291eff00b8b0e12eeb38493a7.jpg)

![3d6f69aadc348dad392a9fa5b899d81ab77f29eae7de3e19bf2ae0b66f1339a6.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/3d6f69aadc348dad392a9fa5b899d81ab77f29eae7de3e19bf2ae0b66f1339a6.jpg)

![45107483555a0bd25058e871bdcf1c748fedbc7aa0bb3da9e9745a454ebcee02.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/45107483555a0bd25058e871bdcf1c748fedbc7aa0bb3da9e9745a454ebcee02.jpg)

![4532d630802bed8f5973ff9f2037dbdc5e6c02fc46ac9b0dbdf94a9b5fcd6141.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/4532d630802bed8f5973ff9f2037dbdc5e6c02fc46ac9b0dbdf94a9b5fcd6141.jpg)

![5c9aa85c7c350333b3479f102f8794c0c906085f5106a112c39943fd714c38e2.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/5c9aa85c7c350333b3479f102f8794c0c906085f5106a112c39943fd714c38e2.jpg)

![68a0a07cfb727eb7dbdd422276ca0bca9ba946e8724ed448a16aaf7ed3d15b5d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/68a0a07cfb727eb7dbdd422276ca0bca9ba946e8724ed448a16aaf7ed3d15b5d.jpg)

![6961b9dfb7f31c74c9ce0e6c40a3a735a7ef7f81f45e3442a543649c8be66e87.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/6961b9dfb7f31c74c9ce0e6c40a3a735a7ef7f81f45e3442a543649c8be66e87.jpg)

![735ed54b23afe12d544eec12ae85cab6381c1c233d87194c51d70743b69434e0.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/735ed54b23afe12d544eec12ae85cab6381c1c233d87194c51d70743b69434e0.jpg)

![91b81245d17d292ef8203325f0df0484594479e961ca9ffb36359c6af80bcfb7.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/91b81245d17d292ef8203325f0df0484594479e961ca9ffb36359c6af80bcfb7.jpg)

![93b52e0f7f8d17e0926421f998cff2aa5eafb2c812756c7e69f4caf94236f0bd.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/93b52e0f7f8d17e0926421f998cff2aa5eafb2c812756c7e69f4caf94236f0bd.jpg)

![982dc8831133a7c6c537a23e4597d513026665ca1166fd5002db5b254562290e.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/982dc8831133a7c6c537a23e4597d513026665ca1166fd5002db5b254562290e.jpg)

![a3fcb1afc03d7b30a6cf2c4b6c2e7729fc0382aa44b39a75a8de5fb21e69d976.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/a3fcb1afc03d7b30a6cf2c4b6c2e7729fc0382aa44b39a75a8de5fb21e69d976.jpg)

![ac5855de35e25cd2dda900395cf2d74484f0be575ee2ce0e44520f5f2493edd0.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/ac5855de35e25cd2dda900395cf2d74484f0be575ee2ce0e44520f5f2493edd0.jpg)

![b573a29b307b8349b31e7302891d0712d111df6340038c14daee21bcf41ddfc9.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/b573a29b307b8349b31e7302891d0712d111df6340038c14daee21bcf41ddfc9.jpg)

![c3cb0cc8b3bed105dc0c4e4d944e24cdbba34f35d5fcd759c0fac62e406f2f2a.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/c3cb0cc8b3bed105dc0c4e4d944e24cdbba34f35d5fcd759c0fac62e406f2f2a.jpg)

![c5d64ac6879d411121de04b82701e96d7b6ea09ef97584adcd348b9073174ed5.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/c5d64ac6879d411121de04b82701e96d7b6ea09ef97584adcd348b9073174ed5.jpg)

![c9bcf829893ecbf6e32e98a397c5dfbfbe97513a8d42e9c21882708f1c49b08e.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/c9bcf829893ecbf6e32e98a397c5dfbfbe97513a8d42e9c21882708f1c49b08e.jpg)

![cd4b900854a4fd669ed68b800d0577d42875c0d26a22bf4f2c5410f9ef04d178.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/cd4b900854a4fd669ed68b800d0577d42875c0d26a22bf4f2c5410f9ef04d178.jpg)

![dc30e9bb80fc14612a50190bb8ce01e6fa287a9a7059716fa8ada294c36654b2.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/dc30e9bb80fc14612a50190bb8ce01e6fa287a9a7059716fa8ada294c36654b2.jpg)

![f773775178747a04efb69d2ae2b15f08263859bd2ce6ffb591073dc83f2c65a7.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/images/f773775178747a04efb69d2ae2b15f08263859bd2ce6ffb591073dc83f2c65a7.jpg)

### Tables

![08350f5162c0c1aab72656045855088dcee6db252d83bae83f6c8f105df7208b.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/08350f5162c0c1aab72656045855088dcee6db252d83bae83f6c8f105df7208b.jpg)

![09e6ccd9ec2a57dad192cf8d223b7c50b6079b59e31c43971873d72ea0287f8f.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/09e6ccd9ec2a57dad192cf8d223b7c50b6079b59e31c43971873d72ea0287f8f.jpg)

![0fed081616f34405f79ed097364f123f405fe1800c18ac3336c3a77fa9f0fb79.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/0fed081616f34405f79ed097364f123f405fe1800c18ac3336c3a77fa9f0fb79.jpg)

![1fc6e12e052241840ee026372aeaaf2abfe98ff61abb94c3af7565a003f24865.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/1fc6e12e052241840ee026372aeaaf2abfe98ff61abb94c3af7565a003f24865.jpg)

![22d1df9d6d07d99cd794e2be27a36a6a1bb6108bae12518897e161ef9e669d4d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/22d1df9d6d07d99cd794e2be27a36a6a1bb6108bae12518897e161ef9e669d4d.jpg)

![265e6aee195e76682c0336c775b270df537469f9c4502659f7a7eca19c903bf4.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/265e6aee195e76682c0336c775b270df537469f9c4502659f7a7eca19c903bf4.jpg)

![2f7faecfe2c6a848307560030785443c7b2f0b54e78c974c6a7cac7ee79d77d0.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/2f7faecfe2c6a848307560030785443c7b2f0b54e78c974c6a7cac7ee79d77d0.jpg)

![3755f06c1fea3248fda9e075f52a3d317af3cc79a0908f708e30b93e4c9c697e.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/3755f06c1fea3248fda9e075f52a3d317af3cc79a0908f708e30b93e4c9c697e.jpg)

![4a666dc9dbcbc72a0f8e4245a318e3ae8c27510a40c8f3ac9f33e132f120f28d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/4a666dc9dbcbc72a0f8e4245a318e3ae8c27510a40c8f3ac9f33e132f120f28d.jpg)

![607ce196127044f04f890c8b8bdce9769fc1714457f88fac77667a617e3031d5.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/607ce196127044f04f890c8b8bdce9769fc1714457f88fac77667a617e3031d5.jpg)

![674e58aa752f9392ba26970249193903130826f8f57df93d0713b8d611d0d4bc.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/674e58aa752f9392ba26970249193903130826f8f57df93d0713b8d611d0d4bc.jpg)

![6bf991839270e028821aa7234e843bca4aff45f77744c1bcacb802abaf4c7106.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/6bf991839270e028821aa7234e843bca4aff45f77744c1bcacb802abaf4c7106.jpg)

![7038a4c7595b3342fc6d163eeb9632038c1632822e695a9ace3f02622e807752.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/7038a4c7595b3342fc6d163eeb9632038c1632822e695a9ace3f02622e807752.jpg)

![75951e4c03341073b46a7d2b97b0490d777de3f694b87c21315f80afc60b2f55.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/75951e4c03341073b46a7d2b97b0490d777de3f694b87c21315f80afc60b2f55.jpg)

![76bc145df0100053eb6a4d2c0c46eea7b3503465182a243a2625c302bd359cdc.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/76bc145df0100053eb6a4d2c0c46eea7b3503465182a243a2625c302bd359cdc.jpg)

![77018223a0b41ec6fecc160b5fbbb1a0c4d67baf447785f82ba5c35f00151c3d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/77018223a0b41ec6fecc160b5fbbb1a0c4d67baf447785f82ba5c35f00151c3d.jpg)

![78c159cf757629851c71a252ca562ec27aeb28d1afb42acd98037dec086de54d.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/78c159cf757629851c71a252ca562ec27aeb28d1afb42acd98037dec086de54d.jpg)

![8c2b95591d629f0d2f10b666da10170cceceaa6591047e4c4d3a911522b64d32.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/8c2b95591d629f0d2f10b666da10170cceceaa6591047e4c4d3a911522b64d32.jpg)

![aab666b046e3720037ae42cc76659c4025aeb0ca278e60b44b769745722d7b2a.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/aab666b046e3720037ae42cc76659c4025aeb0ca278e60b44b769745722d7b2a.jpg)

![bf66eac5a4bc1c891c4709ed37f23c14eed575e82656debf8df44304e534bfe8.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/bf66eac5a4bc1c891c4709ed37f23c14eed575e82656debf8df44304e534bfe8.jpg)

![c2a2d8ea438dff9266b5fc29842f494b20e4977503756b06c2a2e3067a168c98.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/c2a2d8ea438dff9266b5fc29842f494b20e4977503756b06c2a2e3067a168c98.jpg)

![da4830ea5af3c3bd902cbd5a32801189dbeb589ee79b04f340d4ef5c24756d93.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/da4830ea5af3c3bd902cbd5a32801189dbeb589ee79b04f340d4ef5c24756d93.jpg)

![fc219ca92a958225879613867d312502c00c46f64248409f9179986d1e068ed4.jpg](../icml_results/5863_VLA-Arena_%20An%20Open-Source%20Framework%20for%20Benchmarking%20Vision-Language-Action%20Models/tables/fc219ca92a958225879613867d312502c00c46f64248409f9179986d1e068ed4.jpg)

<a id="paper-5815"></a>

## You Don’t Protect if You Don’t Expect: Breaking the Key Assumption behind CLIP’s Test-Time Defenses


### Images

![36a8ef5b4ac0909bd327a73cd1d6f34691f925524a3a78e6fd2d93328b7bf68b.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/images/36a8ef5b4ac0909bd327a73cd1d6f34691f925524a3a78e6fd2d93328b7bf68b.jpg)

![9e397a08bab0d33920081cf3076042c9d5e600b998e894286123324051433d4b.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/images/9e397a08bab0d33920081cf3076042c9d5e600b998e894286123324051433d4b.jpg)

![ba5982919d05309199884d2160dd5dab50ae68725d4a7a7ae1f56ee82149baf8.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/images/ba5982919d05309199884d2160dd5dab50ae68725d4a7a7ae1f56ee82149baf8.jpg)

### Tables

![1ec989b453e219ddf3364054647eec0686397c255abf29ed99b4886fb6f2f119.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/1ec989b453e219ddf3364054647eec0686397c255abf29ed99b4886fb6f2f119.jpg)

![4cba9bd46e25147dd526d98d8035ab8d8f50e2a4571a74f9f8d5658ca8ab4952.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/4cba9bd46e25147dd526d98d8035ab8d8f50e2a4571a74f9f8d5658ca8ab4952.jpg)

![9c507a5116e1d5e6966d75a795ebe6370a15c5dab828a4501af44bf21028f204.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/9c507a5116e1d5e6966d75a795ebe6370a15c5dab828a4501af44bf21028f204.jpg)

![a749eea0179cca3b7c9801db31d57e6342ac9cfcd06c169d03e7652095367e10.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/a749eea0179cca3b7c9801db31d57e6342ac9cfcd06c169d03e7652095367e10.jpg)

![b1c13b532d5e77d3e4325a82ea19b0d26b5345fc42c2053715828c02661f6a62.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/b1c13b532d5e77d3e4325a82ea19b0d26b5345fc42c2053715828c02661f6a62.jpg)

![b6c5586819ffc88bb30f3931637fc2990aad902ac6ec5dbde3460837b0365ad1.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/b6c5586819ffc88bb30f3931637fc2990aad902ac6ec5dbde3460837b0365ad1.jpg)

![bd6ae3c11212d587ae8b366b00a1c917bc55c4c83a34908dd1bce6757ccc210e.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/bd6ae3c11212d587ae8b366b00a1c917bc55c4c83a34908dd1bce6757ccc210e.jpg)

![e06d5e9c2094ec78db8128a9ce6bb6b1d1c6723172233142b3d7e2fa2ca03e6a.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/e06d5e9c2094ec78db8128a9ce6bb6b1d1c6723172233142b3d7e2fa2ca03e6a.jpg)

![e9919a115f3819543ecf00be1e711fe768d1ec98f0ad44ec5ccac7427395cb07.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/e9919a115f3819543ecf00be1e711fe768d1ec98f0ad44ec5ccac7427395cb07.jpg)

![f06935818727c618f6c93d122c8be6a186fc46e4816b348d0ce1699decd7bdb9.jpg](../icml_results/5864_You%20Don%E2%80%99t%20Protect%20if%20You%20Don%E2%80%99t%20Expect_%20Breaking%20the%20Key%20Assumption%20behind%20CLIP%E2%80%99s%20Test-Time%20Defenses/tables/f06935818727c618f6c93d122c8be6a186fc46e4816b348d0ce1699decd7bdb9.jpg)

<a id="paper-5816"></a>

## The Latent Guardian: Defending Collaborative Perception via Feature-Level Consistency Verification


### Images

![06e52a39a2ab71e3000f32bdc225b3c1bf16dc9a1e367a1a23cca3129fa90209.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/06e52a39a2ab71e3000f32bdc225b3c1bf16dc9a1e367a1a23cca3129fa90209.jpg)

![309df85a880d038b0cb36984748b8c5cddaebbbf83471f979a558a83f1735b30.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/309df85a880d038b0cb36984748b8c5cddaebbbf83471f979a558a83f1735b30.jpg)

![3f3358c8a3bc77267c28cd06689b408fcf55aa59d92021017b6b47257d1a6a5c.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/3f3358c8a3bc77267c28cd06689b408fcf55aa59d92021017b6b47257d1a6a5c.jpg)

![5f544fd6e0b1f919e8fdcc019732df0f9c834e880bf991829aa23952ed862a7e.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/5f544fd6e0b1f919e8fdcc019732df0f9c834e880bf991829aa23952ed862a7e.jpg)

![6dca39fce0c2d77c9e66e0426eae319566ee8a155cfce63e73f302222d2abc1d.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/6dca39fce0c2d77c9e66e0426eae319566ee8a155cfce63e73f302222d2abc1d.jpg)

![75134c74ceee5776ead85bab8d114704ce7752607b681418cacd306d107275fc.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/75134c74ceee5776ead85bab8d114704ce7752607b681418cacd306d107275fc.jpg)

![7a4600e1d48cf4ab9941ff4477196da79436de99f2122aea5c64f1e3e246c106.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/7a4600e1d48cf4ab9941ff4477196da79436de99f2122aea5c64f1e3e246c106.jpg)

![b3921f6707e4f5eae83da8c3ccbe6e65cf55475886bb2d9fa36a6109e9d88157.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/b3921f6707e4f5eae83da8c3ccbe6e65cf55475886bb2d9fa36a6109e9d88157.jpg)

![cb8db2b423558430f8892b5343037eafda8f1e0ce902f43879497dda16495050.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/cb8db2b423558430f8892b5343037eafda8f1e0ce902f43879497dda16495050.jpg)

![e168b71a9562031317eb9773133f60e7472f1fffe539d0d5a412d0a2e9f00a5c.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/e168b71a9562031317eb9773133f60e7472f1fffe539d0d5a412d0a2e9f00a5c.jpg)

![e7361cf98c8491ec7ee72a57d863b970265bf15424955e25c8bcb92eed3b37af.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/images/e7361cf98c8491ec7ee72a57d863b970265bf15424955e25c8bcb92eed3b37af.jpg)

### Tables

![27cdb40c738bf362ebe29b586adf77f18893ddc048f1c23ee79def361e637e2c.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/27cdb40c738bf362ebe29b586adf77f18893ddc048f1c23ee79def361e637e2c.jpg)

![2a5f0e73662430bb55dca724a260a125f27ee54ddb9ba79767ef2bc69e59dfda.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/2a5f0e73662430bb55dca724a260a125f27ee54ddb9ba79767ef2bc69e59dfda.jpg)

![30c95de19ef67623ce0354c509f46c2918c5bdc7168b0023a2b1d93f5d5069cb.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/30c95de19ef67623ce0354c509f46c2918c5bdc7168b0023a2b1d93f5d5069cb.jpg)

![3b5f8054862d06a993269b1d17d0072659c134caa36115229d74cbc32c8a5152.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/3b5f8054862d06a993269b1d17d0072659c134caa36115229d74cbc32c8a5152.jpg)

![412658a687f358c3486f8f0c4440f2af58b070691938fd283dbde6df63a0051e.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/412658a687f358c3486f8f0c4440f2af58b070691938fd283dbde6df63a0051e.jpg)

![485e0aa3a8ae55b5de6f89c2c8a5c20a419dbd0c08051a8e0e8f3a75780f6bd0.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/485e0aa3a8ae55b5de6f89c2c8a5c20a419dbd0c08051a8e0e8f3a75780f6bd0.jpg)

![5675c2b706e921a446d16212546c7299af6d60d1137b3cfc9aa9581f7246f22b.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/5675c2b706e921a446d16212546c7299af6d60d1137b3cfc9aa9581f7246f22b.jpg)

![57a605b60ac8a7b05f392f87e94dd542a6bb2487ced8e0b56220cfa4e7ccbc61.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/57a605b60ac8a7b05f392f87e94dd542a6bb2487ced8e0b56220cfa4e7ccbc61.jpg)

![5bad3ed4504828934336b1acd0c0a117fc9c9b64cbbe3985a74b1600b115aa25.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/5bad3ed4504828934336b1acd0c0a117fc9c9b64cbbe3985a74b1600b115aa25.jpg)

![68de048be797a28b294f9f63814c5833102b3066e8fc6cfa24b90a68621b1dd1.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/68de048be797a28b294f9f63814c5833102b3066e8fc6cfa24b90a68621b1dd1.jpg)

![698486ecb6ffd00d7500c0986a370dd4bdc2fe1abd0eaeab1bfb51db02e66151.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/698486ecb6ffd00d7500c0986a370dd4bdc2fe1abd0eaeab1bfb51db02e66151.jpg)

![754cb581dda011f97ca4ed1594368bf411c0ec54e16dfa9d6627b04e968338f1.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/754cb581dda011f97ca4ed1594368bf411c0ec54e16dfa9d6627b04e968338f1.jpg)

![8f54be882399247594c8c10be086e934874dc28793e3caedacb7fffa7b678cc2.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/8f54be882399247594c8c10be086e934874dc28793e3caedacb7fffa7b678cc2.jpg)

![97d1d26383504e13fafb25dc4415e0f206df4cab3ae03fce28ab8acfa1971955.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/97d1d26383504e13fafb25dc4415e0f206df4cab3ae03fce28ab8acfa1971955.jpg)

![bf8bf277064480186fd9074f2917681db3ce80779dd77cf66bbebac8e148d728.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/bf8bf277064480186fd9074f2917681db3ce80779dd77cf66bbebac8e148d728.jpg)

![d997bf29b123f6a981ec55330b9d60aa2bd49bd42e122d6bc234736520c61560.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/d997bf29b123f6a981ec55330b9d60aa2bd49bd42e122d6bc234736520c61560.jpg)

![fbee01b5ae3ed4dc97034ea561c7a06726cd519753227ea539b4da33a8021a0f.jpg](../icml_results/5865_The%20Latent%20Guardian_%20Defending%20Collaborative%20Perception%20via%20Feature-Level%20Consistency%20Verification/tables/fbee01b5ae3ed4dc97034ea561c7a06726cd519753227ea539b4da33a8021a0f.jpg)

<a id="paper-5817"></a>

## Quantifying and Optimizing Simplicity via Polynomial Representations


### Images

![0618716687a3eb3745ea5088a3ca1707a8e52606919a7583335ec0c3282282bb.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/0618716687a3eb3745ea5088a3ca1707a8e52606919a7583335ec0c3282282bb.jpg)

![23b9ef8b4cfa2264de7c45b421f2d2e4ba52e9ba7c9ce4df07731402b6cafc1f.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/23b9ef8b4cfa2264de7c45b421f2d2e4ba52e9ba7c9ce4df07731402b6cafc1f.jpg)

![2c7e71837ad1e16155b5a399ac6ced168dcc3b6a977bc83cf5d0585a6315895c.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/2c7e71837ad1e16155b5a399ac6ced168dcc3b6a977bc83cf5d0585a6315895c.jpg)

![3775017deaca2ecd5b746cfae6e4fe89805e23937f116613a094dc738d3bc668.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/3775017deaca2ecd5b746cfae6e4fe89805e23937f116613a094dc738d3bc668.jpg)

![7a2dbba49be994389e7e94a750d82a2c99f601776a37a4d9ecb146c0d49d5e3d.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/7a2dbba49be994389e7e94a750d82a2c99f601776a37a4d9ecb146c0d49d5e3d.jpg)

![91e0790bafe9bd8d7fa723b67d5381b8e4d20caf791cbc6fc88c3cff96ae33bd.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/91e0790bafe9bd8d7fa723b67d5381b8e4d20caf791cbc6fc88c3cff96ae33bd.jpg)

![a1a6c14f85a537df376738bd879ad5cb2f3ec7682577b8ed106c22794335df09.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/a1a6c14f85a537df376738bd879ad5cb2f3ec7682577b8ed106c22794335df09.jpg)

![d1c9e99219a2067ed12929daea4543f9cf3558b693f3c9be80a2269d40f8b809.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/d1c9e99219a2067ed12929daea4543f9cf3558b693f3c9be80a2269d40f8b809.jpg)

![ef43c93b97b8562db65ff9a50a3f9f7bb0101673f7505769a9df68a3037475ad.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/images/ef43c93b97b8562db65ff9a50a3f9f7bb0101673f7505769a9df68a3037475ad.jpg)

### Tables

![3c69cb913cee717cd55336889956b5603c09d70e8c2e6aff7755190dd9eb3a44.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/3c69cb913cee717cd55336889956b5603c09d70e8c2e6aff7755190dd9eb3a44.jpg)

![46c6b246f56e48246d1185095959d69d872e633f5437fcc69eea9ed208d46c70.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/46c6b246f56e48246d1185095959d69d872e633f5437fcc69eea9ed208d46c70.jpg)

![614443806b7358c5d30ab8f7eccbc0c6ac574324101510451145c83bce667728.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/614443806b7358c5d30ab8f7eccbc0c6ac574324101510451145c83bce667728.jpg)

![63e94426c4271538f414a486e6c224f84e856caa15bcb0d3534b8586bb4d5c9d.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/63e94426c4271538f414a486e6c224f84e856caa15bcb0d3534b8586bb4d5c9d.jpg)

![71e2a25b761140605d362f17cf93b2fb477b9740635439313a0c5c656b7abdbd.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/71e2a25b761140605d362f17cf93b2fb477b9740635439313a0c5c656b7abdbd.jpg)

![aaa0515d138a1d77aa5165e662da63415b7eb9ce2487face2c7c682ce2c82b8d.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/aaa0515d138a1d77aa5165e662da63415b7eb9ce2487face2c7c682ce2c82b8d.jpg)

![bad400bcc3289c04e06d7f2c3f0377ea89b149a8a5803f4029ce043458583fc3.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/bad400bcc3289c04e06d7f2c3f0377ea89b149a8a5803f4029ce043458583fc3.jpg)

![d44162804edd67e6b06ed84bdbd1fdc847d95a6727b7664a872c4e1c150beef5.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/d44162804edd67e6b06ed84bdbd1fdc847d95a6727b7664a872c4e1c150beef5.jpg)

![e07b457c13c0c85f5ed5d2d92b4dca0cd13c06509959830ca363b3c9ffccfc0b.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/e07b457c13c0c85f5ed5d2d92b4dca0cd13c06509959830ca363b3c9ffccfc0b.jpg)

![e3f48e07ce2731f8768c4d13787e05ce8c6e27f0b8d2da37b8cc79305fb3b5a0.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/e3f48e07ce2731f8768c4d13787e05ce8c6e27f0b8d2da37b8cc79305fb3b5a0.jpg)

![ed7424a54dd65fe849615913450699104d89364fbbc85f2c2d7d429b757d71ac.jpg](../icml_results/5866_Quantifying%20and%20Optimizing%20Simplicity%20via%20Polynomial%20Representations/tables/ed7424a54dd65fe849615913450699104d89364fbbc85f2c2d7d429b757d71ac.jpg)

<a id="paper-5818"></a>

## In-Context Learning as Rate–Distortion Optimization


### Images

![05acf0bd08abf7432a846856922b814c0e54cd9454b8ef10d0b5a6f84aad1fa3.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/05acf0bd08abf7432a846856922b814c0e54cd9454b8ef10d0b5a6f84aad1fa3.jpg)

![3952987568a375a15604d66a0e72d1155df438c097a1dbc71cdaff4291e25a2c.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/3952987568a375a15604d66a0e72d1155df438c097a1dbc71cdaff4291e25a2c.jpg)

![965697972f619310187407a0af8a64dc145c94afdb44da04b613b3fe77101747.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/965697972f619310187407a0af8a64dc145c94afdb44da04b613b3fe77101747.jpg)

![9751d1a0b95a39795ddc05dc8dc74f8915e33e2e53210cb6170cabda82d7141d.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/9751d1a0b95a39795ddc05dc8dc74f8915e33e2e53210cb6170cabda82d7141d.jpg)

![d72afc5c2185c71457720b7a9bba319a7b49530ce970d2904b116429179503e2.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/d72afc5c2185c71457720b7a9bba319a7b49530ce970d2904b116429179503e2.jpg)

![ee40308444648c5c34ca4b1aac2724d546800f5cbbf22581e328f07b81dfb93d.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/images/ee40308444648c5c34ca4b1aac2724d546800f5cbbf22581e328f07b81dfb93d.jpg)

### Tables

![4ae050632fc72e6e887920b730e0c4872876e54ca510b14f6399b63757321e95.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/tables/4ae050632fc72e6e887920b730e0c4872876e54ca510b14f6399b63757321e95.jpg)

![663d75a1a961dada09d846e915fe9787c03b319a147a3eba1cb5473fbfc38492.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/tables/663d75a1a961dada09d846e915fe9787c03b319a147a3eba1cb5473fbfc38492.jpg)

![bae34c59f244f7302ed10544c103c523087742cd84f6f3847d5904311f6c1cec.jpg](../icml_results/5867_In-Context%20Learning%20as%20Rate%E2%80%93Distortion%20Optimization/tables/bae34c59f244f7302ed10544c103c523087742cd84f6f3847d5904311f6c1cec.jpg)

<a id="paper-5819"></a>

## Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls


### Images

![2e787611cffdb821d2bdfc5975e6d6e0695e2ec0567b899d1735922c0686903e.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/images/2e787611cffdb821d2bdfc5975e6d6e0695e2ec0567b899d1735922c0686903e.jpg)

![32dc45709780d52c6a8df42937984a5ad24a5f462b96cf6f6e254760eb7cbda7.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/images/32dc45709780d52c6a8df42937984a5ad24a5f462b96cf6f6e254760eb7cbda7.jpg)

![598cb32c15dbfd9bf743169e2993709667c82d219e6570adc86827410e83869a.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/images/598cb32c15dbfd9bf743169e2993709667c82d219e6570adc86827410e83869a.jpg)

![94ce36363517116a4392c3f50050f2180ccfe3cc84cddc6ebbc08bc5757bbed7.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/images/94ce36363517116a4392c3f50050f2180ccfe3cc84cddc6ebbc08bc5757bbed7.jpg)

![d2ab655876513ae628166b56d3ac0b88dad21475de443d4cf02053b61a501b0d.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/images/d2ab655876513ae628166b56d3ac0b88dad21475de443d4cf02053b61a501b0d.jpg)

### Tables

![1fe10c3debd6808d71a263d884120c492e5b8524225f9dcc030a460a206f620e.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/1fe10c3debd6808d71a263d884120c492e5b8524225f9dcc030a460a206f620e.jpg)

![4871628b08f94a36f4aa296e21087425cc5e65437edba7845fa2e0669faa9288.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/4871628b08f94a36f4aa296e21087425cc5e65437edba7845fa2e0669faa9288.jpg)

![4e8eef4012ed065748218f448ac20978d00fd9dea81ad8b57130b201d452d19d.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/4e8eef4012ed065748218f448ac20978d00fd9dea81ad8b57130b201d452d19d.jpg)

![7082f27f2e0a508ef9f8576e963d5fc0fa2b192e6d1390814faf52e548b877ea.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/7082f27f2e0a508ef9f8576e963d5fc0fa2b192e6d1390814faf52e548b877ea.jpg)

![7134303c5d06da590592688f6429ec7c90ce1889dff246bb4415765a3119d78c.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/7134303c5d06da590592688f6429ec7c90ce1889dff246bb4415765a3119d78c.jpg)

![7c1e6fdcc75c78266b0837a5c9b4b4faaaabbd5bd31d2fc2ce7f5822f8c7fa10.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/7c1e6fdcc75c78266b0837a5c9b4b4faaaabbd5bd31d2fc2ce7f5822f8c7fa10.jpg)

![c73709069001a4c0978bb4093f5ea89e64e07bb3824b40c0c47e78d278e37c44.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/c73709069001a4c0978bb4093f5ea89e64e07bb3824b40c0c47e78d278e37c44.jpg)

![c9b139a7d2cb51a1d2bd1413175a3dc400323289774cd85aa503301688c26e33.jpg](../icml_results/5868_Gecko_%20A%20Simulation%20Environment%20with%20Stateful%20Feedback%20for%20Refining%20Agent%20Tool%20Calls/tables/c9b139a7d2cb51a1d2bd1413175a3dc400323289774cd85aa503301688c26e33.jpg)

<a id="paper-5820"></a>

## Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs


### Images

![283c0fd71e50412fb6cb8e274896f8152c46cd204088a587b2ee7d40851fd21f.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/283c0fd71e50412fb6cb8e274896f8152c46cd204088a587b2ee7d40851fd21f.jpg)

![5a69f152c702a22d712349e38d1913056974137b4f794fecbcd60e41c56fecb2.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/5a69f152c702a22d712349e38d1913056974137b4f794fecbcd60e41c56fecb2.jpg)

![7a3d79af908714b033f020a751af15aa0e2dc29ef718b2ff54430964963852a7.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/7a3d79af908714b033f020a751af15aa0e2dc29ef718b2ff54430964963852a7.jpg)

![c351ef942a5548d40d70902d87e6fd9ffde451c14d4d21053645494bede8aab2.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/c351ef942a5548d40d70902d87e6fd9ffde451c14d4d21053645494bede8aab2.jpg)

![dc3757f97e2dc1b43efd61a3ff562cd29eea6f5da43423ae512d037932fcba61.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/dc3757f97e2dc1b43efd61a3ff562cd29eea6f5da43423ae512d037932fcba61.jpg)

![dfcbe85d345512ad2f42710b893a1ec6247ceae538093cdd9b03b008ea164ec5.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/dfcbe85d345512ad2f42710b893a1ec6247ceae538093cdd9b03b008ea164ec5.jpg)

![e20e796c7ce79da049e4257a0217c00a256671047a94e405bd6c753695fbfc7a.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/e20e796c7ce79da049e4257a0217c00a256671047a94e405bd6c753695fbfc7a.jpg)

![fa1323914fd1f0780d2ac833ca7973113dd237dc512d015ab86e9ad73b027219.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/images/fa1323914fd1f0780d2ac833ca7973113dd237dc512d015ab86e9ad73b027219.jpg)

### Tables

![106edbc566980a523e25c6a6712c0356539911ebb362fe86e368b4f136b70e01.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/106edbc566980a523e25c6a6712c0356539911ebb362fe86e368b4f136b70e01.jpg)

![19862f4af64255c91dbdb9747c511d6760f725b3a7cbfe50688ea73bc0825d01.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/19862f4af64255c91dbdb9747c511d6760f725b3a7cbfe50688ea73bc0825d01.jpg)

![2ed3d2e01d5bd54abf7f39c975a98ad043f7284b0eb71058f42bf251381f623f.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/2ed3d2e01d5bd54abf7f39c975a98ad043f7284b0eb71058f42bf251381f623f.jpg)

![389bd6c4cb91c4c5768b1a57cb9aada56313b8517b35499789570b64499a7dc4.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/389bd6c4cb91c4c5768b1a57cb9aada56313b8517b35499789570b64499a7dc4.jpg)

![5ecf0835c708d48cb30a541867f5b9b5c7e42620120ed26a8ed7f758ab7958cd.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/5ecf0835c708d48cb30a541867f5b9b5c7e42620120ed26a8ed7f758ab7958cd.jpg)

![6d2402f136993e8ce189ed1f5ba66adbb1ab5e0c67ebdfc498257a2c969ea149.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/6d2402f136993e8ce189ed1f5ba66adbb1ab5e0c67ebdfc498257a2c969ea149.jpg)

![7fbee147ee6b125bbb9a15b9cf40d0603ffb6080e99ac87d9d4784a8e61d6cab.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/7fbee147ee6b125bbb9a15b9cf40d0603ffb6080e99ac87d9d4784a8e61d6cab.jpg)

![8aa48b1d110e2e3f280d9fcc4ae26f9cbf93d0a001b25488d89a66d918247780.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/8aa48b1d110e2e3f280d9fcc4ae26f9cbf93d0a001b25488d89a66d918247780.jpg)

![9124eb5218484088ed5cdf995833bf1ed90a388a9491db77b8a8f6f80d1b079b.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/9124eb5218484088ed5cdf995833bf1ed90a388a9491db77b8a8f6f80d1b079b.jpg)

![958bc00c06febda6e34bc83b85680dcb96f269469a68bd0db15a0f6452abd9c2.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/958bc00c06febda6e34bc83b85680dcb96f269469a68bd0db15a0f6452abd9c2.jpg)

![9c7c93f09006f4b3071ad62d6c695455d6dbc654055dce3c2c7eb9f7958fdeef.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/9c7c93f09006f4b3071ad62d6c695455d6dbc654055dce3c2c7eb9f7958fdeef.jpg)

![a3aed4431e9a1c79fc85faede1fb66528d52fe4b6f19c60521985e0da79e341b.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/a3aed4431e9a1c79fc85faede1fb66528d52fe4b6f19c60521985e0da79e341b.jpg)

![b0663f9053dbe9efdf6bb1284c7f80fc1651bf3b075a7ac2daa9002a38a6554c.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/b0663f9053dbe9efdf6bb1284c7f80fc1651bf3b075a7ac2daa9002a38a6554c.jpg)

![b7b69806666667917cbdf541e50cb89fed50da2e71cd86468e0990bdb284e322.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/b7b69806666667917cbdf541e50cb89fed50da2e71cd86468e0990bdb284e322.jpg)

![b8cf87dee927e214d00048c12735e470df7f99382bdcf8fe320b4942e3b56378.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/b8cf87dee927e214d00048c12735e470df7f99382bdcf8fe320b4942e3b56378.jpg)

![b984ea96f79b05661dc451261a7f769777fa63ba0f296334ca889ecbf479dd96.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/b984ea96f79b05661dc451261a7f769777fa63ba0f296334ca889ecbf479dd96.jpg)

![d1f7f37f4213a0cf54e86e1cbe26cbba9170c6a27c361d94bbcc4bba6f5b15f5.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/d1f7f37f4213a0cf54e86e1cbe26cbba9170c6a27c361d94bbcc4bba6f5b15f5.jpg)

![e0f1a40c53420f0ef845865883a5201394d5b7d955f61bfe78025858b05ea7b8.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/e0f1a40c53420f0ef845865883a5201394d5b7d955f61bfe78025858b05ea7b8.jpg)

![e2241f135f6a643940123f1b931ba831baca1f139b59fa9131e102dcedda60e5.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/e2241f135f6a643940123f1b931ba831baca1f139b59fa9131e102dcedda60e5.jpg)

![e5a052121eb49488e569b5e5d715235abdf116f807c1e76a22e770d121d392bf.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/e5a052121eb49488e569b5e5d715235abdf116f807c1e76a22e770d121d392bf.jpg)

![fb5d19975d5f3c2c6aab8ed24fa93a465fa30fcd4c7abff801b2a79ff5734630.jpg](../icml_results/5869_Neuromem_%20A%20Granular%20Decomposition%20of%20the%20Streaming%20Lifecycle%20in%20External%20Memory%20for%20LLMs/tables/fb5d19975d5f3c2c6aab8ed24fa93a465fa30fcd4c7abff801b2a79ff5734630.jpg)

<a id="paper-5821"></a>

## Trajectory-Stabilized Inference for Diffusion-Based Video Inpainting


### Images

![12f3f43265778c68d9a7b889f4da0cd4e5c6a959564580de53a97256e2adcc4a.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/12f3f43265778c68d9a7b889f4da0cd4e5c6a959564580de53a97256e2adcc4a.jpg)

![300b71058a7bc224c90656c30ab9fe0f7f024787af19a2c41b529046bbf75e9c.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/300b71058a7bc224c90656c30ab9fe0f7f024787af19a2c41b529046bbf75e9c.jpg)

![63e89f5b90e147ffe1a6ffcddf8ca280dfd0e17178e29a1efaeab4d678cec8a9.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/63e89f5b90e147ffe1a6ffcddf8ca280dfd0e17178e29a1efaeab4d678cec8a9.jpg)

![7924008cb41ee3205dd67380f53835ade82fe508e80b5a1b1a1bcc7d7d2bfc92.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/7924008cb41ee3205dd67380f53835ade82fe508e80b5a1b1a1bcc7d7d2bfc92.jpg)

![7e9db6e989c9e23a629b707eaf9183ab811e5ce4a0af2367d141077edcaa6109.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/7e9db6e989c9e23a629b707eaf9183ab811e5ce4a0af2367d141077edcaa6109.jpg)

![aea5fb2fbfc89cc9675861a2c0506202fcfc534b8a45ff3034b4865d9c1e42c1.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/aea5fb2fbfc89cc9675861a2c0506202fcfc534b8a45ff3034b4865d9c1e42c1.jpg)

![b8535f66922420650bb03e77086275db420abe471869540e8f56902f1c088dff.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/b8535f66922420650bb03e77086275db420abe471869540e8f56902f1c088dff.jpg)

![d60dbcf6821f5c811083b2fe3fdf6efce25295b1927c5480df02f429c7c1da15.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/d60dbcf6821f5c811083b2fe3fdf6efce25295b1927c5480df02f429c7c1da15.jpg)

![e6b92cf143a61effe271cee91ea85f8249b053cad014cf7cda64f6d399ddb7d1.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/e6b92cf143a61effe271cee91ea85f8249b053cad014cf7cda64f6d399ddb7d1.jpg)

![f9df3f4aa59d9896b8a5d7ff14dea5b0b7aee243d8eda91bb2896e21124e8d98.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/images/f9df3f4aa59d9896b8a5d7ff14dea5b0b7aee243d8eda91bb2896e21124e8d98.jpg)

### Tables

![66ffec51916ccc0e9ea334643327736d0e3a1236ad39e32792645fd398c8a6f5.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/tables/66ffec51916ccc0e9ea334643327736d0e3a1236ad39e32792645fd398c8a6f5.jpg)

![ae46a6beacc56b7af17c292af9e888bfe8f9cdc7df2370ff319961c80a207e51.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/tables/ae46a6beacc56b7af17c292af9e888bfe8f9cdc7df2370ff319961c80a207e51.jpg)

![e88a11e7679688a8751224e98aeeb35424e4b9d05dfb3716ebafe5e7a5e4e978.jpg](../icml_results/5870_Trajectory-Stabilized%20Inference%20for%20Diffusion-Based%20Video%20Inpainting/tables/e88a11e7679688a8751224e98aeeb35424e4b9d05dfb3716ebafe5e7a5e4e978.jpg)

<a id="paper-5822"></a>

## CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions


### Images

![08842d3ff473eb2432f2bfa75fafcc553b4af5effd87c7b480d21bd288b71030.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/08842d3ff473eb2432f2bfa75fafcc553b4af5effd87c7b480d21bd288b71030.jpg)

![0b3f8494f28b20c74daaece09e1331ee7ed9da9fde91bd6f9ad9062e2e016542.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/0b3f8494f28b20c74daaece09e1331ee7ed9da9fde91bd6f9ad9062e2e016542.jpg)

![2b73c3e9971f096767b0080402989736a80d485abb0cbe6772cfa567765eb741.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/2b73c3e9971f096767b0080402989736a80d485abb0cbe6772cfa567765eb741.jpg)

![3e8f4f7f24031aa794bcf826d2380b39e8853dc129ff9aa6540f20771d2a8b53.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/3e8f4f7f24031aa794bcf826d2380b39e8853dc129ff9aa6540f20771d2a8b53.jpg)

![4afc6f0242868d48ae238ee008556d1dbe55d908ebca8533974eabb533fffa62.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/4afc6f0242868d48ae238ee008556d1dbe55d908ebca8533974eabb533fffa62.jpg)

![57eb59b498297f456df0c5b4a3312b3f148391d6aaa3698a46cba2c29273c7f8.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/57eb59b498297f456df0c5b4a3312b3f148391d6aaa3698a46cba2c29273c7f8.jpg)

![5e71751d4f348e33bae19dd8ff9b6fc6c83c8d8224913f2d5dfdff2e4876b02c.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/5e71751d4f348e33bae19dd8ff9b6fc6c83c8d8224913f2d5dfdff2e4876b02c.jpg)

![5fb2e48ce557cb02579ed5175605a2d5c6cdd9cbe2eb06e18abf37192837fa10.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/5fb2e48ce557cb02579ed5175605a2d5c6cdd9cbe2eb06e18abf37192837fa10.jpg)

![60de64ab597ea914561db834c3a600c41ada5799adabf9738a2a65cb1bd5196a.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/60de64ab597ea914561db834c3a600c41ada5799adabf9738a2a65cb1bd5196a.jpg)

![8fdb823b5bf8401994e6cb654ba712b2645c743b2c086217dc9033cf0b010ab6.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/8fdb823b5bf8401994e6cb654ba712b2645c743b2c086217dc9033cf0b010ab6.jpg)

![d390b3a7ad97cd872284e65a19f34159a4825743d52e54382b3769c353493ce0.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/d390b3a7ad97cd872284e65a19f34159a4825743d52e54382b3769c353493ce0.jpg)

![ef92c89bd293a51115decf1856f126d46812455ca14b4b5aa8eecf7082c4c279.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/ef92c89bd293a51115decf1856f126d46812455ca14b4b5aa8eecf7082c4c279.jpg)

![efe03f31efdbf729f01291135389d1586354c10c7ad41cadeb9ec83138b85eb2.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/images/efe03f31efdbf729f01291135389d1586354c10c7ad41cadeb9ec83138b85eb2.jpg)

### Tables

![3132179dabaca56096fa0e041aed1021ad53f56532158cf6bcaeee449979d293.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/3132179dabaca56096fa0e041aed1021ad53f56532158cf6bcaeee449979d293.jpg)

![317fd0824fb6435219847317a6f61263944962dee37c097e7c38e88594b575c9.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/317fd0824fb6435219847317a6f61263944962dee37c097e7c38e88594b575c9.jpg)

![52fbcfd27af480d0a9a7e65084f07246e89f27eb412e1eb1618d00c7387903a0.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/52fbcfd27af480d0a9a7e65084f07246e89f27eb412e1eb1618d00c7387903a0.jpg)

![6eb10924ca5e67e69de83f27438822693b76ad58a6ce176f09c3e844658eb545.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/6eb10924ca5e67e69de83f27438822693b76ad58a6ce176f09c3e844658eb545.jpg)

![7f2d59bed8f9253b3f528b090bd96557ca5da09b6dbcea917c5f7a75123fce19.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/7f2d59bed8f9253b3f528b090bd96557ca5da09b6dbcea917c5f7a75123fce19.jpg)

![92286bc3c7af1b478d8bf620221b09c7c08612ebbb43654b548a3e2fd81bd189.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/92286bc3c7af1b478d8bf620221b09c7c08612ebbb43654b548a3e2fd81bd189.jpg)

![994d85f6788b6116f9d3ad7760678a0fd41ee5927488f819f6c63d5c780173f1.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/994d85f6788b6116f9d3ad7760678a0fd41ee5927488f819f6c63d5c780173f1.jpg)

![a77367aae457c38b3f002f120716a381e0d649443fb213b0086489e6f0d767b7.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/a77367aae457c38b3f002f120716a381e0d649443fb213b0086489e6f0d767b7.jpg)

![af75f7bec9581613239b38afaea5b578365eb52b3e577e5291fc5386fe07a2c4.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/af75f7bec9581613239b38afaea5b578365eb52b3e577e5291fc5386fe07a2c4.jpg)

![b90b53a0570847cb0bfef17013a91867593e98c7da265c94b85b47a3e168eb6f.jpg](../icml_results/5871_CauchyNet_%20Compact%20and%20Data-Efficient%20Learning%20using%20Holomorphic%20Activation%20Functions/tables/b90b53a0570847cb0bfef17013a91867593e98c7da265c94b85b47a3e168eb6f.jpg)

<a id="paper-5823"></a>

## Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance


### Images

![01a966bc08ac9a4f2f61203e1b11f1d05a64fa70812c27b9f041ddc607c118a6.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/01a966bc08ac9a4f2f61203e1b11f1d05a64fa70812c27b9f041ddc607c118a6.jpg)

![1c82c926b1794a5a01a1351dcec9965c072566c750b25bcb47c3123aea7e2cef.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/1c82c926b1794a5a01a1351dcec9965c072566c750b25bcb47c3123aea7e2cef.jpg)

![1e3d624af9793b27bd108949855a0f60cf47c4814a845d02a049d7f79aa02e9d.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/1e3d624af9793b27bd108949855a0f60cf47c4814a845d02a049d7f79aa02e9d.jpg)

![300699f9f0c897af5597c134151744f6ec966e076ca6646ba79cd5202b031e5e.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/300699f9f0c897af5597c134151744f6ec966e076ca6646ba79cd5202b031e5e.jpg)

![312944bfec3bcf55ddbaa8f16274417e28dd2fee8f53d39543b410fc43d576c0.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/312944bfec3bcf55ddbaa8f16274417e28dd2fee8f53d39543b410fc43d576c0.jpg)

![3a78841e5e9583d6874ed4936cbaaef73fe96d19b4919fde9e3df84ffb402dca.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/3a78841e5e9583d6874ed4936cbaaef73fe96d19b4919fde9e3df84ffb402dca.jpg)

![45efe6a9c777de6d3d584a71b0d7f4f115fec4c49015bb6d6f2c9efe4992344b.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/45efe6a9c777de6d3d584a71b0d7f4f115fec4c49015bb6d6f2c9efe4992344b.jpg)

![4b0453607fdd3b899a57fed5263e429e65c2c33fe6bc9e1fb1f4d2a3975f1b2d.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/4b0453607fdd3b899a57fed5263e429e65c2c33fe6bc9e1fb1f4d2a3975f1b2d.jpg)

![539784ba6feb39144e9ee0f552989c56489d2dbb76e4fc52f7c7972224fa21f8.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/539784ba6feb39144e9ee0f552989c56489d2dbb76e4fc52f7c7972224fa21f8.jpg)

![58966c1ab5f975c01f80dfb83d5a2a49456e3aaf007c01ec3fd3ee0bb96ac825.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/58966c1ab5f975c01f80dfb83d5a2a49456e3aaf007c01ec3fd3ee0bb96ac825.jpg)

![6c2e9c74b11f6f6858ece173d22787617acbdd51daecf84cafabfa15e8d1573a.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/6c2e9c74b11f6f6858ece173d22787617acbdd51daecf84cafabfa15e8d1573a.jpg)

![70ea8085daa014a6fa71d78f4dedbf7a4f3058d2c2c1a3f65cfa51e2a6e45280.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/70ea8085daa014a6fa71d78f4dedbf7a4f3058d2c2c1a3f65cfa51e2a6e45280.jpg)

![76c96818ee95600a13507424010340c887bfe2c120b06db8c532e52ccb30d414.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/76c96818ee95600a13507424010340c887bfe2c120b06db8c532e52ccb30d414.jpg)

![80d9ffc43bf636b7c8272f4d2f2309d15449a5b39836c420363d156389f30ee7.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/80d9ffc43bf636b7c8272f4d2f2309d15449a5b39836c420363d156389f30ee7.jpg)

![86b89df6b7fc7794161bc098d09a2d028c9839b0a804ecc76c34e6e99a162002.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/86b89df6b7fc7794161bc098d09a2d028c9839b0a804ecc76c34e6e99a162002.jpg)

![87799c06dc03f40a4d6c7d6bdf258da3fcc93eb8c17ff77d2ccb2214d36c39b5.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/87799c06dc03f40a4d6c7d6bdf258da3fcc93eb8c17ff77d2ccb2214d36c39b5.jpg)

![a801e7695227fc30682bb1ddb72071908b1ef8325ec3e479f09b09646be89994.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/a801e7695227fc30682bb1ddb72071908b1ef8325ec3e479f09b09646be89994.jpg)

![bd1149079e51780453acac40f00f28e583f67f740cddca69b47ed6e8b41fa3b2.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/bd1149079e51780453acac40f00f28e583f67f740cddca69b47ed6e8b41fa3b2.jpg)

![c3fef9ce5008c5a7744355a8ea2426d2475794d8878d26fddf3c10058f08ae2d.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/c3fef9ce5008c5a7744355a8ea2426d2475794d8878d26fddf3c10058f08ae2d.jpg)

![d49d14db60cb74f7cc44b5b97a342de2178a5911db64496fe449910389f79fa9.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/d49d14db60cb74f7cc44b5b97a342de2178a5911db64496fe449910389f79fa9.jpg)

![e14971648717d940e9a09330db204d7ef91aba8b1e038f031ac1c772dca7eafb.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/e14971648717d940e9a09330db204d7ef91aba8b1e038f031ac1c772dca7eafb.jpg)

![e881d0dfbfc1f83f9294fa90caee9706c021b4af79e352d47c91a668d9285114.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/e881d0dfbfc1f83f9294fa90caee9706c021b4af79e352d47c91a668d9285114.jpg)

![e89fee4bf2b6468dcd34154cc9113b1db392e445ba859ec4303c7c704d885b4c.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/e89fee4bf2b6468dcd34154cc9113b1db392e445ba859ec4303c7c704d885b4c.jpg)

![e918876702a9641111daaed8de794c2f389b25a12bdc7f1711ca05aa271c6947.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/e918876702a9641111daaed8de794c2f389b25a12bdc7f1711ca05aa271c6947.jpg)

![f6670144e9bdfe1ba53edbd9091ad7391d582965f9b6dd8395ad68dbb7daa805.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/images/f6670144e9bdfe1ba53edbd9091ad7391d582965f9b6dd8395ad68dbb7daa805.jpg)

### Tables

![08a562bda4cd3cfae20bd38454303aeb020b46994bf9de0bf040f1fb13f2ec88.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/08a562bda4cd3cfae20bd38454303aeb020b46994bf9de0bf040f1fb13f2ec88.jpg)

![317cb5a10621674e57af5979d61a6080573c1a60e2b3cd7cae97ffbd6dc8b97a.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/317cb5a10621674e57af5979d61a6080573c1a60e2b3cd7cae97ffbd6dc8b97a.jpg)

![35888799ec62fb9800bfaa5eca11fbab80caad75e9f4662bf6ee108eefb037a7.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/35888799ec62fb9800bfaa5eca11fbab80caad75e9f4662bf6ee108eefb037a7.jpg)

![48e9727096a991750218104750e35be1e03218a651983bf2a29c44edb0879d60.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/48e9727096a991750218104750e35be1e03218a651983bf2a29c44edb0879d60.jpg)

![76438b2186578f298e3f6bfc9aa4bad4ab4959711a73732b6ae8d727b372e2c2.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/76438b2186578f298e3f6bfc9aa4bad4ab4959711a73732b6ae8d727b372e2c2.jpg)

![85936476c8fcedb523cd58a4869da46f01281afa0108c25b41dfb6b12fcdd18b.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/85936476c8fcedb523cd58a4869da46f01281afa0108c25b41dfb6b12fcdd18b.jpg)

![a0b4ec00807f0a0fd70c9962bb28f2ceef97fde45929f016da00058e18474b4d.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/a0b4ec00807f0a0fd70c9962bb28f2ceef97fde45929f016da00058e18474b4d.jpg)

![a91e183b860fbeaa0abea003a288060bcb93dc78151fcf3e51c21dd99bd46087.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/a91e183b860fbeaa0abea003a288060bcb93dc78151fcf3e51c21dd99bd46087.jpg)

![b616a17d13cc821a2be1496e25a8917c33e96c8238cab88be8891a74d7d844fa.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/b616a17d13cc821a2be1496e25a8917c33e96c8238cab88be8891a74d7d844fa.jpg)

![e9b8e3d8305a0e467640441e8f5893ca03323be504c581decc71b38d3ff4113d.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/e9b8e3d8305a0e467640441e8f5893ca03323be504c581decc71b38d3ff4113d.jpg)

![fb8da975446149b5bb71dde71dc4760e703004955fc2f3e6d10ea1e62abf88dd.jpg](../icml_results/5872_Active%20Learning%20with%20Foundation%20Model%20Priors_%20Efficient%20Learning%20under%20Class%20Imbalance/tables/fb8da975446149b5bb71dde71dc4760e703004955fc2f3e6d10ea1e62abf88dd.jpg)

<a id="paper-5824"></a>

## Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs


### Images

![1af7bebd98615461c89dd9f80155a1a05fbf20298f0cf309fc5e8e6efb520417.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/1af7bebd98615461c89dd9f80155a1a05fbf20298f0cf309fc5e8e6efb520417.jpg)

![40c63103012e50425ad693a66b9e8e56ba240c1ab9544c920bb7494de5cfe460.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/40c63103012e50425ad693a66b9e8e56ba240c1ab9544c920bb7494de5cfe460.jpg)

![4119e6e1e1eb9c524f093177a1504fd53bec123542980519ce3d9773933b095f.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/4119e6e1e1eb9c524f093177a1504fd53bec123542980519ce3d9773933b095f.jpg)

![4a9578da66ca8d3fcaf883a22e7d9e43ecda93c1251425db5b69f1e4dbb217e6.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/4a9578da66ca8d3fcaf883a22e7d9e43ecda93c1251425db5b69f1e4dbb217e6.jpg)

![8091198c133c15520060813461ae3c772f135a5be45a4e45ac8f00092cb86153.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/8091198c133c15520060813461ae3c772f135a5be45a4e45ac8f00092cb86153.jpg)

![830de55c23dd4bdea2125c706bda84bd34cf246a0503583ae9021a3b33b79300.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/830de55c23dd4bdea2125c706bda84bd34cf246a0503583ae9021a3b33b79300.jpg)

![dd0360d728dde287220077cbf0ff4e12ed26728abbdb2f49970478a29422a2b8.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/dd0360d728dde287220077cbf0ff4e12ed26728abbdb2f49970478a29422a2b8.jpg)

![e9cb8e5b649eb36c36b51a1153e965425fe90c9d7546efa8876b23bb4a4c1a3a.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/images/e9cb8e5b649eb36c36b51a1153e965425fe90c9d7546efa8876b23bb4a4c1a3a.jpg)

### Tables

![063b21423bd0d7aa403e1a69014fa2cf9ad830c042dd3def15b7b96676cec88d.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/063b21423bd0d7aa403e1a69014fa2cf9ad830c042dd3def15b7b96676cec88d.jpg)

![07b73444d011959644b07a935b3f7ac0c8eb782837eaca3992144d9d787bff10.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/07b73444d011959644b07a935b3f7ac0c8eb782837eaca3992144d9d787bff10.jpg)

![0d2db9c53fcbc5e359d65e24880f05ea3810b00f30609d1c2cc3b3e7705729d6.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/0d2db9c53fcbc5e359d65e24880f05ea3810b00f30609d1c2cc3b3e7705729d6.jpg)

![1b48b8aecae655424dcc914d00eefc347a53d4d28512bc511173741c43e8cb05.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/1b48b8aecae655424dcc914d00eefc347a53d4d28512bc511173741c43e8cb05.jpg)

![256abd773f7414bc61d647b9443829db7555518b0af0e050c029e3b0b8a1de9a.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/256abd773f7414bc61d647b9443829db7555518b0af0e050c029e3b0b8a1de9a.jpg)

![732389c0cc49a38daa7a94764485ac569814d3901cbe908f259d43766b374404.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/732389c0cc49a38daa7a94764485ac569814d3901cbe908f259d43766b374404.jpg)

![741ddf14a271337b8b688edf64ad8d30e45cc49e6764f8a8fb130d80c9f4e1e2.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/741ddf14a271337b8b688edf64ad8d30e45cc49e6764f8a8fb130d80c9f4e1e2.jpg)

![b0cb4ca18a61fe660f2d479de62262d6c146bf46df84105e30716daa2aa89878.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/b0cb4ca18a61fe660f2d479de62262d6c146bf46df84105e30716daa2aa89878.jpg)

![c90fcbc4c5e50ae45f56d2df2ca56c8351f1eaf225161068e19771de1d6b707a.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/c90fcbc4c5e50ae45f56d2df2ca56c8351f1eaf225161068e19771de1d6b707a.jpg)

![c93eae0bca59b3dc20475507d640dc8bd75eae74def8e8fa780439c0ebc243ac.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/c93eae0bca59b3dc20475507d640dc8bd75eae74def8e8fa780439c0ebc243ac.jpg)

![d1a792d3fae014af1442c66945851714aae64ca1441fe93b2b3c3f639f084688.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/d1a792d3fae014af1442c66945851714aae64ca1441fe93b2b3c3f639f084688.jpg)

![d52fd5ebe7b11936b95023ced238f8d8b43828f505d4ec59dfd4b751436285e8.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/d52fd5ebe7b11936b95023ced238f8d8b43828f505d4ec59dfd4b751436285e8.jpg)

![d7f1d9c29de2e82d83f7fbeed8482459fa69874168e474dddf091c2d38a86275.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/d7f1d9c29de2e82d83f7fbeed8482459fa69874168e474dddf091c2d38a86275.jpg)

![e2f3884ab05a4507cb3ab8342befb24ab9b6c79857c4ad6ab6b8f7433802fc58.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/e2f3884ab05a4507cb3ab8342befb24ab9b6c79857c4ad6ab6b8f7433802fc58.jpg)

![f92dcaa2c2c8a6ec50c48e33569c50333835fb6c6b6138f788fa882efb2095a0.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/f92dcaa2c2c8a6ec50c48e33569c50333835fb6c6b6138f788fa882efb2095a0.jpg)

![fa4b770874b46760488abe8e75ec406c6a81bcb0242be01ef8a2f3eb9e6dffc8.jpg](../icml_results/5873_Learning%20a%20Zeroth-Order%20Optimizer%20for%20Fine-Tuning%20LLMs/tables/fa4b770874b46760488abe8e75ec406c6a81bcb0242be01ef8a2f3eb9e6dffc8.jpg)

<a id="paper-5825"></a>

## MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification


### Images

![0e4dab2aa3b6f33452f55b55816c9da1a1889bc26ece469fd83e722c94433736.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/0e4dab2aa3b6f33452f55b55816c9da1a1889bc26ece469fd83e722c94433736.jpg)

![2d7c5aaa1cbb511e7865f54dbadff23c230550492b7924a665e00e0319cff544.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/2d7c5aaa1cbb511e7865f54dbadff23c230550492b7924a665e00e0319cff544.jpg)

![3174a9b9378e9652318c4e4cba5c2518f3f175c47e18c8dfac97e4ad5218a28d.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/3174a9b9378e9652318c4e4cba5c2518f3f175c47e18c8dfac97e4ad5218a28d.jpg)

![4e3813dddb369d00584d489d1a380c07cc16bff9b98426a2da9ed2e184033344.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/4e3813dddb369d00584d489d1a380c07cc16bff9b98426a2da9ed2e184033344.jpg)

![666013483e72addfd708551ad91d64831810b31ef928cf4030cd6e33ecefbbd0.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/666013483e72addfd708551ad91d64831810b31ef928cf4030cd6e33ecefbbd0.jpg)

![66b814f33dc40b18e49b4ad45e228bf50beba40866e5fe16e0625acdfeee2498.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/66b814f33dc40b18e49b4ad45e228bf50beba40866e5fe16e0625acdfeee2498.jpg)

![702ff8f8158079d1ff1f7741ba2cec827d49cad9d66db734a722260c3bd67403.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/702ff8f8158079d1ff1f7741ba2cec827d49cad9d66db734a722260c3bd67403.jpg)

![7daea909fda2b66bc3d040f5d90ebc877f967b84e27acdac682b9b89ac8d4754.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/7daea909fda2b66bc3d040f5d90ebc877f967b84e27acdac682b9b89ac8d4754.jpg)

![8e17fae34e35087e22e1f3083b3d1f84e34a1e8e35695d47e6cb30ca8960a2a7.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/8e17fae34e35087e22e1f3083b3d1f84e34a1e8e35695d47e6cb30ca8960a2a7.jpg)

![de18b132585c10278cae33b0aa87043478e0ffd18b08e4ff5c6e0a9bedbf6427.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/de18b132585c10278cae33b0aa87043478e0ffd18b08e4ff5c6e0a9bedbf6427.jpg)

![f8bc5e2d2a2946f4739547ab21d3916c850ce65fc088fa4081a4233353712b77.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/f8bc5e2d2a2946f4739547ab21d3916c850ce65fc088fa4081a4233353712b77.jpg)

![fc5d465cf21ea14e54aea50fa91fed2a46211ad0022a63ec781124a92490f692.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/fc5d465cf21ea14e54aea50fa91fed2a46211ad0022a63ec781124a92490f692.jpg)

![fe98eb63d5f7daea32c3d0964c89d2373bf9fdc15bce69661d9c2f3d2c939715.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/images/fe98eb63d5f7daea32c3d0964c89d2373bf9fdc15bce69661d9c2f3d2c939715.jpg)

### Tables

![406926aa12e1b4fab269c3ce9994c04d42e6d10b6fc21764cdcdaf00e71a4fde.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/406926aa12e1b4fab269c3ce9994c04d42e6d10b6fc21764cdcdaf00e71a4fde.jpg)

![5d80be0467ec0a2f8bd81f6d7d4218260bad405903d19b091a9f427dbfb030ae.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/5d80be0467ec0a2f8bd81f6d7d4218260bad405903d19b091a9f427dbfb030ae.jpg)

![73a13b5787bb0c779a90e768909b83b1dbbb4e9a09168c3fffec381ecd021e35.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/73a13b5787bb0c779a90e768909b83b1dbbb4e9a09168c3fffec381ecd021e35.jpg)

![75735b50c0c4e3f745a2acacfea26cffb04b8a4cc2c2b2787211c19eb1911b08.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/75735b50c0c4e3f745a2acacfea26cffb04b8a4cc2c2b2787211c19eb1911b08.jpg)

![9ccb2ec302ffd3f2cbe5019d84f2692d750507584b87f25ccc9fc3df7b7dc0e0.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/9ccb2ec302ffd3f2cbe5019d84f2692d750507584b87f25ccc9fc3df7b7dc0e0.jpg)

![b26d02c98265d3f4b169bdc76ac35f7e2734c668dd623611c1e8bb69dc55ed74.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/b26d02c98265d3f4b169bdc76ac35f7e2734c668dd623611c1e8bb69dc55ed74.jpg)

![c3328d2a9b7d9a7a399394000b3f5eeff07493f0a18a3c65d9b988a583696ff5.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/c3328d2a9b7d9a7a399394000b3f5eeff07493f0a18a3c65d9b988a583696ff5.jpg)

![f2e753c7f3090e93e7294a014383f557f693a557676126de69c3fafdb56f8286.jpg](../icml_results/5874_MedMamba_%20Multi-View%20State%20Space%20Models%20with%20Adaptive%20Graph%20Learning%20for%20Medical%20Time%20Series%20Classif/tables/f2e753c7f3090e93e7294a014383f557f693a557676126de69c3fafdb56f8286.jpg)

<a id="paper-5826"></a>

## VELR: Efficient Video Reward Feedback via Ensemble Latent Reward Models


### Images

![0dc7ded54d78f495f869f1d15a80b25ea305c345cf8acf76e1869c3d8a0ebe06.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/0dc7ded54d78f495f869f1d15a80b25ea305c345cf8acf76e1869c3d8a0ebe06.jpg)

![12b923dbfb97190f316a62fa11c24e4758d5ca63b9e5dd25011fc11bde311e3c.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/12b923dbfb97190f316a62fa11c24e4758d5ca63b9e5dd25011fc11bde311e3c.jpg)

![197eee47a04e1383d205ae370481437ee0d8df7ed0a4179db825df4410f12fcd.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/197eee47a04e1383d205ae370481437ee0d8df7ed0a4179db825df4410f12fcd.jpg)

![1eac4f8696cf2d5c355c06554d1254615ca49ef7efda3b9375861aa38d3e654f.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/1eac4f8696cf2d5c355c06554d1254615ca49ef7efda3b9375861aa38d3e654f.jpg)

![33c8c7247f300d6899c91004d4e25b0df38fa1bb81b247fc3f0297ef31b11d53.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/33c8c7247f300d6899c91004d4e25b0df38fa1bb81b247fc3f0297ef31b11d53.jpg)

![388275da78a128591e6e0f9a836ece152d49eb19c40d42ce65a2149dfd639943.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/388275da78a128591e6e0f9a836ece152d49eb19c40d42ce65a2149dfd639943.jpg)

![5142ae4d77bf57c38e7c6d3510b0675ef5a88455c697b36e4e76b4fa7e122828.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/5142ae4d77bf57c38e7c6d3510b0675ef5a88455c697b36e4e76b4fa7e122828.jpg)

![54b1e68d923c7839f1136cead88ca65b48a6a411a78196b43cec2553dc531273.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/54b1e68d923c7839f1136cead88ca65b48a6a411a78196b43cec2553dc531273.jpg)

![69e00df2045e91158a23648b89c31984531db324001586ddbc8e7ff461e33d3f.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/69e00df2045e91158a23648b89c31984531db324001586ddbc8e7ff461e33d3f.jpg)

![755e7fc263a7786c775990bdf58cca7e18eae0c8fb847a9370ed691a8cd8b8de.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/755e7fc263a7786c775990bdf58cca7e18eae0c8fb847a9370ed691a8cd8b8de.jpg)

![87fa12570a0c1b1e25eaf54fff28066e83cc6f10c453af97c8d5465dbb181c9a.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/87fa12570a0c1b1e25eaf54fff28066e83cc6f10c453af97c8d5465dbb181c9a.jpg)

![a078cdba95efd54037080cc4e23ab8b3e1c7240dda0084de6e3bce9a34dd1500.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/a078cdba95efd54037080cc4e23ab8b3e1c7240dda0084de6e3bce9a34dd1500.jpg)

![a5a177b111e239280412d5daaef30052824f8b13910aba6fce26858808415f46.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/a5a177b111e239280412d5daaef30052824f8b13910aba6fce26858808415f46.jpg)

![b0268de860fc6ac77363af7814ac7ce25884338d9cb217f384b55a8cf1f372d3.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/b0268de860fc6ac77363af7814ac7ce25884338d9cb217f384b55a8cf1f372d3.jpg)

![d437c8459cd6a7ee8b07a1c361b1d01c076b5593acfcb361a3c49681c395c78b.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/d437c8459cd6a7ee8b07a1c361b1d01c076b5593acfcb361a3c49681c395c78b.jpg)

![e1b1a51d14158a45365cd01d656833450876bf014989b4cfff3e72131cfbfd76.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/e1b1a51d14158a45365cd01d656833450876bf014989b4cfff3e72131cfbfd76.jpg)

![fbfc264a046f04c6a10cea9b892a7f90c2f65e98c1ec2d032d625ca0323dafbe.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/images/fbfc264a046f04c6a10cea9b892a7f90c2f65e98c1ec2d032d625ca0323dafbe.jpg)

### Tables

![226513e21aaf29785b611a0f5d69638d3d01f6e0afa00fa2fff486f63c3988b4.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/226513e21aaf29785b611a0f5d69638d3d01f6e0afa00fa2fff486f63c3988b4.jpg)

![31a59c6d43cac322412b2e9278f5df9087c8644a904da0ac9a8bf0ab554cb93c.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/31a59c6d43cac322412b2e9278f5df9087c8644a904da0ac9a8bf0ab554cb93c.jpg)

![326701efc58bb770384b326218a99db210cf0a40f58fea46043d90de951f4df0.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/326701efc58bb770384b326218a99db210cf0a40f58fea46043d90de951f4df0.jpg)

![3891ddf5ee63e79a399cf8f621c65667c7e776f9b19871b2856aa85792220019.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/3891ddf5ee63e79a399cf8f621c65667c7e776f9b19871b2856aa85792220019.jpg)

![46ea40cf0727061d4dc4363e983e65ef9f2b0653cfbb79c338219c11bee3d2b0.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/46ea40cf0727061d4dc4363e983e65ef9f2b0653cfbb79c338219c11bee3d2b0.jpg)

![478ae406cffa87b0aea6c9a342b4c8dd9b8693b1296114b9757d63220d6a2274.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/478ae406cffa87b0aea6c9a342b4c8dd9b8693b1296114b9757d63220d6a2274.jpg)

![48072f7ea850974ca376cf29b63480b5701a0b6194742d4855d56586103b68bc.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/48072f7ea850974ca376cf29b63480b5701a0b6194742d4855d56586103b68bc.jpg)

![4b34d122313c3a58536856ad7cfac8e7fb59d6ef53738d4c8a337bc73db0ce32.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/4b34d122313c3a58536856ad7cfac8e7fb59d6ef53738d4c8a337bc73db0ce32.jpg)

![601547cfdb8c82b7a0f399ae309591d7f42df274f4684404b921f5e8517f3e26.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/601547cfdb8c82b7a0f399ae309591d7f42df274f4684404b921f5e8517f3e26.jpg)

![9b28387b04b5cad8bdde2962273f8c53afcf9e3b8d45e0b73de04d209e8eb53e.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/9b28387b04b5cad8bdde2962273f8c53afcf9e3b8d45e0b73de04d209e8eb53e.jpg)

![a4b6dbfcddf8002d6442c64a0588b58d16ef9bc1e6fa854fd2c017036525483c.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/a4b6dbfcddf8002d6442c64a0588b58d16ef9bc1e6fa854fd2c017036525483c.jpg)

![cc9b0b4bb9f563ff4405f66607f9f50d0d2c1405e6740b58e358274d106a7e0e.jpg](../icml_results/5875_VELR_%20Efficient%20Video%20Reward%20Feedback%20via%20Ensemble%20Latent%20Reward%20Models/tables/cc9b0b4bb9f563ff4405f66607f9f50d0d2c1405e6740b58e358274d106a7e0e.jpg)

<a id="paper-5827"></a>

## Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models


### Images

![008d66354223dde67af730c2a3165648ff6a4727aa957dbc397f3e47229cf30d.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/008d66354223dde67af730c2a3165648ff6a4727aa957dbc397f3e47229cf30d.jpg)

![02c04175916ea27666438c6ef56b17c866e218752f126566ce4bf7dd5af07109.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/02c04175916ea27666438c6ef56b17c866e218752f126566ce4bf7dd5af07109.jpg)

![06ce943500ff7537c58274c6c0ab2588fc457dd47e3fef8608b498052e8214c9.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/06ce943500ff7537c58274c6c0ab2588fc457dd47e3fef8608b498052e8214c9.jpg)

![4f1d1993dc515d647047e7368ee88916def1b85d2b17033f6adbede74ea237d8.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/4f1d1993dc515d647047e7368ee88916def1b85d2b17033f6adbede74ea237d8.jpg)

![9264900f4602f0c9d02c87e2aaef5358d0a4ba391530fa174d2d459db612b7af.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/9264900f4602f0c9d02c87e2aaef5358d0a4ba391530fa174d2d459db612b7af.jpg)

![9a05723bf13c2867dfa34f67cc45e676f46a44cfe721dbe97bd2a047b3fccca8.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/9a05723bf13c2867dfa34f67cc45e676f46a44cfe721dbe97bd2a047b3fccca8.jpg)

![cf16a584a9c052b0d81df61fcbaee1cf57c174fe60553a677484c057808ae783.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/cf16a584a9c052b0d81df61fcbaee1cf57c174fe60553a677484c057808ae783.jpg)

![d2261f42d3c87581df31a6b43e4ed85325ca06b1b585b6d454f5ae5fa668c800.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/d2261f42d3c87581df31a6b43e4ed85325ca06b1b585b6d454f5ae5fa668c800.jpg)

![e0c12b940b3ec2316aa64077d5c19bd0a6f15d819ce020d1642e2df0eccec64d.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/e0c12b940b3ec2316aa64077d5c19bd0a6f15d819ce020d1642e2df0eccec64d.jpg)

![f4e99396b08e28cad339a5412d1eaa6c36b79f6958e6bd6902a033ea64226a77.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/images/f4e99396b08e28cad339a5412d1eaa6c36b79f6958e6bd6902a033ea64226a77.jpg)

### Tables

![316812fefe36c7531c00806f628afb288313132fe2c0a1a64ecf3e5f89670f0b.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/tables/316812fefe36c7531c00806f628afb288313132fe2c0a1a64ecf3e5f89670f0b.jpg)

![65c1ddac4cd0b20cd38ac8b167613c631faea42e85bc88f11f970dd1168fb5ec.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/tables/65c1ddac4cd0b20cd38ac8b167613c631faea42e85bc88f11f970dd1168fb5ec.jpg)

![8afc5d89c76850e2960abfac02782ba58a85b9b55e52d2943c6c9a52d3c6f0e2.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/tables/8afc5d89c76850e2960abfac02782ba58a85b9b55e52d2943c6c9a52d3c6f0e2.jpg)

![ed1a033af09a51fc634a763e32ca2745450d586485509099dd37fcaf7fc3aa0d.jpg](../icml_results/5876_Swordsman_%20Entropy-Driven%20Adaptive%20Block%20Partition%20for%20Efficient%20Diffusion%20Language%20Models/tables/ed1a033af09a51fc634a763e32ca2745450d586485509099dd37fcaf7fc3aa0d.jpg)

<a id="paper-5828"></a>

## Rethinking Human Intent-to-CAD: Parametric CAD Model Generation via Cooperative Multi-Task Alignment and Spatial-Aware Reinforcement Learning


### Images

![184c6f30c929880333a2fa64d4a5eebd2b6158532b3818ea39b85a2670bf776d.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/184c6f30c929880333a2fa64d4a5eebd2b6158532b3818ea39b85a2670bf776d.jpg)

![3108fcb956db6c7915dc9bbf2da47e27e7529287fe4131e45542754418e5454e.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/3108fcb956db6c7915dc9bbf2da47e27e7529287fe4131e45542754418e5454e.jpg)

![541a718710a7144e44170af789ec4bf25f3c2bd81a7ccc892dcef88851cc846c.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/541a718710a7144e44170af789ec4bf25f3c2bd81a7ccc892dcef88851cc846c.jpg)

![9a362a87eaf0dad6d294d4c35c7db556cbe1aa73332dbfea43798c867568d0be.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/9a362a87eaf0dad6d294d4c35c7db556cbe1aa73332dbfea43798c867568d0be.jpg)

![9a7c8391cfbd1ab7c7089188ca4cd1e6386eefedb532bb17c4719b8fbc254535.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/9a7c8391cfbd1ab7c7089188ca4cd1e6386eefedb532bb17c4719b8fbc254535.jpg)

![9f30f5c6325cc02bcfe58cf385b2dbb80392bd3a5f1f92a1268526fbf195ebb4.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/9f30f5c6325cc02bcfe58cf385b2dbb80392bd3a5f1f92a1268526fbf195ebb4.jpg)

![d920ee2eb4eb88ab9cefe55fd04b936a9abc461fafb90e729b85bb8f1a3248e2.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/d920ee2eb4eb88ab9cefe55fd04b936a9abc461fafb90e729b85bb8f1a3248e2.jpg)

![e77806dbb925ba5757bd6fec42c59097aa29ddb2dac92eec58e027076dcc2672.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/images/e77806dbb925ba5757bd6fec42c59097aa29ddb2dac92eec58e027076dcc2672.jpg)

### Tables

![20fff0c3c7dc7c71a45d60846e5e678e9bdb37fbad63896767c3722346e28eb2.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/tables/20fff0c3c7dc7c71a45d60846e5e678e9bdb37fbad63896767c3722346e28eb2.jpg)

![877fbe689879a19de123418aefc356fad2ed09f07750ef440dfd12716416b415.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/tables/877fbe689879a19de123418aefc356fad2ed09f07750ef440dfd12716416b415.jpg)

![aa693454c70ede4b550e53a68047f3474d20ff2e93e8f223b09a557f53f9b95d.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/tables/aa693454c70ede4b550e53a68047f3474d20ff2e93e8f223b09a557f53f9b95d.jpg)

![bb772f568ebc01d4b2792ce592df7a8696871561fdd26f9825719962ecbeb2e0.jpg](../icml_results/5877_Rethinking%20Human%20Intent-to-CAD_%20Parametric%20CAD%20Model%20Generation%20via%20Cooperative%20Multi-Task%20Alignment/tables/bb772f568ebc01d4b2792ce592df7a8696871561fdd26f9825719962ecbeb2e0.jpg)

<a id="paper-5829"></a>

## Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images


### Images

![0835ed95d8392f699e81caf53646644971e3c57cee3c3cb38aa422e73498e3d2.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/0835ed95d8392f699e81caf53646644971e3c57cee3c3cb38aa422e73498e3d2.jpg)

![26f1a3c20f92ddeb8bb6d6d3b75f9649bfcac0d08c32e8655cfe0e5078465cbd.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/26f1a3c20f92ddeb8bb6d6d3b75f9649bfcac0d08c32e8655cfe0e5078465cbd.jpg)

![4c994f89ff43d4076c01dfef54a032ffce23d83b468fcfb6e66db2f2c52ed313.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/4c994f89ff43d4076c01dfef54a032ffce23d83b468fcfb6e66db2f2c52ed313.jpg)

![778ec361fcaa5dc6914bbd8c7bf5d5e088a384a9a0c69c4517c798845349c6ca.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/778ec361fcaa5dc6914bbd8c7bf5d5e088a384a9a0c69c4517c798845349c6ca.jpg)

![9698a5ccfa441d3d38bc79e0ee2b16fbb526d22a49d1c16fad8c27a540506e94.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/9698a5ccfa441d3d38bc79e0ee2b16fbb526d22a49d1c16fad8c27a540506e94.jpg)

![ac8b1650146c556386b51f080db337e72e6335e57945da9ce0a81a5aef83238a.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/ac8b1650146c556386b51f080db337e72e6335e57945da9ce0a81a5aef83238a.jpg)

![d1f2de1f256d3eade1bd57d50b3aef17275b1d6850976b5cbca9e39b37d801db.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/d1f2de1f256d3eade1bd57d50b3aef17275b1d6850976b5cbca9e39b37d801db.jpg)

![dcc4def31d5c55299f461311bc7eb7433b13ed48c948d282a9b5f84349552c20.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/dcc4def31d5c55299f461311bc7eb7433b13ed48c948d282a9b5f84349552c20.jpg)

![e30313c2b55999e6e9e6ec2c122393953b16bb224dbdd08d3c76fa95b548df94.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/images/e30313c2b55999e6e9e6ec2c122393953b16bb224dbdd08d3c76fa95b548df94.jpg)

### Tables

![02a3a83c56b2b6d3d4e421d321c9c9745fd6f285bbebd56ef6f4268f0ee7b081.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/02a3a83c56b2b6d3d4e421d321c9c9745fd6f285bbebd56ef6f4268f0ee7b081.jpg)

![09c48c32f0a428b507d34541396fe52ec566f8ee4c6289c3c4276ccb0ab9c03c.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/09c48c32f0a428b507d34541396fe52ec566f8ee4c6289c3c4276ccb0ab9c03c.jpg)

![3177f6fd0589b1f29152c835cc62e0a028c389a66faab3eedf7d046599665be3.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/3177f6fd0589b1f29152c835cc62e0a028c389a66faab3eedf7d046599665be3.jpg)

![61a263b9b648a9e9321094eaeef674baedbf295ecc722dfa6ae0297f2ed807e1.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/61a263b9b648a9e9321094eaeef674baedbf295ecc722dfa6ae0297f2ed807e1.jpg)

![663155c5e574b50f6c4536146f72220aa37d026d44a8541a53afa30ade2d505c.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/663155c5e574b50f6c4536146f72220aa37d026d44a8541a53afa30ade2d505c.jpg)

![895ec8b03bd5fe5a9015a64cd45de5560e12fa297f18d6004e87e2b97e16704e.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/895ec8b03bd5fe5a9015a64cd45de5560e12fa297f18d6004e87e2b97e16704e.jpg)

![d98390dce2254016b7b8fcdbc0b9f4ffd5d2c2141eaee91ebcba3bbd570d4e06.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/d98390dce2254016b7b8fcdbc0b9f4ffd5d2c2141eaee91ebcba3bbd570d4e06.jpg)

![e795cd91d67de1681dca3713b176f420e77d5256ec14716450615ea72700e3c7.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/e795cd91d67de1681dca3713b176f420e77d5256ec14716450615ea72700e3c7.jpg)

![f80e7d549e882c6f0992d40afe0652d3c03ca02ae2faf3eac06c8aba107d8d35.jpg](../icml_results/5878_Structure-Aware%20Riemannian%20Flow%20Matching%20for%20Registration%20and%20Fusion%20of%20Hyperspectral%20and%20Multispect/tables/f80e7d549e882c6f0992d40afe0652d3c03ca02ae2faf3eac06c8aba107d8d35.jpg)

<a id="paper-5830"></a>

## CLIMB: Taming the LoRA Residency Cliff in Multi-LoRA Serving


### Images

![3bd1255399f0a6c8a435436a617067bf666bbfc65348f513147d1bf722cec141.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/3bd1255399f0a6c8a435436a617067bf666bbfc65348f513147d1bf722cec141.jpg)

![569f99e98906ec730db71d51b94254fa2489a47937da5fb49a5dfeee7051f478.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/569f99e98906ec730db71d51b94254fa2489a47937da5fb49a5dfeee7051f478.jpg)

![87b840cd8214a80a1333a3c018c55129187d2520280f36d1abfb20d042722ca9.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/87b840cd8214a80a1333a3c018c55129187d2520280f36d1abfb20d042722ca9.jpg)

![e3433af8ae99fd8ebee9fc0e8562556b6b2c2ad06545965992c31e3f43a2fdef.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/e3433af8ae99fd8ebee9fc0e8562556b6b2c2ad06545965992c31e3f43a2fdef.jpg)

![f5963500a95061f9b1619a74b08bb4c2180e6b9781b25bcbd5288b066e6df7f7.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/f5963500a95061f9b1619a74b08bb4c2180e6b9781b25bcbd5288b066e6df7f7.jpg)

![f8b43dd5e53fef698627bdc831755fcbf857b0eea33ddeea3df278151676a0fa.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/images/f8b43dd5e53fef698627bdc831755fcbf857b0eea33ddeea3df278151676a0fa.jpg)

### Tables

![03cce6653f99c67f2e6ef35b4e733b8b1acf3cd7c85ae2d5580d5f2d8c0b8d3a.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/03cce6653f99c67f2e6ef35b4e733b8b1acf3cd7c85ae2d5580d5f2d8c0b8d3a.jpg)

![04533cf2847bda5bb1b41b452be47bd0c4b247d65f20af2b27f1f7f618a7faad.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/04533cf2847bda5bb1b41b452be47bd0c4b247d65f20af2b27f1f7f618a7faad.jpg)

![0a149945ef4266ea15f54a3843b5784a524661226cf52f6bb70082d2009d6581.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/0a149945ef4266ea15f54a3843b5784a524661226cf52f6bb70082d2009d6581.jpg)

![3a1329f9cab8787e1d61625689261f99efdf30982b6287f80089e0706dbd16c0.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/3a1329f9cab8787e1d61625689261f99efdf30982b6287f80089e0706dbd16c0.jpg)

![49696a86720a049227ee9f9fd9a8cee47c17d5fff6cf04af0f82c7b5f82db8ee.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/49696a86720a049227ee9f9fd9a8cee47c17d5fff6cf04af0f82c7b5f82db8ee.jpg)

![4abe00d7dfc66a983bae490973d365bee6137cd64f466a528e06677bfe98ec61.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/4abe00d7dfc66a983bae490973d365bee6137cd64f466a528e06677bfe98ec61.jpg)

![79797fad71ffc78fb45be55fbfd9ce4464ad639072ce8fce9f22fabb41de7425.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/79797fad71ffc78fb45be55fbfd9ce4464ad639072ce8fce9f22fabb41de7425.jpg)

![ad3da88a398d8ff10d081d29df1b627f42fd351f426ed366357b4670ba7b8fc1.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/ad3da88a398d8ff10d081d29df1b627f42fd351f426ed366357b4670ba7b8fc1.jpg)

![cd00bcc95967d75acb963b9e870b568f99e41c6dd2de5df7514c78d16775e873.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/cd00bcc95967d75acb963b9e870b568f99e41c6dd2de5df7514c78d16775e873.jpg)

![dacb1760c1c127102c3fe5d09756e06a28c49632fe07b86885770ae294676292.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/dacb1760c1c127102c3fe5d09756e06a28c49632fe07b86885770ae294676292.jpg)

![dd93045038199c13447c43a1531bc98ac01035b82097296a9d9dea692a531393.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/dd93045038199c13447c43a1531bc98ac01035b82097296a9d9dea692a531393.jpg)

![deab910e92e49d6bd76a8875d7ca541d9854fd45543825c9f065e8d612505dc0.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/deab910e92e49d6bd76a8875d7ca541d9854fd45543825c9f065e8d612505dc0.jpg)

![e23e4048547520786149467ef595d1f061e5bd740eb79fe640870f9bd1d1ca0e.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/e23e4048547520786149467ef595d1f061e5bd740eb79fe640870f9bd1d1ca0e.jpg)

![e882e6d591c22499e8879f3eba3aba84e773741151227761d8c30ac2eabfd58c.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/e882e6d591c22499e8879f3eba3aba84e773741151227761d8c30ac2eabfd58c.jpg)

![e99737f50d36df500ca2cdc38f7e1b20beafa63f1dd6b98acdbe8424972cb6af.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/e99737f50d36df500ca2cdc38f7e1b20beafa63f1dd6b98acdbe8424972cb6af.jpg)

![f1eaaf503b65e8e47de0d7c6a8fe26b2fb01d97741fc2a8082d44330cf18aa94.jpg](../icml_results/5879_CLIMB_%20Taming%20the%20LoRA%20Residency%20Cliff%20in%20Multi-LoRA%20Serving/tables/f1eaaf503b65e8e47de0d7c6a8fe26b2fb01d97741fc2a8082d44330cf18aa94.jpg)

<a id="paper-5831"></a>

## ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World


### Images

![4386fea83df4b5e1b779c7914841b962a2711b6dc90d543a17d802d6381e200f.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/4386fea83df4b5e1b779c7914841b962a2711b6dc90d543a17d802d6381e200f.jpg)

![702dd6a21ad1fe5dc3c23abdc248dc9ea0c1fbd43283ff083d400217c9bace17.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/702dd6a21ad1fe5dc3c23abdc248dc9ea0c1fbd43283ff083d400217c9bace17.jpg)

![82d0d02fa2b7dd66f61ede8dc5e887da535460f285a59ccd1836d81468e0b5f5.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/82d0d02fa2b7dd66f61ede8dc5e887da535460f285a59ccd1836d81468e0b5f5.jpg)

![9db8f849bb2fe9a33b49eb6bbee07c22ad09ac9929ff59e8164e522cd2100110.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/9db8f849bb2fe9a33b49eb6bbee07c22ad09ac9929ff59e8164e522cd2100110.jpg)

![a7060cd6a14bc83ee28a1f4752bb982cdcdfb458332782fafca72791641857d5.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/a7060cd6a14bc83ee28a1f4752bb982cdcdfb458332782fafca72791641857d5.jpg)

![dd7d3d933e830d22f6c0c7004cb78fbf2f7fbb5ad48b273ae8c93ab3ca1792e6.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/dd7d3d933e830d22f6c0c7004cb78fbf2f7fbb5ad48b273ae8c93ab3ca1792e6.jpg)

![e0941ea79832c11a5682ddc6568ed37cb623e95af7ae379d04bab53ff62962aa.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/images/e0941ea79832c11a5682ddc6568ed37cb623e95af7ae379d04bab53ff62962aa.jpg)

### Tables

![07ba239df2bc62be2a2b5a64626e5f2a07914489e43c42b2e14a26066b70546e.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/07ba239df2bc62be2a2b5a64626e5f2a07914489e43c42b2e14a26066b70546e.jpg)

![0d8c0cc0e1cf04dcd524558ff215e3753943861f8f196293718d66cd0f294c3b.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/0d8c0cc0e1cf04dcd524558ff215e3753943861f8f196293718d66cd0f294c3b.jpg)

![23899b61478f3d56336d9a5df650c6ade80b55177cbf02e9a11e2d490499d4bb.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/23899b61478f3d56336d9a5df650c6ade80b55177cbf02e9a11e2d490499d4bb.jpg)

![2d2c68599a164d427f809ba2c73bb313eb98607a02ce3e62cfea28cfba71684f.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/2d2c68599a164d427f809ba2c73bb313eb98607a02ce3e62cfea28cfba71684f.jpg)

![2f1360329b94f52cb07d8d818d24c9e3546636d1a409589faf3f0a38e3a68642.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/2f1360329b94f52cb07d8d818d24c9e3546636d1a409589faf3f0a38e3a68642.jpg)

![371365d3c9a6527f6cf592957b096bc9f8509f4ef6faa3bf77eea33b3bbfdf19.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/371365d3c9a6527f6cf592957b096bc9f8509f4ef6faa3bf77eea33b3bbfdf19.jpg)

![392b9155d973a98d5f3428314aaff10167918f0a334e01ed85fb7e6cc179aeba.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/392b9155d973a98d5f3428314aaff10167918f0a334e01ed85fb7e6cc179aeba.jpg)

![4253dcc1d5af57c7ff09a01423e5c8c13258336724071fdfb4850497646d4465.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/4253dcc1d5af57c7ff09a01423e5c8c13258336724071fdfb4850497646d4465.jpg)

![515a85c9e207069aca5d4f9eb5dcf5cda262c13a51b179e1f8fed14d27516eba.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/515a85c9e207069aca5d4f9eb5dcf5cda262c13a51b179e1f8fed14d27516eba.jpg)

![58f2f1240d0beaf5613b826e99cba4a89d0344d0d9ec4fc9c46fc649711a1eea.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/58f2f1240d0beaf5613b826e99cba4a89d0344d0d9ec4fc9c46fc649711a1eea.jpg)

![6633b4027d99e6695344a9a8a67ee75b05fce62f6de0c813c3302aafda39ad3f.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/6633b4027d99e6695344a9a8a67ee75b05fce62f6de0c813c3302aafda39ad3f.jpg)

![671722ff6faa4ec868d1d44ca587d438b544f448988494d04d93b66a9d2d2047.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/671722ff6faa4ec868d1d44ca587d438b544f448988494d04d93b66a9d2d2047.jpg)

![785e888fcf37c048c76c3c71b09d952773f36e7d17e0e3e9591e0364b0145e4f.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/785e888fcf37c048c76c3c71b09d952773f36e7d17e0e3e9591e0364b0145e4f.jpg)

![823ed8406c6022b9e7dc40fd81fedb971fcd70f3ceb0b6ba3daffd581d064091.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/823ed8406c6022b9e7dc40fd81fedb971fcd70f3ceb0b6ba3daffd581d064091.jpg)

![96cf7986f608183dfb617c2b1494f69b96c4a720f05fbe7327eb93d251856039.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/96cf7986f608183dfb617c2b1494f69b96c4a720f05fbe7327eb93d251856039.jpg)

![e390fb7dc57da392ac03d4410b15d6f8764cad6603d6f0603b5a331570081e8a.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/e390fb7dc57da392ac03d4410b15d6f8764cad6603d6f0603b5a331570081e8a.jpg)

![e8ba090232a3c6391cfd7637908023a87514d782cb1a0249cdb7bc3a9eca574b.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/e8ba090232a3c6391cfd7637908023a87514d782cb1a0249cdb7bc3a9eca574b.jpg)

![f7ada7b89231b68255cf119efa0b9efac89b5b33c601370b513c89116b432d11.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/f7ada7b89231b68255cf119efa0b9efac89b5b33c601370b513c89116b432d11.jpg)

![fa27a5fe9fc0e8e1fed62445b2ba3e4200ca736a3550595b42523335f307c4a9.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/fa27a5fe9fc0e8e1fed62445b2ba3e4200ca736a3550595b42523335f307c4a9.jpg)

![fb286a19355972a26a90daa6703f9fef236ee1193bef299e75979757e7d56fff.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/fb286a19355972a26a90daa6703f9fef236ee1193bef299e75979757e7d56fff.jpg)

![fcab821cd562d53a1bdc021c48e9586c7ef778c43b5ce5780026e8fb3be1fee1.jpg](../icml_results/5880_ML-Embed_%20Inclusive%20and%20Efficient%20Embeddings%20for%20a%20Multilingual%20World/tables/fcab821cd562d53a1bdc021c48e9586c7ef778c43b5ce5780026e8fb3be1fee1.jpg)

<a id="paper-5832"></a>

## Adversarial Reinforcement Learning for Robust Diffusion Large Language Model Unlearning


### Images

![44527619dc7d8a6b3f81ca475709cddf6598f4ebee0d5390c689ffe031c03b0c.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/44527619dc7d8a6b3f81ca475709cddf6598f4ebee0d5390c689ffe031c03b0c.jpg)

![49fc396b98b9a91133fbb6d1efe573192180670136a48c73419a0f88ae6553b3.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/49fc396b98b9a91133fbb6d1efe573192180670136a48c73419a0f88ae6553b3.jpg)

![73c35a5ee4f8b917789cb5926f2d71fa823328c4ea80cfd07f289ed3f4924d42.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/73c35a5ee4f8b917789cb5926f2d71fa823328c4ea80cfd07f289ed3f4924d42.jpg)

![a6357c7cbc16d2b4578870b223b929bdc2300a7ee459ffadf7ae01662af91fda.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/a6357c7cbc16d2b4578870b223b929bdc2300a7ee459ffadf7ae01662af91fda.jpg)

![e3f3c08fb6f9c751ee38ab4eed2fd59440e06c0f53179aa519b3426b988d041a.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/e3f3c08fb6f9c751ee38ab4eed2fd59440e06c0f53179aa519b3426b988d041a.jpg)

![f446bd544c20cfe4aeb459063a11c3e858557efabe4aedb812c5f07ff4bd6a58.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/images/f446bd544c20cfe4aeb459063a11c3e858557efabe4aedb812c5f07ff4bd6a58.jpg)

### Tables

![104dfbe339160484a454da2246c0be8679ad42ddbbb27d405953af4631d37613.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/tables/104dfbe339160484a454da2246c0be8679ad42ddbbb27d405953af4631d37613.jpg)

![762d77fe9b9bb765179344a22c41e1657bd9dd22bfac6c2d14d3b81f5cef8faa.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/tables/762d77fe9b9bb765179344a22c41e1657bd9dd22bfac6c2d14d3b81f5cef8faa.jpg)

![9e2e83ac4a1c37eb8a7493ba90a495756c3de0118eae540fe1dd783fa2f88c6b.jpg](../icml_results/5881_Adversarial%20Reinforcement%20Learning%20for%20Robust%20Diffusion%20Large%20Language%20Model%20Unlearning/tables/9e2e83ac4a1c37eb8a7493ba90a495756c3de0118eae540fe1dd783fa2f88c6b.jpg)

<a id="paper-5833"></a>

## R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model?


### Images

![026eb7af4e95bf735ae2a98640def5950f92a28c02804a565206c40ee1681ab3.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/026eb7af4e95bf735ae2a98640def5950f92a28c02804a565206c40ee1681ab3.jpg)

![23cf4f11a656d1c8a43dba039bee1dc92d7df126b558f0efcc7db0d579faf712.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/23cf4f11a656d1c8a43dba039bee1dc92d7df126b558f0efcc7db0d579faf712.jpg)

![33d13f71335be91f88ec44d11f2914b1962fdc1d39a23ca9873d59f6d4b198a6.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/33d13f71335be91f88ec44d11f2914b1962fdc1d39a23ca9873d59f6d4b198a6.jpg)

![6195ead7f0ba39dceacea2c80c985c4b5406b60f9f0bbe1f2b543d2520bb5c37.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/6195ead7f0ba39dceacea2c80c985c4b5406b60f9f0bbe1f2b543d2520bb5c37.jpg)

![84b22a9b41845b27d8808777291c8a90a163ea40c0aba6494652070f045bca41.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/84b22a9b41845b27d8808777291c8a90a163ea40c0aba6494652070f045bca41.jpg)

![8dc3f542b50eba8651986f601dffb0e52d9eaac3c0f1e640f530b402edde4248.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/8dc3f542b50eba8651986f601dffb0e52d9eaac3c0f1e640f530b402edde4248.jpg)

![8df64d4d91054b377e75ce596715e002ee4ecafaa5cb264ce1cdec5d1a0f7f10.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/8df64d4d91054b377e75ce596715e002ee4ecafaa5cb264ce1cdec5d1a0f7f10.jpg)

![a5288a9ef451ec16ea40006253db533fdb3048e6ce28a12001eedefabb5a4633.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/a5288a9ef451ec16ea40006253db533fdb3048e6ce28a12001eedefabb5a4633.jpg)

![bc5db5886c286ba16311691dcfded669e30fa8dca8abaca712791f9aaf763d8d.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/bc5db5886c286ba16311691dcfded669e30fa8dca8abaca712791f9aaf763d8d.jpg)

![c05c319edd328772d242a0a82a374ee1b618b5896326949c68ff345a73db44cc.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/c05c319edd328772d242a0a82a374ee1b618b5896326949c68ff345a73db44cc.jpg)

![ecb2045aab41cd495188ab16085d5c6e3b138ae2b8ec7385a84d1886349f6e03.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/ecb2045aab41cd495188ab16085d5c6e3b138ae2b8ec7385a84d1886349f6e03.jpg)

![f53e00f6841deec2553552edfa09ffa330a08bd5f6b35665673c0bb31d9b6f31.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/images/f53e00f6841deec2553552edfa09ffa330a08bd5f6b35665673c0bb31d9b6f31.jpg)

### Tables

![0e4be49d9400626202bbfc4752b927f8ac42c4003933bf60551df3b3bc373c27.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/0e4be49d9400626202bbfc4752b927f8ac42c4003933bf60551df3b3bc373c27.jpg)

![0ea34443e445bdd39491ef1c01f3d7d785a30a7b4fef50c1dc77dbb1e422c8db.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/0ea34443e445bdd39491ef1c01f3d7d785a30a7b4fef50c1dc77dbb1e422c8db.jpg)

![273b72099099cd3e1e42ba5e0c05de45676e0de64b852985dc3e3e2e7c6a6c6d.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/273b72099099cd3e1e42ba5e0c05de45676e0de64b852985dc3e3e2e7c6a6c6d.jpg)

![9c5c525af34119486331aaef600b1bb8cedf540a7b414f4657ec73429c25da4d.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/9c5c525af34119486331aaef600b1bb8cedf540a7b414f4657ec73429c25da4d.jpg)

![af55778b16dd5a88d905722afd774fe2eb8870cc9da6bba0d7b047bb1075022a.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/af55778b16dd5a88d905722afd774fe2eb8870cc9da6bba0d7b047bb1075022a.jpg)

![c46ac8d155735179dbbba87d4cfeea6c156fc9e1d1f50a3456fc6b7b7bcb4fd1.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/c46ac8d155735179dbbba87d4cfeea6c156fc9e1d1f50a3456fc6b7b7bcb4fd1.jpg)

![eb73bfe9b7b1e4e26924c2eae8a355dbee6b1d327c73c49c0414e76410fc9ecd.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/eb73bfe9b7b1e4e26924c2eae8a355dbee6b1d327c73c49c0414e76410fc9ecd.jpg)

![ee5351c988382b5336668a3b066724dc39b4b6df945abefb92f9d87e754257f4.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/ee5351c988382b5336668a3b066724dc39b4b6df945abefb92f9d87e754257f4.jpg)

![f2b91930de30edfe9e7275f1df40de4746a83cc250cc0bf87223ef6b84fc7f03.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/f2b91930de30edfe9e7275f1df40de4746a83cc250cc0bf87223ef6b84fc7f03.jpg)

![f9880dccf6ae7b9acb05a4e0c5413fee4dda633f8effb11bf2963a8cf3f559ed.jpg](../icml_results/5882_R1-SyntheticVL_%20Is%20Synthetic%20Data%20from%20Generative%20Models%20Ready%20for%20Multimodal%20Large%20Language%20Model_/tables/f9880dccf6ae7b9acb05a4e0c5413fee4dda633f8effb11bf2963a8cf3f559ed.jpg)

<a id="paper-5834"></a>

## UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation


### Images

![32a39004fbdb7e879a6b7786ee28eb8dd58cf13e57a81ea3cec755b62397ae8e.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/32a39004fbdb7e879a6b7786ee28eb8dd58cf13e57a81ea3cec755b62397ae8e.jpg)

![3cac19b7312f1fbd1012396150d48f666bbc4f7feeb11077186690d400cee977.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/3cac19b7312f1fbd1012396150d48f666bbc4f7feeb11077186690d400cee977.jpg)

![744287265c5e342ba34b621f44a005b5ba8433de7a4dd889290e2857c0fd609f.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/744287265c5e342ba34b621f44a005b5ba8433de7a4dd889290e2857c0fd609f.jpg)

![7806c01c96f19a43b7bbe4368007156fdfc6f6df34a32d6621051eeb3d93f8a9.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/7806c01c96f19a43b7bbe4368007156fdfc6f6df34a32d6621051eeb3d93f8a9.jpg)

![8d051dc769502950f0bc80d484b9f1683f9908c4686d25bb9b83e6dc11e29b93.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/8d051dc769502950f0bc80d484b9f1683f9908c4686d25bb9b83e6dc11e29b93.jpg)

![f87fccf20ddb96a37e6779e25d8a02be7146bc4439ea925af886d92debaf832a.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/images/f87fccf20ddb96a37e6779e25d8a02be7146bc4439ea925af886d92debaf832a.jpg)

### Tables

![2772411fba31658cca2131e88417ed279cbd26aa1385918f293446483a5d5088.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/tables/2772411fba31658cca2131e88417ed279cbd26aa1385918f293446483a5d5088.jpg)

![c179c347ab4cb59e229cd73d77ce74941da4f6f5b588593212c5d6ff55ace9a4.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/tables/c179c347ab4cb59e229cd73d77ce74941da4f6f5b588593212c5d6ff55ace9a4.jpg)

![fce686bf3e140a6b7a6941725fce42fabecde4e9dff56c8a1df6a8ffd9fddd31.jpg](../icml_results/5883_UFO_%20Chain-of-Evaluation%20for%20Omni-Condition%20Alignment%20in%20Multi-Modal%20Image%20Generation/tables/fce686bf3e140a6b7a6941725fce42fabecde4e9dff56c8a1df6a8ffd9fddd31.jpg)

<a id="paper-5835"></a>

## On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression


### Images

![358888df92f309e4fc9567bfe65632892ee84fa843370c99df7928018d1cb114.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/358888df92f309e4fc9567bfe65632892ee84fa843370c99df7928018d1cb114.jpg)

![658dbd7f2faabb0617c9cb418f4e0884d6b9e9c88d005776ce78856e134144eb.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/658dbd7f2faabb0617c9cb418f4e0884d6b9e9c88d005776ce78856e134144eb.jpg)

![7736cf315c2b53f6a875650e3f5b01866d322092db9623b718c495ede14c69a8.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/7736cf315c2b53f6a875650e3f5b01866d322092db9623b718c495ede14c69a8.jpg)

![8bd7b7421b251dc8f852eee53ee1ef09f5f4c2dada0850ee0624dec9a7502116.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/8bd7b7421b251dc8f852eee53ee1ef09f5f4c2dada0850ee0624dec9a7502116.jpg)

![a5c593eddc0bf4bdfe27cfa1aea00ef2a4b04ee6570a956c82fb2be9f63c9115.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/a5c593eddc0bf4bdfe27cfa1aea00ef2a4b04ee6570a956c82fb2be9f63c9115.jpg)

![c88fd23da45d5803f2e27b3b05ecbb6c9366199fa857b1036bb7644934b70dc8.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/c88fd23da45d5803f2e27b3b05ecbb6c9366199fa857b1036bb7644934b70dc8.jpg)

![f5b1ee264ac00e56a998a093ff80afe59b5ddc95e40ef82e7ecc0967af38dbfc.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/images/f5b1ee264ac00e56a998a093ff80afe59b5ddc95e40ef82e7ecc0967af38dbfc.jpg)

### Tables

![01c679298d10279d59340a9728d7404eb0a6012087d2953a957f2c93b5828813.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/01c679298d10279d59340a9728d7404eb0a6012087d2953a957f2c93b5828813.jpg)

![06756bac757e8f0719941921df84000a9f5778990af91ee132b904bdba5f4927.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/06756bac757e8f0719941921df84000a9f5778990af91ee132b904bdba5f4927.jpg)

![2b8ef9ddba650b7b48ecb32659b310b85dc2e5db0ae340c9e5b75a727ecb8a9d.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/2b8ef9ddba650b7b48ecb32659b310b85dc2e5db0ae340c9e5b75a727ecb8a9d.jpg)

![2c5500bfcddb300af21adf92fb3091dc0dcef93af58c3c03f50e194d095595c8.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/2c5500bfcddb300af21adf92fb3091dc0dcef93af58c3c03f50e194d095595c8.jpg)

![41894a51c9a4b65f74f1fec231c04dfc90c77f6137e2386992e16191292643f6.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/41894a51c9a4b65f74f1fec231c04dfc90c77f6137e2386992e16191292643f6.jpg)

![47535f0e5e00a1804fbd614d7fb86808487724d0543cc691b5ca61e3a8c65435.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/47535f0e5e00a1804fbd614d7fb86808487724d0543cc691b5ca61e3a8c65435.jpg)

![4ef303c94807952865f397a8adb397a081b76d66ecbcb0b91ef09a91be154380.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/4ef303c94807952865f397a8adb397a081b76d66ecbcb0b91ef09a91be154380.jpg)

![645f8305ad4541c7f1a248ae58211caef1ddf833b293baa9bd784f8712056030.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/645f8305ad4541c7f1a248ae58211caef1ddf833b293baa9bd784f8712056030.jpg)

![79e4f518014b95e9b879827d17e2064b99c521dd7864f033b8b879c6f3c61f17.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/79e4f518014b95e9b879827d17e2064b99c521dd7864f033b8b879c6f3c61f17.jpg)

![82e8cb46a4b5c9ab215d9a03c45014d0af5be0f38f92305e2d78ab6ce7b5152f.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/82e8cb46a4b5c9ab215d9a03c45014d0af5be0f38f92305e2d78ab6ce7b5152f.jpg)

![8ecdaa2f7a48f29b93cebcc09462d72057608011f59e3e6a71bdd4c1d0131ecb.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/8ecdaa2f7a48f29b93cebcc09462d72057608011f59e3e6a71bdd4c1d0131ecb.jpg)

![9119b12b7915d587c5478671ba01b398613ac64975ee00b03cad196a3ac0cb03.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/9119b12b7915d587c5478671ba01b398613ac64975ee00b03cad196a3ac0cb03.jpg)

![917075ef7040cebc3788df07873df55f907f0207b821d83376e912343fc791ea.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/917075ef7040cebc3788df07873df55f907f0207b821d83376e912343fc791ea.jpg)

![b3c04ca66985c78f31f49078b5c473c2187aad5c64d38c606171142412085af3.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/b3c04ca66985c78f31f49078b5c473c2187aad5c64d38c606171142412085af3.jpg)

![c324bddc43089b20aeb2a01ca3ba8babc88c37167bfc56be2bd0bb1481d19348.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/c324bddc43089b20aeb2a01ca3ba8babc88c37167bfc56be2bd0bb1481d19348.jpg)

![ca3f482b244caf8dd7dfbce909660294135de135e11bb9d16182b9aa0aa20229.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/ca3f482b244caf8dd7dfbce909660294135de135e11bb9d16182b9aa0aa20229.jpg)

![dca5bed6238a255867f7eefb12405d6ac7782fd9cfdb850eaaee345cf2732cf3.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/dca5bed6238a255867f7eefb12405d6ac7782fd9cfdb850eaaee345cf2732cf3.jpg)

![ddbd4240dbade49a19cdd1b7f04a66a23a1dc52d62dc7f29d96a66291ae7efd5.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/ddbd4240dbade49a19cdd1b7f04a66a23a1dc52d62dc7f29d96a66291ae7efd5.jpg)

![e301c74d54887ef5b4bd9d71be6bd79312f41060433e042c2f24e7fe1dfc4c88.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/e301c74d54887ef5b4bd9d71be6bd79312f41060433e042c2f24e7fe1dfc4c88.jpg)

![e84b6a833f2bf9d0b18a29dec04a42a8b5bcd221b81df59ddce2e0c99611066f.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/e84b6a833f2bf9d0b18a29dec04a42a8b5bcd221b81df59ddce2e0c99611066f.jpg)

![e940ea6cd469bb85ee0621d13d5d70b2d89404a95b9cab4b41f3b7dbb81beca0.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/e940ea6cd469bb85ee0621d13d5d70b2d89404a95b9cab4b41f3b7dbb81beca0.jpg)

![eb4b55d5273e394263bd0fd454a4420e913eacc6d40b59a7375504187975c37c.jpg](../icml_results/5884_On%20the%20Adversarial%20Robustness%20of%20Large%20Vision-Language%20Models%20under%20Visual%20Token%20Compression/tables/eb4b55d5273e394263bd0fd454a4420e913eacc6d40b59a7375504187975c37c.jpg)

<a id="paper-5836"></a>

## InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution


### Images

![21d2a5c19a00d7def7847921ad8db9a6bec23ee415f479ca15aa8abc05cf005e.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/21d2a5c19a00d7def7847921ad8db9a6bec23ee415f479ca15aa8abc05cf005e.jpg)

![2ab5982e1590de059087935b6e05a81bee9bff4863034397ee3ccefb3a103da2.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/2ab5982e1590de059087935b6e05a81bee9bff4863034397ee3ccefb3a103da2.jpg)

![3ec099a082197dec1a44a06beea0ff225bf34c808bcdd7f2326d7a52e1c2a2fa.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/3ec099a082197dec1a44a06beea0ff225bf34c808bcdd7f2326d7a52e1c2a2fa.jpg)

![a3a04c3844a2d2984195b68c8a5414b2b813ceb8cc858c4756f9f268edddaa38.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/a3a04c3844a2d2984195b68c8a5414b2b813ceb8cc858c4756f9f268edddaa38.jpg)

![b71d8fddea1e296eae6f88a42bafeee97a4f17f851fc7946847fd775b6901859.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/b71d8fddea1e296eae6f88a42bafeee97a4f17f851fc7946847fd775b6901859.jpg)

![b7ad80bd9aaca917e9c4f7215f988275a089bdc5f2a0ad09059d889c8aee9224.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/b7ad80bd9aaca917e9c4f7215f988275a089bdc5f2a0ad09059d889c8aee9224.jpg)

![ba810a4c7e9c10553e7cdb47db3b76a4559c0be0d66e49e5de7c9924290432c3.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/ba810a4c7e9c10553e7cdb47db3b76a4559c0be0d66e49e5de7c9924290432c3.jpg)

![cbbafc0c8b989af2c52a6728e7dc0a32d38ce9fef59fd224085f970bc5ee9a53.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/cbbafc0c8b989af2c52a6728e7dc0a32d38ce9fef59fd224085f970bc5ee9a53.jpg)

![d9513792ec85bf4760dd5a3045fc7c2f3e7cfd548ade47d6d3cc2405a801bab3.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/d9513792ec85bf4760dd5a3045fc7c2f3e7cfd548ade47d6d3cc2405a801bab3.jpg)

![db50f142c541324fc327816b6cb64a25ad8e9db0a15399afc4ffcba3ed72dbe3.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/images/db50f142c541324fc327816b6cb64a25ad8e9db0a15399afc4ffcba3ed72dbe3.jpg)

### Tables

![124039ed1c44ea9596b731e3dfa706b7b67e657c8d61ec1bc98aaa67be09bf92.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/124039ed1c44ea9596b731e3dfa706b7b67e657c8d61ec1bc98aaa67be09bf92.jpg)

![55b6995dc4495137c7f2e7fdece9e249065f57c89c23ebc23527a414c7e7da0a.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/55b6995dc4495137c7f2e7fdece9e249065f57c89c23ebc23527a414c7e7da0a.jpg)

![69d2678bb0d8e4d9c08e8694e0f2a7b79f071a1d51a69f245898d9f2b1fc08ec.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/69d2678bb0d8e4d9c08e8694e0f2a7b79f071a1d51a69f245898d9f2b1fc08ec.jpg)

![71bf5d27d98160083d501f6435e24dc437c824b106b98489f374cdd0e13ad60c.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/71bf5d27d98160083d501f6435e24dc437c824b106b98489f374cdd0e13ad60c.jpg)

![8b834cf532d6f5eca43d503833d8f82ae7270f796c76ae74950d864db8b38aaf.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/8b834cf532d6f5eca43d503833d8f82ae7270f796c76ae74950d864db8b38aaf.jpg)

![90ea1b135a02c65969adbaba9e55f3b1aa1d083b423933f3a8c1db119f2d79c1.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/90ea1b135a02c65969adbaba9e55f3b1aa1d083b423933f3a8c1db119f2d79c1.jpg)

![9e3d232313357c4a5b775d0e9d9d43554d7f813503f885894194ae9deb10cdc8.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/9e3d232313357c4a5b775d0e9d9d43554d7f813503f885894194ae9deb10cdc8.jpg)

![a57367f1b5dd77092ef787c8a394e87302ef9d38a9bcdbad58e3da9c30d6b7e1.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/a57367f1b5dd77092ef787c8a394e87302ef9d38a9bcdbad58e3da9c30d6b7e1.jpg)

![d3a1bbc066990a54b22609e2e09648e6005f5c97433de7871e90839163476c41.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/d3a1bbc066990a54b22609e2e09648e6005f5c97433de7871e90839163476c41.jpg)

![e834e8d2665f6c75eedf9ec5cb5dc9685dd6fee013f030678ef7dab5fe6d3d77.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/e834e8d2665f6c75eedf9ec5cb5dc9685dd6fee013f030678ef7dab5fe6d3d77.jpg)

![f0d1c7e7fcb857996d1316f7c4612fb372c7d1da54e0559b792873e3d5162305.jpg](../icml_results/5885_InfVSR_%20Toward%20Consistency-Driven%20Streaming%20Generative%20Video%20Super-Resolution/tables/f0d1c7e7fcb857996d1316f7c4612fb372c7d1da54e0559b792873e3d5162305.jpg)

<a id="paper-5837"></a>

## Regime-Adaptive Bayesian Optimization via Dirichlet Process Mixtures of Gaussian Processes


### Images

![36c81f74d8118944d7d8801a707fd5b64f747192bb6ed15fa107e1a381ff56c7.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/images/36c81f74d8118944d7d8801a707fd5b64f747192bb6ed15fa107e1a381ff56c7.jpg)

![4875bdc43b859e227c1cfde0be28741f81836209b046023ff6d8278f41dcf58f.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/images/4875bdc43b859e227c1cfde0be28741f81836209b046023ff6d8278f41dcf58f.jpg)

![702509b0134fe79b10a923639400bd682480875930c1195b962caf7dd4b7a7e9.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/images/702509b0134fe79b10a923639400bd682480875930c1195b962caf7dd4b7a7e9.jpg)

![de120a986e595ba5c3594327bbc03230195e21adae7588a1e6af391661890a83.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/images/de120a986e595ba5c3594327bbc03230195e21adae7588a1e6af391661890a83.jpg)

### Tables

![6854ced14397a645adf029a8daa3c13977903ed97c4f15e6d8d21677e8e1c20f.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/6854ced14397a645adf029a8daa3c13977903ed97c4f15e6d8d21677e8e1c20f.jpg)

![6ca1d2178c35bdee6e71774bf4b40b657601036a727dee11a28966468313fd27.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/6ca1d2178c35bdee6e71774bf4b40b657601036a727dee11a28966468313fd27.jpg)

![7917320d191fa553e57fadb5a1f448c3ef41fea055c89b20d8c8b6aa7b302002.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/7917320d191fa553e57fadb5a1f448c3ef41fea055c89b20d8c8b6aa7b302002.jpg)

![abf0a4254dae480c69fe56d182c4ff1146cc6cd64a67e1c4296d1eb4ab83708d.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/abf0a4254dae480c69fe56d182c4ff1146cc6cd64a67e1c4296d1eb4ab83708d.jpg)

![b1d2d4ecadf5789a76f8a563b95b49a390323a58976fe2581291a99f5470ba43.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/b1d2d4ecadf5789a76f8a563b95b49a390323a58976fe2581291a99f5470ba43.jpg)

![c8d66f921f2fd2dda0bc88d4a941597d3cd91d2c03480e3b89a22cc1edfcae67.jpg](../icml_results/5886_Regime-Adaptive%20Bayesian%20Optimization%20via%20Dirichlet%20Process%20Mixtures%20of%20Gaussian%20Processes/tables/c8d66f921f2fd2dda0bc88d4a941597d3cd91d2c03480e3b89a22cc1edfcae67.jpg)

<a id="paper-5838"></a>

## Contrastive Flow Map Matching


### Images

![345934b1775321b5607d1c160a20e1bd4d383974833974b642bbdbe100edd416.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/images/345934b1775321b5607d1c160a20e1bd4d383974833974b642bbdbe100edd416.jpg)

![8811146cd112f5d3864c617d4d5332685ef32ce49bc034ba6108db1b259b9083.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/images/8811146cd112f5d3864c617d4d5332685ef32ce49bc034ba6108db1b259b9083.jpg)

![88c8827216012845b2cf426c7d5fea3ab333cbd5d67a5bdf2d8de45b64493432.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/images/88c8827216012845b2cf426c7d5fea3ab333cbd5d67a5bdf2d8de45b64493432.jpg)

![8d49362ed67a2f5eadcca4e2633814d910f73c042f3a4e096f0d869520772a55.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/images/8d49362ed67a2f5eadcca4e2633814d910f73c042f3a4e096f0d869520772a55.jpg)

### Tables

![0490197117117169b3fd0a179d7b31ad8c38ef6b787c5bc161b0171349437573.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/0490197117117169b3fd0a179d7b31ad8c38ef6b787c5bc161b0171349437573.jpg)

![4539c999689411fdd0de1833627a926c4048c5b85ffc0b4be4a3621ed8ec0d66.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/4539c999689411fdd0de1833627a926c4048c5b85ffc0b4be4a3621ed8ec0d66.jpg)

![72b329078ddbfe72519e79ecbb214599bf08a8c5da9d8e99256ebe66383bc976.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/72b329078ddbfe72519e79ecbb214599bf08a8c5da9d8e99256ebe66383bc976.jpg)

![88f81b2541f9b35704cd355a527a49f398e7915ef91021623d3613baacdef85e.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/88f81b2541f9b35704cd355a527a49f398e7915ef91021623d3613baacdef85e.jpg)

![cb54a031784133e95431f75a20eae48d953fc8c02cfe4828663761c23300c472.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/cb54a031784133e95431f75a20eae48d953fc8c02cfe4828663761c23300c472.jpg)

![de6b3637be75614c00ddf4b3d83769c4c64795e7086b200a2bae6b16ae7cac6a.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/de6b3637be75614c00ddf4b3d83769c4c64795e7086b200a2bae6b16ae7cac6a.jpg)

![e6f645bd0372d0736818d0a0889dfea64f319bac5177aa4f2b195c5b7c97c39c.jpg](../icml_results/5887_Contrastive%20Flow%20Map%20Matching/tables/e6f645bd0372d0736818d0a0889dfea64f319bac5177aa4f2b195c5b7c97c39c.jpg)

<a id="paper-5839"></a>

## Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training


### Images

![06dea4e5c31b04c50cf45c17e869501177b9dec7d216f42a57e9a022217b5c95.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/06dea4e5c31b04c50cf45c17e869501177b9dec7d216f42a57e9a022217b5c95.jpg)

![0e3f8255c062de5154013a4f168792759567183d4f43bd16682d76caaeb266b4.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/0e3f8255c062de5154013a4f168792759567183d4f43bd16682d76caaeb266b4.jpg)

![179888ad0bafea6b7f61de4d9b4d710f210813cf766fe179b532ba914dc56a53.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/179888ad0bafea6b7f61de4d9b4d710f210813cf766fe179b532ba914dc56a53.jpg)

![25175f4f5f2c5565e4b5ef38e4a5acc1732aac17ff5c79097d471716265e4f64.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/25175f4f5f2c5565e4b5ef38e4a5acc1732aac17ff5c79097d471716265e4f64.jpg)

![27221826ef349a12ad29ea6a13096a81e4abe762a9c4b763699aadd0c900efb4.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/27221826ef349a12ad29ea6a13096a81e4abe762a9c4b763699aadd0c900efb4.jpg)

![3c91d9e3d5e1267f36f7a050aa4696369eb48b63eb441b9b238350aaa7ff5882.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/3c91d9e3d5e1267f36f7a050aa4696369eb48b63eb441b9b238350aaa7ff5882.jpg)

![462f8cdc81db6f7305826a118da6ef965cd9e4bb939cce635f36407cd0e3f943.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/462f8cdc81db6f7305826a118da6ef965cd9e4bb939cce635f36407cd0e3f943.jpg)

![7d0438a322e86d31eeaaf3b09806f3ebd7b17aa8dda8337a3e83fa02f60cf15c.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/7d0438a322e86d31eeaaf3b09806f3ebd7b17aa8dda8337a3e83fa02f60cf15c.jpg)

![8752be4791caced90f7acf2aae591c23a7204c05ae6e53097fec1e42f5378993.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/8752be4791caced90f7acf2aae591c23a7204c05ae6e53097fec1e42f5378993.jpg)

![8c6e55c72fab7d111aa2e69fd9b23a72a0e763acac83466af8155a5f826edb7f.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/8c6e55c72fab7d111aa2e69fd9b23a72a0e763acac83466af8155a5f826edb7f.jpg)

![a73f77478debf8c9fec2d526b2ed3ace9489328d0fc09d9cd8bf22aa1d5b607b.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/a73f77478debf8c9fec2d526b2ed3ace9489328d0fc09d9cd8bf22aa1d5b607b.jpg)

![afc8ac4aca6324a148f11653b9d516d4e7b8ad90cce8374d8b7618fcb604b6c0.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/afc8ac4aca6324a148f11653b9d516d4e7b8ad90cce8374d8b7618fcb604b6c0.jpg)

![b38c1294680e0c141a53d107413a733d8a40cab7cfa223e9b44c804f2285a027.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/b38c1294680e0c141a53d107413a733d8a40cab7cfa223e9b44c804f2285a027.jpg)

![c997173ae00c4e379df60d432e36a213fd4f850b83a5be1333634047270c9f61.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/c997173ae00c4e379df60d432e36a213fd4f850b83a5be1333634047270c9f61.jpg)

![cb03983313984c33d4e3cf3b13954ecb0cc1a4e3169c01f61119f62d14ef4418.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/cb03983313984c33d4e3cf3b13954ecb0cc1a4e3169c01f61119f62d14ef4418.jpg)

![cf1b45bb21be763d80cefb61ea7ea60ab06a3fd96ca62cdcb6b14fc253a69d31.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/cf1b45bb21be763d80cefb61ea7ea60ab06a3fd96ca62cdcb6b14fc253a69d31.jpg)

![d195c5dd01f7aad979033c9e91f611ff20133a2fda92486c026b154a15365045.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/d195c5dd01f7aad979033c9e91f611ff20133a2fda92486c026b154a15365045.jpg)

![d593d421a3f489c64cc64a64b9d20257e713e2fc531d645b6a4a488ca243e19c.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/d593d421a3f489c64cc64a64b9d20257e713e2fc531d645b6a4a488ca243e19c.jpg)

![d8713dc74c5334afc3c271722b47c1fc347aaf5a894ab3d761002694d40a3977.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/d8713dc74c5334afc3c271722b47c1fc347aaf5a894ab3d761002694d40a3977.jpg)

![e2b731da95757b2146c1f7ce333ccc37fafd14f8216c83172ab5199097d05500.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/e2b731da95757b2146c1f7ce333ccc37fafd14f8216c83172ab5199097d05500.jpg)

![e3bd78ec56ffaa6bf95e1b988910376b92cc17f35e9ccbdaec2f082b822b7bcf.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/e3bd78ec56ffaa6bf95e1b988910376b92cc17f35e9ccbdaec2f082b822b7bcf.jpg)

![e8b8d946d276984e30752c3940c608319c23be4326e15aeeb46aa3c51f864fc3.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/e8b8d946d276984e30752c3940c608319c23be4326e15aeeb46aa3c51f864fc3.jpg)

![eada36ba409669d0120c54ed6e4820aa793a7e19f482a97cbef00f15f7296e11.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/eada36ba409669d0120c54ed6e4820aa793a7e19f482a97cbef00f15f7296e11.jpg)

![ef724d778b217ec055cdedf9dd91537ba932754580eba641e12dcb8c2306d068.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/images/ef724d778b217ec055cdedf9dd91537ba932754580eba641e12dcb8c2306d068.jpg)

### Tables

![13a9cfb3c67e8f12fb7dbd2fa5bdc2a39951ac5bd3a7bbb9edb590a8a8870439.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/13a9cfb3c67e8f12fb7dbd2fa5bdc2a39951ac5bd3a7bbb9edb590a8a8870439.jpg)

![255654b44b7db38467c9cb03e38ee13a238d1e08d1f06928d4f89ca0324f0c46.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/255654b44b7db38467c9cb03e38ee13a238d1e08d1f06928d4f89ca0324f0c46.jpg)

![75ccc0fc7fd5ac325011fd9954bc293850cdb200d2a098fa3682a45662f73e1e.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/75ccc0fc7fd5ac325011fd9954bc293850cdb200d2a098fa3682a45662f73e1e.jpg)

![94da0481ec64623db143ce2e803d211266bc0db5c334b563587728b026e7d619.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/94da0481ec64623db143ce2e803d211266bc0db5c334b563587728b026e7d619.jpg)

![a11cdab39d264ba5eb798527165c4cde4915abea9038014150e255bc45a4855e.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/a11cdab39d264ba5eb798527165c4cde4915abea9038014150e255bc45a4855e.jpg)

![ac4e56b48cfa59b2ee71f45d3c35e13b20bec1556ebbb3f98e2f3eba2c93514f.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/ac4e56b48cfa59b2ee71f45d3c35e13b20bec1556ebbb3f98e2f3eba2c93514f.jpg)

![d67cfcf11db2d70af0fc232409a1649b7037348dc04a74c0866789c040e08368.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/d67cfcf11db2d70af0fc232409a1649b7037348dc04a74c0866789c040e08368.jpg)

![f9ce4f767cffd43d3398dc5efd72290404dcb57124109c5754f93184479f074a.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/f9ce4f767cffd43d3398dc5efd72290404dcb57124109c5754f93184479f074a.jpg)

![fe5e03261f5443baf85da9a2537920b84f6783761a4dc98155f02e53f47f5aa4.jpg](../icml_results/5888_Towards%20On-Policy%20SFT_%20Distribution%20Discriminant%20Theory%20and%20its%20Applications%20in%20LLM%20Training/tables/fe5e03261f5443baf85da9a2537920b84f6783761a4dc98155f02e53f47f5aa4.jpg)

<a id="paper-5840"></a>

## Adaptive Visual Autoregressive Acceleration via Dual-Linkage Entropy Analysis


### Images

![617b5eed6a4c9c292a67ff6c872dcbefb23ca96dda05fb4b52e05ee38e69ee43.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/617b5eed6a4c9c292a67ff6c872dcbefb23ca96dda05fb4b52e05ee38e69ee43.jpg)

![6e74746fba53b5605083168b62b9b2abc4df228290edceb574872762d11cd8e7.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/6e74746fba53b5605083168b62b9b2abc4df228290edceb574872762d11cd8e7.jpg)

![75f6e50084dc7522732bdf7eb41d5b86d66a6fa9124a3e08d33b78bf31bee30b.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/75f6e50084dc7522732bdf7eb41d5b86d66a6fa9124a3e08d33b78bf31bee30b.jpg)

![856ce9dcdfb1073f394b3d6f7b0972eb0380da95fca48ad05a9a048cffd2e996.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/856ce9dcdfb1073f394b3d6f7b0972eb0380da95fca48ad05a9a048cffd2e996.jpg)

![965c8f1b4325f54dca39c60db6953fe1bee76480135ce445f95c031aff7e5a1e.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/965c8f1b4325f54dca39c60db6953fe1bee76480135ce445f95c031aff7e5a1e.jpg)

![aa3284d39b69e53b5b22e2b322d446b039553f2f93043eb78a47b40ffe52c6f3.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/aa3284d39b69e53b5b22e2b322d446b039553f2f93043eb78a47b40ffe52c6f3.jpg)

![b91bd7b51ac6404bf10e54cc96b5f39a202a23bdc38ac94375f918cc097d898d.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/b91bd7b51ac6404bf10e54cc96b5f39a202a23bdc38ac94375f918cc097d898d.jpg)

![c6b12cbc00d78a407328feb65bf8005e077158d2d592ef15b3f366dab024503b.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/c6b12cbc00d78a407328feb65bf8005e077158d2d592ef15b3f366dab024503b.jpg)

![c854e8f2fbef0a6fd6e1f6d40e11666acd401bedcab1225c8a3e229d74c0a90a.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/c854e8f2fbef0a6fd6e1f6d40e11666acd401bedcab1225c8a3e229d74c0a90a.jpg)

![e173555155eed045719c0c433da44c2ce8867cf1cafb5b87aba4e54ae9693af7.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/e173555155eed045719c0c433da44c2ce8867cf1cafb5b87aba4e54ae9693af7.jpg)

![e1b5b1c81ec15d42c7f5cc8ddd322d8a0642c4a11883d38d19b02840eda96b2e.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/e1b5b1c81ec15d42c7f5cc8ddd322d8a0642c4a11883d38d19b02840eda96b2e.jpg)

![e2cf0804f6868fb4339935932660727430741d32dbad827ffa89c79fe6ac2986.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/e2cf0804f6868fb4339935932660727430741d32dbad827ffa89c79fe6ac2986.jpg)

![e7675f225be264619c7c4453d2c96de01a9aa1176e2a6203c83749b39990c5b6.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/e7675f225be264619c7c4453d2c96de01a9aa1176e2a6203c83749b39990c5b6.jpg)

![ed608fd3fb77b60da0c89b290bd9599cc757a3a26ad228859e75be263359c4e3.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/images/ed608fd3fb77b60da0c89b290bd9599cc757a3a26ad228859e75be263359c4e3.jpg)

### Tables

![0b876b06010ec05c3443a80bb7c76978347dbfbbe30f536cb8d48e4bc90458ba.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/0b876b06010ec05c3443a80bb7c76978347dbfbbe30f536cb8d48e4bc90458ba.jpg)

![20ead9ec44ab4fc6721dff0eeb482177aa978c20e6d07acb2cab336e2fae5f9d.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/20ead9ec44ab4fc6721dff0eeb482177aa978c20e6d07acb2cab336e2fae5f9d.jpg)

![303e510e9fad06b1374a3b45a33636930bc379f2a2f3c09aceb139f7ba274324.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/303e510e9fad06b1374a3b45a33636930bc379f2a2f3c09aceb139f7ba274324.jpg)

![4ca4b340a8ed8047e87b80b4a7a6ffd4ac1e04bc77be0e0f849dcf8347ef2be7.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/4ca4b340a8ed8047e87b80b4a7a6ffd4ac1e04bc77be0e0f849dcf8347ef2be7.jpg)

![682731dade083b3ad89b68b111a853651bb019e2ca148e742e436a12e9ee26b0.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/682731dade083b3ad89b68b111a853651bb019e2ca148e742e436a12e9ee26b0.jpg)

![97b2d5adfe6b94f67f3e966eeac89b9a8d4df1d8c5170ae7a99a57d574c234d9.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/97b2d5adfe6b94f67f3e966eeac89b9a8d4df1d8c5170ae7a99a57d574c234d9.jpg)

![9d4713823e1d5b3ea32357912592d36ed3a1e6431fbad6075502a13bb09fdfbc.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/9d4713823e1d5b3ea32357912592d36ed3a1e6431fbad6075502a13bb09fdfbc.jpg)

![abfecad3de8b3aed1121fd14d221a850e5e3587edba8a153fdf2269f2ff0e73b.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/abfecad3de8b3aed1121fd14d221a850e5e3587edba8a153fdf2269f2ff0e73b.jpg)

![b462834174a23960518c1f18a26935a5e4d25a96b3c9ddd278b7503e9f7a6ac6.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/b462834174a23960518c1f18a26935a5e4d25a96b3c9ddd278b7503e9f7a6ac6.jpg)

![bf6d2c8d57071395205109411f75eca3c27ee198f4a348fd2b9a90980c9d1f0b.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/bf6d2c8d57071395205109411f75eca3c27ee198f4a348fd2b9a90980c9d1f0b.jpg)

![c3c8f8b41d5d0ea18d4638b8af99a53a83440323f88c0468972a83c539bafb15.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/c3c8f8b41d5d0ea18d4638b8af99a53a83440323f88c0468972a83c539bafb15.jpg)

![cea2087366c1af2245177eac5fda2f366ddeda11c28623a6a01b49ea7fe2b076.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/cea2087366c1af2245177eac5fda2f366ddeda11c28623a6a01b49ea7fe2b076.jpg)

![dfeef7f1f8368b07ccce4cd730ded21878db3eae4665792db5d3cc2ca5310f48.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/dfeef7f1f8368b07ccce4cd730ded21878db3eae4665792db5d3cc2ca5310f48.jpg)

![e6a71ae5ee491d13ec35f2df11bb49a27aea891f257958365b1726b7aafbc031.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/e6a71ae5ee491d13ec35f2df11bb49a27aea891f257958365b1726b7aafbc031.jpg)

![efcf7ff6ad0ccfd7ea190cdb28d26f3bc9e10ba3804c7d4d80b007a2f173e5f2.jpg](../icml_results/5889_Adaptive%20Visual%20Autoregressive%20Acceleration%20via%20Dual-Linkage%20Entropy%20Analysis/tables/efcf7ff6ad0ccfd7ea190cdb28d26f3bc9e10ba3804c7d4d80b007a2f173e5f2.jpg)

<a id="paper-5841"></a>

## Multimodal Meta-Verifier with Explicit Structured Recalibration

### Images

![1104e6cbe168a16d0f72d73c845db71ed607917600da66a5854b0a1cf92c7a89.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/images/1104e6cbe168a16d0f72d73c845db71ed607917600da66a5854b0a1cf92c7a89.jpg)

![2edd2bd08fec4d790150d728af026aa11575471f16c9152c531f92003c1bcc4a.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/images/2edd2bd08fec4d790150d728af026aa11575471f16c9152c531f92003c1bcc4a.jpg)

![31ce50ea507ed1796e27d838b0b28dee92e2dd89b4d2919f5de1585989e2355a.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/images/31ce50ea507ed1796e27d838b0b28dee92e2dd89b4d2919f5de1585989e2355a.jpg)

![4c6c222bc5bd1b70b20e7114bf5ca09609fd43d0287e395f1e837423c54335c5.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/images/4c6c222bc5bd1b70b20e7114bf5ca09609fd43d0287e395f1e837423c54335c5.jpg)

![7b879bc76ab5bff2590f7376c3f92070b93b0cd432772da48bf854eed5cf1ad5.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/images/7b879bc76ab5bff2590f7376c3f92070b93b0cd432772da48bf854eed5cf1ad5.jpg)

### Tables

![8488391b63e3065bebac186cde0485e6d4a668fff5f729dd8fc725966124380d.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/8488391b63e3065bebac186cde0485e6d4a668fff5f729dd8fc725966124380d.jpg)

![8c72ed18f3e602a17615d1a3859a1d58e6a5a4f39e6add27f667172949691a90.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/8c72ed18f3e602a17615d1a3859a1d58e6a5a4f39e6add27f667172949691a90.jpg)

![9538c41ac92a6122cb614b80117249ababa81ca3297edc21deaab2fd03dffc7d.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/9538c41ac92a6122cb614b80117249ababa81ca3297edc21deaab2fd03dffc7d.jpg)

![cd6d8a9ff0d0b1f6e8408f507623e13eaeae8903a4976aa8b05863d8addc4603.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/cd6d8a9ff0d0b1f6e8408f507623e13eaeae8903a4976aa8b05863d8addc4603.jpg)

![f083c6ff93e38c85d833e924f0e340539450dea75cda7d45666ba6d71ee58110.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/f083c6ff93e38c85d833e924f0e340539450dea75cda7d45666ba6d71ee58110.jpg)

![f19d0d075dc13f29e33b3efaf88e808cf75293a4a55f53c9d7207cfe49098a2c.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/f19d0d075dc13f29e33b3efaf88e808cf75293a4a55f53c9d7207cfe49098a2c.jpg)

![fa5e2b7e97e48905d7df15723c16dd4f8b39b951659706ed099e853fb10fc686.jpg](../icml_results/5890_Multimodal%20Meta-Verifier%20with%20Explicit%20Structured%20Recalibration/tables/fa5e2b7e97e48905d7df15723c16dd4f8b39b951659706ed099e853fb10fc686.jpg)
