# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 57 of 197

## 目录 (Table of Contents)

1. [Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching](#paper-1849)
2. [Sampling from Your Language Model One Byte at a Time](#paper-1850)
3. [Strategic Candidacy in Generative AI Arenas](#paper-1851)
4. [OPIC: Enhancing Language Model Merging via Optimizing In-Context Capability](#paper-1852)
5. [Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning](#paper-1853)
6. [VidLaDA: Bidirectional Diffusion Large Language Models for Efficient Video Understanding](#paper-1854)
7. [PGS: Effective LLM Code Refinement via Property-Oriented and Structurally Minimal Feedback](#paper-1855)
8. [DR$^2$Seg: Decomposed Two-Stage Rollouts for Efficient Reasoning Segmentation in Multimodal Large Language Models](#paper-1856)
9. [Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](#paper-1857)
10. [Self-Prophetic Decoding to Unlock Visual Search in LVLMs](#paper-1858)
11. [Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](#paper-1859)
12. [Co-RedTeam: Orchestrated Security Discovery and Exploitation with LLM Agents](#paper-1860)
13. [Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization](#paper-1861)
14. [Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model](#paper-1862)
15. [Structure-Centric Graph Foundation Model via Geometric Bases](#paper-1863)
16. [Navigating the Flatlands: Dual Adaptive Sharpness-Aware Minimization for Domain Generalization](#paper-1864)
17. [Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration](#paper-1865)
18. [The Accumulation of Score Estimation Error in Diffusion Models](#paper-1866)
19. [Efficient Multi-round LLM Inference over Disaggregated Serving](#paper-1867)
20. [DSGCR: Decomposed Spectral Geometry-Aware Cross-Modal Semantic Representation for 3D Visual Grounding](#paper-1868)
21. [Approximation Error Upper and Lower Bounds for Hölder Class with Transformers](#paper-1869)
22. [Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games](#paper-1870)
23. [HyMTRL: A Hybrid Multi-Task Reinforcement Learning Framework via Phased Policy Evolution](#paper-1871)
24. [Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](#paper-1872)
25. [Can LLMs Reason Structurally? Benchmarking via the lens of Data Structures](#paper-1873)
26. [SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?](#paper-1874)
27. [Doubly Regularized Markov Decision Processes for Robust Reinforcement Learning](#paper-1875)
28. [Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](#paper-1876)
29. [DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models](#paper-1877)
30. [Newton-coupled Dual-Teacher Semi-supervised Learning Framework](#paper-1878)
31. [Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation](#paper-1879)
32. [GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks](#paper-1880)
33. [Demystifying When Pruning Works via Representation Hierarchies](#paper-1881)

---


<a id="paper-1849"></a>

## Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching

### Images

![17c68385f76551f76f5cc90fe477bd47f78830a3197ddd07387a9fe313674d05.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/17c68385f76551f76f5cc90fe477bd47f78830a3197ddd07387a9fe313674d05.jpg)

![40864fa9938a78ff0677fbe2ea56dea0098784fd6da2895d40a4d33b3d237694.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/40864fa9938a78ff0677fbe2ea56dea0098784fd6da2895d40a4d33b3d237694.jpg)

![4b5dd2e97f577c222118619f6764380ab012e8667f55191f14447acab0c5c6c2.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/4b5dd2e97f577c222118619f6764380ab012e8667f55191f14447acab0c5c6c2.jpg)

![54783254b4ab669055fe46b7214c30e592603f643b1ffec1ab0c05894aa5f8d1.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/54783254b4ab669055fe46b7214c30e592603f643b1ffec1ab0c05894aa5f8d1.jpg)

![5e8e724f21da8325265c60b33ec317e6e7bc8d437cbf955ad47325a1ab336650.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/5e8e724f21da8325265c60b33ec317e6e7bc8d437cbf955ad47325a1ab336650.jpg)

![69a30e5cddf185d0a9f79153a0cd8206c039394661ccb5d433f39ba09575141d.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/69a30e5cddf185d0a9f79153a0cd8206c039394661ccb5d433f39ba09575141d.jpg)

![6cdcd659e4119d8d7eddb8b10f694963649843e4e02b2388c4f5fb4cf48827c2.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/6cdcd659e4119d8d7eddb8b10f694963649843e4e02b2388c4f5fb4cf48827c2.jpg)

![6e7c01fb1656bea5504d032e8d65e862c5952a28dce01b23035b634c5024521f.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/6e7c01fb1656bea5504d032e8d65e862c5952a28dce01b23035b634c5024521f.jpg)

![8bbc0ba574bb23305354ab9d754a85504a81f64e48ff65aae228f0c1bfc7c151.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/8bbc0ba574bb23305354ab9d754a85504a81f64e48ff65aae228f0c1bfc7c151.jpg)

![9a3fd02e83a13500c0088bdd6df299c3ead564a4832ffb92290c58c66f28f3cb.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/9a3fd02e83a13500c0088bdd6df299c3ead564a4832ffb92290c58c66f28f3cb.jpg)

![cecf23b390bbfec2c9607ee5f5bf3bce1dd9fea2767af479a4c93aa6da5c3b27.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/cecf23b390bbfec2c9607ee5f5bf3bce1dd9fea2767af479a4c93aa6da5c3b27.jpg)

![d43f3af556841c224775c6f9fc5911a81e39bdcc9f0d00be717104182b49e03f.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/d43f3af556841c224775c6f9fc5911a81e39bdcc9f0d00be717104182b49e03f.jpg)

![fb181dd98d893bf672f3b15b53d490ede1e1c9783bfc8e1b7131d522bf3d64d6.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/fb181dd98d893bf672f3b15b53d490ede1e1c9783bfc8e1b7131d522bf3d64d6.jpg)

![fe399aa0f7b9ca729fb69d2efab576518e24bd8e8f8ceddd9c455525f503edda.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/images/fe399aa0f7b9ca729fb69d2efab576518e24bd8e8f8ceddd9c455525f503edda.jpg)

### Tables

![168abc0a7264db6bb0992788cf05b3cd61175c98fa03760df44301ef0b3f69fa.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/tables/168abc0a7264db6bb0992788cf05b3cd61175c98fa03760df44301ef0b3f69fa.jpg)

![2e801761c97ecfe7077babf52b2b464b62b2caa2b7d870c2ca9fc678114c0822.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/tables/2e801761c97ecfe7077babf52b2b464b62b2caa2b7d870c2ca9fc678114c0822.jpg)

![51dee502e1dfe90d04039cf467e64e21a8c26c6d86a0dd6bc03c27c8d28aaf79.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/tables/51dee502e1dfe90d04039cf467e64e21a8c26c6d86a0dd6bc03c27c8d28aaf79.jpg)

![9d351989777ff421a3b2d47e7f68914c37ac535824d837b58fe8ea07f00252d4.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/tables/9d351989777ff421a3b2d47e7f68914c37ac535824d837b58fe8ea07f00252d4.jpg)

![a63c2c5254a00603480f4f23e29fd88c99c761f4526a8dc95b3f477746dbe6f5.jpg](../icml_results/1873_The%20Heterogeneous%20Safety%20Impacts%20of%20Benign%20Multilingual%20Fine-Tuning/tables/a63c2c5254a00603480f4f23e29fd88c99c761f4526a8dc95b3f477746dbe6f5.jpg)

## Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching


### Tables

![069f995c60d5309a7a7e5b96db116125aadfbe75b9678c33c4bc2310d2465da6.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/069f995c60d5309a7a7e5b96db116125aadfbe75b9678c33c4bc2310d2465da6.jpg)

![179298219704ed0b6b104b82fabb716fbfbd4e144ed3679fcc7d83e678463f47.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/179298219704ed0b6b104b82fabb716fbfbd4e144ed3679fcc7d83e678463f47.jpg)

![1ecb3cecde1f795920bdf27a77fc3aa375a82c08862721bb23eec3cb8a71dc3f.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/1ecb3cecde1f795920bdf27a77fc3aa375a82c08862721bb23eec3cb8a71dc3f.jpg)

![22b0e11c2e6b8b8c3f7f1361e85006d3a123690f68635f56f9233db533d0857c.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/22b0e11c2e6b8b8c3f7f1361e85006d3a123690f68635f56f9233db533d0857c.jpg)

![25a28ed2bb3b716f845c221e806e9651a5681c864c514342741d5df385f33646.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/25a28ed2bb3b716f845c221e806e9651a5681c864c514342741d5df385f33646.jpg)

![2da5b7a5588b88216c55378a02cebdb1298e5c25d1573988130e33ecc7cc213a.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/2da5b7a5588b88216c55378a02cebdb1298e5c25d1573988130e33ecc7cc213a.jpg)

![336dcff23b244fa17aeb086f17b2b407f15e6765cf1847e4aed51510ff0bd6dd.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/336dcff23b244fa17aeb086f17b2b407f15e6765cf1847e4aed51510ff0bd6dd.jpg)

![3e77a5e7265ea465f01b98bd2e03e94be167f72105152281731069514f08ab96.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/3e77a5e7265ea465f01b98bd2e03e94be167f72105152281731069514f08ab96.jpg)

![4a7e1d3ed35e3b102d296c24017c1e4e9dd66ba7eb616d8807027d55775b128e.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/4a7e1d3ed35e3b102d296c24017c1e4e9dd66ba7eb616d8807027d55775b128e.jpg)

![5219abb27b638c41563ed60583d340c6d459bceaea126baa933a780a8075fe68.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/5219abb27b638c41563ed60583d340c6d459bceaea126baa933a780a8075fe68.jpg)

![53f4d3d366047272e97fe56fed1f515eafc3eb37f070afa464a5b309b06bd453.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/53f4d3d366047272e97fe56fed1f515eafc3eb37f070afa464a5b309b06bd453.jpg)

![6fae7826f96e375a5cb5935c3680ef8280eef2d0581cd3e01f62185757b19090.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/6fae7826f96e375a5cb5935c3680ef8280eef2d0581cd3e01f62185757b19090.jpg)

![727c214265578a6389c77e348e60e44fc076d08fafcd2dc00d303ce91767b8b3.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/727c214265578a6389c77e348e60e44fc076d08fafcd2dc00d303ce91767b8b3.jpg)

![81fea51f790d73027c69aa134363395a73f1e2997c33bf083f261df2f9a1555e.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/81fea51f790d73027c69aa134363395a73f1e2997c33bf083f261df2f9a1555e.jpg)

![8dedaf84aa21e7fd58b833d8afa79f5eb56e9304c636f8fa13367b532b03627a.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/8dedaf84aa21e7fd58b833d8afa79f5eb56e9304c636f8fa13367b532b03627a.jpg)

![90690d21768fb5b142b097d773413bdcb34dda54b9ae5f9712a03469bf0d78f7.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/90690d21768fb5b142b097d773413bdcb34dda54b9ae5f9712a03469bf0d78f7.jpg)

![a8089f799334eb37ea32f0ae57f5aed080370efcc4337fd9ccfa41213b1d27fe.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/a8089f799334eb37ea32f0ae57f5aed080370efcc4337fd9ccfa41213b1d27fe.jpg)

![addb400017afdbf16984497d5ad6347126af2adb56cc8059a4d64c9a83602eba.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/addb400017afdbf16984497d5ad6347126af2adb56cc8059a4d64c9a83602eba.jpg)

![b9852bc4311f6229e2bf2af935dfd544f9364a00aaaaf24334c97b0166c2fee3.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/b9852bc4311f6229e2bf2af935dfd544f9364a00aaaaf24334c97b0166c2fee3.jpg)

![b9a2bcff79497aa38bb78a5aeac7e3987bae55a5f974198bf46938c4e406ce7b.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/b9a2bcff79497aa38bb78a5aeac7e3987bae55a5f974198bf46938c4e406ce7b.jpg)

![b9e8642b748f32d8a44d42224886ce57c3347286cb9ffb36275bafbd0ab7d6ef.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/b9e8642b748f32d8a44d42224886ce57c3347286cb9ffb36275bafbd0ab7d6ef.jpg)

![bf06f210246bf050b5039e8bbfba4179e39fc7e5774392fa2d332d7ead288f1c.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/bf06f210246bf050b5039e8bbfba4179e39fc7e5774392fa2d332d7ead288f1c.jpg)

![c50041d4a524bd5182982f6dd69301a1b828e03414525c0c69bffced8d2048f6.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/c50041d4a524bd5182982f6dd69301a1b828e03414525c0c69bffced8d2048f6.jpg)

![cc4f702c5b8656c2e1ba69f47863849d40b92ee18dba2d6aa81d6777283d73db.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/cc4f702c5b8656c2e1ba69f47863849d40b92ee18dba2d6aa81d6777283d73db.jpg)

![d20c55443da4130f539caa5fe3d8cccd4cd322c43acc69e5d3fc3409d2717314.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/d20c55443da4130f539caa5fe3d8cccd4cd322c43acc69e5d3fc3409d2717314.jpg)

![de130230ff3c9c21effbba8ee11c7383e61ea7ec94a51c53d94b92c3726fad39.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/de130230ff3c9c21effbba8ee11c7383e61ea7ec94a51c53d94b92c3726fad39.jpg)

![e744592311f7140be69ce1105625aa2af28bd3a9da091ea18658d431a9c99ed7.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/e744592311f7140be69ce1105625aa2af28bd3a9da091ea18658d431a9c99ed7.jpg)

![f355235425df1ea801593a55c65898a217260ff4ddf6e65b4d49e87c1b6d618a.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/f355235425df1ea801593a55c65898a217260ff4ddf6e65b4d49e87c1b6d618a.jpg)

![f747c9cc6e81b06a470aa8427be46a640de86e603c1a7067294c039c381d5530.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/f747c9cc6e81b06a470aa8427be46a640de86e603c1a7067294c039c381d5530.jpg)

![f7c8c479bf316ff406d5c3dbc261840c1bb192b77e0bf4a6ff0e396cd49ede44.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/f7c8c479bf316ff406d5c3dbc261840c1bb192b77e0bf4a6ff0e396cd49ede44.jpg)

![f8c689b3f99173019a0116f6a4a8788373c0ef394d16409b8fca6ed611ec5b03.jpg](../icml_results/1874_Minibatch%20Optimal%20Transport%20and%20Perplexity%20Bound%20Estimation%20in%20Discrete%20Flow%20Matching/tables/f8c689b3f99173019a0116f6a4a8788373c0ef394d16409b8fca6ed611ec5b03.jpg)

<a id="paper-1850"></a>

## Sampling from Your Language Model One Byte at a Time


### Images

![088dc4f3683f8c2e778680c199650ca078d3f68bb2d4918f027a6e44b6a3f856.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/088dc4f3683f8c2e778680c199650ca078d3f68bb2d4918f027a6e44b6a3f856.jpg)

![421c97c855d67a355aff91d2fe01ab950795f76f783d70d334c777257424ac75.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/421c97c855d67a355aff91d2fe01ab950795f76f783d70d334c777257424ac75.jpg)

![51c4f4b503fad4acbbdeb447173c2a67e51316a94ed65cf95ea4f91867bd9bda.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/51c4f4b503fad4acbbdeb447173c2a67e51316a94ed65cf95ea4f91867bd9bda.jpg)

![5fa86d31d45f65c50739be472d546b06afc1aa6b1ed4b8176337fbbfd6331fb2.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/5fa86d31d45f65c50739be472d546b06afc1aa6b1ed4b8176337fbbfd6331fb2.jpg)

![9078f14f68d3776a9459f3b721f7b79ff02b531da76134ee086fc87f98d83774.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/9078f14f68d3776a9459f3b721f7b79ff02b531da76134ee086fc87f98d83774.jpg)

![9af54a27f88e7fc4bcd06dcdd93f5806a133d7fbc6b84fc68ab13537756d9d0a.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/9af54a27f88e7fc4bcd06dcdd93f5806a133d7fbc6b84fc68ab13537756d9d0a.jpg)

![ece7ed9c32d0ebc2126cafd0b9596f8bc6cbfdb2661e107cdda4d0cedfe2aa6a.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/ece7ed9c32d0ebc2126cafd0b9596f8bc6cbfdb2661e107cdda4d0cedfe2aa6a.jpg)

![f268d647d6d3865dda4bfeb40a099bea6576821802c824135ce1711b1b349037.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/f268d647d6d3865dda4bfeb40a099bea6576821802c824135ce1711b1b349037.jpg)

![f4606998d3751c0db2abd20c4d9e6dd71b881c500ef871ff3042dd8054199ecb.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/images/f4606998d3751c0db2abd20c4d9e6dd71b881c500ef871ff3042dd8054199ecb.jpg)

### Tables

![296da8878bdd2b3f7fa451b648d65f69e5e7b19835063ab4666597f235c8fc63.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/296da8878bdd2b3f7fa451b648d65f69e5e7b19835063ab4666597f235c8fc63.jpg)

![43c39bf218daf8617a23ed241b57258b8fc7ae56205b984b68beeb5e93390220.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/43c39bf218daf8617a23ed241b57258b8fc7ae56205b984b68beeb5e93390220.jpg)

![48346e5aef886c6bee3654617069a1a572bf43ee4a1d6680bedfc99b62c9c861.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/48346e5aef886c6bee3654617069a1a572bf43ee4a1d6680bedfc99b62c9c861.jpg)

![4b36356e92324f447a95c5d3a2c9da5316bc7ef3928193e2ecd60ce2f41ec8ab.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/4b36356e92324f447a95c5d3a2c9da5316bc7ef3928193e2ecd60ce2f41ec8ab.jpg)

![5c3adce0b147e4c653058adddefe0e77967bef61ae6e2ebd9016691bfe001057.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/5c3adce0b147e4c653058adddefe0e77967bef61ae6e2ebd9016691bfe001057.jpg)

![5cbae4025767afe99b429ab79ce9db60443dc14ab41c9b26a934385f848a9f26.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/5cbae4025767afe99b429ab79ce9db60443dc14ab41c9b26a934385f848a9f26.jpg)

![6f72a565d9b8aedf1f5e390eb963dbcf71ec8929df54f70556e16021799681bd.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/6f72a565d9b8aedf1f5e390eb963dbcf71ec8929df54f70556e16021799681bd.jpg)

![865979734e896a86b3c8cd5227fe205d5bc71552f77a387819dcdecf6412af3d.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/865979734e896a86b3c8cd5227fe205d5bc71552f77a387819dcdecf6412af3d.jpg)

![923e9db29aed002a1ae6e08fef9e913a29bb948e43f8e08db6522eecdcb926ae.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/923e9db29aed002a1ae6e08fef9e913a29bb948e43f8e08db6522eecdcb926ae.jpg)

![a3e9844a072ec3a4a8c58536493e0b3b87dfff3aeef04eaa0fdf12822443bcfe.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/a3e9844a072ec3a4a8c58536493e0b3b87dfff3aeef04eaa0fdf12822443bcfe.jpg)

![a5a06f6788d6ae6193dbd314e86235a724d59c910841c3154ceebd1517760cdc.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/a5a06f6788d6ae6193dbd314e86235a724d59c910841c3154ceebd1517760cdc.jpg)

![a5c48291cfa41770b4cd83565981f0d448cebdb22c09d92146b62606af2caee9.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/a5c48291cfa41770b4cd83565981f0d448cebdb22c09d92146b62606af2caee9.jpg)

![cdc1ef0cfd20a39a6fb5e93db002951ea0411c73930cbb32c4b299ee50c271db.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/cdc1ef0cfd20a39a6fb5e93db002951ea0411c73930cbb32c4b299ee50c271db.jpg)

![eeda2b65a191733abf383c22082500bac00f90d4d6eec119184ba59463e1f032.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/eeda2b65a191733abf383c22082500bac00f90d4d6eec119184ba59463e1f032.jpg)

![f82ad8c725a7ea4b73c8634b4e24f04050c4d2775c9f03592a7c54e51d71225a.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/f82ad8c725a7ea4b73c8634b4e24f04050c4d2775c9f03592a7c54e51d71225a.jpg)

![f841dcd740e5724687149e6ca6d96bae21b4dd4d417e63b8fca2b919432eb2d9.jpg](../icml_results/1875_Sampling%20from%20Your%20Language%20Model%20One%20Byte%20at%20a%20Time/tables/f841dcd740e5724687149e6ca6d96bae21b4dd4d417e63b8fca2b919432eb2d9.jpg)

<a id="paper-1851"></a>

## Strategic Candidacy in Generative AI Arenas


### Images

![104840ee72dcb5d35827e7ef8b31fbb39512f39038b8b559f2b99d5030b6c943.jpg](../icml_results/1876_Strategic%20Candidacy%20in%20Generative%20AI%20Arenas/images/104840ee72dcb5d35827e7ef8b31fbb39512f39038b8b559f2b99d5030b6c943.jpg)

![2f070d6eb1930e9ebb1550e9a9e035d3ec3aeab4c838faa155343c2a14fd79ea.jpg](../icml_results/1876_Strategic%20Candidacy%20in%20Generative%20AI%20Arenas/images/2f070d6eb1930e9ebb1550e9a9e035d3ec3aeab4c838faa155343c2a14fd79ea.jpg)

![5c7d9ffa97832508abb7400212921a8ebeedfb7c3ec7b6955f908d29305c6bc7.jpg](../icml_results/1876_Strategic%20Candidacy%20in%20Generative%20AI%20Arenas/images/5c7d9ffa97832508abb7400212921a8ebeedfb7c3ec7b6955f908d29305c6bc7.jpg)

![b44eff1e0e51b319d7b7d2bfaa088a6e57f62d7db788e4c07d151769bb6d5fc2.jpg](../icml_results/1876_Strategic%20Candidacy%20in%20Generative%20AI%20Arenas/images/b44eff1e0e51b319d7b7d2bfaa088a6e57f62d7db788e4c07d151769bb6d5fc2.jpg)

![ccaaac54d8dc8e443040ee988658dd9c957563112fa9af92a746806c066d9578.jpg](../icml_results/1876_Strategic%20Candidacy%20in%20Generative%20AI%20Arenas/images/ccaaac54d8dc8e443040ee988658dd9c957563112fa9af92a746806c066d9578.jpg)

<a id="paper-1852"></a>

## OPIC: Enhancing Language Model Merging via Optimizing In-Context Capability


### Images

![248aa4f89c496d55731c0bf309764391e412f0284a8c5f1100becad3f44263c8.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/images/248aa4f89c496d55731c0bf309764391e412f0284a8c5f1100becad3f44263c8.jpg)

![287fc24f17ca8053b80281041ef795dabd4a4134e197083a35de6522e424d50f.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/images/287fc24f17ca8053b80281041ef795dabd4a4134e197083a35de6522e424d50f.jpg)

![75af068e66007be412a9183f4894c5740ea5f0c137a16888b8e5eabfaf054dac.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/images/75af068e66007be412a9183f4894c5740ea5f0c137a16888b8e5eabfaf054dac.jpg)

![e7c8fffa12d9ffa3d9b4e8928b254a11fd55e20e976740fa994d9c0abb072c12.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/images/e7c8fffa12d9ffa3d9b4e8928b254a11fd55e20e976740fa994d9c0abb072c12.jpg)

### Tables

![0b00d70f7529378686e0c980ac966ecb1150d0e4aa539508973055e0934b1782.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/0b00d70f7529378686e0c980ac966ecb1150d0e4aa539508973055e0934b1782.jpg)

![0e265695d3f41f253ccaf21280f486059ecbc5b5e02dc4a6a04396cc7444697d.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/0e265695d3f41f253ccaf21280f486059ecbc5b5e02dc4a6a04396cc7444697d.jpg)

![1c2015af5767331b31f5f9f2d5079d4b2fe2e966ed46782aa848b7ad30d89d12.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/1c2015af5767331b31f5f9f2d5079d4b2fe2e966ed46782aa848b7ad30d89d12.jpg)

![427163b58697d68c2f7b5c356ec744be9d3a7057b424658a04aa76620777b621.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/427163b58697d68c2f7b5c356ec744be9d3a7057b424658a04aa76620777b621.jpg)

![659710dfb35aa1f59efcb914c69817b11ea821fe50a1e7636c73b8d32c754826.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/659710dfb35aa1f59efcb914c69817b11ea821fe50a1e7636c73b8d32c754826.jpg)

![7dc6ed04d5077c77acf55030971c2708d0018308c882cae879a72b55ed4499e7.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/7dc6ed04d5077c77acf55030971c2708d0018308c882cae879a72b55ed4499e7.jpg)

![8c65d280bc3c4356134c1b194b43c0354e34526e969057a2e2006f9e8b484f9f.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/8c65d280bc3c4356134c1b194b43c0354e34526e969057a2e2006f9e8b484f9f.jpg)

![ad6b607faf740c8ccdeca59f06b0d8be7d2b174a27ddbc2a1310c2ea5d2d6504.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/ad6b607faf740c8ccdeca59f06b0d8be7d2b174a27ddbc2a1310c2ea5d2d6504.jpg)

![c0d7511f6fa973b3f8eb3ebd3a8c14c2a020582cdf561098ca121315a572b3cf.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/c0d7511f6fa973b3f8eb3ebd3a8c14c2a020582cdf561098ca121315a572b3cf.jpg)

![c22358bf672ca67386ac589aa7d2b93d65d0413b7c539cdda9a5da46542438f5.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/c22358bf672ca67386ac589aa7d2b93d65d0413b7c539cdda9a5da46542438f5.jpg)

![e327cfab9df6dfc1a7be75cc30937b22323e13284d7fcbd471906914597e6aa2.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/e327cfab9df6dfc1a7be75cc30937b22323e13284d7fcbd471906914597e6aa2.jpg)

![e36ed991e0ced138cede9fb4d4ee1038c257d5b787df3c79925aa65c5310db6e.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/e36ed991e0ced138cede9fb4d4ee1038c257d5b787df3c79925aa65c5310db6e.jpg)

![fd0ec134f5563979c198a30dd69a644a6e47c39c9f8b14fbedeb0dea1e6a3869.jpg](../icml_results/1877_OPIC_%20Enhancing%20Language%20Model%20Merging%20via%20Optimizing%20In-Context%20Capability/tables/fd0ec134f5563979c198a30dd69a644a6e47c39c9f8b14fbedeb0dea1e6a3869.jpg)

<a id="paper-1853"></a>

## Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning


### Images

![06e8d9f60607db3e89b242495b240d5912a74250d712dce812d58549cbc91a89.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/06e8d9f60607db3e89b242495b240d5912a74250d712dce812d58549cbc91a89.jpg)

![3f48071514297fcd416237f08970fcaa2aeac7aba7ab1c0eb2634ff4ce0eab31.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/3f48071514297fcd416237f08970fcaa2aeac7aba7ab1c0eb2634ff4ce0eab31.jpg)

![5b67ac257b3ba78e953d4568f3e776d04e23a3e220661361827711ec761a3bc9.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/5b67ac257b3ba78e953d4568f3e776d04e23a3e220661361827711ec761a3bc9.jpg)

![917c0eb846f562dd87b79936e6051a5ad3cdef4618600b00fd7706d5ead5b471.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/917c0eb846f562dd87b79936e6051a5ad3cdef4618600b00fd7706d5ead5b471.jpg)

![a5333e107b45a6b855b7a8cf9bfc3afb85720c1a31e692742198a2c966e4d28a.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/a5333e107b45a6b855b7a8cf9bfc3afb85720c1a31e692742198a2c966e4d28a.jpg)

![b080db3a775317c505fa47b2bfefcec5deb2b5b83174f316217420b28554c00d.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/b080db3a775317c505fa47b2bfefcec5deb2b5b83174f316217420b28554c00d.jpg)

![d147f3ec5390b09cb72fd5a676e89ae4109e8ab1b937a7cee86285ac40c6af92.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/d147f3ec5390b09cb72fd5a676e89ae4109e8ab1b937a7cee86285ac40c6af92.jpg)

![eaf0819aad6227df539c5ec8cb5029ab97bfcaaed1a3aa7c6f333e1b6b3d6a88.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/eaf0819aad6227df539c5ec8cb5029ab97bfcaaed1a3aa7c6f333e1b6b3d6a88.jpg)

![ecbeecbb10af635324549a2f24fd0a513e0e41a10d3aca598e7103d336953c13.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/ecbeecbb10af635324549a2f24fd0a513e0e41a10d3aca598e7103d336953c13.jpg)

![f6b5371e11da63ca17653249eb7f4ae9013a055a05f9554b3279db94bc3904c9.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/f6b5371e11da63ca17653249eb7f4ae9013a055a05f9554b3279db94bc3904c9.jpg)

![fd32220d959637724087c1e02034c12b343b4d8d68df9fdb4d88b5b5d8a9f798.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/images/fd32220d959637724087c1e02034c12b343b4d8d68df9fdb4d88b5b5d8a9f798.jpg)

### Tables

![0048b7ce3884e30946e59765c354413ab7766bd52fe71f7a8c77a20863b2269b.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/0048b7ce3884e30946e59765c354413ab7766bd52fe71f7a8c77a20863b2269b.jpg)

![0b6aadfaed8349c15908bf786c5ae5ea82045b4db028677051540d2851df12a4.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/0b6aadfaed8349c15908bf786c5ae5ea82045b4db028677051540d2851df12a4.jpg)

![18e7b2dffd200ae0e57fa9f053ac91ce7f43e3da511b6595738f78bfb9f97c5b.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/18e7b2dffd200ae0e57fa9f053ac91ce7f43e3da511b6595738f78bfb9f97c5b.jpg)

![31f2bab38b18b28daeb9f3c72ed8e0137b1992d7d09a8d5fc24b1ed59f938029.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/31f2bab38b18b28daeb9f3c72ed8e0137b1992d7d09a8d5fc24b1ed59f938029.jpg)

![42d55becdffe9f0c74a5de7f2ed13679c80e39306d5642a39e7203d745aba71d.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/42d55becdffe9f0c74a5de7f2ed13679c80e39306d5642a39e7203d745aba71d.jpg)

![489b4d037db03323cae4a9d61862158ea6b05bb350194f1ac7ed7163fd94fd89.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/489b4d037db03323cae4a9d61862158ea6b05bb350194f1ac7ed7163fd94fd89.jpg)

![58d2fbc6f0c17ab43748c6b842858059acd13308ad96a9e79d4dac5c9a5908af.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/58d2fbc6f0c17ab43748c6b842858059acd13308ad96a9e79d4dac5c9a5908af.jpg)

![67e69db12df17987e72650cb31c2ad8cf664e2607e7d91d5cd153004fe74ca69.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/67e69db12df17987e72650cb31c2ad8cf664e2607e7d91d5cd153004fe74ca69.jpg)

![6cac4578f8a2d6211ba1bfc25abb0c5c96f9fbe9b5a904ed6d98aee75f6d2633.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/6cac4578f8a2d6211ba1bfc25abb0c5c96f9fbe9b5a904ed6d98aee75f6d2633.jpg)

![7321c3ece47ec8345c196ac64b9521eb6a653c70635ab5226ffda32b4b81efa7.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/7321c3ece47ec8345c196ac64b9521eb6a653c70635ab5226ffda32b4b81efa7.jpg)

![7934fd627ed63197a8f2308d452d9dae24d9bcb9430015d723b7a0ebcb1f4263.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/7934fd627ed63197a8f2308d452d9dae24d9bcb9430015d723b7a0ebcb1f4263.jpg)

![7e4074a743cbc69ba522fd1699104695cf52a31576480a9c4134b1cdf6d7cdae.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/7e4074a743cbc69ba522fd1699104695cf52a31576480a9c4134b1cdf6d7cdae.jpg)

![a5c0d507ba1a26fd5751746bbd9c8a8aebd6583181710112105cc367c8a38bc1.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/a5c0d507ba1a26fd5751746bbd9c8a8aebd6583181710112105cc367c8a38bc1.jpg)

![ade1b7b22e1890053ec03ed195cc4187213500497eb85e3ea7fc348c5106e7d1.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/ade1b7b22e1890053ec03ed195cc4187213500497eb85e3ea7fc348c5106e7d1.jpg)

![eaf7083e91d2dbb7c330dad7aa8836b51998dbbf2cd4cfc79b4510aaadf7cf1a.jpg](../icml_results/1878_Reasoning%20to%20Edit_%20Hypothetical%20Instruction-Based%20Image%20Editing%20with%20Visual%20Reasoning/tables/eaf7083e91d2dbb7c330dad7aa8836b51998dbbf2cd4cfc79b4510aaadf7cf1a.jpg)

<a id="paper-1854"></a>

## VidLaDA: Bidirectional Diffusion Large Language Models for Efficient Video Understanding


### Images

![0b521ee84e0ab6b5bb1833f100ffd1be7050d8c2ef5f25e23cd3ade73e521550.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/0b521ee84e0ab6b5bb1833f100ffd1be7050d8c2ef5f25e23cd3ade73e521550.jpg)

![19a22ee861e290d79aca90ec93ae4101c19285c3073da888089891704ba776fd.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/19a22ee861e290d79aca90ec93ae4101c19285c3073da888089891704ba776fd.jpg)

![494d0e7952362ccb439573cd51d04094ce4c14205e4fb93c9f990b45136d280d.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/494d0e7952362ccb439573cd51d04094ce4c14205e4fb93c9f990b45136d280d.jpg)

![5bcbc5bee843b0d15d9912b4afd22ac488f9bc8abbb429f187543d8c5c69c301.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/5bcbc5bee843b0d15d9912b4afd22ac488f9bc8abbb429f187543d8c5c69c301.jpg)

![6ed16a56731e475003ab3acf736d4a789f91d9ce312070114182b5d4405dd504.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/6ed16a56731e475003ab3acf736d4a789f91d9ce312070114182b5d4405dd504.jpg)

![a4ca31d25581fe434806fa597604c2a13b73159e54ff3bba1fc2a927b3fffafa.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/a4ca31d25581fe434806fa597604c2a13b73159e54ff3bba1fc2a927b3fffafa.jpg)

![b07dfa761f0df9bf512a890dc545e96e49c18f8c0f0c716effa43e9baa8f64dc.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/b07dfa761f0df9bf512a890dc545e96e49c18f8c0f0c716effa43e9baa8f64dc.jpg)

![ba366d6533e370b2878a27d9656f7b5eea7f75e12ef80608b941a6ba1e84d4e4.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/ba366d6533e370b2878a27d9656f7b5eea7f75e12ef80608b941a6ba1e84d4e4.jpg)

![d03dbd97ab6aff68b4bd525bbd512a64e40de4191a71e45d5859753f00f552a5.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/d03dbd97ab6aff68b4bd525bbd512a64e40de4191a71e45d5859753f00f552a5.jpg)

![d0ac9693dfff94dc39d9c702223735ee50f69553b1ea285df19c193e2ddc15ec.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/images/d0ac9693dfff94dc39d9c702223735ee50f69553b1ea285df19c193e2ddc15ec.jpg)

### Tables

![08c8192088bd61c314d2ac20ad9e848a8aba79c50251fd0475fb4db987e31bc7.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/08c8192088bd61c314d2ac20ad9e848a8aba79c50251fd0475fb4db987e31bc7.jpg)

![09394eaad22a5ebdbf49998cae358e7835e1f882e47cb8daaa266e44a02c1e30.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/09394eaad22a5ebdbf49998cae358e7835e1f882e47cb8daaa266e44a02c1e30.jpg)

![2450cad42a1ff1b0fee1b7c1537c0087a9b0a7a055de23de90c637ea2dc3fe65.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/2450cad42a1ff1b0fee1b7c1537c0087a9b0a7a055de23de90c637ea2dc3fe65.jpg)

![2b7430756e375182704357487e0b271bef1934189b6fec93f005586219987f39.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/2b7430756e375182704357487e0b271bef1934189b6fec93f005586219987f39.jpg)

![2dbaaccc7f9b18e5410d88296a42c9b10ebaef45d055f9c36cc827e8242e94b9.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/2dbaaccc7f9b18e5410d88296a42c9b10ebaef45d055f9c36cc827e8242e94b9.jpg)

![3235d36a4e309dcfe7a2547bc8e256728b4e6b6b731410b53ef6193ca4bf39c5.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/3235d36a4e309dcfe7a2547bc8e256728b4e6b6b731410b53ef6193ca4bf39c5.jpg)

![5299e92061d395bdb044a5609bc501ee7be17f01f654f3b4dbcee9e4519f1c62.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/5299e92061d395bdb044a5609bc501ee7be17f01f654f3b4dbcee9e4519f1c62.jpg)

![7ff7293c14ee8bdc14b14818455165ba77c3297f9eb90b81197b1f53c40fc871.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/7ff7293c14ee8bdc14b14818455165ba77c3297f9eb90b81197b1f53c40fc871.jpg)

![857c0cbc91a92bf8b005f034f065463a4241746ad9a2f30937a946c23c676ff5.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/857c0cbc91a92bf8b005f034f065463a4241746ad9a2f30937a946c23c676ff5.jpg)

![8fd86793dc0bb1fe1c7742799f1854dd4af41a4e293d49d5e80623182df0d102.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/8fd86793dc0bb1fe1c7742799f1854dd4af41a4e293d49d5e80623182df0d102.jpg)

![b022c464a6f5049402a33e29a144054cda62073fefde380359ede0bb4f2f84d5.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/b022c464a6f5049402a33e29a144054cda62073fefde380359ede0bb4f2f84d5.jpg)

![d41b7a73370b165de2fe3e8cfe3f267a62badd10103ea7f7a4f47837c23080cb.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/d41b7a73370b165de2fe3e8cfe3f267a62badd10103ea7f7a4f47837c23080cb.jpg)

![ea53c55e1df460979801bf4bda2ed538e9c0b1f362e617664b6059a32f1fd8b6.jpg](../icml_results/1879_VidLaDA_%20Bidirectional%20Diffusion%20Large%20Language%20Models%20for%20Efficient%20Video%20Understanding/tables/ea53c55e1df460979801bf4bda2ed538e9c0b1f362e617664b6059a32f1fd8b6.jpg)

<a id="paper-1855"></a>

## PGS: Effective LLM Code Refinement via Property-Oriented and Structurally Minimal Feedback


### Images

![3c7e6e1064ff4baaac8a5643946010ccd9da946692fa0747dcb38ac6ac05688b.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/3c7e6e1064ff4baaac8a5643946010ccd9da946692fa0747dcb38ac6ac05688b.jpg)

![464b86e300844be8bf44e8bf5d0c2f86f78154aa9fd162fdf7d58889084424c2.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/464b86e300844be8bf44e8bf5d0c2f86f78154aa9fd162fdf7d58889084424c2.jpg)

![6e0b626af903a111eaa865e027358e9cfe6299bfbd82122720f386740333ac61.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/6e0b626af903a111eaa865e027358e9cfe6299bfbd82122720f386740333ac61.jpg)

![77bacf3f98fc2b04b071000fdb46d65d4a6f540fbc3aaefda028bb8ff656b7d7.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/77bacf3f98fc2b04b071000fdb46d65d4a6f540fbc3aaefda028bb8ff656b7d7.jpg)

![7828e1a054120cc987d246384650091c838916beb0e87b8223f756a136409ca2.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/7828e1a054120cc987d246384650091c838916beb0e87b8223f756a136409ca2.jpg)

![7e81978553d929651ed5e2779e1757494a2d73ed7e7169dd8b4e2922a251da13.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/7e81978553d929651ed5e2779e1757494a2d73ed7e7169dd8b4e2922a251da13.jpg)

![a0a9ee069e32dfade446efc906cecf65b960724b7eeddd0e1864f54d7042668f.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/a0a9ee069e32dfade446efc906cecf65b960724b7eeddd0e1864f54d7042668f.jpg)

![a0db617e35a3b37f32aefdbcf0fb1672fa5c4f24b792e098e1f83f0d4af87ff0.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/a0db617e35a3b37f32aefdbcf0fb1672fa5c4f24b792e098e1f83f0d4af87ff0.jpg)

![adbed3e10800b15b1ecfc87716b0dc39da3dcf8b720ed4e97227e00eeceab1a0.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/adbed3e10800b15b1ecfc87716b0dc39da3dcf8b720ed4e97227e00eeceab1a0.jpg)

![d9c72bba0f0f142990119fee7c4d312ab870f690ce4caeb4e639e832f90a0912.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/d9c72bba0f0f142990119fee7c4d312ab870f690ce4caeb4e639e832f90a0912.jpg)

![ea40a85da52db0037ea5643e7d3662787a1f34eba20bbb1ed79e1e99c25451d6.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/images/ea40a85da52db0037ea5643e7d3662787a1f34eba20bbb1ed79e1e99c25451d6.jpg)

### Tables

![0be4690f6261d846caf10265373beba25c399010e680adf7cebc68997a58bf2f.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/0be4690f6261d846caf10265373beba25c399010e680adf7cebc68997a58bf2f.jpg)

![300742a73a9753edefc4fc83f574baf3356b7972a76c797044270d28774d61af.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/300742a73a9753edefc4fc83f574baf3356b7972a76c797044270d28774d61af.jpg)

![3278f1f38c0d52aca4bd210e88baeb6fe8422386e1e23bd992dac157825897ff.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/3278f1f38c0d52aca4bd210e88baeb6fe8422386e1e23bd992dac157825897ff.jpg)

![3deb3bb485289d0ed02218694e4f84cfaf662fe146d052cfdd316929af70a8f0.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/3deb3bb485289d0ed02218694e4f84cfaf662fe146d052cfdd316929af70a8f0.jpg)

![4a734488a3b76e1ca95db88e818b9c5173a9187b593969ac887fd96499cba29f.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/4a734488a3b76e1ca95db88e818b9c5173a9187b593969ac887fd96499cba29f.jpg)

![50373268c4a253edfbd0a8f06b22be67fc35eace252494ebf78a29c12249e694.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/50373268c4a253edfbd0a8f06b22be67fc35eace252494ebf78a29c12249e694.jpg)

![6a41e147e1ba911599d1881b6d87aba96b428e2cd699903b3e1a7fca6e19f33a.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/6a41e147e1ba911599d1881b6d87aba96b428e2cd699903b3e1a7fca6e19f33a.jpg)

![86c401672373c6102b5f3e1b4273a39596721d314bb7cc5c918665ea0c312f2e.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/86c401672373c6102b5f3e1b4273a39596721d314bb7cc5c918665ea0c312f2e.jpg)

![95e04bbf0ac08ebe4fc211b3382727aa5709c747f83ec3c4072f2d1be4470c9f.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/95e04bbf0ac08ebe4fc211b3382727aa5709c747f83ec3c4072f2d1be4470c9f.jpg)

![a9a05e9fe57475fefc640ba9792831de06f61085b6c77cfa5df727485315911a.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/a9a05e9fe57475fefc640ba9792831de06f61085b6c77cfa5df727485315911a.jpg)

![eccc589e0b62e21f344ef26ec24c5f29434119ffa5b1258cdc4c7fae724cd529.jpg](../icml_results/1880_PGS_%20Effective%20LLM%20Code%20Refinement%20via%20Property-Oriented%20and%20Structurally%20Minimal%20Feedback/tables/eccc589e0b62e21f344ef26ec24c5f29434119ffa5b1258cdc4c7fae724cd529.jpg)

<a id="paper-1856"></a>

## DR$^2$Seg: Decomposed Two-Stage Rollouts for Efficient Reasoning Segmentation in Multimodal Large Language Models


### Images

![0987fe885251ca1248229fb8cd9b2d14e866d376f0000d1c793b6a42e0784740.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/0987fe885251ca1248229fb8cd9b2d14e866d376f0000d1c793b6a42e0784740.jpg)

![120c74437cc6994209b486691c8501483499f75c381af7885cc34684539b7190.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/120c74437cc6994209b486691c8501483499f75c381af7885cc34684539b7190.jpg)

![130a5bc8942d77f42f4d37814b8a9be0ff3a05d456e72fc2b8e5c4c43f50cfd0.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/130a5bc8942d77f42f4d37814b8a9be0ff3a05d456e72fc2b8e5c4c43f50cfd0.jpg)

![155b581b33b3f680afcbc7084fef000551b005cb5246954ad77b6ad57a070fc6.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/155b581b33b3f680afcbc7084fef000551b005cb5246954ad77b6ad57a070fc6.jpg)

![17737536c146b19b7e74e16cfc613a3b63fb95d35e11c532c04bd58fdb3178c7.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/17737536c146b19b7e74e16cfc613a3b63fb95d35e11c532c04bd58fdb3178c7.jpg)

![1df9e79948e9f3a8dd04c1f0d056ac4abd0664cb39303fa3fdf47ac170f11f9d.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/1df9e79948e9f3a8dd04c1f0d056ac4abd0664cb39303fa3fdf47ac170f11f9d.jpg)

![240a99bfdcc12fdbf11e21ecc87f67d26571f787187a34ecdba43cb1ccb63582.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/240a99bfdcc12fdbf11e21ecc87f67d26571f787187a34ecdba43cb1ccb63582.jpg)

![356c16a4c6e3f736593c7276b1258e18528db87b2906e1af4a5d552575700773.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/356c16a4c6e3f736593c7276b1258e18528db87b2906e1af4a5d552575700773.jpg)

![39b1ec73d0b735214977280c763ed9fbc2b24096cf93971a34af840c2fc895a6.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/39b1ec73d0b735214977280c763ed9fbc2b24096cf93971a34af840c2fc895a6.jpg)

![5a6ab04bbadd69f02f893ecd585d8e39b0bcb76b6426048581d2a596a949dadd.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/5a6ab04bbadd69f02f893ecd585d8e39b0bcb76b6426048581d2a596a949dadd.jpg)

![61134d0a578e5f67e5d35cc2c3d0938ab5adad25f2f3a69ff2e6977da50867bb.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/61134d0a578e5f67e5d35cc2c3d0938ab5adad25f2f3a69ff2e6977da50867bb.jpg)

![83a99bbdc2a5af88d2ac5eecf260554f34e1d56ae369d32f15de0d11e01cea3c.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/83a99bbdc2a5af88d2ac5eecf260554f34e1d56ae369d32f15de0d11e01cea3c.jpg)

![8f44bd99d6bf25c784f97ba2bde8b5a04d068378f3cd8b3c9e3252cf482680ad.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/8f44bd99d6bf25c784f97ba2bde8b5a04d068378f3cd8b3c9e3252cf482680ad.jpg)

![8f48b55dc5574e3cd4bc1b1b85b0b92be9446bd3386d4c7026b382bc547c333f.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/8f48b55dc5574e3cd4bc1b1b85b0b92be9446bd3386d4c7026b382bc547c333f.jpg)

![a2bdff65f68b141e20e9dab99a158b3fea4fe8fc6cb8b5a2a014db7754d8e040.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/a2bdff65f68b141e20e9dab99a158b3fea4fe8fc6cb8b5a2a014db7754d8e040.jpg)

![c3aa581256f67d1696136340684f4ba1c1b372400f9302c696d018d504d98a90.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/c3aa581256f67d1696136340684f4ba1c1b372400f9302c696d018d504d98a90.jpg)

![d64fd04e33e2cca7b37b8348aa78f015d946082fa16a16b43b27117d3279abb8.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/d64fd04e33e2cca7b37b8348aa78f015d946082fa16a16b43b27117d3279abb8.jpg)

![dc80174b6cfa682c8fade5fa008a087219998e4d8912f0bc65b0d86af6c8aaeb.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/dc80174b6cfa682c8fade5fa008a087219998e4d8912f0bc65b0d86af6c8aaeb.jpg)

![dcb5bd5ed7f63aa36659067ff69914574ada8344bc8d02680da79e8a779a83e7.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/dcb5bd5ed7f63aa36659067ff69914574ada8344bc8d02680da79e8a779a83e7.jpg)

![dd645e8c1cc599b065e09ab41d5c21808da64c059ecde223d55f4db9343e8797.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/dd645e8c1cc599b065e09ab41d5c21808da64c059ecde223d55f4db9343e8797.jpg)

![e28c08a7c362b0bd481243a8f56f71757a84cca0a614fbc3b49c588d57bafece.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/e28c08a7c362b0bd481243a8f56f71757a84cca0a614fbc3b49c588d57bafece.jpg)

![f068e938d87b18334de2514b100c9d55fa14e0e824de88f61095c47aa91f1cb9.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/f068e938d87b18334de2514b100c9d55fa14e0e824de88f61095c47aa91f1cb9.jpg)

![f3e596c808df68667b4769f9e1f79cef06ff2a884c48881da92dea041233dd96.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/images/f3e596c808df68667b4769f9e1f79cef06ff2a884c48881da92dea041233dd96.jpg)

### Tables

![169116ac18a6f804f520fff490bc00ea4884781a90a38430f57b2f0c954e9069.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/169116ac18a6f804f520fff490bc00ea4884781a90a38430f57b2f0c954e9069.jpg)

![33fa82ccec9ceafcb298192d2e11fbe7957f258592b87a8128c144b84641291d.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/33fa82ccec9ceafcb298192d2e11fbe7957f258592b87a8128c144b84641291d.jpg)

![4ac330ebc659ae5f3330e580684b9edfedf161da174f81747c813417d0b5c7ea.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/4ac330ebc659ae5f3330e580684b9edfedf161da174f81747c813417d0b5c7ea.jpg)

![62b1be4c1377573b1322c5ef7da58917cfe2fdbfc870362284c7b0ff8453af65.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/62b1be4c1377573b1322c5ef7da58917cfe2fdbfc870362284c7b0ff8453af65.jpg)

![8f75826d6e8ec814a48efd58776efb02e4be1bc1f2eb67bd54b1349b7961b14c.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/8f75826d6e8ec814a48efd58776efb02e4be1bc1f2eb67bd54b1349b7961b14c.jpg)

![90238fba5456984075993796d9a8b93ea511d00a11fd03396c98c1d51e9d8339.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/90238fba5456984075993796d9a8b93ea511d00a11fd03396c98c1d51e9d8339.jpg)

![9482f93331419c92355fb0ebf83d83b1d61c07d08dab83d6b7bd808050b8cd94.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/9482f93331419c92355fb0ebf83d83b1d61c07d08dab83d6b7bd808050b8cd94.jpg)

![e683e87be3f1f5bcdb265eda3ad6a4ee87cf973fe40a0be0d654774260c33349.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/e683e87be3f1f5bcdb265eda3ad6a4ee87cf973fe40a0be0d654774260c33349.jpg)

![fa736e60af89d341c0c63d4a68037ef86f8f9724ae986a9052b87e5a91b42521.jpg](../icml_results/1881_DR%24%5E2%24Seg_%20Decomposed%20Two-Stage%20Rollouts%20for%20Efficient%20Reasoning%20Segmentation%20in%20Multimodal%20Large%20La/tables/fa736e60af89d341c0c63d4a68037ef86f8f9724ae986a9052b87e5a91b42521.jpg)

<a id="paper-1857"></a>

## Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning


### Images

![279e401f761fb47aa50323339da5dc2a9f525a3809896028303f212323a1e997.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/279e401f761fb47aa50323339da5dc2a9f525a3809896028303f212323a1e997.jpg)

![2e3957e90030361bb3a315f5c369546e8a2790f67dd44d09e5e3400efb5ddec4.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/2e3957e90030361bb3a315f5c369546e8a2790f67dd44d09e5e3400efb5ddec4.jpg)

![31747f133559ddfc9782b2bd51f51203e93f7ff59e6fc15ef32176f00b461fb2.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/31747f133559ddfc9782b2bd51f51203e93f7ff59e6fc15ef32176f00b461fb2.jpg)

![5d12310c7f355ef173a5f7dfd95312272048a806a65cb7948c80e40c33b58a8d.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/5d12310c7f355ef173a5f7dfd95312272048a806a65cb7948c80e40c33b58a8d.jpg)

![861bf21a4d3474a5f19ddc93a740a7d407fe7cc3cfa7cdcd538f597034d0d1b1.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/861bf21a4d3474a5f19ddc93a740a7d407fe7cc3cfa7cdcd538f597034d0d1b1.jpg)

![8b6fffd9b2724cb12d7a1c1786c5f7e38d814883468507b3c6e44e46df597b03.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/8b6fffd9b2724cb12d7a1c1786c5f7e38d814883468507b3c6e44e46df597b03.jpg)

![a74c312c952b0f2c001cfec489c508d0e8df997dc5554378a9ae0d94514fab2d.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/a74c312c952b0f2c001cfec489c508d0e8df997dc5554378a9ae0d94514fab2d.jpg)

![c4548c84c1bca3e9a991d097d39d44cdbed19ee38b07924cfe5787303feb07d1.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/c4548c84c1bca3e9a991d097d39d44cdbed19ee38b07924cfe5787303feb07d1.jpg)

![e54423ab827c0ef85d819fd528ea5ba1b884e3d8ee10eccd2c1f674e62f43b81.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/e54423ab827c0ef85d819fd528ea5ba1b884e3d8ee10eccd2c1f674e62f43b81.jpg)

![e6f86d34dc6ebf38cbe72633b7ce4f7151b9913cfd977f0617c24b559e99c652.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/e6f86d34dc6ebf38cbe72633b7ce4f7151b9913cfd977f0617c24b559e99c652.jpg)

![f4b6789a46ace9fd8180a8635f46641afb05fd163fc077e62e69f219ac159a7a.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/f4b6789a46ace9fd8180a8635f46641afb05fd163fc077e62e69f219ac159a7a.jpg)

![f956d81a96100c96a6a7b53889d9683c7a875391394134443c94912666008693.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/images/f956d81a96100c96a6a7b53889d9683c7a875391394134443c94912666008693.jpg)

### Tables

![108d2f53fa9cb78658a05d479f814f527b5e6c938a4fe065cbaa3e7c21960b8c.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/108d2f53fa9cb78658a05d479f814f527b5e6c938a4fe065cbaa3e7c21960b8c.jpg)

![252a333b7d51e2fe8b12a286dfee29b699d3fa066479957f76faba76e5fb4d7c.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/252a333b7d51e2fe8b12a286dfee29b699d3fa066479957f76faba76e5fb4d7c.jpg)

![405064121732014adb61df519f95a32052151657ec9097024b99dc02cb9440dc.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/405064121732014adb61df519f95a32052151657ec9097024b99dc02cb9440dc.jpg)

![54ffad0004d2461496de1995faf0c0bd7739ae55e83b7dbc8ea92fc0968aaf37.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/54ffad0004d2461496de1995faf0c0bd7739ae55e83b7dbc8ea92fc0968aaf37.jpg)

![8290b51f70294ecd2cd7ce7a04f5e1a8b1bdf0170a7ea7b5ae564c3d7603510a.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/8290b51f70294ecd2cd7ce7a04f5e1a8b1bdf0170a7ea7b5ae564c3d7603510a.jpg)

![86fbeb268116927105454afd85614628f3f8d0924c01cbc716a441bda79a0ea5.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/86fbeb268116927105454afd85614628f3f8d0924c01cbc716a441bda79a0ea5.jpg)

![978df85294c403c620d59d2b15911a3cc9f9e2d7086e2d39580a0267c8a37b22.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/978df85294c403c620d59d2b15911a3cc9f9e2d7086e2d39580a0267c8a37b22.jpg)

![ceb8789a3862adfb481aebb577e58dc3c7554cff3292e5015e050228ef2e0583.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/ceb8789a3862adfb481aebb577e58dc3c7554cff3292e5015e050228ef2e0583.jpg)

![da069d62525b56ffa995ada2910f2fe53c86d7a6cfe4a9c35be01ca2f4a7a680.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/da069d62525b56ffa995ada2910f2fe53c86d7a6cfe4a9c35be01ca2f4a7a680.jpg)

![ee463ddfadc852e2c5af8e665f5cc2062785155c75f825cd6a91b50b21b6a453.jpg](../icml_results/1882_Task-Driven%20Subspace%20Decomposition%20for%20Knowledge%20Sharing%20and%20Isolation%20in%20LoRA-based%20Continual%20Learn/tables/ee463ddfadc852e2c5af8e665f5cc2062785155c75f825cd6a91b50b21b6a453.jpg)

<a id="paper-1858"></a>

## Self-Prophetic Decoding to Unlock Visual Search in LVLMs


### Images

![03b585a527daa71c5814d8ecb529ea4543b770f20e869a878239eda24784dad9.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/03b585a527daa71c5814d8ecb529ea4543b770f20e869a878239eda24784dad9.jpg)

![185eb24d45335d45eb38262a5021c15d56a506bfa5702bff95891b096d26559a.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/185eb24d45335d45eb38262a5021c15d56a506bfa5702bff95891b096d26559a.jpg)

![28f7db4065c8c18a9e182849b5ae9d4e24fcec072fcb1af308d394214a66dd71.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/28f7db4065c8c18a9e182849b5ae9d4e24fcec072fcb1af308d394214a66dd71.jpg)

![5018d326ee0208fa4e122b68698d9efdeca802fa014ae525c24fbfb748a5e816.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/5018d326ee0208fa4e122b68698d9efdeca802fa014ae525c24fbfb748a5e816.jpg)

![776aa619cf6149c850aeec458681d48149c62b34ef7caa964ca9852d96136ee5.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/776aa619cf6149c850aeec458681d48149c62b34ef7caa964ca9852d96136ee5.jpg)

![b9e84dabdbd1c8d2db41dcc6b82bd7c082894e618fbcb0bb5ebffb7ac5184401.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/b9e84dabdbd1c8d2db41dcc6b82bd7c082894e618fbcb0bb5ebffb7ac5184401.jpg)

![ea4b460a6e05bad5ae44b2d3a2a8050d7aee2c8550868d4df7c9c093a8fec3b9.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/ea4b460a6e05bad5ae44b2d3a2a8050d7aee2c8550868d4df7c9c093a8fec3b9.jpg)

![ee5b98b1a7760c32f1211a0d558463bd05587214bf63b3672186ee6a2bfaa93f.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/images/ee5b98b1a7760c32f1211a0d558463bd05587214bf63b3672186ee6a2bfaa93f.jpg)

### Tables

![0b244f5ad86a4844e46a5953dbe6daec665177ad2f87d207a6beb2b3e286ad8c.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/tables/0b244f5ad86a4844e46a5953dbe6daec665177ad2f87d207a6beb2b3e286ad8c.jpg)

![0d719848f909b583d95877069e39e8f6866054fe407ff49b49da3ff7062c537a.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/tables/0d719848f909b583d95877069e39e8f6866054fe407ff49b49da3ff7062c537a.jpg)

![56774f8ec8b0a4ac18dc7138bf38e88cd3b59a2e2f4f371bf78083829e30da22.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/tables/56774f8ec8b0a4ac18dc7138bf38e88cd3b59a2e2f4f371bf78083829e30da22.jpg)

![b8e62c902ee2d347b068024f28e212d13b165ab9553ee2fc0e17cd9a80eba98e.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/tables/b8e62c902ee2d347b068024f28e212d13b165ab9553ee2fc0e17cd9a80eba98e.jpg)

![b9a71d2308fb264494b53d386f2c0bdfbe9f07110820010355fafbbb43b3723b.jpg](../icml_results/1883_Self-Prophetic%20Decoding%20to%20Unlock%20Visual%20Search%20in%20LVLMs/tables/b9a71d2308fb264494b53d386f2c0bdfbe9f07110820010355fafbbb43b3723b.jpg)

<a id="paper-1859"></a>

## Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter


### Images

![338fb5a8286025361e9c9e9483cca271a569c51c6f872aaba8f39df80e688ffd.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/338fb5a8286025361e9c9e9483cca271a569c51c6f872aaba8f39df80e688ffd.jpg)

![5cd1d1df998b2be342b16f7fde0e59244dc153c0ee38d1abcf3c7c34d02bd0c6.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/5cd1d1df998b2be342b16f7fde0e59244dc153c0ee38d1abcf3c7c34d02bd0c6.jpg)

![63564463b14afe6646421fe161efe1c993c2d35973dfa17354e0e49f7250d8e2.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/63564463b14afe6646421fe161efe1c993c2d35973dfa17354e0e49f7250d8e2.jpg)

![76d7d534a7b2047c9403e11e5bd7b8d00ca4b6f524a8deaccc5b467635eb110b.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/76d7d534a7b2047c9403e11e5bd7b8d00ca4b6f524a8deaccc5b467635eb110b.jpg)

![8814dbb1d4cf60b58f02ef6a2276fef990dbafbcc1aa78cdace4c232689e4911.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/8814dbb1d4cf60b58f02ef6a2276fef990dbafbcc1aa78cdace4c232689e4911.jpg)

![e3e6b9c7cc1df06011f4dcf739940c8605c98bd6f37b7ac8fa3d233168d06638.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/e3e6b9c7cc1df06011f4dcf739940c8605c98bd6f37b7ac8fa3d233168d06638.jpg)

![e6e7a7e9eb6d8a450eda5bb934621141b8a9d54151131044bacbd6452257444c.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/images/e6e7a7e9eb6d8a450eda5bb934621141b8a9d54151131044bacbd6452257444c.jpg)

### Tables

![1b79c83a32f7bc76fbd86462991395d0f7ff8e9912ef47655b0455e64b5584a7.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/1b79c83a32f7bc76fbd86462991395d0f7ff8e9912ef47655b0455e64b5584a7.jpg)

![202a82bfb4941ce48ee72f2037e63bf755955c4f2de5ad4b5de9821a1406df5c.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/202a82bfb4941ce48ee72f2037e63bf755955c4f2de5ad4b5de9821a1406df5c.jpg)

![234081714461cd4039d877bbe763981a60f43dbfb65ef56d7dc3713a54fe07da.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/234081714461cd4039d877bbe763981a60f43dbfb65ef56d7dc3713a54fe07da.jpg)

![340db7ac1bd50219be5d0f0a193c80f6051b7e5c8ea93e4cefec5981f9be71d4.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/340db7ac1bd50219be5d0f0a193c80f6051b7e5c8ea93e4cefec5981f9be71d4.jpg)

![56410e4a3de248a6e03a499b0a002839b0163788499ed4482d6508621a49d745.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/56410e4a3de248a6e03a499b0a002839b0163788499ed4482d6508621a49d745.jpg)

![667c1c3d15502e1feafa14a7e5a207cd38b994a5bebfac64362a8ae39ec2b673.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/667c1c3d15502e1feafa14a7e5a207cd38b994a5bebfac64362a8ae39ec2b673.jpg)

![77669af8f333f123782cac3b49b69d937a8e56da6a2b2d34154f09455d34e489.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/77669af8f333f123782cac3b49b69d937a8e56da6a2b2d34154f09455d34e489.jpg)

![82150c7b68675aa679c97462da990aa2d9d67bdf770f8e35343385ec69374549.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/82150c7b68675aa679c97462da990aa2d9d67bdf770f8e35343385ec69374549.jpg)

![87b8e44946a2f95a8d7da6a8f58e97bf9413cff44e26140d989ff999c0ebf761.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/87b8e44946a2f95a8d7da6a8f58e97bf9413cff44e26140d989ff999c0ebf761.jpg)

![88214c7bc0a2a1735855dd2e800adac6f466aa23aa281b9d2fb7754b688cb3cf.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/88214c7bc0a2a1735855dd2e800adac6f466aa23aa281b9d2fb7754b688cb3cf.jpg)

![9d53cdc091c13213972a50b357c1823b5bd1ac6e5130394b5fd5b0345c67965f.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/9d53cdc091c13213972a50b357c1823b5bd1ac6e5130394b5fd5b0345c67965f.jpg)

![fb99cc968207b23c67fd988fca1b96e25767f4ce9c70a22592a598033d5b078e.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/fb99cc968207b23c67fd988fca1b96e25767f4ce9c70a22592a598033d5b078e.jpg)

![fccd930287b844dc77578a379c2c032eb40f4e5b7ce4d612f807733dfd6071ef.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/fccd930287b844dc77578a379c2c032eb40f4e5b7ce4d612f807733dfd6071ef.jpg)

![ff0a0c6155bc416cc58120d77eaa5cc3acc95fd45abbcfec0c723478ac032cbb.jpg](../icml_results/1884_Compress%20then%20Merge_%20From%20Multiple%20LoRAs%20into%20One%20Low-Rank%20Adapter/tables/ff0a0c6155bc416cc58120d77eaa5cc3acc95fd45abbcfec0c723478ac032cbb.jpg)

<a id="paper-1860"></a>

## Co-RedTeam: Orchestrated Security Discovery and Exploitation with LLM Agents


### Images

![25a995e2155b5218a7215494a72f6dc32f85390e422f828b200f92a5138cd982.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/images/25a995e2155b5218a7215494a72f6dc32f85390e422f828b200f92a5138cd982.jpg)

![2a7c248b0cc74d503eb25d9f7732cdd90e915938dd032ba894a15d6d2e8b72ab.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/images/2a7c248b0cc74d503eb25d9f7732cdd90e915938dd032ba894a15d6d2e8b72ab.jpg)

![6d6c4df02592a812bc3c50496adb55287a5383df56b29c5ba61d6779152ca713.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/images/6d6c4df02592a812bc3c50496adb55287a5383df56b29c5ba61d6779152ca713.jpg)

![c2de203b2eb4fd7566bd92b28f13737feb11b0ea19bac9e4a3a4a458ff6e2ed1.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/images/c2de203b2eb4fd7566bd92b28f13737feb11b0ea19bac9e4a3a4a458ff6e2ed1.jpg)

### Tables

![00839d542473e49d520cd5c95aa196ebea7f6627db7f9f85828d1ef0c6005371.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/00839d542473e49d520cd5c95aa196ebea7f6627db7f9f85828d1ef0c6005371.jpg)

![091a295673b3df57a54aa51327ab5a609abc671d55a56a7f61c29c46cce48553.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/091a295673b3df57a54aa51327ab5a609abc671d55a56a7f61c29c46cce48553.jpg)

![22e90cea0b2a55ec605fad06adec3469a57f3b924b5ec0fc478aa38d5286a771.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/22e90cea0b2a55ec605fad06adec3469a57f3b924b5ec0fc478aa38d5286a771.jpg)

![5bf973be6ed47453c9aeef6cac7b9c2aae353a3a15ff1947d0402087811f06e8.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/5bf973be6ed47453c9aeef6cac7b9c2aae353a3a15ff1947d0402087811f06e8.jpg)

![67985dd27ce0b96bd25cc661b7efb930707a8991f785093c9b8546dc41d67aa0.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/67985dd27ce0b96bd25cc661b7efb930707a8991f785093c9b8546dc41d67aa0.jpg)

![88119da08d2813cd06fa6b97f56a69b05b90d22e12268398f16abf24af708aa5.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/88119da08d2813cd06fa6b97f56a69b05b90d22e12268398f16abf24af708aa5.jpg)

![a259f3f87075e96f2c93223674e802b7487b0ff47088bfd08ae3abc15055ec43.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/a259f3f87075e96f2c93223674e802b7487b0ff47088bfd08ae3abc15055ec43.jpg)

![a82abe9319ba74c7aef7d2f3d2178ad328a286112654540840abf7f4fbb5b649.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/a82abe9319ba74c7aef7d2f3d2178ad328a286112654540840abf7f4fbb5b649.jpg)

![ae866272b7517fd97cd2aef72ed414a119ebb6ca1a26676320b0cd81d7c3a251.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/ae866272b7517fd97cd2aef72ed414a119ebb6ca1a26676320b0cd81d7c3a251.jpg)

![e7afe8965d0fb45f0db4a77b425536f033f98187a3d611a0deea21a34f2fac57.jpg](../icml_results/1885_Co-RedTeam_%20Orchestrated%20Security%20Discovery%20and%20Exploitation%20with%20LLM%20Agents/tables/e7afe8965d0fb45f0db4a77b425536f033f98187a3d611a0deea21a34f2fac57.jpg)

<a id="paper-1861"></a>

## Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization


### Images

![01a5e0fc6ba84448f05f81101c9a4d0e34ac67e0bc9f5b909b749ffb212848b6.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/01a5e0fc6ba84448f05f81101c9a4d0e34ac67e0bc9f5b909b749ffb212848b6.jpg)

![194491af7093910ad023d31f0aa8292ce280669a8108f3eb8154a13d2fb286bc.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/194491af7093910ad023d31f0aa8292ce280669a8108f3eb8154a13d2fb286bc.jpg)

![33dcb4c892509794dec3c13a376eec97b04e5b0f8f61cd6697bb73d730b15a12.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/33dcb4c892509794dec3c13a376eec97b04e5b0f8f61cd6697bb73d730b15a12.jpg)

![37a25f6c937bb23535a005ff17bfcadd02b9b68579f2e7a594541503bbbbf4a5.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/37a25f6c937bb23535a005ff17bfcadd02b9b68579f2e7a594541503bbbbf4a5.jpg)

![5e30feb748f85ab9cea4c32d931952c1f8777128653515d6f87478809c773f00.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/5e30feb748f85ab9cea4c32d931952c1f8777128653515d6f87478809c773f00.jpg)

![739e7cc0147e6ecf54c79950c29a2e27a4f175d4ef89a4851dcfc96572172c1f.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/739e7cc0147e6ecf54c79950c29a2e27a4f175d4ef89a4851dcfc96572172c1f.jpg)

![99595a8da32d552d1161dcd19d6e157177f30a5326f0d31e21649586f11a482f.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/99595a8da32d552d1161dcd19d6e157177f30a5326f0d31e21649586f11a482f.jpg)

![bd3a960c2331bf39107453f379fb9ea62da5d3d30d7d197806126a1f812a9520.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/bd3a960c2331bf39107453f379fb9ea62da5d3d30d7d197806126a1f812a9520.jpg)

![c82bc922d243f15b400b4e8fb47c70d64f6591e69bff3c54cffb96c9a8ec614d.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/c82bc922d243f15b400b4e8fb47c70d64f6591e69bff3c54cffb96c9a8ec614d.jpg)

![d0414dba0da3936c7b43710e6ba72997fbd175010ee4e439851b27f53c8604f9.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/d0414dba0da3936c7b43710e6ba72997fbd175010ee4e439851b27f53c8604f9.jpg)

![ecab69f297108d3773d15485080fe8967349e03862439fdab49ce512b906464f.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/ecab69f297108d3773d15485080fe8967349e03862439fdab49ce512b906464f.jpg)

![fc362d7083903c761ac0c6e85754cc8ef9b14a743d2ae51b6b591edbcf2d1809.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/images/fc362d7083903c761ac0c6e85754cc8ef9b14a743d2ae51b6b591edbcf2d1809.jpg)

### Tables

![c3a7a86ad2cedeed7665ba9cc6f0fd8f56f177c3b088c693684967829b8c3583.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/tables/c3a7a86ad2cedeed7665ba9cc6f0fd8f56f177c3b088c693684967829b8c3583.jpg)

![de3ab5f4be9d7d59e0088527102c45d3476cf09e9388220692e4289622649777.jpg](../icml_results/1886_Flash-GRPO_%20Efficient%20Alignment%20for%20Video%20Diffusion%20via%20One-Step%20Policy%20Optimization/tables/de3ab5f4be9d7d59e0088527102c45d3476cf09e9388220692e4289622649777.jpg)

<a id="paper-1862"></a>

## Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model


### Images

![0a92c1ac05a778e8a11d65c1f7d1e1c2edf52255bdb8fa626f26feddbe4af913.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/0a92c1ac05a778e8a11d65c1f7d1e1c2edf52255bdb8fa626f26feddbe4af913.jpg)

![10f1e25b860b0f2c99181e298bd6a823524413972fdc1934a60f2840a7e87b8c.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/10f1e25b860b0f2c99181e298bd6a823524413972fdc1934a60f2840a7e87b8c.jpg)

![16865ed5b93be152ce05710568401005486e1a82912a31a40026f2058b485fd4.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/16865ed5b93be152ce05710568401005486e1a82912a31a40026f2058b485fd4.jpg)

![1d4fa87b75a70e3c84535b99efeba495084eeacbb5b52948f993b526d62b9a00.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/1d4fa87b75a70e3c84535b99efeba495084eeacbb5b52948f993b526d62b9a00.jpg)

![31e2573cd5a82108900645654b9613f777212d08f1f910f28f98876f67df74a6.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/31e2573cd5a82108900645654b9613f777212d08f1f910f28f98876f67df74a6.jpg)

![34c60638aa92c5eb90d8fb7c3ba4e0a4616f24b6672b437be06365d8f145dad2.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/34c60638aa92c5eb90d8fb7c3ba4e0a4616f24b6672b437be06365d8f145dad2.jpg)

![356a85d53b46508918eefe3a36a8d6a0675a1186d543c5c907e0f1d69f50f887.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/356a85d53b46508918eefe3a36a8d6a0675a1186d543c5c907e0f1d69f50f887.jpg)

![43f7603721fd8b99deaeb0a372862cdd41c2dd1633241e7e73892ec90135f898.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/43f7603721fd8b99deaeb0a372862cdd41c2dd1633241e7e73892ec90135f898.jpg)

![8ba7b2f6f9062e4dac7f28c1ecd6dd03993f2c035d3b040907936d33e00552bd.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/8ba7b2f6f9062e4dac7f28c1ecd6dd03993f2c035d3b040907936d33e00552bd.jpg)

![990b4bfd2e52eb87bb06e3434cb3888f1af17901af42d5106f3376a481e77fa3.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/990b4bfd2e52eb87bb06e3434cb3888f1af17901af42d5106f3376a481e77fa3.jpg)

![b1adeebecaf91a7cb870b06f3bc0d0776ff33a875cdfbfb201e46bf97574a6eb.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/b1adeebecaf91a7cb870b06f3bc0d0776ff33a875cdfbfb201e46bf97574a6eb.jpg)

![bc81f2ca9b769d19f2c24ba17371f93552a73f94be257b9bd95a8e029198b2dd.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/bc81f2ca9b769d19f2c24ba17371f93552a73f94be257b9bd95a8e029198b2dd.jpg)

![ce8b38095f68f6891e0ddf81185763eaf1d08367685b7958f7f79e432b8cea12.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/ce8b38095f68f6891e0ddf81185763eaf1d08367685b7958f7f79e432b8cea12.jpg)

![ede9c6a16b0dc06fd0d72d93a116036596eed3ae82f1cf861a12d110b0399fc8.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/ede9c6a16b0dc06fd0d72d93a116036596eed3ae82f1cf861a12d110b0399fc8.jpg)

![f83955ff1129f11d03f44439066db79bb2ebf5acdf91954898d69a96ee98c3f6.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/images/f83955ff1129f11d03f44439066db79bb2ebf5acdf91954898d69a96ee98c3f6.jpg)

### Tables

![068674d0d210a043f300b338961751109b0154dcb933f93ab636072fdaf652ed.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/068674d0d210a043f300b338961751109b0154dcb933f93ab636072fdaf652ed.jpg)

![092a05b63a0c372ad3dadca1432219a6b7dda5fa97c5f1c18c8f9deedb8cc55c.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/092a05b63a0c372ad3dadca1432219a6b7dda5fa97c5f1c18c8f9deedb8cc55c.jpg)

![1cd7605a1b33aab1650ae225e55ed724c4642d14294bc7b2ac861e8f70f902aa.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/1cd7605a1b33aab1650ae225e55ed724c4642d14294bc7b2ac861e8f70f902aa.jpg)

![283c668fea38aa8e830a1f6a38ecb9b4bc3fad9f827758a1caac485d9cc8004b.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/283c668fea38aa8e830a1f6a38ecb9b4bc3fad9f827758a1caac485d9cc8004b.jpg)

![58bcb751fbe0942a167f4eb96205df801449576ae56b513c1aa10505d67be1f0.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/58bcb751fbe0942a167f4eb96205df801449576ae56b513c1aa10505d67be1f0.jpg)

![5fe88cf4218beb4486dce6e3607ed8aec6a7736364be9c2009d0802d666c9edc.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/5fe88cf4218beb4486dce6e3607ed8aec6a7736364be9c2009d0802d666c9edc.jpg)

![6c0def83807534a734bd9b5d858bca44a51885abdc7349e2c8f66c42f029ff30.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/6c0def83807534a734bd9b5d858bca44a51885abdc7349e2c8f66c42f029ff30.jpg)

![6fed2fe6c526dde0b4ef699339853079d986c977eed71d91e25b6ca5cd64e98d.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/6fed2fe6c526dde0b4ef699339853079d986c977eed71d91e25b6ca5cd64e98d.jpg)

![8a4e6b9f2014271d7fa4d49ea1625a52386f1581deaa8fd095d386509031023c.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/8a4e6b9f2014271d7fa4d49ea1625a52386f1581deaa8fd095d386509031023c.jpg)

![946c6693631ede247c608422042e9a34e31d5bf0d97f33f1f6fdd1ed8afa8af0.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/946c6693631ede247c608422042e9a34e31d5bf0d97f33f1f6fdd1ed8afa8af0.jpg)

![a51ed550b1c5b813379350895c67afaeae0d7451cb64f0b2687d7bfbd4b95a41.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/a51ed550b1c5b813379350895c67afaeae0d7451cb64f0b2687d7bfbd4b95a41.jpg)

![ac6f01b0c97db72a510f6a90d6c2497769e4da5fe557b199275cf66253fc3f2a.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/ac6f01b0c97db72a510f6a90d6c2497769e4da5fe557b199275cf66253fc3f2a.jpg)

![b4383c081293d46dcd25312230efb8265e391e05e4f940e4ed40f60d34d269f0.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/b4383c081293d46dcd25312230efb8265e391e05e4f940e4ed40f60d34d269f0.jpg)

![b63cbec09c00695581c20c8fef98f1340f2511bbeab9decde19450e9e8e24272.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/b63cbec09c00695581c20c8fef98f1340f2511bbeab9decde19450e9e8e24272.jpg)

![be2d42c6f8e12dbcb07bd733acff29d14ab03c856b15fa8a94594f13777026c4.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/be2d42c6f8e12dbcb07bd733acff29d14ab03c856b15fa8a94594f13777026c4.jpg)

![d0839cca67aa8ef898fb4e76806664800ce7615e6ad012ecd7b119f318938f9b.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/d0839cca67aa8ef898fb4e76806664800ce7615e6ad012ecd7b119f318938f9b.jpg)

![e4cf9e0d162bdc79a06cd8f6eb6005edc248b1d20837b8e706b445ea769c8d98.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/e4cf9e0d162bdc79a06cd8f6eb6005edc248b1d20837b8e706b445ea769c8d98.jpg)

![e5dc6c4016c69e9fe33b6efe89ee267d3068329b0191ca464bd1ca6af7001d21.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/e5dc6c4016c69e9fe33b6efe89ee267d3068329b0191ca464bd1ca6af7001d21.jpg)

![fdb0f6711e0309475e6ea41e5ba8d09881f0fc661c9be7b94bbc1f7782012b34.jpg](../icml_results/1887_Anchored%20Decoding_%20Provably%20Reducing%20Copyright%20Risk%20for%20Any%20Language%20Model/tables/fdb0f6711e0309475e6ea41e5ba8d09881f0fc661c9be7b94bbc1f7782012b34.jpg)

<a id="paper-1863"></a>

## Structure-Centric Graph Foundation Model via Geometric Bases


### Images

![0b11c6e9df03c9342882268f2adee4326d4d4881b4d522e6c3b4afd5a8fcfa81.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/0b11c6e9df03c9342882268f2adee4326d4d4881b4d522e6c3b4afd5a8fcfa81.jpg)

![11009032dfb789bb151640475bf255a56548855a3b1796651c9923c02bcf9903.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/11009032dfb789bb151640475bf255a56548855a3b1796651c9923c02bcf9903.jpg)

![5292c296afe238c9f42b4dd44e2faef9055e33a400d7403ffc1e7090f78fc183.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/5292c296afe238c9f42b4dd44e2faef9055e33a400d7403ffc1e7090f78fc183.jpg)

![7161b7a192ae5450d077f44fada3d03f4dbeab551d12af903b5c8b4a097c19e5.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/7161b7a192ae5450d077f44fada3d03f4dbeab551d12af903b5c8b4a097c19e5.jpg)

![a1106f27ee0957171d143f8ab29a55050b3eddea3cefdd8e5f3911afafb8acf1.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/a1106f27ee0957171d143f8ab29a55050b3eddea3cefdd8e5f3911afafb8acf1.jpg)

![a856a8db998aea7406710dd52989ee86c8e3776e636f851da91311b970616c85.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/a856a8db998aea7406710dd52989ee86c8e3776e636f851da91311b970616c85.jpg)

![adebe5e9c4a62231b8d8310c46bcb4e9063de7b9a3c5306f17294934769ee7c3.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/adebe5e9c4a62231b8d8310c46bcb4e9063de7b9a3c5306f17294934769ee7c3.jpg)

![afbd45f99b82afb8a8443089d770879db31fc5d86ce34e3857870cfb1ec917ef.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/afbd45f99b82afb8a8443089d770879db31fc5d86ce34e3857870cfb1ec917ef.jpg)

![c48e486f0502d78504a80307d6be64937b316a32981af3fbf14cb0ff24ba272f.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/c48e486f0502d78504a80307d6be64937b316a32981af3fbf14cb0ff24ba272f.jpg)

![efe360f041f0d6ff88affab4ed7401274a964f7226775f1bc5402504a0f74cd2.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/efe360f041f0d6ff88affab4ed7401274a964f7226775f1bc5402504a0f74cd2.jpg)

![fc7a1d96bf7eb4effe4ada9c6e54d872d2377275a365172f41e067e1ae07a9db.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/images/fc7a1d96bf7eb4effe4ada9c6e54d872d2377275a365172f41e067e1ae07a9db.jpg)

### Tables

![03f1b4a4fff58215f37958e21bd1a2453d4aad42395ec77f7dacb5367dffdfac.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/03f1b4a4fff58215f37958e21bd1a2453d4aad42395ec77f7dacb5367dffdfac.jpg)

![1a3ad998b9fee41df8f21bb6cf0aa4604599c16597f20b76df6224189a45f4b0.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/1a3ad998b9fee41df8f21bb6cf0aa4604599c16597f20b76df6224189a45f4b0.jpg)

![1febe5315476db1bdc4b86a340b2d5ab4c5d76d03d04f0787fa722b6654e92bc.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/1febe5315476db1bdc4b86a340b2d5ab4c5d76d03d04f0787fa722b6654e92bc.jpg)

![3e4152ccb6ea728579458da9ea5ab7938d60362cdda1e97a18268c0a4d11f682.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/3e4152ccb6ea728579458da9ea5ab7938d60362cdda1e97a18268c0a4d11f682.jpg)

![3ebaa7dbb802cef92c6e5c113e650087e36ce780341214d93193f7bfbd4c8906.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/3ebaa7dbb802cef92c6e5c113e650087e36ce780341214d93193f7bfbd4c8906.jpg)

![3f30e4b4831afcc1c56321431ce108a578802dc0b205e803dfe16f32e2c4176e.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/3f30e4b4831afcc1c56321431ce108a578802dc0b205e803dfe16f32e2c4176e.jpg)

![41c629020f91b89cdf477d7cb23ff83e504ad5c11cedbad1e93b19cd44e3ec29.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/41c629020f91b89cdf477d7cb23ff83e504ad5c11cedbad1e93b19cd44e3ec29.jpg)

![41ec06f72a3f399d23e9da571074184470be4c1638159bc4cd2418f306f57790.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/41ec06f72a3f399d23e9da571074184470be4c1638159bc4cd2418f306f57790.jpg)

![45a40633e9488ef80455d253dfeae79c0d6065cf10484442476a5bc5cbfa951a.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/45a40633e9488ef80455d253dfeae79c0d6065cf10484442476a5bc5cbfa951a.jpg)

![4830c46423a94a1eef0cd681092659a15a4bbb2aa9575aab6f88da6da7d86a36.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/4830c46423a94a1eef0cd681092659a15a4bbb2aa9575aab6f88da6da7d86a36.jpg)

![5bbd8f1fb758cefdcd2688abadd167ec216a783fbe5d3c3e8b82c23e3849d916.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/5bbd8f1fb758cefdcd2688abadd167ec216a783fbe5d3c3e8b82c23e3849d916.jpg)

![5ce1570ffa65a51ff2e782a99f6cca69794e98ed36e284da2ca1606e37f8684d.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/5ce1570ffa65a51ff2e782a99f6cca69794e98ed36e284da2ca1606e37f8684d.jpg)

![8cc67c52bb2f6533f2537edcec28dcee4ba2c0df6d27f172b30229c25dd2fc92.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/8cc67c52bb2f6533f2537edcec28dcee4ba2c0df6d27f172b30229c25dd2fc92.jpg)

![ab86e74279f73efc4cae4ccd9816b49e166a66ad9ac0e170ef3a314076f91ad7.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/ab86e74279f73efc4cae4ccd9816b49e166a66ad9ac0e170ef3a314076f91ad7.jpg)

![b9d7308a03d2d94d59314d7d5e310fd570522e6a0a6f9b25ccbe04e9fa90fbb8.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/b9d7308a03d2d94d59314d7d5e310fd570522e6a0a6f9b25ccbe04e9fa90fbb8.jpg)

![d634d7642368f04b98451ded91f271e0ee26451a2d462045cb95b8f8b14329f3.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/d634d7642368f04b98451ded91f271e0ee26451a2d462045cb95b8f8b14329f3.jpg)

![e9d33d3bc64d6a7050b6dd581311960c01ef7dac1ed5c3b3083122e36468e181.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/e9d33d3bc64d6a7050b6dd581311960c01ef7dac1ed5c3b3083122e36468e181.jpg)

![f5e2064229eaf3b9c18c0a214d0953985e8fadcab00dee8d6aa22290dd4fb0e7.jpg](../icml_results/1888_Structure-Centric%20Graph%20Foundation%20Model%20via%20Geometric%20Bases/tables/f5e2064229eaf3b9c18c0a214d0953985e8fadcab00dee8d6aa22290dd4fb0e7.jpg)

<a id="paper-1864"></a>

## Navigating the Flatlands: Dual Adaptive Sharpness-Aware Minimization for Domain Generalization


### Images

![5db2d96ab16ace260715f68953e68808f16db210f7a014b8d795f11752aae8a4.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/images/5db2d96ab16ace260715f68953e68808f16db210f7a014b8d795f11752aae8a4.jpg)

![d00ddf6ef0250775f07bc06311b66cebfaf63c4f6ce77a374549e5c6a0b9589c.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/images/d00ddf6ef0250775f07bc06311b66cebfaf63c4f6ce77a374549e5c6a0b9589c.jpg)

![ef9a25d969d2db7a490f7d3b6ecb4d2ed15cf019d6130db96b3dbfbb8a603d68.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/images/ef9a25d969d2db7a490f7d3b6ecb4d2ed15cf019d6130db96b3dbfbb8a603d68.jpg)

### Tables

![093a8fb681a6185dda62de6cd92a5f58ff77f65a34112e2b88712b3d5f66ad81.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/093a8fb681a6185dda62de6cd92a5f58ff77f65a34112e2b88712b3d5f66ad81.jpg)

![330e4707ee009e739986948b4a9910a5317cefe3de47ca06b94052186a89512a.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/330e4707ee009e739986948b4a9910a5317cefe3de47ca06b94052186a89512a.jpg)

![343e8cf4f8cfd76fd3127f0e691e5350a93fe3b8b0c7189105ed58ed5eea3e89.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/343e8cf4f8cfd76fd3127f0e691e5350a93fe3b8b0c7189105ed58ed5eea3e89.jpg)

![406668a74fb1bc6539051a19ddf8b61b788b3e7543fa19b9b5a39fbeb0c5258a.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/406668a74fb1bc6539051a19ddf8b61b788b3e7543fa19b9b5a39fbeb0c5258a.jpg)

![59870cd23d0d047e6515fcc184c167ba4ab65116d49ddcfe48e1322eb093bb5c.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/59870cd23d0d047e6515fcc184c167ba4ab65116d49ddcfe48e1322eb093bb5c.jpg)

![61c4166f75a94b985e71f1c253cd7f7b384bf861737c4367fd324b45babfd618.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/61c4166f75a94b985e71f1c253cd7f7b384bf861737c4367fd324b45babfd618.jpg)

![75309fbc9cf2014d0ac9f3a06729b05b95b3ab7ff134d36f42b0d160421e97fc.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/75309fbc9cf2014d0ac9f3a06729b05b95b3ab7ff134d36f42b0d160421e97fc.jpg)

![7848ada5273a6f99cb66e328485aaf2404748c3989e91112f865c1e94651b06a.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/7848ada5273a6f99cb66e328485aaf2404748c3989e91112f865c1e94651b06a.jpg)

![93fd77cf32f0e0210bc7822f6bf4adb4473b954fa3f39716113b6a6ded4cea84.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/93fd77cf32f0e0210bc7822f6bf4adb4473b954fa3f39716113b6a6ded4cea84.jpg)

![9c39c2cd3cb24742cec2ff60e84e4508cb31430b58eca2804a71b1b7f4e13460.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/9c39c2cd3cb24742cec2ff60e84e4508cb31430b58eca2804a71b1b7f4e13460.jpg)

![a9f1621f401878c5b608d495a45cdf251eeadff8ef37748a64a7117dc66bb8cd.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/a9f1621f401878c5b608d495a45cdf251eeadff8ef37748a64a7117dc66bb8cd.jpg)

![b3468f9d4e01ab77cc5d95176e740a062e447be1bb23bd01354d3b0f3775f90d.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/b3468f9d4e01ab77cc5d95176e740a062e447be1bb23bd01354d3b0f3775f90d.jpg)

![ea1ece652d05460d3ad3fbbdf6ed31431dad3df155268ce88ef9a714ad78026d.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/ea1ece652d05460d3ad3fbbdf6ed31431dad3df155268ce88ef9a714ad78026d.jpg)

![ebbb56fe158c8c3104db7c30e23ba3e55a1133427542e013abd9d7c2fab562d4.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/ebbb56fe158c8c3104db7c30e23ba3e55a1133427542e013abd9d7c2fab562d4.jpg)

![f201cd489b5e3aee23e73c0e600e703a8c7de9a0f54888ee869ac5dd59ed4b97.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/f201cd489b5e3aee23e73c0e600e703a8c7de9a0f54888ee869ac5dd59ed4b97.jpg)

![f2de1167b3bd8c6e511acaf9ae8a0e93beac47e6aa6070efd1cacde5dee7a299.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/f2de1167b3bd8c6e511acaf9ae8a0e93beac47e6aa6070efd1cacde5dee7a299.jpg)

![f5827455249b8a3f0328a0b23a4ec0a82057dc9484d23609b24bec645878db23.jpg](../icml_results/1889_Navigating%20the%20Flatlands_%20Dual%20Adaptive%20Sharpness-Aware%20Minimization%20for%20Domain%20Generalization/tables/f5827455249b8a3f0328a0b23a4ec0a82057dc9484d23609b24bec645878db23.jpg)

<a id="paper-1865"></a>

## Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration


### Images

![47782cad461eabc99fbf42e1c7b98af01a579365e2eb7e92efdbbfb4d5c04329.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/47782cad461eabc99fbf42e1c7b98af01a579365e2eb7e92efdbbfb4d5c04329.jpg)

![67dec6333bec39862781b5ded6e53afb84b173d6e8c2f1103df2c71e7e825a50.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/67dec6333bec39862781b5ded6e53afb84b173d6e8c2f1103df2c71e7e825a50.jpg)

![8d21af94be4da6bf485ba691271f7d66a80c8a04455525b115ce89554e1c855b.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/8d21af94be4da6bf485ba691271f7d66a80c8a04455525b115ce89554e1c855b.jpg)

![9a89e37de6b7e5bd07f34ffea1368a25a30b4ace14b3f76f88728bf3254da012.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/9a89e37de6b7e5bd07f34ffea1368a25a30b4ace14b3f76f88728bf3254da012.jpg)

![ad67270e75b4e8bf588ce226069eb1f47811b2d18da7e02c2db5e91bd3b8f771.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/ad67270e75b4e8bf588ce226069eb1f47811b2d18da7e02c2db5e91bd3b8f771.jpg)

![c2fada8f630b229516ad6b1aeaf2053e0c1bddbeade89b8c6f83cc1c038b324f.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/c2fada8f630b229516ad6b1aeaf2053e0c1bddbeade89b8c6f83cc1c038b324f.jpg)

![effe3f022e4b8cd3d96a8757fad6ddeb1a53e7ccc77a804926270bec1beeabf9.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/images/effe3f022e4b8cd3d96a8757fad6ddeb1a53e7ccc77a804926270bec1beeabf9.jpg)

### Tables

![106ead950b37ee44c031763d4b81e754afd485d5bd3052500d43b6b1e9ddc4f3.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/106ead950b37ee44c031763d4b81e754afd485d5bd3052500d43b6b1e9ddc4f3.jpg)

![177a72df4eb343333f83da6be053649667a073ee7b4504825747818781d0670e.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/177a72df4eb343333f83da6be053649667a073ee7b4504825747818781d0670e.jpg)

![18232918ac4aad8b75b16ce218d88dfc4d49661753f8ec8b1f4325109f3f71f2.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/18232918ac4aad8b75b16ce218d88dfc4d49661753f8ec8b1f4325109f3f71f2.jpg)

![45073bcac1aba52bdc13cbaefff5e8cb78a355abdeaefc7cd1b8b8816272e771.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/45073bcac1aba52bdc13cbaefff5e8cb78a355abdeaefc7cd1b8b8816272e771.jpg)

![554a41b07fc232d49219175209415ece0ee3ce79f62a3ae677567ba01d397c75.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/554a41b07fc232d49219175209415ece0ee3ce79f62a3ae677567ba01d397c75.jpg)

![5b0a2947e257bc532f413ca2245680eb79d86dc6abb83c781a84186610b37d00.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/5b0a2947e257bc532f413ca2245680eb79d86dc6abb83c781a84186610b37d00.jpg)

![731d9b33efc95c34e4173758d70619bfc07426d7d83fdfed8c70d25ac699b050.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/731d9b33efc95c34e4173758d70619bfc07426d7d83fdfed8c70d25ac699b050.jpg)

![9ebfb9b5ef3d167599c8a746f6e497f5f6cd973ffcf4af3944842c7b866a63da.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/9ebfb9b5ef3d167599c8a746f6e497f5f6cd973ffcf4af3944842c7b866a63da.jpg)

![b5f47de59848f8b87cf947b1e5aeed83779640a97bb8ca0bc46c9e4d108a6b4c.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/b5f47de59848f8b87cf947b1e5aeed83779640a97bb8ca0bc46c9e4d108a6b4c.jpg)

![b692462648ea68f1fca3bfa7366fe17f0fcf50e9f2427372e8cf790b6334409e.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/b692462648ea68f1fca3bfa7366fe17f0fcf50e9f2427372e8cf790b6334409e.jpg)

![c7e422b92f114e9e1a991457ef87ead7038e9f51aaf4c4672b5ad527b1359f29.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/c7e422b92f114e9e1a991457ef87ead7038e9f51aaf4c4672b5ad527b1359f29.jpg)

![d63e9a457a025022fc061e686c5b18b62114ce6d159fa714d47cfa6bf09b39dd.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/d63e9a457a025022fc061e686c5b18b62114ce6d159fa714d47cfa6bf09b39dd.jpg)

![f094aa7b0dfd2d8d1cf089ffe5b774af9b7f4bc376c7bd7f31120b98484ecb54.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/f094aa7b0dfd2d8d1cf089ffe5b774af9b7f4bc376c7bd7f31120b98484ecb54.jpg)

![f77a78caad534bd8a942c1be3f6f906428787e43777a1db4801203c06649ace9.jpg](../icml_results/1890_Search-R2_%20Enhancing%20Search-Integrated%20Reasoning%20via%20Actor-Refiner%20Collaboration/tables/f77a78caad534bd8a942c1be3f6f906428787e43777a1db4801203c06649ace9.jpg)

<a id="paper-1866"></a>

## The Accumulation of Score Estimation Error in Diffusion Models


### Images

![152b58777c8cf85123539851636bde4ff7a319a4153243d3b9251095b0300ad2.jpg](../icml_results/1891_The%20Accumulation%20of%20Score%20Estimation%20Error%20in%20Diffusion%20Models/images/152b58777c8cf85123539851636bde4ff7a319a4153243d3b9251095b0300ad2.jpg)

### Tables

![927ddc5d63c97c99ba229b60d1360e9f35bd72f6b06ffeb9d2c0c4e48f959ab6.jpg](../icml_results/1891_The%20Accumulation%20of%20Score%20Estimation%20Error%20in%20Diffusion%20Models/tables/927ddc5d63c97c99ba229b60d1360e9f35bd72f6b06ffeb9d2c0c4e48f959ab6.jpg)

![dadba62d7f307832bc23f8e0f08aff345e22ad7522b9cbe4c3e930c959fc9626.jpg](../icml_results/1891_The%20Accumulation%20of%20Score%20Estimation%20Error%20in%20Diffusion%20Models/tables/dadba62d7f307832bc23f8e0f08aff345e22ad7522b9cbe4c3e930c959fc9626.jpg)

<a id="paper-1867"></a>

## Efficient Multi-round LLM Inference over Disaggregated Serving


### Images

![1ebe5b873421cba0afd13ef32bfc35515a980c37d58a68b97b4ed483d2fddc00.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/1ebe5b873421cba0afd13ef32bfc35515a980c37d58a68b97b4ed483d2fddc00.jpg)

![45240990f321800b99b1da1b36dbadfa9fb1458de9ec00fbf926b6bcfe74adf4.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/45240990f321800b99b1da1b36dbadfa9fb1458de9ec00fbf926b6bcfe74adf4.jpg)

![46f9f13083e951263a3c651d3b7e7d284902ea63eb64dc4717f34954171ad30e.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/46f9f13083e951263a3c651d3b7e7d284902ea63eb64dc4717f34954171ad30e.jpg)

![6fe8249d1d8cf03bc24449166acb452f5236d9d634541178ce9ce32041be2e40.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/6fe8249d1d8cf03bc24449166acb452f5236d9d634541178ce9ce32041be2e40.jpg)

![7660ad47870d4a9ad800df87b12a47234d778203f4a7709b42c653dcc5fe5878.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/7660ad47870d4a9ad800df87b12a47234d778203f4a7709b42c653dcc5fe5878.jpg)

![7e0ed83af97b5591cbdffdaf0227185f59394fbdb27f4d3d998af292ab62a78e.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/7e0ed83af97b5591cbdffdaf0227185f59394fbdb27f4d3d998af292ab62a78e.jpg)

![952abdda92f3aebaccf2b3c066b12628b201c6e87ee921f8ce20cd7668e1f8b3.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/952abdda92f3aebaccf2b3c066b12628b201c6e87ee921f8ce20cd7668e1f8b3.jpg)

![a5a8c3e6361188123514a7adade32da54536385d3c495ec0a07ff32c1e894e3b.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/a5a8c3e6361188123514a7adade32da54536385d3c495ec0a07ff32c1e894e3b.jpg)

![b4473609d560526dd6bd387e9687300b22b416fe56717f2a7c4dd87d032a5ecd.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/b4473609d560526dd6bd387e9687300b22b416fe56717f2a7c4dd87d032a5ecd.jpg)

![c34369b41b34763151476bad5b84cad2f0d456723890e521766dffd497f48ed0.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/c34369b41b34763151476bad5b84cad2f0d456723890e521766dffd497f48ed0.jpg)

![f869030ab07e72eafc1c844a8b24162a82e4cefbf957da5f29bcbf232c0a0646.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/images/f869030ab07e72eafc1c844a8b24162a82e4cefbf957da5f29bcbf232c0a0646.jpg)

### Tables

![0d92a8f5cef0bf7a3f64526614444b25590a5366b44b346d67a71b8de4bc226a.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/tables/0d92a8f5cef0bf7a3f64526614444b25590a5366b44b346d67a71b8de4bc226a.jpg)

![d7d2beae27bed545106b804e7bc71998b951c644eedbe7fe061834f12470b79e.jpg](../icml_results/1892_Efficient%20Multi-round%20LLM%20Inference%20over%20Disaggregated%20Serving/tables/d7d2beae27bed545106b804e7bc71998b951c644eedbe7fe061834f12470b79e.jpg)

<a id="paper-1868"></a>

## DSGCR: Decomposed Spectral Geometry-Aware Cross-Modal Semantic Representation for 3D Visual Grounding


### Images

![521f758439544b992bec58ef26054bc45afd818e4e84c9a1575a45cb6ac9166d.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/images/521f758439544b992bec58ef26054bc45afd818e4e84c9a1575a45cb6ac9166d.jpg)

![5e07262f613b5c43f858b8fe06984c99edc5ab2e58e4706c068da349e2cc9ae0.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/images/5e07262f613b5c43f858b8fe06984c99edc5ab2e58e4706c068da349e2cc9ae0.jpg)

![8cdaee41bbe546e4573368c305b2008d04a3e70dd37312d19a93ad837a55bd3b.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/images/8cdaee41bbe546e4573368c305b2008d04a3e70dd37312d19a93ad837a55bd3b.jpg)

![9459e7595cc6881688f2aa3b6e1b30a364e61673d7f991bb0f34907e223a0c22.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/images/9459e7595cc6881688f2aa3b6e1b30a364e61673d7f991bb0f34907e223a0c22.jpg)

![9da29a1fb55c026eef7e92c76819fbbfb1b033a1f27b1e598e35b2a0985cfc5b.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/images/9da29a1fb55c026eef7e92c76819fbbfb1b033a1f27b1e598e35b2a0985cfc5b.jpg)

### Tables

![2106ae0e7bbca59f7500f53cd4ccb51b2101a99e890beffaed048e7146889499.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/2106ae0e7bbca59f7500f53cd4ccb51b2101a99e890beffaed048e7146889499.jpg)

![2d8a3a18aeb176639877412aabb2708a647d622e9e30a84ac075d1e7b7ece5e5.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/2d8a3a18aeb176639877412aabb2708a647d622e9e30a84ac075d1e7b7ece5e5.jpg)

![4b208d0f59fefb5c4981973a782dc502e1c847a138289559d12be3c094ea216b.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/4b208d0f59fefb5c4981973a782dc502e1c847a138289559d12be3c094ea216b.jpg)

![4bc0651bc87a5c8fbbeae7f7d27dd3ff62dc8c5f0b033c5d3f02da761e3c0795.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/4bc0651bc87a5c8fbbeae7f7d27dd3ff62dc8c5f0b033c5d3f02da761e3c0795.jpg)

![7f76596afbd39db70de7c0de389900f43a59d7d9292f15a02ad7990600077030.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/7f76596afbd39db70de7c0de389900f43a59d7d9292f15a02ad7990600077030.jpg)

![b3d877e1c3f69f718681293bf6d55ff811f688065a9b227ec1d352158ddcbb2a.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/b3d877e1c3f69f718681293bf6d55ff811f688065a9b227ec1d352158ddcbb2a.jpg)

![da9f6bf5e50e6e195a8fc6218701549a1612e93df34ccf50669ebc0768dcbce5.jpg](../icml_results/1893_DSGCR_%20Decomposed%20Spectral%20Geometry-Aware%20Cross-Modal%20Semantic%20Representation%20for%203D%20Visual%20Groundin/tables/da9f6bf5e50e6e195a8fc6218701549a1612e93df34ccf50669ebc0768dcbce5.jpg)

<a id="paper-1869"></a>

## Approximation Error Upper and Lower Bounds for Hölder Class with Transformers


### Images

![8e2d054f38b7f402fc778c48dd99d99dabfe27dfb458b18ad2f682800265c995.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/images/8e2d054f38b7f402fc778c48dd99d99dabfe27dfb458b18ad2f682800265c995.jpg)

![f843ab8e0e956fa8dbf4272d2f58d54dd77c3651c356e4556f7cbc9fd5d32125.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/images/f843ab8e0e956fa8dbf4272d2f58d54dd77c3651c356e4556f7cbc9fd5d32125.jpg)

### Tables

![43f136f8226b54777a2cd9b4891e3243d0bf3692664938e6f65d9dc21c3b3b4e.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/tables/43f136f8226b54777a2cd9b4891e3243d0bf3692664938e6f65d9dc21c3b3b4e.jpg)

![64ad3ddb4543bd95166e3c84696eba87194fab4d2db58ceee746c588ee74e133.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/tables/64ad3ddb4543bd95166e3c84696eba87194fab4d2db58ceee746c588ee74e133.jpg)

![cbdc42e6623efe36ede665e630de249f566bfa2b88d5477538072f60da5022fa.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/tables/cbdc42e6623efe36ede665e630de249f566bfa2b88d5477538072f60da5022fa.jpg)

![f556efabce9db30aa412426dea063abe9793d509c6cb553ff7efa01be605bc47.jpg](../icml_results/1894_Approximation%20Error%20Upper%20and%20Lower%20Bounds%20for%20H%C3%B6lder%20Class%20with%20Transformers/tables/f556efabce9db30aa412426dea063abe9793d509c6cb553ff7efa01be605bc47.jpg)

<a id="paper-1870"></a>

## Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games


### Images

![0c1cb66596d1aa8ab63e8bf8939638ecae708a575f48751f08dcbd3192c68ba8.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/0c1cb66596d1aa8ab63e8bf8939638ecae708a575f48751f08dcbd3192c68ba8.jpg)

![50c5e6241396555aaed5c55f79230e923d2817e650ee03f91c1fdf13c542e504.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/50c5e6241396555aaed5c55f79230e923d2817e650ee03f91c1fdf13c542e504.jpg)

![5155dce960f0b0b10e4c81f28955e264f4bcf6a2bef8d3ed9ab015bf4cbbcd89.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/5155dce960f0b0b10e4c81f28955e264f4bcf6a2bef8d3ed9ab015bf4cbbcd89.jpg)

![520cf3c715ae6714f1b035f0177f6a1298299d0a2b893a2c2fa01ba110328759.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/520cf3c715ae6714f1b035f0177f6a1298299d0a2b893a2c2fa01ba110328759.jpg)

![7da824cd54229c645c7bdb7e30a95745b0920ed699a1cfcf94600dde4b508066.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/7da824cd54229c645c7bdb7e30a95745b0920ed699a1cfcf94600dde4b508066.jpg)

![ac1544dd155f360bcc7d5e41aeb5e9f69e93695fb4a303acbf9a13611f96d413.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/images/ac1544dd155f360bcc7d5e41aeb5e9f69e93695fb4a303acbf9a13611f96d413.jpg)

### Tables

![2759543ee01f7071468891428c6592f4f945e2aec2dc6ad93e8513d2b1399873.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/2759543ee01f7071468891428c6592f4f945e2aec2dc6ad93e8513d2b1399873.jpg)

![2cead186050a7d5ef5c5b18a071c1f5e182edfb4da475ce5033a087fefde06a1.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/2cead186050a7d5ef5c5b18a071c1f5e182edfb4da475ce5033a087fefde06a1.jpg)

![9ca8aa1448648f57a39249715e49435301a759da593c9461c6d5fd9af0a0e987.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/9ca8aa1448648f57a39249715e49435301a759da593c9461c6d5fd9af0a0e987.jpg)

![afff8e132e2b4e1bba8a626a9709750203f50e16e20c85d95ca0a50fbf677763.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/afff8e132e2b4e1bba8a626a9709750203f50e16e20c85d95ca0a50fbf677763.jpg)

![bb2022ba44f2e8fbf1029ff5ede63860734dc5cf539582227b236770a6824feb.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/bb2022ba44f2e8fbf1029ff5ede63860734dc5cf539582227b236770a6824feb.jpg)

![c4e1ada1928d7285a8bea662ea01002ec22d53c749884bb1448676ea47b4a2cd.jpg](../icml_results/1895_Strat-Reasoner_%20Reinforcing%20Strategic%20Reasoning%20of%20LLMs%20in%20Multi-Agent%20Games/tables/c4e1ada1928d7285a8bea662ea01002ec22d53c749884bb1448676ea47b4a2cd.jpg)

<a id="paper-1871"></a>

## HyMTRL: A Hybrid Multi-Task Reinforcement Learning Framework via Phased Policy Evolution


### Images

![00f90876d4b63074384ca8ecd180ac806a8a81e76238230c0d51e01f90febd43.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/00f90876d4b63074384ca8ecd180ac806a8a81e76238230c0d51e01f90febd43.jpg)

![16bb5b2a3597acd46300f342626152eb56fe5207457d4496116682c0d0a53db3.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/16bb5b2a3597acd46300f342626152eb56fe5207457d4496116682c0d0a53db3.jpg)

![20e63ea053adcc6799117bcfe7cfe7793409e496d19a31b577563958ffd3857e.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/20e63ea053adcc6799117bcfe7cfe7793409e496d19a31b577563958ffd3857e.jpg)

![2e8e50dac075deb00d6758d2e5d086764aebf85b636a1262b13782b7dc69dae2.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/2e8e50dac075deb00d6758d2e5d086764aebf85b636a1262b13782b7dc69dae2.jpg)

![3411bd8ed9b866b4d9e3d0a7afecb4df3e411fd45a6c0f626880a62634b90ab1.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/3411bd8ed9b866b4d9e3d0a7afecb4df3e411fd45a6c0f626880a62634b90ab1.jpg)

![c670287fa83c085b5c921d80e911daac10dd649b7717c1208791ad0e7315befb.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/c670287fa83c085b5c921d80e911daac10dd649b7717c1208791ad0e7315befb.jpg)

![e4b0a2f202d7048cf0daaf26cb67d1d1daa5b10fff0553259ad374bc45f1ce19.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/images/e4b0a2f202d7048cf0daaf26cb67d1d1daa5b10fff0553259ad374bc45f1ce19.jpg)

### Tables

![1c6d551d1988fb32b5d2a1c22261c8b985e0ab055dc0ee27ba00392bab3fa0d8.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/tables/1c6d551d1988fb32b5d2a1c22261c8b985e0ab055dc0ee27ba00392bab3fa0d8.jpg)

![2863b64176fd5194b1a9cf1e9c7254ad2f1877797021a07d711118604b4290e6.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/tables/2863b64176fd5194b1a9cf1e9c7254ad2f1877797021a07d711118604b4290e6.jpg)

![34a59fa26a49f37c019b77ef918858e22154a5d129589a449088b7a70df3a6ef.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/tables/34a59fa26a49f37c019b77ef918858e22154a5d129589a449088b7a70df3a6ef.jpg)

![b8affb6c0f8042b7a306f4f3d33236dd71a30b80bc7d47938ca81fc84c9107bd.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/tables/b8affb6c0f8042b7a306f4f3d33236dd71a30b80bc7d47938ca81fc84c9107bd.jpg)

![ca5bd6b3d2b41aab815bfe5973dece0254c6067d713ffdf1eef9aa40ec91bab4.jpg](../icml_results/1896_HyMTRL_%20A%20Hybrid%20Multi-Task%20Reinforcement%20Learning%20Framework%20via%20Phased%20Policy%20Evolution/tables/ca5bd6b3d2b41aab815bfe5973dece0254c6067d713ffdf1eef9aa40ec91bab4.jpg)

<a id="paper-1872"></a>

## Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs


### Images

![07801098c4bbb04ada5176ab2e04a777fff262a5b2dab1b436f943a37f084ae0.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/07801098c4bbb04ada5176ab2e04a777fff262a5b2dab1b436f943a37f084ae0.jpg)

![1e6c1494a3c54d0f1f67c40034e3c4f89b0e81ba3dcef08f2a0f50f0dda2558f.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/1e6c1494a3c54d0f1f67c40034e3c4f89b0e81ba3dcef08f2a0f50f0dda2558f.jpg)

![20283c55a35e0eb16ee5db17a6beb2e69c2afbabc0d08300330e96c467506f6a.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/20283c55a35e0eb16ee5db17a6beb2e69c2afbabc0d08300330e96c467506f6a.jpg)

![24aaa398f2e6ef468e375dd4e47541e556ab074a1a91062986421ac828315307.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/24aaa398f2e6ef468e375dd4e47541e556ab074a1a91062986421ac828315307.jpg)

![25bbffa785866ad5480f7d29ee03f0cbc3dc80030f58ac50b6436892a954a307.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/25bbffa785866ad5480f7d29ee03f0cbc3dc80030f58ac50b6436892a954a307.jpg)

![438dbfcec64657a723b7ef40c5c3a7809b8c81f25fba94331966191407ce78dc.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/438dbfcec64657a723b7ef40c5c3a7809b8c81f25fba94331966191407ce78dc.jpg)

![4b4e624e7e3edae3c92332c545fd1fdac593aff0d1af700b0b8e019e721519d6.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/4b4e624e7e3edae3c92332c545fd1fdac593aff0d1af700b0b8e019e721519d6.jpg)

![53bd9d6c480316bd7f0d77204255c30e96635dd53e01a77daaec39072ff95f8a.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/53bd9d6c480316bd7f0d77204255c30e96635dd53e01a77daaec39072ff95f8a.jpg)

![5ab4acb9b8a28132d3b619939adb7e23a30cf2bdd62484f9798c824a6ff32085.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/5ab4acb9b8a28132d3b619939adb7e23a30cf2bdd62484f9798c824a6ff32085.jpg)

![5ea2f9432d441e7a91c2607614d805ba7e87f5a90a8961e492c4c3b67993494d.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/5ea2f9432d441e7a91c2607614d805ba7e87f5a90a8961e492c4c3b67993494d.jpg)

![676f4a821d2137f3befac5b04ce8f9b344362015caf6765fbe6da279beb4cf20.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/676f4a821d2137f3befac5b04ce8f9b344362015caf6765fbe6da279beb4cf20.jpg)

![6c261c7ed592c27a6791ac13e1ce31b12917bfc4387292c7e3a6eaacf5ba62a5.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/6c261c7ed592c27a6791ac13e1ce31b12917bfc4387292c7e3a6eaacf5ba62a5.jpg)

![71745a88867a38917e996b54e20fec4a815a6e912a4caa9e01765e458a4a0f4f.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/71745a88867a38917e996b54e20fec4a815a6e912a4caa9e01765e458a4a0f4f.jpg)

![74478294c65d292c920114f2c2ba8b91acbc56752cd91e6b3a4ef4471daea0d7.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/74478294c65d292c920114f2c2ba8b91acbc56752cd91e6b3a4ef4471daea0d7.jpg)

![79a435b9b8d32665f1c4afd02f137444f28ec938b1c1021fd3da495615b0e238.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/79a435b9b8d32665f1c4afd02f137444f28ec938b1c1021fd3da495615b0e238.jpg)

![7ca71b9c293a37ebdbfb1a96697d7d7dec600e21b84910e69db1ed7cd7fe3a6d.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/7ca71b9c293a37ebdbfb1a96697d7d7dec600e21b84910e69db1ed7cd7fe3a6d.jpg)

![85eb800c4fddb8c162dd83773424ac06973e434e611cc73a9bdeb5d601d71035.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/85eb800c4fddb8c162dd83773424ac06973e434e611cc73a9bdeb5d601d71035.jpg)

![881aa74b900d1727d99cded9bb5393ab8f48b40b448d44a6755006c5aef0ef19.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/881aa74b900d1727d99cded9bb5393ab8f48b40b448d44a6755006c5aef0ef19.jpg)

![8c9c5de7bebfe0e678e7d8fc24f834ee8e26f4205e6b60f3d48069941d7a985f.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/8c9c5de7bebfe0e678e7d8fc24f834ee8e26f4205e6b60f3d48069941d7a985f.jpg)

![9033970c944b0fa9f00fa24e185ab90f80473c15d0dc6d476a069a381c86c422.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/9033970c944b0fa9f00fa24e185ab90f80473c15d0dc6d476a069a381c86c422.jpg)

![98664be3364e38072892fe122d467e5ced2e628362d7cefa419d545c79740953.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/98664be3364e38072892fe122d467e5ced2e628362d7cefa419d545c79740953.jpg)

![a695df7fd147809bbd3c983a377428d3ceffcbb2ce7c5a2c9a89a0fe7d65bd8e.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/a695df7fd147809bbd3c983a377428d3ceffcbb2ce7c5a2c9a89a0fe7d65bd8e.jpg)

![a834867edb67f15f9507f8cfb20ff5ea0fdbc9300420f1f52e515634aae19f5c.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/a834867edb67f15f9507f8cfb20ff5ea0fdbc9300420f1f52e515634aae19f5c.jpg)

![c3a49076b5172dfe4489ca5ba20619c7bd6e6595127059f77d9dd292529b1daa.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/c3a49076b5172dfe4489ca5ba20619c7bd6e6595127059f77d9dd292529b1daa.jpg)

![d67ffc9afaf2b77b9f591143b889e0ce1abe5ddbe629bd7cb2dc08b27367d7c7.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/d67ffc9afaf2b77b9f591143b889e0ce1abe5ddbe629bd7cb2dc08b27367d7c7.jpg)

![e8a433a3c0d0de4ff550aea8ee21bccb52ff4e7cf802872546462b3600abb8a7.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/images/e8a433a3c0d0de4ff550aea8ee21bccb52ff4e7cf802872546462b3600abb8a7.jpg)

### Tables

![0fddaf1398365b4d350bf3263c704682b6b8b768a7b87c7d54c7901c132c0c64.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/0fddaf1398365b4d350bf3263c704682b6b8b768a7b87c7d54c7901c132c0c64.jpg)

![1056d87c7c12d14e8545ca810f877b7bdddafa2716ca31388c0e78b9d4058eb9.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/1056d87c7c12d14e8545ca810f877b7bdddafa2716ca31388c0e78b9d4058eb9.jpg)

![130a85cec59bb21a44d136df3a589a0e372bed92fdbbed1e2a95cc505101e051.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/130a85cec59bb21a44d136df3a589a0e372bed92fdbbed1e2a95cc505101e051.jpg)

![23e3558dd31705725170b7e73dd1fd4902d69c16dca0692dfcdad4933dba38c8.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/23e3558dd31705725170b7e73dd1fd4902d69c16dca0692dfcdad4933dba38c8.jpg)

![2fa68a00c2e6a096024e79b8d15c980dce2939f1674d30dc9aa2821bcb22e9ec.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/2fa68a00c2e6a096024e79b8d15c980dce2939f1674d30dc9aa2821bcb22e9ec.jpg)

![308caa69bca07b65f44d05416ba7d459e869856af638ccebf1a863439dbd9811.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/308caa69bca07b65f44d05416ba7d459e869856af638ccebf1a863439dbd9811.jpg)

![32471798ec30be5cce682857372d00443c4bae2c976cda611117efdaa95cf81e.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/32471798ec30be5cce682857372d00443c4bae2c976cda611117efdaa95cf81e.jpg)

![63fc4c09e61fbd1e3d99bcf451d7f8b087b6e525a1e47548aa53f05182da0778.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/63fc4c09e61fbd1e3d99bcf451d7f8b087b6e525a1e47548aa53f05182da0778.jpg)

![6fb0c08ca7e78d1d807586e5c15bf85b822322a5753305e2d86e2e03dbb3d9b1.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/6fb0c08ca7e78d1d807586e5c15bf85b822322a5753305e2d86e2e03dbb3d9b1.jpg)

![7088fa5883e27a6ed77a30c4f0591d5ffe7ac8b9730ce9b511c52ad5eb2e6fa0.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/7088fa5883e27a6ed77a30c4f0591d5ffe7ac8b9730ce9b511c52ad5eb2e6fa0.jpg)

![a5a7a521e872081fa51667b5a876ba0a5a0697c5797234706d6aa4eab0ea28cc.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/a5a7a521e872081fa51667b5a876ba0a5a0697c5797234706d6aa4eab0ea28cc.jpg)

![c74aee0e714b798ec1a6bb8933dc30383aeaf7a255cb48f5dc534e15b6ea79b1.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/c74aee0e714b798ec1a6bb8933dc30383aeaf7a255cb48f5dc534e15b6ea79b1.jpg)

![ca8e5f8eae6d6178398eb62b084148ff50003f3a1d27f2fb133419bcfc218628.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/ca8e5f8eae6d6178398eb62b084148ff50003f3a1d27f2fb133419bcfc218628.jpg)

![d2097fde9a00deea70a2146b6d15b4f93bd1a2c753172a31a52b8893a32ad9e1.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/d2097fde9a00deea70a2146b6d15b4f93bd1a2c753172a31a52b8893a32ad9e1.jpg)

![d691dde5abfa58fd0502f577bf11d5e72c40cb608c8c0a976762247cf5d9f7ea.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/d691dde5abfa58fd0502f577bf11d5e72c40cb608c8c0a976762247cf5d9f7ea.jpg)

![d85b3bf21758804be05e708110cde3005d5270e8a9e3ee4d278480abb24e9f12.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/d85b3bf21758804be05e708110cde3005d5270e8a9e3ee4d278480abb24e9f12.jpg)

![dc88112cb5618ef27c8626c08ca36db314b44d97642e1251a431b669b6004da1.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/dc88112cb5618ef27c8626c08ca36db314b44d97642e1251a431b669b6004da1.jpg)

![e3be50a6ed0dd9bdde413bbfa2efda22d3318e628cbde1968bb0a89f4111720b.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/e3be50a6ed0dd9bdde413bbfa2efda22d3318e628cbde1968bb0a89f4111720b.jpg)

![e979dd51169cb8583350d80a639773dd86349533bed331da5ffa30347cb6bca9.jpg](../icml_results/1897_Hermite-NGP_%20Gradient-Augmented%20Hash%20Encoding%20for%20Learning%20PDEs/tables/e979dd51169cb8583350d80a639773dd86349533bed331da5ffa30347cb6bca9.jpg)

<a id="paper-1873"></a>

## Can LLMs Reason Structurally? Benchmarking via the lens of Data Structures


### Images

![1b42678d18c95fc332271a0884f17fd6fb7f965c3d7d2c0e6b96733ee5ee7064.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/1b42678d18c95fc332271a0884f17fd6fb7f965c3d7d2c0e6b96733ee5ee7064.jpg)

![1dcb6b2ab10387f8ffcfa0ceab3c432f5a9836806a591c0588f81a3252b2c76a.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/1dcb6b2ab10387f8ffcfa0ceab3c432f5a9836806a591c0588f81a3252b2c76a.jpg)

![29c771f8e26f3de28a2221dd33bd8f54cf25603cf3eaabacd2c75d244564711e.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/29c771f8e26f3de28a2221dd33bd8f54cf25603cf3eaabacd2c75d244564711e.jpg)

![5361cf5b16936c335739142c331aca9ea72af2f3445efdd3afc6b6874de977c9.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/5361cf5b16936c335739142c331aca9ea72af2f3445efdd3afc6b6874de977c9.jpg)

![78bcda42b056241ca58f2d335211d4f843f6ace5f13381afd6cffd5a967a9e74.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/78bcda42b056241ca58f2d335211d4f843f6ace5f13381afd6cffd5a967a9e74.jpg)

![8e263b5006ea4929aa0e2a28a59778104151e331ec5c44dd73bce914333a5f4d.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/8e263b5006ea4929aa0e2a28a59778104151e331ec5c44dd73bce914333a5f4d.jpg)

![8e530575faaf8cdf6c6d7999cc81490bced7852c79f396b08b068bc1b0a430b4.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/8e530575faaf8cdf6c6d7999cc81490bced7852c79f396b08b068bc1b0a430b4.jpg)

![98b06d95ade7abbcbb6b58f872af281a8455d05e1a2243b7289bc124ebb0c2be.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/98b06d95ade7abbcbb6b58f872af281a8455d05e1a2243b7289bc124ebb0c2be.jpg)

![b4ab1e277bc023b26e91cef5bc6d5f7429fe4bcf447bf3b268a03c6e5702eaca.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/b4ab1e277bc023b26e91cef5bc6d5f7429fe4bcf447bf3b268a03c6e5702eaca.jpg)

![d863f25b8af47e5c9bf9b7ae005b57b818ec2686f2bbc5b0b8c0a671e99b7aef.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/images/d863f25b8af47e5c9bf9b7ae005b57b818ec2686f2bbc5b0b8c0a671e99b7aef.jpg)

### Tables

![00300e54c97b2b46ef0c4c45567028365f26c56d16180eb9c0b12ac5c3b12c98.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/00300e54c97b2b46ef0c4c45567028365f26c56d16180eb9c0b12ac5c3b12c98.jpg)

![054d0c7c427e203a6dbce21bb114f7624b5d986d60219dcd85d35669bcecbfac.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/054d0c7c427e203a6dbce21bb114f7624b5d986d60219dcd85d35669bcecbfac.jpg)

![168f651d07835843d04fcc00a75292ed899901d66fa8c0648d850af6d802d6f3.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/168f651d07835843d04fcc00a75292ed899901d66fa8c0648d850af6d802d6f3.jpg)

![191a142b52eee7c943b4e64f145bee35b921b2063dcdfef2b87bfc3d3240f858.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/191a142b52eee7c943b4e64f145bee35b921b2063dcdfef2b87bfc3d3240f858.jpg)

![1a18546f5e9d152023e040fe29d242959581d1b83cf5b737a0485199afe3c29b.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/1a18546f5e9d152023e040fe29d242959581d1b83cf5b737a0485199afe3c29b.jpg)

![1d21cad8597d9a8c6fdda206e7a3144ba99ba18c2cb2e43f137d53efd3990a12.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/1d21cad8597d9a8c6fdda206e7a3144ba99ba18c2cb2e43f137d53efd3990a12.jpg)

![3328f033fd0b55eedd124072622c4bb6579b2ce28dc81cfcf6ffb7303673b8e8.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/3328f033fd0b55eedd124072622c4bb6579b2ce28dc81cfcf6ffb7303673b8e8.jpg)

![33b89024d36a69574ea224d1ab785a0d36e99cb874535dfe96b0365bf55e192f.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/33b89024d36a69574ea224d1ab785a0d36e99cb874535dfe96b0365bf55e192f.jpg)

![4a32935a0a50cffa0cf5a5ef0622bf0589fdd06bcbe32f1b333d24e318f7704d.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/4a32935a0a50cffa0cf5a5ef0622bf0589fdd06bcbe32f1b333d24e318f7704d.jpg)

![4e4b52bec090a1ae2dab8a74ca073fe6421b454c80bdfdcb85a0a6a9f4788a92.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/4e4b52bec090a1ae2dab8a74ca073fe6421b454c80bdfdcb85a0a6a9f4788a92.jpg)

![4e8017811fb3ecec70f0ebf2828d20e155c324e3f47af08e2170a2e85f535cc9.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/4e8017811fb3ecec70f0ebf2828d20e155c324e3f47af08e2170a2e85f535cc9.jpg)

![5eb07c6cab5c905c38ee8e6dbb4d6144d37c73026217ebfbccd544f60c7db140.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/5eb07c6cab5c905c38ee8e6dbb4d6144d37c73026217ebfbccd544f60c7db140.jpg)

![6b5dd9fc5fd6116e57d8be5334f91cd6249f92d7cde9bdc52b1f829ff376b34f.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/6b5dd9fc5fd6116e57d8be5334f91cd6249f92d7cde9bdc52b1f829ff376b34f.jpg)

![7181d5cb431392c5008a2f65c7625630ded7b13b1ef84c05d34a6d8b30c9180a.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/7181d5cb431392c5008a2f65c7625630ded7b13b1ef84c05d34a6d8b30c9180a.jpg)

![75a271fac249219a1bfe807d8a194ed524b043def66d2b1d64d19ca9ad1e8e65.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/75a271fac249219a1bfe807d8a194ed524b043def66d2b1d64d19ca9ad1e8e65.jpg)

![7c937d6a352298c1255689f0eb0d124d3c29eb9970e2c8cec74cfd68b90a8283.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/7c937d6a352298c1255689f0eb0d124d3c29eb9970e2c8cec74cfd68b90a8283.jpg)

![8263e38a6eabf8fb021c42b2d26b3b61db8a79bb3569a934c69cb3dc14600792.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/8263e38a6eabf8fb021c42b2d26b3b61db8a79bb3569a934c69cb3dc14600792.jpg)

![83ccbdf5d3de53b4128d7813cb58c9f76d6336f158c3b60ee29665eca485a51b.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/83ccbdf5d3de53b4128d7813cb58c9f76d6336f158c3b60ee29665eca485a51b.jpg)

![97775509dfa69d2660d110d7eaac8263b60be388f3af3742b9df5d96d4a341ce.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/97775509dfa69d2660d110d7eaac8263b60be388f3af3742b9df5d96d4a341ce.jpg)

![9edfbee8facc56d53840776bf73166da91f0deee3ba45814bbb5d89a8b1f4a48.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/9edfbee8facc56d53840776bf73166da91f0deee3ba45814bbb5d89a8b1f4a48.jpg)

![a320b20a8bac271bd2a4d2e1a4a5cf3ec86054eb45b7d9855c77c07dc64780e2.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/a320b20a8bac271bd2a4d2e1a4a5cf3ec86054eb45b7d9855c77c07dc64780e2.jpg)

![a40eb52b57479265065957155c21aa08fb07fa2223661b653806c28d6d3a742d.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/a40eb52b57479265065957155c21aa08fb07fa2223661b653806c28d6d3a742d.jpg)

![a5cdc9d62024411c6ff0a92c56250447f933ac3d2a312a43b1e06284ae7307d8.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/a5cdc9d62024411c6ff0a92c56250447f933ac3d2a312a43b1e06284ae7307d8.jpg)

![a5ff266e2b5607b03f404fe721ec112117b3df64184cffcddf0e1c144eb4bcb7.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/a5ff266e2b5607b03f404fe721ec112117b3df64184cffcddf0e1c144eb4bcb7.jpg)

![a7faf3e15d57b0c3ef2647b8a877e17a52a896124877ecfa9fa2d05085fdc820.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/a7faf3e15d57b0c3ef2647b8a877e17a52a896124877ecfa9fa2d05085fdc820.jpg)

![b4dae64f9a7c06510f6ba3d042306902e67b7c51ffc92822d304b47d597baa38.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/b4dae64f9a7c06510f6ba3d042306902e67b7c51ffc92822d304b47d597baa38.jpg)

![b8022c38afb8fa1b00242d4aaf44b5cb9862c8a771bee22ed5b1cb2c7814e054.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/b8022c38afb8fa1b00242d4aaf44b5cb9862c8a771bee22ed5b1cb2c7814e054.jpg)

![ba9510469e9a871772d54b7cc6264fdf898fea2d3f60d4113207ec491dec0fff.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/ba9510469e9a871772d54b7cc6264fdf898fea2d3f60d4113207ec491dec0fff.jpg)

![bb6673bebe31128f4728fc4bc89bdab232cd61208a6f700cd8d2a15c28c3bda0.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/bb6673bebe31128f4728fc4bc89bdab232cd61208a6f700cd8d2a15c28c3bda0.jpg)

![bd9aa3aceda2118dc007de3fc2017b8f36c29a719cdb4fe64529dca5d8719c0c.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/bd9aa3aceda2118dc007de3fc2017b8f36c29a719cdb4fe64529dca5d8719c0c.jpg)

![c6f8b2bfb72993927f7c42b3d40efa6824c37e936d4795aab484d38c9598fbbc.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/c6f8b2bfb72993927f7c42b3d40efa6824c37e936d4795aab484d38c9598fbbc.jpg)

![cf650a40534fbab494f984269a4c8a0680f6c85b9d9c254aefc2ea2a2dabba10.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/cf650a40534fbab494f984269a4c8a0680f6c85b9d9c254aefc2ea2a2dabba10.jpg)

![d1e3d0583934fb50f0644d1a578ecef098ae0a557684794e0b4bd665fd3fdf34.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/d1e3d0583934fb50f0644d1a578ecef098ae0a557684794e0b4bd665fd3fdf34.jpg)

![d2fe3c1a599b4e2b137ebe932d9a76cc4124927a4cd418cb3608eebaeaeb8954.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/d2fe3c1a599b4e2b137ebe932d9a76cc4124927a4cd418cb3608eebaeaeb8954.jpg)

![db87de9d29710757cfadbc5c8b1e3215f139b7098dd047dc5d9969c08b4213c1.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/db87de9d29710757cfadbc5c8b1e3215f139b7098dd047dc5d9969c08b4213c1.jpg)

![e82d0817bb52688182fa3f3fdc46fb46b3f12042bebfc3caf2d5622ad8210eec.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/e82d0817bb52688182fa3f3fdc46fb46b3f12042bebfc3caf2d5622ad8210eec.jpg)

![e9ab16606abc685eeaa25b4a57204a200edd1678c8c4068109c12366aceabc3b.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/e9ab16606abc685eeaa25b4a57204a200edd1678c8c4068109c12366aceabc3b.jpg)

![ec4c843d3bb178254eeb1615de1c4dbf23fe7577c27644ff226f3a7fa9e4b558.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/ec4c843d3bb178254eeb1615de1c4dbf23fe7577c27644ff226f3a7fa9e4b558.jpg)

![ec895e59baf9ea8ba7ffce7f0231d6f65d723e54d4beb834395ec7d31145b611.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/ec895e59baf9ea8ba7ffce7f0231d6f65d723e54d4beb834395ec7d31145b611.jpg)

![f999329f53e7fcc8eff37c79bbf680410eb3ecb1023b1e6149a4a92c320d87bb.jpg](../icml_results/1898_Can%20LLMs%20Reason%20Structurally_%20Benchmarking%20via%20the%20lens%20of%20Data%20Structures/tables/f999329f53e7fcc8eff37c79bbf680410eb3ecb1023b1e6149a4a92c320d87bb.jpg)

<a id="paper-1874"></a>

## SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?


### Images

![2826d9fc6ecae23901e09a2078d2acc00a5528bb7a3f0f41df59b0cd8eb5a665.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/2826d9fc6ecae23901e09a2078d2acc00a5528bb7a3f0f41df59b0cd8eb5a665.jpg)

![3f1a6016fda078edc336488d01eb45201b686aed3138f28f8722d141a4778752.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/3f1a6016fda078edc336488d01eb45201b686aed3138f28f8722d141a4778752.jpg)

![5d1e75d4db96ff6b0493428c500798806ac6ffb9e742eebf9bb16e8965a9416d.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/5d1e75d4db96ff6b0493428c500798806ac6ffb9e742eebf9bb16e8965a9416d.jpg)

![7b168a25785ba9af67bfcc133d1094db1f26c860b500998d1d9e6fdfea3ce55f.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/7b168a25785ba9af67bfcc133d1094db1f26c860b500998d1d9e6fdfea3ce55f.jpg)

![7bd2c9059add77b6497d3a3876c6afc5cf6994d21877ecea0d495e82ec37f43f.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/7bd2c9059add77b6497d3a3876c6afc5cf6994d21877ecea0d495e82ec37f43f.jpg)

![8ec06433bd638bab4a642b09694b6ba9934ee12f9d62e360e2d8d90a1bca502f.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/8ec06433bd638bab4a642b09694b6ba9934ee12f9d62e360e2d8d90a1bca502f.jpg)

![8f0739e91018e815fa96b9e83a30c1560c6f96f4bd39ef30aea56370adbdc713.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/8f0739e91018e815fa96b9e83a30c1560c6f96f4bd39ef30aea56370adbdc713.jpg)

![90d265771a1fce79614452a5d0ca6149b32833955b09e9c9deca6ca7fc1a631a.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/90d265771a1fce79614452a5d0ca6149b32833955b09e9c9deca6ca7fc1a631a.jpg)

![bc6c35ae079abb2aef3389f890ea268093cb0ab7762fe48db9a3c78737f72ef9.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/bc6c35ae079abb2aef3389f890ea268093cb0ab7762fe48db9a3c78737f72ef9.jpg)

![f0d03b62a55592accd78547ac5adcb737650741db5abb9913aa1f30fa857990b.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/f0d03b62a55592accd78547ac5adcb737650741db5abb9913aa1f30fa857990b.jpg)

![f8df4e30d96f057ad3387650a2cf520b47455a659a94bd162b216df2bb1f56ed.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/f8df4e30d96f057ad3387650a2cf520b47455a659a94bd162b216df2bb1f56ed.jpg)

![f9d69305f4acdaf6b5aa63548bbb948c1111a92e52e0d9b066a4e29fb6e64df8.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/images/f9d69305f4acdaf6b5aa63548bbb948c1111a92e52e0d9b066a4e29fb6e64df8.jpg)

### Tables

![00b92147a014a44297f0f5dd500d21f3c3ea173a0fb4dbcc60637f47fc30208c.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/00b92147a014a44297f0f5dd500d21f3c3ea173a0fb4dbcc60637f47fc30208c.jpg)

![1fae2939cb115fbc4258a99c57b79801cb8a6548392a82408dd37651ce852118.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/1fae2939cb115fbc4258a99c57b79801cb8a6548392a82408dd37651ce852118.jpg)

![4d207d287aec670cd85086b880dd45da24edfb7bf7a8698a93674ac31d2df8b5.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/4d207d287aec670cd85086b880dd45da24edfb7bf7a8698a93674ac31d2df8b5.jpg)

![8bd9a843bb142aa30de277b2ad201d7152314b57399eefb8bf9708a0f6954b55.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/8bd9a843bb142aa30de277b2ad201d7152314b57399eefb8bf9708a0f6954b55.jpg)

![a5fb6476f0f55a05728209295857146ccb971eb7c786f028882a6c68041866bc.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/a5fb6476f0f55a05728209295857146ccb971eb7c786f028882a6c68041866bc.jpg)

![c1e9b43ed30534d53d2f84f9581ab098b82f2316df7410b47d4a79222d4af091.jpg](../icml_results/1899_SWE-Perf_%20Can%20Language%20Models%20Optimize%20Code%20Performance%20on%20Real-World%20Repositories_/tables/c1e9b43ed30534d53d2f84f9581ab098b82f2316df7410b47d4a79222d4af091.jpg)

<a id="paper-1875"></a>

## Doubly Regularized Markov Decision Processes for Robust Reinforcement Learning


### Images

![63aab52145132d1630190cd4cfd1e996aff11ab0ea82568e2a5ca4e787337127.jpg](../icml_results/1900_Doubly%20Regularized%20Markov%20Decision%20Processes%20for%20Robust%20Reinforcement%20Learning/images/63aab52145132d1630190cd4cfd1e996aff11ab0ea82568e2a5ca4e787337127.jpg)

![ed4cb5542b6ae4069eace7e0cc33481337c218e0cb87fc8e605a83124d14cf81.jpg](../icml_results/1900_Doubly%20Regularized%20Markov%20Decision%20Processes%20for%20Robust%20Reinforcement%20Learning/images/ed4cb5542b6ae4069eace7e0cc33481337c218e0cb87fc8e605a83124d14cf81.jpg)

### Tables

![3ff633d7d64a7476eb7577cfb240c151c91be4497f9696eee4044ec2f7e558ab.jpg](../icml_results/1900_Doubly%20Regularized%20Markov%20Decision%20Processes%20for%20Robust%20Reinforcement%20Learning/tables/3ff633d7d64a7476eb7577cfb240c151c91be4497f9696eee4044ec2f7e558ab.jpg)

![690d8b43b94c2f56918be7ed535b47219a0b2b08c22a409b7f6237bad051f06a.jpg](../icml_results/1900_Doubly%20Regularized%20Markov%20Decision%20Processes%20for%20Robust%20Reinforcement%20Learning/tables/690d8b43b94c2f56918be7ed535b47219a0b2b08c22a409b7f6237bad051f06a.jpg)

![6e06b87ab8fe4eb9b475e31f0acab0177d4a75f5e6c378e87b786ff3c146eb12.jpg](../icml_results/1900_Doubly%20Regularized%20Markov%20Decision%20Processes%20for%20Robust%20Reinforcement%20Learning/tables/6e06b87ab8fe4eb9b475e31f0acab0177d4a75f5e6c378e87b786ff3c146eb12.jpg)

<a id="paper-1876"></a>

## Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference


### Images

![06ffc714fc2347d552c36680c0b0060138e008d9c6c74b96064154b3f88c70f1.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/06ffc714fc2347d552c36680c0b0060138e008d9c6c74b96064154b3f88c70f1.jpg)

![1bfc43d65109b852be1e096b84504a6f5533dfe2f87628187ced0f27790b5568.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/1bfc43d65109b852be1e096b84504a6f5533dfe2f87628187ced0f27790b5568.jpg)

![2e666477cfed69a4a086805b62cab8bde119cc59f69e9e642aec10f70ecebcde.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/2e666477cfed69a4a086805b62cab8bde119cc59f69e9e642aec10f70ecebcde.jpg)

![39f8bb8979e2059a391acaa1b722d42f2c05fa86edb72b2e6e5034b2809744b2.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/39f8bb8979e2059a391acaa1b722d42f2c05fa86edb72b2e6e5034b2809744b2.jpg)

![6008f7f83b91a88e20a6551f97fc8eb245069ce94d0e9df3456b238563eb054d.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/6008f7f83b91a88e20a6551f97fc8eb245069ce94d0e9df3456b238563eb054d.jpg)

![6a157cbdf2a25999e9fd44d7fef780b02f027646701a503f0a380c7a6ac47628.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/6a157cbdf2a25999e9fd44d7fef780b02f027646701a503f0a380c7a6ac47628.jpg)

![74d096c4a0a734ee89beb17a9de99d7863f6f4dfb4a38fd0b222b2fe47cc5c0f.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/74d096c4a0a734ee89beb17a9de99d7863f6f4dfb4a38fd0b222b2fe47cc5c0f.jpg)

![765b4b92a66edfead8d5438bd830a736010b024e13e500a2968fe9a94f69085e.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/765b4b92a66edfead8d5438bd830a736010b024e13e500a2968fe9a94f69085e.jpg)

![801c3059f5cddacd3a09493b685d21bb5f3aada04aaed1697d7d502a9a99448b.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/801c3059f5cddacd3a09493b685d21bb5f3aada04aaed1697d7d502a9a99448b.jpg)

![955ea723256522e82f67cafac09e9bd01b78f7754f27be0136e9d43520fea7b1.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/955ea723256522e82f67cafac09e9bd01b78f7754f27be0136e9d43520fea7b1.jpg)

![9ebdc15702eba759cd945563bab420ddb018650adb3e1ea54185aa1bb94f503f.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/9ebdc15702eba759cd945563bab420ddb018650adb3e1ea54185aa1bb94f503f.jpg)

![a250aedf92bb2f9bdbd273ebd5d0866dba96241a8042b919f624940b7f0dc818.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/a250aedf92bb2f9bdbd273ebd5d0866dba96241a8042b919f624940b7f0dc818.jpg)

![c0f4a0b2fb107af9b3fa1b67858eaf097fb1932d16723d468053ef9d77a4793c.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/c0f4a0b2fb107af9b3fa1b67858eaf097fb1932d16723d468053ef9d77a4793c.jpg)

![c5779947ccee0b0542828adefb7e4fa2deae14bff15b7dded54de6f5d1c43a97.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/c5779947ccee0b0542828adefb7e4fa2deae14bff15b7dded54de6f5d1c43a97.jpg)

![c6b13d85ce4fbf66419e6ad6f60fcbf7283e7f66acd3cbf2fc6122a61691303c.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/c6b13d85ce4fbf66419e6ad6f60fcbf7283e7f66acd3cbf2fc6122a61691303c.jpg)

![cb4204c7a77099676c8d7a17adbacfc539b4b9d2917fc59f1f8cd12c86586f44.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/cb4204c7a77099676c8d7a17adbacfc539b4b9d2917fc59f1f8cd12c86586f44.jpg)

![ce5d530d1b501abc2c79028e3e16e9dfea26753190ff4add535cb579dce2d380.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/ce5d530d1b501abc2c79028e3e16e9dfea26753190ff4add535cb579dce2d380.jpg)

![d155ee9723472bb2eb5c91b37207765e8e644667c4a7984635d3389ae564b870.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/d155ee9723472bb2eb5c91b37207765e8e644667c4a7984635d3389ae564b870.jpg)

![d48fae55c5ddf354d0297ef341b141db064dae5ba71cabeefcf7ebddc35e9472.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/d48fae55c5ddf354d0297ef341b141db064dae5ba71cabeefcf7ebddc35e9472.jpg)

![e05456561cc3df20a648183136e993b2ad146ca04839124ba15c972fcb1d9754.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/e05456561cc3df20a648183136e993b2ad146ca04839124ba15c972fcb1d9754.jpg)

![e1748a76c984dbd97e1d57efa7c6708f3858ca882664fecfd29519141dd96406.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/e1748a76c984dbd97e1d57efa7c6708f3858ca882664fecfd29519141dd96406.jpg)

![e19c1306e3ff6b83f5d5cb20df463715c01927f77bb284ac37bf355e6c19753d.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/e19c1306e3ff6b83f5d5cb20df463715c01927f77bb284ac37bf355e6c19753d.jpg)

![e3cdff5c0ade51645f9f53f31bd9587e4623488e07b08233b48b4bdc4c0e005c.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/e3cdff5c0ade51645f9f53f31bd9587e4623488e07b08233b48b4bdc4c0e005c.jpg)

![e6436cee0892b21d912f276c9cfff7595166fa9929e1fbcab0e93b1a113dc912.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/e6436cee0892b21d912f276c9cfff7595166fa9929e1fbcab0e93b1a113dc912.jpg)

![f8751c0d2ec07a11fadc98defc70fbf4e94b9610cf945f6f5ea9b7290458d28c.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/f8751c0d2ec07a11fadc98defc70fbf4e94b9610cf945f6f5ea9b7290458d28c.jpg)

![ff99766a72f5548d0b6e5cf2e693ed87bd111ecddeb512cd14b5b74937245cf0.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/images/ff99766a72f5548d0b6e5cf2e693ed87bd111ecddeb512cd14b5b74937245cf0.jpg)

### Tables

![3247219dd4dabbc9ac5376a10ff50e640189f6041163bde2da7f03275416253a.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/tables/3247219dd4dabbc9ac5376a10ff50e640189f6041163bde2da7f03275416253a.jpg)

![352f4fbedc46222da03bffbfc5ce280a188bbc6732bdb6b2baa8fa70c2f997d2.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/tables/352f4fbedc46222da03bffbfc5ce280a188bbc6732bdb6b2baa8fa70c2f997d2.jpg)

![6eb62895404d5fba260d42b436f8b2449785eda7fb84964ea8efd95a7a6d7775.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/tables/6eb62895404d5fba260d42b436f8b2449785eda7fb84964ea8efd95a7a6d7775.jpg)

![d1e8d42eeef2c7fbafc429582c5e0aadd2af3d690985750caa2c27f1e5bfa8d7.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/tables/d1e8d42eeef2c7fbafc429582c5e0aadd2af3d690985750caa2c27f1e5bfa8d7.jpg)

![d9e8dc6cba3e33cbf265cc3024fd469032d9670e6ab176b3784b217bc909cf86.jpg](../icml_results/1901_Understand%20and%20Accelerate%20Memory%20Processing%20Pipeline%20for%20Large%20Language%20Model%20Inference/tables/d9e8dc6cba3e33cbf265cc3024fd469032d9670e6ab176b3784b217bc909cf86.jpg)

<a id="paper-1877"></a>

## DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models


### Images

![0a28c008844f61f01907e58ea0f2bf20b00f9e84cf19d12f9bdaaed1c90be55e.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/0a28c008844f61f01907e58ea0f2bf20b00f9e84cf19d12f9bdaaed1c90be55e.jpg)

![199f51c92fefe6cbc1fc69dc2cd0d655abd5897993f9385c381f14776a2c7327.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/199f51c92fefe6cbc1fc69dc2cd0d655abd5897993f9385c381f14776a2c7327.jpg)

![1a064969caa5ca97a5d968f3595c1dd51d39fc96732093bb9e274a57118f1de3.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/1a064969caa5ca97a5d968f3595c1dd51d39fc96732093bb9e274a57118f1de3.jpg)

![1abd35ee6196b22ae1a8302e0201d1be03bf0d74493fb46778e9031780e8edb8.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/1abd35ee6196b22ae1a8302e0201d1be03bf0d74493fb46778e9031780e8edb8.jpg)

![239cb33ae59818f67db419bea2f422ef02a9fcf172d07a716dd8910971f9d1b9.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/239cb33ae59818f67db419bea2f422ef02a9fcf172d07a716dd8910971f9d1b9.jpg)

![32ebcf0fed89d8526844910b951756fb5ba3ee8b5036fda3b18835c488392673.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/32ebcf0fed89d8526844910b951756fb5ba3ee8b5036fda3b18835c488392673.jpg)

![3ae97fad15cf35adc26f5b13a1829b5dd3d12672e2d62a9c93ed5447a9653174.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/3ae97fad15cf35adc26f5b13a1829b5dd3d12672e2d62a9c93ed5447a9653174.jpg)

![3cb129b99f49c2d3b0c816ad855e7da7c62e6ac079af31a9f292789ee0cf9f23.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/3cb129b99f49c2d3b0c816ad855e7da7c62e6ac079af31a9f292789ee0cf9f23.jpg)

![4573fe7d2089b6da8a054c1c464677efa5195f698f281a0658eaacc22d7fb435.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/4573fe7d2089b6da8a054c1c464677efa5195f698f281a0658eaacc22d7fb435.jpg)

![583a8bd78917983834fa1b50c87e9d8316ae60a116d37ccaabaaaa0f0904fb7d.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/583a8bd78917983834fa1b50c87e9d8316ae60a116d37ccaabaaaa0f0904fb7d.jpg)

![5dc837165101e46e3d6c2c0147d729e6a050ee21672c334427f467cd5eea0a43.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/5dc837165101e46e3d6c2c0147d729e6a050ee21672c334427f467cd5eea0a43.jpg)

![5dffacfe5ad03f147edb07ca568d3ebc6acc3e548becd3d8209a6893a368962b.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/5dffacfe5ad03f147edb07ca568d3ebc6acc3e548becd3d8209a6893a368962b.jpg)

![5fb8223836266aeaa70d37af89537511228bca16deb6b8a72c43aad469d281d2.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/5fb8223836266aeaa70d37af89537511228bca16deb6b8a72c43aad469d281d2.jpg)

![6a467471b59eea11c8c999872846038a7a5f9ef94acaa1ab1bba623c7f200b2a.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/6a467471b59eea11c8c999872846038a7a5f9ef94acaa1ab1bba623c7f200b2a.jpg)

![74d6e5eafa51495819248a22ff4c24ba6381dda5211286cb784f9c2301800e0c.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/74d6e5eafa51495819248a22ff4c24ba6381dda5211286cb784f9c2301800e0c.jpg)

![770841abcb59a4ff39b39763b4335fa78cc178a988d6538ba472a3485c7cd0a3.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/770841abcb59a4ff39b39763b4335fa78cc178a988d6538ba472a3485c7cd0a3.jpg)

![7ddde55c6ce38e044b651a3d82ebbdf94fe37e9940860a95694adc4003f54a86.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/7ddde55c6ce38e044b651a3d82ebbdf94fe37e9940860a95694adc4003f54a86.jpg)

![89a766fc95895e861d207bddb56457f89cb2677f3e7fd89dc1c60c8396496235.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/89a766fc95895e861d207bddb56457f89cb2677f3e7fd89dc1c60c8396496235.jpg)

![922c7d6332efe0cba7b15cdcdac4b6ea5fab4ef95664e2ba80167c01e5a09da7.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/922c7d6332efe0cba7b15cdcdac4b6ea5fab4ef95664e2ba80167c01e5a09da7.jpg)

![9aab6cf3de4f4d67e035c7c60dd08760b71537470a44e60036f4e6d8736e3452.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/9aab6cf3de4f4d67e035c7c60dd08760b71537470a44e60036f4e6d8736e3452.jpg)

![a960f2ee9bb1f23e1fd1ce2f014b47595719f70a986255055f16cbcf18dac3cd.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/a960f2ee9bb1f23e1fd1ce2f014b47595719f70a986255055f16cbcf18dac3cd.jpg)

![b90462758a5a3e4d8be2a6024edf818bcfd63bb8bff3492ee80b20c3cd5a6675.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/b90462758a5a3e4d8be2a6024edf818bcfd63bb8bff3492ee80b20c3cd5a6675.jpg)

![c311e2e6b21314aa25af58a4e6d569dfd9d6a7dea52392bc284667abeb914433.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/c311e2e6b21314aa25af58a4e6d569dfd9d6a7dea52392bc284667abeb914433.jpg)

![c6453566679271f258adf85d1a29718b5963e4babdb0b74d9abc7d690c63408d.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/c6453566679271f258adf85d1a29718b5963e4babdb0b74d9abc7d690c63408d.jpg)

![c7bd61fe98c41324929939b61c444a20dc291f125a3ce68b9e524428ece6848b.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/c7bd61fe98c41324929939b61c444a20dc291f125a3ce68b9e524428ece6848b.jpg)

![d59f01990cb679675500301d705a9afc24b1cdcdede0d5908fdc0109d1291edb.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/d59f01990cb679675500301d705a9afc24b1cdcdede0d5908fdc0109d1291edb.jpg)

![e8de7c7e58bcffc6e960fe5dcddf418ea9c75729e842792dc334a2d68d19af6f.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/images/e8de7c7e58bcffc6e960fe5dcddf418ea9c75729e842792dc334a2d68d19af6f.jpg)

### Tables

![2a6e54cf1362a0339169ea021e36df5426198a044a3f4c08ffed05e4642ecc6a.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/2a6e54cf1362a0339169ea021e36df5426198a044a3f4c08ffed05e4642ecc6a.jpg)

![45abf46941d11e8efe8f58fb7b4f874e3f2bc52e227380fe11c28fdb22fd3979.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/45abf46941d11e8efe8f58fb7b4f874e3f2bc52e227380fe11c28fdb22fd3979.jpg)

![4690c27ee5a0bfb69a70ef5086d2d4f635e028d5c7ca48cf353a42e468d9b09a.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/4690c27ee5a0bfb69a70ef5086d2d4f635e028d5c7ca48cf353a42e468d9b09a.jpg)

![4bdecd32009ab4ae9e45f49bae1684e74c24efafd6d7174017dbbbcfb1b3524f.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/4bdecd32009ab4ae9e45f49bae1684e74c24efafd6d7174017dbbbcfb1b3524f.jpg)

![5d197241c36a0464b2bdb4a715dda3d6d5c2bbc1363b4aec03b3d93669f958d4.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/5d197241c36a0464b2bdb4a715dda3d6d5c2bbc1363b4aec03b3d93669f958d4.jpg)

![606c57cd98ad0076779976b9d825e02a2d5c27af23c7ed85b0884a76c1ae9f51.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/606c57cd98ad0076779976b9d825e02a2d5c27af23c7ed85b0884a76c1ae9f51.jpg)

![9cd55daa9abc93da8565f9396c8d5a6b22976e17731c957f53f5f2147a5900d5.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/9cd55daa9abc93da8565f9396c8d5a6b22976e17731c957f53f5f2147a5900d5.jpg)

![ac5cf3f5e6122077a71f836c9461de3429ea0b2accae896046fbf9b8a60415af.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/ac5cf3f5e6122077a71f836c9461de3429ea0b2accae896046fbf9b8a60415af.jpg)

![b8153be61ef6a68dc438dc1f2779e8366731410fe921839e754d3531281b675c.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/b8153be61ef6a68dc438dc1f2779e8366731410fe921839e754d3531281b675c.jpg)

![bb93eeddb8391e176ccfab70357d1a06cae8c1a696945f23dabfc57185ee60a9.jpg](../icml_results/1902_DiffThinker_%20Towards%20Generative%20Multimodal%20Reasoning%20with%20Diffusion%20Models/tables/bb93eeddb8391e176ccfab70357d1a06cae8c1a696945f23dabfc57185ee60a9.jpg)

<a id="paper-1878"></a>

## Newton-coupled Dual-Teacher Semi-supervised Learning Framework


### Images

![17adc50737b4a1aa022fa6105995229b8b6665c7eaccc1dbc72a0a3b290044fe.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/images/17adc50737b4a1aa022fa6105995229b8b6665c7eaccc1dbc72a0a3b290044fe.jpg)

![5ab28d30bbdca78d578369fa7d6426a68cca384c01ed211dda2c20e5809d6c10.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/images/5ab28d30bbdca78d578369fa7d6426a68cca384c01ed211dda2c20e5809d6c10.jpg)

![693cc60dd4aa8ed2d813744a533543c8136b2274da0325d7dd8154f0954c4855.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/images/693cc60dd4aa8ed2d813744a533543c8136b2274da0325d7dd8154f0954c4855.jpg)

![8e4bd25e1704a952e87954002aa3df0cf5a6d67745f89a403eadbe7cce301025.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/images/8e4bd25e1704a952e87954002aa3df0cf5a6d67745f89a403eadbe7cce301025.jpg)

![ec0fd255b24c8a8e32ced97f31b69694cb5df026b2c31b8e83bb889b81e244ab.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/images/ec0fd255b24c8a8e32ced97f31b69694cb5df026b2c31b8e83bb889b81e244ab.jpg)

### Tables

![0ba74b9d17ba9a48444a2f7f895456602a081235bd93ae733689c4bce15192fa.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/0ba74b9d17ba9a48444a2f7f895456602a081235bd93ae733689c4bce15192fa.jpg)

![5949150343ea03a72876bc589051ec49e58a04c77f4e8e27242b76b7ab80b2b2.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/5949150343ea03a72876bc589051ec49e58a04c77f4e8e27242b76b7ab80b2b2.jpg)

![59c3a075ee3100ffdf7da31b8c26a9be04dd63ee2ab587b87b22d945b516fed6.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/59c3a075ee3100ffdf7da31b8c26a9be04dd63ee2ab587b87b22d945b516fed6.jpg)

![91240d60da96ff714c88747c583a108b970cf4683d2af57dfdff425951815c2b.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/91240d60da96ff714c88747c583a108b970cf4683d2af57dfdff425951815c2b.jpg)

![a3df5dafbb4656ae03fe1252f1b519cb01947d75bf2839eea7c12fa780ebc59d.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/a3df5dafbb4656ae03fe1252f1b519cb01947d75bf2839eea7c12fa780ebc59d.jpg)

![e8f8d0093f9d9e3d788f11e5f284a1346d717e62e9bcdea35df65fe59475646a.jpg](../icml_results/1903_Newton-coupled%20Dual-Teacher%20Semi-supervised%20Learning%20Framework/tables/e8f8d0093f9d9e3d788f11e5f284a1346d717e62e9bcdea35df65fe59475646a.jpg)

<a id="paper-1879"></a>

## Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation


### Images

![0b357b864c0d4bb1e36ab72a52646bcc7ea56733d67df8ef496d86f61ecf728b.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/0b357b864c0d4bb1e36ab72a52646bcc7ea56733d67df8ef496d86f61ecf728b.jpg)

![1ab021dbc1a2659dcc1cea236431acf47217bb7fb4f24b5517569a59552a13d9.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/1ab021dbc1a2659dcc1cea236431acf47217bb7fb4f24b5517569a59552a13d9.jpg)

![26d3654bd0e3e8efa3c924eebe223d502bdb3aba098d4f44c3c298f4b440b2aa.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/26d3654bd0e3e8efa3c924eebe223d502bdb3aba098d4f44c3c298f4b440b2aa.jpg)

![2e41ccc8779688644cc3ed5c32f7163c0920865aeed27edbf890d68a8437a053.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/2e41ccc8779688644cc3ed5c32f7163c0920865aeed27edbf890d68a8437a053.jpg)

![6429cf1363fc15aef6af4c8ffddcdd84d8b27e6881a22367fc389373425c7249.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/6429cf1363fc15aef6af4c8ffddcdd84d8b27e6881a22367fc389373425c7249.jpg)

![714234bf40240a601d64397c0103d28515f13613ee39f22e42fee133bbe2800b.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/714234bf40240a601d64397c0103d28515f13613ee39f22e42fee133bbe2800b.jpg)

![75250a7ca0279801d1189b6ea76e656e2053f81d66a8714499eb70b09b10819a.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/75250a7ca0279801d1189b6ea76e656e2053f81d66a8714499eb70b09b10819a.jpg)

![cb6ab19b4ffbd332e5cf179289f334c8a894ba505142dc74d8ed08423ecb0f80.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/cb6ab19b4ffbd332e5cf179289f334c8a894ba505142dc74d8ed08423ecb0f80.jpg)

![cd5596e8c8a3fb140efdf57b1ef31baa43326089020cce759517c998196fbfdb.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/cd5596e8c8a3fb140efdf57b1ef31baa43326089020cce759517c998196fbfdb.jpg)

![dd383ccbc3eef6de59796c5d990c0db1e079cf75a3ee2aeeaa9512d4003555d8.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/dd383ccbc3eef6de59796c5d990c0db1e079cf75a3ee2aeeaa9512d4003555d8.jpg)

![f40ffd8a4de9bd3c186d7129507feb5e5c153094f4ab9e7d249f0646a8d81354.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/images/f40ffd8a4de9bd3c186d7129507feb5e5c153094f4ab9e7d249f0646a8d81354.jpg)

### Tables

![04cc563fc3f8763032307d6ea6b2b20013365223218e4ff32b620afa4e2b0170.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/04cc563fc3f8763032307d6ea6b2b20013365223218e4ff32b620afa4e2b0170.jpg)

![61cdc46416679316a1ba39434262aceea76808c24fc8344eee6ae7b43598e389.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/61cdc46416679316a1ba39434262aceea76808c24fc8344eee6ae7b43598e389.jpg)

![71b4bb7f8dacb903d2ff93f7edda674f88e0dd5430fa1f79a4905cc71d38dfaa.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/71b4bb7f8dacb903d2ff93f7edda674f88e0dd5430fa1f79a4905cc71d38dfaa.jpg)

![71e7a48ad39fdb2d5fea0112d9f5434f7878c1fd4413caa4bdb715ba9f4d3321.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/71e7a48ad39fdb2d5fea0112d9f5434f7878c1fd4413caa4bdb715ba9f4d3321.jpg)

![9430dbc3562786f03dd52e7501d7ed28d3b481c737ac1043d573211215b22ae6.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/9430dbc3562786f03dd52e7501d7ed28d3b481c737ac1043d573211215b22ae6.jpg)

![991006f6e097ef9841edf7dca273b9c886f881831a963b8091795f06d06fa499.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/991006f6e097ef9841edf7dca273b9c886f881831a963b8091795f06d06fa499.jpg)

![a90d5375da32443008993d29e857243d7282372e6dc4c38801967587dcf219ef.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/a90d5375da32443008993d29e857243d7282372e6dc4c38801967587dcf219ef.jpg)

![dc9d161be60125d5796e55d734234d744c80a5d7e244278ad48ba94e86f07da7.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/dc9d161be60125d5796e55d734234d744c80a5d7e244278ad48ba94e86f07da7.jpg)

![f357d3d9bd384db1e2acbadaebef100c34cbf6eb57b00db0b2ee7f81f3b03fd8.jpg](../icml_results/1904_Advantage%20Collapse%20in%20Group%20Relative%20Policy%20Optimization_%20Diagnosis%20and%20Mitigation/tables/f357d3d9bd384db1e2acbadaebef100c34cbf6eb57b00db0b2ee7f81f3b03fd8.jpg)

<a id="paper-1880"></a>

## GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks


### Images

![12dc507ea402f7a3dd3fc735ffd554b1837c565379b0f5b4d316efb2b922d900.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/12dc507ea402f7a3dd3fc735ffd554b1837c565379b0f5b4d316efb2b922d900.jpg)

![18fd2d805079a4b39ea0d835ea8fae1d29dd7c47f0b70ede61dd347f2323928d.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/18fd2d805079a4b39ea0d835ea8fae1d29dd7c47f0b70ede61dd347f2323928d.jpg)

![3c485f73b19ed5c18abc73b749940de53ed61213bd122ef0d1846af164cfd38b.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/3c485f73b19ed5c18abc73b749940de53ed61213bd122ef0d1846af164cfd38b.jpg)

![3dda91966253b178fdfedd6f5382595c7a5196d90621d7488ca08b43f0642830.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/3dda91966253b178fdfedd6f5382595c7a5196d90621d7488ca08b43f0642830.jpg)

![4e822457bf81e197d21b9eed47d0a72d0ae908dd7187393b099ece60d98f347e.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/4e822457bf81e197d21b9eed47d0a72d0ae908dd7187393b099ece60d98f347e.jpg)

![77a604054c63b372183ed41f0bf00c24f64909eecc1760f7ea93e0c597256556.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/77a604054c63b372183ed41f0bf00c24f64909eecc1760f7ea93e0c597256556.jpg)

![a1a3191d3c86657caa79be5ebbf695057dddf3c625408c4e7fa08dc565ea9cc2.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/a1a3191d3c86657caa79be5ebbf695057dddf3c625408c4e7fa08dc565ea9cc2.jpg)

![a911b9256bd0ae29abe10e8f45172e44a4b141d02b3eb84e2d0e3de88c923318.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/a911b9256bd0ae29abe10e8f45172e44a4b141d02b3eb84e2d0e3de88c923318.jpg)

![f9667ed9b2da975fd2560a2efc555260315258b346a1c44c255b16574d18e5a3.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/images/f9667ed9b2da975fd2560a2efc555260315258b346a1c44c255b16574d18e5a3.jpg)

### Tables

![0f00d21aa2535422cbe264edb52e653d2d5e424d6d1b2d676ce434b7329c803d.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/0f00d21aa2535422cbe264edb52e653d2d5e424d6d1b2d676ce434b7329c803d.jpg)

![11378a41e3f0b0f8da3fe9aa65be3f6ae02cbf56acc6132cb87cca712aa96a5f.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/11378a41e3f0b0f8da3fe9aa65be3f6ae02cbf56acc6132cb87cca712aa96a5f.jpg)

![1625e245f2d723b1ad8b2633679bd3b2c1b9cca6624501af2e84a336d7ad0852.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/1625e245f2d723b1ad8b2633679bd3b2c1b9cca6624501af2e84a336d7ad0852.jpg)

![367068a146ca06c8d3bb19fead39fb1ac177267eccb66b6ae7bf40935dc6937c.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/367068a146ca06c8d3bb19fead39fb1ac177267eccb66b6ae7bf40935dc6937c.jpg)

![5254650ebb7666068b5ae58094c9dd859128aedea10e1bd443059461caecd18d.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/5254650ebb7666068b5ae58094c9dd859128aedea10e1bd443059461caecd18d.jpg)

![60af55afde6a431bbcb0fb9cd34f0fdff4f75815e1a04ab08db94cfc7fe93368.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/60af55afde6a431bbcb0fb9cd34f0fdff4f75815e1a04ab08db94cfc7fe93368.jpg)

![7e8268edd1bd9e5936bc15b6eea646027c172c391b3ab6cca4189e2b9c879b5e.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/7e8268edd1bd9e5936bc15b6eea646027c172c391b3ab6cca4189e2b9c879b5e.jpg)

![81b2778ec047ab8e69b0cf1c82c8f15c058f1bb27c545edab7910ea362c73d0b.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/81b2778ec047ab8e69b0cf1c82c8f15c058f1bb27c545edab7910ea362c73d0b.jpg)

![81db4b7f4f2f5b8e116ccf6199d5d708bc0d369d03e617053814db7ac60c5d83.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/81db4b7f4f2f5b8e116ccf6199d5d708bc0d369d03e617053814db7ac60c5d83.jpg)

![be545a8d39ef34131ba897a46e5b544ffc7f51386a68f42dc9febb0dda4ff931.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/be545a8d39ef34131ba897a46e5b544ffc7f51386a68f42dc9febb0dda4ff931.jpg)

![cfc1438fad3c38baf248afaeb8feb6032b4619a6ad7ba2ffec9cdf25fe362795.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/cfc1438fad3c38baf248afaeb8feb6032b4619a6ad7ba2ffec9cdf25fe362795.jpg)

![d2d3241349f33a6689dea7d9a9222b1e6da4265422be5a135bf901e110bec0ab.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/d2d3241349f33a6689dea7d9a9222b1e6da4265422be5a135bf901e110bec0ab.jpg)

![e4a2e1e1c74e117cdf90e0f2c59e6d9f0a5c43b8e83a08d27522aced28c000e7.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/e4a2e1e1c74e117cdf90e0f2c59e6d9f0a5c43b8e83a08d27522aced28c000e7.jpg)

![fd6053076f3ef2886bfd95a189d000394cabe064a4271024c2bf145b43775305.jpg](../icml_results/1905_GP2F_%20Cross-Domain%20Graph%20Prompting%20with%20Adaptive%20Fusion%20of%20Pre-trained%20Graph%20Neural%20Networks/tables/fd6053076f3ef2886bfd95a189d000394cabe064a4271024c2bf145b43775305.jpg)

<a id="paper-1881"></a>

## Demystifying When Pruning Works via Representation Hierarchies

### Images

![10597b89bd425a33024682333aa136d3ca7dd8280a4ba600fd99f79ce3d5cdca.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/10597b89bd425a33024682333aa136d3ca7dd8280a4ba600fd99f79ce3d5cdca.jpg)

![24894a21f02777e0819c57eb734456968cbf84b6f872e94733c1ea115ce88761.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/24894a21f02777e0819c57eb734456968cbf84b6f872e94733c1ea115ce88761.jpg)

![37345ed9bd24ecd43329f02c338e910018b7a318f5d131f887b29475b10022ab.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/37345ed9bd24ecd43329f02c338e910018b7a318f5d131f887b29475b10022ab.jpg)

![3a1ea49a861b937c4e264224fb1403776ff8d4faeca8bd9462b6cf5f750e8a86.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/3a1ea49a861b937c4e264224fb1403776ff8d4faeca8bd9462b6cf5f750e8a86.jpg)

![4f1ae7dd4c3ae1169a8a54f7c147f7591d561c180fcbe195079c33a6d9444fbb.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/4f1ae7dd4c3ae1169a8a54f7c147f7591d561c180fcbe195079c33a6d9444fbb.jpg)

![58245cea913a558318f1bf5a9dddbb10cea1fb801fa84458a9a161e27a019bc2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/58245cea913a558318f1bf5a9dddbb10cea1fb801fa84458a9a161e27a019bc2.jpg)

![77f308037adbed1a68ef2ed354aa066a4e3c09186612b2c1c67dce574187a4fb.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/77f308037adbed1a68ef2ed354aa066a4e3c09186612b2c1c67dce574187a4fb.jpg)

![795968f637089e9792450aaa3be02c5c71b6e6fa6338eac53b8138f9604620f0.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/795968f637089e9792450aaa3be02c5c71b6e6fa6338eac53b8138f9604620f0.jpg)

![7e48b1a6b7e287dbba01a344c3bfac752bb10dbcf63dd3364d25aedead25e7de.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/7e48b1a6b7e287dbba01a344c3bfac752bb10dbcf63dd3364d25aedead25e7de.jpg)

![8577b4d09956da68f7d85890e612729e73de51ae1e4dd1d18944f1a34a2991b4.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/8577b4d09956da68f7d85890e612729e73de51ae1e4dd1d18944f1a34a2991b4.jpg)

![8eebab15400b833d3b34f42c3c2f8d1227032ece70eb4916c93c8710fb3c5163.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/8eebab15400b833d3b34f42c3c2f8d1227032ece70eb4916c93c8710fb3c5163.jpg)

![960e32108288c1fb00cbf770f521e79beb065fe74581d363283e62d953c6fb37.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/960e32108288c1fb00cbf770f521e79beb065fe74581d363283e62d953c6fb37.jpg)

![97ade9ae5421e5d3f4930f339a8ce1ebeaf65d96b6752bf5d7ec5b69d35950c0.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/97ade9ae5421e5d3f4930f339a8ce1ebeaf65d96b6752bf5d7ec5b69d35950c0.jpg)

![9b866f736ffbb0ad7a5e277ccc123eaf7c11df15dfc95cd35bee23860d2e4bc1.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/9b866f736ffbb0ad7a5e277ccc123eaf7c11df15dfc95cd35bee23860d2e4bc1.jpg)

![9ddb5d75798eeb9c71a26bee5a7419f959e36fa90d0cefb5b744265f3d76708e.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/9ddb5d75798eeb9c71a26bee5a7419f959e36fa90d0cefb5b744265f3d76708e.jpg)

![baad57398fe04af1fc77a0eab941febcb569357ab47d6b590355509f4b29b104.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/baad57398fe04af1fc77a0eab941febcb569357ab47d6b590355509f4b29b104.jpg)

![c7672f1a87867cbfa0d7ce7e23026e6cf7f6ac2400a4776beeea014ccbcb04e2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/c7672f1a87867cbfa0d7ce7e23026e6cf7f6ac2400a4776beeea014ccbcb04e2.jpg)

![d043b2652304875d8cd98dc65d979ad9c2a7bbaada7db58be59723bb74245681.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/d043b2652304875d8cd98dc65d979ad9c2a7bbaada7db58be59723bb74245681.jpg)

![d4eea369c5a71fb23949b6611ee20b0839a05079593afb5d226656c17af2e41c.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/d4eea369c5a71fb23949b6611ee20b0839a05079593afb5d226656c17af2e41c.jpg)

![df364304a5eaacce0957cf7673053a6ab3a44d45fdabdffb8d2a5b260d525e0c.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/df364304a5eaacce0957cf7673053a6ab3a44d45fdabdffb8d2a5b260d525e0c.jpg)

![ee30ad07e19cffb1d500a8084510ab8447f18a08858868c7896efe011280ac4b.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/ee30ad07e19cffb1d500a8084510ab8447f18a08858868c7896efe011280ac4b.jpg)

![f5eb272e9a04c710d152b2509c59d75fb3df8900ea35a16b3a29d3e1ffe4af29.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/f5eb272e9a04c710d152b2509c59d75fb3df8900ea35a16b3a29d3e1ffe4af29.jpg)

### Tables

![2cd78a5b6dd269a82ac4da323b20613714fa6c05639099ac31841eb9ec616ad7.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/2cd78a5b6dd269a82ac4da323b20613714fa6c05639099ac31841eb9ec616ad7.jpg)

![50fc64d2b78f538d76fca07ff6f1fc95a0c1b709fea41e5cd7a4e51f8a2d5ed2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/50fc64d2b78f538d76fca07ff6f1fc95a0c1b709fea41e5cd7a4e51f8a2d5ed2.jpg)

![aaf171d12055031929613c86f7097c25f9efc7d7d7f4020ce143b9a89713dbd9.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/aaf171d12055031929613c86f7097c25f9efc7d7d7f4020ce143b9a89713dbd9.jpg)

![b87e2f05a099d4a6c8311aacc951bdc82bc03ecc2a37fcd68673d748ce47b81e.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/b87e2f05a099d4a6c8311aacc951bdc82bc03ecc2a37fcd68673d748ce47b81e.jpg)
