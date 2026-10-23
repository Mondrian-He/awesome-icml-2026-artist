# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 8 of 197

## 目录 (Table of Contents)

1. [Skip a Layer or Loop It? Learning Program-of-Layers in LLMs](#paper-0232)
2. [GFD-EMVC: Evolutionary Multi-View Classification with Label Noise via Gradient and Feature Dual-Perception](#paper-0233)
3. [Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates](#paper-0234)
4. [Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](#paper-0235)
5. [NorMuon: Making Muon more efficient and scalable](#paper-0236)
6. [Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](#paper-0237)
7. [Towards Pareto-Optimal Tool-Integrated Agents with Pareto Ranking Policy Optimization](#paper-0238)
8. [Stop When Further Reasoning Won’t Help: Attention-State Adaptive Generation in Reasoning Models](#paper-0239)
9. [Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum](#paper-0240)
10. [Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies](#paper-0241)
11. [Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks](#paper-0242)
12. [Neuro-evolutionary Continual Reinforcement Learning](#paper-0243)
13. [A Factorized Low-Rank RNN Framework for Uncovering Independent Neural Latent Dynamics and Connectivity](#paper-0244)
14. [From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space](#paper-0245)
15. [Scaling Robot Policy Evaluation via Discrete Diffusion World Model](#paper-0246)
16. [Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders](#paper-0247)
17. [A Recursive Decomposition Framework for Causal Structure Learning in the Presence of Latent Variables](#paper-0248)
18. [SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](#paper-0249)
19. [Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding](#paper-0250)
20. [ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models](#paper-0251)
21. [PLANTAIN: Plan-Answer Interleaved Reasoning](#paper-0252)
22. [Prototype-guided Bilateral Alignment Multimodal Federated Learning](#paper-0253)
23. [WaterSIC: Information-Theoretically (Near) Optimal Linear Layer Quantization](#paper-0254)
24. [From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models](#paper-0255)
25. [MSP: Probabilistically Consistent Multi-Scale Action Generation](#paper-0256)
26. [GR-LoRA: Gradient-Recycling Low-Rank Adaptation for Class-Incremental Learning](#paper-0257)
27. [FlatLand: Personalized Graph Federated Learning via Tailored Lorentz Space](#paper-0258)
28. [World-Model Inspired Emotion-aware Token Refinement for Training-Free Multimodal Emotion Recognition](#paper-0259)
29. [Solving Time-Dependent Differential Equations with Physical Dynamical Systems](#paper-0260)
30. [Online Conformal Prediction via Universal Portfolio Algorithms](#paper-0261)
31. [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](#paper-0262)
32. [Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning](#paper-0263)
33. [Local Covariate Selection for Average Causal Effect Estimation without Pretreatment and Causal Sufficiency Assumptions](#paper-0264)

---


<a id="paper-0232"></a>

## Skip a Layer or Loop It? Learning Program-of-Layers in LLMs

### Images

![36631bd34f100947209d9d65a7c8ede0f5a95949bf68196045c13be39ddcfb18.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/36631bd34f100947209d9d65a7c8ede0f5a95949bf68196045c13be39ddcfb18.jpg)

![3e9f49a77611d4eb19478399e54f51c599d5f0980ed115a9e8d46f284da11674.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/3e9f49a77611d4eb19478399e54f51c599d5f0980ed115a9e8d46f284da11674.jpg)

![721e4812011bd2f1adb1aba2aa6b57ea76e45fb88a4aaa2d410297ee28b78e7a.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/721e4812011bd2f1adb1aba2aa6b57ea76e45fb88a4aaa2d410297ee28b78e7a.jpg)

### Tables

![08b32e7e781d6cdc6fb23ae234ae57e9d5d861657ddc30721af991596f41093e.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/08b32e7e781d6cdc6fb23ae234ae57e9d5d861657ddc30721af991596f41093e.jpg)

![243e5d8c520ad15b5d412d485c13df745524d0517bf7bd2537769d22b09c228b.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/243e5d8c520ad15b5d412d485c13df745524d0517bf7bd2537769d22b09c228b.jpg)

![2cc08817404f8b743e9349925813b5059023b5cdc30536f12acfad0c1f3e86fb.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/2cc08817404f8b743e9349925813b5059023b5cdc30536f12acfad0c1f3e86fb.jpg)

![7338817da293acaa8ce57e3deeb9c29eb879714062bb2048e4ee40601069a3e5.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/7338817da293acaa8ce57e3deeb9c29eb879714062bb2048e4ee40601069a3e5.jpg)

![882e040f867452c85393733f7b6f18e2ff1880776e0e243c4f119d3763608c19.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/882e040f867452c85393733f7b6f18e2ff1880776e0e243c4f119d3763608c19.jpg)

![a9b5c9f0a71eee53264d22f7fda4a3d0277034672487ae31c2421688c6ac37cc.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/a9b5c9f0a71eee53264d22f7fda4a3d0277034672487ae31c2421688c6ac37cc.jpg)

![e8ce5e59ae329de9c156ee5999e1e65323f6ae2bbb228cf215a5e995fc35626f.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/e8ce5e59ae329de9c156ee5999e1e65323f6ae2bbb228cf215a5e995fc35626f.jpg)

## Skip a Layer or Loop It? Learning Program-of-Layers in LLMs


### Images

![550b2de7fc5eb96e2ea8a4ed6254c45185464d6b828d87da5f9672b7572480dc.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/550b2de7fc5eb96e2ea8a4ed6254c45185464d6b828d87da5f9672b7572480dc.jpg)

![59cf6d416f26c8e6ccb441535580d3e90963551d5117156cf19df2b261e79c6e.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/59cf6d416f26c8e6ccb441535580d3e90963551d5117156cf19df2b261e79c6e.jpg)

![5a5f8c9fa5e2401b42fee740405c2e6cb26b2203fd86dff4a95b50fbdeb74d49.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/5a5f8c9fa5e2401b42fee740405c2e6cb26b2203fd86dff4a95b50fbdeb74d49.jpg)

![98b210c41ea03720626c5cfebeb9d95a61d44060476d97cd6c064adaa46321bf.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/98b210c41ea03720626c5cfebeb9d95a61d44060476d97cd6c064adaa46321bf.jpg)

![9a95f31d214ed7ecd1b5b10abbfd8b5f08eb1b8e0107fdc308eaa529a987dd7f.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/9a95f31d214ed7ecd1b5b10abbfd8b5f08eb1b8e0107fdc308eaa529a987dd7f.jpg)

![a70c3410a6433f7a2a711eb72b1521c1ab5107d78b0a1de06124ed8872115830.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/a70c3410a6433f7a2a711eb72b1521c1ab5107d78b0a1de06124ed8872115830.jpg)

![be2dd7f14978f5c394bd8e6f396c6772d55bcf3cf07166bcce57a5d4b55a4d3a.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/be2dd7f14978f5c394bd8e6f396c6772d55bcf3cf07166bcce57a5d4b55a4d3a.jpg)

![fd6933c12a1e1f963bfba72b8b2f8c5d80e5c50dde767e6b7e2018aafba131c6.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/images/fd6933c12a1e1f963bfba72b8b2f8c5d80e5c50dde767e6b7e2018aafba131c6.jpg)

### Tables

![0099507211bb8b1c647166e7791861b16c8a77ccf4a17aaeb85e2296b95a5453.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/0099507211bb8b1c647166e7791861b16c8a77ccf4a17aaeb85e2296b95a5453.jpg)

![1aeb5dec7336a23b7aafe23bad805c52667d7e59ebb88c4b9d1fe873de91ccd5.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/1aeb5dec7336a23b7aafe23bad805c52667d7e59ebb88c4b9d1fe873de91ccd5.jpg)

![4c1a67e796b0897f17a69416f68fa62c99d8e499047b941b5ba40b36b68b7789.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/4c1a67e796b0897f17a69416f68fa62c99d8e499047b941b5ba40b36b68b7789.jpg)

![acd411bd20b24a9021b322789b2b547db15343886121cfda94502083186c74e2.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/acd411bd20b24a9021b322789b2b547db15343886121cfda94502083186c74e2.jpg)

![b6f9a16bd79bf49930d847d4e3c577ccf6a67e9d985982de1d83c2d24ad55661.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/b6f9a16bd79bf49930d847d4e3c577ccf6a67e9d985982de1d83c2d24ad55661.jpg)

![b823ac0fa6039c882fcb04f97ae2812ecd14f41a623cf8dfeb3832ca3cfe5bcd.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/b823ac0fa6039c882fcb04f97ae2812ecd14f41a623cf8dfeb3832ca3cfe5bcd.jpg)

![b900455fc16ea6d9bf662b05fc5231f2a331e7ba21aa83e8ebefbe3b233ea55a.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/b900455fc16ea6d9bf662b05fc5231f2a331e7ba21aa83e8ebefbe3b233ea55a.jpg)

![be46ad8a63343c1cabff8dd127c279c58b6006f7663bc26369941995a06f11de.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/be46ad8a63343c1cabff8dd127c279c58b6006f7663bc26369941995a06f11de.jpg)

![cdb6c107240a5ecfab22ecfcf8111bf806c319bb2906f73efe49841d679fdcd3.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/cdb6c107240a5ecfab22ecfcf8111bf806c319bb2906f73efe49841d679fdcd3.jpg)

![cfdb6973e778b9d7216a5a3fb03621e813708b8e8459da5d3e4eefc49c274180.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/cfdb6973e778b9d7216a5a3fb03621e813708b8e8459da5d3e4eefc49c274180.jpg)

![f961316579d8d5b5f9242be651fee7ded458e7940c5beea4a39963b336a35237.jpg](../icml_results/237_Skip%20a%20Layer%20or%20Loop%20It_%20Learning%20Program-of-Layers%20in%20LLMs/tables/f961316579d8d5b5f9242be651fee7ded458e7940c5beea4a39963b336a35237.jpg)

<a id="paper-0233"></a>

## GFD-EMVC: Evolutionary Multi-View Classification with Label Noise via Gradient and Feature Dual-Perception


### Images

![68cdb2e223b9654c161c0a7433dd876c107b29c6ac4770ba20f086817dc2b71b.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/68cdb2e223b9654c161c0a7433dd876c107b29c6ac4770ba20f086817dc2b71b.jpg)

![73fe5adbb8738899ed9169589082d13eb72368aa2fab0c514a5d08ddb420aafe.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/73fe5adbb8738899ed9169589082d13eb72368aa2fab0c514a5d08ddb420aafe.jpg)

![94139f7fe291dc512ca521f952c51064e50b3f4cf5fbafb81c9f5ce8126c9ff5.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/94139f7fe291dc512ca521f952c51064e50b3f4cf5fbafb81c9f5ce8126c9ff5.jpg)

![b7b1e5393668eaa0ca310fe7810ab27977698c372282035c45977bf47f996e5a.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/b7b1e5393668eaa0ca310fe7810ab27977698c372282035c45977bf47f996e5a.jpg)

![c8abd3d8b37a5eab38e499197548e2dfcd4a2a281be38771eeb7ff29c64d6a94.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/c8abd3d8b37a5eab38e499197548e2dfcd4a2a281be38771eeb7ff29c64d6a94.jpg)

![ddf54b87aa30ce4558d25cc583a8414071bfbc3a1ff0593f1107023af30a2b09.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/ddf54b87aa30ce4558d25cc583a8414071bfbc3a1ff0593f1107023af30a2b09.jpg)

![e70896946685015d30f035a6319a5e952af980fa24f3bfda70529e2dd942b5ed.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/e70896946685015d30f035a6319a5e952af980fa24f3bfda70529e2dd942b5ed.jpg)

![fe60ba456bb1ab56799b1c2cc628f377152f68c5b4246204dc6309b06bf9a2cf.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/images/fe60ba456bb1ab56799b1c2cc628f377152f68c5b4246204dc6309b06bf9a2cf.jpg)

### Tables

![0186b78d3f537452e207d27f32e4e80ee75e1b9a11ff13d344a0c8813c871329.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/0186b78d3f537452e207d27f32e4e80ee75e1b9a11ff13d344a0c8813c871329.jpg)

![16fca613039703cff15ea340fb98df003850492542fce60635f06bc9d0a75131.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/16fca613039703cff15ea340fb98df003850492542fce60635f06bc9d0a75131.jpg)

![392729cca03ef969951c556a504b2095b5a85f1173f88e31d56fe5675a012a59.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/392729cca03ef969951c556a504b2095b5a85f1173f88e31d56fe5675a012a59.jpg)

![3f2f99197997eefa28e00fa6ac39acc9f1b229148ff7ec1f7a19f59a24d2f29d.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/3f2f99197997eefa28e00fa6ac39acc9f1b229148ff7ec1f7a19f59a24d2f29d.jpg)

![56dae12252518af9f1ebd9342711f1f62b9d016fb8fc1f6bf457681339cdf935.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/56dae12252518af9f1ebd9342711f1f62b9d016fb8fc1f6bf457681339cdf935.jpg)

![8f24b26f8e927b2c2fde07543b49f28ee6fbb3ee97097d72fbc1182ec995ae6c.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/8f24b26f8e927b2c2fde07543b49f28ee6fbb3ee97097d72fbc1182ec995ae6c.jpg)

![a6b8b1079b2fb3b9efa680634f3fb402f75faa90f6a8dfcd453bf28ece774c3d.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/a6b8b1079b2fb3b9efa680634f3fb402f75faa90f6a8dfcd453bf28ece774c3d.jpg)

![acbd171647c92ffb7077c0d1cfefef07dff2ae24e42fd45cf7866e78c1890d7f.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/acbd171647c92ffb7077c0d1cfefef07dff2ae24e42fd45cf7866e78c1890d7f.jpg)

![b589c68e1354cc4b23706f281c826fb68ef0237238484471e07ce7d43c2646d3.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/b589c68e1354cc4b23706f281c826fb68ef0237238484471e07ce7d43c2646d3.jpg)

![ed1449996e8e3691ce7509f6fa497ffb5cce3c9e4801804b73cc6de2e8502c5e.jpg](../icml_results/238_GFD-EMVC_%20Evolutionary%20Multi-View%20Classification%20with%20Label%20Noise%20via%20Gradient%20and%20Feature%20Dual-Perc/tables/ed1449996e8e3691ce7509f6fa497ffb5cce3c9e4801804b73cc6de2e8502c5e.jpg)

<a id="paper-0234"></a>

## Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates


### Images

![0ea8723db183132b69fe42d979f1d7aa55219037cfe2af15f3fa90a2ab039a74.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/images/0ea8723db183132b69fe42d979f1d7aa55219037cfe2af15f3fa90a2ab039a74.jpg)

![21422a3491fa05fd27fed63ec2bdcce30f4eb4bbe202280450af37f29d40880e.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/images/21422a3491fa05fd27fed63ec2bdcce30f4eb4bbe202280450af37f29d40880e.jpg)

![6a1461ae6cfb096a6e253e8e6e787eac7fe4c59f5560c71188eb818b6a1f41d1.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/images/6a1461ae6cfb096a6e253e8e6e787eac7fe4c59f5560c71188eb818b6a1f41d1.jpg)

![b1ba24de2895cfd96551aecde8d956782d4209dfcbd0020eb8de89808f28e445.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/images/b1ba24de2895cfd96551aecde8d956782d4209dfcbd0020eb8de89808f28e445.jpg)

![e71eadb4e603bfc997fa9bae52fb985216f9cf98e802144fc955ecf90ab08d99.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/images/e71eadb4e603bfc997fa9bae52fb985216f9cf98e802144fc955ecf90ab08d99.jpg)

### Tables

![03252e93b4a86ab7d4bd5af94b74a83b98694b7210df7658bf368b9e89f43467.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/03252e93b4a86ab7d4bd5af94b74a83b98694b7210df7658bf368b9e89f43467.jpg)

![03cd0b78841b543e938dee6cec26afbdac9366a53dcdd92c76c95f08d22373dc.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/03cd0b78841b543e938dee6cec26afbdac9366a53dcdd92c76c95f08d22373dc.jpg)

![14890ffda44f3ea88210666b09bf8bacf31dbcbf95da36421c4eb54a964b1ed5.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/14890ffda44f3ea88210666b09bf8bacf31dbcbf95da36421c4eb54a964b1ed5.jpg)

![260d027fef0045499a19b450b1ad49410855c01b6d62a054374ec21d6de9e6c3.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/260d027fef0045499a19b450b1ad49410855c01b6d62a054374ec21d6de9e6c3.jpg)

![3af8e1b7b6eff5ffd6c9d3e18f8472c1abc3f9d9ab0e36b55b90c6bbd895d6d2.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/3af8e1b7b6eff5ffd6c9d3e18f8472c1abc3f9d9ab0e36b55b90c6bbd895d6d2.jpg)

![5709ab78c8fce5468ec707e80edebe44683148fb72e23e03b0e4ff0407dc4b7e.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/5709ab78c8fce5468ec707e80edebe44683148fb72e23e03b0e4ff0407dc4b7e.jpg)

![6f0c914df2b8ac47257ecf5ef81a952c7358bae872a3bc575152682082008f40.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/6f0c914df2b8ac47257ecf5ef81a952c7358bae872a3bc575152682082008f40.jpg)

![762d307defaf98d6706641cf65c20dc07ed113e0181cac30fc9895195b49c4fa.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/762d307defaf98d6706641cf65c20dc07ed113e0181cac30fc9895195b49c4fa.jpg)

![79bfe6ac9c0498ec66e0de6566b587074dba46c121294c967bedf627e0ab084f.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/79bfe6ac9c0498ec66e0de6566b587074dba46c121294c967bedf627e0ab084f.jpg)

![892a718dfd808f818498a15c872f0257de5f2614327feb5e694442b314d2cdaa.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/892a718dfd808f818498a15c872f0257de5f2614327feb5e694442b314d2cdaa.jpg)

![920db92ef2a52db8df33df92c2edb9a69ba00768dd6aeedafb594190b09d5587.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/920db92ef2a52db8df33df92c2edb9a69ba00768dd6aeedafb594190b09d5587.jpg)

![95d4e95e1a31df06dafc739db76ee020e55abbc5726527a6443eca7c15032238.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/95d4e95e1a31df06dafc739db76ee020e55abbc5726527a6443eca7c15032238.jpg)

![b518c399fa1f9db0d7c0ca7b583359f723ba46ee8b4051671631e54e7332f264.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/b518c399fa1f9db0d7c0ca7b583359f723ba46ee8b4051671631e54e7332f264.jpg)

![b78dc6121f60dbf4ef117b29d6956c99831aef86f319cfc8ee1530ec6b83dd05.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/b78dc6121f60dbf4ef117b29d6956c99831aef86f319cfc8ee1530ec6b83dd05.jpg)

![bab023ec6c086acff300ee738192839479470f326213b00d61ba72981c265309.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/bab023ec6c086acff300ee738192839479470f326213b00d61ba72981c265309.jpg)

![c4922f7def42ae6aa444fe7146dd8bd1f80974c66d89a2cc409f1b910bc3e976.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/c4922f7def42ae6aa444fe7146dd8bd1f80974c66d89a2cc409f1b910bc3e976.jpg)

![c60c33518fc74fccb04911cb50fc888380ed1b00caef226fe86962ac02e2c73f.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/c60c33518fc74fccb04911cb50fc888380ed1b00caef226fe86962ac02e2c73f.jpg)

![c9a28bbbe6af6ff9d7156937f1a2ed85f91d82e499faf409420beeb2c12139a3.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/c9a28bbbe6af6ff9d7156937f1a2ed85f91d82e499faf409420beeb2c12139a3.jpg)

![d2648186620489ac308160fce386ea899412d285ca720e834367ff58facaf1d9.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/d2648186620489ac308160fce386ea899412d285ca720e834367ff58facaf1d9.jpg)

![f31758b8ba5ad9a053f1ebfd910dd7cc2080cb4a5c7672dc4fbb6e6427c9ccd8.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/f31758b8ba5ad9a053f1ebfd910dd7cc2080cb4a5c7672dc4fbb6e6427c9ccd8.jpg)

![f49c68bdc959b1e1bbfcd38f96471db9e3564f00fd12502734fe7fac18389e3b.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/f49c68bdc959b1e1bbfcd38f96471db9e3564f00fd12502734fe7fac18389e3b.jpg)

![f5d5ed37327a45d3e26aa07eaa834625e291e00576910b3a55b11623bacb64b7.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/f5d5ed37327a45d3e26aa07eaa834625e291e00576910b3a55b11623bacb64b7.jpg)

![f65983120862e2db0f54040cd8a00cab626e3d5fa21e2b4836881ee460aed7b2.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/f65983120862e2db0f54040cd8a00cab626e3d5fa21e2b4836881ee460aed7b2.jpg)

![f84c9285776871d5e35ea2285dc02e9e965e5a7feaf83c8cdf41394498d95917.jpg](../icml_results/239_Just-In-Time%20Reinforcement%20Learning_%20Continual%20Learning%20in%20LLM%20Agents%20Without%20Gradient%20Updates/tables/f84c9285776871d5e35ea2285dc02e9e965e5a7feaf83c8cdf41394498d95917.jpg)

<a id="paper-0235"></a>

## Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior


### Images

![0696fcb7ed7dbdc5a05e2b5d8523875278e09e307ff4ca4660222f8119c482e9.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/0696fcb7ed7dbdc5a05e2b5d8523875278e09e307ff4ca4660222f8119c482e9.jpg)

![0ff14a83c1a624e90989aa5e242e2d31a2e5e450de6b6596efcdbe267501c67d.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/0ff14a83c1a624e90989aa5e242e2d31a2e5e450de6b6596efcdbe267501c67d.jpg)

![2d45a07c80956d364cd257b2dec7f3aaec91ff125faa8e309be52baebe5ca522.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/2d45a07c80956d364cd257b2dec7f3aaec91ff125faa8e309be52baebe5ca522.jpg)

![302bb82b9a234be05e89274fdc7d6869ee5f63ad0e90f6036f293e09257ec383.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/302bb82b9a234be05e89274fdc7d6869ee5f63ad0e90f6036f293e09257ec383.jpg)

![4912695478a491bccb48865635ca24f958cbf78be4d2ecd8defe975824365fe1.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/4912695478a491bccb48865635ca24f958cbf78be4d2ecd8defe975824365fe1.jpg)

![52829fe6298871840e195fc98dba9a20c94c529be85cc2a0a557743fd8f5ab57.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/52829fe6298871840e195fc98dba9a20c94c529be85cc2a0a557743fd8f5ab57.jpg)

![56c16d50c77d34e61736f92af41c57d9480c6d4acbbc7d1fecf29249cfdb7cd6.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/56c16d50c77d34e61736f92af41c57d9480c6d4acbbc7d1fecf29249cfdb7cd6.jpg)

![5826d743a888304725c06d76bc2ec7110266273b9d2cc34c02cd62ef3c9ae669.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/5826d743a888304725c06d76bc2ec7110266273b9d2cc34c02cd62ef3c9ae669.jpg)

![5e1a5bf29766d70345e6ffe0f6e40d88187c61b3c07858b8e9c9ba660c1ffdd8.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/5e1a5bf29766d70345e6ffe0f6e40d88187c61b3c07858b8e9c9ba660c1ffdd8.jpg)

![7a4b2097627910fccced6ef7f48a3c7f4e5d02566f0f23685f6f046d98d84e5b.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/7a4b2097627910fccced6ef7f48a3c7f4e5d02566f0f23685f6f046d98d84e5b.jpg)

![866ef0ff7b3bbc13e6f25222dd9548396c1499e33274d3504adcbdd6185c0a4d.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/866ef0ff7b3bbc13e6f25222dd9548396c1499e33274d3504adcbdd6185c0a4d.jpg)

![86a9e7f3cee68c2b096b2fa70a27049b60bd31a213d91ec900460dfce61f39a9.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/86a9e7f3cee68c2b096b2fa70a27049b60bd31a213d91ec900460dfce61f39a9.jpg)

![8bf817d31cfd6a000372ae8e662afce5a6d0d1957d0707070248f0380bd70aac.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/8bf817d31cfd6a000372ae8e662afce5a6d0d1957d0707070248f0380bd70aac.jpg)

![8c2299014cfec3058ac37537946a5868ca3f7f879305a344477aa9eb0dbe7617.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/8c2299014cfec3058ac37537946a5868ca3f7f879305a344477aa9eb0dbe7617.jpg)

![8c2dd52e8a432e1a7f777d143ddc1d26f8cb1f2f68a06409e6e383b5d653d510.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/8c2dd52e8a432e1a7f777d143ddc1d26f8cb1f2f68a06409e6e383b5d653d510.jpg)

![98e551b033df9f36630c205e7cc9b32f8c16731a45d20d1a7c72d87ad4f1108c.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/98e551b033df9f36630c205e7cc9b32f8c16731a45d20d1a7c72d87ad4f1108c.jpg)

![a5aea678533b1f14cc45eab80f9738bef5a330a2bda5782bf1030e2b4e868934.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/a5aea678533b1f14cc45eab80f9738bef5a330a2bda5782bf1030e2b4e868934.jpg)

![a874d79ca35e91542435a414f3ada7e6af67f79243a10926a45f27a04117fb25.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/a874d79ca35e91542435a414f3ada7e6af67f79243a10926a45f27a04117fb25.jpg)

![b00d5a0a569fa316ab8b164c8b841f8722128621480159ec304bc30a4c26d494.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/b00d5a0a569fa316ab8b164c8b841f8722128621480159ec304bc30a4c26d494.jpg)

![bcd4a2cdaa5d809f13081f01840f2766b272d63d255fba7c1c5673591a243e8f.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/bcd4a2cdaa5d809f13081f01840f2766b272d63d255fba7c1c5673591a243e8f.jpg)

![c0804bfe7f3f58d9a58ca83a931d9a2a80010404959c8487cbe02cc6610056b6.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/c0804bfe7f3f58d9a58ca83a931d9a2a80010404959c8487cbe02cc6610056b6.jpg)

![c795f76873b72616e6022d72876451c9585f53a960118e03db6361c6b581545d.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/c795f76873b72616e6022d72876451c9585f53a960118e03db6361c6b581545d.jpg)

![d37221fded49654108fd5d517fbb32b9c6b625fb3823b02df9d73abc86cd0dc7.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/d37221fded49654108fd5d517fbb32b9c6b625fb3823b02df9d73abc86cd0dc7.jpg)

![eb747db7a195af93aa1ca84893c4945433d84f42740b5ae23b6cd38d063562f4.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/eb747db7a195af93aa1ca84893c4945433d84f42740b5ae23b6cd38d063562f4.jpg)

![f5c454b52898299c421be2c212e391a2ec206f169d567226b59fc4c9f7f99772.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/images/f5c454b52898299c421be2c212e391a2ec206f169d567226b59fc4c9f7f99772.jpg)

### Tables

![0210a78170864625ebcc81f1a9cfcf6f7a89577b8306f92392268b93f768bfc9.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/0210a78170864625ebcc81f1a9cfcf6f7a89577b8306f92392268b93f768bfc9.jpg)

![10b7ca8e2c6e3651b6bbb572bc21ce1ec641362f1e15ebd55b4768aa2952b890.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/10b7ca8e2c6e3651b6bbb572bc21ce1ec641362f1e15ebd55b4768aa2952b890.jpg)

![274d0c0c694ff5f808b621ce13fcd8786b335855f402235f4a864ae702f82543.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/274d0c0c694ff5f808b621ce13fcd8786b335855f402235f4a864ae702f82543.jpg)

![5d533dcde9af037bb834dc53bf1956c267a81291658ac9da2ca39091366c1873.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/5d533dcde9af037bb834dc53bf1956c267a81291658ac9da2ca39091366c1873.jpg)

![677c59e74a6d5e0f8b681391c5a15832c7013667732a70b7012634215fb07ee4.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/677c59e74a6d5e0f8b681391c5a15832c7013667732a70b7012634215fb07ee4.jpg)

![79ba3fb6044614c2ebc6d886d7db271afc0bbea13b7aa30064edfeffaaf04df3.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/79ba3fb6044614c2ebc6d886d7db271afc0bbea13b7aa30064edfeffaaf04df3.jpg)

![8842d7107748cf4eda0f4763b16ea66a3681611d66fa0de53a1e1b385541b307.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/8842d7107748cf4eda0f4763b16ea66a3681611d66fa0de53a1e1b385541b307.jpg)

![96ecfba33fdb6a70676e0c720729e06c629f753b9ba6bb185a08665a93311255.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/96ecfba33fdb6a70676e0c720729e06c629f753b9ba6bb185a08665a93311255.jpg)

![b2d3b2e032a0c9a786c01189ac48c2e43e12d8d977d79e4bac4f6186bc025fa2.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/b2d3b2e032a0c9a786c01189ac48c2e43e12d8d977d79e4bac4f6186bc025fa2.jpg)

![c4b40108efccd585af0f0af6f30ec5affa9eba06d6966faf2754155b02727f99.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/c4b40108efccd585af0f0af6f30ec5affa9eba06d6966faf2754155b02727f99.jpg)

![f64dae7fdfc0d78679e2fb13053463ed80df857607ad6963f37a51d3e1669c79.jpg](../icml_results/240_Initialization%20is%20Half%20the%20Battle_%20Generating%20Diverse%20Images%20from%20a%20Guidance%20Potential%20Posterior/tables/f64dae7fdfc0d78679e2fb13053463ed80df857607ad6963f37a51d3e1669c79.jpg)

<a id="paper-0236"></a>

## NorMuon: Making Muon more efficient and scalable


### Images

![04375e60c797db7dac4b3b2e7bd23058a84f0b9f730fd7bd3267f75307675a13.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/04375e60c797db7dac4b3b2e7bd23058a84f0b9f730fd7bd3267f75307675a13.jpg)

![0d336d14128318b03e238668cdb506c2052b19a8dd15f61f0ee36ef1e754c2bd.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/0d336d14128318b03e238668cdb506c2052b19a8dd15f61f0ee36ef1e754c2bd.jpg)

![1d5f2b94a173d7fd8a24b23a45689cfb31a8a0cdf3c0862a53f85722743207ce.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/1d5f2b94a173d7fd8a24b23a45689cfb31a8a0cdf3c0862a53f85722743207ce.jpg)

![252a6dfb166ae412957ec0ad886637dc525969d63051d34a2d92a464cdefce27.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/252a6dfb166ae412957ec0ad886637dc525969d63051d34a2d92a464cdefce27.jpg)

![2c6e9f4adb67f997c465ec969f6a9237fa5b6035f241cf1eb780779eeee7febd.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/2c6e9f4adb67f997c465ec969f6a9237fa5b6035f241cf1eb780779eeee7febd.jpg)

![5a33ebbc3b4106e1e52c9194c0522b2f5a9abcb088165991a14d3703f5c36e9c.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/5a33ebbc3b4106e1e52c9194c0522b2f5a9abcb088165991a14d3703f5c36e9c.jpg)

![5a68f680d9b68bbf11dae20899ad023fa8b92ca4b1eca0a9231c656b9d806ed2.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/5a68f680d9b68bbf11dae20899ad023fa8b92ca4b1eca0a9231c656b9d806ed2.jpg)

![67e6d397b2fdc435ab3bf8c0a7bb28c108622bbfcc210ca1d2f893638a31536d.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/67e6d397b2fdc435ab3bf8c0a7bb28c108622bbfcc210ca1d2f893638a31536d.jpg)

![8d43917a7d80828b962c2a64033a2d76b775d1ddc9227db80ab70270c3a7df94.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/8d43917a7d80828b962c2a64033a2d76b775d1ddc9227db80ab70270c3a7df94.jpg)

![950cc275f28aa05e4574e487343a3ec0b558fa376c9627dfd4c001b1c98d72be.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/950cc275f28aa05e4574e487343a3ec0b558fa376c9627dfd4c001b1c98d72be.jpg)

![bd35e9e0a001cf317c117959908cb729162e06e684941145889b70f4f37d9f52.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/bd35e9e0a001cf317c117959908cb729162e06e684941145889b70f4f37d9f52.jpg)

![e1ff2451b582f563659c53510dee67aeba1800c314a4e738e57755dedf94d35e.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/e1ff2451b582f563659c53510dee67aeba1800c314a4e738e57755dedf94d35e.jpg)

![fa63c74a5b583edd3124760038de0bab7861743a63d0093cc3b161b6feecd7d1.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/images/fa63c74a5b583edd3124760038de0bab7861743a63d0093cc3b161b6feecd7d1.jpg)

### Tables

![0fe661582e5917cd2f044b195cc2ebd2f315094bd45fd02e9617eaf55f245c73.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/0fe661582e5917cd2f044b195cc2ebd2f315094bd45fd02e9617eaf55f245c73.jpg)

![34834bb131af9bf848e02ca30817447b5924045a29bdc11647e144ece89f36b7.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/34834bb131af9bf848e02ca30817447b5924045a29bdc11647e144ece89f36b7.jpg)

![36fcb309038551addebba83a3c9ad764d68011adb5fb111eb552348e7c5d39bc.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/36fcb309038551addebba83a3c9ad764d68011adb5fb111eb552348e7c5d39bc.jpg)

![4464bb9883e954a1ee37ead3bec7005c954e470e5ff12178b54392358baa09a5.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/4464bb9883e954a1ee37ead3bec7005c954e470e5ff12178b54392358baa09a5.jpg)

![4e8d58db2df78fd618fc31d4e593fe630d777aaa26e6dea33fa0a3517f31286d.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/4e8d58db2df78fd618fc31d4e593fe630d777aaa26e6dea33fa0a3517f31286d.jpg)

![5e848e82ec7e3be48ba04ee686e189495ec0f8a4c930472ee760d9d1a8f7bd90.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/5e848e82ec7e3be48ba04ee686e189495ec0f8a4c930472ee760d9d1a8f7bd90.jpg)

![6b85535bb7e09f944da58c6b6656c437a6609b8a0f94a37f14ab4eedf051d5a4.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/6b85535bb7e09f944da58c6b6656c437a6609b8a0f94a37f14ab4eedf051d5a4.jpg)

![718c2c22ccee1a1203a67c56e7ba4b9d98f1412acf7e637748bd050bfd00195f.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/718c2c22ccee1a1203a67c56e7ba4b9d98f1412acf7e637748bd050bfd00195f.jpg)

![c16059caeb87286f1fe8f7b3833f598009a9e348987b988c024c627f864b8350.jpg](../icml_results/241_NorMuon_%20Making%20Muon%20more%20efficient%20and%20scalable/tables/c16059caeb87286f1fe8f7b3833f598009a9e348987b988c024c627f864b8350.jpg)

<a id="paper-0237"></a>

## Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?


### Images

![40ffe5456d5d06f70cea79efc2d04aaaf8595dc2a3306908e21a67e0cd1a9516.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/40ffe5456d5d06f70cea79efc2d04aaaf8595dc2a3306908e21a67e0cd1a9516.jpg)

![61d8c22a6db38c6eea3214f344c5706240e7974dca85f83cfe344aebeb5576ac.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/61d8c22a6db38c6eea3214f344c5706240e7974dca85f83cfe344aebeb5576ac.jpg)

![7271af0f4e052280c0b423efcdb66534bcd425666568dc7facba99402a9c8dd5.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/7271af0f4e052280c0b423efcdb66534bcd425666568dc7facba99402a9c8dd5.jpg)

![73798f17514133932067fafacc934a6b9c629ebe0f4e1440003dc144daa8cf3c.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/73798f17514133932067fafacc934a6b9c629ebe0f4e1440003dc144daa8cf3c.jpg)

![8494e84655f060278114b71a16a623b15c1076665ecc1d6724de39340af0df7d.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/8494e84655f060278114b71a16a623b15c1076665ecc1d6724de39340af0df7d.jpg)

![91d0cd7fae2c0a2641a266038888608938815b7b22f8572dd0e3dd73fa3a17b2.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/91d0cd7fae2c0a2641a266038888608938815b7b22f8572dd0e3dd73fa3a17b2.jpg)

![c34e5ffc95d1d0bf49cebc081fe74ff00c234f47ac871476ca16f0f27735f30a.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/images/c34e5ffc95d1d0bf49cebc081fe74ff00c234f47ac871476ca16f0f27735f30a.jpg)

### Tables

![21db02db40311b07b2580f9975a0294a3f5449372cd872604e36288a186f2d3b.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/21db02db40311b07b2580f9975a0294a3f5449372cd872604e36288a186f2d3b.jpg)

![364206318df49be63aa8884fbfc05d0d6e98e76621b2e5e8eb60368598e56a05.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/364206318df49be63aa8884fbfc05d0d6e98e76621b2e5e8eb60368598e56a05.jpg)

![4843aef4a753e626e28708e18dfedb0fd26b9e4595beb7e27896f2c18195200f.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/4843aef4a753e626e28708e18dfedb0fd26b9e4595beb7e27896f2c18195200f.jpg)

![4aa38baee6cce6438f5b1279f4933490b9e7a635c9b90460393b44c042928e4b.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/4aa38baee6cce6438f5b1279f4933490b9e7a635c9b90460393b44c042928e4b.jpg)

![5979b18e0c1511353c16ddf0070a40e54a04211354bcf9fbd72e4fa07816152c.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/5979b18e0c1511353c16ddf0070a40e54a04211354bcf9fbd72e4fa07816152c.jpg)

![8fb5cb63134b07edf67ddd17e973fedf6ea6fd1ec8c55723ccdfe7c8e1dae209.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/8fb5cb63134b07edf67ddd17e973fedf6ea6fd1ec8c55723ccdfe7c8e1dae209.jpg)

![b72032d76e58b0242d55bb4992dbaf20ae82fdea62300f3a047b2a3c911a2b0d.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/b72032d76e58b0242d55bb4992dbaf20ae82fdea62300f3a047b2a3c911a2b0d.jpg)

![cbbfc5c5c885821963ee4013d7c4a0b903923b13a8199569243006b4987a9624.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/cbbfc5c5c885821963ee4013d7c4a0b903923b13a8199569243006b4987a9624.jpg)

![e6b25383d501403ca258986fbdfb229bc30d0c88a352aafe25a802eead5407e3.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/e6b25383d501403ca258986fbdfb229bc30d0c88a352aafe25a802eead5407e3.jpg)

![f8d6dcc313710347a35b046a5d98a085e96fec2a7c07a93f78be6b48040e16c9.jpg](../icml_results/242_Mind%20Your%20Margin%20and%20Boundary_%20Are%20Your%20Distilled%20Datasets%20Truly%20Robust_/tables/f8d6dcc313710347a35b046a5d98a085e96fec2a7c07a93f78be6b48040e16c9.jpg)

<a id="paper-0238"></a>

## Towards Pareto-Optimal Tool-Integrated Agents with Pareto Ranking Policy Optimization


### Images

![128937cce351649c33248cdfa1f55646fc07cc6fad687427990ae4c5756f97f8.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/images/128937cce351649c33248cdfa1f55646fc07cc6fad687427990ae4c5756f97f8.jpg)

![8e13b31072bfb6081b0feccabdd39cfd1a923e95995979171624eb1f02ae616b.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/images/8e13b31072bfb6081b0feccabdd39cfd1a923e95995979171624eb1f02ae616b.jpg)

![e40c3fa01ec02b5890079238f51205265ae5f9d0129f2b60fa5c52e6b1f81c63.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/images/e40c3fa01ec02b5890079238f51205265ae5f9d0129f2b60fa5c52e6b1f81c63.jpg)

### Tables

![31d7f610472423431b8c4700090c6fa902d1f05fd76bfe2ad1117ea394baa9aa.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/tables/31d7f610472423431b8c4700090c6fa902d1f05fd76bfe2ad1117ea394baa9aa.jpg)

![3903be3c4064b75ebf8bb228f05a6df4347856fc821eae2435e0af086474aac7.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/tables/3903be3c4064b75ebf8bb228f05a6df4347856fc821eae2435e0af086474aac7.jpg)

![4c3b1318545b1f023ccbc5d3b68b19d4701d407a04e27c7f290449ff37f51992.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/tables/4c3b1318545b1f023ccbc5d3b68b19d4701d407a04e27c7f290449ff37f51992.jpg)

![6be14d9031c80a33de7d0d57be52f4bfb154001f21d27c41b36eeb5d04864ae5.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/tables/6be14d9031c80a33de7d0d57be52f4bfb154001f21d27c41b36eeb5d04864ae5.jpg)

![ab918badbc04c88478eb473d6c6e7f9b724678307c9e4eb9569dd41eb41965c0.jpg](../icml_results/243_Towards%20Pareto-Optimal%20Tool-Integrated%20Agents%20with%20Pareto%20Ranking%20Policy%20Optimization/tables/ab918badbc04c88478eb473d6c6e7f9b724678307c9e4eb9569dd41eb41965c0.jpg)

<a id="paper-0239"></a>

## Stop When Further Reasoning Won’t Help: Attention-State Adaptive Generation in Reasoning Models


### Images

![1426b62b15ed3ddb882d095a8ffa7c16d70cde8b509397258db50ed3bba4d520.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/1426b62b15ed3ddb882d095a8ffa7c16d70cde8b509397258db50ed3bba4d520.jpg)

![40d02e49b4e301087cbbb249af752475b0ef72e13f4ba7777a9ed894699f75ef.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/40d02e49b4e301087cbbb249af752475b0ef72e13f4ba7777a9ed894699f75ef.jpg)

![57b56739aaccd4d15d3afc35309d2a5d1df066131a22837149f8dda0e06b87c7.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/57b56739aaccd4d15d3afc35309d2a5d1df066131a22837149f8dda0e06b87c7.jpg)

![59cfdf3d19efc2e1bec705e8a0c892903d071360b757695ad90359fd6f1fcd46.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/59cfdf3d19efc2e1bec705e8a0c892903d071360b757695ad90359fd6f1fcd46.jpg)

![bd7ac4a8f7e71633517ab33f1dd1f22a4427eeac526337b49567042bc75f336c.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/bd7ac4a8f7e71633517ab33f1dd1f22a4427eeac526337b49567042bc75f336c.jpg)

![c66076443b82cf37806e59fc7ff9f9ec98f23db639931269f845558b682fdf39.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/c66076443b82cf37806e59fc7ff9f9ec98f23db639931269f845558b682fdf39.jpg)

![e58739903be666ce274140006f6a89d390ae71d2563b9d18a2027078bb0ad8db.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/e58739903be666ce274140006f6a89d390ae71d2563b9d18a2027078bb0ad8db.jpg)

![e924d482db3fad8574bd203801993a728aa49d9b8f2cf58d93405c23ee7fa98d.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/e924d482db3fad8574bd203801993a728aa49d9b8f2cf58d93405c23ee7fa98d.jpg)

![f04641cd6aaa0249e93d8e94298d998ebc933c7b87c5bd11e3fd71f9a9262935.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/images/f04641cd6aaa0249e93d8e94298d998ebc933c7b87c5bd11e3fd71f9a9262935.jpg)

### Tables

![3c0f2de67ebcc8805e3db651cb24f602d2fa10b676f0c3f29c93fd64d311f5a6.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/3c0f2de67ebcc8805e3db651cb24f602d2fa10b676f0c3f29c93fd64d311f5a6.jpg)

![59a90af2fe2ddc4af0ba9ffa4520399e2a26fea065117f253ea655154c367577.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/59a90af2fe2ddc4af0ba9ffa4520399e2a26fea065117f253ea655154c367577.jpg)

![6a6922a4a5287f2cc95f22cb3554382bfbfb3981661be1666425564874884e28.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/6a6922a4a5287f2cc95f22cb3554382bfbfb3981661be1666425564874884e28.jpg)

![7b8631ef00c57b2146566da8bbc431435c7675407f54a39f0827778a12ca032b.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/7b8631ef00c57b2146566da8bbc431435c7675407f54a39f0827778a12ca032b.jpg)

![a9a2323fbe2f4c9dbda613ce265a5c6ac7bd259feda5e5fb7078a2caa2c3c0b0.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/a9a2323fbe2f4c9dbda613ce265a5c6ac7bd259feda5e5fb7078a2caa2c3c0b0.jpg)

![b52c2148ea68006406d0d1340959152d9eeb496fad565e76dc7ea6246b718667.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/b52c2148ea68006406d0d1340959152d9eeb496fad565e76dc7ea6246b718667.jpg)

![cc3fb9452c911fabe937c259305b8f6040e9cc2bf49c7e4d72fadb0110112bb4.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/cc3fb9452c911fabe937c259305b8f6040e9cc2bf49c7e4d72fadb0110112bb4.jpg)

![d248605c12edd28d89352f23d5051405ede5033bb9c0e1cfa897efd6af47c002.jpg](../icml_results/244_Stop%20When%20Further%20Reasoning%20Won%E2%80%99t%20Help_%20Attention-State%20Adaptive%20Generation%20in%20Reasoning%20Models/tables/d248605c12edd28d89352f23d5051405ede5033bb9c0e1cfa897efd6af47c002.jpg)

<a id="paper-0240"></a>

## Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum


### Images

![1b01fdec4f8dd73bb8d38adfd508d1d8ebf890ce8334feadde663d1a0c80b79c.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/1b01fdec4f8dd73bb8d38adfd508d1d8ebf890ce8334feadde663d1a0c80b79c.jpg)

![6bf9a2a764721e03db59978cd51d7035f679bd783885c3a0e9332e88d3d08b30.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/6bf9a2a764721e03db59978cd51d7035f679bd783885c3a0e9332e88d3d08b30.jpg)

![790b3488cefc073ec3bb1f9b0dc2f765fe508656fad03b82b9982ad1d205ad41.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/790b3488cefc073ec3bb1f9b0dc2f765fe508656fad03b82b9982ad1d205ad41.jpg)

![a3bc067cc0a02c0b496c8e5e54624ff7ec137efd9c971ca25efef647a8e4c2f8.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/a3bc067cc0a02c0b496c8e5e54624ff7ec137efd9c971ca25efef647a8e4c2f8.jpg)

![b8c4ca2e65b0c788f471122218d3a7343df07be2eb2b5667ca99a400f582ddc9.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/b8c4ca2e65b0c788f471122218d3a7343df07be2eb2b5667ca99a400f582ddc9.jpg)

![db94d40a70e32f220764cde6c6e42f75e3a2876ff1803efba5ea15e98503a5ae.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/db94d40a70e32f220764cde6c6e42f75e3a2876ff1803efba5ea15e98503a5ae.jpg)

![fceed407cb0c6fdd6909c49dbbb8994e156858171a677b1c97aeaf87edfdc83d.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/images/fceed407cb0c6fdd6909c49dbbb8994e156858171a677b1c97aeaf87edfdc83d.jpg)

### Tables

![03c61c8af73daf422047e6d9237b6f476971bfc5c0bc967cfc25717ca44ed8b7.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/03c61c8af73daf422047e6d9237b6f476971bfc5c0bc967cfc25717ca44ed8b7.jpg)

![205fd7d8828ec4908aa71bb8f956f67ae00c752014e9502c68311d3057e55829.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/205fd7d8828ec4908aa71bb8f956f67ae00c752014e9502c68311d3057e55829.jpg)

![39dd3b257b581b2528ff293d8a7f605a6d184462cff9d04698293bce1cfdfdcc.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/39dd3b257b581b2528ff293d8a7f605a6d184462cff9d04698293bce1cfdfdcc.jpg)

![45c1976a8e1058ed9800e12723575a1e76eb2856e97405faa70ec203231d3437.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/45c1976a8e1058ed9800e12723575a1e76eb2856e97405faa70ec203231d3437.jpg)

![48bed78c72d76b084c83c9cef6dcdc44efd4916931e87c5a9323086f090d64af.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/48bed78c72d76b084c83c9cef6dcdc44efd4916931e87c5a9323086f090d64af.jpg)

![53d52c65a1033e70ce834810f1bea0b151cb860ddbef53546a1060102c4b9907.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/53d52c65a1033e70ce834810f1bea0b151cb860ddbef53546a1060102c4b9907.jpg)

![773150afb1ca60b02f85f5e1dff02fd29d45f321e6be843cbd59eb4a2bfa7524.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/773150afb1ca60b02f85f5e1dff02fd29d45f321e6be843cbd59eb4a2bfa7524.jpg)

![8e3c3ddeda20df09ba2bd4e48aafaa1c5bd9e6bde7815def1f06dd7bf053107b.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/8e3c3ddeda20df09ba2bd4e48aafaa1c5bd9e6bde7815def1f06dd7bf053107b.jpg)

![af975851f9a1b7e61ba7d8a3a48f0865f70bcc269b369c3218ac6a3f0d22a9b3.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/af975851f9a1b7e61ba7d8a3a48f0865f70bcc269b369c3218ac6a3f0d22a9b3.jpg)

![e653d9219c4020a4c47c9a92d336d13cf97184517e2f7431eba874242cb40a0b.jpg](../icml_results/245_Beyond%20Log%20Likelihood_%20Probability-Based%20Objectives%20for%20Supervised%20Fine-Tuning%20across%20the%20Model%20Capa/tables/e653d9219c4020a4c47c9a92d336d13cf97184517e2f7431eba874242cb40a0b.jpg)

<a id="paper-0241"></a>

## Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies


### Images

![2fcf1b237dd5ecd25628e1a75bcd0cc6dec6a82132efc9c11d06135c0e8e4baa.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/2fcf1b237dd5ecd25628e1a75bcd0cc6dec6a82132efc9c11d06135c0e8e4baa.jpg)

![847369b7e86cc7a2b435a5c936900cd866c1cc1c2bf69d2ae2874ad1672e090e.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/847369b7e86cc7a2b435a5c936900cd866c1cc1c2bf69d2ae2874ad1672e090e.jpg)

![8a55b10c7568bfc0d4808138fe0787eda9243880f8c445f7a08c5f3101453b12.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/8a55b10c7568bfc0d4808138fe0787eda9243880f8c445f7a08c5f3101453b12.jpg)

![91ee8f07bbc1852aff4e8d53eaf522b9f846ee52e1367213703c3d9a34755510.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/91ee8f07bbc1852aff4e8d53eaf522b9f846ee52e1367213703c3d9a34755510.jpg)

![b2a1343feebf7b3d96dabdf9f78da4786cbd7a2dcef0ee4170f14b0b249fcdfb.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/b2a1343feebf7b3d96dabdf9f78da4786cbd7a2dcef0ee4170f14b0b249fcdfb.jpg)

![b425cbe259839165616debbf83c253c818bdd969c212a6a645b080add37831ec.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/b425cbe259839165616debbf83c253c818bdd969c212a6a645b080add37831ec.jpg)

![e82e377f06379dbb88702c852f56b8f48e5fe59aa6bf0c84d5f0c51c1040cc5e.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/images/e82e377f06379dbb88702c852f56b8f48e5fe59aa6bf0c84d5f0c51c1040cc5e.jpg)

### Tables

![1d95c6a98298ca494b22686161a10253c6f71f78bc5139df088fa5f9c989738b.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/tables/1d95c6a98298ca494b22686161a10253c6f71f78bc5139df088fa5f9c989738b.jpg)

![2fb88328f67b3e7a4ba391e4f7eaf6ce83613e9c3d500b046262c891cfdd2de6.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/tables/2fb88328f67b3e7a4ba391e4f7eaf6ce83613e9c3d500b046262c891cfdd2de6.jpg)

![47e2ceabb5bfd3111193800fda03f8d084628563938b1fa322188c5d03c3d92f.jpg](../icml_results/246_Reverse%20Flow%20Matching_%20A%20Unified%20Framework%20for%20Online%20Reinforcement%20Learning%20with%20Diffusion%20and%20Flow/tables/47e2ceabb5bfd3111193800fda03f8d084628563938b1fa322188c5d03c3d92f.jpg)

<a id="paper-0242"></a>

## Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks


### Images

![0518bfa1cd7a743fed0544d0ebb03d7295b15b560fd57bef29e5e5c170247029.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/0518bfa1cd7a743fed0544d0ebb03d7295b15b560fd57bef29e5e5c170247029.jpg)

![3806d2ebbc33de055cc9c1cbc705f0d437efc0fc22250fe0692abb28c2c9663f.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/3806d2ebbc33de055cc9c1cbc705f0d437efc0fc22250fe0692abb28c2c9663f.jpg)

![4c12d252a0c4a9a095a7a0ce6221e5c36bac16c04f4851979f847cb2c8a75134.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/4c12d252a0c4a9a095a7a0ce6221e5c36bac16c04f4851979f847cb2c8a75134.jpg)

![93a57a99aa2dcd96569d790078f9026c9b200139aa4398d4c389bcc06abda795.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/93a57a99aa2dcd96569d790078f9026c9b200139aa4398d4c389bcc06abda795.jpg)

![ae1d38cbf99a2a101d5e3712934bbc7ab7eb5464fe3e48d4fe0d8175076fdd7e.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/ae1d38cbf99a2a101d5e3712934bbc7ab7eb5464fe3e48d4fe0d8175076fdd7e.jpg)

![b24a8b5870ded1bfe9ab850ad666da1dd0fbfecbdbca00efa74b51a74c9da9fa.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/b24a8b5870ded1bfe9ab850ad666da1dd0fbfecbdbca00efa74b51a74c9da9fa.jpg)

![c88b7ed31117ef92aa13e31aac154e1a033883f5bca54e296414a7237d0f1538.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/c88b7ed31117ef92aa13e31aac154e1a033883f5bca54e296414a7237d0f1538.jpg)

![fc0482ec5f6aefba82ebe5a5f31dd53d1aec104bd76245e712caa530e6987462.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/images/fc0482ec5f6aefba82ebe5a5f31dd53d1aec104bd76245e712caa530e6987462.jpg)

### Tables

![020bf1c13a339837eecfef9f2e03e0ecf635dd54ac9a5f9521175c3693443445.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/020bf1c13a339837eecfef9f2e03e0ecf635dd54ac9a5f9521175c3693443445.jpg)

![08b3f16eb2e76ecae175934e8186ae47bff4fedf440e8753ae89e3fc8534ddd3.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/08b3f16eb2e76ecae175934e8186ae47bff4fedf440e8753ae89e3fc8534ddd3.jpg)

![11ace3c95e3ea40a11705af1fb17f773fe1705d35b888d3bf8ca29cfbe6c81ff.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/11ace3c95e3ea40a11705af1fb17f773fe1705d35b888d3bf8ca29cfbe6c81ff.jpg)

![4e0f931f923b0d005c02f222ddd24290757c63c7e7d150a45eddbdc70bf4adce.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/4e0f931f923b0d005c02f222ddd24290757c63c7e7d150a45eddbdc70bf4adce.jpg)

![4f4b044fe92fe5b378a6981377e13570cac3b1074852704d03bf33db687a7f36.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/4f4b044fe92fe5b378a6981377e13570cac3b1074852704d03bf33db687a7f36.jpg)

![55ca7b846ab94714ca41438f4c77a020aa11de12f2b6f6196952ce7e79824ab5.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/55ca7b846ab94714ca41438f4c77a020aa11de12f2b6f6196952ce7e79824ab5.jpg)

![5f2b7ef58dc2e45df818842ffb7e76a26affd1135c0bd36ef28ccb498e3c24cd.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/5f2b7ef58dc2e45df818842ffb7e76a26affd1135c0bd36ef28ccb498e3c24cd.jpg)

![5fe2d214300c3241583249eca976e0da5bb2359f77807370653682978d7c99ef.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/5fe2d214300c3241583249eca976e0da5bb2359f77807370653682978d7c99ef.jpg)

![69f09bee1bd1eaa6f6d9e0947a51827bee84699873b42aa475741977b7568246.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/69f09bee1bd1eaa6f6d9e0947a51827bee84699873b42aa475741977b7568246.jpg)

![776405ed2f1ba742d4ac2f440c8484038f7fbd06ab4f8cf7eb2291cdd3b5161e.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/776405ed2f1ba742d4ac2f440c8484038f7fbd06ab4f8cf7eb2291cdd3b5161e.jpg)

![b4d04d34b581ba820b27111371198e7e2558a06f3b64b487738f13b3f8643679.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/b4d04d34b581ba820b27111371198e7e2558a06f3b64b487738f13b3f8643679.jpg)

![c41f65c549679bff21a0556736aeb769233756abb748c35617dc607e68bbcdd5.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/c41f65c549679bff21a0556736aeb769233756abb748c35617dc607e68bbcdd5.jpg)

![c455bb763412434374275d7a183a0714a8bafd1bfea3d2526c2078a8da091c4b.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/c455bb763412434374275d7a183a0714a8bafd1bfea3d2526c2078a8da091c4b.jpg)

![c56748f0a19360e9099cda6e2936e2f099463b1e3d51bceb4d4a087c7ba67871.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/c56748f0a19360e9099cda6e2936e2f099463b1e3d51bceb4d4a087c7ba67871.jpg)

![cb306b5c92d2212c9e2f2c3c380407916ea60e9f8f91c4b1893a7ecb29908f2a.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/cb306b5c92d2212c9e2f2c3c380407916ea60e9f8f91c4b1893a7ecb29908f2a.jpg)

![ea60dcde8e85ecf40a5ed21014db29cf0aff7f1229a18b7f61125905352c2de1.jpg](../icml_results/247_Adaptive%20Testing%20for%20LLM%20Evaluation_%20A%20Psychometric%20Alternative%20to%20Static%20Benchmarks/tables/ea60dcde8e85ecf40a5ed21014db29cf0aff7f1229a18b7f61125905352c2de1.jpg)

<a id="paper-0243"></a>

## Neuro-evolutionary Continual Reinforcement Learning


### Images

![152d1c806f5b7faabbe17e73a295de568ce2ae3535decab508aebfacf3a93f77.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/images/152d1c806f5b7faabbe17e73a295de568ce2ae3535decab508aebfacf3a93f77.jpg)

![1e7c492cf34bfb3c50e02d4ba836df983221c11fe2b5ce8f3acc91a527d36957.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/images/1e7c492cf34bfb3c50e02d4ba836df983221c11fe2b5ce8f3acc91a527d36957.jpg)

![61cdb0e86f750b4f77afe21f5ad586799de65184600b5cfa182a9a42bc62f57a.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/images/61cdb0e86f750b4f77afe21f5ad586799de65184600b5cfa182a9a42bc62f57a.jpg)

![856f3aedee07c104f96b087e9f0bbb37eef89d4d43a0fd89b0628831186ced6d.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/images/856f3aedee07c104f96b087e9f0bbb37eef89d4d43a0fd89b0628831186ced6d.jpg)

![9147525a9fcc9903ad009721a81095b404d2573a529f4e534317a608859c5490.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/images/9147525a9fcc9903ad009721a81095b404d2573a529f4e534317a608859c5490.jpg)

### Tables

![0abe53c044dc9fc72c2d81fecc4530a8f459d13b7c67407d946af7c55f44657a.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/0abe53c044dc9fc72c2d81fecc4530a8f459d13b7c67407d946af7c55f44657a.jpg)

![79424550eb588331d928446e9a4b9767a063e0d7bec7f0c4d7428fa3eb918e2a.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/79424550eb588331d928446e9a4b9767a063e0d7bec7f0c4d7428fa3eb918e2a.jpg)

![8b07487468bdb551f87af0f302ad51b5d919e42ebc6ef105e59acdf281876899.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/8b07487468bdb551f87af0f302ad51b5d919e42ebc6ef105e59acdf281876899.jpg)

![8e6c4726869a04d051167d79e3d84862c3b67c15a08cdbfd1042c6301990850d.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/8e6c4726869a04d051167d79e3d84862c3b67c15a08cdbfd1042c6301990850d.jpg)

![9e60d59336ad013390f92b524b55c2d43ac5c87923cad14e4185d534e35378c4.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/9e60d59336ad013390f92b524b55c2d43ac5c87923cad14e4185d534e35378c4.jpg)

![b78bd48866dc630112a7ad636664f1d64a027270b120ece273ce6647e9ea3d96.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/b78bd48866dc630112a7ad636664f1d64a027270b120ece273ce6647e9ea3d96.jpg)

![c807f337f45affbb0d47bf40954938ed95da1a329b8fc22acce57bead76e1ce4.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/c807f337f45affbb0d47bf40954938ed95da1a329b8fc22acce57bead76e1ce4.jpg)

![d69d36641cf0c2b8d3d2656283bc2b31a3723e598de2921d694f9e33f16632df.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/d69d36641cf0c2b8d3d2656283bc2b31a3723e598de2921d694f9e33f16632df.jpg)

![e8331492119dd66aa2cc0cc6fa6cda769e348297f36522a99a10f12c22429e51.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/e8331492119dd66aa2cc0cc6fa6cda769e348297f36522a99a10f12c22429e51.jpg)

![f213f72c6c7074d810d5d43d3e6ff472927f2230cf27f2eaea4cc62acbde1895.jpg](../icml_results/248_Neuro-evolutionary%20Continual%20Reinforcement%20Learning/tables/f213f72c6c7074d810d5d43d3e6ff472927f2230cf27f2eaea4cc62acbde1895.jpg)

<a id="paper-0244"></a>

## A Factorized Low-Rank RNN Framework for Uncovering Independent Neural Latent Dynamics and Connectivity


### Images

![06dde42b093138df7f7122af4edb357c64b270ce21bfc61df603c4e4c23362ff.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/06dde42b093138df7f7122af4edb357c64b270ce21bfc61df603c4e4c23362ff.jpg)

![23b35fccffac4be7b627b8580d577ab731f436216428b06018e05cffe778128b.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/23b35fccffac4be7b627b8580d577ab731f436216428b06018e05cffe778128b.jpg)

![240e52b007153bfb660ebba53496aaa850c830b15edb6d8c4f9ce1d377d3c265.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/240e52b007153bfb660ebba53496aaa850c830b15edb6d8c4f9ce1d377d3c265.jpg)

![2ef1f5be281c624ed655ff934082eeafdddd82d316c7c97bb5f70f5ba545cc59.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/2ef1f5be281c624ed655ff934082eeafdddd82d316c7c97bb5f70f5ba545cc59.jpg)

![4ce85aead61a5f1a51e7cf84c9fd8f1adcc06f7bf82adc3203b5b97ecf7d7cdd.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/4ce85aead61a5f1a51e7cf84c9fd8f1adcc06f7bf82adc3203b5b97ecf7d7cdd.jpg)

![59138d2fbeffad0ef601b74965e97927d8a1dcb630fca38e1f931fffc67283bc.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/59138d2fbeffad0ef601b74965e97927d8a1dcb630fca38e1f931fffc67283bc.jpg)

![9b1aaf22e9c2bca2a1f4b9d1ef4b03fc0e067c103d2315ceeff2319c6fd11be7.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/9b1aaf22e9c2bca2a1f4b9d1ef4b03fc0e067c103d2315ceeff2319c6fd11be7.jpg)

![cd78ee80d7c82a58a450d2312a85593030c0851e384309c3f2b56609c453b9db.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/cd78ee80d7c82a58a450d2312a85593030c0851e384309c3f2b56609c453b9db.jpg)

![d25ea0179716d3544fb4e327fe88e6ab15b3056e059bbc5d3751ff53a5f41eeb.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/d25ea0179716d3544fb4e327fe88e6ab15b3056e059bbc5d3751ff53a5f41eeb.jpg)

![e2736ec849e72b40e20f1646c286dac6a2f0bf087b90fa4cfe98f887ef3da082.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/e2736ec849e72b40e20f1646c286dac6a2f0bf087b90fa4cfe98f887ef3da082.jpg)

![eae633b5134e98b97e1c1a126ba63de5c9034ac6c2c7d13956c1db514a4c5d92.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/images/eae633b5134e98b97e1c1a126ba63de5c9034ac6c2c7d13956c1db514a4c5d92.jpg)

### Tables

![b287a914b08ae1cdfb77d688836a37ede2b91f96f4bb1d34ae24dc1d93e16e19.jpg](../icml_results/249_A%20Factorized%20Low-Rank%20RNN%20Framework%20for%20Uncovering%20Independent%20Neural%20Latent%20Dynamics%20and%20Connectivi/tables/b287a914b08ae1cdfb77d688836a37ede2b91f96f4bb1d34ae24dc1d93e16e19.jpg)

<a id="paper-0245"></a>

## From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space


### Images

![14c9e39c9196a0c2663feb7ea5ce90d2ba314e8e23d217e7557fcc2ec20da705.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/14c9e39c9196a0c2663feb7ea5ce90d2ba314e8e23d217e7557fcc2ec20da705.jpg)

![15718b252e8bba97f683120b0b8088e01c34bf6216917ed99d82bc6c9d448f0b.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/15718b252e8bba97f683120b0b8088e01c34bf6216917ed99d82bc6c9d448f0b.jpg)

![3d8dfe21f0b3c19b2e2782200ad8688aab25ef7b415edd5ebbbe825e572e2793.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/3d8dfe21f0b3c19b2e2782200ad8688aab25ef7b415edd5ebbbe825e572e2793.jpg)

![3e55bb74945355bea7202d6cf74b90b3b97dbcc29ecb16ff2a824c33a2dcd46d.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/3e55bb74945355bea7202d6cf74b90b3b97dbcc29ecb16ff2a824c33a2dcd46d.jpg)

![b74cc7903301df6c011057967db9fa5f78bf2d3aee754d14d8daba4bdb991038.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/b74cc7903301df6c011057967db9fa5f78bf2d3aee754d14d8daba4bdb991038.jpg)

![bc3a733db2391d40cdb9ce12a84a02b633df0c427a09c01f29aa06a017a02263.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/bc3a733db2391d40cdb9ce12a84a02b633df0c427a09c01f29aa06a017a02263.jpg)

![d44bdd99d39f630d3441d935bbbe08438309a8f1369e28cb32edcf6edab9525e.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/d44bdd99d39f630d3441d935bbbe08438309a8f1369e28cb32edcf6edab9525e.jpg)

![d59f1fdc2660d663fa24294f855f32878e093c8d7f161869c39dba975aa3ea5f.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/images/d59f1fdc2660d663fa24294f855f32878e093c8d7f161869c39dba975aa3ea5f.jpg)

### Tables

![3ce3aa005bb5783f113f6049b07a6acb8c10a9a84d3acc849a0941018e625d4d.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/3ce3aa005bb5783f113f6049b07a6acb8c10a9a84d3acc849a0941018e625d4d.jpg)

![56c412fe6ba62129d5240e06964346900439de6305a40103baa2f6b88e7e6e98.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/56c412fe6ba62129d5240e06964346900439de6305a40103baa2f6b88e7e6e98.jpg)

![718abcfad6442bc4cf82d27b2780ce353b01c93e85347ff753481ee268d06583.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/718abcfad6442bc4cf82d27b2780ce353b01c93e85347ff753481ee268d06583.jpg)

![75809e8174ad25b6ab4f458d0f660c4f65141a5b8d5f261783de28e195a9672c.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/75809e8174ad25b6ab4f458d0f660c4f65141a5b8d5f261783de28e195a9672c.jpg)

![98a1ecf3765323a7291da00200174607119b10da6144269dcf1e563ef4752d20.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/98a1ecf3765323a7291da00200174607119b10da6144269dcf1e563ef4752d20.jpg)

![cfd370397e910a471439473aae870ed885c49ed0f067c32900b3859a375c0fcf.jpg](../icml_results/250_From%20Text%20to%20Forecasts_%20Bridging%20Modality%20Gap%20with%20Temporal%20Evolution%20Semantic%20Space/tables/cfd370397e910a471439473aae870ed885c49ed0f067c32900b3859a375c0fcf.jpg)

<a id="paper-0246"></a>

## Scaling Robot Policy Evaluation via Discrete Diffusion World Model


### Images

![0d4d70d51c9ae60437a51d0ae003f4872c296331b236826d6bcd9263c6c1afb6.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/0d4d70d51c9ae60437a51d0ae003f4872c296331b236826d6bcd9263c6c1afb6.jpg)

![213c72c8ba9b6f4e419db11231a12abc0b7904f20476a116e2719590044855f1.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/213c72c8ba9b6f4e419db11231a12abc0b7904f20476a116e2719590044855f1.jpg)

![2720f5dd1315e39e7a58c462ad86cebbc3f01ca588e0b1b45551ec968185ece7.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/2720f5dd1315e39e7a58c462ad86cebbc3f01ca588e0b1b45551ec968185ece7.jpg)

![2b332184fcd932ea26f63bd1d9ab7a028b7e5f381994ff938c83031f2017fe59.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/2b332184fcd932ea26f63bd1d9ab7a028b7e5f381994ff938c83031f2017fe59.jpg)

![68d4d225c89997d24a1f7dc6dc31829a65e11026fb58a2f277b0140d1aa6c59a.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/68d4d225c89997d24a1f7dc6dc31829a65e11026fb58a2f277b0140d1aa6c59a.jpg)

![8016b0c304e926f0a76bcad3c386b65da12819d16319ed049d721772b380a89a.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/8016b0c304e926f0a76bcad3c386b65da12819d16319ed049d721772b380a89a.jpg)

![82274e1631d809b7a049bbb0841b9b0d1c11c6e69aff60b78a3872dd2e86424b.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/82274e1631d809b7a049bbb0841b9b0d1c11c6e69aff60b78a3872dd2e86424b.jpg)

![8c4d5de7c41aab183da1fde57a19550eb791d78e49ca417f35436bc2f4812222.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/8c4d5de7c41aab183da1fde57a19550eb791d78e49ca417f35436bc2f4812222.jpg)

![971890e10b68818886264948a8cfd20c6c3225e4718dfce252caf18be2eec66d.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/971890e10b68818886264948a8cfd20c6c3225e4718dfce252caf18be2eec66d.jpg)

![a98849b5784defcf4d6c27e056c14f28d54a6f1132dee909659225c0bcd8ce99.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/a98849b5784defcf4d6c27e056c14f28d54a6f1132dee909659225c0bcd8ce99.jpg)

![b06474b5cc65787132885b624139dd415917f8547a6d33e5b9719530071b1b71.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/b06474b5cc65787132885b624139dd415917f8547a6d33e5b9719530071b1b71.jpg)

![b413da9275c1ce634c92bd16b78029cff6c77c477104894568dc0773944f9539.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/b413da9275c1ce634c92bd16b78029cff6c77c477104894568dc0773944f9539.jpg)

![f48da37cb95d154cf50a5421fb28c3781f92f7776d1dd09dcaaabc078f32dcd6.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/f48da37cb95d154cf50a5421fb28c3781f92f7776d1dd09dcaaabc078f32dcd6.jpg)

![f609b4c5dd7151d54c97a146a29dfc4d75427c2000b8145febecd8c255196976.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/f609b4c5dd7151d54c97a146a29dfc4d75427c2000b8145febecd8c255196976.jpg)

![ffabc260c43df97c95652a49c6c86fbda994e7f42d56fc0f6a2ae4c413337a05.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/images/ffabc260c43df97c95652a49c6c86fbda994e7f42d56fc0f6a2ae4c413337a05.jpg)

### Tables

![20945eed02474ac2cc16018f05db82a2ceab6f0b46c593af1c84a948250f817d.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/tables/20945eed02474ac2cc16018f05db82a2ceab6f0b46c593af1c84a948250f817d.jpg)

![440a148204f0914ea832888afd498af9946ed5f237a740e812728503dda54189.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/tables/440a148204f0914ea832888afd498af9946ed5f237a740e812728503dda54189.jpg)

![46d3a75857f2d4ee30acc2c803a69cb9bb4ad6e5e0140508ff4abd2cb3141ce5.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/tables/46d3a75857f2d4ee30acc2c803a69cb9bb4ad6e5e0140508ff4abd2cb3141ce5.jpg)

![898dd731ab56cae313d9b7f60b5ad9ed1bf5af197fb055f3c4d5b1e7ccae382d.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/tables/898dd731ab56cae313d9b7f60b5ad9ed1bf5af197fb055f3c4d5b1e7ccae382d.jpg)

![a67897ca77dfd0aee5a5e900c59434068accce3bfe8a8b912473000f7f7de118.jpg](../icml_results/251_Scaling%20Robot%20Policy%20Evaluation%20via%20Discrete%20Diffusion%20World%20Model/tables/a67897ca77dfd0aee5a5e900c59434068accce3bfe8a8b912473000f7f7de118.jpg)

<a id="paper-0247"></a>

## Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders


### Images

![0c94dbca7b2f39e81cd5e069a487ec5c5f5f38f684a417284cbff0e3461632eb.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/0c94dbca7b2f39e81cd5e069a487ec5c5f5f38f684a417284cbff0e3461632eb.jpg)

![3e7f3311560c5f334cac7722a73be4191803e6aecea17ebc940a469fd93c30cc.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/3e7f3311560c5f334cac7722a73be4191803e6aecea17ebc940a469fd93c30cc.jpg)

![437601c6116f7de278cc597adeaffe78f188dafa4556d2efe7d2279246ba069c.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/437601c6116f7de278cc597adeaffe78f188dafa4556d2efe7d2279246ba069c.jpg)

![492080c7396837d5c95e0787d284b575b116b99aa4e6af622905de09e89802a4.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/492080c7396837d5c95e0787d284b575b116b99aa4e6af622905de09e89802a4.jpg)

![55823cd6d30115f1b636949f7b2f68b0b7ba4fc3bd09a2d1249168179a6a2d7a.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/55823cd6d30115f1b636949f7b2f68b0b7ba4fc3bd09a2d1249168179a6a2d7a.jpg)

![6a16eb1122510057a1c270730541fd4f0c7bbb402bb1195c796b19651d6a908a.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/6a16eb1122510057a1c270730541fd4f0c7bbb402bb1195c796b19651d6a908a.jpg)

![8f5c28fa9071338abd1d71202270dab10620914d0f11baca711ed00bcdb4fad9.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/8f5c28fa9071338abd1d71202270dab10620914d0f11baca711ed00bcdb4fad9.jpg)

![90a4583a890961012fd10863280e2808046677db892954c25ae82da0a7e1cf3e.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/90a4583a890961012fd10863280e2808046677db892954c25ae82da0a7e1cf3e.jpg)

![9e2fc16e499d9893ba966a1236c4423d7193222f26d927cfe648f84fac34b2fa.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/9e2fc16e499d9893ba966a1236c4423d7193222f26d927cfe648f84fac34b2fa.jpg)

![a42e6dea340ee9c174272512b34706d9f9d1203a263c8193d729efc61d0ebeea.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/a42e6dea340ee9c174272512b34706d9f9d1203a263c8193d729efc61d0ebeea.jpg)

![b2a9480004deca84e4fedd3697ddaf303eb6aab2b07c79ecbb965b06dea1accd.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/b2a9480004deca84e4fedd3697ddaf303eb6aab2b07c79ecbb965b06dea1accd.jpg)

![bb72fd3e72b6efbb22c504a67c1c44a1c59eff51de4a3a7d5f30480b736fd392.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/bb72fd3e72b6efbb22c504a67c1c44a1c59eff51de4a3a7d5f30480b736fd392.jpg)

![c0619588276f7ac30e2da9b0621140f96d923fdb06f09cd0426d06cb3f6d974a.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/c0619588276f7ac30e2da9b0621140f96d923fdb06f09cd0426d06cb3f6d974a.jpg)

![ca2f550f654ffdf51255be8ed84788ec5ae0865008502f8562bba485a05d0587.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/ca2f550f654ffdf51255be8ed84788ec5ae0865008502f8562bba485a05d0587.jpg)

![d162398b244a9ee81d69e8a166b3eb73401a1c6cae4cb1327cfda6326bd123f7.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/d162398b244a9ee81d69e8a166b3eb73401a1c6cae4cb1327cfda6326bd123f7.jpg)

![d251e7746fc0e41c8394b2362a074acb6cb755b0295e65e711d5f4661024cf92.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/d251e7746fc0e41c8394b2362a074acb6cb755b0295e65e711d5f4661024cf92.jpg)

![efa153abbcc799e3b1ca673a42a64f25269c5211df16ad2edf72ca1353fa7d06.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/efa153abbcc799e3b1ca673a42a64f25269c5211df16ad2edf72ca1353fa7d06.jpg)

![fbb93c04a7a8a1865dc6afa6c3ad3e31d1af2796c64bc80ff48b0333fe6ba18b.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/images/fbb93c04a7a8a1865dc6afa6c3ad3e31d1af2796c64bc80ff48b0333fe6ba18b.jpg)

### Tables

![08c5c89461e46a6112aa56588b623759d7870afc1aabf72c314374ecb433180b.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/08c5c89461e46a6112aa56588b623759d7870afc1aabf72c314374ecb433180b.jpg)

![37c7affdb486802dd5882778b6ca79e5179cf47eaf5fb3537662e040ca5c2df7.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/37c7affdb486802dd5882778b6ca79e5179cf47eaf5fb3537662e040ca5c2df7.jpg)

![4337b29e77c8030ab4b180740513c95490ce5b1ff14cc1a41b075411f47c3cb5.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/4337b29e77c8030ab4b180740513c95490ce5b1ff14cc1a41b075411f47c3cb5.jpg)

![4a6517aae685265ff3e1c17eaac6647d01d59585ab00b1b7684729b52139d994.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/4a6517aae685265ff3e1c17eaac6647d01d59585ab00b1b7684729b52139d994.jpg)

![606fd24c278e9ed4d81103f7fd5664fbf60c5b2a7900cdc23de4ee8a4b834976.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/606fd24c278e9ed4d81103f7fd5664fbf60c5b2a7900cdc23de4ee8a4b834976.jpg)

![609c3b021377b02d1da2f8cac0abc9e503229c1099afe264b378d1c2bad777f1.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/609c3b021377b02d1da2f8cac0abc9e503229c1099afe264b378d1c2bad777f1.jpg)

![6abf39339b967b74175fcd7f8f06c2f1ae103650b8732dc37d3088a8c806eb64.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/6abf39339b967b74175fcd7f8f06c2f1ae103650b8732dc37d3088a8c806eb64.jpg)

![729baabc3db50c56a2b8a7e9e540502d676c9cb17401f717f0b879502fdc63b8.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/729baabc3db50c56a2b8a7e9e540502d676c9cb17401f717f0b879502fdc63b8.jpg)

![78120ac22cbb344191bea5a2c74521c3179a7fed3f5199b66d64e02bd189d744.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/78120ac22cbb344191bea5a2c74521c3179a7fed3f5199b66d64e02bd189d744.jpg)

![8bbbec3c2edc4186e973f4237b90f982173d669d78aea59af6d78b2d1a50419e.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/8bbbec3c2edc4186e973f4237b90f982173d669d78aea59af6d78b2d1a50419e.jpg)

![93d45b52ff5e844ab233b3838116943f532d7b496bc005fcf2dd772e8b061c3d.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/93d45b52ff5e844ab233b3838116943f532d7b496bc005fcf2dd772e8b061c3d.jpg)

![97f6bbf8410dbd134e381a6deb68bb247ea4838c30d55833178dd574fc4d640d.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/97f6bbf8410dbd134e381a6deb68bb247ea4838c30d55833178dd574fc4d640d.jpg)

![9d35fd6605b878ca36d306ec7b69a2409c40918ac6268a6a661c6e33bc1283b3.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/9d35fd6605b878ca36d306ec7b69a2409c40918ac6268a6a661c6e33bc1283b3.jpg)

![9e311b0240f14ddd198dae23159fcc87164ccb5d088a7e60d22bed95d65da055.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/9e311b0240f14ddd198dae23159fcc87164ccb5d088a7e60d22bed95d65da055.jpg)

![a7d23026c77284d50b41bf26e84a7d9d661ad16bfa1856725cc29f092fdc5f0f.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/a7d23026c77284d50b41bf26e84a7d9d661ad16bfa1856725cc29f092fdc5f0f.jpg)

![ab6e333b07b173d75161086025c5ca8fff35c8a047214182c0df086439889130.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/ab6e333b07b173d75161086025c5ca8fff35c8a047214182c0df086439889130.jpg)

![be6f998e020ffe7e844fdd35ed4c9a9368d055990e8dbfd9e2081b70b159c97e.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/be6f998e020ffe7e844fdd35ed4c9a9368d055990e8dbfd9e2081b70b159c97e.jpg)

![c5cb4bec76a6103fa9865a9e005eb3fb0cb745592b118932ced27aac9fe27d6b.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/c5cb4bec76a6103fa9865a9e005eb3fb0cb745592b118932ced27aac9fe27d6b.jpg)

![c8d4201a7bb3f22098f82147f11721900c46f1f78210d4d8351eee1055e6887a.jpg](../icml_results/252_Less%20is%20Enough_%20Synthesizing%20Diverse%20Data%20in%20LLM%20Feature%20Space%20with%20Sparse%20Autoencoders/tables/c8d4201a7bb3f22098f82147f11721900c46f1f78210d4d8351eee1055e6887a.jpg)

<a id="paper-0248"></a>

## A Recursive Decomposition Framework for Causal Structure Learning in the Presence of Latent Variables


### Images

![0fcf0892fe837c6d3cfec628b5a1039ee9bdd52b4e7daeeb314a8e3bd7bd1157.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/0fcf0892fe837c6d3cfec628b5a1039ee9bdd52b4e7daeeb314a8e3bd7bd1157.jpg)

![1cfaf5a57c6b96a0431f1257a605393cd41151ca0c0fcb17c04cd380f6677150.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/1cfaf5a57c6b96a0431f1257a605393cd41151ca0c0fcb17c04cd380f6677150.jpg)

![2b5f7dfde2c78959d8edd520f335a700a15d1c29132cde93d92aa976ee1e0821.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/2b5f7dfde2c78959d8edd520f335a700a15d1c29132cde93d92aa976ee1e0821.jpg)

![3142a91c91a4483a928725e714e39857ccb06abf88a82fe176c57f596860ec24.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/3142a91c91a4483a928725e714e39857ccb06abf88a82fe176c57f596860ec24.jpg)

![42ef493b5e88801f901554c4cb9d138a7e13ac706a66835c783a1d91df918061.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/42ef493b5e88801f901554c4cb9d138a7e13ac706a66835c783a1d91df918061.jpg)

![68f3fc824790b1060136aae3b2d48e2cd8f8768534010f2ad01eff384d8f8fa3.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/68f3fc824790b1060136aae3b2d48e2cd8f8768534010f2ad01eff384d8f8fa3.jpg)

![7cd2959c32e6db3d2ccb825a8cde79febe8de1e603496e8157c73a5a1c5ce911.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/7cd2959c32e6db3d2ccb825a8cde79febe8de1e603496e8157c73a5a1c5ce911.jpg)

![955a2db358f099a66d6a8d8804df0e44b1fccc0f1462a675d850c492c23fbcf7.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/955a2db358f099a66d6a8d8804df0e44b1fccc0f1462a675d850c492c23fbcf7.jpg)

![b40619f4cce4cebcdcfaf2bca3b4d928c65e692a68db07a1c53f85ea08a67e27.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/b40619f4cce4cebcdcfaf2bca3b4d928c65e692a68db07a1c53f85ea08a67e27.jpg)

![ccffad8a880b6893993bd48958f691d3c60e14b186eefa78794feb59288a8a5c.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/ccffad8a880b6893993bd48958f691d3c60e14b186eefa78794feb59288a8a5c.jpg)

![cd9ef10fb5650c5b99b2f9c5386b1b5556c014591fc4e86de8bb849da009c909.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/cd9ef10fb5650c5b99b2f9c5386b1b5556c014591fc4e86de8bb849da009c909.jpg)

![d774cbe3b24c61ca2bcd3b6dd7dec6cd89f087cc7eba9d4917a2c4bbb051c4ff.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/d774cbe3b24c61ca2bcd3b6dd7dec6cd89f087cc7eba9d4917a2c4bbb051c4ff.jpg)

![e9effbcd5b6b2284f17d4a6f4167c253a31781967989dd2680ea87ed586006c0.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/e9effbcd5b6b2284f17d4a6f4167c253a31781967989dd2680ea87ed586006c0.jpg)

![f6fea1b872f1705745dc704f8603ff576786b03e0352d16c1e8c24cf72fce372.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/images/f6fea1b872f1705745dc704f8603ff576786b03e0352d16c1e8c24cf72fce372.jpg)

### Tables

![4d86b00ff0cf6ed85b6b736ebd9e4f1a55beae558427487578ed43e2f33f1f07.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/tables/4d86b00ff0cf6ed85b6b736ebd9e4f1a55beae558427487578ed43e2f33f1f07.jpg)

![813233de2580d4823378c6bd44596f6f0e9410c712e41c7c303fb06665777c74.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/tables/813233de2580d4823378c6bd44596f6f0e9410c712e41c7c303fb06665777c74.jpg)

![cba08e0a8d50ecab2cfeed09116475866bfe9a1fc4126e546f111d081b17b110.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/tables/cba08e0a8d50ecab2cfeed09116475866bfe9a1fc4126e546f111d081b17b110.jpg)

![e86f618f51a63ac6802b0ed9a8f5fc19b2f8318de8a25745947a3ada49b1e43c.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/tables/e86f618f51a63ac6802b0ed9a8f5fc19b2f8318de8a25745947a3ada49b1e43c.jpg)

![f45be74f62c8f6fb6725d54f87a90012bea2d3189475a95d78ec002b32a8b4a5.jpg](../icml_results/253_A%20Recursive%20Decomposition%20Framework%20for%20Causal%20Structure%20Learning%20in%20the%20Presence%20of%20Latent%20Variable/tables/f45be74f62c8f6fb6725d54f87a90012bea2d3189475a95d78ec002b32a8b4a5.jpg)

<a id="paper-0249"></a>

## SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion


### Images

![0238f04b71daff1595c35e67c381f1e1149f8b1693528ca79c9a166acf1f651b.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/0238f04b71daff1595c35e67c381f1e1149f8b1693528ca79c9a166acf1f651b.jpg)

![0251c413c7531cb41fd33aea091f0da685801999695a44b3acdd895fbbd19328.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/0251c413c7531cb41fd33aea091f0da685801999695a44b3acdd895fbbd19328.jpg)

![068502b4cd1e0a8eaaf3403f2c73039d2b47bfc020225812d6ae48261ca4792c.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/068502b4cd1e0a8eaaf3403f2c73039d2b47bfc020225812d6ae48261ca4792c.jpg)

![3cc9cbe2014b461d9efe2d9a4b568005b39ce158f67d007dd7359730f13a35a3.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/3cc9cbe2014b461d9efe2d9a4b568005b39ce158f67d007dd7359730f13a35a3.jpg)

![53f83946751cbafc3ce527ac0ff3cdce2e2c23ddcb5e4833a80dd29078d4a932.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/53f83946751cbafc3ce527ac0ff3cdce2e2c23ddcb5e4833a80dd29078d4a932.jpg)

![74c3ac307eb62b300580c7dd66ab2b486f2ecebcedc4a4a7b04b5fd0a4db6a9b.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/74c3ac307eb62b300580c7dd66ab2b486f2ecebcedc4a4a7b04b5fd0a4db6a9b.jpg)

![a6ff562f8ea4f67d7d4c0fbff1035fd952af4437adbbb3c7a905c7bec35047ba.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/a6ff562f8ea4f67d7d4c0fbff1035fd952af4437adbbb3c7a905c7bec35047ba.jpg)

![a9420ab1e4a3fa177b995f7672bb3fe94bb5434b6e6fb8f875acca2d85884447.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/a9420ab1e4a3fa177b995f7672bb3fe94bb5434b6e6fb8f875acca2d85884447.jpg)

![c718a2d8b1251e1cbe5c18b8d8e166041454f0f3fce25770e1a89508868a4332.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/c718a2d8b1251e1cbe5c18b8d8e166041454f0f3fce25770e1a89508868a4332.jpg)

![c9d65d26b4db49fb52e6c7590973a16ebfc7d005bbc8ceebac7b9eb10572e173.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/c9d65d26b4db49fb52e6c7590973a16ebfc7d005bbc8ceebac7b9eb10572e173.jpg)

![cb72b7491ecb88499d62cae6283ff80a08359b79f9684fafdf8562d83da399a5.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/cb72b7491ecb88499d62cae6283ff80a08359b79f9684fafdf8562d83da399a5.jpg)

![d317f298807c6ba3af1dbb0cb1ced2b12d33570d15be278b813d8be2f57d64c4.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/d317f298807c6ba3af1dbb0cb1ced2b12d33570d15be278b813d8be2f57d64c4.jpg)

![d633b56a460187d08258755a659506a124ca99b282f38a8cbadb5db747225ee4.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/d633b56a460187d08258755a659506a124ca99b282f38a8cbadb5db747225ee4.jpg)

![dd2e73137355ed914ce738694feae8fe61a92dae96af5c6af904fff15dddeabf.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/dd2e73137355ed914ce738694feae8fe61a92dae96af5c6af904fff15dddeabf.jpg)

![e512bbdfaa427a2643cfbcc724c6ed5d1dcf9ed6e98ff25a0877dc8724d4359e.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/e512bbdfaa427a2643cfbcc724c6ed5d1dcf9ed6e98ff25a0877dc8724d4359e.jpg)

![ecbfb48d62b8725d95ebaccbce258060fa3c4702d2d87de31bdf2c845966fac9.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/images/ecbfb48d62b8725d95ebaccbce258060fa3c4702d2d87de31bdf2c845966fac9.jpg)

### Tables

![37786a19f1cfbe55d27d460454d94bfa07e7698d91941a6941c159a3ea3fbf74.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/37786a19f1cfbe55d27d460454d94bfa07e7698d91941a6941c159a3ea3fbf74.jpg)

![4206d80665d23a19c14b058420f80435c426523d6033e9721ef4823db55f6499.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/4206d80665d23a19c14b058420f80435c426523d6033e9721ef4823db55f6499.jpg)

![4c4321704810358b94a3e23fd1ba01b29b3b5f9196a599eb3b341a4a0bf80068.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/4c4321704810358b94a3e23fd1ba01b29b3b5f9196a599eb3b341a4a0bf80068.jpg)

![74f3b727ff080895ce003c4e16e6135f6c422e87a5e481e6c061fe7bdf5fe537.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/74f3b727ff080895ce003c4e16e6135f6c422e87a5e481e6c061fe7bdf5fe537.jpg)

![907bf4791829072fbe67cfec8e51a9b13bbebf3a4e084865ed90aea92d6cfdee.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/907bf4791829072fbe67cfec8e51a9b13bbebf3a4e084865ed90aea92d6cfdee.jpg)

![9c74c8986978b47a925f60f1d5b6150b3b93319ba385cf010910d87bf40e1db9.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/9c74c8986978b47a925f60f1d5b6150b3b93319ba385cf010910d87bf40e1db9.jpg)

![b08343f3944961a901dbd7ef0fcd6ccaf76621a21d711ca7d247fb5a4232ddbc.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/b08343f3944961a901dbd7ef0fcd6ccaf76621a21d711ca7d247fb5a4232ddbc.jpg)

![b1f9795357c4e30c7c76f0ca27087475240606870e8d0a35dbd1257e43f0ffc8.jpg](../icml_results/254_SplAttN_%20Bridging%202D%20and%203D%20with%20Gaussian%20Soft%20Splatting%20and%20Attention%20for%20Point%20Cloud%20Completion/tables/b1f9795357c4e30c7c76f0ca27087475240606870e8d0a35dbd1257e43f0ffc8.jpg)

<a id="paper-0250"></a>

## Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding


### Images

![2b363204ad71f8b269152f9b45091b4bd9eefb70cec5c435fa740c4acba2cb90.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/2b363204ad71f8b269152f9b45091b4bd9eefb70cec5c435fa740c4acba2cb90.jpg)

![5c929e3e2c76113db03c3a2af8ede1e09d350d49f99c6f74b65dc15c2054dd0c.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/5c929e3e2c76113db03c3a2af8ede1e09d350d49f99c6f74b65dc15c2054dd0c.jpg)

![643785c00c6c310ad8d91b9ee9aff4804c596298efb7087a35d593546d4e8cd0.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/643785c00c6c310ad8d91b9ee9aff4804c596298efb7087a35d593546d4e8cd0.jpg)

![64bba8678d56a61ef368ca27945790101a305cc0716719400087dd2c96e850b6.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/64bba8678d56a61ef368ca27945790101a305cc0716719400087dd2c96e850b6.jpg)

![6be73d73e94f67c6c62d1424ffb653516e5f922b8975fe891819d6bf01efe922.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/6be73d73e94f67c6c62d1424ffb653516e5f922b8975fe891819d6bf01efe922.jpg)

![a2dd14e9e909d39c399cc2bcb0179282831183dbeba42eec3cd4d979f46a05ff.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/a2dd14e9e909d39c399cc2bcb0179282831183dbeba42eec3cd4d979f46a05ff.jpg)

![eaa409ef34e10843c3f23a659325d5b0291275aa834f9c404f843099e58df683.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/images/eaa409ef34e10843c3f23a659325d5b0291275aa834f9c404f843099e58df683.jpg)

### Tables

![79856e0ab35f73ae411b2b08ac16bb60ece4f03378bfe2b9b7dedd9f710145a1.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/tables/79856e0ab35f73ae411b2b08ac16bb60ece4f03378bfe2b9b7dedd9f710145a1.jpg)

![8f2ef61c4ebfc86821155edff69fa021ffdd5bd1becc4fc695d543c136df3cd8.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/tables/8f2ef61c4ebfc86821155edff69fa021ffdd5bd1becc4fc695d543c136df3cd8.jpg)

![b0202b50a92c1568ff64a92ab6931e4c847d356fc84e54b2d28a57df394d41ae.jpg](../icml_results/255_Towards%20High-Fidelity%20CAD%20Generation%20via%20LLM-Driven%20Program%20Generation%20and%20Text-Based%20B-Rep%20Primitiv/tables/b0202b50a92c1568ff64a92ab6931e4c847d356fc84e54b2d28a57df394d41ae.jpg)

<a id="paper-0251"></a>

## ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models


### Images

![0530ae3a11c5c8e52b9431cc59a8e7ffa00c1be08cb9c272bc32f70724f2fad2.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/0530ae3a11c5c8e52b9431cc59a8e7ffa00c1be08cb9c272bc32f70724f2fad2.jpg)

![0e292d4cee5a5ef3dde6458ff01e78761172fa97bd17e523fdea1ea59f053394.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/0e292d4cee5a5ef3dde6458ff01e78761172fa97bd17e523fdea1ea59f053394.jpg)

![0ea22d4c9f8f628f01a097d4158bf994ef46432cb74d2e222bbf63eca1fd8d52.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/0ea22d4c9f8f628f01a097d4158bf994ef46432cb74d2e222bbf63eca1fd8d52.jpg)

![18f0319b5f66c490b3fbe586017b0754f4993be05c3e206342f15cbbcb36bbcc.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/18f0319b5f66c490b3fbe586017b0754f4993be05c3e206342f15cbbcb36bbcc.jpg)

![2ea606f6ff60c92aa4b21a078647a471c8ac3f71ed6022e54d9e6a37a7c08005.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/2ea606f6ff60c92aa4b21a078647a471c8ac3f71ed6022e54d9e6a37a7c08005.jpg)

![49189ab0330fcf0d38d16693b405b45e8536093c3f3c4da81b5e8aacf31bb037.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/49189ab0330fcf0d38d16693b405b45e8536093c3f3c4da81b5e8aacf31bb037.jpg)

![49b113ae91094928cad274e6533c3574b74f61dc18214a6fe2547ba73cfd7b1b.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/49b113ae91094928cad274e6533c3574b74f61dc18214a6fe2547ba73cfd7b1b.jpg)

![b55a4b913f7c38b45158c8c8f62ceb07871c0a8a5c42d6782248533bc76abb19.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/b55a4b913f7c38b45158c8c8f62ceb07871c0a8a5c42d6782248533bc76abb19.jpg)

![b799da92b83005362575f49f3b265649e02b0dc904f872e05cc244c67f8e6e01.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/b799da92b83005362575f49f3b265649e02b0dc904f872e05cc244c67f8e6e01.jpg)

![d6df3f7c73c9585f420d4c029c776dbd754e57a789fb8019166e9174494d383d.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/d6df3f7c73c9585f420d4c029c776dbd754e57a789fb8019166e9174494d383d.jpg)

![edd91b59a291d475c744c0769cfeb81cd71d7982ad6af1e40c78a243b4594d82.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/images/edd91b59a291d475c744c0769cfeb81cd71d7982ad6af1e40c78a243b4594d82.jpg)

### Tables

![1638360e426b5d90ef93326db751101326dbb10f97fa999148af891f27855f83.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/1638360e426b5d90ef93326db751101326dbb10f97fa999148af891f27855f83.jpg)

![2ac351dd590a49d4707718d82a619b7c2b67d3e14363314a7e73cdb18f541d9d.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/2ac351dd590a49d4707718d82a619b7c2b67d3e14363314a7e73cdb18f541d9d.jpg)

![3278d2cb2c12601f2fd6b8cbd9c82b81fa4c271e6eea4d1cc767c9e7860a8c7f.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/3278d2cb2c12601f2fd6b8cbd9c82b81fa4c271e6eea4d1cc767c9e7860a8c7f.jpg)

![55b8bbae0c0388095771bb29cdf00dc06723a0b25b3d66d3ee000cc9f02cc692.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/55b8bbae0c0388095771bb29cdf00dc06723a0b25b3d66d3ee000cc9f02cc692.jpg)

![6846551b3f4f8a0aa03b48624b8735a51950c9b016db16970c2492c59f261c0b.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/6846551b3f4f8a0aa03b48624b8735a51950c9b016db16970c2492c59f261c0b.jpg)

![6ff1bc5ce177047d5f296b2253226569dd3f0d3bb6f6e3d05d8721ed596032d5.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/6ff1bc5ce177047d5f296b2253226569dd3f0d3bb6f6e3d05d8721ed596032d5.jpg)

![758cb2bee9ed04c4099c83177d0a77de03f6555294a9d3f892a8f25dfbb593ff.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/758cb2bee9ed04c4099c83177d0a77de03f6555294a9d3f892a8f25dfbb593ff.jpg)

![7ef32272e16bd911eb899bad3e4f34152ab6211c74ce71a2f9387eafe9e86c9a.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/7ef32272e16bd911eb899bad3e4f34152ab6211c74ce71a2f9387eafe9e86c9a.jpg)

![804d77e818a7604b6b49e02757ca65f90c661235c0ab06f04b5adf42e073c10a.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/804d77e818a7604b6b49e02757ca65f90c661235c0ab06f04b5adf42e073c10a.jpg)

![e781ae8595091efa7659d4e8dbbc2de11aff1716f7a0f3226cde96b3a25b71fc.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/e781ae8595091efa7659d4e8dbbc2de11aff1716f7a0f3226cde96b3a25b71fc.jpg)

![fbd9e7a2177c4944eca22c8c73feb3e6679cf2ca5bc2861397c294d34a1d879c.jpg](../icml_results/256_ThreadWeaver_%20Adaptive%20Threading%20for%20Efficient%20Parallel%20Reasoning%20in%20Language%20Models/tables/fbd9e7a2177c4944eca22c8c73feb3e6679cf2ca5bc2861397c294d34a1d879c.jpg)

<a id="paper-0252"></a>

## PLANTAIN: Plan-Answer Interleaved Reasoning


### Images

![9c95be3be0de4e6d2833dc423ec8448de8babd518f349ec10a991ed96ed36963.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/images/9c95be3be0de4e6d2833dc423ec8448de8babd518f349ec10a991ed96ed36963.jpg)

### Tables

![3fecd44f0d702a3f497c54632d78af9e1b760976683ea6167be76a9b0eaf73e7.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/3fecd44f0d702a3f497c54632d78af9e1b760976683ea6167be76a9b0eaf73e7.jpg)

![63c456bf5dc6df9967218b29a7004094f49dd77c3125da7911103b59f524ff7c.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/63c456bf5dc6df9967218b29a7004094f49dd77c3125da7911103b59f524ff7c.jpg)

![659d7a3a133915b29a8dc3b1d345a6d0454a4d402a6a11122ee89357ea0be5b5.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/659d7a3a133915b29a8dc3b1d345a6d0454a4d402a6a11122ee89357ea0be5b5.jpg)

![9b2f5b67c65d067e4c82012326ec99a09c394d590ada683aaa68fc0516f7ed0c.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/9b2f5b67c65d067e4c82012326ec99a09c394d590ada683aaa68fc0516f7ed0c.jpg)

![ac8511242d402fbe9629f15da66bf8157c1cdcc7a233d28254b9df5ed507fc88.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/ac8511242d402fbe9629f15da66bf8157c1cdcc7a233d28254b9df5ed507fc88.jpg)

![be63353cfb24952a8f1a838ed88d29d8fbfbb053b14371c9c481226da9eaa9ff.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/be63353cfb24952a8f1a838ed88d29d8fbfbb053b14371c9c481226da9eaa9ff.jpg)

![c862ecc949a3b292d4a2193d1e9ee47f2916c351f0588c04d27ad6e6c7dd147a.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/c862ecc949a3b292d4a2193d1e9ee47f2916c351f0588c04d27ad6e6c7dd147a.jpg)

![fe7b0a017c99c2645c222285b3eed670f2fd44a324eef5dfca714b52503c670c.jpg](../icml_results/257_PLANTAIN_%20Plan-Answer%20Interleaved%20Reasoning/tables/fe7b0a017c99c2645c222285b3eed670f2fd44a324eef5dfca714b52503c670c.jpg)

<a id="paper-0253"></a>

## Prototype-guided Bilateral Alignment Multimodal Federated Learning


### Images

![1145a813b6735017053123ed79d9f1ded7506f83817a9cabdd94412a830f6733.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/1145a813b6735017053123ed79d9f1ded7506f83817a9cabdd94412a830f6733.jpg)

![23e7252351552096ee887344f7c671ce35178d64f44e862ceaff6ce4ba474414.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/23e7252351552096ee887344f7c671ce35178d64f44e862ceaff6ce4ba474414.jpg)

![3a1dc015184e5b6ddc8dd68557bc2037a3bb7398feadd9e24c7375592b932210.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/3a1dc015184e5b6ddc8dd68557bc2037a3bb7398feadd9e24c7375592b932210.jpg)

![4304e5571ff33d13f36c2e1ce31f3f7fb5059de7c54fc13b004a9ad4fab6cdb4.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/4304e5571ff33d13f36c2e1ce31f3f7fb5059de7c54fc13b004a9ad4fab6cdb4.jpg)

![46f2761fd6a5bf56932a2ad80f77b3fb527f428c450a89f33efa972fa23f1dce.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/46f2761fd6a5bf56932a2ad80f77b3fb527f428c450a89f33efa972fa23f1dce.jpg)

![70120b4a4e2fab5c484e99d1b8a13e8d00af7c14dc08b860be109c38a70100a1.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/70120b4a4e2fab5c484e99d1b8a13e8d00af7c14dc08b860be109c38a70100a1.jpg)

![8bb0be17dd7e79b95cac163ff71f97e71c2f5231e627ad03e5120a06b98e8eeb.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/8bb0be17dd7e79b95cac163ff71f97e71c2f5231e627ad03e5120a06b98e8eeb.jpg)

![8e145774bff7f914ab09293dd0066efb607999a97276c0d002203e6db70723fe.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/8e145774bff7f914ab09293dd0066efb607999a97276c0d002203e6db70723fe.jpg)

![928237339ddd4a5f046a6adce2f8ec27b0d9a09c80d31bee2c767246559f7462.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/928237339ddd4a5f046a6adce2f8ec27b0d9a09c80d31bee2c767246559f7462.jpg)

![9a9b3d5f990797c5927973378be55eef4d39dee8ffc903e5d9ee6c6f845828dc.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/9a9b3d5f990797c5927973378be55eef4d39dee8ffc903e5d9ee6c6f845828dc.jpg)

![b42bf143f062cc4bf59f72fd4e267b967ffcd9683cc79eb73861c6fe777b2652.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/b42bf143f062cc4bf59f72fd4e267b967ffcd9683cc79eb73861c6fe777b2652.jpg)

![b96fde3998975eba5750d347d5fa23a2478358adb88359a48812cf6f2856f657.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/b96fde3998975eba5750d347d5fa23a2478358adb88359a48812cf6f2856f657.jpg)

![bb22146a85994fda956e580774b291badd472615468031156d0093abc3e8786d.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/bb22146a85994fda956e580774b291badd472615468031156d0093abc3e8786d.jpg)

![c0cfa5a0d15b51ff5c6561cc7ba2c21d65d3105d2485d17c1a9cc528c8d4b8cd.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/c0cfa5a0d15b51ff5c6561cc7ba2c21d65d3105d2485d17c1a9cc528c8d4b8cd.jpg)

![c274b593865fdaa8899fa407c3659ad04ea03c417bbcab46910323ca6161b212.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/c274b593865fdaa8899fa407c3659ad04ea03c417bbcab46910323ca6161b212.jpg)

![cb974d715f6e3cc10aa2e77542e0e2ff8fe859ab10296066fbcf47c9e91069ad.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/cb974d715f6e3cc10aa2e77542e0e2ff8fe859ab10296066fbcf47c9e91069ad.jpg)

![e56a4ace7f8a175483055ca9fb52c143bc72371524008bc5ee7f5af2d0fd31c7.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/images/e56a4ace7f8a175483055ca9fb52c143bc72371524008bc5ee7f5af2d0fd31c7.jpg)

### Tables

![124ea8d3aad6dd1cbe0141c57457e1d989b36b21445bee79350f2e7f6a26c15a.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/124ea8d3aad6dd1cbe0141c57457e1d989b36b21445bee79350f2e7f6a26c15a.jpg)

![252ec7c03e1a7f2b227c83b6c2673c3be189ec2d66de140c04c3c1e407d4bee6.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/252ec7c03e1a7f2b227c83b6c2673c3be189ec2d66de140c04c3c1e407d4bee6.jpg)

![510e4bc2c05bbc0d4a796a244443c93cfe5dcff200da3e948f01dfd045c5dce5.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/510e4bc2c05bbc0d4a796a244443c93cfe5dcff200da3e948f01dfd045c5dce5.jpg)

![7783d365d7c09d158a05f9b68f513c98391024c345ba73c8f7ebe7fb67620a49.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/7783d365d7c09d158a05f9b68f513c98391024c345ba73c8f7ebe7fb67620a49.jpg)

![b28a4e74e754a48a2388da399f97bce2d4a9af3b7f1b7a748359d234552183be.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/b28a4e74e754a48a2388da399f97bce2d4a9af3b7f1b7a748359d234552183be.jpg)

![d4aecab0e1a0fff32353868eb1858716ace6e22a5c56db85a011f751a15cd46c.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/d4aecab0e1a0fff32353868eb1858716ace6e22a5c56db85a011f751a15cd46c.jpg)

![e38b14abad648f913eb8c2d2d94ffb31acb10089e2e79fcae1218e39be3173f3.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/e38b14abad648f913eb8c2d2d94ffb31acb10089e2e79fcae1218e39be3173f3.jpg)

![e9a747028c586a37d54caad5873863a9140a00141f558c61723627e04fa9d1ef.jpg](../icml_results/258_Prototype-guided%20Bilateral%20Alignment%20Multimodal%20Federated%20Learning/tables/e9a747028c586a37d54caad5873863a9140a00141f558c61723627e04fa9d1ef.jpg)

<a id="paper-0254"></a>

## WaterSIC: Information-Theoretically (Near) Optimal Linear Layer Quantization


### Images

![2bab7abf75c6e674d665094385b27a21e03d0ed87716d6b2417f39729b26247f.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/2bab7abf75c6e674d665094385b27a21e03d0ed87716d6b2417f39729b26247f.jpg)

![35a12f37ac429d2228c75941fb0c808cf05dd62e8c19c6bf7b5c5f0ff11999d3.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/35a12f37ac429d2228c75941fb0c808cf05dd62e8c19c6bf7b5c5f0ff11999d3.jpg)

![3d9afa18d26cd0dc4f5c8eeafb4933e4159bbc9718830129b8cdffa5fc990a39.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/3d9afa18d26cd0dc4f5c8eeafb4933e4159bbc9718830129b8cdffa5fc990a39.jpg)

![53dfb15be43fc06e13bb1c644b9c402b83b753f677a47d07a239720d3bbd3d2d.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/53dfb15be43fc06e13bb1c644b9c402b83b753f677a47d07a239720d3bbd3d2d.jpg)

![61198c3c4f6ec0591a14f45c856595c2ab5b13ed000fe49328e46154b76958b9.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/61198c3c4f6ec0591a14f45c856595c2ab5b13ed000fe49328e46154b76958b9.jpg)

![64d4845959e428b35439f329a8e4a4af4d317878017f55d7f9c426c82ae7c59d.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/64d4845959e428b35439f329a8e4a4af4d317878017f55d7f9c426c82ae7c59d.jpg)

![6f1117eafd4c86fe6e490ef2057fdf1d8bc31de382ddee9895bfe8ea2cab5695.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/6f1117eafd4c86fe6e490ef2057fdf1d8bc31de382ddee9895bfe8ea2cab5695.jpg)

![791c8f644f444431ac6bc97fcc6795de4de7908f4f9093c0d53ca03d7b735937.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/791c8f644f444431ac6bc97fcc6795de4de7908f4f9093c0d53ca03d7b735937.jpg)

![8361b9aec9a9e7c5490c580a4cd0efce811fe59c1f141c6d28dc00002912b2b0.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/8361b9aec9a9e7c5490c580a4cd0efce811fe59c1f141c6d28dc00002912b2b0.jpg)

![952398a844232296352a22e98fd61023e719a28a8168965ef6959d297465f6d4.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/952398a844232296352a22e98fd61023e719a28a8168965ef6959d297465f6d4.jpg)

![b303de59f14493877136d8863187ab44aaf4ea9347f2e3a960866c73e51cf354.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/b303de59f14493877136d8863187ab44aaf4ea9347f2e3a960866c73e51cf354.jpg)

![b65e609c90326e31a8738bc3918317a218e805c454ebbdf09877b7f5f62c34c2.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/b65e609c90326e31a8738bc3918317a218e805c454ebbdf09877b7f5f62c34c2.jpg)

![c53d7f8f13fac18e65aefd287e4be765cf8d6b8df6da6de42188c7a54ae78474.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/c53d7f8f13fac18e65aefd287e4be765cf8d6b8df6da6de42188c7a54ae78474.jpg)

![e5f304021726214468701cad2dff84c1f4e4dd2b50657404b2eaa9f94854f2b6.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/e5f304021726214468701cad2dff84c1f4e4dd2b50657404b2eaa9f94854f2b6.jpg)

![f07202ce543cd98e36368f769f3c93cdf5366fa9b14f70ef47bd2776e67d66f0.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/images/f07202ce543cd98e36368f769f3c93cdf5366fa9b14f70ef47bd2776e67d66f0.jpg)

### Tables

![0d88c77b651922befc2b9da7c9fcdf6d1f42bf5419b6e60926c890b537ef01e5.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/0d88c77b651922befc2b9da7c9fcdf6d1f42bf5419b6e60926c890b537ef01e5.jpg)

![117fa998c4bf30dd3db17ba1a44250caad536f922ba1ac49a196dd95bb8c4425.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/117fa998c4bf30dd3db17ba1a44250caad536f922ba1ac49a196dd95bb8c4425.jpg)

![236ce9ab58473f223ec591b0337b5e4e7da91dcc8d9b72b3c02aadf8b428b2b6.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/236ce9ab58473f223ec591b0337b5e4e7da91dcc8d9b72b3c02aadf8b428b2b6.jpg)

![23d7340e5fc3485ac97cd4b9b00b9222ed9c808d0c40d4e438ec08f289803f31.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/23d7340e5fc3485ac97cd4b9b00b9222ed9c808d0c40d4e438ec08f289803f31.jpg)

![3545bb07fd205a38462e713a1a37ca0807481735bb32294ee068ab5938894657.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/3545bb07fd205a38462e713a1a37ca0807481735bb32294ee068ab5938894657.jpg)

![3a5458e310c9bc7bc5f17e82a03007057556e51c70f327d43d9381bddca30d7a.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/3a5458e310c9bc7bc5f17e82a03007057556e51c70f327d43d9381bddca30d7a.jpg)

![3e9f2ec828a579dc4a91aeab33c0628736e8b5db98bc48e3160c63bd79d014c6.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/3e9f2ec828a579dc4a91aeab33c0628736e8b5db98bc48e3160c63bd79d014c6.jpg)

![463515b849c54655026a1e1bb6f6d3a9f86dd9b6b5bf2c2228cfcbb6db06086f.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/463515b849c54655026a1e1bb6f6d3a9f86dd9b6b5bf2c2228cfcbb6db06086f.jpg)

![5a47ac0ad12e2a34d52e5f3b1331ad254c285d0756572db198c4f67927c4ae65.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/5a47ac0ad12e2a34d52e5f3b1331ad254c285d0756572db198c4f67927c4ae65.jpg)

![5d1c7dd2501c7f25d8f62f99d255c2e4374756c75648da56cea238e2389bf5fb.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/5d1c7dd2501c7f25d8f62f99d255c2e4374756c75648da56cea238e2389bf5fb.jpg)

![6507b0f1d94fd2e3b9e74186487845d6424f371d5245b514e1570ed31aa49b1d.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/6507b0f1d94fd2e3b9e74186487845d6424f371d5245b514e1570ed31aa49b1d.jpg)

![7737759dfe39dd2a520405af164c52b8c3f8ddedafb5eb5db424bc2b22ad3a0f.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/7737759dfe39dd2a520405af164c52b8c3f8ddedafb5eb5db424bc2b22ad3a0f.jpg)

![8698b83191fdbe3e38c448500f1542cf15548ea61a425c92a0a249906fa9e620.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/8698b83191fdbe3e38c448500f1542cf15548ea61a425c92a0a249906fa9e620.jpg)

![9490ae60b887aa7c74fecbb0689f098c638e871971757e4dd0e33007dfa8ce12.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/9490ae60b887aa7c74fecbb0689f098c638e871971757e4dd0e33007dfa8ce12.jpg)

![b1edeaf79f47dc24b3a7f75e68ae5e72f8ed263adedcdfd9ad8941f9812f4010.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/b1edeaf79f47dc24b3a7f75e68ae5e72f8ed263adedcdfd9ad8941f9812f4010.jpg)

![b418f5cbda4bf4d3b928350fdb62346c174096109b95942b546ae90f1d216e42.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/b418f5cbda4bf4d3b928350fdb62346c174096109b95942b546ae90f1d216e42.jpg)

![b8a941fe9535240cb3d0e158d66505e14d5d58839b098122ccb949924b18cfb6.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/b8a941fe9535240cb3d0e158d66505e14d5d58839b098122ccb949924b18cfb6.jpg)

![cb6dc1746f2bdd3cf577285951939ae978317d22d46159ba831596a72c294711.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/cb6dc1746f2bdd3cf577285951939ae978317d22d46159ba831596a72c294711.jpg)

![db3d054c41d7b9d747a8db346303ceff95afd6a2cfee428de25008d042a3abd0.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/db3d054c41d7b9d747a8db346303ceff95afd6a2cfee428de25008d042a3abd0.jpg)

![eb4c6f956e6d57ef19a167106e6a331cabadb3b09a70053f975670db7733ac69.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/eb4c6f956e6d57ef19a167106e6a331cabadb3b09a70053f975670db7733ac69.jpg)

![f9cfb19a56d302e932857409e33c39e7edc8f6b83c8fba2368c14d5c14c11bf4.jpg](../icml_results/259_WaterSIC_%20Information-Theoretically%20%28Near%29%20Optimal%20Linear%20Layer%20Quantization/tables/f9cfb19a56d302e932857409e33c39e7edc8f6b83c8fba2368c14d5c14c11bf4.jpg)

<a id="paper-0255"></a>

## From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models


### Images

![16ba5a64b1d253b7e1fc961a601ad1ae86a3a4f7171e8214323a9a3de971ab8c.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/16ba5a64b1d253b7e1fc961a601ad1ae86a3a4f7171e8214323a9a3de971ab8c.jpg)

![762a0393d3e84dcfa2ecec0dacf474bd1d5f82ae32f6392cd0558681e39fd850.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/762a0393d3e84dcfa2ecec0dacf474bd1d5f82ae32f6392cd0558681e39fd850.jpg)

![7fa8062d7725b4f408b2571d2e01f171944fac64727652560cb89d47e6221e23.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/7fa8062d7725b4f408b2571d2e01f171944fac64727652560cb89d47e6221e23.jpg)

![8a9f82160fe3f7a70eb9d38842f0a9c415cbd149173e22f548c9dcdffffb9985.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/8a9f82160fe3f7a70eb9d38842f0a9c415cbd149173e22f548c9dcdffffb9985.jpg)

![98a32b6fd752d17cf3e63df6789a4bfa5819ea39ca21b33d44ae7010ce41d65e.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/98a32b6fd752d17cf3e63df6789a4bfa5819ea39ca21b33d44ae7010ce41d65e.jpg)

![b9732865b45d2d5fe0375b5fe5b222a6070c8b3135882728a739bd29b3afaa80.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/b9732865b45d2d5fe0375b5fe5b222a6070c8b3135882728a739bd29b3afaa80.jpg)

![d345037b869740bc1bfdfde351b45fb8321192a8168f5374df9342f9900273cf.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/d345037b869740bc1bfdfde351b45fb8321192a8168f5374df9342f9900273cf.jpg)

![d62e4e70a2cc067f402c6a5adfaa540e56f0b96d4fbdb8903a03e3ac0dfbef19.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/d62e4e70a2cc067f402c6a5adfaa540e56f0b96d4fbdb8903a03e3ac0dfbef19.jpg)

![ef67ed9fbc9b36ef68f0e7c45de5e2de18281c3834d41447c5f114f21207584f.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/ef67ed9fbc9b36ef68f0e7c45de5e2de18281c3834d41447c5f114f21207584f.jpg)

![fd3367f42b634a33b8e34dc3e0eea41b092a0bdbdf070364c8334d14291dc865.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/fd3367f42b634a33b8e34dc3e0eea41b092a0bdbdf070364c8334d14291dc865.jpg)

![fe987a190f34090e738160b42098ba657acb2a9dc3f34c60b92b9d2688819ee1.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/images/fe987a190f34090e738160b42098ba657acb2a9dc3f34c60b92b9d2688819ee1.jpg)

### Tables

![0112b22c560d8a187e306b4633274c01b8545d315ca52f585570998732a10844.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/0112b22c560d8a187e306b4633274c01b8545d315ca52f585570998732a10844.jpg)

![06ca47124a72b7758f64282a2fb16a087563f9c8694fbadc388b33e4f532770a.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/06ca47124a72b7758f64282a2fb16a087563f9c8694fbadc388b33e4f532770a.jpg)

![15c576f63e2bf346ddc7608a7081158684f71fde576292c5ab668937c33f31a2.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/15c576f63e2bf346ddc7608a7081158684f71fde576292c5ab668937c33f31a2.jpg)

![234db698a9f9bccfbbee1ad9dfdf7e86fa0672d9931b58904b43cf7c2eab291b.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/234db698a9f9bccfbbee1ad9dfdf7e86fa0672d9931b58904b43cf7c2eab291b.jpg)

![23dbd46f42b618e781f5b979601f5b8781e87b58395fc67750d53d4681ef9841.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/23dbd46f42b618e781f5b979601f5b8781e87b58395fc67750d53d4681ef9841.jpg)

![424d54afd78cf0bf38ae7dd6d400933b20902e7d457b5ded79eb9216fb11baf5.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/424d54afd78cf0bf38ae7dd6d400933b20902e7d457b5ded79eb9216fb11baf5.jpg)

![46a70fe0cf3709e9ac5742c9e0bf39de066e6a2d80f3037fe42efec37973e9cb.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/46a70fe0cf3709e9ac5742c9e0bf39de066e6a2d80f3037fe42efec37973e9cb.jpg)

![53656f6ff0541d8436867b39696a9197affc923e346f0abe45d1c4aa021b24f0.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/53656f6ff0541d8436867b39696a9197affc923e346f0abe45d1c4aa021b24f0.jpg)

![57276329152e9be750ffe58acef57a2b8a3fcf28ee13be7fe65ff7f02dfc07fb.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/57276329152e9be750ffe58acef57a2b8a3fcf28ee13be7fe65ff7f02dfc07fb.jpg)

![5852ba280c0c0776db86664647b791db61859ff7c98a5224fd967361cdc326d1.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/5852ba280c0c0776db86664647b791db61859ff7c98a5224fd967361cdc326d1.jpg)

![67a24eea023e0eea0a07dc0bc717d968adfa255d6f543b16f02a647c8864879b.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/67a24eea023e0eea0a07dc0bc717d968adfa255d6f543b16f02a647c8864879b.jpg)

![6aa1695a97ab983a732c4c71d47cfbed5f896ae0e8f678a7a26ee06d16e958dd.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/6aa1695a97ab983a732c4c71d47cfbed5f896ae0e8f678a7a26ee06d16e958dd.jpg)

![7545d829120f5318f668450587ba447e18c4b0dedb236a7b54c90ba379e5d258.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/7545d829120f5318f668450587ba447e18c4b0dedb236a7b54c90ba379e5d258.jpg)

![8d7ce12cfd8f2b3244027e9125a8f2fe3d731900044d69c78b2f7577b88c0602.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/8d7ce12cfd8f2b3244027e9125a8f2fe3d731900044d69c78b2f7577b88c0602.jpg)

![b027c3648381835e659bf45a5a4f3e2a5869f810d62b5c34f521c73c2d0e3632.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/b027c3648381835e659bf45a5a4f3e2a5869f810d62b5c34f521c73c2d0e3632.jpg)

![b3314cc8d6bc6dd7a59ae288ae13541d5f0380792e87167952792a028e53660b.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/b3314cc8d6bc6dd7a59ae288ae13541d5f0380792e87167952792a028e53660b.jpg)

![c145bbd487b9137fc6786fcaf0b71c537e656042ebda6103ec35af73e730d3a0.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/c145bbd487b9137fc6786fcaf0b71c537e656042ebda6103ec35af73e730d3a0.jpg)

![c35986681b657c60ed1afe8c26cb76749ead1b29204780f351bf77fa39fa8a75.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/c35986681b657c60ed1afe8c26cb76749ead1b29204780f351bf77fa39fa8a75.jpg)

![d46b3b001bde99a6f46ddcdbea6a5d86a6f3f589177ee06a50dfd119d09d6a7c.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/d46b3b001bde99a6f46ddcdbea6a5d86a6f3f589177ee06a50dfd119d09d6a7c.jpg)

![d6f2f01d4764935f8c2e9c57ae60fc4129bb1185be5e96b8ff5396cf7991cc3e.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/d6f2f01d4764935f8c2e9c57ae60fc4129bb1185be5e96b8ff5396cf7991cc3e.jpg)

![dcc4ea6c7fcef912ad1d068a5184b9046515919d81bd9cd6726023849d304840.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/dcc4ea6c7fcef912ad1d068a5184b9046515919d81bd9cd6726023849d304840.jpg)

![e6ff6978a58e696220380ce07692b3200aa3a4e6e0163c1dd853423f2e4edb4e.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/e6ff6978a58e696220380ce07692b3200aa3a4e6e0163c1dd853423f2e4edb4e.jpg)

![f8b565a6d07537869c70ec550cd5851cc143cc0c350964fd1bf508a8b1ad7220.jpg](../icml_results/260_From%20Pixels%20to%20Tokens_%20A%20Systematic%20Study%20of%20Latent%20Action%20Supervision%20for%20Vision-Language-Action%20Mo/tables/f8b565a6d07537869c70ec550cd5851cc143cc0c350964fd1bf508a8b1ad7220.jpg)

<a id="paper-0256"></a>

## MSP: Probabilistically Consistent Multi-Scale Action Generation


### Images

![3aa27ce85cb6441dc588768959e4ce4d5b375368269dd8f458f4b72a75f02da9.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/3aa27ce85cb6441dc588768959e4ce4d5b375368269dd8f458f4b72a75f02da9.jpg)

![3c13eb1e72c2eac3665eb150c4f4d63c87750b60c7439b1c1240418319289790.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/3c13eb1e72c2eac3665eb150c4f4d63c87750b60c7439b1c1240418319289790.jpg)

![41d15d02fb760c300315d0643552e7feaea77136a28c35608ae69e2e88bc9d31.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/41d15d02fb760c300315d0643552e7feaea77136a28c35608ae69e2e88bc9d31.jpg)

![4c42b4ea64748262ceda44576c72f70f7ad55a193f008ce43fa710c273e8710c.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/4c42b4ea64748262ceda44576c72f70f7ad55a193f008ce43fa710c273e8710c.jpg)

![8c02e3d4b9022d83ac5fd81e1eac9722aa7f90bbe5ab46e756752793fc95d798.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/8c02e3d4b9022d83ac5fd81e1eac9722aa7f90bbe5ab46e756752793fc95d798.jpg)

![a8fcd039a14e665769c1210fcf39c0165e4ecb077b30faff9bb4f1ec282b637c.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/a8fcd039a14e665769c1210fcf39c0165e4ecb077b30faff9bb4f1ec282b637c.jpg)

![aa9023cdcd13992a46ea6838ec02ec1e60b7baff3f8aea4271a21937d39d0b50.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/aa9023cdcd13992a46ea6838ec02ec1e60b7baff3f8aea4271a21937d39d0b50.jpg)

![b037b2f5c2a38c3f9df743f305dad6da86fa6386c194f2fcfdd9a2b4d2edea43.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/b037b2f5c2a38c3f9df743f305dad6da86fa6386c194f2fcfdd9a2b4d2edea43.jpg)

![bb631703642d61c10a202a047665db19a8d49fe7d85fbdac7807c1e1eb15f54c.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/bb631703642d61c10a202a047665db19a8d49fe7d85fbdac7807c1e1eb15f54c.jpg)

![dfeeb9de266a91acf172536f351ba30a2c873013f1d9514404cd0e3242833d5b.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/dfeeb9de266a91acf172536f351ba30a2c873013f1d9514404cd0e3242833d5b.jpg)

![e2be6e3d9ec71fd0afce783d1234f2ba9062c081f30f0c18e3f558dae820e09f.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/images/e2be6e3d9ec71fd0afce783d1234f2ba9062c081f30f0c18e3f558dae820e09f.jpg)

### Tables

![21b216cef1f5fc62eafe4452f40167fade2887fb7423ae925dba17517390e658.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/21b216cef1f5fc62eafe4452f40167fade2887fb7423ae925dba17517390e658.jpg)

![4297e5ab7597c40e605461631f34e36098cbc922434c0ea89699359c7bffbeb2.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/4297e5ab7597c40e605461631f34e36098cbc922434c0ea89699359c7bffbeb2.jpg)

![544c131d4df87e0c76448bd8205a69fbbdc4e0251f3ea50803b2cd5136cc2093.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/544c131d4df87e0c76448bd8205a69fbbdc4e0251f3ea50803b2cd5136cc2093.jpg)

![5893d537339dab5e07a7289c3e8b7d940decee694721fa365f03ee1c80116503.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/5893d537339dab5e07a7289c3e8b7d940decee694721fa365f03ee1c80116503.jpg)

![606a35d32908f1a1a70465aa41103883700ed0fb0ee8a447a95e12241fc8faf7.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/606a35d32908f1a1a70465aa41103883700ed0fb0ee8a447a95e12241fc8faf7.jpg)

![7381917e56e134ef740c1e27250e1866160104c3762111a092328d8b2d901c4c.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/7381917e56e134ef740c1e27250e1866160104c3762111a092328d8b2d901c4c.jpg)

![791279d668eeb168f44959ba6a6b1140538bc7b12d75fa6671e0cbda6d6cfeb2.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/791279d668eeb168f44959ba6a6b1140538bc7b12d75fa6671e0cbda6d6cfeb2.jpg)

![aca6951c2278e63731e82ce99d7c727dd7737b2cd3dd2b65d5ee177b6ba0a9e1.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/aca6951c2278e63731e82ce99d7c727dd7737b2cd3dd2b65d5ee177b6ba0a9e1.jpg)

![ccecec2b368b6e5e5f89c8dd55bdd7d56b6e776d5bd2404a4443dbb6065b6f53.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/ccecec2b368b6e5e5f89c8dd55bdd7d56b6e776d5bd2404a4443dbb6065b6f53.jpg)

![de5ea15aa89affcf0020ab1982a6f545eecf0fd2d03aecc887e9f8e5c0630135.jpg](../icml_results/261_MSP_%20Probabilistically%20Consistent%20Multi-Scale%20Action%20Generation/tables/de5ea15aa89affcf0020ab1982a6f545eecf0fd2d03aecc887e9f8e5c0630135.jpg)

<a id="paper-0257"></a>

## GR-LoRA: Gradient-Recycling Low-Rank Adaptation for Class-Incremental Learning


### Images

![1d0b4957e164a27665d8019eec9c6fc7a1f217e7cfe0867f45a5f84b3b196a00.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/1d0b4957e164a27665d8019eec9c6fc7a1f217e7cfe0867f45a5f84b3b196a00.jpg)

![2dd08b22b65fc6ef3ca3c0331a53cfe3ac91c90c637f9fe1b4ed60a735732290.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/2dd08b22b65fc6ef3ca3c0331a53cfe3ac91c90c637f9fe1b4ed60a735732290.jpg)

![56573da64942e0b64d9847d67f300bb04c64b73f0029956c524a9073b3ec7913.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/56573da64942e0b64d9847d67f300bb04c64b73f0029956c524a9073b3ec7913.jpg)

![6371fb15e57bf5def3743b12b1b8a2d2040c1fcfdbfefb57d4fed5849fe8fbdd.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/6371fb15e57bf5def3743b12b1b8a2d2040c1fcfdbfefb57d4fed5849fe8fbdd.jpg)

![986138dbadcb0ae7eddbd69e10430a18b3805fda902a26c405352305b5bdd77a.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/986138dbadcb0ae7eddbd69e10430a18b3805fda902a26c405352305b5bdd77a.jpg)

![ee7d58aeb148d9cd899a7940992bac95d9afda28070256570920228e270ee8e3.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/images/ee7d58aeb148d9cd899a7940992bac95d9afda28070256570920228e270ee8e3.jpg)

### Tables

![3bcbfc8bf69d4f3139f1507af40ec5302627e65fba47a5690984f6357fc26ebf.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/3bcbfc8bf69d4f3139f1507af40ec5302627e65fba47a5690984f6357fc26ebf.jpg)

![4658d0b8a321c050901bb01019ab5f34c48defcbc115ed6d74c438600eb76b6d.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/4658d0b8a321c050901bb01019ab5f34c48defcbc115ed6d74c438600eb76b6d.jpg)

![47264b8aba3b8939f64efb768ffd6532ce8151ef66901530b2b25b52e5da2687.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/47264b8aba3b8939f64efb768ffd6532ce8151ef66901530b2b25b52e5da2687.jpg)

![68e2baaf31c47632711d077a3e956de33ef9f271804acad0e2e7c4ecf0fc275a.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/68e2baaf31c47632711d077a3e956de33ef9f271804acad0e2e7c4ecf0fc275a.jpg)

![714b5d333cc0d7aa55f84cc3b5a250cb494d67675a8ca8001785c44a3d3c23ec.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/714b5d333cc0d7aa55f84cc3b5a250cb494d67675a8ca8001785c44a3d3c23ec.jpg)

![a4118210795b4e52cc4c424ab9cd1acf34eb9b768a08ecf9e7859eb0b6a9e2e8.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/a4118210795b4e52cc4c424ab9cd1acf34eb9b768a08ecf9e7859eb0b6a9e2e8.jpg)

![aa055b8c87aa589f4c40318c7abb14c7ef72b1a87e32ef916f3f88f33ed48eb2.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/aa055b8c87aa589f4c40318c7abb14c7ef72b1a87e32ef916f3f88f33ed48eb2.jpg)

![b1700009e3714dbe15d2cd316e5933651c3fe3bf8df38d33da2deaa7bab4ba12.jpg](../icml_results/262_GR-LoRA_%20Gradient-Recycling%20Low-Rank%20Adaptation%20for%20Class-Incremental%20Learning/tables/b1700009e3714dbe15d2cd316e5933651c3fe3bf8df38d33da2deaa7bab4ba12.jpg)

<a id="paper-0258"></a>

## FlatLand: Personalized Graph Federated Learning via Tailored Lorentz Space


### Images

![1107d6995583dadccacc9382f62af9c5267785238b483d4741745da81bde38d7.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/1107d6995583dadccacc9382f62af9c5267785238b483d4741745da81bde38d7.jpg)

![39f518fd2138170d488fc304d138e8fa8ab388c88562ed1de2000c19c8212579.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/39f518fd2138170d488fc304d138e8fa8ab388c88562ed1de2000c19c8212579.jpg)

![5b4d6da737c3e6f241c012b254b07c1e3b96132a0a7686b68174d7896d2d508b.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/5b4d6da737c3e6f241c012b254b07c1e3b96132a0a7686b68174d7896d2d508b.jpg)

![918be5307277d091c4ba00b36872cd8ec26aae47573ef107afc6cbf296774cc5.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/918be5307277d091c4ba00b36872cd8ec26aae47573ef107afc6cbf296774cc5.jpg)

![9cf9c2f25595e27a3989ef6372977855b9b6dc462a5eac5d1de7415ac5da4eda.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/9cf9c2f25595e27a3989ef6372977855b9b6dc462a5eac5d1de7415ac5da4eda.jpg)

![bddda35360dc53a12e3c6a7152a6b18fc7236f5b9984dcd8dc8e6e62dc0ab0a9.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/bddda35360dc53a12e3c6a7152a6b18fc7236f5b9984dcd8dc8e6e62dc0ab0a9.jpg)

![c742a245653d60645294d3ca782b878aaf8e75084e79536e507d786bd34cc0bf.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/c742a245653d60645294d3ca782b878aaf8e75084e79536e507d786bd34cc0bf.jpg)

![c9cda1b5265ae2a13ad28fdf925e3ab0191cd70716f1bceda69ad0535bf7604a.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/c9cda1b5265ae2a13ad28fdf925e3ab0191cd70716f1bceda69ad0535bf7604a.jpg)

![d8e670639c475242f5ce510d71237c657b70302fa9b37bfd51f73877cefacb1f.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/d8e670639c475242f5ce510d71237c657b70302fa9b37bfd51f73877cefacb1f.jpg)

![d957757f8b683a8cbd7eb4001765c6183c98c6676c96f56ef6624f2e3483e116.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/images/d957757f8b683a8cbd7eb4001765c6183c98c6676c96f56ef6624f2e3483e116.jpg)

### Tables

![0f480684269f3852a86168a3abe013b58463b2bc516698be6f400eefcf16a1e9.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/0f480684269f3852a86168a3abe013b58463b2bc516698be6f400eefcf16a1e9.jpg)

![105b347184733c9adc597550adccbf7f6c43f122fca0b9619f419cde6b2a54ea.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/105b347184733c9adc597550adccbf7f6c43f122fca0b9619f419cde6b2a54ea.jpg)

![16803f5d576ed659bb6b9a81fb1cdb5d0b11da80ee860c6f8d2cbba36d095cc0.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/16803f5d576ed659bb6b9a81fb1cdb5d0b11da80ee860c6f8d2cbba36d095cc0.jpg)

![4d7011cc8e5a6ba11b3303cbc22eab9d028f84fccdeef9dc5eb0377c3fdb29e5.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/4d7011cc8e5a6ba11b3303cbc22eab9d028f84fccdeef9dc5eb0377c3fdb29e5.jpg)

![6db467d30b7b2eb674c3432ba7fbe6e622586d0c9700c0539b2747a717ae6db2.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/6db467d30b7b2eb674c3432ba7fbe6e622586d0c9700c0539b2747a717ae6db2.jpg)

![783415207961e04612888ac1aa03be4aa52e011d7a9fd01911d639a63666d4a0.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/783415207961e04612888ac1aa03be4aa52e011d7a9fd01911d639a63666d4a0.jpg)

![a7a2195871ad847081b50ab0aed95898e31e6f402cd941346a4913bd1d8df37a.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/a7a2195871ad847081b50ab0aed95898e31e6f402cd941346a4913bd1d8df37a.jpg)

![cce812efbef8605a137a88e23f83f13d371d52dbb935cdb073712c93fc50a24f.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/cce812efbef8605a137a88e23f83f13d371d52dbb935cdb073712c93fc50a24f.jpg)

![ffd6c8771e4a50061624d7accf05be545094b19d1c5beb7e82d0f1472acd4645.jpg](../icml_results/263_FlatLand_%20Personalized%20Graph%20Federated%20Learning%20via%20Tailored%20Lorentz%20Space/tables/ffd6c8771e4a50061624d7accf05be545094b19d1c5beb7e82d0f1472acd4645.jpg)

<a id="paper-0259"></a>

## World-Model Inspired Emotion-aware Token Refinement for Training-Free Multimodal Emotion Recognition


### Images

![22b22669d8f175f4d9fa9be87cd53a474988d5124540808391e66f26a36ca1c6.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/22b22669d8f175f4d9fa9be87cd53a474988d5124540808391e66f26a36ca1c6.jpg)

![47421cabd2e31df3bfab5d30a7f7e9a81aad95495bdebc333be430d9e3543d76.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/47421cabd2e31df3bfab5d30a7f7e9a81aad95495bdebc333be430d9e3543d76.jpg)

![4a8805a6170d43ee3727e11fe43487ba3e9452692c712a2d309490e88847abaf.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/4a8805a6170d43ee3727e11fe43487ba3e9452692c712a2d309490e88847abaf.jpg)

![704a1745f4810dfc870ca806233fdf742254e93c34ac00f084539f03c79df09d.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/704a1745f4810dfc870ca806233fdf742254e93c34ac00f084539f03c79df09d.jpg)

![a1cdfb4d57c77a8946afc57e6a3f6871251ff172b0751dad7952b01359cbb60f.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/a1cdfb4d57c77a8946afc57e6a3f6871251ff172b0751dad7952b01359cbb60f.jpg)

![a8f440ffdb4eb417e1fdb836521b08c8609c6b0fac609c94bddae48b505db122.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/a8f440ffdb4eb417e1fdb836521b08c8609c6b0fac609c94bddae48b505db122.jpg)

![e6981c9788ebe507f4ffefa9207f80a53d8ff62ded24ce72330386f4abd48d5d.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/images/e6981c9788ebe507f4ffefa9207f80a53d8ff62ded24ce72330386f4abd48d5d.jpg)

### Tables

![0eac513ad3fe41e2729d7c2e5efb46d38c583b8271d21daad4f703612d1ffadc.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/0eac513ad3fe41e2729d7c2e5efb46d38c583b8271d21daad4f703612d1ffadc.jpg)

![4edae13911e511cb69c80ed0a22bdb0f681e83152e56959afc1bdd7579df0995.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/4edae13911e511cb69c80ed0a22bdb0f681e83152e56959afc1bdd7579df0995.jpg)

![9a2a2807e2818165cdbc106d17470942d22a7673f5bbf3af2d3061774b951cb1.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/9a2a2807e2818165cdbc106d17470942d22a7673f5bbf3af2d3061774b951cb1.jpg)

![9b9a2024cfd8f3d690446f44046568f59ee98bc0a10948080a6839d5fdee958e.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/9b9a2024cfd8f3d690446f44046568f59ee98bc0a10948080a6839d5fdee958e.jpg)

![b37836329fdb0fd020c9e90629f2c9f9559d34eb1db51fc98e5f6e77ef6859e9.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/b37836329fdb0fd020c9e90629f2c9f9559d34eb1db51fc98e5f6e77ef6859e9.jpg)

![bbc21626badd0c5593877cba63c3e4bf85ba4bda3cf2545bf01c011143e19f01.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/bbc21626badd0c5593877cba63c3e4bf85ba4bda3cf2545bf01c011143e19f01.jpg)

![eee0d6421fca22f616db044a1e1506d593c63020e2dc4c6eeaf44e7901a51656.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/eee0d6421fca22f616db044a1e1506d593c63020e2dc4c6eeaf44e7901a51656.jpg)

![ef2467a9826f497e82ee7701d18bc2ea86e8467e9f1556a32bdfc18bb813a902.jpg](../icml_results/264_World-Model%20Inspired%20Emotion-aware%20Token%20Refinement%20for%20Training-Free%20Multimodal%20Emotion%20Recognition/tables/ef2467a9826f497e82ee7701d18bc2ea86e8467e9f1556a32bdfc18bb813a902.jpg)

<a id="paper-0260"></a>

## Solving Time-Dependent Differential Equations with Physical Dynamical Systems


### Images

![380177ab57743bc1a664006ea9900869e0797e31723dd425b5dbc8d9e994a89d.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/380177ab57743bc1a664006ea9900869e0797e31723dd425b5dbc8d9e994a89d.jpg)

![8820db8fc86144131d3d913f69a085d468845357db95365d6656c0314f4bc332.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/8820db8fc86144131d3d913f69a085d468845357db95365d6656c0314f4bc332.jpg)

![ae62ad875181fcc5db67e0ba59fadefefb5482d2874726fcc3f130c07c84957a.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/ae62ad875181fcc5db67e0ba59fadefefb5482d2874726fcc3f130c07c84957a.jpg)

![cab90c4a9f8280c1eb0835abcbb7ec7fe50e02455de2f69482744e9c985086fa.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/cab90c4a9f8280c1eb0835abcbb7ec7fe50e02455de2f69482744e9c985086fa.jpg)

![dfa35bfdc954995e19a13cd097f2f9a6fe1dbc21d3f9606a5bda6a8d049c2e0d.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/dfa35bfdc954995e19a13cd097f2f9a6fe1dbc21d3f9606a5bda6a8d049c2e0d.jpg)

![eb6dc05b53761af77729dbb017cc51b756463ade8f9b7caf6b364b49cb8b901a.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/eb6dc05b53761af77729dbb017cc51b756463ade8f9b7caf6b364b49cb8b901a.jpg)

![f6d285042abc124d73709946ce6e2cbc02f75ea06f58e5418cf3961ce1ce3d4e.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/images/f6d285042abc124d73709946ce6e2cbc02f75ea06f58e5418cf3961ce1ce3d4e.jpg)

### Tables

![ce8440afd2b626ba1e6a53c41350a052a7f1a7a8fb66217f2fef6d292fd688f4.jpg](../icml_results/265_Solving%20Time-Dependent%20Differential%20Equations%20with%20Physical%20Dynamical%20Systems/tables/ce8440afd2b626ba1e6a53c41350a052a7f1a7a8fb66217f2fef6d292fd688f4.jpg)

<a id="paper-0261"></a>

## Online Conformal Prediction via Universal Portfolio Algorithms


### Images

![0050c294c2767996eab2f1b39df53a62dfdbcce68eecf483fc9d331ec61a1a8e.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/0050c294c2767996eab2f1b39df53a62dfdbcce68eecf483fc9d331ec61a1a8e.jpg)

![03756ce3249701ebe19f03bda9a06eb4a9115c3510fcf56389f7ea3f49ee85ec.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/03756ce3249701ebe19f03bda9a06eb4a9115c3510fcf56389f7ea3f49ee85ec.jpg)

![04351b1ac2b93f8a584edc5a09408909fd4865558f2079f5965541b8ad524052.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/04351b1ac2b93f8a584edc5a09408909fd4865558f2079f5965541b8ad524052.jpg)

![04660246290e43fd30466eed5ba9292a9ef3713b69e63dfe5fb33361e4c9e906.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/04660246290e43fd30466eed5ba9292a9ef3713b69e63dfe5fb33361e4c9e906.jpg)

![06c30e1272c472dcf754644392867ba0c5f48dda51f1b30737af3b758befdfc0.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/06c30e1272c472dcf754644392867ba0c5f48dda51f1b30737af3b758befdfc0.jpg)

![07c2eb44231ac79b20f8bebf90529301f6ede8c2b304be0e5f1c07c3cd4728a4.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/07c2eb44231ac79b20f8bebf90529301f6ede8c2b304be0e5f1c07c3cd4728a4.jpg)

![115c05022a38b05721ee5c472aeb76e27a933f188921533cc12bb0433c52aa87.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/115c05022a38b05721ee5c472aeb76e27a933f188921533cc12bb0433c52aa87.jpg)

![1893fe5b33a7e1acc29d1d9b5972f5932553c2e71d6564f08fb261a330ebbbb8.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/1893fe5b33a7e1acc29d1d9b5972f5932553c2e71d6564f08fb261a330ebbbb8.jpg)

![1b9a9e5d3daf3ace3b7450f0ab2d263cc4cff803d86a7131f245773c60e90eff.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/1b9a9e5d3daf3ace3b7450f0ab2d263cc4cff803d86a7131f245773c60e90eff.jpg)

![1c70e73e0838142ebafea23eeb7d72d84008a63d3bbcd0e040a28843644eb651.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/1c70e73e0838142ebafea23eeb7d72d84008a63d3bbcd0e040a28843644eb651.jpg)

![1d5b5551c76d9262592738f15f01a0b891f803a0292117b32de1a9f852466be5.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/1d5b5551c76d9262592738f15f01a0b891f803a0292117b32de1a9f852466be5.jpg)

![1d98b57df6d68eeb2adfd45b647a3a161389f243dab8618ceec0f251ec181986.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/1d98b57df6d68eeb2adfd45b647a3a161389f243dab8618ceec0f251ec181986.jpg)

![29394a5706d757abd6af7cdc09288af7eef28e769b7602618020230f10f0199a.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/29394a5706d757abd6af7cdc09288af7eef28e769b7602618020230f10f0199a.jpg)

![2a76f4347cf27e4406d1c840afc0fe4315bdc4e783a74fcff0f131c15ef95a4f.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/2a76f4347cf27e4406d1c840afc0fe4315bdc4e783a74fcff0f131c15ef95a4f.jpg)

![34a698172b5d836a9ba45ee70e2d1d6d5cd8df40981ebac480defc7e6aa21552.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/34a698172b5d836a9ba45ee70e2d1d6d5cd8df40981ebac480defc7e6aa21552.jpg)

![3808880de7033128ffa0b19e17549addeab864cae8be32cd2ae5bfa12219d029.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/3808880de7033128ffa0b19e17549addeab864cae8be32cd2ae5bfa12219d029.jpg)

![3d1bd3314c4975927e04fb95596de58a8b31382a2f6fe4813e6030fe24564360.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/3d1bd3314c4975927e04fb95596de58a8b31382a2f6fe4813e6030fe24564360.jpg)

![408ff9b949dd8b3dafd5fcdb2543c50269decd9f4af567d0745af5e3178c6c4a.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/408ff9b949dd8b3dafd5fcdb2543c50269decd9f4af567d0745af5e3178c6c4a.jpg)

![458a93b783b2c9f0113bc37d0e8162d10486cdd3dce9e71b24b53f0dc54e4269.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/458a93b783b2c9f0113bc37d0e8162d10486cdd3dce9e71b24b53f0dc54e4269.jpg)

![47e0fc0872c0dba2aa3596d719e9d785e6203687e3d77b80ec7fcc97d7dedacc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/47e0fc0872c0dba2aa3596d719e9d785e6203687e3d77b80ec7fcc97d7dedacc.jpg)

![55503d21d774d6e01bfb705e18ffc6dcbc76fa18c37dc3c6d83d9cd0313ab838.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/55503d21d774d6e01bfb705e18ffc6dcbc76fa18c37dc3c6d83d9cd0313ab838.jpg)

![5bae69c3303eb085937a2012495bb00ecd3ac7a357c2c61b942e6c5d46e7e202.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/5bae69c3303eb085937a2012495bb00ecd3ac7a357c2c61b942e6c5d46e7e202.jpg)

![5c22fc5fbb48e0d37a28c19d61110f9f4c546e5b32cd63858e70c6b87477f3ee.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/5c22fc5fbb48e0d37a28c19d61110f9f4c546e5b32cd63858e70c6b87477f3ee.jpg)

![623e3cf199b5338f58f0927babe598f65ad14ab299f91c932eb376529b731add.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/623e3cf199b5338f58f0927babe598f65ad14ab299f91c932eb376529b731add.jpg)

![63c12b4b025125386f979bb2db0cd8072a74c26f3d77dc9a6e14a018b8b03751.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/63c12b4b025125386f979bb2db0cd8072a74c26f3d77dc9a6e14a018b8b03751.jpg)

![644379c7147017774dd69ab5c90d187f0eeb9089738ed60cebbfbaaf56c57a42.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/644379c7147017774dd69ab5c90d187f0eeb9089738ed60cebbfbaaf56c57a42.jpg)

![650b400fd82e7709c70fce93d442fdf2db179ea02275262b52f70aa5dd31415f.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/650b400fd82e7709c70fce93d442fdf2db179ea02275262b52f70aa5dd31415f.jpg)

![65dce6065c7676bae35e6890f3880b78b3d6b6ac23da6fc10575fb4fc7d5310c.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/65dce6065c7676bae35e6890f3880b78b3d6b6ac23da6fc10575fb4fc7d5310c.jpg)

![67a50e6e72db35a07d298faa283773d18d1a09b16aaa5e8f6f91be4b06176c12.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/67a50e6e72db35a07d298faa283773d18d1a09b16aaa5e8f6f91be4b06176c12.jpg)

![6a5a21b9287e1912e90d6c760c007082da0eb7f784ee492db128d5e3b5f6f835.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/6a5a21b9287e1912e90d6c760c007082da0eb7f784ee492db128d5e3b5f6f835.jpg)

![6bd589830d314adb433659da4ae7656e39fe3e9977490f5d8889363082caeaa7.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/6bd589830d314adb433659da4ae7656e39fe3e9977490f5d8889363082caeaa7.jpg)

![6c22e5e1414c5afc211ca7dd5334dee4b973204c685f8de9ae2b251deeb043f3.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/6c22e5e1414c5afc211ca7dd5334dee4b973204c685f8de9ae2b251deeb043f3.jpg)

![6ddf11c012efa45166574f1045def26062ab1a47534122efdd889123f4b4b869.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/6ddf11c012efa45166574f1045def26062ab1a47534122efdd889123f4b4b869.jpg)

![73260b589e0d6fd3377371528c678aee4446b3bbb227efd9c6f90e5689340235.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/73260b589e0d6fd3377371528c678aee4446b3bbb227efd9c6f90e5689340235.jpg)

![75d8b830b4c57bedba3f9cbb200c59f9d04ae4450c75b1c8df9528c822b3926c.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/75d8b830b4c57bedba3f9cbb200c59f9d04ae4450c75b1c8df9528c822b3926c.jpg)

![78fb5811d39185c25e2125a3b824a9ae3315dab3123c6ccb0a5296d479052c50.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/78fb5811d39185c25e2125a3b824a9ae3315dab3123c6ccb0a5296d479052c50.jpg)

![796b1f694a2b6f12617720b28812983f831cc4cc0b77d4a310919dea7b42cc8c.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/796b1f694a2b6f12617720b28812983f831cc4cc0b77d4a310919dea7b42cc8c.jpg)

![7b6a2432cadd291a101a3e2d87baca3d59d985eb788ca69d6d1fff396c2c8d69.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/7b6a2432cadd291a101a3e2d87baca3d59d985eb788ca69d6d1fff396c2c8d69.jpg)

![7e4a910b9cec8d44b49165e8c959c33cbb987616bd532e00be89567aadf642b1.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/7e4a910b9cec8d44b49165e8c959c33cbb987616bd532e00be89567aadf642b1.jpg)

![80481c03b6c064968c073133f368abb1e43ae55469be006e3b8767394b7da572.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/80481c03b6c064968c073133f368abb1e43ae55469be006e3b8767394b7da572.jpg)

![8463e14cbecd9eaec7310b5d3211adeb5b5944cb2ddaab8715c3196d94ad57a7.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/8463e14cbecd9eaec7310b5d3211adeb5b5944cb2ddaab8715c3196d94ad57a7.jpg)

![871658bb125ce3581d7e42fbfc734e7c2e824998c444298c1cf7551da6430bbc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/871658bb125ce3581d7e42fbfc734e7c2e824998c444298c1cf7551da6430bbc.jpg)

![8867d776de8add25eec1036c8f914a34e5d3bbfd37541f65a911f13388520511.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/8867d776de8add25eec1036c8f914a34e5d3bbfd37541f65a911f13388520511.jpg)

![8ea8ee8e2de228573e0e4072ae1b5353fce34e781413c074e408312c792dc54a.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/8ea8ee8e2de228573e0e4072ae1b5353fce34e781413c074e408312c792dc54a.jpg)

![9117b549dbda1780f12b1a5682f628286132ba2b57c0a7ab9e9d8bc47144d0f7.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/9117b549dbda1780f12b1a5682f628286132ba2b57c0a7ab9e9d8bc47144d0f7.jpg)

![98262beef3709172b2ca14cd6d286e8c4d86d5bcc809752bbb4b0d8a087d5ac7.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/98262beef3709172b2ca14cd6d286e8c4d86d5bcc809752bbb4b0d8a087d5ac7.jpg)

![989869d209f789e2fb84e96eeb63d9b62a163b1c2cb4362e1aa1b9a957d4571e.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/989869d209f789e2fb84e96eeb63d9b62a163b1c2cb4362e1aa1b9a957d4571e.jpg)

![9edd5beea2129fe48c77997964f2379638d9a34faa8866d0db8cb7809bdba9e5.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/9edd5beea2129fe48c77997964f2379638d9a34faa8866d0db8cb7809bdba9e5.jpg)

![a333ba9280a2f7dea33f40c34df4f001c9234b33379cdac5f5dfdcb46ccd366d.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/a333ba9280a2f7dea33f40c34df4f001c9234b33379cdac5f5dfdcb46ccd366d.jpg)

![a88cec51ce6635ae8ec98b1cb1a4acc0e2ff3a695622a8f6502bbb130bc04f7f.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/a88cec51ce6635ae8ec98b1cb1a4acc0e2ff3a695622a8f6502bbb130bc04f7f.jpg)

![ad8b0de90b45738d6460ea707882e6b907f0ee9dd6271564c77b355feed07d9e.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ad8b0de90b45738d6460ea707882e6b907f0ee9dd6271564c77b355feed07d9e.jpg)

![afbd189f659da48f33787a8b209b6d75267dfc3d202d3e9272404669d3b0720d.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/afbd189f659da48f33787a8b209b6d75267dfc3d202d3e9272404669d3b0720d.jpg)

![b3f2641f96d4805c1936bbe979bcb366be230bb841de1309aa5db9e5d170dc99.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b3f2641f96d4805c1936bbe979bcb366be230bb841de1309aa5db9e5d170dc99.jpg)

![b4f5a76e5927e9aac1a2965bbc43d20138f3e78c25a3cc583679987f7081e057.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b4f5a76e5927e9aac1a2965bbc43d20138f3e78c25a3cc583679987f7081e057.jpg)

![b6dc4b74a27c8143762b4bdd83c406c7f7991364d889f555de7ff400d53be6bb.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b6dc4b74a27c8143762b4bdd83c406c7f7991364d889f555de7ff400d53be6bb.jpg)

![b727db8999a6116fe52583929f7db0474c31f0e049eaaa1618c4f5c8a07a92bb.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b727db8999a6116fe52583929f7db0474c31f0e049eaaa1618c4f5c8a07a92bb.jpg)

![b82c311b9210157e22b7d44cbddd4e9e83ebc47b6c1b87afe19876a53234cb0b.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b82c311b9210157e22b7d44cbddd4e9e83ebc47b6c1b87afe19876a53234cb0b.jpg)

![b9177158b1a29ee43e8d72986cdbf0f0b057d94331c7870f246c5ae5363eb23f.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/b9177158b1a29ee43e8d72986cdbf0f0b057d94331c7870f246c5ae5363eb23f.jpg)

![be572342ae912b43eceb5d7308b5f96b92aea3b7e914914b3e60cd58259918fc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/be572342ae912b43eceb5d7308b5f96b92aea3b7e914914b3e60cd58259918fc.jpg)

![c12055f408a6fb2c86f3b67e7b6c6182acedf58457cdb2b05ad540a4031b14a4.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/c12055f408a6fb2c86f3b67e7b6c6182acedf58457cdb2b05ad540a4031b14a4.jpg)

![c3a25c78742732c39091d84503de4d49439e903d58672a9b6d5952ca2f20ae94.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/c3a25c78742732c39091d84503de4d49439e903d58672a9b6d5952ca2f20ae94.jpg)

![c448fb11a2f94323a4a14c1f22d5c6a9ff23e9d26f84f48769c9fffca1f2f96e.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/c448fb11a2f94323a4a14c1f22d5c6a9ff23e9d26f84f48769c9fffca1f2f96e.jpg)

![ca16c514447127a022bd6e1ee335e69313c41635ccbbb8d09d7953c536c5582c.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ca16c514447127a022bd6e1ee335e69313c41635ccbbb8d09d7953c536c5582c.jpg)

![cb95dda467ff788e89c465bef5d75c4bdf04af5b11c3ae35b79ada5bd78f8051.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/cb95dda467ff788e89c465bef5d75c4bdf04af5b11c3ae35b79ada5bd78f8051.jpg)

![ce07108d1830b65f1383f4ffdc8489ef9c5e62ea6452dbdb19605b9768524f32.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ce07108d1830b65f1383f4ffdc8489ef9c5e62ea6452dbdb19605b9768524f32.jpg)

![ce4763cf1986785157e5ca0c7755cff779caa26c51294dea4a0aae6b889ef4e9.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ce4763cf1986785157e5ca0c7755cff779caa26c51294dea4a0aae6b889ef4e9.jpg)

![d0751da513f2fd83c84f1de4236c7112b93e470ce8ca2fb31427ff1dd70cde43.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/d0751da513f2fd83c84f1de4236c7112b93e470ce8ca2fb31427ff1dd70cde43.jpg)

![d42fe1e8913e9b0f6dd1c02e7a1aa42cb7e145cd6b1a2a5fe6d4af75ed0a5508.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/d42fe1e8913e9b0f6dd1c02e7a1aa42cb7e145cd6b1a2a5fe6d4af75ed0a5508.jpg)

![d5fbe02cad77c2e0c4d1470f1b9d93bf9140b74670c3169f77103f4be2387a7a.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/d5fbe02cad77c2e0c4d1470f1b9d93bf9140b74670c3169f77103f4be2387a7a.jpg)

![e00e38ba5dd227440803797537ff309919131ea20252ed7377e3b7be5046addc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/e00e38ba5dd227440803797537ff309919131ea20252ed7377e3b7be5046addc.jpg)

![e09b4304e956037418f249579302551f0a1418e225b69202e68e9e86b0172efc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/e09b4304e956037418f249579302551f0a1418e225b69202e68e9e86b0172efc.jpg)

![e23e99ac71fb23686c016cb8044eaaf6616d3b2b783136162b8f517cb0e55da6.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/e23e99ac71fb23686c016cb8044eaaf6616d3b2b783136162b8f517cb0e55da6.jpg)

![e4830c363fe5f4f98c8fd424a405aae7b6775e4cc35b67d8f99b25106471c199.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/e4830c363fe5f4f98c8fd424a405aae7b6775e4cc35b67d8f99b25106471c199.jpg)

![e598349cabdaca43c76c0d111cbe2bbc9e7c9a1fc99f39d1653f86abe87c11fd.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/e598349cabdaca43c76c0d111cbe2bbc9e7c9a1fc99f39d1653f86abe87c11fd.jpg)

![ea9fbf1504b4f22bf302d6198458f0870a22345d5c180a0ef0342679dd022e00.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ea9fbf1504b4f22bf302d6198458f0870a22345d5c180a0ef0342679dd022e00.jpg)

![eb2cd8a66c8878e71d17d44df000d003446af755ce600945ab0320c7f6c2e375.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/eb2cd8a66c8878e71d17d44df000d003446af755ce600945ab0320c7f6c2e375.jpg)

![ec498d6b706c58d6d5a428ec124fd4dee417deb1cc227456ea15308bc9d5fc15.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/ec498d6b706c58d6d5a428ec124fd4dee417deb1cc227456ea15308bc9d5fc15.jpg)

![efa1b3643fe1971c51eca5e80d395fa62ce2e612b64cbc54094409f70ba41ade.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/efa1b3643fe1971c51eca5e80d395fa62ce2e612b64cbc54094409f70ba41ade.jpg)

![f09055969a035cea166f533ed152471e6626fb3672ffdfe02cf0cb27401561e7.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/f09055969a035cea166f533ed152471e6626fb3672ffdfe02cf0cb27401561e7.jpg)

![f3f49205119f8e383c77f479927cd766e4d1d2f644d20c8e5dbd031a278f0dbc.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/images/f3f49205119f8e383c77f479927cd766e4d1d2f644d20c8e5dbd031a278f0dbc.jpg)

### Tables

![2160866897d5a2a87aa2075ec8b6ff4d1d7acced0e92d51c1911d7c83c363074.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/2160866897d5a2a87aa2075ec8b6ff4d1d7acced0e92d51c1911d7c83c363074.jpg)

![237624437639860b7cc65fe8f62a4350b224bd97da8808399f73f82834f91130.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/237624437639860b7cc65fe8f62a4350b224bd97da8808399f73f82834f91130.jpg)

![3ee29cc32b3416e59bee63d57bb8389199f568a3665ed8cd00c2876aea1e03d4.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/3ee29cc32b3416e59bee63d57bb8389199f568a3665ed8cd00c2876aea1e03d4.jpg)

![3ee64d5321b73fdd47ca8d9b49ee9cfeb573d58ad53c850bb127a911e454505f.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/3ee64d5321b73fdd47ca8d9b49ee9cfeb573d58ad53c850bb127a911e454505f.jpg)

![4cf4a6a038895d7d506b132f3be7924d53cad4ed138088b3d72f636539eff66a.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/4cf4a6a038895d7d506b132f3be7924d53cad4ed138088b3d72f636539eff66a.jpg)

![533e548a1b1c2159ed9aff1e98d4330f1902e150acb71ddcb3f7dc3b014c3145.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/533e548a1b1c2159ed9aff1e98d4330f1902e150acb71ddcb3f7dc3b014c3145.jpg)

![646fc7d6297c94ce8a6e4cde0ef2521c1c8affccbda19a686befbe413565ff50.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/646fc7d6297c94ce8a6e4cde0ef2521c1c8affccbda19a686befbe413565ff50.jpg)

![6b4bd87061119ce7429d37fc1574a48c20b6ad8401642d0c5f547a1e35bac814.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/6b4bd87061119ce7429d37fc1574a48c20b6ad8401642d0c5f547a1e35bac814.jpg)

![79021222778e5d7c38dac793dc20a55f809fa38f9ebab1dcaafcde47b17682bb.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/79021222778e5d7c38dac793dc20a55f809fa38f9ebab1dcaafcde47b17682bb.jpg)

![971402c0c0855dda6ff516b829b788edc06dfbbc8da6f60294fd20fd60938273.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/971402c0c0855dda6ff516b829b788edc06dfbbc8da6f60294fd20fd60938273.jpg)

![cacb236901c015dfe4b5d5ee9a8c68f86106d762a41e90e7b691a11a3a83c67c.jpg](../icml_results/266_Online%20Conformal%20Prediction%20via%20Universal%20Portfolio%20Algorithms/tables/cacb236901c015dfe4b5d5ee9a8c68f86106d762a41e90e7b691a11a3a83c67c.jpg)

<a id="paper-0262"></a>

## WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference


### Images

![16839077700c2e977ee49b80628c51e0ab017386ba867ba0db8d0a91c2d90b78.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/16839077700c2e977ee49b80628c51e0ab017386ba867ba0db8d0a91c2d90b78.jpg)

![60422608c28249d5cb98bd720b33cd73705642b31c7530c6afb6876aae316c0f.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/60422608c28249d5cb98bd720b33cd73705642b31c7530c6afb6876aae316c0f.jpg)

![74757b98d6196c5176aae31a96afb7aece4e7c9ab18e61610380da520cdb845c.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/74757b98d6196c5176aae31a96afb7aece4e7c9ab18e61610380da520cdb845c.jpg)

![77737e09691e962e96d809156cccccf3452a100f05acd1c5ac9fb9f37bce41ca.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/77737e09691e962e96d809156cccccf3452a100f05acd1c5ac9fb9f37bce41ca.jpg)

![a0a81838fb70fc8ec5fd9207705ce6d7f4f38a86aee173bb42d3775d22aeb134.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/a0a81838fb70fc8ec5fd9207705ce6d7f4f38a86aee173bb42d3775d22aeb134.jpg)

![c113a83e80394bab8c7e34882383f6bd5ffede50c869aca2165fbe7d1ad5aa00.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/c113a83e80394bab8c7e34882383f6bd5ffede50c869aca2165fbe7d1ad5aa00.jpg)

![c72d7b43e13950cacb21143b45b984acbcc4df824a03a31542e014956c39da48.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/c72d7b43e13950cacb21143b45b984acbcc4df824a03a31542e014956c39da48.jpg)

![e00919f7dc93662cd320f2418cfc6d568af1c2b22bf8c01ecd62c09fbfa950ca.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/e00919f7dc93662cd320f2418cfc6d568af1c2b22bf8c01ecd62c09fbfa950ca.jpg)

![f89080c57a09f296c1d4dd32bc329f84fe37a16cf5145add01c11ccbd74e659e.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/images/f89080c57a09f296c1d4dd32bc329f84fe37a16cf5145add01c11ccbd74e659e.jpg)

### Tables

![1aa92afabf858bbb2d20ea61f6c46bc85a4d68297e8c4aa1994e1ad3c5dc03f7.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/tables/1aa92afabf858bbb2d20ea61f6c46bc85a4d68297e8c4aa1994e1ad3c5dc03f7.jpg)

![a729be2c33c9429732d79212116f3f933646aa0eac66ec41099baa18abb383ee.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/tables/a729be2c33c9429732d79212116f3f933646aa0eac66ec41099baa18abb383ee.jpg)

![bf8996008c61e77b1831694ad1caa3a57e98293d4f3d786c93b3a9208957e40b.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/tables/bf8996008c61e77b1831694ad1caa3a57e98293d4f3d786c93b3a9208957e40b.jpg)

![f0d53046a93c960f9c91b50a71b818f1a46cf8ed78d50119137a7b0046aba6c0.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/tables/f0d53046a93c960f9c91b50a71b818f1a46cf8ed78d50119137a7b0046aba6c0.jpg)

![fed33a6568572a0fd216c1d48909b6f6230176383c0b80e608399f833b1feed0.jpg](../icml_results/267_WeDLM_%20Reconciling%20Diffusion%20Language%20Models%20with%20Standard%20Causal%20Attention%20for%20Fast%20Inference/tables/fed33a6568572a0fd216c1d48909b6f6230176383c0b80e608399f833b1feed0.jpg)

<a id="paper-0263"></a>

## Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning


### Images

![0ba4ebaab22c4f02b620559047082920dc5e5b2dd6dddd3f8cf501d265e2791e.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/0ba4ebaab22c4f02b620559047082920dc5e5b2dd6dddd3f8cf501d265e2791e.jpg)

![122bc0d7c844d9189b1f362bc0a49237a7b4d5d7ca21f3cc4322c0bcc4167ad6.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/122bc0d7c844d9189b1f362bc0a49237a7b4d5d7ca21f3cc4322c0bcc4167ad6.jpg)

![3a11089318efb5ba31b33c2e8b67402ee5a04a0a169905c677f1be84d554eb74.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/3a11089318efb5ba31b33c2e8b67402ee5a04a0a169905c677f1be84d554eb74.jpg)

![4aed0a5fe0f43f3035ab4c66c51b8ead7fc99bb8933bc776f5734ee05d6c5a58.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/4aed0a5fe0f43f3035ab4c66c51b8ead7fc99bb8933bc776f5734ee05d6c5a58.jpg)

![4ca7962752f39eb710c852947ded03c99c41f0b1b75602f720144b7a96c35c92.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/4ca7962752f39eb710c852947ded03c99c41f0b1b75602f720144b7a96c35c92.jpg)

![50b1e2f27d88f2cf49384fa38079a5a6cd8a4e1bdfb83fbd8e394df00123b499.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/50b1e2f27d88f2cf49384fa38079a5a6cd8a4e1bdfb83fbd8e394df00123b499.jpg)

![558efddf87c797d0dae84a9bb08130ae31884bd2a7d1ee4ef839a64d1c5c0434.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/558efddf87c797d0dae84a9bb08130ae31884bd2a7d1ee4ef839a64d1c5c0434.jpg)

![77d882f11a6ef2914894a98fa326d9233b2532cd9be2bf65529f8411c5c67278.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/77d882f11a6ef2914894a98fa326d9233b2532cd9be2bf65529f8411c5c67278.jpg)

![947d38d2c451c760ef396af3c9dc8bf776a68ac51faf74ad20b0dc97367b6c96.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/947d38d2c451c760ef396af3c9dc8bf776a68ac51faf74ad20b0dc97367b6c96.jpg)

![9c1e127461e6ce2aa07965a2fdbf53526786b3f021354ca97bbe1ab9d37409e6.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/9c1e127461e6ce2aa07965a2fdbf53526786b3f021354ca97bbe1ab9d37409e6.jpg)

![b482aeb7571327b9e41aa7e5fbcd6b917508c888e012f66eea8f834f8032843b.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/b482aeb7571327b9e41aa7e5fbcd6b917508c888e012f66eea8f834f8032843b.jpg)

![c23476589eb4e8cddc0defe6ad111cb6aa8ea35ec366b3851fc9a2029836e554.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/c23476589eb4e8cddc0defe6ad111cb6aa8ea35ec366b3851fc9a2029836e554.jpg)

![c277e09a94d1795ecd28352025d0b2d19693f24b949139c2f256f5202f718548.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/c277e09a94d1795ecd28352025d0b2d19693f24b949139c2f256f5202f718548.jpg)

![e40cdd634c0ef289ccafa98bb61a3f8387f14ed2b7b72ab5cb4e7b1d02397522.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/images/e40cdd634c0ef289ccafa98bb61a3f8387f14ed2b7b72ab5cb4e7b1d02397522.jpg)

### Tables

![019db7f4a13b70c231424be8e75d75612b07780ae5e02ad1f8bc0c42922b72a7.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/019db7f4a13b70c231424be8e75d75612b07780ae5e02ad1f8bc0c42922b72a7.jpg)

![3d4acbd3deff02e980f520616415579f21cc00fa66d16301e9b1b4ffc4f83dd8.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/3d4acbd3deff02e980f520616415579f21cc00fa66d16301e9b1b4ffc4f83dd8.jpg)

![3da6fe586bf757225b38558c62d3fd901a44abae8940af82f07f510fff4217be.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/3da6fe586bf757225b38558c62d3fd901a44abae8940af82f07f510fff4217be.jpg)

![42ec7033e9bdbaacb4ffb314dee6571ad0134f0cded73557f3ecd364fbd72824.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/42ec7033e9bdbaacb4ffb314dee6571ad0134f0cded73557f3ecd364fbd72824.jpg)

![62f0260f2cd2e0cde11b713568628803803542b377c7683d4c266659d37d7b04.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/62f0260f2cd2e0cde11b713568628803803542b377c7683d4c266659d37d7b04.jpg)

![6e880f9848d285c810d56bda597deb6735eca01393d6dd720d88e7b67aea02e5.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/6e880f9848d285c810d56bda597deb6735eca01393d6dd720d88e7b67aea02e5.jpg)

![927ea0e29429cdd7a3eb9a6894e01da4246d918941bcd6faa90e55bb0dd3feaa.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/927ea0e29429cdd7a3eb9a6894e01da4246d918941bcd6faa90e55bb0dd3feaa.jpg)

![933cc14f5c23f448623a600200ca7b96e9e1644528867663422b4aff4948663f.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/933cc14f5c23f448623a600200ca7b96e9e1644528867663422b4aff4948663f.jpg)

![a022de2ffd580b79feb6dfc774121227ea3f3f7a931f16cf3020a34806dd2e71.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/a022de2ffd580b79feb6dfc774121227ea3f3f7a931f16cf3020a34806dd2e71.jpg)

![ab897da27d23a1ed040b82a0f0075de948b52fd8ba3cf0dbf692545a761967aa.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/ab897da27d23a1ed040b82a0f0075de948b52fd8ba3cf0dbf692545a761967aa.jpg)

![c85339dd71fbe2af600abefe39e4993d5047d2ee61ef7dab1892b0a3eb1ec8cc.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/c85339dd71fbe2af600abefe39e4993d5047d2ee61ef7dab1892b0a3eb1ec8cc.jpg)

![de5819284a7330ff7575da7f210ddae22c614c818c03230d61eb3b5c242efa9e.jpg](../icml_results/268_Pretrained%20Vision-Language-Action%20Models%20are%20Surprisingly%20Resistant%20to%20Forgetting%20in%20Continual%20Learn/tables/de5819284a7330ff7575da7f210ddae22c614c818c03230d61eb3b5c242efa9e.jpg)

<a id="paper-0264"></a>

## Local Covariate Selection for Average Causal Effect Estimation without Pretreatment and Causal Sufficiency Assumptions

### Images

![14e7c6ef9d459b14dd6d6baba7a93b08d9e1bd75cd2613decb8570f6e8ac3e7f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/14e7c6ef9d459b14dd6d6baba7a93b08d9e1bd75cd2613decb8570f6e8ac3e7f.jpg)

![3e36c034ce412f1322ef582da94595b30e9fd644797a25f1f908ada5428c1176.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/3e36c034ce412f1322ef582da94595b30e9fd644797a25f1f908ada5428c1176.jpg)

![4dd246e4cb4a2bb37aceb09da20eabd3d9de7cd57a0c699747faaf43c517f82d.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/4dd246e4cb4a2bb37aceb09da20eabd3d9de7cd57a0c699747faaf43c517f82d.jpg)

![7b89bd493612ff2c5e09a5d51546ead28e9ef26ef85a082e8a8a49c3102fb3b5.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/7b89bd493612ff2c5e09a5d51546ead28e9ef26ef85a082e8a8a49c3102fb3b5.jpg)

![8177d992a91a5b1fa14762ba459d4d7ffc8d4fff3eeecfed6aa0fa1f55b03355.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/8177d992a91a5b1fa14762ba459d4d7ffc8d4fff3eeecfed6aa0fa1f55b03355.jpg)

![a6430daf0b7ced99229151545931f572ddc3a2943f08f235659b1cda406687a4.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/a6430daf0b7ced99229151545931f572ddc3a2943f08f235659b1cda406687a4.jpg)

![ba267c6a75a9ad1b937fec2861a392cb404cb771da771953cb5eae1ce6f9aa36.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/ba267c6a75a9ad1b937fec2861a392cb404cb771da771953cb5eae1ce6f9aa36.jpg)

![c3e60463f8d2c80323e1fa647f82d4a02c71b8fb6a4f9ac67ddee130863a945d.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/c3e60463f8d2c80323e1fa647f82d4a02c71b8fb6a4f9ac67ddee130863a945d.jpg)

![cb56352b8587936f23f1e8cad1cb2e0a0ac7a8ec2196fc72070ec67cd817383e.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/cb56352b8587936f23f1e8cad1cb2e0a0ac7a8ec2196fc72070ec67cd817383e.jpg)

![d932e475842d683200a93fd6ec91b5d922de294c178813b4f5fe480472fb61e7.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/d932e475842d683200a93fd6ec91b5d922de294c178813b4f5fe480472fb61e7.jpg)

![e7a1824278a1f6ccaf91d483039a3ba8680301e66ec442ce650d151bca77552f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/e7a1824278a1f6ccaf91d483039a3ba8680301e66ec442ce650d151bca77552f.jpg)

![fe7a75410ec52414d85af672095710e0603721303d837011e785480e4801d46a.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/images/fe7a75410ec52414d85af672095710e0603721303d837011e785480e4801d46a.jpg)

### Tables

![22adf03773a2d225a7b4d23bc85021bdd19283ae00f9264c89a14204b49dfb19.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/22adf03773a2d225a7b4d23bc85021bdd19283ae00f9264c89a14204b49dfb19.jpg)

![29e80fca1b73400877d84adf591ded0cd7d258fa1650f510c2c23cd1dd02196f.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/29e80fca1b73400877d84adf591ded0cd7d258fa1650f510c2c23cd1dd02196f.jpg)

![958029f29eede0901fe80f6479a12631650a9c23eda9a96d7e5e94c16a02a703.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/958029f29eede0901fe80f6479a12631650a9c23eda9a96d7e5e94c16a02a703.jpg)

![97d0585e8471e938ee014d9d992f94d9031520fd345628cb57cd18b4f4ddda96.jpg](../icml_results/269_Local%20Covariate%20Selection%20for%20Average%20Causal%20Effect%20Estimation%20without%20Pretreatment%20and%20Causal%20Suffi/tables/97d0585e8471e938ee014d9d992f94d9031520fd345628cb57cd18b4f4ddda96.jpg)
