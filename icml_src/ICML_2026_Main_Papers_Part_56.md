# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 56 of 197

## 目录 (Table of Contents)

1. [ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](#paper-1816)
2. [Bring Future Vision: Dynamic Computation Allocation Guided by Lightweight Feature Forecaster](#paper-1817)
3. [Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](#paper-1818)
4. [MRPO: Magnitude-Regularized Policy Optimization via L1 Constraints](#paper-1819)
5. [ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind](#paper-1820)
6. [GaussTrace: Provenance Analysis of 3D Gaussian Splatting Models with Evidence-based LLM Reasoning](#paper-1821)
7. [MusicDET: Zero-Shot AI-Generated Music Detection](#paper-1822)
8. [Benchmarking Physics-Informed Time-Series Models for Operational Global Station Weather Forecasting](#paper-1823)
9. [Optimal Pricing for Data-Augmented AutoML Marketplaces](#paper-1824)
10. [$\textttMetaDistill$: Unlocking the Performance Ceiling for Pretrained Optimizers](#paper-1825)
11. [Veda: Scalable Video Diffusion via Distilled Sparse Attention](#paper-1826)
12. [Non-Uniform Noise-to-Signal Ratio in the REINFORCE Policy-Gradient Estimator](#paper-1827)
13. [Robust Multi-View Fusion via Prototype-Anchored Unbalanced Optimal Transport](#paper-1828)
14. [Pose-ICL: 3D-Aware In-Context Learning for Pose-Controllable Subject Customization](#paper-1829)
15. [Omni-Perception Policy Optimization for Multimodal Emotion Reasoning](#paper-1830)
16. [The Entropic Signature of Class Speciation in Diffusion Models](#paper-1831)
17. [Universal Multiclass Transductive Online Learning](#paper-1832)
18. [ViTok-v2: Scaling Native Resolution Autoencoders to 5 Billion Parameters](#paper-1833)
19. [SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity](#paper-1834)
20. [BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining](#paper-1835)
21. [RACER: Risk-Aware Calibrated Efficient Routing for Large Language Models](#paper-1836)
22. [Hyperbolic Associative Memory Networks](#paper-1837)
23. [AI Cartography: Mapping the Latent Landscape of AI Benchmark Ecosystems](#paper-1838)
24. [Skip-It? Theoretical Conditions for Layer Skipping in Vision–Language Models](#paper-1839)
25. [Geodesic Calculus on Implicitly Defined Latent Manifolds](#paper-1840)
26. [Unsupervised Hierarchical Skill Discovery](#paper-1841)
27. [Exact and Approximate Algorithms for Polytree Learning](#paper-1842)
28. [Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](#paper-1843)
29. [No Data? No Problem: Robust Vision-Tabular Learning with Missing Values](#paper-1844)
30. [Spectral Flow Matching: Stabilizing Stochastic GFlowNets via Frequency-Domain Regularization](#paper-1845)
31. [Bottleneck Communication Delay Minimization for Communication-Efficient Decentralized Learning](#paper-1846)
32. [Effective Distillation to Hybrid xLSTM Architectures](#paper-1847)
33. [The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning](#paper-1848)

---


<a id="paper-1816"></a>

## ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models

### Images

![0c51d8f9e04513736ce0aaa94f7c394fc478599f51dcd21fc5585d7996548a6a.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/0c51d8f9e04513736ce0aaa94f7c394fc478599f51dcd21fc5585d7996548a6a.jpg)

![23d9cf5eef00b2ba9f5f1c8e51b112e4cb45e876d77f04c303901c3f6151faa0.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/23d9cf5eef00b2ba9f5f1c8e51b112e4cb45e876d77f04c303901c3f6151faa0.jpg)

![2b039024375293a9f38f84362d18ec15befcc9c794b0e528b5aad881b8bd57af.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/2b039024375293a9f38f84362d18ec15befcc9c794b0e528b5aad881b8bd57af.jpg)

![5c084c0d4a3abfa56d20ff5e2ad18879c877e7174362ab662e517ebd21f4ebe7.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/5c084c0d4a3abfa56d20ff5e2ad18879c877e7174362ab662e517ebd21f4ebe7.jpg)

![6096889bd935827b4a9944eda328c96865d9517e33464d39a893bd72d9fd6042.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/6096889bd935827b4a9944eda328c96865d9517e33464d39a893bd72d9fd6042.jpg)

![8624f75651093f531eb61a6ad9a20098105bd46dbfac1135a4515ccd7147326d.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/8624f75651093f531eb61a6ad9a20098105bd46dbfac1135a4515ccd7147326d.jpg)

![abd3d55b32af66497c27d6a9e560a1daf1f1d2ac3057b59cf7f5ecf32d9b9a6c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/abd3d55b32af66497c27d6a9e560a1daf1f1d2ac3057b59cf7f5ecf32d9b9a6c.jpg)

![cb024d7065a032f7b903ff2fcb3d6269ec9599c985c99cdfb610436b5ee4fd64.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/cb024d7065a032f7b903ff2fcb3d6269ec9599c985c99cdfb610436b5ee4fd64.jpg)

![daa4aea6cfe427365159eae1e2b62821f0c0e9a284617a3f816fb91a4724ae3d.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/daa4aea6cfe427365159eae1e2b62821f0c0e9a284617a3f816fb91a4724ae3d.jpg)

### Tables

![0a1854bbb45d728959e80b30818572120c41a5eeee9b73b90f27f14358bfb5e7.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/0a1854bbb45d728959e80b30818572120c41a5eeee9b73b90f27f14358bfb5e7.jpg)

![0ba034818a8cc4ed27f729d0647f9b6c61decfc28045fcbb253712368eb7c174.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/0ba034818a8cc4ed27f729d0647f9b6c61decfc28045fcbb253712368eb7c174.jpg)

![2df5e8c8f0ab05f46a4e57face601e098bf0bca9dcfca10216c23f8dbc947db1.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/2df5e8c8f0ab05f46a4e57face601e098bf0bca9dcfca10216c23f8dbc947db1.jpg)

![5f35781f4f5bdb61ffa253487b8ffe98a65ad2e49be4098d9bf5ec6cb1fb9b99.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/5f35781f4f5bdb61ffa253487b8ffe98a65ad2e49be4098d9bf5ec6cb1fb9b99.jpg)

![7ed18a696d8063b642649af3f0deaa8175b24eab6b307ddf59e5af8b0a836f4c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/7ed18a696d8063b642649af3f0deaa8175b24eab6b307ddf59e5af8b0a836f4c.jpg)

![85e15b45145953dda77b84ed91ed5e7f6dd16dfac4a7e36404fae5bd9a306e3e.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/85e15b45145953dda77b84ed91ed5e7f6dd16dfac4a7e36404fae5bd9a306e3e.jpg)

![95f7f896dbeb10197be845aab0f16ae0972763f2a4bdf8ca030b2c66f95f5c0e.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/95f7f896dbeb10197be845aab0f16ae0972763f2a4bdf8ca030b2c66f95f5c0e.jpg)

![a0318afb732560ff134b3253a0c5ff73ad37592edaf14f05910929210ece83b2.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/a0318afb732560ff134b3253a0c5ff73ad37592edaf14f05910929210ece83b2.jpg)

![a3ca3e063af6ebeb8d0ee9a54c42dde8fa260b7b53b9c95e4c1b842f20699aee.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/a3ca3e063af6ebeb8d0ee9a54c42dde8fa260b7b53b9c95e4c1b842f20699aee.jpg)

![aac06f07abb1be5af3eff532ec525e4432033568cc8ba649942210cc83485769.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/aac06f07abb1be5af3eff532ec525e4432033568cc8ba649942210cc83485769.jpg)

![d287b1204f0fdb90f70974513c539020e019cdca96ba62d72bf707a675de1c0c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/d287b1204f0fdb90f70974513c539020e019cdca96ba62d72bf707a675de1c0c.jpg)

## ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models


### Images

![0e7d45b5e65591c24ead01071d3d2b90a364c9579d444ffa30a0e48a86e8d8e7.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/0e7d45b5e65591c24ead01071d3d2b90a364c9579d444ffa30a0e48a86e8d8e7.jpg)

![17ac27d00998490a635b4cc5020cae0ef4f59263c19221e7872170befa1bb38a.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/17ac27d00998490a635b4cc5020cae0ef4f59263c19221e7872170befa1bb38a.jpg)

![21d3b5255e3893a694874754a11ea1e22ecfef29e55d3a408e2ed59abc4ae666.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/21d3b5255e3893a694874754a11ea1e22ecfef29e55d3a408e2ed59abc4ae666.jpg)

![2c995cc6a75e74486f6f63cff8ced9d56f6ff4e677e49e338bd02517699b235f.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/2c995cc6a75e74486f6f63cff8ced9d56f6ff4e677e49e338bd02517699b235f.jpg)

![46098484178e16a5425d46aae8ae7606466e7917f249c4586bef43d240159c61.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/46098484178e16a5425d46aae8ae7606466e7917f249c4586bef43d240159c61.jpg)

![49982807ab669806b952d000d34da43f17d2043a2bbaeb667dcee3b64c471a0f.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/49982807ab669806b952d000d34da43f17d2043a2bbaeb667dcee3b64c471a0f.jpg)

![6dafe2bf5e1852b356c10f97d6320d2968b7e63d6f45a4a7825360074ce931ca.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/6dafe2bf5e1852b356c10f97d6320d2968b7e63d6f45a4a7825360074ce931ca.jpg)

![718b55b1fdac69d3572c1409a048a1f3192c16be6a4367928ce1c9e920175d21.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/718b55b1fdac69d3572c1409a048a1f3192c16be6a4367928ce1c9e920175d21.jpg)

![7e4ca96a0fd8fc28088636b3f7ff48ff9e4956d3f36c6028e38c42adc22c6dd4.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/7e4ca96a0fd8fc28088636b3f7ff48ff9e4956d3f36c6028e38c42adc22c6dd4.jpg)

![828b7d49c44a2e80bfd872f4e0f745e3f48977f784095f97a9fe5334dc0c1f01.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/828b7d49c44a2e80bfd872f4e0f745e3f48977f784095f97a9fe5334dc0c1f01.jpg)

![9da9341e7ebdde6322e2be0cf682471511870e8dd38142f266a0c0daf3b0ff3e.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/9da9341e7ebdde6322e2be0cf682471511870e8dd38142f266a0c0daf3b0ff3e.jpg)

![b187f8408833831498e8f293b083e5d53f03a0ceff1b06b37f5b1e71f6f93e19.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/b187f8408833831498e8f293b083e5d53f03a0ceff1b06b37f5b1e71f6f93e19.jpg)

![b6af74c2361bf5e779d5976ce67160a02edd06cad15954a8d1840eee14753fc8.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/b6af74c2361bf5e779d5976ce67160a02edd06cad15954a8d1840eee14753fc8.jpg)

![e779541106f10cbe6f347afb32c0d68e74c99956f7289c02a74d27eb916753b1.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/e779541106f10cbe6f347afb32c0d68e74c99956f7289c02a74d27eb916753b1.jpg)

![ebeda3794c2f14ece17869a34f73cfe6c7425c4b8c85099766d80e341da6fadc.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/images/ebeda3794c2f14ece17869a34f73cfe6c7425c4b8c85099766d80e341da6fadc.jpg)

### Tables

![07ac578e3bcebc98baf32e092e08dc732fcc77189dc5f368a76e8d87f5e78c35.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/07ac578e3bcebc98baf32e092e08dc732fcc77189dc5f368a76e8d87f5e78c35.jpg)

![1ec0616a00c34a26b128c606438a00ab3c62fba352b1fe02473695fea3a7224f.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/1ec0616a00c34a26b128c606438a00ab3c62fba352b1fe02473695fea3a7224f.jpg)

![24781e44adcf62bb51e8474eecf860c9f9b46ae8a96229523d2d351492dd71e7.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/24781e44adcf62bb51e8474eecf860c9f9b46ae8a96229523d2d351492dd71e7.jpg)

![3ba9a29453f555468b60ec32f7f90777c72028e80b77e73a8ca939b79be324b0.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/3ba9a29453f555468b60ec32f7f90777c72028e80b77e73a8ca939b79be324b0.jpg)

![4d73a93f7025fad25f26e610fab9d7f776ed8a1637c00cac5e955b9765c965bf.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/4d73a93f7025fad25f26e610fab9d7f776ed8a1637c00cac5e955b9765c965bf.jpg)

![5dd9f2788e4ad215fa5d5d85757b70de71c3f05ab4bd628e85bbae24e482d9f0.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/5dd9f2788e4ad215fa5d5d85757b70de71c3f05ab4bd628e85bbae24e482d9f0.jpg)

![5fe5965bde54822be5afc1752e0b13f9ad5d4e4812ce05aa8bf09ca86f8475b6.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/5fe5965bde54822be5afc1752e0b13f9ad5d4e4812ce05aa8bf09ca86f8475b6.jpg)

![68aa4c0806994e58aeff2a2dc9d1b84acd892798b1235fe88c8ac589c4879a2f.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/68aa4c0806994e58aeff2a2dc9d1b84acd892798b1235fe88c8ac589c4879a2f.jpg)

![69de6017a877d85e0a4001a015a1f9b9ba92c5ea50eb674f78a041aab297b784.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/69de6017a877d85e0a4001a015a1f9b9ba92c5ea50eb674f78a041aab297b784.jpg)

![6d5376f4f8e97089327a4d4963da87ea2223a36f19a413b30769eb9984133b7c.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/6d5376f4f8e97089327a4d4963da87ea2223a36f19a413b30769eb9984133b7c.jpg)

![8af0ef3b0e43169b7bdbe422bb9deb1eabe9a89d0fb2afb133d9b215d0bcb5fb.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/8af0ef3b0e43169b7bdbe422bb9deb1eabe9a89d0fb2afb133d9b215d0bcb5fb.jpg)

![b2229973e00d98d89f683e07c71fc71ef72ea2db8c721126466acd8ca5d56e3e.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/b2229973e00d98d89f683e07c71fc71ef72ea2db8c721126466acd8ca5d56e3e.jpg)

![c26309c6d8002dac8985bd40211475e3521021841ff9fe8a512d4af2d6db2e40.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/c26309c6d8002dac8985bd40211475e3521021841ff9fe8a512d4af2d6db2e40.jpg)

![d0873efb32d9d150958e98fe2e94448b26a4931f480a6e0b6cacb0d9a9cc1bcd.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/d0873efb32d9d150958e98fe2e94448b26a4931f480a6e0b6cacb0d9a9cc1bcd.jpg)

![e9205dc2ebc362126fbd015959a7323e3917466b9bc1d62a32d3d8615bfcaa53.jpg](../icml_results/1839_ForceForget_%20Reinforcement%20Concept%20Removal%20for%20Enhancing%20Safety%20in%20Text-to-Image%20Models/tables/e9205dc2ebc362126fbd015959a7323e3917466b9bc1d62a32d3d8615bfcaa53.jpg)

<a id="paper-1817"></a>

## Bring Future Vision: Dynamic Computation Allocation Guided by Lightweight Feature Forecaster


### Images

![4b0785ed0d6bca305235461defba34c865f5a917ce017e852ee3e379930785d6.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/images/4b0785ed0d6bca305235461defba34c865f5a917ce017e852ee3e379930785d6.jpg)

![6908cb8fc0276db1078675bf75ff2528ab06389f128341d9bd0d75c80540c7d4.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/images/6908cb8fc0276db1078675bf75ff2528ab06389f128341d9bd0d75c80540c7d4.jpg)

![c85c886f9600463e99b10c77182b1e7b5e3b9e8c27497d588f4e683b61c0e681.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/images/c85c886f9600463e99b10c77182b1e7b5e3b9e8c27497d588f4e683b61c0e681.jpg)

### Tables

![0531bd1f8445fdb4a6655a9e84cb3d93115831bd0c829d4b5bcb13bf770892ec.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/0531bd1f8445fdb4a6655a9e84cb3d93115831bd0c829d4b5bcb13bf770892ec.jpg)

![1dafd5e683919de9d490f9af16823af20944132a3c1c651468af9b0e0781a0e5.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/1dafd5e683919de9d490f9af16823af20944132a3c1c651468af9b0e0781a0e5.jpg)

![692a05c1640444db54a22fdc53a3e8c10796cfde37c312c041d14a30c6f906ae.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/692a05c1640444db54a22fdc53a3e8c10796cfde37c312c041d14a30c6f906ae.jpg)

![8532ebd7bbc770acfb428bea7d0082774ab1f83597f386be2d0c814397be6d25.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/8532ebd7bbc770acfb428bea7d0082774ab1f83597f386be2d0c814397be6d25.jpg)

![8c3a68f51520f4eae7c899f6617a5991b8df025121a971e3784276432fc7038b.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/8c3a68f51520f4eae7c899f6617a5991b8df025121a971e3784276432fc7038b.jpg)

![998214ef9ec40b1ea4fd85c5276b961471c8b38401248e5ad87b2a6cd831bcff.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/998214ef9ec40b1ea4fd85c5276b961471c8b38401248e5ad87b2a6cd831bcff.jpg)

![9ea4533d61e30c5be01d352400edb96d94b3474f0f64358e984713901d610b08.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/9ea4533d61e30c5be01d352400edb96d94b3474f0f64358e984713901d610b08.jpg)

![ab5b9354a553434ee8da9d1e5f38978a66f3bb3f4565a54fcf39fc0ceeb22612.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/ab5b9354a553434ee8da9d1e5f38978a66f3bb3f4565a54fcf39fc0ceeb22612.jpg)

![b14ad78e6c24ccda146649223fd141f13a16863497167de96a1d4083574cc472.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/b14ad78e6c24ccda146649223fd141f13a16863497167de96a1d4083574cc472.jpg)

![dbf3cf5ab2e3d153ddbcf65f62bf9848946ae723ba86735be50155665d116b99.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/dbf3cf5ab2e3d153ddbcf65f62bf9848946ae723ba86735be50155665d116b99.jpg)

![e0f5ec415d5179776659d87012a8ad8df2582b5374407cd06813ab1bcbfa9361.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/e0f5ec415d5179776659d87012a8ad8df2582b5374407cd06813ab1bcbfa9361.jpg)

![e8b9771428e89d11b8653de8d8c1354f11b87d44ae2d12b8b5e06d66ce8b426c.jpg](../icml_results/1840_Bring%20Future%20Vision_%20Dynamic%20Computation%20Allocation%20Guided%20by%20Lightweight%20Feature%20Forecaster/tables/e8b9771428e89d11b8653de8d8c1354f11b87d44ae2d12b8b5e06d66ce8b426c.jpg)

<a id="paper-1818"></a>

## Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models


### Images

![026034865809aa17b65bbc8c46bcb9589f6c66ae614771580cd5a23ad008062f.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/026034865809aa17b65bbc8c46bcb9589f6c66ae614771580cd5a23ad008062f.jpg)

![08a1b62fccf32d3402e0dbe949ad5d24febc2e46561fcb96209ac10bc2cb1558.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/08a1b62fccf32d3402e0dbe949ad5d24febc2e46561fcb96209ac10bc2cb1558.jpg)

![183f757bd8350feb4662ddddb218b91b7d293095e7c5b0d0b25229778d1b82ca.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/183f757bd8350feb4662ddddb218b91b7d293095e7c5b0d0b25229778d1b82ca.jpg)

![1f307b74b2e424c34e67cffefeed1e4827c122a6f26db1371b650f0588140eb5.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/1f307b74b2e424c34e67cffefeed1e4827c122a6f26db1371b650f0588140eb5.jpg)

![2064d0ba8b910f2135af33e647b5c949dc73012c7c01b3ebeac5728af96f7c04.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/2064d0ba8b910f2135af33e647b5c949dc73012c7c01b3ebeac5728af96f7c04.jpg)

![26bee4e546bd3f0042c25c6bb503cfb0e0c891a9e9eb3d725e83a42e937d4a92.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/26bee4e546bd3f0042c25c6bb503cfb0e0c891a9e9eb3d725e83a42e937d4a92.jpg)

![29a644e40c88e74927d74ce80f4775b690bfde2b063c03d5959724ac49e98e7d.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/29a644e40c88e74927d74ce80f4775b690bfde2b063c03d5959724ac49e98e7d.jpg)

![307ace23281c69522b24c0a29bf6561e66b9fd5a1c5334cd33396d3032dce5c8.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/307ace23281c69522b24c0a29bf6561e66b9fd5a1c5334cd33396d3032dce5c8.jpg)

![33b68127416e6986758beb869c5d25e6f6afc5cc387b8e477dd218e72f921b5c.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/33b68127416e6986758beb869c5d25e6f6afc5cc387b8e477dd218e72f921b5c.jpg)

![3628a573fc3c22cdf4ad3ea66fe53182ad206c1ff077826e7f55e618d87379ef.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/3628a573fc3c22cdf4ad3ea66fe53182ad206c1ff077826e7f55e618d87379ef.jpg)

![388694d92a522445fa84fe687089bbbe50d578c2f3963edbe3e40f4e602f77e1.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/388694d92a522445fa84fe687089bbbe50d578c2f3963edbe3e40f4e602f77e1.jpg)

![389665f8125a45487f7717d3e29bf97deef9cdeacd3839ef6636379674941f69.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/389665f8125a45487f7717d3e29bf97deef9cdeacd3839ef6636379674941f69.jpg)

![3899aeeca04280ff89e4f4267dd1b3c78071b00dda927cfc5506b2d658d40f39.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/3899aeeca04280ff89e4f4267dd1b3c78071b00dda927cfc5506b2d658d40f39.jpg)

![3ae4a87445075446436140f7d7b8aecbc6c71c0fa7790938f682c27add79f944.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/3ae4a87445075446436140f7d7b8aecbc6c71c0fa7790938f682c27add79f944.jpg)

![434b0a977a612ab58f2d0f020950a2724c3e67c4a5302a28bf3abfb4b0956dec.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/434b0a977a612ab58f2d0f020950a2724c3e67c4a5302a28bf3abfb4b0956dec.jpg)

![4d8846de75f32bbae73ac359c7d8fa1837ba2040b832d949a20ff7cb5038b4e2.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/4d8846de75f32bbae73ac359c7d8fa1837ba2040b832d949a20ff7cb5038b4e2.jpg)

![4f8f3162c87d20a427cfa9948757046a82091b412b6d97bfec7e9f057c2be3fb.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/4f8f3162c87d20a427cfa9948757046a82091b412b6d97bfec7e9f057c2be3fb.jpg)

![50a9d32d4b1a648cdada3fb08dada5674da709dafe5d59730664fda4c6a97af2.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/50a9d32d4b1a648cdada3fb08dada5674da709dafe5d59730664fda4c6a97af2.jpg)

![53faad40219771f421f7c3776e1fa21c85ce4a1856bf69631702e33bc788b71b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/53faad40219771f421f7c3776e1fa21c85ce4a1856bf69631702e33bc788b71b.jpg)

![5b0be187ee4855ab7769478dea97a1d3a8632986849b0644aed21616ff365771.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/5b0be187ee4855ab7769478dea97a1d3a8632986849b0644aed21616ff365771.jpg)

![6176e5258cf664ea89af69403803aa89ef7a18a79bad4ad577defb3a1df01776.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/6176e5258cf664ea89af69403803aa89ef7a18a79bad4ad577defb3a1df01776.jpg)

![682f5a700866f1c56edbe42070c4007068c15c9cd8c161fa71de8dce1cdc0a76.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/682f5a700866f1c56edbe42070c4007068c15c9cd8c161fa71de8dce1cdc0a76.jpg)

![6e0bc1d83d542b769baef90357ab76c692850443f5b097c19b9fa77798040370.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/6e0bc1d83d542b769baef90357ab76c692850443f5b097c19b9fa77798040370.jpg)

![7149868ffd3d72d8dcab2252853d64f3cf0106a57001be168d275b149fc54465.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/7149868ffd3d72d8dcab2252853d64f3cf0106a57001be168d275b149fc54465.jpg)

![7319e1b3e20b6edf0a4f6d485090f5f9f8714f16d6ecc651aeb99067ccd60358.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/7319e1b3e20b6edf0a4f6d485090f5f9f8714f16d6ecc651aeb99067ccd60358.jpg)

![8609e54414096d58bb42654413a2b12bdc215cef2b2efcbd59001043b9abcd68.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/8609e54414096d58bb42654413a2b12bdc215cef2b2efcbd59001043b9abcd68.jpg)

![8a8cb4a91af801c4f5787a003d5cc96b6c6e539c96f7245e4975052942a8623d.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/8a8cb4a91af801c4f5787a003d5cc96b6c6e539c96f7245e4975052942a8623d.jpg)

![8cafd7948d6553281a09277ddb1090714e9f71830c99613a320a4925304c24fd.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/8cafd7948d6553281a09277ddb1090714e9f71830c99613a320a4925304c24fd.jpg)

![90dc9908a73721ccf64360f7f7992eff1cc077daabdf1792a6adaeb127413e7f.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/90dc9908a73721ccf64360f7f7992eff1cc077daabdf1792a6adaeb127413e7f.jpg)

![9a8f62af003313540c6f82bae4c98b5941c9b665dceb3b0f0b4d27e790d7cf59.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/9a8f62af003313540c6f82bae4c98b5941c9b665dceb3b0f0b4d27e790d7cf59.jpg)

![9b160217c06c6a1b00c2df6c8e3636cde95875bff6ac53e4d6b80770cc1d9fd2.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/9b160217c06c6a1b00c2df6c8e3636cde95875bff6ac53e4d6b80770cc1d9fd2.jpg)

![aa2f3667d6c0bccb8e256a5ee668450561fcb79e3c166ad2840fe7d68bd57b71.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/aa2f3667d6c0bccb8e256a5ee668450561fcb79e3c166ad2840fe7d68bd57b71.jpg)

![af06c6476ca67fe8b9ac25c3a72b72d1a4a1ed8e6b5fafa6d26b68858d1d5c08.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/af06c6476ca67fe8b9ac25c3a72b72d1a4a1ed8e6b5fafa6d26b68858d1d5c08.jpg)

![af50ba9ed334e84bce3962a474640f1b32fed1f3bd7a54c5c25cd01ed943759b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/af50ba9ed334e84bce3962a474640f1b32fed1f3bd7a54c5c25cd01ed943759b.jpg)

![b0f14d2df5e349d9556a5673127b23995be1f70ef7fad340828390d2142a7afe.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/b0f14d2df5e349d9556a5673127b23995be1f70ef7fad340828390d2142a7afe.jpg)

![b87c6b6a5b22558d81e93b3cb0f4ed3e4ac6733dbf987274dd7ab867b03860f7.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/b87c6b6a5b22558d81e93b3cb0f4ed3e4ac6733dbf987274dd7ab867b03860f7.jpg)

![b88e9eccf2c012424a6cc077a147a10d893d403481463bceebe77cfab21843a5.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/b88e9eccf2c012424a6cc077a147a10d893d403481463bceebe77cfab21843a5.jpg)

![be8162d1e1624f2f44b2a65b11897c6cd9944242d4382dd6f07d66324f7bc277.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/be8162d1e1624f2f44b2a65b11897c6cd9944242d4382dd6f07d66324f7bc277.jpg)

![c341c9d87e1e49418ec12b82500b6ad7fef1a3cd2237aab935ffa9a1d20aaf20.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/c341c9d87e1e49418ec12b82500b6ad7fef1a3cd2237aab935ffa9a1d20aaf20.jpg)

![c56f1bf15de19d39b5bceeb0a917d7f53e7b6d1e45ad83cf8f7d36b6ed7a3385.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/c56f1bf15de19d39b5bceeb0a917d7f53e7b6d1e45ad83cf8f7d36b6ed7a3385.jpg)

![c888e8e9542a0d464a54bc5daccc7f9723184bff338d85650b33194ac2306b97.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/c888e8e9542a0d464a54bc5daccc7f9723184bff338d85650b33194ac2306b97.jpg)

![ccb728fdf091b67bd0f0442ea3762fc869bec15fffe18e3ba0c83a3b6df8d414.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/ccb728fdf091b67bd0f0442ea3762fc869bec15fffe18e3ba0c83a3b6df8d414.jpg)

![cd2151abd03cce689bb546e3e22fc260aa993e68c72de5023825707b096d8cc9.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/cd2151abd03cce689bb546e3e22fc260aa993e68c72de5023825707b096d8cc9.jpg)

![e0622b39b14657392696a8d42238a39aecfa8eefd9ba59ffd6c421c8f9623456.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e0622b39b14657392696a8d42238a39aecfa8eefd9ba59ffd6c421c8f9623456.jpg)

![e222e0a7f5d37eb66c3a2a1ecf686b6818fa55de8bef0c75f5296ed1be47048c.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e222e0a7f5d37eb66c3a2a1ecf686b6818fa55de8bef0c75f5296ed1be47048c.jpg)

![e3fd818289da7c9e36a063e0448efe459747c8d3eb04b73f2a24907c0c5b0732.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e3fd818289da7c9e36a063e0448efe459747c8d3eb04b73f2a24907c0c5b0732.jpg)

![e4409c14ac0ddc5012a4c27104bca25f03e27eee82c60871b368ef8ee9df6ca8.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e4409c14ac0ddc5012a4c27104bca25f03e27eee82c60871b368ef8ee9df6ca8.jpg)

![e8e9f57d3f24d29c00d28e4d23d1bd4b8d0d67505aca66597c5b297e15215e58.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e8e9f57d3f24d29c00d28e4d23d1bd4b8d0d67505aca66597c5b297e15215e58.jpg)

![e977e868691a1d18dc81cd2c77a06ac1b1be7313474dd3b39e65059d8fa532eb.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/e977e868691a1d18dc81cd2c77a06ac1b1be7313474dd3b39e65059d8fa532eb.jpg)

![f227ceff572732fba337a6afb578e40978618ac7f33074a4c3386db138cf8f66.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/images/f227ceff572732fba337a6afb578e40978618ac7f33074a4c3386db138cf8f66.jpg)

### Tables

![0385c59d08195f89ae8c16dd6b5b328bf10ddddd2d449211f98f25cf5321c32a.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/0385c59d08195f89ae8c16dd6b5b328bf10ddddd2d449211f98f25cf5321c32a.jpg)

![04e4fd85c14ff4688d4e7793f376ced07b2b2ff21bdc2a0f7244b7ba8f0c991a.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/04e4fd85c14ff4688d4e7793f376ced07b2b2ff21bdc2a0f7244b7ba8f0c991a.jpg)

![33acf8a20ea3c2b89b2eb699a3bf7934c02daa38339179cf04ed47ea6e2b4048.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/33acf8a20ea3c2b89b2eb699a3bf7934c02daa38339179cf04ed47ea6e2b4048.jpg)

![343026d301f81ae3121056768530c33f7629b85e798eba48fe139bb867fc0adc.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/343026d301f81ae3121056768530c33f7629b85e798eba48fe139bb867fc0adc.jpg)

![38d6ec43565370be50961466e801e08c5b5f933fd274bd0a0ed631e2e3574f28.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/38d6ec43565370be50961466e801e08c5b5f933fd274bd0a0ed631e2e3574f28.jpg)

![3b73c9a03f4aac02727141a435db2e39b7098a492cfcc7271899fabcbe43aa30.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/3b73c9a03f4aac02727141a435db2e39b7098a492cfcc7271899fabcbe43aa30.jpg)

![448805964af134affb8a332f71e0ce38114e6447fd0a131a998f9fcc2a38e3cd.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/448805964af134affb8a332f71e0ce38114e6447fd0a131a998f9fcc2a38e3cd.jpg)

![48343e13f68fd0135cd06879adac42f1bd358f7f0b353568b9219a85dd343ecf.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/48343e13f68fd0135cd06879adac42f1bd358f7f0b353568b9219a85dd343ecf.jpg)

![4c960c549b5c585c5c7d08934057bea951c3916143d971f50c163501a77feeb3.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/4c960c549b5c585c5c7d08934057bea951c3916143d971f50c163501a77feeb3.jpg)

![5020d3044571fbd97ecc8ce47cf3db24b41baf65f317cccfc97121a7972348e4.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/5020d3044571fbd97ecc8ce47cf3db24b41baf65f317cccfc97121a7972348e4.jpg)

![55659e7dd46596cecc9456c9fd8e5b59de694441f651ee5d1b6609fee7b05d81.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/55659e7dd46596cecc9456c9fd8e5b59de694441f651ee5d1b6609fee7b05d81.jpg)

![5aba027b951492175764da1a9ad7bd5e2534c869faf7dafa49ef475686ee0315.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/5aba027b951492175764da1a9ad7bd5e2534c869faf7dafa49ef475686ee0315.jpg)

![5fb9b6a817ca69935e92382fa56b098af53a41a6202618de7562d5d2ec75da6b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/5fb9b6a817ca69935e92382fa56b098af53a41a6202618de7562d5d2ec75da6b.jpg)

![6cc4060846fe855b3e47ef705c1c282bd43a94ca7202743584b37d5420ded5ca.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/6cc4060846fe855b3e47ef705c1c282bd43a94ca7202743584b37d5420ded5ca.jpg)

![6de2ae22eba0f3ca8fbc109c79e0a5504a772a7e8b0a541de18411a4d1c2834e.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/6de2ae22eba0f3ca8fbc109c79e0a5504a772a7e8b0a541de18411a4d1c2834e.jpg)

![6ef387dd68602c8f06728a5991ab5550e9bb37f4411ace4720a2ccda7a0b3362.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/6ef387dd68602c8f06728a5991ab5550e9bb37f4411ace4720a2ccda7a0b3362.jpg)

![75b17ed576c58e43fb119cf0e2b5441585129aef7b125dbe87b905f310ec8058.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/75b17ed576c58e43fb119cf0e2b5441585129aef7b125dbe87b905f310ec8058.jpg)

![7f4ec995d0faebd2682d4681ee4f0e32c9e662a57ff6eee94a4696446bd31fbe.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/7f4ec995d0faebd2682d4681ee4f0e32c9e662a57ff6eee94a4696446bd31fbe.jpg)

![91bba3ecfb6f461c08737467460f9743b623109d28d1782c7c67351b2340f15b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/91bba3ecfb6f461c08737467460f9743b623109d28d1782c7c67351b2340f15b.jpg)

![a09a28d03df6e9400798addbdb6ddd4eb60c3f3d1db653f8a2f23201097ec412.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/a09a28d03df6e9400798addbdb6ddd4eb60c3f3d1db653f8a2f23201097ec412.jpg)

![acdbb661176ac54e171ed1c7506f3421da4b188b32ee66b37dec89d03bd8d57b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/acdbb661176ac54e171ed1c7506f3421da4b188b32ee66b37dec89d03bd8d57b.jpg)

![afbdd2886a962b315d67c4aad29b012b708b32dca726b948376ecc12a72753f9.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/afbdd2886a962b315d67c4aad29b012b708b32dca726b948376ecc12a72753f9.jpg)

![b31f4f25ca95ddc3e17e6f887edc8c91954678eaff013e646662579b670f3f60.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/b31f4f25ca95ddc3e17e6f887edc8c91954678eaff013e646662579b670f3f60.jpg)

![b7e88a6046a6b2e64ea40ac39c87d203a326c73352637bd77abe47515c3601f2.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/b7e88a6046a6b2e64ea40ac39c87d203a326c73352637bd77abe47515c3601f2.jpg)

![bb156984b4546091dabcd46625141a823f46c5b757f66f3ef553de4fe7cd985f.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/bb156984b4546091dabcd46625141a823f46c5b757f66f3ef553de4fe7cd985f.jpg)

![c5e8199ca0c1bbd1805784611874ee1a4bfffedfd4c5c43c5822bae6bb2c79bd.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/c5e8199ca0c1bbd1805784611874ee1a4bfffedfd4c5c43c5822bae6bb2c79bd.jpg)

![d09e172516867bde83769d33660a96871ae9137d103fcee3477a5da7457395cb.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/d09e172516867bde83769d33660a96871ae9137d103fcee3477a5da7457395cb.jpg)

![d12231885144386eb1223addcf54d0f6b1989b0c398d33d73cf842101e08714a.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/d12231885144386eb1223addcf54d0f6b1989b0c398d33d73cf842101e08714a.jpg)

![d1930279598830fc2e7045521c219adfe73d69f5ae23a181a08057b4d6e39f59.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/d1930279598830fc2e7045521c219adfe73d69f5ae23a181a08057b4d6e39f59.jpg)

![d8257558e6e3b4923993f3b94e181f77c9da4f49cc47d175295fa3ee30c4cb20.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/d8257558e6e3b4923993f3b94e181f77c9da4f49cc47d175295fa3ee30c4cb20.jpg)

![db487afaacbe93233b2972d76896b922a3369a103d69c78a2402678f8f2efad6.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/db487afaacbe93233b2972d76896b922a3369a103d69c78a2402678f8f2efad6.jpg)

![db9c5a9a667b61901b7220ba23ecd59319af2230c8d2c028a1dbbf87e4095fd0.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/db9c5a9a667b61901b7220ba23ecd59319af2230c8d2c028a1dbbf87e4095fd0.jpg)

![dc14702f6a566fea10ca4d53351f713e8181c85b329c9aea8a48104c538d62bf.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/dc14702f6a566fea10ca4d53351f713e8181c85b329c9aea8a48104c538d62bf.jpg)

![de305611f9944d7da5584f7cf782543faa070bdcd3793d0614ad309ded7e7a71.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/de305611f9944d7da5584f7cf782543faa070bdcd3793d0614ad309ded7e7a71.jpg)

![e2bed59b73439740896a2ad1ce20c018de7927938f9d24724732cc08dc3cddec.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/e2bed59b73439740896a2ad1ce20c018de7927938f9d24724732cc08dc3cddec.jpg)

![e38cb6755bb316bc93e352152a7776899b2a855b6ccb0663cfab8187a719f92b.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/e38cb6755bb316bc93e352152a7776899b2a855b6ccb0663cfab8187a719f92b.jpg)

![fa7dd0b63662af734f16f846a7276c18f4be276eac184e17f20fdc1efd4666f4.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/fa7dd0b63662af734f16f846a7276c18f4be276eac184e17f20fdc1efd4666f4.jpg)

![fc68eeacb1e2cde9553476f75a5125c1411b593fa7f92ee1393fc96c09e41e66.jpg](../icml_results/1841_Dual%20Mechanisms%20of%20Value%20Expression_%20Intrinsic%20vs.%20Prompted%20Values%20in%20Large%20Language%20Models/tables/fc68eeacb1e2cde9553476f75a5125c1411b593fa7f92ee1393fc96c09e41e66.jpg)

<a id="paper-1819"></a>

## MRPO: Magnitude-Regularized Policy Optimization via L1 Constraints


### Images

![45748799ca551a2365e64537f4ac9b74ec32747872622970224591ac07a321c8.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/images/45748799ca551a2365e64537f4ac9b74ec32747872622970224591ac07a321c8.jpg)

![7fcc86347e5882d84ce4687924c769afe84d53f387e5ced43bfb4c2bcaf37957.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/images/7fcc86347e5882d84ce4687924c769afe84d53f387e5ced43bfb4c2bcaf37957.jpg)

![afca692a27018d81a258ea2dfac001444ff5055a471d3b7f1cdb0b56845e695f.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/images/afca692a27018d81a258ea2dfac001444ff5055a471d3b7f1cdb0b56845e695f.jpg)

### Tables

![18bb925431ade3f031250968641b48d1341812edf90364a5d49ff1bc6a484617.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/18bb925431ade3f031250968641b48d1341812edf90364a5d49ff1bc6a484617.jpg)

![660cba1751a9e15099ba606a19226a4a2126404923ef9eeefd3e2d2ca9340d0b.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/660cba1751a9e15099ba606a19226a4a2126404923ef9eeefd3e2d2ca9340d0b.jpg)

![7c6879c005ccd9e192dfeb905f3c3baecc1945162d52034b8aeca24190b91968.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/7c6879c005ccd9e192dfeb905f3c3baecc1945162d52034b8aeca24190b91968.jpg)

![86a3da66901a2227123d5714f1856c14d18af811150bd772a4e076b1959736b7.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/86a3da66901a2227123d5714f1856c14d18af811150bd772a4e076b1959736b7.jpg)

![91b98d8b0dbc1e3f420eab2fc94cf374506927d118e27902c5d7ca381970318f.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/91b98d8b0dbc1e3f420eab2fc94cf374506927d118e27902c5d7ca381970318f.jpg)

![a2f6a8db587244943ac78982cf8ba58b1fad225d626abf19dd26dad6fa23a207.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/a2f6a8db587244943ac78982cf8ba58b1fad225d626abf19dd26dad6fa23a207.jpg)

![adbf26fe3bfd18f6e0e79f8f9625435b28703e6c66820dd2ed9579e5b493cbc6.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/adbf26fe3bfd18f6e0e79f8f9625435b28703e6c66820dd2ed9579e5b493cbc6.jpg)

![dd6aa2d5184cf78bfc554465e3aaece20e8749869983072269ccd10fe6dae695.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/dd6aa2d5184cf78bfc554465e3aaece20e8749869983072269ccd10fe6dae695.jpg)

![efc9905e96e7d22d4ab04c9b7c9656da394b4ce2323fdcee30adb9a3ff614edd.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/efc9905e96e7d22d4ab04c9b7c9656da394b4ce2323fdcee30adb9a3ff614edd.jpg)

![f94679e833f86c72be9913c87fa0147ab3aa783b1c19f15f9d76f05f028e1327.jpg](../icml_results/1842_MRPO_%20Magnitude-Regularized%20Policy%20Optimization%20via%20L1%20Constraints/tables/f94679e833f86c72be9913c87fa0147ab3aa783b1c19f15f9d76f05f028e1327.jpg)

<a id="paper-1820"></a>

## ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind


### Images

![0fedfd0e4e67f71df0cb40641a7f21627e4c6a29f1e55d6f4f7d0de76a3b3fe0.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/0fedfd0e4e67f71df0cb40641a7f21627e4c6a29f1e55d6f4f7d0de76a3b3fe0.jpg)

![107e6139949e43b0b09aa72d937d611766eaf5fad0ae44ef5a22d72935fdf20a.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/107e6139949e43b0b09aa72d937d611766eaf5fad0ae44ef5a22d72935fdf20a.jpg)

![15fafb6c3f879b3fa2a4462638841dd34c6c70ffd1a2c95dce71d42b1d493dae.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/15fafb6c3f879b3fa2a4462638841dd34c6c70ffd1a2c95dce71d42b1d493dae.jpg)

![23cf7aae0327af28036bc764cec07c9617349bbe6b3fd29cbbff50a142eb9a37.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/23cf7aae0327af28036bc764cec07c9617349bbe6b3fd29cbbff50a142eb9a37.jpg)

![440eb4ba8ba360573bed458b077defa8ac49d02e78b819eb90fb5d0390049840.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/440eb4ba8ba360573bed458b077defa8ac49d02e78b819eb90fb5d0390049840.jpg)

![53115137292d87d161c7df907f2c084fcca1a30362529af54eb81f98b97cac86.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/53115137292d87d161c7df907f2c084fcca1a30362529af54eb81f98b97cac86.jpg)

![8b998ae2a352482b04e30400f0704e0ff131775917bb861b9f386c07424a86b3.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/images/8b998ae2a352482b04e30400f0704e0ff131775917bb861b9f386c07424a86b3.jpg)

### Tables

![00a8ddc45127e1561b48a0e71e094b100d62b0c61f716521bb104941595d647d.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/00a8ddc45127e1561b48a0e71e094b100d62b0c61f716521bb104941595d647d.jpg)

![438b50c866a8e85cfe4453e75c4097ee3bce163b849c1e5f8bdd0030180974b5.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/438b50c866a8e85cfe4453e75c4097ee3bce163b849c1e5f8bdd0030180974b5.jpg)

![56388923d5d7d1eed747c5198dc09350eba22a533e367aae0c424d0a15d87a0c.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/56388923d5d7d1eed747c5198dc09350eba22a533e367aae0c424d0a15d87a0c.jpg)

![6646803675fa17e1d89061430660f468f03c44dfc938fc7b870c945c03265f75.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/6646803675fa17e1d89061430660f468f03c44dfc938fc7b870c945c03265f75.jpg)

![965bc2f7767fbb7742fede3ce5a40ba38f5311b39e1e8f3fb1d956ff235728cb.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/965bc2f7767fbb7742fede3ce5a40ba38f5311b39e1e8f3fb1d956ff235728cb.jpg)

![afdb9830b6aa2cc7add76f01a68b960e2efe4c1195fd404dae662a6c8a9fd6c9.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/afdb9830b6aa2cc7add76f01a68b960e2efe4c1195fd404dae662a6c8a9fd6c9.jpg)

![bc76afeea3797f3146f83e9503e2aff1a8f7a708a1a79520cd8b340d8507ade8.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/bc76afeea3797f3146f83e9503e2aff1a8f7a708a1a79520cd8b340d8507ade8.jpg)

![bedd07f0a0a4d2ca10a8b3ee80b7ee713e6ffcf3283f9b802198a43dd065ffed.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/bedd07f0a0a4d2ca10a8b3ee80b7ee713e6ffcf3283f9b802198a43dd065ffed.jpg)

![e3fc9732407a6d0ef62cee3c1858224ed85c48a8830a78414f5934ff8c2c57ed.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/e3fc9732407a6d0ef62cee3c1858224ed85c48a8830a78414f5934ff8c2c57ed.jpg)

![f26daefe6b83b62f75b59494b47c0753fe4a5aea198b26b95085515a8aff8c92.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/f26daefe6b83b62f75b59494b47c0753fe4a5aea198b26b95085515a8aff8c92.jpg)

![f2c84244b9bc56daabed55e64aa7ce33d6d2e5af3a9187b45a13fbd8d5012e6d.jpg](../icml_results/1843_ToMAP_%20Training%20Opponent-Aware%20LLM%20Persuaders%20with%20Theory%20of%20Mind/tables/f2c84244b9bc56daabed55e64aa7ce33d6d2e5af3a9187b45a13fbd8d5012e6d.jpg)

<a id="paper-1821"></a>

## GaussTrace: Provenance Analysis of 3D Gaussian Splatting Models with Evidence-based LLM Reasoning


### Images

![008cac71c4806f81abcd59b739cfb5760c2f415837a45f5e5b0d8668037adb8b.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/008cac71c4806f81abcd59b739cfb5760c2f415837a45f5e5b0d8668037adb8b.jpg)

![0b6b1eabababfa0e8a18db8b3c49300f3cc3aa96c78229af21bc7de9827b207f.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/0b6b1eabababfa0e8a18db8b3c49300f3cc3aa96c78229af21bc7de9827b207f.jpg)

![1fa4c6165c2de6af8f3c1a3f08a2226daa83b57a87e999ef092b8b481a1e03cc.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/1fa4c6165c2de6af8f3c1a3f08a2226daa83b57a87e999ef092b8b481a1e03cc.jpg)

![2485168b577193f0175d3055020a4bdf1a693a63270a012ff36407aaa8d7f0c6.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/2485168b577193f0175d3055020a4bdf1a693a63270a012ff36407aaa8d7f0c6.jpg)

![2cf0e76cecb0c2fdb0428bb4206af7a0d217ae3befddeb5eabeb5d6e32563242.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/2cf0e76cecb0c2fdb0428bb4206af7a0d217ae3befddeb5eabeb5d6e32563242.jpg)

![3f5b5a61d45900decb6582d44cc2ca7994034709e510a69b1e5f6b6276b7f834.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/3f5b5a61d45900decb6582d44cc2ca7994034709e510a69b1e5f6b6276b7f834.jpg)

![7cd4d32d0b4cd5b86b9408096d621978a147a4b6dab9b0e389b2c3a56b1c1624.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/7cd4d32d0b4cd5b86b9408096d621978a147a4b6dab9b0e389b2c3a56b1c1624.jpg)

![8798bb76832d6deef414fc252d3a1bfbd260031d618a313ba313c9cafa5fa044.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/8798bb76832d6deef414fc252d3a1bfbd260031d618a313ba313c9cafa5fa044.jpg)

![98fd6bba72b737a014c84565e9f8bfd3b3ef6a925901b57e2fc2ca985d236e65.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/98fd6bba72b737a014c84565e9f8bfd3b3ef6a925901b57e2fc2ca985d236e65.jpg)

![a9744ae069fbcce4d7a6a94a92ed8cd5a1072a6467ae3409cb50f7a99d0b590c.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/a9744ae069fbcce4d7a6a94a92ed8cd5a1072a6467ae3409cb50f7a99d0b590c.jpg)

![b2e18aba86ddc92886f2d6f47890931db0e62ce6f62d0a152b1db2d64ad643c0.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/b2e18aba86ddc92886f2d6f47890931db0e62ce6f62d0a152b1db2d64ad643c0.jpg)

![e01e0567f8212921a470a930a4ac06df3c7aa5f0ea6de15588fa2a607377f300.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/e01e0567f8212921a470a930a4ac06df3c7aa5f0ea6de15588fa2a607377f300.jpg)

![f44ddf3e35cca3497eac63811f65d6881a4383786e3f52837f7fa8d1d23bd98b.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/images/f44ddf3e35cca3497eac63811f65d6881a4383786e3f52837f7fa8d1d23bd98b.jpg)

### Tables

![23bfbd5566ef8d5e2642af983356ad0b74ab6ac7f3cdd452aa36e289a590a78e.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/tables/23bfbd5566ef8d5e2642af983356ad0b74ab6ac7f3cdd452aa36e289a590a78e.jpg)

![492ea9618f2ca864db261ba69635db5963aeb124f5e23145f9e4b918657ac4c3.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/tables/492ea9618f2ca864db261ba69635db5963aeb124f5e23145f9e4b918657ac4c3.jpg)

![c1f9c4c552770362e6d1e836d50ee4095f0d598a5ae2ec6a244a90e6a626b076.jpg](../icml_results/1844_GaussTrace_%20Provenance%20Analysis%20of%203D%20Gaussian%20Splatting%20Models%20with%20Evidence-based%20LLM%20Reasoning/tables/c1f9c4c552770362e6d1e836d50ee4095f0d598a5ae2ec6a244a90e6a626b076.jpg)

<a id="paper-1822"></a>

## MusicDET: Zero-Shot AI-Generated Music Detection


### Images

![1148009b44664e8729b6407d71d623677317321771b6821595dc3ef1cc7bd319.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/1148009b44664e8729b6407d71d623677317321771b6821595dc3ef1cc7bd319.jpg)

![456a1547c32049e2546b2ca26c5a0964233d64780d51bdcfcd17757ca13bce5c.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/456a1547c32049e2546b2ca26c5a0964233d64780d51bdcfcd17757ca13bce5c.jpg)

![54a0a206aa4e732966c8d46df68ced99f053f35740b7a88ab5d6a4ef5f25f940.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/54a0a206aa4e732966c8d46df68ced99f053f35740b7a88ab5d6a4ef5f25f940.jpg)

![6dfd9e1a1ac931c620f174fdadef91b935838d06250e63991b0c03e2a5952b60.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/6dfd9e1a1ac931c620f174fdadef91b935838d06250e63991b0c03e2a5952b60.jpg)

![ce6663551381e7a111dd2f33541612c11e44a7789c5edb8b63bc9b1ff1888fc0.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/ce6663551381e7a111dd2f33541612c11e44a7789c5edb8b63bc9b1ff1888fc0.jpg)

![e0dd305e96d85d7850148db9091819b06fae8f9c71fe1aa325590fb9b7e634d4.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/e0dd305e96d85d7850148db9091819b06fae8f9c71fe1aa325590fb9b7e634d4.jpg)

![e7a5ab11ad8150bd0c42a1dd20a7cb644496c7f7732dd2e9af64e79e7eb4cbe2.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/images/e7a5ab11ad8150bd0c42a1dd20a7cb644496c7f7732dd2e9af64e79e7eb4cbe2.jpg)

### Tables

![095f18a6d849456e83f825cb2964d31b19ea7456f2155850cdff575e7c7c9924.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/095f18a6d849456e83f825cb2964d31b19ea7456f2155850cdff575e7c7c9924.jpg)

![c2cdb3feda5004fd88c64eab5c086dd708b921a6bd3a72d77928fa90bf13411a.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/c2cdb3feda5004fd88c64eab5c086dd708b921a6bd3a72d77928fa90bf13411a.jpg)

![d7b5062a30e3868c2adebac01ab7dd9c71cc0b3872ccff6f4966bb78afeec7fb.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/d7b5062a30e3868c2adebac01ab7dd9c71cc0b3872ccff6f4966bb78afeec7fb.jpg)

![ea8a7a7713905aeb7304c77573882afee6a6abd63d6410311d98b1a5a8d4d50c.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/ea8a7a7713905aeb7304c77573882afee6a6abd63d6410311d98b1a5a8d4d50c.jpg)

![f79dd6ca916b87cd521bc0323a0832dc56164fc0891cab3a8ae49a5aeb21ad1d.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/f79dd6ca916b87cd521bc0323a0832dc56164fc0891cab3a8ae49a5aeb21ad1d.jpg)

![fd0d754d7f5edceb425be1d432fc886cd9912b7962cd9f46096a0b2fb085093a.jpg](../icml_results/1845_MusicDET_%20Zero-Shot%20AI-Generated%20Music%20Detection/tables/fd0d754d7f5edceb425be1d432fc886cd9912b7962cd9f46096a0b2fb085093a.jpg)

<a id="paper-1823"></a>

## Benchmarking Physics-Informed Time-Series Models for Operational Global Station Weather Forecasting


### Images

![1388d0e9f0dd4ab1df8aad924db7f29c8439e224dfc50789be2f429d54ac2331.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/1388d0e9f0dd4ab1df8aad924db7f29c8439e224dfc50789be2f429d54ac2331.jpg)

![2d655a0c2460fe38ac4039055379d71554cdc92ac7a0de3df211b8e8f96dae51.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/2d655a0c2460fe38ac4039055379d71554cdc92ac7a0de3df211b8e8f96dae51.jpg)

![3810b36f0d3c0ad58a80a1e03bbe27087bffa67a1b5988c177244f8b8f2335e6.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/3810b36f0d3c0ad58a80a1e03bbe27087bffa67a1b5988c177244f8b8f2335e6.jpg)

![4a125a6af4d0195e551edb0035975cd2bb43450c789d43b1c86301761f79e4e2.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/4a125a6af4d0195e551edb0035975cd2bb43450c789d43b1c86301761f79e4e2.jpg)

![4be14ebd747391d1de95090c86f1390e2568039d0dc3541d778816a60520df5c.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/4be14ebd747391d1de95090c86f1390e2568039d0dc3541d778816a60520df5c.jpg)

![547e45f84bcaa221629997792f4d401da60be4d74f32a076c04e8ce1f7607dca.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/547e45f84bcaa221629997792f4d401da60be4d74f32a076c04e8ce1f7607dca.jpg)

![5e62cc50729652817fb8dfc2d3999dd37543343dc8c67670d320cb28d0c1428e.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/5e62cc50729652817fb8dfc2d3999dd37543343dc8c67670d320cb28d0c1428e.jpg)

![6bdea6c550059da0e659cc7706cd3680d4f65795aa9abf14b39204180c4d6fbd.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/6bdea6c550059da0e659cc7706cd3680d4f65795aa9abf14b39204180c4d6fbd.jpg)

![6c81c00eca35b2c622c2dba960673cd5612182f275362e4d813a8495fc80f117.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/6c81c00eca35b2c622c2dba960673cd5612182f275362e4d813a8495fc80f117.jpg)

![73bc6b26877bf82086b08de45c03c4e5d87100be86f009539150775b43658696.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/73bc6b26877bf82086b08de45c03c4e5d87100be86f009539150775b43658696.jpg)

![77e1aca927d3bf993af90159b2b6ad672993f8fc16332d715238eb9ef5f1d672.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/77e1aca927d3bf993af90159b2b6ad672993f8fc16332d715238eb9ef5f1d672.jpg)

![7f635501abc7566f46c5272e4e6d9bd9104a0160531fb7cb1e240520202cb0c7.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/7f635501abc7566f46c5272e4e6d9bd9104a0160531fb7cb1e240520202cb0c7.jpg)

![87e622e48a817ad481a11466e4de9fb7c233634babb1413cd679265a1803744f.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/87e622e48a817ad481a11466e4de9fb7c233634babb1413cd679265a1803744f.jpg)

![95bb4d35ddbda5227ada5b21506ca302f4e634242741735e8f155e0dccd4b75c.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/95bb4d35ddbda5227ada5b21506ca302f4e634242741735e8f155e0dccd4b75c.jpg)

![a126aa7be0a25ce2bd6c19193ffff4f0cc00429611b9b4a7a8a4a5c62fc82a33.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/a126aa7be0a25ce2bd6c19193ffff4f0cc00429611b9b4a7a8a4a5c62fc82a33.jpg)

![a3c724b6219bb6c322f5d92d17649c673c2d088990f03f2cc85e8df8b1f8d5f4.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/a3c724b6219bb6c322f5d92d17649c673c2d088990f03f2cc85e8df8b1f8d5f4.jpg)

![a8f556fbe02a07a3956bd783c58ad1d5134772671635fbf67721ada198f38075.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/a8f556fbe02a07a3956bd783c58ad1d5134772671635fbf67721ada198f38075.jpg)

![b063b2120e3a0cb8bbbc2227e17c8a55c54718e7797e8dd7ca72325906de22bd.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/b063b2120e3a0cb8bbbc2227e17c8a55c54718e7797e8dd7ca72325906de22bd.jpg)

![b492612b53e6c56e736ba20d9a40e7721c679699220595ea3494a7ce212abcfc.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/b492612b53e6c56e736ba20d9a40e7721c679699220595ea3494a7ce212abcfc.jpg)

![c988f9bd55054fbae47de89790603780ee55be6c30224b67e0709b27e0c6124e.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/c988f9bd55054fbae47de89790603780ee55be6c30224b67e0709b27e0c6124e.jpg)

![cf5f8bdaba69c33517aed3ec9e20c6c5d9b6f36e1ee43746d3e28b6af59a507d.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/images/cf5f8bdaba69c33517aed3ec9e20c6c5d9b6f36e1ee43746d3e28b6af59a507d.jpg)

### Tables

![0e1f511c32d31d594ba7db684a296b26463f32b3bdb935dce31f7987f179ca5b.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/0e1f511c32d31d594ba7db684a296b26463f32b3bdb935dce31f7987f179ca5b.jpg)

![68098bd73bdafcfd4715e1d1656f1e31fc19d45b0781aa507cfa508973374686.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/68098bd73bdafcfd4715e1d1656f1e31fc19d45b0781aa507cfa508973374686.jpg)

![881e5de5aaaa81f9210d8cb0962b03acc253739705c2f84a61343ac3827c0168.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/881e5de5aaaa81f9210d8cb0962b03acc253739705c2f84a61343ac3827c0168.jpg)

![8f6e20264fae2a5d8ad515a5ab8fdaa17e0b00d9c3dc907fcd326ab62dd76843.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/8f6e20264fae2a5d8ad515a5ab8fdaa17e0b00d9c3dc907fcd326ab62dd76843.jpg)

![919c8acdbd8061b7974d115749067753db4fdfe2d598773096f3d466638c6510.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/919c8acdbd8061b7974d115749067753db4fdfe2d598773096f3d466638c6510.jpg)

![cb720915023246f87f5cffef34dd4e0c30545bd114fe5604dfadd4b280543886.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/cb720915023246f87f5cffef34dd4e0c30545bd114fe5604dfadd4b280543886.jpg)

![ccf685bd55a04f6e1200a60a9d8bfb3fe80538456dcb9e56a1c1834a69f5fbfb.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/ccf685bd55a04f6e1200a60a9d8bfb3fe80538456dcb9e56a1c1834a69f5fbfb.jpg)

![d6568ac0208d3365ae46b8933c6cfaf000aef87ffd806f2d5186cefb14b00112.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/d6568ac0208d3365ae46b8933c6cfaf000aef87ffd806f2d5186cefb14b00112.jpg)

![e89a6f73f7cf7484f1319b142ab59a4ac04d90a11ef4660e5d15cddaba788a42.jpg](../icml_results/1846_Benchmarking%20Physics-Informed%20Time-Series%20Models%20for%20Operational%20Global%20Station%20Weather%20Forecasting/tables/e89a6f73f7cf7484f1319b142ab59a4ac04d90a11ef4660e5d15cddaba788a42.jpg)

<a id="paper-1824"></a>

## Optimal Pricing for Data-Augmented AutoML Marketplaces


### Images

![17436e10e3fb08d3c03ff6c884908665a0cfa3e3198deaf0420b49bc70c53b31.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/17436e10e3fb08d3c03ff6c884908665a0cfa3e3198deaf0420b49bc70c53b31.jpg)

![558f8d5034ebb7dbb6aecb064eb69dd076d29cf6cb41c2804a17a6a61a84ff9e.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/558f8d5034ebb7dbb6aecb064eb69dd076d29cf6cb41c2804a17a6a61a84ff9e.jpg)

![73debed0bd16053f28c37afd9804db2d0e8074cd6164cf77536a31c62eaaf98e.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/73debed0bd16053f28c37afd9804db2d0e8074cd6164cf77536a31c62eaaf98e.jpg)

![a56f04e0515cb5e4b9a9f9998c5320b2f9143278619d1a14b329ad3ab58c7e40.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/a56f04e0515cb5e4b9a9f9998c5320b2f9143278619d1a14b329ad3ab58c7e40.jpg)

![bd256d76776b89d7f8a55f83242dce6115af52707baf361aade201b2dc336269.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/bd256d76776b89d7f8a55f83242dce6115af52707baf361aade201b2dc336269.jpg)

![c22770dfe9423be83aa0449affd769d01defeb5c15adbbed2b0651d52734ae01.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/c22770dfe9423be83aa0449affd769d01defeb5c15adbbed2b0651d52734ae01.jpg)

![e8a30ed2d0eff9bd4cb7a1df836f3543e57d56cedfb4be45f8ddf7968b48e177.jpg](../icml_results/1847_Optimal%20Pricing%20for%20Data-Augmented%20AutoML%20Marketplaces/images/e8a30ed2d0eff9bd4cb7a1df836f3543e57d56cedfb4be45f8ddf7968b48e177.jpg)

<a id="paper-1825"></a>

## $\textttMetaDistill$: Unlocking the Performance Ceiling for Pretrained Optimizers


### Images

![0511b72b32b928a102144923338fdba98f18c57851b30483a9446f07c3784fa1.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/0511b72b32b928a102144923338fdba98f18c57851b30483a9446f07c3784fa1.jpg)

![0a0df51bb6203a8b1a5336d502103f53773b46f3e5458584eb6d46f6da29eb74.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/0a0df51bb6203a8b1a5336d502103f53773b46f3e5458584eb6d46f6da29eb74.jpg)

![0f088216bb4c1384b06eb58498ac3255daea228441cbd195a299aca58b9fd4fc.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/0f088216bb4c1384b06eb58498ac3255daea228441cbd195a299aca58b9fd4fc.jpg)

![11da3a000511ea08589d8f4b20ae8d619f5e89979aacf79eeafa771017e82d83.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/11da3a000511ea08589d8f4b20ae8d619f5e89979aacf79eeafa771017e82d83.jpg)

![1310f7a6e55c3ea337a7894aee2fdbcd28fc4999723ef09acd25cd32f92280e1.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/1310f7a6e55c3ea337a7894aee2fdbcd28fc4999723ef09acd25cd32f92280e1.jpg)

![18074921eff5ff838ee50d42d54de75e6e5a58776d8a82bec649e6510a10c01c.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/18074921eff5ff838ee50d42d54de75e6e5a58776d8a82bec649e6510a10c01c.jpg)

![200be4a6167f57a9cb117d4a8e06654a1d0822b1e6a8e1dc20660a76219e2fdf.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/200be4a6167f57a9cb117d4a8e06654a1d0822b1e6a8e1dc20660a76219e2fdf.jpg)

![248bf709cf96f577783ae8dd2f0a9ec4b84badbd79ee21e00a0b6275a99f9982.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/248bf709cf96f577783ae8dd2f0a9ec4b84badbd79ee21e00a0b6275a99f9982.jpg)

![2829e5d16d5960973fac7a16c2d1d14d1c681ca26d9188d31c5970665aaa59f0.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/2829e5d16d5960973fac7a16c2d1d14d1c681ca26d9188d31c5970665aaa59f0.jpg)

![3384284247a488830f766e269ca2208feb1d1e1cd836a57b291155c13aac4957.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/3384284247a488830f766e269ca2208feb1d1e1cd836a57b291155c13aac4957.jpg)

![57a575c954969583c945677fd8ff5f1fe38a4df1c2704735849f5a80c47d85f7.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/57a575c954969583c945677fd8ff5f1fe38a4df1c2704735849f5a80c47d85f7.jpg)

![605c287309ea2c759420dd7ecbb762e5c77e8e39a74229e52b6dbf2e907c9684.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/605c287309ea2c759420dd7ecbb762e5c77e8e39a74229e52b6dbf2e907c9684.jpg)

![7839b4cd0dcb15028762c57f3d5ba81f88d37362660e5cf2b56c4c2a974df9ae.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/7839b4cd0dcb15028762c57f3d5ba81f88d37362660e5cf2b56c4c2a974df9ae.jpg)

![7e5a0560485251da3840f9be1f365648dc3505bce44a875a4fcef5d346a5b329.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/7e5a0560485251da3840f9be1f365648dc3505bce44a875a4fcef5d346a5b329.jpg)

![8047c3e7b6e77dbdc9f777a4155fa02cad12d7a6ae104f663ecbde7220985da4.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/8047c3e7b6e77dbdc9f777a4155fa02cad12d7a6ae104f663ecbde7220985da4.jpg)

![897b107c19ab10f6597817cf0c0e3688c0b0f4fcbd1b7afca66a360c89b6983f.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/897b107c19ab10f6597817cf0c0e3688c0b0f4fcbd1b7afca66a360c89b6983f.jpg)

![89d4d42d54b108360e313a0d3d03237bddc54d3a24942cc32e711b2f8f469c8e.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/89d4d42d54b108360e313a0d3d03237bddc54d3a24942cc32e711b2f8f469c8e.jpg)

![9df51cc4a21f91ad4911b0bdcf5bc699901be56900b26f06218c478f6d5abfdc.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/9df51cc4a21f91ad4911b0bdcf5bc699901be56900b26f06218c478f6d5abfdc.jpg)

![9fda85a1fb24b6698899e35e271292abdcf1a0d549e9a9259ec9d7295ce37fd9.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/9fda85a1fb24b6698899e35e271292abdcf1a0d549e9a9259ec9d7295ce37fd9.jpg)

![aa309827f39f6cf25995d858c80817a763bebb3649e2d2a6839dcf843249f7de.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/aa309827f39f6cf25995d858c80817a763bebb3649e2d2a6839dcf843249f7de.jpg)

![aed32ca5edcaf1b7cb91cca623b5764712cd561ff14b9736a36a93627a065e1e.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/aed32ca5edcaf1b7cb91cca623b5764712cd561ff14b9736a36a93627a065e1e.jpg)

![b19ec4eadfdf78e1a779aa5e03387426169f3bfa71d0221b7c564669a786af84.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/b19ec4eadfdf78e1a779aa5e03387426169f3bfa71d0221b7c564669a786af84.jpg)

![b829cd7a7fccdd5987bb6828be8d3f2312c984969e0401f1b9547596a2856471.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/b829cd7a7fccdd5987bb6828be8d3f2312c984969e0401f1b9547596a2856471.jpg)

![bf8a4ace7fc76fd3ea8b761301d7a152cfe8196d1869ff5ac705cccdd3a1e265.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/bf8a4ace7fc76fd3ea8b761301d7a152cfe8196d1869ff5ac705cccdd3a1e265.jpg)

![c24c7ddf6e548f44460dee9ed9568b12de4e2acb2468318cc9ee7f9a9702070c.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/c24c7ddf6e548f44460dee9ed9568b12de4e2acb2468318cc9ee7f9a9702070c.jpg)

![c96a36a82215e26fcc5c919c084e709472138d0e7c1e51e3717f26e73973395a.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/c96a36a82215e26fcc5c919c084e709472138d0e7c1e51e3717f26e73973395a.jpg)

![cc7c147172ff08484ad50990cedf3ec78eafc52dbf68f489b53a02540eab28b3.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/cc7c147172ff08484ad50990cedf3ec78eafc52dbf68f489b53a02540eab28b3.jpg)

![cdaefa7d6cea976e2bd0f708407b1024d3d3f194cdd5a27d18afc2708e85743e.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/cdaefa7d6cea976e2bd0f708407b1024d3d3f194cdd5a27d18afc2708e85743e.jpg)

![cf1b66defa051c4b11136f002f8683349e3c49dbb19c6626f0909b5cbc295d51.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/cf1b66defa051c4b11136f002f8683349e3c49dbb19c6626f0909b5cbc295d51.jpg)

![e3a95a782d70b795922532c835d2438b9b00678c6d5afd4bdc22e59558dcc143.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/e3a95a782d70b795922532c835d2438b9b00678c6d5afd4bdc22e59558dcc143.jpg)

![f04e88d3058f353af4ebf110086e934577f75ee103fb3761ab2c7e4d46f2b09f.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/f04e88d3058f353af4ebf110086e934577f75ee103fb3761ab2c7e4d46f2b09f.jpg)

![f6d0c39fba8322c0a2aa7f8abf8c3cf4420f0c6a8600638fa89a54f0703062cb.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/images/f6d0c39fba8322c0a2aa7f8abf8c3cf4420f0c6a8600638fa89a54f0703062cb.jpg)

### Tables

![0050c50ba0a5dbaa4b4c93b5bb8e7b6b2c93d6b04b8913b498c62de42b44d1f0.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/0050c50ba0a5dbaa4b4c93b5bb8e7b6b2c93d6b04b8913b498c62de42b44d1f0.jpg)

![16cd191f1fe851135fe00ef94ff85bc976d3ccaca3b2649df15c31987d2b1259.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/16cd191f1fe851135fe00ef94ff85bc976d3ccaca3b2649df15c31987d2b1259.jpg)

![1faa07e4f92583f1540b2de8909fc3fd3b5386e934ff88bcfaa1db13dc264619.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/1faa07e4f92583f1540b2de8909fc3fd3b5386e934ff88bcfaa1db13dc264619.jpg)

![343ec25a1bf775c73394c5d8bb77fa4bb06c161441cec51b6b38ab0c23a5a72a.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/343ec25a1bf775c73394c5d8bb77fa4bb06c161441cec51b6b38ab0c23a5a72a.jpg)

![4716e2c76f57e04ec69e3ed9ec9c34214d50c8f70040a325a68f91d1473c56b8.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/4716e2c76f57e04ec69e3ed9ec9c34214d50c8f70040a325a68f91d1473c56b8.jpg)

![4d244e9e049eca66cbc7ef9bddbeafe8e5aac3b1675f90a121256c3013489e1e.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/4d244e9e049eca66cbc7ef9bddbeafe8e5aac3b1675f90a121256c3013489e1e.jpg)

![6b79616bd4d5dd3eee7cee56070f811be674375010e3219e20654efe8dd11504.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/6b79616bd4d5dd3eee7cee56070f811be674375010e3219e20654efe8dd11504.jpg)

![6ce0e9e32d4484a4c8b183a9461f8f6755a5206291a4c70ef302ae062d0b209a.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/6ce0e9e32d4484a4c8b183a9461f8f6755a5206291a4c70ef302ae062d0b209a.jpg)

![9d5652843a66ec0eb24b1d46ae2ceaf863843eeeacc0241309ca161856afa922.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/9d5652843a66ec0eb24b1d46ae2ceaf863843eeeacc0241309ca161856afa922.jpg)

![a1b39ce4aab14443597cbf741c2f5543609a63fe9a2624d6a057368b7aedc20d.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/a1b39ce4aab14443597cbf741c2f5543609a63fe9a2624d6a057368b7aedc20d.jpg)

![a6f463d4817d4c7f7cf31fe36bf932ca78ac23c37a405d376a350d1015cb03a5.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/a6f463d4817d4c7f7cf31fe36bf932ca78ac23c37a405d376a350d1015cb03a5.jpg)

![bb73ea71760d98f165249c159edb933188c7b2feed0bb63cc03c0290e072ef8c.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/bb73ea71760d98f165249c159edb933188c7b2feed0bb63cc03c0290e072ef8c.jpg)

![c8e7ae1b8a1c3a1bd4d3b266d856ffdf47746b5b3a945f4e67b25d9fed01a5d7.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/c8e7ae1b8a1c3a1bd4d3b266d856ffdf47746b5b3a945f4e67b25d9fed01a5d7.jpg)

![ce60417d21ac8214eb8cda36113ec6560502a22c3d69f43259323d22e692dc64.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/ce60417d21ac8214eb8cda36113ec6560502a22c3d69f43259323d22e692dc64.jpg)

![d827a6add6b4905f76231f0da0fac1bc353c8ff5ccd4d2eadcc66f543d6d2d28.jpg](../icml_results/1848_%24_textttMetaDistill%24_%20Unlocking%20the%20Performance%20Ceiling%20for%20Pretrained%20Optimizers/tables/d827a6add6b4905f76231f0da0fac1bc353c8ff5ccd4d2eadcc66f543d6d2d28.jpg)

<a id="paper-1826"></a>

## Veda: Scalable Video Diffusion via Distilled Sparse Attention


### Images

![316162087a0baf4841277b1e5e16af6f7fdacc82ceb9c3b6fb5646d1112cf464.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/316162087a0baf4841277b1e5e16af6f7fdacc82ceb9c3b6fb5646d1112cf464.jpg)

![45d1f7fd282289df0b4ebe315987803a340e0f4737c95308231477d1e7cb2e36.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/45d1f7fd282289df0b4ebe315987803a340e0f4737c95308231477d1e7cb2e36.jpg)

![7d862aaba884c52add27ea05b280507b0893181fa061295ba7a8d6175c73028f.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/7d862aaba884c52add27ea05b280507b0893181fa061295ba7a8d6175c73028f.jpg)

![99779fffa6c7117fab14849e7dc14a6ff7585a7415d8190ef5e763f859bf5dfa.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/99779fffa6c7117fab14849e7dc14a6ff7585a7415d8190ef5e763f859bf5dfa.jpg)

![9d356e50f73361ef502c8880430c92c44c8274493086f456e26bdb24a323c80f.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/9d356e50f73361ef502c8880430c92c44c8274493086f456e26bdb24a323c80f.jpg)

![aee883d233e3bace688039b5c2ec40aac00c86b5b6298dab5718722c3ff475b9.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/aee883d233e3bace688039b5c2ec40aac00c86b5b6298dab5718722c3ff475b9.jpg)

![c4dcf342b5d900c416b50cfa0e3986fe6f81ea7d4230fbd5c15991bcb9426c92.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/c4dcf342b5d900c416b50cfa0e3986fe6f81ea7d4230fbd5c15991bcb9426c92.jpg)

![db2f7df811dece9a758a972ba1e608411b3c20c7871ff91e84616f7d98db7c91.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/db2f7df811dece9a758a972ba1e608411b3c20c7871ff91e84616f7d98db7c91.jpg)

![fdaca80ad4d69cdb9079ab92453c227f674467983db551beb5f22a8f3a9502b9.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/fdaca80ad4d69cdb9079ab92453c227f674467983db551beb5f22a8f3a9502b9.jpg)

![ffe33d0145edba3aeb2a3bdf457355e0c515efe46ae822602ccb579919f8f38a.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/images/ffe33d0145edba3aeb2a3bdf457355e0c515efe46ae822602ccb579919f8f38a.jpg)

### Tables

![0ca830bc67531b7ed3916533f05272213c00180514d1a2e90215339778bbe85a.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/0ca830bc67531b7ed3916533f05272213c00180514d1a2e90215339778bbe85a.jpg)

![47ac0adc873865e42096bc41bcc46e08d21385dfa6892c4e5ec1aa7e6ce35f54.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/47ac0adc873865e42096bc41bcc46e08d21385dfa6892c4e5ec1aa7e6ce35f54.jpg)

![5c9ace84ebc64fb3ead8e40c7b0c3611eda5d69452e9b715b8512ed9d0450473.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/5c9ace84ebc64fb3ead8e40c7b0c3611eda5d69452e9b715b8512ed9d0450473.jpg)

![67f8f6c13105f7deea71a721e8a1f9d5319e32de67e9c1440569cc68f41a42e3.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/67f8f6c13105f7deea71a721e8a1f9d5319e32de67e9c1440569cc68f41a42e3.jpg)

![9af52af447dbbb6eaa46a4a6f069d4a946f89294d6f8549dffeafb97de588069.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/9af52af447dbbb6eaa46a4a6f069d4a946f89294d6f8549dffeafb97de588069.jpg)

![ab025713587873bbd27231b8b4973c65473d9ed9c5791c80d4b6698a664cd714.jpg](../icml_results/1849_Veda_%20Scalable%20Video%20Diffusion%20via%20Distilled%20Sparse%20Attention/tables/ab025713587873bbd27231b8b4973c65473d9ed9c5791c80d4b6698a664cd714.jpg)

<a id="paper-1827"></a>

## Non-Uniform Noise-to-Signal Ratio in the REINFORCE Policy-Gradient Estimator


### Images

![1dc821341b7e9d893f2877a95653f5049e76fe2a30d54e1088511d42e8db4e6a.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/1dc821341b7e9d893f2877a95653f5049e76fe2a30d54e1088511d42e8db4e6a.jpg)

![487e0604cb75029647a35c28631c82eb414139f9cfecc28203777fbfb3afc958.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/487e0604cb75029647a35c28631c82eb414139f9cfecc28203777fbfb3afc958.jpg)

![49675c19a2b42c64f4c078db639dbbe1cee3e3885f97fd24bc1d27e375cad466.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/49675c19a2b42c64f4c078db639dbbe1cee3e3885f97fd24bc1d27e375cad466.jpg)

![519caffc758ad7541b200443a84bd0b8ca43171a3786ff35d1d531673ba282e5.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/519caffc758ad7541b200443a84bd0b8ca43171a3786ff35d1d531673ba282e5.jpg)

![65f57ee0fae9929e6abe9e9670fff02ae4dc8cee13e24a0435d3c5869a3ee71f.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/65f57ee0fae9929e6abe9e9670fff02ae4dc8cee13e24a0435d3c5869a3ee71f.jpg)

![a30aa3771df0b5701cd64630989f350baae42fcdd31a4c0be3a6b0ea31a2235e.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/a30aa3771df0b5701cd64630989f350baae42fcdd31a4c0be3a6b0ea31a2235e.jpg)

![c49d58cf99ec5dca0ecbd311212452a14dfb19b455802c75caaeca029e631f16.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/c49d58cf99ec5dca0ecbd311212452a14dfb19b455802c75caaeca029e631f16.jpg)

![c972b49eccc559a944a9bab34c9db76f5aac0467c16fbbbdd2a01040080bee27.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/c972b49eccc559a944a9bab34c9db76f5aac0467c16fbbbdd2a01040080bee27.jpg)

![eba687a6bf5157ece7595c8afc952f13cd8c9fca01efb71ab0f71fcbec790fd3.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/eba687a6bf5157ece7595c8afc952f13cd8c9fca01efb71ab0f71fcbec790fd3.jpg)

![feef8e7edcbf754e114a7550aac445c477552bd951803f6781f684f251c5da0e.jpg](../icml_results/1850_Non-Uniform%20Noise-to-Signal%20Ratio%20in%20the%20REINFORCE%20Policy-Gradient%20Estimator/images/feef8e7edcbf754e114a7550aac445c477552bd951803f6781f684f251c5da0e.jpg)

<a id="paper-1828"></a>

## Robust Multi-View Fusion via Prototype-Anchored Unbalanced Optimal Transport


### Images

![5d6492a9278a7f2b26db453ca7aec7cad83353762e87b564ab3a988ad5ad125a.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/images/5d6492a9278a7f2b26db453ca7aec7cad83353762e87b564ab3a988ad5ad125a.jpg)

![f8b96e0d26fedc8eb4f2b4f1f9a11fc96c97bc5506f6518c0228aa4a9e0996ce.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/images/f8b96e0d26fedc8eb4f2b4f1f9a11fc96c97bc5506f6518c0228aa4a9e0996ce.jpg)

### Tables

![0205374317b35a71ec5c25622eeeffbd3a38eb6ced55a29150b64a4270820f81.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/0205374317b35a71ec5c25622eeeffbd3a38eb6ced55a29150b64a4270820f81.jpg)

![046a4caa0e3bb53825c2822d989c2b818b8f386d33ffd419120ac4c69e9d2b40.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/046a4caa0e3bb53825c2822d989c2b818b8f386d33ffd419120ac4c69e9d2b40.jpg)

![2ec1ba3759d107c5defefa4c1b7ee25b24ed043eee862a7a99b8b3ad265cd9ec.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/2ec1ba3759d107c5defefa4c1b7ee25b24ed043eee862a7a99b8b3ad265cd9ec.jpg)

![6a3ba2cbc35fd37bc6b4632c5772243b9aa40790f56b023de4869588a7860795.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/6a3ba2cbc35fd37bc6b4632c5772243b9aa40790f56b023de4869588a7860795.jpg)

![94d853af54aa7e0a33e36bd4a4a99c9d8895d8d51aafa126273f3dd3b5233a3a.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/94d853af54aa7e0a33e36bd4a4a99c9d8895d8d51aafa126273f3dd3b5233a3a.jpg)

![fab81ff422a53ec9153ab3638a02a93890fcbeee8b88ca2f87e41d136339d1aa.jpg](../icml_results/1851_Robust%20Multi-View%20Fusion%20via%20Prototype-Anchored%20Unbalanced%20Optimal%20Transport/tables/fab81ff422a53ec9153ab3638a02a93890fcbeee8b88ca2f87e41d136339d1aa.jpg)

<a id="paper-1829"></a>

## Pose-ICL: 3D-Aware In-Context Learning for Pose-Controllable Subject Customization


### Images

![12a43a45dd88dffbee009897054483049e5e41179bef34d8224f022c46f314ca.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/12a43a45dd88dffbee009897054483049e5e41179bef34d8224f022c46f314ca.jpg)

![26788e6f40f95ae4cea069b502888510b8018eba76d9bfe9b83b0e0f8ed8491b.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/26788e6f40f95ae4cea069b502888510b8018eba76d9bfe9b83b0e0f8ed8491b.jpg)

![3ee19519fa6ac193e6eff97721f03bc10d6685a0a1aaa3dd91d119e729e86c1f.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/3ee19519fa6ac193e6eff97721f03bc10d6685a0a1aaa3dd91d119e729e86c1f.jpg)

![632f88c5405231e8f92ee7096864b61493afc3dcdf6f8d1f1753a35b53d842db.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/632f88c5405231e8f92ee7096864b61493afc3dcdf6f8d1f1753a35b53d842db.jpg)

![799b228473dacb69e86f64add852c04924694506d2066a9406cbfe306f4b89f4.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/799b228473dacb69e86f64add852c04924694506d2066a9406cbfe306f4b89f4.jpg)

![87ec5da7ac88e890e0dc9cbf9be8928bc2abbd92a558491cd01b4799a3e27520.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/87ec5da7ac88e890e0dc9cbf9be8928bc2abbd92a558491cd01b4799a3e27520.jpg)

![88c1ba783ba16c33f3679d9e0daf49133db877e9ee0e9ade46a02db3eba69a1f.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/88c1ba783ba16c33f3679d9e0daf49133db877e9ee0e9ade46a02db3eba69a1f.jpg)

![8f9babeb6d7388e94e80f83dcbdb7a9cfd6b4857b968c90b8a91137072012b94.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/8f9babeb6d7388e94e80f83dcbdb7a9cfd6b4857b968c90b8a91137072012b94.jpg)

![99ef54f4be5ab6458065291e26005cd2f78dd019ac68ac07a1369b8d6661d75e.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/99ef54f4be5ab6458065291e26005cd2f78dd019ac68ac07a1369b8d6661d75e.jpg)

![a842118c8867445adb8db64736dadfcfc8bc6d6e7ad0be34185d52001a9a39b0.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/a842118c8867445adb8db64736dadfcfc8bc6d6e7ad0be34185d52001a9a39b0.jpg)

![c42c15260608a96e16a33c35d70de97bd6d6fb1a7d22df4e2685c6cb14a45fce.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/c42c15260608a96e16a33c35d70de97bd6d6fb1a7d22df4e2685c6cb14a45fce.jpg)

![c86555b9d48bc1e656d61e8ea603c33d669a346cadb34d768d56121f18686aea.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/c86555b9d48bc1e656d61e8ea603c33d669a346cadb34d768d56121f18686aea.jpg)

![d08f3c63a26a7784f99226de11f6577c34c972b2dcba27035ce7c5cf0b477e06.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/d08f3c63a26a7784f99226de11f6577c34c972b2dcba27035ce7c5cf0b477e06.jpg)

![e4522787ac698911eb18193cbd0b31d9e8e318ad9833b06803848f7a69301d64.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/e4522787ac698911eb18193cbd0b31d9e8e318ad9833b06803848f7a69301d64.jpg)

![e8a957f5ddd9b1efbbdc49e7359e0f611dd42fe3bf8bc073a371cd171133deaa.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/images/e8a957f5ddd9b1efbbdc49e7359e0f611dd42fe3bf8bc073a371cd171133deaa.jpg)

### Tables

![358bbe053720ad0bd10fc772d743a066272acca65576e0b5aa5a8da1beb7a85a.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/tables/358bbe053720ad0bd10fc772d743a066272acca65576e0b5aa5a8da1beb7a85a.jpg)

![43e0249ba6149aa8ebb6e5de0d2ec1e12323888a4c795da03e3bc10550286d92.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/tables/43e0249ba6149aa8ebb6e5de0d2ec1e12323888a4c795da03e3bc10550286d92.jpg)

![cabea5327c197f124cd4f6748e0c1e80d9997f72e8fb028858e4549a6e5e4876.jpg](../icml_results/1852_Pose-ICL_%203D-Aware%20In-Context%20Learning%20for%20Pose-Controllable%20Subject%20Customization/tables/cabea5327c197f124cd4f6748e0c1e80d9997f72e8fb028858e4549a6e5e4876.jpg)

<a id="paper-1830"></a>

## Omni-Perception Policy Optimization for Multimodal Emotion Reasoning


### Images

![168db60945748a13c2045d70fd47948de32f14eac07adcd56715c50facd6a4c4.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/images/168db60945748a13c2045d70fd47948de32f14eac07adcd56715c50facd6a4c4.jpg)

![465d9c06199de9ec9845ea5771a03e2f919db11cd6d704801705b247ceeb00ff.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/images/465d9c06199de9ec9845ea5771a03e2f919db11cd6d704801705b247ceeb00ff.jpg)

![806ca45dc9f02f431e769f53ad6e15bbcdfd0d0cb59a6787d03610b6a21039d1.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/images/806ca45dc9f02f431e769f53ad6e15bbcdfd0d0cb59a6787d03610b6a21039d1.jpg)

![cdaab8a5e3257b007d268968c7d3f57e204b573d7d61ec56d0732826f383d468.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/images/cdaab8a5e3257b007d268968c7d3f57e204b573d7d61ec56d0732826f383d468.jpg)

![fd61ce85f1732986ab4ac11b0814aa58012bd59038ccf8a4022c9db0baf5bb43.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/images/fd61ce85f1732986ab4ac11b0814aa58012bd59038ccf8a4022c9db0baf5bb43.jpg)

### Tables

![03edec20de11c561606abb06862725ae1efb046f03a109c563d8b9d18cf56946.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/03edec20de11c561606abb06862725ae1efb046f03a109c563d8b9d18cf56946.jpg)

![0f00aab915047946a6cc5746d9ad8ab7b0ad8fc0a92080a1d7e05fab32c95ca5.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/0f00aab915047946a6cc5746d9ad8ab7b0ad8fc0a92080a1d7e05fab32c95ca5.jpg)

![1b76b80330c7ca982e39f502fdfbd4a3e51fea17aaf54d79add74dadcd6be235.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/1b76b80330c7ca982e39f502fdfbd4a3e51fea17aaf54d79add74dadcd6be235.jpg)

![2218a289f01b8defcedb53942a5be6984b5e84059c478d3f698da852892e834c.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/2218a289f01b8defcedb53942a5be6984b5e84059c478d3f698da852892e834c.jpg)

![2bd87df62f5e963f8a9324d2d744288c6ade539db7fc3eadc23710a89dcc581e.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/2bd87df62f5e963f8a9324d2d744288c6ade539db7fc3eadc23710a89dcc581e.jpg)

![4310126bcc3c1cd063b014ec3eb79899cd975ce6c238718a6bb0034cf4ee402c.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/4310126bcc3c1cd063b014ec3eb79899cd975ce6c238718a6bb0034cf4ee402c.jpg)

![4819525040e84c4d75e24fec072914b5376a12e33c98b2ef958e53cbd5ba789d.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/4819525040e84c4d75e24fec072914b5376a12e33c98b2ef958e53cbd5ba789d.jpg)

![4bfb9cafb4c2083d85c08b1e1a15236a06893bb4ab57d19db168942bc97d88ef.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/4bfb9cafb4c2083d85c08b1e1a15236a06893bb4ab57d19db168942bc97d88ef.jpg)

![6e8e3af4150df076c71e99b73f2f51a571f4c60a1dd37e4f421af82d26d6eeb4.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/6e8e3af4150df076c71e99b73f2f51a571f4c60a1dd37e4f421af82d26d6eeb4.jpg)

![6eb7d88640814853656703c04cc4be80ab379975d0f4b12454e03869465ba5f0.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/6eb7d88640814853656703c04cc4be80ab379975d0f4b12454e03869465ba5f0.jpg)

![73018caeb13c83e5cf271f4e1d50086981451a7e79ebcda5b923ccbb437ed720.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/73018caeb13c83e5cf271f4e1d50086981451a7e79ebcda5b923ccbb437ed720.jpg)

![9737edc7bdbf718f079c3257cb106c8c68e084a0fc1c59ab7de5dbb53503bd1f.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/9737edc7bdbf718f079c3257cb106c8c68e084a0fc1c59ab7de5dbb53503bd1f.jpg)

![9f220f78f229f3d7dfefe6cb942642fa30c9b5bd06a7d49387d3b7f1741f6cb2.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/9f220f78f229f3d7dfefe6cb942642fa30c9b5bd06a7d49387d3b7f1741f6cb2.jpg)

![aaa9af3a93f03d1759031680fe68b09ffde7c34f9f835e7004f70e138893f358.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/aaa9af3a93f03d1759031680fe68b09ffde7c34f9f835e7004f70e138893f358.jpg)

![b0a0d70eb2451323fab5c75e51041d67357455deea8b3248fabfc607efb4fd3b.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/b0a0d70eb2451323fab5c75e51041d67357455deea8b3248fabfc607efb4fd3b.jpg)

![c666dbbffa2c5363cab8ac3f6c37b27e8e6c359b996547690411a820d3d661b9.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/c666dbbffa2c5363cab8ac3f6c37b27e8e6c359b996547690411a820d3d661b9.jpg)

![c6783ab6a7391791d2e1c0ab5560b0917718bd2fbaa68a5016fab57d5bf73a67.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/c6783ab6a7391791d2e1c0ab5560b0917718bd2fbaa68a5016fab57d5bf73a67.jpg)

![e2362066c22e85d4bd1c21968bb761f01a4877bb899b5ce178f0db59d8369908.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/e2362066c22e85d4bd1c21968bb761f01a4877bb899b5ce178f0db59d8369908.jpg)

![e26490a44a44da57e5aeb0613e0caa0427a9b739411b1c0553829a003313e7ab.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/e26490a44a44da57e5aeb0613e0caa0427a9b739411b1c0553829a003313e7ab.jpg)

![eb1bd8fa183d71238b2bbbbe2d530056f5ee783c0f2f47f69295d6e7c7956029.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/eb1bd8fa183d71238b2bbbbe2d530056f5ee783c0f2f47f69295d6e7c7956029.jpg)

![f287a15155e9e1e73bef678bd535ce69fd474b1db272cdf57f59fc6876495d4f.jpg](../icml_results/1853_Omni-Perception%20Policy%20Optimization%20for%20Multimodal%20Emotion%20Reasoning/tables/f287a15155e9e1e73bef678bd535ce69fd474b1db272cdf57f59fc6876495d4f.jpg)

<a id="paper-1831"></a>

## The Entropic Signature of Class Speciation in Diffusion Models


### Images

![03d36574976d4dd324cb07c7f3847d1cd54eaf6d8a29e12a018f29fe707afc98.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/03d36574976d4dd324cb07c7f3847d1cd54eaf6d8a29e12a018f29fe707afc98.jpg)

![128e8854f5d608bd7a6294e10098543057d43a814d590add3eb680a6344445ad.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/128e8854f5d608bd7a6294e10098543057d43a814d590add3eb680a6344445ad.jpg)

![16c598c692721030563c2f40f2ff28f558d7506e55f133caacb629e874391524.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/16c598c692721030563c2f40f2ff28f558d7506e55f133caacb629e874391524.jpg)

![30a1195c0c6697306afa08584e7f848b58b2a8e13b767a2418e5355856064272.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/30a1195c0c6697306afa08584e7f848b58b2a8e13b767a2418e5355856064272.jpg)

![4a7c049399288e596fd54c0ea7e0ccae7768daa7b32335521c8c7404b12a5bc9.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/4a7c049399288e596fd54c0ea7e0ccae7768daa7b32335521c8c7404b12a5bc9.jpg)

![755a86fdfa013bbfc0f019875d095b3faf899df854f27ae76c476d45a29bd2e0.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/755a86fdfa013bbfc0f019875d095b3faf899df854f27ae76c476d45a29bd2e0.jpg)

![9ace4c466449d146422abd7c62adcc0177d7a92cb27afd7468ade47943d7b27f.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/9ace4c466449d146422abd7c62adcc0177d7a92cb27afd7468ade47943d7b27f.jpg)

![a58e248847ab0185b14fe403c1016f004d5e6e0e7e37a882e9e83d20af77de2d.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/a58e248847ab0185b14fe403c1016f004d5e6e0e7e37a882e9e83d20af77de2d.jpg)

![a680537fc6970616024ac4d5a76e0f8d672c7b8d303c58d7633cc63a3278061a.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/a680537fc6970616024ac4d5a76e0f8d672c7b8d303c58d7633cc63a3278061a.jpg)

![a74b6955c5a5969e796ce849977833e50f378ad41197d4fc19df46105ed023f3.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/a74b6955c5a5969e796ce849977833e50f378ad41197d4fc19df46105ed023f3.jpg)

![b9d9b7e929777a40c2a0ed95674e760ed53bcbba9e82a1f0eb08d92ea1349383.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/b9d9b7e929777a40c2a0ed95674e760ed53bcbba9e82a1f0eb08d92ea1349383.jpg)

![bd49158bb8b156653fc924235c45c1f05dc7d808598585ffd7f6677e19e25d57.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/bd49158bb8b156653fc924235c45c1f05dc7d808598585ffd7f6677e19e25d57.jpg)

![be1065f9f490cc3b0938dcae77ef1690261edd27bd58ef2c8a4859acbbfb9248.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/be1065f9f490cc3b0938dcae77ef1690261edd27bd58ef2c8a4859acbbfb9248.jpg)

![c1569e4b3bd6d0070ff3b81a5ea625429a93d9fb9d462b77a64881ed334d75b0.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/c1569e4b3bd6d0070ff3b81a5ea625429a93d9fb9d462b77a64881ed334d75b0.jpg)

![c8f02841fc1a998f1d1550cfc83386df3ad2e546ea108d46757043cf82d88618.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/c8f02841fc1a998f1d1550cfc83386df3ad2e546ea108d46757043cf82d88618.jpg)

![cc74d06839110312c3578538a3853272c021c81f3e67fd2731188b9ecadddeb9.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/cc74d06839110312c3578538a3853272c021c81f3e67fd2731188b9ecadddeb9.jpg)

![ceb7bbf03a76cf020560403dee0b56ac5587e1b4ed1b0671c9c0513904def15e.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/ceb7bbf03a76cf020560403dee0b56ac5587e1b4ed1b0671c9c0513904def15e.jpg)

![f94a6429b683566424d04840a89f97f0c1d5e6facfc2a239b333689bff8442f6.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/f94a6429b683566424d04840a89f97f0c1d5e6facfc2a239b333689bff8442f6.jpg)

![fdb55e9f84b95091b4d7549addd050bbae35074e9c15fc0bb388d1078959e860.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/images/fdb55e9f84b95091b4d7549addd050bbae35074e9c15fc0bb388d1078959e860.jpg)

### Tables

![0225d5844e847e8ba6fd1fab037df367d8b6af890d8a6777f7c2383d653f9607.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/tables/0225d5844e847e8ba6fd1fab037df367d8b6af890d8a6777f7c2383d653f9607.jpg)

![95d71c742c7d0e5d7980551065bdd5c4f5d52a131fd50e12af7529d7a99fe193.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/tables/95d71c742c7d0e5d7980551065bdd5c4f5d52a131fd50e12af7529d7a99fe193.jpg)

![b7d246ad25ff535c50ee71dfe7f5027dee62a134627aad7a6b11931b46376fdd.jpg](../icml_results/1854_The%20Entropic%20Signature%20of%20Class%20Speciation%20in%20Diffusion%20Models/tables/b7d246ad25ff535c50ee71dfe7f5027dee62a134627aad7a6b11931b46376fdd.jpg)

<a id="paper-1832"></a>

## Universal Multiclass Transductive Online Learning


### Images

![3fa71a369d3f93bacbe7d864d71e542df3f13f5b41ae104cb7d829e6fa134380.jpg](../icml_results/1856_Universal%20Multiclass%20Transductive%20Online%20Learning/images/3fa71a369d3f93bacbe7d864d71e542df3f13f5b41ae104cb7d829e6fa134380.jpg)

![79009abc2aadbe27e40e1b0c109cb09ec8efa6c0d7b9d8ec62474dfcd4a7e30a.jpg](../icml_results/1856_Universal%20Multiclass%20Transductive%20Online%20Learning/images/79009abc2aadbe27e40e1b0c109cb09ec8efa6c0d7b9d8ec62474dfcd4a7e30a.jpg)

### Tables

![252a1ba9537cbd90ae1cd47ac2b65ff8caf623ff356f9661a093fa09e2a33c0e.jpg](../icml_results/1856_Universal%20Multiclass%20Transductive%20Online%20Learning/tables/252a1ba9537cbd90ae1cd47ac2b65ff8caf623ff356f9661a093fa09e2a33c0e.jpg)

![c245a3feed6500b022410b8216c7b48484b321f899cc8960d9986238f7484efb.jpg](../icml_results/1856_Universal%20Multiclass%20Transductive%20Online%20Learning/tables/c245a3feed6500b022410b8216c7b48484b321f899cc8960d9986238f7484efb.jpg)

![efa03fc6b06a9eb85ebe8ec63240fbe9290634509e1a2643655be0d21f0d8e6c.jpg](../icml_results/1856_Universal%20Multiclass%20Transductive%20Online%20Learning/tables/efa03fc6b06a9eb85ebe8ec63240fbe9290634509e1a2643655be0d21f0d8e6c.jpg)

<a id="paper-1833"></a>

## ViTok-v2: Scaling Native Resolution Autoencoders to 5 Billion Parameters


### Images

![069361ddaf9b2d038d72523d83412d461c0193d0f8d2a0dc3e85fccc2a7b3311.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/069361ddaf9b2d038d72523d83412d461c0193d0f8d2a0dc3e85fccc2a7b3311.jpg)

![2d63c474951353e87d1887c4708b1b2cc1870e2017ac5b4549935f0189beafab.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/2d63c474951353e87d1887c4708b1b2cc1870e2017ac5b4549935f0189beafab.jpg)

![5b54f428bfcb397455e89b4215cf2a8a6c4bd3cd280c1faebd8f30a8bef44a67.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/5b54f428bfcb397455e89b4215cf2a8a6c4bd3cd280c1faebd8f30a8bef44a67.jpg)

![67f7f2ecf7fb3685b3810e17bd7c3ebe108d49e54199b31d8e63dd548663b6fd.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/67f7f2ecf7fb3685b3810e17bd7c3ebe108d49e54199b31d8e63dd548663b6fd.jpg)

![6a8453df347185113e9374e53f48a8f5f5b772a83ae3f6aab637dc1d63fd6eb8.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/6a8453df347185113e9374e53f48a8f5f5b772a83ae3f6aab637dc1d63fd6eb8.jpg)

![84f7ec3296e8d6db5a907c71de08891958ff0eda95007b6d1268539f5138bf97.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/84f7ec3296e8d6db5a907c71de08891958ff0eda95007b6d1268539f5138bf97.jpg)

![883b55d6e94f6c87e2952df0c9a22a5f9db0be563695d920052a9b185778302b.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/images/883b55d6e94f6c87e2952df0c9a22a5f9db0be563695d920052a9b185778302b.jpg)

### Tables

![04059b06344c2c63e94f6b59a53db990ede7e559a7356691f63d880dfd1766dc.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/04059b06344c2c63e94f6b59a53db990ede7e559a7356691f63d880dfd1766dc.jpg)

![17324da227879e11764bf6c898672e17d72778e58016ebe74712f8e93454734d.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/17324da227879e11764bf6c898672e17d72778e58016ebe74712f8e93454734d.jpg)

![1f6c9099309cf7c2d944f1f37518a2fd951ac67e3a82701391951b0eca2a2108.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/1f6c9099309cf7c2d944f1f37518a2fd951ac67e3a82701391951b0eca2a2108.jpg)

![44cee79f3100a777b4bae0f6c53f4589f7a8a3126fd20a51a4d9507c552f2025.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/44cee79f3100a777b4bae0f6c53f4589f7a8a3126fd20a51a4d9507c552f2025.jpg)

![8c8fe05c4e785ef52845e41e60399ffc828bbb09d28092ce2991549013438cd6.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/8c8fe05c4e785ef52845e41e60399ffc828bbb09d28092ce2991549013438cd6.jpg)

![aca98dfe446a1fc2883ebb6e06ef85e04ae9b11822b20ce2f94bb3ea0e00f940.jpg](../icml_results/1858_ViTok-v2_%20Scaling%20Native%20Resolution%20Autoencoders%20to%205%20Billion%20Parameters/tables/aca98dfe446a1fc2883ebb6e06ef85e04ae9b11822b20ce2f94bb3ea0e00f940.jpg)

<a id="paper-1834"></a>

## SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity


### Images

![37817a5461356f7af17dc4d9fda7767fe405037bff69c0460c71123d4cf9c27a.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/37817a5461356f7af17dc4d9fda7767fe405037bff69c0460c71123d4cf9c27a.jpg)

![37c3b2c607ca998b7742883fc6fe96ffaa5ddbad8a4bd655b49ebec55e7de899.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/37c3b2c607ca998b7742883fc6fe96ffaa5ddbad8a4bd655b49ebec55e7de899.jpg)

![4f9a12087f6c72e25934db4e76ed23edbb2f555d835acd28133d70daab088421.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/4f9a12087f6c72e25934db4e76ed23edbb2f555d835acd28133d70daab088421.jpg)

![6f4e8609d25ab1ccd97f70792718ca1e95b580ed07f350b2a1781f34cda3b72a.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/6f4e8609d25ab1ccd97f70792718ca1e95b580ed07f350b2a1781f34cda3b72a.jpg)

![7071ab0a36075908036a3986b765de22397b6bc46fc211622e1a2e605551da07.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/7071ab0a36075908036a3986b765de22397b6bc46fc211622e1a2e605551da07.jpg)

![8eb7f545daef3b0a2354f18f754f7e5e90807f33e4a1d184247fb14aac24ffd9.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/8eb7f545daef3b0a2354f18f754f7e5e90807f33e4a1d184247fb14aac24ffd9.jpg)

![a03806daf69838304fe0ce2b218c9812ab31e5079790d85652aeff79b4d26e6b.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/a03806daf69838304fe0ce2b218c9812ab31e5079790d85652aeff79b4d26e6b.jpg)

![b3bb2c73cb56714b69022b933a811e4693828c763aee01af4175c489469268d6.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/b3bb2c73cb56714b69022b933a811e4693828c763aee01af4175c489469268d6.jpg)

![d14118fdad99f7d43d478479bc679a51446d1971624a40dd31ba9c7b648eea17.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/d14118fdad99f7d43d478479bc679a51446d1971624a40dd31ba9c7b648eea17.jpg)

![d9988c8f0d8d66b7c3ad96b98cd437ad2fa45635c33ebb9124a25308885fef36.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/images/d9988c8f0d8d66b7c3ad96b98cd437ad2fa45635c33ebb9124a25308885fef36.jpg)

### Tables

![062a807b7634815aaac2e0510aa7854f191cd4c5d6610797cf93a73133785e26.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/062a807b7634815aaac2e0510aa7854f191cd4c5d6610797cf93a73133785e26.jpg)

![122ad6c0ec164f6d6256cbf7e01b510e8c9db567db87d085954da10dc0878290.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/122ad6c0ec164f6d6256cbf7e01b510e8c9db567db87d085954da10dc0878290.jpg)

![18103c5f64bed0f40232e80ccaa5070014be9b205a196a84738e700ce6ff4d61.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/18103c5f64bed0f40232e80ccaa5070014be9b205a196a84738e700ce6ff4d61.jpg)

![1f67badf58da8b30fc9446390e80362bc5ae8f4dc0f66681cbfc6480271697dc.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/1f67badf58da8b30fc9446390e80362bc5ae8f4dc0f66681cbfc6480271697dc.jpg)

![1f95223cf10ed29b431591d022539ee272e47e2405e12951eb42cc7f9c95a8df.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/1f95223cf10ed29b431591d022539ee272e47e2405e12951eb42cc7f9c95a8df.jpg)

![219bdf9bcd196bd1fc97f45c5c8ebb2cb6c9e64fc082eb5b8211a834f300badb.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/219bdf9bcd196bd1fc97f45c5c8ebb2cb6c9e64fc082eb5b8211a834f300badb.jpg)

![326b7a5a7a70539e120416b38d929b7c90b0cd2ed9ffbed835809b4898dce972.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/326b7a5a7a70539e120416b38d929b7c90b0cd2ed9ffbed835809b4898dce972.jpg)

![3272c1306a7750101772835abd4dd9025bd4bc1fa4e4415f40d041e5af27b342.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/3272c1306a7750101772835abd4dd9025bd4bc1fa4e4415f40d041e5af27b342.jpg)

![32e44f51e802367229a40d0582761f42922847e2ce74649a07445ae9ce155c72.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/32e44f51e802367229a40d0582761f42922847e2ce74649a07445ae9ce155c72.jpg)

![34160882688cf290fa140b52661aabf62350a5252bb496fcdedb011c2bd29ac4.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/34160882688cf290fa140b52661aabf62350a5252bb496fcdedb011c2bd29ac4.jpg)

![3b785a8c4362852777d6654fa9b719d24ac389d7b19dc49868031084b94d35f4.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/3b785a8c4362852777d6654fa9b719d24ac389d7b19dc49868031084b94d35f4.jpg)

![434eafbd8ccd35db302593480b0cdc6457e18df06c8b8119f6dee25305f1f32a.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/434eafbd8ccd35db302593480b0cdc6457e18df06c8b8119f6dee25305f1f32a.jpg)

![440ea6958e38b15092e3adbb0e38838b0521cbf87ffd615b8028cb71c209465a.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/440ea6958e38b15092e3adbb0e38838b0521cbf87ffd615b8028cb71c209465a.jpg)

![4959b9ef85989c1822d8dbf479dfbd13bddf7389e830a528a6396428bf91379f.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/4959b9ef85989c1822d8dbf479dfbd13bddf7389e830a528a6396428bf91379f.jpg)

![4bfb4496ef52a5e280f890863198fff30bc310651d466190b3e917279d8366a9.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/4bfb4496ef52a5e280f890863198fff30bc310651d466190b3e917279d8366a9.jpg)

![4cfa23ba7c20c652dd88c05ea6f8bfe8d2ea8f43df35c4b3fb0a34361efb9719.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/4cfa23ba7c20c652dd88c05ea6f8bfe8d2ea8f43df35c4b3fb0a34361efb9719.jpg)

![546dc9612465cf7a19fe8a66e4c52ea16cc988c98e9b8de946677f125b9ec6cb.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/546dc9612465cf7a19fe8a66e4c52ea16cc988c98e9b8de946677f125b9ec6cb.jpg)

![5da60115ab438a4df0e6f1f9e6ad913cfcaa479f68a9115041d25d793fb082d8.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/5da60115ab438a4df0e6f1f9e6ad913cfcaa479f68a9115041d25d793fb082d8.jpg)

![5f77642cd6e7e8619482dd0723d1cf72c95f26aba82da3cbf252d4ea571599cc.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/5f77642cd6e7e8619482dd0723d1cf72c95f26aba82da3cbf252d4ea571599cc.jpg)

![64b3f939aefd3dee8be27f9d75f79023c9691bd8e87c4b25140d3a2eed481d99.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/64b3f939aefd3dee8be27f9d75f79023c9691bd8e87c4b25140d3a2eed481d99.jpg)

![67c156f9031a53e3cfb84a5982e96ff5db6a83e5bf58b5b5aafff9b5b26e84a2.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/67c156f9031a53e3cfb84a5982e96ff5db6a83e5bf58b5b5aafff9b5b26e84a2.jpg)

![6b413908f812b5d87fe432bb05eec969230bf5462134b487eb17f84b7dc53456.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/6b413908f812b5d87fe432bb05eec969230bf5462134b487eb17f84b7dc53456.jpg)

![79ff09ca1b6e6b70df9b3c0438c2005975c6ed94c209fe848d87498c77639138.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/79ff09ca1b6e6b70df9b3c0438c2005975c6ed94c209fe848d87498c77639138.jpg)

![7b2c3158c8cc3d8da7e9f28bb43269ede069d685936b581bef9dc98eb1c60471.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/7b2c3158c8cc3d8da7e9f28bb43269ede069d685936b581bef9dc98eb1c60471.jpg)

![8ceb3e04c4dc5edc66ede0d686d62f92df421088a5c95cf47b1e27a390298bd5.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/8ceb3e04c4dc5edc66ede0d686d62f92df421088a5c95cf47b1e27a390298bd5.jpg)

![8dc0fc1d834fce9c5a6185c6d2327bd0faa4b36830eed8fdabc172bfaafd349b.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/8dc0fc1d834fce9c5a6185c6d2327bd0faa4b36830eed8fdabc172bfaafd349b.jpg)

![9c7795a600e92594e241e64aa6429cbefb17e20f02db7583411d86104620ef41.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/9c7795a600e92594e241e64aa6429cbefb17e20f02db7583411d86104620ef41.jpg)

![9ffcf0917ac2f555b2506db102ab51663c55124b7be8b99189b3976501051c38.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/9ffcf0917ac2f555b2506db102ab51663c55124b7be8b99189b3976501051c38.jpg)

![a2c439c9af2afdbcfce83d7224721f26402205ccf5681af7575ad9bc24c07fe9.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/a2c439c9af2afdbcfce83d7224721f26402205ccf5681af7575ad9bc24c07fe9.jpg)

![a5bc0dc956e9b520342af5778464fc55c78a50907ef853cf7d6a3daea6f09938.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/a5bc0dc956e9b520342af5778464fc55c78a50907ef853cf7d6a3daea6f09938.jpg)

![aac438994b3a355384dbae41e67e78091a0d0415be2e0195fb9ff15fbd651df7.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/aac438994b3a355384dbae41e67e78091a0d0415be2e0195fb9ff15fbd651df7.jpg)

![ac8355a1290968c0d8607bfb170aa4f1e2f1c4612b7ebfd64aaabcb718c4ae2d.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/ac8355a1290968c0d8607bfb170aa4f1e2f1c4612b7ebfd64aaabcb718c4ae2d.jpg)

![addb5b4b3c66831359592292671ad32552e1b92aad1814e3c80e0c63feab9812.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/addb5b4b3c66831359592292671ad32552e1b92aad1814e3c80e0c63feab9812.jpg)

![b1e6dd998af6eba36baf58dd9fd738b5eb359b8ffb46a2a7f23e774b0b43cb3d.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/b1e6dd998af6eba36baf58dd9fd738b5eb359b8ffb46a2a7f23e774b0b43cb3d.jpg)

![b6b8ace524fa5195228ecf3bcb7626c9c4bbd47fafea5aea279872c659d6870d.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/b6b8ace524fa5195228ecf3bcb7626c9c4bbd47fafea5aea279872c659d6870d.jpg)

![c1b81e382b4a4949a7510e72a495bffcbf023e417722dfc2037659b593fccfa5.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/c1b81e382b4a4949a7510e72a495bffcbf023e417722dfc2037659b593fccfa5.jpg)

![d0738d17af4b9cf139926cced3eee89bf8f19a5c099f69c5f3f6e546486a5dc5.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/d0738d17af4b9cf139926cced3eee89bf8f19a5c099f69c5f3f6e546486a5dc5.jpg)

![d729783a0b1eca08be3316302b134cd73085c194270aa2d537dc9f2515b1e8c9.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/d729783a0b1eca08be3316302b134cd73085c194270aa2d537dc9f2515b1e8c9.jpg)

![dc29badbf5d47bb1f81a782fbc7befa38db96b7a95c8c38161a858c46984ff60.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/dc29badbf5d47bb1f81a782fbc7befa38db96b7a95c8c38161a858c46984ff60.jpg)

![dea32dfa55d3e5b7e14a2f13d0b9dca6bf37e3e3ea0fb3d952ee00dd112bce44.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/dea32dfa55d3e5b7e14a2f13d0b9dca6bf37e3e3ea0fb3d952ee00dd112bce44.jpg)

![def86f0fd7bb72f8fed105dd2e258d6a42b13453d7545c07b20bd673894dce8a.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/def86f0fd7bb72f8fed105dd2e258d6a42b13453d7545c07b20bd673894dce8a.jpg)

![df7f9720cd09656007ae9e7bad41b0229ba6259175d8d7b0126eb97735a53a11.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/df7f9720cd09656007ae9e7bad41b0229ba6259175d8d7b0126eb97735a53a11.jpg)

![e02f178a59bc7a854d11d21fcccf739ad0e79456e26738c90ddae307fa16768f.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/e02f178a59bc7a854d11d21fcccf739ad0e79456e26738c90ddae307fa16768f.jpg)

![e9dca134f20b42c6c9b67ebef15fff9804513d571d4de45db3113742f5269ebc.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/e9dca134f20b42c6c9b67ebef15fff9804513d571d4de45db3113742f5269ebc.jpg)

![ef7a35be27b48301a37a2c2561e16e096e6f4478d4e3bc5c1fa00708bf7c4fe3.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/ef7a35be27b48301a37a2c2561e16e096e6f4478d4e3bc5c1fa00708bf7c4fe3.jpg)

![f335b1b0526c03c50059d173adcf60636a9cf55fc55baba619dd7de646883c35.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/f335b1b0526c03c50059d173adcf60636a9cf55fc55baba619dd7de646883c35.jpg)

![f8834346436a08d722cb32f614b8d394a2cf76fda872d8c64e02c2c41314868d.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/f8834346436a08d722cb32f614b8d394a2cf76fda872d8c64e02c2c41314868d.jpg)

![f8ee7709db8decc1384f85c8a09c7aebf7aac43348c34db3e54dad1cb3dd88d4.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/f8ee7709db8decc1384f85c8a09c7aebf7aac43348c34db3e54dad1cb3dd88d4.jpg)

![f9affaf4cec1f4330b3ab3ad82c2c29ec7c5c10a80426161b30ab453a7c1f098.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/f9affaf4cec1f4330b3ab3ad82c2c29ec7c5c10a80426161b30ab453a7c1f098.jpg)

![ff12782a461fb07130c4776d35d46efd2a26faed7447c69326f445bc28b3a415.jpg](../icml_results/1859_SlideSparse_%20Fast%20and%20Flexible%20%282N-2%29_2N%20Structured%20Sparsity/tables/ff12782a461fb07130c4776d35d46efd2a26faed7447c69326f445bc28b3a415.jpg)

<a id="paper-1835"></a>

## BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining


### Images

![020a5e1ab614735ebb3f6056a18b9ffe524d51a9a4b0ef08dbe20b43347cc845.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/020a5e1ab614735ebb3f6056a18b9ffe524d51a9a4b0ef08dbe20b43347cc845.jpg)

![0648542a202d87383ea591555ccee0fe2b0ac26a0696641a895dc191427b9f58.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/0648542a202d87383ea591555ccee0fe2b0ac26a0696641a895dc191427b9f58.jpg)

![077a42703af40de851921418a4739c00608804d3ea94abfec3dcf72a0b0eeab5.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/077a42703af40de851921418a4739c00608804d3ea94abfec3dcf72a0b0eeab5.jpg)

![0e779ab5c56fbb52fc48803b773e16f212d7171c33b2326e93b2f943da5baa82.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/0e779ab5c56fbb52fc48803b773e16f212d7171c33b2326e93b2f943da5baa82.jpg)

![3346c423f3677e2c2f2c74fe23567a22a84559b42575d770e74f9b0912a42e5f.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/3346c423f3677e2c2f2c74fe23567a22a84559b42575d770e74f9b0912a42e5f.jpg)

![3ca0f4d846a721a88368e9eeb7ffaf32d577af535899d7501012160c6ae219e3.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/3ca0f4d846a721a88368e9eeb7ffaf32d577af535899d7501012160c6ae219e3.jpg)

![6a67a3c5c2fb6d83add2a1e9d8e66cbd972fdba8cac873d1fc2560dc1ec6ed8e.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/6a67a3c5c2fb6d83add2a1e9d8e66cbd972fdba8cac873d1fc2560dc1ec6ed8e.jpg)

![6e4105907e71636060e6481994b2f2777237fd51d9eba6d7126a7b7fe13c5114.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/6e4105907e71636060e6481994b2f2777237fd51d9eba6d7126a7b7fe13c5114.jpg)

![702471bd31ad12ad731453c670bff7cdadc548ecedaedc53f6136aa22bacaeb9.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/702471bd31ad12ad731453c670bff7cdadc548ecedaedc53f6136aa22bacaeb9.jpg)

![d4af7145b2af4d748fafdd56ed1db958021b841ff2e7bf505697dcc306597a60.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/images/d4af7145b2af4d748fafdd56ed1db958021b841ff2e7bf505697dcc306597a60.jpg)

### Tables

![09b51f242b2b39651f19aad92faf1b706aeb3fdfb15be4e8299d1c9319a44816.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/09b51f242b2b39651f19aad92faf1b706aeb3fdfb15be4e8299d1c9319a44816.jpg)

![1c70bdd0815f3a1341d9be8369da9f6068035773dad78bc0f736406d89c5ce8c.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/1c70bdd0815f3a1341d9be8369da9f6068035773dad78bc0f736406d89c5ce8c.jpg)

![77d527ebe3e004b711c2f3db042ca5d3d082772be046bffbd1ba141edbe5952e.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/77d527ebe3e004b711c2f3db042ca5d3d082772be046bffbd1ba141edbe5952e.jpg)

![7e4ea937f7b92a7e47e6d814a36baff4f0dbc3b3681e319a738ace7796008d15.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/7e4ea937f7b92a7e47e6d814a36baff4f0dbc3b3681e319a738ace7796008d15.jpg)

![7e7f595e91c07302aea5e8c81f35f0c75ce9fe415ab6350fa6ba2f154d03c802.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/7e7f595e91c07302aea5e8c81f35f0c75ce9fe415ab6350fa6ba2f154d03c802.jpg)

![883ea9541e2e60f43eb822f158416a9b85b236f393ef321e57324eaa3949aca3.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/883ea9541e2e60f43eb822f158416a9b85b236f393ef321e57324eaa3949aca3.jpg)

![d95b05bdd256ab226c16fd72099fa2b2c369e900b0b38839962f5c3ecec5343c.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/d95b05bdd256ab226c16fd72099fa2b2c369e900b0b38839962f5c3ecec5343c.jpg)

![e72f4842bff6d50e8ff1de6913662a64aa8268027f73b8c3e780a0784ffdd78e.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/e72f4842bff6d50e8ff1de6913662a64aa8268027f73b8c3e780a0784ffdd78e.jpg)

![ebddd6e7dc79b560636d49fe8b6c6861690a3e7d23187556e1da09e378ba017a.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/ebddd6e7dc79b560636d49fe8b6c6861690a3e7d23187556e1da09e378ba017a.jpg)

![fa19fccb9ff0a88d08f8b03473b417c71bb39305abaf91d7c142878e2ce5c731.jpg](../icml_results/1860_BLISS_%20A%20Lightweight%20Bilevel%20Influence%20Scoring%20Method%20for%20Data%20Selection%20in%20Language%20Model%20Pretraini/tables/fa19fccb9ff0a88d08f8b03473b417c71bb39305abaf91d7c142878e2ce5c731.jpg)

<a id="paper-1836"></a>

## RACER: Risk-Aware Calibrated Efficient Routing for Large Language Models


### Images

![1f6a9fc8ca74299a37643499961e7c40f1ec556068fe16c02d75cf9f2babd74d.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/1f6a9fc8ca74299a37643499961e7c40f1ec556068fe16c02d75cf9f2babd74d.jpg)

![27257983341051cfcc07710a7713d389b0f723696ac33ef8fb412ef1451719ef.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/27257983341051cfcc07710a7713d389b0f723696ac33ef8fb412ef1451719ef.jpg)

![30ad2fc9a9a249ed79c2afd41bae439c822cc57a3c858d02cdb51762eb2fb1e4.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/30ad2fc9a9a249ed79c2afd41bae439c822cc57a3c858d02cdb51762eb2fb1e4.jpg)

![3e575123c07671e96e457a20fd2a226dfd5c8356bce9f8af6a9e817aa9274e18.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/3e575123c07671e96e457a20fd2a226dfd5c8356bce9f8af6a9e817aa9274e18.jpg)

![5abf8a156f2986df53d5fde8b0335876ce4e096cfe0312477eb5193d26f93e74.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/5abf8a156f2986df53d5fde8b0335876ce4e096cfe0312477eb5193d26f93e74.jpg)

![65836efc94be3a58c7563f6a07957e08814bb1baf9f04aec69b6d5bbc17f96ef.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/65836efc94be3a58c7563f6a07957e08814bb1baf9f04aec69b6d5bbc17f96ef.jpg)

![79346dc9c2166a7e49e7fcefa37a515c69fd30299a793d01a8766b1d93f02215.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/images/79346dc9c2166a7e49e7fcefa37a515c69fd30299a793d01a8766b1d93f02215.jpg)

### Tables

![183d8998534f31ed1ed3969f96f30b1d65c41639c1839e4b19840a579ee08a8a.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/183d8998534f31ed1ed3969f96f30b1d65c41639c1839e4b19840a579ee08a8a.jpg)

![20d18ca3af1e1d48208faa9899a8048e1f30f42bc4c9933beb1e8f1959e7b25e.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/20d18ca3af1e1d48208faa9899a8048e1f30f42bc4c9933beb1e8f1959e7b25e.jpg)

![3fceda788fb390cd80cd39dc34e6805338156c8ac324d419f911fd9131f15e2c.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/3fceda788fb390cd80cd39dc34e6805338156c8ac324d419f911fd9131f15e2c.jpg)

![692e0787a08cd36c2eb8b9082e6ddd9dbeac3c088d45aadcc873ce2f622e2b1a.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/692e0787a08cd36c2eb8b9082e6ddd9dbeac3c088d45aadcc873ce2f622e2b1a.jpg)

![7506af38f3bef127701b59bd1af092eb5c811e5d54565da2b8de4a20721bb9bb.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/7506af38f3bef127701b59bd1af092eb5c811e5d54565da2b8de4a20721bb9bb.jpg)

![8e86b317ce88b320bbe33bf2cf8ed2e9fe1720c85398e05f364db963bd49a0ce.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/8e86b317ce88b320bbe33bf2cf8ed2e9fe1720c85398e05f364db963bd49a0ce.jpg)

![8f0c014401a1e1e9177cc015fae5d67addd4a4dc397cf8348ed80104a0a05246.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/8f0c014401a1e1e9177cc015fae5d67addd4a4dc397cf8348ed80104a0a05246.jpg)

![bac3ff0691e3c5bf8cac7c1769169e25bfd5b6174f2ffbce79c3365fcbf9e405.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/bac3ff0691e3c5bf8cac7c1769169e25bfd5b6174f2ffbce79c3365fcbf9e405.jpg)

![e3f8f1185bb64c1a3a818184094292364f27f6de1f63145d720f31f9dd77a286.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/e3f8f1185bb64c1a3a818184094292364f27f6de1f63145d720f31f9dd77a286.jpg)

![e78331ac716d65e9755585304d267cf79213b5a0829539d3e39553aa666e6103.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/e78331ac716d65e9755585304d267cf79213b5a0829539d3e39553aa666e6103.jpg)

![ff371186a021824989f2c647e9705303047d17a55d3c937c20f43878108e82d3.jpg](../icml_results/1861_RACER_%20Risk-Aware%20Calibrated%20Efficient%20Routing%20for%20Large%20Language%20Models/tables/ff371186a021824989f2c647e9705303047d17a55d3c937c20f43878108e82d3.jpg)

<a id="paper-1837"></a>

## Hyperbolic Associative Memory Networks


### Images

![2b08dbf3574e893c25a9f26cc8b69ff61347dae9366e58da8e585b72aa9e8b15.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/images/2b08dbf3574e893c25a9f26cc8b69ff61347dae9366e58da8e585b72aa9e8b15.jpg)

### Tables

![0270dd2cbef3cd3cb09caf3cea19a55893d329808071bec6a2d0c959dd92aebe.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/0270dd2cbef3cd3cb09caf3cea19a55893d329808071bec6a2d0c959dd92aebe.jpg)

![0819ac358c94b3634bbd2f98f77bd650a4e43158d6abc1dc595a052ecd01181a.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/0819ac358c94b3634bbd2f98f77bd650a4e43158d6abc1dc595a052ecd01181a.jpg)

![0eb9184c1a29316600d5f1e1a290ff295f8735439bb7bd099290c29032d272e5.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/0eb9184c1a29316600d5f1e1a290ff295f8735439bb7bd099290c29032d272e5.jpg)

![109aaad4018d43f0cfa870880cf003bc2d5f5add8c7b11562c2edf52464389b1.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/109aaad4018d43f0cfa870880cf003bc2d5f5add8c7b11562c2edf52464389b1.jpg)

![1cbd9798371c2b083713f3a3938187e29acc2aebd2505398bfaab6f99c1c35e8.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/1cbd9798371c2b083713f3a3938187e29acc2aebd2505398bfaab6f99c1c35e8.jpg)

![28cca4f463871a85dfed8809cde0e97f86357b1010348e9307c66e83263c764b.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/28cca4f463871a85dfed8809cde0e97f86357b1010348e9307c66e83263c764b.jpg)

![380bbbf6801df4836eb40f8583e52513c001034c123c09fbdbfcaae402f2d5a2.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/380bbbf6801df4836eb40f8583e52513c001034c123c09fbdbfcaae402f2d5a2.jpg)

![42de45da7082145da7b53cf218e7d52d15c520d96747b29227df3eb8a4c18c94.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/42de45da7082145da7b53cf218e7d52d15c520d96747b29227df3eb8a4c18c94.jpg)

![577598e4603152260480fa1b47b230c8d9919bcb71b51ee8b5aebd8cde95b799.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/577598e4603152260480fa1b47b230c8d9919bcb71b51ee8b5aebd8cde95b799.jpg)

![759434d087ddaa38c3d50f8818e56222eb49bd7ca0228888ced6c21f52eea0d6.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/759434d087ddaa38c3d50f8818e56222eb49bd7ca0228888ced6c21f52eea0d6.jpg)

![84faa85788ade915949579e6a70762578a9c83f0687b2a0da6e86f8b136dfbcb.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/84faa85788ade915949579e6a70762578a9c83f0687b2a0da6e86f8b136dfbcb.jpg)

![8720cd178d886861bd0e7d5cc189307619ffb81707c90262b989e24892b49f0f.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/8720cd178d886861bd0e7d5cc189307619ffb81707c90262b989e24892b49f0f.jpg)

![920ba2cc493ce41ff0d65dc8770da37fd7280fc56b7b57b08f990e300ddaa5af.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/920ba2cc493ce41ff0d65dc8770da37fd7280fc56b7b57b08f990e300ddaa5af.jpg)

![979162262efa1ea8fbe42b9489aa0f3edaa0cf186e67e539791633c3e23fd133.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/979162262efa1ea8fbe42b9489aa0f3edaa0cf186e67e539791633c3e23fd133.jpg)

![b154fc91dfeb32d9ff8557bb63b95bd4b8e22d7202e2b3ef20775209cb0e7f4f.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/b154fc91dfeb32d9ff8557bb63b95bd4b8e22d7202e2b3ef20775209cb0e7f4f.jpg)

![d2a5128a196d5f7ea028e6a3c14fd2e8d29955a039bd226e0cfcf163cd6b3528.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/d2a5128a196d5f7ea028e6a3c14fd2e8d29955a039bd226e0cfcf163cd6b3528.jpg)

![e206e8de679473030bd2daabf722a864174975c030f03d97d878bc25e5d105f8.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/e206e8de679473030bd2daabf722a864174975c030f03d97d878bc25e5d105f8.jpg)

![f56d86de1ac94f4ad4600e70ec689a84df6128044bf39ad5c5abc68ba117b591.jpg](../icml_results/1862_Hyperbolic%20Associative%20Memory%20Networks/tables/f56d86de1ac94f4ad4600e70ec689a84df6128044bf39ad5c5abc68ba117b591.jpg)

<a id="paper-1838"></a>

## AI Cartography: Mapping the Latent Landscape of AI Benchmark Ecosystems


### Images

![021ec75a81ec3e792125dfa649347158a628794591e13130239a6191e0c3ad7a.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/021ec75a81ec3e792125dfa649347158a628794591e13130239a6191e0c3ad7a.jpg)

![213dd5dd6916a48d509d2606e472e6459d98fcdde9d7dc06e17888cb3428688a.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/213dd5dd6916a48d509d2606e472e6459d98fcdde9d7dc06e17888cb3428688a.jpg)

![639ccb372f739d9123424f21395bb807d19fbb8b1d8ea0db6b35ddbc54b498dc.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/639ccb372f739d9123424f21395bb807d19fbb8b1d8ea0db6b35ddbc54b498dc.jpg)

![99337a6812974aac77398400cb14665ea75520459f26fa42a9fbf01c6acdc81e.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/99337a6812974aac77398400cb14665ea75520459f26fa42a9fbf01c6acdc81e.jpg)

![9ee10ea3ec8cb57b6a006b7bcc494587222372709c3c0391bb48a2fa66aaa34b.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/9ee10ea3ec8cb57b6a006b7bcc494587222372709c3c0391bb48a2fa66aaa34b.jpg)

![b00330379465bbb11c39cb1c22708da53d0849997cd09e29c2c8ab6c7c5858a7.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/b00330379465bbb11c39cb1c22708da53d0849997cd09e29c2c8ab6c7c5858a7.jpg)

![ca62aadfdc367703ed00535b38ea12a167e20f7770aebf75e4059beb4167fb70.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/ca62aadfdc367703ed00535b38ea12a167e20f7770aebf75e4059beb4167fb70.jpg)

![f0008d8c460e414b5ce0549de03825e9071af6a27a3854254559fec8cdb9d578.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/f0008d8c460e414b5ce0549de03825e9071af6a27a3854254559fec8cdb9d578.jpg)

![f7c6d20c5b158b24c71f9ecfa110cf3f15d3b2dadf595b473a62f90f606cd939.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/images/f7c6d20c5b158b24c71f9ecfa110cf3f15d3b2dadf595b473a62f90f606cd939.jpg)

### Tables

![073184f5872ee7ab1c3684ac9fbe2e1c08d8e23e53793c462e8716a619fd33f1.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/073184f5872ee7ab1c3684ac9fbe2e1c08d8e23e53793c462e8716a619fd33f1.jpg)

![0d1488ef680b29d3b0c0e5d1d97f9f898fa88c0301fa85fd12617c43635434b5.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/0d1488ef680b29d3b0c0e5d1d97f9f898fa88c0301fa85fd12617c43635434b5.jpg)

![2bdabea666bf079bc30adc0c6cd235ab1ab3728f167cb63ea53b37c9eb49943f.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/2bdabea666bf079bc30adc0c6cd235ab1ab3728f167cb63ea53b37c9eb49943f.jpg)

![4c437a702846b640b7c3dbfabacaafa5153b8b5a1bbf93aca17bc9a684713347.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/4c437a702846b640b7c3dbfabacaafa5153b8b5a1bbf93aca17bc9a684713347.jpg)

![4d269059f9baa39413256519140a75caf69c5b22fec9c48787e72e35b1d52ef4.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/4d269059f9baa39413256519140a75caf69c5b22fec9c48787e72e35b1d52ef4.jpg)

![5a17ba13f4b7588dc682251490360100b93b33f2371476b043a585419ffa03fa.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/5a17ba13f4b7588dc682251490360100b93b33f2371476b043a585419ffa03fa.jpg)

![5b1b29255a310f52175915923a884856e2be376258d6ce5f7572a8eed68f9663.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/5b1b29255a310f52175915923a884856e2be376258d6ce5f7572a8eed68f9663.jpg)

![637ce72c76dd191e46692b3aa6d0e5837d7a44960513ec1f14091d57c20c5db6.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/637ce72c76dd191e46692b3aa6d0e5837d7a44960513ec1f14091d57c20c5db6.jpg)

![7697d03ed91e037422976821106c2a87a3e846707cd2e8c60881734c7645bcb3.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/7697d03ed91e037422976821106c2a87a3e846707cd2e8c60881734c7645bcb3.jpg)

![7a62fd41f656e6ab5376dfa65b00e387a9df1efa041cadec8d567ac48538d8ca.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/7a62fd41f656e6ab5376dfa65b00e387a9df1efa041cadec8d567ac48538d8ca.jpg)

![7b2274dbee8ae575d02c6e7173bcf3b34b32dc2fc77f729345c9e1f288b48fda.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/7b2274dbee8ae575d02c6e7173bcf3b34b32dc2fc77f729345c9e1f288b48fda.jpg)

![82c8c0f54e5d644b3bdbb93c1173125ac1cdbb888266c45dc9a1df99f9b06ebb.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/82c8c0f54e5d644b3bdbb93c1173125ac1cdbb888266c45dc9a1df99f9b06ebb.jpg)

![88492fce809247e0afcc360f233bfb2723e743263dd0b9d4613ad51e44895801.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/88492fce809247e0afcc360f233bfb2723e743263dd0b9d4613ad51e44895801.jpg)

![c26b0ff8d9387e48b40d1ddeab7eb82a686adf7331c751d69399e427750bcf1c.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/c26b0ff8d9387e48b40d1ddeab7eb82a686adf7331c751d69399e427750bcf1c.jpg)

![cd7176a6160be4d7dc556d156ece5670d2f4c4a6cbdb327833222d4cfa1857c6.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/cd7176a6160be4d7dc556d156ece5670d2f4c4a6cbdb327833222d4cfa1857c6.jpg)

![d80c815a1408ef6231ffbccaca0df56366b70f5af26c0d149e34c7bc71181f2d.jpg](../icml_results/1863_AI%20Cartography_%20Mapping%20the%20Latent%20Landscape%20of%20AI%20Benchmark%20Ecosystems/tables/d80c815a1408ef6231ffbccaca0df56366b70f5af26c0d149e34c7bc71181f2d.jpg)

<a id="paper-1839"></a>

## Skip-It? Theoretical Conditions for Layer Skipping in Vision–Language Models


### Images

![00da022c37486da82e1b6950acc17d0723f3d00ca88dbd0d05fd8c9673713dca.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/00da022c37486da82e1b6950acc17d0723f3d00ca88dbd0d05fd8c9673713dca.jpg)

![090747e1b57cbaad240b5349ea907271af06955ae20c55285bc8e98493843652.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/090747e1b57cbaad240b5349ea907271af06955ae20c55285bc8e98493843652.jpg)

![0e398c240d5b2d061990e547256f4c0cd1f465588d26429dfb9faa8f0a6f10e0.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/0e398c240d5b2d061990e547256f4c0cd1f465588d26429dfb9faa8f0a6f10e0.jpg)

![13d8ac8fed37d7d0a8662fdc3800549d0774fdbeb5a327b9da8bf1ad4a2c8a3b.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/13d8ac8fed37d7d0a8662fdc3800549d0774fdbeb5a327b9da8bf1ad4a2c8a3b.jpg)

![16d9f38f4fb058c92c3a43e0d529e1d0892575c404d626b1f81e82546e861772.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/16d9f38f4fb058c92c3a43e0d529e1d0892575c404d626b1f81e82546e861772.jpg)

![1f93462bbfa7a871ed64c36c45d4047d43d489d7b486adefee9d414f3bf99543.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/1f93462bbfa7a871ed64c36c45d4047d43d489d7b486adefee9d414f3bf99543.jpg)

![20f98c065abb42504c173915b18d6c4826f527e9c2837ee620d0d816a33e91ef.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/20f98c065abb42504c173915b18d6c4826f527e9c2837ee620d0d816a33e91ef.jpg)

![25cda2e44084c82bacfe5a825794296f481da169b8d1fe928142ba737f5402c5.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/25cda2e44084c82bacfe5a825794296f481da169b8d1fe928142ba737f5402c5.jpg)

![2ab8b2bb2966157bcbbb9b44779aa91d8ea81789693d5ca268034c9f19e4d897.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/2ab8b2bb2966157bcbbb9b44779aa91d8ea81789693d5ca268034c9f19e4d897.jpg)

![38be26f6e88acc4885c5f5c10436444e866544229e01808fa564160db9e4339e.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/38be26f6e88acc4885c5f5c10436444e866544229e01808fa564160db9e4339e.jpg)

![3aa3ace49594c16953011c14994da967e007b3bea56dcf6c9457265e37f76131.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/3aa3ace49594c16953011c14994da967e007b3bea56dcf6c9457265e37f76131.jpg)

![3c93bc75ab5b9cdf9df06324073c72bf90ded9fd7b0ebdc0fc2f7628349f49b7.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/3c93bc75ab5b9cdf9df06324073c72bf90ded9fd7b0ebdc0fc2f7628349f49b7.jpg)

![4584c82a202d757473c0dc7246e709017551a6f46727e4ae7f21ec2a822d5d5e.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/4584c82a202d757473c0dc7246e709017551a6f46727e4ae7f21ec2a822d5d5e.jpg)

![4944abc3188d3e122979efdccb6fa274ec388f264552eeed1bf70c97931cc7ec.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/4944abc3188d3e122979efdccb6fa274ec388f264552eeed1bf70c97931cc7ec.jpg)

![5fd703a713c66b1e4d906e75ae7a4eb421960b79f1b9dd059921183e55b24a71.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/5fd703a713c66b1e4d906e75ae7a4eb421960b79f1b9dd059921183e55b24a71.jpg)

![7de9948d2279a86431d5d24ddcd6b0e10dead3022d36469554cfa6c7fa86058c.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/7de9948d2279a86431d5d24ddcd6b0e10dead3022d36469554cfa6c7fa86058c.jpg)

![7f7e7325f7bc5663de9d26afc903cd134e1384f7010ebffbe9b0212b286c5415.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/7f7e7325f7bc5663de9d26afc903cd134e1384f7010ebffbe9b0212b286c5415.jpg)

![81b1a34aaf2931429285bd1f4d7c0b582c457fd1ab3678db0970497e88fd716b.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/81b1a34aaf2931429285bd1f4d7c0b582c457fd1ab3678db0970497e88fd716b.jpg)

![868a8a8e3418c88b4fdc513251f4fc0b4cbbc63818c6538de3aafb455586ead4.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/868a8a8e3418c88b4fdc513251f4fc0b4cbbc63818c6538de3aafb455586ead4.jpg)

![889fd7798d6a5e5009e131cf30b0929a380bc8c41368a5558171eee9ebd279ab.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/889fd7798d6a5e5009e131cf30b0929a380bc8c41368a5558171eee9ebd279ab.jpg)

![8b89af42b91949d5f182aad5aa8c635bf1a14cf5aee16ceaf7bb5614a5698c24.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/8b89af42b91949d5f182aad5aa8c635bf1a14cf5aee16ceaf7bb5614a5698c24.jpg)

![ae11bc926fd44aedaf380f7147291c07da22cd9c246234205d4b0143f4eec3cf.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/ae11bc926fd44aedaf380f7147291c07da22cd9c246234205d4b0143f4eec3cf.jpg)

![b816c04d44d55b43521ef7b561a9c89f8f65f4af882238687ae8d1347bcec81a.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/b816c04d44d55b43521ef7b561a9c89f8f65f4af882238687ae8d1347bcec81a.jpg)

![b87692675238db576af4470c46d034219c6f0448ae36d0a111f86a3063f7ca64.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/b87692675238db576af4470c46d034219c6f0448ae36d0a111f86a3063f7ca64.jpg)

![ba357f597acd926041cdd693528f76911bb7a4a3fa61594eef497c8be321d707.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/ba357f597acd926041cdd693528f76911bb7a4a3fa61594eef497c8be321d707.jpg)

![c468f62ea8bdd8acb0d695f47ebc33fb770c6852830f1677155cefed67a4d389.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/c468f62ea8bdd8acb0d695f47ebc33fb770c6852830f1677155cefed67a4d389.jpg)

![cbd262c19c07e9bb54d6a71087987739331331a7cf409b9cdac9565e03d9e062.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/cbd262c19c07e9bb54d6a71087987739331331a7cf409b9cdac9565e03d9e062.jpg)

![df20eac504e381912defd6073bd218358d108e5459624c82f62ffcaab7283f4f.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/images/df20eac504e381912defd6073bd218358d108e5459624c82f62ffcaab7283f4f.jpg)

### Tables

![9ba58001cf8b5ca042db1f61dd70961d95ba087732954e8608f343c12258f930.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/tables/9ba58001cf8b5ca042db1f61dd70961d95ba087732954e8608f343c12258f930.jpg)

![df33ed80e85bfb0f0c0b77462e25acf5e4d3bd457b507e6307e9a5b972082cad.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/tables/df33ed80e85bfb0f0c0b77462e25acf5e4d3bd457b507e6307e9a5b972082cad.jpg)

![f10a2d218ebb684e2912d16667f2ffb4d60e932d5793c09b33072c9940a10e39.jpg](../icml_results/1864_Skip-It_%20Theoretical%20Conditions%20for%20Layer%20Skipping%20in%20Vision%E2%80%93Language%20Models/tables/f10a2d218ebb684e2912d16667f2ffb4d60e932d5793c09b33072c9940a10e39.jpg)

<a id="paper-1840"></a>

## Geodesic Calculus on Implicitly Defined Latent Manifolds


### Images

![04066fcbf2e47e799a6dee8fbda487448ebf82908e5e0227389fad9cc681f617.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/04066fcbf2e47e799a6dee8fbda487448ebf82908e5e0227389fad9cc681f617.jpg)

![26709104c9353c22b5e1fe796a3cd4dd004de1c4dc86d802de3788867860bf4b.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/26709104c9353c22b5e1fe796a3cd4dd004de1c4dc86d802de3788867860bf4b.jpg)

![26f6d387a54a40bb256818b55c2da8d472bbed9f6435245a47e51a1790598120.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/26f6d387a54a40bb256818b55c2da8d472bbed9f6435245a47e51a1790598120.jpg)

![2f6a668cee6c6accd5a341a042b434239719431ba057f4b53d02e5b16236bf0f.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/2f6a668cee6c6accd5a341a042b434239719431ba057f4b53d02e5b16236bf0f.jpg)

![3a07ea29f93d8876e8813abc0c1bf2753f40f25eea9a6d1551929e8c2a9b529d.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/3a07ea29f93d8876e8813abc0c1bf2753f40f25eea9a6d1551929e8c2a9b529d.jpg)

![4149de8eee3d146ce9d491145b751fa33b3fc999f55a30efe047c03c7990e66f.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/4149de8eee3d146ce9d491145b751fa33b3fc999f55a30efe047c03c7990e66f.jpg)

![49af5f12cd960164734ec22422dee8e8660c8fc2759efb66a476d5b0d5ba6970.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/49af5f12cd960164734ec22422dee8e8660c8fc2759efb66a476d5b0d5ba6970.jpg)

![53a001ec88b43296549078e859f0203cdfde015422346371bb274e075d9d22e8.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/53a001ec88b43296549078e859f0203cdfde015422346371bb274e075d9d22e8.jpg)

![6818fa75652574f18cbb006e7f18e68529b71a1d99ab787ceb6c7b1f51592d49.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/6818fa75652574f18cbb006e7f18e68529b71a1d99ab787ceb6c7b1f51592d49.jpg)

![6ee36cbec886e14af2919093ebb64bee95c3cf7cbc5a465d74c935e941d0788a.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/6ee36cbec886e14af2919093ebb64bee95c3cf7cbc5a465d74c935e941d0788a.jpg)

![823a106cb1a4e039e6dbdbb4ea899705a35ece0a06c55f2c9f8194999a4ede64.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/823a106cb1a4e039e6dbdbb4ea899705a35ece0a06c55f2c9f8194999a4ede64.jpg)

![8df8324794041becea4b6d1b0365f447fc5a85b70932553e004aa56d7958f695.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/8df8324794041becea4b6d1b0365f447fc5a85b70932553e004aa56d7958f695.jpg)

![9333a8cc4681e85faeec82a0ba38f1717a2a964e875d15272e01587259bc05e4.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/9333a8cc4681e85faeec82a0ba38f1717a2a964e875d15272e01587259bc05e4.jpg)

![aa9002f33e0a7f4d354e698b148bc97d2ae2cf0aa1dd098b3322f1b37f429d3e.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/aa9002f33e0a7f4d354e698b148bc97d2ae2cf0aa1dd098b3322f1b37f429d3e.jpg)

![b6ff7235053fdeca1d05ae4cefa6eee6641467f0c91a27be34d836621833a754.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/b6ff7235053fdeca1d05ae4cefa6eee6641467f0c91a27be34d836621833a754.jpg)

![c184830470058853dc8ba1f86ec64fbbf246c5534944a3af4930078151ee5501.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/c184830470058853dc8ba1f86ec64fbbf246c5534944a3af4930078151ee5501.jpg)

![c6c790f4b21db8ef4794b9af0e9d510a6d03494ca1f88d9204343496afa3e658.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/c6c790f4b21db8ef4794b9af0e9d510a6d03494ca1f88d9204343496afa3e658.jpg)

![dc9960275d0ee3a0b552373e9605b1d1015cff44f35277217ee2297fa061a4aa.jpg](../icml_results/1865_Geodesic%20Calculus%20on%20Implicitly%20Defined%20Latent%20Manifolds/images/dc9960275d0ee3a0b552373e9605b1d1015cff44f35277217ee2297fa061a4aa.jpg)

<a id="paper-1841"></a>

## Unsupervised Hierarchical Skill Discovery


### Images

![0eb4c22ca470255d113701ced1fc2837b5489c86591af8a99f55d392fd82aa15.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/0eb4c22ca470255d113701ced1fc2837b5489c86591af8a99f55d392fd82aa15.jpg)

![118d59968a27f9ca9a397740174a6be375060fda7858840853ad72e5dd0830f2.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/118d59968a27f9ca9a397740174a6be375060fda7858840853ad72e5dd0830f2.jpg)

![18673b0f032e4713545dd37ee05737c37cde8bc7a3f774fffdd134a1552440b9.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/18673b0f032e4713545dd37ee05737c37cde8bc7a3f774fffdd134a1552440b9.jpg)

![71a7fc4374598ab7d22d98116eb8063fbb140c811407aa934385613c9025bb5e.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/71a7fc4374598ab7d22d98116eb8063fbb140c811407aa934385613c9025bb5e.jpg)

![762ff2a5ecfba0314a6dff882dd09722092e82b324025127e73516f0045ff7b7.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/762ff2a5ecfba0314a6dff882dd09722092e82b324025127e73516f0045ff7b7.jpg)

![901fb494d240297be0b191e06dc1965804ea8115529713a67e0e74df7d063d26.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/901fb494d240297be0b191e06dc1965804ea8115529713a67e0e74df7d063d26.jpg)

![9a16054dacc0f96a5d0ea327beb7dd8142abf9abbebf2f8343f5a8338770ec5e.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/9a16054dacc0f96a5d0ea327beb7dd8142abf9abbebf2f8343f5a8338770ec5e.jpg)

![9f155e779bc46230a0e57bbc17aac7e6f78c907e8e21150be65d50620a02518a.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/9f155e779bc46230a0e57bbc17aac7e6f78c907e8e21150be65d50620a02518a.jpg)

![a4748a71e78f3cf7fed88a3a8802113e8b0b353857fa4d8bf51ac53d912709f8.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/a4748a71e78f3cf7fed88a3a8802113e8b0b353857fa4d8bf51ac53d912709f8.jpg)

![a967fe3ad1344b96c1723fbe733539bf6ec4bae7dea365f37fc654f581a7f3d9.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/a967fe3ad1344b96c1723fbe733539bf6ec4bae7dea365f37fc654f581a7f3d9.jpg)

![abde8e0bef656dbc423d8946048861f5c9c200f59b2c404b06eef81b43b32d82.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/abde8e0bef656dbc423d8946048861f5c9c200f59b2c404b06eef81b43b32d82.jpg)

![bc45f41a2e524b1311ece9eac0e5fa3ac5fa81c5bc093e429de91e637137cbd9.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/bc45f41a2e524b1311ece9eac0e5fa3ac5fa81c5bc093e429de91e637137cbd9.jpg)

![bc6eba9be2d0a4f434b4d5d7a5cf72b3bcd7056c7dec0a68741677e7e144c82e.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/bc6eba9be2d0a4f434b4d5d7a5cf72b3bcd7056c7dec0a68741677e7e144c82e.jpg)

![c779994dd58343ba1a2cccc29b5bb7bedfa0d80bb94413c047b4fa9af4af36e6.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/c779994dd58343ba1a2cccc29b5bb7bedfa0d80bb94413c047b4fa9af4af36e6.jpg)

![d725dea20664ad4f4413c9204199a268f238a33aecc57bef34ebc8b9b2cc5a41.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/d725dea20664ad4f4413c9204199a268f238a33aecc57bef34ebc8b9b2cc5a41.jpg)

![de952a6988c3f6292aab432756ccee24fc8921955f8da90c749433baaceb9b4c.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/de952a6988c3f6292aab432756ccee24fc8921955f8da90c749433baaceb9b4c.jpg)

![df67411c44b71afdad05d0998e627e70a61e77370e6aae58bef9efa991f5b5c0.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/df67411c44b71afdad05d0998e627e70a61e77370e6aae58bef9efa991f5b5c0.jpg)

![e4573a182ed842d3b2eb73665bcfa7393a4d691d4e2215e0568193bb402df422.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/e4573a182ed842d3b2eb73665bcfa7393a4d691d4e2215e0568193bb402df422.jpg)

![fe6e5a2be8929ec6e35d910433475f997684aee6196387df687de62e40d1aff5.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/fe6e5a2be8929ec6e35d910433475f997684aee6196387df687de62e40d1aff5.jpg)

![fed6ea4e6d550875a8ab33a7b9a2b49bf02d627b9fabbbc253d4fcebc31b0400.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/images/fed6ea4e6d550875a8ab33a7b9a2b49bf02d627b9fabbbc253d4fcebc31b0400.jpg)

### Tables

![093aa52625b1e66a4006d41f3e24124647aa4160c32e03a2c0ba36b3c0a7f69a.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/093aa52625b1e66a4006d41f3e24124647aa4160c32e03a2c0ba36b3c0a7f69a.jpg)

![0b7749e1cd5b7e6fb87b996c9d345f190c36f8e4a7857d77362ff90c8fe88fda.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/0b7749e1cd5b7e6fb87b996c9d345f190c36f8e4a7857d77362ff90c8fe88fda.jpg)

![1a1c9d07ced8866ef781a4c9e301a3e93c5dba78799e434f6f0fff312eee277b.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/1a1c9d07ced8866ef781a4c9e301a3e93c5dba78799e434f6f0fff312eee277b.jpg)

![1b4b19555accc43a87b481e4a70b7f9c33d9521d0dff0a4b1fda4b6a4260d456.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/1b4b19555accc43a87b481e4a70b7f9c33d9521d0dff0a4b1fda4b6a4260d456.jpg)

![1ce42a905b0c1c1b514721807b721529c049df74e54a01465a80fae1d216eaa1.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/1ce42a905b0c1c1b514721807b721529c049df74e54a01465a80fae1d216eaa1.jpg)

![379d96b113ddedca65d2576ccb660251c04b3ec9607699888bbc5ba1621fcb21.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/379d96b113ddedca65d2576ccb660251c04b3ec9607699888bbc5ba1621fcb21.jpg)

![472665a6396086a8dd72ced75c39da7332900320508ddff61d9d0b96a4c2aa46.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/472665a6396086a8dd72ced75c39da7332900320508ddff61d9d0b96a4c2aa46.jpg)

![4c5421eb25df0adfad1db96bf8808de04eb2f26402c9b1e5b90ef6659badfd90.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/4c5421eb25df0adfad1db96bf8808de04eb2f26402c9b1e5b90ef6659badfd90.jpg)

![77b8c755bd0f26913d4179c6e61a8dabfb129a8b334ef465ddc61988b0af963b.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/77b8c755bd0f26913d4179c6e61a8dabfb129a8b334ef465ddc61988b0af963b.jpg)

![7c9a3be55553917fcc6ac6bba9a74707241d6a4a979a7c255766b29c155902df.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/7c9a3be55553917fcc6ac6bba9a74707241d6a4a979a7c255766b29c155902df.jpg)

![8004ee6bb6497713d125d82afe4cfdaee4fda8d89d959689a5a02a78905bf7d1.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/8004ee6bb6497713d125d82afe4cfdaee4fda8d89d959689a5a02a78905bf7d1.jpg)

![a5a177bcf2c21448a754aa61b16564f56adb4bc3fd8e54cbba7d4efc56e38b10.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/a5a177bcf2c21448a754aa61b16564f56adb4bc3fd8e54cbba7d4efc56e38b10.jpg)

![b1037e6daddc5d2c897ac739f3b87412a3c02ff136314176af6192c3197d2b77.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/b1037e6daddc5d2c897ac739f3b87412a3c02ff136314176af6192c3197d2b77.jpg)

![b5f1ef6af15697307976427df1b9c958083d9b334ffd4f3aabcd0a7714a27502.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/b5f1ef6af15697307976427df1b9c958083d9b334ffd4f3aabcd0a7714a27502.jpg)

![b89554b7f5a11bceeb8cf3bd674ed32465b00dc111ecf2b147421c93adf85a0c.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/b89554b7f5a11bceeb8cf3bd674ed32465b00dc111ecf2b147421c93adf85a0c.jpg)

![c1bea37bb1bac5ef73806e9e1d76cdab6b358203f1299b7ed862c7345b7c52cb.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/c1bea37bb1bac5ef73806e9e1d76cdab6b358203f1299b7ed862c7345b7c52cb.jpg)

![c7f0de0bbd9dc6f1ff10efa2efffcf63e46ce2962d2d263e0f33b7451a7a2a37.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/c7f0de0bbd9dc6f1ff10efa2efffcf63e46ce2962d2d263e0f33b7451a7a2a37.jpg)

![c9fb56f62233adb5c8ba8e151da33555f77f3e9f75218e2019e75306d0d155ee.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/c9fb56f62233adb5c8ba8e151da33555f77f3e9f75218e2019e75306d0d155ee.jpg)

![ce5f74a9d3fdc3764219495e6e76a8bfdae5b5fd1401cfabbc6d413beac77085.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/ce5f74a9d3fdc3764219495e6e76a8bfdae5b5fd1401cfabbc6d413beac77085.jpg)

![e2edf6275fd7d89eb2880a971d50397d848f994a13f99803065b7b42fde2cf7b.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/e2edf6275fd7d89eb2880a971d50397d848f994a13f99803065b7b42fde2cf7b.jpg)

![e6a3f824454fe0c837c549317cc9b565a5d5baa0f3545eaef9eac75f2cac4ac6.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/e6a3f824454fe0c837c549317cc9b565a5d5baa0f3545eaef9eac75f2cac4ac6.jpg)

![ec8db6475fe616638ab034056e8a948bf35011e05baaaf8fdbea965c7c4d2be2.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/ec8db6475fe616638ab034056e8a948bf35011e05baaaf8fdbea965c7c4d2be2.jpg)

![fcdababdda6de05b3aa734bf391da9a0853cff259b18bc57fa2e5d98fe2c1b5b.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/fcdababdda6de05b3aa734bf391da9a0853cff259b18bc57fa2e5d98fe2c1b5b.jpg)

![ff4a7f920a2c7f28f1068d38f1b66bc66651c52bd8a46d7508ecc784dd16deb8.jpg](../icml_results/1866_Unsupervised%20Hierarchical%20Skill%20Discovery/tables/ff4a7f920a2c7f28f1068d38f1b66bc66651c52bd8a46d7508ecc784dd16deb8.jpg)

<a id="paper-1842"></a>

## Exact and Approximate Algorithms for Polytree Learning


### Images

![0c145183bc1983d6f5abaf3d01b83cf63d3503103ade9f7f0812192bc255cb0f.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/0c145183bc1983d6f5abaf3d01b83cf63d3503103ade9f7f0812192bc255cb0f.jpg)

![505b88d609229501e6f201325bef6041812e6257ed5e6a39f3476d02a370b29c.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/505b88d609229501e6f201325bef6041812e6257ed5e6a39f3476d02a370b29c.jpg)

![54aa98b006d3d700f9e012d6776bde60aac2d0d0322cfb8e7f3d13ba7b971a00.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/54aa98b006d3d700f9e012d6776bde60aac2d0d0322cfb8e7f3d13ba7b971a00.jpg)

![74251e516434525c48be971ccda8f9f07bd722eafeca731159676893ff0c5b16.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/74251e516434525c48be971ccda8f9f07bd722eafeca731159676893ff0c5b16.jpg)

![c3310de82102d330f16f6fb7f02f287cb967e842cbefb701e37ff6c925028a0a.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/c3310de82102d330f16f6fb7f02f287cb967e842cbefb701e37ff6c925028a0a.jpg)

![e9d863ff3c94d8f5e7d6ca457b8d6779d9e7a581aa47e0a0244aac047ea79196.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/images/e9d863ff3c94d8f5e7d6ca457b8d6779d9e7a581aa47e0a0244aac047ea79196.jpg)

### Tables

![dafb9d3295afba70ab525f5fd342882743669a47186916b1c059c127a049170f.jpg](../icml_results/1867_Exact%20and%20Approximate%20Algorithms%20for%20Polytree%20Learning/tables/dafb9d3295afba70ab525f5fd342882743669a47186916b1c059c127a049170f.jpg)

<a id="paper-1843"></a>

## Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs


### Images

![042e8001734a02f67ff394a30259c2230ff2bff9922cdca754cc815e5c4f3756.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/042e8001734a02f67ff394a30259c2230ff2bff9922cdca754cc815e5c4f3756.jpg)

![1826af4bd7df5e6d77a3b0df6ae329b7c464593b5b18463466d62a0b39f5c475.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/1826af4bd7df5e6d77a3b0df6ae329b7c464593b5b18463466d62a0b39f5c475.jpg)

![66351ceb5782d6d79424ea587f25389c24aa5fdbe2652fe75aef3c0ad0d9058a.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/66351ceb5782d6d79424ea587f25389c24aa5fdbe2652fe75aef3c0ad0d9058a.jpg)

![693047189c4252e5c4f255066a35dff938a3c3e7e1305745380323a78b1b3c27.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/693047189c4252e5c4f255066a35dff938a3c3e7e1305745380323a78b1b3c27.jpg)

![6d3b3ce21cb09b74fe42f1252b30676f424d321d750eaaaabf5c76a51f5d29e0.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/6d3b3ce21cb09b74fe42f1252b30676f424d321d750eaaaabf5c76a51f5d29e0.jpg)

![92fbb5a042b7abe88815015fe48aee856f7c7cc47d99858dd79317cc4ac6b35c.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/92fbb5a042b7abe88815015fe48aee856f7c7cc47d99858dd79317cc4ac6b35c.jpg)

![a01ae062ef9c866fd1fc9bf6d80a84aea03b6f56fba874b3755efba1a1f614dc.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/a01ae062ef9c866fd1fc9bf6d80a84aea03b6f56fba874b3755efba1a1f614dc.jpg)

![cbc28fec5762cdceba53a3a90e7f40032b99c3a8dd37e86132e3aac49d59b6f2.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/images/cbc28fec5762cdceba53a3a90e7f40032b99c3a8dd37e86132e3aac49d59b6f2.jpg)

### Tables

![150754f288a31e5ed81afb178c4902352e8be1e168cc56363ddb3ab5af877471.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/150754f288a31e5ed81afb178c4902352e8be1e168cc56363ddb3ab5af877471.jpg)

![1f801e5e8a1a889aff87ee02fcb2aba634c291150cb84ebddfdad994c71a6514.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/1f801e5e8a1a889aff87ee02fcb2aba634c291150cb84ebddfdad994c71a6514.jpg)

![2bfa515cd08c23778d309ca9fc7cfdf4a9ce2cfeb3a78722322b34a09b2379cd.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/2bfa515cd08c23778d309ca9fc7cfdf4a9ce2cfeb3a78722322b34a09b2379cd.jpg)

![2e6bbcd9de9350803946c19719b9352f7e7c98a4e54449653dd1c41dae044c33.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/2e6bbcd9de9350803946c19719b9352f7e7c98a4e54449653dd1c41dae044c33.jpg)

![4b68557d915005b1d97073bd45f06d39c4f7656009f2cd93a306f7d3d511f0dc.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/4b68557d915005b1d97073bd45f06d39c4f7656009f2cd93a306f7d3d511f0dc.jpg)

![580e0f9b79787662ce75c794af723197a5dc8a31ab47663181390ac2c4072ff0.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/580e0f9b79787662ce75c794af723197a5dc8a31ab47663181390ac2c4072ff0.jpg)

![5e35a18e0679107eb99ac013794b475725e1b3ddeed02626ae234ffa9aabfe5b.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/5e35a18e0679107eb99ac013794b475725e1b3ddeed02626ae234ffa9aabfe5b.jpg)

![6e6860955695493464e8c6c599869910b61871cb6c7b9df7d08e76ec5c6b0969.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/6e6860955695493464e8c6c599869910b61871cb6c7b9df7d08e76ec5c6b0969.jpg)

![76bd702d9ce6ec0bdfc3f6bf7509e1d3bc4b6534c8c2167e2ca9b929d286dc23.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/76bd702d9ce6ec0bdfc3f6bf7509e1d3bc4b6534c8c2167e2ca9b929d286dc23.jpg)

![8c549639d3593ad957e10918362471e4dbf2f199f7bd2abdbd8cafab42743e08.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/8c549639d3593ad957e10918362471e4dbf2f199f7bd2abdbd8cafab42743e08.jpg)

![92a347f612a8ca9553339fe308ea6ba3499e4b14b9c8e7e95665f63e889b92d7.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/92a347f612a8ca9553339fe308ea6ba3499e4b14b9c8e7e95665f63e889b92d7.jpg)

![a69e0515a58f50b395bc07016688cce1b3fbf0246e2b46c2522dbe97aa7d3941.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/a69e0515a58f50b395bc07016688cce1b3fbf0246e2b46c2522dbe97aa7d3941.jpg)

![acabacb684438492a2db5e77725a2264f0554502e759dbb56e5f879848ab63b1.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/acabacb684438492a2db5e77725a2264f0554502e759dbb56e5f879848ab63b1.jpg)

![ad40fa686315e4f9c65897b841cbf25120ff16248ea9c5ad17be2a4156551270.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/ad40fa686315e4f9c65897b841cbf25120ff16248ea9c5ad17be2a4156551270.jpg)

![e51b77a2ac70ed7a94638980eb7d52174d90fde78e931aa4b02d1f2497163f77.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/e51b77a2ac70ed7a94638980eb7d52174d90fde78e931aa4b02d1f2497163f77.jpg)

![eb78a9d7b23ff7b60c009007efac14908cf1e99ed678dbf674f15bd4b267fb64.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/eb78a9d7b23ff7b60c009007efac14908cf1e99ed678dbf674f15bd4b267fb64.jpg)

![ebed2bb532ea1948217a091d69ab97b9231882b55e7d68ae2ab834b39ef1957f.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/ebed2bb532ea1948217a091d69ab97b9231882b55e7d68ae2ab834b39ef1957f.jpg)

![fb8f4ab62a30cbf0c7f28927b33637c12794391c65db346ad58a698c28bcfaff.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/fb8f4ab62a30cbf0c7f28927b33637c12794391c65db346ad58a698c28bcfaff.jpg)

![fe721aae65d8374dad568174b645211b629adc0f33f604a961f501dbf8ff8cc1.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/fe721aae65d8374dad568174b645211b629adc0f33f604a961f501dbf8ff8cc1.jpg)

![ffdeb115ad24c6b0b240a342db81b82b0dcd33183ff573f62bd343e4b3b304e5.jpg](../icml_results/1868_Density-Aware%20Translation%20of%20Spurious%20Correlations%20in%20Zero-Shot%20VLMs/tables/ffdeb115ad24c6b0b240a342db81b82b0dcd33183ff573f62bd343e4b3b304e5.jpg)

<a id="paper-1844"></a>

## No Data? No Problem: Robust Vision-Tabular Learning with Missing Values


### Images

![2fcc4e4db5c8905327af75be43199ea3c6e1a8a4c1d90de080b79beeceeeeb45.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/2fcc4e4db5c8905327af75be43199ea3c6e1a8a4c1d90de080b79beeceeeeb45.jpg)

![3d0b499f383d7f7b3d5a343d3865149770d03172a16fd650455b516db7fc7a7d.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/3d0b499f383d7f7b3d5a343d3865149770d03172a16fd650455b516db7fc7a7d.jpg)

![5cc338584b49417ab7080558ec916202e16cfdbdbe3e180e202d02e156d94ec9.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/5cc338584b49417ab7080558ec916202e16cfdbdbe3e180e202d02e156d94ec9.jpg)

![7e8edf62e21cabe0eb9016145818077a2839bf737bbe344e2dbc639763ac1586.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/7e8edf62e21cabe0eb9016145818077a2839bf737bbe344e2dbc639763ac1586.jpg)

![82e7a69d33269c327db1cf983a6b3c46ec5b48fb923ddcb70e844217087c706c.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/82e7a69d33269c327db1cf983a6b3c46ec5b48fb923ddcb70e844217087c706c.jpg)

![99fae0b3da7986b6a1e6735f39a10eef8764465110b3bc8d80eb7b74f0f2f2d9.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/99fae0b3da7986b6a1e6735f39a10eef8764465110b3bc8d80eb7b74f0f2f2d9.jpg)

![9d7f5cc1486815f8247818ca427426fdf1a808bb67066b9b1b858c898176501a.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/9d7f5cc1486815f8247818ca427426fdf1a808bb67066b9b1b858c898176501a.jpg)

![ce9d680485365e8b946aadd2a352234b455d797effbca8f1d5e3a35151767553.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/ce9d680485365e8b946aadd2a352234b455d797effbca8f1d5e3a35151767553.jpg)

![d1661bb0ef4a3b3cc7f54ad7e70c5f09c4c4ba3f703499941df75349b10f8144.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/d1661bb0ef4a3b3cc7f54ad7e70c5f09c4c4ba3f703499941df75349b10f8144.jpg)

![ff186d79ea0b7c701f0d223d0f5a5b2c54e579393e9af54cebbfbdd9eb3a784a.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/images/ff186d79ea0b7c701f0d223d0f5a5b2c54e579393e9af54cebbfbdd9eb3a784a.jpg)

### Tables

![06a38f27960d96da8899347cf062eeb3ccebe21d0bc5d60da1c85373356c50b4.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/06a38f27960d96da8899347cf062eeb3ccebe21d0bc5d60da1c85373356c50b4.jpg)

![1f50055f699a6e57935ffc2d261bda8cafb689e888b36da78c7df6a7da085468.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/1f50055f699a6e57935ffc2d261bda8cafb689e888b36da78c7df6a7da085468.jpg)

![4126867b952bb6d59c347e40e1a2b334946dbded37e4a5194f03ce2fe28ded8e.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/4126867b952bb6d59c347e40e1a2b334946dbded37e4a5194f03ce2fe28ded8e.jpg)

![766fdee35bf4b11a2e61085ead4250f06826d64e05bec78f85941974943ffec7.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/766fdee35bf4b11a2e61085ead4250f06826d64e05bec78f85941974943ffec7.jpg)

![afbb673f2d6d70c13d2b64a5d4c66d95b5c3e44f772a63699ff5ebbf3ab38983.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/afbb673f2d6d70c13d2b64a5d4c66d95b5c3e44f772a63699ff5ebbf3ab38983.jpg)

![c4a2666f93ab0b173e050307beb31a03f7e49d37ffb6e5b8056603bf839b3340.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/c4a2666f93ab0b173e050307beb31a03f7e49d37ffb6e5b8056603bf839b3340.jpg)

![d711c05d37e61be5a5e3df5e20a701e585d2a7da55d1666d873848ce888475eb.jpg](../icml_results/1869_No%20Data_%20No%20Problem_%20Robust%20Vision-Tabular%20Learning%20with%20Missing%20Values/tables/d711c05d37e61be5a5e3df5e20a701e585d2a7da55d1666d873848ce888475eb.jpg)

<a id="paper-1845"></a>

## Spectral Flow Matching: Stabilizing Stochastic GFlowNets via Frequency-Domain Regularization


### Images

![1059ddb5a61b35ce4e540a568aa8b08037d2087babcb85aa9c402d5347f56809.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/1059ddb5a61b35ce4e540a568aa8b08037d2087babcb85aa9c402d5347f56809.jpg)

![1b85424b56e3a10b1d203cffa8bad4d6e12a9b0103c3ec7f3fb2d39df435163d.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/1b85424b56e3a10b1d203cffa8bad4d6e12a9b0103c3ec7f3fb2d39df435163d.jpg)

![390981c7a1ca9fc140681575c704fd2f4ec0b90202cadfb51441a3bb27fb4a3f.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/390981c7a1ca9fc140681575c704fd2f4ec0b90202cadfb51441a3bb27fb4a3f.jpg)

![7d492c683133cdbbd528f22266ae198a07c2e187a02d8a0bdb4cc46ed9faaaca.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/7d492c683133cdbbd528f22266ae198a07c2e187a02d8a0bdb4cc46ed9faaaca.jpg)

![97acf7b99aabeaafabd5c3e4348ff5faccfcf1b59e7e6ca40fdbd2c88f8a4b99.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/97acf7b99aabeaafabd5c3e4348ff5faccfcf1b59e7e6ca40fdbd2c88f8a4b99.jpg)

![b032f5b3084af4c171ec6796092b125b2870aa07a90728496b2ceb0844cf1c37.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/b032f5b3084af4c171ec6796092b125b2870aa07a90728496b2ceb0844cf1c37.jpg)

![b5f2d15ac6abbf3f598f9db5098ba400e56763289e29682d02eb4f60f20735b0.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/b5f2d15ac6abbf3f598f9db5098ba400e56763289e29682d02eb4f60f20735b0.jpg)

![c98c87e4354aa87b627b7a87665b59350a48eceb2b2b6f0d6e72b2539af026ae.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/c98c87e4354aa87b627b7a87665b59350a48eceb2b2b6f0d6e72b2539af026ae.jpg)

![dd5a5d21698e037f00ee581a33d93df22f5c5cb3a6d6e2527318211c4b39980a.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/images/dd5a5d21698e037f00ee581a33d93df22f5c5cb3a6d6e2527318211c4b39980a.jpg)

### Tables

![1131edd3e73d0cffbe9ab98d0744dcb366e3afe3d74c6398826f3ee6acaf5aa0.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/tables/1131edd3e73d0cffbe9ab98d0744dcb366e3afe3d74c6398826f3ee6acaf5aa0.jpg)

![52165788bbdb561aee24a237b787ff514d14379ecfea110184002e96225f5b13.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/tables/52165788bbdb561aee24a237b787ff514d14379ecfea110184002e96225f5b13.jpg)

![d9f850c453195294a57369afc7cc4079179d5dbc1192e41cfd2d94e14c4b21d8.jpg](../icml_results/1870_Spectral%20Flow%20Matching_%20Stabilizing%20Stochastic%20GFlowNets%20via%20Frequency-Domain%20Regularization/tables/d9f850c453195294a57369afc7cc4079179d5dbc1192e41cfd2d94e14c4b21d8.jpg)

<a id="paper-1846"></a>

## Bottleneck Communication Delay Minimization for Communication-Efficient Decentralized Learning


### Images

![030b5889891088dd70bc5ead71665c5511b0c7565cf6791ab97a35a7b9d24731.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/030b5889891088dd70bc5ead71665c5511b0c7565cf6791ab97a35a7b9d24731.jpg)

![0e20affbe8900ff46606adf6cad995d836605a2ae7bb8500d52f6ff03703f217.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/0e20affbe8900ff46606adf6cad995d836605a2ae7bb8500d52f6ff03703f217.jpg)

![0f52c4f9054224e6a533461422e6fdcff88633a914d94563e1d8a9b4765aecad.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/0f52c4f9054224e6a533461422e6fdcff88633a914d94563e1d8a9b4765aecad.jpg)

![18ceb9705ccac9ff89f92e0af3409f5d9263e262eb986c311b0f1c29f1bac003.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/18ceb9705ccac9ff89f92e0af3409f5d9263e262eb986c311b0f1c29f1bac003.jpg)

![2ec4751121fc289d2c49980af4019d283eb539683ce8b2e71dd09216a6c695c6.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/2ec4751121fc289d2c49980af4019d283eb539683ce8b2e71dd09216a6c695c6.jpg)

![439114ee058175f988164656d8de9e441995591df539e8db059ade44aa37c0d1.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/439114ee058175f988164656d8de9e441995591df539e8db059ade44aa37c0d1.jpg)

![455b402827c5ab17f5f58478cee5ddbb8fbe686333e7be2ff15bc0ba7cbd480a.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/455b402827c5ab17f5f58478cee5ddbb8fbe686333e7be2ff15bc0ba7cbd480a.jpg)

![4b200b2e679b17d51b2824b679c5bc9d97751f920588fc8ca058f3544bcfc3d6.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/4b200b2e679b17d51b2824b679c5bc9d97751f920588fc8ca058f3544bcfc3d6.jpg)

![5ba26d49f95cb9258098e4ab5a8581928c5da036b3b7dde1cbfde78bfd1a4fed.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/5ba26d49f95cb9258098e4ab5a8581928c5da036b3b7dde1cbfde78bfd1a4fed.jpg)

![70e16292664a5743599ea166285b2842e08f3bb8bedce7faaa751e0596ce7420.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/70e16292664a5743599ea166285b2842e08f3bb8bedce7faaa751e0596ce7420.jpg)

![7cfea5a53244236226e460c086eef544ae0fb9f103cc9a445df40b0ecf82913b.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/7cfea5a53244236226e460c086eef544ae0fb9f103cc9a445df40b0ecf82913b.jpg)

![86e79b12e6f2eb64a44245a073dce2e12bed4875b3f384456a5127fff1114bbe.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/86e79b12e6f2eb64a44245a073dce2e12bed4875b3f384456a5127fff1114bbe.jpg)

![8e0f1aacc9cf771322583a5ad5a8e7f1e73cbc181ed2e9d30113226bfb0abc08.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/8e0f1aacc9cf771322583a5ad5a8e7f1e73cbc181ed2e9d30113226bfb0abc08.jpg)

![9d087df1200bed3646c051c5265ab4e75b4320f1e42ba07237dabddf6e70012b.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/9d087df1200bed3646c051c5265ab4e75b4320f1e42ba07237dabddf6e70012b.jpg)

![acad3e4133f234d054cf6f4c735af8bdb65465d1657ae80bdada3ab219f52166.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/acad3e4133f234d054cf6f4c735af8bdb65465d1657ae80bdada3ab219f52166.jpg)

![ae2a7e9a9c7f15eb436affaf7015f8c4af4a7f1ba31df2769eba600983e412bb.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/ae2a7e9a9c7f15eb436affaf7015f8c4af4a7f1ba31df2769eba600983e412bb.jpg)

![ae3243bbecf794ad9c448c10cc92adf87757338ef884c57e18e808dee661a9d1.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/ae3243bbecf794ad9c448c10cc92adf87757338ef884c57e18e808dee661a9d1.jpg)

![b602dff3f5ff913c5bb17c9c826d13606256834d5692886fa635b1fd07c73d2f.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/b602dff3f5ff913c5bb17c9c826d13606256834d5692886fa635b1fd07c73d2f.jpg)

![b663ccf1f34b5ab6972911b5f23cda6b0cf4b7747960e37fb1fd350252d63799.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/b663ccf1f34b5ab6972911b5f23cda6b0cf4b7747960e37fb1fd350252d63799.jpg)

![bdfe50d12b84c1190cedbcde57d4127e1809289f358c6879b099a031598b84d5.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/bdfe50d12b84c1190cedbcde57d4127e1809289f358c6879b099a031598b84d5.jpg)

![ca012547020bd37d878d4bc57b26d8df09c725eeb0747830b9ae44581f758b7d.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/ca012547020bd37d878d4bc57b26d8df09c725eeb0747830b9ae44581f758b7d.jpg)

![ca9384f48f1a383e624d263bae1f444e688ef5597eea165be1c5b7e1257594cf.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/ca9384f48f1a383e624d263bae1f444e688ef5597eea165be1c5b7e1257594cf.jpg)

![cd0445d68b1613f769f991c1c84d78350d45b263659c9fde70d219c0f14989f3.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/cd0445d68b1613f769f991c1c84d78350d45b263659c9fde70d219c0f14989f3.jpg)

![cf3191ef10d77ee8a55a7889b7ab1e1d41c3961049fe066aad64e80aae575abb.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/cf3191ef10d77ee8a55a7889b7ab1e1d41c3961049fe066aad64e80aae575abb.jpg)

![cf7751ab50252d284c94e1031197bbe33d226b977d1d4c2c37fc7c69bd73c9b5.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/cf7751ab50252d284c94e1031197bbe33d226b977d1d4c2c37fc7c69bd73c9b5.jpg)

![dfc1ea936684fb2cd80b67c1c197dccdb02d09013888dd7812c86d056e019562.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/dfc1ea936684fb2cd80b67c1c197dccdb02d09013888dd7812c86d056e019562.jpg)

![e637eae889f3a99572bacb0d1b804a7f846b1cfb10b181272f933b542711778e.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/e637eae889f3a99572bacb0d1b804a7f846b1cfb10b181272f933b542711778e.jpg)

![efd5697ca4550e3c65a1c287f2b089e2bbb167422b7b1be0a75378838f71cd6a.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/efd5697ca4550e3c65a1c287f2b089e2bbb167422b7b1be0a75378838f71cd6a.jpg)

![fd1781c16ada620fc58973a9726979845fd0dc015bdf79b96896e2a0fdffbe13.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/fd1781c16ada620fc58973a9726979845fd0dc015bdf79b96896e2a0fdffbe13.jpg)

![fdc8e17a027807f74f3e0f8bddb6066a814723fb0efd32bab152f35517b06b95.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/images/fdc8e17a027807f74f3e0f8bddb6066a814723fb0efd32bab152f35517b06b95.jpg)

### Tables

![0371c2603d3ebe240521dd40c5c27efa5f42b952adde87f2d733e618ef3bef99.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/tables/0371c2603d3ebe240521dd40c5c27efa5f42b952adde87f2d733e618ef3bef99.jpg)

![06c3ad66b4b385b9475d8d70dcebda84def25a821684e69c93d278f61863e8b1.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/tables/06c3ad66b4b385b9475d8d70dcebda84def25a821684e69c93d278f61863e8b1.jpg)

![54ad21dc6b5c25ec4a34a9de1bb28336afbfee7870b84070a9d292d878a5cb67.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/tables/54ad21dc6b5c25ec4a34a9de1bb28336afbfee7870b84070a9d292d878a5cb67.jpg)

![70ba2424e7e480c239f8f4975ce414b3e0fe34598db946a381abfd62195ebfb4.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/tables/70ba2424e7e480c239f8f4975ce414b3e0fe34598db946a381abfd62195ebfb4.jpg)

![95f75eeeaf2faf0075be3a66e5dc300111ba680b49fd0abcc6d1efc1ee9370e3.jpg](../icml_results/1871_Bottleneck%20Communication%20Delay%20Minimization%20for%20Communication-Efficient%20Decentralized%20Learning/tables/95f75eeeaf2faf0075be3a66e5dc300111ba680b49fd0abcc6d1efc1ee9370e3.jpg)

<a id="paper-1847"></a>

## Effective Distillation to Hybrid xLSTM Architectures


### Images

![00bbbccb5711afdc7b19f3864af6ead659a097e16f65aa8e4d6d605ff4cf5e75.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/00bbbccb5711afdc7b19f3864af6ead659a097e16f65aa8e4d6d605ff4cf5e75.jpg)

![09844da6880871ce881f08a74d91fa42dbaf44f23cc282803bdfe47a69cc8ec8.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/09844da6880871ce881f08a74d91fa42dbaf44f23cc282803bdfe47a69cc8ec8.jpg)

![0d3e106ed57cc108028d1c8cbb4324b19deb2598a78687692b0314a2a82f3114.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/0d3e106ed57cc108028d1c8cbb4324b19deb2598a78687692b0314a2a82f3114.jpg)

![1f6e9d7b9ee3d7e7182fb2803ec9cc3168d041fb36885c654f286a3ac9bbabbf.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/1f6e9d7b9ee3d7e7182fb2803ec9cc3168d041fb36885c654f286a3ac9bbabbf.jpg)

![2670ab2e5c49390af2d2c689a2ce0534988526baef84c3ffe76a539d0b8492c5.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/2670ab2e5c49390af2d2c689a2ce0534988526baef84c3ffe76a539d0b8492c5.jpg)

![320d5a26b8b4da72ca396452acbc78aca96988e5ba1b6ccb8443ce5712da9f2e.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/320d5a26b8b4da72ca396452acbc78aca96988e5ba1b6ccb8443ce5712da9f2e.jpg)

![3a9e9be3ac78f84dcf51946039739acf445a350f62d3692728c3791c38f90bf8.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/3a9e9be3ac78f84dcf51946039739acf445a350f62d3692728c3791c38f90bf8.jpg)

![4550ff254fdd91725afad2cdcd186ab3d2b0b24c234145b7f9f0ed25ecbb4648.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/4550ff254fdd91725afad2cdcd186ab3d2b0b24c234145b7f9f0ed25ecbb4648.jpg)

![51f63dd9071e6cd8efe3211af8b92c7508004e52b4b5503f8d0f0f44463dfaa4.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/51f63dd9071e6cd8efe3211af8b92c7508004e52b4b5503f8d0f0f44463dfaa4.jpg)

![5284bcab5195fc2843b9ebb5994771840aee11dc380451477f875250dd212055.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/5284bcab5195fc2843b9ebb5994771840aee11dc380451477f875250dd212055.jpg)

![5e1c7eeaac6d5d71ff5bfe58384c1647b21bee419dc0a8d3561949ea6f69fd59.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/5e1c7eeaac6d5d71ff5bfe58384c1647b21bee419dc0a8d3561949ea6f69fd59.jpg)

![619be2333dd880d9b17f00239bc1b419396cac1067ce91e42fd847d92cdb932e.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/619be2333dd880d9b17f00239bc1b419396cac1067ce91e42fd847d92cdb932e.jpg)

![77915a61dadd46045d5725eaacfb7b2d566c46fccf93fa432cb3228a5db61517.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/77915a61dadd46045d5725eaacfb7b2d566c46fccf93fa432cb3228a5db61517.jpg)

![7f74810a3cc7c102217555cfdc7b07f900c583c1e5511219a6828d95cde2b177.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/7f74810a3cc7c102217555cfdc7b07f900c583c1e5511219a6828d95cde2b177.jpg)

![8c1b69b2e4aa6eff64372097b497f7f2fba51d923fb5e79bbe9f298cdf03cd38.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/8c1b69b2e4aa6eff64372097b497f7f2fba51d923fb5e79bbe9f298cdf03cd38.jpg)

![900479ac73dffd6971c66baab1667389091f5ecdc318cba83167a1156101c6ae.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/900479ac73dffd6971c66baab1667389091f5ecdc318cba83167a1156101c6ae.jpg)

![d72ddcbcb1d33897f406c654eb5b2cfe596788df558fcf4a39761d9d4c6132d4.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/d72ddcbcb1d33897f406c654eb5b2cfe596788df558fcf4a39761d9d4c6132d4.jpg)

![e4418cb6b2f8992aea13ced6b2216b1c3b72733747d89b6c7aa24854ababa98b.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/e4418cb6b2f8992aea13ced6b2216b1c3b72733747d89b6c7aa24854ababa98b.jpg)

![fb35c09470ddbf96bd93c83cf30ebf7ca556a9b8dbb807dd6762773f083ae6ba.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/images/fb35c09470ddbf96bd93c83cf30ebf7ca556a9b8dbb807dd6762773f083ae6ba.jpg)

### Tables

![07788b0c8e350bac74e709abfb7e7e8e226258de574aa1808debb4d525f79a8f.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/07788b0c8e350bac74e709abfb7e7e8e226258de574aa1808debb4d525f79a8f.jpg)

![46b95bdd49a0470e149eefb9fd91ce8a80e7f7d99597a3f07c3299cbff4dddf0.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/46b95bdd49a0470e149eefb9fd91ce8a80e7f7d99597a3f07c3299cbff4dddf0.jpg)

![48ac8dccbcbccb6a2c2559254a37fcc567e3b3f1469483927bdee290c932e3a6.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/48ac8dccbcbccb6a2c2559254a37fcc567e3b3f1469483927bdee290c932e3a6.jpg)

![86b57afb088bf8c79c5f7c230a8241cd16cdc5b385c1d0a8ae1037ba15caf9b8.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/86b57afb088bf8c79c5f7c230a8241cd16cdc5b385c1d0a8ae1037ba15caf9b8.jpg)

![b10ed4b6b371507a208e75943ed3ab4a1a83865369ffe81ea38582ff834fda66.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/b10ed4b6b371507a208e75943ed3ab4a1a83865369ffe81ea38582ff834fda66.jpg)

![b49bc9e4a6447028992ec0ea1e5bc05976b6b5f000eb2e6d4e9760773040872f.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/b49bc9e4a6447028992ec0ea1e5bc05976b6b5f000eb2e6d4e9760773040872f.jpg)

![ca06e9e64c89baca4f0f99d720fea5d9dfb707c0597c5b9603134bbae6ec51b0.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/ca06e9e64c89baca4f0f99d720fea5d9dfb707c0597c5b9603134bbae6ec51b0.jpg)

![e34422ed44589eca2dfceae1a43aaaa3b381f17eb2be83ede1e584f4a543c6e2.jpg](../icml_results/1872_Effective%20Distillation%20to%20Hybrid%20xLSTM%20Architectures/tables/e34422ed44589eca2dfceae1a43aaaa3b381f17eb2be83ede1e584f4a543c6e2.jpg)

<a id="paper-1848"></a>

## The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning

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
