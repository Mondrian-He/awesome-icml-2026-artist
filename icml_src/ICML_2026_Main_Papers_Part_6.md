# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 6 of 197

## 目录 (Table of Contents)

1. [Control Consistency Losses for Diffusion Bridges](#paper-0166)
2. [The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning](#paper-0167)
3. [Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces](#paper-0168)
4. [From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](#paper-0169)
5. [ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios](#paper-0170)
6. [Rapid Poison: Practical Poisoning Attacks Against the Rapid Response Framework](#paper-0171)
7. [EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models](#paper-0172)
8. [Thinking in Flow: A Dissipative Stabilization Operator for Robust Autoregressive Reasoning](#paper-0173)
9. [Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels](#paper-0174)
10. [Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](#paper-0175)
11. [Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](#paper-0176)
12. [Copyright-Bench: Agentic Evaluation of Copyright Law Compliance](#paper-0177)
13. [Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts](#paper-0178)
14. [Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](#paper-0179)
15. [DiScoFormer: Plug-In Density and Score Estimation with Transformers](#paper-0180)
16. [Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions](#paper-0181)
17. [Non-Euclidean Gradient Descent Operates at the Edge of Stability](#paper-0182)
18. [h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning](#paper-0183)
19. [Learning Unmasking Policies for Diffusion Language Models](#paper-0184)
20. [Efficient Diffusion Models under Nonconvex Equality and Inequality constraints via Landing](#paper-0185)
21. [OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning](#paper-0186)
22. [Manifold-Optimal Guidance: A Unified Riemannian Control View of Diffusion Guidance](#paper-0187)
23. [Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](#paper-0188)
24. [Incremental BPE Tokenization](#paper-0189)
25. [Procedural Pretraining: Warming Up Language Models with Abstract Data](#paper-0190)
26. [VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs](#paper-0191)
27. [Nash Equilibria in Games with Playerwise Concave Coupling Constraints: Existence and Computation](#paper-0192)
28. [The Signal is in the Steps: Local Scoring for Reasoning Data Selection](#paper-0193)
29. [Overcoming PINNs Failure Modes In High Dimension With Low-Rank Fourier Sum](#paper-0194)
30. [Real-Time Visual Attribution Streaming in Thinking Model](#paper-0195)
31. [ThunderAgent: A Fast, Simple, and Program-Aware Agentic Inference System](#paper-0196)
32. [Symmetry in language statistics shapes the geometry of model representations](#paper-0197)
33. [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](#paper-0198)

---


<a id="paper-0166"></a>

## Control Consistency Losses for Diffusion Bridges

### Images

![00a522731012cd772682f9a14a24e9fc74abc451f77853832ba4b8e94ad4f24d.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/00a522731012cd772682f9a14a24e9fc74abc451f77853832ba4b8e94ad4f24d.jpg)

![1e1676b6010e10ebe6f6ee31fd29c2886e0a9d46d34b93693025ed3961c948f0.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/1e1676b6010e10ebe6f6ee31fd29c2886e0a9d46d34b93693025ed3961c948f0.jpg)

![2025430ab85fcfa6a4efc6e2565ab250109cacb0b9cc00654e09339833cb34b2.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/2025430ab85fcfa6a4efc6e2565ab250109cacb0b9cc00654e09339833cb34b2.jpg)

![25de4396c31dc71f72a64b98032c9597707e2ae9988e48d1cb19ebb3eb7595ce.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/25de4396c31dc71f72a64b98032c9597707e2ae9988e48d1cb19ebb3eb7595ce.jpg)

![4b3b7ab5a07f0d476ffc6f0c0f52ad19177724f67590e44dda961de7a58e6e81.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/4b3b7ab5a07f0d476ffc6f0c0f52ad19177724f67590e44dda961de7a58e6e81.jpg)

![637aae2d8e7bf04d0a0356edd79efc17c0935a6b31180c92fb2f37e9c99d347e.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/637aae2d8e7bf04d0a0356edd79efc17c0935a6b31180c92fb2f37e9c99d347e.jpg)

![6d092a7c5e1252adb7f3594db948cf309084e64418bcbae3458537b4177c3d5e.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/6d092a7c5e1252adb7f3594db948cf309084e64418bcbae3458537b4177c3d5e.jpg)

![72725462d621ad031be50f40e3d07a58ba0692994bf143e3f12bf91fd5218864.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/72725462d621ad031be50f40e3d07a58ba0692994bf143e3f12bf91fd5218864.jpg)

![7cab15abedc3c3d9dfdcca6f38821997f4724466f531971b4b0059058c0413ee.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/7cab15abedc3c3d9dfdcca6f38821997f4724466f531971b4b0059058c0413ee.jpg)

![847d1964a8c14da84d1609f0ab7b02a6deaebfa188b982b4f4b94d7ef628b84f.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/847d1964a8c14da84d1609f0ab7b02a6deaebfa188b982b4f4b94d7ef628b84f.jpg)

![86e107732de741472421e56d290ecbb1fc68000151e04c7bdadb6cd68792a7b5.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/86e107732de741472421e56d290ecbb1fc68000151e04c7bdadb6cd68792a7b5.jpg)

![947856f54defa3f253d83692c3600aa85e3af96aae0db53c35d6d6b8849298fc.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/947856f54defa3f253d83692c3600aa85e3af96aae0db53c35d6d6b8849298fc.jpg)

![b4f6b1ceff97e0f2515dd2aac4e06d4c6e23be8760ce1d0b8fff5c2f6a74e241.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/b4f6b1ceff97e0f2515dd2aac4e06d4c6e23be8760ce1d0b8fff5c2f6a74e241.jpg)

![be5cf6dcc507de6a4dcc0049d8f58f4be52c856499e2a1369dc4186dfaf07594.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/be5cf6dcc507de6a4dcc0049d8f58f4be52c856499e2a1369dc4186dfaf07594.jpg)

![c6db180bf3e1310d8a84d926914f118bd83a7dc0349d3304414309d6dadb115d.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/c6db180bf3e1310d8a84d926914f118bd83a7dc0349d3304414309d6dadb115d.jpg)

![c919937b6f22c3134d77ea4a9294d02dd16362f4d5ebfed27e72287b3cccb6ba.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/c919937b6f22c3134d77ea4a9294d02dd16362f4d5ebfed27e72287b3cccb6ba.jpg)

![daa307217e365138687910fbad6eb5f615a2651da7e327cce0a3c49153aabb20.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/daa307217e365138687910fbad6eb5f615a2651da7e327cce0a3c49153aabb20.jpg)

![eb2eb3da0e78c3d19775bd154a39fc8d8895f2015b93674a92d94db5915d1df7.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/eb2eb3da0e78c3d19775bd154a39fc8d8895f2015b93674a92d94db5915d1df7.jpg)

![f59b25a1c2101bb18cf4f1207fb3a69f749cca37e3b06df2292c2ccb76f70c13.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/f59b25a1c2101bb18cf4f1207fb3a69f749cca37e3b06df2292c2ccb76f70c13.jpg)

![faa62ffcdb6cd0175cd190750206f1ad9efebf8b6392a99f1c12dbcb796f47ed.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/faa62ffcdb6cd0175cd190750206f1ad9efebf8b6392a99f1c12dbcb796f47ed.jpg)

![ff491ed9a1d0800ecf15457763017c1ac13fca4499a0b5ec5aab2892e8a6e846.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/ff491ed9a1d0800ecf15457763017c1ac13fca4499a0b5ec5aab2892e8a6e846.jpg)

### Tables

![3610ff7f0d2d8f22f15fdbfa42aced6fbd2fd47f274a2cdbdafa4d79f1be0056.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/3610ff7f0d2d8f22f15fdbfa42aced6fbd2fd47f274a2cdbdafa4d79f1be0056.jpg)

![3ecde2a8cf1bde448dac3aff6e05db2ade3ac8e8c31d404cf5e072dd967d9919.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/3ecde2a8cf1bde448dac3aff6e05db2ade3ac8e8c31d404cf5e072dd967d9919.jpg)

![43c8ed58f9da7a5651f9768c9b1238c2ec46fc1a8242ccecdc8a3717ed6b6185.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/43c8ed58f9da7a5651f9768c9b1238c2ec46fc1a8242ccecdc8a3717ed6b6185.jpg)

![44389b84efe1f2c5ec0ae843ba702fc8e0635b4216e8c5bba62e136eedc71619.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/44389b84efe1f2c5ec0ae843ba702fc8e0635b4216e8c5bba62e136eedc71619.jpg)

![a96249a7c7b78d6bb92e20c7c1b80ad8a54bae42b4da04e4cf999489bc03ace3.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/a96249a7c7b78d6bb92e20c7c1b80ad8a54bae42b4da04e4cf999489bc03ace3.jpg)

![c1a4b7e215fb3cb3c32c2ed5c9327a2bb4a5431d27e970cce7f477634c368e44.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/c1a4b7e215fb3cb3c32c2ed5c9327a2bb4a5431d27e970cce7f477634c368e44.jpg)

![c63fc48796585c5da743c0c4a8e40ed40f8f90c8f5b6cfd8273a72abdca23615.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/c63fc48796585c5da743c0c4a8e40ed40f8f90c8f5b6cfd8273a72abdca23615.jpg)

![e1a841fa39325acc983d73317ccf7cd9d43e0de570fffa2b412583d37413a940.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/e1a841fa39325acc983d73317ccf7cd9d43e0de570fffa2b412583d37413a940.jpg)

![f5e01baac9a07cda73989bb9a1a1e67cb93d63c4e28c978f7ea940e33acd5252.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/f5e01baac9a07cda73989bb9a1a1e67cb93d63c4e28c978f7ea940e33acd5252.jpg)

## Control Consistency Losses for Diffusion Bridges


### Images

![32a1735cfd54fb0b6719a71dc0ca2595e5a46ee1c30e73e4c09c1cf8ab875a6b.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/32a1735cfd54fb0b6719a71dc0ca2595e5a46ee1c30e73e4c09c1cf8ab875a6b.jpg)

![50aefef9151dbc5518f44d58bbe392d76c71adbe0b5ef524ec1e8316a9ce0d48.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/50aefef9151dbc5518f44d58bbe392d76c71adbe0b5ef524ec1e8316a9ce0d48.jpg)

![53cc56557bdc85c319b6fe3e204d51028734b90820e9437a6badf09267745189.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/53cc56557bdc85c319b6fe3e204d51028734b90820e9437a6badf09267745189.jpg)

![547b168295c197cf4d3d95dcc2f9bb1ab4e7658d9ba76685e62951f79501f18d.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/547b168295c197cf4d3d95dcc2f9bb1ab4e7658d9ba76685e62951f79501f18d.jpg)

![5684c481a8eb2024c63c308b221278a0281735f00508cd4a045f818ce4b362ae.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/5684c481a8eb2024c63c308b221278a0281735f00508cd4a045f818ce4b362ae.jpg)

![59f313830a34a99dcb5da016198f1a6e7e2b289796974ffb4b9459c4807f4c72.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/59f313830a34a99dcb5da016198f1a6e7e2b289796974ffb4b9459c4807f4c72.jpg)

![8480b6fa0be40c5afcff1309765a3bd93609d78ea192a1804c7b52b9e096b4fe.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/8480b6fa0be40c5afcff1309765a3bd93609d78ea192a1804c7b52b9e096b4fe.jpg)

![aaf88d97ed1cf97c70cc646571070fadb49cfbd95a73fec132bccfa7d0adb89c.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/aaf88d97ed1cf97c70cc646571070fadb49cfbd95a73fec132bccfa7d0adb89c.jpg)

![aec349c2229aa06167b9285627a0d45dc2f1419fe5844def41c192a460f9e215.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/aec349c2229aa06167b9285627a0d45dc2f1419fe5844def41c192a460f9e215.jpg)

![b58cd78b589231b366e481dee8af9703da83195e268a445f45a4cb8a90bcab76.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/b58cd78b589231b366e481dee8af9703da83195e268a445f45a4cb8a90bcab76.jpg)

![de847d4252e77d54f2a66a70dc9e6c44d1d3b7dfc7721673e813cc3de995583a.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/de847d4252e77d54f2a66a70dc9e6c44d1d3b7dfc7721673e813cc3de995583a.jpg)

![e04d65244bc8d155703d0379390aa95fb7db4885c59076cc007cac5e50e68e75.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/images/e04d65244bc8d155703d0379390aa95fb7db4885c59076cc007cac5e50e68e75.jpg)

### Tables

![174f0af6a2806b8771261befc618916ed84012f4464ad5e87a0053695a92a6a9.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/174f0af6a2806b8771261befc618916ed84012f4464ad5e87a0053695a92a6a9.jpg)

![1e77e2826f0725a0ca81b346e56e3d9ea7f1df1fae3307d94a3ecfa29c621893.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/1e77e2826f0725a0ca81b346e56e3d9ea7f1df1fae3307d94a3ecfa29c621893.jpg)

![6eb6025fa52f6ec12976fc0b525ffcbd459626da49008768f0c3fda6bd55b5c9.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/6eb6025fa52f6ec12976fc0b525ffcbd459626da49008768f0c3fda6bd55b5c9.jpg)

![8e5b6429fa4588a081d220442a82d4a940e536e56da8c2d6188cd7ac314019f5.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/8e5b6429fa4588a081d220442a82d4a940e536e56da8c2d6188cd7ac314019f5.jpg)

![8fda9ff8f8c0a5729c600eaaa41ab8a3c6ff2836bc1f9862331e24c59e7b48f2.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/8fda9ff8f8c0a5729c600eaaa41ab8a3c6ff2836bc1f9862331e24c59e7b48f2.jpg)

![9ceac2d3b6bfa77c5572a8ae0e36c570b8b456101cb760924d9107ad12bf29b7.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/9ceac2d3b6bfa77c5572a8ae0e36c570b8b456101cb760924d9107ad12bf29b7.jpg)

![bab36c2f19b0399a8b0ece86f5914b0ce5dd51ed9b73ef533e19ed8986f24fbf.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/bab36c2f19b0399a8b0ece86f5914b0ce5dd51ed9b73ef533e19ed8986f24fbf.jpg)

![c4ac5b1320e211464904dc71d3ab16d4fefad5d53df71640101e43c79628dfdc.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/c4ac5b1320e211464904dc71d3ab16d4fefad5d53df71640101e43c79628dfdc.jpg)

![e17803351dfedc496c1b2cf4f32e0667b4301e8ee5d5b0e5104d1ee5dc7df09b.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/e17803351dfedc496c1b2cf4f32e0667b4301e8ee5d5b0e5104d1ee5dc7df09b.jpg)

![e94e2cad8be1347bc4875c9347e6a7712cb380a84edd038c8dbe7972dd946ebc.jpg](../icml_results/168_Control%20Consistency%20Losses%20for%20Diffusion%20Bridges/tables/e94e2cad8be1347bc4875c9347e6a7712cb380a84edd038c8dbe7972dd946ebc.jpg)

<a id="paper-0167"></a>

## The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning


### Images

![037aa71b9c62de6dfd2d223126bd1c85dc8b8c3453085226b7083a52e7e20fb8.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/037aa71b9c62de6dfd2d223126bd1c85dc8b8c3453085226b7083a52e7e20fb8.jpg)

![21554a9ef243d07ae4755e70d7e1a6906862aa4bf2be607fd965d8fe969ddaec.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/21554a9ef243d07ae4755e70d7e1a6906862aa4bf2be607fd965d8fe969ddaec.jpg)

![2a834d33744e23f69cc73253db88d71e67b4a929cf538afcec87461787dc1aae.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/2a834d33744e23f69cc73253db88d71e67b4a929cf538afcec87461787dc1aae.jpg)

![2f2ffe4b0905960ca80175e54767f68678597dfdea9b7703084e1a7a857b6918.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/2f2ffe4b0905960ca80175e54767f68678597dfdea9b7703084e1a7a857b6918.jpg)

![3cc06b426c0c940be73ff176b496af7d85e8a0108942bf05c440b99d751ae25d.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/3cc06b426c0c940be73ff176b496af7d85e8a0108942bf05c440b99d751ae25d.jpg)

![457b845b8f7bfa8d225f7a6859156d83549cc9e6d7957bc598f39dc1947e8448.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/457b845b8f7bfa8d225f7a6859156d83549cc9e6d7957bc598f39dc1947e8448.jpg)

![4d7f3b74f941fb2f29560d1f75e67d3feb356b384c2d9d6ed52f20314eea1c08.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/4d7f3b74f941fb2f29560d1f75e67d3feb356b384c2d9d6ed52f20314eea1c08.jpg)

![568c9c26c535e715307439837e491f4d680a15561abc58021cc3ef76a66c3749.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/568c9c26c535e715307439837e491f4d680a15561abc58021cc3ef76a66c3749.jpg)

![71a440eb2a163e556eb18c7da843644ca3c393095f9c4b97bedddd29a264154d.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/71a440eb2a163e556eb18c7da843644ca3c393095f9c4b97bedddd29a264154d.jpg)

![7773daddf920ad97e71538d76266812fb6797c3ab5373f6214f6f43203ff30d5.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/7773daddf920ad97e71538d76266812fb6797c3ab5373f6214f6f43203ff30d5.jpg)

![80bf7a1f24628faa19a080922d8b952905691a986a2324dc9ffd03ff3d2ee93d.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/80bf7a1f24628faa19a080922d8b952905691a986a2324dc9ffd03ff3d2ee93d.jpg)

![83fcbe85d83f6f5439aa63809410fc40bb10a7eb43686f7ecb6fdbdab6803365.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/83fcbe85d83f6f5439aa63809410fc40bb10a7eb43686f7ecb6fdbdab6803365.jpg)

![8dc5d6cc04da53f5805ffea6315c300b79c221a896edbca980e3dd9a9b487a68.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/8dc5d6cc04da53f5805ffea6315c300b79c221a896edbca980e3dd9a9b487a68.jpg)

![a2cb5dbca5c5c50d8fcf6e523580c714ac68284fa5b2e2159def9462b8337e50.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/a2cb5dbca5c5c50d8fcf6e523580c714ac68284fa5b2e2159def9462b8337e50.jpg)

![aaf73feae65b0753ad15a06ed940c8b5a779e4fe519498dd9ff490684758e483.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/aaf73feae65b0753ad15a06ed940c8b5a779e4fe519498dd9ff490684758e483.jpg)

![aaf7ec39c777aaf32b02f0f93e7755c076f1e85493e55a9eaec5fec9a27e28e2.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/aaf7ec39c777aaf32b02f0f93e7755c076f1e85493e55a9eaec5fec9a27e28e2.jpg)

![b0c6ecaf8d6a4ec0bced36f6fd7fe7ad98c6e5c18f08ad4c56dc21667b3be565.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/b0c6ecaf8d6a4ec0bced36f6fd7fe7ad98c6e5c18f08ad4c56dc21667b3be565.jpg)

![bd7ceec0909c2a51bca049df6bb872e0b9f49b794bdc94731370584be89d5eae.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/bd7ceec0909c2a51bca049df6bb872e0b9f49b794bdc94731370584be89d5eae.jpg)

![c663c9ad863e76a46b9122ab77e090c4ae6bdc0abf41edb74234b3559275118f.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/c663c9ad863e76a46b9122ab77e090c4ae6bdc0abf41edb74234b3559275118f.jpg)

![c6965dadcaf271df7945fc571b38063ce69cb97c8a374056d0373d5dfaf0a57f.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/c6965dadcaf271df7945fc571b38063ce69cb97c8a374056d0373d5dfaf0a57f.jpg)

![daa51761dfcd9f4a4eeac598775948e6279199a52273d901a77ac9ebafe45621.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/daa51761dfcd9f4a4eeac598775948e6279199a52273d901a77ac9ebafe45621.jpg)

![e7c2e7be75e5ad894a526e7af31b42ff455e384c5812bd65201db0a6f898863a.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/e7c2e7be75e5ad894a526e7af31b42ff455e384c5812bd65201db0a6f898863a.jpg)

![e874ceb0deb981195215dc8471d63bdca1ac80430148f0d14c6357c77d472771.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/images/e874ceb0deb981195215dc8471d63bdca1ac80430148f0d14c6357c77d472771.jpg)

### Tables

![1cdf75bbcdd1b4db3248fe778d8e1902c5ddffc218781fad375d961eeb40f38c.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/1cdf75bbcdd1b4db3248fe778d8e1902c5ddffc218781fad375d961eeb40f38c.jpg)

![356ad67c0a2adb16bf59fcf46f9cadfaf5b29b1a9a8cb7060aa708a7be8ee67c.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/356ad67c0a2adb16bf59fcf46f9cadfaf5b29b1a9a8cb7060aa708a7be8ee67c.jpg)

![44560b9bb9da03eac10c77ceebc4bd50f4921ea51e2f89b2e6cf1b9f28f4ea34.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/44560b9bb9da03eac10c77ceebc4bd50f4921ea51e2f89b2e6cf1b9f28f4ea34.jpg)

![49449da1e9c58cc8fe0bd42f628e0d5ffac622ab821a9d48297e140eb643d43d.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/49449da1e9c58cc8fe0bd42f628e0d5ffac622ab821a9d48297e140eb643d43d.jpg)

![6ce2cdd1fe9663bb71f341739e79c5558c99caadc99fd13459af4e947d099d06.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/6ce2cdd1fe9663bb71f341739e79c5558c99caadc99fd13459af4e947d099d06.jpg)

![7c20fd8db6fde5ff909127c306809844f6d59217f4b96c9ce19a2c128c17cf87.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/7c20fd8db6fde5ff909127c306809844f6d59217f4b96c9ce19a2c128c17cf87.jpg)

![fddc87bc92bcee5934cf0e12c14cd55a91f4aa8e3ab3488bd958ad22db37bf5e.jpg](../icml_results/169_The%20Double-Edged%20Nature%20of%20the%20Rashomon%20Set%20for%20Trustworthy%20Machine%20Learning/tables/fddc87bc92bcee5934cf0e12c14cd55a91f4aa8e3ab3488bd958ad22db37bf5e.jpg)

<a id="paper-0168"></a>

## Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces


### Images

![1f9b7ce7f3721873c723c92e7a1d78aee2717de9d197cc89f1f2e7fc035c387d.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/1f9b7ce7f3721873c723c92e7a1d78aee2717de9d197cc89f1f2e7fc035c387d.jpg)

![6b1b66ad2b0412f69783c6309804af746a26c1266e3511857c1c9a9959bda3a3.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/6b1b66ad2b0412f69783c6309804af746a26c1266e3511857c1c9a9959bda3a3.jpg)

![723c1c751d80a376123bc738ebf05ede2fcaa8b8a6efe9b9b9c38d77fb799f9a.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/723c1c751d80a376123bc738ebf05ede2fcaa8b8a6efe9b9b9c38d77fb799f9a.jpg)

![99388b2d2febcf8e6ac2b0129b379384fe03529aba24c00bcb8cac78c640e6fa.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/99388b2d2febcf8e6ac2b0129b379384fe03529aba24c00bcb8cac78c640e6fa.jpg)

![ab98d00c32608638086f4d7698c8b4eb9c2f5499488875d493d065e4a81eeab5.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/ab98d00c32608638086f4d7698c8b4eb9c2f5499488875d493d065e4a81eeab5.jpg)

![af1372947ed1d589397002bce68c54f2134f97273f1290d157e385a3c6225f9b.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/af1372947ed1d589397002bce68c54f2134f97273f1290d157e385a3c6225f9b.jpg)

![b254d4df4a391e6340e0cfc98c3a8d4c15de1885bdccb38ac56fec424245363a.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/b254d4df4a391e6340e0cfc98c3a8d4c15de1885bdccb38ac56fec424245363a.jpg)

![d3fdd4aea69c744933d2ddc167f53a09012ee2d4aacec523a10c5491a23f9c5a.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/d3fdd4aea69c744933d2ddc167f53a09012ee2d4aacec523a10c5491a23f9c5a.jpg)

![d66182ed7190350ccfcaa6fd1d278bb09bc2b0dcc592996b6482fa0a20f79409.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/d66182ed7190350ccfcaa6fd1d278bb09bc2b0dcc592996b6482fa0a20f79409.jpg)

![edc1e078d87aaa97a938d910b076e35afd51bd3a4f9657de5fcdef7722656054.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/edc1e078d87aaa97a938d910b076e35afd51bd3a4f9657de5fcdef7722656054.jpg)

![fd4a5aa10626553a0b3412073c7587c3d7c94c38be91d14d7754a124cde4819e.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/images/fd4a5aa10626553a0b3412073c7587c3d7c94c38be91d14d7754a124cde4819e.jpg)

### Tables

![6bfe172fefa0c15a51bfaa9a2b11b64390a37407ac17a860aa96c1eea1dbbb4d.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/6bfe172fefa0c15a51bfaa9a2b11b64390a37407ac17a860aa96c1eea1dbbb4d.jpg)

![9621ea1575b5cc8e99cfb28f8fd5583fb96e6e2f37cb9877941b4329915fbca8.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/9621ea1575b5cc8e99cfb28f8fd5583fb96e6e2f37cb9877941b4329915fbca8.jpg)

![a2c9dc6aaedc4821bfd4313bcea915a3a8fa9486dd912d9a82f6315c63216a6a.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/a2c9dc6aaedc4821bfd4313bcea915a3a8fa9486dd912d9a82f6315c63216a6a.jpg)

![c56669a97c3774a7669710246f2068abdb9697a57f62dfca1ad7de2242804b75.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/c56669a97c3774a7669710246f2068abdb9697a57f62dfca1ad7de2242804b75.jpg)

![e75c8c5c4562f0442d69b307c4339bc0d9a842c69d1d9aa4161dea9d3227b65c.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/e75c8c5c4562f0442d69b307c4339bc0d9a842c69d1d9aa4161dea9d3227b65c.jpg)

![f52384cccea9df500f9ffbcba380677f22ae6bcd5d9ba7847742eaadee8d9cba.jpg](../icml_results/170_Score-Repellent%20Monte%20Carlo_%20Toward%20Efficient%20Non-Markovian%20Sampler%20with%20Constant%20Memory%20in%20General%20/tables/f52384cccea9df500f9ffbcba380677f22ae6bcd5d9ba7847742eaadee8d9cba.jpg)

<a id="paper-0169"></a>

## From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model


### Images

![2a2fff0cb226260d45804ca730d0efeb4b05294827542ccc0cb13c0862ced2e1.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/2a2fff0cb226260d45804ca730d0efeb4b05294827542ccc0cb13c0862ced2e1.jpg)

![4633364c7f1d5afe2965400ca2735a6c784a4ce2657f28e096a0f8ccbefbc596.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/4633364c7f1d5afe2965400ca2735a6c784a4ce2657f28e096a0f8ccbefbc596.jpg)

![4a19d48f87833371499dec715b1242d5c1c7e1c7ffc1643047b2c17fdeac7a27.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/4a19d48f87833371499dec715b1242d5c1c7e1c7ffc1643047b2c17fdeac7a27.jpg)

![65e085e8c6fbcce07cc95442ca7617309c3472f702a8b5e9e58899305a61efd0.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/65e085e8c6fbcce07cc95442ca7617309c3472f702a8b5e9e58899305a61efd0.jpg)

![77bd4856a3b12c7d25f7d441f4121d87f644b508d022e21041b7c070545baa88.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/77bd4856a3b12c7d25f7d441f4121d87f644b508d022e21041b7c070545baa88.jpg)

![b0581038926e47af9f1606dc3a06e25539aac96ea4778d126cd639947902a468.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/b0581038926e47af9f1606dc3a06e25539aac96ea4778d126cd639947902a468.jpg)

![b05cbda4c7adcfee805c1a6b23bbecf02eb3332f270d4bd2a2ce4b8a61970e1c.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/b05cbda4c7adcfee805c1a6b23bbecf02eb3332f270d4bd2a2ce4b8a61970e1c.jpg)

![ca6788bc0c1bc02b263bc4bc6544d94e2668511b6671e688e3e68de6f2d9fba8.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/ca6788bc0c1bc02b263bc4bc6544d94e2668511b6671e688e3e68de6f2d9fba8.jpg)

![ccd3782944ea7a521881311adee5fb0b74bf5fa4b58efb84759090ff46115854.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/images/ccd3782944ea7a521881311adee5fb0b74bf5fa4b58efb84759090ff46115854.jpg)

### Tables

![514da042402dc7155a175aa213770e02de6e19d45b6a0b809e9780bb2fcf8d24.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/514da042402dc7155a175aa213770e02de6e19d45b6a0b809e9780bb2fcf8d24.jpg)

![70009d3501b6c043f8562715fffbce09fc3cf7b5aea37030196e0abd8cf84881.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/70009d3501b6c043f8562715fffbce09fc3cf7b5aea37030196e0abd8cf84881.jpg)

![8da4bdd422d3457781837c1b76abd957ff655e1240c4cd371dbe6191876a7c45.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/8da4bdd422d3457781837c1b76abd957ff655e1240c4cd371dbe6191876a7c45.jpg)

![d4ace02c9ce4b2c9b1367c9a93b40fc6fd8d9a856b280274bc11fea1c1c6f11f.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/d4ace02c9ce4b2c9b1367c9a93b40fc6fd8d9a856b280274bc11fea1c1c6f11f.jpg)

![d4e04948eb12ce570db404269ea14e2cc3b34c298b0714708de6e857c58ca65e.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/d4e04948eb12ce570db404269ea14e2cc3b34c298b0714708de6e857c58ca65e.jpg)

![d7bb76a77cb9f196ce022b0921f6fe0da05eb27545cd125def7dea6a5471d943.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/d7bb76a77cb9f196ce022b0921f6fe0da05eb27545cd125def7dea6a5471d943.jpg)

![f4772dcfe13a68dbe64d99032ebce5ba89539126354052b29cc1a94cae12a310.jpg](../icml_results/171_From%20Abstraction%20to%20Instantiation_%20Learning%20Behavioral%20Representation%20for%20Vision-Language-Action%20Mod/tables/f4772dcfe13a68dbe64d99032ebce5ba89539126354052b29cc1a94cae12a310.jpg)

<a id="paper-0170"></a>

## ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios


### Images

![1a72ced0f838982ca994cdbf3331f57b2dec0ced2d4dab478ac51330858864ba.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/1a72ced0f838982ca994cdbf3331f57b2dec0ced2d4dab478ac51330858864ba.jpg)

![23ddeb98455f89dcbad7f6eed4aafa8d04e16904b130b08a60570ec7c7fce8d4.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/23ddeb98455f89dcbad7f6eed4aafa8d04e16904b130b08a60570ec7c7fce8d4.jpg)

![2969e8a4a16d797742940d4f97bb13485742061d3046c95de7eacaeb7f15b061.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/2969e8a4a16d797742940d4f97bb13485742061d3046c95de7eacaeb7f15b061.jpg)

![65a0866c6cca323638f1265be5df79969009bd53154d4aaaa7a9a5e4303420d4.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/65a0866c6cca323638f1265be5df79969009bd53154d4aaaa7a9a5e4303420d4.jpg)

![682094b02f146682877124e29efcd38b4ac37ef87b49427ce15f7bb97448eaee.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/682094b02f146682877124e29efcd38b4ac37ef87b49427ce15f7bb97448eaee.jpg)

![69fb4fa4a6effad9fd44ba104ed4b4adeb0b5c2844341e4c2173e1d567f196fe.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/69fb4fa4a6effad9fd44ba104ed4b4adeb0b5c2844341e4c2173e1d567f196fe.jpg)

![8cd86ab37cbc7f1b8e142b16afe4ac0f1d153468806c7ef15738c8276d1c880e.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/8cd86ab37cbc7f1b8e142b16afe4ac0f1d153468806c7ef15738c8276d1c880e.jpg)

![bb0d87e34f97254baa7572f9a6e4f7bd2abf3e3ea759721eb727cb0df397bb4c.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/bb0d87e34f97254baa7572f9a6e4f7bd2abf3e3ea759721eb727cb0df397bb4c.jpg)

![d320fba47a40d84ae9daf8f4943e3349d311b2284eecbb20340892d230f6bbb9.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/d320fba47a40d84ae9daf8f4943e3349d311b2284eecbb20340892d230f6bbb9.jpg)

![d69fa962b7f2561e1dbac31b52ba45d77229434dfbc28adb1c288873f5b1112b.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/images/d69fa962b7f2561e1dbac31b52ba45d77229434dfbc28adb1c288873f5b1112b.jpg)

### Tables

![12542b4c08d8c6541a662040c935281cb8fac3b315592142e670182f6d913c0f.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/12542b4c08d8c6541a662040c935281cb8fac3b315592142e670182f6d913c0f.jpg)

![1c09d9a1b28892fec8eaf3297822052ae898ad9c43af1ee7842667ff21e9ae6e.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/1c09d9a1b28892fec8eaf3297822052ae898ad9c43af1ee7842667ff21e9ae6e.jpg)

![2c747774148ff0710b3b144e14553f0c843e1398dd6151d43b67f273c84ecf12.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/2c747774148ff0710b3b144e14553f0c843e1398dd6151d43b67f273c84ecf12.jpg)

![3a4590f1bf2c17551671fa641f776b6431f784ea58f1437dc78619fca971976f.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/3a4590f1bf2c17551671fa641f776b6431f784ea58f1437dc78619fca971976f.jpg)

![7519b8a83defc215b38b7bf68f94d80c665edd54d0377e7e3dc56e4a4f562fc4.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/7519b8a83defc215b38b7bf68f94d80c665edd54d0377e7e3dc56e4a4f562fc4.jpg)

![75503a73f163fbd40211f342f156905ef519e5976f3ce6c39e1471d10ea7919c.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/75503a73f163fbd40211f342f156905ef519e5976f3ce6c39e1471d10ea7919c.jpg)

![8412cb5d7a62fd18298eaf88f4aacce9daa8612f0c4db26c6a66a93a1b16a706.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/8412cb5d7a62fd18298eaf88f4aacce9daa8612f0c4db26c6a66a93a1b16a706.jpg)

![8c6439e19e2857c82a736012243750ad3e0a526a78a3efefd6ed297bb4576dbf.jpg](../icml_results/172_ECHO_%20Elastic%20Speculative%20Decoding%20with%20Sparse%20Gating%20for%20High-Concurrency%20Scenarios/tables/8c6439e19e2857c82a736012243750ad3e0a526a78a3efefd6ed297bb4576dbf.jpg)

<a id="paper-0171"></a>

## Rapid Poison: Practical Poisoning Attacks Against the Rapid Response Framework


### Images

![16708fa99d358809aae33013da0e7f6ef374ba3a1e74e0439fb76604dd03af16.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/16708fa99d358809aae33013da0e7f6ef374ba3a1e74e0439fb76604dd03af16.jpg)

![26a61e67978dd59e50ddcd4a9ee5e295671f1c419e3419152c015024f5144d0d.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/26a61e67978dd59e50ddcd4a9ee5e295671f1c419e3419152c015024f5144d0d.jpg)

![3f49c27dfa1ac57d6e7e21cb5db669f2994fba7d28f11f9a475dba42dd0fea64.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/3f49c27dfa1ac57d6e7e21cb5db669f2994fba7d28f11f9a475dba42dd0fea64.jpg)

![4b4566f7b3d6608fb470cc868c6d944bae3a36a1cca8ad6cc551632473855864.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/4b4566f7b3d6608fb470cc868c6d944bae3a36a1cca8ad6cc551632473855864.jpg)

![4e5b27433ac16712570483d7d10f0860265b1d815625e4b8f4d8a28cad7b0efc.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/4e5b27433ac16712570483d7d10f0860265b1d815625e4b8f4d8a28cad7b0efc.jpg)

![5163174df1bda6e8d8a498801b675b4f2a7192e227e9dc58b1db422e5e8a295b.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/5163174df1bda6e8d8a498801b675b4f2a7192e227e9dc58b1db422e5e8a295b.jpg)

![531b4199043f88b3ef04d0a06f7a63182ebbad2e8ac13edcebe38e5c0717b769.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/531b4199043f88b3ef04d0a06f7a63182ebbad2e8ac13edcebe38e5c0717b769.jpg)

![72eb5dbdcfe0f643b5da6c9165f781321dd877ff27ebe5edc57c1fd6e5e89702.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/72eb5dbdcfe0f643b5da6c9165f781321dd877ff27ebe5edc57c1fd6e5e89702.jpg)

![8cb880140d967df21457756ca995d21f95c4d61ed7f30a5ca147a249a524577d.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/8cb880140d967df21457756ca995d21f95c4d61ed7f30a5ca147a249a524577d.jpg)

![9158df2b498fd486d4fc1a107805b9ba235e5a322f8834aab245b1427ab13f86.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/9158df2b498fd486d4fc1a107805b9ba235e5a322f8834aab245b1427ab13f86.jpg)

![928248b7fc664120139535882f9a833b2b4c1f3a02e41054fe6417b53f51f32f.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/928248b7fc664120139535882f9a833b2b4c1f3a02e41054fe6417b53f51f32f.jpg)

![940470648cb4ee82f7c2bb5da05643a902b81313d363c973c91a67f55cd1e054.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/940470648cb4ee82f7c2bb5da05643a902b81313d363c973c91a67f55cd1e054.jpg)

![9e2052ceb09d87124d7e47927f1629149699b4810ec6b6d30228541b0536603a.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/9e2052ceb09d87124d7e47927f1629149699b4810ec6b6d30228541b0536603a.jpg)

![a95d8ece806f31b8cdffc20feb9790c01a9dd9cb5296dd5274f9187ba55a62e4.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/a95d8ece806f31b8cdffc20feb9790c01a9dd9cb5296dd5274f9187ba55a62e4.jpg)

![d040fd63146e1b38817cc9d3b42f843fbe9d2c18b7945a534de8526f15290462.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/d040fd63146e1b38817cc9d3b42f843fbe9d2c18b7945a534de8526f15290462.jpg)

![d0701c22163ec41b72b22a666293e63b517cbd233b53f308e5361dea10a49af4.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/d0701c22163ec41b72b22a666293e63b517cbd233b53f308e5361dea10a49af4.jpg)

![ea2a67ed39488b63d2e1f8b9fc6496f60238ee6640ed70e3b70976503bd88f85.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/ea2a67ed39488b63d2e1f8b9fc6496f60238ee6640ed70e3b70976503bd88f85.jpg)

![f330957eda1682aed723a8c8ca7c5b570dbb4d4c75280e1a9e0f48e6655bd623.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/f330957eda1682aed723a8c8ca7c5b570dbb4d4c75280e1a9e0f48e6655bd623.jpg)

![f47669cb16e03e7db4f7c7280fdba5b289c638a0ee85baee8e3fdd14df423934.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/f47669cb16e03e7db4f7c7280fdba5b289c638a0ee85baee8e3fdd14df423934.jpg)

![fa869bdc24b5f975d8c89e76f59c3f919b1a8d0a346971e6d45e23656aa10b6d.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/images/fa869bdc24b5f975d8c89e76f59c3f919b1a8d0a346971e6d45e23656aa10b6d.jpg)

### Tables

![9c35b11fbf257451bbe275ba858bb169edece3003973a2c24ff749bb97dfaba5.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/tables/9c35b11fbf257451bbe275ba858bb169edece3003973a2c24ff749bb97dfaba5.jpg)

![c6d0b4930e605b2232c9e5f7f5d3ea63280687fbe32fb74b0159b967cb36f93f.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/tables/c6d0b4930e605b2232c9e5f7f5d3ea63280687fbe32fb74b0159b967cb36f93f.jpg)

![e07982d6f5ae9ba744daf9cc724d413368d19ff544142eb30b46046a8a91adce.jpg](../icml_results/173_Rapid%20Poison_%20Practical%20Poisoning%20Attacks%20Against%20the%20Rapid%20Response%20Framework/tables/e07982d6f5ae9ba744daf9cc724d413368d19ff544142eb30b46046a8a91adce.jpg)

<a id="paper-0172"></a>

## EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models


### Images

![47dee1281d50ce68040036451c53a376f37354f9f31d2ee68f5a8f88c608fb62.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/47dee1281d50ce68040036451c53a376f37354f9f31d2ee68f5a8f88c608fb62.jpg)

![550e9d29b5b1dcce3a91d4718c4db21a482ba5d7aa1b9b4308f6c21ed2185e9b.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/550e9d29b5b1dcce3a91d4718c4db21a482ba5d7aa1b9b4308f6c21ed2185e9b.jpg)

![6c08c5a97aa368ede5336067d31fed23324e28db879446cb159e5f48b523a4e2.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/6c08c5a97aa368ede5336067d31fed23324e28db879446cb159e5f48b523a4e2.jpg)

![71cf0300905399c07e81d75701818adfdc488e81a6428a63d72790f4cc7dcda9.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/71cf0300905399c07e81d75701818adfdc488e81a6428a63d72790f4cc7dcda9.jpg)

![7272905c49f31311e92974076c0589ddebf09ea6a8e738999411414d956b421b.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/7272905c49f31311e92974076c0589ddebf09ea6a8e738999411414d956b421b.jpg)

![8ebdd5223eb74097e62aaa7993dfe0cd00ab80f0b9ed1614a4cad8a8d6fe80ec.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/8ebdd5223eb74097e62aaa7993dfe0cd00ab80f0b9ed1614a4cad8a8d6fe80ec.jpg)

![c91d003445678c0d97a23cd5d8c25d9612ec4988671efdf487544ea5e9f595ae.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/c91d003445678c0d97a23cd5d8c25d9612ec4988671efdf487544ea5e9f595ae.jpg)

![e496e2aaabc300441b8d516515ffc304c01765274f3e6eaa1b6726049918d60c.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/e496e2aaabc300441b8d516515ffc304c01765274f3e6eaa1b6726049918d60c.jpg)

![e8e105f778e0b09f0890d22de9bbd204be1eb5d2a8073c49be4a1e6b86fb396a.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/images/e8e105f778e0b09f0890d22de9bbd204be1eb5d2a8073c49be4a1e6b86fb396a.jpg)

### Tables

![064cfe363c6b62ed795ea41f9f7f20a8991a0a8568256fe00aeb0f970ee01cc0.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/064cfe363c6b62ed795ea41f9f7f20a8991a0a8568256fe00aeb0f970ee01cc0.jpg)

![07ec8920118ad13ada4e289aebf151f00c7dcbdf7a45f9a8e9c29069af959f94.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/07ec8920118ad13ada4e289aebf151f00c7dcbdf7a45f9a8e9c29069af959f94.jpg)

![4eed6662cd100dbe58dc64b6c78b41f6d1ae9085a88085e97198147a7ec3b775.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/4eed6662cd100dbe58dc64b6c78b41f6d1ae9085a88085e97198147a7ec3b775.jpg)

![58b3dbbba4cc8b16b695b40781e70bff12443686629b10232b9a1487bdd14dc4.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/58b3dbbba4cc8b16b695b40781e70bff12443686629b10232b9a1487bdd14dc4.jpg)

![88f6046e9cb6d098851d84bb4198096898ec38f76075855002d8e367a93c1499.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/88f6046e9cb6d098851d84bb4198096898ec38f76075855002d8e367a93c1499.jpg)

![8ebf731187dd5bb0236523f211f658be99b85bb610da97e7d06f4bd01908738b.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/8ebf731187dd5bb0236523f211f658be99b85bb610da97e7d06f4bd01908738b.jpg)

![b813c1a1073abc92574a5095ff8bcfa9b29d2a06937fbc74ed3042a5f1ae4321.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/b813c1a1073abc92574a5095ff8bcfa9b29d2a06937fbc74ed3042a5f1ae4321.jpg)

![bc1c72deab14729792dc26a94b3f3788a9615d9433fa856e5b9be197f784e389.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/bc1c72deab14729792dc26a94b3f3788a9615d9433fa856e5b9be197f784e389.jpg)

![c083eafe72861433ce34ac2ab7fcb4f7e7c16519e25e220b50bc38ec3840d781.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/c083eafe72861433ce34ac2ab7fcb4f7e7c16519e25e220b50bc38ec3840d781.jpg)

![d11a3b5058b01807b058de7d14e618cc403c8998f38ca193e7eb9ef35483a26e.jpg](../icml_results/174_EcoVLA_%20Environment-Aware%20Adaptive%20Pruning%20with%20Interleaved%20Inference%20Orchestration%20for%20Vision-Langu/tables/d11a3b5058b01807b058de7d14e618cc403c8998f38ca193e7eb9ef35483a26e.jpg)

<a id="paper-0173"></a>

## Thinking in Flow: A Dissipative Stabilization Operator for Robust Autoregressive Reasoning


### Images

![203ad871108e083cdb3e62c69b5e3edbd0c225a7d34d93b31b3915c6997ee406.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/203ad871108e083cdb3e62c69b5e3edbd0c225a7d34d93b31b3915c6997ee406.jpg)

![35fd1cd6c5e91afcc85dd2deedad38ffaa7a0de3402b03538bbb744401929a94.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/35fd1cd6c5e91afcc85dd2deedad38ffaa7a0de3402b03538bbb744401929a94.jpg)

![58809c72e05c14ff5a5469513b8c734ac3d6982452fcf0879a6a3ad9cc36b2c1.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/58809c72e05c14ff5a5469513b8c734ac3d6982452fcf0879a6a3ad9cc36b2c1.jpg)

![63a4db50a36280c0af3332c5b28b426edf6387697614f9314a9b918a32739c4c.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/63a4db50a36280c0af3332c5b28b426edf6387697614f9314a9b918a32739c4c.jpg)

![75972202be33a802ef580584d0f2fbf12a6ad178cd8aa5ea3980c073afa46df2.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/75972202be33a802ef580584d0f2fbf12a6ad178cd8aa5ea3980c073afa46df2.jpg)

![a8562afbb629c32d31c31aa93a109cf3af511f8947be3f26a2532609e1e4b408.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/a8562afbb629c32d31c31aa93a109cf3af511f8947be3f26a2532609e1e4b408.jpg)

![abd9750b2909101cbcc3b6917a85ac52fb047c47c3b880f2a84f48f3e1457c45.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/abd9750b2909101cbcc3b6917a85ac52fb047c47c3b880f2a84f48f3e1457c45.jpg)

![b789462a3bad06f889226d853890cdd05570d168af77642160058afb92af6f39.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/b789462a3bad06f889226d853890cdd05570d168af77642160058afb92af6f39.jpg)

![ca67d68a4ad87b0dd8a62ea84e7d0256324f544e033b9b2a0e79328d4e4c458b.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/ca67d68a4ad87b0dd8a62ea84e7d0256324f544e033b9b2a0e79328d4e4c458b.jpg)

![cc08816be8ab0c7ad3e5950f11ee34d3840acbe733dd902361ed2b3bb59cf436.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/cc08816be8ab0c7ad3e5950f11ee34d3840acbe733dd902361ed2b3bb59cf436.jpg)

![d20c43d9f880545aee4858cb1b7465e6050dcfa37d10784d09025774a886e4c1.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/d20c43d9f880545aee4858cb1b7465e6050dcfa37d10784d09025774a886e4c1.jpg)

![d535eddfc16f0b751cc4a4c9469dc3a0cf98d213053b6433050218c2ac55eeaa.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/images/d535eddfc16f0b751cc4a4c9469dc3a0cf98d213053b6433050218c2ac55eeaa.jpg)

### Tables

![1dbd15715ddf7cde14bed0c049332f822b8d05c47ff3ad11dbeec0cdefcb7fa8.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/1dbd15715ddf7cde14bed0c049332f822b8d05c47ff3ad11dbeec0cdefcb7fa8.jpg)

![1f3866e22a5a44d60a2d027f01af37f97a431ce96f69a2b1abd754cfb648b299.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/1f3866e22a5a44d60a2d027f01af37f97a431ce96f69a2b1abd754cfb648b299.jpg)

![23ffc10caf47e5eb69b91fe25b0a2be2147659188f87079709c8ece0d4e54871.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/23ffc10caf47e5eb69b91fe25b0a2be2147659188f87079709c8ece0d4e54871.jpg)

![24e7e2017c2f437a1826c3bdb0bb53e92a8a963a5ef03151035c9deae3479279.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/24e7e2017c2f437a1826c3bdb0bb53e92a8a963a5ef03151035c9deae3479279.jpg)

![2a89dd814828815e0fa02faec6e686bff0fbad341a71d23fb4d1ad5f68c860df.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/2a89dd814828815e0fa02faec6e686bff0fbad341a71d23fb4d1ad5f68c860df.jpg)

![308798a588e6e7404b4d3b55a955bedb9dded740d5813830b693e7c51252a198.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/308798a588e6e7404b4d3b55a955bedb9dded740d5813830b693e7c51252a198.jpg)

![3368053bad072b78bd619c34988af2ae1d171f95275d15dab3f631d6286caf58.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/3368053bad072b78bd619c34988af2ae1d171f95275d15dab3f631d6286caf58.jpg)

![480e41c6a90903373341cfbe599e90e39d80a1d6c692c1570a71b4998eb10a44.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/480e41c6a90903373341cfbe599e90e39d80a1d6c692c1570a71b4998eb10a44.jpg)

![480fa070298bf6b42a0fe908978352035d0a188654a1545203b13600617ca57b.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/480fa070298bf6b42a0fe908978352035d0a188654a1545203b13600617ca57b.jpg)

![648c03f38aaa777312f9b850cc9a90346565d7b2595eaf0b2e3b9dc680e787b2.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/648c03f38aaa777312f9b850cc9a90346565d7b2595eaf0b2e3b9dc680e787b2.jpg)

![9aeba5bde96d3ef322d90aae4f0a787238d7f5b8dcb4f27c16c2ce8bd76aae91.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/9aeba5bde96d3ef322d90aae4f0a787238d7f5b8dcb4f27c16c2ce8bd76aae91.jpg)

![b861bd039ff5c69f517e5bc95e8550f1d32ba1391f5b7d153d62fd2c9e7ae19b.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/b861bd039ff5c69f517e5bc95e8550f1d32ba1391f5b7d153d62fd2c9e7ae19b.jpg)

![c518088786797d8ed59a7b9a61103236239b31943001b185f5f303cdc64a4e5a.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/c518088786797d8ed59a7b9a61103236239b31943001b185f5f303cdc64a4e5a.jpg)

![d60f7a6b983c281c30b4224960c0af1e163bc8065d27a3abe7504a170af383dd.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/d60f7a6b983c281c30b4224960c0af1e163bc8065d27a3abe7504a170af383dd.jpg)

![d616e4b6056951bc525be09192fa0d30fe8a63595cc4c0c7db50ea77aa038da7.jpg](../icml_results/175_Thinking%20in%20Flow_%20A%20Dissipative%20Stabilization%20Operator%20for%20Robust%20Autoregressive%20Reasoning/tables/d616e4b6056951bc525be09192fa0d30fe8a63595cc4c0c7db50ea77aa038da7.jpg)

<a id="paper-0174"></a>

## Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels


### Images

![187dc722afa68b5f1c77819f5371ab280489e6ed666d3fceddfd43d0a7a54d7b.jpg](../icml_results/176_Alignment-Sensitive%20Minimax%20Rates%20for%20Spectral%20Algorithms%20with%20Learned%20Kernels/images/187dc722afa68b5f1c77819f5371ab280489e6ed666d3fceddfd43d0a7a54d7b.jpg)

![3901fc229043ce2c0c5dd7e4ef85c7cc4b1692302481bec509b0ca975bbb0211.jpg](../icml_results/176_Alignment-Sensitive%20Minimax%20Rates%20for%20Spectral%20Algorithms%20with%20Learned%20Kernels/images/3901fc229043ce2c0c5dd7e4ef85c7cc4b1692302481bec509b0ca975bbb0211.jpg)

![94d925cd84ba831ec85ecab5825135d1ae270c07b1d6e8bd37bbf076666bb653.jpg](../icml_results/176_Alignment-Sensitive%20Minimax%20Rates%20for%20Spectral%20Algorithms%20with%20Learned%20Kernels/images/94d925cd84ba831ec85ecab5825135d1ae270c07b1d6e8bd37bbf076666bb653.jpg)

![9666dcc8d3206a90c9b5983f02f7d4588e7fec682c56d14639210ec22ad1765f.jpg](../icml_results/176_Alignment-Sensitive%20Minimax%20Rates%20for%20Spectral%20Algorithms%20with%20Learned%20Kernels/images/9666dcc8d3206a90c9b5983f02f7d4588e7fec682c56d14639210ec22ad1765f.jpg)

![c74fc4c72cdfa77b570ab04b7289cd55f8db63f5b00a65122e1b56376b9081f0.jpg](../icml_results/176_Alignment-Sensitive%20Minimax%20Rates%20for%20Spectral%20Algorithms%20with%20Learned%20Kernels/images/c74fc4c72cdfa77b570ab04b7289cd55f8db63f5b00a65122e1b56376b9081f0.jpg)

<a id="paper-0175"></a>

## Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers


### Images

![01d179016f6d856a0b2825ff9eb7f4739d0a3a327f6982999eef1e06105e12fc.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/01d179016f6d856a0b2825ff9eb7f4739d0a3a327f6982999eef1e06105e12fc.jpg)

![0302ac668aaf4601c4aab6db9f6fdbce74359ea3447b314723ebbe927a1f73a8.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/0302ac668aaf4601c4aab6db9f6fdbce74359ea3447b314723ebbe927a1f73a8.jpg)

![09cc8fa81de9522bfee485ee8d6a55042f2a676ed4a79e87aef71904b2607843.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/09cc8fa81de9522bfee485ee8d6a55042f2a676ed4a79e87aef71904b2607843.jpg)

![2bd644bcbd2daab51369db063216ed7aa0cdd113611e3d0cee00581981381063.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/2bd644bcbd2daab51369db063216ed7aa0cdd113611e3d0cee00581981381063.jpg)

![4486bb3ad01c25594616656360e2fe9ea704d0faaf2f9e6b6824752524842fdf.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/4486bb3ad01c25594616656360e2fe9ea704d0faaf2f9e6b6824752524842fdf.jpg)

![5158f6f7cded7e4f99d95e26be7a4afe0451b43fdaf55963cd7de08df6eb0417.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/5158f6f7cded7e4f99d95e26be7a4afe0451b43fdaf55963cd7de08df6eb0417.jpg)

![62b189850ceda6e316d4f5d69b1cc7084a043a10a3e4d2884061000625e9927d.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/62b189850ceda6e316d4f5d69b1cc7084a043a10a3e4d2884061000625e9927d.jpg)

![78132a5847d8f12c1645fef17cb98c106761a0934a315f3ecdb062d8f86ed5f7.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/78132a5847d8f12c1645fef17cb98c106761a0934a315f3ecdb062d8f86ed5f7.jpg)

![81c806534963862146c1f7cf108d0f874b0042eb002da6aa9cd2d86ae74548d5.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/81c806534963862146c1f7cf108d0f874b0042eb002da6aa9cd2d86ae74548d5.jpg)

![8364a170999821ce16b4f9ee88320a9ca7189833ca672d196f6abc97d408b36d.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/8364a170999821ce16b4f9ee88320a9ca7189833ca672d196f6abc97d408b36d.jpg)

![8e8d14c11a7bbf1e6d3c67764d8c924ead9c6e35baf4e93c75ced661d172cde3.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/8e8d14c11a7bbf1e6d3c67764d8c924ead9c6e35baf4e93c75ced661d172cde3.jpg)

![addce0650fa9d298c5d17310719993762f92dfb693b258143f20c78c3be7a083.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/addce0650fa9d298c5d17310719993762f92dfb693b258143f20c78c3be7a083.jpg)

![b4bc9ecdb40a4fd7da7d2909c341357ab5e5d054e7466128fb25d42e9fc7b14e.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/b4bc9ecdb40a4fd7da7d2909c341357ab5e5d054e7466128fb25d42e9fc7b14e.jpg)

![c6224db78705ba0732f567607cda36fec9eee163faaecc3f19d266993eac34b6.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/c6224db78705ba0732f567607cda36fec9eee163faaecc3f19d266993eac34b6.jpg)

![cf9b8a1e9a2de5d6f58129b6e190c1234eb3fdbcd80bb4dd9441f8b79bb92d19.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/cf9b8a1e9a2de5d6f58129b6e190c1234eb3fdbcd80bb4dd9441f8b79bb92d19.jpg)

![d010a18b2d8077168dc4e88d41bbceba37e6bada99fc4f23b6123f728326684b.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/d010a18b2d8077168dc4e88d41bbceba37e6bada99fc4f23b6123f728326684b.jpg)

![d5250262dfb266fe4a035557fec1bcbbfc0bf128b4513d2d9e59fea2c6a1bfb6.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/d5250262dfb266fe4a035557fec1bcbbfc0bf128b4513d2d9e59fea2c6a1bfb6.jpg)

![ef681c68ab682d8e4f8d93cd51500278329cd9e1adaa1c41c9d70dfb74b0ed5e.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/images/ef681c68ab682d8e4f8d93cd51500278329cd9e1adaa1c41c9d70dfb74b0ed5e.jpg)

### Tables

![11fca094f7dcdde4cfd063402fb1ee6b16ed8228d3666a5d2620ac14641f6015.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/11fca094f7dcdde4cfd063402fb1ee6b16ed8228d3666a5d2620ac14641f6015.jpg)

![2206498406dfdc8533f2dec7ebfa3ecda8b1de796721bd87fc92146ddad2eb42.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/2206498406dfdc8533f2dec7ebfa3ecda8b1de796721bd87fc92146ddad2eb42.jpg)

![25de9e84f72782861231ad84430893f5546b85d7989e4758aaf2cb6bf111a190.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/25de9e84f72782861231ad84430893f5546b85d7989e4758aaf2cb6bf111a190.jpg)

![3186f9924aef5307e45fbdc64120233c98341dd023258f3feeb1b9e626fcb5d2.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/3186f9924aef5307e45fbdc64120233c98341dd023258f3feeb1b9e626fcb5d2.jpg)

![336d7ffd5c611f8af52c4088ec5c9e3bd34b6da1403f44079e9e0720c9ad9d68.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/336d7ffd5c611f8af52c4088ec5c9e3bd34b6da1403f44079e9e0720c9ad9d68.jpg)

![5fd9fef1ccbae963e995e928055e1f000d12f75c5bd7739b9472b612af1e68cc.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/5fd9fef1ccbae963e995e928055e1f000d12f75c5bd7739b9472b612af1e68cc.jpg)

![67fa090ac2ce503783e373c513c9a600209f4615e04c0c5678ed3747d2f4222e.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/67fa090ac2ce503783e373c513c9a600209f4615e04c0c5678ed3747d2f4222e.jpg)

![836d4a91dcb66be466bb39a999cbe005e439e484d2ac3babeb58f889e51e14d2.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/836d4a91dcb66be466bb39a999cbe005e439e484d2ac3babeb58f889e51e14d2.jpg)

![84d1426bc6e6ca5979b893054b561a54ee4929b5ec4debd4bfbd946dedd865f6.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/84d1426bc6e6ca5979b893054b561a54ee4929b5ec4debd4bfbd946dedd865f6.jpg)

![92e123149760e2148ecfa56e5a2be6b6b281987648e768083c48cdce2c269202.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/92e123149760e2148ecfa56e5a2be6b6b281987648e768083c48cdce2c269202.jpg)

![bd16bc72f8924564c1e33417862478e275aa144887310ecd58fce2fc796b4dc5.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/bd16bc72f8924564c1e33417862478e275aa144887310ecd58fce2fc796b4dc5.jpg)

![be74dff0b9522c70ebf1eaa983f46fd48b223baed498ff18dec9d11c854eb610.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/be74dff0b9522c70ebf1eaa983f46fd48b223baed498ff18dec9d11c854eb610.jpg)

![fb3076a9727c7efe6d4061e85b6650f726a7b5283d7c68c84bb5fae0f292b0eb.jpg](../icml_results/177_Dissecting%20Multimodal%20In-Context%20Learning_%20Modality%20Asymmetries%20and%20Circuit%20Dynamics%20in%20modern%20Trans/tables/fb3076a9727c7efe6d4061e85b6650f726a7b5283d7c68c84bb5fae0f292b0eb.jpg)

<a id="paper-0176"></a>

## Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks


### Images

![174dfc63c6b0ad06d2bdaa608a296fdd0300786e05868ad04532545e028dcab0.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/174dfc63c6b0ad06d2bdaa608a296fdd0300786e05868ad04532545e028dcab0.jpg)

![47bbb880a7053df850e3add64d47f98580b1a04fb0e27f84ab4d315d0e0135ad.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/47bbb880a7053df850e3add64d47f98580b1a04fb0e27f84ab4d315d0e0135ad.jpg)

![69454c94e2bfc74b21f89334dfa209c1151628453f467c2f7da7537e73540a80.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/69454c94e2bfc74b21f89334dfa209c1151628453f467c2f7da7537e73540a80.jpg)

![6b46aa9f3c842339ff54a4cc6e2fd00a52ec4f93bb89e094c33949a73c054092.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/6b46aa9f3c842339ff54a4cc6e2fd00a52ec4f93bb89e094c33949a73c054092.jpg)

![700b4e75aa44e2e95cf19bf37fbcf9eccfa3203e95fe19cd9ce7b0599559ddc3.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/700b4e75aa44e2e95cf19bf37fbcf9eccfa3203e95fe19cd9ce7b0599559ddc3.jpg)

![8313cf4b11f804b7397d9953b81a5a7398257c7ceb43483d7d9c03a946bcf9c3.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/8313cf4b11f804b7397d9953b81a5a7398257c7ceb43483d7d9c03a946bcf9c3.jpg)

![8ba79e3f68682e25e0c74b6fe6cac3dcb4a0aa65ad00ffb3178e150920daf16e.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/8ba79e3f68682e25e0c74b6fe6cac3dcb4a0aa65ad00ffb3178e150920daf16e.jpg)

![9284bd442873dae0e36cf921c6b6cae25fe7baf41f9bd5c607e0b2b67bc9ebd9.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/9284bd442873dae0e36cf921c6b6cae25fe7baf41f9bd5c607e0b2b67bc9ebd9.jpg)

![9a917e8b9845c1dcb4a4876b7d2b866d625cf51ffb48311d075cf60b787ab598.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/9a917e8b9845c1dcb4a4876b7d2b866d625cf51ffb48311d075cf60b787ab598.jpg)

![ab1423977f13c30ad896941d031bf5ffc20e96dd1068adc139427d48c9c2e292.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/ab1423977f13c30ad896941d031bf5ffc20e96dd1068adc139427d48c9c2e292.jpg)

![c1de2d4b77cde89230d1a7dac3a9731bc9dbdc6b2cdfa4463ed7ed41d53bb166.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/c1de2d4b77cde89230d1a7dac3a9731bc9dbdc6b2cdfa4463ed7ed41d53bb166.jpg)

![e6979a2ffb79fc5f1f25feaaf07ea33d9046796fb52a67aab94aaba958fd6695.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/e6979a2ffb79fc5f1f25feaaf07ea33d9046796fb52a67aab94aaba958fd6695.jpg)

![e7f453a203d826221c37bd1d5b4fa72f3db69cf9baa15240b6a97f0c95295f4f.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/e7f453a203d826221c37bd1d5b4fa72f3db69cf9baa15240b6a97f0c95295f4f.jpg)

![ecd5f0f661ac226b717967ce1e650a1209cf02d010f610b912a7647730bf41cb.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/images/ecd5f0f661ac226b717967ce1e650a1209cf02d010f610b912a7647730bf41cb.jpg)

### Tables

![00c5502a99b6a67ede26fe2b6f90147657ce9c93d3d65aacf6e639cf7853a978.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/00c5502a99b6a67ede26fe2b6f90147657ce9c93d3d65aacf6e639cf7853a978.jpg)

![047731063e490637bc974a43b93bf08a0198d030bfc92111d50148272c2e32d0.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/047731063e490637bc974a43b93bf08a0198d030bfc92111d50148272c2e32d0.jpg)

![130a495b526affebb63ba31b539cd4742628969a81f517c7a4fb99e59b0e91de.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/130a495b526affebb63ba31b539cd4742628969a81f517c7a4fb99e59b0e91de.jpg)

![5d2373a0fd439ff854494ed8cd0151e93f193861826b433e1c630df8b77388ea.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/5d2373a0fd439ff854494ed8cd0151e93f193861826b433e1c630df8b77388ea.jpg)

![78d89c7650fa79626aaab0590fa202b43e8cb5f9269a14ed4711698fffa0e526.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/78d89c7650fa79626aaab0590fa202b43e8cb5f9269a14ed4711698fffa0e526.jpg)

![8c98859c3e7a49290da023a372e46d542e0fa5eb9f1b5566359464443d233628.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/8c98859c3e7a49290da023a372e46d542e0fa5eb9f1b5566359464443d233628.jpg)

![bf1bc9afc2cd913337bd511d4a718048f896ab455998ba4cea560d66c7b081a4.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/bf1bc9afc2cd913337bd511d4a718048f896ab455998ba4cea560d66c7b081a4.jpg)

![cd23adb5a01f2e1b397719ecce7e07dda2514af26642f9a0d8f18d5c633aae83.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/cd23adb5a01f2e1b397719ecce7e07dda2514af26642f9a0d8f18d5c633aae83.jpg)

![dda59820d970ccb05d908bb93d1b38a1ecbe73e7d92146994267d8bfd52c7517.jpg](../icml_results/178_Chebyshev%20Policies%20and%20the%20Mountain%20Car%20Problem_%20Reinforcement%20Learning%20for%20Low-Dimensional%20Control%20/tables/dda59820d970ccb05d908bb93d1b38a1ecbe73e7d92146994267d8bfd52c7517.jpg)

<a id="paper-0177"></a>

## Copyright-Bench: Agentic Evaluation of Copyright Law Compliance


### Images

![36d16d8d5bb89ffd39df05b2efe6e193dd0875d31d67b1283614e21e8f80f1ad.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/images/36d16d8d5bb89ffd39df05b2efe6e193dd0875d31d67b1283614e21e8f80f1ad.jpg)

![ba4b7954fc0a8b9fa9148f36130e3d7c9bbdbecbed3ec10a1c4518e866e3f67b.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/images/ba4b7954fc0a8b9fa9148f36130e3d7c9bbdbecbed3ec10a1c4518e866e3f67b.jpg)

![e4210642e82c8210b3e72022ec687c04fc8e9f31f40ec436efd7f0bde85aeb4d.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/images/e4210642e82c8210b3e72022ec687c04fc8e9f31f40ec436efd7f0bde85aeb4d.jpg)

### Tables

![9b21ff94d0cb74ec424f7770daf58cccb7d90da2fba8becd94baac047579a045.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/tables/9b21ff94d0cb74ec424f7770daf58cccb7d90da2fba8becd94baac047579a045.jpg)

![d7c1de4a294cfdb06499b32d8d79e2fe1128a49b5bed8d82ef6acf8f2dbcd83e.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/tables/d7c1de4a294cfdb06499b32d8d79e2fe1128a49b5bed8d82ef6acf8f2dbcd83e.jpg)

![dac66c03c3d7a519f218ea50628977bb1c69382925d8204c2429ee5a47dddccb.jpg](../icml_results/179_Copyright-Bench_%20Agentic%20Evaluation%20of%20Copyright%20Law%20Compliance/tables/dac66c03c3d7a519f218ea50628977bb1c69382925d8204c2429ee5a47dddccb.jpg)

<a id="paper-0178"></a>

## Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts


### Images

![128d2e994312e51b09ef9027e125155ec949be2acb50029dbdb92902baf071be.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/128d2e994312e51b09ef9027e125155ec949be2acb50029dbdb92902baf071be.jpg)

![1e56ccfebc8c794736b047912953a22013777074e53f400b7f3ceb24a0a28034.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/1e56ccfebc8c794736b047912953a22013777074e53f400b7f3ceb24a0a28034.jpg)

![23a617df233ebb1b8fe4c11338730535d9c2aaa06f0522d407c49bdb2234faa5.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/23a617df233ebb1b8fe4c11338730535d9c2aaa06f0522d407c49bdb2234faa5.jpg)

![72211c9e16f4f76e0a504f078210e07053e3b2455c8ad82dff54eb5f80391b77.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/72211c9e16f4f76e0a504f078210e07053e3b2455c8ad82dff54eb5f80391b77.jpg)

![8680a2fed83880ac274a97e04d2c63b1be448dcd822e93c7fbea0bc47de72189.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/8680a2fed83880ac274a97e04d2c63b1be448dcd822e93c7fbea0bc47de72189.jpg)

![b3560f2ff2581ebf484fa3ea0e4a94754922d4da66dc523179000fd52bdff1b7.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/b3560f2ff2581ebf484fa3ea0e4a94754922d4da66dc523179000fd52bdff1b7.jpg)

![e666cb414ff270cbeb84afa9e137967f01ed1fab0a453b11e0bb93422b4331b1.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/images/e666cb414ff270cbeb84afa9e137967f01ed1fab0a453b11e0bb93422b4331b1.jpg)

### Tables

![2f0e2a657563b6a38be823b271f75efd3fd25851b14e888c6be76de17cbb7536.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/2f0e2a657563b6a38be823b271f75efd3fd25851b14e888c6be76de17cbb7536.jpg)

![33473d718e7af01126540b7494914286fba8d3a4db2d05ac3cc6916cf002c3cf.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/33473d718e7af01126540b7494914286fba8d3a4db2d05ac3cc6916cf002c3cf.jpg)

![44cc8fbe55749d963ee1c0d89ea5f5cbd13aa9c86062a37589d84cbd05da3db9.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/44cc8fbe55749d963ee1c0d89ea5f5cbd13aa9c86062a37589d84cbd05da3db9.jpg)

![54db0bd6b9b9bcc321d2614322ef51e90a981fe7281df9b76ef0f835907f58c3.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/54db0bd6b9b9bcc321d2614322ef51e90a981fe7281df9b76ef0f835907f58c3.jpg)

![75a4845135ee1568a92eb20efb1abd4623a70d6c5335d74553e13d8ca234e63d.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/75a4845135ee1568a92eb20efb1abd4623a70d6c5335d74553e13d8ca234e63d.jpg)

![75f8614c23eebd5aaa075d168c36a94163f45153db01062f126bc7e9d91556a2.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/75f8614c23eebd5aaa075d168c36a94163f45153db01062f126bc7e9d91556a2.jpg)

![9f72f57ef215826916fb791bc0cb6ba28436b089a1846dcdb862fb093cfbfc22.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/9f72f57ef215826916fb791bc0cb6ba28436b089a1846dcdb862fb093cfbfc22.jpg)

![ae349f628efacb90ee2c5222003e7fd6c37411deae204eb7033a6a72789acf4d.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/ae349f628efacb90ee2c5222003e7fd6c37411deae204eb7033a6a72789acf4d.jpg)

![d1838918d79c1cebb47600dc46354ad53d918be2a37aef7d49886eb694fe78fd.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/d1838918d79c1cebb47600dc46354ad53d918be2a37aef7d49886eb694fe78fd.jpg)

![db0f4fce33218be3a448b78d595cf2b0e647a218e0d5a46572c7b10785551d9d.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/db0f4fce33218be3a448b78d595cf2b0e647a218e0d5a46572c7b10785551d9d.jpg)

![ee8b375b9906d80807fcbe123e284a73de162b68ae256639797b03f8b4f7472e.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/ee8b375b9906d80807fcbe123e284a73de162b68ae256639797b03f8b4f7472e.jpg)

![f3fab6f40f0920cc40f1f014be76fbab19dd2d0d7db07e54ca75f18df34ed09c.jpg](../icml_results/180_Geometric%20and%20Stochastic%20Analysis%20of%20Discontinuities%20in%20Sparse%20Mixture-of-Experts/tables/f3fab6f40f0920cc40f1f014be76fbab19dd2d0d7db07e54ca75f18df34ed09c.jpg)

<a id="paper-0179"></a>

## Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference


### Images

![307bd4735cca5b1e43d759d97cd1dc768aedf04256b77c63a29617d0d0b5b07f.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/307bd4735cca5b1e43d759d97cd1dc768aedf04256b77c63a29617d0d0b5b07f.jpg)

![3273b1abd1f6c0cbd49066e1d0948f939dcfc49c26e9b0e750329821d9253103.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/3273b1abd1f6c0cbd49066e1d0948f939dcfc49c26e9b0e750329821d9253103.jpg)

![398a78827891f192483411cd4328208aa777c3989a867e0785d893975d3d4c68.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/398a78827891f192483411cd4328208aa777c3989a867e0785d893975d3d4c68.jpg)

![65f27d3080dd5035226ae50e844370920c057fa496e553da366048d4913ae422.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/65f27d3080dd5035226ae50e844370920c057fa496e553da366048d4913ae422.jpg)

![7adf23e3ef6d0695cbae185c94852c1e0494096e969d8a7665ca584ac0e4bdfe.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/7adf23e3ef6d0695cbae185c94852c1e0494096e969d8a7665ca584ac0e4bdfe.jpg)

![7deb803a5ba9fb008fdfd78777fff3480103db5b97b3e5207cc241effda2c11d.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/7deb803a5ba9fb008fdfd78777fff3480103db5b97b3e5207cc241effda2c11d.jpg)

![9c2710e43dd184d004d246b4548720b28a2f2501ae60a785a712113fe5770d91.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/images/9c2710e43dd184d004d246b4548720b28a2f2501ae60a785a712113fe5770d91.jpg)

### Tables

![445541524b09a8ddede31850ea3ec19479b422dc2391d001f35b37ec032b4ca8.jpg](../icml_results/181_Correcting%20Split%20Selection%20in%20Online%20Decision%20Trees%20via%20Anytime-Valid%20Inference/tables/445541524b09a8ddede31850ea3ec19479b422dc2391d001f35b37ec032b4ca8.jpg)

<a id="paper-0180"></a>

## DiScoFormer: Plug-In Density and Score Estimation with Transformers


### Images

![0093a80130ba6c6d96b5d63bc256e011d89c40426d79751e5a41cec7c2f5db3e.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/0093a80130ba6c6d96b5d63bc256e011d89c40426d79751e5a41cec7c2f5db3e.jpg)

![08e99c14753a13dc0b6c75490d9237626f5312d299f58f9a1d83f778d3eda6b2.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/08e99c14753a13dc0b6c75490d9237626f5312d299f58f9a1d83f778d3eda6b2.jpg)

![1322dc4a74fbcacb09d2b0edd9be864db547fd6512fa3f927667e64d9be05f98.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/1322dc4a74fbcacb09d2b0edd9be864db547fd6512fa3f927667e64d9be05f98.jpg)

![22baefc0375e97e393de14ef10d5d5abaaaca6f01477b508e148fde3c594c65c.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/22baefc0375e97e393de14ef10d5d5abaaaca6f01477b508e148fde3c594c65c.jpg)

![28528d6a2e1af78f9228ef15c1d637e9032d080b7f86c86963c735d739fe5917.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/28528d6a2e1af78f9228ef15c1d637e9032d080b7f86c86963c735d739fe5917.jpg)

![371b796f430b4b2d023ab808c6f2051ec6d1b1572b803c0711321e3b81b8d2c8.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/371b796f430b4b2d023ab808c6f2051ec6d1b1572b803c0711321e3b81b8d2c8.jpg)

![48549c3943786a85bc8a07e723ac9b2ed7757612004eb6d29181b0d05abb6011.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/48549c3943786a85bc8a07e723ac9b2ed7757612004eb6d29181b0d05abb6011.jpg)

![50a127d818497ae83026af69f8483543e63a0f3bed465afc9b54e45077a81d20.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/50a127d818497ae83026af69f8483543e63a0f3bed465afc9b54e45077a81d20.jpg)

![5e99555f4be5b7d3da8cc4bc4c4fa9d4ec291e4bfab9d1abdc8b8c6b0b0ffc6f.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/5e99555f4be5b7d3da8cc4bc4c4fa9d4ec291e4bfab9d1abdc8b8c6b0b0ffc6f.jpg)

![88f8b3ee51e8b2ed0d5159b36d6772e275fd6f9d9bd64ef945a895ecc87fc2c8.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/88f8b3ee51e8b2ed0d5159b36d6772e275fd6f9d9bd64ef945a895ecc87fc2c8.jpg)

![a448b2634ac625b370b88dddd9e5296d453f03285ab105ff60eb6de5b5eb1097.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/a448b2634ac625b370b88dddd9e5296d453f03285ab105ff60eb6de5b5eb1097.jpg)

![aa9783e4a6bb4cdde2aedd2a47a947c529fcb31e78f263bacce4937c332b4987.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/aa9783e4a6bb4cdde2aedd2a47a947c529fcb31e78f263bacce4937c332b4987.jpg)

![cc5921bc991ec5cf0f152edade12259cece6603539df292a72007730b66bdec1.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/cc5921bc991ec5cf0f152edade12259cece6603539df292a72007730b66bdec1.jpg)

![ea7a4652dac664308141e7957f7ae42e8c420f16296929104e57a1ae6ef1c918.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/ea7a4652dac664308141e7957f7ae42e8c420f16296929104e57a1ae6ef1c918.jpg)

![f7e5d3a31431f05537749ade2da153cd7f97798c8e10d07141a70890ae9ac704.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/images/f7e5d3a31431f05537749ade2da153cd7f97798c8e10d07141a70890ae9ac704.jpg)

### Tables

![561fbbe914f4b6038a6b3f376b9bf7259556e91f64ae11ebccc625c2e67ff258.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/tables/561fbbe914f4b6038a6b3f376b9bf7259556e91f64ae11ebccc625c2e67ff258.jpg)

![7ae94e49bfa3b7e6a956385912df636d04826df7050d0f4df2a36b3e77d8c471.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/tables/7ae94e49bfa3b7e6a956385912df636d04826df7050d0f4df2a36b3e77d8c471.jpg)

![9b6301c1e0ce453a7485e5b4abe291e6ada6bb4cd9b99bd0fd99838c914414da.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/tables/9b6301c1e0ce453a7485e5b4abe291e6ada6bb4cd9b99bd0fd99838c914414da.jpg)

![c49bacad64c32513b09fd68ee5e5f8251ab9618c14e4d7981410ca51af3dabaa.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/tables/c49bacad64c32513b09fd68ee5e5f8251ab9618c14e4d7981410ca51af3dabaa.jpg)

![fe71180a0626c7c51a0fb0ea24a8e59b8d720489f0bab11b1632f6c7ea0d5cc8.jpg](../icml_results/182_DiScoFormer_%20Plug-In%20Density%20and%20Score%20Estimation%20with%20Transformers/tables/fe71180a0626c7c51a0fb0ea24a8e59b8d720489f0bab11b1632f6c7ea0d5cc8.jpg)

<a id="paper-0181"></a>

## Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions


### Images

![303f840fa84b2567f9bd273699da4a8a8cd6fdf68d62dc9df351cd3081fa817f.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/303f840fa84b2567f9bd273699da4a8a8cd6fdf68d62dc9df351cd3081fa817f.jpg)

![49e95afc90ab5ad33a720f53b82d97e979f749f7284c07173fef0988fd54e8c3.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/49e95afc90ab5ad33a720f53b82d97e979f749f7284c07173fef0988fd54e8c3.jpg)

![7d7d98c9cd4025f664f2fd75ab3a8e4e213a46a399e35fbf1c593490c1acff40.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/7d7d98c9cd4025f664f2fd75ab3a8e4e213a46a399e35fbf1c593490c1acff40.jpg)

![8f885e0f46169d8aa926f87c98c56a45a76ea5c07143d50ebdbc46167d992a0f.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/8f885e0f46169d8aa926f87c98c56a45a76ea5c07143d50ebdbc46167d992a0f.jpg)

![a3e55fe6867d20258cddbd0cc5389c60f221fb581f95cdc0407dcc5f049260c7.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/a3e55fe6867d20258cddbd0cc5389c60f221fb581f95cdc0407dcc5f049260c7.jpg)

![be154eda8cac66eb2fdb7c0710062154c80bf22684d2771941c5966c6f018f67.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/be154eda8cac66eb2fdb7c0710062154c80bf22684d2771941c5966c6f018f67.jpg)

![bef37ab302c1fdfa9c8cac181de5c0dd0cdccfad282cafcc688d56877276a2fd.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/bef37ab302c1fdfa9c8cac181de5c0dd0cdccfad282cafcc688d56877276a2fd.jpg)

![c4b726d8ff371e87c27ff08efd221e615dc2495ed8480597ae71c630d1131e20.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/c4b726d8ff371e87c27ff08efd221e615dc2495ed8480597ae71c630d1131e20.jpg)

![cac755c4b3b11edca2de5cd7b8656794007688d226aaf4d1a02f4af88da3a341.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/cac755c4b3b11edca2de5cd7b8656794007688d226aaf4d1a02f4af88da3a341.jpg)

![e09ac7c421c34a8f35cc356eab7638726f8e740a9f06e03d39b22d2f379a75a1.jpg](../icml_results/183_Revenue%20Guarantees%20of%20No-Swap-Regret%20Dynamics%20in%20First%20Price%20Auctions/images/e09ac7c421c34a8f35cc356eab7638726f8e740a9f06e03d39b22d2f379a75a1.jpg)

<a id="paper-0182"></a>

## Non-Euclidean Gradient Descent Operates at the Edge of Stability


### Images

![11680f768b7fbbf5552ab566e0e6340d5c108a17514a3a89dcf1a47fd14f2c73.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/11680f768b7fbbf5552ab566e0e6340d5c108a17514a3a89dcf1a47fd14f2c73.jpg)

![28e20c96e198bd10d73b6448e29ac7c705e560488c76d960dfe5b18765fafce3.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/28e20c96e198bd10d73b6448e29ac7c705e560488c76d960dfe5b18765fafce3.jpg)

![2b909ec9886044d147ba4b42a8327484aa2f3a6d5d90ab7fba3079167244877d.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/2b909ec9886044d147ba4b42a8327484aa2f3a6d5d90ab7fba3079167244877d.jpg)

![35e3ba0dd9409c770a71eedd4c5f7ab32a0c773895451bb6dc43f848037baa0c.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/35e3ba0dd9409c770a71eedd4c5f7ab32a0c773895451bb6dc43f848037baa0c.jpg)

![3e115991c9070270ba3d0916e5972071fcdced9e004fa8fd147e971dc916b1fc.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/3e115991c9070270ba3d0916e5972071fcdced9e004fa8fd147e971dc916b1fc.jpg)

![543990d8b484a5e7dc0e164cedb19ac053db22013af5975327d0064c3775cf0b.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/543990d8b484a5e7dc0e164cedb19ac053db22013af5975327d0064c3775cf0b.jpg)

![5fcef690b98d1bcecb6bb832989e6b768c2248b3a9fe94c173ac496956dd894f.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/5fcef690b98d1bcecb6bb832989e6b768c2248b3a9fe94c173ac496956dd894f.jpg)

![681618a94926f8090a8f2ffc10ad95a263456d4584e3202d68020bcee7e8b9ec.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/681618a94926f8090a8f2ffc10ad95a263456d4584e3202d68020bcee7e8b9ec.jpg)

![8b9f1c05aa10d9c898b6ca4216a54462497d44b0e1f49eb3d88b7119c5bc1c82.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/8b9f1c05aa10d9c898b6ca4216a54462497d44b0e1f49eb3d88b7119c5bc1c82.jpg)

![9153dd095389e70a28365309996ead195dab28523225f43572bb6184604b6246.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/9153dd095389e70a28365309996ead195dab28523225f43572bb6184604b6246.jpg)

![a619263cfeb7eaab6b7de9261ac665854fae1e41cd3a3871b0e19b5b723f319b.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/a619263cfeb7eaab6b7de9261ac665854fae1e41cd3a3871b0e19b5b723f319b.jpg)

![b4bfeb38c47198120acc541f88d3a7fa863baf3286adc335312bd9acd729263e.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/b4bfeb38c47198120acc541f88d3a7fa863baf3286adc335312bd9acd729263e.jpg)

![bb1c081950314da5b82e9ee55adb33994053cb009cf7d692d78b0b5ef7849e36.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/bb1c081950314da5b82e9ee55adb33994053cb009cf7d692d78b0b5ef7849e36.jpg)

![bd0e5dee144da398c6996c2dcaf9133b253da71f21a3c96bbd466beb07389e53.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/bd0e5dee144da398c6996c2dcaf9133b253da71f21a3c96bbd466beb07389e53.jpg)

![c1f1b2a463dc55a5d59ec7bc782c651d07be11c7898ed301ac2b41427bb55eb8.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/c1f1b2a463dc55a5d59ec7bc782c651d07be11c7898ed301ac2b41427bb55eb8.jpg)

![c45813db4ede164efa19e931e16f2dbc9c732a6a498cc3b65a267649daaee15b.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/c45813db4ede164efa19e931e16f2dbc9c732a6a498cc3b65a267649daaee15b.jpg)

![c8733cfe7b0d5d2ab4fcd5ef8d6c41f92c63cabe2f64d4a4bb55273094dfcf61.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/c8733cfe7b0d5d2ab4fcd5ef8d6c41f92c63cabe2f64d4a4bb55273094dfcf61.jpg)

![f73945c6d0b8a2384c110f7a5a303894457fa329c79a591d711c81bafc3cbd1f.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/f73945c6d0b8a2384c110f7a5a303894457fa329c79a591d711c81bafc3cbd1f.jpg)

![f78a6a142deaff804461d360f99d021dcd93b963e5071d75172d605c91661db7.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/images/f78a6a142deaff804461d360f99d021dcd93b963e5071d75172d605c91661db7.jpg)

### Tables

![0a38c3541e31a4c0664e0ff4e1290a1870e050e383b459474941eba4d874ee80.jpg](../icml_results/184_Non-Euclidean%20Gradient%20Descent%20Operates%20at%20the%20Edge%20of%20Stability/tables/0a38c3541e31a4c0664e0ff4e1290a1870e050e383b459474941eba4d874ee80.jpg)

<a id="paper-0183"></a>

## h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning


### Images

![37741f59f550cc8e26a7d299e5d2c97551a784f85e31735a560f36112a862450.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/37741f59f550cc8e26a7d299e5d2c97551a784f85e31735a560f36112a862450.jpg)

![41a7e7941dda9c57a262d4f1681604c547ad6150117a0e714b204fcc727061f1.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/41a7e7941dda9c57a262d4f1681604c547ad6150117a0e714b204fcc727061f1.jpg)

![547c8f3267640fe1bfb7624b54cad4a01af0a07e6439c058f2d7576bbdc90aa1.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/547c8f3267640fe1bfb7624b54cad4a01af0a07e6439c058f2d7576bbdc90aa1.jpg)

![68748bd5f28e80326668494288a43947197a4a75cf100cae911a7c45f530da8e.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/68748bd5f28e80326668494288a43947197a4a75cf100cae911a7c45f530da8e.jpg)

![a9b8d6b2994709d4d3b5ef3c06490199741eda6e85215f06f5ffc31041fdae03.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/a9b8d6b2994709d4d3b5ef3c06490199741eda6e85215f06f5ffc31041fdae03.jpg)

![bb27cfa0f21d190a33071ccfa5b526df6258c88f891af7fb0273d550a21d82cd.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/bb27cfa0f21d190a33071ccfa5b526df6258c88f891af7fb0273d550a21d82cd.jpg)

![c14bd9840806ef7daac41de566f501c33134293e658355c51283ed3a30216df3.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/c14bd9840806ef7daac41de566f501c33134293e658355c51283ed3a30216df3.jpg)

![c91ddf3bd3f5ea1bc1cc83051eac157a91cb4272a93a29547b11eed8ec1f3704.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/images/c91ddf3bd3f5ea1bc1cc83051eac157a91cb4272a93a29547b11eed8ec1f3704.jpg)

### Tables

![122377bc2d53629296d802f217e1d433e39e73ec32e7634fa4b2d29adfbc0357.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/122377bc2d53629296d802f217e1d433e39e73ec32e7634fa4b2d29adfbc0357.jpg)

![2804c58b161bd8800fb6f49cbb0e0e22b31e9853b76684fbccb91bd3b32d689d.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/2804c58b161bd8800fb6f49cbb0e0e22b31e9853b76684fbccb91bd3b32d689d.jpg)

![329d82660a827fc4dc121d8dc4a005289fe10124c79e617c2be8a4118e43efa8.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/329d82660a827fc4dc121d8dc4a005289fe10124c79e617c2be8a4118e43efa8.jpg)

![421c14961eee1d8e7591709bc1c1983c81f5b264c5fcee4ebc2e168de53f5e95.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/421c14961eee1d8e7591709bc1c1983c81f5b264c5fcee4ebc2e168de53f5e95.jpg)

![552886cc6eca91dc50a01b3a202fed54cb4553f7d016275442e7a44484ed5c2e.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/552886cc6eca91dc50a01b3a202fed54cb4553f7d016275442e7a44484ed5c2e.jpg)

![5b202f3c0c62081538d05338761a6d01762b2126596d002ed4b8ebeb9dce9a91.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/5b202f3c0c62081538d05338761a6d01762b2126596d002ed4b8ebeb9dce9a91.jpg)

![74fed25e9d753b95570136785c7f77816d9ceb9d2878b4713a89c054fc28eb2e.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/74fed25e9d753b95570136785c7f77816d9ceb9d2878b4713a89c054fc28eb2e.jpg)

![85bc564ba46af4ba1ded3d4acaac26eb6d6fac9b2a0c461897b5573ceaeee71e.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/85bc564ba46af4ba1ded3d4acaac26eb6d6fac9b2a0c461897b5573ceaeee71e.jpg)

![9ed402033dbc084d05c9f944ebea1fda289959417ee19a0ed2bf701df1c08338.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/9ed402033dbc084d05c9f944ebea1fda289959417ee19a0ed2bf701df1c08338.jpg)

![df44cb57a2445da8161d42e9c13e5c80ae4bc9c640c76926b11aed30515ff292.jpg](../icml_results/185_h1_%20Bootstrapping%20LLMs%20to%20Reason%20over%20Longer%20Horizons%20via%20Reinforcement%20Learning/tables/df44cb57a2445da8161d42e9c13e5c80ae4bc9c640c76926b11aed30515ff292.jpg)

<a id="paper-0184"></a>

## Learning Unmasking Policies for Diffusion Language Models


### Images

![015fe462dc43e457794cfb540749fc3eb9f6307860908d611fb670bff2ebb16c.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/015fe462dc43e457794cfb540749fc3eb9f6307860908d611fb670bff2ebb16c.jpg)

![0eb702fad66eafdc3344209a8cf89973be5f919ee6c39c45ada5a084ad90c44b.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/0eb702fad66eafdc3344209a8cf89973be5f919ee6c39c45ada5a084ad90c44b.jpg)

![1ee361260cc21bd955a41044ab5eea178d48c7e79804ebaa215e8291a33e0401.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/1ee361260cc21bd955a41044ab5eea178d48c7e79804ebaa215e8291a33e0401.jpg)

![24662de66ebbb2fa468ea46c6eaa88bc440e8c37bed4750a6c493b42cce95870.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/24662de66ebbb2fa468ea46c6eaa88bc440e8c37bed4750a6c493b42cce95870.jpg)

![25f0d689e9f954303a4b8782636bd1d916144e1017e449786316c3b546c6797e.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/25f0d689e9f954303a4b8782636bd1d916144e1017e449786316c3b546c6797e.jpg)

![2d9d973cb4f25e348da7961aa58ac9b0fe15289b3ebb5d74c2267d8f396d2165.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/2d9d973cb4f25e348da7961aa58ac9b0fe15289b3ebb5d74c2267d8f396d2165.jpg)

![301ad71c37197c1270ba932f62aed3b2947474e43cfa211bbd4d86305016129e.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/301ad71c37197c1270ba932f62aed3b2947474e43cfa211bbd4d86305016129e.jpg)

![31e8cf81c38613525fb4a9fdcc049b49db49ba424c8f731f4ab9345fc7722ee3.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/31e8cf81c38613525fb4a9fdcc049b49db49ba424c8f731f4ab9345fc7722ee3.jpg)

![376f12304fff42cd6229a62782134cb5f086aeba6bb33937b5d810bbaafd9738.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/376f12304fff42cd6229a62782134cb5f086aeba6bb33937b5d810bbaafd9738.jpg)

![3bc85dfd59abf10c3052c47bc06bf3998e6e4214e14ba1cb07b0a8f666a4b420.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/3bc85dfd59abf10c3052c47bc06bf3998e6e4214e14ba1cb07b0a8f666a4b420.jpg)

![3c2fdf2705b1107b01cff4ab90614e245668f7991f8d2890651b1357456910a5.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/3c2fdf2705b1107b01cff4ab90614e245668f7991f8d2890651b1357456910a5.jpg)

![45b83d6e91635390cd32156b2abd782985eeefc571fdca6dfa8881df89fcf4a8.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/45b83d6e91635390cd32156b2abd782985eeefc571fdca6dfa8881df89fcf4a8.jpg)

![5220daf83181ce98aee06c93597b1d592e5fde501a896f6947ab943858fcc7f2.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/5220daf83181ce98aee06c93597b1d592e5fde501a896f6947ab943858fcc7f2.jpg)

![59ecac0267a0623a266728e405d0cb667f999a5dc025303f22192711b2331ad6.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/59ecac0267a0623a266728e405d0cb667f999a5dc025303f22192711b2331ad6.jpg)

![62700a1170531ba1915b24c4dee22b9fae3b973778dca986b317bb17db11a118.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/62700a1170531ba1915b24c4dee22b9fae3b973778dca986b317bb17db11a118.jpg)

![65131526835a01ff3f7b8bfb000ed3953d672a8118c997165e7948843aae11bf.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/65131526835a01ff3f7b8bfb000ed3953d672a8118c997165e7948843aae11bf.jpg)

![6b518d0d19f6bcc40304db45943f02b82167afdefd62f65b1035048069393d97.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/6b518d0d19f6bcc40304db45943f02b82167afdefd62f65b1035048069393d97.jpg)

![6cf42025cc908d120d17eef732fdcb211b89bbffdc51413f391e8deff22d32b2.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/6cf42025cc908d120d17eef732fdcb211b89bbffdc51413f391e8deff22d32b2.jpg)

![74d7b1e2026c48a7780a7eebb3c43f796fcbe54094ba9b10a2bc9a4c781ef490.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/74d7b1e2026c48a7780a7eebb3c43f796fcbe54094ba9b10a2bc9a4c781ef490.jpg)

![7cb2248d696d147a9f52eec46d9a1896b2a588f4e5ff11d8ea916027c1091068.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/7cb2248d696d147a9f52eec46d9a1896b2a588f4e5ff11d8ea916027c1091068.jpg)

![944fd5abdfd1c12553dc8b6f17be9d5fbb05afd3782e4aabd775e1183e895f1c.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/944fd5abdfd1c12553dc8b6f17be9d5fbb05afd3782e4aabd775e1183e895f1c.jpg)

![95c849699944a3928981c39dc9e764eda242e31fec0d68d80dcd51e4f9b47481.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/95c849699944a3928981c39dc9e764eda242e31fec0d68d80dcd51e4f9b47481.jpg)

![ac94e318b8ee49a09b4dce561d2bcf35c8266cbe5f98b4fa57c5c863946d37df.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/ac94e318b8ee49a09b4dce561d2bcf35c8266cbe5f98b4fa57c5c863946d37df.jpg)

![b37e112b6003235f799fc375fdfa1e460977151938bb9be0ead777815cc253a5.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/b37e112b6003235f799fc375fdfa1e460977151938bb9be0ead777815cc253a5.jpg)

![b9156dcd571bbdf635ff59b6d897c1843b6c5f85a9fe4ea96b1a6551dd6feac6.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/b9156dcd571bbdf635ff59b6d897c1843b6c5f85a9fe4ea96b1a6551dd6feac6.jpg)

![bc5991b15bdeefd6f20aeeb484939f37d05e194610a225ae2e5b6569d93fcc8c.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/bc5991b15bdeefd6f20aeeb484939f37d05e194610a225ae2e5b6569d93fcc8c.jpg)

![bf4e880d7554fad9e2077291b3bd4531abbaf752c9a8a3aa0236843b15b50d50.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/bf4e880d7554fad9e2077291b3bd4531abbaf752c9a8a3aa0236843b15b50d50.jpg)

![c0b24d7e35322a0daa89f11453fb7863b54f3cd43a0d510f9be93aeb668b49e1.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/c0b24d7e35322a0daa89f11453fb7863b54f3cd43a0d510f9be93aeb668b49e1.jpg)

![c3f4c148a9cd0257fa91f09b630fecd07f35edc856cb244d438281cfae07d61c.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/c3f4c148a9cd0257fa91f09b630fecd07f35edc856cb244d438281cfae07d61c.jpg)

![cd92b39b33919ff0ecf4c90d3d6680315b385681425296dda23b68a9b17fcc19.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/cd92b39b33919ff0ecf4c90d3d6680315b385681425296dda23b68a9b17fcc19.jpg)

![d4ff4e85d6586ca4754add0a69a3b41af871fb37284c83ad5d1509a65f712b0a.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/d4ff4e85d6586ca4754add0a69a3b41af871fb37284c83ad5d1509a65f712b0a.jpg)

![da5a9810568ed8a92a1d477f5fedec04d135bca4c8f10505556eb3c6ca429561.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/da5a9810568ed8a92a1d477f5fedec04d135bca4c8f10505556eb3c6ca429561.jpg)

![e24484d89ac0c84e40fe7bd01263fda40563e66ba0df671085d4784003595c63.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/e24484d89ac0c84e40fe7bd01263fda40563e66ba0df671085d4784003595c63.jpg)

![f7962f167e7cddffcdfab90b541efa98abb2ce61f3f7bc90ce22a3ce7828bbc2.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/images/f7962f167e7cddffcdfab90b541efa98abb2ce61f3f7bc90ce22a3ce7828bbc2.jpg)

### Tables

![00713492ceafb7deb60e1fdc45b5c6d38086c1f7578664247575449c9fae77c7.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/00713492ceafb7deb60e1fdc45b5c6d38086c1f7578664247575449c9fae77c7.jpg)

![14b5c1d8e3acb63ca991fc75beaf425d35eef151b12a6f30ec434dc663ae46ed.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/14b5c1d8e3acb63ca991fc75beaf425d35eef151b12a6f30ec434dc663ae46ed.jpg)

![31bce404d45c97b289f91787456f75f8e5167b34c0d4821ecf21dd2a8358c1f2.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/31bce404d45c97b289f91787456f75f8e5167b34c0d4821ecf21dd2a8358c1f2.jpg)

![81b415826d2f838334774a66d5d155dce3cfbbaf173f1d479c49e0154e2f3ef9.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/81b415826d2f838334774a66d5d155dce3cfbbaf173f1d479c49e0154e2f3ef9.jpg)

![ac7984b9c56ec9d78a10ee3510c08ca9aec40968763a166b89b0434eb155684c.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/ac7984b9c56ec9d78a10ee3510c08ca9aec40968763a166b89b0434eb155684c.jpg)

![e9e64bf71fd3e0411d65de530e39f263350aefad6de1cecf09b465a3ce4e584e.jpg](../icml_results/186_Learning%20Unmasking%20Policies%20for%20Diffusion%20Language%20Models/tables/e9e64bf71fd3e0411d65de530e39f263350aefad6de1cecf09b465a3ce4e584e.jpg)

<a id="paper-0185"></a>

## Efficient Diffusion Models under Nonconvex Equality and Inequality constraints via Landing


### Images

![10488edaac7ad5dfe12bd06f352caf27d796caf72ad41603cebe362dfd1fc2d9.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/10488edaac7ad5dfe12bd06f352caf27d796caf72ad41603cebe362dfd1fc2d9.jpg)

![15017723c4c0846fc2074f8dd0afc41a1713f65ddad46795b8171a9c2181f446.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/15017723c4c0846fc2074f8dd0afc41a1713f65ddad46795b8171a9c2181f446.jpg)

![20278eb1b8308fa3df47cbf6a544ab98a271362123f7fa992a10cc54f59c6000.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/20278eb1b8308fa3df47cbf6a544ab98a271362123f7fa992a10cc54f59c6000.jpg)

![25bafa0926f9c9506b4c7a6242636edee663433e760fbbf5660851c86001c3f4.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/25bafa0926f9c9506b4c7a6242636edee663433e760fbbf5660851c86001c3f4.jpg)

![358456cfec05ce687ad36bf6068f130d1edb1a27b17759b3e8b0e293392e1a4e.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/358456cfec05ce687ad36bf6068f130d1edb1a27b17759b3e8b0e293392e1a4e.jpg)

![40976585dfce7095ac5c22aae52823433772cdaf29bbe68b4bcbea6f323e7581.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/40976585dfce7095ac5c22aae52823433772cdaf29bbe68b4bcbea6f323e7581.jpg)

![af2e246696cef42a194825cb003eba668e3512aa74f7b6de2f1fdbc7530f4fdf.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/af2e246696cef42a194825cb003eba668e3512aa74f7b6de2f1fdbc7530f4fdf.jpg)

![b5d4100e38107f1169a05a9a93deec66f5b5a103042b5c31d96e748a5f41d259.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/b5d4100e38107f1169a05a9a93deec66f5b5a103042b5c31d96e748a5f41d259.jpg)

![e6ed4adbfce17d1ef78fea076f126ec8cfb30863f24b6b22663deacb0024d30b.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/images/e6ed4adbfce17d1ef78fea076f126ec8cfb30863f24b6b22663deacb0024d30b.jpg)

### Tables

![072c5d4786e0fb1c25f7f7d539c76c2fc4447049c86fa2c5c5e5a701377d9cae.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/072c5d4786e0fb1c25f7f7d539c76c2fc4447049c86fa2c5c5e5a701377d9cae.jpg)

![11b83a917c01fd16d1d780bdbf9337297c713a712391d295c3d09c686e521d6b.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/11b83a917c01fd16d1d780bdbf9337297c713a712391d295c3d09c686e521d6b.jpg)

![190a89942a0b2d1678f662614807a951e314a06a10c6146c05821e929457d9fc.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/190a89942a0b2d1678f662614807a951e314a06a10c6146c05821e929457d9fc.jpg)

![2e8da6b64114d9de2ae6684c69b37b3d340ad547fef1d586a019542579c499b1.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/2e8da6b64114d9de2ae6684c69b37b3d340ad547fef1d586a019542579c499b1.jpg)

![3cd377691166bff25a0d3e30c3dd8b472bbc1b9610efcf24336bb32756346f22.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/3cd377691166bff25a0d3e30c3dd8b472bbc1b9610efcf24336bb32756346f22.jpg)

![7cf0ad8f0402ddd8b2ce0fdeca1f995646c88496924b47a48376a0398d85daea.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/7cf0ad8f0402ddd8b2ce0fdeca1f995646c88496924b47a48376a0398d85daea.jpg)

![895e104317bb930cd1f86ba376e3fe4bb83ae2a5ccd452206d30bce29e50cd86.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/895e104317bb930cd1f86ba376e3fe4bb83ae2a5ccd452206d30bce29e50cd86.jpg)

![8d127b633edc6fa7b9778c2da137029f4e970add0a38c8889f082ce0ae385c53.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/8d127b633edc6fa7b9778c2da137029f4e970add0a38c8889f082ce0ae385c53.jpg)

![91bc66ecbeb44af95742878ae9e38e25df285ecd9487203b5c8a8409926e233d.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/91bc66ecbeb44af95742878ae9e38e25df285ecd9487203b5c8a8409926e233d.jpg)

![b31672cbdede09dc9833096fcae38cce4965f47acf7c2b7d6ea191269f229da2.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/b31672cbdede09dc9833096fcae38cce4965f47acf7c2b7d6ea191269f229da2.jpg)

![cbf0c509f5a3cd5486fcc5907e9ba1518d98dc62667d16a1fb3ff0ed4fc8f632.jpg](../icml_results/187_Efficient%20Diffusion%20Models%20under%20Nonconvex%20Equality%20and%20Inequality%20constraints%20via%20Landing/tables/cbf0c509f5a3cd5486fcc5907e9ba1518d98dc62667d16a1fb3ff0ed4fc8f632.jpg)

<a id="paper-0186"></a>

## OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning


### Images

![108928a040ece6b978c16c9a63533414c525eab539e6d02abb2d1aaa9aebd67c.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/108928a040ece6b978c16c9a63533414c525eab539e6d02abb2d1aaa9aebd67c.jpg)

![11dd99effa882f35efbf30c64645e5f3dc8d32c9f9842258faded086ae2a62cd.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/11dd99effa882f35efbf30c64645e5f3dc8d32c9f9842258faded086ae2a62cd.jpg)

![22c9ab50dab562d820171887ce5d62a37820fbaf0e877c36a43fa2e9224a7ae9.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/22c9ab50dab562d820171887ce5d62a37820fbaf0e877c36a43fa2e9224a7ae9.jpg)

![47fea1bf37034e484246bcdeced8232d91182295d233254523bb28c325ab6b83.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/47fea1bf37034e484246bcdeced8232d91182295d233254523bb28c325ab6b83.jpg)

![500e099d665ebc11be72ff8b5a2656e8394af95967483a66a5097a553952b2ff.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/500e099d665ebc11be72ff8b5a2656e8394af95967483a66a5097a553952b2ff.jpg)

![59df6a02dca4d14b4a7ad39db3c8b16c86c271cb2d3520afec7127438b34ab8e.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/59df6a02dca4d14b4a7ad39db3c8b16c86c271cb2d3520afec7127438b34ab8e.jpg)

![60af3ac63afafee22903d7c6090422ae290fb38a1706d65a6db37a82fb80fd24.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/60af3ac63afafee22903d7c6090422ae290fb38a1706d65a6db37a82fb80fd24.jpg)

![782f83917299a8861032f0d959aba6f781a3fcc03fcd39943f999a42496a7221.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/782f83917299a8861032f0d959aba6f781a3fcc03fcd39943f999a42496a7221.jpg)

![7c88d2d12ed96609933ec96c5aa2f49c5d088401f2045c98d8b92beef2dcc1fd.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/7c88d2d12ed96609933ec96c5aa2f49c5d088401f2045c98d8b92beef2dcc1fd.jpg)

![7e8a0a076ad1a489336513776a40fb7bb687ff8c283b7e506d18ae6bb47dbfcc.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/7e8a0a076ad1a489336513776a40fb7bb687ff8c283b7e506d18ae6bb47dbfcc.jpg)

![83fb143a342356f238183fde1e1f7580032dccdeec68ff5e784d53dc06a8b9be.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/83fb143a342356f238183fde1e1f7580032dccdeec68ff5e784d53dc06a8b9be.jpg)

![af150e55385c9d6bc482004edf4196bc458f71a0f8dd98c0599079354ee5fbb6.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/af150e55385c9d6bc482004edf4196bc458f71a0f8dd98c0599079354ee5fbb6.jpg)

![b60dea106c75aa72b23a8ca6a8f8ec1f6e814f36b3627e59cfdea716599a6319.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/b60dea106c75aa72b23a8ca6a8f8ec1f6e814f36b3627e59cfdea716599a6319.jpg)

![b9cd48bdaf71a2d64dc1021517891a58c9f2465d2835c50d7e451bd99b69a0cb.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/b9cd48bdaf71a2d64dc1021517891a58c9f2465d2835c50d7e451bd99b69a0cb.jpg)

![c5f472ac8a83bedda6636ee709c2eafec6a38621723e40a6287a974573c0113f.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/c5f472ac8a83bedda6636ee709c2eafec6a38621723e40a6287a974573c0113f.jpg)

![dc952514aeedda3a362123a440b92ab3e626ec2bf84b15afda0206a65284b9d0.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/dc952514aeedda3a362123a440b92ab3e626ec2bf84b15afda0206a65284b9d0.jpg)

![ec80a4d5d4003d3974f128506633808f9cf1696b81a4740208e42883d04701ba.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/ec80a4d5d4003d3974f128506633808f9cf1696b81a4740208e42883d04701ba.jpg)

![f7f9f899dc9325c9c35fe86e0847f1701510be471a8165426982a95f53620958.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/f7f9f899dc9325c9c35fe86e0847f1701510be471a8165426982a95f53620958.jpg)

![fcde5f4b0f5887cae7349383d943ee89dc6a3934460240e6ccf4f52b972a23a5.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/images/fcde5f4b0f5887cae7349383d943ee89dc6a3934460240e6ccf4f52b972a23a5.jpg)

### Tables

![27c1db77c5da3cf31c6d5c38132d725f2caf10d9f40fd66c3f561ce365b5dab5.jpg](../icml_results/188_OXE-AugE_%20A%20Large-Scale%20Robot%20Augmentation%20of%20OXE%20for%20Scaling%20Cross-Embodiment%20Policy%20Learning/tables/27c1db77c5da3cf31c6d5c38132d725f2caf10d9f40fd66c3f561ce365b5dab5.jpg)

<a id="paper-0187"></a>

## Manifold-Optimal Guidance: A Unified Riemannian Control View of Diffusion Guidance


### Images

![03515e4b73ec92f6ccceff5605bdb2480e4db39b7b2d843d9e00bd96628f94f7.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/03515e4b73ec92f6ccceff5605bdb2480e4db39b7b2d843d9e00bd96628f94f7.jpg)

![0390e6c5c4b8f45aa064eca7aac3229a1aa3c446b1226b024cdc2ea390329e12.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/0390e6c5c4b8f45aa064eca7aac3229a1aa3c446b1226b024cdc2ea390329e12.jpg)

![19abd1c913e73287bf86f4c2cad33c722219ad7ed02a16bc0c94ed114bde8f65.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/19abd1c913e73287bf86f4c2cad33c722219ad7ed02a16bc0c94ed114bde8f65.jpg)

![1ff386e1c111db457bc26a8573318c16acf14de22b3eb15975d760990f6b7658.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/1ff386e1c111db457bc26a8573318c16acf14de22b3eb15975d760990f6b7658.jpg)

![36dea01e29affface9b48437046cd778556f59879c36025e92eaea6b0f1ccaac.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/36dea01e29affface9b48437046cd778556f59879c36025e92eaea6b0f1ccaac.jpg)

![795d1b5b4ade382aa347d67709cf7fb8aa671aef5a50a8cb5a0565ec580cf26e.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/795d1b5b4ade382aa347d67709cf7fb8aa671aef5a50a8cb5a0565ec580cf26e.jpg)

![f921dbbb730580b3afdad55ba905549f22367e9f327e727d0c4ab4eb66378678.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/images/f921dbbb730580b3afdad55ba905549f22367e9f327e727d0c4ab4eb66378678.jpg)

### Tables

![06a533eeaedeb1540aa8f2867f4d84bc486093388f32c72ef491325ab894b737.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/06a533eeaedeb1540aa8f2867f4d84bc486093388f32c72ef491325ab894b737.jpg)

![19ea638170cf94889debc94acc2cd448a77a064fdaea5c4c36fd9589f531cb09.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/19ea638170cf94889debc94acc2cd448a77a064fdaea5c4c36fd9589f531cb09.jpg)

![1ec390903a8fd627ae9ecd9984e8b8c07fa97da2941664fb6ea476fa2a0f5f3d.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/1ec390903a8fd627ae9ecd9984e8b8c07fa97da2941664fb6ea476fa2a0f5f3d.jpg)

![2220ec4b11936046f6a421708136bc752a53369a02fe7ba1874ec743e83fd3b3.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/2220ec4b11936046f6a421708136bc752a53369a02fe7ba1874ec743e83fd3b3.jpg)

![5d77e0a30d42a77018310ed851a1e3673543c232230d1acc72cb41f8b9dd1f25.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/5d77e0a30d42a77018310ed851a1e3673543c232230d1acc72cb41f8b9dd1f25.jpg)

![6aadbaca5f508b26550661246ba59814c9ae4cba60415787ef52c9fd8ad02ed9.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/6aadbaca5f508b26550661246ba59814c9ae4cba60415787ef52c9fd8ad02ed9.jpg)

![948a1f7d1a2f3847ca31f75495665308e7e5da3c354f7b31e4e315a9cd64a2b7.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/948a1f7d1a2f3847ca31f75495665308e7e5da3c354f7b31e4e315a9cd64a2b7.jpg)

![a1c5597ea7631d1581b0e21067af3e4e991558a8296fde95effb13d05e7d8c17.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/a1c5597ea7631d1581b0e21067af3e4e991558a8296fde95effb13d05e7d8c17.jpg)

![a3120a0e02baf61c0b652dc7bafe0cbad50b29569840c9acb540765035fc7c4d.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/a3120a0e02baf61c0b652dc7bafe0cbad50b29569840c9acb540765035fc7c4d.jpg)

![ee087c4aba9dd1e7e21c8340a40dc87fe4003fdf449cd511ded0cd2e0327662e.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/ee087c4aba9dd1e7e21c8340a40dc87fe4003fdf449cd511ded0cd2e0327662e.jpg)

![f9e69f525817524052f00f802f5ea4f6105da9c8a04a23fe2342cd3b51b87fc0.jpg](../icml_results/189_Manifold-Optimal%20Guidance_%20A%20Unified%20Riemannian%20Control%20View%20of%20Diffusion%20Guidance/tables/f9e69f525817524052f00f802f5ea4f6105da9c8a04a23fe2342cd3b51b87fc0.jpg)

<a id="paper-0188"></a>

## Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance


### Images

![008e43b99b71d7cb4e19caa2815b2f89c857347881168dc103977db4682d6930.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/008e43b99b71d7cb4e19caa2815b2f89c857347881168dc103977db4682d6930.jpg)

![019c120c1f05483e730bb5b739dea376b7135ab1f11f47710277aeeee8e463e9.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/019c120c1f05483e730bb5b739dea376b7135ab1f11f47710277aeeee8e463e9.jpg)

![099f6fcc8bf303880619de4a7ba0fe6e6a077a5beb9915aaea218b726449db10.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/099f6fcc8bf303880619de4a7ba0fe6e6a077a5beb9915aaea218b726449db10.jpg)

![18bf9f963311eeba381f06e5bcc5de798accf975c778f26a5aed9f53e4ec2738.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/18bf9f963311eeba381f06e5bcc5de798accf975c778f26a5aed9f53e4ec2738.jpg)

![574966850dba0866753b37d6f7f2d4678a4bbdc05cfa29fbd96f67b3d4c49f1b.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/574966850dba0866753b37d6f7f2d4678a4bbdc05cfa29fbd96f67b3d4c49f1b.jpg)

![5fa67c0bbc110c9090520cb10321d5d829aa2f8632c3c4f3bb606960a36172bb.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/5fa67c0bbc110c9090520cb10321d5d829aa2f8632c3c4f3bb606960a36172bb.jpg)

![6979d2a9cee708c6c37c53a490b60195ea1b78da63895017514e9e7e5820887a.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/6979d2a9cee708c6c37c53a490b60195ea1b78da63895017514e9e7e5820887a.jpg)

![7a34b2c466f281bf109b37c6c10e6d7795f7a1dace0b56a06218190bf58a2075.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/7a34b2c466f281bf109b37c6c10e6d7795f7a1dace0b56a06218190bf58a2075.jpg)

![87c3fe42d630d4a307aa20127535b463839dd27ed1a7782c2382e16db70067f6.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/87c3fe42d630d4a307aa20127535b463839dd27ed1a7782c2382e16db70067f6.jpg)

![8d4c1bdd2b2ed3ea0d8300093dd4d11c8d4befa0e456cf373422e22d8df2f165.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/8d4c1bdd2b2ed3ea0d8300093dd4d11c8d4befa0e456cf373422e22d8df2f165.jpg)

![998bde8e33ed27db2455b08e4e6e82d0ef7f5199ee961c4da031df960b083c26.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/998bde8e33ed27db2455b08e4e6e82d0ef7f5199ee961c4da031df960b083c26.jpg)

![ace3c9f406e3821f24381848de38f53910bc0f5d4d5feb905ee7d84a076d1030.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/ace3c9f406e3821f24381848de38f53910bc0f5d4d5feb905ee7d84a076d1030.jpg)

![de65ac65b88e7cc5403eb988f0b472c109c48e7bd1853d203aa6723a2c6f162f.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/de65ac65b88e7cc5403eb988f0b472c109c48e7bd1853d203aa6723a2c6f162f.jpg)

![ed0caebe69c2744b0a6a914390c412a8d537eca89f7fdaf2248112c073e3c367.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/ed0caebe69c2744b0a6a914390c412a8d537eca89f7fdaf2248112c073e3c367.jpg)

![f2cd3d8aac706aa7b346e1df4a0ccc7cca3e9bb8a7f5302c8e4ae90e38a22b31.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/images/f2cd3d8aac706aa7b346e1df4a0ccc7cca3e9bb8a7f5302c8e4ae90e38a22b31.jpg)

### Tables

![1588cfde68197f1f840b737aaf2c19052631591bc98ded2290fabf4843fed72b.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/1588cfde68197f1f840b737aaf2c19052631591bc98ded2290fabf4843fed72b.jpg)

![21fa834c2384bee3d3aa3eebcd9d57bbdc242a1ddf581cd99974a8e844b806c9.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/21fa834c2384bee3d3aa3eebcd9d57bbdc242a1ddf581cd99974a8e844b806c9.jpg)

![36926ed4259ea056ffba5a0eb36a1504ce5026ccc362aeb4228e877dae054f8d.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/36926ed4259ea056ffba5a0eb36a1504ce5026ccc362aeb4228e877dae054f8d.jpg)

![39cd2b67000d8b6ab8ab11db0e925d2e920555ae11f721b2ba8c9087a58fa88e.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/39cd2b67000d8b6ab8ab11db0e925d2e920555ae11f721b2ba8c9087a58fa88e.jpg)

![57441ee6e7581dd3f972d2df3f313067016a42a454a06111bf433835f0ee2217.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/57441ee6e7581dd3f972d2df3f313067016a42a454a06111bf433835f0ee2217.jpg)

![5c79f1b0017c7d91d0b5111486018eb16a1a99c454e9b1f92a5b4b8e885ef600.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/5c79f1b0017c7d91d0b5111486018eb16a1a99c454e9b1f92a5b4b8e885ef600.jpg)

![63c18dc833931b6e4e6457d0165aef52966e82cf2cfab0eed05dfe5a03cda2af.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/63c18dc833931b6e4e6457d0165aef52966e82cf2cfab0eed05dfe5a03cda2af.jpg)

![642f78e9ca5197e3fbc372af4dd16e9defcc582e3c346e9be3060e83f7ea8b95.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/642f78e9ca5197e3fbc372af4dd16e9defcc582e3c346e9be3060e83f7ea8b95.jpg)

![6c69385e7914d57aa1514ebef8fac3d770c5c40ca73e78402b909552313d7b40.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/6c69385e7914d57aa1514ebef8fac3d770c5c40ca73e78402b909552313d7b40.jpg)

![72d27680124d422e8a5035d53a3c2e7389501628dd24926bc9935030ca94118c.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/72d27680124d422e8a5035d53a3c2e7389501628dd24926bc9935030ca94118c.jpg)

![81f289404f5c817f54435e89c6cbea5285713c054f91a9866ca3e19e0262c72e.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/81f289404f5c817f54435e89c6cbea5285713c054f91a9866ca3e19e0262c72e.jpg)

![83b5304a1e00953c6862a8742d6bef2ff4efdeff54ff81b94434ed97eab5248f.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/83b5304a1e00953c6862a8742d6bef2ff4efdeff54ff81b94434ed97eab5248f.jpg)

![929be7b2d195f5b51a0cbec7d4f8eda1bbdf3f16a2156bb0e38305ca68dbc346.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/929be7b2d195f5b51a0cbec7d4f8eda1bbdf3f16a2156bb0e38305ca68dbc346.jpg)

![a54f05e748d1282b155f6b6b950e4dc54401b8b80528bc414d45bd00161ec73c.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/a54f05e748d1282b155f6b6b950e4dc54401b8b80528bc414d45bd00161ec73c.jpg)

![a61c060b1711fd5c42543e9059810e1a44a8a0e528624f551a4deac4af98a47d.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/a61c060b1711fd5c42543e9059810e1a44a8a0e528624f551a4deac4af98a47d.jpg)

![b9e48fba6f446a463c0309a82834170a1a1b614be8bd112f48b2e2b01e8357c0.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/b9e48fba6f446a463c0309a82834170a1a1b614be8bd112f48b2e2b01e8357c0.jpg)

![d1e7b0ce058d55d3c4ad30dfdbd3424483171138c85a375dbd8dc5c476c9e7a6.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/d1e7b0ce058d55d3c4ad30dfdbd3424483171138c85a375dbd8dc5c476c9e7a6.jpg)

![e71432771c548192574ee8d859149222ebf9c2ef2a142411b3ef1462919a199e.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/e71432771c548192574ee8d859149222ebf9c2ef2a142411b3ef1462919a199e.jpg)

![f269fa5c31f1b17654c716a600db995a77437d32fcac399810e69d5af6fe39d8.jpg](../icml_results/190_Weak%20Diffusion%20Priors%20Can%20Still%20Achieve%20Strong%20Inverse-Problem%20Performance/tables/f269fa5c31f1b17654c716a600db995a77437d32fcac399810e69d5af6fe39d8.jpg)

<a id="paper-0189"></a>

## Incremental BPE Tokenization


### Images

![00d920f064fd6967f0337a9f6520267edbd8e9b98d23dcbd9e503b6be23d814b.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/00d920f064fd6967f0337a9f6520267edbd8e9b98d23dcbd9e503b6be23d814b.jpg)

![36eef9160f47f6f1ecc7b39429267b0b00ba4e9526850c3cd1d71b86d21c4781.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/36eef9160f47f6f1ecc7b39429267b0b00ba4e9526850c3cd1d71b86d21c4781.jpg)

![792587b5ea842d4e4a9486dea6b54a35450b9b40c503bf257ad320ab0b6a9db7.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/792587b5ea842d4e4a9486dea6b54a35450b9b40c503bf257ad320ab0b6a9db7.jpg)

![8aa00b8cb7166dcb9230a3c276a628825a0f244ff088d4c7c500e71551aa4caf.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/8aa00b8cb7166dcb9230a3c276a628825a0f244ff088d4c7c500e71551aa4caf.jpg)

![97ebe8e97b8ac244da09a4739b9d84a10478c88347e19a4a3b9b6246b2db63e0.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/97ebe8e97b8ac244da09a4739b9d84a10478c88347e19a4a3b9b6246b2db63e0.jpg)

![b430e84ef3c2a339865a57875821f7c2350e612a51a24fd1da02f67bce8a1896.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/b430e84ef3c2a339865a57875821f7c2350e612a51a24fd1da02f67bce8a1896.jpg)

![d438d0dcd0afdf04234f3bb879e6ad42ba8acc1fca1c864c90d9ec15cc313ca2.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/d438d0dcd0afdf04234f3bb879e6ad42ba8acc1fca1c864c90d9ec15cc313ca2.jpg)

![fae9e355e961910b80b67fb5ff91e1f0447caf573665fea12695a542fa090caa.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/images/fae9e355e961910b80b67fb5ff91e1f0447caf573665fea12695a542fa090caa.jpg)

### Tables

![041d80762ee1811c4dcd46733b948f43b02ce8b067f5b54ecf53f6fd846f6417.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/041d80762ee1811c4dcd46733b948f43b02ce8b067f5b54ecf53f6fd846f6417.jpg)

![274d7cf01f9e7c420fc669e410078707494f520d8260f21de196dce2cefbcf43.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/274d7cf01f9e7c420fc669e410078707494f520d8260f21de196dce2cefbcf43.jpg)

![6493bf812579279992de42796dd833a66e21c02da62953465f91070cf681eccc.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/6493bf812579279992de42796dd833a66e21c02da62953465f91070cf681eccc.jpg)

![7693826294b2480d6975096d9622df4ca616dfe3db28f7bf1e94f07bf045c2e7.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/7693826294b2480d6975096d9622df4ca616dfe3db28f7bf1e94f07bf045c2e7.jpg)

![7d654089b88408854259aa504f72d140d783d21ea83eda1f2e589282d1621925.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/7d654089b88408854259aa504f72d140d783d21ea83eda1f2e589282d1621925.jpg)

![96e2375c4a98748c99a981033ccccfa473a6dd4e7f407938b3dd6c12dfbdaf67.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/96e2375c4a98748c99a981033ccccfa473a6dd4e7f407938b3dd6c12dfbdaf67.jpg)

![be8f42416efc7e4ee5f0e85710cec846a78f82ff799ea5a5c59081a98232c47b.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/be8f42416efc7e4ee5f0e85710cec846a78f82ff799ea5a5c59081a98232c47b.jpg)

![cbc555b9934c083d5760c119d62ecdbe0c17727560f3a95ec720cdc326ac9a53.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/cbc555b9934c083d5760c119d62ecdbe0c17727560f3a95ec720cdc326ac9a53.jpg)

![d0405224b530af5ab7e2c4e0c9aa3cf9cb7291c2495aa7ca3a23ae4de92bfaaa.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/d0405224b530af5ab7e2c4e0c9aa3cf9cb7291c2495aa7ca3a23ae4de92bfaaa.jpg)

![d4dff605f9b62db12ed62466de913465dc1e7b77f30233a8f49afbb5ab00f3ce.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/d4dff605f9b62db12ed62466de913465dc1e7b77f30233a8f49afbb5ab00f3ce.jpg)

![e04bdc569041933104803991bbeed6c636d126b665cd98960f0883f3b3e18022.jpg](../icml_results/191_Incremental%20BPE%20Tokenization/tables/e04bdc569041933104803991bbeed6c636d126b665cd98960f0883f3b3e18022.jpg)

<a id="paper-0190"></a>

## Procedural Pretraining: Warming Up Language Models with Abstract Data


### Images

![00211f5197bad735887edc3eedb850fc655c462ebca8c68d3330db830f4253b7.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/00211f5197bad735887edc3eedb850fc655c462ebca8c68d3330db830f4253b7.jpg)

![0cb347c72aae091fc28f433cfe9f05f896f88a3e5d8af45ca31149e0b8b347cd.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/0cb347c72aae091fc28f433cfe9f05f896f88a3e5d8af45ca31149e0b8b347cd.jpg)

![125f6c4d65039ef4fe17d4a9484d500bdcc0acd7e179cfebd9510c1af0b57b34.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/125f6c4d65039ef4fe17d4a9484d500bdcc0acd7e179cfebd9510c1af0b57b34.jpg)

![1d7bda7b8ba9d308cbc5bcdde5be60db2fa67505a58a0bbb5950d007bed87d57.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/1d7bda7b8ba9d308cbc5bcdde5be60db2fa67505a58a0bbb5950d007bed87d57.jpg)

![2f1b2723b4b13511d44ab963a72bf35440e316a0230507c8f643512b3a7b3f0e.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/2f1b2723b4b13511d44ab963a72bf35440e316a0230507c8f643512b3a7b3f0e.jpg)

![4873af211e21525186f7d6c4b5a5d99a3d6fb163b177cf1003ac8a15e40f0ac7.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/4873af211e21525186f7d6c4b5a5d99a3d6fb163b177cf1003ac8a15e40f0ac7.jpg)

![50e1e37182883562a6c04a4be52c8ba36053f85062eca190c53a6663d32a6cd7.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/50e1e37182883562a6c04a4be52c8ba36053f85062eca190c53a6663d32a6cd7.jpg)

![537aa2adbb7a088ba3ee82eea8c22ffb21e218b1e097cc6ac1159da9af648bb4.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/537aa2adbb7a088ba3ee82eea8c22ffb21e218b1e097cc6ac1159da9af648bb4.jpg)

![53fe1ff0349315389202e78230ca9f9ac18325119879a37d0ee6e8c3c576ebad.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/53fe1ff0349315389202e78230ca9f9ac18325119879a37d0ee6e8c3c576ebad.jpg)

![6228800f7eb15c07782bff2a60878cc966c4e12ee0c5fbf5a053156edac4e232.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/6228800f7eb15c07782bff2a60878cc966c4e12ee0c5fbf5a053156edac4e232.jpg)

![652c9008267bc1e7d92089167829984cf1c263d20aee7cbbd0b67358b5c8f2ef.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/652c9008267bc1e7d92089167829984cf1c263d20aee7cbbd0b67358b5c8f2ef.jpg)

![83a3f6cc20691a156f34a8a2c98a19bc9de4e6cf42b8090121d47f51dfa0c482.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/83a3f6cc20691a156f34a8a2c98a19bc9de4e6cf42b8090121d47f51dfa0c482.jpg)

![86b68dab5a324b04de19d5ee5ef9c2a92948e0415a5522bc713f93974f90e54d.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/86b68dab5a324b04de19d5ee5ef9c2a92948e0415a5522bc713f93974f90e54d.jpg)

![972f3fcfea87d96d117930a5716af5ed4c783c74baf355dff0085de449e12ec8.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/972f3fcfea87d96d117930a5716af5ed4c783c74baf355dff0085de449e12ec8.jpg)

![ab6578a07c5d3748a520e133db09c0825119603a81e158f6806ef94746a59c03.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/ab6578a07c5d3748a520e133db09c0825119603a81e158f6806ef94746a59c03.jpg)

![acf359ce5da845e442a5b0f3e31de5eab2c31b49cb91462699fa2cfa57395d33.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/acf359ce5da845e442a5b0f3e31de5eab2c31b49cb91462699fa2cfa57395d33.jpg)

![b418005638ed6fe2c2b5769123ed1c36237e2444cc5fc472ee4aa2079370c52f.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/b418005638ed6fe2c2b5769123ed1c36237e2444cc5fc472ee4aa2079370c52f.jpg)

![bf2c0f44d680ceb19d6a483c6093d680fedc5394b03cd2a846c7995b77472272.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/bf2c0f44d680ceb19d6a483c6093d680fedc5394b03cd2a846c7995b77472272.jpg)

![c2a102c9a5ee71be79ca7e708fe996b69cd384372ff8abc057bff7a7c8e8231e.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/c2a102c9a5ee71be79ca7e708fe996b69cd384372ff8abc057bff7a7c8e8231e.jpg)

![d25f15c74ed74350dd5c6cb886b629fea51213f85c6f85c9c80b141f52afb92b.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/d25f15c74ed74350dd5c6cb886b629fea51213f85c6f85c9c80b141f52afb92b.jpg)

![dc88e711c4b7d42fe85a572f53197dc249fc90b28b4091e374069adbad9d567e.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/dc88e711c4b7d42fe85a572f53197dc249fc90b28b4091e374069adbad9d567e.jpg)

![e4682802a8d11cd03f3a786cf2eb3af23af5f29760e1fd6dad1f94985b472f6e.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/e4682802a8d11cd03f3a786cf2eb3af23af5f29760e1fd6dad1f94985b472f6e.jpg)

![fc94246e52a13b0ed45f4c86e2977cba9fed42519fe7b326648b1d9716e68e35.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/images/fc94246e52a13b0ed45f4c86e2977cba9fed42519fe7b326648b1d9716e68e35.jpg)

### Tables

![0ec24bf3f519d7e12324d745cbd0bcdb2d1e023898aac26734c207c0af8c2da1.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/0ec24bf3f519d7e12324d745cbd0bcdb2d1e023898aac26734c207c0af8c2da1.jpg)

![16a3877fa70a792aa970373b1681c5669c182e4baa48c6b45a8a3c4f185bd8aa.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/16a3877fa70a792aa970373b1681c5669c182e4baa48c6b45a8a3c4f185bd8aa.jpg)

![2c0c5b767413653666bc49cd131255b39f76b6049d2050ca06fbdb547dbd1ada.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/2c0c5b767413653666bc49cd131255b39f76b6049d2050ca06fbdb547dbd1ada.jpg)

![37ef4af8267987ce20a6408d5063bcaf999dcab84f0f6282a629b708b6620135.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/37ef4af8267987ce20a6408d5063bcaf999dcab84f0f6282a629b708b6620135.jpg)

![84681f95ce395ac319b6908a5f3a0f688016d47b11f11ec0ec26d7382b9d9f87.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/84681f95ce395ac319b6908a5f3a0f688016d47b11f11ec0ec26d7382b9d9f87.jpg)

![869391183181cd694ef6c641b8d3e020ae0434dc61920a32651ee1820987c819.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/869391183181cd694ef6c641b8d3e020ae0434dc61920a32651ee1820987c819.jpg)

![8dfe451959f6a12de8846f3ee0e3444bdac112a523ffa340d5e42ab6193d3bac.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/8dfe451959f6a12de8846f3ee0e3444bdac112a523ffa340d5e42ab6193d3bac.jpg)

![98ae2acb254225ed57ceb1ff7f8bbc10eee838c631318eab371d6228a2c32894.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/98ae2acb254225ed57ceb1ff7f8bbc10eee838c631318eab371d6228a2c32894.jpg)

![a66ab4c66c5d21f66538c2e4e101d08cd007aaa43622116b996b57fa98ef5930.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/a66ab4c66c5d21f66538c2e4e101d08cd007aaa43622116b996b57fa98ef5930.jpg)

![ab353b760bb57d8ab991dc6c3579fa515e4667ba17ad4d8249ac5bbc0acb1af8.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/ab353b760bb57d8ab991dc6c3579fa515e4667ba17ad4d8249ac5bbc0acb1af8.jpg)

![ae896a9338633ba39db3e8cc9953dd8bd256cd0af3975488e799f4990a261beb.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/ae896a9338633ba39db3e8cc9953dd8bd256cd0af3975488e799f4990a261beb.jpg)

![b4f75088f798b1782e2fee6bfd380710bffef0cbd569842244122f8588516d62.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/b4f75088f798b1782e2fee6bfd380710bffef0cbd569842244122f8588516d62.jpg)

![d4ef101002d5836eba7e4766285e3aefd75b6cf546960e717a32516b8d6729c7.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/d4ef101002d5836eba7e4766285e3aefd75b6cf546960e717a32516b8d6729c7.jpg)

![e5fb630fff7b41c04bedd4c9361e014cb1e8ac2423671d9c51cf1da3c3d4c486.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/e5fb630fff7b41c04bedd4c9361e014cb1e8ac2423671d9c51cf1da3c3d4c486.jpg)

![e7c7e5dfca58f733003feaa2b2b8f74a1c90d0fe760edca9b9050b1aa9db8968.jpg](../icml_results/192_Procedural%20Pretraining_%20Warming%20Up%20Language%20Models%20with%20Abstract%20Data/tables/e7c7e5dfca58f733003feaa2b2b8f74a1c90d0fe760edca9b9050b1aa9db8968.jpg)

<a id="paper-0191"></a>

## VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs


### Images

![59644c248eaeb4b6e513afccb4ef409cff0aa114a493d2c82242dfb637bc8bb1.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/59644c248eaeb4b6e513afccb4ef409cff0aa114a493d2c82242dfb637bc8bb1.jpg)

![5add3e1643563146559b0806376bee64af61f8164b098370e3734410e082baf3.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/5add3e1643563146559b0806376bee64af61f8164b098370e3734410e082baf3.jpg)

![71e27a0e288e71bc98c81cc4a2c60ecee90ece1049a9eaca5ff2ce2df3d22299.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/71e27a0e288e71bc98c81cc4a2c60ecee90ece1049a9eaca5ff2ce2df3d22299.jpg)

![726dcf1af0fc6dc32ec5e350f7c56e7313117de2dfdc24c926db2f260b435bcc.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/726dcf1af0fc6dc32ec5e350f7c56e7313117de2dfdc24c926db2f260b435bcc.jpg)

![7a7aa36da0f58d2f9261a2e16b1aa98e02727a313380f482460fa2cf0f98f786.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/7a7aa36da0f58d2f9261a2e16b1aa98e02727a313380f482460fa2cf0f98f786.jpg)

![8ce2775ae7cf72598e579cb12531018d92ed7feef543d918ba0bcff6de401898.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/8ce2775ae7cf72598e579cb12531018d92ed7feef543d918ba0bcff6de401898.jpg)

![9519bde01a09b65701953143636f07f9b07ce14388f85789e57ea1f1fb9cb04f.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/9519bde01a09b65701953143636f07f9b07ce14388f85789e57ea1f1fb9cb04f.jpg)

![c91a11d6e738e1212db32cfe499c5ffd51425809e03877111db961db758fffdc.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/c91a11d6e738e1212db32cfe499c5ffd51425809e03877111db961db758fffdc.jpg)

![c97cd1b7dd68b2df09ac1b6ac5d52e6c49e673b0e3ee540f43d12b03c259cf2d.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/c97cd1b7dd68b2df09ac1b6ac5d52e6c49e673b0e3ee540f43d12b03c259cf2d.jpg)

![d0f9efc2d612d1eedb646a08edac9a893e4220a7352ebe0bfb5edcd84ac76a96.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/d0f9efc2d612d1eedb646a08edac9a893e4220a7352ebe0bfb5edcd84ac76a96.jpg)

![df6fd5745adf9b3de482e31edd76b7500d4807fe47e7c606057bf6abea4149f9.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/df6fd5745adf9b3de482e31edd76b7500d4807fe47e7c606057bf6abea4149f9.jpg)

![e39cd0ca1ca80bdfb02ea7e98385521be584493d1d6474c54e8bafa8803843a7.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/e39cd0ca1ca80bdfb02ea7e98385521be584493d1d6474c54e8bafa8803843a7.jpg)

![ebd1ea852f395c34b776f54163967ff7b73e2aa2c6d1e606904c3ac9e8766a4c.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/ebd1ea852f395c34b776f54163967ff7b73e2aa2c6d1e606904c3ac9e8766a4c.jpg)

![ec5ebc046a5d03eca6dca8df843a9a6cd18feba4215f486fb7bfd735ac087813.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/ec5ebc046a5d03eca6dca8df843a9a6cd18feba4215f486fb7bfd735ac087813.jpg)

![fcbc038d7f808e7c50daf78e510969f8bf54ff1fbb519314ec8470fcf2ee8b91.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/images/fcbc038d7f808e7c50daf78e510969f8bf54ff1fbb519314ec8470fcf2ee8b91.jpg)

### Tables

![0f3e473bf0852323ccd49fcd1976d4a24799dc0b64fa8e4e894c6bafc7e38830.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/0f3e473bf0852323ccd49fcd1976d4a24799dc0b64fa8e4e894c6bafc7e38830.jpg)

![1b9fd9a4d4a417c5fca2ed0b266e64d1ddff7a3e110678e6883e989e238d8fca.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/1b9fd9a4d4a417c5fca2ed0b266e64d1ddff7a3e110678e6883e989e238d8fca.jpg)

![226b49a30e3aefaface8712a1c4132390d90bc2759504c0d11bae099b1b09df4.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/226b49a30e3aefaface8712a1c4132390d90bc2759504c0d11bae099b1b09df4.jpg)

![252223ce1d16cffcafc3d2c68dc4695731f1248406d49cc4f74e24d62456ccc7.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/252223ce1d16cffcafc3d2c68dc4695731f1248406d49cc4f74e24d62456ccc7.jpg)

![3c8b6251e74ee9e0ae5ce77b928028582d3cb84be22b6eb830d128605f4c5e4b.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/3c8b6251e74ee9e0ae5ce77b928028582d3cb84be22b6eb830d128605f4c5e4b.jpg)

![42651358fe8cabd7f3ee4e17dd1eeb51740d0b39cffcd00e27357ce2e47fa92b.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/42651358fe8cabd7f3ee4e17dd1eeb51740d0b39cffcd00e27357ce2e47fa92b.jpg)

![618e050ffd7ae190439d9f24e97c13d2728d6d98111a918a28557683a2d75fc5.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/618e050ffd7ae190439d9f24e97c13d2728d6d98111a918a28557683a2d75fc5.jpg)

![626ca765804778c850ec88db9a5a3ea054099e48fa4f51bbd8c2d84ef4e69843.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/626ca765804778c850ec88db9a5a3ea054099e48fa4f51bbd8c2d84ef4e69843.jpg)

![644ae413b7058d7698cd980535fa67bc65fb8ef6c4186e3e4463a561247b9f2e.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/644ae413b7058d7698cd980535fa67bc65fb8ef6c4186e3e4463a561247b9f2e.jpg)

![69020d3ae088b9c6af39e76872eaf2a879bf55520fdc09043dff698c4cb2fa1f.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/69020d3ae088b9c6af39e76872eaf2a879bf55520fdc09043dff698c4cb2fa1f.jpg)

![7666a842a3bfc2df369e7980d1184c2658b3e74687c62e62f6f24413a0f3baf5.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/7666a842a3bfc2df369e7980d1184c2658b3e74687c62e62f6f24413a0f3baf5.jpg)

![8e9c906aedd54a27c467e8764d5d1c5e15a18611adf26cd658eb9a5accbeda9f.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/8e9c906aedd54a27c467e8764d5d1c5e15a18611adf26cd658eb9a5accbeda9f.jpg)

![92f3f0e960919a3af3e953aebb72dfec108c07b9160b197bc6e82a747b7802fe.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/92f3f0e960919a3af3e953aebb72dfec108c07b9160b197bc6e82a747b7802fe.jpg)

![bc01a1a2cf70c63abed59cfdb2a3ccdaef89ad82bfa501b916893a85d7ee000c.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/bc01a1a2cf70c63abed59cfdb2a3ccdaef89ad82bfa501b916893a85d7ee000c.jpg)

![bf5ed3f29b876042f8d3c26e734bb6c9d9122b03d8ed54be9974dbd6e1fa9db5.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/bf5ed3f29b876042f8d3c26e734bb6c9d9122b03d8ed54be9974dbd6e1fa9db5.jpg)

![c9174fd67688d4ff27f3341a13ba90e231634de0d696189c086bf188f3b23600.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/c9174fd67688d4ff27f3341a13ba90e231634de0d696189c086bf188f3b23600.jpg)

![ce44c22c85cdc0d03cfd3e2acc3d60e88b795fc6dea9af66e9cdeb63608fddd6.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/ce44c22c85cdc0d03cfd3e2acc3d60e88b795fc6dea9af66e9cdeb63608fddd6.jpg)

![da5e769c0329680dbf49ec53d4ee6e355bebb3e20acb14bf9554e9eec0e9d6ac.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/da5e769c0329680dbf49ec53d4ee6e355bebb3e20acb14bf9554e9eec0e9d6ac.jpg)

![e0e3c9da43f1972dd5f86147e02bf2287e36b4d910846035eef7591dcd5d2bd0.jpg](../icml_results/193_VectorWorld_%20Efficient%20Streaming%20World%20Model%20via%20Diffusion%20Flow%20on%20Vector%20Graphs/tables/e0e3c9da43f1972dd5f86147e02bf2287e36b4d910846035eef7591dcd5d2bd0.jpg)

<a id="paper-0192"></a>

## Nash Equilibria in Games with Playerwise Concave Coupling Constraints: Existence and Computation


### Images

![217ad64eb212c031e323ceb14e223d6cd49f28985df4a318e1a23b97041d4d09.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/217ad64eb212c031e323ceb14e223d6cd49f28985df4a318e1a23b97041d4d09.jpg)

![78076dc1ee52cc655ae8f69d2425fdf3fee049cf26c80d1be09ef9dbaa6b824c.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/78076dc1ee52cc655ae8f69d2425fdf3fee049cf26c80d1be09ef9dbaa6b824c.jpg)

![a0ba5efe3f48ac48aa81e5a2592f50a172aebbb6ac5d4691894e7e6b36bf0e56.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/a0ba5efe3f48ac48aa81e5a2592f50a172aebbb6ac5d4691894e7e6b36bf0e56.jpg)

![c1dd7ae5a90d824b095fcc70ea5ec761f472806f560522d508168d095b4de626.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/c1dd7ae5a90d824b095fcc70ea5ec761f472806f560522d508168d095b4de626.jpg)

![d422f47eb0f721d04854569ec7c0a1f5686e2237050e02ea8ebf8ecd0b8943de.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/d422f47eb0f721d04854569ec7c0a1f5686e2237050e02ea8ebf8ecd0b8943de.jpg)

![f8fccd33e9e1ee528cb59ce86363627f17f0507d98fd7aec938d9a345ac89075.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/images/f8fccd33e9e1ee528cb59ce86363627f17f0507d98fd7aec938d9a345ac89075.jpg)

### Tables

![f43bb4e21070a93618310561c89807ed40e952bc4cd831ec068a8b38a59f68bc.jpg](../icml_results/194_Nash%20Equilibria%20in%20Games%20with%20Playerwise%20Concave%20Coupling%20Constraints_%20Existence%20and%20Computation/tables/f43bb4e21070a93618310561c89807ed40e952bc4cd831ec068a8b38a59f68bc.jpg)

<a id="paper-0193"></a>

## The Signal is in the Steps: Local Scoring for Reasoning Data Selection


### Images

![14fd53ef1e1fb92731291f87045b2c9b3b203c8782ef7094bffb147c6d0553ec.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/14fd53ef1e1fb92731291f87045b2c9b3b203c8782ef7094bffb147c6d0553ec.jpg)

![29cf14e50e782db56a80b25f06af184a28852cc0f42e51cac5b2907146ea0563.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/29cf14e50e782db56a80b25f06af184a28852cc0f42e51cac5b2907146ea0563.jpg)

![4261782639046f79054a1486db545f7731d128b6ee33c5d6b53faeca2499f5b3.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/4261782639046f79054a1486db545f7731d128b6ee33c5d6b53faeca2499f5b3.jpg)

![4a933fb264ae0b7cb672631c601c158c3b9fa192dd0765118ed9991d9beb4f06.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/4a933fb264ae0b7cb672631c601c158c3b9fa192dd0765118ed9991d9beb4f06.jpg)

![5d89f57f5b97e395b2a14b8b4372b0624163bf9b72d8b774c4e7c38dbeb7c7ce.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/5d89f57f5b97e395b2a14b8b4372b0624163bf9b72d8b774c4e7c38dbeb7c7ce.jpg)

![6e654cc54682c19b511578f2ac0ee941ceb588e3e9a9a35a8124de5ac1e751fe.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/6e654cc54682c19b511578f2ac0ee941ceb588e3e9a9a35a8124de5ac1e751fe.jpg)

![754f37128ac5254db881a94644dce9c014006fae65d455511b254ae3d26dca04.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/754f37128ac5254db881a94644dce9c014006fae65d455511b254ae3d26dca04.jpg)

![7b785ce175386b7eb23578b830bdfad103371cdae6fcfbf1e0788648ccfd4335.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/7b785ce175386b7eb23578b830bdfad103371cdae6fcfbf1e0788648ccfd4335.jpg)

![ada7478c056d9875b04430fdd47a6aef27ce9eec516a94f4037d2dd14ebfe77b.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/ada7478c056d9875b04430fdd47a6aef27ce9eec516a94f4037d2dd14ebfe77b.jpg)

![da80ba82b9fe2fa4c80980b87847a3f3d34f08972e427cb106de610dac2f65d6.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/images/da80ba82b9fe2fa4c80980b87847a3f3d34f08972e427cb106de610dac2f65d6.jpg)

### Tables

![045b58f781484fad2182fb47321f8580b60502091dd181121a7ea7ee012665a4.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/045b58f781484fad2182fb47321f8580b60502091dd181121a7ea7ee012665a4.jpg)

![050cd058720e4921b52ae492806916197552dc13f75332804ec4cad2726ac749.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/050cd058720e4921b52ae492806916197552dc13f75332804ec4cad2726ac749.jpg)

![054e7f7cddd932fb92cfde0c8e3992ff9b55fab168e1b0a34bea6a71f9245a3f.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/054e7f7cddd932fb92cfde0c8e3992ff9b55fab168e1b0a34bea6a71f9245a3f.jpg)

![0ac625e1eb7e8605f8bd19a62790ca024eba12281320c03927501b2f40d09358.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/0ac625e1eb7e8605f8bd19a62790ca024eba12281320c03927501b2f40d09358.jpg)

![1669010dd525ddd0ac2cfdfc42c9316aede00c6e0df7fcdeebc8cb641751548f.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/1669010dd525ddd0ac2cfdfc42c9316aede00c6e0df7fcdeebc8cb641751548f.jpg)

![1db7c57444ea3ad5fc98711927f505a36df52ce513f83ef52c6de207b55a636a.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/1db7c57444ea3ad5fc98711927f505a36df52ce513f83ef52c6de207b55a636a.jpg)

![366df57d54be939578ff018d31693092a9f167206439bae981e20c14028b997d.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/366df57d54be939578ff018d31693092a9f167206439bae981e20c14028b997d.jpg)

![54dd61acf51a6aeb3dc02fbfda8e01de7fb220ee837bda5db3a76d51de8dca42.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/54dd61acf51a6aeb3dc02fbfda8e01de7fb220ee837bda5db3a76d51de8dca42.jpg)

![55dd59413d382d12deb0e9e77cfd3dff198c98f5b6ce82f662d50438139cbb5a.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/55dd59413d382d12deb0e9e77cfd3dff198c98f5b6ce82f662d50438139cbb5a.jpg)

![607205ac112c6b5eeea5de72b828f4ac8d0289fe1f50acbfa01eccc3d0048df1.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/607205ac112c6b5eeea5de72b828f4ac8d0289fe1f50acbfa01eccc3d0048df1.jpg)

![74ff9a9aead354ae8688a117fc26836ed0e58bdb749ad038bb3d44e884857667.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/74ff9a9aead354ae8688a117fc26836ed0e58bdb749ad038bb3d44e884857667.jpg)

![79efbef4f9f13db14b32d782f84571d3eec75e0392fecf2a215c9e09601fb634.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/79efbef4f9f13db14b32d782f84571d3eec75e0392fecf2a215c9e09601fb634.jpg)

![8296f310b14c590db988795019b603f768ef632ff1738aeaf79a5042d10fb291.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/8296f310b14c590db988795019b603f768ef632ff1738aeaf79a5042d10fb291.jpg)

![86e9cf94ce61440c92955cf401f830727845314971907b0c592622ce611be49b.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/86e9cf94ce61440c92955cf401f830727845314971907b0c592622ce611be49b.jpg)

![91026f67599eb5a32bc5fe8dfe295bd6d94916d337250ee0730891b79c8b8cec.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/91026f67599eb5a32bc5fe8dfe295bd6d94916d337250ee0730891b79c8b8cec.jpg)

![9997967501d07f9cca337d9c1f1f40250ad4d51d7c12c66e35285b49f5b18487.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/9997967501d07f9cca337d9c1f1f40250ad4d51d7c12c66e35285b49f5b18487.jpg)

![b60cc251302f40942b38e122aa4daa5aab28557e8b727b2c5873ad3ebfcd8c9f.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/b60cc251302f40942b38e122aa4daa5aab28557e8b727b2c5873ad3ebfcd8c9f.jpg)

![bd9856658a52fb471d9a862d192a62e42713cafaa71b2a9641c62edac28f47fb.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/bd9856658a52fb471d9a862d192a62e42713cafaa71b2a9641c62edac28f47fb.jpg)

![cddfef507fcca27b2a8c299baa1f247d78aefbd6e07e2b61b9d3a2eaed032a3a.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/cddfef507fcca27b2a8c299baa1f247d78aefbd6e07e2b61b9d3a2eaed032a3a.jpg)

![d0e6dd1c395d1f9a177ebdcfe0530c85459173ae2c327823810e68cbb0108c7b.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/d0e6dd1c395d1f9a177ebdcfe0530c85459173ae2c327823810e68cbb0108c7b.jpg)

![d866ed80bb1ff8170274cee7fb9a7eca5299f58ae9ec0393219b8840cb9f650c.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/d866ed80bb1ff8170274cee7fb9a7eca5299f58ae9ec0393219b8840cb9f650c.jpg)

![e6e4023880e2dfa088f641650ea72d2d0cf9140a16d9f579a07a230108aa69d5.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/e6e4023880e2dfa088f641650ea72d2d0cf9140a16d9f579a07a230108aa69d5.jpg)

![e9985b74723c0b17293d165cf0b7e1d233aaf9ba7c2c3086a1eb99c2d4a2dcb7.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/e9985b74723c0b17293d165cf0b7e1d233aaf9ba7c2c3086a1eb99c2d4a2dcb7.jpg)

![ed1b3d69189c690cc6a5167912a3d753372fa97ea7387959ab9bf9eb28b719aa.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/ed1b3d69189c690cc6a5167912a3d753372fa97ea7387959ab9bf9eb28b719aa.jpg)

![f5f813082da508c5c488de07d52a5c2c8cefe2c84760240b32a13aade702aaff.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/f5f813082da508c5c488de07d52a5c2c8cefe2c84760240b32a13aade702aaff.jpg)

![fce6295800a484c0616e7fcffdcba16a14bb8ee72a396dafe35f4cd94e527aa3.jpg](../icml_results/196_The%20Signal%20is%20in%20the%20Steps_%20Local%20Scoring%20for%20Reasoning%20Data%20Selection/tables/fce6295800a484c0616e7fcffdcba16a14bb8ee72a396dafe35f4cd94e527aa3.jpg)

<a id="paper-0194"></a>

## Overcoming PINNs Failure Modes In High Dimension With Low-Rank Fourier Sum


### Images

![37a975e231d9ec2c4a0f554d43d6d8c1b2cdbf69ac7daa2a3f2241dbffa62c99.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/images/37a975e231d9ec2c4a0f554d43d6d8c1b2cdbf69ac7daa2a3f2241dbffa62c99.jpg)

![af14d0aef7a500c8ee47de922bc0d9c5e85cd4788fa1e34f163051300e7b2c3c.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/images/af14d0aef7a500c8ee47de922bc0d9c5e85cd4788fa1e34f163051300e7b2c3c.jpg)

### Tables

![3852ab5b57a8440eccf1ad3270da66de2a84d45301017fe760f33a395b54dbda.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/tables/3852ab5b57a8440eccf1ad3270da66de2a84d45301017fe760f33a395b54dbda.jpg)

![b38ab97f866868645733dd199818a7e98872c2fd17f2aa63c341926cfb7543da.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/tables/b38ab97f866868645733dd199818a7e98872c2fd17f2aa63c341926cfb7543da.jpg)

![d55b56cf1e75c2da9c35dfad49fd93fb53059f7848ac5fb19404850f82d40c8f.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/tables/d55b56cf1e75c2da9c35dfad49fd93fb53059f7848ac5fb19404850f82d40c8f.jpg)

![d5bf9a9b821b8286998a93c0ea8bb5962f98abffd54c75659278039d914e2909.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/tables/d5bf9a9b821b8286998a93c0ea8bb5962f98abffd54c75659278039d914e2909.jpg)

![fbdd24c11e47a6f2584b3a4ecf7d53f33c64d29cfed5e6b84c946d280173ae72.jpg](../icml_results/197_Overcoming%20PINNs%20Failure%20Modes%20In%20High%20Dimension%20With%20Low-Rank%20Fourier%20Sum/tables/fbdd24c11e47a6f2584b3a4ecf7d53f33c64d29cfed5e6b84c946d280173ae72.jpg)

<a id="paper-0195"></a>

## Real-Time Visual Attribution Streaming in Thinking Model


### Images

![009765ca011b27bff4bfac0735650b2e6e0f760186db9307d2bcf090c2c91037.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/009765ca011b27bff4bfac0735650b2e6e0f760186db9307d2bcf090c2c91037.jpg)

![03447c934a119463f7e2644f46257b4c5e7237b5e104ea2bb76f7f8e36f8e8ad.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/03447c934a119463f7e2644f46257b4c5e7237b5e104ea2bb76f7f8e36f8e8ad.jpg)

![0413b4136c40af1de5ad8a426212c7ff830b90e8d7bde108068e4ff08dce90bd.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/0413b4136c40af1de5ad8a426212c7ff830b90e8d7bde108068e4ff08dce90bd.jpg)

![08c3b67106077476edcd4b3beade23a4461fdd67e1ca3ece817d4355f5d149c1.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/08c3b67106077476edcd4b3beade23a4461fdd67e1ca3ece817d4355f5d149c1.jpg)

![15f44d966fe8f54dd0281eaf0c18d5b9954ffd4cb175f727c96d64033394f90b.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/15f44d966fe8f54dd0281eaf0c18d5b9954ffd4cb175f727c96d64033394f90b.jpg)

![20ad3b5b7173b230ac46304aee608687b6c883d7b22dd55f2c2490abbe0b62da.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/20ad3b5b7173b230ac46304aee608687b6c883d7b22dd55f2c2490abbe0b62da.jpg)

![281ef1f807b7a3b1a07a73724f7f1c9fcecd93fc84034f02c01ca76d1684744f.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/281ef1f807b7a3b1a07a73724f7f1c9fcecd93fc84034f02c01ca76d1684744f.jpg)

![29131619886a71e38d3c5b7ec8c7f4bee11a04b31ca345ad89f934cb82035970.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/29131619886a71e38d3c5b7ec8c7f4bee11a04b31ca345ad89f934cb82035970.jpg)

![2c2b8bd1bd8dbcaf33e50bc040dca39383b4d67955f71fb30fc458711536e313.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/2c2b8bd1bd8dbcaf33e50bc040dca39383b4d67955f71fb30fc458711536e313.jpg)

![2f6c52545267bafa904dcbbc193c60cbf664189470d8bc64b0bea6aefe3d46d5.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/2f6c52545267bafa904dcbbc193c60cbf664189470d8bc64b0bea6aefe3d46d5.jpg)

![325a4228549ab4556797c300c7fcb837e3680adce1c8146a8768a724d9e9bc42.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/325a4228549ab4556797c300c7fcb837e3680adce1c8146a8768a724d9e9bc42.jpg)

![3d2bf28425694c3bb75c15d135319b6919002917f366c486647a3d1a445f1b96.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/3d2bf28425694c3bb75c15d135319b6919002917f366c486647a3d1a445f1b96.jpg)

![42af6dea4ec998fe875a0b185869f506839cbab048bddfecee9976126cd124d8.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/42af6dea4ec998fe875a0b185869f506839cbab048bddfecee9976126cd124d8.jpg)

![4a29222eaf87ccd57e894c60f374976180a42b3af00f4f2a38c68b6300566013.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/4a29222eaf87ccd57e894c60f374976180a42b3af00f4f2a38c68b6300566013.jpg)

![4daa9550cf167c435a6ce00ae00e37cce0e37f5bbc8e920fdfe3f095cdeba8c3.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/4daa9550cf167c435a6ce00ae00e37cce0e37f5bbc8e920fdfe3f095cdeba8c3.jpg)

![502d581d393cfe97ad5af620976f34241948f5357ecc63dafcc4b96249b3d9b2.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/502d581d393cfe97ad5af620976f34241948f5357ecc63dafcc4b96249b3d9b2.jpg)

![54f793c6498a46c934b8e3348c503893ff0283508a5b7ede447aece95e5b3c78.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/54f793c6498a46c934b8e3348c503893ff0283508a5b7ede447aece95e5b3c78.jpg)

![56ed7eb10b029fa3bac81b239a9ccddcc8635c09350cf2b4670e8eb5e66c93fd.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/56ed7eb10b029fa3bac81b239a9ccddcc8635c09350cf2b4670e8eb5e66c93fd.jpg)

![590724489861865f6c74ae49bf30f967a0bede1f50bcd8de16beb0e615f2f667.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/590724489861865f6c74ae49bf30f967a0bede1f50bcd8de16beb0e615f2f667.jpg)

![61ab34caac3eea149bdea7e4e6426e0c38c6954cc2681cc03b16197394b631a8.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/61ab34caac3eea149bdea7e4e6426e0c38c6954cc2681cc03b16197394b631a8.jpg)

![6367dad3b0aa66c9b44f8e220bea0d439cc96d51af2dbbbe95c5c0f341087f0c.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/6367dad3b0aa66c9b44f8e220bea0d439cc96d51af2dbbbe95c5c0f341087f0c.jpg)

![6460cf47222896c0ccd78c9652bb40735cdb836ba9aaa2a28d8d4a75edb938a3.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/6460cf47222896c0ccd78c9652bb40735cdb836ba9aaa2a28d8d4a75edb938a3.jpg)

![729b7a392b6a92bd74adece0e9e436f480977f6453f088d908accbfb7f7715fc.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/729b7a392b6a92bd74adece0e9e436f480977f6453f088d908accbfb7f7715fc.jpg)

![737294be09a59f5a101cef6589ea5e0d085da6bf9f90c836de8c24de6fda8dd7.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/737294be09a59f5a101cef6589ea5e0d085da6bf9f90c836de8c24de6fda8dd7.jpg)

![7ca7e88eb84276a99d6459adec7856d461cbfe1362b12f94d9a43c5c0fffef18.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/7ca7e88eb84276a99d6459adec7856d461cbfe1362b12f94d9a43c5c0fffef18.jpg)

![802e52d3d1cc2077bd61864acb0c5d84abdcf3b429f264443611ed6d40eb150d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/802e52d3d1cc2077bd61864acb0c5d84abdcf3b429f264443611ed6d40eb150d.jpg)

![83425b8cede34e6509a9932c7544bfd48a6230d0426a92768252efefa218b1b4.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/83425b8cede34e6509a9932c7544bfd48a6230d0426a92768252efefa218b1b4.jpg)

![8632e99a7077f67e031cfabdd97b64cd418fc9ab5e5bc1b513e19ca6ba176f7a.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/8632e99a7077f67e031cfabdd97b64cd418fc9ab5e5bc1b513e19ca6ba176f7a.jpg)

![8697ccdf745993e0e99ce6c9fe2ef32ecb9193612dc38855b1113ecc24d8d29d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/8697ccdf745993e0e99ce6c9fe2ef32ecb9193612dc38855b1113ecc24d8d29d.jpg)

![90df01c4635366e1e556f754668a84ead37672e419d1d1c91f46b88eede6f6bf.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/90df01c4635366e1e556f754668a84ead37672e419d1d1c91f46b88eede6f6bf.jpg)

![9d5c6715e95ed02197aa9a02ec3c4e526f53568ed972ddf557dce589cab2cf50.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/9d5c6715e95ed02197aa9a02ec3c4e526f53568ed972ddf557dce589cab2cf50.jpg)

![9fec424f0be39b14c98c6597f80e62943c50cf0a3e05a12061a52912c1edb42d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/9fec424f0be39b14c98c6597f80e62943c50cf0a3e05a12061a52912c1edb42d.jpg)

![a45137671c9b0063b11ef82df09f4a08a791169dc2d3d64ba78b30609cbd4d76.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/a45137671c9b0063b11ef82df09f4a08a791169dc2d3d64ba78b30609cbd4d76.jpg)

![abd373470b76547dd178a22bb953293f49c37b07fe9fc10badcd086a85906a1d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/abd373470b76547dd178a22bb953293f49c37b07fe9fc10badcd086a85906a1d.jpg)

![afa921407089446b6ad5245ac3a025a8a8c95ca3267bf15b275388a61e6b0e64.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/afa921407089446b6ad5245ac3a025a8a8c95ca3267bf15b275388a61e6b0e64.jpg)

![c01f914115bb6483301ea92cbb4b9a4524449e5f47ddabc9dab27003f45b5383.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/c01f914115bb6483301ea92cbb4b9a4524449e5f47ddabc9dab27003f45b5383.jpg)

![caa6adb37e064027ccec9f12792b8a02310b894261009e784cb6ad2b9ac590d0.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/caa6adb37e064027ccec9f12792b8a02310b894261009e784cb6ad2b9ac590d0.jpg)

![cf240a66a27cc65c8e39d96a28193d0972e7dfed9f6b0a73d326e368cf512cbd.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/cf240a66a27cc65c8e39d96a28193d0972e7dfed9f6b0a73d326e368cf512cbd.jpg)

![d36256eb65f7c3f191f3a2599682ca6c466510898073bb063274525f96cd63b9.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/d36256eb65f7c3f191f3a2599682ca6c466510898073bb063274525f96cd63b9.jpg)

![d472174ef19ba68d62c3087e40cf63b2f6e0ce5fc5662085e59aefa5998bc5d7.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/d472174ef19ba68d62c3087e40cf63b2f6e0ce5fc5662085e59aefa5998bc5d7.jpg)

![d578fbc9f9d33c9cb4cf7ec18c27dcaa52f93bdd8a783839c63e1e58d9a0b0d9.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/d578fbc9f9d33c9cb4cf7ec18c27dcaa52f93bdd8a783839c63e1e58d9a0b0d9.jpg)

![d5bead59a17f10ae9e922f301683f887d21664d0b9798924f2780a9367c03f65.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/d5bead59a17f10ae9e922f301683f887d21664d0b9798924f2780a9367c03f65.jpg)

![d6c68c76031b693b5bcb7e5e344a6cc5681e7fc187d3d811ffb4d676cceba008.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/d6c68c76031b693b5bcb7e5e344a6cc5681e7fc187d3d811ffb4d676cceba008.jpg)

![da8c785a2a72a48352bdb5d0714d34762db28781ce81ffbad8897b51bc9a4126.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/da8c785a2a72a48352bdb5d0714d34762db28781ce81ffbad8897b51bc9a4126.jpg)

![dfe9256668f5cdae2f94df7fb1ba4252adb8b2c43106832e52707443e7eed1bb.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/dfe9256668f5cdae2f94df7fb1ba4252adb8b2c43106832e52707443e7eed1bb.jpg)

![e199161d82a9fed7da9f3b08b9f2aedf19651ba7f8cdd20012947bff974aebe6.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/e199161d82a9fed7da9f3b08b9f2aedf19651ba7f8cdd20012947bff974aebe6.jpg)

![ec5acad10d7a19242bdf1c0b5e25130df7bc77fa809eb2b78036feaafe897394.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/ec5acad10d7a19242bdf1c0b5e25130df7bc77fa809eb2b78036feaafe897394.jpg)

![f4102b4b0637bd795cb4c50d70f4bacdb182aee2a14c69a533db3467e35f3a4d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/f4102b4b0637bd795cb4c50d70f4bacdb182aee2a14c69a533db3467e35f3a4d.jpg)

![f7ff7aa6967bf466515bbe4b76b9ae76eecbce9ba5b49f4a6849482b8d455893.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/images/f7ff7aa6967bf466515bbe4b76b9ae76eecbce9ba5b49f4a6849482b8d455893.jpg)

### Tables

![0c5e43b0d9d2b7e887faa77bca08adfee1d1179be70037f98b039c8041e75337.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/0c5e43b0d9d2b7e887faa77bca08adfee1d1179be70037f98b039c8041e75337.jpg)

![13474294d195a0b6a75305004e659f9d0fe970db5462e13815324f5414b5610d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/13474294d195a0b6a75305004e659f9d0fe970db5462e13815324f5414b5610d.jpg)

![1506fc20c3aca1bfc7df36afca2adaff1b997046d149aae85c09d937727f6c03.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/1506fc20c3aca1bfc7df36afca2adaff1b997046d149aae85c09d937727f6c03.jpg)

![1d8bf2a613eb8eef0cd566deaef7d7d841da4347fd9dde44f4b46b7f636aedc2.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/1d8bf2a613eb8eef0cd566deaef7d7d841da4347fd9dde44f4b46b7f636aedc2.jpg)

![2361910b795212b759bc2c141bbfdfc22cc54412316ba5e173c27e6dc2b0d1eb.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/2361910b795212b759bc2c141bbfdfc22cc54412316ba5e173c27e6dc2b0d1eb.jpg)

![264b4c58671f703aae2743e6bac8ed2d62f56aad8a05ac10a4c8636d8342ccf6.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/264b4c58671f703aae2743e6bac8ed2d62f56aad8a05ac10a4c8636d8342ccf6.jpg)

![2ce27c8dafd4834d0a12254b91938ceed9afc1af0bda5c19b0331895c2895101.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/2ce27c8dafd4834d0a12254b91938ceed9afc1af0bda5c19b0331895c2895101.jpg)

![3131d2bf8dcf55faea3e95f1633b9a5b69808200a12d32361c335c3c11c01148.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/3131d2bf8dcf55faea3e95f1633b9a5b69808200a12d32361c335c3c11c01148.jpg)

![3c78fceefa11be184593bfb0e5d6104f4991edc845bbe451089ea016548d1162.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/3c78fceefa11be184593bfb0e5d6104f4991edc845bbe451089ea016548d1162.jpg)

![4965fe92bddea3c883f748585d9bc6a5c28181d09f88a07b05c63e24910e5907.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/4965fe92bddea3c883f748585d9bc6a5c28181d09f88a07b05c63e24910e5907.jpg)

![4d1a00c2a2f195cbbc891c0619d6aa1ac313f0d5f5dd513796e1cdd13dbfad6a.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/4d1a00c2a2f195cbbc891c0619d6aa1ac313f0d5f5dd513796e1cdd13dbfad6a.jpg)

![522f5a5f0b95acf71a31f6acad85971c8627bf1a6b52639546cd4de96f845d6b.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/522f5a5f0b95acf71a31f6acad85971c8627bf1a6b52639546cd4de96f845d6b.jpg)

![537a3996f5f49cae1b00247037db29c130bdcfaf2e92970cf964103f799a78b3.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/537a3996f5f49cae1b00247037db29c130bdcfaf2e92970cf964103f799a78b3.jpg)

![6316e270c8698fa5bca44b62b008b6ccad341b18eb798651bce340c47d6cb718.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/6316e270c8698fa5bca44b62b008b6ccad341b18eb798651bce340c47d6cb718.jpg)

![64fba140e86db783f4e11bb9bc6170a31684eefdb05e6e423a4e81d1ebb6185f.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/64fba140e86db783f4e11bb9bc6170a31684eefdb05e6e423a4e81d1ebb6185f.jpg)

![6fcfe83b43bcbd9c9b71ee04fd9a8b8e1773cff3477b1113f1c51adc6896df2e.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/6fcfe83b43bcbd9c9b71ee04fd9a8b8e1773cff3477b1113f1c51adc6896df2e.jpg)

![7b2184a52f77eb31bc016dcc79d47bbde7d1eeb71e487b7b374d2348d0f398c3.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/7b2184a52f77eb31bc016dcc79d47bbde7d1eeb71e487b7b374d2348d0f398c3.jpg)

![7e2077fc14131dde5f9d94a93154c967e8bef0b7788f1eda6bb418d65197001d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/7e2077fc14131dde5f9d94a93154c967e8bef0b7788f1eda6bb418d65197001d.jpg)

![9d7e525cc4b95800efc0bbc3f1a1fbac7e0707d50e2d3324543b566cb491d76f.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/9d7e525cc4b95800efc0bbc3f1a1fbac7e0707d50e2d3324543b566cb491d76f.jpg)

![a24e03275a247b8319af2e1141912aa65fa327b48c2417e969c33eeb6694a2e5.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/a24e03275a247b8319af2e1141912aa65fa327b48c2417e969c33eeb6694a2e5.jpg)

![a9d33ced5a177346614fa5ee178dbda220b394e27bb09fcb6b0390636d689bab.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/a9d33ced5a177346614fa5ee178dbda220b394e27bb09fcb6b0390636d689bab.jpg)

![ac8f3f210ee9e3c72cd1d655d069da52d98ab894f4bdd82b58c08eb299b60925.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/ac8f3f210ee9e3c72cd1d655d069da52d98ab894f4bdd82b58c08eb299b60925.jpg)

![b06747232dbcbb07c431e979405361bf00b7ed5a97662a81ac3337a00f06792a.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/b06747232dbcbb07c431e979405361bf00b7ed5a97662a81ac3337a00f06792a.jpg)

![b108549705c2a79d374f4defee5446429e46067fe05f0e24942188c4e22153ac.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/b108549705c2a79d374f4defee5446429e46067fe05f0e24942188c4e22153ac.jpg)

![b34919850988d855062028be22ebc35233b9125bbc0db4941ee1e715ba4371db.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/b34919850988d855062028be22ebc35233b9125bbc0db4941ee1e715ba4371db.jpg)

![b3fff12b20f23b91d82c00d2d081f53c64510b12870812831ede584581eacfd0.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/b3fff12b20f23b91d82c00d2d081f53c64510b12870812831ede584581eacfd0.jpg)

![c12bc025ccc51b944a6551f1a17ad73828910d5627d1ba7b7c6bdfabe57d54bd.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/c12bc025ccc51b944a6551f1a17ad73828910d5627d1ba7b7c6bdfabe57d54bd.jpg)

![c16b3e6e7903f7b422041b31d146bcb94faed5de355f2ed442d3881d66e4d11e.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/c16b3e6e7903f7b422041b31d146bcb94faed5de355f2ed442d3881d66e4d11e.jpg)

![c695c0061b8bca2d7048299d7864eac740d4765aabeb9ddd8b1388a989e40ef1.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/c695c0061b8bca2d7048299d7864eac740d4765aabeb9ddd8b1388a989e40ef1.jpg)

![c848c526c46524c257c6060adf5fdeb5fef4f9c3a3231b115377ab748d1ae2b6.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/c848c526c46524c257c6060adf5fdeb5fef4f9c3a3231b115377ab748d1ae2b6.jpg)

![c88a8e231c6370fd06246d24549c8d6c6981c9db8b5bd1036e23f9f2bf35bd08.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/c88a8e231c6370fd06246d24549c8d6c6981c9db8b5bd1036e23f9f2bf35bd08.jpg)

![d30d8984e47b3f14b12464483a7c3883b7fa24cf1f3af49c2a47def070007c8e.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/d30d8984e47b3f14b12464483a7c3883b7fa24cf1f3af49c2a47def070007c8e.jpg)

![d610a7baab103e28132263c197627048a6955f600303c5211d605adbe5136616.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/d610a7baab103e28132263c197627048a6955f600303c5211d605adbe5136616.jpg)

![d9699497ef5ab6ac243a7b91105ca5d7243e4a17c202cce3fc252f6c1aad8c50.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/d9699497ef5ab6ac243a7b91105ca5d7243e4a17c202cce3fc252f6c1aad8c50.jpg)

![dbc70ef686ce5b30f391f9d66421814da76f504f69f2abcfd1735667a7d1476a.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/dbc70ef686ce5b30f391f9d66421814da76f504f69f2abcfd1735667a7d1476a.jpg)

![de0b82bd13f0a9669fc7245f0778db3e3f3515deb4b890e9e6f95bef37950f47.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/de0b82bd13f0a9669fc7245f0778db3e3f3515deb4b890e9e6f95bef37950f47.jpg)

![df59cdc8743f4a9cea556e87b2d9fa71c3466909850ed37896094c2573dcf194.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/df59cdc8743f4a9cea556e87b2d9fa71c3466909850ed37896094c2573dcf194.jpg)

![e13a8aadd3225abfa81cf4d7c4fb81d654dcfc3dbeafd9ccd94be9545374205d.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/e13a8aadd3225abfa81cf4d7c4fb81d654dcfc3dbeafd9ccd94be9545374205d.jpg)

![ea58a3c21f99831f2457a9a18ace2714dfe82adc57b3368cef65aec529c85d9c.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/ea58a3c21f99831f2457a9a18ace2714dfe82adc57b3368cef65aec529c85d9c.jpg)

![ea9259dbae6d74f7e823e0342c1a61b599b8e221803491beca869bf43163e96e.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/ea9259dbae6d74f7e823e0342c1a61b599b8e221803491beca869bf43163e96e.jpg)

![f8b4f6ab648b0d47ef42174cbf235ed8a5eff19867a6f9af24258af2be1ee6b8.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/f8b4f6ab648b0d47ef42174cbf235ed8a5eff19867a6f9af24258af2be1ee6b8.jpg)

![fff67f7c58eae3662d513acd8835e8bdbe3a20b1d1dba322a8a712cc19c5afe5.jpg](../icml_results/198_Real-Time%20Visual%20Attribution%20Streaming%20in%20Thinking%20Model/tables/fff67f7c58eae3662d513acd8835e8bdbe3a20b1d1dba322a8a712cc19c5afe5.jpg)

<a id="paper-0196"></a>

## ThunderAgent: A Fast, Simple, and Program-Aware Agentic Inference System


### Images

![0b1991d08f1373d60b79df33bd01623e3986c5dceb46e638f7d335745778cace.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/0b1991d08f1373d60b79df33bd01623e3986c5dceb46e638f7d335745778cace.jpg)

![10fe09e3c2c8cf895e748c8061e7c9971b0b3fb8475328545286e71604376741.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/10fe09e3c2c8cf895e748c8061e7c9971b0b3fb8475328545286e71604376741.jpg)

![24f8576ff3fba526f7240eb82904a471a7b7d0732513d7850cf4bcdeb5cd1075.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/24f8576ff3fba526f7240eb82904a471a7b7d0732513d7850cf4bcdeb5cd1075.jpg)

![37396831c170c2da230f7c948468fa7f37197228e7dc794169847a97d843f830.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/37396831c170c2da230f7c948468fa7f37197228e7dc794169847a97d843f830.jpg)

![379c43ff7fd0d38141ddb0a697e44fc70e91cc1b0b0b8b7dd6a4f3a5275568be.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/379c43ff7fd0d38141ddb0a697e44fc70e91cc1b0b0b8b7dd6a4f3a5275568be.jpg)

![400387b36b16aaee5da607ad402362eaef7ddf8909bd0c172bbcd92d948ddd5c.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/400387b36b16aaee5da607ad402362eaef7ddf8909bd0c172bbcd92d948ddd5c.jpg)

![4d29d56cc55e934e7586f31bfdd5ec07945d7bdffb5976d4856286f8750b8232.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/4d29d56cc55e934e7586f31bfdd5ec07945d7bdffb5976d4856286f8750b8232.jpg)

![61ef57efc4bb2fec86561000d454b9ec88d49a1dd8264974d589f62f54e80a53.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/61ef57efc4bb2fec86561000d454b9ec88d49a1dd8264974d589f62f54e80a53.jpg)

![6a76d261f66dc377cf362c32986e427d22e06f3f537b800cd3bd94ca37d275f8.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/6a76d261f66dc377cf362c32986e427d22e06f3f537b800cd3bd94ca37d275f8.jpg)

![894f73fe0d30768a7b90d264b82cb34344b6a5fcb5784830873e2f9b3c770a53.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/894f73fe0d30768a7b90d264b82cb34344b6a5fcb5784830873e2f9b3c770a53.jpg)

![a4f342a4c9d229e4d82505856faa7b116d15baa3efcd52c9090ffe6cd92ed8c7.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/a4f342a4c9d229e4d82505856faa7b116d15baa3efcd52c9090ffe6cd92ed8c7.jpg)

![c1b88c9399992b79f950c3356d669c9420ccb49476bc0285f3ded4424151600c.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/c1b88c9399992b79f950c3356d669c9420ccb49476bc0285f3ded4424151600c.jpg)

![c4290b476d3e8018c46741e821a10d9e1c2451557120033e90568dd079a84b1e.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/c4290b476d3e8018c46741e821a10d9e1c2451557120033e90568dd079a84b1e.jpg)

![d2f4b0183ab9f3a143ab9f3329c64ed4a7852bdcbb2a56b99ed734580d70dfd0.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/d2f4b0183ab9f3a143ab9f3329c64ed4a7852bdcbb2a56b99ed734580d70dfd0.jpg)

![deccd80ad0cb228e6e869d4b2facfa898e5c37d4a2ccce99b716a370d2616db6.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/images/deccd80ad0cb228e6e869d4b2facfa898e5c37d4a2ccce99b716a370d2616db6.jpg)

### Tables

![3e4c5244a836f4a54489a189bd03a945bfe67cd11b65145a412e2666a035e022.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/3e4c5244a836f4a54489a189bd03a945bfe67cd11b65145a412e2666a035e022.jpg)

![910e83174e32008445445c9b64c8b1f2af8ed32ab1fbb523e9240ee3d6dbbf88.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/910e83174e32008445445c9b64c8b1f2af8ed32ab1fbb523e9240ee3d6dbbf88.jpg)

![aa9b21d7656442e185abc43f68ad207d407e54a7c2397079b7e28db217301d60.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/aa9b21d7656442e185abc43f68ad207d407e54a7c2397079b7e28db217301d60.jpg)

![b045bc1b56779352e3a7c7d3e012a7e672cb0f91336ccb74d641df14e45f2ab8.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/b045bc1b56779352e3a7c7d3e012a7e672cb0f91336ccb74d641df14e45f2ab8.jpg)

![b0e0bce6c0e5f388a7432dc2eac85b5b2af20eb671d09da106761041e07fef02.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/b0e0bce6c0e5f388a7432dc2eac85b5b2af20eb671d09da106761041e07fef02.jpg)

![bcadbdb4e54851963fd6fc0b1d431e90a6cad765bb8d41b0063dd0b956b6fd20.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/bcadbdb4e54851963fd6fc0b1d431e90a6cad765bb8d41b0063dd0b956b6fd20.jpg)

![ce6bae0b08757f94b888f5cc67fce40ffbd09eccb100ba945badf1717ceff7e2.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/ce6bae0b08757f94b888f5cc67fce40ffbd09eccb100ba945badf1717ceff7e2.jpg)

![dfd894b680b8b586342d55fe919b782a7df31e213dfa8e3ca930704129cb1178.jpg](../icml_results/199_ThunderAgent_%20A%20Fast%2C%20Simple%2C%20and%20Program-Aware%20Agentic%20Inference%20System/tables/dfd894b680b8b586342d55fe919b782a7df31e213dfa8e3ca930704129cb1178.jpg)

<a id="paper-0197"></a>

## Symmetry in language statistics shapes the geometry of model representations


### Images

![07f73931b3544ef480f20b6fe66a01c2161b759a83b8a05f3d572ded8a7c48d0.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/07f73931b3544ef480f20b6fe66a01c2161b759a83b8a05f3d572ded8a7c48d0.jpg)

![0eb21c8b2461906926916558c63219c8ca52f9fb9a0f129858a14a45dab32c4a.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/0eb21c8b2461906926916558c63219c8ca52f9fb9a0f129858a14a45dab32c4a.jpg)

![11ff5fe91fd2f7136267b8bb2308a48ab8b5082e1e4a7841c292c3bdc78b9f6a.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/11ff5fe91fd2f7136267b8bb2308a48ab8b5082e1e4a7841c292c3bdc78b9f6a.jpg)

![3752b24b0184a903c850c72d323cb6223d8a55aef77a2acf31f175d2a3877e01.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/3752b24b0184a903c850c72d323cb6223d8a55aef77a2acf31f175d2a3877e01.jpg)

![694c6812260c989846fe631cce55b093cd3fa3d2142769a643063e3e2f1c3c2e.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/694c6812260c989846fe631cce55b093cd3fa3d2142769a643063e3e2f1c3c2e.jpg)

![6de41f88d9f9abe3aa193ff72f6a917b1f97bafab8d95c249c0baa1335c85ce7.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/6de41f88d9f9abe3aa193ff72f6a917b1f97bafab8d95c249c0baa1335c85ce7.jpg)

![75347b44aa9c52725d07ebd84260d17e7b4546fc5cae4a956a1454b69f66e244.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/75347b44aa9c52725d07ebd84260d17e7b4546fc5cae4a956a1454b69f66e244.jpg)

![76849d4056cd0eda75c09a617f256d3b4cfd2659c569d6cadec83ad1aa4bf83e.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/76849d4056cd0eda75c09a617f256d3b4cfd2659c569d6cadec83ad1aa4bf83e.jpg)

![9b8832a23b3c70dd5f0facdf19e6418927cf2ebba6837379c334403592ec10d5.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/9b8832a23b3c70dd5f0facdf19e6418927cf2ebba6837379c334403592ec10d5.jpg)

![9e85c35abb1b17bc423d1b1a014bd628ee1ce1891be3526ea7149598a8bc09d8.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/9e85c35abb1b17bc423d1b1a014bd628ee1ce1891be3526ea7149598a8bc09d8.jpg)

![a4a807285df4f216f594c18ad8002defbf29b3ac2bcba2810697bb77ee56d63f.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/a4a807285df4f216f594c18ad8002defbf29b3ac2bcba2810697bb77ee56d63f.jpg)

![a8936ef34c93dd53e2c51b372cdb21bde864156ef9923708fb37f2894ab25f71.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/a8936ef34c93dd53e2c51b372cdb21bde864156ef9923708fb37f2894ab25f71.jpg)

![a8ec494b181440c8d47ef4384a7c2d24a458f11f18899b5cb197ccd04d0ba13f.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/a8ec494b181440c8d47ef4384a7c2d24a458f11f18899b5cb197ccd04d0ba13f.jpg)

![a8ef94e8ea43d59a28f365ab89b7d0bd2a2704357b3b651934975fc6da72a45e.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/a8ef94e8ea43d59a28f365ab89b7d0bd2a2704357b3b651934975fc6da72a45e.jpg)

![b8bef2c1899694dc422d7fa8753d4312ef8033a6de1328935a28d17ff80fcce2.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/b8bef2c1899694dc422d7fa8753d4312ef8033a6de1328935a28d17ff80fcce2.jpg)

![bc55adfab5f8698ee0cdbdd1fc0aabdcc7c67accc189307cf57a646f94953a42.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/bc55adfab5f8698ee0cdbdd1fc0aabdcc7c67accc189307cf57a646f94953a42.jpg)

![dcf88d7236f3041b5260cc8ee7589d8451d526d9432bd3b24321784b7f71c81e.jpg](../icml_results/200_Symmetry%20in%20language%20statistics%20shapes%20the%20geometry%20of%20model%20representations/images/dcf88d7236f3041b5260cc8ee7589d8451d526d9432bd3b24321784b7f71c81e.jpg)

<a id="paper-0198"></a>

## Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers

### Images

![0468f573244f16e8b55257e3237cd51e91bf40d3ba8728144b6d08b99933038f.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/0468f573244f16e8b55257e3237cd51e91bf40d3ba8728144b6d08b99933038f.jpg)

![06baa71221f73475157be8ea688a01064a82e13dd56b087c85a79c2f58cc660a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/06baa71221f73475157be8ea688a01064a82e13dd56b087c85a79c2f58cc660a.jpg)

![072af555f2d46ecc13cd3a9ed6cfced2b2f18f65d7a6dc97716ae40c45c5bdb8.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/072af555f2d46ecc13cd3a9ed6cfced2b2f18f65d7a6dc97716ae40c45c5bdb8.jpg)

![2199ec9d3afb240d8ffa49bec9f849e7e72704c9f7a65a68a8eab0c5bd5157db.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/2199ec9d3afb240d8ffa49bec9f849e7e72704c9f7a65a68a8eab0c5bd5157db.jpg)

![423313bd7e9e3e1f8b8e2f4b041d9f70877bcca36965389ef5a68e463b29c92c.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/423313bd7e9e3e1f8b8e2f4b041d9f70877bcca36965389ef5a68e463b29c92c.jpg)

![5226fbe8a5726d8afeeb702ddac7b1086eae69097b0c1b0d421531f3fe1d7b11.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/5226fbe8a5726d8afeeb702ddac7b1086eae69097b0c1b0d421531f3fe1d7b11.jpg)

![680bb4f7492d80af00ccc6e766a30fe528f8a7e4228f748ab6b6f9be018b1c20.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/680bb4f7492d80af00ccc6e766a30fe528f8a7e4228f748ab6b6f9be018b1c20.jpg)

![69c64a150b2603670441c69565b3b93ed85e33fc6ac48fb5e8d816448ce639c1.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/69c64a150b2603670441c69565b3b93ed85e33fc6ac48fb5e8d816448ce639c1.jpg)

![8539e25ab444f985e86ac6f2bf1412f08816bd0afe7ca9f9d381ccba693200a4.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8539e25ab444f985e86ac6f2bf1412f08816bd0afe7ca9f9d381ccba693200a4.jpg)

![8bc6d8a11dcfba2f1deabf41a5db8bc8ca41ecfc4c852d424c69036f66ab5116.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8bc6d8a11dcfba2f1deabf41a5db8bc8ca41ecfc4c852d424c69036f66ab5116.jpg)

![8d6b0980188307ba9d960ddc0a419250d9f8f4a272ddae8a2e299bad7b8861e8.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8d6b0980188307ba9d960ddc0a419250d9f8f4a272ddae8a2e299bad7b8861e8.jpg)

![9521679e41f5347804d95a2593dc86adaefbf29bf319ebf3b566cd61216092b7.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/9521679e41f5347804d95a2593dc86adaefbf29bf319ebf3b566cd61216092b7.jpg)

![9977d119c275cddb13d8c71a996d199137d12f157ac3522236f051dc3f47f0f2.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/9977d119c275cddb13d8c71a996d199137d12f157ac3522236f051dc3f47f0f2.jpg)

![a5a1a7f6c3ef70e3cb0de48a33935758b64c3e24be0d9bea9bbf2ad812eab1ca.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/a5a1a7f6c3ef70e3cb0de48a33935758b64c3e24be0d9bea9bbf2ad812eab1ca.jpg)

![c20a2ae8626a8a90633324f591d5886e310bab2f4bc74d41fd33f5b97868f714.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/c20a2ae8626a8a90633324f591d5886e310bab2f4bc74d41fd33f5b97868f714.jpg)

![cbb4b5a0973a9e78925f502793907ad8a4844ad8b81767317914bbbf86d84f79.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/cbb4b5a0973a9e78925f502793907ad8a4844ad8b81767317914bbbf86d84f79.jpg)

![d074a6c6c98cf691368768ceb61c7dcc9e68ef450b2c40c24f58fc5a570ccc0a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/d074a6c6c98cf691368768ceb61c7dcc9e68ef450b2c40c24f58fc5a570ccc0a.jpg)

![d9b3847d9d6fe7b570d9aa02b0d498773feec6239e7471e54fcda966a7fb552b.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/d9b3847d9d6fe7b570d9aa02b0d498773feec6239e7471e54fcda966a7fb552b.jpg)

![dbff832e889f59bdde02ce174dba4ceac1663c4903114b219bd7b5af24bf663a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/dbff832e889f59bdde02ce174dba4ceac1663c4903114b219bd7b5af24bf663a.jpg)

![f51b097d3b2eaacce6782bf3dfdf6fad6762fc907720aa18925a765092e719aa.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f51b097d3b2eaacce6782bf3dfdf6fad6762fc907720aa18925a765092e719aa.jpg)

![f5eeca36e4c4d10f99393286e86abf501bc5265614afe41d816a95dd3b56f654.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f5eeca36e4c4d10f99393286e86abf501bc5265614afe41d816a95dd3b56f654.jpg)

![f6f130d06253a8363912a06d95a51ecb1d1c6d636d56e23589469e81f959b7e0.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f6f130d06253a8363912a06d95a51ecb1d1c6d636d56e23589469e81f959b7e0.jpg)

### Tables

![99df50d9f89ae14a37ed92553a62dfef2f67eec22584adf52747a1a499e95053.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/99df50d9f89ae14a37ed92553a62dfef2f67eec22584adf52747a1a499e95053.jpg)

![bba7e4fffb09a6df58cda2409b34a97eadbb977a051810fc04b5b21c432116ba.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/bba7e4fffb09a6df58cda2409b34a97eadbb977a051810fc04b5b21c432116ba.jpg)

![f9cd0f220208815d76f940c4ead0b76d50bcbc73aec775c2ed6c423941fbde67.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/f9cd0f220208815d76f940c4ead0b76d50bcbc73aec775c2ed6c423941fbde67.jpg)
