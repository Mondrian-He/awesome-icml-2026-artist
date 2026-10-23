# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 116 of 197

## 目录 (Table of Contents)

1. [HiPER: Hierarchical Plan–Execute RL for Multi-Turn LLM Agents](#paper-3796)
2. [A Close Look at Negative Label Guided Out-of-distribution Detection in Pre-trained Vision-Language Models](#paper-3797)
3. [IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](#paper-3798)
4. [Unified Multimodal Autoregressive Modeling with Shared Context—Visual Tokenizer is Key to Unification](#paper-3799)
5. [PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning](#paper-3800)
6. [STD-Former: Image-Conditioned Texture Dictionary Encoding with Sparse Topological Supervision for Texture Recognition](#paper-3801)
7. [SCNS: Continual Personalization of Diffusion Models via Submodular Concept Neuron Selection](#paper-3802)
8. [Towards Achieving Optimal Strong Regret and Constraint Violation via Computationally Efficient Model-free RL](#paper-3803)
9. [Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs](#paper-3804)
10. [Online Bayesian Experimental Design for Partially Observed Dynamical Systems](#paper-3805)
11. [Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization](#paper-3806)
12. [Generalizing Stochastic Smoothing for Differentiation and Gradient Estimation](#paper-3807)
13. [Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](#paper-3808)
14. [Geometric Entropy and Retrieval Phase Transitions in Continuous Dense Associative Memory](#paper-3809)
15. [Decomposing Out-of-Distribution Error in Conditional Flow Matching via Wasserstein Geometry](#paper-3810)
16. [Revisiting Distribution Correction Estimation for Offline Imitation Learning with Suboptimal Dataset](#paper-3811)
17. [Pruning at Initialisation through the lens of Graphon Limit: Convergence, Expressivity, and Generalisation](#paper-3812)
18. [Automatic Unsupervised Ensemble Outlier Model Selection](#paper-3813)
19. [Learning Manifold Data with Flow Matching](#paper-3814)
20. [Learning Adaptive Perturbation-Conditioned Contexts for Robust Transcriptional Response Prediction](#paper-3815)
21. [MarketSim: Simulating Stock Markets with Large-Scale Generative Agents](#paper-3816)
22. [Infinite-Dimensional Generative Diffusions via Doob’s h-Transform](#paper-3817)
23. [Learning the ESG Geometry with Domain Aware Language Models](#paper-3818)
24. [Temporal Context Reinstatement Drives Episodic-Like Order Memory in Long-Context Language Models](#paper-3819)
25. [ParalESN: Enabling parallel information processing in Reservoir Computing](#paper-3820)
26. [Not All Invariants Are Equal: Curating Training Data to Accelerate Program Verification with SLMs](#paper-3821)
27. [Goal-Oriented Lower-Tail Calibration of Gaussian Processes for Bayesian Optimization](#paper-3822)
28. [Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models](#paper-3823)
29. [A Diagnostic Study of Multi-Agent LLMs for Real-World Debates](#paper-3824)
30. [Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](#paper-3825)
31. [Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](#paper-3826)
32. [Capability Traps in DPO](#paper-3827)
33. [Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models](#paper-3828)

---


<a id="paper-3796"></a>

## HiPER: Hierarchical Plan–Execute RL for Multi-Turn LLM Agents

### Images

![0b29edbc9058e6073a2665b1d3db3c3716052c2636709799e4ec6b32aa113e95.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/0b29edbc9058e6073a2665b1d3db3c3716052c2636709799e4ec6b32aa113e95.jpg)

![13c3a1258747103a1c3a05ee77c8e2ce177c295d711e8f87543054ab98604376.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/13c3a1258747103a1c3a05ee77c8e2ce177c295d711e8f87543054ab98604376.jpg)

![17f669371440099362adfa8052f7661a43a0d8d3eaff30b1de45c079d0c280db.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/17f669371440099362adfa8052f7661a43a0d8d3eaff30b1de45c079d0c280db.jpg)

![2374d6c894b30a2c385358748a0e0810fbf8226f9e1286c9f770a1d82d42cdca.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/2374d6c894b30a2c385358748a0e0810fbf8226f9e1286c9f770a1d82d42cdca.jpg)

![355747370325588d7a7b0819671b5c0ee0a39aecf870bfb8a758d0ba8d5a18cf.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/355747370325588d7a7b0819671b5c0ee0a39aecf870bfb8a758d0ba8d5a18cf.jpg)

![355aaf13aff265978dc94736442d11b74ce4a79f33758c59d8b20c28c2df8a91.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/355aaf13aff265978dc94736442d11b74ce4a79f33758c59d8b20c28c2df8a91.jpg)

![3f30970fb82b50a3f8dbe076d137c3ddb3fbf9a3e96cfa1e6ba8a41226f112ad.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/3f30970fb82b50a3f8dbe076d137c3ddb3fbf9a3e96cfa1e6ba8a41226f112ad.jpg)

![407ed3bb64fc10a2d09e284e3a087d560cf295554b5d283e7ed1b2435474186a.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/407ed3bb64fc10a2d09e284e3a087d560cf295554b5d283e7ed1b2435474186a.jpg)

![4efc6f77a3c9cf1d548a8a38e3b18047d96194d3927bfdd2a829e7030648f900.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/4efc6f77a3c9cf1d548a8a38e3b18047d96194d3927bfdd2a829e7030648f900.jpg)

![5a842ab2a96eaf5109a9179486035efad4e98d9472b4cdac9635ea83f768c428.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/5a842ab2a96eaf5109a9179486035efad4e98d9472b4cdac9635ea83f768c428.jpg)

![5d5f0704139502320ad7197bf7f946470b9c7263dfefd62fec0a13aeb1671fc8.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/5d5f0704139502320ad7197bf7f946470b9c7263dfefd62fec0a13aeb1671fc8.jpg)

![5fde4701c11f96ac982305254a80686c75146f495aef93680567a7669d04e303.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/5fde4701c11f96ac982305254a80686c75146f495aef93680567a7669d04e303.jpg)

![7ab4ad1df3c81c23b8ebd3f4ec4551e0ac62e089f259212c4a18cd0fd6e132a9.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/7ab4ad1df3c81c23b8ebd3f4ec4551e0ac62e089f259212c4a18cd0fd6e132a9.jpg)

![7f01b747935f7d01081aea477a5a8f5c4a56fad7391520f41466506de382ec94.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/7f01b747935f7d01081aea477a5a8f5c4a56fad7391520f41466506de382ec94.jpg)

![7fb0d7afc460108e9eea8d666ab22350551714b610138528fb828aefa9f6aef5.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/7fb0d7afc460108e9eea8d666ab22350551714b610138528fb828aefa9f6aef5.jpg)

![8557185aa57b762d3f4d66930dc729972c0d6a8458c4ff7d84064d656d78b585.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/8557185aa57b762d3f4d66930dc729972c0d6a8458c4ff7d84064d656d78b585.jpg)

![886bffdc4aa51359fed530cbf10d2d8f6cedc3818e27532f7aef697d6e225fea.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/886bffdc4aa51359fed530cbf10d2d8f6cedc3818e27532f7aef697d6e225fea.jpg)

![8c604eec61c7a7f73d334610148d96678f36219b4ab3c42d416789a06c6b318f.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/8c604eec61c7a7f73d334610148d96678f36219b4ab3c42d416789a06c6b318f.jpg)

![94390ba07ed17191a0367f3ee9796a8d9da2f8ab1ec13794be5ef008d165ff27.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/94390ba07ed17191a0367f3ee9796a8d9da2f8ab1ec13794be5ef008d165ff27.jpg)

![95dad2de15c39712a9ef89b8dd02283502947cd8a2102e8a1772d1b38c4fc627.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/95dad2de15c39712a9ef89b8dd02283502947cd8a2102e8a1772d1b38c4fc627.jpg)

![9c3d2a0999292edc2da99460bdf45a38ee5d3d093cd3e83d4feda4913f9fce55.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/9c3d2a0999292edc2da99460bdf45a38ee5d3d093cd3e83d4feda4913f9fce55.jpg)

![a111373a430cbaedab6a87a9dabe040dc5d65b17de1a70fdae6ec09c16a8bc20.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/a111373a430cbaedab6a87a9dabe040dc5d65b17de1a70fdae6ec09c16a8bc20.jpg)

![a88fb0afbded0cf3d8e8c90eed855e2e47626a1e20e131ea405352cb5b8463b8.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/a88fb0afbded0cf3d8e8c90eed855e2e47626a1e20e131ea405352cb5b8463b8.jpg)

![ae7e898e7c7d1637860b467c1040828692a03880866c558e124d9767f2d31a65.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/ae7e898e7c7d1637860b467c1040828692a03880866c558e124d9767f2d31a65.jpg)

![b8b575dd81536d6b8a91f580a594408b5a48fed80067c12a6f3ca7de5b0f61fd.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/b8b575dd81536d6b8a91f580a594408b5a48fed80067c12a6f3ca7de5b0f61fd.jpg)

![bd49b304d903f9365c818e662bd0c2393e717fc704ab9623b4267d6c36688c9f.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/bd49b304d903f9365c818e662bd0c2393e717fc704ab9623b4267d6c36688c9f.jpg)

![c71a2e8991d6ffefac4188cccc9d2c4aaa6758c7852c37d4a92af7414cc8ae5b.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/c71a2e8991d6ffefac4188cccc9d2c4aaa6758c7852c37d4a92af7414cc8ae5b.jpg)

![c89aaa5ae6302900a884b201dfbfa866e8a63140373dfaf5c123a866263d971a.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/c89aaa5ae6302900a884b201dfbfa866e8a63140373dfaf5c123a866263d971a.jpg)

![db0c342e6477a79530cf954aac218c6ab512b4b286d1541776208eb6b06c75d4.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/db0c342e6477a79530cf954aac218c6ab512b4b286d1541776208eb6b06c75d4.jpg)

![df6a1bb9485c0d0fc94f9940e570adca4bcedf8166cf83c20ef4952c3458e3f0.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/df6a1bb9485c0d0fc94f9940e570adca4bcedf8166cf83c20ef4952c3458e3f0.jpg)

![e085fdb33ff5efe4e7d407c4b2c8b36c1c54974745082c6103389d94266ff127.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/e085fdb33ff5efe4e7d407c4b2c8b36c1c54974745082c6103389d94266ff127.jpg)

![e1914f3459a00b4c5fdf833797ca287936aca8e20f2d88b9abf3d011359b68ab.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/e1914f3459a00b4c5fdf833797ca287936aca8e20f2d88b9abf3d011359b68ab.jpg)

![fb582b0c0e7156d80f573e51f3fb4cdac0f225928d8c7dbcf645bcb95f203465.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/images/fb582b0c0e7156d80f573e51f3fb4cdac0f225928d8c7dbcf645bcb95f203465.jpg)

### Tables

![067499e0c39c2c7e720be45e885b1d490d71d12c0693288047d916938e6723b8.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/067499e0c39c2c7e720be45e885b1d490d71d12c0693288047d916938e6723b8.jpg)

![18f6e641d454b51ae68e99786723498741b24bbf2230e87eb3c228e3cdf40c3c.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/18f6e641d454b51ae68e99786723498741b24bbf2230e87eb3c228e3cdf40c3c.jpg)

![26c6c067ee96136ebc61f195f0f4a360031579b167d05c5d872f1b0ed9e05f28.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/26c6c067ee96136ebc61f195f0f4a360031579b167d05c5d872f1b0ed9e05f28.jpg)

![2d78ad45d5c18e09d1e4a400fb31d9f148dd4d4e127cbacba6bc0af4537facd3.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/2d78ad45d5c18e09d1e4a400fb31d9f148dd4d4e127cbacba6bc0af4537facd3.jpg)

![3dbc0d397408932fb949fd58b3ca5a41bc2ce4d9cda50d04a28c40f74d0c7b6c.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/3dbc0d397408932fb949fd58b3ca5a41bc2ce4d9cda50d04a28c40f74d0c7b6c.jpg)

![4172e64f63622f6bc126c78fad9fc1126c07be1f4b3496ecc649c67c7cc0b2c3.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/4172e64f63622f6bc126c78fad9fc1126c07be1f4b3496ecc649c67c7cc0b2c3.jpg)

![5ad0f304c01d29e1506c78822acd563359097703ea07b7b0cf8a07573405364f.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/5ad0f304c01d29e1506c78822acd563359097703ea07b7b0cf8a07573405364f.jpg)

![80808978b2e7acb27479a5c4c63b552dfae2850ca23ebcc29b2159420f343a62.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/80808978b2e7acb27479a5c4c63b552dfae2850ca23ebcc29b2159420f343a62.jpg)

![8b7f4954b69f10d5ab4e4f0b848f326d3e071a5c3172720618d927591ad4bab3.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/8b7f4954b69f10d5ab4e4f0b848f326d3e071a5c3172720618d927591ad4bab3.jpg)

![b2ce98e2bf608bffe7a9621293fc41239d0c4d82a9f583605dac72512974c58b.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/b2ce98e2bf608bffe7a9621293fc41239d0c4d82a9f583605dac72512974c58b.jpg)

![b7a66524e50a6f9c56c4198a6db58d9373b71f7ec1eafc908f3ea8b2dd6d53ef.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/b7a66524e50a6f9c56c4198a6db58d9373b71f7ec1eafc908f3ea8b2dd6d53ef.jpg)

![ba313d75f5b233adbea28055e3500cefebed752a8e1c631c29acacf66aea931d.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/ba313d75f5b233adbea28055e3500cefebed752a8e1c631c29acacf66aea931d.jpg)

![e294ac1a93f63afb5176d6bea7ef7ce126dc89db62cb61833208082dc16314e4.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/e294ac1a93f63afb5176d6bea7ef7ce126dc89db62cb61833208082dc16314e4.jpg)

![e4efd090a74a1d1396ad9dfb3199ba41dafbbd516a9929b7b0328a3921397858.jpg](../icml_results/3835_MIST_%20Moment-Aligned%20Invariant%20Stability%20Transform%20for%20Robust%20Flow%20Matching/tables/e4efd090a74a1d1396ad9dfb3199ba41dafbbd516a9929b7b0328a3921397858.jpg)

## HiPER: Hierarchical Plan–Execute RL for Multi-Turn LLM Agents


### Images

![a1b450f7fd67a261d49a90d73a2f9de7e9cdb6128a1f3845785d652f58a10596.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/images/a1b450f7fd67a261d49a90d73a2f9de7e9cdb6128a1f3845785d652f58a10596.jpg)

![a83e611f67017a7d379708bb28cf6678df17f95bf5a52aa0260d6fcdf9309ed9.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/images/a83e611f67017a7d379708bb28cf6678df17f95bf5a52aa0260d6fcdf9309ed9.jpg)

![b70b941ba072a362ddc553f8a015a23b0fd9f3e7bd7a37043e25a1332d457c53.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/images/b70b941ba072a362ddc553f8a015a23b0fd9f3e7bd7a37043e25a1332d457c53.jpg)

![d1a1e7bbbcf0b6b72f7edaa3bb5d2234a664e6e594cbe7769a266d206f7511c0.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/images/d1a1e7bbbcf0b6b72f7edaa3bb5d2234a664e6e594cbe7769a266d206f7511c0.jpg)

![efe82b632209b0db13091f935094ed58f799867847a7ca5440910615272b2fb3.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/images/efe82b632209b0db13091f935094ed58f799867847a7ca5440910615272b2fb3.jpg)

### Tables

![08e0be737d83af3d750c72bf4a01703c1d3e32eae6c3a3b504bb4f15ef4a4772.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/tables/08e0be737d83af3d750c72bf4a01703c1d3e32eae6c3a3b504bb4f15ef4a4772.jpg)

![52725670d6cd7bc9fe1daac59ad9b3225eae114b28be4ff9c8bbbfbd9ef6de7a.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/tables/52725670d6cd7bc9fe1daac59ad9b3225eae114b28be4ff9c8bbbfbd9ef6de7a.jpg)

![638e3a1bba198018f985eac078b39fbd85eb0eeecfc6eaa626368ebe1055ada2.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/tables/638e3a1bba198018f985eac078b39fbd85eb0eeecfc6eaa626368ebe1055ada2.jpg)

![c65f3a5fc76a0e2d488ac052e9e1913248777432ca1581abd8df5bae70fd9387.jpg](../icml_results/3836_HiPER_%20Hierarchical%20Plan%E2%80%93Execute%20RL%20for%20Multi-Turn%20LLM%20Agents/tables/c65f3a5fc76a0e2d488ac052e9e1913248777432ca1581abd8df5bae70fd9387.jpg)

<a id="paper-3797"></a>

## A Close Look at Negative Label Guided Out-of-distribution Detection in Pre-trained Vision-Language Models


### Images

![398975176ba2b4a5320481bf431d75fb9faa5ced29105abc3b52d5d6cf73f5c5.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/images/398975176ba2b4a5320481bf431d75fb9faa5ced29105abc3b52d5d6cf73f5c5.jpg)

### Tables

![0c2aa7b895db5510a01ab3dc1cc2c803c103b4f3a5b85bb618f01c4204d16190.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/0c2aa7b895db5510a01ab3dc1cc2c803c103b4f3a5b85bb618f01c4204d16190.jpg)

![0fb85d3ff742eefe26e5d844d3ee1972d6ce2a3635d58727962761c94b675c9c.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/0fb85d3ff742eefe26e5d844d3ee1972d6ce2a3635d58727962761c94b675c9c.jpg)

![3ce9ac1b40ac490ef0962e27d2c332a1ec131ce51b95c6b6c6fbd14553236b21.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/3ce9ac1b40ac490ef0962e27d2c332a1ec131ce51b95c6b6c6fbd14553236b21.jpg)

![40b0befc497e2ad49767afaa8cf7b0cea466be7b53dd74f9843c756cab31135a.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/40b0befc497e2ad49767afaa8cf7b0cea466be7b53dd74f9843c756cab31135a.jpg)

![4e4fdac4597ca7595920ae7ab44a7b3a6c05943245710ea0a31fa4fe8126337b.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/4e4fdac4597ca7595920ae7ab44a7b3a6c05943245710ea0a31fa4fe8126337b.jpg)

![8c7659d26ed0c1675c6f746a9ed88f204e291de6d3fe9e244658f8f5c4d6da04.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/8c7659d26ed0c1675c6f746a9ed88f204e291de6d3fe9e244658f8f5c4d6da04.jpg)

![8dac2c43f43a2a6bdd1154222b273bbe6b2db00bc0e63d29365b47337d1050ca.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/8dac2c43f43a2a6bdd1154222b273bbe6b2db00bc0e63d29365b47337d1050ca.jpg)

![bab77e2f18454b15d532468ccf0ef85833db213527efd788c3b23c4b13dfbcff.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/bab77e2f18454b15d532468ccf0ef85833db213527efd788c3b23c4b13dfbcff.jpg)

![de521e5c4eba9c0868e97cabd85f663f147fda7bd19964c653f6a2805a1de9c1.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/de521e5c4eba9c0868e97cabd85f663f147fda7bd19964c653f6a2805a1de9c1.jpg)

![f0f566a53784a865313f5d703f1c78fbf3d9422b251572eb532b100f852a61f8.jpg](../icml_results/3837_A%20Close%20Look%20at%20Negative%20Label%20Guided%20Out-of-distribution%20Detection%20in%20Pre-trained%20Vision-Language%20M/tables/f0f566a53784a865313f5d703f1c78fbf3d9422b251572eb532b100f852a61f8.jpg)

<a id="paper-3798"></a>

## IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring


### Images

![20dab586e43aaef274c823ec3f0c4deb5a33a154f1bb592f233d97eb0a06fd68.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/20dab586e43aaef274c823ec3f0c4deb5a33a154f1bb592f233d97eb0a06fd68.jpg)

![2e2793ab94efb7033db977f7da2295046ead8f800a6a89d4a7b83ba022429d57.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/2e2793ab94efb7033db977f7da2295046ead8f800a6a89d4a7b83ba022429d57.jpg)

![3ad02feeba7d212fedba7815ba0739ad6489b8d85438d24a0152129d2c437f22.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/3ad02feeba7d212fedba7815ba0739ad6489b8d85438d24a0152129d2c437f22.jpg)

![3bf2976dd35503d837c24fee51c64a119e26d9fd4bd11f21763824726473f52a.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/3bf2976dd35503d837c24fee51c64a119e26d9fd4bd11f21763824726473f52a.jpg)

![66a4c0692afca15eb86f02c842257257f0a81062a3223a0a0bce4b04255da6e8.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/66a4c0692afca15eb86f02c842257257f0a81062a3223a0a0bce4b04255da6e8.jpg)

![7fc6af8fca73379579dc808bb43ff49234c1978aa5d5792776486cebfb28dab5.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/7fc6af8fca73379579dc808bb43ff49234c1978aa5d5792776486cebfb28dab5.jpg)

![893925120be16da9c876b21cbb9efbb53099e9e95cabc1459228df68201ee76d.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/893925120be16da9c876b21cbb9efbb53099e9e95cabc1459228df68201ee76d.jpg)

![b51d15e507cf08adfc184184987706fad985771a2270f04bb6eab822f6ba4d8c.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/b51d15e507cf08adfc184184987706fad985771a2270f04bb6eab822f6ba4d8c.jpg)

![bdf40e184382662cbb1f9e4720a42b183c6ff38bd8ffc13e45313a91396b581c.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/bdf40e184382662cbb1f9e4720a42b183c6ff38bd8ffc13e45313a91396b581c.jpg)

![d6acf1fd72a6dbd55fafffc290d244c0e74bc342085160ad42a3715ffe2d0ce7.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/d6acf1fd72a6dbd55fafffc290d244c0e74bc342085160ad42a3715ffe2d0ce7.jpg)

![da9c5cc66ee993bc1e0ff050f5d88856bd95888c5d0d5997a8cc23d086c3881f.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/da9c5cc66ee993bc1e0ff050f5d88856bd95888c5d0d5997a8cc23d086c3881f.jpg)

![e802fd358b1a6ae2cf8c947c5d3c52b3c0ffb167f76758603ae14b68695c30b4.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/e802fd358b1a6ae2cf8c947c5d3c52b3c0ffb167f76758603ae14b68695c30b4.jpg)

![f48c24ab56557c63214ec62a1d222ee77d987412b34122b776a919c93573c952.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/f48c24ab56557c63214ec62a1d222ee77d987412b34122b776a919c93573c952.jpg)

![f5ef5c13b099b07d7d844347a0bc36d458a8e4028f6a7447f36fc3dd09eba15d.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/images/f5ef5c13b099b07d7d844347a0bc36d458a8e4028f6a7447f36fc3dd09eba15d.jpg)

### Tables

![12729d5d71a76e972342f02edeebe23f24fc1ff35d15b3906a04fb97c48245de.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/12729d5d71a76e972342f02edeebe23f24fc1ff35d15b3906a04fb97c48245de.jpg)

![2fc4caa51557ce4a969d9d23e277ad9a7cc781f7479acb322d3a300fdcfb131e.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/2fc4caa51557ce4a969d9d23e277ad9a7cc781f7479acb322d3a300fdcfb131e.jpg)

![3e6cd715645b14cedba06b910be1d7cb67e8278237f7873aebf99c7f9e2d5151.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/3e6cd715645b14cedba06b910be1d7cb67e8278237f7873aebf99c7f9e2d5151.jpg)

![6eee6d86bbeb117dd9bce058a6fe86382f11b6fe610b555fb048db5c6c9d885c.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/6eee6d86bbeb117dd9bce058a6fe86382f11b6fe610b555fb048db5c6c9d885c.jpg)

![766145b849c3d662201cc85709bff5712598947ffae02df782506d1567cc89ea.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/766145b849c3d662201cc85709bff5712598947ffae02df782506d1567cc89ea.jpg)

![7d9a6f71a5723c0be2a15a9c512f23c897a21273970309797dcd690ac35ce7bf.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/7d9a6f71a5723c0be2a15a9c512f23c897a21273970309797dcd690ac35ce7bf.jpg)

![818e042dbdcf3ef37f06371f973fce1debd497eb3ce43d24f8b94b6e482d631e.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/818e042dbdcf3ef37f06371f973fce1debd497eb3ce43d24f8b94b6e482d631e.jpg)

![92dfe3d77adaf718033fe54133bc0112f725c0c2232e38c83174588e28e52163.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/92dfe3d77adaf718033fe54133bc0112f725c0c2232e38c83174588e28e52163.jpg)

![991a02f0392b835b4a417cfc55d73fc1d7c467d63048ed2493f67febf89f8f8f.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/991a02f0392b835b4a417cfc55d73fc1d7c467d63048ed2493f67febf89f8f8f.jpg)

![99dc068b89f3faab993c539f0439c49353972bc6275afeb1e7c36c30213160e1.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/99dc068b89f3faab993c539f0439c49353972bc6275afeb1e7c36c30213160e1.jpg)

![a36f7702626a53e7bbad59b0535cc6904876890a1e5432f1a66f86e250bac0b3.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/a36f7702626a53e7bbad59b0535cc6904876890a1e5432f1a66f86e250bac0b3.jpg)

![a489da2ea0741ef0b3527e75e88cc572bd6c8f75ec2eded05ccf3b5633f3533e.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/a489da2ea0741ef0b3527e75e88cc572bd6c8f75ec2eded05ccf3b5633f3533e.jpg)

![a54f4a13a7e0f1ed17792406816ad45bf6508cdd95833138376d03d72197150d.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/a54f4a13a7e0f1ed17792406816ad45bf6508cdd95833138376d03d72197150d.jpg)

![a962c3eb9274671062fb23514ee018c2fb3e263d94f24a0cf6294aecaa422799.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/a962c3eb9274671062fb23514ee018c2fb3e263d94f24a0cf6294aecaa422799.jpg)

![ced8640a4c15a2bff90b39a07c936db71f298a0c370060926e3e6ab69d0d7b03.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/ced8640a4c15a2bff90b39a07c936db71f298a0c370060926e3e6ab69d0d7b03.jpg)

![ddc5d0ec82a24ab41654ab8580d930fd3471acc4089ad2ee7b8cbc4535d7fd5a.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/ddc5d0ec82a24ab41654ab8580d930fd3471acc4089ad2ee7b8cbc4535d7fd5a.jpg)

![e9aca5d56f9a1ee1b979fd3de833c66446b8a2b693f2ae161086d4160a9d7043.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/e9aca5d56f9a1ee1b979fd3de833c66446b8a2b693f2ae161086d4160a9d7043.jpg)

![fd3b5e07a2e003a2d177bffae19a8047926cc066e9b786ea20a2c6a9251cb752.jpg](../icml_results/3838_IQA-Spider_%20Unifying%20Multi-Granularity%20Image%20Quality%20Assessment%20with%20Reasoning%2C%20Grounding%20and%20Referr/tables/fd3b5e07a2e003a2d177bffae19a8047926cc066e9b786ea20a2c6a9251cb752.jpg)

<a id="paper-3799"></a>

## Unified Multimodal Autoregressive Modeling with Shared Context—Visual Tokenizer is Key to Unification


### Images

![48c2eb6fdc53e8f02949907d59654a980f1d6fdc9e6d11fcd4c7e632ac6c9cc8.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/48c2eb6fdc53e8f02949907d59654a980f1d6fdc9e6d11fcd4c7e632ac6c9cc8.jpg)

![70bbf0a1650f3e8b18172cd266ef85af6f7072f25bdf91af38a15c180bf2b9c2.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/70bbf0a1650f3e8b18172cd266ef85af6f7072f25bdf91af38a15c180bf2b9c2.jpg)

![7b4d839c84467994564caabeac288a855966c24856a7a9e03f22239c87b09a0e.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/7b4d839c84467994564caabeac288a855966c24856a7a9e03f22239c87b09a0e.jpg)

![a90d3304adeea02f1cb54909992b9d983dc7d3327292c37a893cac24cbd84f76.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/a90d3304adeea02f1cb54909992b9d983dc7d3327292c37a893cac24cbd84f76.jpg)

![b35a0fb5a4e9038eddd49b662502821529789e197bb50739406fda15e6ef60e0.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/b35a0fb5a4e9038eddd49b662502821529789e197bb50739406fda15e6ef60e0.jpg)

![da2f01f4aa17a5895cd8550e603704bd2d51566ceceff309b47a436b9f0b0547.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/images/da2f01f4aa17a5895cd8550e603704bd2d51566ceceff309b47a436b9f0b0547.jpg)

### Tables

![087fe919c9d47a2589b710a9cc7b1f284d018d396a8c389366bc22667abc1480.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/087fe919c9d47a2589b710a9cc7b1f284d018d396a8c389366bc22667abc1480.jpg)

![193813f7d5e003ea90f04c17e0325e3b0ef118d796aa90b39d07ac8905123eb1.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/193813f7d5e003ea90f04c17e0325e3b0ef118d796aa90b39d07ac8905123eb1.jpg)

![5d0e1bcad3a096a32926590db8967a6c48250a956c827b346bf99836ce2701c8.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/5d0e1bcad3a096a32926590db8967a6c48250a956c827b346bf99836ce2701c8.jpg)

![6396d257f8ffbcb45e4847356c7eb4b91ccd7c76026db3e4306f77b6862861a1.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/6396d257f8ffbcb45e4847356c7eb4b91ccd7c76026db3e4306f77b6862861a1.jpg)

![a870886b508895c7b75b932877bfd5b391a2b945db28d8592b60259195bd11d6.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/a870886b508895c7b75b932877bfd5b391a2b945db28d8592b60259195bd11d6.jpg)

![a8a2fb64c43780d2d7ee23e72cec69c6ced85962065608e29b8fab3dd937a454.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/a8a2fb64c43780d2d7ee23e72cec69c6ced85962065608e29b8fab3dd937a454.jpg)

![d80edd6bce25384fdd43d49f7bdaec0ee742b169488dc668f4a7af086df8bdad.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/d80edd6bce25384fdd43d49f7bdaec0ee742b169488dc668f4a7af086df8bdad.jpg)

![d8ac7518dfdf4def0d1053b87df0857ac2582262b01c1d3825d6cf8cb266ef34.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/d8ac7518dfdf4def0d1053b87df0857ac2582262b01c1d3825d6cf8cb266ef34.jpg)

![dd86da3202502fd0af38cafbb1a24cb5f9526e761c9050037a0ccf9b2f55a002.jpg](../icml_results/3839_Unified%20Multimodal%20Autoregressive%20Modeling%20with%20Shared%20Context%E2%80%94Visual%20Tokenizer%20is%20Key%20to%20Unificatio/tables/dd86da3202502fd0af38cafbb1a24cb5f9526e761c9050037a0ccf9b2f55a002.jpg)

<a id="paper-3800"></a>

## PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning


### Images

![0e6b54707598598499e023a116a18b24061e0cf56ddb9fd7e56ee3cbd50a2da5.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/images/0e6b54707598598499e023a116a18b24061e0cf56ddb9fd7e56ee3cbd50a2da5.jpg)

![2bb2b5bd70243d17212dce38c620c69d1ce2ce83a9830adf65cda2c403d0ba5c.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/images/2bb2b5bd70243d17212dce38c620c69d1ce2ce83a9830adf65cda2c403d0ba5c.jpg)

![3cdd0570ded022de6258802fc5699507af4fcb4f002b35e9fdd26727738b1997.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/images/3cdd0570ded022de6258802fc5699507af4fcb4f002b35e9fdd26727738b1997.jpg)

![ab154b050511a78966265331aa7c5b41af272a0923bbdbdd9779d074110f1e04.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/images/ab154b050511a78966265331aa7c5b41af272a0923bbdbdd9779d074110f1e04.jpg)

![dc5bd016a2927b8b4f83793d48c1a3ca88083413fcb0cbb7271bf50793c79158.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/images/dc5bd016a2927b8b4f83793d48c1a3ca88083413fcb0cbb7271bf50793c79158.jpg)

### Tables

![1101f0f91be9973bba3e0eca8a41aef73707c588babdc70b3694f2a79ff16077.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/1101f0f91be9973bba3e0eca8a41aef73707c588babdc70b3694f2a79ff16077.jpg)

![1967d54509554f741e5b7e7b48979144457b785d6f65ceaeb58bcd2041ddd7e5.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/1967d54509554f741e5b7e7b48979144457b785d6f65ceaeb58bcd2041ddd7e5.jpg)

![29db968778cd93e23bc8ec16f18e9c080f35d1432758309798de1a6c3c99eeea.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/29db968778cd93e23bc8ec16f18e9c080f35d1432758309798de1a6c3c99eeea.jpg)

![3b564b3c74ae9662eabd7325ec01ffea4fb548aaaede636b29057fe4eb179e14.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/3b564b3c74ae9662eabd7325ec01ffea4fb548aaaede636b29057fe4eb179e14.jpg)

![4e3cc7fb4b5d3c323749f06ade0034b8e2dbe00226d60fa47ba3b8c583dbe22d.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/4e3cc7fb4b5d3c323749f06ade0034b8e2dbe00226d60fa47ba3b8c583dbe22d.jpg)

![4e9977b9aa1f0e88f1bc024eb4c5249951c42eb905b179c77ace0294071362ef.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/4e9977b9aa1f0e88f1bc024eb4c5249951c42eb905b179c77ace0294071362ef.jpg)

![797b3fb66bf11345cf10fc7136646086642f1d171e14f3b25e162b8464a1e5c3.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/797b3fb66bf11345cf10fc7136646086642f1d171e14f3b25e162b8464a1e5c3.jpg)

![7f4b595a9b6fcb9c8a90e9a7029d72bdd9d6e1f45bdaa019d8b43934fe145b5e.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/7f4b595a9b6fcb9c8a90e9a7029d72bdd9d6e1f45bdaa019d8b43934fe145b5e.jpg)

![8b2540fff7af579d98ef9049ded76761e3f92bb55a36cfc9a6a17c8e5204387c.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/8b2540fff7af579d98ef9049ded76761e3f92bb55a36cfc9a6a17c8e5204387c.jpg)

![a154c9192c51ea48cd81380d7702f5aca014b9715834b8031aef87cad87659fc.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/a154c9192c51ea48cd81380d7702f5aca014b9715834b8031aef87cad87659fc.jpg)

![b3feb7d07d74e15eb2cfd64e6cbcff736f7d041178129264651ca7504e1508fa.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/b3feb7d07d74e15eb2cfd64e6cbcff736f7d041178129264651ca7504e1508fa.jpg)

![eca3b723a4f9014d9bc6f6b8c46b72399f19e87e3e20e32054cdc3cff85690c0.jpg](../icml_results/3840_PADD_%20Path-Aligned%20Decompression%20Distillation%20for%20Non-Router%20Teacher%20to%20Guide%20MoE%20Student%20Learning/tables/eca3b723a4f9014d9bc6f6b8c46b72399f19e87e3e20e32054cdc3cff85690c0.jpg)

<a id="paper-3801"></a>

## STD-Former: Image-Conditioned Texture Dictionary Encoding with Sparse Topological Supervision for Texture Recognition


### Images

![0836ac66290da12c598dfa201a9d71fcc89b8ab70dfd43c7140f653a8c6e5eed.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/images/0836ac66290da12c598dfa201a9d71fcc89b8ab70dfd43c7140f653a8c6e5eed.jpg)

![387be115f32a4486f5e77d6874e50f037051448d866713c0f4db1a860bff0040.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/images/387be115f32a4486f5e77d6874e50f037051448d866713c0f4db1a860bff0040.jpg)

![8859aadf020698eca9c4517fed07fc069a78a9e365200b89aaa6783497f44824.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/images/8859aadf020698eca9c4517fed07fc069a78a9e365200b89aaa6783497f44824.jpg)

![a00fb3069f8c8f629b1fc9dd4b67061437b891db21982917077bd9871d382e21.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/images/a00fb3069f8c8f629b1fc9dd4b67061437b891db21982917077bd9871d382e21.jpg)

### Tables

![20e2e75efeedecf3169ef63666c342e5ceab0a0ab080a068139ec913cac6cd9e.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/20e2e75efeedecf3169ef63666c342e5ceab0a0ab080a068139ec913cac6cd9e.jpg)

![2d44cdfefb7eb3b365d6f1fa6381d377bd23b826caac256228f806c3696308d6.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/2d44cdfefb7eb3b365d6f1fa6381d377bd23b826caac256228f806c3696308d6.jpg)

![301d3cf94e066cd5728756338c4948e310f2c19880283631cc399128ccae6222.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/301d3cf94e066cd5728756338c4948e310f2c19880283631cc399128ccae6222.jpg)

![388e005952bff11b7e0247f19aaa29bfc2c478b623bc733fffa6aac9746ebab0.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/388e005952bff11b7e0247f19aaa29bfc2c478b623bc733fffa6aac9746ebab0.jpg)

![40f3181e16431842fa20428b118cdd0247f55fbee7b49fb3b3a683103dd3d329.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/40f3181e16431842fa20428b118cdd0247f55fbee7b49fb3b3a683103dd3d329.jpg)

![440c6b8e114e868d8f80cbeba4628ce7b480ea172f6cba4f396cb6809ed4cdf5.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/440c6b8e114e868d8f80cbeba4628ce7b480ea172f6cba4f396cb6809ed4cdf5.jpg)

![4ddf73ca99e318223254f2aa9b4e8f2183f6efa0d5108c66dc138c38ba59357c.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/4ddf73ca99e318223254f2aa9b4e8f2183f6efa0d5108c66dc138c38ba59357c.jpg)

![4f5b1c0542ae2b7598b6b2a0f0946e40deb29c6cd1b7208242cae664b0b0cafa.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/4f5b1c0542ae2b7598b6b2a0f0946e40deb29c6cd1b7208242cae664b0b0cafa.jpg)

![54c3e85787f030151ce60cd3cf51b42aff485712bad30f7f0dd1d118c76b3afd.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/54c3e85787f030151ce60cd3cf51b42aff485712bad30f7f0dd1d118c76b3afd.jpg)

![74a04f460aaeaa0d4c4b316206c2635f25f7bcbb24e7db98b8571b98707c76bb.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/74a04f460aaeaa0d4c4b316206c2635f25f7bcbb24e7db98b8571b98707c76bb.jpg)

![8d1c0d458d55426450558b052c987132c9ebac477448d3f47be061f96ced4c14.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/8d1c0d458d55426450558b052c987132c9ebac477448d3f47be061f96ced4c14.jpg)

![8f40d8a08e99e68e4479c21ec46b649320f912d580d55ee1179bd93c7c88519b.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/8f40d8a08e99e68e4479c21ec46b649320f912d580d55ee1179bd93c7c88519b.jpg)

![aa7877f32508bd303eb50c5b102ca40ae8a43c594bb5499ef4a44152af674e82.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/aa7877f32508bd303eb50c5b102ca40ae8a43c594bb5499ef4a44152af674e82.jpg)

![b35fc206f2fb48e813c383fd4d81f7896fb10ec03bc5d8185c60f060dfc33488.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/b35fc206f2fb48e813c383fd4d81f7896fb10ec03bc5d8185c60f060dfc33488.jpg)

![b989f0e62683a09318e9d977a4e11fb413e5af5b1e0b126a2a806490a81d9b28.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/b989f0e62683a09318e9d977a4e11fb413e5af5b1e0b126a2a806490a81d9b28.jpg)

![be124e4ae45bb1d7fcbdd16080a39d6d5abbcb7cd4a9435d375c183c86cb20a6.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/be124e4ae45bb1d7fcbdd16080a39d6d5abbcb7cd4a9435d375c183c86cb20a6.jpg)

![ca28cdbeccf0baada6b015ddbd16c30e076b76935159cdd775998528d7e32020.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/ca28cdbeccf0baada6b015ddbd16c30e076b76935159cdd775998528d7e32020.jpg)

![ca6166857298dfaf73a026c944b9e17828744b98834fd6feab3bff73cf43f3e4.jpg](../icml_results/3841_STD-Former_%20Image-Conditioned%20Texture%20Dictionary%20Encoding%20with%20Sparse%20Topological%20Supervision%20for%20Te/tables/ca6166857298dfaf73a026c944b9e17828744b98834fd6feab3bff73cf43f3e4.jpg)

<a id="paper-3802"></a>

## SCNS: Continual Personalization of Diffusion Models via Submodular Concept Neuron Selection


### Images

![01346dd249a1e3a3e21ff12c8ae5b64fe9b872ad0239a8f195a556eba2a8cf13.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/01346dd249a1e3a3e21ff12c8ae5b64fe9b872ad0239a8f195a556eba2a8cf13.jpg)

![2de055f2233be42210ebb2b8269c858fcbefa752bfde324a62e0ddd175ee0c62.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/2de055f2233be42210ebb2b8269c858fcbefa752bfde324a62e0ddd175ee0c62.jpg)

![53881b9b2138cbe6802da3f329ff979a3b3f7196714e679545bcffb50bb51e35.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/53881b9b2138cbe6802da3f329ff979a3b3f7196714e679545bcffb50bb51e35.jpg)

![6cf455fcc9d146db731059754daa341b9f158093fea6f59f8b74792345d0e6ca.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/6cf455fcc9d146db731059754daa341b9f158093fea6f59f8b74792345d0e6ca.jpg)

![72035977f37774639388bde14f98f970a6b43fdbfd6f34ca6a747f254748816e.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/72035977f37774639388bde14f98f970a6b43fdbfd6f34ca6a747f254748816e.jpg)

![8efd7cb70fdac24a5897d2d406e0a87ead749e325dc5a75c18a6ba213b5ff095.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/8efd7cb70fdac24a5897d2d406e0a87ead749e325dc5a75c18a6ba213b5ff095.jpg)

![9559113a18464c54d32c5c6790c5e4ae150b145498659382e5e5909f65750c2b.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/9559113a18464c54d32c5c6790c5e4ae150b145498659382e5e5909f65750c2b.jpg)

![b1b65d6ceb3c8197f91ad9ef086389c7eb3cf2511f675aacddb3d278e1c4fb8d.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/b1b65d6ceb3c8197f91ad9ef086389c7eb3cf2511f675aacddb3d278e1c4fb8d.jpg)

![c02ab4f162126531ad2b2ffad8fc325d03db41d2378641f2cb1cfc6f82199920.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/images/c02ab4f162126531ad2b2ffad8fc325d03db41d2378641f2cb1cfc6f82199920.jpg)

### Tables

![259b6b69f9363eb40be0a6c09fcef4359f02ad599316debbf5e8b55e68aadafa.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/259b6b69f9363eb40be0a6c09fcef4359f02ad599316debbf5e8b55e68aadafa.jpg)

![32e1e6f74e4d2858c0e7366ba3722d2f3f239e65cb6845b74fb5c38c0098abb6.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/32e1e6f74e4d2858c0e7366ba3722d2f3f239e65cb6845b74fb5c38c0098abb6.jpg)

![34547f03f9d6da6b3ed113fd5acfaea1083407fced1331a5a6cecb5b905b84dc.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/34547f03f9d6da6b3ed113fd5acfaea1083407fced1331a5a6cecb5b905b84dc.jpg)

![4334ed31adb20e4d5c4eb4550147a5f24578672b1e6c0092070ccd52630f2715.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/4334ed31adb20e4d5c4eb4550147a5f24578672b1e6c0092070ccd52630f2715.jpg)

![4bf5d6cbaad6544405eb7be2671bea703e13bc4ca4f53528d2025acc91ce0070.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/4bf5d6cbaad6544405eb7be2671bea703e13bc4ca4f53528d2025acc91ce0070.jpg)

![7e0a0b44d860daef3a57d4c7247212c07a286d1ad77d8f0011936b75af819a13.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/7e0a0b44d860daef3a57d4c7247212c07a286d1ad77d8f0011936b75af819a13.jpg)

![89894c0e83b669613139a58b319bfa995fb2fb98339da450e5ff6da9f4537d30.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/89894c0e83b669613139a58b319bfa995fb2fb98339da450e5ff6da9f4537d30.jpg)

![e1ab2032e4f2464558b148253cab43c95191a453553e22ebd858a3f63be20b10.jpg](../icml_results/3842_SCNS_%20Continual%20Personalization%20of%20Diffusion%20Models%20via%20Submodular%20Concept%20Neuron%20Selection/tables/e1ab2032e4f2464558b148253cab43c95191a453553e22ebd858a3f63be20b10.jpg)

<a id="paper-3803"></a>

## Towards Achieving Optimal Strong Regret and Constraint Violation via Computationally Efficient Model-free RL


### Images

![4af1247c411f3395463c7b011bba4fbe271d4c1991bd2937ee3aa49ca74eb774.jpg](../icml_results/3843_Towards%20Achieving%20Optimal%20Strong%20Regret%20and%20Constraint%20Violation%20via%20Computationally%20Efficient%20Model/images/4af1247c411f3395463c7b011bba4fbe271d4c1991bd2937ee3aa49ca74eb774.jpg)

### Tables

![2614ab31566d8e2bbaaf83f9de7c7867343f610e130449ccaf0ed7b77dbaa836.jpg](../icml_results/3843_Towards%20Achieving%20Optimal%20Strong%20Regret%20and%20Constraint%20Violation%20via%20Computationally%20Efficient%20Model/tables/2614ab31566d8e2bbaaf83f9de7c7867343f610e130449ccaf0ed7b77dbaa836.jpg)

![84b0bd87b14bc928a803e73ee6b311b795aa68d0855e33a5cb6ff89ab8b9bd89.jpg](../icml_results/3843_Towards%20Achieving%20Optimal%20Strong%20Regret%20and%20Constraint%20Violation%20via%20Computationally%20Efficient%20Model/tables/84b0bd87b14bc928a803e73ee6b311b795aa68d0855e33a5cb6ff89ab8b9bd89.jpg)

<a id="paper-3804"></a>

## Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs


### Images

![03600c6867b256fda6ec6a4cafcff9c2213cc381c0cddbe47a419ffc74154650.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/03600c6867b256fda6ec6a4cafcff9c2213cc381c0cddbe47a419ffc74154650.jpg)

![17bc6216722202cefec41d91eb98832587dd160b5c228ce5c0a2e0794360124d.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/17bc6216722202cefec41d91eb98832587dd160b5c228ce5c0a2e0794360124d.jpg)

![50d4f4ee96dcb5cd1312d8dbaa6c3116b6ec743ec7f399e4b7ae43b3a699354b.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/50d4f4ee96dcb5cd1312d8dbaa6c3116b6ec743ec7f399e4b7ae43b3a699354b.jpg)

![66be5ad62503ba0887af6df06ecf8b2b41c55fb9154fd1c10fb618aec0b076e5.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/66be5ad62503ba0887af6df06ecf8b2b41c55fb9154fd1c10fb618aec0b076e5.jpg)

![8d60d8bbbe0b734ff5cd40bb4948d49e9c209da13a81824b98d862cf60aebc83.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/8d60d8bbbe0b734ff5cd40bb4948d49e9c209da13a81824b98d862cf60aebc83.jpg)

![941171df8da6397a3e36b27b8cad6daa22e79610e2e0874de5c94eac0e704b7a.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/941171df8da6397a3e36b27b8cad6daa22e79610e2e0874de5c94eac0e704b7a.jpg)

![9601e44d986dfaacd033ab805e62f9678070633617f8050dae7ebcbedbef9b29.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/9601e44d986dfaacd033ab805e62f9678070633617f8050dae7ebcbedbef9b29.jpg)

![98a301b98277282890108f3d106de01ceff51bc15c9e2ac914d1886cd45cb5fa.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/98a301b98277282890108f3d106de01ceff51bc15c9e2ac914d1886cd45cb5fa.jpg)

![aed2f8444ff130b7abeb3f11120240e86c4315afaeb0bae350980369435ddff1.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/aed2f8444ff130b7abeb3f11120240e86c4315afaeb0bae350980369435ddff1.jpg)

![b1a5a38b31ce0fdbaa0a5a01325edea4e179ed87e17b0efb4224723378be49a0.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/b1a5a38b31ce0fdbaa0a5a01325edea4e179ed87e17b0efb4224723378be49a0.jpg)

![b1f24364480ff2c182e115a6cd117fbc300ad4864284da30fbb3fa57adf5a99b.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/b1f24364480ff2c182e115a6cd117fbc300ad4864284da30fbb3fa57adf5a99b.jpg)

![be189250b9e2648da3510f6e57a6e384245974f3e9cc819a283d5a5a31a9dd74.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/be189250b9e2648da3510f6e57a6e384245974f3e9cc819a283d5a5a31a9dd74.jpg)

![c5802920ca7354749bd4e7bf10e921533aeb438b18e282e8264885293389e363.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/c5802920ca7354749bd4e7bf10e921533aeb438b18e282e8264885293389e363.jpg)

![c6f903b943be9d7027b5c5bff85270ca8077f6952402d92da9a860bb8250cd50.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/c6f903b943be9d7027b5c5bff85270ca8077f6952402d92da9a860bb8250cd50.jpg)

![de93ba854cb8b3bd16e134365c30e2df0ec5c9db4c5d4c0dcc5b0e0ba2e47c63.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/de93ba854cb8b3bd16e134365c30e2df0ec5c9db4c5d4c0dcc5b0e0ba2e47c63.jpg)

![eed0b35cc65662fae8c35760108878f1fd70ad38f01d6caf5d43804fa143f557.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/eed0b35cc65662fae8c35760108878f1fd70ad38f01d6caf5d43804fa143f557.jpg)

![faf96cfddc1b4d4337ec93d5ef0b3aa10048cbd50135898ff40c35c56121ba3c.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/images/faf96cfddc1b4d4337ec93d5ef0b3aa10048cbd50135898ff40c35c56121ba3c.jpg)

### Tables

![0881c28c85c5e7e54e3daad08f3d174396e17036428481ea10907c6149335bf5.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/0881c28c85c5e7e54e3daad08f3d174396e17036428481ea10907c6149335bf5.jpg)

![105a1bac866e6265ba181c1510484ad71d27db2d5c838d51ab21d076fd769868.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/105a1bac866e6265ba181c1510484ad71d27db2d5c838d51ab21d076fd769868.jpg)

![138be200bebe77365c7a1e14729867ecf4665c363a957251216d271a95a0576e.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/138be200bebe77365c7a1e14729867ecf4665c363a957251216d271a95a0576e.jpg)

![30482a60d7bbcf02a895d89b90dd3964927d61deb6a3c18f78d5d36e574d33ca.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/30482a60d7bbcf02a895d89b90dd3964927d61deb6a3c18f78d5d36e574d33ca.jpg)

![3dd815ff60cf76d140c371352e6faf91ae8a95931c2af884671a2b6af0bac758.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/3dd815ff60cf76d140c371352e6faf91ae8a95931c2af884671a2b6af0bac758.jpg)

![3f8031f76a00f91148ab93bcc78d880937cb93c6bc893089b8aa036713f660cc.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/3f8031f76a00f91148ab93bcc78d880937cb93c6bc893089b8aa036713f660cc.jpg)

![463737f5dae44f775cb7b674fcb33cca59f235fc790bf937093b895940725066.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/463737f5dae44f775cb7b674fcb33cca59f235fc790bf937093b895940725066.jpg)

![55e8f1475b93b2ab325606fee9ad6cdc243679adb0783b9a86b88192f8a9e116.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/55e8f1475b93b2ab325606fee9ad6cdc243679adb0783b9a86b88192f8a9e116.jpg)

![64da069c3b381fe77fa5b64f35b1871b8b12398e2bcae340b1c1122e75907df3.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/64da069c3b381fe77fa5b64f35b1871b8b12398e2bcae340b1c1122e75907df3.jpg)

![731562d42d7efdfb2c0f439b46dc394a9c4e75d1bc4ea07982d32b86e465dd49.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/731562d42d7efdfb2c0f439b46dc394a9c4e75d1bc4ea07982d32b86e465dd49.jpg)

![7ca1d8b82c251b5df92662c48affef17565b839706a5401ef9732ca99ef4d3c8.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/7ca1d8b82c251b5df92662c48affef17565b839706a5401ef9732ca99ef4d3c8.jpg)

![883dd7fc10927c66b2e7e7ba5f8e822f8e5d0c116169ee93ecf0e4de17150de4.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/883dd7fc10927c66b2e7e7ba5f8e822f8e5d0c116169ee93ecf0e4de17150de4.jpg)

![90faaa089bb092c5573dfb5b36dd0b8d18d1d180e185ee32cf2231fc83cc442c.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/90faaa089bb092c5573dfb5b36dd0b8d18d1d180e185ee32cf2231fc83cc442c.jpg)

![adae0ec3d42a0703efab888b49edacafd9ea3a083a205d72b8bf470b019c51c3.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/adae0ec3d42a0703efab888b49edacafd9ea3a083a205d72b8bf470b019c51c3.jpg)

![be9c3fe6e0cd1833b5107644d8db6a89f942d5f03db572f703c869697459b6aa.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/be9c3fe6e0cd1833b5107644d8db6a89f942d5f03db572f703c869697459b6aa.jpg)

![d0499a5ba705acfd3489d9827266f34395c557a6020d92ad28704f0af19b1d60.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/d0499a5ba705acfd3489d9827266f34395c557a6020d92ad28704f0af19b1d60.jpg)

![e0c895c1f4c1220375bc77b70217960d78292fd39b097df5e3829d5aef81956b.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/e0c895c1f4c1220375bc77b70217960d78292fd39b097df5e3829d5aef81956b.jpg)

![e7d1a685e2068fc3deac1177e41e4839bb0dccd95aef0d9864c7927182ef41af.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/e7d1a685e2068fc3deac1177e41e4839bb0dccd95aef0d9864c7927182ef41af.jpg)

![ee522c64f99fb74e1ef0b9896c1b48cb6ba9e7e91835a2b0a53b73505eca323d.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/ee522c64f99fb74e1ef0b9896c1b48cb6ba9e7e91835a2b0a53b73505eca323d.jpg)

![f48fd6ba82445fbb4c7dfb2c56dcd99323dae42e1ec01e09704597b19279d9d4.jpg](../icml_results/3844_Learning%20Self-Interpretation%20from%20Interpretability%20Artifacts_%20Training%20Lightweight%20Adapters%20on%20Vecto/tables/f48fd6ba82445fbb4c7dfb2c56dcd99323dae42e1ec01e09704597b19279d9d4.jpg)

<a id="paper-3805"></a>

## Online Bayesian Experimental Design for Partially Observed Dynamical Systems


### Images

![4ce745b83ab51fd90e13b0f129485827b2ef056a5fa183e031de8440ed286b85.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/4ce745b83ab51fd90e13b0f129485827b2ef056a5fa183e031de8440ed286b85.jpg)

![52b56c8244aa2930918c5792a783ab644f2f0942346febe471ac2159bf7eb904.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/52b56c8244aa2930918c5792a783ab644f2f0942346febe471ac2159bf7eb904.jpg)

![630bf2f7a80b7cf7937bb911a29a146f9325ccf271d0372dd5602faa2fa22f3c.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/630bf2f7a80b7cf7937bb911a29a146f9325ccf271d0372dd5602faa2fa22f3c.jpg)

![6c5e4409acefa15187888bc7517c44705b200a6191df395f2c7863d686e0f4e9.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/6c5e4409acefa15187888bc7517c44705b200a6191df395f2c7863d686e0f4e9.jpg)

![7c9ed28b0abdcf35b1ca5ca06032fd654276a397fa2f5485bbda5da77bbfe7b4.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/7c9ed28b0abdcf35b1ca5ca06032fd654276a397fa2f5485bbda5da77bbfe7b4.jpg)

![c103d73d1d59b60fa89a20ab38767948f0603a45aac42452e0a665abca3b9043.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/c103d73d1d59b60fa89a20ab38767948f0603a45aac42452e0a665abca3b9043.jpg)

![f63eb693087ce18ee3cabd4a2a7632ac4dd4c924d4364e5673c3f6ac601c8243.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/images/f63eb693087ce18ee3cabd4a2a7632ac4dd4c924d4364e5673c3f6ac601c8243.jpg)

### Tables

![14dc879f515f94a27055966d8da59bcf03e5edf6f57af328462afefcf6b69155.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/14dc879f515f94a27055966d8da59bcf03e5edf6f57af328462afefcf6b69155.jpg)

![2265343cc351ae6a5b5b2aad34d396c6fa2a8aeb484298fe569b1ae4927f45c6.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/2265343cc351ae6a5b5b2aad34d396c6fa2a8aeb484298fe569b1ae4927f45c6.jpg)

![6a91a78da91a31a8e5b3742a8724055272c93fa35bb710f2c1f6254d33b6d584.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/6a91a78da91a31a8e5b3742a8724055272c93fa35bb710f2c1f6254d33b6d584.jpg)

![860641793da384ad519da4c6dcbd29d6c1e08a5974702688b583773de9ef44be.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/860641793da384ad519da4c6dcbd29d6c1e08a5974702688b583773de9ef44be.jpg)

![9f5a7d0bb17ca889d1bdb25ee1d18625e5d354a5efe1c98a7def5e05c6259f86.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/9f5a7d0bb17ca889d1bdb25ee1d18625e5d354a5efe1c98a7def5e05c6259f86.jpg)

![b8bc38e5c919b6333657700f0e39ccb89cc44d9d925c54516062809466229066.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/b8bc38e5c919b6333657700f0e39ccb89cc44d9d925c54516062809466229066.jpg)

![cc4e86df9ed16f92357d3515ae8771e0f5825f27211cfb917953deb269fdffc7.jpg](../icml_results/3845_Online%20Bayesian%20Experimental%20Design%20for%20Partially%20Observed%20Dynamical%20Systems/tables/cc4e86df9ed16f92357d3515ae8771e0f5825f27211cfb917953deb269fdffc7.jpg)

<a id="paper-3806"></a>

## Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization


### Images

![056e3a67069f8cc97e99298edae5178a5a19ddf472780b2b714a13543a2dbe3b.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/056e3a67069f8cc97e99298edae5178a5a19ddf472780b2b714a13543a2dbe3b.jpg)

![10bfe201c1ee301be8d69845c3a5c1cb92a23bde3cd29600cbff57a6f052d0f9.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/10bfe201c1ee301be8d69845c3a5c1cb92a23bde3cd29600cbff57a6f052d0f9.jpg)

![288ce81fd70a758c2462e640916e8e5dc4e75d45103167729cd337fe6151332a.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/288ce81fd70a758c2462e640916e8e5dc4e75d45103167729cd337fe6151332a.jpg)

![32915e63d05c6f3fc24728dbf55328f496882cf695a650a77f30da0f873b7795.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/32915e63d05c6f3fc24728dbf55328f496882cf695a650a77f30da0f873b7795.jpg)

![40c2cfdc3f26d377e5e23c2128ed4c287624e253cf521b3fa1bb7a381dffc304.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/40c2cfdc3f26d377e5e23c2128ed4c287624e253cf521b3fa1bb7a381dffc304.jpg)

![43320b0ca1ea39e7bdbe49e37f8af53ff51ebe89a44ba081002aad2277c2b79a.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/43320b0ca1ea39e7bdbe49e37f8af53ff51ebe89a44ba081002aad2277c2b79a.jpg)

![51979f41f33f197538c2220b44bedbe48519b2dbfbddd1ee2ae24933b2890b6b.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/51979f41f33f197538c2220b44bedbe48519b2dbfbddd1ee2ae24933b2890b6b.jpg)

![73aa9a5a168342214d89296e81dd7996c34bb8875b41d8140075130c222d5be2.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/73aa9a5a168342214d89296e81dd7996c34bb8875b41d8140075130c222d5be2.jpg)

![79374a4eb1f1721a32fa5c733f5a97e5c34f74792c4355c143d64dc0b2f107f1.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/79374a4eb1f1721a32fa5c733f5a97e5c34f74792c4355c143d64dc0b2f107f1.jpg)

![dedfb87738d20016529bb9007ade48f6da36ef5a58b4faa4a84232bc6143474a.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/dedfb87738d20016529bb9007ade48f6da36ef5a58b4faa4a84232bc6143474a.jpg)

![fcb36e1594e746cbbc94a0e2c22c8e5fd4b1b5735c8eea114750054dadc1f685.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/images/fcb36e1594e746cbbc94a0e2c22c8e5fd4b1b5735c8eea114750054dadc1f685.jpg)

### Tables

![0da13f6b9bf71af1667194f2f8e2787f522c5b54f11b1cae738e15e9c029a0dc.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/0da13f6b9bf71af1667194f2f8e2787f522c5b54f11b1cae738e15e9c029a0dc.jpg)

![47c86eb1382b301822ace76eb5754d72e22b9b7de1a285e19d4ebdc93b2da8fa.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/47c86eb1382b301822ace76eb5754d72e22b9b7de1a285e19d4ebdc93b2da8fa.jpg)

![48636f47088c8a0d224e68f4628b4421b189f9ea8b6c69d41b9f7ff7bb33cb97.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/48636f47088c8a0d224e68f4628b4421b189f9ea8b6c69d41b9f7ff7bb33cb97.jpg)

![578eb976e55e53f00d35222fa7bb26ef42d09c2dc6091bf5d7479152f8b2138c.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/578eb976e55e53f00d35222fa7bb26ef42d09c2dc6091bf5d7479152f8b2138c.jpg)

![bfcb70297354a7271ee44f0164a2bdd5ac79387ac42b1e15370d5e198432508a.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/bfcb70297354a7271ee44f0164a2bdd5ac79387ac42b1e15370d5e198432508a.jpg)

![c3d6b8ba55067ae2016832548ecae7dfc9d29fe519ceadb05ee41905a4d4e3af.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/c3d6b8ba55067ae2016832548ecae7dfc9d29fe519ceadb05ee41905a4d4e3af.jpg)

![de95c51a0631d4aabcc2d712724e611f5d8674878d16a717f2cfc545e6802d96.jpg](../icml_results/3846_Inverse%20Entropic%20Optimal%20Transport%20Solves%20Semi-supervised%20Learning%20via%20Data%20Likelihood%20Maximization/tables/de95c51a0631d4aabcc2d712724e611f5d8674878d16a717f2cfc545e6802d96.jpg)

<a id="paper-3807"></a>

## Generalizing Stochastic Smoothing for Differentiation and Gradient Estimation


### Images

![0ac6fec8e6e802a106ac501453423d4c07d5e4f9bc2904b079fac84942c6828a.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/0ac6fec8e6e802a106ac501453423d4c07d5e4f9bc2904b079fac84942c6828a.jpg)

![153072aadd7631db4e5fd51cad471d85db5e19d5cee40988bffb6fda719b46de.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/153072aadd7631db4e5fd51cad471d85db5e19d5cee40988bffb6fda719b46de.jpg)

![25668e8ffb6c5d78746be4b50d40b74600f6f2efdeb557aa26637d3a5c29a504.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/25668e8ffb6c5d78746be4b50d40b74600f6f2efdeb557aa26637d3a5c29a504.jpg)

![3139b54968523ffd295f2c3ccf9f7076152ed59838e97f5a2eda854dfb3d1ce7.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/3139b54968523ffd295f2c3ccf9f7076152ed59838e97f5a2eda854dfb3d1ce7.jpg)

![40fd6cf71f68adb8260a8c80d80b0c48ce43ca37f7b603f0080dfaac3ba5fe20.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/40fd6cf71f68adb8260a8c80d80b0c48ce43ca37f7b603f0080dfaac3ba5fe20.jpg)

![47061d8fcc5dd04ee63fa398f47ffb3405d00de82e1f57a95c78eafccad4f89f.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/47061d8fcc5dd04ee63fa398f47ffb3405d00de82e1f57a95c78eafccad4f89f.jpg)

![4b8f867a51fb8f89aaa4c19f55ada1da7842237e24266f0ac37a5e7074970d76.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/4b8f867a51fb8f89aaa4c19f55ada1da7842237e24266f0ac37a5e7074970d76.jpg)

![533bc104913506f66279f2012dbf978263d8e7689c90c2ff3a27d5c95763592f.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/533bc104913506f66279f2012dbf978263d8e7689c90c2ff3a27d5c95763592f.jpg)

![637253e85e60aa401e3ee5eb9d5de69d5459edb0f94493545097da73d1768297.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/637253e85e60aa401e3ee5eb9d5de69d5459edb0f94493545097da73d1768297.jpg)

![6c781fa300d1e69739812c8cebfe25d6cd8a67ddd995242727b315be22db6baf.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/6c781fa300d1e69739812c8cebfe25d6cd8a67ddd995242727b315be22db6baf.jpg)

![80e30682a2f1f44ca37b2454827953309611649e07fbe878e9a205532c549f52.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/80e30682a2f1f44ca37b2454827953309611649e07fbe878e9a205532c549f52.jpg)

![93468b4146d46ec045bebce17c7dce54693c3ef22570c91e04e914fd2a5e58d4.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/93468b4146d46ec045bebce17c7dce54693c3ef22570c91e04e914fd2a5e58d4.jpg)

![a06bb3ab2410c25d24057c4ea2fb2333a790a2c71ef664ab17a24a4be455c8eb.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/a06bb3ab2410c25d24057c4ea2fb2333a790a2c71ef664ab17a24a4be455c8eb.jpg)

![a599722292602bd620cb73d355faf5309018de53eaa40b62d0454e1d0255144c.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/a599722292602bd620cb73d355faf5309018de53eaa40b62d0454e1d0255144c.jpg)

![c0b62ad2822b18bf1509846ca67adbca0f1a98a9baa5755014e7dad6a14f51dc.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/c0b62ad2822b18bf1509846ca67adbca0f1a98a9baa5755014e7dad6a14f51dc.jpg)

![c5daaff43765a2f544275e1ed020832f6aa30cff9c86e0202cb817a203c8e77d.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/c5daaff43765a2f544275e1ed020832f6aa30cff9c86e0202cb817a203c8e77d.jpg)

![cb8a1205828efe801c8e75dc91b3e4397d6107cb7e8e1f29e85f394f93ac9af8.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/cb8a1205828efe801c8e75dc91b3e4397d6107cb7e8e1f29e85f394f93ac9af8.jpg)

![d1f5549d1b1e21bf36c6c7a8e9e9ead55a7370d28173fe0234b0228bfe21667a.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/d1f5549d1b1e21bf36c6c7a8e9e9ead55a7370d28173fe0234b0228bfe21667a.jpg)

![d94f1acb50d81af590c33b01508ff4a0434d88dff9a79bb4056d0e4f9ffe7fd1.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/d94f1acb50d81af590c33b01508ff4a0434d88dff9a79bb4056d0e4f9ffe7fd1.jpg)

![f8ea050e54b3e863159fd091cb8e28316912fe4052be35133e09a8a3ec297cb3.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/f8ea050e54b3e863159fd091cb8e28316912fe4052be35133e09a8a3ec297cb3.jpg)

![fd038c9c8aefa51a1ab942bc0d08148fd02f3fa81c80ea40474d28f81963d3ca.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/images/fd038c9c8aefa51a1ab942bc0d08148fd02f3fa81c80ea40474d28f81963d3ca.jpg)

### Tables

![382f2a1f8d9cd9658b7568d1d31c77efa95b43f5109af0639f6373efd1a0ecee.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/382f2a1f8d9cd9658b7568d1d31c77efa95b43f5109af0639f6373efd1a0ecee.jpg)

![4dd0d44490fbdf645753bff29902a6c3579ba2106dedf1204ef6fb637e59589e.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/4dd0d44490fbdf645753bff29902a6c3579ba2106dedf1204ef6fb637e59589e.jpg)

![4df055227890342555e0006bbb6a69896877eb5001d9f757a600614173fda935.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/4df055227890342555e0006bbb6a69896877eb5001d9f757a600614173fda935.jpg)

![4ff1fba35142740ec21bf2e0759ea356724a4e989702dc00364d10a6dce7c86a.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/4ff1fba35142740ec21bf2e0759ea356724a4e989702dc00364d10a6dce7c86a.jpg)

![5d3b44178011c96f7dd1bf8e789f6e16b334a3e2acf2134c47c15cccea3dcee4.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/5d3b44178011c96f7dd1bf8e789f6e16b334a3e2acf2134c47c15cccea3dcee4.jpg)

![6c78bb481cf8d59b878ca56237fadfcf29e4ee610bdb06c4b7c5504e12780a21.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/6c78bb481cf8d59b878ca56237fadfcf29e4ee610bdb06c4b7c5504e12780a21.jpg)

![8d66963c775685b7d08a4e3640f9a70d01817d353cc5dc77c3e9247b94e7cd5d.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/8d66963c775685b7d08a4e3640f9a70d01817d353cc5dc77c3e9247b94e7cd5d.jpg)

![9228b9c01c6a893cd465f8bac8a27630e966ed0df8a3e149e70247284b2d5b40.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/9228b9c01c6a893cd465f8bac8a27630e966ed0df8a3e149e70247284b2d5b40.jpg)

![9e8997ead2590fb916b721c9d42460536000fed2cac828643ae57cb618091e3a.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/9e8997ead2590fb916b721c9d42460536000fed2cac828643ae57cb618091e3a.jpg)

![be73dff630cbafc00b0385b326d4b3972daf8537e94bedea314ff56514cf9593.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/be73dff630cbafc00b0385b326d4b3972daf8537e94bedea314ff56514cf9593.jpg)

![c90959564717d26e181ddae49c8d035a4f4a527b20a313b3429d152677905c3a.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/c90959564717d26e181ddae49c8d035a4f4a527b20a313b3429d152677905c3a.jpg)

![e215a6a819360665a730b0901e8c48570db5031d0b3887dd91410e955913bd5b.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/e215a6a819360665a730b0901e8c48570db5031d0b3887dd91410e955913bd5b.jpg)

![e4d86a99609e78a2f8a2d9a7ab8bc7b6d50461efe7d5d64379f7f2bf9ba8c7be.jpg](../icml_results/3847_Generalizing%20Stochastic%20Smoothing%20for%20Differentiation%20and%20Gradient%20Estimation/tables/e4d86a99609e78a2f8a2d9a7ab8bc7b6d50461efe7d5d64379f7f2bf9ba8c7be.jpg)

<a id="paper-3808"></a>

## Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping


### Images

![662941b5e811fdc235ef6e93fa5e1c7a993aafff5b489929c9721b6caf563ec0.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/images/662941b5e811fdc235ef6e93fa5e1c7a993aafff5b489929c9721b6caf563ec0.jpg)

![bb5f0172c70cc746e6b62ceb957fd843c9c45c386f1262be0fdb2d5f86b17812.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/images/bb5f0172c70cc746e6b62ceb957fd843c9c45c386f1262be0fdb2d5f86b17812.jpg)

![de460a543aa7ad6077069951355622f2e3486d69e781a6d10e0d2aa3905281ef.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/images/de460a543aa7ad6077069951355622f2e3486d69e781a6d10e0d2aa3905281ef.jpg)

### Tables

![3feda79324d09052c165d8599a1d7625993b84553c27a4967d69a51a1a640d88.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/tables/3feda79324d09052c165d8599a1d7625993b84553c27a4967d69a51a1a640d88.jpg)

![81f3f626f974d5a8ec9f15b58a1720097c7d2fd741c93e2f2a1249e355bbea84.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/tables/81f3f626f974d5a8ec9f15b58a1720097c7d2fd741c93e2f2a1249e355bbea84.jpg)

![d4412ee36fe7629f0560276dd8be0358b602ba3ba171d18c45a1ec4caf43a944.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/tables/d4412ee36fe7629f0560276dd8be0358b602ba3ba171d18c45a1ec4caf43a944.jpg)

![d8d5b933567d7b7a8186f3c06b034d59ded67f84045ba5f25a55eb7e6ef36b2f.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/tables/d8d5b933567d7b7a8186f3c06b034d59ded67f84045ba5f25a55eb7e6ef36b2f.jpg)

![fea554a22df2ff21fcbe851f425564e1491cba070a38f658b2a2492201798bfc.jpg](../icml_results/3848_Memory%20as%20a%20Markov%20Matrix_%20Sample%20Efficient%20Knowledge%20Expansion%20via%20Token-to-Dictionary%20Mapping/tables/fea554a22df2ff21fcbe851f425564e1491cba070a38f658b2a2492201798bfc.jpg)

<a id="paper-3809"></a>

## Geometric Entropy and Retrieval Phase Transitions in Continuous Dense Associative Memory


### Images

![2e700c94f057db0ee290c8f9f8a84c3d9cdb44b76428a1ef69be0e1d03f60d33.jpg](../icml_results/3849_Geometric%20Entropy%20and%20Retrieval%20Phase%20Transitions%20in%20Continuous%20Dense%20Associative%20Memory/images/2e700c94f057db0ee290c8f9f8a84c3d9cdb44b76428a1ef69be0e1d03f60d33.jpg)

![fb7021f3c793ecb76c07228294919f78c928d80656ce4c1a9fced028a786ce85.jpg](../icml_results/3849_Geometric%20Entropy%20and%20Retrieval%20Phase%20Transitions%20in%20Continuous%20Dense%20Associative%20Memory/images/fb7021f3c793ecb76c07228294919f78c928d80656ce4c1a9fced028a786ce85.jpg)

<a id="paper-3810"></a>

## Decomposing Out-of-Distribution Error in Conditional Flow Matching via Wasserstein Geometry


### Images

![4d96f0699c17e813063bc027f6bc8b0fe9acef5b417cd8fb0233f014452d1afa.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/images/4d96f0699c17e813063bc027f6bc8b0fe9acef5b417cd8fb0233f014452d1afa.jpg)

![57fb67ba80f20ec6147b8758d7b7326bec75adc6d7d86e73f553b8e0ef8d28a0.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/images/57fb67ba80f20ec6147b8758d7b7326bec75adc6d7d86e73f553b8e0ef8d28a0.jpg)

![8110e12e1831629f49dc9719a9d4203bea345c1983174b7ac0b9559c55a6ed17.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/images/8110e12e1831629f49dc9719a9d4203bea345c1983174b7ac0b9559c55a6ed17.jpg)

![cdf25ed85b0a99f68ccb6fa0f87e5c2209c145edec53f7b68739fc674f79a15b.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/images/cdf25ed85b0a99f68ccb6fa0f87e5c2209c145edec53f7b68739fc674f79a15b.jpg)

![f0e229b644618cf086e630f1435560d28394693b58e4954e45aec0c9a90ae9fa.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/images/f0e229b644618cf086e630f1435560d28394693b58e4954e45aec0c9a90ae9fa.jpg)

### Tables

![0bd3b20adc30df30bc3bf31eea19e41f61e7f2fea4c02e0b8949f93d52f1ce8a.jpg](../icml_results/3850_Decomposing%20Out-of-Distribution%20Error%20in%20Conditional%20Flow%20Matching%20via%20Wasserstein%20Geometry/tables/0bd3b20adc30df30bc3bf31eea19e41f61e7f2fea4c02e0b8949f93d52f1ce8a.jpg)

<a id="paper-3811"></a>

## Revisiting Distribution Correction Estimation for Offline Imitation Learning with Suboptimal Dataset


### Images

![1c28794fa6851a9e5d7e078bb06dac5e988e984fcf69fcaf0015374ab1a491ae.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/images/1c28794fa6851a9e5d7e078bb06dac5e988e984fcf69fcaf0015374ab1a491ae.jpg)

![3c17a88c0dce7a4f2e6448c82fe68a6cfdfba9db5e1d9dc5f3ecd98a087e1b4b.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/images/3c17a88c0dce7a4f2e6448c82fe68a6cfdfba9db5e1d9dc5f3ecd98a087e1b4b.jpg)

![425603e22a2ede1afab578bacbb3d31e8b64941877defac63038dcc16e32015a.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/images/425603e22a2ede1afab578bacbb3d31e8b64941877defac63038dcc16e32015a.jpg)

![ad2c0bdd2eba9442bf25914ab0f6cde5cef49010c82060e68337bc09243eb26c.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/images/ad2c0bdd2eba9442bf25914ab0f6cde5cef49010c82060e68337bc09243eb26c.jpg)

![e838505d2b75785afb0c61d59608ecd767a32fea67b582761bb0525ba0c6a778.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/images/e838505d2b75785afb0c61d59608ecd767a32fea67b582761bb0525ba0c6a778.jpg)

### Tables

![013718df2fc5d2f6f512bde70acfe3b391f4720e8f58653df00db74521191bde.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/013718df2fc5d2f6f512bde70acfe3b391f4720e8f58653df00db74521191bde.jpg)

![1150cbb75264a8000ee3e164653e36e6b5c4c10ce7d4e1df119503f977405681.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/1150cbb75264a8000ee3e164653e36e6b5c4c10ce7d4e1df119503f977405681.jpg)

![1a84c191454e27af76e9a9e5820ca972bdc119fa7f41e5c3725eb74272661512.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/1a84c191454e27af76e9a9e5820ca972bdc119fa7f41e5c3725eb74272661512.jpg)

![29ca18be0ca9f888128d9a327d1cb195bb30fe14a3079a48e218a7f08563fddb.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/29ca18be0ca9f888128d9a327d1cb195bb30fe14a3079a48e218a7f08563fddb.jpg)

![2b37ec45fc20d996c9446bf440aa3c7569b453e8fca34a486f76ae0bdc03307a.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/2b37ec45fc20d996c9446bf440aa3c7569b453e8fca34a486f76ae0bdc03307a.jpg)

![79e276e3113196a0e52c9ffbd92fe236d433d4d2c794b867f23e0fddc406e545.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/79e276e3113196a0e52c9ffbd92fe236d433d4d2c794b867f23e0fddc406e545.jpg)

![7f0c263a478cbbddbc9a96f025a04553027e3eef988b30d860c006a87de04bd9.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/7f0c263a478cbbddbc9a96f025a04553027e3eef988b30d860c006a87de04bd9.jpg)

![c06140e436e807f676758a9d5de318157f554d025bedabe69ac38b800c376e1b.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/c06140e436e807f676758a9d5de318157f554d025bedabe69ac38b800c376e1b.jpg)

![d3074b383086ca69aaa94499d8fa7b2611121070426427c3bece67b5208ca97a.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/d3074b383086ca69aaa94499d8fa7b2611121070426427c3bece67b5208ca97a.jpg)

![f547b1a240d530a99c9c49b6279217145948392e8ce730b569408e40b8994b3a.jpg](../icml_results/3851_Revisiting%20Distribution%20Correction%20Estimation%20for%20Offline%20Imitation%20Learning%20with%20Suboptimal%20Dataset/tables/f547b1a240d530a99c9c49b6279217145948392e8ce730b569408e40b8994b3a.jpg)

<a id="paper-3812"></a>

## Pruning at Initialisation through the lens of Graphon Limit: Convergence, Expressivity, and Generalisation


### Images

![0bc1dc93d59f10fcbfebf26849f926d20120fde881aabe42caa08833fedec867.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/0bc1dc93d59f10fcbfebf26849f926d20120fde881aabe42caa08833fedec867.jpg)

![4448c1458cf75fa85e8a917d8dcce94ea047fb2953338d62e991f32a0fb4d5f5.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/4448c1458cf75fa85e8a917d8dcce94ea047fb2953338d62e991f32a0fb4d5f5.jpg)

![69996bdb7129cfbc7dc709d7f1f80a836e1c33d0403b06be2b314f7f5fcc4d55.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/69996bdb7129cfbc7dc709d7f1f80a836e1c33d0403b06be2b314f7f5fcc4d55.jpg)

![98b6b6d4021a59fb9653874335529ae1ef5f55574d574508cf14a391cc542b9f.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/98b6b6d4021a59fb9653874335529ae1ef5f55574d574508cf14a391cc542b9f.jpg)

![9dd98ca9d6a4fb473f8e585d35300b5ee3c5a93e5aab93696af132da2e82ad42.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/9dd98ca9d6a4fb473f8e585d35300b5ee3c5a93e5aab93696af132da2e82ad42.jpg)

![a1db28115c53081adf2e4f90c0a95d23d34858e6b50b9008ab4cfa44f2891aa8.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/a1db28115c53081adf2e4f90c0a95d23d34858e6b50b9008ab4cfa44f2891aa8.jpg)

![a60b2b13c15b2bd623497968deea03d60e37182d18140336cb45adc88d044e20.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/a60b2b13c15b2bd623497968deea03d60e37182d18140336cb45adc88d044e20.jpg)

![b242e8800f27ef6b84833b5d2bbca33c4a71205493bd7437be479092ab2459b0.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/b242e8800f27ef6b84833b5d2bbca33c4a71205493bd7437be479092ab2459b0.jpg)

![bd7c07fbe4b588b85dfe9f91bf5a78fa1e79dc2fdb9d55e0c914e9db60260352.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/bd7c07fbe4b588b85dfe9f91bf5a78fa1e79dc2fdb9d55e0c914e9db60260352.jpg)

![d11afb20aa094f220b31b802b045aeb7b8c55a0d63bbdace6d75d7d58697a79c.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/d11afb20aa094f220b31b802b045aeb7b8c55a0d63bbdace6d75d7d58697a79c.jpg)

![f5fb22e95e8a33c506ef3502ea5b3a70fcb464185f45229e60c44184f9c446c4.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/images/f5fb22e95e8a33c506ef3502ea5b3a70fcb464185f45229e60c44184f9c446c4.jpg)

### Tables

![0f3a3dd15b7d99523fe9bbade8ffac0b7053e8c04f80b3055da34a51c31864b5.jpg](../icml_results/3852_Pruning%20at%20Initialisation%20through%20the%20lens%20of%20Graphon%20Limit_%20Convergence%2C%20Expressivity%2C%20and%20Generali/tables/0f3a3dd15b7d99523fe9bbade8ffac0b7053e8c04f80b3055da34a51c31864b5.jpg)

<a id="paper-3813"></a>

## Automatic Unsupervised Ensemble Outlier Model Selection


### Images

![4afe5d172bc4514c3298a3752ff4e93da3d921cc65984d81ebbfe958f5f910c9.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/images/4afe5d172bc4514c3298a3752ff4e93da3d921cc65984d81ebbfe958f5f910c9.jpg)

![5a28868dba0310791a899dd7a2d5d7a486618edb140fab9853983960e7993878.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/images/5a28868dba0310791a899dd7a2d5d7a486618edb140fab9853983960e7993878.jpg)

![5a60710b2c6bf393e8b14aeaa8763931adc2870feaa38f3cfe35fff016403fbe.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/images/5a60710b2c6bf393e8b14aeaa8763931adc2870feaa38f3cfe35fff016403fbe.jpg)

![d71f7f1c044efb5d533bae154a7cddf0087c0523ef03139cdcd6f2eb30674f59.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/images/d71f7f1c044efb5d533bae154a7cddf0087c0523ef03139cdcd6f2eb30674f59.jpg)

![e749dc849ed0d8f442c62ab2085984569421e0663f82d139fd855775303ae023.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/images/e749dc849ed0d8f442c62ab2085984569421e0663f82d139fd855775303ae023.jpg)

### Tables

![0571e48c200cb4d28b4179f74b4031b9249546825be614bd4d5037ef3b0b64fe.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/0571e48c200cb4d28b4179f74b4031b9249546825be614bd4d5037ef3b0b64fe.jpg)

![0de33932e8d6e0758a7f191bd69d0456f7248bf74902f4bcacc7af3a3b162c10.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/0de33932e8d6e0758a7f191bd69d0456f7248bf74902f4bcacc7af3a3b162c10.jpg)

![13804a3feaede79c459ebac27d5172a130dff34c2e3e260665e412cbad9012cb.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/13804a3feaede79c459ebac27d5172a130dff34c2e3e260665e412cbad9012cb.jpg)

![1a2c2aa09b97b65fa0de2058fe27d7c100b111274e9d77a33b17f1c8b777096d.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/1a2c2aa09b97b65fa0de2058fe27d7c100b111274e9d77a33b17f1c8b777096d.jpg)

![1a3ba547fbe66861dbfc7eb6d6ede1763914fa338c29a18eccaca91ccedf5295.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/1a3ba547fbe66861dbfc7eb6d6ede1763914fa338c29a18eccaca91ccedf5295.jpg)

![1c2ef393800c8ead2c5d485aba823e66cf3f1be049a22dfe8f9dbb3d6482bb3a.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/1c2ef393800c8ead2c5d485aba823e66cf3f1be049a22dfe8f9dbb3d6482bb3a.jpg)

![30a59a33d96a545468cbdf5731d53c65327e816def6521a7066fd18370bd4b5b.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/30a59a33d96a545468cbdf5731d53c65327e816def6521a7066fd18370bd4b5b.jpg)

![3a6925bc5a15261912409e720d6d762c5e804abc72981ae94a6f9e76196cfa98.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/3a6925bc5a15261912409e720d6d762c5e804abc72981ae94a6f9e76196cfa98.jpg)

![3d0968a7e1ed15c3cadd5d1ebc50a1a82f6e69c6d3995261e77c2e8cd9d0483e.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/3d0968a7e1ed15c3cadd5d1ebc50a1a82f6e69c6d3995261e77c2e8cd9d0483e.jpg)

![3e0a78beca07ca9226719b4f5aeea922a8178f709812293aa512decade5f2a44.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/3e0a78beca07ca9226719b4f5aeea922a8178f709812293aa512decade5f2a44.jpg)

![5cecebd693c7e5b2f3452058d7681442b65731170efb2df1ac4f5bceeab643d6.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/5cecebd693c7e5b2f3452058d7681442b65731170efb2df1ac4f5bceeab643d6.jpg)

![5f50099721e355d5acf7ca79df35879347f82cbf1ca76cca3778162f761b85f2.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/5f50099721e355d5acf7ca79df35879347f82cbf1ca76cca3778162f761b85f2.jpg)

![80f169ccc4d4af4a89e9e62ecb2a94d4aad2b10606e4f3fbcecb5ff96e3d7807.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/80f169ccc4d4af4a89e9e62ecb2a94d4aad2b10606e4f3fbcecb5ff96e3d7807.jpg)

![9f44b01d4448eb3100fd0579c7dbb839e24c98266c3c14b4f120d921d060a905.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/9f44b01d4448eb3100fd0579c7dbb839e24c98266c3c14b4f120d921d060a905.jpg)

![a436cd05675de8b71f991203f9723281c4cda6518142a56944994d899b6ca43a.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/a436cd05675de8b71f991203f9723281c4cda6518142a56944994d899b6ca43a.jpg)

![a4bf17db4043715435eadc7d29c7234dc0848a52441ce2e3f0273857bc5e23e4.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/a4bf17db4043715435eadc7d29c7234dc0848a52441ce2e3f0273857bc5e23e4.jpg)

![a6907a214df099bbc781a4d3532fa7866bf758a44113deaa1457ee7aa360a96c.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/a6907a214df099bbc781a4d3532fa7866bf758a44113deaa1457ee7aa360a96c.jpg)

![ba5adf373e53451ef7e88d636abf8b9baf00f3aa084870f1511573f1bb38bf06.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/ba5adf373e53451ef7e88d636abf8b9baf00f3aa084870f1511573f1bb38bf06.jpg)

![bf2282e5ea5478eb691f3e5e7a7015f60bd166243126cc25e4a62259d075f4d4.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/bf2282e5ea5478eb691f3e5e7a7015f60bd166243126cc25e4a62259d075f4d4.jpg)

![d76033e24351be692a3bfc0f84ea86a6b66d046dd60d64f303097cbc22ed10ba.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/d76033e24351be692a3bfc0f84ea86a6b66d046dd60d64f303097cbc22ed10ba.jpg)

![e80506ab93598a28d5e40dbea435bacafff47e9c9832a876d6d3b5adc85ab151.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/e80506ab93598a28d5e40dbea435bacafff47e9c9832a876d6d3b5adc85ab151.jpg)

![f2e433bd8664d7dc84b2df861328beff8ce23830a7cd9ddbe0e8a098815ce034.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/f2e433bd8664d7dc84b2df861328beff8ce23830a7cd9ddbe0e8a098815ce034.jpg)

![f6d3af16f4dac7005f3c1315c7d6523041851502a780b5ba1a365f5fe5adb3b9.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/f6d3af16f4dac7005f3c1315c7d6523041851502a780b5ba1a365f5fe5adb3b9.jpg)

![f7f246a3eb96a8d5997e2a0e1bbb41a972d3d9fd61a296e100c4b9db6b28a7cd.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/f7f246a3eb96a8d5997e2a0e1bbb41a972d3d9fd61a296e100c4b9db6b28a7cd.jpg)

![f8590fca6e811d7a03c69d743cd0747cf578ba681f0ef602d09c5e61b8572d7f.jpg](../icml_results/3853_Automatic%20Unsupervised%20Ensemble%20Outlier%20Model%20Selection/tables/f8590fca6e811d7a03c69d743cd0747cf578ba681f0ef602d09c5e61b8572d7f.jpg)

<a id="paper-3814"></a>

## Learning Manifold Data with Flow Matching


### Images

![06c7bf14a7e59b9af3c761a6313523f938b9c68def395f3aaf99e7483f2b93e0.jpg](../icml_results/3854_Learning%20Manifold%20Data%20with%20Flow%20Matching/images/06c7bf14a7e59b9af3c761a6313523f938b9c68def395f3aaf99e7483f2b93e0.jpg)

![0bb802be07872a4d722e7c5b87070ae78a71360fab33758016a7fe774884ba49.jpg](../icml_results/3854_Learning%20Manifold%20Data%20with%20Flow%20Matching/images/0bb802be07872a4d722e7c5b87070ae78a71360fab33758016a7fe774884ba49.jpg)

![9c9404fe73f0c550d9b2a50a64897de249d164d1ec8a4bc3f4ae540808477772.jpg](../icml_results/3854_Learning%20Manifold%20Data%20with%20Flow%20Matching/images/9c9404fe73f0c550d9b2a50a64897de249d164d1ec8a4bc3f4ae540808477772.jpg)

<a id="paper-3815"></a>

## Learning Adaptive Perturbation-Conditioned Contexts for Robust Transcriptional Response Prediction


### Images

![032c656aa704794c87681aa27919ddddc43451562b8615b23c5c5abc7065388f.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/032c656aa704794c87681aa27919ddddc43451562b8615b23c5c5abc7065388f.jpg)

![05f009c08a6e0d0867b3b05b9992e75697608c0849014b526f3b6b22977510ce.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/05f009c08a6e0d0867b3b05b9992e75697608c0849014b526f3b6b22977510ce.jpg)

![06a00f860823db4841a20deb21aaec102b66da88cdc2c72e70601e4f103cba0a.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/06a00f860823db4841a20deb21aaec102b66da88cdc2c72e70601e4f103cba0a.jpg)

![1462d38851cc3d6abd2d874ae9f51afa7d0fbf22f9dd9980c37197d501edc50d.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/1462d38851cc3d6abd2d874ae9f51afa7d0fbf22f9dd9980c37197d501edc50d.jpg)

![37fb52ff33b44af3853b1eb376df858ee6a451aca548bf55a54b83cfd048e2da.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/37fb52ff33b44af3853b1eb376df858ee6a451aca548bf55a54b83cfd048e2da.jpg)

![3a27059cc16b4c23afc59d7ba5d27ce5fea3dafe6d48a4c1a6a332b061a4584a.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/3a27059cc16b4c23afc59d7ba5d27ce5fea3dafe6d48a4c1a6a332b061a4584a.jpg)

![412ba0d514c9268b047d2a6d2808b6064274d228f80052b97dac36f984ee987a.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/412ba0d514c9268b047d2a6d2808b6064274d228f80052b97dac36f984ee987a.jpg)

![4556f4fd840c7aaa665e01bba353e417662b2f21889ca5046f9e0e3e664705e7.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/4556f4fd840c7aaa665e01bba353e417662b2f21889ca5046f9e0e3e664705e7.jpg)

![4db062ab255ce2af517f2b5cbd613aff493e7606c21c57e232be6d6215c6086a.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/4db062ab255ce2af517f2b5cbd613aff493e7606c21c57e232be6d6215c6086a.jpg)

![57f8c213faf8df34cdaa872c620e9fd1023c9a04107a1dd0d209366c68083d9e.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/57f8c213faf8df34cdaa872c620e9fd1023c9a04107a1dd0d209366c68083d9e.jpg)

![5fba02422e602b7c581594c49012f1af23ab271451a2bf01cf31059f9b9164eb.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/5fba02422e602b7c581594c49012f1af23ab271451a2bf01cf31059f9b9164eb.jpg)

![72b7c5209db1c1847f4cee82a7e74c0367bbcced3589ee77ee520f942313bf3d.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/72b7c5209db1c1847f4cee82a7e74c0367bbcced3589ee77ee520f942313bf3d.jpg)

![7bb1b676028b4fcc089c92013d4213699d973c1b09399b058aeafce1a38f9012.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/7bb1b676028b4fcc089c92013d4213699d973c1b09399b058aeafce1a38f9012.jpg)

![805c0ee82691f977699d6f0672e2c614950d95de6d1974828552d8f10cd804c8.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/805c0ee82691f977699d6f0672e2c614950d95de6d1974828552d8f10cd804c8.jpg)

![9149fd9e4b5da46da698764df18515a3e0cc3f8960f775bed05a7be62ddd61e4.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/9149fd9e4b5da46da698764df18515a3e0cc3f8960f775bed05a7be62ddd61e4.jpg)

![975651323c47e8ee2ca3bc7da9288e5b2e34894dea6c0221eb74a7e512368152.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/975651323c47e8ee2ca3bc7da9288e5b2e34894dea6c0221eb74a7e512368152.jpg)

![9a1d4560995ed571db027d7524f709cc02c47f85bcc9eb85b4a892e394573a73.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/9a1d4560995ed571db027d7524f709cc02c47f85bcc9eb85b4a892e394573a73.jpg)

![9db7885fff01bab4b25f7df4a32a16601effd49f4fd5b6bd9eadcdf8860f5d71.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/9db7885fff01bab4b25f7df4a32a16601effd49f4fd5b6bd9eadcdf8860f5d71.jpg)

![adc757abed1a57d8039270d1e42989117209d1b5e6514cb40e179fd10baa8610.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/adc757abed1a57d8039270d1e42989117209d1b5e6514cb40e179fd10baa8610.jpg)

![c4f2817a4f5ee8f94a8ac531eef6ed1a8a45d4be0fd2def005be4db4a40abc8e.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/c4f2817a4f5ee8f94a8ac531eef6ed1a8a45d4be0fd2def005be4db4a40abc8e.jpg)

![cb1707b2ccff1293807a3df21eb17bc2f66415d01299f9805cc58e34f2ab391d.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/cb1707b2ccff1293807a3df21eb17bc2f66415d01299f9805cc58e34f2ab391d.jpg)

![d08f0cff1a6ec82233f66585b2f87f272589194afdd8b8b00baf8c65bc167038.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/d08f0cff1a6ec82233f66585b2f87f272589194afdd8b8b00baf8c65bc167038.jpg)

![e599e9a1790831dd7859349015da10c65bd331b4ba75f816b53fa48e47cd3a56.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/e599e9a1790831dd7859349015da10c65bd331b4ba75f816b53fa48e47cd3a56.jpg)

![f24729c76151a4053d85ff0879a74d65bc66949b83526a2903b7b8769ea8c8bf.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/f24729c76151a4053d85ff0879a74d65bc66949b83526a2903b7b8769ea8c8bf.jpg)

![f29f0cb77e1a6fde3821aaec8f002f9f295c028fff9a64188db17a3efa3829c7.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/f29f0cb77e1a6fde3821aaec8f002f9f295c028fff9a64188db17a3efa3829c7.jpg)

![f5bf47bb39b41df75a9b7e9a0f2ac7fdeeb2fe728dd5d4644136d2dfb82e345c.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/f5bf47bb39b41df75a9b7e9a0f2ac7fdeeb2fe728dd5d4644136d2dfb82e345c.jpg)

![f8f1d06b5c86e436e314bfa2864e4232392e280766cb91af65063d02872cb8b3.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/f8f1d06b5c86e436e314bfa2864e4232392e280766cb91af65063d02872cb8b3.jpg)

![fd724c51c61c6cb6c472d97fc855579d8611f706119c13e24e3ab015b2a590ad.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/fd724c51c61c6cb6c472d97fc855579d8611f706119c13e24e3ab015b2a590ad.jpg)

![fed4ef69243243171b4922c599f6eb5ddad3de4116daf6217aa8608bc3be23a4.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/images/fed4ef69243243171b4922c599f6eb5ddad3de4116daf6217aa8608bc3be23a4.jpg)

### Tables

![07cc3b711c8d10a21e1b8d1a4389347f590e8a0052fec841d9beff31a46be109.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/07cc3b711c8d10a21e1b8d1a4389347f590e8a0052fec841d9beff31a46be109.jpg)

![1c2a71b8ba9d00c0190e88c07e53ae794adcb5f95a65dd169f91303f9b4f9c54.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/1c2a71b8ba9d00c0190e88c07e53ae794adcb5f95a65dd169f91303f9b4f9c54.jpg)

![23ec7ae24c108dce16cc1bf74034922d6b9a97de77d613407c4886702f6f05ff.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/23ec7ae24c108dce16cc1bf74034922d6b9a97de77d613407c4886702f6f05ff.jpg)

![595b0f246b11c3b502ba642a9ef994d82a8907d4d97f5e799a35a4c843a8f250.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/595b0f246b11c3b502ba642a9ef994d82a8907d4d97f5e799a35a4c843a8f250.jpg)

![602b2d4dc4745cf0ea631e53cde14968fb5c34b322062f4fcce04ab4ff5e2197.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/602b2d4dc4745cf0ea631e53cde14968fb5c34b322062f4fcce04ab4ff5e2197.jpg)

![6f4f36d0d29ee1dfa6ca953698d9f08f98b1c2afd3a368f155fb7353ab2c9f5b.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/6f4f36d0d29ee1dfa6ca953698d9f08f98b1c2afd3a368f155fb7353ab2c9f5b.jpg)

![7daba1f9a1830cf300cda74d136c2093b5679c6dfdeeb7b0fccdc723155c10f2.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/7daba1f9a1830cf300cda74d136c2093b5679c6dfdeeb7b0fccdc723155c10f2.jpg)

![9a5bd0b84507346b1b7537448cb77ebeacb6dbc75f2a2a8543efd2598ed84968.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/9a5bd0b84507346b1b7537448cb77ebeacb6dbc75f2a2a8543efd2598ed84968.jpg)

![aa7b686c122eb9ca9cd0f4ba60e2a7a90067e63453a63cf1c850090f3ea00890.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/aa7b686c122eb9ca9cd0f4ba60e2a7a90067e63453a63cf1c850090f3ea00890.jpg)

![bc30b4027c339df614b9a980966ca34e12e8d44091a16eab2858724e5c6c8322.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/bc30b4027c339df614b9a980966ca34e12e8d44091a16eab2858724e5c6c8322.jpg)

![bef36d8ddb7e7384af7e351c66c80d167f95e0f48333a2722e914a1eebfc20f3.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/bef36d8ddb7e7384af7e351c66c80d167f95e0f48333a2722e914a1eebfc20f3.jpg)

![ca3387fd350306530eaef8583a8ded4c3edcec0b2fce633478db3633daa7db9a.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/ca3387fd350306530eaef8583a8ded4c3edcec0b2fce633478db3633daa7db9a.jpg)

![d28588a91a394767c756b7660b4bfcddaa1a5579ef17c30014874334b70c8d99.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/d28588a91a394767c756b7660b4bfcddaa1a5579ef17c30014874334b70c8d99.jpg)

![d2db7a5e9b0bedb33f13578108120ef6ac5401197d3da028c6035fea72df5f07.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/d2db7a5e9b0bedb33f13578108120ef6ac5401197d3da028c6035fea72df5f07.jpg)

![f0e26457f0d81b5b372c80683255e75d77d2b0fbc5c7c49e99d7711e350de176.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/f0e26457f0d81b5b372c80683255e75d77d2b0fbc5c7c49e99d7711e350de176.jpg)

![fc8bbf9ed9cc8292cf3e9125a294367c162efa882f2c9330582294342d32acae.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/fc8bbf9ed9cc8292cf3e9125a294367c162efa882f2c9330582294342d32acae.jpg)

![ff4cacbb2a37d2d9966936065707849e5e5e9412eee5cfde6fee3bd4d8620ebb.jpg](../icml_results/3855_Learning%20Adaptive%20Perturbation-Conditioned%20Contexts%20for%20Robust%20Transcriptional%20Response%20Prediction/tables/ff4cacbb2a37d2d9966936065707849e5e5e9412eee5cfde6fee3bd4d8620ebb.jpg)

<a id="paper-3816"></a>

## MarketSim: Simulating Stock Markets with Large-Scale Generative Agents


### Images

![01539f3fb3b2ec138c2d4e4d6c998c98734b77037a86870ce9fe913fb3c8832f.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/01539f3fb3b2ec138c2d4e4d6c998c98734b77037a86870ce9fe913fb3c8832f.jpg)

![1ca03180d30fedbf3f70677809b1691cbaf24d5f3ec934d49a5b2d5586424a22.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/1ca03180d30fedbf3f70677809b1691cbaf24d5f3ec934d49a5b2d5586424a22.jpg)

![1ce665393746772f3de26ca096bb91bab7d33db3f4b1cf476072782c03bf0bb3.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/1ce665393746772f3de26ca096bb91bab7d33db3f4b1cf476072782c03bf0bb3.jpg)

![23589f3a895f5c1d281106b669f79df2f0cbdf5965918ce66c1b96de00bafca0.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/23589f3a895f5c1d281106b669f79df2f0cbdf5965918ce66c1b96de00bafca0.jpg)

![26ebc798a2d827faf661881bfe0187d75fc5524c4071ae7f74f5247364bc2ee4.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/26ebc798a2d827faf661881bfe0187d75fc5524c4071ae7f74f5247364bc2ee4.jpg)

![335977a4f4d77ae92ec139ac6acb7fca8b68a3e0bf9ca6d418a32b44bfe33857.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/335977a4f4d77ae92ec139ac6acb7fca8b68a3e0bf9ca6d418a32b44bfe33857.jpg)

![3364dccb0f738749e6d0954c40045ad527dc180c0a375beb312bb680fd8b553b.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/3364dccb0f738749e6d0954c40045ad527dc180c0a375beb312bb680fd8b553b.jpg)

![35948c4cc7f6b19a3cb5f147768bfdee002ef664ef224c6fe04438ba4b75eda1.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/35948c4cc7f6b19a3cb5f147768bfdee002ef664ef224c6fe04438ba4b75eda1.jpg)

![38e79d5639cb28af9a1a1ed64bf4dc6936a9a1bf9be60f50289cd6a95488016f.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/38e79d5639cb28af9a1a1ed64bf4dc6936a9a1bf9be60f50289cd6a95488016f.jpg)

![3e27f78c15b2b0dd78a256c66fb5a7b89b2154d6da2b1fa6f65ffc4876d3eb2e.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/3e27f78c15b2b0dd78a256c66fb5a7b89b2154d6da2b1fa6f65ffc4876d3eb2e.jpg)

![4327cce792579a4cccc7fead28f8c83ba0c9182ed2d7fee1f1dc0b838f9ee435.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/4327cce792579a4cccc7fead28f8c83ba0c9182ed2d7fee1f1dc0b838f9ee435.jpg)

![4b8d15619e91c935676dc0c7d687d52e76d9dc12b230a54954f3f93ed96f1140.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/4b8d15619e91c935676dc0c7d687d52e76d9dc12b230a54954f3f93ed96f1140.jpg)

![4bc3923ee27bfde08322701b58f0fb9d27529a2c656e28a234666b7830966bcb.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/4bc3923ee27bfde08322701b58f0fb9d27529a2c656e28a234666b7830966bcb.jpg)

![511894eda520cf1c8e084c0c7fd8ca3902185c25c8871a197ac3f7a411aa3fc2.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/511894eda520cf1c8e084c0c7fd8ca3902185c25c8871a197ac3f7a411aa3fc2.jpg)

![51f73a2c315bf7bc6a6c5198b83d875ae3e230796857d7c2c5c935e3cccea917.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/51f73a2c315bf7bc6a6c5198b83d875ae3e230796857d7c2c5c935e3cccea917.jpg)

![570ebb209520a9dcf2b0470b819fef59fb35584bedc2b634617e02b19527d717.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/570ebb209520a9dcf2b0470b819fef59fb35584bedc2b634617e02b19527d717.jpg)

![5916a4f6ce2501a6115ab9d696f553a59854183b1aa2677970577dc1e73d60b1.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/5916a4f6ce2501a6115ab9d696f553a59854183b1aa2677970577dc1e73d60b1.jpg)

![5c0b9dd5ccc7aca9a2194bec1b9bef8872c2f0e929c7e84ee02ea7cb517cc596.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/5c0b9dd5ccc7aca9a2194bec1b9bef8872c2f0e929c7e84ee02ea7cb517cc596.jpg)

![5e343c7d1f8a621a9c9d6b2a891560d81645ff7517679303246bb5d3232af0e6.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/5e343c7d1f8a621a9c9d6b2a891560d81645ff7517679303246bb5d3232af0e6.jpg)

![5e739ece63b3d6e371deb78312387f9e8f0c69c8501ba7458ec1ea8ea8006c24.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/5e739ece63b3d6e371deb78312387f9e8f0c69c8501ba7458ec1ea8ea8006c24.jpg)

![5f96f84938bfb6f6fcd0ac25983be8a6174898f074f1c8aee5e4a7a49f8e4fa3.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/5f96f84938bfb6f6fcd0ac25983be8a6174898f074f1c8aee5e4a7a49f8e4fa3.jpg)

![62e24a0338c6bbba119c4e2bc8f5887c055b479b15dc3cd624b58339f910476c.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/62e24a0338c6bbba119c4e2bc8f5887c055b479b15dc3cd624b58339f910476c.jpg)

![67f59218e7be55aecdf021d12e45175d48cae20ad28c7f0a22ef856773e7f882.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/67f59218e7be55aecdf021d12e45175d48cae20ad28c7f0a22ef856773e7f882.jpg)

![685e60d207b5274f69aedc05f1eea2993b448a16d2ad4f3bf0dda31fceec3df8.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/685e60d207b5274f69aedc05f1eea2993b448a16d2ad4f3bf0dda31fceec3df8.jpg)

![6ccae873a4300451c30ffb770e9df4bf176d7994c615e6dcbf4bca35d53e800e.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/6ccae873a4300451c30ffb770e9df4bf176d7994c615e6dcbf4bca35d53e800e.jpg)

![73ce5a119be4a19044c90e875fe447ff7d1f65e22b11cf21cd0976b9db1cc8ab.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/73ce5a119be4a19044c90e875fe447ff7d1f65e22b11cf21cd0976b9db1cc8ab.jpg)

![769e6e030057319b2109d4e919c3394442f5719e03d8e164072297bafae60209.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/769e6e030057319b2109d4e919c3394442f5719e03d8e164072297bafae60209.jpg)

![7d3c0dd353d2cdcb029e5a9f5ac0af1f83ab412182c2088db8f9a9876ff0b75e.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/7d3c0dd353d2cdcb029e5a9f5ac0af1f83ab412182c2088db8f9a9876ff0b75e.jpg)

![8357b60b0a46191d39bc03a34ea6f62422706d555093c1b61079d41dd0ab75d0.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/8357b60b0a46191d39bc03a34ea6f62422706d555093c1b61079d41dd0ab75d0.jpg)

![8427926fe4563b57a9aab992698677177865c70fac9c7e8f08a27913661c087a.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/8427926fe4563b57a9aab992698677177865c70fac9c7e8f08a27913661c087a.jpg)

![882ecd8dc54c81bed76c2eba19a8f84389c053729b28d0c0b15076d45b7075db.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/882ecd8dc54c81bed76c2eba19a8f84389c053729b28d0c0b15076d45b7075db.jpg)

![8a9468d87f974c54069575fc92546a6d91e52301e225fbbd2914a36485cfab03.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/8a9468d87f974c54069575fc92546a6d91e52301e225fbbd2914a36485cfab03.jpg)

![8d317182196be65983bebe23040ec4d3aa3a0baf0d640662bcc79998253cdf7a.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/8d317182196be65983bebe23040ec4d3aa3a0baf0d640662bcc79998253cdf7a.jpg)

![902a19f4b2372a76ab22e66d7446947717eb6b1feed0eeebd448a1d5140350ce.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/902a19f4b2372a76ab22e66d7446947717eb6b1feed0eeebd448a1d5140350ce.jpg)

![924f23a1ce405646ff20a5abcd5701585f0143dc7ae8ae7de7df87806db907c2.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/924f23a1ce405646ff20a5abcd5701585f0143dc7ae8ae7de7df87806db907c2.jpg)

![9e398270944b30ffe8fc6494783a204e2bc66f147af907ca12dcab0af3aeaab0.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/9e398270944b30ffe8fc6494783a204e2bc66f147af907ca12dcab0af3aeaab0.jpg)

![a42f3b10a12a263e52fb0ee84ada5febad1bc2886723cdb096866a1ad08ab393.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/a42f3b10a12a263e52fb0ee84ada5febad1bc2886723cdb096866a1ad08ab393.jpg)

![a532a62352c2e90b25b05150776a9648a0d4bd85cba314b02de06c320a1f00b0.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/a532a62352c2e90b25b05150776a9648a0d4bd85cba314b02de06c320a1f00b0.jpg)

![b078fadcecf326bdac8d4a77a998e7df13bbb98f220858bdc402ce2895f323b8.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/b078fadcecf326bdac8d4a77a998e7df13bbb98f220858bdc402ce2895f323b8.jpg)

![b1001ba250995a5f244281df6cb01a665536e74206e4e91eb13449b6c16cb644.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/b1001ba250995a5f244281df6cb01a665536e74206e4e91eb13449b6c16cb644.jpg)

![b13eb1a5d7f1de9837bc564ed7a4c67f7243fc1c343331e4a5c83033359a519f.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/b13eb1a5d7f1de9837bc564ed7a4c67f7243fc1c343331e4a5c83033359a519f.jpg)

![b80e9178cbd5aad83b1c350d33e1c5ba6a4ed819511adb52cbaaf4fb1c4dd564.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/b80e9178cbd5aad83b1c350d33e1c5ba6a4ed819511adb52cbaaf4fb1c4dd564.jpg)

![c0db33604b126bda98487dc9b7b2dcdfe091717ae6e8389509be6959e4434282.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/c0db33604b126bda98487dc9b7b2dcdfe091717ae6e8389509be6959e4434282.jpg)

![c7abcc7aa1ad889977bfc0d7d73bb960627ce38aeb95d922f9a314fa5cd0c008.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/c7abcc7aa1ad889977bfc0d7d73bb960627ce38aeb95d922f9a314fa5cd0c008.jpg)

![c950dbc01cff67943e57a9cdb6c18a7daf56466b4f219179974f02533551ec52.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/c950dbc01cff67943e57a9cdb6c18a7daf56466b4f219179974f02533551ec52.jpg)

![ca6112342fc45cbeaff67240194cd05af9d9225b9625e588cb01599500e9a372.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/ca6112342fc45cbeaff67240194cd05af9d9225b9625e588cb01599500e9a372.jpg)

![cddbf8da08d22c1af386ae7d9d0598c70188957d988b2766dc186bbe472c323d.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/cddbf8da08d22c1af386ae7d9d0598c70188957d988b2766dc186bbe472c323d.jpg)

![df62759412e9b25e4978ccda4a36c979c63219360b821051f0b7847468e0658a.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/df62759412e9b25e4978ccda4a36c979c63219360b821051f0b7847468e0658a.jpg)

![e0c393894395c64bb51fd323bd11aec5b208f92e962c1f141d2d6c49a6ab9f3b.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/e0c393894395c64bb51fd323bd11aec5b208f92e962c1f141d2d6c49a6ab9f3b.jpg)

![e4ca6084a52e13ad34a77597ba46ac25e1f7f902933e25b8800de43175a863eb.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/e4ca6084a52e13ad34a77597ba46ac25e1f7f902933e25b8800de43175a863eb.jpg)

![ecdf8d9ad303d839d7cdd3882553a2f2df59fa15bd6285041f3bf56c5350cb2f.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/ecdf8d9ad303d839d7cdd3882553a2f2df59fa15bd6285041f3bf56c5350cb2f.jpg)

![fab38f9f5991a86ec206a12995e73ae23025724d32f7a55c2be681003e861693.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/images/fab38f9f5991a86ec206a12995e73ae23025724d32f7a55c2be681003e861693.jpg)

### Tables

![06a570cb0638d0cb27c76d5608e164895a547f686eeea2cca6fd439d0345912a.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/06a570cb0638d0cb27c76d5608e164895a547f686eeea2cca6fd439d0345912a.jpg)

![11ad1ad269597272dfa2e401e3ab1167d437add3e599371e1d21a01366b563ad.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/11ad1ad269597272dfa2e401e3ab1167d437add3e599371e1d21a01366b563ad.jpg)

![300eaba82f84882463a63e0a3462a80fe69562bb8dc28772c83ee8b24732929c.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/300eaba82f84882463a63e0a3462a80fe69562bb8dc28772c83ee8b24732929c.jpg)

![3cfc5e3a3536bd11232ef08065cafc241132923cb9821b09e13427f90ffacefa.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/3cfc5e3a3536bd11232ef08065cafc241132923cb9821b09e13427f90ffacefa.jpg)

![4b0aa2b0ebff4411082dcf2fd4450d637f10e267b704db4590a0739277e0b062.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/4b0aa2b0ebff4411082dcf2fd4450d637f10e267b704db4590a0739277e0b062.jpg)

![5db3468852b665fe25aab0bc29819f89d2b30913e0739ac381e4836203b28e79.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/5db3468852b665fe25aab0bc29819f89d2b30913e0739ac381e4836203b28e79.jpg)

![681645c379018cf5d99a126423ea03742e0fd22e5c49c705f6b079e82228e277.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/681645c379018cf5d99a126423ea03742e0fd22e5c49c705f6b079e82228e277.jpg)

![90828b95a78121ae406efdf45bb6d520b88cdc1f260cc41cb348015b064bdf21.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/90828b95a78121ae406efdf45bb6d520b88cdc1f260cc41cb348015b064bdf21.jpg)

![95d3ea553eaa7cdb519d5a1fe8b4ec6feaf9f65e94a3a56013f94f4efae14b25.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/95d3ea553eaa7cdb519d5a1fe8b4ec6feaf9f65e94a3a56013f94f4efae14b25.jpg)

![b3253d71fa827b8cbb606bcc6092911fecb5b9a787cd56ed61905ea554173ce4.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/b3253d71fa827b8cbb606bcc6092911fecb5b9a787cd56ed61905ea554173ce4.jpg)

![c0b52cecf5388d4ec2e762db9bd10610cf27cfc26a4cd03828c98daa049600c7.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/c0b52cecf5388d4ec2e762db9bd10610cf27cfc26a4cd03828c98daa049600c7.jpg)

![d40421b25e3af36b8ee8d8f8f944f4cb74daba6bcb57e37b33d897dbec7f96cd.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/d40421b25e3af36b8ee8d8f8f944f4cb74daba6bcb57e37b33d897dbec7f96cd.jpg)

![d950c6cc076ff22cd650ff1a73c0f01e7a28c9f8cf68e286bd52656a8fe8894f.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/d950c6cc076ff22cd650ff1a73c0f01e7a28c9f8cf68e286bd52656a8fe8894f.jpg)

![db70c13983494569c950fbe16b9194bc0b8920b0c628192406c5b928514a78a1.jpg](../icml_results/3856_MarketSim_%20Simulating%20Stock%20Markets%20with%20Large-Scale%20Generative%20Agents/tables/db70c13983494569c950fbe16b9194bc0b8920b0c628192406c5b928514a78a1.jpg)

<a id="paper-3817"></a>

## Infinite-Dimensional Generative Diffusions via Doob’s h-Transform


### Images

![2a26093285be06f2213a9e16e4b10cb0e6652b8d02b5d55e3d140a1ebd64de31.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/2a26093285be06f2213a9e16e4b10cb0e6652b8d02b5d55e3d140a1ebd64de31.jpg)

![3b5a357be26b191c7501eca80c7ae3455a1559f83220481554b711451c9b3087.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/3b5a357be26b191c7501eca80c7ae3455a1559f83220481554b711451c9b3087.jpg)

![54932b496b0ad2f42a2d88216c91454ef6d4cc70c2a0b9a7dfc679d30be74a9b.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/54932b496b0ad2f42a2d88216c91454ef6d4cc70c2a0b9a7dfc679d30be74a9b.jpg)

![6ba74e094a9c93c97af43d0394dea9617edaa13bff7ba0e2c9c94e88c98e4bb2.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/6ba74e094a9c93c97af43d0394dea9617edaa13bff7ba0e2c9c94e88c98e4bb2.jpg)

![84cb0d691ec1158a8d6d9187869bd40487ffccda6b15260b90a2d8c9c31274b5.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/84cb0d691ec1158a8d6d9187869bd40487ffccda6b15260b90a2d8c9c31274b5.jpg)

![89ea8bcfdd58191c4919f619eea2fc113922c799210e99029d875af2147804b2.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/89ea8bcfdd58191c4919f619eea2fc113922c799210e99029d875af2147804b2.jpg)

![8ebe468e79deeaf5212c32bdb673609190233d40969be970365b2dbaba1d0737.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/8ebe468e79deeaf5212c32bdb673609190233d40969be970365b2dbaba1d0737.jpg)

![95e95454d92e0cc45a98a4cb025f23ea4de3ba1362908eefb45ab58767d6be33.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/95e95454d92e0cc45a98a4cb025f23ea4de3ba1362908eefb45ab58767d6be33.jpg)

![c11e60eba863e6ad6261997d3cf6c9f0e8b0088229919ba248c75e0234e0ba4d.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/c11e60eba863e6ad6261997d3cf6c9f0e8b0088229919ba248c75e0234e0ba4d.jpg)

![d9cd45699cd824216b7d84a1f06e6b487df0933f5ad66ac9709faef80eebc0c2.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/images/d9cd45699cd824216b7d84a1f06e6b487df0933f5ad66ac9709faef80eebc0c2.jpg)

### Tables

![00a4010945fc496ca5668c0e8c635cc4d8d905cb539be4fc908a923fa7320f4e.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/tables/00a4010945fc496ca5668c0e8c635cc4d8d905cb539be4fc908a923fa7320f4e.jpg)

![b1ee2c21afa78e2efcf9f7334ec905e683f6c59ff046245d0c955809cb582705.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/tables/b1ee2c21afa78e2efcf9f7334ec905e683f6c59ff046245d0c955809cb582705.jpg)

![cbf4513aa6269c66c29052b0dc9c0340df886afbb1a44a38aa6eb119f9f53cb6.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/tables/cbf4513aa6269c66c29052b0dc9c0340df886afbb1a44a38aa6eb119f9f53cb6.jpg)

![e1069006108a93cd0c9adad2667e9d23a3dfdbf2c5d9fe1a599d9da91ab9c665.jpg](../icml_results/3857_Infinite-Dimensional%20Generative%20Diffusions%20via%20Doob%E2%80%99s%20h-Transform/tables/e1069006108a93cd0c9adad2667e9d23a3dfdbf2c5d9fe1a599d9da91ab9c665.jpg)

<a id="paper-3818"></a>

## Learning the ESG Geometry with Domain Aware Language Models


### Images

![0127da0dc472cd15d3b4b3ee87ca046e45bf75c802d20a58ebdd9efa002d7d0c.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/0127da0dc472cd15d3b4b3ee87ca046e45bf75c802d20a58ebdd9efa002d7d0c.jpg)

![1126635127675a3dd77aca5066a02d461155209bc9539b7085624061139bebf5.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/1126635127675a3dd77aca5066a02d461155209bc9539b7085624061139bebf5.jpg)

![1668e1e5db4b1706dddd422089ac75a830248f32159ed7ebaaf85510ba8d6415.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/1668e1e5db4b1706dddd422089ac75a830248f32159ed7ebaaf85510ba8d6415.jpg)

![3726ecf0df09adcf1452001d9a6066a1a7024d5cbb483ee588341f8573ffebab.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/3726ecf0df09adcf1452001d9a6066a1a7024d5cbb483ee588341f8573ffebab.jpg)

![8494acaac56cf8b65bb8e898bcf55f254d2b0aec28c67793a978d533c3336086.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/8494acaac56cf8b65bb8e898bcf55f254d2b0aec28c67793a978d533c3336086.jpg)

![84abf57cf4446029b5a29e0b75120af0aaf661c8db9b5d94a389964e0321c1bc.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/84abf57cf4446029b5a29e0b75120af0aaf661c8db9b5d94a389964e0321c1bc.jpg)

![89101939156d96c7b520b48ff2c0d028547d88f6815dc35b4b1df3c6469079ba.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/89101939156d96c7b520b48ff2c0d028547d88f6815dc35b4b1df3c6469079ba.jpg)

![9c42bccd61ce171af3d53b8d12233e353da1737095838e5ee689fcd40666cab2.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/9c42bccd61ce171af3d53b8d12233e353da1737095838e5ee689fcd40666cab2.jpg)

![acad6a2a76ab9dab389a9c814ef6bfcb05e0671b7695f1ca7604a231f1806d97.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/acad6a2a76ab9dab389a9c814ef6bfcb05e0671b7695f1ca7604a231f1806d97.jpg)

![bbae0e82b1f74ca6fcd8280dba8ad678eda794421583bc5fb8ac8074d6f28649.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/images/bbae0e82b1f74ca6fcd8280dba8ad678eda794421583bc5fb8ac8074d6f28649.jpg)

### Tables

![116beae6471a177e66302cfc8cc17e1bb903b11ae755bf3416aa7a07373a987f.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/116beae6471a177e66302cfc8cc17e1bb903b11ae755bf3416aa7a07373a987f.jpg)

![14bba846823b044bf2191d9f5bd296e9de755b71ce3aafe725bfcd3cdaf32d46.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/14bba846823b044bf2191d9f5bd296e9de755b71ce3aafe725bfcd3cdaf32d46.jpg)

![4b12c3305ef9d269fa4de0f005f6df140284e4c63977da6f80ff131863047cbd.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/4b12c3305ef9d269fa4de0f005f6df140284e4c63977da6f80ff131863047cbd.jpg)

![568d4fd3ffe4ef951837705bd526948fbfe64f3afacf38ed2555924361a3f15e.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/568d4fd3ffe4ef951837705bd526948fbfe64f3afacf38ed2555924361a3f15e.jpg)

![73a7a1b219174ee490536f2f45ea0c2ee61f91d82c0009cf7e83f38520db98be.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/73a7a1b219174ee490536f2f45ea0c2ee61f91d82c0009cf7e83f38520db98be.jpg)

![9b22ae03cbedf2627a6b164b925973d1d201e7dac2afb5ab72b4c9295e57aab3.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/9b22ae03cbedf2627a6b164b925973d1d201e7dac2afb5ab72b4c9295e57aab3.jpg)

![abac23e3d1fe4875f91eaac31f77fca05cb4fa061c95550d401386e1763c9579.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/abac23e3d1fe4875f91eaac31f77fca05cb4fa061c95550d401386e1763c9579.jpg)

![d5b92bdd3c53c3052822f65f95aac883e29ad73a9b921d08e05dbd754b3d748a.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/d5b92bdd3c53c3052822f65f95aac883e29ad73a9b921d08e05dbd754b3d748a.jpg)

![dcd97b022890aeb3c25dcdccd123c4896d0a9a4f8973e498cf0acd2ee8d71bc6.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/dcd97b022890aeb3c25dcdccd123c4896d0a9a4f8973e498cf0acd2ee8d71bc6.jpg)

![e3402f4080a89df1e788aca6a210488d144145e2174023080f952bc178cf3b0e.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/e3402f4080a89df1e788aca6a210488d144145e2174023080f952bc178cf3b0e.jpg)

![e741a60b36d4cb32958a6cf7e0d061fb673d9fd6306beda4bfd511393ac92507.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/e741a60b36d4cb32958a6cf7e0d061fb673d9fd6306beda4bfd511393ac92507.jpg)

![fdec77362a571e5240801278bdf572856fea6925d553372d2789b445b4a00c4f.jpg](../icml_results/3858_Learning%20the%20ESG%20Geometry%20with%20Domain%20Aware%20Language%20Models/tables/fdec77362a571e5240801278bdf572856fea6925d553372d2789b445b4a00c4f.jpg)

<a id="paper-3819"></a>

## Temporal Context Reinstatement Drives Episodic-Like Order Memory in Long-Context Language Models


### Images

![033723423e766affac1b930aa15af409af3091d780e388b7f563553ef115c1a3.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/033723423e766affac1b930aa15af409af3091d780e388b7f563553ef115c1a3.jpg)

![19da3f8ff8b2b4a4b5b7de5f4d1f777ec49d7af19ba17a98bde8049e1a909307.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/19da3f8ff8b2b4a4b5b7de5f4d1f777ec49d7af19ba17a98bde8049e1a909307.jpg)

![201ec29cc4fb236c0568be7800323e100a987f65b9aabafdafd7419d12ec5334.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/201ec29cc4fb236c0568be7800323e100a987f65b9aabafdafd7419d12ec5334.jpg)

![3497fb10b77ec282b8096d0e9fc8a62d1f3c0ed38b9f66f3b27e2a9247a3c85e.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/3497fb10b77ec282b8096d0e9fc8a62d1f3c0ed38b9f66f3b27e2a9247a3c85e.jpg)

![369e9188a9f65789f59d98a70700d4a98cd6fe50fa3e047584e9bb430f0f0146.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/369e9188a9f65789f59d98a70700d4a98cd6fe50fa3e047584e9bb430f0f0146.jpg)

![3753bc862184996380f0be242ae7c9d96e65961f889bbbd0f6567645a4a31def.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/3753bc862184996380f0be242ae7c9d96e65961f889bbbd0f6567645a4a31def.jpg)

![44667a93d375fd370529469eda291641a13fe25f26fcaf7528c401e38666cecc.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/44667a93d375fd370529469eda291641a13fe25f26fcaf7528c401e38666cecc.jpg)

![470c54d9a5075a5c732686f0dabec3f3c7d4468b73dd7228e919daf99583f040.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/470c54d9a5075a5c732686f0dabec3f3c7d4468b73dd7228e919daf99583f040.jpg)

![6ab223ad2556c555f84814bc529618065d2ced81f3c656df3d75d78fa98bb401.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/6ab223ad2556c555f84814bc529618065d2ced81f3c656df3d75d78fa98bb401.jpg)

![777336493bfb60529bab3cc9d0f921b81e94d16df43d01e87d38aa9444dd77f1.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/777336493bfb60529bab3cc9d0f921b81e94d16df43d01e87d38aa9444dd77f1.jpg)

![9d0808e34067dc135074f8f52c313a6b5058778c49c435172846a56f86632de3.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/9d0808e34067dc135074f8f52c313a6b5058778c49c435172846a56f86632de3.jpg)

![b991521651e60d680ad6776e4e52a03ccdbd144770d029f6c51d4bec02a3aeaf.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/b991521651e60d680ad6776e4e52a03ccdbd144770d029f6c51d4bec02a3aeaf.jpg)

![c80ea05364f3fd579706f5b7199aa4e3e7516e594e5fcae7d1782ae2071260c7.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/c80ea05364f3fd579706f5b7199aa4e3e7516e594e5fcae7d1782ae2071260c7.jpg)

![d04d61f081d1aba54a8bdf67d9f17105ea9311a54ecef84285f78c80180295d6.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/d04d61f081d1aba54a8bdf67d9f17105ea9311a54ecef84285f78c80180295d6.jpg)

![d348cd69e75b52826e592bbc850f5325fb993653e5e4aa6f3aa9a1b494c26357.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/d348cd69e75b52826e592bbc850f5325fb993653e5e4aa6f3aa9a1b494c26357.jpg)

![de770853db980e6eceb74a1ccc22616d63a0307314cb263af6c9e10210b6caeb.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/de770853db980e6eceb74a1ccc22616d63a0307314cb263af6c9e10210b6caeb.jpg)

![e0930cda398d0e47d1f86abea7f61d3b29b3db6db6f8215c6202ca9b02c24210.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/e0930cda398d0e47d1f86abea7f61d3b29b3db6db6f8215c6202ca9b02c24210.jpg)

![e7d1d44b6d8a58b71b3ab22153b78e67fd949666e993f85231097d5067204466.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/images/e7d1d44b6d8a58b71b3ab22153b78e67fd949666e993f85231097d5067204466.jpg)

### Tables

![c741f8263fc69132d990678b2893adc7776ed6cbb62d0e27aacfc5b40e08f01f.jpg](../icml_results/3859_Temporal%20Context%20Reinstatement%20Drives%20Episodic-Like%20Order%20Memory%20in%20Long-Context%20Language%20Models/tables/c741f8263fc69132d990678b2893adc7776ed6cbb62d0e27aacfc5b40e08f01f.jpg)

<a id="paper-3820"></a>

## ParalESN: Enabling parallel information processing in Reservoir Computing


### Images

![333bc3119c10a9f89475ebfe4ba9b80f1e0b88242a22aa1c71660a58f87c5fe6.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/333bc3119c10a9f89475ebfe4ba9b80f1e0b88242a22aa1c71660a58f87c5fe6.jpg)

![421feb42819463143d79724376175894d540563ed7b1884cf4712e523546a3e8.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/421feb42819463143d79724376175894d540563ed7b1884cf4712e523546a3e8.jpg)

![525059c061008d0dc7247387ffffa59ae365507bc3b33a865235b953fe3708d5.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/525059c061008d0dc7247387ffffa59ae365507bc3b33a865235b953fe3708d5.jpg)

![847d1b6854d5642c789ada5155932093c0a09a6fb445cab3e41f0f59e7309a2b.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/847d1b6854d5642c789ada5155932093c0a09a6fb445cab3e41f0f59e7309a2b.jpg)

![ad9a24dd05f96c1507a7fbb3eb1432d3489136731a770ce2b4eb2ffd5c9e597d.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/ad9a24dd05f96c1507a7fbb3eb1432d3489136731a770ce2b4eb2ffd5c9e597d.jpg)

![ae0f4d59614cb4503ef3eb09ed76168c0961c6313fa08d2aa48e205dc40c4035.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/ae0f4d59614cb4503ef3eb09ed76168c0961c6313fa08d2aa48e205dc40c4035.jpg)

![b64ab777752e61a3e9107efc0369f9e0c7ecdcc458d0f9b86828b5f4bd2ea78b.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/images/b64ab777752e61a3e9107efc0369f9e0c7ecdcc458d0f9b86828b5f4bd2ea78b.jpg)

### Tables

![0fded409c5f724dbcb52ddefa1e01a6c30ec50091b590060b43a76d459246fb5.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/0fded409c5f724dbcb52ddefa1e01a6c30ec50091b590060b43a76d459246fb5.jpg)

![15556541c1497108146084a3ba6c45f1afbf90caef8bcab9f3af93238b80eb41.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/15556541c1497108146084a3ba6c45f1afbf90caef8bcab9f3af93238b80eb41.jpg)

![185341404758731dac320eb333598c73817c053f24cfdcbc0e4388d032f1fe0f.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/185341404758731dac320eb333598c73817c053f24cfdcbc0e4388d032f1fe0f.jpg)

![1974b59f5e12135fd4537bd512f44f4fa0205747c8e308814ea531313ed02078.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/1974b59f5e12135fd4537bd512f44f4fa0205747c8e308814ea531313ed02078.jpg)

![1f607cd9d3c3d5238067273706d7a80b941bbeb619cb4de640a07ae1ead53570.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/1f607cd9d3c3d5238067273706d7a80b941bbeb619cb4de640a07ae1ead53570.jpg)

![246f28ca7b295cc71e48613ade17254293e76ba98776b08dd9e95d096495dd84.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/246f28ca7b295cc71e48613ade17254293e76ba98776b08dd9e95d096495dd84.jpg)

![57697a83dbc2bc1375886d790bf93c6a672ed409cd92a0edeb7c85ace110db88.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/57697a83dbc2bc1375886d790bf93c6a672ed409cd92a0edeb7c85ace110db88.jpg)

![59a66b4544bccadf7e72877c693b2d9fc8bba29b4862c2f53fd5b8996ba1c76b.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/59a66b4544bccadf7e72877c693b2d9fc8bba29b4862c2f53fd5b8996ba1c76b.jpg)

![6fd3b49c57d0244197dbd9e75c562eb5010dcce52d673f3a433d2abc9e64ec27.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/6fd3b49c57d0244197dbd9e75c562eb5010dcce52d673f3a433d2abc9e64ec27.jpg)

![8332d8ce76ad2f2e3d6d68f3c7a05d8d6884afc345f636a8b751f212c51e31de.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/8332d8ce76ad2f2e3d6d68f3c7a05d8d6884afc345f636a8b751f212c51e31de.jpg)

![9715f54d1d950ecae3e9dc319a859fe0e1180c70e66e42e3b48cc2c7a0228643.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/9715f54d1d950ecae3e9dc319a859fe0e1180c70e66e42e3b48cc2c7a0228643.jpg)

![b0d445faad6fb50073b5eb4662ac4d2c2ca856aa320b65d7e20d1f4634d750e6.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/b0d445faad6fb50073b5eb4662ac4d2c2ca856aa320b65d7e20d1f4634d750e6.jpg)

![bb982906c918f84f293e6172ae311d8dc7173113eab2d4c5ee3656275323fa9b.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/bb982906c918f84f293e6172ae311d8dc7173113eab2d4c5ee3656275323fa9b.jpg)

![edebc43fa99fd01cc5052cfe7a7f0806ffe0e28d6cb9af18a59d0868e86d30e7.jpg](../icml_results/3860_ParalESN_%20Enabling%20parallel%20information%20processing%20in%20Reservoir%20Computing/tables/edebc43fa99fd01cc5052cfe7a7f0806ffe0e28d6cb9af18a59d0868e86d30e7.jpg)

<a id="paper-3821"></a>

## Not All Invariants Are Equal: Curating Training Data to Accelerate Program Verification with SLMs


### Images

![2a0369e1e479b447820831d8e9f7944b79e52aab26cb33a0f311256cfbca00ac.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/2a0369e1e479b447820831d8e9f7944b79e52aab26cb33a0f311256cfbca00ac.jpg)

![2df7e84bd7a10d727b6d5c7ce75d42f0adf6b00fd7539f24211a416386f8eaee.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/2df7e84bd7a10d727b6d5c7ce75d42f0adf6b00fd7539f24211a416386f8eaee.jpg)

![3b519bc1d5a6782e3b2af4b23fa792aea888a1939b68ffd9d444e1e7ea740482.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/3b519bc1d5a6782e3b2af4b23fa792aea888a1939b68ffd9d444e1e7ea740482.jpg)

![426c8e48a95a3c6240316249090042c6770275dad8e746db1a940f0cfd5cbfb4.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/426c8e48a95a3c6240316249090042c6770275dad8e746db1a940f0cfd5cbfb4.jpg)

![8a02d3b3ac19b697fc20c979260db1ae95ab42b922234bdf9f2233f87c0dcb05.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/8a02d3b3ac19b697fc20c979260db1ae95ab42b922234bdf9f2233f87c0dcb05.jpg)

![9c3fdffe82b3d98172ff1671737927a90baef5235564ba030b59195de53c568e.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/9c3fdffe82b3d98172ff1671737927a90baef5235564ba030b59195de53c568e.jpg)

![b7161b563558e5f03b7a86d14ea602f0592d3e3fe0fcf398819928f71190c8de.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/b7161b563558e5f03b7a86d14ea602f0592d3e3fe0fcf398819928f71190c8de.jpg)

![bd2df42a247ceb0dd9e83174c247c7c4df349a2a169c8ccc986ce4b037da373f.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/bd2df42a247ceb0dd9e83174c247c7c4df349a2a169c8ccc986ce4b037da373f.jpg)

![c085eb04f0babc9fdc71953b294fd57dfa67f38935f3e7502f55bb90a82edd0e.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/c085eb04f0babc9fdc71953b294fd57dfa67f38935f3e7502f55bb90a82edd0e.jpg)

![ca9998afd7a3dc3143ff205cfdfe397d09984d7393067fe37c153cd0cd1fb8e3.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/ca9998afd7a3dc3143ff205cfdfe397d09984d7393067fe37c153cd0cd1fb8e3.jpg)

![cdda5ad3aa2050495861025d2605cfebd362b2ee23f9bc66e7a9074a029d2f50.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/images/cdda5ad3aa2050495861025d2605cfebd362b2ee23f9bc66e7a9074a029d2f50.jpg)

### Tables

![0c868b5c36984950a2e20b552a6403fcbb9e0bb8d1246e4497a5268d67514347.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/0c868b5c36984950a2e20b552a6403fcbb9e0bb8d1246e4497a5268d67514347.jpg)

![22fc960de5939a80537b732ff1791d4c94b505b4fcdf8aac8037c9b3260cc9e1.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/22fc960de5939a80537b732ff1791d4c94b505b4fcdf8aac8037c9b3260cc9e1.jpg)

![29ca0891a0e624d018cd7f7131827843f5c61ae9c1740c62be94f35f85f557c7.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/29ca0891a0e624d018cd7f7131827843f5c61ae9c1740c62be94f35f85f557c7.jpg)

![5d338d21fed9474a02b5fa5f04c6eee1fbb9d450d738f113bf33be12d891647f.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/5d338d21fed9474a02b5fa5f04c6eee1fbb9d450d738f113bf33be12d891647f.jpg)

![6a55721a4a952e365130ef431e8690f8d5dd3950080573e485fbf3188410d56b.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/6a55721a4a952e365130ef431e8690f8d5dd3950080573e485fbf3188410d56b.jpg)

![85d9058227af0e43a02d07132908c070566225f893c6ee8bc08682979b172bd0.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/85d9058227af0e43a02d07132908c070566225f893c6ee8bc08682979b172bd0.jpg)

![97fb3b7603f2d5332e4df86baa15691d72a5fb63322a474f93a4dc3dfc3a8b5e.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/97fb3b7603f2d5332e4df86baa15691d72a5fb63322a474f93a4dc3dfc3a8b5e.jpg)

![c1c4dcd0797fe528cb7980db04ff99bba2adc2b0de06995fa49bf8ead970e23c.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/c1c4dcd0797fe528cb7980db04ff99bba2adc2b0de06995fa49bf8ead970e23c.jpg)

![c52c177ac575e0ac3982b1e72010d503d56ca4bbdb6b97c33aff144e5ec51b8a.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/c52c177ac575e0ac3982b1e72010d503d56ca4bbdb6b97c33aff144e5ec51b8a.jpg)

![e4a44aef4bcd63515bae5d10542b91c1eff21c51498d5bd432529b68b5e77a38.jpg](../icml_results/3861_Not%20All%20Invariants%20Are%20Equal_%20Curating%20Training%20Data%20to%20Accelerate%20Program%20Verification%20with%20SLMs/tables/e4a44aef4bcd63515bae5d10542b91c1eff21c51498d5bd432529b68b5e77a38.jpg)

<a id="paper-3822"></a>

## Goal-Oriented Lower-Tail Calibration of Gaussian Processes for Bayesian Optimization


### Images

![185e566af2f04b1c3faf1c82a87f28282b84a6b248161c12aec464834bd0f933.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/185e566af2f04b1c3faf1c82a87f28282b84a6b248161c12aec464834bd0f933.jpg)

![1d20a6078565b765c4d86224dd384e86f309ec6b2a5e521746cb1b3c8c889228.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/1d20a6078565b765c4d86224dd384e86f309ec6b2a5e521746cb1b3c8c889228.jpg)

![27f387f4b9722ea6ad240861c8a0b15135f9d3f1bdac52febdaca0933bc50118.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/27f387f4b9722ea6ad240861c8a0b15135f9d3f1bdac52febdaca0933bc50118.jpg)

![3cb010805d294c7380323e4b51b9abc4b38f535d17a101100fab52fc326207a4.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/3cb010805d294c7380323e4b51b9abc4b38f535d17a101100fab52fc326207a4.jpg)

![6b076cafd74cb08b7677ba34a07aeb21afa5d580564f04d887da3786be9c9e27.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/6b076cafd74cb08b7677ba34a07aeb21afa5d580564f04d887da3786be9c9e27.jpg)

![81cf1372bc64c6e97d883caaec62b85ba4f5b1342c42d5e489bec22e868300dd.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/81cf1372bc64c6e97d883caaec62b85ba4f5b1342c42d5e489bec22e868300dd.jpg)

![84b5c39cb080bbd375c56e1aad35344b94c49358da82205d876462451606a9d8.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/84b5c39cb080bbd375c56e1aad35344b94c49358da82205d876462451606a9d8.jpg)

![9123f28e60d4eb0523764fe98782d4671d7e8a17d213821c858359c86f53e753.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/9123f28e60d4eb0523764fe98782d4671d7e8a17d213821c858359c86f53e753.jpg)

![abf43123e7bca4be54b52d0bcae6fde0db982840454df65b56616e3e55e698c0.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/abf43123e7bca4be54b52d0bcae6fde0db982840454df65b56616e3e55e698c0.jpg)

![be51e187dc2094987b950ad869873d903da61b6c3a2f3f6c2a578614a5b5ff4e.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/be51e187dc2094987b950ad869873d903da61b6c3a2f3f6c2a578614a5b5ff4e.jpg)

![f5c11e8fbaf6d3a60cffb8f60b5f9f303149bac9c52c447eb541411f56f26b34.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/f5c11e8fbaf6d3a60cffb8f60b5f9f303149bac9c52c447eb541411f56f26b34.jpg)

![f92ddbeec8c53323efe05fccc5a8e9b8a52a7218be95b1eab06f4e4b34842b51.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/images/f92ddbeec8c53323efe05fccc5a8e9b8a52a7218be95b1eab06f4e4b34842b51.jpg)

### Tables

![4820e8121a73a11af0d6bf57ba66c8ebd34d21326fec16351c5531c481e61b9f.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/tables/4820e8121a73a11af0d6bf57ba66c8ebd34d21326fec16351c5531c481e61b9f.jpg)

![99a6468fadc0d0e7fc48dc8bdaac7fb79305bbf39e195fe6b7052c159f1e70ed.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/tables/99a6468fadc0d0e7fc48dc8bdaac7fb79305bbf39e195fe6b7052c159f1e70ed.jpg)

![9b0e394dc2812e471d7c0a3d74686d44dd29657161426e50af7de434c0a4fcb0.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/tables/9b0e394dc2812e471d7c0a3d74686d44dd29657161426e50af7de434c0a4fcb0.jpg)

![a9efe7eb10fc7c8d17765b1a1abee9a9631844423c74266f0560b1e3e613c878.jpg](../icml_results/3862_Goal-Oriented%20Lower-Tail%20Calibration%20of%20Gaussian%20Processes%20for%20Bayesian%20Optimization/tables/a9efe7eb10fc7c8d17765b1a1abee9a9631844423c74266f0560b1e3e613c878.jpg)

<a id="paper-3823"></a>

## Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models


### Images

![09a3fddb39c8e2763624264b9742b60042b7fd6c59161321877c4fbca02936fb.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/09a3fddb39c8e2763624264b9742b60042b7fd6c59161321877c4fbca02936fb.jpg)

![0feb6b04f746e92ccc625a3cac7c4f464acdcec3132643e669f027c58294c44f.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/0feb6b04f746e92ccc625a3cac7c4f464acdcec3132643e669f027c58294c44f.jpg)

![24962541f47a91347f1b508dee93229aaba5a9a96749850b603fec8d79552487.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/24962541f47a91347f1b508dee93229aaba5a9a96749850b603fec8d79552487.jpg)

![25a9338b0d04c91dc096753d8d4b0a45ecdc9a9f63caa9cf0093c928d8f49871.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/25a9338b0d04c91dc096753d8d4b0a45ecdc9a9f63caa9cf0093c928d8f49871.jpg)

![311286512960ce0ff13680ac5441ebd682f57a6d78562796c241877c71c4fe3a.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/311286512960ce0ff13680ac5441ebd682f57a6d78562796c241877c71c4fe3a.jpg)

![32ff8296d68427e6e7dd8d67f22479562099975107a1d5f0b44a482e63d61aa1.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/32ff8296d68427e6e7dd8d67f22479562099975107a1d5f0b44a482e63d61aa1.jpg)

![364a06766224990b721d897fd4df56e0114acd148ab18a4dd497758eb6cdbddd.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/364a06766224990b721d897fd4df56e0114acd148ab18a4dd497758eb6cdbddd.jpg)

![45654c2acebfe18d3a88b26485da4060db7abe857f8e61c0df62f77fc4e7da44.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/45654c2acebfe18d3a88b26485da4060db7abe857f8e61c0df62f77fc4e7da44.jpg)

![559af412320fc1799fb587b76cc63c44e376757559e367455edb5705ad0ebfc2.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/559af412320fc1799fb587b76cc63c44e376757559e367455edb5705ad0ebfc2.jpg)

![56145b73fb9db4f392ac61929390eeb3a591ce6844bdf673e5911ec95f71a4f7.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/56145b73fb9db4f392ac61929390eeb3a591ce6844bdf673e5911ec95f71a4f7.jpg)

![565d1339eb2a0e4bec626ff17fd0e903c40956aacdb25f0e35dcf62129ecafc1.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/565d1339eb2a0e4bec626ff17fd0e903c40956aacdb25f0e35dcf62129ecafc1.jpg)

![56aeeb0357a7e51fdde9ad7b06839af47cc24a9854085dc2a19d965fb55c52ef.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/56aeeb0357a7e51fdde9ad7b06839af47cc24a9854085dc2a19d965fb55c52ef.jpg)

![58ba0a77fa8f388a202cd59ccf47e9f310f3729578d05d0f0cd7d17bea035259.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/58ba0a77fa8f388a202cd59ccf47e9f310f3729578d05d0f0cd7d17bea035259.jpg)

![5e145f737d3aba694e9e9fbaf78ab33f42ce7d482ff6ec9738174826284d661e.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/5e145f737d3aba694e9e9fbaf78ab33f42ce7d482ff6ec9738174826284d661e.jpg)

![5ff625762979e798ff3e33cb3acf863821213085738868e96ae417904a899c00.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/5ff625762979e798ff3e33cb3acf863821213085738868e96ae417904a899c00.jpg)

![685badcc0c7e9bd092a884f2ed81d9ac1739f76c99c3ccc0e27c884073ea7b20.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/685badcc0c7e9bd092a884f2ed81d9ac1739f76c99c3ccc0e27c884073ea7b20.jpg)

![6cac4fdaab00d1a1c4fc78ed150bd6c15784898c6cb01db47100c1851a79589c.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/6cac4fdaab00d1a1c4fc78ed150bd6c15784898c6cb01db47100c1851a79589c.jpg)

![6d524d2ebf8f8fb0732ecaad17d2846a6bdafcdd85c02d7aff0e1bd6ac1d6cda.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/6d524d2ebf8f8fb0732ecaad17d2846a6bdafcdd85c02d7aff0e1bd6ac1d6cda.jpg)

![70281e47e41acba3f0d61ef2d1009cc9acb2245da70dfc0bf9aa69ca03e7f91c.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/70281e47e41acba3f0d61ef2d1009cc9acb2245da70dfc0bf9aa69ca03e7f91c.jpg)

![7106e1c8c67d4df21abe8b3b8105ccdb372d5ce8135e12cbe0ca6c279e067c47.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/7106e1c8c67d4df21abe8b3b8105ccdb372d5ce8135e12cbe0ca6c279e067c47.jpg)

![71b2b756746f935b98bf5925f1a512b3b002af95503237c32725b2bfedd21df5.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/71b2b756746f935b98bf5925f1a512b3b002af95503237c32725b2bfedd21df5.jpg)

![741d646fa0c45d05322109c41800357ba494917f49a6a16ca3c18c221e1481d2.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/741d646fa0c45d05322109c41800357ba494917f49a6a16ca3c18c221e1481d2.jpg)

![860e75fcad4ae64b3a12142d8d12ebaedb9da4c55c5e7deebd2c8204e29be302.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/860e75fcad4ae64b3a12142d8d12ebaedb9da4c55c5e7deebd2c8204e29be302.jpg)

![8b991a215c020530437a895d4f9c80f78e0e38be768a7763ca1041c5774339dc.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/8b991a215c020530437a895d4f9c80f78e0e38be768a7763ca1041c5774339dc.jpg)

![920e40526e005b46c5475cf014ad64f79a22ecc1a55f04b8d09e89a65db4ae36.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/920e40526e005b46c5475cf014ad64f79a22ecc1a55f04b8d09e89a65db4ae36.jpg)

![a60e41e11bf7eef5f60a0f46583d5d22f32c05395f8ea593636212c57a09a7ea.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/a60e41e11bf7eef5f60a0f46583d5d22f32c05395f8ea593636212c57a09a7ea.jpg)

![ad7c361dd2478e2e85335c97b1c76675a1a9215085ad0f80080402cc9399a78e.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/ad7c361dd2478e2e85335c97b1c76675a1a9215085ad0f80080402cc9399a78e.jpg)

![ba2a186a51129ac04983f68e2240b7424a41ab638db3b1b91b7bda07076e83eb.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/ba2a186a51129ac04983f68e2240b7424a41ab638db3b1b91b7bda07076e83eb.jpg)

![d3f805951b10b48e8e868843bdfad0ce2ab8bcd283079b31206e4b7362c5f4bb.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/d3f805951b10b48e8e868843bdfad0ce2ab8bcd283079b31206e4b7362c5f4bb.jpg)

![d83701b54a60a86f658618af52a3a4863712f2c61ea450bea7e54e01aa165a0e.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/d83701b54a60a86f658618af52a3a4863712f2c61ea450bea7e54e01aa165a0e.jpg)

![dafe7abea407084975522461781539ef364fce956d854406af5308960340881b.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/dafe7abea407084975522461781539ef364fce956d854406af5308960340881b.jpg)

![ea2cfc3c932d16bbb2cc75b79647aa1e3cd233b439ea53b085c719a15a1190ca.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/ea2cfc3c932d16bbb2cc75b79647aa1e3cd233b439ea53b085c719a15a1190ca.jpg)

![ee0777563d5454623d422b73a222497cd8f608562e6044a63f60666b82f9ce94.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/ee0777563d5454623d422b73a222497cd8f608562e6044a63f60666b82f9ce94.jpg)

![fd5431227a5c775a2506c4b0a77668c3ada97cccdf742ee1bc434c6f688fa6e3.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/fd5431227a5c775a2506c4b0a77668c3ada97cccdf742ee1bc434c6f688fa6e3.jpg)

![fe20053b9c5fa8a9b0cd764e0bc60a4a424766c6ea3ff48f43c859142568e2bd.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/images/fe20053b9c5fa8a9b0cd764e0bc60a4a424766c6ea3ff48f43c859142568e2bd.jpg)

### Tables

![121b674e9a3b8f117baabd47e27e7b47b5e8532ae70af04d51f318f2668d2f27.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/tables/121b674e9a3b8f117baabd47e27e7b47b5e8532ae70af04d51f318f2668d2f27.jpg)

![5f85816add5280d1bde4b65ffccf51b0a6b0c18c842b4d7ace29b19270ec74be.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/tables/5f85816add5280d1bde4b65ffccf51b0a6b0c18c842b4d7ace29b19270ec74be.jpg)

![71724dccc93a95c52f8711d486dbc2e8c52bdfd66bb26dc17197d2a738509cdf.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/tables/71724dccc93a95c52f8711d486dbc2e8c52bdfd66bb26dc17197d2a738509cdf.jpg)

![8193a00a2e8a6125598cda9c4ed97320dc9b756dccefcd0e509c9f51676d7ac1.jpg](../icml_results/3863_Masks%20Can%20Be%20Distracting_%20On%20Context%20Comprehension%20in%20Diffusion%20Language%20Models/tables/8193a00a2e8a6125598cda9c4ed97320dc9b756dccefcd0e509c9f51676d7ac1.jpg)

<a id="paper-3824"></a>

## A Diagnostic Study of Multi-Agent LLMs for Real-World Debates


### Images

![0d241ac3b37e218c0739e7daf5251cabac11315c71d8436f412ae175c78ce65e.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/0d241ac3b37e218c0739e7daf5251cabac11315c71d8436f412ae175c78ce65e.jpg)

![46f97c800cb4c6fed4db37b933d68e4774162514f6acce60f6c90d744e6aef29.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/46f97c800cb4c6fed4db37b933d68e4774162514f6acce60f6c90d744e6aef29.jpg)

![602388a76c1015b5f80d884b45b317208b3ff7c7f150ea46104bec6dac8aeb31.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/602388a76c1015b5f80d884b45b317208b3ff7c7f150ea46104bec6dac8aeb31.jpg)

![9973664dc2f94ee636ec0baf00280562264357009876f2f354796b1625bf8a75.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/9973664dc2f94ee636ec0baf00280562264357009876f2f354796b1625bf8a75.jpg)

![a8eb49a158a80638597bb1c55a989523a9a6c1937b629e94884d9bbb5bac9f45.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/a8eb49a158a80638597bb1c55a989523a9a6c1937b629e94884d9bbb5bac9f45.jpg)

![a9a5c40ba2f1b78d768a353aaaaa5c7aebf380500dddb8351b405803a3be62bc.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/images/a9a5c40ba2f1b78d768a353aaaaa5c7aebf380500dddb8351b405803a3be62bc.jpg)

### Tables

![1296ef9e9370de334671a9e767a3b5f1bd789169b880929ae02634bc7f8ed816.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/tables/1296ef9e9370de334671a9e767a3b5f1bd789169b880929ae02634bc7f8ed816.jpg)

![864443662b6b75f9a3b09bb0e54004311a74e51c659ff78d257628a7c973aaa9.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/tables/864443662b6b75f9a3b09bb0e54004311a74e51c659ff78d257628a7c973aaa9.jpg)

![cd592920c9b7b3dad99875813fa0163721b4e6d481e07f68e65ab6888234dcdc.jpg](../icml_results/3864_A%20Diagnostic%20Study%20of%20Multi-Agent%20LLMs%20for%20Real-World%20Debates/tables/cd592920c9b7b3dad99875813fa0163721b4e6d481e07f68e65ab6888234dcdc.jpg)

<a id="paper-3825"></a>

## Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference


### Images

![30902675dc65ba2affb4524ad263978d755fee6623cfbcd3f81aabe388194522.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/30902675dc65ba2affb4524ad263978d755fee6623cfbcd3f81aabe388194522.jpg)

![59f8c8f4ab604f0ec5a61a5eb7896ed065a4093bba720da440de3f047b0d1688.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/59f8c8f4ab604f0ec5a61a5eb7896ed065a4093bba720da440de3f047b0d1688.jpg)

![8ddab6f06f8a2906abf993add3f243c46ef30a817997c3ef905f15fd9694b5af.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/8ddab6f06f8a2906abf993add3f243c46ef30a817997c3ef905f15fd9694b5af.jpg)

![95bbf27124260d5772ef2e082b41d6c62132bca671e51c4826fe9e6cda790bd9.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/95bbf27124260d5772ef2e082b41d6c62132bca671e51c4826fe9e6cda790bd9.jpg)

![b016b541ba14c30c54fc845258babbd221f8d4afe1f0bd169ab23eefba19a472.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/b016b541ba14c30c54fc845258babbd221f8d4afe1f0bd169ab23eefba19a472.jpg)

![ea4939af17cd54db15fb88eeb1bfab197e97b1946057713a55702ff22d79cfc6.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/images/ea4939af17cd54db15fb88eeb1bfab197e97b1946057713a55702ff22d79cfc6.jpg)

### Tables

![1726c5d0f2895ebf8b92cb25c21b087a657d94ff0e59c44fec1cfe596b2dd27d.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/1726c5d0f2895ebf8b92cb25c21b087a657d94ff0e59c44fec1cfe596b2dd27d.jpg)

![3d41de100c5ee7ad2f0b0d1763ea00a879fd974237caaae9afac55352770cfa0.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/3d41de100c5ee7ad2f0b0d1763ea00a879fd974237caaae9afac55352770cfa0.jpg)

![4ad3972ff0f0513a7fdab55ecf9eeb3a6c2d66d6a457c00c9a848ce7f90a94d4.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/4ad3972ff0f0513a7fdab55ecf9eeb3a6c2d66d6a457c00c9a848ce7f90a94d4.jpg)

![6e497ea788a6394678407d1c3496c659022827eb1c2b8207256685a11ea387b6.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/6e497ea788a6394678407d1c3496c659022827eb1c2b8207256685a11ea387b6.jpg)

![975fa1c260879ce83009c3843240a3ca5afae140d3d7eb0953d20f1a44c3e6d5.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/975fa1c260879ce83009c3843240a3ca5afae140d3d7eb0953d20f1a44c3e6d5.jpg)

![bcea3a1b92f42c07094fede17febc732f40c6ef9849979b77a0107b5c534ec28.jpg](../icml_results/3865_Tailoring%20Strictly%20Proper%20Scoring%20Rules%20for%20Downstream%20Tasks_%20An%20Application%20to%20Causal%20Inference/tables/bcea3a1b92f42c07094fede17febc732f40c6ef9849979b77a0107b5c534ec28.jpg)

<a id="paper-3826"></a>

## Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search


### Images

![063ac948e1835936c70616ad3eba8d6e4c252ce670263cdffc6db8678b5f20ae.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/063ac948e1835936c70616ad3eba8d6e4c252ce670263cdffc6db8678b5f20ae.jpg)

![08abb39d5b50aa0981c399217c590b3983f69de2588c1891c16f1561c9e5848d.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/08abb39d5b50aa0981c399217c590b3983f69de2588c1891c16f1561c9e5848d.jpg)

![0ee5c7d835aeecfd43695f86cf16df4610714067b218c711097c9b0b60e527e5.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/0ee5c7d835aeecfd43695f86cf16df4610714067b218c711097c9b0b60e527e5.jpg)

![15fb029b173b7e9429e684d50386de71198944eee05f651135aa1397d54e242f.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/15fb029b173b7e9429e684d50386de71198944eee05f651135aa1397d54e242f.jpg)

![1a69ee1f30cd10ff3e8258c51bf352283c3e9cf63346a78ac4e93edda2112d30.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/1a69ee1f30cd10ff3e8258c51bf352283c3e9cf63346a78ac4e93edda2112d30.jpg)

![1f7a7056a22a16433b497b9c7ad9303c53817de8cb1705dc5f53e14ef99a4e23.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/1f7a7056a22a16433b497b9c7ad9303c53817de8cb1705dc5f53e14ef99a4e23.jpg)

![2b7b0dd3014b3594571664694c490005c5bd340059b6c8366cad07524a080340.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/2b7b0dd3014b3594571664694c490005c5bd340059b6c8366cad07524a080340.jpg)

![309badfe96271552f05447b420e9e34509bebe1b9979de888900525eba098d5e.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/309badfe96271552f05447b420e9e34509bebe1b9979de888900525eba098d5e.jpg)

![3d88298e45522deaef6e52d629f995915ea51d5536c304035401b11461d5933f.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/3d88298e45522deaef6e52d629f995915ea51d5536c304035401b11461d5933f.jpg)

![3e16037941bc00ceeb23abb4c68324fcb93b9d0af9cc69c9495dfc5c3070c2bb.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/3e16037941bc00ceeb23abb4c68324fcb93b9d0af9cc69c9495dfc5c3070c2bb.jpg)

![3e30364fcaf9a0943276f89f0531e4de0ded9016ab5becda9c60d67b05d8b894.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/3e30364fcaf9a0943276f89f0531e4de0ded9016ab5becda9c60d67b05d8b894.jpg)

![4057e8c7c391926a4f8f43848c6175ea5e86ebc73a7e6ef6695752be3fe0e4ab.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/4057e8c7c391926a4f8f43848c6175ea5e86ebc73a7e6ef6695752be3fe0e4ab.jpg)

![62db0fbc277547c123e5e5eccfc3ee79afbdc77cb026da66025ff193e62a2269.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/62db0fbc277547c123e5e5eccfc3ee79afbdc77cb026da66025ff193e62a2269.jpg)

![6385067d7fd88ab51892ae81fe3e2c43b5149d97121d510232413d8f025f582b.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/6385067d7fd88ab51892ae81fe3e2c43b5149d97121d510232413d8f025f582b.jpg)

![7a86bf7e17a5b61e71a163ca7bc2a67f5423bb7b15a80cefa5b9a2c2cf26baaf.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/7a86bf7e17a5b61e71a163ca7bc2a67f5423bb7b15a80cefa5b9a2c2cf26baaf.jpg)

![7ba49e115683051aeefc0f6d12f9a7b486d14ee07968f0e054b935c50e02f195.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/7ba49e115683051aeefc0f6d12f9a7b486d14ee07968f0e054b935c50e02f195.jpg)

![7fbfc857594402dd22d7ac3d86099a7f09d38e3e2328bf31cb71f60dd5e59d6d.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/7fbfc857594402dd22d7ac3d86099a7f09d38e3e2328bf31cb71f60dd5e59d6d.jpg)

![817bd2df8520922b4c0d1a3cc2a8ef6a9bd8015deb172d63b952f5de4c64d5de.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/817bd2df8520922b4c0d1a3cc2a8ef6a9bd8015deb172d63b952f5de4c64d5de.jpg)

![8b3f956e85dd2aa4550f164a5f03afa461b4b380f694a8191b5cf1737088e203.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/8b3f956e85dd2aa4550f164a5f03afa461b4b380f694a8191b5cf1737088e203.jpg)

![8c0c4e461cd2a78730a76a1ee04e787b9e9f4fbda2421473b9c3eaa142ec5ca0.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/8c0c4e461cd2a78730a76a1ee04e787b9e9f4fbda2421473b9c3eaa142ec5ca0.jpg)

![90dc12147abada6d8ac5d9a9bd3a9d83bc4e2d584c681b3e77975bcc1b1e3f75.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/90dc12147abada6d8ac5d9a9bd3a9d83bc4e2d584c681b3e77975bcc1b1e3f75.jpg)

![a8a0bbe4e44084028e58a07c7f818bf6bbe424358593cf4ccd639208ccd30bc8.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/a8a0bbe4e44084028e58a07c7f818bf6bbe424358593cf4ccd639208ccd30bc8.jpg)

![d069fa26559b3cdeb4c7cf0b15f3b10bd03c06468e130f4e2d36eb1a30e35441.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/d069fa26559b3cdeb4c7cf0b15f3b10bd03c06468e130f4e2d36eb1a30e35441.jpg)

![e26aefd10326cedee6d057cb0d09a23003f1065183ca41642d0530b75bf269c4.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/images/e26aefd10326cedee6d057cb0d09a23003f1065183ca41642d0530b75bf269c4.jpg)

### Tables

![1fb39ed6b792602d97dc6a1f468a9fefd940a174e919b518afb5e3a281eb4c64.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/1fb39ed6b792602d97dc6a1f468a9fefd940a174e919b518afb5e3a281eb4c64.jpg)

![1ffa6dbd96aef7935ed6ec0ff704419cb37e828142ff5a3ffb4725f6ea6a6cfb.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/1ffa6dbd96aef7935ed6ec0ff704419cb37e828142ff5a3ffb4725f6ea6a6cfb.jpg)

![31187cefe9f9e5255e54926feb636ec54345715f2490c9fafe671356ed37299e.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/31187cefe9f9e5255e54926feb636ec54345715f2490c9fafe671356ed37299e.jpg)

![354020b4508d55ab77c5f4ab382fb12316bb04aca803d5c373667e725ab57adf.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/354020b4508d55ab77c5f4ab382fb12316bb04aca803d5c373667e725ab57adf.jpg)

![49afe4886381a84db1237b251636a09bf350696ea11eff5238df7b5cc3e007ef.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/49afe4886381a84db1237b251636a09bf350696ea11eff5238df7b5cc3e007ef.jpg)

![75e830a8b4917b81d2cadb822d3c9c3e88756828256e78f51a7218bf569f8649.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/75e830a8b4917b81d2cadb822d3c9c3e88756828256e78f51a7218bf569f8649.jpg)

![843e7c015cbf29251e941c8fdd7a51ed85b61d4f46544a7cd551783396d03072.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/843e7c015cbf29251e941c8fdd7a51ed85b61d4f46544a7cd551783396d03072.jpg)

![9a48b004d8b7258245a559849345a71340144703ba6885577eedbd553d4469ef.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/9a48b004d8b7258245a559849345a71340144703ba6885577eedbd553d4469ef.jpg)

![a1c5edd83944111002731a2b15d5846b18516a370134f4632b281c5a15df81f2.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/a1c5edd83944111002731a2b15d5846b18516a370134f4632b281c5a15df81f2.jpg)

![bd0d88ed3864faddb05bda812d6fafe0bc801b94c267b2acae428d0786502df8.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/bd0d88ed3864faddb05bda812d6fafe0bc801b94c267b2acae428d0786502df8.jpg)

![dc899b6866887f6babf3744156baabc0cfc0c96257925c2cba5fcedc215cc4b5.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/dc899b6866887f6babf3744156baabc0cfc0c96257925c2cba5fcedc215cc4b5.jpg)

![e0ab142b4e61762ef785207ab3cbd3be50b063d3fea48b7b83f026f54b046ef7.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/e0ab142b4e61762ef785207ab3cbd3be50b063d3fea48b7b83f026f54b046ef7.jpg)

![fa570d922e42e7761f8e47b117ccc1433916b53fdc3df83676ddcaedc07f835e.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/fa570d922e42e7761f8e47b117ccc1433916b53fdc3df83676ddcaedc07f835e.jpg)

![fee156ab41ac103e76c9676939017b7c90d10f12889755af9127c69542070d02.jpg](../icml_results/3866_Exposing%20Hidden%20Biases%20in%20Text-to-Image%20Models%20via%20Automated%20Prompt%20Search/tables/fee156ab41ac103e76c9676939017b7c90d10f12889755af9127c69542070d02.jpg)

<a id="paper-3827"></a>

## Capability Traps in DPO


### Images

![210c397cf4415891f3778ffa3127ddd07a11ed08bbdb00a17d8a4f8a0ae663b8.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/210c397cf4415891f3778ffa3127ddd07a11ed08bbdb00a17d8a4f8a0ae663b8.jpg)

![6a5b63cfc8d089b474b18f0585c34662e02c90266722a76adc7292fcc544470a.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/6a5b63cfc8d089b474b18f0585c34662e02c90266722a76adc7292fcc544470a.jpg)

![9cbb469db7a7a921f02eecdd01f6aec06803927729dd484ff710b3e4b52c2d25.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/9cbb469db7a7a921f02eecdd01f6aec06803927729dd484ff710b3e4b52c2d25.jpg)

![e02b119547fa889a9f4990649fdd0f30a011341124d6f5cd010f13f12e380924.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/e02b119547fa889a9f4990649fdd0f30a011341124d6f5cd010f13f12e380924.jpg)

![efc704838abc4119e5d0882902204801563190e749051559b52e678760043759.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/efc704838abc4119e5d0882902204801563190e749051559b52e678760043759.jpg)

![f25af38f96bdfe82d48c1bf64bcb1121e05ed5a8ed310d97eac3d722e2391191.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/images/f25af38f96bdfe82d48c1bf64bcb1121e05ed5a8ed310d97eac3d722e2391191.jpg)

### Tables

![0a97dd83f5bda25d49cd2bd1e2d57e4b11ec50c44da7ef623bd1b539c46a7873.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/0a97dd83f5bda25d49cd2bd1e2d57e4b11ec50c44da7ef623bd1b539c46a7873.jpg)

![202cb1b0d35fd09fa4065dc0c8053306604b5671328fa7e7bda1df3f33b721a7.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/202cb1b0d35fd09fa4065dc0c8053306604b5671328fa7e7bda1df3f33b721a7.jpg)

![22959d0f10da6ba160083823ce349c1abad4ae58360bbf00e056f9e9dd83cdc3.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/22959d0f10da6ba160083823ce349c1abad4ae58360bbf00e056f9e9dd83cdc3.jpg)

![3c5689c8f9e21d316aeda597c9865d0d51a1a3a972ff26bd2c9c87344d6079ae.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/3c5689c8f9e21d316aeda597c9865d0d51a1a3a972ff26bd2c9c87344d6079ae.jpg)

![40d814a99030c3c0602f2fa905b6615d4219c2f060c8b5db771d50e4ea96a73f.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/40d814a99030c3c0602f2fa905b6615d4219c2f060c8b5db771d50e4ea96a73f.jpg)

![6d2168ea913a12656a589c980a18a12abafe9c26a88c6a0294dfdee9d40a98ee.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/6d2168ea913a12656a589c980a18a12abafe9c26a88c6a0294dfdee9d40a98ee.jpg)

![7485aab7787f2ca36f834a64d4bdbf69ab7a49cdbfc3bc5c3c1d009c5d309bac.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/7485aab7787f2ca36f834a64d4bdbf69ab7a49cdbfc3bc5c3c1d009c5d309bac.jpg)

![8bf43b61bb6038d4bd4354531a37c4b82cfb4c98ae648a1e7198ef8ec082ae58.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/8bf43b61bb6038d4bd4354531a37c4b82cfb4c98ae648a1e7198ef8ec082ae58.jpg)

![a9aed5c95691e8a91a1ff4db0710fc98620e71c11101942235c7056fe9b4cb39.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/a9aed5c95691e8a91a1ff4db0710fc98620e71c11101942235c7056fe9b4cb39.jpg)

![b225c2ac4bdce888b547af2b83faf0cc8b0de63551d190b5019673cbc67e0c65.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/b225c2ac4bdce888b547af2b83faf0cc8b0de63551d190b5019673cbc67e0c65.jpg)

![be677234caaf54ad3ccf4dc089f578c1800eec6876cee8d51532283cb840bc29.jpg](../icml_results/3867_Capability%20Traps%20in%20DPO/tables/be677234caaf54ad3ccf4dc089f578c1800eec6876cee8d51532283cb840bc29.jpg)

<a id="paper-3828"></a>

## Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models

### Images

![054e3579525c1e60aabc42cd56fdce99e4242c8be0005a58dfc559e24115828d.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/054e3579525c1e60aabc42cd56fdce99e4242c8be0005a58dfc559e24115828d.jpg)

![0eca3e83b59f511462808389d0bf2f0c7c1303c53395b6e77e2ebc88db0aa7d5.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/0eca3e83b59f511462808389d0bf2f0c7c1303c53395b6e77e2ebc88db0aa7d5.jpg)

![13efd0b8af64037972e07295a3322853c9af45239a25b10896ae482c1231c654.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/13efd0b8af64037972e07295a3322853c9af45239a25b10896ae482c1231c654.jpg)

![14c3f8e36e4562f5008f4551b4233f334012c584ed2e02d7a8e1e1c725d68fea.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/14c3f8e36e4562f5008f4551b4233f334012c584ed2e02d7a8e1e1c725d68fea.jpg)

![16834a4f67524a5f27eff57eb7cdcac75b633bf2bc326265eddea088c26f4e3e.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/16834a4f67524a5f27eff57eb7cdcac75b633bf2bc326265eddea088c26f4e3e.jpg)

![1874dbd41fb218b5b1c4bc8d9c5ed60067153175bce4d1e18c3f0f208c2632d2.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/1874dbd41fb218b5b1c4bc8d9c5ed60067153175bce4d1e18c3f0f208c2632d2.jpg)

![1ac1f97cf24fae636bb33eb4854dfdddc7ace20d9dc1f6d7957adf33bcc95a67.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/1ac1f97cf24fae636bb33eb4854dfdddc7ace20d9dc1f6d7957adf33bcc95a67.jpg)

![1f14fbaef407ca2cdfaf67d1e21fbea446510cbecd226e5ddb4d6bb8b5f414a9.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/1f14fbaef407ca2cdfaf67d1e21fbea446510cbecd226e5ddb4d6bb8b5f414a9.jpg)

![2d9a08bd3232e4ceef2f3c44ff498bc37f9af72985976b70ddd86410fc9eb53e.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/2d9a08bd3232e4ceef2f3c44ff498bc37f9af72985976b70ddd86410fc9eb53e.jpg)

![2ed539e5c139d0633d638b90f8d10a6d1a94e39cd76541fb6972b243c8ef1b82.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/2ed539e5c139d0633d638b90f8d10a6d1a94e39cd76541fb6972b243c8ef1b82.jpg)

![340ea527b503e370001b6b7fba65a8a1abad0b95a771875b33abb0d1fd1be463.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/340ea527b503e370001b6b7fba65a8a1abad0b95a771875b33abb0d1fd1be463.jpg)

![3768303f0d5ce3cac26f1401d76dfc1b5ea54bdaf20771feea4e040b415583f8.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/3768303f0d5ce3cac26f1401d76dfc1b5ea54bdaf20771feea4e040b415583f8.jpg)

![39862cf9ee551e785da57d42cbdb2be7a43dbb7ba80d68e101e957924c4e8268.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/39862cf9ee551e785da57d42cbdb2be7a43dbb7ba80d68e101e957924c4e8268.jpg)

![4e560a18dca138919a48aa27061927be90a6828afca4f6f6b299827373d79657.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/4e560a18dca138919a48aa27061927be90a6828afca4f6f6b299827373d79657.jpg)

![6e63f4c3d6ddd0421bb75a0baffccdebd384e8fef4f6948746837a26f92b6015.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/6e63f4c3d6ddd0421bb75a0baffccdebd384e8fef4f6948746837a26f92b6015.jpg)

![8070b2b3cd4cb6b1c35db6bb33197662ccf21b2292143160e838eebc6df3fc2f.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/8070b2b3cd4cb6b1c35db6bb33197662ccf21b2292143160e838eebc6df3fc2f.jpg)

![80d7f2a7e4ac634c1d844789e183e7a945f54dbebda3b05e40b9a2cae7770e95.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/80d7f2a7e4ac634c1d844789e183e7a945f54dbebda3b05e40b9a2cae7770e95.jpg)

![852a893d83f33dcbab95189cbb1625d12a081dcc46b7faca67a7842183b53b2d.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/852a893d83f33dcbab95189cbb1625d12a081dcc46b7faca67a7842183b53b2d.jpg)

![8b5387116dfcfcccaaeb7d4da5e84b46c5ade18417767023a4edb1b0af675ea8.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/8b5387116dfcfcccaaeb7d4da5e84b46c5ade18417767023a4edb1b0af675ea8.jpg)

![8d7d38364fe1f9857c9fb4c1764819bfc402fdb3be2fac527b625fd366345d85.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/8d7d38364fe1f9857c9fb4c1764819bfc402fdb3be2fac527b625fd366345d85.jpg)

![92b0d5f85b12030d3b7f1d67181b650917317ff2a8c3570a515c5dab43d8872a.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/92b0d5f85b12030d3b7f1d67181b650917317ff2a8c3570a515c5dab43d8872a.jpg)

![9dff3b1b80daf33fd35de77e85a24fe3d911eabd4b338f18a512e2bf59770f51.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/9dff3b1b80daf33fd35de77e85a24fe3d911eabd4b338f18a512e2bf59770f51.jpg)

![9ed58ed90da6c32205eebb7afb82334e9ab116923734d774b69f849659695ce4.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/9ed58ed90da6c32205eebb7afb82334e9ab116923734d774b69f849659695ce4.jpg)

![a21e422d03b37884d1ab8f5c6b41a5fbe81e0e771a711061f4fd1edf22aaecc1.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/a21e422d03b37884d1ab8f5c6b41a5fbe81e0e771a711061f4fd1edf22aaecc1.jpg)

![a781ac6b5fed55176715872e0902a1eb0d45c729b1daa8aad5c2a5aa21dee046.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/a781ac6b5fed55176715872e0902a1eb0d45c729b1daa8aad5c2a5aa21dee046.jpg)

![a8c831fa8d26d67a6be1314292362dc341b228b2c37a1e5a71d5073fb0d464b9.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/a8c831fa8d26d67a6be1314292362dc341b228b2c37a1e5a71d5073fb0d464b9.jpg)

![a95d7646b3085548238db58e934766842e9ac76bb9deba0aacb9f052c1366540.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/a95d7646b3085548238db58e934766842e9ac76bb9deba0aacb9f052c1366540.jpg)

![ad653f0a7e10f4f099d662ee9e8cdb7dccb029844ae7adae7b6cc90aac7aaace.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/ad653f0a7e10f4f099d662ee9e8cdb7dccb029844ae7adae7b6cc90aac7aaace.jpg)

![ad85afcbbf7a046ed27eb3ed0126c4be7e438adce51de7ab48110578daf6427a.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/ad85afcbbf7a046ed27eb3ed0126c4be7e438adce51de7ab48110578daf6427a.jpg)

![af92865e1fcf5af38bce93c84bdcb71910ef05fdf0189f4846d17e4cb8732170.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/af92865e1fcf5af38bce93c84bdcb71910ef05fdf0189f4846d17e4cb8732170.jpg)

![c1d2561ad6c436212f60a9de98cf6f22b51d7dc2abb54c6da74ca4ed8ac5b873.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/c1d2561ad6c436212f60a9de98cf6f22b51d7dc2abb54c6da74ca4ed8ac5b873.jpg)

![c76d310f04d111a1838dec3cc741de448e06da06ea5806a9081b366e5eeeabd7.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/c76d310f04d111a1838dec3cc741de448e06da06ea5806a9081b366e5eeeabd7.jpg)

![d813ccf9766484b3b083a4e583284f84bbef7473a094462332a3f970db915a19.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/d813ccf9766484b3b083a4e583284f84bbef7473a094462332a3f970db915a19.jpg)

![e931c1a2bd44a6957c5457314c0b55bcf5c257ed57fc8b745ac5432df88dcbf4.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/e931c1a2bd44a6957c5457314c0b55bcf5c257ed57fc8b745ac5432df88dcbf4.jpg)

![ecdd98e49b7557cee707c1b7467ef69cb30e0d2bfddcab0a1dff72749af01a76.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/ecdd98e49b7557cee707c1b7467ef69cb30e0d2bfddcab0a1dff72749af01a76.jpg)

![ee39edfa2a5084bf3d8bb9fbfe7a0a881b33521684798d4e9110ac2bf77bbe14.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/ee39edfa2a5084bf3d8bb9fbfe7a0a881b33521684798d4e9110ac2bf77bbe14.jpg)

![f629d38cff2f87f1742fa4ea7c694f76e50b8818b86965ac4d46747ddf492947.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/images/f629d38cff2f87f1742fa4ea7c694f76e50b8818b86965ac4d46747ddf492947.jpg)

### Tables

![053a059bee97dff6a7fa3b0c9702ba13637ad60c6c9ced26d6732a31f965fd34.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/053a059bee97dff6a7fa3b0c9702ba13637ad60c6c9ced26d6732a31f965fd34.jpg)

![07d997f1f49e5d3e61dae2fdaa5222b5cd53be0cbdb1584089f0fd92ca40e501.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/07d997f1f49e5d3e61dae2fdaa5222b5cd53be0cbdb1584089f0fd92ca40e501.jpg)

![504830134b1810409ce07698bf5e66f4a328ad14f17c8de2dbe2fb3f6be820f6.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/504830134b1810409ce07698bf5e66f4a328ad14f17c8de2dbe2fb3f6be820f6.jpg)

![5411f102d18a02bd91b36a43120e9862999b4741dec35db3418047223cde4487.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/5411f102d18a02bd91b36a43120e9862999b4741dec35db3418047223cde4487.jpg)

![5561baa35f4322854d6d0d58da0f9538db99c777179cb3ec9d9e72a17e924ba5.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/5561baa35f4322854d6d0d58da0f9538db99c777179cb3ec9d9e72a17e924ba5.jpg)

![627adf7c726fc73f58c9fe847a11d45aa3d7a579e15d8be1561fc1119b51a98a.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/627adf7c726fc73f58c9fe847a11d45aa3d7a579e15d8be1561fc1119b51a98a.jpg)

![6923cd610710c5c1cc7cecc820b32f316ce0922c74aa7c03e675a6c7ce08c814.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/6923cd610710c5c1cc7cecc820b32f316ce0922c74aa7c03e675a6c7ce08c814.jpg)

![783295fab955de3bda761a358a0920d0257bb22ea265fa37883dc4ecdd696f0e.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/783295fab955de3bda761a358a0920d0257bb22ea265fa37883dc4ecdd696f0e.jpg)

![ad5df4ac9b7f9881daa86ce9923a86fbf952a89d7cb24e3602460d0be36ed511.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/ad5df4ac9b7f9881daa86ce9923a86fbf952a89d7cb24e3602460d0be36ed511.jpg)

![b9a6fafd121335e3205460de23bc9fca92acf7b4d67e69dbf16a03c44854b514.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/b9a6fafd121335e3205460de23bc9fca92acf7b4d67e69dbf16a03c44854b514.jpg)

![bd31ac69992d256b8b646aa8262d0f4ccab0a659e82ec143e4c6962bbb49ba19.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/bd31ac69992d256b8b646aa8262d0f4ccab0a659e82ec143e4c6962bbb49ba19.jpg)

![db5b1c41c4d161918ad67e2234b013a5d2a6a075d06a4114971ec27107b3023f.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/db5b1c41c4d161918ad67e2234b013a5d2a6a075d06a4114971ec27107b3023f.jpg)

![ded51ee65a9ea5f19ae723e49678ca9ef3436079544f1090bf78a793ca4efd0f.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/ded51ee65a9ea5f19ae723e49678ca9ef3436079544f1090bf78a793ca4efd0f.jpg)

![e81242b40bd2e97f864c187a793496aa6b3b4759b61bccc9c614bb2edada4e83.jpg](../icml_results/3868_Induction%20Meets%20Biology_%20Mechanisms%20of%20Repeat%20Detection%20in%20Protein%20Language%20Models/tables/e81242b40bd2e97f864c187a793496aa6b3b4759b61bccc9c614bb2edada4e83.jpg)
