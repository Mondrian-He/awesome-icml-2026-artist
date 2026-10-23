# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 115 of 197

## 目录 (Table of Contents)

1. [Floating-Point Networks with Automatic Differentiation Can Represent Almost All Floating-Point Functions and Their Gradients](#paper-3763)
2. [On Expressive Power of Floating-Point Transformers](#paper-3764)
3. [STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning](#paper-3765)
4. [Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions](#paper-3766)
5. [Scalable Reinforcement Learning via Adaptive Batch Scaling](#paper-3767)
6. [Quantile-Free Uncertainty Quantification in Graph Neural Networks](#paper-3768)
7. [SeisMark: A Large-Scale Open Benchmark for Robust 3D Seismic Fault Detection](#paper-3769)
8. [Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](#paper-3770)
9. [Generative Neural Operators through Diffusion Last Layer](#paper-3771)
10. [TsLLM: Augmenting LLMs for General Time Series Understanding and Prediction](#paper-3772)
11. [Deep networks learn to parse uniform-depth context-free languages from local statistics](#paper-3773)
12. [Constrained Adaptive Rejection Sampling](#paper-3774)
13. [Insertion Based Sequence Generation with Learnable Order Dynamics](#paper-3775)
14. [OGPO: Sample Efficient Full-Finetuning of Generative Control Policies](#paper-3776)
15. [Assistive Prompt Mediation: Evaluating Language Models Under Accessibility Constraints](#paper-3777)
16. [Riemannian stochastic optimization for sufficient dimension reduction](#paper-3778)
17. [APEX: Approximate-but-exhaustive search for ultra-large combinatorial synthesis libraries](#paper-3779)
18. [FunPhase: A Periodic Functional Autoencoder for Motion Generation via Phase Manifolds](#paper-3780)
19. [When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning](#paper-3781)
20. [Hyperbolic Neural Operator](#paper-3782)
21. [CURE: Context-driven Diffusion with Progressive Expansion for Single Domain Generalization in Time Series Classification](#paper-3783)
22. [Great Minds Think Alike: Contextual Tacit Communication for Decentralized LLM-Agent Cooperation](#paper-3784)
23. [Neural Minimum Weight Perfect Matching for Quantum Error Codes](#paper-3785)
24. [CAMP: Coherent Alignment of Multimodal Prototypes for Explainable Complementary Learning](#paper-3786)
25. [Privileged Information Distillation for Language Models](#paper-3787)
26. [Recursive Binding on a Budget: Subspace Carving in Order-$p$ Tensor Memories](#paper-3788)
27. [Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](#paper-3789)
28. [Measuring and Mitigating Post-Hoc Rationalization in Reverse Chain-of-Thought Generation](#paper-3790)
29. [CauScale: Neural Causal Discovery at Scale](#paper-3791)
30. [LoPhyDA: Low-Rank Tensor and Physics Gradient Guided Diffusion for Atmospheric Data Assimilation](#paper-3792)
31. [Sheaf Neural Networks on SPD Manifolds: Second-Order Geometric Representation Learning](#paper-3793)
32. [TACTIC: Task-Aware Sparse Coordination Graphs for Multi-Task Multi-agent Reinforcement Learning](#paper-3794)
33. [MIST: Moment-Aligned Invariant Stability Transform for Robust Flow Matching](#paper-3795)

---


<a id="paper-3763"></a>

## Floating-Point Networks with Automatic Differentiation Can Represent Almost All Floating-Point Functions and Their Gradients

### Images

![232c0c46672a29e324872c11eb19892b7905a40c7d75260ecd1919d2d9f80ff0.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/232c0c46672a29e324872c11eb19892b7905a40c7d75260ecd1919d2d9f80ff0.jpg)

![35bed1cd8dc035bfb1b9efc280e37feff0cbd030197224dc7eb42116a0c0032a.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/35bed1cd8dc035bfb1b9efc280e37feff0cbd030197224dc7eb42116a0c0032a.jpg)

![53dd29cd8f7e0fdb50590a0d8dacbfe0fc02a7631e25db2d607f668f7804bebb.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/53dd29cd8f7e0fdb50590a0d8dacbfe0fc02a7631e25db2d607f668f7804bebb.jpg)

![67658730e775ab025cb325674cb4490a7ff667050dc1ad82da669d3623434e74.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/67658730e775ab025cb325674cb4490a7ff667050dc1ad82da669d3623434e74.jpg)

![7fe2eaea5bdbd3aedd98448a0bd323fe19cbc9be664cc5ba3bb9aa73be77a874.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/7fe2eaea5bdbd3aedd98448a0bd323fe19cbc9be664cc5ba3bb9aa73be77a874.jpg)

![8b75756cc855167e3c6cf4e0d40d3b5c876d3bf2f1cdfac331b7266c704cfd6f.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/8b75756cc855167e3c6cf4e0d40d3b5c876d3bf2f1cdfac331b7266c704cfd6f.jpg)

![9888d91af2977cafaaf554b2be310cce553ed520a99f24aaa3be6fbd0c392291.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/9888d91af2977cafaaf554b2be310cce553ed520a99f24aaa3be6fbd0c392291.jpg)

![b1ed6b7936318844ffc72836195e5a6e542bf071f5363d515afbfdb73cfc7b36.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/b1ed6b7936318844ffc72836195e5a6e542bf071f5363d515afbfdb73cfc7b36.jpg)

![d2cbe734b2444a2b21f81d5f0936c2e20fad53f8ce9d3188d8083ddb3d190939.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/d2cbe734b2444a2b21f81d5f0936c2e20fad53f8ce9d3188d8083ddb3d190939.jpg)

![e2bc3606399a01451e4e2dbcf7c2e7c4e808eeaee55e82d4f0ef0d7105cca756.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/e2bc3606399a01451e4e2dbcf7c2e7c4e808eeaee55e82d4f0ef0d7105cca756.jpg)

![e8e1f5eba91e6fecd197cf9b2f613c7bbca9010ca198891edbed06130b115120.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/e8e1f5eba91e6fecd197cf9b2f613c7bbca9010ca198891edbed06130b115120.jpg)

### Tables

![0be6d92d9ab5c273eb9894a17bf3784eb5697ce1d6502c69d122aaa6f15f8c68.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/0be6d92d9ab5c273eb9894a17bf3784eb5697ce1d6502c69d122aaa6f15f8c68.jpg)

![32ff13bac62d79cbd08f57e02474c2d004af066ad3df0eae843aeb77ea9ee4aa.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/32ff13bac62d79cbd08f57e02474c2d004af066ad3df0eae843aeb77ea9ee4aa.jpg)

![3c7d6b3d6853505cc3a0e7e76b93b9b2f4b575992f4fe1c3d4e6c2a11dfc1341.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/3c7d6b3d6853505cc3a0e7e76b93b9b2f4b575992f4fe1c3d4e6c2a11dfc1341.jpg)

![489c9982e8b0f96a6fc8338d4648f031177596ea125931d942941d43b5ddf768.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/489c9982e8b0f96a6fc8338d4648f031177596ea125931d942941d43b5ddf768.jpg)

![75afb0e95acd5a5fba09484f5cbde951064db4e5eed0e82c3a9a0b2b3535f88c.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/75afb0e95acd5a5fba09484f5cbde951064db4e5eed0e82c3a9a0b2b3535f88c.jpg)

![8d4e69edd3608336d52495390200f66fe5237e0bbba673960b02897b2279e84a.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/8d4e69edd3608336d52495390200f66fe5237e0bbba673960b02897b2279e84a.jpg)

![c8faa0a096e28d36cd1ea5aa6fe76b9b12db0fb20ac632e71d6d61eec30c5efb.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/c8faa0a096e28d36cd1ea5aa6fe76b9b12db0fb20ac632e71d6d61eec30c5efb.jpg)

![ef1a141f610ce69f1f42e2d379e061c195638e0a6a3ac64efecf08873e901bab.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/ef1a141f610ce69f1f42e2d379e061c195638e0a6a3ac64efecf08873e901bab.jpg)

## Floating-Point Networks with Automatic Differentiation Can Represent Almost All Floating-Point Functions and Their Gradients


### Tables

![6af474dce57ddfe8310e1e03a0dc5b6fdc3d20ae47e065fbdf6202905c89e6f9.jpg](../icml_results/3803_Floating-Point%20Networks%20with%20Automatic%20Differentiation%20Can%20Represent%20Almost%20All%20Floating-Point%20Funct/tables/6af474dce57ddfe8310e1e03a0dc5b6fdc3d20ae47e065fbdf6202905c89e6f9.jpg)

![767a3673808a57e5de2a696d3f0bc1c88952865acb9f93476992166f5b123688.jpg](../icml_results/3803_Floating-Point%20Networks%20with%20Automatic%20Differentiation%20Can%20Represent%20Almost%20All%20Floating-Point%20Funct/tables/767a3673808a57e5de2a696d3f0bc1c88952865acb9f93476992166f5b123688.jpg)

![a0de6c46662a682ddb60cda409c73acb656731acd5793fa5b67a934b14641833.jpg](../icml_results/3803_Floating-Point%20Networks%20with%20Automatic%20Differentiation%20Can%20Represent%20Almost%20All%20Floating-Point%20Funct/tables/a0de6c46662a682ddb60cda409c73acb656731acd5793fa5b67a934b14641833.jpg)

![dc4cfd57d9cf37873ad0ddedbe5c6b41a58efb4a0218185ac8560119715e9838.jpg](../icml_results/3803_Floating-Point%20Networks%20with%20Automatic%20Differentiation%20Can%20Represent%20Almost%20All%20Floating-Point%20Funct/tables/dc4cfd57d9cf37873ad0ddedbe5c6b41a58efb4a0218185ac8560119715e9838.jpg)

<a id="paper-3764"></a>

## On Expressive Power of Floating-Point Transformers


### Tables

![5ed44d009afc4acbd5996b82a5129d88cfbedf82ebaaa5be6adc55dd9de7ec1d.jpg](../icml_results/3804_On%20Expressive%20Power%20of%20Floating-Point%20Transformers/tables/5ed44d009afc4acbd5996b82a5129d88cfbedf82ebaaa5be6adc55dd9de7ec1d.jpg)

<a id="paper-3765"></a>

## STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning


### Images

![1e45d8dfb6c4c869c208a7269f6b9dd00deca3bbc6e21e7b68b5bd9bda9609bb.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/1e45d8dfb6c4c869c208a7269f6b9dd00deca3bbc6e21e7b68b5bd9bda9609bb.jpg)

![393cda4b37a461f5b3ccb40b5c460d5645ebeca0b44a081643bcd97a229ae88b.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/393cda4b37a461f5b3ccb40b5c460d5645ebeca0b44a081643bcd97a229ae88b.jpg)

![4761574a97c5454a6d319e7089da79010469ff6cc329a09b746c025c68bdeb09.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/4761574a97c5454a6d319e7089da79010469ff6cc329a09b746c025c68bdeb09.jpg)

![496f7810422bea0ed67482bf9d75eaee38e505ee4baefc7a386c2690e3507781.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/496f7810422bea0ed67482bf9d75eaee38e505ee4baefc7a386c2690e3507781.jpg)

![7a09f14bf80ebce50cd6276609a906325292f0730ecb7f75981acf6d193e8b61.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/7a09f14bf80ebce50cd6276609a906325292f0730ecb7f75981acf6d193e8b61.jpg)

![ad563b0bd726b83daf516335388a67b0326edfb72f54260b6714c4348538d8c7.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/ad563b0bd726b83daf516335388a67b0326edfb72f54260b6714c4348538d8c7.jpg)

![c5cf74a3f0d8585a7383348726bf8dd3d8fb73c77076c21bcf89ac115400eb7e.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/c5cf74a3f0d8585a7383348726bf8dd3d8fb73c77076c21bcf89ac115400eb7e.jpg)

![e374dce84cf32eac3a29ecb2f0d6f53c4a419497f913aa807adf164df445770c.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/e374dce84cf32eac3a29ecb2f0d6f53c4a419497f913aa807adf164df445770c.jpg)

![e434cd62a9923226a741b73c73b8ceaec469049b2a3525d1b62399a643b91e22.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/e434cd62a9923226a741b73c73b8ceaec469049b2a3525d1b62399a643b91e22.jpg)

![f8eb9e6a50dec3a0e41232f2cdeac1e7dde96358041cb857efc259f7b3a2223b.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/images/f8eb9e6a50dec3a0e41232f2cdeac1e7dde96358041cb857efc259f7b3a2223b.jpg)

### Tables

![197b7d32d670fcd03948ff1b052ac8343e36851250dfd8d1684186e9cf56f737.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/197b7d32d670fcd03948ff1b052ac8343e36851250dfd8d1684186e9cf56f737.jpg)

![3edb8bc91c13e305249be693cd1294984b35ed5f8c97f4471490158f288b149b.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/3edb8bc91c13e305249be693cd1294984b35ed5f8c97f4471490158f288b149b.jpg)

![42bc137b8d2804d8cdf851187e9124e31c162a7f447da61567d997fb47f3deeb.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/42bc137b8d2804d8cdf851187e9124e31c162a7f447da61567d997fb47f3deeb.jpg)

![451ad763c678875798207e34c97b4d51dace42058ec2d362cd09e861e059c136.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/451ad763c678875798207e34c97b4d51dace42058ec2d362cd09e861e059c136.jpg)

![5abf1fbdf51e194d09cfc2068250a23b33180fb19064418b8e8b59ef488a061f.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/5abf1fbdf51e194d09cfc2068250a23b33180fb19064418b8e8b59ef488a061f.jpg)

![8abf1b3dcf7f7dc92a9407c7410bf3b893d688100cf2cf9c332683a1e26b43f0.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/8abf1b3dcf7f7dc92a9407c7410bf3b893d688100cf2cf9c332683a1e26b43f0.jpg)

![962d0d28af6a86a045eea5d802666adb715bf330172bd9cf73f878655c1ef3fa.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/962d0d28af6a86a045eea5d802666adb715bf330172bd9cf73f878655c1ef3fa.jpg)

![b10ddd37ed2da92697869649402dfcc837c28c0b18d61bf24d9cf8ffd3cc0eab.jpg](../icml_results/3805_STAR_%20Rethinking%20MoE%20Routing%20as%20Structure-Aware%20Subspace%20Learning/tables/b10ddd37ed2da92697869649402dfcc837c28c0b18d61bf24d9cf8ffd3cc0eab.jpg)

<a id="paper-3766"></a>

## Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions


### Images

![0d41c0293be1b0f980a08e8fa3136b91af50282eeb5a5d9d9f2b0f02f003e718.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/0d41c0293be1b0f980a08e8fa3136b91af50282eeb5a5d9d9f2b0f02f003e718.jpg)

![0de79d048ac84a3f196c4576ed41f32a5db52aac1cea540b08b6ff774f3b9568.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/0de79d048ac84a3f196c4576ed41f32a5db52aac1cea540b08b6ff774f3b9568.jpg)

![152377c002ed3b2dd5afe076cc4527ca9fc4175553ca92882dad256682af3687.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/152377c002ed3b2dd5afe076cc4527ca9fc4175553ca92882dad256682af3687.jpg)

![299ffe68b952328a37673b9375af6dce0f66bb19b4cca247f626c87ebde7ba87.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/299ffe68b952328a37673b9375af6dce0f66bb19b4cca247f626c87ebde7ba87.jpg)

![2c92a5880b3e2c035aaa98d1e2c1220bed40e9570e2d936d40658b8e0fed44b2.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/2c92a5880b3e2c035aaa98d1e2c1220bed40e9570e2d936d40658b8e0fed44b2.jpg)

![51069066a4d7cf50cfa2728a476fadc3f46668648b2d26d1692a21576de1b2d5.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/51069066a4d7cf50cfa2728a476fadc3f46668648b2d26d1692a21576de1b2d5.jpg)

![51f66b1e467449c7892dd1610a9f56bff3504d36ecfefc75729c9402073c89f8.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/51f66b1e467449c7892dd1610a9f56bff3504d36ecfefc75729c9402073c89f8.jpg)

![7328762fa84af2cc972a70a35b03c7992557c644bf140d3b5fad546a828762fb.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/7328762fa84af2cc972a70a35b03c7992557c644bf140d3b5fad546a828762fb.jpg)

![7fdcde307edec14072082f1603c3cd4da1467a8d80af6e3bdef99d6f5ebb078a.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/7fdcde307edec14072082f1603c3cd4da1467a8d80af6e3bdef99d6f5ebb078a.jpg)

![9c716f90c4a0e8e0ed20cafc119ff64f34f97d259f790c5c08c68c89017f33f5.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/9c716f90c4a0e8e0ed20cafc119ff64f34f97d259f790c5c08c68c89017f33f5.jpg)

![9e31b0810e6fbdc873920e8d9eee0fa50d0097baf9ede10fd340b19b3bc4f8f2.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/9e31b0810e6fbdc873920e8d9eee0fa50d0097baf9ede10fd340b19b3bc4f8f2.jpg)

![ae131ab22f34a1e049f195d247172bc8ba2f5a158661ec10272ff8b99bb8cf1c.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/ae131ab22f34a1e049f195d247172bc8ba2f5a158661ec10272ff8b99bb8cf1c.jpg)

![baf335ad64196c2a9751f7f1154eedc09c07bcafe6149f230abbd29f7589cdbf.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/baf335ad64196c2a9751f7f1154eedc09c07bcafe6149f230abbd29f7589cdbf.jpg)

![c1fa331858c2bffab828197ed13833a5a7eed9b41635321e05aa97d276164b34.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/c1fa331858c2bffab828197ed13833a5a7eed9b41635321e05aa97d276164b34.jpg)

![d7f1ae9117ec63961c4bd5750b3dc03bc887fc82b63582f5c16271d5848ec435.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/images/d7f1ae9117ec63961c4bd5750b3dc03bc887fc82b63582f5c16271d5848ec435.jpg)

### Tables

![11182b65fe25cb2730463105ed4d8fe048fff9c47e101ceef0ce9fed889157d4.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/11182b65fe25cb2730463105ed4d8fe048fff9c47e101ceef0ce9fed889157d4.jpg)

![30853f89a50b33d82613759f55975a5337a41463990c36bc703a8253e5f94b0d.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/30853f89a50b33d82613759f55975a5337a41463990c36bc703a8253e5f94b0d.jpg)

![3c103b35788e83adca7972a6574d3388bad69fd6f73bc5f99941419337a59983.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/3c103b35788e83adca7972a6574d3388bad69fd6f73bc5f99941419337a59983.jpg)

![415b6e39930e4515cf9cd6781c1d71239eeaf711bbfee07e8600f6ffa2dfa13d.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/415b6e39930e4515cf9cd6781c1d71239eeaf711bbfee07e8600f6ffa2dfa13d.jpg)

![75b52bba430ca5ef3cfa364fbfc2f930ead7dd97b12096da3b231f1fdd0a26ec.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/75b52bba430ca5ef3cfa364fbfc2f930ead7dd97b12096da3b231f1fdd0a26ec.jpg)

![771ebfd3e39a3c58309762097a16ec2f7c439f949b4be8d707dadd18be647168.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/771ebfd3e39a3c58309762097a16ec2f7c439f949b4be8d707dadd18be647168.jpg)

![7a0631a58a26159147744911cc3a804d7b39e592093be0df9998fbd2d10bd428.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/7a0631a58a26159147744911cc3a804d7b39e592093be0df9998fbd2d10bd428.jpg)

![80ca283af78050a399859e7e1fa4b5aac0df37a89a010caa0b26fe1a8d43308f.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/80ca283af78050a399859e7e1fa4b5aac0df37a89a010caa0b26fe1a8d43308f.jpg)

![82d0833571da7e0a246a71d61bd982b2dba9969b0ca2b17aacd656834eb83c25.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/82d0833571da7e0a246a71d61bd982b2dba9969b0ca2b17aacd656834eb83c25.jpg)

![9166f5b8c0ef566ee8ed659646e3f83fde69e88ece2b95a52601c0299e400e69.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/9166f5b8c0ef566ee8ed659646e3f83fde69e88ece2b95a52601c0299e400e69.jpg)

![91c4b9769bb788a6595c27b45a607e0e9301a7786864141d24df465fb68e3bf3.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/91c4b9769bb788a6595c27b45a607e0e9301a7786864141d24df465fb68e3bf3.jpg)

![ab6797c62faa397f355456832732ca74ed94b6e007c3b0c10d9b60ed7f91244f.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/ab6797c62faa397f355456832732ca74ed94b6e007c3b0c10d9b60ed7f91244f.jpg)

![abe136486bc46a292a5388a9a1ddca09b8100c5c9685a76ae228983c8563eb82.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/abe136486bc46a292a5388a9a1ddca09b8100c5c9685a76ae228983c8563eb82.jpg)

![cd03fce6a436a122578bbfe2519d03a0e2e4b429e0fc0e6ff785eba2dce5b1ae.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/cd03fce6a436a122578bbfe2519d03a0e2e4b429e0fc0e6ff785eba2dce5b1ae.jpg)

![d45df1fb8e1b32561d69c57d0ea347805171347330db10cc269bbf57453fab9d.jpg](../icml_results/3806_Bridging%20the%20Knowledge-Prediction%20Gap%20in%20LLMs%20on%20Multiple-Choice%20Questions/tables/d45df1fb8e1b32561d69c57d0ea347805171347330db10cc269bbf57453fab9d.jpg)

<a id="paper-3767"></a>

## Scalable Reinforcement Learning via Adaptive Batch Scaling


### Images

![23b0fb6189887b9290dbb141332db2d3869419e10ad55eadac21e85d8edb6d28.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/23b0fb6189887b9290dbb141332db2d3869419e10ad55eadac21e85d8edb6d28.jpg)

![46d2ea05bbe09fab699f622632fa68f11e8e5585602804d883f6f28ca17a3c9f.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/46d2ea05bbe09fab699f622632fa68f11e8e5585602804d883f6f28ca17a3c9f.jpg)

![83f06ee85ec0dc941dcb055386a54b9e08ca35041dd0781f3254b4686b0fb5f5.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/83f06ee85ec0dc941dcb055386a54b9e08ca35041dd0781f3254b4686b0fb5f5.jpg)

![89aa058ecfed1acf858509f45251c50d12bdf61bfdc912f39423006b0ea3fd5a.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/89aa058ecfed1acf858509f45251c50d12bdf61bfdc912f39423006b0ea3fd5a.jpg)

![a1a2b5e1a04a7d164040d1edd85131bfc984460cc105118fd9afb062df464ddb.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/a1a2b5e1a04a7d164040d1edd85131bfc984460cc105118fd9afb062df464ddb.jpg)

![df6174e4ca19d6f91d6d00373776aeb37c48e7eab052c1b33fc497b0a0fb29e5.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/df6174e4ca19d6f91d6d00373776aeb37c48e7eab052c1b33fc497b0a0fb29e5.jpg)

![f6913de4fe31fe6bf5a7739716b55f41a98e9dce22bfa46a080c137f132c8012.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/f6913de4fe31fe6bf5a7739716b55f41a98e9dce22bfa46a080c137f132c8012.jpg)

![fdad906879987cf09e4a63a75e69933e81433546781316423bcbb6b407583a46.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/fdad906879987cf09e4a63a75e69933e81433546781316423bcbb6b407583a46.jpg)

![ffc874bc1d6e8f4b3189f704405c0960179e89f0a7c38027832292012a8d76f1.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/images/ffc874bc1d6e8f4b3189f704405c0960179e89f0a7c38027832292012a8d76f1.jpg)

### Tables

![04c8ae16343f692f886cf17e5c1a4482b96563102c4a6364331748c4ffa33155.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/04c8ae16343f692f886cf17e5c1a4482b96563102c4a6364331748c4ffa33155.jpg)

![5ac5b5cf3e5e7d7237830340f31e368b04a7637d4e5f92fc0a243578c2a94d45.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/5ac5b5cf3e5e7d7237830340f31e368b04a7637d4e5f92fc0a243578c2a94d45.jpg)

![6d38a5f14f312003e5f42662ee33b69b080cf58fe8bb1116859c917f25e6694f.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/6d38a5f14f312003e5f42662ee33b69b080cf58fe8bb1116859c917f25e6694f.jpg)

![83ce4b5a931630833bc4f451cf9d1d400a7de16bf73294fd81eb9316821ce803.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/83ce4b5a931630833bc4f451cf9d1d400a7de16bf73294fd81eb9316821ce803.jpg)

![8534de139b2fe3546ec18b1d7b9e356488f66bf62494b873a85ead341b2885af.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/8534de139b2fe3546ec18b1d7b9e356488f66bf62494b873a85ead341b2885af.jpg)

![abcb7030014cd5054c595bd763f50797c16022b2643f21ce5843465c7dfbb110.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/abcb7030014cd5054c595bd763f50797c16022b2643f21ce5843465c7dfbb110.jpg)

![bf9d6d9accf75a4a6603da3063e26f8d28727f6eefc8d8ab5ffdc2f391aedcc7.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/bf9d6d9accf75a4a6603da3063e26f8d28727f6eefc8d8ab5ffdc2f391aedcc7.jpg)

![cc5840d693109f649f54665ab80a1e178eda8e8fdb87f2ff6f631f95a397b7b4.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/cc5840d693109f649f54665ab80a1e178eda8e8fdb87f2ff6f631f95a397b7b4.jpg)

![eb4251c4665e209c416e5d9fcf3d3cd3ec76c8c0053b639b496f18c836847abc.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/eb4251c4665e209c416e5d9fcf3d3cd3ec76c8c0053b639b496f18c836847abc.jpg)

![f7985647fac6ee3fc6aa18b71f98330f192ede67c929093ed68565bfe444f88e.jpg](../icml_results/3807_Scalable%20Reinforcement%20Learning%20via%20Adaptive%20Batch%20Scaling/tables/f7985647fac6ee3fc6aa18b71f98330f192ede67c929093ed68565bfe444f88e.jpg)

<a id="paper-3768"></a>

## Quantile-Free Uncertainty Quantification in Graph Neural Networks


### Images

![0b8887fc9931e72e7cab6856e82e39d0481404cf23802a62782a79b2d87df7c6.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/0b8887fc9931e72e7cab6856e82e39d0481404cf23802a62782a79b2d87df7c6.jpg)

![4c76e7bec8acbcccf941e98e093127b0364edad0cdd98b30d227c571abf8424c.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/4c76e7bec8acbcccf941e98e093127b0364edad0cdd98b30d227c571abf8424c.jpg)

![4cc3f9551f065816cffd1da19107212550242007e9154048f0e32727220bddae.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/4cc3f9551f065816cffd1da19107212550242007e9154048f0e32727220bddae.jpg)

![74bcefa2df8d591947e6295bc5e57c46bc8041526a0ca5dc4ecf4ceecedd0fbd.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/74bcefa2df8d591947e6295bc5e57c46bc8041526a0ca5dc4ecf4ceecedd0fbd.jpg)

![8393b00003f19efab1359912cc5cc71f8cfcc5830f1d0db0d504c07d34d5f1b7.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/8393b00003f19efab1359912cc5cc71f8cfcc5830f1d0db0d504c07d34d5f1b7.jpg)

![936c84f0a27449d6ceee0e0eefd2983d6b26bf09b4f96a2dc914b3bd12c28f79.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/936c84f0a27449d6ceee0e0eefd2983d6b26bf09b4f96a2dc914b3bd12c28f79.jpg)

![b42da864acd2cf9d3b69a70ce28d055def15ffce6546ad5e701d9303016bdfb2.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/b42da864acd2cf9d3b69a70ce28d055def15ffce6546ad5e701d9303016bdfb2.jpg)

![c9070c211d25dbfc22732b8525f6ec03e91f3b8573ebb65d7e02b7d0b2d220ec.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/c9070c211d25dbfc22732b8525f6ec03e91f3b8573ebb65d7e02b7d0b2d220ec.jpg)

![d09569899aa84220a472377dc60a99c3bba7e3bf10aeb08678c1db1a5e313246.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/d09569899aa84220a472377dc60a99c3bba7e3bf10aeb08678c1db1a5e313246.jpg)

![da556830e3faf5b9febd243bee3430d5a60dc1041e5ee63f47e34e2deeb8b18b.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/da556830e3faf5b9febd243bee3430d5a60dc1041e5ee63f47e34e2deeb8b18b.jpg)

![f33cb665cbd28f552ebfa947956d0f8d35bcdc2fc868fd4d8a21fe6aa224f2e5.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/f33cb665cbd28f552ebfa947956d0f8d35bcdc2fc868fd4d8a21fe6aa224f2e5.jpg)

![ff7e2f526bc47a28737fff9fdcc83942aafab031c167cbe3ed92d1b5d1f3690f.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/images/ff7e2f526bc47a28737fff9fdcc83942aafab031c167cbe3ed92d1b5d1f3690f.jpg)

### Tables

![05cbd03c4b3473996f44cb6456d240e808f0717a2a4673c2521fa55dadc108a9.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/05cbd03c4b3473996f44cb6456d240e808f0717a2a4673c2521fa55dadc108a9.jpg)

![07e6b288091999bf2911f5bdfd39a27bab120f35024fcb06b709dcb0d3bcb7c3.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/07e6b288091999bf2911f5bdfd39a27bab120f35024fcb06b709dcb0d3bcb7c3.jpg)

![0bb48d99aab5520f13813246477930fefad37aae21096cddeb9a49f082054019.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/0bb48d99aab5520f13813246477930fefad37aae21096cddeb9a49f082054019.jpg)

![0e60b3a680f620485a989857da2c937f2fab39243f80894d52f680ce5ad00171.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/0e60b3a680f620485a989857da2c937f2fab39243f80894d52f680ce5ad00171.jpg)

![14fdce550876b65bc6d8e4fcf2243445d719eecd08bbf144c4297f9c0dc4d4d9.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/14fdce550876b65bc6d8e4fcf2243445d719eecd08bbf144c4297f9c0dc4d4d9.jpg)

![2feb4d3a1058ebbe07e268c76ea56d41e1eb534008a55555c65bfcdb38af2867.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/2feb4d3a1058ebbe07e268c76ea56d41e1eb534008a55555c65bfcdb38af2867.jpg)

![3328e124eeca214df53d314ff7d5d9780f40a4d9430223f889bc1e293e991570.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/3328e124eeca214df53d314ff7d5d9780f40a4d9430223f889bc1e293e991570.jpg)

![4a08b77a0af2b93019bc3aa2b9cafe2ab6381e3f50d5ccb93994887269e01259.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/4a08b77a0af2b93019bc3aa2b9cafe2ab6381e3f50d5ccb93994887269e01259.jpg)

![5304f99ad831d4ee66f2cbc33e3ac3f1fc7317933e8ba9e570b197d9f464380d.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/5304f99ad831d4ee66f2cbc33e3ac3f1fc7317933e8ba9e570b197d9f464380d.jpg)

![63b7d65cb8a502696b995a6e043ef38f0fe445dc9a72d04be060e38000f20217.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/63b7d65cb8a502696b995a6e043ef38f0fe445dc9a72d04be060e38000f20217.jpg)

![6401a8d3ac65ad5b6e4034da0438d95c3ac4eed618b820d995911f85d852e391.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/6401a8d3ac65ad5b6e4034da0438d95c3ac4eed618b820d995911f85d852e391.jpg)

![7fb81628a7975d4371687085274345177bcaf0ad1e636c49f02b5abda59a6d0e.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/7fb81628a7975d4371687085274345177bcaf0ad1e636c49f02b5abda59a6d0e.jpg)

![8bb25de1c45603c06c7bd65fd662025e52b852e84b76c143ed05729afaab2cc3.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/8bb25de1c45603c06c7bd65fd662025e52b852e84b76c143ed05729afaab2cc3.jpg)

![993b8563b4e7b40950643e01e5d3670339e65538e27fa0ab874e3926888f9b3e.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/993b8563b4e7b40950643e01e5d3670339e65538e27fa0ab874e3926888f9b3e.jpg)

![a2cb155ba11a1c0b18b3baecb5ce6399818b3f76be2007ab53c0cabcaae3af26.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/a2cb155ba11a1c0b18b3baecb5ce6399818b3f76be2007ab53c0cabcaae3af26.jpg)

![acff5da15d662e3f29f2a9ba6015aedd4f1f75d4ac23af1d1601eee665e8a594.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/acff5da15d662e3f29f2a9ba6015aedd4f1f75d4ac23af1d1601eee665e8a594.jpg)

![ae77a4d66aa479bda75764301971e6235f9444b843eba25609fd3c9fe3fac55f.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/ae77a4d66aa479bda75764301971e6235f9444b843eba25609fd3c9fe3fac55f.jpg)

![aebd61a3e5a0e92a04b7115d6ccb21d3ecb7b521542eecd146a7c3a3605a08ff.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/aebd61a3e5a0e92a04b7115d6ccb21d3ecb7b521542eecd146a7c3a3605a08ff.jpg)

![b567e47f29deaf9272e3c4e1a69fb796048bbd2ac922291dfb758c117b2bda1f.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/b567e47f29deaf9272e3c4e1a69fb796048bbd2ac922291dfb758c117b2bda1f.jpg)

![c5427761c80caee9e9cdcf9fc6e238dfbcb2a3dc147c0c6fdc4b055ce139a76e.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/c5427761c80caee9e9cdcf9fc6e238dfbcb2a3dc147c0c6fdc4b055ce139a76e.jpg)

![d568eb82b7e984ad98f062b0c19cc1032d60e7341dfaee836991ccef1a8f585a.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/d568eb82b7e984ad98f062b0c19cc1032d60e7341dfaee836991ccef1a8f585a.jpg)

![f9d55eb8c2df5ae820004c36db46f329acdf530753bdb3d1fa77c273cde8cc2a.jpg](../icml_results/3808_Quantile-Free%20Uncertainty%20Quantification%20in%20Graph%20Neural%20Networks/tables/f9d55eb8c2df5ae820004c36db46f329acdf530753bdb3d1fa77c273cde8cc2a.jpg)

<a id="paper-3769"></a>

## SeisMark: A Large-Scale Open Benchmark for Robust 3D Seismic Fault Detection


### Images

![263e0946e85909b81aa83a8bad9933a4a2fcfba71f980008cb8c5e8c9f4f1d43.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/images/263e0946e85909b81aa83a8bad9933a4a2fcfba71f980008cb8c5e8c9f4f1d43.jpg)

![48b28ebce1df2f0655dfeac33a76d68598ab381a4919721957d71054585ed1e2.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/images/48b28ebce1df2f0655dfeac33a76d68598ab381a4919721957d71054585ed1e2.jpg)

![ad24ef621b95fa5d7df743d940c6684363380f930432cbf282cca2ba35359d3c.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/images/ad24ef621b95fa5d7df743d940c6684363380f930432cbf282cca2ba35359d3c.jpg)

![ddb9dc3c1a7e0e2994d203bf3e8404eba8fce5cc1dd5f54f92c2d6fd75beea11.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/images/ddb9dc3c1a7e0e2994d203bf3e8404eba8fce5cc1dd5f54f92c2d6fd75beea11.jpg)

![fc8938d0eff1276f142b6d363482dcc11ff376b7e026d40b4477c61356ecfd05.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/images/fc8938d0eff1276f142b6d363482dcc11ff376b7e026d40b4477c61356ecfd05.jpg)

### Tables

![445701f2845ee0279cbc55d7c922971e0f97d88c4cb06bb4581235a011bfda56.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/445701f2845ee0279cbc55d7c922971e0f97d88c4cb06bb4581235a011bfda56.jpg)

![62e823c7c354388f0fbdaf0cf1f1ed80185114823efc68c149bff9635bf0420f.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/62e823c7c354388f0fbdaf0cf1f1ed80185114823efc68c149bff9635bf0420f.jpg)

![77841d233ddc7ff508dc69a0caf014df16613523f925fef017ae46ed6ff9c18a.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/77841d233ddc7ff508dc69a0caf014df16613523f925fef017ae46ed6ff9c18a.jpg)

![a686ec1c7df89d60ff60463543c39b4a1d25a131db0b7b0e24211aeae0b66b20.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/a686ec1c7df89d60ff60463543c39b4a1d25a131db0b7b0e24211aeae0b66b20.jpg)

![b3b7c3093a4ebd7ec6b16f5ea2b8db00af2c5a591ba2df3f2726dd45fae61de9.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/b3b7c3093a4ebd7ec6b16f5ea2b8db00af2c5a591ba2df3f2726dd45fae61de9.jpg)

![c55a01d24d439a1c576ae5d7f08c447db9b59a9529f9b4965ed3b69b4f466935.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/c55a01d24d439a1c576ae5d7f08c447db9b59a9529f9b4965ed3b69b4f466935.jpg)

![e32a8945d2fa0a58d6d5bd7eae71f43c7de03ba20a18440b35f1b1b27eefb59a.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/e32a8945d2fa0a58d6d5bd7eae71f43c7de03ba20a18440b35f1b1b27eefb59a.jpg)

![ef577a2b560d296020b768a7911d08e72ee32ca9535f8b9058e450dc745de769.jpg](../icml_results/3809_SeisMark_%20A%20Large-Scale%20Open%20Benchmark%20for%20Robust%203D%20Seismic%20Fault%20Detection/tables/ef577a2b560d296020b768a7911d08e72ee32ca9535f8b9058e450dc745de769.jpg)

<a id="paper-3770"></a>

## Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion


### Images

![04a047782daf141f8a4a0d7fd161a85486f01f2b6bfeb8cfcca7c81d4fd546f6.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/04a047782daf141f8a4a0d7fd161a85486f01f2b6bfeb8cfcca7c81d4fd546f6.jpg)

![07274a9e64d79bc83cede1681750eb93410d48547c1344a8266e35ff455e4a4b.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/07274a9e64d79bc83cede1681750eb93410d48547c1344a8266e35ff455e4a4b.jpg)

![202795d3f223a80d8acbbc4eb8abecd291e0dfb05f905e48dd8c52eb1b6d3d8d.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/202795d3f223a80d8acbbc4eb8abecd291e0dfb05f905e48dd8c52eb1b6d3d8d.jpg)

![24c72c116ff53f4209426bb25c55428da9778524676aca69183a65bceeec1789.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/24c72c116ff53f4209426bb25c55428da9778524676aca69183a65bceeec1789.jpg)

![2f75a194986ce5cda3ac8de6bfeb55985a28996f57edf505a5063fc776a375a5.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/2f75a194986ce5cda3ac8de6bfeb55985a28996f57edf505a5063fc776a375a5.jpg)

![4f34bb7f305b2baec5d5a0d2992d8df776fc608997a708af8afd63b03a4d7cba.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/4f34bb7f305b2baec5d5a0d2992d8df776fc608997a708af8afd63b03a4d7cba.jpg)

![589e2ddbbeff365fb2648361472bcf8f65b02e3328d4be4a2d86b302df7bd8a9.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/589e2ddbbeff365fb2648361472bcf8f65b02e3328d4be4a2d86b302df7bd8a9.jpg)

![5fe9e7f9f5076335d1633a22295cced3cc350c90d9097ead0f8f9e989687e2f5.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/5fe9e7f9f5076335d1633a22295cced3cc350c90d9097ead0f8f9e989687e2f5.jpg)

![6a7573293ea863d965b5ba9772c3637c7484231750e7d338c81bd210c00edb4f.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/6a7573293ea863d965b5ba9772c3637c7484231750e7d338c81bd210c00edb4f.jpg)

![6e462bfcba3e3f4e9b7b0cc08b22f2ad9a3cee37942cb4f1067cbc295b1af9e4.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/6e462bfcba3e3f4e9b7b0cc08b22f2ad9a3cee37942cb4f1067cbc295b1af9e4.jpg)

![7139dad2e2d1ebef3f26ab37a1d7cb327e87800639eb956c72c9b8531a2c5dd5.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/7139dad2e2d1ebef3f26ab37a1d7cb327e87800639eb956c72c9b8531a2c5dd5.jpg)

![760f0f453dc4dd8ca1d9ccc2d1877cb7abaa6e20212e05dedc270950dd83b36c.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/760f0f453dc4dd8ca1d9ccc2d1877cb7abaa6e20212e05dedc270950dd83b36c.jpg)

![853081babc653bcd95c110c1e5de1bac8a81719813b6be88328bb468c631bc37.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/853081babc653bcd95c110c1e5de1bac8a81719813b6be88328bb468c631bc37.jpg)

![8af94cbe072babef0a549ed78a7a908e9cfe2ad6e2382ebfe8b8d48fac9e4c47.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/8af94cbe072babef0a549ed78a7a908e9cfe2ad6e2382ebfe8b8d48fac9e4c47.jpg)

![8d9d04e55ad0ba0d6e625f0115d0a31589f652a53074fb1fe2fe253101ab225b.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/8d9d04e55ad0ba0d6e625f0115d0a31589f652a53074fb1fe2fe253101ab225b.jpg)

![91ea4dd54fc26cbdcc8f1a86f89b7c0685f7319e55ba39f33ba388e933465d1f.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/91ea4dd54fc26cbdcc8f1a86f89b7c0685f7319e55ba39f33ba388e933465d1f.jpg)

![96c8f1b246bf5daff10b3f67c3cc8e903cf5b561ff5b24c6d5c63b6fdb7d6316.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/96c8f1b246bf5daff10b3f67c3cc8e903cf5b561ff5b24c6d5c63b6fdb7d6316.jpg)

![990719f781e77d9e0e4529ce35c5d435193f19500ebca63b34df7062b4dce1ff.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/990719f781e77d9e0e4529ce35c5d435193f19500ebca63b34df7062b4dce1ff.jpg)

![9d361496e60855025bc05fba734ba9ef993d38df3ec0e52ae1d75f2418635ce7.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/9d361496e60855025bc05fba734ba9ef993d38df3ec0e52ae1d75f2418635ce7.jpg)

![9f8a03f312da4a17d8add67ba761932faf8060d7fc8c3564ebd326c4943687c3.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/9f8a03f312da4a17d8add67ba761932faf8060d7fc8c3564ebd326c4943687c3.jpg)

![b3af43b408eab1584e7d2c660b292b13b01384c90e136c8112b95836f02659be.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/b3af43b408eab1584e7d2c660b292b13b01384c90e136c8112b95836f02659be.jpg)

![b7fc3557b016d05142f9ae2520876ea31e9a774716e681467495724163d3e067.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/b7fc3557b016d05142f9ae2520876ea31e9a774716e681467495724163d3e067.jpg)

![bcfba8021380e29fdd5c18d457008b32bfd0dce3a9f576686bc84e3bba68149a.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/bcfba8021380e29fdd5c18d457008b32bfd0dce3a9f576686bc84e3bba68149a.jpg)

![bf931bfd3dda66a23b794218ee72492f8e1f71dad3150ea5c8de64c1e55b97c5.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/bf931bfd3dda66a23b794218ee72492f8e1f71dad3150ea5c8de64c1e55b97c5.jpg)

![c15fe33ccf69a3115b2e297084a43eb646dd618d3a5d2242d7e259fcf44d8a81.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/c15fe33ccf69a3115b2e297084a43eb646dd618d3a5d2242d7e259fcf44d8a81.jpg)

![df01eeadc597c814e9d63e8d0bafa77921bb931e286cbd2039f045a5fbb973d7.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/df01eeadc597c814e9d63e8d0bafa77921bb931e286cbd2039f045a5fbb973d7.jpg)

![eaa612a87ed0afb6859b317d5674152fc1284097c337866732a5505b1826ca94.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/eaa612a87ed0afb6859b317d5674152fc1284097c337866732a5505b1826ca94.jpg)

![eae907a8b609814911c736fb5c8f76a824aa861beae841da8a5669208cd574e0.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/eae907a8b609814911c736fb5c8f76a824aa861beae841da8a5669208cd574e0.jpg)

![ec3383129edf3f0a81342a5a04ee4d247536fe836559b6cbbb95ed7f205f5b7e.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/ec3383129edf3f0a81342a5a04ee4d247536fe836559b6cbbb95ed7f205f5b7e.jpg)

![edb9793f7cbbc64909fa44d5a0d32b3504f694d3bebc079610073f671f73234d.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/edb9793f7cbbc64909fa44d5a0d32b3504f694d3bebc079610073f671f73234d.jpg)

![f7d40f16f8da2e3cd4217b6928eee72244beb073fbc5b37eb6c3d4baf9fa1b82.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/images/f7d40f16f8da2e3cd4217b6928eee72244beb073fbc5b37eb6c3d4baf9fa1b82.jpg)

### Tables

![0b2bddead895c7b0932f87964c5152f71f51db83107237533fc51396564ded53.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/0b2bddead895c7b0932f87964c5152f71f51db83107237533fc51396564ded53.jpg)

![2b3b98adb29b49362cfd9dfa6d939bbb33297c015eecd24468bc3488f1527388.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/2b3b98adb29b49362cfd9dfa6d939bbb33297c015eecd24468bc3488f1527388.jpg)

![33d36c81520e418ef0b977cd7d714f36b70757ac53e98cf76faa105b549be458.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/33d36c81520e418ef0b977cd7d714f36b70757ac53e98cf76faa105b549be458.jpg)

![408246cbb43327b81db00a264134ec280abee63e7b892fcf00e061ae9b8af56c.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/408246cbb43327b81db00a264134ec280abee63e7b892fcf00e061ae9b8af56c.jpg)

![4f3e36777ee9d95a93c10bc8104ac01b81001e1d8d13d5cd0a5dea37bb101e30.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/4f3e36777ee9d95a93c10bc8104ac01b81001e1d8d13d5cd0a5dea37bb101e30.jpg)

![55d10b232719d5281bfdf0d2020aca88bff76c35c0521c43caa01f48dd2c90c0.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/55d10b232719d5281bfdf0d2020aca88bff76c35c0521c43caa01f48dd2c90c0.jpg)

![5870a04d0f71ebe8806ef91f671f225499d39b464792b21830662ad1d97352fb.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/5870a04d0f71ebe8806ef91f671f225499d39b464792b21830662ad1d97352fb.jpg)

![659a7e5bfbf232a22521b6c871d8007260ab5b31488c668bea6d84800faf06d4.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/659a7e5bfbf232a22521b6c871d8007260ab5b31488c668bea6d84800faf06d4.jpg)

![6a2df853603805b593253904fdb2a4a75146c3ec568b80f41724499e259d382a.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/6a2df853603805b593253904fdb2a4a75146c3ec568b80f41724499e259d382a.jpg)

![70fbeb50de5f84837feca87f9a43cc164e9fd5378106ed5cb6f6605095a76ebc.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/70fbeb50de5f84837feca87f9a43cc164e9fd5378106ed5cb6f6605095a76ebc.jpg)

![8367d7a97422c5ec76bed54f83b82582414da6c46be0d4de54ca26638547eb7a.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/8367d7a97422c5ec76bed54f83b82582414da6c46be0d4de54ca26638547eb7a.jpg)

![a9bbbeb3306ad4bc22937803140c5c20e1635b1443c0417630c4e8b92a8f64a0.jpg](../icml_results/3810_Beyond%20Generative%20Priors_%20Minority%20Sampling%20with%20JEPA-Guided%20Diffusion/tables/a9bbbeb3306ad4bc22937803140c5c20e1635b1443c0417630c4e8b92a8f64a0.jpg)

<a id="paper-3771"></a>

## Generative Neural Operators through Diffusion Last Layer


### Images

![0610358e82331b9970245ea86750c28d9238a8081554085f4701dde1e8f1061a.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/0610358e82331b9970245ea86750c28d9238a8081554085f4701dde1e8f1061a.jpg)

![0d1fab540bfce56ee3782f5a47890bb3a7e3b63ab14675c3279f2717c610f6bb.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/0d1fab540bfce56ee3782f5a47890bb3a7e3b63ab14675c3279f2717c610f6bb.jpg)

![135b9d5ab24b9a3541a48f151abc8c8fc28b74eb686eb884c638abf567da4465.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/135b9d5ab24b9a3541a48f151abc8c8fc28b74eb686eb884c638abf567da4465.jpg)

![17dae62a82fb04a66c01bf283f8bdbf82e32f82da18a25e0cefba9217ceb90f3.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/17dae62a82fb04a66c01bf283f8bdbf82e32f82da18a25e0cefba9217ceb90f3.jpg)

![35246b69d3ee3d621378d4fed7e811c029319f8ed10fdf297b942f2f8ec0e533.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/35246b69d3ee3d621378d4fed7e811c029319f8ed10fdf297b942f2f8ec0e533.jpg)

![90300876aeb4ebc342bd941dcba36c250f5066fd2aa4ae2b2915ddd20e1f2103.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/90300876aeb4ebc342bd941dcba36c250f5066fd2aa4ae2b2915ddd20e1f2103.jpg)

![f7e6c768b0ebaf6791cddbebf46aeaeeff495f0267858af2b044a5878d6971c3.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/images/f7e6c768b0ebaf6791cddbebf46aeaeeff495f0267858af2b044a5878d6971c3.jpg)

### Tables

![0af7d0cc2721956494a79e6ea38c39bb2efd795f0c0537488ddef3cfaa44201b.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/0af7d0cc2721956494a79e6ea38c39bb2efd795f0c0537488ddef3cfaa44201b.jpg)

![1209b0377ced5168c529f48b1e2d76a1cf30368b772d244dd297927ca13ba103.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/1209b0377ced5168c529f48b1e2d76a1cf30368b772d244dd297927ca13ba103.jpg)

![346c777dce6b7754543c49a9a51653fe2d6f4ce9ae02032a9adbcb460b73f830.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/346c777dce6b7754543c49a9a51653fe2d6f4ce9ae02032a9adbcb460b73f830.jpg)

![3beea9d14f7d765ebf01af5e160a4bbe1ff6f1295adc41f9088a279c268b0506.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/3beea9d14f7d765ebf01af5e160a4bbe1ff6f1295adc41f9088a279c268b0506.jpg)

![4807f9614aee878e9afc5be089d9c829d7d227d55e71461e63c612a33165f7e1.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/4807f9614aee878e9afc5be089d9c829d7d227d55e71461e63c612a33165f7e1.jpg)

![49fa6985530638257987ad558338992e7ac575daf24f2d3e04a118352cd32ac4.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/49fa6985530638257987ad558338992e7ac575daf24f2d3e04a118352cd32ac4.jpg)

![52e45d03161b623adb24f741b5e8b7d5c58feb54034cefbf0440a39e7eefe883.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/52e45d03161b623adb24f741b5e8b7d5c58feb54034cefbf0440a39e7eefe883.jpg)

![735f916080866bcf7fb3f5ae87b400a928bcadd9a6ba242affb123f52cf52cbd.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/735f916080866bcf7fb3f5ae87b400a928bcadd9a6ba242affb123f52cf52cbd.jpg)

![893ceb67bcc6a8b7e357ae676f2a07281883b25ea09d43fb742b69dc44ef076f.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/893ceb67bcc6a8b7e357ae676f2a07281883b25ea09d43fb742b69dc44ef076f.jpg)

![9bc8677559fbf9b81367b01dc6a66d02f766dad78f1300f3d5d2b07a19bd9139.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/9bc8677559fbf9b81367b01dc6a66d02f766dad78f1300f3d5d2b07a19bd9139.jpg)

![bea88460a09956b15194c5326b627271aafeedf6c8b80b6f90a9dad558fb2f26.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/bea88460a09956b15194c5326b627271aafeedf6c8b80b6f90a9dad558fb2f26.jpg)

![c28b81b3a94de66a9b8bbfb127cfbe04eef7a282daf28dce6064e308471bdcdc.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/c28b81b3a94de66a9b8bbfb127cfbe04eef7a282daf28dce6064e308471bdcdc.jpg)

![d785aee979986f6a9e0565f7728508f02126128c0740fec46be225da28b894d7.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/d785aee979986f6a9e0565f7728508f02126128c0740fec46be225da28b894d7.jpg)

![dcf84de6f2f9910564757f43e172191f868b6e9aa122269af5918e6d45e429e5.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/dcf84de6f2f9910564757f43e172191f868b6e9aa122269af5918e6d45e429e5.jpg)

![ecaa412cc5abc32832ae080a91842f3b2d7109032329ad6834856fe6a4da77a9.jpg](../icml_results/3811_Generative%20Neural%20Operators%20through%20Diffusion%20Last%20Layer/tables/ecaa412cc5abc32832ae080a91842f3b2d7109032329ad6834856fe6a4da77a9.jpg)

<a id="paper-3772"></a>

## TsLLM: Augmenting LLMs for General Time Series Understanding and Prediction


### Images

![62ff9386e57dd1f942a67d6520611bbe84acda6695370d173c9fedc8a857fe1c.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/images/62ff9386e57dd1f942a67d6520611bbe84acda6695370d173c9fedc8a857fe1c.jpg)

![9053f7a45043fa1a3bfb106768b314d07010e7a0ca900c054b873aa82c9968e2.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/images/9053f7a45043fa1a3bfb106768b314d07010e7a0ca900c054b873aa82c9968e2.jpg)

![af3885dbbe2962a58075a12f1ed2b5785786b5aa4ddd8d17b80164b83bb34c4a.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/images/af3885dbbe2962a58075a12f1ed2b5785786b5aa4ddd8d17b80164b83bb34c4a.jpg)

### Tables

![03b0eb83fe4d2fbac39a6a5b99679f1674e84d502d6f51ab24ff1d06055f0b83.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/03b0eb83fe4d2fbac39a6a5b99679f1674e84d502d6f51ab24ff1d06055f0b83.jpg)

![26979636ae8a1f1a99996fbe6f23d3ec9b5166a9d84e3ba19912ef1c28ae812a.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/26979636ae8a1f1a99996fbe6f23d3ec9b5166a9d84e3ba19912ef1c28ae812a.jpg)

![33bd854e6375be859ac8882353a47beb4cd84a940af96e962bd7a2a249b5a2f6.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/33bd854e6375be859ac8882353a47beb4cd84a940af96e962bd7a2a249b5a2f6.jpg)

![53287fb645075f58928fd6865f27a8292551360fb62cd167d7eed9adf361426e.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/53287fb645075f58928fd6865f27a8292551360fb62cd167d7eed9adf361426e.jpg)

![574f8cd2b40fc13bb7ed184f7c3e420e75d761d577ceab8c6ba1f86f9978e837.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/574f8cd2b40fc13bb7ed184f7c3e420e75d761d577ceab8c6ba1f86f9978e837.jpg)

![67d7099bbe76deb5eadfe92597508b786167052f7e064f2cbc34496a62203489.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/67d7099bbe76deb5eadfe92597508b786167052f7e064f2cbc34496a62203489.jpg)

![7779c54456d8e7de8da0c377ec1f9e8d00a141454b1da6e30dd2b0a0a2f92279.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/7779c54456d8e7de8da0c377ec1f9e8d00a141454b1da6e30dd2b0a0a2f92279.jpg)

![b828dda38c8a707b8233f8424cfd4778ddfc03c8b80e27164b2cd2f4d8efbd53.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/b828dda38c8a707b8233f8424cfd4778ddfc03c8b80e27164b2cd2f4d8efbd53.jpg)

![f4abbe5fd072f8372ea078368b5ff49a2c2a8871eb21571bbb57997ed78e7f1d.jpg](../icml_results/3812_TsLLM_%20Augmenting%20LLMs%20for%20General%20Time%20Series%20Understanding%20and%20Prediction/tables/f4abbe5fd072f8372ea078368b5ff49a2c2a8871eb21571bbb57997ed78e7f1d.jpg)

<a id="paper-3773"></a>

## Deep networks learn to parse uniform-depth context-free languages from local statistics


### Images

![028fde8e1874a9b3735e4c6e64465d528f3a25794136f59e02f2a760626b2e2a.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/028fde8e1874a9b3735e4c6e64465d528f3a25794136f59e02f2a760626b2e2a.jpg)

![02fb308fb797031e349f63159395da3b854925123e462e96af4c786077fcabb8.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/02fb308fb797031e349f63159395da3b854925123e462e96af4c786077fcabb8.jpg)

![0ff8630227392eaec14f718a1461b0752a718e67f2a1d59f3700af45847ded7f.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/0ff8630227392eaec14f718a1461b0752a718e67f2a1d59f3700af45847ded7f.jpg)

![170e1141a341d065a7ecc0939e037d299d40c62e9309be473a5db3bab2264bc7.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/170e1141a341d065a7ecc0939e037d299d40c62e9309be473a5db3bab2264bc7.jpg)

![279e691a5d331dc44a74569ac5ad7f68003608cabe567bc56d423db04334723e.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/279e691a5d331dc44a74569ac5ad7f68003608cabe567bc56d423db04334723e.jpg)

![2ac408ded5b799049e8240f588d51e3bf5e651281f6a572dc36d019578b339f3.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/2ac408ded5b799049e8240f588d51e3bf5e651281f6a572dc36d019578b339f3.jpg)

![3131f2f760e9c099c9b2d008e018711e1e9e2a9c9b365d13091e9541aab4b31c.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/3131f2f760e9c099c9b2d008e018711e1e9e2a9c9b365d13091e9541aab4b31c.jpg)

![3cf67efb5af674f9c7c951a1aafeee18ebcf6844b19a6720a4d4c8e8c62c4770.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/3cf67efb5af674f9c7c951a1aafeee18ebcf6844b19a6720a4d4c8e8c62c4770.jpg)

![3f2612d2576eb1fc1415033f7a428a5bdab3a54bce27f7bb9b77e0d925184212.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/3f2612d2576eb1fc1415033f7a428a5bdab3a54bce27f7bb9b77e0d925184212.jpg)

![706f714b64e7bf7c99548c366b8cb04dbee48e745ea6729d56afc5506fec41d8.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/706f714b64e7bf7c99548c366b8cb04dbee48e745ea6729d56afc5506fec41d8.jpg)

![838377fa2cadca082ef0cba7a17cc638a5d16c85d74b6c8385313895521df981.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/838377fa2cadca082ef0cba7a17cc638a5d16c85d74b6c8385313895521df981.jpg)

![863b64145a994781f3de38e994b9fcdd36ccedb0e7d5b609206a525e627b29d8.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/863b64145a994781f3de38e994b9fcdd36ccedb0e7d5b609206a525e627b29d8.jpg)

![8fd356f34188a757f1c384c040ea47840d3c5b0a0ad6201592eb02f85bfafb4d.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/8fd356f34188a757f1c384c040ea47840d3c5b0a0ad6201592eb02f85bfafb4d.jpg)

![c81317c6ee99f38f04c9448e94d8f3d5d6fe0beaf442d653729247ed8a71b418.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/c81317c6ee99f38f04c9448e94d8f3d5d6fe0beaf442d653729247ed8a71b418.jpg)

![cf3f0d98f300c31824b2e7326d381b320b72108d0654c1e20ea3787f3c9b91e1.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/cf3f0d98f300c31824b2e7326d381b320b72108d0654c1e20ea3787f3c9b91e1.jpg)

![e3067fd93a6ad0d9447ffe2bb6839c7e599bdee02b756e17caa644868699c560.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/e3067fd93a6ad0d9447ffe2bb6839c7e599bdee02b756e17caa644868699c560.jpg)

![e4fed8bd38fef10ef65647e9ab70065ee6f47f5a857412cc9243e47d86c3047f.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/e4fed8bd38fef10ef65647e9ab70065ee6f47f5a857412cc9243e47d86c3047f.jpg)

![f386692d25856d3ceb97a6ad7c14c580752acc15609e56d1f917d9dd7e7149d3.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/f386692d25856d3ceb97a6ad7c14c580752acc15609e56d1f917d9dd7e7149d3.jpg)

![ff4f9f6ab6bbf478fd37edca9b3f722f68759aa7015c74d2024c399f3ce4fa08.jpg](../icml_results/3813_Deep%20networks%20learn%20to%20parse%20uniform-depth%20context-free%20languages%20from%20local%20statistics/images/ff4f9f6ab6bbf478fd37edca9b3f722f68759aa7015c74d2024c399f3ce4fa08.jpg)

<a id="paper-3774"></a>

## Constrained Adaptive Rejection Sampling


### Images

![28f9def9a0804046d581da12f296e1b6563fa3d8cba08430390fc651c55d24d4.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/28f9def9a0804046d581da12f296e1b6563fa3d8cba08430390fc651c55d24d4.jpg)

![3b90dac4b6832d01136479d2f9aeb1e7d6ce4e00e10b4f09c9ccfea7148fe7a8.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/3b90dac4b6832d01136479d2f9aeb1e7d6ce4e00e10b4f09c9ccfea7148fe7a8.jpg)

![3e8477a6b651b8bd5cb36b19121496d9d994e623d3918f9278cb0c47f9a5657c.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/3e8477a6b651b8bd5cb36b19121496d9d994e623d3918f9278cb0c47f9a5657c.jpg)

![796dc49f465408e0eba24875d9ee8dcd46f63c261e243334044fa8ad7afe9953.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/796dc49f465408e0eba24875d9ee8dcd46f63c261e243334044fa8ad7afe9953.jpg)

![970e7b338372f234b516c64d313c6124468b48c14dd663048e86f96655c221a4.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/970e7b338372f234b516c64d313c6124468b48c14dd663048e86f96655c221a4.jpg)

![ad73f6d44fe36816cbc05098bad17e37b0d60879eff2eed0b827af7147a78652.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/ad73f6d44fe36816cbc05098bad17e37b0d60879eff2eed0b827af7147a78652.jpg)

![b826f069be0832b4ab0339518e2fdabd0abb5c0d39a149ae193661b99b0a364c.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/b826f069be0832b4ab0339518e2fdabd0abb5c0d39a149ae193661b99b0a364c.jpg)

![c6ad2a274ec86091408bb3831d6cbcbcfa9c64aec095871bb198ed39781bd14e.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/c6ad2a274ec86091408bb3831d6cbcbcfa9c64aec095871bb198ed39781bd14e.jpg)

![e49b976b3d20dec3559a31e578742b111f53a779293a0616c72e8823f4bafe4d.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/e49b976b3d20dec3559a31e578742b111f53a779293a0616c72e8823f4bafe4d.jpg)

![f2eb2fad28fff942bb9fa8475050cf033e209068d5fd9aeb7890ece7622441be.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/images/f2eb2fad28fff942bb9fa8475050cf033e209068d5fd9aeb7890ece7622441be.jpg)

### Tables

![06fa503a06e012a1e621bd584c21126d8a82b5cfe7fc03cc804c735efab86c13.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/06fa503a06e012a1e621bd584c21126d8a82b5cfe7fc03cc804c735efab86c13.jpg)

![3f7f2cc91838dd6d8325fb7f0264e6e5cbdb844de6160d7194a016241174a818.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/3f7f2cc91838dd6d8325fb7f0264e6e5cbdb844de6160d7194a016241174a818.jpg)

![3f93cb7a2e214460599673206c4e28626cb81371dc354895ed250b9bf6037882.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/3f93cb7a2e214460599673206c4e28626cb81371dc354895ed250b9bf6037882.jpg)

![47462ad5b83c613feccf7b7bb00515504dc2fbc256d621a5f919f98b0a3ca8d3.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/47462ad5b83c613feccf7b7bb00515504dc2fbc256d621a5f919f98b0a3ca8d3.jpg)

![533266bc429cc965bef12459d42c53b6c9ef5ac479061e50f611e32996f60d4b.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/533266bc429cc965bef12459d42c53b6c9ef5ac479061e50f611e32996f60d4b.jpg)

![99535e234fc26a8cd48cb2e707239626e5f4b45466230b745128300f8f7335a0.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/99535e234fc26a8cd48cb2e707239626e5f4b45466230b745128300f8f7335a0.jpg)

![9e5dce1f131b69f21227943875b6402b3635c3106afcb14a6ba718254b6fc060.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/9e5dce1f131b69f21227943875b6402b3635c3106afcb14a6ba718254b6fc060.jpg)

![a734298fe6807a88b77d2c2071e92ed5b00bc5a54e1fca835c07c25f1f256acc.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/a734298fe6807a88b77d2c2071e92ed5b00bc5a54e1fca835c07c25f1f256acc.jpg)

![a853870f566d867455418c3c2c4e1a4d9e7cb9f7769fa35e70eed029aa7ed55c.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/a853870f566d867455418c3c2c4e1a4d9e7cb9f7769fa35e70eed029aa7ed55c.jpg)

![adef9d2a725b38cdfa9bb4351555097972f1fecaef576792c41681b111d5794a.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/adef9d2a725b38cdfa9bb4351555097972f1fecaef576792c41681b111d5794a.jpg)

![b8fb13246f679dc34afc2dc7f45e775b0392d38ab108357cef5c01831d999dd8.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/b8fb13246f679dc34afc2dc7f45e775b0392d38ab108357cef5c01831d999dd8.jpg)

![be48de2f3db7758de1f112a53a63b560610187903c02df7b24cb1f2ae72d2ced.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/be48de2f3db7758de1f112a53a63b560610187903c02df7b24cb1f2ae72d2ced.jpg)

![cc2add4d0552cddce7473163fe0491e5a17a71564e1e6c0b630bec49c67393e0.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/cc2add4d0552cddce7473163fe0491e5a17a71564e1e6c0b630bec49c67393e0.jpg)

![d3c2241eaa7d3c60198eadb0642e4e75439890a7b10ea9d623f36f382e2628ed.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/d3c2241eaa7d3c60198eadb0642e4e75439890a7b10ea9d623f36f382e2628ed.jpg)

![e2f607f3f86e7094cfa24cf3cba3cb93a28f78c6adbb7e57a30ba700e4020050.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/e2f607f3f86e7094cfa24cf3cba3cb93a28f78c6adbb7e57a30ba700e4020050.jpg)

![e98c4178deefe682458f773c826aa738a447517500b7d6eea883e210a521205c.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/e98c4178deefe682458f773c826aa738a447517500b7d6eea883e210a521205c.jpg)

![ead033210c8a9a970b4ea259d35e4f61e99768b19d61b2c8e41187fe8677f762.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/ead033210c8a9a970b4ea259d35e4f61e99768b19d61b2c8e41187fe8677f762.jpg)

![ee6faaf7952cb42a901df7cb3c04ad64cb2d25c8741a8be69c51e32c654e04f4.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/ee6faaf7952cb42a901df7cb3c04ad64cb2d25c8741a8be69c51e32c654e04f4.jpg)

![fbce4bac7d959013edcb4c271ca8ce279f80767b8c7327868d168cb5000000be.jpg](../icml_results/3814_Constrained%20Adaptive%20Rejection%20Sampling/tables/fbce4bac7d959013edcb4c271ca8ce279f80767b8c7327868d168cb5000000be.jpg)

<a id="paper-3775"></a>

## Insertion Based Sequence Generation with Learnable Order Dynamics


### Images

![0bab41c9aa7bc8e4e94af735aca9b3379dc4fff99bafbc86bf8b993966220a8b.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/0bab41c9aa7bc8e4e94af735aca9b3379dc4fff99bafbc86bf8b993966220a8b.jpg)

![17f550f09662b425e2f448f8ef49d62b63a02f8f568369a4ef2d0b5ce894d918.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/17f550f09662b425e2f448f8ef49d62b63a02f8f568369a4ef2d0b5ce894d918.jpg)

![2331d3f21d9b513d711323d6ab703c76508e4110fb06eaafaa0fb4f1caeec657.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/2331d3f21d9b513d711323d6ab703c76508e4110fb06eaafaa0fb4f1caeec657.jpg)

![26e16707c9a3e40929430b12d1be6092a7f0de219bbac6f53fdfee6f0e57cedd.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/26e16707c9a3e40929430b12d1be6092a7f0de219bbac6f53fdfee6f0e57cedd.jpg)

![315c10719409b95ac3c64668ce3293786a7300e6a93bd8a7d6dadbf66182a0da.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/315c10719409b95ac3c64668ce3293786a7300e6a93bd8a7d6dadbf66182a0da.jpg)

![31bb7d9252469706ccb8a6cba5e9ffede70076ac9bdd4b200fbb270de967c1ca.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/31bb7d9252469706ccb8a6cba5e9ffede70076ac9bdd4b200fbb270de967c1ca.jpg)

![46708bb5ac97783718826c4f9b861c98e795f415fd51dbe864d0fe56259e9c87.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/46708bb5ac97783718826c4f9b861c98e795f415fd51dbe864d0fe56259e9c87.jpg)

![5f64beaeb27683243b0023f69f156f2df6b160e43714b522f245163b748118f6.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/5f64beaeb27683243b0023f69f156f2df6b160e43714b522f245163b748118f6.jpg)

![60d22105f889c07185b6c2fc5cd3e16a1cbafa54702a2886e8614a50b4661fa5.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/60d22105f889c07185b6c2fc5cd3e16a1cbafa54702a2886e8614a50b4661fa5.jpg)

![61041b0abd4a78c8d98878dd463fcf68f6e3a15ffc2836d5685270cf191010b6.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/61041b0abd4a78c8d98878dd463fcf68f6e3a15ffc2836d5685270cf191010b6.jpg)

![7587b590f0619df0ad5ebb61037c7d6a08dcb97a3b0824eedf144dce084e288b.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/7587b590f0619df0ad5ebb61037c7d6a08dcb97a3b0824eedf144dce084e288b.jpg)

![83ae5b8fdd2205bf0e95be327f2cd2e0a76fb2c267a62a34ee9cbbfa6b141b3b.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/83ae5b8fdd2205bf0e95be327f2cd2e0a76fb2c267a62a34ee9cbbfa6b141b3b.jpg)

![8ec75c48426c462c78cfeb7afe94991e5cdb3376377c881e585702c520ed834b.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/8ec75c48426c462c78cfeb7afe94991e5cdb3376377c881e585702c520ed834b.jpg)

![9ed65edcef01dcdc3c1cef28552e9239af444a80c5258159995fdc6fe76e1f77.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/9ed65edcef01dcdc3c1cef28552e9239af444a80c5258159995fdc6fe76e1f77.jpg)

![a70a02adcccd9e1992f0e9d4923a0c01263edfaf8fcee1f9640d209a27906110.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/a70a02adcccd9e1992f0e9d4923a0c01263edfaf8fcee1f9640d209a27906110.jpg)

![c6ab47053dc5b3a8b05cadfc0dd5f9f00db60aaf34b08442356ddab69e33f419.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/c6ab47053dc5b3a8b05cadfc0dd5f9f00db60aaf34b08442356ddab69e33f419.jpg)

![df7fc9ddbe096d3f77344c8bfe4b59c1ac6a9104d74cf7802ff0ff69641fc99a.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/images/df7fc9ddbe096d3f77344c8bfe4b59c1ac6a9104d74cf7802ff0ff69641fc99a.jpg)

### Tables

![1b8f7e65a55622fcfd4ceb907a9b403c62ee13e85d9b6b3f8afdd3ff5d0d3421.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/1b8f7e65a55622fcfd4ceb907a9b403c62ee13e85d9b6b3f8afdd3ff5d0d3421.jpg)

![7f70dd818bdbfae31c17798ad28ae7b2b6c9770f41a1b3d9a84941e3334921ee.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/7f70dd818bdbfae31c17798ad28ae7b2b6c9770f41a1b3d9a84941e3334921ee.jpg)

![91d001fdb4cf6b4e5330f26f894d4e96690b562905b40bf8534c768e596b82be.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/91d001fdb4cf6b4e5330f26f894d4e96690b562905b40bf8534c768e596b82be.jpg)

![94a2c2f6127ee97b03b3ae6fce848cb9fc1e1f0e1b2ac7f29584194ad0e64113.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/94a2c2f6127ee97b03b3ae6fce848cb9fc1e1f0e1b2ac7f29584194ad0e64113.jpg)

![9f3288c14b9a51e5f0eeb561f3107aa146471ed0a2f7acc6c2d91b7a9f7f88ba.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/9f3288c14b9a51e5f0eeb561f3107aa146471ed0a2f7acc6c2d91b7a9f7f88ba.jpg)

![aa85268f53ab3abf32609810cbebe02df6d550c7e69f46f082a710bee4efd85c.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/aa85268f53ab3abf32609810cbebe02df6d550c7e69f46f082a710bee4efd85c.jpg)

![ad5037453b9b1d27b7492ec0f52305b9fcd797e5d6815fa56c2b821711d38321.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/ad5037453b9b1d27b7492ec0f52305b9fcd797e5d6815fa56c2b821711d38321.jpg)

![c73088286712fda97fd61ea52065a387e78b2b9fcd0af0cca342e4484159123d.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/c73088286712fda97fd61ea52065a387e78b2b9fcd0af0cca342e4484159123d.jpg)

![d1f907ff3e862902e1c56a8dab059347c42ea982b0e1bd7842b70de4b27e3a50.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/d1f907ff3e862902e1c56a8dab059347c42ea982b0e1bd7842b70de4b27e3a50.jpg)

![fccdc9a21afa5f518bc503b65c1761217bc54c8893814467bf3d8b4a61851d29.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/fccdc9a21afa5f518bc503b65c1761217bc54c8893814467bf3d8b4a61851d29.jpg)

![ff6f9b84f5fc24bcde898fe61bb0b5d7827fbe09698d3cc321c12400412a94ba.jpg](../icml_results/3815_Insertion%20Based%20Sequence%20Generation%20with%20Learnable%20Order%20Dynamics/tables/ff6f9b84f5fc24bcde898fe61bb0b5d7827fbe09698d3cc321c12400412a94ba.jpg)

<a id="paper-3776"></a>

## OGPO: Sample Efficient Full-Finetuning of Generative Control Policies


### Images

![0b65e431a91dcc12c48d5e2f77d4ea08dc7f25115bd406d949e70ff40350c3ed.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/0b65e431a91dcc12c48d5e2f77d4ea08dc7f25115bd406d949e70ff40350c3ed.jpg)

![111c9c9eb75ffcd9351a9190134cf3425a8860848e3cf397ade5942fa670c734.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/111c9c9eb75ffcd9351a9190134cf3425a8860848e3cf397ade5942fa670c734.jpg)

![14b2a0e8b06bf02e180b7abdb876361a30eb5f588887ca76d64d2399f227209d.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/14b2a0e8b06bf02e180b7abdb876361a30eb5f588887ca76d64d2399f227209d.jpg)

![33317621d0d4a2f8776309b4f6c00b5a3e446b5df6ef3a27e8bfd2613f253b80.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/33317621d0d4a2f8776309b4f6c00b5a3e446b5df6ef3a27e8bfd2613f253b80.jpg)

![4997c022f0bf8c9e76e8eb382a33856b772b1a2194e6a40860be16c41b0d8bfa.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/4997c022f0bf8c9e76e8eb382a33856b772b1a2194e6a40860be16c41b0d8bfa.jpg)

![4a66f84afa6a2871f8c77cf774b52233bd52b7cdf82917df7af45305d3787a4d.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/4a66f84afa6a2871f8c77cf774b52233bd52b7cdf82917df7af45305d3787a4d.jpg)

![536a4968b2d5d4ebff967c991123ad23013f0edcffba432fe1522743e18f5c49.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/536a4968b2d5d4ebff967c991123ad23013f0edcffba432fe1522743e18f5c49.jpg)

![75b34cd32e993c5cee5dd1848fef1d5258c16e2a28f2a27e30363310f655529a.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/75b34cd32e993c5cee5dd1848fef1d5258c16e2a28f2a27e30363310f655529a.jpg)

![83f0ce7d59e9e96f29c663fa36b5a674cffc80956cfe147ac78b43524f79055d.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/83f0ce7d59e9e96f29c663fa36b5a674cffc80956cfe147ac78b43524f79055d.jpg)

![9e63f08b94751f7ac656aa1ba44acadb09faae0654a3c63601047fa7ae7162f3.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/9e63f08b94751f7ac656aa1ba44acadb09faae0654a3c63601047fa7ae7162f3.jpg)

![9fb49daf69df62d64d5ce4965c10145e2f1b88dda983e2d0060c8fea26fde590.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/9fb49daf69df62d64d5ce4965c10145e2f1b88dda983e2d0060c8fea26fde590.jpg)

![a4b2fc36cb171183a939ebd6b3eab843b4cb3bbaa1b1683ba873034bc9ba0707.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/a4b2fc36cb171183a939ebd6b3eab843b4cb3bbaa1b1683ba873034bc9ba0707.jpg)

![af01497246d40cd4b735c9d19645035bfe55626ae5208c19ae85110ee5334137.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/af01497246d40cd4b735c9d19645035bfe55626ae5208c19ae85110ee5334137.jpg)

![c6ab487430605ce866dd2f8f4226a9dedf6ddcd9f89f1134caf458a5aded0718.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/c6ab487430605ce866dd2f8f4226a9dedf6ddcd9f89f1134caf458a5aded0718.jpg)

![c8dafccdefc498f28ec52b12fb74918f727ee53e917174f92f32fcf3b00266ec.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/c8dafccdefc498f28ec52b12fb74918f727ee53e917174f92f32fcf3b00266ec.jpg)

![d4fedc88f11a5d3288e2f778d8d841dbd0d8c5f69c19bf9f299acd4598380f03.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/d4fedc88f11a5d3288e2f778d8d841dbd0d8c5f69c19bf9f299acd4598380f03.jpg)

![d844ae9ce041e511266cfec37146e40748a8c8d7e9467cb2532706e50e204cfe.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/d844ae9ce041e511266cfec37146e40748a8c8d7e9467cb2532706e50e204cfe.jpg)

![df2e0587fc782388a62f93050291c903b87bf2d0d07a329e1b5bd60453e10d8b.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/df2e0587fc782388a62f93050291c903b87bf2d0d07a329e1b5bd60453e10d8b.jpg)

![e2e4af251d138bf08375c6479aaeaa1e92cf309231e7509be8a4dcd7204db750.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/e2e4af251d138bf08375c6479aaeaa1e92cf309231e7509be8a4dcd7204db750.jpg)

![e5635e3db90a80b83ec4a7c418d704042d9a801349294af4f3a1dceab2b0d77c.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/e5635e3db90a80b83ec4a7c418d704042d9a801349294af4f3a1dceab2b0d77c.jpg)

![e6653275e6927a8e19fece30155cf9b75a8bb232e065744f58ed164a8b3d3212.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/images/e6653275e6927a8e19fece30155cf9b75a8bb232e065744f58ed164a8b3d3212.jpg)

### Tables

![172f01cbfb7bdba21713ed866cc763e6af7ea399169a56d3840b01c423ba04b8.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/172f01cbfb7bdba21713ed866cc763e6af7ea399169a56d3840b01c423ba04b8.jpg)

![22c422f5a3128b5cac5f15c6f9fa13b9bd1092e05064a936e9d3b8a446e16f2e.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/22c422f5a3128b5cac5f15c6f9fa13b9bd1092e05064a936e9d3b8a446e16f2e.jpg)

![56b2c37b37d1398e56d7b96a515419a94e316a65fcc26a1e88f8b60ea9f85b0c.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/56b2c37b37d1398e56d7b96a515419a94e316a65fcc26a1e88f8b60ea9f85b0c.jpg)

![7d4be72d01d1f36d2f3da6200f6c4564011ed6b5ec6948a6430db7a686699a02.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/7d4be72d01d1f36d2f3da6200f6c4564011ed6b5ec6948a6430db7a686699a02.jpg)

![b0351fab0cd4ad97b3a9dadf5f8a889886904b7e9fcec7650a06538ed5b8dc14.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/b0351fab0cd4ad97b3a9dadf5f8a889886904b7e9fcec7650a06538ed5b8dc14.jpg)

![bbe6cfc7413a7dc90ab70701643861a74d36c894979173bc409a352d22f6cd99.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/bbe6cfc7413a7dc90ab70701643861a74d36c894979173bc409a352d22f6cd99.jpg)

![cf28611ef033f6d4783945fadfb41b72318cdfcd369aaf4efa57b60fcfcb5c3d.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/cf28611ef033f6d4783945fadfb41b72318cdfcd369aaf4efa57b60fcfcb5c3d.jpg)

![d68812674c11820ea0694ff18734371b411cc90607e78dc6b1dde050cb831a0d.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/d68812674c11820ea0694ff18734371b411cc90607e78dc6b1dde050cb831a0d.jpg)

![d7562dce62b5298fb3a030846b370aec2209878a9f485f6bc57e9a1d43b05ab6.jpg](../icml_results/3816_OGPO_%20Sample%20Efficient%20Full-Finetuning%20of%20Generative%20Control%20Policies/tables/d7562dce62b5298fb3a030846b370aec2209878a9f485f6bc57e9a1d43b05ab6.jpg)

<a id="paper-3777"></a>

## Assistive Prompt Mediation: Evaluating Language Models Under Accessibility Constraints


### Images

![179a086ad1c9d6b246264ac46e11e0f305d320c304a6dc79deb3aa2855ce00d5.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/179a086ad1c9d6b246264ac46e11e0f305d320c304a6dc79deb3aa2855ce00d5.jpg)

![34c526bc91b2c8978e0d7b31c5e15821eb72ac53cfaad95559a6a3fe6a382c65.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/34c526bc91b2c8978e0d7b31c5e15821eb72ac53cfaad95559a6a3fe6a382c65.jpg)

![3c8468460b4421a528210c2ba4720160107c1fe0a67e79fdd73f5deb73731122.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/3c8468460b4421a528210c2ba4720160107c1fe0a67e79fdd73f5deb73731122.jpg)

![720d53894fecde60b13fb52b5d4535e2e3f5a6df068272baf27293729aecbfc2.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/720d53894fecde60b13fb52b5d4535e2e3f5a6df068272baf27293729aecbfc2.jpg)

![8f8300a7cae9e15bdecc6f75699d005c40953f469c40b0f4cb04a062103d327e.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/8f8300a7cae9e15bdecc6f75699d005c40953f469c40b0f4cb04a062103d327e.jpg)

![a080580efd4e47aa18cfc2778dc442515667988ec108f2d1d3dce7810aea1486.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/a080580efd4e47aa18cfc2778dc442515667988ec108f2d1d3dce7810aea1486.jpg)

![d09f0ff70f9189d55fe3fb09b145eab6a698146e129310c53872c701e3d8f48f.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/d09f0ff70f9189d55fe3fb09b145eab6a698146e129310c53872c701e3d8f48f.jpg)

![dd3c742dcfdf77a7b20f0bb884998085c47319edc10d6d00b453cc20e88f0c65.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/dd3c742dcfdf77a7b20f0bb884998085c47319edc10d6d00b453cc20e88f0c65.jpg)

![f317e40f87f01a69f8e1a2a0a61ae6ced3fd1d62fe2b6e0473b420d530ad6760.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/images/f317e40f87f01a69f8e1a2a0a61ae6ced3fd1d62fe2b6e0473b420d530ad6760.jpg)

### Tables

![0875f87e80645dca3c78ff409d9cea9d2ebb93cdadcebed85b46a5baaebfcba7.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/0875f87e80645dca3c78ff409d9cea9d2ebb93cdadcebed85b46a5baaebfcba7.jpg)

![2bf1e0e5f082d7a3fd71a1aa2031b5fe3d5c3b05401122f5bc6942924f7df755.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/2bf1e0e5f082d7a3fd71a1aa2031b5fe3d5c3b05401122f5bc6942924f7df755.jpg)

![2de37c23b337c5d6143a7571894b65f8981388b5a0e27e5d5801791708d4feff.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/2de37c23b337c5d6143a7571894b65f8981388b5a0e27e5d5801791708d4feff.jpg)

![35cd7180808283f8dfe8bd6a8d0ad7f96032098ae28adbc387b40b538beb48c4.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/35cd7180808283f8dfe8bd6a8d0ad7f96032098ae28adbc387b40b538beb48c4.jpg)

![485730986e180ccea5370501a0146ff711ea83f8fb580b39ebb0577f30cb1537.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/485730986e180ccea5370501a0146ff711ea83f8fb580b39ebb0577f30cb1537.jpg)

![51e7664546c8185aa644dee5137306944fd726a8ca5c838a5c40d75d1ae18fae.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/51e7664546c8185aa644dee5137306944fd726a8ca5c838a5c40d75d1ae18fae.jpg)

![5b8153c275a94e6c13aabc988fa08fdbcc1cef3036fef325a6d3d5795ab700ae.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/5b8153c275a94e6c13aabc988fa08fdbcc1cef3036fef325a6d3d5795ab700ae.jpg)

![5f5e3b715bd2ec63607023b9500da90314e6f5518f829881464ef326149d0b3e.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/5f5e3b715bd2ec63607023b9500da90314e6f5518f829881464ef326149d0b3e.jpg)

![68b3cc414fe36f71e17d1ee1ae6fa3c2a2ce2e1ba69a3e17a9fd1b395d257ed3.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/68b3cc414fe36f71e17d1ee1ae6fa3c2a2ce2e1ba69a3e17a9fd1b395d257ed3.jpg)

![76ba5c7dfa2031adb813c7ec9489a20e33a4cbb7700c558845a6d53eabf783db.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/76ba5c7dfa2031adb813c7ec9489a20e33a4cbb7700c558845a6d53eabf783db.jpg)

![7adb5965d7a2e67b685163adeca4efe6d28705f1fa0a7f445caf9583aff17cc3.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/7adb5965d7a2e67b685163adeca4efe6d28705f1fa0a7f445caf9583aff17cc3.jpg)

![abe03950e7f5bfbcab4c6c86a36abe70c91a9432211e1831ccdfc5d42dcbe905.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/abe03950e7f5bfbcab4c6c86a36abe70c91a9432211e1831ccdfc5d42dcbe905.jpg)

![b3257eb350ae83290a2caf540c70fd2bf452bf760107c0f66e938d423ce630a1.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/b3257eb350ae83290a2caf540c70fd2bf452bf760107c0f66e938d423ce630a1.jpg)

![b782705ab753add850055478b8cb5e3675fe51eee499c21a0d0a5bfa4406388f.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/b782705ab753add850055478b8cb5e3675fe51eee499c21a0d0a5bfa4406388f.jpg)

![bab3b7e7b237f1ae9aa63c5aa51f4d906808d3cfbcb7faa709f95076370f2c08.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/bab3b7e7b237f1ae9aa63c5aa51f4d906808d3cfbcb7faa709f95076370f2c08.jpg)

![bc0dcfe0eeadba5cb874e2354d3043d8de7f15ba6f01a4cf1d586b43f12e553a.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/bc0dcfe0eeadba5cb874e2354d3043d8de7f15ba6f01a4cf1d586b43f12e553a.jpg)

![e67f0ff7fadedf5442bfcad51a1f3b6540bee88cd116aa5f30f67647e29bcbfd.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/e67f0ff7fadedf5442bfcad51a1f3b6540bee88cd116aa5f30f67647e29bcbfd.jpg)

![f1d2b7775cef27b19b843735a3d298007b6393869e4d8e8e9f86737ad85e0031.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/f1d2b7775cef27b19b843735a3d298007b6393869e4d8e8e9f86737ad85e0031.jpg)

![f3f6fed56b115cca600476a471f4a829c9506a762ab082a873b813ab9f15228d.jpg](../icml_results/3817_Assistive%20Prompt%20Mediation_%20Evaluating%20Language%20Models%20Under%20Accessibility%20Constraints/tables/f3f6fed56b115cca600476a471f4a829c9506a762ab082a873b813ab9f15228d.jpg)

<a id="paper-3778"></a>

## Riemannian stochastic optimization for sufficient dimension reduction


### Images

![02233a756307404b672fc7451c18f4dc386b724e2539f73cfb3aaf995ff78ee5.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/images/02233a756307404b672fc7451c18f4dc386b724e2539f73cfb3aaf995ff78ee5.jpg)

![757697df71f302abb3f315ac96ecf8080deff7846205aa05bc1e5cca53eb6522.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/images/757697df71f302abb3f315ac96ecf8080deff7846205aa05bc1e5cca53eb6522.jpg)

![a203061ac81161116c972b3379c8605d6fd779c53c596a77c8d79c0332ff1965.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/images/a203061ac81161116c972b3379c8605d6fd779c53c596a77c8d79c0332ff1965.jpg)

![c1163fad20dc7481dc1b391086ec67e4447bf08a22c2affba92f894eb4b4a9ce.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/images/c1163fad20dc7481dc1b391086ec67e4447bf08a22c2affba92f894eb4b4a9ce.jpg)

![e299b056333077817ece9c00b7d0507dc3f8de6ccf8e720ca5009b7e095c3097.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/images/e299b056333077817ece9c00b7d0507dc3f8de6ccf8e720ca5009b7e095c3097.jpg)

### Tables

![0a440366055a85655c49f82bb3812b0fc6bb9153caabefaebd952c3cd9a3b6d4.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/0a440366055a85655c49f82bb3812b0fc6bb9153caabefaebd952c3cd9a3b6d4.jpg)

![30cd8ca67aad293ca39a15407456df5be6de31c356ff96d0bb128ca32b5d7a7b.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/30cd8ca67aad293ca39a15407456df5be6de31c356ff96d0bb128ca32b5d7a7b.jpg)

![336503696e07c72f718908dd475d119fc177530dff994aa15b222ca41e185253.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/336503696e07c72f718908dd475d119fc177530dff994aa15b222ca41e185253.jpg)

![3f26dc919802eb7a508d3d3f17f13ce999c7b67c03854a0b75fb6f1003f0a9d4.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/3f26dc919802eb7a508d3d3f17f13ce999c7b67c03854a0b75fb6f1003f0a9d4.jpg)

![7adbca05966146f84e61c1374312a5c8eaf16ef780f34fac093f00b1632991f4.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/7adbca05966146f84e61c1374312a5c8eaf16ef780f34fac093f00b1632991f4.jpg)

![880c865cb28dbcf69a3e82861abec56be4e70ea2955191c5d65f0c9f8ae58e8d.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/880c865cb28dbcf69a3e82861abec56be4e70ea2955191c5d65f0c9f8ae58e8d.jpg)

![8d2f2c57c5cf5bea9f9d453a24eedb55e14f0b582dc8781ba5e27b662c93d43c.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/8d2f2c57c5cf5bea9f9d453a24eedb55e14f0b582dc8781ba5e27b662c93d43c.jpg)

![a7c8b0470acde42dfd32eb3c19d38d4b23be98d2093925ea97431a9176966c0c.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/a7c8b0470acde42dfd32eb3c19d38d4b23be98d2093925ea97431a9176966c0c.jpg)

![c1ace21916be80e79a2e0d11d60630fb7c1672840069352ac9aaa3189700044d.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/c1ace21916be80e79a2e0d11d60630fb7c1672840069352ac9aaa3189700044d.jpg)

![ce4242f2c7e9f37145a40110b6e82523c2cdcbc771b3122d2337d31eee2e1b9b.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/ce4242f2c7e9f37145a40110b6e82523c2cdcbc771b3122d2337d31eee2e1b9b.jpg)

![dd284c7e70bbfd666dd01ae17923c56abe0c6a683e4928cc9fcedb5848a68416.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/dd284c7e70bbfd666dd01ae17923c56abe0c6a683e4928cc9fcedb5848a68416.jpg)

![ded941e5f1fed2d890d30ba63279632ee2aa429b004cd2660c17dd7086447f42.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/ded941e5f1fed2d890d30ba63279632ee2aa429b004cd2660c17dd7086447f42.jpg)

![e8cef39df7eb01817890c0c5dd76f40d6d215e60ff38604ce37c6c1129d57769.jpg](../icml_results/3818_Riemannian%20stochastic%20optimization%20for%20sufficient%20dimension%20reduction/tables/e8cef39df7eb01817890c0c5dd76f40d6d215e60ff38604ce37c6c1129d57769.jpg)

<a id="paper-3779"></a>

## APEX: Approximate-but-exhaustive search for ultra-large combinatorial synthesis libraries


### Images

![1c8b6886d82c939a52810c2e0360adae47f0528063eaf1662741b09914e7c79c.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/1c8b6886d82c939a52810c2e0360adae47f0528063eaf1662741b09914e7c79c.jpg)

![6c93b0ff317b838730dc315341d0bbfbda0b7a358b944fe456bd4265bf547d44.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/6c93b0ff317b838730dc315341d0bbfbda0b7a358b944fe456bd4265bf547d44.jpg)

![6c9a62b4134d2c8d261cf4c6375b143bc2598615f585ca52e78bdba572063be2.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/6c9a62b4134d2c8d261cf4c6375b143bc2598615f585ca52e78bdba572063be2.jpg)

![805ab063ddaedb79871b1d1631140cfdd25d6d6d4f2682c538487c28c6e441d6.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/805ab063ddaedb79871b1d1631140cfdd25d6d6d4f2682c538487c28c6e441d6.jpg)

![810b139b2148dd66f86085e748a25af94c0b72c47fe059af536742864e2b538d.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/810b139b2148dd66f86085e748a25af94c0b72c47fe059af536742864e2b538d.jpg)

![84ae63cb81eb66c9de171f460cfbe15886e2a14ae55e7ff1e812aeedbe33a7f9.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/84ae63cb81eb66c9de171f460cfbe15886e2a14ae55e7ff1e812aeedbe33a7f9.jpg)

![9df4f7b86bc1605f5a9f4996abf590ddfda82139fe0809e8a390d46ef26f7722.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/9df4f7b86bc1605f5a9f4996abf590ddfda82139fe0809e8a390d46ef26f7722.jpg)

![b5fc666ea990577c185d14154105649286fc2d8fe21bddd5b4c2b2469770a749.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/b5fc666ea990577c185d14154105649286fc2d8fe21bddd5b4c2b2469770a749.jpg)

![c11a00d6f393daedefe22ebd4a601365a5046587fbde836faf4ade832da6c712.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/c11a00d6f393daedefe22ebd4a601365a5046587fbde836faf4ade832da6c712.jpg)

![d34fd65af41387c04e30cf8aac8f1938e88dab5075789776ea037872faf53b52.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/d34fd65af41387c04e30cf8aac8f1938e88dab5075789776ea037872faf53b52.jpg)

![fe050e06546479aa56ffda1c2e63436812f7857c34b607de6d3f8f7c8aecf784.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/images/fe050e06546479aa56ffda1c2e63436812f7857c34b607de6d3f8f7c8aecf784.jpg)

### Tables

![49c481188cdf853077b7c1e6c22d8f7dc8784adf5bbca08199db2fc221da108f.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/tables/49c481188cdf853077b7c1e6c22d8f7dc8784adf5bbca08199db2fc221da108f.jpg)

![5dece7884eed9914964d610c1f3e2d0b63c98df104a14cdbae4e3bd2004ae6a4.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/tables/5dece7884eed9914964d610c1f3e2d0b63c98df104a14cdbae4e3bd2004ae6a4.jpg)

![80bf6ac74d668622a3217a0978073b37cf88bf88da44d794bd332494f6b2da4a.jpg](../icml_results/3819_APEX_%20Approximate-but-exhaustive%20search%20for%20ultra-large%20combinatorial%20synthesis%20libraries/tables/80bf6ac74d668622a3217a0978073b37cf88bf88da44d794bd332494f6b2da4a.jpg)

<a id="paper-3780"></a>

## FunPhase: A Periodic Functional Autoencoder for Motion Generation via Phase Manifolds


### Images

![3cff8f1ad2adf2b30593f58f8eff4f925f88b22411e3a0c4e9d674c704d9c95c.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/3cff8f1ad2adf2b30593f58f8eff4f925f88b22411e3a0c4e9d674c704d9c95c.jpg)

![58e13c40bbfa2ecd865d4888aff6bec958237c357fdcaa9981809ea7f642206b.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/58e13c40bbfa2ecd865d4888aff6bec958237c357fdcaa9981809ea7f642206b.jpg)

![7c5d66b2ad44e2e63f9dbaa25e11baa73bf4aff840d5ff655d10cfc17fad541e.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/7c5d66b2ad44e2e63f9dbaa25e11baa73bf4aff840d5ff655d10cfc17fad541e.jpg)

![904e3504fd948105443cb081c27c1d8f2e73e1a75fa14aae82d922c34f0cb62d.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/904e3504fd948105443cb081c27c1d8f2e73e1a75fa14aae82d922c34f0cb62d.jpg)

![b5dd7166eb19f182e9d249289dc871fd64810554334517e9e9f08b7bff82ec2a.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/b5dd7166eb19f182e9d249289dc871fd64810554334517e9e9f08b7bff82ec2a.jpg)

![e1e2ec045ab4b40a3cde32d2531928006f74aee86e049745e487ab26a8491ce0.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/e1e2ec045ab4b40a3cde32d2531928006f74aee86e049745e487ab26a8491ce0.jpg)

![f7931f7b1772bd066a1bf8f3cfc1995ad709309ce3a208f5e15edc271624e542.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/images/f7931f7b1772bd066a1bf8f3cfc1995ad709309ce3a208f5e15edc271624e542.jpg)

### Tables

![191ed71af4546fad1414da2658c9c501c18e9c933b81417ae3648f3c7ef53661.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/191ed71af4546fad1414da2658c9c501c18e9c933b81417ae3648f3c7ef53661.jpg)

![324f04ff5d17f85a240053fb9ec59d84afe239db6c0a1b7b8a68178b19fca1b3.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/324f04ff5d17f85a240053fb9ec59d84afe239db6c0a1b7b8a68178b19fca1b3.jpg)

![831042b265d669526951b97ccf2cdc5d5532279779bee16c1b5f2d899a40d941.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/831042b265d669526951b97ccf2cdc5d5532279779bee16c1b5f2d899a40d941.jpg)

![93de949da38bf3f41823f6941072ad5825aff9d48dee6897baa709ce661f660e.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/93de949da38bf3f41823f6941072ad5825aff9d48dee6897baa709ce661f660e.jpg)

![a3880319dcce61a7373ff89742b2f22086f961826c170dc579939207f7e590c8.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/a3880319dcce61a7373ff89742b2f22086f961826c170dc579939207f7e590c8.jpg)

![ae69e5c8e4e38f203522bc659c5ad2fa245e053df155190015af1c07ba58000d.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/ae69e5c8e4e38f203522bc659c5ad2fa245e053df155190015af1c07ba58000d.jpg)

![d40adc3b3b49b53255c83fe92b89fd455418fcb9079232a11aee771b634a7985.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/d40adc3b3b49b53255c83fe92b89fd455418fcb9079232a11aee771b634a7985.jpg)

![e04b7276d780a55dbacfe1def262619bb71c73a2fcb0121e3608a21f75b601fe.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/e04b7276d780a55dbacfe1def262619bb71c73a2fcb0121e3608a21f75b601fe.jpg)

![e76d25abbccc71706dc0bdee6175f2fc98b0ec22c43f71f67ede6a06fb213999.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/e76d25abbccc71706dc0bdee6175f2fc98b0ec22c43f71f67ede6a06fb213999.jpg)

![e7a47b03daf5b76b3343c6901ac4080bd1b9ef79eaeb8f54943ff5e6429ce392.jpg](../icml_results/3820_FunPhase_%20A%20Periodic%20Functional%20Autoencoder%20for%20Motion%20Generation%20via%20Phase%20Manifolds/tables/e7a47b03daf5b76b3343c6901ac4080bd1b9ef79eaeb8f54943ff5e6429ce392.jpg)

<a id="paper-3781"></a>

## When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning


### Images

![056e18e57700a2d16edaea69c159fac6cc0227300726f218768cbd9b14058825.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/056e18e57700a2d16edaea69c159fac6cc0227300726f218768cbd9b14058825.jpg)

![12c9a1123026ca2ca625474090450fb58c88f51ecd8187ae0e5ab2b50795c898.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/12c9a1123026ca2ca625474090450fb58c88f51ecd8187ae0e5ab2b50795c898.jpg)

![34008c70fa5cb03444aff1fb30f69c57ed5d5671c4dd5ab3399987c632ce30b5.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/34008c70fa5cb03444aff1fb30f69c57ed5d5671c4dd5ab3399987c632ce30b5.jpg)

![4ce9e5e42d455ff5f68a9ef50520527ec1f851f38a5e6e7749168cac6cf42520.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/4ce9e5e42d455ff5f68a9ef50520527ec1f851f38a5e6e7749168cac6cf42520.jpg)

![5a0200ce9a19dc55a17adc8c32b422febe4a06f9ee7b0220dc062a752254430c.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/5a0200ce9a19dc55a17adc8c32b422febe4a06f9ee7b0220dc062a752254430c.jpg)

![5bade74a705a3b9e0cee3d880c8d7545f4dca2b31efcb0584e71671e9df529d6.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/5bade74a705a3b9e0cee3d880c8d7545f4dca2b31efcb0584e71671e9df529d6.jpg)

![76b849a5a8de9578f1ee8def3b062147bf60859312a551a5bc48ab583149e186.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/76b849a5a8de9578f1ee8def3b062147bf60859312a551a5bc48ab583149e186.jpg)

![abb2959dc2d3940da8852133058361bb2b9d95d4a4a4820a833f702db273fb93.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/abb2959dc2d3940da8852133058361bb2b9d95d4a4a4820a833f702db273fb93.jpg)

![bab7bdaea2fd3e64382d4865c1a3784389c068a39b533cd584c5c9cb6f96d9c0.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/bab7bdaea2fd3e64382d4865c1a3784389c068a39b533cd584c5c9cb6f96d9c0.jpg)

![c8913066a0bb7797e511243e2074252b30d0cd63f2555e8af93dd0711da3b52e.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/c8913066a0bb7797e511243e2074252b30d0cd63f2555e8af93dd0711da3b52e.jpg)

![cada1d81049557e7922448e246fee157f49c173975a1865925f6be176509ebd0.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/cada1d81049557e7922448e246fee157f49c173975a1865925f6be176509ebd0.jpg)

![d2993d7ec564eecd6a517bdae2ec1dc4d33be652b0f18260cfe0aeed57713f30.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/d2993d7ec564eecd6a517bdae2ec1dc4d33be652b0f18260cfe0aeed57713f30.jpg)

![ee8d6fa29fddd01c5111fbff5b7cd0f523f50a5c65f2910cb68f46551f59474f.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/images/ee8d6fa29fddd01c5111fbff5b7cd0f523f50a5c65f2910cb68f46551f59474f.jpg)

### Tables

![13075d01ddc130a8111760f7fb50488641b828c59b0a103efb8ef5315a0c4363.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/13075d01ddc130a8111760f7fb50488641b828c59b0a103efb8ef5315a0c4363.jpg)

![1d5067473836a1c8ced52187f9081821cb4e348d4c12be45d006e63c2679e0df.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/1d5067473836a1c8ced52187f9081821cb4e348d4c12be45d006e63c2679e0df.jpg)

![2cfffe961933dd55e8a72d15163eab6c1990f2df0354a797b04bc17ed9ed376d.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/2cfffe961933dd55e8a72d15163eab6c1990f2df0354a797b04bc17ed9ed376d.jpg)

![3452a7cd12f66025617e46d5c48f65da697d316bb16144dc986b6713a14377d6.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/3452a7cd12f66025617e46d5c48f65da697d316bb16144dc986b6713a14377d6.jpg)

![39dcb16e485676eab52aa3578aa269011f215d57d5fe643d32444120216b52da.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/39dcb16e485676eab52aa3578aa269011f215d57d5fe643d32444120216b52da.jpg)

![43f67e4e8a543a3a28d0eddc5be00a9423e6d15975c3803f7445b7748aac37c9.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/43f67e4e8a543a3a28d0eddc5be00a9423e6d15975c3803f7445b7748aac37c9.jpg)

![4eb20cb588460887dac20b3e8a2023e3fe112606cbbfecdab8bce4f748e33ea2.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/4eb20cb588460887dac20b3e8a2023e3fe112606cbbfecdab8bce4f748e33ea2.jpg)

![57a9f96f5841ed635f99b77be02bcb623f245d93b3a1999877cc857d9cc6e041.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/57a9f96f5841ed635f99b77be02bcb623f245d93b3a1999877cc857d9cc6e041.jpg)

![6b67956c75adad146692560bd5f90ec448f1d045e5d4466a89124e12f79baff2.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/6b67956c75adad146692560bd5f90ec448f1d045e5d4466a89124e12f79baff2.jpg)

![70983e884d232148ba15ba7aebd77493f4118fc0a7143f408f4b3781093937a2.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/70983e884d232148ba15ba7aebd77493f4118fc0a7143f408f4b3781093937a2.jpg)

![93882ae46fab91f5c2b0eb47299f0176d49401b82d246835b95e04e1ccfaf8eb.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/93882ae46fab91f5c2b0eb47299f0176d49401b82d246835b95e04e1ccfaf8eb.jpg)

![b56b92622e9e5d2e27eaa06352b4316d6ad5521f28e68aa00ee18a183af46c08.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/b56b92622e9e5d2e27eaa06352b4316d6ad5521f28e68aa00ee18a183af46c08.jpg)

![d2da3e011d86979d98c6863a1effee69da969ad4d2f557249b16ed8686870b40.jpg](../icml_results/3821_When%20LLMs%20Develop%20Languages_%20Symbolic%20Communication%20for%20Efficient%20Multi-Agent%20Reasoning/tables/d2da3e011d86979d98c6863a1effee69da969ad4d2f557249b16ed8686870b40.jpg)

<a id="paper-3782"></a>

## Hyperbolic Neural Operator


### Images

![002e29fe1b0b71c6bcff6e5bfcf593f1f870f1924db370e378a4e8fb15bfc033.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/002e29fe1b0b71c6bcff6e5bfcf593f1f870f1924db370e378a4e8fb15bfc033.jpg)

![0639c22664344055713c7de88fc0b7680a25b81fa9b1b84a8444420062cf9070.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/0639c22664344055713c7de88fc0b7680a25b81fa9b1b84a8444420062cf9070.jpg)

![131dec8a2b1a9cda43c753df60ceaafc9c2b543f8f33304a405c8492b97ca20a.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/131dec8a2b1a9cda43c753df60ceaafc9c2b543f8f33304a405c8492b97ca20a.jpg)

![141ec652e27595554c67cca488e51f5a9da6eae2769d69834fa20459f57990a7.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/141ec652e27595554c67cca488e51f5a9da6eae2769d69834fa20459f57990a7.jpg)

![1941e4839d6bc5135796a318db4b8e14381be143649182450ad42363a89a18e4.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/1941e4839d6bc5135796a318db4b8e14381be143649182450ad42363a89a18e4.jpg)

![1c394a11d21c94e26236706185b31a944d90ada80b7d75a47ff8c920d6391038.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/1c394a11d21c94e26236706185b31a944d90ada80b7d75a47ff8c920d6391038.jpg)

![1ec341c870eeed700bd71200ffeb8eaea0e8f11eb518074dd95f5b0991836364.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/1ec341c870eeed700bd71200ffeb8eaea0e8f11eb518074dd95f5b0991836364.jpg)

![21a4941f63c22ae4f6add480d61e41973bb80b2b8b5eb5f30dcb2e6b07872491.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/21a4941f63c22ae4f6add480d61e41973bb80b2b8b5eb5f30dcb2e6b07872491.jpg)

![2355f069736f52d6a41ecb286c1cbd417efcf40e7bfa1a3182d9080fde2b1f17.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/2355f069736f52d6a41ecb286c1cbd417efcf40e7bfa1a3182d9080fde2b1f17.jpg)

![25125d71879480a39775409a2bf3bada51cb3da0060c39e65e022b9f7c1617c7.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/25125d71879480a39775409a2bf3bada51cb3da0060c39e65e022b9f7c1617c7.jpg)

![28504097f338a07133ecb222d6ad61d08a10e13fd559608e957cc55247deffba.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/28504097f338a07133ecb222d6ad61d08a10e13fd559608e957cc55247deffba.jpg)

![29386cd58346796dd0f882ecbed4074b13f88eb43f49e7332d2ec7f988891bef.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/29386cd58346796dd0f882ecbed4074b13f88eb43f49e7332d2ec7f988891bef.jpg)

![2a370c0140bb516167abd887d44927545dd2b7b01455134dd29902a66a295803.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/2a370c0140bb516167abd887d44927545dd2b7b01455134dd29902a66a295803.jpg)

![2ec9ce6f44cee67b93ff7478d0a9b5b5b397f4cd935fed5c191aad8b973002f5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/2ec9ce6f44cee67b93ff7478d0a9b5b5b397f4cd935fed5c191aad8b973002f5.jpg)

![32241e372f0a4339400b3516d4da3eb1335808cd095b9efd2681e05512a6b887.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/32241e372f0a4339400b3516d4da3eb1335808cd095b9efd2681e05512a6b887.jpg)

![3410daa9e835f40a12c4720b85b24417d2b99fa5c53d8499801595d059174082.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3410daa9e835f40a12c4720b85b24417d2b99fa5c53d8499801595d059174082.jpg)

![3475d4721f397fc53406c81127842387c41b130aedfa34bc133bed2c3d0324f6.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3475d4721f397fc53406c81127842387c41b130aedfa34bc133bed2c3d0324f6.jpg)

![3606c0e11c26c64b46916fd9755c9bc82721a0659f549f9ebab040e5b03e4836.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3606c0e11c26c64b46916fd9755c9bc82721a0659f549f9ebab040e5b03e4836.jpg)

![36368d56a7829a83a930b5e5e627af09903ac0d21e1fc8ed6b6be7da686288fb.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/36368d56a7829a83a930b5e5e627af09903ac0d21e1fc8ed6b6be7da686288fb.jpg)

![3c4621b29dda1a59faa515f43dd3faafa8f2b41dc598c973395480e3c4727b88.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3c4621b29dda1a59faa515f43dd3faafa8f2b41dc598c973395480e3c4727b88.jpg)

![3cb69c69deffc12d179557609614dbb30cb18b8f11502a57bc19b0cf3d9ec6e2.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3cb69c69deffc12d179557609614dbb30cb18b8f11502a57bc19b0cf3d9ec6e2.jpg)

![3d779215d9d4af3116c6c37cdbaf24c4a77b11fed8af338345c4330a91045d71.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/3d779215d9d4af3116c6c37cdbaf24c4a77b11fed8af338345c4330a91045d71.jpg)

![41848645b10d6c2e9acf18c6ef8b187025b60907bee24b083832fcb2074ea908.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/41848645b10d6c2e9acf18c6ef8b187025b60907bee24b083832fcb2074ea908.jpg)

![419a898454ca4501f70d8a69e8a37bb32be5e35fe0f42793cc8da9ed5c4e22f8.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/419a898454ca4501f70d8a69e8a37bb32be5e35fe0f42793cc8da9ed5c4e22f8.jpg)

![455f4e6e9064616a544bacca38f589dac6fb4f68cfd292ca6d3fb9834339d181.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/455f4e6e9064616a544bacca38f589dac6fb4f68cfd292ca6d3fb9834339d181.jpg)

![4826adfb6c134169577371ec32aac975e9fd1703095f4d70376cf13591afeb1f.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/4826adfb6c134169577371ec32aac975e9fd1703095f4d70376cf13591afeb1f.jpg)

![486a07b9164f6874e5f522d0ac82ce7abfa3696dc5429e9e8c6c8aa36878c4ef.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/486a07b9164f6874e5f522d0ac82ce7abfa3696dc5429e9e8c6c8aa36878c4ef.jpg)

![5107fed09fc5be473ec7afd62207ce3587779d71583d06f752f75cae6cf1b327.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/5107fed09fc5be473ec7afd62207ce3587779d71583d06f752f75cae6cf1b327.jpg)

![5388ab0aa4b2811d79f12fade69b8bfb01184b88229335dd4e6b6ae2f9edcef8.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/5388ab0aa4b2811d79f12fade69b8bfb01184b88229335dd4e6b6ae2f9edcef8.jpg)

![53c69bfdadaf69c733ca0eb8788e6efeed59d50d7b70b51c1c2ffde3f7ea244f.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/53c69bfdadaf69c733ca0eb8788e6efeed59d50d7b70b51c1c2ffde3f7ea244f.jpg)

![541baecd34fcf1a98dea4815533cdd95f950165457206d649aba7cb95e9d452d.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/541baecd34fcf1a98dea4815533cdd95f950165457206d649aba7cb95e9d452d.jpg)

![55011b953d24d31e8d69943fb8597e9d00b10e9f0c1e1c03c2d8dc976b3979c7.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/55011b953d24d31e8d69943fb8597e9d00b10e9f0c1e1c03c2d8dc976b3979c7.jpg)

![56f19c3be87e2f88b7a5b498c179319c42241385479cd9b6925d37176fadf504.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/56f19c3be87e2f88b7a5b498c179319c42241385479cd9b6925d37176fadf504.jpg)

![5d9e6dc472690addda2eeb1d87ed630a891c8492fc9572539b4d1babd538d4b2.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/5d9e6dc472690addda2eeb1d87ed630a891c8492fc9572539b4d1babd538d4b2.jpg)

![6163cdcea297fa48b9b7ae038f8a41f38ae9aa8387e4dff966b3f2bc3f8a746d.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/6163cdcea297fa48b9b7ae038f8a41f38ae9aa8387e4dff966b3f2bc3f8a746d.jpg)

![61e92a3ffceee01c3b215fdaf5874dc4022dacce486e9809c7958fe8ad4793ae.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/61e92a3ffceee01c3b215fdaf5874dc4022dacce486e9809c7958fe8ad4793ae.jpg)

![642a67d99288cd3e0115ec6d2e09b2b53025d68ceae301b14153bd866cd6a39c.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/642a67d99288cd3e0115ec6d2e09b2b53025d68ceae301b14153bd866cd6a39c.jpg)

![6deaf186da590c2667db8ef4a0f0bedf8beb9b4152a4e3ec8fc1f76e29cd38a7.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/6deaf186da590c2667db8ef4a0f0bedf8beb9b4152a4e3ec8fc1f76e29cd38a7.jpg)

![6e43866e67737685401d8af33a7452bba0c70a3b2fc31f06e448eb7a4198c21d.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/6e43866e67737685401d8af33a7452bba0c70a3b2fc31f06e448eb7a4198c21d.jpg)

![72aafc27e9408c7746265b65c684e10eb41a37a44b4108484a47abcb7bb0e7c5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/72aafc27e9408c7746265b65c684e10eb41a37a44b4108484a47abcb7bb0e7c5.jpg)

![75982fe4449f02447f4056b4ff7b1089db308229ed4769d2a13b48333ef9d5a9.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/75982fe4449f02447f4056b4ff7b1089db308229ed4769d2a13b48333ef9d5a9.jpg)

![75b3f4fe67aa49120074f849a42641e135efe8791e2a513878c2bcf643719fb9.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/75b3f4fe67aa49120074f849a42641e135efe8791e2a513878c2bcf643719fb9.jpg)

![7990dd676a8e6d5dd74ea190ed3fb4e6fe09362d7c48c980af230702d2c48eb0.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/7990dd676a8e6d5dd74ea190ed3fb4e6fe09362d7c48c980af230702d2c48eb0.jpg)

![7e89f0f108237069e2cd4128a49177de9a5c54765471c579c2c7b19d84401cd5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/7e89f0f108237069e2cd4128a49177de9a5c54765471c579c2c7b19d84401cd5.jpg)

![81d8942e427ffad7f21b7cfac321034c2fc3eae22e05f8cf7fa3ae38c29d5853.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/81d8942e427ffad7f21b7cfac321034c2fc3eae22e05f8cf7fa3ae38c29d5853.jpg)

![8c2a5b145355dbc2ae4d1858ca1be24d55c70b6d00aaf70e26fe02162cd0c134.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/8c2a5b145355dbc2ae4d1858ca1be24d55c70b6d00aaf70e26fe02162cd0c134.jpg)

![9299785976e26be5ec06cb941a44752e7ad290ea4c0d7d60156aa72968f42d52.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9299785976e26be5ec06cb941a44752e7ad290ea4c0d7d60156aa72968f42d52.jpg)

![92cf7651b31b40cdd54fafe6ebd585450090ae6faeab4a63a34e68f67a093780.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/92cf7651b31b40cdd54fafe6ebd585450090ae6faeab4a63a34e68f67a093780.jpg)

![94a2cd58ad8bdeb0daa2933a954ed46d1120fecd88b1a4840e2f9f014c767f44.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/94a2cd58ad8bdeb0daa2933a954ed46d1120fecd88b1a4840e2f9f014c767f44.jpg)

![94c2252b870147ee2b76fae119006302f149080a13853aa3cf29b2eaded4744c.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/94c2252b870147ee2b76fae119006302f149080a13853aa3cf29b2eaded4744c.jpg)

![950ada1a303f0ea778bdaba2bdb13cff028056be90fd931b35d4c87356905fdb.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/950ada1a303f0ea778bdaba2bdb13cff028056be90fd931b35d4c87356905fdb.jpg)

![966a42b636cc8ad0c3ad30c609023be5814ed2f7fa62b9bd681d0d8595b6decd.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/966a42b636cc8ad0c3ad30c609023be5814ed2f7fa62b9bd681d0d8595b6decd.jpg)

![97e739007a2de85825bf721e819a15b9131212a1af04e6f65763d64924e2213b.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/97e739007a2de85825bf721e819a15b9131212a1af04e6f65763d64924e2213b.jpg)

![9a248806b863a0717e88084384c1f8bbef18106b79b5a44560164f9b3f22caa9.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9a248806b863a0717e88084384c1f8bbef18106b79b5a44560164f9b3f22caa9.jpg)

![9bdd3f338cc8368a45d10b108d576e42660596ad0d57401837070a9edfdd8d3b.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9bdd3f338cc8368a45d10b108d576e42660596ad0d57401837070a9edfdd8d3b.jpg)

![9c7c1e8ee29eca05d2286aaeb5286b5c15d911601f635c4f7fe379923cd6e0f5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9c7c1e8ee29eca05d2286aaeb5286b5c15d911601f635c4f7fe379923cd6e0f5.jpg)

![9cc664454ad08665bcc46d7732def8eeca4d90704a0289971161fc34112e4486.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9cc664454ad08665bcc46d7732def8eeca4d90704a0289971161fc34112e4486.jpg)

![9d6c4e5b40bb0033653f956bb2bd74342a20d15eab17f6d103ad593d57801ac1.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/9d6c4e5b40bb0033653f956bb2bd74342a20d15eab17f6d103ad593d57801ac1.jpg)

![a521163f40ee9838399ae976bce91cc452352bb28c8af1e2b93ef9a6b64a0ee2.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/a521163f40ee9838399ae976bce91cc452352bb28c8af1e2b93ef9a6b64a0ee2.jpg)

![a90081ff33b25d82f785c8bd957eef81370bd128b45adb6d914f788ecd178073.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/a90081ff33b25d82f785c8bd957eef81370bd128b45adb6d914f788ecd178073.jpg)

![a9fe9afdc5d95644a10213b7bcc72ff38282177dd394bb321feaa95ebc590bfa.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/a9fe9afdc5d95644a10213b7bcc72ff38282177dd394bb321feaa95ebc590bfa.jpg)

![ad8088196af9cf1f97aafd19e2843fbdfb34c05b90d14bc559d852b66e00fc19.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/ad8088196af9cf1f97aafd19e2843fbdfb34c05b90d14bc559d852b66e00fc19.jpg)

![ae91563f2dc0e2380dc86d7b923a99c67771dacb75dcc32fe5b18b284f79d934.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/ae91563f2dc0e2380dc86d7b923a99c67771dacb75dcc32fe5b18b284f79d934.jpg)

![b43aec83170bd9965e5a27bdfd5f6ebf2a6de8664575fdaadced1e348375a0b3.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/b43aec83170bd9965e5a27bdfd5f6ebf2a6de8664575fdaadced1e348375a0b3.jpg)

![b4858a0a1b26d77e2b54c06698e04559c56e8437db553ff689559465965bb83a.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/b4858a0a1b26d77e2b54c06698e04559c56e8437db553ff689559465965bb83a.jpg)

![b6cef37852dda686ed24066dfb776b233e78fb4c35d3c6b0420d58a44a9f2a30.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/b6cef37852dda686ed24066dfb776b233e78fb4c35d3c6b0420d58a44a9f2a30.jpg)

![b97f779d2e44ec4f67596c574da1d6f0103638a9316e5e5aaf047dbc0760f4ac.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/b97f779d2e44ec4f67596c574da1d6f0103638a9316e5e5aaf047dbc0760f4ac.jpg)

![b9c7fef662e4ea42702ddc9baccecf1711f23979e4cfdfcba351a53270be0e7c.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/b9c7fef662e4ea42702ddc9baccecf1711f23979e4cfdfcba351a53270be0e7c.jpg)

![bc95c738155273cf63c9cc4ec5049b9472f3634cff75ff8ab8a87ec9594252de.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/bc95c738155273cf63c9cc4ec5049b9472f3634cff75ff8ab8a87ec9594252de.jpg)

![c051d4dee80c776d587de4dd36a71dab9b10133853de64ea7ad8dd5b1902d962.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/c051d4dee80c776d587de4dd36a71dab9b10133853de64ea7ad8dd5b1902d962.jpg)

![c18a4f5291b7fb0a441e568833a58418ab57db06df8532ae041894e98666d4fe.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/c18a4f5291b7fb0a441e568833a58418ab57db06df8532ae041894e98666d4fe.jpg)

![c22cfe0248d69a7f23cce0e820b212852872644baf3c1deb86f2600412648a5c.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/c22cfe0248d69a7f23cce0e820b212852872644baf3c1deb86f2600412648a5c.jpg)

![ca2d6476000c9f215b25bc715983fc2ced7781878fc02b174379f21360f97f6f.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/ca2d6476000c9f215b25bc715983fc2ced7781878fc02b174379f21360f97f6f.jpg)

![cf2d94ab185ca90d42e267679a326e021473f841eacc997b5b689b591ea86823.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/cf2d94ab185ca90d42e267679a326e021473f841eacc997b5b689b591ea86823.jpg)

![d0b67b613e39ff5dd8506123ade2f55221fa26752245f80771daf7881d219b34.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d0b67b613e39ff5dd8506123ade2f55221fa26752245f80771daf7881d219b34.jpg)

![d0c96c7da20266105a36ef395b66cfd21a97d66a637f9f4b11a27b5dba1e87d5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d0c96c7da20266105a36ef395b66cfd21a97d66a637f9f4b11a27b5dba1e87d5.jpg)

![d1538f9c283fed50f106e034fc35f218e6e096f7c92d89fe24155938f52be392.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d1538f9c283fed50f106e034fc35f218e6e096f7c92d89fe24155938f52be392.jpg)

![d22f133d94c0e388e30992b5bfae37cd82d15d8656bb0b54fcccc31466cb255e.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d22f133d94c0e388e30992b5bfae37cd82d15d8656bb0b54fcccc31466cb255e.jpg)

![d39fb4d5f0169b0750d3129373f2430f76015411396f72cacfa6890f0f6627aa.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d39fb4d5f0169b0750d3129373f2430f76015411396f72cacfa6890f0f6627aa.jpg)

![d46187bb8c47fc23224b9a4f87592a7ddce4cc76de4692dc5a7c15ac7b735a9a.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d46187bb8c47fc23224b9a4f87592a7ddce4cc76de4692dc5a7c15ac7b735a9a.jpg)

![d472559d439181683ca5e8aedb72fe406e0b039fd0ce9e1ec415f3efac22bc57.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d472559d439181683ca5e8aedb72fe406e0b039fd0ce9e1ec415f3efac22bc57.jpg)

![d4da387d05177db7c1a3ce1e6fc293a42b71fafa4ea2d0e3632b17f5e4b944b1.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d4da387d05177db7c1a3ce1e6fc293a42b71fafa4ea2d0e3632b17f5e4b944b1.jpg)

![d9195f4ba99861615e87a1f482ead722a3336d339fb7b206d41acc791de2da50.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/d9195f4ba99861615e87a1f482ead722a3336d339fb7b206d41acc791de2da50.jpg)

![db4d1c082f36891fc4760ce2dea684cd7b5b099aa2490d9808da750bbf949d82.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/db4d1c082f36891fc4760ce2dea684cd7b5b099aa2490d9808da750bbf949d82.jpg)

![de4a13f1301b47271131f30ae46e4f7f145a3b9073a9bd843c65abe4871fca7a.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/de4a13f1301b47271131f30ae46e4f7f145a3b9073a9bd843c65abe4871fca7a.jpg)

![df9c8601999690fdd0b55666aa4d08d9fd7136f380b6f002e04e04834a5a0766.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/df9c8601999690fdd0b55666aa4d08d9fd7136f380b6f002e04e04834a5a0766.jpg)

![e25098dbde52e7dba6f3e2a319a89e008f020e9e4e993018282fd2f26d1ce5fd.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/e25098dbde52e7dba6f3e2a319a89e008f020e9e4e993018282fd2f26d1ce5fd.jpg)

![e4b007e394e5bc156e4439262a007047fa267096d40b3ce8fe0e9d32feda25e8.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/e4b007e394e5bc156e4439262a007047fa267096d40b3ce8fe0e9d32feda25e8.jpg)

![eb043bdfe27cf1043e4091674ccfcaa9cd30838644b8ac8cb1ab2563eb803ab5.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/eb043bdfe27cf1043e4091674ccfcaa9cd30838644b8ac8cb1ab2563eb803ab5.jpg)

![ed2ef9442832010cc70224bc6f4f2e8944fd00f1214f6f0cbc689965766df779.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/ed2ef9442832010cc70224bc6f4f2e8944fd00f1214f6f0cbc689965766df779.jpg)

![f06b59c9beb3ffe06beeff9bbb072009f50c7f34fe936052780b4b745d5f1ba4.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/f06b59c9beb3ffe06beeff9bbb072009f50c7f34fe936052780b4b745d5f1ba4.jpg)

![f0c6b746e7f96388d5b61df079f32084805f9b1facbab78652652dccac0db18c.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/f0c6b746e7f96388d5b61df079f32084805f9b1facbab78652652dccac0db18c.jpg)

![f0d5f8c54f7050e05c118b260a9c91000bc4502cdaa1a133b7509f686448ef28.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/f0d5f8c54f7050e05c118b260a9c91000bc4502cdaa1a133b7509f686448ef28.jpg)

![f5ea45b7661914394e44d2b35dfd99ca3c50dbf8914b76268df65c43766d5525.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/f5ea45b7661914394e44d2b35dfd99ca3c50dbf8914b76268df65c43766d5525.jpg)

![fb636d498c522ce337f3bb038e19823c58b1ff37c756541efcbc45ca04455c8a.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/images/fb636d498c522ce337f3bb038e19823c58b1ff37c756541efcbc45ca04455c8a.jpg)

### Tables

![0954400d80457beb0028d4330d7a1e8cabd9bdf24141555b0a662f1734bb2d92.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/0954400d80457beb0028d4330d7a1e8cabd9bdf24141555b0a662f1734bb2d92.jpg)

![16c8a2b2d957eeb597ac4bf7362790acde2b203f0b69f76e8c24fd58f0a1edeb.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/16c8a2b2d957eeb597ac4bf7362790acde2b203f0b69f76e8c24fd58f0a1edeb.jpg)

![30cac998c2c62732b85399d8659c24bab13369d9b7f5ce9ddfa4621dac50db46.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/30cac998c2c62732b85399d8659c24bab13369d9b7f5ce9ddfa4621dac50db46.jpg)

![380be7eaa8872851ba950748357b54c8eb858963dd60bf51c0dbbc4a51f28dd6.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/380be7eaa8872851ba950748357b54c8eb858963dd60bf51c0dbbc4a51f28dd6.jpg)

![38787b20aae1373aa8d4280c3c81b98e135b2362268b3208fa1c056c64774df6.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/38787b20aae1373aa8d4280c3c81b98e135b2362268b3208fa1c056c64774df6.jpg)

![7737b2008e925f692800b36ffeab6cd8be7012d7fbb11a9c977d4a0c9ffc3e75.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/7737b2008e925f692800b36ffeab6cd8be7012d7fbb11a9c977d4a0c9ffc3e75.jpg)

![834cffc2ee117fac4637bdcb7a8c738255b23ffa5acf8c155ce62d42915c03fe.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/834cffc2ee117fac4637bdcb7a8c738255b23ffa5acf8c155ce62d42915c03fe.jpg)

![8632156071e84d7efa62e7d2ad585f7b968163c4a0a97fe9d2ff7bf3ba6d4f40.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/8632156071e84d7efa62e7d2ad585f7b968163c4a0a97fe9d2ff7bf3ba6d4f40.jpg)

![932cc7b81b578710afe24665a4eeea22ef3f2851c6c05c3b09a3718c4c164e8e.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/932cc7b81b578710afe24665a4eeea22ef3f2851c6c05c3b09a3718c4c164e8e.jpg)

![98748e7293b22665fccdb9506323e9b8e37c4f0c433f12e833b261b3a7684852.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/98748e7293b22665fccdb9506323e9b8e37c4f0c433f12e833b261b3a7684852.jpg)

![bbac6bd5ff8d1626350f51dbe16d0ca4da0b9f95dfcdea3a1f93695155516e7f.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/bbac6bd5ff8d1626350f51dbe16d0ca4da0b9f95dfcdea3a1f93695155516e7f.jpg)

![c8b8a6faaa2a8868a285f236f82f619b231766493d3377cc29bc104db2a7aa5e.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/c8b8a6faaa2a8868a285f236f82f619b231766493d3377cc29bc104db2a7aa5e.jpg)

![d89cb227ce0943abb02ed17b9bfed9b1d0d60d60877b9ce68ba4119ef7a435c9.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/d89cb227ce0943abb02ed17b9bfed9b1d0d60d60877b9ce68ba4119ef7a435c9.jpg)

![e36c381eb6591711757e2a465d7e5e9d2686197d65668b192a9ed82aeb4ce49f.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/e36c381eb6591711757e2a465d7e5e9d2686197d65668b192a9ed82aeb4ce49f.jpg)

![ecc7bd28a5eb45d591459588628f9a69fab5f17002652ff1ad93a183b8e328de.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/ecc7bd28a5eb45d591459588628f9a69fab5f17002652ff1ad93a183b8e328de.jpg)

![efbab84670093f6a1ce44c5ae811f957b21fbf0f9bbd7f568f5b6dddbbdba5f0.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/efbab84670093f6a1ce44c5ae811f957b21fbf0f9bbd7f568f5b6dddbbdba5f0.jpg)

![fc351451829733cb36752ae6c1428bc219b1deccd177fe8b65ca14bfd49952da.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/fc351451829733cb36752ae6c1428bc219b1deccd177fe8b65ca14bfd49952da.jpg)

![fc724b3acda2d6b23c3369bc6ee3cb86898a7ffbafdb9b44a471312a8f6f39f2.jpg](../icml_results/3822_Hyperbolic%20Neural%20Operator/tables/fc724b3acda2d6b23c3369bc6ee3cb86898a7ffbafdb9b44a471312a8f6f39f2.jpg)

<a id="paper-3783"></a>

## CURE: Context-driven Diffusion with Progressive Expansion for Single Domain Generalization in Time Series Classification


### Images

![830a858cdfbda221c3408f7cc4bb1181a19900a1b53704d432ee44d01f0e24b8.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/images/830a858cdfbda221c3408f7cc4bb1181a19900a1b53704d432ee44d01f0e24b8.jpg)

![a17285a097e3ab952b473f98be3f22c9102f12f87300d6da2b379a9c800d9e5c.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/images/a17285a097e3ab952b473f98be3f22c9102f12f87300d6da2b379a9c800d9e5c.jpg)

![ac4658d029b1416c0ad8e6db476cb21a34b8e4d4b045f2b10e6312e8e322b6e4.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/images/ac4658d029b1416c0ad8e6db476cb21a34b8e4d4b045f2b10e6312e8e322b6e4.jpg)

![dc58ac5efd31acd8cb88712dde87940c1e5ed2c99ffd1175905de380b0a54417.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/images/dc58ac5efd31acd8cb88712dde87940c1e5ed2c99ffd1175905de380b0a54417.jpg)

### Tables

![04680cad32fef9eeead49a6e1d7d2e3836790e25393cf36915790fc6fbb18b6a.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/04680cad32fef9eeead49a6e1d7d2e3836790e25393cf36915790fc6fbb18b6a.jpg)

![186437326f1e7312c2529b1ec1b2fb007a5f9166c0706cfcd4d7197039c71f82.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/186437326f1e7312c2529b1ec1b2fb007a5f9166c0706cfcd4d7197039c71f82.jpg)

![648bd6ddded2df89c7fa55dc8561c4ec453bd8317aeea05018d535e05985f507.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/648bd6ddded2df89c7fa55dc8561c4ec453bd8317aeea05018d535e05985f507.jpg)

![792964389bc6008877e2353b1442e6865fc9f4125530ef82541bbd57d22828ce.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/792964389bc6008877e2353b1442e6865fc9f4125530ef82541bbd57d22828ce.jpg)

![7ba7c7a569b888ad53c41ae7db8d2f92b657017a3d2ecc3b57d321c0474a382d.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/7ba7c7a569b888ad53c41ae7db8d2f92b657017a3d2ecc3b57d321c0474a382d.jpg)

![cc7851b5ea222289c7a13855322aeb69ab7e50ac24c7abf3e233764d77287088.jpg](../icml_results/3823_CURE_%20Context-driven%20Diffusion%20with%20Progressive%20Expansion%20for%20Single%20Domain%20Generalization%20in%20Time%20S/tables/cc7851b5ea222289c7a13855322aeb69ab7e50ac24c7abf3e233764d77287088.jpg)

<a id="paper-3784"></a>

## Great Minds Think Alike: Contextual Tacit Communication for Decentralized LLM-Agent Cooperation


### Images

![7c7e71e8dbcb34160d8b3b5d654b50b613a645623c5a98ccd37d5d35fe33be71.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/images/7c7e71e8dbcb34160d8b3b5d654b50b613a645623c5a98ccd37d5d35fe33be71.jpg)

![a445a6845e2767746f97ffc37f4d978a5b947e543f3663a2c2f5d00250306727.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/images/a445a6845e2767746f97ffc37f4d978a5b947e543f3663a2c2f5d00250306727.jpg)

![ba59b02aef7a2b86a7dda41519c3776ba7b2780a36e3d8f79a90e241cc7a1e4e.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/images/ba59b02aef7a2b86a7dda41519c3776ba7b2780a36e3d8f79a90e241cc7a1e4e.jpg)

### Tables

![050df6354900fb8a57b28a4bd02b457b2806ea6ec2cc028cf8c07ebbd9ffa1b5.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/050df6354900fb8a57b28a4bd02b457b2806ea6ec2cc028cf8c07ebbd9ffa1b5.jpg)

![17fb9db26e88ec1bb130846a8b0a145b10c8c07db12c04cd9e09e67dc1c78f63.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/17fb9db26e88ec1bb130846a8b0a145b10c8c07db12c04cd9e09e67dc1c78f63.jpg)

![40454a878aecc58365a28f23feb7b633741dba6bc082eede908dff5865c32ca2.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/40454a878aecc58365a28f23feb7b633741dba6bc082eede908dff5865c32ca2.jpg)

![560e1cad1e1a3a2776d624501c84a415358a30a56a36cd51424a93ec15fa5902.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/560e1cad1e1a3a2776d624501c84a415358a30a56a36cd51424a93ec15fa5902.jpg)

![6019470beefdbbe5c25e3210f4c92bd73688f7973d343dc0a1c92cb843f14143.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/6019470beefdbbe5c25e3210f4c92bd73688f7973d343dc0a1c92cb843f14143.jpg)

![7a0244befda391a295296e45504cc14605baa33322d13e9894c3f108e5e3adc2.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/7a0244befda391a295296e45504cc14605baa33322d13e9894c3f108e5e3adc2.jpg)

![7faf9227b7996d00d884a609d3dac613b7142cc37d97ba2f6583f58771c0f3b3.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/7faf9227b7996d00d884a609d3dac613b7142cc37d97ba2f6583f58771c0f3b3.jpg)

![8f579a595f9b12f806239729bf87a77812178b687719be50934272a982502dc6.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/8f579a595f9b12f806239729bf87a77812178b687719be50934272a982502dc6.jpg)

![95fe22df8dcb52e02080419c15db21611443ce265aff3b4121e7e77661f5baa3.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/95fe22df8dcb52e02080419c15db21611443ce265aff3b4121e7e77661f5baa3.jpg)

![c54968bdd90171a9fcb84e2fab9eaf620b5f68b643eb0ae5698f692cc25a214a.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/c54968bdd90171a9fcb84e2fab9eaf620b5f68b643eb0ae5698f692cc25a214a.jpg)

![d02c50f68dae486208aa24bfd12b44d5012df2901001b42d577ed17e36e2813e.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/d02c50f68dae486208aa24bfd12b44d5012df2901001b42d577ed17e36e2813e.jpg)

![d62e1ebdedfe9fd9aacd48032dc07ffaf6514e4d5346d2769196bae67400fc08.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/d62e1ebdedfe9fd9aacd48032dc07ffaf6514e4d5346d2769196bae67400fc08.jpg)

![d6905546caa2da609d233f9b4aa28b80beaa33df13393887e587b801510471ed.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/d6905546caa2da609d233f9b4aa28b80beaa33df13393887e587b801510471ed.jpg)

![e6f6c421c3abe0c4ba6f143791edb3830cdb3feb30ad08a9fe2e674193d7dbe3.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/e6f6c421c3abe0c4ba6f143791edb3830cdb3feb30ad08a9fe2e674193d7dbe3.jpg)

![ec1a1194899398c8c056500edacd86a21c8fbea9cf6b4a209a8642ea3743a2d6.jpg](../icml_results/3824_Great%20Minds%20Think%20Alike_%20Contextual%20Tacit%20Communication%20for%20Decentralized%20LLM-Agent%20Cooperation/tables/ec1a1194899398c8c056500edacd86a21c8fbea9cf6b4a209a8642ea3743a2d6.jpg)

<a id="paper-3785"></a>

## Neural Minimum Weight Perfect Matching for Quantum Error Codes


### Images

![0e7e3957af415611628206b7bd1a882ad6d8b2c03d3d9ac0ea5b4053f8e658a0.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/0e7e3957af415611628206b7bd1a882ad6d8b2c03d3d9ac0ea5b4053f8e658a0.jpg)

![14444dbecd88e0a51b64bd639ed531c575485784412812bd18a9e2ca4eea681d.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/14444dbecd88e0a51b64bd639ed531c575485784412812bd18a9e2ca4eea681d.jpg)

![147fd83dab49bf823e6424ba32aba29a8b1135cab2aebe10c2d047a7d16349e6.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/147fd83dab49bf823e6424ba32aba29a8b1135cab2aebe10c2d047a7d16349e6.jpg)

![3fca6ac34b7019c76a355428fc80c679bfddcfcdef3c96c4a8b4729a430db370.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/3fca6ac34b7019c76a355428fc80c679bfddcfcdef3c96c4a8b4729a430db370.jpg)

![a1777443f6a38dd126402ca8d10d3c79ddb29939e11897240a9bb55a55ae8194.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/a1777443f6a38dd126402ca8d10d3c79ddb29939e11897240a9bb55a55ae8194.jpg)

![a37e39bb7fc3261a7fab40173367da08eded8bb3943e8d445c548d7cfa1997f1.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/a37e39bb7fc3261a7fab40173367da08eded8bb3943e8d445c548d7cfa1997f1.jpg)

![a87e94750f9c3651171b98dd7939df68a3db148c2a8faccfadc5e14d01512fbd.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/a87e94750f9c3651171b98dd7939df68a3db148c2a8faccfadc5e14d01512fbd.jpg)

![aae701611c0cf7593acf309422fccd762e188658df372f160fe0061a22b7e071.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/aae701611c0cf7593acf309422fccd762e188658df372f160fe0061a22b7e071.jpg)

![c835cf2bac88f5674b7812bf25aad7cce4cf4d96252ec20614197d4a907a872b.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/images/c835cf2bac88f5674b7812bf25aad7cce4cf4d96252ec20614197d4a907a872b.jpg)

### Tables

![3f21ef1ffc13dc7306aef5519f26af01802939f1b11b3ecc6ddcd7a535971cd5.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/3f21ef1ffc13dc7306aef5519f26af01802939f1b11b3ecc6ddcd7a535971cd5.jpg)

![79e5b61a60ef52bbd470923751015a17fc102e86517605ea14997f86d7636d1c.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/79e5b61a60ef52bbd470923751015a17fc102e86517605ea14997f86d7636d1c.jpg)

![9783e3a8085282dc8724b3890337a97f03e81bdc25c3a1d28aa459d3927167de.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/9783e3a8085282dc8724b3890337a97f03e81bdc25c3a1d28aa459d3927167de.jpg)

![a51745bcd90977da2e3464fdc0cac524340aa90054f4133788f2e6e8ce501351.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/a51745bcd90977da2e3464fdc0cac524340aa90054f4133788f2e6e8ce501351.jpg)

![dd2791eb40cf6e023e3bc1f26be57ca51ecb2b4c7f83da9955914ff96c50b66f.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/dd2791eb40cf6e023e3bc1f26be57ca51ecb2b4c7f83da9955914ff96c50b66f.jpg)

![e3a0b62e42592848e03e574bf9b93a9da48945ae92eca2f918634c7d3d3bd9c0.jpg](../icml_results/3825_Neural%20Minimum%20Weight%20Perfect%20Matching%20for%20Quantum%20Error%20Codes/tables/e3a0b62e42592848e03e574bf9b93a9da48945ae92eca2f918634c7d3d3bd9c0.jpg)

<a id="paper-3786"></a>

## CAMP: Coherent Alignment of Multimodal Prototypes for Explainable Complementary Learning


### Images

![03cc970d34fa15b61f47ff056883df7b75bcd95614634e2d9fda87d8ab8f2ba4.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/images/03cc970d34fa15b61f47ff056883df7b75bcd95614634e2d9fda87d8ab8f2ba4.jpg)

![6987108b5aa5393b6683d59af2fde934ee9a974453c96ad0f0e48bdbf2d68332.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/images/6987108b5aa5393b6683d59af2fde934ee9a974453c96ad0f0e48bdbf2d68332.jpg)

![9ce90b58c26b7217de29d33e9507838f63427eae90cec53661085feaef513f53.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/images/9ce90b58c26b7217de29d33e9507838f63427eae90cec53661085feaef513f53.jpg)

![d6285d18bead78e64b4cf661eeb511c59151adac1b6d697ee4d79a263d535b0e.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/images/d6285d18bead78e64b4cf661eeb511c59151adac1b6d697ee4d79a263d535b0e.jpg)

### Tables

![029d47320ca313268361185a2cefb705bc6cc75b1db4652950536ddba5bfe18e.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/029d47320ca313268361185a2cefb705bc6cc75b1db4652950536ddba5bfe18e.jpg)

![08aabd9d584323d72cf26b55e021ce9d05c4264b8e8c5abd19db0a93a25cbf22.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/08aabd9d584323d72cf26b55e021ce9d05c4264b8e8c5abd19db0a93a25cbf22.jpg)

![0fe3bfc410bf5ebb69695244d4b101b3f0eb23e186073ca3d7f507a9b5ed6de6.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/0fe3bfc410bf5ebb69695244d4b101b3f0eb23e186073ca3d7f507a9b5ed6de6.jpg)

![14990be09211e33fb1f3d6cf848a6cd583fca45da8bb1e5614f9b19b8a96f766.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/14990be09211e33fb1f3d6cf848a6cd583fca45da8bb1e5614f9b19b8a96f766.jpg)

![1a8952613682d3dbbbe8a0fe2ae17dac72722cff72405d50dce30dfeb1ca4c97.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/1a8952613682d3dbbbe8a0fe2ae17dac72722cff72405d50dce30dfeb1ca4c97.jpg)

![2eef4e99e96d7e8956989e8393ebf596afcf341b7adedf38d77c369880baa091.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/2eef4e99e96d7e8956989e8393ebf596afcf341b7adedf38d77c369880baa091.jpg)

![470e320615dd11fe9359f1db0f528d20ce4e20cb587727a2f3b854fd01b979ed.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/470e320615dd11fe9359f1db0f528d20ce4e20cb587727a2f3b854fd01b979ed.jpg)

![5236efafee8aa584291a397bf12bc7556169b56566c984c1c3275aace688f1c6.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/5236efafee8aa584291a397bf12bc7556169b56566c984c1c3275aace688f1c6.jpg)

![5483fdefbfb5b60ae3d38eee5bae635c9bdf326b1b6f31f527312e5717a33a73.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/5483fdefbfb5b60ae3d38eee5bae635c9bdf326b1b6f31f527312e5717a33a73.jpg)

![566b51041b7ce83ab519aa0958ad8a21c04fc9a362752a49aa06f1735f7969d6.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/566b51041b7ce83ab519aa0958ad8a21c04fc9a362752a49aa06f1735f7969d6.jpg)

![6451387009858f85e3217222d5a87c881ad83c45cbc8cad98d110b069baccb5b.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/6451387009858f85e3217222d5a87c881ad83c45cbc8cad98d110b069baccb5b.jpg)

![7301753adc76051fa76a74b7145ff9cb393be46028a0260632c6e5eecb89422f.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/7301753adc76051fa76a74b7145ff9cb393be46028a0260632c6e5eecb89422f.jpg)

![84b384b2684398e3d9c3e8b0bcccdbdec04fc1024c651656c9664396c39b139f.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/84b384b2684398e3d9c3e8b0bcccdbdec04fc1024c651656c9664396c39b139f.jpg)

![9d4e302aa60509dcd7e80f692422de607df306fcc9b0941ca1a40a5240093d41.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/9d4e302aa60509dcd7e80f692422de607df306fcc9b0941ca1a40a5240093d41.jpg)

![9d97c691b832319476d052cae739679c06b7333bf1f9e10815366f54b4eb4c96.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/9d97c691b832319476d052cae739679c06b7333bf1f9e10815366f54b4eb4c96.jpg)

![a19b3c64bec512747e0cd9de8f984d735f2ccadc992568fb2cc37261b6511be8.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/a19b3c64bec512747e0cd9de8f984d735f2ccadc992568fb2cc37261b6511be8.jpg)

![b43643005424fdb9a0984985f7ba0491a07858f6c3d3405cb33cde872f362fef.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/b43643005424fdb9a0984985f7ba0491a07858f6c3d3405cb33cde872f362fef.jpg)

![ba9ef79f4ce0a0aad63aadbfaf32d600b50808db95d380a82cdb7b4a00f6ec70.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/ba9ef79f4ce0a0aad63aadbfaf32d600b50808db95d380a82cdb7b4a00f6ec70.jpg)

![c247ef9f908eed50340597024b36ed9e22daaa3d82713f654d3f83533cbc91c6.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/c247ef9f908eed50340597024b36ed9e22daaa3d82713f654d3f83533cbc91c6.jpg)

![c3232bbc91f06dc6fa4ee239ac77a7304c8cf7829071ea087537acc72c804f2f.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/c3232bbc91f06dc6fa4ee239ac77a7304c8cf7829071ea087537acc72c804f2f.jpg)

![c9577ba662e8e31369ff472068c7fc9bbf0fdc0098fc226502299c879fca901c.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/c9577ba662e8e31369ff472068c7fc9bbf0fdc0098fc226502299c879fca901c.jpg)

![d05a35836018b5a87d84f28878ca1e1b76cede95861c6db70498e3a024678323.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/d05a35836018b5a87d84f28878ca1e1b76cede95861c6db70498e3a024678323.jpg)

![db69919519a0fc8e64b7cece9de2934bb1217ce5969d0c35321982e89b2b9914.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/db69919519a0fc8e64b7cece9de2934bb1217ce5969d0c35321982e89b2b9914.jpg)

![e5edd8153174c649f9c5d4f26e1009e36add53c1d66b17385d7049fe67ddc72a.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/e5edd8153174c649f9c5d4f26e1009e36add53c1d66b17385d7049fe67ddc72a.jpg)

![f281771592b84d6f53c9c8882e10f01e8fd0fb240bce76d4a6b41a5122ac2618.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/f281771592b84d6f53c9c8882e10f01e8fd0fb240bce76d4a6b41a5122ac2618.jpg)

![f431665f0b3045a94f618c8a800f5c33368bf7c736670b1c12df2b1036afbf1e.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/f431665f0b3045a94f618c8a800f5c33368bf7c736670b1c12df2b1036afbf1e.jpg)

![fa6b74d7787d7391fb7de9353885bdbf2e105f8b9836de62630d3027bce9789b.jpg](../icml_results/3826_CAMP_%20Coherent%20Alignment%20of%20Multimodal%20Prototypes%20for%20Explainable%20Complementary%20Learning/tables/fa6b74d7787d7391fb7de9353885bdbf2e105f8b9836de62630d3027bce9789b.jpg)

<a id="paper-3787"></a>

## Privileged Information Distillation for Language Models


### Images

![0c203260696cc00a0abfd62cb213e6fc5c1623f8f984d29490caecf1e9288cac.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/0c203260696cc00a0abfd62cb213e6fc5c1623f8f984d29490caecf1e9288cac.jpg)

![115343f8441cc49d01f342b6c6bbfcf026611d57607f3ec68f3aeed5eae8c82e.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/115343f8441cc49d01f342b6c6bbfcf026611d57607f3ec68f3aeed5eae8c82e.jpg)

![1cd57ebec0e2d6d225f240bd571297356606b4ad3af9961c32f63a8f1efd3f2b.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/1cd57ebec0e2d6d225f240bd571297356606b4ad3af9961c32f63a8f1efd3f2b.jpg)

![1f4763811cbaeba55ca4a5f7d9dd4e6995f9cac1c04ed57c9b2fde501c133372.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/1f4763811cbaeba55ca4a5f7d9dd4e6995f9cac1c04ed57c9b2fde501c133372.jpg)

![209a595e01c02b296f5a64324ac8e6c1e1c8b99ef7e6e9ce1c4b272f2d61f1f5.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/209a595e01c02b296f5a64324ac8e6c1e1c8b99ef7e6e9ce1c4b272f2d61f1f5.jpg)

![2e56743204992447f56a6d6c8a324e671ae185e0977d11614fc9ce8d9177342d.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/2e56743204992447f56a6d6c8a324e671ae185e0977d11614fc9ce8d9177342d.jpg)

![7abedcfee61abfe9f316a96369113f81e9ef80a7f59361b099c847975d1091bb.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/7abedcfee61abfe9f316a96369113f81e9ef80a7f59361b099c847975d1091bb.jpg)

![9ae01dab22b6248233ddc7bf9cb3425c499967ce00bbd7bd5cc417eaef47eaa8.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/9ae01dab22b6248233ddc7bf9cb3425c499967ce00bbd7bd5cc417eaef47eaa8.jpg)

![a09afdfb570f18491f9a198c545c76e1c3102c0e849e3288c4cfe034003f2e42.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/a09afdfb570f18491f9a198c545c76e1c3102c0e849e3288c4cfe034003f2e42.jpg)

![ad4bdcfd1953a88a182b5aa83bf8be86a061bdc5939204c8f60b660d374f7208.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/ad4bdcfd1953a88a182b5aa83bf8be86a061bdc5939204c8f60b660d374f7208.jpg)

![b8238f954d4d8c157a7ba774f52f77b749aab2455cd4ca08b4d024c786010886.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/b8238f954d4d8c157a7ba774f52f77b749aab2455cd4ca08b4d024c786010886.jpg)

![ba7620267783c56650b2820a6b58037936b11f95db82b544d5574b1d5b69f45e.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/ba7620267783c56650b2820a6b58037936b11f95db82b544d5574b1d5b69f45e.jpg)

![bf76d540ca0b248b42fcd87a63769532ced4deba6d38a00fe8fcd6ce9cca0abf.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/bf76d540ca0b248b42fcd87a63769532ced4deba6d38a00fe8fcd6ce9cca0abf.jpg)

![c4a3fe8d3c80661da8baff38de7dae8f6a9ac9a5772282b4dede2629e12f1efb.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/c4a3fe8d3c80661da8baff38de7dae8f6a9ac9a5772282b4dede2629e12f1efb.jpg)

![d89586d912388a762aeaca077a0fa77c6e17434849eee91a2a84efb89cbafc9a.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/d89586d912388a762aeaca077a0fa77c6e17434849eee91a2a84efb89cbafc9a.jpg)

![e1abb5a3646be21ec597a1c7d848c91ab8a102d7e5ee2a363e548914c11b6d7b.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/e1abb5a3646be21ec597a1c7d848c91ab8a102d7e5ee2a363e548914c11b6d7b.jpg)

![e90673c724bbde21e9c16b2c2adbf93708c88f291d4d7dc0af89af8f84107e94.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/e90673c724bbde21e9c16b2c2adbf93708c88f291d4d7dc0af89af8f84107e94.jpg)

![fdfaf3797c048f7503a1e0d01a3e536878d65f7c054d3ed5ce2f7c3bc74588cb.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/images/fdfaf3797c048f7503a1e0d01a3e536878d65f7c054d3ed5ce2f7c3bc74588cb.jpg)

### Tables

![4c728f3831a866ffc9173d98ca3ceed416b262413b372d0dd94a4b3308676667.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/tables/4c728f3831a866ffc9173d98ca3ceed416b262413b372d0dd94a4b3308676667.jpg)

![c366e45a88b21383ee2e2ec794dc8dc50f509231bd1d364bcc0300d57b520111.jpg](../icml_results/3827_Privileged%20Information%20Distillation%20for%20Language%20Models/tables/c366e45a88b21383ee2e2ec794dc8dc50f509231bd1d364bcc0300d57b520111.jpg)

<a id="paper-3788"></a>

## Recursive Binding on a Budget: Subspace Carving in Order-$p$ Tensor Memories


### Images

![04f4d21c6ed52bd87d973ab89e42890ad673ab903266f5294b4d85f9bf05df36.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/04f4d21c6ed52bd87d973ab89e42890ad673ab903266f5294b4d85f9bf05df36.jpg)

![05d29c1c6d9a4177ff2d7d4f2a041d59eca91b75910f31c9f60606d5c8f99247.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/05d29c1c6d9a4177ff2d7d4f2a041d59eca91b75910f31c9f60606d5c8f99247.jpg)

![0b98ad174a3129467c5f7ae882c6e3a84330d7bdb16385ac1dba58129c00fd89.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/0b98ad174a3129467c5f7ae882c6e3a84330d7bdb16385ac1dba58129c00fd89.jpg)

![0cf02e994027c9acf82f5be381f2d6c824d3e09f299221467459cc1337829c65.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/0cf02e994027c9acf82f5be381f2d6c824d3e09f299221467459cc1337829c65.jpg)

![23a41953372b92e0603c75022ca38c378bf04b66d785a9d6918d60b6775f9496.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/23a41953372b92e0603c75022ca38c378bf04b66d785a9d6918d60b6775f9496.jpg)

![40bd9c9803b793e41323a6e5b27cba4f4b11e44cc6889575a7c821a60bebbd4b.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/40bd9c9803b793e41323a6e5b27cba4f4b11e44cc6889575a7c821a60bebbd4b.jpg)

![5be1c206283c11332377ffedb6e98cf0884521d78b2cd1bf7233424d55f24bfa.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/5be1c206283c11332377ffedb6e98cf0884521d78b2cd1bf7233424d55f24bfa.jpg)

![5f61d76f3dadd90563f6bd806aebbb2757a5b2c79473a24a65639125d4021b7a.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/5f61d76f3dadd90563f6bd806aebbb2757a5b2c79473a24a65639125d4021b7a.jpg)

![8d515b307172fea80f1a32d5f37b3ab6195ee7a9060086d03857e932d3df59e6.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/8d515b307172fea80f1a32d5f37b3ab6195ee7a9060086d03857e932d3df59e6.jpg)

![973279e7d1370a15f7791e22a9842b6db38989cd2f43a50d31359f4cbc575cef.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/973279e7d1370a15f7791e22a9842b6db38989cd2f43a50d31359f4cbc575cef.jpg)

![e585f3fbd07bf6258daa881d3853d577db110025617a14c4eb21097cb15ba22b.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/e585f3fbd07bf6258daa881d3853d577db110025617a14c4eb21097cb15ba22b.jpg)

![f6ba01fbef4a1b36042ff99b5bd6b0b9b39cc1682a2fcbf5b7a4d1b1443a0996.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/images/f6ba01fbef4a1b36042ff99b5bd6b0b9b39cc1682a2fcbf5b7a4d1b1443a0996.jpg)

### Tables

![15a055a2dd90d68b967edd318348af103f68c7e0793af19a9f3355e8c1d8555a.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/15a055a2dd90d68b967edd318348af103f68c7e0793af19a9f3355e8c1d8555a.jpg)

![1b8a53bb5b368c49a3328b1dfc55bba43e0016ef636f4cdeeb529e9a77f6301c.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/1b8a53bb5b368c49a3328b1dfc55bba43e0016ef636f4cdeeb529e9a77f6301c.jpg)

![4f790f309d0a76162c9440291c5b930bf99dd12ea6bc3cb6e28f63f07ce84a42.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/4f790f309d0a76162c9440291c5b930bf99dd12ea6bc3cb6e28f63f07ce84a42.jpg)

![596c5e9b9862b6826b8f88d9581195e66bb94ea2d2299a568ca4a856e373b8ce.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/596c5e9b9862b6826b8f88d9581195e66bb94ea2d2299a568ca4a856e373b8ce.jpg)

![6060540ec87f1c9532eccf67e510f98168c937cc5ac3c6d0c07d037a7598199c.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/6060540ec87f1c9532eccf67e510f98168c937cc5ac3c6d0c07d037a7598199c.jpg)

![8b0f1eeacda52e006f379bdb979292c93125e2bf8787746b466ec795f1ac3179.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/8b0f1eeacda52e006f379bdb979292c93125e2bf8787746b466ec795f1ac3179.jpg)

![c8fb480b4edf7d7bc91ea73ac063ef8ede13cd7954579ee88a61bdb97c46d5b6.jpg](../icml_results/3828_Recursive%20Binding%20on%20a%20Budget_%20Subspace%20Carving%20in%20Order-%24p%24%20Tensor%20Memories/tables/c8fb480b4edf7d7bc91ea73ac063ef8ede13cd7954579ee88a61bdb97c46d5b6.jpg)

<a id="paper-3789"></a>

## Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models


### Images

![0ed63a8cff6068d81d3db49b2f806e4eaca495637c7a2468fa9c46c8141b9e0b.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/0ed63a8cff6068d81d3db49b2f806e4eaca495637c7a2468fa9c46c8141b9e0b.jpg)

![2b269708cb49f5a866ef4e054fa4aa82cc1947303c590327e0a4beb1d6ab45a9.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/2b269708cb49f5a866ef4e054fa4aa82cc1947303c590327e0a4beb1d6ab45a9.jpg)

![2ea8e10ebce04f5a5650278a36c9fde06b41bdb97558e6c3fd6e70fcffa52449.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/2ea8e10ebce04f5a5650278a36c9fde06b41bdb97558e6c3fd6e70fcffa52449.jpg)

![331af152e442745dbde30f8c6c64a3ec310c37a703a0078f1740fe26e7af7ef6.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/331af152e442745dbde30f8c6c64a3ec310c37a703a0078f1740fe26e7af7ef6.jpg)

![68ce7ab36c6972ce69880595a93f2c1e40007f9228e4976498a6527e52a5d951.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/68ce7ab36c6972ce69880595a93f2c1e40007f9228e4976498a6527e52a5d951.jpg)

![722f215f33fa4318266cd0c4c49d85df8a3c04676c9c6844127514db02bc2b57.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/722f215f33fa4318266cd0c4c49d85df8a3c04676c9c6844127514db02bc2b57.jpg)

![ccc29f07d1565eb63c729925d0dc75deeb4289b950a95c40ef53bceb63958b20.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/images/ccc29f07d1565eb63c729925d0dc75deeb4289b950a95c40ef53bceb63958b20.jpg)

### Tables

![1651deb18803431bc59bfbcfae67a68715a66f3168ad01ce3ae66003b75d5b75.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/1651deb18803431bc59bfbcfae67a68715a66f3168ad01ce3ae66003b75d5b75.jpg)

![36d179e5800d3d66d63b94fee62f04135831bbb5f69e505c10d204dba4b88302.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/36d179e5800d3d66d63b94fee62f04135831bbb5f69e505c10d204dba4b88302.jpg)

![40c1daa18d3acd51b24f244b4b7f1592c99390c6018879df618f4d03b4ab872c.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/40c1daa18d3acd51b24f244b4b7f1592c99390c6018879df618f4d03b4ab872c.jpg)

![61efd5626a8ab6effddc3ff229a9aac23906701af8714b3f010dd6de8ddc52eb.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/61efd5626a8ab6effddc3ff229a9aac23906701af8714b3f010dd6de8ddc52eb.jpg)

![6f1fa16871d158cb18b5df67302bcee4cb8eb16546da934eb26a8aad701e895d.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/6f1fa16871d158cb18b5df67302bcee4cb8eb16546da934eb26a8aad701e895d.jpg)

![70a26c681e96415d566e016aa6b3acb36a3cc864a437d8f0c6368302b70bf249.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/70a26c681e96415d566e016aa6b3acb36a3cc864a437d8f0c6368302b70bf249.jpg)

![7fb76e20ab73d52fcb77d07119993cdad11dae9fb053a47de5d0f845ad1aa9cb.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/7fb76e20ab73d52fcb77d07119993cdad11dae9fb053a47de5d0f845ad1aa9cb.jpg)

![88189b8c38c593ab3560537823e297b306e615c7e942714554b0146237c7b6aa.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/88189b8c38c593ab3560537823e297b306e615c7e942714554b0146237c7b6aa.jpg)

![8f2870fd0675f1395056ba9478544920168a2b97791cd189df49ef4f041dfa8b.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/8f2870fd0675f1395056ba9478544920168a2b97791cd189df49ef4f041dfa8b.jpg)

![ae97cd1a62abf382f853dd544756196100a88513fa227b6bb22144e7e4a5ea2a.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/ae97cd1a62abf382f853dd544756196100a88513fa227b6bb22144e7e4a5ea2a.jpg)

![b794a90d89bac6abc1bb1305291369e1199642beae993410bec92acc973ca263.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/b794a90d89bac6abc1bb1305291369e1199642beae993410bec92acc973ca263.jpg)

![cd4d337e8ca3e9eab91006a759cf0278048d8bb3812f020b83ea1606d02d1c91.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/cd4d337e8ca3e9eab91006a759cf0278048d8bb3812f020b83ea1606d02d1c91.jpg)

![f9db788857143e322e5ed8700a5544eb6c496914ec3370cacfde17a49da94e47.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/f9db788857143e322e5ed8700a5544eb6c496914ec3370cacfde17a49da94e47.jpg)

![fdb7ca54bdc1194dec9817785a35c42d40323aa6fba1a5cbce05ddc7d7abb116.jpg](../icml_results/3829_Forget%20to%20Know%2C%20Remember%20to%20Use_%20Context-Aware%20Unlearning%20for%20Large%20Language%20Models/tables/fdb7ca54bdc1194dec9817785a35c42d40323aa6fba1a5cbce05ddc7d7abb116.jpg)

<a id="paper-3790"></a>

## Measuring and Mitigating Post-Hoc Rationalization in Reverse Chain-of-Thought Generation


### Images

![1ba4fd7d9bab27a57a8a9f9bd682a62b80241cd386afa1073df8ad1f780cb247.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/1ba4fd7d9bab27a57a8a9f9bd682a62b80241cd386afa1073df8ad1f780cb247.jpg)

![2572a0f1c1fb6e536fd69c754d82355d880cbe9a9ab64ae695b3a9140745682b.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/2572a0f1c1fb6e536fd69c754d82355d880cbe9a9ab64ae695b3a9140745682b.jpg)

![591e78b5999d2af8ce36640ddd5cfdbe5525cc44410e1b4a57141bbf3d4e11a6.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/591e78b5999d2af8ce36640ddd5cfdbe5525cc44410e1b4a57141bbf3d4e11a6.jpg)

![9d146098fc86b52c0df9213d41b2bb9a9872549b260ef261bdc3bf1c88d69f43.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/9d146098fc86b52c0df9213d41b2bb9a9872549b260ef261bdc3bf1c88d69f43.jpg)

![c37314c10011cb3dc255e52aa172a88b923a37f4cb4e0761a4a525e3f470078b.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/c37314c10011cb3dc255e52aa172a88b923a37f4cb4e0761a4a525e3f470078b.jpg)

![d1078e78ca8417094a3f71070cf18d43a848a4be1a78e96c27fe52a1e55d07e6.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/d1078e78ca8417094a3f71070cf18d43a848a4be1a78e96c27fe52a1e55d07e6.jpg)

![f59507e97cb5bf8086780b688181b8d0ba373398f1c01a3e58ea654da97dc588.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/images/f59507e97cb5bf8086780b688181b8d0ba373398f1c01a3e58ea654da97dc588.jpg)

### Tables

![0867b9cd6d4e1ca4afc7b67e5d2473f487c499061d878c7cc456e73352384faf.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/0867b9cd6d4e1ca4afc7b67e5d2473f487c499061d878c7cc456e73352384faf.jpg)

![0b926ffb6a47ac01c81d9891663dd5510d4659e2f55217530e3efae05fb064ba.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/0b926ffb6a47ac01c81d9891663dd5510d4659e2f55217530e3efae05fb064ba.jpg)

![20846a3da9f3490e89246bf9f65083fa1f6ae4b8807e39b2eaaf8e4efbe3588f.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/20846a3da9f3490e89246bf9f65083fa1f6ae4b8807e39b2eaaf8e4efbe3588f.jpg)

![2353af1ce023b419fa71979b8b3ac2d444afd27224cddb056fd6397612aa19cd.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/2353af1ce023b419fa71979b8b3ac2d444afd27224cddb056fd6397612aa19cd.jpg)

![358c2dca13a0b4fb5313c5b0533fd5c7774dce3fa6e4333fa0d4197ee14a613c.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/358c2dca13a0b4fb5313c5b0533fd5c7774dce3fa6e4333fa0d4197ee14a613c.jpg)

![38480210b1ecbfeeb2c1f5fad0d51de9dbab518ca0e0e58af34c7fac4da24059.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/38480210b1ecbfeeb2c1f5fad0d51de9dbab518ca0e0e58af34c7fac4da24059.jpg)

![3f24e135a0a28e8cb09a9a00e8a53422015d327a5abd946704902814e2d50b94.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/3f24e135a0a28e8cb09a9a00e8a53422015d327a5abd946704902814e2d50b94.jpg)

![657eb2a153678780049a9c9a5e3442f7bb3bcb4ee0a33af0d49a4a0a0d12e0fd.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/657eb2a153678780049a9c9a5e3442f7bb3bcb4ee0a33af0d49a4a0a0d12e0fd.jpg)

![877a7e0587a78c7962113c34dc57da97e8a7a4e3a9528a22fe208f1088a0b27e.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/877a7e0587a78c7962113c34dc57da97e8a7a4e3a9528a22fe208f1088a0b27e.jpg)

![8992429014e7dae33cb29a0119660ea44d4fc280aae6fa13e2292a65c5fcffea.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/8992429014e7dae33cb29a0119660ea44d4fc280aae6fa13e2292a65c5fcffea.jpg)

![8cc62da988368157b3481413f4590f92008f8d152c743e794261e4ba76f4cba4.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/8cc62da988368157b3481413f4590f92008f8d152c743e794261e4ba76f4cba4.jpg)

![cccc477d7a7258b4d70589fd95d66c57404d29aa17f6155a6029e9154e170b91.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/cccc477d7a7258b4d70589fd95d66c57404d29aa17f6155a6029e9154e170b91.jpg)

![d5e88332657f11937fdb4ee43186cc41e5c873ca27470c4d3d0091ff7f15c730.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/d5e88332657f11937fdb4ee43186cc41e5c873ca27470c4d3d0091ff7f15c730.jpg)

![f376d68ce2554b4a7e2cde7311f90619975baee13ea2c29557608f6d6e2e7313.jpg](../icml_results/3830_Measuring%20and%20Mitigating%20Post-Hoc%20Rationalization%20in%20Reverse%20Chain-of-Thought%20Generation/tables/f376d68ce2554b4a7e2cde7311f90619975baee13ea2c29557608f6d6e2e7313.jpg)

<a id="paper-3791"></a>

## CauScale: Neural Causal Discovery at Scale


### Images

![3ac4f4170a5066fee20d3cbc871496c568b65b9ee965324f270b2f1e1a16d847.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/3ac4f4170a5066fee20d3cbc871496c568b65b9ee965324f270b2f1e1a16d847.jpg)

![4ad7839ad028292ac92bd766f97c9c39d7e281456c4e1be7b0436b207a67bf05.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/4ad7839ad028292ac92bd766f97c9c39d7e281456c4e1be7b0436b207a67bf05.jpg)

![548ff9b1b32e5f25c0e018f6bd2a8c6a2ae420f6e73d9277df514bc434decb22.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/548ff9b1b32e5f25c0e018f6bd2a8c6a2ae420f6e73d9277df514bc434decb22.jpg)

![63b81e35ddf173e9c0b01b7622467a8ad9bd8e6f114a5bbc63316a5402acb06d.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/63b81e35ddf173e9c0b01b7622467a8ad9bd8e6f114a5bbc63316a5402acb06d.jpg)

![945b682ab44fe9fc9106d50680eb9ccd997cb6ddc376cecbab9ad7cb405dda71.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/945b682ab44fe9fc9106d50680eb9ccd997cb6ddc376cecbab9ad7cb405dda71.jpg)

![9c96b2e6638f25e5d27cc795a3649e70473808284377fe3076f4f2edf2ad6817.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/9c96b2e6638f25e5d27cc795a3649e70473808284377fe3076f4f2edf2ad6817.jpg)

![d41e287b5bacd651b871db6be6cebbc0dbb5d40ac6f326a997879bd823e9367d.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/d41e287b5bacd651b871db6be6cebbc0dbb5d40ac6f326a997879bd823e9367d.jpg)

![ff588bf667a92716060d5e3e75773b84b651a06c473774733a855dd246c500b6.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/images/ff588bf667a92716060d5e3e75773b84b651a06c473774733a855dd246c500b6.jpg)

### Tables

![14e4d1106d7ed165b298cc74662cd0684be53426679dc65f45f4d87b853f6159.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/14e4d1106d7ed165b298cc74662cd0684be53426679dc65f45f4d87b853f6159.jpg)

![308be1b49072c6cb409e6c67e260b03f65db2dd209413a31f8d56e6a48425de2.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/308be1b49072c6cb409e6c67e260b03f65db2dd209413a31f8d56e6a48425de2.jpg)

![397b46650c3293f546d19c1b9b0215fb55b51eb2e29d9eb44d0a10086bc40288.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/397b46650c3293f546d19c1b9b0215fb55b51eb2e29d9eb44d0a10086bc40288.jpg)

![3d9337678c70f56375da854e35a58795687c169f347380d055103a0f31816f27.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/3d9337678c70f56375da854e35a58795687c169f347380d055103a0f31816f27.jpg)

![5913f5d9da8ca220b5355178036857f6b784cf4a383767f62e497dcacead88f7.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/5913f5d9da8ca220b5355178036857f6b784cf4a383767f62e497dcacead88f7.jpg)

![72af106bbc009ffe691f2156317c7871e55f38d6989b0bccc229691d12e208c2.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/72af106bbc009ffe691f2156317c7871e55f38d6989b0bccc229691d12e208c2.jpg)

![76a3fece71d5ed9558c011901bf34ea2634315b128b32158e843b4aba3b0af18.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/76a3fece71d5ed9558c011901bf34ea2634315b128b32158e843b4aba3b0af18.jpg)

![832e9103a40c5b5c5ce4fadaf33ce1f37a1e29db064b37d3260a80382686a60e.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/832e9103a40c5b5c5ce4fadaf33ce1f37a1e29db064b37d3260a80382686a60e.jpg)

![898a0d6f791844f0a4f8c047b0a528fd93f90f9dcb02e6fdcca132043e8306b0.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/898a0d6f791844f0a4f8c047b0a528fd93f90f9dcb02e6fdcca132043e8306b0.jpg)

![b690b82cc27c1d973e57e67537b788506a8a16cf669bacf9754908482461d413.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/b690b82cc27c1d973e57e67537b788506a8a16cf669bacf9754908482461d413.jpg)

![f980e8ed283f591bc1b9fdd1fcaa47efc1646a73263a96f5c67f316c509d417c.jpg](../icml_results/3831_CauScale_%20Neural%20Causal%20Discovery%20at%20Scale/tables/f980e8ed283f591bc1b9fdd1fcaa47efc1646a73263a96f5c67f316c509d417c.jpg)

<a id="paper-3792"></a>

## LoPhyDA: Low-Rank Tensor and Physics Gradient Guided Diffusion for Atmospheric Data Assimilation


### Images

![0dd3609f0aece4a4527d248591d07f7723ee76b08ae4a174095e81a48abef7d9.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/0dd3609f0aece4a4527d248591d07f7723ee76b08ae4a174095e81a48abef7d9.jpg)

![1210d7ae6cabe1309b6a8d5f94e69620eef6bba68f174ed6bee897bedf315f02.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/1210d7ae6cabe1309b6a8d5f94e69620eef6bba68f174ed6bee897bedf315f02.jpg)

![2788ad225c84660cf29ee69369421f0dad4a99f687a3964e8bdf1f6f95344067.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/2788ad225c84660cf29ee69369421f0dad4a99f687a3964e8bdf1f6f95344067.jpg)

![721ce1d4e4d230f4f6602f2b88a0a33aadf90cf34b300c04a7be85aa0bff505c.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/721ce1d4e4d230f4f6602f2b88a0a33aadf90cf34b300c04a7be85aa0bff505c.jpg)

![7cb260b69f9135020304b7fdea1c2f05545f6fc3610e5456cd4f0dd6cfc85019.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/7cb260b69f9135020304b7fdea1c2f05545f6fc3610e5456cd4f0dd6cfc85019.jpg)

![811a35a2fbce75b408dec2c4c5aea0d9af8f2efcdbc00177b0e54e8bedb60c37.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/811a35a2fbce75b408dec2c4c5aea0d9af8f2efcdbc00177b0e54e8bedb60c37.jpg)

![a564462a0e9db333aa8fca21b81080333dd3dd27f6f7a2f03255e162f346404c.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/a564462a0e9db333aa8fca21b81080333dd3dd27f6f7a2f03255e162f346404c.jpg)

![b3baa9b77c525fd726b3f6aa37860edb351f00ebb73a6bd40d72e06af41e019c.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/b3baa9b77c525fd726b3f6aa37860edb351f00ebb73a6bd40d72e06af41e019c.jpg)

![c6bde33b1f0c90f1caa1f72a165c883aaf81efdbecbac35aed73611dcef2673d.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/c6bde33b1f0c90f1caa1f72a165c883aaf81efdbecbac35aed73611dcef2673d.jpg)

![ced53f7781b44f3cc8774219a78df7868f7a4a238e0738620a3930c0c1150eb4.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/ced53f7781b44f3cc8774219a78df7868f7a4a238e0738620a3930c0c1150eb4.jpg)

![dd1f9bac4439245c39e944b8c7c0a7b371f2632c07d67573ed30d9f5ae27e19f.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/dd1f9bac4439245c39e944b8c7c0a7b371f2632c07d67573ed30d9f5ae27e19f.jpg)

![f4e18856f5ca903d8c14a92d0790d6bd985c5064ccbcbe467df4e55032f38857.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/images/f4e18856f5ca903d8c14a92d0790d6bd985c5064ccbcbe467df4e55032f38857.jpg)

### Tables

![49565a8e9fdceaf4026c3b17b5c78573718785fa40b0d73bb001332d43925366.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/49565a8e9fdceaf4026c3b17b5c78573718785fa40b0d73bb001332d43925366.jpg)

![98e226a838b1285f168e8cd8d8e203a78e6db11d7010c9a554d5de1feab7584e.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/98e226a838b1285f168e8cd8d8e203a78e6db11d7010c9a554d5de1feab7584e.jpg)

![9ed24e29e9db16a289404e0fc292d7f36377f301db72bc807fd865346108f548.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/9ed24e29e9db16a289404e0fc292d7f36377f301db72bc807fd865346108f548.jpg)

![a5739648d8505f8fcac0a522e4ef575dc9344766db7275b396923cafa60393b5.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/a5739648d8505f8fcac0a522e4ef575dc9344766db7275b396923cafa60393b5.jpg)

![b9c70d398f8816831a7045c94fbc0d357a6c5a078a157f42349324d87d0fd137.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/b9c70d398f8816831a7045c94fbc0d357a6c5a078a157f42349324d87d0fd137.jpg)

![c17c327bd1bd49dc3cab846d0fdd4fd5e0fe4fd79c9605d7bdd58f2aaf8b07c8.jpg](../icml_results/3832_LoPhyDA_%20Low-Rank%20Tensor%20and%20Physics%20Gradient%20Guided%20Diffusion%20for%20Atmospheric%20Data%20Assimilation/tables/c17c327bd1bd49dc3cab846d0fdd4fd5e0fe4fd79c9605d7bdd58f2aaf8b07c8.jpg)

<a id="paper-3793"></a>

## Sheaf Neural Networks on SPD Manifolds: Second-Order Geometric Representation Learning


### Images

![4cb926d800f678242ebbcbafd00b044e9f14982d32d854722e72ec86b50226ae.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/images/4cb926d800f678242ebbcbafd00b044e9f14982d32d854722e72ec86b50226ae.jpg)

![5c8081335dd070090340c718619ebc6bb7e0adbd5d92000949b26e567a479156.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/images/5c8081335dd070090340c718619ebc6bb7e0adbd5d92000949b26e567a479156.jpg)

![b5366aeb019d9676eb0756f1db372a63e698270a82d5d78a0aa4d33e9734f599.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/images/b5366aeb019d9676eb0756f1db372a63e698270a82d5d78a0aa4d33e9734f599.jpg)

### Tables

![2a04d522c6c63256efafa5e97c2c2cdc1124d8dec57f61fe96515308aa15d171.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/2a04d522c6c63256efafa5e97c2c2cdc1124d8dec57f61fe96515308aa15d171.jpg)

![39881f840ea38fb2682dc1dc2d1123059c650c9f2b30f62d3bd7847ca4819362.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/39881f840ea38fb2682dc1dc2d1123059c650c9f2b30f62d3bd7847ca4819362.jpg)

![3c29c8e07865e65e51512b4542f7f9c9ced7a084cf6d8ff23a969eccbcc2f2db.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/3c29c8e07865e65e51512b4542f7f9c9ced7a084cf6d8ff23a969eccbcc2f2db.jpg)

![3fa66c550ed3ac251f71f1633c3877ef32cebc285bdecc02f472a2ad883e06fe.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/3fa66c550ed3ac251f71f1633c3877ef32cebc285bdecc02f472a2ad883e06fe.jpg)

![6344caf2bf4314ce6eec861391cb75782e9bec7eb0ef9eb3d9799e1eab75278e.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/6344caf2bf4314ce6eec861391cb75782e9bec7eb0ef9eb3d9799e1eab75278e.jpg)

![7190382dd00b03733da833eeb5500929048a76562c019261cfbe04b493a7faf6.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/7190382dd00b03733da833eeb5500929048a76562c019261cfbe04b493a7faf6.jpg)

![98c1564098dc9aba1539691226cc1c9eb2b8e8126332dcd1b068a1f2f355c176.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/98c1564098dc9aba1539691226cc1c9eb2b8e8126332dcd1b068a1f2f355c176.jpg)

![9d3be44ddd008fd2c174f46109148c68bed80493a2e9ae1ee25ce16766069964.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/9d3be44ddd008fd2c174f46109148c68bed80493a2e9ae1ee25ce16766069964.jpg)

![a14c3836c4fe43d515b00af999623eb90388e23b2881204af7dd343cc9d2e748.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/a14c3836c4fe43d515b00af999623eb90388e23b2881204af7dd343cc9d2e748.jpg)

![c049a5489f10154fc2f0fe8f6feb4a0d1ceb067e8df9e40809b06ebee4dec637.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/c049a5489f10154fc2f0fe8f6feb4a0d1ceb067e8df9e40809b06ebee4dec637.jpg)

![db27cc60b3b1072e1832db8bc97b5a0d8e649bde983427dbf05f7f1a2cd0d4b5.jpg](../icml_results/3833_Sheaf%20Neural%20Networks%20on%20SPD%20Manifolds_%20Second-Order%20Geometric%20Representation%20Learning/tables/db27cc60b3b1072e1832db8bc97b5a0d8e649bde983427dbf05f7f1a2cd0d4b5.jpg)

<a id="paper-3794"></a>

## TACTIC: Task-Aware Sparse Coordination Graphs for Multi-Task Multi-agent Reinforcement Learning


### Images

![00b01ec3e22f8733d625a6a54016819511d85913418f4ac368745456e7356d7d.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/00b01ec3e22f8733d625a6a54016819511d85913418f4ac368745456e7356d7d.jpg)

![2f67d4ba36a6eca27cf522af00eb0caa103b7933414a0017291aac25a040e786.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/2f67d4ba36a6eca27cf522af00eb0caa103b7933414a0017291aac25a040e786.jpg)

![69ef2639cf4df2fe83f33fe2a2a8444a3244002e2d9d15e5607db6ee933e5002.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/69ef2639cf4df2fe83f33fe2a2a8444a3244002e2d9d15e5607db6ee933e5002.jpg)

![7ce90cc8982f9d1ef50441b457c3805c18e8c580e6cfc64217ca7539ce6c2b7e.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/7ce90cc8982f9d1ef50441b457c3805c18e8c580e6cfc64217ca7539ce6c2b7e.jpg)

![ab742e05943099c2d2ccaf143b918414eaf57df6f4673593baf2d8ee8fc0711b.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/ab742e05943099c2d2ccaf143b918414eaf57df6f4673593baf2d8ee8fc0711b.jpg)

![e93c07764aa846d9e07d783f6f215a1b7ed0b19af7c49232729bba631bccbe4e.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/e93c07764aa846d9e07d783f6f215a1b7ed0b19af7c49232729bba631bccbe4e.jpg)

![f6f2df65736e4cc97ca57339a98649f85b14044ea1caf937b956c29f5778b4ca.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/images/f6f2df65736e4cc97ca57339a98649f85b14044ea1caf937b956c29f5778b4ca.jpg)

### Tables

![12492f83e8d00d3ed62df46dae47a1e96f4bfb41e2e5d34803429c47705c90ae.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/12492f83e8d00d3ed62df46dae47a1e96f4bfb41e2e5d34803429c47705c90ae.jpg)

![21a4fe77dcbd1cd2bd5fbffcc8b7ba4b79f8d6cc9a8591c8a5fc8397a3569f3d.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/21a4fe77dcbd1cd2bd5fbffcc8b7ba4b79f8d6cc9a8591c8a5fc8397a3569f3d.jpg)

![21fdb1d862003f5fe2be2fa0f428cdb3c3db8fdb9ddc888f2bd233ec5d5c21eb.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/21fdb1d862003f5fe2be2fa0f428cdb3c3db8fdb9ddc888f2bd233ec5d5c21eb.jpg)

![225833879cfe811c87317e0627ba3fc661214e98b5d94ab67f82bd8a85bbf501.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/225833879cfe811c87317e0627ba3fc661214e98b5d94ab67f82bd8a85bbf501.jpg)

![2e6042bdab290b35bab2cdd4d5499bdee047145f6f40059055718b33c0141aec.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/2e6042bdab290b35bab2cdd4d5499bdee047145f6f40059055718b33c0141aec.jpg)

![38956d8554556644bff16a4bfac3059a905c5d259afd5001a4943c1e0eb3c2ff.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/38956d8554556644bff16a4bfac3059a905c5d259afd5001a4943c1e0eb3c2ff.jpg)

![791393740f5f92a4eb12d6de78dcb4c73651db15ab4fe9b8ab83e11ebaed5eb7.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/791393740f5f92a4eb12d6de78dcb4c73651db15ab4fe9b8ab83e11ebaed5eb7.jpg)

![7ab09ac74f34f62bd2cc6b1e59cf8c8e815fd7c796c2b70d125e1ff721a7cc06.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/7ab09ac74f34f62bd2cc6b1e59cf8c8e815fd7c796c2b70d125e1ff721a7cc06.jpg)

![83181cc061202c136baeed2aff3a1bdc341dfa9601f0339cd0553b16705836ee.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/83181cc061202c136baeed2aff3a1bdc341dfa9601f0339cd0553b16705836ee.jpg)

![8ae235d37807bc9de55898743604b4f88957c5f0ed2170919b9d1ae97a6bcbb6.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/8ae235d37807bc9de55898743604b4f88957c5f0ed2170919b9d1ae97a6bcbb6.jpg)

![b2c73b27c0827792f868ee1673218aba15682bed5681fecf23b88ab208a3e8c3.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/b2c73b27c0827792f868ee1673218aba15682bed5681fecf23b88ab208a3e8c3.jpg)

![b938cbde463ea1d5a889b6bdbc418ab9c8451dad57759a89579fba41917c2d16.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/b938cbde463ea1d5a889b6bdbc418ab9c8451dad57759a89579fba41917c2d16.jpg)

![d1e529269db392355a7d679815c602731a8887cbf399bf9d080d7363ed95291c.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/d1e529269db392355a7d679815c602731a8887cbf399bf9d080d7363ed95291c.jpg)

![d2068891b1640e53fef50e1ad3114cd948704c44416e1194f8546d92dab70974.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/d2068891b1640e53fef50e1ad3114cd948704c44416e1194f8546d92dab70974.jpg)

![d505495d91e2eb40c1248c74d00c34089c849216ef4025a331db28ef11fd8396.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/d505495d91e2eb40c1248c74d00c34089c849216ef4025a331db28ef11fd8396.jpg)

![e0dc7b2e9d59de102ed6c211241df80e6d8ec3d50f993cfaa8ba200de29cdbcd.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/e0dc7b2e9d59de102ed6c211241df80e6d8ec3d50f993cfaa8ba200de29cdbcd.jpg)

![e778a81c66fa182aa3b2a063a62bbc3a27c47faf8b9e87b4c73d75f1870f5b2e.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/e778a81c66fa182aa3b2a063a62bbc3a27c47faf8b9e87b4c73d75f1870f5b2e.jpg)

![f7c9ee1e5a0f9f6d822862cbc9dc0cb9bfbff8ffb648b7286d9ca1137f21c44e.jpg](../icml_results/3834_TACTIC_%20Task-Aware%20Sparse%20Coordination%20Graphs%20for%20Multi-Task%20Multi-agent%20Reinforcement%20Learning/tables/f7c9ee1e5a0f9f6d822862cbc9dc0cb9bfbff8ffb648b7286d9ca1137f21c44e.jpg)

<a id="paper-3795"></a>

## MIST: Moment-Aligned Invariant Stability Transform for Robust Flow Matching

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
