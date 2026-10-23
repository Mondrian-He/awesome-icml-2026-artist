# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 37 of 197

## 目录 (Table of Contents)

1. [The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space](#paper-1189)
2. [Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents](#paper-1190)
3. [Global Geometry Is Not Enough for Vision Representations](#paper-1191)
4. [Offline Reinforcement Learning with Universal Horizon Models](#paper-1192)
5. [Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](#paper-1193)
6. [Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](#paper-1194)
7. [Attentive Multi-Layer Fusion for Vision Transformers](#paper-1195)
8. [Conformal Prediction for Early Stopping in Mixed Integer Optimization](#paper-1196)
9. [Learning-Augmented Online Minimization with Dual Predictions](#paper-1197)
10. [Adaptively Robust Resettable Streaming](#paper-1198)
11. [Improved Distribution Estimation in $\ell_∞$](#paper-1199)
12. [MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models](#paper-1200)
13. [Barriers to Counterfactual Credit Attribution for Autoregressive Models](#paper-1201)
14. [Active Learning with Low-Rank Structure for Data Selection](#paper-1202)
15. [Unifying Stacking and Cascading for Efficient Ensemble Inference](#paper-1203)
16. [Prototype-Grounded Concept Models for Verifiable Concept Alignment](#paper-1204)
17. [Multicalibration Yields Better Matchings](#paper-1205)
18. [Constrained hybrid modelling to predict microbial dynamics and organic matter turnover in soil systems](#paper-1206)
19. [Emergent Alignment via Competition](#paper-1207)
20. [RT-Lynx: Putting GEMM Sparsity in the Right Place for Diffusion Models](#paper-1208)
21. [Geometric Conformal Prediction with Spatial Ranks and Multivariate Quantiles](#paper-1209)
22. [Predicting evolutionary rate as a pretraining task improves genome language model representations](#paper-1210)
23. [Fast Mixing Steady-State Control in Markov Decision Processes](#paper-1211)
24. [Beyond Binary: Continuous State Optimization with Graph-Structured Objectives](#paper-1212)
25. [Optimized Deferral for Imbalanced Settings](#paper-1213)
26. [A Theoretical Framework for Modular Learning of Robust Generative Models](#paper-1214)
27. [BLIPs: Bayesian Learned Interatomic Potentials](#paper-1215)
28. [PLATE: Plasticity-Tunable Efficient Adapters for Geometry-Aware Continual Learning](#paper-1216)
29. [Abductive Reasoning with Probabilistic Commonsense](#paper-1217)
30. [Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](#paper-1218)
31. [Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy](#paper-1219)
32. [ANTiC: Adaptive Neural Temporal In Situ Compressor](#paper-1220)
33. [An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants](#paper-1221)

---


<a id="paper-1189"></a>

## The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space

### Images

![066569c3cc5c832f93250b8d5f3bd24767d78787ab60737b2ba3bf05888a2a27.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/066569c3cc5c832f93250b8d5f3bd24767d78787ab60737b2ba3bf05888a2a27.jpg)

![0913ff496cfd4f05ddfb272c0bd18db8fbfa5d75c2de01c16da5996eab0173a4.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/0913ff496cfd4f05ddfb272c0bd18db8fbfa5d75c2de01c16da5996eab0173a4.jpg)

![1093d96669b1be09841b6f2dc3046b169b418af4d4db3c723a60a96eefce70b1.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/1093d96669b1be09841b6f2dc3046b169b418af4d4db3c723a60a96eefce70b1.jpg)

![17c4d3234bfd11ed91de9c1058643e525f3211f527135ec4751ada8e8d4d6b73.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/17c4d3234bfd11ed91de9c1058643e525f3211f527135ec4751ada8e8d4d6b73.jpg)

![21289cf512829b338c43e6069119ce7a2afd1c99ba5e07277b0326cb09a9cb20.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/21289cf512829b338c43e6069119ce7a2afd1c99ba5e07277b0326cb09a9cb20.jpg)

![22f4d7657f2b06700d9208aa55dc071647197c09352505c66258096135c281fe.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/22f4d7657f2b06700d9208aa55dc071647197c09352505c66258096135c281fe.jpg)

![23ad4f03c17a629a2bcc8d20e043b2337d4742aa9014a83bddef1c4e845636a3.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/23ad4f03c17a629a2bcc8d20e043b2337d4742aa9014a83bddef1c4e845636a3.jpg)

![283fd0ce479639b4bbd09310f225c30e5fd88111f0934400785567d1ccc2bf00.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/283fd0ce479639b4bbd09310f225c30e5fd88111f0934400785567d1ccc2bf00.jpg)

![3e4258e5bbcefb46a2985c975cff1b018601cc6cd54b83f40a6f306b830cad46.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/3e4258e5bbcefb46a2985c975cff1b018601cc6cd54b83f40a6f306b830cad46.jpg)

![405bcad77a00da09d5b9e43d820687a5a116aa4bb00c7c9d782f53d939d57b98.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/405bcad77a00da09d5b9e43d820687a5a116aa4bb00c7c9d782f53d939d57b98.jpg)

![4620770e1e8f6bd92d84f6d04da524520c430c6b248d633503697f91f614555b.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/4620770e1e8f6bd92d84f6d04da524520c430c6b248d633503697f91f614555b.jpg)

![4729b73749f91f9963a362f07e2801d00a75214b98fdf04da0f9d2c0e9e96e35.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/4729b73749f91f9963a362f07e2801d00a75214b98fdf04da0f9d2c0e9e96e35.jpg)

![49a5c405ced854fbb4823296cceb1fed3d437416473a6d247dffcfde5f785f52.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/49a5c405ced854fbb4823296cceb1fed3d437416473a6d247dffcfde5f785f52.jpg)

![515ac22d46b0bd4d3a033d6bc1784251ae06a6b8535227cfc7cb4f8778ec000e.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/515ac22d46b0bd4d3a033d6bc1784251ae06a6b8535227cfc7cb4f8778ec000e.jpg)

![54f0ec183ca5d479b2aa6c6c0d2581558ef13cd911324442d6bd7e81318226ae.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/54f0ec183ca5d479b2aa6c6c0d2581558ef13cd911324442d6bd7e81318226ae.jpg)

![59bcefbfbd02abd3f321de985b3728f4506d3e48ed1c10f1ce8f3a721053d108.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/59bcefbfbd02abd3f321de985b3728f4506d3e48ed1c10f1ce8f3a721053d108.jpg)

![5ea518faea7b2f627a830908707338ee5c8f78fab8ea7717a179186715ee11f9.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/5ea518faea7b2f627a830908707338ee5c8f78fab8ea7717a179186715ee11f9.jpg)

![647afc27667f19cb51a37c57bb6820b59bdd60587f358a8e9caf41bdbc255588.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/647afc27667f19cb51a37c57bb6820b59bdd60587f358a8e9caf41bdbc255588.jpg)

![656ee39ef1a0330cc0b98611286ca7bcc5b60fa0ffa34dc9342cbb4340f273c4.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/656ee39ef1a0330cc0b98611286ca7bcc5b60fa0ffa34dc9342cbb4340f273c4.jpg)

![708248ee591536ba4647a5d7dc924664e127afe05f971a99d900a04a99381a3c.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/708248ee591536ba4647a5d7dc924664e127afe05f971a99d900a04a99381a3c.jpg)

![7d79e253f735d9acc4da1759930f0694edb35188ddc867487a8120a45d3f1a4a.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/7d79e253f735d9acc4da1759930f0694edb35188ddc867487a8120a45d3f1a4a.jpg)

![9533fb8193560fed2a5249d13fe5d6ba8a189275b99471fae0f601564b74f4e5.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/9533fb8193560fed2a5249d13fe5d6ba8a189275b99471fae0f601564b74f4e5.jpg)

![966f91d4642b5b913fcb7eafe55abc39cd2e07994918b4df5db1ed458eff872d.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/966f91d4642b5b913fcb7eafe55abc39cd2e07994918b4df5db1ed458eff872d.jpg)

![991d46953566101a0beadf3c7893087fb2924b927ad790c8c5f2028ebedeedd7.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/991d46953566101a0beadf3c7893087fb2924b927ad790c8c5f2028ebedeedd7.jpg)

![a2fa3bc7160d084e7404050cde1e12e116514036b9eb2977fec4687743e77d69.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/a2fa3bc7160d084e7404050cde1e12e116514036b9eb2977fec4687743e77d69.jpg)

![ab97439d74021bf3a0f0ed3030bf3d2ee70b1bd7a0be09e5dd7def4f9fff0500.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/ab97439d74021bf3a0f0ed3030bf3d2ee70b1bd7a0be09e5dd7def4f9fff0500.jpg)

![abf250b3120f7f964af148ff45d9b9bf2cc8dcfe426d1f8e94af5ea29609a244.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/abf250b3120f7f964af148ff45d9b9bf2cc8dcfe426d1f8e94af5ea29609a244.jpg)

![acff658c08019536908817952f22642bb9fd4698d953565241587ba6f526aae6.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/acff658c08019536908817952f22642bb9fd4698d953565241587ba6f526aae6.jpg)

![ad2d6b39037b993c1b6d617102086ca43d2c16dc4f9f901b5a1ed78c3075de96.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/ad2d6b39037b993c1b6d617102086ca43d2c16dc4f9f901b5a1ed78c3075de96.jpg)

![af98b02597522443807958f40a5384539ea8cdd715a50f62ec57cb35c213b12a.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/af98b02597522443807958f40a5384539ea8cdd715a50f62ec57cb35c213b12a.jpg)

![b2ac0eddf0f1e34ad2a30d6af13bbd15ec72cfd75b8ce38a635682b477b1e0e7.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/b2ac0eddf0f1e34ad2a30d6af13bbd15ec72cfd75b8ce38a635682b477b1e0e7.jpg)

![b30826930fe64552d03c280c46b98c0ff27d6e90d71c32a6f3b83d0aaa91cec0.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/b30826930fe64552d03c280c46b98c0ff27d6e90d71c32a6f3b83d0aaa91cec0.jpg)

![b7759c485e2d26bd68c11577886359b3a241c07fbce309eccbd1b1b0cb84d142.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/b7759c485e2d26bd68c11577886359b3a241c07fbce309eccbd1b1b0cb84d142.jpg)

![c5aaa83a9ad11eb71333c57f3c22eb6b1c5ea6899f0d388601479285354b3c54.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/c5aaa83a9ad11eb71333c57f3c22eb6b1c5ea6899f0d388601479285354b3c54.jpg)

![c6050c30ba38a20f620b485e07f772d8f55017a0d584bd9500bad2651bdcca98.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/c6050c30ba38a20f620b485e07f772d8f55017a0d584bd9500bad2651bdcca98.jpg)

![cc5dd271277963e43c15ffcacadff8f776023dd90658fb71b3d3fea6aeaf319e.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/cc5dd271277963e43c15ffcacadff8f776023dd90658fb71b3d3fea6aeaf319e.jpg)

![d4ff76ecf131fe2a71339919e589c685c8a6a08f752cb7453cdcf03532e4768c.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/d4ff76ecf131fe2a71339919e589c685c8a6a08f752cb7453cdcf03532e4768c.jpg)

![d993f6f12814d705301bf7562f75cf7f5b9c9d37e5299fb57619504d63d71ae9.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/d993f6f12814d705301bf7562f75cf7f5b9c9d37e5299fb57619504d63d71ae9.jpg)

![dabd362fb4eaa304f5e4e75d8bf9b2455449e3ceefea9caffa9f809aa688c45a.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/dabd362fb4eaa304f5e4e75d8bf9b2455449e3ceefea9caffa9f809aa688c45a.jpg)

![de3b51ddf56699a179b660536139b82e7be4950001eba643da81b3162259e367.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/de3b51ddf56699a179b660536139b82e7be4950001eba643da81b3162259e367.jpg)

![ec27bfaefdc24f3b3bd1dfd89a8c7b0dcc844323d23203a3dcc0c1170a7b69c8.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/ec27bfaefdc24f3b3bd1dfd89a8c7b0dcc844323d23203a3dcc0c1170a7b69c8.jpg)

![ed0604532c134fc9b191a55f88c7c80a9eeda9eaafa67d99c5c6981c75a25fd4.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/ed0604532c134fc9b191a55f88c7c80a9eeda9eaafa67d99c5c6981c75a25fd4.jpg)

![ed742a813fc28491733534e372612d175937e1404a3cb1dfa85419be79a68051.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/ed742a813fc28491733534e372612d175937e1404a3cb1dfa85419be79a68051.jpg)

![f8e4d0ebdf98fb23be52c633295c9794c3f16f15f7e953a2c0c702af9d909159.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/f8e4d0ebdf98fb23be52c633295c9794c3f16f15f7e953a2c0c702af9d909159.jpg)

![fdbe64355ec4af847e26dcf11bda5d4b5e9b52d4f5b30f01fb7e51ab44828a68.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/images/fdbe64355ec4af847e26dcf11bda5d4b5e9b52d4f5b30f01fb7e51ab44828a68.jpg)

### Tables

![0bfb57b31a562d1f2075a83f506377a2590beeb554fc51cd8413b13e13f7a518.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/tables/0bfb57b31a562d1f2075a83f506377a2590beeb554fc51cd8413b13e13f7a518.jpg)

![70c9949c86512fd0b061e88ea72deb981f94d47e5547e96f838919dd3747fc9f.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/tables/70c9949c86512fd0b061e88ea72deb981f94d47e5547e96f838919dd3747fc9f.jpg)

![8d9d0b780f2afc7dc9ade8a650bf8f5c9937a0faf34bd880b6d6533df393325d.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/tables/8d9d0b780f2afc7dc9ade8a650bf8f5c9937a0faf34bd880b6d6533df393325d.jpg)

![c718169ea23bc75ab9d1a37b1a5b3ec6ee2cc14d73974d11c791b0cf26b215b6.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/tables/c718169ea23bc75ab9d1a37b1a5b3ec6ee2cc14d73974d11c791b0cf26b215b6.jpg)

![f33545ced7adda152aa0fea67f8cafc453badece83feed04e62cd2015996be5b.jpg](../icml_results/1206_Balancing%20Plasticity%20and%20Stability%20with%20Fast%20and%20Slow%20Successor%20Features/tables/f33545ced7adda152aa0fea67f8cafc453badece83feed04e62cd2015996be5b.jpg)

## The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space


### Images

![0d8da7289284a26689de6ac090c469ee5cebf29e543b22440747495c2a37236e.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/0d8da7289284a26689de6ac090c469ee5cebf29e543b22440747495c2a37236e.jpg)

![6796bae4dd4f82caaec2ef0e5968892c1323a5df89ae820a2da8eac1ec01cdab.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/6796bae4dd4f82caaec2ef0e5968892c1323a5df89ae820a2da8eac1ec01cdab.jpg)

![a3cbab36c2b20a908a7fa4bed6d308c978f9e154b45e3887474ea12ba29d4272.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/a3cbab36c2b20a908a7fa4bed6d308c978f9e154b45e3887474ea12ba29d4272.jpg)

![bf01addf2ee00c8241a3fc52854236256c1c9c8938267c5d8de3429b4936aeb9.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/bf01addf2ee00c8241a3fc52854236256c1c9c8938267c5d8de3429b4936aeb9.jpg)

![c05e5fae4d7a829be179a3035dda7a27d34e528a89099c5dd6cc5c4cb9a627c0.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/c05e5fae4d7a829be179a3035dda7a27d34e528a89099c5dd6cc5c4cb9a627c0.jpg)

![c10a1e664202681ec4d63e43d3754365e96cba6bd2b10a13c9e931f393ee0189.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/c10a1e664202681ec4d63e43d3754365e96cba6bd2b10a13c9e931f393ee0189.jpg)

![c3d6f249105681077d7102cd9cf015463ed98840d176ce0064daf66b9896465d.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/c3d6f249105681077d7102cd9cf015463ed98840d176ce0064daf66b9896465d.jpg)

![feb018094d8d50e38ab1298e1a6261414f7e46479c861d9aef58c91b0c3c31da.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/images/feb018094d8d50e38ab1298e1a6261414f7e46479c861d9aef58c91b0c3c31da.jpg)

### Tables

![004857a7607f5e574d659879528669ee00b63f6de6e818df9bf56e234bab3cea.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/tables/004857a7607f5e574d659879528669ee00b63f6de6e818df9bf56e234bab3cea.jpg)

![52d3848d64ab2b4c817ce0f9e0bcaad4d98001864155d32ce4500c3bc53f18b9.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/tables/52d3848d64ab2b4c817ce0f9e0bcaad4d98001864155d32ce4500c3bc53f18b9.jpg)

![b33e60f80e8f14188ea33794da5284fe503001e1f952ecf090a4b194e1f9f56c.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/tables/b33e60f80e8f14188ea33794da5284fe503001e1f952ecf090a4b194e1f9f56c.jpg)

![c0b2abbee711584550c31d3ac211bf64a18028a8849e34b38e50327a66654e53.jpg](../icml_results/1207_The%20Lie%20We%20Tell_%20Correcting%20the%20Euclidean%20Fallacy%20in%20Vision%20Language%20Action%20Policies%20via%20Score%20Match/tables/c0b2abbee711584550c31d3ac211bf64a18028a8849e34b38e50327a66654e53.jpg)

<a id="paper-1190"></a>

## Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents


### Images

![26bb46850e376cfdcc2eea3bdff27422130a5d6b4ead4054d2ce4ceaefafd54f.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/26bb46850e376cfdcc2eea3bdff27422130a5d6b4ead4054d2ce4ceaefafd54f.jpg)

![27faab53ae49dcd1be3e8d156bf9359105d20a23f08007e652b3a5ad778afdf3.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/27faab53ae49dcd1be3e8d156bf9359105d20a23f08007e652b3a5ad778afdf3.jpg)

![38393900c5ec79e309a7db132374fe58ee6b4a4d998b5d9a09017e0ded05777b.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/38393900c5ec79e309a7db132374fe58ee6b4a4d998b5d9a09017e0ded05777b.jpg)

![5b1e25667aa1e678404b6a09188359284d086bc51bab70ca245341c25bceaeb4.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/5b1e25667aa1e678404b6a09188359284d086bc51bab70ca245341c25bceaeb4.jpg)

![5cc046110d615c709e28d08dfeedb9f635910deed6220a04d76020ace23ed106.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/5cc046110d615c709e28d08dfeedb9f635910deed6220a04d76020ace23ed106.jpg)

![75e840b1193ffaef2468e4bdfcf6c0b9eebc9281e80c2a987d78a3488522b79f.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/75e840b1193ffaef2468e4bdfcf6c0b9eebc9281e80c2a987d78a3488522b79f.jpg)

![9a43e83f7c932b3e7b56e563c18edbdf45240a6989431c954214a14c7a17a618.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/9a43e83f7c932b3e7b56e563c18edbdf45240a6989431c954214a14c7a17a618.jpg)

![c65bfee60289a592b9c39c51ddf2c8c0f3195d33ca20cb8cbde754fc18b70df5.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/c65bfee60289a592b9c39c51ddf2c8c0f3195d33ca20cb8cbde754fc18b70df5.jpg)

![c938e274876d4459a7dce8c0cef77bcdc51f1fff1ffd39aa75888c58cfb62403.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/c938e274876d4459a7dce8c0cef77bcdc51f1fff1ffd39aa75888c58cfb62403.jpg)

![d9e8f75245382fbfa90cc7a075c3905432bb7cf1afcb635cfbc13a3bca9d2e74.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/d9e8f75245382fbfa90cc7a075c3905432bb7cf1afcb635cfbc13a3bca9d2e74.jpg)

![dbb935f09766e40f03dcf9f9e9632c818d7bac9784905de7634dad982a5be000.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/dbb935f09766e40f03dcf9f9e9632c818d7bac9784905de7634dad982a5be000.jpg)

![ecdc1104cca4baeec00f9ebbc0b1a25263c5d4af111e7fce1f330d1cf978c6e9.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/ecdc1104cca4baeec00f9ebbc0b1a25263c5d4af111e7fce1f330d1cf978c6e9.jpg)

![fcbe4f4e7151b6f0cdeb6683b229493334d90de2d48d117fee82c2a395a219a1.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/images/fcbe4f4e7151b6f0cdeb6683b229493334d90de2d48d117fee82c2a395a219a1.jpg)

### Tables

![027f4fd98d59295a93b6c7f60bb552b72cf47416a7d75c1cfffbe143be74d147.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/027f4fd98d59295a93b6c7f60bb552b72cf47416a7d75c1cfffbe143be74d147.jpg)

![26478c5577d465a34fe4c9c39e6d734b7421033abb240234c0163b41c85db3cd.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/26478c5577d465a34fe4c9c39e6d734b7421033abb240234c0163b41c85db3cd.jpg)

![31a81e69d38174dff18e019963e7e04e69c3b09cacebeb1551bfdce84441a268.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/31a81e69d38174dff18e019963e7e04e69c3b09cacebeb1551bfdce84441a268.jpg)

![37a7426f7a838a9b2ad007e32a5b0dc927cca00c542d08d6398b22fa2b3d94bf.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/37a7426f7a838a9b2ad007e32a5b0dc927cca00c542d08d6398b22fa2b3d94bf.jpg)

![39f569be5524563db854aa03751d7bf5ed8aa21c9fed6bad06dac069a94d00f4.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/39f569be5524563db854aa03751d7bf5ed8aa21c9fed6bad06dac069a94d00f4.jpg)

![3de5c39743ac7ae59342ddfbcd335fe368b43ad795f7798b9eed35f4ddcb1cb6.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/3de5c39743ac7ae59342ddfbcd335fe368b43ad795f7798b9eed35f4ddcb1cb6.jpg)

![4133eb6e8617cd499687d059ffd2aeb43459dc4866626b9e8fa1b3f87e3e8935.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/4133eb6e8617cd499687d059ffd2aeb43459dc4866626b9e8fa1b3f87e3e8935.jpg)

![418af41327f5d083da9234041d858ea9cee6255d01be6cfc93e5a523bab085cb.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/418af41327f5d083da9234041d858ea9cee6255d01be6cfc93e5a523bab085cb.jpg)

![44f309abeb5119f384f2f600e610add9429f3f00a6bf28fd9943cfb5cab536c4.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/44f309abeb5119f384f2f600e610add9429f3f00a6bf28fd9943cfb5cab536c4.jpg)

![4b629c5ca2bda8f3a555fffb3f2e8060c348c41f6b447fcdb5e15883adca051e.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/4b629c5ca2bda8f3a555fffb3f2e8060c348c41f6b447fcdb5e15883adca051e.jpg)

![5c623c687b5f520e50fed4c69f91a5f166130a2d85f39187de737858e6434c54.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/5c623c687b5f520e50fed4c69f91a5f166130a2d85f39187de737858e6434c54.jpg)

![6508d5e5cd86ba33221f62f89ce3a68f3e832a109a23e5922e60529c5d0df453.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/6508d5e5cd86ba33221f62f89ce3a68f3e832a109a23e5922e60529c5d0df453.jpg)

![679b02e75deaba4fedbed6b17b1b334bf0b860b98c814b56f6a8e0a03b1a2191.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/679b02e75deaba4fedbed6b17b1b334bf0b860b98c814b56f6a8e0a03b1a2191.jpg)

![6b34c19789642a24bfba229208e23e9be8fd6206df82306bcc0b25ca8d4d33c9.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/6b34c19789642a24bfba229208e23e9be8fd6206df82306bcc0b25ca8d4d33c9.jpg)

![6ddbaf154d34405d1741628f946e197fd22136c69ff92fc47c9f5ff85b99f587.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/6ddbaf154d34405d1741628f946e197fd22136c69ff92fc47c9f5ff85b99f587.jpg)

![6f872614c8607d4556ef25219da7b97074d8cfc835c7f0ce58e83876734d13b7.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/6f872614c8607d4556ef25219da7b97074d8cfc835c7f0ce58e83876734d13b7.jpg)

![6fe71dc8e34d86c4c1ae42425f43f238a98148f1e1b02c4dde64371048f80c24.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/6fe71dc8e34d86c4c1ae42425f43f238a98148f1e1b02c4dde64371048f80c24.jpg)

![7e100c9655e09a16b7650e4ab4324ad215f11e0a2f230104d40a624f72ced8aa.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/7e100c9655e09a16b7650e4ab4324ad215f11e0a2f230104d40a624f72ced8aa.jpg)

![84e1a2f59c275154eae9bfd3942fae42a3eb95d2bc00e562ddbe32fbe2185328.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/84e1a2f59c275154eae9bfd3942fae42a3eb95d2bc00e562ddbe32fbe2185328.jpg)

![85c139bee904f91a424b298b1bff9aa6e2900479bd2cfd0d08bb0bffb8332bb6.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/85c139bee904f91a424b298b1bff9aa6e2900479bd2cfd0d08bb0bffb8332bb6.jpg)

![8c16495d0bf8df1a08e1fa8ce4205d1e4c298f9f23423b973ef85aefb1ea2fad.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/8c16495d0bf8df1a08e1fa8ce4205d1e4c298f9f23423b973ef85aefb1ea2fad.jpg)

![91428d418a1e747e5d1ea17873a1a6cc0826737086a77caaa64eb6c8f33489c1.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/91428d418a1e747e5d1ea17873a1a6cc0826737086a77caaa64eb6c8f33489c1.jpg)

![981fd543e3a92f33f8de3671d3a4adc15ce7c5ee9dac4e3afcc6bfe0304728cb.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/981fd543e3a92f33f8de3671d3a4adc15ce7c5ee9dac4e3afcc6bfe0304728cb.jpg)

![a541ffec09262b6685b219bf1ccad8b711a4cdd8168a32dc276cd623e8d5b09d.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/a541ffec09262b6685b219bf1ccad8b711a4cdd8168a32dc276cd623e8d5b09d.jpg)

![b6127c0eefaf340b3b0f402dffe17fd2f5f0b2bff98a388203a787b9baafa316.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/b6127c0eefaf340b3b0f402dffe17fd2f5f0b2bff98a388203a787b9baafa316.jpg)

![b95d19d8074b66f5f7b9d2cc0ee2889d3eff49a1550a2ec035a748595573eb18.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/b95d19d8074b66f5f7b9d2cc0ee2889d3eff49a1550a2ec035a748595573eb18.jpg)

![bb05085b6eb9ec0a6ebf72ec656bc9b057a40daceb74d6819de1e5f9cae559c7.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/bb05085b6eb9ec0a6ebf72ec656bc9b057a40daceb74d6819de1e5f9cae559c7.jpg)

![c41428c969ff05254859ed978c5550e390d5ce4e401301371612428475e0b9d5.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/c41428c969ff05254859ed978c5550e390d5ce4e401301371612428475e0b9d5.jpg)

![c54620daab71122c38a41475124246abc6fac5850a14ea6a4cea6f17369ffb4d.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/c54620daab71122c38a41475124246abc6fac5850a14ea6a4cea6f17369ffb4d.jpg)

![d027e1523f6018aa39d7a5fdeb92771e66fb054ebf4fa32b3b159c40bdea0ca7.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/d027e1523f6018aa39d7a5fdeb92771e66fb054ebf4fa32b3b159c40bdea0ca7.jpg)

![d2fd4be36b7bbc0647b8cda9118aa2aab543d862fe124300102d59e418e3614f.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/d2fd4be36b7bbc0647b8cda9118aa2aab543d862fe124300102d59e418e3614f.jpg)

![d865dd3f530d357fa363ff74cde881c051e6a4637da59049af0963cee960f415.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/d865dd3f530d357fa363ff74cde881c051e6a4637da59049af0963cee960f415.jpg)

![dbdddb4fd1d22c282ab8a0a56cffc8dd6f7be0584006a1f0614d81092f82ba2e.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/dbdddb4fd1d22c282ab8a0a56cffc8dd6f7be0584006a1f0614d81092f82ba2e.jpg)

![deefd65c29babbe850386cbf239b97a225fca6151566923f95625b8dcc9b3e72.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/deefd65c29babbe850386cbf239b97a225fca6151566923f95625b8dcc9b3e72.jpg)

![e0abc435aa62b16fa6f1f09359902044705b39e90b63a64fd0be6e3aec1fd804.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/e0abc435aa62b16fa6f1f09359902044705b39e90b63a64fd0be6e3aec1fd804.jpg)

![eb4dca2334db445fdfd0f7bc8076426b65618f1fe71eb2ad0300f7646548789f.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/eb4dca2334db445fdfd0f7bc8076426b65618f1fe71eb2ad0300f7646548789f.jpg)

![f23ae3d36af28e003133549e2a761a174e7cf1d2bbdce9a35b9be82edb780dd3.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/f23ae3d36af28e003133549e2a761a174e7cf1d2bbdce9a35b9be82edb780dd3.jpg)

![f2556398d407dc6e1e12f635e9bcfaee4e4043a43c547150b36c62b6077c5784.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/f2556398d407dc6e1e12f635e9bcfaee4e4043a43c547150b36c62b6077c5784.jpg)

![f3e2eb891a91517539c6eba1b127487fe0db4fb326f1f7b309a24baa07aa627c.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/f3e2eb891a91517539c6eba1b127487fe0db4fb326f1f7b309a24baa07aa627c.jpg)

![f7a421df6582fbce6888ffbed2b29c48a4bbfa1e198b762a649f4b0a2ff0a7df.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/f7a421df6582fbce6888ffbed2b29c48a4bbfa1e198b762a649f4b0a2ff0a7df.jpg)

![fc11cc2d9910ddf74d20a0448fce53583207fe339d2703472880470e8d733894.jpg](../icml_results/1208_Functional%20Cache%20Grafting_%20Robust%20and%20Rapid%20Code-Policy%20Synthesis%20for%20Embodied%20Agents/tables/fc11cc2d9910ddf74d20a0448fce53583207fe339d2703472880470e8d733894.jpg)

<a id="paper-1191"></a>

## Global Geometry Is Not Enough for Vision Representations


### Images

![45fa7cb4dead4af87ece681a2a8341bad9d1a3364c03786fb663fca8959f8602.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/45fa7cb4dead4af87ece681a2a8341bad9d1a3364c03786fb663fca8959f8602.jpg)

![46455b6114916ad7cf5769c5350b13f31baed4fcb1d78931162eb15d1c5678c9.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/46455b6114916ad7cf5769c5350b13f31baed4fcb1d78931162eb15d1c5678c9.jpg)

![98e93987ed5ff2564aa175d63ec6f98b503a6650ce92bd09c4d4f0645c80f7ee.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/98e93987ed5ff2564aa175d63ec6f98b503a6650ce92bd09c4d4f0645c80f7ee.jpg)

![9f7420899a3121fe122e6406233c94dea83045533132e2f1d08f132faa754c5a.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/9f7420899a3121fe122e6406233c94dea83045533132e2f1d08f132faa754c5a.jpg)

![db07d8b57a5cd78af1dd9f6153536b28ff1b649006d002c017527020936544f2.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/db07d8b57a5cd78af1dd9f6153536b28ff1b649006d002c017527020936544f2.jpg)

![e44f1e69e7f1f0daecec9689803846848a724ab88ab525b91a9ce9bda59b25cb.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/images/e44f1e69e7f1f0daecec9689803846848a724ab88ab525b91a9ce9bda59b25cb.jpg)

### Tables

![0977c0a85b7f183b8ed7f26f5eeff7d5599d288f2a224b14c56ae8ad458ce31d.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/0977c0a85b7f183b8ed7f26f5eeff7d5599d288f2a224b14c56ae8ad458ce31d.jpg)

![0cee7e7be2b01bbf0e08930311f8ad4574252d75a8fd4a35950d2b2c4ec1c34b.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/0cee7e7be2b01bbf0e08930311f8ad4574252d75a8fd4a35950d2b2c4ec1c34b.jpg)

![0f2f2f609a9233cab88ab4ac0dcd6585fc690c70da532ca099f49dfce44adba4.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/0f2f2f609a9233cab88ab4ac0dcd6585fc690c70da532ca099f49dfce44adba4.jpg)

![13b42569d42059280c5aceff6a0eddfaaf0309be8244c716b3ed303a88024062.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/13b42569d42059280c5aceff6a0eddfaaf0309be8244c716b3ed303a88024062.jpg)

![30d9e2057b32106e43cb72e1576d423a5be9af01a9cb1fdd18a284b6ee3f5d8c.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/30d9e2057b32106e43cb72e1576d423a5be9af01a9cb1fdd18a284b6ee3f5d8c.jpg)

![3513ac0545171f4edb51f6b139396f5700e4dc7b513919e37a8c5abc7b971c98.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/3513ac0545171f4edb51f6b139396f5700e4dc7b513919e37a8c5abc7b971c98.jpg)

![3f99b3fea6b7379bf27b4f299a236636b062a701f6f40b70c5be54a6e7167892.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/3f99b3fea6b7379bf27b4f299a236636b062a701f6f40b70c5be54a6e7167892.jpg)

![542aa8bc9cd3685b58ce4d9711ad55ae7c12a63262373e43b08cdd3c3196a443.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/542aa8bc9cd3685b58ce4d9711ad55ae7c12a63262373e43b08cdd3c3196a443.jpg)

![56f0f352daccbb15c26188f9125ce1e3ffa55ca8387fa5c79bc74a2251426684.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/56f0f352daccbb15c26188f9125ce1e3ffa55ca8387fa5c79bc74a2251426684.jpg)

![5d757349f9215427361dd7eaa902a429237cfc0b101f4d68b8f2223900215295.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/5d757349f9215427361dd7eaa902a429237cfc0b101f4d68b8f2223900215295.jpg)

![6a9e6f0e82ccfb85faad3fd1139a6ff6d84d576a3a697d624f12075bef6ffbe2.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/6a9e6f0e82ccfb85faad3fd1139a6ff6d84d576a3a697d624f12075bef6ffbe2.jpg)

![825d2971322fa1f05742390b07d88aaea2390234ff229b6bc4815f51eb46ed36.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/825d2971322fa1f05742390b07d88aaea2390234ff229b6bc4815f51eb46ed36.jpg)

![a7fed7c2cd2cd2a6d96e18490e3236b14b3c380135c79219eaa08226f4e66385.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/a7fed7c2cd2cd2a6d96e18490e3236b14b3c380135c79219eaa08226f4e66385.jpg)

![b3391a402480d84a6a52df5be0a1bb11b242dd44dcdcf78b90f67daddff7ea36.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/b3391a402480d84a6a52df5be0a1bb11b242dd44dcdcf78b90f67daddff7ea36.jpg)

![b97b0d18c579004a3b439421cb0c753c61f97a2d900587c4544b1989a3b9b6a1.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/b97b0d18c579004a3b439421cb0c753c61f97a2d900587c4544b1989a3b9b6a1.jpg)

![c3460b1c04f9d67ffc09e419bc2e72518c0ea2ab986f6458e3edd6f85f7bd7bf.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/c3460b1c04f9d67ffc09e419bc2e72518c0ea2ab986f6458e3edd6f85f7bd7bf.jpg)

![d657196a46d1285f943411596269642e3ace419f87738f53cc9237d0e204b04c.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/d657196a46d1285f943411596269642e3ace419f87738f53cc9237d0e204b04c.jpg)

![f78a151df9ddd6d4e6dc870bf2ce546fc6a2a4d3c3e480dc80aff64ded0da366.jpg](../icml_results/1209_Global%20Geometry%20Is%20Not%20Enough%20for%20Vision%20Representations/tables/f78a151df9ddd6d4e6dc870bf2ce546fc6a2a4d3c3e480dc80aff64ded0da366.jpg)

<a id="paper-1192"></a>

## Offline Reinforcement Learning with Universal Horizon Models


### Images

![2cde6c319caf88ac3b344a0f1ba8fc6fbd76cbda14d6ad316288ca7cb8edd3a1.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/2cde6c319caf88ac3b344a0f1ba8fc6fbd76cbda14d6ad316288ca7cb8edd3a1.jpg)

![6f0061c8c41c89d66bbf034f64601b34b0b16620ccc3ee83ccd2b5e60c2c06c3.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/6f0061c8c41c89d66bbf034f64601b34b0b16620ccc3ee83ccd2b5e60c2c06c3.jpg)

![cab3a73cf49fb4a71ecec44b3031f7623b32bc2e1c41f09977e55da889bc819e.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/cab3a73cf49fb4a71ecec44b3031f7623b32bc2e1c41f09977e55da889bc819e.jpg)

![d3253c144cc7b707ab6e0666b2d1524201d2a9a59e3b23cd8e1cf9415e82f3df.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/d3253c144cc7b707ab6e0666b2d1524201d2a9a59e3b23cd8e1cf9415e82f3df.jpg)

![e51655f315c0c65b3da2a9d5910d2236ab5266f92730a6aab8b8f56151ece0d1.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/e51655f315c0c65b3da2a9d5910d2236ab5266f92730a6aab8b8f56151ece0d1.jpg)

![ef8fb81e13345901c22fda164f21412e056bc217f56b27495e2e7be395cd9b26.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/images/ef8fb81e13345901c22fda164f21412e056bc217f56b27495e2e7be395cd9b26.jpg)

### Tables

![481696536c4592c5a3da3b463cebf67de6e86b538a15b89c6b8b8ef3c16d4bc6.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/481696536c4592c5a3da3b463cebf67de6e86b538a15b89c6b8b8ef3c16d4bc6.jpg)

![634a96ac25bf05e0425eebc03161bee4e486bd984d2e59928d53fb1bee21ce6d.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/634a96ac25bf05e0425eebc03161bee4e486bd984d2e59928d53fb1bee21ce6d.jpg)

![7a6145c6f8e1cab64bd93f66d427877e061335ca64006cb15cec96d5146d1336.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/7a6145c6f8e1cab64bd93f66d427877e061335ca64006cb15cec96d5146d1336.jpg)

![9a45613f2d74303c3cc3935fff99cba4277aa86033df16e42474f405a99defe0.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/9a45613f2d74303c3cc3935fff99cba4277aa86033df16e42474f405a99defe0.jpg)

![a19210224bb0078990d1463d3c04278e4fa3feb2f9b663033a5e11323db7659d.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/a19210224bb0078990d1463d3c04278e4fa3feb2f9b663033a5e11323db7659d.jpg)

![b0f00b660b9cca3ce519cfa28fd641861a7785187a014d04bba7aecda5125cc2.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/b0f00b660b9cca3ce519cfa28fd641861a7785187a014d04bba7aecda5125cc2.jpg)

![e02fafdf9bd607731b55f8979d8991bcc0c1e7d9bf52ff68cbdd6ac237a0891b.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/e02fafdf9bd607731b55f8979d8991bcc0c1e7d9bf52ff68cbdd6ac237a0891b.jpg)

![e606a82ffff341eefa41b17546e58ee24d711d083d40e694dcce90655f53be5f.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/e606a82ffff341eefa41b17546e58ee24d711d083d40e694dcce90655f53be5f.jpg)

![ea60cab4791f95fca79f766d8aef56b4f5e5b4d01bbce8c13a673da85c0473be.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/ea60cab4791f95fca79f766d8aef56b4f5e5b4d01bbce8c13a673da85c0473be.jpg)

![f7c07ebf550ea8c121bf3f099e0cef034b20ba2177b3ce94af775aaa1ccc20c3.jpg](../icml_results/1210_Offline%20Reinforcement%20Learning%20with%20Universal%20Horizon%20Models/tables/f7c07ebf550ea8c121bf3f099e0cef034b20ba2177b3ce94af775aaa1ccc20c3.jpg)

<a id="paper-1193"></a>

## Many-Shot CoT-ICL: Making In-Context Learning Truly Learn


### Images

![1646f46cfd8fcc884551f6d71586acd142df128a7e6148199884127e1ea1da42.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/1646f46cfd8fcc884551f6d71586acd142df128a7e6148199884127e1ea1da42.jpg)

![16ec3e620a2731d0cf66ecbb998b58e6aad66e9c944a45201d1568ce8f478ea3.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/16ec3e620a2731d0cf66ecbb998b58e6aad66e9c944a45201d1568ce8f478ea3.jpg)

![431d6aa7992d9a37fca91a0978f294b313952ffd2b15db39065f2eb86a77ecfa.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/431d6aa7992d9a37fca91a0978f294b313952ffd2b15db39065f2eb86a77ecfa.jpg)

![65e6337041b5772928058a0753f7337bb75acc0f5a8f777c77a8af20b26879f4.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/65e6337041b5772928058a0753f7337bb75acc0f5a8f777c77a8af20b26879f4.jpg)

![71f722bcef23e9b80987491171ea985afd51f07524a1c8739fcc5a9750250817.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/71f722bcef23e9b80987491171ea985afd51f07524a1c8739fcc5a9750250817.jpg)

![7862f1955b03c20e1885cbacc206a27f48f3022c6becd5eb1b570970b67752db.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/7862f1955b03c20e1885cbacc206a27f48f3022c6becd5eb1b570970b67752db.jpg)

![979fba87008f9827526e2a40c3a67b61713a36e06082e023371e80d331d85f71.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/979fba87008f9827526e2a40c3a67b61713a36e06082e023371e80d331d85f71.jpg)

![a634b0758fa806111bb0e88541b2d7329007ab6afa230b5f8160a96ee2725e40.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/a634b0758fa806111bb0e88541b2d7329007ab6afa230b5f8160a96ee2725e40.jpg)

![bd107f90b29df2a61af90f959fd74d4006fa50aa8dacb55316bb95d44a51596f.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/bd107f90b29df2a61af90f959fd74d4006fa50aa8dacb55316bb95d44a51596f.jpg)

![d3e9c1906c719ebe2d0fb01227b67bb577adf37a44fed1a2f8cfdc0749f8c638.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/d3e9c1906c719ebe2d0fb01227b67bb577adf37a44fed1a2f8cfdc0749f8c638.jpg)

![e9266dc6d48e7713a4824cf26c06962b77dbd2d6f270e616644b475e1e9020a6.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/images/e9266dc6d48e7713a4824cf26c06962b77dbd2d6f270e616644b475e1e9020a6.jpg)

### Tables

![13295d80a3b6cb9979ed9d874744bd1b1cff265f7d8609a0918f607c76735353.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/13295d80a3b6cb9979ed9d874744bd1b1cff265f7d8609a0918f607c76735353.jpg)

![37ed2dfb2ba3f1f6c32c3d44cfa76ff8b1c4411df5bce174d003d96c2bd55d29.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/37ed2dfb2ba3f1f6c32c3d44cfa76ff8b1c4411df5bce174d003d96c2bd55d29.jpg)

![53679202dffcd915c144a8e6a58c61a8c80b0053e782b05bb24534200bb4420e.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/53679202dffcd915c144a8e6a58c61a8c80b0053e782b05bb24534200bb4420e.jpg)

![75374dcad309f3f35f8f5a7be903fe5086361bd5b6f550b5efb6b5d2713d0d02.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/75374dcad309f3f35f8f5a7be903fe5086361bd5b6f550b5efb6b5d2713d0d02.jpg)

![8ec6ec024fcd6a49d28e493ecae2f2936db75ccda1bcecd89c9a42e1fe7df79c.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/8ec6ec024fcd6a49d28e493ecae2f2936db75ccda1bcecd89c9a42e1fe7df79c.jpg)

![b154c0f4070b71f19738894cc49978b82753b4b3e8f6f242321cdd72f96d2fd8.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/b154c0f4070b71f19738894cc49978b82753b4b3e8f6f242321cdd72f96d2fd8.jpg)

![b7e166bdb296ce62a7499fa1b7db8b438a64dd49ce2350483f2c681180883f06.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/b7e166bdb296ce62a7499fa1b7db8b438a64dd49ce2350483f2c681180883f06.jpg)

![cb11269c1e0ff03a6e1bb2a21a3f5ddac3f1ff3285861adee0d4efe9d035ef3d.jpg](../icml_results/1211_Many-Shot%20CoT-ICL_%20Making%20In-Context%20Learning%20Truly%20Learn/tables/cb11269c1e0ff03a6e1bb2a21a3f5ddac3f1ff3285861adee0d4efe9d035ef3d.jpg)

<a id="paper-1194"></a>

## Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation


### Images

![0a2edb0ec07c822236f7e1f9dcf195e657fde8e89217e30499b52d7022c31ab6.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/0a2edb0ec07c822236f7e1f9dcf195e657fde8e89217e30499b52d7022c31ab6.jpg)

![391b0e86d2a678b21c7434b5a0da3910e56e5bb62e2dd3baa9fef059f00c7e03.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/391b0e86d2a678b21c7434b5a0da3910e56e5bb62e2dd3baa9fef059f00c7e03.jpg)

![39ba242d0e029354b0bae28324026cd0674ccd2f300f84a1f5cc455eb716a736.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/39ba242d0e029354b0bae28324026cd0674ccd2f300f84a1f5cc455eb716a736.jpg)

![ae07b0d5095ddc9401c2d97330f016d285734ff8df2b71d19e88645b99e2c8ba.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/ae07b0d5095ddc9401c2d97330f016d285734ff8df2b71d19e88645b99e2c8ba.jpg)

![c9464385b911787d8e125a2eb40bc95b084495c07f821afd979c1849c9691f54.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/c9464385b911787d8e125a2eb40bc95b084495c07f821afd979c1849c9691f54.jpg)

![db4b7fe51ee9d12145655770818e9d8cf1336a92d3915a5d7da4447d6436cfe1.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/images/db4b7fe51ee9d12145655770818e9d8cf1336a92d3915a5d7da4447d6436cfe1.jpg)

### Tables

![451798e710341338a5825256d10323e0ad9b35eb79e86dde1c298cff3b19e6c1.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/451798e710341338a5825256d10323e0ad9b35eb79e86dde1c298cff3b19e6c1.jpg)

![47983b622e068de5ceca7cb5584e50d19fa3a391a3b043270887a726dd313544.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/47983b622e068de5ceca7cb5584e50d19fa3a391a3b043270887a726dd313544.jpg)

![4d62f9f5b84c26b3f79fc6776e716a3476d0ef9b0ac9f5bdb74116a22732d20c.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/4d62f9f5b84c26b3f79fc6776e716a3476d0ef9b0ac9f5bdb74116a22732d20c.jpg)

![677a94b5b8b92f3d766a7ac0e2effdaf4c200519ca248a56cadd110c2053879a.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/677a94b5b8b92f3d766a7ac0e2effdaf4c200519ca248a56cadd110c2053879a.jpg)

![8b0b70c2c78b7e761c75fc18cae8bba3ddfb33d290e4f36cdec9c252eee08ba4.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/8b0b70c2c78b7e761c75fc18cae8bba3ddfb33d290e4f36cdec9c252eee08ba4.jpg)

![936e8adb94d1554ae35ca1376e91af110528f31893543a0d85344ef62c5473d8.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/936e8adb94d1554ae35ca1376e91af110528f31893543a0d85344ef62c5473d8.jpg)

![95529fa8b99b11cb4808c772cf26f374da4e34e4ec5b8d2cf8c42d0f5d56c786.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/95529fa8b99b11cb4808c772cf26f374da4e34e4ec5b8d2cf8c42d0f5d56c786.jpg)

![cbf3a54e14fb6b936c14888d2341e1aa8e70e98df3d167df6f2dbd7b2186e871.jpg](../icml_results/1212_Convex%20Distance%20Operator%20Transport_%20A%20Convex%20and%20Geometry-Preserving%20Formulation/tables/cbf3a54e14fb6b936c14888d2341e1aa8e70e98df3d167df6f2dbd7b2186e871.jpg)

<a id="paper-1195"></a>

## Attentive Multi-Layer Fusion for Vision Transformers


### Images

![1a21da9a444e1296cd52d12d8dd514e8ce457f119b0278937c2087372276389c.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/1a21da9a444e1296cd52d12d8dd514e8ce457f119b0278937c2087372276389c.jpg)

![36967f43ad524f5a2d339fd0083893de80cc590bdaa64eeb74cb011eb85c4189.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/36967f43ad524f5a2d339fd0083893de80cc590bdaa64eeb74cb011eb85c4189.jpg)

![46b9a1a0c4c57f84de7985607a67ceb03efe50046be0ef2ec95c3f38ae261637.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/46b9a1a0c4c57f84de7985607a67ceb03efe50046be0ef2ec95c3f38ae261637.jpg)

![4a0f6d6887c2fc2fd2bd5974e9c414436b966ce44450f20b8cc8eb4815dc063c.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/4a0f6d6887c2fc2fd2bd5974e9c414436b966ce44450f20b8cc8eb4815dc063c.jpg)

![65fff35ef8fc6d55c24ab6ce354a19384b7fde802ee5b93aaf726bb1d06f9a62.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/65fff35ef8fc6d55c24ab6ce354a19384b7fde802ee5b93aaf726bb1d06f9a62.jpg)

![6a778ffa5c0e2c5a17ca3ee54141755529171ad7b0a4a6c948b3aadfc7eec4c0.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/6a778ffa5c0e2c5a17ca3ee54141755529171ad7b0a4a6c948b3aadfc7eec4c0.jpg)

![729647becd0ce8a991b31a25c6b2064759ee38f7d013cf9d54676f86ba411d86.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/729647becd0ce8a991b31a25c6b2064759ee38f7d013cf9d54676f86ba411d86.jpg)

![80ed3cd2a298f95bf0de50bcbca13eb8dc9a5f86cff8922754b69195febd980f.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/80ed3cd2a298f95bf0de50bcbca13eb8dc9a5f86cff8922754b69195febd980f.jpg)

![83671b9cc90e5bec355092ac12ea93bfca57194560055fb0fc86cb07aa5d83aa.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/83671b9cc90e5bec355092ac12ea93bfca57194560055fb0fc86cb07aa5d83aa.jpg)

![8ec0d4d686ca290596d2eaf8496d2f7df98951d6aedfe779181e75fc9c547e09.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/8ec0d4d686ca290596d2eaf8496d2f7df98951d6aedfe779181e75fc9c547e09.jpg)

![90c2b2755895f3de487ca8634d88a7a345275ec3446f95753780042bdfcfe1b9.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/90c2b2755895f3de487ca8634d88a7a345275ec3446f95753780042bdfcfe1b9.jpg)

![9c4853d10412ee2cae878346127ad5a0af42e11da3b2d19db94595ec05075411.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/9c4853d10412ee2cae878346127ad5a0af42e11da3b2d19db94595ec05075411.jpg)

![a1302d76c65d411ad4e3132b2d6b9b5ea03ae9ee2ecb7736dca328e9bc995d65.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/a1302d76c65d411ad4e3132b2d6b9b5ea03ae9ee2ecb7736dca328e9bc995d65.jpg)

![ae7f125e1736223d5006dbb7ca88f00ba58a4e8ce74851bb807b0b895debf7c5.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/ae7f125e1736223d5006dbb7ca88f00ba58a4e8ce74851bb807b0b895debf7c5.jpg)

![b92af5543cb0ed4e239420f6f254bce318994d6cb05ed3818f2d1d6fece0f218.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/b92af5543cb0ed4e239420f6f254bce318994d6cb05ed3818f2d1d6fece0f218.jpg)

![ba650ea350c56c6e44f77789daefc8c093149c945620da164172f14695c5e38d.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/ba650ea350c56c6e44f77789daefc8c093149c945620da164172f14695c5e38d.jpg)

![bfabb9768d326208ac9a956ba27376f267cfd3c96c9a43b70afe09f50e5f3567.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/bfabb9768d326208ac9a956ba27376f267cfd3c96c9a43b70afe09f50e5f3567.jpg)

![d379a4d2a47b6cfef9584630e9bdc116ac517212477f7fd58ee30c350aaa98d0.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/d379a4d2a47b6cfef9584630e9bdc116ac517212477f7fd58ee30c350aaa98d0.jpg)

![daa2c5e66933bfd64d3b6fe469a8f3286976d8c61371a3d947d7cacf2543f7d0.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/daa2c5e66933bfd64d3b6fe469a8f3286976d8c61371a3d947d7cacf2543f7d0.jpg)

![e2a490f6fad883df458cf0ea764e9edaa660bb56d46f1f3f8b717b0959a9ef86.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/e2a490f6fad883df458cf0ea764e9edaa660bb56d46f1f3f8b717b0959a9ef86.jpg)

![e9fa65fc090db91318f3399807891b7a304e463b36af2a308997d8bc7e451842.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/e9fa65fc090db91318f3399807891b7a304e463b36af2a308997d8bc7e451842.jpg)

![f08449ed86179ac2fa5209efab174bcf57efaeef74f96cf4d0052432e9faa4f5.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/images/f08449ed86179ac2fa5209efab174bcf57efaeef74f96cf4d0052432e9faa4f5.jpg)

### Tables

![235e64d6cfaf41a4feb58be363ef8b5a252dd8feca992ba69d3ef0c02974ca52.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/tables/235e64d6cfaf41a4feb58be363ef8b5a252dd8feca992ba69d3ef0c02974ca52.jpg)

![4fed8f0eaace095d28efa97c8b66b5d0600697802eed2395b5bb66303dca73ab.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/tables/4fed8f0eaace095d28efa97c8b66b5d0600697802eed2395b5bb66303dca73ab.jpg)

![71292d2a4e863dcaa79bcd9b37f61b20a8ef90ec0963f586f6fea987040446b4.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/tables/71292d2a4e863dcaa79bcd9b37f61b20a8ef90ec0963f586f6fea987040446b4.jpg)

![a786b28e66a0267db4e09474e159bd1e46044a0e2c9c0b94534b803c0b9c3bc9.jpg](../icml_results/1213_Attentive%20Multi-Layer%20Fusion%20for%20Vision%20Transformers/tables/a786b28e66a0267db4e09474e159bd1e46044a0e2c9c0b94534b803c0b9c3bc9.jpg)

<a id="paper-1196"></a>

## Conformal Prediction for Early Stopping in Mixed Integer Optimization


### Images

![0054e5b828bea60f150d63ec2734480b8702536ae6247b277d415c1ced4e3cab.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/images/0054e5b828bea60f150d63ec2734480b8702536ae6247b277d415c1ced4e3cab.jpg)

![057f66d2363c0edb0efb0820a8583f150a6b72e5031bf38dcaf0b0258b3490ec.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/images/057f66d2363c0edb0efb0820a8583f150a6b72e5031bf38dcaf0b0258b3490ec.jpg)

![861377ff64ccbf33a1aa7aaa5becb4377f3bb59f76596e8502a1c35e3118178a.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/images/861377ff64ccbf33a1aa7aaa5becb4377f3bb59f76596e8502a1c35e3118178a.jpg)

![aad3d8f71b8fff5c506060d60140e807ba0b3227429bee6bcdac81be7ea6dc2b.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/images/aad3d8f71b8fff5c506060d60140e807ba0b3227429bee6bcdac81be7ea6dc2b.jpg)

![b8f22047b641aa48026642b641e3a673d034ba4ffbbaf85ddf313db22bf4a032.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/images/b8f22047b641aa48026642b641e3a673d034ba4ffbbaf85ddf313db22bf4a032.jpg)

### Tables

![0356b9609175f8cae79896237774118820e9be7c416978df48adf79462cb17a5.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/0356b9609175f8cae79896237774118820e9be7c416978df48adf79462cb17a5.jpg)

![06f0c390466c64da107ef539b24b04485194f7a7d59623de7bfb95a03b2f43fa.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/06f0c390466c64da107ef539b24b04485194f7a7d59623de7bfb95a03b2f43fa.jpg)

![0aded67215969019639fef32f1593b68e8cfce8f63d02cebec987953c7635379.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/0aded67215969019639fef32f1593b68e8cfce8f63d02cebec987953c7635379.jpg)

![3fbe35c516682bc4e56bf62ac0ca07fc17bf1ec8bc270a1f4569d9bd78793f0b.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/3fbe35c516682bc4e56bf62ac0ca07fc17bf1ec8bc270a1f4569d9bd78793f0b.jpg)

![4be4078b9d73c3c0e4fb2915ee2e61d180b440247eba5665bd33df361b965dbd.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/4be4078b9d73c3c0e4fb2915ee2e61d180b440247eba5665bd33df361b965dbd.jpg)

![4db0abfa8b9d2706d15f27a3349ce4431c494a8091df8112f87b4de4b423629f.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/4db0abfa8b9d2706d15f27a3349ce4431c494a8091df8112f87b4de4b423629f.jpg)

![54abcfc8332f4171f1cf4dc47981c0c79cc503d09571b750fde48076c4bac69f.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/54abcfc8332f4171f1cf4dc47981c0c79cc503d09571b750fde48076c4bac69f.jpg)

![5e0c8326d97f1c0630d42e50621bcefee6d86b1a1da0b460069b51b52ec12d81.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/5e0c8326d97f1c0630d42e50621bcefee6d86b1a1da0b460069b51b52ec12d81.jpg)

![7629bb0fbe7bebf27059dffc5b7b777c1c127e70a8a2cb130d6063400426abc6.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/7629bb0fbe7bebf27059dffc5b7b777c1c127e70a8a2cb130d6063400426abc6.jpg)

![88833557b20936f368ad0639077666f33e2720de162b98b42de392acd19b9a5b.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/88833557b20936f368ad0639077666f33e2720de162b98b42de392acd19b9a5b.jpg)

![8b48ba06a3dcff9abc106cd46075bf5f20fcabbf805a224d96a0058d6ae0f91b.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/8b48ba06a3dcff9abc106cd46075bf5f20fcabbf805a224d96a0058d6ae0f91b.jpg)

![9584dd0d508236769114d1f8ed74ca496d391af65d454d8a8157919a10bd146d.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/9584dd0d508236769114d1f8ed74ca496d391af65d454d8a8157919a10bd146d.jpg)

![a38a01e6e111005043dc3255617febd58d2921974917118d19d45544fbff48a1.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/a38a01e6e111005043dc3255617febd58d2921974917118d19d45544fbff48a1.jpg)

![a4ffea7f8b13a669e4d35214dfe1818940e8c9ec60e9ea55e3f649b602d1840c.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/a4ffea7f8b13a669e4d35214dfe1818940e8c9ec60e9ea55e3f649b602d1840c.jpg)

![cc5b5e7143e3821df6191c4017e8719b543469f4f1e968d425a0c4b1cd8e8916.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/cc5b5e7143e3821df6191c4017e8719b543469f4f1e968d425a0c4b1cd8e8916.jpg)

![e4813f3f07219ec0afcc4965367277285c562bb16b69ff83518542f5b5f6a8f0.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/e4813f3f07219ec0afcc4965367277285c562bb16b69ff83518542f5b5f6a8f0.jpg)

![f8605fa5f7734ad4b615c9016ddfa6f76fdadd78676ade6b25901233f88f4a2c.jpg](../icml_results/1214_Conformal%20Prediction%20for%20Early%20Stopping%20in%20Mixed%20Integer%20Optimization/tables/f8605fa5f7734ad4b615c9016ddfa6f76fdadd78676ade6b25901233f88f4a2c.jpg)

<a id="paper-1197"></a>

## Learning-Augmented Online Minimization with Dual Predictions


### Images

![160f7ca9f60b25570c67d37b664991e366b6ba7886004b0ecabcf3764c5f5602.jpg](../icml_results/1215_Learning-Augmented%20Online%20Minimization%20with%20Dual%20Predictions/images/160f7ca9f60b25570c67d37b664991e366b6ba7886004b0ecabcf3764c5f5602.jpg)

![482f32f4ce212612f46e300a5c7d0fbb88863a2faf1784d848e6f155b11dcd39.jpg](../icml_results/1215_Learning-Augmented%20Online%20Minimization%20with%20Dual%20Predictions/images/482f32f4ce212612f46e300a5c7d0fbb88863a2faf1784d848e6f155b11dcd39.jpg)

![6a85a5a265aee5681fa863f57c29557fdb5ef5251b87a44707e684f7f5266cda.jpg](../icml_results/1215_Learning-Augmented%20Online%20Minimization%20with%20Dual%20Predictions/images/6a85a5a265aee5681fa863f57c29557fdb5ef5251b87a44707e684f7f5266cda.jpg)

![e33128f268112f836c6e3a6c5ad75a7b90f1192832b9189a633980f0f9a2efe8.jpg](../icml_results/1215_Learning-Augmented%20Online%20Minimization%20with%20Dual%20Predictions/images/e33128f268112f836c6e3a6c5ad75a7b90f1192832b9189a633980f0f9a2efe8.jpg)

<a id="paper-1198"></a>

## Adaptively Robust Resettable Streaming


### Tables

![299534e506008ac583af3502adefd4cb8947e6d53ef8a7dfe3460471b1027f2d.jpg](../icml_results/1216_Adaptively%20Robust%20Resettable%20Streaming/tables/299534e506008ac583af3502adefd4cb8947e6d53ef8a7dfe3460471b1027f2d.jpg)

<a id="paper-1199"></a>

## Improved Distribution Estimation in $\ell_∞$


### Images

![4607a8b8b714ef17fcdcaafd7c49ef0407d5d0ca0d93348fcfd82eb3b3f299c4.jpg](../icml_results/1218_Improved%20Distribution%20Estimation%20in%20%24_ell_%E2%88%9E%24/images/4607a8b8b714ef17fcdcaafd7c49ef0407d5d0ca0d93348fcfd82eb3b3f299c4.jpg)

![5481f355fe62a273c0d5e0d1d21b5b788a23c820f107e34a73af697325d21a8d.jpg](../icml_results/1218_Improved%20Distribution%20Estimation%20in%20%24_ell_%E2%88%9E%24/images/5481f355fe62a273c0d5e0d1d21b5b788a23c820f107e34a73af697325d21a8d.jpg)

![685129ef3606d92ab4382375b544cc6c638bec40f4b679e406d857007bc3aa33.jpg](../icml_results/1218_Improved%20Distribution%20Estimation%20in%20%24_ell_%E2%88%9E%24/images/685129ef3606d92ab4382375b544cc6c638bec40f4b679e406d857007bc3aa33.jpg)

<a id="paper-1200"></a>

## MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models


### Images

![9dd9d824dbaa61bebb3cfb483fa6c392cce6e3d8b566b2ccca95e24fcba7a6d2.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/images/9dd9d824dbaa61bebb3cfb483fa6c392cce6e3d8b566b2ccca95e24fcba7a6d2.jpg)

![b4d7459e22fe917100a03f3302ac9de985825015781447c10ac6a645b7157b20.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/images/b4d7459e22fe917100a03f3302ac9de985825015781447c10ac6a645b7157b20.jpg)

![c5cbd50e2d81cb5f66e95ed5a388e356f30b89d349a1d17128994edbe1dfbed6.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/images/c5cbd50e2d81cb5f66e95ed5a388e356f30b89d349a1d17128994edbe1dfbed6.jpg)

![e34f31ef9bf0a20d360af2cf449f1c33b76df372bcf421530b28675b633d374f.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/images/e34f31ef9bf0a20d360af2cf449f1c33b76df372bcf421530b28675b633d374f.jpg)

### Tables

![3da0e082a0808967817a51a7e163b50e58ba52c3d875ebe66736b7b53f169e54.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/3da0e082a0808967817a51a7e163b50e58ba52c3d875ebe66736b7b53f169e54.jpg)

![3dcd648cd9ae4dba71018dd8bc075c2101789b6a74a63c35bd406bf550f8673c.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/3dcd648cd9ae4dba71018dd8bc075c2101789b6a74a63c35bd406bf550f8673c.jpg)

![49499d6588c333cda09bc9ab77a5f6fe4940188e72aab4a78f815b782b018dff.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/49499d6588c333cda09bc9ab77a5f6fe4940188e72aab4a78f815b782b018dff.jpg)

![50d6a4edbef5a73c1cf2f063436990ce5a7c7d250112b26211dd91cfefe8a3c2.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/50d6a4edbef5a73c1cf2f063436990ce5a7c7d250112b26211dd91cfefe8a3c2.jpg)

![6fff8ab73edcb36c409d1849083f3f47a3216a16c1d2a8edfb2411f827d445d7.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/6fff8ab73edcb36c409d1849083f3f47a3216a16c1d2a8edfb2411f827d445d7.jpg)

![81458c293b2641faf05cd4d623b521f1ef96537ed0c6d77060e16f552cf8ee6f.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/81458c293b2641faf05cd4d623b521f1ef96537ed0c6d77060e16f552cf8ee6f.jpg)

![836df35b60084f2846766a212fc266eb5f910516f61429db7672f38bb120acd5.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/836df35b60084f2846766a212fc266eb5f910516f61429db7672f38bb120acd5.jpg)

![b949f4be8ec0f3ba6ad56883fcb66895b3be54f96a9136c98bfde9f7612ac546.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/b949f4be8ec0f3ba6ad56883fcb66895b3be54f96a9136c98bfde9f7612ac546.jpg)

![c57f6a43a5bf99f944cc3da5dd44a5f85d8421c39257338438744a438bea3d00.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/c57f6a43a5bf99f944cc3da5dd44a5f85d8421c39257338438744a438bea3d00.jpg)

![cdb141463309dea82a93d5c60e60041ce2392c0ed984f6ebeefab14ee9a1bfb6.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/cdb141463309dea82a93d5c60e60041ce2392c0ed984f6ebeefab14ee9a1bfb6.jpg)

![e10cdcc550dd2b9d64e9018ebf7cdeea5289c5431c6609b46988afe5db22b5e7.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/e10cdcc550dd2b9d64e9018ebf7cdeea5289c5431c6609b46988afe5db22b5e7.jpg)

![f8ce3a99707e024a4a91597587c05d77e8654ca58989bba95dac9a59e708dd31.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/f8ce3a99707e024a4a91597587c05d77e8654ca58989bba95dac9a59e708dd31.jpg)

![ff6a716da99a17b16a5022d89532da8b54c61bfd51918ea10b833d3b9c59cd45.jpg](../icml_results/1219_MET-Bench_%20Multimodal%20Entity%20Tracking%20for%20Evaluating%20the%20Limitations%20of%20Vision-Language%20and%20Reasonin/tables/ff6a716da99a17b16a5022d89532da8b54c61bfd51918ea10b833d3b9c59cd45.jpg)

<a id="paper-1201"></a>

## Barriers to Counterfactual Credit Attribution for Autoregressive Models


### Images

![81195023700aa539d687d133fa3f3f4af42b5ee09c254338f93fea7aae4c0683.jpg](../icml_results/1220_Barriers%20to%20Counterfactual%20Credit%20Attribution%20for%20Autoregressive%20Models/images/81195023700aa539d687d133fa3f3f4af42b5ee09c254338f93fea7aae4c0683.jpg)

### Tables

![c9c13f309bcff9e08dffe63e3cf333ba0819119fa503f442b26421e21d982bde.jpg](../icml_results/1220_Barriers%20to%20Counterfactual%20Credit%20Attribution%20for%20Autoregressive%20Models/tables/c9c13f309bcff9e08dffe63e3cf333ba0819119fa503f442b26421e21d982bde.jpg)

![f93b0530e16f9445c48d490c31c0d52ef740f5857f528dc172db4f608d5402fa.jpg](../icml_results/1220_Barriers%20to%20Counterfactual%20Credit%20Attribution%20for%20Autoregressive%20Models/tables/f93b0530e16f9445c48d490c31c0d52ef740f5857f528dc172db4f608d5402fa.jpg)

<a id="paper-1202"></a>

## Active Learning with Low-Rank Structure for Data Selection


### Images

![2c6b2bb6c4acea4e9b952aebc6c7fcf358eb48ff5edfbde065d65b5aaf7348a1.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/2c6b2bb6c4acea4e9b952aebc6c7fcf358eb48ff5edfbde065d65b5aaf7348a1.jpg)

![3d11db5faafcb6a7a9d999686621c6d3a449f37bfedc855595f8d53776a6938c.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/3d11db5faafcb6a7a9d999686621c6d3a449f37bfedc855595f8d53776a6938c.jpg)

![4c9731e673cfab80bb3bd0d1119b0771e01400d3593c517572374c883b5662de.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/4c9731e673cfab80bb3bd0d1119b0771e01400d3593c517572374c883b5662de.jpg)

![5d417137c7683baf3004e37b5a2ba59c8fb7ec6e2826b7a8682ce1e0a1d5d5e0.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/5d417137c7683baf3004e37b5a2ba59c8fb7ec6e2826b7a8682ce1e0a1d5d5e0.jpg)

![73634d77c102010592998424dcc8326c9458a2b6f2cf95b36f36052c33188ddd.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/73634d77c102010592998424dcc8326c9458a2b6f2cf95b36f36052c33188ddd.jpg)

![f806beddf2f30901f19c63b4b3a986beda08d7f1cbf06781e107eae15c022c70.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/images/f806beddf2f30901f19c63b4b3a986beda08d7f1cbf06781e107eae15c022c70.jpg)

### Tables

![78f02f2d261021a2959ee3dc63d21ec3024d4cf77fdf9e797c67bcd19ddecc95.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/tables/78f02f2d261021a2959ee3dc63d21ec3024d4cf77fdf9e797c67bcd19ddecc95.jpg)

![8fd13721cc486ef466c460563ef5dad2d0030ac59da4f86e9b1801720f889dde.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/tables/8fd13721cc486ef466c460563ef5dad2d0030ac59da4f86e9b1801720f889dde.jpg)

![a6561e7dbc14d7bf5e6a3c7faf72c6cebc262f185e93feed3a6d1f53bad8e71e.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/tables/a6561e7dbc14d7bf5e6a3c7faf72c6cebc262f185e93feed3a6d1f53bad8e71e.jpg)

![e0e4d7ccf912eba18c963970096bb9aab200fcf5923257f619a96b92a5926649.jpg](../icml_results/1221_Active%20Learning%20with%20Low-Rank%20Structure%20for%20Data%20Selection/tables/e0e4d7ccf912eba18c963970096bb9aab200fcf5923257f619a96b92a5926649.jpg)

<a id="paper-1203"></a>

## Unifying Stacking and Cascading for Efficient Ensemble Inference


### Images

![0e1f91d564e781a53519bb6d6c9dc25065ba432f2808d2759e0af81ab23a754a.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/0e1f91d564e781a53519bb6d6c9dc25065ba432f2808d2759e0af81ab23a754a.jpg)

![274e00bfc0956a095cc73b8d6193ece2a96868e267c3cefefbc974eac63933d4.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/274e00bfc0956a095cc73b8d6193ece2a96868e267c3cefefbc974eac63933d4.jpg)

![27d64a409bb192d078fdfb07fdbfbaa6b431513ed1540288140dc8f86f119ae8.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/27d64a409bb192d078fdfb07fdbfbaa6b431513ed1540288140dc8f86f119ae8.jpg)

![31fed5368a1c725c880d58f392177ba7c4d50a539b6e64f1f356a62bc90ce2f4.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/31fed5368a1c725c880d58f392177ba7c4d50a539b6e64f1f356a62bc90ce2f4.jpg)

![60b5f4e7cc6eca1128541038b2cd5146b5dfa46db24b526290e3d6f8039fa316.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/60b5f4e7cc6eca1128541038b2cd5146b5dfa46db24b526290e3d6f8039fa316.jpg)

![60dc6763eb4b21f3e56e316e6fae5fe535677fa0ef336f7b28e6eaf8778c84c9.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/60dc6763eb4b21f3e56e316e6fae5fe535677fa0ef336f7b28e6eaf8778c84c9.jpg)

![a96f2abb45da05fa52706332b9f8a45a425442b9b4233a4a572001101689270a.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/a96f2abb45da05fa52706332b9f8a45a425442b9b4233a4a572001101689270a.jpg)

![e08b828a4a878c84c2e278374efe707e537addda53a6a2e56b4a9b254965ff0e.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/images/e08b828a4a878c84c2e278374efe707e537addda53a6a2e56b4a9b254965ff0e.jpg)

### Tables

![0140c000c30cdb0fc5c8e86e28d91cbcec7b7ff65f1f119b9b906c1843869812.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/0140c000c30cdb0fc5c8e86e28d91cbcec7b7ff65f1f119b9b906c1843869812.jpg)

![118c5429a6ece7cb0e37ab0497c8e7ef5d03310d361c06f4edafefb7b421d2e6.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/118c5429a6ece7cb0e37ab0497c8e7ef5d03310d361c06f4edafefb7b421d2e6.jpg)

![162a81f7385010c9d7e731a5549a4f12425dc7e803e649f522d76260e6cf064f.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/162a81f7385010c9d7e731a5549a4f12425dc7e803e649f522d76260e6cf064f.jpg)

![17e65f297495a4d9917862c2b50f1b1d18ead8921c5c8242270664baef3f1154.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/17e65f297495a4d9917862c2b50f1b1d18ead8921c5c8242270664baef3f1154.jpg)

![1ce78fe1dac23d40c621f022ed0d335dedfa701f165ed56e7a64c9dd9bed60bf.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/1ce78fe1dac23d40c621f022ed0d335dedfa701f165ed56e7a64c9dd9bed60bf.jpg)

![222108e24057fdb3fd4643a98b9511743869566dca0edefd83ce7d727e33c8fc.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/222108e24057fdb3fd4643a98b9511743869566dca0edefd83ce7d727e33c8fc.jpg)

![296bb8a1dc0b7eacb52fceb16aa5cb3031d5d37839b5eb61b319f415bd897999.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/296bb8a1dc0b7eacb52fceb16aa5cb3031d5d37839b5eb61b319f415bd897999.jpg)

![3311a1a28df8165e8d6d7f231da3b4252cb1fdd3c505d5b8d2124e96090b28a9.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/3311a1a28df8165e8d6d7f231da3b4252cb1fdd3c505d5b8d2124e96090b28a9.jpg)

![4e44dacb223ecc7a976d87be7ee0c8395227021e36606ccba48a9239c68ec7b6.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/4e44dacb223ecc7a976d87be7ee0c8395227021e36606ccba48a9239c68ec7b6.jpg)

![592b3c9aba0dc99cd1fbfae2fc08e767e39cd85bd6211486a3f36705ee5f7a4f.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/592b3c9aba0dc99cd1fbfae2fc08e767e39cd85bd6211486a3f36705ee5f7a4f.jpg)

![6179580ef507d5ce9e6b5bc1055a37612413d40a24927aaf4210abdaece2faf6.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/6179580ef507d5ce9e6b5bc1055a37612413d40a24927aaf4210abdaece2faf6.jpg)

![6622e2e7413d9b00d39e5a5667d68d64fa77500f7e0b978f4e77a7e20810b450.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/6622e2e7413d9b00d39e5a5667d68d64fa77500f7e0b978f4e77a7e20810b450.jpg)

![6dca42fc37f981f59803bf9552bcb857ad5374af8f18a5850fd608fe7ff9bb20.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/6dca42fc37f981f59803bf9552bcb857ad5374af8f18a5850fd608fe7ff9bb20.jpg)

![7474ac600bd63b52960e30bf4f575aed495942e17078dfeb80882d334c3c65a4.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/7474ac600bd63b52960e30bf4f575aed495942e17078dfeb80882d334c3c65a4.jpg)

![7ff4c3daddae0112f158a34179d0bb0ceee6ec17595cb315ba77314e626977b1.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/7ff4c3daddae0112f158a34179d0bb0ceee6ec17595cb315ba77314e626977b1.jpg)

![8b8cad319a08b98c1104571a101638cd04964e965211a167eaeaad17a8f0b661.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/8b8cad319a08b98c1104571a101638cd04964e965211a167eaeaad17a8f0b661.jpg)

![8d49063c30cf7c0c11c1357996e4ab21ffc92490986d78ab2d2667d03714d669.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/8d49063c30cf7c0c11c1357996e4ab21ffc92490986d78ab2d2667d03714d669.jpg)

![9abedeb841f42d30fd239aaa1d0f903c70a56bcb2c1722c89791d11d2fd4231e.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/9abedeb841f42d30fd239aaa1d0f903c70a56bcb2c1722c89791d11d2fd4231e.jpg)

![aa7daa8256ecd1cd2fa4817dd154acf0a47b276ddad11c8b5ae779cf62eb8605.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/aa7daa8256ecd1cd2fa4817dd154acf0a47b276ddad11c8b5ae779cf62eb8605.jpg)

![abac1c9cab9b5b3ba3b227567dcff1734fba47bfb998c6d3d0a9f37b2d56c544.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/abac1c9cab9b5b3ba3b227567dcff1734fba47bfb998c6d3d0a9f37b2d56c544.jpg)

![ad143b2b38bc6a2bf14b526b9894e50f2bb6e3c2b5bdb88ce0718a23c87b9371.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/ad143b2b38bc6a2bf14b526b9894e50f2bb6e3c2b5bdb88ce0718a23c87b9371.jpg)

![b0dc95163db01a03179299a1079bde336e48b6ef906ee73e6028d69d704c81d7.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/b0dc95163db01a03179299a1079bde336e48b6ef906ee73e6028d69d704c81d7.jpg)

![c19656c5876a384bd251b657aa55a37262288047243f0b121d2679b976ae954a.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/c19656c5876a384bd251b657aa55a37262288047243f0b121d2679b976ae954a.jpg)

![c721964a6d3a4fb46e33fc8e45abc9bc681b8afd773c0d55159e673770f79124.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/c721964a6d3a4fb46e33fc8e45abc9bc681b8afd773c0d55159e673770f79124.jpg)

![ceee2cc184bd495acdaed41812802201ee443b0ccb7bcdcb04eaf5e71cf011f7.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/ceee2cc184bd495acdaed41812802201ee443b0ccb7bcdcb04eaf5e71cf011f7.jpg)

![dce1353da7c0f4f6b51271d525f2d364d4e97731528e4a6db9c43c1770262d8c.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/dce1353da7c0f4f6b51271d525f2d364d4e97731528e4a6db9c43c1770262d8c.jpg)

![e34a3821b4d0b113b7143512e911c8beafe5a65901103a002c8f11e8af18cfc4.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/e34a3821b4d0b113b7143512e911c8beafe5a65901103a002c8f11e8af18cfc4.jpg)

![e34adb53be9b23f719b1369044c1b1a8972ada60a565f642b969f67bd3e9dff0.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/e34adb53be9b23f719b1369044c1b1a8972ada60a565f642b969f67bd3e9dff0.jpg)

![eb5ac17e0cc92b712a693f7258f7cfa9638ce33481ac7c61f4105f4ada04d40a.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/eb5ac17e0cc92b712a693f7258f7cfa9638ce33481ac7c61f4105f4ada04d40a.jpg)

![edc172c238c5d0c4a28de56a4645a0673b8f09844dfba80bcee6cf48c484a014.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/edc172c238c5d0c4a28de56a4645a0673b8f09844dfba80bcee6cf48c484a014.jpg)

![fa175ecb0f5723c79135067345c63c5bafea1e5d6cd1a4d04a0a7278f07d3c1d.jpg](../icml_results/1222_Unifying%20Stacking%20and%20Cascading%20for%20Efficient%20Ensemble%20Inference/tables/fa175ecb0f5723c79135067345c63c5bafea1e5d6cd1a4d04a0a7278f07d3c1d.jpg)

<a id="paper-1204"></a>

## Prototype-Grounded Concept Models for Verifiable Concept Alignment


### Images

![636b8573b924e7d3f1021fd065544c3c77f70e54bc3c44454822195f6ef4a696.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/636b8573b924e7d3f1021fd065544c3c77f70e54bc3c44454822195f6ef4a696.jpg)

![6d9b3ef2b9d86b574d53fcade62e0d7bee6a1e51419f9cae66f29ea4572ab4b1.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/6d9b3ef2b9d86b574d53fcade62e0d7bee6a1e51419f9cae66f29ea4572ab4b1.jpg)

![6ff845f78f8223a5945bb630ba3d176fc8c54b6b65ef6075ffe3961816ee9db1.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/6ff845f78f8223a5945bb630ba3d176fc8c54b6b65ef6075ffe3961816ee9db1.jpg)

![8820119cf7d77512de2ff3beb047503636aecb1af00bb533c6ad8eabb754074b.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/8820119cf7d77512de2ff3beb047503636aecb1af00bb533c6ad8eabb754074b.jpg)

![93b4fc67b726ae3c4e393975267bcbc44943b4ef5c567f0f502dc813e6adf0fd.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/93b4fc67b726ae3c4e393975267bcbc44943b4ef5c567f0f502dc813e6adf0fd.jpg)

![9fea4bec7184dd8c57ccf9338669f7012a8b6212064d29aecb410c3439c8ad2a.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/9fea4bec7184dd8c57ccf9338669f7012a8b6212064d29aecb410c3439c8ad2a.jpg)

![cdcc4c3b78d80b53dab668ef8a077d120361e85b513001b9bf7496fe18bc5da3.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/cdcc4c3b78d80b53dab668ef8a077d120361e85b513001b9bf7496fe18bc5da3.jpg)

![d657d9d168ff7103b26d4b34b305855d8c8a5fc67c4e0fc805d9d7e4096d3d7f.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/d657d9d168ff7103b26d4b34b305855d8c8a5fc67c4e0fc805d9d7e4096d3d7f.jpg)

![ed8068abf21ff154198c298634fe72b7a74f61b01685e39b84a341ced1b05651.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/ed8068abf21ff154198c298634fe72b7a74f61b01685e39b84a341ced1b05651.jpg)

![f041ba90428355817b379aaa70de631375452f54e725516911db708ded6ef857.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/f041ba90428355817b379aaa70de631375452f54e725516911db708ded6ef857.jpg)

![f9e27c89fc8e571f3b27480a9eca840e6bb3b501027d43b5cb34a0923428c11e.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/images/f9e27c89fc8e571f3b27480a9eca840e6bb3b501027d43b5cb34a0923428c11e.jpg)

### Tables

![4370fa3112ffedfc207190e75d259cd3aa931769d4e58e9e270a9b86372b12d8.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/tables/4370fa3112ffedfc207190e75d259cd3aa931769d4e58e9e270a9b86372b12d8.jpg)

![cd95b11c26fba0d7ad1dab413b10ab07b058c69ada1a525e1a20acb81df98529.jpg](../icml_results/1223_Prototype-Grounded%20Concept%20Models%20for%20Verifiable%20Concept%20Alignment/tables/cd95b11c26fba0d7ad1dab413b10ab07b058c69ada1a525e1a20acb81df98529.jpg)

<a id="paper-1205"></a>

## Multicalibration Yields Better Matchings


### Images

![29b4350b47084eebd8a00ff1da7474aa72e13073a28a81fb7ab9b23f326befe9.jpg](../icml_results/1224_Multicalibration%20Yields%20Better%20Matchings/images/29b4350b47084eebd8a00ff1da7474aa72e13073a28a81fb7ab9b23f326befe9.jpg)

![3ea60151354475b208e26a3019d4e56dd9962f0a1cc7a7c853444fc2b97a3eac.jpg](../icml_results/1224_Multicalibration%20Yields%20Better%20Matchings/images/3ea60151354475b208e26a3019d4e56dd9962f0a1cc7a7c853444fc2b97a3eac.jpg)

![9fae7567fa368e7fe403b1d08b9befaa79dbfa96199619a8daba192170bdbb54.jpg](../icml_results/1224_Multicalibration%20Yields%20Better%20Matchings/images/9fae7567fa368e7fe403b1d08b9befaa79dbfa96199619a8daba192170bdbb54.jpg)

<a id="paper-1206"></a>

## Constrained hybrid modelling to predict microbial dynamics and organic matter turnover in soil systems


### Images

![1567591a66ab35a3e295e7ff47c90472ab465f7cd64eae290433d767d15194d9.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/1567591a66ab35a3e295e7ff47c90472ab465f7cd64eae290433d767d15194d9.jpg)

![176e80102eb018d541a84895338d43c1afa9ed05bc5b57e8e5907ca5e60323eb.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/176e80102eb018d541a84895338d43c1afa9ed05bc5b57e8e5907ca5e60323eb.jpg)

![29a3af97324ac1181041b84e79b728c4249dd39b0815caa6dfd8f7d485eef53c.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/29a3af97324ac1181041b84e79b728c4249dd39b0815caa6dfd8f7d485eef53c.jpg)

![30921a5895c2881eaeb444ce693d7339352257cde9d3b269e526e96b9407d9e3.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/30921a5895c2881eaeb444ce693d7339352257cde9d3b269e526e96b9407d9e3.jpg)

![95c023008f305a86b959862386788d8febe004ac79fcd6aa9ada4187c693415f.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/95c023008f305a86b959862386788d8febe004ac79fcd6aa9ada4187c693415f.jpg)

![ddd1d0c291f14e2be9dcb4da3b1febc0cede12347a02e0a95ea92b6f37e84aa3.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/ddd1d0c291f14e2be9dcb4da3b1febc0cede12347a02e0a95ea92b6f37e84aa3.jpg)

![edce3f6c293d4b51a5efa2a7465a8b45a18cf58f0b3bf7f52549613770296a35.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/edce3f6c293d4b51a5efa2a7465a8b45a18cf58f0b3bf7f52549613770296a35.jpg)

![ff3b1f94eba991721f57191e3d89d39ee2d31d43811649ed324e27de1e5326d9.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/images/ff3b1f94eba991721f57191e3d89d39ee2d31d43811649ed324e27de1e5326d9.jpg)

### Tables

![0cdc7e795cd75a3c10dcdbfa5e32bacf246fc61d255011d9a024802c1e1a01c5.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/0cdc7e795cd75a3c10dcdbfa5e32bacf246fc61d255011d9a024802c1e1a01c5.jpg)

![0d898d826312328dec7ee36f347cf94d012baff2c2124a358d3dd2d5b2b596c3.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/0d898d826312328dec7ee36f347cf94d012baff2c2124a358d3dd2d5b2b596c3.jpg)

![58cc3cd664d0e1b47f7d9f222874bb6bb38d5830484592627eb3f0449ecf3b4f.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/58cc3cd664d0e1b47f7d9f222874bb6bb38d5830484592627eb3f0449ecf3b4f.jpg)

![735f9ed3f29a867ca921fc2b84cd8a269340fc1c953bb9e8a4c38e5eebe0a5cc.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/735f9ed3f29a867ca921fc2b84cd8a269340fc1c953bb9e8a4c38e5eebe0a5cc.jpg)

![79653baa7789e452a106317eac21ab1641e22dafcb57ac4e887efa469efc9413.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/79653baa7789e452a106317eac21ab1641e22dafcb57ac4e887efa469efc9413.jpg)

![80d799c7d915a32da4ceb5ca6af5fc69f4e66eb7014243ca9f3bf328e2bf2539.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/80d799c7d915a32da4ceb5ca6af5fc69f4e66eb7014243ca9f3bf328e2bf2539.jpg)

![9234bbdcccf6d44bf53b44c595bf97c0882345dea7cac23d8c81b1a66e1e3db7.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/9234bbdcccf6d44bf53b44c595bf97c0882345dea7cac23d8c81b1a66e1e3db7.jpg)

![ad7098f33b2c99e81e654e29aea403088bf4a99d9f66ce63f0e8c96f6e587668.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/ad7098f33b2c99e81e654e29aea403088bf4a99d9f66ce63f0e8c96f6e587668.jpg)

![cd39c99aa96470bb2a461d4af73121610eed784cb84891c95ffb6355e4257f75.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/cd39c99aa96470bb2a461d4af73121610eed784cb84891c95ffb6355e4257f75.jpg)

![f8fe0ee7730b03d5d70599ede92a388bc74a507d19daaaba58727b51ecf82837.jpg](../icml_results/1225_Constrained%20hybrid%20modelling%20to%20predict%20microbial%20dynamics%20and%20organic%20matter%20turnover%20in%20soil%20syste/tables/f8fe0ee7730b03d5d70599ede92a388bc74a507d19daaaba58727b51ecf82837.jpg)

<a id="paper-1207"></a>

## Emergent Alignment via Competition


### Images

![04dcd69d5e024ad2a7755c3e0c26cdc2c911056bfd33475cea12b368d80ec686.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/04dcd69d5e024ad2a7755c3e0c26cdc2c911056bfd33475cea12b368d80ec686.jpg)

![371e83ad8214fc37730f920a37fb5ceeb25fd594fe2e2e11ddd916555512ce8b.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/371e83ad8214fc37730f920a37fb5ceeb25fd594fe2e2e11ddd916555512ce8b.jpg)

![6736423b990a80b7d150c22b1d5e352407e10a3f52092cc980015699e9ccc5f9.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/6736423b990a80b7d150c22b1d5e352407e10a3f52092cc980015699e9ccc5f9.jpg)

![697a5fc83c671839647917f538c4fe3962acf595385a0a2e32d97e361c1a80cf.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/697a5fc83c671839647917f538c4fe3962acf595385a0a2e32d97e361c1a80cf.jpg)

![6bc6334c8f62b3233a7fad9c6bce67702219f65221f1f87dfca8d0bf9cce4920.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/6bc6334c8f62b3233a7fad9c6bce67702219f65221f1f87dfca8d0bf9cce4920.jpg)

![ce0a0b9838cb430c567e1f32644c6c5ac097d0ec003856f684088717fa50850b.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/ce0a0b9838cb430c567e1f32644c6c5ac097d0ec003856f684088717fa50850b.jpg)

![db86187834743eee4ca9aecc7bb266b279cc95f30aa1ca9d1c0b4a8cd27476f9.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/db86187834743eee4ca9aecc7bb266b279cc95f30aa1ca9d1c0b4a8cd27476f9.jpg)

![fdff885dffcf7180f31002a81b8d6761602e960fc2bb9e270ce787fd8faded0c.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/images/fdff885dffcf7180f31002a81b8d6761602e960fc2bb9e270ce787fd8faded0c.jpg)

### Tables

![1a1138493201031cd9200d2ce0363149b84bc7be69b552737b855e8e9463e5c5.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/1a1138493201031cd9200d2ce0363149b84bc7be69b552737b855e8e9463e5c5.jpg)

![34c6b5a7b0fbee32ef6eb14a086f88dd5990687bbc688be0a2dcb059b5243ca5.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/34c6b5a7b0fbee32ef6eb14a086f88dd5990687bbc688be0a2dcb059b5243ca5.jpg)

![591213eda5240bd9c747c63edccc4431af631da9ecb347acfe6a35781cc56bdf.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/591213eda5240bd9c747c63edccc4431af631da9ecb347acfe6a35781cc56bdf.jpg)

![64ae92c681abbd6f82570c9ad21f74955da71720d4a23c391a023de86a777885.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/64ae92c681abbd6f82570c9ad21f74955da71720d4a23c391a023de86a777885.jpg)

![b57eb8f962f6457186c9801854ee97922da193d534a661c251b7787c09f61dcb.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/b57eb8f962f6457186c9801854ee97922da193d534a661c251b7787c09f61dcb.jpg)

![bca65e9a79add49f31c7dd726172f4e118fd6014fef0114c8d86f534e19dfd91.jpg](../icml_results/1226_Emergent%20Alignment%20via%20Competition/tables/bca65e9a79add49f31c7dd726172f4e118fd6014fef0114c8d86f534e19dfd91.jpg)

<a id="paper-1208"></a>

## RT-Lynx: Putting GEMM Sparsity in the Right Place for Diffusion Models


### Images

![0839aefd988d65a9f6c167ddda57d64a90babfac01a8489c8197d1d4878d9a15.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/0839aefd988d65a9f6c167ddda57d64a90babfac01a8489c8197d1d4878d9a15.jpg)

![0ade3483d2022207133388e72eeaf58d4e8738935f6a059db3c0f6bc5e0b2c7b.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/0ade3483d2022207133388e72eeaf58d4e8738935f6a059db3c0f6bc5e0b2c7b.jpg)

![11d61b22ed74d4f37e2a5038e1b3b16e4c3bcd728bd24feaa2393671bf60ba27.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/11d61b22ed74d4f37e2a5038e1b3b16e4c3bcd728bd24feaa2393671bf60ba27.jpg)

![14e3d5b693fd423a97582f86b541ff8ad045d8a09c4a3fec6cd79110573a8218.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/14e3d5b693fd423a97582f86b541ff8ad045d8a09c4a3fec6cd79110573a8218.jpg)

![20ebea37ae2b02132c5b39497f679dd7b5a7aaa1800a6ae0c13fcbe3217cc2f6.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/20ebea37ae2b02132c5b39497f679dd7b5a7aaa1800a6ae0c13fcbe3217cc2f6.jpg)

![2c5a942561292cc8dd4d5f317596a4fe8b6149cb57dd18297e17018e71fd3c7f.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/2c5a942561292cc8dd4d5f317596a4fe8b6149cb57dd18297e17018e71fd3c7f.jpg)

![31fb14cf31c43dca87067f3d2d8ba3b676a3ceaf2963d9b844183fa938be694c.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/31fb14cf31c43dca87067f3d2d8ba3b676a3ceaf2963d9b844183fa938be694c.jpg)

![4665e3584a11777f6537c4a5aed0a178b30b58e5ea9dd127983d9a0d6f59d8e5.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/4665e3584a11777f6537c4a5aed0a178b30b58e5ea9dd127983d9a0d6f59d8e5.jpg)

![57dfffd62ef65189a8c1450d27dc0b21d532b7ba2f37b0390f1f747271d798fd.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/57dfffd62ef65189a8c1450d27dc0b21d532b7ba2f37b0390f1f747271d798fd.jpg)

![6eaed3eeaa0cf53ea5692f024abcd2cb0d7d3228baf7ba7cfe7edfd401b041c8.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/6eaed3eeaa0cf53ea5692f024abcd2cb0d7d3228baf7ba7cfe7edfd401b041c8.jpg)

![750a494229f128ee8436c01f70ce59c969e97706a6b37a0a9b12d1ebada97f33.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/750a494229f128ee8436c01f70ce59c969e97706a6b37a0a9b12d1ebada97f33.jpg)

![a39cf11a88cdd96183a4e4d9018183cbc798535911a884a493ce9a80a798241a.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/a39cf11a88cdd96183a4e4d9018183cbc798535911a884a493ce9a80a798241a.jpg)

![a891c667baaf18c98dbefabaf6bd33f0a98d95aecc05c405f4740672dd0b8520.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/a891c667baaf18c98dbefabaf6bd33f0a98d95aecc05c405f4740672dd0b8520.jpg)

![abb3f97cc22c387a3d6476a48fcb6cdf6c8fa9e1757974288c3ae92ec30b7f52.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/abb3f97cc22c387a3d6476a48fcb6cdf6c8fa9e1757974288c3ae92ec30b7f52.jpg)

![b949b75fdc3fc524ecdb04ebbf58d7ee6fb85a478c41f10e621a59d1b0eb0d4e.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/b949b75fdc3fc524ecdb04ebbf58d7ee6fb85a478c41f10e621a59d1b0eb0d4e.jpg)

![b9fb3e978818c4594be9ace016d5898dbed2fa2fef3ba1caa8bfb6418b695a18.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/b9fb3e978818c4594be9ace016d5898dbed2fa2fef3ba1caa8bfb6418b695a18.jpg)

![c1c24e38f279bfa81c80148a24944410478617610ad265058b40b35312fc9b3e.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/c1c24e38f279bfa81c80148a24944410478617610ad265058b40b35312fc9b3e.jpg)

![ce18108ad803dc327dcbed6126c752f79f9190049cef48ca34130002ebea9306.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/ce18108ad803dc327dcbed6126c752f79f9190049cef48ca34130002ebea9306.jpg)

![d87da9ef90932a8e1c29808731da70f88660adfc8508a500a60001d7a4f89be1.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/d87da9ef90932a8e1c29808731da70f88660adfc8508a500a60001d7a4f89be1.jpg)

![d8b0a4db652d8fd342f193322e1f4deceff5ab0df9e44561ee408ca27908c863.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/d8b0a4db652d8fd342f193322e1f4deceff5ab0df9e44561ee408ca27908c863.jpg)

![de42aa9016cf7a67920c000109da31b477c7bedd8004b6de13c834bd5a011f57.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/de42aa9016cf7a67920c000109da31b477c7bedd8004b6de13c834bd5a011f57.jpg)

![ee5ee60109f92df2c9fc4b12ed5e1c2b6bc2db88ff725254e34a8040fc8897dd.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/ee5ee60109f92df2c9fc4b12ed5e1c2b6bc2db88ff725254e34a8040fc8897dd.jpg)

![fabbbd16fdf0d6e86c54886140cf18ef0334e8b426f05e63e0e29c6d5e6e5c46.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/fabbbd16fdf0d6e86c54886140cf18ef0334e8b426f05e63e0e29c6d5e6e5c46.jpg)

![faca26d107e8b4ec5852f5c718de8d807dc9e289c39b295a5cb607cfca94918f.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/faca26d107e8b4ec5852f5c718de8d807dc9e289c39b295a5cb607cfca94918f.jpg)

![ff1d73852f77c30491a8d2c7647b88d3a77e6aef31bf8bb2f17c20fca00fac31.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/images/ff1d73852f77c30491a8d2c7647b88d3a77e6aef31bf8bb2f17c20fca00fac31.jpg)

### Tables

![10677fbfa55d305006e362f8f1a01855bf9b067d7d0afe623e13cf872ed0da33.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/10677fbfa55d305006e362f8f1a01855bf9b067d7d0afe623e13cf872ed0da33.jpg)

![204278e75bf53d07869fe6e672987cbf819820b1d301e30ce81aa5415be2d78b.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/204278e75bf53d07869fe6e672987cbf819820b1d301e30ce81aa5415be2d78b.jpg)

![233b60876208f55be792a2268a740b19e43376ecb4905396512aa46f1969ecd7.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/233b60876208f55be792a2268a740b19e43376ecb4905396512aa46f1969ecd7.jpg)

![2bee1436aaf1765037ae737069eb39e696fd289a092fc767b87199ae9a07e1dd.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/2bee1436aaf1765037ae737069eb39e696fd289a092fc767b87199ae9a07e1dd.jpg)

![336b8bdf51549fbe859c92bad4a94f498c75a7984919383d12082dce73d0ae28.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/336b8bdf51549fbe859c92bad4a94f498c75a7984919383d12082dce73d0ae28.jpg)

![466a8a4d87abd4ad7a471489dc988ce76d398f30a711d45f92d2c5e858f3c116.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/466a8a4d87abd4ad7a471489dc988ce76d398f30a711d45f92d2c5e858f3c116.jpg)

![5c5223d00b206efe9046b3682ea696d15cbf9e8aaaeb1a44842e63d6e46eb6a8.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/5c5223d00b206efe9046b3682ea696d15cbf9e8aaaeb1a44842e63d6e46eb6a8.jpg)

![92e804d987131b418e89cfed0949808fc54830c6b8a5bde33c0a700ad83a8a4c.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/92e804d987131b418e89cfed0949808fc54830c6b8a5bde33c0a700ad83a8a4c.jpg)

![94c0f6cd0bff40df6614c7606b53e66dfbd996e09b6183ff206037e862a95708.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/94c0f6cd0bff40df6614c7606b53e66dfbd996e09b6183ff206037e862a95708.jpg)

![a0702e7bce48d71d0cac37059620da8ff419ea8dc825f2f34bae835f2860a786.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/a0702e7bce48d71d0cac37059620da8ff419ea8dc825f2f34bae835f2860a786.jpg)

![c1ec235bda1be3092faca53f08afbaa0290bbf698d7cc3f78d0277fcd07b8049.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/c1ec235bda1be3092faca53f08afbaa0290bbf698d7cc3f78d0277fcd07b8049.jpg)

![de741e7ee9c3b306b03d05a318d23c11320d179a8e2cf39c3f5316a02e0916ad.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/de741e7ee9c3b306b03d05a318d23c11320d179a8e2cf39c3f5316a02e0916ad.jpg)

![e91a28361e6fb9691c86c854acd43c8a71283914414c4df4f30748cb78a69e8a.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/e91a28361e6fb9691c86c854acd43c8a71283914414c4df4f30748cb78a69e8a.jpg)

![ea8e3b665ba88dd77df6f2328650e88c3cb3c6ee597c0801ad0e47875ecb3571.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/ea8e3b665ba88dd77df6f2328650e88c3cb3c6ee597c0801ad0e47875ecb3571.jpg)

![fc1ccc16dcca91b5f895ec111c076691293dfe6749527dbfbbfa1ebd9700a702.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/fc1ccc16dcca91b5f895ec111c076691293dfe6749527dbfbbfa1ebd9700a702.jpg)

![fc82aa254c246268ed6ae05755d3bcd6280ebea820afde3b8cc2dd0d71a4bc7b.jpg](../icml_results/1227_RT-Lynx_%20Putting%20GEMM%20Sparsity%20in%20the%20Right%20Place%20for%20Diffusion%20Models/tables/fc82aa254c246268ed6ae05755d3bcd6280ebea820afde3b8cc2dd0d71a4bc7b.jpg)

<a id="paper-1209"></a>

## Geometric Conformal Prediction with Spatial Ranks and Multivariate Quantiles


### Images

![029320f595f1346a54083e936d329fd4fbd4a6482b1b4a34d9681bc889610a52.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/029320f595f1346a54083e936d329fd4fbd4a6482b1b4a34d9681bc889610a52.jpg)

![140cd50d832f5f4abc98cf74b29395532734fb5550968583bf93bd2c8c4dc040.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/140cd50d832f5f4abc98cf74b29395532734fb5550968583bf93bd2c8c4dc040.jpg)

![14229c427f555281615f16586ffe3a0b8cba706554115a82de4b7c484f843336.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/14229c427f555281615f16586ffe3a0b8cba706554115a82de4b7c484f843336.jpg)

![1b3b8c40b0a5ffc43cbd44989ceb1a3580ac0ae7f04473d499a62f658d1cf8fc.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/1b3b8c40b0a5ffc43cbd44989ceb1a3580ac0ae7f04473d499a62f658d1cf8fc.jpg)

![44b694b8be4fab9db6e7e4f062de7e1e364314594064654f0ed206bfd2e753de.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/44b694b8be4fab9db6e7e4f062de7e1e364314594064654f0ed206bfd2e753de.jpg)

![4de62a568152ff9bf14f0d8c7b036ca265151a2eeea73405d7f85c35c9b583a8.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/4de62a568152ff9bf14f0d8c7b036ca265151a2eeea73405d7f85c35c9b583a8.jpg)

![50c0c9b23f09bcec0af040e9be4b5e375d0baa5d4dfdcc31b99bcfcb58acf279.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/50c0c9b23f09bcec0af040e9be4b5e375d0baa5d4dfdcc31b99bcfcb58acf279.jpg)

![5bd0fbf22731f5c633f892ef2fb0d655fd388efa27604c3623bb5583a6cd031c.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/5bd0fbf22731f5c633f892ef2fb0d655fd388efa27604c3623bb5583a6cd031c.jpg)

![62acb2a0074de37749501119b90a66376366197fb22d556be86ac0e6cd308c5a.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/62acb2a0074de37749501119b90a66376366197fb22d556be86ac0e6cd308c5a.jpg)

![9888c219f7ef819c9c71121df2b657580e86b13b7aafb660a712a10f3c35aee3.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/9888c219f7ef819c9c71121df2b657580e86b13b7aafb660a712a10f3c35aee3.jpg)

![9b7a6259f1fd4cc82400eea24aafb8e2c12132c089450fade9d31ca0c9b7dd4e.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/9b7a6259f1fd4cc82400eea24aafb8e2c12132c089450fade9d31ca0c9b7dd4e.jpg)

![a2f6c99f0ec980a982cece90cc57cf28438d7a734eb4c48a7df961e6310db7a2.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/a2f6c99f0ec980a982cece90cc57cf28438d7a734eb4c48a7df961e6310db7a2.jpg)

![ad3df0a857fc693bbc82ddc4ddb68202fe5075b6a640cf41f840c6cf545a83ae.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/ad3df0a857fc693bbc82ddc4ddb68202fe5075b6a640cf41f840c6cf545a83ae.jpg)

![beb51275f5fedc806126e4a833c991a809edb4b47424a9c1e0fa862b93067036.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/beb51275f5fedc806126e4a833c991a809edb4b47424a9c1e0fa862b93067036.jpg)

![c939963b349743506118516ba150f637935b4b9050db3a564b769f2c84015902.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/c939963b349743506118516ba150f637935b4b9050db3a564b769f2c84015902.jpg)

![d01f76b149c07241e951cdc614c7140e24ac0063c4cf4e6ab3fe13adfd0608c7.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/d01f76b149c07241e951cdc614c7140e24ac0063c4cf4e6ab3fe13adfd0608c7.jpg)

![d06810f79853e7937d1ffc040e3f9bc52022478f3410d0d6edb819e8ba21cf07.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/d06810f79853e7937d1ffc040e3f9bc52022478f3410d0d6edb819e8ba21cf07.jpg)

![dab3b04db240ba1aec96c6cb37b6fdd6fd5956d0d8f89a6ad486837ada897e64.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/dab3b04db240ba1aec96c6cb37b6fdd6fd5956d0d8f89a6ad486837ada897e64.jpg)

![e1ff63995f33e05db2ecc56af406e48f663dfb2b04c0ba149d371caeef425b05.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/e1ff63995f33e05db2ecc56af406e48f663dfb2b04c0ba149d371caeef425b05.jpg)

![fdb93173acb2150ff4dc90814e9a3811e13fc53037d825c33029b4c74c65cac8.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/images/fdb93173acb2150ff4dc90814e9a3811e13fc53037d825c33029b4c74c65cac8.jpg)

### Tables

![4c84cb35101394539333c0df2a42761ffb5098df0419bde95cb75d9a33d7808a.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/tables/4c84cb35101394539333c0df2a42761ffb5098df0419bde95cb75d9a33d7808a.jpg)

![5a59e7deb9306c2af1302f1f7f7f51aedbae21bf2e477422a1c0746f004d1e9b.jpg](../icml_results/1228_Geometric%20Conformal%20Prediction%20with%20Spatial%20Ranks%20and%20Multivariate%20Quantiles/tables/5a59e7deb9306c2af1302f1f7f7f51aedbae21bf2e477422a1c0746f004d1e9b.jpg)

<a id="paper-1210"></a>

## Predicting evolutionary rate as a pretraining task improves genome language model representations


### Images

![00499f307aabfc3eb62c0be3da4b25bc86714c6ff1c20a4d68d0872f6a115227.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/00499f307aabfc3eb62c0be3da4b25bc86714c6ff1c20a4d68d0872f6a115227.jpg)

![0a694b211e13348058625253159c4bb9721db8a32568f6d12b06507ecb276990.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/0a694b211e13348058625253159c4bb9721db8a32568f6d12b06507ecb276990.jpg)

![1caf21c1815304301c0ac4af6ad7aa71418f829e7069fe5b7db41a4fda6d64ea.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/1caf21c1815304301c0ac4af6ad7aa71418f829e7069fe5b7db41a4fda6d64ea.jpg)

![4382d784948e0b6279f839a26551998910505eb8eee152ee5c83b761b73d677f.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/4382d784948e0b6279f839a26551998910505eb8eee152ee5c83b761b73d677f.jpg)

![801c92798160e7e3ff6198dac543edb1ade220e19a1265eadbeeb3329cc62c3d.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/801c92798160e7e3ff6198dac543edb1ade220e19a1265eadbeeb3329cc62c3d.jpg)

![8c350c1e14152b13afffb6140a32c3ceec5ffa101b3df2bf4c12920cf173d072.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/8c350c1e14152b13afffb6140a32c3ceec5ffa101b3df2bf4c12920cf173d072.jpg)

![b2c8a428e19b12dd6a66849abfd1e9ad4c1746798f322e67fe026d3567bca7f7.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/b2c8a428e19b12dd6a66849abfd1e9ad4c1746798f322e67fe026d3567bca7f7.jpg)

![c0a774f934eab513560a1c35acb699b69347275aed2a31564f70b93ae246cf00.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/c0a774f934eab513560a1c35acb699b69347275aed2a31564f70b93ae246cf00.jpg)

![c35b26fa7215cfc4e2baa469ab7e5edcab7d7349ea5b655c17c88a53ba4797a2.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/c35b26fa7215cfc4e2baa469ab7e5edcab7d7349ea5b655c17c88a53ba4797a2.jpg)

![ea9386ec2fbcababfba131109b402819004cb79da68e5dcc7810b3b82cca6418.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/ea9386ec2fbcababfba131109b402819004cb79da68e5dcc7810b3b82cca6418.jpg)

![f0ff9e03ae0358f73ed7c5ec596ab053288c8c9eb21f2c6bc2898711144bf781.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/images/f0ff9e03ae0358f73ed7c5ec596ab053288c8c9eb21f2c6bc2898711144bf781.jpg)

### Tables

![1104716de305a7dfa41ba4f82b75b1dabe4486b3f28f4f0fc60195e79e8a8b95.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/1104716de305a7dfa41ba4f82b75b1dabe4486b3f28f4f0fc60195e79e8a8b95.jpg)

![147d6de32fdd7c1be8374add4645a83a65bd282b7b1357230a0e152c761236b4.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/147d6de32fdd7c1be8374add4645a83a65bd282b7b1357230a0e152c761236b4.jpg)

![1f34b91fba5c113dd22cec98a0856c0f5b5794647d80967549bbe896ad01ce9a.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/1f34b91fba5c113dd22cec98a0856c0f5b5794647d80967549bbe896ad01ce9a.jpg)

![32141721f784ff7ce86a89544fe92094e70ff248b727a605776eedc08c15888f.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/32141721f784ff7ce86a89544fe92094e70ff248b727a605776eedc08c15888f.jpg)

![571174e2516b3d80b83973ce5c03463a2a0065b95197c5345933f94940087845.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/571174e2516b3d80b83973ce5c03463a2a0065b95197c5345933f94940087845.jpg)

![691d615f017f26fd9d11b2ecf41a4669f88e7ec5f4401e5c8aa2bc7bf9cfe7ce.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/691d615f017f26fd9d11b2ecf41a4669f88e7ec5f4401e5c8aa2bc7bf9cfe7ce.jpg)

![6dd438afb891f30e0c7dc642fc1c03d7db871cd5335400e65b141fd39fb8b8ef.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/6dd438afb891f30e0c7dc642fc1c03d7db871cd5335400e65b141fd39fb8b8ef.jpg)

![7122909cdcabaa973cf2b2c7652ad6327c8dbad43e43ca84de831727742641c5.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/7122909cdcabaa973cf2b2c7652ad6327c8dbad43e43ca84de831727742641c5.jpg)

![9af61d6e8e1821be6693211cbe023e31dd1a9bcd53ea467ccc323527acebdbcd.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/9af61d6e8e1821be6693211cbe023e31dd1a9bcd53ea467ccc323527acebdbcd.jpg)

![a3283d2ced086aa3c828e871c362dd7cd72f1bb269d53ec2a0b0475d669bd1e7.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/a3283d2ced086aa3c828e871c362dd7cd72f1bb269d53ec2a0b0475d669bd1e7.jpg)

![a804ec59a85ef65fa44770f10b39ace747b0b234c7ab85bbb427b385c255b544.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/a804ec59a85ef65fa44770f10b39ace747b0b234c7ab85bbb427b385c255b544.jpg)

![b022f2740b24f913eb56e2201c0a105c28108a61acf29d13f396549b3864864a.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/b022f2740b24f913eb56e2201c0a105c28108a61acf29d13f396549b3864864a.jpg)

![b4e1b9c24c36da8fd386f2edbe52a21cb3b9625a40039e584b19be1e34f542ee.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/b4e1b9c24c36da8fd386f2edbe52a21cb3b9625a40039e584b19be1e34f542ee.jpg)

![bed38d10e80e429e0f954860d464eb446f99b55dac7ee44249fd24f8ee8aca8a.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/bed38d10e80e429e0f954860d464eb446f99b55dac7ee44249fd24f8ee8aca8a.jpg)

![d5148b8d6a42878e1bd3d6a912c8f6143089ab25693b4aff88d07b960cf881f8.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/d5148b8d6a42878e1bd3d6a912c8f6143089ab25693b4aff88d07b960cf881f8.jpg)

![dc29237734a3788555ffd38c6bf7b053f75dffe4aac34a0a316343a9ad0492d4.jpg](../icml_results/1229_Predicting%20evolutionary%20rate%20as%20a%20pretraining%20task%20improves%20genome%20language%20model%20representations/tables/dc29237734a3788555ffd38c6bf7b053f75dffe4aac34a0a316343a9ad0492d4.jpg)

<a id="paper-1211"></a>

## Fast Mixing Steady-State Control in Markov Decision Processes


### Images

![157fe7e5df661973afb2aaf0cc1976f31321cbe0ad7e3621be0634b1dac5e770.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/157fe7e5df661973afb2aaf0cc1976f31321cbe0ad7e3621be0634b1dac5e770.jpg)

![322e06acfbbe24decb5a722319623ee77543615f8238121491263cc0ef7620f3.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/322e06acfbbe24decb5a722319623ee77543615f8238121491263cc0ef7620f3.jpg)

![3ae6ff54cf8e7ccbdb1aa61d32aba69d9a2ae6973afe899e18f5a68904f486cc.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/3ae6ff54cf8e7ccbdb1aa61d32aba69d9a2ae6973afe899e18f5a68904f486cc.jpg)

![6857cad4bc2049493e334094e9162e55b056be0976c9084b7648a98bc03d5f55.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/6857cad4bc2049493e334094e9162e55b056be0976c9084b7648a98bc03d5f55.jpg)

![695b16d1e1cf69e76084b972e6751dbaffbb9b04f553d33374aa2ee29ffc85a3.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/695b16d1e1cf69e76084b972e6751dbaffbb9b04f553d33374aa2ee29ffc85a3.jpg)

![7308eb4b29120dddbd7b5c76e7a435161e384964f5492c63094961fcb923f6af.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/7308eb4b29120dddbd7b5c76e7a435161e384964f5492c63094961fcb923f6af.jpg)

![7312aebcf024df3f168581beba0b764c461398871c4d4a890e5fd8f5f45545c7.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/7312aebcf024df3f168581beba0b764c461398871c4d4a890e5fd8f5f45545c7.jpg)

![9964ec78b54e0cefa1a0b3e54bf26a1110087893d754539dd6345b70a015fa12.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/9964ec78b54e0cefa1a0b3e54bf26a1110087893d754539dd6345b70a015fa12.jpg)

![a05a109b604dabdc8cc48beb4e69b48d48ba016e18c6403e85ecdc57a817bcde.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/a05a109b604dabdc8cc48beb4e69b48d48ba016e18c6403e85ecdc57a817bcde.jpg)

![b2b08655f11e3f3f6ff1d181290cd547d39af0876dd365671c05fd8c6893a754.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/b2b08655f11e3f3f6ff1d181290cd547d39af0876dd365671c05fd8c6893a754.jpg)

![b474e47a88582ec3cd29895067cd71a7f5f6df3a1b5f9db541ae00c215474286.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/b474e47a88582ec3cd29895067cd71a7f5f6df3a1b5f9db541ae00c215474286.jpg)

![d0791b4a9c9f5d1332054b2c2e907e247ffed1abcf30a8e76180ca97b20b1751.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/d0791b4a9c9f5d1332054b2c2e907e247ffed1abcf30a8e76180ca97b20b1751.jpg)

![d427e90ae71abf0fc3213c9b2d631a4e95ea71b0bd22412dbd490c4644075236.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/d427e90ae71abf0fc3213c9b2d631a4e95ea71b0bd22412dbd490c4644075236.jpg)

![d75d67fe6fd6a9e0ed3f15e17820ee35c79a7e9545e554553f366e35d3b73da8.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/d75d67fe6fd6a9e0ed3f15e17820ee35c79a7e9545e554553f366e35d3b73da8.jpg)

![d8585860b2ff7079877789b5f68382de9bf87cadde3fa040a3d29c4e736a61d4.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/d8585860b2ff7079877789b5f68382de9bf87cadde3fa040a3d29c4e736a61d4.jpg)

![e19ae8922679f36b9b49993549dcbab49cd052d40bae8cefb47dc032a2dddbf6.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/e19ae8922679f36b9b49993549dcbab49cd052d40bae8cefb47dc032a2dddbf6.jpg)

![eccdf69fcc827061341e1a99ca4186403b3d2df91b77bebafd1ef29aa247700f.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/eccdf69fcc827061341e1a99ca4186403b3d2df91b77bebafd1ef29aa247700f.jpg)

![f54e835f7cae70075229750ed8b1e5a538d53072bada3fb54a344c88dc7cf545.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/images/f54e835f7cae70075229750ed8b1e5a538d53072bada3fb54a344c88dc7cf545.jpg)

### Tables

![4b41ec4cec1dc7b65067672b220629d405f46b8fd2bbc330eb5e51b820c1c5e5.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/tables/4b41ec4cec1dc7b65067672b220629d405f46b8fd2bbc330eb5e51b820c1c5e5.jpg)

![4fbc942124fcb7bfa30ab3c22701d8677ce4fdb0d5fe0d5a4a468ce0ad7f23b2.jpg](../icml_results/1230_Fast%20Mixing%20Steady-State%20Control%20in%20Markov%20Decision%20Processes/tables/4fbc942124fcb7bfa30ab3c22701d8677ce4fdb0d5fe0d5a4a468ce0ad7f23b2.jpg)

<a id="paper-1212"></a>

## Beyond Binary: Continuous State Optimization with Graph-Structured Objectives


### Images

![316c3ec4f3e7b77b5248607f056a3eb4b51fa7b146f208a695e1ea47826f548e.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/images/316c3ec4f3e7b77b5248607f056a3eb4b51fa7b146f208a695e1ea47826f548e.jpg)

![49ac2096ae8eee3d0e41f07936503b75bdb342bfbb235272c2f6bb20d541e3ee.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/images/49ac2096ae8eee3d0e41f07936503b75bdb342bfbb235272c2f6bb20d541e3ee.jpg)

![756f86076b6e7745889901fac8db0ebcb5a3d5c04cda440f0e8bd0943abcee96.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/images/756f86076b6e7745889901fac8db0ebcb5a3d5c04cda440f0e8bd0943abcee96.jpg)

![b600bceddfc85f177b2d8f79f60ad1ffd2e6fcc4e5939cc83749954acb95f440.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/images/b600bceddfc85f177b2d8f79f60ad1ffd2e6fcc4e5939cc83749954acb95f440.jpg)

![f39c1dcd06417bf80aaa75579821ddd073613553e9ae742d269e30ae26445630.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/images/f39c1dcd06417bf80aaa75579821ddd073613553e9ae742d269e30ae26445630.jpg)

### Tables

![1e9d951e2c4b89afd4cbf585a3ceeaf8e1995f0a564f0989e0f408596ad0dc7f.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/tables/1e9d951e2c4b89afd4cbf585a3ceeaf8e1995f0a564f0989e0f408596ad0dc7f.jpg)

![5c6801c9bf5e802e3a5d8bd8b1087ce67666037da4152d60bb6e2a87d93cf53f.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/tables/5c6801c9bf5e802e3a5d8bd8b1087ce67666037da4152d60bb6e2a87d93cf53f.jpg)

![a5e4a7495213d21336bd17edf63a6bf82f0335dccddb682aec4532960916dbbe.jpg](../icml_results/1231_Beyond%20Binary_%20Continuous%20State%20Optimization%20with%20Graph-Structured%20Objectives/tables/a5e4a7495213d21336bd17edf63a6bf82f0335dccddb682aec4532960916dbbe.jpg)

<a id="paper-1213"></a>

## Optimized Deferral for Imbalanced Settings


### Images

![2b16d8e94a0cbf7b1bb393ebdb12476faf22b872eb66d5b1b0f883160dac1c9a.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/images/2b16d8e94a0cbf7b1bb393ebdb12476faf22b872eb66d5b1b0f883160dac1c9a.jpg)

### Tables

![0ffc435628f3154356ecf8315869156873b3b741e90c09ae658a9545e48877b4.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/0ffc435628f3154356ecf8315869156873b3b741e90c09ae658a9545e48877b4.jpg)

![243634370cbb1efca4dff595cbb5f0566d09997e796b01b45ea253d00823eaac.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/243634370cbb1efca4dff595cbb5f0566d09997e796b01b45ea253d00823eaac.jpg)

![31b137c325bd9af5a3bd97784a893deb0a16063938c638e8b5b1f3618f916be8.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/31b137c325bd9af5a3bd97784a893deb0a16063938c638e8b5b1f3618f916be8.jpg)

![5ff6063f2781d6b42983cc509a7a42d712bfcbf9206a38da52a83b6edfaf0ad9.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/5ff6063f2781d6b42983cc509a7a42d712bfcbf9206a38da52a83b6edfaf0ad9.jpg)

![61d1de5b416850ada3ac0c667e6aacfd3a5cb40d115c80293bec62cf51b73783.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/61d1de5b416850ada3ac0c667e6aacfd3a5cb40d115c80293bec62cf51b73783.jpg)

![d390815b31b805dacf7ebc34df9578e2512076a9450a6a0c3a1e257c86b23369.jpg](../icml_results/1232_Optimized%20Deferral%20for%20Imbalanced%20Settings/tables/d390815b31b805dacf7ebc34df9578e2512076a9450a6a0c3a1e257c86b23369.jpg)

<a id="paper-1214"></a>

## A Theoretical Framework for Modular Learning of Robust Generative Models


### Images

![0e2c3e71ff3c5412be5ae44159b2e7853cb4002f02f652791fb21022a5a4014c.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/0e2c3e71ff3c5412be5ae44159b2e7853cb4002f02f652791fb21022a5a4014c.jpg)

![15adb776afdeafbd4bbf997a7cc186a37d31c027cdeaae1d7dad90cabd43018f.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/15adb776afdeafbd4bbf997a7cc186a37d31c027cdeaae1d7dad90cabd43018f.jpg)

![267873dd050b6f3e1d65e38484fece1f1b2985e686a8cec692d8590e03351c48.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/267873dd050b6f3e1d65e38484fece1f1b2985e686a8cec692d8590e03351c48.jpg)

![3d2cdaba0af04e0627ae77b6a9ca6e672c9b499d9dbf04d06eab4e55c7d8fa16.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/3d2cdaba0af04e0627ae77b6a9ca6e672c9b499d9dbf04d06eab4e55c7d8fa16.jpg)

![3e69e883158c92102aa9ee252e3d50786d4ecead45898bca425147e136dfa14b.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/3e69e883158c92102aa9ee252e3d50786d4ecead45898bca425147e136dfa14b.jpg)

![72277e2161e0b3bc53aef8e8f9a587ec9852bf27b2fb2529d2a16790207699fa.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/72277e2161e0b3bc53aef8e8f9a587ec9852bf27b2fb2529d2a16790207699fa.jpg)

![7b81f7cbd4c57d173e585d8bc058ab5d44cc98f785d0b85bffe21272728bfd2a.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/7b81f7cbd4c57d173e585d8bc058ab5d44cc98f785d0b85bffe21272728bfd2a.jpg)

![7c5441cc325ddf9ebd90a35d67c5195f6cd6aa1f86c46f83ef6307e04dd88288.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/7c5441cc325ddf9ebd90a35d67c5195f6cd6aa1f86c46f83ef6307e04dd88288.jpg)

![9582e710507b3ddfc07c1125c896394681ad3f69bf0e8a696f9d1a6e3b44146d.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/9582e710507b3ddfc07c1125c896394681ad3f69bf0e8a696f9d1a6e3b44146d.jpg)

![ae6e28df40d2e4f950ee88be02e6352bbe05ea7538ba56e2a903a9a0affa3665.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/ae6e28df40d2e4f950ee88be02e6352bbe05ea7538ba56e2a903a9a0affa3665.jpg)

![ecaf7eb1b47c13676897cef0ba12ffc6556341802e236a121654975b4a8de2f6.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/images/ecaf7eb1b47c13676897cef0ba12ffc6556341802e236a121654975b4a8de2f6.jpg)

### Tables

![58958d04e35c6f9ba8e60e1f6112b110c0968c9b88b4df80436b1f41f7f70432.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/tables/58958d04e35c6f9ba8e60e1f6112b110c0968c9b88b4df80436b1f41f7f70432.jpg)

![8d07e3c2a5a7939301bc5bd964d4517e2f05d813e176df30e247acd2fa8601dc.jpg](../icml_results/1233_A%20Theoretical%20Framework%20for%20Modular%20Learning%20of%20Robust%20Generative%20Models/tables/8d07e3c2a5a7939301bc5bd964d4517e2f05d813e176df30e247acd2fa8601dc.jpg)

<a id="paper-1215"></a>

## BLIPs: Bayesian Learned Interatomic Potentials


### Images

![b44daccf35210a1e0f8ddd9894c6d1e628e4b35b17319f944ee9814e4acf1fca.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/images/b44daccf35210a1e0f8ddd9894c6d1e628e4b35b17319f944ee9814e4acf1fca.jpg)

![ba1cdba3c87f818b462c07a27eaccb092d04a92886c1a9dd8ef7ca895d760af4.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/images/ba1cdba3c87f818b462c07a27eaccb092d04a92886c1a9dd8ef7ca895d760af4.jpg)

![ea8a9283858fdd8860f49c5f9b105423ef2594ff6c4cecd25c02ca84fb8436ee.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/images/ea8a9283858fdd8860f49c5f9b105423ef2594ff6c4cecd25c02ca84fb8436ee.jpg)

![ebe529af1dbba49c706d7f7d2bb4078688b7c666d7b4b3d2dfbcd65eb1859007.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/images/ebe529af1dbba49c706d7f7d2bb4078688b7c666d7b4b3d2dfbcd65eb1859007.jpg)

### Tables

![50dc3520f440a896c2a4ec4ddeb0208b31296a894b6808d86946ef38adbe7584.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/50dc3520f440a896c2a4ec4ddeb0208b31296a894b6808d86946ef38adbe7584.jpg)

![59588db29189610cc155237c140b103ee675018ff222b819ed86e4dbe6f40963.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/59588db29189610cc155237c140b103ee675018ff222b819ed86e4dbe6f40963.jpg)

![745d86f122e31fa1e97cbfc1bc98cdd20c8a8c64dd3d0f82c58da8501fa14701.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/745d86f122e31fa1e97cbfc1bc98cdd20c8a8c64dd3d0f82c58da8501fa14701.jpg)

![abedb829e23407f89b35fc8b8c19db0a4965dd58ad7ae84b42ac6c8172f94744.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/abedb829e23407f89b35fc8b8c19db0a4965dd58ad7ae84b42ac6c8172f94744.jpg)

![ad0a24a991bd1335d4e69642e7a13e67d9ff5cbe219f503c44f6901ea35ff7f4.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/ad0a24a991bd1335d4e69642e7a13e67d9ff5cbe219f503c44f6901ea35ff7f4.jpg)

![e40276f7ae4e1fbd0d8edfebbfe43cb35050376a125671241644ed55438796fb.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/e40276f7ae4e1fbd0d8edfebbfe43cb35050376a125671241644ed55438796fb.jpg)

![f7ac325fc53aca15444b6c2b125973dfc10f3fab4a530c637b7f00fa89bf917e.jpg](../icml_results/1234_BLIPs_%20Bayesian%20Learned%20Interatomic%20Potentials/tables/f7ac325fc53aca15444b6c2b125973dfc10f3fab4a530c637b7f00fa89bf917e.jpg)

<a id="paper-1216"></a>

## PLATE: Plasticity-Tunable Efficient Adapters for Geometry-Aware Continual Learning


### Images

![045e4a27e3161e86c321460fe7a992524e403953bf11e8fbf95c9ea291607789.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/045e4a27e3161e86c321460fe7a992524e403953bf11e8fbf95c9ea291607789.jpg)

![0e7f335a4308e836a33be03c8823bc0451ea1233e366ccb04038b1d9840a5f71.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/0e7f335a4308e836a33be03c8823bc0451ea1233e366ccb04038b1d9840a5f71.jpg)

![10629bdd5f7d3e4d0ef194d72dffb08cc397da8d540588ca592fada68d0b2d37.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/10629bdd5f7d3e4d0ef194d72dffb08cc397da8d540588ca592fada68d0b2d37.jpg)

![1c69c51b0ccd9375c312c4d0a20fc1c394f74a89bebd8aa01e27f54a5c510c72.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/1c69c51b0ccd9375c312c4d0a20fc1c394f74a89bebd8aa01e27f54a5c510c72.jpg)

![2011e21ca28daec4010252d21e054f228d5b0a9603d932d4eae8d04ea1ce673f.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/2011e21ca28daec4010252d21e054f228d5b0a9603d932d4eae8d04ea1ce673f.jpg)

![31e8d144f7cd1f722b768e7e98db2a2f61cc054fd8e025a6e62fea9b14c0cb83.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/31e8d144f7cd1f722b768e7e98db2a2f61cc054fd8e025a6e62fea9b14c0cb83.jpg)

![5ec85b0f7cfb80cb4a669a0a082e89691bab9186560a7f6bb496d0607a3c90e2.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/5ec85b0f7cfb80cb4a669a0a082e89691bab9186560a7f6bb496d0607a3c90e2.jpg)

![725282ca999774c9f9be7eb71f5b9a2764e62cc1b8cafe10195db047120bdcd5.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/725282ca999774c9f9be7eb71f5b9a2764e62cc1b8cafe10195db047120bdcd5.jpg)

![869c9dab9e0d1d1485e17c66c3f9521ff43650dfe4b079d1ea6601ead693675d.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/869c9dab9e0d1d1485e17c66c3f9521ff43650dfe4b079d1ea6601ead693675d.jpg)

![badeaeb12782313bbb0f05f12ea69bc82d9bf741695a5b54ebd515f8d542c48c.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/badeaeb12782313bbb0f05f12ea69bc82d9bf741695a5b54ebd515f8d542c48c.jpg)

![c584b5bdedfa3b6cd87f11352d57c5bf88823cca732ab1b8c6e16a1bdfad29a1.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/c584b5bdedfa3b6cd87f11352d57c5bf88823cca732ab1b8c6e16a1bdfad29a1.jpg)

![c8617d5a2382666f3b32bc1312af1d0fa993c392448569ecd936b090e3d5ea40.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/c8617d5a2382666f3b32bc1312af1d0fa993c392448569ecd936b090e3d5ea40.jpg)

![cb69e26dd0dfeb5aebdc59515c547cbb1105baff2813bba1c8ea43603155d433.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/cb69e26dd0dfeb5aebdc59515c547cbb1105baff2813bba1c8ea43603155d433.jpg)

![df5abdcaafdc4ed5530922f7bb80efbf1739145d85b30ab8b1198c499d970003.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/df5abdcaafdc4ed5530922f7bb80efbf1739145d85b30ab8b1198c499d970003.jpg)

![df6b8efd9bac295bfd1a48d64ef6c2ee4b14a20da4a0439633e71d983864e803.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/df6b8efd9bac295bfd1a48d64ef6c2ee4b14a20da4a0439633e71d983864e803.jpg)

![e8e76bf9321d921b2074b211354c55c36809cf07d2e2fa7673cabdee07f4810a.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/e8e76bf9321d921b2074b211354c55c36809cf07d2e2fa7673cabdee07f4810a.jpg)

![f01a2734d17495dd77875102c1315cf9726dc375fe29ce0bd4a454aa7c8955f0.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/f01a2734d17495dd77875102c1315cf9726dc375fe29ce0bd4a454aa7c8955f0.jpg)

![fb25f2735a4ca10f8c4ac4a3fc90663acc42114b42d6aa24216b106e1f70e971.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/images/fb25f2735a4ca10f8c4ac4a3fc90663acc42114b42d6aa24216b106e1f70e971.jpg)

### Tables

![00a573744b74fb4b82933172e1401341aaf288017ac607f41d42d45978597f2c.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/tables/00a573744b74fb4b82933172e1401341aaf288017ac607f41d42d45978597f2c.jpg)

![bf270e67f75ed59d694b53956c8f529bc02e2a64b811fb006eba3bd3d6e27aa2.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/tables/bf270e67f75ed59d694b53956c8f529bc02e2a64b811fb006eba3bd3d6e27aa2.jpg)

![fbb1f7954b15617e0103c1f00adfc732db99e8acdfbcc049ab2c4fc67ce013b9.jpg](../icml_results/1235_PLATE_%20Plasticity-Tunable%20Efficient%20Adapters%20for%20Geometry-Aware%20Continual%20Learning/tables/fbb1f7954b15617e0103c1f00adfc732db99e8acdfbcc049ab2c4fc67ce013b9.jpg)

<a id="paper-1217"></a>

## Abductive Reasoning with Probabilistic Commonsense


### Images

![4eabaabb4b663fbd05eb6a57754df200d41be6fa1e2f2559851a40dc096d94a0.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/images/4eabaabb4b663fbd05eb6a57754df200d41be6fa1e2f2559851a40dc096d94a0.jpg)

![a64a8410a4fa8942364dad5d4528b0b839ba1e68ecd6f8dc95b96d162c6d0042.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/images/a64a8410a4fa8942364dad5d4528b0b839ba1e68ecd6f8dc95b96d162c6d0042.jpg)

![e26716932e0f67cb7e0fc76a519c8231ff73e3bc7b7bdab3127445c557f93ff8.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/images/e26716932e0f67cb7e0fc76a519c8231ff73e3bc7b7bdab3127445c557f93ff8.jpg)

### Tables

![794a9d42f9ff5ba3291365f44b6824fcd865f3003028f2face4e2c7c27d23b9f.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/tables/794a9d42f9ff5ba3291365f44b6824fcd865f3003028f2face4e2c7c27d23b9f.jpg)

![ac38664e36c77073d2c15d17de6356cc56c242d020367e2dc0bfe461f1ac0bef.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/tables/ac38664e36c77073d2c15d17de6356cc56c242d020367e2dc0bfe461f1ac0bef.jpg)

![cb37c1311b1761e5bf1698286c21562ddbccbce8f7b9fe39dbeb97eb83567bcf.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/tables/cb37c1311b1761e5bf1698286c21562ddbccbce8f7b9fe39dbeb97eb83567bcf.jpg)

![d7578dc6618c01c820ca50231c48367944f5c94bd2aa487dad0fcbb0e176631f.jpg](../icml_results/1236_Abductive%20Reasoning%20with%20Probabilistic%20Commonsense/tables/d7578dc6618c01c820ca50231c48367944f5c94bd2aa487dad0fcbb0e176631f.jpg)

<a id="paper-1218"></a>

## Active Continual Learning with Metaplastic Binary Bayesian Neural Networks


### Images

![0cf1ad469c62856127020f595f2fec30a90c4d0439805ca305cef5ac107e77e4.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/0cf1ad469c62856127020f595f2fec30a90c4d0439805ca305cef5ac107e77e4.jpg)

![14725c265a63900bdc22e6f7fbcfcd7bad3061d30c478883eb86f95a14b9eea6.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/14725c265a63900bdc22e6f7fbcfcd7bad3061d30c478883eb86f95a14b9eea6.jpg)

![1698d8820747ebb3d89a8141a9d30fc6a81291db1a52adc7b4c86db45317b101.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/1698d8820747ebb3d89a8141a9d30fc6a81291db1a52adc7b4c86db45317b101.jpg)

![2fbe71522a73766d4d78b4cb43c090f6365a182feb2dbd4bc5af5176f379e106.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/2fbe71522a73766d4d78b4cb43c090f6365a182feb2dbd4bc5af5176f379e106.jpg)

![52e3979800bcb04e67930a6ac78ca7863971f311e801ecf023df45b094b6bed9.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/52e3979800bcb04e67930a6ac78ca7863971f311e801ecf023df45b094b6bed9.jpg)

![95116d4e67ec0418620ff0bb9a2003e9bdd0472b4f8fc759fa2f80f9ec3a72fd.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/95116d4e67ec0418620ff0bb9a2003e9bdd0472b4f8fc759fa2f80f9ec3a72fd.jpg)

![a5f563dc7962420e8ff715fddf6e76fb44562180f2e8f3f67a9d398099c8d6f1.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/a5f563dc7962420e8ff715fddf6e76fb44562180f2e8f3f67a9d398099c8d6f1.jpg)

![abaf475c915f149d36176cd199178334554767c11ae775b1ec8ef87a1560774d.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/abaf475c915f149d36176cd199178334554767c11ae775b1ec8ef87a1560774d.jpg)

![cdba94b4824177de295f13fcb0f06dae5b1f500a2140d8b6ec324276e4b65ceb.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/cdba94b4824177de295f13fcb0f06dae5b1f500a2140d8b6ec324276e4b65ceb.jpg)

![da3bd4cfdca38726aedce6be28b6ce7e61366739b171f1960dcb28c4c7a8f2a1.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/da3bd4cfdca38726aedce6be28b6ce7e61366739b171f1960dcb28c4c7a8f2a1.jpg)

![e1df7805cd3ad3a9018f8be61364a831394aae1e050204a9c2db0f1bd17dfb73.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/e1df7805cd3ad3a9018f8be61364a831394aae1e050204a9c2db0f1bd17dfb73.jpg)

![e22ad5364b9708c8ded1841e1f56bff45c2c931219142c9c196ad9ed30e91eb0.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/e22ad5364b9708c8ded1841e1f56bff45c2c931219142c9c196ad9ed30e91eb0.jpg)

![e6bb5b750512253479493723105fb6cef6a4234406d21742e41710c82b64178f.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/e6bb5b750512253479493723105fb6cef6a4234406d21742e41710c82b64178f.jpg)

![eed4c51c22357831a0afcd7579bcdd0a59fd621a38da267310733ec212d68b91.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/eed4c51c22357831a0afcd7579bcdd0a59fd621a38da267310733ec212d68b91.jpg)

![f58ecae215d7c268568b9c7461c0346b522c4d8976ff688a5dfefde2075b20e7.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/images/f58ecae215d7c268568b9c7461c0346b522c4d8976ff688a5dfefde2075b20e7.jpg)

### Tables

![1f98339c019e465120254b0e658dfbc9b76dea2d7818265a9e4eb81fcb9d1b8a.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/1f98339c019e465120254b0e658dfbc9b76dea2d7818265a9e4eb81fcb9d1b8a.jpg)

![2621926ef8f1ad65dc705dbe3ffda25a93a5aed252f7a713ad7efa9f0f4d0c17.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/2621926ef8f1ad65dc705dbe3ffda25a93a5aed252f7a713ad7efa9f0f4d0c17.jpg)

![47c8923e7a649df8dc924e1ec0bd4f9fe3e22f9ee449cd0a8dfb9ccb916c71f2.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/47c8923e7a649df8dc924e1ec0bd4f9fe3e22f9ee449cd0a8dfb9ccb916c71f2.jpg)

![50d7fdbbf252b925a6d8f241061a39351278537f61c2cd87bea0305105b62353.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/50d7fdbbf252b925a6d8f241061a39351278537f61c2cd87bea0305105b62353.jpg)

![56e3b5d7de6ad0b745b32b06e470a9c2949f6ff5e8a3ebd44b0b470bc702efc0.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/56e3b5d7de6ad0b745b32b06e470a9c2949f6ff5e8a3ebd44b0b470bc702efc0.jpg)

![581f249accee3f031f9073126bdcf0e6abfe0b16c1566f3700266ceb3c1624b0.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/581f249accee3f031f9073126bdcf0e6abfe0b16c1566f3700266ceb3c1624b0.jpg)

![5c217ecaf5031994921b10855245e0ddf704435c401a0a088e05c70d30f33066.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/5c217ecaf5031994921b10855245e0ddf704435c401a0a088e05c70d30f33066.jpg)

![5c41e3db14e0874e8f62e73f194901e17f05435ab8b3092e7879fa3f78cec31d.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/5c41e3db14e0874e8f62e73f194901e17f05435ab8b3092e7879fa3f78cec31d.jpg)

![61cee47933d06a5aba2c5427246d78cd41f6700d241fb77bf8729a757fa55fba.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/61cee47933d06a5aba2c5427246d78cd41f6700d241fb77bf8729a757fa55fba.jpg)

![6e1f8af1b57d1b15e25c3ba1f3065f81121e880d9feeec32e4917d7e8c14df63.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/6e1f8af1b57d1b15e25c3ba1f3065f81121e880d9feeec32e4917d7e8c14df63.jpg)

![737d3aab29031e00c6840fda4e10ecf4216a886b3c38a27cd6f08b495c42e054.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/737d3aab29031e00c6840fda4e10ecf4216a886b3c38a27cd6f08b495c42e054.jpg)

![7ee5ebb6e1d1bb771fa9b3ffce2202590279b4e059ccaa568f929ab9dd1ad6ad.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/7ee5ebb6e1d1bb771fa9b3ffce2202590279b4e059ccaa568f929ab9dd1ad6ad.jpg)

![86e654183638063e8292e74173e4203df5ba5dd7e665873c883ed56334c75e7a.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/86e654183638063e8292e74173e4203df5ba5dd7e665873c883ed56334c75e7a.jpg)

![89bf032121f7f4273c4af02bb6895d7cf81a296848e2dd09a48bff0b75315b92.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/89bf032121f7f4273c4af02bb6895d7cf81a296848e2dd09a48bff0b75315b92.jpg)

![95b5111b93aa8f5f769ecbea8545d371b3d361a187dae1ab892569e59b7e3d24.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/95b5111b93aa8f5f769ecbea8545d371b3d361a187dae1ab892569e59b7e3d24.jpg)

![a75ef10504c71e64be4a31b72fd648a9384f0a29ddcd902422b14e9af6656df5.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/a75ef10504c71e64be4a31b72fd648a9384f0a29ddcd902422b14e9af6656df5.jpg)

![b813133d1a7be5e0117262efcdf675cc5050a98548f81b6dd87b22458aba8bbb.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/b813133d1a7be5e0117262efcdf675cc5050a98548f81b6dd87b22458aba8bbb.jpg)

![d2134e3609692144feca6b5d48854c10c88e250eb6b478e20d462c92595157c9.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/d2134e3609692144feca6b5d48854c10c88e250eb6b478e20d462c92595157c9.jpg)

![df85e7eeab7229b42679ea814b195d66901c8e90169506fecfe19c372b0db474.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/df85e7eeab7229b42679ea814b195d66901c8e90169506fecfe19c372b0db474.jpg)

![e8b810e5f19f9d0da7f892810e5c46bd4f9f836f7b9421e03d473bc87a876e78.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/e8b810e5f19f9d0da7f892810e5c46bd4f9f836f7b9421e03d473bc87a876e78.jpg)

![efa3fc5b461662e97e7433d701a83e249ba25dd54057f47301d24e9103732394.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/efa3fc5b461662e97e7433d701a83e249ba25dd54057f47301d24e9103732394.jpg)

![f7ef609cb408daf88bf55e934aac702243981a09ed99e1e3a280836333850849.jpg](../icml_results/1237_Active%20Continual%20Learning%20with%20Metaplastic%20Binary%20Bayesian%20Neural%20Networks/tables/f7ef609cb408daf88bf55e934aac702243981a09ed99e1e3a280836333850849.jpg)

<a id="paper-1219"></a>

## Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy


### Images

![1f6d08eb7e87779bd205b76e37e84a9aeb6543e8cb954e075a525de34e6ec835.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/images/1f6d08eb7e87779bd205b76e37e84a9aeb6543e8cb954e075a525de34e6ec835.jpg)

![26df0757b451207bd7ab77d4b3a4bc14562c0f35b08cdd2ccdc2bd46ccb571e1.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/images/26df0757b451207bd7ab77d4b3a4bc14562c0f35b08cdd2ccdc2bd46ccb571e1.jpg)

![553cf3d6047b3f0055ac559d0a667ad4ee361ae91daced81755cd9cb7630b2a5.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/images/553cf3d6047b3f0055ac559d0a667ad4ee361ae91daced81755cd9cb7630b2a5.jpg)

![a8c8cae5c921c05cd1ce5d40e5c260fe0eb8255499771f1b0f60bbf19843dfea.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/images/a8c8cae5c921c05cd1ce5d40e5c260fe0eb8255499771f1b0f60bbf19843dfea.jpg)

### Tables

![1e9c8cd1dd299b8d62db3c2210f14e37bcdb575005f189818f1d5bdeba3f3253.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/tables/1e9c8cd1dd299b8d62db3c2210f14e37bcdb575005f189818f1d5bdeba3f3253.jpg)

![449c90d645b11ba7999f59c0a09398bf062f0280cc8e3e5f11c26ea4f724b47a.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/tables/449c90d645b11ba7999f59c0a09398bf062f0280cc8e3e5f11c26ea4f724b47a.jpg)

![4f73327311eab7695d5b99186dfb3c977670c6d70ee176c6a6ff53d9b28e3c68.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/tables/4f73327311eab7695d5b99186dfb3c977670c6d70ee176c6a6ff53d9b28e3c68.jpg)

![721a8207514c76aa8ffb906c0febe50ac4c61c752093dea5afe921741772ebe0.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/tables/721a8207514c76aa8ffb906c0febe50ac4c61c752093dea5afe921741772ebe0.jpg)

![dc3776902dd1a217e06cebe05d7b7884b9bc08ea403b8ebb4ace33cd6cb59518.jpg](../icml_results/1238_Scalable%20and%20Differentiable%20Point-Cloud%20Registration%20Using%20Maximum%20Mean%20Discrepancy/tables/dc3776902dd1a217e06cebe05d7b7884b9bc08ea403b8ebb4ace33cd6cb59518.jpg)

<a id="paper-1220"></a>

## ANTiC: Adaptive Neural Temporal In Situ Compressor


### Images

![07e051a9a445c397b9d1da2ec3a9594ef4ff72db9b3db8052a44d8a718073dd5.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/07e051a9a445c397b9d1da2ec3a9594ef4ff72db9b3db8052a44d8a718073dd5.jpg)

![141aa9204b771b1e8e382f4ae3d97c96f1a920df8b5d9330a03c4ef19f16b923.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/141aa9204b771b1e8e382f4ae3d97c96f1a920df8b5d9330a03c4ef19f16b923.jpg)

![1b0a9c95c29c3d5ccf6860134b72ac2a8c7e363501b2f97d10a8d83e8a123058.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/1b0a9c95c29c3d5ccf6860134b72ac2a8c7e363501b2f97d10a8d83e8a123058.jpg)

![2510baef6889c5c64bb09b9e8110d19ff328c6290fe93e368526176a1a54e60c.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/2510baef6889c5c64bb09b9e8110d19ff328c6290fe93e368526176a1a54e60c.jpg)

![343668ef46d1adcbf6077bc8418750dca53218f619d05e6b0ae68a10f5868dd2.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/343668ef46d1adcbf6077bc8418750dca53218f619d05e6b0ae68a10f5868dd2.jpg)

![3a7bbbf8c81108ac57d1b38dd8a9c0d518291b7dd13ab5edb00ea8380bdeac8e.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/3a7bbbf8c81108ac57d1b38dd8a9c0d518291b7dd13ab5edb00ea8380bdeac8e.jpg)

![4c48201de3cc0af95c985ce35dba0c6a9351318ba6545bbbe6d710c09ad09583.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/4c48201de3cc0af95c985ce35dba0c6a9351318ba6545bbbe6d710c09ad09583.jpg)

![5ba4228acb15198c82303dc25c3944e87a9f9bb3efcb7906d41e0b49b218e4b8.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/5ba4228acb15198c82303dc25c3944e87a9f9bb3efcb7906d41e0b49b218e4b8.jpg)

![5d60ad7ac7721b5e952768ca22646d50a4ea9ebf29353e8be5ad802ce1ca25d5.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/5d60ad7ac7721b5e952768ca22646d50a4ea9ebf29353e8be5ad802ce1ca25d5.jpg)

![7737834f9f6ef06f8fc1b0f5f7658f66d59b5147c20ab51156b872f89a7b49a2.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/7737834f9f6ef06f8fc1b0f5f7658f66d59b5147c20ab51156b872f89a7b49a2.jpg)

![7ad11abb3f8d240db4b654827d7d7bf2d68d62851b4084803b686b4a24253859.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/7ad11abb3f8d240db4b654827d7d7bf2d68d62851b4084803b686b4a24253859.jpg)

![7cae7d35ab31ad9fd4cc344ac773a81c9dcef175a97d618dc94c95121a99433e.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/7cae7d35ab31ad9fd4cc344ac773a81c9dcef175a97d618dc94c95121a99433e.jpg)

![7e3f795571ad015f135722ab5033fc536a0c703555f234dc0f8ab1822141ff28.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/7e3f795571ad015f135722ab5033fc536a0c703555f234dc0f8ab1822141ff28.jpg)

![814b6f69fab5a95b2a37e2d454dc14a48c15ce13c68a8dbbfcc84a6aed1feb52.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/814b6f69fab5a95b2a37e2d454dc14a48c15ce13c68a8dbbfcc84a6aed1feb52.jpg)

![8c105281d6b6f7485f1875585e8d4f546d3b099811ad0b3ec72040719ee68bb1.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/8c105281d6b6f7485f1875585e8d4f546d3b099811ad0b3ec72040719ee68bb1.jpg)

![a2e0c96cefd26bd1e2dfd97a598c3889de558d63b4c021443547a631858931bc.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/a2e0c96cefd26bd1e2dfd97a598c3889de558d63b4c021443547a631858931bc.jpg)

![a7bd1a594b600df433b235b0566a847dd255f54f0b8e4dc46638a1e7aeff9899.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/a7bd1a594b600df433b235b0566a847dd255f54f0b8e4dc46638a1e7aeff9899.jpg)

![af5c108ca905224cb84881f7ee2525af95650bdf735f459ff6fa384b67c1a88c.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/af5c108ca905224cb84881f7ee2525af95650bdf735f459ff6fa384b67c1a88c.jpg)

![bc28e6479e9a033f34eeda5dfc73ccea8cd9a3ec6e2ae8650a449bb5ccaaa2ca.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/bc28e6479e9a033f34eeda5dfc73ccea8cd9a3ec6e2ae8650a449bb5ccaaa2ca.jpg)

![c95402ffa08cc1893c4e8ae69934b7ce29b869612e64aea4725b08e100dcb077.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/c95402ffa08cc1893c4e8ae69934b7ce29b869612e64aea4725b08e100dcb077.jpg)

![cb5d08d3a22197629a5352cdc253d3f448032c5bfa3e96db807f68476f6d4456.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/cb5d08d3a22197629a5352cdc253d3f448032c5bfa3e96db807f68476f6d4456.jpg)

![d440506dfce84fa8ab1d9811fda1ca6427c0b5171b4a2a89965b2b8fe9bcd3f8.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/d440506dfce84fa8ab1d9811fda1ca6427c0b5171b4a2a89965b2b8fe9bcd3f8.jpg)

![d745043db51f5ee0cf2bfcf82871fdf48196e52d1c1fc8b463549d6f2103774a.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/d745043db51f5ee0cf2bfcf82871fdf48196e52d1c1fc8b463549d6f2103774a.jpg)

![df53bacdaced22fd28afafc47a4b283696659844a9ae957b5fabb997820878bf.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/df53bacdaced22fd28afafc47a4b283696659844a9ae957b5fabb997820878bf.jpg)

![f878de9a81b3bf60c2b5c5317704ecf5a390756fc159d54a2940f816a4a51a56.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/images/f878de9a81b3bf60c2b5c5317704ecf5a390756fc159d54a2940f816a4a51a56.jpg)

### Tables

![5fc71eb491f24aa2648401cb2b8a3cdf49ae9bc24c525581860a414b2be45ca8.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/5fc71eb491f24aa2648401cb2b8a3cdf49ae9bc24c525581860a414b2be45ca8.jpg)

![794cc43c2e718a0f69b890ca0258fe4f58424c27bbc242a26b830bca77db0c3b.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/794cc43c2e718a0f69b890ca0258fe4f58424c27bbc242a26b830bca77db0c3b.jpg)

![8a56c24b3b7ba78622e47287b3a126c952f855542548e4c035e6a0f68a1a131f.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/8a56c24b3b7ba78622e47287b3a126c952f855542548e4c035e6a0f68a1a131f.jpg)

![9293cd9242cfd87c8162e571378c82520739f58cc9bd8e5b56d300bc95baaa1d.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/9293cd9242cfd87c8162e571378c82520739f58cc9bd8e5b56d300bc95baaa1d.jpg)

![aa052f3071e719ab4db0e0fbd8ce1ae83f7971d41601821bab469711c2157167.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/aa052f3071e719ab4db0e0fbd8ce1ae83f7971d41601821bab469711c2157167.jpg)

![b2d493fe456a3d42a0f1178c9e539b33791ec6ae8b79ea638759f057bbbf3709.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/b2d493fe456a3d42a0f1178c9e539b33791ec6ae8b79ea638759f057bbbf3709.jpg)

![d75b01c91735cc4346c3e627207d3c2961db1e2971f439ce1c636d1245169deb.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/d75b01c91735cc4346c3e627207d3c2961db1e2971f439ce1c636d1245169deb.jpg)

![f099a1f599284a6bc04c1d15d523ef5500f183bbf0a7bb72f65589c09fc56edf.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/f099a1f599284a6bc04c1d15d523ef5500f183bbf0a7bb72f65589c09fc56edf.jpg)

![fe9c4ec03a575aa28cad577462b083aa6cac18d220fd139ebcdb7244d9309296.jpg](../icml_results/1239_ANTiC_%20Adaptive%20Neural%20Temporal%20In%20Situ%20Compressor/tables/fe9c4ec03a575aa28cad577462b083aa6cac18d220fd139ebcdb7244d9309296.jpg)

<a id="paper-1221"></a>

## An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants

### Images

![072e3749c6765754aa9f1f6188405596c868a8ffec2fb92e156250d0f5b15355.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/072e3749c6765754aa9f1f6188405596c868a8ffec2fb92e156250d0f5b15355.jpg)

![0b33195ac6e79a36c3f0d43444f07e57df1146a372e2a7fad1fbc5775757d064.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/0b33195ac6e79a36c3f0d43444f07e57df1146a372e2a7fad1fbc5775757d064.jpg)

![0c17e6517dd9d3ebefb35d85fb43ab328f003f939692d0d71f87865fc1a0fdf3.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/0c17e6517dd9d3ebefb35d85fb43ab328f003f939692d0d71f87865fc1a0fdf3.jpg)

![12a91603ce0598e0a9148fbb535b881b7692b5898ddee79f6060e07319f05238.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/12a91603ce0598e0a9148fbb535b881b7692b5898ddee79f6060e07319f05238.jpg)

![25d2762753241874a9f2b09471bd04f48216aea7d603ffa3ca792da2570651f7.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/25d2762753241874a9f2b09471bd04f48216aea7d603ffa3ca792da2570651f7.jpg)

![26c39e45a1bbf54a9448cd924bbdcef880dcf46397a0f77d3f30714823fd1ed0.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/26c39e45a1bbf54a9448cd924bbdcef880dcf46397a0f77d3f30714823fd1ed0.jpg)

![271f4c94b3f18e45768499516775f36072bb6e82140183c5028930e39f3a8e61.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/271f4c94b3f18e45768499516775f36072bb6e82140183c5028930e39f3a8e61.jpg)

![2892808b6c2bd73b405d05438e25bf0780a3e70c41efad424b0a142c3ddc06f3.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/2892808b6c2bd73b405d05438e25bf0780a3e70c41efad424b0a142c3ddc06f3.jpg)

![3920fbe5e3c485b1a134ad012199d88e0c5aeccb693700c0da833e4288518bb3.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/3920fbe5e3c485b1a134ad012199d88e0c5aeccb693700c0da833e4288518bb3.jpg)

![852b51248b87fc565ab1d8679e13ee565689dd7b6fecb120cc6c3535e3a6b3f2.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/852b51248b87fc565ab1d8679e13ee565689dd7b6fecb120cc6c3535e3a6b3f2.jpg)

![be4aff326419a27e1a0ba4d668bab5912b5b17ad2f9cd7ffd30ad98108e151df.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/be4aff326419a27e1a0ba4d668bab5912b5b17ad2f9cd7ffd30ad98108e151df.jpg)

![c6d2d163b4467f6f9833b6502453c9f9d901fd1de7e7aac48c8b98305b3aef1c.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/c6d2d163b4467f6f9833b6502453c9f9d901fd1de7e7aac48c8b98305b3aef1c.jpg)

![c7357ad468b1fb70a4a93f0b006d608449d91c711a8835167d05533da07005de.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/c7357ad468b1fb70a4a93f0b006d608449d91c711a8835167d05533da07005de.jpg)

![ca38477cf0960a1004b1dd2a36de2aab788730f716f92af5ba37956c8abbca6c.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/ca38477cf0960a1004b1dd2a36de2aab788730f716f92af5ba37956c8abbca6c.jpg)

![f8f6efe9450d61fd9fa964afa92e23cfb5a8de2018fff44d09be1d82d1029ff2.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/images/f8f6efe9450d61fd9fa964afa92e23cfb5a8de2018fff44d09be1d82d1029ff2.jpg)

### Tables

![40c5d9f45feb8ff9038847b537ae4f309d09fbe1f21e4df25e2737e013dfec11.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/40c5d9f45feb8ff9038847b537ae4f309d09fbe1f21e4df25e2737e013dfec11.jpg)

![4541ac4058bd5ebc88d896a41d5e3cec938dc5dcad3300122aabf39628141e0b.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/4541ac4058bd5ebc88d896a41d5e3cec938dc5dcad3300122aabf39628141e0b.jpg)

![4cbb81b0645e88b57ae9cbc54c05b2a2ff02b38dd4832a0350424cbf60f0a943.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/4cbb81b0645e88b57ae9cbc54c05b2a2ff02b38dd4832a0350424cbf60f0a943.jpg)

![687bb1310b7ec7406143bbe89378da6759c920b913a7fcb9071632dbc0e7187b.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/687bb1310b7ec7406143bbe89378da6759c920b913a7fcb9071632dbc0e7187b.jpg)

![97541ee699d867536d5f8448c89bd7467e2e60bd408d3052cac0bb2cc2166bc3.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/97541ee699d867536d5f8448c89bd7467e2e60bd408d3052cac0bb2cc2166bc3.jpg)

![9ddaacd0810db3d35e686aa896e823f85547574a8082b5ec5dd02b9d8cddd083.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/9ddaacd0810db3d35e686aa896e823f85547574a8082b5ec5dd02b9d8cddd083.jpg)

![aca8b03d8cae7970c922116951c17593ba7739681b45d1f9632a19aea16c6e80.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/aca8b03d8cae7970c922116951c17593ba7739681b45d1f9632a19aea16c6e80.jpg)

![b4ab233fa1715e0f46623c78163f4f4fea4fe0e33b4627848d6138d5fab7df8e.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/b4ab233fa1715e0f46623c78163f4f4fea4fe0e33b4627848d6138d5fab7df8e.jpg)

![be064ec3f2dea2dfcfe6ffd9ec6f497f1db6c0ef069c7319a3316c23fde3f3a2.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/be064ec3f2dea2dfcfe6ffd9ec6f497f1db6c0ef069c7319a3316c23fde3f3a2.jpg)

![c36cf8a9a10ea8fab0da357ab94ea172869a0366d3aa6ba4ed703c142e773877.jpg](../icml_results/1240_An%20Exploration%20of%20Non-Euclidean%20Gradient%20Descent_%20Muon%20and%20its%20Many%20Variants/tables/c36cf8a9a10ea8fab0da357ab94ea172869a0366d3aa6ba4ed703c142e773877.jpg)
