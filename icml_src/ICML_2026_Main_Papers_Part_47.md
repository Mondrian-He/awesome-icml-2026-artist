# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 47 of 197

## 目录 (Table of Contents)

1. [Bridging RGB and RAW: Single-step Deterministic Flow with Homogeneous Representation Alignment](#paper-1519)
2. [See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model](#paper-1520)
3. [Lookahead-GCG: Improving Universal Multi-Model Optimization-Based Jailbreaking Attacks via Stochastic Nesterov Optimization](#paper-1521)
4. [Self-correcting for Debiasing Large Language Models](#paper-1522)
5. [Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling](#paper-1523)
6. [Demystifying Action Space Design for Robotic Manipulation Policies](#paper-1524)
7. [Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](#paper-1525)
8. [Mantis: Lightweight Foundation Model for Time Series Classification](#paper-1526)
9. [Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling](#paper-1527)
10. [Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings](#paper-1528)
11. [How Good is Post-Hoc Watermarking With Language Model Rephrasing?](#paper-1529)
12. [Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](#paper-1530)
13. [CLASP: Online learning algorithms for Convex Losses And Squared Penalties](#paper-1531)
14. [Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution](#paper-1532)
15. [Evaluating Relational Reasoning in LLMs with REL](#paper-1533)
16. [Unitary Convolutions for Message-passing and Positional Encodings on Directed Graphs](#paper-1534)
17. [RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections](#paper-1535)
18. [Learning to Share: Selective Memory for Efficient Parallel Agentic Systems](#paper-1536)
19. [Understanding Truncated Positional Encodings for Graph Neural Networks](#paper-1537)
20. [On the Anisotropy of Score-Based Generative Models](#paper-1538)
21. [GenCircuit-RL: Reinforcement Learning from Hierarchical Verification for Genetic Circuit Design](#paper-1539)
22. [Posterior Sampling Reinforcement Learning with Gaussian Processes for Continuous Control: Sublinear Regret Bounds for Unbounded State Spaces](#paper-1540)
23. [Is Graph Mixup Beneficial? Investigating Interpolation And Empirical Performance of Graph Mixup Methods](#paper-1541)
24. [Mixing Expertise with Confidence: A Mixture of Experts Framework for Robust Multi-Modal Continual Learning](#paper-1542)
25. [LINGUA: Bridging the Grounding Gap in VideoQA via Typed Memory and Belief-State Reasoning](#paper-1543)
26. [Temporal Difference Calibration in Sequential Tasks: Application to Vision-Language-Action Models](#paper-1544)
27. [Singular Vectors of Attention Heads Align with Features](#paper-1545)
28. [Complexity Bounds for Dirichlet Process Slice Samplers](#paper-1546)
29. [Nonparametric LLM Evaluation from Preference Data](#paper-1547)
30. [Symbol-Equivariant Recurrent Reasoning Models](#paper-1548)
31. [Learning Hamiltonian Dynamics at Scale: A Differential-Geometric Approach](#paper-1549)
32. [Toward Scalable and Valid Conditional Independence Testing with Spectral Representations](#paper-1550)
33. [Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling](#paper-1551)

---


<a id="paper-1519"></a>

## Bridging RGB and RAW: Single-step Deterministic Flow with Homogeneous Representation Alignment

### Images

![239854c65d37d3af7c7049debfa43d0a9aeb2f94004bd9970075bb36c76d7ff6.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/239854c65d37d3af7c7049debfa43d0a9aeb2f94004bd9970075bb36c76d7ff6.jpg)

![36bd97427d87e8d4e332412d765e8684750be4a5208dd84770abf7977f0af105.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/36bd97427d87e8d4e332412d765e8684750be4a5208dd84770abf7977f0af105.jpg)

![59b2aa2946935837326857da2859201e582f29db7e5aca796aee767fe793c2ad.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/59b2aa2946935837326857da2859201e582f29db7e5aca796aee767fe793c2ad.jpg)

![6188b2d9dd9e1e71393ffadc5e554f5d6b7b285cb0da5edf6287b94fc983c9ef.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/6188b2d9dd9e1e71393ffadc5e554f5d6b7b285cb0da5edf6287b94fc983c9ef.jpg)

![63218f480d07b9027f18ad1216347388274c2fd1f3a1c82342d630840651411e.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/63218f480d07b9027f18ad1216347388274c2fd1f3a1c82342d630840651411e.jpg)

![aefead4b25640a8856a5cc877bc446dd93739d5a737a1badbdfc4b93e0c3d17c.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/aefead4b25640a8856a5cc877bc446dd93739d5a737a1badbdfc4b93e0c3d17c.jpg)

![f29f7c1b3eabfc0e22221f32403f99ed7b3c987fc84bfde73e9512290982f429.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/f29f7c1b3eabfc0e22221f32403f99ed7b3c987fc84bfde73e9512290982f429.jpg)

![ff7421c3e2f82a249a50d73d1e51b14b18efd44d59d8aaf59047d759c8fe34bf.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/ff7421c3e2f82a249a50d73d1e51b14b18efd44d59d8aaf59047d759c8fe34bf.jpg)

### Tables

![160d5e83dce15177eb871f5672280c3a6f072c5c3036d4f4ab6d336701b17f36.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/160d5e83dce15177eb871f5672280c3a6f072c5c3036d4f4ab6d336701b17f36.jpg)

![1e8127fc8a877c9547c3ecccd39d212c0c5b3ff7025e7c76566f8190210110b6.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/1e8127fc8a877c9547c3ecccd39d212c0c5b3ff7025e7c76566f8190210110b6.jpg)

![31aa27d29d8f138d44df84aaa8f335f94beae87115d46fe592ec1f15374105e5.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/31aa27d29d8f138d44df84aaa8f335f94beae87115d46fe592ec1f15374105e5.jpg)

![687bc00a6150fed1dbdde9478940a2643bbb6de5d1086c23eb1a427345eacc6c.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/687bc00a6150fed1dbdde9478940a2643bbb6de5d1086c23eb1a427345eacc6c.jpg)

![99746b701a057ae3f7530a247bc9088677112e2460c83071df648710733957d7.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/99746b701a057ae3f7530a247bc9088677112e2460c83071df648710733957d7.jpg)

![b4af1dafa76bd305956401b6a1d6cfdae71a4bca5b0bd609f02a4409b5af912e.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/b4af1dafa76bd305956401b6a1d6cfdae71a4bca5b0bd609f02a4409b5af912e.jpg)

![f42e4783ed6c6575de53ead3e3cd0fe430ae0a2f1051efc729db0de622b629d5.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/f42e4783ed6c6575de53ead3e3cd0fe430ae0a2f1051efc729db0de622b629d5.jpg)

## Bridging RGB and RAW: Single-step Deterministic Flow with Homogeneous Representation Alignment


### Images

![2926784c2c2c164132e01be8b2a7a473620bf34dad9c613e6d34617115763a46.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/2926784c2c2c164132e01be8b2a7a473620bf34dad9c613e6d34617115763a46.jpg)

![4e085febf938f49d0e17aece846c65782bd9df3a6c48c92fa1f364079aeca2d6.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/4e085febf938f49d0e17aece846c65782bd9df3a6c48c92fa1f364079aeca2d6.jpg)

![66782f42cfe5d2387ed9059cd2d29be15e2275bf7323f0b9f9b4974d579383ec.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/66782f42cfe5d2387ed9059cd2d29be15e2275bf7323f0b9f9b4974d579383ec.jpg)

![68624c852802f79ad400d65ec8899d1dbd0d5cd1b669d5a9a6f81f3a490f0676.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/68624c852802f79ad400d65ec8899d1dbd0d5cd1b669d5a9a6f81f3a490f0676.jpg)

![6a1878dbd704d6e0ced84c59d7954f4f546ba627ded56d2d5ec6a14a5ab81cef.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/6a1878dbd704d6e0ced84c59d7954f4f546ba627ded56d2d5ec6a14a5ab81cef.jpg)

![7c026aac2e5815fff67c100727d95e0f122ea338e3401a30f7082cefd6c57d8c.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/7c026aac2e5815fff67c100727d95e0f122ea338e3401a30f7082cefd6c57d8c.jpg)

![9638fe44f0eb4e4e7d12adc5509f6ff33b618ec907d48ee0049ed1995aedff88.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/9638fe44f0eb4e4e7d12adc5509f6ff33b618ec907d48ee0049ed1995aedff88.jpg)

![a08e5dbaff1346e519f89d9a0c6f20fc832391f81ad2441eb19cc919c475090b.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/a08e5dbaff1346e519f89d9a0c6f20fc832391f81ad2441eb19cc919c475090b.jpg)

![b96cb4f31caeec07f19eb3e23583fbe7059207ca61d4f4d6fc623f1e13f96bed.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/b96cb4f31caeec07f19eb3e23583fbe7059207ca61d4f4d6fc623f1e13f96bed.jpg)

![c2284c99dceb35c5ce34e1b445ebf2df4e4f9bb88125e8f81fd7be4fbc4940dd.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/c2284c99dceb35c5ce34e1b445ebf2df4e4f9bb88125e8f81fd7be4fbc4940dd.jpg)

![cfca7a4c99ad9ee2103fb5748449afa0153100205a2fcd55a5cb1537b940a934.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/cfca7a4c99ad9ee2103fb5748449afa0153100205a2fcd55a5cb1537b940a934.jpg)

![d4235f323f6a9492c016ef6e1dc1de0e9c1e4b5d2b2f423feff632eb1b1e67db.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/d4235f323f6a9492c016ef6e1dc1de0e9c1e4b5d2b2f423feff632eb1b1e67db.jpg)

![da1ae96fdec3968610a952ee668202e3298c1a8ed596b7ba27f3b12288988243.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/da1ae96fdec3968610a952ee668202e3298c1a8ed596b7ba27f3b12288988243.jpg)

![e8c3ef324569b2ab19f3935b2abf5bc8784a02f07012a11560ac78439616f0f2.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/e8c3ef324569b2ab19f3935b2abf5bc8784a02f07012a11560ac78439616f0f2.jpg)

![eb6858f48e7e801883bf4f9e7b73afb3d27ffcd4a241c61ad47c0215a99cac1f.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/eb6858f48e7e801883bf4f9e7b73afb3d27ffcd4a241c61ad47c0215a99cac1f.jpg)

![efbded45556e763f0b6a15178a955f20ecb930a8f7ed3e30130b462510f60063.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/images/efbded45556e763f0b6a15178a955f20ecb930a8f7ed3e30130b462510f60063.jpg)

### Tables

![4f83478205cc0e80f150a3f522f5a10d89a3d3bcffde957d73eba444341fd088.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/4f83478205cc0e80f150a3f522f5a10d89a3d3bcffde957d73eba444341fd088.jpg)

![76d2e5ea916631bc42b634620c24a2fe805fa58211cf73e1e20a76b4cedb8dad.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/76d2e5ea916631bc42b634620c24a2fe805fa58211cf73e1e20a76b4cedb8dad.jpg)

![c26b080f12333b21cab78970a2922237bd0bf7bbb614230cecd07cb168047d6d.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/c26b080f12333b21cab78970a2922237bd0bf7bbb614230cecd07cb168047d6d.jpg)

![cd4230d54dd39ecc4f47295475b4e8a12d41c8c4edca2f38c3b789c5beb9a48e.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/cd4230d54dd39ecc4f47295475b4e8a12d41c8c4edca2f38c3b789c5beb9a48e.jpg)

![cd5911211d40d2828a5880e8949cd77bf007e0d3622fcc57fee1654f30eea2ec.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/cd5911211d40d2828a5880e8949cd77bf007e0d3622fcc57fee1654f30eea2ec.jpg)

![d4ea005cc00b7696eb75a6b586695da2a48cf1fe036ec51f1295d9374524311c.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/d4ea005cc00b7696eb75a6b586695da2a48cf1fe036ec51f1295d9374524311c.jpg)

![f2ed2a098e9f305eb92d3ee810aee82114bab288adca50ca96803cc356aec73e.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/f2ed2a098e9f305eb92d3ee810aee82114bab288adca50ca96803cc356aec73e.jpg)

![f421f34b47c73a4cc7a06fe0a7ba933d5bfcc7a5cf869751f8b78d119a7b150f.jpg](../icml_results/1540_Bridging%20RGB%20and%20RAW_%20Single-step%20Deterministic%20Flow%20with%20Homogeneous%20Representation%20Alignment/tables/f421f34b47c73a4cc7a06fe0a7ba933d5bfcc7a5cf869751f8b78d119a7b150f.jpg)

<a id="paper-1520"></a>

## See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model


### Images

![1951d989af9d64bc73d8912d034900e9598016cbb58a01ed10d2d3155381ddda.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/1951d989af9d64bc73d8912d034900e9598016cbb58a01ed10d2d3155381ddda.jpg)

![1d03088cca80c137f1d5bf23e98dbff38b5f65e96efe4fe1bed0d114c21b4cf9.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/1d03088cca80c137f1d5bf23e98dbff38b5f65e96efe4fe1bed0d114c21b4cf9.jpg)

![2f9cb65772a62c3e2a8ef749860dd7317d90e2aec5deda31ea28767809c68ffc.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/2f9cb65772a62c3e2a8ef749860dd7317d90e2aec5deda31ea28767809c68ffc.jpg)

![3e1e47523497810bcee4d0dc5aa4cfb557f77a279459d76dd23cc27c668e5361.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/3e1e47523497810bcee4d0dc5aa4cfb557f77a279459d76dd23cc27c668e5361.jpg)

![528bd2fee9fee91b71a1a1db72189f714a6a20c2e9924e66202afc2240e21ac8.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/528bd2fee9fee91b71a1a1db72189f714a6a20c2e9924e66202afc2240e21ac8.jpg)

![534fd5a1534bb974852fda5787a787ca211e1445465c52d18c164fbb2ea04c78.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/534fd5a1534bb974852fda5787a787ca211e1445465c52d18c164fbb2ea04c78.jpg)

![67e20f809f1435e9737cc936cd978f9a01428af0266c101fc6d071560113b2c2.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/67e20f809f1435e9737cc936cd978f9a01428af0266c101fc6d071560113b2c2.jpg)

![81b0e00adb19fdc0e00323ae4e8e1af4515cb26a59d047d349d8a21ebdcc9476.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/81b0e00adb19fdc0e00323ae4e8e1af4515cb26a59d047d349d8a21ebdcc9476.jpg)

![a37437b1051d317a449f2af8426bb16aea77f0e66cdb8cda56c62fbc05a1942a.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/a37437b1051d317a449f2af8426bb16aea77f0e66cdb8cda56c62fbc05a1942a.jpg)

![b242adaf468a812a2def7f977a568102fdb75006560b220dea7ad6bfd10b1d17.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/b242adaf468a812a2def7f977a568102fdb75006560b220dea7ad6bfd10b1d17.jpg)

![de4f0ffd93cec19216997a07f4731a6f54446771c3ae2d1428c31e317f9170ac.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/de4f0ffd93cec19216997a07f4731a6f54446771c3ae2d1428c31e317f9170ac.jpg)

![fc86eb987ac55845637212709376eaa26f6a38bfbfb3f9037bf2e55e1d10dfce.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/images/fc86eb987ac55845637212709376eaa26f6a38bfbfb3f9037bf2e55e1d10dfce.jpg)

### Tables

![0710be4eb802fe259aa25ec31e93b4f632ed925ca5951bb4aca6286d13a12b8f.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/0710be4eb802fe259aa25ec31e93b4f632ed925ca5951bb4aca6286d13a12b8f.jpg)

![35f1235542126d947f437427daf46d30151db79d1bb60f0bbba5c83b44aca7f6.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/35f1235542126d947f437427daf46d30151db79d1bb60f0bbba5c83b44aca7f6.jpg)

![4912081efc5cde47a2e5ebfb78784313c824bc3c2d86e44dbff581e63cd927ee.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/4912081efc5cde47a2e5ebfb78784313c824bc3c2d86e44dbff581e63cd927ee.jpg)

![504cc968e426850786508e1a4373b94c335d1ee46d95c69cc88b43c2abae8b2f.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/504cc968e426850786508e1a4373b94c335d1ee46d95c69cc88b43c2abae8b2f.jpg)

![5de021855d5911729083a6fca39225db3e13cf7e2fd86d4b52c8f181252200ac.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/5de021855d5911729083a6fca39225db3e13cf7e2fd86d4b52c8f181252200ac.jpg)

![800f277c1aa8a3e2fac5bc95fbb82272871c4d4e4ae364583e09c0161e077563.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/800f277c1aa8a3e2fac5bc95fbb82272871c4d4e4ae364583e09c0161e077563.jpg)

![862b37c6080ad8618c35149c858f88e4c80ef27e4ed7b55cec24c6af60e20772.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/862b37c6080ad8618c35149c858f88e4c80ef27e4ed7b55cec24c6af60e20772.jpg)

![8cb4d201c49e9ecc82ac9759aadaf4e301ac8e26eb510c377d5b7a162665d1f6.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/8cb4d201c49e9ecc82ac9759aadaf4e301ac8e26eb510c377d5b7a162665d1f6.jpg)

![9aee4670c11339834b5993b519730cdb911f974e58fdc7eca4534a0457a5e88e.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/9aee4670c11339834b5993b519730cdb911f974e58fdc7eca4534a0457a5e88e.jpg)

![a6d0e3686ea33c28961376262648aa721c00173154cf72f217d638816cab5649.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/a6d0e3686ea33c28961376262648aa721c00173154cf72f217d638816cab5649.jpg)

![a762d3c185bbd0658bf1c8b73683a7591936157ee0be456fab02f62f42fcd43b.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/a762d3c185bbd0658bf1c8b73683a7591936157ee0be456fab02f62f42fcd43b.jpg)

![aba8935cabaf4313d06135fc877ecce6708f4bc1d4cee9a1906456bb0f220c14.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/aba8935cabaf4313d06135fc877ecce6708f4bc1d4cee9a1906456bb0f220c14.jpg)

![af95c4e8a022e3b380eb141615cb1fbc12124a98acd806395b52180a220ffe78.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/af95c4e8a022e3b380eb141615cb1fbc12124a98acd806395b52180a220ffe78.jpg)

![b792810365deefbb23d44ba56babcc1188d208cc06a7ff74e55a750677ed1a69.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/b792810365deefbb23d44ba56babcc1188d208cc06a7ff74e55a750677ed1a69.jpg)

![b7d4775f6e9bfc577dd0e85aaad9fed77a9d833979e7478c5fe4848a4647cd10.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/b7d4775f6e9bfc577dd0e85aaad9fed77a9d833979e7478c5fe4848a4647cd10.jpg)

![e8a7d105d132cb049de0692273a5fb6a00deb44bfff926fc4b07f57a032048c2.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/e8a7d105d132cb049de0692273a5fb6a00deb44bfff926fc4b07f57a032048c2.jpg)

![f60a5dc223ff7978026ab1d5eb25ebc6f358238f5492f87a558103484a57c923.jpg](../icml_results/1541_See%20What%20Matters_%20Differentiable%20Grid%20Sample%20Pruning%20for%20Generalizable%20Vision-Language-Action%20Model/tables/f60a5dc223ff7978026ab1d5eb25ebc6f358238f5492f87a558103484a57c923.jpg)

<a id="paper-1521"></a>

## Lookahead-GCG: Improving Universal Multi-Model Optimization-Based Jailbreaking Attacks via Stochastic Nesterov Optimization


### Images

![62d62ad146c9edebc925648e78ced13b389ae825738b41306bc5503fab37030e.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/images/62d62ad146c9edebc925648e78ced13b389ae825738b41306bc5503fab37030e.jpg)

![91a03bb8696328b7d6f65fbdd8847af32b6119fd7019c5b5d270e963bbef8d22.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/images/91a03bb8696328b7d6f65fbdd8847af32b6119fd7019c5b5d270e963bbef8d22.jpg)

![cc5d7662d19d0011a9a6d3cdc2847e422ee4557eb729042fe4eca39b19aaab88.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/images/cc5d7662d19d0011a9a6d3cdc2847e422ee4557eb729042fe4eca39b19aaab88.jpg)

![f856c984485c43e58445ab2264905d52efdebeb73ca6fae6a0fad435df4badc7.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/images/f856c984485c43e58445ab2264905d52efdebeb73ca6fae6a0fad435df4badc7.jpg)

### Tables

![44d049a8edb9173660efa3a6bf5f362348b55876522ca826ca9c7800c3f42766.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/44d049a8edb9173660efa3a6bf5f362348b55876522ca826ca9c7800c3f42766.jpg)

![63b3f5618b93566fb811eb86046e9c677f7bf25593c108160068a45e559b1c28.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/63b3f5618b93566fb811eb86046e9c677f7bf25593c108160068a45e559b1c28.jpg)

![89e37dcd3875c90413d3eec92a04706b061a38319a801ba8f24ad35cac420dac.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/89e37dcd3875c90413d3eec92a04706b061a38319a801ba8f24ad35cac420dac.jpg)

![c0d70ee255e4b0f8fd9507d115f50d31b4aca457cf8d011621f3682f6a96b837.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/c0d70ee255e4b0f8fd9507d115f50d31b4aca457cf8d011621f3682f6a96b837.jpg)

![cc4f7d7cda6c58627e70bb87e033094f731cc67a1d9afc17eb8450852f800563.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/cc4f7d7cda6c58627e70bb87e033094f731cc67a1d9afc17eb8450852f800563.jpg)

![e2085e3b62eeafb143d37d77e8bb30b705cecb0951efb3f794f21d140a8178a0.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/e2085e3b62eeafb143d37d77e8bb30b705cecb0951efb3f794f21d140a8178a0.jpg)

![f8b63fbce63946c2ead957926ee7a0a71e590604ed36f2dbe388d4d4f82be6e2.jpg](../icml_results/1542_Lookahead-GCG_%20Improving%20Universal%20Multi-Model%20Optimization-Based%20Jailbreaking%20Attacks%20via%20Stochasti/tables/f8b63fbce63946c2ead957926ee7a0a71e590604ed36f2dbe388d4d4f82be6e2.jpg)

<a id="paper-1522"></a>

## Self-correcting for Debiasing Large Language Models


### Images

![00ca335333c113585e307fb7595c6a8e5ebb751c9d015089f3bcb5f21ebbdef4.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/00ca335333c113585e307fb7595c6a8e5ebb751c9d015089f3bcb5f21ebbdef4.jpg)

![6db1cd7832b7751141d95017161ef7cd266e492842bcd62d84667aeb9ced2d66.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/6db1cd7832b7751141d95017161ef7cd266e492842bcd62d84667aeb9ced2d66.jpg)

![8ef6e465c4c8c1cb3a8506c2029fbd7ac65ce32223bd82bc108401f5e270baed.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/8ef6e465c4c8c1cb3a8506c2029fbd7ac65ce32223bd82bc108401f5e270baed.jpg)

![a271fc7b5098dc592e110668ebf12215a8a9dd275869e6ae45f9887ed285ae2e.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/a271fc7b5098dc592e110668ebf12215a8a9dd275869e6ae45f9887ed285ae2e.jpg)

![c21a8abd5245e8b9e62b477ebad646f0cf8af8b234157755ae6972c36de10581.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/c21a8abd5245e8b9e62b477ebad646f0cf8af8b234157755ae6972c36de10581.jpg)

![c870b0f7e15f82435e9fb506034054035ed793cd55d1a796dc388400b52f6b1d.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/images/c870b0f7e15f82435e9fb506034054035ed793cd55d1a796dc388400b52f6b1d.jpg)

### Tables

![198bf802636cb7e4c2ef521ba9eaab8833a42c419dfc7e3feda3092ed9933903.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/198bf802636cb7e4c2ef521ba9eaab8833a42c419dfc7e3feda3092ed9933903.jpg)

![287e34815dcd6c360957318350db609c536b0e607af597eb7c3858a01d0b150a.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/287e34815dcd6c360957318350db609c536b0e607af597eb7c3858a01d0b150a.jpg)

![2dbdee3ef7782504bb326cc734fec5139dd6f2f5c81234ef8745b25aaefc5f7a.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/2dbdee3ef7782504bb326cc734fec5139dd6f2f5c81234ef8745b25aaefc5f7a.jpg)

![35ce2ac95c07a320f3d5b8e5d66bbd4a418893fba99d47c3ea89bea685128245.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/35ce2ac95c07a320f3d5b8e5d66bbd4a418893fba99d47c3ea89bea685128245.jpg)

![5bb2ee15a60b8dd83bc4609607ded89be5f1ba2bd1082536e212d0deb4d75566.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/5bb2ee15a60b8dd83bc4609607ded89be5f1ba2bd1082536e212d0deb4d75566.jpg)

![65bd64e2bef86c690a52d40501e9075a0634154eaeb46d5a02b1fc2d2890d3c7.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/65bd64e2bef86c690a52d40501e9075a0634154eaeb46d5a02b1fc2d2890d3c7.jpg)

![84f7cb2052fa830cf4e26fdf9a994408b10bc3034413eb1e7a9ef44385b98113.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/84f7cb2052fa830cf4e26fdf9a994408b10bc3034413eb1e7a9ef44385b98113.jpg)

![d3aae3fb26a9733e1d1a106390b09e59ba5d735e81a7faa22e8d74398fef6ba7.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/d3aae3fb26a9733e1d1a106390b09e59ba5d735e81a7faa22e8d74398fef6ba7.jpg)

![de4633b394417b341692490c243a9ce0bcf3e22a36cc031651e90f45eae2831e.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/de4633b394417b341692490c243a9ce0bcf3e22a36cc031651e90f45eae2831e.jpg)

![fa72e880e9f5b7f01e76bf2415d89549910528a12418a4f1e48dd8a453d88996.jpg](../icml_results/1543_Self-correcting%20for%20Debiasing%20Large%20Language%20Models/tables/fa72e880e9f5b7f01e76bf2415d89549910528a12418a4f1e48dd8a453d88996.jpg)

<a id="paper-1523"></a>

## Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling


### Images

![0818c099ff6445a073fd5863e68ff4b62e2ff961d9d66408d2964672b979e236.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/0818c099ff6445a073fd5863e68ff4b62e2ff961d9d66408d2964672b979e236.jpg)

![18e2cb90fc0e5777e4c8a0920282a90404fc3feaacf79fa1265d72ec77e6cfcd.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/18e2cb90fc0e5777e4c8a0920282a90404fc3feaacf79fa1265d72ec77e6cfcd.jpg)

![2332dd98a1381ea806b8a3f3e84cfaf3785c00d6c853913b771201dd683938d1.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/2332dd98a1381ea806b8a3f3e84cfaf3785c00d6c853913b771201dd683938d1.jpg)

![38665fbd19fb92c418b2dac07ee47cae9cce8465926da2a46cebd0863205a9c9.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/38665fbd19fb92c418b2dac07ee47cae9cce8465926da2a46cebd0863205a9c9.jpg)

![43b874b693ff12b8d654f848fb1c12153f0e2f6bf327d9167cef31bd7182fcbc.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/43b874b693ff12b8d654f848fb1c12153f0e2f6bf327d9167cef31bd7182fcbc.jpg)

![694834c8dfe75047d97b791c67de8fb201c4dc26ddfcd8979799ff12f7b97de5.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/694834c8dfe75047d97b791c67de8fb201c4dc26ddfcd8979799ff12f7b97de5.jpg)

![757db39f58f31f4a672bd1118cb582b7115d9a63fd697d68b208c4c19d8a479e.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/757db39f58f31f4a672bd1118cb582b7115d9a63fd697d68b208c4c19d8a479e.jpg)

![7c012e40e9218c18843320df29da6cd87c190434fd780b76e7603d0e10605a71.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/7c012e40e9218c18843320df29da6cd87c190434fd780b76e7603d0e10605a71.jpg)

![7fee3c04a42164f5d37d91739bfbfaf1e7000ed891ea1599dd7b280b8aa74564.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/7fee3c04a42164f5d37d91739bfbfaf1e7000ed891ea1599dd7b280b8aa74564.jpg)

![b7baed890a8246595398e8c0460bd701b68dde003294fa63699f45f8848fd787.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/b7baed890a8246595398e8c0460bd701b68dde003294fa63699f45f8848fd787.jpg)

![cc9ee443eff3a53027a036de4994b0ac22676926e898d2a93217905443807ef7.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/cc9ee443eff3a53027a036de4994b0ac22676926e898d2a93217905443807ef7.jpg)

![da8cce08e1e256f430bc276c73d56a65e36d5d1d25c655cbca20e303f92481e3.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/da8cce08e1e256f430bc276c73d56a65e36d5d1d25c655cbca20e303f92481e3.jpg)

![e4067af37df562fe620124602ffb41e1ba48482fdbdee00e3931fa88f8b3fa42.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/images/e4067af37df562fe620124602ffb41e1ba48482fdbdee00e3931fa88f8b3fa42.jpg)

### Tables

![3607dd228c80947479c5940539ce9e6662575787ee6654a7099a8390b9351e32.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/3607dd228c80947479c5940539ce9e6662575787ee6654a7099a8390b9351e32.jpg)

![4421b52ad3140c80498156d88d1176ef9cd2144723547dcf4aa3f20c302a6d10.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/4421b52ad3140c80498156d88d1176ef9cd2144723547dcf4aa3f20c302a6d10.jpg)

![4adc15db7807c5306401c1d358e3ad2438e8e8c16fcb3faead99ce0cd43b4b06.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/4adc15db7807c5306401c1d358e3ad2438e8e8c16fcb3faead99ce0cd43b4b06.jpg)

![57980f11227868a44a4901510ac573799af61b56dc113d33e79e1ed1ef3ac326.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/57980f11227868a44a4901510ac573799af61b56dc113d33e79e1ed1ef3ac326.jpg)

![5aac45fb1327d69372e344ef50163624e43037e2665467d0b1da0cd634e5d712.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/5aac45fb1327d69372e344ef50163624e43037e2665467d0b1da0cd634e5d712.jpg)

![a113b13e67556aa5cf1bb51e9d8480dee28cf1de00ba9639bd51461bf4609896.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/a113b13e67556aa5cf1bb51e9d8480dee28cf1de00ba9639bd51461bf4609896.jpg)

![a710d57928877f02e9826f038121af587b068f66d4b71807f5cd507b84d54324.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/a710d57928877f02e9826f038121af587b068f66d4b71807f5cd507b84d54324.jpg)

![a86d5a77395f428138be1b44c7b7b866347e70ad51f5fe3676221e666d8c6438.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/a86d5a77395f428138be1b44c7b7b866347e70ad51f5fe3676221e666d8c6438.jpg)

![bbf3a2fda05a06a260a48ad92cdb698345a1e7b03ce8bc1874e7d03a35f73e99.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/bbf3a2fda05a06a260a48ad92cdb698345a1e7b03ce8bc1874e7d03a35f73e99.jpg)

![bea089a697c70651634b933994d25ff5574a7a72e241b41aa409742304882e4b.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/bea089a697c70651634b933994d25ff5574a7a72e241b41aa409742304882e4b.jpg)

![e4f4aa784582eb9b19ffa9b44b840c194f8a30444a51738240a58760b78bfbff.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/e4f4aa784582eb9b19ffa9b44b840c194f8a30444a51738240a58760b78bfbff.jpg)

![e99ac0b6fbf9215b32bb1cb57479a6b3f72ab92edf34e5ea83955ba895119646.jpg](../icml_results/1544_Learning%20Task-Sufficient%20World%20Models%20by%20Synergizing%20Agentic%20Exploration%20and%20Structured%20Modeling/tables/e99ac0b6fbf9215b32bb1cb57479a6b3f72ab92edf34e5ea83955ba895119646.jpg)

<a id="paper-1524"></a>

## Demystifying Action Space Design for Robotic Manipulation Policies


### Images

![028a162e3d7224a852c279d2f09c138f5ba491e17780442fe99ce990a7a788de.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/028a162e3d7224a852c279d2f09c138f5ba491e17780442fe99ce990a7a788de.jpg)

![1fffb3644024c81ee55c028f39414395aff78e09cfd33fa1082f613e160cc702.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/1fffb3644024c81ee55c028f39414395aff78e09cfd33fa1082f613e160cc702.jpg)

![2bc0fecc0d7bbdafb022aca6ef58dc1081d514bddb33bf50a81475ca13d64a88.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/2bc0fecc0d7bbdafb022aca6ef58dc1081d514bddb33bf50a81475ca13d64a88.jpg)

![3226cad48ffe1c6632615609ce95f41b3d2012ab6380508f0a96e66114d032f7.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/3226cad48ffe1c6632615609ce95f41b3d2012ab6380508f0a96e66114d032f7.jpg)

![3d64dbfe93817810cf824b8a1c7110eddf38e0cc8c454e3082ec3d6152fb392d.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/3d64dbfe93817810cf824b8a1c7110eddf38e0cc8c454e3082ec3d6152fb392d.jpg)

![4e7510224d1f2185c8470c103c26c9c8d5c499990dff3e874e02c4ccd591f8e8.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/4e7510224d1f2185c8470c103c26c9c8d5c499990dff3e874e02c4ccd591f8e8.jpg)

![583fe6cd2a05cc1cf99d319b878cf356c550e64d8b7bc172bcffe26f09c96bd8.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/583fe6cd2a05cc1cf99d319b878cf356c550e64d8b7bc172bcffe26f09c96bd8.jpg)

![a20c063fc891023afbe5b1ccba0abc682fed3814008c19d3f6ffec48974c6660.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/a20c063fc891023afbe5b1ccba0abc682fed3814008c19d3f6ffec48974c6660.jpg)

![aa8b42e1f247099127cee59700b5afa8c356ed0e1a73a6be5ea5145701109069.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/aa8b42e1f247099127cee59700b5afa8c356ed0e1a73a6be5ea5145701109069.jpg)

![ad670a3403a506d04d6ee01129ebd02f7ef675b8cecd56b65db3d5bc1502a312.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/ad670a3403a506d04d6ee01129ebd02f7ef675b8cecd56b65db3d5bc1502a312.jpg)

![ae0e70759b14c4007c3cfa9c57d9a8b9aad8f176ac1a8d1d7eb25e203e7287fa.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/ae0e70759b14c4007c3cfa9c57d9a8b9aad8f176ac1a8d1d7eb25e203e7287fa.jpg)

![dc1d53add81f3b9318dcf74d8f1284002582f216b84b9ba0f72e084937b9c539.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/images/dc1d53add81f3b9318dcf74d8f1284002582f216b84b9ba0f72e084937b9c539.jpg)

### Tables

![44db8eda35d6031ebb724fe97c8d2b843edc7620515260726586e0835bf3f814.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/44db8eda35d6031ebb724fe97c8d2b843edc7620515260726586e0835bf3f814.jpg)

![45880860e1ca68cbdb190826b5a2481b434048f11466f8a86d1052b740cf9900.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/45880860e1ca68cbdb190826b5a2481b434048f11466f8a86d1052b740cf9900.jpg)

![67eac804338e2f3361a83b9055090704415b9cbaca3c26591d54c03111b7b50c.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/67eac804338e2f3361a83b9055090704415b9cbaca3c26591d54c03111b7b50c.jpg)

![885d0e158df5d9be071d093339ad5b75de7cf162a56e3996c3ff852c82250363.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/885d0e158df5d9be071d093339ad5b75de7cf162a56e3996c3ff852c82250363.jpg)

![ae9fdd38d19d58b2bddf655eeaeb16a19187ca7bcf7ef906795a884163e965d0.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/ae9fdd38d19d58b2bddf655eeaeb16a19187ca7bcf7ef906795a884163e965d0.jpg)

![db92cc9bb11c8328c9c15b9add78011bf818f6697e9a1651822e8f9204f2ccf3.jpg](../icml_results/1545_Demystifying%20Action%20Space%20Design%20for%20Robotic%20Manipulation%20Policies/tables/db92cc9bb11c8328c9c15b9add78011bf818f6697e9a1651822e8f9204f2ccf3.jpg)

<a id="paper-1525"></a>

## Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos


### Images

![12dd7df79a54186989a3abcf2f07d1f827dc8157abe66f6c0cd47f5d2960736d.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/12dd7df79a54186989a3abcf2f07d1f827dc8157abe66f6c0cd47f5d2960736d.jpg)

![1b4f771874cf45f3936797f90afe74e554f2cd50877ddb463fd968ea6c1f4b5e.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/1b4f771874cf45f3936797f90afe74e554f2cd50877ddb463fd968ea6c1f4b5e.jpg)

![23bcd43a25a7973ec88e23252e81f9a5fd035648ab76c31e8f8cc31068a10c91.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/23bcd43a25a7973ec88e23252e81f9a5fd035648ab76c31e8f8cc31068a10c91.jpg)

![2a9e782ac5ed30a48d91af9541b3da59c47b41e30b7419dddc5e759a3d3f4ca9.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/2a9e782ac5ed30a48d91af9541b3da59c47b41e30b7419dddc5e759a3d3f4ca9.jpg)

![309f7a81bab3d24c07d986e7e880819287f8be0758c8423a5153e40404ea08f5.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/309f7a81bab3d24c07d986e7e880819287f8be0758c8423a5153e40404ea08f5.jpg)

![38a0f0015f9cea4454a853038437978993afd801088a9040f1e156dc5f844254.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/38a0f0015f9cea4454a853038437978993afd801088a9040f1e156dc5f844254.jpg)

![4ab1ecc0a9fdd2b5de1bb2c9654f11c7877835c6abeb35169cb45c86501870f2.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/4ab1ecc0a9fdd2b5de1bb2c9654f11c7877835c6abeb35169cb45c86501870f2.jpg)

![52f2fc31deba0b8747f9db9376cf0535fa261858123c6a9b568d5caf2726d668.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/52f2fc31deba0b8747f9db9376cf0535fa261858123c6a9b568d5caf2726d668.jpg)

![73cf3ccf72a71ab09721140ba4c026a206f3a02df9599595cf6631e96b0262c2.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/73cf3ccf72a71ab09721140ba4c026a206f3a02df9599595cf6631e96b0262c2.jpg)

![7adf852aa9d1d6312862676e46a96af4415ad84a81d876287a6593f5da0e3abe.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/7adf852aa9d1d6312862676e46a96af4415ad84a81d876287a6593f5da0e3abe.jpg)

![80667bb548736c99fa9d05007b5f32026ad9ac1cdcbf8a43bef3f8e0c58a68d0.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/80667bb548736c99fa9d05007b5f32026ad9ac1cdcbf8a43bef3f8e0c58a68d0.jpg)

![ccd60858839dfddd0570682c548697118a5b315a2a3965e8dab9d5de7953217a.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/ccd60858839dfddd0570682c548697118a5b315a2a3965e8dab9d5de7953217a.jpg)

![cfabb05f63a74157baee0cc92bfbbb5e279aeb744ed4c0125a858ea12c039851.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/cfabb05f63a74157baee0cc92bfbbb5e279aeb744ed4c0125a858ea12c039851.jpg)

![d09ad0d7a75939be886fab55bbb3039686c33278a2a129a4083bbd92947f4351.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/d09ad0d7a75939be886fab55bbb3039686c33278a2a129a4083bbd92947f4351.jpg)

![e16afc09a0ceeff194982ab951c5aa7c9fbf4a5214c06d22ac18ec7b9a6f57d7.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/e16afc09a0ceeff194982ab951c5aa7c9fbf4a5214c06d22ac18ec7b9a6f57d7.jpg)

![f50d9061e583daae981cea6795d181840ae1ef202aa5c3c324ef0c5a4125a4fa.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/f50d9061e583daae981cea6795d181840ae1ef202aa5c3c324ef0c5a4125a4fa.jpg)

![fa6c32687924a587fd11ffceabd879f01589b45d96d9bac5811d361b501726b0.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/fa6c32687924a587fd11ffceabd879f01589b45d96d9bac5811d361b501726b0.jpg)

![fc4e3257acaacde01786d7f96506999daa8168c0cb5730c509418419ffb0de37.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/images/fc4e3257acaacde01786d7f96506999daa8168c0cb5730c509418419ffb0de37.jpg)

### Tables

![207e39f0c082243b723406d0db989c0da02f3568fe8bcf35209cc023f676bc42.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/207e39f0c082243b723406d0db989c0da02f3568fe8bcf35209cc023f676bc42.jpg)

![25bf2fca6d85a952d6efdb471886bff840e20ef74b77257521d10765589b4a4a.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/25bf2fca6d85a952d6efdb471886bff840e20ef74b77257521d10765589b4a4a.jpg)

![31084957a8be34406af782bb5fb58452008279b4ef0f202817e90188ff6e6f04.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/31084957a8be34406af782bb5fb58452008279b4ef0f202817e90188ff6e6f04.jpg)

![42f9486425a82804b313711a95e955312e19f5777e278744f78ce622916d8cf2.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/42f9486425a82804b313711a95e955312e19f5777e278744f78ce622916d8cf2.jpg)

![74895eedac126855e1a61846e99671e84f0bf747862d517dc63b6b34d8458bae.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/74895eedac126855e1a61846e99671e84f0bf747862d517dc63b6b34d8458bae.jpg)

![78e113497e184053344aea9aa645cc7092c7b3cb5705c5eb6a50f2545cd24698.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/78e113497e184053344aea9aa645cc7092c7b3cb5705c5eb6a50f2545cd24698.jpg)

![a4fe2c58ba6d18a4a6ef22609e51eb3d4dc285c344e1dd6ecd289ebe7e972a19.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/a4fe2c58ba6d18a4a6ef22609e51eb3d4dc285c344e1dd6ecd289ebe7e972a19.jpg)

![b1c9ac23e234226a2cfadebe9471646635b8f56a40b7c961cd7e2a85cbf1673d.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/b1c9ac23e234226a2cfadebe9471646635b8f56a40b7c961cd7e2a85cbf1673d.jpg)

![eb624951c4e59d6e41b1bdd4eb274e55ef2e535e9215ec9c6182a3a2fbdaae1f.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/eb624951c4e59d6e41b1bdd4eb274e55ef2e535e9215ec9c6182a3a2fbdaae1f.jpg)

![f0e8ffb65956ffeb45d6207b55e15a4b492e8c238426b3e26b309e87cd023849.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/f0e8ffb65956ffeb45d6207b55e15a4b492e8c238426b3e26b309e87cd023849.jpg)

![fb747df3850262ad5014b9a7445ede41358c8f2c969709abfd8125dab0e780c5.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/fb747df3850262ad5014b9a7445ede41358c8f2c969709abfd8125dab0e780c5.jpg)

![feaeb12b2874612c6c839e953f5c53bd8b9fce98c17e9c1d396c3053a0b7e374.jpg](../icml_results/1546_Enhancing%20Train-Free%20Infinite-Frame%20Generation%20for%20Consistent%20Long%20Videos/tables/feaeb12b2874612c6c839e953f5c53bd8b9fce98c17e9c1d396c3053a0b7e374.jpg)

<a id="paper-1526"></a>

## Mantis: Lightweight Foundation Model for Time Series Classification


### Images

![0dcddabcbddd2a0e058dc1c9e51209ca20f57e1ace246a55a8e360301ff84589.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/0dcddabcbddd2a0e058dc1c9e51209ca20f57e1ace246a55a8e360301ff84589.jpg)

![1478360396a968f29fe1cd3c6583e7e3ad101f9170402238cbdecf153b4e33b5.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/1478360396a968f29fe1cd3c6583e7e3ad101f9170402238cbdecf153b4e33b5.jpg)

![1b9de9fd9691b47860574300551b110fa5503377e7b968a6443481eb3ec8f694.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/1b9de9fd9691b47860574300551b110fa5503377e7b968a6443481eb3ec8f694.jpg)

![3b1392cfbfb92df46efddd8d7d899e4f1fc5e6e8d0d3fd5d5e176490e39fa2c1.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/3b1392cfbfb92df46efddd8d7d899e4f1fc5e6e8d0d3fd5d5e176490e39fa2c1.jpg)

![4993a3a401eea1013f232b28a8d3a50cf0a666ee87cb336cedcc7b09363b935d.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/4993a3a401eea1013f232b28a8d3a50cf0a666ee87cb336cedcc7b09363b935d.jpg)

![5800e5348aa800b4dceee9ef0693f2a94daa8d0a6254796b3cb769c059ad5d54.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/5800e5348aa800b4dceee9ef0693f2a94daa8d0a6254796b3cb769c059ad5d54.jpg)

![6c37981f73d0295cfd7aafd2e600c92060ab63e56a6a6ca18da70fe1eb38d52e.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/6c37981f73d0295cfd7aafd2e600c92060ab63e56a6a6ca18da70fe1eb38d52e.jpg)

![6dc11704f517b765c691b046e28dbc087c7041f88c01a0277b011c170c26085d.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/6dc11704f517b765c691b046e28dbc087c7041f88c01a0277b011c170c26085d.jpg)

![76a8ecdea09f5e86abd55773fd734649eef6ed27bccd9ef6f477f16c05fed97e.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/76a8ecdea09f5e86abd55773fd734649eef6ed27bccd9ef6f477f16c05fed97e.jpg)

![8b4ca1a9cb28dca15943a69c8bd915f1427ae634da1b3234c221e7d180c3cac2.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/8b4ca1a9cb28dca15943a69c8bd915f1427ae634da1b3234c221e7d180c3cac2.jpg)

![97e9a1cd8f493cdccaa25bae35ec7da918f6d31de1e8de48d17ec205fdebf5f9.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/97e9a1cd8f493cdccaa25bae35ec7da918f6d31de1e8de48d17ec205fdebf5f9.jpg)

![9990314bb0dd5318733a4cff318da116bb4a67c632488823a4240b98edf23eab.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/9990314bb0dd5318733a4cff318da116bb4a67c632488823a4240b98edf23eab.jpg)

![b32a4860c35c5c1d7474ec656aada1641d1dd0b0de95c5559f517eaa96c9d7ff.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/b32a4860c35c5c1d7474ec656aada1641d1dd0b0de95c5559f517eaa96c9d7ff.jpg)

![b508a1ec01cbb4b9e5d9a8e3b866449beee9d1601f00a69bc7fc58b094e1082e.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/b508a1ec01cbb4b9e5d9a8e3b866449beee9d1601f00a69bc7fc58b094e1082e.jpg)

![c94450fd927607ecb319ea83302e3f4f9316dbb042c6c9834372cb340d834abf.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/c94450fd927607ecb319ea83302e3f4f9316dbb042c6c9834372cb340d834abf.jpg)

![cff3a0a6db345e201a864001fbea997e24f24ae1321e1998ca62d839d887757c.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/cff3a0a6db345e201a864001fbea997e24f24ae1321e1998ca62d839d887757c.jpg)

![dfe05e967bcbe797540b890327816f468fc49e9ff367032f322fb4ac7bef4c16.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/dfe05e967bcbe797540b890327816f468fc49e9ff367032f322fb4ac7bef4c16.jpg)

![e57e4895b4327973a1141242979e8d3114f99fa6160942b2b07f6d3cbd967128.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/e57e4895b4327973a1141242979e8d3114f99fa6160942b2b07f6d3cbd967128.jpg)

![ea7881abdac4bb6dad52f0cf9e2d956c88002be7c72b838b39359a0a3ed0c7c2.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/images/ea7881abdac4bb6dad52f0cf9e2d956c88002be7c72b838b39359a0a3ed0c7c2.jpg)

### Tables

![1b0e130034e3c459dc7b5d65e25fde089211c361591b051b42b35f96e7fadfba.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/1b0e130034e3c459dc7b5d65e25fde089211c361591b051b42b35f96e7fadfba.jpg)

![2367c9b28e2f032850706d25ea457996bde4f1cc6647d4fe8274e03851f5410e.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/2367c9b28e2f032850706d25ea457996bde4f1cc6647d4fe8274e03851f5410e.jpg)

![29b02d9934b1e29c19425ee401d9899c0595dc3470b9775b2244b08291c922a3.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/29b02d9934b1e29c19425ee401d9899c0595dc3470b9775b2244b08291c922a3.jpg)

![33dcb574496cd7ae4882097bda4f0c7fd510c70d3d84ad6e0cf37518247a691f.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/33dcb574496cd7ae4882097bda4f0c7fd510c70d3d84ad6e0cf37518247a691f.jpg)

![37136383393d7f1d2f7630d4851002880e1363581b91b2de39202245a868a5cc.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/37136383393d7f1d2f7630d4851002880e1363581b91b2de39202245a868a5cc.jpg)

![45dcb479b037c591ce7bc91cc186144bd10d50b4a743340dacc37c98161385d8.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/45dcb479b037c591ce7bc91cc186144bd10d50b4a743340dacc37c98161385d8.jpg)

![5641e1e979ccf8a69223867fde2b563a0cf8e3759c34939dfadbe0641a8c2865.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/5641e1e979ccf8a69223867fde2b563a0cf8e3759c34939dfadbe0641a8c2865.jpg)

![6e6fa0dcc6320bbc171db5b145809b30aafb761488b2b582c02d86bb27033741.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/6e6fa0dcc6320bbc171db5b145809b30aafb761488b2b582c02d86bb27033741.jpg)

![6ebcaef6dbdcdc258a34b26b08fe290f719c7b216d2322671242d5bd765a9e6c.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/6ebcaef6dbdcdc258a34b26b08fe290f719c7b216d2322671242d5bd765a9e6c.jpg)

![856b784234795f3e40ac32b8b7ffe57967001a8ec43446cf3b176aec0eb5e30d.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/856b784234795f3e40ac32b8b7ffe57967001a8ec43446cf3b176aec0eb5e30d.jpg)

![87c2cf08b252605e5d0e8003302fc52fbf1ce6d21e122a58728186ca2f7e3ff1.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/87c2cf08b252605e5d0e8003302fc52fbf1ce6d21e122a58728186ca2f7e3ff1.jpg)

![897be57d329bca0a78e9456696d4c87ec29641c4e48e818bb93445456a42064f.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/897be57d329bca0a78e9456696d4c87ec29641c4e48e818bb93445456a42064f.jpg)

![93ed4e3542313dae36ba34c8bb32e221ec214ad167fd07f4e4bf8edca8d16020.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/93ed4e3542313dae36ba34c8bb32e221ec214ad167fd07f4e4bf8edca8d16020.jpg)

![9fddce911e661c94810d62a475dbe80ea1b319e6faa4d97a940ccc69701e807d.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/9fddce911e661c94810d62a475dbe80ea1b319e6faa4d97a940ccc69701e807d.jpg)

![abfe18073f21d4f79fb27720e8a924859d9a91b5ad9f3478e10ebbbe6e17a67d.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/abfe18073f21d4f79fb27720e8a924859d9a91b5ad9f3478e10ebbbe6e17a67d.jpg)

![aeb723d09505bf4862f0967a5d70e3b6c51f4c7c14e91760a57cd043b2766933.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/aeb723d09505bf4862f0967a5d70e3b6c51f4c7c14e91760a57cd043b2766933.jpg)

![ba90807e99120484f748ac45ea3ccde5232cff95d5554e9b4be64f103bded830.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/ba90807e99120484f748ac45ea3ccde5232cff95d5554e9b4be64f103bded830.jpg)

![ed2f85e6a7ee08742a909c63a6a0c74a6d6bfb491aa62888c2ce7d1f9b869985.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/ed2f85e6a7ee08742a909c63a6a0c74a6d6bfb491aa62888c2ce7d1f9b869985.jpg)

![f5e9f93a7beb7404fb553b6d1b3a8269e25d9bcc43cdbfe98ab39193e051330f.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/f5e9f93a7beb7404fb553b6d1b3a8269e25d9bcc43cdbfe98ab39193e051330f.jpg)

![f8abc10461a0121a0d8af7c51134393e80a18ab055cae6d4b48e57059ac1b738.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/f8abc10461a0121a0d8af7c51134393e80a18ab055cae6d4b48e57059ac1b738.jpg)

![f8de63d6d8550a6a7da1127c3d0d3005fdb2fcd9447f25c2847a7199844e4747.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/f8de63d6d8550a6a7da1127c3d0d3005fdb2fcd9447f25c2847a7199844e4747.jpg)

![fbdd83edaccaed3718d2fd5847d45780657b03445736f7548e0086733749c8d6.jpg](../icml_results/1547_Mantis_%20Lightweight%20Foundation%20Model%20for%20Time%20Series%20Classification/tables/fbdd83edaccaed3718d2fd5847d45780657b03445736f7548e0086733749c8d6.jpg)

<a id="paper-1527"></a>

## Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling


### Images

![0d6f7abcf8a353a44de42715480f76f7a014b35022eb06d219aaf99f4f28e5c3.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/images/0d6f7abcf8a353a44de42715480f76f7a014b35022eb06d219aaf99f4f28e5c3.jpg)

![4b7914ef82e25005283eb8c1f5945a1cbd939c491f600df9a004c0c1ee26a68e.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/images/4b7914ef82e25005283eb8c1f5945a1cbd939c491f600df9a004c0c1ee26a68e.jpg)

![a13f674242a3db8f9d6b5bf4d7cd5c037e4fc69fb1301e1626a677f2e03c7678.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/images/a13f674242a3db8f9d6b5bf4d7cd5c037e4fc69fb1301e1626a677f2e03c7678.jpg)

![bd9d3f0c660e1cc5f17903f86b8864146be33f9eb2eed92952c61ae84242fc1f.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/images/bd9d3f0c660e1cc5f17903f86b8864146be33f9eb2eed92952c61ae84242fc1f.jpg)

![d0037a76ef8d22332bfe1aeea0795c1b598ad82b19c7aa4b47f5f36545c33015.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/images/d0037a76ef8d22332bfe1aeea0795c1b598ad82b19c7aa4b47f5f36545c33015.jpg)

### Tables

![089263809916b03a3cedebff54a7c4228aedcbbb26486d17ad3220301147a35e.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/089263809916b03a3cedebff54a7c4228aedcbbb26486d17ad3220301147a35e.jpg)

![56b082c97ca8c3c67825bde63fbb1db65aaaf7d8f32c46116bfcc5ee60bdc3ac.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/56b082c97ca8c3c67825bde63fbb1db65aaaf7d8f32c46116bfcc5ee60bdc3ac.jpg)

![5c072622c8acc2144c2a3b9fe5c3d519a74b00318b0b031eae029f69df2432f6.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/5c072622c8acc2144c2a3b9fe5c3d519a74b00318b0b031eae029f69df2432f6.jpg)

![60bbe1ff30913ef36f992156822160bb02b76c7f69aa69f271f90c3dbaec5776.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/60bbe1ff30913ef36f992156822160bb02b76c7f69aa69f271f90c3dbaec5776.jpg)

![741052db39a1b9733d9eec8d1ed0ffce8c36ed266db319dd967279498508db0b.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/741052db39a1b9733d9eec8d1ed0ffce8c36ed266db319dd967279498508db0b.jpg)

![82bc441c37129327db36da01b53d017d399c9ac40b8f1ed7e18a8beffab118e2.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/82bc441c37129327db36da01b53d017d399c9ac40b8f1ed7e18a8beffab118e2.jpg)

![872e83525471d579643ab58d9e3725d0022d689670d70a807d3c30d61aedef8b.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/872e83525471d579643ab58d9e3725d0022d689670d70a807d3c30d61aedef8b.jpg)

![8ec01d8d2cb180e85f94614d04e1492cb1987650dfd969a3c2f3773e20874e7e.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/8ec01d8d2cb180e85f94614d04e1492cb1987650dfd969a3c2f3773e20874e7e.jpg)

![aee4cf150b3868f48accc3c208ca8e4a5e00d782a720a908b6bbcab1dd5392be.jpg](../icml_results/1548_Softsign_%20Smooth%20Sign%20in%20Your%20Optimizer%20For%20Better%20Parameter%20Heterogeneity%20Handling/tables/aee4cf150b3868f48accc3c208ca8e4a5e00d782a720a908b6bbcab1dd5392be.jpg)

<a id="paper-1528"></a>

## Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings


### Images

![16f2ac2ad5ae2b9e503cadc5078a3987159cc5a0f35af0d5aec6ee193c5dd7a3.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/images/16f2ac2ad5ae2b9e503cadc5078a3987159cc5a0f35af0d5aec6ee193c5dd7a3.jpg)

![6ae802e7b1139ff323db84348557e2a368a0ee2f6caa83a4e0d74279520a9717.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/images/6ae802e7b1139ff323db84348557e2a368a0ee2f6caa83a4e0d74279520a9717.jpg)

![822315486a688ea5502884f5c6d0f0adb46fd328388c84fef1e7e07453cc4b0c.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/images/822315486a688ea5502884f5c6d0f0adb46fd328388c84fef1e7e07453cc4b0c.jpg)

![d2f6a37f36d4afc7246107c66d34aa3863019b557c6faafbf37d5633a9bb4fa6.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/images/d2f6a37f36d4afc7246107c66d34aa3863019b557c6faafbf37d5633a9bb4fa6.jpg)

### Tables

![330344a85b2feabcf24b40a6d6943edcec165cdae2b748057fc602c96632f994.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/tables/330344a85b2feabcf24b40a6d6943edcec165cdae2b748057fc602c96632f994.jpg)

![75c40159bb77105e8983dadf10017a21c8ae31b76e434285127e5cc918926e6d.jpg](../icml_results/1549_Adaptive%20Personalized%20Federated%20Learning%20via%20Multi-task%20Averaging%20of%20Kernel%20Mean%20Embeddings/tables/75c40159bb77105e8983dadf10017a21c8ae31b76e434285127e5cc918926e6d.jpg)

<a id="paper-1529"></a>

## How Good is Post-Hoc Watermarking With Language Model Rephrasing?


### Images

![00381d35ac81ca2ae2bcd229de791a71324418dd81fbbebe22447c381f91b5dd.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/00381d35ac81ca2ae2bcd229de791a71324418dd81fbbebe22447c381f91b5dd.jpg)

![00b2e162332005fa9c53d2d22ce01f949c618fc20e96972c06120780d479c308.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/00b2e162332005fa9c53d2d22ce01f949c618fc20e96972c06120780d479c308.jpg)

![1884033151df01e64e1e1b77b8567661d6909e6b91fc18354ee5daccffdc99f2.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/1884033151df01e64e1e1b77b8567661d6909e6b91fc18354ee5daccffdc99f2.jpg)

![38f47a5a7c25cd30e6c51ebfb3c8e9d856fe049f52259b0a6e9c0f752b5a016d.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/38f47a5a7c25cd30e6c51ebfb3c8e9d856fe049f52259b0a6e9c0f752b5a016d.jpg)

![52aca9aa917aa55bdd893c8293eea8fbdec1c574d334f0c7ec6d7ecd5d3d56a9.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/52aca9aa917aa55bdd893c8293eea8fbdec1c574d334f0c7ec6d7ecd5d3d56a9.jpg)

![54bc9219591ee7f48a614823c1298a09f0b31ede2e2acb5f2e2d96c47f9121c8.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/54bc9219591ee7f48a614823c1298a09f0b31ede2e2acb5f2e2d96c47f9121c8.jpg)

![55e2b4144e53354ff2b572e6d53eddfca19b4e845da2de1d406322dfd80cfcd8.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/55e2b4144e53354ff2b572e6d53eddfca19b4e845da2de1d406322dfd80cfcd8.jpg)

![8c9fe5aa41af2765690b695d3c56006aafeb87b6c251a4bb257a7ae242f5cc50.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/8c9fe5aa41af2765690b695d3c56006aafeb87b6c251a4bb257a7ae242f5cc50.jpg)

![91cd97e71fb8639d59cdb451ff10c6829f21f4e1c46515c7f80cf4755e4cf89a.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/91cd97e71fb8639d59cdb451ff10c6829f21f4e1c46515c7f80cf4755e4cf89a.jpg)

![bcf90478934aeb44953cedc032ea8e1fb2bc62972e9fd5cda748c2a0d221bd07.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/bcf90478934aeb44953cedc032ea8e1fb2bc62972e9fd5cda748c2a0d221bd07.jpg)

![c9d0dbd650fc46e125de929d71b05c8e15b599cec932064dec65af834e9ff7c5.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/c9d0dbd650fc46e125de929d71b05c8e15b599cec932064dec65af834e9ff7c5.jpg)

![db7851285f8707b342a7707605e270a10336c5110017bdf9da6cdc74e60240ec.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/db7851285f8707b342a7707605e270a10336c5110017bdf9da6cdc74e60240ec.jpg)

![f4fed3459096842722cb416687c75c8b57bda6e80e4b0e8938cd875af6a5964e.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/images/f4fed3459096842722cb416687c75c8b57bda6e80e4b0e8938cd875af6a5964e.jpg)

### Tables

![22eaa6bdbeb3de08199a25497b34a4aa3a4b693d48e6c664f0010930689be8f1.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/22eaa6bdbeb3de08199a25497b34a4aa3a4b693d48e6c664f0010930689be8f1.jpg)

![24e22510ca228989aabc0fc96969be1e5fca74ccdd9e315b02fb630f21f6662c.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/24e22510ca228989aabc0fc96969be1e5fca74ccdd9e315b02fb630f21f6662c.jpg)

![25341080c79014c01540191b978f112f655ae698a69028f69adcc1ab4449d1d2.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/25341080c79014c01540191b978f112f655ae698a69028f69adcc1ab4449d1d2.jpg)

![287ea728a501d431eb8cdaaba4ebdc95bd996beb1e800ea755910791f31117e4.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/287ea728a501d431eb8cdaaba4ebdc95bd996beb1e800ea755910791f31117e4.jpg)

![2e2025849455d7445d9d141aa55c29690cd50996219313c9f42898cc5650ebc3.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/2e2025849455d7445d9d141aa55c29690cd50996219313c9f42898cc5650ebc3.jpg)

![3f04280b96dc2a1e89423e90e1f89bf71f5bd1ff1d3b13d646c47f7b39206870.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/3f04280b96dc2a1e89423e90e1f89bf71f5bd1ff1d3b13d646c47f7b39206870.jpg)

![43caafb49a8279d52d8d2faeacfc1c0df7111c41a9dd01e1ef1aa0b5f34528c1.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/43caafb49a8279d52d8d2faeacfc1c0df7111c41a9dd01e1ef1aa0b5f34528c1.jpg)

![445b8d68e750aaf7dbd71c53672299b4a5509b945e10558cc87f12ae952ff948.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/445b8d68e750aaf7dbd71c53672299b4a5509b945e10558cc87f12ae952ff948.jpg)

![5436714df15378758e4e6fd06b676d1035a3c0ba1180560336abf85cddf7288b.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/5436714df15378758e4e6fd06b676d1035a3c0ba1180560336abf85cddf7288b.jpg)

![64f85f595fea3ebe3d79e59ff03d413b52001ae34d7991cb9dac7bd4b6ae6e18.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/64f85f595fea3ebe3d79e59ff03d413b52001ae34d7991cb9dac7bd4b6ae6e18.jpg)

![7336e75447afa6077d4e29016f6c61e5c8bc3f36b31b31183d85e2189130c237.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/7336e75447afa6077d4e29016f6c61e5c8bc3f36b31b31183d85e2189130c237.jpg)

![8f3d06c474246dad5f74593fbae1ad097d38ffb699058461ba116a87a9c7160e.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/8f3d06c474246dad5f74593fbae1ad097d38ffb699058461ba116a87a9c7160e.jpg)

![8f59d9c056c8f4379b624792dff5a925cb9d432461beaeed06af3b57d8112b0a.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/8f59d9c056c8f4379b624792dff5a925cb9d432461beaeed06af3b57d8112b0a.jpg)

![a45360ab742a9cc384037a7996f8cf1319d7ff7cad37cab7f1b23eb26fe5252b.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/a45360ab742a9cc384037a7996f8cf1319d7ff7cad37cab7f1b23eb26fe5252b.jpg)

![adb84983642f27f097910a21adfb65b76442ef86d421b8e8327ba3eb2f89e18a.jpg](../icml_results/1550_How%20Good%20is%20Post-Hoc%20Watermarking%20With%20Language%20Model%20Rephrasing_/tables/adb84983642f27f097910a21adfb65b76442ef86d421b8e8327ba3eb2f89e18a.jpg)

<a id="paper-1530"></a>

## Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos


### Images

![030f15fecfd09e7591de932bac7d940577d991da2f9973ec5a1df010cde9a691.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/030f15fecfd09e7591de932bac7d940577d991da2f9973ec5a1df010cde9a691.jpg)

![14fc880e3a1d00c21cacef7e8f071b6cfdc9dcf0651c7694e6f42506be1c1b28.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/14fc880e3a1d00c21cacef7e8f071b6cfdc9dcf0651c7694e6f42506be1c1b28.jpg)

![1b66528102dc4de1b6251592ed888b9010825150ae06eb3ee1e65688fb4370ca.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/1b66528102dc4de1b6251592ed888b9010825150ae06eb3ee1e65688fb4370ca.jpg)

![213bfb504d47137c1380b13c473c8ebf8e56577155e17a891a55c17ba891dd13.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/213bfb504d47137c1380b13c473c8ebf8e56577155e17a891a55c17ba891dd13.jpg)

![498640e368ac40857190a42b079b102109ffaee997d9338de2d008e5d3aa9022.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/498640e368ac40857190a42b079b102109ffaee997d9338de2d008e5d3aa9022.jpg)

![74de591708332b778f88c06ea35371f4395d23e35487f1b80c4daf301c8fe62e.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/74de591708332b778f88c06ea35371f4395d23e35487f1b80c4daf301c8fe62e.jpg)

![969a9a8723fc76e9e7842143d271c6cc7d5e3144d07163cf071f71bcd88d6b02.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/969a9a8723fc76e9e7842143d271c6cc7d5e3144d07163cf071f71bcd88d6b02.jpg)

![b4e049b46fa518055cd4c9a40ac13452f04e837c7ccb505d0033d48573c6d46f.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/b4e049b46fa518055cd4c9a40ac13452f04e837c7ccb505d0033d48573c6d46f.jpg)

![bef28ad508134bc7b4f343801d6e5c2e786ac0337aa409828d5723bc670788bc.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/bef28ad508134bc7b4f343801d6e5c2e786ac0337aa409828d5723bc670788bc.jpg)

![cf1991451f4e47cdbea2767ec9b40762fe2d95e3557bdef52f7f4b0af94a127f.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/cf1991451f4e47cdbea2767ec9b40762fe2d95e3557bdef52f7f4b0af94a127f.jpg)

![d8bcb1e8071e7317aa7b8cc101a2990baf1a8fc002d4375ac40292a0f575780f.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/d8bcb1e8071e7317aa7b8cc101a2990baf1a8fc002d4375ac40292a0f575780f.jpg)

![e892a29bcc66cbbdac4caa2dbfaa0716b8b86d512edf6d6aff84ba74e34297d7.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/images/e892a29bcc66cbbdac4caa2dbfaa0716b8b86d512edf6d6aff84ba74e34297d7.jpg)

### Tables

![1ebf78160d78b3c456465261ded7f6b9f16cbd5ac047d1349f771a010a33c010.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/1ebf78160d78b3c456465261ded7f6b9f16cbd5ac047d1349f771a010a33c010.jpg)

![1fc9a360e28c3646c84285d618a144d2bf9845fa9108de84940b11aa86df0c92.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/1fc9a360e28c3646c84285d618a144d2bf9845fa9108de84940b11aa86df0c92.jpg)

![21e5dda1b9778fe7e96c01b348a3792f996325e71823da233d2e6ac1b87979b4.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/21e5dda1b9778fe7e96c01b348a3792f996325e71823da233d2e6ac1b87979b4.jpg)

![2b155e907d8ab71e37d86e8c9f5a96670a1e16b26683dd9ad458ce02c4a7f183.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/2b155e907d8ab71e37d86e8c9f5a96670a1e16b26683dd9ad458ce02c4a7f183.jpg)

![2d9b448e597a1be77f9be10f9e2c8a8d2dbd791c0687ea04ed9df3253a4ee9fb.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/2d9b448e597a1be77f9be10f9e2c8a8d2dbd791c0687ea04ed9df3253a4ee9fb.jpg)

![2faebc57a92e76c9fea79dd87d1fe8e1daa8c36d2df370360e0f080fdffa3569.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/2faebc57a92e76c9fea79dd87d1fe8e1daa8c36d2df370360e0f080fdffa3569.jpg)

![2fe5fff93457746be2a9e557e4cee7bbe609c304b26dffc6e47013b79b4c14e2.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/2fe5fff93457746be2a9e557e4cee7bbe609c304b26dffc6e47013b79b4c14e2.jpg)

![3e26970678e540dc29d61949a12e2f780d14b331cea2ca01261bc47c89990850.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/3e26970678e540dc29d61949a12e2f780d14b331cea2ca01261bc47c89990850.jpg)

![9648435276018fdb3dcb8ca240b417d4fd779207891e313d4531ad11c941bb12.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/9648435276018fdb3dcb8ca240b417d4fd779207891e313d4531ad11c941bb12.jpg)

![9983ad94287a919d3bbd85cda6d8ff76a285f8409ba771ada71f8f0f1a06646b.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/9983ad94287a919d3bbd85cda6d8ff76a285f8409ba771ada71f8f0f1a06646b.jpg)

![cc74a95cdfdc0267fa912b08ca64004a0293edf9cf612cffcc03ea71e30ae376.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/cc74a95cdfdc0267fa912b08ca64004a0293edf9cf612cffcc03ea71e30ae376.jpg)

![dad0bbd516223da0ed163595d1a6058083a0e76f078c1665a0e584c4530daf4a.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/dad0bbd516223da0ed163595d1a6058083a0e76f078c1665a0e584c4530daf4a.jpg)

![e06e9733f27615f2c35a43623b216c8c1c94e8bb5d2d197df2b2dcb4127fc5a8.jpg](../icml_results/1551_Dropout%20Universality_%20Scaling%20Laws%20and%20Optimal%20Scheduling%20at%20the%20Edge-of-Chaos/tables/e06e9733f27615f2c35a43623b216c8c1c94e8bb5d2d197df2b2dcb4127fc5a8.jpg)

<a id="paper-1531"></a>

## CLASP: Online learning algorithms for Convex Losses And Squared Penalties


### Images

![2ff7e853db31576d40b099c7d4930e47c2e0dd7814339c2c1f0951b95fc53545.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/images/2ff7e853db31576d40b099c7d4930e47c2e0dd7814339c2c1f0951b95fc53545.jpg)

![5ad84b0006bb2df940156eef336a2cf935b73f90d88cb9fb295d129d6cadc368.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/images/5ad84b0006bb2df940156eef336a2cf935b73f90d88cb9fb295d129d6cadc368.jpg)

![9ce85fd6ec34448690b34175bad1eef0be98df18805446721eb0631477b232a9.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/images/9ce85fd6ec34448690b34175bad1eef0be98df18805446721eb0631477b232a9.jpg)

![b7b2772bdf86bccc2e74b4835d914cec3974d7c6f7ca14100d771d3872091b73.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/images/b7b2772bdf86bccc2e74b4835d914cec3974d7c6f7ca14100d771d3872091b73.jpg)

![c770b2433914b354ae2494e5549124bc6c723498bb4c0779ecff6eb273c5044d.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/images/c770b2433914b354ae2494e5549124bc6c723498bb4c0779ecff6eb273c5044d.jpg)

### Tables

![7c2891f465a9e911bd7eb30e12acf509bea2e8be8c347b0959f443decafa54a2.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/tables/7c2891f465a9e911bd7eb30e12acf509bea2e8be8c347b0959f443decafa54a2.jpg)

![8386b4d1bc3fdaa3455cc7c8a3ca47c8f99e844e0a8516b437d475f585625136.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/tables/8386b4d1bc3fdaa3455cc7c8a3ca47c8f99e844e0a8516b437d475f585625136.jpg)

![f406a32d1d96e728ae5c7c0b9b6c67df062e65cfdc73834376c0a104757865c3.jpg](../icml_results/1552_CLASP_%20Online%20learning%20algorithms%20for%20Convex%20Losses%20And%20Squared%20Penalties/tables/f406a32d1d96e728ae5c7c0b9b6c67df062e65cfdc73834376c0a104757865c3.jpg)

<a id="paper-1532"></a>

## Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution


### Images

![182336594a512c0249a2e075096e7454e92c48bb950cbaf1055f4bfc08aafbf8.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/182336594a512c0249a2e075096e7454e92c48bb950cbaf1055f4bfc08aafbf8.jpg)

![25b3fcb58f22c28dcc5ecf67164efa1e8f540416d3905951d3edc6e4d393e7f9.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/25b3fcb58f22c28dcc5ecf67164efa1e8f540416d3905951d3edc6e4d393e7f9.jpg)

![4233af0219cf568df13e22ba804990ea88e5b8d33fb9d63b86cf011bff1c8986.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/4233af0219cf568df13e22ba804990ea88e5b8d33fb9d63b86cf011bff1c8986.jpg)

![52d130df716184d0c17f4ac892961bc731ab5dee055e2c3bc60fba5b8d6e107e.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/52d130df716184d0c17f4ac892961bc731ab5dee055e2c3bc60fba5b8d6e107e.jpg)

![601e6194efd20f90a0b09165fe3245e2d2e62b2fe378220a6457003bc799baeb.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/601e6194efd20f90a0b09165fe3245e2d2e62b2fe378220a6457003bc799baeb.jpg)

![728b2777fcbc00f34c82f06a4e2dd31071a81b329efa553eec6241dfe077f4a4.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/728b2777fcbc00f34c82f06a4e2dd31071a81b329efa553eec6241dfe077f4a4.jpg)

![9deea953a131e93eeafacab954eebb73bc3f74c6cb02669fe1135450fa199785.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/9deea953a131e93eeafacab954eebb73bc3f74c6cb02669fe1135450fa199785.jpg)

![9eb4b1e307b5886896e263fd6965cc9fe7e3fb9dd5fbabe590293f053afcc4d9.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/9eb4b1e307b5886896e263fd6965cc9fe7e3fb9dd5fbabe590293f053afcc4d9.jpg)

![a452196824282b2573633173cdb87aa25faa1b3bb05aa261cecdb23f2d61d70d.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/a452196824282b2573633173cdb87aa25faa1b3bb05aa261cecdb23f2d61d70d.jpg)

![b492d9390259fd9113004e2c3db5fb7dcef57854fd337e9a8e1d1dbe6bf0769b.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/b492d9390259fd9113004e2c3db5fb7dcef57854fd337e9a8e1d1dbe6bf0769b.jpg)

![d79216492016595cb941d2a49d05ce288c4851174734699493b04cd6956dcfff.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/images/d79216492016595cb941d2a49d05ce288c4851174734699493b04cd6956dcfff.jpg)

### Tables

![04e42958fd0895bd16df11c005b6b6c34bd9df5d60938edad8544964ff36b4aa.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/04e42958fd0895bd16df11c005b6b6c34bd9df5d60938edad8544964ff36b4aa.jpg)

![05791c39ad5a3f2e3efb84d277a8ea430c73065d578ecfa53cb8e6e8bdf2f501.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/05791c39ad5a3f2e3efb84d277a8ea430c73065d578ecfa53cb8e6e8bdf2f501.jpg)

![07e386e1ac2b031c63863196cfed96c3b601203033775b4af4bf0e824710b0a5.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/07e386e1ac2b031c63863196cfed96c3b601203033775b4af4bf0e824710b0a5.jpg)

![1e343e8c77779a6169f88ced8e6cf48e38683604866fea6f11e785ec35ae7afe.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/1e343e8c77779a6169f88ced8e6cf48e38683604866fea6f11e785ec35ae7afe.jpg)

![2092b583d3ace6bd9e9f4d756847187ccdef6b264f929956d6a07f100d6b8a45.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/2092b583d3ace6bd9e9f4d756847187ccdef6b264f929956d6a07f100d6b8a45.jpg)

![23324c502115e8ef86dc7e9d56229680c5a52ff0850b0fe6fc76175194151e1d.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/23324c502115e8ef86dc7e9d56229680c5a52ff0850b0fe6fc76175194151e1d.jpg)

![24e75180d60587429fd85175773e19089e3b1cfef7c2cc8ea83fd20143c1bdf5.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/24e75180d60587429fd85175773e19089e3b1cfef7c2cc8ea83fd20143c1bdf5.jpg)

![28b6440f1ee1cecd356058389a7387a3305dec99209b29028b800c2dc4d89fc9.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/28b6440f1ee1cecd356058389a7387a3305dec99209b29028b800c2dc4d89fc9.jpg)

![2acb4dcad2a96ecbd782027dc4bb4e60c34a02a782dc882c664352e02889c7cc.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/2acb4dcad2a96ecbd782027dc4bb4e60c34a02a782dc882c664352e02889c7cc.jpg)

![2b34d21b8151678c0e711a43170a0688b6babceaa636bf32b918543a4d6ad490.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/2b34d21b8151678c0e711a43170a0688b6babceaa636bf32b918543a4d6ad490.jpg)

![2e3f5ce910899cd4b9a0313884aa70862761a919eeaaaa22cf48004996fc6f54.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/2e3f5ce910899cd4b9a0313884aa70862761a919eeaaaa22cf48004996fc6f54.jpg)

![375d1c47b04b2319320ff156e1fd5fe5692b7455b3795735544d4d55b47b1fce.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/375d1c47b04b2319320ff156e1fd5fe5692b7455b3795735544d4d55b47b1fce.jpg)

![38922e7ee76cd8dad4a92948c40129d81534e29cb2a3b3736e875c4c7bf39a58.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/38922e7ee76cd8dad4a92948c40129d81534e29cb2a3b3736e875c4c7bf39a58.jpg)

![436357cdc0efab43b71373930b8e79a593ad970c12fc25bec2d37e7e78a986ff.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/436357cdc0efab43b71373930b8e79a593ad970c12fc25bec2d37e7e78a986ff.jpg)

![48ce879f57972695b49b71bab0019c5a4324198fa113f9294969c98fca42797a.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/48ce879f57972695b49b71bab0019c5a4324198fa113f9294969c98fca42797a.jpg)

![4d8c213d3eaf11349173d4eb2dfaa81a0df0515bbb26d9c7e91563af6aa05d68.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/4d8c213d3eaf11349173d4eb2dfaa81a0df0515bbb26d9c7e91563af6aa05d68.jpg)

![4f336821fd1f0d47a3c449aabe7f075229da91add84d6cf06449db4938607a5a.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/4f336821fd1f0d47a3c449aabe7f075229da91add84d6cf06449db4938607a5a.jpg)

![574d0bec9bb939ed90336c1c14d09d6550962e2264abfec788d8d69f8e733441.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/574d0bec9bb939ed90336c1c14d09d6550962e2264abfec788d8d69f8e733441.jpg)

![59bc5f9bee53148e00aa94649e364253c8a6a5ccedd29b508aa6a25877c5b5f6.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/59bc5f9bee53148e00aa94649e364253c8a6a5ccedd29b508aa6a25877c5b5f6.jpg)

![5a429b9a282a450557d8a38d60c9725cfb764eefc1a417b148a77a28d5b65630.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/5a429b9a282a450557d8a38d60c9725cfb764eefc1a417b148a77a28d5b65630.jpg)

![60543f179b9ea86d943f825fe8febd412637bb65dca55e19b0f495d6f23969b4.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/60543f179b9ea86d943f825fe8febd412637bb65dca55e19b0f495d6f23969b4.jpg)

![6831d52674013d864ec2cee9ebccd7326f77b46b87235ff8d307594959b17b61.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/6831d52674013d864ec2cee9ebccd7326f77b46b87235ff8d307594959b17b61.jpg)

![6eb339910c52f003db0c023a15c540a8729f9522496bc925f871e511f9b5cfb0.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/6eb339910c52f003db0c023a15c540a8729f9522496bc925f871e511f9b5cfb0.jpg)

![724af12dfd0a1b54709227d485749b3d18aeac005c659619dd7fe0bfaacad90b.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/724af12dfd0a1b54709227d485749b3d18aeac005c659619dd7fe0bfaacad90b.jpg)

![793ca67a679e58ee4c24fc6676b6be61fdac00dfc609a64c1fb283a81a24f54c.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/793ca67a679e58ee4c24fc6676b6be61fdac00dfc609a64c1fb283a81a24f54c.jpg)

![7f1ffa3210daa1548e220ad070e64edcc4784584688d025e5098ff1ce83f27b0.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/7f1ffa3210daa1548e220ad070e64edcc4784584688d025e5098ff1ce83f27b0.jpg)

![81327d3d9f45c1ee48d9e76475ba8fbee17adff3a659df7eebed7082c9052d3d.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/81327d3d9f45c1ee48d9e76475ba8fbee17adff3a659df7eebed7082c9052d3d.jpg)

![857a693000f1ec96fa1889a73eadf774d612507ff432ad6ae546a362dbb5f1f8.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/857a693000f1ec96fa1889a73eadf774d612507ff432ad6ae546a362dbb5f1f8.jpg)

![8660338e0569e7ce7fac235cec36d01cfa33a2d7b19b4ad70683f0f8f3bed6db.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/8660338e0569e7ce7fac235cec36d01cfa33a2d7b19b4ad70683f0f8f3bed6db.jpg)

![8ad09591886661766a4d150123f573630551890c6c4722dee0aced1a7c096b32.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/8ad09591886661766a4d150123f573630551890c6c4722dee0aced1a7c096b32.jpg)

![8bbc63086e7f8cb0ffa4b3267bb550e1ff3779eee066aa3e2adc901619996aa6.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/8bbc63086e7f8cb0ffa4b3267bb550e1ff3779eee066aa3e2adc901619996aa6.jpg)

![8e12544adfa51da7b2b0bbfbd2324199a4d43af0aaee6ab0c0603f92b3c95ba0.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/8e12544adfa51da7b2b0bbfbd2324199a4d43af0aaee6ab0c0603f92b3c95ba0.jpg)

![8f014b237f44945dfd249087effda4f4fbdfa1b11fcbcecaad298321ee257977.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/8f014b237f44945dfd249087effda4f4fbdfa1b11fcbcecaad298321ee257977.jpg)

![9131c66d1747b80a28157a707894cd870246663f59278f94aea05d0ab780baef.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/9131c66d1747b80a28157a707894cd870246663f59278f94aea05d0ab780baef.jpg)

![915f7c71db14aeb60fe6e4708da1c915b8f663d4f5681425689f1d7af9152815.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/915f7c71db14aeb60fe6e4708da1c915b8f663d4f5681425689f1d7af9152815.jpg)

![9aa7d98ed64777fa6cab9ca13a2884202a2071981a22e4a8a118975e020a1ba2.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/9aa7d98ed64777fa6cab9ca13a2884202a2071981a22e4a8a118975e020a1ba2.jpg)

![9d33156a84167eb47845c18c8d0f3abba1d7d6c436c55b1e3145e27f3ad81708.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/9d33156a84167eb47845c18c8d0f3abba1d7d6c436c55b1e3145e27f3ad81708.jpg)

![a43513aca28bcc7daa25d3230f7899143a14f1e4097cc96eb01000bbe1ca3410.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/a43513aca28bcc7daa25d3230f7899143a14f1e4097cc96eb01000bbe1ca3410.jpg)

![a547c5b49e13c4bce69b4ea718d993698496ed9ddeea15343952d67f5cdaf0b3.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/a547c5b49e13c4bce69b4ea718d993698496ed9ddeea15343952d67f5cdaf0b3.jpg)

![a5a1e9ecf3a93a47409bc80a71f7abd941da290df41814b7c206759fb153ae7c.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/a5a1e9ecf3a93a47409bc80a71f7abd941da290df41814b7c206759fb153ae7c.jpg)

![b1daa9520d64c8966055ce332969fd443839a234063e6aaf5138ede92dcc3a1a.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/b1daa9520d64c8966055ce332969fd443839a234063e6aaf5138ede92dcc3a1a.jpg)

![b5d8ea4c7eb66ca9fcd3f1ac104d59f41495540c4286b3a68ca2764facf6cde5.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/b5d8ea4c7eb66ca9fcd3f1ac104d59f41495540c4286b3a68ca2764facf6cde5.jpg)

![c46796380167a2819e375e1c2c272d2080c9fc276a682e125506f89869884289.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/c46796380167a2819e375e1c2c272d2080c9fc276a682e125506f89869884289.jpg)

![d8d7103d67c808ea404cd48aa000ed132e02711737a9076dba9b8e6c94e7982b.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/d8d7103d67c808ea404cd48aa000ed132e02711737a9076dba9b8e6c94e7982b.jpg)

![da8003c3b1cd334f6221b5aa120841759f31f5744b0f8db2dced33f4dc88440b.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/da8003c3b1cd334f6221b5aa120841759f31f5744b0f8db2dced33f4dc88440b.jpg)

![de4d576be0a81fcdfbbb94c0dd9c2665a9c0d42fb03c933cdacf6388d70ae552.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/de4d576be0a81fcdfbbb94c0dd9c2665a9c0d42fb03c933cdacf6388d70ae552.jpg)

![e60a3975cb2b6eba99931b9d069414187580bd81dde07e5a9af819c4b69ef081.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/e60a3975cb2b6eba99931b9d069414187580bd81dde07e5a9af819c4b69ef081.jpg)

![e7c3d0fc1e51b245b342473d8fac09c434bcefb9300be1c6e1b86110d9df6c8a.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/e7c3d0fc1e51b245b342473d8fac09c434bcefb9300be1c6e1b86110d9df6c8a.jpg)

![f2dd24bef10bc2f77729589c05432c9dca5398cdb4ad514a46f4f623233b7b91.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/f2dd24bef10bc2f77729589c05432c9dca5398cdb4ad514a46f4f623233b7b91.jpg)

![f41563cae923186ffe621471597c11672cf4c39431b3191ba25552c659a0d87c.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/f41563cae923186ffe621471597c11672cf4c39431b3191ba25552c659a0d87c.jpg)

![f937941921ef37cc7f759286ba91dc00f77b171e363c45aee3e664eeadea7fd4.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/f937941921ef37cc7f759286ba91dc00f77b171e363c45aee3e664eeadea7fd4.jpg)

![fbdae97cab22c8488cf9faf2d212a8b1e5d2e91c312b3dee4aed9b538044b9b9.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/fbdae97cab22c8488cf9faf2d212a8b1e5d2e91c312b3dee4aed9b538044b9b9.jpg)

![fd2d74ee0f311cc6acd78e20a2a3618b7d14882361366765e71931657ed3849c.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/fd2d74ee0f311cc6acd78e20a2a3618b7d14882361366765e71931657ed3849c.jpg)

![fe61040cdd9680aeaab73c1bb6d0b028efdfbb307d5d5689b5cb40011fe504cf.jpg](../icml_results/1553_Rethinking%20GNNs%20and%20Missing%20Features_%20Challenges%2C%20Evaluation%20and%20a%20Robust%20Solution/tables/fe61040cdd9680aeaab73c1bb6d0b028efdfbb307d5d5689b5cb40011fe504cf.jpg)

<a id="paper-1533"></a>

## Evaluating Relational Reasoning in LLMs with REL


### Images

![21f0db2c9689dd6957a29962ef3379de068b9728d6c01752c8c14c79ba377660.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/21f0db2c9689dd6957a29962ef3379de068b9728d6c01752c8c14c79ba377660.jpg)

![30b28d0dcd272f27050ea13adca8740cb6b2c746f3df4d746a326206f75b85a6.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/30b28d0dcd272f27050ea13adca8740cb6b2c746f3df4d746a326206f75b85a6.jpg)

![35485e0f751e0531db660ef7517e1fdb6480b169b79076a39d40cd7ad2825db3.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/35485e0f751e0531db660ef7517e1fdb6480b169b79076a39d40cd7ad2825db3.jpg)

![4db4f57e283448e307afa9975eee6f4044e49f261a4aae695d4f5013b906e0d5.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/4db4f57e283448e307afa9975eee6f4044e49f261a4aae695d4f5013b906e0d5.jpg)

![70203fbb5763589f4fab25a8fd4ac8ce426382d6508210a09d87bee9a732c0a6.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/70203fbb5763589f4fab25a8fd4ac8ce426382d6508210a09d87bee9a732c0a6.jpg)

![7610c86cc1366a6c9dab7a7f1c0d702f15544adddf0bc688623ec7d033ece3d5.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/7610c86cc1366a6c9dab7a7f1c0d702f15544adddf0bc688623ec7d033ece3d5.jpg)

![767ff190d397fc14551de418a22346f956cd7d877811916d44f3642569aec823.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/767ff190d397fc14551de418a22346f956cd7d877811916d44f3642569aec823.jpg)

![7dfe4319eb0471d087b2b1490b11c81cec1a80827f5b201d6f3737a9cf2c709d.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/7dfe4319eb0471d087b2b1490b11c81cec1a80827f5b201d6f3737a9cf2c709d.jpg)

![8b80c451c045a7deba8a087e2b4272a01e5364e91d324c865aba7023d8cbf082.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/8b80c451c045a7deba8a087e2b4272a01e5364e91d324c865aba7023d8cbf082.jpg)

![a315d08db14858f38b1072c250d5e38809a119610562bb70142d82d9342cc82c.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/a315d08db14858f38b1072c250d5e38809a119610562bb70142d82d9342cc82c.jpg)

![c98a84e4f32d2ab949de168bc2024084cbbc8bab36687826a0590cdfee369b74.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/c98a84e4f32d2ab949de168bc2024084cbbc8bab36687826a0590cdfee369b74.jpg)

![d08a44e7c58e0c5d881e48f62523ff59d79c69955d5b082733e1411ea4f450d8.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/d08a44e7c58e0c5d881e48f62523ff59d79c69955d5b082733e1411ea4f450d8.jpg)

![d13905279d0b8d30cb62b978f807c37ef128466a7770f049a8e597111a1234e2.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/d13905279d0b8d30cb62b978f807c37ef128466a7770f049a8e597111a1234e2.jpg)

![e3b85d591504f9a6572dd320e55e243cde7c9da81d4b938cdd24b1c6ff74b284.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/e3b85d591504f9a6572dd320e55e243cde7c9da81d4b938cdd24b1c6ff74b284.jpg)

![edcaf9c56869a6ca7c7b7d14484d01d86bd9054ce15886b038b722a686541838.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/edcaf9c56869a6ca7c7b7d14484d01d86bd9054ce15886b038b722a686541838.jpg)

![ee6e68d925d4fb3a4e0e3466ce3f49272f1a5c3cc487b69ed34328350510b226.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/images/ee6e68d925d4fb3a4e0e3466ce3f49272f1a5c3cc487b69ed34328350510b226.jpg)

### Tables

![0be990734121f49fd8ddcf61e9ec93b7d8aab0bfc5053ccf15394f80ba75dd46.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/0be990734121f49fd8ddcf61e9ec93b7d8aab0bfc5053ccf15394f80ba75dd46.jpg)

![4a966b8ab4eb8860c6e115824e8ea17af2ea271775e0295a0a28bf5b6ec371f8.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/4a966b8ab4eb8860c6e115824e8ea17af2ea271775e0295a0a28bf5b6ec371f8.jpg)

![4e9f5c1b28130353e87753b10c5d5e94bc226f75431e8c5cf95bcd155cb52229.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/4e9f5c1b28130353e87753b10c5d5e94bc226f75431e8c5cf95bcd155cb52229.jpg)

![71bcb5fd5811c380d08e512583dc68ed61d1ca6446bc53dc5d7d83e800729e29.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/71bcb5fd5811c380d08e512583dc68ed61d1ca6446bc53dc5d7d83e800729e29.jpg)

![92140c35b9433c17a751a67df929e6389bb4ace7d93b8b3620b1950d275d84e8.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/92140c35b9433c17a751a67df929e6389bb4ace7d93b8b3620b1950d275d84e8.jpg)

![94373ed3bd1a39342c779b0c2282a727500ed4d3d09c82e8d74d21fe60271d2b.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/94373ed3bd1a39342c779b0c2282a727500ed4d3d09c82e8d74d21fe60271d2b.jpg)

![94dc2f42f68e02bc0395c60ff28dbc04fc4d40e85aafb8f205d5bd3351401c33.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/94dc2f42f68e02bc0395c60ff28dbc04fc4d40e85aafb8f205d5bd3351401c33.jpg)

![abb238e8fc619ddb3f749a927d2968d69f4a5299ff466ed9f7b0da209a9d8e49.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/abb238e8fc619ddb3f749a927d2968d69f4a5299ff466ed9f7b0da209a9d8e49.jpg)

![afa63482fea10401cbfbd35d4620e028f95dbb4ca9c91e75e65ac4a425cb6a36.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/afa63482fea10401cbfbd35d4620e028f95dbb4ca9c91e75e65ac4a425cb6a36.jpg)

![e8c23a3effb6f8ad88d8bd98729152ed599e4fd7b3de88d7578570e0103a697b.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/e8c23a3effb6f8ad88d8bd98729152ed599e4fd7b3de88d7578570e0103a697b.jpg)

![f4f1e0dc8c88b4744f1ad13a31e1706785cbc4c1c3b303d978640f7a1f6badfc.jpg](../icml_results/1554_Evaluating%20Relational%20Reasoning%20in%20LLMs%20with%20REL/tables/f4f1e0dc8c88b4744f1ad13a31e1706785cbc4c1c3b303d978640f7a1f6badfc.jpg)

<a id="paper-1534"></a>

## Unitary Convolutions for Message-passing and Positional Encodings on Directed Graphs


### Images

![41314ca862151006d8bb2c71b4d873c13ef596d0eecc77a63c943bb8e06367e5.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/41314ca862151006d8bb2c71b4d873c13ef596d0eecc77a63c943bb8e06367e5.jpg)

![4cea213cfe2dc2a9ce334990785d00b72ec52306ccfe3f562a82931ba0a9988d.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/4cea213cfe2dc2a9ce334990785d00b72ec52306ccfe3f562a82931ba0a9988d.jpg)

![4ec5eeaad293bc231de2bddc52262388d0e458be0cdcd8a968244b33bbec62ab.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/4ec5eeaad293bc231de2bddc52262388d0e458be0cdcd8a968244b33bbec62ab.jpg)

![6238e67b37d4f38af9c06129113ff3e519db8fef816b0f30e502cec6155594f3.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/6238e67b37d4f38af9c06129113ff3e519db8fef816b0f30e502cec6155594f3.jpg)

![931ae4ecc8e40e90ba11bcd35a9f275592c69ef00b690d7db1b553e14b322571.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/931ae4ecc8e40e90ba11bcd35a9f275592c69ef00b690d7db1b553e14b322571.jpg)

![c81c9c6a4cc752af1e1d5df0c596fcbfcb970e8cc7626dc596ce95cfaea12502.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/images/c81c9c6a4cc752af1e1d5df0c596fcbfcb970e8cc7626dc596ce95cfaea12502.jpg)

### Tables

![015174ad510c9d7272d2fc3ae94103818ad6e594a9ba9990536245c2cb9ddccc.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/015174ad510c9d7272d2fc3ae94103818ad6e594a9ba9990536245c2cb9ddccc.jpg)

![170238f1847a864f7bce09590da63168c9f8b72f111317986a129510c3efb7a9.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/170238f1847a864f7bce09590da63168c9f8b72f111317986a129510c3efb7a9.jpg)

![25f9fda15819e4f282e860b39b049bad29416964fc078b8fba505b70b9c91de5.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/25f9fda15819e4f282e860b39b049bad29416964fc078b8fba505b70b9c91de5.jpg)

![322d0c6c8d985245fe78bf812dc1da8a13602b82d8dd8cd2f072b90e3cb83d62.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/322d0c6c8d985245fe78bf812dc1da8a13602b82d8dd8cd2f072b90e3cb83d62.jpg)

![6bcaf75b58b60670ffb0eced2c2dbee3f185b4670cfedf9dd108ad3751b8587a.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/6bcaf75b58b60670ffb0eced2c2dbee3f185b4670cfedf9dd108ad3751b8587a.jpg)

![6c6af011a334666f2806b449b1f24fac556cc677cb62c62eee5315f34c547b4a.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/6c6af011a334666f2806b449b1f24fac556cc677cb62c62eee5315f34c547b4a.jpg)

![78a4f45f5086669bb3657db0d1bb2d0543d13059a1fea8169ce88e069450463e.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/78a4f45f5086669bb3657db0d1bb2d0543d13059a1fea8169ce88e069450463e.jpg)

![862cdd35423b197de74d7ecca445eb7357e51371e9c587637d12018a425034af.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/862cdd35423b197de74d7ecca445eb7357e51371e9c587637d12018a425034af.jpg)

![86b80a0fc44cb0a06cb3b9240e3fd5651eb99095a8830fd2f05574f24488f7d7.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/86b80a0fc44cb0a06cb3b9240e3fd5651eb99095a8830fd2f05574f24488f7d7.jpg)

![88bc0e25cf5fa22c4cded32e51dced11bace171de678bf404a60ab3149d100e7.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/88bc0e25cf5fa22c4cded32e51dced11bace171de678bf404a60ab3149d100e7.jpg)

![939b85a0caa08c8aa2c949191392ed5c224b03d2482e77e0135739542d23115e.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/939b85a0caa08c8aa2c949191392ed5c224b03d2482e77e0135739542d23115e.jpg)

![a2d6aff32eedfd03e66e7eaa5bde0404e352d3aefda34936b2914f760eb3f7d5.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/a2d6aff32eedfd03e66e7eaa5bde0404e352d3aefda34936b2914f760eb3f7d5.jpg)

![a49e64739fa9e0e85e0d1f94bb8a4351d650336e3c800eb981c2851b4b3be946.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/a49e64739fa9e0e85e0d1f94bb8a4351d650336e3c800eb981c2851b4b3be946.jpg)

![a627260a2329408e9fa75a7816d13dbfddab4019e23852fd5e3f51ba9d2c875f.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/a627260a2329408e9fa75a7816d13dbfddab4019e23852fd5e3f51ba9d2c875f.jpg)

![ac4b1614b07f34114928bc053a19d0457c75aacdf16bd1f0b41927590fdffd45.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/ac4b1614b07f34114928bc053a19d0457c75aacdf16bd1f0b41927590fdffd45.jpg)

![b68ec2a0782d8cd0ac302f053507f38888daec0eae9bacef4d7db083eb0770ca.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/b68ec2a0782d8cd0ac302f053507f38888daec0eae9bacef4d7db083eb0770ca.jpg)

![c0505034273a6d108c58b51956e3be47cc5fda81fd4699ec824ca5ff413af6dc.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/c0505034273a6d108c58b51956e3be47cc5fda81fd4699ec824ca5ff413af6dc.jpg)

![cad6f84d21852f04d5cc12b9954ac06eceacb1e92a9d8d14a82c88446e32dad0.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/cad6f84d21852f04d5cc12b9954ac06eceacb1e92a9d8d14a82c88446e32dad0.jpg)

![d3afe4dfb53c7cc5f21ac39821410559690b84905c796a1f1e8df855216d46e2.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/d3afe4dfb53c7cc5f21ac39821410559690b84905c796a1f1e8df855216d46e2.jpg)

![d57a9b8afc614f3bb9301d11e622abe415513b27f8816e05dd8526e74e031196.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/d57a9b8afc614f3bb9301d11e622abe415513b27f8816e05dd8526e74e031196.jpg)

![e6f9e03a1bfabd64d9d96572b039d7f5981397e88ef5d7faaeee049d9ed64c7a.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/e6f9e03a1bfabd64d9d96572b039d7f5981397e88ef5d7faaeee049d9ed64c7a.jpg)

![e9a395c335f98f54156351e43d9e93a88c49e5bd5dec96a58ef8f151dd8171e8.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/e9a395c335f98f54156351e43d9e93a88c49e5bd5dec96a58ef8f151dd8171e8.jpg)

![f4d56bbe4b40936e88e9191dab1b723d70cac93e0033e802d88021ccc21a92ec.jpg](../icml_results/1555_Unitary%20Convolutions%20for%20Message-passing%20and%20Positional%20Encodings%20on%20Directed%20Graphs/tables/f4d56bbe4b40936e88e9191dab1b723d70cac93e0033e802d88021ccc21a92ec.jpg)

<a id="paper-1535"></a>

## RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections


### Images

![2603a648f9f8d364a28c2a69e4fd57fb6ed5f95dde3cb7b3d54b1d3467473eb2.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/2603a648f9f8d364a28c2a69e4fd57fb6ed5f95dde3cb7b3d54b1d3467473eb2.jpg)

![4a8166f30393de30533d777398f0497c87864f5df8997ec1e1b334cb5adece9d.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/4a8166f30393de30533d777398f0497c87864f5df8997ec1e1b334cb5adece9d.jpg)

![4c53ac111881e47251c99611990379f00391e97360260503328ef886a1014716.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/4c53ac111881e47251c99611990379f00391e97360260503328ef886a1014716.jpg)

![5d5c820204ce598a1242a5393bb0f1f554c7581d24b526744ef0a66da4f743ec.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/5d5c820204ce598a1242a5393bb0f1f554c7581d24b526744ef0a66da4f743ec.jpg)

![6a09f2b09cb755ec656b73ee566abe9e7b7f7f8a91fd58b4d01029b8bae304a0.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/6a09f2b09cb755ec656b73ee566abe9e7b7f7f8a91fd58b4d01029b8bae304a0.jpg)

![7ccf3ac0b1260ff5656d6cb69ca43e578625f43fa78fece47c29ecb629065232.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/7ccf3ac0b1260ff5656d6cb69ca43e578625f43fa78fece47c29ecb629065232.jpg)

![7db4ddd86473a9d8fd0c4c5682daf15bf8de4101279f72b28933056bb19fd637.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/7db4ddd86473a9d8fd0c4c5682daf15bf8de4101279f72b28933056bb19fd637.jpg)

![a7fd2f06f1031606ae82c782c849d45bb1c770ba5e32a55228270cc6c76a7f6d.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/a7fd2f06f1031606ae82c782c849d45bb1c770ba5e32a55228270cc6c76a7f6d.jpg)

![cd01986bbf9ea8edd116cae68c09cd884571be164f6104cedc823663d51bed70.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/cd01986bbf9ea8edd116cae68c09cd884571be164f6104cedc823663d51bed70.jpg)

![e83af3cb6bbc0a4f88fd363e8669b3b8ca64093cec89b6125d6bd6e725a33368.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/images/e83af3cb6bbc0a4f88fd363e8669b3b8ca64093cec89b6125d6bd6e725a33368.jpg)

### Tables

![107ac3fa789514b622f9b9a6ebb660785204259889b8bcbbd4e848635a380f9b.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/107ac3fa789514b622f9b9a6ebb660785204259889b8bcbbd4e848635a380f9b.jpg)

![1f2666a9901785d060f3bc540faf840a7ef04fcc9085e61ba60574275c9d6062.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/1f2666a9901785d060f3bc540faf840a7ef04fcc9085e61ba60574275c9d6062.jpg)

![20b5eb402c8a6cb3aeb16294bd00b08c6d8f2637dd6bb97ef376a76e6b582f79.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/20b5eb402c8a6cb3aeb16294bd00b08c6d8f2637dd6bb97ef376a76e6b582f79.jpg)

![2ce7c7d02c3ed0889b2aee503b6cf9d6890c06bcf4e43691a84c33165d6aca0d.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/2ce7c7d02c3ed0889b2aee503b6cf9d6890c06bcf4e43691a84c33165d6aca0d.jpg)

![3bcaa91a284eb02b61b84304d301ac7c9779a510b23eaaad537c86d1e2d0c5bf.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/3bcaa91a284eb02b61b84304d301ac7c9779a510b23eaaad537c86d1e2d0c5bf.jpg)

![7bb9b2e938b60694f07b9b8bdb558ac898b3fdea05ccbaa18d4d08ec1f9b2119.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/7bb9b2e938b60694f07b9b8bdb558ac898b3fdea05ccbaa18d4d08ec1f9b2119.jpg)

![904850dc7aefa51291ab340554cb2bff1c09b95e344855ee522d29526028cec6.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/904850dc7aefa51291ab340554cb2bff1c09b95e344855ee522d29526028cec6.jpg)

![96fde28144ac1a2870d723eaf6c70b93157d6f0343e29acc6818b5babbfebdc6.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/96fde28144ac1a2870d723eaf6c70b93157d6f0343e29acc6818b5babbfebdc6.jpg)

![a14cd6d6716a61f54e5fd8177ecc9ad4a10d133138ec808adb86c849fc366b87.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/a14cd6d6716a61f54e5fd8177ecc9ad4a10d133138ec808adb86c849fc366b87.jpg)

![b6d5f20b692edf728c6417374bd0525b4cb72b74097c44a6a15e9f242a0d9b53.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/b6d5f20b692edf728c6417374bd0525b4cb72b74097c44a6a15e9f242a0d9b53.jpg)

![beef7152321890dd788755fdfbf144806643c3dccc7b6dac4275cdbd50828b78.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/beef7152321890dd788755fdfbf144806643c3dccc7b6dac4275cdbd50828b78.jpg)

![cae482cb5b0a7bc1f1905b3c489cad4848117ddc5c8980905e0f316d209b8a40.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/cae482cb5b0a7bc1f1905b3c489cad4848117ddc5c8980905e0f316d209b8a40.jpg)

![ef7bd2fb932dbee7b0baaa17f5cdf2ab5c7519690cda9093f409ad65f993a439.jpg](../icml_results/1556_RAPNet_%20Accelerating%20Algebraic%20Multigrid%20with%20Learned%20Sparse%20Corrections/tables/ef7bd2fb932dbee7b0baaa17f5cdf2ab5c7519690cda9093f409ad65f993a439.jpg)

<a id="paper-1536"></a>

## Learning to Share: Selective Memory for Efficient Parallel Agentic Systems


### Images

![06d9f2e76ce4a91156a04f8ac5fe94321d7b6312e826a03483ec286fbd83059a.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/06d9f2e76ce4a91156a04f8ac5fe94321d7b6312e826a03483ec286fbd83059a.jpg)

![1287549b23d87a5c3561805029c8ea1a1b9ba9fe7be55e3bd9b3b79b9bb44480.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/1287549b23d87a5c3561805029c8ea1a1b9ba9fe7be55e3bd9b3b79b9bb44480.jpg)

![6cdcd40badc21235b4fd1fc8cf63a0402ae798eea7a56336f962847027e192e9.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/6cdcd40badc21235b4fd1fc8cf63a0402ae798eea7a56336f962847027e192e9.jpg)

![8d989d6ce48ba6addaa125d75e092c1069223af59f3dc95e06fc18498a5767d1.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/8d989d6ce48ba6addaa125d75e092c1069223af59f3dc95e06fc18498a5767d1.jpg)

![934fb04b400b84525bc5a09885c9677e1fa2f31a8cb54a77b3c2c6ff39be90c7.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/934fb04b400b84525bc5a09885c9677e1fa2f31a8cb54a77b3c2c6ff39be90c7.jpg)

![a7bc543a8b1c071b360c414c27c1c8610c57e49c2dce67da917b366176465224.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/a7bc543a8b1c071b360c414c27c1c8610c57e49c2dce67da917b366176465224.jpg)

![c5c6a82501b3bf755ca36889cc3a4adb30ed5e46000e535b7ac1783e6ffcad28.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/c5c6a82501b3bf755ca36889cc3a4adb30ed5e46000e535b7ac1783e6ffcad28.jpg)

![cb7673d14271ace822f15fbe9452b73fee2b9747e239175dd3ca33844a350d82.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/cb7673d14271ace822f15fbe9452b73fee2b9747e239175dd3ca33844a350d82.jpg)

![f633ce2e47b40f8c79218c3618501baf4033cdae2d52b976ea2b8fd051f1ff9b.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/images/f633ce2e47b40f8c79218c3618501baf4033cdae2d52b976ea2b8fd051f1ff9b.jpg)

### Tables

![1cfc4f6bc50e521651c438ff641710a39afdb6f68acc866600df868a15e222c6.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/1cfc4f6bc50e521651c438ff641710a39afdb6f68acc866600df868a15e222c6.jpg)

![1e6ba2ba032df3a800bc72be3e556d1bb1b5e1db7860cbcdbbc980bbf9cf9483.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/1e6ba2ba032df3a800bc72be3e556d1bb1b5e1db7860cbcdbbc980bbf9cf9483.jpg)

![23219f65a50ea3c86b8ecee71116fc03c0001ca0d38e6df0022fda5aeaeb41f9.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/23219f65a50ea3c86b8ecee71116fc03c0001ca0d38e6df0022fda5aeaeb41f9.jpg)

![3da2f5ce8891f3ef0327b2978cb7eb9a0ef9f0f15ece3dfe195875c9f943e813.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/3da2f5ce8891f3ef0327b2978cb7eb9a0ef9f0f15ece3dfe195875c9f943e813.jpg)

![42f3d86b14badaf3c8f73724d70e52e755c1f19ce41522bec720f48e719ee750.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/42f3d86b14badaf3c8f73724d70e52e755c1f19ce41522bec720f48e719ee750.jpg)

![9608b1e2031e488d2ee35ca16095115af55bb4ebaa1aed00fd697685fc5b780d.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/9608b1e2031e488d2ee35ca16095115af55bb4ebaa1aed00fd697685fc5b780d.jpg)

![daee9a3321dc79c828b78c02cedde7570d01e4ccb29a280ccb22d2ee4a9843b5.jpg](../icml_results/1557_Learning%20to%20Share_%20Selective%20Memory%20for%20Efficient%20Parallel%20Agentic%20Systems/tables/daee9a3321dc79c828b78c02cedde7570d01e4ccb29a280ccb22d2ee4a9843b5.jpg)

<a id="paper-1537"></a>

## Understanding Truncated Positional Encodings for Graph Neural Networks


### Images

![5c6bc3d6f143d18620a4843c328b9d507c8b07c11e3385f667308113c056879b.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/images/5c6bc3d6f143d18620a4843c328b9d507c8b07c11e3385f667308113c056879b.jpg)

![997e40eb59c6b6dcf3254494ae8b3e75d1f00b5cca911c77d0489394bb6b23c0.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/images/997e40eb59c6b6dcf3254494ae8b3e75d1f00b5cca911c77d0489394bb6b23c0.jpg)

![c098fce159a78902e891e80aca1a66ba1ca7de0a111addae3a019aa154e569f3.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/images/c098fce159a78902e891e80aca1a66ba1ca7de0a111addae3a019aa154e569f3.jpg)

![ed88d085c22ae4526652d81c50d62c9f168af46ab7bdab842a677dac5298b72d.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/images/ed88d085c22ae4526652d81c50d62c9f168af46ab7bdab842a677dac5298b72d.jpg)

### Tables

![2558df122adb46c167fe66919c86c75aff9a78b2b1d3556ce4793815397eeea3.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/2558df122adb46c167fe66919c86c75aff9a78b2b1d3556ce4793815397eeea3.jpg)

![766cab1df93ad5b05f571b837c0b97e7f61f517c538577c956e50a74338a6f4d.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/766cab1df93ad5b05f571b837c0b97e7f61f517c538577c956e50a74338a6f4d.jpg)

![7e878084b233291ff867c4d512f1dbf68bd5656b6dba76a7223fd55e34e2a37f.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/7e878084b233291ff867c4d512f1dbf68bd5656b6dba76a7223fd55e34e2a37f.jpg)

![88ba36fee3b1d723157b583a48d2a5d1f55b50d3ee601d13bfbec7a74a2797e4.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/88ba36fee3b1d723157b583a48d2a5d1f55b50d3ee601d13bfbec7a74a2797e4.jpg)

![aacd38026882c63c23d6845cb04ceb960d67b335a1ed82afd1dbd9076c43e448.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/aacd38026882c63c23d6845cb04ceb960d67b335a1ed82afd1dbd9076c43e448.jpg)

![ae630032254d075bd285d59204ca209519345406686bfbaeeffc46d0555f6a61.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/ae630032254d075bd285d59204ca209519345406686bfbaeeffc46d0555f6a61.jpg)

![e54880ff3434845bde56e8ebbb2b9ce2838a9b24b517ead765b15e186ff529d4.jpg](../icml_results/1558_Understanding%20Truncated%20Positional%20Encodings%20for%20Graph%20Neural%20Networks/tables/e54880ff3434845bde56e8ebbb2b9ce2838a9b24b517ead765b15e186ff529d4.jpg)

<a id="paper-1538"></a>

## On the Anisotropy of Score-Based Generative Models


### Images

![06696408be94a79b688169fc0ffc8e8127454233254c2d4806c3fa1f5a97f46f.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/06696408be94a79b688169fc0ffc8e8127454233254c2d4806c3fa1f5a97f46f.jpg)

![389f21718a7103ccda63a74603f45903e0fb377c082434b6d8ec1c6edc7ccdd9.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/389f21718a7103ccda63a74603f45903e0fb377c082434b6d8ec1c6edc7ccdd9.jpg)

![39b76661bcefa24e2dd55a1449f637dccf7376e012b8e37d3f6e056e97c8e4a9.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/39b76661bcefa24e2dd55a1449f637dccf7376e012b8e37d3f6e056e97c8e4a9.jpg)

![574a0a65fd39da7f58bf3854388af9c2837c47a5a0358d6362a5784726195129.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/574a0a65fd39da7f58bf3854388af9c2837c47a5a0358d6362a5784726195129.jpg)

![8cc12f8b74391dc3f0e88f3e6b8cf7241876e9e79b3b8e635f4bc295f20c20c0.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/8cc12f8b74391dc3f0e88f3e6b8cf7241876e9e79b3b8e635f4bc295f20c20c0.jpg)

![a1527e82a5b2a9774cff1109a3a30ec6798dc6c52f9ff001fada46b80c297704.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/a1527e82a5b2a9774cff1109a3a30ec6798dc6c52f9ff001fada46b80c297704.jpg)

![b04ef366ea88f68f08e749c22020429520e422558bbb8af1b5d86225c3b19bf1.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/b04ef366ea88f68f08e749c22020429520e422558bbb8af1b5d86225c3b19bf1.jpg)

![bfeae46e510571824fa6f894f5e698d6a6934fd9a2e9ca6d8682fdadfec523e1.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/bfeae46e510571824fa6f894f5e698d6a6934fd9a2e9ca6d8682fdadfec523e1.jpg)

![e47bc0038858c27e3064e4ca75d4003aaf0758dc5b08177381b06105789c3725.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/images/e47bc0038858c27e3064e4ca75d4003aaf0758dc5b08177381b06105789c3725.jpg)

### Tables

![aaede8f6715a84b30c3c30412989d6203e362256aa86f081ffcfcee2db235143.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/tables/aaede8f6715a84b30c3c30412989d6203e362256aa86f081ffcfcee2db235143.jpg)

![dfe2b2101d3f0de9ad3ef7974e01d47691f1b7c3fcece609e475b281c0d2fc94.jpg](../icml_results/1559_On%20the%20Anisotropy%20of%20Score-Based%20Generative%20Models/tables/dfe2b2101d3f0de9ad3ef7974e01d47691f1b7c3fcece609e475b281c0d2fc94.jpg)

<a id="paper-1539"></a>

## GenCircuit-RL: Reinforcement Learning from Hierarchical Verification for Genetic Circuit Design


### Images

![087a36a26eaa5bcd51120a05fdb33f95b773996c39909c9fc718d1d3e493e336.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/087a36a26eaa5bcd51120a05fdb33f95b773996c39909c9fc718d1d3e493e336.jpg)

![43d69d2a117b2983b9a69e9fa2552ff0888392108b5c8f97ba358cfd73717f66.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/43d69d2a117b2983b9a69e9fa2552ff0888392108b5c8f97ba358cfd73717f66.jpg)

![559e5178d27dfcb418db5302352c6224325dd0d3595806e79122cc4019c63c11.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/559e5178d27dfcb418db5302352c6224325dd0d3595806e79122cc4019c63c11.jpg)

![8b01d9df8a0d897a25ec8bf08f5b74b9e3c58165f5f084163e54411c24ac13fe.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/8b01d9df8a0d897a25ec8bf08f5b74b9e3c58165f5f084163e54411c24ac13fe.jpg)

![9f44d140894743af4c6a7d17f483095c2d1b7f4ce181aaae13a9eadc892606b5.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/9f44d140894743af4c6a7d17f483095c2d1b7f4ce181aaae13a9eadc892606b5.jpg)

![ad40187c5f61796eced21aca0ea20003f740227ee95b2d6765a552a516fa706d.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/ad40187c5f61796eced21aca0ea20003f740227ee95b2d6765a552a516fa706d.jpg)

![c0fba5050279b32d0bd66295326956ac4256e387fd6f641906c89d1316c2f820.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/c0fba5050279b32d0bd66295326956ac4256e387fd6f641906c89d1316c2f820.jpg)

![fa5178f4df567c44e938565dc0f8f5bcd92b5c2af0b83f5da796fece31b9bb62.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/images/fa5178f4df567c44e938565dc0f8f5bcd92b5c2af0b83f5da796fece31b9bb62.jpg)

### Tables

![0483ba026080a9472b40d4154f661bacdbaa4a2e1249bd3f6bc86b2135b2657b.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/0483ba026080a9472b40d4154f661bacdbaa4a2e1249bd3f6bc86b2135b2657b.jpg)

![08c99dba2434622a9e8f66acbb849e4c2f7b774e3ebf8849419aae37886f001a.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/08c99dba2434622a9e8f66acbb849e4c2f7b774e3ebf8849419aae37886f001a.jpg)

![0ead861f9bd9276b3e04781f9e0bb2dd07b61fda7218eecf757b7688d3cdf6da.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/0ead861f9bd9276b3e04781f9e0bb2dd07b61fda7218eecf757b7688d3cdf6da.jpg)

![13cb84532542f96a5f49bc7a0503e959cf07eaf150a0ad48f72d581489b61b88.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/13cb84532542f96a5f49bc7a0503e959cf07eaf150a0ad48f72d581489b61b88.jpg)

![1ab7ee4be147a16aa05b4f77f919e7ebfa0b539c8c8c384d39efefa0cb2a3699.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/1ab7ee4be147a16aa05b4f77f919e7ebfa0b539c8c8c384d39efefa0cb2a3699.jpg)

![1c92983c2fb470e3d80f0f946eb5e82c6f83e10040ad50db927cc9ca21e39543.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/1c92983c2fb470e3d80f0f946eb5e82c6f83e10040ad50db927cc9ca21e39543.jpg)

![1f63aa909c1ac0444ce79dc685a5b62e1423b285e85c7880a74ff0e248423c64.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/1f63aa909c1ac0444ce79dc685a5b62e1423b285e85c7880a74ff0e248423c64.jpg)

![217b18841fe0fc2fd2936b90e81d4515f4c2dfe8af041563aaba0b6acb447020.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/217b18841fe0fc2fd2936b90e81d4515f4c2dfe8af041563aaba0b6acb447020.jpg)

![2585f9c22be1de392b028577877ffd30ea2eaa6810a37dbc669aa9697c7ca764.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/2585f9c22be1de392b028577877ffd30ea2eaa6810a37dbc669aa9697c7ca764.jpg)

![26ded390f0acf56297bd8a0a914a2799900ef828ad0d1c74aac8e2b4c1e3061e.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/26ded390f0acf56297bd8a0a914a2799900ef828ad0d1c74aac8e2b4c1e3061e.jpg)

![2c98b066426956544e228c13c15220eff85cdf8d9f887a096bc972a722a25e06.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/2c98b066426956544e228c13c15220eff85cdf8d9f887a096bc972a722a25e06.jpg)

![2c9bd8bcffd75aad80f75d27325e85e54d04e3b295e773c43109ae62673e3e5f.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/2c9bd8bcffd75aad80f75d27325e85e54d04e3b295e773c43109ae62673e3e5f.jpg)

![35b3fad7d9e20b44dd70d91def1570572c1a65118226eb95b3c96148b37879fd.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/35b3fad7d9e20b44dd70d91def1570572c1a65118226eb95b3c96148b37879fd.jpg)

![365c63bc7650a60308b6e74620dbb8df7ce5dcd5a2eed45abe920a2d88f09870.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/365c63bc7650a60308b6e74620dbb8df7ce5dcd5a2eed45abe920a2d88f09870.jpg)

![38c61e2fca35fc162afab075d155160520201a6eaaefd303cf394240feece0c2.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/38c61e2fca35fc162afab075d155160520201a6eaaefd303cf394240feece0c2.jpg)

![3c957b9d8009fca5bf885816f2d84b409ce445c17bda6f641bb1064cfc680b10.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/3c957b9d8009fca5bf885816f2d84b409ce445c17bda6f641bb1064cfc680b10.jpg)

![3f607433ca8a3d8ca9f432cba3fbfe3ce7ba0c428d2b23fb8d21de502c82ef65.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/3f607433ca8a3d8ca9f432cba3fbfe3ce7ba0c428d2b23fb8d21de502c82ef65.jpg)

![42c98a346cbd660711037ddc4de94186ea52172108153f5fc076b410cce299a4.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/42c98a346cbd660711037ddc4de94186ea52172108153f5fc076b410cce299a4.jpg)

![53c94b3c21e7eab72c9d5123644816d949f5d7f405992b0570ef6d35229cc372.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/53c94b3c21e7eab72c9d5123644816d949f5d7f405992b0570ef6d35229cc372.jpg)

![53f2f3de4d3475eba6dc145568fbd543428a477f137c19d8ef41539ad87903d2.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/53f2f3de4d3475eba6dc145568fbd543428a477f137c19d8ef41539ad87903d2.jpg)

![583bbf1f3aab991a48835d4c647dfa82530c0bbf216633bbb08295dec4c255ac.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/583bbf1f3aab991a48835d4c647dfa82530c0bbf216633bbb08295dec4c255ac.jpg)

![5860bcc720cc6d37be9d28da442d7d8854f1200d19d02f2c9172ff83741fedd9.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/5860bcc720cc6d37be9d28da442d7d8854f1200d19d02f2c9172ff83741fedd9.jpg)

![5cef6f3f26ccc621203df70cca990a06a837a3deda4ff07dff79263f6da2329f.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/5cef6f3f26ccc621203df70cca990a06a837a3deda4ff07dff79263f6da2329f.jpg)

![612f43d531a975d67f81950ba567312b4357cdc4459dadf1334bd670057816ac.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/612f43d531a975d67f81950ba567312b4357cdc4459dadf1334bd670057816ac.jpg)

![62a514d57d96f3d6a591e42d4c1e70a5ccdb4ec847dc4f0729970ffa8012c422.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/62a514d57d96f3d6a591e42d4c1e70a5ccdb4ec847dc4f0729970ffa8012c422.jpg)

![6432918e3af0578234c7b2d4e81d1d181ecbc28fc610529a3b0e7f504a590070.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/6432918e3af0578234c7b2d4e81d1d181ecbc28fc610529a3b0e7f504a590070.jpg)

![69ec284b717c19a7a3dc79971903eee6837ec4cf8ca6438d89a29a75bb10ac57.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/69ec284b717c19a7a3dc79971903eee6837ec4cf8ca6438d89a29a75bb10ac57.jpg)

![6a5a0c2a47bf259189aa846915c64beb9dd05d6314e90c14b03127374553fe77.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/6a5a0c2a47bf259189aa846915c64beb9dd05d6314e90c14b03127374553fe77.jpg)

![6e4cb13326c2fa210bc1860f28c6a16c2e7cd534021dc92cb2f4fee8549ba3dd.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/6e4cb13326c2fa210bc1860f28c6a16c2e7cd534021dc92cb2f4fee8549ba3dd.jpg)

![7d38583d74ff781241c4e7ccc116c1e1e831189f6b1862eaeb8e6f288bec5bfe.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/7d38583d74ff781241c4e7ccc116c1e1e831189f6b1862eaeb8e6f288bec5bfe.jpg)

![846d8a88180bf1ab6c309a5da1a137a1e36858d2a5465bcaa07d425c94121bb0.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/846d8a88180bf1ab6c309a5da1a137a1e36858d2a5465bcaa07d425c94121bb0.jpg)

![856e526f7b93e143d4877bca00e441d8b677dd31877f6bfc4a410ad227547c81.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/856e526f7b93e143d4877bca00e441d8b677dd31877f6bfc4a410ad227547c81.jpg)

![8636b338aee7557f8ee28c465e679806b4f0067adbf24fcaf14f7fb34d51ec7e.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/8636b338aee7557f8ee28c465e679806b4f0067adbf24fcaf14f7fb34d51ec7e.jpg)

![8a01babefdce8c2fa37445e0cedce28838465b485d1ce3ff8c4e0f60aa6adcc9.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/8a01babefdce8c2fa37445e0cedce28838465b485d1ce3ff8c4e0f60aa6adcc9.jpg)

![9bed41fa97d34e7b2de3a7d52a9d12b76ad264f787334e55a9f55b4fb0776029.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/9bed41fa97d34e7b2de3a7d52a9d12b76ad264f787334e55a9f55b4fb0776029.jpg)

![a723992b0bf464811c6aea9a2c36d1a52d93cd7af7a822a149d913977328cd52.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/a723992b0bf464811c6aea9a2c36d1a52d93cd7af7a822a149d913977328cd52.jpg)

![ab6770dbc068e09b7def8d44501391e6a10d0d7c03f8eddb3932b5a247e0215d.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/ab6770dbc068e09b7def8d44501391e6a10d0d7c03f8eddb3932b5a247e0215d.jpg)

![ac21679dad5d5af5cec6072c3509b8057e83a5f849a566e3540689fa08b354a1.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/ac21679dad5d5af5cec6072c3509b8057e83a5f849a566e3540689fa08b354a1.jpg)

![ac5b5a10ec297d19019450a4ae554e74fcd668510d6322b33f8c3c6993b3ee40.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/ac5b5a10ec297d19019450a4ae554e74fcd668510d6322b33f8c3c6993b3ee40.jpg)

![ac8ab9474b22560e18b26f17cc63ef6c03246a3423e24ac71530ef2af3c78dc7.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/ac8ab9474b22560e18b26f17cc63ef6c03246a3423e24ac71530ef2af3c78dc7.jpg)

![b0f0aff66349b14a6a3ed9a3c812b8e64daec96076fbd2ace4a5e92cd7b82f17.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/b0f0aff66349b14a6a3ed9a3c812b8e64daec96076fbd2ace4a5e92cd7b82f17.jpg)

![b3fd19f020f8cd4aae53bf3b9925fbc5d362242da2e9ef7d6b2edf7e04fcf0ff.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/b3fd19f020f8cd4aae53bf3b9925fbc5d362242da2e9ef7d6b2edf7e04fcf0ff.jpg)

![b97a18476adcc0ff7ed889d403c54b2e2ac2c8656f59237fdaed483bb3fa8c01.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/b97a18476adcc0ff7ed889d403c54b2e2ac2c8656f59237fdaed483bb3fa8c01.jpg)

![bbb1f49989b52055cd6b4b117fc9623a2f043d7404aa833abcb1a02a1c49ef01.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/bbb1f49989b52055cd6b4b117fc9623a2f043d7404aa833abcb1a02a1c49ef01.jpg)

![be320353895b681b4c02d0ed2d3ba3002ec1ee9dc7ad8b1ce94d933f3a71f9ab.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/be320353895b681b4c02d0ed2d3ba3002ec1ee9dc7ad8b1ce94d933f3a71f9ab.jpg)

![c13046a6689103ccbe59516475ed65d7f363b924f827ca889ac7ac771fc2efbc.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/c13046a6689103ccbe59516475ed65d7f363b924f827ca889ac7ac771fc2efbc.jpg)

![c32c14d53bc7114cdcc5bc7e868c523ecd5792021185ef170c8d11fe615c5ae8.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/c32c14d53bc7114cdcc5bc7e868c523ecd5792021185ef170c8d11fe615c5ae8.jpg)

![c93c4d2e98985c6cbb2284a8e8559f9e02f62120156291b9ae3aca899b71708b.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/c93c4d2e98985c6cbb2284a8e8559f9e02f62120156291b9ae3aca899b71708b.jpg)

![cad0e08e42a7ce5fe2544e2ce056a27d813fc7ecfdf177a0ee237ba4a681893c.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/cad0e08e42a7ce5fe2544e2ce056a27d813fc7ecfdf177a0ee237ba4a681893c.jpg)

![cd9e4fe147e38498532218f79c38da17353bc3264387ca2eaf64e0040b859409.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/cd9e4fe147e38498532218f79c38da17353bc3264387ca2eaf64e0040b859409.jpg)

![d72ce1e208b67c10ee07a773dd01421458ceaad9deb0a380e228fade00dd3188.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/d72ce1e208b67c10ee07a773dd01421458ceaad9deb0a380e228fade00dd3188.jpg)

![da245878c1ed514f92afa6f05fd6e79c98046551d0051e30569f1dc3ba3e1dcd.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/da245878c1ed514f92afa6f05fd6e79c98046551d0051e30569f1dc3ba3e1dcd.jpg)

![e1c1fe35eb86f1106cb4e59b3492cb864e50fe969806b251c539f4f90fa65aa1.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/e1c1fe35eb86f1106cb4e59b3492cb864e50fe969806b251c539f4f90fa65aa1.jpg)

![e3c3f820e6fd3f94a465bbd6ef7c35e704f68885ef5929fd4b7842d78f056451.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/e3c3f820e6fd3f94a465bbd6ef7c35e704f68885ef5929fd4b7842d78f056451.jpg)

![e41f6509f52567f636bedea668d81347a611ee99248df750369948e7f792caf1.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/e41f6509f52567f636bedea668d81347a611ee99248df750369948e7f792caf1.jpg)

![e49d330d4175c13a2e9480dcaa51c84ce7871e5936d16650d1d6c15e0b93351e.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/e49d330d4175c13a2e9480dcaa51c84ce7871e5936d16650d1d6c15e0b93351e.jpg)

![e552e21199ef58eba5634883897461995b34d9f48da87d2780df18730dc58c2b.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/e552e21199ef58eba5634883897461995b34d9f48da87d2780df18730dc58c2b.jpg)

![eb65013b0bca38dbada33e05fce0971e1640e4efa613a67248a0154f22bde3d5.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/eb65013b0bca38dbada33e05fce0971e1640e4efa613a67248a0154f22bde3d5.jpg)

![ed906ffe7874c563dda983320263c67172641618551b811e48ca61fc18afbc5e.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/ed906ffe7874c563dda983320263c67172641618551b811e48ca61fc18afbc5e.jpg)

![f106124bec99854578c4f483993fc6d203bc7e507b2cb861f21a70c860cb54a1.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/f106124bec99854578c4f483993fc6d203bc7e507b2cb861f21a70c860cb54a1.jpg)

![f6a22e499bf53ba0082241c73327c69604a243e1dee193b0a1ea1c52ea58e130.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/f6a22e499bf53ba0082241c73327c69604a243e1dee193b0a1ea1c52ea58e130.jpg)

![f8c0f090681983c9f04cfb8b2fed539a8299858d6f1fe56f897e0b793d7cca17.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/f8c0f090681983c9f04cfb8b2fed539a8299858d6f1fe56f897e0b793d7cca17.jpg)

![fe95d580a0bebb5ca18c3700bd0288daf19d51be5dce1de957585c7a4e860c83.jpg](../icml_results/1560_GenCircuit-RL_%20Reinforcement%20Learning%20from%20Hierarchical%20Verification%20for%20Genetic%20Circuit%20Design/tables/fe95d580a0bebb5ca18c3700bd0288daf19d51be5dce1de957585c7a4e860c83.jpg)

<a id="paper-1540"></a>

## Posterior Sampling Reinforcement Learning with Gaussian Processes for Continuous Control: Sublinear Regret Bounds for Unbounded State Spaces


### Images

![2c0eadf3c3c04df760ace6b63002c85214364e481f2a542e92d33561073c103d.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/2c0eadf3c3c04df760ace6b63002c85214364e481f2a542e92d33561073c103d.jpg)

![480470f2bb784046413f717f001dbe00a52e50a314dc9ec2a7adb699c29dc382.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/480470f2bb784046413f717f001dbe00a52e50a314dc9ec2a7adb699c29dc382.jpg)

![4b96bc84dac3dd3772949ab703f6cfe12d4ad0e56823720ecfe6353f2f7d2c0f.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/4b96bc84dac3dd3772949ab703f6cfe12d4ad0e56823720ecfe6353f2f7d2c0f.jpg)

![5be98abd300426cef8b52a0dd4220359083e1a6869a1811820f1545ff9dc2f95.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/5be98abd300426cef8b52a0dd4220359083e1a6869a1811820f1545ff9dc2f95.jpg)

![83b64285a054a3fe587ab66371909d9f1e435f65e3e28c2edf14b9eb9421c3ad.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/83b64285a054a3fe587ab66371909d9f1e435f65e3e28c2edf14b9eb9421c3ad.jpg)

![a469676bbad48a44687381a1a66e0c004729476ac04d21237cb25a03468dd1d8.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/a469676bbad48a44687381a1a66e0c004729476ac04d21237cb25a03468dd1d8.jpg)

![a8f5180de28d5525377d089ad5d6aad6c612c55180d36e87c3be0f5ff6d43fde.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/a8f5180de28d5525377d089ad5d6aad6c612c55180d36e87c3be0f5ff6d43fde.jpg)

![bdfe86d447b96b0bd75e608279450c1835e3d43fedf1c83b5d75077c5246fc90.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/images/bdfe86d447b96b0bd75e608279450c1835e3d43fedf1c83b5d75077c5246fc90.jpg)

### Tables

![80b9d909ca8839823aba24ef25c08467ddc38c8bab3ebf92d87ec39b27ec4111.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/tables/80b9d909ca8839823aba24ef25c08467ddc38c8bab3ebf92d87ec39b27ec4111.jpg)

![8504ab5e5244668f5dccc77f4b83e42cc8e62d589e7dd66935db88784032c722.jpg](../icml_results/1561_Posterior%20Sampling%20Reinforcement%20Learning%20with%20Gaussian%20Processes%20for%20Continuous%20Control_%20Sublinear%20/tables/8504ab5e5244668f5dccc77f4b83e42cc8e62d589e7dd66935db88784032c722.jpg)

<a id="paper-1541"></a>

## Is Graph Mixup Beneficial? Investigating Interpolation And Empirical Performance of Graph Mixup Methods


### Images

![0a0117bd805a669aef7ff5491091b6830cc044d35e01422e778849bf3e884885.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/0a0117bd805a669aef7ff5491091b6830cc044d35e01422e778849bf3e884885.jpg)

![1685fc1ae473649c5846550430fc9011faf181a9ad4227dec5e2a48f3622ffa5.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/1685fc1ae473649c5846550430fc9011faf181a9ad4227dec5e2a48f3622ffa5.jpg)

![1905da9e51798ac7f99b376c2efe77b831cab4924f7b14d0d1c98df3706c2af5.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/1905da9e51798ac7f99b376c2efe77b831cab4924f7b14d0d1c98df3706c2af5.jpg)

![1b8c89dbff0690d0e55545df7ae0047a28db60b2a7972430f531f53c25eabff5.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/1b8c89dbff0690d0e55545df7ae0047a28db60b2a7972430f531f53c25eabff5.jpg)

![1cff65efca49d7ed7ad26b159c08b53c42f61bfdad6cbf32ce27f2919726e9c7.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/1cff65efca49d7ed7ad26b159c08b53c42f61bfdad6cbf32ce27f2919726e9c7.jpg)

![28d28481795c52b1199a98428fd11a0a408ec9c7c7e884cdf4435f77e702559f.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/28d28481795c52b1199a98428fd11a0a408ec9c7c7e884cdf4435f77e702559f.jpg)

![56755a7dbd3f4ac9bdb111be2be011eb8e8abdba785cbd7c09bc60bee4c8b760.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/56755a7dbd3f4ac9bdb111be2be011eb8e8abdba785cbd7c09bc60bee4c8b760.jpg)

![595ee91586b8defc4fc75dadb25ecaec26dd892b9c034c0f9ee14bf79f101ea0.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/595ee91586b8defc4fc75dadb25ecaec26dd892b9c034c0f9ee14bf79f101ea0.jpg)

![82415567b77c0a8214e4dd7c0b39ef19628c74785aa1dd25067f267449e7a0b8.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/82415567b77c0a8214e4dd7c0b39ef19628c74785aa1dd25067f267449e7a0b8.jpg)

![957a3936eeb2ceff271a632e17553235b690b5017d496ce1148386c753d23fc3.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/957a3936eeb2ceff271a632e17553235b690b5017d496ce1148386c753d23fc3.jpg)

![9c7a68f8bfe6dcf485b19638da9a84d93373c4aa365c419836c78cba20c2a4e4.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/9c7a68f8bfe6dcf485b19638da9a84d93373c4aa365c419836c78cba20c2a4e4.jpg)

![9e739f4c6c3f40185c66ab642bb17d293d38450bb24ad87432f364c30fd0d6d1.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/9e739f4c6c3f40185c66ab642bb17d293d38450bb24ad87432f364c30fd0d6d1.jpg)

![a74cd6ed4271e755d0ef241c4fbdf7b00ffb9d8e59ef2ecba95626a4b7af841d.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/a74cd6ed4271e755d0ef241c4fbdf7b00ffb9d8e59ef2ecba95626a4b7af841d.jpg)

![aaa051885ae56e0b2ff07b9f406f227b52a58614b4058602e8ed26d14ab2911c.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/aaa051885ae56e0b2ff07b9f406f227b52a58614b4058602e8ed26d14ab2911c.jpg)

![affe337ac6978b630fdbc607356430bd26ee320a181fb8489fc7e4cdf83955ae.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/affe337ac6978b630fdbc607356430bd26ee320a181fb8489fc7e4cdf83955ae.jpg)

![c4a5fd5a4fc2370fc8610a19972d940fe9e50a169c60d85b52ffe806cccb7813.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/c4a5fd5a4fc2370fc8610a19972d940fe9e50a169c60d85b52ffe806cccb7813.jpg)

![d489a8d2e12639b1a48f3577f5cadd5e9090c2e13b0347858047f573c62a031e.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/d489a8d2e12639b1a48f3577f5cadd5e9090c2e13b0347858047f573c62a031e.jpg)

![dda78031ea9b98e2ed99e7031796fd392a47d8459e7874666d911542e2dcf863.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/images/dda78031ea9b98e2ed99e7031796fd392a47d8459e7874666d911542e2dcf863.jpg)

### Tables

![4b468e384566c78a65034077f4942de7fb022cca660f554b7bfc542e35e35b69.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/4b468e384566c78a65034077f4942de7fb022cca660f554b7bfc542e35e35b69.jpg)

![4eecc500a7382e1ad7f0a7efe57966f290c592435d3d60eaeec1a627e6e2159e.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/4eecc500a7382e1ad7f0a7efe57966f290c592435d3d60eaeec1a627e6e2159e.jpg)

![82dbc33f84fc079f143fb6208d57e98e5c5adbe8a0a6df9ddc76b64675765854.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/82dbc33f84fc079f143fb6208d57e98e5c5adbe8a0a6df9ddc76b64675765854.jpg)

![8d6dd6b147927ac53d4214352e82c5df41e866efa7ec780b4c2d5ad3d33da02c.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/8d6dd6b147927ac53d4214352e82c5df41e866efa7ec780b4c2d5ad3d33da02c.jpg)

![970fc1d4b1f6954ff232d063ecf20ada38684dc11d0d74e042ca36c3a21196ee.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/970fc1d4b1f6954ff232d063ecf20ada38684dc11d0d74e042ca36c3a21196ee.jpg)

![99f240c32ffd53c83a001083859c3f7209d1f2ae82179ea015630072fc419b24.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/99f240c32ffd53c83a001083859c3f7209d1f2ae82179ea015630072fc419b24.jpg)

![9a90adbba847ca2a138b028e084b30fb344bb03df02a65905f146c9303619925.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/9a90adbba847ca2a138b028e084b30fb344bb03df02a65905f146c9303619925.jpg)

![a1a6c13e4d04c3413d3a0b79c952948054308c7daa63fd44a583aefcfa696aa4.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/a1a6c13e4d04c3413d3a0b79c952948054308c7daa63fd44a583aefcfa696aa4.jpg)

![bb3507d01a0b09ff61f1e28822ef6408679092e4e642b9ec9b71085529c798d0.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/bb3507d01a0b09ff61f1e28822ef6408679092e4e642b9ec9b71085529c798d0.jpg)

![c7a4a7a71fea6453bf68697438f536d224752e2412d922d804ecd6029f72ebb5.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/c7a4a7a71fea6453bf68697438f536d224752e2412d922d804ecd6029f72ebb5.jpg)

![e1f0fb4dd78b602872f9d63064a7072b1cd9cdf0c2c99ea58c981df5e1545679.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/e1f0fb4dd78b602872f9d63064a7072b1cd9cdf0c2c99ea58c981df5e1545679.jpg)

![ef9b53df7ae87cdb02deb7966e9a9fde0565596591645b89447e69d819e5f8b7.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/ef9b53df7ae87cdb02deb7966e9a9fde0565596591645b89447e69d819e5f8b7.jpg)

![eff3d51bd9c9bf922c0a4d6d3d35518edcb417d0b383e41ab6f2730a40f6a393.jpg](../icml_results/1562_Is%20Graph%20Mixup%20Beneficial_%20Investigating%20Interpolation%20And%20Empirical%20Performance%20of%20Graph%20Mixup%20Meth/tables/eff3d51bd9c9bf922c0a4d6d3d35518edcb417d0b383e41ab6f2730a40f6a393.jpg)

<a id="paper-1542"></a>

## Mixing Expertise with Confidence: A Mixture of Experts Framework for Robust Multi-Modal Continual Learning


### Images

![2d82039c1c09c77eba8dd3b49ab280802c5b842f309cf3da80829a629c7570ed.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/2d82039c1c09c77eba8dd3b49ab280802c5b842f309cf3da80829a629c7570ed.jpg)

![3745e871b1b4d10aa1a5fdf04629d4c16cd82f656fb44a7287af42616f3942d7.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/3745e871b1b4d10aa1a5fdf04629d4c16cd82f656fb44a7287af42616f3942d7.jpg)

![55b46ef916cb854a5d9c7636a43ea25512cfa81db62f8b42452e4d33dba63894.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/55b46ef916cb854a5d9c7636a43ea25512cfa81db62f8b42452e4d33dba63894.jpg)

![55c278a90e344b3c293abd932b865b14ce2f261c75fc47d8eadb0ef18823a927.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/55c278a90e344b3c293abd932b865b14ce2f261c75fc47d8eadb0ef18823a927.jpg)

![6eff21bb1703ddb6d7372f0fe7ac352e3de19746cb6052a53d95f1ea261aba9e.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/6eff21bb1703ddb6d7372f0fe7ac352e3de19746cb6052a53d95f1ea261aba9e.jpg)

![7549be50841abeff2a0e8182929867a9219edd12cf9a1a8a136055ab50cbcb58.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/7549be50841abeff2a0e8182929867a9219edd12cf9a1a8a136055ab50cbcb58.jpg)

![a226d76530920a7c6c9a2090a0f7dfd36d753a9283b2c3e6a5b27dbbff7f4b57.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/a226d76530920a7c6c9a2090a0f7dfd36d753a9283b2c3e6a5b27dbbff7f4b57.jpg)

![aa1dbebd510bdab2c86b0802bcd0ee6179768ced299d45b8925a6639abfc6732.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/aa1dbebd510bdab2c86b0802bcd0ee6179768ced299d45b8925a6639abfc6732.jpg)

![cf49f75073445eb2139f1c6d209be72e93ca34ff44f1468faf02caf4cf3e23d8.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/cf49f75073445eb2139f1c6d209be72e93ca34ff44f1468faf02caf4cf3e23d8.jpg)

![db36afae982302b4c1198ff34a39fe45b1f5453d917102153aad28b48efd2a5d.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/db36afae982302b4c1198ff34a39fe45b1f5453d917102153aad28b48efd2a5d.jpg)

![eee3f5e034742232c617307c0c14503691c0df92ed2b3188064030a34c9dd089.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/eee3f5e034742232c617307c0c14503691c0df92ed2b3188064030a34c9dd089.jpg)

![f02e82a817f08d149cd257b1d87c26b050bb93b336b4941f4986ea1cfe9d6b9c.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/images/f02e82a817f08d149cd257b1d87c26b050bb93b336b4941f4986ea1cfe9d6b9c.jpg)

### Tables

![1345039462366c877f4b19070dbd1b8d7214c9d06d95ac59357a6fe03d1c498c.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/1345039462366c877f4b19070dbd1b8d7214c9d06d95ac59357a6fe03d1c498c.jpg)

![1ff7b21fafcf5f86deca6f1391b07417ae399bb47d5dfcaf4e1675be548ece96.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/1ff7b21fafcf5f86deca6f1391b07417ae399bb47d5dfcaf4e1675be548ece96.jpg)

![2168fa959495145620b10210794453d8ad8ee8ab1716c863b9dacc0ee857c0f0.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/2168fa959495145620b10210794453d8ad8ee8ab1716c863b9dacc0ee857c0f0.jpg)

![29a0630fe6c562f7ec06ce0ac8f1a354e5df7fb317f223fc922f9457d99f463d.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/29a0630fe6c562f7ec06ce0ac8f1a354e5df7fb317f223fc922f9457d99f463d.jpg)

![370cf740c5c8205a516c45d8e8cce18c754580e8f55495a63f35b741b5c3b076.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/370cf740c5c8205a516c45d8e8cce18c754580e8f55495a63f35b741b5c3b076.jpg)

![59c4f88ab62df35d0a5a7c87d7f7174b5d83a3033784f313db7189c72561323e.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/59c4f88ab62df35d0a5a7c87d7f7174b5d83a3033784f313db7189c72561323e.jpg)

![5ed83686546483e3af68dee42c6bf48c2e07ddfc353e55e4cf665d3876eed99b.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/5ed83686546483e3af68dee42c6bf48c2e07ddfc353e55e4cf665d3876eed99b.jpg)

![61f32d02254fc50b0dd3d4a40ef0e4234410c3a71cc6a7fb986f481edd353838.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/61f32d02254fc50b0dd3d4a40ef0e4234410c3a71cc6a7fb986f481edd353838.jpg)

![88388e53045468d926f1399c6aaf769e06eae86705f9bd1e330224cde3dfce8e.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/88388e53045468d926f1399c6aaf769e06eae86705f9bd1e330224cde3dfce8e.jpg)

![8dce1dc973c4863be196c6c3b2354de9e4abbfac946c071847f891ca10416cf9.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/8dce1dc973c4863be196c6c3b2354de9e4abbfac946c071847f891ca10416cf9.jpg)

![9c4b2752f516b4e0e86d4a3edef557d642013a77986f82c825dd9c9a43cd136c.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/9c4b2752f516b4e0e86d4a3edef557d642013a77986f82c825dd9c9a43cd136c.jpg)

![a66faa465fb295e6250b6638006b419ac698236fb45bcd1cad522c2d7f022bea.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/a66faa465fb295e6250b6638006b419ac698236fb45bcd1cad522c2d7f022bea.jpg)

![ae83f24fce754fc6da034f3c305632cdb0c0a3ba18f2c5c512f764c58dbf4285.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/ae83f24fce754fc6da034f3c305632cdb0c0a3ba18f2c5c512f764c58dbf4285.jpg)

![b8fcb4f123edd89e09950de32a05f19922e7a1960d46f538b7259ef7f0f3d4d2.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/b8fcb4f123edd89e09950de32a05f19922e7a1960d46f538b7259ef7f0f3d4d2.jpg)

![baf5fccb772ec94b0c9fedac2d057f7f2a5bc26cb969252fcc8c5b3847664040.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/baf5fccb772ec94b0c9fedac2d057f7f2a5bc26cb969252fcc8c5b3847664040.jpg)

![c085b6768f479fbba2168492ef73a025bd634239c3c143067805e2b534396a71.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/c085b6768f479fbba2168492ef73a025bd634239c3c143067805e2b534396a71.jpg)

![e03cc28e12ad092d1a4dab5d43948c93b8f4dccf8d28dc39d9467361f6cb9e3f.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/e03cc28e12ad092d1a4dab5d43948c93b8f4dccf8d28dc39d9467361f6cb9e3f.jpg)

![f681fadcc1ee5e6b2061db429b2324b8f625821b764fcb0ed95f741a10adc099.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/f681fadcc1ee5e6b2061db429b2324b8f625821b764fcb0ed95f741a10adc099.jpg)

![f70a62048944066f02795819dc000cf7516d155151c1c1ab1656e26648cdf8ec.jpg](../icml_results/1563_Mixing%20Expertise%20with%20Confidence_%20A%20Mixture%20of%20Experts%20Framework%20for%20Robust%20Multi-Modal%20Continual%20Le/tables/f70a62048944066f02795819dc000cf7516d155151c1c1ab1656e26648cdf8ec.jpg)

<a id="paper-1543"></a>

## LINGUA: Bridging the Grounding Gap in VideoQA via Typed Memory and Belief-State Reasoning


### Images

![0adb22f84c1670e1f1f547c8990202f9ca9a7a36129b68e53c63f2d7d6f638f0.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/0adb22f84c1670e1f1f547c8990202f9ca9a7a36129b68e53c63f2d7d6f638f0.jpg)

![24574eb7e9dd87c34a2b56aad46022542a4017fa98fa1243d11f8ceb7c5025a5.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/24574eb7e9dd87c34a2b56aad46022542a4017fa98fa1243d11f8ceb7c5025a5.jpg)

![24a55fe1e6c8f2d0639c31e9f351b8be27ec77d5abd45902bd1df4758392948b.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/24a55fe1e6c8f2d0639c31e9f351b8be27ec77d5abd45902bd1df4758392948b.jpg)

![30dda8099b8ce3a37d107315a366da46f9a12751a1bba78806e1a088ed07ea83.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/30dda8099b8ce3a37d107315a366da46f9a12751a1bba78806e1a088ed07ea83.jpg)

![3f96acefffa3300781407e99d556cef3c4405f26977ba90ad89a6e6b94c1a987.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/3f96acefffa3300781407e99d556cef3c4405f26977ba90ad89a6e6b94c1a987.jpg)

![4f6be8f4cfd467ecae08c81a20b12634fe8883815ef64cf182f0b58f66cab5a5.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/4f6be8f4cfd467ecae08c81a20b12634fe8883815ef64cf182f0b58f66cab5a5.jpg)

![8b5ce3a04a5a14ddd6c9d5f07a6948cfc3dc7fe8a382c438b32221efca09318b.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/8b5ce3a04a5a14ddd6c9d5f07a6948cfc3dc7fe8a382c438b32221efca09318b.jpg)

![8d17403b480a63359b3253310b874894359ba264a00a6da7c4cba55e12723a5c.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/8d17403b480a63359b3253310b874894359ba264a00a6da7c4cba55e12723a5c.jpg)

![8fbeeb3c5517c48a26e1b116733b362e90ccf6b6173b2849a165c9fdea25507b.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/8fbeeb3c5517c48a26e1b116733b362e90ccf6b6173b2849a165c9fdea25507b.jpg)

![a92e5d614f674f38bebb62e4607a0396e55c2952b1b707814ca16c76f384abdf.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/a92e5d614f674f38bebb62e4607a0396e55c2952b1b707814ca16c76f384abdf.jpg)

![bd5c8d3001389b73598608a8cd4f2afb94989352a33630964b7c6efb21962c2c.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/bd5c8d3001389b73598608a8cd4f2afb94989352a33630964b7c6efb21962c2c.jpg)

![c3504446a86a2bb3ad3517546623bbad795891004ced917cc56efcff58278d25.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/c3504446a86a2bb3ad3517546623bbad795891004ced917cc56efcff58278d25.jpg)

![c9c8d030fbedd2a96fcdf683d8556db899df3524d4fba7a5a8dcca5d974350c0.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/c9c8d030fbedd2a96fcdf683d8556db899df3524d4fba7a5a8dcca5d974350c0.jpg)

![d8bf80838ba1fcabbd53117e76b62f7009d1835144585a7c6c7c78217b901844.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/d8bf80838ba1fcabbd53117e76b62f7009d1835144585a7c6c7c78217b901844.jpg)

![fbaf93c9970afb3daec4ea268fd52cc1e32c56dde6709c87837d37fb137eeb81.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/images/fbaf93c9970afb3daec4ea268fd52cc1e32c56dde6709c87837d37fb137eeb81.jpg)

### Tables

![0b8686cde195dab20694576028cc394e0d834e3de09cf8e79b30bdc44d0f488d.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/0b8686cde195dab20694576028cc394e0d834e3de09cf8e79b30bdc44d0f488d.jpg)

![0e98616127b435f4dc005b5146117065d71f961bcf49293f4973ca20629917f9.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/0e98616127b435f4dc005b5146117065d71f961bcf49293f4973ca20629917f9.jpg)

![10bebb5d6512ca581b11418432fd7a4100271fd2b73201737404a30dd8e2216c.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/10bebb5d6512ca581b11418432fd7a4100271fd2b73201737404a30dd8e2216c.jpg)

![1c1877cf745d0bd05495a19bab13b7842f5e81ddb504d0c0277ead464a4a6731.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/1c1877cf745d0bd05495a19bab13b7842f5e81ddb504d0c0277ead464a4a6731.jpg)

![225aa01f6ae10825e6392c5f8ad166f89df6c4605c137e035d91f852088b6dd5.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/225aa01f6ae10825e6392c5f8ad166f89df6c4605c137e035d91f852088b6dd5.jpg)

![318a4c60a5f511124872a34cdcce9ee92061f5eb06d1ba03308b45bde5fb8d3c.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/318a4c60a5f511124872a34cdcce9ee92061f5eb06d1ba03308b45bde5fb8d3c.jpg)

![347d69ebbe7ae7ebd1bc8e73af8146bec7b44f6b5a7d2572310bbe4e7f89906f.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/347d69ebbe7ae7ebd1bc8e73af8146bec7b44f6b5a7d2572310bbe4e7f89906f.jpg)

![46def7544221b2fb414b2254bd8d1378596c847f79ed863b4488d37077cb827a.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/46def7544221b2fb414b2254bd8d1378596c847f79ed863b4488d37077cb827a.jpg)

![55ca36d89898dabcacb33aea138cc4869eaa03c8121426124298e05312345b10.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/55ca36d89898dabcacb33aea138cc4869eaa03c8121426124298e05312345b10.jpg)

![573a419fa357227e8ae3364259839ab634a412a43d5ac74672b8f81fcc0c893c.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/573a419fa357227e8ae3364259839ab634a412a43d5ac74672b8f81fcc0c893c.jpg)

![5862309abd27979a64301c5ad0b18c7e93aecf721976bfa38168bd215ee611f1.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/5862309abd27979a64301c5ad0b18c7e93aecf721976bfa38168bd215ee611f1.jpg)

![5e1fa87fdb694ec26291b0d037b9fb8b2874535abd77efb44de8104eebe34855.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/5e1fa87fdb694ec26291b0d037b9fb8b2874535abd77efb44de8104eebe34855.jpg)

![61d8f51d691d5db46707ccced2ce5a9f7236dbb56a363e208661b42cac00d76d.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/61d8f51d691d5db46707ccced2ce5a9f7236dbb56a363e208661b42cac00d76d.jpg)

![637e70326713d01894c5443de19f5e6d590f8ead2b33dd7176b89dce3b2c47fc.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/637e70326713d01894c5443de19f5e6d590f8ead2b33dd7176b89dce3b2c47fc.jpg)

![69dea4f6efd8ab5cdc30854a88ac2c41d4512fbe8eac2020ac7453f590072ffc.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/69dea4f6efd8ab5cdc30854a88ac2c41d4512fbe8eac2020ac7453f590072ffc.jpg)

![71e874e980f10c103298d9d9b037324674e55fefe0d852fff0fbe8c9da2b2e53.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/71e874e980f10c103298d9d9b037324674e55fefe0d852fff0fbe8c9da2b2e53.jpg)

![9c6301141486dc105bfa45ac90748b60bb6862f8716f56b5a288a54167a2a54d.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/9c6301141486dc105bfa45ac90748b60bb6862f8716f56b5a288a54167a2a54d.jpg)

![9e28b73714f46c69f395b8363bef6168c4930dd2fbd6ca5b4f6c1406398263ff.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/9e28b73714f46c69f395b8363bef6168c4930dd2fbd6ca5b4f6c1406398263ff.jpg)

![a474d5206acdfcca93fc6ded3958859ca677c7a9afc423bced0a5056e70a46c0.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/a474d5206acdfcca93fc6ded3958859ca677c7a9afc423bced0a5056e70a46c0.jpg)

![af7796abf5553b03fa9c07b913fb9606821a58e0588e890569b503b39e0f6b47.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/af7796abf5553b03fa9c07b913fb9606821a58e0588e890569b503b39e0f6b47.jpg)

![b34fd5f54d5b7b5dc4fd3ccc7332861b19109dccfbbd83fea0f943d400fb4e11.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/b34fd5f54d5b7b5dc4fd3ccc7332861b19109dccfbbd83fea0f943d400fb4e11.jpg)

![b84c9a4a8d373d967bca1a19418abbbb3c862cd414c5b753a9fdfbfcccb29227.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/b84c9a4a8d373d967bca1a19418abbbb3c862cd414c5b753a9fdfbfcccb29227.jpg)

![d02acdd29c63e621b91df7b4d6ff8588446e6df65ac70723ec9bf0ff13930c98.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/d02acdd29c63e621b91df7b4d6ff8588446e6df65ac70723ec9bf0ff13930c98.jpg)

![d8f435630ae1ca1ef511a9deb19fc3f8573e037530d65a7cdeea51465327eeb7.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/d8f435630ae1ca1ef511a9deb19fc3f8573e037530d65a7cdeea51465327eeb7.jpg)

![dfc8b8f5ad954a6aea58b9791eea9ca86c8d6681a90cc2b1fcfb61677f5f931a.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/dfc8b8f5ad954a6aea58b9791eea9ca86c8d6681a90cc2b1fcfb61677f5f931a.jpg)

![e1d6c4739e3cb9f1f4a1da612ea413a5f90f1079389821e1210af59bdf2aeb15.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/e1d6c4739e3cb9f1f4a1da612ea413a5f90f1079389821e1210af59bdf2aeb15.jpg)

![e9191477525c02e4077c681f3c3cd2059b587a14584def3051b78ee426f665a8.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/e9191477525c02e4077c681f3c3cd2059b587a14584def3051b78ee426f665a8.jpg)

![f1f9af1f19f8017df84addba354a6787f551f9749eaccada9d9ff19724b281a9.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/f1f9af1f19f8017df84addba354a6787f551f9749eaccada9d9ff19724b281a9.jpg)

![faaf7dfec994d78c86237ee6fb84865ef8b6f1d6fea8dddbea9db7e4d733db07.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/faaf7dfec994d78c86237ee6fb84865ef8b6f1d6fea8dddbea9db7e4d733db07.jpg)

![fef7753b2f9c7d946cba18bed04a218b63ae9608a71e9c6e1a2d63919bcede78.jpg](../icml_results/1564_LINGUA_%20Bridging%20the%20Grounding%20Gap%20in%20VideoQA%20via%20Typed%20Memory%20and%20Belief-State%20Reasoning/tables/fef7753b2f9c7d946cba18bed04a218b63ae9608a71e9c6e1a2d63919bcede78.jpg)

<a id="paper-1544"></a>

## Temporal Difference Calibration in Sequential Tasks: Application to Vision-Language-Action Models


### Images

![15d8fd61113cf22bc8d30cabcf23f7ee7a2be69fd56849434b6aa5a114cb434a.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/15d8fd61113cf22bc8d30cabcf23f7ee7a2be69fd56849434b6aa5a114cb434a.jpg)

![23225540d980e179653eeb22b3f867c0f72ff1a225f331229ffb4cba026b747b.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/23225540d980e179653eeb22b3f867c0f72ff1a225f331229ffb4cba026b747b.jpg)

![3109893f50432ec7f57c1027f116fdfb7602c9b1f2f3fdb88adc055662ad1ce9.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/3109893f50432ec7f57c1027f116fdfb7602c9b1f2f3fdb88adc055662ad1ce9.jpg)

![3a84016e0d56bcb3c42baef8335084326df19aad574a864dc892c2ed4025ac11.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/3a84016e0d56bcb3c42baef8335084326df19aad574a864dc892c2ed4025ac11.jpg)

![3ab583c0783268e01b0cee8db69ea700fbcadae5f5b3d6ec8bae6604a022f8bd.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/3ab583c0783268e01b0cee8db69ea700fbcadae5f5b3d6ec8bae6604a022f8bd.jpg)

![47bc72dbf9c0bfd1f8f1cab4206a0849c7858dd1d0472fc336129895ba08460e.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/47bc72dbf9c0bfd1f8f1cab4206a0849c7858dd1d0472fc336129895ba08460e.jpg)

![4e72768b62276ae50803a31abb4bbdaf6111997aa639498635697af98161d4b2.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/4e72768b62276ae50803a31abb4bbdaf6111997aa639498635697af98161d4b2.jpg)

![5ae064e27f569f6885d79e3842d08ba86c2deab0f18475b6f5eb96b5527efa69.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/5ae064e27f569f6885d79e3842d08ba86c2deab0f18475b6f5eb96b5527efa69.jpg)

![6047d7194d06987e0c87fccc56a5e64832ef41206af22c376ab158da8d9fdf7d.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/6047d7194d06987e0c87fccc56a5e64832ef41206af22c376ab158da8d9fdf7d.jpg)

![76101e04d62cb73bfa7110acab6a900bd8a2474bc1a6fbac94c11d8ecaa70738.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/76101e04d62cb73bfa7110acab6a900bd8a2474bc1a6fbac94c11d8ecaa70738.jpg)

![849cd28feec339f8045575670861cc0b8f85982e1c8776a214c8674ca91ff18d.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/849cd28feec339f8045575670861cc0b8f85982e1c8776a214c8674ca91ff18d.jpg)

![8ab26ade1e38e385338953c319b4f473d7ba84d760aeff9ba27b1314ff685935.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/8ab26ade1e38e385338953c319b4f473d7ba84d760aeff9ba27b1314ff685935.jpg)

![90ca693cd44803cc9f801f900e27aee1d5e71566bf208a143243f94af9f383f2.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/90ca693cd44803cc9f801f900e27aee1d5e71566bf208a143243f94af9f383f2.jpg)

![a39fa4ea0092217baa3c71be738237fb99c695e6dedeab2b45fce4efd6365ce0.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/a39fa4ea0092217baa3c71be738237fb99c695e6dedeab2b45fce4efd6365ce0.jpg)

![b2e493a7615be632e257d8b4f6bc7c5c888bc0e2a76e42e692a88d4e87114e76.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/b2e493a7615be632e257d8b4f6bc7c5c888bc0e2a76e42e692a88d4e87114e76.jpg)

![b2f63292696bc7d0cb61068d5bad2e63dc03c5d9f04fbb1e89811583d35ff871.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/b2f63292696bc7d0cb61068d5bad2e63dc03c5d9f04fbb1e89811583d35ff871.jpg)

![b423d996e6c9dac1f1c627258cc07c678c94bef1685d1385d782ff8af8a9421d.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/b423d996e6c9dac1f1c627258cc07c678c94bef1685d1385d782ff8af8a9421d.jpg)

![b855297138546c8bfe860a35ea46a7ce36d606824d384c08d296245093dd465a.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/b855297138546c8bfe860a35ea46a7ce36d606824d384c08d296245093dd465a.jpg)

![c24c3a7343926431fb40c02349a4fa015ae74bf4b094b213377637518d4f4a19.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/c24c3a7343926431fb40c02349a4fa015ae74bf4b094b213377637518d4f4a19.jpg)

![ca57cb9f30386b905c858e790c9e35237c6dfea164cf248b34ef722d95a94abf.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/ca57cb9f30386b905c858e790c9e35237c6dfea164cf248b34ef722d95a94abf.jpg)

![ce5c1a4132f0874e7c1a73d29a7c84ef8b6bed39b6d47e7b17d78f44207a0755.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/ce5c1a4132f0874e7c1a73d29a7c84ef8b6bed39b6d47e7b17d78f44207a0755.jpg)

![dd446225061ec49242c980e52111302c76e13251d6dd8f007441bea5ba23a358.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/images/dd446225061ec49242c980e52111302c76e13251d6dd8f007441bea5ba23a358.jpg)

### Tables

![0cd106d1a862789bca9abe9ced621b1128481eb6c33321b5c573b495e5175be3.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/0cd106d1a862789bca9abe9ced621b1128481eb6c33321b5c573b495e5175be3.jpg)

![22278bef36916fc30e2dbdbfcbcd8729d505e45df420137f5eb710b5ec328a38.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/22278bef36916fc30e2dbdbfcbcd8729d505e45df420137f5eb710b5ec328a38.jpg)

![3320ed4dd0b0ce59c6430f0168eaf18b182ffcab5056f4216e43bb4ce332a48b.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/3320ed4dd0b0ce59c6430f0168eaf18b182ffcab5056f4216e43bb4ce332a48b.jpg)

![478b6c9323bb7543b55fabcb58c8926c36917df0aed470c90c39c0a0eed27a49.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/478b6c9323bb7543b55fabcb58c8926c36917df0aed470c90c39c0a0eed27a49.jpg)

![608d9a12b0b904107e5186c4643bed0e5df2e8544dbc93e529cf38bde5990818.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/608d9a12b0b904107e5186c4643bed0e5df2e8544dbc93e529cf38bde5990818.jpg)

![609c3b2284e8f111b4da1e6ef1262497a88b0f89ffd61228ac3a767fe2a056eb.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/609c3b2284e8f111b4da1e6ef1262497a88b0f89ffd61228ac3a767fe2a056eb.jpg)

![7b412409dc3395a7e2abc2ddf648a637dd820435da6da5b9023716399a456583.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/7b412409dc3395a7e2abc2ddf648a637dd820435da6da5b9023716399a456583.jpg)

![8d4ba8172c70f25d09ab3b69cdb30b1f830309fc3af53d1872b7e8db88884db5.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/8d4ba8172c70f25d09ab3b69cdb30b1f830309fc3af53d1872b7e8db88884db5.jpg)

![8eacba6fc213c10f5fa9c670c211533197713bce3b99f855af366d92672229b4.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/8eacba6fc213c10f5fa9c670c211533197713bce3b99f855af366d92672229b4.jpg)

![8f3ae10c6165a057d4dbae2ada81fe8c090d3e4789d524c0c1a997788aca1541.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/8f3ae10c6165a057d4dbae2ada81fe8c090d3e4789d524c0c1a997788aca1541.jpg)

![9aed22da89452a53385d7775e957a91fea5c611c6014989d1411087146e136a8.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/9aed22da89452a53385d7775e957a91fea5c611c6014989d1411087146e136a8.jpg)

![9ff6011b270d580f1bf185ae310af73984225c00849772b78241da32be473ba1.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/9ff6011b270d580f1bf185ae310af73984225c00849772b78241da32be473ba1.jpg)

![a73ec1766046edcd145df8e64a13e5513a0010420b5e6b08f9a739ccb62c9a1b.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/a73ec1766046edcd145df8e64a13e5513a0010420b5e6b08f9a739ccb62c9a1b.jpg)

![bcc695b0719859e325e9ae367a509ed6b3d65882f46fc6764617ba9edc9636c6.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/bcc695b0719859e325e9ae367a509ed6b3d65882f46fc6764617ba9edc9636c6.jpg)

![ef06d967c0124627dab1fb36e0732ad60b7753b63430c86dfc714157ab92a08f.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/ef06d967c0124627dab1fb36e0732ad60b7753b63430c86dfc714157ab92a08f.jpg)

![fa1a6a1faf246f1b626c5fe552c519125c97cfa0ee17bb73f60efe590ae8571f.jpg](../icml_results/1565_Temporal%20Difference%20Calibration%20in%20Sequential%20Tasks_%20Application%20to%20Vision-Language-Action%20Models/tables/fa1a6a1faf246f1b626c5fe552c519125c97cfa0ee17bb73f60efe590ae8571f.jpg)

<a id="paper-1545"></a>

## Singular Vectors of Attention Heads Align with Features


### Images

![02e2dd1600a882df13eaa889c360fcfb0da96aab16f9af25d930d73ba73cae61.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/02e2dd1600a882df13eaa889c360fcfb0da96aab16f9af25d930d73ba73cae61.jpg)

![11f8548ae8f8821fef25f29ea8ba93ee1f82723fbff3a19fc18cc6f2b40a671f.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/11f8548ae8f8821fef25f29ea8ba93ee1f82723fbff3a19fc18cc6f2b40a671f.jpg)

![19f4bfce1dfcdc5b89eb0a3f5527b3d5d09e94e3416ccd2d1898d68c6141a7d6.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/19f4bfce1dfcdc5b89eb0a3f5527b3d5d09e94e3416ccd2d1898d68c6141a7d6.jpg)

![2f76b381d012d1fbb6afdabf1b5f74e7ba75f4f3a5016c54ecf58722292afa0a.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/2f76b381d012d1fbb6afdabf1b5f74e7ba75f4f3a5016c54ecf58722292afa0a.jpg)

![364f72688b095c0ad1e1530507e0c60b2c464b4618b6ffafa4f9800ab5cb6460.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/364f72688b095c0ad1e1530507e0c60b2c464b4618b6ffafa4f9800ab5cb6460.jpg)

![3ea17ec3ffbc21ec96c275504053148a033dce2accdcf286f9448c7a62eb3f30.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/3ea17ec3ffbc21ec96c275504053148a033dce2accdcf286f9448c7a62eb3f30.jpg)

![513191d9e69abd60a8d95a268e7fa7217ad43a45cd62685f6877cda1b21964b7.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/513191d9e69abd60a8d95a268e7fa7217ad43a45cd62685f6877cda1b21964b7.jpg)

![538d1ca5869fe305ffaf995e83c7b9790d58094bf48fc2740c7b36ac941b3ea2.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/538d1ca5869fe305ffaf995e83c7b9790d58094bf48fc2740c7b36ac941b3ea2.jpg)

![5501a627913861743a1315d55b89afbb67d793df695b444ba483b580e77391e0.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/5501a627913861743a1315d55b89afbb67d793df695b444ba483b580e77391e0.jpg)

![555b8983778b58f2f99b163ac68a72e83f2287a0380ea9c549fd3773d43278b4.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/555b8983778b58f2f99b163ac68a72e83f2287a0380ea9c549fd3773d43278b4.jpg)

![76379341b437b99376aabb47a4d176eaa0fbee058b1c692058ee7f4724a05f0b.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/76379341b437b99376aabb47a4d176eaa0fbee058b1c692058ee7f4724a05f0b.jpg)

![80a39da643bbf2efdbbb0f28bc88f642ab70f2de0f7abce3260eb39366c79ae0.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/80a39da643bbf2efdbbb0f28bc88f642ab70f2de0f7abce3260eb39366c79ae0.jpg)

![842ffd7f932061a4d571298cfd60102c968bd4ccb65c133ebfae5ee9d3471ace.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/842ffd7f932061a4d571298cfd60102c968bd4ccb65c133ebfae5ee9d3471ace.jpg)

![8f6abe1393bc20ec740639c5a39107f29ed2fdf8bf62ae1fea99f7126c781275.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/8f6abe1393bc20ec740639c5a39107f29ed2fdf8bf62ae1fea99f7126c781275.jpg)

![97a89dbc56f2f633f01b871f077b734c31ce4d5f0f7bdab2d2e8545174305774.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/97a89dbc56f2f633f01b871f077b734c31ce4d5f0f7bdab2d2e8545174305774.jpg)

![9be3bca104e02e83f491fe28c61818ee0fa55b585713cfa980a47c9b4c7b302b.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/9be3bca104e02e83f491fe28c61818ee0fa55b585713cfa980a47c9b4c7b302b.jpg)

![a534e6d98379620ddd2e1395a850da285fe42bb8ac2ab425fdc84bd08c404bea.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/a534e6d98379620ddd2e1395a850da285fe42bb8ac2ab425fdc84bd08c404bea.jpg)

![c652114342968f3bde67e4fdbdc85919330bbac9655d49ddf77ecf0e383d9171.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/c652114342968f3bde67e4fdbdc85919330bbac9655d49ddf77ecf0e383d9171.jpg)

![cd541efd6cb196db6ecbf321982d070e31809d9cc81cbbfab4ebfdd59f14bfda.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/cd541efd6cb196db6ecbf321982d070e31809d9cc81cbbfab4ebfdd59f14bfda.jpg)

![d1852a0c2f87944f02f034546fc5b350066e913964cad6629a9402cd5a64f6a6.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/d1852a0c2f87944f02f034546fc5b350066e913964cad6629a9402cd5a64f6a6.jpg)

![d6582af4d8a366e00672c6c6223b55d19ffd5ce06ad1f453d7c64e7657ddf4d6.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/d6582af4d8a366e00672c6c6223b55d19ffd5ce06ad1f453d7c64e7657ddf4d6.jpg)

![da6d655123597bd5b8aab6fb528fdd4242fde7724a11e5bec747ca5528b8a0a7.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/da6d655123597bd5b8aab6fb528fdd4242fde7724a11e5bec747ca5528b8a0a7.jpg)

![ee2af73fefa20f0f58e5db5edd0c770514cb4a73e015b71c276fc6f0befe7884.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/ee2af73fefa20f0f58e5db5edd0c770514cb4a73e015b71c276fc6f0befe7884.jpg)

![f97104fcc7f6964cef87c0589672c212512321571f199cb133ccf761ad8a16ad.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/f97104fcc7f6964cef87c0589672c212512321571f199cb133ccf761ad8a16ad.jpg)

![fb0a5521b56d1bba0cfc9eaa7ad28745d0ae1837b9d62c6f7497d44db1740128.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/images/fb0a5521b56d1bba0cfc9eaa7ad28745d0ae1837b9d62c6f7497d44db1740128.jpg)

### Tables

![22ce068db769e17aad3b39e51a0401e0c68aa5984754f8c0682f4090afb50d11.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/tables/22ce068db769e17aad3b39e51a0401e0c68aa5984754f8c0682f4090afb50d11.jpg)

![c6e1398cebf51c804b1866330b9d23ed9d8808c77ee499c407cec7286ae6addf.jpg](../icml_results/1566_Singular%20Vectors%20of%20Attention%20Heads%20Align%20with%20Features/tables/c6e1398cebf51c804b1866330b9d23ed9d8808c77ee499c407cec7286ae6addf.jpg)

<a id="paper-1546"></a>

## Complexity Bounds for Dirichlet Process Slice Samplers


### Images

![058ff2c1f6664adb968149387ea3b3122659392a49e14268a340653818e9bdbc.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/058ff2c1f6664adb968149387ea3b3122659392a49e14268a340653818e9bdbc.jpg)

![090fc32645159acfd007e33c254f7f27d11d403a1dad903ec4f7ad105ec1b88c.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/090fc32645159acfd007e33c254f7f27d11d403a1dad903ec4f7ad105ec1b88c.jpg)

![3ba2090f50ea03082d25a591c028e7e4d387e8248a55a909f71008d677d1dc6c.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/3ba2090f50ea03082d25a591c028e7e4d387e8248a55a909f71008d677d1dc6c.jpg)

![45483a3fdd921b7744cc9e95936ba620fc7b2fe28b102f1c8fe51fbb96380439.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/45483a3fdd921b7744cc9e95936ba620fc7b2fe28b102f1c8fe51fbb96380439.jpg)

![63b6c9f06a742273afaed2fae5fdc24ba0253172ae9aad301ad96ea53ddce7bc.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/63b6c9f06a742273afaed2fae5fdc24ba0253172ae9aad301ad96ea53ddce7bc.jpg)

![6dd77ae1f74321881b100784beb02a653c2afd704d23668852d89a9e25defb00.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/6dd77ae1f74321881b100784beb02a653c2afd704d23668852d89a9e25defb00.jpg)

![7cdaa06108e9c250c082a57754b47ecddf5a7b8d5578f2ac0669962e6e83e752.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/7cdaa06108e9c250c082a57754b47ecddf5a7b8d5578f2ac0669962e6e83e752.jpg)

![91ec2b028a08aeb5ebd69875bae8e8b1692a7beff0c53783b68498ddcef2509d.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/91ec2b028a08aeb5ebd69875bae8e8b1692a7beff0c53783b68498ddcef2509d.jpg)

![95ed0b7b144f25b569f95d8eb4053ef05d70970fd4bc1b7977b4cc377e8697cb.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/95ed0b7b144f25b569f95d8eb4053ef05d70970fd4bc1b7977b4cc377e8697cb.jpg)

![ab3121c18b67b1e143502f27cfa4fef7d5a053daadd6963cf3e3df619eb1aaa0.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/ab3121c18b67b1e143502f27cfa4fef7d5a053daadd6963cf3e3df619eb1aaa0.jpg)

![abfd672c18fb5c86d1262a71c7a8b158f948b3b9aef6dcf2a427f52e6adec0d5.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/abfd672c18fb5c86d1262a71c7a8b158f948b3b9aef6dcf2a427f52e6adec0d5.jpg)

![ad359fad590d2bcd54d3c5d488dd120d940ebf0f0f17a5a05b6e77bbfa2e0042.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/ad359fad590d2bcd54d3c5d488dd120d940ebf0f0f17a5a05b6e77bbfa2e0042.jpg)

![b52e2441100b93db981452a0be68e10b29ddc70c4e76ce516f61a81dbb34493d.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/b52e2441100b93db981452a0be68e10b29ddc70c4e76ce516f61a81dbb34493d.jpg)

![b543f3046576f5fae1d7a3d8555ac746b6be2e9b2eec96da5e22cccd4f26b384.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/b543f3046576f5fae1d7a3d8555ac746b6be2e9b2eec96da5e22cccd4f26b384.jpg)

![cd2eb0e18b750835b3b17ef7ddbe70c76f05ea7e8f23651cec94fa6d3321fabe.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/cd2eb0e18b750835b3b17ef7ddbe70c76f05ea7e8f23651cec94fa6d3321fabe.jpg)

![dc287407db3e596eb8e86d6a2f73bc33d928fb97b08cb26b4e8368a179e6ad23.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/images/dc287407db3e596eb8e86d6a2f73bc33d928fb97b08cb26b4e8368a179e6ad23.jpg)

### Tables

![31a51f765e006711bddb119bc01a934e26b7289c905946330b24d86a13066f8b.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/tables/31a51f765e006711bddb119bc01a934e26b7289c905946330b24d86a13066f8b.jpg)

![814e7d7f901bc8536c84c3124436611886ca7e548756182bd46ef3223ead94e0.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/tables/814e7d7f901bc8536c84c3124436611886ca7e548756182bd46ef3223ead94e0.jpg)

![857c3264f8498d7bde5b68097cd1a8cb0f0bdbd836e51dac18b5388101d0ba3a.jpg](../icml_results/1567_Complexity%20Bounds%20for%20Dirichlet%20Process%20Slice%20Samplers/tables/857c3264f8498d7bde5b68097cd1a8cb0f0bdbd836e51dac18b5388101d0ba3a.jpg)

<a id="paper-1547"></a>

## Nonparametric LLM Evaluation from Preference Data


### Images

![1bf3523b082147d9aa80178fc995c557aaec7b79dd059e6cba7dce72498af717.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/1bf3523b082147d9aa80178fc995c557aaec7b79dd059e6cba7dce72498af717.jpg)

![1c448f37150eb66740fa87129ece15ee9b232ccc2005fe5aa461d17b637891af.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/1c448f37150eb66740fa87129ece15ee9b232ccc2005fe5aa461d17b637891af.jpg)

![527a5875a46536d2000e4c6907b75473b9d851e96644874819f35a73dc2ea866.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/527a5875a46536d2000e4c6907b75473b9d851e96644874819f35a73dc2ea866.jpg)

![598a6647f7bb5d43405ad6467c3ade133facbaa18d797f4268fd575b2752058e.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/598a6647f7bb5d43405ad6467c3ade133facbaa18d797f4268fd575b2752058e.jpg)

![827f8b4925635b759cd656f8bb411c60298f2a142d6b16249c0d2dccbec72137.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/827f8b4925635b759cd656f8bb411c60298f2a142d6b16249c0d2dccbec72137.jpg)

![9a551109acf93283591df977be222b67ba58f5f3fa57a86a0b7f2346b3ab0883.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/images/9a551109acf93283591df977be222b67ba58f5f3fa57a86a0b7f2346b3ab0883.jpg)

### Tables

![70171a2d67f17a7c2c55fcc32344ed33a8b6b31f4ba11bfe6dff96e64d56f501.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/70171a2d67f17a7c2c55fcc32344ed33a8b6b31f4ba11bfe6dff96e64d56f501.jpg)

![a5d5ad08911e21989ab5913db634d4c22b6dedbd516429015b62eca47da8ad90.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/a5d5ad08911e21989ab5913db634d4c22b6dedbd516429015b62eca47da8ad90.jpg)

![c366753eb0e1ad6d3dbcdad25555791c428405a717ca1cd20968660c41a26dee.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/c366753eb0e1ad6d3dbcdad25555791c428405a717ca1cd20968660c41a26dee.jpg)

![cbb0fb569d0a1b4c3c4d0e3a77b3bb89e9b0c0ae1807458aae5e0679a077ed85.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/cbb0fb569d0a1b4c3c4d0e3a77b3bb89e9b0c0ae1807458aae5e0679a077ed85.jpg)

![d8dc7752105e7bba629ea01dd727e058613f1d36841c5f366615b67c00f6e480.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/d8dc7752105e7bba629ea01dd727e058613f1d36841c5f366615b67c00f6e480.jpg)

![dca339777ec9e303e4d09c75e0376434ec6fae0685d58e105e2e4e1339d12514.jpg](../icml_results/1568_Nonparametric%20LLM%20Evaluation%20from%20Preference%20Data/tables/dca339777ec9e303e4d09c75e0376434ec6fae0685d58e105e2e4e1339d12514.jpg)

<a id="paper-1548"></a>

## Symbol-Equivariant Recurrent Reasoning Models


### Images

![05c810c9ba04697dcc911628f1e7d0be5f49b6516361fc8a37412acdae39f2eb.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/images/05c810c9ba04697dcc911628f1e7d0be5f49b6516361fc8a37412acdae39f2eb.jpg)

![20051fb9bfe2164f0f9d389777c06d461084f1ab30a2f4a64dfec36204cbeb2d.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/images/20051fb9bfe2164f0f9d389777c06d461084f1ab30a2f4a64dfec36204cbeb2d.jpg)

![4cd080552e9c89663cd4537035d57807541da8699c49090d1846a85198831478.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/images/4cd080552e9c89663cd4537035d57807541da8699c49090d1846a85198831478.jpg)

![5977131c4741bbb83f13583db8c1542acc339e9a2d5d2d9c7e9aad87f09a0c60.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/images/5977131c4741bbb83f13583db8c1542acc339e9a2d5d2d9c7e9aad87f09a0c60.jpg)

### Tables

![027045d46ca24dcf19ec750412365e16db4cd0bc1025083eb8c8d1d5b22930b0.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/027045d46ca24dcf19ec750412365e16db4cd0bc1025083eb8c8d1d5b22930b0.jpg)

![1d8d1c62f7bc3e4a2a54fd893f8afdf313aac7c6f9d94d45d0b71fe67bc3ab06.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/1d8d1c62f7bc3e4a2a54fd893f8afdf313aac7c6f9d94d45d0b71fe67bc3ab06.jpg)

![2432e82965b9a2170d719c9fb2cb9ec902927fcc82e92b5e1afb62e3ff02650b.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/2432e82965b9a2170d719c9fb2cb9ec902927fcc82e92b5e1afb62e3ff02650b.jpg)

![357edc2cd2fcb38958bfda8dceb261446bedcc3fe879e8e70b8ea819ec157d2b.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/357edc2cd2fcb38958bfda8dceb261446bedcc3fe879e8e70b8ea819ec157d2b.jpg)

![588b4d2637b902ac25234851b29d5fdc743830892045f4e493ba89a4e45a7498.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/588b4d2637b902ac25234851b29d5fdc743830892045f4e493ba89a4e45a7498.jpg)

![68bde05509833decaea99138d150dddadefd4a89c51d83e72f56e4c00122a250.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/68bde05509833decaea99138d150dddadefd4a89c51d83e72f56e4c00122a250.jpg)

![6ab73808db6bf08b0553e500b72a911064843f107643442a5a64ab9fdfd85b7d.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/6ab73808db6bf08b0553e500b72a911064843f107643442a5a64ab9fdfd85b7d.jpg)

![78d262b07ccb8884b80b4c53ea7dcfa620fe3e875faf018dbbd01367b1156ee4.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/78d262b07ccb8884b80b4c53ea7dcfa620fe3e875faf018dbbd01367b1156ee4.jpg)

![b5228421b2e338e36d95c041b591c33d9449ce71718077e6be0ec99e3afab9ba.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/b5228421b2e338e36d95c041b591c33d9449ce71718077e6be0ec99e3afab9ba.jpg)

![e215e9bcba29e49c682067dedcb8b49374463b850dc0472c8abd214b5536d4d9.jpg](../icml_results/1569_Symbol-Equivariant%20Recurrent%20Reasoning%20Models/tables/e215e9bcba29e49c682067dedcb8b49374463b850dc0472c8abd214b5536d4d9.jpg)

<a id="paper-1549"></a>

## Learning Hamiltonian Dynamics at Scale: A Differential-Geometric Approach


### Images

![01376ddaac9cf6f0132a0cb76cf089281e21978be7dd2da1f795cd63521d2b61.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/01376ddaac9cf6f0132a0cb76cf089281e21978be7dd2da1f795cd63521d2b61.jpg)

![04fd4381baea34c9e4012b6db358726fc8eae7138094fce3b10bf166be1461a4.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/04fd4381baea34c9e4012b6db358726fc8eae7138094fce3b10bf166be1461a4.jpg)

![05c4fd236bf3dcc8bfb2a25613b62107a787617ef06e98cc2ad1abe75d577720.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/05c4fd236bf3dcc8bfb2a25613b62107a787617ef06e98cc2ad1abe75d577720.jpg)

![0f48fbacc9351b48a07b537726e4ed07e7df270ba139aeccde833ae47982227d.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/0f48fbacc9351b48a07b537726e4ed07e7df270ba139aeccde833ae47982227d.jpg)

![26181816c566e7d0e3e7b559abcee51ba178ec20109a1f8b619c03c30aef9530.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/26181816c566e7d0e3e7b559abcee51ba178ec20109a1f8b619c03c30aef9530.jpg)

![3dda939b8a4700bdbd897adb3cf24f913794a23fdae50500b8aff80388f6a5d8.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/3dda939b8a4700bdbd897adb3cf24f913794a23fdae50500b8aff80388f6a5d8.jpg)

![42fbed4fd8326cfec43b941ba0cf495199202ba69d2364cc49ce16489043f653.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/42fbed4fd8326cfec43b941ba0cf495199202ba69d2364cc49ce16489043f653.jpg)

![54c81494a7a8683983d265f6c2923326f9db14f486e014621c5bf4d3fdfd8269.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/54c81494a7a8683983d265f6c2923326f9db14f486e014621c5bf4d3fdfd8269.jpg)

![679308a4eae54331e559268e9f03e73fdc49039d74fbf6e4674f01fc31f2a3e3.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/679308a4eae54331e559268e9f03e73fdc49039d74fbf6e4674f01fc31f2a3e3.jpg)

![698e3db8e058770ebbd706c045632743ee3a1a220e8e2c9a8c07cf87a4b5d3be.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/698e3db8e058770ebbd706c045632743ee3a1a220e8e2c9a8c07cf87a4b5d3be.jpg)

![706bd51edf2935262eb58e3dd9bcc2fcce83a2c17b0e5d65a16b32a74858c037.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/706bd51edf2935262eb58e3dd9bcc2fcce83a2c17b0e5d65a16b32a74858c037.jpg)

![7fc5cbf434f974bfb86ea647adba031d5d0712325b0154abf88839246f2906ae.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/7fc5cbf434f974bfb86ea647adba031d5d0712325b0154abf88839246f2906ae.jpg)

![82eabcb37ff1c5dfd3162414140db99a51cc5a1f0f528a5092106cdf663e945f.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/82eabcb37ff1c5dfd3162414140db99a51cc5a1f0f528a5092106cdf663e945f.jpg)

![949f9bb117717b3ac17316b1c8353f7bfd4314e5812f774935417cbd5991b69a.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/949f9bb117717b3ac17316b1c8353f7bfd4314e5812f774935417cbd5991b69a.jpg)

![959b0193e899bfa80864fc665b66c2fb7dce3d8b369e311bc421769f783c8451.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/959b0193e899bfa80864fc665b66c2fb7dce3d8b369e311bc421769f783c8451.jpg)

![b021bdb269bced5222710819c28f17e010b86455bfa505f13d33b1a8dd177078.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/b021bdb269bced5222710819c28f17e010b86455bfa505f13d33b1a8dd177078.jpg)

![b16aa2ed60f28447d709e573439cec529da01cfe920a72e2b1e57f330c749f78.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/b16aa2ed60f28447d709e573439cec529da01cfe920a72e2b1e57f330c749f78.jpg)

![bef018c8923b40827904664773fd1a42b82307611a1acae0c085bdaaf4ec83e8.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/bef018c8923b40827904664773fd1a42b82307611a1acae0c085bdaaf4ec83e8.jpg)

![cc87adbbbbbdcc88efb36d3f10f4d9e73549434b9f76e678c534ccc41e991c41.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/cc87adbbbbbdcc88efb36d3f10f4d9e73549434b9f76e678c534ccc41e991c41.jpg)

![e76180fd73e77a740a35fe0675de676b38482429c924fd0182312c820adff992.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/e76180fd73e77a740a35fe0675de676b38482429c924fd0182312c820adff992.jpg)

![f95807671169d46df7b25dd74577825f5504f87fafb07abff22abb1199ae7d84.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/images/f95807671169d46df7b25dd74577825f5504f87fafb07abff22abb1199ae7d84.jpg)

### Tables

![0960dbf4fb988f0628642b0f7887b185d8759e1a71f76934ced31c308c090a89.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/0960dbf4fb988f0628642b0f7887b185d8759e1a71f76934ced31c308c090a89.jpg)

![333434ae9bda296764a82083b780cf215a4f158a3fafabd4eb0b38466a9ba667.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/333434ae9bda296764a82083b780cf215a4f158a3fafabd4eb0b38466a9ba667.jpg)

![405a5a8943c8685fc81ae547c772dddac4ffa29e3ec606383bce6023cc21e86a.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/405a5a8943c8685fc81ae547c772dddac4ffa29e3ec606383bce6023cc21e86a.jpg)

![64c1e8f209ba1231255aa2b3e320f584476a1b09b116782175c3929e3287bbf5.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/64c1e8f209ba1231255aa2b3e320f584476a1b09b116782175c3929e3287bbf5.jpg)

![8bdac4036e182e6ba52c1814427ea7ce318393ff173e19690047f3f435dfed31.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/8bdac4036e182e6ba52c1814427ea7ce318393ff173e19690047f3f435dfed31.jpg)

![8e35f206d5d2fcaa627b71a226893acd67a3b4a9f478d84927f54cc00c3da84a.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/8e35f206d5d2fcaa627b71a226893acd67a3b4a9f478d84927f54cc00c3da84a.jpg)

![a24ec3bf85b75757e9a6ea25d416f541a6bf3f3d07c5c84712aa61032a029441.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/a24ec3bf85b75757e9a6ea25d416f541a6bf3f3d07c5c84712aa61032a029441.jpg)

![a2975b876fd845f665e15ba6a75e1082a711b7fa3070fe3280625cc9e42d9fd5.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/a2975b876fd845f665e15ba6a75e1082a711b7fa3070fe3280625cc9e42d9fd5.jpg)

![a723d884c9ffc64484708e5fad926108f3cdf54d5f3c7a9f1b14abd27b5f165e.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/a723d884c9ffc64484708e5fad926108f3cdf54d5f3c7a9f1b14abd27b5f165e.jpg)

![b754b4edb0cfd8aa77fd0b8f408ea18236e2843f3c7a6e9672755df21759aff8.jpg](../icml_results/1570_Learning%20Hamiltonian%20Dynamics%20at%20Scale_%20A%20Differential-Geometric%20Approach/tables/b754b4edb0cfd8aa77fd0b8f408ea18236e2843f3c7a6e9672755df21759aff8.jpg)

<a id="paper-1550"></a>

## Toward Scalable and Valid Conditional Independence Testing with Spectral Representations


### Images

![069e29d72b9ed40289a305683308d15a680229b7e01aa8e7bd287d304f4849ec.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/069e29d72b9ed40289a305683308d15a680229b7e01aa8e7bd287d304f4849ec.jpg)

![0e81d7a3cd2660e5a12e97e8e42bcbdc688a248dd917dab3b47c82452343b5ae.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/0e81d7a3cd2660e5a12e97e8e42bcbdc688a248dd917dab3b47c82452343b5ae.jpg)

![26e9ecfced0adfa955cb56edd3de675d7502bed146315e7668bd46afbb02a1db.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/26e9ecfced0adfa955cb56edd3de675d7502bed146315e7668bd46afbb02a1db.jpg)

![3dd1a9b78b3346508923041dd1375b7e832c54027c798b960f91b35df1e56d7d.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/3dd1a9b78b3346508923041dd1375b7e832c54027c798b960f91b35df1e56d7d.jpg)

![47502319447b109c317e226f9cecdc33b750b813b5f411ef6df911522cd964cf.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/47502319447b109c317e226f9cecdc33b750b813b5f411ef6df911522cd964cf.jpg)

![4ba77837583528145d44155c832581b04961592493ebd26ad765d0832381e190.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/4ba77837583528145d44155c832581b04961592493ebd26ad765d0832381e190.jpg)

![65efb938884869f8a7a3e91dd4fc9510be1f8f90c05a5eb25bac9ade0d32aa62.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/65efb938884869f8a7a3e91dd4fc9510be1f8f90c05a5eb25bac9ade0d32aa62.jpg)

![7369406af2d2253fde844c949e24d8e484ec2461ed17a40ed930f19405adf0c3.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/7369406af2d2253fde844c949e24d8e484ec2461ed17a40ed930f19405adf0c3.jpg)

![75b44c24317d9a01ad0725c53370b30596057e6247a37efb8eb1074cca0895ff.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/75b44c24317d9a01ad0725c53370b30596057e6247a37efb8eb1074cca0895ff.jpg)

![80c047a6d55e033a9009e7b04fc812625424ec726cc8336f4046b113091b22c1.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/80c047a6d55e033a9009e7b04fc812625424ec726cc8336f4046b113091b22c1.jpg)

![bc88f6c28542ff20790bf849b8549244d6c47bd9033fee69617a2fce2cb6617f.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/bc88f6c28542ff20790bf849b8549244d6c47bd9033fee69617a2fce2cb6617f.jpg)

![c2333e321ada68efbb975a8b4dee9cd05c94c65e16b2a5bbb158e3e3fcb5c4a5.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/c2333e321ada68efbb975a8b4dee9cd05c94c65e16b2a5bbb158e3e3fcb5c4a5.jpg)

![ea2ded5b15dcdc53ffb6699ea27520ca02e1c018718da27f3f556473a2355861.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/ea2ded5b15dcdc53ffb6699ea27520ca02e1c018718da27f3f556473a2355861.jpg)

![eca9aeeb4e655e8dcccdfc873617ba9785ab4122f9df7fdb399c7655e5699940.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/images/eca9aeeb4e655e8dcccdfc873617ba9785ab4122f9df7fdb399c7655e5699940.jpg)

### Tables

![6001f516982d009d14d02197404d57cdfa7286c870eac13fe890b6fd8472fb2b.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/6001f516982d009d14d02197404d57cdfa7286c870eac13fe890b6fd8472fb2b.jpg)

![6ce048d7936ed70edc5ce21f503031261c29574e6794cec54d1025c06d8bbf8d.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/6ce048d7936ed70edc5ce21f503031261c29574e6794cec54d1025c06d8bbf8d.jpg)

![793f203592f6e8d53753ede7980bd24fef4f1303bc24622335b312be9168d250.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/793f203592f6e8d53753ede7980bd24fef4f1303bc24622335b312be9168d250.jpg)

![a2a359fb354078fdc21b64cc6e4b1d4ed33fbd5c3c32d293e9688d9b33a6c09b.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/a2a359fb354078fdc21b64cc6e4b1d4ed33fbd5c3c32d293e9688d9b33a6c09b.jpg)

![ab836559566aa0b21bdabf5e768451fc8f017ce037260e7d730cf3e0655939b4.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/ab836559566aa0b21bdabf5e768451fc8f017ce037260e7d730cf3e0655939b4.jpg)

![b16a0a5f51212281e0cf2cb25fd0812760d7adeb34eddb5ec47d52038bb95b67.jpg](../icml_results/1571_Toward%20Scalable%20and%20Valid%20Conditional%20Independence%20Testing%20with%20Spectral%20Representations/tables/b16a0a5f51212281e0cf2cb25fd0812760d7adeb34eddb5ec47d52038bb95b67.jpg)

<a id="paper-1551"></a>

## Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling

### Images

![034259992ca81413d3e69cc86b9aad0ed17498e17f1aa6c78b8d41720ab0c2f8.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/034259992ca81413d3e69cc86b9aad0ed17498e17f1aa6c78b8d41720ab0c2f8.jpg)

![1d3c8e5dc855170500067c8befe8abe1511adbd2e98dc88105bfe3c4fc6e15df.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/1d3c8e5dc855170500067c8befe8abe1511adbd2e98dc88105bfe3c4fc6e15df.jpg)

![2745e24c9739cc8c2bc13cc6150b9af420d29c9d5b7030301850fc4c6ef48145.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/2745e24c9739cc8c2bc13cc6150b9af420d29c9d5b7030301850fc4c6ef48145.jpg)

![2a8b2fd9e571dd2e86f54d24e6f53ae8316a944e06b44fdf691fbb6bd771f7db.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/2a8b2fd9e571dd2e86f54d24e6f53ae8316a944e06b44fdf691fbb6bd771f7db.jpg)

![3a3be20e0132a55f6da72d7cb8592c4e8a899906d48caecb657bac12976ea0ef.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/3a3be20e0132a55f6da72d7cb8592c4e8a899906d48caecb657bac12976ea0ef.jpg)

![4406be4205d622f32c64532074984de11cdb42ce0e6faae3e5058c9a9451b90d.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/4406be4205d622f32c64532074984de11cdb42ce0e6faae3e5058c9a9451b90d.jpg)

![48452e5901f861de41f4a7a928bb7022f1d48fb9eb409a2e3c508f2be4344e94.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/48452e5901f861de41f4a7a928bb7022f1d48fb9eb409a2e3c508f2be4344e94.jpg)

![4a972a1f63b51dc72e17c8fb2b0fd32894dbf5f854a61f8e68240dfc8eb88464.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/4a972a1f63b51dc72e17c8fb2b0fd32894dbf5f854a61f8e68240dfc8eb88464.jpg)

![51440c6283a36c2200cff7928613f5244a78bc01d0bc35ca7067a50aa54c2cc0.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/51440c6283a36c2200cff7928613f5244a78bc01d0bc35ca7067a50aa54c2cc0.jpg)

![5a05bb75d431306c90389f3e05c658114a1d18978e6f961e59a5aa88b4f989ac.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/5a05bb75d431306c90389f3e05c658114a1d18978e6f961e59a5aa88b4f989ac.jpg)

![62eaf270a93c635e0ef388b8db6c6320ba3b7768ac8e677f69c52dcb57805b23.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/62eaf270a93c635e0ef388b8db6c6320ba3b7768ac8e677f69c52dcb57805b23.jpg)

![753b39ecc91a24430700902d85452f091358f59a48eae73c957ce8fb11ec5f44.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/753b39ecc91a24430700902d85452f091358f59a48eae73c957ce8fb11ec5f44.jpg)

![b4a7a679b406a386ce6efc2be321f81645dbf3b901a9346c00de56a8341a4d2e.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/b4a7a679b406a386ce6efc2be321f81645dbf3b901a9346c00de56a8341a4d2e.jpg)

![b579527db65a01098b511005351a912f75169e576dd36b444cde547e74a50afe.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/b579527db65a01098b511005351a912f75169e576dd36b444cde547e74a50afe.jpg)

![d0b5ab441fd7dd56657310ad138dca3c14a63250dc7da186ad16fb389ac4023a.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/d0b5ab441fd7dd56657310ad138dca3c14a63250dc7da186ad16fb389ac4023a.jpg)

![f54f5d13acfc2009c72537e1f3253c967a5abccd8200a220577e483dd131477e.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/images/f54f5d13acfc2009c72537e1f3253c967a5abccd8200a220577e483dd131477e.jpg)

### Tables

![062bb9b55b277f916bbca0b28f7e211597128535df8cec556e38a176a1b6828b.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/062bb9b55b277f916bbca0b28f7e211597128535df8cec556e38a176a1b6828b.jpg)

![0d53560136bb00348394aa859f4d4eba613f4e259f8aa426abf8289d69f37968.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/0d53560136bb00348394aa859f4d4eba613f4e259f8aa426abf8289d69f37968.jpg)

![1224b3d390fff1843ecac4d76dd5250a7ab848b6a305969116b071d24caa046c.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/1224b3d390fff1843ecac4d76dd5250a7ab848b6a305969116b071d24caa046c.jpg)

![1796f5040309f450590ff768f149bfc6fbfbe889d1f14190b785b4618e86c21c.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/1796f5040309f450590ff768f149bfc6fbfbe889d1f14190b785b4618e86c21c.jpg)

![1d49bfc3f304b180f2656bf5294789c43dc107b83ebf863e2b54fded1216ddab.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/1d49bfc3f304b180f2656bf5294789c43dc107b83ebf863e2b54fded1216ddab.jpg)

![36385b1a74c43df4451a4ce9235a3496408efa6bdae0ee1d1af494db665e808f.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/36385b1a74c43df4451a4ce9235a3496408efa6bdae0ee1d1af494db665e808f.jpg)

![4fd5f071bc41ad56f84e221bf15b148ff8e488c559076eefe0de3efbc2471be1.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/4fd5f071bc41ad56f84e221bf15b148ff8e488c559076eefe0de3efbc2471be1.jpg)

![5933a0adf6d922904172e9e921882a9a70c03ea2fc0d8c2bce23430dba0380b2.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/5933a0adf6d922904172e9e921882a9a70c03ea2fc0d8c2bce23430dba0380b2.jpg)

![5dd3f5ce4551a64b09e8f0d4f53d998603d2f08e46aaa8b36ce40b5d8a484781.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/5dd3f5ce4551a64b09e8f0d4f53d998603d2f08e46aaa8b36ce40b5d8a484781.jpg)

![8bec39b006c75039e7f50c44815a193328d03bf2ec9a93d66674ac70a9643c88.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/8bec39b006c75039e7f50c44815a193328d03bf2ec9a93d66674ac70a9643c88.jpg)

![a1d98efee414e49bbf0a481154a552e83af078291b44f8a4fb01a35a266dfb01.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/a1d98efee414e49bbf0a481154a552e83af078291b44f8a4fb01a35a266dfb01.jpg)

![a7a28063b6ed9f529567bb0c038d42dbc2d583060c717f60bcc5f5ca1f453599.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/a7a28063b6ed9f529567bb0c038d42dbc2d583060c717f60bcc5f5ca1f453599.jpg)

![b778d40aef43cf372679568f12f1df54e6af79e2549cd4ea73b86e24a2a6aa7c.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/b778d40aef43cf372679568f12f1df54e6af79e2549cd4ea73b86e24a2a6aa7c.jpg)

![bd8107899d96fabdfea43689b820fde95c63b81c11a518d9ef8c6f3d47476826.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/bd8107899d96fabdfea43689b820fde95c63b81c11a518d9ef8c6f3d47476826.jpg)

![e528b84b3b4c3847290ff18390b9a4fc2203c7b2f638a0a724b07a15aca125ed.jpg](../icml_results/1572_Q-Sched_%20Pushing%20the%20Boundaries%20of%20Few-Step%20Diffusion%20Models%20with%20Quantization-Aware%20Scheduling/tables/e528b84b3b4c3847290ff18390b9a4fc2203c7b2f638a0a724b07a15aca125ed.jpg)
