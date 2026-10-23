# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 163 of 197

## 目录 (Table of Contents)

1. [Quaternion Self-Attention with Shared Scores](#paper-5347)
2. [LLM-Guided Diagnostic Evidence Alignment for Medical Vision–Language Pretraining under Limited Pairing](#paper-5348)
3. [Taylor-Gaussians-Flow: Towards Non-uniform Motion for Novel View Synthesis from Monocular Video](#paper-5349)
4. [Proact-VL: A Proactive VideoLLM for Real-Time AI Companions](#paper-5350)
5. [Beyond Accuracy: Latent Perturbations for Cognitive-Aware Diagnosis](#paper-5351)
6. [InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](#paper-5352)
7. [Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs](#paper-5353)
8. [Rethink the Role of Neural Decoders in Quantum Error Correction](#paper-5354)
9. [Rethinking Generative Image Pretraining: How Far Are We From Scaling Up Next-Pixel Prediction?](#paper-5355)
10. [GLARE: Scalable Neuro-Symbolic Reward Shaping for LLM Agents via Group-Level Automata](#paper-5356)
11. [Invariant Representation Learning for Source-Free Time Series Forecasting with LLM-Centric Proxy Denoising](#paper-5357)
12. [Design Linear Constrained Neural Layers with Implicit Convex Optimization](#paper-5358)
13. [MemIncept: Steering LLM Agents via Cooperative Stealthy Memory Injections](#paper-5359)
14. [Arboreal Neural Network](#paper-5360)
15. [ORBIT: A Prognostic World Model for Ocular Reasoning Based on Imagined Trajectories](#paper-5361)
16. [PLoRA: Efficient Concurrent LoRA Training for Large Language Models](#paper-5362)
17. [Can VLMs Diagnose and Recover from VLA Manipulation Faults?](#paper-5363)
18. [From Knowledge to Inference: Formalizing Specialized Public Health Reasoning on GlobalHealthAtlas](#paper-5364)
19. [CircuitPrint: Mechanistic Circuit Fingerprints for Large Language Models](#paper-5365)
20. [SPUR: Scale-Partitioned Uncertainty Rectification for Robust UAV-on-UAV Interception](#paper-5366)
21. [Rethinking Serialization in Linear 3D Vision: Decoupling Anisotropic Geometry from Isotropic Semantics](#paper-5367)
22. [GRASP: Awakening Latent Spatial Reasoning in LVLMs via Training-free Geometric Rectification](#paper-5368)
23. [Robust Federated Learning Against Adaptive Compression](#paper-5369)
24. [CE$^4$L: Continual Ego, Exo, and Ego-Exo Learning](#paper-5370)
25. [A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data Analysis](#paper-5371)
26. [SpecExit: Accelerating Large Reasoning Model via Speculative Exit](#paper-5372)
27. [Preconditioning Neural Tangent Kernel for Adaptive Optimization](#paper-5373)
28. [Length Generalization Bounds for Transformers](#paper-5374)
29. [MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier](#paper-5375)
30. [Towards Understanding Generalization of Federated Adversarial Learning: Perspective of Algorithmic Stability](#paper-5376)
31. [PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents](#paper-5377)
32. [Concept Concentration for Faithful Representation Intervention](#paper-5378)
33. [PISA: Privacy-Preserving Split Adaptation with Model IP Protection](#paper-5379)

---


<a id="paper-5347"></a>

## Quaternion Self-Attention with Shared Scores

### Images

![057a590388a5d32f748fb6d263cc611b857a8d25e6fe48d14a471373dd21780c.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/057a590388a5d32f748fb6d263cc611b857a8d25e6fe48d14a471373dd21780c.jpg)

![0ef23c6b1b405be48e1bdcf7bb23e727f0a0f863f2c7091465d2b173883eb894.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/0ef23c6b1b405be48e1bdcf7bb23e727f0a0f863f2c7091465d2b173883eb894.jpg)

![14f9caa77fd720b5af865fcf66be383565a92112cee4bcdf2d5c8232b9acf885.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/14f9caa77fd720b5af865fcf66be383565a92112cee4bcdf2d5c8232b9acf885.jpg)

![17936c2237bab079d013248318bd8483977e032f84f7ceac5ca0bd5de6aaf4f1.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/17936c2237bab079d013248318bd8483977e032f84f7ceac5ca0bd5de6aaf4f1.jpg)

![18af74275f19bcee8e3e7c68eae684155458d3ea562e912ac104ede20a09f464.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/18af74275f19bcee8e3e7c68eae684155458d3ea562e912ac104ede20a09f464.jpg)

![1a2c3565added2b65d52c204bd561a2292f1a23835117da99eea5bd927322521.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/1a2c3565added2b65d52c204bd561a2292f1a23835117da99eea5bd927322521.jpg)

![35b61d31834319a8696ca5542c27157a3cad1401f4c2ae646ea81eb0d3a59875.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/35b61d31834319a8696ca5542c27157a3cad1401f4c2ae646ea81eb0d3a59875.jpg)

![4195175f0047d678bbf95b53e0b8d5d5b4ec2a2d42089d139920b446012b238c.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/4195175f0047d678bbf95b53e0b8d5d5b4ec2a2d42089d139920b446012b238c.jpg)

![429b7cf4caa37f3836885fcc7a48026323e7a96b785542002105e569ed7513f9.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/429b7cf4caa37f3836885fcc7a48026323e7a96b785542002105e569ed7513f9.jpg)

![58d662876630b7ab74395e7b7dcb64743dc0a5bd1182d962e12d98e0f5c88be2.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/58d662876630b7ab74395e7b7dcb64743dc0a5bd1182d962e12d98e0f5c88be2.jpg)

![5bc41334abecfb03e8997a826c635701959f00ae71d64523624415f444033465.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/5bc41334abecfb03e8997a826c635701959f00ae71d64523624415f444033465.jpg)

![646b2ae960a495732c763bb34dfa3f4b07eb0f4bd9bba64502fdc23c90495d99.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/646b2ae960a495732c763bb34dfa3f4b07eb0f4bd9bba64502fdc23c90495d99.jpg)

![67e053e2a2cc77dc20d19e862411b8e7764dc90a3ff1db9cee8476f5ddd4365a.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/67e053e2a2cc77dc20d19e862411b8e7764dc90a3ff1db9cee8476f5ddd4365a.jpg)

![6ccb875f5516a18b4475cf073f9e7cdbb4cd2415ecbe3dc4a02faafadd205a3c.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/6ccb875f5516a18b4475cf073f9e7cdbb4cd2415ecbe3dc4a02faafadd205a3c.jpg)

![72453c0c0718c801855d9a85d8b9ac7e673b70c18e0927025ad65de181737815.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/72453c0c0718c801855d9a85d8b9ac7e673b70c18e0927025ad65de181737815.jpg)

![94a9d1973d17aa342993933ebf0be86c1e8d9e72bab26f79dc1f3c4a068acf54.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/94a9d1973d17aa342993933ebf0be86c1e8d9e72bab26f79dc1f3c4a068acf54.jpg)

![95ee78116ba5ac883d17402e03c1c01630a6d2c8c4c382665372f6702466ca40.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/95ee78116ba5ac883d17402e03c1c01630a6d2c8c4c382665372f6702466ca40.jpg)

![a58456e70cf5c8a40d7fa19a7d076b4b2bce0325defc687f4beef1d12362c586.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/a58456e70cf5c8a40d7fa19a7d076b4b2bce0325defc687f4beef1d12362c586.jpg)

![b04d5d91f21e56776b9acba6a371059e8b46e242b26cab3920c55fed700ca518.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/b04d5d91f21e56776b9acba6a371059e8b46e242b26cab3920c55fed700ca518.jpg)

![b095451d343420d24e8b46d5130d4a12d589be6897e1df667bc4092aaa92659f.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/b095451d343420d24e8b46d5130d4a12d589be6897e1df667bc4092aaa92659f.jpg)

![ba0dbefca14be2be3352b0e8b5b4a20b897e07faea95764bdc432c21d3457d0e.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/ba0dbefca14be2be3352b0e8b5b4a20b897e07faea95764bdc432c21d3457d0e.jpg)

![c3a069c2ce58f5623ca854db540c1565e856859049d9a5389c12d29df6c00a93.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/c3a069c2ce58f5623ca854db540c1565e856859049d9a5389c12d29df6c00a93.jpg)

![c956ae09b5ee551af5a5bb0311e5a65112e49be7548180e563f66ad2f92a65b3.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/c956ae09b5ee551af5a5bb0311e5a65112e49be7548180e563f66ad2f92a65b3.jpg)

![cb4dbe613dc538e178f91feaaa631014d33d39721452734f665a8d3cf83b79bb.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/cb4dbe613dc538e178f91feaaa631014d33d39721452734f665a8d3cf83b79bb.jpg)

![cd5a506ee4eb402333f77f4453f355b823e40860716b85011d73cda292551cce.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/cd5a506ee4eb402333f77f4453f355b823e40860716b85011d73cda292551cce.jpg)

![dcac8be0b42dfd508cc8e51b071d362eab4e7ee943f43612e00c893318396d43.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/dcac8be0b42dfd508cc8e51b071d362eab4e7ee943f43612e00c893318396d43.jpg)

![f16c944e346eade8cb371db8c63778be84717b1b64370553eea76caf531c5218.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/images/f16c944e346eade8cb371db8c63778be84717b1b64370553eea76caf531c5218.jpg)

### Tables

![09119460d878728892822dcf6ac53363c97cde0ff3bf0b91855473618a2f0762.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/tables/09119460d878728892822dcf6ac53363c97cde0ff3bf0b91855473618a2f0762.jpg)

![0e594c1ec981e49ec3ae0393726143db465ec3ca9e447d1f69fe1d8afc3db2d1.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/tables/0e594c1ec981e49ec3ae0393726143db465ec3ca9e447d1f69fe1d8afc3db2d1.jpg)

![76b0089ae008fad67d3bccf110832234b2ac744dfeaa0f9aebe24ca00b0477ce.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/tables/76b0089ae008fad67d3bccf110832234b2ac744dfeaa0f9aebe24ca00b0477ce.jpg)

![c864b12ae3a1183296ef9c5b530c42e6c294e366a8d44666c4502f1a0f572d8e.jpg](../icml_results/5394_Left%E2%80%93Right%20Symmetry%20Breaking%20in%20CLIP-Style%20Vision-Language%20Models%20Trained%20on%20Synthetic%20Spatial-Relat/tables/c864b12ae3a1183296ef9c5b530c42e6c294e366a8d44666c4502f1a0f572d8e.jpg)

## Quaternion Self-Attention with Shared Scores


### Images

![454ae09887380a2916a43f1542a2c12624ce26b93f202977ca42ffc0c213604e.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/454ae09887380a2916a43f1542a2c12624ce26b93f202977ca42ffc0c213604e.jpg)

![62abc7cef36702584ed542b2d7557e25e7a8c8ab1dedb5c60861e5161c804810.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/62abc7cef36702584ed542b2d7557e25e7a8c8ab1dedb5c60861e5161c804810.jpg)

![72da977cb25fd6484e71a13a7f057c2ec0b8f24e892a093ca93a40c06f75d0d2.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/72da977cb25fd6484e71a13a7f057c2ec0b8f24e892a093ca93a40c06f75d0d2.jpg)

![97e7ab7bb9fa12a88133cd201d866b7c2e78f9d4db2642a14e084e2638fa2370.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/97e7ab7bb9fa12a88133cd201d866b7c2e78f9d4db2642a14e084e2638fa2370.jpg)

![9998587189c6f166b3dba91827da7ef7fb55f24f1cfd87447e037672dc6c1df9.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/9998587189c6f166b3dba91827da7ef7fb55f24f1cfd87447e037672dc6c1df9.jpg)

![af379a88cb24e250b383440ac829882a176095a2982854d1696d1da8ccec457e.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/images/af379a88cb24e250b383440ac829882a176095a2982854d1696d1da8ccec457e.jpg)

### Tables

![3107e6a02513bc83ba9b4443adece77360f363e79d4dc8d5cb12d5848181f363.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/3107e6a02513bc83ba9b4443adece77360f363e79d4dc8d5cb12d5848181f363.jpg)

![33a14f956e75a9ee0c3293dd60c5e5d4f30c7ecd766bbc8f1cd9ce0da6b4fa5c.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/33a14f956e75a9ee0c3293dd60c5e5d4f30c7ecd766bbc8f1cd9ce0da6b4fa5c.jpg)

![3405e7c93baec45813ffadcf8d52da88b618c2067b21a3dd4195c79df042fddb.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/3405e7c93baec45813ffadcf8d52da88b618c2067b21a3dd4195c79df042fddb.jpg)

![34abcae687e978226600806447b050dba70567945d43f303912eb7e6d1f84b74.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/34abcae687e978226600806447b050dba70567945d43f303912eb7e6d1f84b74.jpg)

![5145437128377a9ba4318955cc078f76038949e08fe8a5bcb1299763b6339abf.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/5145437128377a9ba4318955cc078f76038949e08fe8a5bcb1299763b6339abf.jpg)

![5b6d9e9a75a154bdefededd8ca27e8497c91c4b2f996109ca7e7f5edae0ace43.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/5b6d9e9a75a154bdefededd8ca27e8497c91c4b2f996109ca7e7f5edae0ace43.jpg)

![5cbacec3b281d4447a5cc96db8d57f563168c0a53d1ee956dbbfb1185b1292e2.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/5cbacec3b281d4447a5cc96db8d57f563168c0a53d1ee956dbbfb1185b1292e2.jpg)

![629eaba175b1352777723a0db6be2f6355162b4814323e931b51f04382ece380.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/629eaba175b1352777723a0db6be2f6355162b4814323e931b51f04382ece380.jpg)

![64dadc60a02e0d1e42476e3c64392b613dab81709ab61c8341c9b7b432945ac1.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/64dadc60a02e0d1e42476e3c64392b613dab81709ab61c8341c9b7b432945ac1.jpg)

![69ce736e12176c05d2b85adfd9837b0d3053bcfe5b18b1e541e340b65e395443.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/69ce736e12176c05d2b85adfd9837b0d3053bcfe5b18b1e541e340b65e395443.jpg)

![807dc46651de8817b6116bf18fc2ac3338fd557d90fc7669da2c7978876f04fa.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/807dc46651de8817b6116bf18fc2ac3338fd557d90fc7669da2c7978876f04fa.jpg)

![b439e3b10a98a16cb6c3e878c8be938e750be8af15a38fb666f615657c7637a6.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/b439e3b10a98a16cb6c3e878c8be938e750be8af15a38fb666f615657c7637a6.jpg)

![b7f44bf7d922d4fb9b00749aba5589e7a3d6af076f29a254a391af56af2ebf8b.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/b7f44bf7d922d4fb9b00749aba5589e7a3d6af076f29a254a391af56af2ebf8b.jpg)

![d21c7949874279b5d919dd4843bc7134f4c801f6d937715aa76134ca10c869c5.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/d21c7949874279b5d919dd4843bc7134f4c801f6d937715aa76134ca10c869c5.jpg)

![d2c91c61f87e2627bee526e99bb119b1d34b37c5369adf355db9c2ba9dcdf53c.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/d2c91c61f87e2627bee526e99bb119b1d34b37c5369adf355db9c2ba9dcdf53c.jpg)

![ee6d31f609f364fe5ad2f3c81eb94e1e7493787e91d6b589ed9b7a5bc361da8a.jpg](../icml_results/5395_Quaternion%20Self-Attention%20with%20Shared%20Scores/tables/ee6d31f609f364fe5ad2f3c81eb94e1e7493787e91d6b589ed9b7a5bc361da8a.jpg)

<a id="paper-5348"></a>

## LLM-Guided Diagnostic Evidence Alignment for Medical Vision–Language Pretraining under Limited Pairing


### Images

![0830706b5da0cb8478404d16d159952f253ad190e9a048ff214b69b2e594af1d.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/0830706b5da0cb8478404d16d159952f253ad190e9a048ff214b69b2e594af1d.jpg)

![2d1c42220d988039063a25c2f4eb65cc87dd0d352cfddf252ed392a2b48a936a.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/2d1c42220d988039063a25c2f4eb65cc87dd0d352cfddf252ed392a2b48a936a.jpg)

![6aed01001517755bb0bead0f1ffa3f67cb1eeda421ce7bbd529b6ba6091bac23.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/6aed01001517755bb0bead0f1ffa3f67cb1eeda421ce7bbd529b6ba6091bac23.jpg)

![7bb232fc7794ea6d01554f2b6f3cc8f53b7a2ea2213ce784630dd632e88a8c79.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/7bb232fc7794ea6d01554f2b6f3cc8f53b7a2ea2213ce784630dd632e88a8c79.jpg)

![85860270df3dd47429450bd5e9952c89f227478d48abf828ccb1bc97a9dc9122.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/85860270df3dd47429450bd5e9952c89f227478d48abf828ccb1bc97a9dc9122.jpg)

![dbd8b5f919447fad0595ca103ea8ddb368eba3be73bd2b93c7b34401fbc02b9d.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/dbd8b5f919447fad0595ca103ea8ddb368eba3be73bd2b93c7b34401fbc02b9d.jpg)

![fad75de80c53afa9dd4f2a8ec96fce64d071d85a8c3e901de58cb5c78ec100df.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/images/fad75de80c53afa9dd4f2a8ec96fce64d071d85a8c3e901de58cb5c78ec100df.jpg)

### Tables

![058e65f6e88f593ab0e2f7bde5e81ce6ccd6c5549b12a041478604188f50770b.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/058e65f6e88f593ab0e2f7bde5e81ce6ccd6c5549b12a041478604188f50770b.jpg)

![38f304b491e2bd8f816e81eb62d447a20831a8785042ff9cb5beda6b795c0da8.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/38f304b491e2bd8f816e81eb62d447a20831a8785042ff9cb5beda6b795c0da8.jpg)

![3ccffcd4637d6083f3fd6c8d65d7ae15d721b7a2faba79d7eb0298d227bc91ff.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/3ccffcd4637d6083f3fd6c8d65d7ae15d721b7a2faba79d7eb0298d227bc91ff.jpg)

![3f45dcb394b637cf1d8e98a65387d73d62b1cd04fb215a1e17627dbdf21aa22b.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/3f45dcb394b637cf1d8e98a65387d73d62b1cd04fb215a1e17627dbdf21aa22b.jpg)

![6028c1de2d5e1eee6c2b9c1e29eb830f139d3641166c4d0e4ec0e5089cde2448.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/6028c1de2d5e1eee6c2b9c1e29eb830f139d3641166c4d0e4ec0e5089cde2448.jpg)

![7b9adcd333141be00c2cc531dfc6c98f073ac3b902ea40ef0c6b61aab2202d8c.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/7b9adcd333141be00c2cc531dfc6c98f073ac3b902ea40ef0c6b61aab2202d8c.jpg)

![98dde744484ad3d111f49b4e24ef7e6ec516757b285bdcfba3c55f518624632a.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/98dde744484ad3d111f49b4e24ef7e6ec516757b285bdcfba3c55f518624632a.jpg)

![b33cd51fe68f0cec7a129cb92a850c990ccf0c3b3a6f4339b34469e46828799f.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/b33cd51fe68f0cec7a129cb92a850c990ccf0c3b3a6f4339b34469e46828799f.jpg)

![b9ee072ae3f76cfaeb8a794440dfc82154a5f4571110424911b36a3cdf8a2c4e.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/b9ee072ae3f76cfaeb8a794440dfc82154a5f4571110424911b36a3cdf8a2c4e.jpg)

![c03317564835d2cd50c0a9a4622122cfee41984226c558b0e363ada4d46bb859.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/c03317564835d2cd50c0a9a4622122cfee41984226c558b0e363ada4d46bb859.jpg)

![e1a4278706ec7f2ec32c7c09693f8f03e2c3dfc13174b9b263a63416e52a84ca.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/e1a4278706ec7f2ec32c7c09693f8f03e2c3dfc13174b9b263a63416e52a84ca.jpg)

![ef946890502361790d434a57260ba797accf91791906c24f591a0941c26708da.jpg](../icml_results/5396_LLM-Guided%20Diagnostic%20Evidence%20Alignment%20for%20Medical%20Vision%E2%80%93Language%20Pretraining%20under%20Limited%20Pairi/tables/ef946890502361790d434a57260ba797accf91791906c24f591a0941c26708da.jpg)

<a id="paper-5349"></a>

## Taylor-Gaussians-Flow: Towards Non-uniform Motion for Novel View Synthesis from Monocular Video


### Images

![0fab21fd72a18474d6377e003179aad58ffa80ff597f1ca1c440a46119fd2976.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/0fab21fd72a18474d6377e003179aad58ffa80ff597f1ca1c440a46119fd2976.jpg)

![57db0cf84d97a774338633706781345fb9f46dbd8fd45d24a9915bd4c9756f72.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/57db0cf84d97a774338633706781345fb9f46dbd8fd45d24a9915bd4c9756f72.jpg)

![64ef3250352c623581850d30d6944ef5cab7d8623e02b89c922deb3dadcb7660.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/64ef3250352c623581850d30d6944ef5cab7d8623e02b89c922deb3dadcb7660.jpg)

![67808282b50fec065573126f9dc75db3496cfc355fe34eccc230255afc2be78b.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/67808282b50fec065573126f9dc75db3496cfc355fe34eccc230255afc2be78b.jpg)

![6f3aba089ce4230bd347119c9386af83435954dcc5c5cba1c636de13e9308787.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/6f3aba089ce4230bd347119c9386af83435954dcc5c5cba1c636de13e9308787.jpg)

![727042da5eed329b4157a9c6630a3369c36077c886bd002cf9f466c5c8935b75.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/727042da5eed329b4157a9c6630a3369c36077c886bd002cf9f466c5c8935b75.jpg)

![ba74084e6e24acec39483125bf2d0b3006975528e7e446f357ad7ad0a743ca8c.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/ba74084e6e24acec39483125bf2d0b3006975528e7e446f357ad7ad0a743ca8c.jpg)

![f02a143d1c2bb096fb2088e99e24d954b1b55b8e98799b9368d830903cb47291.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/f02a143d1c2bb096fb2088e99e24d954b1b55b8e98799b9368d830903cb47291.jpg)

![f2abb49c5a5770f77ffd9c9d216ec0caf090344e78a47c25b44348459c4e1d3e.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/images/f2abb49c5a5770f77ffd9c9d216ec0caf090344e78a47c25b44348459c4e1d3e.jpg)

### Tables

![0b2595eadeb4db21a065a8159fbec6df47ca96f5a65e7ffe31c7c3ac379078df.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/0b2595eadeb4db21a065a8159fbec6df47ca96f5a65e7ffe31c7c3ac379078df.jpg)

![2bb9e34a06d281f181c7d17b216ad250e7237aed4c5f29737468735ad414982a.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/2bb9e34a06d281f181c7d17b216ad250e7237aed4c5f29737468735ad414982a.jpg)

![60787b29aa432b10e792c7f846f45976cd4ab42c642928f987b8e35c57aeb968.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/60787b29aa432b10e792c7f846f45976cd4ab42c642928f987b8e35c57aeb968.jpg)

![69cc49d3a459aa81cfeed90ee7a1c5ccebe64607a08b7fd528148c722d509cd1.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/69cc49d3a459aa81cfeed90ee7a1c5ccebe64607a08b7fd528148c722d509cd1.jpg)

![6fccb96d9d9c1333f3138e14bba3e604003e2a63dd0750fde893f11d1d92791a.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/6fccb96d9d9c1333f3138e14bba3e604003e2a63dd0750fde893f11d1d92791a.jpg)

![7a826ce222063265cc1035c39eae4275a36c147a16e66ff02d5bd6f63861e674.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/7a826ce222063265cc1035c39eae4275a36c147a16e66ff02d5bd6f63861e674.jpg)

![82721b274bdeea66a65a6c606c0042faf552b4902e005203c34dc1011d8c4a19.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/82721b274bdeea66a65a6c606c0042faf552b4902e005203c34dc1011d8c4a19.jpg)

![9d76c9ed67c123137a11aa69618ce0702e4de7d289e4cd98ee6dbdcb86c45462.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/9d76c9ed67c123137a11aa69618ce0702e4de7d289e4cd98ee6dbdcb86c45462.jpg)

![caac96e9f26f036d324e8bda5a2fbcc898accc82a6d12c0f1480db27fa87b838.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/caac96e9f26f036d324e8bda5a2fbcc898accc82a6d12c0f1480db27fa87b838.jpg)

![fd6134b798ffa26daab9c8a035ea76720694e8be8b951a515b785dae682e41ee.jpg](../icml_results/5397_Taylor-Gaussians-Flow_%20Towards%20Non-uniform%20Motion%20for%20Novel%20View%20Synthesis%20from%20Monocular%20Video/tables/fd6134b798ffa26daab9c8a035ea76720694e8be8b951a515b785dae682e41ee.jpg)

<a id="paper-5350"></a>

## Proact-VL: A Proactive VideoLLM for Real-Time AI Companions


### Images

![2256c1a08422665b02269273df757a4f4bf052f5479b04396d674f2d0696b111.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/2256c1a08422665b02269273df757a4f4bf052f5479b04396d674f2d0696b111.jpg)

![297e53a31ddbac256ddbda4ad040b22505768bd9749591fca43d0b5dbcc035e9.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/297e53a31ddbac256ddbda4ad040b22505768bd9749591fca43d0b5dbcc035e9.jpg)

![30113475c8d9906a33cd99db3e5cd2f752c0e96e328c6f19ae266e3b67f35a6b.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/30113475c8d9906a33cd99db3e5cd2f752c0e96e328c6f19ae266e3b67f35a6b.jpg)

![3d17edad3d9983a9f05eeeded6c1879e8ae8a31f17800acc3c1de4af8b041780.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/3d17edad3d9983a9f05eeeded6c1879e8ae8a31f17800acc3c1de4af8b041780.jpg)

![4597744d3f19ae3ca1dd2239fe7cd51d3d1b9fccdb90e0c8ed0fcfcc60bb85b1.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/4597744d3f19ae3ca1dd2239fe7cd51d3d1b9fccdb90e0c8ed0fcfcc60bb85b1.jpg)

![62733809bcb0f0124fb3170907054021bc8a3ee62391ccfe07029e0cdfa34649.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/62733809bcb0f0124fb3170907054021bc8a3ee62391ccfe07029e0cdfa34649.jpg)

![69a9e4006f53ee31335128c82286b9e6386a97505d40afebb52e09c4fb2e64ee.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/69a9e4006f53ee31335128c82286b9e6386a97505d40afebb52e09c4fb2e64ee.jpg)

![73e53760f7fe673044fab19b294d4d1f4873c9b77a0b82c81b863e6692b069b8.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/73e53760f7fe673044fab19b294d4d1f4873c9b77a0b82c81b863e6692b069b8.jpg)

![75e51bd253ae793ad19bb5cb370c4f37879c4a1aa63b99445cd25d3799942447.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/75e51bd253ae793ad19bb5cb370c4f37879c4a1aa63b99445cd25d3799942447.jpg)

![76ac5b6f0b0006ec1f142fd55e91ce7bb02f8fb8b6b8299b6558636b9bdcec61.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/76ac5b6f0b0006ec1f142fd55e91ce7bb02f8fb8b6b8299b6558636b9bdcec61.jpg)

![7b24f4a0553a6e71c20d9632ff5fc131983c10aa3ee10779320d6932e453ab71.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/7b24f4a0553a6e71c20d9632ff5fc131983c10aa3ee10779320d6932e453ab71.jpg)

![88708c9efdd155abc60f695751459e9633da619b9dd23c45b2a3f6a7156677f8.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/88708c9efdd155abc60f695751459e9633da619b9dd23c45b2a3f6a7156677f8.jpg)

![8b10f5773fa2e8810055cb30ce9d23190fd081c31c608c39da266620425b5962.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/8b10f5773fa2e8810055cb30ce9d23190fd081c31c608c39da266620425b5962.jpg)

![b299ea28c00e0df925baabdae1019b437af2572e1fdfbac058651d1cc748726a.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/b299ea28c00e0df925baabdae1019b437af2572e1fdfbac058651d1cc748726a.jpg)

![b9034475ac4a904d48d0c091b2c0f1ab9f2f7d015a57f25627372d5857b8ef38.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/b9034475ac4a904d48d0c091b2c0f1ab9f2f7d015a57f25627372d5857b8ef38.jpg)

![c613eafde18be3db34b0e01e88ea70efe70a9ea20fddbb2dfe0a5035445f3d7b.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/c613eafde18be3db34b0e01e88ea70efe70a9ea20fddbb2dfe0a5035445f3d7b.jpg)

![c7b31446931e4e42c112866bae61c2ef7f0a7174e5422d9d2e414cec7b2b297e.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/images/c7b31446931e4e42c112866bae61c2ef7f0a7174e5422d9d2e414cec7b2b297e.jpg)

### Tables

![0bd1be279dc883a2acc3e3542de3b5a3c6104b8dbdd162ef622f1b0d8f88f14f.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/0bd1be279dc883a2acc3e3542de3b5a3c6104b8dbdd162ef622f1b0d8f88f14f.jpg)

![104f4a186c2a017685244fb909e056cc2dec52684726f016f4362c8334d195db.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/104f4a186c2a017685244fb909e056cc2dec52684726f016f4362c8334d195db.jpg)

![188bebbb9ddb10a71f98aeb15955b2618f9273209d5d38ff83cca3222750ef43.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/188bebbb9ddb10a71f98aeb15955b2618f9273209d5d38ff83cca3222750ef43.jpg)

![30e26959d43179c93a1327a1a546958ba1bc9b6e0887e2c05e75493dfe1a1cff.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/30e26959d43179c93a1327a1a546958ba1bc9b6e0887e2c05e75493dfe1a1cff.jpg)

![4b3bcf17838c31c8b82c23afa7fdf4ae7a8a58feddcf239b0b95c420b0d825bd.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/4b3bcf17838c31c8b82c23afa7fdf4ae7a8a58feddcf239b0b95c420b0d825bd.jpg)

![6cc33fd12739e242c466c85f4b531d6cc8864b4a0e0f4b06439248583b82ff86.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/6cc33fd12739e242c466c85f4b531d6cc8864b4a0e0f4b06439248583b82ff86.jpg)

![71df834dfdd69b2e9adb0537b6e0b9addcd82d43ffe57b354ed2c441e84fb71d.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/71df834dfdd69b2e9adb0537b6e0b9addcd82d43ffe57b354ed2c441e84fb71d.jpg)

![76794061c072184f647a7f2606e457087fc44dd0b6d9baae8cb07fedf9054a1d.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/76794061c072184f647a7f2606e457087fc44dd0b6d9baae8cb07fedf9054a1d.jpg)

![7f5dcc1cc755f0d840cb952ff11c1c356fe4a871a7b989e34612c119e0ac5bf0.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/7f5dcc1cc755f0d840cb952ff11c1c356fe4a871a7b989e34612c119e0ac5bf0.jpg)

![853c264d906b0aa536ee2c0d384b618a02b750d98a3f4223535c36cc03e2c8fe.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/853c264d906b0aa536ee2c0d384b618a02b750d98a3f4223535c36cc03e2c8fe.jpg)

![86a30f5e47a24cd620e80d33b044f26212c2b0364912391bcd7d384f14586780.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/86a30f5e47a24cd620e80d33b044f26212c2b0364912391bcd7d384f14586780.jpg)

![8cb004380598133583cdcf8d5a649a3c8a4efcd661fe3259f1179879168c9f6c.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/8cb004380598133583cdcf8d5a649a3c8a4efcd661fe3259f1179879168c9f6c.jpg)

![90a8b09fe3afe9d69b8f87ddcbed1149599c2f0c016a772bc644e770ed5f25ea.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/90a8b09fe3afe9d69b8f87ddcbed1149599c2f0c016a772bc644e770ed5f25ea.jpg)

![ab5c5f38b89ddbeed21bee8ec462bd2ac9ccf9f49ec604add57f9a805f56e3d9.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/ab5c5f38b89ddbeed21bee8ec462bd2ac9ccf9f49ec604add57f9a805f56e3d9.jpg)

![aed8c1f3043c80b746a4ab4c2fc2bf3ac8548b4148dc5b285a94c1f41af87597.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/aed8c1f3043c80b746a4ab4c2fc2bf3ac8548b4148dc5b285a94c1f41af87597.jpg)

![b5470b697be06760c283ebd59dba5c3eb061067946fc5b679cec6cdc9147ef26.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/b5470b697be06760c283ebd59dba5c3eb061067946fc5b679cec6cdc9147ef26.jpg)

![ba82529cba7ee947be252b88f50bac02272f1c89c123a84d43a6ab083c83f7d9.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/ba82529cba7ee947be252b88f50bac02272f1c89c123a84d43a6ab083c83f7d9.jpg)

![d3da37ed72614763ec61a6e5b658a4dff66fb3d75018ef121700e617346849e8.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/d3da37ed72614763ec61a6e5b658a4dff66fb3d75018ef121700e617346849e8.jpg)

![d61544e50e6dfe9d272273eb7f037d3daf3db5902746c2942eb7e9fc42817402.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/d61544e50e6dfe9d272273eb7f037d3daf3db5902746c2942eb7e9fc42817402.jpg)

![f5ce7b00d44b870aeff253812fe5e8af00214ba47168253171a402555380ad96.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/f5ce7b00d44b870aeff253812fe5e8af00214ba47168253171a402555380ad96.jpg)

![fa2aa2d1d91f0c8ca09ef5cf44cf4ca42d7e5f624c4646e44f90ef7097e4c6e7.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/fa2aa2d1d91f0c8ca09ef5cf44cf4ca42d7e5f624c4646e44f90ef7097e4c6e7.jpg)

![fca6e2aaf334b8d87de5da4f8753e04a4a71e61d6c88856ecd7425e6e912289a.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/fca6e2aaf334b8d87de5da4f8753e04a4a71e61d6c88856ecd7425e6e912289a.jpg)

![fffed8ea2345e7520dcbd9a96fda81eedd68f249389f65fe6636f356324a5f0d.jpg](../icml_results/5398_Proact-VL_%20A%20Proactive%20VideoLLM%20for%20Real-Time%20AI%20Companions/tables/fffed8ea2345e7520dcbd9a96fda81eedd68f249389f65fe6636f356324a5f0d.jpg)

<a id="paper-5351"></a>

## Beyond Accuracy: Latent Perturbations for Cognitive-Aware Diagnosis


### Images

![26a3a85e30f84c9a39f617818b8c5f7a62ff15b46fd9c3d9598bae82782cea9b.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/26a3a85e30f84c9a39f617818b8c5f7a62ff15b46fd9c3d9598bae82782cea9b.jpg)

![77f4e4886f840870050fcaf851b8820833c9cce5a2ff64af8b100f2ae3b4cdc2.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/77f4e4886f840870050fcaf851b8820833c9cce5a2ff64af8b100f2ae3b4cdc2.jpg)

![8486cebc9b1323a0ce8753644f5f067e2f3cabba559fd63a85202e91fe562fd4.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/8486cebc9b1323a0ce8753644f5f067e2f3cabba559fd63a85202e91fe562fd4.jpg)

![8624618b1d9430f6ff5ed84c476ce1e0e957b7e05a5c259c05893548122a39a3.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/8624618b1d9430f6ff5ed84c476ce1e0e957b7e05a5c259c05893548122a39a3.jpg)

![b4d2e40dbe83e1db2272d44685698e4a6a3268377d9f2260a27910d92ba27d2c.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/b4d2e40dbe83e1db2272d44685698e4a6a3268377d9f2260a27910d92ba27d2c.jpg)

![d2176c6fa70680df83c2833a2c17bcede035d7a2f35289cf97b93c9f65d37713.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/d2176c6fa70680df83c2833a2c17bcede035d7a2f35289cf97b93c9f65d37713.jpg)

![feae4a8e4c4d6dc40f53e824595c56d7fd34848a4f4c7e321ac712376c5b6f28.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/images/feae4a8e4c4d6dc40f53e824595c56d7fd34848a4f4c7e321ac712376c5b6f28.jpg)

### Tables

![03406cffa01fc5b154f8bd980721015a873e41b85e557f13a8f376e9caf808a4.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/03406cffa01fc5b154f8bd980721015a873e41b85e557f13a8f376e9caf808a4.jpg)

![3074348b2fdf80791cf50fd7dee781c56c32b18d89474596282fba28c0761feb.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/3074348b2fdf80791cf50fd7dee781c56c32b18d89474596282fba28c0761feb.jpg)

![4ef67645752f5b0e6b36c51c4293fe5d03d018101276a2a2f53bb53fe5aa9b6c.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/4ef67645752f5b0e6b36c51c4293fe5d03d018101276a2a2f53bb53fe5aa9b6c.jpg)

![5c120d549ab9db3cda74cafcf5a90835ff79496bafb8ef8f4e320e0cfded1769.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/5c120d549ab9db3cda74cafcf5a90835ff79496bafb8ef8f4e320e0cfded1769.jpg)

![5d42df38561308c5de66bba9ba1f0c9099d4002c9f880a577453873eae45f6a0.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/5d42df38561308c5de66bba9ba1f0c9099d4002c9f880a577453873eae45f6a0.jpg)

![65e4b45e0c8e2ba9bb8ecfc833b9b99a6c53dd9d43de348a47afd4e2d9c8af47.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/65e4b45e0c8e2ba9bb8ecfc833b9b99a6c53dd9d43de348a47afd4e2d9c8af47.jpg)

![65fae462c9146a53094bbd3bbe7eb34e42c1a1cfb2c1ff5b6f9a1a993196e2fd.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/65fae462c9146a53094bbd3bbe7eb34e42c1a1cfb2c1ff5b6f9a1a993196e2fd.jpg)

![8fe25ae27397a827772ea09715ca7a2fff26b147537c56cc83c35d096bcf808c.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/8fe25ae27397a827772ea09715ca7a2fff26b147537c56cc83c35d096bcf808c.jpg)

![f92a735ff2ecbd43efd151790a4823fe3ab06934d87f22a9eebe0b793c616860.jpg](../icml_results/5399_Beyond%20Accuracy_%20Latent%20Perturbations%20for%20Cognitive-Aware%20Diagnosis/tables/f92a735ff2ecbd43efd151790a4823fe3ab06934d87f22a9eebe0b793c616860.jpg)

<a id="paper-5352"></a>

## InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning


### Images

![00a39946225ba73ad8681f2357b72ba5e60b2e9f0b90716991b307eff5e2e109.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/00a39946225ba73ad8681f2357b72ba5e60b2e9f0b90716991b307eff5e2e109.jpg)

![03f18ca9cd2293fda1ab90098a9fd9009bf8c20f130a477278892b49bf23ebd6.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/03f18ca9cd2293fda1ab90098a9fd9009bf8c20f130a477278892b49bf23ebd6.jpg)

![1287dd836838182f1b0a04fcf64916931d7ba84f0ef2203cf45abf7fea543e60.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/1287dd836838182f1b0a04fcf64916931d7ba84f0ef2203cf45abf7fea543e60.jpg)

![132456c4bd1d006d45ee8b8fb5697d0c001aa949e5727cc192141c7de49fb72a.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/132456c4bd1d006d45ee8b8fb5697d0c001aa949e5727cc192141c7de49fb72a.jpg)

![1ae21b66f2a19d5ae5f40f4f53829b05255e1b437e68d68a37c8937a04d3745d.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/1ae21b66f2a19d5ae5f40f4f53829b05255e1b437e68d68a37c8937a04d3745d.jpg)

![3839d7c86601f86a33be1b369202021a8d692801dadca1b763de1fb12a1a3101.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/3839d7c86601f86a33be1b369202021a8d692801dadca1b763de1fb12a1a3101.jpg)

![39939a12b6ee3c4c30ec2e3dec638c31241feecc9bfd33ee247fc6576cf5ced7.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/39939a12b6ee3c4c30ec2e3dec638c31241feecc9bfd33ee247fc6576cf5ced7.jpg)

![46e09544319c1d1b2af0218980ea59ff05d13e6c106f7808d54dd98e93741a0d.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/46e09544319c1d1b2af0218980ea59ff05d13e6c106f7808d54dd98e93741a0d.jpg)

![5025de115e1143a7661f6e86338440e2fb63406daa0c17bdfb8a67a9016c8dae.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/5025de115e1143a7661f6e86338440e2fb63406daa0c17bdfb8a67a9016c8dae.jpg)

![52cd5c2a5da58d1e49e8ef6677f88f9d042c815b232e3c502fd6b09d33a145cf.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/52cd5c2a5da58d1e49e8ef6677f88f9d042c815b232e3c502fd6b09d33a145cf.jpg)

![53ebd1c268bfe8865467e99d95d495f2307634c7e180d3b2324edaa2f61c6432.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/53ebd1c268bfe8865467e99d95d495f2307634c7e180d3b2324edaa2f61c6432.jpg)

![58475024a6836775ab17fa035802503d1de386cd665043468065d300822fa7d0.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/58475024a6836775ab17fa035802503d1de386cd665043468065d300822fa7d0.jpg)

![62b5b2c42a9e89cf05160793b63fc7686e4b492c1d160ad25d7f4c3e0f60c106.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/62b5b2c42a9e89cf05160793b63fc7686e4b492c1d160ad25d7f4c3e0f60c106.jpg)

![62ba3557267819d07770b84e8224ceed16a3c155af573c16251f9696b4840232.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/62ba3557267819d07770b84e8224ceed16a3c155af573c16251f9696b4840232.jpg)

![6a1579769794fc74499a4a03801ccecc57c09092d7a8d9ce73f750539bf95f45.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/6a1579769794fc74499a4a03801ccecc57c09092d7a8d9ce73f750539bf95f45.jpg)

![7ca5caa5feab398eec02a4016a6081e38becc3961a146b9121252b453a039860.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/7ca5caa5feab398eec02a4016a6081e38becc3961a146b9121252b453a039860.jpg)

![8005851a6c917a167747371842670426030de0168760de530651417659ee63dc.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/8005851a6c917a167747371842670426030de0168760de530651417659ee63dc.jpg)

![84dc26f8f4119985e1ee4bd525dc959e8d8dda2b84aea385b1d0fcc88e2453a5.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/84dc26f8f4119985e1ee4bd525dc959e8d8dda2b84aea385b1d0fcc88e2453a5.jpg)

![87db46508e3e034c24726a8f5471795377f11c60fba454552238b1cf345b6834.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/87db46508e3e034c24726a8f5471795377f11c60fba454552238b1cf345b6834.jpg)

![8dc37431b0844dc8178bd60b86d9cf1be6b4ec0d43e8faa816c620d04d19d593.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/8dc37431b0844dc8178bd60b86d9cf1be6b4ec0d43e8faa816c620d04d19d593.jpg)

![a16d6f958569c7f0b1bcfaba9c6ba1f8e5ca7f4f7f906f54d605cc5d0035e694.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/a16d6f958569c7f0b1bcfaba9c6ba1f8e5ca7f4f7f906f54d605cc5d0035e694.jpg)

![a1880d22e6202025055814cc45ff8421bf9e24b39c7bad8b5fd845a839a7d438.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/a1880d22e6202025055814cc45ff8421bf9e24b39c7bad8b5fd845a839a7d438.jpg)

![a779cf9aa4091002b820327d3464ff5fd307ea0393a4f1cc118ed75705ff2cfc.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/a779cf9aa4091002b820327d3464ff5fd307ea0393a4f1cc118ed75705ff2cfc.jpg)

![ac926a9d8644ddaead9705f0d02e9729659a46f357e32da064dd659a5cdc7579.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/ac926a9d8644ddaead9705f0d02e9729659a46f357e32da064dd659a5cdc7579.jpg)

![b286a2e45031b6fc56ae038b2eb12b8879444bc9eda7602d7159c2fd07ec6a91.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/b286a2e45031b6fc56ae038b2eb12b8879444bc9eda7602d7159c2fd07ec6a91.jpg)

![b6fb8f817a1ffc16e8da796fc0c005a4fcfd82423f442bca6491edd15bed2ed7.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/b6fb8f817a1ffc16e8da796fc0c005a4fcfd82423f442bca6491edd15bed2ed7.jpg)

![d837484f770dffdbcdab0839cb2e8e48942f8db6147af592ef626e210e51ce80.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/d837484f770dffdbcdab0839cb2e8e48942f8db6147af592ef626e210e51ce80.jpg)

![d8d0ed5e730326b67dad2ee444a0b99921e072d6628a5621409ca8ed088feed2.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/d8d0ed5e730326b67dad2ee444a0b99921e072d6628a5621409ca8ed088feed2.jpg)

![ddeb544f094eb465d256da7685e5524b7d1104018583373cc8a1332604d66d7e.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/ddeb544f094eb465d256da7685e5524b7d1104018583373cc8a1332604d66d7e.jpg)

![df95a7aed0b8e808e1b63159d2417c7e3f6fe684098f4f663936bd2d2131c9e2.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/df95a7aed0b8e808e1b63159d2417c7e3f6fe684098f4f663936bd2d2131c9e2.jpg)

![e2487939c32b1e3005a545ae8111922c8bc2bf770b99fb11eaff8af37b44ed6c.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/e2487939c32b1e3005a545ae8111922c8bc2bf770b99fb11eaff8af37b44ed6c.jpg)

![ed641abca6009413022802472ee9cf0a40515c5218a53e803f9c3f943596c8db.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/ed641abca6009413022802472ee9cf0a40515c5218a53e803f9c3f943596c8db.jpg)

![ee70d8ce9d450b945c7fc6d2f74ea083e2673acef8507a134e2c527fe7e647e7.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/ee70d8ce9d450b945c7fc6d2f74ea083e2673acef8507a134e2c527fe7e647e7.jpg)

![f54d433b93757830a5d32fc51f1ad71201c5079ae504fbb2482bae5a0b9ef4a7.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/images/f54d433b93757830a5d32fc51f1ad71201c5079ae504fbb2482bae5a0b9ef4a7.jpg)

### Tables

![02139d7f7b2f1dec9f982eaffc09064bf043e8f83aec51e8ec8e3176d087bcd5.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/02139d7f7b2f1dec9f982eaffc09064bf043e8f83aec51e8ec8e3176d087bcd5.jpg)

![09c5cafdf7d43e8bd5f3c4c9a520b88986136182dceb8dcc0bd301caae4cfa37.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/09c5cafdf7d43e8bd5f3c4c9a520b88986136182dceb8dcc0bd301caae4cfa37.jpg)

![1174234ac499362151c51a36f07a987b65c078ba189aa60b0fea9cfb9bb6f12a.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/1174234ac499362151c51a36f07a987b65c078ba189aa60b0fea9cfb9bb6f12a.jpg)

![182bd7d50c43942577787f253eb99d71abc53be0a0772290fe0698904278f34b.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/182bd7d50c43942577787f253eb99d71abc53be0a0772290fe0698904278f34b.jpg)

![1d260eee90e13c71ba3514f79cf0ae3ee757417e744b28deb0e95cdf4128d0dd.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/1d260eee90e13c71ba3514f79cf0ae3ee757417e744b28deb0e95cdf4128d0dd.jpg)

![5d1688609376dacfcf109a1fe44cef586da28e163f864b5b57f10740842d6130.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/5d1688609376dacfcf109a1fe44cef586da28e163f864b5b57f10740842d6130.jpg)

![6d093aa5db13e1135714ac8e73fc97379b13d2d5bbe0506b7b0c76cbd99642f9.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/6d093aa5db13e1135714ac8e73fc97379b13d2d5bbe0506b7b0c76cbd99642f9.jpg)

![7b09e672646ee29dfb575fa67eaf48c01c94788a9ccea89e33d981d8c1992eda.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/7b09e672646ee29dfb575fa67eaf48c01c94788a9ccea89e33d981d8c1992eda.jpg)

![84c102e5daa677568de78c33bc22f1847fcc287248b119cb673c456bc37185ff.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/84c102e5daa677568de78c33bc22f1847fcc287248b119cb673c456bc37185ff.jpg)

![b4f0f3f6dddd30b5a4d5caf8464ad506262382caaea7aedcdab518eb90fe4306.jpg](../icml_results/5400_InftyThink%2B_%20Effective%20and%20Efficient%20Infinite-Horizon%20Reasoning%20via%20Reinforcement%20Learning/tables/b4f0f3f6dddd30b5a4d5caf8464ad506262382caaea7aedcdab518eb90fe4306.jpg)

<a id="paper-5353"></a>

## Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs


### Images

![06bec0883fed60cec59f1b0bcc2236f324ebf7462a1af1bcc2a4b629bf8de69e.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/06bec0883fed60cec59f1b0bcc2236f324ebf7462a1af1bcc2a4b629bf8de69e.jpg)

![0b94e4f5b1de8d772e125f080a462777c92c3a7bb56cfbe679f3dcd7349d8590.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/0b94e4f5b1de8d772e125f080a462777c92c3a7bb56cfbe679f3dcd7349d8590.jpg)

![0f4d3140c039ff9c7087429524f1ed9e28c752d2b76cc99de4762e8c5cd3406c.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/0f4d3140c039ff9c7087429524f1ed9e28c752d2b76cc99de4762e8c5cd3406c.jpg)

![1270ad4ef392a32b06afe949cf46dd0bef9a43dc3034a3bb12857f594bf0f040.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/1270ad4ef392a32b06afe949cf46dd0bef9a43dc3034a3bb12857f594bf0f040.jpg)

![1a32849291e00efae6bc4660747c4b2687839a14d684876bde0a1490e9bfa4e1.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/1a32849291e00efae6bc4660747c4b2687839a14d684876bde0a1490e9bfa4e1.jpg)

![1e5c21d176e081e5e50aba762541c77eda8f81959565d59eda20910d268edfcb.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/1e5c21d176e081e5e50aba762541c77eda8f81959565d59eda20910d268edfcb.jpg)

![330750e7df77a82586a3a5f7ba0d39c277d60901e30932f231428a60b808c83c.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/330750e7df77a82586a3a5f7ba0d39c277d60901e30932f231428a60b808c83c.jpg)

![48efc231de219bf832fe3614d4e202becc5c6d1aaf66490ac31a9d9c14ad4f39.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/48efc231de219bf832fe3614d4e202becc5c6d1aaf66490ac31a9d9c14ad4f39.jpg)

![4c2c6a8ed3b3c3b8ba85f4e391ac4081ab724739817ff695602618911205496a.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/4c2c6a8ed3b3c3b8ba85f4e391ac4081ab724739817ff695602618911205496a.jpg)

![5b0d7b724f0cf8041e9ee5f0139c59910e7c2b5ade3ff2aedd44efad759bfbdc.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/5b0d7b724f0cf8041e9ee5f0139c59910e7c2b5ade3ff2aedd44efad759bfbdc.jpg)

![5c757f3353cc07a2e013ae08f91c5f373a384862669ff85d693b48fd5656320f.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/5c757f3353cc07a2e013ae08f91c5f373a384862669ff85d693b48fd5656320f.jpg)

![6555286ecea960a90eebea3e8df9410e7489fb636378929d0a7b3989c2590085.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/6555286ecea960a90eebea3e8df9410e7489fb636378929d0a7b3989c2590085.jpg)

![7957aefa70bcc4faa87c64a1396ac8dcf139d01adbb72e4d2c9604356857aecd.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/7957aefa70bcc4faa87c64a1396ac8dcf139d01adbb72e4d2c9604356857aecd.jpg)

![7b3a17519dd8f0aeea5fd50d51a39f8a4fcf93d0c042c0cd95ec991bfcdceab8.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/7b3a17519dd8f0aeea5fd50d51a39f8a4fcf93d0c042c0cd95ec991bfcdceab8.jpg)

![8234961dbf4ac5d4ceef9b0b7e96f635effbed5a8569c8053d53c53df262a73e.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/8234961dbf4ac5d4ceef9b0b7e96f635effbed5a8569c8053d53c53df262a73e.jpg)

![89bab0c91a0f2abd36497fb7cafe66dfdfc0f1128a9c1a30e684e076f3c8e594.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/89bab0c91a0f2abd36497fb7cafe66dfdfc0f1128a9c1a30e684e076f3c8e594.jpg)

![8fa5456568a2f3f8b50ac2f73a8b4ed5e989c581f944565438c6c785ae20f559.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/8fa5456568a2f3f8b50ac2f73a8b4ed5e989c581f944565438c6c785ae20f559.jpg)

![96a8b7393dadc6e3574bcb818be81b97bde4cd42145ec6897dde483340866751.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/96a8b7393dadc6e3574bcb818be81b97bde4cd42145ec6897dde483340866751.jpg)

![a5dca23c71008f6a3811b578f13768440c9657f9625c74ef708a4c78b8c2be10.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/a5dca23c71008f6a3811b578f13768440c9657f9625c74ef708a4c78b8c2be10.jpg)

![af3544787522bd4fbb8bd2a6c817b43c32f53ccf105442d8f53c4a7c7457c4b6.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/af3544787522bd4fbb8bd2a6c817b43c32f53ccf105442d8f53c4a7c7457c4b6.jpg)

![b0700cba5a1245dd59e181a0ee58e48fc03234bf1183269841254e48a70701b9.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/b0700cba5a1245dd59e181a0ee58e48fc03234bf1183269841254e48a70701b9.jpg)

![bae0776362dfa001c0ba86ab7332fcd6885f37861776ecbeca7dd2538ac9920f.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/bae0776362dfa001c0ba86ab7332fcd6885f37861776ecbeca7dd2538ac9920f.jpg)

![bf454345e2424c98e2595d7b219a136d33c2f77930e0d36ed6b95b0b61430c43.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/bf454345e2424c98e2595d7b219a136d33c2f77930e0d36ed6b95b0b61430c43.jpg)

![c1b9aac3212f27734e28b95ef8eb789f118d7fa4da4e4ead30731f33e67a3245.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/c1b9aac3212f27734e28b95ef8eb789f118d7fa4da4e4ead30731f33e67a3245.jpg)

![d2657c76ef7f19f25db8eb51c2e4dd12aca2803c8b7468c7328bfbe866aadcdb.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/d2657c76ef7f19f25db8eb51c2e4dd12aca2803c8b7468c7328bfbe866aadcdb.jpg)

![e629107f9a491869cda4c4da1b62704bb4c4706d9010ac3ca68116b799d560f6.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/e629107f9a491869cda4c4da1b62704bb4c4706d9010ac3ca68116b799d560f6.jpg)

![ea8e3d02f72357a619c1ed89da9e98e8f3fdb78084f139df26c0886dff90edb9.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/ea8e3d02f72357a619c1ed89da9e98e8f3fdb78084f139df26c0886dff90edb9.jpg)

![f5ebe7c96512b63b423d3a27184b0f6c8a3e4fc5725d76f662ee9885f9851844.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/images/f5ebe7c96512b63b423d3a27184b0f6c8a3e4fc5725d76f662ee9885f9851844.jpg)

### Tables

![712efa91530f77c7b7b7b58f301cc8295278df4792cbef1518eb4a81762f7e04.jpg](../icml_results/5401_Spurious%20Rewards%20Paradox_%20Mechanistically%20Understanding%20How%20RLVR%20Activates%20Memorization%20Shortcuts%20in/tables/712efa91530f77c7b7b7b58f301cc8295278df4792cbef1518eb4a81762f7e04.jpg)

<a id="paper-5354"></a>

## Rethink the Role of Neural Decoders in Quantum Error Correction


### Images

![0604c676861c016c80df28f48dcdb3fb6ec9f42baded0d67f5bfdc34bf0591b7.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/0604c676861c016c80df28f48dcdb3fb6ec9f42baded0d67f5bfdc34bf0591b7.jpg)

![0b2f885c7288128c7b777fc2aa765eec0182a909f3fb2bb554b251bbbfcda1c0.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/0b2f885c7288128c7b777fc2aa765eec0182a909f3fb2bb554b251bbbfcda1c0.jpg)

![2f0d8da8fdecdf73b42003f873757ab376ca8b60e3c0b31ffc28e10f11ea8a2c.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/2f0d8da8fdecdf73b42003f873757ab376ca8b60e3c0b31ffc28e10f11ea8a2c.jpg)

![30b9bea5d7d95b1f5db045e321edc079de5d162b189d18fe977d18e7f01c1624.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/30b9bea5d7d95b1f5db045e321edc079de5d162b189d18fe977d18e7f01c1624.jpg)

![48e63490767c7aaf244ac5f19b24b86d85df3d7a56945ec9332992b7e8e86d08.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/48e63490767c7aaf244ac5f19b24b86d85df3d7a56945ec9332992b7e8e86d08.jpg)

![5269bfb0cc3ea39d7feef54b72dd2aa3df69f214fb8740e1be25fb20944084a3.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/5269bfb0cc3ea39d7feef54b72dd2aa3df69f214fb8740e1be25fb20944084a3.jpg)

![698e30cfb13fda72dc5dfe806d7bcaaa5d27b404908413a4b0a9c9087292e012.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/698e30cfb13fda72dc5dfe806d7bcaaa5d27b404908413a4b0a9c9087292e012.jpg)

![9faa8f1547e54c9fae5693c84d6aaf3e63181afb5363cb99e1b92f59307e6a2f.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/9faa8f1547e54c9fae5693c84d6aaf3e63181afb5363cb99e1b92f59307e6a2f.jpg)

![a7a677a4a35a5b9ebb2598f3e1cfa70beae44688ae0f98f7a7840b1835885e0f.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/a7a677a4a35a5b9ebb2598f3e1cfa70beae44688ae0f98f7a7840b1835885e0f.jpg)

![c346154ceb5559c1f1203f35559a302581323d203d5547a7905b153c53e06ce3.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/c346154ceb5559c1f1203f35559a302581323d203d5547a7905b153c53e06ce3.jpg)

![d64f5abf67bddb9c41b490a93d6eb5a3f17d01475c1eac339a46da81f2ef09ba.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/images/d64f5abf67bddb9c41b490a93d6eb5a3f17d01475c1eac339a46da81f2ef09ba.jpg)

### Tables

![14dd4ebb316fdb2c20b2c85f5019de9045f354e209e988c44ba12e5477aefb2c.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/14dd4ebb316fdb2c20b2c85f5019de9045f354e209e988c44ba12e5477aefb2c.jpg)

![269ed22c745710362915d86874690f4dc3bccd14bbac066dc6a34a87f0ce4389.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/269ed22c745710362915d86874690f4dc3bccd14bbac066dc6a34a87f0ce4389.jpg)

![2c1e26c8824a1af00d5635e75770a5f8663da747a7483a537abfe3473c175b4d.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/2c1e26c8824a1af00d5635e75770a5f8663da747a7483a537abfe3473c175b4d.jpg)

![32a432f8cda7e6281a82bd1bd98e957a853a76d001bb52c16a310aac11a3a380.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/32a432f8cda7e6281a82bd1bd98e957a853a76d001bb52c16a310aac11a3a380.jpg)

![3bcdbb9c3e4dd28edccf8fadbff130f833cb2c992b07f2add835d8f85f0b12c6.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/3bcdbb9c3e4dd28edccf8fadbff130f833cb2c992b07f2add835d8f85f0b12c6.jpg)

![4b6ddaf964fb80693a58f6712178b40f6d8f7c8af7e7c0d8a7d239f368e12c70.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/4b6ddaf964fb80693a58f6712178b40f6d8f7c8af7e7c0d8a7d239f368e12c70.jpg)

![60355ec0f573175863f7360c6d9d5e6d6225b648c65afec5278a6dde91ddb141.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/60355ec0f573175863f7360c6d9d5e6d6225b648c65afec5278a6dde91ddb141.jpg)

![67e21aa1ea24b8101abfd1690e15827263f846b507695001bd4e2c29c251a9a4.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/67e21aa1ea24b8101abfd1690e15827263f846b507695001bd4e2c29c251a9a4.jpg)

![6f424ea542d71bb891a1d75829437c47a8896c8dbd0a258b45d58c7aaf4c18b6.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/6f424ea542d71bb891a1d75829437c47a8896c8dbd0a258b45d58c7aaf4c18b6.jpg)

![81150f344d504adf9db169d88408b1ce1da610ac227320a1a3ad173307a3818c.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/81150f344d504adf9db169d88408b1ce1da610ac227320a1a3ad173307a3818c.jpg)

![a7b4d5c453ddc1d2c536d9ad1b5d5e10adafefe1c6511c211bf7a89f80d5082a.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/a7b4d5c453ddc1d2c536d9ad1b5d5e10adafefe1c6511c211bf7a89f80d5082a.jpg)

![af31ad6c48a52cdcd985071befb063a920d4500c633ef35d18a044a946d32cf2.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/af31ad6c48a52cdcd985071befb063a920d4500c633ef35d18a044a946d32cf2.jpg)

![c8239bf0a031e9c7e05fd0fecb3a7d3fc1da40c7a4e89159a9100d1eecc1d964.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/c8239bf0a031e9c7e05fd0fecb3a7d3fc1da40c7a4e89159a9100d1eecc1d964.jpg)

![d1ce80ddc7465d31b1f30d08e69b86247cc2c94ba9d12dcb10819255a3b3ca8c.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/d1ce80ddc7465d31b1f30d08e69b86247cc2c94ba9d12dcb10819255a3b3ca8c.jpg)

![db98916f2d66569ac59d05d52a3bd74c9ebd237476918ed9f561c3fd43b030e9.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/db98916f2d66569ac59d05d52a3bd74c9ebd237476918ed9f561c3fd43b030e9.jpg)

![dbee0a7a385257d7f4a3a8eda61d806d39155119c41dcca80c8fa6e9b13ca518.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/dbee0a7a385257d7f4a3a8eda61d806d39155119c41dcca80c8fa6e9b13ca518.jpg)

![e1a86491770852b52a72eb92cb26de5df6bb724ba9358e7c300faeccce2846f3.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/e1a86491770852b52a72eb92cb26de5df6bb724ba9358e7c300faeccce2846f3.jpg)

![e3b91d06565ace785f6ab8ec1c0bce84716fad0385439e7b43e18eed43d8fe70.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/e3b91d06565ace785f6ab8ec1c0bce84716fad0385439e7b43e18eed43d8fe70.jpg)

![e6fc47647a2692692fe1e0daa7e058ba63892a15f7df1437ace6bb2d5a80e479.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/e6fc47647a2692692fe1e0daa7e058ba63892a15f7df1437ace6bb2d5a80e479.jpg)

![f0f7fc0d0e8de46c8cfea97019c353b626f2e4e15719c873556cdd074c2a742a.jpg](../icml_results/5402_Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/tables/f0f7fc0d0e8de46c8cfea97019c353b626f2e4e15719c873556cdd074c2a742a.jpg)

<a id="paper-5355"></a>

## Rethinking Generative Image Pretraining: How Far Are We From Scaling Up Next-Pixel Prediction?


### Images

![06118348aec5bfae81ae688c00ad721c2a7da5def2b4bab1b1b8f39d6f7d851b.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/06118348aec5bfae81ae688c00ad721c2a7da5def2b4bab1b1b8f39d6f7d851b.jpg)

![187ab9985dc591725131fbbc7b89a53006a2053e9ac8453a89d176e4dc0acf3f.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/187ab9985dc591725131fbbc7b89a53006a2053e9ac8453a89d176e4dc0acf3f.jpg)

![2923f8164bf61a4399d31c09d86fc496df46f8940e3ec28dd1f7831663b0b3db.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/2923f8164bf61a4399d31c09d86fc496df46f8940e3ec28dd1f7831663b0b3db.jpg)

![3d579829c482df8689764b9a1b7a58da86439edb59a4c187b05c7fb224516962.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/3d579829c482df8689764b9a1b7a58da86439edb59a4c187b05c7fb224516962.jpg)

![71fb8fed6ef0f0016a30f5f14e920e889434763f56bf875437d4ec0d35804d08.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/71fb8fed6ef0f0016a30f5f14e920e889434763f56bf875437d4ec0d35804d08.jpg)

![8cd17b7a860958f4b1baf67aae292d2fabe3a5621a6a5cfa281cb51bab313642.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/8cd17b7a860958f4b1baf67aae292d2fabe3a5621a6a5cfa281cb51bab313642.jpg)

![8f4fad831cf7c1e53a5ddcbb90cdba2b88f37f98bb45a743345a01614c4cfc66.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/8f4fad831cf7c1e53a5ddcbb90cdba2b88f37f98bb45a743345a01614c4cfc66.jpg)

![9b3f47bf309b824f60ed3868709d115809762c493a7ba0212123dd5af050f39f.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/9b3f47bf309b824f60ed3868709d115809762c493a7ba0212123dd5af050f39f.jpg)

![9c3c2ab0ba00473c863743be536f28bc6e05b76c2c2c0e5037f18f1f64cc1fbf.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/9c3c2ab0ba00473c863743be536f28bc6e05b76c2c2c0e5037f18f1f64cc1fbf.jpg)

![be4763b3f62de974e13fe0fc21489cbfc1b414115e693b3743da5c4701161688.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/be4763b3f62de974e13fe0fc21489cbfc1b414115e693b3743da5c4701161688.jpg)

![ce5a4a0cd616fa91aaeabc7680825a952dcbd567d01591a57f062b305a8a4e5a.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/ce5a4a0cd616fa91aaeabc7680825a952dcbd567d01591a57f062b305a8a4e5a.jpg)

![d0543c096b8c32dbbf5e78121aa960ac01d77d6333be6ce787915cb77c2d887e.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/d0543c096b8c32dbbf5e78121aa960ac01d77d6333be6ce787915cb77c2d887e.jpg)

![de91e98e9a1d42f638bf41ee7671e90006fe5ebf59efe9a07f7a101492ff8d17.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/de91e98e9a1d42f638bf41ee7671e90006fe5ebf59efe9a07f7a101492ff8d17.jpg)

![fdb0d3e5056d1d1dd73d47a627087c8c35bfcdc12ab6814345eceff0cde4323f.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/images/fdb0d3e5056d1d1dd73d47a627087c8c35bfcdc12ab6814345eceff0cde4323f.jpg)

### Tables

![2a6d0670534cf739bde9c7826ff77a83cff5f94bd7d6ae22ce04a35e0170356f.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/tables/2a6d0670534cf739bde9c7826ff77a83cff5f94bd7d6ae22ce04a35e0170356f.jpg)

![5f3512459d1fef48859b1c48a361148dda03a25ba46ca0953d7884d48af2d2fb.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/tables/5f3512459d1fef48859b1c48a361148dda03a25ba46ca0953d7884d48af2d2fb.jpg)

![7beac0d03aa71eb3f00ddb7ec0432d9316e7fb2054405b8d16f83e749604082e.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/tables/7beac0d03aa71eb3f00ddb7ec0432d9316e7fb2054405b8d16f83e749604082e.jpg)

![98cce2cd88444a7eef17ee29a15557374ea232f5e4bee1a09a655fe7e3ae89a9.jpg](../icml_results/5403_Rethinking%20Generative%20Image%20Pretraining_%20How%20Far%20Are%20We%20From%20Scaling%20Up%20Next-Pixel%20Prediction_/tables/98cce2cd88444a7eef17ee29a15557374ea232f5e4bee1a09a655fe7e3ae89a9.jpg)

<a id="paper-5356"></a>

## GLARE: Scalable Neuro-Symbolic Reward Shaping for LLM Agents via Group-Level Automata


### Images

![4a41f50881402fdfa75fc9cc2434ff6e338df7fe05dd514cd48e70973cab16da.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/4a41f50881402fdfa75fc9cc2434ff6e338df7fe05dd514cd48e70973cab16da.jpg)

![7b808dd075b8a02ac462906f801c2da94da555bf4477f0467c67a9c28c10ee7f.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/7b808dd075b8a02ac462906f801c2da94da555bf4477f0467c67a9c28c10ee7f.jpg)

![7c37daa338200e1f7947ae36ca53fc9fe39915aa1e07833d2901ecf336bd2e39.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/7c37daa338200e1f7947ae36ca53fc9fe39915aa1e07833d2901ecf336bd2e39.jpg)

![82dbf3b48f67d0fd380ca7f3e9d991ce3574e30c1fb074920407759d7df226a2.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/82dbf3b48f67d0fd380ca7f3e9d991ce3574e30c1fb074920407759d7df226a2.jpg)

![9b353591a17875d5173dd9cbc2e95f5eced7e354e0b4fba456a200925759fa57.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/9b353591a17875d5173dd9cbc2e95f5eced7e354e0b4fba456a200925759fa57.jpg)

![c034b7136eb72b5d46309960a56d8804ee3f525fa165b038a9b732613a97ea23.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/c034b7136eb72b5d46309960a56d8804ee3f525fa165b038a9b732613a97ea23.jpg)

![c4836e92a7c8276dfa6f0150bd308d11c9ab80523a777e6eff31ded88f64d9ff.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/c4836e92a7c8276dfa6f0150bd308d11c9ab80523a777e6eff31ded88f64d9ff.jpg)

![dd279a5afe0008c5f03fbdf78ca9ff771493baa4e1b387667b01ba7163de231e.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/dd279a5afe0008c5f03fbdf78ca9ff771493baa4e1b387667b01ba7163de231e.jpg)

![de66776de57bb06c6e286267133f8dca048e0c87c01e515b8d86744b7b6c7e4f.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/de66776de57bb06c6e286267133f8dca048e0c87c01e515b8d86744b7b6c7e4f.jpg)

![de74d44a810e2b5984c5d8174dc9ca4a5bb415434f937f4fe245792864e85b66.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/de74d44a810e2b5984c5d8174dc9ca4a5bb415434f937f4fe245792864e85b66.jpg)

![f1256c3b5ec4d3565682fe0ec6e36b92664f252116a3efcbb1fac2cd4c1dcf7b.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/images/f1256c3b5ec4d3565682fe0ec6e36b92664f252116a3efcbb1fac2cd4c1dcf7b.jpg)

### Tables

![3764ac86b227d375925524058df785d153105c2ceb99597c162f9a1c15f3a375.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/tables/3764ac86b227d375925524058df785d153105c2ceb99597c162f9a1c15f3a375.jpg)

![eb718087c6aabfe9d926b9a422843177b79980870bbf95090f77bf6726895c61.jpg](../icml_results/5404_GLARE_%20Scalable%20Neuro-Symbolic%20Reward%20Shaping%20for%20LLM%20Agents%20via%20Group-Level%20Automata/tables/eb718087c6aabfe9d926b9a422843177b79980870bbf95090f77bf6726895c61.jpg)

<a id="paper-5357"></a>

## Invariant Representation Learning for Source-Free Time Series Forecasting with LLM-Centric Proxy Denoising


### Images

![04f0e70ab2462788421da4ee887fa814245b628656f60cf6edcf169fd44ae792.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/04f0e70ab2462788421da4ee887fa814245b628656f60cf6edcf169fd44ae792.jpg)

![1a3b04146bc2590c72dbad2ffdf162bbf12d4000b6fa2d7fc12499b9da2f986e.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/1a3b04146bc2590c72dbad2ffdf162bbf12d4000b6fa2d7fc12499b9da2f986e.jpg)

![2fab31c567ce4e3dfadba7f1fa847b946e7287372b54bfa544f7d95f96d61dd7.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/2fab31c567ce4e3dfadba7f1fa847b946e7287372b54bfa544f7d95f96d61dd7.jpg)

![3044e3635675127b46441f190cc5bad62cc0f06de4711ad5ff5730207552665e.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/3044e3635675127b46441f190cc5bad62cc0f06de4711ad5ff5730207552665e.jpg)

![36247ca407dcb7469eede1ca73b2771cf7f860c6465efe8a1feb370d1c9b5c64.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/36247ca407dcb7469eede1ca73b2771cf7f860c6465efe8a1feb370d1c9b5c64.jpg)

![39508d5d9556d6729b26b6fd895760ab28cfd8c5c83549119df0c7b98d989ae6.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/39508d5d9556d6729b26b6fd895760ab28cfd8c5c83549119df0c7b98d989ae6.jpg)

![3cc2c927c2b96ca452200dd681610ad652353388c2e303a693b65d1f2144ac6f.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/3cc2c927c2b96ca452200dd681610ad652353388c2e303a693b65d1f2144ac6f.jpg)

![456d0e471e4ab69a48f79986f69cb60a6f0ebb17c36896fb49b7e180f8d9d4f8.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/456d0e471e4ab69a48f79986f69cb60a6f0ebb17c36896fb49b7e180f8d9d4f8.jpg)

![65d26c6838b9fd13d2179de6a06370a33106d144be96c6b742ee01b99ecdb84c.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/65d26c6838b9fd13d2179de6a06370a33106d144be96c6b742ee01b99ecdb84c.jpg)

![877daaf29dee8f92d07bd49ca23ec7709410a1bf051fbae4db6ebbafbd5d0c9a.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/877daaf29dee8f92d07bd49ca23ec7709410a1bf051fbae4db6ebbafbd5d0c9a.jpg)

![8a741597f3d08902a9476e0e15f0d53b41b07e8e04b02acfa3d0eb5e1c8a8bb8.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/8a741597f3d08902a9476e0e15f0d53b41b07e8e04b02acfa3d0eb5e1c8a8bb8.jpg)

![9aa76bd018709f74613e15f0d1698667f0eb82d2c7c4e938bb9c44945047be1e.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/9aa76bd018709f74613e15f0d1698667f0eb82d2c7c4e938bb9c44945047be1e.jpg)

![b8bf75fb3ee242c400f43cf8d3e9159b1b07f949b4432620e42c9c29b7d55677.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/b8bf75fb3ee242c400f43cf8d3e9159b1b07f949b4432620e42c9c29b7d55677.jpg)

![bd406e350410526e237b355a187fe62803cb0acdcc60b59335f7d82508ec9784.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/bd406e350410526e237b355a187fe62803cb0acdcc60b59335f7d82508ec9784.jpg)

![d82f6f6d1eb18a6f2b0e4e7de1bbb97a618a2c0734307b9bfba04cd34bacfbea.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/d82f6f6d1eb18a6f2b0e4e7de1bbb97a618a2c0734307b9bfba04cd34bacfbea.jpg)

![e126014d90ef757d6679393f0727496366a2857a881a35df48eb2352e5d1a18a.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/e126014d90ef757d6679393f0727496366a2857a881a35df48eb2352e5d1a18a.jpg)

![e2291776c459a4ec09404a3ba966cfda022d29326c6348f9ed714e269e8dfde3.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/e2291776c459a4ec09404a3ba966cfda022d29326c6348f9ed714e269e8dfde3.jpg)

![e76b5d0ce0fe82c9ab8d4266eecf7e669ae64b99ef3d1435d549d51e118c4a5f.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/e76b5d0ce0fe82c9ab8d4266eecf7e669ae64b99ef3d1435d549d51e118c4a5f.jpg)

![ef66baa31d27ecba59044a83237500e5e76f25beadd0b18c8fc5c89e8a7fa9a5.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/ef66baa31d27ecba59044a83237500e5e76f25beadd0b18c8fc5c89e8a7fa9a5.jpg)

![f915f393d99e7ac1cda62e86e9cfb59019b18c857de92811fbf040fde11f6336.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/images/f915f393d99e7ac1cda62e86e9cfb59019b18c857de92811fbf040fde11f6336.jpg)

### Tables

![0470c9850b6084201d29c309e1c447c797f0ee89ba67c307b62fa0d6d6568ba6.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/0470c9850b6084201d29c309e1c447c797f0ee89ba67c307b62fa0d6d6568ba6.jpg)

![0910ceb6e08333972f5a8a706d6e421787f1faf34fc6752e0b1c4be1c552bd74.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/0910ceb6e08333972f5a8a706d6e421787f1faf34fc6752e0b1c4be1c552bd74.jpg)

![3fb3f8adf8c972a8832c1eab20b60e1a63a4259d74f353f4e065688d2c6da3e8.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/3fb3f8adf8c972a8832c1eab20b60e1a63a4259d74f353f4e065688d2c6da3e8.jpg)

![539bc6bae06b81e011790ba0b250c1106860e093682e6be23992cba18405b7c5.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/539bc6bae06b81e011790ba0b250c1106860e093682e6be23992cba18405b7c5.jpg)

![6743d5f6f8748b5bc7d7ab1dc9ecda48c34ed46d63ebbc64cbd9e6941ca03053.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/6743d5f6f8748b5bc7d7ab1dc9ecda48c34ed46d63ebbc64cbd9e6941ca03053.jpg)

![75f964f4c1414f3563b38a162aa855f82bd266a81a3df1466f9c212bc1d6b959.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/75f964f4c1414f3563b38a162aa855f82bd266a81a3df1466f9c212bc1d6b959.jpg)

![c69bd7385f8105be93ba688a12390ad3e1cbf1aa1a7816f2c581f7783e0bab09.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/c69bd7385f8105be93ba688a12390ad3e1cbf1aa1a7816f2c581f7783e0bab09.jpg)

![e067e9640842399d4f769fa912e3164295d1f079025ccdba6cfb9d989a892e77.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/e067e9640842399d4f769fa912e3164295d1f079025ccdba6cfb9d989a892e77.jpg)

![e3dd2bb3e4728dc666a30fc196a64f605bb42a4bb041b4b5e871d4e78b2144cf.jpg](../icml_results/5405_Invariant%20Representation%20Learning%20for%20Source-Free%20Time%20Series%20Forecasting%20with%20LLM-Centric%20Proxy%20Den/tables/e3dd2bb3e4728dc666a30fc196a64f605bb42a4bb041b4b5e871d4e78b2144cf.jpg)

<a id="paper-5358"></a>

## Design Linear Constrained Neural Layers with Implicit Convex Optimization


### Images

![880de0884636c05191331e85b09065fc482c97147bbe336ee227132e29c2b895.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/images/880de0884636c05191331e85b09065fc482c97147bbe336ee227132e29c2b895.jpg)

![ac6e624f6cfaf04b672fb942161dc7643b3d36ded6ca411ed41cc35e18e6938d.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/images/ac6e624f6cfaf04b672fb942161dc7643b3d36ded6ca411ed41cc35e18e6938d.jpg)

### Tables

![02859263f2ebb7db6fea64862601a6e95d1724ff02f460e92f07a076f43b652c.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/02859263f2ebb7db6fea64862601a6e95d1724ff02f460e92f07a076f43b652c.jpg)

![3e2409595aaa0ed62f61fb053736b27bdc4c57f803a140a1b71e6fdee0d2c82a.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/3e2409595aaa0ed62f61fb053736b27bdc4c57f803a140a1b71e6fdee0d2c82a.jpg)

![52fef53f49e0adbd4676251310807ebc4ed56260d3ec55b3cb3440bb6fafe3d5.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/52fef53f49e0adbd4676251310807ebc4ed56260d3ec55b3cb3440bb6fafe3d5.jpg)

![62d5c379a21d2f32b0c36699a45f68f93b5000c99e3659b301886ec8deb32fcf.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/62d5c379a21d2f32b0c36699a45f68f93b5000c99e3659b301886ec8deb32fcf.jpg)

![7628629054764844b9a28db7f112944b179c53a5c9d92114a279090cfb063dd9.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/7628629054764844b9a28db7f112944b179c53a5c9d92114a279090cfb063dd9.jpg)

![a3000ab73db9301d61e122c6772087628911dfc9e50fef9d30f9b9f986b13472.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/a3000ab73db9301d61e122c6772087628911dfc9e50fef9d30f9b9f986b13472.jpg)

![ab8ecbbe654fd657deec1b4d9da85319fa2a453cd9996f7f11ec722ce65e9097.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/ab8ecbbe654fd657deec1b4d9da85319fa2a453cd9996f7f11ec722ce65e9097.jpg)

![b1b8ed04186bc61a05a25e3bc832f9f1159ec02d1cf0a1ae3d8f11e5eb92d8a8.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/b1b8ed04186bc61a05a25e3bc832f9f1159ec02d1cf0a1ae3d8f11e5eb92d8a8.jpg)

![b52340c62a00d6f963b0ed897b0af89063e790ab483bc65023c58fa586f8b21f.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/b52340c62a00d6f963b0ed897b0af89063e790ab483bc65023c58fa586f8b21f.jpg)

![c4184ad49f9a60155fb130ae9d90dc0e7abfdc5891dc22d494633d1f365f17c8.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/c4184ad49f9a60155fb130ae9d90dc0e7abfdc5891dc22d494633d1f365f17c8.jpg)

![c5dce4d307a99c3ef3885625e5785e3824816c666abf462b7471793115bd0de4.jpg](../icml_results/5406_Design%20Linear%20Constrained%20Neural%20Layers%20with%20Implicit%20Convex%20Optimization/tables/c5dce4d307a99c3ef3885625e5785e3824816c666abf462b7471793115bd0de4.jpg)

<a id="paper-5359"></a>

## MemIncept: Steering LLM Agents via Cooperative Stealthy Memory Injections


### Images

![0a20db036427c4f6a2a7d6f4ddabff527d4f0f5058608154b4f847751197f729.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/0a20db036427c4f6a2a7d6f4ddabff527d4f0f5058608154b4f847751197f729.jpg)

![a3b37dd6070734c0a53acdd118877b9da506c78ad0f113ee82a8c61c59ee4099.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/a3b37dd6070734c0a53acdd118877b9da506c78ad0f113ee82a8c61c59ee4099.jpg)

![c3b5b1828751bfbb3583f0100ceea3fe70233adae0fee53e0137af3f502932d8.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/c3b5b1828751bfbb3583f0100ceea3fe70233adae0fee53e0137af3f502932d8.jpg)

![dbd6666924d21b3e22570770fdaa2b1ae847e1aab5308e0ab76ceee80714f1d2.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/dbd6666924d21b3e22570770fdaa2b1ae847e1aab5308e0ab76ceee80714f1d2.jpg)

![e7923bfc049205002ce060f1ea45339ff5fd315de55f844d851ba2fd9e62fa19.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/e7923bfc049205002ce060f1ea45339ff5fd315de55f844d851ba2fd9e62fa19.jpg)

![f2b825882ee0540a9b35ca1b99610160f4689df1afc19cdd5be70f9afb53d022.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/f2b825882ee0540a9b35ca1b99610160f4689df1afc19cdd5be70f9afb53d022.jpg)

![fbaeb9560acf20eba6af0245afd0400190e3f7a7db50185a10cd01ad73ed6f94.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/images/fbaeb9560acf20eba6af0245afd0400190e3f7a7db50185a10cd01ad73ed6f94.jpg)

### Tables

![0055a3c952af59eb7a969a1383469af3345e40bdd6b216ea36d72255b06f26ec.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/0055a3c952af59eb7a969a1383469af3345e40bdd6b216ea36d72255b06f26ec.jpg)

![04445653dc9ff429b2393d922b639d0f88f197cf6ecd4de875326eb3d2e95feb.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/04445653dc9ff429b2393d922b639d0f88f197cf6ecd4de875326eb3d2e95feb.jpg)

![0c35df831fa3762855d121c32e6801ac13932cc9027f575cb480b3c0cdd1f37c.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/0c35df831fa3762855d121c32e6801ac13932cc9027f575cb480b3c0cdd1f37c.jpg)

![0c377e91ed4e5dff597783cfc02c9f9557a45fb7b0b34835e0a380c03c3e8b0c.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/0c377e91ed4e5dff597783cfc02c9f9557a45fb7b0b34835e0a380c03c3e8b0c.jpg)

![2df754e57b8b81d254cca9d53ab423327f5a8bf7b288d14decc84553aa6ad279.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/2df754e57b8b81d254cca9d53ab423327f5a8bf7b288d14decc84553aa6ad279.jpg)

![3e41baa8bea583d6f3609bf73e1f357cf00ed153da5b2b570287f5aaae75747d.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/3e41baa8bea583d6f3609bf73e1f357cf00ed153da5b2b570287f5aaae75747d.jpg)

![4138b0d8c86fee6899f368dcc20acf182bf0c838a498e9e838dda0623b0f10ad.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/4138b0d8c86fee6899f368dcc20acf182bf0c838a498e9e838dda0623b0f10ad.jpg)

![4763cb9ce9a0fbbe7c0c8721a706962aac69015da532fa50840b6ee529b962b1.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/4763cb9ce9a0fbbe7c0c8721a706962aac69015da532fa50840b6ee529b962b1.jpg)

![6d86cafadf844cc09fcd7c53deb64bb645ed821100c5fca68a1cd33692e3b372.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/6d86cafadf844cc09fcd7c53deb64bb645ed821100c5fca68a1cd33692e3b372.jpg)

![72e80edf18361b59273b95ab626445856a3de75a506892244dd325eb013b46c5.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/72e80edf18361b59273b95ab626445856a3de75a506892244dd325eb013b46c5.jpg)

![c0060bde6e89e4c8db1e684a01f139e2e9d8742574e67dac7593931bcd85b9dc.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/c0060bde6e89e4c8db1e684a01f139e2e9d8742574e67dac7593931bcd85b9dc.jpg)

![c662c3ca2c77693770a75f850e93608f62e6548e00db0277349c2d321b6368d0.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/c662c3ca2c77693770a75f850e93608f62e6548e00db0277349c2d321b6368d0.jpg)

![fd78012a69c9babfa50be55286aa54ffda82127cbfd2d3265eff0c85b42f6ef8.jpg](../icml_results/5407_MemIncept_%20Steering%20LLM%20Agents%20via%20Cooperative%20Stealthy%20Memory%20Injections/tables/fd78012a69c9babfa50be55286aa54ffda82127cbfd2d3265eff0c85b42f6ef8.jpg)

<a id="paper-5360"></a>

## Arboreal Neural Network


### Images

![3fa1bc5a4655c1a8c91ebbaeec50e3833b99b9ac36faaf06904dfd9ec5fe77d0.jpg](../icml_results/5408_Arboreal%20Neural%20Network/images/3fa1bc5a4655c1a8c91ebbaeec50e3833b99b9ac36faaf06904dfd9ec5fe77d0.jpg)

![94521d98bb7476d77dfa88c7aa66bad4e3a56c8e85d5952b3cc669159c1c5ca8.jpg](../icml_results/5408_Arboreal%20Neural%20Network/images/94521d98bb7476d77dfa88c7aa66bad4e3a56c8e85d5952b3cc669159c1c5ca8.jpg)

![ccbeab99f17a310bd0a9745c3f57eecf064d754199fc1cbcd82d0619eecfadfd.jpg](../icml_results/5408_Arboreal%20Neural%20Network/images/ccbeab99f17a310bd0a9745c3f57eecf064d754199fc1cbcd82d0619eecfadfd.jpg)

### Tables

![224fd49e2a2b66cc3f8a421834e9ae60f318180c2a02fa847b394e39e8486f5a.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/224fd49e2a2b66cc3f8a421834e9ae60f318180c2a02fa847b394e39e8486f5a.jpg)

![4516c798643aaa51540eb651ca037ae5d6da64e829e14c7f56ceb6e031234b8e.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/4516c798643aaa51540eb651ca037ae5d6da64e829e14c7f56ceb6e031234b8e.jpg)

![54b291086b4c5306ba383f64df11f84ad73b00e0283de2899fcb6b13ccec085e.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/54b291086b4c5306ba383f64df11f84ad73b00e0283de2899fcb6b13ccec085e.jpg)

![ba4a0d816b19e43f60838954a638d7998a71f66b0d01a2fd31e46e78e6891321.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/ba4a0d816b19e43f60838954a638d7998a71f66b0d01a2fd31e46e78e6891321.jpg)

![e2e5b5c194a23629d08a2c15977189e3354e9e777eb5c89e66332057948a5d64.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/e2e5b5c194a23629d08a2c15977189e3354e9e777eb5c89e66332057948a5d64.jpg)

![e4fa1aca59516a0d7dad717e86eb4a9a21c336e337457e8b9d764c5db346fc4d.jpg](../icml_results/5408_Arboreal%20Neural%20Network/tables/e4fa1aca59516a0d7dad717e86eb4a9a21c336e337457e8b9d764c5db346fc4d.jpg)

<a id="paper-5361"></a>

## ORBIT: A Prognostic World Model for Ocular Reasoning Based on Imagined Trajectories


### Images

![20bec2b19e5a84c1ac971fe5c2b495046f00bc2ab420efc0fc51031cfaf04b5f.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/20bec2b19e5a84c1ac971fe5c2b495046f00bc2ab420efc0fc51031cfaf04b5f.jpg)

![2b94970cbaf0a6b7e0d360cbf88f62a8bad7005c41e2f4687f55f403922d3d8a.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/2b94970cbaf0a6b7e0d360cbf88f62a8bad7005c41e2f4687f55f403922d3d8a.jpg)

![2bac75af700b7c690402a41e6416382e0d69b03babfe59077df3fd893158d45a.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/2bac75af700b7c690402a41e6416382e0d69b03babfe59077df3fd893158d45a.jpg)

![2f2c2ab162aa28d8b133be02191cad545f0ed80cfb413b21cfd7fb8262b98c56.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/2f2c2ab162aa28d8b133be02191cad545f0ed80cfb413b21cfd7fb8262b98c56.jpg)

![42cc72f424c1dcd7cb2368cf1472e04b8b0cebb52963f966a9444c2486ded0ab.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/42cc72f424c1dcd7cb2368cf1472e04b8b0cebb52963f966a9444c2486ded0ab.jpg)

![46137798ab4dbe13cfaa2feb8e88e3e175936b8c6dea1a57266f31b7299a1693.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/46137798ab4dbe13cfaa2feb8e88e3e175936b8c6dea1a57266f31b7299a1693.jpg)

![508f53d161f2a2d39733ed0eec1641a0777ff2c96f2a21cbc85fa0f33db74a0c.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/508f53d161f2a2d39733ed0eec1641a0777ff2c96f2a21cbc85fa0f33db74a0c.jpg)

![523ee6310e5bbfffa549298b1a3ec131f1a18837d2b14d75f6a7a2616324279f.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/523ee6310e5bbfffa549298b1a3ec131f1a18837d2b14d75f6a7a2616324279f.jpg)

![5dfa3238669ffa5c1fd0ecd883260847608971e169e44778e5d3998319eadd51.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/5dfa3238669ffa5c1fd0ecd883260847608971e169e44778e5d3998319eadd51.jpg)

![796c7befa5608963349a1d25d1d7084ff833288a77c8611af36449cd27f5cd2c.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/796c7befa5608963349a1d25d1d7084ff833288a77c8611af36449cd27f5cd2c.jpg)

![9d940dafc194da877fe03c6f1a9ace25a33ee6b9fe9a9aae9de73698c77fbcd9.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/9d940dafc194da877fe03c6f1a9ace25a33ee6b9fe9a9aae9de73698c77fbcd9.jpg)

![9dffca68797c8a0df06539224328898895eee82ee65c21db7076bb196cfe311f.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/9dffca68797c8a0df06539224328898895eee82ee65c21db7076bb196cfe311f.jpg)

![a7467b45ca9b36317b5a19847eaa0ab212e734c1d419a30cbbeb1b831cb3c1f0.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/a7467b45ca9b36317b5a19847eaa0ab212e734c1d419a30cbbeb1b831cb3c1f0.jpg)

![a8105d794d2761e2a8e41a036771436c03527876a7af7d4f7432c7a2e096a787.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/a8105d794d2761e2a8e41a036771436c03527876a7af7d4f7432c7a2e096a787.jpg)

![a89b9f2030c1ba71c014923c52f993240701a581ff22e71bd31d83304df3559a.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/a89b9f2030c1ba71c014923c52f993240701a581ff22e71bd31d83304df3559a.jpg)

![b2ae7e4cde05f8a6fae10da4b81697985f2c2deb81e8034db73764c796716607.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/b2ae7e4cde05f8a6fae10da4b81697985f2c2deb81e8034db73764c796716607.jpg)

![bfbcbeeb53747d983453f82864e5c134f5835adf26ef380a60fb1f72c4d52cbb.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/bfbcbeeb53747d983453f82864e5c134f5835adf26ef380a60fb1f72c4d52cbb.jpg)

![ca118bc001454b7844892684ef24e102f55f3196d135fa6220335b937e685e9d.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/ca118bc001454b7844892684ef24e102f55f3196d135fa6220335b937e685e9d.jpg)

![ce86f93f7359b449055b16c7397164df98d71111ba3bfd673b6663c24092f9f2.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/ce86f93f7359b449055b16c7397164df98d71111ba3bfd673b6663c24092f9f2.jpg)

![da7d5ba717fa7c2327faba7b555df349bc490758359f249a4a99d8f7cb550027.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/da7d5ba717fa7c2327faba7b555df349bc490758359f249a4a99d8f7cb550027.jpg)

![dc38923f6e53ecf94adf9fadc02199fd1f6566d0eb28785311a5993db3b9a664.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/dc38923f6e53ecf94adf9fadc02199fd1f6566d0eb28785311a5993db3b9a664.jpg)

![e044a151832f16d720e26375dda61d883bad6e211454db3cde2703efc5111757.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/e044a151832f16d720e26375dda61d883bad6e211454db3cde2703efc5111757.jpg)

![fd1ac980c5c9ba8e249ea5d6592de5949f461d3433b0eebee90146149be2258a.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/images/fd1ac980c5c9ba8e249ea5d6592de5949f461d3433b0eebee90146149be2258a.jpg)

### Tables

![40bb1ed241b1674ed1b423b2ef0be5367b6f49b1547d4f982ab872a8c877d509.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/40bb1ed241b1674ed1b423b2ef0be5367b6f49b1547d4f982ab872a8c877d509.jpg)

![4d8dea416e7e633faf65693174a0134efc18a789433169ff6cea30ab10ca1f0c.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/4d8dea416e7e633faf65693174a0134efc18a789433169ff6cea30ab10ca1f0c.jpg)

![5c0760ba7e5db0faaddb7361955a13b329a0ea8114ef46d430a511985067a6c5.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/5c0760ba7e5db0faaddb7361955a13b329a0ea8114ef46d430a511985067a6c5.jpg)

![7c188efc7b70af37566fe2b7e3c849c4159a448a24cd97056bc74c5909bdde0c.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/7c188efc7b70af37566fe2b7e3c849c4159a448a24cd97056bc74c5909bdde0c.jpg)

![80475048c81f0f144ca27b07196285933c0f99a77cde3ac7676295a710968ef4.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/80475048c81f0f144ca27b07196285933c0f99a77cde3ac7676295a710968ef4.jpg)

![8ba28943db50056e36843ccaf67d4af259b78a7756366f7805853b30b8688a3b.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/8ba28943db50056e36843ccaf67d4af259b78a7756366f7805853b30b8688a3b.jpg)

![8d3f164850a8f28b9e330efb2e05d1b1c9b987e114a809392762088030ac5cba.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/8d3f164850a8f28b9e330efb2e05d1b1c9b987e114a809392762088030ac5cba.jpg)

![cb3f7d59369f3d39d4dc14224ecd578022747805412cd9fbeb36ca3de9b1745b.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/cb3f7d59369f3d39d4dc14224ecd578022747805412cd9fbeb36ca3de9b1745b.jpg)

![f6f971b289bc8ff523a1f5fa640d395a5f1ae4edc35e98ad5e49da98d49d25d9.jpg](../icml_results/5409_ORBIT_%20A%20Prognostic%20World%20Model%20for%20Ocular%20Reasoning%20Based%20on%20Imagined%20Trajectories/tables/f6f971b289bc8ff523a1f5fa640d395a5f1ae4edc35e98ad5e49da98d49d25d9.jpg)

<a id="paper-5362"></a>

## PLoRA: Efficient Concurrent LoRA Training for Large Language Models


### Images

![0738bc5c9a5803c837f11270e3ac848dc229ade0982c18f683ac33381c3e96ac.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/0738bc5c9a5803c837f11270e3ac848dc229ade0982c18f683ac33381c3e96ac.jpg)

![1482249792cb8f522301f6fecfa7d921db36fe650b4db8eff440eda97bb85da5.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/1482249792cb8f522301f6fecfa7d921db36fe650b4db8eff440eda97bb85da5.jpg)

![19e60ad7918818e3355d4031a5299a727d51930f165e6cc75fcfa4de9b5083df.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/19e60ad7918818e3355d4031a5299a727d51930f165e6cc75fcfa4de9b5083df.jpg)

![74831151c78bcbced722dca62c8d809bbf7a767057a087468692f5895079c439.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/74831151c78bcbced722dca62c8d809bbf7a767057a087468692f5895079c439.jpg)

![c6f01ec5e97824aeb0c2e6df937041f8bcddd274ceef78519a30ba86e896be94.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/c6f01ec5e97824aeb0c2e6df937041f8bcddd274ceef78519a30ba86e896be94.jpg)

![d7fcc3f567a9995d3419472c4bb35f7886f0d3da7678526ebde7545e48970f24.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/d7fcc3f567a9995d3419472c4bb35f7886f0d3da7678526ebde7545e48970f24.jpg)

![deb087859dd2fa7f8d8b46264b892bb2c076f9c28aba2d93b1276e81b5d6b925.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/deb087859dd2fa7f8d8b46264b892bb2c076f9c28aba2d93b1276e81b5d6b925.jpg)

![ffa48bc55c13a27768d50698c64de754a25005892090b067d06dc5814af825f6.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/images/ffa48bc55c13a27768d50698c64de754a25005892090b067d06dc5814af825f6.jpg)

### Tables

![10665fbf9d2aed5d91917a9005b1fa5c139d8c4fa3717a5c345fe45d6be10aaf.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/10665fbf9d2aed5d91917a9005b1fa5c139d8c4fa3717a5c345fe45d6be10aaf.jpg)

![23c676d8a6ff2f5d1857f12cb3a40a45b8a44127b48f087761621b480fb1ce1e.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/23c676d8a6ff2f5d1857f12cb3a40a45b8a44127b48f087761621b480fb1ce1e.jpg)

![33755e901f4ad8bab3d52926fa1308aae6ee7b6468a22afadab0d02176febab7.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/33755e901f4ad8bab3d52926fa1308aae6ee7b6468a22afadab0d02176febab7.jpg)

![b1be24943b6c95e7dc6b458978d10c3f28d030b1f2f6f0d284ef91d7c56ae217.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/b1be24943b6c95e7dc6b458978d10c3f28d030b1f2f6f0d284ef91d7c56ae217.jpg)

![b5ead8b0269f982e3a9172aa9e1b7eaacb5b5208d644f7d5a5ed1b0b43c9dabc.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/b5ead8b0269f982e3a9172aa9e1b7eaacb5b5208d644f7d5a5ed1b0b43c9dabc.jpg)

![c153d5ffffe7dcceff05b6cb94b1a5a4ce8dfd361c242ce8b731fcfd8a622271.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/c153d5ffffe7dcceff05b6cb94b1a5a4ce8dfd361c242ce8b731fcfd8a622271.jpg)

![c771e707366c9099425f3cccad25b752f62dd69336c6ca5d330605aef7270d83.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/c771e707366c9099425f3cccad25b752f62dd69336c6ca5d330605aef7270d83.jpg)

![e111d00be81f3c5713a32ca50dc65ad3afcc6f5b2f0f83d514c0b61e4f68d946.jpg](../icml_results/5410_PLoRA_%20Efficient%20Concurrent%20LoRA%20Training%20for%20Large%20Language%20Models/tables/e111d00be81f3c5713a32ca50dc65ad3afcc6f5b2f0f83d514c0b61e4f68d946.jpg)

<a id="paper-5363"></a>

## Can VLMs Diagnose and Recover from VLA Manipulation Faults?


### Images

![02fecb92f0c1a55cd19b6945d59b7bdd449fba8d442ec7fbcfbe79457df4c0b8.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/02fecb92f0c1a55cd19b6945d59b7bdd449fba8d442ec7fbcfbe79457df4c0b8.jpg)

![21456cf2f61a4669cef0422276eebc7f10e55ebda4d84dca1d29d1df0b94cb97.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/21456cf2f61a4669cef0422276eebc7f10e55ebda4d84dca1d29d1df0b94cb97.jpg)

![26b9002de2f523ea33e34cb11a13d13626f87ad1019e3c6c4c0aa22f0a769e5f.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/26b9002de2f523ea33e34cb11a13d13626f87ad1019e3c6c4c0aa22f0a769e5f.jpg)

![27f2b1d92e3ff2feb4b2e2796a81bb3a026b0366fe90bba086d188e6cf8c4d5a.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/27f2b1d92e3ff2feb4b2e2796a81bb3a026b0366fe90bba086d188e6cf8c4d5a.jpg)

![63666c9bbdba48977e7bb54ea8b8e31be051ffd800008e8c4fcc501bd0b88887.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/63666c9bbdba48977e7bb54ea8b8e31be051ffd800008e8c4fcc501bd0b88887.jpg)

![9e10a1b403105eb9e485f893b35f7a86cb239742efb4fea6d683706c2c8c6e99.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/9e10a1b403105eb9e485f893b35f7a86cb239742efb4fea6d683706c2c8c6e99.jpg)

![a9c115c6f7f63fdae777f3a785ef942e81545c8c952b25d5233dd05ae6cf7b9c.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/a9c115c6f7f63fdae777f3a785ef942e81545c8c952b25d5233dd05ae6cf7b9c.jpg)

![ebc8012fba69147ecd007bf8ffc2d7f221011e541de35a5c17791f7965d00e26.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/ebc8012fba69147ecd007bf8ffc2d7f221011e541de35a5c17791f7965d00e26.jpg)

![f0dfd2f6255ab1c6abe0d2fae8b29632b2676abf95bb6fc061b6a25c05ba0fac.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/f0dfd2f6255ab1c6abe0d2fae8b29632b2676abf95bb6fc061b6a25c05ba0fac.jpg)

![f91e59c9686a3be53e2cf1c5225becc0fa9488992a008fc9eb9b35517968eeb2.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/images/f91e59c9686a3be53e2cf1c5225becc0fa9488992a008fc9eb9b35517968eeb2.jpg)

### Tables

![075520bf2e639828c46e5429574d5a125cf55d566b8a581e01b93b27c95cc0df.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/tables/075520bf2e639828c46e5429574d5a125cf55d566b8a581e01b93b27c95cc0df.jpg)

![da1c21e48bc4e07449fead025e16c8ea2251550493c28912785ac666b90aeb1d.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/tables/da1c21e48bc4e07449fead025e16c8ea2251550493c28912785ac666b90aeb1d.jpg)

![da55b28a784f969889bf3c37900f06c545214aa8a4ed1081f74c5133fd49ff56.jpg](../icml_results/5411_Can%20VLMs%20Diagnose%20and%20Recover%20from%20VLA%20Manipulation%20Faults_/tables/da55b28a784f969889bf3c37900f06c545214aa8a4ed1081f74c5133fd49ff56.jpg)

<a id="paper-5364"></a>

## From Knowledge to Inference: Formalizing Specialized Public Health Reasoning on GlobalHealthAtlas


### Images

![11902460501eb20db60314c5a772ffcdd25b560e54273de491fa4feb9406071e.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/11902460501eb20db60314c5a772ffcdd25b560e54273de491fa4feb9406071e.jpg)

![399f3783c3dca9fdb6a30f9ff5f2d20c6b616ea6561b239b628b7b199644f1e9.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/399f3783c3dca9fdb6a30f9ff5f2d20c6b616ea6561b239b628b7b199644f1e9.jpg)

![3d2492100ebf58fd4b4eb3b86bd8490b89f7f849e2f8958f163bed9016c8da76.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/3d2492100ebf58fd4b4eb3b86bd8490b89f7f849e2f8958f163bed9016c8da76.jpg)

![7c9182cd382a8ae7dd3b72e63be4b56d518b1b7889960da9b2a3c7fae8257b67.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/7c9182cd382a8ae7dd3b72e63be4b56d518b1b7889960da9b2a3c7fae8257b67.jpg)

![acf5b517ca0e7781442815ff63b6cca7e3a400c19cbfc5957e3d0a6ea7e89636.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/acf5b517ca0e7781442815ff63b6cca7e3a400c19cbfc5957e3d0a6ea7e89636.jpg)

![b29d32a808898ba87dd16ce5ab23f59c93dbfff8fbd705c0f5b70c74b0a56d50.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/b29d32a808898ba87dd16ce5ab23f59c93dbfff8fbd705c0f5b70c74b0a56d50.jpg)

![bafb81824b05d4888b7c16ab84df5c8ad2017a4d96ff28e4607226559a3a7909.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/bafb81824b05d4888b7c16ab84df5c8ad2017a4d96ff28e4607226559a3a7909.jpg)

![bdebedf56571ea92c37c15c956c5a71172ff820fe8deaa2b996c12325695bf33.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/bdebedf56571ea92c37c15c956c5a71172ff820fe8deaa2b996c12325695bf33.jpg)

![d19f057ae55504e2c29fc54f8f655090ed0e3bab853b79420c0f6389586a4637.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/d19f057ae55504e2c29fc54f8f655090ed0e3bab853b79420c0f6389586a4637.jpg)

![e006ed7410afec62e325c0c6ce899e87c8cfcafa95f095bf71369ebb02e3de20.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/images/e006ed7410afec62e325c0c6ce899e87c8cfcafa95f095bf71369ebb02e3de20.jpg)

### Tables

![01f9228177b4c159019eb7a73cb521d03ef99aa74bde06179670a1f310d2fe6d.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/01f9228177b4c159019eb7a73cb521d03ef99aa74bde06179670a1f310d2fe6d.jpg)

![0bfa627c3454e9750c68f63528a97c5a012fd4267e3d8bfa6e3deb441702fd30.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/0bfa627c3454e9750c68f63528a97c5a012fd4267e3d8bfa6e3deb441702fd30.jpg)

![2a16cc211df5f3fd7f3f81824ba6938784a14fe175465a4722d49fbba63cac83.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/2a16cc211df5f3fd7f3f81824ba6938784a14fe175465a4722d49fbba63cac83.jpg)

![2c8bfd0efc09e9d979e2e7c5a48cdd9107a0a145cb407858266468eb9dacb3af.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/2c8bfd0efc09e9d979e2e7c5a48cdd9107a0a145cb407858266468eb9dacb3af.jpg)

![3bd688bf1247b8750a26cacc7f4adc314d7a0d1190af1bd2774af28a370563f4.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/3bd688bf1247b8750a26cacc7f4adc314d7a0d1190af1bd2774af28a370563f4.jpg)

![3eecd40ef927b247d5114fa16edcc919a8620672c84d1a7adf86cd66688354e6.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/3eecd40ef927b247d5114fa16edcc919a8620672c84d1a7adf86cd66688354e6.jpg)

![56761171a02e534abd9f8d5ea039e29ee914e2b48439b2db3bb4eddd938d2a1c.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/56761171a02e534abd9f8d5ea039e29ee914e2b48439b2db3bb4eddd938d2a1c.jpg)

![589f92cf141d5144bf1f1e96b9cf4037e040287c5fff9fefa2c95a706caa6e95.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/589f92cf141d5144bf1f1e96b9cf4037e040287c5fff9fefa2c95a706caa6e95.jpg)

![7001d4d6f91c41a316f2a9efe3878b88cba6304737287d41bda243c63fd00006.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/7001d4d6f91c41a316f2a9efe3878b88cba6304737287d41bda243c63fd00006.jpg)

![7ba114d70eba2c2ab7f48af13d7f7951471e56f6b37434075f96b384b7c95fce.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/7ba114d70eba2c2ab7f48af13d7f7951471e56f6b37434075f96b384b7c95fce.jpg)

![85585ff5c8fe471de9cc0711613277aca5f25c7b50517a18074645d2c0d08af9.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/85585ff5c8fe471de9cc0711613277aca5f25c7b50517a18074645d2c0d08af9.jpg)

![86f919af9db3f19d17b72d1bb0970341fce1efe909f9536056d256d1a681db87.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/86f919af9db3f19d17b72d1bb0970341fce1efe909f9536056d256d1a681db87.jpg)

![88afe6aec6a1dd74682af4eb7c274edab2a0b5b282d3a7a3e59e21a91d6ee9c1.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/88afe6aec6a1dd74682af4eb7c274edab2a0b5b282d3a7a3e59e21a91d6ee9c1.jpg)

![8d2309dde3beeb01b9a3e941542c5a16448ace551249bc91c82c30001a936b30.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/8d2309dde3beeb01b9a3e941542c5a16448ace551249bc91c82c30001a936b30.jpg)

![947a8a5383b0e650fac9c3e22a4482bada0e53fb19857f24115d7441b16d06b1.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/947a8a5383b0e650fac9c3e22a4482bada0e53fb19857f24115d7441b16d06b1.jpg)

![9f629b1f9a8b9fb7cc8ca7d1a0bd3691e544245acd7fbfeded7fdc874b2aadca.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/9f629b1f9a8b9fb7cc8ca7d1a0bd3691e544245acd7fbfeded7fdc874b2aadca.jpg)

![a6029d132460368b1e552695d371f5a601bc6feb69a40a94f339432e22a5995b.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/a6029d132460368b1e552695d371f5a601bc6feb69a40a94f339432e22a5995b.jpg)

![b138ce22450ff8ee9267b4ed06446f532e957cab9228be01fda2f2fd6988af97.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/b138ce22450ff8ee9267b4ed06446f532e957cab9228be01fda2f2fd6988af97.jpg)

![b8d5669e2f0f2fe7ca314e8b7c15cc8fe678c4348c23768b2d66e8816872f9fb.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/b8d5669e2f0f2fe7ca314e8b7c15cc8fe678c4348c23768b2d66e8816872f9fb.jpg)

![df1e1a78bef41ea0f147b6020470cf595ddc2e21ca318167cd1e56cf993bcdb7.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/df1e1a78bef41ea0f147b6020470cf595ddc2e21ca318167cd1e56cf993bcdb7.jpg)

![df8a2cab5df839d52456c3ef5b3f6f8be352c0284eda8ca868240f0ad127c968.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/df8a2cab5df839d52456c3ef5b3f6f8be352c0284eda8ca868240f0ad127c968.jpg)

![e1d009683706572c8e07f9511328646302ae5e41bf1800a8b76c381cffc095a2.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/e1d009683706572c8e07f9511328646302ae5e41bf1800a8b76c381cffc095a2.jpg)

![eab5599cf974cf725b051f3d06714c6fdb5fc46938efe18e1aa370207a629070.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/eab5599cf974cf725b051f3d06714c6fdb5fc46938efe18e1aa370207a629070.jpg)

![f17b88817abd366d5e4010e09cc25e9e2b7fd8698e69d194fcfbe24a86ee37a9.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/f17b88817abd366d5e4010e09cc25e9e2b7fd8698e69d194fcfbe24a86ee37a9.jpg)

![f3fc8587a13be177358beb7413d08474805cd8f9a2d87bdea61339ff0137a897.jpg](../icml_results/5412_From%20Knowledge%20to%20Inference_%20Formalizing%20Specialized%20Public%20Health%20Reasoning%20on%20GlobalHealthAtlas/tables/f3fc8587a13be177358beb7413d08474805cd8f9a2d87bdea61339ff0137a897.jpg)

<a id="paper-5365"></a>

## CircuitPrint: Mechanistic Circuit Fingerprints for Large Language Models


### Images

![1030d2d766f2602c4595f7db9c2b605b92a3bc4f4e49ca7097a6ac6dc721ba5f.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/1030d2d766f2602c4595f7db9c2b605b92a3bc4f4e49ca7097a6ac6dc721ba5f.jpg)

![10c7c44f3da4003d41ef9a7f88dd55211818928477fd9b6b5fa16a60e532a960.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/10c7c44f3da4003d41ef9a7f88dd55211818928477fd9b6b5fa16a60e532a960.jpg)

![1ea86a088d026d043cb79504cfd1682adae25087e8d719ec0de4817ce074a7e5.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/1ea86a088d026d043cb79504cfd1682adae25087e8d719ec0de4817ce074a7e5.jpg)

![3527ae1bcffdd97476dbe95917444fac43efafa6688c55ef54ebe8a8d2863aae.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/3527ae1bcffdd97476dbe95917444fac43efafa6688c55ef54ebe8a8d2863aae.jpg)

![394c8bda6d9b988e0c56dabfa45506a7ec9a3f28b5b5fc49589199208bfcf226.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/394c8bda6d9b988e0c56dabfa45506a7ec9a3f28b5b5fc49589199208bfcf226.jpg)

![3a4fbac14336539655b56f8c3c933d5b9530278ce1bbc1b45a18188d0c96afbd.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/3a4fbac14336539655b56f8c3c933d5b9530278ce1bbc1b45a18188d0c96afbd.jpg)

![4cf577be87ff90f5d08b0208f655df05b4e51af458b3adba61a4ed1fd88b3997.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/4cf577be87ff90f5d08b0208f655df05b4e51af458b3adba61a4ed1fd88b3997.jpg)

![4fed9ca558f036acaf6716251a3811f1c4e7fd9f97d9ce464232a0e87fe32622.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/4fed9ca558f036acaf6716251a3811f1c4e7fd9f97d9ce464232a0e87fe32622.jpg)

![58c09804fb352d7799aecedf74104c90dfb016497666c587e5f1c04571e46dca.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/58c09804fb352d7799aecedf74104c90dfb016497666c587e5f1c04571e46dca.jpg)

![5c279f523ed004ec147387f1e3eb9146d2fdaa6e50ee4b705f3f2263b2f5c097.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/5c279f523ed004ec147387f1e3eb9146d2fdaa6e50ee4b705f3f2263b2f5c097.jpg)

![67c1d796ca20282fd5617756abafb3eb518fe60f2569d4716dc90fca99b36ef7.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/67c1d796ca20282fd5617756abafb3eb518fe60f2569d4716dc90fca99b36ef7.jpg)

![7ada48f0e0b125f4e41d9634cab9d857760fd9ed6d411a42d5b2dfb4b7a0d4fc.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/7ada48f0e0b125f4e41d9634cab9d857760fd9ed6d411a42d5b2dfb4b7a0d4fc.jpg)

![a52132134361f9f464c949de95f4ae7aec8b7e9a2751831bf2a920e59696973f.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/a52132134361f9f464c949de95f4ae7aec8b7e9a2751831bf2a920e59696973f.jpg)

![a59f2d4a4392e8d1a329816aa2b8d63a070e71b26ee9aa5858b13bbeaab22717.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/a59f2d4a4392e8d1a329816aa2b8d63a070e71b26ee9aa5858b13bbeaab22717.jpg)

![b69edbcc15fc6554ecb54c9c5896d50aa0b4450b03710edeb280e290a87629ef.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/b69edbcc15fc6554ecb54c9c5896d50aa0b4450b03710edeb280e290a87629ef.jpg)

![bb536c27ffca13a10ccf199ddaf8c05f6319c7a157feee9cc835d77087e20ca9.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/bb536c27ffca13a10ccf199ddaf8c05f6319c7a157feee9cc835d77087e20ca9.jpg)

![be3ce3233be08cfc7b95c3092761894a7c90ea41f22e3d073020a1bf4eca2624.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/be3ce3233be08cfc7b95c3092761894a7c90ea41f22e3d073020a1bf4eca2624.jpg)

![c0601f46916456ae757d6b76a6cc6fa7b545025edb70eed679acabb95b17a4d5.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/c0601f46916456ae757d6b76a6cc6fa7b545025edb70eed679acabb95b17a4d5.jpg)

![d730f910736c8bf0575d7dd3eea725d122232289eefb732b33a947f752ed62e0.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/images/d730f910736c8bf0575d7dd3eea725d122232289eefb732b33a947f752ed62e0.jpg)

### Tables

![0268fd18493507ae1ed87a7ec1080d7589c501485c39b693943b04a60749e0a2.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/0268fd18493507ae1ed87a7ec1080d7589c501485c39b693943b04a60749e0a2.jpg)

![3395369d7cb58a5f07e7e7486e588419c3213dbd97ecf6964faa6b98e1be4973.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/3395369d7cb58a5f07e7e7486e588419c3213dbd97ecf6964faa6b98e1be4973.jpg)

![340a7940dddac01ab809f68a93b364e2af3fbe77bd83805e7b7c13511101e755.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/340a7940dddac01ab809f68a93b364e2af3fbe77bd83805e7b7c13511101e755.jpg)

![3f4633e3517166d7bab2dc077eab7ac8299c72f1cd1711bc6031e872192db551.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/3f4633e3517166d7bab2dc077eab7ac8299c72f1cd1711bc6031e872192db551.jpg)

![3f48f9a2e34d78f55e9fc075393e0b57a1892b38d56e76499b5caaa0385802a0.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/3f48f9a2e34d78f55e9fc075393e0b57a1892b38d56e76499b5caaa0385802a0.jpg)

![40314354e9df92233a9b8d6e3731a791beebcf45a07d9ea91380157c57d0b9d6.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/40314354e9df92233a9b8d6e3731a791beebcf45a07d9ea91380157c57d0b9d6.jpg)

![43297cdce6c45b3b285a27ffa18386d6212e336fc066a5436ab5cde05fa3d4a2.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/43297cdce6c45b3b285a27ffa18386d6212e336fc066a5436ab5cde05fa3d4a2.jpg)

![63fbea61f2ec66349eddbcd831776e6fad2c9b9a82f50e1835b842c8c6da094f.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/63fbea61f2ec66349eddbcd831776e6fad2c9b9a82f50e1835b842c8c6da094f.jpg)

![7c9b736d4e6aab0c5fa7a48d24010848eddab94b6bad87a41ab98517f4ea26e8.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/7c9b736d4e6aab0c5fa7a48d24010848eddab94b6bad87a41ab98517f4ea26e8.jpg)

![81dbd58118d3d7571ed8ca140adaf35813d7040751e318959e08cf489054e03a.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/81dbd58118d3d7571ed8ca140adaf35813d7040751e318959e08cf489054e03a.jpg)

![df5abc0823ee81228fcd636da092a129e17ea6fc46dab13412e2400f59aa251f.jpg](../icml_results/5413_CircuitPrint_%20Mechanistic%20Circuit%20Fingerprints%20for%20Large%20Language%20Models/tables/df5abc0823ee81228fcd636da092a129e17ea6fc46dab13412e2400f59aa251f.jpg)

<a id="paper-5366"></a>

## SPUR: Scale-Partitioned Uncertainty Rectification for Robust UAV-on-UAV Interception


### Images

![3ec3749b0905a3b2f0ea8554bc54a7523ec0de824a8326d7a241683b68c59c14.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/images/3ec3749b0905a3b2f0ea8554bc54a7523ec0de824a8326d7a241683b68c59c14.jpg)

![43b42987b7901aee438dd50fcdd00d3a5a1f673af6ac17d28816e737217c01ef.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/images/43b42987b7901aee438dd50fcdd00d3a5a1f673af6ac17d28816e737217c01ef.jpg)

![83dec5345eb480548759a57f49b10a32c8ddd51eb1c7f2688f4de3b7af1eadb9.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/images/83dec5345eb480548759a57f49b10a32c8ddd51eb1c7f2688f4de3b7af1eadb9.jpg)

### Tables

![5b53cdfd11f4533974db6bb0e125ea5a43d69b9800021f8e401368ac61717033.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/5b53cdfd11f4533974db6bb0e125ea5a43d69b9800021f8e401368ac61717033.jpg)

![6ad0d3822ac01ced64ca6319bc560ca3f967acd70896ee78437f52fddeae6e47.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/6ad0d3822ac01ced64ca6319bc560ca3f967acd70896ee78437f52fddeae6e47.jpg)

![6bf12f0a0346539d3d566b65272e0e8ae332b35fb2bd270002bbcc58ca6ac417.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/6bf12f0a0346539d3d566b65272e0e8ae332b35fb2bd270002bbcc58ca6ac417.jpg)

![99dc200801a7cf9f995675c654f41b5d16cb85049988f538ece119d7a91415d4.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/99dc200801a7cf9f995675c654f41b5d16cb85049988f538ece119d7a91415d4.jpg)

![9b18628ae51dbe9900182a5b1e8d256e3c946e3a99c1073c966026af4605926d.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/9b18628ae51dbe9900182a5b1e8d256e3c946e3a99c1073c966026af4605926d.jpg)

![9b722990d96ef87ae53ac282790e1c9f13cdfa42bc465232eb82bde621950725.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/9b722990d96ef87ae53ac282790e1c9f13cdfa42bc465232eb82bde621950725.jpg)

![a63bef6c29d809b87464ee8e8be9a47b31e3fe485f487bd916852b2ecb8c5eff.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/a63bef6c29d809b87464ee8e8be9a47b31e3fe485f487bd916852b2ecb8c5eff.jpg)

![b4ad6c1a736b39022c4f1f71378b241df051b3adb1b5973570c9c2c2939218af.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/b4ad6c1a736b39022c4f1f71378b241df051b3adb1b5973570c9c2c2939218af.jpg)

![bd215ce4de444ffc1b8e51eb2d3deb3287d56536e8fd3c112e34da9b1ce21591.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/bd215ce4de444ffc1b8e51eb2d3deb3287d56536e8fd3c112e34da9b1ce21591.jpg)

![d2d62129497c8a2ffc051e12768dbf961c6598e3a3240e3d8cacfe93d7914b91.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/d2d62129497c8a2ffc051e12768dbf961c6598e3a3240e3d8cacfe93d7914b91.jpg)

![df62a4ff9ac15d4b20111716b0629c7914542b53388b3b08a18beb7a3f859696.jpg](../icml_results/5414_SPUR_%20Scale-Partitioned%20Uncertainty%20Rectification%20for%20Robust%20UAV-on-UAV%20Interception/tables/df62a4ff9ac15d4b20111716b0629c7914542b53388b3b08a18beb7a3f859696.jpg)

<a id="paper-5367"></a>

## Rethinking Serialization in Linear 3D Vision: Decoupling Anisotropic Geometry from Isotropic Semantics


### Images

![597c8b143b9f39ce551490499b394f5276edada5afc4a62494f9a70db2b878cc.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/images/597c8b143b9f39ce551490499b394f5276edada5afc4a62494f9a70db2b878cc.jpg)

![65232e7278df246bdd3d732904a699176fdac999ac5a352c5044f7acc84dbdca.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/images/65232e7278df246bdd3d732904a699176fdac999ac5a352c5044f7acc84dbdca.jpg)

![6d23608b3dad92b55aff789c4cfa975b2ca4a6500a915df19143fdb80ef5c7ec.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/images/6d23608b3dad92b55aff789c4cfa975b2ca4a6500a915df19143fdb80ef5c7ec.jpg)

![7f885fd1f720265ee21eefe688039e5c6afecda806562f6367d364c9fc74f6b8.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/images/7f885fd1f720265ee21eefe688039e5c6afecda806562f6367d364c9fc74f6b8.jpg)

### Tables

![11f950ad9cff8aff8d5202b9900f982ab8a25775ac47135bdf37d1fcec91ec69.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/11f950ad9cff8aff8d5202b9900f982ab8a25775ac47135bdf37d1fcec91ec69.jpg)

![196450ab2ada12758fb52ca39dd63b966bec374b9a5a968c9dc4853bda0d4c4c.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/196450ab2ada12758fb52ca39dd63b966bec374b9a5a968c9dc4853bda0d4c4c.jpg)

![207d9f5f374a7da5a5ee567d8a2d7cf7ae4408f09de8b8c27df50cdf8e276d48.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/207d9f5f374a7da5a5ee567d8a2d7cf7ae4408f09de8b8c27df50cdf8e276d48.jpg)

![2c6ab44a79f7564698387866ee4bb1fade140b9f386b480ad5e2a09a6fe7135e.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/2c6ab44a79f7564698387866ee4bb1fade140b9f386b480ad5e2a09a6fe7135e.jpg)

![41ad9f91a188445e622463c8a0ea0deefe59924942adb95aa4333ebaf6b11287.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/41ad9f91a188445e622463c8a0ea0deefe59924942adb95aa4333ebaf6b11287.jpg)

![48f2de3e84b68ca9b3d33cc526db016a3d6055240b5e683f324e878b1abec53e.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/48f2de3e84b68ca9b3d33cc526db016a3d6055240b5e683f324e878b1abec53e.jpg)

![4a05bd1bbcecb6475a980b26531e5b974e9472d7122a1e749295279ab927a7f8.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/4a05bd1bbcecb6475a980b26531e5b974e9472d7122a1e749295279ab927a7f8.jpg)

![538f609a4158791f8622b87312d7fab30f15bb9a14c2ca87d0f48918a6bb0518.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/538f609a4158791f8622b87312d7fab30f15bb9a14c2ca87d0f48918a6bb0518.jpg)

![67444ffeb27b525ae92a5db2960323889daf6a2e25db326f67a452e82a5dc8db.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/67444ffeb27b525ae92a5db2960323889daf6a2e25db326f67a452e82a5dc8db.jpg)

![8ce2e850f971f48bf0bf03446e43303e2e655fbf9851438ce5e7c80a9f888e94.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/8ce2e850f971f48bf0bf03446e43303e2e655fbf9851438ce5e7c80a9f888e94.jpg)

![9fcebafaa863bab712e744472b3d9fdad520da9cd5b8f76a1bf433fc03f85e8f.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/9fcebafaa863bab712e744472b3d9fdad520da9cd5b8f76a1bf433fc03f85e8f.jpg)

![a0471137cdf93afc829bf6a0cdeb525e5778523f0f11c0fb029361a35cdd6eae.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/a0471137cdf93afc829bf6a0cdeb525e5778523f0f11c0fb029361a35cdd6eae.jpg)

![d6c3ffa8676d3e390f53ebc739b10a040abab40e250d5439f7433281826f2c8d.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/d6c3ffa8676d3e390f53ebc739b10a040abab40e250d5439f7433281826f2c8d.jpg)

![f1e34e82b11305d6b45aa6bf5f3ac2a82ad2ca9343d8337e4456a5ce3a97f44a.jpg](../icml_results/5415_Rethinking%20Serialization%20in%20Linear%203D%20Vision_%20Decoupling%20Anisotropic%20Geometry%20from%20Isotropic%20Semanti/tables/f1e34e82b11305d6b45aa6bf5f3ac2a82ad2ca9343d8337e4456a5ce3a97f44a.jpg)

<a id="paper-5368"></a>

## GRASP: Awakening Latent Spatial Reasoning in LVLMs via Training-free Geometric Rectification


### Images

![35da84f6575a33d37615f75df8a06aff4e19e2d696badd4102335d1a61e9b302.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/35da84f6575a33d37615f75df8a06aff4e19e2d696badd4102335d1a61e9b302.jpg)

![5acc81b04310b65e35360bd76872e0cf1654a81d15d48d037aa18a5c5125a8e6.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/5acc81b04310b65e35360bd76872e0cf1654a81d15d48d037aa18a5c5125a8e6.jpg)

![5e7f2c0c7b772abcb3e55335475838dc225cc77422462a1e97f37070b1409319.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/5e7f2c0c7b772abcb3e55335475838dc225cc77422462a1e97f37070b1409319.jpg)

![82f19acd9b1c1484471f6121d77c86f1d8b59e140f631e10aa2d13bc45673caa.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/82f19acd9b1c1484471f6121d77c86f1d8b59e140f631e10aa2d13bc45673caa.jpg)

![8506a0466d397d0130e8b4feb88620a3f2ff48e1d0c77d66fe5f9c24acdce837.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/8506a0466d397d0130e8b4feb88620a3f2ff48e1d0c77d66fe5f9c24acdce837.jpg)

![8b1dab1e628cc8728e7e1b7ced2b9ecf02760c85ee032a7d16f462c0da37404d.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/8b1dab1e628cc8728e7e1b7ced2b9ecf02760c85ee032a7d16f462c0da37404d.jpg)

![a07710cbc6cc80b9ffb185f8f059f82e0f98fbf18711efe779f19034e379c1d8.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/a07710cbc6cc80b9ffb185f8f059f82e0f98fbf18711efe779f19034e379c1d8.jpg)

![aecd31576e09ac768dd2016a11ed0de235886390fcea09059be2130da64445cd.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/aecd31576e09ac768dd2016a11ed0de235886390fcea09059be2130da64445cd.jpg)

![bd273a2ee6773482376ce69f83c0f77c72b44534f1b40f91dd91d74d51ddf848.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/bd273a2ee6773482376ce69f83c0f77c72b44534f1b40f91dd91d74d51ddf848.jpg)

![df2f53904e1fbbd80d3385b9a8a2d76e6f875e94af807d1d2b397728e6f08868.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/df2f53904e1fbbd80d3385b9a8a2d76e6f875e94af807d1d2b397728e6f08868.jpg)

![e7dbfe08ff593c5fa260bb078eca1853e86536f7c03c5fb33cf8d427fc419ef1.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/e7dbfe08ff593c5fa260bb078eca1853e86536f7c03c5fb33cf8d427fc419ef1.jpg)

![ffdae567c43284a8e0c13996de0cc029ed06e1dff0c6369f8391ed0925495b5b.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/images/ffdae567c43284a8e0c13996de0cc029ed06e1dff0c6369f8391ed0925495b5b.jpg)

### Tables

![00a08e11ee8d885bdbb8822e7fc240b80e104faeceb38ed15d5a4b5442e87786.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/00a08e11ee8d885bdbb8822e7fc240b80e104faeceb38ed15d5a4b5442e87786.jpg)

![0c5676c17930fc0902ad5ca994dfe2b90a83c444220303302406948f8baa3b26.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/0c5676c17930fc0902ad5ca994dfe2b90a83c444220303302406948f8baa3b26.jpg)

![135784ce4c666df0b3b817c4214da70d2ac4032aa4f7df88177c66c76dd9e2aa.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/135784ce4c666df0b3b817c4214da70d2ac4032aa4f7df88177c66c76dd9e2aa.jpg)

![14c48ecfdc40ee025cec3a00f19f21e4176b06f2af2961ca211ad0d52f800a8d.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/14c48ecfdc40ee025cec3a00f19f21e4176b06f2af2961ca211ad0d52f800a8d.jpg)

![4f3690d86411a49117d14fa645b9375a55c3f42ca91ebaa8fc2434c84a08bd46.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/4f3690d86411a49117d14fa645b9375a55c3f42ca91ebaa8fc2434c84a08bd46.jpg)

![664decc0a26eb6d00725c4ff8ce526bb4bff571ab57cc7102d9c0bef762ed061.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/664decc0a26eb6d00725c4ff8ce526bb4bff571ab57cc7102d9c0bef762ed061.jpg)

![6bb435edb5108cf29ef6f9f5209dee26ffe76de5664216f69652c9b0ab37f86c.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/6bb435edb5108cf29ef6f9f5209dee26ffe76de5664216f69652c9b0ab37f86c.jpg)

![76395896b9bc6238ab88556cfa5f7bc639ed2d93b5b9e080004be24a3b622ae5.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/76395896b9bc6238ab88556cfa5f7bc639ed2d93b5b9e080004be24a3b622ae5.jpg)

![77ce22a075b0d715b7a7b26bcf1b938347bee703c190da1c6d5ee6d0d86ad1a1.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/77ce22a075b0d715b7a7b26bcf1b938347bee703c190da1c6d5ee6d0d86ad1a1.jpg)

![77f2c8cc899c94fc5e6ef4c3c22cd52c6fa2d515f8ca7ef1a4d587beecef1931.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/77f2c8cc899c94fc5e6ef4c3c22cd52c6fa2d515f8ca7ef1a4d587beecef1931.jpg)

![85a7c431cec57ef3e1788311c94ec1a5b2a7426475b8b21951b58b1d7450e4e3.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/85a7c431cec57ef3e1788311c94ec1a5b2a7426475b8b21951b58b1d7450e4e3.jpg)

![96b68d8c41d68869bedc1de5c85d358093e107ca98d22658489abe558e27f68d.jpg](../icml_results/5416_GRASP_%20Awakening%20Latent%20Spatial%20Reasoning%20in%20LVLMs%20via%20Training-free%20Geometric%20Rectification/tables/96b68d8c41d68869bedc1de5c85d358093e107ca98d22658489abe558e27f68d.jpg)

<a id="paper-5369"></a>

## Robust Federated Learning Against Adaptive Compression


### Images

![0a4651208948403b4e7f3dd926867fa41a44da2f07785fd6690851dc78dfee29.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/0a4651208948403b4e7f3dd926867fa41a44da2f07785fd6690851dc78dfee29.jpg)

![0eb6de65cd27d18a8e479f7b706b25d866108fb0f18a4d5980d12719bd570201.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/0eb6de65cd27d18a8e479f7b706b25d866108fb0f18a4d5980d12719bd570201.jpg)

![2efd1e18ac180f1860cd085c7eb87898e6790d54e3aeec79d603b49a6a650f95.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/2efd1e18ac180f1860cd085c7eb87898e6790d54e3aeec79d603b49a6a650f95.jpg)

![516747f19d6d35b6feb743489aa0c9d1340da2a36049336706e5b70a3cbfebe2.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/516747f19d6d35b6feb743489aa0c9d1340da2a36049336706e5b70a3cbfebe2.jpg)

![56d36715eba60c1ead1c128ee5e744033370fab8faa74a0a98d92bfd81780bb3.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/56d36715eba60c1ead1c128ee5e744033370fab8faa74a0a98d92bfd81780bb3.jpg)

![88c74a5889f923f8e64dd1fbcce57a278c1ac19c30cf8ec3c1cc1b0f25613a9d.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/88c74a5889f923f8e64dd1fbcce57a278c1ac19c30cf8ec3c1cc1b0f25613a9d.jpg)

![924f891e31b6cfd40e7553e0340913afd9b593ea50c1a17284e9dcaa2b50f566.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/924f891e31b6cfd40e7553e0340913afd9b593ea50c1a17284e9dcaa2b50f566.jpg)

![a72a8f16ab77bfc98eacbb73b50497208fae941d6b09a50f470dfbbfff15a534.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/a72a8f16ab77bfc98eacbb73b50497208fae941d6b09a50f470dfbbfff15a534.jpg)

![c3d8365b9620c6035c3e0240ca0ec4824e4ac68a2d604fbff3980578d41a92ea.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/c3d8365b9620c6035c3e0240ca0ec4824e4ac68a2d604fbff3980578d41a92ea.jpg)

![d586e6edc8a8a1c0b7d02f8a8dd5d350893dc0087d4d0dcb6bc92318a26347ec.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/d586e6edc8a8a1c0b7d02f8a8dd5d350893dc0087d4d0dcb6bc92318a26347ec.jpg)

![fc0f8412227c6d2844571e604f1a53b27262ed6908ae7aecc71f00815fbe10bc.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/images/fc0f8412227c6d2844571e604f1a53b27262ed6908ae7aecc71f00815fbe10bc.jpg)

### Tables

![46a553f6e5ea1f725dd8a1253f591587b0a552424f7a626c772206b1ee9dfe5c.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/tables/46a553f6e5ea1f725dd8a1253f591587b0a552424f7a626c772206b1ee9dfe5c.jpg)

![90896fad6ad9427b91d124362cb1e62d35278afe8dc70f62a767ea66b3452929.jpg](../icml_results/5417_Robust%20Federated%20Learning%20Against%20Adaptive%20Compression/tables/90896fad6ad9427b91d124362cb1e62d35278afe8dc70f62a767ea66b3452929.jpg)

<a id="paper-5370"></a>

## CE$^4$L: Continual Ego, Exo, and Ego-Exo Learning


### Images

![098f62c3dcb807605676fccdbe0efe0f677b5cb947c5469b435b202fc5cdce5a.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/098f62c3dcb807605676fccdbe0efe0f677b5cb947c5469b435b202fc5cdce5a.jpg)

![622f53dc3142f6304aa374cacdb9e5c1fbfbc814cadfb3ea05c3e05bca4ecdee.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/622f53dc3142f6304aa374cacdb9e5c1fbfbc814cadfb3ea05c3e05bca4ecdee.jpg)

![65019c7c0f5d39526808a9e717f35c55531509e3300cc434d30679dad2873ebd.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/65019c7c0f5d39526808a9e717f35c55531509e3300cc434d30679dad2873ebd.jpg)

![6bd4a927c7a5f96f956c2b30bab6495a003a35b30c844cbf99e7b35e0dc9013c.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/6bd4a927c7a5f96f956c2b30bab6495a003a35b30c844cbf99e7b35e0dc9013c.jpg)

![7ea7c7ec672e125fa27be3cf738929137785e50e5794db7193b161ea6ad94e31.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/7ea7c7ec672e125fa27be3cf738929137785e50e5794db7193b161ea6ad94e31.jpg)

![84dfebba7ff8d250ddbaeb4dff94f015343755c4e5350cda585e8c8bf0f1ff66.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/84dfebba7ff8d250ddbaeb4dff94f015343755c4e5350cda585e8c8bf0f1ff66.jpg)

![ae2d353cae9751e9b030abdda478890e3418ddce7c2c7b047089a93663101742.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/ae2d353cae9751e9b030abdda478890e3418ddce7c2c7b047089a93663101742.jpg)

![bca912031bba6e35f7a8a7e05c5ab7d4d4530f5495a634655e809082f0fbd85c.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/bca912031bba6e35f7a8a7e05c5ab7d4d4530f5495a634655e809082f0fbd85c.jpg)

![e70bbb2fe3dd8a5b184496e7c82111349018c9a7ddf1d2d860ca726ce9a84531.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/e70bbb2fe3dd8a5b184496e7c82111349018c9a7ddf1d2d860ca726ce9a84531.jpg)

![e76701abec9b70f73f7fa3b4bad9f703725fea360c3df2e2506baddb631e3b92.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/images/e76701abec9b70f73f7fa3b4bad9f703725fea360c3df2e2506baddb631e3b92.jpg)

### Tables

![012804454aaa3b3141f309ee2292b135e4ec2c02abe7168fa831d6eed9d6b1ca.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/012804454aaa3b3141f309ee2292b135e4ec2c02abe7168fa831d6eed9d6b1ca.jpg)

![036515acfb9123c0dc5ee5ec17ff4a9cca418fe5a2cf6e50f9e6c0403ea27da9.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/036515acfb9123c0dc5ee5ec17ff4a9cca418fe5a2cf6e50f9e6c0403ea27da9.jpg)

![065003b08593f7d11cb9c22475522b989d4fed052ae7be821a6233ca42528b1c.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/065003b08593f7d11cb9c22475522b989d4fed052ae7be821a6233ca42528b1c.jpg)

![073d7e35276cdbef3c9f678def8c055e27c9a6445f92ba82c896ea47a3a1b57b.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/073d7e35276cdbef3c9f678def8c055e27c9a6445f92ba82c896ea47a3a1b57b.jpg)

![1075bf05b8f91c2fb92ec8bbe98c165968ff831abbb5d32733527fe7615e1d22.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/1075bf05b8f91c2fb92ec8bbe98c165968ff831abbb5d32733527fe7615e1d22.jpg)

![13a2f4e5ab6b569350f2be8749f16ff0828d0b02ff860ec97817c67cba216413.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/13a2f4e5ab6b569350f2be8749f16ff0828d0b02ff860ec97817c67cba216413.jpg)

![238dab38194e050b05c1b305866770fb0b66b8b34885f51828c966c9ae5e2c56.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/238dab38194e050b05c1b305866770fb0b66b8b34885f51828c966c9ae5e2c56.jpg)

![28628d8048f650c319edfea598d5bbe00b8fc3ee317b5f3c881f4bb587df800d.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/28628d8048f650c319edfea598d5bbe00b8fc3ee317b5f3c881f4bb587df800d.jpg)

![2a919431942e28dc8a7b2554ea49c2ba9207b13bb11a8be474c7c1a51444e4af.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/2a919431942e28dc8a7b2554ea49c2ba9207b13bb11a8be474c7c1a51444e4af.jpg)

![3121e8e9515470776bb899576af805261ad0b3eca0fb9a4414be286a1ee15c43.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/3121e8e9515470776bb899576af805261ad0b3eca0fb9a4414be286a1ee15c43.jpg)

![3470733d8109072cf17cc94fe14a41bbee68468ecb052064f937bc4997f233c4.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/3470733d8109072cf17cc94fe14a41bbee68468ecb052064f937bc4997f233c4.jpg)

![4949453209ba519c91a48e42e8ff69ed1ff2ab9f430afaf839310bc7f6a87cad.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/4949453209ba519c91a48e42e8ff69ed1ff2ab9f430afaf839310bc7f6a87cad.jpg)

![545724ca465cd87761529b42e6b40e8a59dee99456ed4cdebf427662cbb29831.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/545724ca465cd87761529b42e6b40e8a59dee99456ed4cdebf427662cbb29831.jpg)

![6aba3a6bffb3ff7799b7d49169a53d6115b04186ea5455b24b910b9c4a7ed446.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/6aba3a6bffb3ff7799b7d49169a53d6115b04186ea5455b24b910b9c4a7ed446.jpg)

![6da2c7213097f7ddc6bad13cf33d1c320071fa5b352e38ad414c180d7d858b9f.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/6da2c7213097f7ddc6bad13cf33d1c320071fa5b352e38ad414c180d7d858b9f.jpg)

![6e8313aa926eb921c248744c5db03ec640416ba9ccdbb34c62572cf7ad0a7d14.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/6e8313aa926eb921c248744c5db03ec640416ba9ccdbb34c62572cf7ad0a7d14.jpg)

![6f782bf22ef52a9b5ceaaebaf23ad76c07d0cc94b3f4fd67d849ca6721031c3f.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/6f782bf22ef52a9b5ceaaebaf23ad76c07d0cc94b3f4fd67d849ca6721031c3f.jpg)

![8054348a65e6d442eb51e24ed9b9f18224e59ac7be653a2c13671697ed0fae49.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/8054348a65e6d442eb51e24ed9b9f18224e59ac7be653a2c13671697ed0fae49.jpg)

![85e3f789ef4892f3f425731626d1ff0e289260e790ac68da8ce3e5fed0f85acd.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/85e3f789ef4892f3f425731626d1ff0e289260e790ac68da8ce3e5fed0f85acd.jpg)

![986d70fbb9d651cae6286ec12a9430632f5f48ea77965149ef4482d09fd70fdd.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/986d70fbb9d651cae6286ec12a9430632f5f48ea77965149ef4482d09fd70fdd.jpg)

![a04b30fcc4d7903280ddc7b25b85e5b292a4b57fda3df520e74765231bc5f32a.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/a04b30fcc4d7903280ddc7b25b85e5b292a4b57fda3df520e74765231bc5f32a.jpg)

![b0e6ec22b738b4870435a43b96f4add983b426c89e6f469c766e5e3b53118569.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/b0e6ec22b738b4870435a43b96f4add983b426c89e6f469c766e5e3b53118569.jpg)

![b55aea5c574af16a3b5e18d1aeda853f6ed50c5ce30636b6e4b1f209d600210c.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/b55aea5c574af16a3b5e18d1aeda853f6ed50c5ce30636b6e4b1f209d600210c.jpg)

![b6161f4888d23643892b8e898612563b29fc496e793ada806044b7cb9d0b12b6.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/b6161f4888d23643892b8e898612563b29fc496e793ada806044b7cb9d0b12b6.jpg)

![ba92dc575deb3657119e5a6f680c0b7598f2974c8bc3d1e647b2f733e78dbe0f.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/ba92dc575deb3657119e5a6f680c0b7598f2974c8bc3d1e647b2f733e78dbe0f.jpg)

![c854f150df2aae6958e60ee681bf76876e4e61d38d0d7723cd9da259a1c0c31a.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/c854f150df2aae6958e60ee681bf76876e4e61d38d0d7723cd9da259a1c0c31a.jpg)

![dd065df8872c21a899876d1637135d3140b8751e9841c191d56db340840f2985.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/dd065df8872c21a899876d1637135d3140b8751e9841c191d56db340840f2985.jpg)

![f0583a53bfa38923a1970f08d20caf513697b8ad3a72920e71eb0ec45fa75aaf.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/f0583a53bfa38923a1970f08d20caf513697b8ad3a72920e71eb0ec45fa75aaf.jpg)

![f4710aaa07546e23a62ae875a0b3859c9a0e47a139787c2dc48e0fe6459649ca.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/f4710aaa07546e23a62ae875a0b3859c9a0e47a139787c2dc48e0fe6459649ca.jpg)

![f728b75b5c7f15ff0495eb9f48e716864a923fc8278e6f5a157918fbf7d8ce9e.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/f728b75b5c7f15ff0495eb9f48e716864a923fc8278e6f5a157918fbf7d8ce9e.jpg)

![fb8b801507f6a58a4b01d12baf8ced2806ef40c8b6ee2a83ca432974c3641afc.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/fb8b801507f6a58a4b01d12baf8ced2806ef40c8b6ee2a83ca432974c3641afc.jpg)

![fc409c7efc17ba5229a710d346c5bcc182d1f7de8a99660da57b3a8e2e779b17.jpg](../icml_results/5418_CE%24%5E4%24L_%20Continual%20Ego%2C%20Exo%2C%20and%20Ego-Exo%20Learning/tables/fc409c7efc17ba5229a710d346c5bcc182d1f7de8a99660da57b3a8e2e779b17.jpg)

<a id="paper-5371"></a>

## A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data Analysis


### Images

![529388ff3cfa23f396a9e780dc37ba6bfa943d8fe470643109f034d3fabd67f8.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/529388ff3cfa23f396a9e780dc37ba6bfa943d8fe470643109f034d3fabd67f8.jpg)

![74293e1b6b7141fb86787be3a2519f46a1c7e43566e43aefe2a584d986015db7.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/74293e1b6b7141fb86787be3a2519f46a1c7e43566e43aefe2a584d986015db7.jpg)

![7fff3c3f3e48bd09f07a80fcdf5bfcdb6fe953bf4ebb4e346cb7ba38c426b09e.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/7fff3c3f3e48bd09f07a80fcdf5bfcdb6fe953bf4ebb4e346cb7ba38c426b09e.jpg)

![95cbb2163ef9f440c30f4847cd4cfcb0711cbf1fe914e34a8ada8195ec86c233.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/95cbb2163ef9f440c30f4847cd4cfcb0711cbf1fe914e34a8ada8195ec86c233.jpg)

![a88dc55b999013bfcd0e2a9d746af2d55f5ec4a6d9d6e0c25438c15f5cb919b1.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/a88dc55b999013bfcd0e2a9d746af2d55f5ec4a6d9d6e0c25438c15f5cb919b1.jpg)

![a8d74d0c3cd4bd4f763248196c09705b0e1b42f818616dcad26c34dcb7b6dfbb.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/images/a8d74d0c3cd4bd4f763248196c09705b0e1b42f818616dcad26c34dcb7b6dfbb.jpg)

### Tables

![8854beb5d18cdf3e5a40aea0a4c6ed6cb34c2c8a85387ff9e1d65050e1f08303.jpg](../icml_results/5419_A%20Strictly%20Proper%20Scoring%20Rule%20and%20a%20Calibration%20Metric%20for%20Interval-Censored%20Data%20Analysis/tables/8854beb5d18cdf3e5a40aea0a4c6ed6cb34c2c8a85387ff9e1d65050e1f08303.jpg)

<a id="paper-5372"></a>

## SpecExit: Accelerating Large Reasoning Model via Speculative Exit


### Images

![086694a8ee66a30ffc79b4956cbf1b6a9627459e1a7c82926a618f071a02b8ea.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/086694a8ee66a30ffc79b4956cbf1b6a9627459e1a7c82926a618f071a02b8ea.jpg)

![2285e1a8ad12500218ac0ded77cfdb94f6296ef9118aaded70a798ea7d0eee1d.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/2285e1a8ad12500218ac0ded77cfdb94f6296ef9118aaded70a798ea7d0eee1d.jpg)

![29c19b0da1ceb1c690db40979614d48ae006e2a2a811a5fa7afbcb4655575593.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/29c19b0da1ceb1c690db40979614d48ae006e2a2a811a5fa7afbcb4655575593.jpg)

![38d916c253035221783f7fd7dc942035f49a7df37ef7b58d305044dfdfdc00e4.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/38d916c253035221783f7fd7dc942035f49a7df37ef7b58d305044dfdfdc00e4.jpg)

![439b72f802d134b0e900e56f8f170f54900e1783bcd5f95209bceb38e3c6b844.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/439b72f802d134b0e900e56f8f170f54900e1783bcd5f95209bceb38e3c6b844.jpg)

![630f9927de7800a7eb97fc5b963a04ccfd932a641caee4d6cafd67c9f638e4b1.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/630f9927de7800a7eb97fc5b963a04ccfd932a641caee4d6cafd67c9f638e4b1.jpg)

![6cc26ed476b54efe75d6f26712d6b6674135b40deabdbf15fdd6ea05d7aa2fdd.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/6cc26ed476b54efe75d6f26712d6b6674135b40deabdbf15fdd6ea05d7aa2fdd.jpg)

![722b8a092a4936ad84e6747ee9d3f84c31b4327fe8fee526411cfd471db18053.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/722b8a092a4936ad84e6747ee9d3f84c31b4327fe8fee526411cfd471db18053.jpg)

![74f3bc7fe03fea365059677f20dd99e9562871b453e38c6a8fd08183ec14a154.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/74f3bc7fe03fea365059677f20dd99e9562871b453e38c6a8fd08183ec14a154.jpg)

![78eecc8bfa8fc57b5de04580a96f3e11b0e2d8a003a9e1332e83ed5dfc6d1d94.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/78eecc8bfa8fc57b5de04580a96f3e11b0e2d8a003a9e1332e83ed5dfc6d1d94.jpg)

![79fbf5ccaefe5f4b9625deec4d7ad0a758c01d1a7f0c9cfbd72ba96f319db2e7.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/79fbf5ccaefe5f4b9625deec4d7ad0a758c01d1a7f0c9cfbd72ba96f319db2e7.jpg)

![842d867913840145813c4696e8b0be54dd68833d34c74abe8555b3391c26d6de.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/842d867913840145813c4696e8b0be54dd68833d34c74abe8555b3391c26d6de.jpg)

![906e2c52c234de025b9ca7b10d3f4fd1d77fb1c284d5531cbd206ab4bd64c1b6.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/906e2c52c234de025b9ca7b10d3f4fd1d77fb1c284d5531cbd206ab4bd64c1b6.jpg)

![944b73704d6aa10d3ebd07d33b7a8284f6b2b6ba5acd9d6acd9859a7c0eef0a5.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/944b73704d6aa10d3ebd07d33b7a8284f6b2b6ba5acd9d6acd9859a7c0eef0a5.jpg)

![ba651992b441c1967dc3796a0774310866708ca43c79e08ab7df988e75f1aa4d.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/ba651992b441c1967dc3796a0774310866708ca43c79e08ab7df988e75f1aa4d.jpg)

![bdfd6b277c1729d2021e64728b52e1920e41a8094d2e6d7e947aec609252f318.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/bdfd6b277c1729d2021e64728b52e1920e41a8094d2e6d7e947aec609252f318.jpg)

![f315c5ffc794a235332b875ed7a42d03c8cc32478cc15b0fb98fe6932bb962d4.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/f315c5ffc794a235332b875ed7a42d03c8cc32478cc15b0fb98fe6932bb962d4.jpg)

![f9732bb52604da9f7ce49c0405334c6745304446335e5a21398bce61d54593e9.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/images/f9732bb52604da9f7ce49c0405334c6745304446335e5a21398bce61d54593e9.jpg)

### Tables

![6a31781231861cef85245a6d25a8093933a6d55f558a9c2baf11e76fb3c593cc.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/tables/6a31781231861cef85245a6d25a8093933a6d55f558a9c2baf11e76fb3c593cc.jpg)

![79d0b557ed7bdaad055aa65d1c85b42cf6461bffe56c29a52ad8c8d6ec3624f9.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/tables/79d0b557ed7bdaad055aa65d1c85b42cf6461bffe56c29a52ad8c8d6ec3624f9.jpg)

![f087c034f4f53c5ae90af9c99adcabdac84230637436e2ee6f9199d30214fb34.jpg](../icml_results/5420_SpecExit_%20Accelerating%20Large%20Reasoning%20Model%20via%20Speculative%20Exit/tables/f087c034f4f53c5ae90af9c99adcabdac84230637436e2ee6f9199d30214fb34.jpg)

<a id="paper-5373"></a>

## Preconditioning Neural Tangent Kernel for Adaptive Optimization


### Images

![3048c4752ca92f2abbf3828eba6ad4abdbae402cb73644bdcdd693c577ce9d89.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/images/3048c4752ca92f2abbf3828eba6ad4abdbae402cb73644bdcdd693c577ce9d89.jpg)

![31bdf773d26050335316224d9fb8c5cefa76d5a006f8fd6c3c5ae17b6efff1ef.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/images/31bdf773d26050335316224d9fb8c5cefa76d5a006f8fd6c3c5ae17b6efff1ef.jpg)

![d170172b6e355df9f2dd3dc3313a8634b0e61acfdb35fa206180d017ed154fbb.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/images/d170172b6e355df9f2dd3dc3313a8634b0e61acfdb35fa206180d017ed154fbb.jpg)

### Tables

![0af3038c54a3e5d6ebe0335634f4689063bc710b0563a78307cd75cea9e22a3b.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/0af3038c54a3e5d6ebe0335634f4689063bc710b0563a78307cd75cea9e22a3b.jpg)

![23103f76291251c543e6744829a26b857ee986f126039e907bc0d5c9d6c0cf75.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/23103f76291251c543e6744829a26b857ee986f126039e907bc0d5c9d6c0cf75.jpg)

![3321c9f5844cba401825bb38f58dea49829d712c866dfcd114e668cf66a216b4.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/3321c9f5844cba401825bb38f58dea49829d712c866dfcd114e668cf66a216b4.jpg)

![76459a6d9833478f8dd286fcec7ed8535b70cf32b54190966d5c47a23be0620e.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/76459a6d9833478f8dd286fcec7ed8535b70cf32b54190966d5c47a23be0620e.jpg)

![87182252059281e71251818b85ed9e4a2dbf1119136f91fe2c1cd8107b5452f7.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/87182252059281e71251818b85ed9e4a2dbf1119136f91fe2c1cd8107b5452f7.jpg)

![a571c4c7626bc13ff8903c4e7a7e5dd9f0a215325a46ce4a694b65531f73a9a7.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/a571c4c7626bc13ff8903c4e7a7e5dd9f0a215325a46ce4a694b65531f73a9a7.jpg)

![b40c7462a0dc533ffaea60962c1e4c9774dd11ca18a6f6ea0436d4342ef9de66.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/b40c7462a0dc533ffaea60962c1e4c9774dd11ca18a6f6ea0436d4342ef9de66.jpg)

![c0eca26818b417e1cab6795f34ecf87fddd1dca692f63ecc4e5d37977415d6ed.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/c0eca26818b417e1cab6795f34ecf87fddd1dca692f63ecc4e5d37977415d6ed.jpg)

![d5673bfb5977946d3e0693196ac7bd8ab3d5fc72d97134d5be1307f2f214a81b.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/d5673bfb5977946d3e0693196ac7bd8ab3d5fc72d97134d5be1307f2f214a81b.jpg)

![e956c475af81f20e1abc0765937497002dcb85682879b5948c0183b1172d2493.jpg](../icml_results/5421_Preconditioning%20Neural%20Tangent%20Kernel%20for%20Adaptive%20Optimization/tables/e956c475af81f20e1abc0765937497002dcb85682879b5948c0183b1172d2493.jpg)

<a id="paper-5374"></a>

## Length Generalization Bounds for Transformers


### Images

![7cf16d8cb32df73d0998ba92d74f0c6b0ad4581d2347279e248e1d64ee2083e8.jpg](../icml_results/5422_Length%20Generalization%20Bounds%20for%20Transformers/images/7cf16d8cb32df73d0998ba92d74f0c6b0ad4581d2347279e248e1d64ee2083e8.jpg)

<a id="paper-5375"></a>

## MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier


### Images

![0ad8761dc67f307cc5ca265b20e5674682a70db39323fd3b3e71136e437aa36a.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/images/0ad8761dc67f307cc5ca265b20e5674682a70db39323fd3b3e71136e437aa36a.jpg)

![11242598e60e5a92af6eadb3cdbf79ba2e85e93b40cde31518c0e281a70f804d.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/images/11242598e60e5a92af6eadb3cdbf79ba2e85e93b40cde31518c0e281a70f804d.jpg)

![85d531bc9236ff78d5443eea923afd46d0776bd0007ea86bba2f054380946cd5.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/images/85d531bc9236ff78d5443eea923afd46d0776bd0007ea86bba2f054380946cd5.jpg)

![a4391dda87a061373e0d3f41e19dee6e79648f902f0bbd04f4f8a1d127eddb06.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/images/a4391dda87a061373e0d3f41e19dee6e79648f902f0bbd04f4f8a1d127eddb06.jpg)

### Tables

![1d7c67fb936071db86f86555b1c4e91501acdf522f5382dd43db401ed9df1689.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/1d7c67fb936071db86f86555b1c4e91501acdf522f5382dd43db401ed9df1689.jpg)

![2013d29ac187225333af749082e5e1bbf4b43e0014fceedee28c1f21493776ca.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/2013d29ac187225333af749082e5e1bbf4b43e0014fceedee28c1f21493776ca.jpg)

![203c127478b6914cb022b3dcdce2e4a885b98ec34f3bf02dd100d299f79a4efe.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/203c127478b6914cb022b3dcdce2e4a885b98ec34f3bf02dd100d299f79a4efe.jpg)

![31a44140569dda53db44e7596ac18496f7b295910237148a95b2c1785c4dba64.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/31a44140569dda53db44e7596ac18496f7b295910237148a95b2c1785c4dba64.jpg)

![64863189d72cbf82dd22d7d7c9fb1516c6a0730239cf9533d53e5b6e2e2cb5ed.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/64863189d72cbf82dd22d7d7c9fb1516c6a0730239cf9533d53e5b6e2e2cb5ed.jpg)

![7d71911ed4370c93489225ca2876d5905fe4961ed2bfe906035c15928ebecac3.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/7d71911ed4370c93489225ca2876d5905fe4961ed2bfe906035c15928ebecac3.jpg)

![8e26b0d58f0de2153f4c62f43d7bc85f4e0dc8bd8bbc9a3a15dc11b1b5bd47db.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/8e26b0d58f0de2153f4c62f43d7bc85f4e0dc8bd8bbc9a3a15dc11b1b5bd47db.jpg)

![af55c02319f3675d0009ba0288c6f0a37420f4b81a6484c9957155e8f930d658.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/af55c02319f3675d0009ba0288c6f0a37420f4b81a6484c9957155e8f930d658.jpg)

![b06bc6bca1ed4b046a2ddc2a5749a026aa4af83fa494edfbc6a8ed9edc1545f2.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/b06bc6bca1ed4b046a2ddc2a5749a026aa4af83fa494edfbc6a8ed9edc1545f2.jpg)

![e4e4b9385399d7d54d8ba90cf52a221ec5ed8be0b6861d602650065c02ec4d6f.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/e4e4b9385399d7d54d8ba90cf52a221ec5ed8be0b6861d602650065c02ec4d6f.jpg)

![f8d2dfe8e4698aad386a37c17993e8efde871d6aae1d8e412d864e3f20c7b076.jpg](../icml_results/5423_MOOSE-Star_%20Unlocking%20Tractable%20Training%20for%20Scientific%20Discovery%20by%20Breaking%20the%20Complexity%20Barrier/tables/f8d2dfe8e4698aad386a37c17993e8efde871d6aae1d8e412d864e3f20c7b076.jpg)

<a id="paper-5376"></a>

## Towards Understanding Generalization of Federated Adversarial Learning: Perspective of Algorithmic Stability


### Images

![2b12ec09e24cfcb2593f0dd20d2ca5e84951f8f31e849ea4f1cc3c5457f958fa.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/2b12ec09e24cfcb2593f0dd20d2ca5e84951f8f31e849ea4f1cc3c5457f958fa.jpg)

![7631745a11a36f46d3700f50cfef0fd53e707802bdb8494c9cd0b2cb46688850.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/7631745a11a36f46d3700f50cfef0fd53e707802bdb8494c9cd0b2cb46688850.jpg)

![7ba321ea700907387dc83102e20f97cce9eec7a1d28f20b802d35e374e91cd77.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/7ba321ea700907387dc83102e20f97cce9eec7a1d28f20b802d35e374e91cd77.jpg)

![887ce6cee418e3af26f1d305afeae299b40b741878fc9d5dd51fbd40a3181f04.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/887ce6cee418e3af26f1d305afeae299b40b741878fc9d5dd51fbd40a3181f04.jpg)

![ab1b4430bccf5a3d0bd73ea9cbd008acf14871aba9d84addcf136168127cacb6.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/ab1b4430bccf5a3d0bd73ea9cbd008acf14871aba9d84addcf136168127cacb6.jpg)

![b0e3bd8dae5a6b6b84694c39a0242c95613a52d772783e1f727f45f783865952.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/b0e3bd8dae5a6b6b84694c39a0242c95613a52d772783e1f727f45f783865952.jpg)

![b5dd2742ca72b16a8b6d2c2c23855608e102804c6cd4ab53eda27c106b0fff17.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/images/b5dd2742ca72b16a8b6d2c2c23855608e102804c6cd4ab53eda27c106b0fff17.jpg)

### Tables

![1fad036707e146150e9751da54769bc59a2e5e10c86cacd8ab30339efcc4925b.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/1fad036707e146150e9751da54769bc59a2e5e10c86cacd8ab30339efcc4925b.jpg)

![35fd4499bf74ef6a51e940e870c9c85e0c46ba83dbc32a2eddc15cfbcec5e971.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/35fd4499bf74ef6a51e940e870c9c85e0c46ba83dbc32a2eddc15cfbcec5e971.jpg)

![47dffe0381a507709a08654c8dcd01fed57487607b326d60cdcd1281fcdbf908.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/47dffe0381a507709a08654c8dcd01fed57487607b326d60cdcd1281fcdbf908.jpg)

![601aff14f04f13660e486e652b22b2ba677167770b1f0e48243ccd459be81c18.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/601aff14f04f13660e486e652b22b2ba677167770b1f0e48243ccd459be81c18.jpg)

![742b587f0d804d073eccf53aae052eec88ddc00bc1051808bfa61b3700e3e9a1.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/742b587f0d804d073eccf53aae052eec88ddc00bc1051808bfa61b3700e3e9a1.jpg)

![f51758ade4c3c7d8ec3cd9dcf3bf565c8efd45e3eb326eee5bdaf6f92772de83.jpg](../icml_results/5424_Towards%20Understanding%20Generalization%20of%20Federated%20Adversarial%20Learning_%20Perspective%20of%20Algorithmic%20S/tables/f51758ade4c3c7d8ec3cd9dcf3bf565c8efd45e3eb326eee5bdaf6f92772de83.jpg)

<a id="paper-5377"></a>

## PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents


### Images

![18b120896b76e2f5ce9aa336fab2a6ca7c25ed2ca397dc68f9ef563c23a92281.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/images/18b120896b76e2f5ce9aa336fab2a6ca7c25ed2ca397dc68f9ef563c23a92281.jpg)

![229a232431fdc8573fe323eccf4042a71ed9e28b78972157eeea63dccf5b750a.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/images/229a232431fdc8573fe323eccf4042a71ed9e28b78972157eeea63dccf5b750a.jpg)

![50050898ff75817825ac90c3c37f89a28e135fe91b14a0c5dd89a37930fe6f66.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/images/50050898ff75817825ac90c3c37f89a28e135fe91b14a0c5dd89a37930fe6f66.jpg)

![94258680f0fec2741dbd28d2711106a1e19ca3e6a011261b77313ae61013c3b8.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/images/94258680f0fec2741dbd28d2711106a1e19ca3e6a011261b77313ae61013c3b8.jpg)

![acc72953201b91c8b0935a7fee1679214be8d135397cb0dcf20c797bc19123c5.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/images/acc72953201b91c8b0935a7fee1679214be8d135397cb0dcf20c797bc19123c5.jpg)

### Tables

![0bcb5b5e60b597d347fd3430af5b42144b3c4cb4e160459e5128bc93989f8a4f.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/0bcb5b5e60b597d347fd3430af5b42144b3c4cb4e160459e5128bc93989f8a4f.jpg)

![0f0660ef5bdfd8d1f1a2275366cc2ad202c2c14e672372f7aad4639d4199032a.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/0f0660ef5bdfd8d1f1a2275366cc2ad202c2c14e672372f7aad4639d4199032a.jpg)

![0fe089c67c05da218b98ade785633f3585119012a032e8be97f48aa5649d2055.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/0fe089c67c05da218b98ade785633f3585119012a032e8be97f48aa5649d2055.jpg)

![121f717f9e8f17f4c9e69f341e154d32a889127f35c88a77da164bef3bc7ceea.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/121f717f9e8f17f4c9e69f341e154d32a889127f35c88a77da164bef3bc7ceea.jpg)

![18fd8913994283becd86ac775ea1eb082a118f37c2e95f756c3e50dd9d54e480.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/18fd8913994283becd86ac775ea1eb082a118f37c2e95f756c3e50dd9d54e480.jpg)

![192313a798a1545a6138493edca87bdc65fb7b1e6de1368362d3299d9d068ce0.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/192313a798a1545a6138493edca87bdc65fb7b1e6de1368362d3299d9d068ce0.jpg)

![1dc0f7ca9c04f4f189b202bc19ff60b77acc78099e74b5868b60ba87f98d5c4e.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/1dc0f7ca9c04f4f189b202bc19ff60b77acc78099e74b5868b60ba87f98d5c4e.jpg)

![3e45491a66c5b75707652354bd31d2a643144b04c1d405dc1dc6ae0157f361b2.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/3e45491a66c5b75707652354bd31d2a643144b04c1d405dc1dc6ae0157f361b2.jpg)

![3f89478662d1b700be55f0ab598aa4bdf61db3bc7a5875524d74634a1717101a.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/3f89478662d1b700be55f0ab598aa4bdf61db3bc7a5875524d74634a1717101a.jpg)

![57f2dd5d9820985a723493f4ac638a19b71dffed53e5ce0eaedfd6e9a6d03f17.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/57f2dd5d9820985a723493f4ac638a19b71dffed53e5ce0eaedfd6e9a6d03f17.jpg)

![5ef015603b890bd311e8fae94e66866a93df0047f36f2534ebafae6fa35a05cf.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/5ef015603b890bd311e8fae94e66866a93df0047f36f2534ebafae6fa35a05cf.jpg)

![685e70ce51458592cbe966e781e5b56dea8bd5e889af4370d1049175daa10f7b.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/685e70ce51458592cbe966e781e5b56dea8bd5e889af4370d1049175daa10f7b.jpg)

![72458816b6dbb62efa92fdb89c1c4e980e6106514829823d110c1dbb380f5419.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/72458816b6dbb62efa92fdb89c1c4e980e6106514829823d110c1dbb380f5419.jpg)

![74a565c16bdf5789e55f9abd52037bed58e17506c814973adba6befb19bb8194.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/74a565c16bdf5789e55f9abd52037bed58e17506c814973adba6befb19bb8194.jpg)

![909cc713c02c2ec23dd2ae95c31178f48a7aba9ba432e4789ed1c914a77f5b56.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/909cc713c02c2ec23dd2ae95c31178f48a7aba9ba432e4789ed1c914a77f5b56.jpg)

![91033c0208e681f4e416c5831297bf5a33fbda7ec5e51a781b3ecddf0fd4f2ba.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/91033c0208e681f4e416c5831297bf5a33fbda7ec5e51a781b3ecddf0fd4f2ba.jpg)

![aaaa6229955d0ea0e37c076b9a5a2e04a523288fb760e8b1461fe1642a1fe5a3.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/aaaa6229955d0ea0e37c076b9a5a2e04a523288fb760e8b1461fe1642a1fe5a3.jpg)

![b2e0b75acf7e10370ac51945204295c496e60fdad4aeac56cb0fb569fee6211d.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/b2e0b75acf7e10370ac51945204295c496e60fdad4aeac56cb0fb569fee6211d.jpg)

![c4be5777313531d15deedd7208f30b90da1a0e6f7902e6d1bcb92636a842a09f.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/c4be5777313531d15deedd7208f30b90da1a0e6f7902e6d1bcb92636a842a09f.jpg)

![c4c9888cee0d1c64d0ef53c7d475348e26bb0dd0fe44faa6a270c8c7b6ebb21b.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/c4c9888cee0d1c64d0ef53c7d475348e26bb0dd0fe44faa6a270c8c7b6ebb21b.jpg)

![e3c9ffce205ec1e32b7c04b7aa0f56b2a7c853cdfc93e8312f648c7092d140b8.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/e3c9ffce205ec1e32b7c04b7aa0f56b2a7c853cdfc93e8312f648c7092d140b8.jpg)

![f572897129b3bf1a098214f428acfdcbe46c1093363785faa5bb0ff699c34bce.jpg](../icml_results/5425_PlugMem_%20A%20Task-Agnostic%20Plugin%20Memory%20Module%20for%20LLM%20Agents/tables/f572897129b3bf1a098214f428acfdcbe46c1093363785faa5bb0ff699c34bce.jpg)

<a id="paper-5378"></a>

## Concept Concentration for Faithful Representation Intervention


### Images

![2bec56345fc4e24d8cc1451eac95d1f0af865c32c860736da827ff7079c7a337.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/images/2bec56345fc4e24d8cc1451eac95d1f0af865c32c860736da827ff7079c7a337.jpg)

![684de832feeba1cd5708c18e63975b91d56c2d1913bdef8e112410b9d477b6c7.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/images/684de832feeba1cd5708c18e63975b91d56c2d1913bdef8e112410b9d477b6c7.jpg)

![9381f4b311f7086263bc6f35c8e354b3097306bdbd1583f3ee3cc838db624e63.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/images/9381f4b311f7086263bc6f35c8e354b3097306bdbd1583f3ee3cc838db624e63.jpg)

![c3b526f27d13ec710564013c1abb4ac21da9b11e62f6ffd6bcd1c28f17dd289b.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/images/c3b526f27d13ec710564013c1abb4ac21da9b11e62f6ffd6bcd1c28f17dd289b.jpg)

![cf71f18eb00baafd25678e98bf9ed8c79f3bb3aed35de4f9f6321f85c0c388a6.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/images/cf71f18eb00baafd25678e98bf9ed8c79f3bb3aed35de4f9f6321f85c0c388a6.jpg)

### Tables

![25376a9141aca6c78980a5a0d1728431293bf4ea23af0a73c9d77686c78a39da.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/25376a9141aca6c78980a5a0d1728431293bf4ea23af0a73c9d77686c78a39da.jpg)

![4c1ed877363f4e4bb0fe36639316e36acd9614271a3cedbac7c3c941f5a8d0cb.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/4c1ed877363f4e4bb0fe36639316e36acd9614271a3cedbac7c3c941f5a8d0cb.jpg)

![70f087876dac8da46a1988beb867d6a1dd976c953d9a218789f1f59d0092f148.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/70f087876dac8da46a1988beb867d6a1dd976c953d9a218789f1f59d0092f148.jpg)

![89e9452224d6c31cd8976c6bd935de14ea59c705712ecc78a1e6834c72977188.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/89e9452224d6c31cd8976c6bd935de14ea59c705712ecc78a1e6834c72977188.jpg)

![8ec749132c11c22db23ef7bec8cac51f15f6848d36e8483bd0d20f90a9f9ee47.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/8ec749132c11c22db23ef7bec8cac51f15f6848d36e8483bd0d20f90a9f9ee47.jpg)

![9ffdd0374b7344ed3bd55e04f79e21a2101c90d9c2170107d16c86c31ff74316.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/9ffdd0374b7344ed3bd55e04f79e21a2101c90d9c2170107d16c86c31ff74316.jpg)

![a6a76669a5c9cec3882405972f5203667596181c626ad96d582c89acd6fd1fbf.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/a6a76669a5c9cec3882405972f5203667596181c626ad96d582c89acd6fd1fbf.jpg)

![a7e9f8b3dfbdf63ed966d8effb02a792493535cd6130b6a34b1c738e4df5f647.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/a7e9f8b3dfbdf63ed966d8effb02a792493535cd6130b6a34b1c738e4df5f647.jpg)

![afe35bab6c844e43465f235c51ec8957513116effd3c1cb9377c234e4be9ab5d.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/afe35bab6c844e43465f235c51ec8957513116effd3c1cb9377c234e4be9ab5d.jpg)

![bb39daf7028f3ffd646c412f9e4c61f4183d01aacd3ec252aba6b39c0dce305f.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/bb39daf7028f3ffd646c412f9e4c61f4183d01aacd3ec252aba6b39c0dce305f.jpg)

![d35358972e214d5e1d05dfaf77d4916ac7b20b5689ba284f82db164bd4c30753.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/d35358972e214d5e1d05dfaf77d4916ac7b20b5689ba284f82db164bd4c30753.jpg)

![ef354e7508de68aa3a69e66740ed92d4bcada9823109267778ab0161976c00fb.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/ef354e7508de68aa3a69e66740ed92d4bcada9823109267778ab0161976c00fb.jpg)

![ef38366760bc27fa493519f131ae588651422dc09a0cf6ba3afee581a13a0317.jpg](../icml_results/5426_Concept%20Concentration%20for%20Faithful%20Representation%20Intervention/tables/ef38366760bc27fa493519f131ae588651422dc09a0cf6ba3afee581a13a0317.jpg)

<a id="paper-5379"></a>

## PISA: Privacy-Preserving Split Adaptation with Model IP Protection

### Images

![61ed6d5318e0f26d46b1d0c76e47ab54c270fdba03409bf1d3dbff7436b4e4e6.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/images/61ed6d5318e0f26d46b1d0c76e47ab54c270fdba03409bf1d3dbff7436b4e4e6.jpg)

![7e27a7134218e8bfc3afdb51dc879ba2ebdc4605b816eb2120f39c7670023603.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/images/7e27a7134218e8bfc3afdb51dc879ba2ebdc4605b816eb2120f39c7670023603.jpg)

![dbed5db323158ab9f52e101fd7f9ad4bcb57d1fb161f17200f2c9ae218c16752.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/images/dbed5db323158ab9f52e101fd7f9ad4bcb57d1fb161f17200f2c9ae218c16752.jpg)

### Tables

![1addc80c01592767c1ea12833695fe0ae0a2656ba32c19cd2dd01e03c908c7cb.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/1addc80c01592767c1ea12833695fe0ae0a2656ba32c19cd2dd01e03c908c7cb.jpg)

![2901fdf60c6bb09f22d5f3bf36b4361b8a17ad91b372b4ee97acb5d28fb406a6.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/2901fdf60c6bb09f22d5f3bf36b4361b8a17ad91b372b4ee97acb5d28fb406a6.jpg)

![34eb9ae4aa50d6924700a8c3927c40d2ed967e641eb384c669e6e86039ca9a21.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/34eb9ae4aa50d6924700a8c3927c40d2ed967e641eb384c669e6e86039ca9a21.jpg)

![73b8272ea89dc8845f89e6646dd5f4b99bf13d8ed51e17396b1682caacf5d550.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/73b8272ea89dc8845f89e6646dd5f4b99bf13d8ed51e17396b1682caacf5d550.jpg)

![823120379d06ec205a2ab0c2f2e76ed849e4143b88e0de8043d72c5bf5630149.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/823120379d06ec205a2ab0c2f2e76ed849e4143b88e0de8043d72c5bf5630149.jpg)

![a095eaf604bfa41762619de20a19d24ed8484a242cb912383fc925f87dbc52a6.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/a095eaf604bfa41762619de20a19d24ed8484a242cb912383fc925f87dbc52a6.jpg)

![b12a2a5bf82edb7cc707acfcc748a12b2762fe10496a3d1598128b0b57470367.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/b12a2a5bf82edb7cc707acfcc748a12b2762fe10496a3d1598128b0b57470367.jpg)

![c71b0727d869e7313a5c61116eb591084fa451fbfe0a4cfa04ef5a86fdbe2079.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/c71b0727d869e7313a5c61116eb591084fa451fbfe0a4cfa04ef5a86fdbe2079.jpg)

![d77421893961e9f13f1632f32a45326453acd43754dbf55240b1eb2caac788d8.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/d77421893961e9f13f1632f32a45326453acd43754dbf55240b1eb2caac788d8.jpg)

![e3a0e81a992728b71fcfd5258e682c013b09a7d412b3198c109e640b2fe5822a.jpg](../icml_results/5427_PISA_%20Privacy-Preserving%20Split%20Adaptation%20with%20Model%20IP%20Protection/tables/e3a0e81a992728b71fcfd5258e682c013b09a7d412b3198c109e640b2fe5822a.jpg)
