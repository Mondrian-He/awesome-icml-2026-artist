# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 94 of 197

## 目录 (Table of Contents)

1. [The Appeal and Reality of Recycling LoRAs with Adaptive Merging](#paper-3070)
2. [Accelerating Regression Tasks with Quantum Algorithms](#paper-3071)
3. [CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Priors](#paper-3072)
4. [Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning](#paper-3073)
5. [LARA: Latent Action Representation Alignment for Vision-Language-Action Models](#paper-3074)
6. [Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL](#paper-3075)
7. [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](#paper-3076)
8. [How does Bayesian Sampling help Membership Inference Attacks?](#paper-3077)
9. [TopoDistill: Distilling Global System Topology for Causal Discovery in Multivariate Time Series](#paper-3078)
10. [SpeedCP: Fast Kernel-based Conditional Conformal Prediction](#paper-3079)
11. [DIYHealth Suite: Dataset, Model, and Benchmark for Health Management at Home](#paper-3080)
12. [Continual GUI Agents](#paper-3081)
13. [Inverse Depth Scaling From Most Layers Being Similar](#paper-3082)
14. [Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic Matrices in Decentralized Learning](#paper-3083)
15. [Entropy-aware Span-Constrained Optimal Transport for Robust Cross-Tokenizer Knowledge Distillation](#paper-3084)
16. [RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization](#paper-3085)
17. [Toward Effective Multimodal Graph Foundation Model: A Divide-and-Conquer Based Approach](#paper-3086)
18. [Random Selection Reveals Implicit Knowledge Consensus in Code Generation](#paper-3087)
19. [Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services](#paper-3088)
20. [Geometric Rate–Distortion Invariance for Domain Generalization](#paper-3089)
21. [Hom-PGD$^+$: Fast Reparameterized Optimization over Non-convex Ball-Homeomorphic Set](#paper-3090)
22. [Geometric Pocket-Centric Protein Encoding for Polypharmacology-Guided Multi-Target Drug Design](#paper-3091)
23. [HSGG: Training-Free Hierarchical Scene Graph Generation with Geometry-Guided Relation Reasoning](#paper-3092)
24. [CAST: Modeling Visual State Transitions for Consistent Video Retrieval](#paper-3093)
25. [Causal Dependency-Aware Unsupervised Routing for Large Reasoning Models](#paper-3094)
26. [XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling](#paper-3095)
27. [Universal One-third Time Scaling in Learning Peaked Distributions](#paper-3096)
28. [Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search](#paper-3097)
29. [UniDrag: Unified Multi-Field Prediction and Robust Shape Optimization for Vehicle Aerodynamics](#paper-3098)
30. [SAGE: A Dataflow-Native Framework for Modular, Controllable, and Transparent LLM-Augmented Reasoning](#paper-3099)
31. [Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](#paper-3100)
32. [SLASH the Sink: Sharpening Structural Attention Inside LLMs](#paper-3101)
33. [Next-Gen CAPTCHAs: Leveraging the Cognitive Gap for Scalable and Diverse GUI-Agent Defense](#paper-3102)

---


<a id="paper-3070"></a>

## The Appeal and Reality of Recycling LoRAs with Adaptive Merging

### Images

![12eb43ce26385996a7ccec2c707430e618d819496d1f371a4cb4401325392b3d.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/12eb43ce26385996a7ccec2c707430e618d819496d1f371a4cb4401325392b3d.jpg)

![2f3a942948d7187bde505b3cdc6a4f066f80bfcc47c6f2f906e7e597eccd3d6b.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/2f3a942948d7187bde505b3cdc6a4f066f80bfcc47c6f2f906e7e597eccd3d6b.jpg)

![51818bea5a5aaeb42223b2e8ccb207ec02e5a9a6c511d2a1d7c31940bc727b54.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/51818bea5a5aaeb42223b2e8ccb207ec02e5a9a6c511d2a1d7c31940bc727b54.jpg)

![54bb9a552efe9f791568a485cecfa408361ab6ae1007568f860fb7b8a34efb0f.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/54bb9a552efe9f791568a485cecfa408361ab6ae1007568f860fb7b8a34efb0f.jpg)

![9ef37ced9a570ddb5a4dd5a6dcae08cb6a6f355b473f6cbc8e7e6fbd6b523df7.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/9ef37ced9a570ddb5a4dd5a6dcae08cb6a6f355b473f6cbc8e7e6fbd6b523df7.jpg)

![b5e7af41d1ac8f9d0e61a02d43830a6ad0ebfc1ec041656b2da30719bb4faecf.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/b5e7af41d1ac8f9d0e61a02d43830a6ad0ebfc1ec041656b2da30719bb4faecf.jpg)

![d73869c9d3ed8670b4ee2f7160edaca8df1eba5448af05912fcdf296a33c7a6f.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/images/d73869c9d3ed8670b4ee2f7160edaca8df1eba5448af05912fcdf296a33c7a6f.jpg)

### Tables

![7b14e95b0c3e24d2b32b6b62b5bca6810d27a408e529ad12660be16003d6a38e.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/7b14e95b0c3e24d2b32b6b62b5bca6810d27a408e529ad12660be16003d6a38e.jpg)

![c79f0b0ef365c3733e82f69b18311c03865c2f03fb41625e7533f666f40bf433.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/c79f0b0ef365c3733e82f69b18311c03865c2f03fb41625e7533f666f40bf433.jpg)

![da3425dd094dfb6f6a4290158a80867246d78e8201e444640cbea2b9e54568e2.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/da3425dd094dfb6f6a4290158a80867246d78e8201e444640cbea2b9e54568e2.jpg)

![db9469235ac6edc5599271347a7cb994697e1539a03313facfae2029afbc7ce9.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/db9469235ac6edc5599271347a7cb994697e1539a03313facfae2029afbc7ce9.jpg)

![e37ddd8e51ee19e8b4703ffc8629c79f02539a3881a8dc4d711b6a3a76c2b93a.jpg](../icml_results/3108_ConvexBench_%20Can%20LLMs%20Recognize%20Convex%20Functions_/tables/e37ddd8e51ee19e8b4703ffc8629c79f02539a3881a8dc4d711b6a3a76c2b93a.jpg)

## The Appeal and Reality of Recycling LoRAs with Adaptive Merging


### Images

![05e84d90f20a940160f21f163d2e9e5d2f1ffa11714923ac264fd372e9a90da5.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/05e84d90f20a940160f21f163d2e9e5d2f1ffa11714923ac264fd372e9a90da5.jpg)

![1581a7c5a9b5ea41e49184ac4371fde5ff3b2fd69e1746beca8566e2a689d72f.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/1581a7c5a9b5ea41e49184ac4371fde5ff3b2fd69e1746beca8566e2a689d72f.jpg)

![1791e6884cbba77557c5ddbba9d007387e554e93815dffa3e4e098e72fff603d.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/1791e6884cbba77557c5ddbba9d007387e554e93815dffa3e4e098e72fff603d.jpg)

![1d1f4428e7396ec311d7bb3419d9c2b2e5504e99b5fd58c18b3b29399867a887.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/1d1f4428e7396ec311d7bb3419d9c2b2e5504e99b5fd58c18b3b29399867a887.jpg)

![270807e862411b8c9679cc224c23fb81572bcee070d008c050ab1204e913fa90.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/270807e862411b8c9679cc224c23fb81572bcee070d008c050ab1204e913fa90.jpg)

![6451a1be3af4ee8d0f3e0b2482c5ed9da8f06e0742ffa65935d1a4ef109f11ce.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/6451a1be3af4ee8d0f3e0b2482c5ed9da8f06e0742ffa65935d1a4ef109f11ce.jpg)

![69fb8533010f2cde994670afe0d68234a8229dcc0b2c708d4fe3fcc0d49838b9.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/69fb8533010f2cde994670afe0d68234a8229dcc0b2c708d4fe3fcc0d49838b9.jpg)

![74877abdde674cd054f18845f68052d3f59b71ff9071026bbcd5e60f1672d3b3.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/74877abdde674cd054f18845f68052d3f59b71ff9071026bbcd5e60f1672d3b3.jpg)

![933df7024069352f478c03347a8dd2b53fac983355af7d0c830489af3c1f2850.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/933df7024069352f478c03347a8dd2b53fac983355af7d0c830489af3c1f2850.jpg)

![aca092f87f1272c53da7ebf4b18a5e4cbece52762b8a24b9fd7d98a3e4e83adb.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/aca092f87f1272c53da7ebf4b18a5e4cbece52762b8a24b9fd7d98a3e4e83adb.jpg)

![b369efbb49b869028b2c490dd9ec8a65b6f94a920a8c1e69dca8d0eb6568a5f0.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/b369efbb49b869028b2c490dd9ec8a65b6f94a920a8c1e69dca8d0eb6568a5f0.jpg)

![c1a238dd4edb76e58fdfcf9fd0e7367db9dfe4493346b154620497add34e9687.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/c1a238dd4edb76e58fdfcf9fd0e7367db9dfe4493346b154620497add34e9687.jpg)

![cb381d26f3148f67bea2b3ab5a94fc172e6576b7f3a22f656ec7105701e359fc.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/cb381d26f3148f67bea2b3ab5a94fc172e6576b7f3a22f656ec7105701e359fc.jpg)

![d51d05281c207fe1dcbb4a168d6190e1774e7398bbc80ffb85a37385bba7b9ab.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/d51d05281c207fe1dcbb4a168d6190e1774e7398bbc80ffb85a37385bba7b9ab.jpg)

![f66586179d3141aada4f984c1efbb5b909c3454e6353589c97e4119ec3ec2516.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/f66586179d3141aada4f984c1efbb5b909c3454e6353589c97e4119ec3ec2516.jpg)

![fe0052c459efe74293dfa7a53ad2f43cf4c54a913467e99aa8918118558ec458.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/images/fe0052c459efe74293dfa7a53ad2f43cf4c54a913467e99aa8918118558ec458.jpg)

### Tables

![01c480036328e40e554dfc778ee50f28e80587234647ce85ee579bdfe03a2eaa.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/01c480036328e40e554dfc778ee50f28e80587234647ce85ee579bdfe03a2eaa.jpg)

![02bbac5bf1d7d7346a51e8c305754b3bffdd8975ba4f17f18abc92104f02f332.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/02bbac5bf1d7d7346a51e8c305754b3bffdd8975ba4f17f18abc92104f02f332.jpg)

![4d02189a1bb6eac00c0870787e8a984fed50c318de0e35350dbf43a10a53081a.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/4d02189a1bb6eac00c0870787e8a984fed50c318de0e35350dbf43a10a53081a.jpg)

![4eac9863de1ec8043e9b48f8e0a21a0ec040673f9517fa3eb06f1a5256ac3c3b.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/4eac9863de1ec8043e9b48f8e0a21a0ec040673f9517fa3eb06f1a5256ac3c3b.jpg)

![52b0e14a0ac247a27930bc8ebd059cc0bb5dc3a89f9009a8c62f367cd04bc037.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/52b0e14a0ac247a27930bc8ebd059cc0bb5dc3a89f9009a8c62f367cd04bc037.jpg)

![5e7c3f95038f9cc33e58a79728003088df70a4af60447f7e7ce2a099fedb1da4.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/5e7c3f95038f9cc33e58a79728003088df70a4af60447f7e7ce2a099fedb1da4.jpg)

![817a47bf332de0761a214d72f2e54d70b77a72dd42632eb203f5c75f1be81253.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/817a47bf332de0761a214d72f2e54d70b77a72dd42632eb203f5c75f1be81253.jpg)

![86e69f68e59d44aba357906890a0061ac363a17651d83a7cb012c4c53083e549.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/86e69f68e59d44aba357906890a0061ac363a17651d83a7cb012c4c53083e549.jpg)

![8d7608fad1064ace321a72bfe72788d08830980ffb0c70f7ec467ae99f70be63.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/8d7608fad1064ace321a72bfe72788d08830980ffb0c70f7ec467ae99f70be63.jpg)

![8fd0f7cfc9ebc32a95309ee0fee17ebc2531465ab87b9c1fd40ee9708fbf2401.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/8fd0f7cfc9ebc32a95309ee0fee17ebc2531465ab87b9c1fd40ee9708fbf2401.jpg)

![9c70d1398965f1fafbbb84d93607bde90e211a9608b5bad96b8beedf18549ec8.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/9c70d1398965f1fafbbb84d93607bde90e211a9608b5bad96b8beedf18549ec8.jpg)

![a41161f9ed5bb9c62a2ea8866a5ed7c26c02a3b939458c83c20f1512aa67552f.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/a41161f9ed5bb9c62a2ea8866a5ed7c26c02a3b939458c83c20f1512aa67552f.jpg)

![b4a36eb8688b0f7050f71f6e69669359fa016d713b7abeaee1d7af41e1036aec.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/b4a36eb8688b0f7050f71f6e69669359fa016d713b7abeaee1d7af41e1036aec.jpg)

![dcdf1d4cd36d443a4fba505d326c8294fea69af58ec5d1e307de393b92c4e7b4.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/dcdf1d4cd36d443a4fba505d326c8294fea69af58ec5d1e307de393b92c4e7b4.jpg)

![e4e06bf08b2ba15a1ae3b65cb5fad2fce144c8acd8453a16ddaf5f86f9331c3c.jpg](../icml_results/3109_The%20Appeal%20and%20Reality%20of%20Recycling%20LoRAs%20with%20Adaptive%20Merging/tables/e4e06bf08b2ba15a1ae3b65cb5fad2fce144c8acd8453a16ddaf5f86f9331c3c.jpg)

<a id="paper-3071"></a>

## Accelerating Regression Tasks with Quantum Algorithms


### Images

![add652b113726273310156297edba9e0909e971cf103e91d08641fb59967e2ec.jpg](../icml_results/3110_Accelerating%20Regression%20Tasks%20with%20Quantum%20Algorithms/images/add652b113726273310156297edba9e0909e971cf103e91d08641fb59967e2ec.jpg)

### Tables

![9f6418d58ebfc7930e3679c16b92eb9c8ae2a19c0120ee6f5ba4e9a8502ab820.jpg](../icml_results/3110_Accelerating%20Regression%20Tasks%20with%20Quantum%20Algorithms/tables/9f6418d58ebfc7930e3679c16b92eb9c8ae2a19c0120ee6f5ba4e9a8502ab820.jpg)

<a id="paper-3072"></a>

## CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Priors


### Images

![12cd72eb259d858f8b4f29831c8008e4c81588b1ecad52a643c1fdd835f62e50.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/12cd72eb259d858f8b4f29831c8008e4c81588b1ecad52a643c1fdd835f62e50.jpg)

![258ee0d80748af4d0f8ae4ca4d1fdbf53c7cc427c023f5fe720636bec976ddb7.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/258ee0d80748af4d0f8ae4ca4d1fdbf53c7cc427c023f5fe720636bec976ddb7.jpg)

![273e53c6aa1faef67997a4b01fb3b6188bb997051a38359caa0caefc840a0a4f.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/273e53c6aa1faef67997a4b01fb3b6188bb997051a38359caa0caefc840a0a4f.jpg)

![53af42d168ac1f1a349599c1164a78676bb2600facc6234cd97cbbb37f4bd620.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/53af42d168ac1f1a349599c1164a78676bb2600facc6234cd97cbbb37f4bd620.jpg)

![7fbf4133f4ef798469dc39ec2bc060e89d81f5a4c334ca4e0b507fbd0ae90376.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/7fbf4133f4ef798469dc39ec2bc060e89d81f5a4c334ca4e0b507fbd0ae90376.jpg)

![8f6b4292ee8c5837cf74dba24b7d1394c6d6d70b9debb75992b7ba86f75e26ea.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/8f6b4292ee8c5837cf74dba24b7d1394c6d6d70b9debb75992b7ba86f75e26ea.jpg)

![94463bd75ce490fe39cb2fa9a616fbb3a290226dd8a4f15fc6edc304a1492256.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/94463bd75ce490fe39cb2fa9a616fbb3a290226dd8a4f15fc6edc304a1492256.jpg)

![945a0caba933a96975f2e0a57ac3f12f626113a5ddb55266e0700c18cf340a2e.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/945a0caba933a96975f2e0a57ac3f12f626113a5ddb55266e0700c18cf340a2e.jpg)

![951d1dba9b1bff5c918b4e52d8cb25368c18629f72573a871e0e22511acb9c19.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/951d1dba9b1bff5c918b4e52d8cb25368c18629f72573a871e0e22511acb9c19.jpg)

![b8c693b54f1c8cbd35c56cb8bb10d25d5d593a86919d12fc00f27b81fe7b1911.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/b8c693b54f1c8cbd35c56cb8bb10d25d5d593a86919d12fc00f27b81fe7b1911.jpg)

![c33b90a53575902f9c9ce8297cadb0675346bb1f07e343b5fde963c603ac8e76.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/c33b90a53575902f9c9ce8297cadb0675346bb1f07e343b5fde963c603ac8e76.jpg)

![c8a479943277c5a5332402f47cf8ed9ed8d94781e06bd198039416ac6440c030.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/c8a479943277c5a5332402f47cf8ed9ed8d94781e06bd198039416ac6440c030.jpg)

![e83e5700840ba01697e738bb7492f35304caaa200b5c381789e6a281d8766f48.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/e83e5700840ba01697e738bb7492f35304caaa200b5c381789e6a281d8766f48.jpg)

![e862a6075976247d709694987d18919fa04bbfe9e7da54854cb937936c40d116.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/e862a6075976247d709694987d18919fa04bbfe9e7da54854cb937936c40d116.jpg)

![ff1070dab634d828a5dee8f5aeb6478ddd7960f77d5f36bba2b334e41162fb78.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/images/ff1070dab634d828a5dee8f5aeb6478ddd7960f77d5f36bba2b334e41162fb78.jpg)

### Tables

![00cc97ced33d8fd9a3dffccf5c57c90095d3742274c26397ac5f5b1f073cbe91.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/00cc97ced33d8fd9a3dffccf5c57c90095d3742274c26397ac5f5b1f073cbe91.jpg)

![0ca277bfe1c1adc8e8b12934d94343adb0d3c829c86e2c23bd4f2f67f7fc5abd.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/0ca277bfe1c1adc8e8b12934d94343adb0d3c829c86e2c23bd4f2f67f7fc5abd.jpg)

![1fc883a1c2a44dacc0bc28c2758f62cb3a749be54095ede5955db0319791db67.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/1fc883a1c2a44dacc0bc28c2758f62cb3a749be54095ede5955db0319791db67.jpg)

![20a211086df701c34f21840bb159755dc0f08efeed8535aa73380a5967e300dd.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/20a211086df701c34f21840bb159755dc0f08efeed8535aa73380a5967e300dd.jpg)

![33ba4159058ca9fffcd3379e9d424bcadda7d8bbdc9e8087cf1a2426ad7133cf.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/33ba4159058ca9fffcd3379e9d424bcadda7d8bbdc9e8087cf1a2426ad7133cf.jpg)

![42837c6c673909416c49f29c1585041c1ebea63830108ce3b33ce273a4e57544.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/42837c6c673909416c49f29c1585041c1ebea63830108ce3b33ce273a4e57544.jpg)

![c34d713edbbfe6bf881c5023fff3d6942dc679337feec9b6dd52d6fc30fdffd6.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/c34d713edbbfe6bf881c5023fff3d6942dc679337feec9b6dd52d6fc30fdffd6.jpg)

![ed01210cec888566f8d88a7ba333baa05262e4ea7f50b89ec9b691d3666ef6c4.jpg](../icml_results/3111_CamGeo_%20Sparse%20Camera-Conditioned%20Image-to-Video%20Generation%20with%203D%20Geometry%20Priors/tables/ed01210cec888566f8d88a7ba333baa05262e4ea7f50b89ec9b691d3666ef6c4.jpg)

<a id="paper-3073"></a>

## Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning


### Images

![1c4170d746a75b9d3a0c27f2db2e4b0c8374deff36a2e11da5bdb8c40c60cd97.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/1c4170d746a75b9d3a0c27f2db2e4b0c8374deff36a2e11da5bdb8c40c60cd97.jpg)

![408372e2f1f3b68c19a78288b3fbb6c54c5ef773682d9493a8138b4809a2997e.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/408372e2f1f3b68c19a78288b3fbb6c54c5ef773682d9493a8138b4809a2997e.jpg)

![45383b6c9dce932670a3db5d3365d894451fcd83fdecd82630bc48de8c82b2c1.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/45383b6c9dce932670a3db5d3365d894451fcd83fdecd82630bc48de8c82b2c1.jpg)

![5bdb0d646b426db9cca687335ab0ec00984e919a11a388e10e735964f4e2532e.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/5bdb0d646b426db9cca687335ab0ec00984e919a11a388e10e735964f4e2532e.jpg)

![81930aa90957c289ad2ed54da1f945d7bc35d9653ef2414a28d9c56d0d78c535.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/81930aa90957c289ad2ed54da1f945d7bc35d9653ef2414a28d9c56d0d78c535.jpg)

![9e6b00bcc7d8582fd50c0feb4b3532a963bf1f3af89c9e1ae13bf51dfa3dbbc4.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/9e6b00bcc7d8582fd50c0feb4b3532a963bf1f3af89c9e1ae13bf51dfa3dbbc4.jpg)

![b139dbcb9b88a90392bbf1686d5146122e63703c8992741e4eabe71960194ae0.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/b139dbcb9b88a90392bbf1686d5146122e63703c8992741e4eabe71960194ae0.jpg)

![e0310cbe4a17e7023f788665d6b2b303b9aa30bb8c7ad7192203a836b594863b.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/e0310cbe4a17e7023f788665d6b2b303b9aa30bb8c7ad7192203a836b594863b.jpg)

![e5148b4ca430c499dc92b83a463a6084e11fb9df6a5147329af81f3ed8424a86.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/images/e5148b4ca430c499dc92b83a463a6084e11fb9df6a5147329af81f3ed8424a86.jpg)

### Tables

![2daeab78cf04209cc0a738867edfa1e60667898e272f7f4acf6222160ed54789.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/2daeab78cf04209cc0a738867edfa1e60667898e272f7f4acf6222160ed54789.jpg)

![3687a3f163bd3b530eaa4be25aeed6ac68dc2d4e8a028453654f9e4daecd1e98.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/3687a3f163bd3b530eaa4be25aeed6ac68dc2d4e8a028453654f9e4daecd1e98.jpg)

![4061cf50f458e726da41e3d887bf2ef7152f8f55e87eb67f5df980ea7da7ff1a.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/4061cf50f458e726da41e3d887bf2ef7152f8f55e87eb67f5df980ea7da7ff1a.jpg)

![4e41cde5c64ec58166519b451d941b8140bcc3077476511625794ff69b1eaff7.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/4e41cde5c64ec58166519b451d941b8140bcc3077476511625794ff69b1eaff7.jpg)

![757d9c0e28de0be0ab69bada023ef96ebcd7ec1efbadc0eb4543eb524b12e4ac.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/757d9c0e28de0be0ab69bada023ef96ebcd7ec1efbadc0eb4543eb524b12e4ac.jpg)

![a8896f600d4a8aa76ab661e396918ed562cb92b2db2a345a45bc0529616bfb86.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/a8896f600d4a8aa76ab661e396918ed562cb92b2db2a345a45bc0529616bfb86.jpg)

![c0f6b37bce56bab2d3fd470afa2d96151450a5b5fee17ef3edcaef8c5eba4f71.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/c0f6b37bce56bab2d3fd470afa2d96151450a5b5fee17ef3edcaef8c5eba4f71.jpg)

![ea2804dbe5489179fdb7342fb843bfd9f4114ce9d43c0b6523ae3f3715f4c303.jpg](../icml_results/3112_Equivariant%20Covariance%20Tensors_%20Guaranteed%20SPD%20Uncertainty%20for%20Tensor-Valued%20Geometric%20Learning/tables/ea2804dbe5489179fdb7342fb843bfd9f4114ce9d43c0b6523ae3f3715f4c303.jpg)

<a id="paper-3074"></a>

## LARA: Latent Action Representation Alignment for Vision-Language-Action Models


### Images

![34bfaad3d691766ec69fe0e52a1d33b89a2b0930f24ab6a7432be7a9f5a73593.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/34bfaad3d691766ec69fe0e52a1d33b89a2b0930f24ab6a7432be7a9f5a73593.jpg)

![35e85b0c6377c5df925fc2559b96363112605737bafbdc8d8f0f618dc67ae0e4.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/35e85b0c6377c5df925fc2559b96363112605737bafbdc8d8f0f618dc67ae0e4.jpg)

![4168484511de075ea3831909076b216fdf985a3d85e7472f17d08df1df507c33.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/4168484511de075ea3831909076b216fdf985a3d85e7472f17d08df1df507c33.jpg)

![4a7373cfc6ef39fc6dbb22cc081f49a796af60b4282d8c28507385a2e23e701e.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/4a7373cfc6ef39fc6dbb22cc081f49a796af60b4282d8c28507385a2e23e701e.jpg)

![56eae3e26428fe2090ec9582d9f8f73570fa817134a035e2b6c10064f8f3cf2e.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/56eae3e26428fe2090ec9582d9f8f73570fa817134a035e2b6c10064f8f3cf2e.jpg)

![6caaa5a07229e698aef87df557c6dcb24c724fa5a952ed44a5900958ef033a40.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/6caaa5a07229e698aef87df557c6dcb24c724fa5a952ed44a5900958ef033a40.jpg)

![6daf592de2d3a6528060811e21fbb474935453492c34290a134fb77490e04651.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/6daf592de2d3a6528060811e21fbb474935453492c34290a134fb77490e04651.jpg)

![c9cea45e8562e1842b4db56ab33932d97fcb14e66d3f7286832814e3d6dc5476.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/c9cea45e8562e1842b4db56ab33932d97fcb14e66d3f7286832814e3d6dc5476.jpg)

![e644964360b2844b24d22da98b6730fd00c644a813727193e5d07eac1bef0b9a.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/e644964360b2844b24d22da98b6730fd00c644a813727193e5d07eac1bef0b9a.jpg)

![f497f9c226011a89a791782034a7b2de5d9899d13469ef94e622e0ffab60aea3.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/images/f497f9c226011a89a791782034a7b2de5d9899d13469ef94e622e0ffab60aea3.jpg)

### Tables

![1df4cdd1389e5afcdb998aadedabdf0d7ab6aa261845e06e48d396ade57efd52.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/1df4cdd1389e5afcdb998aadedabdf0d7ab6aa261845e06e48d396ade57efd52.jpg)

![21cf6970d2ac41bbba46f35a89795a7b5f09e214f132fc3c40c3d261a5a05f78.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/21cf6970d2ac41bbba46f35a89795a7b5f09e214f132fc3c40c3d261a5a05f78.jpg)

![2302ed117d7abfcfcfd5ed7ff9eb675dbe5208f4b4cd1d893972837a94557a11.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/2302ed117d7abfcfcfd5ed7ff9eb675dbe5208f4b4cd1d893972837a94557a11.jpg)

![3bdd9b3fe684c560074672e480b4517c3312a157d141161bb0defdde6f7b099a.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/3bdd9b3fe684c560074672e480b4517c3312a157d141161bb0defdde6f7b099a.jpg)

![99fc2847f1a8a5b870b742d368766bf6a686b899f46dac777a99e61a6b32129b.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/99fc2847f1a8a5b870b742d368766bf6a686b899f46dac777a99e61a6b32129b.jpg)

![cde4edc1262bd4e085900a55e8e2fd22a049880238ff45da9381b58e6869e56b.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/cde4edc1262bd4e085900a55e8e2fd22a049880238ff45da9381b58e6869e56b.jpg)

![df1bc6c05ca656c0dfa6276fe579ab0090e3ef55f2ea9b7ef4d6181b44b31423.jpg](../icml_results/3113_LARA_%20Latent%20Action%20Representation%20Alignment%20for%20Vision-Language-Action%20Models/tables/df1bc6c05ca656c0dfa6276fe579ab0090e3ef55f2ea9b7ef4d6181b44b31423.jpg)

<a id="paper-3075"></a>

## Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL


### Images

![0247a5808ed7b5952fcea1b00e8f939c07d3585bf25c6338cb777081db80c715.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/0247a5808ed7b5952fcea1b00e8f939c07d3585bf25c6338cb777081db80c715.jpg)

![07f285c3f2cd0a14a15713f6e30f898e518a25dc9a3a67c9747275d178c5f2ec.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/07f285c3f2cd0a14a15713f6e30f898e518a25dc9a3a67c9747275d178c5f2ec.jpg)

![0a444fae049e697b77c83cef861f669893a528e49f88a7bfc4c18666a131c6f8.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/0a444fae049e697b77c83cef861f669893a528e49f88a7bfc4c18666a131c6f8.jpg)

![127042fac864332c9e64d08daff4e5362676b232892264ad560456eeb9c8aae1.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/127042fac864332c9e64d08daff4e5362676b232892264ad560456eeb9c8aae1.jpg)

![221efb2686ac3049f8ea6949daaecbff05daa3d34c78de2bba51270e5cc3b8b7.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/221efb2686ac3049f8ea6949daaecbff05daa3d34c78de2bba51270e5cc3b8b7.jpg)

![350d8d48c3b464c3a61740df21a688e6845d5e7d77d2a8cc8b022b718b37c270.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/350d8d48c3b464c3a61740df21a688e6845d5e7d77d2a8cc8b022b718b37c270.jpg)

![4620f2e3e770778c52eaa61f3267dd1d04b8c2af28319eea289201b0364abf69.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/4620f2e3e770778c52eaa61f3267dd1d04b8c2af28319eea289201b0364abf69.jpg)

![49f7be5e615e3b9d81e0289b4f4b187375c6f5b35530539b1a1916d56d083558.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/49f7be5e615e3b9d81e0289b4f4b187375c6f5b35530539b1a1916d56d083558.jpg)

![598f6965fb9c7d183d510fe8bb08db959c3a14da5cc35ae91ea99d849629a153.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/598f6965fb9c7d183d510fe8bb08db959c3a14da5cc35ae91ea99d849629a153.jpg)

![5aa1b23e6fdefe85993c481661442d79e3faea166d09e418f7ee1e5cec5436d9.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/5aa1b23e6fdefe85993c481661442d79e3faea166d09e418f7ee1e5cec5436d9.jpg)

![64d247d3ec3143d7aeaa98894e32a0c620158b8c42086a45f1ea7dd77ac6a52b.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/64d247d3ec3143d7aeaa98894e32a0c620158b8c42086a45f1ea7dd77ac6a52b.jpg)

![7256181874b2d4b622d2e25277e8e9b953ffb0a899004a9664b2423392ab26e6.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/7256181874b2d4b622d2e25277e8e9b953ffb0a899004a9664b2423392ab26e6.jpg)

![975111238284b3eacf1fd778dd3db4f8f57a83a298acd63d8c5b4d3bc0fff6f8.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/975111238284b3eacf1fd778dd3db4f8f57a83a298acd63d8c5b4d3bc0fff6f8.jpg)

![a64fcf3aaf9c7a3f6062c0964467cb07524789b68dca54ce5dc021b803e24c35.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/a64fcf3aaf9c7a3f6062c0964467cb07524789b68dca54ce5dc021b803e24c35.jpg)

![b264a17d1705affbad89bdc75e02de05c2310f20f932cbe022b10e98ce36dcdd.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/b264a17d1705affbad89bdc75e02de05c2310f20f932cbe022b10e98ce36dcdd.jpg)

![c037bc6062a9e6c809c49c008f9ba2e9e0bd404d58e9bad48a61a70ea642eae0.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/c037bc6062a9e6c809c49c008f9ba2e9e0bd404d58e9bad48a61a70ea642eae0.jpg)

![c6ba034b2a6072f2a22db7e1ed1113bfecdb5302f7acdcf10239bbbd089954d1.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/c6ba034b2a6072f2a22db7e1ed1113bfecdb5302f7acdcf10239bbbd089954d1.jpg)

![cadc0cbb42c8ab3658dab3dcf5eb4c204abe7df7989583ed3a3b4b6f7d52a5d4.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/cadc0cbb42c8ab3658dab3dcf5eb4c204abe7df7989583ed3a3b4b6f7d52a5d4.jpg)

![cf14b3c658e70b9fb83fedf08fce746dfbfc32e704d58ab495f6c12c178610c8.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/cf14b3c658e70b9fb83fedf08fce746dfbfc32e704d58ab495f6c12c178610c8.jpg)

![d172f338a381564051d5d8e9f3104da55955f2d554b04860311206d2f8c7bab1.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/d172f338a381564051d5d8e9f3104da55955f2d554b04860311206d2f8c7bab1.jpg)

![e71d41dafdf27afbd5e5208e292a46e05f11c60280f3fcec0f5374394b2a8196.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/e71d41dafdf27afbd5e5208e292a46e05f11c60280f3fcec0f5374394b2a8196.jpg)

![ea8322d18f5bc579a9842ffdde71af9e68b19c9865ea1bad42aed52de3e7217d.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/ea8322d18f5bc579a9842ffdde71af9e68b19c9865ea1bad42aed52de3e7217d.jpg)

![f9b4b68d46dd94e21849c137c297f78caa5a17fa9c3fde8716f05582634e7192.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/images/f9b4b68d46dd94e21849c137c297f78caa5a17fa9c3fde8716f05582634e7192.jpg)

### Tables

![08015cf073ffe19936c99b0f2a8240bd88d31fa6c2e9960acfd8450d055aa7eb.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/08015cf073ffe19936c99b0f2a8240bd88d31fa6c2e9960acfd8450d055aa7eb.jpg)

![1beaeaa17498a71a4b86c1ba1c2caf288d0f2b8906d41e1399ba1c08bda3c43e.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/1beaeaa17498a71a4b86c1ba1c2caf288d0f2b8906d41e1399ba1c08bda3c43e.jpg)

![1fc86b39bf1443f3ac4e8080a1dea123f805fea8f5a8cde5f07a3d8462facdf8.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/1fc86b39bf1443f3ac4e8080a1dea123f805fea8f5a8cde5f07a3d8462facdf8.jpg)

![2078457ad61bd253decea0b6577545bb5263adf623201abdc7d294e2e91d3b01.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/2078457ad61bd253decea0b6577545bb5263adf623201abdc7d294e2e91d3b01.jpg)

![21dde71d60a848212406307508290c9f2b2487d1fb39f0073604a9d94e75ead1.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/21dde71d60a848212406307508290c9f2b2487d1fb39f0073604a9d94e75ead1.jpg)

![429c6ac78fbdfe594cafe3c2791b380d6a3577dd5887f4ff6fa120945272eb24.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/429c6ac78fbdfe594cafe3c2791b380d6a3577dd5887f4ff6fa120945272eb24.jpg)

![4814c10c6b5286b531962d3b612c448e5f161069326157e65d9ae32f48604ccc.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/4814c10c6b5286b531962d3b612c448e5f161069326157e65d9ae32f48604ccc.jpg)

![91545906101802371eafc12c8892680297e06d10bb66d8e852e9c30899c5025a.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/91545906101802371eafc12c8892680297e06d10bb66d8e852e9c30899c5025a.jpg)

![d8e5cdd7ee2dedf37e0773ee700803e88d9478f5fb56b9ef36d194f7940608a9.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/d8e5cdd7ee2dedf37e0773ee700803e88d9478f5fb56b9ef36d194f7940608a9.jpg)

![db6ad34ff4a3c18d7a76c5c4c771b28ab1faec7a32354bc6fd6636d42212af82.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/db6ad34ff4a3c18d7a76c5c4c771b28ab1faec7a32354bc6fd6636d42212af82.jpg)

![f08e635c9fe23af420422e18abd23417e33e660cfe98b7baaaf216f007fd9953.jpg](../icml_results/3114_Clipping%20Low-Probability%20Tokens%20in%20SFT%20Yields%20a%20Generalizable%20Initialization%20for%20RL/tables/f08e635c9fe23af420422e18abd23417e33e660cfe98b7baaaf216f007fd9953.jpg)

<a id="paper-3076"></a>

## Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models


### Images

![03e0a422d7a94486335db547c4f0e3985991000291b32f5e01688b28eba1296d.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/03e0a422d7a94486335db547c4f0e3985991000291b32f5e01688b28eba1296d.jpg)

![58eec049dbcea8156b1404b3faa9d075ff818b2fba938547c545ad6a87ab0071.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/58eec049dbcea8156b1404b3faa9d075ff818b2fba938547c545ad6a87ab0071.jpg)

![658c9ea36b756487176a25d1a263ec8f2687dedb3cbd12343ed059cd2c398dfb.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/658c9ea36b756487176a25d1a263ec8f2687dedb3cbd12343ed059cd2c398dfb.jpg)

![7646803a5cfcef65cada2ccb42262ae3534437b79ff58bec7ae86f4ac6dda54e.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/7646803a5cfcef65cada2ccb42262ae3534437b79ff58bec7ae86f4ac6dda54e.jpg)

![7dddfc3f31c3d9906452cc1e3eb5670d04908056294b905558eaba3dde09cfd0.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/7dddfc3f31c3d9906452cc1e3eb5670d04908056294b905558eaba3dde09cfd0.jpg)

![88a37e59ba1184ea69af32a291e31040d97fccd750e3bdba6118750652ea678e.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/88a37e59ba1184ea69af32a291e31040d97fccd750e3bdba6118750652ea678e.jpg)

![e25519bdf9ff3f98470c45c53c1ca9ab25802be13ba7b3f8322ba14d46c8c690.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/images/e25519bdf9ff3f98470c45c53c1ca9ab25802be13ba7b3f8322ba14d46c8c690.jpg)

### Tables

![01fb2d949f428f4687bbb79e25f60ad5ed8b4458681ec49ec76d90c9d959bb44.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/01fb2d949f428f4687bbb79e25f60ad5ed8b4458681ec49ec76d90c9d959bb44.jpg)

![1a9dd73e2fe380dc62243f0b6233119abee3b1030276b132dbd08875753558e2.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/1a9dd73e2fe380dc62243f0b6233119abee3b1030276b132dbd08875753558e2.jpg)

![1bb1cfd6890839dcf31114ecaf3d98e3e3794ebd027d59a0c5175a6b73c860db.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/1bb1cfd6890839dcf31114ecaf3d98e3e3794ebd027d59a0c5175a6b73c860db.jpg)

![29bded0ca759589800b13afd9574b01aac0d31dbec17d307aaf5491e7e32ba61.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/29bded0ca759589800b13afd9574b01aac0d31dbec17d307aaf5491e7e32ba61.jpg)

![398bd0ac1623a0affc3261a0aa4f092041bee3bd0014f920bd86cd697c39cbc1.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/398bd0ac1623a0affc3261a0aa4f092041bee3bd0014f920bd86cd697c39cbc1.jpg)

![45d585d9339d3307b72fd23fb8bbc8d33eff070df86dfb8584f931777c51fbf2.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/45d585d9339d3307b72fd23fb8bbc8d33eff070df86dfb8584f931777c51fbf2.jpg)

![5433679f9f1c3d4668a0bc1a111a57414f061600291b37bbd49f8b1a7c9296e2.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/5433679f9f1c3d4668a0bc1a111a57414f061600291b37bbd49f8b1a7c9296e2.jpg)

![58a8e33bcb0dc18a9f28e9056179ad994e8f1eb872be1e22900df64562d21eda.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/58a8e33bcb0dc18a9f28e9056179ad994e8f1eb872be1e22900df64562d21eda.jpg)

![6160f57d86808689077023c3bb2d670c564e175ba9c3eeae2730ce1544004c84.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/6160f57d86808689077023c3bb2d670c564e175ba9c3eeae2730ce1544004c84.jpg)

![71d291433090b1128fea8877a98f7d06a9714d087b07ece0b6fabcb969f45afe.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/71d291433090b1128fea8877a98f7d06a9714d087b07ece0b6fabcb969f45afe.jpg)

![7600f5111636af0749a0dcd8c1e5039bb13cee1d41c04b6302ae556b6c906398.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/7600f5111636af0749a0dcd8c1e5039bb13cee1d41c04b6302ae556b6c906398.jpg)

![810f433acb35ae716297061e9734f3868512618efc0f548654327dd949f5b947.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/810f433acb35ae716297061e9734f3868512618efc0f548654327dd949f5b947.jpg)

![8279c1952e95340940952e913dce62482eaf773d481818756ac989af0877f522.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/8279c1952e95340940952e913dce62482eaf773d481818756ac989af0877f522.jpg)

![8cd9634be0c78270ad39bd36a997940e4261c26f2c25d46158f629750dcf4239.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/8cd9634be0c78270ad39bd36a997940e4261c26f2c25d46158f629750dcf4239.jpg)

![8fd0c006f3ec131aa91f19b2b6dcff75000a3943b07ac573abdc3886a5d37a2b.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/8fd0c006f3ec131aa91f19b2b6dcff75000a3943b07ac573abdc3886a5d37a2b.jpg)

![9388c0977c1520e50b1f4556d81cf1bfa42eb8b52164d8acbc56153cc920d601.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/9388c0977c1520e50b1f4556d81cf1bfa42eb8b52164d8acbc56153cc920d601.jpg)

![a335742c3ba4cac3fd25e9b278b23b15967fab9fbb2f70a9da9a21c9e98579e7.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/a335742c3ba4cac3fd25e9b278b23b15967fab9fbb2f70a9da9a21c9e98579e7.jpg)

![a63ec7cfb67b24e54fe75ea2b1b2c2501c2e9aac78c6e2feb5c6c38e56a3b854.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/a63ec7cfb67b24e54fe75ea2b1b2c2501c2e9aac78c6e2feb5c6c38e56a3b854.jpg)

![b93eda469fa6f2de84547a17fa3ebc832b0d85554371f0ae31c30505a83ae9c6.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/b93eda469fa6f2de84547a17fa3ebc832b0d85554371f0ae31c30505a83ae9c6.jpg)

![ee2ae70c11b85105097d28d589761643dd0dc3ba219d5b7add0ea74c4dde9f66.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/ee2ae70c11b85105097d28d589761643dd0dc3ba219d5b7add0ea74c4dde9f66.jpg)

![f64186c3e265ea036f302a54b08b91a03c45e08d09abe9c1d04f1ecc6f2ccafb.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/f64186c3e265ea036f302a54b08b91a03c45e08d09abe9c1d04f1ecc6f2ccafb.jpg)

![fccf120a834104ca3199119c5d695a6c46e4ef3362625c52cb91256a521b03fb.jpg](../icml_results/3115_Chasing%20Moving%20Targets%20with%20Online%20Self-Play%20Reinforcement%20Learning%20for%20Safer%20Language%20Models/tables/fccf120a834104ca3199119c5d695a6c46e4ef3362625c52cb91256a521b03fb.jpg)

<a id="paper-3077"></a>

## How does Bayesian Sampling help Membership Inference Attacks?


### Images

![04043412bfdeceb92c8eefd702451c637dc5b71a41692e9bbd950308bda77d30.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/04043412bfdeceb92c8eefd702451c637dc5b71a41692e9bbd950308bda77d30.jpg)

![241896476a3ccf89187b1e1230f841fa28453e076800a876c19b21fa148a666e.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/241896476a3ccf89187b1e1230f841fa28453e076800a876c19b21fa148a666e.jpg)

![3f2c9b09330c0b5784ca591dd7a647c0cb91f7a1d258e32a85673cb952c27ea8.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/3f2c9b09330c0b5784ca591dd7a647c0cb91f7a1d258e32a85673cb952c27ea8.jpg)

![54f8f990fb3b72e20bed4f1a89cd57dc360903c683773d96b71700d891023eaf.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/54f8f990fb3b72e20bed4f1a89cd57dc360903c683773d96b71700d891023eaf.jpg)

![621eb1932200409a51c30edab8b79430c4b7cc70fa3dee497f17b0038dc24985.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/621eb1932200409a51c30edab8b79430c4b7cc70fa3dee497f17b0038dc24985.jpg)

![64525f56bb9f684d77caf5a45f645df48a964f043ee5ed52f27a7c14d96c8a94.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/64525f56bb9f684d77caf5a45f645df48a964f043ee5ed52f27a7c14d96c8a94.jpg)

![6b847b44ef36137a35c9f56297d3fe22beee48bc67e77b9c3902b4bdb98ffe45.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/6b847b44ef36137a35c9f56297d3fe22beee48bc67e77b9c3902b4bdb98ffe45.jpg)

![be9aa0c56233211bcb1894dc4430eff59ab0535f95e9791c567eb4120f404786.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/be9aa0c56233211bcb1894dc4430eff59ab0535f95e9791c567eb4120f404786.jpg)

![bf4073d32983e9bf3c313c82add0efb9670bb2a649d3a55c5da301b2f6c19441.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/bf4073d32983e9bf3c313c82add0efb9670bb2a649d3a55c5da301b2f6c19441.jpg)

![eb04adf0ac8c7762c1396800d933af6c76efda086b1a0f2d0dfc36b003315998.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/images/eb04adf0ac8c7762c1396800d933af6c76efda086b1a0f2d0dfc36b003315998.jpg)

### Tables

![1364ecd8c842febd782d8ea3477ecad9ad153d76adddf6e16cc8a70a825317c8.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/1364ecd8c842febd782d8ea3477ecad9ad153d76adddf6e16cc8a70a825317c8.jpg)

![1be38a96e693082db373ba7f0f6e30fb3a11f6bc89ab20b676fd07515edc5a69.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/1be38a96e693082db373ba7f0f6e30fb3a11f6bc89ab20b676fd07515edc5a69.jpg)

![255ffa06c5e43063a31083e989c3d8c2246240a222bf81fd88bc7433004dc6a6.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/255ffa06c5e43063a31083e989c3d8c2246240a222bf81fd88bc7433004dc6a6.jpg)

![49e5abaa1331184f3deb540b5ca899ceda8dc63b830635d00d8c3cad12dbef19.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/49e5abaa1331184f3deb540b5ca899ceda8dc63b830635d00d8c3cad12dbef19.jpg)

![60fd959e834b19352b0c397dea9da7df09cf90e03481b930502726be5842bf69.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/60fd959e834b19352b0c397dea9da7df09cf90e03481b930502726be5842bf69.jpg)

![7c69652c52d0d696f740cfe36a0f37d616ad6147fc93d0bcc536106d654ff8a9.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/7c69652c52d0d696f740cfe36a0f37d616ad6147fc93d0bcc536106d654ff8a9.jpg)

![b2718698865a889a6067ae5f87a582c29118e5a5224dcbc9bee05699754675f1.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/b2718698865a889a6067ae5f87a582c29118e5a5224dcbc9bee05699754675f1.jpg)

![d835e8b333d7bbad0d4ae3351b527d2f2f84d894d0b4900eb5470ebab62d9c86.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/d835e8b333d7bbad0d4ae3351b527d2f2f84d894d0b4900eb5470ebab62d9c86.jpg)

![df09cb6a4d5f7575cbe0629fec25f40595688cfda136e14e1890384782b9a679.jpg](../icml_results/3116_How%20does%20Bayesian%20Sampling%20help%20Membership%20Inference%20Attacks_/tables/df09cb6a4d5f7575cbe0629fec25f40595688cfda136e14e1890384782b9a679.jpg)

<a id="paper-3078"></a>

## TopoDistill: Distilling Global System Topology for Causal Discovery in Multivariate Time Series


### Images

![605052d485e866048cd83bbef9a2a0504e3d79067fcc3af9c42bdeb6a5c3f9be.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/images/605052d485e866048cd83bbef9a2a0504e3d79067fcc3af9c42bdeb6a5c3f9be.jpg)

![64d405d2dcf624a41915d1a16ef08c0b14cdc92ba0160fe6d66c71b22989dd8f.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/images/64d405d2dcf624a41915d1a16ef08c0b14cdc92ba0160fe6d66c71b22989dd8f.jpg)

![7fc99ddec51f5f69649cdf72335d8bea16b9583d9793fda906cf6efa597a8999.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/images/7fc99ddec51f5f69649cdf72335d8bea16b9583d9793fda906cf6efa597a8999.jpg)

![eac8ec0d11e754d2f93561c463e2133e78214979d90e44cf8b5e95fddec4fb87.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/images/eac8ec0d11e754d2f93561c463e2133e78214979d90e44cf8b5e95fddec4fb87.jpg)

### Tables

![462aca8fd589e6efb15e33800354229c1c15ad8f1c739e3df4be2057340f158c.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/tables/462aca8fd589e6efb15e33800354229c1c15ad8f1c739e3df4be2057340f158c.jpg)

![71c1445bc14e5aa22798158d88ac8aa0d4d697ee1ea2d8337b97aff6a02a902f.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/tables/71c1445bc14e5aa22798158d88ac8aa0d4d697ee1ea2d8337b97aff6a02a902f.jpg)

![7e57c28ecafd6597fed54cf14a9c54ebf8407509a7391a3ebe96f129f7c9baba.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/tables/7e57c28ecafd6597fed54cf14a9c54ebf8407509a7391a3ebe96f129f7c9baba.jpg)

![c71cf5be27745e04585d1c13773c3feb4d130819a3fb1d4db588b68b4d0fe5ed.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/tables/c71cf5be27745e04585d1c13773c3feb4d130819a3fb1d4db588b68b4d0fe5ed.jpg)

![d38a5b9c6b849f15acec486322143e47d6557bdf92100469a126a908084a5d1f.jpg](../icml_results/3117_TopoDistill_%20Distilling%20Global%20System%20Topology%20for%20Causal%20Discovery%20in%20Multivariate%20Time%20Series/tables/d38a5b9c6b849f15acec486322143e47d6557bdf92100469a126a908084a5d1f.jpg)

<a id="paper-3079"></a>

## SpeedCP: Fast Kernel-based Conditional Conformal Prediction


### Images

![2ce3c9390781b99697d902252e90b784d5f02a79cfa305b1cb7c2f0c4ce5eff1.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/2ce3c9390781b99697d902252e90b784d5f02a79cfa305b1cb7c2f0c4ce5eff1.jpg)

![33eb8cac2acff46f53986560918eb56b5e178cebb9cf13451d9eac413d11ecee.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/33eb8cac2acff46f53986560918eb56b5e178cebb9cf13451d9eac413d11ecee.jpg)

![4d263dbab07afd81e44875eec47524654d57be867873f0c6eebeaf3de6f15d8b.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/4d263dbab07afd81e44875eec47524654d57be867873f0c6eebeaf3de6f15d8b.jpg)

![55c74457d5d1520df8f574d3b202c014fe058ac8614a317f813cff66452b6492.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/55c74457d5d1520df8f574d3b202c014fe058ac8614a317f813cff66452b6492.jpg)

![5ce0097652618f1999737c1727880b6c643ec23f2d60ba8753bc3001f4cdc652.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/5ce0097652618f1999737c1727880b6c643ec23f2d60ba8753bc3001f4cdc652.jpg)

![5ef9083c9fb4d6e6a14963a0f9288bb795ea4b085f5844222df20bbff2429ebd.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/5ef9083c9fb4d6e6a14963a0f9288bb795ea4b085f5844222df20bbff2429ebd.jpg)

![91e0b8c016ed65404ecf043690ce336351848036387fe6eb4906f4e6282f35c3.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/91e0b8c016ed65404ecf043690ce336351848036387fe6eb4906f4e6282f35c3.jpg)

![9cc1fac45b18e15fa5d89d18af3e86184ca364f77f17596c83960ae3e4385ac7.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/9cc1fac45b18e15fa5d89d18af3e86184ca364f77f17596c83960ae3e4385ac7.jpg)

![9e51cf3a3ecd6912af8f07ae46257879c88b4e0df983da9188e6f3a58b68932f.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/9e51cf3a3ecd6912af8f07ae46257879c88b4e0df983da9188e6f3a58b68932f.jpg)

![b314740aa4eafb86054b994569e582cdde4dd0553f47a5393dea36af41bb4057.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/b314740aa4eafb86054b994569e582cdde4dd0553f47a5393dea36af41bb4057.jpg)

![b4318f1485a27e763e1546fd73a32ee61f7e0f4103d3faf657f5604bddb62e45.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/b4318f1485a27e763e1546fd73a32ee61f7e0f4103d3faf657f5604bddb62e45.jpg)

![bddbeb62394eb707cc613be8ab8162892d8a7ebff30c50f47f81eb5361769ffc.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/bddbeb62394eb707cc613be8ab8162892d8a7ebff30c50f47f81eb5361769ffc.jpg)

![cb55911151f9fa7c46e051f9abb1d528e0414eef0a186b1af0786c9444692573.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/cb55911151f9fa7c46e051f9abb1d528e0414eef0a186b1af0786c9444692573.jpg)

![d13398b3d9db30d0d8155a110b9db8c54fae5ea34ec98c73e5ce3ac14f38a4ef.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/d13398b3d9db30d0d8155a110b9db8c54fae5ea34ec98c73e5ce3ac14f38a4ef.jpg)

![df66b070f9c48fb794d1a962f38f076399f36ca2aa8e6919f3333026d1295680.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/df66b070f9c48fb794d1a962f38f076399f36ca2aa8e6919f3333026d1295680.jpg)

![e21c9f3c4ea19f3f61a77fee5188be9e585b87fb3e741e76d36c6f2e4673efc6.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/e21c9f3c4ea19f3f61a77fee5188be9e585b87fb3e741e76d36c6f2e4673efc6.jpg)

![e3dc5e1c0714cbb983ace0d341d2e810103968dbf31718f0354be627a860cb9a.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/e3dc5e1c0714cbb983ace0d341d2e810103968dbf31718f0354be627a860cb9a.jpg)

![eed8c10a26309de0b8da17d1f4e5c2636a35ccf925db12a2275be9250188d42f.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/images/eed8c10a26309de0b8da17d1f4e5c2636a35ccf925db12a2275be9250188d42f.jpg)

### Tables

![336a12ba4968e4e4f06a6b95d2031a99d6b02fa2c79dba336d8588ac98d737c5.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/336a12ba4968e4e4f06a6b95d2031a99d6b02fa2c79dba336d8588ac98d737c5.jpg)

![36f7902a68b60c13cf1de558ac14afffe2aa870db96e2ef14350b8df1b2cb15a.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/36f7902a68b60c13cf1de558ac14afffe2aa870db96e2ef14350b8df1b2cb15a.jpg)

![4704c9f1375a4a5c8702316f031805a848dcc4a6912ba9950c9c4888343cf507.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/4704c9f1375a4a5c8702316f031805a848dcc4a6912ba9950c9c4888343cf507.jpg)

![5ccb3e70b36d5eb4ff77264a7b837985b1dd2dd2582adefc08beb879674c68e3.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/5ccb3e70b36d5eb4ff77264a7b837985b1dd2dd2582adefc08beb879674c68e3.jpg)

![6c665712fd942b3112d4375edf02f5eb387968e0e1b58f81679bf510717a08eb.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/6c665712fd942b3112d4375edf02f5eb387968e0e1b58f81679bf510717a08eb.jpg)

![74b71762b3794179571e919f201d2d7e131a3e3e9df1049760ce7ecfb70fb40e.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/74b71762b3794179571e919f201d2d7e131a3e3e9df1049760ce7ecfb70fb40e.jpg)

![8a7c3a49bfa10451091bd44450e29ef83fe00b50c2276c60be39f946a9ef8b0d.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/8a7c3a49bfa10451091bd44450e29ef83fe00b50c2276c60be39f946a9ef8b0d.jpg)

![a23fa10865aeedb0f72275f014bd490a61497e96ca5f8b5d580924d2d79cc512.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/a23fa10865aeedb0f72275f014bd490a61497e96ca5f8b5d580924d2d79cc512.jpg)

![bffa7c33d58a5196a65ce3b5268cc94d675ac88f1146f1ef8becf73538cc426b.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/bffa7c33d58a5196a65ce3b5268cc94d675ac88f1146f1ef8becf73538cc426b.jpg)

![cb581affc6ff9d0ed045f2642ac5d021a86b00acbd1ee581edce797f26a64777.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/cb581affc6ff9d0ed045f2642ac5d021a86b00acbd1ee581edce797f26a64777.jpg)

![d541bd53fcfe0020b58a0e3bb49bc2d0b9c8f90993112d4cb3aa6e197868a94c.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/d541bd53fcfe0020b58a0e3bb49bc2d0b9c8f90993112d4cb3aa6e197868a94c.jpg)

![faffecfcad83ab70fd01b0b2f87b6bbb09d423e1f3bc0d567afa9d5fa619e0b2.jpg](../icml_results/3118_SpeedCP_%20Fast%20Kernel-based%20Conditional%20Conformal%20Prediction/tables/faffecfcad83ab70fd01b0b2f87b6bbb09d423e1f3bc0d567afa9d5fa619e0b2.jpg)

<a id="paper-3080"></a>

## DIYHealth Suite: Dataset, Model, and Benchmark for Health Management at Home


### Images

![1600ef984c1693f660c12aa1162ca43405590baf8ba2746bb3edc52de2c52d2e.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/1600ef984c1693f660c12aa1162ca43405590baf8ba2746bb3edc52de2c52d2e.jpg)

![167e65688977b7025949ec3987ca7e1475e5b4a6ad774bbe2b6dac4e2818909d.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/167e65688977b7025949ec3987ca7e1475e5b4a6ad774bbe2b6dac4e2818909d.jpg)

![1ed66a79ab551b789a8fbbc6aa2b8d1d6c2d8ad2f2fd643fb913378bbc0aabba.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/1ed66a79ab551b789a8fbbc6aa2b8d1d6c2d8ad2f2fd643fb913378bbc0aabba.jpg)

![1f59c280c15614c9fe9a84489c767919595d1e2272c3f5e733e0e3bf8b97498b.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/1f59c280c15614c9fe9a84489c767919595d1e2272c3f5e733e0e3bf8b97498b.jpg)

![24a8ebadfb8797f62fad54a7b2b935bad9b9f65b29d9543385949abf2d727820.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/24a8ebadfb8797f62fad54a7b2b935bad9b9f65b29d9543385949abf2d727820.jpg)

![2c279603522b40d8e3601ec1c1cf437450e0e3abcf47a341bfdd4fa7d411ecef.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/2c279603522b40d8e3601ec1c1cf437450e0e3abcf47a341bfdd4fa7d411ecef.jpg)

![317e371e37e4b803f90f477aaa1bd4594075c744388d886fd8ddec3ad3f12232.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/317e371e37e4b803f90f477aaa1bd4594075c744388d886fd8ddec3ad3f12232.jpg)

![34cc43cad519372bd8e8a7cdf31d9fd6ebe599812693c97119d51f9d160b1c16.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/34cc43cad519372bd8e8a7cdf31d9fd6ebe599812693c97119d51f9d160b1c16.jpg)

![3bf2face55adc98f64dc1cb4e7a6a32d6298505335fc9f04245765790a5a68f1.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/3bf2face55adc98f64dc1cb4e7a6a32d6298505335fc9f04245765790a5a68f1.jpg)

![47f6288b53fda9da2e35b85711c6b467f0f1bdde4ec41032c754628884dc2e40.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/47f6288b53fda9da2e35b85711c6b467f0f1bdde4ec41032c754628884dc2e40.jpg)

![4af768fdf4fdd4b0781431b8fc043b7cbfa85213e54d1a59722c63c71b8811b0.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/4af768fdf4fdd4b0781431b8fc043b7cbfa85213e54d1a59722c63c71b8811b0.jpg)

![7d5f11ee8d1e9d039b1178b619ff000a2a4eb13916f8a29b341f37d2dde1db9d.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/7d5f11ee8d1e9d039b1178b619ff000a2a4eb13916f8a29b341f37d2dde1db9d.jpg)

![7eb064d9c5dcbfb396ed3007e5106fe1bad0a25c48fed65fd07039a8eca49ec5.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/7eb064d9c5dcbfb396ed3007e5106fe1bad0a25c48fed65fd07039a8eca49ec5.jpg)

![861d2254cd67b8f6395e9d5cf758ea2fab3fd00e4172ab3c2652a146a34e3776.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/861d2254cd67b8f6395e9d5cf758ea2fab3fd00e4172ab3c2652a146a34e3776.jpg)

![8b6d12d0c5e4ade052613244f928989353b08bb5dbd8e261e009b76096018d08.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/8b6d12d0c5e4ade052613244f928989353b08bb5dbd8e261e009b76096018d08.jpg)

![a4b4774d95f11880ac623b9914d4b0bae6a6a50a8d9d4a82a71765884f602947.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/a4b4774d95f11880ac623b9914d4b0bae6a6a50a8d9d4a82a71765884f602947.jpg)

![ad81f3d4d767ea9e726fb01fa1b7439a5d9c109b385248507c188d978aac6e25.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/ad81f3d4d767ea9e726fb01fa1b7439a5d9c109b385248507c188d978aac6e25.jpg)

![c066da6c9a2de1eda83358633d4c02271f9c8a6d058f28ffefe3f8cc0d83de6e.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/c066da6c9a2de1eda83358633d4c02271f9c8a6d058f28ffefe3f8cc0d83de6e.jpg)

![c129dbbac072ec412ad3684723b6c23f85d409d95550ce3a2b69388d65f214e6.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/c129dbbac072ec412ad3684723b6c23f85d409d95550ce3a2b69388d65f214e6.jpg)

![c4cd7e002b5382e13061a801873db9326d7b3632d25d02d46d9578cb2fbaf682.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/c4cd7e002b5382e13061a801873db9326d7b3632d25d02d46d9578cb2fbaf682.jpg)

![cf907cda3c2865693619d84cf45ed5bb55fa313b67d5cdba371c37d7eb68188d.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/cf907cda3c2865693619d84cf45ed5bb55fa313b67d5cdba371c37d7eb68188d.jpg)

![d193205a7cfd2e151c5e511801a4d92c26eba28d42f4e6448437516b91f6d7fb.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/d193205a7cfd2e151c5e511801a4d92c26eba28d42f4e6448437516b91f6d7fb.jpg)

![db3803ab9bfc084f13d58cccfc5086b8bf2471a7fc6ff211fadde16fc2cc4dfd.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/db3803ab9bfc084f13d58cccfc5086b8bf2471a7fc6ff211fadde16fc2cc4dfd.jpg)

![e73ede97e0189db037c887119f27483f4ee8a382fe601d5d8f0ed5e24570eb4a.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/e73ede97e0189db037c887119f27483f4ee8a382fe601d5d8f0ed5e24570eb4a.jpg)

![ef1948f8add2dba35d3c0a37cc9bb58711a48350b2eab5a31b20549c3b9ed07e.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/ef1948f8add2dba35d3c0a37cc9bb58711a48350b2eab5a31b20549c3b9ed07e.jpg)

![fd1052d344eb44866d9db6a93db1b8f97cfc1396204fb070da539005a30b14e9.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/fd1052d344eb44866d9db6a93db1b8f97cfc1396204fb070da539005a30b14e9.jpg)

![fdbbf34a4b3f3b99e95b59463e2bf987f69e9d1e3804ac78083ef2e2102867f0.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/images/fdbbf34a4b3f3b99e95b59463e2bf987f69e9d1e3804ac78083ef2e2102867f0.jpg)

### Tables

![18dee4e94eab167640462ddfe087e4959549794bfe9cbca771f9688ad29ec860.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/18dee4e94eab167640462ddfe087e4959549794bfe9cbca771f9688ad29ec860.jpg)

![2bbe9c6764152d3319237ccf20687f8f1879331eabcabe25a5270f13a44c7417.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/2bbe9c6764152d3319237ccf20687f8f1879331eabcabe25a5270f13a44c7417.jpg)

![353df0ea7e59865fadd53efcebb5f700921227e403516dfbb481d60e733f3248.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/353df0ea7e59865fadd53efcebb5f700921227e403516dfbb481d60e733f3248.jpg)

![409b6fab8428e4732aa1277fa3d92a6aca5282201411e26622a04c2e8439a072.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/409b6fab8428e4732aa1277fa3d92a6aca5282201411e26622a04c2e8439a072.jpg)

![45db4d5fa993e026eae42cc00eb0d9ab706ccc5a37f265606d150b81ba54fb7e.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/45db4d5fa993e026eae42cc00eb0d9ab706ccc5a37f265606d150b81ba54fb7e.jpg)

![4d084f5fe897a373a6307dafdd2415a1a3637833006835e6c83250e002da9110.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/4d084f5fe897a373a6307dafdd2415a1a3637833006835e6c83250e002da9110.jpg)

![8a9078b5d5e8b500efd8e0963bb33bf0d4d69f560d849d94519c01e1c8f24a8c.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/8a9078b5d5e8b500efd8e0963bb33bf0d4d69f560d849d94519c01e1c8f24a8c.jpg)

![93275d8e764f981d567a6bfdc88a04bdb486757de518a654f04456da643ca377.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/93275d8e764f981d567a6bfdc88a04bdb486757de518a654f04456da643ca377.jpg)

![943216303000f4929c11cafa9cea15324c9d21e9fe57118cf7325e4eeaf52184.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/943216303000f4929c11cafa9cea15324c9d21e9fe57118cf7325e4eeaf52184.jpg)

![9882425e8580933e0aa54e4389602a68d8c914a9c3d92aaae7c20bb4229862ff.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/9882425e8580933e0aa54e4389602a68d8c914a9c3d92aaae7c20bb4229862ff.jpg)

![c13b2765d5763646e0b575088ce49111cb641d2b095aaf77c91f763d66f5a4ac.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/c13b2765d5763646e0b575088ce49111cb641d2b095aaf77c91f763d66f5a4ac.jpg)

![c71efe70de45580e46d7e64b3b6b0147e23619ff009175e3c5161b5298cc2c85.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/c71efe70de45580e46d7e64b3b6b0147e23619ff009175e3c5161b5298cc2c85.jpg)

![e46997d922521a4fe977217ce8ee2aa01e9ec8dd9bc229738278dc0046fbecad.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/e46997d922521a4fe977217ce8ee2aa01e9ec8dd9bc229738278dc0046fbecad.jpg)

![f133ece9c9b3fa2ce0c01146f3a73141043832aeb5816e61aed576a99cbdf476.jpg](../icml_results/3119_DIYHealth%20Suite_%20Dataset%2C%20Model%2C%20and%20Benchmark%20for%20Health%20Management%20at%20Home/tables/f133ece9c9b3fa2ce0c01146f3a73141043832aeb5816e61aed576a99cbdf476.jpg)

<a id="paper-3081"></a>

## Continual GUI Agents


### Images

![01eed3496a8ef093afdf66e0cf30193c36869a4e3139471e0c8e9b9abf74e9b7.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/01eed3496a8ef093afdf66e0cf30193c36869a4e3139471e0c8e9b9abf74e9b7.jpg)

![2e9108a8d774e5cd3a90890cdeee12e8211aa6009781eae69d15bb8755cf7dc9.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/2e9108a8d774e5cd3a90890cdeee12e8211aa6009781eae69d15bb8755cf7dc9.jpg)

![3c99cf39d4788c79de2d84cdd7116fb3d3255552f234dba49bb085bc0cbbfab8.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/3c99cf39d4788c79de2d84cdd7116fb3d3255552f234dba49bb085bc0cbbfab8.jpg)

![3f85f63679cb6532d798959c0726496e104950cffb16c9cc80fa0989dfe47e67.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/3f85f63679cb6532d798959c0726496e104950cffb16c9cc80fa0989dfe47e67.jpg)

![6339c7b8e060c2d1d1724e9190505b84a1d67effab14ee052ba44cf02f4abea4.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/6339c7b8e060c2d1d1724e9190505b84a1d67effab14ee052ba44cf02f4abea4.jpg)

![68604349fcb6b878f4300b16f180903953c4497dfd4627f985d0dff2faa72923.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/68604349fcb6b878f4300b16f180903953c4497dfd4627f985d0dff2faa72923.jpg)

![7fd7800ea32fd70dabe34cad7d2cbbaaa6826e382e86f017960dbde46ef71db2.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/7fd7800ea32fd70dabe34cad7d2cbbaaa6826e382e86f017960dbde46ef71db2.jpg)

![86f0f6433766dab3d530d6116dac4d206af99a6833ad077bfb4d9d6255754d6b.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/86f0f6433766dab3d530d6116dac4d206af99a6833ad077bfb4d9d6255754d6b.jpg)

![9ecac03e1a2fb845853206158a55cfcee7acedd8298a6743ec78372ac31d7076.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/9ecac03e1a2fb845853206158a55cfcee7acedd8298a6743ec78372ac31d7076.jpg)

![b2c89b167926c62cb3e3e8e36d29414e7eff760d28a47377496845283aa53d44.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/b2c89b167926c62cb3e3e8e36d29414e7eff760d28a47377496845283aa53d44.jpg)

![e8f6223e745bb4227285cd50a2de9a75014474a7d9f4d9ebea590c94b9d72eb9.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/e8f6223e745bb4227285cd50a2de9a75014474a7d9f4d9ebea590c94b9d72eb9.jpg)

![ed94558d4aa102da4947848031eb6452c3f94fbb49deeab74845a0836f5d059c.jpg](../icml_results/3120_Continual%20GUI%20Agents/images/ed94558d4aa102da4947848031eb6452c3f94fbb49deeab74845a0836f5d059c.jpg)

### Tables

![6dd1b585b67644979d616f3ac6de12a6531af356118dfab343786f041069085e.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/6dd1b585b67644979d616f3ac6de12a6531af356118dfab343786f041069085e.jpg)

![87ae60727d7a166d9984b8024b4986d5d120e82ffff1437f7ee16c1ff224d30d.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/87ae60727d7a166d9984b8024b4986d5d120e82ffff1437f7ee16c1ff224d30d.jpg)

![88efaa74b44dc4344d11d14bbf599a6604f10c2d7104032400c5662d5d01fb71.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/88efaa74b44dc4344d11d14bbf599a6604f10c2d7104032400c5662d5d01fb71.jpg)

![a93a899b51955ca318d26f7aa5558c6730f3c7bba7fdea8017a975e75d8b000b.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/a93a899b51955ca318d26f7aa5558c6730f3c7bba7fdea8017a975e75d8b000b.jpg)

![cfc1faff472731bb2cd9f4a186af092e31115a0a717ed59c3783cebba0cb970a.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/cfc1faff472731bb2cd9f4a186af092e31115a0a717ed59c3783cebba0cb970a.jpg)

![e5dc3510bd26a268fe9f00956a4e195d43a406a3415896f90c5fc5cdd30565c9.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/e5dc3510bd26a268fe9f00956a4e195d43a406a3415896f90c5fc5cdd30565c9.jpg)

![edf88abcce770ae72794f7379d6ccb9cd581b367c30727938a9d831940935bc0.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/edf88abcce770ae72794f7379d6ccb9cd581b367c30727938a9d831940935bc0.jpg)

![fe8a61c3e3a03feed67fbed0ff349ba611b4e00d899eb644799ec886dacad2e6.jpg](../icml_results/3120_Continual%20GUI%20Agents/tables/fe8a61c3e3a03feed67fbed0ff349ba611b4e00d899eb644799ec886dacad2e6.jpg)

<a id="paper-3082"></a>

## Inverse Depth Scaling From Most Layers Being Similar


### Images

![06f8095b0aefb8336844a1768d4de4df895e6c875361f3e31ff49a0f2f0fe821.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/06f8095b0aefb8336844a1768d4de4df895e6c875361f3e31ff49a0f2f0fe821.jpg)

![080c9bdc9db5ea211e820b98b8328770f80b9186c254ebcaef7c34d13284d4f0.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/080c9bdc9db5ea211e820b98b8328770f80b9186c254ebcaef7c34d13284d4f0.jpg)

![219cfb5b0bc8a4c08e3eb5d57d6ebfeef6767d2d67edab2c52983822ac5b64ba.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/219cfb5b0bc8a4c08e3eb5d57d6ebfeef6767d2d67edab2c52983822ac5b64ba.jpg)

![219d63e6c0593339e964dfb82d21019ed759a7efedae0d8fff09c57853dbcb67.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/219d63e6c0593339e964dfb82d21019ed759a7efedae0d8fff09c57853dbcb67.jpg)

![252ef48a213da4ede8dde2bee50cf1de250ee8d0c190cc7774725d2f3722f67c.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/252ef48a213da4ede8dde2bee50cf1de250ee8d0c190cc7774725d2f3722f67c.jpg)

![291a47cb5f5ea4ba43380f1c3c38e9baacbf3820e1f0d115398f17242ea9ee56.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/291a47cb5f5ea4ba43380f1c3c38e9baacbf3820e1f0d115398f17242ea9ee56.jpg)

![29e03c5aab21bc08e5971dd44737c3adb01113696ff01e3fd080b09e66f15d28.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/29e03c5aab21bc08e5971dd44737c3adb01113696ff01e3fd080b09e66f15d28.jpg)

![306524e4102520fca4fcf5163a5c4d4e9df2cc40f902226c79048b29f2050c17.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/306524e4102520fca4fcf5163a5c4d4e9df2cc40f902226c79048b29f2050c17.jpg)

![485745a07667840e01b19cca22685e974d9dbdb78644498ca51d31a6b13f86fa.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/485745a07667840e01b19cca22685e974d9dbdb78644498ca51d31a6b13f86fa.jpg)

![4bad735039af9e01348a4928b3e30d598cb96dc50175a1e4c62d70753ee75d03.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/4bad735039af9e01348a4928b3e30d598cb96dc50175a1e4c62d70753ee75d03.jpg)

![4ea43341ec3263363be06fcd2a4dfd8ebf667790b70ff168a308456849d6334b.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/4ea43341ec3263363be06fcd2a4dfd8ebf667790b70ff168a308456849d6334b.jpg)

![4ffc28ff757920391914c68246996dc4cdee7209d0a9256b58cde90c7d9ab3b5.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/4ffc28ff757920391914c68246996dc4cdee7209d0a9256b58cde90c7d9ab3b5.jpg)

![562153e7d3357816f9c4d7419fe79126829b2da2c4bfed92e5942ec77de6077c.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/562153e7d3357816f9c4d7419fe79126829b2da2c4bfed92e5942ec77de6077c.jpg)

![660e894b559aee55789de93fe0ab6ad2c8d7950ef57080358fc0cbd8c2e984ff.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/660e894b559aee55789de93fe0ab6ad2c8d7950ef57080358fc0cbd8c2e984ff.jpg)

![6cf7b07f97e95bb53052b726f6efdeb94d460dd429f272f24798ef649984cd56.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/6cf7b07f97e95bb53052b726f6efdeb94d460dd429f272f24798ef649984cd56.jpg)

![7d27adbde81115006c06b11f7934d34dfe2929dccf3b6336e57a48a595451a79.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/7d27adbde81115006c06b11f7934d34dfe2929dccf3b6336e57a48a595451a79.jpg)

![9e9cbedcb0ea176eedc5424c01778ddf4448a1a18aeb28101eea01278a9d729e.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/9e9cbedcb0ea176eedc5424c01778ddf4448a1a18aeb28101eea01278a9d729e.jpg)

![be7dc8c3167b4e8c06467c8bf65bc748cb6b10b2f1e183b7920f8e0e8a9c65e8.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/be7dc8c3167b4e8c06467c8bf65bc748cb6b10b2f1e183b7920f8e0e8a9c65e8.jpg)

![bf8a82a320e4d4c311069efd4596fdd21ef8d099e86c24ded404626c67960e4b.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/bf8a82a320e4d4c311069efd4596fdd21ef8d099e86c24ded404626c67960e4b.jpg)

![c545ea83af9e042715a4c7821ec717363dc627780d0be52e96e72661bb187bea.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/c545ea83af9e042715a4c7821ec717363dc627780d0be52e96e72661bb187bea.jpg)

![cee75a705f9e24723911b620c07ac31c7bb3c31e7edcfbb87053f695f822f60c.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/cee75a705f9e24723911b620c07ac31c7bb3c31e7edcfbb87053f695f822f60c.jpg)

![d23e72c068671314ea3b9d337177a6c9f0697c0c733707afe5d0de4f01c6d6e1.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/d23e72c068671314ea3b9d337177a6c9f0697c0c733707afe5d0de4f01c6d6e1.jpg)

![de8e4571690cc0dfee729671dd037a5340c7853fcb47cc8db21154ce2e326aa7.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/de8e4571690cc0dfee729671dd037a5340c7853fcb47cc8db21154ce2e326aa7.jpg)

![ef8c0e4c85f7345da76d4cfd5de5ad42a4e906fc40b35f37e46d50bec9331242.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/ef8c0e4c85f7345da76d4cfd5de5ad42a4e906fc40b35f37e46d50bec9331242.jpg)

![f1171eb428eb6600cddff1277d34fdade987de433ee990235874e055f4441b6b.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/f1171eb428eb6600cddff1277d34fdade987de433ee990235874e055f4441b6b.jpg)

![f5ccc720851b646e4843daf543834b4c48f84918dd1cff00b56dc7534491af44.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/f5ccc720851b646e4843daf543834b4c48f84918dd1cff00b56dc7534491af44.jpg)

![fa99d616a7d68fcb8b2aa5e1385095037dd5acded2908bf75b2a31c49363381b.jpg](../icml_results/3121_Inverse%20Depth%20Scaling%20From%20Most%20Layers%20Being%20Similar/images/fa99d616a7d68fcb8b2aa5e1385095037dd5acded2908bf75b2a31c49363381b.jpg)

<a id="paper-3083"></a>

## Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic Matrices in Decentralized Learning


### Images

![08bd9d627202dbaf5ac7d1edb8a25dbc76ff95f4c9eed38e9f7aa1c79956940c.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/08bd9d627202dbaf5ac7d1edb8a25dbc76ff95f4c9eed38e9f7aa1c79956940c.jpg)

![5120c7261097db25f0a6377262bb605e3316d360692781520754945a1a4af4a5.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/5120c7261097db25f0a6377262bb605e3316d360692781520754945a1a4af4a5.jpg)

![891f91ec8dbe2f85ed7eff13753c90a878a77663f38240a3ec5f27e65668d4f7.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/891f91ec8dbe2f85ed7eff13753c90a878a77663f38240a3ec5f27e65668d4f7.jpg)

![c83147d385e2119ec316537761f18a5b563cb9b608cad619d3650bea9f778a8c.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/c83147d385e2119ec316537761f18a5b563cb9b608cad619d3650bea9f778a8c.jpg)

![e841ad6e03c2a5fe446a0b87364f61a6095ef3dde16a7d43fac6c8ed52b58135.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/e841ad6e03c2a5fe446a0b87364f61a6095ef3dde16a7d43fac6c8ed52b58135.jpg)

![ea6df7976885655125d2e69bc2c1f57025bcafcacce094c830bb26f18518f602.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/ea6df7976885655125d2e69bc2c1f57025bcafcacce094c830bb26f18518f602.jpg)

![ee97f042172edc7631b8dbd56115041ad476dbff781d7e5ea6147abe861a2a95.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/ee97f042172edc7631b8dbd56115041ad476dbff781d7e5ea6147abe861a2a95.jpg)

![f940a06968803fa2bbd7c1624f90d71a9c84c96c3231c4144d0394cd41860072.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/f940a06968803fa2bbd7c1624f90d71a9c84c96c3231c4144d0394cd41860072.jpg)

![f9c8513fbb7434474e3cc937fe9662bbfe13d00c641f9b8ede7c8d1acb67da87.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/images/f9c8513fbb7434474e3cc937fe9662bbfe13d00c641f9b8ede7c8d1acb67da87.jpg)

### Tables

![371313dbf6b1edf5e04dd963546ad9907602bf6dc6a310ba8492220dad114b6c.jpg](../icml_results/3122_Row-Stochastic%20Matrices%20Can%20Provably%20Outperform%20Doubly%20Stochastic%20Matrices%20in%20Decentralized%20Learning/tables/371313dbf6b1edf5e04dd963546ad9907602bf6dc6a310ba8492220dad114b6c.jpg)

<a id="paper-3084"></a>

## Entropy-aware Span-Constrained Optimal Transport for Robust Cross-Tokenizer Knowledge Distillation


### Images

![00ef76c714d7a6c3e93ff654750848e82c34a34f13d04c558339590e79cb6341.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/00ef76c714d7a6c3e93ff654750848e82c34a34f13d04c558339590e79cb6341.jpg)

![121c1a58eeaebda5d7926dcbdce7e90c91e9caf6b61d532756b0546bf9618c0b.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/121c1a58eeaebda5d7926dcbdce7e90c91e9caf6b61d532756b0546bf9618c0b.jpg)

![123d11fdba918513aaa84243435664a46318c243350939398c38fcf31b3f0f05.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/123d11fdba918513aaa84243435664a46318c243350939398c38fcf31b3f0f05.jpg)

![2bfe8a2fc7bc481dad801f17afc5189dd4943f23d096b37fd5577a3d4cfd7499.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/2bfe8a2fc7bc481dad801f17afc5189dd4943f23d096b37fd5577a3d4cfd7499.jpg)

![2de7b83b16ab0620b982f8fa9fbc50ae8fec92291f87f894edb7f85764d8cccc.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/2de7b83b16ab0620b982f8fa9fbc50ae8fec92291f87f894edb7f85764d8cccc.jpg)

![6fc8ad06f0ef2d0d0713d9e827b02aae2163d7cada91c772a74ad200cc112925.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/images/6fc8ad06f0ef2d0d0713d9e827b02aae2163d7cada91c772a74ad200cc112925.jpg)

### Tables

![191c064868f0f82e96b501dc5111258f63f1f0199f026727e5db6a22cb28b82d.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/191c064868f0f82e96b501dc5111258f63f1f0199f026727e5db6a22cb28b82d.jpg)

![19b45662541821c413458b4935f29a04e08937b8626e82a7192c1ff6372d8b21.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/19b45662541821c413458b4935f29a04e08937b8626e82a7192c1ff6372d8b21.jpg)

![2ef32faec6fd97db8b482e1db7c97f8ad6600ddca78d94674710fb7612727fd4.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/2ef32faec6fd97db8b482e1db7c97f8ad6600ddca78d94674710fb7612727fd4.jpg)

![50e7a63a419eed4c50d89a70f349f81e4e8168494bf24a1f7e122f55da62ab51.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/50e7a63a419eed4c50d89a70f349f81e4e8168494bf24a1f7e122f55da62ab51.jpg)

![6185bf8a09b60e823dcdebf0e954017e19aeb329d9658be53bc58f9ff823d0d5.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/6185bf8a09b60e823dcdebf0e954017e19aeb329d9658be53bc58f9ff823d0d5.jpg)

![b59a9d74e4a23354aa55a02f4b170f41fea5c8d638a04ae99e3c88db8ed7fea5.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/b59a9d74e4a23354aa55a02f4b170f41fea5c8d638a04ae99e3c88db8ed7fea5.jpg)

![d9ebe9d16336f9ae7fd657a907c3da224e3ce4922a197fad1aacc49de901d297.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/d9ebe9d16336f9ae7fd657a907c3da224e3ce4922a197fad1aacc49de901d297.jpg)

![ea6b5fe3a8c524c3cf0078b8844139c2bcb8683006cf73bc1c4c1290971cc1ad.jpg](../icml_results/3123_Entropy-aware%20Span-Constrained%20Optimal%20Transport%20for%20Robust%20Cross-Tokenizer%20Knowledge%20Distillation/tables/ea6b5fe3a8c524c3cf0078b8844139c2bcb8683006cf73bc1c4c1290971cc1ad.jpg)

<a id="paper-3085"></a>

## RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization


### Images

![154eb5407e0a07b71ec1f6632b90390dd80f84ec03209779899de62b74fd6a12.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/154eb5407e0a07b71ec1f6632b90390dd80f84ec03209779899de62b74fd6a12.jpg)

![280cf4618e43edade8fb59ace7a9d2ca384e1c2bde677007c0655df261d1c659.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/280cf4618e43edade8fb59ace7a9d2ca384e1c2bde677007c0655df261d1c659.jpg)

![33ce4c259e87fcb8dc98bac5e5980beba7f3d4fc5de191e2e5f634d91e7814e9.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/33ce4c259e87fcb8dc98bac5e5980beba7f3d4fc5de191e2e5f634d91e7814e9.jpg)

![3c11f184c7724d621e3b0931d71759e531c5e142c3de2eddb7ec31fc5fe92e5e.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/3c11f184c7724d621e3b0931d71759e531c5e142c3de2eddb7ec31fc5fe92e5e.jpg)

![3fc733ec6414239e5d172c19f34425af58e868770df451660edbb27fb642266a.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/3fc733ec6414239e5d172c19f34425af58e868770df451660edbb27fb642266a.jpg)

![5af3c5ebc46ba56201392623e822abca69271324f62776453a5b6eff6c1bcfba.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/5af3c5ebc46ba56201392623e822abca69271324f62776453a5b6eff6c1bcfba.jpg)

![66d41ab6d61a67d70096bedf9df09f81b06ba895828961abf168dd4c737403ba.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/66d41ab6d61a67d70096bedf9df09f81b06ba895828961abf168dd4c737403ba.jpg)

![7ccda22dac6c776ed345df98db6402483e5c63453ce0d652a281fe88511f355a.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/7ccda22dac6c776ed345df98db6402483e5c63453ce0d652a281fe88511f355a.jpg)

![c0911c726939e7a76ea6aecc857a001312693b1d93f455a4ff0252ab90cd7682.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/c0911c726939e7a76ea6aecc857a001312693b1d93f455a4ff0252ab90cd7682.jpg)

![c51cc48ed825804ff57071e9787b2a5d92ba591a75a38778f5a01a051a6c76f0.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/c51cc48ed825804ff57071e9787b2a5d92ba591a75a38778f5a01a051a6c76f0.jpg)

![c86b4f7ace6e476f50b4f717b30c9707d053b8426ed1c490bdf50d2449d003fe.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/c86b4f7ace6e476f50b4f717b30c9707d053b8426ed1c490bdf50d2449d003fe.jpg)

![c9ad34069f3187cca420ddc9fc9c361b813a16ec72f94f74850c457fd8c433f1.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/c9ad34069f3187cca420ddc9fc9c361b813a16ec72f94f74850c457fd8c433f1.jpg)

![d353f5e94353825043555ff67ebcf48426c31c5a44bf959a5a962ef13d94976f.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/d353f5e94353825043555ff67ebcf48426c31c5a44bf959a5a962ef13d94976f.jpg)

![ddc7d16020bf7938bf1767c5239e69b30334f58b7672e7a66207545c3b72fc6b.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/ddc7d16020bf7938bf1767c5239e69b30334f58b7672e7a66207545c3b72fc6b.jpg)

![ec13bfac84226f270626777faa6ea276cfb149e9904a59d157bc5d76e69b86f0.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/ec13bfac84226f270626777faa6ea276cfb149e9904a59d157bc5d76e69b86f0.jpg)

![ff3e75204e9982b4d98c5c250d906bda169ad429ef59be546d6d10472db3e85d.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/images/ff3e75204e9982b4d98c5c250d906bda169ad429ef59be546d6d10472db3e85d.jpg)

### Tables

![1542fe989aa37a0ae361e6e7c033003eee9f0919cc92c7388cb1f335594823ec.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/1542fe989aa37a0ae361e6e7c033003eee9f0919cc92c7388cb1f335594823ec.jpg)

![3cd91c816a83b9610954fa9515d70c62bbcc96b3c9dec2c5fdebe86633c8cd09.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/3cd91c816a83b9610954fa9515d70c62bbcc96b3c9dec2c5fdebe86633c8cd09.jpg)

![60eaf8f6f00d05c49614fa8e24105319698b6ee942092c6edecbda318932420e.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/60eaf8f6f00d05c49614fa8e24105319698b6ee942092c6edecbda318932420e.jpg)

![6d2270757565cd2f1a3e64d454c4a7d9d7e8a19685738ae988b30070487dbdb7.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/6d2270757565cd2f1a3e64d454c4a7d9d7e8a19685738ae988b30070487dbdb7.jpg)

![a0e98b392879a9e1593fdda4548f8e252841ed0e84d730b6d35ad25db721030f.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/a0e98b392879a9e1593fdda4548f8e252841ed0e84d730b6d35ad25db721030f.jpg)

![a4488c7d0d400bc4fb46ec4ab0393dc0f7f97fb31ca0b2f6938053969bce6cd5.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/a4488c7d0d400bc4fb46ec4ab0393dc0f7f97fb31ca0b2f6938053969bce6cd5.jpg)

![aef2cd6de44ae6fdb85eedeab781e2cd99c4c8565c384c855510827040eba7fc.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/aef2cd6de44ae6fdb85eedeab781e2cd99c4c8565c384c855510827040eba7fc.jpg)

![be241ec9335744fe2f26f8c8c5f7e1d957b0ed23c8abd678cf293bbe92b06d5e.jpg](../icml_results/3124_RDT2_%20Exploring%20the%20Scaling%20Limit%20of%20UMI%20Data%20Towards%20Zero-Shot%20Cross-Embodiment%20Generalization/tables/be241ec9335744fe2f26f8c8c5f7e1d957b0ed23c8abd678cf293bbe92b06d5e.jpg)

<a id="paper-3086"></a>

## Toward Effective Multimodal Graph Foundation Model: A Divide-and-Conquer Based Approach


### Images

![41a9454ae7bfa1df332d4adcaa06e64f427d2c6e4116f1aeaa4c85527ba76a60.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/41a9454ae7bfa1df332d4adcaa06e64f427d2c6e4116f1aeaa4c85527ba76a60.jpg)

![7d1f84a2c02f50cfbf87148f6019c8b6da490d2016703f32fb844602c5fe865b.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/7d1f84a2c02f50cfbf87148f6019c8b6da490d2016703f32fb844602c5fe865b.jpg)

![9c30ee4dedfbfd8794b0d118eeec1f22e8948ea54fe91dbcb0be1ef0040e87a1.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/9c30ee4dedfbfd8794b0d118eeec1f22e8948ea54fe91dbcb0be1ef0040e87a1.jpg)

![a3606b03aa94a315994c9efda70ec4ad4125490d02b62ecd7c6eac1f8293953f.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/a3606b03aa94a315994c9efda70ec4ad4125490d02b62ecd7c6eac1f8293953f.jpg)

![ecd8426f3154d171e2fac4edb3b3f58589d11c025785e02adb718c47f616c0d2.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/ecd8426f3154d171e2fac4edb3b3f58589d11c025785e02adb718c47f616c0d2.jpg)

![ed6e9819b5bfcb29e7ce98ba9461c3982ebbd0bc8b7bc4cd7586a4241a708fd6.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/images/ed6e9819b5bfcb29e7ce98ba9461c3982ebbd0bc8b7bc4cd7586a4241a708fd6.jpg)

### Tables

![03810e1590b2d255492125ad71e262e0844b8342ac358f738cd13a652a35f184.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/03810e1590b2d255492125ad71e262e0844b8342ac358f738cd13a652a35f184.jpg)

![421259ae46e0a03727bac082e7e70cb170c60fd16381fc23b9fa94ec3598710a.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/421259ae46e0a03727bac082e7e70cb170c60fd16381fc23b9fa94ec3598710a.jpg)

![4749d0cf9d51025430eed5c8aa23f7c363677a87596651c1156e807a7e5f9d6d.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/4749d0cf9d51025430eed5c8aa23f7c363677a87596651c1156e807a7e5f9d6d.jpg)

![575ebeeb0e3ae136350e4aa1ddfb32391f6d245cb9c253fdfff4d468ff29a365.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/575ebeeb0e3ae136350e4aa1ddfb32391f6d245cb9c253fdfff4d468ff29a365.jpg)

![5bfecf632da271b35af19e901ea6cc833bef4288767752dc0ed556f0d0c457d1.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/5bfecf632da271b35af19e901ea6cc833bef4288767752dc0ed556f0d0c457d1.jpg)

![72b9a601eb35a3ca478f81501262aba74e8a2213a057d1a851154d79791c572d.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/72b9a601eb35a3ca478f81501262aba74e8a2213a057d1a851154d79791c572d.jpg)

![74c4f94accd1ae4a89ba3060301eb14f4bda8e70ef29bc2123f4b2cd403329f0.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/74c4f94accd1ae4a89ba3060301eb14f4bda8e70ef29bc2123f4b2cd403329f0.jpg)

![879bf07ffb4dfa4f7de6cd93e7b548feebcc42d52802bc967e08de6acb25879c.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/879bf07ffb4dfa4f7de6cd93e7b548feebcc42d52802bc967e08de6acb25879c.jpg)

![985b4bebc7aac97e24308f72bd336f0982cb2a90c6a0b873a7a17ebe592ea188.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/985b4bebc7aac97e24308f72bd336f0982cb2a90c6a0b873a7a17ebe592ea188.jpg)

![c0d2a8105822fbf144934de6374d6fa8ec185760a0957d5d8b07877661753640.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/c0d2a8105822fbf144934de6374d6fa8ec185760a0957d5d8b07877661753640.jpg)

![c3b9334fbce49bf720a0a9f3367ba27fe580ebbbe1325abad056c1d73c85703a.jpg](../icml_results/3125_Toward%20Effective%20Multimodal%20Graph%20Foundation%20Model_%20A%20Divide-and-Conquer%20Based%20Approach/tables/c3b9334fbce49bf720a0a9f3367ba27fe580ebbbe1325abad056c1d73c85703a.jpg)

<a id="paper-3087"></a>

## Random Selection Reveals Implicit Knowledge Consensus in Code Generation


### Images

![3be27a721744801bed9d6ad26c0166be6811ec917fe7eb5d0d59958d31bfc0c9.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/images/3be27a721744801bed9d6ad26c0166be6811ec917fe7eb5d0d59958d31bfc0c9.jpg)

![775e22bdc525f0f9dbc65220e1a6e01fbd0da62eb979bd637b9fa42ad7ea089a.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/images/775e22bdc525f0f9dbc65220e1a6e01fbd0da62eb979bd637b9fa42ad7ea089a.jpg)

![90673bb826258e4d8871f545f6e6d0ac24e65bfa2445636ae5ffca0405c08ca2.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/images/90673bb826258e4d8871f545f6e6d0ac24e65bfa2445636ae5ffca0405c08ca2.jpg)

![e1c6cf6994e0e159aa82dc7100b6c5e89ecc74d1a77e65139fc1b0758e7f6027.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/images/e1c6cf6994e0e159aa82dc7100b6c5e89ecc74d1a77e65139fc1b0758e7f6027.jpg)

### Tables

![138fa3cf44d11152c76e5826c5b1ff3587948e97575b7758b2ad8fe1b8942412.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/138fa3cf44d11152c76e5826c5b1ff3587948e97575b7758b2ad8fe1b8942412.jpg)

![2b60ec8add1662ff36d795953829f605afd0bbbca33e081246cca965e22071b7.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/2b60ec8add1662ff36d795953829f605afd0bbbca33e081246cca965e22071b7.jpg)

![300202b5872b61160f0df121d3b8d4e3525b7f0a972d481655a79cacc0e83526.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/300202b5872b61160f0df121d3b8d4e3525b7f0a972d481655a79cacc0e83526.jpg)

![474e2346183f07ca29aae7f2135801739b36001b89b0175b24323898145f7a64.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/474e2346183f07ca29aae7f2135801739b36001b89b0175b24323898145f7a64.jpg)

![576ce5065ebe291c6db0d4d7e8586172919d198e67a482316a7479be3e9b2160.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/576ce5065ebe291c6db0d4d7e8586172919d198e67a482316a7479be3e9b2160.jpg)

![636bcf8b7df5e7e74f4f0fd6534c1dd1aed6807532544843e14adaea55adb0bc.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/636bcf8b7df5e7e74f4f0fd6534c1dd1aed6807532544843e14adaea55adb0bc.jpg)

![651adc59c4a643da2cb6064f720df9d3dc6fff912262230c574cc37679357832.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/651adc59c4a643da2cb6064f720df9d3dc6fff912262230c574cc37679357832.jpg)

![72b576703ac1106c86b452bfb25bfc732e9ff6b600f813aa0f10799ca3c0721d.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/72b576703ac1106c86b452bfb25bfc732e9ff6b600f813aa0f10799ca3c0721d.jpg)

![74f996396c520ac1296fc7ff25f9bb1faf37d07625f35cda8c6a14821b55dad8.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/74f996396c520ac1296fc7ff25f9bb1faf37d07625f35cda8c6a14821b55dad8.jpg)

![898aad7bc3b16738ba2ed163552a70ce85810615dbe2366d27b0452b7e561113.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/898aad7bc3b16738ba2ed163552a70ce85810615dbe2366d27b0452b7e561113.jpg)

![8c241d72e96f4ea20d6568257f8b9ed58a7fc22263b248f4a72cddbb91274f29.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/8c241d72e96f4ea20d6568257f8b9ed58a7fc22263b248f4a72cddbb91274f29.jpg)

![9f8fb27d6fe450cb8912a9f6d928d08c21fdf36453761706e57738271bfd5f85.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/9f8fb27d6fe450cb8912a9f6d928d08c21fdf36453761706e57738271bfd5f85.jpg)

![a27f538a03e256c2c055a27da74e09d6133dc34779b0841860a0c6b287efda88.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/a27f538a03e256c2c055a27da74e09d6133dc34779b0841860a0c6b287efda88.jpg)

![c60522e488ab44b57a68ca310109eb6803ca0e494dc293c65ce026bb35a157d6.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/c60522e488ab44b57a68ca310109eb6803ca0e494dc293c65ce026bb35a157d6.jpg)

![d040af9a99a93be08826ac66ae30e46935af1121786b22a2c57387c31dd220b8.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/d040af9a99a93be08826ac66ae30e46935af1121786b22a2c57387c31dd220b8.jpg)

![df1ff9eebf7e9c9d933c625aa78cac959c54a73443fbb53c87d9f5642fe827e0.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/df1ff9eebf7e9c9d933c625aa78cac959c54a73443fbb53c87d9f5642fe827e0.jpg)

![f543d00a3662062b4ac245dfefd8a13384b77b8361c2a227b655e4d790d8f6a2.jpg](../icml_results/3126_Random%20Selection%20Reveals%20Implicit%20Knowledge%20Consensus%20in%20Code%20Generation/tables/f543d00a3662062b4ac245dfefd8a13384b77b8361c2a227b655e4d790d8f6a2.jpg)

<a id="paper-3088"></a>

## Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services


### Images

![53fd03cced9620901aa515c712f7b476d25e1a09f5fb213755c8fca665256ce7.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/53fd03cced9620901aa515c712f7b476d25e1a09f5fb213755c8fca665256ce7.jpg)

![552088b60c33d0aac29c9a9319758971ba3276eb1214be839e9ac32c21d85126.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/552088b60c33d0aac29c9a9319758971ba3276eb1214be839e9ac32c21d85126.jpg)

![83219af0e975e04b66ba001cf851eec13bc2effcc7ee1706617287c80f5c6908.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/83219af0e975e04b66ba001cf851eec13bc2effcc7ee1706617287c80f5c6908.jpg)

![947f3d2518a42d24a2fa547d627511be6089e48ad5c130eb3f6aab32dff75cf9.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/947f3d2518a42d24a2fa547d627511be6089e48ad5c130eb3f6aab32dff75cf9.jpg)

![bfc2058037e6fa839aec132229037d8570478a0bd6a54bf276e0c2e4ca48a965.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/bfc2058037e6fa839aec132229037d8570478a0bd6a54bf276e0c2e4ca48a965.jpg)

![da5fb40c4cfa0683f1bbb7c62651fd13615b2edb8b83fb1bbd83d5438f71b6ac.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/da5fb40c4cfa0683f1bbb7c62651fd13615b2edb8b83fb1bbd83d5438f71b6ac.jpg)

![daee8bb5aa95ff577c344b42be8a879a3140a06c7801ffbb266255ba93286b15.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/daee8bb5aa95ff577c344b42be8a879a3140a06c7801ffbb266255ba93286b15.jpg)

![f756f7b71439495c8634d8f2df8f5701477f9d0f581aa1509c3b3d9141408173.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/images/f756f7b71439495c8634d8f2df8f5701477f9d0f581aa1509c3b3d9141408173.jpg)

### Tables

![4a5920e55eb41c820d6cc147de084d24bf513ec85fd32230aa185abe80d4433c.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/tables/4a5920e55eb41c820d6cc147de084d24bf513ec85fd32230aa185abe80d4433c.jpg)

![6db58833f93ad9cd81b8e8f7b19fd19099ad50bf7ed3e127ffbaeb86bf868651.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/tables/6db58833f93ad9cd81b8e8f7b19fd19099ad50bf7ed3e127ffbaeb86bf868651.jpg)

![6e3240ed4e61005278c56951f770d8847badf7dfc3265fcca456436c354df5ef.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/tables/6e3240ed4e61005278c56951f770d8847badf7dfc3265fcca456436c354df5ef.jpg)

![714ffb36dd9005b7fc94cfe7edd503cd33f9c3bfbccac2bdc234f87f5504ac13.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/tables/714ffb36dd9005b7fc94cfe7edd503cd33f9c3bfbccac2bdc234f87f5504ac13.jpg)

![f83344bf3d09c432563993d6f1dc60c77d75a697224b34f2c71d82fc29a78465.jpg](../icml_results/3127_Beyond%20Static%20Endpoints_%20Tool%20Programs%20as%20an%20Interface%20for%20Flexible%20Agentic%20Web%20Services/tables/f83344bf3d09c432563993d6f1dc60c77d75a697224b34f2c71d82fc29a78465.jpg)

<a id="paper-3089"></a>

## Geometric Rate–Distortion Invariance for Domain Generalization


### Images

![1bef96e321d804a1fa7fe70180d368578117b9a3619ec664c50554d016349a50.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/1bef96e321d804a1fa7fe70180d368578117b9a3619ec664c50554d016349a50.jpg)

![2c699cf4e4c2624f72a6086dfa710f9579c893fa042d79fe8d29edffebecd687.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/2c699cf4e4c2624f72a6086dfa710f9579c893fa042d79fe8d29edffebecd687.jpg)

![554b025ad57071074745fe34292f66a165061e6d0774165e786295e737047bb7.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/554b025ad57071074745fe34292f66a165061e6d0774165e786295e737047bb7.jpg)

![57b6527a36ea9b525ea8764c5a6f0efc4ccecf6017b8d3318290609d6ac4f70e.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/57b6527a36ea9b525ea8764c5a6f0efc4ccecf6017b8d3318290609d6ac4f70e.jpg)

![6e9116a4e2ee82ba2ddcc13f45966a0881fc509e69826e278f34bbb82e367006.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/6e9116a4e2ee82ba2ddcc13f45966a0881fc509e69826e278f34bbb82e367006.jpg)

![76c96d65347a4bdd15b8eb9ca787aac7e0107356570dde65e6097ece5cd9ab93.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/76c96d65347a4bdd15b8eb9ca787aac7e0107356570dde65e6097ece5cd9ab93.jpg)

![80d6c81948fbdc9c9849b3c89df922b944521d0c96922bde49d056580918c7bf.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/80d6c81948fbdc9c9849b3c89df922b944521d0c96922bde49d056580918c7bf.jpg)

![82160e6d943066672e0a5ace3ed87df2cd6db5807f86370b790f7852b442dfc8.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/82160e6d943066672e0a5ace3ed87df2cd6db5807f86370b790f7852b442dfc8.jpg)

![b4e92fb26c419eb50bf6ce834b96b61b3e54f1703d73c9f9ae3cb248ad24f29c.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/b4e92fb26c419eb50bf6ce834b96b61b3e54f1703d73c9f9ae3cb248ad24f29c.jpg)

![e7b941e9d63b9362822ea79c772fe92db11cd86b1a6e909ae0a268846dd7c643.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/images/e7b941e9d63b9362822ea79c772fe92db11cd86b1a6e909ae0a268846dd7c643.jpg)

### Tables

![1067ed8e52fa91f19f5312871e0041aa4ade2db69522ef0852b24a3b6c4c3810.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/1067ed8e52fa91f19f5312871e0041aa4ade2db69522ef0852b24a3b6c4c3810.jpg)

![13b6683382d8d044f43fd8bc477336d92cce2a7b5aafb2d2151d8040109cad2d.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/13b6683382d8d044f43fd8bc477336d92cce2a7b5aafb2d2151d8040109cad2d.jpg)

![15c601e951581d79c64d156d32ac6385595baaaba62349288cb12e4a4f0dcf8b.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/15c601e951581d79c64d156d32ac6385595baaaba62349288cb12e4a4f0dcf8b.jpg)

![38b1697f6f710ecb6119c5160dafc518172acc55a5b866f14b162f177b1fe071.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/38b1697f6f710ecb6119c5160dafc518172acc55a5b866f14b162f177b1fe071.jpg)

![3e0402098098e079f4b876991b3a749a580186e3337fd6524bfbd15337d3e2e8.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/3e0402098098e079f4b876991b3a749a580186e3337fd6524bfbd15337d3e2e8.jpg)

![42564d7740782aa847b0d1edc39808093480befbe4e78e72f8816a7b37706738.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/42564d7740782aa847b0d1edc39808093480befbe4e78e72f8816a7b37706738.jpg)

![503398c5df0c7476f3242677f717b55802aec0787c743f46c5d7f5233f009d99.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/503398c5df0c7476f3242677f717b55802aec0787c743f46c5d7f5233f009d99.jpg)

![5410aeb9219592fd0e625d833468a38003c454f5def57090de8dc4d38eb07f8a.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/5410aeb9219592fd0e625d833468a38003c454f5def57090de8dc4d38eb07f8a.jpg)

![56afea9a9811765cafda92317dcdb1d258272068bb4cbfa8629b6bf41e2302a3.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/56afea9a9811765cafda92317dcdb1d258272068bb4cbfa8629b6bf41e2302a3.jpg)

![57a07c45d49565590bad51924c4e2bb1f4ec1355323b4c5abf1596a93e1e0235.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/57a07c45d49565590bad51924c4e2bb1f4ec1355323b4c5abf1596a93e1e0235.jpg)

![667f1390ef92785918d6203fa619a172d4460c93c4a48ce3563e58fbf14a7244.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/667f1390ef92785918d6203fa619a172d4460c93c4a48ce3563e58fbf14a7244.jpg)

![66a2572aeb7425822c3ccf4560c5d450399aac6d4f30accf70961a31388ccaac.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/66a2572aeb7425822c3ccf4560c5d450399aac6d4f30accf70961a31388ccaac.jpg)

![6bebc751b6d96476b2e53e32ccf966bee33f330a2193add3a3ab2ba4826f7f9f.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/6bebc751b6d96476b2e53e32ccf966bee33f330a2193add3a3ab2ba4826f7f9f.jpg)

![75f740c0df551d95f849a5cb830201addac571474da8db1c246b60750f2fe18f.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/75f740c0df551d95f849a5cb830201addac571474da8db1c246b60750f2fe18f.jpg)

![95885223278f3f3343e75ee412d83a3b31a90551c408d4ebd68c070c931eba8e.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/95885223278f3f3343e75ee412d83a3b31a90551c408d4ebd68c070c931eba8e.jpg)

![9b052a02aa7b291d92b8b9f0f23ff4cc80ab0350394e4a8b02033d0648369a0a.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/9b052a02aa7b291d92b8b9f0f23ff4cc80ab0350394e4a8b02033d0648369a0a.jpg)

![a539e97a59949ef93395602cdaae968c4478579520d1f07d676b53c4db79b5ce.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/a539e97a59949ef93395602cdaae968c4478579520d1f07d676b53c4db79b5ce.jpg)

![a55adb360cc849b8bf3f6413770a0684a3d992575953c2e03ff15c6568a63fb3.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/a55adb360cc849b8bf3f6413770a0684a3d992575953c2e03ff15c6568a63fb3.jpg)

![ae532382001b6f2b19aad67c1626a2aeaf09951147a67ca3e21cc82f5a2db441.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/ae532382001b6f2b19aad67c1626a2aeaf09951147a67ca3e21cc82f5a2db441.jpg)

![b955ad892ee059888ed1392eae7c05ff9f0aecd021d6da129d79f564b1111970.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/b955ad892ee059888ed1392eae7c05ff9f0aecd021d6da129d79f564b1111970.jpg)

![be0a7cb908c619725cfed51a0ffa917a7aedff56a0e1b391cda33e6feac891a7.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/be0a7cb908c619725cfed51a0ffa917a7aedff56a0e1b391cda33e6feac891a7.jpg)

![c41fd085c801714d27363b1ade3baaf920f3235cf82abcb9cb2b90b6b8612a4b.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/c41fd085c801714d27363b1ade3baaf920f3235cf82abcb9cb2b90b6b8612a4b.jpg)

![cc09ad61cedbc08441c1cd3dad723f5629b8713114df42116509684f5b7db15b.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/cc09ad61cedbc08441c1cd3dad723f5629b8713114df42116509684f5b7db15b.jpg)

![e66c3d76d354b9891bf9721b053c382ec56f0c7388adf9aa737e4334fdbb4fe9.jpg](../icml_results/3128_Geometric%20Rate%E2%80%93Distortion%20Invariance%20for%20Domain%20Generalization/tables/e66c3d76d354b9891bf9721b053c382ec56f0c7388adf9aa737e4334fdbb4fe9.jpg)

<a id="paper-3090"></a>

## Hom-PGD$^+$: Fast Reparameterized Optimization over Non-convex Ball-Homeomorphic Set


### Images

![09e27af1a66891f1ad32234ecaf22a6bd3c9739a7c5c45442922913a13208528.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/09e27af1a66891f1ad32234ecaf22a6bd3c9739a7c5c45442922913a13208528.jpg)

![128e9d70047a2763f2c9f9e4e10e477b54aa990d7737a2df02a9e6cafa498ce0.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/128e9d70047a2763f2c9f9e4e10e477b54aa990d7737a2df02a9e6cafa498ce0.jpg)

![1f6946812e167bb73eb55c8bcdf4473d184f66b3e94c6b20c6124c426b121981.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/1f6946812e167bb73eb55c8bcdf4473d184f66b3e94c6b20c6124c426b121981.jpg)

![8c37730bee3327083255f97d2c587a0bd2f9e94d77d801d86d8e0d4a663a6019.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/8c37730bee3327083255f97d2c587a0bd2f9e94d77d801d86d8e0d4a663a6019.jpg)

![a103f0146ff01654d29dd90ca6e9b98fd09fb5b8423b0537cb6716c2712c24f1.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/a103f0146ff01654d29dd90ca6e9b98fd09fb5b8423b0537cb6716c2712c24f1.jpg)

![ae383e0eeace27aa5823c10e5994966fd1788aa7533e7f5f9bc574b960cd4234.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/ae383e0eeace27aa5823c10e5994966fd1788aa7533e7f5f9bc574b960cd4234.jpg)

![b84f8b4ba4cc12c64373239533d52fa68adc8e4e0b07bb260ca8fcdc1217567c.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/b84f8b4ba4cc12c64373239533d52fa68adc8e4e0b07bb260ca8fcdc1217567c.jpg)

![e9acae5b408cb6ed60a82fc1c0958832bb814bfb0c1125cda2ae80244a54391f.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/e9acae5b408cb6ed60a82fc1c0958832bb814bfb0c1125cda2ae80244a54391f.jpg)

![f0ac254f5eb462100dc4aa1b9983180e618048cc104549ccddba794566d9cbb8.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/f0ac254f5eb462100dc4aa1b9983180e618048cc104549ccddba794566d9cbb8.jpg)

![f51dc42d1595f467c3b07b94c2c51198dd120dac07f3d0b319c437af0a60bdd6.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/f51dc42d1595f467c3b07b94c2c51198dd120dac07f3d0b319c437af0a60bdd6.jpg)

![fcdf948c34752be1fdb10e85a014ee22e3749a7dd13eaab2065da3365c8252fc.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/fcdf948c34752be1fdb10e85a014ee22e3749a7dd13eaab2065da3365c8252fc.jpg)

![fe812f66dc4eb84da15b614fb06fdf8698cbabc531033639f187d588a7eff66b.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/images/fe812f66dc4eb84da15b614fb06fdf8698cbabc531033639f187d588a7eff66b.jpg)

### Tables

![3923bb4d603f3e7381b02e3267b4248bff47b4615ba345831ff19a7c8ffcc37d.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/3923bb4d603f3e7381b02e3267b4248bff47b4615ba345831ff19a7c8ffcc37d.jpg)

![5064701f0ec7a1c324ca95b08657a880b917337546e48bde649ced493242b3ea.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/5064701f0ec7a1c324ca95b08657a880b917337546e48bde649ced493242b3ea.jpg)

![5d9b3c71a58588eb62aca8fb2308441cdb9f8872c3ae5d28bd0b65ea573653ab.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/5d9b3c71a58588eb62aca8fb2308441cdb9f8872c3ae5d28bd0b65ea573653ab.jpg)

![6af2cf1c338c0dae57f0e7390f7d07519a48d77c7dcaf03ec80cf545346682e6.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/6af2cf1c338c0dae57f0e7390f7d07519a48d77c7dcaf03ec80cf545346682e6.jpg)

![73032e7b2191a23f64c724b83f07984a860ddf7f3e07df826d78da6edfb1c926.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/73032e7b2191a23f64c724b83f07984a860ddf7f3e07df826d78da6edfb1c926.jpg)

![b9f9e3cd169d1fa70e5a4644fb5c98b24c82cdca555510919c606783caab1f56.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/b9f9e3cd169d1fa70e5a4644fb5c98b24c82cdca555510919c606783caab1f56.jpg)

![f27e2643a0484f73adc9f4b20317052f577df23b8ed96e9239c8425b3574303b.jpg](../icml_results/3129_Hom-PGD%24%5E%2B%24_%20Fast%20Reparameterized%20Optimization%20over%20Non-convex%20Ball-Homeomorphic%20Set/tables/f27e2643a0484f73adc9f4b20317052f577df23b8ed96e9239c8425b3574303b.jpg)

<a id="paper-3091"></a>

## Geometric Pocket-Centric Protein Encoding for Polypharmacology-Guided Multi-Target Drug Design


### Images

![0dcfe0b8f0c28fe19024bf5af7e1566921bce0c4a14a112515b84d8ee419033e.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/0dcfe0b8f0c28fe19024bf5af7e1566921bce0c4a14a112515b84d8ee419033e.jpg)

![28cbedc47c0858f14343fc15404481baa3e96b75d88369d1f2f853a2caaefc2a.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/28cbedc47c0858f14343fc15404481baa3e96b75d88369d1f2f853a2caaefc2a.jpg)

![5041345945132aa1dbaf9f44fddbfc9d154e4172b07693c399ee02e69ba15684.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/5041345945132aa1dbaf9f44fddbfc9d154e4172b07693c399ee02e69ba15684.jpg)

![5b80bbf4f7a9978ff7da2c62ae3f185273409e023b522553a987025b2127a689.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/5b80bbf4f7a9978ff7da2c62ae3f185273409e023b522553a987025b2127a689.jpg)

![5e25d166d4a08121e240e96e258f302f6eb10fe0fa9296890d85e424396c4944.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/5e25d166d4a08121e240e96e258f302f6eb10fe0fa9296890d85e424396c4944.jpg)

![6353f1791ac8678c15e706735a6f8b2395d4beb3613145ea63c32bc5e448a069.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/6353f1791ac8678c15e706735a6f8b2395d4beb3613145ea63c32bc5e448a069.jpg)

![67899844dae4bb994d6ad81d60547313d09b1a46850fb50c7ddb69f43258ab25.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/67899844dae4bb994d6ad81d60547313d09b1a46850fb50c7ddb69f43258ab25.jpg)

![97ff4444b0bd50874e4c202d71bc916176ee072775b14abb8afdfea84a7ae530.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/97ff4444b0bd50874e4c202d71bc916176ee072775b14abb8afdfea84a7ae530.jpg)

![a86650394f983695c50fb5e2886cdfec2882c490ad203f8887206737e32ad935.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/a86650394f983695c50fb5e2886cdfec2882c490ad203f8887206737e32ad935.jpg)

![f4fc09bc5df38f4bf73cd2aa68b0b6023e31bb203fca11a6230accc7b745c17b.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/images/f4fc09bc5df38f4bf73cd2aa68b0b6023e31bb203fca11a6230accc7b745c17b.jpg)

### Tables

![02f814809d2349e37c4deaa1a5a72fe8861c1047f1cd309602d2aa0e70d79227.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/tables/02f814809d2349e37c4deaa1a5a72fe8861c1047f1cd309602d2aa0e70d79227.jpg)

![5f6e394225eb703e424a1baaf1faf5ed4c73f51baae1879a56fe0a3a8b6bc725.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/tables/5f6e394225eb703e424a1baaf1faf5ed4c73f51baae1879a56fe0a3a8b6bc725.jpg)

![8721d3ede144fc2047b8e7526c46844a48b07da81b489ac717cbfc002db8c467.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/tables/8721d3ede144fc2047b8e7526c46844a48b07da81b489ac717cbfc002db8c467.jpg)

![a58a1665745624d9a934bac389ee0c9f742712859986f90099337b29ae4e31d2.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/tables/a58a1665745624d9a934bac389ee0c9f742712859986f90099337b29ae4e31d2.jpg)

![c0a760db0cd66cd6dd4bec44284568840c795fd5c283c90fc7635b6afad40292.jpg](../icml_results/3130_Geometric%20Pocket-Centric%20Protein%20Encoding%20for%20Polypharmacology-Guided%20Multi-Target%20Drug%20Design/tables/c0a760db0cd66cd6dd4bec44284568840c795fd5c283c90fc7635b6afad40292.jpg)

<a id="paper-3092"></a>

## HSGG: Training-Free Hierarchical Scene Graph Generation with Geometry-Guided Relation Reasoning


### Images

![00c3ecdb3d35021f58ab5a327278b8ad493453918f6a7c0d3e05124a84523aee.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/00c3ecdb3d35021f58ab5a327278b8ad493453918f6a7c0d3e05124a84523aee.jpg)

![0f99f632183054deccf08a590f5fb95deb7248203bd73e6702529bfe348a873b.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/0f99f632183054deccf08a590f5fb95deb7248203bd73e6702529bfe348a873b.jpg)

![4d7e0112b225c45ab72510f161394b148040ce153e423a92cefaa95d2fdb0746.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/4d7e0112b225c45ab72510f161394b148040ce153e423a92cefaa95d2fdb0746.jpg)

![62b768fe9e525ad72bfaaf9aa44a756f989582e4b7a5c8e1657549a5e1969518.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/62b768fe9e525ad72bfaaf9aa44a756f989582e4b7a5c8e1657549a5e1969518.jpg)

![6f0f1d5bbbdd7ad53171115708c947f2a1ff25ce80d913b972cac9b7a341e739.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/6f0f1d5bbbdd7ad53171115708c947f2a1ff25ce80d913b972cac9b7a341e739.jpg)

![8822bea1fa301404947d1c79a37b446373652b83f3a3dc60c733b0c1eda189b3.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/8822bea1fa301404947d1c79a37b446373652b83f3a3dc60c733b0c1eda189b3.jpg)

![9c2fee493708cefceaaefc5bd7c7b075e3460b8959afacfba2f9e0e2debfd438.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/9c2fee493708cefceaaefc5bd7c7b075e3460b8959afacfba2f9e0e2debfd438.jpg)

![d136a3d4a6fd236742c0ef79b40b4843ff377f22e20d16662698ed029b5196c6.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/d136a3d4a6fd236742c0ef79b40b4843ff377f22e20d16662698ed029b5196c6.jpg)

![e7ce6d775e7b57a44922a64728ae4d84e7f8cc3ac74e57a12c9d16dd1045e484.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/images/e7ce6d775e7b57a44922a64728ae4d84e7f8cc3ac74e57a12c9d16dd1045e484.jpg)

### Tables

![0214dd1c343924c563c05fc1c43986a054579457ee893578e8b3c3ca6e0bfa98.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/0214dd1c343924c563c05fc1c43986a054579457ee893578e8b3c3ca6e0bfa98.jpg)

![0a709a407592d88e85b01668e95c0e79195dd1579dde8cda5823b16de78e7ea3.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/0a709a407592d88e85b01668e95c0e79195dd1579dde8cda5823b16de78e7ea3.jpg)

![23829aaf60ce15cae33f3ce2e9199bb7193a5072ffff5043a020d772f6fb1987.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/23829aaf60ce15cae33f3ce2e9199bb7193a5072ffff5043a020d772f6fb1987.jpg)

![4df357473eaf5de2ed173eb1ddc80acc9b6770677b9acceefd3a5bc0d72e4a87.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/4df357473eaf5de2ed173eb1ddc80acc9b6770677b9acceefd3a5bc0d72e4a87.jpg)

![4eef66022ea93b6c68ea6171ef7de370d87d568bc74a7f5280cd1e81af500d81.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/4eef66022ea93b6c68ea6171ef7de370d87d568bc74a7f5280cd1e81af500d81.jpg)

![63ef4e9f299156c20f9ffa0621e5644a99b2f0c902c3683146d4706e85787d94.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/63ef4e9f299156c20f9ffa0621e5644a99b2f0c902c3683146d4706e85787d94.jpg)

![86a3d85159836d784485ae1babce511df4cf8e00f65cd84dbb3fdd626cd4a4b9.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/86a3d85159836d784485ae1babce511df4cf8e00f65cd84dbb3fdd626cd4a4b9.jpg)

![9f25425def588f39e627e73bfebd92a7e545c085264ac7b0b40a8ffa73178d99.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/9f25425def588f39e627e73bfebd92a7e545c085264ac7b0b40a8ffa73178d99.jpg)

![a0b9e0a945f6b407fc1bd280a7d8689c8700f205ead38b8adb3524f01e5ea64b.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/a0b9e0a945f6b407fc1bd280a7d8689c8700f205ead38b8adb3524f01e5ea64b.jpg)

![acb0100a7df48dc6e0e93e1b337f016ea9814f1f17519164b541af9b631fcc9a.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/acb0100a7df48dc6e0e93e1b337f016ea9814f1f17519164b541af9b631fcc9a.jpg)

![c456a2d508ef4cfc2d74a9103ba49e780047faa250385e2ed4bcd86f47a61547.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/c456a2d508ef4cfc2d74a9103ba49e780047faa250385e2ed4bcd86f47a61547.jpg)

![fd1654975141140afdac0216bf490fccf8f288d72ac097248e4d6246a7931f10.jpg](../icml_results/3131_HSGG_%20Training-Free%20Hierarchical%20Scene%20Graph%20Generation%20with%20Geometry-Guided%20Relation%20Reasoning/tables/fd1654975141140afdac0216bf490fccf8f288d72ac097248e4d6246a7931f10.jpg)

<a id="paper-3093"></a>

## CAST: Modeling Visual State Transitions for Consistent Video Retrieval


### Images

![06428d55651c67dc6f3fbffb5e631725c54998309ed76cf2d208d5216fbd5944.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/06428d55651c67dc6f3fbffb5e631725c54998309ed76cf2d208d5216fbd5944.jpg)

![2f468861fe08d1131bdd98f991d9fa73663ab92a92cd7b4eb23b8dd483910708.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/2f468861fe08d1131bdd98f991d9fa73663ab92a92cd7b4eb23b8dd483910708.jpg)

![32d28cf14007193059c721641c8f29d2c2f62bdb9709c16378961d4f1ae05a0c.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/32d28cf14007193059c721641c8f29d2c2f62bdb9709c16378961d4f1ae05a0c.jpg)

![3e98c776655410e6293dbb80e455ab17f593fa95c34e044e5026bd9e3c5a098e.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/3e98c776655410e6293dbb80e455ab17f593fa95c34e044e5026bd9e3c5a098e.jpg)

![42c76dd9142a5b501eae8f7d880c272a9aa79c3a2756cbf178f7e1f9c46d2d3a.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/42c76dd9142a5b501eae8f7d880c272a9aa79c3a2756cbf178f7e1f9c46d2d3a.jpg)

![bb8876d7330085dcebf0178a2f859bd8c04a374081d60a590055c1e8a4f64dac.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/bb8876d7330085dcebf0178a2f859bd8c04a374081d60a590055c1e8a4f64dac.jpg)

![e6b64416324efb06e834dd474ed9e1de47a3513d5dcdde934b363860b4f0bd14.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/e6b64416324efb06e834dd474ed9e1de47a3513d5dcdde934b363860b4f0bd14.jpg)

![fe6fffd348004294426e236253c61bc5d2f691e34b604dc2e8ed6ec6b4e5bfbb.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/images/fe6fffd348004294426e236253c61bc5d2f691e34b604dc2e8ed6ec6b4e5bfbb.jpg)

### Tables

![11c13377fe434d96839dc80f6cddfe70aa7300060ec175523786b1dd2585840c.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/11c13377fe434d96839dc80f6cddfe70aa7300060ec175523786b1dd2585840c.jpg)

![189fd6f63b8865b40cdce3d77c2ddaea65e0db716cb2b31d8bc9fc8331a32282.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/189fd6f63b8865b40cdce3d77c2ddaea65e0db716cb2b31d8bc9fc8331a32282.jpg)

![2b6e5ebdfc0652897f278db4a0b31ead7c4a322988ccc175d40080c3876d3e18.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/2b6e5ebdfc0652897f278db4a0b31ead7c4a322988ccc175d40080c3876d3e18.jpg)

![2cd5cb4063e5d4ebceb3535a68504820a991131d1dcc5dd175137baf02743553.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/2cd5cb4063e5d4ebceb3535a68504820a991131d1dcc5dd175137baf02743553.jpg)

![4069f3f2d65c998cdaefb4442bc22eacce41b4d2acf425a6d6a8d00d3806f6a1.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/4069f3f2d65c998cdaefb4442bc22eacce41b4d2acf425a6d6a8d00d3806f6a1.jpg)

![70eea1014c70a611edf5ec718530eb523cdf2d76ea5edcb8eb62a6648bd9c88c.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/70eea1014c70a611edf5ec718530eb523cdf2d76ea5edcb8eb62a6648bd9c88c.jpg)

![7ba17efec10d52da18d15d3748eb30fa27ed693dc13698f2905100f106d89adb.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/7ba17efec10d52da18d15d3748eb30fa27ed693dc13698f2905100f106d89adb.jpg)

![884d159d12e56e3977c06068f34d1a7fb16e213c3766321bcddfd1123eba04b3.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/884d159d12e56e3977c06068f34d1a7fb16e213c3766321bcddfd1123eba04b3.jpg)

![937e7d50c367ed1020002c3b1d06ccd7c7de39b0d5da8310e20b2b66826911a8.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/937e7d50c367ed1020002c3b1d06ccd7c7de39b0d5da8310e20b2b66826911a8.jpg)

![b39d3db1de0176eb606d34fb703c569d1796e748d373fb8e100da144f6d9cd30.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/b39d3db1de0176eb606d34fb703c569d1796e748d373fb8e100da144f6d9cd30.jpg)

![ca989200d6b6e7ed1fdbb562953baf1f750b2e6398c3a25e9b42bf1ff718e11e.jpg](../icml_results/3132_CAST_%20Modeling%20Visual%20State%20Transitions%20for%20Consistent%20Video%20Retrieval/tables/ca989200d6b6e7ed1fdbb562953baf1f750b2e6398c3a25e9b42bf1ff718e11e.jpg)

<a id="paper-3094"></a>

## Causal Dependency-Aware Unsupervised Routing for Large Reasoning Models


### Images

![3767a7acbc24968e5e029b48e0597f5742b537557b6ae1c6830680d07ad97aec.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/3767a7acbc24968e5e029b48e0597f5742b537557b6ae1c6830680d07ad97aec.jpg)

![387bd553c556f079a09c2dfc3e05d996189bc5d3af9eb0d8afb7405b9ca517bf.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/387bd553c556f079a09c2dfc3e05d996189bc5d3af9eb0d8afb7405b9ca517bf.jpg)

![41cc09f5a510c4de7a0a5d7b1bb3a3f4da8fc7644894229950a8d0bf6d4864c4.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/41cc09f5a510c4de7a0a5d7b1bb3a3f4da8fc7644894229950a8d0bf6d4864c4.jpg)

![45d6eb1c8eb6b211881fac2ad8d3823b012c16f3df2ade3aa722e3b3887eb7e1.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/45d6eb1c8eb6b211881fac2ad8d3823b012c16f3df2ade3aa722e3b3887eb7e1.jpg)

![c47b07c739c6aadf931f5133b79a7d693a81d88016c23207e1c4bf26c1d203bf.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/c47b07c739c6aadf931f5133b79a7d693a81d88016c23207e1c4bf26c1d203bf.jpg)

![cb52ff2fd1f4f24fdd728ebd0e803e362b0aec7c59126ed520834dce9165a510.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/cb52ff2fd1f4f24fdd728ebd0e803e362b0aec7c59126ed520834dce9165a510.jpg)

![dba0ae9ec3831c9dd94a40a01ed495ef08ee43b6755967b0152642d59d65784d.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/images/dba0ae9ec3831c9dd94a40a01ed495ef08ee43b6755967b0152642d59d65784d.jpg)

### Tables

![1bb159d68059fbd276d673e224085c5cd78007ac58637ece70d7be01d870a9b9.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/tables/1bb159d68059fbd276d673e224085c5cd78007ac58637ece70d7be01d870a9b9.jpg)

![b7cc4a4c8567f43660eec4605ca91a39f0d96c14a0a56ec16abb870815bfe542.jpg](../icml_results/3133_Causal%20Dependency-Aware%20Unsupervised%20Routing%20for%20Large%20Reasoning%20Models/tables/b7cc4a4c8567f43660eec4605ca91a39f0d96c14a0a56ec16abb870815bfe542.jpg)

<a id="paper-3095"></a>

## XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling


### Images

![04aaeea4a245b6addb784044116c741f8215e2a37b14a557bf0548f5a74e5520.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/images/04aaeea4a245b6addb784044116c741f8215e2a37b14a557bf0548f5a74e5520.jpg)

![336e3884c5d042637432950b3fd3599fd1d623b7a614a0458e5792de822a3746.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/images/336e3884c5d042637432950b3fd3599fd1d623b7a614a0458e5792de822a3746.jpg)

![3705a8b4b50203845d11587395e983751bec4b5d57f9345f4049ab30099e20d6.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/images/3705a8b4b50203845d11587395e983751bec4b5d57f9345f4049ab30099e20d6.jpg)

![7029eddca217dbbf0085ef2db9171904435b83e9d5d7e1b22094d47a6bee5031.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/images/7029eddca217dbbf0085ef2db9171904435b83e9d5d7e1b22094d47a6bee5031.jpg)

![87094d1c01688363dff34e14e50247c5d684e6e2ece8101e2b5bdc273751b2ab.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/images/87094d1c01688363dff34e14e50247c5d684e6e2ece8101e2b5bdc273751b2ab.jpg)

### Tables

![01d907a8c0017f873b582d36ded64cfc3ef385f8ea910e7db554eb77f16f795a.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/01d907a8c0017f873b582d36ded64cfc3ef385f8ea910e7db554eb77f16f795a.jpg)

![2363b5ac0669dd779ce5d3ade9bdf11933457cbbf17e2f1d089f4f3348d13175.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/2363b5ac0669dd779ce5d3ade9bdf11933457cbbf17e2f1d089f4f3348d13175.jpg)

![4b84d5d247bea516e7e4fe1d8a8ea7e8c621830261178df77483fb0a1ca806c0.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/4b84d5d247bea516e7e4fe1d8a8ea7e8c621830261178df77483fb0a1ca806c0.jpg)

![e10ba65d4c5208bf96dacfec0e0bc04da631f9aa04ed2fc6560a82028b5ec8bc.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/e10ba65d4c5208bf96dacfec0e0bc04da631f9aa04ed2fc6560a82028b5ec8bc.jpg)

![e9b20264dd8afe31e8357cb0206a039be3b30f0b097a8fa884336291ed858359.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/e9b20264dd8afe31e8357cb0206a039be3b30f0b097a8fa884336291ed858359.jpg)

![f8cbcc557bc266b5a1bc7bdbea52cf275153b28c9b4e17127ec87c44a7b1415a.jpg](../icml_results/3134_XYZFlow_%20Scaling%20Multidimensional%20Shortcut%20Flows%20for%20Efficient%20Generative%20Modeling/tables/f8cbcc557bc266b5a1bc7bdbea52cf275153b28c9b4e17127ec87c44a7b1415a.jpg)

<a id="paper-3096"></a>

## Universal One-third Time Scaling in Learning Peaked Distributions


### Images

![00da7e8b6a61c084903058fbc6060e90c8232fcf79e1ea79c6fea44be4a91d2e.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/00da7e8b6a61c084903058fbc6060e90c8232fcf79e1ea79c6fea44be4a91d2e.jpg)

![01fcf1699580c3b550da918d845ad343a987f8e55a93bcc99a06971d526e585d.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/01fcf1699580c3b550da918d845ad343a987f8e55a93bcc99a06971d526e585d.jpg)

![043e038fdcc04422cf9ab54319b11200b0bc38f5a1c3901dceaf303eb5867172.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/043e038fdcc04422cf9ab54319b11200b0bc38f5a1c3901dceaf303eb5867172.jpg)

![1269be3129ae7b1fe724c3fd8a71b4728c1a0b372e7661b70ef18302f3c92ab8.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/1269be3129ae7b1fe724c3fd8a71b4728c1a0b372e7661b70ef18302f3c92ab8.jpg)

![212197d9ea7b9d3822bd3400cedf22b1bc778b820368e322c2c79ba3732243b8.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/212197d9ea7b9d3822bd3400cedf22b1bc778b820368e322c2c79ba3732243b8.jpg)

![2626f3b1d4024e0222ffc4945f243cbbf2b4a56ebeb9a21d77fd0a670ccfc26e.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/2626f3b1d4024e0222ffc4945f243cbbf2b4a56ebeb9a21d77fd0a670ccfc26e.jpg)

![2f9a4f6b20ec5d2cdccbd60b36d01811db7fa9a0ea2175cb000613ecd3b84140.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/2f9a4f6b20ec5d2cdccbd60b36d01811db7fa9a0ea2175cb000613ecd3b84140.jpg)

![30574624caf5dfd8c8c5fff517aefbe5de85dbb915c8373ee62de24f682fb6d6.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/30574624caf5dfd8c8c5fff517aefbe5de85dbb915c8373ee62de24f682fb6d6.jpg)

![337c3baba341a126ebd40a1a1e0d7f523861f02d9d1f67a5890cf5efdc369b7b.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/337c3baba341a126ebd40a1a1e0d7f523861f02d9d1f67a5890cf5efdc369b7b.jpg)

![368b50648b866dfd623dc05d3bc05ebfcb4316c4474007bf45c9dc4d21d8146e.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/368b50648b866dfd623dc05d3bc05ebfcb4316c4474007bf45c9dc4d21d8146e.jpg)

![3ba5edf574c847cd44b5a4ac3704fb7b5f26202e690b8db548d986219f8d8a1b.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/3ba5edf574c847cd44b5a4ac3704fb7b5f26202e690b8db548d986219f8d8a1b.jpg)

![3c211bdd91d7f7f0cb42b2bdd58f1b632da52f2f75136afd7ad434ddefbb5102.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/3c211bdd91d7f7f0cb42b2bdd58f1b632da52f2f75136afd7ad434ddefbb5102.jpg)

![4e3ca0d5b309b4cad09ea8e3f1a8f35072d9eb03b813eaea57c6bdc97b3a2d45.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/4e3ca0d5b309b4cad09ea8e3f1a8f35072d9eb03b813eaea57c6bdc97b3a2d45.jpg)

![51ba349007ab71422c605fc53afc5e8fade64a619f044b6f23678bf48702016d.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/51ba349007ab71422c605fc53afc5e8fade64a619f044b6f23678bf48702016d.jpg)

![699e510322f269cbd25ed156a2d4cfc0f4316e120b48e99bd2f52d8b56b92ac0.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/699e510322f269cbd25ed156a2d4cfc0f4316e120b48e99bd2f52d8b56b92ac0.jpg)

![7151e838aa76df0392e5dd9e85f618c2864572995b1fed3aa8d494c97504eab7.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/7151e838aa76df0392e5dd9e85f618c2864572995b1fed3aa8d494c97504eab7.jpg)

![7bf52d23c17fa969cfe1fce44359721d5364192a899ec5541c747be17b607550.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/7bf52d23c17fa969cfe1fce44359721d5364192a899ec5541c747be17b607550.jpg)

![89aed1e4b886eb3abb2e1d8f9781d61b188de477a097f9f28319f203af0e4720.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/89aed1e4b886eb3abb2e1d8f9781d61b188de477a097f9f28319f203af0e4720.jpg)

![8dc17cece424204c09489331949c1cbf0a570ed1f93ee0c07babff0caeb63723.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/8dc17cece424204c09489331949c1cbf0a570ed1f93ee0c07babff0caeb63723.jpg)

![9738ea6c6c74548b6a5d12411e97ba19e34b328edabb4e3aaee8af9d018d5303.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/9738ea6c6c74548b6a5d12411e97ba19e34b328edabb4e3aaee8af9d018d5303.jpg)

![99201789941ac74b8dbabc818490c5aaa191d0b5c59f368278b357654ab5e9e6.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/99201789941ac74b8dbabc818490c5aaa191d0b5c59f368278b357654ab5e9e6.jpg)

![a0d6901e55a61f5db74eba3def2043b9e25b9ea7a7c99b07ce7cca65a10f40f5.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/a0d6901e55a61f5db74eba3def2043b9e25b9ea7a7c99b07ce7cca65a10f40f5.jpg)

![a4f23be65f88f1feb92252561cbb705d47fa0fadb9a715a26a191f54ef642817.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/a4f23be65f88f1feb92252561cbb705d47fa0fadb9a715a26a191f54ef642817.jpg)

![a51134ad80fe6e4ddae2c24afeba6e32fd3b1ffa957b86634e69dd4337dc51d5.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/a51134ad80fe6e4ddae2c24afeba6e32fd3b1ffa957b86634e69dd4337dc51d5.jpg)

![ac33fbb55e1ecc5ea5d1dd10ee0a764dbf83cb2ea1f3bb074e8122771770bf53.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/ac33fbb55e1ecc5ea5d1dd10ee0a764dbf83cb2ea1f3bb074e8122771770bf53.jpg)

![b733c74e74af9cad3b30a9913c45e501f4eeffa87545e444d67bab7490d881c6.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/b733c74e74af9cad3b30a9913c45e501f4eeffa87545e444d67bab7490d881c6.jpg)

![bb12c037621941ee812d844d410ce25bd0654e463b36a6fce44211679e2934a3.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/bb12c037621941ee812d844d410ce25bd0654e463b36a6fce44211679e2934a3.jpg)

![d47a33b72103ba0d8c2803981a1befc72314f115b1ba9dcaa8fcc2374ac03a90.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/d47a33b72103ba0d8c2803981a1befc72314f115b1ba9dcaa8fcc2374ac03a90.jpg)

![d818dbebb031517b4b88aca633ee5139989da62ee95808fa134d5340eeebfb9b.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/d818dbebb031517b4b88aca633ee5139989da62ee95808fa134d5340eeebfb9b.jpg)

![eed7396204d2555970f5b4a067b2b249e1a5f6efe78e9a6f8ca50339e1aec70d.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/eed7396204d2555970f5b4a067b2b249e1a5f6efe78e9a6f8ca50339e1aec70d.jpg)

![f5eee029e48c7edafa2b5ed0fe07fad251264e4fc334703930b9ebc24183e9b1.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/f5eee029e48c7edafa2b5ed0fe07fad251264e4fc334703930b9ebc24183e9b1.jpg)

![fa94dc82ba5b83e247570500420ecf76928d71b12a0b7ec58ffbdc7d20068d51.jpg](../icml_results/3135_Universal%20One-third%20Time%20Scaling%20in%20Learning%20Peaked%20Distributions/images/fa94dc82ba5b83e247570500420ecf76928d71b12a0b7ec58ffbdc7d20068d51.jpg)

<a id="paper-3097"></a>

## Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search


### Images

![0dbfacef0426be283ae1d6738ca485c6e02b091d50a9d8adafd23356fe6f64f7.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/images/0dbfacef0426be283ae1d6738ca485c6e02b091d50a9d8adafd23356fe6f64f7.jpg)

![1d1f69a4bbf53f15148cb715199c559496c9440143551f9f31c1aad0c516ab1b.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/images/1d1f69a4bbf53f15148cb715199c559496c9440143551f9f31c1aad0c516ab1b.jpg)

![812e12a1e679e67d6e2f5e3aa6f355d5230d1068ebf7d0c2356aebdd52a7a065.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/images/812e12a1e679e67d6e2f5e3aa6f355d5230d1068ebf7d0c2356aebdd52a7a065.jpg)

![e06cc3ab4a6775926f823ffd716d38d1a1e2ad4d19f7ea98ebf50bbdfc4b346d.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/images/e06cc3ab4a6775926f823ffd716d38d1a1e2ad4d19f7ea98ebf50bbdfc4b346d.jpg)

### Tables

![596d48fdc95acb41179683cbd1e871a6b069f044ee0e0bb1cc73a6122b026641.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/596d48fdc95acb41179683cbd1e871a6b069f044ee0e0bb1cc73a6122b026641.jpg)

![84a09d3de174473547e6d09d1edbf8ad47858136b8003bf2908a76e3b0e605b5.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/84a09d3de174473547e6d09d1edbf8ad47858136b8003bf2908a76e3b0e605b5.jpg)

![8c9d4031f0dc1ad22c40f1ee14e4b8bfd48ee4146422fcfe2f7aa05ff8461cfe.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/8c9d4031f0dc1ad22c40f1ee14e4b8bfd48ee4146422fcfe2f7aa05ff8461cfe.jpg)

![9040f3377acb71f03c66fb5dd13f7ee7c662a15f05304679dd7eaa23a6b8fd51.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/9040f3377acb71f03c66fb5dd13f7ee7c662a15f05304679dd7eaa23a6b8fd51.jpg)

![97284ddd47da5e5ecc162f93acefc05139eb1d32845820b68fc9311dee7c7760.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/97284ddd47da5e5ecc162f93acefc05139eb1d32845820b68fc9311dee7c7760.jpg)

![9f46fbb613d3f449e825c5722a64feb30e5f57b1fee681c82d8d83eebab24869.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/9f46fbb613d3f449e825c5722a64feb30e5f57b1fee681c82d8d83eebab24869.jpg)

![bf32e9f9e8bb1f3de1e43a9efa13e7b9e103d353dc33eb63bf403f00683793bb.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/bf32e9f9e8bb1f3de1e43a9efa13e7b9e103d353dc33eb63bf403f00683793bb.jpg)

![d1deed529f0b815cb388e05fc98fd466250847bd6d2fe4fbe22597f11d761e37.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/d1deed529f0b815cb388e05fc98fd466250847bd6d2fe4fbe22597f11d761e37.jpg)

![d4304afe6fef944716e29d417fd5bf906801053ba57a6d18722951ee2000fb37.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/d4304afe6fef944716e29d417fd5bf906801053ba57a6d18722951ee2000fb37.jpg)

![d80e91c6b4586b6ee31657b31400d9148939c71ebbaf30e5a1b18445d8471693.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/d80e91c6b4586b6ee31657b31400d9148939c71ebbaf30e5a1b18445d8471693.jpg)

![e141648532c7a473f1af2d12ecfa517c021dace5cb1d0998e862033fddb83ab5.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/e141648532c7a473f1af2d12ecfa517c021dace5cb1d0998e862033fddb83ab5.jpg)

![fa91dfd17ad91e04c1e5ccb442367012904ecab6214d5303854acda7bc6767b1.jpg](../icml_results/3136_Structured%20Progressive%20Knowledge%20Activation%20for%20LLM-Driven%20Neural%20Architecture%20Search/tables/fa91dfd17ad91e04c1e5ccb442367012904ecab6214d5303854acda7bc6767b1.jpg)

<a id="paper-3098"></a>

## UniDrag: Unified Multi-Field Prediction and Robust Shape Optimization for Vehicle Aerodynamics


### Images

![1276894405086e3f4612800b05af70538f7d886f6e88e36c84503c20d7a27a54.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/1276894405086e3f4612800b05af70538f7d886f6e88e36c84503c20d7a27a54.jpg)

![2cf8aa5daeabe2654ee5f0731973f6b6d3dcbb53e3dce4646bbfa15447a49787.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/2cf8aa5daeabe2654ee5f0731973f6b6d3dcbb53e3dce4646bbfa15447a49787.jpg)

![30e73ee40dbe5cb327eb5fb38cf579c3b03533f3ba5038b7da62348fb9982c91.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/30e73ee40dbe5cb327eb5fb38cf579c3b03533f3ba5038b7da62348fb9982c91.jpg)

![31d3b004acb39d4bb9bdd9eb24f47355c80941e635b5e067e723c528816eba7b.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/31d3b004acb39d4bb9bdd9eb24f47355c80941e635b5e067e723c528816eba7b.jpg)

![5ceb641354becde411989ba8ad320e7a3dc9c577d435e3b5c55a9a351afb9a41.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/5ceb641354becde411989ba8ad320e7a3dc9c577d435e3b5c55a9a351afb9a41.jpg)

![775260da2b822d7d985ab68600bc9d2ea9ec068174cbe572fd99a6cc7fcffcc4.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/775260da2b822d7d985ab68600bc9d2ea9ec068174cbe572fd99a6cc7fcffcc4.jpg)

![79297db84069b1de7a55f7cb8ebd064c16a4f6582354c883078ee4c5a445adde.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/79297db84069b1de7a55f7cb8ebd064c16a4f6582354c883078ee4c5a445adde.jpg)

![845253fbe300c352906d317bdc97fc6adb451279a141a8788730204601460276.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/845253fbe300c352906d317bdc97fc6adb451279a141a8788730204601460276.jpg)

![8cf65f4c2edf0924d84420122b211b502117de852ccc23bf769b4f40419938ec.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/8cf65f4c2edf0924d84420122b211b502117de852ccc23bf769b4f40419938ec.jpg)

![9ea8e0e003109c0c4be724d61775c61b2a4dfb33502d17448071e9ba0d41a31b.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/9ea8e0e003109c0c4be724d61775c61b2a4dfb33502d17448071e9ba0d41a31b.jpg)

![b1fac5ede3ef2de93874d4d10ba7b14fffe8c9791fcf7cb1b00331e598bc97db.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/b1fac5ede3ef2de93874d4d10ba7b14fffe8c9791fcf7cb1b00331e598bc97db.jpg)

![b3dbeb969028f710f4a96fbb7e32e20545f5789f995c571ee1cda838c9b24664.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/b3dbeb969028f710f4a96fbb7e32e20545f5789f995c571ee1cda838c9b24664.jpg)

![d3c1a2d54f0abfe0652fddde108f4eb1f9fb7e0994c3d06e0601ead61ae1f85c.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/d3c1a2d54f0abfe0652fddde108f4eb1f9fb7e0994c3d06e0601ead61ae1f85c.jpg)

![e1c423c2a295e0f238cc4908270f5e395d5633761ef697dee520b3a17cccc103.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/e1c423c2a295e0f238cc4908270f5e395d5633761ef697dee520b3a17cccc103.jpg)

![ee181ae05b0ef1f90ba04ed10395ecd620ae29546db5b53f118fafbf1d704d91.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/ee181ae05b0ef1f90ba04ed10395ecd620ae29546db5b53f118fafbf1d704d91.jpg)

![ee2dcba32ce821b2447de7618aad1664a3142cc939c2f6af68725c59077d5fef.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/images/ee2dcba32ce821b2447de7618aad1664a3142cc939c2f6af68725c59077d5fef.jpg)

### Tables

![10f426fac11c61c6905cd637d92e0fe5bdcd3897bd4da3316db790f63a5741b9.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/10f426fac11c61c6905cd637d92e0fe5bdcd3897bd4da3316db790f63a5741b9.jpg)

![7f8af96187d560fd901f1e5d7e1ffeb3b70dfa5e35314ae17ff1c67bb430121b.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/7f8af96187d560fd901f1e5d7e1ffeb3b70dfa5e35314ae17ff1c67bb430121b.jpg)

![8afe211e66ac66951dff0bbf280f3b92250d346af7842da32315c625dd4e173f.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/8afe211e66ac66951dff0bbf280f3b92250d346af7842da32315c625dd4e173f.jpg)

![8cb71dd82a2aecad63bd88638d1cb53c959f45ec057195f7956b61c9680d4d5f.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/8cb71dd82a2aecad63bd88638d1cb53c959f45ec057195f7956b61c9680d4d5f.jpg)

![97e5af508409b87c669f24d653a79838d07dfb8dbe9fb75ed74e61685daa586b.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/97e5af508409b87c669f24d653a79838d07dfb8dbe9fb75ed74e61685daa586b.jpg)

![9831b5d6d5f6a7d3e68f3cae88654521e3691ab83ab460ad52584a650a115039.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/9831b5d6d5f6a7d3e68f3cae88654521e3691ab83ab460ad52584a650a115039.jpg)

![be023a19410ffdcdd19828dad7e70a025f9d4fdd45cac10253f654668a99be36.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/be023a19410ffdcdd19828dad7e70a025f9d4fdd45cac10253f654668a99be36.jpg)

![c12e311360c20cb7231f65b0f6c74dedd5c8ca2b0150c14cc15ff4226fd1532f.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/c12e311360c20cb7231f65b0f6c74dedd5c8ca2b0150c14cc15ff4226fd1532f.jpg)

![c285ee91369210a956849d1096877a80f955228a883a08b262d5e93c9f39e83e.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/c285ee91369210a956849d1096877a80f955228a883a08b262d5e93c9f39e83e.jpg)

![c54ba7c37f340ebec2fe7577d6db2e83664f888a968032e685e232eadf7788ed.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/c54ba7c37f340ebec2fe7577d6db2e83664f888a968032e685e232eadf7788ed.jpg)

![c6417ce57858759ecfffb35bd3f90c6a3a33814e42331b6fb0db41130e0bc356.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/c6417ce57858759ecfffb35bd3f90c6a3a33814e42331b6fb0db41130e0bc356.jpg)

![d39e93eb1c84fc9b8a3991b3a356b7daa8d575b1bd580fcd1361781c20978cf8.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/d39e93eb1c84fc9b8a3991b3a356b7daa8d575b1bd580fcd1361781c20978cf8.jpg)

![dd402d25414d096a6dca572bbfe74a28582e54de0e3c3329b1fd90cdfaa35d8c.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/dd402d25414d096a6dca572bbfe74a28582e54de0e3c3329b1fd90cdfaa35d8c.jpg)

![e1be16948cb302a24c7429d89179a3b9097a25560bf7391fdc040720a8a2f742.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/e1be16948cb302a24c7429d89179a3b9097a25560bf7391fdc040720a8a2f742.jpg)

![e25a8d1f77c89440a9f0e3f2d2b78011d03c7d1e4b792f417ad7cb132adb7c04.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/e25a8d1f77c89440a9f0e3f2d2b78011d03c7d1e4b792f417ad7cb132adb7c04.jpg)

![eab0bd83fdd6549fc62b3e6eeafdeebea776fa2d5d412af9a17aff974c990925.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/eab0bd83fdd6549fc62b3e6eeafdeebea776fa2d5d412af9a17aff974c990925.jpg)

![f20b707af6507a8dbe331a7088bb102037d7c4f9ed714bcf0e3724de3b7fa20f.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/f20b707af6507a8dbe331a7088bb102037d7c4f9ed714bcf0e3724de3b7fa20f.jpg)

![f891d38e01b7642202e123ae08539761fe64631c3bb5ce48da48441b7b34a343.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/f891d38e01b7642202e123ae08539761fe64631c3bb5ce48da48441b7b34a343.jpg)

![fbd5920fa263687a612945ce42bd5d0de02a16cbe950668fde1408b923a2c803.jpg](../icml_results/3137_UniDrag_%20Unified%20Multi-Field%20Prediction%20and%20Robust%20Shape%20Optimization%20for%20Vehicle%20Aerodynamics/tables/fbd5920fa263687a612945ce42bd5d0de02a16cbe950668fde1408b923a2c803.jpg)

<a id="paper-3099"></a>

## SAGE: A Dataflow-Native Framework for Modular, Controllable, and Transparent LLM-Augmented Reasoning


### Images

![335416daae07ef123f32425da767079b421b27ccc5f32b8b857133449c6e0cc3.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/335416daae07ef123f32425da767079b421b27ccc5f32b8b857133449c6e0cc3.jpg)

![425136da46fd15f6dd82065c73df37e4dac9debe1da16f57ff0bad9d41d5de2c.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/425136da46fd15f6dd82065c73df37e4dac9debe1da16f57ff0bad9d41d5de2c.jpg)

![481566df1c33c39502ff0a0d34a68ca75f1d71717c15cc927f0a2fa5061b57b1.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/481566df1c33c39502ff0a0d34a68ca75f1d71717c15cc927f0a2fa5061b57b1.jpg)

![6181ba2a2a150b3a6c2cc6a9ff56b43b2a04aadec6ba0d7296d3a7124a040c49.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/6181ba2a2a150b3a6c2cc6a9ff56b43b2a04aadec6ba0d7296d3a7124a040c49.jpg)

![928aa91ee9a410450d8675f1afebabb71ac37c32e7da8e974534fc6972274a51.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/928aa91ee9a410450d8675f1afebabb71ac37c32e7da8e974534fc6972274a51.jpg)

![c9ef5ecf26391c4d17ca46307f75309a71f302e1dc5db517ac7eb1e56d195039.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/c9ef5ecf26391c4d17ca46307f75309a71f302e1dc5db517ac7eb1e56d195039.jpg)

![f3135bae8962fba3677656baff634d5f1517a8c0cd34f3df7f51f61b234d1a35.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/images/f3135bae8962fba3677656baff634d5f1517a8c0cd34f3df7f51f61b234d1a35.jpg)

### Tables

![13ba6c5eec09ad444f92bdbccb057e012b67c61aa4ef343873368c9482ca97f4.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/13ba6c5eec09ad444f92bdbccb057e012b67c61aa4ef343873368c9482ca97f4.jpg)

![1e7d10de23711a1a991f48dba552f015cdcc1213f4a7e35a14b75b39da2f0939.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/1e7d10de23711a1a991f48dba552f015cdcc1213f4a7e35a14b75b39da2f0939.jpg)

![28c86f47db2055c79eda8f47dbc0f5a7babbabce8025c71dc0309f0d6b52201b.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/28c86f47db2055c79eda8f47dbc0f5a7babbabce8025c71dc0309f0d6b52201b.jpg)

![4aa205b8bbf79c9ec526455b381b7c98253ccd5074b2057d048b8611e60b53b8.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/4aa205b8bbf79c9ec526455b381b7c98253ccd5074b2057d048b8611e60b53b8.jpg)

![5463a56a321eefb0214bc7f65fd1ec19d5953837d9f2f68a2b1f78657c547536.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/5463a56a321eefb0214bc7f65fd1ec19d5953837d9f2f68a2b1f78657c547536.jpg)

![5f4176fc1bb54646758e0d6000cf832cf15acc8f89b3a98cd286356c0585b30a.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/5f4176fc1bb54646758e0d6000cf832cf15acc8f89b3a98cd286356c0585b30a.jpg)

![72f5746fb959bc859103ee5d5c3b828d76512decb93c805d2c0f7edab5d38d05.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/72f5746fb959bc859103ee5d5c3b828d76512decb93c805d2c0f7edab5d38d05.jpg)

![be1cefa098f31f238c1beaa3a6d0a65c5738918beb45eb89a5d89478187d7183.jpg](../icml_results/3138_SAGE_%20A%20Dataflow-Native%20Framework%20for%20Modular%2C%20Controllable%2C%20and%20Transparent%20LLM-Augmented%20Reasoning/tables/be1cefa098f31f238c1beaa3a6d0a65c5738918beb45eb89a5d89478187d7183.jpg)

<a id="paper-3100"></a>

## Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach


### Images

![34a9fbc96cb456d5653ec1536000daf28a40309f9669b870244eb3d1ea86b986.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/34a9fbc96cb456d5653ec1536000daf28a40309f9669b870244eb3d1ea86b986.jpg)

![655a3a0560114ed59a91d6294f23505e39a25a6188b229b9d9cb90d34bf1a38e.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/655a3a0560114ed59a91d6294f23505e39a25a6188b229b9d9cb90d34bf1a38e.jpg)

![856282d0df90dec66b289387df046ea46c2ec5c63b0dd0cc1f5084b39ebe6e77.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/856282d0df90dec66b289387df046ea46c2ec5c63b0dd0cc1f5084b39ebe6e77.jpg)

![9d3df30487e853bc222219052938e9753a5a077a19dc1cdf26f4f1bad39bf020.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/9d3df30487e853bc222219052938e9753a5a077a19dc1cdf26f4f1bad39bf020.jpg)

![a4b0ddecfbbc3f523ddb2429fa887c4fe2a331b99f4761a4ceed288ab6a45c87.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/a4b0ddecfbbc3f523ddb2429fa887c4fe2a331b99f4761a4ceed288ab6a45c87.jpg)

![a982f6a463ad9ffdf9cbd74264007ada94b7b7b733253d3b05fe0ddfeb963079.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/a982f6a463ad9ffdf9cbd74264007ada94b7b7b733253d3b05fe0ddfeb963079.jpg)

![dc536900c8aa5c4d8412bc1bac3fbc61edf59f9e9d22e48cfd499e9cff53adb3.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/images/dc536900c8aa5c4d8412bc1bac3fbc61edf59f9e9d22e48cfd499e9cff53adb3.jpg)

### Tables

![14fcff2c8a5cc00b00026a1b787dccf93b23e5f840df3daacdde0ab5356634dd.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/14fcff2c8a5cc00b00026a1b787dccf93b23e5f840df3daacdde0ab5356634dd.jpg)

![42232fa0330b2b957b964ec2426c639dd17fbe208ed07caa61924cc681ad4f90.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/42232fa0330b2b957b964ec2426c639dd17fbe208ed07caa61924cc681ad4f90.jpg)

![4a1efd92a9a0542ea38661594c779c221b0fc0b9b5acb5d2cfa7170d7d1a18bc.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/4a1efd92a9a0542ea38661594c779c221b0fc0b9b5acb5d2cfa7170d7d1a18bc.jpg)

![51303b3cbe30804d45537b37969df98547daa7a0d4a1d5636cfe745a21020387.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/51303b3cbe30804d45537b37969df98547daa7a0d4a1d5636cfe745a21020387.jpg)

![5a90a9e173a3fdbf2d215868c2d95e75ffa00f7609a12f80a7c35c32e766bd75.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/5a90a9e173a3fdbf2d215868c2d95e75ffa00f7609a12f80a7c35c32e766bd75.jpg)

![6915ee670a3a2ac6b40f57303e943e6663de3e711a3fba981c3d30a9b9323fef.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/6915ee670a3a2ac6b40f57303e943e6663de3e711a3fba981c3d30a9b9323fef.jpg)

![89fc934a359f1afc93582bd93de0fdad3a3dcd3d1da9f764394a9136096ca68d.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/89fc934a359f1afc93582bd93de0fdad3a3dcd3d1da9f764394a9136096ca68d.jpg)

![91ab92d88abf5062ba382260e92bfd6d8e161332da1625e4de72c9ddc3328461.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/91ab92d88abf5062ba382260e92bfd6d8e161332da1625e4de72c9ddc3328461.jpg)

![957f6516f8e295146ad55b1a56e5e50d2b42e68dbda75be275bd3de5e6f6b908.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/957f6516f8e295146ad55b1a56e5e50d2b42e68dbda75be275bd3de5e6f6b908.jpg)

![98aa41ec3af92daf2b97bb5a8314b702f88c3e1370292cd6afaacc8c153f05bf.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/98aa41ec3af92daf2b97bb5a8314b702f88c3e1370292cd6afaacc8c153f05bf.jpg)

![b8f9903008eb78506e51e4b989c43ef80aadccf70a55bf067910ed7e47379491.jpg](../icml_results/3139_Rethinking%20Feature%20Alignment%20in%20Generalist%20Graph%20Anomaly%20Detection_%20A%20Relational%20Fingerprint-based%20A/tables/b8f9903008eb78506e51e4b989c43ef80aadccf70a55bf067910ed7e47379491.jpg)

<a id="paper-3101"></a>

## SLASH the Sink: Sharpening Structural Attention Inside LLMs


### Images

![04e64ab6d1a8b2cdcbd4f9b59e1edb244a966c3dce694dc0897d90faf46b2224.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/04e64ab6d1a8b2cdcbd4f9b59e1edb244a966c3dce694dc0897d90faf46b2224.jpg)

![23f5a8124b4f0285d3c42ed27156ec761c505db5747c66d8cbb8583ae76b3e6c.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/23f5a8124b4f0285d3c42ed27156ec761c505db5747c66d8cbb8583ae76b3e6c.jpg)

![2c1e33a4208dcaa4bc436e2de165e54aea30978031910069d40fd530c10ea327.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/2c1e33a4208dcaa4bc436e2de165e54aea30978031910069d40fd530c10ea327.jpg)

![32135111c5934181fcf211b8b3c5fa4c03dbfce1713a077180d10fa2c3a302cb.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/32135111c5934181fcf211b8b3c5fa4c03dbfce1713a077180d10fa2c3a302cb.jpg)

![3769f681906ba09226eb161cada34e8fd738813e3ff6c6aade455267155b3d01.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/3769f681906ba09226eb161cada34e8fd738813e3ff6c6aade455267155b3d01.jpg)

![4c292bb57e5c31e1a8cea4482d223bed2beb279bf7a66864665685ee2f75a02a.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/4c292bb57e5c31e1a8cea4482d223bed2beb279bf7a66864665685ee2f75a02a.jpg)

![555683c6d8f9e9691f8fe7e21d97db5c5d1d8138f171dd6a82387aa4ba68379f.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/555683c6d8f9e9691f8fe7e21d97db5c5d1d8138f171dd6a82387aa4ba68379f.jpg)

![6e2fcee29dce6f76930cd7d0fa4121c3ccea3f1ce1e697ef4da74c1ed9665a5d.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/6e2fcee29dce6f76930cd7d0fa4121c3ccea3f1ce1e697ef4da74c1ed9665a5d.jpg)

![78979ba959e52ff0197915a530e454433561c77169bc8ccd35beefb39bd10d16.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/78979ba959e52ff0197915a530e454433561c77169bc8ccd35beefb39bd10d16.jpg)

![b513418b358b9e1145bacccc97586848c4c9efa240b6a2d7dc315faa61020dec.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/b513418b358b9e1145bacccc97586848c4c9efa240b6a2d7dc315faa61020dec.jpg)

![d2c24fe9193ff519480199a185b8cf4e1af36b0c9764873aaf4bc1463bdbed88.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/d2c24fe9193ff519480199a185b8cf4e1af36b0c9764873aaf4bc1463bdbed88.jpg)

![fbb9f83f0bf509a3622f90fefd3c4df60e3f5c74fbe4f9a4939d781b55fa0f9d.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/images/fbb9f83f0bf509a3622f90fefd3c4df60e3f5c74fbe4f9a4939d781b55fa0f9d.jpg)

### Tables

![03bac5c05da8d575b9f153ea33a7f433adbb97cf97ec32dc9efc4013f30bdb25.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/03bac5c05da8d575b9f153ea33a7f433adbb97cf97ec32dc9efc4013f30bdb25.jpg)

![19451c2120e31c33a6dc7278849162d06ba9604d892f952a2feaa71fad9d5ff7.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/19451c2120e31c33a6dc7278849162d06ba9604d892f952a2feaa71fad9d5ff7.jpg)

![3460db6bef41ed70fe8333fca832591eed6caaa1622e6f8526e2fa5f2e6c93c4.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/3460db6bef41ed70fe8333fca832591eed6caaa1622e6f8526e2fa5f2e6c93c4.jpg)

![4c94fc5dbf936c128099d0a1b71ff34ea589e5ce69f8d726ba88bca218052548.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/4c94fc5dbf936c128099d0a1b71ff34ea589e5ce69f8d726ba88bca218052548.jpg)

![95cca3a62a82ededa9a5184627ab50045e01245944b8236556a4e5e5ff163576.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/95cca3a62a82ededa9a5184627ab50045e01245944b8236556a4e5e5ff163576.jpg)

![b0d91613f5cc7b2fb16c9824ac2811198bb3649e7cfa20faa8cc98ef9fa7d2fa.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/b0d91613f5cc7b2fb16c9824ac2811198bb3649e7cfa20faa8cc98ef9fa7d2fa.jpg)

![b5a8a940be728cb79812ad30806b8cb0557cfda18599325f8e121e0fdb3be282.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/b5a8a940be728cb79812ad30806b8cb0557cfda18599325f8e121e0fdb3be282.jpg)

![ba8e9eb4e01c454e813d1d7da83089951058a70e7057fc7f4701c527d3ee0818.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/ba8e9eb4e01c454e813d1d7da83089951058a70e7057fc7f4701c527d3ee0818.jpg)

![e57dd1174c1c3f27689f9a973db6cb7fccfd87bb374a259fa036e5518ccf3f7f.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/e57dd1174c1c3f27689f9a973db6cb7fccfd87bb374a259fa036e5518ccf3f7f.jpg)

![f12a039488ff9054f7621095f10aabc667b9a1903cb37a47a4505b4b76c35cee.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/f12a039488ff9054f7621095f10aabc667b9a1903cb37a47a4505b4b76c35cee.jpg)

![f52a1cc1df04cd1e8eacaf67f1a0c79b08987980d74c65b9829bbab7c6ff6170.jpg](../icml_results/3140_SLASH%20the%20Sink_%20Sharpening%20Structural%20Attention%20Inside%20LLMs/tables/f52a1cc1df04cd1e8eacaf67f1a0c79b08987980d74c65b9829bbab7c6ff6170.jpg)

<a id="paper-3102"></a>

## Next-Gen CAPTCHAs: Leveraging the Cognitive Gap for Scalable and Diverse GUI-Agent Defense

### Images

![0136d7c610c8c5763b877bf50bfd4dedd689ebbec7461516fc3db50be054ce02.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/0136d7c610c8c5763b877bf50bfd4dedd689ebbec7461516fc3db50be054ce02.jpg)

![098dcb2e2be6f6fa3b0346cfeaf7978ba28bf3d4155c7f67293f5452603e970e.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/098dcb2e2be6f6fa3b0346cfeaf7978ba28bf3d4155c7f67293f5452603e970e.jpg)

![0d2894c16232533b4a731e47ca18ffe284608e9330d4f9f293b5e03a9c851038.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/0d2894c16232533b4a731e47ca18ffe284608e9330d4f9f293b5e03a9c851038.jpg)

![24da2928f1be82de40e4e948fb0fabe815707f24b01b2cd4afdde1fe8c262519.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/24da2928f1be82de40e4e948fb0fabe815707f24b01b2cd4afdde1fe8c262519.jpg)

![62266a66c94b11d19f183cca2de5ba236baea6eb997867659065895f2d78b303.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/62266a66c94b11d19f183cca2de5ba236baea6eb997867659065895f2d78b303.jpg)

![662833443ecb20ee1db66ec7df896df4f89f635c7aefd2f4ab2de3bd17af6bbd.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/662833443ecb20ee1db66ec7df896df4f89f635c7aefd2f4ab2de3bd17af6bbd.jpg)

![8b62391541cb1497df9f6a6dd3849e9079d7a4c2e24f73e99d4d21e8c4893164.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/8b62391541cb1497df9f6a6dd3849e9079d7a4c2e24f73e99d4d21e8c4893164.jpg)

![9e8b00edbeee4037a8ede4e5d103bc1b87fafff23c84fdd574039349095cd8b7.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/9e8b00edbeee4037a8ede4e5d103bc1b87fafff23c84fdd574039349095cd8b7.jpg)

![a10d6f6fa1bcacc43e797c5c3fd64c4f982fccfcc5cad6e112cb2840fc7da35a.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/a10d6f6fa1bcacc43e797c5c3fd64c4f982fccfcc5cad6e112cb2840fc7da35a.jpg)

![a87683c1287b47867b0e2fa7838aa060895dc730474734b64d60a19c66614330.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/a87683c1287b47867b0e2fa7838aa060895dc730474734b64d60a19c66614330.jpg)

![ab15419c712cd68ef09072003c4dc04d85027a5e78f3d063418da0f249f967e7.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/ab15419c712cd68ef09072003c4dc04d85027a5e78f3d063418da0f249f967e7.jpg)

![f5a4aa4ac8ed684d276980f54e0f09d156735753c8f213ca5d3d6a19917ed565.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/f5a4aa4ac8ed684d276980f54e0f09d156735753c8f213ca5d3d6a19917ed565.jpg)

### Tables

![0d9ef0641dbfeff665959f1852c8ca925b7e9f9a3cb1d6d4d6e865adb628ae87.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/0d9ef0641dbfeff665959f1852c8ca925b7e9f9a3cb1d6d4d6e865adb628ae87.jpg)

![19ce8d139f6bbd6ca1440e39c68577f441366687d9f7d130203a6e212d8fc804.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/19ce8d139f6bbd6ca1440e39c68577f441366687d9f7d130203a6e212d8fc804.jpg)

![578c1f0fd06ea5d922cbdad5b2acf18017ec5d65597f2790e111d28caa023c72.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/578c1f0fd06ea5d922cbdad5b2acf18017ec5d65597f2790e111d28caa023c72.jpg)

![7c1d0abbad357d1f8cd6804448edecfb64d235667beecc49b6759d906366952e.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/7c1d0abbad357d1f8cd6804448edecfb64d235667beecc49b6759d906366952e.jpg)

![b0888192709d1e944bf705cbaa1e0c7aa6c7c3f5ed231cc9cc85127c46b78384.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/b0888192709d1e944bf705cbaa1e0c7aa6c7c3f5ed231cc9cc85127c46b78384.jpg)

![b64490d737c9c5f3eb11d8925ff5ee1beb0c1c6d907380723e2d5881726382da.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/b64490d737c9c5f3eb11d8925ff5ee1beb0c1c6d907380723e2d5881726382da.jpg)

![d02c0eac02570d0d4d7cb7341176594477e2dfb312545154856467f011cf79e1.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/d02c0eac02570d0d4d7cb7341176594477e2dfb312545154856467f011cf79e1.jpg)
