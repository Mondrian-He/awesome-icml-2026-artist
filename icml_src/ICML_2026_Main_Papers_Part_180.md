# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 180 of 197

## 目录 (Table of Contents)

1. [GeoMoLa: Geometry-Aware Motion Latents for Learning Robust Manipulation Policies](#paper-5908)
2. [Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](#paper-5909)
3. [Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks](#paper-5910)
4. [DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving](#paper-5911)
5. [Reference-Free Meta-Learning for Generalized Implicit Neural Representation in Efficient MRI Reconstruction](#paper-5912)
6. [Utonia: Toward One Encoder for All Point Clouds](#paper-5913)
7. [dTRPO : Trajectory Reduction in Policy Optimization of Diffusion Large Language Models](#paper-5914)
8. [See First, Reason Later: Mutual Information-Guided Reinforcement Learning for Vision-Language Models](#paper-5915)
9. [MUSA-PINN: Multi-scale Weak-form Physics-Informed Neural Networks for Fluid Flow in Complex Geometries](#paper-5916)
10. [AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](#paper-5917)
11. [Who Transfers Safety? Identifying and Targeting Cross-Lingual Shared Safety Neurons](#paper-5918)
12. [Agentic Confidence Calibration](#paper-5919)
13. [Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](#paper-5920)
14. [Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](#paper-5921)
15. [Benchmarking and Enhancing VLM for Compressed Image Understanding](#paper-5922)
16. [SafeCompass: Dynamic Chain-of-Thought Steering via Inference-Time Safety Signals](#paper-5923)
17. [Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning](#paper-5924)
18. [Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](#paper-5925)
19. [AOEB: Benchmarking Agent-Oriented Multimodal Embeddings](#paper-5926)
20. [4DPC$^2$hat: Towards Dynamic Point Cloud Understanding with Failure-Aware Bootstrapping](#paper-5927)
21. [Backjump-on-Graph: Empowering Large Language Models with Reinforced Retrospective Exploration for Agentic Knowledge Graph Reasoning](#paper-5928)
22. [ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](#paper-5929)
23. [SemBind: Binding Diffusion Watermarks to Semantics Against Black-Box Forgery Attacks](#paper-5930)
24. [BTSP-CAM: A Brain-Inspired Geometric Memory for Class-Incremental Learning](#paper-5931)
25. [Focus, Align, and Sustain: Counteracting Gradient Dilution in Incremental Object Detection](#paper-5932)
26. [Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](#paper-5933)
27. [Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](#paper-5934)
28. [Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](#paper-5935)
29. [A Game-Theoretic Framework for Measuring and Explaining Metric Compatibility in Fair Machine Learning](#paper-5936)
30. [GenAlign: Towards Unified Alignment Framework of MLLMs via Generative Reward Model](#paper-5937)
31. [AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models](#paper-5938)
32. [Cross-Modal Semantic Decoupling and Transfer for Text-to-Visible-Infrared Person Re-Identification](#paper-5939)
33. [Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook Strategy](#paper-5940)

---


<a id="paper-5908"></a>

## GeoMoLa: Geometry-Aware Motion Latents for Learning Robust Manipulation Policies

### Images

![0b81df73ed728c7d0eb3751be076436a81d327c6db8a6967055c3f014c3c832d.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/0b81df73ed728c7d0eb3751be076436a81d327c6db8a6967055c3f014c3c832d.jpg)

![1b59b1e12e16f3e1afb3314cc4b4fd247f99199758b6d4540cc3ae6a2a4d4361.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/1b59b1e12e16f3e1afb3314cc4b4fd247f99199758b6d4540cc3ae6a2a4d4361.jpg)

![4748c80706f3174df770f64cbe8caf6a597dc0be436c8540b9c64a0f2f24a5af.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/4748c80706f3174df770f64cbe8caf6a597dc0be436c8540b9c64a0f2f24a5af.jpg)

![4bf9960958df9745b885927e9516bec2767951451a5feee5ed4592a087e857b1.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/4bf9960958df9745b885927e9516bec2767951451a5feee5ed4592a087e857b1.jpg)

![4c8f92168446f0597a37fb1d2234e736ef270f19609b219c53c88bd21ba59e28.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/4c8f92168446f0597a37fb1d2234e736ef270f19609b219c53c88bd21ba59e28.jpg)

![4de212c47fe16a7a6ee3e5fb39b395fc8634ea444cbcff67bca16c362266d691.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/4de212c47fe16a7a6ee3e5fb39b395fc8634ea444cbcff67bca16c362266d691.jpg)

![5a5ebeee0062b75cd72da9f5866439030c9dd74ee072bcc9fd961cd941b5fb3c.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/5a5ebeee0062b75cd72da9f5866439030c9dd74ee072bcc9fd961cd941b5fb3c.jpg)

![75cf67fa15640cce3c66dacf9783082ff862f48c3b95c0772b5b270a4581501e.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/75cf67fa15640cce3c66dacf9783082ff862f48c3b95c0772b5b270a4581501e.jpg)

![971c33ef245620f9e408265f314d413dc6010c3ea4549b30e8b6dab45e610c82.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/images/971c33ef245620f9e408265f314d413dc6010c3ea4549b30e8b6dab45e610c82.jpg)

### Tables

![abafff24b7e63ff14442a2d060ca22dc2c1382d7d27e4652bc0a3950c3112e50.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/tables/abafff24b7e63ff14442a2d060ca22dc2c1382d7d27e4652bc0a3950c3112e50.jpg)

![d4bc3dd3abed44221e334daddf095e8b70d34dd8aec89856b5fa6c318c691641.jpg](../icml_results/5956_When%20and%20How%20Human%20Curation%20Backfires_%20Preference%20Alignment%20under%20Multi-Model%20Self-Consuming%20Loop/tables/d4bc3dd3abed44221e334daddf095e8b70d34dd8aec89856b5fa6c318c691641.jpg)

## GeoMoLa: Geometry-Aware Motion Latents for Learning Robust Manipulation Policies


### Images

![03d2e5a8f4f745e5f49e3922db9e8132dc1fc9f621d0e16b6c3fee17887b5c33.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/03d2e5a8f4f745e5f49e3922db9e8132dc1fc9f621d0e16b6c3fee17887b5c33.jpg)

![0768fd4c1b66d061b76fbd560fbe13f0ada9507fef12a5bc38d3b0bf6dd84e44.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/0768fd4c1b66d061b76fbd560fbe13f0ada9507fef12a5bc38d3b0bf6dd84e44.jpg)

![0b9b74b4e2bc49f871091260174a268bf40a3614dd1cb44470f1d3b51e65c47d.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/0b9b74b4e2bc49f871091260174a268bf40a3614dd1cb44470f1d3b51e65c47d.jpg)

![171c475eb9addd28eb501eac517fb2e7db0792b18d629c2914524a329ffb54a2.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/171c475eb9addd28eb501eac517fb2e7db0792b18d629c2914524a329ffb54a2.jpg)

![3df3e60a985175515f9ef6230ef1e0bfd1fcdb4d80425b4c77b49959d28c3767.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/3df3e60a985175515f9ef6230ef1e0bfd1fcdb4d80425b4c77b49959d28c3767.jpg)

![af4f8df3ef7a8b02c11db6cbe1e2f8149dd2c663f1d0f9af03cadbc3fa68e77e.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/af4f8df3ef7a8b02c11db6cbe1e2f8149dd2c663f1d0f9af03cadbc3fa68e77e.jpg)

![c5a39b5a3d7dff5168fe9c6eced79143588f9082b708dbe73a8c4f5853428ec8.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/c5a39b5a3d7dff5168fe9c6eced79143588f9082b708dbe73a8c4f5853428ec8.jpg)

![ceff814049ecc8f3a94127566e9fa5f52ae0c0e2677a409f01691a26ea8cc512.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/ceff814049ecc8f3a94127566e9fa5f52ae0c0e2677a409f01691a26ea8cc512.jpg)

![f32a7d05da6b353d5c3a1e526af982c93b43463de6618f98691b0ed7a3f4e153.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/images/f32a7d05da6b353d5c3a1e526af982c93b43463de6618f98691b0ed7a3f4e153.jpg)

### Tables

![143ceae1907d3d213224c62580c5e0abd8afe1f98f95f0e8a14061c003d87860.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/143ceae1907d3d213224c62580c5e0abd8afe1f98f95f0e8a14061c003d87860.jpg)

![284ab2d68020cfc1234cab74cc0e972631cd598e2dfc8f79d87a09a0f2198e59.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/284ab2d68020cfc1234cab74cc0e972631cd598e2dfc8f79d87a09a0f2198e59.jpg)

![28d26b3d2dcc345d9ceea07459bcfe615f73ddc2ae1624bd5b06cd31dd616a2e.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/28d26b3d2dcc345d9ceea07459bcfe615f73ddc2ae1624bd5b06cd31dd616a2e.jpg)

![73b26ab6d9c0dc487c858b9877a2d3865bf4d6469e2622be6300d380a1cc2ade.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/73b26ab6d9c0dc487c858b9877a2d3865bf4d6469e2622be6300d380a1cc2ade.jpg)

![7f97f99726f57daaae8599ae0159e86449369bdff1fa16b72b771d81aa1605ad.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/7f97f99726f57daaae8599ae0159e86449369bdff1fa16b72b771d81aa1605ad.jpg)

![925e44f72025da73616f13d20fae4d2fad1f812c2f58ac3619604d4d0ef09431.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/925e44f72025da73616f13d20fae4d2fad1f812c2f58ac3619604d4d0ef09431.jpg)

![eb52388c86f512915b7f014389365596ca0ed6cea229bb4f30f92c1c25a77b92.jpg](../icml_results/5957_GeoMoLa_%20Geometry-Aware%20Motion%20Latents%20for%20Learning%20Robust%20Manipulation%20Policies/tables/eb52388c86f512915b7f014389365596ca0ed6cea229bb4f30f92c1c25a77b92.jpg)

<a id="paper-5909"></a>

## Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation


### Images

![14cf4499937c7ec905d42220ad21ee79cfc4c8d51e5362c056455a7d33b1332d.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/14cf4499937c7ec905d42220ad21ee79cfc4c8d51e5362c056455a7d33b1332d.jpg)

![1fdebddd0654a95752e33a7c05e9e00ec9247d6eb13c4805da6a067e632aac97.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/1fdebddd0654a95752e33a7c05e9e00ec9247d6eb13c4805da6a067e632aac97.jpg)

![45350e140f29b64c96cd83d86edc088bcb25f2c63029757c42ddca6fedc776ca.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/45350e140f29b64c96cd83d86edc088bcb25f2c63029757c42ddca6fedc776ca.jpg)

![52a39e82b285c1db2cab84bf2a655cc60dec0361d1614b729cc50e16e3393fdd.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/52a39e82b285c1db2cab84bf2a655cc60dec0361d1614b729cc50e16e3393fdd.jpg)

![638a82e9caa124a474887697686f595d64205cf164a4d7fd4474849b43e0b9ae.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/638a82e9caa124a474887697686f595d64205cf164a4d7fd4474849b43e0b9ae.jpg)

![87a5e62c8355a787d7fcc2be10ed089b9c198c45de74706862ce49becffd1e79.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/87a5e62c8355a787d7fcc2be10ed089b9c198c45de74706862ce49becffd1e79.jpg)

![8a68b1079613036b734820f4e4aed0fa69d876ef5c46e5b86d9db86d392ac625.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/8a68b1079613036b734820f4e4aed0fa69d876ef5c46e5b86d9db86d392ac625.jpg)

![d1943f7582d7e48e8477760ddc9af90c2c776327b76ee0b770143872cfe233f7.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/images/d1943f7582d7e48e8477760ddc9af90c2c776327b76ee0b770143872cfe233f7.jpg)

### Tables

![01cc669b1de9fc18677a5f83d4444c1bc5f15267a4aaa67b3d26b2fbbbc5f785.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/tables/01cc669b1de9fc18677a5f83d4444c1bc5f15267a4aaa67b3d26b2fbbbc5f785.jpg)

![802fdf706cd9aa9888f53c42f0b48ad82b721cc8eaba4a8d8e49a623da5f0add.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/tables/802fdf706cd9aa9888f53c42f0b48ad82b721cc8eaba4a8d8e49a623da5f0add.jpg)

![a2f1efce68deedebc2fa7db392a4edad88b2b361828396b6b7d09f6d1fe7a335.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/tables/a2f1efce68deedebc2fa7db392a4edad88b2b361828396b6b7d09f6d1fe7a335.jpg)

![a860c982dedc5e31135376c4389c809d41560ee0dcef2c277c4080184d3e6788.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/tables/a860c982dedc5e31135376c4389c809d41560ee0dcef2c277c4080184d3e6788.jpg)

![aaa5cff954f38ee88e7483702b8cfa17136dd8a5614cd57cb6ecf0a100fec0ee.jpg](../icml_results/5958_Decompose%20and%20Recompose_%20Reasoning%20New%20Skills%20from%20Existing%20Abilities%20for%20Cross-Task%20Robotic%20Manipul/tables/aaa5cff954f38ee88e7483702b8cfa17136dd8a5614cd57cb6ecf0a100fec0ee.jpg)

<a id="paper-5910"></a>

## Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks


### Images

![01264f12f6445d9d98be52b8143355c354ac2b241731bbdcaf1bb1f867c5df6e.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/01264f12f6445d9d98be52b8143355c354ac2b241731bbdcaf1bb1f867c5df6e.jpg)

![1f1d5405d5aa35317b43be268cbbd816b4b5008167778d1eb5d1f26cd9e7e5dd.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/1f1d5405d5aa35317b43be268cbbd816b4b5008167778d1eb5d1f26cd9e7e5dd.jpg)

![6d746166fca1c97d77842e5d1eef044cc418368b1a6e0865b98954207bf9ba81.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/6d746166fca1c97d77842e5d1eef044cc418368b1a6e0865b98954207bf9ba81.jpg)

![7eec9f8fbfeeac5bc1deb19daf520d56c2a785183b408ea4c55b0e829bfb5090.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/7eec9f8fbfeeac5bc1deb19daf520d56c2a785183b408ea4c55b0e829bfb5090.jpg)

![95d0e7a8e54c124afe9af4fb14896dd35c0d5da7af99d98e7f17415c2ffd278a.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/95d0e7a8e54c124afe9af4fb14896dd35c0d5da7af99d98e7f17415c2ffd278a.jpg)

![e748bf40fabac38d614558655940719a6b4af67401fe6a3a8f630c7a4ae0392b.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/e748bf40fabac38d614558655940719a6b4af67401fe6a3a8f630c7a4ae0392b.jpg)

![eda60042da7749b911b1f17ce460037e81a07c0f4185eb733e6215d17377a3f9.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/images/eda60042da7749b911b1f17ce460037e81a07c0f4185eb733e6215d17377a3f9.jpg)

### Tables

![156c99556449e90b1fc5d8835e9a1d5af3afdeeacb42738cf75e11f4fdd9ecd9.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/156c99556449e90b1fc5d8835e9a1d5af3afdeeacb42738cf75e11f4fdd9ecd9.jpg)

![32d7dbf0550ecdc9c58944056131a7950cf8c68c895612dfdcc136bbb1d637de.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/32d7dbf0550ecdc9c58944056131a7950cf8c68c895612dfdcc136bbb1d637de.jpg)

![51567f88f782e67238a64d44b9ffeaf8b41daf2aa97aa63d40c456185437c856.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/51567f88f782e67238a64d44b9ffeaf8b41daf2aa97aa63d40c456185437c856.jpg)

![60657bc8645fc629a30a5666fc2386f176ff6a12ee573fd581b8bbc295aa101d.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/60657bc8645fc629a30a5666fc2386f176ff6a12ee573fd581b8bbc295aa101d.jpg)

![6a53412c112c335f841672e3ca668b2f5e895110c06ff0ff8fe37e3eebbd9b8a.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/6a53412c112c335f841672e3ca668b2f5e895110c06ff0ff8fe37e3eebbd9b8a.jpg)

![7353b1c112de944c0ee432aa1adea7b4342686351bd3d1d1abdfab682ad42bb3.jpg](../icml_results/5959_Hyperparameter%20Transfer%20Laws%20for%20Non-Recurrent%20Multi-Path%20Neural%20Networks/tables/7353b1c112de944c0ee432aa1adea7b4342686351bd3d1d1abdfab682ad42bb3.jpg)

<a id="paper-5911"></a>

## DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving


### Images

![05199884cefb780ad26517e3f367562cc75178a819e5ebc68d3c1e5bca07c7b3.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/05199884cefb780ad26517e3f367562cc75178a819e5ebc68d3c1e5bca07c7b3.jpg)

![7e545e68e6d3f1348d94abc17fed4a8e6114a288896d9740bd11395c3603159c.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/7e545e68e6d3f1348d94abc17fed4a8e6114a288896d9740bd11395c3603159c.jpg)

![9e930f9703a76c24106be9f4e91e5303c9b557e0041fd328200b960b291a83f4.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/9e930f9703a76c24106be9f4e91e5303c9b557e0041fd328200b960b291a83f4.jpg)

![a0111b00dd31b80f2e04ae198ba3f5574b24b8b2156d5c441bfb1db15808e97e.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/a0111b00dd31b80f2e04ae198ba3f5574b24b8b2156d5c441bfb1db15808e97e.jpg)

![d651e9179e51ba7cafe49d51685cc08ee6d2638b5e329a052f0a261a424e44a0.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/d651e9179e51ba7cafe49d51685cc08ee6d2638b5e329a052f0a261a424e44a0.jpg)

![ed37ee59bbb19c0de5c05204304953fcbd9851df79c62b55417afadd4ad16dd3.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/images/ed37ee59bbb19c0de5c05204304953fcbd9851df79c62b55417afadd4ad16dd3.jpg)

### Tables

![09789d850687cea6c18c1b140406f61aad2c287f1ffff725fa0288d774ef1ae6.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/09789d850687cea6c18c1b140406f61aad2c287f1ffff725fa0288d774ef1ae6.jpg)

![0de96d158774492dc0ec2e6ac1f070be00bf538b9245f7e7a3b4b6a83bed6b19.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/0de96d158774492dc0ec2e6ac1f070be00bf538b9245f7e7a3b4b6a83bed6b19.jpg)

![5b506259a83beece377644c0233761795084d6d2ebe4651b24406398e544dbba.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/5b506259a83beece377644c0233761795084d6d2ebe4651b24406398e544dbba.jpg)

![707597df7322cf113c955d8e751151fa8d13122742dd0384526dfd86cdf2ed83.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/707597df7322cf113c955d8e751151fa8d13122742dd0384526dfd86cdf2ed83.jpg)

![83e1d63fe284d2ac6a1121c5ec0f7e1cb333bfe14cdfc0f82b071ffbbdbfe1d4.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/83e1d63fe284d2ac6a1121c5ec0f7e1cb333bfe14cdfc0f82b071ffbbdbfe1d4.jpg)

![9f42de4b4c235f410a66b92318e5153f231fc785d691e778e54cdf865fc26086.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/9f42de4b4c235f410a66b92318e5153f231fc785d691e778e54cdf865fc26086.jpg)

![a052b57d8102e071b8719a51d64b75579521b89e03d93e0fe946ca14fb5f1d0a.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/a052b57d8102e071b8719a51d64b75579521b89e03d93e0fe946ca14fb5f1d0a.jpg)

![b1bea0eb2753ab925c87471d667d53cfa573542771ea5df46c53389272bf3a3f.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/b1bea0eb2753ab925c87471d667d53cfa573542771ea5df46c53389272bf3a3f.jpg)

![e0e51b138876329d8141bd278b097fc4e16f81d1ca27c94efb54eee333f03fef.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/e0e51b138876329d8141bd278b097fc4e16f81d1ca27c94efb54eee333f03fef.jpg)

![e52ad6dc09e02e43e796b5d26f0e71738e459bfdfdee256694f0ffbbeba64048.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/e52ad6dc09e02e43e796b5d26f0e71738e459bfdfdee256694f0ffbbeba64048.jpg)

![f9218f9e0ed5211de876f3c692f4870a6bb1f6370c23191527537b6c618e3124.jpg](../icml_results/5960_DeepSight_%20Long-Horizon%20World%20Modeling%20via%20Latent%20States%20Prediction%20for%20End-to-End%20Autonomous%20Drivin/tables/f9218f9e0ed5211de876f3c692f4870a6bb1f6370c23191527537b6c618e3124.jpg)

<a id="paper-5912"></a>

## Reference-Free Meta-Learning for Generalized Implicit Neural Representation in Efficient MRI Reconstruction


### Images

![154fd82057c84dc1126047d007869663fd4bd3f76bc585e2717de05ec2873468.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/154fd82057c84dc1126047d007869663fd4bd3f76bc585e2717de05ec2873468.jpg)

![21d342028ba890c182fb069fed1fe4eff5a1f01794b9af398012f6cb588f5c8c.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/21d342028ba890c182fb069fed1fe4eff5a1f01794b9af398012f6cb588f5c8c.jpg)

![37469ecb7d611c8b217d62a0f300180dfde97be56a8cdcbc8b0128cc22042561.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/37469ecb7d611c8b217d62a0f300180dfde97be56a8cdcbc8b0128cc22042561.jpg)

![5ef686f5457fae46948682a10381f8bbde7bb4f79236c81643b8b3c53eeeb407.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/5ef686f5457fae46948682a10381f8bbde7bb4f79236c81643b8b3c53eeeb407.jpg)

![619df71287d50c64c23b8437277f105490b5f2709c3eb433d1f1145c3db4eb10.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/619df71287d50c64c23b8437277f105490b5f2709c3eb433d1f1145c3db4eb10.jpg)

![655cc2fa0f06eea42a23f227a27f1eef8e2bad11db3b67be7e787c1e40d6967c.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/655cc2fa0f06eea42a23f227a27f1eef8e2bad11db3b67be7e787c1e40d6967c.jpg)

![68aff5cd0bedda83617ec0a46f4a2009f58d9a476695db60e974c33ca2ef3796.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/68aff5cd0bedda83617ec0a46f4a2009f58d9a476695db60e974c33ca2ef3796.jpg)

![a7c7412568692afd60967612eeb8c3951c5ddcf2cd523d22582e072499781c99.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/a7c7412568692afd60967612eeb8c3951c5ddcf2cd523d22582e072499781c99.jpg)

![b5b248582b3e94816dcf672a87560d16a437561344da2441313dc4dd35afe5fa.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/b5b248582b3e94816dcf672a87560d16a437561344da2441313dc4dd35afe5fa.jpg)

![cb89eb6b8e27453927701f78221ccf5fe8671485e9ee8bf71cf535b568f07bf7.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/cb89eb6b8e27453927701f78221ccf5fe8671485e9ee8bf71cf535b568f07bf7.jpg)

![d328188b8d955b6f08a31eb21a62c876173c4bc46c8a757ec800a77b2c6eddc3.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/d328188b8d955b6f08a31eb21a62c876173c4bc46c8a757ec800a77b2c6eddc3.jpg)

![dac31e59ce82c7d11eab1bd9e236751c52b3c87431cacbf94e309f9d5707c19d.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/dac31e59ce82c7d11eab1bd9e236751c52b3c87431cacbf94e309f9d5707c19d.jpg)

![de58654b4c845e960426e20335dfe2d22de8d7c8a9afdd83f0a40031c642e594.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/de58654b4c845e960426e20335dfe2d22de8d7c8a9afdd83f0a40031c642e594.jpg)

![e43f6452bf7945122ffa1bb26dc2950aba1c5b399bb98f9972aa53baaef0089b.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/images/e43f6452bf7945122ffa1bb26dc2950aba1c5b399bb98f9972aa53baaef0089b.jpg)

### Tables

![44b23bb840aadfb87142754f7c2d7d641ec7aaf04738cef738283c2c83c0bc6e.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/44b23bb840aadfb87142754f7c2d7d641ec7aaf04738cef738283c2c83c0bc6e.jpg)

![52eeb063569aa7976551ed07e53d4c03c6d87aa48c95c7664bb6497599ea7cb1.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/52eeb063569aa7976551ed07e53d4c03c6d87aa48c95c7664bb6497599ea7cb1.jpg)

![628fc618a54458c7f36ba2b9cd3ba32db8d3aa82cc382feae7e0f44e77773a0a.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/628fc618a54458c7f36ba2b9cd3ba32db8d3aa82cc382feae7e0f44e77773a0a.jpg)

![71ba9957e8120757d9780d3bb5c212b620449393e3114cff3fdc2e6153904659.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/71ba9957e8120757d9780d3bb5c212b620449393e3114cff3fdc2e6153904659.jpg)

![7493af32c4227082fcda0bd1f1f1866efd0c56273ac7ed0cdf2ebb800a51e24a.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/7493af32c4227082fcda0bd1f1f1866efd0c56273ac7ed0cdf2ebb800a51e24a.jpg)

![d27cd3bba5be3b41e307a03ef002ab56a38a17af60526eda5a8ea532abe68598.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/d27cd3bba5be3b41e307a03ef002ab56a38a17af60526eda5a8ea532abe68598.jpg)

![e35f0e3370b0504310ef4c416753dac183628e3a63c8d7289581b25abd6ffabb.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/e35f0e3370b0504310ef4c416753dac183628e3a63c8d7289581b25abd6ffabb.jpg)

![fe8d084f9c908f0d6b89abeea2fceecf64b759048a427ec9b55e54703fff821a.jpg](../icml_results/5961_Reference-Free%20Meta-Learning%20for%20Generalized%20Implicit%20Neural%20Representation%20in%20Efficient%20MRI%20Reconst/tables/fe8d084f9c908f0d6b89abeea2fceecf64b759048a427ec9b55e54703fff821a.jpg)

<a id="paper-5913"></a>

## Utonia: Toward One Encoder for All Point Clouds


### Images

![39cf7112ae2c767a372ee0735d8edd4c9edaf9a32a19289b6f2da0b4c5123e92.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/39cf7112ae2c767a372ee0735d8edd4c9edaf9a32a19289b6f2da0b4c5123e92.jpg)

![6535e3abb4f0e2807e4bfa911c4c3396d7de2dd67c31eba3d1a200b4f74d55c3.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/6535e3abb4f0e2807e4bfa911c4c3396d7de2dd67c31eba3d1a200b4f74d55c3.jpg)

![664dab9652b147a4dab0b9ff9790639b895c9bd6bed353a22edb82a29ab88f48.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/664dab9652b147a4dab0b9ff9790639b895c9bd6bed353a22edb82a29ab88f48.jpg)

![7e5e73bcadc9c69d1ca28f34460d4e7648303686647a8f15f614c3b6811f4ecf.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/7e5e73bcadc9c69d1ca28f34460d4e7648303686647a8f15f614c3b6811f4ecf.jpg)

![a3e16b0e6e7ff4c79579c44ffba6eb7372c1c421644ec469a5e6228d3e8a2ecf.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/a3e16b0e6e7ff4c79579c44ffba6eb7372c1c421644ec469a5e6228d3e8a2ecf.jpg)

![e7d5d0a8edaa6beaaafedf041219c90aa62df23c2ed3b9010d402d461c01235b.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/images/e7d5d0a8edaa6beaaafedf041219c90aa62df23c2ed3b9010d402d461c01235b.jpg)

### Tables

![0b66cf55679eb592bee19d15e30df278aff409ff719aaaa0e67b62c64226034c.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/0b66cf55679eb592bee19d15e30df278aff409ff719aaaa0e67b62c64226034c.jpg)

![1b3e33cdff58ff13e079b4c43ee98e918cac265c16e39288060b5f3aad6b3865.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/1b3e33cdff58ff13e079b4c43ee98e918cac265c16e39288060b5f3aad6b3865.jpg)

![232c913d4ed4a4399bb1bc38d3b88436d138360da2885c44188eac3606d26b12.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/232c913d4ed4a4399bb1bc38d3b88436d138360da2885c44188eac3606d26b12.jpg)

![2455003b8ac3f1c08697f64c9d6cec7807fe7070615e92d0aaf916b72766dfec.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/2455003b8ac3f1c08697f64c9d6cec7807fe7070615e92d0aaf916b72766dfec.jpg)

![248e7c5187432c1659f7668256349dcff82d499eb8122380470bee13d6a2e192.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/248e7c5187432c1659f7668256349dcff82d499eb8122380470bee13d6a2e192.jpg)

![263ff8c95bb0b27c02658ae540e161beaa7d4810781a803fda2f841c7e0317ba.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/263ff8c95bb0b27c02658ae540e161beaa7d4810781a803fda2f841c7e0317ba.jpg)

![2edc7cf66888f091f0e3543d3613f0d4868add1ac599a0e48ede2a10ccdb46e0.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/2edc7cf66888f091f0e3543d3613f0d4868add1ac599a0e48ede2a10ccdb46e0.jpg)

![418bae048df333a98a9536a37f13c7f556a6a98ad39bf53e404e1225344baab3.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/418bae048df333a98a9536a37f13c7f556a6a98ad39bf53e404e1225344baab3.jpg)

![7a441f9c9b8b6a5db801edb77903b88efc0eaaf08e5c6358e30d82684e33f044.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/7a441f9c9b8b6a5db801edb77903b88efc0eaaf08e5c6358e30d82684e33f044.jpg)

![8eb5f6da65a1c94618ed9e36832c47b4e156b02211ae1790aef4e50a6541057a.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/8eb5f6da65a1c94618ed9e36832c47b4e156b02211ae1790aef4e50a6541057a.jpg)

![97e5c7bedcbd813440f4e577771a562e5cf9a8b8a3dbf12e36f24aa930393ad4.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/97e5c7bedcbd813440f4e577771a562e5cf9a8b8a3dbf12e36f24aa930393ad4.jpg)

![9e15195fc039669ee4754eff6d3f4eda6fbd97b6556ec6f16569b8e537fb54cc.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/9e15195fc039669ee4754eff6d3f4eda6fbd97b6556ec6f16569b8e537fb54cc.jpg)

![ac5e9a6575885c23b16b501dc33e9aa0a0ae2ed7ea35fe9be0c29bdf359c850b.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/ac5e9a6575885c23b16b501dc33e9aa0a0ae2ed7ea35fe9be0c29bdf359c850b.jpg)

![e0ed76225252cb3bbc15ed89038f9f67257a4dce6c739518e5cddcdb344e73a5.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/e0ed76225252cb3bbc15ed89038f9f67257a4dce6c739518e5cddcdb344e73a5.jpg)

![e753de0776f2ded271a16be14fdd7f9ce211b9859b80d4a138f977bbe46a290a.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/e753de0776f2ded271a16be14fdd7f9ce211b9859b80d4a138f977bbe46a290a.jpg)

![ea9be86b1e0119b5868126decac75682e1dac4b363a4eb6be5ab25759a6c1d15.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/ea9be86b1e0119b5868126decac75682e1dac4b363a4eb6be5ab25759a6c1d15.jpg)

![fd642c8a566c4d97934c6c8109975af2a9b97da9cbdeeeafeb8aecfd3e8d2991.jpg](../icml_results/5962_Utonia_%20Toward%20One%20Encoder%20for%20All%20Point%20Clouds/tables/fd642c8a566c4d97934c6c8109975af2a9b97da9cbdeeeafeb8aecfd3e8d2991.jpg)

<a id="paper-5914"></a>

## dTRPO : Trajectory Reduction in Policy Optimization of Diffusion Large Language Models


### Images

![0060112c6f43542d8e5c7185ba5b24ad593a5b80d59408bd05c7302ba5ffefe2.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/0060112c6f43542d8e5c7185ba5b24ad593a5b80d59408bd05c7302ba5ffefe2.jpg)

![030d0177f97a0ee1939210d8708728f2c2d240be34b66a1f476506baa42c935e.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/030d0177f97a0ee1939210d8708728f2c2d240be34b66a1f476506baa42c935e.jpg)

![16f9350670774c52f5b991a1ae075ba75fcd324b40d7f30190fc78c6d5dc9388.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/16f9350670774c52f5b991a1ae075ba75fcd324b40d7f30190fc78c6d5dc9388.jpg)

![a47ebf06b451c3ef51fef9472d714802f2eea8e7abb82be1d5ed41940fd0f933.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/a47ebf06b451c3ef51fef9472d714802f2eea8e7abb82be1d5ed41940fd0f933.jpg)

![f061572e62c5ffedf7a368f838a0c875ee6fadf043e94d683404ebe91d648847.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/f061572e62c5ffedf7a368f838a0c875ee6fadf043e94d683404ebe91d648847.jpg)

![f675ca72080a4b0b74121c044957bfed58374fc37704c0f2a0f8eaef9e774fef.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/images/f675ca72080a4b0b74121c044957bfed58374fc37704c0f2a0f8eaef9e774fef.jpg)

### Tables

![0786732abd385168d9e6e451433f828ff28f434a9166722fe237f22c60c9df2d.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/0786732abd385168d9e6e451433f828ff28f434a9166722fe237f22c60c9df2d.jpg)

![12bd73b457ab52d91ecf254f6e209041a59865cd8bf982bcaa59bce8665dd6a4.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/12bd73b457ab52d91ecf254f6e209041a59865cd8bf982bcaa59bce8665dd6a4.jpg)

![2eba03b33c576bb0b15a68169eb1dcc461b087c298358f6a70cd12663b9ef16e.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/2eba03b33c576bb0b15a68169eb1dcc461b087c298358f6a70cd12663b9ef16e.jpg)

![74c21853f99732fc7e13edf3ef9539d119b7b0d5e8bcece3e7b06b88ac053be4.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/74c21853f99732fc7e13edf3ef9539d119b7b0d5e8bcece3e7b06b88ac053be4.jpg)

![89c0cd2fa8288c66b81a3e3057d9bd8a5c9705e559a2453744922837e1e3fffb.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/89c0cd2fa8288c66b81a3e3057d9bd8a5c9705e559a2453744922837e1e3fffb.jpg)

![96a47affc785720224d736863aa4d8cba91b8f201818ed4832f7bc381f14b377.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/96a47affc785720224d736863aa4d8cba91b8f201818ed4832f7bc381f14b377.jpg)

![a1819e8cc75327e2009df4007f6137257f33ec3aa94d6c2b1bfaa266e89cf5ad.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/a1819e8cc75327e2009df4007f6137257f33ec3aa94d6c2b1bfaa266e89cf5ad.jpg)

![fe70ede5fe7b7efa579627ffbea44cd63372cbd13b5c3eadf4f2544f121de6c3.jpg](../icml_results/5963_dTRPO%20_%20Trajectory%20Reduction%20in%20Policy%20Optimization%20of%20Diffusion%20Large%20Language%20Models/tables/fe70ede5fe7b7efa579627ffbea44cd63372cbd13b5c3eadf4f2544f121de6c3.jpg)

<a id="paper-5915"></a>

## See First, Reason Later: Mutual Information-Guided Reinforcement Learning for Vision-Language Models


### Images

![1998849678ac623d15bc5353de25c195f5004399bb9685875e1f6d95c161982c.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/images/1998849678ac623d15bc5353de25c195f5004399bb9685875e1f6d95c161982c.jpg)

![1c379298fbd0b8caea5f6daf1a9f5dd7308fa44522c96f9f70bd6c8dca4028ff.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/images/1c379298fbd0b8caea5f6daf1a9f5dd7308fa44522c96f9f70bd6c8dca4028ff.jpg)

![2f1c4da6f6027e5e0feeb3768dc0219f97e3c806efb3bf2c3e8a5cd2e02855a5.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/images/2f1c4da6f6027e5e0feeb3768dc0219f97e3c806efb3bf2c3e8a5cd2e02855a5.jpg)

![711d2ce7dff87fd01f07ac6f311a7dba73db1058ecad105bcd7b57e21401ab65.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/images/711d2ce7dff87fd01f07ac6f311a7dba73db1058ecad105bcd7b57e21401ab65.jpg)

### Tables

![07ece21e22f566841904f37dd53fea153734f968c902c06216082cfa41446c7d.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/tables/07ece21e22f566841904f37dd53fea153734f968c902c06216082cfa41446c7d.jpg)

![3cec0bd4c472db7e5db039b9ffc0e575e98c1580b94fc006842c8464a23095f3.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/tables/3cec0bd4c472db7e5db039b9ffc0e575e98c1580b94fc006842c8464a23095f3.jpg)

![868c6e30b4920c417cbd6253d07e211503a6d69a5fc57e0ad550809c4fbfd733.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/tables/868c6e30b4920c417cbd6253d07e211503a6d69a5fc57e0ad550809c4fbfd733.jpg)

![b7b43dfb86dad7650dbdb08a344fa0da5eb5a6dfa4fd7c79b2dc7bda05e6a82a.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/tables/b7b43dfb86dad7650dbdb08a344fa0da5eb5a6dfa4fd7c79b2dc7bda05e6a82a.jpg)

![e354a31d8912d0a4d330b5474ee6fea8dc194ca0811db2bd8334e5c77e404de3.jpg](../icml_results/5964_See%20First%2C%20Reason%20Later_%20Mutual%20Information-Guided%20Reinforcement%20Learning%20for%20Vision-Language%20Models/tables/e354a31d8912d0a4d330b5474ee6fea8dc194ca0811db2bd8334e5c77e404de3.jpg)

<a id="paper-5916"></a>

## MUSA-PINN: Multi-scale Weak-form Physics-Informed Neural Networks for Fluid Flow in Complex Geometries


### Images

![1bc83c7dbae10708aca65754e03d3e92f0188d873b3938f5700d21083681bb13.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/1bc83c7dbae10708aca65754e03d3e92f0188d873b3938f5700d21083681bb13.jpg)

![3097f9b4bb8dc04e191bd0e78ed8124a74a7600d7946d9baa76f495ba36f1b29.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/3097f9b4bb8dc04e191bd0e78ed8124a74a7600d7946d9baa76f495ba36f1b29.jpg)

![383eedf504155b1ac271b8bb4195990281169c854acb2084d6e07d4be1fc5d12.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/383eedf504155b1ac271b8bb4195990281169c854acb2084d6e07d4be1fc5d12.jpg)

![3c26e5e57483b2596f50e6d1b4b99f30e9f9467aa0885f8362b107b43d8a6b55.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/3c26e5e57483b2596f50e6d1b4b99f30e9f9467aa0885f8362b107b43d8a6b55.jpg)

![45e23a53ff5e2614a8a3c243903ff79099d158a618f19abb39c63fcfc481996f.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/45e23a53ff5e2614a8a3c243903ff79099d158a618f19abb39c63fcfc481996f.jpg)

![6051555acae5c59f5166de9335d335272e111a934b57593e1d8728077472c334.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/6051555acae5c59f5166de9335d335272e111a934b57593e1d8728077472c334.jpg)

![6f86c364f0df93b768d733e6127834395d65ce5c4262aed486f5994a9d40c12a.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/6f86c364f0df93b768d733e6127834395d65ce5c4262aed486f5994a9d40c12a.jpg)

![72405b5f0dd25bdf8970803c687476a9886619b0279ecdc03ce3144b5f6a0bc0.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/72405b5f0dd25bdf8970803c687476a9886619b0279ecdc03ce3144b5f6a0bc0.jpg)

![749af00be1bd07966eb9ceed5c411ffad74403a37634f4ff102bda8135031b58.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/749af00be1bd07966eb9ceed5c411ffad74403a37634f4ff102bda8135031b58.jpg)

![b92972ab2ae825a1d53971aac376653def8d3813c9602dc8a015cee320fcdd13.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/b92972ab2ae825a1d53971aac376653def8d3813c9602dc8a015cee320fcdd13.jpg)

![b9ea0e60ce95a65bbaed5b264db5c7046b854bb19001a90244170c9059886185.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/b9ea0e60ce95a65bbaed5b264db5c7046b854bb19001a90244170c9059886185.jpg)

![bc56821fa5ab27b0bc69673b44789f3b5e06404c54446e2f8de491c055337865.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/bc56821fa5ab27b0bc69673b44789f3b5e06404c54446e2f8de491c055337865.jpg)

![dbe86a47ab50a0976defbccf6415612eb7aaa2d48feb184d445e27e721d2c5c6.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/dbe86a47ab50a0976defbccf6415612eb7aaa2d48feb184d445e27e721d2c5c6.jpg)

![ed2d09a7578918f6912682c01e40f8f15155c4d7c06df7cd7bcd758f643aec8c.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/ed2d09a7578918f6912682c01e40f8f15155c4d7c06df7cd7bcd758f643aec8c.jpg)

![f3c90d015d5b22c2dcd48648ab53c1fba38dff020d81556b11c9239a01c1d5f2.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/images/f3c90d015d5b22c2dcd48648ab53c1fba38dff020d81556b11c9239a01c1d5f2.jpg)

### Tables

![068f0a26709738626de697a5650258c7a6dffc5d253b070d181dff839d24690f.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/068f0a26709738626de697a5650258c7a6dffc5d253b070d181dff839d24690f.jpg)

![1c682c99e8eae65390db68c80692a70d78f8a59e976d4fc5a12e37e2051e8b1f.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/1c682c99e8eae65390db68c80692a70d78f8a59e976d4fc5a12e37e2051e8b1f.jpg)

![81712cf841073c572899fdef8eabcf39fee673699d6c01c4fa61f5ec0b108d8f.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/81712cf841073c572899fdef8eabcf39fee673699d6c01c4fa61f5ec0b108d8f.jpg)

![8619cbc461314cdde71fdb9772164efe85c3fc30b3c9ac1355b195b0cb0bedc1.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/8619cbc461314cdde71fdb9772164efe85c3fc30b3c9ac1355b195b0cb0bedc1.jpg)

![9afc8e2cff68baeb9a63700811ae5fb4ab4d5f6fea48f06bf6b086f226dd139d.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/9afc8e2cff68baeb9a63700811ae5fb4ab4d5f6fea48f06bf6b086f226dd139d.jpg)

![ddb1a350a50740423f9f88f6983035cba3e225b67f0fa5ba314ec86758a6895a.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/ddb1a350a50740423f9f88f6983035cba3e225b67f0fa5ba314ec86758a6895a.jpg)

![e11fce3e4c36ab8e259da8b321751650f1e1dc17983a0d1dd2385a3fa763b93e.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/e11fce3e4c36ab8e259da8b321751650f1e1dc17983a0d1dd2385a3fa763b93e.jpg)

![f256ceba7cb4d189ff89c7d18d8ac8781053265db9a31e0968613e7f90fe2023.jpg](../icml_results/5965_MUSA-PINN_%20Multi-scale%20Weak-form%20Physics-Informed%20Neural%20Networks%20for%20Fluid%20Flow%20in%20Complex%20Geometri/tables/f256ceba7cb4d189ff89c7d18d8ac8781053265db9a31e0968613e7f90fe2023.jpg)

<a id="paper-5917"></a>

## AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching


### Images

![486bb7662259378b6ba86a14867fa8f7428edd7130631a1ba9c711ca43b66d5f.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/images/486bb7662259378b6ba86a14867fa8f7428edd7130631a1ba9c711ca43b66d5f.jpg)

![5fe8cff2ff376a0673d8a18d2954deba5624aa92ead6f4fe9049092feba4446b.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/images/5fe8cff2ff376a0673d8a18d2954deba5624aa92ead6f4fe9049092feba4446b.jpg)

![71a97b3b588eac2d0d7f5b1cc5252081cc5274c317b0085a436b02a391607cbe.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/images/71a97b3b588eac2d0d7f5b1cc5252081cc5274c317b0085a436b02a391607cbe.jpg)

![ca95f0492b35c8a9bc9c3dfeb1a4912c0bd7c48c2c42be48b766d6b53ac5235e.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/images/ca95f0492b35c8a9bc9c3dfeb1a4912c0bd7c48c2c42be48b766d6b53ac5235e.jpg)

### Tables

![269ebd83109d3dadc5ba04dee799464025aa18b6bb4f1aefe53d7278acc98e85.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/269ebd83109d3dadc5ba04dee799464025aa18b6bb4f1aefe53d7278acc98e85.jpg)

![28e94651d950807d525fe5c971a4928b2fa611c083d127efb5444c69212080f0.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/28e94651d950807d525fe5c971a4928b2fa611c083d127efb5444c69212080f0.jpg)

![2f6996cb191fe3f30c87b87abe50699758c1c30ec5b98fc175b71e951e6d2aa9.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/2f6996cb191fe3f30c87b87abe50699758c1c30ec5b98fc175b71e951e6d2aa9.jpg)

![371c4984534a874021ad33c99f162f521118cf46063972d2f18f417c13f152a5.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/371c4984534a874021ad33c99f162f521118cf46063972d2f18f417c13f152a5.jpg)

![51fb3fa069f2b09c9f75d09a291af9d44017ab1e3110fc1e006d3e05b8650b93.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/51fb3fa069f2b09c9f75d09a291af9d44017ab1e3110fc1e006d3e05b8650b93.jpg)

![8f8ff7694f9e250538d7c96577e5cdb10a97db8846496c0bb0ebee44f2c1538e.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/8f8ff7694f9e250538d7c96577e5cdb10a97db8846496c0bb0ebee44f2c1538e.jpg)

![aaccd2a22de73cbd8af67341a0e6303080d7396b780aa5e668b31be51c4cacba.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/aaccd2a22de73cbd8af67341a0e6303080d7396b780aa5e668b31be51c4cacba.jpg)

![b7078e9ba799d2972474417b51da1f68c40f3dcf541c022111f0d5c77260fcb0.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/b7078e9ba799d2972474417b51da1f68c40f3dcf541c022111f0d5c77260fcb0.jpg)

![d05f85bd16cc19148dcc2d9f69a591782efeb84261a774f8e122b7115e6f0be0.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/d05f85bd16cc19148dcc2d9f69a591782efeb84261a774f8e122b7115e6f0be0.jpg)

![d295b6de0f7952f2a551fca5e9ea853ece3bca12e7932c5073e2926922dabdeb.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/d295b6de0f7952f2a551fca5e9ea853ece3bca12e7932c5073e2926922dabdeb.jpg)

![d4c66eb79a3ec7536b0a2aed89e2c364cb2bbf494dd85a12003f0bc791aa5449.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/d4c66eb79a3ec7536b0a2aed89e2c364cb2bbf494dd85a12003f0bc791aa5449.jpg)

![dc91c426ce03888b04b9b14a42a5af963d187b0389b39d52bf89998765795452.jpg](../icml_results/5966_AG-REPA_%20Causal%20Layer%20Selection%20for%20Representation%20Alignment%20in%20Audio%20Flow%20Matching/tables/dc91c426ce03888b04b9b14a42a5af963d187b0389b39d52bf89998765795452.jpg)

<a id="paper-5918"></a>

## Who Transfers Safety? Identifying and Targeting Cross-Lingual Shared Safety Neurons


### Images

![31af72bc6c89163a9ab44275b2f7275968de7b4bfbb105f660ec621815ddc885.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/images/31af72bc6c89163a9ab44275b2f7275968de7b4bfbb105f660ec621815ddc885.jpg)

![6cb10ee2e2be486b7c45f21b01a478786022679c6be9d0128edbcdc9acfb1f9d.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/images/6cb10ee2e2be486b7c45f21b01a478786022679c6be9d0128edbcdc9acfb1f9d.jpg)

![96fccd719bea951635cde421485edeac2229d02671128becb01363acb1aed062.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/images/96fccd719bea951635cde421485edeac2229d02671128becb01363acb1aed062.jpg)

![b5939c4ec6ec463e49e9eaada15ed1292fe8645d19db4d0ef6568dda7c066e65.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/images/b5939c4ec6ec463e49e9eaada15ed1292fe8645d19db4d0ef6568dda7c066e65.jpg)

![f3b70ba1c20cb0ea0e1c3b0f20ef4dbdbe449add758c80f3c349282b1f589afe.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/images/f3b70ba1c20cb0ea0e1c3b0f20ef4dbdbe449add758c80f3c349282b1f589afe.jpg)

### Tables

![1c3283c754e0d609be1dc9d6772a8675bb302e01adcf15c802ebd6dee275c436.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/1c3283c754e0d609be1dc9d6772a8675bb302e01adcf15c802ebd6dee275c436.jpg)

![2b4928956cc4300040356660f7dfe2d1fd83b5d1e7368b1f0760e01c1a17f706.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/2b4928956cc4300040356660f7dfe2d1fd83b5d1e7368b1f0760e01c1a17f706.jpg)

![33bddeac1c0086af4102e78f7cbbeb1b988bff420ce786757babe029a296e21d.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/33bddeac1c0086af4102e78f7cbbeb1b988bff420ce786757babe029a296e21d.jpg)

![34f6aaa10d52770193db4c3799b94ff73563c338ef1149d20bd5374a954cde3d.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/34f6aaa10d52770193db4c3799b94ff73563c338ef1149d20bd5374a954cde3d.jpg)

![3e14a291dc023668d0617201b423f3041ee8a9631e351867e3354b4e740d24a8.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/3e14a291dc023668d0617201b423f3041ee8a9631e351867e3354b4e740d24a8.jpg)

![459b9eacfec38f60991e80a99bb19edad44449ec61a25291a68bab56fd991980.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/459b9eacfec38f60991e80a99bb19edad44449ec61a25291a68bab56fd991980.jpg)

![679ccea4bd3d5909f4dbabfb1a233d253ff86527684fc327088658ad24b853b0.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/679ccea4bd3d5909f4dbabfb1a233d253ff86527684fc327088658ad24b853b0.jpg)

![8d771c7f0635864db0eda23f8c7092aaf3b33c6a902f81279c59c09430f37548.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/8d771c7f0635864db0eda23f8c7092aaf3b33c6a902f81279c59c09430f37548.jpg)

![93b9f522926ac4d4a5bce8f15dd2d6db2ef5566da3494b3b7025fc2f507ba8c8.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/93b9f522926ac4d4a5bce8f15dd2d6db2ef5566da3494b3b7025fc2f507ba8c8.jpg)

![9b8f9c40ee7d8ac6b6665d8bc4ab37d5233e7f8402109071aa6f648b31a43309.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/9b8f9c40ee7d8ac6b6665d8bc4ab37d5233e7f8402109071aa6f648b31a43309.jpg)

![9d8dc6f9817e9de63d84e5c152ae1bca7582c284ce31fc3b8144306e8acaa297.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/9d8dc6f9817e9de63d84e5c152ae1bca7582c284ce31fc3b8144306e8acaa297.jpg)

![aec17a68375ad8542d60c91f6e6d7ee4beabef7234e09e12e1020daa13000bcc.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/aec17a68375ad8542d60c91f6e6d7ee4beabef7234e09e12e1020daa13000bcc.jpg)

![b5541dbd90ac76aa5ef68c332e09e5d897156c2c00af14892085c6a42c7e985d.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/b5541dbd90ac76aa5ef68c332e09e5d897156c2c00af14892085c6a42c7e985d.jpg)

![c05748844da492b1fd3ddfa2c6ddced46a593b5bff8fc6ef08023f147ada7fcb.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/c05748844da492b1fd3ddfa2c6ddced46a593b5bff8fc6ef08023f147ada7fcb.jpg)

![e61906b7a22bcef330667972c6a385bb66ed9a305d9c3ff7845459c3f5c8a519.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/e61906b7a22bcef330667972c6a385bb66ed9a305d9c3ff7845459c3f5c8a519.jpg)

![f47ca1de693b45dad99df4c568b97a2ccba2591a4592b22f58fca173aa192a6f.jpg](../icml_results/5967_Who%20Transfers%20Safety_%20Identifying%20and%20Targeting%20Cross-Lingual%20Shared%20Safety%20Neurons/tables/f47ca1de693b45dad99df4c568b97a2ccba2591a4592b22f58fca173aa192a6f.jpg)

<a id="paper-5919"></a>

## Agentic Confidence Calibration


### Images

![0d5078a9ce481289a16c13232aac7667a7ad9e18982444f33346edd6ec084136.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/0d5078a9ce481289a16c13232aac7667a7ad9e18982444f33346edd6ec084136.jpg)

![0df7268ca4b02652ff215983575ee4f7c106260ea849b0518ee870d2e98e31e0.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/0df7268ca4b02652ff215983575ee4f7c106260ea849b0518ee870d2e98e31e0.jpg)

![0e03263fbf91af5ecd3052576302e71209b9bcd50bc015901b48c99560d53e37.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/0e03263fbf91af5ecd3052576302e71209b9bcd50bc015901b48c99560d53e37.jpg)

![2212f00e69136a3c11833a6e17b44cf658fd3aa9449224fa0b4a92ae406cf801.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/2212f00e69136a3c11833a6e17b44cf658fd3aa9449224fa0b4a92ae406cf801.jpg)

![431593132d9321d54067e1424a782471a117610ae281535354878fbde81d267d.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/431593132d9321d54067e1424a782471a117610ae281535354878fbde81d267d.jpg)

![5fcdbeff5fd80e117e832c3bec4253deec4b460e54116a5414efdf4d0fa8412d.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/5fcdbeff5fd80e117e832c3bec4253deec4b460e54116a5414efdf4d0fa8412d.jpg)

![73bbf625d85598dbc4b7d8053f07dd8de86136956e9bc610dd42c828b8d161c3.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/73bbf625d85598dbc4b7d8053f07dd8de86136956e9bc610dd42c828b8d161c3.jpg)

![808bcd6249e13001ad8558a6b81b0d5aec3975be8b9d10699b959155ff3037d6.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/808bcd6249e13001ad8558a6b81b0d5aec3975be8b9d10699b959155ff3037d6.jpg)

![9301f4245a3214d81273c497540c831e3f255b7fb3fe3662be261afde21f4c34.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/9301f4245a3214d81273c497540c831e3f255b7fb3fe3662be261afde21f4c34.jpg)

![9e933062e9e6c016aaceb31638aea84b3fac1dc5670a9f5e0c4de473b26eef79.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/9e933062e9e6c016aaceb31638aea84b3fac1dc5670a9f5e0c4de473b26eef79.jpg)

![a68d9271df40b2037641bffa79c213ec76793c0f82963e70f53334a7bfb54a60.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/a68d9271df40b2037641bffa79c213ec76793c0f82963e70f53334a7bfb54a60.jpg)

![b61247ecd9e46db5ffc96f790a0175a9038f0b4b019924a3592bfc616effc563.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/b61247ecd9e46db5ffc96f790a0175a9038f0b4b019924a3592bfc616effc563.jpg)

![ba537b25ddd61d517ee0884681eff3a0f7ea773c9939980959b7cbf72aa69475.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/ba537b25ddd61d517ee0884681eff3a0f7ea773c9939980959b7cbf72aa69475.jpg)

![bbe44845edcea6343935d13a2a5655f41a174b684f9125a4f03efcdcd70e6734.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/bbe44845edcea6343935d13a2a5655f41a174b684f9125a4f03efcdcd70e6734.jpg)

![c7d818785fb7850a7d025690420f0e289a4f2208d84b146be49a30a904222c2c.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/c7d818785fb7850a7d025690420f0e289a4f2208d84b146be49a30a904222c2c.jpg)

![dbbf96a3db74c371c619c58b0be3912f92248b20983c2bf3fbcc049a40eac3ae.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/dbbf96a3db74c371c619c58b0be3912f92248b20983c2bf3fbcc049a40eac3ae.jpg)

![e3e371bb61db37d5ba2041ea0073d4e07feeb403f3ed69eb138cd6d7fce7cd90.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/e3e371bb61db37d5ba2041ea0073d4e07feeb403f3ed69eb138cd6d7fce7cd90.jpg)

![edf084edfc530de849b2a8faff8d260be68162fd74b435615c39956be01bda12.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/edf084edfc530de849b2a8faff8d260be68162fd74b435615c39956be01bda12.jpg)

![ef4caa78fafbfc9f8eb72134ab421eec628d8a867e44b1815425b22f555ca4f2.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/images/ef4caa78fafbfc9f8eb72134ab421eec628d8a867e44b1815425b22f555ca4f2.jpg)

### Tables

![0f069897d57293c1cede208a46cb57eedeb622991650be58293192a747158c4c.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/0f069897d57293c1cede208a46cb57eedeb622991650be58293192a747158c4c.jpg)

![1865c57d5f2454f07f6298b5372eb1bcca51bb153c6c1ccfdcf12fb5cd36ef8e.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/1865c57d5f2454f07f6298b5372eb1bcca51bb153c6c1ccfdcf12fb5cd36ef8e.jpg)

![51a227c4f9b79827b61c9c1b548f1055ea3f58cab5bd6716dc2b779014742cff.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/51a227c4f9b79827b61c9c1b548f1055ea3f58cab5bd6716dc2b779014742cff.jpg)

![6e0d9e4cb2813545a4b29dcbf7328f777aa6551c81554918f98779958c234863.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/6e0d9e4cb2813545a4b29dcbf7328f777aa6551c81554918f98779958c234863.jpg)

![7741776d16a1e7d981d0e51855edca8af7d37620b0894087228e4ab0ee70fe29.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/7741776d16a1e7d981d0e51855edca8af7d37620b0894087228e4ab0ee70fe29.jpg)

![77cbd41618a51eb141429dc2d573f6e0d777447d701ef494ec08bc2ffaeed733.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/77cbd41618a51eb141429dc2d573f6e0d777447d701ef494ec08bc2ffaeed733.jpg)

![b3ab4df652d74c531f942c3d802435a2b875342e10a5a39cbfc4ff6a026a740a.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/b3ab4df652d74c531f942c3d802435a2b875342e10a5a39cbfc4ff6a026a740a.jpg)

![d43dbc2fa285bf83a3c419dd7a4e8103e4f3e56c7a685c31fdf5c83063c7888b.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/d43dbc2fa285bf83a3c419dd7a4e8103e4f3e56c7a685c31fdf5c83063c7888b.jpg)

![e210813451ef4d460c49a818494169cdbdcb4b00ec4f01e0b365dae65a1a1af7.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/e210813451ef4d460c49a818494169cdbdcb4b00ec4f01e0b365dae65a1a1af7.jpg)

![e8535d59bbf56082508cd5ff4c1b89e018513b4e9bf604609e1024d8428110e7.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/e8535d59bbf56082508cd5ff4c1b89e018513b4e9bf604609e1024d8428110e7.jpg)

![edd49bdb7c8921563f7524a98295a55dfc63262e120f43cc92db852882f12a0e.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/edd49bdb7c8921563f7524a98295a55dfc63262e120f43cc92db852882f12a0e.jpg)

![f57b5da3371d5b806b95d5a7f705a6795f4b61df1bc2d84cf5a920839e342c7c.jpg](../icml_results/5968_Agentic%20Confidence%20Calibration/tables/f57b5da3371d5b806b95d5a7f705a6795f4b61df1bc2d84cf5a920839e342c7c.jpg)

<a id="paper-5920"></a>

## Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models


### Images

![06a70ad6c005cd157d82b83bba24d526a7f1a488b2d7745d04e4f6708a18d192.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/06a70ad6c005cd157d82b83bba24d526a7f1a488b2d7745d04e4f6708a18d192.jpg)

![0d4bb43d9643e148088bfd3a2f61ec43976aaf30f10cc7c9d87fab02b03fa237.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/0d4bb43d9643e148088bfd3a2f61ec43976aaf30f10cc7c9d87fab02b03fa237.jpg)

![1245a45eff0e923cd1d4615fd95d9931b6c74dde0db3a5a20f629690a2e4903c.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/1245a45eff0e923cd1d4615fd95d9931b6c74dde0db3a5a20f629690a2e4903c.jpg)

![13fb8926696ea26e2a48f4ce82128eb698b9e0a9c41584b2bb31d649ebaca8a1.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/13fb8926696ea26e2a48f4ce82128eb698b9e0a9c41584b2bb31d649ebaca8a1.jpg)

![179fb1481c8e76723a2b2c596b2cbf232b7b6e9cea3632118ab22bb29d491e2c.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/179fb1481c8e76723a2b2c596b2cbf232b7b6e9cea3632118ab22bb29d491e2c.jpg)

![1c7770a9bd86c22d0f208fc143ed24ffde4f65ebb8e8588f1175959aea5a081b.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/1c7770a9bd86c22d0f208fc143ed24ffde4f65ebb8e8588f1175959aea5a081b.jpg)

![2be505ee30361c899fe8d5a51a2f22bca3019411eaf496ce5d1c908e1a268b4b.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/2be505ee30361c899fe8d5a51a2f22bca3019411eaf496ce5d1c908e1a268b4b.jpg)

![3f2e51e6c85aba977c9f3c67ff6c6f258a7d056ff505c234601189236b131d3c.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/3f2e51e6c85aba977c9f3c67ff6c6f258a7d056ff505c234601189236b131d3c.jpg)

![3f345d5f468f5ab697ad8c053621c266fb08d5589987f9eb4ce028cfa4b78f35.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/3f345d5f468f5ab697ad8c053621c266fb08d5589987f9eb4ce028cfa4b78f35.jpg)

![6188e0d222b41b9fcaae0239160b3229234d58b3b3be87c9856e0c0d9f95880e.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/6188e0d222b41b9fcaae0239160b3229234d58b3b3be87c9856e0c0d9f95880e.jpg)

![6b57086b1cc0ee0a234d2b72ddbaec00c6b12d4dd38a82faefe4c97d42f46a1b.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/6b57086b1cc0ee0a234d2b72ddbaec00c6b12d4dd38a82faefe4c97d42f46a1b.jpg)

![78c794e99df8de26749647a8858dfed56c2590bef8d6d033a357b3b2fb021fbd.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/78c794e99df8de26749647a8858dfed56c2590bef8d6d033a357b3b2fb021fbd.jpg)

![97d9cf4f0edb7f2fab456f0e2a85c9ff471fe1a5ea16572daf8bb51b244d8745.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/97d9cf4f0edb7f2fab456f0e2a85c9ff471fe1a5ea16572daf8bb51b244d8745.jpg)

![d70b73c34c3d7ff9e04e320200155ac130bd55d18cef324fe50df5e46198b608.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/d70b73c34c3d7ff9e04e320200155ac130bd55d18cef324fe50df5e46198b608.jpg)

![d9516189f912208696d5ce94c30c9aec84ade767619573c8c6d002eccd1d77df.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/d9516189f912208696d5ce94c30c9aec84ade767619573c8c6d002eccd1d77df.jpg)

![e26ccc1cacef4d20e7904b67c1a01a40a37803c53571901e8b256aa04124f725.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/e26ccc1cacef4d20e7904b67c1a01a40a37803c53571901e8b256aa04124f725.jpg)

![fa5c716fb47b0493dcff4ec0ba21a68f1fb7a84cc669c765760ee683de3b7a64.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/images/fa5c716fb47b0493dcff4ec0ba21a68f1fb7a84cc669c765760ee683de3b7a64.jpg)

### Tables

![26afd872bf6a98e5dda036a3f4b55e8d59a033ca8d25fd741b06167a7239da7a.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/26afd872bf6a98e5dda036a3f4b55e8d59a033ca8d25fd741b06167a7239da7a.jpg)

![2dd9c791ab40e28db4c0843980ce1dad3d032ab9b29719715dd082772d6e8299.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/2dd9c791ab40e28db4c0843980ce1dad3d032ab9b29719715dd082772d6e8299.jpg)

![5fa935d91351f6d7505544417b3605f2b6532e77b6e818d1956a7baf7b44ce3c.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/5fa935d91351f6d7505544417b3605f2b6532e77b6e818d1956a7baf7b44ce3c.jpg)

![8af80a1dabc55ac710786897823899efc62977da1758fc69118a87e4abf1e1d1.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/8af80a1dabc55ac710786897823899efc62977da1758fc69118a87e4abf1e1d1.jpg)

![c2049af0ad045f65300af955d17ce888009257962f7ab043839ff153dc508378.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/c2049af0ad045f65300af955d17ce888009257962f7ab043839ff153dc508378.jpg)

![ebf1bc452e8840cdd05e1be999cd7d688608d96fe507da5e2e5f4af5dea815f0.jpg](../icml_results/5969_Embodied%20Interpretability_%20Linking%20Causal%20Understanding%20to%20Generalization%20in%20Vision-Language-Action%20/tables/ebf1bc452e8840cdd05e1be999cd7d688608d96fe507da5e2e5f4af5dea815f0.jpg)

<a id="paper-5921"></a>

## Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise


### Images

![03d6faaefda9c3ea5196a49944502355c185d512fd9f58d28285c40e586e7f21.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/03d6faaefda9c3ea5196a49944502355c185d512fd9f58d28285c40e586e7f21.jpg)

![10928242e41e81bc76dec92e695ce587b77b2a1a7f54cfe522d973b5a1fbeaf1.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/10928242e41e81bc76dec92e695ce587b77b2a1a7f54cfe522d973b5a1fbeaf1.jpg)

![25a9f30554782b2d24abfb3eac1f0a8b8911696b0a12f02ed10a4a602a404f92.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/25a9f30554782b2d24abfb3eac1f0a8b8911696b0a12f02ed10a4a602a404f92.jpg)

![3a19e20442ece579ab7d048ea498e133ac9c797c316c7f06898b66d66cdb92c3.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/3a19e20442ece579ab7d048ea498e133ac9c797c316c7f06898b66d66cdb92c3.jpg)

![ae1dd97a9bd462c827dcc4dd65ffd8d18e58f4c3a618fe34787b9a88e5010155.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/ae1dd97a9bd462c827dcc4dd65ffd8d18e58f4c3a618fe34787b9a88e5010155.jpg)

![ae4e6b590c135c6cbe3814e7ddd275cca2193ed3f3af62d87573a1d38e2adf19.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/ae4e6b590c135c6cbe3814e7ddd275cca2193ed3f3af62d87573a1d38e2adf19.jpg)

![dbaaeb04f8681a9b8deefe3d2885100c689865909d6f515c30aa2759f9632f78.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/dbaaeb04f8681a9b8deefe3d2885100c689865909d6f515c30aa2759f9632f78.jpg)

![e4a2ef8632acbfa1b701fef9363c82a60347c63988e4504960f35e58d68334e7.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/images/e4a2ef8632acbfa1b701fef9363c82a60347c63988e4504960f35e58d68334e7.jpg)

### Tables

![12e209be3497b939b1c3ae97b35b530633e10eb67b92e08c3115f1bb60d87bbb.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/12e209be3497b939b1c3ae97b35b530633e10eb67b92e08c3115f1bb60d87bbb.jpg)

![23538e870f3c88a08242a0edd0343b6d2793bb22b308fbd073744d9fde746da7.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/23538e870f3c88a08242a0edd0343b6d2793bb22b308fbd073744d9fde746da7.jpg)

![58a75cfc0472d8c05a8da1eefd7371d94dfebeeb52e2b12ddacf955180bb1121.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/58a75cfc0472d8c05a8da1eefd7371d94dfebeeb52e2b12ddacf955180bb1121.jpg)

![6c61369ec92e95c04247db6811c8415eb2b9ea8aebcd9ebd8cb791415d28d341.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/6c61369ec92e95c04247db6811c8415eb2b9ea8aebcd9ebd8cb791415d28d341.jpg)

![8fb45d16ac0287dc74c0d44e4a7231f1e8946522b9f71b1992ab5d19cd99d925.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/8fb45d16ac0287dc74c0d44e4a7231f1e8946522b9f71b1992ab5d19cd99d925.jpg)

![c2fa1999cfd9ca442811e17d4adeeee0e66c4d7c2ddbf980ef1f871537d5e1cf.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/c2fa1999cfd9ca442811e17d4adeeee0e66c4d7c2ddbf980ef1f871537d5e1cf.jpg)

![d40892fe38b6bb42837adaa05eed3dff76ae09f965a753c35c1f67ab41368dc7.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/d40892fe38b6bb42837adaa05eed3dff76ae09f965a753c35c1f67ab41368dc7.jpg)

![d8837cc5bff49294806b422c89a45bd3c54ab8ab89ca1429473f2eb8fd8f2c38.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/d8837cc5bff49294806b422c89a45bd3c54ab8ab89ca1429473f2eb8fd8f2c38.jpg)

![dddac5eaed26606a4a2fd4233f937df02c03663997d946fa65431bc3cb25e090.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/dddac5eaed26606a4a2fd4233f937df02c03663997d946fa65431bc3cb25e090.jpg)

![e7459a188c71d31b285ec6ee10d373b6b3498fb54fff17d3e0298214efcec42b.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/e7459a188c71d31b285ec6ee10d373b6b3498fb54fff17d3e0298214efcec42b.jpg)

![f90ededba0417dcd24a8c9e15d6426880af2b11df541ca7364dc11617f9ee99a.jpg](../icml_results/5970_Data%20Augmentation%20of%20Contrastive%20Learning%20is%20Estimating%20Positive-incentive%20Noise/tables/f90ededba0417dcd24a8c9e15d6426880af2b11df541ca7364dc11617f9ee99a.jpg)

<a id="paper-5922"></a>

## Benchmarking and Enhancing VLM for Compressed Image Understanding


### Images

![019991116cd75cf46ea10041b05e3393f71636da77ab683c849e08d26c920548.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/019991116cd75cf46ea10041b05e3393f71636da77ab683c849e08d26c920548.jpg)

![04891a8d6b7eb962749775e775e858f6247f3b401f8eccffac894f7ce63d5b26.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/04891a8d6b7eb962749775e775e858f6247f3b401f8eccffac894f7ce63d5b26.jpg)

![219f8be1119c450d0486f61072f01d6aacc7da1ac2d0ca834af9e2de108e7de0.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/219f8be1119c450d0486f61072f01d6aacc7da1ac2d0ca834af9e2de108e7de0.jpg)

![28169c224270fdb0526ee29322f614b8a3ba6c77157aa9cdf7320fc8c65f8c87.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/28169c224270fdb0526ee29322f614b8a3ba6c77157aa9cdf7320fc8c65f8c87.jpg)

![2f979430ba9643350a6f55a17091555bb933ee103e362ccec1012c17d270a988.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/2f979430ba9643350a6f55a17091555bb933ee103e362ccec1012c17d270a988.jpg)

![3d5949f0b4e90a57b79f7994af1402f46d50a0e51d6d8857cb1e3023e119cba3.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/3d5949f0b4e90a57b79f7994af1402f46d50a0e51d6d8857cb1e3023e119cba3.jpg)

![45f5a553d06bdd4b6124ed2282bd66ddc823f22d6cc081f1ebca79671ab42eb7.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/45f5a553d06bdd4b6124ed2282bd66ddc823f22d6cc081f1ebca79671ab42eb7.jpg)

![4b505906121c0448e727796941c49ee8ab120910446fc7203e97404c3e8eda40.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/4b505906121c0448e727796941c49ee8ab120910446fc7203e97404c3e8eda40.jpg)

![4b814ef6e09c077b8672958407ccfe648a449e69951dde00193546373b0c6bd4.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/4b814ef6e09c077b8672958407ccfe648a449e69951dde00193546373b0c6bd4.jpg)

![4bf9db8b9c953b14a9b56d28c4c5bc7971511e6255a213d8574e69d2f4087430.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/4bf9db8b9c953b14a9b56d28c4c5bc7971511e6255a213d8574e69d2f4087430.jpg)

![502d6220b0328acfc2c12efd6b3886e84d2a41222729bb60fc43acd577f41293.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/502d6220b0328acfc2c12efd6b3886e84d2a41222729bb60fc43acd577f41293.jpg)

![6b7ae0c172ba6c27899ea2eff182f40487e6fd7053ec6c6677bf15ae9c9c1500.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/6b7ae0c172ba6c27899ea2eff182f40487e6fd7053ec6c6677bf15ae9c9c1500.jpg)

![861a2a46b13c2b5b09c4daab5846705088f93b8974e70c3a15899c1e2cc53cd9.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/861a2a46b13c2b5b09c4daab5846705088f93b8974e70c3a15899c1e2cc53cd9.jpg)

![8e5d8bed830c82464c4e056952b85d8bc02a085513188b9e3d47e2403d4678e4.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/8e5d8bed830c82464c4e056952b85d8bc02a085513188b9e3d47e2403d4678e4.jpg)

![937fdd5eae1ddbbffbf9d02260b6a02c71249927ce5ce65e19433d0cda0a3074.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/937fdd5eae1ddbbffbf9d02260b6a02c71249927ce5ce65e19433d0cda0a3074.jpg)

![b37907a41f4c20620eb8c8528ee163073953de1733e0c053d1bc231e337eba3c.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/b37907a41f4c20620eb8c8528ee163073953de1733e0c053d1bc231e337eba3c.jpg)

![b76dea19c9c28e2f816927b8cc537621bb6173ee8e30686d728e92fb037d8669.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/b76dea19c9c28e2f816927b8cc537621bb6173ee8e30686d728e92fb037d8669.jpg)

![bda0bc5d6a0a9b19f81207d073acbff742f921af1b50953aaaca3c255f2c0794.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/bda0bc5d6a0a9b19f81207d073acbff742f921af1b50953aaaca3c255f2c0794.jpg)

![c6ce696a75618fb2a9f64c125e9b70caf125dca30649b9d847ee67dd37770419.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/c6ce696a75618fb2a9f64c125e9b70caf125dca30649b9d847ee67dd37770419.jpg)

![cbb8fd5dbb93c34dab445965427fb480058ea8ee13dccc7b2e870c239f5c8240.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/cbb8fd5dbb93c34dab445965427fb480058ea8ee13dccc7b2e870c239f5c8240.jpg)

![cef6b22d41adab92c2e352fa4714c9f75bc6b8e6c0cf0dc52cf4211ef7525d22.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/cef6b22d41adab92c2e352fa4714c9f75bc6b8e6c0cf0dc52cf4211ef7525d22.jpg)

![d5aa9e116f7ab15d306de9edec7f9267e73f5f7c6537e6e9ae33a8d7be92f275.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/images/d5aa9e116f7ab15d306de9edec7f9267e73f5f7c6537e6e9ae33a8d7be92f275.jpg)

### Tables

![0ac6dd41f6903b7164002f136665c7c22c7a44fee295a337007ae19b4e564e8e.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/0ac6dd41f6903b7164002f136665c7c22c7a44fee295a337007ae19b4e564e8e.jpg)

![1856ed49bb99bd93526c630f11576d4cb5aacc5e775c48ef14290ef14bab828b.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/1856ed49bb99bd93526c630f11576d4cb5aacc5e775c48ef14290ef14bab828b.jpg)

![1e7ffecf6f345f94a193d37f43171ad1eb040c4ca07f7a76fec1ec2aa692fce1.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/1e7ffecf6f345f94a193d37f43171ad1eb040c4ca07f7a76fec1ec2aa692fce1.jpg)

![3a35518e8cc61d8a219de107f499b2dcbed94e0bd589f8785ca756484a5b8518.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/3a35518e8cc61d8a219de107f499b2dcbed94e0bd589f8785ca756484a5b8518.jpg)

![3f03bcbb3a63063babc036f4d375f3ecad0086b00085177d7c35bc8f37e4c518.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/3f03bcbb3a63063babc036f4d375f3ecad0086b00085177d7c35bc8f37e4c518.jpg)

![428b280e5edeaf0a50c6dbe73043109b308e4d18f069896a0f379f6d4d1abe99.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/428b280e5edeaf0a50c6dbe73043109b308e4d18f069896a0f379f6d4d1abe99.jpg)

![573d61a859e13e1305a80558b1aa16577c539e75d677394e10a9a17fb48d788d.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/573d61a859e13e1305a80558b1aa16577c539e75d677394e10a9a17fb48d788d.jpg)

![616b020539bc162185f89ae4b0e1cdab64220b17c84053224b944de9304102d0.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/616b020539bc162185f89ae4b0e1cdab64220b17c84053224b944de9304102d0.jpg)

![8e3f20ec53724f9c0e677791f68457f86817303262160f467c0e6ec3bfd34873.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/8e3f20ec53724f9c0e677791f68457f86817303262160f467c0e6ec3bfd34873.jpg)

![a81892d205d091b526c211f9984c0034d13aa7ca5a375f02df78aa33bb89745c.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/a81892d205d091b526c211f9984c0034d13aa7ca5a375f02df78aa33bb89745c.jpg)

![a9682aad1f8f0a6f832e5e79cdae9f59eedeaa62ada2af8ebd6911e874d0150a.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/a9682aad1f8f0a6f832e5e79cdae9f59eedeaa62ada2af8ebd6911e874d0150a.jpg)

![b38b9bb051ded051a9334779e2a45103f2a33edf2bf614bab67c9bdacfac19fd.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/b38b9bb051ded051a9334779e2a45103f2a33edf2bf614bab67c9bdacfac19fd.jpg)

![c2ab698c8c81d24478ddbdc4a2fc9a1fc5b68b3fb021b21625e2fd7ac0842e75.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/c2ab698c8c81d24478ddbdc4a2fc9a1fc5b68b3fb021b21625e2fd7ac0842e75.jpg)

![d0f7842112760b90fc6d0e96748ae90a1951774932449bd9cc640d3480277dd5.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/d0f7842112760b90fc6d0e96748ae90a1951774932449bd9cc640d3480277dd5.jpg)

![dc1e5cea39e9acfd90e1c0d4b471ecf2765831256feaa1f253801f346f004857.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/dc1e5cea39e9acfd90e1c0d4b471ecf2765831256feaa1f253801f346f004857.jpg)

![e6558e588402f7ca4d9bc58e8f8a7c03feba9df6e93ba8d8276d15fee67cfebf.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/e6558e588402f7ca4d9bc58e8f8a7c03feba9df6e93ba8d8276d15fee67cfebf.jpg)

![e99220b5f94bf75c2ff9916c9ec2466ea1f4f2e604fab678151617ffe450f885.jpg](../icml_results/5971_Benchmarking%20and%20Enhancing%20VLM%20for%20Compressed%20Image%20Understanding/tables/e99220b5f94bf75c2ff9916c9ec2466ea1f4f2e604fab678151617ffe450f885.jpg)

<a id="paper-5923"></a>

## SafeCompass: Dynamic Chain-of-Thought Steering via Inference-Time Safety Signals


### Images

![042296594951a05263b67b3bf0911970c181bae8a0c47adc076bbc37e53869d2.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/042296594951a05263b67b3bf0911970c181bae8a0c47adc076bbc37e53869d2.jpg)

![478879e39bbaab1fddafa1e9d909f74b01133faabefd956e5d24a0f6ab96033d.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/478879e39bbaab1fddafa1e9d909f74b01133faabefd956e5d24a0f6ab96033d.jpg)

![9972faa6eb6f0e2be93f3c26f87cfe31c8c6e07d87d9326d792c34b2a9842418.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/9972faa6eb6f0e2be93f3c26f87cfe31c8c6e07d87d9326d792c34b2a9842418.jpg)

![af5e539b366bbf80a64d47dbb895132177be9f37598cd128315bdf8a5a503760.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/af5e539b366bbf80a64d47dbb895132177be9f37598cd128315bdf8a5a503760.jpg)

![c799dbd7f370a9a74acc36a22822ae364191635601013415c0c0000ec239676b.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/c799dbd7f370a9a74acc36a22822ae364191635601013415c0c0000ec239676b.jpg)

![ce79aa6c38020947dc44655edc7c8d91f7e56e4bf63774337762c01415c6d967.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/ce79aa6c38020947dc44655edc7c8d91f7e56e4bf63774337762c01415c6d967.jpg)

![de86ecac0e14f9cb96798c7374fbded9ced82dc927eea3ce3bf1d8491d6be063.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/de86ecac0e14f9cb96798c7374fbded9ced82dc927eea3ce3bf1d8491d6be063.jpg)

![e44908aec65d91ceba42d360227f602697f9900474320bbdfc6bd80044bbda46.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/e44908aec65d91ceba42d360227f602697f9900474320bbdfc6bd80044bbda46.jpg)

![e87866d8c3b428a726f8008d64f84ae90c4d74c15a4221b209427add7b187350.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/e87866d8c3b428a726f8008d64f84ae90c4d74c15a4221b209427add7b187350.jpg)

![fc3f4baa4dd0923258c7b93b388e16e532d07955d216710f77acb4112b9f9077.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/images/fc3f4baa4dd0923258c7b93b388e16e532d07955d216710f77acb4112b9f9077.jpg)

### Tables

![079a298158e7de023c879831cec32bbc8c2d61ca80d0a0fd99599704d95630c7.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/079a298158e7de023c879831cec32bbc8c2d61ca80d0a0fd99599704d95630c7.jpg)

![0d1bc0c01df1fc10e04d697853e9c3b2a80c1f787cab7214a66c5c20a8493a3f.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/0d1bc0c01df1fc10e04d697853e9c3b2a80c1f787cab7214a66c5c20a8493a3f.jpg)

![1cf25c93f784b6c2837ed12701cd495963d2297036d2c50437fd7f8509ad1fbf.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/1cf25c93f784b6c2837ed12701cd495963d2297036d2c50437fd7f8509ad1fbf.jpg)

![462693eb03bfbcb15dfdc1c34c98d930a00e41e63e0e5dac5ab13a2a01b3fd6e.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/462693eb03bfbcb15dfdc1c34c98d930a00e41e63e0e5dac5ab13a2a01b3fd6e.jpg)

![7c46cae3e1a8fd039fd390f4a6feeabbc5e87c3f1bc7a7edb4aa8f5ac9e01eab.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/7c46cae3e1a8fd039fd390f4a6feeabbc5e87c3f1bc7a7edb4aa8f5ac9e01eab.jpg)

![af58a8c8ffb4242579826ae89b73143db0f5019c5c66721b619c51509d1ff4e4.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/af58a8c8ffb4242579826ae89b73143db0f5019c5c66721b619c51509d1ff4e4.jpg)

![b0d87aa701016275a8336c6538716e4cf373d2899483e067dc69d75cc001dc5e.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/b0d87aa701016275a8336c6538716e4cf373d2899483e067dc69d75cc001dc5e.jpg)

![b3a754265361bb01f65fac843a58271ba9bf17b72146e0dc8153fd9562206253.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/b3a754265361bb01f65fac843a58271ba9bf17b72146e0dc8153fd9562206253.jpg)

![ef9ef6c315b60536d6db134ead0157b786c20efcea67a4eec9c3ef99dd9dd5bc.jpg](../icml_results/5972_SafeCompass_%20Dynamic%20Chain-of-Thought%20Steering%20via%20Inference-Time%20Safety%20Signals/tables/ef9ef6c315b60536d6db134ead0157b786c20efcea67a4eec9c3ef99dd9dd5bc.jpg)

<a id="paper-5924"></a>

## Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning


### Images

![048bb6db48af707903a124fe1d5bf19241d4393d6becdcf6df4901023581d895.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/048bb6db48af707903a124fe1d5bf19241d4393d6becdcf6df4901023581d895.jpg)

![128e3bae845d5df4cce0db76996f500867940f97d0af337e07968e725d062d91.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/128e3bae845d5df4cce0db76996f500867940f97d0af337e07968e725d062d91.jpg)

![278dd5ca21b91bdd7ab6fc5aef8d2ac16bce5fea49e00a22bf4f71e1aa4dc37b.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/278dd5ca21b91bdd7ab6fc5aef8d2ac16bce5fea49e00a22bf4f71e1aa4dc37b.jpg)

![34b875233d70fb6cee6648f2ca1b8a88c1e84dc035d3469329e27033fc1a881e.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/34b875233d70fb6cee6648f2ca1b8a88c1e84dc035d3469329e27033fc1a881e.jpg)

![4e7ccc08a8bd01d56028234fd51e22429326750034b5a6e10c40b0d4d2bd0db0.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/4e7ccc08a8bd01d56028234fd51e22429326750034b5a6e10c40b0d4d2bd0db0.jpg)

![599e8ca88543d5e5de6ed97ccfe3e93f03acb31b55aec5f749fed4a03e33e60a.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/599e8ca88543d5e5de6ed97ccfe3e93f03acb31b55aec5f749fed4a03e33e60a.jpg)

![5a3ec5f0fec5493af707a5ee1281d4089ad92cc5988df5a709abfa7bf513cde5.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/5a3ec5f0fec5493af707a5ee1281d4089ad92cc5988df5a709abfa7bf513cde5.jpg)

![7870048895dba676f610558b43ae3c88736b64bfe2ac325eb8d8528a9f6fd452.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/7870048895dba676f610558b43ae3c88736b64bfe2ac325eb8d8528a9f6fd452.jpg)

![7b6376e000273d00d2c2172bac871c4464bc1edfb1c6941a32ae0d12bbaa098b.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/7b6376e000273d00d2c2172bac871c4464bc1edfb1c6941a32ae0d12bbaa098b.jpg)

![7f5aa17dd9fa3d57cf847be12e514379669a843317ede6418b9b9ccf55a8de3d.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/7f5aa17dd9fa3d57cf847be12e514379669a843317ede6418b9b9ccf55a8de3d.jpg)

![8e18e88c00a452bd46e384f2434ddb5b0c98f699f989ab0fea57cffcb95348cb.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/8e18e88c00a452bd46e384f2434ddb5b0c98f699f989ab0fea57cffcb95348cb.jpg)

![8e9b070b1bd45d3a2056748501d859b54fc132cacac1a79759c2bf72412d5d52.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/8e9b070b1bd45d3a2056748501d859b54fc132cacac1a79759c2bf72412d5d52.jpg)

![97efa58b20962564699737b82fcfd47a275a09369ab03fcf71646e9365dba324.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/97efa58b20962564699737b82fcfd47a275a09369ab03fcf71646e9365dba324.jpg)

![bb00a7aecda5c657f5d6633f992509de091c2f73e7a5074a10e2787ad321814d.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/bb00a7aecda5c657f5d6633f992509de091c2f73e7a5074a10e2787ad321814d.jpg)

![d2ee456adc1d9302cf7ce069533389a7ba32f72b90ceb3863e5687de4a9407c0.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/d2ee456adc1d9302cf7ce069533389a7ba32f72b90ceb3863e5687de4a9407c0.jpg)

![d4bb4f5e0f2931465dff1753a756ff4cab908812d2eaa4af1fd02c29bece2c63.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/d4bb4f5e0f2931465dff1753a756ff4cab908812d2eaa4af1fd02c29bece2c63.jpg)

![e37e268e55b3a421d8e3d4b5b55950658d2097abd296bc79d18f66d69c0e8683.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/images/e37e268e55b3a421d8e3d4b5b55950658d2097abd296bc79d18f66d69c0e8683.jpg)

### Tables

![1c5b12f441abade8a20a1ae0d5e9f6dc01d71239b3d876add217e50d8fafcc80.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/1c5b12f441abade8a20a1ae0d5e9f6dc01d71239b3d876add217e50d8fafcc80.jpg)

![1d5dd048977dc75754cd102e50df250dcc9fb2a3beeb05b68c147aa2dbd150bc.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/1d5dd048977dc75754cd102e50df250dcc9fb2a3beeb05b68c147aa2dbd150bc.jpg)

![370ac28437a41879cfca2bf51e67e49ae1a5ea835958d6ec2da5c487e7448196.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/370ac28437a41879cfca2bf51e67e49ae1a5ea835958d6ec2da5c487e7448196.jpg)

![3c81a6f1dfdf78ab72afb287bd3a384ab6e7ba1a47484d4ed3b69d1d130c5251.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/3c81a6f1dfdf78ab72afb287bd3a384ab6e7ba1a47484d4ed3b69d1d130c5251.jpg)

![5c47fc9a8781af7b8d754e19919e2f5a9ab847362daa2d0e3d09b87fd615b2cf.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/5c47fc9a8781af7b8d754e19919e2f5a9ab847362daa2d0e3d09b87fd615b2cf.jpg)

![61f2b0fa900da0024d1b31083b4f58c240ca5fe38aa606da07c2d6ed896d30b3.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/61f2b0fa900da0024d1b31083b4f58c240ca5fe38aa606da07c2d6ed896d30b3.jpg)

![6ff457ca5568869141439664f4c670eefbc0de565109df89b86c61f6e68c2565.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/6ff457ca5568869141439664f4c670eefbc0de565109df89b86c61f6e68c2565.jpg)

![831da411c70d90950f423e7363831351684ce884f59235d62b718ca1c98fddd1.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/831da411c70d90950f423e7363831351684ce884f59235d62b718ca1c98fddd1.jpg)

![8997faa4f7dba05354531886d6fc01d11bc83b8df2649a83d1ef20af6e8379d2.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/8997faa4f7dba05354531886d6fc01d11bc83b8df2649a83d1ef20af6e8379d2.jpg)

![b19816e215f148e7af6e1c0e74fa1e98fc40a843d4f28338d0ee1798cc89c569.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/b19816e215f148e7af6e1c0e74fa1e98fc40a843d4f28338d0ee1798cc89c569.jpg)

![b291a482f12c4edf9e9f7e21ddaa61318928a73040db2fb0e393784f3d3dce5f.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/b291a482f12c4edf9e9f7e21ddaa61318928a73040db2fb0e393784f3d3dce5f.jpg)

![d24b0dc3a89429b24615164ea2586ce022155668cca37775bd93edcf7252e7a6.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/d24b0dc3a89429b24615164ea2586ce022155668cca37775bd93edcf7252e7a6.jpg)

![da82dd305caa482489f6375dad67c1c9613fb6ce64540bb8d22a0b2836a545a6.jpg](../icml_results/5973_Good%20SFT%20Optimizes%20for%20SFT%2C%20Better%20SFT%20Prepares%20for%20Reinforcement%20Learning/tables/da82dd305caa482489f6375dad67c1c9613fb6ce64540bb8d22a0b2836a545a6.jpg)

<a id="paper-5925"></a>

## Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification


### Images

![09dfa2870aa5be45d9a25f1759ee8ac482c52c65509af83b1f8919461047cdf1.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/09dfa2870aa5be45d9a25f1759ee8ac482c52c65509af83b1f8919461047cdf1.jpg)

![112aa5efedf1fcfb07ff2bd2645c2cca20e8c81973bf2bb81a864a65b6017de7.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/112aa5efedf1fcfb07ff2bd2645c2cca20e8c81973bf2bb81a864a65b6017de7.jpg)

![19137ac127f993e1e65cd8a5ca927ffb7febb3dc1386329e81cc7c7edad69971.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/19137ac127f993e1e65cd8a5ca927ffb7febb3dc1386329e81cc7c7edad69971.jpg)

![1f793c5a96d1ad5ad9355db3db383c33a67dbd4bc00f674be4fcedc565df41f8.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/1f793c5a96d1ad5ad9355db3db383c33a67dbd4bc00f674be4fcedc565df41f8.jpg)

![53b6b066bc464fb0ada9675d1a25918a2625ec3f9a0512b1f88a16e503896862.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/53b6b066bc464fb0ada9675d1a25918a2625ec3f9a0512b1f88a16e503896862.jpg)

![70a37557069b5fba44ac3c217b2b9595595c6fd72fb1d53df5b2e03fc482b8c4.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/70a37557069b5fba44ac3c217b2b9595595c6fd72fb1d53df5b2e03fc482b8c4.jpg)

![77e0815ad34e59d2aa366fdc8e7cdd61b1beee36377ce94a3525f3e18fb1abbc.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/77e0815ad34e59d2aa366fdc8e7cdd61b1beee36377ce94a3525f3e18fb1abbc.jpg)

![7977cb36820e7837143bf751dd1b1ecf35e4425942ffddacafc1f6f9d093f01c.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/7977cb36820e7837143bf751dd1b1ecf35e4425942ffddacafc1f6f9d093f01c.jpg)

![8a5c31d547ca2cc1203e241d9b57ee8afbb89ea23b28cd39e3d7f2e28f7ba653.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/8a5c31d547ca2cc1203e241d9b57ee8afbb89ea23b28cd39e3d7f2e28f7ba653.jpg)

![90fc3d31a8c374fe68248fd4270b15006bc842abb2905260346ec34175755b40.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/90fc3d31a8c374fe68248fd4270b15006bc842abb2905260346ec34175755b40.jpg)

![d1d956201e0b076f968419d6d2057b64737d54c8c399f9ccccbf64e9240685cb.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/d1d956201e0b076f968419d6d2057b64737d54c8c399f9ccccbf64e9240685cb.jpg)

![dcbc45d53daf9a4488983535174a8cd5b54794f15c4a908f98b48eb3493366a3.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/dcbc45d53daf9a4488983535174a8cd5b54794f15c4a908f98b48eb3493366a3.jpg)

![fddf1e2b3dde93e3ae302d05a5cea01330eb59452e54014b8bc865e829d8147d.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/images/fddf1e2b3dde93e3ae302d05a5cea01330eb59452e54014b8bc865e829d8147d.jpg)

### Tables

![029ffaf8304e9d83d6fb56cedadcd9359de7fce0d0b5e70d5011feb383ad12e8.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/029ffaf8304e9d83d6fb56cedadcd9359de7fce0d0b5e70d5011feb383ad12e8.jpg)

![0ac592a2050cef8749ee648a7a9e17f44ecf5d05c6af7ddfc66e382aa137de7e.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/0ac592a2050cef8749ee648a7a9e17f44ecf5d05c6af7ddfc66e382aa137de7e.jpg)

![16d4bbbd0ffcb52c88267149cf8d8c6703ba936428914ca8d1164a77d8ef9d94.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/16d4bbbd0ffcb52c88267149cf8d8c6703ba936428914ca8d1164a77d8ef9d94.jpg)

![1e6aef2cd34241272795971d7dd5e49e9283c5e967633e1c1daf2991f6931cef.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/1e6aef2cd34241272795971d7dd5e49e9283c5e967633e1c1daf2991f6931cef.jpg)

![2122eb4d7eef923cb2efe1d4d3d1bb0df3f97c6b886da0c09ff91ea6aa5f0dc4.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/2122eb4d7eef923cb2efe1d4d3d1bb0df3f97c6b886da0c09ff91ea6aa5f0dc4.jpg)

![352b769a1efcfad01a67e7a607a40323aefbaba71ca8622740ece6f72c489740.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/352b769a1efcfad01a67e7a607a40323aefbaba71ca8622740ece6f72c489740.jpg)

![36119d362cc0f71dc9a93fd7e8acc4e780b705f0f049ef91b618f98213b66f2b.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/36119d362cc0f71dc9a93fd7e8acc4e780b705f0f049ef91b618f98213b66f2b.jpg)

![5b5ab5a023700e2f219b271ccae7bd314c1857b6fcdbf8517e827821bc476e9c.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/5b5ab5a023700e2f219b271ccae7bd314c1857b6fcdbf8517e827821bc476e9c.jpg)

![5b616ee151bb4daa949df5ada8440063037d87dee4aef448d06c03677d9c0c39.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/5b616ee151bb4daa949df5ada8440063037d87dee4aef448d06c03677d9c0c39.jpg)

![5e4112834b9453b50cfa88618f783e17c0b714d4fc5bf8bfae856288555ed25b.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/5e4112834b9453b50cfa88618f783e17c0b714d4fc5bf8bfae856288555ed25b.jpg)

![62b2e81389d12ea4e7ce57975eb81031e315b0d1e17ed1c627be12a600633cfc.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/62b2e81389d12ea4e7ce57975eb81031e315b0d1e17ed1c627be12a600633cfc.jpg)

![6a7356575444c98549a8f99ba3718514d25d5dd7d1611664d4ff296f8caa990d.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/6a7356575444c98549a8f99ba3718514d25d5dd7d1611664d4ff296f8caa990d.jpg)

![7f04f68a17366444e5456e291f58790789634eef83d025e2f558c65c2698a360.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/7f04f68a17366444e5456e291f58790789634eef83d025e2f558c65c2698a360.jpg)

![86ceac469af27e829885fcb78a81ce0b1c55f13f9b1ccd04978473acbcdab478.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/86ceac469af27e829885fcb78a81ce0b1c55f13f9b1ccd04978473acbcdab478.jpg)

![8c8a4b7833a81fc658d08fd3f47d16984b1638a5f1869bfcb21fb3536ae191ef.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/8c8a4b7833a81fc658d08fd3f47d16984b1638a5f1869bfcb21fb3536ae191ef.jpg)

![9afa1db264232b9e9ee8f981f53955cd8f99671ca200a9eb939fc4a12f6cc8b1.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/9afa1db264232b9e9ee8f981f53955cd8f99671ca200a9eb939fc4a12f6cc8b1.jpg)

![aac1b16089c2569a51ad83eeb1cc3a23ca41fd591537f9c7440a46822ae1df36.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/aac1b16089c2569a51ad83eeb1cc3a23ca41fd591537f9c7440a46822ae1df36.jpg)

![b2ea95ad081d26c87a48a7a09918293479a6726ba5ce0a2856efb374bae29318.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/b2ea95ad081d26c87a48a7a09918293479a6726ba5ce0a2856efb374bae29318.jpg)

![b99e2299b352078b1df7b65472cd893f6b06d6c9aa4b968d8e41c28d1c827761.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/b99e2299b352078b1df7b65472cd893f6b06d6c9aa4b968d8e41c28d1c827761.jpg)

![d0361d6a57a2db1c8a1197180876f1522245135da8c8bccbe7067b4ba4f43a71.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/d0361d6a57a2db1c8a1197180876f1522245135da8c8bccbe7067b4ba4f43a71.jpg)

![fbcb2d78493f78ac1fd07023d6930f045c56761e1929725689fd690fbcd41b8b.jpg](../icml_results/5974_Target-Agnostic%20Calibration%20under%20Distribution%20Shift%20with%20Frequency-Aware%20Gradient%20Rectification/tables/fbcb2d78493f78ac1fd07023d6930f045c56761e1929725689fd690fbcd41b8b.jpg)

<a id="paper-5926"></a>

## AOEB: Benchmarking Agent-Oriented Multimodal Embeddings


### Images

![3312e3340eed5745eed3aaf1477dc09fc0cd68eab8b60efa38c17c885d8d6164.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/3312e3340eed5745eed3aaf1477dc09fc0cd68eab8b60efa38c17c885d8d6164.jpg)

![40a165b574c9cf04af1598e908b79ff0876bc64d92bf6691531ae027667ba0bb.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/40a165b574c9cf04af1598e908b79ff0876bc64d92bf6691531ae027667ba0bb.jpg)

![5828875945b5febd2d52ba932c57fc4e890a6c8fc59dccc83bd62a2966fdece3.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/5828875945b5febd2d52ba932c57fc4e890a6c8fc59dccc83bd62a2966fdece3.jpg)

![61165d18fe11a90926f71469c3e20979d87340a808227d82b49843e1e7d71164.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/61165d18fe11a90926f71469c3e20979d87340a808227d82b49843e1e7d71164.jpg)

![7dc4293fbd0231b90942b7cc1d31ca642a92b9a8b06850afbdd391554b3ce98f.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/7dc4293fbd0231b90942b7cc1d31ca642a92b9a8b06850afbdd391554b3ce98f.jpg)

![a540e6cc31c05db313649bf01726f1d71113630c40f8c5657be5d780541308ad.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/a540e6cc31c05db313649bf01726f1d71113630c40f8c5657be5d780541308ad.jpg)

![b77a3899a873a19e6d7b2633209c2df3d59fb6d135383f87d877df9456b3ece8.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/b77a3899a873a19e6d7b2633209c2df3d59fb6d135383f87d877df9456b3ece8.jpg)

![bc746e2e27df99bb09429205f23ff1ed949db3632314d49d9cc0ff4735dc2acf.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/images/bc746e2e27df99bb09429205f23ff1ed949db3632314d49d9cc0ff4735dc2acf.jpg)

### Tables

![0f7d592417f5f74b3d08c819dadafc41d97168167614e603b6287c7890115e7f.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/0f7d592417f5f74b3d08c819dadafc41d97168167614e603b6287c7890115e7f.jpg)

![15135107e8f055b56541a6ee3b92a10989919c8595165b4dd867bd0932db95b2.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/15135107e8f055b56541a6ee3b92a10989919c8595165b4dd867bd0932db95b2.jpg)

![2e6bfc6e6b763f573d751e8362d74ba8e74e60a3fb0f4157aa65d74494c20269.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/2e6bfc6e6b763f573d751e8362d74ba8e74e60a3fb0f4157aa65d74494c20269.jpg)

![3886fd310f07c9affa117f8b1f55cd5af776641235dc63b3af2a171caa3b8ce2.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/3886fd310f07c9affa117f8b1f55cd5af776641235dc63b3af2a171caa3b8ce2.jpg)

![4207ebd3d997f36396ee79e3827c30170102f5b42b49e98e0324ce6e252aba23.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/4207ebd3d997f36396ee79e3827c30170102f5b42b49e98e0324ce6e252aba23.jpg)

![49897e39ec20af866f12f59536c02ebc0534a83f240df8ceb7076d87094b12cd.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/49897e39ec20af866f12f59536c02ebc0534a83f240df8ceb7076d87094b12cd.jpg)

![6c07e3e9d4f76dd759d7179121862d36deb48cac5d284245d98ad246c2ea6570.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/6c07e3e9d4f76dd759d7179121862d36deb48cac5d284245d98ad246c2ea6570.jpg)

![7ae1268cbc7e58d0608daf0e329b9202730f6f49db8496cb8d3d5fb84f993ae1.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/7ae1268cbc7e58d0608daf0e329b9202730f6f49db8496cb8d3d5fb84f993ae1.jpg)

![cb67dd27634493279afd4d7aeb790f83ed76f7aaf73bf987c4287fb70d2caffa.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/cb67dd27634493279afd4d7aeb790f83ed76f7aaf73bf987c4287fb70d2caffa.jpg)

![ceb7c4ee8f679589586d6e2937abcec8c0a3e2067cc3ccbe4467a1659ddf00f9.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/ceb7c4ee8f679589586d6e2937abcec8c0a3e2067cc3ccbe4467a1659ddf00f9.jpg)

![cf8fe88075b2fbac71518531a01dff7372086d9feb7bbf8fb2be3d39c862a6c6.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/cf8fe88075b2fbac71518531a01dff7372086d9feb7bbf8fb2be3d39c862a6c6.jpg)

![ecfec5e61abcaada2c4df12fee5bbc71dd24078329b46c66fcdd91fa112808c4.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/ecfec5e61abcaada2c4df12fee5bbc71dd24078329b46c66fcdd91fa112808c4.jpg)

![f61779f543cb76097fc87368f54cfe888a565e8b6ae56a8af0045656ec28b171.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/f61779f543cb76097fc87368f54cfe888a565e8b6ae56a8af0045656ec28b171.jpg)

![fd347cd07d65cc77506d9b614cfdfa3a404feb50ef9e9e90a70b513ae8663f45.jpg](../icml_results/5975_AOEB_%20Benchmarking%20Agent-Oriented%20Multimodal%20Embeddings/tables/fd347cd07d65cc77506d9b614cfdfa3a404feb50ef9e9e90a70b513ae8663f45.jpg)

<a id="paper-5927"></a>

## 4DPC$^2$hat: Towards Dynamic Point Cloud Understanding with Failure-Aware Bootstrapping


### Images

![04addb388e058c87628bd025d6add6c38e9c210dccbc7e078aee6339363fab00.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/04addb388e058c87628bd025d6add6c38e9c210dccbc7e078aee6339363fab00.jpg)

![092fee701e83410a4e428955d78c4f7c193d619da52ae80e943e2cbfd599767e.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/092fee701e83410a4e428955d78c4f7c193d619da52ae80e943e2cbfd599767e.jpg)

![0f7890222a1ff44a4b7ac911c2d7d633cf5c88678507b14a9d35154980e91eab.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/0f7890222a1ff44a4b7ac911c2d7d633cf5c88678507b14a9d35154980e91eab.jpg)

![150b4fb753fa2ca10d11aefcb793917914fff0e8f77c1253631922de8b31a743.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/150b4fb753fa2ca10d11aefcb793917914fff0e8f77c1253631922de8b31a743.jpg)

![35c648df4d7c18e2139a3671460a295629381da38a8d8a1b69365ad6dc66ef20.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/35c648df4d7c18e2139a3671460a295629381da38a8d8a1b69365ad6dc66ef20.jpg)

![a0279a36754fd7903a0e23b48d7a27fc5e6e74c381a4eec425725d90dab1d842.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/a0279a36754fd7903a0e23b48d7a27fc5e6e74c381a4eec425725d90dab1d842.jpg)

![ac2dc99998f0849d117ac68a95d55223d57b1df8adcc70c42a689857a7801ffc.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/ac2dc99998f0849d117ac68a95d55223d57b1df8adcc70c42a689857a7801ffc.jpg)

![b4832de9867f2787e260b7b029fef29287eea8d2bc4e1dd55ae21fd4fd4b0d91.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/b4832de9867f2787e260b7b029fef29287eea8d2bc4e1dd55ae21fd4fd4b0d91.jpg)

![b4ac4a4a98c49580186e91b68ef55d4f7c2abe0874c23a7f66963548e35cab1d.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/b4ac4a4a98c49580186e91b68ef55d4f7c2abe0874c23a7f66963548e35cab1d.jpg)

![c213ab195d852f45bb94aee876dc093957b37ae03b9bfbafa379b8a01cb81073.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/c213ab195d852f45bb94aee876dc093957b37ae03b9bfbafa379b8a01cb81073.jpg)

![d68c9ba1b55b6c76a28de1e99bcfde9203d092667e1d0dc45518dbd00180b0ef.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/d68c9ba1b55b6c76a28de1e99bcfde9203d092667e1d0dc45518dbd00180b0ef.jpg)

![e7f1c2f83a67c6e2a4199388639feeae53e34e0ec31b3257742d5b0f8f38d9b9.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/images/e7f1c2f83a67c6e2a4199388639feeae53e34e0ec31b3257742d5b0f8f38d9b9.jpg)

### Tables

![15adf3b4ce41307b094a7d9254fce3d470101ba492f966631b3d97303beea8fc.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/15adf3b4ce41307b094a7d9254fce3d470101ba492f966631b3d97303beea8fc.jpg)

![1a7949132ffc23b85fe32cf0ee825cd13a5d4048c82939c1b1dd893277cecba1.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/1a7949132ffc23b85fe32cf0ee825cd13a5d4048c82939c1b1dd893277cecba1.jpg)

![33e6be2efde10e652bfce3c922fb339ba5f888b23c17c6d8134723a10cec493f.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/33e6be2efde10e652bfce3c922fb339ba5f888b23c17c6d8134723a10cec493f.jpg)

![37f7cd2f83ed8d12e1bb367c1090596e60bd9b4fc3cd7f2c4cc83742b8f520e6.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/37f7cd2f83ed8d12e1bb367c1090596e60bd9b4fc3cd7f2c4cc83742b8f520e6.jpg)

![66bd124776f7e500cc8570d7d75fc222e9fdc9c7cf53c972c43492c77e795d0a.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/66bd124776f7e500cc8570d7d75fc222e9fdc9c7cf53c972c43492c77e795d0a.jpg)

![684b39acde9d76d3e9ead8fcd37c4bbaef9c0988e21baf2175ea9bed3b732f7f.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/684b39acde9d76d3e9ead8fcd37c4bbaef9c0988e21baf2175ea9bed3b732f7f.jpg)

![708650acede5f6db4f32aafec86e031a9b6d87c6afc589ea057a40da35b1be86.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/708650acede5f6db4f32aafec86e031a9b6d87c6afc589ea057a40da35b1be86.jpg)

![b24f8347b0a4386ca9f56a4f177b10cda5e4af74dc32669ce8fed1dd0df9242a.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/b24f8347b0a4386ca9f56a4f177b10cda5e4af74dc32669ce8fed1dd0df9242a.jpg)

![e072936906ca136bdef8adb3ba0d778d2bcebb27d0d9f1dfc665ca6109709351.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/e072936906ca136bdef8adb3ba0d778d2bcebb27d0d9f1dfc665ca6109709351.jpg)

![e93e66fd79b7351a77f9a722c015998050f6cce18ea1859bb5e9728b159692e6.jpg](../icml_results/5976_4DPC%24%5E2%24hat_%20Towards%20Dynamic%20Point%20Cloud%20Understanding%20with%20Failure-Aware%20Bootstrapping/tables/e93e66fd79b7351a77f9a722c015998050f6cce18ea1859bb5e9728b159692e6.jpg)

<a id="paper-5928"></a>

## Backjump-on-Graph: Empowering Large Language Models with Reinforced Retrospective Exploration for Agentic Knowledge Graph Reasoning


### Images

![0002e94b4ca4897d0aeb874b5da5022be9f9d1d9100df2ab0565aa687320fd81.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/0002e94b4ca4897d0aeb874b5da5022be9f9d1d9100df2ab0565aa687320fd81.jpg)

![28314857a6a75697b1de6152d3751eb5738c208107b0c59861da1745501006b5.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/28314857a6a75697b1de6152d3751eb5738c208107b0c59861da1745501006b5.jpg)

![4a9fd8c11ee0363654b2f5bd14a0848dd6f34b6c7ae94d981d7bf8ec5550abe5.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/4a9fd8c11ee0363654b2f5bd14a0848dd6f34b6c7ae94d981d7bf8ec5550abe5.jpg)

![5fb128058bf575c8e7b469a47fec4373567ff31380591a18d2f4455636e3c299.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/5fb128058bf575c8e7b469a47fec4373567ff31380591a18d2f4455636e3c299.jpg)

![b6c1ed6b90aa70247bc1692d526385cfcd40456e49aa0a07b646de6ac6858ce8.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/b6c1ed6b90aa70247bc1692d526385cfcd40456e49aa0a07b646de6ac6858ce8.jpg)

![de2ea81cb8dd6f323d42ba4303d81e249fce764b3a57cadfbb9e25d2409fb3d9.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/images/de2ea81cb8dd6f323d42ba4303d81e249fce764b3a57cadfbb9e25d2409fb3d9.jpg)

### Tables

![25d3f2a4dc652f20fd3969b3c7cb42744efcca0245885b0282380596d9aa777f.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/25d3f2a4dc652f20fd3969b3c7cb42744efcca0245885b0282380596d9aa777f.jpg)

![3b3e47b1d2492e67800bec1d9b689054e5b763d7030bdb33a33a2f27b01bdb18.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/3b3e47b1d2492e67800bec1d9b689054e5b763d7030bdb33a33a2f27b01bdb18.jpg)

![54d0505633726327aa0cfcae8a2ea42589fd3313b8aaf31c673ef613a8abd1e7.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/54d0505633726327aa0cfcae8a2ea42589fd3313b8aaf31c673ef613a8abd1e7.jpg)

![7e73ee76406aca87e3dc184445181c592ec98db79205b492ac1fcd0050b4f5f7.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/7e73ee76406aca87e3dc184445181c592ec98db79205b492ac1fcd0050b4f5f7.jpg)

![e0b31c0c09a47edddc258dde75c3a45dc6ca5228c58d016c9dc408a08db358d3.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/e0b31c0c09a47edddc258dde75c3a45dc6ca5228c58d016c9dc408a08db358d3.jpg)

![e8f276ca57abc4bc48519bca509003e73da4512cc5a05491bd3a3cc8ef37748d.jpg](../icml_results/5977_Backjump-on-Graph_%20Empowering%20Large%20Language%20Models%20with%20Reinforced%20Retrospective%20Exploration%20for%20Ag/tables/e8f276ca57abc4bc48519bca509003e73da4512cc5a05491bd3a3cc8ef37748d.jpg)

<a id="paper-5929"></a>

## ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning


### Images

![39eb1298b0b4fae5fe576602b17195c7b593eed5ad103da558bd816007fc3ee6.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/images/39eb1298b0b4fae5fe576602b17195c7b593eed5ad103da558bd816007fc3ee6.jpg)

![5d4e53841cf8e76f9d24327e97330b591f21033fd49ce008fc7a8e25df647b86.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/images/5d4e53841cf8e76f9d24327e97330b591f21033fd49ce008fc7a8e25df647b86.jpg)

![7a71707143d9959d74705b788f02c17cca8f8b6601cc9702ad272c339f07150f.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/images/7a71707143d9959d74705b788f02c17cca8f8b6601cc9702ad272c339f07150f.jpg)

![d84e8ef2dee34f707ecc49d7a802b9c494aa10a73b67018077d862fd40aa93b0.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/images/d84e8ef2dee34f707ecc49d7a802b9c494aa10a73b67018077d862fd40aa93b0.jpg)

![e77a2151a84fcc772b5887f8c24da42ee3a9b69af8abc508f1304756eb3352a4.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/images/e77a2151a84fcc772b5887f8c24da42ee3a9b69af8abc508f1304756eb3352a4.jpg)

### Tables

![0a185d76c13cbc406669b31b05aff69d056734a1eb250ea289e34f6e67c588c7.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/0a185d76c13cbc406669b31b05aff69d056734a1eb250ea289e34f6e67c588c7.jpg)

![0ebe2e0db9234c7e25f029c1e7992e1f3ebd7d17ac6d9b0e3c380ae407467438.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/0ebe2e0db9234c7e25f029c1e7992e1f3ebd7d17ac6d9b0e3c380ae407467438.jpg)

![5a6fa14188b300d40ac8b300592959ac95cd0a3266665f0c8969dc0cb065d256.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/5a6fa14188b300d40ac8b300592959ac95cd0a3266665f0c8969dc0cb065d256.jpg)

![6a8b4328b83b3bb98906e35f25521620578ce5cb2653cfb8a756272804463dd4.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/6a8b4328b83b3bb98906e35f25521620578ce5cb2653cfb8a756272804463dd4.jpg)

![71e1a1a949700a340ed32e6ccb57db3a2c3b2d0dfeb5f12f19d8990ccc0a7490.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/71e1a1a949700a340ed32e6ccb57db3a2c3b2d0dfeb5f12f19d8990ccc0a7490.jpg)

![78d0b395b6a1dda35b252f49e1b33c717fc864afbb807779eb60d956a9baa408.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/78d0b395b6a1dda35b252f49e1b33c717fc864afbb807779eb60d956a9baa408.jpg)

![95a74b4175554abf2217920b79ae1407573b03ff78b46174276bb26a52e89958.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/95a74b4175554abf2217920b79ae1407573b03ff78b46174276bb26a52e89958.jpg)

![96063ce3bf8b4a669fbff8655b866f89b4fbba65c7d974a0b8d1af105e9e275e.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/96063ce3bf8b4a669fbff8655b866f89b4fbba65c7d974a0b8d1af105e9e275e.jpg)

![9e53de2ce0211cf28739ab8389fb84174379a4e1fbfbb154e665702108448af9.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/9e53de2ce0211cf28739ab8389fb84174379a4e1fbfbb154e665702108448af9.jpg)

![d21aa260876c79f7a3df88459de64592c2473fc826f318e8afa1c4712d2f4a8f.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/d21aa260876c79f7a3df88459de64592c2473fc826f318e8afa1c4712d2f4a8f.jpg)

![fa45fc5d5905794a69b735912418fce562af4b75e25026ee2a901be263124c7d.jpg](../icml_results/5978_ScaLoRA_%20Optimally%20Scaled%20Low-Rank%20Adaptation%20for%20Efficient%20High-Rank%20Fine-Tuning/tables/fa45fc5d5905794a69b735912418fce562af4b75e25026ee2a901be263124c7d.jpg)

<a id="paper-5930"></a>

## SemBind: Binding Diffusion Watermarks to Semantics Against Black-Box Forgery Attacks


### Images

![1d1f92c67b3a705afac61a8fa312f2275c702ce31d22cdc7908081c033cdd80e.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/1d1f92c67b3a705afac61a8fa312f2275c702ce31d22cdc7908081c033cdd80e.jpg)

![23cd7caa12b2660561331092d174cb4dbc4981b389dfab02d71bfdc795df30e8.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/23cd7caa12b2660561331092d174cb4dbc4981b389dfab02d71bfdc795df30e8.jpg)

![2bc8a68c738c3d9b88c0a314d284da6a67b5ba7bbf89a068f0638417afadd12f.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/2bc8a68c738c3d9b88c0a314d284da6a67b5ba7bbf89a068f0638417afadd12f.jpg)

![36dcef5c04c078860b8a38b2413414576bbfffad48e1af5c0873ddf25bfefcbc.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/36dcef5c04c078860b8a38b2413414576bbfffad48e1af5c0873ddf25bfefcbc.jpg)

![daa1aa08240300a295faf45e7315b2f355afd923e3f5409d6cb7a10ed932f317.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/daa1aa08240300a295faf45e7315b2f355afd923e3f5409d6cb7a10ed932f317.jpg)

![ec482034fdc2fe13faa307b3df38d1e64b520815c58ed77d41fd13016737d9d6.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/images/ec482034fdc2fe13faa307b3df38d1e64b520815c58ed77d41fd13016737d9d6.jpg)

### Tables

![1da43310a5bd34a763379f631787e5a4188c4e02885a5f0fb5ba88603a6a1f57.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/1da43310a5bd34a763379f631787e5a4188c4e02885a5f0fb5ba88603a6a1f57.jpg)

![3f6e14ebea4bcd8fa9607046f059858173f6c5f38f6698d915d031ead6c75a5b.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/3f6e14ebea4bcd8fa9607046f059858173f6c5f38f6698d915d031ead6c75a5b.jpg)

![3fcd017d47a6dc638889e029b3c3f073d07df88adf7dfc559d7560688123a781.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/3fcd017d47a6dc638889e029b3c3f073d07df88adf7dfc559d7560688123a781.jpg)

![495308e18a7b818953bd9db9a1e35e8efaa1b3a330427c1b120ec7f13f3cf00a.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/495308e18a7b818953bd9db9a1e35e8efaa1b3a330427c1b120ec7f13f3cf00a.jpg)

![49eaa9b5364046fe2c4a7a43bb59aa3b3aa25f0e92130370c7e1623f6b2ddf50.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/49eaa9b5364046fe2c4a7a43bb59aa3b3aa25f0e92130370c7e1623f6b2ddf50.jpg)

![4ace40228e1754476d201b4363d4200862ee92b1b47549b4e5039e963de5ce9d.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/4ace40228e1754476d201b4363d4200862ee92b1b47549b4e5039e963de5ce9d.jpg)

![4edaeb14ff36e687ec4963ee98d305f3134587e7c75e49fcae95b645123050aa.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/4edaeb14ff36e687ec4963ee98d305f3134587e7c75e49fcae95b645123050aa.jpg)

![524044bdc5ad0f4418051c1a37612ded892d715d1ada0ba36c66f6434e795862.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/524044bdc5ad0f4418051c1a37612ded892d715d1ada0ba36c66f6434e795862.jpg)

![5dab61bc6efdfcd837f8d0202c1a1b647942c5f42ce23d99023c98991dffa8c0.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/5dab61bc6efdfcd837f8d0202c1a1b647942c5f42ce23d99023c98991dffa8c0.jpg)

![66451297b0263cfc67b049502e2d6adaff4aa65313dbda57dda6868755cabd4f.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/66451297b0263cfc67b049502e2d6adaff4aa65313dbda57dda6868755cabd4f.jpg)

![73ad217194eb618942f9aed230b4c4d7eeea5d11a208efb4c6fd256179d3e01f.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/73ad217194eb618942f9aed230b4c4d7eeea5d11a208efb4c6fd256179d3e01f.jpg)

![75812789604e595d81ecaa36f22fadbac2005fb2520931822ddfe4401f877af4.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/75812789604e595d81ecaa36f22fadbac2005fb2520931822ddfe4401f877af4.jpg)

![7ff9ee67b708980d20d74d46fc58c3e8c09e0d0199153b7f9125fae475506027.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/7ff9ee67b708980d20d74d46fc58c3e8c09e0d0199153b7f9125fae475506027.jpg)

![94292bffd7cb29f6900124fd599359ac2869745601fe3ef8d657cea264537aef.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/94292bffd7cb29f6900124fd599359ac2869745601fe3ef8d657cea264537aef.jpg)

![ada5fdc590fbb298f701fa91b9d7f82ab033a761897baf17ceb1686a650f9622.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/ada5fdc590fbb298f701fa91b9d7f82ab033a761897baf17ceb1686a650f9622.jpg)

![afcc1d0cca77e9f9a18986c983d69b5fd28eab53d0f4d41d3fc9663c2a363d04.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/afcc1d0cca77e9f9a18986c983d69b5fd28eab53d0f4d41d3fc9663c2a363d04.jpg)

![b613481dc6a63b834eb8aac650c4ea02bf31378b67821a609f20b5d44b7d0ac2.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/b613481dc6a63b834eb8aac650c4ea02bf31378b67821a609f20b5d44b7d0ac2.jpg)

![b941e56500bfe31606f501032d007fd9dbb8d71661c8f255f2cf353dbdf0430a.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/b941e56500bfe31606f501032d007fd9dbb8d71661c8f255f2cf353dbdf0430a.jpg)

![cec49af304cb21d4939b897aaa7199bf5d0bb5a00c4fac9156e2cc91378beb61.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/cec49af304cb21d4939b897aaa7199bf5d0bb5a00c4fac9156e2cc91378beb61.jpg)

![d4b0dec3b9b26e1203c21e8d84140e30265bbd714f49f017d2d1b42982555a9d.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/d4b0dec3b9b26e1203c21e8d84140e30265bbd714f49f017d2d1b42982555a9d.jpg)

![d9b7d95b7b4d43747bab1201a8fd45e345a7835a57797b8c159141ab7e5319f9.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/d9b7d95b7b4d43747bab1201a8fd45e345a7835a57797b8c159141ab7e5319f9.jpg)

![e3c07398d41d27fc80a9cf41dc6826f564fa824162e13982f0a07b48b2420337.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/e3c07398d41d27fc80a9cf41dc6826f564fa824162e13982f0a07b48b2420337.jpg)

![ebbb6dca6e434e6b06c0b9c424e8583fa99b90283b98ac9788ab11c7a3515817.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/ebbb6dca6e434e6b06c0b9c424e8583fa99b90283b98ac9788ab11c7a3515817.jpg)

![fb7ac5fcfe389f59c7133882a53094b11efd7e2d69fb55fad6b49506663e5624.jpg](../icml_results/5979_SemBind_%20Binding%20Diffusion%20Watermarks%20to%20Semantics%20Against%20Black-Box%20Forgery%20Attacks/tables/fb7ac5fcfe389f59c7133882a53094b11efd7e2d69fb55fad6b49506663e5624.jpg)

<a id="paper-5931"></a>

## BTSP-CAM: A Brain-Inspired Geometric Memory for Class-Incremental Learning


### Images

![8b8939bb6811a2cc1cae5fd1e8e4bbd874efa3ad891a4285bf97a42ac553c2dd.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/images/8b8939bb6811a2cc1cae5fd1e8e4bbd874efa3ad891a4285bf97a42ac553c2dd.jpg)

![91f1cc85ec905281609baae93f9175b0eb8804da58cce1739d6a3548e0dba182.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/images/91f1cc85ec905281609baae93f9175b0eb8804da58cce1739d6a3548e0dba182.jpg)

![bb2b89e25aee74b6bbae471d04241e120b8e5ddeb2bb355e4cb4fc65275784ea.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/images/bb2b89e25aee74b6bbae471d04241e120b8e5ddeb2bb355e4cb4fc65275784ea.jpg)

![d8a671edbe2abc639e8c1930bb9f5e5acb88b679c9c2abf4b002badb7f423115.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/images/d8a671edbe2abc639e8c1930bb9f5e5acb88b679c9c2abf4b002badb7f423115.jpg)

### Tables

![45703a99c25937efe2bf856f8d17043a7b4bae1065bfe34a3fc61b8be9e39dd6.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/45703a99c25937efe2bf856f8d17043a7b4bae1065bfe34a3fc61b8be9e39dd6.jpg)

![505fcd86f48366f7426cc12e450950c8abeee0f0c135214d95d2f7206943db93.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/505fcd86f48366f7426cc12e450950c8abeee0f0c135214d95d2f7206943db93.jpg)

![52e0ec316abd0b22fcfea89d1e96fe19c359e4d0afeed38e8b747aa0e58b172d.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/52e0ec316abd0b22fcfea89d1e96fe19c359e4d0afeed38e8b747aa0e58b172d.jpg)

![59fb578407c13069c2a6722f31ea500b0c232511d962ae6dc6ec62f45e24ccb4.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/59fb578407c13069c2a6722f31ea500b0c232511d962ae6dc6ec62f45e24ccb4.jpg)

![65cfc7a3eee70bc906c5e8036b2e9067e0fa3e83d988ef089913a27004a96c6e.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/65cfc7a3eee70bc906c5e8036b2e9067e0fa3e83d988ef089913a27004a96c6e.jpg)

![af8bf35bb2bdc13d7f8cac51aa4a78dc0e831656fb93fa7574af77e8af8cb648.jpg](../icml_results/5980_BTSP-CAM_%20A%20Brain-Inspired%20Geometric%20Memory%20for%20Class-Incremental%20Learning/tables/af8bf35bb2bdc13d7f8cac51aa4a78dc0e831656fb93fa7574af77e8af8cb648.jpg)

<a id="paper-5932"></a>

## Focus, Align, and Sustain: Counteracting Gradient Dilution in Incremental Object Detection


### Images

![0c051646a106025160f11b42a9c18b31528496378c287f620c38c2294a823d8b.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/0c051646a106025160f11b42a9c18b31528496378c287f620c38c2294a823d8b.jpg)

![10245b288574967e6d859d69afa651278fe73c0eaeef34c39d61283d342b030e.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/10245b288574967e6d859d69afa651278fe73c0eaeef34c39d61283d342b030e.jpg)

![1e28d7558b805ec5b1777257efce75de1e919606416e7ba9720e6d7a4ad63d26.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/1e28d7558b805ec5b1777257efce75de1e919606416e7ba9720e6d7a4ad63d26.jpg)

![5cde2471d318d03a844bf3c8262150985397b72dc9f24e2fddaa5168910732a8.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/5cde2471d318d03a844bf3c8262150985397b72dc9f24e2fddaa5168910732a8.jpg)

![685db69a416d4b5e92a3fdde8cb20d34998f8a63d51ddde0faadd70b84dd1841.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/685db69a416d4b5e92a3fdde8cb20d34998f8a63d51ddde0faadd70b84dd1841.jpg)

![91325bf58eb1fee2c8c6084131f7de88ab05489525623d790575343d836e7b1b.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/91325bf58eb1fee2c8c6084131f7de88ab05489525623d790575343d836e7b1b.jpg)

![b25067bb4d1ac6e64824f8d9fdb52f279a46f7adff7ef58d635c216b3dae57e6.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/b25067bb4d1ac6e64824f8d9fdb52f279a46f7adff7ef58d635c216b3dae57e6.jpg)

![e932151937eec034144d317497aa3d78e6318aaee3283aed36126f5c4eee210c.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/images/e932151937eec034144d317497aa3d78e6318aaee3283aed36126f5c4eee210c.jpg)

### Tables

![03023d88744bfc6db9d8ffc3c039480e8abbd52c82febee1c4b94b8a40488d4a.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/tables/03023d88744bfc6db9d8ffc3c039480e8abbd52c82febee1c4b94b8a40488d4a.jpg)

![2dfef6f51086ff7660cccf07c4f4f494d0c61332a27a3c0c648571e3ccd6f9ea.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/tables/2dfef6f51086ff7660cccf07c4f4f494d0c61332a27a3c0c648571e3ccd6f9ea.jpg)

![74affddc95220376c862f0599353d228953c4eedb80861f70961f1165962b8a1.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/tables/74affddc95220376c862f0599353d228953c4eedb80861f70961f1165962b8a1.jpg)

![8a5f7ba46ebc690a3e8b2a3102b3bcc0f82970cd1d67e8dda778571879e6d52d.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/tables/8a5f7ba46ebc690a3e8b2a3102b3bcc0f82970cd1d67e8dda778571879e6d52d.jpg)

![fb9f8f6b4a7a91cb9f97dab6d9f452c078c5ba3f06dc453abbe3da7306e12864.jpg](../icml_results/5981_Focus%2C%20Align%2C%20and%20Sustain_%20Counteracting%20Gradient%20Dilution%20in%20Incremental%20Object%20Detection/tables/fb9f8f6b4a7a91cb9f97dab6d9f452c078c5ba3f06dc453abbe3da7306e12864.jpg)

<a id="paper-5933"></a>

## Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution


### Images

![066d4fd62ef4648424881fd186ff009d6933d69a096a4b2dd7fe61a3b890cbaa.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/066d4fd62ef4648424881fd186ff009d6933d69a096a4b2dd7fe61a3b890cbaa.jpg)

![299ac614fd8ae8aae8488e80971708a2f0e2c7d0530966543ec397343331b178.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/299ac614fd8ae8aae8488e80971708a2f0e2c7d0530966543ec397343331b178.jpg)

![2d1cf3de9f3ca44cfd28b3dead97787e94af9711323f4d43ba3bd1ec9863a5ed.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/2d1cf3de9f3ca44cfd28b3dead97787e94af9711323f4d43ba3bd1ec9863a5ed.jpg)

![538fd8f65403abd1026b75373c05a92c9e22611c49ca3274e99a86daebdfc9d9.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/538fd8f65403abd1026b75373c05a92c9e22611c49ca3274e99a86daebdfc9d9.jpg)

![7c8bbbed186a4430f17e9cc50d7165b721cb49abd02403f67805f3aa49e7ec62.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/7c8bbbed186a4430f17e9cc50d7165b721cb49abd02403f67805f3aa49e7ec62.jpg)

![9fed61e552cef6e686e5b6001f6269faa459f4a1f9b05286b0652a5de38132ba.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/9fed61e552cef6e686e5b6001f6269faa459f4a1f9b05286b0652a5de38132ba.jpg)

![de5545db6355630be53cd242010fcce268eede9f4565c83edbbc4d8e5badaa62.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/de5545db6355630be53cd242010fcce268eede9f4565c83edbbc4d8e5badaa62.jpg)

![e4eea423ffc8ea8e91726fa463ce0b11a46ec92db3841eee2f3127cb0e5de6c0.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/e4eea423ffc8ea8e91726fa463ce0b11a46ec92db3841eee2f3127cb0e5de6c0.jpg)

![fbe32d294d98896e08e14a267331025716aadfff3acb53705718d7da3a7f03de.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/images/fbe32d294d98896e08e14a267331025716aadfff3acb53705718d7da3a7f03de.jpg)

### Tables

![24b5b2e1247f1dad11ed3a602688eb229a432d4414723d75b740f383b72577a1.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/24b5b2e1247f1dad11ed3a602688eb229a432d4414723d75b740f383b72577a1.jpg)

![5dbcf814f2a6d2e97542ef59dbebc1123c8ffa6b18d4f7b41a5bfc1047468bc8.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/5dbcf814f2a6d2e97542ef59dbebc1123c8ffa6b18d4f7b41a5bfc1047468bc8.jpg)

![82074a1540df05955c5bedc91bdf6f72f80b5dc3c862d523abe669474ec25fac.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/82074a1540df05955c5bedc91bdf6f72f80b5dc3c862d523abe669474ec25fac.jpg)

![8d0db329f2934a8a70e68c7772114ad94a9947cd8e77a726d5bef64adfdae40c.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/8d0db329f2934a8a70e68c7772114ad94a9947cd8e77a726d5bef64adfdae40c.jpg)

![a75c9901e12ff0462d9ce44001a4afd048ed60b8ec2765edbb8fbe153f5ca42c.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/a75c9901e12ff0462d9ce44001a4afd048ed60b8ec2765edbb8fbe153f5ca42c.jpg)

![c5395b667349d5100fc1fb738cd6adecbc79a2b35d9451841945b81a616dc521.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/c5395b667349d5100fc1fb738cd6adecbc79a2b35d9451841945b81a616dc521.jpg)

![c6a5922835505526d7fd87607390e59c374d1a380383735c4c535d6e7b3630b3.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/c6a5922835505526d7fd87607390e59c374d1a380383735c4c535d6e7b3630b3.jpg)

![cf1592f07a60ab6b13dad34daca73f23cfd1d11938429402fcd271e858f45c4c.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/cf1592f07a60ab6b13dad34daca73f23cfd1d11938429402fcd271e858f45c4c.jpg)

![e3852d2c80af2151eb8b1a1ff4786a812db8ad87f8d5a9882cdf607dcb5879b2.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/e3852d2c80af2151eb8b1a1ff4786a812db8ad87f8d5a9882cdf607dcb5879b2.jpg)

![f0692ef02fa9793df0110c217c00e39680ea5603b2949cef859362cdb6264aea.jpg](../icml_results/5982_Q-DiT4SR_%20Exploration%20of%20Detail-Preserving%20Diffusion%20Transformer%20Quantization%20for%20Real-World%20Image%20S/tables/f0692ef02fa9793df0110c217c00e39680ea5603b2949cef859362cdb6264aea.jpg)

<a id="paper-5934"></a>

## Active Tabular Augmentation via Policy-Guided Diffusion Inpainting


### Images

![0894894243ef90a4930a86bfbfe8742b9aedc61b7212e8c12e09bb5ebdf55513.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/0894894243ef90a4930a86bfbfe8742b9aedc61b7212e8c12e09bb5ebdf55513.jpg)

![1dcc72c13b4bb003057c4b6c494e67c1fef7b51af85d2bf8b40139d1fc311ab3.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/1dcc72c13b4bb003057c4b6c494e67c1fef7b51af85d2bf8b40139d1fc311ab3.jpg)

![22a925738243347b0d5f77d678201fcd36921ac3436c7d225b7a65916ae7e0ea.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/22a925738243347b0d5f77d678201fcd36921ac3436c7d225b7a65916ae7e0ea.jpg)

![8fdf44f02d1c5ee3a0a4f3d3e4557981c8efd7e390b2428df8e80bc8b69fb2c7.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/8fdf44f02d1c5ee3a0a4f3d3e4557981c8efd7e390b2428df8e80bc8b69fb2c7.jpg)

![dca1fb6ed0e5d4af3773a5bd305978e6de6a3f141707b34d1ebee2b82f2cc834.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/dca1fb6ed0e5d4af3773a5bd305978e6de6a3f141707b34d1ebee2b82f2cc834.jpg)

![fd0184ddd6e2759fa2e85e3bcda41a8e21b57434ab1424541965322adb506634.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/images/fd0184ddd6e2759fa2e85e3bcda41a8e21b57434ab1424541965322adb506634.jpg)

### Tables

![0dbb82afc6afc4d5338e4e50f5ab7bd52ab4bd5c6fd4e696515f47c885330eb2.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/0dbb82afc6afc4d5338e4e50f5ab7bd52ab4bd5c6fd4e696515f47c885330eb2.jpg)

![1e55025e8f8968887dc7ae8e17b6f23f68d664014be97df3c8d8a6f2d1c53523.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/1e55025e8f8968887dc7ae8e17b6f23f68d664014be97df3c8d8a6f2d1c53523.jpg)

![24834dc11cc2b192fc0bc36ff35f94401d1659b30424c8402f2d19336b4be9d8.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/24834dc11cc2b192fc0bc36ff35f94401d1659b30424c8402f2d19336b4be9d8.jpg)

![283fe5e5a55568fe4b0ca2ab4e34b2c90c095bec253e271c2e1730bd88d34c72.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/283fe5e5a55568fe4b0ca2ab4e34b2c90c095bec253e271c2e1730bd88d34c72.jpg)

![33ab6b7ad03ccbd218abfa7290ace869c211cc015e21d30ea1b6a0712b9ed7d4.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/33ab6b7ad03ccbd218abfa7290ace869c211cc015e21d30ea1b6a0712b9ed7d4.jpg)

![390c9cb4d0475b1a4b04c17db7721ea3f7ca66a7d936d6ee7884db42db9407cd.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/390c9cb4d0475b1a4b04c17db7721ea3f7ca66a7d936d6ee7884db42db9407cd.jpg)

![3a162dc9658b7e75d2ce8c59524e25a550686aca6b24be8d23a7b9173353ff47.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/3a162dc9658b7e75d2ce8c59524e25a550686aca6b24be8d23a7b9173353ff47.jpg)

![42ce85a5a5593b2f64510b8afbd1707cc9d216aa8dffdee0e14d686825d04a3e.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/42ce85a5a5593b2f64510b8afbd1707cc9d216aa8dffdee0e14d686825d04a3e.jpg)

![4752b770b60dd6a232759891e310ae84ea19e9391a58c56a664dfc142999f67d.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/4752b770b60dd6a232759891e310ae84ea19e9391a58c56a664dfc142999f67d.jpg)

![544aab8d21bfe97b09723aa496a91c3c756463d0a17df46c12084f2feb69858c.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/544aab8d21bfe97b09723aa496a91c3c756463d0a17df46c12084f2feb69858c.jpg)

![67b2f2ad89d63102137ad1fa25edaf36720ed42155abc1373b30723d87bb34a9.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/67b2f2ad89d63102137ad1fa25edaf36720ed42155abc1373b30723d87bb34a9.jpg)

![71a46040f62883148c0cb42c5f424f3adc1cdd556a944221db136e8f45fc4f94.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/71a46040f62883148c0cb42c5f424f3adc1cdd556a944221db136e8f45fc4f94.jpg)

![748b1cea305df458bb715b39eb24fcbe6903698706ce69fc849ca70cb49658c2.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/748b1cea305df458bb715b39eb24fcbe6903698706ce69fc849ca70cb49658c2.jpg)

![7eb381318e1cd95bb196bd91d542ee7507274afdc8ab7fcfbd3faea70d6c3dde.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/7eb381318e1cd95bb196bd91d542ee7507274afdc8ab7fcfbd3faea70d6c3dde.jpg)

![971a293c7ae067606747977d98c5f2fe3fc3bf5ddac3cf72379fd4c539f873bb.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/971a293c7ae067606747977d98c5f2fe3fc3bf5ddac3cf72379fd4c539f873bb.jpg)

![cc53817eea9c1977f786b073f205494595c94d530188ed909b2057dc660c8b20.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/cc53817eea9c1977f786b073f205494595c94d530188ed909b2057dc660c8b20.jpg)

![ddcbccaecc27b08ed560f916c91fb461dcc35931e5dd4fe317c8fc72f93aba08.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/ddcbccaecc27b08ed560f916c91fb461dcc35931e5dd4fe317c8fc72f93aba08.jpg)

![f60c0bcf543614f976073839a0ab84d54ae371d61761d97f23366bde5a70f80d.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/f60c0bcf543614f976073839a0ab84d54ae371d61761d97f23366bde5a70f80d.jpg)

![f8764788bde64261fbb5f071e920c68a9e06532adbabf5c647011c99fd620037.jpg](../icml_results/5983_Active%20Tabular%20Augmentation%20via%20Policy-Guided%20Diffusion%20Inpainting/tables/f8764788bde64261fbb5f071e920c68a9e06532adbabf5c647011c99fd620037.jpg)

<a id="paper-5935"></a>

## Global Policy-Space Response Oracles for Two-Player Zero-Sum Games


### Images

![6dab08c4ebd18369569293be19c390accbc16339aeb4668d448a404a7956eaf4.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/6dab08c4ebd18369569293be19c390accbc16339aeb4668d448a404a7956eaf4.jpg)

![877dd684ac8b1492c30bf22ea9c512adf4472117562e7b90d85d0784497b9e6d.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/877dd684ac8b1492c30bf22ea9c512adf4472117562e7b90d85d0784497b9e6d.jpg)

![878e0aa8dcff0e271b2522336b4f4c24f59287679f631f994650638212588295.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/878e0aa8dcff0e271b2522336b4f4c24f59287679f631f994650638212588295.jpg)

![9605adac41351e7bd7a56198308a473a6f1774d0c0fbd53212e893e1c7718272.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/9605adac41351e7bd7a56198308a473a6f1774d0c0fbd53212e893e1c7718272.jpg)

![96d72eae1e6e75f4e03fd64ff92753aab4f7580e41931922f1858e2538946d83.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/96d72eae1e6e75f4e03fd64ff92753aab4f7580e41931922f1858e2538946d83.jpg)

![a1a9ce27ba6aefe149f0d2b2cfa976480469de26549d22e76b8bfd76d465bc5d.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/a1a9ce27ba6aefe149f0d2b2cfa976480469de26549d22e76b8bfd76d465bc5d.jpg)

![a5ffcec2d9d07298b0809db3159b57f08fa15c694b14c99dea2fce996affd5f2.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/a5ffcec2d9d07298b0809db3159b57f08fa15c694b14c99dea2fce996affd5f2.jpg)

![acd0bc48587eea565d8f154775b640bf4cf2183efbd8581d3fb10771f41a3c5a.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/acd0bc48587eea565d8f154775b640bf4cf2183efbd8581d3fb10771f41a3c5a.jpg)

![b805c91520ee01cd9c357b9d2214c50c901b40a4067724bc2424f09a3026ff28.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/b805c91520ee01cd9c357b9d2214c50c901b40a4067724bc2424f09a3026ff28.jpg)

![c38ea21c9c606338ac34522536cbb05508d00874912564eada24894223313a36.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/c38ea21c9c606338ac34522536cbb05508d00874912564eada24894223313a36.jpg)

![d8b01e625b74bb4c4248251af0bb5b12b88ce16500b66c2b0d156629932c4e67.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/d8b01e625b74bb4c4248251af0bb5b12b88ce16500b66c2b0d156629932c4e67.jpg)

![ee50580dbe9654f3416e075ddbbf9e6a32bd8714a1b1814b9a07e2f4aa62a5fe.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/images/ee50580dbe9654f3416e075ddbbf9e6a32bd8714a1b1814b9a07e2f4aa62a5fe.jpg)

### Tables

![2c3a33d1edebc9830cc3764a7f47e25fec3c79b1216166c43d40fbdad0d878ac.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/2c3a33d1edebc9830cc3764a7f47e25fec3c79b1216166c43d40fbdad0d878ac.jpg)

![3b6bbd3153bd5872df4406aded03992e69478e3e0f1a46d5686bc0470a9eccc4.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/3b6bbd3153bd5872df4406aded03992e69478e3e0f1a46d5686bc0470a9eccc4.jpg)

![3c698cc63c7e5e9616bf46811f9208f60bf4438555fea96591ee80b804e08c21.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/3c698cc63c7e5e9616bf46811f9208f60bf4438555fea96591ee80b804e08c21.jpg)

![6240e51155c607b0fca36013953b387a9540c52c61f8805505831f4c0a040773.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/6240e51155c607b0fca36013953b387a9540c52c61f8805505831f4c0a040773.jpg)

![657c18e81c15bec63586da490815063e5f2fc3738bf5eacbb0ce074804714df7.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/657c18e81c15bec63586da490815063e5f2fc3738bf5eacbb0ce074804714df7.jpg)

![8b8ac5b2b69315d76782663e2be37e809bbac719af738420aae2fbbab3c1fcea.jpg](../icml_results/5984_Global%20Policy-Space%20Response%20Oracles%20for%20Two-Player%20Zero-Sum%20Games/tables/8b8ac5b2b69315d76782663e2be37e809bbac719af738420aae2fbbab3c1fcea.jpg)

<a id="paper-5936"></a>

## A Game-Theoretic Framework for Measuring and Explaining Metric Compatibility in Fair Machine Learning


### Images

![566500d58aa6cf38a94b875e6b477bd03b3e22e27bc9fe8877a9ac55cd12f9e1.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/566500d58aa6cf38a94b875e6b477bd03b3e22e27bc9fe8877a9ac55cd12f9e1.jpg)

![9aaaeed45dcdb4cf933a7bd8818babf407ae33f7528834a4d3ab7da0d7509970.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/9aaaeed45dcdb4cf933a7bd8818babf407ae33f7528834a4d3ab7da0d7509970.jpg)

![a9ed412bd6eeb5efecd23b2e57ce588aa51c89466e71c84b68645eed249f66f4.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/a9ed412bd6eeb5efecd23b2e57ce588aa51c89466e71c84b68645eed249f66f4.jpg)

![ee3f34846857015c2c1f1dba879ad4db3d5f0bbb2724b0f64fc7b473a63a3bce.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/ee3f34846857015c2c1f1dba879ad4db3d5f0bbb2724b0f64fc7b473a63a3bce.jpg)

![ee5cc78bb2f721519949c3c357ccb03d7be232716e45080e7fbcb390493d0b79.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/ee5cc78bb2f721519949c3c357ccb03d7be232716e45080e7fbcb390493d0b79.jpg)

![f5cca2bf64c796f9ec02c38e6332ee72796e8fd214419263f472db5a3a461672.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/f5cca2bf64c796f9ec02c38e6332ee72796e8fd214419263f472db5a3a461672.jpg)

![fc2250efb69db87f52581a10525dddfc4f48bbee2d7ffec6b2e3fcffa0559de7.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/images/fc2250efb69db87f52581a10525dddfc4f48bbee2d7ffec6b2e3fcffa0559de7.jpg)

### Tables

![0e7eb5d9608cc08df74e89bfc113d94f253d70bce8692222623691bd072af927.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/0e7eb5d9608cc08df74e89bfc113d94f253d70bce8692222623691bd072af927.jpg)

![3027178b2376e0a97951cf844eeb31266f8f785b358ffc9556004fa700e2908e.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/3027178b2376e0a97951cf844eeb31266f8f785b358ffc9556004fa700e2908e.jpg)

![4c32f3b640389f20036f5d703653c4653b0d216395f1c6f27567e08cf9e996f0.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/4c32f3b640389f20036f5d703653c4653b0d216395f1c6f27567e08cf9e996f0.jpg)

![73623b5b11953af263c18d6d0f3882b5d32db9ababf9a6a1996c4660be074c81.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/73623b5b11953af263c18d6d0f3882b5d32db9ababf9a6a1996c4660be074c81.jpg)

![7415598f31e1473436121f0f60b5fe067bd6761dd8cc342d837191a3d13f16d0.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/7415598f31e1473436121f0f60b5fe067bd6761dd8cc342d837191a3d13f16d0.jpg)

![79a6b973b8134c1a3326945bca91868262f1a977c503dd4c3d1984ae15b3be42.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/79a6b973b8134c1a3326945bca91868262f1a977c503dd4c3d1984ae15b3be42.jpg)

![8143e6b65f9eefffbaca32e9383f2aca0e57ee1182b6f59db36eca56f843c2a4.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/8143e6b65f9eefffbaca32e9383f2aca0e57ee1182b6f59db36eca56f843c2a4.jpg)

![96f4184ad2cdb17a95672db210a4a4c6a61054493dd24432fdae749775aeaca6.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/96f4184ad2cdb17a95672db210a4a4c6a61054493dd24432fdae749775aeaca6.jpg)

![a77fe136ab45fe7c0e48402ab419a909aee100173a14763dd7dcb70355b0b041.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/a77fe136ab45fe7c0e48402ab419a909aee100173a14763dd7dcb70355b0b041.jpg)

![cd74c9fc9948dbe76e9f5e16eb6747c7a81d23928f8407f3c38da4abcc26f775.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/cd74c9fc9948dbe76e9f5e16eb6747c7a81d23928f8407f3c38da4abcc26f775.jpg)

![d268e25ac9e320b6d0c075f8ee5c2b9718a3432bf7c5a4262265cb21a5c55895.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/d268e25ac9e320b6d0c075f8ee5c2b9718a3432bf7c5a4262265cb21a5c55895.jpg)

![d6e8a842ed9589f95ffa6de277f4b73a9b71d6b675ed8f1965ea63d40f25add1.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/d6e8a842ed9589f95ffa6de277f4b73a9b71d6b675ed8f1965ea63d40f25add1.jpg)

![f5bbbdf7e6eb1d4d96a0a6c42f116e0bf99edefac982627b2cabb7a0d34e463b.jpg](../icml_results/5985_A%20Game-Theoretic%20Framework%20for%20Measuring%20and%20Explaining%20Metric%20Compatibility%20in%20Fair%20Machine%20Learnin/tables/f5bbbdf7e6eb1d4d96a0a6c42f116e0bf99edefac982627b2cabb7a0d34e463b.jpg)

<a id="paper-5937"></a>

## GenAlign: Towards Unified Alignment Framework of MLLMs via Generative Reward Model


### Images

![5440bcd0258c6197b73f1f865f4324d05ec9122718361dd5a223db433810b8f0.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/5440bcd0258c6197b73f1f865f4324d05ec9122718361dd5a223db433810b8f0.jpg)

![6ed29a9abfdb0f395f7c11fc20f8837ff65db3528bf8b162e3e0c02d44601c51.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/6ed29a9abfdb0f395f7c11fc20f8837ff65db3528bf8b162e3e0c02d44601c51.jpg)

![8b455e09aa717579f1461ff279906f35f851bfd1c615ded71b8f63937769daed.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/8b455e09aa717579f1461ff279906f35f851bfd1c615ded71b8f63937769daed.jpg)

![8f9631ef2f427b0bd63bb9b85ae21534fadd3dbc61e0a9a1d47dc7ae7056103d.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/8f9631ef2f427b0bd63bb9b85ae21534fadd3dbc61e0a9a1d47dc7ae7056103d.jpg)

![9f016d0c1d15b224d28edab22a216ce2e52725f7cfa222c56e260c61d2d06735.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/9f016d0c1d15b224d28edab22a216ce2e52725f7cfa222c56e260c61d2d06735.jpg)

![9f07613aa627656e34fd8792b8816bf8b7378c3bb2b442d2ea06fe57fd66db2c.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/9f07613aa627656e34fd8792b8816bf8b7378c3bb2b442d2ea06fe57fd66db2c.jpg)

![be986aa6769930e301b267e494068a9c613022c16730b770a351adf109de1100.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/be986aa6769930e301b267e494068a9c613022c16730b770a351adf109de1100.jpg)

![e4981c51a406bdd2d329b7e9740e6002f66548593952a746c66525166d7fbe92.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/images/e4981c51a406bdd2d329b7e9740e6002f66548593952a746c66525166d7fbe92.jpg)

### Tables

![3cd1d88544d8438dd959c4b54d2143c2a8dacbd0d732dd7ee98c84dd93f046dd.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/3cd1d88544d8438dd959c4b54d2143c2a8dacbd0d732dd7ee98c84dd93f046dd.jpg)

![5dda82c57a5c00a3c2ad224335e9ad6b7e78e6d970251d0f0a38cf074f45cf6e.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/5dda82c57a5c00a3c2ad224335e9ad6b7e78e6d970251d0f0a38cf074f45cf6e.jpg)

![95b8f0a3519ac789d5f4c51b6bf8eaaccd4707994b026e1818dcf810504143e2.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/95b8f0a3519ac789d5f4c51b6bf8eaaccd4707994b026e1818dcf810504143e2.jpg)

![e651f8c83c5f54ad5cdd05f36a261fab3c77bb540b6f3d242c32380efe9c487b.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/e651f8c83c5f54ad5cdd05f36a261fab3c77bb540b6f3d242c32380efe9c487b.jpg)

![e8c906ad309b43a35262ae9d321af227296f3aec4a556d04451ccf35a1426b98.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/e8c906ad309b43a35262ae9d321af227296f3aec4a556d04451ccf35a1426b98.jpg)

![f954c148261cf9be1fda7db1659531bde245fcfef43c521a50139e75f5dc5ac2.jpg](../icml_results/5986_GenAlign_%20Towards%20Unified%20Alignment%20Framework%20of%20MLLMs%20via%20Generative%20Reward%20Model/tables/f954c148261cf9be1fda7db1659531bde245fcfef43c521a50139e75f5dc5ac2.jpg)

<a id="paper-5938"></a>

## AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models


### Images

![25f1141353142227d68a37ab58d22bb77f9c9bc5e4cda749be339dcbd36bfe0a.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/25f1141353142227d68a37ab58d22bb77f9c9bc5e4cda749be339dcbd36bfe0a.jpg)

![3289cf02ecba4cc95100fa689b604456cfbcc589cb8369ddf545beec7dee1c53.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/3289cf02ecba4cc95100fa689b604456cfbcc589cb8369ddf545beec7dee1c53.jpg)

![358947c63fee709553e77cc121da8c7d8c483741d27e7a159cbe30ee5bd20593.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/358947c63fee709553e77cc121da8c7d8c483741d27e7a159cbe30ee5bd20593.jpg)

![535e51a0c13fa1648dea4a278f5a73ced9bb9ba71d330450473bd8f1d61d31ce.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/535e51a0c13fa1648dea4a278f5a73ced9bb9ba71d330450473bd8f1d61d31ce.jpg)

![57fa025b57e051f20d40fd2ff340721875def4730e8bd93e4a739d1b3ff6a4db.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/57fa025b57e051f20d40fd2ff340721875def4730e8bd93e4a739d1b3ff6a4db.jpg)

![8651d00fbaca44f3633b4e789a29e3e3eab4f36f278e0c6c4e36bea8a13db89f.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/8651d00fbaca44f3633b4e789a29e3e3eab4f36f278e0c6c4e36bea8a13db89f.jpg)

![b0f207b8e59fb061e435d064943e62651366e8f9e06ec1a995cbff843b840b19.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/b0f207b8e59fb061e435d064943e62651366e8f9e06ec1a995cbff843b840b19.jpg)

![b26005d47a6d77a8e3bc4e4f827595766c27f56c7571cd6a35a0597024c0ca65.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/b26005d47a6d77a8e3bc4e4f827595766c27f56c7571cd6a35a0597024c0ca65.jpg)

![b6b2a9b2c0624dc16834c006170f670e6e7321fe7467b48ec817ad1bb9b488e3.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/images/b6b2a9b2c0624dc16834c006170f670e6e7321fe7467b48ec817ad1bb9b488e3.jpg)

### Tables

![050069ad7c192608262c9935b03251f2eb274131989a1fb83240ad85ec091dc5.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/tables/050069ad7c192608262c9935b03251f2eb274131989a1fb83240ad85ec091dc5.jpg)

![3b854824e0083b8684dc2156e7000e033daeb3e3fec80f3ed9af000a473a5d30.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/tables/3b854824e0083b8684dc2156e7000e033daeb3e3fec80f3ed9af000a473a5d30.jpg)

![4bdfec9c7adbe056fe0e1e7918e3259c6ae601b22de423a268e48b400d601534.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/tables/4bdfec9c7adbe056fe0e1e7918e3259c6ae601b22de423a268e48b400d601534.jpg)

![562321768330c1b7cbb777de68058cbb9f98b85973588e7ce0b99c86171a5bd8.jpg](../icml_results/5987_AReaL-DTA_%20Dynamic%20Tree%20Attention%20for%20Efficient%20Reinforcement%20Learning%20of%20Large%20Language%20Models/tables/562321768330c1b7cbb777de68058cbb9f98b85973588e7ce0b99c86171a5bd8.jpg)

<a id="paper-5939"></a>

## Cross-Modal Semantic Decoupling and Transfer for Text-to-Visible-Infrared Person Re-Identification


### Images

![0832f02a253cb054bf6d98f831e810297d7671e55e2ab6c3bee16e99e61c1673.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/0832f02a253cb054bf6d98f831e810297d7671e55e2ab6c3bee16e99e61c1673.jpg)

![26bcea937cf8987f2bdb40fab81a90973d4d1349de1d27ede65caed9722e2176.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/26bcea937cf8987f2bdb40fab81a90973d4d1349de1d27ede65caed9722e2176.jpg)

![5115680773f6ce1367d4d49eb2d53d47c099ebe581a078a7990e7c78756b73a2.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/5115680773f6ce1367d4d49eb2d53d47c099ebe581a078a7990e7c78756b73a2.jpg)

![56ec6b5fa451698461a1873e2e2277276fc26bfebc7d3d0751ff0687e88fb9b3.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/56ec6b5fa451698461a1873e2e2277276fc26bfebc7d3d0751ff0687e88fb9b3.jpg)

![5defd6a40049f060dbb2eb659fde0ca86223fe822c4d364cadab2319f4949c17.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/5defd6a40049f060dbb2eb659fde0ca86223fe822c4d364cadab2319f4949c17.jpg)

![62988ed5eddebdf99bf8e450c07a90e29958b0526d641cbfd18429c22870af57.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/62988ed5eddebdf99bf8e450c07a90e29958b0526d641cbfd18429c22870af57.jpg)

![6c0afd137fbf1df236a1afcfdad15069accc92fa720acb8dc72b86cbe435a21f.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/6c0afd137fbf1df236a1afcfdad15069accc92fa720acb8dc72b86cbe435a21f.jpg)

![6d450bfb857846b842e458e2d9fcce005b311db5c472c90631c5b683a63efaf3.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/6d450bfb857846b842e458e2d9fcce005b311db5c472c90631c5b683a63efaf3.jpg)

![84c55f76e7605a5404132611cbc652bc3c78669140a6f10cde455d8ee0ff4b5a.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/84c55f76e7605a5404132611cbc652bc3c78669140a6f10cde455d8ee0ff4b5a.jpg)

![8d01a2f916523861bdb71113c41145fe67a1ff4fe257fd872f1613bddd1780b8.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/8d01a2f916523861bdb71113c41145fe67a1ff4fe257fd872f1613bddd1780b8.jpg)

![8db77006c652c6a771b9062a75d5905a53999f5e8f90075d445ad60126f2f287.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/8db77006c652c6a771b9062a75d5905a53999f5e8f90075d445ad60126f2f287.jpg)

![c4323eade3bb6e254254e7ef532cbfab8326811012439dfbc9c7338cc4d6b4a8.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/c4323eade3bb6e254254e7ef532cbfab8326811012439dfbc9c7338cc4d6b4a8.jpg)

![e6806c5bc1dd78b8b3af09bb28f9e598bbcd8d937e0a0b06202302ec5f0b1c98.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/images/e6806c5bc1dd78b8b3af09bb28f9e598bbcd8d937e0a0b06202302ec5f0b1c98.jpg)

### Tables

![1026259b64a9d7296029209c7fe4f4d143348fb6d3666a9b04b2211f422f1b99.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/tables/1026259b64a9d7296029209c7fe4f4d143348fb6d3666a9b04b2211f422f1b99.jpg)

![3758d488d963ef81ae43b52b7e2a527fe2f7d19378d4dfd7eb7f148dcfdbe93b.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/tables/3758d488d963ef81ae43b52b7e2a527fe2f7d19378d4dfd7eb7f148dcfdbe93b.jpg)

![cb917de69c8ce4e1d33f92c1e64c457fb0ad3e9035c4b4de715bd260085a43c9.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/tables/cb917de69c8ce4e1d33f92c1e64c457fb0ad3e9035c4b4de715bd260085a43c9.jpg)

![e274fde53da0e9dccf22e97273a357b06266c25ef3e469808a66aca2649035f8.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/tables/e274fde53da0e9dccf22e97273a357b06266c25ef3e469808a66aca2649035f8.jpg)

![f8d5476f9d2b5a45b274f1f7e1ee9c4b87113fa84ca4e20b7cf34f9d3dde780e.jpg](../icml_results/5988_Cross-Modal%20Semantic%20Decoupling%20and%20Transfer%20for%20Text-to-Visible-Infrared%20Person%20Re-Identification/tables/f8d5476f9d2b5a45b274f1f7e1ee9c4b87113fa84ca4e20b7cf34f9d3dde780e.jpg)

<a id="paper-5940"></a>

## Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook Strategy

### Images

![19650c97e977950113882568cece9ad7b6556fb99995a05cec496114080b2144.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/19650c97e977950113882568cece9ad7b6556fb99995a05cec496114080b2144.jpg)

![2ee79b3e080ef32120b6528aa1b16b5ed5a94c71345daec42eb469c8da0c2a3d.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/2ee79b3e080ef32120b6528aa1b16b5ed5a94c71345daec42eb469c8da0c2a3d.jpg)

![3774039eba0b0cf86285e1c9850f5c8a71b971e72d1b941e6e44eaefca6f5aa3.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/3774039eba0b0cf86285e1c9850f5c8a71b971e72d1b941e6e44eaefca6f5aa3.jpg)

![56363f967aeac445dcbdabd551dadd1c63ba8954023fc45bdc88a2fc7d525a9a.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/56363f967aeac445dcbdabd551dadd1c63ba8954023fc45bdc88a2fc7d525a9a.jpg)

![6afe9b5961ba9d8ae2cd95dfb2165a888432e110eda444e24187b2f2728b94bf.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/6afe9b5961ba9d8ae2cd95dfb2165a888432e110eda444e24187b2f2728b94bf.jpg)

![7b5361967950cb5d490d2436d137c573175bcb8c610125d93aae2464644d1fa5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/7b5361967950cb5d490d2436d137c573175bcb8c610125d93aae2464644d1fa5.jpg)

![95fc80dd22dd0bf39ae08286f274ddf1b35a0217d54ae9136b7b3c822b4b55fa.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/95fc80dd22dd0bf39ae08286f274ddf1b35a0217d54ae9136b7b3c822b4b55fa.jpg)

![b5b01c84f082f07e56ed7a61b5adcb0e00eadb57bedd4e587da7e521bf41c2c5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/b5b01c84f082f07e56ed7a61b5adcb0e00eadb57bedd4e587da7e521bf41c2c5.jpg)

![c5be1340424a3bd77c0a97701212fa87b8b1a217df7b7c3061c35734fa3e2a84.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/c5be1340424a3bd77c0a97701212fa87b8b1a217df7b7c3061c35734fa3e2a84.jpg)

![e1f8fc50c480a0533ce5b704a092ca25f4fbfb33dc5826344dd5ca09a2142cd2.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e1f8fc50c480a0533ce5b704a092ca25f4fbfb33dc5826344dd5ca09a2142cd2.jpg)

![e2b3b2eca9f52c639dc00363ce834bf1d1cc99745033ae7cd7607ce8ea578daf.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e2b3b2eca9f52c639dc00363ce834bf1d1cc99745033ae7cd7607ce8ea578daf.jpg)

![e7a19b07b133070682ef53daae1939e7858f4c820e47eb0bf00a0f495176bf13.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e7a19b07b133070682ef53daae1939e7858f4c820e47eb0bf00a0f495176bf13.jpg)

![edbc8997bd75788267dbe1d42e7682d961285c0f0b85bcc4ac6cb867014fb401.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/edbc8997bd75788267dbe1d42e7682d961285c0f0b85bcc4ac6cb867014fb401.jpg)

### Tables

![25d2d0f805f9902663475d308cd3a614ef35292ee9e955c019c1b5803ed19f22.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/25d2d0f805f9902663475d308cd3a614ef35292ee9e955c019c1b5803ed19f22.jpg)

![786d948135960e981ca06ce8452fbd6ff28e4305a6fb0bf0f99dda6f380d1bc3.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/786d948135960e981ca06ce8452fbd6ff28e4305a6fb0bf0f99dda6f380d1bc3.jpg)

![7efadef478ceab2be94ead9e2ea45af11fe36024197acf922a282cfaea772518.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/7efadef478ceab2be94ead9e2ea45af11fe36024197acf922a282cfaea772518.jpg)

![90a20a456bd8f1ca8bf36e3deb47c55dd783d163b1f10fb0816c2dec74841e5b.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/90a20a456bd8f1ca8bf36e3deb47c55dd783d163b1f10fb0816c2dec74841e5b.jpg)

![9ad9b8e69cb64405d54e761195aa1f8165e93bfbf1231d7d876a1a6fb5d6f21b.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/9ad9b8e69cb64405d54e761195aa1f8165e93bfbf1231d7d876a1a6fb5d6f21b.jpg)

![bd1ab632c9b728053f67cd3c14b2f8e1738d02e9307c817a3a636c688d0d98b5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/bd1ab632c9b728053f67cd3c14b2f8e1738d02e9307c817a3a636c688d0d98b5.jpg)

![e08fe0338fe0cb28bad6243a92cb36875d6b558a14f53274ab5dfdd9c5d40a4e.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/e08fe0338fe0cb28bad6243a92cb36875d6b558a14f53274ab5dfdd9c5d40a4e.jpg)

![eebc3d6c87548f933bc7bac865f7f39db4d32b9aa16f8bf861b58ae139a2b5e4.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/eebc3d6c87548f933bc7bac865f7f39db4d32b9aa16f8bf861b58ae139a2b5e4.jpg)
