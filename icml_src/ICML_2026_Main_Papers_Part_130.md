# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 130 of 197

## 目录 (Table of Contents)

1. [HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning](#paper-4258)
2. [Domain Transfer Becomes Identifiable via a Single Alignment](#paper-4259)
3. [Size Transferability of Graph Convolutional Networks across Sparsity: A Generalized Graphon Perspective](#paper-4260)
4. [HybridOM: Hybrid Physics-Based and Data-Driven Global Ocean Modeling with Efficient Spatial Downscaling](#paper-4261)
5. [OSF: On Pre-training and Scaling of Sleep Foundation Models](#paper-4262)
6. [The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency](#paper-4263)
7. [General Analysis of LMO-based Optimizers: Beyond Bounded Variance](#paper-4264)
8. [Understanding MARS: When Scaling Momentum Correction Provably Helps](#paper-4265)
9. [Two-dimensional quantization for geometry-aware audio coding](#paper-4266)
10. [FLAG: Foundation model representation with Latent diffusion Alignment via Graph for spatial gene expression prediction](#paper-4267)
11. [Towards Execution-Grounded Automated AI Research](#paper-4268)
12. [Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement](#paper-4269)
13. [Geometry-Aware Probabilistic Circuits via Voronoi Tessellations](#paper-4270)
14. [Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](#paper-4271)
15. [Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](#paper-4272)
16. [3DPoV: Improving 3D understanding via Patch Ordering on Videos](#paper-4273)
17. [Training with Honeypots: Reshaping How LLMs Fail Under Adversarial Attacks](#paper-4274)
18. [On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](#paper-4275)
19. [Unsupervised Disentanglement Without Compromises : How Functional Orthogonality Enforces Identifiability](#paper-4276)
20. [Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies](#paper-4277)
21. [Optimal Fair Aggregation of Crowdsourced Noisy Labels using Demographic Parity Constraints](#paper-4278)
22. [Curriculum-Guided Layer Scaling for Language Model Pretraining](#paper-4279)
23. [V1: Unifying Generation and Self-Verification for Parallel Reasoners](#paper-4280)
24. [RL with Learnable Textual Feedback: A Bilevel Approach](#paper-4281)
25. [DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone](#paper-4282)
26. [Evaluating Object-Centric Models beyond Object Discovery](#paper-4283)
27. [VIBE: Disentangling Social Dynamics via Kinematics-Informed Variational Inference for Behavioral Emotion](#paper-4284)
28. [Divide and Learn: Multi-Objective Combinatorial Optimization at Scale](#paper-4285)
29. [LLM Priors for ERM over Programs](#paper-4286)
30. [Accurate, Private, Secure, Federated U-statistics with Higher Degree](#paper-4287)
31. [Trainable Nonexpansive Denoisers for Contractive Image Reconstruction](#paper-4288)
32. [Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling](#paper-4289)
33. [Implicit Intelligence - Evaluating Agents on What Users Don’t Say](#paper-4290)

---


<a id="paper-4258"></a>

## HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning

### Images

![0cea3b0a44eb2d5d131152dc669dc091847d8b8737d023e1538bedbd1efabafd.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/0cea3b0a44eb2d5d131152dc669dc091847d8b8737d023e1538bedbd1efabafd.jpg)

![0d3f560c5eb87d60d3d901732448bdf72c438d78caf19519839ad37e4ead2741.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/0d3f560c5eb87d60d3d901732448bdf72c438d78caf19519839ad37e4ead2741.jpg)

![106f637a270bfaed70a7436ce3edf143f7bc7e9057679041e65468d6b148e125.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/106f637a270bfaed70a7436ce3edf143f7bc7e9057679041e65468d6b148e125.jpg)

![14006cf82cb79942e29eaed74ca28a3411291fe13571bcaf6771385145e36f3c.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/14006cf82cb79942e29eaed74ca28a3411291fe13571bcaf6771385145e36f3c.jpg)

![21de9e6b616d9fcbc8cd6bfe4a65d591f08ca38dae7f5038e66f3607cbb96cd3.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/21de9e6b616d9fcbc8cd6bfe4a65d591f08ca38dae7f5038e66f3607cbb96cd3.jpg)

![2b1cdb06c853ad5c5cc83d0bc7e0840222e730fa60c5669e63226ef7157dfaf4.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/2b1cdb06c853ad5c5cc83d0bc7e0840222e730fa60c5669e63226ef7157dfaf4.jpg)

![53ffb9b72107a47c46cd78093f934d1ae9d35522e9bd518de6492485982529d5.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/53ffb9b72107a47c46cd78093f934d1ae9d35522e9bd518de6492485982529d5.jpg)

![84fd90eb4fd5e151ec2663a83e4940138282dd518f041fc208a57eecc1405abb.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/84fd90eb4fd5e151ec2663a83e4940138282dd518f041fc208a57eecc1405abb.jpg)

![b967780d945fb883f0b5e3506424a64904012962f29a1c1bc4e47fe050723466.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/b967780d945fb883f0b5e3506424a64904012962f29a1c1bc4e47fe050723466.jpg)

![b99398431381aca6c38897bfd5cf3ef1bc50001d9ad1bce53eda5790ea602a15.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/b99398431381aca6c38897bfd5cf3ef1bc50001d9ad1bce53eda5790ea602a15.jpg)

![c346a838af7b86dc4223f8ac79d9a19fa98b37ba4db66ddfe724e30ce1603277.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/c346a838af7b86dc4223f8ac79d9a19fa98b37ba4db66ddfe724e30ce1603277.jpg)

![d18cec520324a8e4213dfa50922a6dbda7074ba18f13f1cafab748a24f0a9c66.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/d18cec520324a8e4213dfa50922a6dbda7074ba18f13f1cafab748a24f0a9c66.jpg)

![de6bb9c10ac3d9822ed35137b2945064780874f5d52669f95afe9d304e79cb05.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/de6bb9c10ac3d9822ed35137b2945064780874f5d52669f95afe9d304e79cb05.jpg)

![f743c3ec7a659fa057e0797a818d9c19ee885398fefa7d3e70c56728f2cb240a.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/f743c3ec7a659fa057e0797a818d9c19ee885398fefa7d3e70c56728f2cb240a.jpg)

![fcec80744a22cb31b4dd7a0dc9a19d36072da6d8d36616d07e7fc335086f22c1.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/fcec80744a22cb31b4dd7a0dc9a19d36072da6d8d36616d07e7fc335086f22c1.jpg)

### Tables

![3cdb45d98fce117e0df1df242fb26986ff12f4f26b0e90f15d181d96f9d47df4.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/3cdb45d98fce117e0df1df242fb26986ff12f4f26b0e90f15d181d96f9d47df4.jpg)

![45c48d84cfd2618ff2455c33ce19a01db027c4232a57b85620043bfd9d78d3ee.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/45c48d84cfd2618ff2455c33ce19a01db027c4232a57b85620043bfd9d78d3ee.jpg)

![49284e161b784d8fe3faab761d20c7fc853fe4ba6c5144628015bf1ef9b4be58.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/49284e161b784d8fe3faab761d20c7fc853fe4ba6c5144628015bf1ef9b4be58.jpg)

![5086adcd66b8fbd620777f3d7c99375d8711036499fac342d82e523644f5f721.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/5086adcd66b8fbd620777f3d7c99375d8711036499fac342d82e523644f5f721.jpg)

![5506bf6ddef1a752f4de1af1d58601b30e9ded0124ec9aaebf63b1c5082fdd47.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/5506bf6ddef1a752f4de1af1d58601b30e9ded0124ec9aaebf63b1c5082fdd47.jpg)

![69cbb45e462d17f66545409c673ecfb5304733d971739d2a37e878fe37aeb325.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/69cbb45e462d17f66545409c673ecfb5304733d971739d2a37e878fe37aeb325.jpg)

![7f9ae42269e16f7bd4a353ecac751420d3a08d1bfa7da6bab6742c81e2e2c548.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/7f9ae42269e16f7bd4a353ecac751420d3a08d1bfa7da6bab6742c81e2e2c548.jpg)

![89ac3fb83504eb91b51df945091fdd440f9de38bf091dc5ae0d6ebd94e38753a.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/89ac3fb83504eb91b51df945091fdd440f9de38bf091dc5ae0d6ebd94e38753a.jpg)

![941025c795c52483c100e2d2d8a455243238724aa39bf4e870d9332c4eea8b4f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/941025c795c52483c100e2d2d8a455243238724aa39bf4e870d9332c4eea8b4f.jpg)

![afa29f2117594022db3620a3c252fa188ada6e661edee49214ba95ff77d9099f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/afa29f2117594022db3620a3c252fa188ada6e661edee49214ba95ff77d9099f.jpg)

![d72aff98c3d74830b1e75eb3ac3773bd2f34b507928a4a12e1a4174edb78ab7e.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/d72aff98c3d74830b1e75eb3ac3773bd2f34b507928a4a12e1a4174edb78ab7e.jpg)

![e6425815bec40b66cd82f06385d0da489b793107b38bccd6d8d7d708638e4b2e.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/e6425815bec40b66cd82f06385d0da489b793107b38bccd6d8d7d708638e4b2e.jpg)

![ea6025b69d6789b71ba271238fc932823fd9a69c3556e747ac5568828d9daa5f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/ea6025b69d6789b71ba271238fc932823fd9a69c3556e747ac5568828d9daa5f.jpg)

## HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning


### Images

![00892616eb11d18266feb09486031b9a63acb193be231067f5a1a100be25a5a8.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/00892616eb11d18266feb09486031b9a63acb193be231067f5a1a100be25a5a8.jpg)

![32e039fc2a31d2d6b04c1c63a98d07a960be641eec604c34143e0fbfd5ebf8ac.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/32e039fc2a31d2d6b04c1c63a98d07a960be641eec604c34143e0fbfd5ebf8ac.jpg)

![49c4ee541e56fac67f9cc7c3ccedab6895b675e99f8b784dcae0827f7e8ae03f.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/49c4ee541e56fac67f9cc7c3ccedab6895b675e99f8b784dcae0827f7e8ae03f.jpg)

![67b20eb98ddbf7df940334c39ae4e9b3385302e3d73c51c66812836691ebdf62.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/67b20eb98ddbf7df940334c39ae4e9b3385302e3d73c51c66812836691ebdf62.jpg)

![8143b8a28dac4f78feefd63aec9ad82adf1edcca4a13c70427fb1509b7144735.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/8143b8a28dac4f78feefd63aec9ad82adf1edcca4a13c70427fb1509b7144735.jpg)

![830a0e41acdad397bf703737d342ddb178eb44f1f6c7e97672c376dcc7bcd487.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/830a0e41acdad397bf703737d342ddb178eb44f1f6c7e97672c376dcc7bcd487.jpg)

![a6102ecda504cdfd9acde7263c93c36096a36139e597f52ac6d3ac76c31ec9a8.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/a6102ecda504cdfd9acde7263c93c36096a36139e597f52ac6d3ac76c31ec9a8.jpg)

![c2c969bb1ce92581013c214f655a402a6d9b68659c6a5ffacc4849e9e02f48f7.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/c2c969bb1ce92581013c214f655a402a6d9b68659c6a5ffacc4849e9e02f48f7.jpg)

![f671dc84319e82e0a90813ba358e74255f274b48dec258b36d497bedada7786c.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/images/f671dc84319e82e0a90813ba358e74255f274b48dec258b36d497bedada7786c.jpg)

### Tables

![01d89c04f5d6fc95e11d0bb8b62a703ac5762912e2bb1241f2597e6c277d096f.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/01d89c04f5d6fc95e11d0bb8b62a703ac5762912e2bb1241f2597e6c277d096f.jpg)

![5565b4d83d72bc82e95d3a625628053946755073f6002bed98907dce8e92a498.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/5565b4d83d72bc82e95d3a625628053946755073f6002bed98907dce8e92a498.jpg)

![5ab5db2d4843778241c3dd2a5df2748c5e3a99131fa6b6fcc28cc6167abe3e64.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/5ab5db2d4843778241c3dd2a5df2748c5e3a99131fa6b6fcc28cc6167abe3e64.jpg)

![73dd54c36b4650a801fc6dd8843c62794332dc60c1247eca3010d1df78f804bf.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/73dd54c36b4650a801fc6dd8843c62794332dc60c1247eca3010d1df78f804bf.jpg)

![806fd222555e41b047a3f40bc55050b7b6c55b18bad43d3d81f91d9b6a6175a0.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/806fd222555e41b047a3f40bc55050b7b6c55b18bad43d3d81f91d9b6a6175a0.jpg)

![8750edec9e36cdaa0f14ef4c9b9170f647a8b8ed790bcc1238f9aaf5b3ac4e5b.jpg](../icml_results/4301_HALO_%20A%20Unified%20Vision-Language-Action%20Model%20for%20Embodied%20Multimodal%20Chain-of-Thought%20Reasoning/tables/8750edec9e36cdaa0f14ef4c9b9170f647a8b8ed790bcc1238f9aaf5b3ac4e5b.jpg)

<a id="paper-4259"></a>

## Domain Transfer Becomes Identifiable via a Single Alignment


### Images

![1b0081a0cfe8776c09575fb088e509017d99cd06f93864f52f7378ecf77a7bc7.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/1b0081a0cfe8776c09575fb088e509017d99cd06f93864f52f7378ecf77a7bc7.jpg)

![51588d45012a667603f9e7aea9e61e99bcd3972eae5d6a679f19f473f2474051.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/51588d45012a667603f9e7aea9e61e99bcd3972eae5d6a679f19f473f2474051.jpg)

![5856772770e104f24f95c2a2d74c528cba82e2661f545f2d09b5751622142721.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/5856772770e104f24f95c2a2d74c528cba82e2661f545f2d09b5751622142721.jpg)

![59701b043417c342e8e55964e5c3a43d5fe84de0d9f17c5772ee23df87981780.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/59701b043417c342e8e55964e5c3a43d5fe84de0d9f17c5772ee23df87981780.jpg)

![60e3e2b3c3df810cf3e40e591e45a1238f6ab56437153ace97c59f9e0df9fd9c.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/60e3e2b3c3df810cf3e40e591e45a1238f6ab56437153ace97c59f9e0df9fd9c.jpg)

![76a61bb8137c99522fb433b9ee483e5b761cfb68ff6dc542f56ea6a8414efc13.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/76a61bb8137c99522fb433b9ee483e5b761cfb68ff6dc542f56ea6a8414efc13.jpg)

![7e8ffc9d60854119c8ea088bc70521f933596cf99f9563a7eb4a4b7bba4e5613.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/7e8ffc9d60854119c8ea088bc70521f933596cf99f9563a7eb4a4b7bba4e5613.jpg)

![9dfead239bc4328ea473e027a3f594353280b68cf6f08d5838f89bd3e688dbfd.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/9dfead239bc4328ea473e027a3f594353280b68cf6f08d5838f89bd3e688dbfd.jpg)

![a8fffd7168091781675c514af3d9ab706995ec13ff52497017a0dd832f221642.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/a8fffd7168091781675c514af3d9ab706995ec13ff52497017a0dd832f221642.jpg)

![bec0cbcd843436da51b520affc526abec48d820aebd8315948cd320a0e7fdb9f.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/images/bec0cbcd843436da51b520affc526abec48d820aebd8315948cd320a0e7fdb9f.jpg)

### Tables

![0f8d8e785da1227b05942ab94c8bf68e4c28afbe6f33ba56c11ad2891b3f7543.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/0f8d8e785da1227b05942ab94c8bf68e4c28afbe6f33ba56c11ad2891b3f7543.jpg)

![1fae8f4bffccbe42d5cb9daf1131988820b88e590f754daf9daa76afdd90baf1.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/1fae8f4bffccbe42d5cb9daf1131988820b88e590f754daf9daa76afdd90baf1.jpg)

![242f616c9a5dbbd4ae047d1d7a3df6b9e99de87d412fb90e6b65f52f7c7bc783.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/242f616c9a5dbbd4ae047d1d7a3df6b9e99de87d412fb90e6b65f52f7c7bc783.jpg)

![42d782364f3efed595feeed7d31b280bdfe05287932f3632f7f3400722106bea.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/42d782364f3efed595feeed7d31b280bdfe05287932f3632f7f3400722106bea.jpg)

![b6afc6a33bfd8dabc15efc9a8f044eb46a234b365d81c5110c00cae3dda00d4f.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/b6afc6a33bfd8dabc15efc9a8f044eb46a234b365d81c5110c00cae3dda00d4f.jpg)

![eb2e0805007115780fe977fe383db7c0af8afa9c5064b315318796db6e578733.jpg](../icml_results/4302_Domain%20Transfer%20Becomes%20Identifiable%20via%20a%20Single%20Alignment/tables/eb2e0805007115780fe977fe383db7c0af8afa9c5064b315318796db6e578733.jpg)

<a id="paper-4260"></a>

## Size Transferability of Graph Convolutional Networks across Sparsity: A Generalized Graphon Perspective


### Images

![06888a37297184c2d8b9545e2e1dc9d356c319b0fc313137fe969f85637c5664.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/06888a37297184c2d8b9545e2e1dc9d356c319b0fc313137fe969f85637c5664.jpg)

![11d098ce4915d759d1dc0c9c04a5ff82b57da470dedf35d86cede749860fb5cd.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/11d098ce4915d759d1dc0c9c04a5ff82b57da470dedf35d86cede749860fb5cd.jpg)

![60dfa0a8833f1ac62c89dfcac3ed5a11904b3fdd8495a2f9ed0413da309d42e9.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/60dfa0a8833f1ac62c89dfcac3ed5a11904b3fdd8495a2f9ed0413da309d42e9.jpg)

![723caf81f4e0891a40a9efde82383e953e758988898f43619cc6862d32008b45.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/723caf81f4e0891a40a9efde82383e953e758988898f43619cc6862d32008b45.jpg)

![b8032402e46e99dd80b50a9a85575045d07dcb6856856663f98879f283f89069.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/b8032402e46e99dd80b50a9a85575045d07dcb6856856663f98879f283f89069.jpg)

![ddddf9f8ef162aec871f85a058469e1fccadf96e5bac974af73adcc46f76a174.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/ddddf9f8ef162aec871f85a058469e1fccadf96e5bac974af73adcc46f76a174.jpg)

![dec8081ee5336a95085d53452605ad467450e6ad7f84543fc4c5745eef688d9d.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/dec8081ee5336a95085d53452605ad467450e6ad7f84543fc4c5745eef688d9d.jpg)

![ed1b2c4f86300c802a2c47462b8963c8543fce4db6f92868933caa8669381e8d.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/ed1b2c4f86300c802a2c47462b8963c8543fce4db6f92868933caa8669381e8d.jpg)

![f9d879b487677c1fd4774ffcae034e704725e5d1ffb0ff653ad1be460f534ab9.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/images/f9d879b487677c1fd4774ffcae034e704725e5d1ffb0ff653ad1be460f534ab9.jpg)

### Tables

![c2bf0fde9d7cfa3a2ea2f79572b38abe74f9afb3e59ef94207734f745db8bd19.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/tables/c2bf0fde9d7cfa3a2ea2f79572b38abe74f9afb3e59ef94207734f745db8bd19.jpg)

![f6631347344c3e2c254284a759d77026b07728794c8af5c904ce5eac81c29f64.jpg](../icml_results/4303_Size%20Transferability%20of%20Graph%20Convolutional%20Networks%20across%20Sparsity_%20A%20Generalized%20Graphon%20Perspect/tables/f6631347344c3e2c254284a759d77026b07728794c8af5c904ce5eac81c29f64.jpg)

<a id="paper-4261"></a>

## HybridOM: Hybrid Physics-Based and Data-Driven Global Ocean Modeling with Efficient Spatial Downscaling


### Images

![1f7c81385a713f1bae5a08212b6fbc2541a61641ec9cf32279c4f041ffdee1d1.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/1f7c81385a713f1bae5a08212b6fbc2541a61641ec9cf32279c4f041ffdee1d1.jpg)

![3310427c399364e9b5dd2f46420e85cd5c00af36566f755b59fb50264dcdea02.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/3310427c399364e9b5dd2f46420e85cd5c00af36566f755b59fb50264dcdea02.jpg)

![3ad558a403771cf62acd02c7d61f79a2225a27fa304e4e4c112a281e276b64fd.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/3ad558a403771cf62acd02c7d61f79a2225a27fa304e4e4c112a281e276b64fd.jpg)

![48e6362e459fa8be1a743e366892fb3f4c1e15d6be2f4d8fd03ebe8ab39833d6.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/48e6362e459fa8be1a743e366892fb3f4c1e15d6be2f4d8fd03ebe8ab39833d6.jpg)

![72143bc8e8d5c964a958bcba3243feda8faedf94ebd4889a029aeb86f2418af0.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/72143bc8e8d5c964a958bcba3243feda8faedf94ebd4889a029aeb86f2418af0.jpg)

![934642bfb58346d4d5a0a798b72c0747a7ca1e190bed98a6fe56349033440915.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/934642bfb58346d4d5a0a798b72c0747a7ca1e190bed98a6fe56349033440915.jpg)

![aa1797fa7de6c606307a0ad65b4e11e46bf0886593b1eabd194d1ee6fa4afb49.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/aa1797fa7de6c606307a0ad65b4e11e46bf0886593b1eabd194d1ee6fa4afb49.jpg)

![b696178ae2dbdf5c276568c5e1309e01092d23d0469ae0e663755bafd77c01c4.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/b696178ae2dbdf5c276568c5e1309e01092d23d0469ae0e663755bafd77c01c4.jpg)

![cc57fdeee84de3408bf97790c705a5492a957702bbe199538c799b25b3c3bcae.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/cc57fdeee84de3408bf97790c705a5492a957702bbe199538c799b25b3c3bcae.jpg)

![e6e7886ea7ce4473ba0dbeb4bba5455d83d93968589eb7030569c05ddebdbfb3.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/e6e7886ea7ce4473ba0dbeb4bba5455d83d93968589eb7030569c05ddebdbfb3.jpg)

![f281f51370aef5954614010edd7d0e346947ed73466d14c3eca9e657c40ad7cc.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/f281f51370aef5954614010edd7d0e346947ed73466d14c3eca9e657c40ad7cc.jpg)

![f99b5749784255e0fb86c7f4d2bad84ef42e4cfafda6b551ee96143aa47ae7fc.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/images/f99b5749784255e0fb86c7f4d2bad84ef42e4cfafda6b551ee96143aa47ae7fc.jpg)

### Tables

![2d3df4430755ab19c1ba2a90abded1223231a7e06a46a0e2c39e19efcd7b2b78.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/2d3df4430755ab19c1ba2a90abded1223231a7e06a46a0e2c39e19efcd7b2b78.jpg)

![41869070b3dd06114a8b01c29ffc1746fe9c6e948c81ce73771363b8c4df23c8.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/41869070b3dd06114a8b01c29ffc1746fe9c6e948c81ce73771363b8c4df23c8.jpg)

![4194e2579d6a41b8fe3ef3fc2999faa8e7fc47dae068add1237e1404803d9037.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/4194e2579d6a41b8fe3ef3fc2999faa8e7fc47dae068add1237e1404803d9037.jpg)

![453931875e62b689353cc894041e005928a258558499f392314ab66b872fc411.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/453931875e62b689353cc894041e005928a258558499f392314ab66b872fc411.jpg)

![4584e8938f6a6f99afc086969ffd8b4b5c6dfe465c45c0305062dffc1894be7f.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/4584e8938f6a6f99afc086969ffd8b4b5c6dfe465c45c0305062dffc1894be7f.jpg)

![4bc11898ed11d4c3acc98049866865f75859911104645a91c6e5b1915315081b.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/4bc11898ed11d4c3acc98049866865f75859911104645a91c6e5b1915315081b.jpg)

![587b78ccb4094bf345d54687efbc5235bcbc25c2c9e2827125edc97b3cdd1a63.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/587b78ccb4094bf345d54687efbc5235bcbc25c2c9e2827125edc97b3cdd1a63.jpg)

![6a104d00513ebabcf956901b0cc64d40509c9ec9de3d447ac3e3839697eac8ec.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/6a104d00513ebabcf956901b0cc64d40509c9ec9de3d447ac3e3839697eac8ec.jpg)

![6b64caff4acc011fade27ad09a09c73438918d8871a1b32529fe3a498e7b2b88.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/6b64caff4acc011fade27ad09a09c73438918d8871a1b32529fe3a498e7b2b88.jpg)

![a4dc91b7c409b36fc802c43ad6c2986e2e9b6cd4ff71242d1f4b833d7deb5ea9.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/a4dc91b7c409b36fc802c43ad6c2986e2e9b6cd4ff71242d1f4b833d7deb5ea9.jpg)

![ba5af004c579decd3d49f56e6eeb0d796d67a2ac5e357774f85b313488c86faf.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/ba5af004c579decd3d49f56e6eeb0d796d67a2ac5e357774f85b313488c86faf.jpg)

![bca88668dfe429ee3f34335694bd3a93e3a9bb60fd53f72d25931a1c36bc8b57.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/bca88668dfe429ee3f34335694bd3a93e3a9bb60fd53f72d25931a1c36bc8b57.jpg)

![c8d4c6d11a8e97cb67008086007ea4c4cc1aaa85dbd3e7a091a4285246e73ffd.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/c8d4c6d11a8e97cb67008086007ea4c4cc1aaa85dbd3e7a091a4285246e73ffd.jpg)

![e4b49d148b707d5f915c7a704828440369edaaa844db329dc8133b7619065791.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/e4b49d148b707d5f915c7a704828440369edaaa844db329dc8133b7619065791.jpg)

![e88452b17611700e70c4dde48c84c623e43c31122ff6890ade72311b5f32c14b.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/e88452b17611700e70c4dde48c84c623e43c31122ff6890ade72311b5f32c14b.jpg)

![ee191e597ed02c70d4576bd8607dfb1a040b40b466434c38a539c0da461d360f.jpg](../icml_results/4304_HybridOM_%20Hybrid%20Physics-Based%20and%20Data-Driven%20Global%20Ocean%20Modeling%20with%20Efficient%20Spatial%20Downscal/tables/ee191e597ed02c70d4576bd8607dfb1a040b40b466434c38a539c0da461d360f.jpg)

<a id="paper-4262"></a>

## OSF: On Pre-training and Scaling of Sleep Foundation Models


### Images

![004c21cd7aecf90b126e178fa2e5f3e19397f4091462bf686bfef119251304c3.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/004c21cd7aecf90b126e178fa2e5f3e19397f4091462bf686bfef119251304c3.jpg)

![1ee0e49a09649bfed12873773f5596f634554e5c9a69e0053130d16f79d6c3d0.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/1ee0e49a09649bfed12873773f5596f634554e5c9a69e0053130d16f79d6c3d0.jpg)

![20c0c7d3ffb696e3226e28f0c5dc5f27a90b793fc27e484194f3e68a77079a7f.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/20c0c7d3ffb696e3226e28f0c5dc5f27a90b793fc27e484194f3e68a77079a7f.jpg)

![24a53f924b1742b1185cc86a2d8f1e6ca0034d16ccda462693839d08b7ddfed5.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/24a53f924b1742b1185cc86a2d8f1e6ca0034d16ccda462693839d08b7ddfed5.jpg)

![2c4ec9ca38d1386e5e58916e724e36fddb23b91fd291d8bc384c6677ef2c3499.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/2c4ec9ca38d1386e5e58916e724e36fddb23b91fd291d8bc384c6677ef2c3499.jpg)

![2d67dd443aabd24a963656d417da76c91385a3d511367fc048ad7ed75acef516.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/2d67dd443aabd24a963656d417da76c91385a3d511367fc048ad7ed75acef516.jpg)

![361f83c0651cd173126fdb334c4b5b50ef9740b088f4af3d87b4dd324375db4e.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/361f83c0651cd173126fdb334c4b5b50ef9740b088f4af3d87b4dd324375db4e.jpg)

![4c75ffd3ee276505a89edcb0329872815cd4a9beb4d2bf50fda5e59ae6375c6f.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/4c75ffd3ee276505a89edcb0329872815cd4a9beb4d2bf50fda5e59ae6375c6f.jpg)

![70e0b157284b2ba155aea8b9cc540d9cd2915164b08660585d318a2bf481c5c0.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/70e0b157284b2ba155aea8b9cc540d9cd2915164b08660585d318a2bf481c5c0.jpg)

![820a4e3787ebacbf0c7dd0d06a7d77aef652254d5125c503167833c032feb042.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/820a4e3787ebacbf0c7dd0d06a7d77aef652254d5125c503167833c032feb042.jpg)

![ad053ff3a553eceb875dad343119df4ffa1640a3b99640cb103e6d72c0cb0603.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/ad053ff3a553eceb875dad343119df4ffa1640a3b99640cb103e6d72c0cb0603.jpg)

![c60fc5b482132e66ed5da9b77f87acd3695d1aedc4df73e5d50ebdbe686525b1.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/c60fc5b482132e66ed5da9b77f87acd3695d1aedc4df73e5d50ebdbe686525b1.jpg)

![cc5939ccdb0ee47e2cc1059ecd1fe6898cb4014372ff394a2c106a637a386699.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/cc5939ccdb0ee47e2cc1059ecd1fe6898cb4014372ff394a2c106a637a386699.jpg)

![df9c5d128738a5f0b0667a7ea0c982e1d0c03f68c3cc359a30ec773be79f05ff.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/df9c5d128738a5f0b0667a7ea0c982e1d0c03f68c3cc359a30ec773be79f05ff.jpg)

![f09b333ba7f9e00736af653ca7cbe0ce00db1355e0c2a27c961bd3807b74483f.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/images/f09b333ba7f9e00736af653ca7cbe0ce00db1355e0c2a27c961bd3807b74483f.jpg)

### Tables

![0dbb109ca182538a62835f6a7a7ad1d334029d18e2f505ce32dc89ad6e105611.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/0dbb109ca182538a62835f6a7a7ad1d334029d18e2f505ce32dc89ad6e105611.jpg)

![13835308add76db5fbedcf0e42dc4d925426aa84ba3383f405e40c0bcf430096.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/13835308add76db5fbedcf0e42dc4d925426aa84ba3383f405e40c0bcf430096.jpg)

![235fd88a438d636231f092a3ebb6322074322850a151a4432ee75ac68d9e8501.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/235fd88a438d636231f092a3ebb6322074322850a151a4432ee75ac68d9e8501.jpg)

![2ad178a680cccd312b8d54e63a5883a52d2f37687fc303e5d02066f69bb2b2eb.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/2ad178a680cccd312b8d54e63a5883a52d2f37687fc303e5d02066f69bb2b2eb.jpg)

![2b8c5e525176600da0d3422180653aa7dcbad5913f87d0ae1a782f5c743911ef.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/2b8c5e525176600da0d3422180653aa7dcbad5913f87d0ae1a782f5c743911ef.jpg)

![2c83319c5ac8a750857156aaef0e99e8f5cf62c520d738f8fe2515679392ad23.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/2c83319c5ac8a750857156aaef0e99e8f5cf62c520d738f8fe2515679392ad23.jpg)

![3259515ba8deadba925bb0ebd9aa3af5d5e4dc13b77037888de44f87064bef75.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/3259515ba8deadba925bb0ebd9aa3af5d5e4dc13b77037888de44f87064bef75.jpg)

![36cf46654a48650f40abd1ec1cbd3606cd8edb5b46f59588d7c9ef1751ed5743.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/36cf46654a48650f40abd1ec1cbd3606cd8edb5b46f59588d7c9ef1751ed5743.jpg)

![3e3a40cfe61e21c357a6db9b5a9d3bc4b9a31674ec75118af23031f028ae43cd.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/3e3a40cfe61e21c357a6db9b5a9d3bc4b9a31674ec75118af23031f028ae43cd.jpg)

![40bfdc13fc9ecce38fc98f2e24def91d3f01d1e7346170069b43e338804c819d.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/40bfdc13fc9ecce38fc98f2e24def91d3f01d1e7346170069b43e338804c819d.jpg)

![412a5b6eaea6d378cd169c8b55435bde366d47d5fdf900d14a9fd9e9da74c444.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/412a5b6eaea6d378cd169c8b55435bde366d47d5fdf900d14a9fd9e9da74c444.jpg)

![434197a1d61215858bb34f59dc62306624eee758f2f61e5bee46720289f67135.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/434197a1d61215858bb34f59dc62306624eee758f2f61e5bee46720289f67135.jpg)

![49230e68f9c0202861551dcf73d4642f83874e697ea114d7b6148de82cc4e790.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/49230e68f9c0202861551dcf73d4642f83874e697ea114d7b6148de82cc4e790.jpg)

![4a22ff1400fc8206b206590e68d75ece84d3c26b6b65489c5696fe56bfb0596b.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/4a22ff1400fc8206b206590e68d75ece84d3c26b6b65489c5696fe56bfb0596b.jpg)

![4cb08aca62aea0ea54be5d0acd50b42a9f69c82f00670cf4f9c1cc474ae0f110.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/4cb08aca62aea0ea54be5d0acd50b42a9f69c82f00670cf4f9c1cc474ae0f110.jpg)

![58a6f250baf56414916217512edbca83083c52a300ffc174c15401b4c54610a1.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/58a6f250baf56414916217512edbca83083c52a300ffc174c15401b4c54610a1.jpg)

![61ce4568e84513906699d112f2850abce2742c8c655b16b9a038d0bd8b3a3e21.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/61ce4568e84513906699d112f2850abce2742c8c655b16b9a038d0bd8b3a3e21.jpg)

![6506e4684fadf2a9ad8bc413f9652a3dfcf866d0af337eedbb68945542834200.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/6506e4684fadf2a9ad8bc413f9652a3dfcf866d0af337eedbb68945542834200.jpg)

![68661bb33eb9e0e7277452340524e6b6119d66bd26e47b8a915f483ed7bb34e0.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/68661bb33eb9e0e7277452340524e6b6119d66bd26e47b8a915f483ed7bb34e0.jpg)

![6e63521df9467adbcbdc32e162150eb5ac6a34f0f1ed6b99e96c38d6a0912d95.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/6e63521df9467adbcbdc32e162150eb5ac6a34f0f1ed6b99e96c38d6a0912d95.jpg)

![70f46b53e90dbf96d2891b2344d5b1041bdbf1447304404115f2ca079c62cb5a.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/70f46b53e90dbf96d2891b2344d5b1041bdbf1447304404115f2ca079c62cb5a.jpg)

![7501ec37a3df89553b69c59b1ef004681a73b26d038da5b9fab6acf32104df57.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/7501ec37a3df89553b69c59b1ef004681a73b26d038da5b9fab6acf32104df57.jpg)

![758f85c542c33fae3818a8b114f086928603a098a0f0b78271cf4badd0fb5ad4.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/758f85c542c33fae3818a8b114f086928603a098a0f0b78271cf4badd0fb5ad4.jpg)

![7c497eb0e01b02637a5cd196921bb15937c2ed2247db729be0b6a7163f7e1ec9.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/7c497eb0e01b02637a5cd196921bb15937c2ed2247db729be0b6a7163f7e1ec9.jpg)

![8ad3d21460028cdaac373dacd22c90b3c3406174226e173778779e0bb933282b.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/8ad3d21460028cdaac373dacd22c90b3c3406174226e173778779e0bb933282b.jpg)

![8b25d099a39ce5bf3b1b62db779fc2333a25c9fe9d24268b05eea7ae6c6f5c84.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/8b25d099a39ce5bf3b1b62db779fc2333a25c9fe9d24268b05eea7ae6c6f5c84.jpg)

![97e749976ebb1720aad65e3946c84adf600a257fe8709d90c1502b4ff4a4977c.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/97e749976ebb1720aad65e3946c84adf600a257fe8709d90c1502b4ff4a4977c.jpg)

![9991ad3c831f9fc430c49c378aa86c3ee453f74726fa5278c287f1e24d826a06.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/9991ad3c831f9fc430c49c378aa86c3ee453f74726fa5278c287f1e24d826a06.jpg)

![9ad94a4c842bfdf857f8df718969197a8534eac3a550825a45a6c98b4adcc96f.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/9ad94a4c842bfdf857f8df718969197a8534eac3a550825a45a6c98b4adcc96f.jpg)

![a2f2be326fe17fdf7b1837f3b08e99e1265154be7014cc8f49932184affbb55b.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/a2f2be326fe17fdf7b1837f3b08e99e1265154be7014cc8f49932184affbb55b.jpg)

![ac6bf81766359a6d6e5dd33428c0298d62dca249b33afab1d2f5de063cbf783e.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/ac6bf81766359a6d6e5dd33428c0298d62dca249b33afab1d2f5de063cbf783e.jpg)

![b21f4cd2e7938d42e12cc0f4aed9ba47fed0866e9234cdc2c7c0becf9fc960fe.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/b21f4cd2e7938d42e12cc0f4aed9ba47fed0866e9234cdc2c7c0becf9fc960fe.jpg)

![b4fbf14d15f5122701cc1c17c2c865195996cfd7206ebaf238b99fffca61b65a.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/b4fbf14d15f5122701cc1c17c2c865195996cfd7206ebaf238b99fffca61b65a.jpg)

![c4868f11d4414ceb02e4d8f2b599a14de1fa1af6f72f7412eebaa4d147c79bb2.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/c4868f11d4414ceb02e4d8f2b599a14de1fa1af6f72f7412eebaa4d147c79bb2.jpg)

![cc41ad980730c084e7dd0f7dab6aaba18bc29157dd392db18a488fe97aa0a7bf.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/cc41ad980730c084e7dd0f7dab6aaba18bc29157dd392db18a488fe97aa0a7bf.jpg)

![d1c550c165f61ec2199f922b4deb9a4010d036b42147b826f07db690f7dd87a7.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/d1c550c165f61ec2199f922b4deb9a4010d036b42147b826f07db690f7dd87a7.jpg)

![d3b2f3fffa927c039eb06e0336fa5058f4988adb059afeb6a0acfdbeb5a12373.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/d3b2f3fffa927c039eb06e0336fa5058f4988adb059afeb6a0acfdbeb5a12373.jpg)

![d9203db8e6170eaba694fde5d3ab872bdcfd2af60423dd4d429ffd7033551769.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/d9203db8e6170eaba694fde5d3ab872bdcfd2af60423dd4d429ffd7033551769.jpg)

![ea67a39beb5867f1d57f56f0546d53ee2d5aba5a41d02ba40c2bf8b5d26c6d04.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/ea67a39beb5867f1d57f56f0546d53ee2d5aba5a41d02ba40c2bf8b5d26c6d04.jpg)

![ed0f439ef823bec486df09f1126773a8bc900de72fabecf0e773146af19b574f.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/ed0f439ef823bec486df09f1126773a8bc900de72fabecf0e773146af19b574f.jpg)

![fb15a5e3fc6b1c4217e7056b2ccf2ac2fe8958dbe0c8731e4097aeff0b233e7c.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/fb15a5e3fc6b1c4217e7056b2ccf2ac2fe8958dbe0c8731e4097aeff0b233e7c.jpg)

![ffdd6739622debec18344cbf7c149770993b4abdba3e72ba90646fb8d307c5ba.jpg](../icml_results/4305_OSF_%20On%20Pre-training%20and%20Scaling%20of%20Sleep%20Foundation%20Models/tables/ffdd6739622debec18344cbf7c149770993b4abdba3e72ba90646fb8d307c5ba.jpg)

<a id="paper-4263"></a>

## The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency


### Images

![2eee90c7ebfcb44327cabefb062894d56b73985371567ef521c2592cabe659b7.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/2eee90c7ebfcb44327cabefb062894d56b73985371567ef521c2592cabe659b7.jpg)

![50e1be3ae58839cc936a4e9ce3a7bf12ca8eee3bf61f72f002598071dfdb0523.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/50e1be3ae58839cc936a4e9ce3a7bf12ca8eee3bf61f72f002598071dfdb0523.jpg)

![72f155d713795f550deb1f70547da1954138aefaeee7b960e23e5ff381a9c328.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/72f155d713795f550deb1f70547da1954138aefaeee7b960e23e5ff381a9c328.jpg)

![7cce7584e8d06610391923f2152963dca43f5d38cb958996b46a3228f2984a8e.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/7cce7584e8d06610391923f2152963dca43f5d38cb958996b46a3228f2984a8e.jpg)

![a24b3bd1b16ab8c78d4d820704def232beb00bc8c7e7ef0e1acd5e5b564f4575.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/a24b3bd1b16ab8c78d4d820704def232beb00bc8c7e7ef0e1acd5e5b564f4575.jpg)

![a44a3a46de986e61dcd2fd4d5b68e2554854c6f64cdfe7147f05c0c40e1182e0.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/a44a3a46de986e61dcd2fd4d5b68e2554854c6f64cdfe7147f05c0c40e1182e0.jpg)

![c6ca1dbb896b89aca76901ee84b3332c5088ba61ce4c88d14a690c5e89ab1539.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/c6ca1dbb896b89aca76901ee84b3332c5088ba61ce4c88d14a690c5e89ab1539.jpg)

![e4d2010f776fc0c46b4456217c0d8c586974fd8319b9e565ac44c4828eed60df.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/e4d2010f776fc0c46b4456217c0d8c586974fd8319b9e565ac44c4828eed60df.jpg)

![f230dffa6b6974d9117bb99827ca58b18a58d4e1a6fb9944615c4c583eedd50b.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/f230dffa6b6974d9117bb99827ca58b18a58d4e1a6fb9944615c4c583eedd50b.jpg)

![ff5d453fcf28111e1b89114561c73ad38b9bbc28b25e3af94de62fb5213fc4aa.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/images/ff5d453fcf28111e1b89114561c73ad38b9bbc28b25e3af94de62fb5213fc4aa.jpg)

### Tables

![4a68e5ecb85252befe9c60aac449bbb7a0a535db8e745031584d8c993217f2bd.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/4a68e5ecb85252befe9c60aac449bbb7a0a535db8e745031584d8c993217f2bd.jpg)

![91568a45ec2174752a61da76090ea8fa9d8ded30b6186425eebdd67a4837a836.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/91568a45ec2174752a61da76090ea8fa9d8ded30b6186425eebdd67a4837a836.jpg)

![9626b53c702960879269adf59cec313c675a7a87348001104c472148a8831192.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/9626b53c702960879269adf59cec313c675a7a87348001104c472148a8831192.jpg)

![a8904dbb65b22e51ce1b79616c4b2c6cd23c87e337c15c8e8fab4aab46ecf2e3.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/a8904dbb65b22e51ce1b79616c4b2c6cd23c87e337c15c8e8fab4aab46ecf2e3.jpg)

![ad0620463d90eaff446d3b802881ff6b3a5ab7d46153ed056218f940e6960535.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/ad0620463d90eaff446d3b802881ff6b3a5ab7d46153ed056218f940e6960535.jpg)

![cffb1a1c62d4033785fc43d8e830d123352c6f16fb07eb53ff4c9ba41cc7d9a8.jpg](../icml_results/4306_The%20Hippocampal%20Place%20Field%20Gradient_%20A%20Bio-inspired%20Framework%20Building%20Multiscale%20Representation%20fo/tables/cffb1a1c62d4033785fc43d8e830d123352c6f16fb07eb53ff4c9ba41cc7d9a8.jpg)

<a id="paper-4264"></a>

## General Analysis of LMO-based Optimizers: Beyond Bounded Variance


### Images

![04ac77c409eb867d3a94aa7a32459d84160ade17b810e441fa150f4d8f9873b0.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/04ac77c409eb867d3a94aa7a32459d84160ade17b810e441fa150f4d8f9873b0.jpg)

![1aac5feed9f173850f1801d6af4d3b3881dd21466508ddf74015c1e465c57acc.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/1aac5feed9f173850f1801d6af4d3b3881dd21466508ddf74015c1e465c57acc.jpg)

![803143b77a42b8a7d8b8070902fb32bd0346c8e956873774b3a4fcf3a207813d.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/803143b77a42b8a7d8b8070902fb32bd0346c8e956873774b3a4fcf3a207813d.jpg)

![84e7d8ed293fff7d76cc97e76bd0df8dc90a82f8cff5cfd2f10e1582adbe87f1.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/84e7d8ed293fff7d76cc97e76bd0df8dc90a82f8cff5cfd2f10e1582adbe87f1.jpg)

![ae31b6fa4fb7f554edbe8dfe78939d821250589ad933942b6efe682991b470fa.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/ae31b6fa4fb7f554edbe8dfe78939d821250589ad933942b6efe682991b470fa.jpg)

![d0fd6ed35ecb470df53446e8bf0d1997ab90897d688a96bcba4ebaa50f62c2dd.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/d0fd6ed35ecb470df53446e8bf0d1997ab90897d688a96bcba4ebaa50f62c2dd.jpg)

![ecd1d08b43725c5904b77f010e69afaff943953996d5ba5b10413f2cb820b11b.jpg](../icml_results/4307_General%20Analysis%20of%20LMO-based%20Optimizers_%20Beyond%20Bounded%20Variance/images/ecd1d08b43725c5904b77f010e69afaff943953996d5ba5b10413f2cb820b11b.jpg)

<a id="paper-4265"></a>

## Understanding MARS: When Scaling Momentum Correction Provably Helps


### Images

![23fcebdedce6b72e9d084d4c2ef838ef75664d1c646be59c51d1db94f01e9115.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/23fcebdedce6b72e9d084d4c2ef838ef75664d1c646be59c51d1db94f01e9115.jpg)

![319a15015bc3f3b2192aa41ff29a0c30dfd4338030fe72bac56dc8e100c08a41.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/319a15015bc3f3b2192aa41ff29a0c30dfd4338030fe72bac56dc8e100c08a41.jpg)

![44d08bb6dd2e64ecc13fbd29de2ae3970f34367c9d02c355c665da3c2851b3dc.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/44d08bb6dd2e64ecc13fbd29de2ae3970f34367c9d02c355c665da3c2851b3dc.jpg)

![576eaf7f2b56031b91ba95563339a1c770dfc3b4a7e6be8f25fec24a48a55b42.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/576eaf7f2b56031b91ba95563339a1c770dfc3b4a7e6be8f25fec24a48a55b42.jpg)

![a0a8eed49156db3e6fc2e9493d63787228c3abeeefb531ca10e73151170c3065.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/a0a8eed49156db3e6fc2e9493d63787228c3abeeefb531ca10e73151170c3065.jpg)

![a8d4181f3c408f6b9aaf86721ebece5edf039e56f19b8fa030b308e93bad8964.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/images/a8d4181f3c408f6b9aaf86721ebece5edf039e56f19b8fa030b308e93bad8964.jpg)

### Tables

![0e654b57e051a84014ca46e0bd30a4a27282076f97bc81f6eb67ad24c1b0eb94.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/tables/0e654b57e051a84014ca46e0bd30a4a27282076f97bc81f6eb67ad24c1b0eb94.jpg)

![a99980649dc11a8a6843575869c4e497b7bbd1310f1456a63196092a5c5547a7.jpg](../icml_results/4308_Understanding%20MARS_%20When%20Scaling%20Momentum%20Correction%20Provably%20Helps/tables/a99980649dc11a8a6843575869c4e497b7bbd1310f1456a63196092a5c5547a7.jpg)

<a id="paper-4266"></a>

## Two-dimensional quantization for geometry-aware audio coding


### Images

![08a693d9deff28ac5f94be7054bdcf7630004b114a81dc8984c39f0795e41596.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/images/08a693d9deff28ac5f94be7054bdcf7630004b114a81dc8984c39f0795e41596.jpg)

![25bb322a7bf90daec1ff5df694dc620e472110a98485407b9c37ec4a163ea859.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/images/25bb322a7bf90daec1ff5df694dc620e472110a98485407b9c37ec4a163ea859.jpg)

![caf335fef32c0d9df5c7eedb9b1d98f7bd2f775a96ca0a2c521ad34f9301cb14.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/images/caf335fef32c0d9df5c7eedb9b1d98f7bd2f775a96ca0a2c521ad34f9301cb14.jpg)

### Tables

![0868c9a23778828c303eea867d34e67cdb3480ba7310ce265b15136db0d5d2e9.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/0868c9a23778828c303eea867d34e67cdb3480ba7310ce265b15136db0d5d2e9.jpg)

![2d14f377694a369c9f3ab8307c5047adf16b0aefcdfb0277ae9f3202b2f02bdb.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/2d14f377694a369c9f3ab8307c5047adf16b0aefcdfb0277ae9f3202b2f02bdb.jpg)

![3015539101cfa268572c2d9bf3878e435a679655eadf4d827a9d13f5eede9457.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/3015539101cfa268572c2d9bf3878e435a679655eadf4d827a9d13f5eede9457.jpg)

![3d8e71ca640d15830cd81f428391e03095f750336822fcef7bb7c00e3ea5a673.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/3d8e71ca640d15830cd81f428391e03095f750336822fcef7bb7c00e3ea5a673.jpg)

![410ee353c50c7bc03d17aa916d7f940182cf6893fe0e6a41a219e628e9e8a397.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/410ee353c50c7bc03d17aa916d7f940182cf6893fe0e6a41a219e628e9e8a397.jpg)

![83baaa5e641ccac947a7f3b2164624c6357dd91a64b3c635ab985804978adb4f.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/83baaa5e641ccac947a7f3b2164624c6357dd91a64b3c635ab985804978adb4f.jpg)

![87cb9f9fb03f86b512a5880e1a61498115f753062c29754fedbeebc3ed38a607.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/87cb9f9fb03f86b512a5880e1a61498115f753062c29754fedbeebc3ed38a607.jpg)

![a6980327b0f92cfc1dab950b1ddd04a7cc4a3caee06df04376b3cf75525c5098.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/a6980327b0f92cfc1dab950b1ddd04a7cc4a3caee06df04376b3cf75525c5098.jpg)

![a9822313e47bd4586440444c1030d9fb303ef79ebb9716aecdabdf72a50f811e.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/a9822313e47bd4586440444c1030d9fb303ef79ebb9716aecdabdf72a50f811e.jpg)

![b467fdb8d81970d84a432d2d3fdaa6caad57374dcafbc73d382e7236bf03eae0.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/b467fdb8d81970d84a432d2d3fdaa6caad57374dcafbc73d382e7236bf03eae0.jpg)

![cbc82e6fea4490589cf8471145ae06fe6535f716806348152559f39cc8d9f973.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/cbc82e6fea4490589cf8471145ae06fe6535f716806348152559f39cc8d9f973.jpg)

![cd0d94894c4e6d91da01ac63835353b64a866ccc720068b3a108b5f2e63850a1.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/cd0d94894c4e6d91da01ac63835353b64a866ccc720068b3a108b5f2e63850a1.jpg)

![d16b2016944fd64a43b6d1343074e5e2c6fd07fabc5b9b32dd6a6dcd7603408f.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/d16b2016944fd64a43b6d1343074e5e2c6fd07fabc5b9b32dd6a6dcd7603408f.jpg)

![eb325fb38f15da8a1325aa46ed0cdedde40defe944e6c4756addba623cf4d52b.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/eb325fb38f15da8a1325aa46ed0cdedde40defe944e6c4756addba623cf4d52b.jpg)

![f3283ec11e25566084cf4c37cb3863e2fc6e9d9375c98d87f3f647cc084b71f8.jpg](../icml_results/4309_Two-dimensional%20quantization%20for%20geometry-aware%20audio%20coding/tables/f3283ec11e25566084cf4c37cb3863e2fc6e9d9375c98d87f3f647cc084b71f8.jpg)

<a id="paper-4267"></a>

## FLAG: Foundation model representation with Latent diffusion Alignment via Graph for spatial gene expression prediction


### Images

![0237322cbfea935684e5a9e1705465d71d0f26c894d4e68f4c9d79c19db9072e.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/0237322cbfea935684e5a9e1705465d71d0f26c894d4e68f4c9d79c19db9072e.jpg)

![0854386c23d33ce54e951407a30ea6627f69577a718531bb68f21af4363e68c2.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/0854386c23d33ce54e951407a30ea6627f69577a718531bb68f21af4363e68c2.jpg)

![17cbe30bb23b3899f5f8895d3d57d7237f32fb7c775241e5787d74bdc896374d.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/17cbe30bb23b3899f5f8895d3d57d7237f32fb7c775241e5787d74bdc896374d.jpg)

![2007f93ea8383b8bee04373867a13f96b81de7d3e3728579728b8377da346a31.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/2007f93ea8383b8bee04373867a13f96b81de7d3e3728579728b8377da346a31.jpg)

![21d783af11654e2279a2d94337c76c53bb600784849be1612496301018e07a1c.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/21d783af11654e2279a2d94337c76c53bb600784849be1612496301018e07a1c.jpg)

![3740ce7a21c6c468e52eaf747f416cebe49715d0daa6b6a8d852bc848d36042e.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/3740ce7a21c6c468e52eaf747f416cebe49715d0daa6b6a8d852bc848d36042e.jpg)

![4aa5cfac984404e9cddb74eb9b1ee3d86993478dda4551ec87b3c440e7ae6410.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/4aa5cfac984404e9cddb74eb9b1ee3d86993478dda4551ec87b3c440e7ae6410.jpg)

![561bf8dd41a4fb2dd091c3465d92a5af224cfc783071ba595b8b228527a27404.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/561bf8dd41a4fb2dd091c3465d92a5af224cfc783071ba595b8b228527a27404.jpg)

![7e47bb341a486b8297c1775dcf71eb4cdf2d65969688185c1affe38174fbbba4.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/7e47bb341a486b8297c1775dcf71eb4cdf2d65969688185c1affe38174fbbba4.jpg)

![9ed8cd4c7006451b1319a31ea0b00d0040e4eaacdfde6d3f95126a4db46fb3ce.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/9ed8cd4c7006451b1319a31ea0b00d0040e4eaacdfde6d3f95126a4db46fb3ce.jpg)

![be28a605d14fcc26d3bcde8338fe910ca5cf137acc40292fee039b4d59598089.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/be28a605d14fcc26d3bcde8338fe910ca5cf137acc40292fee039b4d59598089.jpg)

![d2f88c75ecb7e8e008ac23f7fd1cada9710e929a8d139baf38fb8d43180c0acb.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/d2f88c75ecb7e8e008ac23f7fd1cada9710e929a8d139baf38fb8d43180c0acb.jpg)

![e4207cad23275758f9a784056e998a71a5de0cf7037ad6683a08f31d4efad66c.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/e4207cad23275758f9a784056e998a71a5de0cf7037ad6683a08f31d4efad66c.jpg)

![fd91dd87e6656c7152e7ae39a814335f446dbba4f77a7098beba0e8b1241b1bf.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/images/fd91dd87e6656c7152e7ae39a814335f446dbba4f77a7098beba0e8b1241b1bf.jpg)

### Tables

![098d9d3e55c5a995eb5c93cf86bf2afa3d8c3c482a8a4730bac5205c8f2e5c9c.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/098d9d3e55c5a995eb5c93cf86bf2afa3d8c3c482a8a4730bac5205c8f2e5c9c.jpg)

![286668824c5d66bb9aecd1e93c4c539b44a3b3679f37595eeb6aa6e6990bfcc3.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/286668824c5d66bb9aecd1e93c4c539b44a3b3679f37595eeb6aa6e6990bfcc3.jpg)

![4a137506b50573e9229c3286c35e68d8da53296d26e01eacabac28aec4859c18.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/4a137506b50573e9229c3286c35e68d8da53296d26e01eacabac28aec4859c18.jpg)

![50555c8ca241ea7afb0ee7106e0da40c74cf0b7c5a279359615cfeebf43e9381.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/50555c8ca241ea7afb0ee7106e0da40c74cf0b7c5a279359615cfeebf43e9381.jpg)

![7a9a713cbbea756c1f6dce1dd97d06b1af2b01313e15e2ade287d912bda84568.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/7a9a713cbbea756c1f6dce1dd97d06b1af2b01313e15e2ade287d912bda84568.jpg)

![80a10bd2d301e5a96ed06fd33d7b499ed447536fd4dfcad31d7122d23b11d952.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/80a10bd2d301e5a96ed06fd33d7b499ed447536fd4dfcad31d7122d23b11d952.jpg)

![834dc18fd64489dbadeba3b393114ab3890f470058b74801d955bbf6389cda69.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/834dc18fd64489dbadeba3b393114ab3890f470058b74801d955bbf6389cda69.jpg)

![8f4580401b2ffaa7aa3474bf545c0c3236fe3b1b46ce91f988e7da10c5d8df5c.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/8f4580401b2ffaa7aa3474bf545c0c3236fe3b1b46ce91f988e7da10c5d8df5c.jpg)

![b43bc29d49a1385b959ce8a6b1b5877630ca4e41e705fdd0a1a49aadee9f415f.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/b43bc29d49a1385b959ce8a6b1b5877630ca4e41e705fdd0a1a49aadee9f415f.jpg)

![ce9cd2e9c11f41197ac1e54d69e4fdb3f8c7327566a673989b368c39dec95b78.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/ce9cd2e9c11f41197ac1e54d69e4fdb3f8c7327566a673989b368c39dec95b78.jpg)

![d378fcd333a56d2b4d999af6edae881f5250916674ee5b494dbddd6192cb8c25.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/d378fcd333a56d2b4d999af6edae881f5250916674ee5b494dbddd6192cb8c25.jpg)

![dc0ffb0daa1d5e1b97627117e1526cad6aa45948c5edb42ebed9617103d038c0.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/dc0ffb0daa1d5e1b97627117e1526cad6aa45948c5edb42ebed9617103d038c0.jpg)

![f0eda007dff4543a5a6094a7a778f791cf81254865838e711f92c59dad75eee4.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/f0eda007dff4543a5a6094a7a778f791cf81254865838e711f92c59dad75eee4.jpg)

![f938998e263c4d4822078f702ef3b89eaf0492aab10e3ae6818eb70b9c555d05.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/f938998e263c4d4822078f702ef3b89eaf0492aab10e3ae6818eb70b9c555d05.jpg)

![fb85a61bb173a7f24d438c8b65255322aae71af53c4c39fdd3c247168a02c433.jpg](../icml_results/4310_FLAG_%20Foundation%20model%20representation%20with%20Latent%20diffusion%20Alignment%20via%20Graph%20for%20spatial%20gene%20exp/tables/fb85a61bb173a7f24d438c8b65255322aae71af53c4c39fdd3c247168a02c433.jpg)

<a id="paper-4268"></a>

## Towards Execution-Grounded Automated AI Research


### Images

![322e0f7bb997d297ac0aa63763d73287b5eceb9a92351259e91e558fd1d176df.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/322e0f7bb997d297ac0aa63763d73287b5eceb9a92351259e91e558fd1d176df.jpg)

![3682aae143ce81b8d6818271de1d753b0f0a6b42883dfd335203393fc0a1bbaf.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/3682aae143ce81b8d6818271de1d753b0f0a6b42883dfd335203393fc0a1bbaf.jpg)

![7a5d72231d4c7ebbf2f3ec7d3e30f06f2cc97a6e10edaf202b5517fca2536cea.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/7a5d72231d4c7ebbf2f3ec7d3e30f06f2cc97a6e10edaf202b5517fca2536cea.jpg)

![918b5ef9ca81726cb3a585fca545a374c026705887c50091a7a2e4168e75bd4d.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/918b5ef9ca81726cb3a585fca545a374c026705887c50091a7a2e4168e75bd4d.jpg)

![92cb37c8ba0b013ca269ba9b17fde49a56e36d6fb4354d4332d364cb432befc6.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/92cb37c8ba0b013ca269ba9b17fde49a56e36d6fb4354d4332d364cb432befc6.jpg)

![99a799e8974ddef325b4707f8ff88e0007b746e93480805b0c4b9ee0ec28e1b8.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/99a799e8974ddef325b4707f8ff88e0007b746e93480805b0c4b9ee0ec28e1b8.jpg)

![a9a384bad89604e65f3d5ae064e8eefc5c9ba26931cbacf115675556043123da.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/a9a384bad89604e65f3d5ae064e8eefc5c9ba26931cbacf115675556043123da.jpg)

![b62ff2a0e727e2a302958e7e8683d501d609383fc63fc5f255610d8e01287fd4.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/b62ff2a0e727e2a302958e7e8683d501d609383fc63fc5f255610d8e01287fd4.jpg)

![b6f104b9bf3a66b367481313ca2609a86f8ce0d2e64daa7b3c21f3a9f8d487a3.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/b6f104b9bf3a66b367481313ca2609a86f8ce0d2e64daa7b3c21f3a9f8d487a3.jpg)

![c59732cf9ed7c0a3d5b6983d3393aa6bf87bdc3fce547028c09a80dfc72b92f5.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/c59732cf9ed7c0a3d5b6983d3393aa6bf87bdc3fce547028c09a80dfc72b92f5.jpg)

![c67a0c4878d950acc7c30225830c23f0f82032d3a540ff2c9ffc911cd6834010.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/c67a0c4878d950acc7c30225830c23f0f82032d3a540ff2c9ffc911cd6834010.jpg)

![d8e5a46f1810440d11a2097ec24f591de6017e51c208b332044e2d24cfe516dc.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/images/d8e5a46f1810440d11a2097ec24f591de6017e51c208b332044e2d24cfe516dc.jpg)

### Tables

![43763694c3fe97ef821e239e5429b6624c63fa80c338aa7533928d2b9c1509b6.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/43763694c3fe97ef821e239e5429b6624c63fa80c338aa7533928d2b9c1509b6.jpg)

![498358abd6e18ab0e897e7140a3c4162b333bfbd108b4b3d82a083979386cde5.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/498358abd6e18ab0e897e7140a3c4162b333bfbd108b4b3d82a083979386cde5.jpg)

![49f50f3419707ae09ff1cffbd5c004b313ca03ce0ef54644953594700f92d6a3.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/49f50f3419707ae09ff1cffbd5c004b313ca03ce0ef54644953594700f92d6a3.jpg)

![6051239feb759fa8d2f8a7c6240eaf54499d37273291c8528fb70e619e97abea.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/6051239feb759fa8d2f8a7c6240eaf54499d37273291c8528fb70e619e97abea.jpg)

![8e63c6071921859b72921ec04579c022b3731d2fb670e99f8cda1b882a35d0e7.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/8e63c6071921859b72921ec04579c022b3731d2fb670e99f8cda1b882a35d0e7.jpg)

![9b64d5fec9ed9d4a5bf1e94e4ba616d8fc79625e351870d29273f59de9d2a6d2.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/9b64d5fec9ed9d4a5bf1e94e4ba616d8fc79625e351870d29273f59de9d2a6d2.jpg)

![9c939562ede6b8921cfe6f6f4bfa3e23dc04fd885c3ba66fdc26af4b9a8b2303.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/9c939562ede6b8921cfe6f6f4bfa3e23dc04fd885c3ba66fdc26af4b9a8b2303.jpg)

![a29b076a09bd584897e2b78dc394e94f1fa349aaae3ea8f424c9d88753c16aed.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/a29b076a09bd584897e2b78dc394e94f1fa349aaae3ea8f424c9d88753c16aed.jpg)

![a9daf9034688a269d8aa0d807d3d146e1c4dc34990cbacde68601bd4d50821c7.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/a9daf9034688a269d8aa0d807d3d146e1c4dc34990cbacde68601bd4d50821c7.jpg)

![aa845e103849d0066df793d62aae2e679ddf1505a93b2b3bd13b1883000e910e.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/aa845e103849d0066df793d62aae2e679ddf1505a93b2b3bd13b1883000e910e.jpg)

![d151f1e56be49f1c96ba7ff2e9941c7551c1513c45a6c277310ecaff9bbf7056.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/d151f1e56be49f1c96ba7ff2e9941c7551c1513c45a6c277310ecaff9bbf7056.jpg)

![e53a9875fb67d5882fab930d688781b83e53a6321ba3c6f4e54ae9c3f0b71328.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/e53a9875fb67d5882fab930d688781b83e53a6321ba3c6f4e54ae9c3f0b71328.jpg)

![ea563f75f3edbb442aab0289f4fc249dc54f0f4b52f04aeff9cab3f4b9d9ce63.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/ea563f75f3edbb442aab0289f4fc249dc54f0f4b52f04aeff9cab3f4b9d9ce63.jpg)

![edfb21c8a72ceedee28c1ca8a76cfdef5d17d9f40cd0b336d4433cd2b53a19c9.jpg](../icml_results/4311_Towards%20Execution-Grounded%20Automated%20AI%20Research/tables/edfb21c8a72ceedee28c1ca8a76cfdef5d17d9f40cd0b336d4433cd2b53a19c9.jpg)

<a id="paper-4269"></a>

## Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement


### Images

![16b810d7851184d77b946759b6c95e4ed834e8c02e92e029a5b804da7a24bfd7.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/16b810d7851184d77b946759b6c95e4ed834e8c02e92e029a5b804da7a24bfd7.jpg)

![53b5b11ebed9647a8df68e8272886fb7291350ef06976e43913f17a3c0f3453b.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/53b5b11ebed9647a8df68e8272886fb7291350ef06976e43913f17a3c0f3453b.jpg)

![5fb1895b48f34dda9eee7af9169b02f5d2516dff5eeef81579710771b2107cef.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/5fb1895b48f34dda9eee7af9169b02f5d2516dff5eeef81579710771b2107cef.jpg)

![74e22a852266d5212c2c54cc68d599fd5b5ebaafb969e15a0c3eb470eae0b9e3.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/74e22a852266d5212c2c54cc68d599fd5b5ebaafb969e15a0c3eb470eae0b9e3.jpg)

![8abce2fbd2040d63c2045954833d2c67f6424ea9a772c85d25c8d1112f659920.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/8abce2fbd2040d63c2045954833d2c67f6424ea9a772c85d25c8d1112f659920.jpg)

![8cee4bf081afa41537186977be1637eeb763c81274b4bd82162933614723da8c.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/images/8cee4bf081afa41537186977be1637eeb763c81274b4bd82162933614723da8c.jpg)

### Tables

![75b38865e06d29b704b20ffaadc1cd3efb46e43d1e95a396a08e86f8fef71b83.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/75b38865e06d29b704b20ffaadc1cd3efb46e43d1e95a396a08e86f8fef71b83.jpg)

![90afdbb61c97816cfa4bbfd71a8760d595d93d48c5fa87b321b789e4a7433c07.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/90afdbb61c97816cfa4bbfd71a8760d595d93d48c5fa87b321b789e4a7433c07.jpg)

![a55f6eb0f1fd8b721bb6463cbe7ebb3a37db8c25c572bc3a427f4e1dd25cc52b.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/a55f6eb0f1fd8b721bb6463cbe7ebb3a37db8c25c572bc3a427f4e1dd25cc52b.jpg)

![a8e702ccaac540493a1cf083f7541f85f5d19e4a65a8a4a85972f8e6c9d12894.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/a8e702ccaac540493a1cf083f7541f85f5d19e4a65a8a4a85972f8e6c9d12894.jpg)

![be7892bc82ae1af7814e58bf6915d929a21dbf236dcfab89604a5a25ffa424a5.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/be7892bc82ae1af7814e58bf6915d929a21dbf236dcfab89604a5a25ffa424a5.jpg)

![c0b32256a8822f754b6aace8516187df3e574704220ad4a2fda7694e68bdb9aa.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/c0b32256a8822f754b6aace8516187df3e574704220ad4a2fda7694e68bdb9aa.jpg)

![c3b2c9c40d435d529463eba16e186a2223e1622e57bb67da02d8f6c43651557a.jpg](../icml_results/4312_Mobility-Embedded%20POIs_%20Learning%20What%20A%20Place%20Is%20and%20How%20It%20Is%20Used%20from%20Human%20Movement/tables/c3b2c9c40d435d529463eba16e186a2223e1622e57bb67da02d8f6c43651557a.jpg)

<a id="paper-4270"></a>

## Geometry-Aware Probabilistic Circuits via Voronoi Tessellations


### Images

![1d9986617545d991e2466cc6b882ff60bfff234227d8f504340d272d0ba95e61.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/images/1d9986617545d991e2466cc6b882ff60bfff234227d8f504340d272d0ba95e61.jpg)

![475cbf1e06da0a9fae413f9f20a59e50c8328a10afa6b40e11c589addea347b7.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/images/475cbf1e06da0a9fae413f9f20a59e50c8328a10afa6b40e11c589addea347b7.jpg)

![9fec0d4c619d188fb93face6662623f0cf9b61f6003981d285390eeb5c1cd8e1.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/images/9fec0d4c619d188fb93face6662623f0cf9b61f6003981d285390eeb5c1cd8e1.jpg)

![f420d5cae6bf7a1458720f48c14f74e8cd42d306bda49433e656f34d36509afc.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/images/f420d5cae6bf7a1458720f48c14f74e8cd42d306bda49433e656f34d36509afc.jpg)

![f77c940196f55be183aea2a2403a70fb94834e2594e26d10cacad97c92fb6393.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/images/f77c940196f55be183aea2a2403a70fb94834e2594e26d10cacad97c92fb6393.jpg)

### Tables

![8b1fcd7897ef4b5d5bc49198ba529af63599a8918c4c13565f17a4fa3129feb1.jpg](../icml_results/4313_Geometry-Aware%20Probabilistic%20Circuits%20via%20Voronoi%20Tessellations/tables/8b1fcd7897ef4b5d5bc49198ba529af63599a8918c4c13565f17a4fa3129feb1.jpg)

<a id="paper-4271"></a>

## Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis


### Images

![16302d6450430265de3108cc1b5e83f7270e1b42116ca47072ee8c341a38c769.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/16302d6450430265de3108cc1b5e83f7270e1b42116ca47072ee8c341a38c769.jpg)

![284f17a05fa6a5befab9879fb0ee65dc6b5b16b7ea89da4261ba16fd4f0b8840.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/284f17a05fa6a5befab9879fb0ee65dc6b5b16b7ea89da4261ba16fd4f0b8840.jpg)

![43bcfcd4c8f1be3fdf37cfa2e514b9e51359f952476e66e8530cb73f6038fda1.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/43bcfcd4c8f1be3fdf37cfa2e514b9e51359f952476e66e8530cb73f6038fda1.jpg)

![54181e83ff00e5da55d7c1d47620c1abd1c65f31f830fdd9653332e1a756a272.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/54181e83ff00e5da55d7c1d47620c1abd1c65f31f830fdd9653332e1a756a272.jpg)

![87bf640756eea0feee3fd54360b480d946173f826559296ad985b1072d6ae32a.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/87bf640756eea0feee3fd54360b480d946173f826559296ad985b1072d6ae32a.jpg)

![a5a4a8fa7cad0996ef898ac8be392a50f68cdb27331b49afc58e600e9b63f7cb.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/a5a4a8fa7cad0996ef898ac8be392a50f68cdb27331b49afc58e600e9b63f7cb.jpg)

![b769cbad91c2763dde476767f333a6ab1da4ab3dc15a3fde3b0b224019e02687.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/b769cbad91c2763dde476767f333a6ab1da4ab3dc15a3fde3b0b224019e02687.jpg)

![c72eaccc897c44dc7bf2758ac5aa293a4ed032bd799802d5954d539329c92b27.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/c72eaccc897c44dc7bf2758ac5aa293a4ed032bd799802d5954d539329c92b27.jpg)

![cc47ebcda6f34f08b9862f5aa809224c65a415d4607dde237b4d1b13f60e159e.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/cc47ebcda6f34f08b9862f5aa809224c65a415d4607dde237b4d1b13f60e159e.jpg)

![cec0688b0c5bdb5d0f99eec56a0323976f701da5aba241798142cc8fcd19e4da.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/cec0688b0c5bdb5d0f99eec56a0323976f701da5aba241798142cc8fcd19e4da.jpg)

![f25df69bd99b6273a853b3808ec8352d5af082d36fc14d86b4bedbff2b158e0f.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/f25df69bd99b6273a853b3808ec8352d5af082d36fc14d86b4bedbff2b158e0f.jpg)

![f9a74e03e2826464889d553bc5fcf47b66798dfcb7f5a6524f5875713e4aef6d.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/images/f9a74e03e2826464889d553bc5fcf47b66798dfcb7f5a6524f5875713e4aef6d.jpg)

### Tables

![0316672b84f4c05a6f26d0f5cb48c383b7b194f548d01bdeacb6adb53eda3f30.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/0316672b84f4c05a6f26d0f5cb48c383b7b194f548d01bdeacb6adb53eda3f30.jpg)

![3bd9536db874921643cf0729832e71d187f6b13ec0857ac23af73c95fed96678.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/3bd9536db874921643cf0729832e71d187f6b13ec0857ac23af73c95fed96678.jpg)

![4e1723340687827e889281788ffdaf60eb94c17cd71ec7e524a7e442ee663de7.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/4e1723340687827e889281788ffdaf60eb94c17cd71ec7e524a7e442ee663de7.jpg)

![515150d73ad890c0ae0db4096aa3169fd165c819a8cdaa8d779c15d3232519b6.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/515150d73ad890c0ae0db4096aa3169fd165c819a8cdaa8d779c15d3232519b6.jpg)

![6a1acaca89bcfe219a2fc98f804ac145a9e8fa862b768d6ad26318c587f95df9.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/6a1acaca89bcfe219a2fc98f804ac145a9e8fa862b768d6ad26318c587f95df9.jpg)

![8f965cd529069bb20ada18eedb40e09ecf636ddfb3057f081c0064bf9110a313.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/8f965cd529069bb20ada18eedb40e09ecf636ddfb3057f081c0064bf9110a313.jpg)

![9b1b7f8b4248eeb72a96f742ac184d8c7aa847df6485de7f2d0cf20bbbad6b8a.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/9b1b7f8b4248eeb72a96f742ac184d8c7aa847df6485de7f2d0cf20bbbad6b8a.jpg)

![a5a1350a16456e0b61d8754576b08504da48ec3c15d9bfe90529e2b2bd680a2b.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/a5a1350a16456e0b61d8754576b08504da48ec3c15d9bfe90529e2b2bd680a2b.jpg)

![b2f71793fe6789431c5b0be118b5bc87d4a46c3e981f38fbc2346cb6240a206c.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/b2f71793fe6789431c5b0be118b5bc87d4a46c3e981f38fbc2346cb6240a206c.jpg)

![c2f16e7bbc0c3efbe7c174a63f6a388bf54394d69e2f70a946ea0d63b7dc0955.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/c2f16e7bbc0c3efbe7c174a63f6a388bf54394d69e2f70a946ea0d63b7dc0955.jpg)

![d9db2cdc890b2cad0215dac8bd8a04b79bbf5a1348db2b221559215388ffb6b3.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/d9db2cdc890b2cad0215dac8bd8a04b79bbf5a1348db2b221559215388ffb6b3.jpg)

![dc085f088ac8f39e391a386d0d64b6942e1f1a8bcb0cfae679799c8d425f884d.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/dc085f088ac8f39e391a386d0d64b6942e1f1a8bcb0cfae679799c8d425f884d.jpg)

![eb060ea3e77a574062e9037e23b904a42da75190375d65d9f7a995bcfaed8d23.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/eb060ea3e77a574062e9037e23b904a42da75190375d65d9f7a995bcfaed8d23.jpg)

![f60756bebbf9ec5a2ff87539f2418af725d88d101a41f6d1f43fe0d6e8f1f69d.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/f60756bebbf9ec5a2ff87539f2418af725d88d101a41f6d1f43fe0d6e8f1f69d.jpg)

![fa1e7ee6a305e8bd16e0085106ecf269b0f97fa1450401a6f8fb0313449ba70a.jpg](../icml_results/4314_Learning%20Multi-Scale%20Hypergraph%20for%20High-Order%20Brain%20Connectivity%20Analysis/tables/fa1e7ee6a305e8bd16e0085106ecf269b0f97fa1450401a6f8fb0313449ba70a.jpg)

<a id="paper-4272"></a>

## Old Habits Die Hard: How Conversational History Geometrically Traps LLMs


### Images

![1d4dd3e7f133f9461fa0f3a76db111d36404158acad15fd9fad0dccb18d99d01.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/1d4dd3e7f133f9461fa0f3a76db111d36404158acad15fd9fad0dccb18d99d01.jpg)

![49bdb9d61dd9a27c808525f0170ada63a9cadfd49c5fbfef85b695194914c632.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/49bdb9d61dd9a27c808525f0170ada63a9cadfd49c5fbfef85b695194914c632.jpg)

![52d517e803f5e4807aee06c1c15c218bc61d1a1c8da3e0c9c3b98fe3608abc36.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/52d517e803f5e4807aee06c1c15c218bc61d1a1c8da3e0c9c3b98fe3608abc36.jpg)

![5f4fc2218c8d1655ca1ee1a08bd66e6a5259f2849cdd1826891752fe8266715e.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/5f4fc2218c8d1655ca1ee1a08bd66e6a5259f2849cdd1826891752fe8266715e.jpg)

![763f3312d4858d9e0e2d9e2754f5ea4497301cd4ed04b744ee39d9dd7e15de72.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/763f3312d4858d9e0e2d9e2754f5ea4497301cd4ed04b744ee39d9dd7e15de72.jpg)

![833011c91afc2dde1c99dafd3774a9a28aa9810eb1deff6537ee3638e88c35b4.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/833011c91afc2dde1c99dafd3774a9a28aa9810eb1deff6537ee3638e88c35b4.jpg)

![89303545456aea11223dce356fc7a9befd809a277c3576580ffb70fc381f286e.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/89303545456aea11223dce356fc7a9befd809a277c3576580ffb70fc381f286e.jpg)

![d42accb5f08dafc22683683812f406f8cebca97cdfc6b8f0b4c0d4bf9e86e66e.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/d42accb5f08dafc22683683812f406f8cebca97cdfc6b8f0b4c0d4bf9e86e66e.jpg)

![d7bf0851881019ab911c78491570c7aeaebb8153b2e888bee76717000064823b.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/d7bf0851881019ab911c78491570c7aeaebb8153b2e888bee76717000064823b.jpg)

![ea6a0b83bf165d4f2c3885f756e1a45dd3672775df0db6c016f6056c790cdea2.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/ea6a0b83bf165d4f2c3885f756e1a45dd3672775df0db6c016f6056c790cdea2.jpg)

![efc9bf080a6aca123efaee8949a730258b12f450b435533354588f37ac7fa3b5.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/efc9bf080a6aca123efaee8949a730258b12f450b435533354588f37ac7fa3b5.jpg)

![fc5bfa7c5162b32b0cb94cb23c9a6b6e428ecc95506711e0308121c1e6484f74.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/images/fc5bfa7c5162b32b0cb94cb23c9a6b6e428ecc95506711e0308121c1e6484f74.jpg)

### Tables

![079b50886f340dfa8fa58b8c8352f30d5113371b8e01591fa31f04764ca218c1.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/079b50886f340dfa8fa58b8c8352f30d5113371b8e01591fa31f04764ca218c1.jpg)

![3c269cb2a3513abf89b5081dc1bd4784a1aca1726e4e9e93cd03fa933421834d.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/3c269cb2a3513abf89b5081dc1bd4784a1aca1726e4e9e93cd03fa933421834d.jpg)

![3d1beaef5e46e30d7d9fd82f9ad77a0229841f120c1897f52dc7f4ae418abfcf.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/3d1beaef5e46e30d7d9fd82f9ad77a0229841f120c1897f52dc7f4ae418abfcf.jpg)

![3fd87f0e7757861aef38d44b642b091ab027ffabba770d30440bbffc6fcb5658.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/3fd87f0e7757861aef38d44b642b091ab027ffabba770d30440bbffc6fcb5658.jpg)

![677514d34f2fc6f61f44edb2e3b74b2780bc975f06b828203251437403b8bc4e.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/677514d34f2fc6f61f44edb2e3b74b2780bc975f06b828203251437403b8bc4e.jpg)

![6df8ded4017b80458e76e0ce30ab2d8490930007b1b53d611a8aef3424e71989.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/6df8ded4017b80458e76e0ce30ab2d8490930007b1b53d611a8aef3424e71989.jpg)

![897bb720986c8aa0bf87d23031563c3d7bacd12aa7ea5c6cadf1398070cc6440.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/897bb720986c8aa0bf87d23031563c3d7bacd12aa7ea5c6cadf1398070cc6440.jpg)

![a35e1948312e718559f0f5b289dd33ea7e33a74f43ebd5349f820c186455c518.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/a35e1948312e718559f0f5b289dd33ea7e33a74f43ebd5349f820c186455c518.jpg)

![c72cf0a19b65071de2989749c39ce85a50bc06566d7e5a67c79dc2021ebce552.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/c72cf0a19b65071de2989749c39ce85a50bc06566d7e5a67c79dc2021ebce552.jpg)

![f760b5118fc45b5c26c6fb16e6fa5db0d6118aaa1677f989c9ce7d38d0fb5a7e.jpg](../icml_results/4315_Old%20Habits%20Die%20Hard_%20How%20Conversational%20History%20Geometrically%20Traps%20LLMs/tables/f760b5118fc45b5c26c6fb16e6fa5db0d6118aaa1677f989c9ce7d38d0fb5a7e.jpg)

<a id="paper-4273"></a>

## 3DPoV: Improving 3D understanding via Patch Ordering on Videos


### Images

![0adebf3cc38c698791f73b8696852ed34acd230e2a688adf96455ae1904a8bc2.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/0adebf3cc38c698791f73b8696852ed34acd230e2a688adf96455ae1904a8bc2.jpg)

![29a290f6bdebc6abb5d3f39388fab86f250f5ca2eb54fda564a5276963a6240a.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/29a290f6bdebc6abb5d3f39388fab86f250f5ca2eb54fda564a5276963a6240a.jpg)

![35a694f45c0833fe2955aab6f5e1a15a88bfa1b1ae5fd62dedb44e527461b9ba.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/35a694f45c0833fe2955aab6f5e1a15a88bfa1b1ae5fd62dedb44e527461b9ba.jpg)

![50a7776fd332aff6380ba283e1423f8a5c87f03b8619d2f1728d1d2a50587e9f.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/50a7776fd332aff6380ba283e1423f8a5c87f03b8619d2f1728d1d2a50587e9f.jpg)

![61f50ac8d35e2226000309ab39c0d0e83da1b276326277b1b858b20467086e3e.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/61f50ac8d35e2226000309ab39c0d0e83da1b276326277b1b858b20467086e3e.jpg)

![67fa63d0e1baa8e95aa98f30cd5f785e12ed73209f76ca716143fdbf5ceddae6.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/67fa63d0e1baa8e95aa98f30cd5f785e12ed73209f76ca716143fdbf5ceddae6.jpg)

![6a66a69a92f41f3ab7d910205b2a6d44be4fe192999b9b7023b85f49d78feb37.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/6a66a69a92f41f3ab7d910205b2a6d44be4fe192999b9b7023b85f49d78feb37.jpg)

![a3409fb0a8b44e99bad0bb173a4196ea9c39bca7f2b47c7c6da7747bfec697a3.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/a3409fb0a8b44e99bad0bb173a4196ea9c39bca7f2b47c7c6da7747bfec697a3.jpg)

![ac1122d72176c601538317cdf6d362c37d7aa938699aa84e21c2a29b03bc8401.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/ac1122d72176c601538317cdf6d362c37d7aa938699aa84e21c2a29b03bc8401.jpg)

![c20f143a311f1984b310e2cd1f88be62325d7eeaf736ffc7546cce49ca18307b.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/c20f143a311f1984b310e2cd1f88be62325d7eeaf736ffc7546cce49ca18307b.jpg)

![c355f8178e54cc59abeff3ac24017af11a7f031d0957d689c3c51e189b6fb65d.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/c355f8178e54cc59abeff3ac24017af11a7f031d0957d689c3c51e189b6fb65d.jpg)

![cda81d47f2fdb5ce04cd74a5035f909451eb55dee8602294f475aaaa0abdddcc.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/cda81d47f2fdb5ce04cd74a5035f909451eb55dee8602294f475aaaa0abdddcc.jpg)

![d08113ca80019309b12758aab68a91ecf4be50c3d7f9ae475310271851a5954f.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/d08113ca80019309b12758aab68a91ecf4be50c3d7f9ae475310271851a5954f.jpg)

![dd06b5290cd9938684c0c5f2267eea309dc4b9d447e3d97d0cd85af1edf8cf53.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/dd06b5290cd9938684c0c5f2267eea309dc4b9d447e3d97d0cd85af1edf8cf53.jpg)

![df140c8a9b260f87d671219f279a1be9c80cea1e4d6ee410a398753fe9705e77.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/images/df140c8a9b260f87d671219f279a1be9c80cea1e4d6ee410a398753fe9705e77.jpg)

### Tables

![0029830b291ac49112351bc370a51a8f800f43a539866ef8f2751d1594fa7dd9.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/0029830b291ac49112351bc370a51a8f800f43a539866ef8f2751d1594fa7dd9.jpg)

![07f87f4c415056b959727a957e9ed0ee30b723c6f465a456d505790c14044adb.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/07f87f4c415056b959727a957e9ed0ee30b723c6f465a456d505790c14044adb.jpg)

![0cf490279aa8092c150d3a308cfb8cad5a64dd7ec05dce34ec168fb5c6fca693.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/0cf490279aa8092c150d3a308cfb8cad5a64dd7ec05dce34ec168fb5c6fca693.jpg)

![0cff67eeab3959b2c63eb30f72a6a293888da2f67b3bed5a5f5219698d187f04.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/0cff67eeab3959b2c63eb30f72a6a293888da2f67b3bed5a5f5219698d187f04.jpg)

![0e2193a717784ece4018d9d0ddafd98619eb9f5d5830587f6848380b72e16b24.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/0e2193a717784ece4018d9d0ddafd98619eb9f5d5830587f6848380b72e16b24.jpg)

![1012155db969d7a1dc5c2b33c4ae402dbfbafa528345e871be2ac7d594d14bef.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/1012155db969d7a1dc5c2b33c4ae402dbfbafa528345e871be2ac7d594d14bef.jpg)

![193a788effb9fc37bcb2b14632191fa5e28ddc162263fb9f64e07625e2ed1d25.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/193a788effb9fc37bcb2b14632191fa5e28ddc162263fb9f64e07625e2ed1d25.jpg)

![22187bb0b4b76079172f153a05f528746bc2a993a4b0be903064757ac33af7dd.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/22187bb0b4b76079172f153a05f528746bc2a993a4b0be903064757ac33af7dd.jpg)

![2d3c6a0fda5234f65ad16b6a64abaaccac56a3160d24ac7889b4569bd9f66010.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/2d3c6a0fda5234f65ad16b6a64abaaccac56a3160d24ac7889b4569bd9f66010.jpg)

![462fccc528878f1463efbf6c9d6eb0b4c1b53042c1eff7db5a5b77f7885b1dd1.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/462fccc528878f1463efbf6c9d6eb0b4c1b53042c1eff7db5a5b77f7885b1dd1.jpg)

![4bb6a72dfb3f0a17164f54fdaf3604d859eb77afbc4cb25adec34d53a487973d.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/4bb6a72dfb3f0a17164f54fdaf3604d859eb77afbc4cb25adec34d53a487973d.jpg)

![5775004a75f5bd3181c8f32f12591b5649453239d5a4fe5c4d5d8bb51eb025ab.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/5775004a75f5bd3181c8f32f12591b5649453239d5a4fe5c4d5d8bb51eb025ab.jpg)

![5cc02071adcb3e08758276c9d354cd3e13aa975d3707715e0f2c5da771fe17dc.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/5cc02071adcb3e08758276c9d354cd3e13aa975d3707715e0f2c5da771fe17dc.jpg)

![66ce562d2aaad97622c4c1ac5ee5a488f8608b04f97dceb359141ba813fa75fa.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/66ce562d2aaad97622c4c1ac5ee5a488f8608b04f97dceb359141ba813fa75fa.jpg)

![677df96f2c5840abb8385c3662b44c6abe2d17fff2e5d870883936dbfdc9bb1d.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/677df96f2c5840abb8385c3662b44c6abe2d17fff2e5d870883936dbfdc9bb1d.jpg)

![703e19155441a4d9db1324e9e0a7bf135c491ea3fd513250b28cef67cb21ed5e.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/703e19155441a4d9db1324e9e0a7bf135c491ea3fd513250b28cef67cb21ed5e.jpg)

![70ea081211abef8605c8cf4ed46dee7c886a75242a68d8a518acaf9776d8b2d2.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/70ea081211abef8605c8cf4ed46dee7c886a75242a68d8a518acaf9776d8b2d2.jpg)

![77d3c6a39d215553a33a5b0df1c494238de5bab8124aab70a01f26cb733a09d6.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/77d3c6a39d215553a33a5b0df1c494238de5bab8124aab70a01f26cb733a09d6.jpg)

![7da43c29594b266418937125c4a21fe04d753df9f243acacdfb764c39b19cf7b.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/7da43c29594b266418937125c4a21fe04d753df9f243acacdfb764c39b19cf7b.jpg)

![82d7a4c7fa8d76f7d38c13a12f00dbf924b42f25b70d15e0518b6a96ad1aca20.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/82d7a4c7fa8d76f7d38c13a12f00dbf924b42f25b70d15e0518b6a96ad1aca20.jpg)

![88a40c70732c7ffb4f1f5116a344fe3b223b65126b691969559413ad9afab382.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/88a40c70732c7ffb4f1f5116a344fe3b223b65126b691969559413ad9afab382.jpg)

![8b07ff1fea6ef06eef92b2dc23a48dab84b4ad0111951da7fc0d8800ccc29288.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/8b07ff1fea6ef06eef92b2dc23a48dab84b4ad0111951da7fc0d8800ccc29288.jpg)

![90dd46a3d3faed3338e2b9c999b3eb14d8b70e3e5b73c74579a5d945897247d7.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/90dd46a3d3faed3338e2b9c999b3eb14d8b70e3e5b73c74579a5d945897247d7.jpg)

![962d1be99ba2489a0183e0a17e45bc4b1948af576eda89389f819c1e74606fe4.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/962d1be99ba2489a0183e0a17e45bc4b1948af576eda89389f819c1e74606fe4.jpg)

![a0740e5a66d745efbf16275a800629c86af06176b510956f517dacd372690347.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/a0740e5a66d745efbf16275a800629c86af06176b510956f517dacd372690347.jpg)

![b857bb1b1608e09dc344b153cb2f39e5e91f7d02be013c4fca1eb5aef6938870.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/b857bb1b1608e09dc344b153cb2f39e5e91f7d02be013c4fca1eb5aef6938870.jpg)

![b91746a253afff0805d32a600fc518060574fa0cf27241a1a531af2914ec2457.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/b91746a253afff0805d32a600fc518060574fa0cf27241a1a531af2914ec2457.jpg)

![d0509d9d40b78df6a13a69bae00da957c7f33e2c6a0549303225a5c130b3f762.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/d0509d9d40b78df6a13a69bae00da957c7f33e2c6a0549303225a5c130b3f762.jpg)

![d43c29f555ccb969f3d9a13a199c846badefabd84d4acb2bce48966cff194fd8.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/d43c29f555ccb969f3d9a13a199c846badefabd84d4acb2bce48966cff194fd8.jpg)

![d63f5415b6958f9d722a3b016af7718d5a50130a342105fba76a56c2890503f6.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/d63f5415b6958f9d722a3b016af7718d5a50130a342105fba76a56c2890503f6.jpg)

![e1102d237dfce58de3155e4bda4f2caab78e712596b87ea36e495c9413832219.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/e1102d237dfce58de3155e4bda4f2caab78e712596b87ea36e495c9413832219.jpg)

![e87b07071af167919cbeb1539fa2588b22103b1f2556215baa382623a9a6630e.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/e87b07071af167919cbeb1539fa2588b22103b1f2556215baa382623a9a6630e.jpg)

![efe06e6f0709e0007520f94904051035a32e1d8b503d47819871ef51d3710985.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/efe06e6f0709e0007520f94904051035a32e1d8b503d47819871ef51d3710985.jpg)

![f166297eac6e5b027a25a7161e6431a43840b81f1f6cf343472001bff9af86dd.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/f166297eac6e5b027a25a7161e6431a43840b81f1f6cf343472001bff9af86dd.jpg)

![f48ec2a9ed9f32944b3114cbe5ee660607b0e83d85621ba4db0b485b3a3774f0.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/f48ec2a9ed9f32944b3114cbe5ee660607b0e83d85621ba4db0b485b3a3774f0.jpg)

![f70057b3e31ad88e0e3954b762998ab3fb2e0fcb180b2ca67a6fa488fbc2d79d.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/f70057b3e31ad88e0e3954b762998ab3fb2e0fcb180b2ca67a6fa488fbc2d79d.jpg)

![fd8e0eb81bd10039c9b01402bedc5e3e11965bf0048e1df53f95445aa25e173f.jpg](../icml_results/4316_3DPoV_%20Improving%203D%20understanding%20via%20Patch%20Ordering%20on%20Videos/tables/fd8e0eb81bd10039c9b01402bedc5e3e11965bf0048e1df53f95445aa25e173f.jpg)

<a id="paper-4274"></a>

## Training with Honeypots: Reshaping How LLMs Fail Under Adversarial Attacks


### Images

![622ffe3995f5af1bda37edf4aeb5786300d299aca76eb5c68c747bd93d748f12.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/images/622ffe3995f5af1bda37edf4aeb5786300d299aca76eb5c68c747bd93d748f12.jpg)

![75c53d4a9fec8962544fecc673ca796806b75b61d09deb8bf2f2895cea957997.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/images/75c53d4a9fec8962544fecc673ca796806b75b61d09deb8bf2f2895cea957997.jpg)

![fde03ae5ee07fd02968a76b2ceed86da2bf1407dd3332c07881deb4118ff7cc5.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/images/fde03ae5ee07fd02968a76b2ceed86da2bf1407dd3332c07881deb4118ff7cc5.jpg)

### Tables

![1243b9d6e30c3f4843f84e979ed58dfde2c35d5bfb0101dafa6d45f1e3007e5d.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/1243b9d6e30c3f4843f84e979ed58dfde2c35d5bfb0101dafa6d45f1e3007e5d.jpg)

![1824d165174aea332c4df7187073a9129c9499587716b436e2efe9da6ea3da51.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/1824d165174aea332c4df7187073a9129c9499587716b436e2efe9da6ea3da51.jpg)

![2305a2762ffbeedf6cdd866626e09f36d1687e7d0536696360620aac27bb03bf.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/2305a2762ffbeedf6cdd866626e09f36d1687e7d0536696360620aac27bb03bf.jpg)

![39de6a504ac902931b23ea9aada84dc5c0cff47b0a080e69c868ebf21daf2fea.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/39de6a504ac902931b23ea9aada84dc5c0cff47b0a080e69c868ebf21daf2fea.jpg)

![3fa7f79a18e872a87e53768311e30cbdcf944f0aa45c1cdffb96f44409726b10.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/3fa7f79a18e872a87e53768311e30cbdcf944f0aa45c1cdffb96f44409726b10.jpg)

![b19053f0480f7166062a92c73c7aa7ae761ede5f3c12385fdd3f578eeb22db47.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/b19053f0480f7166062a92c73c7aa7ae761ede5f3c12385fdd3f578eeb22db47.jpg)

![b52373015d1679e2dd32bad8d33aaee1f9bd07c3fc7d17fce60354f40c3c421f.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/b52373015d1679e2dd32bad8d33aaee1f9bd07c3fc7d17fce60354f40c3c421f.jpg)

![b835092a20fd114d392b3f62ae372caef6dea211a9d97a38ea7b70e31f402178.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/b835092a20fd114d392b3f62ae372caef6dea211a9d97a38ea7b70e31f402178.jpg)

![c423cda17842a3a0b27454ccfdd47c35a719595e2e3d37624bc03b95b42c7002.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/c423cda17842a3a0b27454ccfdd47c35a719595e2e3d37624bc03b95b42c7002.jpg)

![c86ca6e6daa89c664e78f5e832f560c2fdb1a05edd120afb535e8e371279dd62.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/c86ca6e6daa89c664e78f5e832f560c2fdb1a05edd120afb535e8e371279dd62.jpg)

![cc8a4fec986099d26b2642734090dfb0e50be57a089c3645394aaf7f4be1df94.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/cc8a4fec986099d26b2642734090dfb0e50be57a089c3645394aaf7f4be1df94.jpg)

![d290f5d5f0fbd7dcee860e5f4cb7f4a87b3aa1de6c5a3fd03546a8184aa8cc46.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/d290f5d5f0fbd7dcee860e5f4cb7f4a87b3aa1de6c5a3fd03546a8184aa8cc46.jpg)

![e21fdf7c9fc91be2eff201d5e798e6a2a4a064da57f4f39d2632b30796bc4bb5.jpg](../icml_results/4317_Training%20with%20Honeypots_%20Reshaping%20How%20LLMs%20Fail%20Under%20Adversarial%20Attacks/tables/e21fdf7c9fc91be2eff201d5e798e6a2a4a064da57f4f39d2632b30796bc4bb5.jpg)

<a id="paper-4275"></a>

## On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders


### Images

![0ee23909ea3ffcc2e02f8e94f0443cebdfdb860b2e6dc59aa4151b4a1c9db691.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/0ee23909ea3ffcc2e02f8e94f0443cebdfdb860b2e6dc59aa4151b4a1c9db691.jpg)

![1101abfacc2af8f0637cefea1605618f2d35fc3b13c8fc20d11720476990a033.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/1101abfacc2af8f0637cefea1605618f2d35fc3b13c8fc20d11720476990a033.jpg)

![115eddddb7d94041fa167b5f2ade0b8607c9948f526ab24669241dbd3b5abb11.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/115eddddb7d94041fa167b5f2ade0b8607c9948f526ab24669241dbd3b5abb11.jpg)

![33658b24e0d8d51720cf34f089838aa382e821f29437828ca403ec6830f3347b.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/33658b24e0d8d51720cf34f089838aa382e821f29437828ca403ec6830f3347b.jpg)

![346c5acbbbe17a93c7c1710dcec07cae9db80c5d21b44e5f3feb7d1a7bb7ba1f.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/346c5acbbbe17a93c7c1710dcec07cae9db80c5d21b44e5f3feb7d1a7bb7ba1f.jpg)

![3bd64ffaf3685ed48014501bd2c4eb57c6d0680e4a99ac0568a745d612b741c3.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/3bd64ffaf3685ed48014501bd2c4eb57c6d0680e4a99ac0568a745d612b741c3.jpg)

![4ef7b402381b69b866336ab352bc0e7fcd0f782c452048a0b5c7049e74421db0.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/4ef7b402381b69b866336ab352bc0e7fcd0f782c452048a0b5c7049e74421db0.jpg)

![680049449fb1279d2b5151ac14cff5158f6ac10843ebc3bed94f52ee7b746862.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/680049449fb1279d2b5151ac14cff5158f6ac10843ebc3bed94f52ee7b746862.jpg)

![6e2f0601e4c670968e76537987dcd6c5facd396d4bf8349fe6059fdd9f2b4198.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/6e2f0601e4c670968e76537987dcd6c5facd396d4bf8349fe6059fdd9f2b4198.jpg)

![6e40508d2e98765413e0ac0917b84c35211a86cbeb56429c6b37a41030257136.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/6e40508d2e98765413e0ac0917b84c35211a86cbeb56429c6b37a41030257136.jpg)

![7993243c48dcf980d61cd755366dc54d84ae1af620e2bf2e56f3576817d44ce8.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/7993243c48dcf980d61cd755366dc54d84ae1af620e2bf2e56f3576817d44ce8.jpg)

![8d9c34f0811ccbc619e76b2cde1be0abdd83133d7ccfe6adf56802651848e62e.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/8d9c34f0811ccbc619e76b2cde1be0abdd83133d7ccfe6adf56802651848e62e.jpg)

![8f989c9e11f3796f2e40b7d2fca94d33d9e24cab8252b635e56f28efafb5ee8d.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/8f989c9e11f3796f2e40b7d2fca94d33d9e24cab8252b635e56f28efafb5ee8d.jpg)

![90ceb45591fa1fd0c0d4886cb29a07c6776b28f945eee4c74842fcf3931e2fdd.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/90ceb45591fa1fd0c0d4886cb29a07c6776b28f945eee4c74842fcf3931e2fdd.jpg)

![998103a80a71dcc3ec4b2b6066fe413cd85f3befb74bcb8a1c962ae58e167f41.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/998103a80a71dcc3ec4b2b6066fe413cd85f3befb74bcb8a1c962ae58e167f41.jpg)

![9ca7799f187535e6a5147cb968ec226917272c1808686b3bd63f131e9761c9aa.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/9ca7799f187535e6a5147cb968ec226917272c1808686b3bd63f131e9761c9aa.jpg)

![9fbe3f5fabddd9b1c4b864d13009ed08b10b9f4d1e7df492006499056071efe1.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/9fbe3f5fabddd9b1c4b864d13009ed08b10b9f4d1e7df492006499056071efe1.jpg)

![a0cf6d88c7b10a7fdd76cef9c825c59c505abe8c1483b14bffa8185f0263599e.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/a0cf6d88c7b10a7fdd76cef9c825c59c505abe8c1483b14bffa8185f0263599e.jpg)

![aca7d619e27872eeaff4be8fdcb77112972c18657f78e108a63646ae1cf62e9c.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/aca7d619e27872eeaff4be8fdcb77112972c18657f78e108a63646ae1cf62e9c.jpg)

![af7fcc2dfdaaaa9c406c1ad4b705fb2344ffc800f4697ff0417a270885019a45.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/af7fcc2dfdaaaa9c406c1ad4b705fb2344ffc800f4697ff0417a270885019a45.jpg)

![b5404251e98808ef22e91c863ad379833201c4d7271a60e8fb215dfe3fca6408.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/b5404251e98808ef22e91c863ad379833201c4d7271a60e8fb215dfe3fca6408.jpg)

![b5bcc38b2dfba637f20310b54d3e6a920ef558c9ec8e1b4ca0465a754bf5ff3f.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/b5bcc38b2dfba637f20310b54d3e6a920ef558c9ec8e1b4ca0465a754bf5ff3f.jpg)

![b801bda4d6ceb3d2f138cb726f1936581495168a03de89d495ce18e4021a0055.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/b801bda4d6ceb3d2f138cb726f1936581495168a03de89d495ce18e4021a0055.jpg)

![c2586ad762d6229a3b32b9db8315015e34aff44f7d9efeaea25d24cd12cba704.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/c2586ad762d6229a3b32b9db8315015e34aff44f7d9efeaea25d24cd12cba704.jpg)

![dd78763d95976597daa77fc0aaa15525e3377ef06d04be98b945146d3e97f91a.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/dd78763d95976597daa77fc0aaa15525e3377ef06d04be98b945146d3e97f91a.jpg)

![e1b4cd096482032d1215d341dc9c981ac4923a16c8d5ac9740ae75d47f0b91a1.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/e1b4cd096482032d1215d341dc9c981ac4923a16c8d5ac9740ae75d47f0b91a1.jpg)

![fc7e99941cb15ffb212011d4d39625c6648764ed3e46fe2a9bb41b0050610a7d.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/images/fc7e99941cb15ffb212011d4d39625c6648764ed3e46fe2a9bb41b0050610a7d.jpg)

### Tables

![0ce03690c52bcee27938b33e529c32605b99a100c5a09e257dda7107689c7e4b.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/0ce03690c52bcee27938b33e529c32605b99a100c5a09e257dda7107689c7e4b.jpg)

![5821a0607861d47e6c57416555696427fd4288f753d53e39178391ac71517e39.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/5821a0607861d47e6c57416555696427fd4288f753d53e39178391ac71517e39.jpg)

![6108e30342ecdeb69c8dfa46e3a527df936fa18666578feed347b669d897f881.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/6108e30342ecdeb69c8dfa46e3a527df936fa18666578feed347b669d897f881.jpg)

![6398ecdff5387f1d39a06ace213c7bddb2a16baeaca26f26a9562e4755685b0e.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/6398ecdff5387f1d39a06ace213c7bddb2a16baeaca26f26a9562e4755685b0e.jpg)

![90d4ae5d9d5c49e99383c51dd86f96650f88fb6d34ddbe81f913baa37fdce0ac.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/90d4ae5d9d5c49e99383c51dd86f96650f88fb6d34ddbe81f913baa37fdce0ac.jpg)

![9886ddac5b4ca689b8db2882c758d078070afb2cbfba800f5037358b4c4f4c19.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/9886ddac5b4ca689b8db2882c758d078070afb2cbfba800f5037358b4c4f4c19.jpg)

![a22940b52b7b767b41e7adfac86e45eb106b60298168aae62b4ac8f8812e8b1c.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/a22940b52b7b767b41e7adfac86e45eb106b60298168aae62b4ac8f8812e8b1c.jpg)

![b3929c51e3147693f9a28c4c4a3d49f74b38e6f8a35bb47142189f0e6fa305f7.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/b3929c51e3147693f9a28c4c4a3d49f74b38e6f8a35bb47142189f0e6fa305f7.jpg)

![c6f6d8e73b7d8d0a7120c0a37ef6c4ef17c872900f20f7eb3d6d91907d98091e.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/c6f6d8e73b7d8d0a7120c0a37ef6c4ef17c872900f20f7eb3d6d91907d98091e.jpg)

![e89b7e2c5ca8041fc4902002fe994dfb2a79e825b08b12f5f025aef2bb581b0a.jpg](../icml_results/4318_On%20the%20Relationship%20Between%20Activation%20Outliers%20and%20Feature%20Death%20in%20Sparse%20Autoencoders/tables/e89b7e2c5ca8041fc4902002fe994dfb2a79e825b08b12f5f025aef2bb581b0a.jpg)

<a id="paper-4276"></a>

## Unsupervised Disentanglement Without Compromises : How Functional Orthogonality Enforces Identifiability


### Images

![087bb18913f63d34f8ae0bc9a97797748d4bb3b7872dd70a10b98c5a3c555cc9.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/087bb18913f63d34f8ae0bc9a97797748d4bb3b7872dd70a10b98c5a3c555cc9.jpg)

![12236f26b26bb5f09bcc5369a6ed5ecadca016a7b190fd5d26b9f9db03db32b3.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/12236f26b26bb5f09bcc5369a6ed5ecadca016a7b190fd5d26b9f9db03db32b3.jpg)

![1b61658a508726510894dca07c1926754530fa9414f53b403069669bd1845b2a.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/1b61658a508726510894dca07c1926754530fa9414f53b403069669bd1845b2a.jpg)

![1f5c58eca771bb0d9cc54983cf5f298cf88cc81b17a549aa334130be8883bfc3.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/1f5c58eca771bb0d9cc54983cf5f298cf88cc81b17a549aa334130be8883bfc3.jpg)

![49e8d7d674b5cfaf65ddff4fd39159dfa8ba716954de3badf555c32121ac7ebd.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/49e8d7d674b5cfaf65ddff4fd39159dfa8ba716954de3badf555c32121ac7ebd.jpg)

![7ab9da30a065a147be7ff26874bd884d45a8bf528b0f6eedf1ed5e19c5e59f81.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/7ab9da30a065a147be7ff26874bd884d45a8bf528b0f6eedf1ed5e19c5e59f81.jpg)

![9ba4fbc7090e29c924ccf021b2e35f39439f1fc7023e24bc2386298c75bc8048.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/9ba4fbc7090e29c924ccf021b2e35f39439f1fc7023e24bc2386298c75bc8048.jpg)

![9d9780e2a4f0a8e047e7c86d4305731caeb3ae7577d2ca6a860e235e223e092d.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/9d9780e2a4f0a8e047e7c86d4305731caeb3ae7577d2ca6a860e235e223e092d.jpg)

![acf7199c9a5e8e439a69917a182593b8bc9ab6937f878f57ebb79f895e1ad826.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/acf7199c9a5e8e439a69917a182593b8bc9ab6937f878f57ebb79f895e1ad826.jpg)

![ad7ceaf0a5b4a3ca1db7b9c39f0f743eb5db9cca377ce27c687da3fa91c02301.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/ad7ceaf0a5b4a3ca1db7b9c39f0f743eb5db9cca377ce27c687da3fa91c02301.jpg)

![bd9f8e2e923a14eb408aafa15724b6e11e86fb1c38202682c8ba1003098e72b6.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/bd9f8e2e923a14eb408aafa15724b6e11e86fb1c38202682c8ba1003098e72b6.jpg)

![c5dbf1b52e062c7ffdc7bc35da04778f2c59a2f924e7269dde0191851af5271a.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/c5dbf1b52e062c7ffdc7bc35da04778f2c59a2f924e7269dde0191851af5271a.jpg)

![c82710b29d0bb4c5525b8612bb3e502f09056c26333733d10f26e34b414d5af4.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/c82710b29d0bb4c5525b8612bb3e502f09056c26333733d10f26e34b414d5af4.jpg)

![e370fe19e6cd75ee4341a2142cf7ddeb9f89cdfd47c1977804b8b9797d40403e.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/e370fe19e6cd75ee4341a2142cf7ddeb9f89cdfd47c1977804b8b9797d40403e.jpg)

![e505d586e6f23029cb8fd2bfc9d1e33f30022a9c2a46b8550952a51f2a73df09.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/e505d586e6f23029cb8fd2bfc9d1e33f30022a9c2a46b8550952a51f2a73df09.jpg)

![fa11ff7d7b6054e9cf63e1e47cae9d8d435a67a6928d6267e4b0b6cc8acdf468.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/images/fa11ff7d7b6054e9cf63e1e47cae9d8d435a67a6928d6267e4b0b6cc8acdf468.jpg)

### Tables

![020f022f578a289a8a05835bee5ba1bc808deb848c578c157cadff9d1ad82131.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/tables/020f022f578a289a8a05835bee5ba1bc808deb848c578c157cadff9d1ad82131.jpg)

![6f5c4f0c8fd6ddbef5956a9f42c429141a91892e114ad39b64ab3686d3008d57.jpg](../icml_results/4319_Unsupervised%20Disentanglement%20Without%20Compromises%20_%20How%20Functional%20Orthogonality%20Enforces%20Identifiabi/tables/6f5c4f0c8fd6ddbef5956a9f42c429141a91892e114ad39b64ab3686d3008d57.jpg)

<a id="paper-4277"></a>

## Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies


### Images

![0290f3b6f864ea6503356e56ba24866fc3e438fb28a187c433729e95c0567daf.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/0290f3b6f864ea6503356e56ba24866fc3e438fb28a187c433729e95c0567daf.jpg)

![209aa0ba652c043832eda0cb0ed9b3bba1a545e193580d9691a0c90fdaf4fee7.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/209aa0ba652c043832eda0cb0ed9b3bba1a545e193580d9691a0c90fdaf4fee7.jpg)

![33c981a30c4c83a167604190997f2754735cbfa6a2432f3a2a8012367324151a.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/33c981a30c4c83a167604190997f2754735cbfa6a2432f3a2a8012367324151a.jpg)

![3671b0218f5b1a6b09ea14b632f471e2b10ca47541c42148d2ff3593565d5e24.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/3671b0218f5b1a6b09ea14b632f471e2b10ca47541c42148d2ff3593565d5e24.jpg)

![372d08b3e6c3bfc4633806da42fef6b48c32e909163c0646acb9a64c75513100.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/372d08b3e6c3bfc4633806da42fef6b48c32e909163c0646acb9a64c75513100.jpg)

![375cf5cd51ae38b47d2486928b3bd5d344d0105a09b75e7046b9684aa6da9099.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/375cf5cd51ae38b47d2486928b3bd5d344d0105a09b75e7046b9684aa6da9099.jpg)

![4e34989fb501de09c498ca3108d85954d7899a44bc78229eef25068e1eacd6ee.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/4e34989fb501de09c498ca3108d85954d7899a44bc78229eef25068e1eacd6ee.jpg)

![535ddb061caa256e8a337c2037a35361ce3770d7e94e996545b7941ef940c08d.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/535ddb061caa256e8a337c2037a35361ce3770d7e94e996545b7941ef940c08d.jpg)

![62e25517a7d4965287d5ddb2940862aa2310f15036fdd7f2157cad5442af07c2.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/62e25517a7d4965287d5ddb2940862aa2310f15036fdd7f2157cad5442af07c2.jpg)

![7757f74df97d479e08ec8376be35c8f83a9c641a934c29b39c77749c9397d2d6.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/7757f74df97d479e08ec8376be35c8f83a9c641a934c29b39c77749c9397d2d6.jpg)

![860adcdd8a37355f04f6698788ba7e900ce45596e7e372af1488bb58b1cb6ae6.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/860adcdd8a37355f04f6698788ba7e900ce45596e7e372af1488bb58b1cb6ae6.jpg)

![d29cc0e7bc43d4e685e635790378b76fdf09a4d4c02faae4702fff4f9de2e0a2.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/d29cc0e7bc43d4e685e635790378b76fdf09a4d4c02faae4702fff4f9de2e0a2.jpg)

![fefe124693caea66f077df820191acba0138e7bd10a14dc1662e10a7b630f858.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/images/fefe124693caea66f077df820191acba0138e7bd10a14dc1662e10a7b630f858.jpg)

### Tables

![15180969082107d7c4d3baa13aa8451e1a7cd9791fe6e97660c3c85bbd0fe921.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/15180969082107d7c4d3baa13aa8451e1a7cd9791fe6e97660c3c85bbd0fe921.jpg)

![216a33a5f715ec0781b2c9c7a38703baa67749e900da8b9f22bca15cde73c0af.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/216a33a5f715ec0781b2c9c7a38703baa67749e900da8b9f22bca15cde73c0af.jpg)

![6c3fc032f22e8c8502f0f4db5c4b76d03fbf3a80b9597fc3bd5bb1d8619f7753.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/6c3fc032f22e8c8502f0f4db5c4b76d03fbf3a80b9597fc3bd5bb1d8619f7753.jpg)

![97cb4d1ab33c03cc5f18fa347928e7bd8ed1c6b5d6cbd2c4b19c89895dcadc3d.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/97cb4d1ab33c03cc5f18fa347928e7bd8ed1c6b5d6cbd2c4b19c89895dcadc3d.jpg)

![b50398bdda5288282ca95512bb63e6699763f1e1237de3c5b4a1306d7f18f796.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/b50398bdda5288282ca95512bb63e6699763f1e1237de3c5b4a1306d7f18f796.jpg)

![f45bc957b61f79664e42c4f3cb4896bc65815fb4024ceabf3d9cf1c16e30aa88.jpg](../icml_results/4320_Lagrangian%20Perturbation%20Diffusion%20Steering_%20Latent%20Reinforcement%20Learning%20for%20Generative%20Policies/tables/f45bc957b61f79664e42c4f3cb4896bc65815fb4024ceabf3d9cf1c16e30aa88.jpg)

<a id="paper-4278"></a>

## Optimal Fair Aggregation of Crowdsourced Noisy Labels using Demographic Parity Constraints


### Images

![429ed4976513b342ea1c00a42b7a5c5f191c6a72341c9e10c7243707482230b9.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/images/429ed4976513b342ea1c00a42b7a5c5f191c6a72341c9e10c7243707482230b9.jpg)

![48a387d9df7cb7abe15b2994a2b317a59af42948df9a6223960a88af89d44838.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/images/48a387d9df7cb7abe15b2994a2b317a59af42948df9a6223960a88af89d44838.jpg)

![680e5af72a319283bf0c9f3bc2c28e3d56fc28bcd2f0943c8f231a3608520553.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/images/680e5af72a319283bf0c9f3bc2c28e3d56fc28bcd2f0943c8f231a3608520553.jpg)

![6b968b3ebbc2122e42b48d791d25b5b050660d14372413555f3144d78870021d.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/images/6b968b3ebbc2122e42b48d791d25b5b050660d14372413555f3144d78870021d.jpg)

![e2f67cc52f6ebb7dbbcef85863b96db42a01725477ffbb7be4d6cf208ce7b80c.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/images/e2f67cc52f6ebb7dbbcef85863b96db42a01725477ffbb7be4d6cf208ce7b80c.jpg)

### Tables

![bc4aaad12b29a965d1b0811ce79771d83ed872818d7cff7e5814fe06518bc6dc.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/tables/bc4aaad12b29a965d1b0811ce79771d83ed872818d7cff7e5814fe06518bc6dc.jpg)

![c0e8852e8ea110184f5b6644bd3f055a17209cc315a432cf1c6473573842e9bf.jpg](../icml_results/4321_Optimal%20Fair%20Aggregation%20of%20Crowdsourced%20Noisy%20Labels%20using%20Demographic%20Parity%20Constraints/tables/c0e8852e8ea110184f5b6644bd3f055a17209cc315a432cf1c6473573842e9bf.jpg)

<a id="paper-4279"></a>

## Curriculum-Guided Layer Scaling for Language Model Pretraining


### Images

![1bf064f912a9a76b6524516e7a18ac7c0d54eaf6d6cd293b8e4c0a671d13abcf.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/images/1bf064f912a9a76b6524516e7a18ac7c0d54eaf6d6cd293b8e4c0a671d13abcf.jpg)

![72b984f73c5015a2a5ba06b1fd57f632a9adffe98fed094b39074984723cfc89.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/images/72b984f73c5015a2a5ba06b1fd57f632a9adffe98fed094b39074984723cfc89.jpg)

![bc779ac4beee5770b9bad73cab78d1ab1ae095cff49b945b0bb3efae07f87953.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/images/bc779ac4beee5770b9bad73cab78d1ab1ae095cff49b945b0bb3efae07f87953.jpg)

![d004cc63757db403e413bde588f37d2d42db2efbdcae95a014a1b95b3e7ed023.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/images/d004cc63757db403e413bde588f37d2d42db2efbdcae95a014a1b95b3e7ed023.jpg)

![e112e821c17c9ba71a073d5057dd3f6899ecd69db6d38aa8c7839cb94a9385ff.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/images/e112e821c17c9ba71a073d5057dd3f6899ecd69db6d38aa8c7839cb94a9385ff.jpg)

### Tables

![0a74f06e3716faff8725119270cdf298aa576fda689c3969e87490491f52f928.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/0a74f06e3716faff8725119270cdf298aa576fda689c3969e87490491f52f928.jpg)

![3dec01b295f8ec6c621dbbcf2727ea826f5cd9e46df5aeee18e00e9433565d61.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/3dec01b295f8ec6c621dbbcf2727ea826f5cd9e46df5aeee18e00e9433565d61.jpg)

![3f063f41dec30e849ec73e1e9889eacdf43317d8cd2c80bfd5cfe3cf421b6c02.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/3f063f41dec30e849ec73e1e9889eacdf43317d8cd2c80bfd5cfe3cf421b6c02.jpg)

![528d2d38c0514694cd35f5933ee863eba85df07adeda30e769cd57bebd680893.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/528d2d38c0514694cd35f5933ee863eba85df07adeda30e769cd57bebd680893.jpg)

![75ba9dae10b744f803c02a58ebb5031dca52955c0a5650d8de53843f2b771c9e.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/75ba9dae10b744f803c02a58ebb5031dca52955c0a5650d8de53843f2b771c9e.jpg)

![764cb589782dab5733df2ded8d072db3c74be467f3859328dd9595b34886314a.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/764cb589782dab5733df2ded8d072db3c74be467f3859328dd9595b34886314a.jpg)

![92c48b0e67d31b274f64fd98b97494d4a39f31185bcd3dacd4097a56f2a9aced.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/92c48b0e67d31b274f64fd98b97494d4a39f31185bcd3dacd4097a56f2a9aced.jpg)

![a2909caf5299c65dd770ce77f69f683b3f2f8970b112e53cfe376f580acae900.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/a2909caf5299c65dd770ce77f69f683b3f2f8970b112e53cfe376f580acae900.jpg)

![ae3e67e850169f2bf59040a6cfe3bdc72d04824dde70368a7bddb1db37bc2cb5.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/ae3e67e850169f2bf59040a6cfe3bdc72d04824dde70368a7bddb1db37bc2cb5.jpg)

![cdea023272904afa34fddce0ca773f1e259868b0649369d8883983b133a463e3.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/cdea023272904afa34fddce0ca773f1e259868b0649369d8883983b133a463e3.jpg)

![fc690ecefa3c8cd5a4b8684b3c3503c48b45f78191211044ebe0f1a047d2dc75.jpg](../icml_results/4322_Curriculum-Guided%20Layer%20Scaling%20for%20Language%20Model%20Pretraining/tables/fc690ecefa3c8cd5a4b8684b3c3503c48b45f78191211044ebe0f1a047d2dc75.jpg)

<a id="paper-4280"></a>

## V1: Unifying Generation and Self-Verification for Parallel Reasoners


### Images

![1a2042bd7c77cba03d573720f6c973b299d2a3e5beb5bcc19f1f41dc0264f7cd.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/1a2042bd7c77cba03d573720f6c973b299d2a3e5beb5bcc19f1f41dc0264f7cd.jpg)

![1ea3b16ea6d63e883cf57fbe158abf513a499b11ea0fe1a9020f0434eda8bb08.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/1ea3b16ea6d63e883cf57fbe158abf513a499b11ea0fe1a9020f0434eda8bb08.jpg)

![2cecbdc793563f8bb71ce62130926143945ea59719697d91e93493a94d866da2.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/2cecbdc793563f8bb71ce62130926143945ea59719697d91e93493a94d866da2.jpg)

![4e5fb8c0a501f634e279809600e911cc5fdadbf41582e02ca3991d26d7bc7772.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/4e5fb8c0a501f634e279809600e911cc5fdadbf41582e02ca3991d26d7bc7772.jpg)

![66d4fe0b09c52e8451a73f4b6b6f78cdec41af8b9bdb2b7a574d5a644216b873.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/66d4fe0b09c52e8451a73f4b6b6f78cdec41af8b9bdb2b7a574d5a644216b873.jpg)

![66e7077ff64d93db6a90895b36a080492142965e8b5d4611b78468fe2c6c0a08.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/66e7077ff64d93db6a90895b36a080492142965e8b5d4611b78468fe2c6c0a08.jpg)

![709f55a85fcde78d0c5ace9a2c41af0f883b29110019f6ad51d67f287066ca28.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/709f55a85fcde78d0c5ace9a2c41af0f883b29110019f6ad51d67f287066ca28.jpg)

![76276026975c0a4dce457aa0f010fde55bb98a21ed60f062af1a9c85e162e57a.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/76276026975c0a4dce457aa0f010fde55bb98a21ed60f062af1a9c85e162e57a.jpg)

![783506055f5b7e53c06b33ce273c90c9ee198a76a8e6f359ffa1768aebfef5cc.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/783506055f5b7e53c06b33ce273c90c9ee198a76a8e6f359ffa1768aebfef5cc.jpg)

![800b688e1c649d3c61059bffb12ec203bf8a48f43d85b5e1f9fc72b8f8042643.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/800b688e1c649d3c61059bffb12ec203bf8a48f43d85b5e1f9fc72b8f8042643.jpg)

![90c204228d00d666c54d619218b7e6a6e2f47563c67d51bfa3a13237beaffda3.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/90c204228d00d666c54d619218b7e6a6e2f47563c67d51bfa3a13237beaffda3.jpg)

![916a94bd92fea893bad5d3ec005ff237f7c4b28a8e4bcc761b761fbc4fc66b2b.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/916a94bd92fea893bad5d3ec005ff237f7c4b28a8e4bcc761b761fbc4fc66b2b.jpg)

![ab541897a02cbd928592abd225cc911e4da8964a86088be9677e74aeddcc68d7.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/ab541897a02cbd928592abd225cc911e4da8964a86088be9677e74aeddcc68d7.jpg)

![b0bceb9294d469991edd9fdff4929bf4be2ebd96ee7d8ad32d1a44106a105bee.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/b0bceb9294d469991edd9fdff4929bf4be2ebd96ee7d8ad32d1a44106a105bee.jpg)

![c9502021f896f3bc43e8d623c18ac37f113611fc8d084c6e04fb61f6ea236b78.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/c9502021f896f3bc43e8d623c18ac37f113611fc8d084c6e04fb61f6ea236b78.jpg)

![f013e78ef25c8d3a8c7fac73566812048b9656bac6dcabe5cd50bc36ba75738d.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/f013e78ef25c8d3a8c7fac73566812048b9656bac6dcabe5cd50bc36ba75738d.jpg)

![f2cc578838de170f5170e5f6b83aa26d7fd7dd74b9e26a130b7f16569369e9bc.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/images/f2cc578838de170f5170e5f6b83aa26d7fd7dd74b9e26a130b7f16569369e9bc.jpg)

### Tables

![01b88e5242f18ba9b9d6413f3a1404b77b51568d5dafae96d8e1d0447ad75717.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/01b88e5242f18ba9b9d6413f3a1404b77b51568d5dafae96d8e1d0447ad75717.jpg)

![0f638e40902269eef6a9d81bd44d63c78583eade6a12ddec45f62cbd4af1f141.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/0f638e40902269eef6a9d81bd44d63c78583eade6a12ddec45f62cbd4af1f141.jpg)

![1c30be20d53ea7a5cf4e96de8b5bffef4f276d8b6475869040304c4d2ace3d98.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/1c30be20d53ea7a5cf4e96de8b5bffef4f276d8b6475869040304c4d2ace3d98.jpg)

![3784d56ad45b08a607cf01b06f94610327f8679c0675ed007cea8fa4932efd69.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/3784d56ad45b08a607cf01b06f94610327f8679c0675ed007cea8fa4932efd69.jpg)

![402bb62b19d45fdf8a6f113821cbe39a9e8624a89fe83c50ae95b9d449963f61.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/402bb62b19d45fdf8a6f113821cbe39a9e8624a89fe83c50ae95b9d449963f61.jpg)

![470f6413d4453b587300484d0130f96973fe70148e9795cc5f462321ef817ea8.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/470f6413d4453b587300484d0130f96973fe70148e9795cc5f462321ef817ea8.jpg)

![67c92346b8a421a2294d4f4ebc9d6bd0703164976b936e65112c0098702cdc5b.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/67c92346b8a421a2294d4f4ebc9d6bd0703164976b936e65112c0098702cdc5b.jpg)

![d08f1772de767242cad5025a61e147422f1cde9926558661a1a282b29ae6d87a.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/d08f1772de767242cad5025a61e147422f1cde9926558661a1a282b29ae6d87a.jpg)

![d7c86b49ab727d8fad7f78bb1b565b644078851408bd067091c59c3857a0bc2f.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/d7c86b49ab727d8fad7f78bb1b565b644078851408bd067091c59c3857a0bc2f.jpg)

![e585b435e50e3f3b2e7aba1f43407f8cf569929e31753583380af96d20380c7e.jpg](../icml_results/4323_V1_%20Unifying%20Generation%20and%20Self-Verification%20for%20Parallel%20Reasoners/tables/e585b435e50e3f3b2e7aba1f43407f8cf569929e31753583380af96d20380c7e.jpg)

<a id="paper-4281"></a>

## RL with Learnable Textual Feedback: A Bilevel Approach


### Images

![3f8d9b98c9294a9a2ab7bd56ffd87bddcadfb12d69e420adfdc8d3c5b6ff3214.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/images/3f8d9b98c9294a9a2ab7bd56ffd87bddcadfb12d69e420adfdc8d3c5b6ff3214.jpg)

![87d69ae8e981089aec7a5d48588f3ee4e9fae39a217a056cc76530f5fb32a10f.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/images/87d69ae8e981089aec7a5d48588f3ee4e9fae39a217a056cc76530f5fb32a10f.jpg)

![cc9d3a77d971155d9ee66f0ce2c19f2ad1b578d08e7ce70a4fc4acc2af8820bc.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/images/cc9d3a77d971155d9ee66f0ce2c19f2ad1b578d08e7ce70a4fc4acc2af8820bc.jpg)

![d5d50d03e32dda935587e5b7744f510b74b366d87def153c86424f182fa4b2d8.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/images/d5d50d03e32dda935587e5b7744f510b74b366d87def153c86424f182fa4b2d8.jpg)

![e649c7f666101ccd1ab7b7a8ec0a8c0473b672ec53b982c8fd13039969691d29.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/images/e649c7f666101ccd1ab7b7a8ec0a8c0473b672ec53b982c8fd13039969691d29.jpg)

### Tables

![15657ced1d8354a661cf9c907ad645a713d1dec678d4cda7d474f638a330f562.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/15657ced1d8354a661cf9c907ad645a713d1dec678d4cda7d474f638a330f562.jpg)

![1940e42838c8946398c7b0453335c80052eba524856ab91aa3b4815746c2726c.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/1940e42838c8946398c7b0453335c80052eba524856ab91aa3b4815746c2726c.jpg)

![1fad0cfea2f5c5d8873dd7e7157df03a68e611a02a35a7a7e4f90a8924b0af64.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/1fad0cfea2f5c5d8873dd7e7157df03a68e611a02a35a7a7e4f90a8924b0af64.jpg)

![26402b4471a2b72a0f20affec4f780714792c2a6c3c15c9d6e15c1d50317cc94.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/26402b4471a2b72a0f20affec4f780714792c2a6c3c15c9d6e15c1d50317cc94.jpg)

![610ab3fe17ab69982ff22bb9bfbe157b942f3cecb61a769ce1692493ea2a2915.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/610ab3fe17ab69982ff22bb9bfbe157b942f3cecb61a769ce1692493ea2a2915.jpg)

![7a00a3f860fed5dad8077bdcc45018b112059d7fe0e241e40060446ffa18e766.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/7a00a3f860fed5dad8077bdcc45018b112059d7fe0e241e40060446ffa18e766.jpg)

![8e0dbda41c3d00922966130afe5d3e2b052069f2a56b98af25f876f73209b066.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/8e0dbda41c3d00922966130afe5d3e2b052069f2a56b98af25f876f73209b066.jpg)

![8f7f7506213d7c9c51f8c36fad0332c256b10c8c47e558880910dc70041e2882.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/8f7f7506213d7c9c51f8c36fad0332c256b10c8c47e558880910dc70041e2882.jpg)

![9cf19ed5c39c28ea2f64c2bb704bbd2b88f5ec8c43fd92c753cddd531cf2a2cd.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/9cf19ed5c39c28ea2f64c2bb704bbd2b88f5ec8c43fd92c753cddd531cf2a2cd.jpg)

![ae99b497f95d9e542037e868ffbf521b3b4e2288d4e17eaf83f599eaf8548af6.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/ae99b497f95d9e542037e868ffbf521b3b4e2288d4e17eaf83f599eaf8548af6.jpg)

![e13e6cd10c65b7539208f813e22ca2f6106a254eb0bcc55239c9b74182e2ccae.jpg](../icml_results/4324_RL%20with%20Learnable%20Textual%20Feedback_%20A%20Bilevel%20Approach/tables/e13e6cd10c65b7539208f813e22ca2f6106a254eb0bcc55239c9b74182e2ccae.jpg)

<a id="paper-4282"></a>

## DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone


### Images

![2c4c9600d6fbf626cf3ea20bca7761ff4b81dc6a01abe9ccdc85293a79aea943.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/images/2c4c9600d6fbf626cf3ea20bca7761ff4b81dc6a01abe9ccdc85293a79aea943.jpg)

![790726135c64c621141158e77b7dd881e65a32b79e8797d9570237cb5d874cc4.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/images/790726135c64c621141158e77b7dd881e65a32b79e8797d9570237cb5d874cc4.jpg)

![8d9a73b30aca62a8a773eeba8292885b584b0afe437a92f291375073b4252403.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/images/8d9a73b30aca62a8a773eeba8292885b584b0afe437a92f291375073b4252403.jpg)

![8ecb090b8f8a59711c4afff58de6cb4b72bc42fdbc6b763c9e642290928e1dc4.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/images/8ecb090b8f8a59711c4afff58de6cb4b72bc42fdbc6b763c9e642290928e1dc4.jpg)

![dc2041d4108cfe0ac77fe9a36c6a1a77262462d23e429bbacd4cb3475105aeae.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/images/dc2041d4108cfe0ac77fe9a36c6a1a77262462d23e429bbacd4cb3475105aeae.jpg)

### Tables

![17a8d6dd8bac96cc3ce73749678daf9a7fd27602b0cefe799c577be795dff5af.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/17a8d6dd8bac96cc3ce73749678daf9a7fd27602b0cefe799c577be795dff5af.jpg)

![1a8acaba4839757886af3530e59044345ed135aeeeddc2a6c333edeac0ed605b.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/1a8acaba4839757886af3530e59044345ed135aeeeddc2a6c333edeac0ed605b.jpg)

![2f7ccb985d42878aabdea63d8df18712514d04088b2a0b4538c3301d9c3c6ec4.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/2f7ccb985d42878aabdea63d8df18712514d04088b2a0b4538c3301d9c3c6ec4.jpg)

![3d2773cdb30f6b6df3bdbfc32126cc9cfad28523b2db28037686c869f2ff6b7c.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/3d2773cdb30f6b6df3bdbfc32126cc9cfad28523b2db28037686c869f2ff6b7c.jpg)

![4c4b8ceb1799543f5c7a4e99b44ca4f4ffeeb936664678c53b03d8b1ee63ba67.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/4c4b8ceb1799543f5c7a4e99b44ca4f4ffeeb936664678c53b03d8b1ee63ba67.jpg)

![5bdd956d2ca6c894c3d96e56923e5cac7bc9422f69fb400393f781461f5875b7.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/5bdd956d2ca6c894c3d96e56923e5cac7bc9422f69fb400393f781461f5875b7.jpg)

![769c323d3f83d2ec3d9f3e6f20831ea849e789cd631a655ae08ec4ee812d83d2.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/769c323d3f83d2ec3d9f3e6f20831ea849e789cd631a655ae08ec4ee812d83d2.jpg)

![8376f979c0a4e7090091608ca3c46557bd87d53234c6679a87ac48dbf63b0d03.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/8376f979c0a4e7090091608ca3c46557bd87d53234c6679a87ac48dbf63b0d03.jpg)

![8da66c9e92ee1c9f2a046035514f915d172ffa0eecab401a66c12c55a0abe352.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/8da66c9e92ee1c9f2a046035514f915d172ffa0eecab401a66c12c55a0abe352.jpg)

![946c8fe729f4945118e73d38625e02b70e49e447bb34c7f4349f15ec54fdfa3e.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/946c8fe729f4945118e73d38625e02b70e49e447bb34c7f4349f15ec54fdfa3e.jpg)

![9980eb8954e0536bfff36aca7e10899274d320d6d59bcd0af9cec1b1b6a94642.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/9980eb8954e0536bfff36aca7e10899274d320d6d59bcd0af9cec1b1b6a94642.jpg)

![ac3bed2f5f16e7ec13837ff950195e4c93fe3e28faed2f010fbc7570c2bbcaaf.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/ac3bed2f5f16e7ec13837ff950195e4c93fe3e28faed2f010fbc7570c2bbcaaf.jpg)

![e4fc9f3223eca4aad839a44d3464595e5d2021d5ca955124ed22e9d4527a9644.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/e4fc9f3223eca4aad839a44d3464595e5d2021d5ca955124ed22e9d4527a9644.jpg)

![e63bd264cee5bfc4a7abc28bb3b8f36d756cacaece566e005b4e76a11885163c.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/e63bd264cee5bfc4a7abc28bb3b8f36d756cacaece566e005b4e76a11885163c.jpg)

![e9a6e08353359d1233eb79689aac5086177bda196e12ee7fe17767c81ad8091c.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/e9a6e08353359d1233eb79689aac5086177bda196e12ee7fe17767c81ad8091c.jpg)

![ed6bede698662c5b6c1c279a183fcb474c3d0ec62748d1f005c71fb406de775a.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/ed6bede698662c5b6c1c279a183fcb474c3d0ec62748d1f005c71fb406de775a.jpg)

![f3e7308a39d0f257d9da6913c77fb6bf336db2d67b6f971b6182857f93d4086e.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/f3e7308a39d0f257d9da6913c77fb6bf336db2d67b6f971b6182857f93d4086e.jpg)

![fc826812b016975e48bb71b364914a43218d7ba5e4870fc5585bf30ce09ae8d1.jpg](../icml_results/4325_DiffuMamba_%20High-Throughput%20Diffusion%20LMs%20with%20Mamba%20Backbone/tables/fc826812b016975e48bb71b364914a43218d7ba5e4870fc5585bf30ce09ae8d1.jpg)

<a id="paper-4283"></a>

## Evaluating Object-Centric Models beyond Object Discovery


### Images

![53b490689fa4d927b2fa7b83b064472f768f3abd32b55ea587ca850733fc77d5.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/53b490689fa4d927b2fa7b83b064472f768f3abd32b55ea587ca850733fc77d5.jpg)

![62b6ca472127028b76252e764fe254d89a1246dbf8543105794dfbf74338a9ce.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/62b6ca472127028b76252e764fe254d89a1246dbf8543105794dfbf74338a9ce.jpg)

![b8f192dc2ea830aa32f31619b0a3196fa22f644ce9b0e69d16d64d89b5359b6a.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/b8f192dc2ea830aa32f31619b0a3196fa22f644ce9b0e69d16d64d89b5359b6a.jpg)

![c95ae48d97a03b9bc377609d562a03349abb27083d1fae317129110d9b30513d.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/c95ae48d97a03b9bc377609d562a03349abb27083d1fae317129110d9b30513d.jpg)

![d84972304975990b8ed5c625194543e0b47abed897ca46e4f0ac038b7f7a7925.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/d84972304975990b8ed5c625194543e0b47abed897ca46e4f0ac038b7f7a7925.jpg)

![e201524f8b1e4064e192a8333fa5d530534b70839710d317de97fa712bf81901.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/e201524f8b1e4064e192a8333fa5d530534b70839710d317de97fa712bf81901.jpg)

![ecd426721d044f9bcf6f9e1db99f032f02aae55c5f46bcf957b41d0cc5b98311.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/ecd426721d044f9bcf6f9e1db99f032f02aae55c5f46bcf957b41d0cc5b98311.jpg)

![f20766901331512f2007817c330753997a90cf6ed32417e0ff5346b093dd52d5.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/images/f20766901331512f2007817c330753997a90cf6ed32417e0ff5346b093dd52d5.jpg)

### Tables

![0547717dc0a11c789aeb3012f7cfb452fb132bd3dd2d953da9eac0a58f1e4a48.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/0547717dc0a11c789aeb3012f7cfb452fb132bd3dd2d953da9eac0a58f1e4a48.jpg)

![1a34d929a97ecd4fee288f6c2b399153957c75d9d666c46a75199eeff448577c.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/1a34d929a97ecd4fee288f6c2b399153957c75d9d666c46a75199eeff448577c.jpg)

![25da9b9ca8b0f4fbd5a59e750eac85b1f033a1530f5bf63cf030f115c3fd0883.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/25da9b9ca8b0f4fbd5a59e750eac85b1f033a1530f5bf63cf030f115c3fd0883.jpg)

![281c9e2f795dd8e9848979faeec7e7d6292c12fcc650886e80264dbe8f8d07c8.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/281c9e2f795dd8e9848979faeec7e7d6292c12fcc650886e80264dbe8f8d07c8.jpg)

![2a9d99f6c468f2ea3c8b3830fd8b6cef249dc3b51fa50402d4ce53f87be5c64f.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/2a9d99f6c468f2ea3c8b3830fd8b6cef249dc3b51fa50402d4ce53f87be5c64f.jpg)

![2faa5b6d3cfef9c9e50ed84d465bb0ec053fc3fb0ae9ee17fe750565ee12b1dd.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/2faa5b6d3cfef9c9e50ed84d465bb0ec053fc3fb0ae9ee17fe750565ee12b1dd.jpg)

![789a2f76c3d0b649f4cb1387b446421b6f2b3433ae6159c2330b9042036f6d04.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/789a2f76c3d0b649f4cb1387b446421b6f2b3433ae6159c2330b9042036f6d04.jpg)

![7ac8f029b57021dd4b5cd4fd524330b6584fe63c3b7cdfbb3353465b805e9a5f.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/7ac8f029b57021dd4b5cd4fd524330b6584fe63c3b7cdfbb3353465b805e9a5f.jpg)

![86e3ae174cd188a3d4ef7f5731724ee26c93fa7a24018fdd94430ad3bd2550a9.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/86e3ae174cd188a3d4ef7f5731724ee26c93fa7a24018fdd94430ad3bd2550a9.jpg)

![8afe68df840866e688e1d91f3ca48ce4ca1afa4994b10ad2663a1146beca09f8.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/8afe68df840866e688e1d91f3ca48ce4ca1afa4994b10ad2663a1146beca09f8.jpg)

![9d245220fffda981255d6d715cec61717c6cf50c60001be1a51257e68437191b.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/9d245220fffda981255d6d715cec61717c6cf50c60001be1a51257e68437191b.jpg)

![e915de975d526c3a796f8488af189b9f9d304db81b4a00eca95642c0b395fce5.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/e915de975d526c3a796f8488af189b9f9d304db81b4a00eca95642c0b395fce5.jpg)

![f4ac6c41d92b24559c544e50bed492d38308459c65948473f472e6fb2d28c6ca.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/f4ac6c41d92b24559c544e50bed492d38308459c65948473f472e6fb2d28c6ca.jpg)

![f58abd5379bb09038a837b2eec477efa8b090852e4e1bec1757863f7f4ff2f95.jpg](../icml_results/4327_Evaluating%20Object-Centric%20Models%20beyond%20Object%20Discovery/tables/f58abd5379bb09038a837b2eec477efa8b090852e4e1bec1757863f7f4ff2f95.jpg)

<a id="paper-4284"></a>

## VIBE: Disentangling Social Dynamics via Kinematics-Informed Variational Inference for Behavioral Emotion


### Images

![1b626a54a1863178bc6c534a03f3503067fc9a172a0e3af46ab6730a571ba76d.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/images/1b626a54a1863178bc6c534a03f3503067fc9a172a0e3af46ab6730a571ba76d.jpg)

![44169f8d0f453988dc5e4ebbfcb18b6a6e48ff981c715ab78f6583cf839423a3.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/images/44169f8d0f453988dc5e4ebbfcb18b6a6e48ff981c715ab78f6583cf839423a3.jpg)

![be1988a458dff207ec5a25590f9671b7920a24cf6645dc13b59b42e5a3314d4b.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/images/be1988a458dff207ec5a25590f9671b7920a24cf6645dc13b59b42e5a3314d4b.jpg)

![ca0eea8b1fa4a7fa521e29308a6ae210d2567e56249ce3e1a41552c12f189e34.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/images/ca0eea8b1fa4a7fa521e29308a6ae210d2567e56249ce3e1a41552c12f189e34.jpg)

### Tables

![1387267af3624fb7e13494472f343906148d8af89cc964f09fb63c78047fd32e.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/1387267af3624fb7e13494472f343906148d8af89cc964f09fb63c78047fd32e.jpg)

![34ab2b3d62cf883dd99dafdf7a557861ceeb0265b5d285c7549d98cef3099953.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/34ab2b3d62cf883dd99dafdf7a557861ceeb0265b5d285c7549d98cef3099953.jpg)

![430c77846359bd049a16331873bf9955feec56d340546b5a678d649ec9343844.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/430c77846359bd049a16331873bf9955feec56d340546b5a678d649ec9343844.jpg)

![8e3cd723c3ceb49706ded3b30fc07925e9230ec1818e9abae31b709d4d6a8c8f.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/8e3cd723c3ceb49706ded3b30fc07925e9230ec1818e9abae31b709d4d6a8c8f.jpg)

![acae6d10a0893b319b8056593202b721d82f10a459fe9e3ec9dfa1e98b905c3e.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/acae6d10a0893b319b8056593202b721d82f10a459fe9e3ec9dfa1e98b905c3e.jpg)

![ad412026e3980f288ea9ad815734ba9966bff228723cb1fac604d70ffe0274c9.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/ad412026e3980f288ea9ad815734ba9966bff228723cb1fac604d70ffe0274c9.jpg)

![be05aff4c09464de8495cc193dd41dd1af79aaf261eacc64ddc5f5bf8c357881.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/be05aff4c09464de8495cc193dd41dd1af79aaf261eacc64ddc5f5bf8c357881.jpg)

![c90994507931698a900dc86cf35fb32df7a65418ee36c2810e3bf1f3c72d3e99.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/c90994507931698a900dc86cf35fb32df7a65418ee36c2810e3bf1f3c72d3e99.jpg)

![e0ed0ad1709cd85526e2c966692affb16ea1cddbc8d392617a21fddb3a60d9f5.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/e0ed0ad1709cd85526e2c966692affb16ea1cddbc8d392617a21fddb3a60d9f5.jpg)

![eed1f7b66184dfc24ee68fc464e735b923859558fd158a2ceedc914df16ac987.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/eed1f7b66184dfc24ee68fc464e735b923859558fd158a2ceedc914df16ac987.jpg)

![fda133d6f69fc788ac4fb0389fdcfba0e8edfee85439a1256d87292779678990.jpg](../icml_results/4328_VIBE_%20Disentangling%20Social%20Dynamics%20via%20Kinematics-Informed%20Variational%20Inference%20for%20Behavioral%20Emo/tables/fda133d6f69fc788ac4fb0389fdcfba0e8edfee85439a1256d87292779678990.jpg)

<a id="paper-4285"></a>

## Divide and Learn: Multi-Objective Combinatorial Optimization at Scale


### Images

![05b7b761e7abfc296e580b58185416f79f1dcf4b6d539d5969b7150cef00076b.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/05b7b761e7abfc296e580b58185416f79f1dcf4b6d539d5969b7150cef00076b.jpg)

![06e0518551b27a52841547c5657851196c198577fb2ea2e88d6a6283a5727533.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/06e0518551b27a52841547c5657851196c198577fb2ea2e88d6a6283a5727533.jpg)

![0d2b36dc7fb57493bc98b7211133ceb6c6a6eb0e2fe4a0f42a77109707537f95.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/0d2b36dc7fb57493bc98b7211133ceb6c6a6eb0e2fe4a0f42a77109707537f95.jpg)

![1d410560b1ff873cb6b93113868705b408fdb9db76b71ff53f61aa8cf190323d.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/1d410560b1ff873cb6b93113868705b408fdb9db76b71ff53f61aa8cf190323d.jpg)

![48ddb2b666243e13e9c0079f7583733565e7253448539143cacd33bc5609a3d3.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/48ddb2b666243e13e9c0079f7583733565e7253448539143cacd33bc5609a3d3.jpg)

![535ee8f708dbd1884a036c7e5915039fdf3489fc5e7c93b137bea8796a20c879.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/535ee8f708dbd1884a036c7e5915039fdf3489fc5e7c93b137bea8796a20c879.jpg)

![598cf07afde70dc2e2df190dc2f4a369d469eb8f6f9233e6cc2e4c05340f2057.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/598cf07afde70dc2e2df190dc2f4a369d469eb8f6f9233e6cc2e4c05340f2057.jpg)

![5f71191974309835a95e60b6dca8c2e36db1b6e5f9b75dbc09eae4f85d0df9c4.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/5f71191974309835a95e60b6dca8c2e36db1b6e5f9b75dbc09eae4f85d0df9c4.jpg)

![5fabceb83a257154c7bbab867ea51731a5855a2c8e0a4884bbf90a2b9befa00b.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/5fabceb83a257154c7bbab867ea51731a5855a2c8e0a4884bbf90a2b9befa00b.jpg)

![646df8ef309758a748c73f4f7f8cc893229a6b492b6b757d2029fab3b6c8dc66.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/646df8ef309758a748c73f4f7f8cc893229a6b492b6b757d2029fab3b6c8dc66.jpg)

![67da26a25acd2f79284752b2ae04dff722026c5e13b421b63d9e88e48a51090a.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/67da26a25acd2f79284752b2ae04dff722026c5e13b421b63d9e88e48a51090a.jpg)

![7267c139c072e3cfaa5e9b0379780b628e88f7e0e4859c52b9c62f471b6ab346.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/7267c139c072e3cfaa5e9b0379780b628e88f7e0e4859c52b9c62f471b6ab346.jpg)

![7f7e6a567874ec09896521a5985e40f765eaf8d4f4d404a55ebaa537a1132768.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/7f7e6a567874ec09896521a5985e40f765eaf8d4f4d404a55ebaa537a1132768.jpg)

![87569c4eedf356dfb39a4a162dfc01cf78d22526cc1501131bb57afc00beec26.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/87569c4eedf356dfb39a4a162dfc01cf78d22526cc1501131bb57afc00beec26.jpg)

![8af39f569112582f7c950517bb1d17596ff5c7107ccaf359cc2a6c0f4548546c.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/8af39f569112582f7c950517bb1d17596ff5c7107ccaf359cc2a6c0f4548546c.jpg)

![9e21e84c3650936494992ea62a3401f446dc0729e54e1be006a6871506f6604c.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/9e21e84c3650936494992ea62a3401f446dc0729e54e1be006a6871506f6604c.jpg)

![a5ea04056af7b1bc807b76ae1bfc809a2df52cbcfd0b9e2250ec80b02bd37c46.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/a5ea04056af7b1bc807b76ae1bfc809a2df52cbcfd0b9e2250ec80b02bd37c46.jpg)

![a7e520901cb3a33d4f652c98c102bea6c4465bfd206a4e630c5b8de4d6a4cb1e.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/a7e520901cb3a33d4f652c98c102bea6c4465bfd206a4e630c5b8de4d6a4cb1e.jpg)

![ab9ce508e8437dd83b5e86d3c3d22b04c01a7986b817370e93eda9b962a687d9.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/ab9ce508e8437dd83b5e86d3c3d22b04c01a7986b817370e93eda9b962a687d9.jpg)

![adc894f4897bc579bfe5924c238feb4f997b71335c168c0f254a63554edd8f4b.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/adc894f4897bc579bfe5924c238feb4f997b71335c168c0f254a63554edd8f4b.jpg)

![c0028dd589e461e9cccf3c1f58f78c3ac88a1d004eeb7aef80b357e1a60c69c9.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/c0028dd589e461e9cccf3c1f58f78c3ac88a1d004eeb7aef80b357e1a60c69c9.jpg)

![c84fd85283e2ed6638be04431a6a76a21146a18d77c44338267e5fe96a37b9c6.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/c84fd85283e2ed6638be04431a6a76a21146a18d77c44338267e5fe96a37b9c6.jpg)

![c9253618441493b56f7063fec3f04f7364dfdb56665caad6e6b30dcdef963e49.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/c9253618441493b56f7063fec3f04f7364dfdb56665caad6e6b30dcdef963e49.jpg)

![d9891aa17f4a8b356a594560ae093125fabc795a13c41e6c190a1303321c4c05.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/d9891aa17f4a8b356a594560ae093125fabc795a13c41e6c190a1303321c4c05.jpg)

![e320c112517f51e0ada3df48bad86859694119c2c396d7ae8f82a1594593c061.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/e320c112517f51e0ada3df48bad86859694119c2c396d7ae8f82a1594593c061.jpg)

![ecdfd4325d4c9eb606fc8b40056ce1e15677c0f90c5219359aabc93f45ace711.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/ecdfd4325d4c9eb606fc8b40056ce1e15677c0f90c5219359aabc93f45ace711.jpg)

![f1faf06fab1f449d666d3bb8d7bca64db25a28942964495c8a9dfa82bcb7b506.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/images/f1faf06fab1f449d666d3bb8d7bca64db25a28942964495c8a9dfa82bcb7b506.jpg)

### Tables

![0e5b8a9742245c32a704a3413b4cc0b1cc6a522d5ccae12e6f0403a773ed43e2.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/0e5b8a9742245c32a704a3413b4cc0b1cc6a522d5ccae12e6f0403a773ed43e2.jpg)

![166096a2293772a02b944d4fd58beb08b44059d3c04297a2c28a07cdbc353785.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/166096a2293772a02b944d4fd58beb08b44059d3c04297a2c28a07cdbc353785.jpg)

![1ddb554df025ccb831d83e5421497f8480cd0f1a5df89d226e803b2e3b2ba0c3.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/1ddb554df025ccb831d83e5421497f8480cd0f1a5df89d226e803b2e3b2ba0c3.jpg)

![200b4608778e9632c74b53a49748874f12b4cbb12947dfba5196c76431492acf.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/200b4608778e9632c74b53a49748874f12b4cbb12947dfba5196c76431492acf.jpg)

![2496497016d7bd12de796058e4f0466b165add3d48335509a777fb6a25d97c08.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/2496497016d7bd12de796058e4f0466b165add3d48335509a777fb6a25d97c08.jpg)

![304bfe118ea5ef5ad4ee762cb0c5a0072242d1e759b4211a2b5d785567cc825e.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/304bfe118ea5ef5ad4ee762cb0c5a0072242d1e759b4211a2b5d785567cc825e.jpg)

![4da0f560a14075c94de0451fd32bbc70514fe5406513f11706d95a4aede70a1f.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/4da0f560a14075c94de0451fd32bbc70514fe5406513f11706d95a4aede70a1f.jpg)

![6d242d307e71f33429bcd4123ec971cd46722b00b64d8daa566ed0f0251d8696.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/6d242d307e71f33429bcd4123ec971cd46722b00b64d8daa566ed0f0251d8696.jpg)

![85de4e9245f15be9536d617b5bbd645147699ed7e600d5c6e03ebf75225333e7.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/85de4e9245f15be9536d617b5bbd645147699ed7e600d5c6e03ebf75225333e7.jpg)

![8f06be0b750bae73d50337abb7b92d65eef2d3403ea7eba11e6479cdd47cf72e.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/8f06be0b750bae73d50337abb7b92d65eef2d3403ea7eba11e6479cdd47cf72e.jpg)

![90b4ef547e1bb090e53e128fdcec72410b5f0bafcf1a6db0878967b74e015fe3.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/90b4ef547e1bb090e53e128fdcec72410b5f0bafcf1a6db0878967b74e015fe3.jpg)

![9abdf4eec66a908b05557360a5e42970bcd6b8f26a6fd0de2da38679a04afdc5.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/9abdf4eec66a908b05557360a5e42970bcd6b8f26a6fd0de2da38679a04afdc5.jpg)

![9c14c690591fdf32b0a88f7eaa8521d79b4b2220d8e6217e3973145443bd3e44.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/9c14c690591fdf32b0a88f7eaa8521d79b4b2220d8e6217e3973145443bd3e44.jpg)

![a0124b3e5fda7582d21a0e43cfec0519c28ac3d8e2f8b56d21602594fc2a8283.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/a0124b3e5fda7582d21a0e43cfec0519c28ac3d8e2f8b56d21602594fc2a8283.jpg)

![bb605700be1021436a863da079cfa5ce4f75b5ffd838b300f5c402416578da09.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/bb605700be1021436a863da079cfa5ce4f75b5ffd838b300f5c402416578da09.jpg)

![bbe5467412265b5ca7b41198776eba6243d13bba88d6d0ce7051f9ba946a6765.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/bbe5467412265b5ca7b41198776eba6243d13bba88d6d0ce7051f9ba946a6765.jpg)

![c7078811481cd2037e20e7f8855b6be11d6629b3de1587b93eecd57c0250554d.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/c7078811481cd2037e20e7f8855b6be11d6629b3de1587b93eecd57c0250554d.jpg)

![cd5f061f09dddc36a88cf9c0df08f73d57a5dafd677f0ae17aa841a01222cec0.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/cd5f061f09dddc36a88cf9c0df08f73d57a5dafd677f0ae17aa841a01222cec0.jpg)

![d12a3129cbb6ee6d19a6b78d46359604bfea10ac5273ed3cfd57fb7d8b86af8f.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/d12a3129cbb6ee6d19a6b78d46359604bfea10ac5273ed3cfd57fb7d8b86af8f.jpg)

![d4acd4831893cdb9a3592b18c82b2c2d28b95950876df1431c055fa037633d04.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/d4acd4831893cdb9a3592b18c82b2c2d28b95950876df1431c055fa037633d04.jpg)

![e4cc68f18bff8951a00ba5c4dc7b9a8e818d920cacb1bca54d74125653ffd60e.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/e4cc68f18bff8951a00ba5c4dc7b9a8e818d920cacb1bca54d74125653ffd60e.jpg)

![e8c5c6033b9e5e2efdb485ae0e4e67e03a528a0edd01a79c2b4b854ea0a3bdca.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/e8c5c6033b9e5e2efdb485ae0e4e67e03a528a0edd01a79c2b4b854ea0a3bdca.jpg)

![eefe072353a1f186736bf71127fe1c44f59d1aadbbf4075b0afbbdc5154b2c18.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/eefe072353a1f186736bf71127fe1c44f59d1aadbbf4075b0afbbdc5154b2c18.jpg)

![f4bce76b42e734c5181be58d875b7da1b7d043aee6f55394352c507d8d822af8.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/f4bce76b42e734c5181be58d875b7da1b7d043aee6f55394352c507d8d822af8.jpg)

![f739cf281833decc020d78d0e554bd6cb465794a54e610739b5a024b3e23b1a4.jpg](../icml_results/4329_Divide%20and%20Learn_%20Multi-Objective%20Combinatorial%20Optimization%20at%20Scale/tables/f739cf281833decc020d78d0e554bd6cb465794a54e610739b5a024b3e23b1a4.jpg)

<a id="paper-4286"></a>

## LLM Priors for ERM over Programs


### Images

![13bbfa32cba0183e6fc28e048527b6ea80867a1dc41c5e6256dbf3a20d76ad71.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/13bbfa32cba0183e6fc28e048527b6ea80867a1dc41c5e6256dbf3a20d76ad71.jpg)

![3623706fc4fbe3b3cc06049eb10450411fcde27794bae9f9d59ff4502933559e.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/3623706fc4fbe3b3cc06049eb10450411fcde27794bae9f9d59ff4502933559e.jpg)

![57e3d8e510e4b9e41a3ccc21973d5962a8ee01444a6f56042d1602a3c975091d.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/57e3d8e510e4b9e41a3ccc21973d5962a8ee01444a6f56042d1602a3c975091d.jpg)

![64e0021cd287df314124fc008e57a01256b287af93fe5f0bea1f8a9dd5793940.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/64e0021cd287df314124fc008e57a01256b287af93fe5f0bea1f8a9dd5793940.jpg)

![973e27565a71933e1236588b5df6cb272950f4e95fac172ec500096bab37b764.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/973e27565a71933e1236588b5df6cb272950f4e95fac172ec500096bab37b764.jpg)

![d7f6adcc8a7c409c6d9eeb1000ae8e1c0f53e9bcaff06094c1c0931f38832d04.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/images/d7f6adcc8a7c409c6d9eeb1000ae8e1c0f53e9bcaff06094c1c0931f38832d04.jpg)

### Tables

![01d45fed1de2ab8878bd383539c935fe3916c6c70c4fe88bec70384e5615bf79.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/01d45fed1de2ab8878bd383539c935fe3916c6c70c4fe88bec70384e5615bf79.jpg)

![0d3ce988903a356ef649e8c362a7c44ca9e3421ecb765c2342816c829b074b4d.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/0d3ce988903a356ef649e8c362a7c44ca9e3421ecb765c2342816c829b074b4d.jpg)

![47f96c7917c187cbd9ae3a496a10da49acb9aefa3ad91b3752ac365e0bf94cf9.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/47f96c7917c187cbd9ae3a496a10da49acb9aefa3ad91b3752ac365e0bf94cf9.jpg)

![6fe75c30e3e23fdc906d4e8b347cf48eed3371e26d9b99f988d378d52db9297b.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/6fe75c30e3e23fdc906d4e8b347cf48eed3371e26d9b99f988d378d52db9297b.jpg)

![9484eeaddb754f4dd6c7f03ecc143d4cc6e08b507769c4d97a3eaa49e9d2e3a2.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/9484eeaddb754f4dd6c7f03ecc143d4cc6e08b507769c4d97a3eaa49e9d2e3a2.jpg)

![9e935373142bf6958aec4d1e6acaea2763e5b109e6e6faf8853524d02eac2a1e.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/9e935373142bf6958aec4d1e6acaea2763e5b109e6e6faf8853524d02eac2a1e.jpg)

![e3ece0860f0207c848dd8dc6a73519a13fb92c7dfbc1079f4c494cdbba2b2213.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/e3ece0860f0207c848dd8dc6a73519a13fb92c7dfbc1079f4c494cdbba2b2213.jpg)

![ff503791d4f1c7a9f615d0f0c431cb169e5a04c6dfd3606c8e30c6001b63b36d.jpg](../icml_results/4330_LLM%20Priors%20for%20ERM%20over%20Programs/tables/ff503791d4f1c7a9f615d0f0c431cb169e5a04c6dfd3606c8e30c6001b63b36d.jpg)

<a id="paper-4287"></a>

## Accurate, Private, Secure, Federated U-statistics with Higher Degree


### Images

![0404472ca95f79e99fbe780367c229cc25f7563298168bbc74f811435d73fc1d.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/0404472ca95f79e99fbe780367c229cc25f7563298168bbc74f811435d73fc1d.jpg)

![17bdcb829a922afedbf698a0985250971796b163bd59d72baa89f7a0331906f4.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/17bdcb829a922afedbf698a0985250971796b163bd59d72baa89f7a0331906f4.jpg)

![27bccdafc0017a47664cb7cbe467c6cb2195b5cde193dd4bd18bbd49aeae297f.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/27bccdafc0017a47664cb7cbe467c6cb2195b5cde193dd4bd18bbd49aeae297f.jpg)

![6225147552aee4574c4f25800203ccb1a72266372b4d59125371fe79c4db6584.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/6225147552aee4574c4f25800203ccb1a72266372b4d59125371fe79c4db6584.jpg)

![7e5c8e044dbabc859ba0c2cff23e90c59c47faea15b1c1889ae716c90fbe4ec4.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/7e5c8e044dbabc859ba0c2cff23e90c59c47faea15b1c1889ae716c90fbe4ec4.jpg)

![d3a79ef218f94b017199b5aaa5ded2cfc0d87fad67e99a5cac29c77ed1c64c46.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/images/d3a79ef218f94b017199b5aaa5ded2cfc0d87fad67e99a5cac29c77ed1c64c46.jpg)

### Tables

![05e18c43a6b949959cc8539fd5d51704954b385be7548a139346ccae47a525cc.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/05e18c43a6b949959cc8539fd5d51704954b385be7548a139346ccae47a525cc.jpg)

![3b5bfa18dc05e051dfbbe014de6dadaf5b0969e8e9595d206be28b4422dd027d.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/3b5bfa18dc05e051dfbbe014de6dadaf5b0969e8e9595d206be28b4422dd027d.jpg)

![7b14ac7e3c961e82fecedc43cd7437e232ec2f8a9d51e93077615096b185a7c3.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/7b14ac7e3c961e82fecedc43cd7437e232ec2f8a9d51e93077615096b185a7c3.jpg)

![9d2cacafcffbb5b80246649f36e2d03fbcf94b9f5c8986cfd684355c45dd8a3f.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/9d2cacafcffbb5b80246649f36e2d03fbcf94b9f5c8986cfd684355c45dd8a3f.jpg)

![e87c71cef9bbf68e69168614aec82d5dcb1a706b65cd9ba30a7b1c7810ca5dcb.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/e87c71cef9bbf68e69168614aec82d5dcb1a706b65cd9ba30a7b1c7810ca5dcb.jpg)

![f832555c6ab7185802d5188e2873650b8fcc6c666446e81b114a9339991cc781.jpg](../icml_results/4331_Accurate%2C%20Private%2C%20Secure%2C%20Federated%20U-statistics%20with%20Higher%20Degree/tables/f832555c6ab7185802d5188e2873650b8fcc6c666446e81b114a9339991cc781.jpg)

<a id="paper-4288"></a>

## Trainable Nonexpansive Denoisers for Contractive Image Reconstruction


### Images

![0c944907bb25c45f416a415b18e4c9aae3dc151c384bce955cc8196d3ad4ec54.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/0c944907bb25c45f416a415b18e4c9aae3dc151c384bce955cc8196d3ad4ec54.jpg)

![632bd46661cdb016c2938e0e1fff0c4c1641b73ce881a589b479b3427643d1b9.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/632bd46661cdb016c2938e0e1fff0c4c1641b73ce881a589b479b3427643d1b9.jpg)

![6f3bd935b18776109a3f06901a6f332460ee2e36b77a873d57b5893dd211523a.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/6f3bd935b18776109a3f06901a6f332460ee2e36b77a873d57b5893dd211523a.jpg)

![83539a29937c047aa116328bc36370d9221b0f33a64fc344a06709f5d5134ad3.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/83539a29937c047aa116328bc36370d9221b0f33a64fc344a06709f5d5134ad3.jpg)

![85dee1128a0f76eac48e96f58f71131005b5e61f90d628acdf644839e3153b4a.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/85dee1128a0f76eac48e96f58f71131005b5e61f90d628acdf644839e3153b4a.jpg)

![8fe9258cdf5767aa316e32ae3b4828b17017816aaf5273fd08ca606ced89bbe6.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/8fe9258cdf5767aa316e32ae3b4828b17017816aaf5273fd08ca606ced89bbe6.jpg)

![a2f2edb8ae4d942b6156f6634f02aa9978b0a3d4efa3dba86b7a6ff916ca22c0.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/a2f2edb8ae4d942b6156f6634f02aa9978b0a3d4efa3dba86b7a6ff916ca22c0.jpg)

![b45ddfbe3d9699cd5eff9c7596beca552938ccb26ce71d59691333e59a828268.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/images/b45ddfbe3d9699cd5eff9c7596beca552938ccb26ce71d59691333e59a828268.jpg)

### Tables

![585db82dfe58da14290def065e1762499048111a67ee189174216ed729dd59c7.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/585db82dfe58da14290def065e1762499048111a67ee189174216ed729dd59c7.jpg)

![6f7d2abf42794dcccf60bc1e1fbe70375930a03670e47c09e719bff0b676d587.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/6f7d2abf42794dcccf60bc1e1fbe70375930a03670e47c09e719bff0b676d587.jpg)

![718e6ecf5295b08eae9f253ad0e1de9627583bdfd15635ec49fc9ca922a3b7d5.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/718e6ecf5295b08eae9f253ad0e1de9627583bdfd15635ec49fc9ca922a3b7d5.jpg)

![9c93864283fd49e875106dfe43e8bdf6edeaa4ecffe199c354ce74a3545e591f.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/9c93864283fd49e875106dfe43e8bdf6edeaa4ecffe199c354ce74a3545e591f.jpg)

![c03d05195a28889dfffbf8dc250c2a1baee1009679157627762952fa3fc19961.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/c03d05195a28889dfffbf8dc250c2a1baee1009679157627762952fa3fc19961.jpg)

![d2311c5fac6847648f812ce1b708e5e4e9ad08ea1d9b8a3e16d8b35d49cebd6f.jpg](../icml_results/4332_Trainable%20Nonexpansive%20Denoisers%20for%20Contractive%20Image%20Reconstruction/tables/d2311c5fac6847648f812ce1b708e5e4e9ad08ea1d9b8a3e16d8b35d49cebd6f.jpg)

<a id="paper-4289"></a>

## Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling


### Images

![037e590401378a6f0ebc87b413c9960dc2309aca7b485c73d4f1f2940bcaf4c9.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/images/037e590401378a6f0ebc87b413c9960dc2309aca7b485c73d4f1f2940bcaf4c9.jpg)

![252ad2e4367e581ce1983cdc05f6692ef4b887f1fc3ead93bad840dc2d40a9a6.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/images/252ad2e4367e581ce1983cdc05f6692ef4b887f1fc3ead93bad840dc2d40a9a6.jpg)

![d532504aa2cdb9787ae81f2d4c85205428a0262020dd44e0503e6328e56f6428.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/images/d532504aa2cdb9787ae81f2d4c85205428a0262020dd44e0503e6328e56f6428.jpg)

### Tables

![09ccb17475cd4ea83720bfc3b4af361be41bbd14f9929c004e7ae42a4cb1fa85.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/09ccb17475cd4ea83720bfc3b4af361be41bbd14f9929c004e7ae42a4cb1fa85.jpg)

![25e6f6b7ca38109921edc254a0ea2d59283daa1d7fdf1299af709bc6103667e0.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/25e6f6b7ca38109921edc254a0ea2d59283daa1d7fdf1299af709bc6103667e0.jpg)

![527d5bc6c92b1493b1c46368fce7e05867f2035186e7ea338ec3caceaacbb026.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/527d5bc6c92b1493b1c46368fce7e05867f2035186e7ea338ec3caceaacbb026.jpg)

![63c435d247dcc7e8c521007d7472ea071682f70cc1a3fbc0d389484e039338f6.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/63c435d247dcc7e8c521007d7472ea071682f70cc1a3fbc0d389484e039338f6.jpg)

![aa6772ed66ac5377e785fc8dceeefbc296d58b58164f969c43998d4483312482.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/aa6772ed66ac5377e785fc8dceeefbc296d58b58164f969c43998d4483312482.jpg)

![b33c020aabb17b920167a5556acfaa74d6e7cff8c1e3b831c8214352bc03d558.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/b33c020aabb17b920167a5556acfaa74d6e7cff8c1e3b831c8214352bc03d558.jpg)

![cd9cd920575190ab4ffc3affe194e8f77c2cd1f7614aebd71d6641488fa4d16b.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/cd9cd920575190ab4ffc3affe194e8f77c2cd1f7614aebd71d6641488fa4d16b.jpg)

![f41b47630ced4b33eedb4dc3d0540c1efffbfbc17c72608f2f76ee603a1a5e2b.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/f41b47630ced4b33eedb4dc3d0540c1efffbfbc17c72608f2f76ee603a1a5e2b.jpg)

![f61eec7e0dd9c8e4e532cdd5e904f6c26822b3a13e52b0f83648608821ebd511.jpg](../icml_results/4333_Expected%20Return%20Causes%20Outcome-Level%20Mode%20Collapse%20in%20Reinforcement%20Learning%20and%20How%20to%20Fix%20It%20with%20/tables/f61eec7e0dd9c8e4e532cdd5e904f6c26822b3a13e52b0f83648608821ebd511.jpg)

<a id="paper-4290"></a>

## Implicit Intelligence - Evaluating Agents on What Users Don’t Say

### Images

![00d1e192d04d9363b51f92425e96cfd1e406f11dac1788a678fded1e6b40604e.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/00d1e192d04d9363b51f92425e96cfd1e406f11dac1788a678fded1e6b40604e.jpg)

![44cbe28cd116f299da47db0d54574e502cc025ef25ee89ecf59fc0d7eba24228.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/44cbe28cd116f299da47db0d54574e502cc025ef25ee89ecf59fc0d7eba24228.jpg)

![6475131c95ee182fa4c5dd805f919a4e395bb8660d679d140ae7d730233bb360.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/6475131c95ee182fa4c5dd805f919a4e395bb8660d679d140ae7d730233bb360.jpg)

![67c30222366be74704a7825d46386ceba0a57dc029a95bf2f41cce4483b11848.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/67c30222366be74704a7825d46386ceba0a57dc029a95bf2f41cce4483b11848.jpg)

![6daba154ce7da7722de6100c8a1dfec6d027eaed1fc8ef5e722f1ceb3606a36d.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/6daba154ce7da7722de6100c8a1dfec6d027eaed1fc8ef5e722f1ceb3606a36d.jpg)

![76648469e8f6aabdbf112305e90f570b44d1ba6a7b0bea42f6c34a38e375f144.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/76648469e8f6aabdbf112305e90f570b44d1ba6a7b0bea42f6c34a38e375f144.jpg)

![945e472f414a0be016d74af38a7db583171456cf8f060033b7c066cf306b7e79.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/945e472f414a0be016d74af38a7db583171456cf8f060033b7c066cf306b7e79.jpg)

![a036fa0a2b2d92bffeae75e8f0d5f243dab0d6f4be1ec2fb6837438663ade5da.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/images/a036fa0a2b2d92bffeae75e8f0d5f243dab0d6f4be1ec2fb6837438663ade5da.jpg)

### Tables

![050375c0b331156bcf7e97f09131320a69e79e657396c82b0a1c6cac2a2e46ba.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/050375c0b331156bcf7e97f09131320a69e79e657396c82b0a1c6cac2a2e46ba.jpg)

![09dc70edadfeaa5bf1fe019f541229ea72b976def329ad4157d27679ee321462.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/09dc70edadfeaa5bf1fe019f541229ea72b976def329ad4157d27679ee321462.jpg)

![396a45c7584b74314b83123d90beca3be55f1d33f3041c409e856fbf84597875.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/396a45c7584b74314b83123d90beca3be55f1d33f3041c409e856fbf84597875.jpg)

![430f6bcaf93d9828c5c821bf64bf6be3ed7fa3d69edfe092b3a5186d92fba0e9.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/430f6bcaf93d9828c5c821bf64bf6be3ed7fa3d69edfe092b3a5186d92fba0e9.jpg)

![6baa4354547c6334ea283edb01f6966880107b8f39aff90679d69436c5f1f686.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/6baa4354547c6334ea283edb01f6966880107b8f39aff90679d69436c5f1f686.jpg)

![7f09726ef2a4cb767beec6febc2c05e5e41ec3e34a5ae7ffc5eecf3937be8de9.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/7f09726ef2a4cb767beec6febc2c05e5e41ec3e34a5ae7ffc5eecf3937be8de9.jpg)

![c0756a8fe9c7a7fc7de4eeb3beca6fc6820d05c673aab4b090f65a72690d70c7.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/c0756a8fe9c7a7fc7de4eeb3beca6fc6820d05c673aab4b090f65a72690d70c7.jpg)

![c09cc9daf69f8fbb0afc5ae67edc933784b83895b4c491a6c158a9b4bb7efbe7.jpg](../icml_results/4334_Implicit%20Intelligence%20-%20Evaluating%20Agents%20on%20What%20Users%20Don%E2%80%99t%20Say/tables/c09cc9daf69f8fbb0afc5ae67edc933784b83895b4c491a6c158a9b4bb7efbe7.jpg)
